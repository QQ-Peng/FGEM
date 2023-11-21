import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from utils import * 

import random,time,copy

import argparse, gc

from models import *
from graph_learner import graph_learner_cosine_similarity

def fix_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train():
    
    model.train()
    # print(data.y)
    
    y_onehot = one_hot(data.y,args.num_class).to(data.x.device)
    data.y_onehot = y_onehot
    

    QGQYModel_weight = None

    best_qy_loss_val = 10000
    best_py_loss_val = 10000
    
    train_QGQY = True
    train_PY = True

    EPOCH = args.EPOCH
    
    del data.init_pt_graph
    gc.collect()
    torch.cuda.empty_cache()
    
    
        
    if train_PY:
        print(f'start train PY Model.')
        patience = args.patience
        optimizer = torch.optim.Adam(model.PYModel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1,EPOCH):
            model.train()
            if patience < 0:
                break
            hidden, out = model.PYModel(data)
            loss_py = model.loss_fn.nll_loss_(out[data.train_mask],data.y[data.train_mask],True)
            optimizer.zero_grad()
            loss_py.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                hidden, py = model.PYModel(data) 
                py = F.log_softmax(py,dim=1)
                py_acc_train, py_acc_val, py_acc_test = metrics(py,data)
                py_loss_val = model.loss_fn.nll_loss_(py[data.val_mask],data.y[data.val_mask],True)
                # print(f'qy_loss_val: {qy_loss_val}, qy_acc_tain: {qy_acc_train}, qy_acc_val: {qy_acc_val}, qy_acc_test: {qy_acc_test}')
                if best_py_loss_val > py_loss_val:
                    best_py_loss_val = py_loss_val
                    # print(f'epoch: {epoch}, best_py_loss_val: {best_py_loss_val}, py_acc_train: {py_acc_train}, py_acc_val: {py_acc_val}, py_acc_test: {py_acc_test}')
                    patience = args.patience
                    PY_weight = copy.deepcopy(model.PYModel.state_dict())
        print(f'Train PY Model done.')

    if args.dataset in ['cora','citeseer','pubmed']:
        model.PYModel.load_state_dict(PY_weight)
        model.eval()
        with torch.no_grad():
            hidden, py = model.PYModel(data)
            py = F.log_softmax(py,dim=1)
            data.Gobs = torch.zeros(args.node_num,args.node_num).to(args.device)
            for x in [data.x,hidden,py]:
                dense_adj = graph_learner_cosine_similarity(x)
                data.Gobs += weight_adj_to_adj(dense_adj,args.k)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    if train_QGQY:
        print(f'start train QY_QG Model.')
        optimizer = torch.optim.Adam(model.QG_QY_Model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        edge_index_train = edge_sampling(data.Gobs,1.0,0.5)
        data.edge_index_train = edge_index_train
        patience = args.patience
        for epoch in range(1,EPOCH):
            
            model.train()
            if patience < 0:
                break
            
            qy, qg = model.QG_QY_Model(data) 
            
            loss_qy_obs = model.loss_fn.nll_loss_(qy[data.train_mask],data.y[data.train_mask],True) # (g)
            alpha = torch.sigmoid(model.QG_QY_Model.P_GObs_G_Model.alpha)
            beta = torch.sigmoid(model.QG_QY_Model.P_GObs_G_Model.beta)
            
            loss_PG_obs = model.loss_fn.P_Gobs(data.Gobs,qg,alpha,beta,edge_index_train,data.Gobs_num,False) # (d)
            
            loss = loss_qy_obs + 0.5*loss_PG_obs
            
            _, py = model.PYModel(data)
            kl_y_ub = 2*model.loss_fn.PY_QY_ub(py[~data.train_mask],qy[~data.train_mask])
            
            loss += kl_y_ub
            
            if args.G_prior is not None:
                # print('add PG KL')
                qy_softmax = F.softmax(qy,dim=1)
                qy_prob = torch.where(data.train_mask.unsqueeze(1), data.y_onehot,qy_softmax)

                if args.G_prior == 'sbm':
                    p0, p1 = model.QG_QY_Model.PGModel(data)
                    kl_PG_QG = model.loss_fn.PG_QG((qy_prob,qg,p0,p1,edge_index_train),False,'sbm')
                elif args.G_prior == 'lsm':
                    edge_prior = model.QG_QY_Model.PGModel(data,qy_prob)
                    kl_PG_QG = model.loss_fn.PG_QG((qg,edge_prior,edge_index_train),False,'lsm')
                loss += 0.5*kl_PG_QG
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.eval()
                qy, qg = model.QG_QY_Model(data) 
                qy = F.log_softmax(qy,dim=1)
                qy_acc_train, qy_acc_val, qy_acc_test = metrics(qy,data)
                qy_loss_val = model.loss_fn.nll_loss_(qy[data.val_mask],data.y[data.val_mask],True)
                
                if best_qy_loss_val>qy_loss_val:
                        best_qy_loss_val = qy_loss_val
                        print("*"*20)
                        # print(f'epoch: {epoch}, qy_acc_train: {qy_acc_train}, min_qy_loss_val: {min_qy_loss_val}, qy_acc_val: {qy_acc_val}, qy_acc_test: {qy_acc_test}, alpha: {alpha}, beta: {beta}')
                        if args.G_prior == 'sbm':
                            print(f'epoch: {epoch}, best_qy_loss_val: {best_qy_loss_val},qy_acc_train: {qy_acc_train},  qy_acc_val: {qy_acc_val}, qy_acc_test: {qy_acc_test}, alpha: {alpha}, beta: {beta}, p0: {p0}, p1: {p1}')
                        elif args.G_prior == 'lsm':
                            print(f'epoch: {epoch}, best_qy_loss_val: {best_qy_loss_val},qy_acc_train: {qy_acc_train},  qy_acc_val: {qy_acc_val}, qy_acc_test: {qy_acc_test}, alpha: {alpha}, beta: {beta}')

                        QGQYModel_weight = copy.deepcopy(model.QG_QY_Model.state_dict())
                        patience = args.patience
            patience -= 1
        
            
    with torch.no_grad():
        model.PYModel.load_state_dict(PY_weight)
        model.QG_QY_Model.load_state_dict(QGQYModel_weight)
        model.eval()
        qy, qg = model.QG_QY_Model(data) 
        qy = F.log_softmax(qy,dim=1)
        qy_acc_val = acc(qy[data.val_mask],data.y[data.val_mask])
        qy_acc_test = acc(qy[data.test_mask],data.y[data.test_mask])
        qy_acc_train = acc(qy[data.train_mask],data.y[data.train_mask])
        qy_loss_val = model.loss_fn.nll_loss_(qy[data.val_mask],data.y[data.val_mask],True)
        print(f'seed: {args.seed}, final result: qy_acc_train: {qy_acc_train}, qy_acc_val: {qy_acc_val}, qy_acc_test: {qy_acc_test}')
        torch.save(qg.cpu(),f'./out/qg_{args.dataset}.pkl')
    
    

parser = argparse.ArgumentParser()

parser.add_argument('--input_dim', type=int, default=1433)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_class', type=int, default=7)

parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)


parser.add_argument('--alpha', type=float,default=-0.2,help="Init true positive rate of the inverse function of sigmoid") 
parser.add_argument('--beta', type=float,default=-2.5,help="Init false positive rate of the inverse function of sigmoid") 
parser.add_argument('--P0', type=float,default=2.198,help='The p0 value of the inverse function of sigmoid')
parser.add_argument('--P1', type=float,default=-2.194,help='The p1 value of the inverse function of sigmoid')
parser.add_argument('--k', type=int, default=9, help='k for knn graph')
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--G_prior', default='sbm')


parser.add_argument('--dataset', default='cora')
parser.add_argument('--EPOCH', type=int, default=1000)
parser.add_argument('--verbose', type=int, default=-1)
parser.add_argument('--learn_sbm', action='store_true',help='whether to optimize the p0, p1, default not')
parser.add_argument('--lc', type=int,default=20)
parser.add_argument('--er', type=float,default=1.0)
parser.add_argument('--lmd', type=float,default=0.5, help='lambda')



args = parser.parse_args()
device = args.device 
data = load_data(args.dataset).to(device)

args.node_num = data.x.shape[0]
data.x = data.x.float()

args.input_dim = data.x.shape[1]
args.num_class = data.num_classes 

data.Gobs_num = 5 if args.dataset == 'multiPPI' else 3
if args.dataset == 'multiPPI':
    for i in range(data.x.shape[0]):
        if data.Gobs[i,i].item() == 0:
            data.Gobs[i,i] = 1

    data.edge_index = data.Gobs.to_sparse().indices()
    data.adj = copy.deepcopy(data.Gobs)
    data.adj[data.Gobs!=0] = 1
    data.adj_norm = gcn_norm_pyg([data.adj])[0]
    data = data.to(device)

else:
    edge_index = limit_edge(data.edge_index,args.er)
    data.edge_index = edge_index
    data.adj = edge_index_to_sparse_matrix(data).float()
    data.adj_norm = gcn_norm_pyg([data.adj])[0]
    data = data.to(args.device)
    train_mask = limit_label(data.train_mask,data.y,data.num_classes,args.lc)
    data.train_mask = train_mask
    
if args.seed is not None:
    fix_seed(args.seed)


model = GenerativeModel(args).to(device)
train()

