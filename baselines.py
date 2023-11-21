import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv, APPNP, SGConv, ChebConv, SAGEConv
import argparse, random, json
from utils import load_data, metrics, limit_label, limit_edge
import gc

class MLP(nn.Module):
    def __init__(self,
                 args,
                 ):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.num_classes)

        self.dropout = args.dropout
        self.activation = F.relu

    def forward(self, data):
        x = data.x
        x1 = self.activation(self.fc1(x))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.fc2(x1)
        return x1,x2

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.input_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.num_classes)
        self.activation = F.relu
        self.dropout = args.dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.activation(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            args.input_dim, args.hidden_dim, heads=args.num_heads, dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden_dim * args.num_heads, args.num_classes, dropout=args.dropout)

        self.dropout = args.dropout
        self.activation = F.relu

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x1 = F.dropout(x, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2

class APPNPNet(nn.Module):
    def __init__(self, args):
        super(APPNPNet, self).__init__()
        self.lin1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.lin2 = nn.Linear(args.hidden_dim, args.num_classes)
        self.prop1 = APPNP(args.K,args.alpha)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        self.conv1 = SGConv(args.input_dim, args.hidden_dim)
        self.conv2 = SGConv(args.hidden_dim, args.num_classes)
        self.activation = F.relu
        self.dropout = args.dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.activation(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2


class ChebNet(nn.Module):
    def __init__(self, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(args.input_dim, args.hidden_dim, K=args.K,normalization='sym')
        self.conv2 = ChebConv(args.hidden_dim, args.num_classes, K=args.K,normalization='sym')
        self.activation = F.relu
        self.dropout = args.dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.activation(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2

class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(args.input_dim, args.hidden_dim)
        self.conv2 = SAGEConv(args.hidden_dim, args.num_classes)
        self.activation = F.relu
        self.dropout = args.dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.activation(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2




def train(args,data):
    model = eval(args.model)(args).to(args.device)
    optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss_val = 10000000
    patience = args.patience
    best_acc_test = 0
    for epoch in range(args.EPOCH):
        model.train()

        if patience < 0:
            break
        pred_y = model(data)
        log_py = F.log_softmax(pred_y,dim=1)
        loss = F.nll_loss(log_py[data.train_mask],data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_y = model(data)
            log_py = F.log_softmax(pred_y,dim=1)
            loss_val = F.nll_loss(log_py[data.val_mask],data.y[data.val_mask])
            acc_train, acc_val, acc_test = metrics(pred_y,data)
            if min_loss_val > loss_val:
                min_loss_val = loss_val
                best_acc_test = acc_test
                # print("*"*30)
                # print(f'epoch: {epoch}, min_loss_val: {min_loss_val}, acc_train: {acc_train}, acc_val: {acc_val}, acc_test: {acc_test}')
                patience = args.patience
        patience -= 1
    return best_acc_test


def run_all_baselines(args):
    result = {}
    for dataset in ['pubmed','cora', 'citeseer']:
        for baseline in ['SGC','GCN','ChebNet','GAT','APPNPNet','SAGE']:
            for seed in [0,1,2,3,4]:
                fix_seed(seed)
                if baseline == 'APPNP':
                    args.K = 10
                    args.alpha = 0.1
                if baseline == 'ChebNet' : 
                    if dataset == 'cora':
                        args.K = 3
                    else:
                        args.K = 2
                data = load_data(dataset)
                data.train_mask = limit_label(data.train_mask,data.y,data.num_classes,args.lc)
                data.edge_index = limit_edge(data.edge_index,args.er)
               
                data = data.to(args.device)
                args.input_dim = data.x.shape[1]
                data.num_classes = data.num_classes
                args.model = baseline
                
                acc_test = train(args,data)
                if result.get(dataset) is None:
                    result[dataset] = {}
                if result[dataset].get(baseline) is None:
                    result[dataset][baseline] = {}
                result[dataset][baseline][seed] = acc_test
                print(f'dataset: {dataset}, baseline: {baseline}, seed: {seed} done.')

    with open(f'./baseline_benchmark_result_lc{str(args.lc)}_er{str(args.er)}.json','w') as f:
        f.write(json.dumps(result))




def run_all_baselines_PPI(args):
    result = {}
    for dataset in ['CPDB','IREF','IREF_2015','MULTINET','STRINGdb','RGE','PGE']:
        for baseline in ['SGC','GCN','ChebNet','GAT','APPNPNet','SAGE']:
            for seed in [0,1,2,3,4]:
                fix_seed(seed)
                if baseline == 'APPNP':
                    args.K = 10
                    args.alpha = 0.1
                if baseline == 'ChebNet' : 
                    if dataset == 'cora':
                        args.K = 3
                    else:
                        args.K = 2
                data = load_data(dataset)
                data.edge_index = data.Gobs.to_sparse().indices()
                data.x = data.x.float()
                del data.Gobs
                gc.collect()
                torch.cuda.empty_cache()
                

                data = data.to(args.device)
                args.input_dim = data.x.shape[1]
                data.num_classes = data.num_classes
                args.model = baseline
                # print(data)
                acc_test = train(args,data)
                if result.get(dataset) is None:
                    result[dataset] = {}
                if result[dataset].get(baseline) is None:
                    result[dataset][baseline] = {}
                result[dataset][baseline][seed] = acc_test
                print(f'dataset: {dataset}, baseline: {baseline}, seed: {seed} done.')

    with open(f'./baseline_PPI_result.json','w') as f:
        f.write(json.dumps(result))




def fix_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='GCN')
    parser.add_argument('--input_dim', type=int, default=1433)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--EPOCH', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lc', type=int,default=20)
    parser.add_argument('--er', type=float,default=1.0)
    args = parser.parse_args()


    run_all_baselines(args)
    run_all_baselines_PPI(args)




