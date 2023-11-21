from torch_geometric.datasets import Planetoid
import os
import torch
import torch_geometric.transforms as transform
from torch_geometric.utils import add_self_loops,negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import numpy as np
import h5py,time, random
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity as cos



def load_data(dataset="cora",
              num_labels_per_class=20,
              missing_edge=False,
              verbose=0):
    # Load data.
    path = os.path.join("./", 'data')
    # if verbose:
    #     print("loading data from %s. %d labels per class." %
    #           (path, num_labels_per_class))
    assert dataset in ["cora", "pubmed", "citeseer","multiPPI",'CPDB','IREF','IREF_2015','MULTINET','STRINGdb','RGE','PGE']
    if dataset in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(
            root=path, name=dataset, transform=transform.NormalizeFeatures())

        data = dataset[0]
        data.num_classes = dataset.num_classes

        
        # Original Planetoid setting.
        if num_labels_per_class == 20:
            return data

        # Get one-hot labels.
        temp = data.y.numpy()
        labels = np.zeros((len(temp), temp.max() + 1))
        for i in range(len(labels)):
            labels[i, temp[i]] = 1

        all_idx = list(range(len(labels)))

        # Select a fixed number of training data per class.
        idx_train = []
        class_cnt = np.zeros(
            labels.shape[1])  # number of nodes selected for each class
        for i in all_idx:
            if (class_cnt >= num_labels_per_class).all():
                break
            if ((class_cnt + labels[i]) > num_labels_per_class).any():
                continue
            class_cnt += labels[i]
            idx_train.append(i)
        if verbose:
            print("number of training data: ", len(idx_train))

        train_mask = np.zeros((len(labels), ), dtype=int)
        val_mask = np.zeros((len(labels), ), dtype=int)
        test_mask = np.zeros((len(labels), ), dtype=int)
        for i in all_idx:
            if i in idx_train:
                train_mask[i] = 1
            elif sum(val_mask) < 500:  # select 500 validation data
                val_mask[i] = 1
            else:
                test_mask[i] = 1
        data.train_mask = torch.ByteTensor(train_mask)
        data.val_mask = torch.ByteTensor(val_mask)
        data.test_mask = torch.ByteTensor(test_mask)
    elif dataset in ['multiPPI','CPDB','IREF','IREF_2015','MULTINET','STRINGdb','RGE','PGE']:
        '''
        RGE: random graph estimated by algorithm 'Estimating network structure from unreliable measurements'
        PGE: poisson graph estimated by algorithm 'A principled approach for weighted multilayer network aggregation'
        '''
        
        y = torch.load(f'{path}/EMOGI/data/y.pkl').long()
        train_mask = torch.load(f'{path}/EMOGI/data/train_mask.pkl')
        val_mask = torch.load(f'{path}/EMOGI/data/val_mask.pkl')
        test_mask = torch.load(f'{path}/EMOGI/data/test_mask.pkl')
        
        x = torch.load(f'{path}/EMOGI/data/features.pkl')
        if dataset == 'multiPPI':
            Gobs = torch.load(f'{path}/EMOGI/data/Gobs_aggr.pkl',map_location='cpu')
    
            data = Data(x=x, Gobs=Gobs,y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,num_classes=2,Gobs_num=5)
        elif dataset in ['CPDB','IREF','IREF_2015','MULTINET','STRINGdb']:
            Gobs = torch.load(f'{path}/EMOGI/data/Gobs_{dataset}_dense_adj.pkl',map_location='cpu')
            data = Data(x=x, Gobs=Gobs,y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,num_classes=2,Gobs_num=1)
        elif dataset in ['RGE','PGE']:
            Gobs = torch.load(f'{path}/EMOGI/data/{dataset}0.5.pkl')
            
            data = Data(x=x, Gobs=Gobs,y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,num_classes=2,Gobs_num=1)

    return data



def process_PPI(path):

    gene_names = []
    features= []

    feature_names_fix = ['MF: KIRP', 'MF: BLCA', 'MF: THCA', 'MF: ESCA', 'MF: CESC', 'MF: COAD', 'MF: LIHC', 'MF: LUAD', 'MF: PRAD', 'MF: STAD', 'MF: HNSC', 'MF: LUSC', 'MF: UCEC', 'MF: BRCA', 'MF: READ', 'MF: KIRC', 'METH: KIRP', 'METH: BLCA', 'METH: THCA', 'METH: ESCA', 'METH: CESC', 'METH: COAD', 'METH: LIHC', 'METH: LUAD', 'METH: PRAD', 'METH: STAD', 'METH: HNSC', 'METH: LUSC', 'METH: UCEC', 'METH: BRCA', 'METH: READ', 'METH: KIRC', 'GE: KIRP', 'GE: BLCA', 'GE: THCA', 'GE: ESCA', 'GE: CESC', 'GE: COAD', 'GE: LIHC', 'GE: LUAD', 'GE: PRAD', 'GE: STAD', 'GE: HNSC', 'GE: LUSC', 'GE: UCEC', 'GE: BRCA', 'GE: READ', 'GE: KIRC', 'CNA: KIRP', 'CNA: BLCA', 'CNA: THCA', 'CNA: ESCA', 'CNA: CESC', 'CNA: COAD', 'CNA: LIHC', 'CNA: LUAD', 'CNA: PRAD', 'CNA: STAD', 'CNA: HNSC', 'CNA: LUSC', 'CNA: UCEC', 'CNA: BRCA', 'CNA: READ', 'CNA: KIRC']

    genes_to_features = {}

    # process feaure matrix
    print(f'start process feature matrix.')
    for file in os.listdir(path):
        if file.endswith('.h5'):
            f = h5py.File(f'{path}/{file}', 'r')
            # print(f'*'*40)
            feature_raw = np.array(f['features_raw'])
            feature_names = [i.decode() for i in np.array(f['feature_names'])]
            gn = [i.decode() for i in np.array(f['gene_names'])[:,1]]
            feature_col_ind = [feature_names.index(i) for i in feature_names_fix]
            feature_raw = feature_raw[:,feature_col_ind]
            for i in range(len(gn)):
                if genes_to_features.get(gn[i]) is None:
                    genes_to_features[gn[i]] = feature_raw[i]
                else:
                    assert (not (genes_to_features[gn[i]] != feature_raw[i]).sum()) 
    gene_names = []
    features_raw = []
    for g, f in genes_to_features.items():
        gene_names.append(g)
        features_raw.append(f.tolist())
    features_raw = np.array(features_raw)
    gene_names = np.array(gene_names)
    
    torch.save(torch.tensor(features_raw),f'{path}/features.pkl')
    with open(f'{path}/gene_names.txt','w') as f:
        for g in gene_names:
            f.write(g+'\n')
    
    # process adjacent matrix
    print(f'start process adjant matrix.')
    gene_to_index = {gene_names[i]:i for i in range(len(gene_names))}
    # Gobs = np.zeros([len(gene_names),len(gene_names)])
    Gobs_aggr = torch.zeros([len(gene_names),len(gene_names)]).to('cuda')
    for file in os.listdir(path):
        if file.endswith('.h5'):
            f = h5py.File(f'{path}/{file}', 'r')
            gn = np.array([i.decode() for i in np.array(f['gene_names'])[:,1]])
            adj = torch.tensor(np.array(f['network']))
            index = adj.to_sparse().indices()
            s,t = index[0],index[1]
            s_name = gn[s]
            t_name = gn[t]
            s_new = [gene_to_index[n] for n in s_name]
            t_new = [gene_to_index[n] for n in t_name]
            
            s_new = torch.tensor(s_new).to('cuda')
            t_new = torch.tensor(t_new).to('cuda')
            Gobs = torch.zeros([len(gene_names),len(gene_names)]).to('cuda')
            Gobs[s_new,t_new] = 1
            fine_name = file.split('_multi')[0]
            torch.save(Gobs,f'{path}/Gobs_{fine_name}_dense_adj.pkl')
            Gobs_aggr[s_new,t_new] += 1
            print(f'have processed a PPI data.')
            
    torch.save(Gobs_aggr,f'{path}/Gobs_aggr.pkl')

def split_train_val_test_gene(GeneFile='',DriverGeneFile='',NonDriverGeneFile=''):
    GeneFile = './data/EMOGI/data/gene_names.txt'
    DriverGeneFile='./data/EMOGI/data/796true.txt'
    NonDriverGeneFile='./data/EMOGI/data/2187false.txt'
    ongene = './data/EMOGI/data/ongene_human.txt'
    oncokb = './data/EMOGI/data/OncoKB_cancerGeneList.tsv'
    ongene = [i.split('\t')[1] for i in open('./data/EMOGI/data/ongene_human.txt','r').readlines()[1:]]
    oncokb = [i.split('\t')[0] for i in open('./data/EMOGI/data/OncoKB_cancerGeneList.tsv','r').readlines()[1:]]


    genes = [i.strip() for i in open(GeneFile,'r').readlines()]
    driver_genes = [i.strip() for i in open(DriverGeneFile,'r').readlines()]
    driver_genes = driver_genes + ongene + oncokb
    driver_genes = list(set(driver_genes).intersection(set(genes)))

    non_driver_genes = [i.strip() for i in open(NonDriverGeneFile,'r').readlines()]

    conflict_genes = list(set(driver_genes).intersection(non_driver_genes))

    driver_genes = [i for i in driver_genes if i not in conflict_genes]
    non_driver_genes = [i for i in non_driver_genes if i not in conflict_genes]

    labels = [1]*len(driver_genes) + [0]*len(non_driver_genes)
    labels = np.array(labels).reshape(-1,1)
    labelled_gene_ind = [genes.index(i) for i in driver_genes+non_driver_genes]
    labelled_gene_ind = np.array(labelled_gene_ind).reshape(-1,1)
    labelled_driver_ind = np.array([genes.index(i) for i in driver_genes])
    labelled_nondriver_ind = np.array([genes.index(i) for i in non_driver_genes])
    y = np.zeros(len(genes))
    y[labelled_driver_ind] = 1
    y[labelled_nondriver_ind] = 0

    data = np.concatenate([labelled_gene_ind,labels],axis=1)

    train, val_test = train_test_split(data,test_size=0.40,random_state=123,shuffle=True)
    val, test = train_test_split(val_test,test_size=0.50,random_state=456,shuffle=True)

    train_mask = np.zeros(len(genes))
    train_mask[train[:,0]] = 1

    val_mask = np.zeros(len(genes))
    val_mask[val[:,0]] = 1

    test_mask = np.zeros(len(genes))
    test_mask[test[:,0]] = 1
    
    
    torch.save(torch.tensor(y),'./data/EMOGI/data/y.pkl')
    torch.save(torch.tensor(train_mask.astype('bool')),'./data/EMOGI/data/train_mask.pkl')
    torch.save(torch.tensor(val_mask.astype('bool')),'./data/EMOGI/data/val_mask.pkl')
    torch.save(torch.tensor(test_mask.astype('bool')),'./data/EMOGI/data/test_mask.pkl')



def one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)



def edge_index_to_sparse_matrix(data):
    
    init_adj_sparse = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.tensor([1]*data.edge_index.shape[1],device=data.x.device),
                                   sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    
    init_adj = init_adj_sparse.to_dense()
    return init_adj


def weight_adj_to_adj(weight_adj,k, grad=False):
    # weight_adj = torch.rand([3,4]).softmax(dim=1)
    _,ind = weight_adj.topk(k,dim=1)
    adj = torch.zeros_like(weight_adj,requires_grad=False)
    adj.scatter_(1,ind,1)
    if grad:
        adj = (adj-weight_adj).detach() + weight_adj
    return adj

def weight_adj_to_adj_threshold(weight_adj,threshold):
    # weight_adj = torch.rand([3,4]).softmax(dim=1)
    adj = torch.zeros_like(weight_adj)
    adj[weight_adj>threshold] = 1.0
    adj = (adj-weight_adj).detach() + weight_adj
    return adj

def same_cluster_prob(distr_prob):
    """
    qy = torch.tensor([[0.7,0.2,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
    print(torch.mm(qy,qy.t()))
    """
    return torch.mm(distr_prob,distr_prob.t())

def graph_learner_normalize(adj):
        adj = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=1e-12)
        return adj


def acc(pred,target):
    pred = pred.max(1)[1]
    acc = pred.eq(target).sum().item() / pred.shape[0]
    return acc

def metrics(pred,data):
    acc_val = acc(pred[data.val_mask],data.y[data.val_mask])
    acc_test = acc(pred[data.test_mask],data.y[data.test_mask])
    acc_train = acc(pred[data.train_mask],data.y[data.train_mask])
    return acc_train,acc_val,acc_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.detach().cpu().numpy()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv_sqrt = np.sqrt(r_inv)
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    mx = r_mat_inv_sqrt.dot(mx)
    mx = mx.dot(r_mat_inv_sqrt)
    return mx

def normalize_torch(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv_sqrt = rowsum.pow_(-0.5)
    r_inv_sqrt.masked_fill_(r_inv_sqrt == float('inf'), 0.)
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.mm(r_mat_inv_sqrt,mx)
    mx = torch.mm(mx,r_mat_inv_sqrt)
    return mx

def gcn_norm_pyg(Gobs:list):
    Gobs_norm = []
    for obs in Gobs:
        obs = obs.to_sparse()
        ind = obs.indices()
        weight = obs.values()
        _edge_index, edge_weight = gcn_norm(
            ind, weight, Gobs[0].shape[0], False,
            dtype=Gobs[0].dtype)
        row, col = _edge_index
        obs_norm = SparseTensor(row=col, col=row, value=edge_weight,
                                        sparse_sizes=(Gobs[0].shape[0], Gobs[0].shape[0]))
    
        Gobs_norm.append(obs_norm.to_dense())
    return Gobs_norm


def edge_sampling(Gobs, neg_ration = 1.0,total_ration=1.0):
    
    tmp = Gobs
    tmp = tmp.int().to_sparse()
    edge_index = tmp.indices()
    edge_index = add_self_loops(edge_index,num_nodes=Gobs[0].shape[0])[0]
    # print(edge_index)
    edge_index_ub = negative_sampling(edge_index,Gobs[0].shape[0], num_neg_samples = int(neg_ration*edge_index.shape[1]))
    edge_index_obs = edge_index
    if total_ration<1.0:
        mask = torch.rand(edge_index_obs.shape[1]+edge_index_ub.shape[1])<total_ration
        edge_index_sample = torch.cat([edge_index_obs, edge_index_ub],dim=1)
        return edge_index_sample.t()[mask].t()
    return torch.cat([edge_index_obs, edge_index_ub],dim=1)


def edge_sampling_logical_and(Gobs:list, ration = 1.0):
    tmp = torch.ones_like(Gobs[0])
    for obs in Gobs:
        tmp = tmp.logical_and(obs)
    tmp = tmp.int().to_sparse()
    edge_index = tmp.indices()
    edge_index = add_self_loops(edge_index,num_nodes=Gobs[0].shape[0])[0]
    # print(edge_index)
    edge_index_ub = negative_sampling(edge_index,Gobs[0].shape[0], num_neg_samples = int(ration*edge_index.shape[1]))
    edge_index_obs = edge_index
    return torch.cat([edge_index_obs, edge_index_ub],dim=1)



def limit_label(train_mask,label,num_classes,lc=20):
    label_to_index = {i:[] for i in range(num_classes)}
    for i in range(train_mask.shape[0]):
        if train_mask[i].item()==True:
            label_to_index[label[i].item()].append(i)
    train_mask_limit = torch.zeros_like(train_mask).to(train_mask.device)

    for i in label_to_index.keys():
        train_mask_limit[label_to_index[i][:lc]] = True

    return train_mask_limit

def limit_edge(edge_index, percent=1.0):
    if percent == 1.0:
        return edge_index
    ind = list(range(edge_index.shape[1]))
    random.shuffle(ind)
    ind = torch.tensor(ind).to(edge_index.device)
    ind = ind[:int(ind.shape[0]*percent)]
    edge_index = edge_index.t()[ind].t()
    return edge_index

