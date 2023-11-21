import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from encoder import GCN ,Dense_GCN_Net
from utils import * 
from graph_learner import GraphLearner
from loss import LossFn


class PYModel(nn.Module):
    def __init__(self, args):
        super(PYModel, self).__init__()
        self.py_encoder = GCN(args.input_dim, args.num_class, args.hidden_dim, args.dropout)
        self.dropout = args.dropout
        self.args = args

    def forward(self, data):
        x = data.x
        hidden, out = self.py_encoder(x,data.edge_index)
        return hidden, out

    
class PGModel(nn.Module):
    '''
    Prior on G
    '''
    def __init__(self, args):
        super(PGModel, self).__init__()
        self.args = args
        self.dropout = args.dropout

        if args.G_prior == 'sbm':
            if args.learn_sbm:
                self.P0 = nn.Parameter(torch.tensor(args.P0))
                self.P1 = nn.Parameter(torch.tensor(args.P1))
            else:
                self.P0 = torch.tensor(args.P0)
                self.P1 = torch.tensor(args.P1)

        elif args.G_prior == 'lsm':
            self.x_trans = nn.Linear(args.input_dim,args.hidden_dim)
            self.w = nn.Linear(2*(args.hidden_dim+args.num_class),1)

    def forward(self,data,y_prob=None):
        if self.args.G_prior =='sbm':
            p0 = torch.sigmoid(self.P0)
            p1 = torch.sigmoid(self.P1)
            return p0, p1
        elif self.args.G_prior == 'lsm':
            x = F.dropout(data.x, p=self.dropout,training=self.training)
            x_trans = F.relu(self.x_trans(x))
            
            # compute lsm prior of edges
            edge_index_train = data.edge_index_train
            source, target = edge_index_train[0], edge_index_train[1]
            x_source = F.embedding(source,x_trans)
            y_source = F.embedding(source,y_prob)
            x_target = F.embedding(target,x_trans)
            y_target = F.embedding(target,y_prob)
            xy = torch.cat([x_source,y_source,x_target,y_target],dim=1)
            edge_prob = self.w(xy)
            edge_prob = torch.sigmoid(edge_prob)
            return edge_prob

class P_GObs_G_Model(nn.Module):
    '''
    parameter for likelihood of Gobs
    '''
    def __init__(self,args):
        super(P_GObs_G_Model, self).__init__()
       
        self.alpha = nn.Parameter(torch.tensor(args.alpha))
        self.beta = nn.Parameter(torch.tensor(args.beta))

        self.init_alpha = torch.tensor(args.alpha)
        self.init_beta = torch.tensor(args.beta)

    def reset_parameter(self):
        self.alpha = nn.Parameter(self.init_alpha)
        self.beta = nn.Parameter(self.init_beta)


class QG_QY_Model(nn.Module):
    def __init__(self,args):
        super(QG_QY_Model, self).__init__()
        self.args = args
        self.P_GObs_G_Model = P_GObs_G_Model(args)
        self.PGModel = PGModel(args)

        self.loss_fn = LossFn()
        self.dropout=args.dropout
        self.qy_encoder = Dense_GCN_Net(args.input_dim,args.hidden_dim,args.num_class,args.dropout)
        self.graph_learner = GraphLearner(args.input_dim,args.num_heads)

    def forward(self, data):
        
        raw_qg = self.graph_learner(data.x)
        qg = raw_qg / torch.clamp(torch.sum(raw_qg, dim=-1, keepdim=True), min=1e-12)
        qg = (1-self.args.lmd)*qg + self.args.lmd*data.adj_norm
        raw_qg = (1-self.args.lmd)*raw_qg + self.args.lmd*data.adj
        _,qy_1 = self.qy_encoder(data.x,data.adj_norm)
        _,qy_2 = self.qy_encoder(data.x,qg)
        qy = 0.5*qy_1 + 0.5*qy_2
        return qy,raw_qg


        
class GenerativeModel(torch.nn.Module):
    def __init__(self,args):
        super(GenerativeModel, self).__init__()
        self.PYModel = PYModel(args)
        self.QG_QY_Model = QG_QY_Model(args)
        self.args = args
        self.loss_fn = LossFn()
    
