import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

import math


class GCN(nn.Module):
    def __init__(self, num_feature, num_class, hidden_size, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_class)

        self.dropout = dropout
        self.activation = F.relu

    def forward(self, feature, adj,edge_weight=None):
        x1 = self.activation(self.conv1(feature, adj,edge_weight=edge_weight))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, adj,edge_weight=edge_weight)
        return x1, x2



class DenseGCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super(DenseGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.input_dim)
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Dense_GCN_Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.5) -> None:
        super(Dense_GCN_Net,self).__init__()
        self.gcn1 = DenseGCN(input_dim,hidden_dim)
        self.gcn2 = DenseGCN(hidden_dim,output_dim)
        self.dropout = dropout
    def forward(self,x,adj):
        # x = F.dropout(x,p=self.dropout,training=self.training)
        x1 = F.relu(self.gcn1(x,adj))
        x1 = F.dropout(x1,p=self.dropout,training=self.training)
        x2 = self.gcn2(x1,adj)
        return x1, x2

