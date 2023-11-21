import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphLearner(nn.Module):
    """adapted from WSGNN"""
    def __init__(self, input_size, num_pers=4):
        super(GraphLearner, self).__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        self.reset_parameters()
    def reset_parameters(self):
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
        mask = (attention > 0).detach().float()
        attention = attention * mask + 0 * (1 - mask)

        return attention

def graph_learner_cosine_similarity(embedding):
    """
    embedding = torch.randn(3,4).abs()
    embedding2 = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    similarity = torch.mm(embedding2, embedding2.T)
    """

    embedding = embedding / (torch.norm(embedding, dim=-1, keepdim=True)+1e-8)
    similarity = torch.mm(embedding, embedding.T)
    mask = (similarity > 0).detach().float()
    similarity = similarity * mask + 0 * (1 - mask)
    return similarity
