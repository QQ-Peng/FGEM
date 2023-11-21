import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import same_cluster_prob

class LossFn(nn.Module):
    def __init__(self):
        super(LossFn,self).__init__()
    
    def nll_loss_(self,y,y_target,log_softmax=True):
        log_py = F.log_softmax(y,dim=1) if log_softmax else y
        loss = F.nll_loss(log_py,y_target)
        return loss

    def P_Gobs(self,Gobs,qg,alpha,beta,edge_index_train,num_Gobs=4,sum=True):
        if sum:
            E = sum(Gobs)
        else:
            E = Gobs
        E = E.detach()

        s,t = edge_index_train[0], edge_index_train[1]
        E_train = E[s,t]
        E_train = E_train.detach()
        qg_train = qg[s,t]
        
        loss_P_Gobs = -torch.mean(qg_train*(E_train*torch.log(alpha)+(num_Gobs-E_train)*torch.log(1-alpha))+\
            (1-qg_train)*(E_train*torch.log(beta)+(num_Gobs-E_train)*torch.log(1-beta)))
        return loss_P_Gobs

    def PY_ub(self,py_ub,qy_ub,log_softmax=True):
        if log_softmax:
            log_py_ub = F.log_softmax(py_ub,dim=1)
            log_qy_ub = F.log_softmax(qy_ub,dim=1)
            qy_ub = torch.exp(log_qy_ub)
        else:
            log_py_ub = py_ub
        loss = -torch.mean(qy_ub*log_py_ub)
        return loss

    def QY_ub(self,qy_ub,log_softmax=True):
        log_qy_ub = F.log_softmax(qy_ub,dim=1) if log_softmax else qy_ub
        qy_ub = torch.exp(log_qy_ub)
        loss = torch.mean(qy_ub*log_qy_ub)
        return loss
    
    def PY_QY_ub(self,py_ub,qy_ub,log_softmax=True):
        if log_softmax:
            log_py_ub = F.log_softmax(py_ub,dim=1)
            log_qy_ub = F.log_softmax(qy_ub,dim=1)
            qy_ub = torch.exp(log_qy_ub)
        else:
            log_py_ub = py_ub
        loss = -torch.mean(qy_ub*log_py_ub) + torch.mean(qy_ub*log_qy_ub)
        return loss
    
    def PG_QG(self,res,softmax=True,G_prior='sbm'):
        if G_prior == 'sbm':
            qy,qg,p0,p1,edge_index_train = res
            s,t = edge_index_train[0], edge_index_train[1]
            if softmax:
                qy = F.softmax(qy,dim=1)
            same_class_prob = same_cluster_prob(qy)
            qg_train = qg[s,t]
            same_class_prob_train = same_class_prob[s,t]

            loss_pg = -torch.mean(qg_train*(torch.log(p0)*same_class_prob_train+torch.log(p1)*(1-same_class_prob_train))+\
                (1-qg_train)*(torch.log(p0)*same_class_prob_train+torch.log(p1)*(1-same_class_prob_train)))
            
            qg_train_add = qg_train + 1e-5
            loss_qg = torch.mean(qg_train*torch.log(qg_train_add))
            return loss_pg + loss_qg

        elif G_prior == 'lsm':
            qg, edge_prior, edge_index_train = res
            s,t = edge_index_train[0], edge_index_train[1]
            qg_train = qg[s,t]
            qg_train = qg_train.view(*edge_prior.shape)
            
            loss = -torch.mean(qg_train*torch.log((edge_prior+1e-6)/(qg_train+1e-6)) + \
                (1-qg_train)*torch.log((edge_prior+1e-6)/(1-qg_train+1e-6)))
            return loss


