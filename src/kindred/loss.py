import torch
import torch.nn as nn
import torch.nn.functional as F

class RankingLoss:
    def __init__(self, temperature):
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()
    
    def __call__(self, query_embs, pos_psg_embs, neg_psg_embs):
        '''
        query_embs: B * dim (B is actuall n_query of this gpu)
        pos_psg_embs: (B + x) * dim (x is the total number of postive psgs of other gpus)
        neg_psg_embs: x' * dim, x' is the total number of negative psgs of other gpus, Optional
        '''
        n_query = len(query_embs) # n_query = per_gpu_batch_size
        doc_embs = pos_psg_embs
        if neg_psg_embs is not None:
            doc_embs = torch.cat([pos_psg_embs, neg_psg_embs], dim=0)
        score_mat = query_embs.mm(doc_embs.T) # n_query * n_psgs
        score_mat /= self.temperature
        label_mat = torch.arange(n_query).to(query_embs.device)  # only the first n_query docs are positive
        loss = self.loss_func(score_mat, label_mat)

        return loss
    

class KDLoss:
    def __init__(self, reduction='mean'):
        self.loss_func = nn.MSELoss(reduction=reduction)
    
    def __call__(self, student_embs, teacher_embs):
        teacher_embs = teacher_embs.detach()
        assert teacher_embs.requires_grad == False
        return self.loss_func(student_embs, teacher_embs)



class Regularizer:
    def __init__(self, reg_type):
        self.reg_type = reg_type
    
    def __call__(self, embs):
        if self.reg_type == "L0":
            return torch.count_nonzero(embs, dim=-1).float().mean()
        elif self.reg_type == "L1":
            return torch.sum(torch.abs(embs), dim=-1).mean()
        elif self.reg_type == "FLOPS":
            return torch.sum(torch.mean(torch.abs(embs), dim=0) ** 2)
        else:
            raise NotImplementedError