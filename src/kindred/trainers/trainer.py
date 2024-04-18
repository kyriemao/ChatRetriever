import math
from IPython import embed
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Trainer

from dataclasses import dataclass

from kindred.models.model_utils import unified_model_forward
from kindred.loss import RankingLoss, KDLoss, Regularizer


# mix instruction tuning and matching learning
class MatchingTrainer(Trainer):
    
    def __init__(self, 
                 model_args,
                 match_training_args,
                 freezed_model: torch.nn.Module = None,  
                 **kwargs):
        super(MatchingTrainer, self).__init__(**kwargs)
        self.model_type = model_args.model_type
        self.normalize_emb = model_args.normalize_emb
        
        self.ranking_loss_func = None
        self.kd_loss_func = None
        self.regularizer = None
        
        if 'ranking' in match_training_args.loss_type:
            self.ranking_loss_func = RankingLoss(match_training_args.temperature)
        if 'kd' in match_training_args.loss_type:
            self.kd_loss_func = KDLoss()
        if match_training_args.regularizer_type:
            self.regularizer = Regularizer(reg_type=match_training_args.regularizer_type)
        self.ranking_loss_weight = match_training_args.ranking_loss_weight
        self.kd_loss_weight = match_training_args.kd_loss_weight
        self.regularization_weight = match_training_args.regularization_weight
        self.inst_loss_weight = match_training_args.inst_loss_weight
        self.label_smoother = LabelSmoother()
        
        if match_training_args.min_lr > 0:
            assert self.args.lr_scheduler_type == 'cosine'
            assert match_training_args.min_lr < self.args.learning_rate
            num_cycles = self.get_num_cycles_for_cosine_lr_scheduler(self.args.learning_rate, match_training_args.min_lr)
            self.args.lr_scheduler_kwargs['num_cycles'] = num_cycles
            
        self.use_query_mask = match_training_args.use_query_mask
        
        self.freezed_model = freezed_model
        self.freezed_model_type = model_args.freezed_model_type
    
    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor], emb_dim: int, dtype):
        '''
        Support gathering different sizes of tensors (even 0) from the other gpus through padding
        refer to https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
        '''
        if t is not None:
            t = t.contiguous()
        
        cuda_device = f'cuda:{torch.distributed.get_rank()}'
        world_size = dist.get_world_size()
        local_size = torch.tensor(t.shape[0] if t is not None else 0, device=cuda_device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        size_diff = max(all_sizes).item() - local_size.item()
        # if all gpus have no data, return None
        if max(all_sizes).item() == 0:
            return None
        if size_diff > 0:
            padding = torch.zeros((size_diff, emb_dim), device=cuda_device, dtype=dtype)
            t = torch.cat((t, padding)) if t is not None else padding
            
        all_tensors_padded = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, t)
        # cut the padding
        all_tensors = []
        for iter_t, size in zip(all_tensors_padded, all_sizes):
            all_tensors.append(iter_t[:size])
        # always put tensors of the current rank at the first place
        all_tensors[dist.get_rank()] = t
        all_tensors.pop(dist.get_rank())
        all_tensors = [t] + all_tensors
    
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    
    def compute_loss(self, model, inputs):
        sample_ids = inputs.pop('sample_ids', None)
        query_input_encodings = inputs.pop('query_input_encodings')
        pos_psg_input_encodings = inputs.pop('pos_psg_input_encodings')
        neg_psg_input_encodings = inputs.pop('neg_psg_input_encodings')
        teacher_input_encodings = inputs.pop('teacher_input_encodings')
        inst_input_encodings = inputs.pop('inst_input_encodings', None)
        
        to_run_model = model
        to_run_model_type = self.model_type
        query_embs = unified_model_forward(to_run_model, to_run_model_type, query_input_encodings, self.normalize_emb)
        
        if self.freezed_model:
            to_run_model = self.freezed_model
            to_run_model.to(query_embs.device)
            to_run_model_type = self.freezed_model_type
            
        ranking_loss, kd_loss, reg_loss, inst_loss = 0.0, 0.0, 0.0, 0.0
        pos_psg_embs, neg_psg_embs, teacher_embs = None, None, None
        
        if self.ranking_loss_func:
            pos_psg_embs = unified_model_forward(to_run_model, to_run_model_type, pos_psg_input_encodings, self.normalize_emb)
            if neg_psg_input_encodings:
                neg_psg_embs = unified_model_forward(to_run_model, to_run_model_type, neg_psg_input_encodings, self.normalize_emb)
            
            emb_dim = query_embs.shape[1] # for cross gpu broadcasting
            dtype = query_embs.dtype # for cross gpu broadcasting
            pos_psg_embs = self._dist_gather_tensor(pos_psg_embs, emb_dim, dtype)
            neg_psg_embs = self._dist_gather_tensor(neg_psg_embs, emb_dim, dtype)
            ranking_loss = self.ranking_loss_func(query_embs, pos_psg_embs, neg_psg_embs) * self.ranking_loss_weight
        
        if self.kd_loss_func:
            teacher_embs = unified_model_forward(to_run_model, to_run_model_type, teacher_input_encodings, self.normalize_emb)
            kd_loss = self.kd_loss_func(query_embs, teacher_embs) * self.kd_loss_weight
            
        if self.regularizer:
            reg_loss = self.regularizer(query_embs)
            reg_loss = reg_loss * self.regularization_weight

        # instruction tuning
        if inst_input_encodings and self.inst_loss_weight > 0:
            inst_labels = inst_input_encodings.pop("labels")
            if self.use_query_mask:
                attention_mask = inst_input_encodings.pop("qm_attn_mask")
            else:
                attention_mask = inst_input_encodings.pop("attention_mask")
            input_ids = inst_input_encodings.pop("input_ids")
            
            if "qwen15" in self.model_type or "mistrial" in self.model_type:
                inst_outputs = model(input_ids, attention_mask=attention_mask.unsqueeze(1))
            else:
                inst_outputs = model(input_ids, attention_mask=attention_mask, use_4d_attn_mask=self.use_query_mask)
            inst_loss = self.label_smoother(inst_outputs, inst_labels, shift_labels=True)
            inst_loss *= self.inst_loss_weight

        loss = ranking_loss + kd_loss + reg_loss + inst_loss
        # print("ranking loss {}, kd loss {}, reg loss {}, inst loss {}".format(ranking_loss, kd_loss, reg_loss, inst_loss))
        return loss
    
    def get_num_cycles_for_cosine_lr_scheduler(self, init_lr, min_lr):
        y = 2 * (min_lr / init_lr) - 1
        num_cycles = math.acos(y) / math.pi * 0.5
        return num_cycles
    


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss