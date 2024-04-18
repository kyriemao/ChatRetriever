import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5PreTrainedModel


class GTR(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # T5EncoderModel.__init__(self, config)
        self.t5_encoder = T5EncoderModel(config)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size, bias=False) # gtr has
        self.activation = torch.nn.Identity()
        self.model_parallel = False
        
    def pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1)
        original_dtype = sum_embeddings.dtype
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_embs = sum_embeddings / sum_mask
        pooled_embs = pooled_embs.to(dtype=original_dtype)
        return pooled_embs
    
    def forward(self, input_ids, attention_mask):
        output = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.pooling(output, attention_mask)   
        output = self.activation(self.embeddingHead(output))
        output = F.normalize(output, p=2, dim=1)
        
        return output
