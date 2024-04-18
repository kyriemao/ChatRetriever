from IPython import embed

import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from datasets import IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import HfArgumentParser, AutoTokenizer, AutoModel, AutoModelForCausalLM

from kindred.arguments import ModelArguments, DataArguments, EvalArguments
from kindred.models.model_utils import load_model, unified_model_forward
from kindred.models.splade import Splade
from kindred.data.dataset import IndexingCollator, json_psg_generator
from kindred.utils import write_running_args, pstore, pload
from kindred.retrieval_toolkits.splade_tool import IndexDictOfArray
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import gc

gc.collect()
torch.cuda.empty_cache()

import torch

def create_random_attention_mask(bs, seq_length, num_head, mask_prob=0.2):
    # Create a full attention mask
    attention_mask = torch.ones((bs, num_head, seq_length, seq_length))
    
    # Generate a random mask with the specified probability
    random_mask = torch.bernoulli(torch.full((bs, num_head, seq_length, seq_length), mask_prob))
    
    # Apply the random mask to the attention mask (0s in random_mask will block attention)
    attention_mask = attention_mask * (1 - random_mask)

    return attention_mask



if __name__ == "__main__":
    device = "cpu" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
    # model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = ["the cat sat here", "I like you", "I am a good boy but you are not."]
    inputs = tokenizer(texts, padding=True)
    inputs['input_ids'] = torch.tensor(inputs['input_ids'])
    bs = len(inputs['input_ids'])
    seq_len = len(inputs['attention_mask'][0])
    input_ids = inputs['input_ids'].to(device)
    num_heads = 16
    attention_mask = create_random_attention_mask(bs, seq_len, num_heads)
    # attention_mask = attention_mask.squeeze(1)
    attention_mask = torch.tensor(attention_mask)
    attention_mask=attention_mask.bool().to(device)
  
    # a = model(input_ids)
    # attention_mask = 'aa'
    embed()
    input()
    a = model(input_ids, attention_mask=attention_mask)
    a = model(input_ids, attention_mask=attention_mask, use_4d_attn_mask=True)
    
    # logits_0 = model(input_0).logits
    # logits_1 = model(input_1, attention_mask=mask_1.bool())
    
