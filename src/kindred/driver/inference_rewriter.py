import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from transformers import HfArgumentParser
from kindred.arguments import ModelArguments, DataArguments, EvalArguments
from kindred.models.model_utils import load_model
from kindred.data.data_utils import load_dataset
from kindred.data.rewriter_dataset import RewriterDataset, inference_collator
from kindred.utils import write_running_args, mkdirs
from IPython import embed
from tqdm import tqdm

import os
import re
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_func(output):
    pattern = r"### Assistant:\nRewritten Query:\n"
    match = re.search(pattern + "(.*)", output, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    if data_args.data_output_dir:
        mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
    write_running_args(data_args.data_output_dir, [model_args, data_args, eval_args])
    
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)
    
    # 2. load data
    matching_dataset = load_dataset(tokenizer, model_args.model_type, data_args)
    inference_dataset = RewriterDataset(matching_dataset, inference=True)
    
    inference_dataloader = DataLoader(inference_dataset, 
                                      batch_size=eval_args.per_device_eval_batch_size, 
                                      shuffle=False, 
                                      collate_fn=lambda x:inference_collator(x, tokenizer, data_args.max_q_len))
    
    # 3. inference
    kwargs = {"max_new_tokens": 128}
    output_file = os.path.join(data_args.data_output_dir, "ultrachat_rewrite.qaw.jsonl")
    with open(output_file, "a+") as fw:
        for batch in tqdm(inference_dataloader, desc="Inference..."):
            batch['input_ids'] = batch['input_ids'].to('cuda')
            batch['attention_mask'] = batch['attention_mask'].to('cuda')
            generated_tokens = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], do_sample=True, **kwargs)
            
            for sample_idx, tokens in zip(batch['sample_ids'], generated_tokens):
                decoded_output = tokenizer.decode(tokens, skip_special_tokens=True)
                rewrite = parse_func(decoded_output)
                if rewrite is not None:
                    fw.write(json.dumps({"sample_idx": sample_idx, "rewrite": rewrite}) + '\n')
                    fw.flush()
   
if __name__ == '__main__':
    main()