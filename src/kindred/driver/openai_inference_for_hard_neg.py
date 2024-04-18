from openai import OpenAI
from transformers import HfArgumentParser
from kindred.arguments import DataArguments
from kindred.data.data_utils import load_dataset
from kindred.utils import write_running_args, mkdirs
from IPython import embed
from tqdm import tqdm

import os
import time
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Given a query and its positive relevant passage, generate three diverse hard negative passages that only appear relevant to the query. Each passage should be no more than 256 tokens. Your output must always be a JSON object only, containing key "Neg-#number" for each generated neg passage. Do not explain yourself or output anything else. Be creative!

Query: {query}

Positive Passage: {pos_passage}
""".lstrip('\n')



def main():
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()[0]
    if data_args.data_output_dir:
        mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir, allow_dir_exist=data_args.allow_dir_exist)
    write_running_args(data_args.data_output_dir, [data_args])
    
    # 1. model and data
    openai_key = ""
    client = OpenAI(api_key=openai_key)
    model_type = "openai"
    tokenizer = None
    matching_dataset = load_dataset(tokenizer, model_type, data_args)
    
    rewrite_dict = {}
    with open("/share/kelong/kindred/src/rewrites/ultrachat_rewrite.qaw.jsonl", "r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            rewrite_dict[line['sample_idx']] = line['rewrite']

    # begin to call openai API
    output_file = os.path.join(data_args.data_output_dir, "hard_neg.jsonl")
    finished_sample_ids = set()
    with open(output_file) as fr:
        for line in tqdm(fr):
            line = json.loads(line)
            finished_sample_ids.add(line['sample_idx'])
    
    # 2. invoke openai  
    with open(output_file, "a+") as fw:
        for sample in tqdm(matching_dataset.samples):
            if sample.sample_idx in finished_sample_ids:
                continue
            if sample.sample_idx in rewrite_dict:
                query = rewrite_dict[sample.sample_idx]
            else:
                # query = sample.query
                continue
            pos = " ".join(sample.pos_psg.split(" ")[ :384])
            prompt = PROMPT_TEMPLATE.format(query=query, pos_passage=pos)
            
            max_try = 2
            success = False
            while max_try > 0:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        response_format={ "type": "json_object" },
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "{}".format(prompt)},
                        ],
                    )
                    success = True
                    break
                except:
                    max_try -= 1
                    time.sleep(20)
                
            if success:
                res = json.loads(response.choices[0].message.content)
                res['sample_idx'] = sample.sample_idx
                
                fw.write(json.dumps(res) + '\n')
                fw.flush()
      
        
if __name__ == '__main__':
    main()