import json
import copy
from IPython import embed
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer

from .dataset import MatchingDataset, MatchingSample
from kindred.arguments import DataArguments


class ConvSearchDataset(MatchingDataset):
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent):
        if not isinstance(use_data_percent, list):
            use_data_percent = [use_data_percent]
        super().__init__(tokenizer, model_type, data_args, use_data_percent)             
        self.only_last_response = data_args.only_last_response
        
        self.samples = self.load_data_from_file(data_args.convsearch_data_path_list)
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self._text_formatting()
        
    
    def load_data_from_file(self, data_path_list: List[str]):
        all_samples = []
        data_idx = 0
        for data_path in tqdm(data_path_list, desc="Processing convsearch data..."):
            samples = []
            with open(data_path, 'r') as f:
                data = json.load(f)
            for conv in tqdm(data, desc="Processing {}...".format(data_path)):
                history = []
                history_token_size_list = []
            
                for turn in conv['turns']:
                    sample_idx = "{}_{}".format(conv['conv_id'], turn['turn_id'])
                    query = turn['question']
                    rewrite = None
                    if "manual_rewrite" in turn:
                        rewrite = turn['manual_rewrite']
                        
                    # passage
                    pos_psgs, neg_psgs = [], []
                    if "pos_psg_text" in turn:
                        pos_psgs = turn['pos_psg_text']
         
                    if self.neg_type == 'in_batch':
                        neg_psgs = []
                    elif self.neg_type == 'bm25_hard':
                        neg_psgs = turn['bm25_hard_neg_psgs_text']

                    pos_psg = None if len(pos_psgs) == 0 else pos_psgs[0]
                    
                    cur_query_len = len(self.tokenizer.encode(query))
                    while sum(history_token_size_list) + cur_query_len > self.max_q_len - 30: # 30 for template tokens
                        history = history[2:]
                        history_token_size_list = history_token_size_list[2:]
                    
                    sample = MatchingSample(sample_idx=sample_idx,
                                            query=query,
                                            rewrite=rewrite,
                                            history=copy.deepcopy(history),
                                            pos_psg=pos_psg,
                                            neg_psgs=neg_psgs)                
                    samples.append(sample)

                    # update history
                    if self.only_last_response and "response" in turn and len(history) > 0:
                        history.pop()   # pop the last response
                    history.append({"role":"User", "text": query})
                    history_token_size_list.append(cur_query_len)
                    if "response" in turn: 
                        history.append({"role": "Assistant", "text": turn['response']})
                        history_token_size_list.append(len(self.tokenizer.encode(turn['response'])))
            
            samples = self.sample_part_of_data(samples, self.use_data_percent[data_idx])
            all_samples.extend(samples)
            data_idx += 1

        return all_samples
    