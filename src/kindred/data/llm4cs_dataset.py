import json
import copy
from IPython import embed
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer

from .dataset import MatchingDataset, MatchingSample
from kindred.arguments import DataArguments


class LLM4CSDataset(MatchingDataset):
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent):
        if not isinstance(use_data_percent, list):
            use_data_percent = [use_data_percent]
        super().__init__(tokenizer, model_type, data_args, use_data_percent)             
        self.only_last_response = data_args.only_last_response
        
        self.samples = self.load_data_from_file(data_args.llm4cs_data_path_list)
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self._text_formatting()
        
    
    def load_data_from_file(self, data_path_list: List[str]):
        all_samples = []
        data_idx = 0
        for data_path in tqdm(data_path_list, desc="Processing llm4cs data..."):
            samples = []
            with open(data_path, 'r') as f:
                for line in tqdm(f):
                    line = json.loads(line)
                    sample_idx = line['sample_id']
                    rewrites = line['predicted_rewrite']
                    rewrite = rewrites[0]
                    query = rewrite
                    
                    history, pos_psg, neg_psgs = [], [], []
                    sample = MatchingSample(sample_idx=sample_idx,
                                            query=query,
                                            rewrite=rewrite,
                                            history=history,
                                            pos_psg=pos_psg,
                                            neg_psgs=neg_psgs)                
                    samples.append(sample)
            
            samples = self.sample_part_of_data(samples, self.use_data_percent[data_idx])
            all_samples.extend(samples)
            data_idx += 1

        return all_samples
    