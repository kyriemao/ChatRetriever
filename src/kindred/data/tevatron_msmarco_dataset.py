import json
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
from .data_sample import MatchingSample
from .dataset import MatchingDataset
from kindred.arguments import DataArguments
from IPython import embed
import random

# Specilized to peitian's dataset format
class TevatronMsmarcoDataset(MatchingDataset):
    def __init__(self, 
                 tokenizer:AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent):
        if not isinstance(use_data_percent, list):
            use_data_percent = [use_data_percent]
        super().__init__(tokenizer, model_type, data_args, use_data_percent)
        self.samples = self.load_data_from_file(data_args.tevatron_msmarco_data_path_list)
        
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self._text_formatting()
    
    def load_data_from_file(self, data_path_list: List[str]):
        all_samples = []
        data_idx = 0
        for data_path in tqdm(data_path_list, desc="Processing tevatron msmarco data..."):
            samples = []
            with open(data_path, 'r') as f:
                for line in tqdm(f, desc='loading {}...'.format(data_path)):
                    line = json.loads(line)
                    # if len(samples) > 20000:
                    #     break
                    query = line['query']
                    pos_psgs, neg_psgs = [], []
                    if 'positive_passages' in line:
                        pos_psgs = line['positive_passages']
                        pos_psgs = [x['title'] + " " + x['text'] for x in pos_psgs]
                    if self.neg_type == 'hard':
                        neg_psgs = line['negative_passages']
                        neg_psgs = [x['title'] + " " + x['text'] for x in neg_psgs]

                    sample_idx = str(line['query_id'])
                    for pos_psg in pos_psgs:
                        sample = MatchingSample(sample_idx=sample_idx, query=query, pos_psg=pos_psg, neg_psgs=neg_psgs)
                        samples.append(sample)
                        break
            
            samples = self.sample_part_of_data(samples, self.use_data_percent[data_idx])
            all_samples.extend(samples)
            data_idx += 1                
            
        return all_samples