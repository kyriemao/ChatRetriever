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
class MsmarcoDataset(MatchingDataset):
    def __init__(self, tokenizer:AutoTokenizer, model_type: str, data_args: DataArguments):
        super().__init__(tokenizer, model_type, data_args)
        self.samples = self.load_data_from_file(data_args.msmarco_data_path_list)
        
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self.sample_part_of_data()
        self._text_formatting()
    
    def load_data_from_file(self, data_path_list: List[str]):
        samples = []
        for data_path in tqdm(data_path_list, desc="Processing msmarco data..."):
            with open(data_path, 'r') as f:
                for line in tqdm(f, desc='loading {}...'.format(data_path)):
                    line = json.loads(line)
                    query = line['query']
                    pos_psgs, neg_psgs = [], []
                    if 'pos' in line:
                        pos_psgs = line['pos']
                    if self.neg_type == 'hard':
                        neg_psgs = line['neg'][15:] # top 15 passages are likely to be false negative
                        neg_psgs = random.sample(neg_psgs, self.neg_num)
                        
                    sample_idx = str(line['query_id'])
                    for pos_psg in pos_psgs:
                        sample = MatchingSample(sample_idx=sample_idx, query=query, pos_psg=pos_psg, neg_psgs=neg_psgs)
                        samples.append(sample)

        return samples
