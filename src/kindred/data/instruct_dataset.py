import json
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
from .dataset import MatchingSample, MatchingDataset
from kindred.arguments import DataArguments


class InstructDataset(MatchingDataset):
    def __init__(self, 
                 tokenizer:AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent):
        if not isinstance(use_data_percent, list):
            use_data_percent = [use_data_percent]
        super().__init__(tokenizer, model_type, data_args, use_data_percent)
        self.samples = self.load_data_from_file(data_args.instruct_data_path_list)
        
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self._text_formatting()
    
    def load_data_from_file(self, data_path_list: List[str]):
        all_samples = []
        data_idx = 0
        for data_path in tqdm(data_path_list, desc="Processing instruct data..."):
            samples = []
            with open(data_path, 'r') as f:
                for line in tqdm(f, desc='loading {}...'.format(data_path)):
                    line = json.loads(line)
                    # if len(samples) > 20000:
                    #     break
                    sample_idx = line['id']
                    query = line['instruction']
                    if len(query) == 0:
                        continue
                    pos_psg = line['response']
                    sample = MatchingSample(sample_idx=sample_idx, query=query, pos_psg=pos_psg)
                    samples.append(sample)

            samples = self.sample_part_of_data(samples, self.use_data_percent[data_idx])
            all_samples.extend(samples)
            data_idx += 1
            
        return all_samples