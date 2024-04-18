import json
import copy
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer
from .dataset import MatchingSample, MatchingDataset
from kindred.arguments import DataArguments


class ChatDataset(MatchingDataset):
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent):
        if not isinstance(use_data_percent, list):
            use_data_percent = [use_data_percent]
        super().__init__(tokenizer, model_type, data_args, use_data_percent)
        self.samples = self.load_data_from_file(data_args.chat_data_path_list)
        
        if self.filter_no_pos:
            self.filter_no_pos_sample()
        self._text_formatting()

    def load_data_from_file(self, data_path_list: List[str]):
        all_samples = []
        data_idx = 0
        for data_path in tqdm(data_path_list, desc="Processing chat data..."):
            chat_samples = []
            with open(data_path, 'r') as f:
                for line in tqdm(f, desc='loading {}...'.format(data_path)):
                    line = json.loads(line)
                    # if len(chat_samples) > 20000:
                    #     break
                    conv_id = line['conv_id'] # tag_conv-idx
                    conversation = line['conversation']
                    history = []
                    turn_idx = 0
                    if len(conversation) == 0 or conversation[0]['role'] != 'User':
                        continue
                    
                    for i in range(len(conversation)-1):
                        if conversation[i]['role'] == 'User':
                            query = conversation[i]['text']
                            if 'rewrite' in conversation[i]:
                                rewrite = conversation[i]['rewrite']
                            else:
                                rewrite = query
                            assert conversation[i+1]['role'] == 'Assistant'
                            pos_psg = conversation[i+1]['text']
                            neg_psgs = None
                            if 'neg_texts' in conversation[i+1]:
                                neg_psgs = conversation[i+1]['neg_texts']
                            
                        elif conversation[i]['role'] == 'Assistant':
                            history.append(conversation[i])
                            continue
                        else:
                            raise KeyError("Unknown role: {}".format(conversation[i]['role']))
                        
                        sample_idx = "{}-{}".format(conv_id, turn_idx)
                        turn_idx += 1
                        if len(query) == 0:
                            continue
                        sample = MatchingSample(sample_idx=sample_idx, 
                                                query=query, 
                                                rewrite=rewrite,
                                                history=copy.deepcopy(history),
                                                pos_psg=pos_psg,
                                                neg_psgs=neg_psgs)
                        chat_samples.append(sample)
                        history.append(conversation[i])
            
            samples = self.sample_part_of_data(chat_samples, self.use_data_percent[data_idx])
            data_idx += 1
            all_samples.extend(samples)

        return all_samples