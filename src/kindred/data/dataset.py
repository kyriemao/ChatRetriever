from IPython import embed
import json
import random
from tqdm import tqdm, trange
from .data_sample import MatchingSample
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from kindred.arguments import DataArguments  
from kindred.data.text_format import TextFormatter

import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

    
class MatchingDataset(Dataset):
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments,
                 use_data_percent: float=1.0):
        self.samples = []
        
        self.max_q_len = data_args.max_q_len
        self.max_p_len = data_args.max_p_len
        
        self.neg_type = data_args.neg_type
        self.neg_num = data_args.neg_num
        
        self.use_data_percent = use_data_percent
        
        self.filter_no_pos = data_args.filter_no_pos
        
        self.tokenizer = tokenizer
        self.text_formatter = TextFormatter(self.tokenizer, model_type, data_args)
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]
    
    def filter_no_pos_sample(self):
        filtered_samples = []
        for sample in self.samples:
            if sample.pos_psg:
                filtered_samples.append(sample)
      
        self.samples = filtered_samples
        logger.info("Filtered {} samples with positive passages.".format(len(self.samples)))         
        
    def sample_part_of_data(self, samples, use_data_percent):
        if use_data_percent >= 1.0:
            return samples
        n = int(use_data_percent * len(samples))  
        random.seed(7)
        random.shuffle(samples)
        return samples[:n]
            
    def merge(self, dataset):
        self.samples += dataset.samples
        
    def _text_formatting(self):
        for i in trange(len(self.samples), desc="Performing text formatting..."):
            self.text_formatter(self.samples[i])


class MatchingCollator:
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 data_args: DataArguments,
                 q_suffix: str,
                 p_suffix: str,
                 is_eval: bool, 
                 mix_inst: bool=False,
                 use_query_mask: bool=False):
        self.tokenizer = tokenizer
        self.max_q_len = data_args.max_q_len
        self.max_p_len = data_args.max_p_len
        self.neg_num = data_args.neg_num
        self.directly_filter_too_long_session = data_args.directly_filter_too_long_session
        
        self.q_suffix = q_suffix
        self.p_suffix = p_suffix
        self.q_suffix_token_id, self.p_suffix_token_id = None, None 
        
        # suffix_token_id can be a list
        if self.q_suffix is not None:
            self.q_suffix_token_id = tokenizer.encode(self.q_suffix, add_special_tokens=False)
        if self.p_suffix is not None:
            self.p_suffix_token_id = tokenizer.encode(self.p_suffix, add_special_tokens=False)

        self.query_field_name = data_args.query_field_name
        self.is_eval = is_eval
        self.mix_inst = mix_inst
        self.use_query_mask = use_query_mask
        
    def __call__(self, batch):    
        sample_ids = []
        query_inputs = []
        teacher_inputs = []
        pos_psg_inputs = []
        neg_psg_inputs = []
        
        inst_inputs = []
        
        for i in range(len(batch)):
            sample = batch[i]
            if self.query_field_name == 'query':
                query = sample.query
            elif self.query_field_name == 'rewrite':
                query = sample.rewrite
            elif self.query_field_name == 'session':
                query = sample.session
            else:
                raise KeyError("query_field_name {} is not supported".format(self.query_field_name))
            
            # The query can be too long for in conversation data, so we filter these too long samples
            if self.directly_filter_too_long_session and len(self.tokenizer.encode(query)) > self.max_q_len:
                if i < len(batch) - 1 or len(query_inputs) > 0:
                    continue
    
            sample_ids.append(sample.sample_idx)
            query_inputs.append(query)
            
            if not self.is_eval:
                if sample.pos_psg:
                    pos_psg_inputs.append(sample.pos_psg) # only use the first positive passage
                if sample.neg_psgs:
                    if len(sample.neg_psgs) < self.neg_num:
                        selected_neg_psgs = sample.neg_psgs
                    else:
                        selected_neg_psgs = random.sample(sample.neg_psgs, self.neg_num)
                    neg_psg_inputs += selected_neg_psgs
                if sample.rewrite:
                    teacher_inputs.append(sample.rewrite)
            
            if sample.pos_psg and self.mix_inst:
                inst_inputs.append("{}\n{}".format(query, sample.pos_psg))
            
            
        # query
        query_input_encodings = self.tokenizer(query_inputs, padding="longest", max_length=self.max_q_len, truncation=True, return_tensors='pt')  
        query_input_encodings = self.ensure_suffix_in_inputs(query_input_encodings, is_query=True)

        # psg
        pos_psg_input_encodings, neg_psg_input_encodings = None, None
        if not self.is_eval:
            if len(pos_psg_inputs) > 0:
                pos_psg_input_encodings = self.tokenizer(pos_psg_inputs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt')      
                pos_psg_input_encodings = self.ensure_suffix_in_inputs(pos_psg_input_encodings, is_query=False)
            if len(neg_psg_inputs) > 0:
                neg_psg_input_encodings = self.tokenizer(neg_psg_inputs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt') 
                neg_psg_input_encodings = self.ensure_suffix_in_inputs(neg_psg_input_encodings, is_query=False)
        
        teacher_input_encodings = None
        if not self.is_eval:
            if len(teacher_inputs) > 0:
                teacher_input_encodings = self.tokenizer(teacher_inputs, padding="longest", max_length=self.max_q_len, truncation=True, return_tensors='pt')      
                teacher_input_encodings = self.ensure_suffix_in_inputs(teacher_input_encodings, is_query=True)
        
        # instruction tuning
        inst_input_encodings = None
        if len(inst_inputs) > 0:
            inst_input_encodings = self._sft_data_org(inst_inputs, max_seq_len=self.max_q_len+self.max_p_len, use_query_mask=self.use_query_mask)
            inst_input_encodings = self.ensure_suffix_in_inputs(inst_input_encodings, is_query=False)
        
        return {'sample_ids': sample_ids,
                'query_input_encodings': query_input_encodings,
                'pos_psg_input_encodings': pos_psg_input_encodings,
                'neg_psg_input_encodings': neg_psg_input_encodings,
                'teacher_input_encodings': teacher_input_encodings,
                'inst_input_encodings': inst_input_encodings}

    def build_query_masked_attention_mask(self, input_ids,  response_template_ids):
        seq_len = len(input_ids[0])
        qm_attn_mask_list = []
        
        for i in range(len(input_ids)):
            response_token_ids_start_idx = None
            for idx in np.where(input_ids[i] == response_template_ids[0])[0]:
                if (
                    response_template_ids
                    == input_ids[i][idx: idx + len(response_template_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx
                    break   # only use the first one (i.e., the query's special token)
        
            qm_attn_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
            if response_token_ids_start_idx:
                # build customized 4D attention mask matrix
                # (bs, seq_len, seq_len)
                real_response_token_start_idx = response_token_ids_start_idx + len(response_template_ids) + 1    
                # query mask
                qm_attn_mask[real_response_token_start_idx:, :response_token_ids_start_idx] = False        
            qm_attn_mask_list.append(qm_attn_mask.unsqueeze(0))
        
        all_qm_attn_mask = torch.cat(qm_attn_mask_list, dim=0)
        return all_qm_attn_mask
            
        
    def _sft_data_org(self, examples, max_seq_len, use_query_mask):
            n_sample = len(examples)
            inputs = self.tokenizer(examples, padding="longest", max_length=max_seq_len, truncation=True, return_tensors='pt')  
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            labels = input_ids.detach().clone()
            ignore_idx = -100
            labels[attention_mask == 0] = ignore_idx
            
            # mask the prompt part
            response_template_ids = self.q_suffix_token_id
            for i in range(n_sample):
                response_token_ids_start_idx = None
                for idx in np.where(labels[i] == response_template_ids[0])[0]:
                    if (
                        response_template_ids
                        == labels[i][idx: idx + len(response_template_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx
                        break   # only use the first one (i.e., the query's special token)
                if response_token_ids_start_idx is None:
                    logger.warning(
                        f"Could not find response key `{response_template_ids}` in the "
                        f'following instance: {i} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    labels[i, :] = ignore_idx
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(response_template_ids)
                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    labels[i, :response_token_ids_end_idx] = ignore_idx

            # query mask
            if use_query_mask:
                qm_attn_mask = self.build_query_masked_attention_mask(input_ids, response_template_ids)
            else:
                qm_attn_mask = None
            
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "qm_attn_mask": qm_attn_mask,
                    "labels": labels}
         
    def ensure_suffix_in_inputs(self, inputs, is_query: bool):
        suffix_token_id = self.q_suffix_token_id if is_query else self.p_suffix_token_id
        if suffix_token_id is not None:
            n = len(suffix_token_id)
            real_lengths = inputs['attention_mask'].sum(dim=1)
            for i in range(1, n+1):
                # inputs['attention_mask'][torch.arange(inputs['attention_mask'].size(0)), real_lengths-i] = 1
                inputs['input_ids'][torch.arange(inputs['input_ids'].size(0)), real_lengths-i] = suffix_token_id[n-i]
            
        return inputs
    
    
    
class IndexingCollator(MatchingCollator):
    def __call__(self, batch: list):    
        sample_ids, psgs = zip(*batch)
        psgs = list(psgs)
        inputs = self.tokenizer(psgs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt')
        inputs = self.ensure_suffix_in_inputs(inputs, is_query=False)
        inputs['sample_ids'] = sample_ids
        
        return inputs
    

# Iteratively loading a large passage collection for indexing
def json_psg_generator(collection_path: str, text_formatter: TextFormatter):
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc='loading colletion {}...'.format(collection_path)):
            obj = json.loads(line)
            psg_id = obj['id']
            psg = obj['text']
            if 'title' in obj:
                title = obj['title']
            else:
                title = ""
            if len(title) > 0:
                psg = title + ". " + psg
            psg = text_formatter(psg, is_only_psg=True).pos_psg
            yield [psg_id, psg]