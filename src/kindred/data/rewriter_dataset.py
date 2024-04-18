import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from IPython import embed
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAIN_REWRITER_TEMPLATE = """
### Human:
Transform the current user query into a context-independent version that encapsulates the user's information needs comprehensively, without relying on the context provided by previous interactions.

- Current User Query:
{query}

- Previous Interactions:
{history}

### Assistant:
Rewritten Query:
{rewrite}
""".strip('\n')

INFERENCE_REWRITER_TEMPLATE = """
### Human:
Transform the current user query into a context-independent version that encapsulates the user's information needs comprehensively, without relying on the context provided by previous interactions.

- Current User Query:
{query}

- Previous Interactions:
{history}

### Assistant:
""".lstrip('\n')

class RewriterDataset(Dataset):
    def __init__(self, matching_dataset, inference=False):
        self.samples = matching_dataset.samples
        self.tokenizer = matching_dataset.tokenizer
        self.data = []
        if inference:
            self._format_inference_text()
        else:
            self._format_train_text()
            self.response_template = "### Assistant:\n"
        
    def _format_train_text(self):
        for sample in tqdm(self.samples):
            turn_idx = 0
            history_text = []
            for i in range(0, len(sample.history), 2):
                turn_idx += 1
                history_text.append("Query of Turn-{}: {}".format(turn_idx, sample.history[i]['text']))
            if len(sample.history) > 0:
                history_text.append("Response of Turn-{}: {}".format(turn_idx, sample.history[-1]['text']))
            history_text = "\n".join(history_text)
            
            text = TRAIN_REWRITER_TEMPLATE.format(query=sample.query, history=history_text, rewrite=sample.rewrite)
            self.data.append(text)
    
    def _format_inference_text(self):
        for sample in tqdm(self.samples):
            if len(sample.history) == 0:
                continue
            turn_idx = 0
            history_text = []
            for i in range(0, len(sample.history), 2):
                turn_idx += 1
                history_text.append("Query of Turn-{}: {}".format(turn_idx, sample.history[i]['text']))
            if len(sample.history) > 0:
                response = " ".join(sample.history[-1]['text'].split(" ")[:300])
                history_text.append("Response of Turn-{}: {}".format(turn_idx, response))
            history_text = "\n".join(history_text)
            
            text = INFERENCE_REWRITER_TEMPLATE.format(query=sample.query, history=history_text)
            self.data.append([sample.sample_idx, text])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
def sft_data_collator(examples, tokenizer, max_seq_len, response_template):
        n_sample = len(examples)
        inputs = tokenizer(examples, padding="longest", max_length=max_seq_len, truncation=True, return_tensors='pt')  
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        labels = input_ids.detach().clone()
        ignore_idx = -100
        labels[labels == tokenizer.pad_token_id] = ignore_idx
        
        # mask the prompt part
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        for i in range(n_sample):
            response_token_ids_start_idx = None
            for idx in np.where(labels[i] == response_template_ids[0])[0]:
                if (
                    response_template_ids
                    == labels[i][idx: idx + len(response_template_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

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
        
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}
        

def inference_collator(examples, tokenizer, max_seq_len):
    sample_ids = [example[0] for example in examples]
    input_texts = [example[1] for example in examples]
    inputs = tokenizer(input_texts, padding="longest", max_length=max_seq_len, truncation=True, return_tensors='pt')  
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    return {"sample_ids": sample_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask}