import torch.distributed as dist

from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer

from kindred.arguments import ModelArguments, DataArguments, MatchingTrainingArguments, ConvSearchArguments
from kindred.data.data_utils import load_dataset
from kindred.utils import pstore
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MatchingTrainingArguments, ConvSearchArguments, TrainingArguments))
    model_args, data_args, match_training_args, conv_args, training_args = parser.parse_args_into_dataclasses()

    # 1. load model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # 2. load data
    my_dataset = load_dataset(tokenizer, data_args, conv_args)
    
    # 3. statistic data
    from tqdm import tqdm
    import numpy as np
    
    query_lens = []
    psg_lens = []
    for sample in tqdm(my_dataset.samples):
        q_len = len(tokenizer.encode(sample.query))
        query_lens.append(q_len)
        psg_lens.extend([len(tokenizer.encode(psg)) for psg in sample.pos_psgs])
        if sample.neg_psgs:
            psg_lens.extend([ len(tokenizer.encode(psg)) for psg in sample.neg_psgs])

    
    query_lens = np.array(query_lens)
    psg_lens = np.array(psg_lens)

    pstore(query_lens, "query_lens.pkl")
    pstore(psg_lens, "psg_lens.pkl")
    embed()
    input()
    
    
if __name__ == '__main__':
    main()