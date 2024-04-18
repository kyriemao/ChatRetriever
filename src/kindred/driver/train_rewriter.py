import torch.distributed as dist

from transformers import HfArgumentParser, TrainingArguments

from kindred.arguments import ModelArguments, DataArguments, MatchingTrainingArguments
from kindred.models.model_utils import load_model
from kindred.data.data_utils import load_dataset
from kindred.data.rewriter_dataset import RewriterDataset, sft_data_collator
from trl import SFTTrainer
from kindred.utils import write_running_args
from datasets import Dataset
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)
    
    # 2. load data
    matching_train_dataset = load_dataset(tokenizer, model_args.model_type, data_args)
    train_dataset = RewriterDataset(matching_train_dataset)
    # hf_dataset = Dataset.from_dict({"text": train_dataset.data})

    # 3. train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=lambda x: sft_data_collator(x, tokenizer, data_args.max_q_len, train_dataset.response_template),
        dataset_num_proc=64,
        packing=True,
    )

    trainer.train()
    
    # 4. save
    trainer.save_model(training_args.output_dir)
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, training_args])
    
    
if __name__ == '__main__':
    main()