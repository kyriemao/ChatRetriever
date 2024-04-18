import torch.distributed as dist

from transformers import HfArgumentParser, TrainingArguments
from transformers import Trainer
from kindred.arguments import ModelArguments, DataArguments, MatchingTrainingArguments
from kindred.models.model_utils import load_model, set_all_requires_grad
from kindred.data.data_utils import load_dataset
from kindred.trainers.trainer import MatchingTrainer
from kindred.data.dataset import MatchingCollator
from kindred.utils import write_running_args
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MatchingTrainingArguments, TrainingArguments))
    model_args, data_args, match_training_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)
    freezed_model, _ = load_model(model_args, is_freezed_model=True, for_eval=True)
    set_all_requires_grad(freezed_model, state=False)
    
    # 2. load data
    train_dataset = load_dataset(tokenizer, model_args.model_type, data_args)
    train_collator = MatchingCollator(tokenizer, 
                                      data_args,
                                      train_dataset.text_formatter.q_suffix,
                                      train_dataset.text_formatter.p_suffix, 
                                      is_eval=False,
                                      mix_inst=match_training_args.inst_loss_weight > 0,
                                      use_query_mask=match_training_args.use_query_mask)
    
    # 3. train
    trainer = MatchingTrainer(model_args=model_args,
                              match_training_args=match_training_args,
                              freezed_model=freezed_model,
                              model=model,
                              tokenizer=tokenizer,
                              train_dataset=train_dataset,
                              data_collator=train_collator,
                              args=training_args)

    trainer.train()

    # 4. save model and training args
    if model_args.model_type == 'splade':
        if dist.get_rank() == 0:
            model.save_model(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
        
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, match_training_args, training_args])
    
    
if __name__ == '__main__':
    main()