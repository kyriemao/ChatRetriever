from IPython import embed

import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from datasets import IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import HfArgumentParser

from kindred.arguments import ModelArguments, DataArguments, EvalArguments
from kindred.models.model_utils import load_model, unified_model_forward
from kindred.data.dataset import IndexingCollator, json_psg_generator
from kindred.data.text_format import TextFormatter
from kindred.utils import write_running_args, mkdirs
from kindred.retrieval_toolkits.splade_tool import IndexDictOfArray

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        if data_args.data_output_dir:
            mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
        write_running_args(data_args.data_output_dir, [model_args, data_args, eval_args])
    
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=True)
    model.to("cuda")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 2. load data
    text_formatter = TextFormatter(tokenizer, model_args.model_type, data_args)
    indexing_dataset = IterableDataset.from_generator(json_psg_generator, 
                                                      gen_kwargs={"collection_path": data_args.collection_path,
                                                                  "text_formatter": text_formatter})
    
    indexing_collator = IndexingCollator(tokenizer,
                                         data_args,
                                         q_suffix=text_formatter.q_suffix,
                                         p_suffix=text_formatter.p_suffix,
                                         is_eval=True)
    
    sub_indexing_dataset = split_dataset_by_node(indexing_dataset, world_size=dist.get_world_size(), rank=local_rank)
    indexing_dataloader = DataLoader(sub_indexing_dataset, 
                                     batch_size=eval_args.per_device_eval_batch_size, 
                                     collate_fn=indexing_collator,
                                     num_workers=4)
    
    # 3. add passage embeddings to the index
    splade_index = IndexDictOfArray()
    index_dir = os.path.join(data_args.data_output_dir, "rank_{}".format(local_rank))
    os.makedirs(index_dir)

    for batch in tqdm(indexing_dataloader, desc='Splade indexing...', position=0, leave=True):
        inputs = {k: v.to("cuda") for k, v in batch.items() if k not in {"sample_ids"}}
        embs = unified_model_forward(model, model_args.model_type, inputs, model_args.normalize_emb)
        splade_index.add_psg_lexical_embs(embs.detach().cpu().numpy(), batch['sample_ids'])

    splade_index.save(index_dir)
    dist.barrier()
    
    # 4. reload and merge all splade indexes to one
    if dist.get_rank() == 0:
        for rank in range(dist.get_world_size()):
            if rank == 0:
                continue
            another_index = IndexDictOfArray()
            another_index_dir = os.path.join(data_args.data_output_dir, "rank_{}".format(rank))
            another_index.load(another_index_dir)
            splade_index.merge(another_index)
            
            logger.info("remove {}...".format(another_index_dir))
            os.system("rm -r {}".format(another_index_dir))

        logger.info("remove {}...".format(index_dir))
        os.system("rm -r {}".format(index_dir))
        logger.info("all splade indexes are merged into {}...".format(data_args.data_output_dir))
        splade_index.save(data_args.data_output_dir)    


if __name__ == "__main__":
    main()