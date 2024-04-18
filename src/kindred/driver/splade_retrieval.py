from IPython import embed
import numpy as np
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from kindred.arguments import ModelArguments, DataArguments, EvalArguments
from kindred.models.model_utils import load_model, unified_model_forward
from kindred.data.data_utils import load_dataset
from kindred.data.dataset import MatchingCollator
from kindred.utils import get_qrel_sample_ids, write_running_args, mkdirs
from kindred.retrieval_toolkits.splade_tool import SpladeRetrievalTool
from kindred.driver.evaluate import evaluate

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def get_query_embs(model, model_args, data_args, eval_dataloader):
    model.eval()
    model.to("cuda")

    qrel_sample_ids = get_qrel_sample_ids(data_args.qrel_path) # get all sample_ids that needed to be evaluated
    
    query_embs = []
    sample_ids = []
    for batch in tqdm(eval_dataloader, desc="Encoding queries..."):
        inputs = batch['query_input_encodings']
        inputs = {k: v.to("cuda") if k not in {"sample_ids"} else v for k, v in inputs.items()}
        embs = unified_model_forward(model, model_args.model_type, inputs, model_args.normalize_emb)
        embs = embs.detach().cpu().numpy()
        for i, sample_idx in enumerate(batch['sample_ids']):
            if sample_idx in qrel_sample_ids:
                sample_ids.append(sample_idx)
                query_embs.append(embs[i].reshape(1, -1))

    query_embs = np.concatenate(query_embs, axis=0)
    query_embs = query_embs.astype(np.float32) if query_embs.dtype != np.float32 else query_embs

    logger.info("#Total queries for evaluation: {}".format(len(query_embs)))
    
    model.to("cpu")
    torch.cuda.empty_cache()
    
    return sample_ids, query_embs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    if data_args.data_output_dir:
            mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
    write_running_args(data_args.data_output_dir, [model_args, data_args, eval_args])

    # 1. get query embeddings
    model, tokenizer = load_model(model_args, for_eval=True)
    
    # 2. load data
    eval_dataset = load_dataset(tokenizer, model_args.model_type, data_args)
    eval_collator = MatchingCollator(tokenizer, 
                                     data_args, 
                                     eval_dataset.text_formatter.q_suffix,
                                     eval_dataset.text_formatter.p_suffix, 
                                     is_eval=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_args.per_device_eval_batch_size, shuffle=False, collate_fn=eval_collator)
    
    # 3. get query embeddings
    sample_ids, query_embs = get_query_embs(model, model_args, data_args, eval_dataloader)

    # 4. faiss retrieval
    splade_retrieval_tool = SpladeRetrievalTool(index_dir=data_args.index_dir, top_n=eval_args.top_n)
    scores_mat, psg_ids_mat = splade_retrieval_tool.retrieve(query_embs)
    
    # 5. output retrieval results
    logger.info('begin to write the retrieval output...')

    output_path = os.path.join(data_args.data_output_dir, 'run.json')
    output_trec_path = os.path.join(data_args.data_output_dir, 'run.trec')
    with open(output_path, "w") as fw, open(output_trec_path, "w") as fw_trec:
        for i in range(len(sample_ids)):
            sample_idx = sample_ids[i]
            rank = 0
            for psg_idx, score in zip(psg_ids_mat[i], scores_mat[i]):
                fw.write(json.dumps({"sample_id": str(sample_idx), "psg": "", "psg_idx": psg_idx, "rank": rank, "retrieval_score": score}) + "\n")
                fw_trec.write("{} Q0 {} {} {}".format(sample_idx, psg_idx, rank, eval_args.top_n-rank, model_args.model_name_or_path) + "\n")
                rank += 1
                
    logger.info('finish writing the retrieval output...')

    # 6. evaluation 
    eval_args.run_trec_dir = data_args.data_output_dir
    evaluate(data_args, eval_args)



if __name__ == '__main__':
    main()
