import os
import json
import pytrec_eval
import numpy as np
from pprint import pprint
from IPython import embed

from transformers import HfArgumentParser
from kindred.arguments import DataArguments, EvalArguments
from kindred.utils import mkdirs

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trec_eval(run_trec_path: str, qrel_trec_path: str, rel_threshold: str, need_turn_level_result: bool = False):
    # process run trec file
    with open(run_trec_path, 'r' )as f:
        run_data = f.readlines()
    runs = {}
    for line in run_data:
        line = line.split(" ")
        sample_id = line[0]
        doc_id = line[2]
        score = float(line[4])
        if sample_id not in runs:
            runs[sample_id] = {}
        runs[sample_id][doc_id] = score

    # process qrel trec file
    with open(qrel_trec_path, 'r') as f:
        qrel_data = f.readlines()
    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        record = line.strip().split("\t")
        if len(record) == 1:
            record = line.strip().split(" ")
        query = record[0]
        doc_id = record[2]
        rel = int(record[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][doc_id] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][doc_id] = rel
 

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100", "recall.1000"})
    res_others = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res_others.values()]
    mrr_list = [v['recip_rank'] for v in res_others.values()]
    recall_5_list = [v['recall_5'] for v in res_others.values()]
    recall_10_list = [v['recall_10'] for v in res_others.values()]
    recall_20_list = [v['recall_20'] for v in res_others.values()]
    recall_100_list = [v['recall_100'] for v in res_others.values()]
    recall_1000_list = [v['recall_1000'] for v in res_others.values()]

    # get mrr@k
    mrr_10_list = get_mrr_k(qrels, runs, k=10)
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res_ndcg_3 = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res_ndcg_3.values()]
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.10"})
    res_ndcg_10 = evaluator.evaluate(runs)
    ndcg_10_list = [v['ndcg_cut_10'] for v in res_ndcg_10.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "MRR@10": np.average(mrr_10_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
            "Recall@1000": np.average(recall_1000_list),
            "NDCG@3": np.average(ndcg_3_list), 
            "NDCG@10": np.average(ndcg_10_list),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    pprint(res)
    
    # write results to files
    output_path = os.path.dirname(run_trec_path)
    with open(os.path.join(output_path, "all_metric.res"), "w") as f:
        f.write(json.dumps(res, indent=4))
    if need_turn_level_result:
        with open(os.path.join(output_path, "turn_level_others.res"), "w") as f:
            f.write(json.dumps(res_others, indent=4))
        with open(os.path.join(output_path, "turn_level_ndcg.res"), "w") as f:
            f.write(json.dumps(res_ndcg_3, indent=4))

    return res


def get_mrr_k(qrels, runs, k):
    new_runs = {}
    # only keep top10 psg per sample
    for sample_idx in runs:
        sorted_dict = dict(sorted(runs[sample_idx].items(), key=lambda x: x[1], reverse=True)[:k])
        new_runs[sample_idx] = sorted_dict
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    res = evaluator.evaluate(new_runs)
    mrr_list = [v['recip_rank'] for v in res.values()]
    
    return mrr_list
    
    

def doc_level_agg(run_trec_path):
    res_file = os.path.join(run_trec_path)
    with open(run_trec_path, 'r' ) as f:
        run_data = f.readlines()
    
    model_name_or_path = "no_name"
    agg_run = {}
    for line in run_data:
        line = line.strip().split(" ")
        model_name_or_path = line[-1]
        if len(line) == 1:
            line = line.strip().split('\t')
        sample_id = line[0]
        if sample_id not in agg_run:
            agg_run[sample_id] = {}
        doc_id = "_".join(line[2].split('_')[:2])
        score = float(line[4])

        if doc_id not in agg_run[sample_id]:
            agg_run[sample_id][doc_id] = 0
        agg_run[sample_id][doc_id] = max(agg_run[sample_id][doc_id], score)
    
    agg_run = {k: sorted(v.items(), key=lambda item: item[1], reverse=True) for k, v in agg_run.items()}
    with open(os.path.join(run_trec_path + ".agg"), "w") as f:
        for sample_id in agg_run:
            doc_scores = agg_run[sample_id]
            rank = 1
            for doc_id, real_score in doc_scores:
                rank_score = 9999 - rank
                f.write("{} Q0 {} {} {} {}\n".format(sample_id, doc_id, rank, rank_score, real_score, model_name_or_path))
                rank += 1                


def evaluate(data_args, eval_args):
    run_trec_path = os.path.join(eval_args.run_trec_dir, "run.0.trec")
    if eval_args.need_doc_level_agg:
        doc_level_agg(run_trec_path)
        run_trec_path = run_trec_path + ".agg"
    
    trec_eval(run_trec_path, data_args.qrel_path, eval_args.rel_threshold, eval_args.need_turn_level_result)
    
def main():
    parser = HfArgumentParser((DataArguments, EvalArguments))
    data_args, eval_args = parser.parse_args_into_dataclasses()
    if data_args.data_output_dir:
        mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
    
    evaluate(data_args, eval_args)


if __name__ == "__main__":
    main()
