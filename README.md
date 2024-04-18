# ChatRetriever: Adapting Large Language Models for Generalized and Robust Conversational Dense Retrieval

This is the anonymous repository for our 2024 ARR April submission: *ChatRetriever: Adapting Large Language Models for Generalized and Robust Conversational Dense Retrieval*


## Preparation
We fine-tune Qwen-Chat-7B on the `question_about_the_world` subset of UltraChat and MSMARCO to get ChatRetriever. We then evaluate on five conversational search datasets. All datasets are open and you can downloaded them from:

- UltraChat: https://github.com/thunlp/UltraChat
- MSMARCO: https://microsoft.github.io/msmarco/
- QReCC: https://github.com/apple/ml-qrecc
- TopiOCQA: https://github.com/McGill-NLP/topiocqa
- CAsT-19,20,21: https://www.treccast.ai/ 

Put your data into the right path based on the following scripts.

## Train
To train the model, run
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="train-lora-qwenchat7b"

MODEL_NAME_OR_PATH="Qwen/Qwen-7B-Chat"

torchrun --nproc_per_node=8 \
--master_port 28553 \
src/kindred/driver/train.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="qwen_chat_cot_lora" \
--chat_data_path_list="../cis_datasets/ultrachat/train.jsonl" \
--tevatron_msmarco_data_path_list="../cis_datasets/msmarco/train.jsonl" \
--query_field_name="session" \
--filter_no_pos \
--directly_filter_too_long_session \
--normalize_emb \
--temperature=0.01 \
--use_data_percent=0.1 \
--gradient_accumulation_steps=4 \
--max_q_len=1024 \
--max_p_len=384 \
--loss_type="ranking" \
--neg_type="hard" \
--neg_num=4 \
--ranking_loss_weight=1.0 \
--inst_loss_weight=0.2 \
--use_query_mask \
--per_device_train_batch_size=8 \
--warmup_steps=50 \
--learning_rate=1e-4 \
--num_train_epochs=1 \
--logging_steps=1 \
--save_strategy='steps' \
--save_steps=200 \
--save_total_limit=100 \
--log_level="info" \
--report_to="wandb" \
--run_name="1" \
--output_dir="./checkpoints" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="ds_config.1.json" \
```

## Index
Before evaluation, we should get the dense index. Taking CAsT-21 for example, run the following script to index the corpus. Since the corpus contains over ten millions of passages, the indexing process can be long.
```sh
export CUDA_VISIBLE_DEVICES=4,5,6,7

DATA_FOLDER="./"
MODEL_PATH="./checkpoints"

torchrun --nproc_per_node=4 \
--master_port 28145 \
kindred/driver/dense_indexing.py \
--model_name_or_path=$MODEL_PATH \
--model_type="qwen_chat_cot_lora_eval" \
--normalize_emb \
--max_p_len=384 \
--collection_path=$DATA_FOLDER"./cis_collections/cast21/cast21_collection.jsonl" \
--per_device_eval_batch_size=2 \
--num_psg_per_block=1000000 \
--data_output_dir=$DATA_FOLDER"./cis_indexes/cast21/index" \
--force_emptying_dir \
```

## Test
To get the retrieval performance, run:
```sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
DATA_FOLDER="."

MODEL_PATH="./checkpoints"

python kindred/driver/faiss_retrieval.py \
--model_name_or_path=$MODEL_PATH \
--model_type="qwen_chat_cot_lora_eval" \
--embedding_size=4096 \
--convsearch_data_path_list="$DATA_FOLDER./cis_datasets/cast21/test.json" \
--query_field_name="session" \
--qrel_path=$DATA_FOLDER"./cis_datasets/cast21/qrels.txt" \
--rel_threshold=2 \
--max_q_len=1024 \
--need_doc_level_agg \
--per_device_eval_batch_size=4 \
--index_dir=$DATA_FOLDER"./cis_indexes/cast21" \
--data_output_dir="./results/" \
--force_emptying_dir \
```