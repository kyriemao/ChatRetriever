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

