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