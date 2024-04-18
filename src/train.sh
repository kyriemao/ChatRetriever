export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="train-lora-qwenchat7b"

MODEL_NAME_OR_PATH="Qwen/Qwen-7B-Chat"

torchrun --nproc_per_node=8 \
--master_port 28553 \
kindred/driver/train.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="qwen_chat_cot_lora" \
--chat_data_path_list="../cis_datasets/ultrachat/train.qaw.jsonl" \
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

