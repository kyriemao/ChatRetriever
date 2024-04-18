import torch
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, LlamaForCausalLM, AutoModelForCausalLM, LlamaModel

from .gtr import GTR
from .ance import ANCE
from .splade import Splade

from IPython import embed


MODEL_TYPE_CLS = {
    "gtr": GTR,
    "ance": ANCE,
    "bge": AutoModel,
    "splade": Splade,
    "qwen_lm": AutoModelForMaskedLM,
    "qwen_lm_inference": AutoModelForMaskedLM,
    "qwen": AutoModelForCausalLM,
    "qwen_chat": AutoModelForCausalLM,
    "qwen_chat_cot": AutoModelForCausalLM,
    "qwen_chat_lora": AutoModelForCausalLM,
    "qwen_chat_lora_eval": AutoModelForCausalLM,
    "qwen_chat_cot_lora": AutoModelForCausalLM,
    "qwen_chat_cot_lora_eval": AutoModelForCausalLM,
    "qwen15": AutoModelForCausalLM,
    "qwen15_chat": AutoModelForCausalLM,
    "qwen15_chat_cot": AutoModelForCausalLM,
    "qwen15_chat_lora": AutoModelForCausalLM,
    "qwen15_chat_lora_eval": AutoModelForCausalLM,
    "qwen15_chat_cot_lora": AutoModelForCausalLM,
    "qwen15_chat_cot_lora_eval": AutoModelForCausalLM,
    "e5_mistrial": AutoModel, 
    "e5_mistrial-train": AutoModel,
    "mistrial_cot-train": AutoModelForCausalLM,
    "mistrial_chat_cot-train": AutoModelForCausalLM,
    "mistrial_cot": AutoModelForCausalLM,
    "mistrial_chat_cot": AutoModelForCausalLM,
    "repllama": LlamaModel,
    "repllama-train": LlamaModel,
    "repllama_v2": LlamaModel,
    "repllama_v2-train": LlamaModel,
    "repllama_v2-continue_train": LlamaModel,
    "repllama_chat": LlamaModel,
    "repllama_chat-train": LlamaModel,
    "repllama_chat_cot": LlamaModel,
    "repllama_chat_cot-train": LlamaModel,
    "llm_embedder": AutoModel,
    "llama": LlamaForCausalLM,
    "bert": AutoModel,
    "splade": Splade,
    "openai": None,
}


def load_model(model_args, is_freezed_model=False, for_eval=False):
    model_name_or_path = model_args.model_name_or_path if not is_freezed_model else model_args.freezed_model_name_or_path
    model_type = model_args.model_type if not is_freezed_model else model_args.freezed_model_type
    model_dtype = model_args.model_dtype if not is_freezed_model else model_args.freezed_model_dtype
    if model_args.freezed_model_name_or_path:
        assert model_args.model_dtype == model_args.freezed_model_dtype
    
    if model_dtype == 'fp16':
        model_dtype = torch.float16
    elif model_dtype == 'fp32':
        model_dtype = torch.float32
    elif model_dtype == 'bf16':
        model_dtype = torch.bfloat16
    
    if model_name_or_path is None:
        return None, None
        
    assert model_type in MODEL_TYPE_CLS
    
    model_cls = MODEL_TYPE_CLS[model_type]
    if model_type == "splade":
        model = model_cls(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # elif model_type == 'instructor':
    #     model = INSTRUCTOR(model_name_or_path)
    elif model_type in ['repllama', 'repllama_v2', 'repllama_v2-continue_train', 'repllama_chat','repllama_chat_cot']:
        peft_model_name = model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = model_cls.from_pretrained(config.base_model_name_or_path, torch_dtype=model_dtype)
        if "train" in model_type:
            base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, peft_model_name, config=config)
        if "train" in model_type:
            # enable lora weights requires_grad
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
        if "train" not in model_type:
            model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if "cot" in model_type:
            tokenizer.add_special_tokens({"additional_special_tokens":["<|emb_0|>", 
                                                                       "<|emb_1|>", 
                                                                       "<|emb_2|>"]}) 
            model.resize_token_embeddings(len(tokenizer))
    elif model_type in ['e5_mistrial']:
        model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        
    elif model_type in ['e5_mistrial-train', 'mistrial_cot-train', 'mistrial_chat_cot-train']:
        base_model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
        base_model.enable_input_require_grads() # add this when using gradient checkpointings
        peft_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False
        )
        
        model = get_peft_model(base_model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        
    elif model_type in ['mistrial_cot', 'mistrial_chat_cot']:
        peft_model_name = model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = model_cls.from_pretrained(config.base_model_name_or_path, torch_dtype=model_dtype)
        model = PeftModel.from_pretrained(base_model, peft_model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
      
    elif model_type in ['repllama-train', 'repllama_v2-train', 'repllama_chat-train', 'repllama_chat_cot-train']:
        base_model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype, trust_remote_code=True, use_cache=False)
        base_model.enable_input_require_grads() # add this when using gradient checkpointings
         
        peft_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False
        )
        model = get_peft_model(base_model, peft_config)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        if "cot" in model_type:
            tokenizer.add_special_tokens({"additional_special_tokens":["<|emb_0|>", 
                                                                       "<|emb_1|>", 
                                                                       "<|emb_2|>"]}) 
            model.resize_token_embeddings(len(tokenizer))
    
        return model, tokenizer
    
    elif model_type in ['qwen', 'qwen_chat', 'qwen_chat_lora', 'qwen_chat_cot_lora', 'qwen_lm', 'qwen_chat_cot']:
        model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype, trust_remote_code=True, use_flash_attn=False)
        if 'lora' in model_type:
            model.enable_input_require_grads() # add this when using gradient checkpointings
            peft_config = LoraConfig(
                base_model_name_or_path=model_args.model_name_or_path,
                task_type=TaskType.FEATURE_EXTRACTION,
                r=32,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj", "w1", "w2"],
                inference_mode=False
            )
            model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|extra_3|>'})
    
    elif model_type in ['qwen15', 'qwen15_chat', 'qwen15_chat_lora', 'qwen15_chat_cot_lora', 'qwen15_chat_cot']:
        model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
        if 'lora' in model_type:
            model.enable_input_require_grads() # add this when using gradient checkpointings
            peft_config = LoraConfig(
                base_model_name_or_path=model_args.model_name_or_path,
                task_type=TaskType.FEATURE_EXTRACTION,
                r=16,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj",
                                "k_proj",
                                "v_proj",
                                "o_proj",
                                "up_proj",
                                "gate_proj",
                                "down_proj"],
                inference_mode=False
            )
            model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        
    elif model_type in ['qwen_lm_inference']:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            return_dict=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|extra_3|>'})
        tokenizer.padding_side = "left"
        
    elif model_type in ['qwen_chat_lora_eval', 'qwen_chat_cot_lora_eval']:
        peft_model_name = model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = model_cls.from_pretrained(config.base_model_name_or_path, torch_dtype=model_dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, peft_model_name, config=config)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|extra_3|>'})
    elif model_type in ['qwen15_chat_lora_eval', 'qwen15_chat_cot_lora_eval']:
        peft_model_name = model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = model_cls.from_pretrained(config.base_model_name_or_path, torch_dtype=model_dtype)
        model = PeftModel.from_pretrained(base_model, peft_model_name, config=config)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        model = model_cls.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if for_eval:
        model.eval()
        
    return model, tokenizer


def unified_model_forward(model, model_type: str, inputs: dict, normalize_emb: bool):
    inputs.pop("sample_ids", None)
    inputs.pop("input_texts", None)
    
    if model_type in ['bge', 'llm_embedder']:
        output = model(**inputs)
        embs = output.last_hidden_state[:, 0]
    elif model_type in set(['qwen', 'qwen_chat', 'qwen_chat_cot', 'qwen_chat_lora', 'qwen_chat_lora_eval', 'qwen_chat_cot_lora', 'qwen_chat_cot_lora_eval', 'qwen15_chat_cot_lora_eval', 'repllama', 'llama', 'repllama-train', 'repllama_v2', 'repllama_v2-train', 'repllama_v2-continue_train', 'repllama_chat', 'repllama_chat-train', 'repllama_chat_cot', 'repllama_chat_cot-train', 'mistrial_cot-train', 'mistrial_chat_cot-train', 'mistrial_cot', 'mistrial_chat_cot', 'qwen15', 'qwen15_chat', 'qwen15_chat_lora', 'qwen15_chat_cot_lora', 'qwen15_chat_cot']):        
        inputs['output_hidden_states'] = True
        output = model(**inputs)
        hidden_states = output.hidden_states[-1]
        last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
        embs = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
    elif model_type in ['e5_mistrial', 'e5_mistrial-train']:
        outputs = model(**inputs)
        embs = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
    elif model_type in ['ance', 'gtr', 'splade']:
        embs = model(**inputs)
    elif model_type in ['bert']:
        output = model(**inputs)
        embs = output.pooler_output
    else:
        raise NotImplementedError("Model type {} is not supported now.".format(model_type))    
    
    if normalize_emb:
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
    return embs



def set_all_requires_grad(model, state: bool):
    if model:
        for param in model.parameters():
            param.requires_grad = state
            

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]