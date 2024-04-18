from IPython import embed

import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from datasets import IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import HfArgumentParser, AutoTokenizer, AutoModel, AutoModelForCausalLM

from kindred.arguments import ModelArguments, DataArguments, EvalArguments
from kindred.models.model_utils import load_model, unified_model_forward
from kindred.models.splade import Splade
from kindred.data.dataset import IndexingCollator, json_psg_generator
from kindred.utils import write_running_args, pstore, pload
from kindred.retrieval_toolkits.splade_tool import IndexDictOfArray
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    embed()
    input()
#    base_dir = "/share/peitian/Data/Datasets/llm-embedder/qa/msmarco/"
#    output_dir = "/share/kelong/cis_collections/msmarco"
#    with open(os.path.join(base_dir, "corpus.json")) as fr, open(os.path.join(output_dir, "msmarco_collection.psg.jsonl"), "w") as fw:
#        for line in tqdm(fr):
#             line = json.loads(line)
#             doc_id = line.pop("docid")
#             line['id'] = doc_id
#             fw.write(json.dumps(line))
#             fw.write('\n')

    # model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # embed()
    # input()
 
    # test qwen
    
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from transformers.generation import GenerationConfig
#     MODEL_NAME_OR_PATH="/share/LMs/hub/models--Qwen--Qwen-1_8B-Chat/snapshots/63f5577cdabcf601944ed2adfd70663c6615d69e"
    
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME_OR_PATH,
#         device_map="auto",
#         trust_remote_code=True
#     ).eval()
    
#     text = """<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# How can mindful leadership benefit an organization in the long run? Does it lead to better financial performance?<|im_end|>
# <|im_start|>assistant
#     """
#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_NAME_OR_PATH,
#         trust_remote_code=True
#     )
    
#     input_ids = tokenizer.encode(text)
#     embed()
#     input()
#     output = model.generate(input_ids)
#     response = tokenizer.decode(output)
#     print(response)
#     embed()
#     input()

    # def _qwen_chat_text_format(query, history):
        
    #     def _wrap_text(system, query, history):
    #         im_start, im_end = "<|im_start|>", "<|im_end|>"
            
    #         def _format_raw(role, content):
    #             return f"{role}\n{content}"
    #         system_text = _format_raw("system", content=system)
    #         raw_text = ""
            
    #         for turn_query, turn_response in reversed(history):
    #             query_text = _format_raw("user", turn_query)
    #             response_text = _format_raw(
    #                 "assistant", turn_response
    #             )
    #             prev_chat = (
    #                 f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
    #             )
    #             raw_text = prev_chat + raw_text

    #         raw_text = f"{im_start}{system_text}{im_end}" + raw_text
    #         raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
    #         return raw_text

    #     session = _wrap_text(system="You are a helpful assistant.",
    #                                 query=query,
    #                                 history=history)
        
    #     return session

    # query = "what you name"
    # history = [["111", "222"], ["333", "444"]]
    # print(_qwen_chat_text_format(query, history))
