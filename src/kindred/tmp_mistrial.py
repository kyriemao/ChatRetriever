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
import torch.nn.functional as F
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    x = tokenizer.decode(encodeds[0].tolist())
    embed()
    input()
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])