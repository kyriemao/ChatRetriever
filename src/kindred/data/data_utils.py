from transformers import AutoTokenizer

from kindred.arguments import DataArguments
from .dataset import MatchingDataset
from .instruct_dataset import InstructDataset
from .chat_dataset import ChatDataset
from .convsearch_dataset import ConvSearchDataset
from .msmarco_dataset import MsmarcoDataset
from .tevatron_msmarco_dataset import TevatronMsmarcoDataset
from .llm4cs_dataset import LLM4CSDataset
from IPython import embed

def load_dataset(tokenizer: AutoTokenizer, 
                 model_type: str, 
                 data_args: DataArguments):
    dataset = MatchingDataset(tokenizer, model_type, data_args, use_data_percent=1.0)
    use_data_percent_list = data_args.use_data_percent
    used_num = 0
    if data_args.chat_data_path_list:
        dataset.merge(ChatDataset(tokenizer, 
                                  model_type, 
                                  data_args, 
                                  use_data_percent_list[used_num:used_num+len(data_args.chat_data_path_list)]))
        used_num += len(data_args.chat_data_path_list)
        
    if data_args.instruct_data_path_list:
        dataset.merge(InstructDataset(tokenizer, 
                                      model_type, 
                                      data_args,
                                      use_data_percent_list[used_num:used_num+len(data_args.instruct_data_path_list)]))
        used_num += len(data_args.instruct_data_path_list)

    if data_args.tevatron_msmarco_data_path_list:
        dataset.merge(TevatronMsmarcoDataset(tokenizer, 
                                             model_type, 
                                             data_args,
                                             use_data_percent_list[used_num:used_num+len(data_args.tevatron_msmarco_data_path_list)]))
        used_num += len(data_args.tevatron_msmarco_data_path_list)

    if data_args.convsearch_data_path_list:
        dataset.merge(ConvSearchDataset(tokenizer, 
                                        model_type, 
                                        data_args,
                                        use_data_percent_list[used_num:used_num+len(data_args.convsearch_data_path_list)]))
        used_num += len(data_args.convsearch_data_path_list)
        
    if data_args.llm4cs_data_path_list:
        dataset.merge(LLM4CSDataset(tokenizer, 
                                        model_type, 
                                        data_args,
                                        use_data_percent_list[used_num:used_num+len(data_args.llm4cs_data_path_list)]))
        used_num += len(data_args.llm4cs_data_path_list)
    
                

    return dataset
