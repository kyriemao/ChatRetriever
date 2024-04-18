from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "Type of the pretrained model or model identifier from huggingface.co/models"}
    )
    model_dtype: Literal['bf16', 'fp16', 'fp32', 'auto'] = field(default="auto", metadata={"help": "the data type of the model"})
    freezed_model_name_or_path: str = field(default=None,
        metadata={"help": "Path to freezed model or model identifier from huggingface.co/models"}
    )
    freezed_model_type: str = field(default=None,
        metadata={"help": "Type of the freezed model or model identifier from huggingface.co/models"}
    )
    freezed_model_dtype: Literal['bf16', 'fp16', 'fp32', 'auto'] = field(default="auto", metadata={"help": "the data type of the freezed model"})
    
    normalize_emb: bool = field(default=False)


@dataclass
class DataArguments:
    data_path_list: List[str] = field(default=None, metadata={"help": "the list of normal (matching) dataset names"})
    tevatron_msmarco_data_path_list: List[str] = field(default=None, metadata={"help": "the list of msmarco dataset names"})
    msmarco_data_path_list: List[str] = field(default=None, metadata={"help": "the list of msmarco dataset names"})
    instruct_data_path_list: List[str] = field(default=None, metadata={"help": "the list of instruct dataset names"})
    chat_data_path_list: List[str] = field(default=None, metadata={"help": "the list of chat-version dataset names"})
    convsearch_data_path_list: List[str] = field(default=None, metadata={"help": "the list of convsearch dataset names"})
    llm4cs_data_path_list: List[str] = field(default=None, metadata={"help": "the list of llm4cs dataset names"})
    aug_convsearch_data_path_list: List[str] = field(default=None, metadata={"help": "the list of convsearch dataset names"})
    neg_type: Literal['hard', 'in_batch'] = field(default='in_batch', metadata={"help": "the type of negative samples"})
    neg_num: int = field(default=0, metadata={"help": "the number of negative samples"})
    
    collection_path: str = field(default=None, metadata={"help": "the collection path for indexing"})
    num_psg_per_block: int = field(default=5000000, metadata={"help": "the number of passages in a block to avoid OOM for indexing"})
    
    # for compatible with session
    query_field_name: Literal["query", "rewrite", "session", None] = field(default=None, metadata={"help": "the field name to use as the query"})
    only_last_response: bool = field(default=False, metadata={"help": "whether to only use the last response in a session for evaluation."})

    max_q_len: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_p_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    
    filter_no_pos: bool = field(default=False, metadata={"help": "whether to filter out samples without positive passages."})
    directly_filter_too_long_session: bool = field(default=False, metadata={"help": "whether to filter out sessions exceed max_q_len at the begining"})
    use_data_percent: List[str] = field(default="1.0", metadata={"help": "The percent of training data to use. The default order is chat->instruct->msmarco->convsearch"})
    data_percent_has_been_used: float = field(default=0.0, metadata={"help": "The percent of training data has been used."})
    
    data_output_dir: str = field(default=None, metadata={"help": "the dir to store outputs, which can be used for various tasks"})
    force_emptying_dir: bool = field(default=False, metadata={"help": "force to emptying the dir"})
    allow_dir_exist: bool = field(default=False, metadata={"help": "whether to allow the output dir exists"})
    
    # for retrieval
    index_dir: str = field(default=None, metadata={"help": "the dir to store pre-encoded passage embeddings, i.e., pre-built index"})
    embedding_size: int = field(default=768, metadata={"help": "the embedding size of the passage embs"})
    num_split_block: int = field(default=1, metadata={"help": "the number of splits of each psg block to avoid OOM "})
    qrel_path: str = field(default=None, metadata={"help": "the path to the qrel file for evaluation"})
    
    def __post_init__(self):
        if self.msmarco_data_path_list:
            self.msmarco_data_path_list = self.msmarco_data_path_list[0].split(' ')
        if self.instruct_data_path_list:
            self.instruct_data_path_list = self.instruct_data_path_list[0].split(' ')
        if self.chat_data_path_list:
            self.chat_data_path_list = self.chat_data_path_list[0].split(' ')
        if self.convsearch_data_path_list:
            self.convsearch_data_path_list = self.convsearch_data_path_list[0].split(' ')
        if self.llm4cs_data_path_list:
            self.llm4cs_data_path_list = self.llm4cs_data_path_list[0].split(' ')
        if self.aug_convsearch_data_path_list:
            self.aug_convsearch_data_path_list = self.convsearch_data_path_list[0].split(' ')

        if not self.use_data_percent:
            self.use_data_percent = [1.0]
        else:
            self.use_data_percent = self.use_data_percent[0].split(' ')
            for i in range(len(self.use_data_percent)):
                self.use_data_percent[i] = float(self.use_data_percent[i])

@dataclass
class MatchingTrainingArguments:
    min_lr: float = field(default=0.0, metadata={"help": "The minimum learning rate in the cosine annealing scheduler."})
    
    # about loss
    temperature: float = field(default=1.0, metadata={"help": "The temperature of the ranking loss."})
    loss_type: Literal['ranking', 'kd', 'kd+ranking'] = field(default='ranking', metadata={"help": "The type of loss."})
    regularizer_type: Literal['L0', 'L1', 'FLOPS', None] = field(default=None, metadata={"help": "The type of regularizer."})
    ranking_loss_weight: float = field(default=0.0, metadata={"help": "The weight of the ranking loss."})
    kd_loss_weight: float = field(default=0.0, metadata={"help": "The weight of the kd loss."})
    regularization_weight: float = field(default=0.0, metadata={"help": "The weight of the regularization loss."})
    inst_loss_weight: float = field(default=0.0, metadata={"help": "The weight of the instruction tuning loss."})
    use_query_mask: bool = field(default=False, metadata={"help": "whether to use query mask for instruction tuning"})

    
@dataclass
class EvalArguments:
    top_n: int = field(default=100, metadata={"help": "the number of passages to retrieve for each query"})
    rel_threshold: float = field(default=1, metadata={"help": "the threshold to determine whether a passage is relevant to a query"})
    need_doc_level_agg: bool = field(default=False, metadata={"help": "whether to aggregate the retrieval results at the document level"})
    need_turn_level_result: bool = field(default=False, metadata={"help": "whether to output the turn-level results"})
    run_trec_dir: str = field(default=None, metadata={"help": "the dir to store the run.trec file"})
    
    per_device_eval_batch_size: int = field(default=64, metadata={"help": "eval batch size per device for inference"})
    

@dataclass
class ContextUnderstandingTestArguments:
    demo_path: str = field(default=None, metadata={"help": "the path to the demonstration file"})
    test_mode: Literal['chat', 'base', None] = field(default=None, metadata={"help": "the test mode"})
    