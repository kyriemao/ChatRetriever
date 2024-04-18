
from .data_sample import MatchingSample
from IPython import embed


class TextFormatter:
    def __init__(self, tokenizer, model_type, data_args):
        self.tokenizer = tokenizer
        self.model_type = model_type
        
        # set suffix
        if self.model_type in ['ance', 'bge', 'llm_embedder', 'splade', 'gtr', 'qwen_lm_inference', 'openai']:
            self.q_suffix = None
            self.p_suffix = None
        elif self.model_type in ['repllama', 'repllama-train', 'repllama_v2', 'repllama_v2-train', 'repllama_chat', 'repllama_chat-train', 'e5_mistrial', 'e5_mistrial-train']:
            self.q_suffix = "</s>"
            self.p_suffix = "</s>"
        elif self.model_type in ['repllama_chat_cot', 'repllama_chat_cot-train']:
            self.q_suffix = "<|emb_0|><|emb_1|><|emb_2|>"
            self.p_suffix = "<|emb_0|><|emb_1|><|emb_2|>"
        elif self.model_type in ['qwen', 'qwen_chat', 'qwen_chat_lora', 'qwen_chat_lora_eval', 'qwen15', 'qwen15_chat', 'qwen15_chat_lora', 'qwen15_chat_lora_eval', 'qwen_lm']:
            self.q_suffix = "<|endoftext|>"
            self.p_suffix = "<|endoftext|>"
        elif self.model_type in ['qwen_chat_cot', 'qwen_chat_cot_lora', 'qwen_chat_cot_lora_eval']:
            self.q_suffix = "<|extra_0|><|extra_1|><|extra_2|>"
            self.p_suffix = "<|extra_0|><|extra_1|><|extra_2|>"
        elif self.model_type in ['qwen15_chat_cot', 'qwen15_chat_cot_lora', 'qwen15_chat_cot_lora_eval']:
            self.q_suffix = "<|endoftext|><|endoftext|><|endoftext|>"
            self.p_suffix = "<|endoftext|><|endoftext|><|endoftext|>"
        elif self.model_type in ['mistrial_cot-train', 'mistrial_chat_cot-train', 'mistrial_cot', 'mistrial_chat_cot']:
            self.q_suffix = "</s></s></s>"
            self.p_suffix = "</s></s></s>"
        else:
            raise KeyError("Unknown model type: {}".format(self.model_type))

        # To form session
        self.only_last_response = data_args.only_last_response
        
    def __call__(self, sample, is_only_psg=False):
        if is_only_psg:
            # change a passage into a pseudo sample
            sample = MatchingSample(sample_idx=None, pos_psg=sample)
        
        if self.model_type in ['repllama', 'repllama-train']:
            self._repllama_text_format(sample)
        elif self.model_type in ['repllama_v2', 'repllama_v2-train', 'repllama_v2-continue_train']:
            self._repllama_v2_text_format(sample)
        elif self.model_type in ['repllama_chat', 'repllama_chat-train']:
            self._repllama_chat_text_format(sample)
        elif self.model_type in ['repllama_chat_cot', 'repllama_chat_cot-train']:
            self._repllama_chat_cot_text_format(sample)
        elif self.model_type in ['qwen', 'qwen15']:
            self._qwen_text_format(sample)
        elif self.model_type in ['qwen_chat', 'qwen_chat_lora', 'qwen_chat_lora_eval', 'qwen15_chat', 'qwen15_chat_lora', 'qwen15_chat_lora_eval']:
            self._qwen_chat_text_format(sample)
        elif self.model_type in ['qwen_chat_cot', 'qwen_chat_cot_lora', 'qwen_chat_cot_lora_eval']:
            self._qwen_chat_cot_text_format(sample)
        elif self.model_type in ['qwen15_chat_cot', 'qwen15_chat_cot_lora', 'qwen15_chat_cot_lora_eval']:
            self._qwen15_chat_cot_text_format(sample)
        elif self.model_type == 'bge':
            self._bge_text_format(sample)
        elif self.model_type == 'llm_embedder':
            self._llm_embedder_text_format(sample)
        elif self.model_type == 'splade':
            self._splade_text_format(sample)
        elif self.model_type == 'qwen_lm':
            self._qwen_lm_text_format(sample)
        elif self.model_type == 'qwen_lm_inference':
            self._qwen_lm_inference_text_format(sample)
        elif self.model_type == 'openai':
            self._openai_text_format(sample)
        elif self.model_type in ['e5_mistrial', 'e5_mistrial-train']:
            self._e5_mistrial_text_format(sample)
        elif self.model_type in ['mistrial_cot-train', 'mistrial_cot']:
            self._mistrial_cot_text_format(sample)
        elif self.model_type in ['mistrial_chat_cot-train', 'mistrial_chat_cot']:
            self._mistrial_chat_cot_text_format(sample)
        else:
            self._vanilla_text_format(sample)
        
        return sample
    
    
    def _qwen15_chat_cot_text_format(self, sample: MatchingSample):
        def _wrap_text(system, enable_sys_prompt=False, query=None, history=None, psg=None):
            def _format_raw(role, content):
                return f"{role}\n{content}"
            
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            to_concat = []
            if enable_sys_prompt:
                system_text = _format_raw("system", content=system)
                to_concat.append(f"{im_start}{system_text}{im_end}")
            
            if query is not None:
                if history is None:
                    history = []     
                for turn in history:
                    if turn['role'] == 'User':
                        text = _format_raw("user", turn['text'])
                    elif turn['role'] == 'Assistant':
                        text = _format_raw("assistant", turn['text'])
                    else:
                        raise KeyError("Unknown role: {}".format(turn['role']))
                    to_concat.append(f"{im_start}{text}{im_end}")
                
                text = _format_raw("user", query)
                to_concat.append(f"{im_start}{text}{im_end}")
                raw_text = "\n".join(to_concat)
                
            elif psg is not None:     
                text = _format_raw("assistant", psg)
                raw_text = f"{im_start}{text}{im_end}"
                 
            return raw_text
        
        if sample.query:
            sample.session = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.query,
                                        history=sample.history)
            sample.session = "{}<|endoftext|><|endoftext|><|endoftext|>".format(sample.session)
            sample.query = _wrap_text(system="You are a helpful assistant.",
                                      query=sample.query, 
                                      history=[])
            sample.query = "{}<|endoftext|><|endoftext|><|endoftext|>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = _wrap_text(system="Repeat the content.",
                                        psg=sample.pos_psg,
                                        history=[])
            sample.pos_psg = "{}<|endoftext|><|endoftext|><|endoftext|>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = [_wrap_text(system="Repeat the content.",
                                          psg=psg,
                                          history=[]) for psg in sample.neg_psgs]
            sample.neg_psgs = ["{}<|endoftext|><|endoftext|><|endoftext|>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.rewrite,
                                        history=[])
            sample.rewrite = "{}<|endoftext|><|endoftext|><|endoftext|>".format(sample.rewrite)
    
    
    
    def _mistrial_chat_cot_text_format(self, sample: MatchingSample):
        def _wrap_mistrial_chat_template(query, history=[], is_psg=False):
            if is_psg:
                return "[INST] [/INST]{}".format(query)
            
            res = ""
            for i in range(len(history)):
                if i % 2 == 0:
                    res += "[INST] {} [/INST]".format(history[i])
                else:
                    res += "{}</s> ".format(history[i])
            res += "[INST] {} [/INST]".format(query)
            return res
            
        if sample.query:
            sample.session = _wrap_mistrial_chat_template(sample.query, sample.history)
            sample.session = "{}</s></s></s>".format(sample.session)
            sample.query = _wrap_mistrial_chat_template(sample.query)
            sample.query = "{}</s></s></s>".format(sample.query)
        
        if sample.pos_psg:
            sample.pos_psg = _wrap_mistrial_chat_template(query=sample.pos_psg, is_psg=True)
            sample.pos_psg = "{}</s></s></s>".format(sample.pos_psg)
        if sample.neg_psgs:
            neg_psgs = []
            for psg in sample.neg_psgs:
                neg_psgs.append(_wrap_mistrial_chat_template(query=psg, is_psg=True))                
            sample.neg_psgs = ["{}</s></s></s>".format(psg) for psg in neg_psgs]
        
        if sample.rewrite:
            sample.rewrite = _wrap_mistrial_chat_template(sample.rewrite)
            sample.rewrite = "{}</s></s></s>".format(sample.rewrite)
            
            
            
    def _mistrial_cot_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="chat")
            sample.session = "{}</s></s></s>".format(sample.session)
            sample.query = "User: {}</s></s></s>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}</s></s></s>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}</s></s></s>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "User: {}</s></s></s>".format(sample.rewrite)
            
            
    def _e5_mistrial_text_format(self, sample: MatchingSample):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'
        
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="reverse_concat")
            sample.session = get_detailed_instruct(task, sample.session)
            sample.session = "{}</s>".format(sample.session)
            sample.query = get_detailed_instruct(task, sample.query)
            sample.query = "{}</s>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}</s>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}</s>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = get_detailed_instruct(task, sample.rewrite)
            sample.rewrite = "{}</s>".format(sample.rewrite)
            
        
    def _repllama_chat_cot_text_format(self, sample: MatchingSample):
        _START = "[INST] <<SYS>>\n\n<</SYS>>\n\n"   # note that llama2 tokenizer will automatically add <s> at the beginning
        
        def _wrap_session(query, history):
            # handle the first turn
            if not history:
                return "{}{} [/INST]<|emb_0|><|emb_1|><|emb_2|>".format(_START, query)
            
            session = _START
            for i in range(len(history)):
                if history[i]['role'] == 'User':
                    if i == 0:
                        session += "{} [/INST]".format(history[i]['text'])
                    else:
                        session += "<s>[INST] {} [/INST]".format(history[i]['text'])
                elif history[i]['role'] == 'Assistant':
                    session += " {} </s>".format(history[i]['text'])
                else:
                    raise ValueError("Unknown role: {}".format(history[i]['role']))
            session += "<s>[INST] {} [/INST]<|emb_0|><|emb_1|><|emb_2|>".format(query)
            return session
        
        if sample.query:
            sample.session = _wrap_session(sample.query, sample.history)
            sample.query = "{}{} [/INST]<|emb_0|><|emb_1|><|emb_2|>".format(_START, sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}{} [/INST] {} <|emb_0|><|emb_1|><|emb_2|>".format(_START, "", sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}{} [/INST] {} <|emb_0|><|emb_1|><|emb_2|>".format(_START, "", psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "{}{} [/INST]<|emb_0|><|emb_1|><|emb_2|>".format(_START, sample.rewrite)
            
            
    def _repllama_chat_text_format(self, sample: MatchingSample):
        _START = "[INST] <<SYS>>\n\n<</SYS>>\n\n"   # note that llama2 tokenizer will automatically add <s> at the beginning
        
        def _wrap_session(query, history):
            # handle the first turn
            if not history:
                return "{}{} [/INST]</s>".format(_START, query)
            
            session = _START
            for i in range(len(history)):
                if history[i]['role'] == 'User':
                    if i == 0:
                        session += "{} [/INST]".format(history[i]['text'])
                    else:
                        session += "<s>[INST] {} [/INST]".format(history[i]['text'])
                elif history[i]['role'] == 'Assistant':
                    session += " {} </s>".format(history[i]['text'])
                else:
                    raise ValueError("Unknown role: {}".format(history[i]['role']))
            session += "<s>[INST] {} [/INST]</s>".format(query)
            return session
        
        if sample.query:
            sample.session = _wrap_session(sample.query, sample.history)
            sample.query = "{}{} [/INST]</s>".format(_START, sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}{} [/INST] {} </s>".format(_START, "", sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}{} [/INST] {} </s>".format(_START, "", psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "{}{} [/INST]</s>".format(_START, sample.rewrite)
            
    def _qwen_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="chat")
            sample.session = "{}<|endoftext|>".format(sample.session)
            sample.query = "User: {}<|endoftext|>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}<|endoftext|>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}<|endoftext|>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "User: {}<|endoftext|>".format(sample.rewrite)
            
    
    def _qwen_chat_text_format(self, sample: MatchingSample):
        
        def _wrap_text(system, enable_sys_prompt=False, query=None, history=None, psg=None):
            def _format_raw(role, content):
                return f"{role}\n{content}"
            
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            to_concat = []
            if enable_sys_prompt:
                system_text = _format_raw("system", content=system)
                to_concat.append(f"{im_start}{system_text}{im_end}")
            
            if query is not None:
                if history is None:
                    history = []     
                for turn in history:
                    if turn['role'] == 'User':
                        text = _format_raw("user", turn['text'])
                    elif turn['role'] == 'Assistant':
                        text = _format_raw("assistant", turn['text'])
                    else:
                        raise KeyError("Unknown role: {}".format(turn['role']))
                    to_concat.append(f"{im_start}{text}{im_end}")
                
                text = _format_raw("user", query)
                to_concat.append(f"{im_start}{text}{im_end}")
                raw_text = "\n".join(to_concat)
                
            elif psg is not None:     
                text = _format_raw("assistant", psg)
                raw_text = f"{im_start}{text}{im_end}"
                 
            return raw_text
        
        if sample.query:
            sample.session = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.query,
                                        history=sample.history)
            sample.session = "{}<|endoftext|>".format(sample.session)
            sample.query = _wrap_text(system="You are a helpful assistant.",
                                      query=sample.query, 
                                      history=[])
            sample.query = "{}<|endoftext|>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = _wrap_text(system="Repeat the content.",
                                        psg=sample.pos_psg,
                                        history=[])
            sample.pos_psg = "{}<|endoftext|>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = [_wrap_text(system="Repeat the content.",
                                          psg=psg,
                                          history=[]) for psg in sample.neg_psgs]
            sample.neg_psgs = ["{}<|endoftext|>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.rewrite,
                                        history=[])
            sample.rewrite = "{}<|endoftext|>".format(sample.rewrite)
            
    def _qwen_chat_cot_text_format(self, sample: MatchingSample):
        
        def _wrap_text(system, enable_sys_prompt=False, query=None, history=None, psg=None):
            def _format_raw(role, content):
                return f"{role}\n{content}"
            
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            to_concat = []
            if enable_sys_prompt:
                system_text = _format_raw("system", content=system)
                to_concat.append(f"{im_start}{system_text}{im_end}")
            
            if query is not None:
                if history is None:
                    history = []     
                for turn in history:
                    if turn['role'] == 'User':
                        text = _format_raw("user", turn['text'])
                    elif turn['role'] == 'Assistant':
                        text = _format_raw("assistant", turn['text'])
                    else:
                        raise KeyError("Unknown role: {}".format(turn['role']))
                    to_concat.append(f"{im_start}{text}{im_end}")
                
                text = _format_raw("user", query)
                to_concat.append(f"{im_start}{text}{im_end}")
                raw_text = "\n".join(to_concat)
                
            elif psg is not None:     
                text = _format_raw("assistant", psg)
                raw_text = f"{im_start}{text}{im_end}"
                 
            return raw_text
        
        if sample.query:
            sample.session = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.query,
                                        history=sample.history)
            sample.session = "{}<|extra_0|><|extra_1|><|extra_2|>".format(sample.session)
            sample.query = _wrap_text(system="You are a helpful assistant.",
                                      query=sample.query, 
                                      history=[])
            sample.query = "{}<|extra_0|><|extra_1|><|extra_2|>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = _wrap_text(system="Repeat the content.",
                                        psg=sample.pos_psg,
                                        history=[])
            sample.pos_psg = "{}<|extra_0|><|extra_1|><|extra_2|>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = [_wrap_text(system="Repeat the content.",
                                          psg=psg,
                                          history=[]) for psg in sample.neg_psgs]
            sample.neg_psgs = ["{}<|extra_0|><|extra_1|><|extra_2|>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = _wrap_text(system="You are a helpful assistant.",
                                        query=sample.rewrite,
                                        history=[])
            sample.rewrite = "{}<|extra_0|><|extra_1|><|extra_2|>".format(sample.rewrite)
    
    def _qwen_lm_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.query = "{}<|endoftext|>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}<|endoftext|>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}<|endoftext|>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "{}<|endoftext|>".format(sample.rewrite)
    
    def _qwen_lm_inference_text_format(self, sample: MatchingSample):
        return sample
    
    def _openai_text_format(self, sample: MatchingSample):
        return sample
        
    def _repllama_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="chat")
            sample.session = "query: {}</s>".format(sample.session)
            sample.query = "query: {}</s>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "passage: {}</s>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["passage: {}</s>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "query: {}</s>".format(sample.rewrite)

    def _repllama_v2_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="chat")
            sample.session = "{}</s>".format(sample.session)
            sample.query = "User: {}</s>".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}</s>".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}</s>".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "User: {}</s>".format(sample.rewrite)
        
    def _bge_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="reverse_concat")
            sample.query = "Represent this sentence for searching relevant passages: {}".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "{}".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "Represent this sentence for searching relevant passages: {}".format(sample.rewrite)
        
    def _llm_embedder_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="llm_embedder")
            sample.session = "Encode this query and context for searching relevant passages: {}".format(sample.session)
            sample.query = "Represent this query for retrieving relevant documents: {}".format(sample.query)
        if sample.pos_psg:
            sample.pos_psg = "Represent this document for retrieval: {}".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["Represent this document for retrieval: {}".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "Represent this query for retrieving relevant documents: {}".format(sample.rewrite)
        
    def _vanilla_text_format(self, sample: MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="reverse_concat")
            sample.query = "{}".format(sample.query)
        if sample.pos_psg:       
            sample.pos_psg = "{}".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "{}".format(sample.rewrite)
            
    def _splade_text_format(self, sample:MatchingSample):
        if sample.query:
            sample.session = self._form_session(sample.query, sample.history, strategy="splade")
            sample.query = "{}".format(sample.query)
        if sample.pos_psg:       
            sample.pos_psg = "{}".format(sample.pos_psg)
        if sample.neg_psgs:
            sample.neg_psgs = ["{}".format(psg) for psg in sample.neg_psgs]
        if sample.rewrite:
            sample.rewrite = "{}".format(sample.rewrite)

    def _form_session(self, query, history, strategy):
        if strategy == 'reverse_concat':
            return self._form_session_reverse_concat(query, history)
        elif strategy == 'llm_embedder':
            return self._form_session_llm_embedder(query, history)
        elif strategy == 'chat':
            return self._form_session_chat(query, history)
        elif strategy == 'splade':
            return self._form_session_splade(query, history)
        else:
            raise NotImplementedError("Session organize strategy {} not implemented.".format(strategy))
        

    def _form_session_reverse_concat(self, query, history):
        to_concat = []
        if history:
            history = [h['text'] for h in history]
            to_concat.extend(history)
        to_concat.append(query)
        to_concat.reverse()
        sep_token = self.tokenizer.sep_token
        if sep_token is None:
            assert self.tokenizer.eos_token
            sep_token = self.tokenizer.eos_token
        
        return " {} ".format(sep_token).join(to_concat)


    def _form_session_llm_embedder(self, query, history):
        # the original method in LLM embedder paper
        to_concat = []
        if history:
            history = [h['text'] for h in history]
            to_concat.extend(history)
        to_concat.append(query)
        return " ".join(to_concat)
        
    # A chat format to organize the session, hope llm-based retriever can understand the search intents from such a complex input
    # Refer the chat template in kindred/src/kindred/data/chat_dataset.py: CHAT_TEMPLATE
    def _form_session_chat(self, query, history):
        to_concat = []
        if history:
            for i in range(len(history)):
                to_concat.append("{}: {}".format(history[i]['role'], history[i]['text']))
                
        to_concat.append("User: {}".format(query))    
        return "\n\n".join(to_concat)

    
    def _form_session_splade(self, query, history):
        flat_concat = []
        if len(history) == 0:
            ctx_utts_text = []
            last_response_text = ""
        else:
            ctx_utts_text = [x['text'] for x in history[:-1]]
            last_response_text = history[-1]['text']
        cur_utt = self.tokenizer.encode(query, add_special_tokens = True, max_length = 32)
        flat_concat.extend(cur_utt)

        if len(last_response_text) > 0:
            last_response = self.tokenizer.encode(last_response_text, add_special_tokens=True, max_length=128)
            flat_concat.extend(last_response)

        for j in range(len(ctx_utts_text) - 1, -1, -1):
            utt = self.tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=32) # not remove [CLS]
            if len(flat_concat) + len(utt) > 256:
                flat_concat += utt[:256 - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                break
            else:
                flat_concat.extend(utt)
        
        flat_concat = self.tokenizer.decode(flat_concat)

        return flat_concat
        
    
    