import json
from IPython import embed

from transformers import HfArgumentParser
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
from kindred.models.model_utils import load_model
from kindred.arguments import ModelArguments, DataArguments, ConvSearchArguments, ContextUnderstandingTestArguments 
from kindred.data.data_utils import load_dataset
from tqdm import tqdm


# CQR ability
CQR_CHAT_WITH_DEMO_TEMPLATE = """
Below is a information-seeking conversation between a user and an AI assistant. Reformulate the current user question into a rewrite that can comprehensively express the user‘s information needs without the need of context. I will first provide you a few examples. Directly output the rewrite in your response and avoid to include any other text.

{demo}

[Your Task]
{conv}

Current Question: {query}
""".strip('\n')

# CQR_CHAT_WITH_DEMO_TEMPLATE = """
# Below is a information-seeking conversation between a user and an AI assistant. Reformulate the current user question into a rewrite that can comprehensively express the user‘s information needs without the need of context.

# [Conversation context]:
# {conv}

# [Current Question]:
# {query}

# """.strip('\n')



CQR_BASE_TEMPLATE = """
Below is a information-seeking conversation between a user and an AI assistant. Reformulate the user's lastest question into a into a rewrite that can fully express the user‘s information needs without the need of the conversation context.

{demo}

{this_conv}
""".strip('\n')



# Chat response generation ability
RG_BASE_TEMPLATE = """
{}

Assistant:
""".strip('\n')

RG_CHAT_TEMPLATE = """
Generate the response to the user's utterance.

{}
""".strip('\n')


class CQRTester:
    def __init__(self, test_mode, model, model_type, tokenizer, demo_path):
        self.test_mode = test_mode
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        
        self.demo = self.form_demo_field(demo_path)
        
    def chat_test(self, conv, query):
        prompt = CQR_CHAT_WITH_DEMO_TEMPLATE.format(demo=self.demo, conv=conv, query=query)
        if self.model_type == 'qwen':
            response, history = self.model.chat(self.tokenizer, prompt, history=None)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(input_ids)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response, prompt
    
    def base_test(self, conv):
        prompt = CQR_BASE_TEMPLATE.format(demo=self.demo, this_conv=conv)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response, prompt

    def test(self, conv, query=None):
        if self.test_mode == "chat":
            return self.chat_test(conv, query)
        
        elif self.test_mode == "base":
            return self.base_test(conv)
        else:
            raise NotImplementedError
        
    def form_demo_field(self, demo_path):
        if not demo_path:
            return ""
        
        with open(demo_path, 'r') as f:
            conversations = json.load(f)
        
        conversations = conversations[:2]
        print("load demo...")
        demo_prompt = []
        for conv_idx, conv in enumerate(conversations):
            turns = conv['turns']
            to_concat = []
            for turn_idx, turn in enumerate(turns):
                this_conv = []
                this_conv.append("User: {}".format(turn['question']))
                this_conv.append("Assistant: {}".format(turn['response']))
                this_conv.append("Rewrite: {}".format(turn['manual_rewrite']))
                to_concat.append("\n".join(this_conv))
            to_concat = "[Example #{}]:".format(conv_idx+1) + "\n" + "\n\n".join(to_concat)
            demo_prompt.append(to_concat)
            
        return "\n\n".join(demo_prompt)
    

        

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, ConvSearchArguments, ContextUnderstandingTestArguments))
    model_args, data_args, conv_args, cu_args = parser.parse_args_into_dataclasses()

    # 1. load model
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
    if model_args.model_type == 'qwen':
        model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

    # 2. load data
    dataset = load_dataset(tokenizer, data_args, conv_args)
    
    # 3. CQR test
    tester = CQRTester(cu_args.test_mode, model, model_args.model_type, tokenizer, demo_path=cu_args.demo_path)
    
    for sample in tqdm(dataset.samples):
        rewrite, prompt = tester.test(conv=sample.query, query=sample.raw_utt)
        print('outputput')
        embed()
        input()
        
        


if __name__ == '__main__':
    main()
        