import torch
from transformers import DistilBertForMaskedLM
from kindred.retrieval_toolkits.splade_tool import NullContextManager
from IPython import embed



class Splade(torch.nn.Module):
    def __init__(self, model_name_or_path:str, agg="max", fp16=True):
        super(Splade, self).__init__()
     
        self.transformer = DistilBertForMaskedLM.from_pretrained(model_name_or_path)
        self.fp16 = fp16
        assert agg in ("sum", "max") 
        self.agg = agg

    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast() if self.fp16 else NullContextManager():
            # tokens: output of HF tokenizer
            transformer_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            out = self.transformer(**transformer_input)
            out = out['logits']  

        if self.agg == "sum":
                lexical_reps = torch.sum(torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1)
        else:
            lexical_reps, max_indices = torch.max(torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1)  
            
        return lexical_reps
    
    def save_model(self, path):
        self.transformer.save_pretrained(path)
        print("successfully saved splade model.")
