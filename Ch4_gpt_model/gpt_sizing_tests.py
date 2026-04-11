from config import (GPT_CONFIG_MED, GPT_CONFIG_LG, GPT_CONFIG_XL)
from gpt_model import GPTModel

def calculate_params(model):
    return sum(p.numel() for p in model.parameters())

def compute_size(model):
    total_params = calculate_params(model)
    total_size = total_params * 4 / (1024*1024)
    
    return print(f"Model has {total_params:,} parameters and is {total_size: .2f} MB in size.\n")

if __name__ == "__main__":
    gpt_med = GPTModel(GPT_CONFIG_MED)
    gpt_lg = GPTModel(GPT_CONFIG_LG)
    gpt_xl = GPTModel(GPT_CONFIG_XL)

    for i in [gpt_med, gpt_lg, gpt_xl]:
        compute_size(i)