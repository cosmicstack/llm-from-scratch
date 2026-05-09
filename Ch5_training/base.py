
import torch
import tiktoken

from Ch4_gpt_model.gpt_model import GPTModel
from Ch4_gpt_model.run import generate_text_example

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_example(
        model,
        text_to_token_ids(start_context, tokenizer),
        10,
        GPT_CONFIG_124M["context_length"]
    )

    print(f"Output text:\n{token_ids_to_text(token_ids, tokenizer)}")
