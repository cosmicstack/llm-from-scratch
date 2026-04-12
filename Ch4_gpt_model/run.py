import tiktoken
import torch
from config import GPT_CONFIG_124M
from gpt_model import GPTModel


def generate_text_example(model, idx, max_new_tokens, context_size):
    '''
    idx.shape is (batch, n_tokens)
    n_tokens is what's added by user and has to be <= context_size
    '''
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad(): # <- what happens here?
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] # logits.shape is (batch, n_tokens, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1) # n_tokens dim
    
    return idx

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)
    print(f"Encoded Tensor shape:{encoded.shape}")

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out = generate_text_example(
        model,
        encoded,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print(f"Output: {out}")
    print(f"Output lenght: {len(out[0])}\n")

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)