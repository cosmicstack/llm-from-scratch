import tiktoken
import torch
from torch.cuda import temperature
from Ch4_gpt_model.config import GPT_CONFIG_124M
from Ch4_gpt_model.gpt_model import GPTModel


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
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) #greedy
        idx = torch.cat((idx, idx_next), dim=1) # n_tokens dim
    
    return idx

def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
    
        # Top-k Sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # Followed by Temparature scaling
        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
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