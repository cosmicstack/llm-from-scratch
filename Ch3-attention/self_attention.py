import torch
import torch.nn as nn
import numpy as np


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        query = x @ self.W_query
        key = x @ self.W_key
        value = x @ self.W_value
        attn_weights = torch.softmax((query @ key.T) / key.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ value


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        attn_weights = torch.softmax((query @ key.T) / key.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ value


if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    attn = SelfAttentionV1(3, 2)
    context_vec = attn(inputs)
    print("Self Attention v1:\n{}\n\n".format(context_vec))


    attn2 = SelfAttentionV2(3, 2)

    # accessing nn.Module ingormation. using .data in order to not break autograd
    attn2.W_key.weight.data = attn.W_key.T
    attn2.W_query.weight.data = attn.W_query.T
    attn2.W_value.weight.data = attn.W_value.T

        
    context_vec2 = attn2(inputs)
    print("Self Attention v2:\n{}\n\n".format(context_vec2))

    print(context_vec - context_vec2)