import torch
from torch.functional import _return_counts


def get_token_dist(probas, inverse_vocab, n_iter=1000):
    """
    Monte Carlo Sample from a probability vector.
    Map the token_id to the word from vocabulary.
    Present the distribution.
    """
    samples = torch.multinomial(probas, num_samples=n_iter, replacement=True)
    token_ids, next_token_counts = torch.unique(samples, return_counts=True)
    return {inverse_vocab[token_ids[i].item()]: next_token_counts[i].item()/n_iter for i in range(len(token_ids))}

def temp_scale(logits, temperature):
    return torch.softmax(logits/temperature, dim=0)

if __name__ == "__main__":
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8
    }

    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
    probas = torch.softmax(next_token_logits, dim=0)

    dist = get_token_dist(probas, inverse_vocab)
    print(dist)
    print("===")

    temps = [1, 0.1, 5]
    prob_tensors = [temp_scale(next_token_logits, T) for T in temps]
    for i in range(len(prob_tensors)):
        print("---")
        print(f"For Temperature={temps[i]}: Monte Carlo Dist={get_token_dist(prob_tensors[i], inverse_vocab)}")
        print(f"Scaled Probability={prob_tensors[i]}")