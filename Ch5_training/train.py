import torch
import tiktoken
from Ch4_gpt_model.gpt_model import GPTModel
from Ch5_training.train_utils import train_model_simple, plot_losses, train_loader, val_loader
from Ch5_training.base import GPT_CONFIG_124M

if __name__ == "__main__":
    torch.manual_seed(123)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.0004,
        weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tiktoken.get_encoding("gpt2")
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)