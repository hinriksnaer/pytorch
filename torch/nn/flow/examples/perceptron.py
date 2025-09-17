from torch.nn.flow.chain import Chain
import torch
from torch.nn.flow.examples.utils.mnist_trainer import train_and_evaluate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, epochs, lr = 128, 5, 1e-3
    torch.manual_seed(0)

    model = (
        Chain()
        .tensor.flatten(start_dim=1)     # keep batch dim intact
        .nn.Linear(28*28, 128)
        .nn.GELU()                       # add a nonlinearity
        .nn.LayerNorm(128)
        .nn.Dropout(0.1)
        .nn.Linear(128, 10)              # logits for CrossEntropyLoss
        .freeze()
        .to(device)
    )

    # Use the compiled model if you want compile speedups
    model = torch.compile(model)

    train_and_evaluate(model, epochs=epochs, lr=lr, batch_size=batch_size)
