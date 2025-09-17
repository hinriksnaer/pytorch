from torch.nn.flow.chain import Chain
import torch

# mnist_mlp_no_torchvision.py
import os, gzip, urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch.nn.flow.examples.utils.mnist_trainer import train_and_evaluate

def linear_block(tc: Chain, in_features, out_features, dropout=0.0):
    return (
        tc
        .nn.Linear(in_features, out_features)
        .nn.LayerNorm(out_features)
        .nn.ReLU()
        .nn.Dropout(dropout)
    )

def stacked_linear_block(tc: Chain, in_features, out_features, num_layers, dropout=0.0):
    return (
        tc
        .repeat(
            num_layers,
            linear_block,
            in_features=in_features,
            out_features=out_features,
            dropout=dropout
        )
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, epochs, lr = 128, 5, 1e-3
    torch.manual_seed(0)

    '''
    A deeper MLP with submodules, this is equivalent to:
    model = (
            Chain()
            .flatten()
            .linear(28*28, 256)
            .layernorm(256)
            .relu()
            .dropout(0.1)
            .linear(256, 256)
            .layernorm(256)
            .relu()
            .dropout(0.1)
            .linear(256, 256)
            .layernorm(256)
            .relu()
            .dropout(0.1)
            .linear(256, 10)
            .freeze()
            .to(device)
    )
    '''

    model = (
            Chain()
            .tensor.flatten(start_dim=1)     # keep batch dim intact
            .pipe(
                linear_block,
                in_features=28*28,
                out_features=256,
                dropout=0.1
            )
            .pipe(
                stacked_linear_block,
                in_features=256,
                out_features=256,
                num_layers=2,
                dropout=0.1
            )
            .nn.Linear(256, 10)
            .freeze()
            .to(device)
    )

    model = torch.compile(model)  # Optional: compile the model for performance

    trained_model = train_and_evaluate(model, epochs=epochs, lr=lr, batch_size=batch_size)
