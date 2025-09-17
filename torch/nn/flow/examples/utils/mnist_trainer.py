from torch.nn.flow.chain import Chain
import torch

# mnist_mlp_no_torchvision.py
import os, gzip, urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


URLS = {
    "train_images": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train_labels": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "test_images":  "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
}

def download_mnist(path="./data"):
    os.makedirs(path, exist_ok=True)
    for _, url in URLS.items():
        out = os.path.join(path, url.split("/")[-1])
        if not os.path.exists(out):
            print("Downloading", url)
            urllib.request.urlretrieve(url, out)

def load_images(path):
    # Read whole file, then slice off the 16-byte IDX header
    with gzip.open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)[16:]          # no 'offset' arg
    imgs = arr.reshape(-1, 28, 28).astype(np.float32) / 255.0
    return imgs

def load_labels(path):
    # Slice off the 8-byte IDX header
    with gzip.open(path, "rb") as f:
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8)[8:]        # no 'offset' arg
    return labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N,1,28,28)
        self.labels = torch.from_numpy(labels).long()
    def __len__(self): return self.labels.shape[0]
    def __getitem__(self, i): return self.images[i], self.labels[i]


def train_and_evaluate(model: Chain, epochs=5, lr=1e-3, batch_size=128, eval_every=100):

    # load data
    download_mnist()
    train_images = load_images("./data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./data/train-labels-idx1-ubyte.gz")
    test_images  = load_images("./data/t10k-images-idx3-ubyte.gz")
    test_labels  = load_labels("./data/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(MNISTDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(MNISTDataset(test_images, test_labels),   batch_size=1024)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    torch.compile(model)  # Optional: compile the model for performance

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % eval_every == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                      f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
              f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")

    print("Training complete.")
    print("Final model performance:")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
          f" ({100. * correct / len(test_loader.dataset):.0f}%)")

    return model

