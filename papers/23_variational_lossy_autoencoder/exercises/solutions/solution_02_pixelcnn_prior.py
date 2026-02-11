"""
Solution 02: Gated PixelCNN Prior

Building a standalone autoregressive density estimator for MNIST.
Demonstrates pixel-by-pixel modeling using Masked Convolutions and Gated Activations.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path to import implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import MaskedConv2d, GatedActivation


class SimplePixelCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layers=4):
        super().__init__()
        self.initial_conv = MaskedConv2d('A', in_channels, hidden_dim, 7, padding=3)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                MaskedConv2d('B', hidden_dim, hidden_dim * 2, 3, padding=1),
                GatedActivation()
            ) for _ in range(n_layers)
        ])
        
        self.final_conv = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x):
        h = self.initial_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.final_conv(h)


def train_pixelcnn():
    batch_size = 64
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimplePixelCNN(1, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training PixelCNN on {device}...")
    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            loss = criterion(logits, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    print("Training complete.")


if __name__ == "__main__":
    train_pixelcnn()
