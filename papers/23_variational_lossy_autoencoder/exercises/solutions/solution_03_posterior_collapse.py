"""
Solution 03: Reproducing Posterior Collapse

Experiment showing how a high-capacity decoder without architectural 
constraints chooses the 'easy' path of ignoring the latent code z.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import VLAE, loss_function

def train_collapsing_vae():
    batch_size = 64
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # To ensure collapse, use many decoder layers (e.g. 8) and NO flow prior
    model = VLAE(input_dim=1, latent_dim=32, n_layers=8, use_flow=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training High-Capacity VAE (Check for Collapse) on {device}...")
    
    kl_history = []
    
    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            
            # VLAE forward returns 5 items
            logits, mu, logvar, z, log_det = model(x)
            
            # VLAE loss_function expects 6 items
            loss = loss_function(logits, x, mu, logvar, z, log_det)
            
            # Monitor KL component
            with torch.no_grad():
                # Analytical KL for Gaussian
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_history.append(kld.item() / x.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item()/x.size(0):.4f} KL: {kl_history[-1]:.4f}")

    print("Training complete.")
    
    plt.plot(kl_history)
    plt.title("KL Divergence (Low/Descending = Collapse)")
    plt.xlabel("Step")
    plt.ylabel("KL (nats)")
    plt.show()

if __name__ == "__main__":
    train_collapsing_vae()
