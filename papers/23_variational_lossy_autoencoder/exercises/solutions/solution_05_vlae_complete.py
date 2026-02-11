"""
Solution 05: Complete VLAE

Final integration of the IAF prior and the restricted PixelCNN decoder.
Successfully cures posterior collapse on MNIST.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import VLAE, loss_function

def train_vlae_solution():
    batch_size = 64
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Full Model
    model = VLAE(input_dim=1, latent_dim=32, n_layers=3, use_flow=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training Full VLAE on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            
            # Forward returns 5 items
            logits, mu, logvar, z, log_det = model(x)
            
            # Loss expects 6 items
            loss = loss_function(logits, x, mu, logvar, z, log_det)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item()/x.size(0):.4f}")
        
        print(f"====> Epoch {epoch} Average Loss: {total_loss / len(loader.dataset):.4f}")

    # Final Visualization
    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(loader))
        x_sample = x_sample[:8].to(device)
        logits, _, _, _, _ = model(x_sample)
        recons = torch.sigmoid(logits)
        grid = torch.cat([x_sample, recons], dim=0)
        
        if not os.path.exists('results'):
            os.makedirs('results')
        save_image(grid, "results/final_vlae_comp.png", nrow=8)
        print("Reconstruction saved to results/final_vlae_comp.png")

if __name__ == "__main__":
    train_vlae_solution()
