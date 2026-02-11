"""
train_minimal.py - Variational Lossy Autoencoder (VLAE) Training CLI

Train a VLAE on binarized MNIST to demonstrate the cure for posterior collapse.
The script tracks Reconstruction Loss, KL Divergence, and Bits-per-dimension (BPD).

Usage:
    # Basic training
    python train_minimal.py --epochs 10 --latent-dim 32
    
    # Training with IAF Prior Flow
    python train_minimal.py --epochs 10 --use-flow
    
    # Compare posterior collapse by changing decoder layers
    python train_minimal.py --decoder-layers 8  # Use many layers to see collapse

Reference: Chen et al. (2017) "Variational Lossy Autoencoder"
Author: 30u30 Project
"""

import argparse
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from typing import Dict, List

from implementation import VLAE, loss_function
from visualization import (
    plot_training_trajectories, 
    plot_reconstructions, 
    plot_latent_sampling, 
    plot_bits_heatmap,
    plot_latent_traversals
)


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        logits, mu, logvar, z, log_det = model(data)
        loss = loss_function(logits, data, mu, logvar, z, log_det)
        
        # Individual components for tracking
        with torch.no_grad():
            recon_part = torch.nn.functional.binary_cross_entropy_with_logits(logits, data, reduction='sum')
            kl_part = loss - recon_part

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_part.item()
        total_kl += kl_part.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f} (R: {recon_part.item()/len(data):.4f}, KL: {kl_part.item()/len(data):.4f})')

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon / len(train_loader.dataset)
    avg_kl = total_kl / len(train_loader.dataset)
    
    # BPD = Total Loss in bits / (dim_x * log(2))
    bpd = avg_loss / (28 * 28 * 0.6931) 
    
    return avg_loss, avg_recon, avg_kl, bpd


def main():
    parser = argparse.ArgumentParser(description='VLAE Training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--latent-dim', type=int, default=32, help='latent dimension size')
    parser.add_argument('--use-flow', action='store_true', help='use IAF prior flow')
    parser.add_argument('--decoder-layers', type=int, default=3, help='number of gated pixelcnn layers')
    parser.add_argument('--output-dir', type=str, default='results', help='directory for results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=" * 60)
    print(f"Training VLAE (Flow: {args.use_flow}, Decoder Layers: {args.decoder_layers})")
    print(f"Device: {device}")
    print("=" * 60)

    # Data Loader (Binarized MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=args.batch_size, shuffle=False)

    # Model & Optimizer
    model = VLAE(input_dim=1, latent_dim=args.latent_dim, n_layers=args.decoder_layers, use_flow=args.use_flow).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'train_recon': [], 'train_kl': [], 'bpd': []}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, recon, kl, bpd = train(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(loss)
        history['train_recon'].append(recon)
        history['train_kl'].append(kl)
        history['bpd'].append(bpd)
        print(f"====> Epoch {epoch} Average Loss: {loss:.4f} | BPD: {bpd:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.1f}s")

    # Final Visualizations
    print("\nGenerating final plots...")
    plot_training_trajectories(history, args.output_dir)
    
    test_data, _ = next(iter(test_loader))
    plot_reconstructions(model, test_data[:8], device, args.output_dir)
    
    # Bits allocation for mu/logvar from the last batch
    # (Extracting mu/logvar from last forward pass)
    with torch.no_grad():
        _, mu, logvar, _, _ = model(test_data[:100].to(device))
        plot_bits_heatmap(mu, logvar, args.output_dir)

    # Sampling (Slow, only doing 16 samples)
    plot_latent_sampling(model, device, args.latent_dim, n_samples=16, output_dir=args.output_dir)
    
    # Latent Traversals
    print("Generating latent traversals...")
    plot_latent_traversals(model, test_data[:1], device, args.latent_dim, dim_idx=0, output_dir=args.output_dir)
    
    # Save Model
    save_path = os.path.join(args.output_dir, "vlae_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
