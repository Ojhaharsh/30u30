"""
Day 15: Training Script for Bahdanau Attention Seq2Seq

Trains a sequence-to-sequence model with attention on a toy copy/reverse task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
from implementation import create_model, plot_attention


# ============================================================================
# Toy Dataset: Number Reversal
# ============================================================================

class ReversalDataset(Dataset):
    """
    Toy dataset: reverse a sequence of numbers.
    
    Example: [5, 3, 8, 2] -> [2, 8, 3, 5]
    
    This is perfect for testing attention because:
    - Clear alignment pattern (reversed diagonal)
    - No complex semantics to learn
    - Easy to verify correctness
    """
    
    def __init__(self, num_samples=10000, min_len=5, max_len=15, vocab_size=100):
        self.samples = []
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            # Generate random sequence (avoid special tokens 0, 1, 2)
            src = [random.randint(3, vocab_size - 1) for _ in range(length)]
            # Target is reversed
            trg = list(reversed(src))
            self.samples.append((src, trg))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        # Add SOS and EOS to target
        trg = [self.sos_idx] + trg + [self.eos_idx]
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    """
    Custom collate function for padding.
    """
    srcs, trgs = zip(*batch)
    
    # Get lengths
    src_lengths = torch.tensor([len(s) for s in srcs])
    
    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    trg_padded = nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=0)
    
    return src_padded, src_lengths, trg_padded


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    for src, src_lengths, trg in dataloader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(src, src_lengths, trg)
        
        # Reshape for loss: [batch * (trg_len-1), vocab_size]
        output_dim = outputs.shape[-1]
        outputs = outputs.reshape(-1, output_dim)
        
        # Targets: skip first token (SOS), flatten
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, src_lengths, trg in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            trg = trg.to(device)
            
            outputs, _ = model(src, src_lengths, trg)
            
            output_dim = outputs.shape[-1]
            outputs = outputs.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(outputs, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def calculate_accuracy(model, dataset, device, num_samples=100):
    """Calculate exact match accuracy."""
    model.eval()
    correct = 0
    
    for i in range(min(num_samples, len(dataset))):
        src, trg = dataset[i]
        src = src.unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(src[0])]).to(device)
        
        with torch.no_grad():
            pred, _ = model.translate(
                src, src_lengths, 
                max_len=len(trg) + 5,
                sos_idx=1, eos_idx=2
            )
        
        # Remove EOS and compare
        pred = pred[0].cpu().tolist()
        target = trg[1:-1].tolist()  # Remove SOS and EOS
        
        # Trim prediction at EOS
        if 2 in pred:
            pred = pred[:pred.index(2)]
        
        if pred == target:
            correct += 1
    
    return correct / min(num_samples, len(dataset))


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Configuration
    VOCAB_SIZE = 100
    EMBED_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ReversalDataset(num_samples=10000, vocab_size=VOCAB_SIZE)
    val_dataset = ReversalDataset(num_samples=1000, vocab_size=VOCAB_SIZE)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        src_vocab_size=VOCAB_SIZE,
        trg_vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        
        # Calculate accuracy every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            accuracy = calculate_accuracy(model, val_dataset, device)
            print(f"Epoch {epoch:3d} | Time: {elapsed:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Accuracy: {accuracy:.2%}")
        else:
            print(f"Epoch {epoch:3d} | Time: {elapsed:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    print("=" * 60)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load('best_model.pt'))
    final_accuracy = calculate_accuracy(model, val_dataset, device, num_samples=500)
    print(f"Test Accuracy: {final_accuracy:.2%}")
    
    # Demonstrate on a few examples
    print("\nExample Predictions:")
    print("-" * 40)
    
    model.eval()
    for i in range(3):
        src, trg = val_dataset[i]
        src_tensor = src.unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(src)]).to(device)
        
        with torch.no_grad():
            pred, attention = model.translate(
                src_tensor, src_lengths,
                max_len=len(trg) + 5,
                sos_idx=1, eos_idx=2
            )
        
        pred = pred[0].cpu().tolist()
        if 2 in pred:
            pred = pred[:pred.index(2)]
        
        print(f"Source:  {src.tolist()}")
        print(f"Target:  {trg[1:-1].tolist()}")
        print(f"Predict: {pred}")
        print(f"Match:   {'✓' if pred == trg[1:-1].tolist() else '✗'}")
        print()
    
    # Visualize attention for one example
    print("Saving attention visualization...")
    src, trg = val_dataset[0]
    src_tensor = src.unsqueeze(0).to(device)
    src_lengths = torch.tensor([len(src)]).to(device)
    
    with torch.no_grad():
        _, attention = model.translate(
            src_tensor, src_lengths,
            max_len=len(trg),
            sos_idx=1, eos_idx=2
        )
    
    # Convert to tokens (just use numbers)
    src_tokens = [str(x) for x in src.tolist()]
    pred_tokens = [str(x) for x in reversed(src.tolist())]
    
    # Trim attention to actual length
    attention = attention[0, :len(pred_tokens), :len(src_tokens)]
    
    fig = plot_attention(
        attention,
        src_tokens,
        pred_tokens,
        figsize=(8, 8),
        save_path='attention_visualization.png'
    )
    print("Saved to attention_visualization.png")
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
