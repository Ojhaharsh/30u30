"""
Solution 4: Train on a Toy Dataset

Complete training loop for the reversal task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os

# Import from solutions
from solution_1 import AdditiveAttention
from solution_2 import BidirectionalEncoder
from solution_3 import AttentionDecoder


# ============================================================================
# Dataset
# ============================================================================

class ReversalDataset(Dataset):
    """
    Dataset that generates (sequence, reversed_sequence) pairs.
    
    Special tokens:
    - 0: PAD
    - 1: SOS (start of sequence)
    - 2: EOS (end of sequence)
    - 3+: actual tokens
    """
    
    def __init__(self, num_samples=5000, min_len=4, max_len=10, vocab_size=50):
        self.samples = []
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.vocab_size = vocab_size
        
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            seq = [random.randint(3, vocab_size - 1) for _ in range(length)]
            reversed_seq = list(reversed(seq))
            self.samples.append((seq, reversed_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        trg = [self.sos_idx] + trg + [self.eos_idx]
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    """Pad sequences to same length."""
    srcs, trgs = zip(*batch)
    
    src_lengths = torch.tensor([len(s) for s in srcs])
    
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    trg_padded = nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=0)
    
    return src_padded, src_lengths, trg_padded


# ============================================================================
# Model Wrapper
# ============================================================================

class Seq2SeqModel(nn.Module):
    """Wraps encoder and decoder into a single model."""
    
    def __init__(self, encoder, decoder, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=1.0):
        src_mask = (src == self.pad_idx)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        outputs, attentions = self.decoder(
            trg, hidden, encoder_outputs, src_mask, teacher_forcing_ratio
        )
        return outputs, attentions
    
    @torch.no_grad()
    def translate(self, src, src_lengths, max_len=20, sos_idx=1, eos_idx=2):
        """Greedy decoding."""
        self.eval()
        
        batch_size = src.size(0)
        device = src.device
        
        src_mask = (src == self.pad_idx)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        prev_token = torch.full((batch_size,), sos_idx, device=device)
        translations = []
        attentions = []
        
        for _ in range(max_len):
            output, hidden, attn = self.decoder.forward_step(
                prev_token, hidden, encoder_outputs, src_mask
            )
            prev_token = output.argmax(dim=-1)
            translations.append(prev_token)
            attentions.append(attn)
            
            if (prev_token == eos_idx).all():
                break
        
        return torch.stack(translations, dim=1), torch.stack(attentions, dim=1)


# ============================================================================
# Training Functions (Complete Solutions)
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    This is the SOLUTION - complete implementation.
    """
    model.train()
    total_loss = 0
    
    for src, src_lengths, trg in loader:
        # Move to device
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(src, src_lengths, trg)
        
        # Reshape for loss computation
        # outputs: (batch, trg_len-1, vocab_size)
        # We compare against trg[:, 1:] (skip SOS)
        output_dim = outputs.shape[-1]
        outputs = outputs.reshape(-1, output_dim)  # (batch * (trg_len-1), vocab_size)
        trg_flat = trg[:, 1:].reshape(-1)  # (batch * (trg_len-1),)
        
        # Compute loss (CrossEntropyLoss ignores padding via ignore_index)
        loss = criterion(outputs, trg_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, src_lengths, trg in loader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            trg = trg.to(device)
            
            # Forward pass (no teacher forcing for evaluation)
            outputs, _ = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)
            
            # Reshape for loss
            output_dim = outputs.shape[-1]
            outputs = outputs.reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            
            loss = criterion(outputs, trg_flat)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def calculate_accuracy(model, dataset, device, num_samples=200):
    """Calculate exact sequence match accuracy."""
    model.eval()
    correct = 0
    
    for i in range(min(num_samples, len(dataset))):
        src, trg = dataset[i]
        src = src.unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(src[0])]).to(device)
        
        pred, _ = model.translate(src, src_lengths)
        pred = pred[0].cpu().tolist()
        
        # Remove EOS and everything after
        if 2 in pred:
            pred = pred[:pred.index(2)]
        
        target = trg[1:-1].tolist()  # Remove SOS and EOS
        
        if pred == target:
            correct += 1
    
    return correct / min(num_samples, len(dataset))


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Hyperparameters
    VOCAB_SIZE = 50
    EMBED_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    print("Creating datasets...")
    train_data = ReversalDataset(num_samples=5000, vocab_size=VOCAB_SIZE)
    val_data = ReversalDataset(num_samples=500, vocab_size=VOCAB_SIZE)
    
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    # Model
    print("Creating model...")
    encoder = BidirectionalEncoder(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    decoder = AttentionDecoder(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        encoder_hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    model = Seq2SeqModel(encoder, decoder).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train!
    print("\nTraining...")
    print("=" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Could save model here: torch.save(model.state_dict(), 'best_model.pt')
        
        if epoch % 5 == 0 or epoch == EPOCHS:
            acc = calculate_accuracy(model, val_data, device)
            print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2%}")
        else:
            print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    # Final evaluation
    print("=" * 50)
    final_acc = calculate_accuracy(model, val_data, device, num_samples=500)
    print(f"\nFinal Accuracy: {final_acc:.2%}")
    
    if final_acc > 0.9:
        print("🎉 Success! Model achieves >90% accuracy!")
    elif final_acc > 0.7:
        print("Good progress! Try more epochs or tuning hyperparameters.")
    else:
        print("Keep training or check your implementation!")
    
    # Show some examples
    print("\n" + "=" * 50)
    print("Examples:")
    print("=" * 50)
    
    model.eval()
    for i in range(5):
        src, trg = val_data[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        pred, attentions = model.translate(src_t, src_len)
        pred = pred[0].cpu().tolist()
        if 2 in pred:
            pred = pred[:pred.index(2)]
        
        target = trg[1:-1].tolist()
        correct = "✓" if pred == target else "✗"
        
        print(f"\n{correct} Example {i+1}:")
        print(f"   Input:  {src.tolist()}")
        print(f"   Pred:   {pred}")
        print(f"   Target: {target}")
    
    return model, val_data


if __name__ == '__main__':
    model, data = main()
