"""
Exercise 4: Train on a Toy Dataset

Time to put it all together. We train on a simple task:
reversing a sequence of numbers.

    Input:  [5, 3, 8, 2, 1]
    Output: [1, 2, 8, 3, 5]

Why this task?
1. Easy to verify correctness
2. Clear attention pattern (reversed diagonal)
3. Tests if the model can learn complex dependencies
4. Fast to train (no large datasets needed)

Your task: Complete the training loop and achieve >90% accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from solutions, fall back to implementation
try:
    from solutions.solution_1 import AdditiveAttention
    from solutions.solution_2 import BidirectionalEncoder
    from solutions.solution_3 import AttentionDecoder
except ImportError:
    from implementation import BahdanauAttention as AdditiveAttention
    from implementation import Encoder as BidirectionalEncoder
    from implementation import AttentionDecoder


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
            # Use tokens 3 to vocab_size-1 (avoid special tokens)
            seq = [random.randint(3, vocab_size - 1) for _ in range(length)]
            reversed_seq = list(reversed(seq))
            self.samples.append((seq, reversed_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        # Add SOS and EOS to target
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
    
    def forward(self, src, src_lengths, trg):
        # Create source mask
        src_mask = (src == self.pad_idx)
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Decode
        outputs, attentions = self.decoder(trg, hidden, encoder_outputs, src_mask)
        
        return outputs, attentions
    
    @torch.no_grad()
    def translate(self, src, src_lengths, max_len=20, sos_idx=1, eos_idx=2):
        """Greedy decoding."""
        self.eval()
        
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        src_mask = (src == self.pad_idx)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Decode
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
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """
    TODO: Implement one training epoch.
    
    Steps:
    1. Set model to training mode
    2. Loop through batches
    3. Move data to device
    4. Zero gradients
    5. Forward pass
    6. Reshape outputs and targets for loss
    7. Compute loss (ignore padding)
    8. Backward pass
    9. Clip gradients (max norm = 1.0)
    10. Optimizer step
    11. Accumulate loss
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for src, src_lengths, trg in loader:
        # TODO: Implement the training step
        
        # Move to device
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        
        # TODO: Complete the rest
        # optimizer.zero_grad()
        # outputs, _ = model(src, src_lengths, trg)
        # ... reshape and compute loss ...
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()
        
        raise NotImplementedError("Implement train_epoch!")
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    TODO: Implement evaluation (similar to training but no gradients).
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, src_lengths, trg in loader:
            # TODO: Implement evaluation step
            raise NotImplementedError("Implement evaluate!")
    
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
        
        # Remove EOS and after
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
        dropout=DROPOUT
    )
    
    model = Seq2SeqModel(encoder, decoder).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train!
    print("\nTraining...")
    print("=" * 50)
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if epoch % 5 == 0:
            acc = calculate_accuracy(model, val_data, device)
            print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2%}")
        else:
            print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    # Final test
    print("=" * 50)
    final_acc = calculate_accuracy(model, val_data, device, num_samples=500)
    print(f"\nFinal Accuracy: {final_acc:.2%}")
    
    if final_acc > 0.9:
        print("Success: Model achieves >90% accuracy.")
    else:
        print("Accuracy below 90%. Adjust hyperparameters and retrain.")
    
    # Show some examples
    print("\nExamples:")
    for i in range(3):
        src, trg = val_data[i]
        src_t = src.unsqueeze(0).to(device)
        src_len = torch.tensor([len(src)]).to(device)
        
        pred, _ = model.translate(src_t, src_len)
        pred = pred[0].cpu().tolist()
        if 2 in pred:
            pred = pred[:pred.index(2)]
        
        print(f"  In:  {src.tolist()}")
        print(f"  Out: {pred}")
        print(f"  Tgt: {trg[1:-1].tolist()}")
        print()


if __name__ == '__main__':
    main()
