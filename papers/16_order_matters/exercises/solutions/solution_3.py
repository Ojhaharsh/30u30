"""
Solution 3: Training Loop for Sorting

Complete training implementation with proper loss computation and accuracy tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class SortingDataset(Dataset):
    """Generate random sorting problems."""
    
    def __init__(self, num_samples=10000, min_len=5, max_len=10):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: Unsorted numbers [seq_len]
            target: Sorted indices [seq_len]
        """
        seq_len = np.random.randint(self.min_len, self.max_len + 1)
        
        # Random numbers between 0 and 1
        sequence = torch.rand(seq_len)
        
        # Get sorting indices (argsort gives indices that would sort the array)
        target = torch.argsort(sequence)
        
        return sequence, target


class SimplePointerNetwork(nn.Module):
    """Simplified pointer network for sorting."""
    
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        
        # Encoder: LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Decoder: LSTM
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Pointer mechanism
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [batch, seq_len]
        Returns:
            pointers: [batch, seq_len, seq_len] (logits)
        """
        batch_size, seq_len = inputs.shape
        
        # Expand to [batch, seq_len, 1]
        inputs = inputs.unsqueeze(-1)
        
        # Encode
        encoder_outputs, (h, c) = self.encoder(inputs)
        
        # Decoder initial state = encoder final state
        decoder_state = (h, c)
        
        # Decoder input: zeros (or could use previous pointer)
        decoder_input = torch.zeros(batch_size, 1, self.decoder.input_size, device=inputs.device)
        
        all_pointers = []
        
        for t in range(seq_len):
            # Decode one step
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # Compute attention scores
            # decoder_output: [batch, 1, hidden_dim]
            # encoder_outputs: [batch, seq_len, hidden_dim]
            
            query = self.W_query(decoder_output)  # [batch, 1, hidden_dim]
            keys = self.W_key(encoder_outputs)     # [batch, seq_len, hidden_dim]
            
            # Additive attention
            scores = self.v(torch.tanh(query + keys.unsqueeze(1)))  # [batch, 1, seq_len, 1]
            scores = scores.squeeze(-1).squeeze(1)  # [batch, seq_len]
            
            all_pointers.append(scores)
            
            # Next decoder input: pointed encoder output
            pointer_probs = torch.softmax(scores, dim=-1)  # [batch, seq_len]
            context = torch.bmm(pointer_probs.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_dim]
            decoder_input = context
        
        # Stack all pointers
        pointers = torch.stack(all_pointers, dim=1)  # [batch, seq_len, seq_len]
        
        return pointers


def compute_loss(logits, targets):
    """
    Args:
        logits: [batch, seq_len, seq_len] - raw scores
        targets: [batch, seq_len] - target indices
    Returns:
        loss: scalar
    """
    batch_size, seq_len, _ = logits.shape
    
    # Reshape for cross-entropy
    logits_flat = logits.view(-1, seq_len)      # [batch*seq_len, seq_len]
    targets_flat = targets.view(-1)              # [batch*seq_len]
    
    # Cross-entropy loss
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    
    return loss


def compute_accuracy(logits, targets):
    """
    Compute percentage of perfectly sorted sequences.
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]
    
    # Check if entire sequence matches
    correct = (predictions == targets).all(dim=1)  # [batch]
    
    accuracy = correct.float().mean()
    
    return accuracy.item()


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_acc = 0
    
    for batch_inputs, batch_targets in tqdm(dataloader, desc="Training"):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        logits = model(batch_inputs)
        
        # Compute loss
        loss = compute_loss(logits, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for stability!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, batch_targets)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_acc


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            logits = model(batch_inputs)
            
            # Compute metrics
            loss = compute_loss(logits, batch_targets)
            acc = compute_accuracy(logits, batch_targets)
            
            total_loss += loss.item()
            total_acc += acc
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    return avg_loss, avg_acc


def test_sorting():
    """Test the complete training pipeline."""
    print("🧪 Testing Sorting Training Pipeline")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = SortingDataset(num_samples=1000, min_len=5, max_len=8)
    val_dataset = SortingDataset(num_samples=200, min_len=5, max_len=8)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = SimplePointerNetwork(input_dim=1, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test one sample
    print("\n📊 Testing on one sample:")
    sample_input = torch.tensor([[0.7, 0.2, 0.9, 0.1, 0.5]])
    print(f"Input: {sample_input[0].tolist()}")
    
    expected_order = torch.argsort(sample_input[0])
    print(f"Expected order (indices): {expected_order.tolist()}")
    print(f"Sorted values: {[sample_input[0, i].item() for i in expected_order]}")
    
    # Train for a few epochs
    print("\n🏋️ Training for 3 epochs...")
    
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/3")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
    
    # Test on sample again
    print("\n📊 After training:")
    model.eval()
    with torch.no_grad():
        logits = model(sample_input.to(device))
        predictions = torch.argmax(logits, dim=-1)[0]
    
    print(f"Predicted order: {predictions.cpu().tolist()}")
    print(f"Expected order:  {expected_order.tolist()}")
    
    if torch.equal(predictions.cpu(), expected_order):
        print("✅ Perfect match!")
    else:
        print("⚠️ Not perfect yet (needs more training)")


if __name__ == "__main__":
    test_sorting()
    
    print("\n" + "=" * 60)
    print("🎯 Solution 3 Summary")
    print("=" * 60)
    print("""
Key implementation details:

1. **Dataset Generation**
   - Random sequences of floats
   - Variable length (makes model robust)
   - Target = argsort(sequence) gives sorting indices

2. **Loss Function**
   - Cross-entropy at each decoding step
   - Treats each step as classification over input positions
   - Formula: CE(logits[t], target[t]) for each t
   
3. **Accuracy Metric**
   - Must get ENTIRE sequence correct
   - Partial credit doesn't make sense for sorting
   - This is why training can be slow initially

4. **Training Tips**
   - Gradient clipping (max_norm=1.0) is CRITICAL
   - Without it, training explodes
   - Start with small sequences (5-8)
   - Gradually increase difficulty

5. **Why It's Hard**
   - Must learn both "what to point at" and "when to point"
   - Early in training: random pointers
   - Middle training: gets some elements right
   - Late training: perfect sorting

6. **Debugging Checklist**
   ✅ Loss decreasing? (should drop from ~2.0 to <0.1)
   ✅ Accuracy improving? (0% → 80%+ on small sequences)
   ✅ Gradients flowing? (check with gradient clipping)
   ✅ Overfitting? (train acc >> val acc means yes)

Expected performance:
- Epoch 1-5: Random guessing (~0% accuracy)
- Epoch 10-20: Starting to learn (~20-40% accuracy)
- Epoch 50+: Good performance (~80%+ on len=5-8)
    """)
