"""
NTM Training Script: The Copy Task
==================================

This script implements the "Copy Task" defined in section 4.1 of 
Graves et al. (2014). It serves as the primary benchmark for verifying 
that the NTM has learned a simple algorithmic procedure.

Tasks:
1. Input a sequence of random 8-bit vectors.
2. Input a unique delimiter.
3. Observe the NTM outputting the exact same sequence in order.

A standard LSTM often fails this task on long sequences because it tries 
to compress the whole history into a fixed-size vector. The NTM solves it 
by using its memory like a tape (Section 4.1).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from implementation import NTM

def generate_copy_instance(batch_size, seq_len, width):
    """
    Generates data for a single training step.
    
    Format (Section 4.1):
    - Input phase: [Data bits] followed by [1 at index Width+1]
    - Output phase: Model must output [Data bits] given zero input.
    """
    # Create random bit vectors {0, 1}^width
    data = torch.randn(batch_size, seq_len, width).sign().add(1).div(2)
    
    # Input sequence: [data] + [delimiter] + [zeros for output phase]
    # Total length: L (input) + 1 (delim) + L (output)
    full_len = 2 * seq_len + 1
    inputs = torch.zeros(batch_size, full_len, width + 1)
    targets = torch.zeros(batch_size, full_len, width)
    
    inputs[:, :seq_len, :width] = data
    inputs[:, seq_len, width] = 1.0 # Delimiter bit
    
    # Target is the data during the second L timesteps
    targets[:, seq_len+1:, :] = data
    
    return inputs.transpose(0, 1), targets.transpose(0, 1)

def train():
    # Experimental Hyperparameters (Section 4)
    batch_size = 32
    max_seq_len = 20 # Can be increased to test generalization (Section 4.1)
    width = 8
    controller_size = 100
    N, M = 128, 20
    
    model = NTM(width + 1, width, controller_size, N, M)
    
    # Section 4: Graves recommends RMSProp for NTM stability.
    # Learning rate 1e-4 is a common stable starting point.
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9, alpha=0.95)
    
    # Binary Cross Entropy is ideal for bit-copy tasks.
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting Copy Task training (N={N}, M={M}, Controller={controller_size})...")
    
    for i in range(10001):
        optimizer.zero_grad()
        
        # Vary sequence length to force the model to learn the 'rule' of copying 
        # instead of a fixed-length pattern (Section 4.1).
        cur_len = random.randint(1, max_seq_len)
        inputs, targets = generate_copy_instance(batch_size, cur_len, width)
        
        state = model.reset(batch_size)
        out_list = []
        
        # Sequentially process the entire task sequence
        for t in range(inputs.size(0)):
            out, state = model(inputs[t], state)
            out_list.append(out)
            
        out_seq = torch.stack(out_list)
        
        # Only compute loss on the recall phase
        loss = criterion(out_seq[cur_len+1:], targets[cur_len+1:])
        loss.backward()
        
        # CRITICAL STABILITY (Section 4):
        # Gradient clipping prevents the circular shift gradients from exploding.
        # Without this, the model will likely produce NaNs within 100 iterations.
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        
        if i % 100 == 0:
            with torch.no_grad():
                # Bit Error Rate (BER): % of bits that are wrong
                preds = (out_seq[cur_len+1:] > 0).float()
                error = torch.abs(preds - targets[cur_len+1:]).mean().item()
            print(f"Iter {i:5d} | Len {cur_len:2d} | Loss {loss.item():.4f} | BER {error*100:.2f}%")

        if i % 1000 == 0 and i > 0:
            # Convergence check: Usually < 1% BER is considered 'solved'.
            pass

    # Save weights for visualization
    torch.save(model.state_dict(), "papers/17_neural_turing_machines/ntm_copy_final.pth")
    print("Training complete. Weights saved to papers/17_neural_turing_machines/ntm_copy_final.pth")

if __name__ == "__main__":
    train()
