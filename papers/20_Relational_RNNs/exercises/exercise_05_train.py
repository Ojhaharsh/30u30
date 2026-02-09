import torch
import torch.nn as nn
import torch.optim as optim
from implementation import RelationalRNN
from exercise_04_nth_farthest import generate_nth_farthest_batch

def train():
    # Hyperparameters
    seq_len = 10
    dim = 16
    slots = 4
    mem_size = 32
    heads = 4
    epochs = 500
    batch_size = 32
    
    # TODO: Initialize the RelationalRNN model
    # Use the hyperparameters above. Output size == dim (regression)
    model = # YOUR CODE HERE
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Starting Training...")
    
    for step in range(epochs):
        # TODO: Generate a batch of data
        inputs, targets = # YOUR CODE HERE
        
        optimizer.zero_grad()
        
        # TODO: Forward pass
        output = # YOUR CODE HERE
        
        # TODO: Calculate loss and backprop
        loss = # YOUR CODE HERE
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    print("\nTraining Complete!")
    
    # Verification
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_nth_farthest_batch(1, seq_len, dim)
        output = model(inputs)
        print(f"Final Loss: {criterion(output, targets).item():.6f}")

if __name__ == "__main__":
    train()
