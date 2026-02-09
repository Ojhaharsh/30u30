import torch

def generate_nth_farthest_batch(batch_size, seq_len, vector_dim, n_index=0):
    inputs = torch.rand(batch_size, seq_len, vector_dim) * 2 - 1
    targets = []
    
    for i in range(batch_size):
        seq = inputs[i]
        ref_vec = seq[n_index]
        
        # Calculate distances
        dists = torch.norm(seq - ref_vec, dim=1)
        
        # Find farthest
        farthest_idx = torch.argmax(dists)
        target_vec = seq[farthest_idx]
        targets.append(target_vec)
        
    targets = torch.stack(targets)
    return inputs, targets

def test_generator():
    print("Testing Data Generator...")
    B, L, D = 2, 5, 2
    inputs, targets = generate_nth_farthest_batch(B, L, D)
    
    assert inputs.shape == (B, L, D)
    assert targets.shape == (B, D)
    
    seq = inputs[0]
    ref = seq[0]
    dists = torch.norm(seq - ref, dim=1)
    max_idx = torch.argmax(dists)
    expected = seq[max_idx]
    
    assert torch.allclose(targets[0], expected), "Target verification failed!"
    print("Test Passed!")

if __name__ == "__main__":
    test_generator()
