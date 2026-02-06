"""
Exercise 5: Greedy Decoding Loop
Standardized for Day 18 of 30u30

Your task: Complete the autoregressive decoding loop for a Pointer Network.
You need to generate a pointer at each step, update the decoder input, and use it
for the next step.

This is the core "Write" phase of the Read-Process-Write framework.
"""

import torch
import torch.nn as nn

def greedy_decode(decoder, pointer_head, encoder_outputs, batch_size, seq_len, hidden_size):
    """
    Perform greedy decoding to generate pointers.
    
    Args:
        decoder (RNN): The decoder LSTM/GRU.
        pointer_head (Module): The Pointer Attention module.
        encoder_outputs (Tensor): (batch, seq_len, hidden_size)
    """
    device = encoder_outputs.device
    
    # Initial decoder input (zeros)
    decoder_input = torch.zeros(batch_size, 1, hidden_size, device=device)
    # Initial hidden state (assume zeros for this exercise)
    decoder_hidden = None 
    
    pointers = []
    
    # TODO: Loop for seq_len steps
    # 1. Pass decoder_input through decoder
    # 2. Get log_probs from pointer_head using decoder_output and encoder_outputs
    # 3. Pick the index with highest probability (argmax)
    # 4. Use the encoder_output at that index as the NEXT decoder_input
    # 5. Append the index to the pointers list
    
    # return torch.tensor(pointers)
    pass

if __name__ == "__main__":
    print("Exercise 5: Greedy Decoding")
    print("=" * 50)
    
    # Mock components
    batch_size, seq_len, hidden_size = 1, 3, 4
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    
    # Dummy decoder and pointer head that just returns something valid
    class MockDecoder:
        def __call__(self, x, h):
            return torch.randn(batch_size, 1, hidden_size), None
            
    class MockPointer:
        def __call__(self, d, e, mask=None):
            # Just return scores that favor the next index in line
            scores = torch.zeros(batch_size, seq_len)
            return scores
            
    try:
        results = greedy_decode(MockDecoder(), MockPointer(), encoder_outputs, batch_size, seq_len, hidden_size)
        print(f"Decoded Pointers: {results}")
        
        if results is not None and len(results) == seq_len:
            print("\n[OK] Decoding loop produced the correct number of pointers!")
        else:
            print("\n[FAIL] Decoding loop incomplete or returned wrong shape.")
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
