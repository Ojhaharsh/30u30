"""
Solution 5: Greedy Decoding Loop
Standardized for Day 18 of 30u30

Based on: Vinyals et al. (2015) - "Pointer Networks"
"""

import torch

def greedy_decode(decoder, pointer_head, encoder_outputs, batch_size, seq_len, hidden_size):
    """
    Perform greedy decoding to generate pointers.
    """
    device = encoder_outputs.device
    decoder_input = torch.zeros(batch_size, 1, hidden_size, device=device)
    decoder_hidden = None 
    
    pointers = []
    
    for _ in range(seq_len):
        # 1. Decoder step
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # 2. Pointer Attention scores
        log_probs = pointer_head(decoder_output.squeeze(1), encoder_outputs)
        
        # 3. Greedy selection
        idx = log_probs.argmax(dim=-1)
        pointers.append(idx.item())
        
        # 4. Update next input with the chosen element's representation
        # Assuming batch_size=1 for simplicity in this solution
        decoder_input = encoder_outputs[0, idx].unsqueeze(1)
        
    return torch.tensor(pointers)
