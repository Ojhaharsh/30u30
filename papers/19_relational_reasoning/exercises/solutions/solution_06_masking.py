import torch

def masked_aggregation(g_out: torch.Tensor) -> torch.Tensor:
    """
    Solution 6: Masked aggregation (ignoring self-relations).
    """
    batch, n, n_inner, d = g_out.shape
    
    # 1. Create Identity Mask
    mask = torch.eye(n, device=g_out.device).bool()
    # Invert mask: 0 for diagonal, 1 for others
    mask = ~mask
    
    # 2. Expand mask to match g_out shape: (N, N) -> (batch, N, N, D)
    mask = mask.view(1, n, n, 1).expand(batch, -1, -1, d)
    
    # 3. Apply mask and sum
    masked_g = g_out * mask.float()
    return masked_g.sum(dim=(1, 2))
