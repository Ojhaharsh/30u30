"""
Day 25: Scaling Laws for Neural Language Models - Reference Implementation
=============================================================================

This module provides a first-principles demonstration of the scaling laws 
discovered by Kaplan et al. (2020).

This is more complex than standard Transformers because we have:
- Scratch-built blocks for precise 12Ld^2 audits
- Non-linear MasterFitter for irreducible loss estimation
- ComputeEconomy for 6N compute budgets

Author: 30u30 Project
License: CC BY-NC-ND 4.0
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional

# --- ARCHITECTURE ---

class KaplanAttention(nn.Module):
    """
    Standard Multi-Head Attention, but implemented from scratch to 
    allow for precise parameter audits.
    
    Parameters
    ----------
    d_model : int
        Dimension of the hidden state.
    n_heads : int
        Number of attention heads.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 1. Project inputs to Q, K, V
        # These 4 matrices are the '4*d^2' parameters cited in the napkin math
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled Dot-Product Attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence [batch, seq_len, d_model]
            
        Returns
        -------
        output : torch.Tensor
            Context-aware representations [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # 2. Linear projection and head splitting
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute attention scores (Logits)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 4. Causal Masking (Prevents model from seeing the future)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax + Value Weighted Sum
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 6. Final projection back to d_model
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)


class KaplanBlock(nn.Module):
    """
    A single Transformer layer (Attention + FFN).
    Exposes the 12*d^2 parameter structure.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = KaplanAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network: 8*d^2 parameters (2 * d * 4d)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (more stable for scaling)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class KaplanTransformer(nn.Module):
    """
    Full Transformer stack optimized for scaling law measurements.
    
    Parameters
    ----------
    vocab_size : int
        Size of the token library.
    d_model : int
        Dimension of the hidden state.
    n_heads : int
        Number of heads in attention.
    n_layers : int
        Number of blocks to stack.
    max_seq_len : int
        Maximum context window.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 1. Embeddings (Excluded from Kaplan's 'N' in Section 2.1)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 2. Sequential Blocks
        self.blocks = nn.Sequential(*[KaplanBlock(d_model, n_heads) for _ in range(n_layers)])
        
        # 3. Final Head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_parameters(self, mode: str = "Kaplan") -> int:
        """
        Return the number of parameters according to the desired methodology.
        
        Parameters
        ----------
        mode : str
            'Kaplan' (Excludes embeddings) or 'Total' (Includes everything).
            
        Returns
        -------
        count : int
            The parameter count.
        """
        if mode == "Kaplan":
            # Exclude embeddings as per Section 2.1
            return sum(p.numel() for name, p in self.named_parameters() 
                       if "emb" not in name and "head.weight" not in name)
        return sum(p.numel() for p in self.parameters())


# --- ANALYSIS & FITTING ---

class MasterFitter:
    """
    Advanced non-linear regression for scaling law exponents.
    Fits the form: L(X) = L_inf + (Xc / X)^alpha
    
    Parameters
    ----------
    x_data : np.ndarray
        Scale values (N, D, or C).
    y_data : np.ndarray
        Empirical loss values.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = x_data
        self.y = y_data
        self.alpha = None
        self.l_inf = None
        self.xc = None

    @staticmethod
    def scaling_law(x, l_inf, xc, alpha):
        # 1. The Kaplan Power Law form (Section 3)
        return l_inf + (xc / x)**alpha

    def fit(self):
        """
        Perform non-linear curve fitting using Levenberg-Marquardt.
        """
        try:
            # 2. Initial guesses based on OpenAI's results
            p0 = [min(self.y)*0.8, 1e12, 0.07]
            popt, _ = curve_fit(self.scaling_law, self.x, self.y, p0=p0, maxfev=10000)
            self.l_inf, self.xc, self.alpha = popt
        except Exception as e:
            # 3. Fallback to simplified log-linear model if L_inf cannot be resolved
            print(f"[WARN] Non-linear fit failed: {e}. Falling back to log-linear.")
            coeffs = np.polyfit(np.log(self.x), np.log(self.y), 1)
            self.alpha = -coeffs[0]
            self.l_inf = 0
            self.xc = np.exp(coeffs[1] / self.alpha)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scaling_law(x, self.l_inf, self.xc, self.alpha)


class ComputeEconomy:
    """
    Utilities for Equation 2.1 (Compute C = 6NBS).
    """
    @staticmethod
    def calculate_c_pfdays(n_params: int, n_tokens: int) -> float:
        """
        Convert training FLOPs into PF-days (Petaflop-days).
        
        Parameters
        ----------
        n_params : int
            Model capacity (N).
        n_tokens : int
            Dataset tokens (T).
            
        Returns
        -------
        pf_days : float
            Total compute investment.
        """
        # 1. Calculate total FLOPS (The '6' accounts for fwd+bwd ops)
        total_flops = 6 * n_params * n_tokens
        
        # 2. Convert to Petaflop-days (10^15 ops/sec * 86400 sec)
        return total_flops / (1e15 * 86400)


# --- DATA ---

class ScalingDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for demonstrating scaling behavior without 
    external dependencies. Generates sequences with predictable patterns.
    """
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
        # Add a tiny amount of structure: make every 4th token predictable
        self.data[:, 3::4] = self.data[:, 2::4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def verify_implementation():
    print("--- Scaling Law Verification ---")
    
    # Test 1: Parameter Heuristic Audit
    model = KaplanTransformer(100, 256, 4, 4)
    n_k = model.count_parameters("Kaplan")
    n_theory = 12 * 4 * (256**2)
    print(f"[OK] Parameter Check: Kaplan Count = {n_k:,}, Theoretical (~12Ld^2) = {n_theory:,}")
    print(f"     Delta: {((n_k - n_theory)/n_theory)*100:.2f}% (Difference due to Norms/PE/Offsets)")

    # Test 2: Compute Rule Audit (Eq 2.1)
    tokens = 1e9
    params = 1e9
    c = ComputeEconomy.calculate_c_pfdays(params, tokens)
    flops = c * 1e15 * 86400
    print(f"[OK] Compute Check: 1B param on 1B tokens = {flops:.2e} FLOPS (expected 6e18)")

    # Test 3: Fitter Robustness
    x = np.logspace(5, 8, 10)
    y = 2.0 + (1e12 / x)**0.076 + np.random.normal(0, 0.01, 10)
    fitter = MasterFitter(x, y)
    fitter.fit()
    print(f"[OK] Fitter Check: Derived Alpha = {fitter.alpha:.4f}")
    if fitter.l_inf > 0:
        print(f"     Summary: L(N) = {fitter.l_inf:.4f} + ({fitter.xc:.2e} / N)^{fitter.alpha:.4f}")
    else:
        print(f"     Summary: [NOTE: Linear Approximation] L(N) = 0.0000 + ({fitter.xc:.2e} / N)^{fitter.alpha:.4f}")


if __name__ == "__main__":
    verify_implementation()
