"""
Solution 4: Training Pipeline
==============================

Complete solution for Exercise 4.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size):
    """Create mask for causal (autoregressive) attention."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1) == 0
    return mask


class Batch:
    """
    Holds a batch of data with masks for training.
    """
    
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        
        if tgt is not None:
            # Decoder input: all tokens except last
            self.tgt = tgt[:, :-1]
            # Target for loss: all tokens except first
            self.tgt_y = tgt[:, 1:]
            # Combined mask
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # Count non-pad tokens
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create mask to hide padding AND future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class LabelSmoothing(nn.Module):
    """
    Label Smoothing Loss.
    """
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        """Compute label smoothing loss."""
        assert x.size(1) == self.size
        
        # Create smoothed target distribution
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        # Zero out rows for padding targets
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def rate(step, d_model, factor=1.0, warmup=4000):
    """
    Noam learning rate schedule.
    
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    if step == 0:
        step = 1
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class NoamOpt:
    """
    Optimizer wrapper with Noam learning rate schedule.
    """
    
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        """Update parameters and learning rate."""
        self._step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._rate = lr
        self.optimizer.step()
    
    def rate(self, step=None):
        """Compute current learning rate."""
        if step is None:
            step = self._step
        return rate(step, self.model_size, self.factor, self.warmup)
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class SimpleLossCompute:
    """Simple loss computation and training step wrapper."""
    
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        # Generate log probabilities
        x = self.generator(x)
        
        # Compute loss
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1)
        ) / norm
        
        # If training, backprop and step
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        
        return loss.data.item() * norm


# =============================================================================
# TESTS
# =============================================================================

def test_batch():
    """Test Batch class."""
    print("Testing Batch...")
    
    src = torch.LongTensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
    tgt = torch.LongTensor([[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]])
    
    batch = Batch(src, tgt, pad=0)
    
    assert batch.src_mask.shape == (2, 1, 5)
    assert batch.src_mask[0, 0, 3] == False
    assert batch.src_mask[1, 0, 3] == True
    
    assert batch.tgt.shape == (2, 4)
    assert batch.tgt_y.shape == (2, 4)
    
    assert batch.ntokens > 0
    
    print(f"✓ Source mask shape: {batch.src_mask.shape}")
    print(f"✓ Target input shape: {batch.tgt.shape}")
    print(f"✓ Target output shape: {batch.tgt_y.shape}")
    print(f"✓ Non-pad tokens: {batch.ntokens}")
    print("✓ Batch test passed!")
    return True


def test_label_smoothing():
    """Test LabelSmoothing loss."""
    print("\nTesting LabelSmoothing...")
    
    vocab_size = 10
    ls = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    
    preds = torch.randn(5, vocab_size)
    preds = F.log_softmax(preds, dim=-1)
    
    targets = torch.LongTensor([1, 2, 3, 0, 1])
    
    loss = ls(preds, targets)
    
    assert loss >= 0
    assert ls.true_dist is not None
    assert ls.true_dist.shape == (5, vocab_size)
    assert ls.true_dist[3].sum() == 0
    assert ls.true_dist[0, 1] == 0.9
    
    print(f"✓ Loss: {loss:.4f}")
    print("✓ Label smoothing test passed!")
    return True


def test_rate_schedule():
    """Test Noam learning rate schedule."""
    print("\nTesting Noam LR schedule...")
    
    d_model = 512
    warmup = 4000
    
    rates = []
    for step in [1, 1000, 2000, 4000, 8000, 16000]:
        r = rate(step, d_model, warmup=warmup)
        rates.append(r)
        
    assert rates[0] < rates[1] < rates[2] < rates[3]
    assert rates[3] > rates[4] > rates[5]
    assert rates[3] == max(rates)
    
    print("✓ LR increases during warmup")
    print("✓ LR peaks at warmup step")
    print("✓ LR decreases after warmup")
    print("✓ Rate schedule test passed!")
    return True


def test_noam_opt():
    """Test NoamOpt wrapper."""
    print("\nTesting NoamOpt...")
    
    model = nn.Linear(64, 64)
    base_opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98))
    opt = NoamOpt(model_size=64, factor=1, warmup=100, optimizer=base_opt)
    
    rates = []
    for i in range(200):
        x = torch.randn(2, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        rates.append(opt._rate)
    
    assert rates[0] > 0
    assert rates[99] > rates[0]
    assert rates[199] < rates[99]
    
    print(f"✓ Initial LR: {rates[0]:.6f}")
    print(f"✓ Peak LR (step 100): {rates[99]:.6f}")
    print(f"✓ Final LR (step 200): {rates[199]:.6f}")
    print("✓ NoamOpt test passed!")
    return True


def test_simple_loss_compute():
    """Test SimpleLossCompute."""
    print("\nTesting SimpleLossCompute...")
    
    vocab_size = 100
    d_model = 64
    
    generator = nn.Sequential(
        nn.Linear(d_model, vocab_size),
        nn.LogSoftmax(dim=-1)
    )
    
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    loss_compute = SimpleLossCompute(generator, criterion, opt=None)
    
    x = torch.randn(2, 10, d_model)
    y = torch.randint(1, vocab_size, (2, 10))
    
    loss = loss_compute(x, y, norm=20)
    
    assert loss >= 0
    
    print(f"✓ Loss: {loss:.4f}")
    print("✓ SimpleLossCompute test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SOLUTION 4: TRAINING PIPELINE")
    print("=" * 60)
    
    test_batch()
    test_label_smoothing()
    test_rate_schedule()
    test_noam_opt()
    test_simple_loss_compute()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
