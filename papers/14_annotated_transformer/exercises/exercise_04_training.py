"""
Exercise 4: Training Pipeline
==============================

Build the complete training infrastructure for the Transformer.

Training a Transformer requires careful handling of:
1. Batch creation with proper masking
2. Label smoothing for better generalization
3. Special learning rate schedule (warmup + decay)
4. Loss computation and optimization

Your tasks:
1. Implement the Batch class
2. Implement LabelSmoothing loss
3. Implement the Noam learning rate schedule
4. Put it together in a training loop
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
    
    Handles:
    - Source padding mask (hide <pad> tokens)
    - Target shift (decoder input vs. target)
    - Target causal mask (can't see future + hide padding)
    
    Args:
        src: Source tokens (batch, src_seq)
        tgt: Target tokens (batch, tgt_seq) - optional
        pad: Padding token index (default 0)
    
    Attributes:
        src: Source tokens
        src_mask: Source padding mask (batch, 1, src_seq)
        tgt: Decoder input = target[:-1] (batch, tgt_seq-1)
        tgt_y: Target for loss = target[1:] (batch, tgt_seq-1)
        tgt_mask: Combined padding + causal mask
        ntokens: Number of non-pad tokens for loss normalization
    """
    
    def __init__(self, src, tgt=None, pad=0):
        # TODO: Store source and create source mask
        # self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        
        # TODO: If target provided, process it
        # if tgt is not None:
        #     # Decoder input: all tokens except last
        #     self.tgt = tgt[:, :-1]
        #     # Target for loss: all tokens except first
        #     self.tgt_y = tgt[:, 1:]
        #     # Combined mask
        #     self.tgt_mask = self.make_std_mask(self.tgt, pad)
        #     # Count non-pad tokens
        #     self.ntokens = (self.tgt_y != pad).data.sum()
        
        raise NotImplementedError("Implement Batch.__init__")
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create mask to hide padding AND future words.
        
        Combines:
        1. Padding mask: where tgt != pad
        2. Subsequent mask: can only attend to positions <= current
        
        Args:
            tgt: Target tokens (batch, tgt_seq)
            pad: Padding token index
        
        Returns:
            Combined mask (batch, tgt_seq, tgt_seq)
        """
        # TODO: Create combined mask
        # tgt_mask = (tgt != pad).unsqueeze(-2)
        # tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # return tgt_mask
        
        raise NotImplementedError("Implement Batch.make_std_mask")


class LabelSmoothing(nn.Module):
    """
    Label Smoothing Loss.
    
    Instead of one-hot targets, spreads some probability mass to other tokens.
    This prevents overconfidence and improves generalization.
    
    Standard cross-entropy: target = [0, 0, 1, 0, 0] (100% confident)
    Label smoothing (ε=0.1): target = [0.025, 0.025, 0.9, 0.025, 0.025]
    
    Args:
        size: Vocabulary size
        padding_idx: Index of padding token (excluded from smoothing)
        smoothing: Amount of probability to redistribute (default 0.0)
    
    The loss is computed using KL divergence.
    """
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        # TODO: Initialize
        # self.criterion = nn.KLDivLoss(reduction="sum")
        # self.padding_idx = padding_idx
        # self.confidence = 1.0 - smoothing
        # self.smoothing = smoothing
        # self.size = size
        # self.true_dist = None
        
        raise NotImplementedError("Implement LabelSmoothing.__init__")
    
    def forward(self, x, target):
        """
        Compute label smoothing loss.
        
        Args:
            x: Model log probabilities (batch * seq, vocab_size)
            target: Target indices (batch * seq,)
        
        Returns:
            KL divergence loss
        
        Steps:
        1. Create smoothed distribution
        2. Fill with smoothing / (size - 2)  (excluding padding and true)
        3. Set true class to confidence value
        4. Zero out padding positions
        5. Compute KL divergence
        """
        # TODO: Implement label smoothing
        # assert x.size(1) == self.size
        #
        # # Create smoothed target distribution
        # true_dist = x.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        #
        # # Zero out rows for padding targets
        # mask = torch.nonzero(target.data == self.padding_idx)
        # if mask.dim() > 0:
        #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
        #
        # self.true_dist = true_dist
        # return self.criterion(x, true_dist.clone().detach())
        
        raise NotImplementedError("Implement LabelSmoothing.forward")


def rate(step, d_model, factor=1.0, warmup=4000):
    """
    Noam learning rate schedule.
    
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    
    This schedule:
    - Increases linearly during warmup
    - Decays proportionally to 1/sqrt(step) after warmup
    
    Args:
        step: Current training step
        d_model: Model dimension
        factor: Scaling factor (default 1.0)
        warmup: Number of warmup steps (default 4000)
    
    Returns:
        Learning rate for this step
    """
    # TODO: Implement Noam schedule
    # if step == 0:
    #     step = 1
    # return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
    
    raise NotImplementedError("Implement rate")


class NoamOpt:
    """
    Optimizer wrapper with Noam learning rate schedule.
    
    Wraps a standard optimizer (usually Adam) and updates the
    learning rate according to the Noam schedule before each step.
    
    Args:
        model_size: d_model for the schedule
        factor: Scaling factor
        warmup: Number of warmup steps
        optimizer: Base optimizer (e.g., Adam)
    """
    
    def __init__(self, model_size, factor, warmup, optimizer):
        # TODO: Initialize
        # self.optimizer = optimizer
        # self._step = 0
        # self.warmup = warmup
        # self.factor = factor
        # self.model_size = model_size
        # self._rate = 0
        
        raise NotImplementedError("Implement NoamOpt.__init__")
    
    def step(self):
        """Update parameters and learning rate."""
        # TODO: Increment step, compute new LR, update optimizer, step
        # self._step += 1
        # rate = self.rate()
        # for p in self.optimizer.param_groups:
        #     p['lr'] = rate
        # self._rate = rate
        # self.optimizer.step()
        
        raise NotImplementedError("Implement NoamOpt.step")
    
    def rate(self, step=None):
        """Compute current learning rate."""
        # TODO: Compute rate for given step (or current step)
        # if step is None:
        #     step = self._step
        # return rate(step, self.model_size, self.factor, self.warmup)
        
        raise NotImplementedError("Implement NoamOpt.rate")
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class SimpleLossCompute:
    """
    Simple loss computation and training step wrapper.
    
    Args:
        generator: Output projection layer (linear + log_softmax)
        criterion: Loss criterion (LabelSmoothing)
        opt: Optimizer (NoamOpt) - None for evaluation
    """
    
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        """
        Compute loss and optionally update.
        
        Args:
            x: Decoder output (batch, seq, d_model)
            y: Target tokens (batch, seq)
            norm: Normalization factor (usually ntokens)
        
        Returns:
            Loss value (raw, not normalized)
        """
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
    
    # Create sample data
    src = torch.LongTensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
    tgt = torch.LongTensor([[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]])
    
    batch = Batch(src, tgt, pad=0)
    
    # Check source mask
    assert batch.src_mask.shape == (2, 1, 5), \
        f"Wrong src_mask shape: {batch.src_mask.shape}"
    assert batch.src_mask[0, 0, 3] == False, "Padding should be masked"
    assert batch.src_mask[1, 0, 3] == True, "Non-padding should not be masked"
    
    # Check target shift
    assert batch.tgt.shape == (2, 4), f"Wrong tgt shape: {batch.tgt.shape}"
    assert batch.tgt_y.shape == (2, 4), f"Wrong tgt_y shape: {batch.tgt_y.shape}"
    
    # Check ntokens
    assert batch.ntokens > 0, "ntokens should be positive"
    
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
    
    # Create fake predictions (log probs)
    preds = torch.randn(5, vocab_size)
    preds = F.log_softmax(preds, dim=-1)
    
    # Create targets
    targets = torch.LongTensor([1, 2, 3, 0, 1])  # 0 is padding
    
    loss = ls(preds, targets)
    
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    assert ls.true_dist is not None, "true_dist not stored"
    
    # Check smoothing distribution
    assert ls.true_dist.shape == (5, vocab_size)
    
    # Check padding row is all zeros
    assert ls.true_dist[3].sum() == 0, "Padding row should be zero"
    
    # Check confidence on true class
    assert ls.true_dist[0, 1] == 0.9, f"Confidence wrong: {ls.true_dist[0, 1]}"
    
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
        
    # LR should increase during warmup
    assert rates[0] < rates[1] < rates[2] < rates[3], \
        "LR should increase during warmup"
    
    # LR should decrease after warmup
    assert rates[3] > rates[4] > rates[5], \
        "LR should decrease after warmup"
    
    # Peak should be at warmup
    assert rates[3] == max(rates), "Peak should be at warmup step"
    
    print("✓ LR increases during warmup")
    print("✓ LR peaks at warmup step")
    print("✓ LR decreases after warmup")
    print("✓ Rate schedule test passed!")
    return True


def test_noam_opt():
    """Test NoamOpt wrapper."""
    print("\nTesting NoamOpt...")
    
    # Create simple model
    model = nn.Linear(64, 64)
    base_opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98))
    opt = NoamOpt(model_size=64, factor=1, warmup=100, optimizer=base_opt)
    
    # Simulate training steps
    rates = []
    for i in range(200):
        x = torch.randn(2, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        rates.append(opt._rate)
    
    # Check rate changes correctly
    assert rates[0] > 0, "Initial rate should be positive"
    assert rates[99] > rates[0], "Rate should increase during warmup"
    assert rates[199] < rates[99], "Rate should decrease after warmup"
    
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
    
    # Simple generator
    generator = nn.Sequential(
        nn.Linear(d_model, vocab_size),
        nn.LogSoftmax(dim=-1)
    )
    
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    loss_compute = SimpleLossCompute(generator, criterion, opt=None)
    
    # Test forward
    x = torch.randn(2, 10, d_model)
    y = torch.randint(1, vocab_size, (2, 10))
    
    loss = loss_compute(x, y, norm=20)
    
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    
    print(f"✓ Loss: {loss:.4f}")
    print("✓ SimpleLossCompute test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EXERCISE 4: TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        test_batch()
        test_label_smoothing()
        test_rate_schedule()
        test_noam_opt()
        test_simple_loss_compute()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext: Exercise 5 - Inference and Decoding")
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("Implement the TODO sections and run again!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    run_all_tests()
