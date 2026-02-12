"""
setup.py - Verification Suite for Day 24 (GPipe)

This script performs unit tests to ensure that GPipe maintains 
mathematical equivalence to standard sequential execution while 
providing the infrastructure for pipeline parallelism.
"""

import torch
import torch.nn as nn
import unittest
from implementation import GPipe, get_peak_memory

class TestGPipe(unittest.TestCase):
    def setUp(self):
        # Create a simple deep model
        self.in_dim = 16
        self.hidden_dim = 64
        self.out_dim = 1
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.batch_size = 16
        self.x = torch.randn(self.batch_size, self.in_dim)
        self.y = torch.randn(self.batch_size, self.out_dim)

    def test_output_equivalence(self):
        """Verify GPipe output matches Sequential output exactly."""
        with torch.no_grad():
            expected = self.model(self.x)
            
            # GPipe with 2 partitions, 4 micro-batches
            gpipe_model = GPipe(self.model, n_partitions=2, n_microbatches=4, use_checkpoint=False)
            actual = gpipe_model(self.x)
            
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
            print("[OK] Output equivalence verified.")

    def test_gradient_equivalence(self):
        """Verify gradients accumulated by GPipe match Sequential gradients."""
        # 1. Standard Gradient
        model_seq = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        # Deep copy to ensure identical weights
        import copy
        model_gpipe_inner = copy.deepcopy(model_seq)
        
        x = torch.randn(4, 2)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        # Seq Pass
        out_seq = model_seq(x)
        loss_seq = criterion(out_seq, y)
        loss_seq.backward()
        
        # GPipe Pass
        gpipe = GPipe(model_gpipe_inner, n_partitions=2, n_microbatches=2, use_checkpoint=False)
        out_gpipe = gpipe(x)
        loss_gpipe = criterion(out_gpipe, y)
        loss_gpipe.backward()
        
        # Check gradients of the first layer
        grad_seq = model_seq[0].weight.grad
        grad_gpipe = gpipe.stages[0].layers[0].weight.grad
        
        self.assertTrue(torch.allclose(grad_seq, grad_gpipe, atol=1e-6))
        print("[OK] Gradient equivalence verified.")

    def test_checkpointing_impact(self):
        """Verify activation checkpointing reduces memory footprint on a larger model."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, cannot measure memory impact reliably.")
            
        # Large model to see difference
        large_model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(20)])
        x_huge = torch.randn(128, 1024).cuda()
        
        # Pass 1: No Checkpointing
        torch.cuda.reset_peak_memory_stats()
        gpipe_no_cp = GPipe(large_model, n_partitions=4, n_microbatches=4, use_checkpoint=False).cuda()
        out = gpipe_no_cp(x_huge)
        loss = out.sum()
        loss.backward()
        mem_no_cp = get_peak_memory()
        
        # Pass 2: With Checkpointing
        torch.cuda.reset_peak_memory_stats()
        gpipe_cp = GPipe(large_model, n_partitions=4, n_microbatches=4, use_checkpoint=True).cuda()
        out = gpipe_cp(x_huge)
        loss = out.sum()
        loss.backward()
        mem_cp = get_peak_memory()
        
        print(f"Memory (No Checkpoint): {mem_no_cp:.2f} MB")
        print(f"Memory (With Checkpoint): {mem_cp:.2f} MB")
        self.assertLess(mem_cp, mem_no_cp)
        print("[OK] Checkpointing memory reduction verified.")

if __name__ == "__main__":
    unittest.main()
