"""
GPipe: Efficient Training of Giant Neural Networks with Pipeline Parallelism

This implementation follows Huang et al. (2018/2019) "GPipe: Efficient Training of 
Giant Neural Networks using Pipeline Parallelism".

The core challenge addressed is training models that are too large for a single 
accelerator's memory. We solve this by:
1. Pipeline Parallelism: Partitioning the model into K stages.
2. Batch Splitting: Slicing mini-batches into M micro-batches to fill the pipeline.
3. Synchronous Updates: Accumulating gradients across M steps for consistency.
4. Re-materialization: Trading computation for 80%+ memory savings.

Reference: https://arxiv.org/abs/1811.06965

Author: 30u30 Project
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Any, Optional
import time

# ============================================================================
# SECTION 1: PIPELINE STAGE (RE-MATERIALIZATION)
# ============================================================================
# A PipelineStage encapsulates a contiguous sub-sequence of the full model.
# The most critical feature here is Activation Checkpointing (Re-materialization).
# As described in Section 3.2 of the paper, we store only the input to the 
# stage and discard internal activations during the forward pass. This reduces
# memory from O(L) to O(L/K) per partition.
# ============================================================================

class PipelineStage(nn.Module):
    """
    Wraps a portion of the model as a pipeline stage.
    Handles activation checkpointing to trade computation for memory.
    Ref: Section 3.2 (Re-materialization)
    """
    def __init__(self, layers: nn.Sequential, stage_idx: int, use_checkpoint: bool = False):
        super().__init__()
        self.layers = layers
        self.stage_idx = stage_idx
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional re-materialization.
        Note: We use `use_reentrant=False` which is the modern PyTorch standard
        for checkpointing, avoiding some common graph-break issues.
        """
        if self.use_checkpoint and self.training:
            # We trade a ~33% increase in compute time for a dramatic reduction
            # in activation memory. This is what enables 6B+ parameter models.
            return checkpoint(self.layers, x, use_reentrant=False)
        return self.layers(x)

# ============================================================================
# SECTION 2: GPIPE WRAPPER (ORCHESTRATION)
# ============================================================================
# The GPipe class handles the logic of partitioning the model and 
# scheduling micro-batches through the stages. 
#
# A "Pipeline Bubble" occurs at the start (filling) and end (draining).
# The efficiency is M / (M + K - 1). 
# Ref: Section 3.1 (Batch Splitting and Pipelining)
# ============================================================================

class GPipe(nn.Module):
    """
    GPipe implementation: Model parallelism + Pipelining.
    
    Architecture:
    1. Model is split into K partitions based on layer count.
    2. Mini-batch is split into M micro-batches sized N/M.
    3. Pipeline executes in a synchronous schedule for identity training.
    """
    def __init__(self, model: nn.Sequential, n_partitions: int, n_microbatches: int, use_checkpoint: bool = True):
        super().__init__()
        self.n_partitions = n_partitions
        self.n_microbatches = n_microbatches
        self.use_checkpoint = use_checkpoint
        
        # Partition the model: Balanced approach
        # We assume uniform computation per layer; real-world implementations
        # might use profiling-based load balancing.
        total_layers = len(model)
        layers_per_partition = max(1, total_layers // n_partitions)
        
        self.stages = nn.ModuleList()
        for i in range(n_partitions):
            start = i * layers_per_partition
            # Ensure the last partition captures all remaining layers
            end = (i + 1) * layers_per_partition if i < n_partitions - 1 else total_layers
            
            stage_layers = model[start:end]
            self.stages.append(PipelineStage(stage_layers, i, use_checkpoint))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the Synchronous Pipeline Schedule.
        Each micro-batch goes through the stages sequentially, but overlapping
        in the pipeline schedule. This ensures hardware is rarely idle.
        """
        # Split mini-batch into M micro-batches
        # If N is not divisible by M, torch.chunk handles it but results in
        # uneven micro-batch sizes. Divisibility is preferred for stability.
        micro_batches = torch.chunk(x, self.n_microbatches, dim=0)
        
        # outputs[microbatch_idx][stage_idx] stores the data for the next step.
        # This simulates the cross-device communication buffers.
        outputs = [[None for _ in range(self.n_partitions)] for _ in range(self.n_microbatches)]
        
        # Scheduling Logic: step 0 to (M + K - 2)
        # T0: MB0 processed by S0
        # T1: MB0 processed by S1, MB1 processed by S0
        for step in range(self.n_microbatches + self.n_partitions - 1):
            for m in range(self.n_microbatches):
                k = step - m
                if 0 <= k < self.n_partitions:
                    input_tensor = micro_batches[m] if k == 0 else outputs[m][k-1]
                    # Intermediate results require gradients throughout the pipeline
                    outputs[m][k] = self.stages[k](input_tensor)
        
        # Synchronous Update: Concatenate for loss calculation
        # This matches the paper's promise of consistency with standard SGD.
        final_outputs = [outputs[m][-1] for m in range(self.n_microbatches)]
        return torch.cat(final_outputs, dim=0)

# ============================================================================
# SECTION 3: UTILS & REPORTING
# ============================================================================

def get_peak_memory():
    """Returns peak GPU memory usage in MB if available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0

def summarize_results(config: dict, stats: dict) -> str:
    """Professional results summary (Day 6 Style)."""
    line = "=" * 60
    header = "GPIPE TRAINING SUMMARY"
    
    # Calculate theoretical efficiency
    k, m = config['partitions'], config['micro_batches']
    efficiency = m / (m + k - 1)
    
    lines = [
        line,
        f"{header:^60}",
        line,
        f"CONFIG:",
        f"  Total Layers:     {config['layers']}",
        f"  Partitions (K):   {k}",
        f"  Micro-batches (M):{m}",
        f"  Checkpointing:    {'[ENABLED]' if config['use_checkpoint'] else '[DISABLED]'}",
        "",
        f"PERFORMANCE:",
        f"  Total Time:       {stats['total_time']:.4f}s",
        f"  Peak Memory:      {stats['peak_mem']:.2f} MB",
        f"  Pipeline Efficiency: {efficiency:.2%} (Theoretical)",
        line
    ]
    return "\n".join(lines)
