"""
Exercise 4: SortaGrad Curriculum Learning
Difficulty: Easy (2/5) | Time: 20 min

SortaGrad (Section 3.1): in the first training epoch, sort examples
by utterance length (shortest first). In subsequent epochs, use
random order. This stabilizes early training because short utterances
produce more stable CTC gradients.

Your job:
    1. Implement the sortagrad_sampler function
    2. Implement a length-aware batch creator
    3. Verify that epoch 0 is sorted and epoch 1+ is random
"""

import torch
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class AudioSample:
    """Simplified audio sample for this exercise."""
    audio_length: int  # Length of audio in samples
    transcript: str


def sortagrad_sampler(dataset: List[AudioSample], epoch: int) -> List[int]:
    """Return sample indices in SortaGrad order.

    Section 3.1: "In the first epoch, we iterate over the training
    set in increasing order of utterance length. After the first
    epoch, training reverts to random order."

    Args:
        dataset: List of AudioSample objects
        epoch: Current epoch number (0-indexed)

    Returns:
        List of indices into the dataset

    Example:
        epoch=0: [2, 0, 3, 1] (sorted by audio_length, ascending)
        epoch=1: [1, 3, 0, 2] (random)
        epoch=2: [3, 0, 2, 1] (random)
    """
    # TODO: implement SortaGrad sampling
    # Epoch 0: sort indices by dataset[i].audio_length (ascending)
    # Epoch 1+: random shuffle
    pass  # YOUR CODE HERE


def create_length_sorted_batches(
    dataset: List[AudioSample],
    batch_size: int,
    epoch: int
) -> List[List[int]]:
    """Create batches of sample indices using SortaGrad order.

    Args:
        dataset: List of AudioSample objects
        batch_size: Number of samples per batch
        epoch: Current epoch number

    Returns:
        List of batches, where each batch is a list of indices

    Example with 10 samples, batch_size=3:
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    # TODO: Get indices using sortagrad_sampler, then chunk into batches
    pass  # YOUR CODE HERE


def test_sortagrad():
    """Test your SortaGrad implementation."""
    np.random.seed(42)

    # Create dataset with varying lengths
    dataset = [
        AudioSample(audio_length=5000, transcript="long sentence here"),
        AudioSample(audio_length=1000, transcript="hi"),
        AudioSample(audio_length=3000, transcript="medium text"),
        AudioSample(audio_length=500, transcript="a"),
        AudioSample(audio_length=2000, transcript="short"),
    ]

    # Test epoch 0: should be sorted by audio_length
    indices_0 = sortagrad_sampler(dataset, epoch=0)
    assert indices_0 is not None, "sortagrad_sampler returned None"
    assert len(indices_0) == len(dataset), f"Expected {len(dataset)} indices"

    lengths_0 = [dataset[i].audio_length for i in indices_0]
    assert lengths_0 == sorted(lengths_0), (
        f"Epoch 0 should be sorted by length. Got lengths: {lengths_0}"
    )
    print(f"[OK] Epoch 0 indices: {indices_0}")
    print(f"[OK] Epoch 0 lengths (sorted): {lengths_0}")

    # Test epoch 1: should be random (just check it is a valid permutation)
    indices_1 = sortagrad_sampler(dataset, epoch=1)
    assert len(indices_1) == len(dataset)
    assert set(indices_1) == set(range(len(dataset))), "Not a valid permutation"
    print(f"[OK] Epoch 1 indices (random): {indices_1}")

    # Test epoch 2: should also be random, likely different from epoch 1
    indices_2 = sortagrad_sampler(dataset, epoch=2)
    assert set(indices_2) == set(range(len(dataset)))
    print(f"[OK] Epoch 2 indices (random): {indices_2}")

    # Test batching
    batches = create_length_sorted_batches(dataset, batch_size=2, epoch=0)
    assert batches is not None, "create_length_sorted_batches returned None"
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"
    assert len(batches[0]) == 2, f"First batch should have 2 samples"
    assert len(batches[-1]) == 1, f"Last batch should have 1 sample"

    # All indices should appear exactly once
    all_indices = [i for batch in batches for i in batch]
    assert sorted(all_indices) == list(range(len(dataset))), "Missing or duplicate indices"
    print(f"[OK] Batches (epoch 0): {batches}")

    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_sortagrad()
