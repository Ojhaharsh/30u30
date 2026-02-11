"""
Solution 4: SortaGrad Curriculum Learning
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class AudioSample:
    """Simplified audio sample for this exercise."""
    audio_length: int
    transcript: str


def sortagrad_sampler(dataset: List[AudioSample], epoch: int) -> List[int]:
    """Return sample indices in SortaGrad order."""
    indices = list(range(len(dataset)))

    if epoch == 0:
        # First epoch: sort by audio length (ascending)
        indices.sort(key=lambda i: dataset[i].audio_length)
    else:
        # Subsequent epochs: random shuffle
        np.random.shuffle(indices)

    return indices


def create_length_sorted_batches(
    dataset: List[AudioSample],
    batch_size: int,
    epoch: int
) -> List[List[int]]:
    """Create batches using SortaGrad order."""
    indices = sortagrad_sampler(dataset, epoch)

    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(indices[i:i + batch_size])

    return batches


def test_sortagrad():
    """Test the SortaGrad implementation."""
    np.random.seed(42)

    dataset = [
        AudioSample(audio_length=5000, transcript="long sentence here"),
        AudioSample(audio_length=1000, transcript="hi"),
        AudioSample(audio_length=3000, transcript="medium text"),
        AudioSample(audio_length=500, transcript="a"),
        AudioSample(audio_length=2000, transcript="short"),
    ]

    # Epoch 0: sorted
    indices_0 = sortagrad_sampler(dataset, epoch=0)
    lengths_0 = [dataset[i].audio_length for i in indices_0]
    assert lengths_0 == sorted(lengths_0)
    print(f"[OK] Epoch 0: {indices_0} (lengths: {lengths_0})")

    # Epoch 1: random
    indices_1 = sortagrad_sampler(dataset, epoch=1)
    assert set(indices_1) == set(range(5))
    print(f"[OK] Epoch 1: {indices_1}")

    # Batches
    batches = create_length_sorted_batches(dataset, batch_size=2, epoch=0)
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[-1]) == 1
    all_indices = [i for b in batches for i in b]
    assert sorted(all_indices) == list(range(5))
    print(f"[OK] Batches: {batches}")
    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_sortagrad()
