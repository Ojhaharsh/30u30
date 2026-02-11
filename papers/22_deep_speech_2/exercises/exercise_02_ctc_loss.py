"""
Exercise 2: CTC Loss and Greedy Decoding
Difficulty: Medium (3/5) | Time: 45 min

CTC (Connectionist Temporal Classification) is the loss function that
makes end-to-end speech recognition possible (Section 3.1, 3.3).
It handles the alignment between variable-length audio and variable-length
text without requiring frame-level labels.

Your job:
    1. Build a character encoder (text <-> integer indices)
    2. Implement CTC greedy decoding (collapse repeats, remove blanks)
    3. Wire up PyTorch's CTC loss for a simple test case
"""

import torch
import torch.nn.functional as F
from typing import List


class CharEncoder:
    """Maps between characters and integer indices.

    The alphabet (Section 3.1):
        - Index 0: CTC blank symbol
        - Indices 1-26: 'a' through 'z'
        - Index 27: space ' '
        - Index 28: apostrophe "'"

    Total vocab size: 29
    """

    def __init__(self):
        # TODO: Build the character-to-index and index-to-character mappings
        # blank = 0, 'a'=1, 'b'=2, ..., 'z'=26, ' '=27, "'"=28
        self.blank_idx = 0
        self.char_to_idx = {}  # YOUR CODE HERE
        self.idx_to_char = {}  # YOUR CODE HERE
        self.vocab_size = None  # YOUR CODE HERE

    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices.

        Args:
            text: Input string (will be lowercased)

        Returns:
            List of integer indices

        Example:
            encode("hi") -> [8, 9]
        """
        # TODO: implement encoding
        pass  # YOUR CODE HERE

    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text (no CTC collapsing).

        Args:
            indices: List of integer indices

        Returns:
            Decoded string
        """
        # TODO: implement decoding
        pass  # YOUR CODE HERE

    def ctc_decode(self, indices: List[int]) -> str:
        """CTC greedy decode: collapse repeated characters, remove blanks.

        The CTC collapsing rule:
            1. Remove consecutive duplicate indices
            2. Remove all blank (index 0) symbols
            3. Map remaining indices to characters

        Example:
            [0, 0, 8, 8, 0, 9, 9, 0] -> "hi"
            (blanks removed, repeated 8s collapsed to one 'h', repeated 9s to 'i')

        Args:
            indices: Raw model output (argmax at each timestep)

        Returns:
            Decoded string after CTC collapsing
        """
        # TODO: implement CTC greedy decoding
        pass  # YOUR CODE HERE


def compute_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor
) -> torch.Tensor:
    """Compute CTC loss using PyTorch's built-in function.

    Args:
        log_probs: shape (time, batch, vocab_size) — log probabilities
        targets: shape (sum(target_lengths),) — concatenated target indices
        input_lengths: shape (batch,) — length of each input sequence
        target_lengths: shape (batch,) — length of each target sequence

    Returns:
        Scalar loss value

    Hint:
        Use F.ctc_loss with blank=0, reduction='mean', zero_infinity=True
    """
    # TODO: compute CTC loss
    pass  # YOUR CODE HERE


def test_ctc():
    """Test your CTC implementation."""
    torch.manual_seed(42)

    # Test CharEncoder
    encoder = CharEncoder()
    assert encoder.vocab_size == 29, f"Expected vocab_size=29, got {encoder.vocab_size}"
    assert encoder.blank_idx == 0

    # Test encode/decode roundtrip
    text = "hello"
    encoded = encoder.encode(text)
    decoded = encoder.decode(encoded)
    assert decoded == text, f"Roundtrip failed: '{text}' -> {encoded} -> '{decoded}'"
    print(f"[OK] Encode/decode roundtrip: '{text}' -> {encoded} -> '{decoded}'")

    # Test CTC decode
    # Simulated output: blank, blank, h, h, blank, i, i, blank
    raw = [0, 0, 8, 8, 0, 9, 9, 0]
    ctc_result = encoder.ctc_decode(raw)
    assert ctc_result == "hi", f"CTC decode failed: {raw} -> '{ctc_result}', expected 'hi'"
    print(f"[OK] CTC decode: {raw} -> '{ctc_result}'")

    # Test with repeated same char: blank, a, blank, a, blank -> "aa"
    raw2 = [0, 1, 0, 1, 0]
    ctc_result2 = encoder.ctc_decode(raw2)
    assert ctc_result2 == "aa", f"CTC decode failed: {raw2} -> '{ctc_result2}', expected 'aa'"
    print(f"[OK] CTC decode (repeated chars): {raw2} -> '{ctc_result2}'")

    # Test CTC loss
    batch_size = 2
    time_steps = 10
    vocab_size = encoder.vocab_size

    log_probs = torch.randn(time_steps, batch_size, vocab_size)
    log_probs = F.log_softmax(log_probs, dim=-1)

    targets = torch.tensor([8, 9, 3, 1, 20])  # "hi" + "cat"
    input_lengths = torch.tensor([10, 10])
    target_lengths = torch.tensor([2, 3])

    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths)
    assert loss is not None, "compute_ctc_loss returned None"
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"[OK] CTC loss: {loss.item():.4f}")

    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_ctc()
