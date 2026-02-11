"""
Solution 2: CTC Loss and Greedy Decoding
"""

import torch
import torch.nn.functional as F
from typing import List


class CharEncoder:
    """Maps between characters and integer indices."""

    def __init__(self):
        self.blank_idx = 0
        chars = list("abcdefghijklmnopqrstuvwxyz '")
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.idx_to_char[0] = ''  # blank
        self.vocab_size = len(chars) + 1  # 28 chars + blank = 29

    def encode(self, text: str) -> List[int]:
        text = text.lower()
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char.get(i, '') for i in indices)

    def ctc_decode(self, indices: List[int]) -> str:
        """CTC greedy decode: collapse repeated characters, remove blanks."""
        result = []
        prev = None
        for idx in indices:
            if idx != prev:
                if idx != self.blank_idx:
                    result.append(self.idx_to_char.get(idx, ''))
            prev = idx
        return ''.join(result)


def compute_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor
) -> torch.Tensor:
    """Compute CTC loss."""
    return F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=0, reduction='mean', zero_infinity=True
    )


def test_ctc():
    """Test the CTC implementation."""
    torch.manual_seed(42)

    encoder = CharEncoder()
    assert encoder.vocab_size == 29

    # Roundtrip
    text = "hello"
    encoded = encoder.encode(text)
    decoded = encoder.decode(encoded)
    assert decoded == text
    print(f"[OK] Roundtrip: '{text}' -> {encoded} -> '{decoded}'")

    # CTC decode
    raw = [0, 0, 8, 8, 0, 9, 9, 0]
    assert encoder.ctc_decode(raw) == "hi"
    print(f"[OK] CTC decode: {raw} -> '{encoder.ctc_decode(raw)}'")

    raw2 = [0, 1, 0, 1, 0]
    assert encoder.ctc_decode(raw2) == "aa"
    print(f"[OK] CTC decode: {raw2} -> '{encoder.ctc_decode(raw2)}'")

    # CTC loss
    log_probs = F.log_softmax(torch.randn(10, 2, 29), dim=-1)
    targets = torch.tensor([8, 9, 3, 1, 20])
    input_lengths = torch.tensor([10, 10])
    target_lengths = torch.tensor([2, 3])

    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths)
    assert loss.dim() == 0
    assert loss.item() > 0
    print(f"[OK] CTC loss: {loss.item():.4f}")
    print(f"[OK] All tests passed.")


if __name__ == '__main__':
    test_ctc()
