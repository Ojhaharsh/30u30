"""
Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
====================================================================
Amodei et al. (2015) — https://arxiv.org/abs/1512.02595

This implements the Deep Speech 2 architecture: a model that takes raw audio
spectrograms as input and outputs character sequences directly, bypassing the
traditional HMM/GMM pipeline entirely. Trained end-to-end with CTC loss.

Architecture (Section 3):
    Spectrogram -> Conv layers -> Bidirectional RNN/GRU -> FC -> Softmax -> CTC

Key components implemented:
    1. Spectrogram feature extraction (log power spectrogram, 20ms windows)
    2. Convolutional front-end (1-3 layers, 2D convolution over time+frequency)
    3. Bidirectional RNN/GRU layers with sequence-wise batch normalization
    4. Clipped ReLU activation: min(max(x, 0), 20)
    5. CTC loss and greedy/beam search decoding
    6. SortaGrad curriculum learning (Section 3.1)

References to paper sections are marked inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AudioSample:
    """A single audio sample with its transcription.

    Attributes:
        audio: Raw waveform tensor, shape (num_samples,)
        sample_rate: Audio sample rate in Hz
        transcript: Ground truth text
    """
    audio: torch.Tensor
    sample_rate: int
    transcript: str


@dataclass
class SpectrogramBatch:
    """A batch of spectrograms ready for the model.

    Attributes:
        features: Padded spectrograms, shape (batch, channels, freq, time)
        feature_lengths: Actual lengths before padding, shape (batch,)
        targets: Encoded target sequences (concatenated), shape (total_target_len,)
        target_lengths: Length of each target, shape (batch,)
        texts: Original transcription strings
    """
    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: List[str]

    def to(self, device: torch.device) -> 'SpectrogramBatch':
        return SpectrogramBatch(
            features=self.features.to(device),
            feature_lengths=self.feature_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            texts=self.texts
        )


# =============================================================================
# Character Encoding
# =============================================================================

class CharEncoder:
    """Maps between characters and integer indices.

    The alphabet follows the paper (Section 3.1):
    - English: {a-z, space, apostrophe, blank}
    - blank is index 0 (CTC convention)
    """

    def __init__(self, language: str = 'english'):
        if language == 'english':
            # Paper Section 3.1: "outputs the graphemes of a language
            # along with a blank symbol"
            chars = list("abcdefghijklmnopqrstuvwxyz '")
        elif language == 'mandarin':
            # Paper uses ~6000 characters for Mandarin (Section 3.7)
            # We use a simplified set for demonstration
            chars = list("abcdefghijklmnopqrstuvwxyz '")
        else:
            raise ValueError(f"Unsupported language: {language}")

        self.blank_idx = 0
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.idx_to_char[0] = ''  # blank maps to empty string
        self.vocab_size = len(chars) + 1  # +1 for blank

    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices."""
        text = text.lower()
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text (no CTC collapsing)."""
        return ''.join(self.idx_to_char.get(i, '') for i in indices)

    def ctc_decode(self, indices: List[int]) -> str:
        """CTC greedy decode: collapse repeats, remove blanks."""
        result = []
        prev = None
        for idx in indices:
            if idx != prev:
                if idx != self.blank_idx:
                    result.append(self.idx_to_char.get(idx, ''))
            prev = idx
        return ''.join(result)


# =============================================================================
# Spectrogram Features
# =============================================================================

def compute_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    window_ms: float = 20.0,
    stride_ms: float = 10.0,
    n_fft: int = 512
) -> torch.Tensor:
    """Compute log power spectrogram from raw audio.

    The paper uses spectrograms of power-normalized audio clips,
    calculated on 20ms windows (Section 3.1).

    Args:
        audio: Raw waveform, shape (num_samples,)
        sample_rate: Sample rate in Hz
        window_ms: Window size in milliseconds (paper: 20ms)
        stride_ms: Stride in milliseconds (paper: 10ms)
        n_fft: FFT size

    Returns:
        Log power spectrogram, shape (n_freq, n_time)
    """
    window_size = int(sample_rate * window_ms / 1000)
    stride = int(sample_rate * stride_ms / 1000)

    # win_length must be <= n_fft for torch.stft
    win_length = min(window_size, n_fft)

    # Pad audio to fit window
    if len(audio) < win_length:
        audio = F.pad(audio, (0, win_length - len(audio)))

    # Hann window for spectral analysis
    window = torch.hann_window(win_length, device=audio.device)

    # STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=stride,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    # Power spectrogram + log scaling
    power = spec.abs().pow(2)
    log_spec = torch.log(power + 1e-10)

    # Normalize per utterance (zero mean, unit variance)
    # Paper: "power normalized" (Section 3.1)
    log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-10)

    return log_spec


# =============================================================================
# Model Components
# =============================================================================

class ClippedReLU(nn.Module):
    """Clipped Rectified Linear Unit.

    Section 3.1: "the clipped rectified-linear (ReLU) activation function
    sigma(x) = min{max{x, 0}, 20}"

    Clipping at 20 prevents hidden state values from exploding,
    which is important for numerical stability in deep RNNs.
    """

    def __init__(self, clip_value: float = 20.0):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.clip_value)


class SequenceBatchNorm(nn.Module):
    """Sequence-wise Batch Normalization.

    Section 3.2: "the most effective approach we found was to apply
    batch normalization on the output activations... the mean and variance
    statistics are computed over all items in a minibatch over the
    entire length of the sequence."

    Unlike standard BatchNorm that normalizes per-timestep, this computes
    statistics across the full (batch x time) dimension. This is critical
    because speech utterances have variable length -- per-timestep stats
    would be unreliable at later timesteps where fewer sequences contribute.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, time, features) or (batch, features)
        """
        if self.training:
            # Compute mean/var over batch AND time dimensions
            if x.dim() == 3:
                # (batch, time, features) -> stats over (batch * time, features)
                flat = x.reshape(-1, self.num_features)
            else:
                flat = x

            mean = flat.mean(dim=0)
            var = flat.var(dim=0, unbiased=False)

            # Update running stats
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class ConvBlock(nn.Module):
    """2D Convolutional block for the front-end.

    Section 3.1: The model uses 1-3 convolutional layers that process
    the spectrogram in both time and frequency dimensions before
    feeding into the recurrent layers.

    Each block: Conv2D -> BatchNorm -> ClippedReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0)
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm handles the bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = ClippedReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, channels, freq, time)
        Returns:
            shape (batch, out_channels, freq', time')
        """
        return self.activation(self.bn(self.conv(x)))


class BidirectionalRNNLayer(nn.Module):
    """A single bidirectional RNN/GRU layer with sequence-wise BatchNorm.

    Section 3.1: "bidirectional recurrent layers"
    Section 3.2: "we apply batch normalization on the output activations"

    The paper found that GRUs generally train faster and are less prone
    to divergence compared to simple RNNs (Section 3.1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = 'gru',
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_batchnorm = use_batchnorm

        # Paper explores both simple RNN and GRU (Section 3.1)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size,
                batch_first=True,
                bidirectional=True
            )
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size, hidden_size,
                batch_first=True,
                bidirectional=True,
                nonlinearity='relu'
            )
        else:
            raise ValueError(f"rnn_type must be 'gru' or 'rnn', got '{rnn_type}'")

        if use_batchnorm:
            # BatchNorm on concatenated forward+backward outputs
            self.bn = SequenceBatchNorm(hidden_size * 2)

        self.activation = ClippedReLU()

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape (batch, time, features)
            lengths: actual sequence lengths, shape (batch,)
        Returns:
            shape (batch, time, hidden_size * 2)
        """
        if lengths is not None:
            # Pack padded sequences for efficient computation
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            output, _ = self.rnn(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        else:
            output, _ = self.rnn(x)

        if self.use_batchnorm:
            output = self.bn(output)

        return self.activation(output)


# =============================================================================
# Deep Speech 2 Model
# =============================================================================

class DeepSpeech2(nn.Module):
    """Deep Speech 2: End-to-End Speech Recognition.

    Architecture (Section 3):
        Input spectrogram
        -> 1-3 convolutional layers (2D over time + frequency)
        -> 1-7 bidirectional RNN/GRU layers with sequence-wise BatchNorm
        -> 1 fully connected layer
        -> Softmax over alphabet + blank

    The paper systematically evaluates depth vs. width, finding that
    "increasing depth is more effective for scaling model size with
    larger datasets" (Section 3.5).

    Args:
        n_freq: Number of frequency bins in the spectrogram
        vocab_size: Size of the output alphabet (including blank)
        n_conv: Number of convolutional layers (paper: 1-3)
        n_rnn: Number of recurrent layers (paper: 1-7)
        rnn_hidden: Hidden size per direction for RNN layers
        rnn_type: 'gru' or 'rnn' (paper finds GRU trains faster)
        use_batchnorm: Whether to use sequence-wise BatchNorm (Section 3.2)
    """

    def __init__(
        self,
        n_freq: int = 257,
        vocab_size: int = 29,
        n_conv: int = 2,
        n_rnn: int = 5,
        rnn_hidden: int = 512,
        rnn_type: str = 'gru',
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.n_conv = n_conv
        self.n_rnn = n_rnn

        # --- Convolutional front-end (Section 3.1) ---
        # The paper uses 2D convolutions over time and frequency
        conv_layers = []
        if n_conv >= 1:
            # First conv: large kernel to capture spectral patterns
            conv_layers.append(ConvBlock(
                1, 32,
                kernel_size=(41, 11),  # (freq, time)
                stride=(2, 2),
                padding=(20, 5)
            ))
        if n_conv >= 2:
            conv_layers.append(ConvBlock(
                32, 32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5)
            ))
        if n_conv >= 3:
            conv_layers.append(ConvBlock(
                32, 96,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5)
            ))
        self.conv = nn.Sequential(*conv_layers)

        # Compute the frequency dimension after convolutions
        freq_after_conv = self._compute_freq_after_conv(n_freq)
        rnn_input_size = freq_after_conv * (96 if n_conv >= 3 else 32 if n_conv >= 1 else 1)

        # --- Recurrent layers (Section 3.1) ---
        # "The model consists of several layers of bidirectional
        # recurrent layers" — 1 to 7 layers explored
        rnn_layers = []
        for i in range(n_rnn):
            input_dim = rnn_input_size if i == 0 else rnn_hidden * 2
            rnn_layers.append(BidirectionalRNNLayer(
                input_dim, rnn_hidden,
                rnn_type=rnn_type,
                use_batchnorm=use_batchnorm
            ))
        self.rnn_layers = nn.ModuleList(rnn_layers)

        # --- Fully connected + output (Section 3.1) ---
        self.fc = nn.Linear(rnn_hidden * 2, vocab_size)

    def _compute_freq_after_conv(self, n_freq: int) -> int:
        """Calculate output frequency dimension after conv layers."""
        freq = n_freq
        if self.n_conv >= 1:
            freq = (freq + 2 * 20 - 41) // 2 + 1  # stride 2
        if self.n_conv >= 2:
            freq = (freq + 2 * 10 - 21) // 2 + 1  # stride 2
        if self.n_conv >= 3:
            freq = (freq + 2 * 10 - 21) // 2 + 1  # stride 2
        return freq

    def _compute_time_after_conv(self, time_len: torch.Tensor) -> torch.Tensor:
        """Calculate output time dimension after conv layers (for CTC)."""
        lengths = time_len.clone().float()
        if self.n_conv >= 1:
            lengths = torch.floor((lengths + 2 * 5 - 11) / 2 + 1)
        # Remaining conv layers use stride 1 in time, so length is preserved
        return lengths.long()

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the full DS2 pipeline.

        Args:
            x: Spectrograms, shape (batch, 1, freq, time)
            lengths: Original time lengths, shape (batch,)

        Returns:
            log_probs: Log probabilities, shape (time, batch, vocab_size)
            output_lengths: Length of each output sequence, shape (batch,)
        """
        # Convolutional layers
        x = self.conv(x)  # (batch, channels, freq', time')

        # Reshape for RNN: merge channel and frequency dims
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, channels * freq)  # (batch, time, features)

        # Compute output lengths after convolution
        if lengths is not None:
            output_lengths = self._compute_time_after_conv(lengths)
        else:
            output_lengths = torch.full(
                (batch,), time, dtype=torch.long, device=x.device
            )

        # Recurrent layers
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, output_lengths)

        # Fully connected + log softmax
        x = self.fc(x)  # (batch, time, vocab_size)
        log_probs = F.log_softmax(x, dim=-1)

        # CTC expects (time, batch, vocab_size)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs, output_lengths


# =============================================================================
# CTC Decoding
# =============================================================================

def greedy_decode(
    log_probs: torch.Tensor,
    encoder: CharEncoder,
    lengths: Optional[torch.Tensor] = None
) -> List[str]:
    """Greedy CTC decoding: take argmax at each timestep, then collapse.

    This is the simplest decoding strategy. The paper also uses a beam
    search decoder with a language model for production use (Section 3.8),
    but greedy decoding works well for demonstration.

    Args:
        log_probs: shape (time, batch, vocab_size)
        encoder: Character encoder for index-to-char mapping
        lengths: Output sequence lengths, shape (batch,)

    Returns:
        List of decoded strings
    """
    # Argmax over vocabulary at each timestep
    argmax = log_probs.argmax(dim=-1)  # (time, batch)
    argmax = argmax.permute(1, 0)  # (batch, time)

    results = []
    for i in range(argmax.size(0)):
        seq_len = lengths[i].item() if lengths is not None else argmax.size(1)
        indices = argmax[i, :seq_len].tolist()
        text = encoder.ctc_decode(indices)
        results.append(text)

    return results


def beam_search_decode(
    log_probs: torch.Tensor,
    encoder: CharEncoder,
    beam_width: int = 10,
    lengths: Optional[torch.Tensor] = None
) -> List[str]:
    """Beam search CTC decoding.

    Section 3.8: The paper uses beam search with a language model
    for production deployment. This is a simplified version without
    the language model component.

    Args:
        log_probs: shape (time, batch, vocab_size)
        encoder: Character encoder
        beam_width: Number of beams to keep
        lengths: Output sequence lengths

    Returns:
        List of decoded strings (best beam for each sample)
    """
    batch_size = log_probs.size(1)
    results = []

    for b in range(batch_size):
        seq_len = lengths[b].item() if lengths is not None else log_probs.size(0)
        probs = log_probs[:seq_len, b, :]  # (time, vocab)

        # Each beam: (log_prob, prefix_indices)
        beams = [(0.0, [])]

        for t in range(seq_len):
            new_beams = {}

            for score, prefix in beams:
                for c in range(probs.size(-1)):
                    new_score = score + probs[t, c].item()

                    # CTC collapsing logic
                    if c == encoder.blank_idx:
                        key = tuple(prefix)
                    elif len(prefix) > 0 and prefix[-1] == c:
                        key = tuple(prefix)  # repeated char -> no new char
                    else:
                        key = tuple(prefix + [c])

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Keep top beams
            sorted_beams = sorted(
                new_beams.items(), key=lambda x: x[1], reverse=True
            )[:beam_width]
            beams = [(score, list(prefix)) for prefix, score in sorted_beams]

        # Best beam
        best_prefix = beams[0][1] if beams else []
        text = encoder.decode(best_prefix)
        results.append(text)

    return results


# =============================================================================
# Metrics
# =============================================================================

def edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein edit distance between two strings."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Word Error Rate (WER) — the standard ASR metric.

    WER = (substitutions + insertions + deletions) / total_reference_words

    The paper reports WER throughout (Tables 3-5, Section 3.5-3.6).
    """
    total_words = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += len(ref_words)
        total_errors += edit_distance(ref_words, hyp_words)

    return total_errors / max(total_words, 1)


def character_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Character Error Rate (CER) — used alongside WER for Mandarin.

    The paper uses CER for Mandarin evaluation (Section 3.7).
    """
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        total_chars += len(ref)
        total_errors += edit_distance(ref, hyp)

    return total_errors / max(total_chars, 1)


# =============================================================================
# SortaGrad
# =============================================================================

def sortagrad_sampler(
    dataset: List[AudioSample],
    epoch: int
) -> List[int]:
    """SortaGrad curriculum learning.

    Section 3.1: "we sort the training examples by length and pack
    them into minibatches of similar length utterances... In the
    first epoch, we iterate over the training set in increasing
    order of utterance length. After the first epoch, training
    reverts to random order."

    This helps with numerical stability during early training:
    short utterances produce more stable gradients.

    Args:
        dataset: List of audio samples
        epoch: Current epoch number (0-indexed)

    Returns:
        List of sample indices in the order to iterate
    """
    indices = list(range(len(dataset)))

    if epoch == 0:
        # First epoch: sort by audio length (ascending)
        indices.sort(key=lambda i: len(dataset[i].audio))
    else:
        # Subsequent epochs: random shuffle
        np.random.shuffle(indices)

    return indices


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_audio(
    text: str,
    sample_rate: int = 16000,
    duration_per_char: float = 0.1
) -> torch.Tensor:
    """Generate a synthetic audio waveform for a given text.

    This is NOT real speech synthesis. It generates a deterministic
    pattern of sine waves where each character maps to a distinct
    frequency, producing an audio signal that a model can learn to
    "decode." This lets us test the full pipeline without needing
    real speech data.

    Args:
        text: Input text
        sample_rate: Audio sample rate
        duration_per_char: Duration in seconds per character

    Returns:
        Audio waveform, shape (num_samples,)
    """
    num_samples = int(len(text) * duration_per_char * sample_rate)
    t = torch.linspace(0, len(text) * duration_per_char, num_samples)

    waveform = torch.zeros(num_samples)
    samples_per_char = num_samples // max(len(text), 1)

    for i, char in enumerate(text.lower()):
        start = i * samples_per_char
        end = min((i + 1) * samples_per_char, num_samples)

        if char == ' ':
            freq = 100.0  # low frequency for space
        elif char == "'":
            freq = 110.0
        elif 'a' <= char <= 'z':
            # Each letter gets a distinct frequency: 200-720 Hz
            freq = 200.0 + (ord(char) - ord('a')) * 20.0
        else:
            freq = 150.0

        segment_t = t[start:end]
        waveform[start:end] = torch.sin(2 * math.pi * freq * segment_t)

    # Add slight noise for realism
    noise = torch.randn_like(waveform) * 0.01
    waveform = waveform + noise

    return waveform


def generate_synthetic_dataset(
    n_samples: int = 100,
    sample_rate: int = 16000,
    min_words: int = 1,
    max_words: int = 5
) -> List[AudioSample]:
    """Generate a synthetic dataset for testing.

    Creates audio-text pairs where the audio is a deterministic
    function of the text (sine waves at character-specific frequencies).
    This lets us verify the pipeline end-to-end without real speech data.

    Args:
        n_samples: Number of samples to generate
        sample_rate: Audio sample rate
        min_words: Minimum words per utterance
        max_words: Maximum words per utterance

    Returns:
        List of AudioSample objects
    """
    # Simple word vocabulary for synthetic data
    words = [
        'the', 'a', 'is', 'in', 'it', 'of', 'to', 'and', 'on', 'at',
        'cat', 'dog', 'sun', 'run', 'big', 'red', 'hat', 'box', 'cup', 'fly',
        'go', 'up', 'do', 'my', 'no', 'so', 'we', 'he', 'or', 'if',
        'hot', 'old', 'new', 'day', 'way', 'man', 'one', 'two', 'six', 'ten'
    ]

    dataset = []
    for _ in range(n_samples):
        n_words = np.random.randint(min_words, max_words + 1)
        chosen = [words[np.random.randint(len(words))] for _ in range(n_words)]
        text = ' '.join(chosen)

        audio = generate_synthetic_audio(text, sample_rate)
        dataset.append(AudioSample(
            audio=audio,
            sample_rate=sample_rate,
            transcript=text
        ))

    return dataset


# =============================================================================
# Batching
# =============================================================================

def collate_batch(
    samples: List[AudioSample],
    encoder: CharEncoder,
    n_fft: int = 512
) -> SpectrogramBatch:
    """Collate a list of AudioSamples into a padded SpectrogramBatch.

    Computes spectrograms, pads to the longest in the batch,
    and encodes targets for CTC loss.

    Args:
        samples: List of AudioSample objects
        encoder: Character encoder
        n_fft: FFT size for spectrogram computation

    Returns:
        SpectrogramBatch ready for the model
    """
    specs = []
    texts = []
    targets = []
    target_lengths = []

    for sample in samples:
        spec = compute_spectrogram(
            sample.audio,
            sample_rate=sample.sample_rate,
            n_fft=n_fft
        )
        specs.append(spec)
        texts.append(sample.transcript)

        encoded = encoder.encode(sample.transcript)
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    # Pad spectrograms to same time length
    max_time = max(s.size(1) for s in specs)
    feature_lengths = torch.tensor([s.size(1) for s in specs], dtype=torch.long)

    padded = torch.zeros(len(specs), 1, specs[0].size(0), max_time)
    for i, spec in enumerate(specs):
        padded[i, 0, :, :spec.size(1)] = spec

    return SpectrogramBatch(
        features=padded,
        feature_lengths=feature_lengths,
        targets=torch.tensor(targets, dtype=torch.long),
        target_lengths=torch.tensor(target_lengths, dtype=torch.long),
        texts=texts
    )


# =============================================================================
# Training Utilities
# =============================================================================

def train_step(
    model: DeepSpeech2,
    batch: SpectrogramBatch,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device('cpu')
) -> float:
    """Single training step with CTC loss.

    The paper trains end-to-end with CTC (Sections 3.1, 3.3).
    Gradient clipping at 400 is used for stability (common practice
    for CTC-based models).

    Returns:
        Loss value for this step
    """
    model.train()
    batch = batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    log_probs, output_lengths = model(batch.features, batch.feature_lengths)

    # CTC loss
    # log_probs: (time, batch, vocab)
    # targets: (sum of target_lengths,)
    loss = F.ctc_loss(
        log_probs,
        batch.targets,
        output_lengths,
        batch.target_lengths,
        blank=0,
        reduction='mean',
        zero_infinity=True  # numerical safety
    )

    loss.backward()

    # Gradient clipping (common practice for CTC training stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)

    optimizer.step()

    return loss.item()


def evaluate(
    model: DeepSpeech2,
    dataset: List[AudioSample],
    encoder: CharEncoder,
    batch_size: int = 16,
    device: torch.device = torch.device('cpu'),
    n_fft: int = 512
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Returns WER, CER, and average CTC loss.
    """
    model.eval()
    all_refs = []
    all_hyps = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_samples = dataset[i:i + batch_size]
            batch = collate_batch(batch_samples, encoder, n_fft=n_fft)
            batch = batch.to(device)

            log_probs, output_lengths = model(batch.features, batch.feature_lengths)

            # CTC loss
            loss = F.ctc_loss(
                log_probs,
                batch.targets,
                output_lengths,
                batch.target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )
            total_loss += loss.item()
            n_batches += 1

            # Decode predictions
            decoded = greedy_decode(log_probs, encoder, output_lengths)
            all_refs.extend([s.transcript for s in batch_samples])
            all_hyps.extend(decoded)

    return {
        'wer': word_error_rate(all_refs, all_hyps),
        'cer': character_error_rate(all_refs, all_hyps),
        'loss': total_loss / max(n_batches, 1),
        'n_samples': len(all_refs)
    }
