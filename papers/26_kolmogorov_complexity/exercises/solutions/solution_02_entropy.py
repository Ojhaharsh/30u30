""" Day 26: Bit-rate vs Shannon Entropy | Solution 02 | Part of 30u30 """
"""
Solution for Exercise 2
=======================
"""

from collections import Counter
import numpy as np
from implementation import HuffmanCoder, ComplexityMetrics

def analyze_bitrate(text):
    # 1. Get Huffman complexity
    h_coder = HuffmanCoder()
    h_bits = h_coder.get_complexity(text)
    
    # 2. Get Shannon Entropy (bits per sequence)
    s_bits = ComplexityMetrics.shannon_entropy(text)
    
    # Calculate bits-per-character for both
    h_bpc = h_bits / len(text)
    s_bpc = s_bits / len(text)
    
    return h_bpc, s_bpc

if __name__ == "__main__":
    texts = {
        "Uniform": "abcdefghij" * 10,
        "Biased": "aaaaaaaaaa" + "b" * 90,
        "English": "The entropy of a message is the average level of information."
    }
    
    print(f"{'Type':10} | {'Huffman BPC':12} | {'Shannon BPC':12} | {'Diff':10}")
    print("-" * 55)
    for name, t in texts.items():
        h, s = analyze_bitrate(t)
        print(f"{name:10} | {h:12.4f} | {s:12.4f} | {h-s:10.4f}")
