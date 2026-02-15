""" Day 26: Bit-rate vs Shannon Entropy | Exercise 02 | Part of 30u30 """
"""
Exercise 2: Bit-rate vs Shannon Entropy
=======================================

In this exercise, you will compare the empirical compression 
achieved by Huffman coding against the theoretical limit 
defined by Shannon Entropy.

Goal:
1. Calculate the average bits per character for different strings.
2. Verify that Bits/Char >= Entropy.
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
    
    # TODO: Calculate bits-per-character for both
    h_bpc = 0.0 # h_bits / len(text)
    s_bpc = 0.0 # s_bits / len(text)
    
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
