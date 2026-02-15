""" Day 26: The Incompressibility Challenge | Exercise 05 | Part of 30u30 """
"""
Exercise 5: The Incompressibility Challenge
===========================================

Kolmogorov's most important proof is that for any length N, 
at least one string exists that is incompressible.

Goal:
Write a function that generates a string of length 100 which 
cannot be compressed by more than 5% by any of our coders.
"""

import random
import string
from implementation import HuffmanCoder

def create_incompressible_string(length=100):
    # TODO: Try different strategies to find a string that resists compression
    # Hint: Random noise is usually incompressible.
    return "example"

if __name__ == "__main__":
    h_coder = HuffmanCoder()
    s = create_incompressible_string(100)
    bits = h_coder.get_complexity(s)
    raw_bits = len(s) * 8
    
    print(f"String: {s[:20]}...")
    print(f"Raw bits: {raw_bits}")
    print(f"Compressed: {bits}")
    print(f"Reduction: {(1 - bits/raw_bits)*100:.2f}%")
    
    # For printable chars (log2(62) approx 5.9 bits), 
    # a ratio > 0.70 is excellent for a Huffman coder.
    if bits >= raw_bits * 0.70:
        print("SUCCESS: String is incompressible!")
    else:
        print("FAILURE: String was too easy to compress.")
