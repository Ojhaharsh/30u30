""" Day 26: Arithmetic Range Narrowing | Exercise 04 | Part of 30u30 """
"""
Exercise 4: Arithmetic Range Narrowing
======================================

Arithmetic coding is superior to Huffman because it doesn't 
assign a whole number of bits to each character. It encodes 
the entire message into an interval on the number line.

Goal:
Implement the `low` and `high` range update logic.
"""

import numpy as np
from collections import Counter

def get_probabilities(text):
    freqs = Counter(test_str)
    total = len(test_str)
    probs = {}
    cumulative = 0.0
    for char in sorted(freqs.keys()):
        p = freqs[char] / total
        probs[char] = (cumulative, cumulative + p)
        cumulative += p
    return probs

def arithmetic_encode_range(text):
    probs = get_probabilities(text)
    
    low = 0.0
    high = 1.0
    
    print(f"{'Char':5} | {'Low':10} | {'High':10} | {'Range':10}")
    print("-" * 50)
    
    for char in text:
        p_low, p_high = probs[char]
        
        # TODO: Calculate current range width
        # TODO: Update high and low based on p_high and p_low of current character
        
        width = high - low
        print(f"{char:5} | {low:10.8f} | {high:10.8f} | {width:10.8f}")
        
    return low, high

if __name__ == "__main__":
    test_str = "KOLMOGOROV"
    l, h = arithmetic_encode_range(test_str)
    
    theoretical_bits = -np.log2(h - l)
    print(f"\nFinal Interval: [{l:.10f}, {h:.10f})")
    print(f"Theoretical Complexity: {theoretical_bits:.2f} bits")
