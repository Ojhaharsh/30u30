""" Day 26: Arithmetic Range Narrowing | Solution 04 | Part of 30u30 """
"""
Solution for Exercise 4
=======================
"""

import numpy as np
from collections import Counter

def get_probabilities(text):
    freqs = Counter(text)
    total = len(text)
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
        
        width = high - low
        high = low + width * p_high
        low = low + width * p_low
        
        print(f"{char:5} | {low:10.8f} | {high:10.8f} | {width:10.8f}")
        
    return low, high

if __name__ == "__main__":
    test_str = "KOLMOGOROV"
    l, h = arithmetic_encode_range(test_str)
    
    theoretical_bits = -np.log2(h - l)
    print(f"\nFinal Interval: [{l:.10f}, {h:.10f})")
    print(f"Theoretical Complexity: {theoretical_bits:.2f} bits")
