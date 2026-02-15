""" Day 26: The Incompressibility Challenge | Solution 05 | Part of 30u30 """
"""
Solution for Exercise 5
=======================
"""

import random
import string
from implementation import HuffmanCoder

def create_incompressible_string(length=100):
    # Random strings are the best way to achieve high Kolmogorov complexity
    # relative to a specific encoder.
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))

if __name__ == "__main__":
    h_coder = HuffmanCoder()
    s = create_incompressible_string(100)
    bits = h_coder.get_complexity(s)
    raw_bits = len(s) * 8
    
    print(f"String: {s[:20]}...")
    print(f"Raw bits: {raw_bits}")
    print(f"Compressed: {bits}")
    print(f"Reduction: {(1 - bits/raw_bits)*100:.2f}%")
    
    # ASCII chars have approx 5.9 bits of entropy, so 5.9/8 = 0.73
    if bits >= raw_bits * 0.70:
        print("SUCCESS: String is incompressible!")
    else:
        print("FAILURE: String was too easy to compress.")
