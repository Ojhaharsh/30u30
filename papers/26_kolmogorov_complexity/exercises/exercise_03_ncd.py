""" Day 26: Similarity Check (NCD) | Exercise 03 | Part of 30u30 """
"""
Exercise 3: Similarity Check (NCD)
==================================

Using Normalized Compression Distance to identify patterns.

Goal:
Verify that NCD behaves like a distance metric.
- NCD(x, x) should be close to 0.
- NCD(x, random) should be close to 1.
"""

from implementation import HuffmanCoder, ComplexityMetrics
import random
import string

def test_ncd():
    h_coder = HuffmanCoder()
    compressor = lambda x: h_coder.get_complexity(x)
    
    s_base = "Machine learning is the study of computer algorithms that improve through experience."
    s_same = s_base[:]
    s_diff = "I like green eggs and ham, I do not like them Sam I am."
    s_rand = "".join(random.choices(string.printable, k=len(s_base)))
    
    # TODO: Calculate NCD for the combinations
    # d_identity = ComplexityMetrics.ncd(s_base, s_same, compressor)
    # d_different = ...
    # d_random = ...
    
    print(f"Distance (Identity):  {0.0:.4f}") # TODO: Print actual
    print(f"Distance (Different): {0.0:.4f}")
    print(f"Distance (Random):    {0.0:.4f}")

if __name__ == "__main__":
    test_ncd()
