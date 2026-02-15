""" Day 26: Similarity Check (NCD) | Solution 03 | Part of 30u30 """
"""
Solution for Exercise 3
=======================
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
    
    # Calculate NCD for the combinations
    d_identity = ComplexityMetrics.ncd(s_base, s_same, compressor)
    d_different = ComplexityMetrics.ncd(s_base, s_diff, compressor)
    d_random = ComplexityMetrics.ncd(s_base, s_rand, compressor)
    
    print(f"Distance (Identity):  {d_identity:.4f}")
    print(f"Distance (Different): {d_different:.4f}")
    print(f"Distance (Random):    {d_random:.4f}")

if __name__ == "__main__":
    test_ncd()
