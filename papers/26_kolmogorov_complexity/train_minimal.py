""" Day 26: Complexity Diagnostic Sweep | Verification Script | Part of 30u30 """
"""
Complexity Diagnostic Sweep
==========================

Verifying algorithmic randomness properties through compression.
Day 26 of 30u30.

This script runs a sweep across different sequence types to verify
that our estimators correctly identify "Randomness" as high complexity.
"""

import time
import random
import string
import json
from implementation import HuffmanCoder, ArithmeticCoder, ComplexityMetrics

def run_diagnostic_sweep():
    print("="*60)
    print("DAY 26: KOLMOGOROV COMPLEXITY DIAGNOSTIC")
    print("="*60)
    
    lengths = [100, 500, 1000, 5000]
    huffman = HuffmanCoder()
    arithmetic = ArithmeticCoder()
    
    results = []
    
    for length in lengths:
        print(f"\n[SWEEP] Data Length: {length} chars")
        
        # Test Cases
        cases = {
            "Constant": "A" * length,
            "Periodic": "ABC" * (length // 3),
            "English": "Complexity is the amount of information required to describe an object. " * (length // 70),
            "Random": "".join(random.choices(string.ascii_letters + string.digits, k=length))
        }
        
        for name, text in cases.items():
            start_t = time.time()
            
            h_bits = huffman.get_complexity(text)
            entropy = ComplexityMetrics.shannon_entropy(text)
            
            # For Arithmetic, we estimate bits based on range width
            val, _ = arithmetic.encode(text[:100]) # Limit length for float precision
            a_bits_estimated = arithmetic.calculate_min_bits(val, val + 1e-15) # Heuristic
            
            elapsed = time.time() - start_t
            compression_ratio = h_bits / (len(text) * 8)
            
            print(f"  {name:10} | Huffman: {h_bits:5} bits | Entropy: {entropy:8.2f} | Ratio: {compression_ratio:.3f}")
            
            results.append({
                "length": length,
                "type": name,
                "huffman_bits": h_bits,
                "shannon_entropy": entropy,
                "ratio": compression_ratio,
                "time": elapsed
            })

    # Save results
    with open("complexity_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n[OK] Diagnostic complete. Results saved to complexity_results.json")

if __name__ == "__main__":
    run_diagnostic_sweep()
