""" Day 26: Complexity Spectrum Visualization | Plotting Logic | Part of 30u30 """
"""
Complexity Spectrum Visualization
=================================

Visualizing the "Algorithmic Information" vs "Randomness" tradeoff.
Day 26 of 30u30.

This script demonstrates:
1. The Complexity Spectrum (Constant vs. Patterned vs. Random)
2. NCD Similarity Heatmap (Identifying clusters through compression)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from implementation import HuffmanCoder, ComplexityMetrics
import random
import string

def generate_data():
    length = 500
    # 1. Constant (Lowest complexity)
    constant = "a" * length
    
    # 2. Periodic/Patterned (Medium-Low complexity)
    periodic = "abcd" * (length // 4)
    
    # 3. Structure with noise (Medium complexity)
    structured = "".join([random.choice("abc") if i % 2 == 0 else "x" for i in range(length)])
    
    # 4. Pure Random (Highest complexity)
    random_str = "".join(random.choices(string.ascii_lowercase, k=length))
    
    return {
        "Constant": constant,
        "Periodic": periodic,
        "Structured": structured,
        "Random": random_str
    }

def plot_complexity_spectrum(data):
    coder = HuffmanCoder()
    names = list(data.keys())
    
    huffman_bits = [coder.get_complexity(s) for s in data.values()]
    entropy_bits = [ComplexityMetrics.shannon_entropy(s) for s in data.values()]
    raw_bits = [len(s) * 8 for s in data.values()] # 8 bits per char
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, raw_bits, width, label='Raw Bits (C_max)', color='#2e2a28', alpha=0.3)
    plt.bar(x, huffman_bits, width, label='Huffman Complexity', color='#e07a5f')
    plt.bar(x + width, entropy_bits, width, label='Shannon Entropy', color='#3d5a45')
    
    plt.ylabel('Complexity (Bits)')
    plt.title('The Complexity Spectrum: From Constant to Random')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('complexity_spectrum.png')
    print("[OK] Saved complexity_spectrum.png")


# =============================================================================
# ======= SECTION 2: NCD HEATMAPS =======
# =============================================================================

def plot_ncd_heatmap():
    # Generate snippets from different categories
    cats = {
        "Math": ["The square root of two is irrational", "Prime numbers are infinite", "Calculus is the study of change"],
        "Code": ["def hello_world(): print('Hi')", "if x > 10: return True", "while True: continue"],
        "Strings": ["aaaaaaaaaaaaaaaaaaaa", "abababababababababab", "abcabcabcabcabcabcab"]
    }
    
    all_texts = []
    labels = []
    for cat, texts in cats.items():
        all_texts.extend(texts)
        labels.extend([cat] * len(texts))
        
    n = len(all_texts)
    matrix = np.zeros((n, n))
    
    huffman = HuffmanCoder()
    compressor = lambda x: huffman.get_complexity(x)
    
    for i in range(n):
        for j in range(n):
            matrix[i, j] = ComplexityMetrics.ncd(all_texts[i], all_texts[j], compressor)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap='RdYlGn_r', fmt=".2f")
    plt.title('Normalized Compression Distance (NCD) Heatmap')
    plt.tight_layout()
    plt.savefig('ncd_heatmap.png')
    print("[OK] Saved ncd_heatmap.png")


# =============================================================================
# ======= SECTION 3: MAIN EXECUTION =======
# =============================================================================

if __name__ == "__main__":
    print("Generating complexity visualizations...")
    data = generate_data()
    plot_complexity_spectrum(data)
    plot_ncd_heatmap()
    print("\nVisualizations complete. Check complexity_spectrum.png and ncd_heatmap.png")
