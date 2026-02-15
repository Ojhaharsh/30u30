""" Day 26: Huffman Tree Construction | Exercise 01 | Part of 30u30 """
"""
Exercise 1: Huffman Tree Construction
======================================

In this exercise, you will implement the core of frequency-based encoding.
Huffman coding is the basic yardstick for $C(x)$ — the complexity of a 
string when we assume its characters are independent.

Goal:
1. Build a leaf node for each unique character.
2. Implement the tree-merging logic Using a min-priority queue.
3. Generate bit-codes by traversing the tree.
"""

import heapq
from collections import Counter

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    if not text: return None
    
    # 1. Count frequencies
    frequencies = Counter(text)
    
    # 2. Initialize priority queue (heap)
    # TODO: Create a list of Node objects and heapify it
    heap = [] 
    
    # 3. Merge nodes until only one root remains
    while len(heap) > 1:
        # TODO: Pop two lowest-frequency nodes
        # TODO: Create a parent node with sum frequency
        # TODO: Set children and push back to heap
        pass

    return heapq.heappop(heap) if heap else None

def generate_codes(node, prefix="", codes={}):
    if node is None: return codes
    
    # TODO: If leaf node (node.char is not None), store prefix in codes
    # TODO: Recursively call for left (append '0') and right (append '1')
    
    return codes

if __name__ == "__main__":
    test_str = "algorithmic information theory is beautiful"
    root = build_huffman_tree(test_str)
    codes = generate_codes(root)
    
    print(f"Text: {test_str}")
    print("Character Codes:")
    for char in sorted(codes.keys()):
        print(f"  '{char}': {codes[char]}")
    
    encoded = "".join([codes[c] for c in test_str])
    print(f"\nOriginal bits: {len(test_str)*8}")
    print(f"Compressed bits: {len(encoded)}")
    print(f"Compression ratio: {len(encoded)/(len(test_str)*8):.3f}")
