""" Day 26: Huffman Tree Construction | Solution 01 | Part of 30u30 """
"""
Solution for Exercise 1
=======================
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
    frequencies = Counter(text)
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heapq.heappop(heap)

def generate_codes(node, prefix="", codes=None):
    if codes is None: codes = {}
    if node is None: return codes
    if node.char is not None:
        codes[node.char] = prefix
        return codes
    generate_codes(node.left, prefix + "0", codes)
    generate_codes(node.right, prefix + "1", codes)
    return codes

if __name__ == "__main__":
    test_str = "algorithmic information theory is beautiful"
    root = build_huffman_tree(test_str)
    codes = generate_codes(root)
    
    print(f"Text: {test_str}")
    encoded = "".join([codes[c] for c in test_str])
    print(f"Compression ratio: {len(encoded)/(len(test_str)*8):.3f}")
