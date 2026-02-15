""" Day 26: Kolmogorov Complexity Estimators | Core Implementation | Part of 30u30 """
"""
Kolmogorov Complexity Estimators
===============================

Reference implementation for Day 26 of 30u30.
This module implements computable upper bounds for Kolmogorov Complexity
as discussed in Shen, Uspensky, and Vereshchagin (2017).

Included:
- HuffmanCoder: Frequency-based complexity (Shannon-optimal)
- ArithmeticCoder: Range-based probabilistic complexity (Optimal for skewed data)
- ComplexityMetrics: NCD and Bit-rate estimators

Traceability:
- Section 1.1 (Complexity of finite objects)
- Section 2.1 (Prefix-free complexity)
- Section 3.4 (Coding theorems)

Author: 30u30 Project
License: CC BY-NC-ND 4.0
"""


# =============================================================================
# ======= SECTION 1: HUFFMAN CODING (FREQUENCY-BASED) =======
# =============================================================================

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoder:
    """
    Implements Huffman Coding (Section 3.4 coding theorems).
    Provides a Shannon-optimal prefix code for a given frequency distribution.
    """
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}

    def build_tree(self, text):
        if not text:
            return None
            
        frequencies = Counter(text)
        heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)

        return heapq.heappop(heap)

    def _generate_codes(self, node, current_code):
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_codes[current_code] = node.char
            return

        self._generate_codes(node.left, current_code + "0")
        self._generate_codes(node.right, current_code + "1")

    def encode(self, text):
        """
        Returns the encoded string of bits.
        """
        if not text:
            return ""
            
        root = self.build_tree(text)
        self.codes = {}
        self._generate_codes(root, "")
        
        return "".join([self.codes[char] for char in text])

    def get_complexity(self, text):
        """
        Returns the length of the compressed string in bits.
        This provides an upper bound for C(text).
        """
        encoded = self.encode(text)
        return len(encoded)


# =============================================================================
# ======= SECTION 2: ARITHMETIC CODING (PROBABILISTIC) =======
# =============================================================================

class ArithmeticCoder:
    """
    Implements Arithmetic Coding (Section 3.4).
    Encodes a sequence into a single fractional number in [0, 1).
    More efficient than Huffman for skewed probabilities where -log2(p) is not integer.
    """
    def __init__(self, precision=32):
        self.precision = precision

    def _get_probabilities(self, text):
        freqs = Counter(text)
        total = len(text)
        probs = {}
        cumulative = 0.0
        
        # Sort keys for deterministic ranges
        for char in sorted(freqs.keys()):
            p = freqs[char] / total
            probs[char] = (cumulative, cumulative + p)
            cumulative += p
        return probs

    def encode(self, text):
        """
        Basic implementation of range-based arithmetic encoding.
        Note: This uses standard floats and is susceptible to precision issues 
        for very long strings. In production, fixed-point integer math is used.
        """
        if not text:
            return 0.0, 0
            
        probs = self._get_probabilities(text)
        low = 0.0
        high = 1.0
        
        for char in text:
            p_low, p_high = probs[char]
            range_width = high - low
            high = low + range_width * p_high
            low = low + range_width * p_low
            
        return (low + high) / 2.0, len(text)

    @staticmethod
    def calculate_min_bits(low, high):
        """
        Calculates the theoretical minimum bits required to represent the range.
        Log2(range_width) provides the bit-count for complexity estimation.
        """
        if high <= low: return 0
        return int(np.ceil(-np.log2(high - low)))


# =============================================================================
# ======= SECTION 3: COMPLEXITY METRICS & NCD =======
# =============================================================================

class ComplexityMetrics:
    """
    High-level metrics for Algorithmic Information Theory.
    """
    @staticmethod
    def shannon_entropy(text):
        """
        H(X) = -sum(p(x) * log2(p(x)))
        Theoretical limit for i.i.d. sources.
        """
        if not text: return 0.0
        freqs = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in freqs.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy * total # Total bits

    @staticmethod
    def ncd(x, y, compressor_func):
        """
        Normalized Compression Distance:
        NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        
        Approximates the normalized version of Kolmogorov distance.
        Range [0, 1], where 0 is identical and 1 is maximal difference.
        """
        len_x = compressor_func(x)
        len_y = compressor_func(y)
        len_xy = compressor_func(x + y)
        
        mx = max(len_x, len_y)
        if mx == 0: return 0.0
        return (len_xy - min(len_x, len_y)) / mx


# =============================================================================
# ======= SECTION 4: DIAGNOSTIC DEMONSTRATION =======
# =============================================================================

if __name__ == "__main__":
    # Diagnostic Demonstration
    test_str_1 = "abababababababababab" # Patterned (Low KC)
    test_str_2 = "4c1j5b2p9n7m3q3k8r1t" # Pseudo-Random (High KC)
    
    huffman = HuffmanCoder()
    print(f"Text 1: {test_str_1}")
    print(f"  Entropy bits: {ComplexityMetrics.shannon_entropy(test_str_1):.2f}")
    print(f"  Huffman bits: {huffman.get_complexity(test_str_1)}")
    
    print(f"\nText 2: {test_str_2}")
    print(f"  Entropy bits: {ComplexityMetrics.shannon_entropy(test_str_2):.2f}")
    print(f"  Huffman bits: {huffman.get_complexity(test_str_2)}")
