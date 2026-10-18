import heapq
import math
from collections import Counter

LATVIAN_ALPHABET = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"


class HuffmanNode:
    def __init__(self, char: str | None, weight: int, left = None, right = None):
        self.char = char
        self.weight = weight
        self.left = left
        self.right = right

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.weight < other.weight


def compute_letter_frequencies(words: list[str]) -> list[tuple[str, int, float]]:
    """Returns (letter, count, percentage) sorted descending by count."""
    alphabet_set = set(LATVIAN_ALPHABET)
    counts = Counter({letter: 0 for letter in LATVIAN_ALPHABET})

    for word in words:
        for char in word:
            if char in alphabet_set:
                counts[char] += 1

    total_letters = sum(counts.values())
    if total_letters == 0:
        return []

    frequencies = [
        (letter, count, (count/total_letters)*100.0)
        for letter, count in counts.items()
    ]
    return sorted(frequencies, key = lambda x: x[1], reverse=True)


def build_huffman_codes(frequencies: list[tuple[str, int, float]]) -> dict[str, str]:
    """Builds an optimal prefix-free binary Huffman code from letter counts."""
    heap = [HuffmanNode(letter, count) for letter, count, _ in frequencies if count > 0]
    heapq.heapify(heap)

    if not heap:
        return {}
    if len(heap) == 1:
        return {heap[0].char: "0"}

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.weight + right.weight, left, right)
        heapq.heappush(heap, merged)

    codes: dict[str, str] = {}

    def traverse(node: HuffmanNode | None, prefix: str) -> None:
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = prefix
            return
        traverse(node.left, prefix + "0")
        traverse(node.right, prefix + "1")

    traverse(heap[0], "")
    return codes


def evaluate_encoding(frequencies: list[tuple[str, int, float]], codes: dict[str, str]) -> dict[str, float]:
    """Computes Shannon entropy H(X), average Huffman code length L, and encoding efficiency (H / L)."""
    total_letters = sum(count for _, count, _ in frequencies)
    entropy = 0.0
    avg_code_length = 0.0

    for letter, count, _ in frequencies:
        if count > 0:
            p = count / total_letters
            entropy -= p*math.log2(p)
            avg_code_length += p*len(codes[letter])

    efficiency = (entropy/avg_code_length)*100.0 if avg_code_length > 0 else 0.0

    return {
        "shannon_entropy": entropy,
        "avg_code_length": avg_code_length,
        "efficiency_pct": efficiency,
    }