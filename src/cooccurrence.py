import csv
import os
from collections import defaultdict


def extract_top_neighbors(sentences: list[list[str]], top_n: int = 100, output_path: str = "data/top_100_neighbors.csv") -> list[tuple[int, str, int]]:
    """Counts unique adjacent words within sentences and saves the top_n to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    neighbors: dict[str, set[str]] = defaultdict(set)

    for sentence in sentences:
        for w1, w2 in zip(sentence, sentence[1:]):
            if w1 != w2:
                neighbors[w1].add(w2)
                neighbors[w2].add(w1)

    sorted_words = sorted(((word, len(adj)) for word, adj in neighbors.items()), key = lambda x: x[1], reverse = True)[:top_n]

    results = [(rank, word, count) for rank, (word, count) in enumerate(sorted_words, start = 1)]

    with open(output_path, "w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "unique_neighbors"])
        writer.writerows(results)

    return results