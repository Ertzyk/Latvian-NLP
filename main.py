from src.corpus import load_corpus
from src.information_theory import (
    build_huffman_codes,
    compute_letter_frequencies,
    evaluate_encoding,
)


def main():
    words = load_corpus("data/latvian_corpus.txt")
    freqs = compute_letter_frequencies(words)
    codes = build_huffman_codes(freqs)
    metrics = evaluate_encoding(freqs, codes)

    total_letters = sum(count for _, count, _ in freqs)
    print(f"Total words: {len(words):,} | Total letters: {total_letters:,}\n")

    print(f"{'Rank':<6}{'Letter':<8}{'Count':<12}{'Percentage':<14}{'Huffman Code':<14}{'Bits'}")
    print("-" * 60)
    for rank, (letter, count, pct) in enumerate(freqs, start=1):
        code = codes[letter]
        print(f"{rank:<6}{letter:<8}{count:<12,}{f'{pct:.4f}%':<14}{code:<14}{len(code)}")

    print("-" * 60)
    print(f"Shannon Entropy             : {metrics['shannon_entropy']:.4f} bits/letter")
    print(f"Huffman Average Code Length : {metrics['avg_code_length']:.4f} bits/letter")
    print(f"Encoding Efficiency         : {metrics['efficiency_pct']:.2f}%")


if __name__ == "__main__":
    main()