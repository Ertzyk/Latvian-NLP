from src.corpus import load_corpus, load_sentences
from src.information_theory import (
    build_huffman_codes,
    compute_letter_frequencies,
    evaluate_encoding,
)
from src.zipf import (
    compute_word_frequencies,
    fit_zipf_power_law,
    plot_zipf_distribution,
    extract_core_vocabulary,
)
from src.cooccurrence import extract_top_neighbors


def main():
    # Letter frequencies & Huffman
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

    # Zipf's Law Analysis
    word_freqs = compute_word_frequencies(words)
    zipf_metrics = fit_zipf_power_law(word_freqs)
    plot_path = "data/zipf_distribution.png"
    plot_zipf_distribution(word_freqs, zipf_metrics, output_path = plot_path)

    print(f"Unique words (Vocabulary Size) : {len(word_freqs):,}")
    print(f"Fitted Zipf Exponent (s)       : {zipf_metrics['exponent_s']:.4f}")
    print(f"Log-Log Goodness of Fit (R^2)  : {zipf_metrics['r_squared']:.4f}\n")

    print(f"{'Rank':<8}{'Word':<16}{'Count':<10}{'Percentage':<14}{'r * f'}")
    print("-"*60)
    for rank, word, count, pct in word_freqs[:20]:
        print(f"{rank:<8}{word:<16}{count:<10,}{f'{pct:.4f}%':<14}{rank * count:,}")

    print("-"*60)
    print(f"Saved Zipf log-log plot to: {plot_path}")

    core_path = "data/core_vocabulary_90.csv"
    core_metrics = extract_core_vocabulary(word_freqs, target_coverage_pct = 90.0, output_path = core_path)

    print(
        f"90% Corpus Coverage            : {core_metrics['core_word_count']:,} / {len(word_freqs):,} words "
        f"({core_metrics['vocab_share_pct']:.2f}% of vocabulary)"
    )
    print(f"Exact Cumulative Coverage    : {core_metrics['actual_coverage_pct']:.4f}%")
    print(f"Saved 90% core vocabulary to : {core_path}")

    # Word Co-occurrence / Neighbors
    sentences = load_sentences("data/latvian_corpus.txt")
    neighbors_path = "data/top_100_neighbors.csv"
    top_neighbors = extract_top_neighbors(sentences, top_n = 100, output_path = neighbors_path)

    print("-"*60)
    print(f"Saved top 100 words by unique neighbors : {neighbors_path}\n")
    print(f"{'Rank':<8}{'Word':<18}{'Unique Neighbors'}")
    print("-"*60)
    for rank, word, count in top_neighbors[:20]:
        print(f"{rank:<8}{word:<18}{count:,}")

if __name__ == "__main__":
    main()