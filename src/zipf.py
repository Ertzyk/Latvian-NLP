import math
import os
from collections import Counter
import matplotlib.pyplot as plt
import csv


def compute_word_frequencies(words: list[str]) -> list[tuple[int, str, int, float]]:
    total_words = len(words)
    if total_words == 0:
        return []

    counts = Counter(words)
    sorted_words = counts.most_common()

    return [
        (rank, word, count, (count/total_words)*100.0)
        for rank, (word, count) in enumerate(sorted_words, start=1)
    ]


def fit_zipf_power_law(word_freqs: list[tuple[int, str, int, float]]) -> dict[str, float]:
    """
    Fits log10(f) = log10(C) - s * log10(r) using Ordinary Least Squares (OLS).
    Returns the empirical Zipf exponent 's', intercept 'log10_C', and 'r_squared'.
    """
    n = len(word_freqs)
    if n < 2:
        return {"exponent_s": 0.0, "log10_c": 0.0, "r_squared": 0.0}

    log_ranks = [math.log10(rank) for rank, _, _, _ in word_freqs]
    log_counts = [math.log10(count) for _, _, count, _ in word_freqs]

    mean_x = sum(log_ranks)/n
    mean_y = sum(log_counts)/n
    ss_xy = sum((x - mean_x)*(y - mean_y) for x, y in zip(log_ranks, log_counts))
    ss_xx = sum((x - mean_x)**2 for x in log_ranks)
    ss_yy = sum((y - mean_y)**2 for y in log_counts)

    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    r_squared = (ss_xy ** 2)/(ss_xx * ss_yy) if (ss_xx > 0 and ss_yy > 0) else 0.0

    return {
        "exponent_s": -slope,
        "log10_c": intercept,
        "r_squared": r_squared,
    }


def plot_zipf_distribution(word_freqs: list[tuple[int, str, int, float]], fit_metrics: dict[str, float],
    output_path: str = "data/zipf_distribution.png",) -> None:
    """Generates and saves a log-log plot comparing empirical frequencies to Zipf's Law."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ranks = [rank for rank, _, _, _ in word_freqs]
    counts = [count for _, _, count, _ in word_freqs]

    s = fit_metrics["exponent_s"]
    c = 10 ** fit_metrics["log10_c"]
    r2 = fit_metrics["r_squared"]

    fitted_counts = [c/(r**s) for r in ranks]
    theoretical_counts = [counts[0]/r for r in ranks]

    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, counts, ".", alpha=0.35, markersize=4, label="Empirical Latvian Corpus")
    plt.loglog(ranks, fitted_counts, "r-", linewidth=1.8, label=f"OLS Power-Law Fit ($s = {s:.3f}$, $R^2 = {r2:.4f}$)",)
    plt.loglog(ranks, theoretical_counts, "k--", linewidth=1.2, alpha=0.7, label="Theoretical Zipf ($s = 1.000$)",)

    plt.title("Zipf's Law Distribution — Latvian Corpus (Log-Log Scale)")
    plt.xlabel("Word Rank $r$ (log scale)")
    plt.ylabel("Word Frequency $f$ (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def extract_core_vocabulary(word_freqs: list[tuple[int, str, int, float]], target_coverage_pct: float = 90.0,
    output_path: str = "data/core_vocabulary_90.csv",) -> dict[str, int | float]:
    """
    Finds the minimal set of top words that account for at least `target_coverage_pct`
    of the total corpus tokens and exports the table to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_words = sum(count for _, _, count, _ in word_freqs)
    if total_words == 0:
        return {"core_word_count": 0, "vocab_share_pct": 0.0, "actual_coverage_pct": 0.0}

    core_words = []
    cumulative_count = 0

    for rank, word, count, pct in word_freqs:
        cumulative_count += count
        cumulative_pct = (cumulative_count / total_words) * 100.0
        core_words.append((rank, word, count, round(pct, 6), round(cumulative_pct, 6)))
        if cumulative_pct > target_coverage_pct:
            break

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count", "percentage", "cumulative_percentage"])
        writer.writerows(core_words)

    vocab_share_pct = (len(core_words) / len(word_freqs)) * 100.0
    actual_coverage_pct = (cumulative_count / total_words) * 100.0

    return {
        "core_word_count": len(core_words),
        "vocab_share_pct": vocab_share_pct,
        "actual_coverage_pct": actual_coverage_pct,
    }