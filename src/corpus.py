import os
import re
from datasets import load_dataset


def generate_corpus(target_words: int = 100000, output_file: str = "data/latvian_corpus.txt") -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok = True)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.lv", split = "train")

    word_count = 0
    latvian_alphabet = set("aābcčdeēfgģhiījkķlļmnņoprsštuūvzž")

    with open(output_file, "w", encoding = "utf-8") as f:
        for article in dataset:
            raw_text = article["text"].lower()
            cleaned_text = re.sub(r"[—–]", " ", raw_text)
            cleaned_text = re.sub(r"[^\w\s\-]", "", cleaned_text)
            cleaned_text = re.sub(r"[\d_]", "", cleaned_text)

            raw_words = cleaned_text.split()

            words = []
            for w in raw_words:
                w = w.strip("-")
                if not w:
                    continue
                if all((char in latvian_alphabet or char == "-") for char in w):
                    words.append(w)

            if not words:
                continue

            f.write("\n".join(words) + "\n")
            word_count += len(words)

            if word_count >= target_words:
                break


def load_corpus(filepath: str = "data/latvian_corpus.txt", target_words: int = 200000) -> list[str]:
    """Loads the corpus from disk, or generates it first if missing."""
    if not os.path.exists(filepath):
        generate_corpus(target_words = target_words, output_file = filepath)

    with open(filepath, "r", encoding = "utf-8") as f:
        return [line.strip() for line in f if line.strip()]