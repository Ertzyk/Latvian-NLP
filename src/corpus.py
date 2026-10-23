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
            raw_sentences = re.split(r"[.!?\n;]+", raw_text)

            for sentence in raw_sentences:
                cleaned = re.sub(r"[—–]", " ", sentence)
                cleaned = re.sub(r"[^\w\s\-]", "", cleaned)
                cleaned = re.sub(r"[\d_]", "", cleaned)

                words = []
                for w in cleaned.split():
                    w = w.strip("-")
                    if not w:
                        continue
                    if all((char in latvian_alphabet or char == "-") for char in w):
                        words.append(w)

                if words:
                    f.write(" ".join(words) + "\n")
                    word_count += len(words)

            if word_count >= target_words:
                break


def load_corpus(filepath: str = "data/latvian_corpus.txt", target_words: int = 100000) -> list[str]:
    """Loads all words as a flat list, generating the corpus first if missing."""
    if not os.path.exists(filepath):
        generate_corpus(target_words = target_words, output_file = filepath)

    with open(filepath, "r", encoding = "utf-8") as f:
        return f.read().split()


def load_sentences(filepath: str = "data/latvian_corpus.txt", target_words: int = 100000) -> list[list[str]]:
    """Loads the corpus as a list of sentences (each sentence is a list of words)."""
    if not os.path.exists(filepath):
        generate_corpus(target_words = target_words, output_file = filepath)

    with open(filepath, "r", encoding = "utf-8") as f:
        return [line.split() for line in f if line.strip()]