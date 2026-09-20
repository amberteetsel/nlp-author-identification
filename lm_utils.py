"""
Methods for data preprocessing, BPE tokenization, N-gram language models, and author identification.
"""

# Load dependencies
import re
import os
import csv
import json
import math
import pickle
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
print("Dependencies loaded successfully.")

# Resolve Path
def find_base_dir(marker=".git"):
    """Navigates up from cwd to repo root"""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Unable to locate repo root (searching for '{marker}')")

# ------------------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------------------

def remove_gutenberg(text):
    """Strips text outside Gutenberg header and footer"""
    start_txt = re.search(r"\*\*\* START OF .*?\*\*\*", text)
    end_txt = re.search(r"\*\*\* END OF .*?\*\*\*", text)
    start = start_txt.end() if start_txt else 0
    end = end_txt.start() if end_txt else len(text)
    return text[start:end].strip()

def remove_front_extra(text):
    """Removes everything before first real chapter heading.
    Requires 'CHAPTER|Chapter' + ' ' + roman numeral on the same line,
    to distinguish from table of contents, but allows leading indentation before
    'Chapter'"""
    ch1 = re.search(
        r"\n[ \t]*(CHAPTER|Chapter)[ \t]+[IVXLCDM]+\.?[ \t]*$",
        text,
        flags=re.MULTILINE,
    )
    if ch1:
        return text[ch1.start():].strip()
    else:
        return text

def remove_chapter_headings(text):
    """Removes 'Chapter I', "CHAPTER IX", etc."""
    pattern = r"^\s*(CHAPTER|Chapter)\s+[IVXLCDM]+\.?\s*$"
    return re.sub(pattern, "", text, flags=re.MULTILINE)

def clean_text(inpath, outpath):
    """Cleans text files and saves to a new file"""
    with open(inpath, encoding="utf-8") as f:
        text = f.read()
    text = remove_gutenberg(text)
    text = remove_front_extra(text)
    text = remove_chapter_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text) # remove excess newlines
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# ------------------------------------------------------------------------------
# Split sentences, train-holdout split
# ------------------------------------------------------------------------------

def get_sentences(text):
    """Splits text into sentences using regex."""
    s = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", s)

# AI Code
def split_train_holdout(text, holdout_frac=0.1):
    # Collapse ALL whitespace/newlines into single spaces —
    # don't trust newline count to mean anything structurally
    flat = re.sub(r"\s+", " ", text).strip()
    
    # Split into sentences: break after ., !, or ? followed by a space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", flat)
    
    n_holdout = max(1, int(len(sentences) * holdout_frac))
    
    train = sentences[:-n_holdout]
    holdout = sentences[-n_holdout:]
    
    return " ".join(train), " ".join(holdout)

# ------------------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, vocab_size=5000, pre_tokenizer="byte_level"):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        if pre_tokenizer == "byte_level":
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
        elif pre_tokenizer == "whitespace":
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            # no matching decoder so use default
        else:
            raise ValueError(f"Unknown pre_tokenizer: {pre_tokenizer}")

        self.special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]

    def train(self, filepaths):
        """Train BPE on one or more text files (paths as a list)."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train(filepaths, trainer)

    def encode(self, text):
        """Normalize input and returns list of token ids."""
        text = re.sub(r"\s+", " ", text).strip()
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Returns string from list of token ids."""
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

def encode_sentences(sentences, tokenizer):
    """Tokenizes each sentence and wraps with <s> and </s>"""
    s_id = tokenizer.tokenizer.token_to_id("<s>")
    e_id = tokenizer.tokenizer.token_to_id("</s>")
    encoded_sentences = []
    for sent in sentences:
        ids = tokenizer.encode(sent)
        if ids:
            encoded_sentences.append([s_id] + ids + [e_id])
    return encoded_sentences

# ------------------------------------------------------------------------------
# N-gram model
# ------------------------------------------------------------------------------

# N-Gram model
class NGramModel:
    "Unigram, bigram and trigram counts with add-k smoothing"

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.bigram_context_counts = Counter()
        self.trigram_context_counts = Counter()

    def train(self, encoded_sentences):
        """encoded_sentences: list of token id lists wrapped with <s> and </s>"""
        for sent in encoded_sentences:
            for i, token in enumerate(sent):
                self.unigram_counts[token] += 1

                if i >= 1:
                    bigram = (sent[i-1], sent[i])
                    self.bigram_counts[bigram] += 1
                    self.bigram_context_counts[(sent[i-1]),] += 1

                if i >= 2:
                    trigram = (sent[i-2], sent[i-1], sent[i])
                    self.trigram_counts[trigram] += 1
                    self.trigram_context_counts[(sent[i-2], sent[i-1])] += 1

    def bigram_prob(self, w1, w2, k=1.0):
        """P(w2 | w1) with add-k smoothing"""
        count_bigram = self.bigram_counts[(w1, w2)]
        count_context = self.bigram_context_counts[(w1,)]
        return (count_bigram + k) / (count_context + k * self.vocab_size)

    def trigram_prob(self, w1, w2, w3, k=1.0):
        """P(w3 | w1, w2) with add-k smoothing"""
        count_trigram = self.trigram_counts[(w1, w2, w3)]
        count_context = self.trigram_context_counts[(w1,w2)]
        return (count_trigram + k) / (count_context + k * self.vocab_size)

    def neg_log_prob(self, sent, n=2, k=1.0):
        """Returns total -logP(sentence) using n_gram model (n=2 for bigram, n=3 for trigram)"""
        total = 0.0
        for i in range(1, len(sent)) if n==2 else range(2, len(sent)):
            if n==2:
                p = self.bigram_prob(sent[i-1], sent[i], k)
            elif n==3:
                p = self.trigram_prob(sent[i-2], sent[i-1], sent[i], k)
            else:
                raise ValueError("n must be 2 (bigram) or 3 (trigram)")
            total += -math.log(p)
        return total

    def perplexity(self, sentences, n=2, k=1.0):
        """Computes perplexity over list of encoded sentences (list of token id lists)"""
        total_neg_log_prob = 0.0
        total_tokens = 0

        for sent in sentences:
            total_neg_log_prob += self.neg_log_prob(sent, n=n, k=k)
            # count predicted tokens, excl. <s>
            # same logic inside neg_log_prob: start at 1 for bigram, 2 for trigram
            n_predicted = len(sent)-1 if n==2 else len(sent)-2
            total_tokens += n_predicted

        avg_neg_log_prob = total_neg_log_prob / total_tokens
        return math.exp(avg_neg_log_prob)


# ------------------------------------------------------------------------------
# Author Classifier
# ------------------------------------------------------------------------------

def predict_author(text, tolkien_model, doyle_model, bpe, n=2, k=0.01):
    """Predicts author of input text based on perplexity"""
    sentences = get_sentences(text)
    encoded_sentences = encode_sentences(sentences, bpe)

    tolkien_ppl = tolkien_model.perplexity(encoded_sentences, n=n, k=k)
    doyle_ppl = doyle_model.perplexity(encoded_sentences, n=n, k=k)

    author_id = "J.R.R. Tolkien" if tolkien_ppl < doyle_ppl else "Arthur Conan Doyle"

    return author_id, tolkien_ppl, doyle_ppl

# ------------------------------------------------------------------------------
# Saving/Loading Models
# ------------------------------------------------------------------------------

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)