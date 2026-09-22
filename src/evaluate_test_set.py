"""
Consolidated Script: cleans raw training texts (if needed),
trains BPE tokenizer and two author N-gram models (if needed),
cleans and scores a provided test set of passages,
and writes predictions to a CSV file.

On first run (or with --force-retrain), this will:
    1. Clean the raw Gutenberg texts (data/hobbit_raw.txt, data/lostworld_raw.txt)
    2. Split each into train/holdout
    3. Train a shared BPE tokenizer on the combined train splits
    4. Train separate bigram N-gram models for each author
    5. Save the tokenizer and models to disk

On subsequent runs, it loads the saved tokenizer/models directly and
skips straight to evaluation, unless --force-retrain is passed.

Usage:
    python evaluate_test_set.py path/to/test_set
    python evaluate_test_set.py path/to/test_set --output my_predictions.csv
    python evaluate_test_set.py path/to/test_set --n 2 --k 0.01
    python evaluate_test_set.py path/to/test_set --force-retrain

Test set formats accepted:
    - a tab-separated .txt file: "ID<TAB>passage text", one item per line
      (this is the actual HW2 test set format — IDs are taken directly
      from the file, not auto-generated)
    - a directory of .txt files (one passage per file)
    - a single .txt file, one passage per line OR blank-line separated
    - a .csv file with a text/passage column (and optional id column)
    - a .json file: a list of strings, or a list of {"id": ..., "text": ...}

Output format:
    - if --output ends in .txt (the default): plain "ID<TAB>Label" lines,
      no header, no extra columns — matches the assignment's required
      submission format exactly.
    - if --output ends in .csv: full CSV with id, prediction, tolkien_ppl,
      doyle_ppl columns (useful for your own analysis/report, not for
      submission).
"""

import argparse
import csv
import json
import os
import re

from lm_utils import (
    find_base_dir,
    clean_text,
    split_train_holdout,
    get_sentences,
    encode_sentences,
    BPETokenizer,
    NGramModel,
    save_model,
    load_model,
    predict_author,
)

# ------------------------------------------------------------------------------
# Test set loading (format auto-detection) ** AI Code **
# ------------------------------------------------------------------------------

def load_test_passages(path):
    """Returns a list of (id, passage_text) tuples."""
    if os.path.isdir(path):
        return _load_from_directory(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_from_csv(path)
    elif ext == ".json":
        return _load_from_json(path)
    elif ext == ".txt":
        return _load_from_txt(path)
    else:
        raise ValueError(f"Unrecognized file type: {ext}")


def _load_from_directory(path):
    filenames = sorted(f for f in os.listdir(path) if f.endswith(".txt"))
    passages = []
    for fname in filenames:
        with open(os.path.join(path, fname), encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            passages.append((fname, text))
    return passages


def _load_from_txt(path):
    """
    Parses the HW2 test set format: every line is
        ID<whitespace>passage text
    where the ID appears in the file as "ITEM-<number>" but must be
    reformatted to "ID<number>" for output, per the assignment's
    example format (e.g. "ID795", "ID21").
    """
    passages = []
    with open(path, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Line {line_num} doesn't have an ID and passage text: {line!r}")
            raw_id, text = parts

            m = re.match(r"ITEM-(\d+)", raw_id)
            if not m:
                raise ValueError(f"Line {line_num}: couldn't parse an ID number from {raw_id!r}")
            item_id = f"ID{m.group(1)}"

            passages.append((item_id, text.strip()))
    return passages


def _load_from_csv(path):
    passages = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        text_col = _find_column(reader.fieldnames, ["text", "passage", "content"])
        id_col = _find_column(reader.fieldnames, ["id", "ID", "filename", "name"])
        if text_col is None:
            raise ValueError(f"No text/passage/content column found in {path}. "
                              f"Columns present: {reader.fieldnames}")
        for i, row in enumerate(reader):
            pid = row[id_col] if id_col else f"passage_{i}"
            passages.append((pid, row[text_col].strip()))
    return passages


def _load_from_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    passages = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                passages.append((f"passage_{i}", item.strip()))
            elif isinstance(item, dict):
                text = item.get("text") or item.get("passage") or item.get("content")
                pid = item.get("id", f"passage_{i}")
                if text is None:
                    raise ValueError(f"Item {i} has no 'text'/'passage'/'content' field: {item}")
                passages.append((pid, text.strip()))
    else:
        raise ValueError(f"Expected a JSON list, got {type(data)}")
    return passages


def _find_column(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None

# ------------------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------------------

def build_pipeline(data_dir, models_dir, vocab_size=5000, force_retrain=False):
    """
    Guarantee trained tokenizer and author models exist on disk.
    Returns (bpe, tolkien_model, doyle_model).
    """
    tokenizer_path = models_dir / "bpe_tokenizer_final.json"
    tolkien_path = models_dir / "tolkien_model.pkl"
    doyle_path = models_dir / "doyle_model.pkl"

    artifacts_exist = tokenizer_path.exists() and tolkien_path.exists() and doyle_path.exists()

    if artifacts_exist and not force_retrain:
        print("Found existing trained tokenizer and models — loading from disk.")
        bpe = BPETokenizer(vocab_size=vocab_size, pre_tokenizer="byte_level")
        bpe.load(str(tokenizer_path))
        tolkien_model = load_model(tolkien_path)
        doyle_model = load_model(doyle_path)
        return bpe, tolkien_model, doyle_model

    print("Artifacts not found or forced retrain. Running full pipeline...")

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    hobbit_raw = data_dir / "hobbit_raw.txt"
    lostworld_raw = data_dir / "lostworld_raw.txt"
    for raw_path, name in [(hobbit_raw, "Hobbit"), (lostworld_raw, "Lost World")]:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw source text for {name}, expected as {raw_path}. Cannot continue without it."
            )

    # 1. Preprocessing
    print("Cleaning raw texts...")
    hobbit_clean = clean_text(str(hobbit_raw), str(data_dir / "hobbit_clean.txt"))
    lostworld_clean = clean_text(str(lostworld_raw), str(data_dir / "lostworld_clean.txt"))

    # 2. Split into Training and Holdout Sets
    print("Splitting train/holdout...")
    hobbit_train, hobbit_holdout = split_train_holdout(hobbit_clean)
    lostworld_train, lostworld_holdout = split_train_holdout(lostworld_clean)

    hobbit_train_path = data_dir / "hobbit_train.txt"
    lostworld_train_path = data_dir / "lostworld_train.txt"
    with open(hobbit_train_path, "w", encoding="utf-8") as f:
        f.write(hobbit_train)
    with open(data_dir / "hobbit_holdout.txt", "w", encoding="utf-8") as f:
        f.write(hobbit_holdout)
    with open(lostworld_train_path, "w", encoding="utf-8") as f:
        f.write(lostworld_train)
    with open(data_dir / "lostworld_holdout.txt", "w", encoding="utf-8") as f:
        f.write(lostworld_holdout)

    # 3. Train Tokenizer
    print(f"Training BPE tokenizer (vocab_size= {vocab_size})...")
    bpe = BPETokenizer(vocab_size=vocab_size, pre_tokenizer="byte_level")
    bpe.train([str(hobbit_train_path), str(lostworld_train_path)])
    bpe.save(str(tokenizer_path))

    # 4. Train author N-gram models
    print("Training author N-gram models...")
    hobbit_encoded = encode_sentences(get_sentences(hobbit_train), bpe)
    lostworld_encoded = encode_sentences(get_sentences(lostworld_train), bpe)

    tolkien_model = NGramModel(vocab_size=bpe.get_vocab_size())
    tolkien_model.train(hobbit_encoded)

    doyle_model = NGramModel(vocab_size=bpe.get_vocab_size())
    doyle_model.train(lostworld_encoded)

    save_model(tolkien_model, tolkien_path)
    save_model(doyle_model, doyle_path)

    print("Success! Tokenizer and author models saved to disk.")
    return bpe, tolkien_model, doyle_model

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

def batch_predict(id_text_pairs, tolkien_model, doyle_model, bpe, n=2, k=0.01):
    """Runs predict_author on each (id, text) pair."""
    results = []
    for pid, text in id_text_pairs:
        author_id, tolkien_ppl, doyle_ppl = predict_author(
            text, tolkien_model, doyle_model, bpe, n=n, k=k
        )
        results.append({
            "id": pid,
            "prediction": author_id,
            "tolkien_ppl": tolkien_ppl,
            "doyle_ppl": doyle_ppl,
        })
    return results


def write_predictions(results, output_path):
    """
    Writes predictions in the format matching the output file's extension:
      .txt -> plain 'ID<TAB>Label' lines, no header (assignment submission format)
      .csv -> full CSV with id, prediction, tolkien_ppl, doyle_ppl columns
    """
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction", "tolkien_ppl", "doyle_ppl"])
            for r in results:
                writer.writerow([
                    r["id"],
                    r["prediction"],
                    f"{r['tolkien_ppl']:.4f}",
                    f"{r['doyle_ppl']:.4f}",
                ])
    else:
        # Plain submission format: "ID<TAB>Label", one line per result, no extras.
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r['id']}\t{r['prediction']}\n")

# ------------------------------------------------------------------------------
# Main ** AI Code **
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="One-stop-shop: builds (if needed) and runs the author-ID classifier on a test set."
    )
    parser.add_argument("test_path", help="Path to test set file or directory")
    parser.add_argument("--output", default="predictions.txt", help="Output path (.txt for submission format, .csv for full detail)")
    parser.add_argument("--n", type=int, default=2, help="N-gram order (2=bigram, 3=trigram)")
    parser.add_argument("--k", type=float, default=0.01, help="Add-k smoothing value")
    parser.add_argument("--vocab-size", type=int, default=5000, help="BPE vocab size (only used if (re)training)")
    parser.add_argument("--force-retrain", action="store_true",
                         help="Rebuild tokenizer/models from raw text even if saved artifacts exist")
    args = parser.parse_args()

    BASE_DIR = find_base_dir()
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    print(f"Repo root resolved to: {BASE_DIR}")

    bpe, tolkien_model, doyle_model = build_pipeline(
        DATA_DIR, MODELS_DIR,
        vocab_size=args.vocab_size,
        force_retrain=args.force_retrain,
    )

    test_passages = load_test_passages(args.test_path)
    print(f"Loaded {len(test_passages)} test passages.")
    if test_passages:
        print(f"First passage preview: {test_passages[0]}")

    results = batch_predict(test_passages, tolkien_model, doyle_model, bpe, n=args.n, k=args.k)

    write_predictions(results, args.output)

    # Sanity check: line counts must match exactly, per assignment instructions
    n_input = len(test_passages)
    with open(args.output, encoding="utf-8") as f:
        n_output = sum(1 for _ in f)
    if n_input != n_output:
        raise AssertionError(f"Line count mismatch! input={n_input}, output={n_output}")

    print(f"Saved {len(results)} predictions to {args.output} (line count verified).")


if __name__ == "__main__":
    main()