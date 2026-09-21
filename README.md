# Language Models and Author Identification

Implementation of a BPE tokenizer, N-gram language models (bigram + trigram) with add-k smoothing, perplexity evaluation, and author identification classifier distinguishing **J.R.R. Tolkien** (*The Hobbit*) from **Arthur Conan Doyle** (*The Lost World*).

## Repository Structure

- `data/`
  - `hobbit_raw.txt` — original source text
  - `hobbit_clean.txt` — after Gutenberg/front-matter/chapter cleaning
  - `hobbit_train.txt` — 90% split
  - `hobbit_holdout.txt` — 10% split
  - `lostworld_raw.txt`
  - `lostworld_clean.txt`
  - `lostworld_train.txt`
  - `lostworld_holdout.txt`
  - `bpe_tokenizer_final.json` — saved trained BPE tokenizer
- `models/`
  - `tolkien_model.pkl` — pickled trained NGramModel (Hobbit)
  - `doyle_model.pkl` — pickled trained NGramModel (Lost World)
- `src/`
   - `lm_utils.py` — core library: cleaning, tokenizer, n-gram model, classifier
   - `evaluate_test_set.py` — one-stop-shop script: builds the pipeline if needed, scores a test set
   - `notebook.ipynb` — exploratory pipeline: vocab-size sweep, k sweep, error analysis, validation
- `requirements.txt`
- `report.pdf` — written report
- `README.md`

## Setup

**1. Clone the repo:**
```bash
git clone <repo-url>
cd nlp-author-identification
```

**2. Create and activate an environment, then install dependencies:**
```bash
conda create -n nlp_env python=3.10
conda activate nlp_env
pip install -r requirements.txt
```
(or use `venv` instead of conda — either works, as long as the same
environment is active for both any notebook kernel and any terminal
commands you run.)

**3. Raw source texts.** This repo expects `data/hobbit_raw.txt` and
`data/lostworld_raw.txt` to be present (plain-text editions of
*The Hobbit* and *The Lost World*).

## Running on a test set

`evaluate_test_set.py` is a one-stop-shop script. On first run, if a
trained tokenizer and author models aren't already present in `data/` and
`models/`, it automatically runs the full pipeline from the raw texts
(clean → split → train tokenizer → train both author N-gram models →
save to disk) before scoring anything. On subsequent runs it loads the
saved tokenizer/models directly.

```bash
python evaluate_test_set.py path/to/test_set
```

**Optional flags:**
```bash
python evaluate_test_set.py path/to/test_set --output my_predictions.csv
python evaluate_test_set.py path/to/test_set --n 2 --k 0.01        # n-gram order / smoothing
python evaluate_test_set.py path/to/test_set --vocab-size 5000     # only used if (re)training
python evaluate_test_set.py path/to/test_set --force-retrain       # rebuild from raw text
```

**Accepted test set formats** (auto-detected from the path):
- a directory of `.txt` files (one passage per file)
- a single `.txt` file, one passage per line **or** blank-line-separated passages
- a `.csv` file with a `text`/`passage`/`content` column (and optional `id` column)
- a `.json` file: either a list of strings, or a list of `{"id": ..., "text": ...}` objects

Output is written as a CSV with columns `id, prediction, tolkien_ppl, doyle_ppl`.

Individual test passages are lightly, safely cleaned automatically inside
`predict_author()` (stripping any stray Gutenberg boilerplate or
illustration tags and normalizing whitespace) — no separate cleaning step
is required by the caller.

## Pipeline Overview

1. **Cleaning** (`clean_text`): strips Gutenberg header/footer, title
   page/table of contents, and chapter heading lines from the raw novels.
2. **Train/holdout split** (`split_train_holdout`): 90/10 split by
   sentence, performed *after* cleaning and *before* tokenizer training.
3. **Tokenizer** (`BPETokenizer`): Hugging Face `tokenizers` BPE
   implementation, `ByteLevel` pre-tokenizer/decoder, vocab_size=5000,
   trained on the combined train splits of both books (a single shared
   vocabulary, so token ids are comparable across the two author models).
4. **N-gram models** (`NGramModel`): unigram/bigram/trigram counts with
   add-k smoothing. Final configuration — **bigram, k = 0.01** — was
   selected via a systematic perplexity sweep over n-gram order and k on
   held-out data (see `report.pdf` for full results and reasoning).
5. **Classifier** (`predict_author`): predicts author by comparing a
   passage's perplexity under two independently trained author models
   (Tolkien-only, Doyle-only) and selecting whichever is lower.

See `report.pdf` for the full write-up, including the vocabulary-size
exploration, the bigram-vs-trigram/k-sweep results, validation accuracy
by passage length, out-of-domain validation on other Tolkien/Doyle
novels, and error analysis.

## Use of pre-existing / AI-generated code

*All AI-generated code was carefully reviewed and validated.*

- **BPE implementation**: Hugging Face `tokenizers` library, used as
  instructed by the assignment (not reimplemented).
- **Test Set Loading**: AI-generated code to parse format of test passages and load them.
- **Final Evaluation Call**: AI-generated code for `main` in `evaluate_test_set.py`.
- **Everything else** (`BPETokenizer` wrapper, text cleaning, sentence
  splitting, `NGramModel`, the author classifier) is original code, developed with AI
  assistance (Claude, Anthropic) for design discussion, debugging, and
  code review. Specific difficulties encountered and resolved with this
  assistance (e.g. a decoder/pre-tokenizer mismatch causing incorrect
  word reconstruction, a train/inference preprocessing inconsistency, and
  several rounds of Gutenberg text-cleaning refinement) are documented in
  `report.pdf`.
- **README.md**: AI-generated text for this README.
