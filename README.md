# Fake News Headline Classifier

Binary NLP classifier that distinguishes **real** from **fake** news headlines using TF-IDF features and a Linear SVM.

| | |
|---|---|
| **Task** | Binary text classification |
| **Model** | TF-IDF (word + char n-grams) + LinearSVC |
| **CV Accuracy** | **96.67% ± 0.27%** (5-fold stratified) |
| **Training set** | 34,151 headlines |
| **Test set** | 9,983 headlines |

---

## Labels

| Label | Meaning |
|---|---|
| `0` | Fake news |
| `1` | Real news |

---

## Project Structure

```
.
├── training_data.csv              # Labeled training set (tab-sep, no header)
├── testing_data.csv               # Unlabeled test set (label column = 2)
├── testing_data_predicted.csv     # ← Output: labels 2 replaced by 0/1
├── fake_news_classifier.ipynb     # Full pipeline notebook
├── fake_news_nlp.ppt              # Project overview presentation
└── README.md
```

---

## Method

### Why this approach?

Benchmarked five configurations via 5-fold stratified cross-validation:

| Model | CV Accuracy | Δ vs baseline |
|---|---|---|
| TF-IDF (word 1-2g) + Logistic Regression *(baseline)* | 94.88% | — |
| TF-IDF (word 1-2g) + LinearSVC | 94.99% | +0.11 pp |
| Soft Voting ensemble: LR + SVC + NB | 95.19% | +0.31 pp |
| TF-IDF (word+char) + Logistic Regression | 96.52% | +1.64 pp |
| **TF-IDF (word+char) + LinearSVC ✓** | **96.67%** | **+1.79 pp** |

**Key insight:** Adding character n-grams (+1.79 pp) is far more impactful than ensembling (+0.31 pp). Character n-grams capture stylistic signals that word tokens miss entirely — ALL-CAPS, punctuation abuse (`!!!`), sensationalist suffixes (`-gate`), and non-standard spelling.

### Pipeline

```
Raw headline
    │
    ▼
Text cleaning          # lowercase · fix encoding artefacts · collapse whitespace
    │
    ▼
FeatureUnion
    ├── TfidfVectorizer(word, ngram_range=(1,2), max_features=60k, sublinear_tf=True)
    └── TfidfVectorizer(char_wb, ngram_range=(3,5), max_features=40k, sublinear_tf=True)
    │
    ▼
LinearSVC(C=0.5, max_iter=2000)
    │
    ▼
0 (fake) or 1 (real)
```

**Why `char_wb`?** The `char_wb` analyzer pads word boundaries with spaces before extracting character n-grams. This prevents n-grams from bleeding across word boundaries — better suited for short texts like headlines.

---

## Quickstart

### Requirements

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
```

### Run the notebook

```bash
jupyter notebook fake_news_classifier_v2.ipynb
```

Make sure `training_data.csv` and `testing_data.csv` are in the same directory as the notebook.

### What the notebook does

1. Loads both CSVs (tab-separated, no header, `utf-8-sig` encoding)
2. Cleans text — fixes encoding artefacts, lowercases, collapses whitespace
3. Defines the TF-IDF + LinearSVC pipeline
4. Runs 5-fold stratified cross-validation and reports accuracy
5. Trains the final model on the full training set
6. Plots confusion matrix and most informative features (word vs char)
7. Predicts on the test set
8. Saves `testing_data_predicted.csv` in the original format

---

## Output Format

`testing_data_predicted.csv` preserves the exact original format:

- Tab-separated (`\t`)
- No header row
- Two columns: `label` (0 or 1) and `headline`

```
0	copycat muslim terrorist arrested with assault weapons
0	wow! chicago protester caught on camera admits...
1	germany's fdp look to fill schaeuble's big shoes
```

---

## Results

| Metric | Value |
|---|---|
| CV Accuracy (5-fold) | **96.67% ± 0.27%** |
| Fold scores | 96.53 · 97.00 · 96.79 · 96.22 · 96.79 |
| Test: predicted fake (0) | 4,934 (49.4%) |
| Test: predicted real (1) | 5,050 (50.6%) |

---

## Upgrade Path

If you need accuracy above ~98%, here are the natural next steps:

| Approach | Expected accuracy | Training time |
|---|---|---|
| **Current** — TF-IDF + LinearSVC | ~96.7% | < 30s (CPU) |
| `sentence-transformers` embeddings + LR/SVM | ~97–98% | 2–5 min (CPU) |
| Fine-tuned `distilbert-base-uncased` | ~98–99% | 10–30 min (GPU) |
| Fine-tuned `roberta-base` | ~99%+ | 20–45 min (GPU) |

For transformer fine-tuning, see the [HuggingFace Transformers docs](https://huggingface.co/docs/transformers/training).

---

## Data Format Reference

Both input files are tab-separated with no header:

```
<label>\t<headline>
```

- `training_data.csv` — labels are `0` or `1`
- `testing_data.csv` — labels are `2` (placeholder, replaced by the model)

