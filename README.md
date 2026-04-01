# Fake News Headline Classifier

| | Notebook 1 | Notebook 2 |
|---|---|---|
| **File** | `fake_news_classifier_v2.ipynb` | `fake_news_classifier_DistilBERT.ipynb` |
| **Model** | TF-IDF (word + char n-grams) + LinearSVC | DistilBERT fine-tuned |
| **Accuracy** | **96.67%** (5-fold CV ± 0.27%) | **98.19%** (val accuracy, epoch 2) |
| **Environment** | Local / Jupyter — no GPU needed | Google Colab — free T4 GPU |
| **Training time** | < 30 seconds | ~7 minutes |
| **Interpretable** | Yes — inspect LinearSVC coefficients | No — transformer black-box |
| **Output file** | `testing_data_predicted.csv` | `testing_data_predicted_distilBERT.csv` |

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
├── training_data.csv                          # Labeled training set (tab-sep, no header)
├── testing_data.csv                           # Unlabeled test set (label column = 2)
│
├── fake_news_classifier_v2.ipynb              # Notebook 1: TF-IDF + LinearSVC
├── fake_news_classifier_DistilBERT.ipynb      # Notebook 2: DistilBERT fine-tuning
│
├── testing_data_predicted.csv                 # Output from Notebook 1
├── testing_data_predicted_distilBERT.csv      # Output from Notebook 2
│
├── fake_news_nlp.pptx                         # PPT presentation Summary
│
└── README.md
```

---

## Notebook 1 — TF-IDF + LinearSVC

### When to use this
- You want results in under a minute on any machine
- You need to understand *why* a headline was classified a certain way (inspect model coefficients)
- You don't have GPU access

### Requirements

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
```

### How to run

```bash
jupyter notebook fake_news_classifier_v2.ipynb
```

Make sure `training_data.csv` and `testing_data.csv` are in the same directory as the notebook.

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

**Key insight:** Adding character n-grams (+1.79 pp over baseline) is far more effective than ensembling (+0.31 pp). Character n-grams capture stylistic fingerprints — ALL-CAPS, punctuation runs (`!!!`), sensationalist suffixes — that word tokens miss entirely. `char_wb` pads word boundaries with spaces so n-grams don't bleed across words.

### Benchmark results (5-fold stratified CV)

| Model | CV Accuracy | Δ vs baseline |
|---|---|---|
| TF-IDF (word 1-2g) + Logistic Regression *(baseline)* | 94.88% | — |
| TF-IDF (word 1-2g) + LinearSVC | 94.99% | +0.11 pp |
| Soft Voting ensemble: LR + SVC + NB | 95.19% | +0.31 pp |
| TF-IDF (word+char) + Logistic Regression | 96.52% | +1.64 pp |
| **TF-IDF (word+char) + LinearSVC ✓** | **96.67%** | **+1.79 pp** |

---

## Notebook 2 — DistilBERT Fine-tuning

### When to use this
- You want maximum accuracy (~98%)
- You have access to a GPU (Google Colab T4 is free and sufficient)
- You are comfortable with HuggingFace Transformers

### Environment

This notebook is designed for **Google Colab**. Before running:
1. Open in Colab and go to **Runtime → Change runtime type → T4 GPU**
2. Upload `training_data.csv` and `testing_data.csv` via the file manager (folder icon in the left sidebar) or by running the upload cell

### Architecture

```
Raw headline
    │
    ▼
Text cleaning          # lowercase · fix encoding artefacts
    │
    ▼
DistilBERT Tokenizer   # max_length=64 · padding=True · truncation=True
    │
    ▼
distilbert-base-uncased
    │
    ▼
Classification head    # 2-class linear layer (randomly initialized, learned from scratch)
    │
    ▼
0 (fake) or 1 (real)
```

### Model choice: why DistilBERT?

`distilbert-base-uncased` retains 97% of BERT's performance while being 40% smaller and 60% faster. The `uncased` variant lowercases all input — consistent with our cleaning step. We fine-tune the full model (not just the classification head) so the internal attention weights adapt to the specific patterns of fake news headlines.

### Key training decisions

| Argument | Value | Why |
|---|---|---|
| `max_length` | 64 | Headlines are short (< 50 tokens). 512 would waste ~8× GPU memory. |
| `num_train_epochs` | 2 | Val accuracy plateaued at epoch 2 (98.19%). More epochs risk overfitting. |
| `per_device_train_batch_size` | 32 | Fits in T4 VRAM (16 GB) with max_length=64. |
| `warmup_steps` | 100 | Ramps LR from 0 to avoid large updates on the randomly-initialized head. |
| `weight_decay` | 0.1 | L2 regularisation across all non-bias weights. |
| `load_best_model_at_end` | `True` | Restores the best checkpoint (lowest val loss) after training. |

### Training results

| Epoch | Train Loss | Val Loss | Val Accuracy |
|---|---|---|---|
| 1 | 0.020 | 0.090 | 98.07% |
| **2** | **0.009** | **0.082** | **98.19%** ✓ |

Training time: ~7 minutes on Colab T4.

---

## Output Format

Both output files preserve the exact original format:

- Tab-separated (`\t`)
- No header row
- Two columns: `label` (0 or 1) and `headline`

```
0	copycat muslim terrorist arrested with assault weapons
1	germany's fdp look to fill schaeuble's big shoes
```

---

## Choosing Between the Two Notebooks

```
Do you have a GPU / Colab access?
        │
        ├── No  →  Use Notebook 1 (TF-IDF + LinearSVC)
        │          96.67% · < 30s · fully interpretable
        │
        └── Yes →  Do you need maximum accuracy?
                        │
                        ├── No  →  Notebook 1 is still great
                        │
                        └── Yes →  Use Notebook 2 (DistilBERT)
                                   98.19% · ~7 min · Colab T4
```

---

## Data Format Reference

Both input files are tab-separated with no header:

```
<label>\t<headline>
```

- `training_data.csv` — labels are `0` (fake) or `1` (real)
- `testing_data.csv` — labels are `2` (placeholder, replaced by the model)

