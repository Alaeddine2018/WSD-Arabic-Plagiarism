# WSD-Enhanced Arabic Plagiarism Detection

A complete, single-file experiment pipeline for **Arabic plagiarism detection enhanced with Word Sense Disambiguation (WSD)**, built on the AraPlagDet / ExAra corpus and the Arabic WordNet (AWN).

The pipeline trains and compares transformer baselines (AraBERT, MARBERT) against CNN and BiLSTM encoders augmented with WSD-derived sense features, and produces all paper-ready tables: main results, ablation studies, WSD-coverage stratified analysis, bootstrap significance tests, and confusion matrices.

---

## Table of Contents

1. [Dataset Information](#dataset-information)
2. [Code Overview](#code-overview)
3. [Methodology](#methodology)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Repository Structure](#repository-structure)
8. [Citations](#citations)
9. [License](#license)

---

## Dataset Information

| Resource | Description | Source |
|----------|-------------|--------|
| **AraPlagDet / ExAra Corpus** | Evaluation corpus for Arabic external plagiarism detection; used in the AraPlagDet PAN@FIRE 2015 shared task. Contains suspicious and source document pairs with plagiarism annotations. | [Zenodo DOI 10.5281/zenodo.6607799](https://doi.org/10.5281/zenodo.6607799) |
| **AWN.txt** | Arabic WordNet synonym sets in CSV form (`word,synonym1,synonym2,...`), used for WSD-based feature extraction and data augmentation. | Included in this repository |

> **Note:** The AraPlagDet corpus is **not** redistributed here, in line with the authors' distribution terms. Download it from the official DOI above, place it as `AraPlagDet.zip` in the repository root, and cite Bensalem et al. (2015) (see [Citations](#citations)).

---

## Code Overview

`run_experiment.py` runs the entire pipeline end to end:

| Stage | Description |
|-------|-------------|
| Part 0 | Arabic preprocessing (diacritics removal, normalization) and Arabic WordNet loading |
| Part 1 | Extraction of suspicious/source sentence pairs from the AraPlagDet corpus |
| Part 2 | Data augmentation (synonym replacement via AWN, word/phrase shuffling, token dropout) — 3× training data |
| Part 3 | Model definitions: CNN encoder, BiLSTM encoder, WSD-augmented plagiarism detector, BERT classifiers |
| Part 4 | Training of all models (AraBERT, MARBERT, CNN±WSD, BiLSTM±WSD) |
| Part 5 | Evaluation: ablation studies, bootstrap significance testing (5,000 iterations), WSD-coverage strata |
| Part 6 | Saving results (`results/all_results.json`, `.npy` predictions) and printing formatted tables |

---

## Methodology

### Preprocessing
Arabic text is normalized by removing diacritics (tashkeel), standardizing alef/ya/ta-marbuta variants, and tokenizing by whitespace after punctuation removal.

### WSD Feature Extraction
The `ArabicWordNet` class loads AWN.txt and provides:
- **Synonym lookup** per word
- **Sense overlap** between two word lists (Jaccard-based)
- **WSD feature vectors** (dimension: 50) encoding sense overlap, coverage ratio, and synonym co-occurrence between a suspicious/source sentence pair

### Data Augmentation (3× multiplier)
Four strategies applied to the training set:
- **Synonym replacement** — replace 30% of content words with AWN synonyms
- **Word shuffle** — local reordering within a window of 3 tokens
- **Phrase shuffle** — reordering of comma/conjunction-separated phrases
- **Token dropout** — random deletion of 15% of tokens

### Models

| Model | Architecture |
|-------|-------------|
| **AraBERT** | `aubmindlab/bert-base-arabertv02` fine-tuned as binary classifier (hidden: 256, dropout: 0.3) |
| **MARBERT** | `UBC-NLP/MARBERT` fine-tuned as binary classifier (hidden: 256, dropout: 0.3) |
| **CNN** | Character/token n-gram CNN encoder (filters: 100, kernels: [2,3,4]) → interaction features → classifier |
| **CNN+WSD** | CNN encoder + 50-dim WSD feature vector concatenated before classifier |
| **BiLSTM** | 2-layer bidirectional LSTM (hidden: 128) → interaction features → classifier |
| **BiLSTM+WSD** | BiLSTM encoder + 50-dim WSD feature vector concatenated before classifier |

All encoder models use an element-wise difference + product interaction layer, a 256-unit hidden classifier, and 0.5 dropout. Training: Adam (lr=1e-3), BERT models: AdamW (lr=2e-5), batch size 32, max 15 epochs, early stopping (patience=3).

### Evaluation
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score** (macro)
- 5-fold cross-validation
- Bootstrap significance testing (5,000 iterations) between model pairs
- Stratified analysis by WSD coverage (low / medium / high)

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU recommended (expected runtime: **3–5 hours** on an RTX-class GPU)

Key libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 1.13 | Model training |
| `transformers` | ≥ 4.30 | AraBERT / MARBERT |
| `arabert` | ≥ 1.0 | AraBERT tokenizer |
| `scikit-learn` | ≥ 1.2 | Metrics, cross-validation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Data handling |
| `tqdm` | ≥ 4.65 | Progress bars |
| `matplotlib` / `seaborn` | ≥ 3.7 / 0.12 | Visualizations |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare the data
Download the AraPlagDet corpus from [https://doi.org/10.5281/zenodo.6607799](https://doi.org/10.5281/zenodo.6607799) and place it in the repository root:

```
wsd-arabic-plagiarism/
├── AraPlagDet.zip      ← place here
├── AWN.txt
├── run_experiment.py
└── ...
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the pipeline

```bash
python run_experiment.py
```

The script will automatically:
1. Extract and preprocess the corpus
2. Build and augment sentence pairs
3. Train all six models
4. Evaluate and print formatted result tables

### Outputs

| File | Description |
|------|-------------|
| `results/all_results.json` | All metrics for every model and analysis |
| `results/preds_*.npy` | Raw model predictions for reproducibility |
| `results/y_true.npy` | Ground-truth labels |
| Stdout | Main results table, ablation, significance tests, confusion matrix |

---

## Configuration

All hyperparameters are in the `Config` class at the top of `run_experiment.py`:

```python
BATCH_SIZE      = 32
MAX_EPOCHS      = 15
PATIENCE        = 3          # early stopping
LR              = 1e-3       # encoder models
BERT_LR         = 2e-5       # BERT models
BILSTM_HIDDEN   = 128
CNN_FILTERS     = 100
WSD_DIM         = 50
DROPOUT         = 0.5
N_FOLDS         = 5
BOOTSTRAP_N     = 5000
AUG_MULTIPLIER  = 3.0
SEED            = 42
```

---

## Repository Structure

```
.
├── run_experiment.py     # Complete experiment pipeline (single file)
├── AWN.txt               # Arabic WordNet synonym sets
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
├── AraPlagDet.zip        # ← not included; obtain from Zenodo
├── data/                 # Generated: extracted corpus + sentence pairs
├── results/              # Generated: metrics, predictions, tables
└── saved_models/         # Generated: model checkpoints
```

---

## Citations

If you use this code, please cite the accompanying paper (to be updated upon publication):

```bibtex
@article{wsd_arabic_plagiarism,
  title   = {WSD-Enhanced Arabic Plagiarism Detection},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

If you use the ExAra / AraPlagDet corpus, its authors request citing:

```bibtex
@inproceedings{bensalem2015araplagdet,
  title     = {Overview of the {AraPlagDet} {PAN}@{FIRE}2015 Shared Task on Arabic Plagiarism Detection},
  author    = {Bensalem, Imene and Boukhalfa, Imene and Rosso, Paolo and Abouenour, Lahsen, and Darwish, Kareem and Chikhi, Salim},
  booktitle = {Post Proceedings of the Workshops at the 7th Forum for Information Retrieval Evaluation (FIRE 2015)},
  series    = {CEUR Workshop Proceedings},
  volume    = {1587},
  pages     = {111--122},
  year      = {2015}
}
```

---

## License

This code is released under the **MIT License** (see `LICENSE`).  
The ExAra/AraPlagDet corpus and the Arabic WordNet are subject to their own respective licenses; please obtain them from their official sources.
