# WSD-Enhanced Arabic Plagiarism Detection

A complete, single-file experiment pipeline for **Arabic plagiarism detection enhanced with Word Sense Disambiguation (WSD)**, built on the AraPlagDet corpus and the Arabic WordNet (AWN).

The pipeline trains and compares transformer baselines (AraBERT, MARBERT) against CNN and BiLSTM encoders augmented with WSD-derived sense features, and produces all paper-ready tables: main results, ablations, WSD-coverage stratified analysis, bootstrap significance tests, and confusion matrices.

## Pipeline Overview

`run_experiment.py` runs everything end to end:

| Stage | Description |
|-------|-------------|
| Part 0 | Arabic preprocessing (diacritics removal, normalization) and Arabic WordNet loading |
| Part 1 | Extraction of suspicious/source sentence pairs from the AraPlagDet corpus |
| Part 2 | Data augmentation (synonym replacement via AWN, word/phrase shuffling, dropout) — 3× training data |
| Part 3 | Model definitions: CNN encoder, BiLSTM encoder, plagiarism detector with WSD features, BERT classifiers |
| Part 4 | Training of all models (AraBERT, MARBERT, CNN±WSD, BiLSTM±WSD) |
| Part 5 | Evaluation: ablation studies, bootstrap significance testing (5,000 iterations), WSD-coverage strata |
| Part 6 | Saving results (`results/all_results.json`, `.npy` predictions) and printing formatted tables |

## Requirements

- Python ≥ 3.9
- A CUDA-capable GPU is strongly recommended (expected runtime: **3–5 hours** on an RTX-class GPU)

```bash
pip install -r requirements.txt
```

## Data

1. **AraPlagDet / ExAra Corpus** — the Evaluation Corpus for Arabic External Plagiarism Detection, used in the AraPlagDet PAN@FIRE 2015 shared task. Download it from its official Zenodo record — [https://doi.org/10.5281/zenodo.6607799](https://doi.org/10.5281/zenodo.6607799) — and place it as `AraPlagDet.zip` in the repository root. *The corpus is intentionally **not** redistributed here, in line with its authors' distribution terms; please obtain it from the official DOI and cite Bensalem et al. (2015) below.*
2. **AWN.txt** — Arabic WordNet synonym sets in CSV form (`word,synonym1,synonym2,...`), included in this repository for the WSD component.

## Usage

```bash
python run_experiment.py
```

The script automatically extracts the corpus, builds and augments sentence pairs, trains all models, and writes:

- `results/all_results.json` — all metrics for every model and analysis
- `results/preds_*.npy`, `results/y_true.npy` — raw predictions for reproducibility
- Formatted tables printed to stdout (main results, ablation, significance, strata, confusion matrix)

## Configuration

All hyperparameters live in the `Config` class at the top of `run_experiment.py` (batch size, learning rates, BiLSTM/CNN dimensions, WSD vector dimension, augmentation ratios, seed).

## Repository Structure

```
.
├── run_experiment.py     # Complete experiment pipeline (single file)
├── AWN.txt               # Arabic WordNet synonym sets
├── requirements.txt      # Python dependencies
├── AraPlagDet.zip        # (not included — obtain separately)
├── data/                 # Generated: extracted corpus + sentence pairs
├── results/              # Generated: metrics, predictions, tables
└── saved_models/         # Generated: model checkpoints
```

## Citation

If you use this code, please cite the accompanying paper (reference to be added upon publication):

```bibtex
@article{wsd_arabic_plagiarism,
  title   = {WSD-Enhanced Arabic Plagiarism Detection},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

If you use the ExAra corpus, its authors request citing:

```bibtex
@inproceedings{bensalem2015araplagdet,
  title     = {Overview of the {AraPlagDet} {PAN}@{FIRE}2015 Shared Task on Arabic Plagiarism Detection},
  author    = {Bensalem, Imene and Boukhalfa, Imene and Rosso, Paolo and Abouenour, Lahsen and Darwish, Kareem and Chikhi, Salim},
  booktitle = {Post Proceedings of the Workshops at the 7th Forum for Information Retrieval Evaluation (FIRE 2015)},
  series    = {CEUR Workshop Proceedings},
  volume    = {1587},
  pages     = {111--122},
  year      = {2015}
}
```

## License

This code is released under the MIT License (see `LICENSE`). The ExAra/AraPlagDet corpus and the Arabic WordNet are subject to their own respective licenses and distribution terms; they should be obtained from their official sources.
