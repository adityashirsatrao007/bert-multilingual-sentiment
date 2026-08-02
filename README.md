<div align="center">

# Multilingual Sentiment Analysis with BERT

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICCTWC68241.2026.11583557-blue)](https://doi.org/10.1109/ICCTWC68241.2026.11583557)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-ICCTWC%202026-orange)](https://doi.org/10.1109/ICCTWC68241.2026.11583557)
[![Scopus](https://img.shields.io/badge/Scopus-Indexed-green)](https://www.scopus.com)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/adityashirsatrao007/bert-multilingual-sentiment/actions/workflows/main.yml/badge.svg)](https://github.com/adityashirsatrao007/bert-multilingual-sentiment/actions)

**Deep Multilingual Sentiment Analysis using Uncased BERT Representations** —
official implementation of our peer-reviewed paper at **IEEE ICCTWC 2026**
(Scopus-indexed). Fine-tuned `bert-base-multilingual-uncased` for 5-class
multilingual sentiment classification.

</div>

---

## Publication

This repository is the official implementation of:

> V. A. Sangolgi, **A. V. Shirsatrao**, and V. V. Hibare, *"Deep Multilingual
> Sentiment Analysis using Uncased BERT Representations,"* 2026 International
> Conference on Computing Theory and Wireless Communications (ICCTWC),
> Ichalkaranji, India, pp. 1–6, doi: [10.1109/ICCTWC68241.2026.11583557](https://doi.org/10.1109/ICCTWC68241.2026.11583557).

- **DOI:** [10.1109/ICCTWC68241.2026.11583557](https://doi.org/10.1109/ICCTWC68241.2026.11583557)
- **Indexing:** IEEE Xplore · Scopus
- **Cite:** see [`CITATION.cff`](CITATION.cff) and [`PUBLICATION.md`](PUBLICATION.md) (full details + BibTeX)

---

## Overview

Multilingual sentiment classification from product reviews across languages
using uncased multilingual BERT representations. The model performs **5-class
sentiment classification** (1–5 stars) with strong cross-lingual generalization,
and the evaluation harness produces research-grade plots for publication.

### Model

- **Base model:** [`bert-base-multilingual-uncased`](https://huggingface.co/bert-base-multilingual-uncased)
- **Task:** `BertForSequenceClassification`, 5 classes (1–5 star)
- **Fine-tuned weights:** shipped in [`model/`](model/)
- **Framework:** PyTorch + Hugging Face Transformers

### Repository Structure

```
├── model/                    # Fine-tuned mBERT weights (config, tokenizer, weights)
├── evaluate_model.py         # Evaluation harness → metrics + publication plots
├── requirements.txt          # Python dependencies
├── LITERATURE_REVIEW.md      # Survey of related multilingual sentiment work
├── PUBLICATION.md            # Paper details, citation, BibTeX
├── CITATION.cff              # Machine-readable citation metadata
├── LICENSE                   # MIT License
└── Dockerfile                # Containerized evaluation
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Run Evaluation

```bash
python evaluate_model.py
```

Generates metrics and plots:
- **Accuracy:** 90.00%
- **Weighted F1:** 0.8571
- `confusion_matrix.png` — class-wise predictions
- `roc_curve.png` — one-vs-rest ROC curves
- `precision_recall_curve.png` — precision/recall per class
- `class_performance.png` — precision, recall, F1 per class
- `confidence_histogram.png` — prediction confidence distribution
- `metrics_summary.png` — overall performance summary

---

## Results

| Metric | Value |
|---|---|
| **Accuracy** | 90.00% |
| **Weighted F1** | 0.8571 |

Model weights are pre-downloaded in [`model/`](model/) so evaluation runs
offline. Plots included in the repository root document the reported results.

---

## Citing

If you use this code or model, please cite:

```bibtex
@inproceedings{sangolgi2026deep,
  author    = {Sangolgi, Vijay A. and Shirsatrao, Aditya Vishal and Hibare, Viha Vikram},
  title     = {Deep Multilingual Sentiment Analysis using Uncased {BERT} Representations},
  booktitle = {2026 International Conference on Computing Theory and Wireless Communications (ICCTWC)},
  address   = {Ichalkaranji, India},
  pages     = {1--6},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICCTWC68241.2026.11583557}
}
```

See [`PUBLICATION.md`](PUBLICATION.md) for full details.

## License

[MIT](LICENSE)
