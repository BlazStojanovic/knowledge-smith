---
arxiv: '2107.07511'
authors:
- Anastasios N. Angelopoulos
- Stephen Bates
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2021-a-gentle-introduction-to-conformal-prediction-and.md
raw_pdf: raw/papers/pdf/2021-a-gentle-introduction-to-conformal-prediction-and.pdf
read: false
slug: a-gentle-introduction-to-conformal-prediction-and
tags:
- uncertainty
- calibration
- distribution-free
- survey
title: A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
  Quantification
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2107.07511v6
venue: null
year: 2021
---

# A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification

> *Anastasios N. Angelopoulos, Stephen Bates* — arXiv 2107.07511, 2021

## Abstract

Black-box machine learning models are now routinely used in high-risk settings, like medical diagnostics, which demand uncertainty quantification to avoid consequential model failures. Conformal prediction is a user-friendly paradigm for creating statistically rigorous uncertainty sets/intervals for the predictions of such models. Critically, the sets are valid in a distribution-free sense: they possess explicit, non-asymptotic guarantees even without distributional assumptions or model assumptions. One can use conformal prediction with any pre-trained model, such as a neural network, to produce sets that are guaranteed to contain the ground truth with a user-specified probability, such as 90%. It is easy-to-understand, easy-to-use, and general, applying naturally to problems arising in the fields of computer vision, natural language processing, deep reinforcement learning, and so on.
  This hands-on introduction is aimed to provide the reader a working understanding of conformal prediction and related distribution-free uncertainty quantification techniques with one self-contained document. We lead the reader through practical theory for and examples of conformal prediction and describe its extensions to complex machine learning tasks involving structured outputs, distribution shift, time-series, outliers, models that abstain, and more. Throughout, there are many explanatory illustrations, examples, and code samples in Python. With each code sample comes a Jupyter notebook implementing the method on a real-data example; the notebooks can be accessed and easily run using our codebase.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-a-gentle-introduction-to-conformal-prediction-and]]
- PDF: [[raw/papers/pdf/2021-a-gentle-introduction-to-conformal-prediction-and.pdf]]
- arXiv: <http://arxiv.org/abs/2107.07511v6>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-angelopoulos-conformal.md` before that tree was retired.*

Conformal prediction — distribution-free coverage wrapper applicable to GBDTs.
