---
arxiv: '2307.14338'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Nikolay Kartashev
- Daniil Shlenskii
- Akim Kotelnikov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2307.14338.md
raw_pdf: raw/papers/2307.14338.pdf
read: false
slug: tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023
tags: []
title: 'TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2307.14338v2
venue: null
year: 2023
---

# TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023

> *Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev…* — arXiv 2307.14338, 2023

## Abstract

Deep learning (DL) models for tabular data problems (e.g. classification, regression) are currently receiving increasingly more attention from researchers. However, despite the recent efforts, the non-DL algorithms based on gradient-boosted decision trees (GBDT) remain a strong go-to solution for these problems. One of the research directions aimed at improving the position of tabular DL involves designing so-called retrieval-augmented models. For a target object, such models retrieve other objects (e.g. the nearest neighbors) from the available training data and use their features and labels to make a better prediction.
  In this work, we present TabR -- essentially, a feed-forward network with a custom k-Nearest-Neighbors-like component in the middle. On a set of public benchmarks with datasets up to several million objects, TabR marks a big step forward for tabular DL: it demonstrates the best average performance among tabular DL models, becomes the new state-of-the-art on several datasets, and even outperforms GBDT models on the recently proposed "GBDT-friendly" benchmark (see Figure 1). Among the important findings and technical details powering TabR, the main ones lie in the attention-like mechanism that is responsible for retrieving the nearest neighbors and extracting valuable signal from them. In addition to the much higher performance, TabR is simple and significantly more efficient compared to prior retrieval-based tabular DL models.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2307.14338]]
- PDF: `raw/papers/2307.14338.pdf`
- arXiv: <http://arxiv.org/abs/2307.14338v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2024-gorishniy-tabr]]
