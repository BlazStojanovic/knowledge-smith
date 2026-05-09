---
arxiv: '2207.08815'
authors:
- Léo Grinsztajn
- Edouard Oyallon
- Gaël Varoquaux
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2207.08815.md
raw_pdf: raw/papers/2207.08815.pdf
read: false
slug: why-do-tree-based-models-still-outperform-deep-learning-on
tags: []
title: Why do tree-based models still outperform deep learning on tabular data?
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2207.08815v1
venue: null
year: 2022
---

# Why do tree-based models still outperform deep learning on tabular data?

> *Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux* — arXiv 2207.08815, 2022

## Abstract

While deep learning has enabled tremendous progress on text and image datasets, its superiority on tabular data is not clear. We contribute extensive benchmarks of standard and novel deep learning methods as well as tree-based models such as XGBoost and Random Forests, across a large number of datasets and hyperparameter combinations. We define a standard set of 45 datasets from varied domains with clear characteristics of tabular data and a benchmarking methodology accounting for both fitting models and finding good hyperparameters. Results show that tree-based models remain state-of-the-art on medium-sized data ($\sim$10K samples) even without accounting for their superior speed. To understand this gap, we conduct an empirical investigation into the differing inductive biases of tree-based models and Neural Networks (NNs). This leads to a series of challenges which should guide researchers aiming to build tabular-specific NNs: 1. be robust to uninformative features, 2. preserve the orientation of the data, and 3. be able to easily learn irregular functions. To stimulate research on tabular architectures, we contribute a standard benchmark and raw data for baselines: every point of a 20 000 compute hours hyperparameter search for each learner.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2207.08815]]
- PDF: `raw/papers/2207.08815.pdf`
- arXiv: <http://arxiv.org/abs/2207.08815v1>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2022-shwartz-ziv-tabular]]
