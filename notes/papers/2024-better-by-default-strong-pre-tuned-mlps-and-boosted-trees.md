---
arxiv: '2407.04491'
authors:
- David Holzmüller
- Léo Grinsztajn
- Ingo Steinwart
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2407.04491.md
raw_pdf: raw/papers/2407.04491.pdf
read: false
slug: better-by-default-strong-pre-tuned-mlps-and-boosted-trees
tags: []
title: 'Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2407.04491v3
venue: null
year: 2024
---

# Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data

> *David Holzmüller, Léo Grinsztajn, Ingo Steinwart* — arXiv 2407.04491, 2024

## Abstract

For classification and regression on tabular data, the dominance of gradient-boosted decision trees (GBDTs) has recently been challenged by often much slower deep learning methods with extensive hyperparameter tuning. We address this discrepancy by introducing (a) RealMLP, an improved multilayer perceptron (MLP), and (b) strong meta-tuned default parameters for GBDTs and RealMLP. We tune RealMLP and the default parameters on a meta-train benchmark with 118 datasets and compare them to hyperparameter-optimized versions on a disjoint meta-test benchmark with 90 datasets, as well as the GBDT-friendly benchmark by Grinsztajn et al. (2022). Our benchmark results on medium-to-large tabular datasets (1K--500K samples) show that RealMLP offers a favorable time-accuracy tradeoff compared to other neural baselines and is competitive with GBDTs in terms of benchmark scores. Moreover, a combination of RealMLP and GBDTs with improved default parameters can achieve excellent results without hyperparameter tuning. Finally, we demonstrate that some of RealMLP's improvements can also considerably improve the performance of TabR with default parameters.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2407.04491]]
- PDF: `raw/papers/2407.04491.pdf`
- arXiv: <http://arxiv.org/abs/2407.04491v3>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2024-holzmuller-realmlp]]
