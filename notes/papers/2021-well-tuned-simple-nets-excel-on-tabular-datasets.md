---
arxiv: '2106.11189'
authors:
- Arlind Kadra
- Marius Lindauer
- Frank Hutter
- Josif Grabocka
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2106.11189
  raw: '[[raw/papers/md/2021-well-tuned-simple-nets-excel-on-tabular-datasets]]'
  source: http://arxiv.org/abs/2106.11189v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-well-tuned-simple-nets-excel-on-tabular-datasets.md
raw_pdf: raw/papers/pdf/2021-well-tuned-simple-nets-excel-on-tabular-datasets.pdf
read: false
slug: well-tuned-simple-nets-excel-on-tabular-datasets
tags:
- type/paper
- tabular
- optimization
- generalization
- gradient-boosting
- status/stub
title: Well-tuned Simple Nets Excel on Tabular Datasets
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# Well-tuned Simple Nets Excel on Tabular Datasets

> *Arlind Kadra, Marius Lindauer, Frank Hutter…* — arXiv 2106.11189, 2021

## Abstract

Tabular datasets are the last "unconquered castle" for deep learning, with traditional ML methods like Gradient-Boosted Decision Trees still performing strongly even against recent specialized neural architectures. In this paper, we hypothesize that the key to boosting the performance of neural networks lies in rethinking the joint and simultaneous application of a large set of modern regularization techniques. As a result, we propose regularizing plain Multilayer Perceptron (MLP) networks by searching for the optimal combination/cocktail of 13 regularization techniques for each dataset using a joint optimization over the decision on which regularizers to apply and their subsidiary hyperparameters. We empirically assess the impact of these regularization cocktails for MLPs in a large-scale empirical study comprising 40 tabular datasets and demonstrate that (i) well-regularized plain MLPs significantly outperform recent state-of-the-art specialized neural network architectures, and (ii) they even outperform strong traditional ML methods, such as XGBoost.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-well-tuned-simple-nets-excel-on-tabular-datasets]]
- PDF: [[raw/papers/pdf/2021-well-tuned-simple-nets-excel-on-tabular-datasets.pdf]]
- arXiv: <http://arxiv.org/abs/2106.11189v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-kadra-well-tuned-nets.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub had arXiv ID 2106.03253; the actual ID is **2106.11189** per the bib entry and verified abstract.

- **ArXiv:** 2106.11189
- **Authors:** Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka
- **Year:** 2021
- **Venue:** NeurIPS 2021
- **Raw:** [[raw/papers/2021-kadra-well-tuned-nets.pdf]]

## Core claim

A plain MLP, jointly tuned over a *cocktail* of 13 modern regularisation techniques, **outperforms specialised tabular DL architectures and outperforms XGBoost** on a 40-dataset benchmark. Tabular DL had been losing to GBDTs not because deep networks lacked expressivity but because the field had been comparing under-regularised MLPs to bespoke architectures and tuned trees. The recipe lives in the regularisation, not the backbone.

## Architecture / Method

The architecture is a vanilla MLP. The methodological contribution is a *joint hyperparameter optimisation* over (a) which regularisers to apply and (b) their continuous hyperparameters. The 13 regularisers cover:

- **Weight decay / L2 / L1**, classic shrinkage.
- **Dropout** (standard + variational).
- **Batch / Layer Normalization.**
- **Mixup** (input-space and feature-space convex combinations).
- **Stochastic Depth.**
- **Standard data augmentation** (per-feature jitter where applicable).
- **Lookahead** optimizer wrapper.
- **Snapshot ensembling.**
- **Early stopping.**
- **Knowledge distillation** (from a tuned tree-based model where applicable).

A Bayesian optimiser searches over the joint space of (binary inclusion, continuous hyperparameters) per dataset. The search budget is comparable to what specialised tabular DL papers use for their own architectures.

## Key result

- **40 tabular datasets** from OpenML's CC-18 + complementary collections.
- Well-regularised MLP (the "regularisation cocktail") outperforms TabNet, NODE, MLP, ResNet, and tuned XGBoost on the suite.
- The headline finding: **architecture innovation is not the dominant variable; regularisation tuning is.**

## Why it matters for §2.4.4 (the methodological correction)

Kadra et al. is the most-cited single piece of evidence that "the deep-tabular gap was a tuning gap." For §2.4.4 it provides the controlled experiment:

- **Hold the architecture constant** (plain MLP).
- **Vary the regularisation discipline** (cocktail vs. standard).
- **Observe** that the cocktail closes — and on this benchmark, beats — the GBDT and specialised-DL gap.

This is the controlled experiment §2.4.4 needs. The paper lands awkwardly for the architectural-innovation school (it implies many of the architectural wins were tuning effects), and is later confirmed and extended by RealMLP [@holzmuller2024realmlp] (meta-learned defaults), TabM [@gorishniy2025tabm] (parameter-efficient ensembling), and Zabergja et al. [@zabergja2024nlpinspired] (NLP-inspired tabular tricks fail under fair comparison).

The diagnostic reading: §2.1, §2.2, §2.3 still characterise real inductive-bias problems, but the architecture-zoo's verdict is "the right *training procedure* on a plain MLP closes more of the gap than the wrong training procedure on a specialised architecture."

## Caveats

- The 40-dataset suite skews toward small-to-medium problems where the cocktail's variance-reducing effects are most powerful. Behaviour on production-scale tabular workloads (millions of rows) is not the paper's focus.
- "Beats XGBoost" is on this benchmark; later fair-protocol benchmarks (TabReD [@rubachev2025tabred], TabArena [@erickson2025tabarena]) re-rank methods and the cocktail-MLP's relative position depends on the specific HPO budget used per comparator.
- The cocktail's HPO budget is not free — joint search over 13 regularisers' inclusion + hyperparameters is expensive, and "well-tuned MLP" hides that cost. Practitioners should treat the cocktail as evidence that *the tuning ceiling is high*, not as a drop-in replacement.
