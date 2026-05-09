---
arxiv: '2106.01342'
authors:
- Gowthami Somepalli
- Micah Goldblum
- Avi Schwarzschild
- C. Bayan Bruss
- Tom Goldstein
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2106.01342.md
raw_pdf: raw/papers/2106.01342.pdf
read: false
slug: saint-improved-neural-networks-for-tabular-data-via-row
tags:
- tabular
- attention
- contrastive
- pretraining
title: 'SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive
  Pre-Training'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2106.01342v1
venue: null
year: 2021
---

# SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training

> *Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild…* — arXiv 2106.01342, 2021

## Abstract

Tabular data underpins numerous high-impact applications of machine learning from fraud detection to genomics and healthcare. Classical approaches to solving tabular problems, such as gradient boosting and random forests, are widely used by practitioners. However, recent deep learning methods have achieved a degree of performance competitive with popular techniques. We devise a hybrid deep learning approach to solving tabular data problems. Our method, SAINT, performs attention over both rows and columns, and it includes an enhanced embedding method. We also study a new contrastive self-supervised pre-training method for use when labels are scarce. SAINT consistently improves performance over previous deep learning methods, and it even outperforms gradient boosting methods, including XGBoost, CatBoost, and LightGBM, on average over a variety of benchmark tasks.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2106.01342]]
- PDF: `raw/papers/2106.01342.pdf`
- arXiv: <http://arxiv.org/abs/2106.01342v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-somepalli-saint.md` before that tree was retired.*

## Core claim

A hybrid attention design with two complementary mechanisms — *self-attention over features* and *intersample attention over rows* — plus a contrastive self-supervised pretraining stage, beats prior tabular DL methods and outperforms gradient-boosting (XGBoost, CatBoost, LightGBM) on average across a benchmark suite. SAINT is the first tabular transformer to pair attention-between-rows with attention-between-columns in the same architecture.

## Architecture

Each SAINT block contains *two* attention layers:

1. **Self-attention over features** (within a row). Standard token-mixing attention over the per-feature embeddings, equivalent to FT-Transformer's [@gorishniy2021ftt] feature attention. Captures cross-feature interactions per instance.
2. **Intersample attention** (across rows). The block treats the rows in a batch as tokens and attends across them — each instance's representation is contextualised by the other instances in its batch. This is the "row attention" of the title and the architectural innovation versus single-instance transformers.

Inputs: every feature is embedded (numerical via per-feature MLP, categorical via lookup) and concatenated with a `[CLS]`-style class token, similar to FT-Transformer.

The self-supervised pretraining stage uses **contrastive learning with CutMix-style and mixup-style augmentations** in feature and embedding space. Two views of each row are produced via random feature corruption + mixup; the contrastive loss pulls embeddings of the same row's two views together and pushes different rows apart. Pretraining helps especially in low-label regimes.

## Key result

- Benchmark covers a mix of UCI and OpenML datasets, plus tasks where prior tabular DL methods had been claimed strong.
- SAINT outperforms TabNet, TabTransformer, and FT-Transformer on most datasets, and outperforms tuned XGBoost, CatBoost, and LightGBM **on average**.
- Contrastive pretraining adds further gains in low-label regimes (semi-supervised setting).

The paper's framing positions SAINT as the first deep tabular method that beats GBDT *on average* — a stronger claim than TabNet or FT-Transformer.

## Why it matters for §2.4.3 (tests of irrelevant-feature robustness)

SAINT is the canonical "intersample-attention" instance — attention is the soft-feature-selector lever (per §2.3), and intersample attention extends it across rows so the model can build a per-batch context similar to NPT's [@kossen2021npt] whole-dataset attention, but at batch granularity. Two diagnostic notes:

- **Two priors at once.** SAINT pulls the §2.3 irrelevant-features lever (column attention as soft feature gating) *and* a partial §2.4.5 instance-based-conditioning lever (row attention pools information across in-batch instances, like a soft k-NN). The dual lever is the architectural reason it outperforms its single-axis predecessors.
- **GBDT-on-average.** "Beats GBDT on average" is the strongest 2021-era claim, but later fair-protocol evaluations (TabZilla [@mcelfresh2023tabzilla], TabReD [@rubachev2025tabred]) shift the standing — SAINT remains competitive but isn't a universal-best DL choice.

## Caveats

- arXiv preprint, presented at the ML4Finance NeurIPS 2021 workshop; not at the main NeurIPS track.
- Pretraining helps in low-label regimes specifically; in fully-supervised settings SAINT's gains are mostly architectural.
- The "outperforms GBDT on average" claim is bench-specific; sample sizes per dataset and HPO budgets matter, and the average can mask per-dataset losses.
