---
arxiv: '2309.17130'
authors:
- Sascha Marton
- Stefan Lüdtke
- Christian Bartelt
- Heiner Stuckenschmidt
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2309.17130.md
raw_pdf: raw/papers/2309.17130.pdf
read: false
slug: grande-gradient-based-decision-tree-ensembles-for-tabular
tags: []
title: 'GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2309.17130v3
venue: null
year: 2023
---

# GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data

> *Sascha Marton, Stefan Lüdtke, Christian Bartelt…* — arXiv 2309.17130, 2023

## Abstract

Despite the success of deep learning for text and image data, tree-based ensemble models are still state-of-the-art for machine learning with heterogeneous tabular data. However, there is a significant need for tabular-specific gradient-based methods due to their high flexibility. In this paper, we propose $\text{GRANDE}$, $\text{GRA}$die$\text{N}$t-Based $\text{D}$ecision Tree $\text{E}$nsembles, a novel approach for learning hard, axis-aligned decision tree ensembles using end-to-end gradient descent. GRANDE is based on a dense representation of tree ensembles, which affords to use backpropagation with a straight-through operator to jointly optimize all model parameters. Our method combines axis-aligned splits, which is a useful inductive bias for tabular data, with the flexibility of gradient-based optimization. Furthermore, we introduce an advanced instance-wise weighting that facilitates learning representations for both, simple and complex relations, within a single model. We conducted an extensive evaluation on a predefined benchmark with 19 classification datasets and demonstrate that our method outperforms existing gradient-boosting and deep learning frameworks on most datasets. The method is available under: https://github.com/s-marton/GRANDE

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2309.17130]]
- PDF: `raw/papers/2309.17130.pdf`
- arXiv: <http://arxiv.org/abs/2309.17130v3>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2024-marton-grande]]
