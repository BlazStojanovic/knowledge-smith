---
arxiv: '2107.01830'
authors:
- Shaofeng Cai
- Kaiping Zheng
- Gang Chen
- H. V. Jagadish
- Beng Chin Ooi
- Meihui Zhang
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2107.01830
  raw: '[[raw/papers/md/2021-arm-net-adaptive-relation-modeling-network-for-structured]]'
  source: http://arxiv.org/abs/2107.01830v1
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-arm-net-adaptive-relation-modeling-network-for-structured.md
raw_pdf: raw/papers/pdf/2021-arm-net-adaptive-relation-modeling-network-for-structured.pdf
read: false
slug: arm-net-adaptive-relation-modeling-network-for-structured
tags:
- type/paper
- tabular
- attention
- feature-encoding
- interpretability
- status/stub
title: 'ARM-Net: Adaptive Relation Modeling Network for Structured Data'
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# ARM-Net: Adaptive Relation Modeling Network for Structured Data

> *Shaofeng Cai, Kaiping Zheng, Gang Chen…* — arXiv 2107.01830, 2021

## Abstract

Relational databases are the de facto standard for storing and querying structured data, and extracting insights from structured data requires advanced analytics. Deep neural networks (DNNs) have achieved super-human prediction performance in particular data types, e.g., images. However, existing DNNs may not produce meaningful results when applied to structured data. The reason is that there are correlations and dependencies across combinations of attribute values in a table, and these do not follow simple additive patterns that can be easily mimicked by a DNN. The number of possible such cross features is combinatorial, making them computationally prohibitive to model. Furthermore, the deployment of learning models in real-world applications has also highlighted the need for interpretability, especially for high-stakes applications, which remains another issue of concern to DNNs.
  In this paper, we present ARM-Net, an adaptive relation modeling network tailored for structured data, and a lightweight framework ARMOR based on ARM-Net for relational data analytics. The key idea is to model feature interactions with cross features selectively and dynamically, by first transforming the input features into exponential space, and then determining the interaction order and interaction weights adaptively for each cross feature. We propose a novel sparse attention mechanism to dynamically generate the interaction weights given the input tuple, so that we can explicitly model cross features of arbitrary orders with noisy features filtered selectively. Then during model inference, ARM-Net can specify the cross features being used for each prediction for higher accuracy and better interpretability. Our extensive experiments on real-world datasets demonstrate that ARM-Net consistently outperforms existing models and provides more interpretable predictions for data-driven decision making.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-arm-net-adaptive-relation-modeling-network-for-structured]]
- PDF: [[raw/papers/pdf/2021-arm-net-adaptive-relation-modeling-network-for-structured.pdf]]
- arXiv: <http://arxiv.org/abs/2107.01830v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-cai-armnet.md` before that tree was retired.*

An adaptive feature-interaction model that selects per-instance cross-features through a sparse attention layer, framing tabular prediction as relation modeling — an early academic transformer-on-tables instance from the database community, predating TabTransformer and FT-Transformer.
