---
arxiv: '1611.00144'
authors:
- Yanru Qu
- Han Cai
- Kan Ren
- Weinan Zhang
- Yong Yu
- Ying Wen
- Jun Wang
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1611.00144.md
raw_pdf: raw/papers/1611.00144.pdf
read: false
slug: product-based-neural-networks-for-user-response-prediction
tags:
- ctr-prediction
- recsys
- feature-encoding
- tabular
title: Product-based Neural Networks for User Response Prediction
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1611.00144v1
venue: null
year: 2016
---

# Product-based Neural Networks for User Response Prediction

> *Yanru Qu, Han Cai, Kan Ren…* — arXiv 1611.00144, 2016

## Abstract

Predicting user responses, such as clicks and conversions, is of great importance and has found its usage in many Web applications including recommender systems, web search and online advertising. The data in those applications is mostly categorical and contains multiple fields; a typical representation is to transform it into a high-dimensional sparse binary feature representation via one-hot encoding. Facing with the extreme sparsity, traditional models may limit their capacity of mining shallow patterns from the data, i.e. low-order feature combinations. Deep models like deep neural networks, on the other hand, cannot be directly applied for the high-dimensional input because of the huge feature space. In this paper, we propose a Product-based Neural Networks (PNN) with an embedding layer to learn a distributed representation of the categorical data, a product layer to capture interactive patterns between inter-field categories, and further fully connected layers to explore high-order feature interactions. Our experimental results on two large-scale real-world ad click datasets demonstrate that PNNs consistently outperform the state-of-the-art models on various metrics.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1611.00144]]
- PDF (gitignored): `raw/papers/1611.00144.pdf`
- arXiv: <http://arxiv.org/abs/1611.00144v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2016-qu-pnn.md` before that tree was retired.*

Product-based Neural Network; explicit inner/outer product interaction layers.
