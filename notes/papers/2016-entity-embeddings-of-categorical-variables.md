---
arxiv: '1604.06737'
authors:
- Cheng Guo
- Felix Berkhahn
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1604.06737.md
raw_pdf: raw/papers/1604.06737.pdf
read: false
slug: entity-embeddings-of-categorical-variables
tags:
- tabular
- feature-encoding
- deep-learning
title: Entity Embeddings of Categorical Variables
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1604.06737v1
venue: null
year: 2016
---

# Entity Embeddings of Categorical Variables

> *Cheng Guo, Felix Berkhahn* — arXiv 1604.06737, 2016

## Abstract

We map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables. The mapping is learned by a neural network during the standard supervised training process. Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables. We applied it successfully in a recent Kaggle competition and were able to reach the third position with relative simple features. We further demonstrate in this paper that entity embedding helps the neural network to generalize better when the data is sparse and statistics is unknown. Thus it is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit. We also demonstrate that the embeddings obtained from the trained neural network boost the performance of all tested machine learning methods considerably when used as the input features instead. As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1604.06737]]
- PDF (gitignored): `raw/papers/1604.06737.pdf`
- arXiv: <http://arxiv.org/abs/1604.06737v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2016-guo-entity-embeddings.md` before that tree was retired.*

Seminal entity embeddings for categoricals from the Rossmann Kaggle winner; still baked into every modern tabular net.
