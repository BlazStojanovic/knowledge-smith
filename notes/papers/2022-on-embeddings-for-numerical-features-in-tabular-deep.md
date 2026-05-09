---
arxiv: '2203.05556'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2203.05556.md
raw_pdf: raw/papers/2203.05556.pdf
read: false
slug: on-embeddings-for-numerical-features-in-tabular-deep
tags: []
title: On Embeddings for Numerical Features in Tabular Deep Learning
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2203.05556v4
venue: null
year: 2022
---

# On Embeddings for Numerical Features in Tabular Deep Learning

> *Yury Gorishniy, Ivan Rubachev, Artem Babenko* — arXiv 2203.05556, 2022

## Abstract

Recently, Transformer-like deep architectures have shown strong performance on tabular data problems. Unlike traditional models, e.g., MLP, these architectures map scalar values of numerical features to high-dimensional embeddings before mixing them in the main backbone. In this work, we argue that embeddings for numerical features are an underexplored degree of freedom in tabular DL, which allows constructing more powerful DL models and competing with GBDT on some traditionally GBDT-friendly benchmarks. We start by describing two conceptually different approaches to building embedding modules: the first one is based on a piecewise linear encoding of scalar values, and the second one utilizes periodic activations. Then, we empirically demonstrate that these two approaches can lead to significant performance boosts compared to the embeddings based on conventional blocks such as linear layers and ReLU activations. Importantly, we also show that embedding numerical features is beneficial for many backbones, not only for Transformers. Specifically, after proper embeddings, simple MLP-like models can perform on par with the attention-based architectures. Overall, we highlight embeddings for numerical features as an important design aspect with good potential for further improvements in tabular DL.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2203.05556]]
- PDF: `raw/papers/2203.05556.pdf`
- arXiv: <http://arxiv.org/abs/2203.05556v4>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2022-gorishniy-numerical-embeddings]]
