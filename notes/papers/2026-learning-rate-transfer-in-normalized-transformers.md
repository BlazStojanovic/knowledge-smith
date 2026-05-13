---
arxiv: '2604.27077'
authors:
- Boris Shigida
- Boris Hanin
- Andrey Gromov
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.27077
  raw: '[[raw/papers/md/2026-learning-rate-transfer-in-normalized-transformers]]'
  source: https://arxiv.org/abs/2604.27077
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-learning-rate-transfer-in-normalized-transformers.md
raw_pdf: raw/papers/pdf/2026-learning-rate-transfer-in-normalized-transformers.pdf
read: false
slug: learning-rate-transfer-in-normalized-transformers
tags:
- type/paper
- status/stub
title: Learning Rate Transfer in Normalized Transformers
type: note
updated: '2026-05-11'
year: 2026
---

# Learning Rate Transfer in Normalized Transformers

> *Boris Shigida, Boris Hanin, Andrey Gromov* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

The Normalized Transformer, or nGPT (arXiv:2410.01131) achieves impressive training speedups and does not require weight decay or learning rate warmup. However, despite having hyperparameters that explicitly scale with model size, we observe that nGPT does not exhibit learning rate transfer across model dimension and token horizon. To rectify this, we combine numerical experiments with a principled use of alignment exponents (arXiv:2407.05872) to revisit and modify the $μ$P approach to hyperparameter transfer (arXiv:2011.14522). The result is a novel nGPT parameterization we call $ν$GPT. Through extensive empirical validation, we find $ν$GPT exhibits learning rate transfer across width, depth, and token horizon.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.27077>
- PDF: [[raw/papers/pdf/2026-learning-rate-transfer-in-normalized-transformers.pdf]]
- Raw markdown: [[raw/papers/md/2026-learning-rate-transfer-in-normalized-transformers]]
