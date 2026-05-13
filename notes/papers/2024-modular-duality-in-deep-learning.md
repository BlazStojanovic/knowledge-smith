---
arxiv: '2410.21265'
authors:
- Jeremy Bernstein
- Laker Newhouse
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2410.21265
  raw: '[[raw/papers/md/2024-modular-duality-in-deep-learning]]'
  source: https://arxiv.org/abs/2410.21265
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-modular-duality-in-deep-learning.md
raw_pdf: raw/papers/pdf/2024-modular-duality-in-deep-learning.pdf
read: false
slug: modular-duality-in-deep-learning
tags:
- type/paper
- status/stub
title: Modular Duality in Deep Learning
type: note
updated: '2026-05-11'
year: 2024
---

# Modular Duality in Deep Learning

> *Jeremy Bernstein, Laker Newhouse* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

An old idea in optimization theory says that since the gradient is a dual vector it may not be subtracted from the weights without first being mapped to the primal space where the weights reside. We take this idea seriously in this paper and construct such a duality map for general neural networks. Our map, which we call modular dualization, forms a unifying theoretical basis for training algorithms that are a) fast and b) scalable. Modular dualization involves first assigning operator norms to layers based on the semantics of each layer, and then using these layerwise norms to recursively induce a duality map on the weight space of the full neural architecture. We conclude by deriving GPU-friendly algorithms for dualizing Embed, Linear and Conv2D layers -- the latter two methods are based on a rectangular Newton-Schulz iteration (Kovarik, 1970; Björck & Bowie, 1971). A variant of our methods was used to set speed records for training NanoGPT. Overall, we hope that our theory of modular duality will yield a next generation of fast and scalable optimizers for general neural architectures.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2410.21265>
- PDF: [[raw/papers/pdf/2024-modular-duality-in-deep-learning.pdf]]
- Raw markdown: [[raw/papers/md/2024-modular-duality-in-deep-learning]]
