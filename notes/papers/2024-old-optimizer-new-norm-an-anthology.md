---
arxiv: '2409.20325'
authors:
- Jeremy Bernstein
- Laker Newhouse
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2024-old-optimizer-new-norm-an-anthology.md
raw_pdf: raw/papers/pdf/2024-old-optimizer-new-norm-an-anthology.pdf
read: false
slug: old-optimizer-new-norm-an-anthology
tags:
- optimization
- theory
title: 'Old Optimizer, New Norm: An Anthology'
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2409.20325
venue: null
year: 2024
---

# Old Optimizer, New Norm: An Anthology

> *Jeremy Bernstein, Laker Newhouse* — arXiv 2409.20325, 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Deep learning optimizers are often motivated through a mix of convex and approximate second-order theory. The authors select three methods -- Adam, Shampoo and Prodigy -- and argue that each can be understood as a first-order method without convexity assumptions. After switching off exponential moving averages, each method is equivalent to steepest descent under a particular norm. By generalizing this observation, the paper charts a new design space for training algorithms. Different operator norms should be assigned to different tensors based on the role that the tensor plays within the network. For example, while linear and embedding layers may have the same weight space, these layers play different roles and should be assigned different norms. The authors propose that carefully metrizing the neural architecture might lead to more stable, scalable and faster training.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2024-old-optimizer-new-norm-an-anthology]]
- PDF: [[raw/papers/pdf/2024-old-optimizer-new-norm-an-anthology.pdf]]
- arXiv: <https://arxiv.org/abs/2409.20325>
