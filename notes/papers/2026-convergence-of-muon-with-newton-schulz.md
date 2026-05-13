---
arxiv: '2601.19156'
authors:
- Gyu Yeol Kim
- Min-hwan Oh
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.19156
  raw: '[[raw/papers/md/2026-convergence-of-muon-with-newton-schulz]]'
  source: https://arxiv.org/abs/2601.19156
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-convergence-of-muon-with-newton-schulz.md
raw_pdf: raw/papers/pdf/2026-convergence-of-muon-with-newton-schulz.pdf
read: false
slug: convergence-of-muon-with-newton-schulz
tags:
- type/paper
- status/stub
title: Convergence of Muon with Newton-Schulz
type: note
updated: '2026-05-11'
year: 2026
---

# Convergence of Muon with Newton-Schulz

> *Gyu Yeol Kim, Min-hwan Oh* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We analyze Muon as originally proposed and used in practice -- using the momentum orthogonalization with a few Newton-Schulz steps. The prior theoretical results replace this key step in Muon with an exact SVD-based polar factor. We prove that Muon with Newton-Schulz converges to a stationary point at the same rate as the SVD-polar idealization, up to a constant factor for a given number $q$ of Newton-Schulz steps. We further analyze this constant factor and prove that it converges to 1 doubly exponentially in $q$ and improves with the degree of the polynomial used in Newton-Schulz for approximating the orthogonalization direction. We also prove that Muon removes the typical square-root-of-rank loss compared to its vector-based counterpart, SGD with momentum. Our results explain why Muon with a few low-degree Newton-Schulz steps matches exact-polar (SVD) behavior at a much faster wall-clock time and explain how much momentum matrix orthogonalization via Newton-Schulz benefits over the vector-based optimizer. Overall, our theory justifies the practical Newton-Schulz design of Muon, narrowing its practice-theory gap.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.19156>
- PDF: [[raw/papers/pdf/2026-convergence-of-muon-with-newton-schulz.pdf]]
- Raw markdown: [[raw/papers/md/2026-convergence-of-muon-with-newton-schulz]]
