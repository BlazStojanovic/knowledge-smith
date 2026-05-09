---
arxiv: '2003.04560'
authors:
- Ronen Basri
- Meirav Galun
- Amnon Geifman
- David Jacobs
- Yoni Kasten
- Shira Kritchman
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2003.04560.md
raw_pdf: raw/papers/2003.04560.pdf
read: false
slug: frequency-bias-in-neural-networks-for-input-of-non-uniform
tags: []
title: Frequency Bias in Neural Networks for Input of Non-Uniform Density
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2003.04560v1
venue: null
year: 2020
---

# Frequency Bias in Neural Networks for Input of Non-Uniform Density

> *Ronen Basri, Meirav Galun, Amnon Geifman…* — arXiv 2003.04560, 2020

## Abstract

Recent works have partly attributed the generalization ability of over-parameterized neural networks to frequency bias -- networks trained with gradient descent on data drawn from a uniform distribution find a low frequency fit before high frequency ones. As realistic training sets are not drawn from a uniform distribution, we here use the Neural Tangent Kernel (NTK) model to explore the effect of variable density on training dynamics. Our results, which combine analytic and empirical observations, show that when learning a pure harmonic function of frequency $κ$, convergence at a point $\x \in \Sphere^{d-1}$ occurs in time $O(κ^d/p(\x))$ where $p(\x)$ denotes the local density at $\x$. Specifically, for data in $\Sphere^1$ we analytically derive the eigenfunctions of the kernel associated with the NTK for two-layer networks. We further prove convergence results for deep, fully connected networks with respect to the spectral decomposition of the NTK. Our empirical study highlights similarities and differences between deep and shallow networks in this model.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2003.04560]]
- PDF: `raw/papers/2003.04560.pdf`
- arXiv: <http://arxiv.org/abs/2003.04560v1>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2020-basri-frequency-bias]]
