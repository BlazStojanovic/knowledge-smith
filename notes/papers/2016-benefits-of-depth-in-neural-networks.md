---
arxiv: '1602.04485'
authors:
- Matus Telgarsky
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1602.04485
  raw: '[[raw/papers/md/2016-benefits-of-depth-in-neural-networks]]'
  source: http://arxiv.org/abs/1602.04485v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2016-benefits-of-depth-in-neural-networks.md
raw_pdf: raw/papers/pdf/2016-benefits-of-depth-in-neural-networks.pdf
read: false
slug: benefits-of-depth-in-neural-networks
tags:
- type/paper
- theory
- deep-learning
- generalization
- status/stub
title: Benefits of depth in neural networks
type: note
updated: '2026-05-09'
venue: null
year: 2016
---

# Benefits of depth in neural networks

> *Matus Telgarsky* — arXiv 1602.04485, 2016

## Abstract

For any positive integer $k$, there exist neural networks with $Θ(k^3)$ layers, $Θ(1)$ nodes per layer, and $Θ(1)$ distinct parameters which can not be approximated by networks with $\mathcal{O}(k)$ layers unless they are exponentially large --- they must possess $Ω(2^k)$ nodes. This result is proved here for a class of nodes termed "semi-algebraic gates" which includes the common choices of ReLU, maximum, indicator, and piecewise polynomial functions, therefore establishing benefits of depth against not just standard networks with ReLU gates, but also convolutional networks with ReLU and maximization gates, sum-product networks, and boosted decision trees (in this last case with a stronger separation: $Ω(2^{k^3})$ total tree nodes are required).

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2016-benefits-of-depth-in-neural-networks]]
- PDF (gitignored): [[raw/papers/pdf/2016-benefits-of-depth-in-neural-networks.pdf]]
- arXiv: <http://arxiv.org/abs/1602.04485v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2016-telgarsky-benefits-of-depth.md` before that tree was retired.*

Constructs a $\Theta(k^3)$-layer ReLU network whose function any $o(k)$-layer network requires $\Omega(2^k)$ width to approximate, formalizing a depth-vs-width separation theorem that anchors expressivity-side arguments about why deep networks can represent functions shallow ones cannot.
