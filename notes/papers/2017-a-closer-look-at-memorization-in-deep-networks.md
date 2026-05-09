---
arxiv: '1706.05394'
authors:
- Devansh Arpit
- Stanisław Jastrzębski
- Nicolas Ballas
- David Krueger
- Emmanuel Bengio
- Maxinder S. Kanwal
- Tegan Maharaj
- Asja Fischer
- Aaron Courville
- Yoshua Bengio
- Simon Lacoste-Julien
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1706.05394.md
raw_pdf: raw/papers/1706.05394.pdf
read: false
slug: a-closer-look-at-memorization-in-deep-networks
tags:
- generalization
- ml
- theory
- optimization
title: A Closer Look at Memorization in Deep Networks
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1706.05394v2
venue: null
year: 2017
---

# A Closer Look at Memorization in Deep Networks

> *Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas…* — arXiv 1706.05394, 2017

## Abstract

We examine the role of memorization in deep learning, drawing connections to capacity, generalization, and adversarial robustness. While deep networks are capable of memorizing noise data, our results suggest that they tend to prioritize learning simple patterns first. In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks (DNNs) on noise vs. real data. We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data. Our analysis suggests that the notions of effective capacity which are dataset independent are unlikely to explain the generalization performance of deep networks when trained with gradient based methods because training data itself plays an important role in determining the degree of memorization.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1706.05394]]
- PDF (gitignored): `raw/papers/1706.05394.pdf`
- arXiv: <http://arxiv.org/abs/1706.05394v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-arpit-memorization.md` before that tree was retired.*

MLPs have a smoothness/simplicity bias; foundational for why they can't fit jagged tabular surfaces.
