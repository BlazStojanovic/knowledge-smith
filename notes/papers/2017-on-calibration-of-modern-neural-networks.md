---
arxiv: '1706.04599'
authors:
- Chuan Guo
- Geoff Pleiss
- Yu Sun
- Kilian Q. Weinberger
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1706.04599.md
raw_pdf: raw/papers/1706.04599.pdf
read: false
slug: on-calibration-of-modern-neural-networks
tags: []
title: On Calibration of Modern Neural Networks
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1706.04599v2
venue: null
year: 2017
---

# On Calibration of Modern Neural Networks

> *Chuan Guo, Geoff Pleiss, Yu Sun…* — arXiv 1706.04599, 2017

## Abstract

Confidence calibration -- the problem of predicting probability estimates representative of the true correctness likelihood -- is important for classification models in many applications. We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated. Through extensive experiments, we observe that depth, width, weight decay, and Batch Normalization are important factors influencing calibration. We evaluate the performance of various post-processing calibration methods on state-of-the-art architectures with image and document classification datasets. Our analysis and experiments not only offer insights into neural network learning, but also provide a simple and straightforward recipe for practical settings: on most datasets, temperature scaling -- a single-parameter variant of Platt Scaling -- is surprisingly effective at calibrating predictions.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1706.04599]]
- PDF (gitignored): `raw/papers/1706.04599.pdf`
- arXiv: <http://arxiv.org/abs/1706.04599v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2017-guo-calibration]]
