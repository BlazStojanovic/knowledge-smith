---
arxiv: '1612.01474'
authors:
- Balaji Lakshminarayanan
- Alexander Pritzel
- Charles Blundell
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1612.01474.md
raw_pdf: raw/papers/1612.01474.pdf
read: false
slug: simple-and-scalable-predictive-uncertainty-estimation-using
tags: []
title: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1612.01474v3
venue: null
year: 2016
---

# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

> *Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell* — arXiv 1612.01474, 2016

## Abstract

Deep neural networks (NNs) are powerful black box predictors that have recently achieved impressive performance on a wide spectrum of tasks. Quantifying predictive uncertainty in NNs is a challenging and yet unsolved problem. Bayesian NNs, which learn a distribution over weights, are currently the state-of-the-art for estimating predictive uncertainty; however these require significant modifications to the training procedure and are computationally expensive compared to standard (non-Bayesian) NNs. We propose an alternative to Bayesian NNs that is simple to implement, readily parallelizable, requires very little hyperparameter tuning, and yields high quality predictive uncertainty estimates. Through a series of experiments on classification and regression benchmarks, we demonstrate that our method produces well-calibrated uncertainty estimates which are as good or better than approximate Bayesian NNs. To assess robustness to dataset shift, we evaluate the predictive uncertainty on test examples from known and unknown distributions, and show that our method is able to express higher uncertainty on out-of-distribution examples. We demonstrate the scalability of our method by evaluating predictive uncertainty estimates on ImageNet.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1612.01474]]
- PDF (gitignored): `raw/papers/1612.01474.pdf`
- arXiv: <http://arxiv.org/abs/1612.01474v3>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2017-lakshminarayanan-deep-ensembles]]
