---
arxiv: '1906.02530'
authors:
- Yaniv Ovadia
- Emily Fertig
- Jie Ren
- Zachary Nado
- D Sculley
- Sebastian Nowozin
- Joshua V. Dillon
- Balaji Lakshminarayanan
- Jasper Snoek
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2019-can-you-trust-your-model-s-uncertainty-evaluating.md
raw_pdf: raw/papers/pdf/2019-can-you-trust-your-model-s-uncertainty-evaluating.pdf
read: false
slug: can-you-trust-your-model-s-uncertainty-evaluating
tags:
- uncertainty
- calibration
- benchmark
- ml
title: Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under
  Dataset Shift
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1906.02530v2
venue: null
year: 2019
---

# Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift

> *Yaniv Ovadia, Emily Fertig, Jie Ren…* — arXiv 1906.02530, 2019

## Abstract

Modern machine learning methods including deep learning have achieved great success in predictive accuracy for supervised learning tasks, but may still fall short in giving useful estimates of their predictive {\em uncertainty}. Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model's output should (or should not) be trusted. Many probabilistic deep learning methods, including Bayesian-and non-Bayesian methods, have been proposed in the literature for quantifying predictive uncertainty, but to our knowledge there has not previously been a rigorous large-scale empirical comparison of these methods under dataset shift. We present a large-scale benchmark of existing state-of-the-art methods on classification problems and investigate the effect of dataset shift on accuracy and calibration. We find that traditional post-hoc calibration does indeed fall short, as do several other previous methods. However, some methods that marginalize over models give surprisingly strong results across a broad spectrum of tasks.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2019-can-you-trust-your-model-s-uncertainty-evaluating]]
- PDF: [[raw/papers/pdf/2019-can-you-trust-your-model-s-uncertainty-evaluating.pdf]]
- arXiv: <http://arxiv.org/abs/1906.02530v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2019-ovadia-can-you-trust.md` before that tree was retired.*

Systematic comparison of uncertainty methods under shift; softmax is notoriously poor.
