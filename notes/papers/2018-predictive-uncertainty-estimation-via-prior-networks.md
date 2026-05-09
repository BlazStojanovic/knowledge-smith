---
arxiv: '1802.10501'
authors:
- Andrey Malinin
- Mark Gales
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2018-predictive-uncertainty-estimation-via-prior-networks.md
raw_pdf: raw/papers/pdf/2018-predictive-uncertainty-estimation-via-prior-networks.pdf
read: false
slug: predictive-uncertainty-estimation-via-prior-networks
tags:
- uncertainty
- calibration
- ml
title: Predictive Uncertainty Estimation via Prior Networks
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1802.10501v4
venue: null
year: 2018
---

# Predictive Uncertainty Estimation via Prior Networks

> *Andrey Malinin, Mark Gales* — arXiv 1802.10501, 2018

## Abstract

Estimating how uncertain an AI system is in its predictions is important to improve the safety of such systems. Uncertainty in predictive can result from uncertainty in model parameters, irreducible data uncertainty and uncertainty due to distributional mismatch between the test and training data distributions. Different actions might be taken depending on the source of the uncertainty so it is important to be able to distinguish between them. Recently, baseline tasks and metrics have been defined and several practical methods to estimate uncertainty developed. These methods, however, attempt to model uncertainty due to distributional mismatch either implicitly through model uncertainty or as data uncertainty. This work proposes a new framework for modeling predictive uncertainty called Prior Networks (PNs) which explicitly models distributional uncertainty. PNs do this by parameterizing a prior distribution over predictive distributions. This work focuses on uncertainty for classification and evaluates PNs on the tasks of identifying out-of-distribution (OOD) samples and detecting misclassification on the MNIST dataset, where they are found to outperform previous methods. Experiments on synthetic and MNIST and CIFAR-10 data show that unlike previous non-Bayesian methods PNs are able to distinguish between data and distributional uncertainty.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2018-predictive-uncertainty-estimation-via-prior-networks]]
- PDF: [[raw/papers/pdf/2018-predictive-uncertainty-estimation-via-prior-networks.pdf]]
- arXiv: <http://arxiv.org/abs/1802.10501v4>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2018-malinin-prior-networks.md` before that tree was retired.*

Prior networks with explicit distributional uncertainty.
