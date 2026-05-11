---
arxiv: '1706.02515'
authors:
- Günter Klambauer
- Thomas Unterthiner
- Andreas Mayr
- Sepp Hochreiter
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1706.02515
  raw: '[[raw/papers/md/2017-self-normalizing-neural-networks]]'
  source: http://arxiv.org/abs/1706.02515v5
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2017-self-normalizing-neural-networks.md
raw_pdf: raw/papers/pdf/2017-self-normalizing-neural-networks.pdf
read: false
slug: self-normalizing-neural-networks
tags:
- type/paper
- ml
- optimization
- generalization
- tabular
- status/stub
title: Self-Normalizing Neural Networks
type: note
updated: '2026-05-09'
venue: null
year: 2017
---

# Self-Normalizing Neural Networks

> *Günter Klambauer, Thomas Unterthiner, Andreas Mayr…* — arXiv 1706.02515, 2017

## Abstract

Deep Learning has revolutionized vision via convolutional neural networks (CNNs) and natural language processing via recurrent neural networks (RNNs). However, success stories of Deep Learning with standard feed-forward neural networks (FNNs) are rare. FNNs that perform well are typically shallow and, therefore cannot exploit many levels of abstract representations. We introduce self-normalizing neural networks (SNNs) to enable high-level abstract representations. While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations. This convergence property of SNNs allows to (1) train deep networks with many layers, (2) employ strong regularization, and (3) to make learning highly robust. Furthermore, for activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible. We compared SNNs on (a) 121 tasks from the UCI machine learning repository, on (b) drug discovery benchmarks, and on (c) astronomy tasks with standard FNNs and other machine learning methods such as random forests and support vector machines. SNNs significantly outperformed all competing FNN methods at 121 UCI tasks, outperformed all competing methods at the Tox21 dataset, and set a new record at an astronomy data set. The winning SNN architectures are often very deep. Implementations are available at: github.com/bioinf-jku/SNNs.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-self-normalizing-neural-networks]]
- PDF (gitignored): [[raw/papers/pdf/2017-self-normalizing-neural-networks.pdf]]
- arXiv: <http://arxiv.org/abs/1706.02515v5>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-klambauer-selu.md` before that tree was retired.*

SELU/SNN — the most-cited "deep nets finally work on tables" paper of 2017; over-promised, under-delivered.
