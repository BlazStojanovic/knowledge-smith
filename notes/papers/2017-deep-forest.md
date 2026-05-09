---
arxiv: '1702.08835'
authors:
- Zhi-Hua Zhou
- Ji Feng
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2017-deep-forest.md
raw_pdf: raw/papers/pdf/2017-deep-forest.pdf
read: false
slug: deep-forest
tags:
- deep-forest
- decision-tree
- tabular
- ml
title: Deep Forest
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1702.08835v4
venue: null
year: 2017
---

# Deep Forest

> *Zhi-Hua Zhou, Ji Feng* — arXiv 1702.08835, 2017

## Abstract

Current deep learning models are mostly build upon neural networks, i.e., multiple layers of parameterized differentiable nonlinear modules that can be trained by backpropagation. In this paper, we explore the possibility of building deep models based on non-differentiable modules. We conjecture that the mystery behind the success of deep neural networks owes much to three characteristics, i.e., layer-by-layer processing, in-model feature transformation and sufficient model complexity. We propose the gcForest approach, which generates \textit{deep forest} holding these characteristics. This is a decision tree ensemble approach, with much less hyper-parameters than deep neural networks, and its model complexity can be automatically determined in a data-dependent way. Experiments show that its performance is quite robust to hyper-parameter settings, such that in most cases, even across different data from different domains, it is able to get excellent performance by using the same default setting. This study opens the door of deep learning based on non-differentiable modules, and exhibits the possibility of constructing deep models without using backpropagation.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-deep-forest]]
- PDF: `raw/papers/pdf/2017-deep-forest.pdf`
- arXiv: <http://arxiv.org/abs/1702.08835v4>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-zhou-deep-forest.md` before that tree was retired.*

gcForest — "deep" via cascaded forests, not backprop.
