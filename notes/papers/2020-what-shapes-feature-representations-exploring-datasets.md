---
arxiv: '2006.12433'
authors:
- Katherine L. Hermann
- Andrew K. Lampinen
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2020-what-shapes-feature-representations-exploring-datasets.md
raw_pdf: raw/papers/pdf/2020-what-shapes-feature-representations-exploring-datasets.pdf
read: false
slug: what-shapes-feature-representations-exploring-datasets
tags:
- interpretability
- generalization
- feature-encoding
- ml
title: What shapes feature representations? Exploring datasets, architectures, and
  training
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2006.12433v2
venue: null
year: 2020
---

# What shapes feature representations? Exploring datasets, architectures, and training

> *Katherine L. Hermann, Andrew K. Lampinen* — arXiv 2006.12433, 2020

## Abstract

In naturalistic learning problems, a model's input contains a wide range of features, some useful for the task at hand, and others not. Of the useful features, which ones does the model use? Of the task-irrelevant features, which ones does the model represent? Answers to these questions are important for understanding the basis of models' decisions, as well as for building models that learn versatile, adaptable representations useful beyond the original training task. We study these questions using synthetic datasets in which the task-relevance of input features can be controlled directly. We find that when two features redundantly predict the labels, the model preferentially represents one, and its preference reflects what was most linearly decodable from the untrained model. Over training, task-relevant features are enhanced, and task-irrelevant features are partially suppressed. Interestingly, in some cases, an easier, weakly predictive feature can suppress a more strongly predictive, but more difficult one. Additionally, models trained to recognize both easy and hard features learn representations most similar to models that use only the easy feature. Further, easy features lead to more consistent representations across model runs than do hard features. Finally, models have greater representational similarity to an untrained model than to models trained on a different task. Our results highlight the complex processes that determine which features a model represents.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2020-what-shapes-feature-representations-exploring-datasets]]
- PDF: [[raw/papers/pdf/2020-what-shapes-feature-representations-exploring-datasets.pdf]]
- arXiv: <http://arxiv.org/abs/2006.12433v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-katzir-netdnf.md` before that tree was retired.*

Net-DNF — architectural prior mimicking DNF formulas.
