---
arxiv: '2106.02584'
authors:
- Jannik Kossen
- Neil Band
- Clare Lyle
- Aidan N. Gomez
- Tom Rainforth
- Yarin Gal
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2106.02584.md
raw_pdf: raw/papers/2106.02584.pdf
read: false
slug: self-attention-between-datapoints-going-beyond-individual
tags: []
title: 'Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs
  in Deep Learning'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2106.02584v2
venue: null
year: 2021
---

# Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning

> *Jannik Kossen, Neil Band, Clare Lyle…* — arXiv 2106.02584, 2021

## Abstract

We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms. However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction. Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models. We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2106.02584]]
- PDF: `raw/papers/2106.02584.pdf`
- arXiv: <http://arxiv.org/abs/2106.02584v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2021-kossen-npt]]
