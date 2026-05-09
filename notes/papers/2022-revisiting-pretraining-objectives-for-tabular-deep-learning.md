---
arxiv: '2207.03208'
authors:
- Ivan Rubachev
- Artem Alekberov
- Yury Gorishniy
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2207.03208.md
raw_pdf: raw/papers/2207.03208.pdf
read: false
slug: revisiting-pretraining-objectives-for-tabular-deep-learning
tags: []
title: Revisiting Pretraining Objectives for Tabular Deep Learning
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2207.03208v2
venue: null
year: 2022
---

# Revisiting Pretraining Objectives for Tabular Deep Learning

> *Ivan Rubachev, Artem Alekberov, Yury Gorishniy…* — arXiv 2207.03208, 2022

## Abstract

Recent deep learning models for tabular data currently compete with the traditional ML models based on decision trees (GBDT). Unlike GBDT, deep models can additionally benefit from pretraining, which is a workhorse of DL for vision and NLP. For tabular problems, several pretraining methods were proposed, but it is not entirely clear if pretraining provides consistent noticeable improvements and what method should be used, since the methods are often not compared to each other or comparison is limited to the simplest MLP architectures.
  In this work, we aim to identify the best practices to pretrain tabular DL models that can be universally applied to different datasets and architectures. Among our findings, we show that using the object target labels during the pretraining stage is beneficial for the downstream performance and advocate several target-aware pretraining objectives. Overall, our experiments demonstrate that properly performed pretraining significantly increases the performance of tabular DL models, which often leads to their superiority over GBDTs.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2207.03208]]
- PDF: `raw/papers/2207.03208.pdf`
- arXiv: <http://arxiv.org/abs/2207.03208v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2022-rubachev-revisiting-pretraining]]
