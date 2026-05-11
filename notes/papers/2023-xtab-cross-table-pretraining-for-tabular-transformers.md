---
arxiv: '2305.06090'
authors:
- Bingzhao Zhu
- Xingjian Shi
- Nick Erickson
- Mu Li
- George Karypis
- Mahsa Shoaran
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2305.06090
  raw: '[[raw/papers/md/2023-xtab-cross-table-pretraining-for-tabular-transformers]]'
  source: http://arxiv.org/abs/2305.06090v1
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-xtab-cross-table-pretraining-for-tabular-transformers.md
raw_pdf: raw/papers/pdf/2023-xtab-cross-table-pretraining-for-tabular-transformers.pdf
read: false
slug: xtab-cross-table-pretraining-for-tabular-transformers
tags:
- type/paper
- tabular
- pretraining
- transformer
- self-supervised
- status/stub
title: 'XTab: Cross-table Pretraining for Tabular Transformers'
type: note
updated: '2026-05-09'
venue: null
year: 2023
---

# XTab: Cross-table Pretraining for Tabular Transformers

> *Bingzhao Zhu, Xingjian Shi, Nick Erickson…* — arXiv 2305.06090, 2023

## Abstract

The success of self-supervised learning in computer vision and natural language processing has motivated pretraining methods on tabular data. However, most existing tabular self-supervised learning models fail to leverage information across multiple data tables and cannot generalize to new tables. In this work, we introduce XTab, a framework for cross-table pretraining of tabular transformers on datasets from various domains. We address the challenge of inconsistent column types and quantities among tables by utilizing independent featurizers and using federated learning to pretrain the shared component. Tested on 84 tabular prediction tasks from the OpenML-AutoML Benchmark (AMLB), we show that (1) XTab consistently boosts the generalizability, learning speed, and performance of multiple tabular transformers, (2) by pretraining FT-Transformer via XTab, we achieve superior performance than other state-of-the-art tabular deep learning models on various tasks such as regression, binary, and multiclass classification.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2023-xtab-cross-table-pretraining-for-tabular-transformers]]
- PDF: [[raw/papers/pdf/2023-xtab-cross-table-pretraining-for-tabular-transformers.pdf]]
- arXiv: <http://arxiv.org/abs/2305.06090v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-zhu-xtab.md` before that tree was retired.*

XTab — cross-table pretraining for tabular transformers.
