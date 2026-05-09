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
parser: ar5iv
raw_md: raw/papers/2305.06090.md
raw_pdf: raw/papers/2305.06090.pdf
read: false
slug: xtab-cross-table-pretraining-for-tabular-transformers
tags: []
title: 'XTab: Cross-table Pretraining for Tabular Transformers'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2305.06090v1
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

- Raw markdown: [[raw/papers/2305.06090]]
- PDF: `raw/papers/2305.06090.pdf`
- arXiv: <http://arxiv.org/abs/2305.06090v1>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2023-zhu-xtab]]
