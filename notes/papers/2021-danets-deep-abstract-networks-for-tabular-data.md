---
arxiv: '2112.02962'
authors:
- Jintai Chen
- Kuanlun Liao
- Yao Wan
- Danny Z. Chen
- Jian Wu
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2112.02962
  raw: '[[raw/papers/md/2021-danets-deep-abstract-networks-for-tabular-data]]'
  source: http://arxiv.org/abs/2112.02962v4
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-danets-deep-abstract-networks-for-tabular-data.md
raw_pdf: raw/papers/pdf/2021-danets-deep-abstract-networks-for-tabular-data.pdf
read: false
slug: danets-deep-abstract-networks-for-tabular-data
tags:
- type/paper
- tabular
- ml
- feature-encoding
- status/stub
title: 'DANets: Deep Abstract Networks for Tabular Data Classification and Regression'
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# DANets: Deep Abstract Networks for Tabular Data Classification and Regression

> *Jintai Chen, Kuanlun Liao, Yao Wan…* — arXiv 2112.02962, 2021

## Abstract

Tabular data are ubiquitous in real world applications. Although many commonly-used neural components (e.g., convolution) and extensible neural networks (e.g., ResNet) have been developed by the machine learning community, few of them were effective for tabular data and few designs were adequately tailored for tabular data structures. In this paper, we propose a novel and flexible neural component for tabular data, called Abstract Layer (AbstLay), which learns to explicitly group correlative input features and generate higher-level features for semantics abstraction. Also, we design a structure re-parameterization method to compress the learned AbstLay, thus reducing the computational complexity by a clear margin in the reference phase. A special basic block is built using AbstLays, and we construct a family of Deep Abstract Networks (DANets) for tabular data classification and regression by stacking such blocks. In DANets, a special shortcut path is introduced to fetch information from raw tabular features, assisting feature interactions across different levels. Comprehensive experiments on seven real-world tabular datasets show that our AbstLay and DANets are effective for tabular data classification and regression, and the computational complexity is superior to competitive methods. Besides, we evaluate the performance gains of DANet as it goes deep, verifying the extendibility of our method. Our code is available at https://github.com/WhatAShot/DANet.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-danets-deep-abstract-networks-for-tabular-data]]
- PDF: [[raw/papers/pdf/2021-danets-deep-abstract-networks-for-tabular-data.pdf]]
- arXiv: <http://arxiv.org/abs/2112.02962v4>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-chen-danets.md` before that tree was retired.*

DANets — deep abstract layers with sparse feature grouping.
