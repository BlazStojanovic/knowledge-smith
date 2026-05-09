---
arxiv: '2012.07421'
authors:
- Pang Wei Koh
- Shiori Sagawa
- Henrik Marklund
- Sang Michael Xie
- Marvin Zhang
- Akshay Balsubramani
- Weihua Hu
- Michihiro Yasunaga
- Richard Lanas Phillips
- Irena Gao
- Tony Lee
- Etienne David
- Ian Stavness
- Wei Guo
- Berton A. Earnshaw
- Imran S. Haque
- Sara Beery
- Jure Leskovec
- Anshul Kundaje
- Emma Pierson
- Sergey Levine
- Chelsea Finn
- Percy Liang
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2012.07421.md
raw_pdf: raw/papers/2012.07421.pdf
read: false
slug: wilds-a-benchmark-of-in-the-wild-distribution-shifts
tags: []
title: 'WILDS: A Benchmark of in-the-Wild Distribution Shifts'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2012.07421v3
venue: null
year: 2020
---

# WILDS: A Benchmark of in-the-Wild Distribution Shifts

> *Pang Wei Koh, Shiori Sagawa, Henrik Marklund…* — arXiv 2012.07421, 2020

## Abstract

Distribution shifts -- where the training distribution differs from the test distribution -- can substantially degrade the accuracy of machine learning (ML) systems deployed in the wild. Despite their ubiquity in the real-world deployments, these distribution shifts are under-represented in the datasets widely used in the ML community today. To address this gap, we present WILDS, a curated benchmark of 10 datasets reflecting a diverse range of distribution shifts that naturally arise in real-world applications, such as shifts across hospitals for tumor identification; across camera traps for wildlife monitoring; and across time and location in satellite imaging and poverty mapping. On each dataset, we show that standard training yields substantially lower out-of-distribution than in-distribution performance. This gap remains even with models trained by existing methods for tackling distribution shifts, underscoring the need for new methods for training models that are more robust to the types of distribution shifts that arise in practice. To facilitate method development, we provide an open-source package that automates dataset loading, contains default model architectures and hyperparameters, and standardizes evaluations. Code and leaderboards are available at https://wilds.stanford.edu.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2012.07421]]
- PDF: `raw/papers/2012.07421.pdf`
- arXiv: <http://arxiv.org/abs/2012.07421v3>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2021-koh-wilds]]
