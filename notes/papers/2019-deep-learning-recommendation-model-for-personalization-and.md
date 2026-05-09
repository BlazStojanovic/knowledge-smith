---
arxiv: '1906.00091'
authors:
- Maxim Naumov
- Dheevatsa Mudigere
- Hao-Jun Michael Shi
- Jianyu Huang
- Narayanan Sundaraman
- Jongsoo Park
- Xiaodong Wang
- Udit Gupta
- Carole-Jean Wu
- Alisson G. Azzolini
- Dmytro Dzhulgakov
- Andrey Mallevich
- Ilia Cherniavskii
- Yinghai Lu
- Raghuraman Krishnamoorthi
- Ansha Yu
- Volodymyr Kondratenko
- Stephanie Pereira
- Xianjie Chen
- Wenlin Chen
- Vijay Rao
- Bill Jia
- Liang Xiong
- Misha Smelyanskiy
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2019-deep-learning-recommendation-model-for-personalization-and.md
raw_pdf: raw/papers/pdf/2019-deep-learning-recommendation-model-for-personalization-and.pdf
read: false
slug: deep-learning-recommendation-model-for-personalization-and
tags:
- recsys
- ctr-prediction
- ml
- feature-encoding
title: Deep Learning Recommendation Model for Personalization and Recommendation Systems
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1906.00091v1
venue: null
year: 2019
---

# Deep Learning Recommendation Model for Personalization and Recommendation Systems

> *Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi…* — arXiv 1906.00091, 2019

## Abstract

With the advent of deep learning, neural network-based recommendation models have emerged as an important tool for tackling personalization and recommendation tasks. These networks differ significantly from other deep learning networks due to their need to handle categorical features and are not well studied or understood. In this paper, we develop a state-of-the-art deep learning recommendation model (DLRM) and provide its implementation in both PyTorch and Caffe2 frameworks. In addition, we design a specialized parallelization scheme utilizing model parallelism on the embedding tables to mitigate memory constraints while exploiting data parallelism to scale-out compute from the fully-connected layers. We compare DLRM against existing recommendation models and characterize its performance on the Big Basin AI platform, demonstrating its usefulness as a benchmark for future algorithmic experimentation and system co-design.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2019-deep-learning-recommendation-model-for-personalization-and]]
- PDF: `raw/papers/pdf/2019-deep-learning-recommendation-model-for-personalization-and.pdf`
- arXiv: <http://arxiv.org/abs/1906.00091v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2019-naumov-dlrm.md` before that tree was retired.*

Meta's DLRM — open-source reference for billion-scale recsys tables.
