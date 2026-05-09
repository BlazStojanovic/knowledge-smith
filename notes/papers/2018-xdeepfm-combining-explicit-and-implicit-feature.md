---
arxiv: '1803.05170'
authors:
- Jianxun Lian
- Xiaohuan Zhou
- Fuzheng Zhang
- Zhongxia Chen
- Xing Xie
- Guangzhong Sun
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1803.05170.md
raw_pdf: raw/papers/1803.05170.pdf
read: false
slug: xdeepfm-combining-explicit-and-implicit-feature
tags:
- recsys
- ctr-prediction
- feature-encoding
- ml
title: 'xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender
  Systems'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1803.05170v3
venue: null
year: 2018
---

# xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

> *Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang…* — arXiv 1803.05170, 2018

## Abstract

Combinatorial features are essential for the success of many commercial models. Manually crafting these features usually comes with high cost due to the variety, volume and velocity of raw data in web-scale systems. Factorization based models, which measure interactions in terms of vector product, can learn patterns of combinatorial features automatically and generalize to unseen features as well. With the great success of deep neural networks (DNNs) in various fields, recently researchers have proposed several DNN-based factorization model to learn both low- and high-order feature interactions. Despite the powerful ability of learning an arbitrary function from data, plain DNNs generate feature interactions implicitly and at the bit-wise level. In this paper, we propose a novel Compressed Interaction Network (CIN), which aims to generate feature interactions in an explicit fashion and at the vector-wise level. We show that the CIN share some functionalities with convolutional neural networks (CNNs) and recurrent neural networks (RNNs). We further combine a CIN and a classical DNN into one unified model, and named this new model eXtreme Deep Factorization Machine (xDeepFM). On one hand, the xDeepFM is able to learn certain bounded-degree feature interactions explicitly; on the other hand, it can learn arbitrary low- and high-order feature interactions implicitly. We conduct comprehensive experiments on three real-world datasets. Our results demonstrate that xDeepFM outperforms state-of-the-art models. We have released the source code of xDeepFM at \url{https://github.com/Leavingseason/xDeepFM}.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1803.05170]]
- PDF: `raw/papers/1803.05170.pdf`
- arXiv: <http://arxiv.org/abs/1803.05170v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2018-lian-xdeepfm.md` before that tree was retired.*

xDeepFM — Compressed Interaction Network for high-order explicit feature crosses.
