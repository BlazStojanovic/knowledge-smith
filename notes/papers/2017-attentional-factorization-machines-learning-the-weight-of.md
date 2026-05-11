---
arxiv: '1708.04617'
authors:
- Jun Xiao
- Hao Ye
- Xiangnan He
- Hanwang Zhang
- Fei Wu
- Tat-Seng Chua
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1708.04617
  raw: '[[raw/papers/md/2017-attentional-factorization-machines-learning-the-weight-of]]'
  source: http://arxiv.org/abs/1708.04617v1
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2017-attentional-factorization-machines-learning-the-weight-of.md
raw_pdf: raw/papers/pdf/2017-attentional-factorization-machines-learning-the-weight-of.pdf
read: false
slug: attentional-factorization-machines-learning-the-weight-of
tags:
- type/paper
- recsys
- attention
- ctr-prediction
- feature-encoding
- status/stub
title: 'Attentional Factorization Machines: Learning the Weight of Feature Interactions
  via Attention Networks'
type: note
updated: '2026-05-09'
venue: null
year: 2017
---

# Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

> *Jun Xiao, Hao Ye, Xiangnan He…* — arXiv 1708.04617, 2017

## Abstract

Factorization Machines (FMs) are a supervised learning approach that enhances the linear regression model by incorporating the second-order feature interactions. Despite effectiveness, FM can be hindered by its modelling of all feature interactions with the same weight, as not all feature interactions are equally useful and predictive. For example, the interactions with useless features may even introduce noises and adversely degrade the performance. In this work, we improve FM by discriminating the importance of different feature interactions. We propose a novel model named Attentional Factorization Machine (AFM), which learns the importance of each feature interaction from data via a neural attention network. Extensive experiments on two real-world datasets demonstrate the effectiveness of AFM. Empirically, it is shown on regression task AFM betters FM with a $8.6\%$ relative improvement, and consistently outperforms the state-of-the-art deep learning methods Wide&Deep and DeepCross with a much simpler structure and fewer model parameters. Our implementation of AFM is publicly available at: https://github.com/hexiangnan/attentional_factorization_machine

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-attentional-factorization-machines-learning-the-weight-of]]
- PDF: [[raw/papers/pdf/2017-attentional-factorization-machines-learning-the-weight-of.pdf]]
- arXiv: <http://arxiv.org/abs/1708.04617v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-xiao-afm.md` before that tree was retired.*

Attentional Factorization Machine with attention weights on pairwise interactions.
