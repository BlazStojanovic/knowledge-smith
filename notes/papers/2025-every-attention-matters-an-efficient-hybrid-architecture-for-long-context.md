---
arxiv: '2510.19338'
authors:
- Ling Team
- Bin Han
- Caizhi Tang
- Chen Liang
- Donghao Zhang
- Fan Yuan
- Feng Zhu
- Jie Gao
- Jingyu Hu
- Longfei Li
- Meng Li
- Mingyang Zhang
- Peijie Jiang
- Peng Jiao
- Qian Zhao
- Qingyuan Yang
- Wenbo Shen
- Xinxing Yang
- Yalin Zhang
- Yankun Ren
- Yao Zhao
- Yibo Cao
- Yixuan Sun
- Yue Zhang
- Yuchen Fang
- Zibin Lin
- Zixuan Cheng
- Jun Zhou
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.19338
  raw: '[[raw/papers/md/2025-every-attention-matters-an-efficient-hybrid-architecture-for-long-context]]'
  source: https://arxiv.org/abs/2510.19338
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-every-attention-matters-an-efficient-hybrid-architecture-for-long-context.md
raw_pdf: raw/papers/pdf/2025-every-attention-matters-an-efficient-hybrid-architecture-for-long-context.pdf
read: false
slug: every-attention-matters-an-efficient-hybrid-architecture-for-long-context
tags:
- type/paper
- status/stub
title: 'Every Attention Matters: An Efficient Hybrid Architecture for Long-Context
  Reasoning'
type: note
updated: '2026-05-11'
year: 2025
---

# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning

> *Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.19338>
- PDF: [[raw/papers/pdf/2025-every-attention-matters-an-efficient-hybrid-architecture-for-long-context.pdf]]
- Raw markdown: [[raw/papers/md/2025-every-attention-matters-an-efficient-hybrid-architecture-for-long-context]]
