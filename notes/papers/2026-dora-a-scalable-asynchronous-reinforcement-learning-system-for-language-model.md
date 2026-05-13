---
arxiv: '2604.26256'
authors:
- Tianhao Hu
- Xiangcheng Liu
- Youshao Xiao
- Yang Zheng
- Xuan Huang
- Jinrui Ding
- Yufei Zhang
- Tao Liang
- Hongyu Zang
- Quan Chen
- Yueqing Sun
- Wenjie Shi
- Chao Zhang
- Wei Wang
- Qi Gu
- Yerui Sun
- Yucheng Xie
- Xunliang Cai
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.26256
  raw: '[[raw/papers/md/2026-dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model]]'
  source: https://arxiv.org/abs/2604.26256
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model.md
raw_pdf: raw/papers/pdf/2026-dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model.pdf
read: false
slug: dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model
tags:
- type/paper
- status/stub
title: 'DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model
  Training'
type: note
updated: '2026-05-11'
year: 2026
---

# DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training

> *Tianhao Hu, Xiangcheng Liu, Youshao Xiao, Yang Zheng, Xuan Huang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning (RL) has become a critical paradigm for LLM post-training, yet the rollout phase -- accounting for 50--80% of total step time -- is bottlenecked by skewed generation: long-tailed trajectories indispensable for model performance block the entire training pipeline. Asynchronous training offers a natural remedy by overlapping generation with training, but introduces a fundamental tension between efficiency and algorithmic correctness. We identify three constraints in asynchronous training to preserve convergence: intra-trajectory policy consistency, data integrity, and bounded staleness. Existing approaches fail to intrinsically address the long-tailed trajectory problem, which is further exacerbated by the imbalance characteristic of Mix-of-Experts models, or deviate from the standard RL training formulation, thereby hindering model convergence. Therefore, we propose DORA (Dynamic ORchestration for Asynchronous Rollout), which addresses this challenge through algorithm-system co-design. DORA introduces multi-version streaming rollout, a novel asynchronous paradigm that maintains multiple policy versions concurrently -- simultaneously achieving full bubble elimination without compromising algorithmic constraints. Experimental results demonstrate that our DORA system achieves substantial improvements in throughput -- up to 2--3 times higher than state-of-the-art systems on open-source benchmarks -- without compromising convergence. Furthermore, in large-scale industrial applications with tens of thousands of accelerators, DORA accelerates RL training by 2--4 times compared to synchronous training across various scenarios. The resultant open-source models, LongCat-Flash-Thinking, exhibit competitive performance on complex reasoning benchmarks, matching the capability of most advanced LLMs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.26256>
- PDF: [[raw/papers/pdf/2026-dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model.pdf]]
- Raw markdown: [[raw/papers/md/2026-dora-a-scalable-asynchronous-reinforcement-learning-system-for-language-model]]
