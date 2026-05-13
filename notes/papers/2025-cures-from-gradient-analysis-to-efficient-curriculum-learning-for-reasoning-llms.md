---
arxiv: '2510.01037'
authors:
- Yongcheng Zeng
- Zexu Sun
- Bokai Ji
- Erxue Min
- Hengyi Cai
- Shuaiqiang Wang
- Dawei Yin
- Haifeng Zhang
- Xu Chen
- Jun Wang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.01037
  raw: '[[raw/papers/md/2025-cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms]]'
  source: https://arxiv.org/abs/2510.01037
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms.md
raw_pdf: raw/papers/pdf/2025-cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms.pdf
read: false
slug: cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms
tags:
- type/paper
- status/stub
title: 'CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning
  LLMs'
type: note
updated: '2026-05-11'
year: 2025
---

# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs

> *Yongcheng Zeng, Zexu Sun, Bokai Ji, Erxue Min, Hengyi Cai, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by +3.30 points and +4.82 points with 1.5B and 7B models, respectively, and exceeds the best prior sample efficient methods by +2.12 points on average across eight math reasoning benchmarks. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.01037>
- PDF: [[raw/papers/pdf/2025-cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms.pdf]]
- Raw markdown: [[raw/papers/md/2025-cures-from-gradient-analysis-to-efficient-curriculum-learning-for-reasoning-llms]]
