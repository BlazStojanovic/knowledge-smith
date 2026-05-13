---
arxiv: '2510.11370'
authors:
- Wenhan Ma
- Hailin Zhang
- Liang Zhao
- Yifan Song
- Yudong Wang
- Zhifang Sui
- Fuli Luo
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.11370
  raw: '[[raw/papers/md/2025-stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference]]'
  source: https://arxiv.org/abs/2510.11370
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference.md
raw_pdf: raw/papers/pdf/2025-stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference.pdf
read: false
slug: stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference
tags:
- type/paper
- status/stub
title: Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers
type: note
updated: '2026-05-11'
year: 2025
---

# Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers

> *Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning (RL) has emerged as a crucial approach for enhancing the capabilities of large language models. However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse. We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes. To address this foundational inconsistency, we propose Rollout Routing Replay (R3), a method that records routing distributions from the inference engine and replays them during training. R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed. Extensive experiments on various settings confirm that R3 succeeds in stabilizing RL training, preventing collapse and outperforming methods such as GSPO and TIS. We believe this work can offer a new solution for stabilizing RL in MoE models.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.11370>
- PDF: [[raw/papers/pdf/2025-stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference.pdf]]
- Raw markdown: [[raw/papers/md/2025-stabilizing-moe-reinforcement-learning-by-aligning-training-and-inference]]
