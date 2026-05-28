---
arxiv: '2605.25604'
authors:
- Guochao Jiang
- Jingyi Song
- Guofeng Quan
- Chuzhan Hao
- Guohua Liu
- Yuewei Zhang
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.25604
  raw: '[[raw/papers/md/2026-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward]]'
  source: https://arxiv.org/abs/2605.25604
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward.md
raw_pdf: raw/papers/pdf/2026-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward.pdf
read: false
slug: dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward
tags:
- type/paper
- status/stub
title: 'DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement
  Learning'
type: note
updated: '2026-05-28'
year: 2026
---

# DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning

> *Guochao Jiang, Jingyi Song, Guofeng Quan, Chuzhan Hao, Guohua Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement Learning has become a standard paradigm for aligning Large Language Models with human intent and task requirements. While Group Relative Policy Optimization offers an efficient, value-model-free alternative to Proximal Policy Optimization, adapting it to real-world multi-reward settings remains challenging. Standard scalarization practices, such as Reward Combination and Advantage Combination, suffer from significant drawbacks: Reward Combination frequently generates advantages with excessively large squared magnitudes that lead to training instability, while Advantage Combination relies on static hyperparameters and ignores cross-objective correlations. To address these limitations, we propose Dynamic Variance-adaptive Advantage Optimization (DVAO), which dynamically adjusts combination weights based on the empirical reward variance of each objective within a rollout group, effectively up-weighting objectives with a stronger learning signal while suppressing noisy ones. We mathematically prove that DVAO maintains bounded advantage magnitudes for stable training and introduces a self-adaptive cross-objective regularization mechanism. Extensive experiments on mathematical reasoning and tool-use benchmarks using Qwen3 and Qwen2.5 models demonstrate that DVAO significantly outperforms baseline methods, achieving a superior multi-objective Pareto frontier and robust training stability.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.25604>
- PDF: [[raw/papers/pdf/2026-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward.pdf]]
- Raw markdown: [[raw/papers/md/2026-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward]]
