---
arxiv: '2512.13607'
authors:
- Boxin Wang
- Chankyu Lee
- Nayeon Lee
- Sheng-Chieh Lin
- Wenliang Dai
- Yang Chen
- Yangyi Chen
- Zhuolin Yang
- Zihan Liu
- Mohammad Shoeybi
- Bryan Catanzaro
- Wei Ping
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.13607
  raw: '[[raw/papers/md/2025-nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose]]'
  source: https://arxiv.org/abs/2512.13607
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose.md
raw_pdf: raw/papers/pdf/2025-nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose.pdf
read: false
slug: nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose
tags:
- type/paper
- status/stub
title: 'Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose
  Reasoning Models'
type: note
updated: '2026-05-11'
year: 2025
---

# Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models

> *Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Building general-purpose reasoning models with reinforcement learning (RL) entails substantial cross-domain heterogeneity, including large variation in inference-time response lengths and verification latency. Such variability complicates the RL infrastructure, slows training, and makes training curriculum (e.g., response length extension) and hyperparameter selection challenging. In this work, we propose cascaded domain-wise reinforcement learning (Cascade RL) to develop Nemotron-Cascade, capable of operating in both instruct and deep thinking modes, without any performance gap relative to a thinking-only counterpart. Departing from conventional approaches that blend heterogeneous prompts from different domains, Cascade RL orchestrates sequential, domain-wise RL, reducing engineering complexity and delivering state-of-the-art performance across a wide range of benchmarks. Notably, RLHF for alignment, when used as a pre-step, boosts the model's reasoning ability far beyond mere preference optimization, and subsequent domain-wise RLVR stages rarely degrade the benchmark performance attained in earlier domains and may even improve it (see an illustration in Figure 1). Our 14B model, after RL, outperforms its SFT teacher, DeepSeek-R1-0528, on LiveCodeBench v5/v6/Pro and achieves silver-medal performance in the 2025 International Olympiad in Informatics (IOI). We transparently share our training and data recipes.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.13607>
- PDF: [[raw/papers/pdf/2025-nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose.pdf]]
- Raw markdown: [[raw/papers/md/2025-nemotron-cascade-scaling-cascaded-reinforcement-learning-for-general-purpose]]
