---
arxiv: '2602.02710'
authors:
- Fahim Tajwar
- Guanning Zeng
- Yueer Zhou
- Yuda Song
- Daman Arora
- Yiding Jiang
- Jeff Schneider
- Ruslan Salakhutdinov
- Haiwen Feng
- Andrea Zanette
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.02710
  raw: '[[raw/papers/md/2026-maximum-likelihood-reinforcement-learning]]'
  source: https://arxiv.org/abs/2602.02710
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-maximum-likelihood-reinforcement-learning.md
raw_pdf: raw/papers/pdf/2026-maximum-likelihood-reinforcement-learning.pdf
read: false
slug: maximum-likelihood-reinforcement-learning
tags:
- type/paper
- rl
- theory
- optimization
- status/stub
title: Maximum Likelihood Reinforcement Learning
type: note
updated: '2026-05-09'
venue: null
year: 2026
---

# Maximum Likelihood Reinforcement Learning

> *Fahim Tajwar, Guanning Zeng, Yueer Zhou…* — arXiv 2602.02710, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning is the method of choice to train models in sampling-based setups with binary outcome feedback, such as navigation, code generation, and mathematical problem solving. In such settings, models implicitly induce a likelihood over correct rollouts. However, we observe that reinforcement learning does not maximize this likelihood, and instead optimizes only a lower-order approximation. Inspired by this observation, we introduce Maximum Likelihood Reinforcement Learning (MaxRL), a sampling-based framework to approximate maximum likelihood using reinforcement learning techniques. MaxRL addresses the challenges of non-differentiable sampling by defining a compute-indexed family of sample-based objectives that interpolate between standard reinforcement learning and exact maximum likelihood as additional sampling compute is allocated. The resulting objectives admit a simple, unbiased policy-gradient estimator and converge to maximum likelihood optimization in the infinite-compute limit. Empirically, we show that MaxRL Pareto-dominates existing methods in all models and tasks we tested, achieving up to 20x test-time scaling efficiency gains compared to its GRPO-trained counterpart. We also observe MaxRL to scale better with additional data and compute. Our results suggest MaxRL is a promising framework for scaling RL training in correctness based settings.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-maximum-likelihood-reinforcement-learning]]
- PDF: [[raw/papers/pdf/2026-maximum-likelihood-reinforcement-learning.pdf]]
- arXiv: <https://arxiv.org/abs/2602.02710>
