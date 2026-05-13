---
arxiv: '2510.13786'
authors:
- Devvrit Khatri
- Lovish Madaan
- Rishabh Tiwari
- Rachit Bansal
- Sai Surya Duvvuri
- Manzil Zaheer
- Inderjit S. Dhillon
- David Brandfonbrener
- Rishabh Agarwal
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.13786
  raw: '[[raw/papers/md/2025-art-of-scaling-reinforcement-learning-compute-for-llms]]'
  source: https://arxiv.org/abs/2510.13786
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-art-of-scaling-reinforcement-learning-compute-for-llms.md
raw_pdf: raw/papers/pdf/2025-art-of-scaling-reinforcement-learning-compute-for-llms.pdf
read: false
slug: art-of-scaling-reinforcement-learning-compute-for-llms
tags:
- type/paper
- status/stub
title: The Art of Scaling Reinforcement Learning Compute for LLMs
type: note
updated: '2026-05-11'
year: 2025
---

# The Art of Scaling Reinforcement Learning Compute for LLMs

> *Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning (RL) has become central to training large language models (LLMs), yet the field lacks predictive scaling methodologies comparable to those established for pre-training. Despite rapidly rising compute budgets, there is no principled understanding of how to evaluate algorithmic improvements for scaling RL compute. We present the first large-scale systematic study, amounting to more than 400,000 GPU-hours, that defines a principled framework for analyzing and predicting RL scaling in LLMs. We fit sigmoidal compute-performance curves for RL training and ablate a wide range of common design choices to analyze their effects on asymptotic performance and compute efficiency. We observe: (1) Not all recipes yield similar asymptotic performance, (2) Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote, and (3) Stable, scalable recipes follow predictable scaling trajectories, enabling extrapolation from smaller-scale runs. Combining these insights, we propose a best-practice recipe, ScaleRL, and demonstrate its effectiveness by successfully scaling and predicting validation performance on a single RL run scaled up to 100,000 GPU-hours. Our work provides both a scientific framework for analyzing scaling in RL and a practical recipe that brings RL training closer to the predictability long achieved in pre-training.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.13786>
- PDF: [[raw/papers/pdf/2025-art-of-scaling-reinforcement-learning-compute-for-llms.pdf]]
- Raw markdown: [[raw/papers/md/2025-art-of-scaling-reinforcement-learning-compute-for-llms]]
