---
arxiv: '2509.04259'
authors:
- Idan Shenfeld
- Jyothish Pari
- Pulkit Agrawal
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2509.04259
  raw: '[[raw/papers/md/2025-rl-s-razor-why-online-reinforcement-learning-forgets-less]]'
  source: https://arxiv.org/abs/2509.04259
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-rl-s-razor-why-online-reinforcement-learning-forgets-less.md
raw_pdf: raw/papers/pdf/2025-rl-s-razor-why-online-reinforcement-learning-forgets-less.pdf
read: false
slug: rl-s-razor-why-online-reinforcement-learning-forgets-less
tags:
- type/paper
- status/stub
title: 'RL''s Razor: Why Online Reinforcement Learning Forgets Less'
type: note
updated: '2026-05-11'
year: 2025
---

# RL's Razor: Why Online Reinforcement Learning Forgets Less

> *Idan Shenfeld, Jyothish Pari, Pulkit Agrawal* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Comparison of fine-tuning models with reinforcement learning (RL) and supervised fine-tuning (SFT) reveals that, despite similar performance at a new task, RL preserves prior knowledge and capabilities significantly better. We find that the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task. Our analysis reveals that on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model. We validate these findings through experiments with large language models and robotic foundation models and further provide theoretical justification for why on-policy RL updates lead to a smaller KL change. We term this principle $\textit{RL's Razor}$: among all ways to solve a new task, RL prefers those closest in KL to the original model.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2509.04259>
- PDF: [[raw/papers/pdf/2025-rl-s-razor-why-online-reinforcement-learning-forgets-less.pdf]]
- Raw markdown: [[raw/papers/md/2025-rl-s-razor-why-online-reinforcement-learning-forgets-less]]
