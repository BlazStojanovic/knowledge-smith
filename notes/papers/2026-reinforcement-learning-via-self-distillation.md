---
arxiv: '2601.20802'
authors:
- Jonas Hübotter
- Frederike Lübeck
- Lejs Behric
- Anton Baumann
- Marco Bagatella
- Daniel Marta
- Ido Hakimi
- Idan Shenfeld
- Thomas Kleine Buening
- Carlos Guestrin
- Andreas Krause
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.20802
  raw: '[[raw/papers/md/2026-reinforcement-learning-via-self-distillation]]'
  source: https://arxiv.org/abs/2601.20802
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-reinforcement-learning-via-self-distillation.md
raw_pdf: raw/papers/pdf/2026-reinforcement-learning-via-self-distillation.pdf
read: false
slug: reinforcement-learning-via-self-distillation
tags:
- type/paper
- status/stub
title: Reinforcement Learning via Self-Distillation
type: note
updated: '2026-05-11'
year: 2026
---

# Reinforcement Learning via Self-Distillation

> *Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed. We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context. Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines. Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts. Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.20802>
- PDF: [[raw/papers/pdf/2026-reinforcement-learning-via-self-distillation.pdf]]
- Raw markdown: [[raw/papers/md/2026-reinforcement-learning-via-self-distillation]]
