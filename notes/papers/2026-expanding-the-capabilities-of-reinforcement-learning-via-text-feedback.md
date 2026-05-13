---
arxiv: '2602.02482'
authors:
- Yuda Song
- Lili Chen
- Fahim Tajwar
- Remi Munos
- Deepak Pathak
- J. Andrew Bagnell
- Aarti Singh
- Andrea Zanette
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.02482
  raw: '[[raw/papers/md/2026-expanding-the-capabilities-of-reinforcement-learning-via-text-feedback]]'
  source: https://arxiv.org/abs/2602.02482
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-expanding-the-capabilities-of-reinforcement-learning-via-text-feedback.md
raw_pdf: raw/papers/pdf/2026-expanding-the-capabilities-of-reinforcement-learning-via-text-feedback.pdf
read: false
slug: expanding-the-capabilities-of-reinforcement-learning-via-text-feedback
tags:
- type/paper
- status/stub
title: Expanding the Capabilities of Reinforcement Learning via Text Feedback
type: note
updated: '2026-05-11'
year: 2026
---

# Expanding the Capabilities of Reinforcement Learning via Text Feedback

> *Yuda Song, Lili Chen, Fahim Tajwar, Remi Munos, Deepak Pathak, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

The success of RL for LLM post-training stems from an unreasonably uninformative source: a single bit of information per rollout as binary reward or preference label. At the other extreme, distillation offers dense supervision but requires demonstrations, which are costly and difficult to scale. We study text feedback as an intermediate signal: richer than scalar rewards, yet cheaper than complete demonstrations. Textual feedback is a natural mode of human interaction and is already abundant in many real-world settings, where users, annotators, and automated judges routinely critique LLM outputs. Towards leveraging text feedback at scale, we formalize a multi-turn RL setup, RL from Text Feedback (RLTF), where text feedback is available during training but not at inference. Therefore, models must learn to internalize the feedback in order to improve their test-time single-turn performance. To do this, we propose two methods: Self Distillation (RLTF-SD), which trains the single-turn policy to match its own feedback-conditioned second-turn generations; and Feedback Modeling (RLTF-FM), which predicts the feedback as an auxiliary objective. We provide theoretical analysis on both methods, and empirically evaluate on reasoning puzzles, competition math, and creative writing tasks. Our results show that both methods consistently outperform strong baselines across benchmarks, highlighting the potential of RL with an additional source of rich supervision at scale.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.02482>
- PDF: [[raw/papers/pdf/2026-expanding-the-capabilities-of-reinforcement-learning-via-text-feedback.pdf]]
- Raw markdown: [[raw/papers/md/2026-expanding-the-capabilities-of-reinforcement-learning-via-text-feedback]]
