---
arxiv: '2604.27063'
authors:
- Aditya A. Ramesh
- Alex Lewandowski
- Jürgen Schmidhuber
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.27063
  raw: '[[raw/papers/md/2026-learning-to-forget-continual-learning-with-adaptive-weight-decay]]'
  source: https://arxiv.org/abs/2604.27063
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-learning-to-forget-continual-learning-with-adaptive-weight-decay.md
raw_pdf: raw/papers/pdf/2026-learning-to-forget-continual-learning-with-adaptive-weight-decay.pdf
read: false
slug: learning-to-forget-continual-learning-with-adaptive-weight-decay
tags:
- type/paper
- status/stub
title: 'Learning to Forget: Continual Learning with Adaptive Weight Decay'
type: note
updated: '2026-05-11'
year: 2026
---

# Learning to Forget: Continual Learning with Adaptive Weight Decay

> *Aditya A. Ramesh, Alex Lewandowski, Jürgen Schmidhuber* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Continual learning agents with finite capacity must balance acquiring new knowledge with retaining the old. This requires controlled forgetting of knowledge that is no longer needed, freeing up capacity to learn. Weight decay, viewed as a mechanism for forgetting, can serve this role by gradually discarding information stored in the weights. However, a fixed scalar weight decay drives this forgetting uniformly over time and uniformly across all parameters, even when some encode stable knowledge while others track rapidly changing targets. We introduce Forgetting through Adaptive Decay (FADE), which adapts per-parameter weight decay rates online via approximate meta-gradient descent. We derive FADE for the online linear setting and apply it to the final layer of neural networks. Our empirical analysis shows that FADE automatically discovers distinct decay rates for different parameters, complements step-size adaptation, and consistently improves over fixed weight decay across online tracking and streaming classification problems.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.27063>
- PDF: [[raw/papers/pdf/2026-learning-to-forget-continual-learning-with-adaptive-weight-decay.pdf]]
- Raw markdown: [[raw/papers/md/2026-learning-to-forget-continual-learning-with-adaptive-weight-decay]]
