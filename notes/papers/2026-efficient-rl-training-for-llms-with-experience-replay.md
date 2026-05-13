---
arxiv: '2604.08706'
authors:
- Charles Arnal
- Vivien Cabannes
- Taco Cohen
- Julia Kempe
- Remi Munos
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.08706
  raw: '[[raw/papers/md/2026-efficient-rl-training-for-llms-with-experience-replay]]'
  source: https://arxiv.org/abs/2604.08706
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-efficient-rl-training-for-llms-with-experience-replay.md
raw_pdf: raw/papers/pdf/2026-efficient-rl-training-for-llms-with-experience-replay.pdf
read: false
slug: efficient-rl-training-for-llms-with-experience-replay
tags:
- type/paper
- status/stub
title: Efficient RL Training for LLMs with Experience Replay
type: note
updated: '2026-05-11'
year: 2026
---

# Efficient RL Training for LLMs with Experience Replay

> *Charles Arnal, Vivien Cabannes, Taco Cohen, Julia Kempe, Remi Munos* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

While Experience Replay - the practice of storing rollouts and reusing them multiple times during training - is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance. In this work, we challenge this assumption. We present a systematic study of replay buffers for LLM post-training, formalizing the optimal design as a trade-off between staleness-induced variance, sample diversity and the high computational cost of generation. We show that strict on-policy sampling is suboptimal when generation is expensive. Empirically, we show that a well-designed replay buffer can drastically reduce inference compute without degrading - and in some cases even improving - final model performance, while preserving policy entropy.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.08706>
- PDF: [[raw/papers/pdf/2026-efficient-rl-training-for-llms-with-experience-replay.pdf]]
- Raw markdown: [[raw/papers/md/2026-efficient-rl-training-for-llms-with-experience-replay]]
