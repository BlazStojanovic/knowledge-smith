---
arxiv: '2509.19128'
authors:
- Alexandre Piché
- Ehsan Kamalloo
- Rafael Pardinas
- Xiaoyin Chen
- Dzmitry Bahdanau
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2509.19128
  raw: '[[raw/papers/md/2025-pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation]]'
  source: https://arxiv.org/abs/2509.19128
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation.md
raw_pdf: raw/papers/pdf/2025-pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation.pdf
read: false
slug: pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation
tags:
- type/paper
- status/stub
title: 'PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation'
type: note
updated: '2026-05-11'
year: 2025
---

# PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation

> *Alexandre Piché, Ehsan Kamalloo, Rafael Pardinas, Xiaoyin Chen, Dzmitry Bahdanau* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement Learning (RL) is increasingly utilized to enhance the reasoning capabilities of Large Language Models (LLMs). However, effectively scaling these RL methods presents significant challenges, primarily due to the difficulty in maintaining high AI accelerator utilization without generating stale, off-policy data that harms common RL algorithms. This paper introduces PipelineRL, an approach designed to achieve a superior trade-off between hardware efficiency and data on-policyness for LLM training. PipelineRL employs concurrent asynchronous data generation and model training, distinguished by the novel in-flight weight updates. This mechanism allows the LLM generation engine to receive updated model weights with minimal interruption during the generation of token sequences, thereby maximizing both the accelerator utilization and the freshness of training data. Experiments conducted on long-form reasoning tasks using 128 H100 GPUs demonstrate that PipelineRL achieves approximately $\sim 2x$ faster learning compared to conventional RL baselines while maintaining highly on-policy training data. A scalable and modular open-source implementation of PipelineRL is also released as a key contribution.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2509.19128>
- PDF: [[raw/papers/pdf/2025-pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation.pdf]]
- Raw markdown: [[raw/papers/md/2025-pipelinerl-faster-on-policy-reinforcement-learning-for-long-sequence-generation]]
