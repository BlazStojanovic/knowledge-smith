---
arxiv: '2401.01335'
authors:
- Zixiang Chen
- Yihe Deng
- Huizhuo Yuan
- Kaixuan Ji
- Quanquan Gu
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2401.01335
  raw: null
  source: https://arxiv.org/abs/2401.01335
owner: blaz
raw_pdf: raw/papers/pdf/2024-spin-self-play-fine-tuning.pdf
read: false
slug: spin-self-play-fine-tuning
tags:
- type/paper
- source/primary
- status/stub
- domain/synth-data
- stage/sft
title: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
type: note
updated: '2026-05-10'
year: 2024
---

# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

Iterative self-play where the generator improves by contrasting its own outputs with human demonstrations.

- **Authors**: Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu
- **Venue**: ICML 2024
- **arXiv**: [2401.01335](https://arxiv.org/abs/2401.01335)
- **Raw**: [[raw/papers/pdf/2024-spin-self-play-fine-tuning]]

## Core contribution

SPIN (Self-Play Fine-Tuning): the model plays against itself in an iterative loop, generating responses and then learning to distinguish its own outputs from human demonstrations. Breaks the ceiling of static SFT supervision without requiring extra human annotations or stronger teacher models.

## Connections

- Concept: iterative self-improvement without external reward models
