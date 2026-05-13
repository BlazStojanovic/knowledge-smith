---
arxiv: '2601.19897'
authors:
- Idan Shenfeld
- Mehul Damani
- Jonas Hübotter
- Pulkit Agrawal
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.19897
  raw: '[[raw/papers/md/2026-self-distillation-enables-continual-learning]]'
  source: https://arxiv.org/abs/2601.19897
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-self-distillation-enables-continual-learning.md
raw_pdf: raw/papers/pdf/2026-self-distillation-enables-continual-learning.pdf
read: false
slug: self-distillation-enables-continual-learning
tags:
- type/paper
- status/stub
title: Self-Distillation Enables Continual Learning
type: note
updated: '2026-05-11'
year: 2026
---

# Self-Distillation Enables Continual Learning

> *Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce Self-Distillation Fine-Tuning (SDFT), a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.19897>
- PDF: [[raw/papers/pdf/2026-self-distillation-enables-continual-learning.pdf]]
- Raw markdown: [[raw/papers/md/2026-self-distillation-enables-continual-learning]]
