---
arxiv: '2601.18734'
authors:
- Siyan Zhao
- Zhihui Xie
- Mengchen Liu
- Jing Huang
- Guan Pang
- Feiyu Chen
- Aditya Grover
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2601.18734
  raw: '[[raw/papers/md/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models]]'
  source: https://arxiv.org/abs/2601.18734
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models.md
raw_pdf: raw/papers/pdf/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models.pdf
read: false
slug: self-distilled-reasoner-on-policy-self-distillation-for-large-language-models
tags:
- type/paper
- status/stub
title: 'Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models'
type: note
updated: '2026-05-11'
year: 2026
---

# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

> *Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self, we introduce On-Policy Self-Distillation (OPSD), a learning algorithm where a single LLM acts as both teacher and student with different contexts. The teacher policy conditions on privileged information (e.g., verified reasoning traces) while the student policy sees only the question; training minimizes the per-token divergence between these distributions over the student's own rollouts. We demonstrate the efficacy of our method on multiple mathematical reasoning benchmarks, achieving superior token efficiency compared to reinforcement learning methods and better performance over off-policy distillation methods. Code repo: https://github.com/siyan-zhao/OPSD.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2601.18734>
- PDF: [[raw/papers/pdf/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models]]
