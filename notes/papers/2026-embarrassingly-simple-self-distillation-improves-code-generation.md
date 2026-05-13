---
arxiv: '2604.01193'
authors:
- Ruixiang Zhang
- Richard He Bai
- Huangjie Zheng
- Navdeep Jaitly
- Ronan Collobert
- Yizhe Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.01193
  raw: '[[raw/papers/md/2026-embarrassingly-simple-self-distillation-improves-code-generation]]'
  source: https://arxiv.org/abs/2604.01193
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-embarrassingly-simple-self-distillation-improves-code-generation.md
raw_pdf: raw/papers/pdf/2026-embarrassingly-simple-self-distillation-improves-code-generation.pdf
read: false
slug: embarrassingly-simple-self-distillation-improves-code-generation
tags:
- type/paper
- status/stub
title: Embarrassingly Simple Self-Distillation Improves Code Generation
type: note
updated: '2026-05-11'
year: 2026
---

# Embarrassingly Simple Self-Distillation Improves Code Generation

> *Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Can a large language model (LLM) improve at code generation using only its own raw outputs, without a verifier, a teacher model, or reinforcement learning? We answer in the affirmative with simple self-distillation (SSD): sample solutions from the model with certain temperature and truncation configurations, then fine-tune on those samples with standard supervised fine-tuning. SSD improves Qwen3-30B-Instruct from 42.4% to 55.3% pass@1 on LiveCodeBench v6, with gains concentrating on harder problems, and it generalizes across Qwen and Llama models at 4B, 8B, and 30B scale, including both instruct and thinking variants. To understand why such a simple method can work, we trace these gains to a precision-exploration conflict in LLM decoding and show that SSD reshapes token distributions in a context-dependent way, suppressing distractor tails where precision matters while preserving useful diversity where exploration matters. Taken together, SSD offers a complementary post-training direction for improving LLM code generation.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.01193>
- PDF: [[raw/papers/pdf/2026-embarrassingly-simple-self-distillation-improves-code-generation.pdf]]
- Raw markdown: [[raw/papers/md/2026-embarrassingly-simple-self-distillation-improves-code-generation]]
