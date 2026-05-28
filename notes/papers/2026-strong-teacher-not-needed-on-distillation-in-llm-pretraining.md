---
arxiv: '2605.23857'
authors:
- Taiming Lu
- Zhuang Liu
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.23857
  raw: '[[raw/papers/md/2026-strong-teacher-not-needed-on-distillation-in-llm-pretraining]]'
  source: https://arxiv.org/abs/2605.23857
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-strong-teacher-not-needed-on-distillation-in-llm-pretraining.md
raw_pdf: raw/papers/pdf/2026-strong-teacher-not-needed-on-distillation-in-llm-pretraining.pdf
read: false
slug: strong-teacher-not-needed-on-distillation-in-llm-pretraining
tags:
- type/paper
- status/stub
title: Strong Teacher Not Needed? On Distillation in LLM Pretraining
type: note
updated: '2026-05-28'
year: 2026
---

# Strong Teacher Not Needed? On Distillation in LLM Pretraining

> *Taiming Lu, Zhuang Liu* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Knowledge distillation generally assumes a strong-to-weak relationship where stronger teachers yield better students. In this work, we examine this assumption about distillation in large language model pretraining. By varying architecture sizes and training token budgets, we create strong-to-weak, same-level, and weak-to-strong teacher-student relationships, and study distillation's effectiveness under each. We find that the teacher need not be strong: with proper mixing of the language modeling and knowledge distillation losses, even small and undertrained teachers improve larger students. At the same time, a stronger teacher is not always better: pushing the teacher further, through more parameters or more training tokens, can saturate or even reverse the distillation gains. We further observe that distillation improves generalization (out-of-distribution and downstream performance) more readily than in-domain fitting. Together, these results challenge the common belief that distillation pretraining always requires a strong teacher.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.23857>
- PDF: [[raw/papers/pdf/2026-strong-teacher-not-needed-on-distillation-in-llm-pretraining.pdf]]
- Raw markdown: [[raw/papers/md/2026-strong-teacher-not-needed-on-distillation-in-llm-pretraining]]
