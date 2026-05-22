---
arxiv: '2605.10781'
authors:
- Jeonghye Kim
- Jiwon Jeon
- Dongsheng Li
- Yuqing Yang
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.10781
  raw: '[[raw/papers/md/2026-rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with]]'
  source: https://arxiv.org/abs/2605.10781
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with.md
raw_pdf: raw/papers/pdf/2026-rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with.pdf
read: false
slug: rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with
tags:
- type/paper
- status/stub
title: 'Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with
  Self-Distilled RLVR'
type: note
updated: '2026-05-22'
year: 2026
---

# Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR

> *Jeonghye Kim, Jiwon Jeon, Dongsheng Li, Yuqing Yang* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Self-distillation has emerged as a powerful framework for post-training LLMs, where a teacher conditioned on extra information guides a student without it, both from the same model. While this guidance is useful when the student has failed, on successful rollouts, the same mechanism instead overwrites the student's choices and suppresses it's own reasoning. Therefore, we propose reading the original self-distillation signal in reverse: when the student succeeds along a path the teacher would not have predicted, these tokens reflect its self-driven reasoning. Building on this, we propose RLRT (RLVR with Reversed Teacher), which augments GRPO by reinforcing these tokens on correct rollouts. We interpret this as a new form of exploration in RLVR: not uniform diversity, but valuable exploration grounded in the student's own success. Across base, instruction-tuned, and thinking-tuned Qwen3 checkpoints, RLRT substantially outperforms self-distillation and exploration-based baselines, establishing information asymmetry as a new, principled design axis for RLVR.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.10781>
- PDF: [[raw/papers/pdf/2026-rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with.pdf]]
- Raw markdown: [[raw/papers/md/2026-rebellious-student-reversing-teacher-signals-for-reasoning-exploration-with]]
