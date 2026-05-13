---
arxiv: '2512.07783'
authors:
- Charlie Zhang
- Graham Neubig
- Xiang Yue
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.07783
  raw: '[[raw/papers/md/2025-on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language]]'
  source: https://arxiv.org/abs/2512.07783
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language.md
raw_pdf: raw/papers/pdf/2025-on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language.pdf
read: false
slug: on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language
tags:
- type/paper
- status/stub
title: On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language
  Models
type: note
updated: '2026-05-11'
year: 2025
---

# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models

> *Charlie Zhang, Graham Neubig, Xiang Yue* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether post-training truly extends a model's reasoning ability beyond what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines: large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models along two axes: extrapolative generalization to more complex compositions and contextual generalization across surface contexts. Using this framework, we reconcile competing views on RL's effectiveness. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model's edge of competence, tasks at the boundary that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding and improving reasoning LM training strategies.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.07783>
- PDF: [[raw/papers/pdf/2025-on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language.pdf]]
- Raw markdown: [[raw/papers/md/2025-on-the-interplay-of-pre-training-mid-training-and-rl-on-reasoning-language]]
