---
arxiv: '2602.13949'
authors:
- Taiwei Shi
- Sihao Chen
- Bowen Jiang
- Linxin Song
- Longqi Yang
- Jieyu Zhao
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.13949
  raw: '[[raw/papers/md/2026-experiential-reinforcement-learning]]'
  source: https://arxiv.org/abs/2602.13949
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-experiential-reinforcement-learning.md
raw_pdf: raw/papers/pdf/2026-experiential-reinforcement-learning.pdf
read: false
slug: experiential-reinforcement-learning
tags:
- type/paper
- status/stub
title: Experiential Reinforcement Learning
type: note
updated: '2026-05-11'
year: 2026
---

# Experiential Reinforcement Learning

> *Taiwei Shi, Sihao Chen, Bowen Jiang, Linxin Song, Longqi Yang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning has become the central approach for language models (LMs) to learn from environmental reward or feedback. In practice, the environmental feedback is usually sparse and delayed. Learning from such signals is challenging, as LMs must implicitly infer how observed failures should translate into behavioral changes for future iterations. We introduce Experiential Reinforcement Learning (ERL), a training paradigm that embeds an explicit experience-reflection-consolidation loop into the reinforcement learning process. Given a task, the model generates an initial attempt, receives environmental feedback, and produces a reflection that guides a refined second attempt, whose success is reinforced and internalized into the base policy. This process converts feedback into structured behavioral revision, improving exploration and stabilizing optimization while preserving gains at deployment without additional inference cost. Across sparse-reward control environments and agentic reasoning benchmarks, ERL consistently improves learning efficiency and final performance over strong reinforcement learning baselines, achieving gains of up to +81% in complex multi-step environments and up to +11% in tool-using reasoning tasks. These results suggest that integrating explicit self-reflection into policy training provides a practical mechanism for transforming feedback into durable behavioral improvement.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.13949>
- PDF: [[raw/papers/pdf/2026-experiential-reinforcement-learning.pdf]]
- Raw markdown: [[raw/papers/md/2026-experiential-reinforcement-learning]]
