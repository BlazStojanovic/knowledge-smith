---
arxiv: '2506.13351'
authors:
- Yifei Xu
- Tusher Chakraborty
- Srinagesh Sharma
- Leonardo Nunes
- Swati Sharma
- Kate Drakos Demopulos
- Emre Kıcıman
- Songwu Lu
- Ranveer Chandra
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.13351
  raw: '[[raw/papers/md/2025-direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric]]'
  source: https://arxiv.org/abs/2506.13351
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric.md
raw_pdf: raw/papers/pdf/2025-direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric.pdf
read: false
slug: direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric
tags:
- type/paper
- status/stub
title: 'Direct Reasoning Optimization: Token-Level Reasoning Reflectivity Meets Rubric
  Gates for Unverifiable Tasks'
type: note
updated: '2026-05-11'
year: 2025
---

# Direct Reasoning Optimization: Token-Level Reasoning Reflectivity Meets Rubric Gates for Unverifiable Tasks

> *Yifei Xu, Tusher Chakraborty, Srinagesh Sharma, Leonardo Nunes, Swati Sharma, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning (RL) training of large language models (LLMs) on unverifiable tasks is challenging even when a reasonable-quality reference answer is available. We propose a constrained RL training framework that (i) optimizes a token-level dense Reasoning Reflection Reward (R3) aligned with reasoning quality, and (ii) enforces rubric-gating as feasibility constraints at the rollout group level. R3 measures the model's token-level certainty of a reference answer under its chain-of-thought (CoT) prefix, and selectively emphasizes tokens with high cross-rollout variance, which we call reasoning-reflective tokens, that would otherwise be diluted by the bulk of low-variance tokens. The same variance signal also drives a filter that discards queries with insufficient signal for comparative learning. Rubric-gating complements R3 by operationalizing principled task criteria as hard accept/reject checks on final answers. Empirically, across four datasets spanning scientific writing, medicine, legal contracts, and finance, our framework outperforms strong baselines, achieves faster, more sample-efficient learning, and respects feasibility constraints.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2506.13351>
- PDF: [[raw/papers/pdf/2025-direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric.pdf]]
- Raw markdown: [[raw/papers/md/2025-direct-reasoning-optimization-token-level-reasoning-reflectivity-meets-rubric]]
