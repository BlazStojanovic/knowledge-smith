---
arxiv: '2406.14491'
authors:
- Daixuan Cheng
- Yuxian Gu
- Shaohan Huang
- Junyu Bi
- Minlie Huang
- Furu Wei
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2406.14491
  raw: '[[raw/papers/md/2024-instruction-pre-training]]'
  source: https://arxiv.org/abs/2406.14491
owner: blaz
raw_pdf: raw/papers/pdf/2024-instruction-pre-training.pdf
read: false
slug: instruction-pre-training
tags:
- type/paper
- status/stub
- domain/synth-data
- domain/pretraining
title: 'Instruction Pre-Training: Language Models are Supervised Multitask Learners'
type: note
updated: '2026-05-10'
year: 2024
---

# Instruction Pre-Training: Language Models are Supervised Multitask Learners

## Citation

- URL: https://arxiv.org/abs/2406.14491
- PDF: https://arxiv.org/pdf/2406.14491
- Authors: Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, Furu Wei
- Year / venue: 2024 (arXiv, June 2024)

## Core Claim

Incorporate supervised instruction-response pairs into pre-training (not just post-training). The authors synthesise 200M instruction pairs spanning 40+ task categories using an efficient instruction generator and show that an 8B model trained with instruction pre-training can match or exceed a 70B baseline.

## Key Paper Ideas

- **Supervised multitask pre-training.** Mixing instruction-response data into the pre-training corpus rather than reserving it for SFT; hypothesis is that the model learns task structure earlier.
- **Instruction generator.** A lightweight pipeline to produce 200M pairs at scale, covering 40+ task categories — spans general QA, summarisation, code, and reasoning.
- **Continual pre-training application.** Demonstrates that instruction pre-training can be applied to domain-adaptive continual pre-training, not just from-scratch runs.
- **Efficiency gains.** An 8B model reaches 70B-level performance on target benchmarks, suggesting a parameter-efficiency benefit from richer pre-training supervision.

## Relevance To Poolside

Relevant to thinking about how synthetic instruction data can be mixed earlier (pre-train stage) rather than saved for SFT — potential synergy with the rephrasing / synthetic-data pipeline work.

## Related Notes

- Concepts: [[concepts/synthetic-data-formalism]], [[concepts/reasoning-data-generation]]
- Maps: [[maps/evaluation/landscape]]

## Caveats

Stub — created 2026-04-28.
