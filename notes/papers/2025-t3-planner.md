---
arxiv: '2510.16767'
authors:
- Jia Li
- Guoxiang Zhao
- et al
created: 2026-04-24
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.16767
  raw: https://arxiv.org/pdf/2510.16767
  source: https://arxiv.org/abs/2510.16767
owner: blaz
raw_pdf: raw/papers/pdf/2025-t3-planner.pdf
read: false
slug: t3-planner
tags:
- type/paper
- status/stub
- source/primary
- domain/agents
- domain/reasoning
title: 'T³ Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with
  Temporal Logic'
type: note
updated: '2026-05-10'
year: 2025
---

# T³ Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic

## Citation

- URL: https://arxiv.org/abs/2510.16767
- PDF: https://arxiv.org/pdf/2510.16767
- Authors: Jia Li, Guoxiang Zhao, et al.
- Year / venue: 2025
- arXiv: 2510.16767

## Core Claim

Self-correcting planning framework that verifies LLM-generated plans against Signal Temporal Logic (STL) specifications and repairs them when needed, producing verified plan-and-trajectory traces usable for evaluation or learning.

## Why KB cites this

Cited by [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data|LLM Data Auditor §7]] for STL verification-in-the-loop as V-signal, plan repair traces as synthetic training data, SafetySat metric. See [[concepts/trajectory-synthesis]] §Agent-data product taxonomy.

## Related Notes

- [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]]
- [[concepts/trajectory-synthesis]]
