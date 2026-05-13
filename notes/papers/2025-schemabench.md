---
aliases:
- SchemaBench
arxiv: '2502.18878'
authors:
- Lu et al
created: 2026-04-24
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.18878
  raw: '[[raw/papers/md/2025-schemabench]]'
  source: https://arxiv.org/abs/2502.18878
owner: blaz
raw_pdf: raw/papers/pdf/2025-schemabench.pdf
read: false
slug: schemabench
tags:
- type/paper
- status/stub
- source/primary
- domain/synth-data
- domain/llm
title: 'SchemaBench: Learning to Generate Structured Output with Schema Reinforcement
  Learning'
type: note
updated: '2026-05-10'
year: 2025
---

# SchemaBench: Learning to Generate Structured Output with Schema Reinforcement Learning

## Citation

- arXiv: 2502.18878
- Authors: Lu et al.
- Year: 2025

## Core Claim

Introduces SchemaBench (~40K JSON schemas) and improves structured generation by incorporating RL with a fine-grained schema validator as reward signal. Outperforms standard SFT baselines for JSON schema adherence.

## Relevance

Cited by [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]] §5.1.2 as representative learning-based method for JSON generation. Demonstrates RL with verifier-style rewards for structural compliance — the JSON analogue of execution-verified code synthesis. Links to [[concepts/verification-signals]] (schema validator as oracle) and [[concepts/process-vs-outcome-reward]] (schema validator reward).
