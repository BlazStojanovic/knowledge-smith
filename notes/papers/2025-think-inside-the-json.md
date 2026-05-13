---
aliases:
- Think Inside the JSON
arxiv: '2502.14905'
authors:
- Agarwal et al
created: 2026-04-24
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.14905
  raw: '[[raw/papers/md/2025-think-inside-the-json]]'
  source: https://arxiv.org/abs/2502.14905
owner: blaz
raw_pdf: raw/papers/pdf/2025-think-inside-the-json.pdf
read: false
slug: think-inside-the-json
tags:
- type/paper
- status/stub
- source/primary
- domain/synth-data
- domain/llm
title: 'Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence'
type: note
updated: '2026-05-10'
year: 2025
---

# Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence

## Citation

- arXiv: 2502.14905
- Authors: Agarwal et al.
- Year: 2025

## Core Claim

Applies Group Relative Policy Optimization (GRPO) to train smaller models with custom rewards for strict JSON schema adherence. Demonstrates effective improvements in schema consistency without constrained decoding.

## Relevance

Cited by [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]] §5.1.2, §5.2.2. GRPO with schema validator rewards is the JSON-domain instance of RL with verifier rewards — parallel to code RL with test-execution rewards. Links to [[concepts/verification-signals]], [[concepts/generation-intervention-loci]] (Locus 3: alignment-based control).
