---
aliases:
- LogBench
arxiv: '2307.05950'
authors:
- Li et al
created: 2026-04-24
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2307.05950
  raw: '[[raw/papers/md/2024-logbench]]'
  source: https://arxiv.org/abs/2307.05950
owner: blaz
raw_pdf: raw/papers/pdf/2024-logbench.pdf
read: false
slug: logbench
tags:
- type/paper
- status/stub
- source/primary
- domain/evals
- domain/code
title: 'Exploring the Effectiveness of LLMs in Automated Logging Generation: An Empirical
  Study'
type: note
updated: '2026-05-10'
year: 2024
---

# Exploring the Effectiveness of LLMs in Automated Logging Generation: An Empirical Study

## Citation

- arXiv: 2307.05950
- Authors: Li et al.
- Year: 2024 (IEEE TSE)

## Core Claim

LLMs can accurately predict logging attributes (level, variables) but fail to produce full log statements mimicking human-written code — best BLEU score of 0.249. Introduces LogBench and LogBench-T (semantics-preserving code transformations). Performance degrades consistently under transformed contexts, indicating brittleness to semantics-preserving code transformations.

## Relevance

Cited by [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]] §5.1.3, §5.2.3. The semantic-vs-surface gap finding — models understand *what* to log but not *how* to write the log statement — mirrors broader code generation challenges. LogBench-T's transformation-robustness testing is methodologically transferable to code robustness evaluation.
