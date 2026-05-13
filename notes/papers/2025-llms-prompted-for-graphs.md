---
aliases:
- LLMs Prompted for Graphs
arxiv: '2409.00159'
authors:
- Richardeau et al
created: 2026-04-24
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2409.00159
  raw: '[[raw/papers/md/2025-llms-prompted-for-graphs]]'
  source: https://arxiv.org/abs/2409.00159
owner: blaz
raw_pdf: raw/papers/pdf/2025-llms-prompted-for-graphs.pdf
read: false
slug: llms-prompted-for-graphs
tags:
- type/paper
- status/stub
- source/primary
- domain/evals
- domain/llm
title: 'LLMs Prompted for Graphs: Hallucinations and Generative Capabilities'
type: note
updated: '2026-05-10'
year: 2025
---

# LLMs Prompted for Graphs: Hallucinations and Generative Capabilities

## Citation

- arXiv: 2409.00159
- Authors: Richardeau et al.
- Year: 2025 (v3)

## Core Claim

Studies structural hallucinations in LLM-generated graphs — plausible but incorrect graph structures. Introduces Graph Atlas Distance (GAD) based on graph edit distance to canonical atlas graphs, plus a capped variant for robust aggregation. Also defines Syntactic Correctness Rate and Degree-distribution Deviation as complementary robustness checks.

## Relevance

Cited by [[notes/papers/2026-the-llm-data-auditor-a-metric-oriented-survey-on-quality-and-trustworthiness-in-evaluating-synthetic-data]] §5.3.1 Robustness. The hallucination-detection methodology is transferable: the concept of measuring structural deviation from expected targets parallels code-structure hallucination detection. GAD's use of edit distance to canonical targets is analogous to diff-based evaluation in code.
