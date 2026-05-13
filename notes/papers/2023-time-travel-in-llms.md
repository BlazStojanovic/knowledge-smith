---
arxiv: '2308.08493'
authors:
- Shahriar Golchin
- Mihai Surdeanu
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2308.08493
  raw: null
  source: https://arxiv.org/abs/2308.08493
owner: blaz
raw_pdf: raw/papers/pdf/2023-time-travel-in-llms.pdf
read: false
slug: time-travel-in-llms
tags:
- type/paper
- source/primary
- status/stub
- domain/evals
title: 'Time Travel in LLMs: Tracing Data Contamination in Large Language Models'
type: note
updated: '2026-05-10'
year: 2023
---

# Time Travel in LLMs: Tracing Data Contamination in Large Language Models

Black-box contamination detection via guided instruction prompts containing dataset names and initial segments of reference instances.

- **Authors**: Shahriar Golchin, Mihai Surdeanu
- **Venue**: ICLR 2024 (Spotlight)
- **arXiv**: [2308.08493](https://arxiv.org/abs/2308.08493)
- **Raw**: [[raw/papers/pdf/2023-time-travel-in-llms]]

## Core contribution

Uses "guided instruction" prompts: provide the model with a dataset name and the beginning of an instance, then check if the model can reproduce the rest. Detects contamination at both instance and partition level.

## Key results

- 92–100% detection accuracy across seven datasets
- GPT-4 shows contamination with AG News, WNLI, and XSum datasets

## Connections

- Map: [[maps/model-evaluation/contamination-methods]] — black-box, behavioral signal (guided completion)
- Concept: [[concepts/benchmark-contamination]]
- Related: [[notes/papers/2024-benchmarking-benchmark-leakage]], [[notes/papers/2024-min-k-percent]]
