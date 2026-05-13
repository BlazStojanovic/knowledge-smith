---
arxiv: '2604.07190'
authors:
- Nathan Lambert
- Florian Brand (Interconnects AI)
created: 2026-05-04
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.07190
  raw: '[[raw/papers/md/2026-atom-project]]'
  source: https://arxiv.org/abs/2604.07190
owner: blaz
read: true
slug: atom-report
tags:
- type/paper
- status/stub
- domain/models
- domain/evals
- source/primary
title: 'The ATOM Report: Measuring the Open Language Model Ecosystem'
type: note
updated: '2026-05-10'
year: 2026
---

# The ATOM Report: Measuring the Open Language Model Ecosystem

## Citation

- URL: https://arxiv.org/abs/2604.07190
- Authors: Nathan Lambert, Florian Brand (Interconnects AI)
- Year / venue: 2026, arXiv preprint
- Raw PDF: [[raw/papers/pdf/2026-atom-project]]

## Core Claim

Comprehensive adoption snapshot of ~1.5K tracked open language models (Nov 2023–Mar 2026) using HuggingFace downloads, OpenRouter inference token share, Arena Elo, and Artificial Analysis Index. Chinese models overtook U.S. models in cumulative downloads in July 2025 and in OpenRouter inference token share by ~mid-2025; Qwen is the single most-used open model family.

## Key Results

- China reached 1.15B cumulative downloads vs 723M (USA) by March 2026 (Fig 1)
- Chinese models' OpenRouter inference token share: 2.8% → >70% in 14 months (Fig 4)
- Qwen: ~942M downloads by March 2026, surpassing Llama (476M) in Sept 2025; derivative share 1% → 69% (Jan 2024 – Feb 2026)
- Introduces [[metrics/relative-adoption-metric]] (RAM): download count for model m normalized by top-10 size-bucket median at matched post-release milestone

## Caveats

- HuggingFace downloads inflate for small models via CI/CD pipelines and automated pulls
- OpenRouter reports top-10 only; orgs spread across many models are undercounted
- Regional attribution is by org HQ — imperfect nationality proxy
