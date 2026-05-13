---
arxiv: '2512.16676'
authors:
- Hao Liang
- Xiaochen Ma
- Zhou Liu
- 'et al. (35 authors total; affiliation: Peking University / BAAI)'
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.16676
  raw: '[[raw/papers/md/2025-dataflow]]'
  source: https://arxiv.org/abs/2512.16676
owner: blaz
raw_pdf: raw/papers/pdf/2025-dataflow.pdf
read: false
slug: dataflow
tags:
- type/paper
- status/stub
- domain/synth-data
- domain/infra
title: 'DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow
  Automation in the Era of Data-Centric AI'
type: note
updated: '2026-05-10'
year: 2025
---

# DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI

## Citation

- URL: https://arxiv.org/abs/2512.16676
- PDF: https://arxiv.org/pdf/2512.16676
- Authors: Hao Liang, Xiaochen Ma, Zhou Liu, et al. (35 authors total; affiliation: Peking University / BAAI)
- Year / venue: 2025 (arXiv, December 2025)

## Core Claim

DataFlow is a modular, LLM-driven data-preparation framework providing ~200 composable operators and an agent that translates natural-language pipeline specs into executable code. Targets scalable, reproducible data curation for LLM pre-training and fine-tuning.

## Key Paper Ideas

- **~200 operators.** Covers filtering, deduplication, quality scoring, augmentation, and format conversion — composable as directed graph pipelines.
- **NL-to-pipeline agent.** An LLM agent interprets a natural-language description of the desired data transformation and emits the corresponding operator DAG — reducing manual pipeline engineering.
- **Benchmark gains.** Reports improvements on Text-to-SQL accuracy and on math/code/knowledge benchmarks from models trained on DataFlow-processed data.
- **Data-centric framing.** Positions itself as infrastructure for the "data-centric AI" paradigm — i.e., systematic, reproducible iteration on data rather than model architecture.

## Relevance To Poolside

Relevant as a reference architecture for data-pipeline tooling; the operator abstraction and agent-driven composition could inform how Poolside structures its own synthetic-data processing pipelines.

## Related Notes

- Concepts: [[concepts/data-filtering-paradigms]], [[concepts/synthetic-data-formalism]]

## Caveats

Stub — created 2026-04-28.
