---
arxiv: '2303.08896'
authors:
- Potsawee Manakul
- Adian Liusie
- Mark J.F. Gales
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2303.08896
  raw: null
  source: https://arxiv.org/abs/2303.08896
owner: blaz
raw_pdf: raw/papers/pdf/2023-selfcheckgpt.pdf
read: false
slug: selfcheckgpt
tags:
- type/paper
- source/primary
- status/stub
- domain/llm
- domain/evals
title: 'SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative
  Large Language Models'
type: note
updated: '2026-05-10'
year: 2023
---

# SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

Consistency sampling to detect hallucinated content without external knowledge.

- **Authors**: Potsawee Manakul, Adian Liusie, Mark J.F. Gales
- **Venue**: EMNLP 2023
- **arXiv**: [2303.08896](https://arxiv.org/abs/2303.08896)
- **Raw**: [[raw/papers/pdf/2023-selfcheckgpt]]

## Core contribution

Detects hallucinations by sampling multiple responses and measuring consistency: factual statements tend to be consistent across samples while hallucinations vary. Zero-resource — requires no external knowledge base. Can flag or filter likely hallucinated content from generated datasets.

## Connections

- Related: [[notes/papers/2024-chain-of-verification-reduces-hallucination]]
- Related: [[notes/papers/2023-rarr-researching-and-revising-what-language-models-say]]
