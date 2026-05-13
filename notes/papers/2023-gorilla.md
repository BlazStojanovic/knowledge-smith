---
arxiv: '2305.15334'
authors:
- Shishir G. Patil
- Buduo Zhang
- Pedro Rodriguez
- Joseph E. Gonzalez
- et al
created: 2026-04-23
kind: paper
links:
  code: https://github.com/ShishirPatil/gorilla
  paper: https://arxiv.org/abs/2305.15334
  raw: null
  source: https://arxiv.org/abs/2305.15334
owner: blaz
read: false
slug: gorilla
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/agents
title: 'Gorilla: Large Language Model Connected with Massive APIs'
type: note
updated: '2026-05-10'
year: 2023
---

# Gorilla: Large Language Model Connected with Massive APIs

## Citation

- URL: https://arxiv.org/abs/2305.15334
- Authors: Shishir G. Patil, Buduo Zhang, Pedro Rodriguez, Joseph E. Gonzalez, et al.
- Year / venue: 2023 arXiv preprint
- arXiv: 2305.15334

## Short Summary

LLM fine-tuned for API calling using a self-instruct dataset over 1,645 API docs from TorchHub, TensorHub, and HuggingFace. Introduces retrieval-aware training to reduce hallucination of API parameters. Parent project behind the Berkeley Function Calling Leaderboard ([[evals/bfcl-v3]], [[evals/bfcl-v4]]).

## Relevance To Poolside

Parent project for BFCL, our primary function-calling eval. Referenced in [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]] as related tool-use work.

## Related Notes

- [[evals/bfcl-v3]], [[evals/bfcl-v4]] — downstream eval from this project
- [[notes/papers/2024-toolbench]] — contrasting approach (real APIs vs curated API docs)
- [[notes/papers/2024-quality-matters-evaluating-synthetic-data-for-tool-using-llms]]

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
