---
created: 2026-04-23
developer: '[needs verification]'
family: Gemma 3n
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: gemma-3n
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: Gemma 3n
type: note
updated: '2026-05-10'
year: 2025
---

# Gemma 3n

Google's mobile-optimized variant of the Gemma 3 family. Introduces Per-Layer Embedding (PLE) and MatFormer (Matryoshka Transformer) for efficient deployment on resource-constrained devices.

## Model Family

| Property | Value |
|---|---|
| Developer | Google |
| Release date | 2025 |
| Family | Gemma 3n |
| Variants | [nv] |
| License | Gemma license |

## Architecture

Source: Raschka comparison article.

| Property | Value |
|---|---|
| Parameters (total) | [nv] |
| Parameters (active) | [nv] (variable via MatFormer slicing) |
| Layers | [nv] |
| Hidden dim | [nv] |
| FFN dim | [nv] |
| Attention heads | [nv] |
| KV heads | [nv] |
| Vocab size | [nv] |
| Max context length | [nv] |
| Attention | [nv] |
| Positional encoding | [nv] |
| Normalization | [nv] |
| Activation | [nv] |

## Key Architecture Choices

- **Per-Layer Embedding (PLE)**: uses layer-specific embedding parameters, allowing different layers to operate on different representation subspaces. Optimized for efficient mobile inference.
- **MatFormer (Matryoshka Transformer)**: enables layered model slicing — a single trained model can be deployed at multiple capability/cost tradeoffs by using different subsets of the model's dimensions, similar to Matryoshka embeddings but applied to the full transformer.
- **Mobile-first design**: architectural choices driven by on-device deployment constraints (memory, latency, power) rather than datacenter-scale throughput.

## Training

| Property | Value |
|---|---|
| Training tokens | [needs verification] |
| Stages | [needs verification] |
| Data mix (high-level) | [needs verification] |
| Compute | [needs verification] |

## Reported Evals

Scores to be filled from primary source.

| Eval | Score | Notes |
|---|---|---|
| [[evals/mmlu]] | — | |
| [[evals/hellaswag]] | — | |
| [[evals/arc]] | — | |

## Blaz Notes

- 

## Related Notes

- Parent family: [[notes/model-cards/2025-gemma-3]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Most architecture details not available from the Raschka article — needs primary source.
- PLE and MatFormer are novel techniques; concept notes do not yet exist.
