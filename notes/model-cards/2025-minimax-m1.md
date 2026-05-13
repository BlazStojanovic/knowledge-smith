---
created: 2026-04-23
developer: '[needs verification]'
family: MiniMax-M1
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: minimax-m1
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: MiniMax-M1
type: note
updated: '2026-05-10'
year: 2025
---

# MiniMax-M1

MiniMax's first-generation production MoE model. The first production model to use linear attention (Lightning Attention). Also notable for Partial RoPE (first 50% of head dimensions only). Later superseded by M2 which reverted to standard attention.

## Model Family

| Property | Value |
|---|---|
| Developer | MiniMax |
| Release date | 2025 [nv] |
| Family | MiniMax-M |
| Variants | 456B (46B active) |
| License | [nv] |

## Architecture

Source: Raschka comparison article.

| Property | Value |
|---|---|
| Parameters (total) | 456B |
| Parameters (active) | 46B |
| Layers | [nv] |
| Hidden dim | [nv] |
| FFN dim | [nv] |
| Attention heads | [nv] |
| KV heads | [nv] |
| Vocab size | [nv] |
| Max context length | [nv] |
| Attention | Lightning Attention (linear variant) |
| Positional encoding | Partial RoPE (first 50% of head dimensions) |
| Normalization | [nv] |
| Activation | [nv] |

| MoE Property | Value |
|---|---|
| Total experts | [nv] (sparse) |
| Active experts | [nv] |
| Routing | [nv] |

## Key Architecture Choices

- **Lightning Attention (linear)**: first production model to deploy linear attention at scale. Linear attention reduces the O(n²) complexity of standard attention to O(n), enabling much longer effective context. However, M2 later reverted from this, suggesting quality limitations.
- **Partial RoPE (50%)**: applies rotary position embeddings to only the first half of each head's dimensions, leaving the second half position-independent. This is an unusual approach also adopted by M2.
- **Large active parameter ratio**: 46B active out of 456B total (10.1% activation ratio), higher than DeepSeek V3 (5.5%) or Qwen3 MoE 235B (9.4%).

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
| [[evals/math-500]] | — | |
| [[evals/humaneval]] | — | |

## Blaz Notes

- 

## Related Notes

- Successor: [[notes/model-cards/2025-minimax-m2]] (reverted from linear attention)
- Compare linear attention: [[notes/model-cards/2025-kimi-linear]], [[notes/model-cards/2025-qwen3-next]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Linear attention subsequently abandoned in M2 for reasoning quality concerns — M1 may represent a cautionary data point.
- Most architecture dimensions not available from Raschka.
