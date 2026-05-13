---
created: 2026-04-23
developer: '[needs verification]'
family: Qwen3-Next
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: qwen3-next
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: Qwen3-Next
type: note
updated: '2026-05-10'
year: 2025
---

# Qwen3-Next

Alibaba Qwen's experimental next-generation model. Major architectural departure from Qwen3: replaces standard attention with a Gated DeltaNet + Gated Attention hybrid (3:1 ratio), reintroduces shared experts, and dramatically increases expert count to 1024. Supports Multi-Token Prediction.

## Model Family

| Property | Value |
|---|---|
| Developer | Alibaba / Qwen |
| Release date | 2025 [nv] |
| Family | Qwen3-Next |
| Variants | 80B-A3B (80B total, 3B active); Qwen3-Coder-Next (same arch, code-specialized training) |
| License | [nv] |

## Architecture

Source: Raschka comparison article.

Sources: Raschka diagram (Figure 35).

| Property | Value |
|---|---|
| Parameters (total) | 80B |
| Parameters (active) | ~3B |
| Layers | [nv] |
| Hidden dim (d_model) | 2048 |
| Expert input size | 2048 |
| Attention heads | [nv] |
| KV heads | [nv] |
| Vocab size | 151K |
| Max context length | 262K (native) |
| Attention | Gated DeltaNet + Gated Attention hybrid (3:1 ratio) |
| Positional encoding | RoPE + YaRN |
| Normalization | [nv] |
| Activation | [nv] |

| MoE Property | Value |
|---|---|
| Total experts | 1024 |
| Active experts | 3 |
| Shared experts | Yes (reintroduced; absent in Qwen3 MoE) |
| Routing | [nv] |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-35-qwen3-vs-qwen3next.png]]
*Qwen3 vs Qwen3-Next design changes — from Raschka*

![[raw/images/articles/raschka-llm-architecture/figure-42-qwen3next-vs-kimilinear.png]]
*Qwen3-Next vs Kimi Linear architecture comparison*

## Key Architecture Choices

- **Linear attention hybrid**: 3:1 ratio of Gated DeltaNet (linear attention variant) to Gated Attention (standard attention). The linear layers provide efficient long-sequence processing while periodic standard attention layers maintain modeling quality.
- **Massive expert count (1024)**: 8× more experts than Qwen3 MoE (128), with only 3 active per token (0.3% activation ratio). Extremely fine-grained specialization.
- **Shared expert reintroduced**: reverses the Qwen3 MoE design decision that omitted shared experts, returning to the DeepSeek-style approach.
- **Multi-Token Prediction (MTP)**: supports predicting multiple future tokens, enabling speculative decoding for faster inference.
- **262K native context**: one of the longest native context lengths, achieved via YaRN scaling on top of the linear attention efficiency.
- **Qwen3-Coder-Next**: identical architecture specialized for code through training data and post-training.

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
| [[evals/humaneval]] | — | |
| [[evals/math-500]] | — | |
| [[evals/livecodebench]] | — | |
| [[evals/aime-2025]] | — | |

## Blaz Notes

- 

## Related Notes

- Predecessor: [[notes/model-cards/2025-qwen3]], [[notes/model-cards/2025-qwen3-moe]]
- Compare linear attention: [[notes/model-cards/2025-kimi-linear]], [[notes/model-cards/2025-minimax-m1]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Most architecture dimensions not yet verified — needs primary source/paper.
- Gated DeltaNet is a relatively novel attention variant; concept note does not yet exist.
