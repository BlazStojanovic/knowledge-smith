---
created: 2026-04-23
developer: '[needs verification]'
family: Kimi Linear
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: kimi-linear
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: Kimi Linear
type: note
updated: '2026-05-10'
year: 2025
---

# Kimi Linear

Moonshot AI's 48B dense model exploring linear attention hybrids. Uses Kimi Delta Attention (a refined Gated DeltaNet) combined with MLA in a 3:1 ratio. Achieves near-full-attention quality with linear-attention speed.

## Model Family

| Property | Value |
|---|---|
| Developer | Moonshot AI |
| Release date | 2025 |
| Family | Kimi Linear |
| Variants | 48B |
| License | [nv] |

## Architecture

Source: Raschka comparison article.

| Property | Value |
|---|---|
| Parameters (total) | 48B |
| Parameters (active) | 48B (dense) |
| Layers | [nv] |
| Hidden dim | [nv] |
| FFN dim | [nv] |
| Attention heads | [nv] |
| KV heads | [nv] |
| Vocab size | [nv] |
| Max context length | [nv] |
| Attention | Kimi Delta Attention + MLA hybrid (3:1 ratio) |
| Positional encoding | NoPE in MLA layers; channel-wise gating in Delta layers |
| Normalization | [nv] |
| Activation | [nv] |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-42-qwen3next-vs-kimilinear.png]]
*Qwen3-Next vs Kimi Linear architecture comparison — from Raschka*

## Key Architecture Choices

- **Kimi Delta Attention**: a refined version of Gated DeltaNet, a linear attention variant. Provides O(n) complexity per token rather than O(n²), enabling efficient long-sequence processing.
- **3:1 hybrid ratio**: 3 Kimi Delta Attention layers per 1 MLA layer. The periodic MLA layers maintain full attention modeling quality while the linear layers handle most of the compute efficiently.
- **NoPE in MLA layers**: the MLA (full attention) layers use No Positional Embeddings, relying on causal masking for position information.
- **Channel-wise gating in Delta layers**: applies gating at the channel (feature dimension) level rather than token level in the linear attention layers.
- **Improvement over Qwen3-Next**: Raschka notes this model improves upon the Gated DeltaNet approach used in [[notes/model-cards/2025-qwen3-next]], maintaining speed while improving reasoning quality.

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
| [[evals/gpqa]] | — | |

## Blaz Notes

- 

## Related Notes

- Compare: [[notes/model-cards/2025-kimi-k2]] (same org, full MLA)
- Compare: [[notes/model-cards/2025-qwen3-next]] (similar linear attention hybrid)
- Compare: [[notes/model-cards/2025-minimax-m1]] (earlier linear attention attempt)
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Most architecture dimensions not available from Raschka — needs primary paper.
- Kimi Delta Attention is novel; no concept note exists yet.
