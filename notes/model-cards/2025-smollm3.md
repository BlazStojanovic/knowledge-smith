---
created: 2026-04-23
developer: '[needs verification]'
family: SmolLM3
kind: model-card
links:
  code: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
model_type: llm
owner: blaz
read: false
slug: smollm3
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: SmolLM3
type: note
updated: '2026-05-10'
year: 2025
---

# SmolLM3

Hugging Face's 3B parameter model. Notable for experimenting with No Positional Embeddings (NoPE), applying explicit positional encoding only every 4th layer and relying on causal masking for position information elsewhere.

## Model Family

| Property | Value |
|---|---|
| Developer | Hugging Face |
| Release date | 2025 |
| Family | SmolLM |
| Variants | 3B |
| License | Apache 2.0 [nv] |

## Architecture

Sources: [HuggingFace config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), Raschka comparison article.

| Property | Value |
|---|---|
| Parameters (total) | 3B |
| Parameters (active) | 3B (dense) |
| Layers | 36 |
| Hidden dim (d_model) | 2048 |
| FFN dim (d_ff) | 11008 |
| Attention heads | 16 |
| KV heads | 4 (GQA) |
| Head dim | 128 |
| Vocab size | 128,256 |
| Max context length | 65,536 |
| Attention | GQA |
| Positional encoding | NoPE hybrid — RoPE on 27/36 layers, disabled every 4th layer |
| Normalization | RMSNorm |
| Activation | SiLU |
| Tie embeddings | Yes |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-21-qwen3-vs-smollm3.png]]
*Qwen3 4B vs SmolLM3 3B architecture comparison — from Raschka*

## Key Architecture Choices

- **NoPE hybrid (RoPE disabled every 4th layer)**: 9 of 36 layers (positions 3, 7, 11, ..., 35) have RoPE disabled, relying on causal masking for position information. The remaining 27 layers use standard RoPE (theta=5M). Config: `no_rope_layer_interval: 4`.
- **Better length generalization**: the NoPE approach is motivated by research showing that models with fewer positional encoding layers generalize better to sequence lengths unseen during training.
- **Small-scale efficiency**: designed to be a strong model at the 3B scale, competing with larger models on per-parameter efficiency.

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
| [[evals/gsm8k-v2]] | — | |
| [[evals/hellaswag]] | — | |
| [[evals/arc]] | — | |

## Blaz Notes

- 

## Related Notes

- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- NoPE is a relatively novel choice — few production models use this approach, making direct architectural comparisons limited.
- Tied embeddings (`tie_word_embeddings: true`) is unusual among recent models this size.
