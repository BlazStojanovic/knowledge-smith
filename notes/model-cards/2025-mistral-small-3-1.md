---
created: 2026-04-23
developer: '[needs verification]'
family: Mistral Small 3.1
kind: model-card
links:
  code: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://mistral.ai/news/mistral-small-3-1
model_type: llm
owner: blaz
read: false
slug: mistral-small-3-1
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: Mistral Small 3.1
type: note
updated: '2026-05-10'
year: 2025
---

# Mistral Small 3.1

Mistral AI's 24B dense model. Optimized for inference speed with reduced KV cache and layer count relative to comparable-sized models like Gemma 3 27B.

## Model Family

| Property | Value |
|---|---|
| Developer | Mistral AI |
| Release date | 2025-03 |
| Family | Mistral Small |
| Variants | 24B (base, instruct) |
| License | Apache 2.0 |

## Architecture

Sources: [HuggingFace config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503), Raschka comparison article.

| Property | Value |
|---|---|
| Parameters (total) | 24B |
| Parameters (active) | 24B (dense) |
| Layers | 40 |
| Hidden dim (d_model) | 5120 |
| FFN dim (d_ff) | 32768 |
| Attention heads | 32 |
| KV heads | 8 (GQA) |
| Head dim | 128 |
| Vocab size | 131,072 |
| Max context length | 131,072 |
| Attention | GQA |
| Sliding window | Disabled (null in config) |
| Positional encoding | RoPE (theta=1B) |
| Normalization | RMSNorm |
| Activation | SiLU |
| Tie embeddings | [nv] |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-16-gemma3-vs-mistral.png]]
*Gemma 3 27B vs Mistral Small 3.1 24B — from Raschka*

## Key Architecture Choices

- **Sliding window disabled by default**: although the architecture supports sliding-window attention, it is disabled in the default configuration, using full attention throughout.
- **Custom tokenizer**: 131K vocab — one of the largest among open models, larger than Qwen3 (152K) but in the same range.
- **Very high RoPE theta (1B)**: theta=1,000,000,000 — orders of magnitude higher than Qwen3 (1M) or OLMo 2 (500K), enabling native 128K context without explicit RoPE scaling.
- **Inference-optimized**: reduced layer count (40 vs deeper alternatives) with very wide FFN (32768 = 6.4× hidden dim), favouring width over depth.
- **Dense architecture**: no MoE — all 24B parameters active per token.

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
| [[evals/ifeval]] | — | |
| [[evals/arena-hard-auto]] | — | |

## Blaz Notes

- 

## Related Notes

- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]
- Compare: [[notes/model-cards/2025-gemma-3]] (similar size, different attention strategy)

## Caveats

- `tie_word_embeddings` not specified in HuggingFace config.
- Multimodal (Pixtral vision encoder) config exists but not documented here — text architecture only.
