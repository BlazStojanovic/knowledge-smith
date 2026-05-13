---
created: 2026-04-23
developer: '[needs verification]'
family: gpt-oss
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: gpt-oss
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: gpt-oss
type: note
updated: '2026-05-10'
year: 2025
---

# gpt-oss

OpenAI's first open-weight model release. Two MoE variants with GQA, sliding window attention on alternating layers, and unusual attention bias units with learned attention sink logits.

## Model Family

| Property | Value |
|---|---|
| Developer | OpenAI |
| Release date | 2025 |
| Family | gpt-oss |
| Variants | gpt-oss-20b (20B total, 3.6B active), gpt-oss-120b (120B total) |
| License | [nv] |

## Architecture

Source: Raschka comparison article.

| Property | 20B | 120B |
|---|---|---|
| Parameters (total) | 20B | 120B |
| Parameters (active) | 3.6B | [nv] |
| Layers | [nv] | [nv] |
| Hidden dim | [nv] | [nv] |
| FFN dim | [nv] | [nv] |
| Attention heads | [nv] | [nv] |
| KV heads | [nv] (GQA) | [nv] (GQA) |
| Vocab size | [nv] | [nv] |
| Max context length | [nv] | [nv] |

All variants share:

| Property | Value |
|---|---|
| Attention | GQA + sliding window (every other layer) |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | [nv] |

| MoE Property | Value |
|---|---|
| Total experts | 32 |
| Active experts | 4 |
| Shared experts | [nv] |
| Expert size | Larger individual experts than Qwen3 MoE |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-26-gpt-oss-overview.png]]
*gpt-oss-20b and gpt-oss-120b architecture overview — from Raschka*

## Key Architecture Choices

- **Sliding window on alternating layers**: every other layer uses a sliding window attention pattern; remaining layers use full attention. This differs from Gemma 3's 5:1 ratio approach.
- **Attention bias units**: uses bias terms in attention layers, which is uncommon in modern LLMs (most have attention_bias=false). These biases provide learned offsets to attention logits.
- **Learned attention sink logits**: per-head learned bias logits for attention sinks — designated positions that absorb excess attention mass. This addresses the attention sink phenomenon with a learned rather than heuristic solution.
- **Larger experts (32 total, 4 active)**: fewer experts than Qwen3 MoE (128) or DeepSeek V3 (256), but each expert is larger, creating a coarser-grained MoE.
- **First OpenAI open weights**: architecturally significant as the first public look at OpenAI's internal design choices.

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
| [[evals/gpqa]] | — | |
| [[evals/ifeval]] | — | |
| [[evals/livecodebench]] | — | |

## Blaz Notes

- 

## Related Notes

- Compare MoE: [[notes/model-cards/2024-deepseek-v3]], [[notes/model-cards/2025-qwen3-moe]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Architecture dimensions not available from Raschka — needs primary source.
- Attention sink mechanism is novel for open-weight models; no concept note exists yet.
