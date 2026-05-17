---
created: 2026-04-23
developer: '[needs verification]'
family: Qwen3 (MoE)
kind: model-card
links:
  code: https://github.com/QwenLM/Qwen3
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://qwenlm.github.io/blog/qwen3/
model_type: llm
owner: blaz
read: false
slug: qwen3-moe
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: Qwen3 (MoE)
type: note
updated: '2026-05-10'
year: 2025
---

# Qwen3 (MoE)

Qwen3's sparse Mixture-of-Experts variants. Notable for using 128 experts with no shared expert — a departure from Qwen2.5-MoE and the DeepSeek approach which include shared experts.

## Model Family

| Property | Value |
|---|---|
| Developer | Alibaba / Qwen |
| Release date | 2025-04 |
| Family | Qwen3 MoE |
| Variants | 30B-A3B (30B total, 3B active), 235B-A22B (235B total, 22B active) |
| License | Apache 2.0 |

## Architecture

Sources: HuggingFace configs for [30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) and [235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B).

| Property | 30B-A3B | 235B-A22B |
|---|---|---|
| Parameters (total) | 30B | 235B |
| Parameters (active) | ~3B | ~22B |
| Layers | 48 | 94 |
| Hidden dim (d_model) | 2048 | 4096 |
| FFN dim (dense) | 6144 | 12288 |
| Expert intermediate dim | 768 | 1536 |
| Attention heads | 32 | 64 |
| KV heads | 4 | 4 |
| Head dim | 128 | 128 |
| Vocab size | 151,936 | 151,936 |
| Max context length | 40,960 | 40,960 |

All variants share:

| Property | Value |
|---|---|
| Attention | GQA |
| Positional encoding | RoPE (theta=1M) |
| Normalization | RMSNorm (pre-norm) |
| Activation | SiLU |
| Tie embeddings | No |

| MoE Property            | 30B-A3B     | 235B-A22B   |
| ----------------------- | ----------- | ----------- |
| Total experts           | 128         | 128         |
| Active experts          | 8           | 8           |
| Shared experts          | 0           | 0           |
| Expert intermediate dim | 768         | 1536        |
| MoE frequency           | Every layer | Every layer |
| Routing aux loss coef   | 0.001       | 0.001       |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-19-deepseekv3-vs-qwen3moe.png]]
*DeepSeek V3 vs Qwen3 235B-A22B MoE comparison — from Raschka*

## Key Architecture Choices

- **No shared expert**: unlike DeepSeek V3 (1 shared expert) and Qwen2.5-MoE, Qwen3 MoE omits shared experts entirely. All capacity is in routed experts.
- **128 experts, top-8 routing**: moderate expert count with high activation ratio (8/128 = 6.25%), contrasting with DeepSeek V3's 256 experts at 3.1% activation.
- **Very deep for active params**: the 235B-A22B uses 94 layers — among the deepest MoE architectures — while activating only 22B params per token.
- **Small expert intermediate dim**: experts use small FFN dims (768 for 30B, 1536 for 235B) relative to the dense FFN dim, creating many fine-grained experts.
- **Consistent with dense Qwen3**: same vocab, RoPE theta, head dim, and normalization as dense variants. See [[notes/model-cards/2025-qwen3]].

## Training

| Property | Value |
|---|---|
| Training tokens | [needs verification] |
| Stages | Pretrain → SFT → RL (thinking mode) |
| Data mix (high-level) | [needs verification] |
| Compute | [needs verification] |

## Reported Evals

Scores to be filled from primary source.

| Eval | Score | Notes |
|---|---|---|
| [[evals/mmlu]] | — | |
| [[evals/mmlu-pro]] | — | |
| [[evals/gpqa]] | — | |
| [[evals/math-500]] | — | |
| [[evals/humaneval]] | — | |
| [[evals/ifeval]] | — | |
| [[evals/livecodebench]] | — | |
| [[evals/arena-hard-auto]] | — | |
| [[evals/aime-2025]] | — | |

## Blaz Notes

- 

## Related Notes

- Dense variants: [[notes/model-cards/2025-qwen3]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]
- Architecture scaling: [[concepts/architecture-scaling]]

## Caveats

- Active parameter counts are approximate (naming convention, not exact from config).
