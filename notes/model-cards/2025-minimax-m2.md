---
created: 2026-04-23
developer: '[needs verification]'
family: MiniMax-M2
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: minimax-m2
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: MiniMax-M2
type: note
updated: '2026-05-10'
year: 2025
---

# MiniMax-M2

MiniMax's second-generation model. Reverted from M1's linear attention back to full Grouped-Query Attention after determining linear attention was problematic for reasoning. Uses per-layer QK-Norm (unique RMSNorm per attention head) and higher MoE sparsity than M1.

## Model Family

| Property | Value |
|---|---|
| Developer | MiniMax |
| Release date | 2025 [nv] |
| Family | MiniMax-M |
| Variants | 230B |
| License | [nv] |

## Architecture

Source: Raschka comparison article.

Sources: Raschka diagram (Figure 39).

| Property | Value |
|---|---|
| Parameters (total) | 230B |
| Parameters (active) | ~10B |
| Layers | [nv] |
| Hidden dim (d_model) | 3072 |
| Expert input size | 3072 |
| Attention heads | [nv] |
| KV heads | [nv] |
| Vocab size | 191K |
| Max context length | [nv] |
| Attention | GQA (reverted from M1's linear attention) |
| Positional encoding | Partial RoPE (50% of head dimensions, carried over from M1) |
| Normalization | RMSNorm + per-layer QK-Norm |
| Activation | [nv] |

| MoE Property | Value |
|---|---|
| Total experts | [nv] |
| Active experts | 8 |
| Active parameters | ~10B (4.3% of 230B) |
| Routing | [nv] |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-39-qwen3-vs-minimaxm2.png]]
*Qwen3 vs MiniMax-M2 architecture comparison — from Raschka*

## Key Architecture Choices

- **Reverted from linear to full GQA**: after M1's Lightning Attention deployment, MiniMax determined that linear attention was problematic for reasoning tasks and reverted to standard GQA. This is one of the strongest empirical data points against production linear attention.
- **Per-layer QK-Norm**: applies a unique RMSNorm per attention head at each layer, normalizing queries and keys independently per head. More fine-grained than the standard QK-norm used by OLMo 2.
- **Partial RoPE (50%, continued from M1)**: retains M1's approach of applying RoPE to only half of each head's dimensions.
- **Higher sparsity**: ~10B active params from 230B total (4.3% activation), much sparser than M1's 10.1%. Fewer but presumably larger or more specialized experts.
- **Better benchmark performance**: Raschka notes M2 outperforms M1 on benchmarks, validating the reversion to standard attention.

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

- Predecessor: [[notes/model-cards/2025-minimax-m1]] (linear attention, superseded)
- Compare QK-norm: [[notes/model-cards/2025-olmo-2]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Most architecture dimensions not available from Raschka.
- The linear→GQA reversion is architecturally significant but the exact quality delta is not quantified in the Raschka article.
