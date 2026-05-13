---
created: 2026-04-23
developer: '[needs verification]'
family: DeepSeek V3.2
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
model_type: llm
owner: blaz
read: false
slug: deepseek-v3-2
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: DeepSeek V3.2
type: note
updated: '2026-05-10'
year: 2025
---

# DeepSeek V3.2

Incremental update to DeepSeek V3, adding sparse attention mechanisms for improved efficiency. Same base MoE/MLA architecture as V3 with attention-level sparsity on top.

## Model Family

| Property | Value |
|---|---|
| Developer | DeepSeek |
| Release date | 2025 [nv] |
| Family | DeepSeek-V3 |
| Variants | 671B (37B active) |
| License | [nv] |

## Architecture

Source: Raschka comparison article. Shares base architecture with [[notes/model-cards/2024-deepseek-v3]].

| Property | Value |
|---|---|
| Parameters (total) | 671B |
| Parameters (active) | 37B |
| Layers | 61 (same as V3) |
| Hidden dim (d_model) | 7168 (same as V3) |
| Attention | MLA + sparse attention layer |
| Positional encoding | RoPE + YaRN |
| Normalization | RMSNorm (pre-norm) |
| Vocab size | 129,280 (same as V3) |

MoE configuration same as V3 (256 routed + 1 shared, top-8 routing).

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-48-deepseekv32-sparse.png]]
*DeepSeek V3.2 with sparse attention — from Raschka*

## Key Architecture Choices

- **Sparse attention addition**: adds a sparse attention mechanism on top of the MLA architecture from V3, further reducing compute in attention for long sequences.
- **Otherwise identical to V3**: same MoE configuration, MLA, layer structure, and dimensions. The update is attention-level, not architecture-level.

## Training

| Property | Value |
|---|---|
| Training tokens | [needs verification] |
| Stages | [needs verification] |
| Data mix (high-level) | [needs verification] |
| Compute | [needs verification] |

## Reported Evals

Raschka notes V3.2 is "competitive with GPT-5.1 and Gemini 3.0 Pro." Scores to be filled from primary source.

| Eval | Score | Notes |
|---|---|---|
| [[evals/mmlu]] | — | |
| [[evals/mmlu-pro]] | — | |
| [[evals/math-500]] | — | |
| [[evals/humaneval]] | — | |
| [[evals/gpqa]] | — | |
| [[evals/livecodebench]] | — | |

## Blaz Notes

- 

## Related Notes

- Base architecture: [[notes/model-cards/2024-deepseek-v3]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]

## Caveats

- Sparse attention details not yet documented — needs primary source.
- Release date and detailed specs need verification.
