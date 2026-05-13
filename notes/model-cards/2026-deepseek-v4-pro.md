---
created: '2026-05-11'
developer: DeepSeek
family: DeepSeek-V4
kind: model-card
license: MIT
links:
  code: null
  paper: null
  raw: null
  source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
model_type: llm
owner: blaz
parameters_active: 49B
parameters_total: 1.6T
read: false
slug: deepseek-v4-pro
tags:
- type/model-card
- status/stub
- domain/models
title: DeepSeek-V4-Pro
type: note
updated: '2026-05-11'
variants:
- V4-Pro-Base
- V4-Pro
- V4-Flash-Base
- V4-Flash
year: 2026
---

# DeepSeek-V4-Pro

> *DeepSeek AI* — released 2026 (preview)

## Overview

(stub — fill in after reading the card)

## Model Family

| Property | Value |
|---|---|
| Developer    | DeepSeek AI |
| Release date | 2026 |
| Family       | DeepSeek-V4 |
| Variants     | V4-Pro-Base (FP8), V4-Pro (FP4+FP8), V4-Flash-Base (284B/13B), V4-Flash (284B/13B) |
| License      | MIT |

## Architecture

| Property              | Value           |
|---|---|
| Parameters (total)    | 1.6T |
| Parameters (active)   | 49B |
| Layers                | [needs verification] |
| Hidden dim            | [needs verification] |
| Attention             | Hybrid: Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA) |
| Heads                 | [needs verification] |
| Positional encoding   | [needs verification] |
| Normalization         | [needs verification] |
| Activation            | [needs verification] |
| Vocabulary size       | [needs verification] |
| Context length        | 1M tokens |

### MoE properties

| Property              | Value           |
|---|---|
| Experts (total)       | [needs verification] |
| Experts (active)      | [needs verification] |
| Routing               | [needs verification] |
| Load balancing        | [needs verification] |

## Key Architecture Choices

- Hybrid attention: CSA + HCA reduces single-token inference FLOPs to 27% vs V3.2
- Three reasoning modes: Non-think, Think High, Think Max
- Mixed-precision FP4/FP8 inference for Pro tier

## Training

| Property         | Value             |
|---|---|
| Data tokens      | [needs verification] |
| Optimizer        | [needs verification] |
| Schedule         | [needs verification] |
| Stages           | [needs verification] |
| Compute          | [needs verification] |
| Precision        | FP8 (base), FP4+FP8 (Pro) |

## Reported Evals

| Eval | Score | Source |
|---|---|---|
| (pending) | — | — |

## Related

- (link DeepSeek-V3 / V3.2 model cards as available)

## Caveats

- Most architecture/training fields `[needs verification]` — HF card is sparse; await technical report.

## Source

- HuggingFace: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
