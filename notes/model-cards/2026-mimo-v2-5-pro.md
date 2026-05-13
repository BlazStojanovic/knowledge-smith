---
created: '2026-05-11'
developer: Xiaomi
family: MiMo-V2.5
kind: model-card
license: MIT
links:
  code: null
  paper: null
  raw: null
  source: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
model_type: llm
owner: blaz
parameters_active: 42B
parameters_total: 1.02T
read: false
slug: mimo-v2-5-pro
tags:
- type/model-card
- status/stub
- domain/models
title: MiMo-V2.5-Pro
type: note
updated: '2026-05-11'
variants:
- MiMo-V2.5-Pro
- MiMo-V2.5-Pro-Base
year: 2026
---

# MiMo-V2.5-Pro

> *Xiaomi MiMo* — released 2026

## Overview

(stub — fill in after reading the card)

## Model Family

| Property | Value |
|---|---|
| Developer    | Xiaomi MiMo |
| Release date | 2026 |
| Family       | MiMo-V2.5 |
| Variants     | MiMo-V2.5-Pro (1M ctx, FP8), MiMo-V2.5-Pro-Base (256K ctx, FP8) |
| License      | MIT |

## Architecture

| Property              | Value           |
|---|---|
| Parameters (total)    | 1.02T |
| Parameters (active)   | 42B |
| Layers                | 70 (1 dense + 69 MoE) |
| Hidden dim            | 6144 |
| Attention             | Hybrid: interleaved SWA + global attention with GQA |
| Heads                 | 128 (8 KV heads) |
| Head dim              | 192 (QK) / 128 (V) |
| Positional encoding   | [needs verification] |
| Normalization         | [needs verification] |
| Activation            | [needs verification] |
| Vocabulary size       | [needs verification] |
| Context length        | 1M tokens (Pro), 256K (Pro-Base) |

### MoE properties

| Property              | Value           |
|---|---|
| Experts (total)       | [needs verification] |
| Experts (active)      | [needs verification] |
| Routing               | [needs verification] |
| Load balancing        | [needs verification] |

## Key Architecture Choices

- Hybrid attention combining interleaved sliding window and global attention with GQA
- Three-layer multi-token prediction (MTP) for ~3x faster inference
- FP8 mixed-precision training
- 1M-token context window

## Training

| Property         | Value             |
|---|---|
| Data tokens      | 27T |
| Optimizer        | [needs verification] |
| Schedule         | [needs verification] |
| Stages           | [needs verification] |
| Compute          | [needs verification] |
| Precision        | FP8 mixed |

## Reported Evals

| Eval | Score | Source |
|---|---|---|
| (pending) | — | — |

## Related

- (link peer MoE model cards / underlying paper as available)

## Caveats

- Multiple architectural fields marked `[needs verification]`; pull from official report when available.

## Source

- HuggingFace: <https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro>
