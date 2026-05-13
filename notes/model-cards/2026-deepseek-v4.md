---
created: 2026-05-04
developer: '[needs verification]'
family: DeepSeek V4 (Pro & Flash)
kind: model-card
links:
  code: null
  paper: null
  raw: '[[raw/articles/2026-raschka-llm-architecture-gallery]]'
  source: https://sebastianraschka.com/llm-architecture-gallery/
model_type: llm
owner: blaz
read: false
slug: deepseek-v4
tags:
- type/model-card
- status/stub
- domain/models
- confidential/public-source
- model-type/llm
title: DeepSeek V4 (Pro & Flash)
type: note
updated: '2026-05-10'
year: 2026
---

# DeepSeek V4 (Pro & Flash)

DeepSeek's most architecture-heavy release this cycle. Both **V4-Pro** (1.6T) and **V4-Flash** (284B) introduce the same novel attention/connection stack — **MLA-style CSA/HCA with manifold-constrained hyper-connections (mHC)** — extend native context to **1,048,576 tokens (1M)**, and use **hash-based routing** in the MoE path. This is a clear architectural step beyond the V3 / V3.2 lineage rather than another scale increment.

## Model Family

| Property | Value |
|---|---|
| Developer | DeepSeek |
| Release date | 2026-04-24 |
| Family | DeepSeek V4 |
| Variants | V4-Pro (1.6T total, 49B active); V4-Flash (284B total, 13B active) |
| License | [nv] |

## Architecture

Source: Raschka LLM Architecture Gallery cards.

### Shared

| Property | Value |
|---|---|
| Attention | MLA-style CSA / HCA (compressed sparse attention / hyper-compressed attention) with mHC |
| Connections | Manifold-constrained hyper-connections (mHC) |
| Routing | Hash-based routing (early layers per Blaz framing); MTP-capable ("V4 MTP path") |
| Positional encoding | [nv] (RoPE assumed; verify) |
| Normalization | [nv] |
| Activation | [nv] |
| Vocab size | [nv] |
| Max context length | 1,048,576 (1M) |

### V4-Flash (284B)

| Property | Value |
|---|---|
| Parameters (total) | 284B |
| Parameters (active) | 13B (4.6% activation ratio) |
| Layers | 43 (CSA / HCA) |
| Total experts | 256 |
| Active experts | 6 routed + 1 shared = 7 per token |
| Shared experts | 1 |
| KV cache (per Raschka) | 5.4 KiB · "very low" class |
| Decoder type | Sparse MoE |

### V4-Pro (1.6T)

| Property | Value |
|---|---|
| Parameters (total) | 1.6T |
| Parameters (active) | 49B (3.1% activation ratio) |
| Layers | 61 (CSA / HCA) |
| Total experts | 384 |
| Active experts | 6 routed + 1 shared = 7 per token |
| Shared experts | 1 |
| KV cache (per Raschka) | 7.7 KiB · "very low" class |
| Decoder type | Sparse MoE |

## Key Architecture Choices

- **Manifold-constrained hyper-connections (mHC)**: novel residual / connection mechanism flagged in the gallery as central to V4. Mechanism not explained by Raschka — needs the DeepSeek V4 technical report. Hyper-connections in prior literature generalize residual streams across layers; the "manifold-constrained" qualifier suggests the connections are restricted to a learned manifold rather than the full embedding space. **[concept needed]** once tech report is out.
- **CSA / HCA — compressed (sparse) attention variants**: replaces the V3.2 "MLA + DeepSeek Sparse Attention" stack. Gallery describes V4-Flash as "keeps the million-token architecture while reducing the MoE scale" and V4-Pro as introducing "compressed sparse attention plus manifold-constrained hyper-connections for million-token contexts." The very low KV-cache class (5.4–7.7 KiB) is the operational signal — far below DeepSeek V3.2's footprint at the same context — so the compression delivers real per-token KV-cache savings.
- **Hash-based routing (early layers)**: per Blaz's framing, early layers route via hash rather than learned gating. Hash routing trades adaptivity for zero-cost, deterministic load balancing — sensible at the start of the stack where signal is still general. Gallery confirms hash-based routing for V4-Flash; assumed shared with V4-Pro.
- **1M-token native context**: 5× DeepSeek V3.2's 128K. Achieved through CSA + mHC rather than RoPE / YaRN scaling alone; the very-low KV cache class makes the 1M context economically tractable.
- **Pro / Flash split, shared architecture**: V4-Pro at 1.6T / 49B active and V4-Flash at 284B / 13B active share the layer-stack design (CSA/HCA + mHC + hash routing + MTP). Pro doubles routed expert count (384 vs 256) and depth (61 vs 43) over Flash.
- **Tighter expert configuration vs V3**: 256-expert Flash with 6 routed + 1 shared returns to V3's 256-routed scheme but with fewer active experts (6 vs V3's 8). Pro's 384 routed experts is the largest DeepSeek expert pool to date.
- **MTP retained**: "V4 MTP path" preserves multi-token prediction for speculative decoding.

## Training

| Property | Value |
|---|---|
| Training tokens | [needs verification] |
| Stages | [needs verification] |
| Data mix (high-level) | [needs verification] |
| Compute | [needs verification] |
| Optimizer | [needs verification] |

## Reported Evals

Scores not in the gallery cards.

| Eval | Score | Notes |
|---|---|---|
| [[evals/mmlu]] | — | |
| [[evals/mmlu-pro]] | — | |
| [[evals/math-500]] | — | |
| [[evals/humaneval]] | — | |
| [[evals/gpqa]] | — | |
| [[evals/livecodebench]] | — | |
| [[evals/aime-2025]] | — | |
| [[evals/swe-bench-verified-harbor]] | — | |

## Blaz Notes

-

## Related Notes

- Predecessors: [[notes/model-cards/2024-deepseek-v3]], [[notes/model-cards/2025-deepseek-v3-2]]
- Compare 1T+ MoE: [[notes/model-cards/2025-kimi-k2]], [[notes/model-cards/2026-kimi-k2-6]]
- Compare DeepSeek-attention adopters: [[notes/model-cards/2026-glm-5-1]]
- Source: [[raw/articles/2026-raschka-llm-architecture-gallery]]

## Caveats

- mHC, CSA, and HCA are not defined in the gallery — these are novel terms whose mechanisms need the DeepSeek V4 technical report. Treat current architecture statements as headline-level only.
- Whether hash routing applies to all layers or only "early layers" is Blaz's framing in the request prompt; gallery only confirms hash routing exists. Verify with technical report.
- KV-cache class "very low" at 1M context is the most operationally striking claim — implies compression ratios well beyond MLA alone. Worth a dedicated concept note once the mechanism is published.
- 1.6T (Pro) is the largest open-weight LLM at release if confirmed — exceeds Kimi K2's 1T. License needs verification.
