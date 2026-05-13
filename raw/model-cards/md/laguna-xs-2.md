---
source: model-announcement
url: https://poolside.ai/blog/laguna-a-deeper-dive
retrieved: 2026-05-04
title: "Laguna XS.2 and M.1: A Deeper Dive"
author: Poolside
publication: poolside.ai blog
license: blog-text — model weights Apache 2.0
---

Companion deeper-dive blog to "Introducing Laguna XS.2 and Laguna M.1" (https://poolside.ai/blog/introducing-laguna-xs2-m1). The intro post itself ships placeholder zeros for benchmark numbers; the deeper-dive page is the authoritative public source for Laguna XS.2's reported scores.

## Headline numbers (Laguna XS.2 33B-A3B)

- SWE-bench Verified: 68.2%
- SWE-bench Multilingual: 62.4%
- SWE-bench Pro: 44.5%
- Terminal-Bench 2.0: 30.1%

Statistical approach: "mean pass@1 averaged over" 3–7 runs per benchmark.

## Methodology

- Harness: "the Laude Institute's Harbor Framework with our agent harness"
- Max steps: 500
- Sandbox: 8 GB RAM / 2 CPUs (Terminal-Bench 2.0 exception: 48 GB RAM / 32 CPUs)
- Sampling: temperature=0.7, top_k=20

## Comparator scores (verbatim from the same tables)

### SWE-bench Verified

| Model | Score |
|---|---|
| Laguna XS.2 33B-A3B | 68.2 |
| Devstral Small 2 24B dense | 68.0 |
| Gemma 4 31B dense | 52.0 |
| Qwen3.5 35B-A3B | 69.2 |
| Qwen3.6 35B-A3B | 73.4 |
| Claude Haiku 4.5 | 73.3 |
| GPT-5.4 Nano | — |

### SWE-bench Multilingual

| Model | Score |
|---|---|
| Laguna XS.2 33B-A3B | 62.4 |
| Devstral Small 2 24B dense | 55.7 |
| Gemma 4 31B dense | 51.7 |
| Qwen3.5 35B-A3B | 60.3 |
| Qwen3.6 35B-A3B | 67.2 |
| Claude Haiku 4.5 | — |
| GPT-5.4 Nano | — |

### SWE-bench Pro

| Model | Score |
|---|---|
| Laguna XS.2 33B-A3B | 44.5 |
| Devstral Small 2 24B dense | — |
| Gemma 4 31B dense | 35.7 |
| Qwen3.5 35B-A3B | 44.6 |
| Qwen3.6 35B-A3B | 49.5 |
| Claude Haiku 4.5 | 39.5 |
| GPT-5.4 Nano | 52.4 |

### Terminal-Bench 2.0

| Model | Score |
|---|---|
| Laguna XS.2 33B-A3B | 30.1 |
| Devstral Small 2 24B dense | 22.5 |
| Gemma 4 31B dense | 42.9 |
| Qwen3.5 35B-A3B | 40.5 |
| Qwen3.6 35B-A3B | 51.5 |
| Claude Haiku 4.5 | 29.8 |
| GPT-5.4 Nano | 46.3 |

## Architecture / training (from blog body)

- 33B total parameters, 3B active per token (MoE)
- 30T training tokens
- Stages: pretraining + post-training, including async off-policy agent RL
- Optimizer: Muon (per HuggingFace card)
- Weights: Apache 2.0 on Hugging Face (`poolside/Laguna-XS.2`); also via OpenRouter and Ollama
- "free to use for a limited time via our API"

Per-layer architecture details (40 layers = 30 SWA + 10 global, 3:1 ratio, 512 SWA window, FP8 KV cache, 256 routed experts + 1 shared) come from the HuggingFace model card at https://huggingface.co/poolside/Laguna-XS.2 — not mirrored here (HF card is stable / discoverable).
