---
created: 2026-04-23
developer: '[needs verification]'
family: DeepSeek V3
kind: model-card
links:
  code: https://github.com/deepseek-ai/DeepSeek-V3
  paper: null
  raw: '[[raw/articles/2026-raschka-big-llm-architecture-comparison]]'
  source: https://arxiv.org/abs/2412.19437
model_type: llm
owner: blaz
read: false
slug: deepseek-v3
tags:
- type/model-card
- status/draft
- domain/models
- confidential/public-source
- model-type/llm
title: DeepSeek V3
type: note
updated: '2026-05-10'
year: 2024
---

# DeepSeek V3

DeepSeek's flagship MoE model (Dec 2024). Introduces Multi-Head Latent Attention (MLA) for KV cache compression and auxiliary-loss-free load balancing. DeepSeek-R1 (Jan 2025) uses the same architecture with RL-based training.

## Model Family

| Property | Value |
|---|---|
| Developer | DeepSeek |
| Release date | 2024-12-26 |
| Family | DeepSeek-V3 |
| Variants | DeepSeek-V3 (671B), DeepSeek-R1 (671B, same arch, RL training) |
| License | MIT |

## Architecture

Source: [HuggingFace config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json).

| Property               | Value                                         |
| ---------------------- | --------------------------------------------- |
| Parameters (total)     | 671B                                          |
| Parameters (active)    | 37B                                           |
| Layers                 | 61                                            |
| Hidden dim (d_model)   | 7168                                          |
| FFN dim (dense layers) | 18432                                         |
| Attention heads        | 128                                           |
| KV heads               | 128 (compressed via MLA)                      |
| KV LoRA rank           | 512                                           |
| Q LoRA rank            | 1536                                          |
| QK nope head dim       | 128                                           |
| QK RoPE head dim       | 64                                            |
| V head dim             | 128                                           |
| Vocab size             | 129,280                                       |
| Max context length     | 163,840 (pretrained at 4K, extended via YaRN) |
| Attention              | Multi-Head Latent Attention (MLA)             |
| Positional encoding    | RoPE + YaRN (factor=40)                       |
| Normalization          | RMSNorm (pre-norm)                            |
| Activation             | SiLU                                          |
| Tie embeddings         | No                                            |

| MoE Property            | Value                                |
| ----------------------- | ------------------------------------ |
| Total experts           | 256 routed                           |
| Active experts          | 8 routed + 1 shared = 9 per token    |
| Shared experts          | 1                                    |
| Expert intermediate dim | 2048                                 |
| Dense layers            | First 3 layers (no MoE)              |
| Routing                 | Sigmoid scoring, top-k with noaux_tc |
| Expert grouping         | 8 groups, top-4 groups selected      |

## Architecture Diagrams

![[raw/images/articles/raschka-llm-architecture/figure-03-mha-vs-mla.png]]
*MHA vs MLA — from Raschka, "The Big LLM Architecture Comparison"*

![[raw/images/articles/raschka-llm-architecture/figure-05-moe-vs-ffn.png]]
*DeepSeek V3 MoE module vs standard FFN*

![[raw/images/articles/raschka-llm-architecture/figure-17-deepseekv3-vs-llama4.png]]
*DeepSeek V3 vs Llama 4 Maverick architecture comparison*

## Key Architecture Choices

- **Multi-Head Latent Attention (MLA)**: compresses KV cache by projecting keys and values into a low-rank latent space (rank 512) before caching, dramatically reducing KV cache memory. Q also compressed (rank 1536). Decoupled RoPE applied to a separate 64-dim subspace. [[concepts/architecture-scaling]]
- **Auxiliary-loss-free load balancing**: avoids the standard auxiliary loss for expert balancing, preventing degradation of model quality from the balancing objective.
- **Multi-Token Prediction (MTP)**: trained to predict multiple future tokens, used as auxiliary training objective and for speculative decoding at inference.
- **First 3 layers dense**: initial layers use standard dense FFN (intermediate=18432) before MoE routing begins, establishing low-level representations.
- **FP8 mixed-precision training**: first model trained end-to-end with FP8 compute, reducing training cost.
- **Expert grouping**: 256 experts organized into 8 groups of 32; routing selects top-4 groups, then top-2 experts within each selected group (8 total active routed experts).

## Training

| Property | Value |
|---|---|
| Training tokens | 14.8T |
| Stages | Pretrain → SFT → RL |
| Data mix (high-level) | [needs verification from paper] |
| Compute | 2.788M H800 GPU hours |

DeepSeek-R1 applies large-scale RL on top of the V3 base, developing chain-of-thought reasoning without supervised fine-tuning on reasoning traces.

## Reported Evals

Scores to be filled from primary paper. Evals commonly reported:

| Eval | Score | Notes |
|---|---|---|
| [[evals/mmlu]] | — | |
| [[evals/mmlu-pro]] | — | |
| [[evals/math-500]] | — | |
| [[evals/humaneval]] | — | |
| [[evals/gpqa]] | — | |
| [[evals/ifeval]] | — | |
| [[evals/simpleqa]] | — | |
| [[evals/livecodebench]] | — | |
| [[evals/arena-hard-auto]] | — | |
| [[evals/drop]] | — | |
| [[evals/bigcodebench]] | — | |

## Blaz Notes

- 

## Related Notes

- Architecture concepts: [[concepts/architecture-scaling]]
- Source article: [[raw/articles/2026-raschka-big-llm-architecture-comparison]]
- See also: [[notes/model-cards/2025-deepseek-v3-2]] (V3.2 successor)

## Caveats

- Eval scores not yet filled from the paper — marked as "—".
- MLA attention dims are non-standard; KV heads=128 in config but the effective KV cache is compressed to rank 512.
