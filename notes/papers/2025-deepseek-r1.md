---
arxiv: '2501.12948'
authors:
- DeepSeek-AI
- Daya Guo
- Dejian Yang
- Haowei Zhang
- Junxiao Song
- Peiyi Wang
- Qihao Zhu
- Runxin Xu
- et al. (~200 authors)
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2501.12948
  raw: https://arxiv.org/pdf/2501.12948
  source: https://arxiv.org/abs/2501.12948
owner: blaz
read: false
slug: deepseek-r1
tags:
- type/paper
- source/primary
- status/stub
- domain/reasoning
- domain/math
- domain/code
- stage/rl-rule
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning'
type: note
updated: '2026-05-10'
year: 2025
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## Citation

- URL: https://arxiv.org/abs/2501.12948
- Authors: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, et al. (~200 authors)
- Year / venue: 2025 / arXiv (also published in Nature vol. 645, pp. 633–638)
- arXiv: [2501.12948](https://arxiv.org/abs/2501.12948)

## Core Claim

Reasoning abilities in LLMs can be incentivized through pure reinforcement learning with verifiable rewards, without human-labelled reasoning trajectories. The trained model exhibits emergent advanced reasoning patterns (self-reflection, verification, dynamic strategy adaptation) and achieves performance comparable to OpenAI o1-1217.

## Key Paper Ideas

- **R1-Zero**: pure RL (GRPO) on DeepSeek-V3-Base with no SFT cold-start data. Uses only accuracy rewards (rule-based verification for math, compiler + tests for code) and format rewards (thinking in `<think>` tags). No neural reward model — explicitly avoided to prevent reward hacking.
- **Emergent behaviours**: self-verification, extended thinking (hundreds to thousands of tokens), "aha moment" where the model learns to re-evaluate its initial approach, language mixing.
- **Full R1 pipeline** (4 stages): cold-start SFT → reasoning-oriented RL (GRPO) → rejection sampling + SFT (~600k reasoning + ~200k non-reasoning) → secondary RL for all scenarios.
- **Distillation**: SFT-only distillation into Qwen2.5 (1.5B–32B) and Llama-3.1/3.3 (8B–70B) using 800k curated R1 samples. Distillation significantly outperforms applying RL directly to smaller models.

## Key Results

| Benchmark | R1-Zero | R1 (full) |
|---|---|---|
| AIME 2024 | 71.0% pass@1 (from 15.6% base) | 79.8% pass@1 |
| MATH-500 | — | 97.3% pass@1 |
| Codeforces | — | 2,029 Elo (96.3rd percentile) |
| GPQA Diamond | — | 71.5% |
| MMLU | — | 90.8% |

R1-Zero with majority voting: 86.7% on AIME 2024 (comparable to OpenAI o1-0912).

Distillation results: 7B achieves 55.5% AIME (surpasses QwQ-32B-Preview); 32B achieves 72.6% AIME, 94.3% MATH-500; 70B achieves 70.0% AIME, 94.5% GPQA Diamond.

## Core Concepts

- Existing concepts: [[concepts/rlvr]], [[concepts/grpo]], [[concepts/process-vs-outcome-reward]], [[concepts/verification-signals]]
- Concepts to extract: emergent reasoning from RL, distillation vs. direct RL for small models

## Relevance To Poolside

*Our interpretation*: canonical demonstration that RLVR + GRPO is sufficient for frontier reasoning without expensive human preference data. The distillation result (SFT from R1 trajectories outperforms direct RL on small models) is load-bearing for Poolside's training pipeline design — it suggests reasoning capability can be "compressed" into smaller models via trajectory distillation rather than running expensive RL on each model size.

## Key Follow-Ups / Jumping-Off Points

- [[notes/papers/2024-deepseekmath-pushing-the-limits-of-mathematical-reasoning]] — introduces GRPO, reused here
- DeepSeek-Prover-V2 (arXiv [2504.21801](https://arxiv.org/abs/2504.21801)) — extends to formal proofs
- [[concepts/trajectory-synthesis]] — R1 trajectories as distillation data (archetype 3)
- [[concepts/reasoning-data-generation]] — R1 is the canonical instance of archetype 3 (rewarded-rollout trajectory harvesting)

## Related Notes

- Concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rlvr]], [[concepts/grpo]]
- Maps: [[maps/rl-environments/landscape]]
