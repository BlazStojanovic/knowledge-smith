---
arxiv: '2506.14245'
authors:
- Xumeng Wen
- Zihan Liu
- Shun Zheng
- Shengyu Ye
- Zhirong Wu
- Yang Wang
- Zhijian Xu
- Xiao Liang
- Junjie Li
- Ziming Miao
- Jiang Bian
- Mao Yang
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2506.14245
  raw: null
  source: https://arxiv.org/abs/2506.14245
owner: blaz
read: false
slug: rlvr-implicitly-incentivizes-reasoning
tags:
- type/paper
- source/primary
- status/stub
- domain/reasoning
- domain/training
title: Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct
  Reasoning in Base LLMs
type: note
updated: '2026-05-10'
year: 2025
---

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

## Citation

- URL: https://arxiv.org/abs/2506.14245
- Authors: Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang
- Year / venue: 2025 / arXiv
- arXiv: [2506.14245](https://arxiv.org/abs/2506.14245)

## Core Claim

RLVR (as used in DeepSeek-R1's GRPO) genuinely enhances reasoning abilities rather than merely boosting sampling efficiency. Provides a theoretical framework showing how RLVR encourages correct reasoning even when rewards are based solely on answer correctness (not intermediate steps).

## Key Paper Ideas

- Revisits Pass@K experiments and demonstrates RLVR extends the reasoning boundary for both math and coding tasks.
- Introduces **CoT-Pass@K**, a novel metric capturing reasoning success by accounting for both the final answer and intermediate reasoning steps.
- Presents a theoretical framework explaining RLVR's incentive mechanism.
- Training dynamics analysis shows RLVR incentivizes correct reasoning early in the training process.

## Core Concepts

- Existing concepts: [[concepts/rlvr]], [[concepts/grpo]], [[concepts/process-vs-outcome-reward]]
- Concepts to extract: CoT-Pass@K metric (extends pass@k to account for reasoning quality, not just answer correctness)

## Relevance To Poolside

*Our interpretation*: resolves a key open question — whether RLVR teaches new reasoning or just makes existing capabilities more accessible. The paper argues for genuine enhancement, which strengthens the case for investing in RLVR infrastructure.

## Key Follow-Ups / Jumping-Off Points

- [[notes/papers/2025-deepseek-r1]] — the system whose training regime this paper analyses
- "Does RL Really Incentivize Reasoning Beyond Base Model" (arXiv [2504.13837](https://arxiv.org/abs/2504.13837)) — opposing perspective [ID?]
- [[concepts/rlvr]] — the RLVR concept note

## Related Notes

- Concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rlvr]], [[concepts/grpo]]
