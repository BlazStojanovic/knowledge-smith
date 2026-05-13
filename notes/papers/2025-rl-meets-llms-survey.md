---
arxiv: '2509.16679'
authors:
- Keliang Liu
- Dingkang Yang
- Ziyun Qian
- et al
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2509.16679
  raw: null
  source: https://arxiv.org/abs/2509.16679
owner: blaz
read: false
slug: rl-meets-llms-survey
tags:
- type/paper
- source/primary
- status/stub
- domain/training
- domain/reasoning
title: 'Reinforcement Learning Meets Large Language Models: A Survey of Advancements
  and Applications Across the LLM Lifecycle'
type: note
updated: '2026-05-10'
year: 2025
---

# Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle

## Citation

- URL: https://arxiv.org/abs/2509.16679
- Authors: Keliang Liu, Dingkang Yang, Ziyun Qian, et al.
- Year / venue: 2025 / arXiv
- arXiv: [2509.16679](https://arxiv.org/abs/2509.16679)

## Core Claim

Comprehensive lifecycle survey of how RL empowers LLMs across pre-training, alignment fine-tuning, and reinforced reasoning, with emphasis on RLVR as the pivotal driver for pushing reasoning to its limits.

## Key Paper Ideas

- Covers the full RL-for-LLMs lifecycle: data generation, pretraining augmentation, post-training alignment, test-time inference.
- Collates datasets, evaluation benchmarks (human-annotated, AI-assisted preference, program-verification corpora), and open-source training frameworks.
- Emphasises RLVR as the current frontier, distinct from earlier RLHF approaches.
- Provides a structured overview useful as a reference map for the field.

## Core Concepts

- Existing concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rlvr]], [[concepts/grpo]]
- Concepts to extract: RL across the full LLM lifecycle (not just post-training)

## Relevance To Poolside

*Our interpretation*: good reference survey for understanding where RL fits in the full training pipeline. Useful for contextualising Poolside's RL work within the broader landscape.

## Related Notes

- Concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rl-training-frameworks]]
- Maps: [[maps/rl-environments/landscape]]
