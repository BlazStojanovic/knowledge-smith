---
arxiv: '2402.03300'
authors:
- DeepSeek-AI
- Zhihong Shao
- Peiyi Wang
- Qihao Zhu
- Runxin Xu
- Junxiao Song
- et al
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2402.03300
  raw: null
  source: https://arxiv.org/abs/2402.03300
owner: blaz
read: false
slug: deepseekmath-pushing-the-limits-of-mathematical-reasoning
tags:
- type/paper
- source/primary
- status/stub
- domain/math
- domain/reasoning
- stage/rl-rule
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
type: note
updated: '2026-05-10'
year: 2024
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

Introduces Group Relative Policy Optimization (GRPO) for scalable preference optimization in mathematical reasoning.

- **Authors**: DeepSeek-AI, Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, et al.
- **Venue**: arXiv 2024
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)
- **Raw**: [[raw/papers/pdf/2024-deepseekmath]]

## Core contribution

Achieves near-frontier mathematical reasoning with a 7B model through large-scale math-focused pretraining (120B math tokens from Common Crawl) and Group Relative Policy Optimization (GRPO). GRPO normalizes rewards across a group of sampled outputs rather than using a separate critic model, enabling scalable RL without the overhead of a value network.

## Connections

- Concept: [[concepts/grpo]] — full treatment of the GRPO algorithm introduced here
- GRPO is reused in [[notes/papers/2025-deepseek-r1]] for reasoning training
- [[concepts/rl-for-llm-post-training]] — foundational RL framing
- [[concepts/rlvr]] — RLVR paradigm that GRPO enables
