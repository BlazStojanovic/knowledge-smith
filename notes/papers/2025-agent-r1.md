---
arxiv: '2511.14460'
authors:
- Mingyue Cheng
- Jie Ouyang
- Shuo Yu
- et al
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2511.14460
  raw: null
  source: https://arxiv.org/abs/2511.14460
owner: blaz
read: false
slug: agent-r1
tags:
- type/paper
- source/primary
- status/stub
- domain/agents
- domain/training
- stage/rl-exec
title: 'Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning'
type: note
updated: '2026-05-10'
year: 2025
---

# Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning

## Citation

- URL: https://arxiv.org/abs/2511.14460
- Authors: Mingyue Cheng, Jie Ouyang, Shuo Yu, et al.
- Year / venue: 2025 / arXiv
- arXiv: [2511.14460](https://arxiv.org/abs/2511.14460)

## Core Claim

Extends the MDP framework to comprehensively define LLM agent components and introduces Agent-R1, a modular RL training framework for LLM agents. Argues that RL tailored specifically for the LLM agent context (tool use, environment interaction) is underdeveloped and provides both a formalism and a flexible framework.

## Key Paper Ideas

- Formalises the LLM agent as an MDP with tool-use actions and environment observations.
- Modular framework: separates environment interfaces, reward computation, and policy training.
- Validated on multi-hop QA benchmarks.
- Handles heterogeneous environment interfaces (different agent tasks, different sandboxes) with consistent abstractions.

## Core Concepts

- Existing concepts: [[concepts/rl-for-llm-post-training]], [[concepts/rl-environment-construction]], [[concepts/trajectory-synthesis]]
- Concepts to extract: MDP formalism for LLM agents, consistent environment interfaces for heterogeneous tasks

## Relevance To Poolside

*Our interpretation*: provides a practical framework and formalism for extending RL beyond single-turn code/math to multi-turn agent tasks. Relevant if Poolside wants to train agent capabilities via RL rather than just SFT on trajectories.

## Related Notes

- Concepts: [[concepts/rl-training-frameworks]], [[concepts/rl-environment-construction]]
- Maps: [[maps/rl-environments/landscape]]
