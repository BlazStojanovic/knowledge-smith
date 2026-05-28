---
arxiv: '2605.24220'
authors:
- Binfeng Xu
- Hao Zhang
- Shaokun Zhang
- Songyang Han
- Mingjie Liu
- Jian Hu
- Shizhe Diao
- Zhenghui Jin
- Yunheng Zou
- Michael Demoret
- Jan Kautz
- Yi Dong
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.24220
  raw: '[[raw/papers/md/2026-polar-agentic-rl-on-any-harness-at-scale]]'
  source: https://arxiv.org/abs/2605.24220
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-polar-agentic-rl-on-any-harness-at-scale.md
raw_pdf: raw/papers/pdf/2026-polar-agentic-rl-on-any-harness-at-scale.pdf
read: false
slug: polar-agentic-rl-on-any-harness-at-scale
tags:
- type/paper
- status/stub
title: 'Polar: Agentic RL on Any Harness at Scale'
type: note
updated: '2026-05-28'
year: 2026
---

# Polar: Agentic RL on Any Harness at Scale

> *Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning for language agents increasingly depends on custom harnesses that manage long-running context, multi-turn tool use and multi-agent orchestration. However, porting these harnesses into RL environment interfaces remains difficult and often loses important training signals. We bridge this gap with polar, a rollout framework for scalable asynchronous RL over arbitrary agent harnesses. Polar treats the agent harness as a black box: it proxies LLM API calls, records token-level model interactions, and reconstructs token-faithful trajectories for training. Each rollout node efficiently manages runtime prewarming, agent execution, trajectory reconstruction, and evaluation in parallel, exposing asynchronous service endpoints that can be consumed by independent trainers at scale. This decoupled design makes Polar agnostic to agent harnesses, training infrastructure, and RL algorithms while improving compute utilization for long-running agent workloads. We validate polar by training agents on software-engineering tasks with popular coding harnesses. Using simple GRPO, polar improves Qwen3.5-4B by 22.6, 4.8, 0.6 and 6.2 points on SWE-Bench Verified with the Codex, Claude Code, Qwen Code and Pi harnesses, respectively. We further demonstrate Polar for offline data generation over custom harnesses and ablate trajectory reconstruction strategies. Polar rewrites its preceding work, Prorl Agent, and has been registered as one of NeMo Gym environments.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.24220>
- PDF: [[raw/papers/pdf/2026-polar-agentic-rl-on-any-harness-at-scale.pdf]]
- Raw markdown: [[raw/papers/md/2026-polar-agentic-rl-on-any-harness-at-scale]]
