---
arxiv: '2605.25624'
authors:
- Bowen Wang
- Dunjie Lu
- Junli Wang
- Tianyi Bai
- Shixuan Liu
- Zhipeng Zhang
- Haiquan Wang
- Hao Hu
- Tianbao Xie
- Shuai Bai
- Dayiheng Liu
- Que Shen
- Junyang Lin
- Tao Yu
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.25624
  raw: '[[raw/papers/md/2026-cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use]]'
  source: https://arxiv.org/abs/2605.25624
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use.md
raw_pdf: raw/papers/pdf/2026-cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use.pdf
read: false
slug: cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use
tags:
- type/paper
- status/stub
title: 'CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use
  Agents'
type: note
updated: '2026-05-28'
year: 2026
---

# CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents

> *Bowen Wang, Dunjie Lu, Junli Wang, Tianyi Bai, Shixuan Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has driven breakthroughs in domains such as math, tool-use, and software engineering, yet its extension to computer-use agents (CUAs) has been bottlenecked by the scarcity of scalable training data with deterministic rewards. Constructing such data for CUAs requires consistent task instruction, executable environment, and verifiable reward. However, hand-curated benchmarks achieve high reward fidelity but cover few applications and LLM-as-judge-based datasets scale broadly but lack reliable verification. We present CUA-Gym, a scalable pipeline that co-generates task instructions, environment states, and reward functions. Concretely, a Generator agent constructs the initial and golden environment states, and a separate Discriminator agent writes the reward function from the task specification. An orchestrator agent drives the two through iterative rounds upon execution. Generated tuples then pass a final filter combining LLM majority voting and agent rollouts, ensuring quality beyond the per-task adversarial loop. To address the scarcity of training environments, we further synthesize CUA-Gym-Hub, a broad suite of high-fidelity mock web applications grounded in real-world software-use distributions, expanding the scale of CUA RLVR data by magnitude. Using this pipeline, we construct CUA-Gym, a dataset of 32,112 verified RLVR training tuples grounded in 110 environments. Trained with GSPO on CUA-Gym, our CUA-Gym-A3B and CUA-Gym-A17B achieve 62.1% and 72.6% on OSWorld-Verified, outperforming prior open-source CUAs at comparable scales, with performance scaling smoothly in both data volume and environment diversity. The same checkpoints also improve on the held-out WebArena benchmark, indicating transfer beyond the training environments. We will open-source the full synthesis pipeline, dataset, CUA-Gym-Hub environments, and models.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.25624>
- PDF: [[raw/papers/pdf/2026-cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use.pdf]]
- Raw markdown: [[raw/papers/md/2026-cua-gym-scaling-verifiable-training-environments-and-tasks-for-computer-use]]
