---
arxiv: '2412.21139'
authors:
- '[needs verification]'
created: '2026-05-10'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2412.21139
  raw: null
  source: https://arxiv.org/abs/2412.21139
owner: blaz
raw_pdf: raw/papers/pdf/2024-swe-gym.pdf
read: false
slug: swe-gym
tags:
- type/paper
- source/primary
- status/verified
- domain/code
- domain/agents
- stage/sft
title: SWE-Gym
type: note
updated: '2026-05-10'
year: 2024
---

# SWE-Gym

- **arXiv**: [2412.21139](https://arxiv.org/abs/2412.21139)
- **Authors / affiliation**: Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang
- **Year / venue**: 2024, ICML 2025
- **Raw**: [[raw/papers/pdf/2024-swe-gym]]
- **Grounding axis**: [[maps/grounding/repository-multifile]]
- **Output shape**: Agent trajectories (real rollouts) + verifier training data
- **Filter / verification**: Hidden-test execution — only trajectories whose final patch passes the hidden tests are kept.
- **Training stage**: Trajectory-SFT + verifier training

## Method

First open training environment for real-world SWE agents. Contains **2,438** Python task instances, each comprising:

- a codebase snapshot,
- an executable runtime environment with pre-installed dependencies,
- expert-validated unit tests,
- a task specified in natural language.

Tasks are sourced from pull requests in 11 popular Python repositories, in alignment with SWE-Bench's construction.

Training recipe:

1. Run existing agents (e.g. OpenHands) on SWE-Gym tasks to collect trajectories.
2. Filter trajectories by execution success on hidden tests.
3. Fine-tune the agent policy on the filtered trajectories (trajectory-SFT).
4. Train verifiers on the outcome labels for inference-time selection.

## Key result

- Up to **+19% absolute** gain in resolve rate on SWE-Bench Verified and Lite when used to train LM-based SWE agents.
- Combined SFT-agent + inference-time verifier scaling: **32.0%** on SWE-Bench Verified and **26.0%** on Lite — reported as new state-of-the-art for open-weight SWE agents at release.

## Notes

- Trajectories are *real* rollouts (executed tools over real repos), not LM-fabricated. Filtering is execution-based.
- Contrast with [[notes/papers/2025-r2e-gym]] (procedurally generated environments) and [[notes/papers/2025-swe-rl]] (rule-based-reward RL, no tests required). Together these three frame the current [[concepts/trajectory-synthesis]] frontier.
- Code: [github.com/SWE-Gym/SWE-Gym](https://github.com/SWE-Gym/SWE-Gym). HF: [huggingface.co/SWE-Gym](https://huggingface.co/SWE-Gym).
