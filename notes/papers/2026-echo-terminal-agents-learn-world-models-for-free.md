---
authors:
- Vaishnavi Shrivastava
- Ahmed Awadallah
- Dimitris Papailiopoulos
created: '2026-05-18'
kind: paper
links:
  code: https://github.com/microsoft/echo-rl
  paper: https://github.com/microsoft/echo-rl/blob/main/echo.pdf
  raw: '[[raw/papers/md/2026-echo-terminal-agents-learn-world-models-for-free]]'
  source: https://github.com/microsoft/echo-rl
owner: blaz
parser: read
raw_md: raw/papers/md/2026-echo-terminal-agents-learn-world-models-for-free.md
raw_pdf: raw/papers/pdf/2026-echo-terminal-agents-learn-world-models-for-free.pdf
read: false
slug: echo-terminal-agents-learn-world-models-for-free
tags:
- type/paper
- status/stub
- reinforcement-learning
- agents
- world-models
- llm
title: 'ECHO: Terminal Agents Learn World Models for Free'
type: note
updated: '2026-05-18'
year: 2026
---

# ECHO: Terminal Agents Learn World Models for Free

> *Vaishnavi Shrivastava, Ahmed Awadallah, Dimitris Papailiopoulos* — Microsoft Research, 2026

## TL;DR

ECHO (Environment Cross-entropy Hybrid Objective) is an auxiliary loss for multi-turn agent RL: alongside the standard GRPO policy-gradient loss on **action** tokens, it adds a cross-entropy loss that trains the policy to **predict the environment-observation tokens** (stdout, errors, file contents, logs, traces) that its own actions produce — `L_ECHO = L_GRPO(A) + λ·L_Env(O')`. The insight: GRPO's sparse outcome-level reward discards the rich environment-response signal already sitting in every rollout, so failed rollouts yield almost no policy-gradient signal even though they contain evidence about how the environment behaves. ECHO reuses the **same forward pass** as GRPO — no teacher model, no extra rollouts. It roughly doubles GRPO pass@1 on TerminalBench-2.0 (Qwen3-8B 2.70→5.17%, Qwen3-14B 5.17→10.79%), produces policies that better predict terminal dynamics even on held-out trajectories, matches expert-SFT-then-GRPO **without** the ~15k expert demos, and — with the environment-prediction loss alone — can drive verifier-free self-improvement on unseen OOD tasks. Framed as terminal agents picking up a "world model" for free. (Summary from abstract + intro; note unread.)

## Abstract

CLI agents are the closest thing language models have to an embodied setting: the model emits commands, the terminal executes them, and the returned stream — stdout, errors, file contents, logs, and traces — records the consequences of its actions. We argue that this stream is a supervision signal, but standard agent RL largely discards it: GRPO-style training updates action tokens with sparse outcome-level rewards while ignoring environment responses, even though rollouts already contain them. As a result, failed rollouts often provide little policy-gradient signal, even though they contain rich evidence about how the environment responds. We introduce ECHO (Environment Cross-entropy Hybrid Objective), a hybrid loss objective that combines the standard policy-gradient loss on action tokens with an auxiliary loss that trains the policy to predict the environment observation tokens resulting from its own actions. ECHO reuses the same forward pass as GRPO, requires no additional rollouts, and turns terminal feedback into dense supervision that enables learning from all rollouts. ECHO roughly doubles GRPO pass@1 on TerminalBench-2.0: Qwen3-8B improves from 2.70% to 5.17%, and Qwen3-14B from 5.17% to 10.79%. We find that ECHO produces policies that are better at predicting terminal dynamics, even for trajectories they didn't generate: across held-out rollouts, it sharply reduces environment-token cross-entropy while GRPO alone barely changes it. From base Qwen3-8B, ECHO matches expert-SFT-then-GRPO performance on held-out terminal tasks without the expert demonstrations required for SFT, and recovers roughly half of the expert-SFT initialization benefit on TerminalBench-2.0. In certain settings, ECHO with only environment prediction loss can enable verifier-free self-improvement, allowing policies to improve on unseen OOD tasks purely by learning from environment interactions. Together, these results suggest that environment observations are not merely context for future actions, but a dense, on-policy supervision signal already present in every agent rollout.

## Notes

(stub)

## Source

- Code / paper: <https://github.com/microsoft/echo-rl>
- Announcement: <https://x.com/DimitrisPapail/status/2056368948870811746>
- PDF: [[raw/papers/pdf/2026-echo-terminal-agents-learn-world-models-for-free.pdf]]
- Raw markdown: [[raw/papers/md/2026-echo-terminal-agents-learn-world-models-for-free]]
