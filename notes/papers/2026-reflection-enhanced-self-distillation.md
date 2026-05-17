---
authors:
- Yuwei Zhang
- Sha Li
- Changlong Yu
- Qin Lu
- Shuowei Jin
- Chengyu Dong
- Haoran Liu
- Ilgee Hong
- Xintong Li
- Zhenyu Shi
- Bing Yin
- Jingbo Shang
created: '2026-05-15'
kind: paper
links:
  code: https://github.com/horizon-llm/RESD
  paper: null
  raw: '[[raw/papers/md/2026-reflection-enhanced-self-distillation]]'
  source: https://github.com/horizon-llm/RESD
owner: blaz
parser: read
raw_md: raw/papers/md/2026-reflection-enhanced-self-distillation.md
raw_pdf: raw/papers/pdf/2026-reflection-enhanced-self-distillation.pdf
read: false
slug: reflection-enhanced-self-distillation
tags:
- type/paper
- status/stub
- rl
- distillation
- agents
- llm
title: Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation
type: note
updated: '2026-05-15'
venue: Preprint
year: 2026
---

# Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation

> *Yuwei Zhang, Sha Li, Changlong Yu et al. (UC San Diego, Amazon, Georgia Tech)* — Preprint, 2026

## TL;DR

Post-training framework for rare-success regimes. RESD turns raw failure feedback into *active* corrective supervision: it generates retrospective reflections on failed trajectories to diagnose local errors, and curates a persistent global "playbook" of reusable lessons across training steps. This enriched context lets the on-policy self-teacher give actionable token-level supervision even when there are no successful rollouts. Reported to beat standard self-distillation baselines and to reach faster early-stage improvement than GRPO with 8× samples, using a single rollout per prompt.

## Abstract

Enabling Large Language Models (LLMs) to continuously improve from environmental interactions is a central challenge in post-training. While on-policy self-distillation offers a promising paradigm, existing methods predominantly treat environmental feedback as a passive conditioning signal. Consequently, they heavily rely on successful demonstrations and struggle to learn in rare-success regimes. To bridge this gap, we introduce Reflection-Enhanced Self-Distillation (RESD), a framework that transforms raw failure feedback into an active source of corrective supervision. Instead of passively appending feedback, RESD interprets failed trajectories by generating retrospective reflections to diagnose local errors, and curates a persistent global playbook to preserve reusable lessons across training steps. The enriched context enables the self-teacher to provide actionable token-level supervision even in the absence of successful rollouts. Empirical evaluations on multiple continual learning tasks demonstrate that RESD substantially outperforms standard self-distillation baselines. Furthermore, RESD achieves significantly faster early-stage improvement than GRPO with 8× samples using only a single rollout per prompt, highlighting its superior interaction efficiency.

## Notes

(your synthesis — anything beyond the abstract belongs here)

## Source

- PDF: [[raw/papers/pdf/2026-reflection-enhanced-self-distillation.pdf]]
- Source: <https://github.com/horizon-llm/RESD>
