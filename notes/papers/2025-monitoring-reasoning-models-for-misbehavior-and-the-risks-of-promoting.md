---
arxiv: '2503.11926'
authors:
- Bowen Baker
- Joost Huizinga
- Leo Gao
- Zehao Dou
- Melody Y. Guan
- Aleksander Madry
- Wojciech Zaremba
- Jakub Pachocki
- David Farhi
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2503.11926
  raw: '[[raw/papers/md/2025-monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting]]'
  source: https://arxiv.org/abs/2503.11926
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting.md
raw_pdf: raw/papers/pdf/2025-monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting.pdf
read: false
slug: monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting
tags:
- type/paper
- status/stub
title: Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation
type: note
updated: '2026-05-11'
year: 2025
---

# Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation

> *Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Mitigating reward hacking--where AI systems misbehave due to flaws or misspecifications in their learning objectives--remains a key challenge in constructing capable and aligned models. We show that we can monitor a frontier reasoning model, such as OpenAI o3-mini, for reward hacking in agentic coding environments by using another LLM that observes the model's chain-of-thought (CoT) reasoning. CoT monitoring can be far more effective than monitoring agent actions and outputs alone, and we further found that a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model. Because CoT monitors can be effective at detecting exploits, it is natural to ask whether those exploits can be suppressed by incorporating a CoT monitor directly into the agent's training objective. While we show that integrating CoT monitors into the reinforcement learning reward can indeed produce more capable and more aligned agents in the low optimization regime, we find that with too much optimization, agents learn obfuscated reward hacking, hiding their intent within the CoT while still exhibiting a significant rate of reward hacking. Because it is difficult to tell when CoTs have become obfuscated, it may be necessary to pay a monitorability tax by not applying strong optimization pressures directly to the chain-of-thought, ensuring that CoTs remain monitorable and useful for detecting misaligned behavior.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2503.11926>
- PDF: [[raw/papers/pdf/2025-monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting.pdf]]
- Raw markdown: [[raw/papers/md/2025-monitoring-reasoning-models-for-misbehavior-and-the-risks-of-promoting]]
