---
arxiv: '2605.15155'
authors:
- Zhengxi Lu
- Zhiyuan Yao
- Zhuowen Han
- Zi-Han Wang
- Jinyang Wu
- Qi Gu
- Xunliang Cai
- Weiming Lu
- Jun Xiao
- Yueting Zhuang
- Yongliang Shen
created: '2026-05-18'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.15155
  raw: '[[raw/papers/md/2026-self-distilled-agentic-reinforcement-learning]]'
  source: https://arxiv.org/abs/2605.15155
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-self-distilled-agentic-reinforcement-learning.md
raw_pdf: raw/papers/pdf/2026-self-distilled-agentic-reinforcement-learning.pdf
read: false
slug: self-distilled-agentic-reinforcement-learning
tags:
- type/paper
- status/stub
- reinforcement-learning
- agents
- distillation
- llm
title: Self-Distilled Agentic Reinforcement Learning
type: note
updated: '2026-05-18'
year: 2026
---

# Self-Distilled Agentic Reinforcement Learning

> *Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, et al.* — arXiv 2026

## TL;DR

SDAR adds On-Policy Self-Distillation (OPSD) — dense token-level guidance from a teacher branch given privileged context — as a **gated auxiliary** objective on top of GRPO, which remains the primary optimizer, for multi-turn agents. The problem it fixes: naive GRPO+OPSD destabilizes in multi-turn settings, and negative teacher rejections are ambiguous (they may come from imperfect skill retrieval, not bad actions). SDAR routes detached token-level signals through a sigmoid gate that **strengthens** distillation on teacher-endorsed positive-gap tokens and **softly attenuates** negative teacher rejections. On Qwen2.5/Qwen3 across ALFWorld, WebShop, and Search-QA it beats GRPO (+9.4% ALFWorld, +7.0% Search-QA, +10.2% WebShop-Acc) without the instability of the naive hybrid. (Summary from abstract; note unread.)

## Abstract

Reinforcement learning (RL) has emerged as a central paradigm for post-training LLM agents, yet its trajectory-level reward signal provides only coarse supervision for long-horizon interaction. On-Policy Self-Distillation (OPSD) complements RL by introducing dense token-level guidance from a teacher branch augmented with privileged context. However, transferring OPSD to multi-turn agents proves problematic: compounding multi-turn instability destabilizes supervision, while skill-conditioned privileged guidance requires asymmetric treatment for negative teacher rejections may arise from imperfect skills retrieval or utilization. We introduce SDAR (Self-Distilled Agentic Reinforcement Learning), which treats OPSD as a gated auxiliary objective while keeping RL as the primary optimization backbone. SDAR maps detached token-level signals into a sigmoid gate, strengthening distillation on teacher-endorsed positive-gap tokens and softly attenuating negative teacher rejections. Across the Qwen2.5 and Qwen3 families on ALFWorld, WebShop, and Search-QA, SDAR substantially improves over GRPO (+9.4% on ALFWorld, +7.0% on Search-QA, +10.2% on WebShop-Acc), avoids the instability of naive GRPO+OPSD, and consistently outperforms hybrid RL--OPSD baselines across model scales.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.15155>
- PDF: [[raw/papers/pdf/2026-self-distilled-agentic-reinforcement-learning.pdf]]
- Raw markdown: [[raw/papers/md/2026-self-distilled-agentic-reinforcement-learning]]
