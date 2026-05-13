---
arxiv: '2604.06268'
authors:
- Zihan Wang
- Chi Gui
- Xing Jin
- Qineng Wang
- Licheng Liu
- Kangrui Wang
- Shiqi Chen
- Linjie Li
- Zhengyuan Yang
- Pingyue Zhang
- Yiping Lu
- Jiajun Wu
- Li Fei-Fei
- Lijuan Wang
- Yejin Choi
- Manling Li
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.06268
  raw: '[[raw/papers/md/2026-ragen-2-reasoning-collapse-in-agentic-rl]]'
  source: https://arxiv.org/abs/2604.06268
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-ragen-2-reasoning-collapse-in-agentic-rl.md
raw_pdf: raw/papers/pdf/2026-ragen-2-reasoning-collapse-in-agentic-rl.pdf
read: false
slug: ragen-2-reasoning-collapse-in-agentic-rl
tags:
- type/paper
- status/stub
title: 'RAGEN-2: Reasoning Collapse in Agentic RL'
type: note
updated: '2026-05-11'
year: 2026
---

# RAGEN-2: Reasoning Collapse in Agentic RL

> *Zihan Wang, Chi Gui, Xing Jin, Qineng Wang, Licheng Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

RL training of multi-turn LLM agents is inherently unstable, and reasoning quality directly determines task performance. Entropy is widely used to track reasoning stability. However, entropy only measures diversity within the same input, and cannot tell whether reasoning actually responds to different inputs. In RAGEN-2, we find that even with stable entropy, models can rely on fixed templates that look diverse but are input-agnostic. We call this template collapse, a failure mode invisible to entropy and all existing metrics. To diagnose this failure, we decompose reasoning quality into within-input diversity (Entropy) and cross-input distinguishability (Mutual Information, MI), and introduce a family of mutual information proxies for online diagnosis. Across diverse tasks, mutual information correlates with final performance much more strongly than entropy, making it a more reliable proxy for reasoning quality. We further explain template collapse with a signal-to-noise ratio (SNR) mechanism. Low reward variance weakens task gradients, letting regularization terms dominate and erase cross-input reasoning differences. To address this, we propose SNR-Aware Filtering to select high-signal prompts per iteration using reward variance as a lightweight proxy. Across planning, math reasoning, web navigation, and code execution, the method consistently improves both input dependence and task performance.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.06268>
- PDF: [[raw/papers/pdf/2026-ragen-2-reasoning-collapse-in-agentic-rl.pdf]]
- Raw markdown: [[raw/papers/md/2026-ragen-2-reasoning-collapse-in-agentic-rl]]
