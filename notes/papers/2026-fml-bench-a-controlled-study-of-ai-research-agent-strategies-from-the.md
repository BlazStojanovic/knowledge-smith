---
arxiv: '2605.17373'
authors:
- Qiran Zou
- Hou Hei Lam
- Wenhao Zhao
- Tingting Chen
- Yiming Tang
- Samson Yu
- Yingtao Zhu
- Srinivas Anumasa
- Zufeng Zhang
- Tianyi Zhang
- Chang Liu
- Zhengyao Jiang
- Anirudh Goyal
- Dianbo Liu
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.17373
  raw: '[[raw/papers/md/2026-fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the]]'
  source: https://arxiv.org/abs/2605.17373
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the.md
raw_pdf: raw/papers/pdf/2026-fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the.pdf
read: false
slug: fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the
tags:
- type/paper
- status/stub
title: 'FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective
  of Search Dynamics'
type: note
updated: '2026-06-01'
year: 2026
---

# FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective of Search Dynamics

> *Qiran Zou, Hou Hei Lam, Wenhao Zhao, Tingting Chen, Yiming Tang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

AI research agents accelerate ML research by automating hypothesis generation, experimentation, and empirical refinement. Existing agent strategies range from greedy hill-climbing to tree search and evolutionary optimization, yet which strategy choices drive performance remains unclear. Answering this question requires a benchmark that separates agent strategy (e.g., search topology) from execution infrastructure (e.g., code editor), so that performance differences are attributable to strategy rather than infrastructure, and that provides process-level metrics beyond final scores to analyze exploration behaviors. Existing benchmarks offer limited support. We propose FML-Bench, a benchmark of 18 fundamental ML research tasks across 10 domains that separates agent strategy from execution infrastructure and defines 12 process-level behavioral metrics. Evaluating six representative agents, we find that: (1) strategy complexity alone does not guarantee strong performance: a simple greedy hill-climber nearly matches the best-performing tree-search agent, both well above the remaining agents; (2) our analysis suggests this pattern relates to improvement opportunity structure: greedy search tends to be more effective when opportunities are dense, while tree-search and evolutionary strategies tend to be more effective when opportunities are sparse; an adaptive agent built on this insight switches to broader exploration upon detecting improvement stagnation and outperforms the other six agents, lending initial support to this observation; and (3) process-level analysis reveals that early convergence and directionally focused exploration are significantly associated with final performance, while solution diversity and compute cost are not. Our benchmark is available at: https://github.com/qrzou/FML-bench.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.17373>
- PDF: [[raw/papers/pdf/2026-fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the.pdf]]
- Raw markdown: [[raw/papers/md/2026-fml-bench-a-controlled-study-of-ai-research-agent-strategies-from-the]]
