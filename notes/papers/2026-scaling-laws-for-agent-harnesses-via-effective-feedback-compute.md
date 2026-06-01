---
arxiv: '2605.29682'
authors:
- Xuanliang Zhang
- Dingzirui Wang
- Keyan Xu
- Qingfu Zhu
- Wanxiang Che
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.29682
  raw: '[[raw/papers/md/2026-scaling-laws-for-agent-harnesses-via-effective-feedback-compute]]'
  source: https://arxiv.org/abs/2605.29682
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-scaling-laws-for-agent-harnesses-via-effective-feedback-compute.md
raw_pdf: raw/papers/pdf/2026-scaling-laws-for-agent-harnesses-via-effective-feedback-compute.pdf
read: false
slug: scaling-laws-for-agent-harnesses-via-effective-feedback-compute
tags:
- type/paper
- status/stub
title: Scaling Laws for Agent Harnesses via Effective Feedback Compute
type: note
updated: '2026-06-01'
year: 2026
---

# Scaling Laws for Agent Harnesses via Effective Feedback Compute

> *Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Agent harnesses increasingly determine the performance of language-model systems by deciding how models call tools, receive feedback, verify intermediate states, store memory, and revise solutions. Yet current test-time scaling analyses often parameterize this process by raw expenditure -- tokens, tool calls, operations, wall time, or cost -- which does not distinguish useful feedback from redundant or unstable interaction. We introduce \emph{Effective Feedback Compute} (EFC), a trace-level scaling coordinate that credits feedback only when it is informative, valid, non-redundant, and retained for subsequent decisions, and we normalize it by task demand when comparing tasks with different feedback requirements. Across synthetic controllable tasks, executable code tasks, real benchmark traces, held-out splits, and a prospective validation batch, EFC-based coordinates consistently predict failure rates better than raw-compute baselines and a strong multivariate SAS baseline. In controlled scaling, raw tokens and tool calls explain limited variation ($R^2=0.33$ and $0.42$), SAS reaches $0.88$, while Oracle-EFC and Estimated-EFC reach $0.94$ and Oracle-EFC/$D_{\mathrm{task}}$ reaches $0.99$. Matched-budget interventions show that improving feedback quality raises success from $0.27$ to $0.90$ while raw cost and tool calls are fixed. On mixed real traces, NRS-EFC/$D_{\mathrm{task}}$ reaches $R^2=0.92$ while raw compute has near-zero or negative fit, and it remains the best predictor in a prospective holdout ($R^2=0.85$). These results suggest that harness scaling is governed less by how much computation is spent than by how efficiently raw budget is converted into durable, task-sufficient feedback.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.29682>
- PDF: [[raw/papers/pdf/2026-scaling-laws-for-agent-harnesses-via-effective-feedback-compute.pdf]]
- Raw markdown: [[raw/papers/md/2026-scaling-laws-for-agent-harnesses-via-effective-feedback-compute]]
