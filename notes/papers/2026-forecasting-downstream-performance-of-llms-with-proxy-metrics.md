---
arxiv: '2605.18607'
authors:
- Arkil Patel
- Siva Reddy
- Marius Mosbach
- Dzmitry Bahdanau
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.18607
  raw: '[[raw/papers/md/2026-forecasting-downstream-performance-of-llms-with-proxy-metrics]]'
  source: https://arxiv.org/abs/2605.18607
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-forecasting-downstream-performance-of-llms-with-proxy-metrics.md
raw_pdf: raw/papers/pdf/2026-forecasting-downstream-performance-of-llms-with-proxy-metrics.pdf
read: false
slug: forecasting-downstream-performance-of-llms-with-proxy-metrics
tags:
- type/paper
- status/stub
title: Forecasting Downstream Performance of LLMs With Proxy Metrics
type: note
updated: '2026-05-22'
year: 2026
---

# Forecasting Downstream Performance of LLMs With Proxy Metrics

> *Arkil Patel, Siva Reddy, Marius Mosbach, Dzmitry Bahdanau* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Progress in language model development is often driven by comparative decisions: which architecture to adopt, which pretraining corpus to use, or which training recipe to apply. Making these decisions well requires reliable performance forecasts, yet the two commonly used signals are fundamentally limited. Cross-entropy loss is poorly aligned with downstream capabilities, and direct downstream evaluation is expensive, sparse, and often uninformative at early training stages. Instead, we propose to construct proxy metrics by aggregating token-level statistics, such as entropy, top-k accuracy, and expert token rank, from a candidate model's next token distribution over expert-written solutions. Across three settings, our proxies consistently outperform loss- and compute-based baselines: 1) For cross-family model selection, they rank a heterogeneous population of reasoning models with mean Spearman Rho = 0.81 (vs. Rho = 0.36 for cross-entropy loss); 2) For pretraining data selection, they reliably rank 25 candidate corpora for a target model at roughly $10{,}000\times$ less compute than direct evaluation, pushing the Pareto frontier beyond existing methods; and 3) for training-time forecasting, they extrapolate downstream accuracy across an $18\times$ compute horizon with roughly half the error of existing alternatives. Together, these results suggest that expert trajectories are a broadly useful source of signal for assessing model capabilities, enabling reliable performance forecasting throughout the model development life cycle.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.18607>
- PDF: [[raw/papers/pdf/2026-forecasting-downstream-performance-of-llms-with-proxy-metrics.pdf]]
- Raw markdown: [[raw/papers/md/2026-forecasting-downstream-performance-of-llms-with-proxy-metrics]]
