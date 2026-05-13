---
arxiv: '2604.01411'
authors:
- Nicholas Roberts
- Sungjun Cho
- Zhiqi Gao
- Tzu-Heng Huang
- Albert Wu
- Gabriel Orlanski
- Avi Trost
- Kelly Buchanan
- Aws Albarghouthi
- Frederic Sala
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.01411
  raw: '[[raw/papers/md/2026-test-time-scaling-makes-overtraining-compute-optimal]]'
  source: https://arxiv.org/abs/2604.01411
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-test-time-scaling-makes-overtraining-compute-optimal.md
raw_pdf: raw/papers/pdf/2026-test-time-scaling-makes-overtraining-compute-optimal.pdf
read: false
slug: test-time-scaling-makes-overtraining-compute-optimal
tags:
- type/paper
- status/stub
title: Test-Time Scaling Makes Overtraining Compute-Optimal
type: note
updated: '2026-05-11'
year: 2026
---

# Test-Time Scaling Makes Overtraining Compute-Optimal

> *Nicholas Roberts, Sungjun Cho, Zhiqi Gao, Tzu-Heng Huang, Albert Wu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Modern LLMs scale at test-time, e.g. via repeated sampling, where inference cost grows with model size and the number of samples. This creates a trade-off that pretraining scaling laws, such as Chinchilla, do not address. We present Train-to-Test ($T^2$) scaling laws that jointly optimize model size, training tokens, and number of inference samples under fixed end-to-end budgets. $T^2$ modernizes pretraining scaling laws with pass@$k$ modeling used for test-time scaling, then jointly optimizes pretraining and test-time decisions. Forecasts from $T^2$ are robust over distinct modeling approaches: measuring joint scaling effect on the task loss and modeling impact on task accuracy. Across eight downstream tasks, we find that when accounting for inference cost, optimal pretraining decisions shift radically into the overtraining regime, well-outside of the range of standard pretraining scaling suites. We validate our results by pretraining heavily overtrained models in the optimal region that $T^2$ scaling forecasts, confirming their substantially stronger performance compared to pretraining scaling alone. Finally, as frontier LLMs are post-trained, we show that our findings survive the post-training stage, making $T^2$ scaling meaningful in modern deployments.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.01411>
- PDF: [[raw/papers/pdf/2026-test-time-scaling-makes-overtraining-compute-optimal.pdf]]
- Raw markdown: [[raw/papers/md/2026-test-time-scaling-makes-overtraining-compute-optimal]]
