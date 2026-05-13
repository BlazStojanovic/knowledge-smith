---
arxiv: '2604.19295'
authors:
- Qingyang Zhang
- Xinke Kong
- Haitao Wu
- Qinghua Hu
- Minghao Wu
- Baosong Yang
- Yu Cheng
- Yun Luo
- Ganqu Cui
- Changqing Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.19295
  raw: '[[raw/papers/md/2026-tempo-scaling-test-time-training-for-large-reasoning-models]]'
  source: https://arxiv.org/abs/2604.19295
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-tempo-scaling-test-time-training-for-large-reasoning-models.md
raw_pdf: raw/papers/pdf/2026-tempo-scaling-test-time-training-for-large-reasoning-models.pdf
read: false
slug: tempo-scaling-test-time-training-for-large-reasoning-models
tags:
- type/paper
- status/stub
title: 'TEMPO: Scaling Test-time Training for Large Reasoning Models'
type: note
updated: '2026-05-11'
year: 2026
---

# TEMPO: Scaling Test-time Training for Large Reasoning Models

> *Qingyang Zhang, Xinke Kong, Haitao Wu, Qinghua Hu, Minghao Wu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Test-time training (TTT) adapts model parameters on unlabeled test instances during inference time, which continuously extends capabilities beyond the reach of offline training. Despite initial gains, existing TTT methods for LRMs plateau quickly and do not benefit from additional test-time compute. Without external calibration, the self-generated reward signal increasingly drifts as the policy model evolves, leading to both performance plateaus and diversity collapse. We propose TEMPO, a TTT framework that interleaves policy refinement on unlabeled questions with periodic critic recalibration on a labeled dataset. By formalizing this alternating procedure through the Expectation-Maximization (EM) algorithm, we reveal that prior methods can be interpreted as incomplete variants that omit the crucial recalibration step. Reintroducing this step tightens the evidence lower bound (ELBO) and enables sustained improvement. Across diverse model families (Qwen3 and OLMO3) and reasoning tasks, TEMPO improves OLMO3-7B on AIME 2024 from 33.0% to 51.1% and Qwen3-14B from 42.3% to 65.8%, while maintaining high diversity.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.19295>
- PDF: [[raw/papers/pdf/2026-tempo-scaling-test-time-training-for-large-reasoning-models.pdf]]
- Raw markdown: [[raw/papers/md/2026-tempo-scaling-test-time-training-for-large-reasoning-models]]
