---
arxiv: '2406.09405'
authors:
- '[needs verification]'
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2406.09405
  raw: null
  source: https://arxiv.org/abs/2406.09405
owner: blaz
read: false
slug: why-warmup-learning-rate
tags:
- type/paper
- status/stub
- source/paper
- domain/training
title: Why Warmup the Learning Rate? Underlying Mechanisms and Improvements
type: note
updated: '2026-05-10'
year: 2024
---

# Why Warmup the Learning Rate? Underlying Mechanisms and Improvements

## Citation

- URL: https://arxiv.org/abs/2406.09405
- Year / venue: 2024 / NeurIPS 2024
- arXiv: 2406.09405

## Core Claim

LR warmup's primary benefit is enabling larger peak learning rates by traversing high-curvature regions near random initialization safely. The optimal warmup duration is not about the number of steps per se, but about reaching the peak LR while Adam statistics stabilize.

## Related Notes

- [[concepts/hyperparameter-scaling]] — warmup theory
- [[maps/scaling-laws/landscape]] — hyperparameters domain
