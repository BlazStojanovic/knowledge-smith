---
arxiv: '2410.05192'
authors:
- '[needs verification]'
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2410.05192
  raw: null
  source: https://arxiv.org/abs/2410.05192
owner: blaz
read: false
slug: wsd-learning-rates
tags:
- type/paper
- status/stub
- source/paper
- domain/training
title: 'Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape
  Perspective'
type: note
updated: '2026-05-10'
year: 2024
---

# Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective

## Citation

- URL: https://arxiv.org/abs/2410.05192
- Year / venue: 2024 / arXiv
- arXiv: 2410.05192

## Core Claim

The WSD (warmup-stable-decay) learning rate schedule outperforms cosine decay at scale. The theoretical basis is a "river valley" loss landscape: high LR during the stable phase enables fast progress along the valley while oscillating perpendicular to it; decay collapses the oscillations to reveal the true optimum.

## Related Notes

- [[concepts/hyperparameter-scaling]] — WSD schedule
- [[maps/scaling-laws/landscape]] — hyperparameters domain
