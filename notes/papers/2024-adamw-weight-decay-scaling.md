---
arxiv: '2405.13698'
authors:
- '[needs verification]'
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2405.13698
  raw: null
  source: https://arxiv.org/abs/2405.13698
owner: blaz
read: false
slug: adamw-weight-decay-scaling
tags:
- type/paper
- status/stub
- source/paper
- domain/training
title: How to Set AdamW's Weight Decay as You Scale Model and Dataset Size
type: note
updated: '2026-05-10'
year: 2024
---

# How to Set AdamW's Weight Decay as You Scale Model and Dataset Size

## Citation

- URL: https://arxiv.org/abs/2405.13698
- Year / venue: 2024 / ICML 2025
- arXiv: 2405.13698

## Core Claim

AdamW weight decay should be understood through the EMA timescale τ = B/(ηλD). The optimal τ (in epochs) is roughly constant across model and dataset sizes, providing a principled scaling rule: as D increases, λ should decrease; as N increases, λ should increase.

## Related Notes

- [[concepts/hyperparameter-scaling]] — weight decay scaling
- [[maps/scaling-laws/landscape]] — hyperparameters domain
