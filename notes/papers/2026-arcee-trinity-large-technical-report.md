---
arxiv: '2602.17004'
authors:
- Varun Singh
- Lucas Krauss
- Sami Jaghouar et al. (Arcee AI)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.17004
  raw: https://arxiv.org/pdf/2602.17004
  source: https://arxiv.org/abs/2602.17004
owner: blaz
read: false
slug: arcee-trinity-large-technical-report
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/models
- domain/training
title: Arcee Trinity Large Technical Report
type: note
updated: '2026-05-10'
year: 2026
---

# Arcee Trinity Large Technical Report

## Citation

- URL: https://arxiv.org/abs/2602.17004
- PDF: https://arxiv.org/pdf/2602.17004
- Authors: Varun Singh, Lucas Krauss, Sami Jaghouar et al. (Arcee AI)
- Year / venue: 2026-02 arXiv preprint
- arXiv: 2602.17004

## Short Summary

Sparse MoE model family: Trinity Nano (6B total / 1B active), Trinity Mini (26B total / 3B active), Trinity Large (400B total / 13B active). Architecture features: interleaved local/global attention, gated attention, depth-scaled sandwich norm, sigmoid routing for MoE. Novel load-balancing strategy: SMEBU (Soft-clamped Momentum Expert Bias Updates) for Trinity Large. Trained with Muon optimizer. Token budgets: Nano/Mini on 10T tokens, Large on 17T tokens. Zero loss spikes throughout training.

## Open Threads

- Training data composition and synthetic data usage?
- How does SMEBU compare to standard expert load balancing?
- Downstream benchmark results vs. comparable dense models?
