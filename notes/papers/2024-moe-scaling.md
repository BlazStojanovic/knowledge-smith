---
arxiv: '2402.07871'
authors:
- Jakub Ludziejewski
- Maciej Piróg
- Kamil Ciebiera
- Krystian Król
- Jan Ludziejewski
- Marek Cygan
- Sebastian Jaszczur
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2402.07871
  raw: '[[raw/papers/md/2024-moe-scaling]]'
  source: https://arxiv.org/abs/2402.07871
owner: blaz
raw_pdf: raw/papers/pdf/2024-moe-scaling.pdf
read: false
slug: moe-scaling
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/training
title: Scaling Laws for Fine-Grained Mixture of Experts
type: note
updated: '2026-05-10'
year: 2024
---

# Scaling Laws for Fine-Grained Mixture of Experts

## Citation

- URL: https://arxiv.org/abs/2402.07871
- Authors: Jakub Ludziejewski, Maciej Piróg, Kamil Ciebiera, Krystian Król, Jan Ludziejewski, Marek Cygan, Sebastian Jaszczur
- Affiliation: University of Warsaw, IDEAS NCBR
- Year / venue: 2024 / arXiv
- arXiv: 2402.07871
- **Raw**: [[raw/papers/pdf/2024-moe-scaling.pdf]]

## Core Claim

Granularity (the ratio of total experts to expert size) is a first-class scaling hyperparameter for MoE models. Scaling laws that account for N_total, N_active, D, and granularity predict MoE performance more accurately than laws that treat N_total alone.

## Key Ideas

- Fine-grained experts (many small) can outperform coarse-grained (fewer large) at the same compute, but with higher routing overhead
- The optimal number of experts scales with compute budget
- MoE efficiency leverage: the ratio N_total / N_active determines the advantage over dense models; follows a power law in compute
- MoE models can be more memory-efficient than equivalent-compute dense models when properly scaled

## Related Notes

- [[concepts/architecture-scaling]] — MoE as an architecture component
- [[maps/scaling-laws/landscape]] — architecture components domain
