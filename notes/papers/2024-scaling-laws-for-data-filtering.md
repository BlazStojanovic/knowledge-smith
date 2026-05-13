---
arxiv: '2404.07177'
authors:
- Sachin Goyal
- Pratyush Maini
- Zachary C. Lipton
- Aditi Raghunathan
- J. Zico Kolter
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.07177
  raw: '[[raw/papers/md/2024-scaling-laws-for-data-filtering]]'
  source: https://arxiv.org/abs/2404.07177
owner: blaz
raw_pdf: raw/papers/pdf/2024-scaling-laws-for-data-filtering.pdf
read: false
slug: scaling-laws-for-data-filtering
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/data-mix
title: Scaling Laws for Data Filtering — Data Curation Cannot be Compute Agnostic
type: note
updated: '2026-05-10'
year: 2024
---

# Scaling Laws for Data Filtering — Data Curation Cannot be Compute Agnostic

## Citation

- URL: https://arxiv.org/abs/2404.07177
- Authors: Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, J. Zico Kolter
- Year / venue: CVPR 2024 (pp. 22702–22711); arXiv Apr 2024
- arXiv: 2404.07177
- **Raw**: [[raw/papers/pdf/2024-scaling-laws-for-data-filtering.pdf]]

## Core Claim

Data curation decisions must account for total training compute budget. High-quality data loses its utility advantage when repeated; at large compute budgets, including lower-quality data (once) can outperform repeating high-quality data. The optimal quality threshold depends on C.

## Key Ideas

- Characterizes utility differences across data quality subsets as a function of compute
- At low compute: aggressive filtering (high quality only) wins
- At high compute: filtering too aggressively leaves insufficient unique data → repetition penalty exceeds quality penalty
- The optimal filter threshold is a decreasing function of compute budget
- Connects to [[concepts/data-repetition]]: the cost of repetition scales differently from the cost of low-quality data

## Relevance To Poolside

Directly relevant to Poolside's data curation pipeline. The optimal quality filter threshold for a Laguna-scale run may differ from the threshold validated at smaller scale.

## Related Notes

- [[concepts/data-repetition]] — repetition vs quality tradeoff
- [[maps/scaling-laws/landscape]] — data quality domain
- [[maps/evaluation/recipe-level]] — scaling-law-based recipe comparison
- [[questions/evaluation]] — intrinsic vs utility meta-question
