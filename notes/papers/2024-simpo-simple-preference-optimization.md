---
arxiv: '2405.14734'
authors:
- Yu Meng
- Mengzhou Xia
- Danqi Chen
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2405.14734
  raw: null
  source: https://arxiv.org/abs/2405.14734
owner: blaz
read: false
slug: simpo-simple-preference-optimization
tags:
- type/paper
- source/primary
- status/stub
- domain/synth-data
- stage/dpo
title: 'SimPO: Simple Preference Optimization with a Reference-Free Reward'
type: note
updated: '2026-05-10'
year: 2024
---

# SimPO: Simple Preference Optimization with a Reference-Free Reward

Reference-free preference optimization that eliminates the memory-heavy reference model.

- **Authors**: Yu Meng, Mengzhou Xia, Danqi Chen
- **Venue**: NeurIPS 2024
- **arXiv**: [2405.14734](https://arxiv.org/abs/2405.14734)
- **Raw**: [[raw/papers/pdf/2024-simpo]]

## Core contribution

Proposes a reference-free reward formulation for preference optimization that uses average log probability as an implicit reward. Eliminates the need for a separate reference model, reducing memory overhead. Includes a length normalization term that mitigates verbosity bias without separate reward modeling.

## Connections

- Related: [[notes/papers/2024-orpo-monolithic-preference-optimization]]
- Related: [[notes/papers/2024-spin-self-play-fine-tuning]]
