---
arxiv: '2403.07691'
authors:
- Jiwoo Hong
- Noah Lee
- James Thorne
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2403.07691
  raw: null
  source: https://arxiv.org/abs/2403.07691
owner: blaz
read: false
slug: orpo-monolithic-preference-optimization
tags:
- type/paper
- source/primary
- status/stub
- domain/synth-data
- stage/dpo
title: 'ORPO: Monolithic Preference Optimization without Reference Model'
type: note
updated: '2026-05-10'
year: 2024
---

# ORPO: Monolithic Preference Optimization without Reference Model

Odds ratio-based preference optimization that integrates instruction following directly into the alignment loss.

- **Authors**: Jiwoo Hong, Noah Lee, James Thorne
- **Venue**: EMNLP 2024
- **arXiv**: [2403.07691](https://arxiv.org/abs/2403.07691)
- **Raw**: [[raw/papers/pdf/2024-orpo]]

## Core contribution

Introduces Odds Ratio Preference Optimization: a reference-free alignment objective that combines SFT and preference alignment into a single training stage. Uses an odds ratio term to penalize disfavored responses, eliminating the two-stage RLHF pipeline. Helps mitigate length bias without separate reward modeling.

## Connections

- Related: [[notes/papers/2024-simpo-simple-preference-optimization]]
