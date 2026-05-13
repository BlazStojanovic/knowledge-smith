---
arxiv: '2504.11393'
authors:
- Ian Magnusson
- Nguyen Tai
- Ben Bogin
- David Heineman
- Jena D. Hwang
- Luca Soldaini
- Akshita Bhagia
- Jiacheng Liu
- Dirk Groeneveld
- Oyvind Tafjord
- Noah A. Smith
- Pang Wei Koh
- Jesse Dodge (AI2)
created: 2026-04-28
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.11393
  raw: '[[raw/papers/md/2025-datadecide]]'
  source: https://arxiv.org/abs/2504.11393
owner: blaz
raw_pdf: raw/papers/pdf/2025-datadecide.pdf
read: false
slug: datadecide
tags:
- type/paper
- status/stub
- domain/synth-data
- domain/pretraining
- domain/evals
title: 'DataDecide: How to Predict Best Pretraining Data with Small Experiments'
type: note
updated: '2026-05-10'
year: 2025
---

# DataDecide: How to Predict Best Pretraining Data with Small Experiments

## Citation

- URL: https://arxiv.org/abs/2504.11393
- PDF: https://arxiv.org/pdf/2504.11393
- Authors: Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena D. Hwang, Luca Soldaini, Akshita Bhagia, Jiacheng Liu, Dirk Groeneveld, Oyvind Tafjord, Noah A. Smith, Pang Wei Koh, Jesse Dodge (AI2)
- Year / venue: 2025 (arXiv, April 2025)

## Core Claim

Small-scale experiments (~150M params) reliably predict which pre-training data corpus will produce the best large-scale (~1B param) model. Continuous likelihood metrics can predict data rankings using as little as **0.01% of target compute**. Validated across 25 corpora up to 100B tokens.

## Key Paper Ideas

- **Rank stability across scales.** Corpus rankings at 150M parameters strongly correlate with rankings at 1B — a confirmation of the "small proxy" intuition with broad empirical coverage (25 corpora).
- **Continuous likelihood > discrete accuracy.** Likelihood-based metrics are more reliable data-selection signals than binary benchmark accuracy at small scale.
- **DataDecide benchmark suite.** Open release: models, data, and evaluations for 25 corpora — a shared testbed for data-curation research.
- **0.01% compute threshold.** The finding that predictions stabilise at 0.01% of target compute has direct implications for how much a pre-training team needs to invest in proxy runs.

## Relevance To Poolside

Highly relevant to Poolside's data-selection and ablation work: provides empirical grounding for how small proxy runs should be sized and which metrics to track. Complements the internal xs-ablation flow — see [[technical/xs-ablation-flow]].

## Related Notes

- Concepts: [[concepts/synthetic-data-formalism]], [[concepts/data-filtering-paradigms]]
- Technical: [[technical/xs-ablation-flow]]
- Maps: [[maps/evaluation/landscape]]

## Caveats

Stub — created 2026-04-28.
