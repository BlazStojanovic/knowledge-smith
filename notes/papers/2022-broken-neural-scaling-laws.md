---
arxiv: '2210.14891'
authors:
- Ethan Caballero
- Kshitij Gupta
- Irina Rish
- David Krueger
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2210.14891
  raw: '[[raw/papers/md/2022-broken-neural-scaling-laws]]'
  source: https://arxiv.org/abs/2210.14891
owner: blaz
raw_pdf: raw/papers/pdf/2022-broken-neural-scaling-laws.pdf
read: false
slug: broken-neural-scaling-laws
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/evals
title: Broken Neural Scaling Laws
type: note
updated: '2026-05-10'
year: 2022
---

# Broken Neural Scaling Laws

## Citation

- URL: https://arxiv.org/abs/2210.14891
- Authors: Ethan Caballero, Kshitij Gupta, Irina Rish, David Krueger
- Affiliation: Mila, Université de Montréal
- Year / venue: 2022 / ICML 2023 Workshop on Computational Sustainability, later ICLR 2024 spotlight
- arXiv: 2210.14891
- **Raw**: [[raw/papers/pdf/2022-broken-neural-scaling-laws.pdf]]

## Core Claim

Standard power law forms L(x) = ax^b + c fail to capture non-linearities, inflection points, and double descent in downstream task scaling. The BNSL (Broken Neural Scaling Law) functional form captures these phenomena:

L(D) = E + (b·D^{-c₀}) ∏(1 + (D/dᵢ)^{1/fᵢ})^{-cᵢfᵢ}

BNSL uses a smoothly-connected piecewise power law that can model multiple "breaks" — transitions where the scaling exponent changes.

## Key Ideas

- Power laws work in the scaling regime but fail at transitions (random guessing → power law, power law → saturation, intermediate breaks)
- BNSL predicts downstream performance more accurately than power laws, especially for tasks with phase transitions
- Relates to the emergence debate: apparent "emergent" abilities may be smooth BNSL curves that look like phase transitions when measured with discontinuous metrics
- Alternative to M4 estimator proposed by Alabdulmohsin et al. (2022)

## Relevance To Poolside

When Poolside's eval scores show non-smooth scaling behavior, BNSL may provide better extrapolation than standard power law fits. Especially relevant for tasks that show apparent emergence (AIME, GPQA) where standard scaling predictions fail.

## Related Notes

- [[concepts/scaling-laws-foundational]] — BNSL as an alternative to standard power law forms
- [[concepts/evaluation-scaling-laws]] — emergence vs metric artifact debate
- [[maps/scaling-laws/landscape]] — pathologies domain
- [[raw/articles/2023-epoch-ai-scaling-laws-literature-review]] — covers BNSL alongside M4
