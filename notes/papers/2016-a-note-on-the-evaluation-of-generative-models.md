---
aliases:
- Theis 2016
- Note on evaluation of generative models
arxiv: '1511.01844'
authors:
- Lucas Theis
- Aäron van den Oord
- Matthias Bethge
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1511.01844
  raw: https://arxiv.org/pdf/1511.01844
  source: https://arxiv.org/abs/1511.01844
owner: blaz
read: false
slug: a-note-on-the-evaluation-of-generative-models
tags:
- type/paper
- source/primary
- status/stub
title: A Note on the Evaluation of Generative Models (Theis et al. 2016)
type: note
updated: '2026-05-10'
year: 2016
---

# A Note on the Evaluation of Generative Models (Theis et al. 2016)

## Citation

- arXiv: [1511.01844](https://arxiv.org/abs/1511.01844)
- Authors: Lucas Theis, Aäron van den Oord, Matthias Bethge.
- Year / venue: ICLR 2016.

## Core claim (stub)

Argues that average log-likelihood, sample quality, and the ability of a model to perform well on downstream tasks are *largely independent* of one another for generative models. A model with good likelihood can produce poor samples and vice versa; ranking generators by any single metric is misleading. This motivates the multi-metric framing that the rest of the field eventually settled into (and that [[maps/evaluation/distributional-level]] explicitly enumerates).

## Why it's load-bearing

Conceptual ancestor of "fidelity vs diversity is not one number". Anyone who tries to defend a single-scalar evaluation is making a claim Theis 2016 already refuted at the conceptual level.

## Status

Stub. Full read deferred.
