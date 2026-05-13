---
aliases:
- CPC
- InfoNCE
- van den Oord 2018
arxiv: '1807.03748'
authors:
- Aäron van den Oord
- Yazhe Li
- Oriol Vinyals
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1807.03748
  raw: https://arxiv.org/pdf/1807.03748
  source: https://arxiv.org/abs/1807.03748
owner: blaz
read: false
slug: representation-learning-with-contrastive-predictive-coding
tags:
- type/paper
- source/primary
- status/stub
title: Representation Learning with Contrastive Predictive Coding (van den Oord et
  al. 2018)
type: note
updated: '2026-05-10'
year: 2018
---

# Representation Learning with Contrastive Predictive Coding (van den Oord et al. 2018)

## Citation

- arXiv: [1807.03748](https://arxiv.org/abs/1807.03748)
- Authors: Aäron van den Oord, Yazhe Li, Oriol Vinyals.
- Year / venue: arXiv 2018.

## Core claim (stub)

Introduces **InfoNCE**, a contrastive loss whose minimum is a lower bound on mutual information. The model learns to discriminate a positive pair drawn from the joint distribution against $K-1$ negative pairs drawn from the marginals. As $K$ increases the bound becomes tighter, with $\log K$ as the asymptotic ceiling.

## Why it's load-bearing

InfoNCE is the more practically stable cousin of MINE for mutual-information estimation at scale. For the [[maps/evaluation/distributional-level]] §Axis 3 corpus-scale provenance frontier — estimating $I(O; S)$ between seed and output corpora — InfoNCE-style contrastive estimators are the leading candidate, especially when paired with a learned embedding.

## Status

Stub. Full read deferred. See companion stub [[notes/papers/2018-mine-mutual-information-neural-estimation]].
