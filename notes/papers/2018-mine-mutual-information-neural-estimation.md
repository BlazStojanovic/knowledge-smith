---
aliases:
- MINE
- Belghazi 2018
arxiv: '1801.04062'
authors:
- Mohamed Ishmael Belghazi
- Aristide Baratin
- Sai Rajeswar
- Sherjil Ozair
- Yoshua Bengio
- Aaron Courville
- R Devon Hjelm
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1801.04062
  raw: https://arxiv.org/pdf/1801.04062
  source: https://arxiv.org/abs/1801.04062
owner: blaz
read: false
slug: mine-mutual-information-neural-estimation
tags:
- type/paper
- source/primary
- status/stub
title: 'MINE: Mutual Information Neural Estimation (Belghazi et al. 2018)'
type: note
updated: '2026-05-10'
year: 2018
---

# MINE: Mutual Information Neural Estimation (Belghazi et al. 2018)

## Citation

- arXiv: [1801.04062](https://arxiv.org/abs/1801.04062)
- Authors: Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, R Devon Hjelm.
- Year / venue: ICML 2018.

## Core claim (stub)

A scalable neural estimator for mutual information $I(X; Y)$ between high-dimensional random variables, based on the Donsker–Varadhan dual representation of KL divergence. Trains a neural network to recognise samples from the joint distribution vs samples from the product of marginals; the trained network's score is a lower bound on MI and can be optimised end-to-end.

## Why it's load-bearing

For the corpus-scale provenance frontier ($I(O; S)$ at billion-document scale, the open question in [[maps/evaluation/distributional-level]] §Axis 3), MINE-style estimators are the realistic path. The alternative — InfoNCE / contrastive estimation — is also tracked separately (see [[notes/papers/2018-representation-learning-with-contrastive-predictive-coding]]). MINE has known instability issues at high MI; InfoNCE is the more robust choice in practice but the two should be benchmarked together.

## Status

Stub. Full read deferred until corpus-scale MI estimation becomes an active engineering target.
