---
aliases:
- Kynkäänniemi 2019
- improved P/R
- k-NN precision/recall
arxiv: '1904.06991'
authors:
- Tuomas Kynkäänniemi
- Tero Karras
- Samuli Laine
- Jaakko Lehtinen
- Timo Aila
created: 2026-04-27
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1904.06991
  raw: https://arxiv.org/pdf/1904.06991
  source: https://arxiv.org/abs/1904.06991
owner: blaz
read: false
slug: improved-precision-and-recall-metric-for-assessing-generative-models
tags:
- type/paper
- source/primary
- status/stub
title: Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi
  et al. 2019)
type: note
updated: '2026-05-10'
year: 2019
---

# Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi et al. 2019)

## Citation

- arXiv: [1904.06991](https://arxiv.org/abs/1904.06991)
- Authors: Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila.
- Year / venue: NeurIPS 2019.

## Core claim (stub)

Refines the precision/recall framework of [[notes/papers/2018-precision-and-recall-for-generative-models|Sajjadi et al. 2018]] by replacing clustering / quantisation with **k-NN-based manifold estimation**. Each sample's manifold neighbourhood is the union of $k$-nearest-neighbour balls; precision is the fraction of generated samples falling inside the reference manifold, recall is the fraction of reference samples falling inside the generated manifold. More stable than the original under finite samples; standard step on the path to the Naeem 2020 density/coverage refinement.

## Why it's load-bearing

The middle link in the precision-recall lineage that ends with the modern density/coverage standard ([[notes/papers/2020-reliable-fidelity-and-diversity-metrics-for-generative-models]]). Worth tracking because some 2020–2022 work cites Kynkäänniemi P/R rather than the later refinement.

## Status

Stub.
