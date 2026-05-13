---
arxiv: '2510.14865'
authors:
- Emmy Liu
- Graham Neubig
- Chenyan Xiong
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2510.14865
  raw: '[[raw/papers/md/2025-midtraining-bridges-pretraining-and-posttraining-distributions]]'
  source: https://arxiv.org/abs/2510.14865
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-midtraining-bridges-pretraining-and-posttraining-distributions.md
raw_pdf: raw/papers/pdf/2025-midtraining-bridges-pretraining-and-posttraining-distributions.pdf
read: false
slug: midtraining-bridges-pretraining-and-posttraining-distributions
tags:
- type/paper
- status/stub
title: Midtraining Bridges Pretraining and Posttraining Distributions
type: note
updated: '2026-05-11'
year: 2025
---

# Midtraining Bridges Pretraining and Posttraining Distributions

> *Emmy Liu, Graham Neubig, Chenyan Xiong* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Midtraining, the practice of mixing specialized data with more general pretraining data in an intermediate training phase, has become widespread in language model development, yet there is little understanding of what makes it effective. We propose that midtraining functions as distributional bridging by providing better initialization for posttraining. We conduct controlled pretraining experiments, and find that midtraining benefits are largest for domains distant from general pretraining data, such as code and math, and scale with the proximity advantage the midtraining data provides toward the target distribution. In these domains, midtraining consistently outperforms continued pretraining on specialized data alone both in-domain and in terms of mitigating forgetting. We further conduct an investigation on the starting time and mixture weight of midtraining data, using code as a case study, and find that time of introduction and mixture weight interact strongly such that early introduction of specialized data is amenable to high mixture weights, while late introduction requires lower ones. This suggests that late introduction of specialized data outside a plasticity window cannot be compensated for by increasing data mixtures later in training. Beyond midtraining itself, this suggests that distributional transitions between any training phases may benefit from similar bridging strategies.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2510.14865>
- PDF: [[raw/papers/pdf/2025-midtraining-bridges-pretraining-and-posttraining-distributions.pdf]]
- Raw markdown: [[raw/papers/md/2025-midtraining-bridges-pretraining-and-posttraining-distributions]]
