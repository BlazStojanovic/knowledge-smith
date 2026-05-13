---
arxiv: '2605.01640'
authors:
- Justin Lovelace
- Christian Belardi
- Srivatsa Kundurthy
- Shriya Sudhakar
- Kilian Q. Weinberger
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.01640
  raw: '[[raw/papers/md/2026-prescriptive-scaling-laws-for-data-constrained-training]]'
  source: https://arxiv.org/abs/2605.01640
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-prescriptive-scaling-laws-for-data-constrained-training.md
raw_pdf: raw/papers/pdf/2026-prescriptive-scaling-laws-for-data-constrained-training.pdf
read: false
slug: prescriptive-scaling-laws-for-data-constrained-training
tags:
- type/paper
- status/stub
title: Prescriptive Scaling Laws for Data Constrained Training
type: note
updated: '2026-05-11'
year: 2026
---

# Prescriptive Scaling Laws for Data Constrained Training

> *Justin Lovelace, Christian Belardi, Srivatsa Kundurthy, Shriya Sudhakar, Kilian Q. Weinberger* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Training compute is increasingly outpacing the availability of high-quality data. This shifts the central challenge from optimal compute allocation to extracting maximum value from limited data. The widely adopted Chinchilla scaling law assumes every training token is unique. This limits its ability to guide pretraining decisions in data-constrained regimes. We model the excess loss under repetition with a simple additive overfitting penalty and find that it accurately describes model behavior. Our scaling law yields qualitatively new compute-optimal allocation advice. Beyond a point, further repetition is counterproductive and compute is better spent on model capacity. We show that following our law's recommended configuration improves performance in data-constrained regimes. Finally, because our one-parameter form isolates overfitting in a single coefficient, it enables direct comparison across training configurations. As a case study, we show that strong weight decay ($λ=1.0$) reduces this coefficient by approximately 70%, providing a scaling-law explanation for recent findings that optimal weight decay in data-constrained regimes is an order of magnitude larger than standard practice.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.01640>
- PDF: [[raw/papers/pdf/2026-prescriptive-scaling-laws-for-data-constrained-training.pdf]]
- Raw markdown: [[raw/papers/md/2026-prescriptive-scaling-laws-for-data-constrained-training]]
