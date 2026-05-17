---
arxiv: '2605.12715'
authors:
- Anastasiia Sedova
- Skyler Seto
- Natalie Schluter
- Pierre Ablin
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.12715
  raw: '[[raw/papers/md/2026-scaling-laws-for-mixture-pretraining-under-data-constraints]]'
  source: https://arxiv.org/abs/2605.12715
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-scaling-laws-for-mixture-pretraining-under-data-constraints.md
raw_pdf: raw/papers/pdf/2026-scaling-laws-for-mixture-pretraining-under-data-constraints.pdf
read: false
slug: scaling-laws-for-mixture-pretraining-under-data-constraints
tags:
- type/paper
- status/stub
- scaling-laws
- pretraining
- llm
- theory
title: Scaling Laws for Mixture Pretraining Under Data Constraints
type: note
updated: '2026-05-15'
year: 2026
---

# Scaling Laws for Mixture Pretraining Under Data Constraints

> *Anastasiia Sedova, Skyler Seto, Natalie Schluter, Pierre Ablin* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

As language models scale, the amount of data they require grows -- yet many target data sources, such as low-resource languages or specialized domains, are inherently limited in size. A common strategy is to mix this scarce but valuable target data with abundant generic data, which presents a fundamental trade-off: too little target data in the mixture underexposes the model to the target domain, while too much target data repeats the same examples excessively, yielding diminishing returns and eventual overfitting. We study this trade-off across more than 2,000 language-model training runs spanning multiple model and target dataset sizes, as well as several data types, including multilingual, domain-specific, and quality-filtered mixtures. Across all settings, we find that repetition is a central driver of target-domain performance, and that mixture training tolerates much higher repetition than single-source training: scarce target corpora can be reused 15-20 times, with the optimal number of repetitions depending on the target data size, compute budget, and model scale. Next, we introduce a repetition-aware mixture scaling law that accounts for the decreasing value of repeated target tokens and the regularizing role of generic data. Optimizing the scaling law provides a principled way to compute effective mixture configurations, yielding practical mixture recommendations for pretraining under data constraints.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.12715>
- PDF: [[raw/papers/pdf/2026-scaling-laws-for-mixture-pretraining-under-data-constraints.pdf]]
- Raw markdown: [[raw/papers/md/2026-scaling-laws-for-mixture-pretraining-under-data-constraints]]
