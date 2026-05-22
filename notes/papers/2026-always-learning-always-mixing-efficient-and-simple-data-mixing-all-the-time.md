---
arxiv: '2605.15220'
authors:
- Michael Y. Hu
- Apurva Gandhi
- Kyunghyun Cho
- Tal Linzen
- Pratyusha Sharma
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.15220
  raw: '[[raw/papers/md/2026-always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time]]'
  source: https://arxiv.org/abs/2605.15220
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time.md
raw_pdf: raw/papers/pdf/2026-always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time.pdf
read: false
slug: always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time
tags:
- type/paper
- status/stub
title: 'Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time'
type: note
updated: '2026-05-22'
year: 2026
---

# Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time

> *Michael Y. Hu, Apurva Gandhi, Kyunghyun Cho, Tal Linzen, Pratyusha Sharma* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Data mixing decides how to combine different sources or types of data and is a consequential problem throughout language model training. In pretraining, data composition is a key determinant of model quality; in continual learning and adaptation, it governs what is retained and acquired. Yet existing data mixing methods address only one phase of this lifecycle at a time: some require smaller proxy models tied to a single training phase, others assume a fixed domain set, and continual learning lacks principled guidance altogether. We argue that data mixing is fundamentally an online decision making problem -- one that recurs throughout training and demands a single, unified solution. We introduce OP-Mix (On-Policy Mix), a data mixing algorithm that operates across the entire language model training lifecycle. Our main insight is that candidate data mixtures can be cheaply simulated by interpolating between low-rank adapters trained directly on the current model, eliminating separate proxy models and ensuring the search is always grounded in the model's actual learning dynamics. Across pretraining, continual midtraining, and continual instruction tuning, OP-Mix consistently finds near-optimal mixtures while using a fraction of the compute of the baselines. In pretraining, OP-Mix improves upon training without mixing by 6.3% in average perplexity. For continual learning, OP-Mix matches the performance of both retraining and on-policy distillation while using 66% and 95% less overall compute, respectively. OP-Mix suggests a different view of language model training: not a sequence of distinct phases, but a single continuous process of learning from data.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.15220>
- PDF: [[raw/papers/pdf/2026-always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time.pdf]]
- Raw markdown: [[raw/papers/md/2026-always-learning-always-mixing-efficient-and-simple-data-mixing-all-the-time]]
