---
arxiv: '2507.12466'
authors:
- David Mizrahi
- Anders Boesen Lindbo Larsen
- Jesse Allardice
- Suzie Petryk
- Yuri Gorokhov
- Jeffrey Li
- Alex Fang
- Josh Gardner
- Tom Gunter
- Afshin Dehghan
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2507.12466
  raw: '[[raw/papers/md/2025-language-models-improve-when-pretraining-data-matches-target-tasks]]'
  source: https://arxiv.org/abs/2507.12466
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-language-models-improve-when-pretraining-data-matches-target-tasks.md
raw_pdf: raw/papers/pdf/2025-language-models-improve-when-pretraining-data-matches-target-tasks.pdf
read: false
slug: language-models-improve-when-pretraining-data-matches-target-tasks
tags:
- type/paper
- status/stub
title: Language Models Improve When Pretraining Data Matches Target Tasks
type: note
updated: '2026-05-11'
year: 2025
---

# Language Models Improve When Pretraining Data Matches Target Tasks

> *David Mizrahi, Anders Boesen Lindbo Larsen, Jesse Allardice, Suzie Petryk, Yuri Gorokhov, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Every data selection method inherently has a target. In practice, these targets often emerge implicitly through benchmark-driven iteration: researchers develop selection strategies, train models, measure benchmark performance, then refine accordingly. This raises a natural question: what happens when we make this optimization explicit? To explore this, we propose benchmark-targeted ranking (BETR), a simple method that selects pretraining documents based on similarity to benchmark training examples. BETR embeds benchmark examples and a sample of pretraining documents in a shared space, scores this sample by similarity to benchmarks, then trains a lightweight classifier to predict these scores for the full corpus. We compare data selection methods by training over 500 models spanning $10^{19}$ to $10^{22}$ FLOPs and fitting scaling laws to them. From this, we find that simply aligning pretraining data to evaluation benchmarks using BETR achieves a 2.1x compute multiplier over DCLM-Baseline (4.7x over unfiltered data) and improves performance on 9 out of 10 tasks across all scales. BETR also generalizes well: when targeting a diverse set of benchmarks disjoint from our evaluation suite, it still matches or outperforms baselines. Our scaling analysis further reveals a clear trend: larger models require less aggressive filtering. Overall, our findings show that directly matching pretraining data to target tasks precisely shapes model capabilities and highlight that optimal selection strategies must adapt to model scale.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2507.12466>
- PDF: [[raw/papers/pdf/2025-language-models-improve-when-pretraining-data-matches-target-tasks.pdf]]
- Raw markdown: [[raw/papers/md/2025-language-models-improve-when-pretraining-data-matches-target-tasks]]
