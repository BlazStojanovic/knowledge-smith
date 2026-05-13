---
arxiv: '2504.11393'
authors:
- Ian Magnusson
- Nguyen Tai
- Ben Bogin
- David Heineman
- Jena D. Hwang
- Luca Soldaini
- Akshita Bhagia
- Jiacheng Liu
- Dirk Groeneveld
- Oyvind Tafjord
- Noah A. Smith
- Pang Wei Koh
- Jesse Dodge
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.11393
  raw: '[[raw/papers/md/2025-datadecide-how-to-predict-best-pretraining-data-with-small-experiments]]'
  source: https://arxiv.org/abs/2504.11393
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-datadecide-how-to-predict-best-pretraining-data-with-small-experiments.md
raw_pdf: raw/papers/pdf/2025-datadecide-how-to-predict-best-pretraining-data-with-small-experiments.pdf
read: false
slug: datadecide-how-to-predict-best-pretraining-data-with-small-experiments
tags:
- type/paper
- status/stub
title: 'DataDecide: How to Predict Best Pretraining Data with Small Experiments'
type: note
updated: '2026-05-11'
year: 2025
---

# DataDecide: How to Predict Best Pretraining Data with Small Experiments

> *Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena D. Hwang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Because large language models are expensive to pretrain on different datasets, using smaller-scale experiments to decide on data is crucial for reducing costs. Which benchmarks and methods of making decisions from observed performance at small scale most accurately predict the datasets that yield the best large models? To empower open exploration of this question, we release models, data, and evaluations in DataDecide -- the most extensive open suite of models over differences in data and scale. We conduct controlled pretraining experiments across 25 corpora with differing sources, deduplication, and filtering up to 100B tokens, model sizes up to 1B parameters, and 3 random seeds. We find that the ranking of models at a single, small size (e.g., 150M parameters) is a strong baseline for predicting best models at our larger target scale (1B) (~80% of com parisons correct). No scaling law methods among 8 baselines exceed the compute-decision frontier of single-scale predictions, but DataDecide can measure improvement in future scaling laws. We also identify that using continuous likelihood metrics as proxies in small experiments makes benchmarks including MMLU, ARC, HellaSwag, MBPP, and HumanEval >80% predictable at the target 1B scale with just 0.01% of the compute.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2504.11393>
- PDF: [[raw/papers/pdf/2025-datadecide-how-to-predict-best-pretraining-data-with-small-experiments.pdf]]
- Raw markdown: [[raw/papers/md/2025-datadecide-how-to-predict-best-pretraining-data-with-small-experiments]]
