---
arxiv: '1812.06162'
authors:
- Sam McCandlish
- Jared Kaplan
- Dario Amodei
- OpenAI team
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1812.06162
  raw: '[[raw/papers/md/2018-empirical-model-large-batch-training]]'
  source: https://arxiv.org/abs/1812.06162
owner: blaz
raw_pdf: raw/papers/pdf/2018-empirical-model-large-batch-training.pdf
read: false
slug: empirical-model-large-batch-training
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/training
title: An Empirical Model of Large-Batch Training
type: note
updated: '2026-05-10'
year: 2018
---

# An Empirical Model of Large-Batch Training

## Citation

- URL: https://arxiv.org/abs/1812.06162
- Authors: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI team
- Affiliation: OpenAI, Johns Hopkins University
- Year / venue: 2018 / arXiv
- arXiv: 1812.06162
- **Raw**: [[raw/papers/pdf/2018-empirical-model-large-batch-training.pdf]]

## Core Claim

The **gradient noise scale** B_noise — the ratio of gradient variance to squared gradient norm — predicts the **critical batch size** B_crit, the threshold below which doubling batch size halves training time and above which returns diminish. B_noise can be measured cheaply during training.

## Key Ideas

- B_crit = B_noise / L (loss). As training progresses and loss decreases, B_crit increases — larger batches become more efficient later in training
- Below B_crit: perfect data-parallel scaling (2× batch = 2× speed). Above B_crit: diminishing returns converging to constant speed
- Practical batch size strategy: start small, increase during training (used in GPT-3, PaLM, LLaMA)
- The gradient noise scale is a simple statistic that can be computed from two gradient estimates at different batch sizes
- Validated across multiple domains (language modeling, image classification, game-playing)

## Relevance To Poolside

Directly applicable to batch size scheduling in Poolside training runs. The critical batch size framework predicts when to increase batch size for maximum compute efficiency.

## Related Notes

- [[concepts/hyperparameter-scaling]] — critical batch size as a scaling quantity
- [[concepts/compute-optimal-methodology]] — batch size as part of the HP optimization problem
- [[maps/scaling-laws/landscape]] — hyperparameters domain
