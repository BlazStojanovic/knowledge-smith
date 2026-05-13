---
arxiv: '2411.12925'
authors:
- David Brandfonbrener
- Nikhil Anand
- Nikhil Vyas
- Eran Malach
- Sham M. Kakade
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2411.12925
  raw: '[[raw/papers/md/2024-loss-to-loss-prediction]]'
  source: https://arxiv.org/abs/2411.12925
owner: blaz
raw_pdf: raw/papers/pdf/2024-loss-to-loss-prediction.pdf
read: false
slug: loss-to-loss-prediction
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/evals
title: 'Loss-to-Loss Prediction: Scaling Laws for All Datasets'
type: note
updated: '2026-05-10'
year: 2024
---

# Loss-to-Loss Prediction: Scaling Laws for All Datasets

## Citation

- URL: https://arxiv.org/abs/2411.12925
- Authors: David Brandfonbrener, Nikhil Anand, Nikhil Vyas, Eran Malach, Sham M. Kakade
- Year / venue: 2024 / arXiv
- arXiv: 2411.12925
- **Raw**: [[raw/papers/pdf/2024-loss-to-loss-prediction.pdf]]

## Core Claim

Loss on one dataset can predict loss on another dataset via shifted power law relationships, without needing to evaluate downstream tasks. The form L_test(L_train) = a·(L_train - c)^b + d extrapolates reliably at 20× the largest FLOP budget.

## Key Ideas

- Three prediction modes: train-to-train, train-to-test, test-to-test
- Shifted power law captures the relationship between losses on different distributions
- Enables predicting performance on new datasets from small-scale experiments without ever evaluating on the target
- Complementary to Gadre et al.'s loss-to-downstream pipeline — this predicts loss-to-loss rather than loss-to-accuracy

## Relevance To Poolside

Could enable predicting performance on Poolside's internal eval distributions from public benchmark losses, reducing eval compute.

## Related Notes

- [[concepts/compute-optimal-methodology]] — loss-to-loss as methodology
- [[concepts/evaluation-scaling-laws]] — downstream prediction
- [[maps/scaling-laws/landscape]] — evaluation/downstream domain
