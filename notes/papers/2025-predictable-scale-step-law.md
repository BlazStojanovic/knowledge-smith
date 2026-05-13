---
arxiv: '2503.04715'
authors:
- (Step AI / joint team)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2503.04715
  raw: '[[raw/papers/md/2025-predictable-scale-step-law]]'
  source: https://arxiv.org/abs/2503.04715
owner: blaz
raw_pdf: raw/papers/pdf/2025-predictable-scale-step-law.pdf
read: false
slug: predictable-scale-step-law
tags:
- type/paper
- status/stub
- source/paper
- domain/pretraining
- domain/training
title: 'Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in Large Language
  Model Pretraining'
type: note
updated: '2026-05-10'
year: 2025
---

# Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in Large Language Model Pretraining

## Citation

- URL: https://arxiv.org/abs/2503.04715
- Authors: (Step AI / joint team)
- Year / venue: 2025 / arXiv
- arXiv: 2503.04715
- **Raw**: [[raw/papers/pdf/2025-predictable-scale-step-law.pdf]]

## Core Claim

Optimal training hyperparameters (learning rate, batch size, weight decay) follow power law relationships with model size N and dataset size D. These laws, validated across 3,700 LLMs and ~1M GPU-hours, predict hyperparameters within 0.094% of exhaustive search global optimum.

## Key Ideas

- Optimal LR follows power law in both N and D: η_opt ∝ N^a × D^b
- Optimal batch size scales primarily with D: B_opt ∝ D^c
- First framework to unify dense and MoE models across diverse data recipes
- 3,700 LLMs trained from scratch across 100 trillion tokens
- Interactive tool at step-law.github.io for practical use
- Represents the most comprehensive empirical HP scaling study to date

## Relevance To Poolside

Could directly inform Poolside's hyperparameter selection for new training runs. Instead of expensive HP sweeps, use the step law to predict optimal LR and batch size from model size and data budget.

## Related Notes

- [[concepts/hyperparameter-scaling]] — step law as the most actionable HP scaling result
- [[concepts/compute-optimal-methodology]] — reduces the cost of HP tuning
- [[maps/scaling-laws/landscape]] — hyperparameters domain
