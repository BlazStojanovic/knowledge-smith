---
arxiv: '2408.03314'
authors:
- Charlie Snell
- Jaehoon Lee
- Kelvin Xu
- Aviral Kumar
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2408.03314
  raw: '[[raw/papers/md/2024-scaling-compute-optimal-test-time-compute]]'
  source: https://arxiv.org/abs/2408.03314
owner: blaz
raw_pdf: raw/papers/pdf/2024-scaling-compute-optimal-test-time-compute.pdf
read: false
slug: scaling-compute-optimal-test-time-compute
tags:
- type/paper
- status/stub
- source/paper
- domain/inference
- domain/reasoning
title: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling
  Model Parameters
type: note
updated: '2026-05-10'
year: 2024
---

# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

## Citation

- URL: https://arxiv.org/abs/2408.03314
- Authors: Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- Affiliation: UC Berkeley, Google DeepMind
- Year / venue: 2024 / arXiv
- arXiv: 2408.03314
- **Raw**: [[raw/papers/pdf/2024-scaling-compute-optimal-test-time-compute.pdf]]

## Core Claim

Test-time compute scaling can be more effective than scaling model parameters. A smaller model with optimally allocated test-time compute can match or exceed a 14× larger model on reasoning tasks. The optimal allocation strategy depends on problem difficulty.

## Key Ideas

- Two mechanisms for test-time scaling: (1) searching against a process-based verifier (PRM), (2) adaptively updating the model's distribution (e.g., revising a response)
- Compute-optimal test-time allocation: for easy problems, a single sample is sufficient; for hard problems, repeated sampling + verification is optimal
- The crossover: below a difficulty threshold, scale model size; above it, scale test-time compute
- Practical implication: given a fixed total compute budget (train + inference), smaller models with more inference compute can outperform larger models with less inference

## Methodology

MATH benchmark and variants. PaLM 2 models at multiple scales. Process reward models for verification. Best-of-N sampling and sequential revision as test-time strategies.

## Relevance To Poolside

Directly relevant to inference cost decisions. If Poolside trains a smaller model and invests in test-time scaling (verification, repeated sampling), it may achieve better performance per dollar than a larger model with fewer inference passes.

## Related Notes

- [[concepts/scaling-laws-foundational]] — test-time scaling as an alternative to pretrain scaling
- [[maps/scaling-laws/landscape]] — test-time compute domain
- [[concepts/pass-at-k-methodology]] — repeated sampling is the simplest test-time scaling
