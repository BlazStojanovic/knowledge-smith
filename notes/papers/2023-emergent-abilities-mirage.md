---
arxiv: '2304.15004'
authors:
- Schaeffer
- Miranda
- Koyejo (Stanford
- UT Austin)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2304.15004
  raw: '[[raw/papers/md/2023-emergent-abilities-mirage]]'
  source: https://arxiv.org/abs/2304.15004
owner: blaz
raw_pdf: raw/papers/pdf/2023-emergent-abilities-mirage.pdf
read: false
slug: emergent-abilities-mirage
tags:
- type/paper
- status/draft
- domain/evals
- domain/reasoning
- source/primary
title: Are Emergent Abilities of Large Language Models a Mirage?
type: note
updated: '2026-05-10'
year: 2023
---

# Are Emergent Abilities of Large Language Models a Mirage?

## Citation

- URL: https://arxiv.org/abs/2304.15004
- Authors: Schaeffer, Miranda, Koyejo (Stanford, UT Austin)
- Year / venue: 2023 / arXiv
- **Raw**: [[raw/papers/pdf/2023-emergent-abilities-mirage]]

## Core Claim

Emergent abilities are primarily a mirage caused by the researcher choosing a metric that nonlinearly or discontinuously deforms per-token error rates, and secondarily by insufficient test data to estimate small-model performance. Over 92% of claimed emergent abilities on BIG-Bench appear under just 2 metrics: Multiple Choice Grade (discontinuous) and Exact String Match (nonlinear). Switching to linear/continuous metrics reveals smooth improvement.

## Key Paper Ideas

- **Metric-driven emergence**: for multi-token exact match, Accuracy(N) ∝ exp(-L·(N/c)^α) — nonlinear transformation of smooth per-token improvement creates apparent sharp transition. For Multiple Choice Grade, a step function at 0.5 threshold creates discontinuity.
- **Resolution**: measurement resolution is 1/(N·L) where N=test examples, L=target length. Small models appear to have zero performance when resolution is too coarse.
- **Metric taxonomy**: nonlinear (Exact Match), discontinuous (MCG), linear (Token Edit Distance), continuous (Brier Score). Linear/continuous metrics reveal smooth improvement where nonlinear/discontinuous show emergence.
- **92% statistic**: of 39 BIG-Bench metrics, at most 5 produce emergent task-metric-model triples. Over 92% of hand-annotated emergent abilities appear under just MCG and Exact String Match.
- **Emergence can be manufactured**: demonstrated in vision tasks (FC, CNN, Transformer) by choosing appropriate discontinuous/nonlinear metrics — emergence is not specific to LLMs or even language.

## Methodology

Three complementary tests:
1. **GPT-3 family**: change metric from Accuracy to Token Edit Distance → emergence disappears. Increase test data → smooth above-chance accuracy visible at all model sizes.
2. **BIG-Bench meta-analysis**: emergent abilities cluster by metric, not by task. LaMDA emergent abilities under MCG disappear under Brier Score.
3. **Vision experiments**: induce emergence in autoencoders, LeNets, Transformers on CIFAR100/MNIST/Omniglot by choosing threshold-based metrics.

## Key Results

- 92% of claimed BIG-Bench emergent abilities appear under just 2 metrics (MCG, Exact String Match)
- GPT-3 arithmetic: under Token Edit Distance, smooth improvement visible across all sizes (350M–175B)
- With increased test data, even 350M model achieves above-chance accuracy
- LaMDA emergent abilities under MCG disappear under Brier Score

## Core Concepts

- [[concepts/evaluation-scaling-laws]] — emergence as metric artifact vs real capability transition
- [[concepts/evaluation-variance]] — test set size as a resolution constraint
- [[concepts/benchmark-saturation]] — the inverse problem: benchmarks appearing "too hard" due to metric choice

## Relevance To Poolside

Directly actionable: when evaluating scaling behavior during training, report results under both discrete/threshold metrics AND continuous/graded metrics. If emergence appears only under one class of metric, it is likely an artifact. For Poolside's eval suite, this means tracking Brier scores or per-token metrics alongside accuracy, especially for smaller checkpoint evaluations.

## Blaz Notes

- 

## Key Follow-Ups / Jumping-Off Points

- Does the metric-artifact explanation account for CoT emergence? (CoT being harmful at small scale, beneficial at large scale, is not obviously a metric effect)
- The 92% statistic is compelling but relies on BIG-Bench's metric assignments — what about non-BIG-Bench emergence claims?
- What is the resolution requirement for Poolside's benchmarks at different model scales?

## Related Notes

- Papers: [[notes/papers/2022-emergent-abilities]] (the paper being rebutted)
- Concepts: [[concepts/evaluation-scaling-laws]]
- Questions: [[questions/model-evaluation-methodology]] §proxy evals across scales

## Caveats

- Does not claim LLMs cannot have emergent abilities — only that prior claims are likely metric artifacts
- Independence assumption (per-token errors independent) is empirically false
- Limited to publicly queryable models (4 GPT-3 sizes)
- Does not address all forms of emergence (e.g., qualitatively new capabilities like CoT reasoning)
- Caballero et al. alternative (broken neural scaling laws) remains possible
