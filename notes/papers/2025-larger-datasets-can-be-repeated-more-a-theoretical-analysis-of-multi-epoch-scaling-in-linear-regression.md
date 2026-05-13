---
arxiv: '2511.13421'
authors:
- Tingkai Yan
- Haodong Wen
- Binghui Li
- Kairong Luo
- Wenguang Chen
- Kaifeng Lyu
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2511.13421
  raw: '[[raw/papers/md/2025-larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scaling-in-linear-regression]]'
  source: https://arxiv.org/abs/2511.13421
owner: blaz
raw_pdf: raw/papers/pdf/2025-larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scaling-in-linear-regression.pdf
read: false
slug: larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scaling-in-linear-regression
tags:
- type/paper
- status/stub
- source/paper
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/repetition
- domain/data-mix
- domain/models
title: 'Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch
  Scaling in Linear Regression'
type: note
updated: '2026-05-10'
year: 2025
---

# Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression

## Citation

- URL: https://arxiv.org/abs/2511.13421
- PDF: https://arxiv.org/pdf/2511.13421
- Authors: Tingkai Yan, Haodong Wen, Binghui Li, Kairong Luo, Wenguang Chen, Kaifeng Lyu
- Year / venue: 2025-11-17 arXiv preprint
- arXiv: 2511.13421v2
- Categories: cs.LG, stat.ML
- Raw PDF: [[raw/papers/pdf/2025-larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scaling-in-linear-regression.pdf]]
- Source filename: `2511.13421v2.pdf`

## Short Summary

While data scaling laws of large language models (LLMs) have been widely examined in the one-pass regime with massive corpora, their form under limited data and repeated epochs remains largely unexplored. This paper presents a theoretical analysis of how a common workaround, training for multiple epochs on the same dataset, reshapes the data scaling laws in linear regression.

## Relevance To Poolside

Our interpretation: keep this as an unread source for future grounding. Use it when its method or claim becomes load-bearing for a Poolside hypothesis, experiment, model note, or data-method decision.

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Repetition mode: theoretical multi-epoch reuse, with an LLM validation run at 0.3B parameters on DCLM.
- Memo-grounded claim: the effective reuse threshold is not scale-invariant; larger unique datasets can tolerate more epochs before repeated data saturates.
- Memo-grounded formula: the review records a fit of roughly `K(lambda=0.75, N) ~= 0.80 log N + 5.21` for the 0.3B LLM validation setting.
- Poolside implication: the "4 epochs are okay" prior should not be treated as universal; it may shift with unique-data scale and schedule.

## Related Notes

- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
- [[concepts/scaling-laws-foundational]] — challenges Muennighoff's scale-invariance assumption
- [[concepts/data-repetition]] — scale-dependent reuse
- [[maps/scaling-laws/landscape]] — data-constrained domain

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
