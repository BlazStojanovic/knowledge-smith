---
arxiv: '2305.13230'
authors:
- Fuzhao Xue
- Yao Fu
- Wangchunshu Zhou
- Zangwei Zheng
- Yang You
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2305.13230
  raw: '[[raw/papers/md/2023-to-repeat-or-not-to-repeat-insights-from-scaling-llm-under-token-crisis]]'
  source: https://arxiv.org/abs/2305.13230
owner: blaz
raw_pdf: raw/papers/pdf/2023-to-repeat-or-not-to-repeat-insights-from-scaling-llm-under-token-crisis.pdf
read: false
slug: to-repeat-or-not-to-repeat-insights-from-scaling-llm-under-token-crisis
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
title: 'To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis'
type: note
updated: '2026-05-10'
year: 2023
---

# To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis

## Citation

- URL: https://arxiv.org/abs/2305.13230
- PDF: https://arxiv.org/pdf/2305.13230
- Authors: Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, Yang You
- Year / venue: 2023-05-22 arXiv preprint
- arXiv: 2305.13230v2
- Categories: cs.LG, cs.AI, cs.CL
- Raw PDF: [[raw/papers/pdf/2023-to-repeat-or-not-to-repeat-insights-from-scaling-llm-under-token-crisis.pdf]]
- Source filename: `2305.13230v2.pdf`

## Short Summary

Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text data on the web is approaching its scaling limit for LLMs.

## Relevance To Poolside

Our interpretation: keep this as an unread source for future grounding. Use it when its method or claim becomes load-bearing for a Poolside hypothesis, experiment, model note, or data-method decision.

## Extracted From Repetition Memo

- Source review: [[raw/reviews/2026-scaling-laws-data-repetition-review]].
- Repetition mode: full-dataset multi-epoch training under token-crisis conditions.
- Memo-grounded claim: larger T5-style models overfit faster under repetition at fixed compute; parameter count matters more than raw FLOPs for multi-epoch degradation.
- Memo-grounded mitigation: dropout is highlighted as the strongest regularization intervention among the tested options.
- Poolside implication: if Laguna-scale runs enter repeated-data regimes, model size and regularization choices should be treated as part of the repetition experiment.

## Related Notes

- [[hypotheses/seed-repetition-at-laguna-xs-can-hurt-quality]]
- [[concepts/scaling-laws-foundational]] — data-constrained scaling
- [[concepts/data-repetition]] — token-crisis multi-epoch regime
- [[maps/scaling-laws/landscape]] — data-constrained domain

## Reading State

- Tagged `read/unread`; Blaz has not marked this as read yet.
