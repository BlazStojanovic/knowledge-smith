---
arxiv: '2505.22232'
authors:
- Mehdi Ali
- Manuel Brack
- Max Lübbering
- Elias Wendt
- Abbas Goher Khan
- Richard Rutmann
- et al
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.22232
  raw: https://arxiv.org/pdf/2505.22232
  source: https://arxiv.org/abs/2505.22232
owner: blaz
read: false
slug: jql-judging-quality-across-languages
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'Judging Quality Across Languages: A Multilingual Approach to Pretraining Data
  Filtering with Language Models'
type: note
updated: '2026-05-10'
year: 2025
---

# Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models

## Citation

- URL: https://arxiv.org/abs/2505.22232
- PDF: https://arxiv.org/pdf/2505.22232
- Authors: Mehdi Ali, Manuel Brack, Max Lübbering, Elias Wendt, Abbas Goher Khan, Richard Rutmann, et al.
- Year / venue: 2025-05 arXiv preprint

## Short Summary

Systematic approach to multilingual data quality annotation using LLM-as-judge distillation. Three-stage pipeline: (1) human annotators create ground-truth quality labels on monolingual documents, (2) labels translated across 35 languages to create multilingual ground truth, (3) top-performing LLM judges selected and distilled into lightweight annotators based on pretrained multilingual embeddings. The lightweight annotators exhibit robust cross-lingual transfer, even for unseen languages and scripts. Substantially outperforms heuristic filtering (FineWeb-2) on downstream model quality and increases data retention rates.

## Open Threads

- How does the cross-lingual transfer quality degrade for very low-resource languages?
- Is the human annotation step scalable, or does it become a bottleneck for new languages?
- How does JQL interact with per-language FastText classifiers (FineWeb-2-HQ approach)?
