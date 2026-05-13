---
arxiv: '2511.01066'
authors:
- Stephan Oepen et al. (31 co-authors)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2511.01066
  raw: https://arxiv.org/pdf/2511.01066
  source: https://arxiv.org/abs/2511.01066
owner: blaz
read: false
slug: hplt-3-very-large-scale-multilingual-resources
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'HPLT 3.0: Very Large-Scale Multilingual Resources for LLM and MT'
type: note
updated: '2026-05-10'
year: 2025
---

# HPLT 3.0: Very Large-Scale Multilingual Resources for LLM and MT

## Citation

- URL: https://arxiv.org/abs/2511.01066
- PDF: https://arxiv.org/pdf/2511.01066
- Authors: Stephan Oepen et al. (31 co-authors)
- Year / venue: 2025-11 arXiv preprint

## Short Summary

30-trillion-token multilingual dataset for ~200 languages with a complete open-source pipeline for: document selection from web archives, text extraction, language identification, deduplication, annotation with register labels, text quality estimates, and PII removal. Represents the broadest multi-label annotation effort at web scale — each document receives register labels, quality estimates, and PII flags. Also includes 57 monolingual encoder-decoder models and parallel corpora.

## Open Threads

- How do the register annotations compare to Myntti et al.'s CORE-based classification?
- What is the quality-estimate methodology — heuristic, classifier, or LLM-based?
- Can the register + quality + PII triple be used for composable data selection similar to Propella-1's multi-property approach?
