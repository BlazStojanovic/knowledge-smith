---
aliases:
- T5
- C4
arxiv: '1910.10683'
authors:
- Colin Raffel
- Noam Shazeer
- Adam Roberts
- Katherine Lee
- Sharan Narang
- Michael Matena
- Yanqi Zhou
- Wei Li
- Peter J. Liu (Google)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1910.10683
  raw: https://arxiv.org/pdf/1910.10683
  source: https://arxiv.org/abs/1910.10683
owner: blaz
read: false
slug: exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/pretraining
- domain/llm
title: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
type: note
updated: '2026-05-10'
year: 2020
---

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

## Citation

- URL: https://arxiv.org/abs/1910.10683
- PDF: https://arxiv.org/pdf/1910.10683
- Authors: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (Google)
- Year / venue: 2019-10 arXiv; JMLR 2020

## Short Summary

Introduces the T5 model and the Colossal Clean Crawled Corpus (C4) — a 750GB cleaned subset of CommonCrawl. C4 filtering uses heuristic rules: language detection (langdetect), deduplication (three-sentence dedup), removal of pages with obscenities (blocklist), removal of pages containing "{" (JavaScript/code), sentences not ending in terminal punctuation, and pages with fewer than 5 sentences. Canonical reference for rule-based heuristic filtering as a pretraining data pipeline component. FineWeb's ablation study later showed several C4 rules have low or negative impact on downstream performance.
