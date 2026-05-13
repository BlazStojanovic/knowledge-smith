---
aliases:
- RedPajama V2
- RedPajama-Data-v2
arxiv: '2411.12372'
authors:
- Maurice Weber
- Daniel Y. Fu
- Quentin Anthony
- Yonatan Oren
- Shane Adams
- Anton Alexandrov
- Xiaozhong Lyu
- Huu Nguyen
- Xiaozhe Yao
- Virginia Adams
- Ben Athiwaratkun
- Rahul Chalamala
- Kezhen Chen
- Max Ryabinin
- Tri Dao
- Percy Liang
- Christopher Ré
- Irina Rish
- Ce Zhang (Together AI
- Stanford
- Mila)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2411.12372
  raw: https://arxiv.org/pdf/2411.12372
  source: https://arxiv.org/abs/2411.12372
owner: blaz
read: false
slug: redpajama-data-v2
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/pretraining
- domain/data-mix
title: 'RedPajama-Data-v2: An Open Dataset for Training Large Language Models'
type: note
updated: '2026-05-10'
year: 2024
---

# RedPajama-Data-v2: An Open Dataset for Training Large Language Models

## Citation

- URL: https://arxiv.org/abs/2411.12372
- PDF: https://arxiv.org/pdf/2411.12372
- Authors: Maurice Weber, Daniel Y. Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, Ce Zhang (Together AI, Stanford, Mila)
- Year / venue: 2024-11 arXiv preprint

## Short Summary

Over 100 trillion tokens of multilingual web text (5 languages, 84 CommonCrawl snapshots) accompanied by precomputed per-document quality signals including: (1) heuristic rule-based signals (word count, sentence count, character ratios), (2) n-gram perplexity scores, (3) fastText quality classifier scores, and (4) deduplication signatures. Positions itself as a "signals + metadata" approach — providing quality signals alongside raw data rather than a single filtered dataset, enabling downstream users to compose their own filtering pipelines.

## Open Threads

- How do the precomputed quality signals compare to more expensive classifiers (FineWeb-Edu, QuRating) for downstream model quality?
- Does the multi-signal approach enable Pareto-optimal filtering strategies that single-score approaches cannot reach?
- How does scale (100T+ tokens) interact with filtering threshold — does the Scaling Laws for Data Filtering insight apply here?
