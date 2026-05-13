---
aliases:
- DCLM
- dclm
- datacomp-lm
arxiv: '2406.11794'
authors:
- Jeffrey Li
- Alex Fang
- Georgios Smyrnis
- Maor Goldfarb
- Jordan Hoffman
- et al. (DataComp consortium)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2406.11794
  raw: '[[raw/papers/md/2024-dclm-datacomp-for-language-models]]'
  source: https://arxiv.org/abs/2406.11794
owner: blaz
raw_pdf: raw/papers/pdf/2024-dclm-datacomp-for-language-models.pdf
read: false
slug: dclm-datacomp-for-language-models
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/data-mix
- domain/evals
title: 'DCLM: DataComp for Language Models'
type: note
updated: '2026-05-10'
year: 2024
---

# DCLM: DataComp for Language Models

## Citation

- URL: https://arxiv.org/abs/2406.11794
- PDF: https://arxiv.org/pdf/2406.11794
- Authors: Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Goldfarb, Jordan Hoffman, et al. (DataComp consortium)
- Year / venue: 2024-06 arXiv preprint
- Raw PDF: [[raw/papers/pdf/2024-dclm-datacomp-for-language-models.pdf]]

## Short Summary

Benchmark and dataset for studying data curation decisions for LLM pretraining. Holds model architecture and training compute fixed; data pipeline is the variable. Releases DCLM-Baseline, a 7T-token filtered CommonCrawl dataset, and a benchmark suite (DCLM-Eval) spanning 53 tasks. DCLM-Baseline trains competitive 7B models at Chinchilla-optimal compute. Key finding: text-quality filtering (fastText classifier trained on high-quality web text) matters more than deduplication at this scale. Referenced extensively in the rephrasing literature (FinePhrase, REWIRE, BeyondWeb) as a strong web-text baseline.

## Open Threads

- What is the exact filtering pipeline: heuristics, dedup, quality classifier — and what ablations exist?
- How does the fastText quality classifier compare to LLM-as-judge approaches (e.g., FineWeb-Edu)?
- At what model size / token budget does DCLM-Baseline's advantage over alternatives hold?
- What is the relationship between DCLM the benchmark and DCLM-Baseline the dataset — can other recipes be slotted in?
- How does DCLM-Baseline perform as a mix-in (as used in FinePhrase) vs. as a standalone training corpus?
