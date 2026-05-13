---
aliases:
- CCNet
- ccnet
arxiv: '1911.00359'
authors:
- Guillaume Wenzek
- Marie-Anne Lample
- Alexis Conneau
- Ludovic Denoyer
- Marc'Aurelio Ranzato
- Armand Joulin (Facebook AI Research)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1911.00359
  raw: '[[raw/papers/md/2019-ccnet-extracting-high-quality-monolingual-datasets-from-web-crawl-data]]'
  source: https://arxiv.org/abs/1911.00359
owner: blaz
raw_pdf: raw/papers/pdf/2019-ccnet-extracting-high-quality-monolingual-datasets-from-web-crawl-data.pdf
read: false
slug: ccnet-extracting-high-quality-monolingual-datasets-from-web-crawl-data
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/data-mix
title: 'CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data'
type: note
updated: '2026-05-10'
year: 2019
---

# CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data

## Citation

- URL: https://arxiv.org/abs/1911.00359
- PDF: https://arxiv.org/pdf/1911.00359
- Authors: Guillaume Wenzek, Marie-Anne Lample, Alexis Conneau, Ludovic Denoyer, Marc'Aurelio Ranzato, Armand Joulin (Facebook AI Research)
- Year / venue: 2019-11 arXiv preprint; LREC 2020
- Raw PDF: [[raw/papers/pdf/2019-ccnet-extracting-high-quality-monolingual-datasets-from-web-crawl-data.pdf]]

## Short Summary

Pipeline for building large monolingual datasets from CommonCrawl. Three-stage process: (1) paragraph-level deduplication using SHA-1 hashing, (2) language identification via fastText, (3) perplexity-based quality filtering using KenLM 5-gram models trained on Wikipedia. Documents are bucketed into head/middle/tail perplexity bins; the head bin (most Wikipedia-like) is typically used for training. The canonical reference for perplexity filtering as a data-quality signal. Used as a component in LLaMA, CCNet-derived corpora, and cited as a baseline in most subsequent web-text curation work.

## Open Threads

- What are the exact perplexity thresholds used to define head/middle/tail bins, and are they language-specific?
- How does KenLM perplexity filtering compare to classifier-based approaches (DCLM fastText, FineWeb-Edu LLM judge) in downstream model quality?
- Does perplexity filtering introduce topic/style bias (Wikipedia-like text preferred) that affects downstream task coverage?
- CCNet operates at paragraph level for dedup but document level for perplexity — what is the effect of this granularity mismatch?
- How does perplexity filtering interact with non-English languages where Wikipedia coverage is thin?
