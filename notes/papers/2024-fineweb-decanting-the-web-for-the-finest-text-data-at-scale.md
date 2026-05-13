---
arxiv: '2406.17557'
authors:
- Guilherme Penedo
- Hynek Kydlíček
- Loubna Ben Allal
- Anton Lozhkov
- Margaret Mitchell
- Colin Raffel
- Leandro von Werra
- Thomas Wolf (HuggingFace)
created: 2026-04-23
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2406.17557
  raw: '[[raw/papers/md/2024-fineweb-decanting-the-web-for-the-finest-text-data-at-scale]]'
  source: https://arxiv.org/abs/2406.17557
owner: blaz
raw_pdf: raw/papers/pdf/2024-fineweb-decanting-the-web-for-the-finest-text-data-at-scale.pdf
read: false
slug: fineweb-decanting-the-web-for-the-finest-text-data-at-scale
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/llm
- domain/pretraining
- domain/data-mix
title: 'FineWeb: Decanting the Web for the Finest Text Data at Scale'
type: note
updated: '2026-05-10'
year: 2024
---

# FineWeb: Decanting the Web for the Finest Text Data at Scale

## Citation

- URL: https://arxiv.org/abs/2406.17557
- PDF: https://arxiv.org/pdf/2406.17557
- Authors: Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf (HuggingFace)
- Year / venue: 2024-06 arXiv preprint
- Raw PDF: [[raw/papers/pdf/2024-fineweb-decanting-the-web-for-the-finest-text-data-at-scale.pdf]]

## Short Summary

15T-token English pretraining dataset derived from 96 CommonCrawl snapshots. Introduces FineWeb-Edu, a filtered subset scored by an LLM-as-judge educational value classifier distilled into a small model. FineWeb-Edu outperforms FineWeb and other open web datasets on several knowledge/reasoning benchmarks at matched compute. The pipeline, data, and filtering code are released publicly.

## Open Threads

- What are the exact filtering steps and their individual contribution (heuristics vs. dedup vs. quality)?
- How does FineWeb-Edu's educational-value filter interact with topic bias, as the paper reportedly acknowledges?
- How does it compare to DCLM, RedPajama-v2, and Dolma at the same token budget?
- What model size and token budget was used for the downstream ablations?
- FineWeb-2 is referenced in propella-1 — what changed between FineWeb and FineWeb-2?
