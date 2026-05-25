---
arxiv: '2602.05910'
authors:
- Seoirse Murray
- Allison Qi
- Timothy Qian
- John Schulman
- Collin Burns
- Sara Price
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.05910
  raw: '[[raw/papers/md/2026-chunky-post-training-data-driven-failures-of-generalization]]'
  source: https://arxiv.org/abs/2602.05910
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-chunky-post-training-data-driven-failures-of-generalization.md
raw_pdf: raw/papers/pdf/2026-chunky-post-training-data-driven-failures-of-generalization.pdf
read: false
slug: chunky-post-training-data-driven-failures-of-generalization
tags:
- type/paper
- status/stub
title: 'Chunky Post-Training: Data Driven Failures of Generalization'
type: note
updated: '2026-05-25'
year: 2026
---

# Chunky Post-Training: Data Driven Failures of Generalization

> *Seoirse Murray, Allison Qi, Timothy Qian, John Schulman, Collin Burns, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

LLM post-training involves many diverse datasets, each targeting a specific behavior. But these datasets encode incidental patterns alongside intended ones: correlations between formatting and content, narrow phrasings across diverse problems, and implicit associations arising from the discrete data curation process. These patterns are often invisible to developers yet salient to models, producing behaviors that surprise their creators, such as rejecting true facts presented in a particular question format. We call this chunky post-training: the model learns spurious correlations as a result of distinct chunks of post-training data. We introduce SURF, a black-box pipeline which surfaces these unintended behaviors at run time, and TURF, a tool that traces these failures back to specific post-training data. Applying these tools to frontier models (Claude 4.5, GPT-5.1, Grok 4.1, Gemini 3) and open models (Tülu 3), we show that chunky post-training produces miscalibrated behaviors, which often result from imbalanced or underspecified chunks of post-training data.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.05910>
- PDF: [[raw/papers/pdf/2026-chunky-post-training-data-driven-failures-of-generalization.pdf]]
- Raw markdown: [[raw/papers/md/2026-chunky-post-training-data-driven-failures-of-generalization]]
