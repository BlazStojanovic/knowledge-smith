---
arxiv: '2412.08905'
authors:
- Marah Abdin et al. (Microsoft
- 27 authors)
created: 2026-04-22
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2412.08905
  raw: https://arxiv.org/pdf/2412.08905
  source: https://arxiv.org/abs/2412.08905
owner: blaz
raw_pdf: raw/papers/pdf/2024-phi-4-technical-report.pdf
read: false
slug: phi-4-technical-report
tags:
- type/paper
- status/stub
- source/primary
- confidential/public-source
- domain/synth-data
- domain/pretraining
- domain/llm
- domain/models
title: Phi-4 Technical Report
type: note
updated: '2026-05-10'
year: 2024
---

# Phi-4 Technical Report

## Citation

- URL: https://arxiv.org/abs/2412.08905
- PDF: https://arxiv.org/pdf/2412.08905
- Authors: Marah Abdin et al. (Microsoft, 27 authors)
- Year / venue: 2024-12 arXiv preprint
- arXiv: 2412.08905

## Short Summary

14B parameter language model. Training recipe is centrally focused on data quality, with synthetic data incorporated throughout the pretraining process — a departure from the web-corpus-first approach of most models. Surpasses its GPT-4 teacher on STEM-focused QA despite being much smaller. Strong performance on reasoning benchmarks attributed to data quality and training curriculum rather than architectural changes from phi-3.

## Open Threads

- How is synthetic data mixed and at what ratio across pretraining stages?
- What types of synthetic data (rephrasing, generation, distillation) does phi-4 use?
- How does phi-4's synthetic data strategy compare to [[notes/papers/2026-finephrase-systematic-study-of-pretraining-data-rephrasing]]?
