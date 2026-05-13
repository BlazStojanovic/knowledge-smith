---
tags: [type/source, status/verified, source/paper, confidential/public-source]
source_type: paper
url: https://jmlr.org/papers/v26/24-1000.html
pdf: https://jmlr.org/papers/volume26/24-1000/24-1000.pdf
neurips: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html
openreview: https://openreview.net/forum?id=j5BuTrEj35
arxiv: https://arxiv.org/abs/2305.16264
retrieved: 2026-04-22
title: Scaling Data-Constrained Language Models
authors: Niklas Muennighoff; Alexander M. Rush; Boaz Barak; Teven Le Scao; Aleksandra Piktus; Nouamane Tazi; Sampo Pyysalo; Thomas Wolf; Colin Raffel
venue: JMLR 26(53), 2025; NeurIPS 2023 oral
license: CC-BY-4.0
local_pdf: raw/papers/2023-scaling-data-constrained-language-models.pdf
---

# Scaling Data-Constrained Language Models - Raw Source Notes

## Relevant Extract

- The JMLR abstract reports experiments over data repetition and compute budget, up to 900B training tokens and 9B parameter models.
- The key claim for this vault: in constrained-data settings with fixed compute, repeated data for up to 4 epochs has negligible loss impact relative to unique data.
- The same abstract says that beyond this regime, additional repetition has diminishing value.
- The introduction adds that validation-loss differences are generally insignificant up to 4 epochs and do not show downstream task differences.

## Local File

- PDF: `raw/papers/2023-scaling-data-constrained-language-models.pdf`

## Source Pointers

- JMLR page: https://jmlr.org/papers/v26/24-1000.html
- JMLR PDF: https://jmlr.org/papers/volume26/24-1000/24-1000.pdf
- NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=j5BuTrEj35
