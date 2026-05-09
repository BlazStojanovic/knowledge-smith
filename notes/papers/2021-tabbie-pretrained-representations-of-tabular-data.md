---
arxiv: '2105.02584'
authors:
- Hiroshi Iida
- Dung Thai
- Varun Manjunatha
- Mohit Iyyer
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2021-tabbie-pretrained-representations-of-tabular-data.md
raw_pdf: raw/papers/pdf/2021-tabbie-pretrained-representations-of-tabular-data.pdf
read: false
slug: tabbie-pretrained-representations-of-tabular-data
tags:
- tabular
- pretraining
- transformer
- self-supervised
title: 'TABBIE: Pretrained Representations of Tabular Data'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2105.02584v1
venue: null
year: 2021
---

# TABBIE: Pretrained Representations of Tabular Data

> *Hiroshi Iida, Dung Thai, Varun Manjunatha…* — arXiv 2105.02584, 2021

## Abstract

Existing work on tabular representation learning jointly models tables and associated text using self-supervised objective functions derived from pretrained language models such as BERT. While this joint pretraining improves tasks involving paired tables and text (e.g., answering questions about tables), we show that it underperforms on tasks that operate over tables without any associated text (e.g., populating missing cells). We devise a simple pretraining objective (corrupt cell detection) that learns exclusively from tabular data and reaches the state-of-the-art on a suite of table based prediction tasks. Unlike competing approaches, our model (TABBIE) provides embeddings of all table substructures (cells, rows, and columns), and it also requires far less compute to train. A qualitative analysis of our model's learned cell, column, and row representations shows that it understands complex table semantics and numerical trends.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-tabbie-pretrained-representations-of-tabular-data]]
- PDF: `raw/papers/pdf/2021-tabbie-pretrained-representations-of-tabular-data.pdf`
- arXiv: <http://arxiv.org/abs/2105.02584v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-iida-tabbie.md` before that tree was retired.*

TABBIE — NAACL dual-encoder over Wikipedia tables.
