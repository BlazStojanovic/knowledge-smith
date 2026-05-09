---
arxiv: '2205.09328'
authors:
- Zifeng Wang
- Jimeng Sun
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2205.09328.md
raw_pdf: raw/papers/2205.09328.pdf
read: false
slug: transtab-learning-transferable-tabular-transformers-across
tags:
- tabular
- transformer
- pretraining
- fine-tuning
title: 'TransTab: Learning Transferable Tabular Transformers Across Tables'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2205.09328v2
venue: null
year: 2022
---

# TransTab: Learning Transferable Tabular Transformers Across Tables

> *Zifeng Wang, Jimeng Sun* — arXiv 2205.09328, 2022

## Abstract

Tabular data (or tables) are the most widely used data format in machine learning (ML). However, ML models often assume the table structure keeps fixed in training and testing. Before ML modeling, heavy data cleaning is required to merge disparate tables with different columns. This preprocessing often incurs significant data waste (e.g., removing unmatched columns and samples). How to learn ML models from multiple tables with partially overlapping columns? How to incrementally update ML models as more columns become available over time? Can we leverage model pretraining on multiple distinct tables? How to train an ML model which can predict on an unseen table?
  To answer all those questions, we propose to relax fixed table structures by introducing a Transferable Tabular Transformer (TransTab) for tables. The goal of TransTab is to convert each sample (a row in the table) to a generalizable embedding vector, and then apply stacked transformers for feature encoding. One methodology insight is combining column description and table cells as the raw input to a gated transformer model. The other insight is to introduce supervised and self-supervised pretraining to improve model performance. We compare TransTab with multiple baseline methods on diverse benchmark datasets and five oncology clinical trial datasets. Overall, TransTab ranks 1.00, 1.00, 1.78 out of 12 methods in supervised learning, feature incremental learning, and transfer learning scenarios, respectively; and the proposed pretraining leads to 2.3% AUC lift on average over the supervised learning.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2205.09328]]
- PDF: `raw/papers/2205.09328.pdf`
- arXiv: <http://arxiv.org/abs/2205.09328v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-wang-transtab.md` before that tree was retired.*

TransTab — transformer that serializes rows with column-name tokens across heterogeneous tables; direct precursor to TabPFN/CARTE framing.
