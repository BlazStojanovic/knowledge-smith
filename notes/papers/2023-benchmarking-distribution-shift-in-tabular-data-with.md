---
arxiv: '2312.07577'
authors:
- Josh Gardner
- Zoran Popovic
- Ludwig Schmidt
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2023-benchmarking-distribution-shift-in-tabular-data-with.md
raw_pdf: raw/papers/pdf/2023-benchmarking-distribution-shift-in-tabular-data-with.pdf
read: false
slug: benchmarking-distribution-shift-in-tabular-data-with
tags:
- tabular
- benchmark
- generalization
- fairness
title: Benchmarking Distribution Shift in Tabular Data with TableShift
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2312.07577v3
venue: null
year: 2023
---

# Benchmarking Distribution Shift in Tabular Data with TableShift

> *Josh Gardner, Zoran Popovic, Ludwig Schmidt* — arXiv 2312.07577, 2023

## Abstract

Robustness to distribution shift has become a growing concern for text and image models as they transition from research subjects to deployment in the real world. However, high-quality benchmarks for distribution shift in tabular machine learning tasks are still lacking despite the widespread real-world use of tabular data and differences in the models used for tabular data in comparison to text and images. As a consequence, the robustness of tabular models to distribution shift is poorly understood. To address this issue, we introduce TableShift, a distribution shift benchmark for tabular data. TableShift contains 15 binary classification tasks in total, each with an associated shift, and includes a diverse set of data sources, prediction targets, and distribution shifts. The benchmark covers domains including finance, education, public policy, healthcare, and civic participation, and is accessible using only a few lines of Python code via the TableShift API. We conduct a large-scale study comparing several state-of-the-art tabular data models alongside robust learning and domain generalization methods on the benchmark tasks. Our study demonstrates (1) a linear trend between in-distribution (ID) and out-of-distribution (OOD) accuracy; (2) domain robustness methods can reduce shift gaps but at the cost of reduced ID accuracy; (3) a strong relationship between shift gap (difference between ID and OOD performance) and shifts in the label distribution.
  The benchmark data, Python package, model implementations, and more information about TableShift are available at https://github.com/mlfoundations/tableshift and https://tableshift.org .

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2023-benchmarking-distribution-shift-in-tabular-data-with]]
- PDF: `raw/papers/pdf/2023-benchmarking-distribution-shift-in-tabular-data-with.pdf`
- arXiv: <http://arxiv.org/abs/2312.07577v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-gardner-tableshift.md` before that tree was retired.*

TableShift — purpose-built tabular distribution-shift benchmark with 15 tasks.
