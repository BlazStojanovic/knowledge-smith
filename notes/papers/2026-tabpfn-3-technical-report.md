---
authors:
- Prior Labs Team
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://storage.googleapis.com/prior-labs-tabpfn-public/reports/TabPFN_3_model_report.pdf
  raw: '[[raw/papers/md/2026-tabpfn-3-technical-report]]'
  source: https://priorlabs.ai/technical-reports/tabpfn-3
owner: blaz
parser: read
raw_md: raw/papers/md/2026-tabpfn-3-technical-report.md
raw_pdf: raw/papers/pdf/2026-tabpfn-3-technical-report.pdf
read: false
slug: tabpfn-3-technical-report
tags:
- type/paper
- status/stub
- tabular
- synthetic-data
- foundation-model
- benchmark
title: 'TabPFN-3: Technical Report'
type: note
updated: '2026-05-15'
venue: Technical Report
year: 2026
---

# TabPFN-3: Technical Report

> *Prior Labs Team* — Technical Report, 2026

## TL;DR

Tabular foundation model pretrained exclusively on synthetic data drawn from a prior. TabPFN-3 scales single-forward-pass tabular prediction to 1M training rows and 200 features and adds a test-time "thinking" mode (TabPFN-3-Plus). SOTA on the TabArena benchmark — a forward pass beats tuned and ensembled baselines, and thinking adds ~200+ Elo, pareto-dominating an 8-hour-tuned gradient-boosted-tree ensemble at ~10× lower runtime. Extends beyond classification/regression to time-series, relational, and tabular-text data. Released May 2026 under the TABPFN-3.0 License v1.0.

## Abstract

Tabular data underpins most high-value prediction problems in science and industry, and TabPFN has driven the foundation model revolution for this modality. Designed with feedback from users, TabPFN-3 builds on this foundation to scale state-of-the-art performance to datasets with 1M training rows and substantially reduce training and inference time. Pretrained exclusively on synthetic data from our prior, TabPFN-3 dramatically pushes the frontier of tabular prediction and brings substantial gains on time series, relational, and tabular-text data. On the standard tabular benchmark TabArena, a forward pass of TabPFN-3 outperforms all other models, including tuned and ensembled baselines, by a significant margin. TabPFN-3 introduces test-time compute scaling to tabular foundation models via TabPFN-3-Plus (Thinking). The model is released under the TABPFN-3.0 License v1.0 (permissive for research and internal evaluation); TabPFN-3-Plus is available via API and enterprise licensing.

## Notes

(your synthesis — anything beyond the abstract belongs here. Architecture, many-class decoder, synthetic prior, and thinking-mode details are in §2 of the report.)

## Source

- PDF: [[raw/papers/pdf/2026-tabpfn-3-technical-report.pdf]]
- Source: <https://priorlabs.ai/technical-reports/tabpfn-3>
- Docs: <https://docs.priorlabs.ai>
