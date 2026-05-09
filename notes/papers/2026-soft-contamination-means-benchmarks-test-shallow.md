---
arxiv: '2602.12413'
authors:
- Ari Spiesberger
- Juan J. Vazquez
- Nicky Pochinkov
- Tomáš Gavenčiak
- Peli Grietzer
- Gavin Leech
- Nandi Schoots
created: '2026-05-09'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2026-soft-contamination-means-benchmarks-test-shallow.md
raw_pdf: raw/papers/pdf/2026-soft-contamination-means-benchmarks-test-shallow.pdf
read: false
slug: soft-contamination-means-benchmarks-test-shallow
tags:
- benchmark
- generalization
- evaluation
title: Soft Contamination Means Benchmarks Test Shallow Generalization
type: note
updated: '2026-05-09'
url: https://arxiv.org/abs/2602.12413
venue: null
year: 2026
---

# Soft Contamination Means Benchmarks Test Shallow Generalization

> *Ari Spiesberger, Juan J. Vazquez, Nicky Pochinkov…* — arXiv 2602.12413, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

If LLM training data is polluted with benchmark test data, then benchmark performance gives biased estimates of out-of-distribution (OOD) generalization. Typical decontamination filters use n-gram matching which fail to detect semantic duplicates: sentences with equivalent (or near-equivalent) content that are not close in string space. We study this soft contamination of training data by semantic duplicates. Among other experiments, we embed the Olmo3 training corpus and find that: 1) contamination remains widespread, e.g. we find semantic duplicates for 78% of CodeForces and exact duplicates for 50% of ZebraLogic problems; 2) including semantic duplicates of benchmark data in training does improve benchmark performance; and 3) when finetuning on duplicates of benchmark datapoints, performance also improves on truly-held-out datapoints from the same benchmark. We argue that recent benchmark gains are thus confounded: the prevalence of soft contamination means gains reflect both genuine capability improvements and the accumulation of test data and effective test data in growing training corpora.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-soft-contamination-means-benchmarks-test-shallow]]
- PDF: [[raw/papers/pdf/2026-soft-contamination-means-benchmarks-test-shallow.pdf]]
- arXiv: <https://arxiv.org/abs/2602.12413>
