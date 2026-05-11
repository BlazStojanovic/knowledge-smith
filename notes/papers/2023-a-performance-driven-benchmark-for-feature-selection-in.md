---
arxiv: '2311.05877'
authors:
- Valeriia Cherepanova
- Roman Levin
- Gowthami Somepalli
- Jonas Geiping
- C. Bayan Bruss
- Andrew Gordon Wilson
- Tom Goldstein
- Micah Goldblum
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2311.05877
  raw: '[[raw/papers/md/2023-a-performance-driven-benchmark-for-feature-selection-in]]'
  source: http://arxiv.org/abs/2311.05877v1
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-a-performance-driven-benchmark-for-feature-selection-in.md
raw_pdf: raw/papers/pdf/2023-a-performance-driven-benchmark-for-feature-selection-in.pdf
read: false
slug: a-performance-driven-benchmark-for-feature-selection-in
tags:
- type/paper
- tabular
- benchmark
- feature-selection
- transformer
- status/stub
title: A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning
type: note
updated: '2026-05-09'
venue: null
year: 2023
---

# A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning

> *Valeriia Cherepanova, Roman Levin, Gowthami Somepalli…* — arXiv 2311.05877, 2023

## Abstract

Academic tabular benchmarks often contain small sets of curated features. In contrast, data scientists typically collect as many features as possible into their datasets, and even engineer new features from existing ones. To prevent overfitting in subsequent downstream modeling, practitioners commonly use automated feature selection methods that identify a reduced subset of informative features. Existing benchmarks for tabular feature selection consider classical downstream models, toy synthetic datasets, or do not evaluate feature selectors on the basis of downstream performance. Motivated by the increasing popularity of tabular deep learning, we construct a challenging feature selection benchmark evaluated on downstream neural networks including transformers, using real datasets and multiple methods for generating extraneous features. We also propose an input-gradient-based analogue of Lasso for neural networks that outperforms classical feature selection methods on challenging problems such as selecting from corrupted or second-order features.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2023-a-performance-driven-benchmark-for-feature-selection-in]]
- PDF: [[raw/papers/pdf/2023-a-performance-driven-benchmark-for-feature-selection-in.pdf]]
- arXiv: <http://arxiv.org/abs/2311.05877v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-cherepanova-feature-selection.md` before that tree was retired.*

## Core claim

Deep tabular models differ sharply in their robustness to irrelevant input features. **MLPs degrade substantially** when noise / redundant columns are added; **FT-Transformers are roughly as robust as XGBoost**. The gap between deep tabular and GBDT on irrelevant-features-laden data is therefore an *MLP-architecture* problem, not a generic deep-tabular problem.

## Benchmark setup

- **12 real-world datasets** (8 classification + 4 regression).
- **Models:** MLP, FT-Transformer, XGBoost (the GBDT reference).
- **Feature selection algorithms tested (10):** Lasso, Random Forest importance, XGBoost importance, Deep Lasso, and others.
- **Three modes of irrelevant-feature corruption:**
  1. *Random features.* Append Gaussian-noise columns.
  2. *Corrupted features.* Replace with $x_c = 0.5 x + 0.5 \varepsilon$ — partial information loss.
  3. *Second-order features.* Products of randomly chosen original-feature pairs (high-correlation distractors).
- **Corruption ratios:** 50% and 75% of total features.

## Headline finding

Direct quote: *"the FT-Transformer model is roughly as robust to noisy features as the XGBoost model"* — but **MLPs degrade markedly** under all three corruption modes, with the gap widening as the corruption ratio rises.

This is a refinement of [@grinsztajn2022tree]'s "robust to uninformative features" challenge: Grinsztajn showed MLPs are worse than ResNets are worse than FT-Transformers; Cherepanova confirms the FT-Transformer's robustness reaches GBDT levels under their irrelevant-feature stress tests.

## Why it matters for §2.3 — and a correction to the post outline

The §2 outline currently says: *"Cherepanova et al. construct a performance-driven benchmark for feature selection in tabular deep learning and show the gap to GBDTs is still substantial as of 2023."* That over-collapses the result. The accurate version:

- The MLP-vs-GBDT gap on irrelevant-features data is substantial as of 2023.
- The FT-Transformer-vs-GBDT gap on the same data is roughly closed.
- Whether explicit feature selection (Deep Lasso etc.) helps MLPs match GBDTs is what the benchmark is built to measure.

This refines §2.3: the irrelevant-features failure mode is specifically *the MLP's smooth-function prior* failing to zero out noise columns, not a uniformly architecture-independent deep-tabular problem. The architectural prior of attention (in FT-Transformer) does enough column-level filtering to nearly close the gap.

The §2.3 prose should reflect both findings: the MLP failure mode is the canonical inductive-bias gap, and the FT-Transformer's near-parity is itself a partial gap-closing result that fits the §2.4 "what closes the gap" pattern (here, attention as an implicit feature-selection prior).

## Caveats

- 12 datasets is a smaller sample than Grinsztajn's 45 — generalisation to broader tabular regimes is a follow-up question.
- The "noise feature" construction is synthetic; real enterprise tables have *informative-but-weak* columns, which is a different stress test.
