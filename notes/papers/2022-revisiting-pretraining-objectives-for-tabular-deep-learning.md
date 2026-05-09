---
arxiv: '2207.03208'
authors:
- Ivan Rubachev
- Artem Alekberov
- Yury Gorishniy
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2207.03208.md
raw_pdf: raw/papers/2207.03208.pdf
read: false
slug: revisiting-pretraining-objectives-for-tabular-deep-learning
tags:
- tabular
- pretraining
- self-supervised
- gradient-boosting
title: Revisiting Pretraining Objectives for Tabular Deep Learning
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2207.03208v2
venue: null
year: 2022
---

# Revisiting Pretraining Objectives for Tabular Deep Learning

> *Ivan Rubachev, Artem Alekberov, Yury Gorishniy…* — arXiv 2207.03208, 2022

## Abstract

Recent deep learning models for tabular data currently compete with the traditional ML models based on decision trees (GBDT). Unlike GBDT, deep models can additionally benefit from pretraining, which is a workhorse of DL for vision and NLP. For tabular problems, several pretraining methods were proposed, but it is not entirely clear if pretraining provides consistent noticeable improvements and what method should be used, since the methods are often not compared to each other or comparison is limited to the simplest MLP architectures.
  In this work, we aim to identify the best practices to pretrain tabular DL models that can be universally applied to different datasets and architectures. Among our findings, we show that using the object target labels during the pretraining stage is beneficial for the downstream performance and advocate several target-aware pretraining objectives. Overall, our experiments demonstrate that properly performed pretraining significantly increases the performance of tabular DL models, which often leads to their superiority over GBDTs.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2207.03208]]
- PDF: `raw/papers/2207.03208.pdf`
- arXiv: <http://arxiv.org/abs/2207.03208v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-rubachev-revisiting-pretraining.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub had wrong arXiv ID 2209.07286; the correct ID per the bib entry and verified abstract is **2207.03208**.

- **ArXiv:** 2207.03208
- **Authors:** Ivan Rubachev, Artem Alekberov, Yury Gorishniy, Artem Babenko
- **Year:** 2022
- **Venue:** arXiv preprint (Yandex)
- **Raw:** [[raw/papers/2022-rubachev-revisiting-pretraining.pdf]]

## Core claim

The 2020–2022 wave of tabular SSL methods (VIME [@yoon2020vime], SCARF [@bahri2022scarf], SubTab [@ucar2021subtab], TabNet's [@arik2021tabnet] SSL stage, etc.) had been compared inconsistently and tuned variably; under a fair, unified protocol, *most published gains do not survive*. The paper benchmarks tabular pretraining objectives across architectures and datasets, identifies which objectives are robust, and argues for **target-aware pretraining** — using the supervised target during the pretraining stage — as the consistently helpful objective.

## Method

The paper:

1. **Benchmarks** a broad set of tabular SSL methods (VIME, SCARF, SubTab, MLM, contrastive variants) under matched HPO budgets and matched architectures.
2. **Decouples** the pretraining-objective choice from the architectural-backbone choice (MLP, ResNet, FT-Transformer), running each (objective × architecture × dataset) cell.
3. **Introduces target-aware variants** that use the supervised target during pretraining (e.g., predict the target from corrupted features, or use the target as an auxiliary signal in the contrastive loss).

## Key result

- Most prior tabular-SSL gains are **fragile to tuning** — under fair HPO they shrink or disappear.
- **Target-aware pretraining objectives** consistently improve downstream performance and frequently push tabular DL models above tuned GBDTs on the benchmarked datasets.
- **The architecture-objective interaction is non-trivial** — different backbones favour different objectives; a single "best objective" claim across architectures is unsupported.

The headline conceptual contribution is the *target-aware* framing: if you have access to the labels at pretraining time, use them. This contradicts the strict-SSL framing imported from vision/NLP where pretraining and downstream supervision are kept separate.

## Why it matters for §2.4.5 (single-table SSL limit)

Rubachev et al. is the **fair-protocol checkpoint** for the tabular-SSL programme. For §2.4.5 the paper provides:

1. **Empirical evidence that single-table SSL is fragile.** Many headline claims from VIME, SCARF, etc. shrink under matched HPO. The literature's average gain is real but smaller than the per-paper claims suggest.
2. **Diagnostic for the ceiling.** Single-table SSL extracts limited additional signal beyond what the supervised loss already provides — once HPO and architecture are matched, the marginal benefit of pretraining-on-the-same-table is bounded.
3. **A pointer to the next move.** Target-aware pretraining helps; cross-table pretraining (TransTab [@wang2022transtab], XTab) is implied as the natural follow-up. The Chapter 5 story (TabPFN's synthetic priors; LLM-for-tables's real-table corpus aggregation) is the *answer* to the ceiling Rubachev documents — change the pretraining corpus, not the objective.

For Chapter 3's argument the paper closes the §2.4.5 sub-section: the single-table-SSL line was a credible attempt that hit a real ceiling, and the field's response (Chapter 5) had to leave the single-table regime.

## Caveats

- arXiv preprint, not formally peer-reviewed at a major venue.
- "Most prior gains shrink" is on the paper's specific benchmark; particular methods or datasets may still show robust SSL gains.
- The paper's target-aware pretraining recommendation is not "do supervised learning twice" — it's "use the target as auxiliary supervision during pretraining," which the paper distinguishes carefully but readers can elide.
