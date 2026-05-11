---
arxiv: '2407.04491'
authors:
- David Holzmüller
- Léo Grinsztajn
- Ingo Steinwart
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.04491
  raw: '[[raw/papers/md/2024-better-by-default-strong-pre-tuned-mlps-and-boosted-trees]]'
  source: http://arxiv.org/abs/2407.04491v3
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-better-by-default-strong-pre-tuned-mlps-and-boosted-trees.md
raw_pdf: raw/papers/pdf/2024-better-by-default-strong-pre-tuned-mlps-and-boosted-trees.pdf
read: false
slug: better-by-default-strong-pre-tuned-mlps-and-boosted-trees
tags:
- type/paper
- tabular
- gradient-boosting
- benchmark
- mlp
- status/stub
title: 'Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data'
type: note
updated: '2026-05-09'
venue: null
year: 2024
---

# Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data

> *David Holzmüller, Léo Grinsztajn, Ingo Steinwart* — arXiv 2407.04491, 2024

## Abstract

For classification and regression on tabular data, the dominance of gradient-boosted decision trees (GBDTs) has recently been challenged by often much slower deep learning methods with extensive hyperparameter tuning. We address this discrepancy by introducing (a) RealMLP, an improved multilayer perceptron (MLP), and (b) strong meta-tuned default parameters for GBDTs and RealMLP. We tune RealMLP and the default parameters on a meta-train benchmark with 118 datasets and compare them to hyperparameter-optimized versions on a disjoint meta-test benchmark with 90 datasets, as well as the GBDT-friendly benchmark by Grinsztajn et al. (2022). Our benchmark results on medium-to-large tabular datasets (1K--500K samples) show that RealMLP offers a favorable time-accuracy tradeoff compared to other neural baselines and is competitive with GBDTs in terms of benchmark scores. Moreover, a combination of RealMLP and GBDTs with improved default parameters can achieve excellent results without hyperparameter tuning. Finally, we demonstrate that some of RealMLP's improvements can also considerably improve the performance of TabR with default parameters.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2024-better-by-default-strong-pre-tuned-mlps-and-boosted-trees]]
- PDF: [[raw/papers/pdf/2024-better-by-default-strong-pre-tuned-mlps-and-boosted-trees.pdf]]
- arXiv: <http://arxiv.org/abs/2407.04491v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-holzmuller-realmlp.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub listed authors "David Holzmüller, Viktor Garg, Jan Hammerla" — the actual authors per the bib entry and arXiv abstract are **David Holzmüller, Léo Grinsztajn, Ingo Steinwart**. Corrected.

- **ArXiv:** 2407.04491
- **Authors:** David Holzmüller, Léo Grinsztajn, Ingo Steinwart
- **Year:** 2024
- **Venue:** NeurIPS 2024
- **Raw:** [[raw/papers/2024-holzmuller-realmlp.pdf]]

## Core claim

Two contributions, both targeting the methodological correction Kadra et al. [@kadra2021welltuned] opened. (a) *RealMLP* — an improved MLP design (small but consequential changes to normalisation, optimizer, learning-rate schedule, regularisation defaults). (b) *Meta-tuned default parameters* — for both RealMLP and tuned GBDTs (XGBoost, LightGBM, CatBoost) — found via meta-training on 118 datasets and tested on a *disjoint* meta-test benchmark of 90 datasets plus the GBDT-friendly Grinsztajn et al. [@grinsztajn2022tree] benchmark. The contribution is "good defaults" rather than another architecture.

## Method

**RealMLP architecture changes** (collectively called the "improved" recipe):
- Scaled Layer Normalization placement.
- AdamW with warmup + cosine schedule.
- Standardised per-feature numerical input encoding (quantile-bin embeddings or PLE-style, building on [@gorishniy2022embeddings]).
- Carefully tuned default dropout / weight-decay / depth / width values.

**Meta-tuning protocol.** A search over default parameters for RealMLP and for each GBDT package, where the search objective is *average-rank performance on the meta-train benchmark*. The defaults that win the meta-train are evaluated on the meta-test, with no further per-dataset tuning. This separates "good default" claims from "good with HPO" claims — the meta-test datasets are unseen at default-search time.

## Key result

- **RealMLP** is competitive with GBDTs on medium-to-large datasets (1K–500K samples) and gives a favourable time-accuracy trade-off versus other neural baselines.
- **GBDT + meta-tuned defaults** matches or beats GBDT-with-HPO on meta-test datasets, suggesting the field's common HPO defaults are sub-optimal.
- **Combined RealMLP + GBDT (defaults only, no HPO)** achieves excellent results — a strong out-of-the-box baseline.
- **Transfer to TabR.** Some of RealMLP's improvements (input encoding, optimizer schedule) also improve TabR [@gorishniy2024tabr] under default settings, suggesting the recipe is not RealMLP-specific.

## Why it matters for §2.4.4 (the methodological correction)

RealMLP extends Kadra's "the gap is a tuning gap" verdict in two ways §2.4.4 cares about:

1. **From cocktail-tuning to meta-defaults.** Kadra showed expensive joint HPO closes the gap. Holzmüller et al. show that *good defaults found by meta-learning on a held-out collection* close it without per-task HPO. The practical implication is stronger: the gap was not "GBDTs win because their HPO is cheaper" — it was "DL needed better off-the-shelf settings."
2. **The defaults transfer.** Improvements found for RealMLP help TabR. This suggests there's a stable "right way to train deep tabular models in 2024" that's mostly architecture-agnostic.

For §2.4.4 the diagnostic reading: when you control for tuning regime properly (meta-defaults, no per-task HPO), the architectural rabbit holes (TabNet, NODE, large transformers) lose, and a well-trained MLP and a well-defaulted GBDT are jointly the strong baselines.

## Caveats

- The benchmark is medium-to-large (1K–500K rows). Sub-1K and >500K regimes are out of scope and may shift rankings.
- "Meta-defaults" carries its own collinearity risks — defaults found on meta-train datasets may overfit the dataset distribution. The meta-test result mitigates but doesn't eliminate this.
- The RealMLP recipe is concrete and reproducible (see paper's appendix) but its improvements are accumulated small wins rather than one big idea. The right way to read the paper is "tuning ceiling for plain MLPs is significantly higher than the field's defaults," not "here's a new architecture."
