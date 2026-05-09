---
arxiv: '2207.08815'
authors:
- Léo Grinsztajn
- Edouard Oyallon
- Gaël Varoquaux
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2022-why-do-tree-based-models-still-outperform-deep-learning-on.md
raw_pdf: raw/papers/pdf/2022-why-do-tree-based-models-still-outperform-deep-learning-on.pdf
read: false
slug: why-do-tree-based-models-still-outperform-deep-learning-on
tags:
- tabular
- benchmark
- gradient-boosting
- decision-tree
title: Why do tree-based models still outperform deep learning on tabular data?
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2207.08815v1
venue: null
year: 2022
---

# Why do tree-based models still outperform deep learning on tabular data?

> *Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux* — arXiv 2207.08815, 2022

## Abstract

While deep learning has enabled tremendous progress on text and image datasets, its superiority on tabular data is not clear. We contribute extensive benchmarks of standard and novel deep learning methods as well as tree-based models such as XGBoost and Random Forests, across a large number of datasets and hyperparameter combinations. We define a standard set of 45 datasets from varied domains with clear characteristics of tabular data and a benchmarking methodology accounting for both fitting models and finding good hyperparameters. Results show that tree-based models remain state-of-the-art on medium-sized data ($\sim$10K samples) even without accounting for their superior speed. To understand this gap, we conduct an empirical investigation into the differing inductive biases of tree-based models and Neural Networks (NNs). This leads to a series of challenges which should guide researchers aiming to build tabular-specific NNs: 1. be robust to uninformative features, 2. preserve the orientation of the data, and 3. be able to easily learn irregular functions. To stimulate research on tabular architectures, we contribute a standard benchmark and raw data for baselines: every point of a 20 000 compute hours hyperparameter search for each learner.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2022-why-do-tree-based-models-still-outperform-deep-learning-on]]
- PDF: `raw/papers/pdf/2022-why-do-tree-based-models-still-outperform-deep-learning-on.pdf`
- arXiv: <http://arxiv.org/abs/2207.08815v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-grinsztajn-tree-outperform.md` before that tree was retired.*

## Core claim

On **medium-sized** (~10K-row) tabular datasets with heterogeneous columns, tree-based ensembles (XGBoost, GBT/HistGBT, Random Forest) remain state-of-the-art **under fair hyperparameter-search budgets** and beat MLP/ResNet/FT-Transformer/SAINT both in raw accuracy and per unit of compute. The performance gap traces to three structural inductive-bias mismatches between MLP-family models and tabular target functions, each diagnosed by a controlled-transformation experiment that narrows or widens the gap in the direction the bias predicts.

## Benchmark setup

- **45 datasets** drawn from OpenML, organised into four benchmark splits:
  - 15 numerical-only classification (electricity, covertype, pol, MagicTelescope, jannis, MiniBooNE, Higgs, california, ...)
  - ~20 numerical-only regression (cpu_act, elevators, houses, diamonds, year, nyc-taxi-green-dec-2016, ...)
  - 7 mixed-type classification (electricity, eye_movements, KDDCup09_upselling, road-safety, compass, ...)
  - ~14 mixed-type regression
  - The 45-total reflects deduplication across splits — several datasets appear in multiple split-pairs after the per-split feature filtering. Full dataset list in Appendix A.1 of the paper.
- **Selection criteria** (Appendix §3.1): real-world (not artificial), i.i.d. (not stream/time-series), heterogeneous columns (excludes pixel/sensor data), $d/n < 1/10$, $n \ge 3000$, $d \ge 4$, **not-too-easy** (logistic-regression baseline must trail a default ResNet *and* a default HistGBT by ≥5%), **not deterministic** (excludes poker/chess game datasets).
- **Preprocessing** (§3.2, §3.5): training set truncated to ~10,000 samples per dataset for the medium-data regime; missing data dropped by row+column; classes binarised to two-most-numerous balanced; categorical features capped at 20 cardinality, numerical at ≥10 unique values; for NN training, features are **Gaussianised via QuantileTransformer**, log-transform applied to heavy-tailed regression targets.
- **HPO budget** (§3.3): ~400 random-search iterations per (method, dataset). For each $n \in \{1, ..., 400\}$, the best-on-validation hyperparameter set across 15 random shuffles of the search trajectory is bootstrap-evaluated on test. Total compute: **~20,000 GPU/CPU-hours per learner.**
- **Aggregation across datasets:** **ADTM** (Average Distance To the Minimum) — accuracies are affinely renormalised between the 10%-quantile (classification) / 50%-quantile (regression) test-error model and the top-performing model, then averaged.
- **Methods compared.**
  - Tree family: Random Forest, GradientBoostingTrees (HistGradientBoosting when categorical), XGBoost.
  - Deep family: MLP (Gorishniy 2021 spec, with `ReduceOnPlateau` LR scheduler), ResNet (MLP + skip + BN/LN + dropout), FT-Transformer (Gorishniy 2021 — transformer over feature-tokenised numericals + categoricals), SAINT (transformer + cross-row attention + contrastive SSL pretraining).

Tree-based models remain Pareto-best across all random-search budgets and across both numerical-only and mixed-type splits — even when accounting only for accuracy and ignoring the trees' substantial speed advantage per random-search iteration.

## The three challenges (paper's own framing)

The paper isolates three properties of tabular target functions that MLP-family models don't natively respect. Each is supported by a controlled-transformation experiment that narrows or widens the gap.

### 1. Be robust to uninformative features (§5.3, Finding 2)

- **Setup A:** remove columns in increasing order of Random Forest feature-importance.
- **Setup B:** *add* uncorrelated Gaussian-noise columns.
- **Observation A:** removing uninformative features *reduces* the MLP-vs-tree gap.
- **Observation B:** adding uninformative features *widens* it.
- **Per-method ordering** (most degradation → least): MLP > ResNet > FT-Transformer ≈ XGBoost. The smoother the architecture, the worse it copes with uninformative features.
- **Why this happens (paper §5.3):** "the test accuracy of a GBT trained on the removed features [...] is very low up to 20% of features removed", confirming that real tabular tables contain many genuinely uninformative — not just redundant — columns, and the MLP smooth-function prior cannot drive their weights cleanly to zero.

### 2. Preserve the orientation of the data — rotational non-invariance (§5.4, Finding 3)

- **Setup:** apply random orthogonal rotations to the feature space; retrain everything on the rotated data. Repeat after dropping the 50% least-important features (so rotations don't pull noise into the informative subspace).
- **Observation:** under rotation, *"NNs are now above tree-based models and ResNets above FT Transformers"* (§5.4). Rotation is *detrimental* on the original axis-aligned features (trees lose what they were exploiting) and *helpful* for rotationally invariant learners (the rotation gives them no new information but levels the playing field).
- **Direct empirical confirmation** of [Ng's 2004 sample-complexity bound](2004-ng-feature-selection-l1.md). The paper's restatement in §5.4 (verbatim, including "rotationallly" typo): *"any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features."* And the intuition: *"to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones."*
- **Implication for design:** *"the sheer presence of an embedding which breaks the invariance is a key part of [the embedding-based] improvements"* (§5.4) — both Gorishniy 2022 numerical embeddings and SAINT/FT-Transformer per-feature tokenisation break rotation invariance, which is hypothesised by the authors to be why both work.

### 3. Be able to easily learn irregular functions (§5.2, Finding 1)

- **Setup:** smooth the regression target on the training set with a Gaussian kernel of varying length-scale before training (covariance = data covariance times length-scale²); retrain everything.
- **Observation:** at small length-scales, target smoothing *markedly decreases* tree-based model accuracy and *barely affects* NN accuracy. Translation: NNs already produce smooth surrogates so the smoothing is a no-op for them; trees were exploiting the irregularity and lose that advantage when the target is smoothed.
- **Why this matters:** "the target functions in our datasets are not smooth, and [...] NNs struggle to fit these irregular functions compared to tree-based models" — consistent with Rahaman et al.'s spectral-bias rate ([[2019-rahaman-spectral-bias]]) but established here directly via the smoothing-experiment proxy. Trees' piecewise-constant fits don't have this bias.
- The authors note this *could also* explain Neural-GAM's ExU activations and Gorishniy's periodic embeddings: both inject high-frequency basis on the input side and let the smoothness prior bypass the spectral bottleneck.

## Important constraints on what the paper proves

- **No quantitative decomposition.** The paper does not break down what fraction of the overall MLP-vs-GBDT gap each of the three challenges accounts for. Each experiment establishes that the gap *moves in the predicted direction* under the controlled transformation; combining the three into "X% from rotation, Y% from smoothness" is over-reading.
- **Medium-data regime only.** Results are for ~10K-row training sets. Very-small ($n < 3000$) and very-large ($n \gg 50K$) regimes are explicitly out of scope — the paper does not refute the possibility that broader pretraining or much larger datasets could close the gap.
- **Heterogeneous-column selection bias.** "Heterogeneous columns" is one of the inclusion criteria, so high-dimensional dense-categorical CTR-style problems are *excluded by construction*. The findings are about the domain where each column carries individual semantic content.
- **Categorical handling is not the main story.** The paper explicitly checks: numerical-only and mixed-type splits show similar tree-vs-NN gaps (§4.2). The categorical-encoding bottleneck often blamed for NN failure on tabular data is not what's driving this gap.

## Why it matters for §2.1–§2.3 of Chapters 2 and 3

Grinsztajn et al. is the **empirical spine** of the diagnostic chapter (Chapter 2):
- Their **rotation experiment** (§5.4) is what §2.1 cites as the empirical confirmation that real tabular features are pre-aligned to semantic axes — under rotation, NNs catch up to trees, exactly the asymmetry Ng's 2004 bound predicts.
- Their **uninformative-features experiment** (§5.3) is what §2.3 cites as the empirical confirmation that the MLP smooth-function prior cannot cleanly suppress noise columns and that attention's per-instance feature selection can.
- Their **smoothing experiment** (§5.2) is what §2.2 cites as the empirical confirmation that real tabular targets are not smooth — and that the gap moves in the direction Rahaman/Basri's spectral-bias rate predicts.

Together with Ng (rotational-invariance theorem), Rahaman (spectral-bias mechanism), Cherepanova (sharpened irrelevant-feature stress test on 12 datasets), and the gap-closing follow-ups (Gorishniy numerical embeddings, RealMLP, TabR), this paper supplies the controlled-experiment evidence that the gap is **inductive-bias rather than tuning**.

## Useful for downstream sections

- **§D2 (attention) — borrowing primitives.** Cherepanova's later result (FT-Transformer ≈ XGBoost robustness to noise features) is the natural pair to Grinsztajn's per-method ordering: the same Pareto frontier shows up in two independent benchmarks.
- **§D3 (input embeddings).** The paper's hypothesis that "the sheer presence of an embedding which breaks the invariance is a key part of these improvements" directly motivates Gorishniy 2022 numerical embeddings and the wider input-representation literature.
- **§D4 (training discipline).** The fair-protocol HPO methodology (~400 random-search iterations × 15 shuffles, 20K compute hours per learner) is the methodological precedent that RealMLP / TabReD / TabArena later built on.
