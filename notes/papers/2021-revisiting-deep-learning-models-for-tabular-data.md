---
arxiv: '2106.11959'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Valentin Khrulkov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2021-revisiting-deep-learning-models-for-tabular-data.md
raw_pdf: raw/papers/pdf/2021-revisiting-deep-learning-models-for-tabular-data.pdf
read: false
slug: revisiting-deep-learning-models-for-tabular-data
tags:
- tabular
- transformer
- benchmark
- gradient-boosting
title: Revisiting Deep Learning Models for Tabular Data
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2106.11959v5
venue: null
year: 2021
---

# Revisiting Deep Learning Models for Tabular Data

> *Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov…* — arXiv 2106.11959, 2021

## Abstract

The existing literature on deep learning for tabular data proposes a wide range of novel architectures and reports competitive results on various datasets. However, the proposed models are usually not properly compared to each other and existing works often use different benchmarks and experiment protocols. As a result, it is unclear for both researchers and practitioners what models perform best. Additionally, the field still lacks effective baselines, that is, the easy-to-use models that provide competitive performance across different problems.
  In this work, we perform an overview of the main families of DL architectures for tabular data and raise the bar of baselines in tabular DL by identifying two simple and powerful deep architectures. The first one is a ResNet-like architecture which turns out to be a strong baseline that is often missing in prior works. The second model is our simple adaptation of the Transformer architecture for tabular data, which outperforms other solutions on most tasks. Both models are compared to many existing architectures on a diverse set of tasks under the same training and tuning protocols. We also compare the best DL models with Gradient Boosted Decision Trees and conclude that there is still no universally superior solution.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-revisiting-deep-learning-models-for-tabular-data]]
- PDF: `raw/papers/pdf/2021-revisiting-deep-learning-models-for-tabular-data.pdf`
- arXiv: <http://arxiv.org/abs/2106.11959v5>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-gorishniy-fttransformer.md` before that tree was retired.*

## Core claim

Most prior deep-tabular results compared specialised architectures (TabNet, NODE, TabTransformer, etc.) against weak MLP baselines under inconsistent protocols. Under fair HPO and a unified benchmark, two conceptually simple designs — a *ResNet-style MLP* and an *FT-Transformer* (Feature Tokenizer + standard transformer) — match or beat all the prior specialised architectures, while still trailing well-tuned GBDTs on a non-trivial fraction of tasks. This paper re-baselined the field.

## Architecture

The paper introduces two baselines:

**ResNet-tabular.** A plain MLP with skip connections (BatchNorm + ReLU + Linear residual blocks). Built to be the "missing strong MLP baseline" — most prior tabular-DL papers compared against vanilla MLPs without modern training tricks (residuals, batchnorm, careful HPO).

**FT-Transformer.** A *Feature Tokenizer* embeds every input feature (both numerical and categorical) into a $d$-dimensional vector — numerical features via a per-feature linear projection $z_i = b_i + W_i x_i$, categoricals via a lookup table — plus a learnable `[CLS]` token. The resulting sequence of $n+1$ tokens passes through a standard pre-LN transformer encoder. The `[CLS]`-token output is fed to a prediction head.

The Feature Tokenizer is the load-bearing change versus TabTransformer [@huang2020tabtransformer]: numerical features are *included* in the attention block instead of bypassed. This unifies the architecture (one mechanism for all feature types) and lifts performance on numerical-feature-heavy tasks.

## Key result

- Unified benchmark suite: 11 public datasets (California Housing, Adult, Higgs Small, MS LTR, Year, etc.).
- HPO budget matched across methods (TPE Bayesian search, ~100 iterations per (method, dataset)), 15 random seeds, 3-model ensembles at evaluation.
- ResNet-tabular and FT-Transformer match or beat all evaluated specialised architectures (TabNet, NODE, AutoInt, SNN, MLP-PLR, etc.) under this protocol.
- GBDT (XGBoost, CatBoost) remains competitive — the paper concludes "there is still no universally superior solution."

The headline is methodological as much as architectural: the message is that previously claimed deep-tabular wins were partly tuning artefacts, and the right baselines are a strong MLP and a transformer-with-feature-tokenisation.

## Why it matters for §2.4.3 (tests of irrelevant-feature robustness)

FT-Transformer is the strongest fair-comparison anchor of the 2020–2022 transformer-on-tables wave. For §2.4.3:

- **What it tries:** unified attention over *all* features (numerical + categorical) via Feature Tokenizer, with attention as soft per-instance feature interaction.
- **What it pulls:** the §2.3 irrelevant-features lever, but for numerical features too. [@grinsztajn2022tree] later finds FT-Transformer is markedly more robust to noise-column injection than ResNet-tabular and plain MLPs (matching XGBoost more closely under feature-noise stress) — a controlled-experiment confirmation that attention-on-tokenised-numerical-features partially closes the §2.3 gap.

For §2.4.3 the FT-Transformer / ResNet-tabular pair is the *right* benchmark anchor: prior tabular-DL papers that beat "MLP" were often beating an under-tuned MLP. The Gorishniy et al. baselines are the strong ones; results above them are the real wins.

## Caveats

- The "no universally superior solution" framing means GBDT often still wins; FT-Transformer is competitive, not dominant.
- 11 datasets is moderate; later benchmark work (TabZilla [@mcelfresh2023tabzilla], TabReD [@rubachev2025tabred], TabArena [@erickson2025tabarena]) gives larger samples that mostly preserve the qualitative ranking but shift specific positions.
- The ResNet-tabular baseline is plain — proper *numerical embeddings* per [@gorishniy2022embeddings] add another step on top, often closing the FT-Transformer gap with a vanilla MLP.
