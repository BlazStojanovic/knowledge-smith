---
arxiv: '2203.05556'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2022-on-embeddings-for-numerical-features-in-tabular-deep.md
raw_pdf: raw/papers/pdf/2022-on-embeddings-for-numerical-features-in-tabular-deep.pdf
read: false
slug: on-embeddings-for-numerical-features-in-tabular-deep
tags:
- tabular
- feature-encoding
- transformer
- gradient-boosting
title: On Embeddings for Numerical Features in Tabular Deep Learning
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2203.05556v4
venue: null
year: 2022
---

# On Embeddings for Numerical Features in Tabular Deep Learning

> *Yury Gorishniy, Ivan Rubachev, Artem Babenko* — arXiv 2203.05556, 2022

## Abstract

Recently, Transformer-like deep architectures have shown strong performance on tabular data problems. Unlike traditional models, e.g., MLP, these architectures map scalar values of numerical features to high-dimensional embeddings before mixing them in the main backbone. In this work, we argue that embeddings for numerical features are an underexplored degree of freedom in tabular DL, which allows constructing more powerful DL models and competing with GBDT on some traditionally GBDT-friendly benchmarks. We start by describing two conceptually different approaches to building embedding modules: the first one is based on a piecewise linear encoding of scalar values, and the second one utilizes periodic activations. Then, we empirically demonstrate that these two approaches can lead to significant performance boosts compared to the embeddings based on conventional blocks such as linear layers and ReLU activations. Importantly, we also show that embedding numerical features is beneficial for many backbones, not only for Transformers. Specifically, after proper embeddings, simple MLP-like models can perform on par with the attention-based architectures. Overall, we highlight embeddings for numerical features as an important design aspect with good potential for further improvements in tabular DL.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2022-on-embeddings-for-numerical-features-in-tabular-deep]]
- PDF: [[raw/papers/pdf/2022-on-embeddings-for-numerical-features-in-tabular-deep.pdf]]
- arXiv: <http://arxiv.org/abs/2203.05556v4>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-gorishniy-numerical-embeddings.md` before that tree was retired.*

## Core claim

Replacing raw scalar numerical features with learned per-feature embeddings — periodic or piecewise-linear — closes most of the gap between MLPs and FT-Transformer-style attention models on standard tabular benchmarks. The lever is at the input representation, not in the architecture: a vanilla MLP with proper embeddings matches a tuned transformer on numerical-feature datasets.

## The two embedding forms

**Periodic embedding (per-feature).** For numerical input $x$:

$$f_i(x) = \text{concat}\big[\sin(\mathbf{v}), \cos(\mathbf{v})\big], \quad \mathbf{v} = [2\pi c_1 x, \ldots, 2\pi c_k x]^\top$$

with **trained** coefficients $c_i \sim \mathcal{N}(0, \sigma^2)$ at initialisation. Hyperparameters $\sigma$ and $k$ are tuned per dataset. Note: per-feature, no cross-feature mixing during embedding.

**Piecewise Linear Encoding (PLE).** Bin the feature into $T$ bins $\{b_0, \ldots, b_T\}$ (quantile-based or decision-tree-based boundaries). Each bin gives one coordinate of the embedding:

$$e_t(x) = \begin{cases} 0 & x < b_{t-1} \\ 1 & x \ge b_t \\ (x - b_{t-1}) / (b_t - b_{t-1}) & \text{otherwise} \end{cases}$$

PLE explicitly bakes a piecewise-constant / piecewise-linear basis into the input, which is exactly the basis trees use natively.

## Benchmark setup

- 11 public tabular datasets (California Housing, Adult, Higgs Small, etc.).
- Methods compared: vanilla MLP, ResNet, FT-Transformer, plus CatBoost and XGBoost baselines.
- HPO: TPE (Optuna) Bayesian search, 100 iterations per (method, dataset), 15 random seeds, 3-model ensemble at evaluation.

## Headline numbers

California Housing (RMSE, lower better):
- Vanilla MLP: 0.495
- MLP-PLR (Periodic + Linear + ReLU embedding head): **0.453**
- FT-Transformer-L: 0.465

The pattern repeats across the eleven datasets: with proper embeddings, the simple MLP matches or beats the attention-based architectures. Direct quote: *"after proper embeddings, simple MLP-like models can perform on par with the attention-based architectures."*

## Mechanism — what the authors actually claim

The authors are explicit that the mechanism is **not fully nailed down**:

> "It is still to be explained how exactly the discussed embedding modules help optimization on the fundamental level."

What they do say:

- The inspiration is [@tancik2020fourier]'s Fourier-features result on coordinate networks. *"Changing the input space alleviates [the optimization] issue."*
- Their adaptation differs from Tancik in two ways: **per-feature embedding (no cross-feature mixing)**, and **trained pre-activation coefficients** rather than fixed random ones. They tested the original Tancik formulation on tables (Appendix D.2): it performs *worse* than vanilla MLP. Per-feature is essential; cross-feature Fourier mixing breaks tabular structure.
- Both periodic and PLE work; the paper does not single out one mechanism. PLE is closer in spirit to tree splits; periodic is closer to spectral-bias remediation. The improvement comes from injecting *some* basis, with the right per-feature locality.

## Why it matters for §2.4

This is the gap-closing fix that diagnoses the §2.2 spectral-bias story most cleanly. The reading the post should give:

- **Conservative version** (what Gorishniy explicitly supports): proper input embeddings close most of the MLP↔Transformer gap on numerical-feature tables; the mechanism is empirical, the analogy to Tancik is the authors' own framing.
- **Stronger version** (consistent with the paper but not directly proven): the gap was driven by spectral bias on raw scalar inputs, and the embeddings fix it by giving the network a high-frequency basis on the input side, exactly as Tancik's NTK argument predicts.

For §2.4 the conservative version is what we should land on — over-claiming "spectral bias *was* the bottleneck" goes past what the paper proves. The empirical pattern (proper embeddings ≈ proper architecture) is what carries the diagnostic argument.
