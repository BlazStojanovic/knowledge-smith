---
arxiv: '2407.03257'
authors:
- Han-Jia Ye
- Huai-Hong Yin
- De-Chuan Zhan
- Wei-Lun Chao
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2407.03257.md
raw_pdf: raw/papers/2407.03257.pdf
read: false
slug: revisiting-nearest-neighbor-for-tabular-data-a-deep-tabular
tags:
- tabular
- contrastive
- metric-learning
- benchmark
title: 'Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two
  Decades Later'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2407.03257v2
venue: null
year: 2024
---

# Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later

> *Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan…* — arXiv 2407.03257, 2024

## Abstract

The widespread enthusiasm for deep learning has recently expanded into the domain of tabular data. Recognizing that the advancement in deep tabular methods is often inspired by classical methods, e.g., integration of nearest neighbors into neural networks, we investigate whether these classical methods can be revitalized with modern techniques. We revisit a differentiable version of $K$-nearest neighbors (KNN) -- Neighbourhood Components Analysis (NCA) -- originally designed to learn a linear projection to capture semantic similarities between instances, and seek to gradually add modern deep learning techniques on top. Surprisingly, our implementation of NCA using SGD and without dimensionality reduction already achieves decent performance on tabular data, in contrast to the results of using existing toolboxes like scikit-learn. Further equipping NCA with deep representations and additional training stochasticity significantly enhances its capability, being on par with the leading tree-based method CatBoost and outperforming existing deep tabular models in both classification and regression tasks on 300 datasets. We conclude our paper by analyzing the factors behind these improvements, including loss functions, prediction strategies, and deep architectures. The code is available at https://github.com/qile2000/LAMDA-TALENT.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2407.03257]]
- PDF: `raw/papers/2407.03257.pdf`
- arXiv: <http://arxiv.org/abs/2407.03257v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2025-ye-modernnca.md` before that tree was retired.*

> **2026-05-05 corrections.** Previous stub had wrong arXiv ID (2410.02010, an unrelated medical-imaging paper) and wrong title ("Modern Neighborhood Components Analysis for Tabular Data"). Authors were also incomplete ("Han-Jia Ye, Huai-Hong Zhu, De-Chuan Zhan"). All corrected against the bib entry and verified arXiv abstract.

- **ArXiv:** 2407.03257
- **Authors:** Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan, Wei-Lun Chao
- **Year:** 2025
- **Venue:** ICLR 2025 (introduces ModernNCA)
- **Raw:** [[raw/papers/2025-ye-modernnca.pdf]]

## Core claim

A modernised version of *Neighbourhood Components Analysis* (NCA), a 2004 differentiable nearest-neighbour method by Goldberger et al., outperforms most existing deep tabular models on classification and regression across a 300-dataset benchmark and matches CatBoost on average. The paper's argument: classical instance-based methods, when retrofitted with deep representations and modern training stochasticity, are stronger than the architectural-zoo proliferation of 2017–2024 has acknowledged.

## Architecture

ModernNCA layers four ingredients on the original NCA backbone:

1. **NCA core.** A learned embedding $\phi: x \mapsto h$ such that the soft-nearest-neighbour classifier in $\phi$-space minimises the leave-one-out classification loss. The objective is differentiable: each training point's prediction is a soft-attention-weighted average over all *other* training points' labels, weighted by negative squared distance in $\phi$-space.

2. **Deep encoder.** $\phi$ is a multi-layer MLP (rather than the 2004-paper's linear projection), giving non-linear metric learning. Combined with proper numerical input encodings per [@gorishniy2022embeddings], the encoder has the expressive capacity NCA needed.

3. **Stochastic Neighbourhood Sampling (SNS).** Rather than computing soft-attention over the entire training set per query (which is $O(N^2)$ in dataset size), SNS samples a stochastic subset of neighbours per training step, making large-data training tractable. The stochasticity also acts as implicit regularisation.

4. **Regression extension.** The original NCA was classification-only; ModernNCA extends the soft-neighbour weighting to a regression objective via target averaging.

The architectural family closely resembles TabR's [@gorishniy2024tabr] retrieval-augmented MLP, but with the simpler design of "make NCA work properly with modern DL" rather than "build a hybrid with a separate retrieval module."

## Key result

- **300 datasets** evaluated (large-scale benchmark).
- ModernNCA is on par with **CatBoost** on average across classification and regression.
- ModernNCA outperforms existing deep tabular models on most datasets.
- Reduced model size and training time versus competing deep tabular methods.

## Why it matters for §2.4.5 (sidestepping the prior)

ModernNCA pairs with TabR [@gorishniy2024tabr] as the §2.4.5 instance-based-conditioning evidence:

- **TabR (2023/2024):** retrieval-augmented MLP with attention-based neighbour aggregation. Best average DL model on its benchmark, beats GBDT on Grinsztajn's [@grinsztajn2022tree] suite.
- **ModernNCA (2024/2025):** classical NCA with deep encoder + SNS. Matches CatBoost on 300 datasets, outperforms most deep tabular models.

Both methods *sidestep* the §2.1/§2.2/§2.3 priors via instance-based conditioning. Together they make the strongest empirical case in §2.4.5: when you change the inference regime from "evaluate a globally-smooth function" to "look up similar training instances and aggregate their labels," the inductive-bias gap to trees largely evaporates.

The diagnostic reading: ModernNCA's strength on a large benchmark suggests the result is robust, not benchmark-specific. And the historical framing — a 2004 method matches 2025 SOTA when properly adapted — supports the §2.4.5 conclusion that the "deep representation" was the missing ingredient, not the "non-trees architecture."

## Caveats

- "On par with CatBoost on average" — the paper does not claim universal dominance over CatBoost or other strong GBDTs.
- 300 datasets is a large suite, but particular regimes (very-low-data, very-high-cardinality categoricals, strong temporal structure) may shift rankings.
- The paper's title in some versions appears as "Modern Neighborhood Components Analysis" — the canonical bib entry title is "Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later" with a "Introduces ModernNCA" note.
