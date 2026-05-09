---
arxiv: '2309.17130'
authors:
- Sascha Marton
- Stefan Lüdtke
- Christian Bartelt
- Heiner Stuckenschmidt
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2023-grande-gradient-based-decision-tree-ensembles-for-tabular.md
raw_pdf: raw/papers/pdf/2023-grande-gradient-based-decision-tree-ensembles-for-tabular.pdf
read: false
slug: grande-gradient-based-decision-tree-ensembles-for-tabular
tags:
- tabular
- decision-tree
- gradient-boosting
- optimization
title: 'GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2309.17130v3
venue: null
year: 2023
---

# GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data

> *Sascha Marton, Stefan Lüdtke, Christian Bartelt…* — arXiv 2309.17130, 2023

## Abstract

Despite the success of deep learning for text and image data, tree-based ensemble models are still state-of-the-art for machine learning with heterogeneous tabular data. However, there is a significant need for tabular-specific gradient-based methods due to their high flexibility. In this paper, we propose $\text{GRANDE}$, $\text{GRA}$die$\text{N}$t-Based $\text{D}$ecision Tree $\text{E}$nsembles, a novel approach for learning hard, axis-aligned decision tree ensembles using end-to-end gradient descent. GRANDE is based on a dense representation of tree ensembles, which affords to use backpropagation with a straight-through operator to jointly optimize all model parameters. Our method combines axis-aligned splits, which is a useful inductive bias for tabular data, with the flexibility of gradient-based optimization. Furthermore, we introduce an advanced instance-wise weighting that facilitates learning representations for both, simple and complex relations, within a single model. We conducted an extensive evaluation on a predefined benchmark with 19 classification datasets and demonstrate that our method outperforms existing gradient-boosting and deep learning frameworks on most datasets. The method is available under: https://github.com/s-marton/GRANDE

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2023-grande-gradient-based-decision-tree-ensembles-for-tabular]]
- PDF: `raw/papers/pdf/2023-grande-gradient-based-decision-tree-ensembles-for-tabular.pdf`
- arXiv: <http://arxiv.org/abs/2309.17130v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-marton-grande.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub had wrong arXiv ID (2309.17429, an unrelated physics paper on KMOC waveforms). Corrected to **2309.17130** per the bib entry and arXiv search.

- **ArXiv:** 2309.17130
- **Authors:** Sascha Marton, Stefan Lüdtke, Christian Bartelt, Heiner Stuckenschmidt
- **Year:** 2024
- **Venue:** ICLR 2024
- **Raw:** [[raw/papers/2024-marton-grande.pdf]]

## Core claim

End-to-end gradient learning of *hard, axis-aligned* decision-tree ensembles via a dense matrix representation and a straight-through estimator. GRANDE positions itself as the 2024 endpoint of the differentiable-tree arc that NODE [@popov2020node] opened in 2019: keeping the right inductive bias (axis-aligned splits) for tables while gaining the flexibility of joint gradient-based optimisation. The mechanism that's new versus NODE is a per-instance leaf-weighting scheme that lets a single ensemble learn representations for both simple and complex relations.

## Architecture

GRANDE represents the entire ensemble as dense parameter tensors: split-feature indices, split thresholds, leaf values, and instance-wise leaf weights all live as differentiable parameters. The forward pass uses a *straight-through operator* — hard axis-aligned routing in the forward direction (so the model's expressive class is that of a hard-split ensemble), but a soft, differentiable surrogate in the backward direction (so gradients can flow through the splits). All parameters update jointly under standard gradient descent.

The instance-wise weighting is the second new element: each input gets per-instance leaf weights via a learned head, so trees specialise softly on subsets of the input distribution rather than committing globally. This is a softer version of mixture-of-experts gating, applied at the leaf granularity within a single ensemble.

## Key result

Evaluated on a predefined benchmark of **19 classification datasets** under a fair HPO protocol, GRANDE outperforms existing gradient-boosting frameworks (XGBoost, CatBoost, LightGBM) and existing deep tabular methods on most datasets. The paper does not claim universal dominance — the headline is "outperforms on most" rather than "always best."

## Why it matters for §2.4.1 (tests of rotational invariance)

GRANDE is the most-recent and most-committed instance of the "axis-alignment-as-architecture" strategy. Together with NODE it bookends the differentiable-tree arc:

- **NODE (2019/2020):** soft routing via entmax, multi-layer DenseNet-style stacking. First credible differentiable axis-aligned ensemble.
- **GRANDE (2023/2024):** straight-through hard-split forward + soft-gradient backward, dense parameterisation, instance-wise weighting. Better engineering, broader benchmark wins, but the same conceptual lever.

The diagnostic reading for §2.4.1: explicitly baking the §2.1 rotational-invariance fix into the architecture *can* narrow the gap, but the gain is bounded — across the same era, MLPs with proper numerical embeddings ([@gorishniy2022embeddings]) and ensembling ([@gorishniy2025tabm]) close at least as much of the gap with simpler architecture. The rotational-invariance lever isn't useless, but it isn't the binding constraint either.

## Caveats

- 19 datasets is a moderate benchmark; full TabZilla-scale validation [@mcelfresh2023tabzilla] is not reported in the paper.
- "Outperforms on most" is consistent with strong-but-not-universal performance; competing fair-protocol benchmarks (TabReD [@rubachev2025tabred], TabArena [@erickson2025tabarena]) include or post-date GRANDE and are the right comparator for grounded claims about its standing.
