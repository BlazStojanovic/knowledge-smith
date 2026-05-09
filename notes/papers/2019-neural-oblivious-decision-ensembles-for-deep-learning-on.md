---
arxiv: '1909.06312'
authors:
- Sergei Popov
- Stanislav Morozov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2019-neural-oblivious-decision-ensembles-for-deep-learning-on.md
raw_pdf: raw/papers/pdf/2019-neural-oblivious-decision-ensembles-for-deep-learning-on.pdf
read: false
slug: neural-oblivious-decision-ensembles-for-deep-learning-on
tags:
- tabular
- decision-tree
- gradient-boosting
- ml
title: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1909.06312v2
venue: null
year: 2019
---

# Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

> *Sergei Popov, Stanislav Morozov, Artem Babenko* — arXiv 1909.06312, 2019

## Abstract

Nowadays, deep neural networks (DNNs) have become the main instrument for machine learning tasks within a wide range of domains, including vision, NLP, and speech. Meanwhile, in an important case of heterogenous tabular data, the advantage of DNNs over shallow counterparts remains questionable. In particular, there is no sufficient evidence that deep learning machinery allows constructing methods that outperform gradient boosting decision trees (GBDT), which are often the top choice for tabular problems. In this paper, we introduce Neural Oblivious Decision Ensembles (NODE), a new deep learning architecture, designed to work with any tabular data. In a nutshell, the proposed NODE architecture generalizes ensembles of oblivious decision trees, but benefits from both end-to-end gradient-based optimization and the power of multi-layer hierarchical representation learning. With an extensive experimental comparison to the leading GBDT packages on a large number of tabular datasets, we demonstrate the advantage of the proposed NODE architecture, which outperforms the competitors on most of the tasks. We open-source the PyTorch implementation of NODE and believe that it will become a universal framework for machine learning on tabular data.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2019-neural-oblivious-decision-ensembles-for-deep-learning-on]]
- PDF: `raw/papers/pdf/2019-neural-oblivious-decision-ensembles-for-deep-learning-on.pdf`
- arXiv: <http://arxiv.org/abs/1909.06312v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-popov-node.md` before that tree was retired.*

## Core claim

Tabular DL had no convincing baseline against GBDTs through 2019; NODE proposes a deep, end-to-end-differentiable architecture that *generalises ensembles of oblivious decision trees* and reports outperforming the leading GBDT packages on most tabular benchmarks. The thesis is structural: if the right inductive bias for tables is axis-aligned splits, build that bias into a deep network rather than fighting it.

## Architecture

The atomic NODE layer is a *differentiable oblivious decision tree* (ODT). An ODT is the same tree CatBoost uses internally — a fully balanced tree where every node at the same depth tests the same feature with the same threshold. NODE replaces ODT's hard splits with `entmax`-based soft routing so the tree is differentiable end-to-end:

- Each tree has $d$ depth, $2^d$ leaves, and $d$ split decisions, each a soft-attention over input features.
- Splits use an `entmax` activation (a sparsity-tunable softmax variant) that drives the routing toward axis-aligned, hard-split behaviour at the limit while remaining differentiable in the interior.
- Multiple trees are stacked layer-wise (DenseNet-style residual stacking): each layer's outputs concatenate to the next layer's input features, enabling hierarchical representation learning across multiple "rounds" of tree splits.

The result is a network whose *expressive class* is roughly that of a CatBoost-style model but whose *optimization story* is end-to-end SGD with backprop, allowing combination with any differentiable head and joint training inside larger pipelines.

## Key result

Across a large set of tabular benchmarks (Higgs, Microsoft, Yahoo, Click, etc.), NODE outperforms tuned XGBoost, CatBoost, and LightGBM on most tasks under matched HPO budgets. The paper's headline framing: NODE is the first deep tabular architecture to credibly compete with GBDTs as a default choice.

## Why it matters for §2.4.1 (tests of rotational invariance)

NODE is the strongest "axis-aligned-by-construction" deep architecture in the literature — the most committed implementation of the prior §2.1 names as load-bearing. If the rotational-invariance diagnosis is correct (any rotationally invariant learner pays $\Omega(K)$ irrelevant-feature sample cost; trees pay $O(\log K)$), then explicitly baking axis-alignment into the architecture should narrow the gap. NODE is the controlled experiment: it does narrow the gap, and on some benchmarks closes it. But subsequent reproductions ([@gorishniy2021ftt; @holzmuller2024realmlp]) found that NODE's wins are sensitive to HPO budget and benchmark choice, and a well-tuned MLP with proper input embeddings often catches up or beats it ([@gorishniy2022embeddings]). The rotational-invariance lever, by itself, isn't enough to dethrone GBDTs.

## Caveats

- NODE's training is markedly slower than tuned GBDTs (multiple-hour fits where CatBoost converges in minutes), which the paper acknowledges and which has limited NODE's adoption in practice.
- The "outperforms GBDT on most tasks" framing depends on the baseline tuning protocol — fair-protocol comparisons in [@gorishniy2021ftt] and [@grinsztajn2022tree] are less favourable.
