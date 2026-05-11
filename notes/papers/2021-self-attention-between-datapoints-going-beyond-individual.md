---
arxiv: '2106.02584'
authors:
- Jannik Kossen
- Neil Band
- Clare Lyle
- Aidan N. Gomez
- Tom Rainforth
- Yarin Gal
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2106.02584
  raw: '[[raw/papers/md/2021-self-attention-between-datapoints-going-beyond-individual]]'
  source: http://arxiv.org/abs/2106.02584v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-self-attention-between-datapoints-going-beyond-individual.md
raw_pdf: raw/papers/pdf/2021-self-attention-between-datapoints-going-beyond-individual.pdf
read: false
slug: self-attention-between-datapoints-going-beyond-individual
tags:
- type/paper
- attention
- tabular
- transformer
- self-supervised
- status/stub
title: 'Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs
  in Deep Learning'
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning

> *Jannik Kossen, Neil Band, Clare Lyle…* — arXiv 2106.02584, 2021

## Abstract

We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms. However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction. Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models. We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-self-attention-between-datapoints-going-beyond-individual]]
- PDF: [[raw/papers/pdf/2021-self-attention-between-datapoints-going-beyond-individual.pdf]]
- arXiv: <http://arxiv.org/abs/2106.02584v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-kossen-npt.md` before that tree was retired.*

> **2026-05-05 correction.** Previous stub had wrong arXiv ID 2104.01136. Corrected to **2106.02584** per the bib entry and verified abstract.

- **ArXiv:** 2106.02584
- **Authors:** Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, Yarin Gal
- **Year:** 2021
- **Venue:** NeurIPS 2021
- **Raw:** [[raw/papers/2021-kossen-npt.pdf]]

## Core claim

A "Non-Parametric Transformer" (NPT) takes the **entire dataset** as input and uses self-attention to reason about relationships between datapoints, breaking the standard supervised-learning assumption that prediction depends only on a single input's features. The architecture realises non-parametric models (k-NN, kernel methods) using a *parametric* attention mechanism whose use of context is learned end-to-end. On tabular benchmarks NPT is highly competitive, and on synthetic tasks requiring cross-datapoint lookup or relational reasoning, it solves problems that single-input deep models can't.

## Architecture

The NPT input is a *matrix* of $n$ rows × $d$ features. The architecture alternates two attention types:

1. **Attention between attributes (ABA).** Within each row, attention runs across the $d$ features — the standard column-wise transformer attention seen in FT-Transformer / TabTransformer / SAINT.
2. **Attention between datapoints (ABD).** Across rows, attention runs over the $n$ datapoints, using their combined feature embeddings as keys/queries/values. The model is allowed to look up other rows' representations to inform the current row's prediction.

Targets are masked at training time (BERT-style): the model predicts targets for masked rows using both unmasked-row context and the test row's own features. At inference time the test row's target is masked and the training rows' contexts inform prediction.

Crucially, NPT does **not** restrict attention to k nearest neighbours — the model learns end-to-end which other datapoints to attend to. The architectural inductive bias is "predictions can use other datapoints"; the *which* and *how* are learned.

## Key result

- Tabular benchmarks: NPT is "highly competitive" with leading tabular DL and GBDT methods (specific dataset-by-dataset rankings depend on the comparator and protocol; the paper does not claim universal dominance).
- Synthetic cross-datapoint reasoning tasks: NPT solves tasks unsolvable by standard deep models (e.g., tasks requiring lookup of values from other rows under various corruptions).
- Early CIFAR-10 results: NPT can also be applied to small image datasets via dataset-as-input, though that's not the main contribution.

## Why it matters for §2.4.3 (tests of irrelevant-feature robustness) and §2.4.5 (sidestepping the prior)

NPT is the conceptual midpoint between the within-row attention models (§2.4.3) and the retrieval-augmented / instance-based models (§2.4.5).

- For **§2.4.3 (irrelevant features):** ABA is column attention as soft feature gating, the same lever as TabTransformer / FT-Transformer / SAINT.
- For **§2.4.5 (sidestepping the smoothness prior):** ABD lets the model condition on a learned subset of training rows for each test prediction, sidestepping the "single-input parametric MLP must integrate over all training rows" regime where smoothness bias dominates. This is the same mechanism TabR [@gorishniy2024tabr] formalises later as retrieval-augmented prediction, and the spiritual precursor to TabPFN's [@hollmann2023tabpfn] in-context-learning framing (Chapter 5 territory).

For Chapter 3's argument NPT marks the **paradigm shift**: once you allow attention-between-datapoints, you have an architecture whose inductive bias is "use other rows" rather than "fit a smooth function." That's a different lever than §2.1/§2.2/§2.3 names, and it's the bridge from §2.4.5 to Chapter 5's pretraining-as-paradigm story.

## Caveats

- ABD's complexity is $O(n^2)$ in dataset size, limiting NPT to small/medium-data tabular tasks. Larger datasets need approximation (mini-batch attention, retrieval-restricted attention) — TabR is one such practical instantiation.
- "Highly competitive" rather than dominance; NPT is a conceptual milestone more than a deployable best-DL-on-tables model.
- The paper precedes TabPFN by a year and reads, in retrospect, as foreshadowing the in-context-learning regime that Chapter 5 develops.
