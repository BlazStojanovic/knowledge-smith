---
arxiv: '2307.14338'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Nikolay Kartashev
- Daniil Shlenskii
- Akim Kotelnikov
- Artem Babenko
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2023-tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023.md
raw_pdf: raw/papers/pdf/2023-tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023.pdf
read: false
slug: tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023
tags:
- tabular
- retrieval
- attention
- benchmark
title: 'TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2307.14338v2
venue: null
year: 2023
---

# TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023

> *Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev…* — arXiv 2307.14338, 2023

## Abstract

Deep learning (DL) models for tabular data problems (e.g. classification, regression) are currently receiving increasingly more attention from researchers. However, despite the recent efforts, the non-DL algorithms based on gradient-boosted decision trees (GBDT) remain a strong go-to solution for these problems. One of the research directions aimed at improving the position of tabular DL involves designing so-called retrieval-augmented models. For a target object, such models retrieve other objects (e.g. the nearest neighbors) from the available training data and use their features and labels to make a better prediction.
  In this work, we present TabR -- essentially, a feed-forward network with a custom k-Nearest-Neighbors-like component in the middle. On a set of public benchmarks with datasets up to several million objects, TabR marks a big step forward for tabular DL: it demonstrates the best average performance among tabular DL models, becomes the new state-of-the-art on several datasets, and even outperforms GBDT models on the recently proposed "GBDT-friendly" benchmark (see Figure 1). Among the important findings and technical details powering TabR, the main ones lie in the attention-like mechanism that is responsible for retrieving the nearest neighbors and extracting valuable signal from them. In addition to the much higher performance, TabR is simple and significantly more efficient compared to prior retrieval-based tabular DL models.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2023-tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023]]
- PDF: `raw/papers/pdf/2023-tabr-tabular-deep-learning-meets-nearest-neighbors-in-2023.pdf`
- arXiv: <http://arxiv.org/abs/2307.14338v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-gorishniy-tabr.md` before that tree was retired.*

## Core claim

A feed-forward MLP with a custom attention-based k-NN-like retrieval module in the middle achieves the best average performance among tabular DL models on a public benchmark suite, becomes new SOTA on several individual datasets, and *outperforms GBDT models on the GBDT-friendly Grinsztajn et al. benchmark* [@grinsztajn2022tree]. TabR is the strongest 2024 instance of the retrieval-augmented direction in tabular DL.

## Architecture

The TabR pipeline:

1. **Feature encoder.** A standard MLP (with proper numerical encodings per [@gorishniy2022embeddings]) projects each input into a $d$-dimensional embedding $h$.
2. **Retrieval module (the load-bearing component).** For a query $x$ with embedding $h_x$, retrieve the $k$ nearest training-set neighbours by embedding similarity. An *attention-like* mechanism then computes context vectors from the retrieved neighbours and their labels — the attention weights are functions of (query embedding, neighbour embedding, neighbour label), so the retrieval is *learned end-to-end* with the prediction task.
3. **Prediction head.** A second MLP takes the query embedding plus the retrieval-context vector and produces the final prediction.

The architectural innovation versus prior retrieval-tabular methods is the **value extraction**: the model doesn't just average neighbour labels, it learns *what signal to extract* from each neighbour conditional on the query. The attention is over training rows (similar in spirit to NPT [@kossen2021npt]) but restricted to a learned-relevant subset (the k nearest), giving practical scaling beyond NPT's whole-batch attention.

## Key result

- Public-benchmark suite (datasets up to several million objects).
- TabR achieves **best average performance among tabular DL models** evaluated.
- Becomes new SOTA on several individual datasets.
- **Outperforms GBDT models on Grinsztajn et al.'s "GBDT-friendly" benchmark** [@grinsztajn2022tree] — the benchmark designed to favour tree-style methods. This is the strongest 2024-era result for a deep tabular method versus GBDT.

## Why it matters for §2.4.5 (sidestepping the prior)

TabR is the canonical "instance-based conditioning" instance in §2.4.5. The §2.1/§2.2/§2.3 priors describe what an MLP gets wrong globally (rotational invariance, spectral bias, irrelevant-feature sensitivity). TabR's retrieval mechanism *sidesteps* all three by conditioning each prediction on a learned-relevant *subset* of training rows:

- Retrieval breaks the globally-smooth surface that spectral bias prefers — local neighbour aggregation can produce sharp boundaries where the data warrants them.
- Per-query feature relevance is implicit in the neighbour-similarity computation: only the features that matter for "looks like this query" contribute to the retrieval, sidestepping irrelevant-feature degradation.
- Rotational invariance in the encoder is irrelevant when the head's prediction depends on retrieved-neighbour labels, not on globally-smooth function fitting.

The diagnostic reading: TabR's win against GBDTs on Grinsztajn's benchmark is the strongest signal that *the gap closes when you change the inference regime*, not just the architecture. This conceptually rhymes with TabPFN's [@hollmann2023tabpfn] in-context-learning regime (Chapter 5), where the model conditions on training rows directly. TabR is the supervised-learning analogue: train the retriever and predictor jointly, then use them at inference like a learned nearest-neighbours classifier.

## Caveats

- Inference cost depends on the retrieval-index size and embedding dimensionality; for large training sets this can be non-trivial vs. a forward pass through a plain MLP.
- "Best average DL performance" is contingent on the benchmark suite; TabM [@gorishniy2025tabm] later challenges this within the MLP family with parameter-efficient ensembling.
- The Grinsztajn-benchmark win is on a specific (medium-sized, heterogeneous-numerical) regime; production-scale tabular workloads with high-cardinality categoricals or strong temporal structure are out of scope here.
