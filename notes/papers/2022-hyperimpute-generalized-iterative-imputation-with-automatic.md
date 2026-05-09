---
arxiv: '2206.07769'
authors:
- Daniel Jarrett
- Bogdan Cebere
- Tennison Liu
- Alicia Curth
- Mihaela van der Schaar
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2206.07769.md
raw_pdf: raw/papers/2206.07769.pdf
read: false
slug: hyperimpute-generalized-iterative-imputation-with-automatic
tags:
- missing-data
- tabular
- automl
title: 'HyperImpute: Generalized Iterative Imputation with Automatic Model Selection'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2206.07769v1
venue: null
year: 2022
---

# HyperImpute: Generalized Iterative Imputation with Automatic Model Selection

> *Daniel Jarrett, Bogdan Cebere, Tennison Liu…* — arXiv 2206.07769, 2022

## Abstract

Consider the problem of imputing missing values in a dataset. One the one hand, conventional approaches using iterative imputation benefit from the simplicity and customizability of learning conditional distributions directly, but suffer from the practical requirement for appropriate model specification of each and every variable. On the other hand, recent methods using deep generative modeling benefit from the capacity and efficiency of learning with neural network function approximators, but are often difficult to optimize and rely on stronger data assumptions. In this work, we study an approach that marries the advantages of both: We propose *HyperImpute*, a generalized iterative imputation framework for adaptively and automatically configuring column-wise models and their hyperparameters. Practically, we provide a concrete implementation with out-of-the-box learners, optimizers, simulators, and extensible interfaces. Empirically, we investigate this framework via comprehensive experiments and sensitivities on a variety of public datasets, and demonstrate its ability to generate accurate imputations relative to a strong suite of benchmarks. Contrary to recent work, we believe our findings constitute a strong defense of the iterative imputation paradigm.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2206.07769]]
- PDF: `raw/papers/2206.07769.pdf`
- arXiv: <http://arxiv.org/abs/2206.07769v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2022-perez-lebel-benchmark-missing.md` before that tree was retired.*

Benchmark on real health tables; GBDTs with native NA handling remain strong.
