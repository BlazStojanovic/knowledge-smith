---
arxiv: null
authors:
- Ege Beyazit
- Jonathan Kozaczuk
- Bo Li
- Vanessa Wallace
- Bilal H. Fadlallah
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: null
  raw: null
  source: null
owner: blaz
parser: none
raw_md: null
raw_pdf: null
read: false
slug: 2023-beyazit-inductive-bias
tags:
- type/paper
- tabular
- ml
- generalization
- status/stub
title: An Inductive Bias for Tabular Deep Learning
type: note
updated: '2026-05-09'
venue: NeurIPS 2023
year: 2023
---

# An Inductive Bias for Tabular Deep Learning

*NeurIPS 2023, 2023*

## TL;DR

(stub — fill in after reading)

## Notes

(your synthesis)

## Source

- PDF: [[raw/papers/pdf/2023-beyazit-inductive-bias.pdf]]

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2023-beyazit-inductive-bias.md` before that tree was retired.*

> **2026-05-03 correction.** The previous stub had wrong title ("Inductive Biases of Neural Networks on Tabular Benchmarks"), wrong authors ("Hecke Moreno Beyazit, Farivar, Salemi, An, Habermehl"), and wrong arXiv ID (2307.10236, which is a paper on LLM uncertainty estimation). All three were apparent fabrications. The note has been rewritten against the OpenReview record and the bib entry.

- **OpenReview:** [XEUc1JegGt](https://openreview.net/forum?id=XEUc1JegGt)
- **Authors:** Ege Beyazit, Jonathan Kozaczuk, Bo Li, Vanessa Wallace, Bilal H. Fadlallah
- **Year:** 2023
- **Venue:** NeurIPS 2023
- **Raw:** [[raw/papers/2023-beyazit-inductive-bias.pdf]]

## Core claim

Neural networks underperform GBDTs on tabular data because (i) tabular targets are *irregular* in a spectral sense and (ii) MLPs have a smoothness bias that pulls them away from such targets. The paper makes both halves quantitative via spectral analysis and proposes a *frequency-reduction layer* — an inductive-bias layer that pushes the network toward learning low-frequency representations of features, so the optimisation problem lives in a smoother space.

The mechanism is the same diagnosis as [@rahaman2019spectralbias] and [@gorishniy2022embeddings] approached from the opposite direction: rather than injecting high-frequency basis functions on the input, smooth the *target representation* on the output side via a learned layer.

## What the paper actually shows

- **Empirical spectral analysis of tabular targets.** Functions described by tabular datasets exhibit measurably higher irregularity than vision/text counterparts.
- **Smoothness transformations as a baseline.** Per-feature scaling and ranking transforms reduce target irregularity and improve NN performance — but the gain depends on careful per-feature tuning, and the transforms either lose information or hurt the loss landscape.
- **Frequency-reduction layer.** A learned layer that promotes low-frequency representations of inputs. Lower computational complexity than a fully connected layer; reported to *significantly* improve NN performance and convergence speed across **14 tabular datasets**.

## Why it matters for §2.2

Two-fold:

1. *Independent confirmation* of the spectral-bias diagnosis on tabular data, with direct measurement of target irregularity.
2. *Counterpoint to the Gorishniy embedding direction.* Gorishniy injects high-frequency basis functions on the input; Beyazit smooths the representation toward low frequencies. Both close part of the gap. This is itself diagnostic — both directions of spectral remediation help, which strengthens the conclusion that the spectral mismatch was the bottleneck.

## Caveats

- The paper proposes its own method (frequency-reduction layer) rather than just diagnosing the spectral mismatch in the abstract. For §2 of post 1 we want the *diagnostic* part — tabular targets are spectrally irregular — not necessarily the proposed solution.
- 14 datasets is moderate; broader generalisation untested.
