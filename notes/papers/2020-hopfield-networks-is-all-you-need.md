---
arxiv: '2008.02217'
authors:
- Hubert Ramsauer
- Bernhard Schäfl
- Johannes Lehner
- Philipp Seidl
- Michael Widrich
- Thomas Adler
- Lukas Gruber
- Markus Holzleitner
- Milena Pavlović
- Geir Kjetil Sandve
- Victor Greiff
- David Kreil
- Michael Kopp
- Günter Klambauer
- Johannes Brandstetter
- Sepp Hochreiter
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2020-hopfield-networks-is-all-you-need.md
raw_pdf: raw/papers/pdf/2020-hopfield-networks-is-all-you-need.pdf
read: false
slug: hopfield-networks-is-all-you-need
tags:
- attention
- transformer
- ml
- theory
title: Hopfield Networks is All You Need
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2008.02217v3
venue: null
year: 2020
---

# Hopfield Networks is All You Need

> *Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner…* — arXiv 2008.02217, 2020

## Abstract

We introduce a modern Hopfield network with continuous states and a corresponding update rule. The new Hopfield network can store exponentially (with the dimension of the associative space) many patterns, retrieves the pattern with one update, and has exponentially small retrieval errors. It has three types of energy minima (fixed points of the update): (1) global fixed point averaging over all patterns, (2) metastable states averaging over a subset of patterns, and (3) fixed points which store a single pattern. The new update rule is equivalent to the attention mechanism used in transformers. This equivalence enables a characterization of the heads of transformer models. These heads perform in the first layers preferably global averaging and in higher layers partial averaging via metastable states. The new modern Hopfield network can be integrated into deep learning architectures as layers to allow the storage of and access to raw input data, intermediate results, or learned prototypes. These Hopfield layers enable new ways of deep learning, beyond fully-connected, convolutional, or recurrent networks, and provide pooling, memory, association, and attention mechanisms. We demonstrate the broad applicability of the Hopfield layers across various domains. Hopfield layers improved state-of-the-art on three out of four considered multiple instance learning problems as well as on immune repertoire classification with several hundreds of thousands of instances. On the UCI benchmark collections of small classification tasks, where deep learning methods typically struggle, Hopfield layers yielded a new state-of-the-art when compared to different machine learning methods. Finally, Hopfield layers achieved state-of-the-art on two drug design datasets. The implementation is available at: https://github.com/ml-jku/hopfield-layers

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2020-hopfield-networks-is-all-you-need]]
- PDF: `raw/papers/pdf/2020-hopfield-networks-is-all-you-need.pdf`
- arXiv: <http://arxiv.org/abs/2008.02217v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-ramsauer-modern-hopfield.md` before that tree was retired.*

Modern Hopfield networks equal transformer attention; theoretical basis for Hopular.
