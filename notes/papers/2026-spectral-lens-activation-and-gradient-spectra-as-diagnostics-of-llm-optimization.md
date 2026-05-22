---
arxiv: '2605.05683'
authors:
- Andy Zeyi Liu
- Elliot Paquette
- John Sous
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.05683
  raw: '[[raw/papers/md/2026-spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization]]'
  source: https://arxiv.org/abs/2605.05683
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization.md
raw_pdf: raw/papers/pdf/2026-spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization.pdf
read: false
slug: spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization
tags:
- type/paper
- status/stub
title: 'Spectral Lens: Activation and Gradient Spectra as Diagnostics of LLM Optimization'
type: note
updated: '2026-05-22'
year: 2026
---

# Spectral Lens: Activation and Gradient Spectra as Diagnostics of LLM Optimization

> *Andy Zeyi Liu, Elliot Paquette, John Sous* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Training loss and throughput can hide distinct internal representation in language-model training. To examine these hidden mechanics, we use spectral measurements as practical and operational diagnostics. Using a controlled family of decoder-only models adapted from the modded NanoGPT codebase, we introduce an empirical protocol based on activation covariance and per-sample gradient SVD spectra. This dual-view reveals three empirical findings and one mechanistic explanation. First, batch size acts as a latent determinant of representation geometry: runs that reach equal loss settle into systematically distinct activation spectra. Second, the activation covariance tail measured early in training reliably forecasts downstream token efficiency. Third, movement of the activation spectrum head (leading modes), together with gradient spectra, characterizes underlying learning-dynamics changes, separating learning-side architectural improvements from primarily execution-side gains. These predictive and diagnostic signals persist across the 12-, 36-, and 48-layer model tiers. Finally, a mechanistic model proves the main observations and explains how activation covariance spectra correlate with task-aligned feature learning.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.05683>
- PDF: [[raw/papers/pdf/2026-spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization.pdf]]
- Raw markdown: [[raw/papers/md/2026-spectral-lens-activation-and-gradient-spectra-as-diagnostics-of-llm-optimization]]
