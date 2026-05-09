---
arxiv: '2408.10717'
authors:
- Yifu Han
- Francois P. Hamon
- Louis J. Durlofsky
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2408.10717.md
raw_pdf: raw/papers/2408.10717.pdf
read: false
slug: accelerated-training-of-deep-learning-surrogate-models-for
tags:
- ml
- surrogate-model
- uncertainty
- simulation
- geoscience
title: Accelerated training of deep learning surrogate models for surface displacement
  and flow, with application to MCMC-based history matching of CO2 storage operations
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2408.10717v1
venue: null
year: 2024
---

# Accelerated training of deep learning surrogate models for surface displacement and flow, with application to MCMC-based history matching of CO2 storage operations

> *Yifu Han, Francois P. Hamon, Louis J. Durlofsky* — arXiv 2408.10717, 2024

## Abstract

Deep learning surrogate modeling shows great promise for subsurface flow applications, but the training demands can be substantial. Here we introduce a new surrogate modeling framework to predict CO2 saturation, pressure and surface displacement for use in the history matching of carbon storage operations. Rather than train using a large number of expensive coupled flow-geomechanics simulation runs, training here involves a large number of inexpensive flow-only simulations combined with a much smaller number of coupled runs. The flow-only runs use an effective rock compressibility, which is shown to provide accurate predictions for saturation and pressure for our system. A recurrent residual U-Net architecture is applied for the saturation and pressure surrogate models, while a new residual U-Net model is introduced to predict surface displacement. The surface displacement surrogate accepts, as inputs, geomodel quantities along with saturation and pressure surrogate predictions. Median relative error for a diverse test set is less than 4% for all variables. The surrogate models are incorporated into a hierarchical Markov chain Monte Carlo history matching workflow. Surrogate error is included using a new treatment involving the full model error covariance matrix. A high degree of prior uncertainty, with geomodels characterized by uncertain geological scenario parameters (metaparameters) and associated realizations, is considered. History matching results for a synthetic true model are generated using in-situ monitoring-well data only, surface displacement data only, and both data types. The enhanced uncertainty reduction achieved with both data types is quantified. Posterior saturation and surface displacement fields are shown to correspond well with the true solution.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2408.10717]]
- PDF: `raw/papers/2408.10717.pdf`
- arXiv: <http://arxiv.org/abs/2408.10717v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-ye-talent.md` before that tree was retired.*

TALENT — a closer look at deep tabular methods across diverse benchmarks.
