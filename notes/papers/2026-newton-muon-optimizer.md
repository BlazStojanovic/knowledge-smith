---
arxiv: '2604.01472'
authors:
- Zhehang Du
- Weijie Su
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.01472
  raw: '[[raw/papers/md/2026-newton-muon-optimizer]]'
  source: https://arxiv.org/abs/2604.01472
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-newton-muon-optimizer.md
raw_pdf: raw/papers/pdf/2026-newton-muon-optimizer.pdf
read: false
slug: newton-muon-optimizer
tags:
- type/paper
- status/stub
title: The Newton-Muon Optimizer
type: note
updated: '2026-05-11'
year: 2026
---

# The Newton-Muon Optimizer

> *Zhehang Du, Weijie Su* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

The Muon optimizer has received considerable attention for its strong performance in training large language models, yet the design principle behind its matrix-gradient orthogonalization remains largely elusive. In this paper, we introduce a surrogate model that not only sheds new light on the design of Muon, but more importantly leads to a new optimizer. In the same spirit as the derivation of Newton's method, the surrogate approximates the loss as a quadratic function of the perturbation to a weight matrix $W$ using only three matrices: the gradient $G$, an output-space curvature matrix $H$, and the data matrix $Z$ that stacks the layer inputs. By minimizing this surrogate in one step and adopting a certain isotropic assumption on the weights, we obtain the closed-form update rule (up to momentum and weight decay) $W \leftarrow W - η\cdot \mathrm{msgn}(G(ZZ^\top)^{-1})$, where $η$ is the learning rate and $\mathrm{msgn}(X)=UV^\top$ if $X=USV^\top$ is a compact singular value decomposition. This new optimization method, which we refer to as Newton-Muon, shows that standard Muon can be interpreted as an implicit Newton-type method that neglects the right preconditioning induced by the input second moment. Empirically, on a reproduction of the earliest publicly released Modded-NanoGPT speedrun configuration using Muon for GPT-2 pretraining, Newton-Muon reaches the target validation loss in 6\% fewer iteration steps and reduces wall-clock training time by about 4\%.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.01472>
- PDF: [[raw/papers/pdf/2026-newton-muon-optimizer.pdf]]
- Raw markdown: [[raw/papers/md/2026-newton-muon-optimizer]]
