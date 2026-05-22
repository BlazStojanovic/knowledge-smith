---
arxiv: '2605.19376'
authors:
- Junyeob Baek
- Mingyu Jo
- Minsu Kim
- Mengye Ren
- Yoshua Bengio
- Sungjin Ahn
created: '2026-05-22'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.19376
  raw: '[[raw/papers/md/2026-generative-recursive-reasoning]]'
  source: https://arxiv.org/abs/2605.19376
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-generative-recursive-reasoning.md
raw_pdf: raw/papers/pdf/2026-generative-recursive-reasoning.pdf
read: false
slug: generative-recursive-reasoning
tags:
- type/paper
- status/stub
title: Generative Recursive Reasoning
type: note
updated: '2026-05-22'
year: 2026
---

# Generative Recursive Reasoning

> *Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

How should future neural reasoning systems implement extended computation? Recursive Reasoning Models (RRMs) offer a promising alternative to autoregressive sequence extension by performing iterative latent-state refinement with shared transition functions. Yet existing RRMs are largely deterministic, following a single latent trajectory and converging to a single prediction. We introduce Generative Recursive reAsoning Models (GRAM), a framework that turns recursive latent reasoning into probabilistic multi-trajectory computation. GRAM models reasoning as a stochastic latent trajectory, enabling multiple hypotheses, alternative solution strategies, and inference-time scaling through both recursive depth and parallel trajectory sampling. This yields a latent-variable generative model supporting conditional reasoning via $p_θ(y \mid x)$ and, with fixed or absent inputs, unconditional generation via $p_θ(x)$. Trained with amortized variational inference, GRAM improves over deterministic recurrent and recursive baselines on structured reasoning and multi-solution constraint satisfaction tasks, while demonstrating an unconditional generation capability. https://ahn-ml.github.io/gram-website

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.19376>
- PDF: [[raw/papers/pdf/2026-generative-recursive-reasoning.pdf]]
- Raw markdown: [[raw/papers/md/2026-generative-recursive-reasoning]]
