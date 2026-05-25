---
arxiv: '2605.21488'
authors:
- Benhao Huang
- Zhengyang Geng
- Zico Kolter
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.21488
  raw: '[[raw/papers/md/2026-equilibrium-reasoners-learning-attractors-enables-scalable-reasoning]]'
  source: https://arxiv.org/abs/2605.21488
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-equilibrium-reasoners-learning-attractors-enables-scalable-reasoning.md
raw_pdf: raw/papers/pdf/2026-equilibrium-reasoners-learning-attractors-enables-scalable-reasoning.pdf
read: false
slug: equilibrium-reasoners-learning-attractors-enables-scalable-reasoning
tags:
- type/paper
- status/stub
title: 'Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning'
type: note
updated: '2026-05-25'
year: 2026
---

# Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning

> *Benhao Huang, Zhengyang Geng, Zico Kolter* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Scaling test-time compute by iteratively updating a latent state has emerged as a powerful paradigm for reasoning. Yet the internal mechanisms that enable these iterative models to generalize beyond memorized patterns remain unclear. We hypothesize that generalizable reasoning arises from learning task-conditioned attractors: latent dynamical systems whose stable fixed points correspond to valid solutions.
  We formalize this process through Equilibrium Reasoners (EqR), which enable test-time scaling without external verifiers or task-specific priors. EqR scales internal dynamics along two axes: depth, by running more iterations, and breadth, by aggregating stochastic trajectories from multiple initializations. Empirically, gains from test-time scaling are tightly coupled with stronger convergence toward solution-aligned attractors.
  This attractor perspective allows neural networks to adaptively allocate test-time compute based on task difficulty. While simple cases converge within 1 to 5 iteration steps, harder cases benefit from massive test-time scaling. By unrolling up to the equivalent of 40,000 layers, scalable latent reasoning boosts accuracy from 2.6% for feedforward models to over 99% on Sudoku-Extreme. These results suggest that learned attractor landscapes provide a useful mechanistic lens for understanding scalable reasoning in iterative latent models.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.21488>
- PDF: [[raw/papers/pdf/2026-equilibrium-reasoners-learning-attractors-enables-scalable-reasoning.pdf]]
- Raw markdown: [[raw/papers/md/2026-equilibrium-reasoners-learning-attractors-enables-scalable-reasoning]]
