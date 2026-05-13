---
arxiv: '2604.21999'
authors:
- Grigory Sapunov
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.21999
  raw: '[[raw/papers/md/2026-universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive]]'
  source: https://arxiv.org/abs/2604.21999
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive.md
raw_pdf: raw/papers/pdf/2026-universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive.pdf
read: false
slug: universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive
tags:
- type/paper
- status/stub
title: 'Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive
  Reasoning'
type: note
updated: '2026-05-11'
year: 2026
---

# Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning

> *Grigory Sapunov* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We study learned memory tokens as a computational scratchpad for a single-block Universal Transformer with Adaptive Computation Time (ACT) on Sudoku-Extreme, a combinatorial reasoning benchmark. Memory tokens are empirically necessary: no configuration without them reaches non-trivial performance. The optimal count has a sharp lower threshold (T=0 always fails, T=8 reliably succeeds) followed by a stable plateau (T=8-32, 57.4% +/- 0.7% exact-match) and a dilution boundary at T=64. Under halt-side pressure (lambda warmup), mean halt drops monotonically with memory size across the plateau (from 11.6 at T=8 to 8.3 at T=64), showing that memory tokens and ponder depth substitute as resources at fixed accuracy.
  We also identify a router initialization trap that causes the majority of training runs to fail: both default zero-bias and Graves' recommended positive bias settle into a shallow halt equilibrium the model cannot escape. Inverting the bias to -3 ("deep start") eliminates the failure mode, and ablation shows the trap is inherent to ACT initialization rather than an artifact of our architecture.
  With reliable training, ACT yields an order of magnitude lower seed variance than fixed-depth processing (+/-0.7 vs +/-9.3 pp); lambda warmup recovers 34% of compute at matched accuracy; and attention heads specialize into memory readers, constraint propagators, and integrators across recursive depth. Code: https://github.com/che-shr-cat/utm-jax.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.21999>
- PDF: [[raw/papers/pdf/2026-universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive.pdf]]
- Raw markdown: [[raw/papers/md/2026-universal-transformers-need-memory-depth-state-trade-offs-in-adaptive-recursive]]
