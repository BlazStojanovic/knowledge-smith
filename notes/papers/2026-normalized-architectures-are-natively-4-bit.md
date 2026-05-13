---
arxiv: '2605.06067'
authors:
- Maxim Fishman
- Brian Chmiel
- Ron Banner
- Daniel Soudry
- Boris Ginsburg
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.06067
  raw: '[[raw/papers/md/2026-normalized-architectures-are-natively-4-bit]]'
  source: https://arxiv.org/abs/2605.06067
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-normalized-architectures-are-natively-4-bit.md
raw_pdf: raw/papers/pdf/2026-normalized-architectures-are-natively-4-bit.pdf
read: false
slug: normalized-architectures-are-natively-4-bit
tags:
- type/paper
- status/stub
title: Normalized Architectures are Natively 4-Bit
type: note
updated: '2026-05-11'
year: 2026
---

# Normalized Architectures are Natively 4-Bit

> *Maxim Fishman, Brian Chmiel, Ron Banner, Daniel Soudry, Boris Ginsburg* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Training large language models at 4-bit precision is critical for efficiency. We show that nGPT, an architecture that constrains weights and hidden representations to the unit hypersphere, is inherently more robust to low-precision arithmetic. This removes the need for interventions-such as applying random Hadamard transforms and performing per-tensor scaling calculations-to preserve model quality, and it enables stable end-to-end NVFP4 training. We validate this approach on both a 1.2B dense model and hybrid (Mamba-Transformer) MoE models of up to 3B/30B parameters. We trace this robustness to the dot product: while quantization noise remains largely uncorrelated in both standard and normalized architectures, the signal behaves differently. In nGPT, the hypersphere constraint enhances weak positive correlations among the element-wise products, leading to a constructive accumulation of the signal across the hidden dimension while the noise continues to average out. This yields a higher effective signal-to-noise ratio and a flatter loss landscape, with the effect strengthening as the hidden dimension grows, suggesting increasing advantages at scale. A reference implementation is available at https://github.com/anonymous452026/ngpt-nvfp4

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.06067>
- PDF: [[raw/papers/pdf/2026-normalized-architectures-are-natively-4-bit.pdf]]
- Raw markdown: [[raw/papers/md/2026-normalized-architectures-are-natively-4-bit]]
