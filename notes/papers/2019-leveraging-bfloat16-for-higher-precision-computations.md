---
arxiv: '1904.06376'
authors:
- Greg Henry
- Ping Tak Peter Tang
- Alexander Heinecke
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1904.06376
  raw: '[[raw/papers/md/2019-leveraging-bfloat16-for-higher-precision-computations]]'
  source: https://arxiv.org/abs/1904.06376
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2019-leveraging-bfloat16-for-higher-precision-computations.md
raw_pdf: raw/papers/pdf/2019-leveraging-bfloat16-for-higher-precision-computations.pdf
read: false
slug: leveraging-bfloat16-for-higher-precision-computations
tags:
- type/paper
- status/stub
title: Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision
  Computations
type: note
updated: '2026-05-25'
venue: ARITH 2019
year: 2019
---

# Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations

> *Greg Henry, Ping Tak Peter Tang, Alexander Heinecke* — ARITH 2019

## TL;DR

(stub — fill in after reading)

## Abstract

Bfloat16 (BF16), a truncated FP32 with 8-bit exponent and 7-bit mantissa, is shipping in upcoming AI accelerators (Intel Cooper Lake / Sapphire Rapids, Google TPUs, ARM). The paper shows that FP32-range computations like vector inner products and matrix-matrix products can be assembled by decomposing each FP32 input into 2-3 BF16 components and using a BF16-input / FP32-accumulate FMA unit, with projected speed-ups up to ~5.2x over native FP32 on representative kernels. Provides error analyses for the decomposed kernels and a path to "higher-than-BF16" precision without paying for a full FP32 ALU.

## Notes

(stub — Poolside context: BF16/FP32 mixed-precision is the default in titan; this paper is one of the foundational references for thinking about precision-vs-throughput trade-offs in matmul.)

## Source

- arXiv: <https://arxiv.org/abs/1904.06376>
- PDF: [[raw/papers/pdf/2019-leveraging-bfloat16-for-higher-precision-computations.pdf]]
- Raw markdown: [[raw/papers/md/2019-leveraging-bfloat16-for-higher-precision-computations]]
