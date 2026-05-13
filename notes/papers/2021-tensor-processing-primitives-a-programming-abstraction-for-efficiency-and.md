---
arxiv: '2104.05755'
authors:
- Evangelos Georganas
- Dhiraj Kalamkar
- Sasikanth Avancha
- Menachem Adelman
- Deepti Aggarwal
- Cristina Anderson
- Alexander Breuer
- Jeremy Bruestle
- Narendra Chaudhary
- Abhisek Kundu
- Denise Kutnick
- Frank Laub
- Vasimuddin Md
- Sanchit Misra
- Ramanarayan Mohanty
- Hans Pabst
- Brian Retford
- Barukh Ziv
- Alexander Heinecke
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2104.05755
  raw: '[[raw/papers/md/2021-tensor-processing-primitives-a-programming-abstraction-for-efficiency-and]]'
  source: https://arxiv.org/abs/2104.05755
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-tensor-processing-primitives-a-programming-abstraction-for-efficiency-and.md
raw_pdf: raw/papers/pdf/2021-tensor-processing-primitives-a-programming-abstraction-for-efficiency-and.pdf
read: false
slug: tensor-processing-primitives-a-programming-abstraction-for-efficiency-and
tags:
- type/paper
- status/stub
title: 'Tensor Processing Primitives: A Programming Abstraction for Efficiency and
  Portability in Deep Learning & HPC Workloads'
type: note
updated: '2026-05-11'
year: 2021
---

# Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads

> *Evangelos Georganas, Dhiraj Kalamkar, Sasikanth Avancha, Menachem Adelman, Deepti Aggarwal, et al.* — arXiv 2021

## TL;DR

(stub — fill in after reading)

## Abstract

During the past decade, novel Deep Learning (DL) algorithms, workloads and hardware have been developed to tackle a wide range of problems. Despite the advances in workload and hardware ecosystems, the programming methodology of DL systems is stagnant. DL workloads leverage either highly-optimized, yet platform-specific and inflexible kernels from DL libraries, or in the case of novel operators, reference implementations are built via DL framework primitives with underwhelming performance. This work introduces the Tensor Processing Primitives (TPP), a programming abstraction striving for efficient, portable implementation of DL workloads with high-productivity. TPPs define a compact, yet versatile set of 2D-tensor operators (or a virtual Tensor ISA), which subsequently can be utilized as building-blocks to construct complex operators on high-dimensional tensors. The TPP specification is platform-agnostic, thus code expressed via TPPs is portable, whereas the TPP implementation is highly-optimized and platform-specific. We demonstrate the efficacy and viability of our approach using standalone kernels and end-to-end DL & HPC workloads expressed entirely via TPPs that outperform state-of-the-art implementations on multiple platforms.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2104.05755>
- PDF: [[raw/papers/pdf/2021-tensor-processing-primitives-a-programming-abstraction-for-efficiency-and.pdf]]
- Raw markdown: [[raw/papers/md/2021-tensor-processing-primitives-a-programming-abstraction-for-efficiency-and]]
