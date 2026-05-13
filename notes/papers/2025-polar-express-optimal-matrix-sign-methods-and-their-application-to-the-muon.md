---
arxiv: '2505.16932'
authors:
- Noah Amsel
- David Persson
- Christopher Musco
- Robert M. Gower
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.16932
  raw: '[[raw/papers/md/2025-polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon]]'
  source: https://arxiv.org/abs/2505.16932
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon.md
raw_pdf: raw/papers/pdf/2025-polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon.pdf
read: false
slug: polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon
tags:
- type/paper
- status/stub
title: 'The Polar Express: Optimal Matrix Sign Methods and Their Application to the
  Muon Algorithm'
type: note
updated: '2026-05-11'
year: 2025
---

# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

> *Noah Amsel, David Persson, Christopher Musco, Robert M. Gower* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Computing the polar decomposition and the related matrix sign function has been a well-studied problem in numerical analysis for decades. Recently, it has emerged as an important subroutine within the Muon optimizer for training deep neural networks. However, the requirements of this application differ sharply from classical settings: deep learning demands GPU-friendly algorithms that prioritize high throughput over high precision. We introduce Polar Express, a new method for computing the polar decomposition. Like Newton-Schulz and other classical polynomial methods, our approach uses only matrix-matrix multiplications, making it very efficient on GPUs. Inspired by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the update rule at each iteration by solving a minimax optimization problem. We prove that this strategy minimizes error in a worst-case sense, allowing Polar Express to converge as rapidly as possible both in the early iterations and asymptotically. We also address finite-precision issues, making it practical to use in bfloat16. When integrated into Muon, our method yields consistent improvements in validation loss for a GPT-2 model trained on one to ten billion tokens from the FineWeb dataset, outperforming recent alternatives across a range of learning rates.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2505.16932>
- PDF: [[raw/papers/pdf/2025-polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon.pdf]]
- Raw markdown: [[raw/papers/md/2025-polar-express-optimal-matrix-sign-methods-and-their-application-to-the-muon]]
