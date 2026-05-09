---
arxiv: '2006.10739'
authors:
- Matthew Tancik
- Pratul P. Srinivasan
- Ben Mildenhall
- Sara Fridovich-Keil
- Nithin Raghavan
- Utkarsh Singhal
- Ravi Ramamoorthi
- Jonathan T. Barron
- Ren Ng
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2006.10739.md
raw_pdf: raw/papers/2006.10739.pdf
read: false
slug: fourier-features-let-networks-learn-high-frequency
tags:
- vision
- ml
- theory
- optimization
title: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
  Domains
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2006.10739v1
venue: null
year: 2020
---

# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

> *Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall…* — arXiv 2006.10739, 2020

## Abstract

We show that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains. These results shed light on recent advances in computer vision and graphics that achieve state-of-the-art results by using MLPs to represent complex 3D objects and scenes. Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP fails to learn high frequencies both in theory and in practice. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth. We suggest an approach for selecting problem-specific Fourier features that greatly improves the performance of MLPs for low-dimensional regression tasks relevant to the computer vision and graphics communities.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2006.10739]]
- PDF: `raw/papers/2006.10739.pdf`
- arXiv: <http://arxiv.org/abs/2006.10739v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-tancik-fourier-features.md` before that tree was retired.*

## Core claim

A coordinate-MLP cannot fit high-frequency content in its input — the manifestation of [@rahaman2019spectralbias]'s spectral bias for low-dimensional regression problems. Replacing the raw input $\mathbf{v}$ with a Fourier-feature mapping $\gamma(\mathbf{v})$ before the MLP shifts the NTK eigenvalue spectrum so high-frequency components become learnable. The fix is at the input, not in the network.

## The Fourier-feature mapping

$$\gamma(\mathbf{v}) = \big[a_1 \cos(2\pi \mathbf{b}_1^\top \mathbf{v}), a_1 \sin(2\pi \mathbf{b}_1^\top \mathbf{v}), \ldots, a_m \cos(2\pi \mathbf{b}_m^\top \mathbf{v}), a_m \sin(2\pi \mathbf{b}_m^\top \mathbf{v})\big]^\top$$

For the recommended "Gaussian" variant: $\mathbf{B} \in \mathbb{R}^{m \times d}$ has entries i.i.d. $\mathcal{N}(0, \sigma^2)$, with $\sigma$ tuned per task; amplitudes $a_j = 1$. The single hyperparameter that matters is $\sigma$ (kernel bandwidth).

## NTK-based mechanism

- **Without Fourier features.** A standard MLP's NTK has eigenvalues that decay rapidly with frequency. Gradient flow on the squared loss converges component-wise at rate $\eta \lambda_i$, so high-frequency eigenfunctions converge exponentially more slowly — exactly the spectral bias.
- **With Fourier features.** The composed NTK becomes stationary (shift-invariant): $h_{\text{NTK}}(h_\gamma(\mathbf{v}_1 - \mathbf{v}_2))$, with effective frequency support tunable via $\sigma$. Narrow $\sigma$ → underfitting (kernel concentrated at low frequencies); wide $\sigma$ → aliasing / overfitting. Optimal $\sigma$ broadens the support to cover task-relevant frequencies.

## Headline empirical result

2D natural-image regression, 4–8 layer MLPs, 256 channels, 256 Fourier features:
- **No mapping**: 19.32 PSNR
- **Gaussian RFF**: 25.57 PSNR (+6.25 dB ≈ ~40% relative error reduction)

Five regression tasks total: 2D image regression, 3D shape occupancy, 2D CT reconstruction, 3D MRI from Fourier samples, novel-view synthesis (NeRF). The fix transfers across all of them.

## Why it matters for §2 of post 1

Tancik et al. is the *mechanism paper* for the §2.4 numerical-embeddings result. The story:

1. §2.2 establishes the spectral-bias problem (Rahaman, Basri).
2. Tancik shows in a different domain (implicit neural representations, coordinate regression) that the bias is fixable by injecting high-frequency basis functions on the input side, with NTK eigenvalue analysis explaining why.
3. [@gorishniy2022embeddings] adapts the same idea to tabular data via periodic and piecewise-linear numerical embeddings — and reports most of the MLP↔Transformer gap closes.

The connection is conceptual rather than direct: Gorishniy's periodic embeddings differ from Tancik's in two ways (per-feature, no mixing; trained coefficients rather than fixed). And Gorishniy themselves are careful: *"it is still to be explained how exactly the discussed embedding modules help optimization on the fundamental level."* So Tancik is the inspiration and the spectral-bias narrative; the tabular adaptation has its own empirical-validation story.
