---
arxiv: '2003.04560'
authors:
- Ronen Basri
- Meirav Galun
- Amnon Geifman
- David Jacobs
- Yoni Kasten
- Shira Kritchman
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2003.04560.md
raw_pdf: raw/papers/2003.04560.pdf
read: false
slug: frequency-bias-in-neural-networks-for-input-of-non-uniform
tags:
- theory
- generalization
- ml
- optimization
title: Frequency Bias in Neural Networks for Input of Non-Uniform Density
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2003.04560v1
venue: null
year: 2020
---

# Frequency Bias in Neural Networks for Input of Non-Uniform Density

> *Ronen Basri, Meirav Galun, Amnon Geifman…* — arXiv 2003.04560, 2020

## Abstract

Recent works have partly attributed the generalization ability of over-parameterized neural networks to frequency bias -- networks trained with gradient descent on data drawn from a uniform distribution find a low frequency fit before high frequency ones. As realistic training sets are not drawn from a uniform distribution, we here use the Neural Tangent Kernel (NTK) model to explore the effect of variable density on training dynamics. Our results, which combine analytic and empirical observations, show that when learning a pure harmonic function of frequency $κ$, convergence at a point $\x \in \Sphere^{d-1}$ occurs in time $O(κ^d/p(\x))$ where $p(\x)$ denotes the local density at $\x$. Specifically, for data in $\Sphere^1$ we analytically derive the eigenfunctions of the kernel associated with the NTK for two-layer networks. We further prove convergence results for deep, fully connected networks with respect to the spectral decomposition of the NTK. Our empirical study highlights similarities and differences between deep and shallow networks in this model.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2003.04560]]
- PDF: `raw/papers/2003.04560.pdf`
- arXiv: <http://arxiv.org/abs/2003.04560v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2020-basri-frequency-bias.md` before that tree was retired.*

## Core claim

The spectral-bias result of [@rahaman2019spectralbias] strengthens — and gains a quantitative density dependence — when the input distribution is non-uniform. The number of training iterations needed to fit a frequency-$\kappa$ component scales **inversely with the *minimum* density across the input space**, so non-uniform tabular features (which always have some sparse regions) are bottlenecked by their thinnest support, not their average density.

## Theorem 1 (informal statement)

For learning a pure sine $g(x) = \sin(\kappa x)$ on the circle to error $\delta$ via gradient descent on a depth-2 ReLU network, the iteration count satisfies

$$t = \tilde{O}\!\left(\frac{\kappa^2}{p_*}\right)$$

where $p_*$ is the **minimum density** of the input distribution $p(x)$. In $d$ dimensions for a frequency-$k$ harmonic the conjecture is $O(k^d / p_*)$ iterations.

## Mechanism — NTK eigenvalue argument

The convergence rate of each frequency component is set by the corresponding NTK eigenvalue. For uniform input the eigenvalues decay as $O(1/\kappa^d)$ (this recovers Rahaman's bound). For piecewise-constant non-uniform density $p(x)$:

- Eigenfunctions become piecewise sinusoids whose **local frequency scales as $\sqrt{p(x)}$** — high-frequency content is learned faster in dense regions and dramatically slower in sparse ones.
- Eigenvalues incorporate the density-weighted integral $\Psi(x) = \int \sqrt{p(\tilde{x})}\, d\tilde{x}$ via a $Z^2$ normalisation factor, where $Z$ is the total mass of $\sqrt{p}$.
- Convergence time at frequency $k$ in a region of density $p$ scales as $\kappa^2 / p$ — directly observed in their Figure 7 (density ratio 1:2:4 across three regions; convergence time scales linearly in $1/p$).

## Why it matters for §2.2 / tabular data

Real tabular features have non-uniform density by default — long tails, mode-concentration, target-conditioned imbalance, sparse high-cardinality categorical embeddings. Basri's result says the spectral-bias effect on tables is **worse than the uniform-input bound predicts**: the iterations needed to learn a sharp boundary somewhere in the feature space are set by the *least-populated* region the boundary cuts through.

Concretely, this is why MLPs on tabular data with rare-but-informative regions (the kind of `if-rare-flag-then-different-decision` rules trees handle natively) have a hard time even with infinite training: the iteration bound is divided by the local density of the rare regime.

## Caveats

- Theory is for shallow networks (depth-2 ReLU) and full-batch gradient descent.
- Conjecture extends the result to higher dimensions; the paper does not prove the $d$-dimensional generalisation rigorously, but provides empirical support.
- Result is asymptotic; practical training with finite data and stochastic optimisation can deviate.
