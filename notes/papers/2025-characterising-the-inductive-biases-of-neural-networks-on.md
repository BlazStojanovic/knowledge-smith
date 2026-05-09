---
arxiv: '2505.24060'
authors:
- Chris Mingard
- Lukas Seier
- Niclas Göring
- Andrei-Vlad Badelita
- Charles London
- Ard Louis
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2025-characterising-the-inductive-biases-of-neural-networks-on.md
raw_pdf: raw/papers/pdf/2025-characterising-the-inductive-biases-of-neural-networks-on.pdf
read: false
slug: characterising-the-inductive-biases-of-neural-networks-on
tags:
- theory
- generalization
- interpretability
- ml
title: Characterising the Inductive Biases of Neural Networks on Boolean Data
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2505.24060v1
venue: null
year: 2025
---

# Characterising the Inductive Biases of Neural Networks on Boolean Data

> *Chris Mingard, Lukas Seier, Niclas Göring…* — arXiv 2505.24060, 2025

## Abstract

Deep neural networks are renowned for their ability to generalise well across diverse tasks, even when heavily overparameterized. Existing works offer only partial explanations (for example, the NTK-based task-model alignment explanation neglects feature learning). Here, we provide an end-to-end, analytically tractable case study that links a network's inductive prior, its training dynamics including feature learning, and its eventual generalisation. Specifically, we exploit the one-to-one correspondence between depth-2 discrete fully connected networks and disjunctive normal form (DNF) formulas by training on Boolean functions. Under a Monte Carlo learning algorithm, our model exhibits predictable training dynamics and the emergence of interpretable features. This framework allows us to trace, in detail, how inductive bias and feature formation drive generalisation.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2025-characterising-the-inductive-biases-of-neural-networks-on]]
- PDF: [[raw/papers/pdf/2025-characterising-the-inductive-biases-of-neural-networks-on.pdf]]
- arXiv: <http://arxiv.org/abs/2505.24060v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2025-mingard-boolean-inductive-bias.md` before that tree was retired.*

## Core claim

Depth-2 fully connected networks on Boolean data have an analytically tractable inductive prior that assigns exponentially smaller probability to high-complexity Boolean functions:

$$P(f) \lesssim 2^{-K(f) + O(1)}$$

where $P(f)$ is the prior probability a random initialisation computes function $f$ and $K(f)$ is the DNF complexity (minimum literal count). Specific scaling laws (Table 2):

- Constant functions: $P(f) \approx 1$.
- $t$-entropy functions: $P(f) \sim 2^{-O(t (4/3)^n)}$.
- $k$-parity functions: $P(f) \sim 2^{-\Theta(k 2^{n-1})}$ — exponentially suppressed in the input dimension.

This makes high-degree / high-frequency Boolean targets exponentially rare under the network's prior — a discrete analogue of the continuous spectral-bias story in [@rahaman2019spectralbias] and [@basri2020frequency].

## Setup — and an important caveat

- Networks are **discretised**: weights live in $\{-1, 0, 1\}$. Inputs and outputs are Boolean.
- Training uses **Markov-chain Monte Carlo (Metropolis-Hastings)** with a 0–1 likelihood, or a discrete greedy local search. **Not standard SGD on continuous weights.**
- One-to-one correspondence between depth-2 DFCNs (their term) and DNF formulas (Proposition 2.7) is what makes the analysis tractable.
- Experiments scale only to $n \le 7$ Boolean inputs (computational tractability ceiling).
- Authors' own caveat: *"our training algorithms do not capture all properties of continuous optimisation with SGD."*

## Why it matters for §2.2

Mingard et al. is *not* a direct claim that standard SGD-trained MLPs concentrate on low-frequency / low-degree Boolean functions. It is an analytic case study showing that the **prior** (the distribution over functions induced by random initialisation) of a discrete depth-2 network has this concentration; whether continuous SGD inherits the bias is left open.

The right way to cite this in §2.2: as a *complementary* discrete-side result that strengthens the spectral-bias picture established empirically by Rahaman and theoretically by Basri on continuous inputs — not as an independent proof that real MLPs do this. The §2 outline currently says *"give an analytic treatment on Boolean data: the MLP prior concentrates on low-degree / low-frequency solutions as a function of input dimension"* — that's directionally right but should be qualified to "depth-2 discrete networks under MCMC training" before the prose lands.

## What the post should cite Mingard for

- The clean *parity* scaling — $k$-parity is $2^{-\Theta(k 2^{n-1})}$ rare under the prior. Parity is the canonical example of a high-degree Boolean function trees can represent at depth $k$ but MLPs notoriously struggle with; Mingard makes the prior-side cost quantitative.
- Evidence that the simplicity-bias / spectral-bias story has a discrete complement, so it isn't an artefact of the continuous Fourier basis.
