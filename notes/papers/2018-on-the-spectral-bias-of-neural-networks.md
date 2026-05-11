---
arxiv: '1806.08734'
authors:
- Nasim Rahaman
- Aristide Baratin
- Devansh Arpit
- Felix Draxler
- Min Lin
- Fred A. Hamprecht
- Yoshua Bengio
- Aaron Courville
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/1806.08734
  raw: '[[raw/papers/md/2018-on-the-spectral-bias-of-neural-networks]]'
  source: http://arxiv.org/abs/1806.08734v3
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2018-on-the-spectral-bias-of-neural-networks.md
raw_pdf: raw/papers/pdf/2018-on-the-spectral-bias-of-neural-networks.pdf
read: false
slug: on-the-spectral-bias-of-neural-networks
tags:
- type/paper
- theory
- generalization
- optimization
- ml
- status/stub
title: On the Spectral Bias of Neural Networks
type: note
updated: '2026-05-09'
venue: null
year: 2018
---

# On the Spectral Bias of Neural Networks

> *Nasim Rahaman, Aristide Baratin, Devansh Arpit…* — arXiv 1806.08734, 2018

## Abstract

Neural networks are known to be a class of highly expressive functions able to fit even random input-output mappings with $100\%$ accuracy. In this work, we present properties of neural networks that complement this aspect of expressivity. By using tools from Fourier analysis, we show that deep ReLU networks are biased towards low frequency functions, meaning that they cannot have local fluctuations without affecting their global behavior. Intuitively, this property is in line with the observation that over-parameterized networks find simple patterns that generalize across data samples. We also investigate how the shape of the data manifold affects expressivity by showing evidence that learning high frequencies gets \emph{easier} with increasing manifold complexity, and present a theoretical understanding of this behavior. Finally, we study the robustness of the frequency components with respect to parameter perturbation, to develop the intuition that the parameters must be finely tuned to express high frequency functions.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2018-on-the-spectral-bias-of-neural-networks]]
- PDF: [[raw/papers/pdf/2018-on-the-spectral-bias-of-neural-networks.pdf]]
- arXiv: <http://arxiv.org/abs/1806.08734v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2019-rahaman-spectral-bias.md` before that tree was retired.*

## Core claim

Trained-by-gradient-descent ReLU MLPs preferentially fit low-frequency Fourier components of a target function before high-frequency ones, regardless of their relative amplitudes. The bias is structural — it emerges from the Fourier spectrum of the function class itself and from a frequency-dependent gradient-descent convergence rate — not from optimisation artifacts that disappear with longer training.

## What they actually measure

- **Mathematical object.** Fourier coefficients $|\tilde{f}_\theta(\mathbf{k})|$ of the function the network represents at training step $t$.
- **Synthetic targets.** Superpositions of sinusoids on $[0,1]$, $\lambda(z) = \sum_i A_i \sin(2\pi k_i z + \phi_i)$, with parameterised frequencies $k_i$ and amplitudes $A_i$.
- **Experimental protocol.** 6-layer, 256-unit ReLU MLP, full-batch gradient descent on 200 equally-spaced samples; track normalised spectrum $|\tilde{f}_\theta(k_i)|/A_i$ throughout training.

## The two formal results

**Theorem 1 (Fourier spectrum bound).** For a ReLU network the Fourier components decay rationally:

$$\tilde{f}_\theta(\mathbf{k}) = \sum_{n=0}^{d} \frac{C_n(\theta,\mathbf{k}) \cdot \mathbf{1}_{H_n^\theta}(\mathbf{k})}{k^{n+1}}$$

with the numerator scaling with the Lipschitz constant $L_f$. High frequencies have smaller amplitude *bounds*; expressing them requires either large parameters or fine-grained activation patterns.

**Frequency-dependent convergence rate.** Under continuous gradient flow on the squared loss in Fourier space, the residual amplitude at frequency $\mathbf{k}$ contracts at rate

$$\left|\frac{d\tilde{h}(\mathbf{k})}{dt}\right| = \mathcal{O}(k^{-\Delta}), \quad 1 \le \Delta \le d,$$

so high-$k$ components evolve more slowly than low-$k$ ones for any fixed step size. Combined with the Lipschitz constant growing only gradually during training, this is what produces the empirical "low frequencies first" pattern.

## Headline empirical findings

1. **Synthetic sums of sines.** With equal or *increasing* amplitudes, low-frequency components reach the target amplitude first. The ordering is by frequency, not amplitude — the inductive bias dominates.
2. **MNIST with radial-noise targets.** Adding $\sin(k\|\mathbf{x}\|)$-type radial noise at varying $k$ to binary classification labels shows generalisation degrades under *low-frequency* noise but is robust to *high-frequency* noise — networks fit low frequencies first and treat high-frequency content as noise.
3. **Generalised frequency via RBF eigenfunctions.** Projecting onto Gaussian-RBF kernel eigenfunctions confirms the spectral bias persists outside the strict Fourier-basis setting.

## Scope and caveats

- Result is for ReLU (continuous piecewise-linear) networks; the rational-decay bound exploits CPWL structure. Generalisation to other activations is open.
- Synthetic experiments use simple targets (sinusoid superpositions); the MNIST/CIFAR-10 transfer is a "proof of concept" that rests on a manifold-hypothesis assumption.
- The Lipschitz growth is observed empirically across training, not formally guaranteed.
- Result is for full-batch gradient descent; behaviour under SGD with batch noise / momentum / adaptive optimizers is not covered.

## Why it matters for tabular

Tabular targets — `if age > 65 AND income < 30k AND has_prior_claim` — are piecewise-constant in the input space, hence high-frequency in any Fourier sense. The spectral-bias result predicts MLPs will smooth them out rather than reproduce the cliffs. Trees, by construction, *are* piecewise-constant. This is one of the three formal arguments §2 leans on for the inductive-bias gap between MLPs and trees on tabular data.

The mechanism cuts both ways: it predicts what fixes the gap. If the optimiser cannot reach high-frequency content, *injecting* high-frequency basis functions into the input — Fourier features [@tancik2020fourier], or the periodic / piecewise-linear numerical embeddings of [@gorishniy2022embeddings] — should bypass the bias. The §2.4 numerical-embeddings story rests on this prediction being borne out empirically.
