---
arxiv: null
authors: []
created: '2026-05-09'
doi: null
kind: paper
parser: docling
raw_md: raw/papers/md/2004-ng-feature-selection-l1.md
raw_pdf: raw/papers/pdf/2004-ng-feature-selection-l1.pdf
read: false
slug: 2004-ng-feature-selection-l1
tags:
- optimization
- generalization
- tabular
title: Feature Selection, L1 vs. L2 Regularization, and Rotational Invariance
type: note
updated: '2026-05-09'
url: null
venue: ICML 2004 (Banff, Canada)
year: 2004
---

# Feature Selection, L1 vs. L2 Regularization, and Rotational Invariance

*ICML 2004 (Banff, Canada), 2004*

## TL;DR

(stub — fill in after reading)

## Notes

(your synthesis)

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2004-ng-feature-selection-l1.md` before that tree was retired.*

## Core claim

Two matched results that bracket the sample complexity of feature selection by regulariser geometry:

- **L1-regularised logistic regression — upper bound (Theorem 3.1).** With at most $r$ relevant features (each weight bounded by $K$), with hold-out cross-validation over $B \in \{0, 1, 2, 4, \ldots, C\}$ on the L1 budget, the algorithm needs $m = \Omega\big((\log n)\cdot \mathrm{poly}(r, K, 1/\varepsilon, \log(1/\delta))\big)$ training examples to achieve excess logloss at most $\varepsilon$ with probability $1-\delta$. **Logarithmic in the input dimension $n$.**
- **Any rotationally invariant learner — lower bound (Theorem 4.3).** There exists a "trivial" learning problem — labels are a deterministic threshold $y = \mathbf{1}[x_1 \ge t]$ on a single coordinate — on which any rotationally invariant algorithm needs at least $m = \Omega(n/\varepsilon)$ training examples to attain $\varepsilon$ misclassification error. **Linear in the input dimension $n$.**

The contrast: with $K = n - r$ irrelevant features, L1 needs $\sim \log K$ examples while any rotationally invariant procedure needs $\sim K$ in the worst case. The gap is exponential and structural — not a constant-factor artefact.

## What "rotationally invariant" means (Definition 4.1)

A deterministic learning algorithm $L$ is **rotationally invariant** if for every training set $S = \{(x^{(i)}, y^{(i)})\}$, every rotation matrix $M \in \mathcal{M}$ (orthogonal with $|M|=1$), and every test point $x$:
$$
L[S](x) \;=\; L[MS](Mx),
$$
where $MS = \{(Mx^{(i)}, y^{(i)})\}$. The algorithm produces the *same predicted label* on $x$ when trained on $S$ as it does on $Mx$ when trained on the rotated dataset $MS$. For stochastic algorithms the same equality holds in distribution.

> **Caution against a common misstatement.** Rotational invariance does *not* mean $f(Mx) = f(x)$ for the trained classifier $f$ on a fixed dataset (which would be far stronger and obviously false for any non-trivial classifier). It means the algorithm-as-a-mapping from datasets to classifiers commutes with rotation: rotating the inputs and retraining produces a rotated copy of the original classifier.

The intuition Ng emphasises: a rotationally invariant algorithm "doesn't know which axis is which" — only the inner-product geometry of the inputs matters, not the choice of coordinate basis.

## Members of the rotationally invariant class (per Ng §4)

The lower bound applies to a wide family:
- **L2-regularised logistic regression** (proved in Proposition 4.2 via $R(\theta) = \theta^\top \theta = (M\theta)^\top(M\theta)$).
- **SVMs** with kernels that depend only on $x^\top x$, $x^\top z$, $z^\top z$ — linear, polynomial, RBF/Gaussian, and the L1 soft-margin variant (the per-example slack L1 doesn't break rotational invariance).
- **Multilayer neural networks trained by backpropagation** — under the technical assumption that weights are initialised from a spherically symmetric distribution (e.g. $\mathcal{N}(0, \varepsilon)$ per weight, independent of input dimension). Standard symmetric inits (Xavier/He) satisfy this.
- **Unregularised logistic regression** (where unique).
- **Perceptron.**
- **PCA / ICA preprocessing** followed by any algorithm — unless the preprocessing rescales each feature to unit variance.
- **Gaussian discriminant analysis** with a full covariance matrix.

Non-rotationally-invariant — and so escaping the lower bound:
- **L1-regularised logistic regression** (the algorithmic asymmetry behind the upper bound).
- **Naive Bayes.**
- **Decision trees making axis-aligned splits.**
- **Winnow** (Littlestone), **EG** (Kivinen & Warmuth).
- **Most feature-selection algorithms.**

## The lower bound's construction (sketch — Appendix B)

Pick the linear-threshold concept class $\mathcal{C} = \{\theta^\top x \ge \beta\}$ (VC dimension $n+1$). A standard PAC lower bound says some distribution / target $h^* \in \mathcal{C}$ requires $m = \Omega(n/\varepsilon)$ samples.

Now, by rotational invariance, any axis-aligned threshold problem (e.g. $y = \mathbf{1}[x_1 \ge t]$) is *just as hard* as any rotated version of it. Pick the rotation $M$ whose first row is $\theta^*$ — the bound transports: there exists at least one "axis-aligned-looking" problem on which $L$ needs $\Omega(n/\varepsilon)$ samples even though the true generating concept depends on only one feature. The orientation-finding step is what costs the bound.

The proof exploits Ehrenfeucht–Haussler–Kearns–Valiant's PAC lower bound and Vapnik's $n+1$ VC bound on linear threshold concepts.

## The L1 upper bound's construction (sketch — Appendix A)

Bartlett (1998) and Zhang (2002) covering-number bounds for the L1-norm-bounded linear function class: for $\mathcal{G} = \{x \mapsto \theta^\top x : \|\theta\|_1 \le B\}$ with inputs in $[-1,1]^n$, the L1 covering number $\mathcal{N}_1(\mathcal{G}, \varepsilon, m)$ depends on $\log n$, $B$, $1/\varepsilon$, and $\log m$ — polynomially in $B$ and $1/\varepsilon$, **logarithmically in $n$**.

Fitting the budget $\hat B$ in the cross-validation loop satisfies $rK \le \hat B \le 2rK$ (where $K$ here is Ng's bound on individual weights, not the irrelevant count — note the notation clash with §2.1 of the post). Logloss is Lipschitz in the model output, so the covering bound on $\mathcal{G}$ transfers to the loss class. Standard uniform convergence then closes the argument.

## Empirical experiments (§5)

A 1-relevant / many-irrelevant logistic regression problem (inputs from a multivariate normal, target a logistic function of one coordinate, $\theta_1 = 10$) confirms the asymptotic story:
- With 100 training examples and a 1000-dimensional input, **L1-LR remains near-perfect**; **L2-LR's misclassification rate climbs to 0.5** (random-chance) as $n$ grows.
- Repeated with three relevant features and with an exponential-decay relevance pattern; same direction in all cases.
- Repeated with 200 training examples; same direction.

The empirical figure is a clean visual demonstration of the gap: the L2 curve ramps toward chance as the irrelevant count grows, while L1 stays flat.

## SVM / margin reconciliation (§4 Remark)

An apparent tension: SVMs work in infinite-dimensional kernel spaces under large-margin guarantees, yet Theorem 4.3 says they're as bad as anyone else when irrelevant features are added. The reconciliation: SVM generalisation depends on **margin / diameter**, not margin alone. Adding irrelevant Gaussian features doesn't shrink the margin, but it **inflates the diameter** of the data in $L_2$, so the ratio degrades and the generalisation bound loosens. (Vapnik 1998.)

## Why it matters for §2.1 of Chapter 2

- This is the *only* sample-complexity-style theorem that directly distinguishes axis-aligned learners from rotationally invariant ones on irrelevant-feature-laden data — exactly the regime real production tabular schemas sit in (informative-only-conditionally columns, weak features, denormalisation noise).
- The bound makes precise what §2.1 calls "the cost of not knowing which axis is which": the orientation-finding step is the $\Omega(K)$ tax, and trees skip it entirely because the column index is given.
- Direct empirical operationalisation comes from Grinsztajn et al. (2022) — they apply random orthogonal rotations to a 45-dataset benchmark and observe the predicted reversal: under rotation, NNs catch up to trees (because rotation is neutral to NNs and damaging to trees). See [[2022-grinsztajn-tree-outperform]] §4.2.

## Caveats / things to watch when citing

- **The upper bound is for L1-regularised logistic regression with hold-out cross-validation**, not L1-LR per se. Different selection rules can change the bound.
- **The lower bound is worst-case over distributions** — it shows there exist tabular distributions where the gap is at least linear, not that every tabular dataset puts MLPs in this regime. Real datasets may live well inside the worst case.
- **MLP membership in the rotationally invariant class** depends on (i) spherically symmetric weight init (Xavier, He, $\mathcal{N}(0, \sigma^2)$ all qualify), and (ii) rotation-symmetric loss + L2 weight decay. If any component breaks rotation symmetry — e.g. per-coordinate weight decay, group L1 over input weights, batch normalisation behaving asymmetrically across input dimensions — the bound technically does not apply, though the spirit usually does.
- **Notation clash with §2.1 of Chapter 2.** Ng uses $K$ for the bound on individual $|\theta_i|$ in Theorem 3.1, and $r$ for the count of relevant features. The post (and Grinsztajn) use $K$ for the count of irrelevant features. Translating: the post's $K$ is Ng's $n - r$.
