---
arxiv: '2601.19156'
authors:
- Gyu Yeol Kim
- Min-hwan Oh
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: Convergence of Muon with Newton-Schulz
url: https://arxiv.org/abs/2601.19156
year: 2026
---

# Convergence of Muon with Newton–Schulz

Gyu Yeol Kim
  
Seoul National University
  
Seoul, South Korea
  
gyuyeolkim@snu.ac.kr
  
&Min-hwan Oh
  
Seoul National University
  
Seoul, South Korea
  
minoh@snu.ac.kr

###### Abstract

We analyze Muon as originally proposed and used in practice—using the momentum orthogonalization with a few Newton–Schulz steps.
The prior theoretical results replace this key step in Muon with an exact SVD-based polar factor.
We prove that Muon with Newton–Schulz converges to a stationary point at the same rate as the SVD-polar idealization, up to a constant factor for a given number qq of Newton–Schulz steps.
We further analyze this constant factor and prove that it converges to 1 doubly exponentially in qq and improves with
the degree of the polynomial used in Newton–Schulz for approximating the orthogonalization direction.
We also prove that Muon removes the typical square-root-of-rank loss compared to its vector-based counterpart, SGD with momentum.
Our results explain why Muon with a few low-degree Newton–Schulz steps matches exact-polar (SVD) behavior at a much faster wall-clock time
and explain how much momentum matrix orthogonalization via Newton–Schulz benefits over the vector-based optimizer.
Overall, our theory justifies the practical Newton–Schulz design of Muon, narrowing its practice–theory gap.

## 1 Introduction

Modern deep neural networks comprise billions of parameters and demand highly efficient training procedures.
A persistent challenge is that most widely used optimizers—such as stochastic gradient descent (SGD) (Robbins and Monro, [1951](#bib.bib31 "A stochastic approximation method")) and adaptive methods such as Adam (Kingma and Ba, [2015](#bib.bib27 "Adam: A method for stochastic optimization"))—operate on *vectorized* parameters, thereby discarding the native *matrix* structure present in linear layers and attention projections.
Optimizers that explicitly respect matrix structure can, in principle, yield search directions that are better aligned with the underlying geometry while remaining computationally efficient at scale.

Muon (Jordan et al., [2024](#bib.bib1 "Muon: an optimizer for hidden layers in neural networks")) is an optimizer designed for matrix-structured parameters.
At each iteration, instead of following the raw momentum, Muon *orthogonalizes* the mntum matrix and then uses this orthogonalized direction to update the weights.
In practice, this orthogonalization is not computed via an exact singular value decomposition (SVD)—which is accurate but expensive—but is approximated efficiently by a small, fixed number of Newton–Schulz steps.
Empirical studies (Jordan et al., [2024](#bib.bib1 "Muon: an optimizer for hidden layers in neural networks"); Liu et al., [2025a](#bib.bib2 "Muon is scalable for llm training")) have reported strong performance at scale with this SVD-free implementation, making Muon an attractive alternative to vector-based optimizers.
Despite recent attempts to analyze the convergence of Muon (Shen et al., [2025](#bib.bib6 "On the convergence analysis of muon"); Li and Hong, [2025](#bib.bib7 "A note on the convergence of muon and further"); Sato et al., [2025](#bib.bib17 "Analysis of muon’s convergence and critical batch size")), theory still lags behind practice.
Existing analyses typically study an *idealized* variant that replaces the Newton–Schulz step—central to practical Muon—with an exact polar step computed by SVD for analytical convenience. This leaves open whether the *actual* SVD-free orthogonalization used in practice—i.e., a finite number of Newton–Schulz steps—admits principled nonconvex convergence guarantees, and how the Newton–Schulz approximation impacts rank dependence and efficiency.
Therefore, the following research questions remain open:

Research questions.

* •

  Does Muon with Newton–Schulz admit nonconvex convergence guarantees,
  and how do its rates compare to the exact SVD–polar idealization?
* •

  How does the Newton–Schulz steps qq and Newton–Schulz polynomial degree κ\kappa control the accuracy–compute trade‑off?
  In particular, how large is the gap caused by using Newton–Schulz, and how does that gap become negligible?
* •

  Can we show that Muon converges faster than the vector-counterpart, SGD with momentum?
  What geometric mechanisms and rank dependence drive this gap?

To address these questions,
we analyze Muon as originally proposed (Jordan et al., [2024](#bib.bib1 "Muon: an optimizer for hidden layers in neural networks")) and as used in practice:
Muon with momentum orthogonalization computed via Newton–Schulz.
For nonconvex objectives, under standard smoothness assumptions,
we prove the convergence of Muon to a stationary point, measured by the nuclear norm of the gradient.
We establish that the convergence rate in the number of iterations matches the idealized (but not used in practice) SVD‑based exact polar variant,
up to a constant factor that depends on the polar approximation error εq\varepsilon\_{q} (defined in Definition [3](#Thmdefinition3 "Definition 3 (Orthogonality residual and polar approximation error). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")) for the fixed number of Newton–Schulz steps qq.
Moreover, we show that the polar approximation error εq\varepsilon\_{q} shrinks doubly exponentially as qq grows and decays with larger κ\kappa,
which is the degree of the polynomial used in Newton–Schulz steps.
Recursive updates using this polynomial allow the optimizer to find the approximated orthogonalization direction of the momentum matrix.
Hence, with only a few Newton–Schulz steps, the convergence rate of Muon quickly tends to the convergence rate of the exact polar step via SVD.
Consequently, our results imply that, because a few Newton–Schulz steps are far cheaper per iteration than SVD,
the practical Muon implementation with Newton–Schulz attains substantially faster wall-clock convergence.

Our main contributions are summarized as follows:

* •

  The first convergence result of Muon with Newton–Schulz.
  To our knowledge, we present the first nonconvex convergence guarantees for Muon with a finite number of Newton–Schulz steps (Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")),
  as originally proposed and practically used.
  It is important to note that even for convex optimization, the convergence of Muon with Newton–Schulz has not been shown previously.
  The key distinction from the existing analyses of Muon is that we do not replace Newton–Schulz in the original Muon with the exact polar computed by SVD.
* •

  Analysis of polar approximation error and wall-clock convergence.
  We prove that the polar approximation error εq\varepsilon\_{q} due to using Newton–Schulz instead of SVD\mathrm{SVD},
  in the Muon optimizer, decays doubly exponentially with the number of Newton–Schulz steps qq and decays with the degree κ\kappa of the polynomial
  required to approximate the orthogonalization direction of the momentum matrix (Definition [2](#Thmdefinition2 "Definition 2 (Newton–Schulz polynomial). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")).
  Thus, even with a few steps of Newton–Schulz, the convergence of Muon with Newton–Schulz becomes arbitrarily close to that of the SVD-variant in the number of iterations (Theorems [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).
  Hence, given that per-iteration computation is much more efficient for Newton–Schulz steps compared to SVD,
  the overall convergence in wall-clock time is much faster for Muon with Newton–Schulz.
* •

  Sharper rank dependence in Muon with Newton–Schulz.
  To prove the comparative advantage of
  Muon against the vector-based counterpart,
  we demonstrate for the first time that Muon with Newton–Schulz sharpens the convergence rate by a factor of the square root of the rank of the momentum matrix (see Table [1](#S4.T1 "Table 1 ‣ 4.3 Proof Sketch for Theorem 1. ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).

## 2 Related work

Muon and momentum orthogonalization.
Jordan et al. ([2024](#bib.bib1 "Muon: an optimizer for hidden layers in neural networks")) introduced Muon,
which orthogonalizes a momentum matrix via a few Newton–Schulz steps (SVD-free),
and reported strong empirical results at LLM scale (Liu et al., [2025a](#bib.bib2 "Muon is scalable for llm training")).
Earlier work orthogonalized gradients by SVD before applying momentum (Orthogonal-SGDM; Tuddenham et al. ([2022](#bib.bib15 "Orthogonalising gradients to speed up neural network optimisation"))),
whereas Muon applies momentum before orthogonalization and replaces SVD with Newton–Schulz,
resulting in faster computation with only matrix multiplications.
More details about Muon are described in [3.3](#S3.SS3 "3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").

Second-order preconditioners vs. Muon.
Matrix-aware optimizers such as Shampoo (Gupta et al., [2018](#bib.bib18 "Shampoo: preconditioned stochastic tensor optimization")) SOAP (Vyas et al., [2024](#bib.bib19 "Soap: improving and stabilizing shampoo using adam")) and their variants (An et al., [2025](#bib.bib3 "Asgo: adaptive structured gradient optimization")) are *second-order preconditioners*:
they maintain layerwise curvature (Kronecker-factored second moments) and periodically apply inverse-root preconditioning.
By contrast, *Muon is not a second-order method*;
it neither estimates nor inverts curvature but orthogonalizes the momentum matrix via a few Newton–Schulz iterations (SVD-free, using only matrix multiplications).
These methods target different mechanisms—curvature preconditioning vs. projection/normalization—and can be complementary rather than directly comparable.

Practical efficiency of Muon.
Large-scale training reports for Muon (Liu et al., [2025a](#bib.bib2 "Muon is scalable for llm training"); Shah et al., [2025](#bib.bib8 "Practical efficiency of muon for pretraining"); Tveit et al., [2025](#bib.bib9 "Muon optimizer accelerates grokking")) and communication/memory-aware variants (Ahn and Dion, [2025](#bib.bib5 "A communication-efficient optimizer for large models"); Liu et al., [2025b](#bib.bib4 "Cosmos: a hybrid adaptive optimizer for memory-efficient training of llms")) motivate a theory that is SVD-free and GPU-aligned.
Our analysis of Newton–Schulz, a key step in the Muon optimizer, adopts precisely that stance.
The analysis provided in this work can be adapted to other variants of Muon.

Convergence analysis of Muon.
Several recent analyses examine Muon but typically either idealize the orthogonalization by assuming an exact SVD polar step (Shen et al., [2025](#bib.bib6 "On the convergence analysis of muon")),
work under Frobenius-smoothness with dimension-driven constants (Li and Hong, [2025](#bib.bib7 "A note on the convergence of muon and further")),
focus on stability/variant phenomena (Sato et al., [2025](#bib.bib17 "Analysis of muon’s convergence and critical batch size")),
or offer complementary lenses (steepest descent under norms, trust-region views, implicit constraints, or LMO/Frank-Wolfe formulations)
without addressing Newton–Schulz step accuracy in nonconvex rates (Bernstein and Newhouse, [2024](#bib.bib12 "Old optimizer, new norm: an anthology"); Kovalev, [2025](#bib.bib13 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization"); Chen et al., [2025](#bib.bib11 "Muon optimizes under spectral norm constraints"); Riabinin et al., [2025](#bib.bib10 "Gluon: making muon & scion great again!(bridging theory and practice of lmo-based optimizers for llms)")).

Key distinctions compared to the existing analyses of Muon.
Prior work either assumes exact SVD polar steps,
measures progress in a geometry that obscures rank benefits,
or does not quantify the approximation from Newton–Schulz in Muon.
Our work is the first to analyze how two key parameters—the number of Newton–Schulz steps and the degree of the polynomial used for finding the orthogonalization direction of the momentum matrix approximately—affect the convergence rate of Muon.
The results indicate a nonconvex convergence rate to a stationary point,
an explicit and rapidly vanishing constant factor derived from Newton–Schulz instead of SVD,
and sharper rank dependence than SGD with momentum under the same metric.
Overall, our work explains why Muon converges quickly (particularly in wall-clock time) as well as why only a few steps of Newton–Schulz suffice in practice.

## 3 Preliminaries

### 3.1 Notations

For matrix X∈ℝm×nX\in\mathbb{R}^{m\times n}, X⊤X^{\top} is its transpose.
For X=(xi​j)∈ℝn×nX=(x\_{ij})\in\mathbb{R}^{n\times n}, tr⁡(X):=∑i=1nxi​i\operatorname{tr}(X):=\sum\_{i=1}^{n}x\_{ii}.
We write ‖X‖∗\|X\|\_{\*}, ‖X‖op\|X\|\_{\mathrm{op}}, and ‖X‖F\|X\|\_{F} for the nuclear, spectral (operator), and Frobenius norms, respectively,
and ⟨X,Y⟩F:=tr⁡(X⊤​Y)\langle X,Y\rangle\_{F}:=\operatorname{tr}(X^{\top}Y).
For a thin SVD X=U​Σ​V⊤X=U\Sigma V^{\top}, the *polar factor* is Polar⁡(X):=U​V⊤\operatorname{Polar}(X):=UV^{\top}, a partial isometry with
‖Polar⁡(X)‖op≤1\|\operatorname{Polar}(X)\|\_{\mathrm{op}}\leq 1 and ⟨X,Polar⁡(X)⟩F=‖X‖∗\langle X,\operatorname{Polar}(X)\rangle\_{F}=\|X\|\_{\*}
(See Appendix [A.1](#A1.SS1 "A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).
For two sequences {an}n=1∞\{a\_{n}\}\_{n=1}^{\infty} and {bn}n=1∞\{b\_{n}\}\_{n=1}^{\infty},
an=𝒪​(bn)a\_{n}=\mathcal{O}(b\_{n}) implies that there exists a constant C>0C>0 such that an≤C​bna\_{n}\leq Cb\_{n} holds for all n≥1n\geq 1.
We use 𝔼​[⋅]\mathbb{E}[\cdot] for expectations over all algorithmic randomness.

### 3.2 Problem Setting: Nonconvex Optimization

We consider the stochastic optimization of a matrix-valued parameter:
W∈ℝm×nW\in\mathbb{R}^{m\times n}

|  |  |  |
| --- | --- | --- |
|  | minW∈ℝm×n⁡f​(W)=𝔼ξ​[f​(W;ξ)],\displaystyle\min\_{W\in\mathbb{R}^{m\times n}}f(W)\;=\;\mathbb{E}\_{\xi}[f(W;\xi)], |  |

where the objective ff is nonconvex with f∗:=infWf​(W)>−∞f^{\*}:=\inf\_{W}f(W)>-\infty.
We denote by r:=min⁡{m,n}r:=\min\{m,n\} the maximal possible rank of the matrix.
At iteration tt,
a mini-batch ξt=(ξt,1,…,ξt,B)\xi\_{t}=(\xi\_{t,1},\ldots,\xi\_{t,B}) is drawn with {ξt,i}i\{\xi\_{t,i}\}\_{i} i.i.d., and {ξt}t\{\xi\_{t}\}\_{t} are independent across t=1,…,Tt=1,\ldots,T.
At t=0t=0, the model parameter W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n} is initialized, and
we define the initial sub-optimality
as D:=f​(W0)−f∗D:=f(W\_{0})-f^{\*}.
In this paper, the following assumptions are made for the convergence analysis:

###### Assumption 1 (Lipschitz smoothness).

The objective function f:ℝm×n→ℝf:\mathbb{R}^{m\times n}\to\mathbb{R} is continuously differentiable and LL-Lipschitz smooth, i.e., for all X,Y∈ℝm×nX,Y\in\mathbb{R}^{m\times n},

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(X)−∇f​(Y)‖∗≤L​‖X−Y‖op.\displaystyle\|\nabla f(X)-\nabla f(Y)\|\_{\*}\;\leq\;L\,\|X-Y\|\_{\mathrm{op}}. |  |

We use smoothness with respect to the operator norm (and the nuclear norm as its dual).

###### Assumption 2 (Bounded variance).

We assume ∇f​(W;ξ)\nabla f(W;\xi) is an unbiased stochastic estimator of the true gradient ∇f​(W)\nabla f(W) and has bounded variance for all WW and a single sample ξ\xi, i.e.

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[∇f​(W;ξ)]=∇f​(W),𝔼​[‖∇f​(W;ξ)−∇f​(W)‖F2]≤σ2.\displaystyle\mathbb{E}[\nabla f(W;\xi)]=\nabla f(W),\qquad\mathbb{E}\big[\|\nabla f(W;\xi)-\nabla f(W)\|\_{F}^{2}\big]\leq\sigma^{2}. |  |

For a mini-batch of size BB, the variance is at most σ2/B\sigma^{2}/B.

Assumptions [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") are standard for analyzing first-order methods in stochastic optimization (Boyd and Vandenberghe, [2004](#bib.bib34 "Convex optimization"); Nemirovski et al., [2009](#bib.bib32 "Robust stochastic approximation approach to stochastic programming"); Candes and Recht, [2012](#bib.bib35 "Exact matrix completion via convex optimization"); Shapiro et al., [2021](#bib.bib33 "Lectures on stochastic programming: modeling and theory"); Reddi et al., [2018](#bib.bib28 "On the convergence of adam and beyond"); Zou et al., [2019](#bib.bib36 "A sufficient condition for convergences of adam and rmsprop")).
LL‑smoothness is typically defined using a norm and its dual,
and the specific operator-nuclear geometry chosen here is a natural fit when working with matrix parameters and polar or orthogonal updates  (Nesterov, [2013](#bib.bib22 "Introductory lectures on convex optimization: a basic course"); Beck, [2017](#bib.bib23 "First-order methods in optimization"); Jaggi, [2013](#bib.bib24 "Revisiting frank-wolfe: projection-free sparse convex optimization")).
The bounded-variance assumption, where the mini-batch variance is σ2/B\sigma^{2}/B,
is a foundational concept for algorithms like SGD in both convex and nonconvex scenarios
 (Bottou et al., [2018](#bib.bib25 "Optimization methods for large-scale machine learning"); Ghadimi and Lan, [2013](#bib.bib26 "Stochastic first-and zeroth-order methods for nonconvex stochastic programming")).

In this paper, the metric for the convergence rate is defined as follows:

###### Definition 1 (ϵ\epsilon-stationary point).

We call W∈ℝm×nW\in\mathbb{R}^{m\times n} an ϵ\epsilon-stationary point (in the nuclear norm)
if 𝔼​[‖∇f​(W)‖∗]≤ϵ\mathbb{E}[\|\nabla f(W)\|\_{\*}]\leq\epsilon.
Equivalently, we say an algorithm attains ϵ\epsilon-stationarity in TT steps if

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]≤ϵ.\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}]\leq\epsilon. |  |

Note that when working with functions that have matrix inputs,
using the nuclear norm to find a stationary point
provides a more restrictive and precise condition than using the standard Frobenius norm.
In other words, if a point satisfies the stationarity condition for the nuclear norm,
it is guaranteed to also satisfy the condition for the Frobenius norm.

### 3.3 Muon Algorithm and Newton–Schulz Orthogonalization

Algorithm 1  Muon (with the illustration of Newton–Schulz orthogonalization)

0: learning rate η>0\eta>0, momentum β∈[0,1)\beta\in[0,1), Newton–Schulz steps q∈ℕq\in\mathbb{N},
Newton–Schulz polynomial pκp\_{\kappa} (degree κ)\kappa), batch size BB, total iteration TT.

1: Initialize: M0←0M\_{0}\leftarrow 0, W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n}

2: for t=1t=1 to TT do

3:  Gt←1B​∑i=1B∇f​(Wt−1;ξt,i)G\_{t}\leftarrow\frac{1}{B}\sum\_{i=1}^{B}\nabla f(W\_{t-1};\xi\_{t,i})
⊳\triangleright Compute (batch) gradients

4:  Mt←β​Mt−1+GtM\_{t}\leftarrow\beta M\_{t-1}+G\_{t}

5:  Xt,0←Mt/αtX\_{t,0}\leftarrow M\_{t}/\alpha\_{t} with αt=max⁡{1,‖Mt‖F}\alpha\_{t}=\max\{1,\|M\_{t}\|\_{F}\}
⊳\triangleright Pre-Newton–Schulz scaling

6:  for j=1j=1 to qq do

7:   Xt,j←pκ​(Xt,j−1​Xt,j−1⊤)​Xt,j−1X\_{t,j}\leftarrow p\_{\kappa}(X\_{t,j-1}X\_{t,j-1}^{\top})\,X\_{t,j-1} ⊳\triangleright Newton–Schulz steps (Lines 6-9)

8:  end for

9:  Ot←Xt,qO\_{t}\leftarrow X\_{t,q}

10:  Wt←Wt−1−η​OtW\_{t}\leftarrow W\_{t-1}-\eta O\_{t}
⊳\triangleright Update parameters

11: end for

For clarity of exposition, we present pseudocode for Muon with an explicit illustration of the Newton–Schulz steps (Lines 5–9) in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
Note that this is not a new algorithm; it is the original method of Jordan et al. ([2024](#bib.bib1 "Muon: an optimizer for hidden layers in neural networks")),
here written in a more general mini-batch form with a step-by-step illustration of Newton–Schulz.
Rather than orthogonalizing via SVD,
Muon approximates the orthogonalization direction using only matrix multiplications;
the key mechanism enabling this is the Newton–Schulz-based orthogonalization.

The key advantages of using Newton–Schulz for orthogonalization are as follows:

* •

  Newton–Schulz makes Muon inversion-free and SVD-free.
  SVD is computationally expensive and makes each iteration costly.
  In contrast, the Newton–Schulz approach relies solely on matrix multiplications,
  yielding substantially better per-iteration efficiency—especially for large parameter matrices.
* •

  Newton–Schulz is an iterative method that allows for precise control over the degree of orthogonality by adjusting the number of iterations.
  The number of Newton–Schulz steps provides a direct trade-off between computational cost and the degree of orthogonality, offering valuable flexibility.

### 3.4 Newton–Schulz polynomial

In the Muon algorithm,
at every iteration, a scaled momentum matrix XX is orthogonalized via Newton–Schulz steps.
First, the matrix X​X⊤XX^{\top} is formed and is then passed to a polynomial function pκp\_{\kappa} with degree κ\kappa.
Recursive updates by this polynomial make the matrix XX nearly orthogonal, i.e., X​X⊤=IXX^{\top}=I.
We first define this function pκp\_{\kappa} in Definition [2](#Thmdefinition2 "Definition 2 (Newton–Schulz polynomial). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")
and state the properties of this polynomial used in Newton–Schulz.

###### Definition 2 (Newton–Schulz polynomial).

For degree κ∈ℕ\kappa\in\mathbb{N}, the Newton–Schulz polynomial is
the Taylor truncation of 1/λ1/\sqrt{\lambda} at λ=1\lambda=1, i.e.,

|  |  |  |
| --- | --- | --- |
|  | p(s)​(1)=dsd​λs​λ−1/2|λ=1\displaystyle p^{(s)}(1)=\frac{d^{s}}{d\lambda^{s}}\lambda^{-1/2}\Bigg|\_{\lambda=1} |  |

for s=1,…,κs=1,\ldots,\kappa.
The explicit form of the Newton–Schulz polynomial for degree κ\kappa is

|  |  |  |
| --- | --- | --- |
|  | pκ​(λ)=∑s=0κcs​(1−λ)s,cs=(2​s)!4s​(s!)2>0.\displaystyle p\_{\kappa}(\lambda)=\sum\_{s=0}^{\kappa}c\_{s}(1-\lambda)^{s},\qquad c\_{s}=\frac{(2s)!}{4^{s}(s!)^{2}}>0. |  |

Equivalently, with reparametrization u=1−λ∈[0,1]u=1-\lambda\in[0,1], pκ​(1−u)=∑s=0κcs​usp\_{\kappa}(1-u)=\sum\_{s=0}^{\kappa}c\_{s}u^{s}.

###### Proposition 1 (Properties of pκp\_{\kappa}).

For λ∈[0,1]\lambda\in[0,1]:

* •

  Positivity.
  pκ​(λ)>0p\_{\kappa}(\lambda)>0 and pκ​(λ)≥1p\_{\kappa}(\lambda)\geq 1 with equality iff λ=1\lambda=1.
* •

  Monotonicity of τ\tau.
  Let τ​(λ):=λ​[pκ​(λ)]2\tau(\lambda):=\lambda[p\_{\kappa}(\lambda)]^{2}, then
  we have τ\tau non‑decreasing on [0,1][0,1] and τ​(1)=1\tau(1)=1.

Consequently, for any symmetric A⪰0A\succeq 0 with spectrum in [0,1][0,1],
the Newton–Schulz update A↦pκ​(A)​A​pκ​(A)A\mapsto p\_{\kappa}(A)Ap\_{\kappa}(A) satisfies
‖pκ​(A)​A​pκ​(A)‖op≤1\|p\_{\kappa}(A)Ap\_{\kappa}(A)\|\_{\mathrm{op}}\leq 1: Newton–Schulz steps preserve the unit spectral ball (see Appendix [A.3](#A1.SS3 "A.3 Newton–Schulz polynomial ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).
Moreover, the property of the function τ\tau is used when proving how fast does one step of Newton–Schulz make the momentum orthogonal (Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).

In order to quantify the degree of orthogonality of the output matrix Xt,qX\_{t,q} after qq steps of Newton–Schulz,
and to measure the approximation error derived from Newton–Schulz compared to the exact-polar method via SVD under operator-norm,
we define the following:

###### Definition 3 (Orthogonality residual and polar approximation error).

For fixed tt, let Πt\Pi\_{t} be the orthogonal projector onto range​(Mt)\mathrm{range}(M\_{t}).
With {Xt,j}j=0q\{X\_{t,j}\}\_{j=0}^{q} from Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
define the orthogonality residual δt,j\delta\_{t,j} and the polar approximation error εt,q\varepsilon\_{t,q} by

|  |  |  |
| --- | --- | --- |
|  | δt,j:=‖Πt−Xt,j​Xt,j⊤‖op∈[0,1),εt,q:=‖Xt,q−Polar⁡(Mt)‖op.\displaystyle\delta\_{t,j}:=\|\Pi\_{t}-X\_{t,j}X\_{t,j}^{\top}\|\_{\mathrm{op}}\in[0,1),\qquad\varepsilon\_{t,q}:=\|X\_{t,q}-\operatorname{Polar}(M\_{t})\|\_{\mathrm{op}}. |  |

Define εq:=suptεt,q\varepsilon\_{q}:=\sup\_{t}\varepsilon\_{t,q} and δ0:=suptδt,0\delta\_{0}:=\sup\_{t}\delta\_{t,0}.

###### Remark 1.

In Muon,
there is a scaling step for the momentum matrix before applying it to the recursive update by the Newton–Schulz polynomial
(Line 5 in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")).
This scaling ensures ‖Xt,0‖op≤1\|X\_{t,0}\|\_{\mathrm{op}}\leq 1,
which is required to apply the Newton–Schulz polynomial properties described in Proposition [1](#Thmprop1 "Proposition 1 (Properties of 𝑝_𝜅). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
In parallel, the initial residual is strictly less than 1, i.e., δt,0∈[0,1)\delta\_{t,0}\in[0,1) for every iteration tt (see Appendix [D](#A4 "Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")).

## 4 Main Results

We begin by stating our two main theorems:
(i) a nonconvex convergence rate for Muon with a finite number of Newton–Schulz steps (Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")); and
(ii) an explicit bound on the multiplicative constant induced by Newton–Schulz,
together with its (doubly) exponential decay in qq (Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).
We then provide brief proof sketches for both theorems in Sections [4.3](#S4.SS3 "4.3 Proof Sketch for Theorem 1. ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and [4.4](#S4.SS4 "4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
For comparisons, Section [4.5](#S4.SS5 "4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") also presents convergence rates for the idealized Muon with an exact SVD-based polar step and for SGD with momentum—the vector-based baseline—stated under the same nuclear-norm stationarity metric.

### 4.1 Convergence of Muon (with Newton–Schulz)

###### Theorem 1 (Convergence of Muon with Newton–Schulz).

Suppose Assumptions [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") hold,
and run Muon (Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")) with initialization W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n}.
Choose the stepsize and momentum as
η=(1−β)​DT​L\eta=\sqrt{\frac{(1-\beta)D}{TL}}
and β=1−min⁡{L​D​Bσ​r​T,1}\beta=1-\min\left\{\frac{\sqrt{LDB}}{\sigma\sqrt{rT}},1\right\} where r=min⁡{m,n}r=\min\{m,n\}.
Then there exists a factor χq>0\chi\_{q}>0,
depending only on the number qq of Newton–Schulz steps,
such that

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]≤χq⋅𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4)\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\bigl[\|\nabla f(W\_{t-1})\|\_{\*}\bigr]\leq\chi\_{q}\cdot\mathcal{O}\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right) |  |

Consequently, Muon with Newton–Schulz attains ϵ\epsilon‑stationarity with an iteration complexity of
T=𝒪​(max⁡{χq2​L​Dϵ2,χq2​r2​σ2B​ϵ2,χq4​r​σ2​L​DB​ϵ4})T=\mathcal{O}\left(\max\left\{\frac{\chi\_{q}^{2}LD}{\epsilon^{2}},\frac{\chi\_{q}^{2}r^{2}\sigma^{2}}{B\epsilon^{2}},\frac{\chi\_{q}^{4}r\sigma^{2}LD}{B\epsilon^{4}}\right\}\right)
iterations.

##### Discussions of Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") guarantees that Muon with Newton–Schulz converges to an ϵ\epsilon-stationary point.
To the best of our knowledge, this convergence guarantee is the first result for Muon with Newton–Schulz.
Moreover, as shown later by comparison with the SVD-based polar variant, the iteration complexity of Muon with Newton–Schulz matches the exact-polar rate up to a multiplicative factor χq\chi\_{q} that depends on the polar-approximation error εq\varepsilon\_{q} (e.g., Table [1](#S4.T1 "Table 1 ‣ 4.3 Proof Sketch for Theorem 1. ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).
Crucially, we show later in Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") that χq→1\chi\_{q}\to 1 converges at an exponential rate in the number of Newton–Schulz steps qq,
so the convergence gap (in the number of iterations) to the ideal SVD-polar rate can be made arbitrarily small.
Since each Newton–Schulz step is substantially cheaper than an SVD, these results provide the first theoretical explanation for the superior practical performance observed for the original (SVD-free) Muon.

### 4.2 Decay Rate of εq\varepsilon\_{q} and Convergence Rate of χq→1\chi\_{q}\to 1

We now quantify how fast χq\chi\_{q} approaches 11.
For a given number qq of Newton–Schulz steps, we can show that the polar-approximation error εq\varepsilon\_{q}
decays *doubly exponentially* in qq (with faster decay for larger κ\kappa).
Since χq\chi\_{q} is controlled by εq\varepsilon\_{q},
it follows that χq→1\chi\_{q}\to 1 is at the same rate.
The following theorem formalizes this result.

###### Theorem 2 (Upper-bounds on εq\varepsilon\_{q} and χq\chi\_{q}).

For the Newton–Schulz polynomial with degree κ\kappa and for any tt,
δt,q≤δt,0(κ+1)q\delta\_{t,q}\leq\delta\_{t,0}^{(\kappa+1)^{q}}.
Hence, the bound of the polar approximation error εq\varepsilon\_{q} and the factor χq\chi\_{q} occurred by Newton–Schulz is

|  |  |  |
| --- | --- | --- |
|  | εq≤1−1−δ0(κ+1)q≤δ0(κ+1)q,χq=11−εq≤11−δ0(κ+1)q,\displaystyle\varepsilon\_{q}\leq 1-\sqrt{1-\delta\_{0}^{(\kappa+1)^{q}}}\leq\delta\_{0}^{(\kappa+1)^{q}},\qquad\chi\_{q}=\frac{1}{1-\varepsilon\_{q}}\leq\frac{1}{\sqrt{1-\delta\_{0}^{(\kappa+1)^{q}}}}, |  |

where δ0:=suptδt,0<1\delta\_{0}:=\sup\_{t}\delta\_{t,0}<1.

##### Discussion of Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

The theorem shows that εq\varepsilon\_{q} is bounded by δ0(κ+1)q\delta\_{0}^{\,(\kappa+1)^{q}}.
Hence, εq\varepsilon\_{q} vanishes *doubly exponentially* in qq (and improves with larger κ\kappa).
Therefore, χq→1\chi\_{q}\to 1 at the same doubly exponential rate.
Together with Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),
this result implies that the iteration-complexity gap between Newton–Schulz and the idealized SVD-polar update becomes negligible after only a few Newton–Schulz steps.

##### Practical implication.

A finite number of Newton–Schulz steps yields iteration complexity essentially indistinguishable from exact SVD updates
(up to the factor χq→1\chi\_{q}\to 1 doubly exponentially fast),
while dramatically reducing per-iteration cost by using only matrix multiplications.

### 4.3 Proof Sketch for Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

We briefly outline the main ideas.
First, introduce the scaled momentum Nt=(1−β)​MtN\_{t}=(1-\beta)M\_{t} (faithfully following the original update rule),
so the EMA becomes Nt←β​Nt−1+(1−β)​GtN\_{t}\leftarrow\beta N\_{t-1}+(1-\beta)G\_{t}.
Next, apply the descent lemma (Lemma [6](#Thmlemma6 "Lemma 6 (Descent Lemma). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")) to the update Wt←Wt−1−η​OtW\_{t}\leftarrow W\_{t-1}-\eta O\_{t},
which yields a term of the form ⟨∇f​(Wt−1),Ot⟩F\langle\nabla f(W\_{t-1}),O\_{t}\rangle\_{F}.
Decompose this inner product as

|  |  |  |
| --- | --- | --- |
|  | ⟨∇f​(Wt−1),Ot⟩F=⟨Nt,Ot⟩F+⟨∇f​(Wt−1)−Nt,Ot⟩F,\displaystyle\langle\nabla f(W\_{t-1}),O\_{t}\rangle\_{F}=\langle N\_{t},O\_{t}\rangle\_{F}+\langle\nabla f(W\_{t-1})-N\_{t},O\_{t}\rangle\_{F}, |  |

thereby isolating the momentum mismatch.
Prior analyses typically stop here and average over iterations.
In contrast, we further split ⟨Nt,Ot⟩F\langle N\_{t},O\_{t}\rangle\_{F} as

|  |  |  |
| --- | --- | --- |
|  | ⟨Nt,Ot⟩F=⟨Nt,Pt⟩F+⟨Nt,Ot−Pt⟩F,\displaystyle\langle N\_{t},O\_{t}\rangle\_{F}=\langle N\_{t},P\_{t}\rangle\_{F}+\langle N\_{t},O\_{t}-P\_{t}\rangle\_{F}, |  |

separating the exact polar factor PtP\_{t} from the Newton–Schulz orthogonalizer (the output matrix of the Newton–Schulz routine).
As we define the *polar approximation error* εq\varepsilon\_{q} as the discrepancy between the exact polar factor
Pt=Polar⁡(Nt)=Polar⁡(Mt)P\_{t}=\operatorname{Polar}(N\_{t})=\operatorname{Polar}(M\_{t}) and the actual step OtO\_{t} produced by qq steps of Newton–Schulz,
we can control this part with respect to εq\varepsilon\_{q}.
This yields a one-step descent inequality for Muon that explicitly includes the Newton–Schulz-induced error εq\varepsilon\_{q}.
Averaging this inequality over t=1,…,Tt=1,\ldots,T and choosing η\eta and β\beta as specified in Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") produces the stated convergence rate. Full details appear in Appendix [B](#A2 "Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz").

Table 1: Comparison of convergence rates.

| Method | Convergence rate |
| --- | --- |
| SGD with momentum (Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) | 𝒪​(r​L​DT+(r2​σ2​L​DB​T)1/4)\mathcal{O}\!\left(\sqrt{\frac{rLD}{T}}+\left(\frac{r^{2}\sigma^{2}LD}{BT}\right)^{1/4}\right) |
| Muon with SVD (Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) | 𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4)\mathcal{O}\!\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right) |
| Muon (Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) | χq⋅𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4)\chi\_{q}\cdot\mathcal{O}\!\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right) |

D=f​(W0)−f∗D=f(W\_{0})-f^{\*}, r=min⁡{m,n}r=\min\{m,n\}, iterations TT, batch size BB, Lipschitz constant LL, variance bound σ2\sigma^{2}
  
The factor χq\chi\_{q} converges to 11 at exponential rate in step qq. (Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")):
χq≤[1−δ0(κ+1)q]−1/2\chi\_{q}\leq[1-\delta\_{0}^{(\kappa+1)^{q}}]^{-1/2}, where qq is the number of Newton–Schulz step and κ\kappa is the degree of the Newton–Schulz polynomial (Def. [2](#Thmdefinition2 "Definition 2 (Newton–Schulz polynomial). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")).

### 4.4 Proof Sketch for Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")

We outline how Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") follows;
full proofs appear in Appendix [D](#A4 "Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz").
Recall that in Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),
the convergence rate contains the polar-approximation error εq\varepsilon\_{q} and the multiplicative factor χq\chi\_{q} depending on εq\varepsilon\_{q}.
To bound the resulting multiplicative factor χq\chi\_{q}, we relate εq\varepsilon\_{q} to an *orthogonality residual*
that quantifies how close the Newton–Schulz iterate is to having orthonormal columns.
Concretely, letting Xt,qX\_{t,q} be the output matrix after qq Newton–Schulz steps applied to the (scaled) momentum at iteration tt,
define δt,q:=‖I−Xt,q​Xt,q⊤‖op\delta\_{t,q}:=\|I-X\_{t,q}X\_{t,q}^{\top}\|\_{\mathrm{op}}.

The next lemma provides the spectral link between the residual and the polar-approximation error:

###### Lemma 1 (Orthogonality residual vs. Polar approximation error).

Let λmin+\lambda\_{\min}^{+} be the smallest positive eigenvalue of Xt,q​Xt,q⊤X\_{t,q}X\_{t,q}^{\top}
restricted to range​(Mt)\mathrm{range}(M\_{t}) (set λmin+=1\lambda\_{\min}^{+}=1 if rank​(Mt)=0\mathrm{rank}(M\_{t})=0).
Then

|  |  |  |
| --- | --- | --- |
|  | δt,q=1−λmin+,εt,q=1−λmin+= 1−1−δt,q.\displaystyle\delta\_{t,q}=1-\lambda\_{\min}^{+},\qquad\varepsilon\_{t,q}=1-\sqrt{\lambda\_{\min}^{+}}\,=\,1-\sqrt{1-\delta\_{t,q}}. |  |

Next, we describe how a single Newton–Schulz step transforms the residual through the degree-κ\kappa polynomial pκp\_{\kappa}:

###### Lemma 2 (Residual update).

For Newton–Schulz polynomial pκp\_{\kappa}, the orthogonality residual δt,j\delta\_{t,j} is updated by Newton–Schulz per step as

|  |  |  |
| --- | --- | --- |
|  | δt,j+1=ϕ​(δt,j),\displaystyle\delta\_{t,j+1}=\phi(\delta\_{t,j}), |  |

where ϕ​(u):=1−(1−u)​[pκ​(1−u)]2=1−τ​(1−u)\phi(u):=1-(1-u)\left[p\_{\kappa}(1-u)\right]^{2}=1-\tau(1-u).

To prove Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"), we introduce τ​(λ):=λ​[pκ​(λ)]2\tau(\lambda):=\lambda\,[p\_{\kappa}(\lambda)]^{2}, note that τ\tau is non-decreasing on [0,1][0,1] and satisfies τ​(1)≤1\tau(1)\leq 1 (Proposition [1](#Thmprop1 "Proposition 1 (Properties of 𝑝_𝜅). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")), and then translate this monotonicity to the residual map ϕ\phi.

Finally, we obtain a contraction bound and its multi-step consequence:

###### Lemma 3 (Residual decay by Newton–Schulz polynomial).

For Newton–Schulz polynomial pκp\_{\kappa}, ϕ​(u)≤uκ+1\phi(u)\leq u^{\kappa+1} on [0,1][0,1] where ϕ\phi is a function defined in Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
Hence, for every tt and all j≥0j\geq 0,

|  |  |  |
| --- | --- | --- |
|  | δt,j+1≤δt,jκ+1,δt,q≤δt,0(κ+1)q.\displaystyle\delta\_{t,j+1}\ \leq\ \delta\_{t,j}^{\,\kappa+1},\qquad\delta\_{t,q}\ \leq\ \delta\_{t,0}^{\,(\kappa+1)^{q}}. |  |

### 4.5 Comparisons with SGD with Momentum and Muon with SVD\mathrm{SVD}

To enable a fair comparison with the original Muon with Newton–Schulz method,
we also establish convergence guarantees for two baselines:
(i) *SGD with momentum* and
(ii) the *idealized Muon with an exact polar step computed by SVD*.
All results are derived under the same smoothness and bounded-variance assumptions,
and progress is measured by the nuclear norm of the gradient, so the rates are directly comparable.

The convergence bound in Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") for SGD with momentum (Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) is,
to our knowledge, new and may be of independent interest. It provides a clean reference point for vector-based updates under the same stationarity metric.

###### Theorem 3 (Convergence of SGD with momentum).

Suppose Assumptions [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") hold,
and run SGD with momentum.
Choosing η=min⁡{1−βL,(1−β)24​L}\eta=\min\left\{\frac{1-\beta}{L},\frac{(1-\beta)^{2}}{4L}\right\} and β=1−min⁡{L​D​Bσ​T,1}\beta=1-\min\{\frac{\sqrt{LDB}}{\sigma\sqrt{T}},1\}, the following holds

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​‖∇f​(Wt−1)‖∗≤𝒪​(r​L​DT+(r2​σ2​L​DB​T)1/4),\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|\nabla f(W\_{t-1})\|\_{\*}\leq\mathcal{O}\!\left(\sqrt{\frac{rLD}{T}}+\left(\frac{r^{2}\sigma^{2}LD}{BT}\right)^{1/4}\right), |  |

Consequently, SGD with momentum attains ϵ\epsilon‑stationarity with an iteration complexity of
𝒪​(max⁡{r​L​Dϵ2,r2​σ2​L​DB​ϵ4})\mathcal{O}\!\left(\max\left\{\frac{rLD}{\epsilon^{2}},\frac{r^{2}\sigma^{2}LD}{B\epsilon^{4}}\right\}\right).

Note that the rate for Muon with SVD (Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) is obtained as a special case of Muon with Newton–Schulz analysis by setting the polar-approximation error to zero (i.e., replacing Newton–Schulz with an exact polar step).
This “oracle” baseline clarifies the gap that Newton–Schulz needs to close in practice and is useful for interpreting the effect of a finite number of Newton–Schulz steps.

###### Theorem 4 (Muon with SVD).

Under Assumptions [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
setting εq=0\varepsilon\_{q}=0 in Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") yields

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​‖∇f​(Wt−1)‖∗≤𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4),\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|\nabla f(W\_{t-1})\|\_{\*}\;\leq\;\mathcal{O}\!\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right), |  |

Consequently, the idealized Muon with SVD\mathrm{SVD} attains ϵ\epsilon‑stationarity with iteration complexity of
𝒪​(max⁡{L​Dϵ2,r2​σ2B​ϵ2,r​σ2​L​DB​ϵ4})\mathcal{O}\left(\max\left\{\frac{LD}{\epsilon^{2}},\frac{r^{2}\sigma^{2}}{B\epsilon^{2}},\frac{r\sigma^{2}LD}{B\epsilon^{4}}\right\}\right)

##### Discussion of Theorems [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

Relative to SGD with momentum,
the SVD-based polar variant of Muon removes the r\sqrt{r} factor from the deterministic (first) term and sharpens rank dependence in the stochastic terms under the same nuclear-norm stationarity metric.
Geometrically, the polar step aligns the update with the leading singular structure of the gradient (via spectral–nuclear duality),
converting a Frobenius-aligned descent direction into one that is optimally aligned for the nuclear norm, thereby sharpening the rr dependence.

Turning to the practical Muon with Newton–Schulz,
Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") shows that its iteration complexity matches that of the SVD-based polar variant (Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")) up to a multiplicative factor χq\chi\_{q}.
By Theorem [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"), χq→1\chi\_{q}\!\to\!1 *doubly exponentially* fast in the number of Newton–Schulz steps qq (and improves with the polynomial degree κ\kappa).
Consequently, a small qq already yields rates that are essentially indistinguishable from the ideal SVD-based baseline in iteration count.
Since each Newton–Schulz step uses only matrix multiplications (and avoids SVD),
the per-iteration cost is much lower, which explains the superior wall-clock performance of the original SVD-free Muon observed in practice.

## 5 Numerical Experiments

Setup.
We conduct a numerical experiment with the CIFAR‑10 (50k/10k) dataset
and a CNN model, specifically CifarNet, which has approximately 22M parameters.
We compare optimizers: SGD with momentum (baseline), idealized Muon with SVD, and Muon with Newton–Schulz (q∈{1,2,3}q\in\{1,2,3\}).
For the Newton–Schulz step sweep, we use the Newton–Schulz polynomial pκp\_{\kappa} with degree κ=2\kappa=2.
We run 50 epochs, and the batch size is B=512B=512.
Results are plotted in Fig. [1](#S5.F1 "Figure 1 ‣ 5 Numerical Experiments ‣ Convergence of Muon with Newton–Schulz").
Performance is assessed by plotting the training loss (left column) and test loss (right column)
over epochs (top row) as well as the cumulative wall-clock time (bottom row).
Results represent the average of five runs with different random seeds, including standard deviations.
More detailed numerical settings are described in Appendix [F.1](#A6.SS1 "F.1 Experimental Setting ‣ Appendix F Numerical Experiments Detail ‣ Convergence of Muon with Newton–Schulz").

Ablation on Newton–Schulz step qq.
As qq increases, the learning dynamics per epoch steadily improve:
Muon with q=1q=1 already outperforms SGD‑M,
and q∈2,3q\in{2,3} nearly coincides with the SVD‑based Muon in both train loss and test loss,
in line with Theorems [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and [2](#Thmtheorem2 "Theorem 2 (Upper-bounds on 𝜀_𝑞 and 𝜒_𝑞). ‣ 4.2 Decay Rate of 𝜀_𝑞 and Convergence Rate of 𝜒_𝑞→1 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),
which state that the Newton–Schulz variant matches the SVD iteration complexity up to a factor χq→1\chi\_{q}\to 1 that decays doubly exponentially in qq.
At the same time, the bottom row of Fig. [1](#S5.F1 "Figure 1 ‣ 5 Numerical Experiments ‣ Convergence of Muon with Newton–Schulz") shows that Muon with q=2q=2 or 33 reaches a given test loss substantially faster in wall‑clock time than the SVD variant, reflecting the lower per‑iteration cost of the Newton–Schulz update.

!(/html/2601.19156/assets/x1.png)

Figure 1: Newton–Schulz steps (qq) ablation.
Muon with Newton–Schulz for q∈{1,2,3}q\in\{1,2,3\} vs. Muon (SVD) and SGD with momentum (SGD‑M, baseline).

We additionally performed numerical experiments on various datasets using models at different scales (with the number of parameters indicated in parentheses): a multilayer perceptron (MLP) with 0.5M parameters on MNIST; ResNet-18 (11.2M) on CIFAR-100; WideResNet-28-10 (36.6M) on Tiny-ImageNet; NanoGPT (124M, Transformer) on FineWeb; and a GPT-2–based model (1.3B, Transformer) on FineWeb. All additional experimental results are presented in Appendix [G](#A7 "Appendix G Additional Numerical Experiments ‣ Convergence of Muon with Newton–Schulz").

Ablation on Newton–Schulz polynomial degree κ\kappa.
We perform a controlled ablation that varies the degree of the Newton–Schulz polynomial κ\kappa
while fixing the number of Newton–Schulz steps to q=3q=3 for all variants.
Increasing the degree κ∈{1,…,5}\kappa\in\{1,\dots,5\} improves optimization (the loss drops faster at a fixed epoch)
but lengthens each step, yielding a clear accuracy–time trade-off (See Appendix [H.1](#A8.SS1 "H.1 Newton–Schulz–polynomial degree-𝜅 ablations. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz")).
This mirrors the theory that the residual contracts as δj+1≤δjκ+1\delta\_{j+1}\leq\delta\_{j}^{\kappa+1} (Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")),
while computation scales with polynomial evaluations.

Rank dependence.
We vary the monitored layer’s effective rank r∈{16,32,64,128,216}r\in\{16,32,64,128,216\} and plot the epoch-averaged ‖∇f​(W)‖∗\|\nabla f(W)\|\_{\*} on a log–log scale.
SGD-M shows a positive slope of approximately 0.30.3 (grows with rr),
whereas Muon and its variants are nearly flat.
These observations are precisely in line with Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") and Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"), which state that
orthogonalizing momentum removes the deterministic r\sqrt{r} penalty and softens the rank dependence of the stochastic terms.
The more detailed experimental settings and results are deferred to Appendix [H.2](#A8.SS2 "H.2 Rank–dependence. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz").

## 6 Conclusion

We analyzed practical Muon with finite Newton–Schulz steps and proved nonconvex convergence to ϵ\epsilon-stationarity with iteration complexity matching the SVD-polar idealization, up to a factor χq\chi\_{q} that shrinks doubly exponentially in qq (and improves with κ\kappa).
Thus, a few Newton–Schulz steps recover the idealized rate while remaining SVD-free and GPU-friendly.
We also provided baselines for SGD with momentum and the SVD-polar variant,
showing that Muon weakens rank dependence under the same metric
—closing the theory–practice gap and explaining its practical performance.

## Use of Large Language Models

Large Language Models (LLMs) are used solely as assistive tools for writing.
Specifically, we employed an LLM to improve the clarity, grammar, and style of exposition.
No part of the research ideation, algorithm design, theoretical analysis, or experimental results involved the use of LLMs.
The authors take full responsibility for the content of the paper.

## References

* K. Ahn and B. X. Dion (2025)
  A communication-efficient optimizer for large models.
  arXiv preprint arXiv:2504.05295.
  Cited by: [§2](#S2.p3.5 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* K. An, Y. Liu, R. Pan, Y. Ren, S. Ma, D. Goldfarb, and T. Zhang (2025)
  Asgo: adaptive structured gradient optimization.
  arXiv preprint arXiv:2503.20762.
  Cited by: [§2](#S2.p2.1 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* A. Beck (2017)
  First-order methods in optimization.
   SIAM.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* J. Bernstein and L. Newhouse (2024)
  Old optimizer, new norm: an anthology.
  arXiv preprint arXiv:2409.20325.
  Cited by: [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* L. Bottou, F. E. Curtis, and J. Nocedal (2018)
  Optimization methods for large-scale machine learning.
  SIAM review 60 (2),  pp. 223–311.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* S. P. Boyd and L. Vandenberghe (2004)
  Convex optimization.
   Cambridge university press.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* E. Candes and B. Recht (2012)
  Exact matrix completion via convex optimization.
  Communications of the ACM 55 (6),  pp. 111–119.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* L. Chen, J. Li, and Q. Liu (2025)
  Muon optimizes under spectral norm constraints.
  arXiv preprint arXiv:2506.15054.
  Cited by: [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* S. Ghadimi and G. Lan (2013)
  Stochastic first-and zeroth-order methods for nonconvex stochastic programming.
  SIAM journal on optimization 23 (4),  pp. 2341–2368.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* G. H. Golub and C. Reinsch (1971)
  Singular value decomposition and least squares solutions.
  In Linear algebra,
   pp. 134–151.
  Cited by: [1st item](#A5.I1.i1.p1.4 "In Per-iteration Orthogonalization FLOPs. ‣ Appendix E Wall-Clock via Computational Complexity. ‣ Convergence of Muon with Newton–Schulz").
* V. Gupta, T. Koren, and Y. Singer (2018)
  Shampoo: preconditioned stochastic tensor optimization.
  In International Conference on Machine Learning,
   pp. 1842–1850.
  Cited by: [§2](#S2.p2.1 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* M. Jaggi (2013)
  Revisiting frank-wolfe: projection-free sparse convex optimization.
  In International conference on machine learning,
   pp. 427–435.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein (2024)
  Muon: an optimizer for hidden layers in neural networks.
  External Links: [Link](https://kellerjordan.github.io/posts/muon/)
  Cited by: [§1](#S1.p2.9 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§1](#S1.p4.16 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p1.7 "2 Related work ‣ Convergence of Muon with Newton–Schulz"),
  [§3.3](#S3.SS3.p1.5 "3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* D. P. Kingma and J. Ba (2015)
  Adam: A method for stochastic optimization.
  In 3rd International Conference on Learning Representations, ICLR 2015,
  San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings,
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Convergence of Muon with Newton–Schulz").
* D. Kovalev (2025)
  Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization.
  arXiv preprint arXiv:2503.12645.
  Cited by: [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* J. Li and M. Hong (2025)
  A note on the convergence of muon and further.
  arXiv e-prints,  pp. arXiv–2502.
  Cited by: [§1](#S1.p2.9 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, et al. (2025a)
  Muon is scalable for llm training.
  arXiv preprint arXiv:2502.16982.
  Cited by: [§1](#S1.p2.9 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p1.7 "2 Related work ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p3.5 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* L. Liu, Z. Xu, Z. Zhang, H. Kang, Z. Li, C. Liang, W. Chen, and T. Zhao (2025b)
  Cosmos: a hybrid adaptive optimizer for memory-efficient training of llms.
  arXiv preprint arXiv:2502.17410.
  Cited by: [§2](#S2.p3.5 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro (2009)
  Robust stochastic approximation approach to stochastic programming.
  SIAM Journal on optimization 19 (4),  pp. 1574–1609.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* Y. Nesterov (2013)
  Introductory lectures on convex optimization: a basic course.
  Vol. 87, Springer Science & Business Media.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* S. J. Reddi, S. Kale, and S. Kumar (2018)
  On the convergence of adam and beyond.
  In International Conference on Learning Representations,
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* A. Riabinin, E. Shulgin, K. Gruntkowska, and P. Richtárik (2025)
  Gluon: making muon & scion great again!(bridging theory and practice of lmo-based optimizers for llms).
  arXiv preprint arXiv:2505.13416.
  Cited by: [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* H. Robbins and S. Monro (1951)
  A stochastic approximation method.
  The annals of mathematical statistics,  pp. 400–407.
  Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Convergence of Muon with Newton–Schulz").
* N. Sato, H. Naganuma, and H. Iiduka (2025)
  Analysis of muon’s convergence and critical batch size.
  arXiv preprint arXiv:2507.01598.
  Cited by: [§1](#S1.p2.9 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* I. Shah, A. M. Polloreno, K. Stratos, P. Monk, A. Chaluvaraju, A. Hojel, A. Ma, A. Thomas, A. Tanwer, D. J. Shah, et al. (2025)
  Practical efficiency of muon for pretraining.
  arXiv preprint arXiv:2505.02222.
  Cited by: [§2](#S2.p3.5 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* A. Shapiro, D. Dentcheva, and A. Ruszczynski (2021)
  Lectures on stochastic programming: modeling and theory.
   SIAM.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").
* W. Shen, R. Huang, M. Huang, C. Shen, and J. Zhang (2025)
  On the convergence analysis of muon.
  arXiv preprint arXiv:2505.23737.
  Cited by: [§1](#S1.p2.9 "1 Introduction ‣ Convergence of Muon with Newton–Schulz"),
  [§2](#S2.p4.3 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* M. Tuddenham, A. Prügel-Bennett, and J. Hare (2022)
  Orthogonalising gradients to speed up neural network optimisation.
  arXiv preprint arXiv:2202.07052.
  Cited by: [§2](#S2.p1.7 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* A. Tveit, B. Remseth, and A. Skogvold (2025)
  Muon optimizer accelerates grokking.
  arXiv preprint arXiv:2504.16041.
  Cited by: [§2](#S2.p3.5 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* N. Vyas, D. Morwani, R. Zhao, M. Kwun, I. Shapira, D. Brandfonbrener, L. Janson, and S. Kakade (2024)
  Soap: improving and stabilizing shampoo using adam.
  arXiv preprint arXiv:2409.11321.
  Cited by: [§2](#S2.p2.1 "2 Related work ‣ Convergence of Muon with Newton–Schulz").
* F. Zou, L. Shen, Z. Jie, W. Zhang, and W. Liu (2019)
  A sufficient condition for convergences of adam and rmsprop.
  In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition,
   pp. 11127–11135.
  Cited by: [§3.2](#S3.SS2.p2.2 "3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz").

## Appendix A Appendix

### A.1 Basic Facts for Matrix Norms

###### Definition 4 (Schatten Norms).

For each p∈[1,∞]p\in[1,\infty], the Schatten-pp norm is defined as ‖X‖Sp:=(∑iσi​(X)p)1/p\|X\|\_{S\_{p}}:=\left(\sum\_{i}\sigma\_{i}(X)^{p}\right)^{1/p}, where σi​(X)\sigma\_{i}(X) are the singular values of XX.

* •

  Nuclear/trace norm: ‖X‖S1=‖X‖∗=∑iσi​(X)\|X\|\_{S\_{1}}=\|X\|\_{\*}=\sum\_{i}\sigma\_{i}(X) (Sum of singular values)
* •

  spectral/operator norm: ‖X‖S∞=‖X‖op=maxi⁡σi​(X)\|X\|\_{S\_{\infty}}=\|X\|\_{\mathrm{op}}=\max\_{i}\sigma\_{i}(X) (largest singular value)
* •

  Frobenius norm: ‖X‖S2=‖X‖F=∑iσi​(X)2\|X\|\_{S\_{2}}=\|X\|\_{F}=\sqrt{\sum\_{i}\sigma\_{i}(X)^{2}}

For any conjugate pairs (p,q)(p,q), i.e., 1p+1q=1\frac{1}{p}+\frac{1}{q}=1,
the norms ∥⋅∥Sp\|\cdot\|\_{S\_{p}} and ∥⋅∥Sq\|\cdot\|\_{S\_{q}} are dual to each other.
In particular, the nuclear norm and the spectral norm are duals.

###### Lemma 4 (Hölder’s inequality).

For any A,B∈ℝm×nA,B\in\mathbb{R}^{m\times n},

|  |  |  |
| --- | --- | --- |
|  | |⟨A,B⟩F|≤‖A‖∗​‖B‖op.\displaystyle|\langle A,B\rangle\_{F}|\ \leq\ \|A\|\_{\*}\,\|B\|\_{\mathrm{op}}. |  |

###### Proof.

Let SVD​(A)=(U,Σ,V)\mathrm{SVD}(A)=(U,\Sigma,V).

|  |  |  |
| --- | --- | --- |
|  | ⟨A,B⟩F=tr⁡(A⊤​B)=tr⁡((U​Σ​V⊤)⊤​B)=tr⁡(V​Σ​U⊤​B)=tr⁡(Σ​U⊤​B​V)\displaystyle\langle A,B\rangle\_{F}=\operatorname{tr}(A^{\top}B)=\operatorname{tr}((U\Sigma V^{\top})^{\top}B)=\operatorname{tr}(V\Sigma U^{\top}B)=\operatorname{tr}(\Sigma U^{\top}BV) |  |

Let’s define a new matrix C=U⊤​B​VC=U^{\top}BV.
Since U⊤U^{\top} and VV are orthogonal matrices, their operator norm is 11.

|  |  |  |
| --- | --- | --- |
|  | ‖C‖op=‖U⊤​B​V‖op≤‖U⊤‖op​‖B‖op​‖V‖op=1⋅‖B‖op⋅1=‖B‖op\displaystyle\|C\|\_{\mathrm{op}}=\|U^{\top}BV\|\_{\mathrm{op}}\leq\|U^{\top}\|\_{\mathrm{op}}\|B\|\_{\mathrm{op}}\|V\|\_{\mathrm{op}}=1\cdot\|B\|\_{\mathrm{op}}\cdot 1=\|B\|\_{\mathrm{op}} |  |

Since Σ\Sigma is diagonal, tr⁡(Σ​C)=∑iσi​(A)​Ci​i\operatorname{tr}(\Sigma C)=\sum\_{i}\sigma\_{i}(A)C\_{ii} where σi​(A)\sigma\_{i}(A) is the ii-th singular value of AA.

|  |  |  |
| --- | --- | --- |
|  | |⟨A,B⟩F|=|tr⁡(Σ​C)|=|∑iσi​(A)​Ci​i|≤∑iσi​(A)​|Ci​i|≤∑iσi​(A)​‖C‖op=‖A‖∗​‖C‖op\displaystyle|\langle A,B\rangle\_{F}|=|\operatorname{tr}(\Sigma C)|=\left|\sum\_{i}\sigma\_{i}(A)C\_{ii}\right|\leq\sum\_{i}\sigma\_{i}(A)|C\_{ii}|\leq\sum\_{i}\sigma\_{i}(A)\|C\|\_{\mathrm{op}}=\|A\|\_{\*}\|C\|\_{\mathrm{op}} |  |

Combining inequalities, we have |⟨A,B⟩F|≤‖A‖∗​‖B‖op|\langle A,B\rangle\_{F}|\leq\|A\|\_{\*}\|B\|\_{\mathrm{op}}.
∎

###### Lemma 5.

For any A,B∈ℝm×nA,B\in\mathbb{R}^{m\times n} and any constant β∈(0,1)\beta\in(0,1),

|  |  |  |
| --- | --- | --- |
|  | ‖A+B‖F2≤1β​‖A‖F2+11−β​‖B‖F2.\displaystyle\|A+B\|\_{F}^{2}\;\leq\;\frac{1}{\beta}\|A\|\_{F}^{2}+\frac{1}{1-\beta}\|B\|\_{F}^{2}. |  |

###### Proof.

By Young’s inequality (i.e., ⟨x,y⟩F≤ϵ2​‖x‖F2+12​ϵ​‖y‖F2\langle x,y\rangle\_{F}\leq\frac{\epsilon}{2}\|x\|\_{F}^{2}+\frac{1}{2\epsilon}\|y\|\_{F}^{2} with ϵ>0\epsilon>0), for any positive constant c>0c>0,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖A+B‖F2\displaystyle\|A+B\|\_{F}^{2} | =‖A‖F2+‖B‖F2+2​⟨A,B⟩F\displaystyle=\|A\|\_{F}^{2}+\|B\|\_{F}^{2}+2\langle A,B\rangle\_{F} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤‖A‖2+‖B‖F2+c​‖A‖F2+1c​‖B‖F2\displaystyle\leq\|A\|^{2}+\|B\|\_{F}^{2}+c\|A\|\_{F}^{2}+\frac{1}{c}\|B\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1+c)​‖A‖F2+(1+1c)​‖B‖F2\displaystyle=(1+c)\|A\|\_{F}^{2}+\left(1+\frac{1}{c}\right)\|B\|\_{F}^{2} |  |

If we choose c=1−ββc=\frac{1-\beta}{\beta} where β∈(0,1)\beta\in(0,1), we get

|  |  |  |
| --- | --- | --- |
|  | ‖A+B‖F2≤1β​‖A‖F2+11−β​‖B‖F2\displaystyle\|A+B\|\_{F}^{2}\leq\frac{1}{\beta}\|A\|\_{F}^{2}+\frac{1}{1-\beta}\|B\|\_{F}^{2} |  |

∎

###### Proposition 2.

Let X∈ℝm×nX\in\mathbb{R}^{m\times n} with r=min⁡{m,n}r=\min\{m,n\}.
Denote that its singular values are {σi}i=1r\{\sigma\_{i}\}\_{i=1}^{r}.
Then the following holds:

1. (i)

   ‖X‖F=tr⁡(X⊤​X)=∑i​jXi,j2\|X\|\_{F}=\sqrt{\operatorname{tr}(X^{\top}X)}=\sqrt{\sum\_{ij}X\_{i,j}^{2}}

   ###### Proof.

   Let SVD​(X)=(U,Σ,V)\mathrm{SVD}(X)=(U,\Sigma,V). Then

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | tr⁡(X⊤​X)\displaystyle\operatorname{tr}(X^{\top}X) | =tr⁡((U​Σ​V⊤)⊤​(U​Σ​V⊤))=tr⁡(V​Σ⊤​U⊤​U​Σ​V⊤)=tr⁡(V​Σ⊤​Σ​V⊤)\displaystyle=\operatorname{tr}\left((U\Sigma V^{\top})^{\top}(U\Sigma V^{\top})\right)=\operatorname{tr}(V\Sigma^{\top}U^{\top}U\Sigma V^{\top})=\operatorname{tr}(V\Sigma^{\top}\Sigma V^{\top}) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =tr⁡(V⊤​V​Σ⊤​Σ)=tr⁡(Σ⊤​Σ)=∑iσi​(X)2=‖X‖F2\displaystyle=\operatorname{tr}(V^{\top}V\Sigma^{\top}\Sigma)=\operatorname{tr}(\Sigma^{\top}\Sigma)=\sum\_{i}\sigma\_{i}(X)^{2}=\|X\|\_{F}^{2} |  |

   Since (X⊤)j​i=Xi​j(X^{\top})\_{ji}=X\_{ij}, we have

   |  |  |  |
   | --- | --- | --- |
   |  | (X⊤​X)j​j=∑i=1m(X⊤)j​i​Xi​j=∑i=1mXi​j2\displaystyle(X^{\top}X)\_{jj}=\sum\_{i=1}^{m}(X^{\top})\_{ji}X\_{ij}=\sum\_{i=1}^{m}X\_{ij}^{2} |  |

   Hence, tr⁡(X⊤​X)=∑j=1n(X⊤​X)j​j=∑j=1n∑i=1mXi​j2\operatorname{tr}(X^{\top}X)=\sum\_{j=1}^{n}(X^{\top}X)\_{jj}=\sum\_{j=1}^{n}\sum\_{i=1}^{m}X\_{ij}^{2}
   ∎
2. (ii)

   If P=Polar⁡(X)P=\operatorname{Polar}(X), then ⟨X,P⟩F=‖X‖∗\langle X,P\rangle\_{F}=\|X\|\_{\*}.

   ###### Proof.

   Let SVD​(X)=(U,Σ,V)\mathrm{SVD}(X)=(U,\Sigma,V). Then P=Polar⁡(X)=U​V⊤P=\operatorname{Polar}(X)=UV^{\top}.

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ⟨X,P⟩F\displaystyle\langle X,P\rangle\_{F} | =⟨U​Σ​V⊤,U​V⊤⟩F=tr⁡((U​Σ​V⊤)⊤​U​V⊤)=tr⁡(V​Σ​U⊤​U​V⊤)\displaystyle=\langle U\Sigma V^{\top},UV^{\top}\rangle\_{F}=\operatorname{tr}((U\Sigma V^{\top})^{\top}UV^{\top})=\operatorname{tr}(V\Sigma U^{\top}UV^{\top}) |  |
   |  |  |  |  |
   | --- | --- | --- | --- |
   |  |  | =tr⁡(V​Σ​V⊤)=tr⁡(V⊤​V​Σ)=tr⁡(Σ)=∑iσi​(X)=‖X‖∗\displaystyle=\operatorname{tr}(V\Sigma V^{\top})=\operatorname{tr}(V^{\top}V\Sigma)=\operatorname{tr}(\Sigma)=\sum\_{i}\sigma\_{i}(X)=\|X\|\_{\*} |  |

   ∎
3. (iii)

   ‖X‖∗≤r​‖X‖F\|X\|\_{\*}\leq\sqrt{r}\,\|X\|\_{F}.

   ###### Proof.

   By Cauchy-Schwarz inequality,

   |  |  |  |
   | --- | --- | --- |
   |  | ‖X‖∗=∑i=1rσi​(X)≤r​∑i=1rσi​(X)2=r​‖X‖F\displaystyle\|X\|\_{\*}=\sum\_{i=1}^{r}\sigma\_{i}(X)\leq\sqrt{r\sum\_{i=1}^{r}\sigma\_{i}(X)^{2}}=\sqrt{r}\|X\|\_{F} |  |

   ∎
4. (iv)

   ‖X‖op≤‖X‖F≤‖X‖∗\|X\|\_{\mathrm{op}}\leq\|X\|\_{F}\leq\|X\|\_{\*}.

   ###### Proof.

   Since singular values are always non-negative (σi​(X)≥0)(\sigma\_{i}(X)\geq 0),

   |  |  |  |
   | --- | --- | --- |
   |  | (maxi⁡σi​(X))2≤∑iσi​(X)2≤(∑iσi​(X))2\displaystyle\left(\max\_{i}\sigma\_{i}(X)\right)^{2}\leq\sum\_{i}\sigma\_{i}(X)^{2}\leq\left(\sum\_{i}\sigma\_{i}(X)\right)^{2} |  |

   Taking the square root, we get ‖X‖op≤‖X‖F≤‖X‖∗\|X\|\_{\mathrm{op}}\leq\|X\|\_{F}\leq\|X\|\_{\*}.
   ∎
5. (v)

   The polar factor is scale-invariant: Polar⁡(c​X)=Polar⁡(X)\operatorname{Polar}(cX)=\operatorname{Polar}(X) for all c>0c>0.

   ###### Proof.

   Let SVD​(X)=(U,Σ,V)\mathrm{SVD}(X)=(U,\Sigma,V). Then P=Polar⁡(X)=U​V⊤P=\operatorname{Polar}(X)=UV^{\top}.
   Also, SVD​(c​X)=(U,c​Σ,V)\mathrm{SVD}(cX)=(U,c\Sigma,V), so Polar⁡(c​X)=U​V⊤\operatorname{Polar}(cX)=UV^{\top}.
   Thus, Polar⁡(c​X)=Polar⁡(X)\operatorname{Polar}(cX)=\operatorname{Polar}(X) for all c>0c>0.
   ∎

### A.2 Lemmas under Assumptions

#### A.2.1 Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")

###### Definition 5 (LL-smoothness).

A differentiable function f:ℝm×n→ℝf:\mathbb{R}^{m\times n}\to\mathbb{R} is LL-smooth if its gradient is LL-Lipschitz continuous, i.e., for all X,Y∈ℝm×nX,Y\in\mathbb{R}^{m\times n},

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(X)−∇f​(Y)‖∗≤L​‖X−Y‖op\displaystyle\|\nabla f(X)-\nabla f(Y)\|\_{\*}\leq L\|X-Y\|\_{\mathrm{op}} |  |

We use smoothness with respect to the operator norm (and the nuclear norm as its dual).

###### Lemma 6 (Descent Lemma).

Let f:ℝm×n→ℝf:\mathbb{R}^{m\times n}\to\mathbb{R} be LL-smooth.
Then, for all X,Y∈ℝm×nX,Y\in\mathbb{R}^{m\times n},

|  |  |  |
| --- | --- | --- |
|  | f​(Y)≤f​(X)+⟨∇f​(X),Y−X⟩F+L2​‖Y−X‖op2.\displaystyle f(Y)\ \leq\ f(X)+\langle\nabla f(X),Y-X\rangle\_{F}+\frac{L}{2}\|Y-X\|\_{\mathrm{op}}^{2}. |  |

###### Proof.

Define an auxiliary scalar function g:[0,1]→ℝg:[0,1]\to\mathbb{R} by considering the value of ff along the line segment from XX to YY:

|  |  |  |
| --- | --- | --- |
|  | g​(s):=f​(X+s​(Y−X))\displaystyle g(s):=f(X+s(Y-X)) |  |

By the fundamental theorem of calculus,

|  |  |  |
| --- | --- | --- |
|  | f​(Y)−f​(X)=g​(1)−g​(0)=∫01g′​(s)​𝑑s\displaystyle f(Y)-f(X)=g(1)-g(0)=\int\_{0}^{1}g^{\prime}(s)\,ds |  |

Using the chain rule, g′​(s)=⟨∇f​(X+s​(Y−X)),Y−X⟩Fg^{\prime}(s)=\langle\nabla f(X+s(Y-X)),Y-X\rangle\_{F}.
Substituting this back into the integral equation gives:

|  |  |  |
| --- | --- | --- |
|  | f​(Y)−f​(X)=∫01⟨∇f​(X+s​(Y−X)),Y−X⟩F​𝑑s\displaystyle f(Y)-f(X)=\int\_{0}^{1}\langle\nabla f(X+s(Y-X)),Y-X\rangle\_{F}ds |  |

Add and subtract ⟨∇f​(X),Y−X⟩F=∫01⟨∇f​(X),Y−X⟩F​𝑑s\langle\nabla f(X),Y-X\rangle\_{F}=\int\_{0}^{1}\langle\nabla f(X),Y-X\rangle\_{F}ds, we have:

|  |  |  |
| --- | --- | --- |
|  | f​(Y)−f​(X)=⟨∇f​(X),Y−X⟩F+∫01⟨∇f​(X+s​(Y−X))−∇f​(X),Y−X⟩F​𝑑s\displaystyle f(Y)-f(X)=\langle\nabla f(X),Y-X\rangle\_{F}+\int\_{0}^{1}\langle\nabla f(X+s(Y-X))-\nabla f(X),Y-X\rangle\_{F}ds |  |

We now use the generalized Cauchy-Schwarz inequality or Hölder’s inequality (Lemma [4](#Thmlemma4 "Lemma 4 (Hölder’s inequality). ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")),
which states that for any matrices A,B∈ℝm×nA,B\in\mathbb{R}^{m\times n} and |⟨A,B⟩F|≤‖A‖∗​‖B‖op|\langle A,B\rangle\_{F}|\leq\|A\|\_{\*}\|B\|\_{\mathrm{op}}, the nuclear norm (∥⋅∥∗)(\|\cdot\|\_{\*}) is the dual norm to the operator norm (∥⋅∥op)(\|\cdot\|\_{\mathrm{op}}).
Applying this to the integrand:

|  |  |  |
| --- | --- | --- |
|  | ⟨∇f​(X+s​(Y−X))−∇f​(X),Y−X⟩F≤‖∇f​(X+s​(Y−X))−∇f​(X)‖∗​‖Y−X‖op\displaystyle\langle\nabla f(X+s(Y-X))-\nabla f(X),Y-X\rangle\_{F}\leq\|\nabla f(X+s(Y-X))-\nabla f(X)\|\_{\*}\|Y-X\|\_{\mathrm{op}} |  |

By the LL-smoothness of ff:

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(X+s​(Y−X))−∇f​(X)‖∗≤L​‖s​(Y−X)‖op=L​s​‖Y−X‖op\displaystyle\|\nabla f(X+s(Y-X))-\nabla f(X)\|\_{\*}\leq L\|s(Y-X)\|\_{\mathrm{op}}=Ls\|Y-X\|\_{\mathrm{op}} |  |

Combining these inequalities,
we obtain a bound on the integrand, yielding the final form:

|  |  |  |
| --- | --- | --- |
|  | f​(Y)−f​(X)≤⟨∇f​(X),Y−X⟩F+∫01L​s​‖Y−X‖op2​𝑑s=⟨∇f​(X),Y−X⟩F+L2​‖Y−X‖op2\displaystyle f(Y)-f(X)\leq\langle\nabla f(X),Y-X\rangle\_{F}+\int\_{0}^{1}Ls\|Y-X\|\_{\mathrm{op}}^{2}ds=\langle\nabla f(X),Y-X\rangle\_{F}+\frac{L}{2}\|Y-X\|\_{\mathrm{op}}^{2} |  |

∎

###### Lemma 7 (Gradient-gap inequality).

Under Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
for any WW, we have

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(W)‖F2≤2​L​(f​(W)−f∗)\displaystyle\|\nabla f(W)\|\_{F}^{2}\leq 2L\left(f(W)-f^{\*}\right) |  |

###### Proof.

In the descent lemma (Lemma [6](#Thmlemma6 "Lemma 6 (Descent Lemma). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")),
let Y=X−1L​∇f​(X)Y=X-\frac{1}{L}\nabla f(X).
Then, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(X−1L​∇f​(X))\displaystyle f\left(X-\frac{1}{L}\nabla f(X)\right) | ≤f​(X)+⟨∇f​(X),X−1L​∇f​(X)−X⟩F+L2​‖X−1L​∇f​(X)−X‖op2\displaystyle\leq f(X)+\left\langle\nabla f(X),X-\frac{1}{L}\nabla f(X)-X\right\rangle\_{F}+\frac{L}{2}\left\|X-\frac{1}{L}\nabla f(X)-X\right\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(X)−1L​‖∇f​(X)‖F+12​L​‖∇f​(X)‖op2\displaystyle=f(X)-\frac{1}{L}\|\nabla f(X)\|\_{F}+\frac{1}{2L}\|\nabla f(X)\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(X)−1L​‖∇f​(X)‖F+12​L​‖∇f​(X)‖F2\displaystyle\leq f(X)-\frac{1}{L}\|\nabla f(X)\|\_{F}+\frac{1}{2L}\|\nabla f(X)\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(X)−12​L​‖∇f​(X)‖F\displaystyle=f(X)-\frac{1}{2L}\|\nabla f(X)\|\_{F} |  |

where the last inequality is due to Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iv).
Since f∗≤f​(X−1L​∇f​(X))f^{\*}\leq f\left(X-\frac{1}{L}\nabla f(X)\right), we have

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(W)‖F2≤2​L​(f​(W)−f∗)\displaystyle\|\nabla f(W)\|\_{F}^{2}\leq 2L\left(f(W)-f^{\*}\right) |  |

∎

#### A.2.2 Assumption [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")

###### Lemma 8 (Unbiasedness and bounded variance).

If Assumption [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") holds, then

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖Gt−∇f​(Wt)‖F2]≤σ2Band𝔼​[‖Gt−∇f​(Wt)‖F]≤σB.\displaystyle\mathbb{E}\left[\|G\_{t}-\nabla f(W\_{t})\|\_{F}^{2}\right]\leq\frac{\sigma^{2}}{B}\quad\text{and}\quad\mathbb{E}[\|G\_{t}-\nabla f(W\_{t})\|\_{F}]\leq\frac{\sigma}{\sqrt{B}}. |  |

where BB is the batch size.

###### Proof.

Since Gt=1B​∑i=1BgiG\_{t}=\frac{1}{B}\sum\_{i=1}^{B}g\_{i} with gi=∇f​(Wt;ξt,i)g\_{i}=\nabla f(W\_{t};\xi\_{t,i}),

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Gt−∇f​(Wt)‖F2]\displaystyle\mathbb{E}\left[\|G\_{t}-\nabla f(W\_{t})\|\_{F}^{2}\right] | =𝔼​[‖1B​∑i=1B(∇f​(Wt;ξt,i)−∇f​(Wt))‖F2]\displaystyle=\mathbb{E}\left[\left\|\frac{1}{B}\sum\_{i=1}^{B}\left(\nabla f(W\_{t};\xi\_{t,i})-\nabla f(W\_{t})\right)\right\|\_{F}^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1B2​𝔼​[‖∑i=1B(∇f​(Wt;ξt,i)−∇f​(Wt))‖F2]\displaystyle=\frac{1}{B^{2}}\mathbb{E}\left[\left\|\sum\_{i=1}^{B}\left(\nabla f(W\_{t};\xi\_{t,i})-\nabla f(W\_{t})\right)\right\|\_{F}^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1B2​∑i=1B𝔼​[‖∇f​(Wt;ξt,i)−∇f​(Wt)‖F2]≤σ2B.\displaystyle=\frac{1}{B^{2}}\sum\_{i=1}^{B}\mathbb{E}\!\left[\|\nabla f(W\_{t};\xi\_{t,i})-\nabla f(W\_{t})\|\_{F}^{2}\right]\leq\frac{\sigma^{2}}{B}. |  |

where the last equality is due to i.i.d. uniform sampling with 𝔼​[∇f​(Wt;ξt,i)]=∇f​(Wt)\mathbb{E}[\nabla f(W\_{t};\xi\_{t,i})]=\nabla f(W\_{t}) and independence.
The last inequality is due to 𝔼​[‖∇f​(W;ξt,i)−∇f​(W)‖F2]≤σ2\mathbb{E}[\|\nabla f(W;\xi\_{t,i})-\nabla f(W)\|\_{F}^{2}]\leq\sigma^{2}.
According to Jensen’s inequality, we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖Gt−∇f​(Wt)‖F]≤𝔼​[‖Gt−∇f​(Wt)‖F2]=σB.\displaystyle\mathbb{E}[\|G\_{t}-\nabla f(W\_{t})\|\_{F}]\leq\sqrt{\mathbb{E}[\|G\_{t}-\nabla f(W\_{t})\|\_{F}^{2}]}=\frac{\sigma}{\sqrt{B}}. |  |

∎

### A.3 Newton–Schulz polynomial

Newton–Schulz steps orthogonalize a matrix XX via Newton–Schulz steps applied to A=X​X⊤A=XX^{\top}.
Define Newton–Schulz polynomial pκp\_{\kappa} the degree is κ\kappa.
The following are the properties of pκp\_{\kappa} along with their proofs.

Newton–Schulz polynomial pκp\_{\kappa}.

For degree κ∈ℕ\kappa\in\mathbb{N}, the Newton–Schulz polynomial is
the Taylor truncation of 1/λ1/\sqrt{\lambda} at λ=1\lambda=1, i.e.,

|  |  |  |
| --- | --- | --- |
|  | p(s)​(1)=dsd​λs​λ−1/2|λ=1\displaystyle p^{(s)}(1)=\frac{d^{s}}{d\lambda^{s}}\lambda^{-1/2}\Bigg|\_{\lambda=1} |  |

for s=1,…,κs=1,\ldots,\kappa.
The explicit form of the Newton–Schulz polynomial for degree κ\kappa is

|  |  |  |
| --- | --- | --- |
|  | pκ​(λ)=∑s=0κcs​(1−λ)s,cs=(2​s)!4s​(s!)2>0.\displaystyle p\_{\kappa}(\lambda)=\sum\_{s=0}^{\kappa}c\_{s}(1-\lambda)^{s},\qquad c\_{s}=\frac{(2s)!}{4^{s}(s!)^{2}}>0. |  |

Equivalently, with reparametrization u=1−λ∈[0,1]u=1-\lambda\in[0,1] and pκ​(1−u)=∑s=0κcs​usp\_{\kappa}(1-u)=\sum\_{s=0}^{\kappa}c\_{s}u^{s}.

Proposition [1](#Thmprop1 "Proposition 1 (Properties of 𝑝_𝜅). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") (Properties of pκp\_{\kappa}).
For λ∈[0,1]\lambda\in[0,1]:

* •

  Positivity.
  pκ​(λ)>0p\_{\kappa}(\lambda)>0 and pκ​(λ)≥1p\_{\kappa}(\lambda)\geq 1 with equality if and only if λ=1\lambda=1.
* •

  Monotonicity of τ\tau.
  Let τ​(λ):=λ​[pκ​(λ)]2\tau(\lambda):=\lambda[p\_{\kappa}(\lambda)]^{2}; then
  we have τ\tau non‑decreasing on [0,1][0,1] and τ​(1)=1\tau(1)=1.

Consequently, for any symmetric A⪰0A\succeq 0 with spectrum in [0,1][0,1],
the Newton–Schulz update A↦pκ​(A)​A​pκ​(A)A\mapsto p\_{\kappa}(A)Ap\_{\kappa}(A) satisfies
‖pκ​(A)​A​pκ​(A)‖op≤1\|p\_{\kappa}(A)Ap\_{\kappa}(A)\|\_{\mathrm{op}}\leq 1: Newton–Schulz steps preserve the unit spectral ball
(see Appendix [A.3](#A1.SS3 "A.3 Newton–Schulz polynomial ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).

proof.

Positivity.
pκ​(λ)>0p\_{\kappa}(\lambda)>0 and pκ​(λ)≥1p\_{\kappa}(\lambda)\geq 1 with equality if and only if λ=1\lambda=1.

###### Proof.

Separating the first term (for s=0s=0) from the summation that defines pκ​(λ)p\_{\kappa}(\lambda):

|  |  |  |
| --- | --- | --- |
|  | pκ​(λ)=∑s=0κcs​(1−λ)s=c0​(a−λ)0+∑s=1κcs​(1−λ)s\displaystyle p\_{\kappa}(\lambda)=\sum\_{s=0}^{\kappa}c\_{s}(1-\lambda)^{s}=c\_{0}(a-\lambda)^{0}+\sum\_{s=1}^{\kappa}c\_{s}(1-\lambda)^{s} |  |

The coefficient c0c\_{0} is (2⋅0)!40​(0!)2=1\frac{(2\cdot 0)!}{4^{0}(0!)^{2}}=1.
The rest of the polynomial is the sum ∑s=1κcs​(1−λ)s\sum\_{s=1}^{\kappa}c\_{s}(1-\lambda)^{s}.
For any term in this sum and for any λ∈[0,1]\lambda\in[0,1], the coefficients csc\_{s} are strictly positive for all s≥1s\geq 1.
Also, for any exponent s≥1s\geq 1, (1−λ)s(1-\lambda)^{s} is non-negative in the range [0,1][0,1].
Each term cs​(1−λ)sc\_{s}(1-\lambda)^{s} is a product of a positive number and a non-negative number, which means each term is non-negative.
Hence,

|  |  |  |
| --- | --- | --- |
|  | pκ​(λ)=1+∑s=1κcs​(1−λ)s≥1+0=1\displaystyle p\_{\kappa}(\lambda)=1+\sum\_{s=1}^{\kappa}c\_{s}(1-\lambda)^{s}\geq 1+0=1 |  |

This proves that pκ​(λ)≥1p\_{\kappa}(\lambda)\geq 1 for all λ∈[0,1]\lambda\in[0,1].
It is also trivially true that pκ​(λ)>0p\_{\kappa}(\lambda)>0.
∎

Monotonicity of τ​(λ):=λ​[pκ​(λ)]2\tau(\lambda):=\lambda[p\_{\kappa}(\lambda)]^{2}.

###### Proof.

First note τ​(1)=1⋅pκ​(1)2=1\tau(1)=1\cdot p\_{\kappa}(1)^{2}=1.
For monotonicity, differentiate:

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=pκ​(λ)2+2​λ​pκ​(λ)​pκ′​(λ)=pκ​(λ)​(pκ​(λ)+2​λ​pκ′​(λ)).\displaystyle\tau^{\prime}(\lambda)=p\_{\kappa}(\lambda)^{2}+2\lambda p\_{\kappa}(\lambda)p\_{\kappa}^{\prime}(\lambda)=p\_{\kappa}(\lambda)\left(p\_{\kappa}(\lambda)+2\lambda p\_{\kappa}^{\prime}(\lambda)\right). |  |

Set u=1−λu=1-\lambda and define q​(u):=pκ​(1−u)=∑s=0κcs​usq(u):=p\_{\kappa}(1-u)=\sum\_{s=0}^{\kappa}c\_{s}u^{s}.
Then

|  |  |  |
| --- | --- | --- |
|  | pκ​(λ)=q​(u),pκ′​(λ)=dd​λ​q​(1−λ)=−q′​(u),λ=1−u.\displaystyle p\_{\kappa}(\lambda)=q(u),\qquad p\_{\kappa}^{\prime}(\lambda)=\frac{d}{d\lambda}q(1-\lambda)=-q^{\prime}(u),\qquad\lambda=1-u. |  |

Hence

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=q​(u)​(q​(u)−2​(1−u)​q′​(u))⏟:=S​(u).\displaystyle\tau^{\prime}(\lambda)=q(u)\underbrace{\left(q(u)-2(1-u)q^{\prime}(u)\right)}\_{:=S(u)}. |  |

Now, we compute S​(u)S(u) in closed form.
Since

|  |  |  |
| --- | --- | --- |
|  | q​(u)=∑s=0κcs​us,q′​(u)=∑s=1κs​cs​us−1,\displaystyle q(u)=\sum\_{s=0}^{\kappa}c\_{s}u^{s},\qquad q^{\prime}(u)=\sum\_{s=1}^{\kappa}sc\_{s}u^{s-1}, |  |

we obtain

|  |  |  |
| --- | --- | --- |
|  | S​(u)=∑s=0κcs​us−2​(1−u)​∑s=1κs​cs​us−1.\displaystyle S(u)=\sum\_{s=0}^{\kappa}c\_{s}u^{s}-2(1-u)\sum\_{s=1}^{\kappa}sc\_{s}u^{s-1}. |  |

Expanding and collecting coefficients of utu^{t} gives

|  |  |  |
| --- | --- | --- |
|  | S​(u)=(c0−2​c1)+∑t=1κ−1((2​t+1)​ct−2​(t+1)​ct+1)​ut+(2​κ+1)​cκ​uκ.\displaystyle S(u)=(c\_{0}-2c\_{1})+\sum\_{t=1}^{\kappa-1}\left((2t+1)c\_{t}-2(t+1)c\_{t+1}\right)u^{t}+(2\kappa+1)c\_{\kappa}u^{\kappa}. |  |

Using the ratio identity

|  |  |  |
| --- | --- | --- |
|  | cs+1cs=(2​s+2)!​(s!)2​4s(2​s)!​((s+1)!)2​4s+1=(2​s+2)​(2​s+1)4​(s+1)2=2​s+12​(s+1),\displaystyle\frac{c\_{s+1}}{c\_{s}}=\frac{(2s+2)!(s!)^{2}4^{s}}{(2s)!((s+1)!)^{2}4^{s+1}}=\frac{(2s+2)(2s+1)}{4(s+1)^{2}}=\frac{2s+1}{2(s+1)}, |  |

for t=0,…,κ−1t=0,\ldots,\kappa-1,

|  |  |  |
| --- | --- | --- |
|  | (2​t+1)​ct−2​(t+1)​ct+1=(2​t+1)​ct−2​(t+1)​(2​t+12​(t+1)​ct)=0,\displaystyle(2t+1)c\_{t}-2(t+1)c\_{t+1}=(2t+1)c\_{t}-2(t+1)\left(\frac{2t+1}{2(t+1)}c\_{t}\right)=0, |  |

and also c0−2​c1=1−2⋅12=0c\_{0}-2c\_{1}=1-2\cdot\frac{1}{2}=0.
Therefore, all coefficients up to degree κ−1\kappa-1 vanish, and we are left with

|  |  |  |
| --- | --- | --- |
|  | S​(u)=(2​κ+1)​cκ​uκ≥0for ​u∈[0,1],\displaystyle S(u)=(2\kappa+1)c\_{\kappa}u^{\kappa}\geq 0\qquad\text{for }u\in[0,1], |  |

with strict positivity for u>0u>0 when κ≥1\kappa\geq 1.
Since q​(u)>0q(u)>0 on [0,1][0,1], we conclude

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=q​(u)​S​(u)≥0for all ​λ∈[0,1],\displaystyle\tau^{\prime}(\lambda)=q(u)S(u)\geq 0\qquad\text{for all }\lambda\in[0,1], |  |

i.e., τ\tau is non-decreasing on [0,1][0,1], and τ′​(λ)>0\tau^{\prime}(\lambda)>0 for λ∈[0,1)\lambda\in[0,1) when κ≥1\kappa\geq 1.
Together with τ​(1)=1\tau(1)=1, this proves the proposition.
∎

##### NS step preserves the unit spectral ball.

Let A⪰0A\succeq 0 be symmetric with spectrum σ​(A)⊂[0,1]\sigma(A)\subset[0,1].
By spectral calculus,

|  |  |  |
| --- | --- | --- |
|  | pκ​(A)​A​pκ​(A)=U​diag⁡(λi​pκ​(λi)2)​U⊤,\displaystyle p\_{\kappa}(A)Ap\_{\kappa}(A)=U\operatorname{diag}\left(\lambda\_{i}p\_{\kappa}(\lambda\_{i})^{2}\right)U^{\top}, |  |

where A=U​diag⁡(λi)​U⊤A=U\operatorname{diag}(\lambda\_{i})U^{\top}. Thus

|  |  |  |
| --- | --- | --- |
|  | ‖pκ​(A)​A​pκ​(A)‖op=maxi⁡(λi​pκ​(λi)2)=maxi⁡τ​(λi)≤τ​(1)=1,\displaystyle\|p\_{\kappa}(A)Ap\_{\kappa}(A)\|\_{\mathrm{op}}=\max\_{i}\left(\lambda\_{i}p\_{\kappa}(\lambda\_{i})^{2}\right)=\max\_{i}\tau(\lambda\_{i})\leq\tau(1)=1, |  |

because τ\tau is non-decreasing on [0,1][0,1].
Hence, the Newton–Schulz update maps the unit spectral ball into itself.

## Appendix B Muon with Finite Newton–Schulz Iteration

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")  Muon with Newton–Schulz Orthogonalization

0: learning rate η>0\eta>0, momentum β∈[0,1)\beta\in[0,1), Newton–Schulz steps q∈ℕq\in\mathbb{N}, Newton–Schulz polynomial pκp\_{\kappa} with degree κ\kappa, batch size BB, total iteration TT.

1: Initialize: M0←0M\_{0}\leftarrow 0, W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n}

2: for t=1t=1 to TT do

3:  Gt←1B​∑i=1B∇f​(Wt−1;ξt,i)G\_{t}\leftarrow\frac{1}{B}\sum\_{i=1}^{B}\nabla f(W\_{t-1};\xi\_{t,i})

4:  Mt←β​Mt−1+GtM\_{t}\leftarrow\beta M\_{t-1}+G\_{t}

5:  Xt,0←Mt/αtX\_{t,0}\leftarrow M\_{t}/\alpha\_{t} with αt=max⁡{1,‖Mt‖F}\alpha\_{t}=\max\{1,\|M\_{t}\|\_{F}\} *(scaling ensures ‖Xt,0‖op≤1\|X\_{t,0}\|\_{\mathrm{op}}\leq 1)*

6:  for j=1j=1 to qq do

7:   Xt,j←pκ​(Xt,j−1​Xt,j−1⊤)​Xt,j−1X\_{t,j}\leftarrow p\_{\kappa}(X\_{t,j-1}X\_{t,j-1}^{\top})\,X\_{t,j-1}

8:  end for

9:  Ot←Xt,qO\_{t}\leftarrow X\_{t,q}

10:  Wt←Wt−1−η​OtW\_{t}\leftarrow W\_{t-1}-\eta O\_{t}

11: end for

###### Proof of Theorem [1](#Thmtheorem1 "Theorem 1 (Convergence of Muon with Newton–Schulz). ‣ 4.1 Convergence of Muon (with Newton–Schulz) ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

First, we introduce the scaled EMA momentum: Nt:=(1−β)​MtN\_{t}:=(1-\beta)M\_{t}.
Then, we get Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t}.
Note that the polar factor is scale-invariant (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(v)).
Let PtP\_{t} be the polar factor of MtM\_{t}, i.e., Pt=Polar⁡(Mt)P\_{t}=\operatorname{Polar}(M\_{t}).
Hence, Pt:=Polar⁡(Nt)=Polar⁡(Mt)P\_{t}:=\operatorname{Polar}(N\_{t})=\operatorname{Polar}(M\_{t}).

By Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
we start from descent lemma (Lemma [6](#Thmlemma6 "Lemma 6 (Descent Lemma). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).
Since Wt=Wt−1−η​OtW\_{t}=W\_{t-1}-\eta O\_{t}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)+⟨∇f​(Wt−1),Wt−Wt−1⟩F+L2​‖Wt−Wt−1‖op2\displaystyle\leq f(W\_{t-1})+\langle\nabla f(W\_{t-1}),W\_{t}-W\_{t-1}\rangle\_{F}+\frac{L}{2}\|W\_{t}-W\_{t-1}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨∇f​(Wt−1),Ot⟩F+L2​η2​‖Ot‖op2\displaystyle=f(W\_{t-1})-\eta\langle\nabla f(W\_{t-1}),O\_{t}\rangle\_{F}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨Nt,Ot⟩F+η​⟨Nt−∇f​(Wt−1),Ot⟩F+L2​η2​‖Ot‖op2\displaystyle=f(W\_{t-1})-\eta\langle N\_{t},O\_{t}\rangle\_{F}+\eta\langle N\_{t}-\nabla f(W\_{t-1}),O\_{t}\rangle\_{F}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η​⟨Nt,Ot⟩F+η​‖Nt−∇f​(Wt−1)‖∗​‖Ot‖op+L2​η2​‖Ot‖op2\displaystyle\leq f(W\_{t-1})-\eta\langle N\_{t},O\_{t}\rangle\_{F}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|O\_{t}\|\_{\mathrm{op}}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨Nt,Pt⟩F+η​⟨Nt,Pt−Ot⟩F+η​‖Nt−∇f​(Wt−1)‖∗​‖Ot‖op+L2​η2​‖Ot‖op2\displaystyle=f(W\_{t-1})-\eta\langle N\_{t},P\_{t}\rangle\_{F}+\eta\langle N\_{t},P\_{t}-O\_{t}\rangle\_{F}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|O\_{t}\|\_{\mathrm{op}}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​‖Nt‖∗+η​⟨Nt,Pt−Ot⟩F+η​‖Nt−∇f​(Wt−1)‖∗​‖Ot‖op+L2​η2​‖Ot‖op2\displaystyle=f(W\_{t-1})-\eta\|N\_{t}\|\_{\*}+\eta\langle N\_{t},P\_{t}-O\_{t}\rangle\_{F}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|O\_{t}\|\_{\mathrm{op}}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η​‖Nt‖∗+η​‖Nt‖∗​‖Pt−Ot‖op+η​‖Nt−∇f​(Wt−1)‖∗​‖Ot‖op+L2​η2​‖Ot‖op2\displaystyle\leq f(W\_{t-1})-\eta\|N\_{t}\|\_{\*}+\eta\|N\_{t}\|\_{\*}\|P\_{t}-O\_{t}\|\_{\mathrm{op}}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|O\_{t}\|\_{\mathrm{op}}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |

where the second inequality and the third inequality are due to Hölder’s inequality (Lemma [4](#Thmlemma4 "Lemma 4 (Hölder’s inequality). ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")), and
the last equality is due to ⟨Nt,Pt⟩F=‖Nt‖∗\langle N\_{t},P\_{t}\rangle\_{F}=\|N\_{t}\|\_{\*} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(ii)).

Now, we define the polar approximation error, which is the error between the exact polar factor Pt=Polar⁡(Nt)=Polar⁡(Mt)P\_{t}=\operatorname{Polar}(N\_{t})=\operatorname{Polar}(M\_{t}) and the actual step OtO\_{t} generated by qq-step Newton–Schulz steps, i.e., Ot=Newton–Schulz​(Mt,q)O\_{t}=\text{$\textsc{Newton--Schulz}$}(M\_{t},q), measured in the operator norm.

|  |  |  |
| --- | --- | --- |
|  | εt,q:=‖Ot−Pt‖opandεq:=suptεt,q\displaystyle\varepsilon\_{t,q}:=\|O\_{t}-P\_{t}\|\_{\mathrm{op}}\quad\text{and}\quad\varepsilon\_{q}:=\sup\_{t}\varepsilon\_{t,q} |  |

Since ‖Pt‖op≤1\|P\_{t}\|\_{\mathrm{op}}\leq 1,
‖Ot‖op≤1+εt,q≤1+εq\|O\_{t}\|\_{\mathrm{op}}\leq 1+\varepsilon\_{t,q}\leq 1+\varepsilon\_{q}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)−η​(1−‖Pt−Ot‖op)​‖Nt‖∗+η​‖Nt−∇f​(Wt−1)‖∗​‖Ot‖op+L2​η2​‖Ot‖op2\displaystyle\leq f(W\_{t-1})-\eta(1-\|P\_{t}-O\_{t}\|\_{\mathrm{op}})\|N\_{t}\|\_{\*}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|O\_{t}\|\_{\mathrm{op}}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η​(1−εt,q)​‖Nt‖∗+η​(1+εt,q)​‖Nt−∇f​(Wt−1)‖∗+L2​η2​(1+εt,q)2\displaystyle\leq f(W\_{t-1})-\eta(1-\varepsilon\_{t,q})\|N\_{t}\|\_{\*}+\eta(1+\varepsilon\_{t,q})\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}+\frac{L}{2}\eta^{2}(1+\varepsilon\_{t,q})^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η​(1−εt,q)​‖∇f​(Wt−1)‖∗+η​(1−εt,q)​‖∇f​(Wt−1)−Nt‖∗\displaystyle\leq f(W\_{t-1})-\eta(1-\varepsilon\_{t,q})\|\nabla f(W\_{t-1})\|\_{\*}+\eta(1-\varepsilon\_{t,q})\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +η​(1+εt,q)​‖Nt−∇f​(Wt−1)‖∗+L2​η2​(1+εt,q)2\displaystyle\qquad+\eta(1+\varepsilon\_{t,q})\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}+\frac{L}{2}\eta^{2}(1+\varepsilon\_{t,q})^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​(1−εt,q)​‖∇f​(Wt−1)‖∗+2​η​‖Nt−∇f​(Wt)‖∗+L2​η2​(1+εt,q)2\displaystyle=f(W\_{t-1})-\eta(1-\varepsilon\_{t,q})\|\nabla f(W\_{t-1})\|\_{\*}+2\eta\|N\_{t}-\nabla f(W\_{t})\|\_{\*}+\frac{L}{2}\eta^{2}(1+\varepsilon\_{t,q})^{2} |  |

where the last inequality is due to −‖A‖∗≤−‖B‖∗+‖A−B‖∗-\|A\|\_{\*}\leq-\|B\|\_{\*}+\|A-B\|\_{\*}
and the last equality holds because (1−εt,q)+(1+εt,q)=2(1-\varepsilon\_{t,q})+(1+\varepsilon\_{t,q})=2.
Since εt,q≤εq\varepsilon\_{t,q}\leq\varepsilon\_{q}, we arrive at the clean one-step inequality as

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)≤f​(Wt−1)−η​(1−εq)​‖∇f​(Wt−1)‖∗+2​η​‖∇f​(Wt−1)−Nt‖∗+L2​η2​(1+εq)2.f(W\_{t})\leq f(W\_{t-1})-\eta(1-\varepsilon\_{q})\|\nabla f(W\_{t-1})\|\_{\*}+2\eta\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}+\tfrac{L}{2}\eta^{2}(1+\varepsilon\_{q})^{2}. |  | (1) |

Rearranging and taking the expectation yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​‖∇f​(Wt−1)‖∗≤𝔼​[f​(Wt−1)−f​(Wt)]η​(1−εq)+21−εq​𝔼​‖∇f​(Wt−1)−Nt‖∗+L​η​(1+εq)22​(1−εq).\mathbb{E}\|\nabla f(W\_{t-1})\|\_{\*}\leq\frac{\mathbb{E}[f(W\_{t-1})-f(W\_{t})]}{\eta(1-\varepsilon\_{q})}+\frac{2}{1-\varepsilon\_{q}}\mathbb{E}\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}+\frac{L\eta(1+\varepsilon\_{q})^{2}}{2(1-\varepsilon\_{q})}. |  | (2) |

Bounding 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}].

In order to bound 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}],
we introduce the true scaled momentum N¯t\bar{N}\_{t} defined by the true (full-batch) gradient ∇f​(Wt)\nabla f(W\_{t}) instead of GtG\_{t} for each step tt:

* •

  N¯t=β​N¯t−1+(1−β)​∇f​(Wt−1)\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t-1}) for t>0t>0
  and N¯0=0\bar{N}\_{0}=0.
* •

  Note that Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t} for t>0t>0 and N0=0N\_{0}=0.

Then we can decompose 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}] as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)−Nt‖∗]≤𝔼​[‖∇f​(Wt−1)−N¯t‖∗]⏟Term (A)+𝔼​[‖N¯t−Nt‖∗]⏟Term (B)\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}]\leq\underbrace{\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{\*}]}\_{\text{Term (A)}}+\underbrace{\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{\*}]}\_{\text{Term (B)}} |  | (3) |

Bounding the Term (A).
Let Dt=‖∇f​(Wt−1)−N¯t‖∗D\_{t}=\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{\*}.
From the recursion,

|  |  |  |
| --- | --- | --- |
|  | ∇f​(Wt−1)−N¯t=β​(∇f​(Wt−1)−N¯t−1)\displaystyle\nabla f(W\_{t-1})-\bar{N}\_{t}=\beta(\nabla f(W\_{t-1})-\bar{N}\_{t-1}) |  |

Hence, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dt\displaystyle D\_{t} | =β​‖∇f​(Wt−1)−N¯t−1‖∗\displaystyle=\beta\|\nabla f(W\_{t-1})-\bar{N}\_{t-1}\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤β​‖∇f​(Wt−2)−N¯t−1‖+β​‖∇f​(Wt−1)−∇f​(Wt−2)‖∗\displaystyle\leq\beta\|\nabla f(W\_{t-2})-\bar{N}\_{t-1}\|+\beta\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​Dt−1+β​‖∇f​(Wt−1)−∇f​(Wt−2)‖∗.\displaystyle=\beta D\_{t-1}+\beta\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*}. |  |

Applying Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and using the fact that
‖Ot−1‖op≤1+εt−1,q≤1+εq\|O\_{t-1}\|\_{\mathrm{op}}\leq 1+\varepsilon\_{t-1,q}\leq 1+\varepsilon\_{q}, we have

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(Wt−1)−∇f​(Wt−2)‖∗≤L​‖Wt−1−Wt−2‖op=L​η​‖Ot−1‖op≤L​η​(1+εq)\displaystyle\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*}\leq L\|W\_{t-1}-W\_{t-2}\|\_{\mathrm{op}}=L\eta\|O\_{t-1}\|\_{\mathrm{op}}\leq L\eta(1+\varepsilon\_{q}) |  |

Hence, we have the following recursion,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dt\displaystyle D\_{t} | ≤β​Dt−1+β​L​η​(1+εq)\displaystyle\leq\beta D\_{t-1}+\beta L\eta(1+\varepsilon\_{q}) |  |

Since N¯0=0\bar{N}\_{0}=0,
we have

|  |  |  |
| --- | --- | --- |
|  | D1=‖∇f​(W0)−N¯1‖∗=‖∇f​(W0)−(β​N¯0+(1−β)​∇f​(W0))‖∗=β​‖∇f​(W0)‖∗\displaystyle D\_{1}=\|\nabla f(W\_{0})-\bar{N}\_{1}\|\_{\*}=\|\nabla f(W\_{0})-(\beta\bar{N}\_{0}+(1-\beta)\nabla f(W\_{0}))\|\_{\*}=\beta\|\nabla f(W\_{0})\|\_{\*} |  |

By unrolling the recursion and taking the expectation, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[Dt]≤βt​𝔼​[‖∇f​(W0)‖∗]+∑i=1tβi​L​η​(1+εq)≤βt​‖∇f​(W0)‖∗+β​L​η​(1+εq)1−β\displaystyle\mathbb{E}[D\_{t}]\leq\beta^{t}\mathbb{E}[\|\nabla f(W\_{0})\|\_{\*}]+\sum\_{i=1}^{t}\beta^{i}L\eta(1+\varepsilon\_{q})\leq\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\frac{\beta L\eta(1+\varepsilon\_{q})}{1-\beta} |  | (4) |

Bounding the Term (B).
When we unroll the EMA recursion of both NtN\_{t} and N¯t\bar{N}\_{t},

|  |  |  |
| --- | --- | --- |
|  | Nt=β​Nt−1+(1−β)​Gt=βt​N0+(1−β)​∑i=1tβt−i​Gi\displaystyle N\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t}=\beta^{t}N\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}G\_{i} |  |
|  |  |  |
| --- | --- | --- |
|  | N¯t=β​N¯t−1+(1−β)​∇f​(Wt)=βt​N¯0+(1−β)​∑i=1tβt−i​∇f​(Wi)\displaystyle\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t})=\beta^{t}\bar{N}\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}\nabla f(W\_{i}) |  |

Then, we compute the expectation of the Frobenius norm bias caused by MtM\_{t} and M¯t\bar{M}\_{t},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F]≤(1−β)​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}]\leq(1-\beta)\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}\right] |  | (5) |

Applying Jensen’s inequality and the linearity of expectation gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1−β)​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F]\displaystyle(1-\beta)\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}\right] | ≤(1−β)2​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F2]\displaystyle\leq\sqrt{(1-\beta)^{2}\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−β)​𝔼[∑i=1tβ2​(t−i)∥Gi−∇f(Wi))∥F2]\displaystyle=(1-\beta)\sqrt{\mathbb{E}\left[\sum\_{i=1}^{t}\beta^{2(t-i)}\left\|G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−β)​∑i=1tβ2​(t−i)​𝔼​[‖Gi−∇f​(Wi)‖F2]\displaystyle=(1-\beta)\sqrt{\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i})\|\_{F}^{2}]} |  |

By Lemma [8](#Thmlemma8 "Lemma 8 (Unbiasedness and bounded variance). ‣ A.2.2 Assumption 2 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz"), Eq.([5](#A2.E5 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")) becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}] | ≤(1−β)​∑i=1tβ2​(t−i)​𝔼​[‖Gi−∇f​(Wi)‖F2]\displaystyle\leq(1-\beta)\sqrt{\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i})\|\_{F}^{2}]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤(1−β)​1−β2​t1−β2​σ2B≤1−β1+β​σB\displaystyle\leq(1-\beta)\sqrt{\frac{1-\beta^{2t}}{1-\beta^{2}}\frac{\sigma^{2}}{B}}\leq\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sigma}{\sqrt{B}} |  |

Using the fact that ‖X‖∗≤r​‖X‖F\|X\|\_{\*}\leq\sqrt{r}\|X\|\_{F} for X∈ℝm×nX\in\mathbb{R}^{m\times n} with r=min⁡{m,n}r=\min\{m,n\} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iii)), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖N¯t−Nt‖∗]≤r​𝔼​[‖N¯t−Nt‖F]≤1−β1+β​r​σB\displaystyle\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{\*}]\leq\sqrt{r}\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{F}]\leq\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}} |  | (6) |

Plugging Eq.([4](#A2.E4 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")) and Eq.([6](#A2.E6 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")) into Eq.([3](#A2.E3 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)−Nt‖∗]≤β​L​η​(1+εq)1−β+βt​‖∇f​(W0)‖∗+1−β1+β​r​σB\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}]\leq\frac{\beta L\eta(1+\varepsilon\_{q})}{1-\beta}+\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}} |  | (7) |

Averaging and tuning.
Plugging Eq.([7](#A2.E7 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")) into Eq.([2](#A2.E2 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")),
we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤𝔼​[f​(Wt−1)−f​(Wt)]η​(1−εq)\displaystyle\leq\frac{\mathbb{E}[f(W\_{t-1})-f(W\_{t})]}{\eta(1-\varepsilon\_{q})} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +21−εq​(β​L​η​(1+εq)1−β+βt​‖∇f​(W0)‖∗+1−β1+β​r​σB)+L​η​(1+εq)22​(1−εq)\displaystyle\quad+\frac{2}{1-\varepsilon\_{q}}\left(\frac{\beta L\eta(1+\varepsilon\_{q})}{1-\beta}+\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right)+\frac{L\eta(1+\varepsilon\_{q})^{2}}{2(1-\varepsilon\_{q})} |  |

Averaging over t=1,…​Tt=1,\ldots T, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤DT​η​(1−εq)+21−εq​(β​L​η​(1+εq)1−β+1−β1+β​r​σB)\displaystyle\leq\frac{D}{T\eta(1-\varepsilon\_{q})}+\frac{2}{1-\varepsilon\_{q}}\left(\frac{\beta L\eta(1+\varepsilon\_{q})}{1-\beta}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +21−εq​(1T​∑t=1Tβt​‖∇f​(W0)‖∗)+L​η​(1+εq)22​(1−εq)\displaystyle\quad+\frac{2}{1-\varepsilon\_{q}}\left(\frac{1}{T}\sum\_{t=1}^{T}\beta^{t}\|\nabla f(W\_{0})\|\_{\*}\right)+\frac{L\eta(1+\varepsilon\_{q})^{2}}{2(1-\varepsilon\_{q})} |  |

Using Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iii) and Lemma [7](#Thmlemma7 "Lemma 7 (Gradient-gap inequality). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz"),
‖∇f​(W0)‖∗≤r​‖∇f​(W0)‖F≤2​r​L​D\|\nabla f(W\_{0})\|\_{\*}\leq\sqrt{r}\,\|\nabla f(W\_{0})\|\_{F}\leq\sqrt{2rLD},
where D=f​(W0)−f∗D=f(W\_{0})-f^{\*}.
Then, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤DT​η​(1−εq)+21−εq​(β​L​η​(1+εq)1−β+1−β1+β​r​σB)\displaystyle\leq\frac{D}{T\eta(1-\varepsilon\_{q})}+\frac{2}{1-\varepsilon\_{q}}\left(\frac{\beta L\eta(1+\varepsilon\_{q})}{1-\beta}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +2​β​2​r​L​D(1−εq)​T​(1−β)+L​η​(1+εq)22​(1−εq)\displaystyle\quad+\frac{2\beta\sqrt{2rLD}}{(1-\varepsilon\_{q})T(1-\beta)}+\frac{L\eta(1+\varepsilon\_{q})^{2}}{2(1-\varepsilon\_{q})} |  |

where D=f​(W0)−f∗D=f(W\_{0})-f^{\*}.
By setting η=(1−β)​DT​L\eta=\sqrt{\frac{(1-\beta)D}{TL}},
we obtain

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] |  |
|  |  |  |
| --- | --- | --- |
|  | ≤11−εq​(DT​η+L​η2​(1+εq)2+2​β​L​η​(1+εq)1−β+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β))\displaystyle\qquad\leq\frac{1}{1-\varepsilon\_{q}}\left(\frac{D}{T\eta}+\frac{L\eta}{2}(1+\varepsilon\_{q})^{2}+\frac{2\beta L\eta(1+\varepsilon\_{q})}{1-\beta}+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =11−εq​(L​DT​(1+2​β​(1+εq)1−β+(1+εq)2​1−β2)+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β))\displaystyle\qquad=\frac{1}{1-\varepsilon\_{q}}\left(\sqrt{\frac{LD}{T}}\left(\frac{1+2\beta(1+\varepsilon\_{q})}{\sqrt{1-\beta}}+\frac{(1+\varepsilon\_{q})^{2}\sqrt{1-\beta}}{2}\right)+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)}\right) |  |

Setting β=1−min⁡{L​D​Bσ​r​T,1}\beta=1-\min\left\{\frac{\sqrt{LDB}}{\sigma\sqrt{rT}},1\right\}, we get

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] |  |
|  |  |  |
| --- | --- | --- |
|  | ≤11−εq​((1+εq)2​1−β2​L​DT+1+2​β​(1+εq)1−β​L​DT+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β))\displaystyle\leq\frac{1}{1-\varepsilon\_{q}}\left(\frac{(1+\varepsilon\_{q})^{2}\sqrt{1-\beta}}{2}\sqrt{\frac{LD}{T}}+\frac{1+2\beta(1+\varepsilon\_{q})}{\sqrt{1-\beta}}\sqrt{\frac{LD}{T}}+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)}\right) |  |
|  |  |  |
| --- | --- | --- |
|  | =11−εq​((1+εq)2​1−β2​L​DT+(1+2​β​(1+εq)+21+β)​(r​σ2​L​DB​T)1/4+2​2​β​σ​rB​T)\displaystyle\qquad=\frac{1}{1-\varepsilon\_{q}}\left(\frac{(1+\varepsilon\_{q})^{2}\sqrt{1-\beta}}{2}\sqrt{\frac{LD}{T}}+\left(1+2\beta(1+\varepsilon\_{q})+\frac{2}{\sqrt{1+\beta}}\right)\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}+2\sqrt{2}\beta\frac{\sigma r}{\sqrt{BT}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =11−εq⋅𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4)\displaystyle\qquad=\frac{1}{1-\varepsilon\_{q}}\cdot\mathcal{O}\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right) |  | (8) |

From Newton–Schulz residual to factor χq\chi\_{q}.
By the Newton–Schulz residual–error link (Lemma [1](#Thmlemma1 "Lemma 1 (Orthogonality residual vs. Polar approximation error). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"))
and the residual contraction (Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"))
as assembled in Corollary [1](#Thmcorollary1 "Corollary 1 (Final constant factor bound). ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz"), we have

|  |  |  |
| --- | --- | --- |
|  | εq=suptεt,q=supt(1−1−δt,q)≤1−1−δ0(κ+1)q.\displaystyle\varepsilon\_{q}=\sup\_{t}\varepsilon\_{t,q}=\sup\_{t}\bigl(1-\sqrt{1-\delta\_{t,q}}\bigr)\leq 1-\sqrt{1-\delta\_{0}^{(\kappa+1)^{q}}}. |  |

Hence, we bound the factor χq\chi\_{q} as

|  |  |  |
| --- | --- | --- |
|  | χq=(1−εq)−1≤[1−δ0(κ+1)q]−1/2\displaystyle\chi\_{q}=(1-\varepsilon\_{q})^{-1}\leq[1-\delta\_{0}^{(\kappa+1)^{q}}]^{-1/2} |  |

with δ0=suptδt,0<1\delta\_{0}=\sup\_{t}\delta\_{t,0}<1 (Remark [1](#Thmremark1 "Remark 1. ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")).
Finally, we conclude from Eq.([8](#A2.E8 "In Proof of Theorem 1. ‣ Appendix B Muon with Finite Newton–Schulz Iteration ‣ Convergence of Muon with Newton–Schulz")) that

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​‖∇f​(Wt−1)‖∗≤χq⋅𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4),χq=11−εq,\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|\nabla f(W\_{t-1})\|\_{\*}\;\leq\;\chi\_{q}\cdot\mathcal{O}\!\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right),\qquad\chi\_{q}=\frac{1}{1-\varepsilon\_{q}}, |  |

with the constant factor bounded by

|  |  |  |
| --- | --- | --- |
|  | χq≤[ 1−δ0(κ+1)q]−1/2.\displaystyle\chi\_{q}\;\leq\;\bigl[\,1-\delta\_{0}^{(\kappa+1)^{q}}\,\bigr]^{-1/2}. |  |

Finally,
we can find an ϵ\epsilon-nuclear norm stationary point of ff with a complexity of

|  |  |  |
| --- | --- | --- |
|  | 𝒪​(max⁡{χq2​L​Dϵ2,χq2​r2​σ2B​ϵ2,χq4​r​σ2​L​DB​ϵ4})\displaystyle\mathcal{O}\left(\max\left\{\frac{\chi\_{q}^{2}LD}{\epsilon^{2}},\frac{\chi\_{q}^{2}r^{2}\sigma^{2}}{B\epsilon^{2}},\frac{\chi\_{q}^{4}r\sigma^{2}LD}{B\epsilon^{4}}\right\}\right) |  |

∎

## Appendix C Muon with SVD\mathrm{SVD} and SGD with momentum

### C.1 Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") (Muon with SVD)

Algorithm 2  Muon with SVD\mathrm{SVD}

0: learning rate η>0\eta>0, momentum β∈[0,1)\beta\in[0,1), Newton–Schulz steps q∈ℕq\in\mathbb{N}, batch size BB, Total iteration TT.

1: Initialize: M0←0M\_{0}\leftarrow 0, W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n}

2: for t=1t=1 to TT do

3:  Gt←1B​∑i=1B∇f​(Wt−1;ξt,i)G\_{t}\leftarrow\frac{1}{B}\sum\_{i=1}^{B}\nabla f(W\_{t-1};\xi\_{t,i})

4:  Mt←β​Mt−1+GtM\_{t}\leftarrow\beta M\_{t-1}+G\_{t}

5:  Ot←Polar⁡(Mt)O\_{t}\leftarrow\operatorname{Polar}(M\_{t}) (SVD​(Mt)=(Ut,Σt,Vt)\mathrm{SVD}(M\_{t})=(U\_{t},\Sigma\_{t},V\_{t}), then Polar⁡(Mt)=Ut​Vt⊤\operatorname{Polar}(M\_{t})=U\_{t}V\_{t}^{\top})

6:  Wt←Wt−1−η​OtW\_{t}\leftarrow W\_{t-1}-\eta O\_{t}

7: end for

8: return WTW\_{T}

###### Proof of Theorem [4](#Thmtheorem4 "Theorem 4 (Muon with SVD). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

First, we introduce the scaled EMA momentum: Nt:=(1−β)​MtN\_{t}:=(1-\beta)M\_{t}.
Then, we get Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t}.
Note that the polar factor is scale-invariant (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(v)).
Let PtP\_{t} be the polar factor of MtM\_{t}, i.e., Pt=Polar⁡(Mt)P\_{t}=\operatorname{Polar}(M\_{t}).
Hence, Pt:=Polar⁡(Nt)=Polar⁡(Mt)P\_{t}:=\operatorname{Polar}(N\_{t})=\operatorname{Polar}(M\_{t}).

By Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
we start from descent lemma (Lemma [6](#Thmlemma6 "Lemma 6 (Descent Lemma). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).
Since Wt=Wt−1−η​OtW\_{t}=W\_{t-1}-\eta O\_{t}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)+⟨∇f​(Wt−1),Wt−Wt−1⟩F+L2​‖Wt−Wt−1‖op2\displaystyle\leq f(W\_{t-1})+\langle\nabla f(W\_{t-1}),W\_{t}-W\_{t-1}\rangle\_{F}+\frac{L}{2}\|W\_{t}-W\_{t-1}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨∇f​(Wt−1),Ot⟩F+L2​η2​‖Ot‖op2\displaystyle=f(W\_{t-1})-\eta\langle\nabla f(W\_{t-1}),O\_{t}\rangle\_{F}+\frac{L}{2}\eta^{2}\|O\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨∇f​(Wt−1),Pt⟩F+L2​η2​‖Pt‖op2\displaystyle=f(W\_{t-1})-\eta\langle\nabla f(W\_{t-1}),P\_{t}\rangle\_{F}+\frac{L}{2}\eta^{2}\|P\_{t}\|\_{\mathrm{op}}^{2} |  |

where the last equality is due to Ot=Pt=Polar⁡(Mt)O\_{t}=P\_{t}=\operatorname{Polar}(M\_{t}).
Let SVD​(Mt)=(Ut,Σt,Vt)\mathrm{SVD}(M\_{t})=(U\_{t},\Sigma\_{t},V\_{t}). Then Pt=Polar⁡(Mt)=Ut​Vt⊤P\_{t}=\operatorname{Polar}(M\_{t})=U\_{t}V\_{t}^{\top}.
Since Pt=Polar⁡(Mt)P\_{t}=\operatorname{Polar}(M\_{t}) is a partial isometry, ‖Pt‖op≤1\|P\_{t}\|\_{\mathrm{op}}\leq 1 (and ‖Pt‖op=1\|P\_{t}\|\_{\mathrm{op}}=1 if rank​(Mt)>0\mathrm{rank}(M\_{t})>0).
Then, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)−η​⟨∇f​(Wt−1),Pt⟩F+L2​η2\displaystyle\leq f(W\_{t-1})-\eta\langle\nabla f(W\_{t-1}),P\_{t}\rangle\_{F}+\frac{L}{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​⟨Nt,Pt⟩F+η​⟨Nt−∇f​(Wt−1),Pt⟩F+L​η22\displaystyle=f(W\_{t-1})-\eta\langle N\_{t},P\_{t}\rangle\_{F}+\eta\langle N\_{t}-\nabla f(W\_{t-1}),P\_{t}\rangle\_{F}+\frac{L\eta^{2}}{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​‖Nt‖∗+η​⟨Nt−∇f​(Wt−1),Pt⟩F+L​η22\displaystyle=f(W\_{t-1})-\eta\|N\_{t}\|\_{\*}+\eta\langle N\_{t}-\nabla f(W\_{t-1}),P\_{t}\rangle\_{F}+\frac{L\eta^{2}}{2} |  |

where the last equality is due to ⟨Nt,Pt⟩F=‖Nt‖∗\langle N\_{t},P\_{t}\rangle\_{F}=\|N\_{t}\|\_{\*} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(ii)).
By Hölder’s inequality (Lemma [4](#Thmlemma4 "Lemma 4 (Hölder’s inequality). ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")) and the triangle inequality, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)−η​‖Nt‖∗+η​‖Nt−∇f​(Wt−1)‖∗​‖Pt‖op+L​η22\displaystyle\leq f(W\_{t-1})-\eta\|N\_{t}\|\_{\*}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}\|P\_{t}\|\_{\mathrm{op}}+\frac{L\eta^{2}}{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η​‖∇f​(Wt−1)‖∗+η​‖∇f​(Wt−1)−Nt‖∗+η​‖Nt−∇f​(Wt−1)‖∗+L​η22\displaystyle\leq f(W\_{t-1})-\eta\|\nabla f(W\_{t-1})\|\_{\*}+\eta\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}+\eta\|N\_{t}-\nabla f(W\_{t-1})\|\_{\*}+\frac{L\eta^{2}}{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η​‖∇f​(Wt−1)‖∗+2​η​‖∇f​(Wt−1)−Nt‖∗+L​η22.\displaystyle=f(W\_{t-1})-\eta\|\nabla f(W\_{t-1})\|\_{\*}+2\eta\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}+\frac{L\eta^{2}}{2}. |  |

Rearranging and taking expectation gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)‖∗]≤𝔼​[f​(Wt−1)−f​(Wt)]η+2​𝔼​[‖∇f​(Wt−1)−Nt‖∗]+L​η2\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}]\leq\frac{\mathbb{E}[f(W\_{t-1})-f(W\_{t})]}{\eta}+2\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}]+\frac{L\eta}{2} |  | (9) |

Bounding 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}].

In order to bound 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}],
we introduce the true scaled momentum N¯t\bar{N}\_{t} defined by the true (full-batch) gradient ∇f​(Wt)\nabla f(W\_{t}) instead of GtG\_{t} for each step tt:

* •

  N¯t=β​N¯t−1+(1−β)​∇f​(Wt−1)\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t-1}) for t>0t>0
  and N¯0=0\bar{N}\_{0}=0.
* •

  Note that Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t} for t>0t>0 and N0=0N\_{0}=0.

Then we can decompose 𝔼​[‖∇f​(Wt−1)−Nt‖∗]\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}] as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)−Nt‖∗]≤𝔼​[‖∇f​(Wt−1)−N¯t‖∗]⏟Term (A)+𝔼​[‖N¯t−Nt‖∗]⏟Term (B)\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}]\leq\underbrace{\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{\*}]}\_{\text{Term (A)}}+\underbrace{\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{\*}]}\_{\text{Term (B)}} |  | (10) |

Bounding the Term (A).
Let Dt=‖∇f​(Wt−1)−N¯t‖∗D\_{t}=\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{\*}.
From the recursion,

|  |  |  |
| --- | --- | --- |
|  | ∇f​(Wt−1)−N¯t=β​(∇f​(Wt−1)−N¯t−1)\displaystyle\nabla f(W\_{t-1})-\bar{N}\_{t}=\beta(\nabla f(W\_{t-1})-\bar{N}\_{t-1}) |  |

Hence, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dt\displaystyle D\_{t} | =β​‖∇f​(Wt−1)−N¯t−1‖∗\displaystyle=\beta\|\nabla f(W\_{t-1})-\bar{N}\_{t-1}\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤β​‖∇f​(Wt−2)−N¯t−1‖+β​‖∇f​(Wt−1)−∇f​(Wt−2)‖∗\displaystyle\leq\beta\|\nabla f(W\_{t-2})-\bar{N}\_{t-1}\|+\beta\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​Dt−1+β​‖∇f​(Wt−1)−∇f​(Wt−2)‖∗.\displaystyle=\beta D\_{t-1}+\beta\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*}. |  |

Applying Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and using the fact that
‖Ot−1‖op=‖Pt−1‖op≤1\|O\_{t-1}\|\_{\mathrm{op}}=\|P\_{t-1}\|\_{\mathrm{op}}\leq 1, we have

|  |  |  |
| --- | --- | --- |
|  | ‖∇f​(Wt−1)−∇f​(Wt−2)‖∗≤L​‖Wt−1−Wt−2‖op=L​η​‖Ot−1‖op≤L​η\displaystyle\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*}\leq L\|W\_{t-1}-W\_{t-2}\|\_{\mathrm{op}}=L\eta\|O\_{t-1}\|\_{\mathrm{op}}\leq L\eta |  |

Hence, we have the following recursion,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Dt\displaystyle D\_{t} | ≤β​Dt−1+β​L​η\displaystyle\leq\beta D\_{t-1}+\beta L\eta |  |

Since N¯0=0\bar{N}\_{0}=0,
we have

|  |  |  |
| --- | --- | --- |
|  | D1=‖∇f​(W0)−N¯1‖∗=‖∇f​(W0)−(β​N¯0+(1−β)​∇f​(W0))‖∗=β​‖∇f​(W0)‖∗\displaystyle D\_{1}=\|\nabla f(W\_{0})-\bar{N}\_{1}\|\_{\*}=\|\nabla f(W\_{0})-(\beta\bar{N}\_{0}+(1-\beta)\nabla f(W\_{0}))\|\_{\*}=\beta\|\nabla f(W\_{0})\|\_{\*} |  |

By unrolling the recursion and taking the expectation, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[Dt]≤βt​𝔼​[‖∇f​(W0)‖∗]+∑i=1tβi​L​η≤βt​‖∇f​(W0)‖∗+β​L​η1−β\displaystyle\mathbb{E}[D\_{t}]\leq\beta^{t}\mathbb{E}[\|\nabla f(W\_{0})\|\_{\*}]+\sum\_{i=1}^{t}\beta^{i}L\eta\leq\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\frac{\beta L\eta}{1-\beta} |  | (11) |

Bounding the Term (B).
When we unroll the EMA recursion of both NtN\_{t} and N¯t\bar{N}\_{t},

|  |  |  |
| --- | --- | --- |
|  | Nt=β​Nt−1+(1−β)​Gt=βt​N0+(1−β)​∑i=1tβt−i​Gi\displaystyle N\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t}=\beta^{t}N\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}G\_{i} |  |
|  |  |  |
| --- | --- | --- |
|  | N¯t=β​N¯t−1+(1−β)​∇f​(Wt)=βt​N¯0+(1−β)​∑i=1tβt−i​∇f​(Wi)\displaystyle\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t})=\beta^{t}\bar{N}\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}\nabla f(W\_{i}) |  |

Then, we compute the expectation of the Frobenius norm bias caused by MtM\_{t} and M¯t\bar{M}\_{t},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F]≤(1−β)​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}]\leq(1-\beta)\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}\right] |  | (12) |

Applying Jensen’s inequality and the linearity of expectation gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1−β)​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F]\displaystyle(1-\beta)\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}\right] | ≤(1−β)2​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F2]\displaystyle\leq\sqrt{(1-\beta)^{2}\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−β)​𝔼[∑i=1tβ2​(t−i)∥Gi−∇f(Wi))∥F2]\displaystyle=(1-\beta)\sqrt{\mathbb{E}\left[\sum\_{i=1}^{t}\beta^{2(t-i)}\left\|G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−β)​∑i=1tβ2​(t−i)​𝔼​[‖Gi−∇f​(Wi)‖F2]\displaystyle=(1-\beta)\sqrt{\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i})\|\_{F}^{2}]} |  |

By Lemma [8](#Thmlemma8 "Lemma 8 (Unbiasedness and bounded variance). ‣ A.2.2 Assumption 2 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz"), Eq.([12](#A3.E12 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) becomes

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}] | ≤(1−β)​∑i=1tβ2​(t−i)​𝔼​[‖Gi−∇f​(Wi)‖F2]\displaystyle\leq(1-\beta)\sqrt{\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i})\|\_{F}^{2}]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤(1−β)​1−β2​t1−β2​σ2B≤1−β1+β​σB\displaystyle\leq(1-\beta)\sqrt{\frac{1-\beta^{2t}}{1-\beta^{2}}\frac{\sigma^{2}}{B}}\leq\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sigma}{\sqrt{B}} |  |

Using the fact that ‖X‖∗≤r​‖X‖F\|X\|\_{\*}\leq\sqrt{r}\|X\|\_{F} for X∈ℝm×nX\in\mathbb{R}^{m\times n} with r=min⁡{m,n}r=\min\{m,n\} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iii)), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖N¯t−Nt‖∗]≤r​𝔼​[‖N¯t−Nt‖F]≤1−β1+β​r​σB\displaystyle\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{\*}]\leq\sqrt{r}\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{F}]\leq\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}} |  | (13) |

Plugging Eq.([11](#A3.E11 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) and Eq.([13](#A3.E13 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) into Eq.([10](#A3.E10 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)−Nt‖∗]≤β​L​η1−β+βt​‖∇f​(W0)‖∗+1−β1+β​r​σB\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{\*}]\leq\frac{\beta L\eta}{1-\beta}+\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}} |  | (14) |

Averaging and tuning.
Plugging Eq.([14](#A3.E14 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) into Eq.([9](#A3.E9 "In Proof of Theorem 4. ‣ C.1 Theorem 4 (Muon with SVD) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")),
we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤𝔼​[f​(Wt−1)−f​(Wt)]η+2​(β​L​η1−β+βt​‖∇f​(W0)‖∗+1−β1+β​r​σB)+L​η2\displaystyle\leq\frac{\mathbb{E}[f(W\_{t-1})-f(W\_{t})]}{\eta}+2\left(\frac{\beta L\eta}{1-\beta}+\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right)+\frac{L\eta}{2} |  |

Averaging over t=1,…​Tt=1,\ldots T, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤DT​η+2​(β​L​η1−β+1−β1+β​r​σB)+2T​∑t=1Tβt​‖∇f​(W0)‖∗+L​η2\displaystyle\leq\frac{D}{T\eta}+2\left(\frac{\beta L\eta}{1-\beta}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right)+\frac{2}{T}\sum\_{t=1}^{T}\beta^{t}\|\nabla f(W\_{0})\|\_{\*}+\frac{L\eta}{2} |  |

Using Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iii) and Lemma [7](#Thmlemma7 "Lemma 7 (Gradient-gap inequality). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz"),
‖∇f​(W0)‖∗≤r​‖∇f​(W0)‖F≤2​r​L​D\|\nabla f(W\_{0})\|\_{\*}\leq\sqrt{r}\,\|\nabla f(W\_{0})\|\_{F}\leq\sqrt{2rLD},
where D=f​(W0)−f∗D=f(W\_{0})-f^{\*}.
Then, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤DT​η+2​(β​L​η1−β+1−β1+β​r​σB)+2​β​2​r​L​DT​(1−β)+L​η2\displaystyle\leq\frac{D}{T\eta}+2\left(\frac{\beta L\eta}{1-\beta}+\sqrt{\frac{1-\beta}{1+\beta}}\frac{\sqrt{r}\sigma}{\sqrt{B}}\right)+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)}+\frac{L\eta}{2} |  |

where D=f​(W0)−f∗D=f(W\_{0})-f^{\*}.
By setting η=(1−β)​DT​L\eta=\sqrt{\frac{(1-\beta)D}{TL}},
we obtain

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]≤DT​η+L​η2+2​β​L​η1−β+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β)\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}]\leq\frac{D}{T\eta}+\frac{L\eta}{2}+\frac{2\beta L\eta}{1-\beta}+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)} |  |
|  |  |  |
| --- | --- | --- |
|  | =L​DT​(1+2​β1−β+1−β2)+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β)\displaystyle\qquad=\sqrt{\frac{LD}{T}}\left(\frac{1+2\beta}{\sqrt{1-\beta}}+\frac{\sqrt{1-\beta}}{2}\right)+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)} |  |

Setting β=1−min⁡{L​D​Bσ​r​T,1}\beta=1-\min\left\{\frac{\sqrt{LDB}}{\sigma\sqrt{rT}},1\right\}, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖∗]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{\*}] | ≤1−β2​L​DT+1+2​β1−β​L​DT+2​σ​rB​1−β1+β+2​β​2​r​L​DT​(1−β)\displaystyle\leq\frac{\sqrt{1-\beta}}{2}\sqrt{\frac{LD}{T}}+\frac{1+2\beta}{\sqrt{1-\beta}}\sqrt{\frac{LD}{T}}+\frac{2\sigma\sqrt{r}}{\sqrt{B}}\sqrt{\frac{1-\beta}{1+\beta}}+\frac{2\beta\sqrt{2rLD}}{T(1-\beta)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−β2​L​DT+(1+2​β+21+β)​(r​σ2​L​DB​T)1/4+2​2​β​σ​rB​T\displaystyle=\frac{\sqrt{1-\beta}}{2}\sqrt{\frac{LD}{T}}+\left(1+2\beta+\frac{2}{\sqrt{1+\beta}}\right)\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}+2\sqrt{2}\beta\frac{\sigma r}{\sqrt{BT}} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝒪​(L​DT+σ​rB​T+(r​σ2​L​DB​T)1/4)\displaystyle=\mathcal{O}\left(\sqrt{\frac{LD}{T}}+\frac{\sigma r}{\sqrt{BT}}+\left(\frac{r\sigma^{2}LD}{BT}\right)^{1/4}\right) |  | (15) |

Thus, we can find an ϵ\epsilon-nuclear norm stationary point of ff with a complexity of

|  |  |  |
| --- | --- | --- |
|  | 𝒪​(max⁡{L​Dϵ2,r2​σ2B​ϵ2,r​σ2​L​DB​ϵ4})\displaystyle\mathcal{O}\left(\max\left\{\frac{LD}{\epsilon^{2}},\frac{r^{2}\sigma^{2}}{B\epsilon^{2}},\frac{r\sigma^{2}LD}{B\epsilon^{4}}\right\}\right) |  |

∎

### C.2 Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") (SGD with Momentum)

Algorithm 3  SGD with momentum (SGD-M)

0: learning rate η>0\eta>0, momentum β∈[0,1)\beta\in[0,1), batch size BB

1: Initialize: M0←0M\_{0}\leftarrow 0, W0∈ℝm×nW\_{0}\in\mathbb{R}^{m\times n}

2: for t=1t=1 to TT do

3:  Gt←1B​∑i=1B∇f​(Wt;ξt,i)G\_{t}\leftarrow\tfrac{1}{B}\sum\_{i=1}^{B}\nabla f(W\_{t};\xi\_{t,i})

4:  Mt←β​Mt−1+GtM\_{t}\leftarrow\beta M\_{t-1}+G\_{t}

5:  Wt←Wt−1−η​MtW\_{t}\leftarrow W\_{t-1}-\eta M\_{t}

6: end for

7: return WTW\_{T}

###### Proof of Theorem [3](#Thmtheorem3 "Theorem 3 (Convergence of SGD with momentum). ‣ 4.5 Comparisons with SGD with Momentum and Muon with SVD ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

First, we introduce the scaled EMA momentum: Nt:=(1−β)​MtN\_{t}:=(1-\beta)M\_{t}.
Then, we get Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t} with N0=0N\_{0}=0 and the scaled learning rate η~:=η1−β\tilde{\eta}:=\frac{\eta}{1-\beta}, yielding
Wt=Wt−1−η~​NtW\_{t}=W\_{t-1}-\tilde{\eta}N\_{t}.

By Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"),
we start from the descent lemma (Lemma [6](#Thmlemma6 "Lemma 6 (Descent Lemma). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")).
Since Wt=Wt−1−η~​NtW\_{t}=W\_{t-1}-\tilde{\eta}N\_{t}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)+⟨∇f​(Wt−1),Wt−Wt−1⟩F+L2​‖Wt−Wt−1‖op2\displaystyle\leq f(W\_{t-1})+\langle\nabla f(W\_{t-1}),W\_{t}-W\_{t-1}\rangle\_{F}+\frac{L}{2}\|W\_{t}-W\_{t-1}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η~​⟨∇f​(Wt−1),Nt⟩F+L2​η~2​‖Nt‖op2\displaystyle=f(W\_{t-1})-\tilde{\eta}\langle\nabla f(W\_{t-1}),N\_{t}\rangle\_{F}+\frac{L}{2}\tilde{\eta}^{2}\|N\_{t}\|\_{\mathrm{op}}^{2} |  |

By using ⟨a,b⟩F=12​(‖a‖F2+‖b‖F2−‖a−b‖F2)\langle a,b\rangle\_{F}=\frac{1}{2}\left(\|a\|\_{F}^{2}+\|b\|\_{F}^{2}-\|a-b\|\_{F}^{2}\right), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(Wt)\displaystyle f(W\_{t}) | ≤f​(Wt−1)−η~2​‖∇f​(Wt−1)‖F2−η~2​‖Nt‖F2+η~2​‖Nt−∇f​(Wt−1)‖F2+L2​η~2​‖Nt‖op2\displaystyle\leq f(W\_{t-1})-\frac{\tilde{\eta}}{2}\|\nabla f(W\_{t-1})\|\_{F}^{2}-\frac{\tilde{\eta}}{2}\|N\_{t}\|\_{F}^{2}+\frac{\tilde{\eta}}{2}\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}+\frac{L}{2}\tilde{\eta}^{2}\|N\_{t}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η~2​‖∇f​(Wt−1)‖F2−η~2​‖Nt‖F2+η~2​‖Nt−∇f​(Wt−1)‖F2+L2​η~2​‖Nt‖F2\displaystyle\leq f(W\_{t-1})-\frac{\tilde{\eta}}{2}\|\nabla f(W\_{t-1})\|\_{F}^{2}-\frac{\tilde{\eta}}{2}\|N\_{t}\|\_{F}^{2}+\frac{\tilde{\eta}}{2}\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}+\frac{L}{2}\tilde{\eta}^{2}\|N\_{t}\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(Wt−1)−η~2​‖∇f​(Wt−1)‖F2+η~2​‖Nt−∇f​(Wt−1)‖F2−η~​(1−L​η~)2​‖Nt‖F2\displaystyle=f(W\_{t-1})-\frac{\tilde{\eta}}{2}\|\nabla f(W\_{t-1})\|\_{F}^{2}+\frac{\tilde{\eta}}{2}\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}-\frac{\tilde{\eta}(1-L\tilde{\eta})}{2}\|N\_{t}\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤f​(Wt−1)−η~2​‖∇f​(Wt−1)‖F2+η~2​‖Nt−∇f​(Wt−1)‖F2\displaystyle\leq f(W\_{t-1})-\frac{\tilde{\eta}}{2}\|\nabla f(W\_{t-1})\|\_{F}^{2}+\frac{\tilde{\eta}}{2}\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2} |  |

where the second inequality is due to ‖Nt‖op≤‖Nt‖F\|N\_{t}\|\_{\mathrm{op}}\leq\|N\_{t}\|\_{F} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iv)).
The last inequality holds by choosing η~≤1/L\tilde{\eta}\leq 1/L so that we can drop the last term.

Rearranging and taking expectation gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖∇f​(Wt−1)‖F2]\displaystyle\mathbb{E}\left[\|\nabla f(W\_{t-1})\|\_{F}^{2}\right] | ≤2​𝔼​[f​(Wt−1)−f​(Wt)]η~+𝔼​[‖Nt−∇f​(Wt−1)‖F2]\displaystyle\leq\frac{2\mathbb{E}[f(W\_{t-1})-f(W\_{t})]}{\tilde{\eta}}+\mathbb{E}[\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}] |  |

Averaging over t=1t=1 to TT, we obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖F2]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\left[\|\nabla f(W\_{t-1})\|\_{F}^{2}\right] | ≤2​DT​η~+1T​∑t=1T𝔼​[‖Nt−∇f​(Wt−1)‖F2]\displaystyle\leq\frac{2D}{T\tilde{\eta}}+\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}] |  | (16) |

where D=f​(W0)−f∗D=f(W\_{0})-f^{\*}.
Denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | SA:=1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖F2]≤2​DT​η~+SB,SB:=1T​∑t=1T𝔼​[‖Nt−∇f​(Wt−1)‖F2]\displaystyle S\_{A}:=\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{F}^{2}]\leq\frac{2D}{T\tilde{\eta}}+S\_{B},\qquad S\_{B}:=\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}] |  | (17) |

Bounding 1T​∑t=1T𝔼​[‖Nt−∇f​(Wt−1)‖F2]\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}].

We introduce the true scaled momentum N¯t\bar{N}\_{t} defined by the true (full-batch) gradient ∇f​(Wt)\nabla f(W\_{t}) instead of GtG\_{t} for each step tt:

* •

  N¯t=β​N¯t−1+(1−β)​∇f​(Wt)\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t}) for t>0t>0 and N¯0=0\bar{N}\_{0}=0.
* •

  Note that Nt=β​Nt−1+(1−β)​GtN\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t} for t>0t>0 and N0=0N\_{0}=0.

Then, we can decompose 1T​∑t=1T𝔼​[‖∇f​(Wt−1)−Nt‖F2]\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{F}^{2}] using ‖a+b‖F2≤2​‖a‖F2+2​‖b‖F2\|a+b\|\_{F}^{2}\leq 2\|a\|\_{F}^{2}+2\|b\|\_{F}^{2},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)−Nt‖F2]≤2​1T​∑t=0T−1𝔼​[‖∇f​(Wt−1)−N¯t‖F2]⏟Term (A)+2​1T​∑t=1T𝔼​[‖N¯t−Nt‖F2]⏟Term (B)\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{F}^{2}]\leq 2\underbrace{\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{F}^{2}]}\_{\text{Term (A)}}+2\underbrace{\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\bar{N}\_{t}-N\_{t}\|\_{F}^{2}]}\_{\text{Term (B)}} |  | (18) |

Bounding the Term (A).
Let et:=∇f​(Wt−1)−N¯te\_{t}:=\nabla f(W\_{t-1})-\bar{N}\_{t}.
Using the recursion for N¯t\bar{N}\_{t}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | et\displaystyle e\_{t} | =∇f(Wt−1)−(βN¯t−1+(1−β)∇f(Wt)))\displaystyle=\nabla f(W\_{t-1})-\left(\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t}))\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​(∇f​(Wt−1)−N¯t−1)+(1−β)​(∇f​(Wt−1)−∇f​(Wt))\displaystyle=\beta\left(\nabla f(W\_{t-1})-\bar{N}\_{t-1}\right)+(1-\beta)\left(\nabla f(W\_{t-1})-\nabla f(W\_{t})\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​et−1+β​(∇f​(Wt−1)−∇f​(Wt−2))+(1−β)​(∇f​(Wt−1)−∇f​(Wt))\displaystyle=\beta e\_{t-1}+\beta\left(\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\right)+(1-\beta)\left(\nabla f(W\_{t-1})-\nabla f(W\_{t})\right) |  |

Apply the variation of Young’s inequality (Lemma [5](#Thmlemma5 "Lemma 5. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")) with c=1−ββc=\frac{1-\beta}{\beta}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖et‖F2\displaystyle\|e\_{t}\|\_{F}^{2} | =‖β​et−1+β​(∇f​(Wt−1)−∇f​(Wt−2))+(1−β)​(∇f​(Wt−1)−∇f​(Wt))‖F2\displaystyle=\|\beta e\_{t-1}+\beta\left(\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\right)+(1-\beta)\left(\nabla f(W\_{t-1})-\nabla f(W\_{t})\right)\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1+c)​β2​‖et−1‖F2+(1+1/c)​‖β​(∇f​(Wt−1)−∇f​(Wt−2))+(1−β)​(∇f​(Wt−1)−∇f​(Wt))‖F2\displaystyle=(1+c)\beta^{2}\|e\_{t-1}\|\_{F}^{2}+(1+1/c)\|\beta\left(\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\right)+(1-\beta)\left(\nabla f(W\_{t-1})-\nabla f(W\_{t})\right)\|\_{F}^{2} |  |

Since 1+c=1β1+c=\frac{1}{\beta} and 1+1/c=1/(1−β)1+1/c=1/(1-\beta), and
‖x+y‖2≤2​‖x‖2+2​‖y‖2\|x+y\|^{2}\leq 2\|x\|^{2}+2\|y\|^{2}, we get

|  |  |  |
| --- | --- | --- |
|  | ‖et‖F2≤β​‖et−1‖F2+2​β21−β​‖∇f​(Wt−1)−∇f​(Wt−2)‖F2+2​(1−β)​‖∇f​(Wt−1)−∇f​(Wt)‖F2.\displaystyle\|e\_{t}\|\_{F}^{2}\leq\beta\|e\_{t-1}\|\_{F}^{2}+\frac{2\beta^{2}}{1-\beta}\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{F}^{2}+2(1-\beta)\|\nabla f(W\_{t-1})-\nabla f(W\_{t})\|\_{F}^{2}. |  |

By Assumption [1](#Thmassumption1 "Assumption 1 (Lipschitz smoothness). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") and the norm monotonicity ∥⋅∥op≤∥⋅∥F≤∥⋅∥∗\|\cdot\|\_{\mathrm{op}}\leq\|\cdot\|\_{F}\leq\|\cdot\|\_{\*} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iv)),
we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∇f​(Wt)−∇f​(Wt−1)‖F2\displaystyle\|\nabla f(W\_{t})-\nabla f(W\_{t-1})\|\_{F}^{2} | ≤‖∇f​(Wt)−∇f​(Wt−1)‖∗2≤L2​‖Wt−Wt−1‖op2\displaystyle\leq\|\nabla f(W\_{t})-\nabla f(W\_{t-1})\|\_{\*}^{2}\leq L^{2}\|W\_{t}-W\_{t-1}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =L2​η~2​‖Nt‖op2≤L2​η~2​‖Nt‖F2\displaystyle=L^{2}\tilde{\eta}^{2}\|N\_{t}\|\_{\mathrm{op}}^{2}\leq L^{2}\tilde{\eta}^{2}\|N\_{t}\|\_{F}^{2} |  |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∇f​(Wt−1)−∇f​(Wt−2)‖F2\displaystyle\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{F}^{2} | ≤‖∇f​(Wt−1)−∇f​(Wt−2)‖∗2≤L2​‖Wt−1−Wt−2‖op2\displaystyle\leq\|\nabla f(W\_{t-1})-\nabla f(W\_{t-2})\|\_{\*}^{2}\leq L^{2}\|W\_{t-1}-W\_{t-2}\|\_{\mathrm{op}}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =L2​η~2​‖Nt−1‖op2≤L2​η~2​‖Nt−1‖F2\displaystyle=L^{2}\tilde{\eta}^{2}\|N\_{t-1}\|\_{\mathrm{op}}^{2}\leq L^{2}\tilde{\eta}^{2}\|N\_{t-1}\|\_{F}^{2} |  |

Hence, we have the following recursion,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖et‖F2\displaystyle\|e\_{t}\|\_{F}^{2} | ≤β​‖et−1‖F2+2​β2​L2​η~21−β​‖Nt−1‖F2+2​(1−β)​L2​η~​‖Nt‖F2\displaystyle\leq\beta\|e\_{t-1}\|\_{F}^{2}+\frac{2\beta^{2}L^{2}\tilde{\eta}^{2}}{1-\beta}\|N\_{t-1}\|\_{F}^{2}+2(1-\beta)L^{2}\tilde{\eta}\|N\_{t}\|\_{F}^{2} |  |

Taking expectation and averaging over t=1t=1 to TT, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)−N¯t‖F2]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{F}^{2}] | ≤β​1T​∑t=1T‖∇f​(Wt−2)−N¯t−1‖F2\displaystyle\leq\beta\frac{1}{T}\sum\_{t=1}^{T}\|\nabla f(W\_{t-2})-\bar{N}\_{t-1}\|\_{F}^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +2​β2​L2​η~21−β​1T​∑t=1T𝔼​[‖Nt−1‖F2]+2​(1−β)​L2​η~2​1T​∑t=1T𝔼​[‖Nt‖F2]\displaystyle\qquad+\frac{2\beta^{2}L^{2}\tilde{\eta}^{2}}{1-\beta}\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t-1}\|\_{F}^{2}]+2(1-\beta)L^{2}\tilde{\eta}^{2}\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}\|\_{F}^{2}] |  |

Since 1T​∑t=1T𝔼​[‖∇f​(Wt−2)−N¯t−1‖F2]≤1T​𝔼​‖e0‖F2+1T​∑t=1T𝔼​[‖∇f​(Wt−1)−N¯t‖F2]\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-2})-\bar{N}\_{t-1}\|\_{F}^{2}]\leq\frac{1}{T}\mathbb{E}\|e\_{0}\|\_{F}^{2}+\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{F}^{2}]
and
1T​∑t=1T𝔼​‖Nt−1‖F2≤1T​∑t=1T𝔼​‖Nt‖F2\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|N\_{t-1}\|\_{F}^{2}\leq\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|N\_{t}\|\_{F}^{2},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖∇f​(Wt−1)−N¯t‖F2]≤β​𝔼​‖e0‖F2T​(1−β)+2​L2​η~21−β​(β21−β+1−β)​(1T​∑t=1T𝔼​‖Nt‖F2).\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t-1})-\bar{N}\_{t}\|\_{F}^{2}]\leq\frac{\beta\mathbb{E}\|e\_{0}\|\_{F}^{2}}{T(1-\beta)}+\frac{2L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)\left(\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|N\_{t}\|\_{F}^{2}\right). |  | (19) |

Bounding the Term (B).
When we unroll the EMA recursion of both NtN\_{t} and N¯t\bar{N}\_{t},

|  |  |  |
| --- | --- | --- |
|  | Nt=β​Nt−1+(1−β)​Gt=βt​N0+(1−β)​∑i=1tβt−i​Gi\displaystyle N\_{t}=\beta N\_{t-1}+(1-\beta)G\_{t}=\beta^{t}N\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}G\_{i} |  |
|  |  |  |
| --- | --- | --- |
|  | N¯t=β​N¯t−1+(1−β)​∇f​(Wt)=βt​N¯0+(1−β)​∑i=1tβt−i​∇f​(Wi)\displaystyle\bar{N}\_{t}=\beta\bar{N}\_{t-1}+(1-\beta)\nabla f(W\_{t})=\beta^{t}\bar{N}\_{0}+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}\nabla f(W\_{i}) |  |

Then, we compute the expectation of the Frobenius squared norm bias caused by NtN\_{t} and N¯t\bar{N}\_{t},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F2]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}^{2}] | ≤𝔼​[‖βt​(N0−N¯0)+(1−β)​∑i=1tβt−i​(Gi−∇f​(Wi))‖F2]\displaystyle\leq\mathbb{E}\left[\left\|\beta^{t}(N\_{0}-\bar{N}\_{0})+(1-\beta)\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β2​t​𝔼​[‖N0−N¯0‖F2]+(1−β)2​𝔼​[‖∑i=1tβt−i​(Gi−∇f​(Wi))‖F2]\displaystyle=\beta^{2t}\mathbb{E}[\|N\_{0}-\bar{N}\_{0}\|\_{F}^{2}]+(1-\beta)^{2}\mathbb{E}\left[\left\|\sum\_{i=1}^{t}\beta^{t-i}(G\_{i}-\nabla f(W\_{i}))\right\|\_{F}^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β2​t𝔼[∥N0−N¯0∥F2]+(1−β)2∑i=1tβ2​(t−i)𝔼[∥Gi−∇f(Wi))∥F2]\displaystyle=\beta^{2t}\mathbb{E}[\|N\_{0}-\bar{N}\_{0}\|\_{F}^{2}]+(1-\beta)^{2}\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i}))\|\_{F}^{2}] |  |

where equalities are due to Assumption [2](#Thmassumption2 "Assumption 2 (Bounded variance). ‣ 3.2 Problem Setting: Nonconvex Optimization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz"), which states 𝔼​[Gt]=∇f​(Wt)\mathbb{E}[G\_{t}]=\nabla f(W\_{t}) so that the cross-term vanishes.
Note that 𝔼​[‖N0−N¯0‖F2]=0\mathbb{E}[\|N\_{0}-\bar{N}\_{0}\|\_{F}^{2}]=0.
By Lemma [8](#Thmlemma8 "Lemma 8 (Unbiasedness and bounded variance). ‣ A.2.2 Assumption 2 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz"), we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖Nt−N¯t‖F2]\displaystyle\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}^{2}] | ≤(1−β)2∑i=1tβ2​(t−i)𝔼[∥Gi−∇f(Wi))∥F2]\displaystyle\leq(1-\beta)^{2}\sum\_{i=1}^{t}\beta^{2(t-i)}\mathbb{E}[\|G\_{i}-\nabla f(W\_{i}))\|\_{F}^{2}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤(1−β)2​∑i=1tβ2​(t−i)​σ2B≤1−β1+β​σ2B\displaystyle\leq(1-\beta)^{2}\sum\_{i=1}^{t}\beta^{2(t-i)}\frac{\sigma^{2}}{B}\leq\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B} |  |

By averaging over t=1,…,Tt=1,\ldots,T, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖Nt−N¯t‖F2]≤1T​∑t=1T(1−β1+β​σ2B)≤1−β1+β​σ2B\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}-\bar{N}\_{t}\|\_{F}^{2}]\leq\frac{1}{T}\sum\_{t=1}^{T}\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)\leq\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B} |  | (20) |

Plugging Eq.([19](#A3.E19 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) and Eq.([20](#A3.E20 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) to Eq.([18](#A3.E18 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")), we obtain

|  |  |  |
| --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)−Nt‖F2]\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})-N\_{t}\|\_{F}^{2}] |  |
|  |  |  |
| --- | --- | --- |
|  | ≤2(1−β1+βσ2B)+2(β​𝔼​‖e0‖F2T​(1−β)+2​L2​η~21−β(β21−β+1−β)(1T∑t=1T𝔼∥Nt∥F2).)\displaystyle\leq 2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)+2\left(\frac{\beta\mathbb{E}\|e\_{0}\|\_{F}^{2}}{T(1-\beta)}+\frac{2L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)\left(\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}\|N\_{t}\|\_{F}^{2}\right).\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤2​(1−β1+β​σ2B)+2​β​𝔼​‖e0‖F2T​(1−β)\displaystyle\leq 2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)+\frac{2\beta\mathbb{E}\|e\_{0}\|\_{F}^{2}}{T(1-\beta)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | +8​L2​η~21−β​(β21−β+1−β)​(1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖F2]+1T​∑t=1T𝔼​[‖Nt−∇f​(Wt−1)‖F2])\displaystyle\quad+\frac{8L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)\left(\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{F}^{2}]+\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|N\_{t}-\nabla f(W\_{t-1})\|\_{F}^{2}]\right) |  | (21) |

where the last inequality is due to ‖a+b‖F2≤2​‖a‖F2+2​‖b‖F2\|a+b\|\_{F}^{2}\leq 2\|a\|\_{F}^{2}+2\|b\|\_{F}^{2}.
Eq.([21](#A3.E21 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) can be expressed as

|  |  |  |  |
| --- | --- | --- | --- |
|  | SB\displaystyle S\_{B} | ≤2​(1−β1+β​σ2B)+2​β​𝔼​‖e0‖F2T​(1−β)+8​L2​η~21−β​(β21−β+1−β)​(SA+SB)\displaystyle\leq 2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)+\frac{2\beta\mathbb{E}\|e\_{0}\|\_{F}^{2}}{T(1-\beta)}+\frac{8L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)\left(S\_{A}+S\_{B}\right) |  |

By Lemma [7](#Thmlemma7 "Lemma 7 (Gradient-gap inequality). ‣ A.2.1 Assumption 1 ‣ A.2 Lemmas under Assumptions ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz") and Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iv),
we have 𝔼​‖e0‖F2=𝔼​‖∇f​(W0)‖F2≤𝔼​‖∇f​(W0)‖∗2≤2​L​D\mathbb{E}\|e\_{0}\|\_{F}^{2}=\mathbb{E}\|\nabla f(W\_{0})\|\_{F}^{2}\leq\mathbb{E}\|\nabla f(W\_{0})\|\_{\*}^{2}\leq 2LD.
Therefore,

|  |  |  |  |
| --- | --- | --- | --- |
|  | SB\displaystyle S\_{B} | ≤2​(1−β1+β​σ2B)+4​β​L​DT​(1−β)+8​L2​η~21−β​(β21−β+1−β)​(SA+SB)\displaystyle\leq 2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)+\frac{4\beta LD}{T(1-\beta)}+\frac{8L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)\left(S\_{A}+S\_{B}\right) |  |

Let θ:=4​L2​η~21−β​(β21−β+1−β)=4​L2​η~2(1−β)2​K\theta:=\frac{4L^{2}\tilde{\eta}^{2}}{1-\beta}\left(\frac{\beta^{2}}{1-\beta}+1-\beta\right)=\frac{4L^{2}\tilde{\eta}^{2}}{(1-\beta)^{2}}K and K:=β2+(1−β)2K:=\beta^{2}+(1-\beta)^{2}.
Then, we have

|  |  |  |
| --- | --- | --- |
|  | SB≤2​θ​(SA+SB)+4​β​L​D(1−β)​T+2​(1−β1+β​σ2B)\displaystyle S\_{B}\leq 2\theta(S\_{A}+S\_{B})+\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right) |  |

Hence

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1−2​θ)​SB≤2​θ​SA+4​β​L​D(1−β)​T+2​(1−β1+β​σ2B).\displaystyle(1-2\theta)S\_{B}\leq 2\theta S\_{A}+\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right). |  | (22) |

Insert Eq.([22](#A3.E22 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")) into Eq.([17](#A3.E17 "In Proof of Theorem 3. ‣ C.2 Theorem 3 (SGD with Momentum) ‣ Appendix C Muon with SVD and SGD with momentum ‣ Convergence of Muon with Newton–Schulz")), we obtain

|  |  |  |
| --- | --- | --- |
|  | SA≤2​DT​η~+SB≤2​DT​η~+2​θ1−2​θ​SA+11−2​θ​(4​β​L​D(1−β)​T+2​(1−β1+β​σ2B))\displaystyle S\_{A}\leq\frac{2D}{T\tilde{\eta}}+S\_{B}\leq\frac{2D}{T\tilde{\eta}}+\frac{2\theta}{1-2\theta}S\_{A}+\frac{1}{1-2\theta}\left(\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)\right) |  |

Therefore, provided θ<14\theta<\tfrac{1}{4},

|  |  |  |  |
| --- | --- | --- | --- |
|  | SA≤2​(1−2​θ)1−4​θ⋅Dη~​T+11−4​θ​(4​β​L​D(1−β)​T+2​(1−β1+β​σ2B)).\displaystyle S\_{A}\leq\frac{2(1-2\theta)}{1-4\theta}\cdot\frac{D}{\tilde{\eta}T}+\frac{1}{1-4\theta}\left(\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)\right). |  | (23) |

Finally, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=1T𝔼​[‖∇f​(Wt−1)‖F2]≤2​(1−2​θ)1−4​θ⋅(1−β)​Dη​T+11−4​θ​(4​β​L​D(1−β)​T+2​(1−β1+β​σ2B)).\displaystyle\frac{1}{T}\sum\_{t=1}^{T}\mathbb{E}[\|\nabla f(W\_{t-1})\|\_{F}^{2}]\leq\frac{2(1-2\theta)}{1-4\theta}\cdot\frac{(1-\beta)D}{\eta T}+\frac{1}{1-4\theta}\left(\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)\right). |  | (24) |

Tuning η\eta and β\beta.
We need η~≤1/L\tilde{\eta}\leq 1/L and θ<14\theta<\frac{1}{4}.
Since K=β2+(1−β)2∈[1/2,1]K=\beta^{2}+(1-\beta)^{2}\in[1/2,1],
a convenient sufficient choice is

|  |  |  |
| --- | --- | --- |
|  | η≤min⁡{1−βL,(1−β)24​L​K}\displaystyle\eta\leq\min\left\{\frac{1-\beta}{L},\frac{(1-\beta)^{2}}{4L\sqrt{K}}\right\} |  |

Applying Jensen’s inequality and using the fact that ‖X‖∗≤r​‖X‖F\|X\|\_{\*}\leq\sqrt{r}\|X\|\_{F} for X∈ℝm×nX\in\mathbb{R}^{m\times n} with r=min⁡{m,n}r=\min\{m,n\} (Proposition [2](#Thmprop2 "Proposition 2. ‣ A.1 Basic Facts for Matrix Norms ‣ Appendix A Appendix ‣ Convergence of Muon with Newton–Schulz")(iii)), we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖∇f​(Wt)‖∗]\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t})\|\_{\*}] | ≤r​(1T​∑t=0T−1𝔼​[‖∇f​(Wt)‖F])≤r​1T​∑t=0T−1𝔼​[‖∇f​(Wt)‖F2]\displaystyle\leq\sqrt{r}\left(\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t})\|\_{F}]\right)\leq\sqrt{r}\sqrt{\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t})\|\_{F}^{2}]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤r​2​(1−2​θ)1−4​θ⋅(1−β)​Dη​T+11−4​θ​(4​β​L​D(1−β)​T+2​(1−β1+β​σ2B))\displaystyle\leq\sqrt{r}\sqrt{\frac{2(1-2\theta)}{1-4\theta}\cdot\frac{(1-\beta)D}{\eta T}+\frac{1}{1-4\theta}\left(\frac{4\beta LD}{(1-\beta)T}+2\left(\frac{1-\beta}{1+\beta}\frac{\sigma^{2}}{B}\right)\right)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤2​r​(1−2​θ)1−4​θ⋅(1−β)​Dη​T+r1−4​θ⋅4​β​L​D(1−β)​T+r1−4​θ⋅2​(1−β)1+β⋅σ2B\displaystyle\leq\sqrt{\frac{2r(1-2\theta)}{1-4\theta}\cdot\frac{(1-\beta)D}{\eta T}}+\sqrt{\frac{r}{1-4\theta}\cdot\frac{4\beta LD}{(1-\beta)T}}+\sqrt{\frac{r}{1-4\theta}\cdot\frac{2(1-\beta)}{1+\beta}\cdot\frac{\sigma^{2}}{B}} |  |

where the last inequality is due to a+b≤a+b\sqrt{a+b}\leq\sqrt{a}+\sqrt{b}.
By setting η=min⁡{1−βL,(1−β)24​L​K}\eta=\min\left\{\frac{1-\beta}{L},\frac{(1-\beta)^{2}}{4L\sqrt{K}}\right\},
which guarantees η~≤1/L\tilde{\eta}\leq 1/L and θ<14\theta<\frac{1}{4},
we obtain,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖∇f​(Wt)‖∗]\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t})\|\_{\*}] | ≤𝒪​(r​L​DT)⏟Deterministic Term+𝒪​(r​β​L​D(1−β)​T)+𝒪​(r​σ2​(1−β)(1+β)​B)⏟Stochastic Term\displaystyle\leq\underbrace{\mathcal{O}\left(\sqrt{\frac{rLD}{T}}\right)}\_{\text{Deterministic Term}}+\underbrace{\mathcal{O}\left(\sqrt{\frac{r\beta LD}{(1-\beta)T}}\right)+\mathcal{O}\left(\sqrt{\frac{r\sigma^{2}(1-\beta)}{(1+\beta)B}}\right)}\_{\text{Stochastic Term}} |  |

By setting β=1−min⁡{L​D​Bσ​T,1}\beta=1-\min\left\{\frac{\sqrt{LDB}}{\sigma\sqrt{T}},1\right\}, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1T​∑t=0T−1𝔼​[‖∇f​(Wt)‖∗]\displaystyle\frac{1}{T}\sum\_{t=0}^{T-1}\mathbb{E}[\|\nabla f(W\_{t})\|\_{\*}] | ≤𝒪​(r​L​DT)+𝒪​(r​β​L​D(1−β)​T)+𝒪​(r​σ2​(1−β)(1+β)​B)\displaystyle\leq\mathcal{O}\left(\sqrt{\frac{rLD}{T}}\right)+\mathcal{O}\left(\sqrt{\frac{r\beta LD}{(1-\beta)T}}\right)+\mathcal{O}\left(\sqrt{\frac{r\sigma^{2}(1-\beta)}{(1+\beta)B}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤𝒪​(r​L​DT+(r2​σ2​L​DB​T)1/4)\displaystyle\leq\mathcal{O}\left(\sqrt{\frac{rLD}{T}}+\left(\frac{r^{2}\sigma^{2}LD}{BT}\right)^{1/4}\right) |  |

Thus, we can find an ϵ\epsilon-nuclear norm stationary point of ff with a complexity of

|  |  |  |
| --- | --- | --- |
|  | 𝒪​(max⁡{r​L​Dϵ2,r2​σ2​L​DB​ϵ4})\displaystyle\mathcal{O}\left(\max\left\{\frac{rLD}{\epsilon^{2}},\frac{r^{2}\sigma^{2}LD}{B\epsilon^{4}}\right\}\right) |  |

∎

## Appendix D Newton–Schulz Lemmas: Proofs

Remark [1](#Thmremark1 "Remark 1. ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz") (Initial residual below one).

With Xt,0=Mt/αtX\_{t,0}=M\_{t}/\alpha\_{t} and αt=max⁡{1,‖Mt‖F}\alpha\_{t}=\max\{1,\|M\_{t}\|\_{F}\},
we have δt,0∈[0,1)\delta\_{t,0}\in[0,1) for every tt;
moreover δt,0=0\delta\_{t,0}=0 when Mt=0M\_{t}=0.

###### Proof.

Let Mt=U​Σ​V⊤M\_{t}=U\Sigma V^{\top} (rank rtr\_{t}).
Then Xt,0​Xt,0⊤=U​(Σ2/αt2)​U⊤X\_{t,0}X\_{t,0}^{\top}=U(\Sigma^{2}/\alpha\_{t}^{2})U^{\top},
so on range​(Mt)\mathrm{range}(M\_{t}) the eigenvalues are σi2/αt2∈(0,1]\sigma\_{i}^{2}/\alpha\_{t}^{2}\in(0,1].
If rt=0r\_{t}=0, set δt,0=0\delta\_{t,0}=0.
Otherwise, the minimal positive eigenvalue λmin+=mini≤rt⁡σi2/αt2>0\lambda\_{\min}^{+}=\min\_{i\leq r\_{t}}\sigma\_{i}^{2}/\alpha\_{t}^{2}>0, hence by Lemma [1](#Thmlemma1 "Lemma 1 (Orthogonality residual vs. Polar approximation error). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),
δt,0=1−λmin+<1\delta\_{t,0}=1-\lambda\_{\min}^{+}<1.
∎

###### Lemma 9.

(Polar factor invariance under Newton–Schulz).

As Newton–Schulz iterates by the polynomial pκp\_{\kappa},
i.e., Xt,0=Mt/αtX\_{t,0}=M\_{t}/\alpha\_{t}, Xt,j+1=pκ​(Xt,j​Xt,j⊤)​Xt,jX\_{t,j+1}=p\_{\kappa}(X\_{t,j}X\_{t,j}^{\top})X\_{t,j},
the polar factor is invariant: Polar⁡(Mt)=Polar⁡(Xt,j)\operatorname{Polar}(M\_{t})=\operatorname{Polar}(X\_{t,j}) for all j≥0j\geq 0.

###### Proof.

First, the polar factor is invariant under scalar multiplication.
(∵\because Let SVD​(X)=(U,Σ,V)\mathrm{SVD}(X)=(U,\Sigma,V). Then Polar⁡(X)=U​V⊤\operatorname{Polar}(X)=UV^{\top}.
Now, let Y=c​XY=cX for c>0c>0. Then SVD​(Y)=SVD​(c​X)=(U,c​Σ,V)\mathrm{SVD}(Y)=\mathrm{SVD}(cX)=(U,c\Sigma,V). Thus, Polar⁡(Y)=U​V⊤\operatorname{Polar}(Y)=UV^{\top}.
Hence, Pt=Polar⁡(Mt)=Polar⁡(Mt/αt)=Polar⁡(Xt,0)P\_{t}=\operatorname{Polar}(M\_{t})=\operatorname{Polar}(M\_{t}/\alpha\_{t})=\operatorname{Polar}(X\_{t,0}) holds.

Now, to prove by induction on j≥0j\geq 0, we assume that Pt=Polar⁡(Xt,j)P\_{t}=\operatorname{Polar}(X\_{t,j}).
Note that one step of Newton–Schulz for polynomial pp is defined as

|  |  |  |
| --- | --- | --- |
|  | Xt,j+1=p​(Xt,j​Xt,j⊤)​Xt,j\displaystyle X\_{t,j+1}=p(X\_{t,j}X\_{t,j}^{\top})X\_{t,j} |  |

Let SVD​(Xt,j)=(U,Σ,V)\mathrm{SVD}(X\_{t,j})=(U,\Sigma,V) with Σ≻0\Sigma\succ 0 (on range​(Xt,j)\mathrm{range}(X\_{t,j})), so that Polar⁡(Xt,j)=U​V⊤\operatorname{Polar}(X\_{t,j})=UV^{\top}.
Then

|  |  |  |
| --- | --- | --- |
|  | p​(Xt,j​Xt,j⊤)​Xt,j=p​(U​Σ​V⊤​(U​Σ​V⊤)⊤)​U​Σ​V⊤=p​(U​Σ2​U⊤)​U​Σ​V⊤=U​p​(Σ2)​Σ​V⊤\displaystyle p(X\_{t,j}X\_{t,j}^{\top})X\_{t,j}=p(U\Sigma V^{\top}(U\Sigma V^{\top})^{\top})U\Sigma V^{\top}=p(U\Sigma^{2}U^{\top})U\Sigma V^{\top}=Up(\Sigma^{2})\Sigma V^{\top} |  |

Thus, p​(Xt,j​Xt,j⊤)​Xt,jp(X\_{t,j}X\_{t,j}^{\top})X\_{t,j} has the same left/right singular vectors U,VU,V as Xt,jX\_{t,j}.
Hence,

|  |  |  |
| --- | --- | --- |
|  | Polar⁡(Xt,j+1)=Polar⁡(p​(Xt,j​Xt,j⊤)​Xt,j)=Polar⁡(Xt,j)=U​V⊤\displaystyle\operatorname{Polar}(X\_{t,j+1})=\operatorname{Polar}(p(X\_{t,j}X\_{t,j}^{\top})X\_{t,j})=\operatorname{Polar}(X\_{t,j})=UV^{\top} |  |

By induction, we have

|  |  |  |
| --- | --- | --- |
|  | Pt=Polar⁡(Mt)=Polar⁡(Xt,0)=Polar⁡(Xt,1)=…=Polar⁡(Xt,q)\displaystyle P\_{t}=\operatorname{Polar}(M\_{t})=\operatorname{Polar}(X\_{t,0})=\operatorname{Polar}(X\_{t,1})=\ldots=\operatorname{Polar}(X\_{t,q}) |  |

which implies the polar factor invariance under Newton–Schulz updates.
∎

###### Lemma 10.

(Support invariance under Newton–Schulz).

As Newton–Schulz iterates by the polynomial pκp\_{\kappa},
i.e., Xt,0=Mt/αtX\_{t,0}=M\_{t}/\alpha\_{t}, Xt,j+1=pκ​(Xt,j​Xt,j⊤)​Xt,jX\_{t,j+1}=p\_{\kappa}(X\_{t,j}X\_{t,j}^{\top})X\_{t,j},
the support is invariant: range​(Mt)=range​(Xt,j)\mathrm{range}(M\_{t})=\mathrm{range}(X\_{t,j}) for all j≥0j\geq 0,
and pκ​(Xt,j​Xt,j⊤)p\_{\kappa}(X\_{t,j}X\_{t,j}^{\top}) is positive definite on range​(Xt,j)\mathrm{range}(X\_{t,j}).

###### Proof.

On range​(Xt,j)\mathrm{range}(X\_{t,j}) the spectrum of A:=Xt,j​Xt,j⊤A:=X\_{t,j}X\_{t,j}^{\top} lies in (0,1](0,1].
pκp\_{\kappa} is strictly positive on (0,1](0,1] (Proposition [1](#Thmprop1 "Proposition 1 (Properties of 𝑝_𝜅). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")),
so pκ​(A)|range​(Xt,j)p\_{\kappa}(A)|\_{\mathrm{range}(X\_{t,j})} is invertible.
Hence

|  |  |  |
| --- | --- | --- |
|  | range​(Xt,j+1)=range​(pκ​(A)​Xt,j)=range​(Xt,j),\displaystyle\mathrm{range}(X\_{t,j+1})=\mathrm{range}(p\_{\kappa}(A)X\_{t,j})=\mathrm{range}(X\_{t,j}), |  |

and by induction range​(Xt,j)=range​(Mt)\mathrm{range}(X\_{t,j})=\mathrm{range}(M\_{t}) for all jj.
∎

###### Remark 2 (Support‑aware details).

As in the main text,
all Newton–Schulz quantities are restricted to range​(Mt)\mathrm{range}(M\_{t}) (no standing full‑rank assumption). Lemma [10](#Thmlemma10 "Lemma 10. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz") preserves the column space;
PtP\_{t}, Xt,j​Xt,j⊤X\_{t,j}X\_{t,j}^{\top}, and Πt\Pi\_{t} vanish on range​(Mt)⟂\mathrm{range}(M\_{t})^{\perp}.

Lemma [1](#Thmlemma1 "Lemma 1 (Orthogonality residual vs. Polar approximation error). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") (Orthogonality residual vs. Polar approximation error).

Let λmin+\lambda\_{\min}^{+} be the smallest positive eigenvalue of Xt,q​Xt,q⊤X\_{t,q}X\_{t,q}^{\top}
restricted to range​(Mt)\mathrm{range}(M\_{t}) (set λmin+=1\lambda\_{\min}^{+}=1 if rank​(Mt)=0\mathrm{rank}(M\_{t})=0).
Then

|  |  |  |
| --- | --- | --- |
|  | δt,q=1−λmin+,εt,q=1−λmin+= 1−1−δt,q.\displaystyle\delta\_{t,q}=1-\lambda\_{\min}^{+},\qquad\varepsilon\_{t,q}=1-\sqrt{\lambda\_{\min}^{+}}\,=\,1-\sqrt{1-\delta\_{t,q}}. |  |

###### Proof.

Let Xt,q=U​Σ​V⊤X\_{t,q}=U\Sigma V^{\top} and Πt=U​U⊤\Pi\_{t}=UU^{\top} with Σ=diag⁡(σi)\Sigma=\operatorname{diag}(\sigma\_{i}).
Then,

|  |  |  |
| --- | --- | --- |
|  | Πt−Xt,q​Xt,q⊤=U​(I−Σ2)​U⊤\displaystyle\Pi\_{t}-X\_{t,q}X\_{t,q}^{\top}=U(I-\Sigma^{2})U^{\top} |  |

on range​(Mt)\mathrm{range}(M\_{t}) and vanishes on range​(Mt)⟂\mathrm{range}(M\_{t})^{\perp}.
Thus, the orthogonality residual δt,q\delta\_{t,q} is computed as

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,q=‖Πt−Xt,q​Xt,q⊤‖op=maxi⁡|1−σi2|=1−λmin+\displaystyle\delta\_{t,q}=\|\Pi\_{t}-X\_{t,q}X\_{t,q}^{\top}\|\_{\mathrm{op}}=\max\_{i}|1-\sigma\_{i}^{2}|=1-\lambda\_{\min}^{+} |  | (25) |

where λmin+\lambda\_{\min}^{+} is the smallest positive eigenvalue of Xt,q​Xt,q⊤X\_{t,q}X\_{t,q}^{\top} on range​(Mt)\mathrm{range}(M\_{t}).
By the polar factor invariance under Newton–Schulz updates (Lemma [9](#Thmlemma9 "Lemma 9. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),
we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | εt,q=‖Xt,q−Polar⁡(Mt)‖op=‖U​Σ​V⊤−U​V⊤‖op=‖Σ−I‖op=1−mini⁡σi=1−λmin+\displaystyle\varepsilon\_{t,q}=\|X\_{t,q}-\operatorname{Polar}(M\_{t})\|\_{\mathrm{op}}=\|U\Sigma V^{\top}-UV^{\top}\|\_{\mathrm{op}}=\|\Sigma-I\|\_{\mathrm{op}}=1-\min\_{i}\sigma\_{i}=1-\sqrt{\lambda\_{\min}^{+}} |  | (26) |

Combining Eq.([25](#A4.E25 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")) and Eq.([26](#A4.E26 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")), we conclude

|  |  |  |
| --- | --- | --- |
|  | εt,q=1−λmin+=1−1−δt,j\displaystyle\varepsilon\_{t,q}=1-\sqrt{\lambda\_{\min}^{+}}=1-\sqrt{1-\delta\_{t,j}} |  |

∎

Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") (Residual update).

For Newton–Schulz polynomial pκp\_{\kappa}
the orthogonality residual δt,j\delta\_{t,j} is updated by Newton–Schulz per step as

|  |  |  |
| --- | --- | --- |
|  | δt,j+1=ϕ​(δt,j),\displaystyle\delta\_{t,j+1}\;=\;\phi(\delta\_{t,j}), |  |

where ϕ​(u):=1−(1−u)​[pκ​(1−u)]2\phi(u):=1-(1-u)\left[p\_{\kappa}(1-u)\right]^{2}.

###### Proof.

Recall that Newton–Schulz update is defined as

|  |  |  |
| --- | --- | --- |
|  | Xt,j+1=p​(Xt,j​Xt,j⊤)​Xt,j,j=0,1,…,q−1\displaystyle X\_{t,j+1}=p(X\_{t,j}X\_{t,j}^{\top})X\_{t,j},\quad j=0,1,\dots,q-1 |  |

where ‖Xt,0‖op≤1\|X\_{t,0}\|\_{\mathrm{op}}\leq 1.
Let At,jA\_{t,j} be Xt,j​Xt,j⊤X\_{t,j}X\_{t,j}^{\top}.
Then At,jA\_{t,j} is symmetric, because

|  |  |  |
| --- | --- | --- |
|  | At,j⊤=(Xt,j​Xt,j⊤)⊤=(Xt,j⊤)⊤​Xt,j⊤=Xt,j​Xt,j⊤=At,j\displaystyle A\_{t,j}^{\top}=(X\_{t,j}X\_{t,j}^{\top})^{\top}=(X\_{t,j}^{\top})^{\top}X\_{t,j}^{\top}=X\_{t,j}X\_{t,j}^{\top}=A\_{t,j} |  |

Applying Newton–Schulz update, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | At,j+1\displaystyle A\_{t,j+1} | =Xt,j+1​Xt,j+1⊤=(p​(At,j)​Xt,j)​(p​(At,j)​Xt,j)⊤\displaystyle=X\_{t,j+1}X\_{t,j+1}^{\top}=(p(A\_{t,j})X\_{t,j})(p(A\_{t,j})X\_{t,j})^{\top} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =p​(At,j)​Xt,j​Xt,j⊤​p​(At,j)⊤=p​(At,j)​At,j​p​(At,j)⊤\displaystyle=p(A\_{t,j})X\_{t,j}X\_{t,j}^{\top}p(A\_{t,j})^{\top}=p(A\_{t,j})A\_{t,j}p(A\_{t,j})^{\top} |  |

Since At,jA\_{t,j} is symmetric, any polynomial p​(At,j)p(A\_{t,j}) is also symmetric.
Thus, p​(At,j)⊤=p​(At,j)p(A\_{t,j})^{\top}=p(A\_{t,j}).
Therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | At,j+1=p​(At,j)​At,j​p​(At,j)\displaystyle A\_{t,j+1}=p(A\_{t,j})A\_{t,j}p(A\_{t,j}) |  | (27) |

Since At,jA\_{t,j} is symmetric, At,jA\_{t,j} is orthogonally diagonalizable.
Let {λ0,λ1,…}\{\lambda\_{0},\lambda\_{1},\ldots\} be the eigenvalues of At,jA\_{t,j}.
Hence, we can write its spectral decomposition as
At,j=U​Λ​U⊤=U​diag⁡(λl)​U⊤A\_{t,j}=U\Lambda U^{\top}=U\operatorname{diag}(\lambda\_{l})U^{\top}.
By the key property of matrix polynomials,
p​(At,j)=U​p​(Λ)​U⊤=U​diag⁡(p​(λl))​U⊤p(A\_{t,j})=Up(\Lambda)U^{\top}=U\operatorname{diag}(p(\lambda\_{l}))U^{\top}.
Now, we substitute the spectral decompositions of At,jA\_{t,j} and p​(At,j)p(A\_{t,j}) into Eq.([27](#A4.E27 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),

|  |  |  |
| --- | --- | --- |
|  | At,j+1=(U​p​(Λ)​U⊤)​(U​Λ​U⊤)​(U​p​(Λ)​U⊤)=U​(p​(Λ)​Λ​p​(Λ))​U⊤\displaystyle A\_{t,j+1}=(Up(\Lambda)U^{\top})(U\Lambda U^{\top})(Up(\Lambda)U^{\top})=U(p(\Lambda)\Lambda p(\Lambda))U^{\top} |  |

Let the new diagonal matrix be Λ′:=p​(Λ)​Λ​p​(Λ)\Lambda^{\prime}:=p(\Lambda)\Lambda p(\Lambda).
Then, the diagonal entries of Λ′\Lambda^{\prime} are λℓ​[p​(λℓ)]2\lambda\_{\ell}[p(\lambda\_{\ell})]^{2}.
The expression for At,j+1=U​Λ′​U⊤A\_{t,j+1}=U\Lambda^{\prime}U^{\top},
which is the spectral decomposition of At,j+1A\_{t,j+1}.
This form explicitly shows that the eigenvalues of At,j+1A\_{t,j+1} are the diagonal entries of Λ′\Lambda^{\prime},
which are λℓ​[p​(λℓ)]2\lambda\_{\ell}[p(\lambda\_{\ell})]^{2}.

We conclude that At,j+1=p​(At,j)​At,j​p​(At,j)=U​diag⁡(λi​[p​(λℓ)]2)​U⊤A\_{t,j+1}=p(A\_{t,j})A\_{t,j}p(A\_{t,j})=U\,\operatorname{diag}(\lambda\_{i}[p(\lambda\_{\ell})]^{2})\,U^{\top} on range​(Mt)\mathrm{range}(M\_{t})
and both Πt\Pi\_{t} and At,jA\_{t,j} vanish on range​(Mt)⟂\mathrm{range}(M\_{t})^{\perp}.

Claim: ‖Xt,j‖op≤1\|X\_{t,j}\|\_{\mathrm{op}}\leq 1 for all j≥0j\geq 0.

For j=0j=0, it is trivial.
Assume that ‖Xt,j‖op≤1\|X\_{t,j}\|\_{\mathrm{op}}\leq 1 holds.
Then, ‖At,j‖op=‖Xt,j​Xt,j⊤‖op=‖Xt,j‖op2≤1\|A\_{t,j}\|\_{\mathrm{op}}=\|X\_{t,j}X\_{t,j}^{\top}\|\_{\mathrm{op}}=\|X\_{t,j}\|\_{\mathrm{op}}^{2}\leq 1.
Hence, the largest singular value of At,jA\_{t,j} is in [0,1][0,1], which implies all eigenvalues of symmetric At,jA\_{t,j} are in [0,1][0,1], i.e., maxl⁡λl=λmax≤1\max\_{l}\lambda\_{l}=\lambda\_{\text{max}}\leq 1.

|  |  |  |
| --- | --- | --- |
|  | ‖Xt,j+1‖op2=‖Xt,j+1​Xt,j+1⊤‖op=‖At,j+1‖op=maxℓ⁡(λℓ​[p​(λℓ)]2)\displaystyle\|X\_{t,j+1}\|\_{\mathrm{op}}^{2}=\|X\_{t,j+1}X\_{t,j+1}^{\top}\|\_{\mathrm{op}}=\|A\_{t,j+1}\|\_{\mathrm{op}}=\max\_{\ell}\left(\lambda\_{\ell}[p(\lambda\_{\ell})]^{2}\right) |  |

Since τ​(λ)=λ​[p​(λ)]2\tau(\lambda)=\lambda[p(\lambda)]^{2} is non-decreasing on [0,1][0,1]
with τ​(1)≤1\tau(1)\leq 1 (Proposition [1](#Thmprop1 "Proposition 1 (Properties of 𝑝_𝜅). ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")),
we have

|  |  |  |
| --- | --- | --- |
|  | ‖Xt,j+1‖op2=maxℓ⁡(τ​(λl))=τ​(maxℓ⁡λl)=τ​(λmax)≤τ​(1)≤1\displaystyle\|X\_{t,j+1}\|\_{\mathrm{op}}^{2}=\max\_{\ell}\left(\tau(\lambda\_{l})\right)=\tau(\max\_{\ell}\lambda\_{l})=\tau(\lambda\_{\text{max}})\leq\tau(1)\leq 1 |  |

By induction, we get ‖Xt,j‖op≤1\|X\_{t,j}\|\_{\mathrm{op}}\leq 1 for all j≥0j\geq 0.
The claim also implies ‖At,j‖op≤1\|A\_{t,j}\|\_{\mathrm{op}}\leq 1 for all j≥0j\geq 0.

Now, recall that the orthogonality residual at step jj is defined as

|  |  |  |
| --- | --- | --- |
|  | δt,j=‖Πt−Xt,j​Xt,j⊤‖op=‖Πt−At,j‖op\displaystyle\delta\_{t,j}=\left\|\Pi\_{t}-X\_{t,j}X\_{t,j}^{\top}\right\|\_{\mathrm{op}}=\left\|\Pi\_{t}-A\_{t,j}\right\|\_{\mathrm{op}} |  |

Let λmin+\lambda\_{\min}^{+} be the minimum eigenvalue of At,jA\_{t,j}, i.e., minl⁡λl=λmin+≤1\min\_{l}\lambda\_{l}=\lambda\_{\min}^{+}\leq 1.
Then, by the claim, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,j=‖Πt−At,j‖op=maxl⁡|1−λl|=1−λmin+\displaystyle\delta\_{t,j}=\left\|\Pi\_{t}-A\_{t,j}\right\|\_{\mathrm{op}}=\max\_{l}|1-\lambda\_{l}|=1-\lambda\_{\min}^{+} |  | (28) |

Now, we consider the next step residual δt,j+1\delta\_{t,j+1},

|  |  |  |
| --- | --- | --- |
|  | δt,j+1=‖Πt−At,j+1‖op=maxl⁡|1−λℓ​[p​(λℓ)]2|=maxl⁡(1−λℓ​[p​(λℓ)]2)\displaystyle\delta\_{t,j+1}=\left\|\Pi\_{t}-A\_{t,j+1}\right\|\_{\mathrm{op}}=\max\_{l}\left|1-\lambda\_{\ell}[p(\lambda\_{\ell})]^{2}\right|=\max\_{l}\left(1-\lambda\_{\ell}[p(\lambda\_{\ell})]^{2}\right) |  |

where the last equality is due to ‖At,j+1‖op≤1\|A\_{t,j+1}\|\_{\mathrm{op}}\leq 1.
Since τ​(λ)=λ​[p​(λ)]2\tau(\lambda)=\lambda[p(\lambda)]^{2} is non-decreasing on [0,1][0,1] with τ​(1)≤1\tau(1)\leq 1, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,j+1\displaystyle\delta\_{t,j+1} | =1−λmin+​[p​(λmin+)]2\displaystyle=1-\lambda\_{\min}^{+}[p(\lambda\_{\min}^{+})]^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−δt,j)​[p​(1−δt,j)]2\displaystyle=1-(1-\delta\_{t,j})[p(1-\delta\_{t,j})]^{2} |  |

where the last equality is due to Eq.([28](#A4.E28 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")).
Therefore, we conclude that the orthogonality residual is updated by Newton–Schulz as

|  |  |  |
| --- | --- | --- |
|  | δt,j+1=ϕ​(δt,j)\displaystyle\delta\_{t,j+1}=\phi(\delta\_{t,j}) |  |

where ϕ​(u)=1−(1−u)​[p​(1−u)]2\phi(u)=1-(1-u)[p(1-u)]^{2}
∎

Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")
(Residual Decay by Newton–Schulz Polynomial).
For Newton–Schulz polynomial pκp\_{\kappa}, ϕ​(u)≤uκ+1\phi(u)\leq u^{\kappa+1} on [0,1][0,1] where ϕ\phi is a function defined in Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
Hence, for every tt and all j≥0j\geq 0,

|  |  |  |
| --- | --- | --- |
|  | δt,j+1≤δt,jκ+1,δt,q≤δt,0(κ+1)q.\displaystyle\delta\_{t,j+1}\ \leq\ \delta\_{t,j}^{\,\kappa+1},\qquad\delta\_{t,q}\ \leq\ \delta\_{t,0}^{\,(\kappa+1)^{q}}. |  |

###### Proof.

The Newton–Schulz polynomial p​(λ)p(\lambda) of degree κ\kappa is defined as the truncation of the Taylor series of g​(λ)=λ−1/2g(\lambda)=\lambda^{-1/2} expanded around λ=1\lambda=1.
This means that the first κ\kappa derivatives of p​(λ)p(\lambda) and g​(λ)g(\lambda) are identical at λ=1\lambda=1:

|  |  |  |
| --- | --- | --- |
|  | p(s)​(1)=g(s)​(1)for ​s=0,1,…,κ\displaystyle p^{(s)}(1)=g^{(s)}(1)\qquad\text{for }s=0,1,\ldots,\kappa |  |

Let’s consider the variable substitution λ=1−u\lambda=1-u.
The function becomes f​(u)=g​(1−u)=(1−u)−1/2f(u)=g(1-u)=(1-u)^{-1/2}.
The polynomial p​(1−u)p(1-u) is then the Taylor expansion of f​(u)f(u) around u=0u=0 up to the term uκu^{\kappa}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(1−u)=∑s=0κf(s)​(0)s!​us=∑s=0κcs​us\displaystyle p(1-u)=\sum\_{s=0}^{\kappa}\frac{f^{(s)}(0)}{s!}u^{s}=\sum\_{s=0}^{\kappa}c\_{s}u^{s} |  | (29) |

The derivatives of f​(u)=(1−u)−1/2f(u)=(1-u)^{-1/2} at u=0u=0 are f(s)​(0)=(2​s)!s!​4sf^{(s)}(0)=\frac{(2s)!}{s!4^{s}},
so that the coefficients are cs=(2​s)!(s!)2​4sc\_{s}=\frac{(2s)!}{(s!)^{2}4^{s}}.
Moreover, since p​(1−u)p(1-u) and f​(u)f(u) share the same first κ\kappa derivatives at u=0u=0, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | f(s)​(0)=(−1)s​p(s)​(1),for ​s=1,…,κ\displaystyle f^{(s)}(0)=(-1)^{s}p^{(s)}(1),\qquad\text{for }s=1,\ldots,\kappa |  | (30) |

Consider the function τ​(λ):=λ​[p​(λ)]2\tau(\lambda):=\lambda[p(\lambda)]^{2}.

Claim: τ​(λ)\tau(\lambda) is non-decreasing on [0,1][0,1] with τ​(1)≤1\tau(1)\leq 1.

τ​(1)=1⋅[p​(1)]2=[g​(1)]2=1\tau(1)=1\cdot[p(1)]^{2}=[g(1)]^{2}=1 holds.
Now, the derivative of τ​(λ)\tau(\lambda) is

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=[p​(λ)]2+2​λ​p​(λ)​p′​(λ)=p​(λ)​(p​(λ)+2​λ​p′​(λ))\displaystyle\tau^{\prime}(\lambda)=[p(\lambda)]^{2}+2\lambda p(\lambda)p^{\prime}(\lambda)=p(\lambda)\left(p(\lambda)+2\lambda p^{\prime}(\lambda)\right) |  |

With λ=1−u\lambda=1-u, we have p′​(λ)=−p′​(1−u)p^{\prime}(\lambda)=-p^{\prime}(1-u).
Then, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ′​(λ)=p​(1−u)​(p​(1−u)−2​(1−u)​p′​(1−u))⏟:=S​(u)\displaystyle\tau^{\prime}(\lambda)=p(1-u)\underbrace{\left(p(1-u)-2(1-u)p^{\prime}(1-u)\right)}\_{:=S(u)} |  | (31) |

Since all coefficients of p​(1−u)p(1-u) are positive, i.e., cs=(2​s)!(s!)2​4s>0c\_{s}=\frac{(2s)!}{(s!)^{2}4^{s}}>0, we have

|  |  |  |
| --- | --- | --- |
|  | p​(1−u)>0​ for ​u∈[0,1]\displaystyle p(1-u)>0\text{ for }u\in[0,1] |  |

Now, it suffices to prove S​(u)≥0S(u)\geq 0 for u∈[0,1]u\in[0,1].
From Eq.([29](#A4.E29 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")), we can compute S​(u)S(u):

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(u)\displaystyle S(u) | =p​(1−u)−2​(1−u)​p′​(1−u)\displaystyle=p(1-u)-2(1-u)p^{\prime}(1-u) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑s=0κcs​us−2​(1−u)​∑s=1κs​cs​us−1\displaystyle=\sum\_{s=0}^{\kappa}c\_{s}u^{s}-2(1-u)\sum\_{s=1}^{\kappa}sc\_{s}u^{s-1} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(c0−2​c1)+∑s=1κ−1((2​s+1)​cs−2​(s+1)​cs+1)​us+(2​κ+1)​cκ​uκ\displaystyle=(c\_{0}-2c\_{1})+\sum\_{s=1}^{\kappa-1}\left((2s+1)c\_{s}-2(s+1)c\_{s+1}\right)u^{s}+(2\kappa+1)c\_{\kappa}u^{\kappa} |  | (32) |

Note that the ratio for the coefficients is,

|  |  |  |
| --- | --- | --- |
|  | cs+1cs=(2​s+2)!((s+1)!)2​4s+1(2​s)!(s!)2​4s=2​s+12​(s+1)\displaystyle\frac{c\_{s+1}}{c\_{s}}=\frac{\frac{(2s+2)!}{((s+1)!)^{2}4^{s+1}}}{\frac{(2s)!}{(s!)^{2}4^{s}}}=\frac{2s+1}{2(s+1)} |  |

Therefore,

|  |  |  |
| --- | --- | --- |
|  | (2​s+1)​cs−2​(s+1)​cs+1=(2​s+1)​cs−2​(s+1)​(2​s+12​(s+1)​cs)=0,\displaystyle(2s+1)c\_{s}-2(s+1)c\_{s+1}=(2s+1)c\_{s}-2(s+1)\left(\frac{2s+1}{2(s+1)}c\_{s}\right)=0, |  |

and also c0−2​c1=1−2⋅12=0c\_{0}-2c\_{1}=1-2\cdot\frac{1}{2}=0.
Hence, all coefficients up to degree κ−1\kappa-1 cancel in Eq.([32](#A4.E32 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),
leaving the exact factorization

|  |  |  |  |
| --- | --- | --- | --- |
|  | S​(u)=(2​κ+1)​cκ​uκ,cκ=(2​κ)!4κ​(κ!)2>0.\displaystyle S(u)=(2\kappa+1)c\_{\kappa}u^{\kappa},\qquad c\_{\kappa}=\frac{(2\kappa)!}{4^{\kappa}(\kappa!)^{2}}>0. |  | (33) |

Putting Eq.([33](#A4.E33 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")) into Eq.([31](#A4.E31 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")), which is τ′​(λ)=p​(1−u)​S​(u)\tau^{\prime}(\lambda)=p(1-u)S(u), gives

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=p​(1−u)​(2​κ+1)​cκ​uκ,with ​u=1−λ∈[0,1].\displaystyle\tau^{\prime}(\lambda)=p(1-u)(2\kappa+1)c\_{\kappa}u^{\kappa},\qquad\text{with }u=1-\lambda\in[0,1]. |  |

Since p​(1−u)>0p(1-u)>0, cκ>0c\_{\kappa}>0, and uκ≥0u^{\kappa}\geq 0 on [0,1][0,1],
we have τ′​(λ)≥0\tau^{\prime}(\lambda)\geq 0 for λ∈[0,1]\lambda\in[0,1].
Moreover, τ′​(λ)>0\tau^{\prime}(\lambda)>0 for λ∈[0,1)\lambda\in[0,1) (because then u>0u>0), and τ′​(1)=0\tau^{\prime}(1)=0.
Therefore, τ\tau is non-decreasing on [0,1][0,1] (in fact, it is strictly increasing on [0,1)[0,1)), as claimed.

Now, we can apply Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"), since
τ​(λ)\tau(\lambda) is non-decreasing on [0,1][0,1] with τ​(1)≤1\tau(1)\leq 1.
From Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),
we know the orthogonality residual is updated according to the rule:

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,j+1=ϕ​(δt,j)\displaystyle\delta\_{t,j+1}=\phi(\delta\_{t,j}) |  | (34) |

where the function ϕ​(u)\phi(u) is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(u)=1−(1−u)​[p​(1−u)]2\displaystyle\phi(u)=1-(1-u)[p(1-u)]^{2} |  | (35) |

Claim: ϕ(s)​(0)=0\phi^{(s)}(0)=0 for all s=1,…,κs=1,\ldots,\kappa.
The function ϕ​(u)\phi(u) can be rewritten using f​(u)=(1−u)−1/2f(u)=(1-u)^{-1/2} and p​(1−u)p(1-u):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(u)=1−1f​(u)2​[p​(1−u)]2\displaystyle\phi(u)=1-\frac{1}{f(u)^{2}}[p(1-u)]^{2} |  | (36) |

Let’s define a remainder function Rκ​(u)R\_{\kappa}(u),
which represents the difference between f​(u)f(u) and its Taylor approximation p​(1−u)p(1-u):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Rκ​(u)=f​(u)−p​(1−u)\displaystyle R\_{\kappa}(u)=f(u)-p(1-u) |  | (37) |

Since p​(1−u)p(1-u) matches the derivatives of f​(u)f(u) up to order κ\kappa at u=0u=0, which implies Eq.([30](#A4.E30 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),
the remainder function and its first κ\kappa derivatives are all zero at u=0u=0:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Rκ(s)​(0)=f(s)​(0)−p(s)​(1)⋅(−1)s=0,for ​s=1,…,κ\displaystyle R\_{\kappa}^{(s)}(0)=f^{(s)}(0)-p^{(s)}(1)\cdot(-1)^{s}=0,\qquad\text{for }s=1,\ldots,\kappa |  | (38) |

Substituting p​(1−u)=f​(u)−Rκ​(u)p(1-u)=f(u)-R\_{\kappa}(u) from Eq.([37](#A4.E37 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")) to Eq.([36](#A4.E36 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(u)\displaystyle\phi(u) | =1−[f​(u)−Rκ​(u)]2f​(u)2\displaystyle=1-\frac{[f(u)-R\_{\kappa}(u)]^{2}}{f(u)^{2}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−f​(u)2−2​f​(u)​Rκ​(u)+Rκ​(u)2f​(u)2\displaystyle=1-\frac{f(u)^{2}-2f(u)R\_{\kappa}(u)+R\_{\kappa}(u)^{2}}{f(u)^{2}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−2​R​(u)f​(u)+Rκ​(u)2f​(u)2)\displaystyle=1-\left(1-\frac{2R(u)}{f(u)}+\frac{R\_{\kappa}(u)^{2}}{f(u)^{2}}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =2​Rκ​(u)f​(u)−Rκ​(u)2f​(u)2=Rκ​(u)⋅(2​f​(u)−Rκ​(u)f​(u)2)\displaystyle=\frac{2R\_{\kappa}(u)}{f(u)}-\frac{R\_{\kappa}(u)^{2}}{f(u)^{2}}=R\_{\kappa}(u)\cdot\left(\frac{2f(u)-R\_{\kappa}(u)}{f(u)^{2}}\right) |  |

Let’s define the term in the brackets as a new function, H​(u)H(u):

|  |  |  |
| --- | --- | --- |
|  | H​(u)=2​f​(u)−Rκ​(u)f​(u)2\displaystyle H(u)=\frac{2f(u)-R\_{\kappa}(u)}{f(u)^{2}} |  |

So, we have a simple product form:
ϕ​(u)=Rκ​(u)​H​(u)\phi(u)=R\_{\kappa}(u)H(u).
Note that H​(u)H(u) is well-behaved around u=0u=0 since f​(0)=1f(0)=1 and Rκ​(0)=0R\_{\kappa}(0)=0, making the denominator non-zero.
To find the derivatives of ϕ​(u)\phi(u),
we use the Leibniz rule (the generalized product rule) for the ss-th derivative of a product:

|  |  |  |
| --- | --- | --- |
|  | ϕ(s)​(u)=dsd​us​[Rκ​(u)​H​(u)]=∑j=0s(sj)​Rκ(j)​(u)​H(s−j)​(u)\displaystyle\phi^{(s)}(u)=\frac{d^{s}}{du^{s}}[R\_{\kappa}(u)H(u)]=\sum\_{j=0}^{s}\binom{s}{j}R\_{\kappa}^{(j)}(u)H^{(s-j)}(u) |  |

Now, we evaluate this ss-th derivative at u=0u=0:

|  |  |  |
| --- | --- | --- |
|  | ϕ(s)​(0)=∑j=0s(sj)​Rκ(j)​(0)​H(s−j)​(0)\displaystyle\phi^{(s)}(0)=\sum\_{j=0}^{s}\binom{s}{j}R^{(j)}\_{\kappa}(0)H^{(s-j)}(0) |  |

Let’s consider any integer ss in the range 0≤s≤κ0\leq s\leq\kappa.
In the summation above, the index jj runs from 0 to ss.
For every term in this sum, the condition j≤s≤κj\leq s\leq\kappa holds.
From Eq.([38](#A4.E38 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")), we know that Rκ(j)​(0)=0R\_{\kappa}^{(j)}(0)=0 for all j≤κj\leq\kappa.
Therefore, every single term in the summation contains a factor of Rκ(j)​(0)R\_{\kappa}^{(j)}(0) which is equal to zero.

|  |  |  |
| --- | --- | --- |
|  | ϕ(s)​(0)=∑j=0s(sj)​(0)⋅H(s−j)​(0)=0\displaystyle\phi^{(s)}(0)=\sum\_{j=0}^{s}\binom{s}{j}(0)\cdot H^{(s-j)}(0)=0 |  |

This result holds for all s=1,…,κs=1,\ldots,\kappa.
Thus, we have proven that the first κ\kappa derivatives of ϕ​(u)\phi(u) at u=0u=0 are all zero:

|  |  |  |
| --- | --- | --- |
|  | ϕ​(0)=ϕ′​(u)=⋯=ϕ(κ)​(0)=0\displaystyle\phi(0)=\phi^{\prime}(u)=\cdots=\phi^{(\kappa)}(0)=0 |  |

This implies that the Taylor series for ϕ​(u)\phi(u) starts with a term of order uκ+1u^{\kappa+1}:

|  |  |  |
| --- | --- | --- |
|  | ϕ​(u)=ϕ(κ+1)​(0)(κ+1)!​uκ+1+𝒪​(uκ+2)\displaystyle\phi(u)=\frac{\phi^{(\kappa+1)}(0)}{(\kappa+1)!}u^{\kappa+1}+\mathcal{O}(u^{\kappa+2}) |  |

Find the leading term of ϕ​(u)\phi(u).

Recall that we can express p​(1−u)p(1-u) using the Taylor remainder theorem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(1−u)=f​(u)−Rκ​(u)\displaystyle p(1-u)=f(u)-R\_{\kappa}(u) |  | (39) |

where Rκ​(u)R\_{\kappa}(u) is the remainder term, which is of order 𝒪​(uκ+1)\mathcal{O}(u^{\kappa+1}).
Substituting Eq.([39](#A4.E39 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")) into the expression for ϕ​(u)\phi(u) in Eq.([35](#A4.E35 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(u)\displaystyle\phi(u) | =1−(1−u)​[f​(u)−Rκ​(u)]2\displaystyle=1-(1-u)[f(u)-R\_{\kappa}(u)]^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−u)​[f​(u)2−2​f​(u)​Rκ​(u)+Rκ​(u)2]\displaystyle=1-(1-u)[f(u)^{2}-2f(u)R\_{\kappa}(u)+R\_{\kappa}(u)^{2}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−u)​[11−u−2​Rκ​(u)1−u+Rκ​(u)2]\displaystyle=1-(1-u)\left[\frac{1}{1-u}-\frac{2R\_{\kappa}(u)}{\sqrt{1-u}}+R\_{\kappa}(u)^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−[1−2​1−u​Rκ​(u)+(1−u)​Rκ​(u)2]\displaystyle=1-\left[1-2\sqrt{1-u}R\_{\kappa}(u)+(1-u)R\_{\kappa}(u)^{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =2​1−u​Rκ​(u)−(1−u)​Rκ​(u)2\displaystyle=2\sqrt{1-u}R\_{\kappa}(u)-(1-u)R\_{\kappa}(u)^{2} |  |

The remainder Rκ​(u)R\_{\kappa}(u) is given by Rκ​(u)=f(κ+1)​(c)(κ+1)!​uκ+1R\_{\kappa}(u)=\frac{f^{(\kappa+1)}(c)}{(\kappa+1)!}u^{\kappa+1} for some c∈(0,u)c\in(0,u).
As u→0u\to 0, the leading term of ϕ​(u)\phi(u) is determined by 2​1−u⋅f(κ+1)​(0)(κ+1)!​uκ+12\sqrt{1-u}\cdot\frac{f^{(\kappa+1)}(0)}{(\kappa+1)!}u^{\kappa+1}.
The derivatives of f​(u)=(1−u)−1/2f(u)=(1-u)^{-1/2} at u=0u=0 are f(s)​(0)=(2​s)!s!​4sf^{(s)}(0)=\frac{(2s)!}{s!4^{s}}.
Thus, the leading coefficient of the Taylor series for ϕ​(u)\phi(u) is:

|  |  |  |
| --- | --- | --- |
|  | Cκ=2​f(κ+1)​(0)(κ+1)!=2(κ+1)!​(2​(κ+1))!(κ+1)!​4κ+1=24κ+1​(2​κ+2κ+1)\displaystyle C\_{\kappa}=\frac{2f^{(\kappa+1)}(0)}{(\kappa+1)!}=\frac{2}{(\kappa+1)!}\frac{(2(\kappa+1))!}{(\kappa+1)!4^{\kappa+1}}=\frac{2}{4^{\kappa+1}}\binom{2\kappa+2}{\kappa+1} |  |

For any κ≥1\kappa\geq 1, this coefficient is less than 11.
For example, for κ=1\kappa=1, C1=216​(42)=1216=34<1C\_{1}=\frac{2}{16}\binom{4}{2}=\frac{12}{16}=\frac{3}{4}<1.
In general, using Stirling’s approximation (2​nn)≤4nπ​n\binom{2n}{n}\leq\frac{4^{n}}{\sqrt{\pi n}}, we have:

|  |  |  |
| --- | --- | --- |
|  | Cκ=24κ+1​(2​κ+2κ+1)≤24κ+1​4κ+1π​(κ+1)=2π​(κ+1)<1for ​κ≥1.\displaystyle C\_{\kappa}=\frac{2}{4^{\kappa+1}}\binom{2\kappa+2}{\kappa+1}\leq\frac{2}{4^{\kappa+1}}\frac{4^{\kappa+1}}{\sqrt{\pi(\kappa+1)}}=\frac{2}{\sqrt{\pi(\kappa+1)}}<1\qquad\text{for }\kappa\geq 1. |  |

Since ϕ​(u)=Cκ​uκ+1+𝒪​(uκ+2)\phi(u)=C\_{\kappa}u^{\kappa+1}+\mathcal{O}(u^{\kappa+2}) and the leading coefficient CκC\_{\kappa} is less than 11, there exists a sufficiently small ρκ∈(0,1)\rho\_{\kappa}\in(0,1) such that ϕ​(u)≤uκ+1\phi(u)\leq u^{\kappa+1} for all u∈[0,ρκ]u\in[0,\rho\_{\kappa}].
Specifically, you can choose any θ∈(Cκ,1)\theta\in(C\_{\kappa},1) and take ρκ\rho\_{\kappa}
so that |ϕ​(u)−Cκ​uκ+1|≤(θ−Cκ)​uκ+1|\phi(u)-C\_{\kappa}u^{\kappa+1}|\leq(\theta-C\_{\kappa})u^{\kappa+1} on [0,ρκ][0,\rho\_{\kappa}].
Then

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(u)≤θ​uκ+1≤uκ+1\displaystyle\phi(u)\leq\theta u^{\kappa+1}\leq u^{\kappa+1} |  | (40) |

which makes the "exists ρκ\rho\_{\kappa}" completely explicit.

Combining the recursion Eq.([34](#A4.E34 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")) and Eq.([40](#A4.E40 "In Proof. ‣ Appendix D Newton–Schulz Lemmas: Proofs ‣ Convergence of Muon with Newton–Schulz")),
we have

|  |  |  |
| --- | --- | --- |
|  | δt,j+1=ϕ​(δt,j)≤(δt,j)κ+1\displaystyle\delta\_{t,j+1}=\phi(\delta\_{t,j})\leq(\delta\_{t,j})^{\kappa+1} |  |

After qq steps, we obtain the final result by repeated application:

|  |  |  |
| --- | --- | --- |
|  | δt,q≤(δt,q−1)κ+1≤((δt,q−2)κ+1)κ+1≤⋯​(δt,0)(κ+1)q\displaystyle\delta\_{t,q}\leq(\delta\_{t,q-1})^{\kappa+1}\leq((\delta\_{t,q-2})^{\kappa+1})^{\kappa+1}\leq\cdots(\delta\_{t,0})^{(\kappa+1)^{q}} |  |

Finally, we conclude:

|  |  |  |
| --- | --- | --- |
|  | δt,q≤δt,0(κ+1)q\displaystyle\delta\_{t,q}\leq\delta\_{t,0}^{(\kappa+1)^{q}} |  |

This establishes that the orthogonality residual decays with an order of κ+1\kappa+1
at each step of the Newton–Schulz if Newton–Schulz polynomial is of degree κ\kappa.
∎

###### Corollary 1 (Final constant factor bound).

Let δ0:=suptδt,0<1\delta\_{0}:=\sup\_{t}\delta\_{t,0}<1 (Remark [1](#Thmremark1 "Remark 1. ‣ 3.4 Newton–Schulz polynomial ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")). Then for all tt and q≥1q\geq 1,

|  |  |  |
| --- | --- | --- |
|  | δt,q≤δ0(κ+1)q,εq≤1− 1−δ0(κ+1)q,χq=(1−εq)−1≤[1−δ0(κ+1)q]−1/2.\displaystyle\delta\_{t,q}\ \leq\ \delta\_{0}^{(\kappa+1)^{q}},\qquad\varepsilon\_{q}\leq 1-\sqrt{\,1-\delta\_{0}^{(\kappa+1)^{q}}\,},\qquad\chi\_{q}=(1-\varepsilon\_{q})^{-1}\leq\bigl[1-\delta\_{0}^{(\kappa+1)^{q}}\bigr]^{-1/2}. |  |

###### Proof.

Combine Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") with Lemma [1](#Thmlemma1 "Lemma 1 (Orthogonality residual vs. Polar approximation error). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
∎

###### Corollary 2 (Case when κ∈{1,2}\kappa\in\{1,2\}).

For the polynomials p1​(λ)=32−12​λp\_{1}(\lambda)=\tfrac{3}{2}-\tfrac{1}{2}\lambda and
p2​(λ)=158−54​λ+38​λ2p\_{2}(\lambda)=\tfrac{15}{8}-\tfrac{5}{4}\lambda+\tfrac{3}{8}\lambda^{2},
the residual recursion specializes to
δt,j+1≤δt,j2\delta\_{t,j+1}\leq\delta\_{t,j}^{2} and
δt,j+1≤δt,j3\delta\_{t,j+1}\leq\delta\_{t,j}^{3} for all δt,j∈[0,1]\delta\_{t,j}\in[0,1].
These are concrete instances of the general residual map in Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").

###### Proof.

Direct substitution into ϕ​(u)=1−(1−u)​[pκ​(1−u)]2\phi(u)=1-(1-u)[p\_{\kappa}(1-u)]^{2} gives
δj+1=δj2​(3+δj)4≤δj2\delta\_{j+1}=\frac{\delta\_{j}^{2}(3+\delta\_{j})}{4}\leq\delta\_{j}^{2} since 3+δ≤43+\delta\leq 4 on [0,1][0,1] for κ=1\kappa=1.
Also,
δj+1=δj3​(40+15​δj+9​δj2)64≤δj3\delta\_{j+1}=\frac{\delta\_{j}^{3}(40+15\delta\_{j}+9\delta\_{j}^{2})}{64}\leq\delta\_{j}^{3} since
40+15​δ+9​δ2≤6440+15\delta+9\delta^{2}\leq 64 on [0,1][0,1] for κ=2\kappa=2 (check at δ=1\delta=1).
∎

### D.1 Detailed proofs for case when κ∈{1,2}\kappa\in\{1,2\}

Residual Decay by 1st-order Newton–Schulz Polynomial.
Define the polynomial p​(λ)p(\lambda) for Newton–Schulz steps as

|  |  |  |
| --- | --- | --- |
|  | p​(λ)=32−12​λ\displaystyle p(\lambda)=\frac{3}{2}-\frac{1}{2}\lambda |  |

Then for any fixed tt and any j≥0j\geq 0, the residual δt,j\delta\_{t,j} satisfies

|  |  |  |
| --- | --- | --- |
|  | δt,j+1≤(δt,j)2\displaystyle\delta\_{t,j+1}\leq(\delta\_{t,j})^{2} |  |

Consequently, if δt,0≤ρ<1\delta\_{t,0}\leq\rho<1 is sufficiently small for all tt, then δt,q≤ρ2q\delta\_{t,q}\leq\rho^{2^{q}} for all tt.

###### Proof.

Deriving the function τ​(λ)\tau(\lambda) in Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),

|  |  |  |
| --- | --- | --- |
|  | τ​(λ)=λ​[p​(λ)]2=λ​(32−12​λ)2=94​λ−32​λ2+14​λ3\displaystyle\tau(\lambda)=\lambda[p(\lambda)]^{2}=\lambda\left(\frac{3}{2}-\frac{1}{2}\lambda\right)^{2}=\frac{9}{4}\lambda-\frac{3}{2}\lambda^{2}+\frac{1}{4}\lambda^{3} |  |

Then the derivative of τ​(λ)\tau(\lambda) is

|  |  |  |
| --- | --- | --- |
|  | τ′​(λ)=94−3​λ+34​λ2=34​(λ−1)​(λ−3)\displaystyle\tau^{\prime}(\lambda)=\frac{9}{4}-3\lambda+\frac{3}{4}\lambda^{2}=\frac{3}{4}(\lambda-1)(\lambda-3) |  |

Since τ′​(λ)≥0\tau^{\prime}(\lambda)\geq 0 for λ∈[0,1]\lambda\in[0,1] and τ​(1)=1\tau(1)=1,
we can apply Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
The orthogonality residual δt,j\delta\_{t,j} is updated by one Newton–Schulz step as

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,j+1\displaystyle\delta\_{t,j+1} | =ϕ​(δt,j)=1−(1−δt,j)​[p​(1−δt,j)]2\displaystyle=\phi(\delta\_{t,j})=1-(1-\delta\_{t,j})[p(1-\delta\_{t,j})]^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−δt,j)​(32−12​(1−δt,j))2=δt,j2​(3+δt,j)4≤δt,j2\displaystyle=1-(1-\delta\_{t,j})\left(\frac{3}{2}-\frac{1}{2}(1-\delta\_{t,j})\right)^{2}=\frac{\delta\_{t,j}^{2}(3+\delta\_{t,j})}{4}\leq\delta\_{t,j}^{2} |  |

By induction, after qq steps, we get the relationship,
δt,q≤(δt,0)2q\delta\_{t,q}\leq(\delta\_{t,0})^{2^{q}}.
Given the assumption that δt,0≤ρ<1\delta\_{t,0}\leq\rho<1 for all tt, the conclusion follows directly:

|  |  |  |
| --- | --- | --- |
|  | δt,q≤ρ2q\displaystyle\delta\_{t,q}\leq\rho^{2^{q}} |  |

This shows that the orthogonality residual decreases quadratically with each iteration, which is a very rapid rate of convergence.
∎

Residual Decay by 2nd-order Newton–Schulz Polynomial.
Define the polynomial p​(λ)p(\lambda) for Newton–Schulz steps as

|  |  |  |
| --- | --- | --- |
|  | p​(λ)=158−54​λ+38​λ2\displaystyle p(\lambda)=\frac{15}{8}-\frac{5}{4}\lambda+\frac{3}{8}\lambda^{2} |  |

Then for any fixed tt and any j≥0j\geq 0, the residual δt,j\delta\_{t,j} satisfies

|  |  |  |
| --- | --- | --- |
|  | δt,j+1≤(δt,j)3\displaystyle\delta\_{t,j+1}\leq(\delta\_{t,j})^{3} |  |

Consequently, if δt,0≤ρ<1\delta\_{t,0}\leq\rho<1 is sufficiently small for all tt, then δt,q≤ρ3q\delta\_{t,q}\leq\rho^{3^{q}} for all tt.

###### Proof.

Deriving the function τ​(λ)\tau(\lambda) in Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz"),

|  |  |  |
| --- | --- | --- |
|  | τ​(λ)=λ​[p​(λ)]2=λ​(158−54​λ+38​λ2)2\displaystyle\tau(\lambda)=\lambda[p(\lambda)]^{2}=\lambda\left(\frac{15}{8}-\frac{5}{4}\lambda+\frac{3}{8}\lambda^{2}\right)^{2} |  |

Then the derivative of τ​(λ)\tau(\lambda) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | τ′​(λ)\displaystyle\tau^{\prime}(\lambda) | =(158−54​λ+38​λ2)2+2​λ​(158−54​λ+38​λ2)\displaystyle=\left(\frac{15}{8}-\frac{5}{4}\lambda+\frac{3}{8}\lambda^{2}\right)^{2}+2\lambda\left(\frac{15}{8}-\frac{5}{4}\lambda+\frac{3}{8}\lambda^{2}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(158−54​λ+38​λ2)​(158+34​λ+38​λ2)=164​(20+(5−3​λ)2)​(4+(1+λ)2)\displaystyle=\left(\frac{15}{8}-\frac{5}{4}\lambda+\frac{3}{8}\lambda^{2}\right)\left(\frac{15}{8}+\frac{3}{4}\lambda+\frac{3}{8}\lambda^{2}\right)=\frac{1}{64}\left(20+(5-3\lambda)^{2}\right)\left(4+(1+\lambda)^{2}\right) |  |

Since τ′​(λ)≥0\tau^{\prime}(\lambda)\geq 0 for λ∈[0,1]\lambda\in[0,1] and τ​(1)=1\tau(1)=1,
we can apply Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz").
The orthogonality residual δt,j\delta\_{t,j} is updated by one Newton–Schulz step as

|  |  |  |  |
| --- | --- | --- | --- |
|  | δt,j+1\displaystyle\delta\_{t,j+1} | =ϕ​(δt,j)=1−(1−δt,j)​[p​(1−δt,j)]2\displaystyle=\phi(\delta\_{t,j})=1-(1-\delta\_{t,j})[p(1-\delta\_{t,j})]^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−δt,j)​(158−54​(1−δt,j)+38​(1−δt,j)2)2\displaystyle=1-(1-\delta\_{t,j})\left(\frac{15}{8}-\frac{5}{4}(1-\delta\_{t,j})+\frac{3}{8}(1-\delta\_{t,j})^{2}\right)^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−δt,j)​(1+12​δt,j+38​δt,j2)2\displaystyle=1-(1-\delta\_{t,j})\left(1+\frac{1}{2}\delta\_{t,j}+\frac{3}{8}\delta\_{t,j}^{2}\right)^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =1−(1−δt,j)​(1+δt,j+δt,j2+38​δt,j3+964​δt,j4)\displaystyle=1-(1-\delta\_{t,j})\left(1+\delta\_{t,j}+\delta\_{t,j}^{2}+\frac{3}{8}\delta\_{t,j}^{3}+\frac{9}{64}\delta\_{t,j}^{4}\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =δt,j3​(40+15​δt,j+9​δt,j2)64≤δt,j3\displaystyle=\frac{\delta\_{t,j}^{3}(40+15\delta\_{t,j}+9\delta\_{t,j}^{2})}{64}\leq\delta\_{t,j}^{3} |  |

By induction, after qq steps, we get the relationship,
δt,q≤(δt,0)3q\delta\_{t,q}\leq(\delta\_{t,0})^{3^{q}}.
Given the assumption that δt,0≤ρ<1\delta\_{t,0}\leq\rho<1 for all tt, the conclusion follows directly:

|  |  |  |
| --- | --- | --- |
|  | δt,q≤ρ3q\displaystyle\delta\_{t,q}\leq\rho^{3^{q}} |  |

This shows that the orthogonality residual decreases cubically with each iteration, which is a very rapid rate of convergence.
∎

## Appendix E Wall-Clock via Computational Complexity.

At each iteration tt, Muon performs an orthogonalization of the momentum matrix MtM\_{t} via either SVD\mathrm{SVD} or Newton–Schulz (NS).
We write Φgemm\Phi\_{\text{gemm}} for the effective GEMM throughput (FLOP/s),
and Φsvd\Phi\_{\text{svd}} for the effective throughput of the SVD\mathrm{SVD} routine.
In practice Φgemm≫Φsvd\Phi\_{\text{gemm}}\gg\Phi\_{\text{svd}} due to far higher hardware utilization of GEMM.

##### Per-iteration Orthogonalization FLOPs.

For a single layer index by ℓ\ell with m≤nm\leq n (for m>nm>n, apply to Mt⊤M\_{t}^{\top} and transpose-trick):

* •

  Muon with SVD.
  A thin SVD\mathrm{SVD} of Mt∈ℝm×nM\_{t}\in\mathbb{R}^{m\times n} and extracting the polar factor Ut​Vt⊤U\_{t}V\_{t}^{\top} costs (Golub and Reinsch, [1971](#bib.bib21 "Singular value decomposition and least squares solutions")):

  |  |  |  |
  | --- | --- | --- |
  |  | FLOPssvd(ℓ)​(m,n)=Θ​(4​m2​n+8​m3)\displaystyle\text{FLOPs}\_{\text{svd}}^{(\ell)}(m,n)=\Theta(4m^{2}n+8m^{3}) |  |

  Wall-clock time per layer is tsvd(ℓ)=FLOPssvd(ℓ)​(m,n)/Φsvdt\_{\text{svd}}^{(\ell)}=\text{FLOPs}\_{\text{svd}}^{(\ell)}(m,n)/\Phi\_{\text{svd}}.
* •

  Muon with NS (qq-steps, κ\kappa-degree).
  Newton–Schulz follows Horner’s rule when recursively updating the scaled momentum matrix using the Newton–Schulz polynomial.
  Newton–Schulz forms A=X​X⊤∈ℝm×mA=XX^{\top}\in\mathbb{R}^{m\times m} and applies the degree-κ\kappa polynomial to XX via Horner’s rule.
  Each NS step needs one m×nm\times n by n×mn\times m GEMM to build AA and κ\kappa multiplies A​YAY (each m×mm\times m by m×nm\times n). Hence,

  |  |  |  |
  | --- | --- | --- |
  |  | FLOPsns(ℓ)​(m,n;q,κ)=Θ​(2​q​(κ+1)​m2​n)\displaystyle\text{FLOPs}\_{\text{ns}}^{(\ell)}(m,n;q,\kappa)=\Theta(2q(\kappa+1)m^{2}n) |  |

  Wall-clock time per layer is tns(ℓ)=FLOPsns(ℓ)​(m,n;q,κ)/Φgemmt\_{\text{ns}}^{(\ell)}=\text{FLOPs}\_{\text{ns}}^{(\ell)}(m,n;q,\kappa)/\Phi\_{\text{gemm}}.

###### Lemma 11.

For a layer with m≤nm\leq n, the wall-clock time ratio between Muon with SVD\mathrm{SVD} and Muon with Newton–Schulz (qq-steps, κ\kappa-degree) is,

|  |  |  |
| --- | --- | --- |
|  | tsvd(ℓ)tns(ℓ)=Θ​(4​m2​n+8​m3)Θ​(2​q​(κ+1)​m2​n)⋅ΦgemmΦsvd=Θ​(2+4​mnq​(κ+1))⋅ΦgemmΦsvd⏟efficiency ratio ≫1\displaystyle\frac{t\_{\text{svd}}^{(\ell)}}{t\_{\text{ns}}^{(\ell)}}=\frac{\Theta(4m^{2}n+8m^{3})}{\Theta(2q(\kappa+1)m^{2}n)}\cdot\frac{\Phi\_{\text{gemm}}}{\Phi\_{\text{svd}}}=\Theta\left(\frac{2+4\tfrac{m}{n}}{q(\kappa+1)}\right)\cdot\underbrace{\frac{\Phi\_{\text{gemm}}}{\Phi\_{\text{svd}}}}\_{\text{efficiency ratio }\gg 1} |  |

##### Discussion of Lemma [11](#Thmlemma11 "Lemma 11. ‣ Per-iteration Orthogonalization FLOPs. ‣ Appendix E Wall-Clock via Computational Complexity. ‣ Convergence of Muon with Newton–Schulz").

With the practical setting q∈{2,3}q\in\{2,3\} and κ∈{1,2}\kappa\in\{1,2\},
the algebraic factor 2+4​(m/n)q​(κ+1)\frac{2+4(m/n)}{q(\kappa+1)} is 𝒪​(0.3∼1)\mathcal{O}(0.3{\sim}1),
so the wall-clock speedup is essentially the GEMM/SVD\mathrm{SVD} efficiency ratio.
On modern GPUs, Φgemm/Φsvd\Phi\_{\text{gemm}}/\Phi\_{\text{svd}} is often 4∼104{\sim}10,
so NS typically yields a multi-×\times speedup per iteration over SVD\mathrm{SVD}, matching empirical observations in Figure [1](#S5.F1 "Figure 1 ‣ 5 Numerical Experiments ‣ Convergence of Muon with Newton–Schulz").

Practical Interpretation.
Newton–Schulz scales linearly in qq (accuracy knob) and uses only GEMMs, which map efficiently to GPUs.
Exact SVD pays an additional m3m^{3} term and typically incurs larger constants.
Hence for small qq and modest κ\kappa, Newton–Schulz is substantially cheaper per update than a full SVD while achieving near-exact orthogonalization in practice.

## Appendix F Numerical Experiments Detail

### F.1 Experimental Setting

Task and metric.
All experiments are conducted on CIFAR‑10 (50k train / 10k test) with the standard channel‑wise normalization
mean=(0.4914,0.4822,0.4465)\text{mean}=(0.4914,0.4822,0.4465) and std=(0.2470,0.2435,0.2616)\text{std}=(0.2470,0.2435,0.2616).

We report cross‑entropy train loss and test loss.
Plots show the (m​e​a​n±1⋅s​t​d)(mean\pm 1\cdot std) over 55 independent runs,
and we also report the wall-clock time (in seconds)
accumulated from the beginning of training.

Model.
We use a compact convolutional network
(CifarNet, approximately 2​M2M parameters): a fixed whitening 2×22\times 2 convolution (weights frozen, bias trainable),
followed by three ConvGroups
(each: 3×33\times 3 conv →\rightarrow MaxPool2d(2) →\rightarrow GELU →\rightarrow 3×33\times 3 conv →\rightarrow GELU)
with widths (64,256,256)(64,256,256) and a linear head.
BatchNorm layers keep affine weights frozen (biases are trainable).
For stability, the first portion of each convolution is Dirac-initialized, and the linear head is variance-normalized.

Hyperparameters and schedules.
Unless stated otherwise,
we train for 5050 epochs with a batch size of B=512B=512 on the GPU
(the CPU fallback uses B=256B=256).
Each run uses a different seed.

* •

  Main optimizer (one of Muon (q)(q), Muon (SVD\mathrm{SVD}), or SGD-M) on convolutional filters:
  learning rate η=0.0860632\eta=0.0860632,
  momentum β=0.730778\beta=0.730778.
* •

  Auxiliary SGD
  (whitening bias, BatchNorm biases, linear head):
  learning rates ηother=1.4949×10−3\eta\_{\text{other}}=1.4949\times 10^{-3} and
  ηhead=1.72446\eta\_{\text{head}}=1.72446 with momentum 0.9897030.989703 (Nesterov).
* •

  Label smoothing 0.20.2, and
  gradient clipping ∥⋅∥2≤1.0\|\cdot\|\_{2}\leq 1.0.
* •

  Schedule: The same warm‑up + cosine schedule is applied to all optimizers:
  a 5% linear warm‑up of total steps followed by cosine decay to 0.
  Schedulers step once per update.

Evaluation.
At the end of each epoch, we evaluate the test loss using the same normalization (no augmentation).
We record
(i) epoch-wise training loss,
(ii) test loss,
and (iii) the cumulative wall-clock time.
Results are aggregated across 55 runs.
Figures report (m​e​a​n±1⋅s​t​d)(mean\pm 1\cdot std) and include both epoch-aligned and time-aligned views to disentangle statistical and systemic effects.

### F.2 Newton–Schulz steps (qq) ablations.

Optimizers under comparison.
We compare:

1. 1.

   Muon qq-step: SGD with momentum followed by an orthogonalization of the momentum matrix using qq Newton–Schulz steps (Algorithm [1](#alg1 "Algorithm 1 ‣ 3.3 Muon Algorithm and Newton–Schulz Orthogonalization ‣ 3 Preliminaries ‣ Convergence of Muon with Newton–Schulz")); q∈{0,1,2,3}q\in\{0,1,2,3\}.
   The case q=0q=0 is a normalization‑only ablation (no orthogonalization).
2. 2.

   Muon (SVD\mathrm{SVD}): Muon with an exact polar step U​V⊤UV^{\top} via SVD\mathrm{SVD}.
3. 3.

   SGD-M: SGD with momentum baseline with identical schedules.

All methods share identical schedules and auxiliary updates (Appendix [F.1](#A6.SS1 "F.1 Experimental Setting ‣ Appendix F Numerical Experiments Detail ‣ Convergence of Muon with Newton–Schulz")).

Orthogonalization details.
For Muon with qq-Newton–Schulz steps, we use Newton–Schulz polynomial, pκp\_{\kappa}

|  |  |  |
| --- | --- | --- |
|  | p​(λ)=a+b​λ+c​λ2,(a,b,c)=(15/8,−5/4,3/8),\displaystyle p(\lambda)=a+b\lambda+c\lambda^{2},\qquad(a,b,c)=(15/8,-5/4,3/8), |  |

applied as X←a​X+(b​A+c​A2)​XX\leftarrow aX+(bA+cA^{2})X with A=X​X⊤A=XX^{\top} and X=M/‖M‖FX=M/\|M\|\_{F};
if the matrix is tall, we employ the transpose trick.
The SVD\mathrm{SVD} variant normalizes MM by its Frobenius norm and returns U​V⊤UV^{\top}.
In all Muon variants, each weight tensor is re-normalized once per step to the Frobenius norm out\_channels\sqrt{\text{out\\_channels}} to stabilize scales.

Ablations on qq.
Our main comparison varies the number of Newton–Schulz steps q∈{1,2,3}q\in\{1,2,3\}
while holding the polynomial fixed to pκp\_{\kappa},
and contrasts these with Muon with SVD\mathrm{SVD} and SGD-M.
All other components (architecture, schedules, augmentation) are
kept identical across methods to ensure a fair comparison.

## Appendix G Additional Numerical Experiments

Optimizers compared:

* •

  SGD with Momentum
* •

  Muon (Newton–Schulz) with 1, 2, 3 steps per update
* •

  Muon with exact SVD (polar factor)

Dataset and Model:

* •

  MNIST (60K) / MLP (0.5M)
* •

  CIFAR-10 (50K) / CifarNet (2M) : Main text
* •

  CIFAR-100 (50K) / ResNet-18 (11.2M)
* •

  Tiny-ImageNet (100K) / WideResNet-28-10 (36.6M)
* •

  FineWeb (10M tokens from sample-10BT) / NanoGPT (124.2M) & GPT-2 (1.3B)

  + –

    block\_size = 1024
  + –

    num\_blocks = (10M - 1) // 1024 = 9,765 sequences (for causal LM)
  + –

    n\_layer: 12 & 24 (transformer blocks)
  + –

    n\_head: 12 & 16
  + –

    n\_embd: 768 & 2048
  + –

    Vocab size: tokenizer.vocab\_size from GPT2TokenizerFast (50257)
  + –

    Token embedding: nn.Embedding(vocab\_size, 768)
  + –

    Positional embedding: nn.Embedding(block\_size, 768)
  + –

    Each of the 12 transformer blocks

    - \*

      LayerNorm →\rightarrow multi-head causal self-attention (QKV + output projection) →\rightarrow residual
    - \*

      LayerNorm →\rightarrow 4×\times-wide MLP (3072 hidden) →\rightarrow residual
  + –

    Final LayerNorm
  + –

    Total n\_params: 124.2M & 1313.63M (1.31B)

##### Hyper-parameters

* •

  MLP on MNIST:

  + –

    model: 784 →\rightarrow 512→\rightarrow 256 →\rightarrow 10
  + –

    learning rate: 0.08; momentum: 0.7
  + –

    256 batch size; 50 epochs, 5 runs
* •

  ResNet-18 on CIFAR-100:

  + –

    model: torchvision.models.resnet18
  + –

    learning rate: 0.08; momentum: 0.7
  + –

    512 batch size; 50 epochs, 5 runs
* •

  WideResNet-28-10 on Tiny-ImageNet:

  + –

    model: WideResNet(depth=28, widen\_factor=10, num\_classes=200, drop\_rate=0.0)
  + –

    learning rate: 0.08; momentum: 0.7
  + –

    128 batch size; 30 epochs, 3 runs
* •

  NanoGPT & GPT-1.3B on FineWeb:

  + –

    model: NanoGPT & GPT-1.3B
  + –

    learning rate: 0.02; momentum: 0.95; batch size: 8
  + –

    10M training tokens, 1M validation tokens
  + –

    8 RTX-3090 GPUs
  + –

    max steps: 6000

### G.1 MLP on MNIST

!(/html/2601.19156/assets/x2.png)

Figure 2: Train losses of MLP on MNIST across wall-clock time and epochs

Table 2: Wall-clock time training performance of MLP (0.5M) on MNIST dataset

|  | Wall-clock time train loss | | | | |
| --- | --- | --- | --- | --- | --- |
| Optimizer | 50 sec | 100 sec | 150 sec | 200 sec | 250 sec |
| SGD-M | 0.550 | 0.525 | 0.517 | 0.514 | 0.513 |
| Muon with SVD | 0.616 | 0.612 | 0.600 | 0.583 | 0.565 |
| Muon (qq=1) | 0.526 | 0.514 | 0.506 | 0.502 | 0.501 |
| Muon (qq=2) | 0.532 | 0.518 | 0.509 | 0.503 | 0.501 |
| Muon (qq=3) | 0.548 | 0.529 | 0.511 | 0.503 | 0.501 |

### G.2 CifarNet on CIFAR-10

!(/html/2601.19156/assets/x3.png)

Figure 3: 
Train losses of
CifarNet on CIFAR-10
across wall-clock time and epochs

Table 3: Wall-clock time training performance of CifarNet (2M) on CIFAR-10

|  | Wall-clock time train loss | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Optimizer | 10 sec | 20 sec | 30 sec | 40 sec | 50 sec | 60 sec |
| SGD-M | 1.366 | 1.157 | 1.084 | 1.045 | 1.021 | 1.010 |
| Muon with SVD | 2.211 | 1.334 | 1.227 | 1.207 | 1.193 | 1.182 |
| Muon (qq=1) | 1.130 | 1.014 | 0.945 | 0.905 | 0.888 | 0.884 |
| Muon (qq=2) | 1.129 | 1.064 | 1.001 | 0.932 | 0.888 | 0.876 |
| Muon (qq=3) | 1.191 | 1.145 | 1.079 | 0.990 | 0.905 | 0.876 |

### G.3 ResNet-18 on CIFAR-100

!(/html/2601.19156/assets/x4.png)

Figure 4: 
Train losses of
ResNet-18 on CIFAR-100
across wall-clock time and epochs

Table 4: Wall-clock time training performance of ResNet-18 (11.2M) on CIFAR-100

|  | Wall-clock time train loss | | | | |
| --- | --- | --- | --- | --- | --- |
| Optimizer | 100 sec | 200 sec | 300 sec | 400 sec | 500 sec |
| SGD-M | 3.096 | 2.612 | 2.324 | 2.157 | 2.072 |
| Muon with SVD | 3.202 | 2.767 | 2.668 | 2.594 | 2.522 |
| Muon (qq=1) | 2.427 | 1.986 | 1.716 | 1.563 | 1.514 |
| Muon (qq=2) | 2.407 | 2.127 | 1.826 | 1.584 | 1.495 |
| Muon (qq=3) | 2.554 | 2.329 | 1.985 | 1.624 | 1.481 |

### G.4 WideResNet-28-10 on Tiny-ImageNet

!(/html/2601.19156/assets/x5.png)

Figure 5: 
Train losses of
WideResNet-28-10 on Tiny-ImageNet
across wall-clock time and epochs

Table 5: Wall-clock time training performance of WideResNet-28-10 (36.6M) on Tiny-ImageNet

|  | Wall-clock time train loss | | | | |
| --- | --- | --- | --- | --- | --- |
| Optimizer | 400 sec | 800 sec | 1200 sec | 1600 sec | 2000 sec |
| SGD-M | 4.694 | 4.299 | 4.045 | 3.914 | 3.857 |
| Muon with SVD | 4.267 | 4.020 | 3.846 | 3.734 | 3.520 |
| Muon (qq=1) | 4.035 | 3.369 | 2.903 | 2.495 | 2.144 |
| Muon (qq=2) | 3.766 | 3.246 | 2.815 | 2.311 | 1.856 |
| Muon (qq=3) | 3.666 | 3.317 | 2.929 | 2.372 | 1.800 |

### G.5 NanoGPT on FineWeb

!(/html/2601.19156/assets/x6.png)

Figure 6: 
Train losses of
NanoGPT on FineWeb
across wall-clock time and epochs

Table 6: Wall-clock time training performance of NanoGPT (124M) on FineWeb

|  | Wall-clock time train loss | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Optimizer | 100 sec | 200 sec | 300 sec | 400 sec | 500 sec | 600 sec |
| SGD-M | 10.938 | 7.656 | 6.000 | 5.375 | 4.938 | 4.594 |
| Muon with SVD | 10.250 | 9.188 | 8.625 | 8.062 | 7.594 | 7.125 |
| Muon (qq=1) | 5.969 | 4.094 | 2.469 | 1.367 | 0.879 | 0.637 |
| Muon (qq=2) | 5.000 | 2.344 | 1.047 | 0.395 | 0.157 | 0.030 |
| Muon (qq=3) | 3.984 | 2.125 | 1.180 | 0.523 | 0.142 | 0.026 |

### G.6 GPT-2 based model (1.3B) on FineWeb

!(/html/2601.19156/assets/x7.png)

Figure 7: 
Train losses of
GPT-2 based model (1.3B) on FineWeb
across wall-clock time and epochs

Table 7: Wall-clock time training performance of GPT-2 based model (1.3B) on FineWeb

|  | Wall-clock time train loss | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Optimizer | 600 sec | 1200 sec | 1800 sec | 2400 sec | 3000 sec | 3600 sec |
| SGD-M | 6.4062 | 5.3750 | 4.2812 | 3.2969 | 2.4219 | 2.0156 |
| Muon with SVD | 8.2280 | 8.0810 | 7.9340 | 7.7870 | 7.6400 | 7.4930 |
| Muon (qq=1) | 6.0625 | 3.5156 | 1.0781 | 0.1436 | 0.1338 | 0.0109 |
| Muon (qq=2) | 5.1562 | 2.8906 | 0.5898 | 0.1133 | 0.0349 | 0.0113 |
| Muon (qq=3) | 5.6562 | 3.2031 | 1.2188 | 0.4609 | 0.1201 | 0.0286 |

## Appendix H Additional Ablation Experiments

### H.1 Newton–Schulz–polynomial degree-κ\kappa ablations.

!(/html/2601.19156/assets/x8.png)

Figure 8: Degree κ\kappa sweep.
Newton–Schulz polynomial degree κ∈{1,…,5}\kappa\in\{1,\ldots,5\} at fixed q=3q=3.
Larger κ\kappa improves train/test loss but increases time per step.

Beyond the step‑sweep in Fig. [1](#S5.F1 "Figure 1 ‣ 5 Numerical Experiments ‣ Convergence of Muon with Newton–Schulz"),
we perform a controlled ablation that varies the degree κ\kappa of the Newton–Schulz polynomial while fixing the number of Newton–Schulz steps to q=3q=3 for all variants.
Concretely, we instantiate a family of Muon with degree-κ\kappa polynomial optimizers whose orthogonalization step applies, per layer, the update

|  |  |  |
| --- | --- | --- |
|  | X←pκ​(X​X⊤)​X,X=M‖M‖F\displaystyle X\leftarrow p\_{\kappa}(XX^{\top})X,\quad X=\frac{M}{\|M\|\_{F}} |  |

where MM is the momentum matrix,
and pκ​(λ)=∑m=0κcm​λmp\_{\kappa}(\lambda)=\sum\_{m=0}^{\kappa}c\_{m}\lambda^{m} is the degree‑κ\kappa Newton–Schulz polynomial that matches the derivatives of λ−1/2\lambda^{-1/2} at λ=1\lambda=1 up to order κ\kappa.
The coefficients are generated analytically from the Taylor series of λ−1/2\lambda^{-1/2} at 11,
ensuring the residual recursion δj+1≤(δj)κ+1\delta\_{j+1}\leq(\delta\_{j})^{\kappa+1}
(Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).
Tall matrices are handled using the transpose trick.

Ablation on degree κ\kappa.
Fixing q=3q=3,
increasing the degree κ∈{1,…,5}\kappa\in\{1,\dots,5\} improves optimization
(loss drops faster at a fixed epoch)
but lengthens each step,
yielding a clear accuracy–time trade-off
(see Fig. [8](#A8.F8 "Figure 8 ‣ H.1 Newton–Schulz–polynomial degree-𝜅 ablations. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz")).
(This mirrors the theory that
the residual contracts as δj+1≤δjκ+1\delta\_{j+1}\leq\delta\_{j}^{\kappa+1} (Lemma [3](#Thmlemma3 "Lemma 3 (Residual decay by Newton–Schulz polynomial). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")),
while computation scales with polynomial evaluations.)

Compared optimizers.
We evaluate κ∈{1,2,3,4,5}\kappa\in\{1,2,3,4,5\} under a fixed iteration budget q=3q=3.
This ablation isolates the effect of the polynomial degree κ\kappa at a fixed step qq,
directly testing the theory-driven prediction that larger κ\kappa accelerates orthogonality residual decay
(see Table [1](#S4.T1 "Table 1 ‣ 4.3 Proof Sketch for Theorem 1. ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz")).

### H.2 Rank–dependence.

!(/html/2601.19156/assets/x9.png)

Figure 9: Rank dependence.
Epoch‑averaged gradient norm vs. rank rr
(left: raw log–log; right: normalized by r\sqrt{r}).
Muon variants are nearly rr‑invariant; SGD-M scales up with rank.

Table 8: log–log slopes (ω\omega) from Fig. [9](#A8.F9 "Figure 9 ‣ H.2 Rank–dependence. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz").

| Method | ω\omega (raw) | ω\omega (normalized by  r\sqrt{r} ) |
| --- | --- | --- |
| SGD-M | 0.2920.292 | −0.208-0.208 |
| Muon (SVD) | 0.1020.102 | −0.398-0.398 |
| Muon (NS) | −0.106-0.106 | −0.606-0.606 |

Rank dependence.
We vary the monitored layer’s effective rank r∈{16,32,64,128,216}r\in\{16,32,64,128,216\} and plot the epoch-averaged ‖∇f​(W)‖∗\|\nabla f(W)\|\_{\*} on a log–log scale.
SGD-M shows a positive slope ≈0.3\approx 0.3 (grows with rr),
whereas Muon and its variants are nearly flat.
After normalizing by r\sqrt{r}, the two Muon variants follow the predicted r−1/2r^{-1/2} trend, while SGD-M remains non-flat (see Fig. [9](#A8.F9 "Figure 9 ‣ H.2 Rank–dependence. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz")).

Goal and prediction.
We test how the nuclear–norm of the per–step gradient on a monitored layer scales with the layer’s dimension parameter
r:=min⁡{m,n}r:=\min\{m,n\} for a weight W∈ℝm×nW\in\mathbb{R}^{m\times n}.
The theory predicts that orthogonalizing the momentum
suppresses the rr‑growth which occurs in either Muon with Newton–Schulz or Muon with SVD\mathrm{SVD}, while SGD-M has r\sqrt{r} dependence.

Controlling rr.
We vary the out‑channels of the first convolution in the first ConvGroup and keep all other widths fixed.
For this layer, the weight tensor has shape
[𝚘𝚞𝚝,𝚒𝚗, 3,3][{\tt out},\,{\tt in},\,3,3] with 𝚒𝚗=24{\tt in}=24
(from the fixed "whitening" 2×22\times 2 conv).
Flattening by rows yields a matrix of shape m×nm\times n with
m=𝚘𝚞𝚝m={\tt out} and n=𝚒𝚗⋅3⋅3=24⋅9=216n={\tt in}\cdot 3\cdot 3=24\cdot 9=216,
so r=min⁡{𝚘𝚞𝚝,216}r=\min\{{\tt out},216\}.
We sweep 𝚘𝚞𝚝∈{16,32,64,128,216}{\tt out}\in\{16,32,64,128,216\},
hence r∈{16,32,64,128,216}r\in\{16,32,64,128,216\}.

Optimizers under comparison.
We compare:

* •

  SGD-M (baseline),
* •

  Muon (SVD\mathrm{SVD}) (exact polar U​V⊤UV^{\top}), and
* •

  Muon (NS\mathrm{NS}) with q=3q=3 Newton–Schulz steps.

All methods share identical schedules and auxiliary updates (Appendix [F.1](#A6.SS1 "F.1 Experimental Setting ‣ Appendix F Numerical Experiments Detail ‣ Convergence of Muon with Newton–Schulz")).

Results.
Across r∈{16,32,64,128,216}r\in\{16,32,64,128,216\},
SGD-M exhibits a positive log–log slope (about 0.30.3),
while both Muon variants are nearly flat.
After dividing by r\sqrt{r}, the Muon curves show a slope of about −0.5-0.5
(Muon with SVD\mathrm{SVD}: -0.4 and Muon with Newton–Schulz: -0.61)
, as predicted, whereas SGD-M becomes almost flat (slope with −0.21-0.21).

### H.3 Batch size BB ablations.

We sweep B∈{64,128,256,512,1024}B\in\{64,128,256,512,1024\} with Muon (q=3q=3)
under identical schedules and report epoch–aligned and time–aligned views (Fig. [10](#A8.F10 "Figure 10 ‣ H.3 Batch size 𝐵 ablations. ‣ Appendix H Additional Ablation Experiments ‣ Convergence of Muon with Newton–Schulz")).
The schedule is step–based, so larger BB implies fewer total steps over EE epochs.
We report both epoch–aligned and wall–clock-aligned curves.

From a systems perspective,
increasing BB improves throughput up to a regime of diminishing returns.
From an optimization perspective, a larger BB reduces gradient noise but also decreases the frequency of orthogonalization steps per epoch
(Nproj=Ntrain/BN\_{\mathrm{proj}}=N\_{\text{train}}/B).

In our runs, the best time–to-accuracy is achieved for a large batch size B=1024B=1024,
while a small BB suffers from noise, taking a greater amount of time.

!(/html/2601.19156/assets/x10.png)

Figure 10: Batch size.
Train/test loss vs. epoch and wall‑clock for B∈{64,128,256,512,1024}B\in\{64,128,256,512,1024\}.

### H.4 Degree-2 NS polynomial vs. Ad-hoc degree-2 NS polynomial

!(/html/2601.19156/assets/x11.png)

Figure 11: NS polynomial vs. Ad-hoc polynomial.
Train/test loss vs. epoch and wall‑clock.

Analysis of pad-hocp\_{\text{ad-hoc}}

Let pad-hoc​(λ)=3.4445−4.7750​λ+2.0315​λ2p\_{\text{ad-hoc}}(\lambda)=3.4445-4.7750\,\lambda+2.0315\,\lambda^{2} and τ​(λ)=λ​p​(λ)2\tau(\lambda)=\lambda p(\lambda)^{2}.

The statement that τad-hoc​(λ)\tau\_{\text{ad-hoc}}(\lambda) is monotone non-decreasing on [0,1][0,1] is false, and the underlying condition that pad-hoc​(λ)∈[0,1]p\_{\text{ad-hoc}}(\lambda)\in[0,1] is also not met.

On the interval [0,1][0,1], the range of p​(λ)p(\lambda) is [0.701,3.4445][0.701,3.4445].
Since this range is not contained within [0,1][0,1],
the premise p​(λ)∈[0,1]p(\lambda)\in[0,1] is false.

A function is monotone non-decreasing if its derivative is greater than or equal to zero over the entire interval.
The derivative of τ​(λ)\tau(\lambda) is:

|  |  |  |
| --- | --- | --- |
|  | τad-hoc′​(λ)=dd​λ​τad-hoc​(λ)=20.635​λ4−77.5824​λ3+110.3556​λ2−65.781​λ+11.8641\displaystyle\tau\_{\text{ad-hoc}}^{\prime}(\lambda)=\frac{d}{d\lambda}\tau\_{\text{ad-hoc}}(\lambda)=20.635\lambda^{4}-77.5824\lambda^{3}+110.3556\lambda^{2}-65.781\lambda+11.8641 |  |

To check for monotonicity,
we can evaluate the derivative at the endpoints of the interval [0,1][0,1]:

* •

  At λ=0\lambda=0:
  τad-hoc′​(0)=11.8641\tau\_{\text{ad-hoc}}^{\prime}(0)=11.8641
* •

  At λ=1\lambda=1:
  τad-hoc′​(1)=20.635−77.5824+110.3556−65.781+11.8641=−0.5087\tau\_{\text{ad-hoc}}^{\prime}(1)=20.635-77.5824+110.3556-65.781+11.8641=-0.5087

Since τad-hoc′​(0)>0\tau\_{\text{ad-hoc}}^{\prime}(0)>0 and τad-hoc′​(1)<0\tau\_{\text{ad-hoc}}^{\prime}(1)<0,
the derivative changes from positive to negative within the interval.
This means the function τad-hoc​(λ)\tau\_{\text{ad-hoc}}(\lambda) increases for a portion of the interval and then decreases.

Therefore, the claim that τad-hoc​(λ)\tau\_{\text{ad-hoc}}(\lambda) is monotone non-decreasing on [0,1][0,1] is false.
The function has a local maximum at approximately λ≈0.308\lambda\approx 0.308.

Hence, the monotonicity premise of Lemma [2](#Thmlemma2 "Lemma 2 (Residual update). ‣ 4.4 Proof Sketch for Theorem 2 ‣ 4 Main Results ‣ Convergence of Muon with Newton–Schulz") fails for pad-hocp\_{\text{ad-hoc}},
even though τ​(1)=p​(1)2=0.7012<1\tau(1)=p(1)^{2}=0.701^{2}<1 holds.
This explains why our guarantees apply to pκp\_{\kappa} but not to the ad-hoc quadratic,
which remains an empirical heuristic.
