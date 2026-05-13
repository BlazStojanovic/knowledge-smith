---
arxiv: '2604.01472'
authors:
- Zhehang Du
- Weijie Su
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: The Newton-Muon Optimizer
url: https://arxiv.org/abs/2604.01472
year: 2026
---

[2604.01472] The Newton–Muon Optimizer















function detectColorScheme(){
var theme="light";
var current\_theme = localStorage.getItem("ar5iv\_theme");
if(current\_theme){
if(current\_theme == "dark"){
theme = "dark";
} }
else if(!window.matchMedia) { return false; }
else if(window.matchMedia("(prefers-color-scheme: dark)").matches) {
theme = "dark"; }
if (theme=="dark") {
document.documentElement.setAttribute("data-theme", "dark");
} else {
document.documentElement.setAttribute("data-theme", "light"); } }
detectColorScheme();
function toggleColorScheme(){
var current\_theme = localStorage.getItem("ar5iv\_theme");
if (current\_theme) {
if (current\_theme == "light") {
localStorage.setItem("ar5iv\_theme", "dark"); }
else {
localStorage.setItem("ar5iv\_theme", "light"); } }
else {
localStorage.setItem("ar5iv\_theme", "dark"); }
detectColorScheme(); }



# The Newton–Muon Optimizer

Zhehang Du
  
Weijie Su

(University of Pennsylvania
  
)

###### Abstract

The Muon optimizer has received considerable attention for its strong performance in training large language models, yet the design principle behind its matrix-gradient orthogonalization remains largely elusive. In this paper, we introduce a surrogate model that not only sheds new light on the design of Muon, but more importantly leads to a new optimizer. In the same spirit as the derivation of Newton’s method, the surrogate approximates the loss as a quadratic function of the perturbation to a weight matrix WW using only three matrices: the gradient GG, an output-space curvature matrix HH, and the data matrix ZZ that stacks the layer inputs. By minimizing this surrogate in one step and adopting a certain isotropic assumption on the weights, we obtain the closed-form update rule (up to momentum and weight decay)

|  |  |  |
| --- | --- | --- |
|  | W←W−η⋅msgn​(G​(Z​Z⊤)−1),W\leftarrow W-\eta\cdot\mathrm{msgn}(G(ZZ^{\top})^{-1}), |  |

where η\eta is the learning rate and msgn​(X)=U​V⊤\mathrm{msgn}(X)=UV^{\top} if X=U​S​V⊤X=USV^{\top} is a compact singular value decomposition. This new optimization method, which we refer to as Newton–Muon, shows that standard Muon can be interpreted as an implicit Newton-type method that neglects the right preconditioning induced by the input second moment. Empirically, on a reproduction of the earliest publicly released Modded-NanoGPT speedrun configuration using Muon for GPT-2 pretraining, Newton–Muon reaches the target validation loss in 6% fewer iteration steps and reduces wall-clock training time by about 4%.

\NoHyper††footnotetext: Emails: {duz,suw}@wharton.upenn.edu.\endNoHyper\NoHyper††footnotetext: Code is available at <https://github.com/zhehangdu/Newton-Muon>.\endNoHyper

## 1 Introduction

Since early 2025, there has been a surge of interest in matrix-structured optimization methods for training deep neural networks and large language models (LLMs). A prominent optimizer at the center of this flurry of research activity is Muon (jordan2024muon), closely related to spectral gradient descent (carlson2015preconditioned). Specifically, letting f​(W)f({W}) be the loss function and W{W} be a layer weight matrix, we consider the optimization problem

|  |  |  |
| --- | --- | --- |
|  | minW∈ℝm×n⁡f​(W).\min\_{W\in\mathbb{R}^{m\times n}}f(W). |  |

Let G≔∇Wf​(W)∈ℝm×n{G}\coloneqq\nabla\_{{W}}f({W})\in\mathbb{R}^{m\times n} be the gradient matrix or, in practice, an approximation computed from a mini-batch. Writing the matrix sign msgn​(G)≔U​V⊤\mathrm{msgn}({G})\coloneqq{U}{V}^{\top} if G=U​S​V⊤{G}={U}{S}{V}^{\top} is a compact singular value decomposition (SVD), Muon updates the weight matrix as111In practice, Muon approximates msgn​(G)\mathrm{msgn}({G}) using Newton–Schulz iterations and momentum is applied to the gradient.

|  |  |  |
| --- | --- | --- |
|  | W←W−η⋅msgn​(G)W\leftarrow W-\eta\cdot\mathrm{msgn}({G}) |  |

for some learning rate η\eta. Compared to AdamW (kingma2014adam; loshchilov2017decoupled), Muon has been reported to offer a faster convergence rate and lower wall-clock time to reach the same level of loss across a broad range of model sizes (liu2025muon; essentialai2025practical; wen2025fantastic). Furthermore, it has been used to train state-of-the-art open-source models (kimiteam2026kimik25visualagentic; zeng2026glm), and many extensions of Muon have been proposed (li2025normuon; pethick2025training; ahn2025dion2; he2025low; xu2026fismo; qi2026delving; gu2026mano).

While matrix-based optimizers have demonstrated highly effective empirical performance, the theoretical mechanisms underlying Muon remain largely mysterious. For instance, it is natural to ask why preserving the matrix structure of the gradient is beneficial and why simply discarding its singular values is empirically effective. This is, however, not entirely surprising, as the nonconvex nature of deep learning optimization makes it notoriously difficult to analyze. Given that a rigorous theoretical understanding remains out of reach, a practical approach is to establish an intuitive yet principled framework for designing deep learning optimizers (bernstein2024old; pethick2025training; lau2025polargrad; gong2026aro). Among these, su2025isotropic introduced the isotropic curvature model as a surrogate for approximating the loss function by assuming isotropic curvature for preconditioning and isotropic input activations. A one-step descent analysis applied to this model suggests that the optimal update direction naturally preserves the matrix subspace of the gradient, thereby partially justifying gradient orthogonalization.

However, the unspecified curvature function in the isotropic curvature model limits its utility for deriving practical optimization methods for LLM training. In this paper, we address this challenge by introducing a more tractable surrogate model for understanding these optimizers and, more importantly, for proposing a new method. Taking Q∈ℝm×nQ\in\mathbb{R}^{m\times n} to be a potential update direction and letting HH denote a curvature matrix,222Here, the curvature matrix HH is not the full parameter-space Hessian with respect to the vectorized weights. In particular, the curvature matrix HH has size m×mm\times m. this paper introduces a surrogate model of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(W−Q)−f​(W)≈−tr​(Q​G⊤)+12​N​tr​(H​Q​(Z​Z⊤)​Q⊤),f(W-Q)-f(W)\approx-\mathrm{tr}({Q}{G}^{\top})+\frac{1}{2N}\mathrm{tr}\Big({H}{Q}({Z}{Z}^{\top}){Q}^{\top}\Big), |  | (1) |

where Z=[𝒛1,…,𝒛N]Z=[\boldsymbol{z}\_{1},\ldots,\boldsymbol{z}\_{N}] denotes the collection of all activation inputs from the NN training data points to the layer WW. Because it involves three components—namely GG, HH, and ZZ—this model is referred to as the triplet quadratic surrogate model. By design, the gradient matrix GG and the update direction QQ naturally retain their matrix forms. The linear term −tr​(Q​G⊤)-\mathrm{tr}({Q}{G}^{\top}) captures the first-order approximation, as in the isotropic curvature model, while the quadratic term 12​N​tr​(H​Q​(Z​Z⊤)​Q⊤)\frac{1}{2N}\mathrm{tr}\Big({H}{Q}({Z}{Z}^{\top}){Q}^{\top}\Big) is considerably simpler than the curvature function in the isotropic curvature model. Notably, this quadratic term is the matrix-form expression induced by the Kronecker-factored curvature approximation in K-FAC (martens2015optimizing).

This triplet quadratic surrogate for approximating f​(W−Q)−f​(W)f(W-Q)-f(W) is essentially of the form “−linear gradient term+quadratic curvature term-\textit{linear gradient term}+\textit{quadratic curvature term}”, which closely resembles the derivation of Newton’s method. As such, this surrogate model serves as a means of deriving a Newton-type method in a manner that fully leverages the matrix structure of the gradient via a one-step descent analysis; that is, by minimizing the right-hand side of ([1](#S1.E1 "In 1 Introduction ‣ The Newton–Muon Optimizer")) over QQ. In fact, although second-order methods such as Newton’s method are rarely used in training large-scale neural networks, recent work demonstrates a 33–5×5\times iteration speedup when using the Gauss–Newton method compared to common optimizers in deep learning (abreu2025potential). Thus, there is strong motivation to develop an implicit Newton-type method for LLM training, provided that the per-iteration computational cost remains roughly comparable to that of AdamW or Muon.

Despite the ease of minimizing the triplet quadratic surrogate model, a key difficulty arises because the optimal solution involves the unknown curvature matrix HH. To circumvent this, we make an assumption on the displacement W−W⋆W-W^{\star}, which is the difference between the current weight matrix WW and an optimal weight matrix W⋆W^{\star}.333Since there are generally exponentially many optimal weight matrices, we simply consider one that is locally closest to WW. We assume that this displacement is isotropic, in the sense that no direction is favored over another on average. Surprisingly, under this least-informative assumption, we obtain a closed-form expression for the Newton-type update: Q⋆∝msgn​(G​(Z​Z⊤)−1)Q^{\star}\propto\mathrm{msgn}(G(ZZ^{\top})^{-1}). Accordingly, our new method updates the weight matrix according to the rule444In practice, momentum and other common implementation tricks are also used.

|  |  |  |
| --- | --- | --- |
|  | W←W−η⋅msgn​(G​(Z​Z⊤)−1).W\leftarrow W-\eta\cdot\mathrm{msgn}(G(ZZ^{\top})^{-1}). |  |

This method differs from standard Muon in that the gradient matrix GG is right-preconditioned by the inverse second moment of the input data. This can be interpreted as incorporating the geometry of the data distribution to yield an update direction distinct from that of Muon. To implement Newton–Muon efficiently in practice, we maintain a running estimate of the second moment matrix Z​Z⊤ZZ^{\top} and recompute both this estimate and its inverse only periodically, rather than at every optimization step. Between refreshes, the cached inverse is reused. We compute the damped inverse (Z​Z⊤+γ​In)−1(ZZ^{\top}+\gamma I\_{n})^{-1} via a Cholesky factorization followed by triangular solves. Even for very large matrices, this computational cost is no more than about 10×10\times that of a single matrix multiplication on a modern GPU. This step is then followed by a Newton–Schulz polynomial approximation to the matrix sign, such as the Polar Express method (amsel2025polar) or Gram Newton–Schulz (GramNewtonSchulz).

Because this optimization method is derived in the same spirit as Newton’s method, we call it Newton–Muon. Standard Muon can be recovered as a special case when the input data’s second moment is approximately isotropic, that is, Z​Z⊤∝InZZ^{\top}\propto I\_{n}. Put differently, Muon can be viewed as an implicit Newton-type method that does not account for the geometry of the input data. However, this simplification is not consistent with our observation that Z​Z⊤ZZ^{\top} is highly anisotropic in practice.

The right-preconditioning by the input data distribution in the update rule endows Newton–Muon with its empirical advantage over standard Muon. Indeed, compared to our reproduction of the earliest publicly logged Modded-NanoGPT speedrun (modded\_nanogpt\_2024) configuration using Muon, Newton–Muon achieves a notable 6%6\% reduction in the number of iterations required to reach the target validation loss, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ The Newton–Muon Optimizer"). In terms of wall-clock time, replacing standard Muon with Newton–Muon yields a reduction of about 4%4\%.

![Refer to caption](/html/2604.01472/assets/x1.png)

![Refer to caption](/html/2604.01472/assets/x2.png)

Figure 1: Top: short track Record #4 validation loss comparison on the Modded-NanoGPT speedrun benchmark. Record #4 is the earliest publicly released configuration using Muon, and our reproduction on a single H100 GPU is denoted Muon. Newton–Muon adds the activation right-preconditioner before the Newton–Schulz iterations. Newton–Muon reaches the Muon baseline final validation loss in 6%6\% fewer steps; despite a 1.8%1.8\% higher per-step cost from right-preconditioning, it reduces wall-clock time to that loss by about 4%4\%. Bottom: CIFAR-10 experiments (Appendix [C](#A3 "Appendix C CIFAR-10 Experiment ‣ The Newton–Muon Optimizer")) on a 32-layer residual MLP show that Newton–Muon outperforms both Muon and AdamW in both per-step efficiency and overall wall-clock time.

### 1.1 Structure of the Paper

Section [2](#S2 "2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer") introduces the triplet quadratic surrogate model and derives the Newton–Muon update. Section [3](#S3 "3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer") analyzes Newton–Muon on a simple quadratic case study. Section [4](#S4 "4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer") develops a one-step analysis with numerical experiments under spiked activation. Section [5](#S5 "5 LLM Experiments ‣ The Newton–Muon Optimizer") presents our experiments on LLMs. Section [6](#S6 "6 Discussion ‣ The Newton–Muon Optimizer") discusses limitations and open directions. Appendix [A](#A1 "Appendix A LLM Experimental Details ‣ The Newton–Muon Optimizer") provides the LLM training configurations. Appendix [B](#A2 "Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer") details efficient computation of the activation second moment inverse. Appendix [C](#A3 "Appendix C CIFAR-10 Experiment ‣ The Newton–Muon Optimizer") provides the CIFAR-10 training configurations. Appendix [D](#A4 "Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer") derives the Kronecker-factored curvature. Appendix [E](#A5 "Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer") gives quadratic score formulas under isotropic activation. Appendix [F](#A6 "Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer") discusses the non-isotropic assumption.

### 1.2 Related Work

#### Matrix-based optimizers.

Existing matrix-based optimizers can be grouped into several broad directions. One line uses matrix structure for preconditioning, curvature approximation, or adaptive updates, including spectral descent (carlson2015preconditioned), K-FAC (martens2015optimizing; george2018fast), Shampoo (gupta2018shampoo; morwani2024new), and SOAP (vyas2024soap), which performs Adafactor-style (shazeer2018adafactor) updates. Another line improves training efficiency through approximations, for example by combining preconditioning with variance reduction, as in MARS (yuan2024mars), or by using low-rank gradient projections, as in GaLore (zhao2024galore; su2025galore), or by accelerating matrix-function evaluation, as in PRISM (yang2026prism). More recent work studies matrix geometry itself as the design principle, including LMO-based methods such as Scion (pethick2025training), cheaper Muon variants such as Dion2 (ahn2025dion2), adaptive choices of learning rate such as PolarGrad (lau2025polargrad), and rotated coordinate updates such as ARO (gong2026aro).

#### Understanding Muon.

A common view is that Muon implements normalized steepest descent under the spectral norm (bernstein2024old; crawshaw2025exploration; riabinin2025gluon), which is further generalized by Lion-𝒦\mathcal{K} (chen2025muon). We instead connect Muon to the classical Euclidean Newton step under a local quadratic surrogate, providing a complementary explanation for its strong performance. Related theory studies Muon’s implicit bias and geometric interpretation (fan2025implicit; pethick2025training), including simplicity bias (dragutinovic2026use), separates curvature and gradient anisotropy (lau2025polargrad), and analyzes convergence in stochastic nonconvex settings and structured models (li2025muon; sato2025analysis; ma2026preconditioning), including under heavy-tailed gradient noise (yu2026sign). Beyond asymptotic results, local one-step analysis has also been studied (su2025isotropic; davis2025spectral; gonon2026insights).

#### Kronecker-factored curvature.

Layerwise curvature factorizations and matrix preconditioners have been widely studied, notably in K-FAC (martens2015optimizing) and Shampoo (gupta2018shampoo). We start from the same structural observation: for a given layer, the parameter-space curvature of a local quadratic model can be approximated by the Kronecker product (Z​Z⊤/N)⊗H({Z}{Z}^{\top}/N)\otimes{H}, where Z​Z⊤/N{Z}{Z}^{\top}/N is the activation second moment and H{H} is an output-space curvature factor. K-FAC explicitly estimates H{H} as a generalized Gauss–Newton/Fisher factor from gradient second moments and then applies the two-sided preconditioner H−1​G​(Z​Z⊤)−1{H}^{-1}{G}({Z}{Z}^{\top})^{-1}. In contrast, we show that Muon’s matrix sign can be interpreted as an implicit left preconditioner that approximates the effect of H−1{H}^{-1}.

## 2 Derivation of Newton–Muon

In this section, we first propose an amenable surrogate function for approximating the objective function we wish to minimize, followed by the derivation of the Newton–Muon method via a one-step minimization of this surrogate. We then draw a connection between Newton’s method and standard Muon to provide a principled interpretation of the latter.

### 2.1 Triplet Quadratic Surrogate

Starting from the current iterate, we estimate the loss change induced by a candidate update direction using a local quadratic model, in the same spirit as the derivation of Newton’s method.

Assume the loss function f:ℝm×n→ℝf:\mathbb{R}^{m\times n}\to\mathbb{R} is twice continuously differentiable, and let G≔∇Wf​(W)∈ℝm×n{G}\coloneqq\nabla\_{{W}}f({W})\in\mathbb{R}^{m\times n} be the gradient matrix at the current iterate W∈ℝm×n{W}\in\mathbb{R}^{m\times n}. For each sample ii, let 𝒛i∈ℝn\boldsymbol{z}\_{i}\in\mathbb{R}^{n} denote the layer input, which is the activated output from the previous layer, and write Z=[𝒛1,…,𝒛N]∈ℝn×N{Z}=[\boldsymbol{z}\_{1},\dots,\boldsymbol{z}\_{N}]\in\mathbb{R}^{n\times N}. For a candidate update direction Q∈ℝm×n{Q}\in\mathbb{R}^{m\times n}, the layer output for sample ii changes by −Q​𝒛i∈ℝm-{Q}\boldsymbol{z}\_{i}\in\mathbb{R}^{m}. To motivate the quadratic term, we write the averaged loss as f​(W)=(1/N)​∑i=1NLi​(W​𝒛i)f({W})=(1/N)\sum\_{i=1}^{N}L\_{i}({W}\boldsymbol{z}\_{i}). Under the perturbation W↦W−Q{W}\mapsto{W}-{Q}, the ii-th output changes from W​𝒛i{W}\boldsymbol{z}\_{i} to W​𝒛i−Q​𝒛i{W}\boldsymbol{z}\_{i}-{Q}\boldsymbol{z}\_{i}. By the integral remainder form of Taylor’s theorem,

|  |  |  |
| --- | --- | --- |
|  | Li​(W​𝒛i−Q​𝒛i)=Li​(W​𝒛i)−∇Li​(W​𝒛i)⊤​(Q​𝒛i)+h​(Q​𝒛i,W​𝒛i),L\_{i}({W}\boldsymbol{z}\_{i}-{Q}\boldsymbol{z}\_{i})=L\_{i}({W}\boldsymbol{z}\_{i})-\nabla L\_{i}({W}\boldsymbol{z}\_{i})^{\top}({Q}\boldsymbol{z}\_{i})+h({Q}\boldsymbol{z}\_{i},{W}\boldsymbol{z}\_{i}), |  |

where

|  |  |  |
| --- | --- | --- |
|  | h​(Q​𝒛i,W​𝒛i)≔(Q​𝒛i)⊤​[∫01(1−t)​∇2Li​(W​𝒛i−t​Q​𝒛i)​dt]​(Q​𝒛i).h({Q}\boldsymbol{z}\_{i},{W}\boldsymbol{z}\_{i})\coloneqq({Q}\boldsymbol{z}\_{i})^{\top}\left[\int\_{0}^{1}(1-t)\nabla^{2}L\_{i}({W}\boldsymbol{z}\_{i}-t{Q}\boldsymbol{z}\_{i})\mathrm{d}t\right]({Q}\boldsymbol{z}\_{i}). |  |

We then approximate this path-dependent integral ∫01(1−t)​∇2Li​(W​𝒛i−t​Q​𝒛i)​dt≈(1/2)​H\int\_{0}^{1}(1-t)\nabla^{2}L\_{i}({W}\boldsymbol{z}\_{i}-t{Q}\boldsymbol{z}\_{i})\mathrm{d}t\approx(1/2)H by a fixed average curvature matrix HH shared across samples, so that the contribution of sample ii to the quadratic term is approximated by 1/(2​N)​(Q​𝒛i)⊤​H​(Q​𝒛i)1/(2N)({Q}\boldsymbol{z}\_{i})^{\top}{H}({Q}\boldsymbol{z}\_{i}). Summing over the NN samples gives the surrogate objective

|  |  |  |  |
| --- | --- | --- | --- |
|  | minQ∈ℝm×n⁡J​(Q)=−tr​(Q​G⊤)+12​N​∑i=1N(Q​𝒛i)⊤​H​(Q​𝒛i)≔−tr​(Q​G⊤)+12​N​tr​(H​Q​Z​Z⊤​Q⊤).\min\_{{Q}\in\mathbb{R}^{m\times n}}J({Q})=-\mathrm{tr}({Q}{G}^{\top})+\frac{1}{2N}\sum\_{i=1}^{N}({Q}\boldsymbol{z}\_{i})^{\top}{H}({Q}\boldsymbol{z}\_{i})\coloneqq-\mathrm{tr}({Q}{G}^{\top})+\frac{1}{2N}\mathrm{tr}\Big(HQZZ^{\top}Q^{\top}\Big). |  | (2) |

This objective is closely related to the isotropic curvature model (su2025isotropic):

|  |  |  |
| --- | --- | --- |
|  | minQ∈ℝm×n−tr​(Q​G⊤)+𝔼ζ​h​(‖Q​ζ‖),\min\_{{Q}\in\mathbb{R}^{m\times n}}-\mathrm{tr}(QG^{\top})+\mathbb{E}\_{\zeta}h(\|Q\zeta\|), |  |

where ζ\zeta is sampled uniformly from the unit sphere and hh is a univariate curvature function. The linear term is the same as in ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")), but the higher-order term is modeled isotropically through a radial curvature function hh, assumed to have super-quadratic growth. In contrast, our triplet surrogate replaces the isotropic sampling of ζ\zeta with the empirical distribution from the columns of ZZ, removes the isotropy assumption on the curvature, and explicitly captures the interaction among HH, ZZ, and GG in ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")).

### 2.2 Minimization of the Triplet Model

Minimizing ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) over QQ is straightforward, but it becomes practically useful only if we can establish a relationship between the curvature matrix HH and the gradient matrix GG. To this end, we first establish a connection between ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) and the full parameter-space second-order expansion. Define the parameter-space Hessian
ℋW≔∇vec​(W)2f​(W)∈ℝ(m​n)×(m​n){\mathcal{H}}\_{{W}}\coloneqq\nabla^{2}\_{\mathrm{vec}({W})}f({W})\in\mathbb{R}^{(mn)\times(mn)},
where vec​(W)∈ℝm​n\mathrm{vec}({W})\in\mathbb{R}^{mn} stacks W{W} into a vector using a standard convention. The second-order Taylor expansion around vec​(W)\mathrm{vec}({W}) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(W−Q)≈f​(W)−tr​(Q​G⊤)+12​vec​(Q)⊤​ℋW​vec​(Q).f({W}-{Q})\approx f({W})-\mathrm{tr}(QG^{\top})+\frac{1}{2}\mathrm{vec}({Q})^{\top}{\mathcal{H}}\_{{W}}\mathrm{vec}({Q}). |  | (3) |

For the quadratic proxy of ([3](#S2.E3 "In 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")), we employ the approximation vec​(G)≈ℋW​vec​(W−W⋆)\mathrm{vec}({G})\approx{\mathcal{H}}\_{{W}}\mathrm{vec}({W}-W^{\star}), where W⋆W^{\star} denotes a nearby local minimizer, or more generally, a nearby reference point at which ∇Wf​(W⋆)\nabla\_{W}f(W^{\star}) is assumed to be negligibly small.

To relate ℋW\mathcal{H}\_{W} to the curvature matrix HH in the surrogate model ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")), we use the Kronecker-factored approximation (see its application in K-FAC (martens2015optimizing) and Appendix [D](#A4 "Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer") for details):

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℋW≈(Z​Z⊤/N)⊗H,{\mathcal{H}}\_{{W}}\approx({Z}{Z}^{\top}/N)\otimes{H}, |  | (4) |

where ⊗\otimes denotes the Kronecker product. We then obtain

|  |  |  |
| --- | --- | --- |
|  | vec​(G)≈((Z​Z⊤/N)⊗H)​vec​(W−W⋆)=vec​(H​(W−W⋆)​(Z​Z⊤/N)).\mathrm{vec}({G})\approx\big(({Z}{Z}^{\top}/N)\otimes{H}\big)\mathrm{vec}({W}-W^{\star})=\mathrm{vec}\Big({H}({W}-W^{\star})({Z}{Z}^{\top}/N)\Big). |  |

This immediately implies

|  |  |  |
| --- | --- | --- |
|  | G≈H​(W−W⋆)​Z​Z⊤/N.G\approx H({W}-W^{\star}){Z}{Z}^{\top}/N. |  |

In light of the above, we make the following assumption.

###### Assumption 1.

We assume G=H​(W−W⋆)​Z​Z⊤/N,H≻0,Z​Z⊤≻0{G}={H}({W}-W^{\star}){Z}{Z}^{\top}/N,\quad{H}\succ 0,\quad{Z}{Z}^{\top}\succ 0.

The last two conditions ensure that the quadratic surrogate in ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) is strictly convex in Q{Q} and hence has a unique minimizer. If the curvature matrix HH is not positive definite but merely non-degenerate, we can instead solve for a stationary point of the surrogate model.

Under this assumption, the optimal solution to the surrogate objective ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) takes a closed-form expression that does not explicitly involve the unknown curvature matrix HH.

###### Proposition 1.

Denote by ΣW≔(W−W⋆)​(W−W⋆)⊤∈ℝm×m{\Sigma}\_{{W}}\coloneqq({W}-W^{\star})({W}-W^{\star})^{\top}\in\mathbb{R}^{m\times m} the displacement second moment matrix and write ΣW1/2{\Sigma}\_{{W}}^{1/2} for its unique positive semidefinite square root. Then under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), the unique minimizer Q⋆{Q}^{\star} of ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) takes the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | Q⋆=ΣW1/2​msgn​(ΣW1/2​G​(Z​Z⊤)−1).{Q}^{\star}={\Sigma}\_{{W}}^{1/2}\mathrm{msgn}\Big({\Sigma}\_{{W}}^{1/2}{G}({Z}{Z}^{\top})^{-1}\Big). |  | (5) |

###### Proof of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer").

Define the compact SVD W−W⋆=UQ​SQ​VQ⊤{W}-W^{\star}={U}\_{Q}{S}\_{Q}{V}\_{Q}^{\top}, where UQ∈ℝm×r{U}\_{Q}\in\mathbb{R}^{m\times r} and VQ∈ℝn×r{V}\_{Q}\in\mathbb{R}^{n\times r} have orthonormal columns, and SQ∈ℝr×r{S}\_{Q}\in\mathbb{R}^{r\times r} is a diagonal matrix with positive entries. Then ΣW=UQ​SQ2​UQ⊤{\Sigma}\_{{W}}={U}\_{Q}{S}\_{Q}^{2}{U}\_{Q}^{\top} and ΣW1/2=UQ​SQ​UQ⊤{\Sigma}\_{{W}}^{1/2}={U}\_{Q}{S}\_{Q}{U}\_{Q}^{\top}. From Assumption [1](#Thmassumption1 "Assumption 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), we have G​(Z​Z⊤)−1=H​(W−W⋆)/N{G}({Z}{Z}^{\top})^{-1}={H}(W-W^{\star})/N, so

|  |  |  |
| --- | --- | --- |
|  | ΣW1/2​G​(Z​Z⊤)−1=1N​ΣW1/2​H​(W−W⋆)=1N​UQ​SQ​(UQ⊤​H​UQ)​SQ​VQ⊤.{\Sigma}\_{{W}}^{1/2}{G}({Z}{Z}^{\top})^{-1}=\frac{1}{N}{\Sigma}\_{{W}}^{1/2}{H}(W-W^{\star})=\frac{1}{N}{U}\_{Q}{S}\_{Q}\big({U}\_{Q}^{\top}{H}{U}\_{Q}\big){S}\_{Q}{V}\_{Q}^{\top}. |  |

Since H≻0{H}\succ 0 and UQ{U}\_{Q} has full column rank, the r×rr\times r matrix UQ⊤​H​UQ{U}\_{Q}^{\top}{H}{U}\_{Q} is symmetric positive definite. Hence, SQ​(UQ⊤​H​UQ)​SQ{S}\_{Q}\big({U}\_{Q}^{\top}{H}{U}\_{Q}\big){S}\_{Q} is also symmetric positive definite. Therefore,

|  |  |  |
| --- | --- | --- |
|  | msgn​(ΣW1/2​G​(Z​Z⊤)−1)=UQ​VQ⊤.\mathrm{msgn}\Big({\Sigma}\_{{W}}^{1/2}{G}({Z}{Z}^{\top})^{-1}\Big)={U}\_{Q}{V}\_{Q}^{\top}. |  |

Multiplying on the left by ΣW1/2{\Sigma}\_{{W}}^{1/2} gives

|  |  |  |
| --- | --- | --- |
|  | ΣW1/2​msgn​(ΣW1/2​G​(Z​Z⊤)−1)=UQ​SQ​UQ⊤​UQ​VQ⊤=UQ​SQ​VQ⊤=W−W⋆.{\Sigma}\_{{W}}^{1/2}\mathrm{msgn}\Big({\Sigma}\_{{W}}^{1/2}{G}({Z}{Z}^{\top})^{-1}\Big)={U}\_{Q}{S}\_{Q}{U}\_{Q}^{\top}{U}\_{Q}{V}\_{Q}^{\top}={U}\_{Q}{S}\_{Q}{V}\_{Q}^{\top}={W}-W^{\star}. |  |

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), the quadratic term in ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) is equivalent to the vectorized quadratic term in ([3](#S2.E3 "In 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")). Thus, the solution Q⋆{Q}^{\star} to ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")), that is, the Newton direction, satisfies vec​(Q⋆)=ℋW−1​vec​(G)=ℋW−1​ℋW​vec​(W−W⋆)=vec​(W−W⋆)\mathrm{vec}({Q}^{\star})={\mathcal{H}}\_{{W}}^{-1}\mathrm{vec}({G})={\mathcal{H}}\_{{W}}^{-1}{\mathcal{H}}\_{{W}}\mathrm{vec}({W}-W^{\star})=\mathrm{vec}({W}-W^{\star}). Hence Q⋆=W−W⋆{Q}^{\star}={W}-W^{\star}, from which ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) follows.
∎

### 2.3 Newton–Muon

The Newton-type update ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) can be readily used as long as the unknown displacement second moment ΣW=(W−W⋆)​(W−W⋆)⊤\Sigma\_{W}=(W-W^{\star})(W-W^{\star})^{\top} can be estimated. While the proof of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer") shows that Q⋆=W−W⋆Q^{\star}=W-W^{\star}, meaning that the right-hand side of ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) inherently depends on Q⋆Q^{\star}, a closer look reveals that ΣW\Sigma\_{W} is a much coarser object to approximate than Q⋆Q^{\star} itself. Specifically, approximating ΣW\Sigma\_{W} does not require knowledge of the right singular spaces of W−W⋆W-W^{\star}.

Perhaps the least-informative approximation is to assume that ΣW\Sigma\_{W} is a multiple of the identity matrix. This isotropic proxy, ΣW∝Im{\Sigma}\_{{W}}\propto{I}\_{m}, is plausible, especially at initialization and during the early stages of training when the columns of W−W⋆W-W^{\star} are approximately independent and no specific directional structure has yet emerged. Since ΣW=(W−W⋆)​(W−W⋆)⊤{\Sigma}\_{{W}}=({W}-W^{\star})({W}-W^{\star})^{\top} aggregates the second moments of the columns of W−W⋆{W}-W^{\star}, applying this isotropic proxy amounts to treating these unknown column directions as having no preferred orientation in aggregate.

This yields a closed-form update direction depending only on observable quantities, as shown in the following result.

###### Theorem 1.

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer") and the isotropic proxy ΣW∝Im{\Sigma}\_{{W}}\propto{I}\_{m}, we have

|  |  |  |
| --- | --- | --- |
|  | Q⋆∝msgn​(G​(Z​Z⊤)−1).{Q}^{\star}\propto\mathrm{msgn}\Big({G}({Z}{Z}^{\top})^{-1}\Big). |  |

Although this is a corollary of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), we choose to present it as a theorem because it formally establishes the core method introduced in this paper. At iteration tt, we update the weight matrix according to the rule:

|  |  |  |
| --- | --- | --- |
|  | Wt+1=Wt−ηt⋅msgn​(Gt​(Zt​Zt⊤)−1),W\_{t+1}=W\_{t}-\eta\_{t}\cdot\mathrm{msgn}\Big(G\_{t}(Z\_{t}Z\_{t}^{\top})^{-1}\Big), |  |

where ηt\eta\_{t} denotes the learning rate. We call this method Newton–Muon because it implicitly acts as a Newton-type method, derived by minimizing a quadratic surrogate. Strictly speaking, when m>nm>n, it mathematically cannot hold that ΣW∝Im{\Sigma}\_{{W}}\propto{I}\_{m}, since ΣW=(W−W⋆)​(W−W⋆)⊤{\Sigma}\_{{W}}=({W}-W^{\star})({W}-W^{\star})^{\top} has rank at most nn. Thus, the relation ΣW∝Im{\Sigma}\_{{W}}\propto{I}\_{m} should be understood purely as an isotropic proxy for the unknown displacement second moment.

The following result shows that the Newton–Muon update is always a descent direction.

###### Proposition 2.

For the Newton–Muon direction Q=msgn​(G​(Z​Z⊤)−1)Q=\mathrm{msgn}\big(G(ZZ^{\top})^{-1}\big), we have tr​(G⊤​Q)≥0\mathrm{tr}(G^{\top}Q)\geq 0.

###### Proof of Proposition [2](#Thmproposition2 "Proposition 2. ‣ 2.3 Newton–Muon ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer").

Let Gr≔G​(Z​Z⊤)−1{G}\_{\mathrm{r}}\coloneqq G(ZZ^{\top})^{-1}, and write its compact SVD as Gr=UG​SG​VG⊤{G}\_{\mathrm{r}}={U}\_{G}{S}\_{G}{V}\_{G}^{\top}. Then Q=msgn​(Gr)=UG​VG⊤Q=\mathrm{msgn}({G}\_{\mathrm{r}})={U}\_{G}{V}\_{G}^{\top}. Since G=Gr​Z​Z⊤G={G}\_{\mathrm{r}}ZZ^{\top}, we have

|  |  |  |
| --- | --- | --- |
|  | tr​(G⊤​Q)=tr​((Gr​Z​Z⊤)⊤​Q)=tr​(Z​Z⊤​Gr⊤​Q)=tr​(Z​Z⊤​VG​SG​VG⊤).\mathrm{tr}(G^{\top}Q)=\mathrm{tr}\big(({G}\_{\mathrm{r}}ZZ^{\top})^{\top}Q\big)=\mathrm{tr}\big(ZZ^{\top}{G}\_{\mathrm{r}}^{\top}Q\big)=\mathrm{tr}\big(ZZ^{\top}{V}\_{G}{S}\_{G}{V}\_{G}^{\top}\big). |  |

Now Z​Z⊤ZZ^{\top} and VG​SG​VG⊤{V}\_{G}{S}\_{G}{V}\_{G}^{\top} are both positive semidefinite. Therefore, the trace of their product is nonnegative, so tr​(G⊤​Q)≥0\mathrm{tr}(G^{\top}Q)\geq 0.
∎

Making a connection to the standard Muon optimizer, we establish the following result.

###### Corollary 1 (Isotropic activations recover Muon).

Under the assumptions of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.3 Newton–Muon ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), if additionally
Z​Z⊤∝In{Z}{Z}^{\top}\propto{I}\_{n}, then

|  |  |  |
| --- | --- | --- |
|  | Q⋆∝msgn​(G).{Q}^{\star}\propto\mathrm{msgn}({G}). |  |

That is, the triplet quadratic surrogate model recovers the standard Muon update in this case.

However, as we will demonstrate in Section [5](#S5 "5 LLM Experiments ‣ The Newton–Muon Optimizer"), Z​Z⊤{Z}{Z}^{\top} is highly anisotropic in practice. Therefore, the right preconditioner (Z​Z⊤)−1({Z}{Z}^{\top})^{-1}, which is readily available during training, can significantly alter the singular spaces of the gradient matrix GG. Furthermore, the added computational cost of computing (G​(Z​Z⊤)−1)(G(ZZ^{\top})^{-1}) is insignificant compared to the empirical performance gains of Newton–Muon over standard Muon.

It is also instructive to interpret Newton–Muon through the lens of orthogonal equivariance. Standard Muon is basis-free in the sense that for any orthogonal matrices Om∈ℝm×m{O}\_{m}\in\mathbb{R}^{m\times m} and On∈ℝn×n{O}\_{n}\in\mathbb{R}^{n\times n}, we have

|  |  |  |
| --- | --- | --- |
|  | msgn​(Om​G​On)=Om​msgn​(G)​On.\mathrm{msgn}({O}\_{m}{G}{O}\_{n})={O}\_{m}\mathrm{msgn}({G}){O}\_{n}. |  |

Thus, rotating the gradient matrix on the left or right correspondingly rotates the Muon update in the exact same manner. For Newton–Muon, the analogous commutative diagram holds if the right rotation of the gradient matrix GG is accompanied by the coordinate transformation Z↦On⊤​ZZ\mapsto O\_{n}^{\top}Z. Consequently,

|  |  |  |
| --- | --- | --- |
|  | msgn​(Om​G​On​((On⊤​Z)​(On⊤​Z)⊤)−1)=Om​msgn​(G​(Z​Z⊤)−1)​On.\mathrm{msgn}\left({O}\_{m}{G}{O}\_{n}\big((O\_{n}^{\top}Z)(O\_{n}^{\top}Z)^{\top}\big)^{-1}\right)={O}\_{m}\mathrm{msgn}\big({G}({Z}{Z}^{\top})^{-1}\big){O}\_{n}. |  |

In contrast, if one rotates G{G} on the right while keeping Z{Z} fixed, right equivariance generally fails unless On{O}\_{n} commutes with Z​Z⊤{Z}{Z}^{\top}, as illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2.3 Newton–Muon ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer").

|  |  |  |
| --- | --- | --- |
|  | G{G}msgn​(G){\mathrm{msgn}(G)}Om​G​On{{O}\_{m}G{O}\_{n}}Om​msgn​(G)​On{{O}\_{m}\mathrm{msgn}(G){O}\_{n}}  (G,Z){(G,Z)}msgn​(G​(Z​Z⊤)−1){\mathrm{msgn}\big(G(ZZ^{\top})^{-1}\big)}(Om​G​On,On⊤​Z){({O}\_{m}G{O}\_{n},{O}\_{n}^{\top}Z)}Om​msgn​(G​(Z​Z⊤)−1)​On{{O}\_{m}\mathrm{msgn}\big(G(ZZ^{\top})^{-1}\big){O}\_{n}} |  |

Figure 2: Left: standard Muon is orthogonally equivariant. Right: Newton–Muon takes the pair (G,Z)(G,Z) as input, and the diagram commutes if the right rotation of GG is accompanied by the transformation Z↦On⊤​ZZ\mapsto O\_{n}^{\top}Z.

## 3 Convergence Analysis of Newton–Muon: Case Study

This section presents a case study of Newton–Muon in a simple quadratic model under a single spike assumption on Z​Z⊤{Z}{Z}^{\top}, with one spiked eigendirection and an isotropic bulk. A fully general convergence analysis is difficult, so we study this simple model in which explicit rates can be obtained. The main finding is that Newton–Muon achieves a contraction rate independent of the spike condition number κ\kappa, whereas the rates of gradient descent and Muon scale as 1/κ1/\kappa as κ\kappa increases. We consider the iterative optimization of W{W} under full-batch gradient descent, Muon, and Newton–Muon, using the learning rate sequence {ηt}t≥0\{\eta\_{t}\}\_{t\geq 0}. We consider the objective

|  |  |  |
| --- | --- | --- |
|  | minW⁡f​(W)≔12​‖W​Z−W⋆​Z‖F2,\min\_{W}f({W})\coloneqq\frac{1}{2}\|{W}Z-W^{\star}{Z}\|\_{F}^{2}, |  |

where Z∈ℝn×N{Z}\in\mathbb{R}^{n\times N} is fixed, W⋆∈ℝm×nW^{\star}\in\mathbb{R}^{m\times n} is the ground-truth matrix, and ∥⋅∥F\|\cdot\|\_{F} denotes the Frobenius norm. The exact gradient is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇Wf​(W)=(W−W⋆)​Z​Z⊤.\nabla\_{{W}}f({W})=({W}-W^{\star}){Z}{Z}^{\top}. |  | (6) |

Denote WtW\_{t} as the weight matrix at iteration tt. The resulting updates are

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GD:Wt+1\displaystyle\text{GD:}\quad W\_{t+1} | =Wt−ηt​∇Wf​(Wt)=Wt−ηt​(Wt−W⋆)​Z​Z⊤,\displaystyle=W\_{t}-\eta\_{t}\nabla\_{W}f(W\_{t})=W\_{t}-\eta\_{t}(W\_{t}-W^{\star})ZZ^{\top}, |  | (7) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Muon:Wt+1\displaystyle\text{Muon:}\quad W\_{t+1} | =Wt−ηt​msgn​(∇Wf​(Wt))=Wt−ηt​msgn​((Wt−W⋆)​Z​Z⊤),\displaystyle=W\_{t}-\eta\_{t}\mathrm{msgn}\big(\nabla\_{W}f(W\_{t})\big)=W\_{t}-\eta\_{t}\mathrm{msgn}\big((W\_{t}-W^{\star})ZZ^{\top}\big), |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Newton–Muon:Wt+1\displaystyle\text{Newton--Muon:}\quad W\_{t+1} | =Wt−ηt​msgn​(∇Wf​(Wt)​(Z​Z⊤)−1)=Wt−ηt​msgn​(Wt−W⋆).\displaystyle=W\_{t}-\eta\_{t}\mathrm{msgn}\big(\nabla\_{W}f(W\_{t})(ZZ^{\top})^{-1}\big)=W\_{t}-\eta\_{t}\mathrm{msgn}(W\_{t}-W^{\star}). |  | (9) |

Before going into the theoretical details, we can already see the potential advantage of Newton–Muon over gradient descent and Muon. From the update equations ([7](#S3.E7 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer"))–([9](#S3.E9 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), gradient descent and Muon apply the right-multiplication factor Z​Z⊤ZZ^{\top} to the displacement Wt−W⋆W\_{t}-W^{\star}. When Z​Z⊤ZZ^{\top} is highly anisotropic, as in LLM training, where the activation matrix ZZ has bounded stable rank (davis2025spectral) and a very large condition number (Table [4](#S5.T4 "Table 4 ‣ Results. ‣ 5.2 Quadratic Score ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer")), this anisotropy can distort the effective update, over-emphasizing some directions toward W⋆W^{\star} while suppressing others. In contrast, Newton–Muon explicitly cancels this right-side anisotropy by multiplying by (Z​Z⊤)−1(ZZ^{\top})^{-1}.

#### Single spike model.

We now introduce the single-spike model and restrict to a low-dimensional invariant subspace spanned by a single spiked eigendirection of Z​Z⊤{Z}{Z}^{\top} and its isotropic complement.
We initialize W0=0{W}\_{0}=0 and assume that the singular values of Z{Z} satisfy σ12=κ>1\sigma\_{1}^{2}=\kappa>1 and σ22=⋯=σn2=1\sigma\_{2}^{2}=\cdots=\sigma\_{n}^{2}=1, so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Z​Z⊤=UZ​diag​(κ,1,…,1)​UZ⊤,κ>1,{Z}{Z}^{\top}={U}\_{Z}\mathrm{diag}(\kappa,1,\ldots,1){U}\_{Z}^{\top},\qquad\kappa>1, |  | (10) |

for some orthogonal matrix UZ{U}\_{Z}. Let 𝒆1∈ℝn\boldsymbol{e}\_{1}\in\mathbb{R}^{n} denote the first column of UZU\_{Z}. We assume W⋆W^{\star} has the following structured rank-rr form

|  |  |  |  |
| --- | --- | --- | --- |
|  | W⋆=−𝒖1​(α1,0​𝒆1⊤+β1,0​𝒃1⊤)−∑i=2r𝒖i​(βi,0​𝒃i⊤),W^{\star}=-\boldsymbol{u}\_{1}\big(\alpha\_{1,0}\boldsymbol{e}\_{1}^{\top}+\beta\_{1,0}\boldsymbol{b}\_{1}^{\top}\big)-\sum\_{i=2}^{r}\boldsymbol{u}\_{i}\big(\beta\_{i,0}\boldsymbol{b}\_{i}^{\top}\big), |  | (11) |

where 𝒖1,…,𝒖r∈ℝm\boldsymbol{u}\_{1},\dots,\boldsymbol{u}\_{r}\in\mathbb{R}^{m} are orthonormal, and 𝒃1,…,𝒃r∈ℝn\boldsymbol{b}\_{1},\dots,\boldsymbol{b}\_{r}\in\mathbb{R}^{n} are orthonormal with 𝒃i⟂𝒆1\boldsymbol{b}\_{i}\perp\boldsymbol{e}\_{1} for all ii. In particular, this requires r≤mr\leq m and r≤n−1r\leq n-1. We assume (α1,0,β1,0)≠(0,0)(\alpha\_{1,0},\beta\_{1,0})\neq(0,0) and βi,0≠0\beta\_{i,0}\neq 0 for i≥2i\geq 2.
This assumption ensures that (i) W⋆W^{\star} is rank-rr, and (ii) only the first mode mixes the spiked direction 𝒆1\boldsymbol{e}\_{1} with a non-spike direction; all other modes lie entirely in the isotropic subspace 𝒆1⟂\boldsymbol{e}\_{1}^{\perp}.
Since W0=0{W}\_{0}=0, the initial residual W0−W⋆=−W⋆{W}\_{0}-W^{\star}=-W^{\star} lies in the same family. We will show that, for all tt,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wt−W⋆=𝒖1​(α1,t​𝒆1⊤+β1,t​𝒃1⊤)+∑i=2r𝒖i​(βi,t​𝒃i⊤).W\_{t}-W^{\star}=\boldsymbol{u}\_{1}\big(\alpha\_{1,t}\boldsymbol{e}\_{1}^{\top}+\beta\_{1,t}\boldsymbol{b}\_{1}^{\top}\big)+\sum\_{i=2}^{r}\boldsymbol{u}\_{i}\big(\beta\_{i,t}\boldsymbol{b}\_{i}^{\top}\big). |  | (12) |

The main consequence of ([10](#S3.E10 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) and ([11](#S3.E11 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) is that the dynamics of gradient descent, Muon, and Newton–Muon decouple across the modes i=1,…,ri=1,\dots,r, yielding explicit one-dimensional recursions for the coefficients. We adopt the convention that msgn​(0)=0\mathrm{msgn}(0)=0, β/|β|=0\beta/|\beta|=0 when β=0\beta=0, and (α,β)/α2+β2=(0,0)(\alpha,\beta)/\sqrt{\alpha^{2}+\beta^{2}}=(0,0) when (α,β)=(0,0)(\alpha,\beta)=(0,0).

###### Lemma 1 (Dynamics under the single spike model).

Under ([10](#S3.E10 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) and ([11](#S3.E11 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), gradient descent ([7](#S3.E7 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), Muon ([8](#S3.E8 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), and Newton–Muon ([9](#S3.E9 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) preserve the decomposition ([12](#S3.E12 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")). In particular, the dynamics decouple across modes i=1,…,ri=1,\dots,r, and the coefficients satisfy the following recursions.

* •

  For gradient descent,

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α1,t+1=(1−ηt​κ)​α1,t,βi,t+1=(1−ηt)​βi,t,i=1,…,r.\alpha\_{1,t+1}=(1-\eta\_{t}\kappa)\alpha\_{1,t},\qquad\beta\_{i,t+1}=(1-\eta\_{t})\beta\_{i,t},\quad i=1,\dots,r. |  | (13) |
* •

  For Muon,

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α1,t+1=α1,t−ηt​κ​α1,tκ2​α1,t2+β1,t2,βi,t+1={β1,t−ηt​(β1,t/κ2​α1,t2+β1,t2),i=1,βi,t−ηt​(βi,t/|βi,t|),i=2,…,r.\alpha\_{1,t+1}=\alpha\_{1,t}-\eta\_{t}\frac{\kappa\alpha\_{1,t}}{\sqrt{\kappa^{2}\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}},\qquad\beta\_{i,t+1}=\begin{cases}\beta\_{1,t}-\eta\_{t}\bigl(\beta\_{1,t}/\sqrt{\kappa^{2}\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}\bigr),&i=1,\\ \beta\_{i,t}-\eta\_{t}\bigl(\beta\_{i,t}/|\beta\_{i,t}|\bigr),&i=2,\dots,r.\end{cases} |  | (14) |
* •

  For Newton–Muon,

  |  |  |  |  |
  | --- | --- | --- | --- |
  |  | α1,t+1=α1,t−ηt​α1,tα1,t2+β1,t2,βi,t+1={β1,t−ηt​(β1,t/α1,t2+β1,t2),i=1,βi,t−ηt​(βi,t/|βi,t|),i=2,…,r.\alpha\_{1,t+1}=\alpha\_{1,t}-\eta\_{t}\frac{\alpha\_{1,t}}{\sqrt{\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}},\qquad\beta\_{i,t+1}=\begin{cases}\beta\_{1,t}-\eta\_{t}\bigl(\beta\_{1,t}/\sqrt{\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}\bigr),&i=1,\\ \beta\_{i,t}-\eta\_{t}\bigl(\beta\_{i,t}/|\beta\_{i,t}|\bigr),&i=2,\dots,r.\end{cases} |  | (15) |

###### Proof of Lemma [1](#Thmlemma1 "Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer").

Define 𝒗1,t≔α1,t​𝒆1+β1,t​𝒃1\boldsymbol{v}\_{1,t}\coloneqq\alpha\_{1,t}\boldsymbol{e}\_{1}+\beta\_{1,t}\boldsymbol{b}\_{1} and, for i≥2i\geq 2, 𝒗i,t≔βi,t​𝒃i\boldsymbol{v}\_{i,t}\coloneqq\beta\_{i,t}\boldsymbol{b}\_{i}, so that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wt−W⋆=∑i=1r𝒖i​𝒗i,t⊤.W\_{t}-W^{\star}=\sum\_{i=1}^{r}\boldsymbol{u}\_{i}\boldsymbol{v}\_{i,t}^{\top}. |  | (16) |

Since {𝒖i}i=1r\{\boldsymbol{u}\_{i}\}\_{i=1}^{r} are orthonormal and {𝒃i}i=1r\{\boldsymbol{b}\_{i}\}\_{i=1}^{r} are orthonormal with 𝒃i⟂𝒆1\boldsymbol{b}\_{i}\perp\boldsymbol{e}\_{1}, we have 𝒗i,t⟂𝒗j,t\boldsymbol{v}\_{i,t}\perp\boldsymbol{v}\_{j,t} for i≠ji\neq j for every tt.
Under ([10](#S3.E10 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), Z​Z⊤​𝒆1=κ​𝒆1ZZ^{\top}\boldsymbol{e}\_{1}=\kappa\boldsymbol{e}\_{1}, and the eigenspace with eigenvalue 11 is 𝒆1⟂\boldsymbol{e}\_{1}^{\perp}, so Z​Z⊤​𝒃i=𝒃iZZ^{\top}\boldsymbol{b}\_{i}=\boldsymbol{b}\_{i} for all i=1,…,ri=1,\dots,r. Hence

|  |  |  |  |
| --- | --- | --- | --- |
|  | Z​Z⊤​𝒗1,t=κ​α1,t​𝒆1+β1,t​𝒃1∈span​{𝒆1,𝒃1},Z​Z⊤​𝒗i,t=𝒗i,t∈span​{𝒃i}(i≥2).ZZ^{\top}\boldsymbol{v}\_{1,t}=\kappa\alpha\_{1,t}\boldsymbol{e}\_{1}+\beta\_{1,t}\boldsymbol{b}\_{1}\in\mathrm{span}\{\boldsymbol{e}\_{1},\boldsymbol{b}\_{1}\},\qquad ZZ^{\top}\boldsymbol{v}\_{i,t}=\boldsymbol{v}\_{i,t}\in\mathrm{span}\{\boldsymbol{b}\_{i}\}\ \ (i\geq 2). |  | (17) |

Moreover, the transformed vectors {Z​Z⊤​𝒗i,t}i=1r\{ZZ^{\top}\boldsymbol{v}\_{i,t}\}\_{i=1}^{r} remain mutually orthogonal across ii. Using ([6](#S3.E6 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) and ([16](#S3.E16 "In Proof of Lemma 1. ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")),

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇Wf​(Wt)=(Wt−W⋆)​Z​Z⊤=∑i=1r𝒖i​((Z​Z⊤​𝒗i,t)⊤).\nabla\_{W}f(W\_{t})=(W\_{t}-W^{\star})ZZ^{\top}=\sum\_{i=1}^{r}\boldsymbol{u}\_{i}\big((ZZ^{\top}\boldsymbol{v}\_{i,t})^{\top}\big). |  | (18) |

In particular, by ([17](#S3.E17 "In Proof of Lemma 1. ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")),

|  |  |  |
| --- | --- | --- |
|  | (Z​Z⊤​𝒗1,t)⊤=(κ​α1,t)​𝒆1⊤+β1,t​𝒃1⊤,(Z​Z⊤​𝒗i,t)⊤=βi,t​𝒃i⊤​(i≥2).(ZZ^{\top}\boldsymbol{v}\_{1,t})^{\top}=(\kappa\alpha\_{1,t})\boldsymbol{e}\_{1}^{\top}+\beta\_{1,t}\boldsymbol{b}\_{1}^{\top},\qquad(ZZ^{\top}\boldsymbol{v}\_{i,t})^{\top}=\beta\_{i,t}\boldsymbol{b}\_{i}^{\top}\ (i\geq 2). |  |

For gradient descent, substituting ([18](#S3.E18 "In Proof of Lemma 1. ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) into ([7](#S3.E7 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) gives

|  |  |  |
| --- | --- | --- |
|  | Wt+1−W⋆=(Wt−W⋆)−ηt​(Wt−W⋆)​Z​Z⊤,W\_{t+1}-W^{\star}=(W\_{t}-W^{\star})-\eta\_{t}(W\_{t}-W^{\star})ZZ^{\top}, |  |

and matching coefficients in the basis {𝒖1​𝒆1⊤,𝒖1​𝒃1⊤,𝒖i​𝒃i⊤}\{\boldsymbol{u}\_{1}\boldsymbol{e}\_{1}^{\top},\boldsymbol{u}\_{1}\boldsymbol{b}\_{1}^{\top},\boldsymbol{u}\_{i}\boldsymbol{b}\_{i}^{\top}\} yields α1,t+1=(1−ηt​κ)​α1,t\alpha\_{1,t+1}=(1-\eta\_{t}\kappa)\alpha\_{1,t} and βi,t+1=(1−ηt)​βi,t\beta\_{i,t+1}=(1-\eta\_{t})\beta\_{i,t} for all i=1,…,ri=1,\dots,r, i.e., ([13](#S3.E13 "In 1st item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")).

For Muon, since the right factors {Z​Z⊤​𝒗i,t}i=1r\{ZZ^{\top}\boldsymbol{v}\_{i,t}\}\_{i=1}^{r} in ([18](#S3.E18 "In Proof of Lemma 1. ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) are mutually orthogonal and the left factors {𝒖i}i=1r\{\boldsymbol{u}\_{i}\}\_{i=1}^{r} are orthonormal, the matrix sign of (Wt−W⋆)​Z​Z⊤(W\_{t}-W^{\star})ZZ^{\top} is given by

|  |  |  |
| --- | --- | --- |
|  | msgn​((Wt−W⋆)​Z​Z⊤)=∑i:‖Z​Z⊤​𝒗i,t‖2>0𝒖i​((Z​Z⊤​𝒗i,t)⊤‖Z​Z⊤​𝒗i,t‖2).\mathrm{msgn}\big((W\_{t}-W^{\star})ZZ^{\top}\big)=\sum\_{i:\|ZZ^{\top}\boldsymbol{v}\_{i,t}\|\_{2}>0}\boldsymbol{u}\_{i}\left(\frac{(ZZ^{\top}\boldsymbol{v}\_{i,t})^{\top}}{\|ZZ^{\top}\boldsymbol{v}\_{i,t}\|\_{2}}\right). |  |

For i≥2i\geq 2, Z​Z⊤​𝒗i,t=𝒗i,tZZ^{\top}\boldsymbol{v}\_{i,t}=\boldsymbol{v}\_{i,t} and ‖Z​Z⊤​𝒗i,t‖2=|βi,t|\|ZZ^{\top}\boldsymbol{v}\_{i,t}\|\_{2}=|\beta\_{i,t}|, yielding the i≥2i\geq 2 case in ([14](#S3.E14 "In 2nd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")). For i=1i=1, we have
Z​Z⊤​𝒗1,t=κ​α1,t​𝒆1+β1,t​𝒃1ZZ^{\top}\boldsymbol{v}\_{1,t}=\kappa\alpha\_{1,t}\boldsymbol{e}\_{1}+\beta\_{1,t}\boldsymbol{b}\_{1} and
‖Z​Z⊤​𝒗1,t‖2=κ2​α1,t2+β1,t2\|ZZ^{\top}\boldsymbol{v}\_{1,t}\|\_{2}=\sqrt{\kappa^{2}\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}, so substitution into ([8](#S3.E8 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) yields the i=1i=1 case in ([14](#S3.E14 "In 2nd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")). This also preserves ([12](#S3.E12 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")).

For Newton–Muon, since the right factors {𝒗i,t}i=1r\{\boldsymbol{v}\_{i,t}\}\_{i=1}^{r} are mutually orthogonal and the left factors {𝒖i}i=1r\{\boldsymbol{u}\_{i}\}\_{i=1}^{r} are orthonormal, the matrix sign of Wt−W⋆W\_{t}-W^{\star} is obtained mode-by-mode:

|  |  |  |
| --- | --- | --- |
|  | msgn​(Wt−W⋆)=∑i:‖𝒗i,t‖2>0𝒖i​(𝒗i,t‖𝒗i,t‖2)⊤.\mathrm{msgn}(W\_{t}-W^{\star})=\sum\_{i:\|\boldsymbol{v}\_{i,t}\|\_{2}>0}\boldsymbol{u}\_{i}\left(\frac{\boldsymbol{v}\_{i,t}}{\|\boldsymbol{v}\_{i,t}\|\_{2}}\right)^{\top}. |  |

Substituting into ([9](#S3.E9 "In 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) gives

|  |  |  |
| --- | --- | --- |
|  | Wt+1−W⋆=(Wt−W⋆)−ηt​msgn​(Wt−W⋆),W\_{t+1}-W^{\star}=(W\_{t}-W^{\star})-\eta\_{t}\mathrm{msgn}(W\_{t}-W^{\star}), |  |

and matching coefficients yields the i=1i=1 and i≥2i\geq 2 cases in ([15](#S3.E15 "In 3rd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")). This preserves ([12](#S3.E12 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) and completes the proof.
∎

Lemma [1](#Thmlemma1 "Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer") reduces the matrix iterates to scalar recursions for the coefficients {α1,t,βi,t}\{\alpha\_{1,t},\beta\_{i,t}\}. In particular, convergence of Wt→W⋆W\_{t}\to W^{\star} is equivalent to α1,t→0\alpha\_{1,t}\to 0 and βi,t→0\beta\_{i,t}\to 0 for all i=1,…,ri=1,\dots,r, and the effect of the spike κ\kappa is isolated to the mixed (α1,t,β1,t)(\alpha\_{1,t},\beta\_{1,t}) mode.

###### Corollary 2 (Convergence rates).

Assume ([10](#S3.E10 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) and ([11](#S3.E11 "In Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), and let r0>0r\_{0}>0 satisfy |α1,0|≤r0|\alpha\_{1,0}|\leq r\_{0} and |βi,0|≤r0|\beta\_{i,0}|\leq r\_{0} for all i=1,…,ri=1,\dots,r. Define rtr\_{t} recursively so that |α1,t|≤rt|\alpha\_{1,t}|\leq r\_{t} and |βi,t|≤rt|\beta\_{i,t}|\leq r\_{t} for all i=1,…,ri=1,\dots,r, and at each step choose ηt\eta\_{t} greedily to minimize the worst-case next bound rt+1r\_{t+1}. Then, for any target 0<ε<r00<\varepsilon<r\_{0}, gradient descent, Newton–Muon, and Muon reach |α1,t|≤ε|\alpha\_{1,t}|\leq\varepsilon and |βi,t|≤ε|\beta\_{i,t}|\leq\varepsilon for all i=1,…,ri=1,\dots,r after at most TGD​(ε)=O​(κ​log⁡(r0/ε))T\_{\mathrm{GD}}(\varepsilon)=O(\kappa\log(r\_{0}/\varepsilon)), TNM​(ε)=O​(log⁡(r0/ε))T\_{\mathrm{NM}}(\varepsilon)=O(\log(r\_{0}/\varepsilon)), and TM​(ε)=O​(κ​log⁡(r0/ε))T\_{\mathrm{M}}(\varepsilon)=O(\kappa\log(r\_{0}/\varepsilon)) iterations, respectively. In particular, Newton–Muon converges faster than both gradient descent and Muon by a factor of order κ\kappa.

###### Proof of Corollary [2](#Thmcorollary2 "Corollary 2 (Convergence rates). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer").

For gradient descent, ([13](#S3.E13 "In 1st item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) gives |α1,t+1|=|1−ηt​κ|​|α1,t||\alpha\_{1,t+1}|=|1-\eta\_{t}\kappa||\alpha\_{1,t}| and |βi,t+1|=|1−ηt|​|βi,t||\beta\_{i,t+1}|=|1-\eta\_{t}||\beta\_{i,t}| for i=1,…,ri=1,\dots,r. Hence, if |α1,t|,|βi,t|≤rt|\alpha\_{1,t}|,|\beta\_{i,t}|\leq r\_{t}, then rt+1≤max⁡{|1−ηt​κ|,|1−ηt|}​rtr\_{t+1}\leq\max\{|1-\eta\_{t}\kappa|,\ |1-\eta\_{t}|\}r\_{t}. The greedy choice minimizes the right-hand side by solving |1−ηt​κ|=|1−ηt||1-\eta\_{t}\kappa|=|1-\eta\_{t}|, which gives ηt=2/(κ+1)\eta\_{t}=2/(\kappa+1) and yields rt+1=(κ−1)/(κ+1)​rtr\_{t+1}=(\kappa-1)/(\kappa+1)r\_{t}.

For Newton–Muon, let st≔α1,t2+β1,t2s\_{t}\coloneqq\sqrt{\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}. By ([15](#S3.E15 "In 3rd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), α1,t+1=(1−ηt/st)​α1,t\alpha\_{1,t+1}=\bigl(1-\eta\_{t}/s\_{t}\bigr)\alpha\_{1,t} and β1,t+1=(1−ηt/st)​β1,t\beta\_{1,t+1}=\bigl(1-\eta\_{t}/s\_{t}\bigr)\beta\_{1,t}. Since |α1,t|,|β1,t|≤rt|\alpha\_{1,t}|,|\beta\_{1,t}|\leq r\_{t}, we have st≤2​rts\_{t}\leq\sqrt{2}\,r\_{t}. We first bound α1,t+1\alpha\_{1,t+1} and β1,t+1\beta\_{1,t+1}. If ηt≤st\eta\_{t}\leq s\_{t}, we obtain 1−ηt/st≤1−ηt/(2​rt)1-\eta\_{t}/s\_{t}\leq 1-\eta\_{t}/(\sqrt{2}\,r\_{t}), so |α1,t+1|≤(1−ηt/(2​rt))​|α1,t|≤rt−ηt/2|\alpha\_{1,t+1}|\leq\bigl(1-\eta\_{t}/(\sqrt{2}\,r\_{t})\bigr)|\alpha\_{1,t}|\leq r\_{t}-\eta\_{t}/\sqrt{2}, and similarly |β1,t+1|≤rt−ηt/2|\beta\_{1,t+1}|\leq r\_{t}-\eta\_{t}/\sqrt{2}. If instead ηt≥st\eta\_{t}\geq s\_{t}, then |1−ηt/st|=ηt/st−1|1-\eta\_{t}/s\_{t}|=\eta\_{t}/s\_{t}-1, and therefore |α1,t+1|=(ηt/st−1)​|α1,t|≤(ηt/st)​|α1,t|≤ηt|\alpha\_{1,t+1}|=(\eta\_{t}/s\_{t}-1)|\alpha\_{1,t}|\leq(\eta\_{t}/s\_{t})|\alpha\_{1,t}|\leq\eta\_{t}. The same argument gives |β1,t+1|≤ηt|\beta\_{1,t+1}|\leq\eta\_{t}. Combining the two cases, |α1,t+1|,|β1,t+1|≤max⁡{ηt,rt−ηt/2}|\alpha\_{1,t+1}|,|\beta\_{1,t+1}|\leq\max\{\eta\_{t},\ r\_{t}-\eta\_{t}/\sqrt{2}\}. Next, for i=2,…,ri=2,\dots,r, ([15](#S3.E15 "In 3rd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")) gives |βi,t+1|=||βi,t|−ηt||\beta\_{i,t+1}|=\bigl||\beta\_{i,t}|-\eta\_{t}\bigr|. Since |βi,t|≤rt|\beta\_{i,t}|\leq r\_{t}, this implies |βi,t+1|≤max⁡{ηt,rt−ηt}≤max⁡{ηt,rt−ηt/2}|\beta\_{i,t+1}|\leq\max\{\eta\_{t},\ r\_{t}-\eta\_{t}\}\leq\max\{\eta\_{t},\ r\_{t}-\eta\_{t}/\sqrt{2}\}. Hence rt+1≤max⁡{ηt,rt−ηt/2}r\_{t+1}\leq\max\{\eta\_{t},\ r\_{t}-\eta\_{t}/\sqrt{2}\}. The greedy choice minimizes the right-hand side by equalizing the two terms: ηt=rt−ηt/2\eta\_{t}=r\_{t}-\eta\_{t}/\sqrt{2}. Solving gives ηt=(2−2)​rt\eta\_{t}=(2-\sqrt{2})\,r\_{t}, and therefore rt+1=(2−2)​rtr\_{t+1}=(2-\sqrt{2})r\_{t}.

For Muon, let st≔κ2​α1,t2+β1,t2s\_{t}\coloneqq\sqrt{\kappa^{2}\alpha\_{1,t}^{2}+\beta\_{1,t}^{2}}. By ([14](#S3.E14 "In 2nd item ‣ Lemma 1 (Dynamics under the single spike model). ‣ Single spike model. ‣ 3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer")), we have α1,t+1=(1−ηt​κ/st)​α1,t\alpha\_{1,t+1}=(1-\eta\_{t}\kappa/s\_{t})\alpha\_{1,t} and β1,t+1=(1−ηt/st)​β1,t\beta\_{1,t+1}=(1-\eta\_{t}/s\_{t})\beta\_{1,t}. Since |α1,t|,|β1,t|≤rt|\alpha\_{1,t}|,|\beta\_{1,t}|\leq r\_{t}, we have st≤κ2+1​rts\_{t}\leq\sqrt{\kappa^{2}+1}\,r\_{t}.
Applying the same argument as above, we have rt+1≤max⁡{ηt,rt−ηt/κ2+1}r\_{t+1}\leq\max\{\eta\_{t},\ r\_{t}-\eta\_{t}/\sqrt{\kappa^{2}+1}\}. The greedy choice equalizes the two terms, so ηt=rt−ηt/κ2+1\eta\_{t}=r\_{t}-\eta\_{t}/\sqrt{\kappa^{2}+1}, and therefore rt+1=κ2+1/(κ2+1+1)​rtr\_{t+1}=\sqrt{\kappa^{2}+1}/(\sqrt{\kappa^{2}+1}+1)r\_{t}.

Iterating the three recursions shows that gradient descent, Newton–Muon, and Muon all decrease rtr\_{t} geometrically. Solving rt≤εr\_{t}\leq\varepsilon gives TGD​(ε)=O​(log⁡(r0/ε)/log⁡((κ+1)/(κ−1)))T\_{\mathrm{GD}}(\varepsilon)=O\big(\log(r\_{0}/\varepsilon)/\log((\kappa+1)/(\kappa-1))\big), TNM​(ε)=O​(log⁡(r0/ε))T\_{\mathrm{NM}}(\varepsilon)=O(\log(r\_{0}/\varepsilon)), and TM​(ε)=O​(log⁡(r0/ε)/log⁡(1+1/κ2+1))T\_{\mathrm{M}}(\varepsilon)=O\big(\log(r\_{0}/\varepsilon)/\log(1+1/\sqrt{\kappa^{2}+1})\big). Finally, for large κ\kappa,

|  |  |  |
| --- | --- | --- |
|  | log⁡(κ+1κ−1)≍1κ,log⁡(1+1κ2+1)≍1κ.\log\Big(\frac{\kappa+1}{\kappa-1}\Big)\asymp\frac{1}{\kappa},\qquad\log\Big(1+\frac{1}{\sqrt{\kappa^{2}+1}}\Big)\asymp\frac{1}{\kappa}. |  |

Therefore TGD​(ε)=O​(κ​log⁡(r0/ε))T\_{\mathrm{GD}}(\varepsilon)=O(\kappa\log(r\_{0}/\varepsilon)), TNM​(ε)=O​(log⁡(r0/ε))T\_{\mathrm{NM}}(\varepsilon)=O(\log(r\_{0}/\varepsilon)), and TM​(ε)=O​(κ​log⁡(r0/ε))T\_{\mathrm{M}}(\varepsilon)=O(\kappa\log(r\_{0}/\varepsilon)).
∎

## 4 One-Step Analysis of Newton–Muon

The convergence analysis in Section [3](#S3 "3 Convergence Analysis of Newton–Muon: Case Study ‣ The Newton–Muon Optimizer") relies on a simple quadratic case study in order to obtain explicit global dynamics. In this section, we provide a complementary one-step analysis that can be carried out more generally through local approximation.
It has two parts: (i) we introduce a scale-invariant quadratic score that measures the best achievable one-step decrease along a chosen direction after an optimal line search; and (ii) we use a simple spiked activation model to obtain a concrete numerical study. The theoretical derivation for isotropic activations is deferred to Appendix [E](#A5 "Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer").

### 4.1 Quadratic Score

We compare several update directions by their predicted one-step improvement under the surrogate ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")).
Note first that the objective ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) is a quadratic function of the update
direction Q{Q}, and we only need to choose a direction Q{Q} and then perform a one-dimensional line search along that direction. This reduces the comparison to a direction choice followed by one-dimensional line search.

Concretely, for a fixed direction Q∈ℝm×n{Q}\in\mathbb{R}^{m\times n} and step size η∈ℝ\eta\in\mathbb{R}, consider the update W↦W−η​Q{W}\mapsto{W}-\eta{Q}.
Plugging η​Q\eta{Q} into the quadratic surrogate ([2](#S2.E2 "In 2.1 Triplet Quadratic Surrogate ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) yields the one-dimensional model

|  |  |  |
| --- | --- | --- |
|  | J​(η​Q)=−η​tr​(Q​G⊤)+η22​N​tr​(H​Q​(Z​Z⊤)​Q⊤).J(\eta{Q})=-\eta\mathrm{tr}({Q}{G}^{\top})+\frac{\eta^{2}}{2N}\mathrm{tr}\Big({H}{Q}({Z}{Z}^{\top}){Q}^{\top}\Big). |  |

Under Assumption [1](#Thmassumption1 "Assumption 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"), the optimal step size is η⋆=tr​(Q​G⊤)/tr​(H​Q​(Z​Z⊤/N)​Q⊤)\eta^{\star}=\mathrm{tr}({Q}{G}^{\top})/\mathrm{tr}\big({H}{Q}({Z}{Z}^{\top}/N){Q}^{\top}\big).
Thus, we define the score of a direction Q{Q} to be proportional to the best one-step loss decrease under the surrogate

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(Q)≔−2​(J​(η⋆​Q)−J​(0))=tr​(Q​G⊤)2tr​(H​Q​(Z​Z⊤/N)​Q⊤).s({Q})\coloneqq-2\big(J(\eta^{\star}{Q})-J(0)\big)=\frac{\mathrm{tr}({Q}{G}^{\top})^{2}}{\mathrm{tr}\Big({H}{Q}({Z}{Z}^{\top}/N){Q}^{\top}\Big)}. |  | (19) |

This provides a general, scale-invariant criterion for comparing different update directions.

### 4.2 Numerical Study with Spiked Activation

We numerically evaluate the score s​(Q)s({Q}) in ([19](#S4.E19 "In 4.1 Quadratic Score ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer")) for the six directions under G=H​(W−W⋆)​(Z​Z⊤/N){G}={H}({W}-W^{\star})({Z}{Z}^{\top}/N).
These directions are
(i) Q=G{Q}={G},
(ii) QMuon​-​SVD=msgn​(G){Q}\_{\mathrm{Muon\text{-}SVD}}=\mathrm{msgn}({G}) (exact SVD),
(iii) QMuon​-​NS{Q}\_{\mathrm{Muon\text{-}NS}}, obtained by applying five Newton–Schulz iterations to G{G}, specifically, QMuon​-​NS=X5{Q}\_{\mathrm{Muon\text{-}NS}}={X}\_{5}, where X0=G/‖G‖F{X}\_{0}={G}/\|{G}\|\_{F} and Xt+1=(3.4445​Im−4.7750​Xt​Xt⊤+2.0315​(Xt​Xt⊤)2)​Xt{X}\_{t+1}=\big(3.4445{I}\_{m}-4.7750{X}\_{t}{X}\_{t}^{\top}+2.0315({X}\_{t}{X}\_{t}^{\top})^{2}\big){X}\_{t} for t=0,…,4t=0,\dots,4,
(iv) QNewton​-​Muon​-​SVD=msgn​(G​(Z​Z⊤)−1){Q}\_{\mathrm{Newton\text{-}Muon\text{-}SVD}}=\mathrm{msgn}\big({G}({Z}{Z}^{\top})^{-1}\big) (exact SVD),
(v) QNewton​-​Muon​-​NS{Q}\_{\mathrm{Newton\text{-}Muon\text{-}NS}}, obtained by applying the same five Newton–Schulz iterations to G​(Z​Z⊤)−1{G}({Z}{Z}^{\top})^{-1},
and (vi) QNewton=H−1​G​(Z​Z⊤)−1{Q}\_{\mathrm{Newton}}={H}^{-1}{G}({Z}{Z}^{\top})^{-1}, which is proportional to W−W⋆{W}-W^{\star}.
Throughout, we fix the square setting m=n=512m=n=512, λmax=1\lambda\_{\max}=1, and λmin=10−4\lambda\_{\min}=10^{-4}. We generate the eigenvalues of H{H} by the stretched-exponential rule

|  |  |  |
| --- | --- | --- |
|  | λk=λmax​exp⁡(−τ​(k−1)p),τ=log⁡(λmax/λmin)(m−1)p,\lambda\_{k}=\lambda\_{\max}\exp\big(-\tau(k-1)^{p}\big),\qquad\tau=\frac{\log(\lambda\_{\max}/\lambda\_{\min})}{(m-1)^{p}}, |  |

so that λm=λmin\lambda\_{m}=\lambda\_{\min}, then sample a random orthogonal P{P} and set H=P​diag​(λ1,…,λm)​P⊤{H}={P}\mathrm{diag}(\lambda\_{1},\ldots,\lambda\_{m}){P}^{\top}. We sample W−W⋆∈ℝm×n{W}-W^{\star}\in\mathbb{R}^{m\times n} with i.i.d. 𝒩​(0,1)\mathcal{N}(0,1) entries. For the activation matrix, we sample the columns 𝒛i∈ℝn\boldsymbol{z}\_{i}\in\mathbb{R}^{n} of Z{Z} independently from 𝒩​(0,diag​(κ,1,…,1))\mathcal{N}(0,\mathrm{diag}(\kappa,1,\ldots,1)) with κ=64\kappa=64, so that the population activation second moment is spiked. We use three choices of (N,p)(N,p): the baseline case (8192,0.3)(8192,0.3), a more top-uniform curvature case (8192,2.4)(8192,2.4), and a smaller-sample case (1024,0.3)(1024,0.3). For each setting, we run 10241024 independent simulations and report the mean score with the 2.5%–97.5% interval.

Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Numerical Study with Spiked Activation ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer") shows substantially higher scores for Newton–Muon than for Muon when both activation anisotropy and curvature anisotropy are strong, with Newton–Muon being closer to the optimal Newton direction. Muon also substantially outperforms gradient descent when the curvature is anisotropic.

![Refer to caption](/html/2604.01472/assets/x3.png)


Figure 3: Numerical study with spiked activation second moment diag​(κ,1,…,1)\mathrm{diag}(\kappa,1,\ldots,1) and κ=64\kappa=64. Top: baseline case (N,p)=(8192,0.3)(N,p)=(8192,0.3). Middle: more uniform curvature (N,p)=(8192,2.4)(N,p)=(8192,2.4). Bottom: smaller sample size (N,p)=(1024,0.3)(N,p)=(1024,0.3). The left column shows the spectrum of H{H}, and the right column shows the corresponding mean absolute scores s​(Q)s({Q}).

## 5 LLM Experiments

### 5.1 Pretraining Benchmark Records on Modded-NanoGPT

#### Benchmark setup.

We first compare Newton–Muon, Muon, and AdamW using historical records from the Modded-NanoGPT speedrun benchmark (modded\_nanogpt\_2024). This benchmark is designed around a fixed objective: train a GPT-style language model on FineWeb (penedo2024the) and report the wall-clock time needed to reach a target validation loss under a fixed hardware configuration. The benchmark has two tracks: the short track targets a validation loss of 3.28, while the medium track lowers the target to 2.92. For reproducibility, several important implementation details are presented in Appendix [A](#A1 "Appendix A LLM Experimental Details ‣ The Newton–Muon Optimizer").

#### Reference records.

In this benchmark, each Record #k refers to a historical leaderboard submission together with its released training configuration and runtime log. In our experiments, we use these records for reproduction and comparison. We use Record #4 from the short track as a baseline; this run was submitted shortly after Muon was introduced. To benchmark
against a stronger setup, for the short track we adapt a training script that is close in configuration to
Record #28. For the medium track, we compare against Record #17.

However, note that the later records are heavily optimized around the original Muon update, which makes them difficult to surpass without extensive hyperparameter tuning.
Since we only tune the learning rate and Newton–Muon hyperparameters, the baseline comparison should be viewed as the most direct comparison of Newton–Muon, while other comparisons should be interpreted as evaluations under training configurations that were heavily tuned for the original Muon update rather than for Newton–Muon.
Algorithm [1](#algorithm1 "In Reference records. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") summarizes the implementation of Newton–Muon.

Input: For each Muon layer ℓ\ell: input activations Zℓ∈ℝn×N{Z}\_{\ell}\in\mathbb{R}^{n\times N}, gradient Gℓ∈ℝm×n{G}\_{\ell}\in\mathbb{R}^{m\times n}, running second moment Kℓ∈ℝn×n{K}\_{\ell}\in\mathbb{R}^{n\times n} (initialize with 10−3​In10^{-3}{I}\_{n}), stored inverse Kℓ−1{K}\_{\ell}^{-1}.

Input: Hyperparameters: EWMA coefficient β\beta, ridge scaling γ\gamma, refresh interval kk.

for *step=0,1,…=0,1,\dots* do

foreach *ℓ\ell* do

if *(*step*+1)modk=0(\texttt{step}+1)\bmod k=0* then

Kℓ←β​Kℓ+(1−β)​Zℓ​Zℓ⊤/N{K}\_{\ell}\leftarrow\beta{K}\_{\ell}+(1-\beta){Z}\_{\ell}{Z}\_{\ell}^{\top}/N

// Compute via a symmetric rank-NN update (Appendix [B.1](#A2.SS1 "B.1 Symmetric Matrix Multiplication ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer"))

γℓ←γ⋅tr​(Kℓ)/n\gamma\_{\ell}\leftarrow\gamma\cdot\mathrm{tr}({K}\_{\ell})/n

Kℓ−1←(Kℓ+γℓ​In)−1{K}\_{\ell}^{-1}\leftarrow({K}\_{\ell}+\gamma\_{\ell}{I}\_{n})^{-1}

// Compute via Cholesky inverse (Appendix [B.2](#A2.SS2 "B.2 Cholesky Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer")) or a polynomial iteration with a custom symmetric-output matmul (Appendix [B.3](#A2.SS3 "B.3 Polynomial Iteration Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer"))

Gℓ←Gℓ​Kℓ−1{G}\_{\ell}\leftarrow{G}\_{\ell}{K}\_{\ell}^{-1}

// Kℓ−1{K}\_{\ell}^{-1} is applied to the raw layer gradient, before momentum, weight decay, and the remaining Muon pipeline

Apply the standard Muon update using the right-preconditioned gradient Gℓ{G}\_{\ell}

Algorithm 1 Newton–Muon

#### Baseline.

We begin with the short track Record #4 with a single NVIDIA H100 GPU. This run trains a 124M GPT-2 architecture with 3.25B training tokens. The Muon baseline
learning rate is 0.00360.0036. We modified the code to run on a single GPU without touching any training pipeline. For Newton–Muon, we use learning
rate 0.00400.0040, EWMA coefficient β=0.95\beta=0.95, ridge scaling γ=0.2\gamma=0.2, and refresh interval k=32k=32. For AdamW, we assign the transformer block parameters a reduced AdamW learning rate 0.0005760.000576, selected from a learning-rate sweep. All wall-clock numbers reported below are measured in our environment under this configuration.
Table [1](#S5.T1 "Table 1 ‣ Baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") summarizes the final validation losses and total training times. Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ The Newton–Muon Optimizer") compares the validation loss trajectories.

| Method | Loss | Time (s) |
| --- | --- | --- |
| AdamW | 3.3801 | 7228.4 |
| Muon | 3.2793 | 7314.1 |
| Newton–Muon | 3.2611 | 7443.3 |

Table 1: Short track Record #4 setting (single H100).

#### Ablation for baseline.

Unless otherwise stated, we fix the Newton–Muon settings to learning rate 0.00400.0040, EWMA coefficient β=0.95\beta=0.95, ridge scaling
γ=0.2\gamma=0.2, and refresh interval k=32k=32, and vary only the ablation parameters. Figure [4](#S5.F4 "Figure 4 ‣ Ablation for baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer")
shows a two-dimensional sweep over refresh interval kk and EWMA coefficient β\beta.
Figure [5](#S5.F5 "Figure 5 ‣ Ablation for baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") sweeps ridge scaling γ\gamma and the learning rate.

![Refer to caption](/html/2604.01472/assets/x4.png)


Figure 4: Refresh ablation for Newton–Muon on short track Record #4. Grouped bar plot over refresh interval kk, with one bar
per EWMA coefficient β\beta (ridge scaling fixed at γ=0.2\gamma=0.2 and learning rate fixed at 0.00400.0040).

![Refer to caption](/html/2604.01472/assets/x5.png)


Figure 5: Left: Ridge-scaling ablation for Newton–Muon on short track Record #4. Bar plot over ridge scaling γ\gamma with
k=32k=32, β=0.95\beta=0.95, and learning rate 0.00400.0040 fixed. Right: Learning-rate ablation for Newton–Muon on short track Record #4. Bar plot over learning rate with
k=32k=32, β=0.95\beta=0.95, and γ=0.2\gamma=0.2 fixed.

These ablations suggest three patterns. First, Figure [4](#S5.F4 "Figure 4 ‣ Ablation for baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") shows that very frequent refreshes (k∈{1,2}k\in\{1,2\}) consistently underperform, while
moderate refresh intervals (k∈{16,32,64}k\in\{16,32,64\}) yield the best final losses.
Second, ridge scaling is essential to make the right preconditioner numerically well behaved. Figure [5](#S5.F5 "Figure 5 ‣ Ablation for baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") indicates that overly small ridge values can degrade
training slightly since the right-preconditioner may be ill-conditioned. In contrast, a wide mid-range (roughly
γ∈[0.02,0.5]\gamma\in[0.02,0.5] here) performs similarly. Very large ridge values eventually move the update back toward
standard Muon, which corresponds to the loss rising toward the baseline 3.2793 as γ\gamma increases.
Third, the learning-rate sweep in Figure [5](#S5.F5 "Figure 5 ‣ Ablation for baseline. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") is relatively flat around the best region, and the best Newton–Muon learning rate is very close to the best Muon learning rate.

#### Short Track Record #28.

We reproduce a short track configuration similar to a record submitted around the same time as Record #28 on a single NVIDIA L40S GPU. This run trains a ∼\sim275M-parameter model with about 670670M training tokens. We use the training script at
<https://github.com/KellerJordan/modded-nanogpt/blob/9d9dc969c451c87b7ad3c84f807db2c2d9109f41/train_gpt.py>. For Newton–Muon, we do not change any other configurations including the learning rate. We use EWMA coefficient β=0.8\beta=0.8, ridge scaling γ=0.2\gamma=0.2, and refresh interval k=16k=16; since
the run is shorter than Record #4, we refresh more aggressively.
For AdamW, we assign the transformer block matrices a reduced AdamW learning rate 0.000550.00055.
Because this setting is noisy, we run experiments four times and summarize results in Table [2](#S5.T2 "Table 2 ‣ Short Track Record #28. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer"). Figure [6](#S5.F6 "Figure 6 ‣ Short Track Record #28. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") compares the validation-loss curves of
the second-best run (by final validation loss) from each group. Although Newton–Muon has slightly higher total runtime due to preconditioning overhead (Table [2](#S5.T2 "Table 2 ‣ Short Track Record #28. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer")), its loss-versus-time curve indicates a small advantage in time to reach comparable validation loss (Figure [6](#S5.F6 "Figure 6 ‣ Short Track Record #28. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer")).

| Method | Run 1 | Run 2 | Run 3 | Run 4 | Avg. loss | Avg. time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| AdamW | 3.4677 | 3.4631 | 3.4640 | 3.4566 | 3.4628 | 4272.0 |
| Muon | 3.2758 | 3.2777 | 3.2783 | 3.2830 | 3.2787 | 4305.9 |
| Newton–Muon | 3.2733 | 3.2736 | 3.2740 | 3.2745 | 3.2739 | 4342.4 |

Table 2: Short track Record #28 setting (single L40S).

![Refer to caption](/html/2604.01472/assets/x6.png)


Figure 6: Validation loss trajectories for the short track Record #28 setting (single L40S). We plot the second-best run for AdamW, Muon, and Newton–Muon, and show loss versus step (left) and loss versus wall-clock time (right).

#### Medium Track Record #17.

We next reproduce the medium track Record #17 setting. This run trains a ∼\sim455M-parameter model with about 3.123.12B training tokens. Similarly, for Newton–Muon we keep other configurations fixed. We use EWMA coefficient β=0.9\beta=0.9, ridge scaling γ=0.2\gamma=0.2, and refresh interval k=24k=24. We run both Muon and Newton–Muon three times and summarize results in Table [3](#S5.T3 "Table 3 ‣ Medium Track Record #17. ‣ 5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer"). AdamW is not included for the medium track comparison. The improvement in final validation loss is only marginal in this setting.

| Method | Run 1 | Run 2 | Run 3 | Avg. loss |
| --- | --- | --- | --- | --- |
| Muon | 2.9190 | 2.9208 | 2.9190 | 2.9196 |
| Newton–Muon | 2.9175 | 2.9191 | 2.9181 | 2.9183 |

Table 3: Medium track Record #17 setting. Wall-clock time is not reported since we did not observe a substantial improvement.

### 5.2 Quadratic Score

We evaluate candidate directions using the same one-dimensional surrogate, but with the exact parameter-space Hessian in the denominator. Given a parameter matrix W{W} and a candidate direction Q{Q}, we define the score

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(Q)=tr​(G⊤​Q)2vec​(Q)⊤​ℋW​vec​(Q).s({Q})=\frac{\mathrm{tr}({G}^{\top}{Q})^{2}}{\mathrm{vec}({Q})^{\top}{\mathcal{H}}\_{{W}}\mathrm{vec}({Q})}. |  | (20) |

When vec​(Q)⊤​ℋW​vec​(Q)>0\mathrm{vec}({Q})^{\top}{\mathcal{H}}\_{{W}}\mathrm{vec}({Q})>0, this quantity is, up to a constant factor, the predicted best decrease of the local quadratic model after an optimal line search along Q{Q}.

We compute the curvature term vec​(Q)⊤​ℋW​vec​(Q)\mathrm{vec}({Q})^{\top}{\mathcal{H}}\_{{W}}\mathrm{vec}({Q}) using exact Hessian-vector products for the batch loss, without any Kronecker approximation.
To make scores comparable across methods and layers, we normalize each layer’s raw gradient to unit Frobenius norm
G^≔G/‖G‖F\widehat{{G}}\coloneqq{G}/\|{G}\|\_{F}.
Each candidate direction is also normalized
Q^≔Q/‖Q‖F\widehat{{Q}}\coloneqq{Q}/\|{Q}\|\_{F}.
We then report three quantities. First, the alignment term
tr​(G^⊤​Q^)\mathrm{tr}(\widehat{{G}}^{\top}\widehat{{Q}}).
Second, the curvature term
(1/2)​vec​(Q^)⊤​ℋW​vec​(Q^)(1/2)\mathrm{vec}(\widehat{{Q}})^{\top}{\mathcal{H}}\_{{W}}\mathrm{vec}(\widehat{{Q}}).
Third, the resulting score
s​(Q^)s(\widehat{{Q}}).

#### Experimental setup.

We use Pythia-70M (biderman2023pythia) and evaluate two checkpoints, step 1000 and step 50000. We stream the Pile (gao2020pile) and form a large batch of 20482048 sequences of length 10241024, totaling 1024×16×128=2,097,1521024\times 16\times 128=2{,}097{,}152 tokens. We study four weight matrices in the fourth transformer block: the attention output projection, the attention QKV projection, the MLP expansion, and the MLP contraction. To stabilize computation, we replace Z​Z⊤{Z}{Z}^{\top} by the damped matrix Kγ≔Z​Z⊤+γ​In{K}\_{\gamma}\coloneqq{Z}{Z}^{\top}+\gamma{I}\_{n} with γ>0\gamma>0, starting from a small damping value and multiplying γ\gamma by 1010 until Kγ{K}\_{\gamma} admits a stable float64 Cholesky factorization.

We compare five directions per layer. The gradient descent direction is Q=G^{Q}=\widehat{{G}}. Muon-NS5 and Muon-NS32 apply 5 and 32 Newton–Schulz iterations to G^\widehat{{G}}, respectively. Newton–Muon-NS5 and Newton–Muon-NS32 apply 5 and 32 Newton–Schulz steps to the right-preconditioned gradient.

#### Results.

Figure [7](#S5.F7 "Figure 7 ‣ Results. ‣ 5.2 Quadratic Score ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") summarizes the results. For these four matrices at the two sampled checkpoints, the ranking is consistent. Newton–Muon achieves the highest score, Muon is intermediate, and the raw gradient is worst. By construction, the gradient direction has the largest alignment with itself, but it also typically has a large curvature term. Both Muon and Newton–Muon substantially reduce this curvature term. Relative to Muon, Newton–Muon usually reduces the curvature term more than it reduces the alignment term, which explains its higher score. The advantage of Newton–Muon over Muon is smaller at step 50000 than at step 1000, and this trend appears across all four matrices. Also, NS5 and NS32 are close, suggesting that a small number of Newton–Schulz steps often suffices.

Table [4](#S5.T4 "Table 4 ‣ Results. ‣ 5.2 Quadratic Score ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer") reports diagnostics of the input activation second moment computed on the same batch. The matrices are strongly anisotropic, with large diagonal spread, substantial off-diagonal mass, and very large condition numbers.

| Module | nn | κ​(Z​Z⊤)\kappa({Z}{Z}^{\top}) | dmax/dmind\_{\max}/d\_{\min} | o¯/dmean\bar{o}/d\_{\mathrm{mean}} | γ/dmean\gamma/d\_{\mathrm{mean}} |
| --- | --- | --- | --- | --- | --- |
| Attn out | 512 | 9.18×1039.18\times 10^{3} | 2.84×1012.84\times 10^{1} | 9.89×1019.89\times 10^{1} | 1.0×10−61.0\times 10^{-6} |
| Attn QKV | 512 | 2.04×1062.04\times 10^{6} | 3.00×1003.00\times 10^{0} | 6.59×1016.59\times 10^{1} | 1.0×10−61.0\times 10^{-6} |
| MLP expansion | 512 | ∞\infty | 3.01×1003.01\times 10^{0} | 6.56×1016.56\times 10^{1} | 1.0×10−41.0\times 10^{-4} |
| MLP contraction | 2048 | – | 7.23×1017.23\times 10^{1} | 1.88×1021.88\times 10^{2} | 1.0×10−61.0\times 10^{-6} |
| Attn out | 512 | 2.34×1032.34\times 10^{3} | 3.02×1013.02\times 10^{1} | 7.21×1017.21\times 10^{1} | 1.0×10−61.0\times 10^{-6} |
| Attn QKV | 512 | 9.77×1049.77\times 10^{4} | 1.66×1021.66\times 10^{2} | 4.22×1014.22\times 10^{1} | 1.0×10−61.0\times 10^{-6} |
| MLP expansion | 512 | 8.22×1068.22\times 10^{6} | 1.21×1021.21\times 10^{2} | 3.97×1013.97\times 10^{1} | 1.0×10−61.0\times 10^{-6} |
| MLP contraction | 2048 | – | 7.46×1027.46\times 10^{2} | 1.88×1021.88\times 10^{2} | 1.0×10−61.0\times 10^{-6} |

Table 4: Diagnostics of the activation second moment Z​Z⊤{Z}{Z}^{\top} for four modules at step 1000 (top block) and step 50000 (bottom block). Here dmind\_{\min}, dmeand\_{\mathrm{mean}}, and dmaxd\_{\max} are the minimum, mean, and maximum diagonal entries of Z​Z⊤{Z}{Z}^{\top}, o¯=(1/n)​∑i∑j≠i|(Z​Z⊤)i​j|\bar{o}=(1/n)\sum\_{i}\sum\_{j\neq i}|({Z}{Z}^{\top})\_{ij}|, and κ​(Z​Z⊤)\kappa({Z}{Z}^{\top}) is the condition number of Z​Z⊤{Z}{Z}^{\top}. The value κ​(Z​Z⊤)\kappa({Z}{Z}^{\top}) is reported when eigenvalues were explicitly computed; “∞\infty” indicates numerical instability in the eigenspectrum estimate (a tiny negative eigenvalue at step 1000), and “–” indicates that the spectrum was not explicitly computed at n=2048n=2048. The last column reports the relative damping level selected by the adaptive Cholesky procedure.



![Refer to caption](/html/2604.01472/assets/x7.png)

![Refer to caption](/html/2604.01472/assets/x8.png)

Figure 7: Quadratic score comparison on four layer matrices from Pythia at checkpoints step 1000 (top) and step 50000 (bottom), corresponding to the one-step analysis in ([20](#S5.E20 "In 5.2 Quadratic Score ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer")).

## 6 Discussion

In this work, we introduce a triplet quadratic surrogate model that offers a local second-order view of Muon and leads to a new optimizer, Newton–Muon. Under an isotropic proxy, one-step minimization yields the update msgn​(G​(Z​Z⊤)−1)\mathrm{msgn}(G(ZZ^{\top})^{-1}), suggesting that Muon is an implicit Newton method without right preconditioning by the input second moment. Empirically, on our reproductions of historical Modded-NanoGPT speedrun benchmark configurations, Newton–Muon reaches the target validation loss in 6% fewer steps and reduces wall-clock time to that loss by over 4% relative to the Muon baseline. However, several limitations of the current framework remain and point to important directions for future work.

First, Newton–Muon relies on the isotropic proxy ΣW∝Im{\Sigma}\_{W}\propto{I}\_{m}. This choice is simple and robust, but it discards potentially useful information about the displacement distribution and may become inaccurate later in training. It is therefore better viewed as a practical proxy than as a faithful model throughout optimization. Indeed, weight matrices in classification models often exhibit low dimensional geometric structure, as suggested by neural collapse (papyan2020prevalence) and minority collapse (fang2021exploring); related phenomena, including linguistic collapse (wu2024linguistic) and cluster formation in attention dynamics (geshkovski2023emergence), have also been observed in transformers. A natural direction for future work is to estimate ΣW{\Sigma}\_{{W}} from training dynamics. With a sufficiently accurate estimate of ΣW{\Sigma}\_{{W}}, one need not explicitly compute its SVD or symmetric square root in order to apply ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")); it suffices to obtain a factorization ΣW=M​M⊤{\Sigma}\_{{W}}={M}{M}^{\top}, and then use M{M} to implement the same update direction. More details on this factorized implementation are deferred to Appendix [F](#A6 "Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer"). However, our preliminary experiments in Appendix [F](#A6 "Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer") with non-identity proxies, such as diagonal and factorized forms, did not improve performance, suggesting that estimating ΣW{\Sigma}\_{{W}} reliably is nontrivial and may introduce feedback bias. Developing stable, cheap estimators for non-isotropic proxies remains an important open problem.

Second, the Kronecker approximation of the parameter-space Hessian omits explicit token coupling in transformer attention mechanisms. In general, for token indices t≠st\neq s, one can have Ht​s=∂2L/(∂𝒚t​∂𝒚s)≠0H\_{ts}=\partial^{2}L/(\partial\boldsymbol{y}\_{t}\partial\boldsymbol{y}\_{s})\neq 0, but these cross-token curvature terms are discarded when passing from the exact token-coupled formula ([21](#A4.E21 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")) to the averaged Kronecker approximation ([22](#A4.E22 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")). We also use a single shared curvature matrix HH across all samples and positions, whereas the local output space curvature may vary with sample or token. As a result, the approximation ℋW≈(Z​Z⊤/N)⊗H\mathcal{H}\_{W}\approx(ZZ^{\top}/N)\otimes H may become less accurate in late training, when the Hessian can deviate further from Kronecker structure. These may help explain why Muon’s advantage decays in very late training (wen2025fantastic). Future work should develop tractable approximations that incorporate token-coupled curvature and relax the shared HH assumption.

Third, our experiments computed the activation second moment inverse using a damped Cholesky solve of Kγ=Z​Z⊤+γ​In{K}\_{\gamma}={Z}{Z}^{\top}+\gamma I\_{n}. In most cases we used the full inverse of Kγ{K}\_{\gamma}, but for MLP contraction matrices of shape d×(4​d)d\times(4d) we instead used a block-diagonal approximation with four d×dd\times d blocks, which performed well in practice (see Appendix [A](#A1 "Appendix A LLM Experimental Details ‣ The Newton–Muon Optimizer")). This suggests that structured approximations to Kγ{K}\_{\gamma} may already suffice. Therefore, future work could study structured approximations such as block matrices or low-rank factorizations, which may substantially reduce memory and computation while preserving most of the benefit of Newton–Muon. Future work could also compute the activation second-moment inverse via polynomial iteration (Appendix [B.3](#A2.SS3 "B.3 Polynomial Iteration Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer")), which may be faster. It is also natural to replace our heuristic damping γ∝tr​(Z​Z⊤)/n\gamma\propto\mathrm{tr}(ZZ^{\top})/n by a shifted Cholesky rule (fukaya2020shifted), for example γ∝‖Z​Z⊤‖2\gamma\propto\|ZZ^{\top}\|\_{2} or γ∝‖Z​Z⊤‖F/n\gamma\propto\|ZZ^{\top}\|\_{F}/\sqrt{n}.

Fourth, future work should evaluate Newton–Muon in distributed training, since we only test it on a single GPU. In particular, it remains important to understand how to compute and invert the activation second moment Z​Z⊤{Z}{Z}^{\top} in multi-GPU training. This includes both efficient implementations and approximations that preserve most of the benefit of Newton–Muon. Studying these systems issues at scale is an important direction for future work.

### Acknowledgments

This research was supported in part by the Wharton AI fund.

\appendixpage\addappheadtotoc

## Appendix A LLM Experimental Details

This appendix collects the implementation and benchmarking details for the LLM experiments in Section [5.1](#S5.SS1 "5.1 Pretraining Benchmark Records on Modded-NanoGPT ‣ 5 LLM Experiments ‣ The Newton–Muon Optimizer"), including several details that are important for reproducing the reported results.

#### Comparison protocol.

For a fair comparison, we use several publicly logged benchmark records as configuration references and report wall-clock results from our reproductions. To keep the comparison controlled, for each selected record we modify only the optimizer and tune learning rates when needed. For Newton–Muon, we replace the Muon update with Newton–Muon. For AdamW, to keep the comparison controlled, we replace the Muon-optimized parameters by AdamW, set β=(0.9,0.999)\beta=(0.9,0.999), and tune the AdamW learning rate over a wide range, reporting the setting that achieves the lowest validation loss, while keeping the remaining training settings unchanged. All other training settings are kept identical to the original record; we may make implementation-level changes (e.g., custom kernels) only to improve efficiency without altering the underlying training pipeline, and we apply any such changes uniformly across methods within the same comparison. For each selected record, we reproduce the run in our own environment and report wall-clock time from these reproductions. We do not claim that our measured times match the leaderboard submission times; our focus is the within-environment wall-clock difference among Newton–Muon, Muon, and AdamW under otherwise identical training settings.

#### Layerwise second moment computation.

For the attention QKV projection and the MLP expansion, the input dimension is dd, so we form a single d×dd\times d
activation second moment matrix from their inputs. For the attention output projection, the input dimension is also dd,
but the activations are taken from the attention output immediately before applying the output projection.
For the MLP contraction, the input dimension is 4​d4d. To avoid forming a full (4​d)×(4​d)(4d)\times(4d) second moment, we use a
block-diagonal approximation: reshape each activation 𝒛∈ℝ4​d\boldsymbol{z}\in\mathbb{R}^{4d} into four
contiguous blocks 𝒛=[𝒛(1);…;𝒛(4)]\boldsymbol{z}=[\boldsymbol{z}^{(1)};\ldots;\boldsymbol{z}^{(4)}] with
𝒛(b)∈ℝd\boldsymbol{z}^{(b)}\in\mathbb{R}^{d}, and form four separate d×dd\times d second moments
Z(b)​Z(b)⊤{Z}^{(b)}{{Z}^{(b)}}^{\top}. The corresponding inverse is represented as four independent d×dd\times d
inverses, and applying it is done by splitting the MLP contraction gradient into four d×dd\times d blocks along the input
dimension and right-multiplying each block by its matching inverse.

#### Numerical precision.

There are also a few important implementation details regarding numerical precision. First, the activation second moment update
Zℓ​Zℓ⊤{Z}\_{\ell}{Z}\_{\ell}^{\top} is relatively well behaved, and in practice the accumulation can be done with bfloat16. However, the inverse computation is much more sensitive. We compute
Kℓ−1=(Kℓ+γℓ​In)−1{K}\_{\ell}^{-1}=({K}\_{\ell}+\gamma\_{\ell}{I}\_{n})^{-1} using a float32 Cholesky factorization followed by a Cholesky inverse (see Appendix [B.2](#A2.SS2 "B.2 Cholesky Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer")).
Second, when applying the right-preconditioner, the matrix multiply Gℓ​Kℓ−1{G}\_{\ell}{K}\_{\ell}^{-1} should also be performed in
float32. If the gradients are stored in bfloat16, we can upcast Gℓ{G}\_{\ell} to float32 before the multiplication.

## Appendix B Optimizing Computation

### B.1 Symmetric Matrix Multiplication

Our implementation exploits the fact that the activation second moment
Z​Z⊤{Z}{Z}^{\top} (and any polynomial p​(Z​Z⊤)p({Z}{Z}^{\top})) is symmetric. This
structure allows us to avoid redundant computation by computing only one triangle and reconstructing the other by
symmetry.

#### SYRK (symmetric rank-kk update).

Forming the second moment

|  |  |  |
| --- | --- | --- |
|  | K←Z​Z⊤{K}\leftarrow{Z}{Z}^{\top} |  |

is exactly a SYRK pattern: it suffices to compute either the lower or upper triangle of K{K} and fill the
other half by mirroring across the diagonal.

#### SYPP (Symmetric polynomial product).

In the polynomial inverse, we repeatedly multiply factors that are polynomials in Z​Z⊤{Z}{Z}^{\top}.
Since Z​Z⊤{Z}{Z}^{\top} is symmetric, any polynomial p​(Z​Z⊤)p({Z}{Z}^{\top}) is also
symmetric. Moreover, because these factors are functions of the same matrix, they commute, so products such as
p​(Z​Z⊤)​q​(Z​Z⊤)p({Z}{Z}^{\top})q({Z}{Z}^{\top}) remain symmetric. Consequently, we can compute
only one triangle of the product and write the mirrored entries to complete the matrix, reducing nearly half of the work
compared to a dense matrix multiplication.

PyTorch (paszke2019pytorch) does not expose a simple interface for enforcing this kind of triangular compute. Instead, we implement Triton (tillet2019triton) custom kernels, adapted from
the Modded-NanoGPT repository (modded\_nanogpt\_2024).

### B.2 Cholesky Inverse

When Kγ=Z​Z⊤+γ​In≻0{K}\_{\gamma}={Z}{Z}^{\top}+\gamma{I}\_{n}\succ 0, Cholesky factorization gives
Kγ=L​L⊤{K}\_{\gamma}={L}{L}^{\top} with L{L} lower triangular. We then explicitly form the
inverse Kγ−1=(L​L⊤)−1=L−⊤​L−1{K}\_{\gamma}^{-1}=({L}{L}^{\top})^{-1}={L}^{-\top}{L}^{-1}.
We use torch.linalg.cholesky\_ex followed by torch.cholesky\_inverse in float32.

### B.3 Polynomial Iteration Inverse

We also consider a polynomial iteration that explicitly constructs an approximation to
Kγ−1{K}\_{\gamma}^{-1} using only SYPP. When Kγ{K}\_{\gamma} has moderate condition number after damping, this
approach can be cheaper than Cholesky.

#### Principle.

Let Kγ≻0{K}\_{\gamma}\succ 0 and choose a scalar α>0\alpha>0 such that the scaled matrix

|  |  |  |
| --- | --- | --- |
|  | K~≔α​Kγ\widetilde{{K}}\coloneqq\alpha{K}\_{\gamma} |  |

has spectrum contained in (ε,1](\varepsilon,1], i.e.,

|  |  |  |
| --- | --- | --- |
|  | ε<λmin​(K~)≤λmax​(K~)≤1\varepsilon<\lambda\_{\min}(\widetilde{{K}})\leq\lambda\_{\max}(\widetilde{{K}})\leq 1 |  |

for a small safety margin ε>0\varepsilon>0. Define the residual matrix

|  |  |  |
| --- | --- | --- |
|  | R0≔In−K~.{R}\_{0}\coloneqq{I}\_{n}-\widetilde{{K}}. |  |

With this scaling, the spectrum of R0{R}\_{0} lies in [0,1−ε][0,1-\varepsilon]. Since
K~=In−R0\widetilde{{K}}={I}\_{n}-{R}\_{0},

|  |  |  |
| --- | --- | --- |
|  | Kγ−1=α​K~−1=α​(In−R0)−1.{K}\_{\gamma}^{-1}=\alpha\widetilde{{K}}^{-1}=\alpha({I}\_{n}-{R}\_{0})^{-1}. |  |

The goal is therefore to build an explicit approximation X≈(In−R0)−1{X}\approx({I}\_{n}-{R}\_{0})^{-1}.
We initialize the inverse estimate as

|  |  |  |
| --- | --- | --- |
|  | X0=In.{X}\_{0}={I}\_{n}. |  |

At step k=1,…,Tk=1,\dots,T, we pick a polynomial qk​(⋅)q\_{k}(\cdot) and apply it to the current residual matrix

|  |  |  |
| --- | --- | --- |
|  | Qk≔qk​(Rk−1),Xk≔Xk−1​Qk.{Q}\_{k}\coloneqq q\_{k}({R}\_{k-1}),\qquad{X}\_{k}\coloneqq{X}\_{k-1}{Q}\_{k}. |  |

This induces a residual update. Using Rk−1=In−K~​Xk−1{R}\_{k-1}={I}\_{n}-\widetilde{{K}}{X}\_{k-1}, we have
K~​Xk−1=In−Rk−1\widetilde{{K}}{X}\_{k-1}={I}\_{n}-{R}\_{k-1}, hence

|  |  |  |
| --- | --- | --- |
|  | Rk=In−K~​Xk=In−K~​Xk−1​Qk=In−(In−Rk−1)​qk​(Rk−1)=ϕk​(Rk−1),{R}\_{k}={I}\_{n}-\widetilde{{K}}{X}\_{k}={I}\_{n}-\widetilde{{K}}{X}\_{k-1}{Q}\_{k}={I}\_{n}-({I}\_{n}-{R}\_{k-1})q\_{k}({R}\_{k-1})=\phi\_{k}({R}\_{k-1}), |  |

where the induced scalar map on eigenvalues is

|  |  |  |
| --- | --- | --- |
|  | r+=ϕk​(r)≔1−(1−r)​qk​(r).r^{+}=\phi\_{k}(r)\coloneqq 1-(1-r)q\_{k}(r). |  |

Crucially, the admissible interval for the residual is iteration dependent. If Rk−1{R}\_{k-1} has spectrum contained
in an interval r∈ℐk−1r\in\mathcal{I}\_{k-1} (ℐ0=[0,1−ε]\mathcal{I}\_{0}=[0,1-\varepsilon]), then the next spectrum is contained in

|  |  |  |
| --- | --- | --- |
|  | r+∈ϕk​(ℐk−1),r^{+}\in\phi\_{k}(\mathcal{I}\_{k-1}), |  |

and the interval shrinks as kk grows. As the residual contracts toward zero, the effective design interval becomes much
smaller than [0,1−ε][0,1-\varepsilon], which allows later polynomials to be optimized for a tighter range and achieve stronger
contraction per SYPP.

#### Convergence guarantee.

Define s0≔1−εs\_{0}\coloneqq 1-\varepsilon, so that the initial residual interval is ℐ0=[0,s0]\mathcal{I}\_{0}=[0,s\_{0}]. If the first polynomial q1q\_{1} is chosen so that

|  |  |  |
| --- | --- | --- |
|  | supr∈ℐ0|ϕ1​(r)|≤s1<s0,\sup\_{r\in\mathcal{I}\_{0}}|\phi\_{1}(r)|\leq s\_{1}<s\_{0}, |  |

and for each later step k≥2k\geq 2 the polynomial qkq\_{k} is chosen so that

|  |  |  |
| --- | --- | --- |
|  | sup|r|≤sk−1|ϕk​(r)|≤sk<sk−1,\sup\_{|r|\leq s\_{k-1}}|\phi\_{k}(r)|\leq s\_{k}<s\_{k-1}, |  |

then ‖Rk‖2≤sk\|{R}\_{k}\|\_{2}\leq s\_{k} and the residual interval shrinks monotonically to 0. Repeating a short
sequence of such contractive steps yields RT{R}\_{T} close to zero and XT{X}\_{T} as an explicit
approximation to (In−R0)−1=K~−1({I}\_{n}-{R}\_{0})^{-1}=\widetilde{{K}}^{-1}. Finally,

|  |  |  |
| --- | --- | --- |
|  | Kγ−1≈α​XT.{K}\_{\gamma}^{-1}\approx\alpha{X}\_{T}. |  |

#### How the polynomials are chosen.

At the first step we choose q1q\_{1} to minimize s1=supr∈ℐ0|ϕ1​(r)|s\_{1}=\sup\_{r\in\mathcal{I}\_{0}}|\phi\_{1}(r)| on the one-sided interval ℐ0=[0,s0]\mathcal{I}\_{0}=[0,s\_{0}]. For each later step k≥2k\geq 2, we take the current symmetric spectral bound sk−1s\_{k-1} and choose qkq\_{k} to minimize
the next bound sk=sup|r|≤sk−1|ϕk​(r)|s\_{k}=\sup\_{|r|\leq s\_{k-1}}|\phi\_{k}(r)| under coefficient-magnitude constraints and an explicit robustness margin
that accounts for finite precision and modeling error. We implement this minimax design on a grid (via a linear
program) and then chain multiple steps under a total SYPP budget (via dynamic programming). The first step is designed on a
one-sided interval, and subsequent steps are designed on the symmetric interval to remain stable once roundoff introduces
small negative eigenvalues.

#### Numerical safeguards for precision.

Two hyperparameters in the LP directly control how conservative the bound update sk−1↦sks\_{k-1}\mapsto s\_{k} is.
First, INTERVAL\_PAD\_REL enlarges the design interval. For the first step, instead of optimizing ϕ1\phi\_{1} only on ℐ0=[0,s0]\mathcal{I}\_{0}=[0,s\_{0}], the solver designs on

|  |  |  |
| --- | --- | --- |
|  | ℐ0design=[0,(1+INTERVAL\_PAD\_REL)​s0].\mathcal{I}\_{0}^{\mathrm{design}}=[0,(1+\texttt{INTERVAL\\_PAD\\_REL})s\_{0}]. |  |

For each later step k≥2k\geq 2, instead of optimizing ϕk\phi\_{k} only on the current spectral
bound |r|≤sk−1|r|\leq s\_{k-1}, the solver designs on |r|≤skdesign|r|\leq s\_{k}^{\mathrm{design}} with

|  |  |  |
| --- | --- | --- |
|  | skdesign=(1+INTERVAL\_PAD\_REL)​sk−1.s\_{k}^{\mathrm{design}}=(1+\texttt{INTERVAL\\_PAD\\_REL})s\_{k-1}. |  |

This guards against underestimating the current residual bound by enforcing contraction on a slightly wider interval than
the nominal bound.
Second, NOISE\_ABS is a worst-case absolute perturbation radius for the scalar residual map. Concretely, the LP is
solved under the robust requirement that for all rr in the design interval and for all perturbations
δ∈[−NOISE\_ABS,NOISE\_ABS]\delta\in[-\texttt{NOISE\\_ABS},\texttt{NOISE\\_ABS}], the perturbed update remains bounded

|  |  |  |
| --- | --- | --- |
|  | |ϕk​(r)+δ|≤γk.\bigl|\phi\_{k}(r)+\delta\bigr|\leq\gamma\_{k}. |  |

Equivalently, the solver enforces |ϕk​(r)|≤γk−NOISE\_ABS|\phi\_{k}(r)|\leq\gamma\_{k}-\texttt{NOISE\\_ABS} on the grid, and then reports the robust bound
as sk=γk=sup|ϕk​(r)|+NOISE\_ABSs\_{k}=\gamma\_{k}=\sup|\phi\_{k}(r)|+\texttt{NOISE\\_ABS}.
The algorithm is shown in Algorithm [2](#algorithm2 "In Numerical safeguards for precision. ‣ B.3 Polynomial Iteration Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer").

Input: positive semidefinite matrix K∈ℝn×n{K}\in\mathbb{R}^{n\times n}, damping γ>0\gamma>0, SYPP budget BB.

Output: Explicit approximation Kγ−1~≈(K+γ​In)−1\widetilde{{K}\_{\gamma}^{-1}}\approx({K}+\gamma{I}\_{n})^{-1}.

Kγ←K+γ​In{K}\_{\gamma}\leftarrow{K}+\gamma{I}\_{n}

Estimate any upper bound of the largest eigenvalue λ¯max​(K)≥λmax​(K)\overline{\lambda}\_{\max}({K})\geq\lambda\_{\max}({K})

λ¯max​(Kγ)←λ¯max​(K)+γ\overline{\lambda}\_{\max}({K}\_{\gamma})\leftarrow\overline{\lambda}\_{\max}({K})+\gamma

α←1/λ¯max​(Kγ)\alpha\leftarrow 1/\overline{\lambda}\_{\max}({K}\_{\gamma}), K~←α​Kγ\widetilde{{K}}\leftarrow\alpha{K}\_{\gamma}, ε¯←α​γ\bar{\varepsilon}\leftarrow\alpha\gamma, X0←In{X}\_{0}\leftarrow{I}\_{n}, R0←In−K~{R}\_{0}\leftarrow{I}\_{n}-\widetilde{{K}}

Choose a plan (q1,…,qT)(q\_{1},\dots,q\_{T}) from Table [5](#A2.T5 "Table 5 ‣ Numerical safeguards for precision. ‣ B.3 Polynomial Iteration Inverse ‣ Appendix B Optimizing Computation ‣ The Newton–Muon Optimizer") with tabulated ε≤ε¯\varepsilon\leq\bar{\varepsilon} and total SYPP cost ≤B\leq B

for *k←1k\leftarrow 1 to T−1T-1* do

Qk←qk​(Rk−1){Q}\_{k}\leftarrow q\_{k}({R}\_{k-1})

Xk←Xk−1​Qk{X}\_{k}\leftarrow{X}\_{k-1}{Q}\_{k}

Rk←In−(In−Rk−1)​Qk{R}\_{k}\leftarrow{I}\_{n}-({I}\_{n}-{R}\_{k-1}){Q}\_{k}

QT←qT​(RT−1){Q}\_{T}\leftarrow q\_{T}({R}\_{T-1})

XT←XT−1​QT{X}\_{T}\leftarrow{X}\_{T-1}{Q}\_{T}

return Kγ−1~←α​XT\widetilde{{K}\_{\gamma}^{-1}}\leftarrow\alpha{X}\_{T} such that ‖In−Kγ​Kγ−1~‖2≤sout\left\|{I}\_{n}-{K}\_{\gamma}\widetilde{{K}\_{\gamma}^{-1}}\right\|\_{2}\leq s\_{\mathrm{out}}, where souts\_{\mathrm{out}} is the certified residual bound associated with the selected plan

Algorithm 2 Inverse of Kγ=K+γ​In{K}\_{\gamma}={K}+\gamma{I}\_{n} via polynomial approximation



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| ε\varepsilon | Total | souts\_{\mathrm{out}} | SYPP | Polynomial qk​(x)q\_{k}(x) |
| 0.0015 | 12 | 0.030717 | 3 | q1​(x)=1.991037−15.856588​x+31.760959​x2q\_{1}(x)=1.991037-15.856588x+31.760959x^{2} |
|  |  |  | 4 | q2​(x)=0.102569+0.102569​x+7.383161​x2+7.383161​x3q\_{2}(x)=0.102569+0.102569x+7.383161x^{2}+7.383161x^{3} |
|  |  |  | 3 | q3​(x)=1+2.541910​x+2.541910​x2q\_{3}(x)=1+2.541910x+2.541910x^{2} |
|  |  |  | 2 | q4​(x)=1+1.192261​x+1.192261​x2q\_{4}(x)=1+1.192261x+1.192261x^{2} |
| 0.0015 | 13 | 0.004865 | 3 | q1​(x)=1.991037−15.856588​x+31.760959​x2q\_{1}(x)=1.991037-15.856588x+31.760959x^{2} |
|  |  |  | 3 | q2​(x)=1+3.839962​x+3.839963​x2q\_{2}(x)=1+3.839962x+3.839963x^{2} |
|  |  |  | 3 | q3​(x)=1+2.989700​x+2.989700​x2q\_{3}(x)=1+2.989700x+2.989700x^{2} |
|  |  |  | 2 | q4​(x)=1.244063+1.244063​xq\_{4}(x)=1.244063+1.244063x |
|  |  |  | 2 | q5​(x)=1+1.047265​x+1.047265​x2q\_{5}(x)=1+1.047265x+1.047265x^{2} |
| 0.003 | 10 | 0.019885 | 3 | q1​(x)=1.964953−15.439061​x+31.064790​x2q\_{1}(x)=1.964953-15.439061x+31.064790x^{2} |
|  |  |  | 3 | q2​(x)=1+3.346712​x+3.346712​x2q\_{2}(x)=1+3.346712x+3.346712x^{2} |
|  |  |  | 2 | q3​(x)=1.403255+1.403255​xq\_{3}(x)=1.403255+1.403255x |
|  |  |  | 2 | q4​(x)=1+1.140006​x+1.140006​x2q\_{4}(x)=1+1.140006x+1.140006x^{2} |
| 0.003 | 11 | 0.002839 | 3 | q1​(x)=1.964953−15.439061​x+31.064790​x2q\_{1}(x)=1.964953-15.439061x+31.064790x^{2} |
|  |  |  | 3 | q2​(x)=1+3.346712​x+3.346712​x2q\_{2}(x)=1+3.346712x+3.346712x^{2} |
|  |  |  | 3 | q3​(x)=1+1.757644​x+1.757644​x2q\_{3}(x)=1+1.757644x+1.757644x^{2} |
|  |  |  | 2 | q4​(x)=1+1.028634​x+1.028634​x2q\_{4}(x)=1+1.028634x+1.028634x^{2} |
| 0.006 | 8 | 0.047094 | 3 | q1​(x)=1.915935−14.653845​x+29.754543​x2q\_{1}(x)=1.915935-14.653845x+29.754543x^{2} |
|  |  |  | 3 | q2​(x)=1+2.716205​x+2.716205​x2q\_{2}(x)=1+2.716205x+2.716205x^{2} |
|  |  |  | 2 | q3​(x)=1+1.262596​x+1.262596​x2q\_{3}(x)=1+1.262596x+1.262596x^{2} |
| 0.006 | 9 | 0.014106 | 3 | q1​(x)=1.915935−14.653845​x+29.754543​x2q\_{1}(x)=1.915935-14.653845x+29.754543x^{2} |
|  |  |  | 4 | q2​(x)=0.639753+0.639753​x+4.060692​x2+4.060692​x3q\_{2}(x)=0.639753+0.639753x+4.060692x^{2}+4.060692x^{3} |
|  |  |  | 2 | q3​(x)=1+1.108734​x+1.108734​x2q\_{3}(x)=1+1.108734x+1.108734x^{2} |
| 0.006 | 10 | 0.002087 | 3 | q1​(x)=1.915935−14.653845​x+29.754543​x2q\_{1}(x)=1.915935-14.653845x+29.754543x^{2} |
|  |  |  | 3 | q2​(x)=1+2.716205​x+2.716205​x2q\_{2}(x)=1+2.716205x+2.716205x^{2} |
|  |  |  | 2 | q3​(x)=1.160973+1.160973​xq\_{3}(x)=1.160973+1.160973x |
|  |  |  | 2 | q4​(x)=1+1.020113​x+1.020111​x2q\_{4}(x)=1+1.020113x+1.020111x^{2} |
| 0.012 | 7 | 0.047594 | 3 | q1​(x)=1.828900−13.257429​x+27.420730​x2q\_{1}(x)=1.828900-13.257429x+27.420730x^{2} |
|  |  |  | 3 | q2​(x)=1+2.072900​x+2.072900​x2q\_{2}(x)=1+2.072900x+2.072900x^{2} |
|  |  |  | 1 | q3​(x)=1.046594+1.046594​xq\_{3}(x)=1.046594+1.046594x |
| 0.012 | 8 | 0.008118 | 3 | q1​(x)=1.828900−13.257429​x+27.420730​x2q\_{1}(x)=1.828900-13.257429x+27.420730x^{2} |
|  |  |  | 3 | q2​(x)=1+2.072900​x+2.072900​x2q\_{2}(x)=1+2.072900x+2.072900x^{2} |
|  |  |  | 2 | q3​(x)=1+1.071558​x+1.071558​x2q\_{3}(x)=1+1.071558x+1.071558x^{2} |
| 0.025 | 6 | 0.048057 | 4 | q1​(x)=1.528164+1.400800​x−12.902311​x2+32​x4q\_{1}(x)=1.528164+1.400800x-12.902311x^{2}+32x^{4} |
|  |  |  | 2 | q2​(x)=1+1.266514​x+1.266514​x2q\_{2}(x)=1+1.266514x+1.266514x^{2} |
| 0.025 | 7 | 0.008458 | 3 | q1​(x)=1.679044−10.844625​x+23.373926​x2q\_{1}(x)=1.679044-10.844625x+23.373926x^{2} |
|  |  |  | 2 | q2​(x)=1.301562+1.301562​xq\_{2}(x)=1.301562+1.301562x |
|  |  |  | 2 | q3​(x)=1+1.073878​x+1.073878​x2q\_{3}(x)=1+1.073878x+1.073878x^{2} |

Table 5: Polynomial plans with CMAX=32 (the maximum absolute polynomial coefficient), INTERVAL\_PAD\_REL=0.001, and NOISE\_ABS=0.001. Total denotes the total number of symmetric polynomial products (SYPP) used by the plan, and the SYPP column reports the SYPP cost of each polynomial qkq\_{k}. The quantity souts\_{\mathrm{out}} is the certified final residual bound, i.e., an upper bound on ‖In−Kγ​Kγ−1~‖2\left\|{I}\_{n}-{K}\_{\gamma}\widetilde{{K}\_{\gamma}^{-1}}\right\|\_{2}.

## Appendix C CIFAR-10 Experiment

#### Dataset and model.

This appendix describes the CIFAR-10 (krizhevsky2009learning) experiment shown in the second row of Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ The Newton–Muon Optimizer"), where we report test accuracy versus training step and training time. The 50,000 training images are split into 45,000 training examples and 5,000 validation examples, and the standard 10,000-image test set is used for final evaluation. Training augmentation consists of random cropping with padding 4 and random horizontal flipping. All images are normalized channelwise using the standard CIFAR-10 mean and standard deviation. The model is a residual MLP with 32 hidden layers of width 512. Each hidden layer consists of a linear map followed by LayerNorm and GELU. Residual connections are used whenever the input and output dimensions match. Thus, the first layer 3072→5123072\to 512 has no skip connection, while the remaining 31 hidden layers of shape 512→512512\to 512 use residual additions. The output layer is a linear classifier from 512 to 10.

#### Training setup.

All three methods are trained for 100 epochs with batch size 4096 on an A100 GPU. The learning-rate schedule is linear warmup followed by cosine decay, with 100 warmup steps and minimum learning-rate ratio 0.1. Validation is evaluated every 24 training steps during hyperparameter tuning. For Muon and Newton–Muon, only the hidden-layer weight matrices are assigned to the matrix optimizer, while the remaining parameters are optimized by AdamW. For the pure AdamW baseline, all trainable parameters are optimized by AdamW. We tuned the hyperparameters of all methods on the validation split. We then retrained each method with its selected hyperparameters on the full 50,000-image CIFAR-10 training set and report test accuracy versus training step and training time in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ The Newton–Muon Optimizer").

#### Final hyperparameters.

The final AdamW baseline uses learning rate 8×10−48\times 10^{-4}, weight decay 10−210^{-2}, and (β1,β2)=(0.9,0.999)(\beta\_{1},\beta\_{2})=(0.9,0.999). The final Muon configuration uses AdamW on the non-matrix parameters with learning rate 1.6×10−31.6\times 10^{-3}, weight decay 10−210^{-2}, and (β1,β2)=(0.9,0.999)(\beta\_{1},\beta\_{2})=(0.9,0.999), and applies Muon to the hidden-layer weight matrices with matrix learning rate 0.160.16, matrix weight decay 10−310^{-3}, momentum 0.80.8. The final Newton–Muon configuration uses AdamW on the non-matrix parameters with learning rate 8×10−48\times 10^{-4}, weight decay 10−210^{-2}, and (β1,β2)=(0.9,0.999)(\beta\_{1},\beta\_{2})=(0.9,0.999), and applies Newton–Muon to the hidden-layer weight matrices with matrix learning rate 0.160.16, matrix weight decay 3×10−43\times 10^{-4}, momentum 0.750.75, together with EWMA β=0.95\beta=0.95, ridge γ=0.05\gamma=0.05, and refresh interval k=16k=16.

## Appendix D Kronecker-Factored Curvature

Here we derive the Kronecker-factored approximation in ([4](#S2.E4 "In 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")).
In a transformer with sequence length NN, let 𝒛t∈ℝn\boldsymbol{z}\_{t}\in\mathbb{R}^{n} denote the input activation at token tt and let 𝒚t∈ℝm\boldsymbol{y}\_{t}\in\mathbb{R}^{m} denote the output of a linear map 𝒚t=W​𝒛t\boldsymbol{y}\_{t}={W}\boldsymbol{z}\_{t} for t=1,…,Nt=1,\dots,N. For an update direction Q∈ℝm×n{Q}\in\mathbb{R}^{m\times n}, the output perturbation at token tt is Δ​𝒚t=Q​𝒛t\Delta\boldsymbol{y}\_{t}={Q}\boldsymbol{z}\_{t}.
To expose the Kronecker structure, first consider a single token loss Lt​(𝒚t)L\_{t}(\boldsymbol{y}\_{t}) with output-space curvature Ht≔∇𝒚t2Lt​(𝒚t)∈ℝm×m{H}\_{t}\coloneqq\nabla\_{\boldsymbol{y}\_{t}}^{2}L\_{t}(\boldsymbol{y}\_{t})\in\mathbb{R}^{m\times m}.
Then the exact second-order change satisfies

|  |  |  |
| --- | --- | --- |
|  | δ2​Lt=(Δ​𝒚t)⊤​Ht​(Δ​𝒚t)=(Q​𝒛t)⊤​Ht​(Q​𝒛t).\delta^{2}L\_{t}=(\Delta\boldsymbol{y}\_{t})^{\top}{H}\_{t}(\Delta\boldsymbol{y}\_{t})=({Q}\boldsymbol{z}\_{t})^{\top}{H}\_{t}({Q}\boldsymbol{z}\_{t}). |  |

Using vec​(Q​𝒛t)=(𝒛t⊤⊗Im)​vec​(Q)\mathrm{vec}({Q}\boldsymbol{z}\_{t})=(\boldsymbol{z}\_{t}^{\top}\otimes{I}\_{m})\mathrm{vec}({Q}), we obtain

|  |  |  |
| --- | --- | --- |
|  | (Q​𝒛t)⊤​Ht​(Q​𝒛t)=vec​(Q)⊤​((𝒛t​𝒛t⊤)⊗Ht)​vec​(Q).({Q}\boldsymbol{z}\_{t})^{\top}{H}\_{t}({Q}\boldsymbol{z}\_{t})=\mathrm{vec}({Q})^{\top}\Big((\boldsymbol{z}\_{t}\boldsymbol{z}\_{t}^{\top})\otimes{H}\_{t}\Big)\mathrm{vec}({Q}). |  |

For token-coupled losses, stack token outputs into 𝒚≔[𝒚1⊤,…,𝒚N⊤]⊤∈ℝm​N\boldsymbol{y}\coloneqq[\boldsymbol{y}\_{1}^{\top},\dots,\boldsymbol{y}\_{N}^{\top}]^{\top}\in\mathbb{R}^{mN},
and let the averaged scalar loss be f​(𝒚)≔L​(𝒚)/Nf(\boldsymbol{y})\coloneqq L(\boldsymbol{y})/N, where L​(𝒚)L(\boldsymbol{y}) denotes the summed loss over the NN tokens. The token-coupled output-space curvature of LL is

|  |  |  |
| --- | --- | --- |
|  | ℋ≔∇𝒚2L​(𝒚)∈ℝ(m​N)×(m​N).\mathscr{H}\coloneqq\nabla\_{\boldsymbol{y}}^{2}L(\boldsymbol{y})\in\mathbb{R}^{(mN)\times(mN)}. |  |

Write ℋ\mathscr{H} in N×NN\times N blocks of size m×mm\times m:

|  |  |  |
| --- | --- | --- |
|  | ℋ=[H11⋯H1​N⋮⋱⋮HN​1⋯HN​N],Ht​s∈ℝm×m.\mathscr{H}=\begin{bmatrix}{H}\_{11}&\cdots&{H}\_{1N}\\ \vdots&\ddots&\vdots\\ {H}\_{N1}&\cdots&{H}\_{NN}\end{bmatrix},\qquad{H}\_{ts}\in\mathbb{R}^{m\times m}. |  |

Then

|  |  |  |
| --- | --- | --- |
|  | δ2​f=1N​∑t=1N∑s=1N(Q​𝒛t)⊤​Ht​s​(Q​𝒛s).\delta^{2}f=\frac{1}{N}\sum\_{t=1}^{N}\sum\_{s=1}^{N}({Q}\boldsymbol{z}\_{t})^{\top}{H}\_{ts}({Q}\boldsymbol{z}\_{s}). |  |

Each term can be written as

|  |  |  |
| --- | --- | --- |
|  | (Q​𝒛t)⊤​Ht​s​(Q​𝒛s)=vec​(Q)⊤​((𝒛t​𝒛s⊤)⊗Ht​s)​vec​(Q),({Q}\boldsymbol{z}\_{t})^{\top}{H}\_{ts}({Q}\boldsymbol{z}\_{s})=\mathrm{vec}({Q})^{\top}\Big((\boldsymbol{z}\_{t}\boldsymbol{z}\_{s}^{\top})\otimes{H}\_{ts}\Big)\mathrm{vec}({Q}), |  |

so the exact parameter-space Hessian for the token-coupled loss is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℋW=1N∑t=1N∑s=1N(𝒛t𝒛s⊤)⊗Ht​s∈ℝ(m​n)×(m​n).{\mathcal{H}}\_{{W}}=\frac{1}{N}\sum\_{t=1}^{N}\sum\_{s=1}^{N}(\boldsymbol{z}\_{t}\boldsymbol{z}\_{s}^{\top})\otimes{H}\_{ts}\qquad\in\mathbb{R}^{(mn)\times(mn)}. |  | (21) |

Define the input activation matrix Z=[𝒛1,…,𝒛N]∈ℝn×N{Z}=[\boldsymbol{z}\_{1},\dots,\boldsymbol{z}\_{N}]\in\mathbb{R}^{n\times N},
the diagonal-block average curvature

|  |  |  |
| --- | --- | --- |
|  | H≔1N​∑t=1NHt​t.{H}\coloneqq\frac{1}{N}\sum\_{t=1}^{N}{H}\_{tt}. |  |

A simplifying approximation retains only diagonal-token curvature contributions and replaces Ht​t{H}\_{tt} by their average H{H},
yielding

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℋW≈1N​∑t=1N(𝒛t​𝒛t⊤)⊗H=(Z​Z⊤/N)⊗H.{\mathcal{H}}\_{{W}}\approx\frac{1}{N}\sum\_{t=1}^{N}(\boldsymbol{z}\_{t}\boldsymbol{z}\_{t}^{\top})\otimes{H}=({Z}{Z}^{\top}/N)\otimes{H}. |  | (22) |

#### Remark.

For architectures where loss decomposes across tokens/examples such as MLPs, the parameter-space Hessian ([21](#A4.E21 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")) is block-diagonal, i.e., Ht​s=𝟎{H}\_{ts}=\mathbf{0} for t≠st\neq s.
In this case, ([21](#A4.E21 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")) reduces to the diagonal-token form

|  |  |  |
| --- | --- | --- |
|  | ℋW=1N​∑t=1N(𝒛t​𝒛t⊤)⊗Ht​t,{\mathcal{H}}\_{{W}}=\frac{1}{N}\sum\_{t=1}^{N}(\boldsymbol{z}\_{t}\boldsymbol{z}\_{t}^{\top})\otimes{H}\_{tt}, |  |

so the only approximation in ([22](#A4.E22 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")) comes from replacing the varying blocks Ht​t{H}\_{tt} by their average H{H}.
As a result, the Kronecker estimator ([22](#A4.E22 "In Appendix D Kronecker-Factored Curvature ‣ The Newton–Muon Optimizer")) is expected to be more accurate when the loss is not token-coupled.

## Appendix E Theoretical Quadratic Score Under Isotropic Activation

This appendix develops the theoretical quadratic score analysis under an isotropic activation second moment, deriving explicit formulas and approximations for gradient descent, Muon, Newton–Muon, and Newton, and then using them to interpret the corresponding numerical study.

### E.1 Score Formulas for Different Directions

We use ([19](#S4.E19 "In 4.1 Quadratic Score ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer")) to compare four update directions under the quadratic surrogate:
(i) the raw GD direction Q=G{Q}={G}, (ii) the Muon direction QMuon=msgn​(G){Q}\_{\mathrm{Muon}}=\mathrm{msgn}({G}), (iii) the Newton–Muon direction QNewton​-​Muon=msgn​(G​(Z​Z⊤)−1){Q}\_{\mathrm{Newton\text{-}Muon}}=\mathrm{msgn}\big({G}({Z}{Z}^{\top})^{-1}\big), and (iv) the Newton direction QNewton=H−1​G​(Z​Z⊤)−1{Q}\_{\mathrm{Newton}}={H}^{-1}{G}({Z}{Z}^{\top})^{-1}.

#### GD direction.

For Q=G{Q}={G}, the numerator is tr​(G​G⊤)=‖G‖F2\mathrm{tr}({G}{G}^{\top})=\|{G}\|\_{F}^{2} and the denominator is
tr​(H​G​(Z​Z⊤/N)​G⊤)\mathrm{tr}\big({H}{G}({Z}{Z}^{\top}/N){G}^{\top}\big). Using G=H​(W−W⋆)​(Z​Z⊤/N){G}={H}({W}-W^{\star})({Z}{Z}^{\top}/N),
we have

|  |  |  |
| --- | --- | --- |
|  | tr​(G​G⊤)=tr​(H2​(W−W⋆)​(Z​Z⊤/N)2​(W−W⋆)⊤).\mathrm{tr}({G}{G}^{\top})=\mathrm{tr}\Big({H}^{2}({W}-W^{\star})({Z}{Z}^{\top}/N)^{2}({W}-W^{\star})^{\top}\Big). |  |

Similarly,

|  |  |  |
| --- | --- | --- |
|  | tr​(H​G​(Z​Z⊤/N)​G⊤)=tr​(H3​(W−W⋆)​(Z​Z⊤/N)3​(W−W⋆)⊤).\mathrm{tr}\big({H}{G}({Z}{Z}^{\top}/N){G}^{\top}\big)=\mathrm{tr}\Big({H}^{3}({W}-W^{\star})({Z}{Z}^{\top}/N)^{3}({W}-W^{\star})^{\top}\Big). |  |

Hence the score for the GD direction has the closed form

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(G)=tr​(H2​(W−W⋆)​(Z​Z⊤/N)2​(W−W⋆)⊤)2tr​(H3​(W−W⋆)​(Z​Z⊤/N)3​(W−W⋆)⊤).s({G})=\frac{\mathrm{tr}\Big({H}^{2}({W}-W^{\star})({Z}{Z}^{\top}/N)^{2}({W}-W^{\star})^{\top}\Big)^{2}}{\mathrm{tr}\Big({H}^{3}({W}-W^{\star})({Z}{Z}^{\top}/N)^{3}({W}-W^{\star})^{\top}\Big)}. |  | (23) |

#### Muon direction.

We consider the matrix sign QMuon≔msgn​(G){Q}\_{\mathrm{Muon}}\coloneqq\mathrm{msgn}(G). For G=U​S​V⊤G=USV^{\top},

|  |  |  |
| --- | --- | --- |
|  | tr​(QMuon​G⊤)=tr​(U​V⊤​(V​S​U⊤))=tr​(S)=‖G‖∗=‖H​(W−W⋆)​(Z​Z⊤/N)‖∗,\mathrm{tr}({Q}\_{\mathrm{Muon}}{G}^{\top})=\mathrm{tr}\big({U}{V}^{\top}({V}S{U}^{\top})\big)=\mathrm{tr}(S)=\|{G}\|\_{\ast}=\big\|{H}({W}-W^{\star})({Z}{Z}^{\top}/N)\big\|\_{\ast}, |  |

and the denominator becomes

|  |  |  |
| --- | --- | --- |
|  | tr​(H​QMuon​(Z​Z⊤/N)​QMuon⊤)=tr​(H​U​V⊤​(Z​Z⊤/N)​V​U⊤)=tr​(U⊤​H​U⋅V⊤​(Z​Z⊤/N)​V).\mathrm{tr}\Big({H}{Q}\_{\mathrm{Muon}}({Z}{Z}^{\top}/N){Q}\_{\mathrm{Muon}}^{\top}\Big)=\mathrm{tr}\Big({H}{U}{V}^{\top}({Z}{Z}^{\top}/N){V}{U}^{\top}\Big)=\mathrm{tr}\Big({U}^{\top}{H}{U}\cdot{V}^{\top}({Z}{Z}^{\top}/N){V}\Big). |  |

Therefore, the Muon-direction score is

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(QMuon)=‖H​(W−W⋆)​(Z​Z⊤/N)‖∗2tr​(U⊤​H​U⋅V⊤​(Z​Z⊤/N)​V).s({Q}\_{\mathrm{Muon}})=\frac{\big\|{H}({W}-W^{\star})({Z}{Z}^{\top}/N)\big\|\_{\ast}^{2}}{\mathrm{tr}\Big({U}^{\top}{H}{U}\cdot{V}^{\top}({Z}{Z}^{\top}/N){V}\Big)}. |  | (24) |

The score for Newton–Muon can be derived similarly. We do not analyze it theoretically here; instead, we evaluate its predicted score numerically and compare it with Muon in our numerical study.

#### Newton direction.

The Newton step is QNewton∝H−1​G​(Z​Z⊤)−1{Q}\_{\mathrm{Newton}}\propto{H}^{-1}{G}({Z}{Z}^{\top})^{-1}. Substituting into
([19](#S4.E19 "In 4.1 Quadratic Score ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer")) gives

|  |  |  |
| --- | --- | --- |
|  | tr​(QNewton​G⊤)=tr​(H−1​G​(Z​Z⊤)−1​G⊤),\mathrm{tr}({Q}\_{\mathrm{Newton}}{G}^{\top})=\mathrm{tr}\big({H}^{-1}{G}({Z}{Z}^{\top})^{-1}{G}^{\top}\big), |  |

and, using H​QNewton​(Z​Z⊤/N)=G/N{H}{Q}\_{\mathrm{Newton}}({Z}{Z}^{\top}/N)={G}/N,

|  |  |  |
| --- | --- | --- |
|  | tr​(H​QNewton​(Z​Z⊤/N)​QNewton⊤)=1N​tr​(G​QNewton⊤)=1N​tr​(QNewton​G⊤).\mathrm{tr}\Big({H}{Q}\_{\mathrm{Newton}}({Z}{Z}^{\top}/N){Q}\_{\mathrm{Newton}}^{\top}\Big)=\frac{1}{N}\mathrm{tr}\big({G}{Q}\_{\mathrm{Newton}}^{\top}\big)=\frac{1}{N}\mathrm{tr}({Q}\_{\mathrm{Newton}}{G}^{\top}). |  |

Therefore, s​(QNewton)=N​tr​(QNewton​G⊤)s({Q}\_{\mathrm{Newton}})=N\mathrm{tr}({Q}\_{\mathrm{Newton}}{G}^{\top}).
Under G=H​(W−W⋆)​(Z​Z⊤/N){G}={H}({W}-W^{\star})({Z}{Z}^{\top}/N), this can be rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(QNewton)=tr​(H​(W−W⋆)​(Z​Z⊤/N)​(W−W⋆)⊤).s({Q}\_{\mathrm{Newton}})=\mathrm{tr}\Big({H}({W}-W^{\star})({Z}{Z}^{\top}/N)({W}-W^{\star})^{\top}\Big). |  | (25) |

### E.2 Isotropic Baseline Numerical Study

From ([23](#A5.E23 "In GD direction. ‣ E.1 Score Formulas for Different Directions ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), ([24](#A5.E24 "In Muon direction. ‣ E.1 Score Formulas for Different Directions ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), and ([25](#A5.E25 "In Newton direction. ‣ E.1 Score Formulas for Different Directions ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), the quadratic score
([19](#S4.E19 "In 4.1 Quadratic Score ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer")) yields explicit expressions for the GD, Muon, and Newton directions. By using different distributions on H{H}, Z{Z}, and (W−W⋆)({W}-W^{\star}), we can evaluate how
anisotropy in curvature, activations, and displacement interacts to favor different update geometries, either analytically when expectations simplify or
numerically by Monte Carlo simulation.
Here we study the simplest baseline in which both the activation and the displacement are isotropic Gaussian; note that the same framework can be reused under more structured, non-isotropic
specifications by changing only the distribution for H{H}, Z{Z}, and (W−W⋆)({W}-W^{\star}).

#### Assumptions.

First, we sample Z{Z} with i.i.d. standard normal entries. For the theoretical analysis, we assume the activation second moment is isotropic (while in the simulations we use a finite sample size):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Z​Z⊤/N=In.{Z}{Z}^{\top}/N={I}\_{n}. |  | (26) |

Second, we assume the square case m=nm=n. Third, we further assume the displacement matrix W−W⋆{W}-W^{\star} has i.i.d. standard normal entries. Under ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), G=H​(W−W⋆){G}={H}({W}-W^{\star}) and the score ([19](#S4.E19 "In 4.1 Quadratic Score ‣ 4 One-Step Analysis of Newton–Muon ‣ The Newton–Muon Optimizer")) simplifies to s​(Q)=tr​(Q​G⊤)2/tr​(H​Q​Q⊤)s({Q})=\mathrm{tr}({Q}{G}^{\top})^{2}/\mathrm{tr}\big({H}{Q}{Q}^{\top}\big). Let DH≔diag​(λ1,…,λm){D}\_{{H}}\coloneqq\mathrm{diag}(\lambda\_{1},\dots,\lambda\_{m}). We fix this diagonal spectrum (to analyze different levels of curvature anisotropy through the choice of {λk}\{\lambda\_{k}\}), then sample a random orthogonal matrix P∈ℝm×m{P}\in\mathbb{R}^{m\times m} and define H≔P​DH​P⊤{H}\coloneqq{P}{D}\_{{H}}{P}^{\top}.

#### GD score.

Under ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), ([23](#A5.E23 "In GD direction. ‣ E.1 Score Formulas for Different Directions ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) reduces to

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(G)=tr​(H2​(W−W⋆)​(W−W⋆)⊤)2tr​(H3​(W−W⋆)​(W−W⋆)⊤).s({G})=\frac{\mathrm{tr}\Big({H}^{2}({W}-W^{\star})({W}-W^{\star})^{\top}\Big)^{2}}{\mathrm{tr}\Big({H}^{3}({W}-W^{\star})({W}-W^{\star})^{\top}\Big)}. |  | (27) |

Moreover, for k∈{2,3}k\in\{2,3\},

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[tr​(Hk​(W−W⋆)​(W−W⋆)⊤)]=tr​(Hk​𝔼​[(W−W⋆)​(W−W⋆)⊤])=n​tr​(Hk).\mathbb{E}\left[\mathrm{tr}\Big({H}^{k}({W}-W^{\star})({W}-W^{\star})^{\top}\Big)\right]=\mathrm{tr}\Big({H}^{k}\mathbb{E}\big[({W}-W^{\star})({W}-W^{\star})^{\top}\big]\Big)=n\mathrm{tr}({H}^{k}). |  |

In the large-nn regime, the random traces above concentrate around their means, so a standard
spectrum-level approximation to ([27](#A5.E27 "In GD score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) is

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(G)≈n​tr​(H2)2tr​(H3)=n​(∑j=1mλj2)2∑j=1mλj3.s({G})\approx n\frac{\mathrm{tr}({H}^{2})^{2}}{\mathrm{tr}({H}^{3})}=n\frac{\left(\sum\_{j=1}^{m}\lambda\_{j}^{2}\right)^{2}}{\sum\_{j=1}^{m}\lambda\_{j}^{3}}. |  | (28) |

#### Muon score.

In the square full-rank case, U∈ℝm×m{U}\in\mathbb{R}^{m\times m} is orthogonal, so tr​(U⊤​H​U)=tr​(H)\mathrm{tr}({U}^{\top}{H}{U})=\mathrm{tr}({H}). Thus

|  |  |  |  |
| --- | --- | --- | --- |
|  | s​(QMuon)=‖H​(W−W⋆)‖∗2tr​(U⊤​H​U)=‖H​(W−W⋆)‖∗2tr​(H).s({Q}\_{\mathrm{Muon}})=\frac{\|{H}({W}-W^{\star})\|\_{\ast}^{2}}{\mathrm{tr}({U}^{\top}{H}{U})}=\frac{\|{H}({W}-W^{\star})\|\_{\ast}^{2}}{\mathrm{tr}({H})}. |  | (29) |

While 𝔼​‖H​(W−W⋆)‖∗2\mathbb{E}\|{H}({W}-W^{\star})\|\_{\ast}^{2} does not simplify to a closed form in general,
it admits an accurate high-dimensional deterministic approximation that depends only on the spectrum of H{H}
and can be computed numerically. Let

|  |  |  |
| --- | --- | --- |
|  | S≔1m​H​(W−W⋆)​(W−W⋆)⊤​H∈ℝm×m.{S}\coloneqq\frac{1}{m}{H}({W}-W^{\star})({W}-W^{\star})^{\top}{H}\in\mathbb{R}^{m\times m}. |  |

The eigenvalues of S{S} are the squared singular values of H​(W−W⋆){H}({W}-W^{\star}) divided by mm.
Consequently,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖H​(W−W⋆)‖∗=∑i=1mσi​(H​(W−W⋆))=m​∑i=1mλi​(S).\|{H}({W}-W^{\star})\|\_{\ast}=\sum\_{i=1}^{m}\sigma\_{i}\big({H}({W}-W^{\star})\big)=\sqrt{m}\sum\_{i=1}^{m}\sqrt{\lambda\_{i}({S})}. |  | (30) |

In the regime m=n→∞m=n\to\infty with H{H} deterministic and W−W⋆{W}-W^{\star} having i.i.d. standard normal entries,
Silverstein–Choi’s analysis of sample-covariance type matrices implies that the empirical spectral distribution of S{S}
converges to a nonrandom limiting law characterized by a fixed-point equation for its Stieltjes transform
(silverstein1995empirical). In particular, writing the eigenvalues of H{H} as {λk}k=1m\{\lambda\_{k}\}\_{k=1}^{m}, the Stieltjes transform mS​(z)m\_{S}(z) of the limiting law is the unique solution in the
upper half-plane to the equation

|  |  |  |  |
| --- | --- | --- | --- |
|  | mS​(z)=−1z⋅1m​∑k=1m11+λk2​mS​(z),z∈ℂ+.m\_{S}(z)=-\frac{1}{z}\cdot\frac{1}{m}\sum\_{k=1}^{m}\frac{1}{1+\lambda\_{k}^{2}m\_{S}(z)},\qquad z\in\mathbb{C}\_{+}. |  | (31) |

Given mS​(z)m\_{S}(z), the limiting spectral density ρS​(x)\rho\_{S}(x) is obtained by the standard inversion formula

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρS​(x)=limη↓01π​ℑ⁡mS​(x+i​η),x>0.\rho\_{S}(x)=\lim\_{\eta\downarrow 0}\frac{1}{\pi}\Im m\_{S}(x+i\eta),\qquad x>0. |  | (32) |

Equations ([31](#A5.E31 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) and ([32](#A5.E32 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) suggest a concrete numerical pipeline:

1. 1.

   Choose a small η>0\eta>0 and a grid of xx values covering the support of the spectrum of S{S}.
2. 2.

   For each z=x+i​ηz=x+i\eta, solve the fixed-point equation ([31](#A5.E31 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")).
3. 3.

   Approximate ρS​(x)≈(1/π)​ℑ⁡mS​(x+i​η)\rho\_{S}(x)\approx(1/\pi)\Im m\_{S}(x+i\eta).
4. 4.

   Compute the limiting mean singular value of H​(W−W⋆){H}({W}-W^{\star}) via

   |  |  |  |
   | --- | --- | --- |
   |  | μ1/2≔∫0∞x​ρS​(x)​dx,\mu\_{1/2}\coloneqq\int\_{0}^{\infty}\sqrt{x}\rho\_{S}(x)\mathrm{d}x, |  |

   using numerical quadrature on the grid.
5. 5.

   Using ([30](#A5.E30 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), the leading-order scaling is

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ‖H​(W−W⋆)‖∗≈m3/2​μ1/2,hence𝔼​‖H​(W−W⋆)‖∗2≈m3​μ1/22,\|{H}({W}-W^{\star})\|\_{\ast}\approx m^{3/2}\mu\_{1/2},\qquad\text{hence}\qquad\mathbb{E}\|{H}({W}-W^{\star})\|\_{\ast}^{2}\approx m^{3}\mu\_{1/2}^{2}, |  | (33) |

   where the approximation captures the dominant m3m^{3} growth and depends only on the spectrum of H{H} through ([31](#A5.E31 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")).

Substituting ([33](#A5.E33 "In item 5 ‣ Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) into ([29](#A5.E29 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) yields a corresponding practical approximation for the
expected Muon score,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​s​(QMuon)≈𝔼​‖H​(W−W⋆)‖∗2tr​(H)≈m3​μ1/22tr​(H).\mathbb{E}s({Q}\_{\mathrm{Muon}})\approx\frac{\mathbb{E}\|{H}({W}-W^{\star})\|\_{\ast}^{2}}{\mathrm{tr}({H})}\approx\frac{m^{3}\mu\_{1/2}^{2}}{\mathrm{tr}({H})}. |  | (34) |

We will use ([31](#A5.E31 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer"))–([34](#A5.E34 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) as the basis for numerically evaluating the Muon
numerator (and thus the score) from the eigenvalues of H{H}.

#### Newton–Muon score.

Under ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), Newton–Muon coincides exactly with Muon:

|  |  |  |
| --- | --- | --- |
|  | QNewton​-​Muon=msgn​(G​(Z​Z⊤)−1)=msgn​(G)=QMuon.{Q}\_{\mathrm{Newton\text{-}Muon}}=\mathrm{msgn}\big({G}({Z}{Z}^{\top})^{-1}\big)=\mathrm{msgn}({G})={Q}\_{\mathrm{Muon}}. |  |

Consequently, we use the same approximation

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​s​(QNewton​-​Muon)≈m3​μ1/22tr​(H).\mathbb{E}s({Q}\_{\mathrm{Newton\text{-}Muon}})\approx\frac{m^{3}\mu\_{1/2}^{2}}{\mathrm{tr}({H})}. |  | (35) |

#### Newton score.

Applying ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) to ([25](#A5.E25 "In Newton direction. ‣ E.1 Score Formulas for Different Directions ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) yields

|  |  |  |
| --- | --- | --- |
|  | s​(QNewton)=tr​(H​(W−W⋆)​(W−W⋆)⊤).s({Q}\_{\mathrm{Newton}})=\mathrm{tr}\Big({H}({W}-W^{\star})({W}-W^{\star})^{\top}\Big). |  |

Taking expectation gives the exact identity

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[s​(QNewton)]=tr​(H⋅n​Im)=n​tr​(H)=n​∑k=1mλk.\mathbb{E}\left[s({Q}\_{\mathrm{Newton}})\right]=\mathrm{tr}\Big({H}\cdot n{I}\_{m}\Big)=n\mathrm{tr}({H})=n\sum\_{k=1}^{m}\lambda\_{k}. |  | (36) |

#### Comparison.

Under the simplifying assumptions ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) and m=nm=n, we can compare the three directions at the level of the
theoretical quantities already derived for Newton, GD, Muon, and Newton–Muon. Newton has an exact expected score
([36](#A5.E36 "In Newton score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")):

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[s​(QNewton)]=n​tr​(H)=n​∑k=1mλk.\mathbb{E}\left[s({Q}\_{\mathrm{Newton}})\right]=n\mathrm{tr}({H})=n\sum\_{k=1}^{m}\lambda\_{k}. |  |

For the GD direction, the spectrum-level approximation ([28](#A5.E28 "In GD score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) gives

|  |  |  |
| --- | --- | --- |
|  | s​(G)≈n​tr​(H2)2tr​(H3)=n​(∑k=1mλk2)2∑k=1mλk3.s({G})\approx n\frac{\mathrm{tr}({H}^{2})^{2}}{\mathrm{tr}({H}^{3})}=n\frac{\left(\sum\_{k=1}^{m}\lambda\_{k}^{2}\right)^{2}}{\sum\_{k=1}^{m}\lambda\_{k}^{3}}. |  |

For nonnegative eigenvalues λk≥0\lambda\_{k}\geq 0, apply Cauchy inequality to the two sequences λk3/2\lambda\_{k}^{3/2} and λk1/2\lambda\_{k}^{1/2}:

|  |  |  |
| --- | --- | --- |
|  | (∑kλk2)2∑kλk3≤∑kλk⟹s​(G)≲n​∑kλk=𝔼​[s​(QNewton)].\frac{\left(\sum\_{k}\lambda\_{k}^{2}\right)^{2}}{\sum\_{k}\lambda\_{k}^{3}}\leq\sum\_{k}\lambda\_{k}\quad\Longrightarrow\quad s({G})\lesssim n\sum\_{k}\lambda\_{k}=\mathbb{E}\left[s({Q}\_{\mathrm{Newton}})\right]. |  |

Equality in Cauchy holds if and only if λk\lambda\_{k} is constant across kk. Thus, GD matches Newton only when H∝Im{H}\propto{I}\_{m}.

Newton–Muon and Muon also simplify in this equal-eigenvalues case, and its spectrum-only approximation ([34](#A5.E34 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) becomes
fully explicit. When H=λ​Im{H}=\lambda{I}\_{m} and m=nm=n, the limiting law of
S=(1/m)​H​(W−W⋆)​(W−W⋆)⊤​H{S}=(1/m){H}({W}-W^{\star})({W}-W^{\star})^{\top}{H}
is λ2\lambda^{2} times the Marchenko–Pastur law at aspect ratio 11, which yields the closed form

|  |  |  |
| --- | --- | --- |
|  | μ1/2=83​π​λ.\mu\_{1/2}=\frac{8}{3\pi}\lambda. |  |

Substituting into ([34](#A5.E34 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) gives

|  |  |  |
| --- | --- | --- |
|  | 𝔼​s​(QMuon)≈m3​μ1/22tr​(H)=m3​λ2​(83​π)2m​λ=(649​π2)​m2​λ=(649​π2)​𝔼​[s​(QNewton)],\mathbb{E}s({Q}\_{\mathrm{Muon}})\approx\frac{m^{3}\mu\_{1/2}^{2}}{\mathrm{tr}({H})}=\frac{m^{3}\lambda^{2}\left(\frac{8}{3\pi}\right)^{2}}{m\lambda}=\left(\frac{64}{9\pi^{2}}\right)m^{2}\lambda=\left(\frac{64}{9\pi^{2}}\right)\mathbb{E}\left[s({Q}\_{\mathrm{Newton}})\right], |  |

so the isotropic asymptotic approximation predicts that Muon achieves a constant fraction of the Newton score (about 0.720.72), while GD coincides with Newton.

Beyond H=λ​Im{H}=\lambda{I}\_{m}, the Muon and Newton–Muon approximation ([34](#A5.E34 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) and ([35](#A5.E35 "In Newton–Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) still depend only on the
spectrum of H{H} through μ1/2\mu\_{1/2} (computed from ([31](#A5.E31 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) and ([32](#A5.E32 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer"))),
but there is no longer a simple closed-form comparison between the Muon/Newton–Muon score ([34](#A5.E34 "In Muon score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")) and the GD
approximation ([28](#A5.E28 "In GD score. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), as we will show next in our numerical study.

Under the isotropic-input theory ([26](#A5.E26 "In Assumptions. ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")), the Newton–Muon and Muon theoretical predictions coincide exactly; any empirical separation between them comes solely from finite-sample deviations of Z​Z⊤/N{Z}{Z}^{\top}/N from In{I}\_{n}.
In all experiments below we fix λmin=10−4\lambda\_{\min}=10^{-4} and vary only (N,p)(N,p). For each setting, we run 10241024 independent simulations and report the mean score together with the 2.5%–97.5% interval. The theory marker in the plot denotes the theoretical score in the limit of infinite data and infinite dimensions m,n→∞m,n\to\infty.

#### Baseline (N=8192N=8192, p=0.3p=0.3).

Figure [8](#A5.F8 "Figure 8 ‣ Baseline (𝑁=8192, 𝑝=0.3). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer") shows a clearly anisotropic spectrum of H{H}.
In this ill-conditioned regime, both Muon and Newton–Muon yield substantial score improvements over the raw
GD direction, and Newton–Muon is slightly better than Muon.

![Refer to caption](/html/2604.01472/assets/x9.png)

![Refer to caption](/html/2604.01472/assets/x10.png)

Figure 8: Baseline configuration (N=8192N=8192, p=0.3p=0.3): spectrum of H{H} (left) and mean absolute
scores s​(Q)s({Q}) for GD, Muon, Newton–Muon, and Newton (right).

#### Uniform curvature (N=8192N=8192, p=2.4p=2.4).

Figure [9](#A5.F9 "Figure 9 ‣ Uniform curvature (𝑁=8192, 𝑝=2.4). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer") corresponds to a spectrum that is nearly flat at the top and drops sharply only near the tail.
In this more uniform-curvature setting, the score gaps among GD, Muon, and Newton–Muon narrow, indicating that Newton–Muon and Muon help when H{H} is strongly anisotropic. Here, Newton–Muon is still slightly better than Muon.

![Refer to caption](/html/2604.01472/assets/x11.png)

![Refer to caption](/html/2604.01472/assets/x12.png)

Figure 9: More top-uniform curvature (N=8192N=8192, p=2.4p=2.4): spectrum of H{H} (left) and mean absolute scores (right).

#### Smaller data (N=1024N=1024, p=0.3p=0.3).

Figure [10](#A5.F10 "Figure 10 ‣ Smaller data (𝑁=1024, 𝑝=0.3). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer") keeps the same curvature shape as the baseline but uses a smaller sample size NN to form Z​Z⊤/N{Z}{Z}^{\top}/N, making the empirical second moment noisier. Relative to Figure [8](#A5.F8 "Figure 8 ‣ Baseline (𝑁=8192, 𝑝=0.3). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer"), the gap between Newton–Muon and Muon becomes much larger, suggesting that Newton–Muon more effectively compensates for activation anisotropy when the sample size is small.

![Refer to caption](/html/2604.01472/assets/x13.png)

![Refer to caption](/html/2604.01472/assets/x14.png)

Figure 10: Smaller-NN data (N=1024N=1024, p=0.3p=0.3): spectrum of H{H} (left) and mean absolute scores (right).

#### Conclusion.

Across these three runs, Muon and Newton–Muon achieve their largest score gains when curvature is strongly anisotropic (Figure [8](#A5.F8 "Figure 8 ‣ Baseline (𝑁=8192, 𝑝=0.3). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")); when H{H} becomes more uniform, the benefit shrinks or even vanishes (Figure [9](#A5.F9 "Figure 9 ‣ Uniform curvature (𝑁=8192, 𝑝=2.4). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")). Newton–Muon outperforms Muon in these three experiments, and especially when NN is reduced so that Z​Z⊤/N{Z}{Z}^{\top}/N is less isotropic, the advantage of Newton–Muon over Muon further increases (Figure [10](#A5.F10 "Figure 10 ‣ Smaller data (𝑁=1024, 𝑝=0.3). ‣ E.2 Isotropic Baseline Numerical Study ‣ Appendix E Theoretical Quadratic Score Under Isotropic Activation ‣ The Newton–Muon Optimizer")).

## Appendix F Non-Isotropic Assumption

Here we report attempts to replace the isotropic assumption ΣW∝Im\Sigma\_{W}\propto I\_{m} by estimating ΣW\Sigma\_{W} from training dynamics. ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) motivates the use of a non-isotropic ΣW{\Sigma}\_{W}. When ΣW{\Sigma}\_{W} admits a factorization of the form ΣW=M​M⊤{\Sigma}\_{W}=MM^{\top}, the update in ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")) can be implemented without explicitly constructing ΣW1/2\Sigma\_{W}^{1/2}.

###### Proposition 3 (Factorized form of ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer"))).

Suppose ΣW=M​M⊤{\Sigma}\_{{W}}={M}{M}^{\top} for some M∈ℝm×r{M}\in\mathbb{R}^{m\times r}. Then

|  |  |  |
| --- | --- | --- |
|  | Q⋆=M​msgn​(M⊤​G​(Z​Z⊤)−1).{Q}^{\star}={M}\mathrm{msgn}\Big({M}^{\top}{G}({Z}{Z}^{\top})^{-1}\Big). |  |

###### Proof of Proposition [3](#Thmproposition3 "Proposition 3 (Factorized form of (5)). ‣ Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer").

Let k≔rank​(M)k\coloneqq\mathrm{rank}(M) and take a compact SVD M=UM​SM​VM⊤M=U\_{M}S\_{M}V\_{M}^{\top}, where UM∈ℝm×kU\_{M}\in\mathbb{R}^{m\times k} and VM∈ℝr×kV\_{M}\in\mathbb{R}^{r\times k} have orthonormal columns and SM∈ℝk×kS\_{M}\in\mathbb{R}^{k\times k} is diagonal with positive entries. Then ΣW=M​M⊤=UM​SM2​UM⊤\Sigma\_{W}=MM^{\top}=U\_{M}S\_{M}^{2}U\_{M}^{\top}, so ΣW1/2=UM​SM​UM⊤\Sigma\_{W}^{1/2}=U\_{M}S\_{M}U\_{M}^{\top}.
Hence

|  |  |  |
| --- | --- | --- |
|  | ΣW1/2​G​(Z​Z⊤)−1=UM​SM​UM⊤​G​(Z​Z⊤)−1.\Sigma\_{W}^{1/2}G(ZZ^{\top})^{-1}=U\_{M}S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}. |  |

Using the compact-SVD definition of the rectangular matrix sign, if UMU\_{M} has orthonormal columns, then msgn​(UM​X)=UM​msgn​(X)\mathrm{msgn}(U\_{M}X)=U\_{M}\mathrm{msgn}(X) for any matrix XX. Applying this with X=SM​UM⊤​G​(Z​Z⊤)−1X=S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1} gives

|  |  |  |
| --- | --- | --- |
|  | msgn​(ΣW1/2​G​(Z​Z⊤)−1)=UM​msgn​(SM​UM⊤​G​(Z​Z⊤)−1).\mathrm{msgn}\Big(\Sigma\_{W}^{1/2}G(ZZ^{\top})^{-1}\Big)=U\_{M}\mathrm{msgn}\Big(S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}\Big). |  |

Substituting into ([5](#S2.E5 "In Proposition 1. ‣ 2.2 Minimization of the Triplet Model ‣ 2 Derivation of Newton–Muon ‣ The Newton–Muon Optimizer")), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | Q⋆=ΣW1/2​msgn​(ΣW1/2​G​(Z​Z⊤)−1)=UM​SM​msgn​(SM​UM⊤​G​(Z​Z⊤)−1).Q^{\star}=\Sigma\_{W}^{1/2}\mathrm{msgn}\Big(\Sigma\_{W}^{1/2}G(ZZ^{\top})^{-1}\Big)=U\_{M}S\_{M}\mathrm{msgn}\Big(S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}\Big). |  | (37) |

On the other hand, M⊤​G​(Z​Z⊤)−1=VM​SM​UM⊤​G​(Z​Z⊤)−1M^{\top}G(ZZ^{\top})^{-1}=V\_{M}S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}. Applying the same identity with VMV\_{M} yields

|  |  |  |
| --- | --- | --- |
|  | msgn​(M⊤​G​(Z​Z⊤)−1)=VM​msgn​(SM​UM⊤​G​(Z​Z⊤)−1).\mathrm{msgn}\Big(M^{\top}G(ZZ^{\top})^{-1}\Big)=V\_{M}\mathrm{msgn}\Big(S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}\Big). |  |

Therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | M​msgn​(M⊤​G​(Z​Z⊤)−1)\displaystyle M\mathrm{msgn}\Big(M^{\top}G(ZZ^{\top})^{-1}\Big) | =(UM​SM​VM⊤)​(VM​msgn​(SM​UM⊤​G​(Z​Z⊤)−1))\displaystyle=(U\_{M}S\_{M}V\_{M}^{\top})\Big(V\_{M}\mathrm{msgn}\big(S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}\big)\Big) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =UM​SM​msgn​(SM​UM⊤​G​(Z​Z⊤)−1),\displaystyle=U\_{M}S\_{M}\mathrm{msgn}\Big(S\_{M}U\_{M}^{\top}G(ZZ^{\top})^{-1}\Big), |  |

which matches ([37](#A6.E37 "In Proof of Proposition 3. ‣ Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer")).
∎

Motivated by this identity, we explored practical estimators of M{M} in benchmark training runs, including momentum-based and diagonal variants derived from update statistics. However, these non-isotropic variants did not yield consistent improvements over Newton–Muon.

#### Keeping a momentum buffer of the actual update.

We maintain a momentum buffer of the actual parameter updates, denoted by M{M}.
Heuristically, if the update directions tend to point toward an optimum W⋆W^{\star}, then this buffer can be viewed as a coarse proxy of
W−W⋆{W}-W^{\star}.
In practice, we formed two candidate directions

|  |  |  |
| --- | --- | --- |
|  | QI=msgn​(G​(Z​Z⊤)−1),QM=M​msgn​(M⊤​G​(Z​Z⊤)−1),{Q}\_{I}=\mathrm{msgn}\big({G}({Z}{Z}^{\top})^{-1}\big),\qquad{Q}\_{M}={M}\mathrm{msgn}\Big({M}^{\top}{G}({Z}{Z}^{\top})^{-1}\Big), |  |

and then combined them by a convex average after normalizing their scales

|  |  |  |
| --- | --- | --- |
|  | Q=(1−λ)​QI‖QI‖F+λ​QM‖QM‖F,λ∈[0,1].{Q}=(1-\lambda)\frac{{Q}\_{I}}{\|{Q}\_{I}\|\_{F}}+\lambda\frac{{Q}\_{M}}{\|{Q}\_{M}\|\_{F}},\qquad\lambda\in[0,1]. |  |

Despite the motivation from Proposition [3](#Thmproposition3 "Proposition 3 (Factorized form of (5)). ‣ Appendix F Non-Isotropic Assumption ‣ The Newton–Muon Optimizer"), this averaging did not yield consistent gains, and typically underperformed
the Newton–Muon update QI{Q}\_{I}.

#### Using diagonal estimation of the covariance from update statistics.

We also tried a computationally cheaper approximation that estimates a diagonal ΣW{\Sigma}\_{{W}} from the second moment of the actual update directions. Concretely, for each matrix parameter W∈ℝm×n{W}\in\mathbb{R}^{m\times n} we maintain a state vector
𝒖∈ℝm\boldsymbol{u}\in\mathbb{R}^{m} that tracks the row-wise magnitudes of
the applied Muon directions via an EWMA:

|  |  |  |
| --- | --- | --- |
|  | 𝒖←β​𝒖+(1−β)​‖Q‖row,\boldsymbol{u}\leftarrow\beta\boldsymbol{u}+(1-\beta)\big\|{Q}\big\|\_{\text{row}}, |  |

where ∥⋅∥row\|\cdot\|\_{\text{row}} denotes the ℓ2\ell\_{2} norm along the column dimension.
We then form a damped diagonal estimate of ΣW{\Sigma}\_{{W}} using the squared row magnitudes,

|  |  |  |
| --- | --- | --- |
|  | 𝒖⊙2≈diag​(ΣW),M2=diag​(λ​𝒖⊙2+(1−λ)​𝒖⊙2¯​ 1m),\boldsymbol{u}^{\odot 2}\ \approx\ \mathrm{diag}({\Sigma}\_{{W}}),\qquad{M}^{2}=\mathrm{diag}\Big(\lambda\boldsymbol{u}^{\odot 2}+(1-\lambda)\overline{\boldsymbol{u}^{\odot 2}}\,\mathbf{1}\_{m}\Big), |  |

where 𝒖⊙2¯\overline{\boldsymbol{u}^{\odot 2}} is the mean of the entries of 𝒖⊙2\boldsymbol{u}^{\odot 2}, λ∈[0,1]\lambda\in[0,1] is a damping coefficient, and 𝟏m∈ℝm\mathbf{1}\_{m}\in\mathbb{R}^{m} is the all-ones vector. Let M{M} be the positive diagonal square root of M2{M}^{2}. The resulting update takes the form

|  |  |  |
| --- | --- | --- |
|  | QM∝M​msgn​(M​G​(Z​Z⊤)−1).{Q}\_{M}\propto{M}\mathrm{msgn}\Big({M}{G}({Z}{Z}^{\top})^{-1}\Big). |  |

Despite being inexpensive, this diagonal ΣW{\Sigma}\_{{W}} variant did not outperform Newton–Muon in our experiments and in some cases performed worse than standard Muon.

#### Conclusion.

Overall, if the update direction is systematically misaligned with W−W⋆{W}-W^{\star}, then the resulting estimate of ΣW{\Sigma}\_{{W}} inherits this bias, and since the next update direction depends on this estimate, the bias can reinforce itself, leading to slower training.
By using an isotropic proxy (i.e., ΣW∝Im{\Sigma}\_{{W}}\propto{I}\_{m}), we avoid imposing assumptions about the unknown distribution of displacements while still capturing the most robust geometric component of the update through the matrix sign. This helps explain both the computational simplicity and the stability of Newton–Muon.

[◄](/html/2604.01471)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2604.01472)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.01472)
[View original  
on arXiv](https://arxiv.org/abs/2604.01472)[►](/html/2604.01473)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed May 6 02:47:34 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

var canMathML = typeof(MathMLElement) == "function";
if (!canMathML) {
var body = document.querySelector("body");
body.firstElementChild.setAttribute('style', 'opacity: 0;');
var loading = document.createElement("div");
loading.setAttribute("id", "mathjax-loading-spinner");
var message = document.createElement("div");
message.setAttribute("id", "mathjax-loading-message");
message.innerText = "Typesetting Equations...";
body.prepend(loading);
body.prepend(message);
var el = document.createElement("script");
el.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
document.querySelector("head").appendChild(el);
window.MathJax = {
startup: {
pageReady: () => {
return MathJax.startup.defaultPageReady().then(() => {
body.removeChild(loading);
body.removeChild(message);
body.firstElementChild.removeAttribute('style');
}); } } };
}

// Auxiliary function, building the preview feature when
// an inline citation is clicked
function clicked\_cite(e) {
e.preventDefault();
let cite = this.closest('.ltx\_cite');
let next = cite.nextSibling;
if (next && next.nodeType == Node.ELEMENT\_NODE && next.getAttribute('class') == "ar5iv-bibitem-preview") {
next.remove();
return; }
// Before adding a preview modal,
// cleanup older previews, in case they're still open
document.querySelectorAll('span.ar5iv-bibitem-preview').forEach(function(node) {
node.remove();
})
// Create the preview
preview = document.createElement('span');
preview.setAttribute('class','ar5iv-bibitem-preview');
let target = document.getElementById(this.getAttribute('href').slice(1));
target.childNodes.forEach(function (child) {
preview.append(child.cloneNode(true));
});
let close\_x = document.createElement('button');
close\_x.setAttribute("aria-label","Close modal for bibliography item preview");
close\_x.textContent = "×";
close\_x.setAttribute('class', 'ar5iv-button-close-preview');
close\_x.setAttribute('onclick','this.parentNode.remove()');
preview.append(close\_x);
preview.querySelectorAll('.ltx\_tag\_bibitem').forEach(function(node) {
node.remove();
});
cite.parentNode.insertBefore(preview, cite.nextSibling);
return;
}
// Global Document initialization:
// - assign the preview feature to all inline citation links
document.querySelectorAll(".ltx\_cite .ltx\_ref").forEach(function (link) {
link.addEventListener("click", clicked\_cite);
});
