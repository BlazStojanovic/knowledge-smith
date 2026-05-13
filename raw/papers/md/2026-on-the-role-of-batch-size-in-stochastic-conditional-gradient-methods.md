---
arxiv: '2603.21191'
authors:
- Rustem Islamov
- Roman Machacek
- Aurelien Lucchi
- Antonio Silveti-Falls
- Eduard Gorbunov
- Volkan Cevher
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: On the Role of Batch Size in Stochastic Conditional Gradient Methods
url: https://arxiv.org/abs/2603.21191
year: 2026
---

[2603.21191] On the Role of Batch Size in Stochastic Conditional Gradient Methods














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



# On the Role of Batch Size in Stochastic Conditional Gradient Methods

Rustem Islamov1,4,†\dagger



Roman Machacek2



Aurelien Lucchi1



Antonio Silveti-Falls3
  
Eduard Gorbunov4,⋆\star



Volkan Cevher5,⋆\star

###### Abstract

We study the role of batch size in stochastic conditional gradient methods under a μ\mu-Kurdyka–Łojasiewicz (μ\mu-KL) condition. Focusing on momentum-based stochastic conditional gradient algorithms (e.g., Scion), we derive a new analysis that explicitly captures the interaction between stepsize, batch size, and stochastic noise. Our study reveals a regime-dependent behavior: increasing the batch size initially improves optimization accuracy but, beyond a critical threshold, the benefits saturate and can eventually degrade performance under a fixed token budget. Notably, the theory predicts the magnitude of the optimal stepsize and aligns well with empirical practices observed in large-scale training. Leveraging these insights, we derive principled guidelines for selecting the batch size and stepsize, and propose an adaptive strategy that increases batch size and sequence length during training while preserving convergence guarantees. Experiments on NanoGPT are consistent with the theoretical predictions and illustrate the emergence of the predicted scaling regimes. Overall, our results provide a theoretical framework for understanding batch size scaling in stochastic conditional gradient methods and offer guidance for designing efficient training schedules in large-scale optimization.

$\dagger$$\dagger$footnotetext: Most of this work was done when Rustem Islamov was a visiting student in the group of Prof. Eduard Gorbunov at MBZUAI, UAE.$\star$$\star$footnotetext: The last two authors share senior authorship.

## 1 Introduction

Large-scale language model training is constrained by a token budget TT rather than by a fixed number of optimization steps.
In this regime, we face a familiar batch size tradeoff: increasing the batch size BB improves hardware utilization, yet beyond a certain scale it can degrade optimization efficiency and hurt generalization (Goyal et al., [2017](#bib.bib97 "Accurate, large minibatch sgd: training imagenet in 1 hour"); Keskar et al., [2017](#bib.bib100 "On large-batch training for deep learning: generalization gap and sharp minima"); Smith et al., [2018](#bib.bib102 "Don’t decay the learning rate, increase the batch size"); Shallue et al., [2019](#bib.bib99 "Measuring the effects of data parallelism on neural network training")).

A token budget-aware viewpoint makes this tradeoff explicit.
With batch size BB and sequence length SS, the number of parameter updates is
K≔TB​S,K\coloneqq\frac{T}{BS},
and hence (B,S)(B,S) and the stepsize jointly determine how effectively the token budget is converted into optimization progress.
This coupling raises a central question in model training:
*how should (B,S)(B,S) and the stepsize be chosen, and adapted, to optimize performance under a fixed token budget TT?*

Recent empirical studies have further refined this picture. In particular, critical batch sizes – the point at which scaling BB stops being beneficial – appear to scale primarily with the effective data size and only weakly with model size under a fixed token budget (Zhang et al., [2025](#bib.bib103 "How does critical batch size scale in pre-training?"); Bergsma et al., [2025](#bib.bib104 "Power lines: scaling laws for weight decay and batch size in LLM pre-training")). Additionally, the critical batch threshold is often stage-dependent, motivating warmup and stage-wise training schedules (Merrill et al., [2025](#bib.bib105 "Critical batch size revisited: a simple empirical approach to large-batch language model training")). Taken together, these findings suggest that the batch size should be treated as a dynamic optimization variable rather than a fixed hyperparameter. However, these insights remain largely empirical: they do not provide explicit optimization error laws as functions of (B,S,T)(B,S,T), nor do they characterize when increasing batch size becomes provably detrimental under a fixed token budget.

In parallel, hyperparameter transfer frameworks such as μ\muP have shown that, with appropriate parameterization and initialization, gradient magnitudes can be kept Θ​(1)\Theta(1) across model scales, enabling stable training without retuning learning rates (Yang and Hu, [2020](#bib.bib63 "Feature learning in infinite-width neural networks"); Yang et al., [2021](#bib.bib93 "Tuning large neural networks via zero-shot hyperparameter transfer"), [2022](#bib.bib57 "Tensor programs v: tuning large neural networks via zero-shot hyperparameter transfer")).
However, these results are inherently local: they ensure that individual updates neither explode nor vanish, but do not address how batch size, sequence length, and stepsize should scale *globally* with the token budget.

Our work bridges this gap by showing that hyperparameters that are locally optimal for a given (B,S,T)(B,S,T) can become provably suboptimal as the token budget increases, even under μ\muP-style initialization. To obtain such global scaling laws, we derive an analysis for stochastic conditional gradient (SCG) methods (Pethick et al., [2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")), a projection-free framework that underlies several modern norm-constrained training algorithms. This class of algorithms is closely aligned with modern optimizers such as Muon (Jordan et al., [2024b](#bib.bib95 "Muon: an optimizer for hidden layers in neural networks")).

Our analysis is carried out for stochastic optimization ([1](#S3.E1 "Equation 1 ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) under smoothness ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) in a general norm, norm equivalence ([A2](#S3.Ex2 "Equation A2 ‣ Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), and a μ\mu-Kurdyka–Łojasiewicz (μ\mu-KL) error bound ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) (Karimi et al., [2016](#bib.bib114 "Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition"); Bolte et al., [2007](#bib.bib113 "The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems")).
The μ\mu-KL condition is particularly well matched to SCG geometry, as it relates first-order stationarity to suboptimality measured in the dual norm induced by the linear minimization oracle (LMO).

Specializing our convergence bounds to the fixed-token setting T=K​B​ST=KBS yields an explicit, non-monotone dependence of the achievable optimization error on the effective batch–sequence scale B​SBS.
Three regimes emerge:
*(i)* a noise-dominated regime where increasing B​SBS improves performance,
*(ii)* an intermediate regime where the best achievable error is essentially independent of B​SBS, and
*(iii)* a large-batch regime where performance deteriorates as B​SBS grows under a fixed token budget.

Balancing the dominant terms yields a *critical* effective batch–sequence–token (BST) scale rule
B​S≍T2/3BS\asymp T^{2/3}
up to problem-dependent factors that we derive in this work,
revealing how curvature, noise, geometry, and error-bound strength shift the optimal operating point.
Importantly, our analysis shows that large batch sizes do not inherently degrade performance:
when batch size, sequence length, and learning rate are chosen according to our BST scaling rule, large-batch training remains effective and token-efficient. In contrast to μ\muP, our perspective disentangles local stability, as controlled by parameterization and initialization, from global efficiency, as governed by token-budget–aware optimization.

Our contributions are as follows:

* •

  Convergence guarantees for momentum SCG under μ\mu-KL.
  We establish convergence guarantees for [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") under the μ\mu-KL condition ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) in a general normed geometry, explicitly tracking the effects of momentum, smoothness, and stochastic gradient noise.
  Our bounds hold *in expectation* under bounded-variance and LL-smoothness assumptions.
* •

  A token-budget view of batch, sequence length, and stepsize scaling.
  By translating iteration complexity into token complexity via T=K​B​ST=KBS, we obtain explicit (B,S,T)(B,S,T)-dependent error laws and identify the *critical* effective batch size B​SBS that separates beneficial from harmful scaling.
* •

  Actionable adaptive scheduling rules.
  We turn the theory into concrete recipes for choosing and *updating* (β,B,S)(\beta,B,S) during training under a fixed token budget, yielding the scaling relations ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))–([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and a two-stage (and more generally multi-stage) protocol validated empirically on NanoGPT (cf., [Figure˜2](#S4.F2 "In 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

Our results complement classical large-batch heuristics such as linear learning rate scaling with warmup (Goyal et al., [2017](#bib.bib97 "Accurate, large minibatch sgd: training imagenet in 1 hour")) and adaptive batch size schedules (Smith et al., [2018](#bib.bib102 "Don’t decay the learning rate, increase the batch size")), while offering a *projection-free* viewpoint rooted in conditional gradient geometry.
They are also consistent with empirical observations that there exists a largest useful batch size depending on training stage and problem statistics (McCandlish et al., [2018](#bib.bib101 "An empirical model of large-batch training"); Shallue et al., [2019](#bib.bib99 "Measuring the effects of data parallelism on neural network training")), and provide an explicit optimization-side mechanism for the “too-large batch hurts” regime under a fixed token budget.

## 2 Related Works

##### Assumptions in SCG methods: smoothness.

Convergence analyses for stochastic conditional gradient (SCG) (aka Frank–Wolfe) methods and, more broadly, *LMO-based* methods, have been conducted under various assumptions. Most analyses, including our analysis, assume standard LL-smoothness. However, recent works consider relaxed notions, such as (L0,L1)(L\_{0},L\_{1})-smoothness (Zhang et al., [2019](#bib.bib36 "Why gradient clipping accelerates training: a theoretical justification for adaptivity")) and other extensions beyond global smoothness (Pethick et al., [2025b](#bib.bib2 "Generalized gradient norm clipping & non-euclidean (L0,L1)-smoothness"); Riabinin et al., [2025](#bib.bib3 "Gluon: making Muon & Scion great again! (Bridging theory and practice of LMO-based optimizers for LLMs)")).
Extending our analysis to these generalized smoothness settings is an interesting direction for future work, but it lies beyond the scope of the present paper.

##### Assumptions in SCG methods: structured nonconvexity.

Most prior work considers either general nonconvex or (strongly) convex objectives, failing to capture practical learning rate and batch size scaling effects observed in large-scale training. This limitation motivates our study under structured nonconvexity.

Several recent works study structured nonconvexity for LMO-based or related methods. Yang et al. ([2024](#bib.bib1 "Adaptive gradient normalization and independent sampling for (stochastic) generalized-smooth optimization")) derives an analysis under a generalized Polyak–Łojasiewicz condition, which recovers our [˜3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") as a special case. Their method, however, does not use momentum and assumes almost surely affine bounded noise, in contrast to the bounded variance setting considered here.

Kovalev ([2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization")) studies stochastic conditional gradient methods under star-convexity, a condition closely related to the μ\mu-KL condition. However, our work empirically validates the μ\mu-KL condition in large-scale language models training and uses it to derive a principled BST scaling rule under a fixed token budget. Finally, Riabinin et al. ([2025](#bib.bib3 "Gluon: making Muon & Scion great again! (Bridging theory and practice of LMO-based optimizers for LLMs)")) study an LMO-based method with adaptive layer-wise learning rates under the classical Polyak-Łojasiewicz (PL) condition (Polyak, [1963](#bib.bib28 "Gradient methods for the minimisation of functionals"); Łojasiewicz, [1963](#bib.bib27 "A topological property of real analytic subsets")), restricted to the deterministic setting without momentum, limiting its applicability to the large-scale stochastic settings.

##### Works on Hyperparameter Transfer.

Transferring hyperparameters (HPs) tuned on small proxy models to large-scale training has become increasingly important as model sizes grow.
This line of work was initiated by the μ\muP framework (Yang and Hu, [2020](#bib.bib63 "Feature learning in infinite-width neural networks"); Yang et al., [2021](#bib.bib93 "Tuning large neural networks via zero-shot hyperparameter transfer"), [2022](#bib.bib57 "Tensor programs v: tuning large neural networks via zero-shot hyperparameter transfer")), which enables zero-shot transfer of learning rates across model *width*, and was later extended to other aspects of the model architecture, such as depth (Yang et al., [2023](#bib.bib120 "Feature learning in infinite-depth neural networks"); Dey et al., [2025](#bib.bib121 "Don’t be lazy: completep enables compute-efficient deep transformers")).

Technically, μ\muP-style analyses focus on parameterizations that ensure gradient magnitudes and parameter updates remain Θ​(1)\Theta(1) around initialization. These analyses assume a fixed number of tokens processed per step and do not characterize optimization behavior when the number of optimization steps is significantly larger than the model width.

To reason about the latter regime, we analyze SCG methods under a μ\mu-KL condition and derive convergence guarantees that explicitly depend on the batch size BB, sequence length SS, and total token budget TT. This trajectory-level analysis allows us to characterize how optimization error accumulates as a function of (B,S,T)(B,S,T) and to derive principled scaling rules for jointly adapting batch size, sequence length, and stepsize, in contrast to prior hyperparameter transfer works that focus on local, per-step stability governing early training behavior.

##### Batch Size Scheduling.

Adapting the batch size during training is a well-established and practical strategy, motivated by both computational efficiency and optimization dynamics. Increasing the batch size can serve as an alternative to learning rate decay, reduce the number of parameter updates, and improve parallel utilization (Smith et al., [2018](#bib.bib102 "Don’t decay the learning rate, increase the batch size")). However, compared to small-batch training, large batches often lead to worse generalization performance and tend to converge to sharper minima (Keskar et al., [2017](#bib.bib100 "On large-batch training for deep learning: generalization gap and sharp minima")).

A complementary empirical view suggests a *critical batch size* (CBS), beyond which increasing BB yields diminishing token efficiency; McCandlish et al. ([2018](#bib.bib101 "An empirical model of large-batch training")) relate CBS to the gradient noise scale and argue that it evolves during training. In the LLM setting, scaling-law work (Kaplan et al., [2020](#bib.bib32 "Scaling laws for neural language models"); Hoffmann et al., [2022](#bib.bib29 "Training compute-optimal large language models (2022)")) primarily addresses how to allocate a fixed compute budget across model size and training tokens, rather than prescribing within-run batch size schedules. More recently, Bi et al. ([2024](#bib.bib30 "Deepseek llm: scaling open-source language models with longtermism")) report empirical power-law relations between compute budget, batch size, and learning rate that perform well at scale.

Taken together, these works reinforce a central practical message: the best batch size is typically not a fixed constant, but depends on the training stage, optimization hyperparameters, and budget. Motivated by this, we seek *principled, token-budget–aware* rules that characterize how the optimal effective batch–sequence scale and stepsize should co-vary with TT, and how (B,S,β)(B,S,\beta) should be adapted.

Algorithm 1  Stochastic Conditional Gradient (SCG)

Input: x0,m0∈𝒳x\_{0},m\_{0}\in\mathcal{X}, parameters α,β∈(0,1),η>0\alpha,\beta\in(0,1),\eta>0

for k=0,…,K−1k=0,\ldots,K-1 do

sample ξk∼𝒟\xi\_{k}\sim\mathcal{D}

compute mk+1=(1−α)​mk+α​g​(xk;ξk)m\_{k+1}=(1-\alpha)m\_{k}+\alpha g(x\_{k};\xi\_{k})

compute dk+1=arg​mind∈𝒳⁡⟨mk+1,d⟩d\_{k+1}={\rm arg}\min\_{d\in\mathcal{X}}\langle m\_{k+1},d\rangle s.t. ‖d‖≤1\|d\|\leq 1

compute xk+1=(1−β)​xk+β​η​dk+1x\_{k+1}=(1-\beta)x\_{k}+\beta\eta d\_{k+1}

end for

## 3 Problem Formulation and Assumptions

We consider the following problem template:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minx∈𝒳⁡f​(x),\min\_{x\in\mathcal{X}}f(x), |  | (1) |

where the space 𝒳\mathcal{X} is equipped with a standard Euclidean norm ∥⋅∥2\|\cdot\|\_{2} induced by the inner product ⟨⋅,⋅⟩\langle\cdot,\cdot\rangle, i.e., ‖x‖2=⟨x,x⟩\|x\|\_{2}=\sqrt{\langle x,x\rangle}, and another norm ∥⋅∥\|\cdot\|, which possibly does not coincide with the Euclidean one. For the norm ∥⋅∥\|\cdot\|, we define the associated dual norm ‖x‖∗≔sup‖x′‖≤1⟨x,x′⟩\|x\|\_{\*}\coloneqq\sup\_{\|x^{\prime}\|\leq 1}\langle x,x^{\prime}\rangle for all x∈𝒳x\in\mathcal{X}. We seek to solve ([1](#S3.E1 "Equation 1 ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) using [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

###### Assumption 3.1.

Let the gradient ∇f​(⋅)\nabla f(\cdot) be Lipschitz continuous with respect to the norm ∥⋅∥\|\cdot\|:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∇f​(x)−∇f​(x′)‖∗≤L​‖x−x′‖for all ​x,x′∈𝒳,\|\nabla f(x)-\nabla f(x^{\prime})\|\_{\*}\leq L\|x-x^{\prime}\|\quad\text{for all }x,x^{\prime}\in\mathcal{X}, |  | (A1) |

where L>0L>0 is the gradient Lipschitz constant.

###### Assumption 3.2.

There exist a constant ρ>0\rho>0 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖x‖∗≤ρ​‖x‖2for all ​x∈𝒳.\|x\|\_{\*}\leq\rho\|x\|\_{2}\quad\text{for all }x\in\mathcal{X}. |  | (A2) |

Note that such a constant always exists by norm equivalence, which always holds in finite-dimensional spaces 𝒳\mathcal{X}.

###### Assumption 3.3.

The objective function f​(x)f(x) is μ\mu-KL for some μ>0\mu>0:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖∇f​(x)‖∗≥μ​(f​(x)−f⋆)for all ​x∈𝒳\displaystyle\|\nabla f(x)\|\_{\*}\geq\mu(f(x)-f^{\star})\quad\text{for all }x\in\mathcal{X} |  | (A3) |

where f⋆=minx∈𝒳⁡f​(x)f^{\star}=\min\_{x\in\mathcal{X}}f(x).

Note that condition ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) is closely related to the Polyak-Łojasiewicz (PL) condition ‖∇f​(x)‖22≥μ​(f​(x)−f⋆)\|\nabla f(x)\|\_{2}^{2}\geq\mu(f(x)-f^{\star}) originally studied in Polyak ([1963](#bib.bib28 "Gradient methods for the minimisation of functionals")); Łojasiewicz ([1963](#bib.bib27 "A topological property of real analytic subsets")). Variants of the PL condition have been investigated for over-parameterized models (Liu et al., [2022](#bib.bib33 "Loss landscapes and optimization in over-parameterized non-linear systems and neural networks")). A key distinction between the μ\mu-PL and μ\mu-KL conditions lies in the exponent of the gradient norm, making the difference between them significant when the norm is small.

|  |
| --- |
| Refer to caption |

Figure 1: Empirical verification of the validity of [˜3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") during the training of a 124M NanoGPT model. The points with a loss below 5 fit a linear function well, with a slope equal to μ\mu.

Nevertheless, condition ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) has been extensively used in the optimization literature to analyze gradient descent under the Euclidean norm (Bolte et al., [2014](#bib.bib79 "Proximal alternating linearized minimization for nonconvex and nonsmooth problems"); Fatkhullin et al., [2022](#bib.bib76 "Sharp analysis of stochastic optimization under global kurdyka-łojasiewicz inequality")). For problems with a bounded domain,the μ\mu-KL condition is closely related to ζ\zeta-quasar convexity (ζ\zeta-QC) (Hardt et al., [2018](#bib.bib21 "Gradient descent learns linear dynamical systems"); Guminov et al., [2017](#bib.bib19 "Accelerated methods for α-weakly-quasi-convex problems")), which requires ⟨∇f​(x),x−x⋆⟩≥ζ​(f​(x)−f⋆)\langle\nabla f(x),x-x^{\star}\rangle\geq\zeta(f(x)-f^{\star}) for some x⋆∈𝒳x^{\star}\in\mathcal{X} and all x∈𝒳x\in\mathcal{X}. ζ\zeta-QC naturally arises in the training of neural networks (Zhou et al., [2019](#bib.bib24 "Sgd converges to global minimum in deep learning via star-convex path"); Kleinberg et al., [2018](#bib.bib22 "An alternative view: when does sgd escape local minima?")). When 𝒳\mathcal{X} is bounded with diameter RR with respect to the norm ∥⋅∥\|\cdot\|, ζ\zeta-QC
implies the μ\mu-KL condition with μ=ζ/R\mu=\zeta/R.

In this work, we extend the applicability of the standard μ\mu-KL assumption beyond the Euclidean norm. To demonstrate its validity in practice, we track the train loss and dual gradient norm during the training of a 124M NanoGPT model. In [Figure˜1](#S3.F1 "In 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we observe that the measurements fit a linear function well, especially when the loss is below 5
(cf., the description of the full setting in [Section˜6.2](#S6.SS2 "6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))

We make the assumption below for the gradient noise.

###### Assumption 3.4.

We have access to the unbiased estimator g​(⋅;ξ):𝒳→𝒳g(\cdot;\xi)\colon\mathcal{X}\to\mathcal{X} of the gradient ∇f​(⋅)\nabla f(\cdot), where ξ∼𝒟\xi\sim\mathcal{D} is a random variable sampled from a probability distribution 𝒟\mathcal{D}. We assume that the stochastic gradient estimator g​(⋅;ξ)g(\cdot;\xi) is unbiased and has σ\sigma-bounded variance for some σ≥0\sigma\geq 0:

|  |  |  |
| --- | --- | --- |
|  | 𝔼ξ∼𝒟​[g​(x;ξ)]=∇f​(x)and𝔼ξ∼𝒟​[‖g​(x;ξ)−∇f​(x)‖22]≤σ2for all ​x∈𝒳.\displaystyle\mathbb{E}\_{\xi\sim\mathcal{D}}[g(x;\xi)]=\nabla f(x)\quad\text{and}\quad\mathbb{E}\_{\xi\sim\mathcal{D}}[\|g(x;\xi)-\nabla f(x)\|\_{2}^{2}]\leq\sigma^{2}\quad\text{for all }x\in\mathcal{X}. |  |

Additionally, let σ2=σ⋆2B​S\sigma^{2}=\frac{\sigma\_{\star}^{2}}{BS}, where BB and SS are batch size and sequence length respectively.

[˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") is a classical assumption for the in-expectation convergence analysis of stochastic methods (Ghadimi and Lan, [2012](#bib.bib84 "Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization i: a generic algorithmic framework"), [2013](#bib.bib85 "Stochastic first-and zeroth-order methods for nonconvex stochastic programming")).
We verify the validity of [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") during the training in [Figure˜2](#S4.F2 "In 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") (cf., the description of the full setting in [Section˜6.1](#S6.SS1 "6.1 Verification of Assumption 3.4 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

## 4 Theoretical Analysis

This section establishes convergence guarantees for [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), guiding how to choose the batch size BB, sequence length SS, and stepsize β\beta under a fixed token budget TT. The proof and the full statement of the following theorem are deferred to [Appendix˜D](#A4 "Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

###### Theorem 4.1.

Let Assumptions ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([A2](#S3.Ex2 "Equation A2 ‣ Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), and ([3.4](#S3.Ex5 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) hold. Let m0=g​(x0;ξ0)m\_{0}=g(x\_{0};\xi\_{0}). Let the parameters of
[Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") and initialization x0x\_{0} be chosen as follows

|  |  |  |
| --- | --- | --- |
|  | β=𝒪​(1K),η=𝒪~​(1μ),α=min⁡{1,𝒪​((ε​μ)2(ρ​σ)2)},2​‖x0‖≤η,and\displaystyle\beta=\mathcal{O}\left(\frac{1}{K}\right),\quad\eta=\widetilde{\mathcal{O}}\left(\frac{1}{\mu}\right),\quad\alpha=\min\left\{1,{\mathcal{O}}\left(\frac{(\varepsilon\mu)^{2}}{(\rho\sigma)^{2}}\right)\right\},\quad 2\|x\_{0}\|\leq\eta,\quad\text{and} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | K=max⁡[𝒪~​(1),𝒪~​(max⁡{Lε​μ2,ρ​σε​μ,L​(ρ​σ)2μ​(ε​μ)3,(ρ​σ)3(ε​μ)3})],\displaystyle K=\max\left[\widetilde{\mathcal{O}}(1),\widetilde{\mathcal{O}}\left(\max\left\{\frac{L}{\varepsilon\mu^{2}},\frac{\rho\sigma}{\varepsilon\mu},\frac{L(\rho\sigma)^{2}}{\mu(\varepsilon\mu)^{3}},\frac{(\rho\sigma)^{3}}{(\varepsilon\mu)^{3}}\right\}\right)\right], |  | (2) |

where 𝒪\mathcal{O} hides all numerical constants and 𝒪~\tilde{\mathcal{O}} hides all numerical and logarithmic factors. Then, the output of [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") after KK iterations satisfies 𝔼​[f​(xK)−f⋆]≤ε\mathbb{E}[f(x\_{K})-f^{\star}]\leq\varepsilon.

###### Remark 4.1.

Convergence bounds for SCG were derived in Pethick et al. ([2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")) for the Frank-Wolfe gap, then similar results to [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") were given by Kovalev ([2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization"))111Kovalev ([2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization")) studies a stochastic first-order non-Euclidean trust-region method with momentum and weight decay, which is equivalent to [Algorithm 1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). under star-convexity, a special case of ζ\zeta-quasar convexity with ζ=1\zeta=1. In light of the relationship between the μ\mu-KL condition and ζ\zeta-QC in Section [3](#S3 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), this similarity is expected.

Our work goes beyond this connection in two important ways. First, we provide empirical justification for the use of the μ\mu-KL condition in the analysis. Second, building on this framework, we derive new theory-guided scaling rules for both the learning rate and the batch size.

In practice, the number of iterations KK cannot be arbitrarily large. In fact, KK is trivially constrained by the available token budget TT, the two being related by the simple identity T=K⋅B⋅ST=K\cdot B\cdot S. Consequently, the requirement on KK in [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")222We ignore the requirement K=𝒪~​(1)K=\widetilde{\mathcal{O}}(1), as it is always satisfied in practice; see also [Corollary D.1](#A4.Thmcorollary1 "Corollary D.1 (Full statement of Corollary˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") for the details. In the sequel, we omit numerical constants for clarity. can be equivalently expressed as a condition on TT by multiplying both sides by B​SBS:

|  |  |  |
| --- | --- | --- |
|  | T=𝒪~​(max​{L​B​Sε​μ2,ρ​σ​B​Sε​μ,L​(ρ​σ)2​B​Sμ​(ε​μ)3,(ρ​σ)3​B​S(ε​μ)3))T=\tilde{\mathcal{O}}\left(\max\left\{\frac{LBS}{\varepsilon\mu^{2}},\frac{\rho\sigma BS}{\varepsilon\mu},\frac{L(\rho\sigma)^{2}BS}{\mu(\varepsilon\mu)^{3}},\frac{(\rho\sigma)^{3}BS}{(\varepsilon\mu)^{3}}\right)\right) |  |

Under a fixed token budget, the expression above indicates that we cannot achieve an arbitrary optimization error ε\varepsilon. Instead, [Corollary˜4.1](#S4.Thmcorollary1 "Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") lower bounds the achievable error.

###### Corollary 4.1 (BST Scaling Rule).

Under the setup of [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), running the algorithm with parameters from [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") for TB​S\frac{T}{BS} iterations, we achieve the optimization error

|  |  |  |  |
| --- | --- | --- | --- |
|  | ε=𝒪~​(max⁡{L​B​Sμ2​T,(L​ρ2​σ⋆2μ4​T)1/3,ρ​σ⋆μ​(T2​B​S)1/6}),\varepsilon=\tilde{\mathcal{O}}\left(\max\left\{\frac{LBS}{\mu^{2}T},\left(\frac{L\rho^{2}\sigma\_{\star}^{2}}{\mu^{4}T}\right)^{1/3},\frac{\rho\sigma\_{\star}}{\mu(T^{2}BS)^{1/6}}\right\}\right), |  | (3) |

where 𝒪~\tilde{\mathcal{O}} hides numerical and logarithmic factors.

|  |  |
| --- | --- |
| Refer to caption | Refer to caption |

Figure 2: Empirical gradient variance and fitted power-law models as functions of batch size BB with fixed sequence length S=1024S=1024 (left) and sequence length SS with fixed batch size B=512B=512 (right) when training a 124M NanoGPT model on the FineWeb dataset under a fixed token budget T=2.7T=2.7B. For the left plot, the estimated scaling exponent is λ≈0.9\lambda\approx 0.9 and Bshift≈90B\_{\rm shift}\approx 90, while for the right plot they are λ≈1.1\lambda\approx 1.1 and Sshift≈35S\_{\rm shift}\approx 35. The fitted models support the validity of [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

[Corollary˜4.1](#S4.Thmcorollary1 "Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") provides key insights into how the error ε\varepsilon typically varies as the product B​SBS changes:

1. 1.

   For small batch sizes, the third term in ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) dominates, and ε\varepsilon improves as B​SBS increases.
2. 2.

   When the batch size exceeds (μ​ρ​σ⋆L)2\left(\frac{\mu\rho\sigma\_{\star}}{L}\right)^{2}, the second term in ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) becomes dominant. In this regime, the error is independent of the batch size and sequence length, and the error instead scales as ∼T−1/3\sim T^{-1/3}.
3. 3.

   Further increasing the batch size moves the system into an iteration-starved regime where the first term dominates, causing the error to deteriorate linearly.

[Corollary˜4.1](#S4.Thmcorollary1 "Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") indicates that the optimal achievable performance lies in the second regime, where the optimization error ε\varepsilon is independent of both the batch size and the sequence length. From a practical perspective, however, larger batch sizes are often preferred to improve GPU utilization (Narayanan et al., [2021](#bib.bib87 "Efficient large-scale language model training on gpu clusters using megatron-lm")).

This motivates us to select the batch size and sequence length at the crossover between the second and third regimes. Following this intuition, we choose BB and SS as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lμ2​B​ST=(L​ρ2​σ⋆2μ4​T)1/3⇔B​S=(T​μ​ρ​σ⋆L)2/3,\hskip-8.53581pt\frac{L}{\mu^{2}}\frac{BS}{T}=\left(\frac{L\rho^{2}\sigma\_{\star}^{2}}{\mu^{4}T}\right)^{1/3}\Leftrightarrow BS=\left(\frac{T\mu\rho\sigma\_{\star}}{L}\right)^{2/3}, |  | (4) |

balancing final performance and hardware efficiency. Next, the BST rule results in the Frank–Wolfe stepsize

|  |  |  |  |
| --- | --- | --- | --- |
|  | β⋆∼1K.\beta\_{\star}\sim\frac{1}{K}. |  | (5) |

Notably, a Frank–Wolfe stepsize of this form is used in practice when employing decoupled weight decay (Loshchilov and Hutter, [2019](#bib.bib90 "Decoupled weight decay regularization")) to train LLMs near the Chinchilla-optimal token-per-parameter (TPP) regime (Xiao, [2024](#bib.bib89 "Rethinking conventional wisdom in machine learning: from generalization to scaling"); Qiu et al., [2025](#bib.bib88 "Hyperparameter transfer enables consistent gains of matrix-preconditioned optimizers across scales")), where the model depth scales proportionally with the token budget.

Using ε=(L​ρ2​σ⋆2μ4​T)1/3\varepsilon=\left(\frac{L\rho^{2}\sigma\_{\star}^{2}}{\mu^{4}T}\right)^{1/3}, B​S=(T​μ​ρ​σ⋆L)2/3BS=\left(\frac{T\mu\rho\sigma\_{\star}}{L}\right)^{2/3}, and [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") in [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we obtain that the momentum parameter α\alpha

|  |  |  |
| --- | --- | --- |
|  | α∼μ2​B​Sρ2​σ⋆2⋅(L​ρ2​σ⋆2μ4​T)2/3=(Lμ​ρ​σ⋆​T)2/3​(T​μ​ρ​σ⋆L)2/3=Const.\alpha\sim\frac{\mu^{2}BS}{\rho^{2}\sigma\_{\star}^{2}}\cdot\left(\frac{L\rho^{2}\sigma\_{\star}^{2}}{\mu^{4}T}\right)^{2/3}=\left(\frac{L}{\mu\rho\sigma\_{\star}T}\right)^{2/3}\left(\frac{T\mu\rho\sigma\_{\star}}{L}\right)^{2/3}=\text{Const}. |  |

This suggests that if we find an optimal momentum parameter α\alpha for a small model, under the BST scaling rule, it transfers to the larger setting.

To summarize, the BST scaling rule suggests the following choice of parameters in [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"):

B​S∼T2/3,β∼1K,α=Const.BS\sim T^{2/3},\quad\beta\sim\frac{1}{K},\quad\alpha=\text{Const}.

(6)

In [Section˜5](#S5 "5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we provide a more detailed explanation of the BST scaling rule for the parameter choice when training under a fixed TPP or increasing the token budget for the same model.

## 5 Strategies for Hyperparameter Choice

##### Training Setup.

Training a model such that ([4](#S4.E4 "Equation 4 ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) holds establishes working strategies on how to train a larger model of size D1D\_{\rm 1} efficiently, given that we have a tuned configuration (i.e., the tuned values of Frank–Wolfe stepsize β0\beta\_{\rm 0}, momentum parameter α\alpha, batch size B0,B\_{\rm 0}, and sequence length S0S\_{\rm 0}) for a smaller model of size D0D\_{\rm 0}. We consider the training under a fixed TPP, which implies that the available token budget increases proportionally to the model size, i.e., T1/T0=D1/D0.\nicefrac{{T\_{\rm 1}}}{{T\_{\rm 0}}}=\nicefrac{{D\_{\rm 1}}}{{D\_{\rm 0}}}. Moreover, we assume that the problem constants L=L​(D),μ=μ​(D),L=L(D),\mu=\mu(D), and ρ=ρ​(D)\rho=\rho(D) change with model size. We denote the constants with subscripts 11 and 0 for models of size D1D\_{\rm 1} and D0D\_{\rm 0}, respectively.

###### Remark 5.1.

In this work, we assume that the variance constant σ⋆2\sigma\_{\star}^{2} in [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") does not depend on the model size, as estimating its scaling with model size is computationally infeasible. We acknowledge, however, that in practice σ⋆2\sigma\_{\star}^{2} may change as the model size grows.

### 5.1 Increasing Batch Size

We assume that the optimal batch size B0⋆B\_{0}^{\star}, sequence length S0⋆S\_{0}^{\star}, and β0⋆\beta\_{0}^{\star} are tuned for a small model333Ideally, we want all hyperparameters of the optimizer and model to be tuned for a small model, including radii η\eta or the initialization. However, such a task is infeasible even for a small model. Therefore, we focus on the main hyperparameters that affect the final performance the most: batch size, sequence length, and Frank–Wolfe stepsize, while we set the rest according to default values obtained from prior work. of size D0D\_{\rm 0} and satisfy ([4](#S4.E4 "Equation 4 ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), namely

|  |  |  |
| --- | --- | --- |
|  | B0⋆​S0⋆∼(T0​μ0​ρ0​σ⋆L0)2/3.B\_{\rm 0}^{\star}S\_{\rm 0}^{\star}\sim\left(\frac{T\_{\rm 0}\mu\_{\rm 0}\rho\_{\rm 0}\sigma\_{\star}}{L\_{\rm 0}}\right)^{2/3}. |  |

We now determine B1B\_{1} and S1S\_{1} such that ([4](#S4.E4 "Equation 4 ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) remains satisfied for a larger model. By simple manipulation, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | B1​S1=B0⋆​S0⋆​(T1T0​μ1μ0​ρ1ρ0L1L0)2/3.\displaystyle B\_{\rm 1}S\_{\rm 1}=B\_{\rm 0}^{\star}S\_{\rm 0}^{\star}\left(\frac{\frac{T\_{\rm 1}}{T\_{\rm 0}}\frac{\mu\_{\rm 1}}{\mu\_{\rm 0}}\frac{\rho\_{\rm 1}}{\rho\_{\rm 0}}}{\frac{L\_{\rm 1}}{L\_{\rm 0}}}\right)^{2/3}. |  | (7) |

Note that the ratio T1/T0\nicefrac{{T\_{1}}}{{T\_{0}}} can be replaced by D1/D0\nicefrac{{D\_{1}}}{{D\_{0}}} under fixed TPP. Knowing how L,μ,ρL,\mu,\rho change with model size and batch size,444In real-world applications, the change of constants with a model size might be ignored for simplicity, but later we provide estimates for them that we use in [Section 6](#S6 "6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). we can adjust the batch size and sequence length for a larger model.

### 5.2 Tuning the Frank–Wolfe Stepsize

From ([5](#S4.E5 "Equation 5 ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) we know that the optimal Frank–Wolfe stepsize β\beta should scale as 1K\frac{1}{K}; therefore, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | β0⋆β1=B0⋆​S0⋆/T0B1​S1/T1⇒β1=β0⋆​B1​S1B0⋆​S0⋆​T0T1.\hskip-8.53581pt\frac{\beta\_{\rm 0}^{\star}}{\beta\_{\rm 1}}=\frac{\nicefrac{{B\_{\rm 0}^{\star}S\_{\rm 0}^{\star}}}{{T\_{0}}}}{\nicefrac{{B\_{\rm 1}S\_{\rm 1}}}{{T\_{1}}}}\Rightarrow\beta\_{\rm 1}=\beta\_{\rm 0}^{\star}\frac{B\_{\rm 1}S\_{\rm 1}}{B\_{\rm 0}^{\star}S\_{\rm 0}^{\star}}\frac{T\_{0}}{T\_{1}}. |  | (8) |

Since we increase batch size and sequence length according to ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), then the optimal Frank–Wolfe stepsize for a larger model is expected to be around

|  |  |  |  |
| --- | --- | --- | --- |
|  | β1=β0⋆​(T0T1​μ1μ0​ρ1ρ0L1L0)2/3.\beta\_{\rm 1}=\beta\_{\rm 0}^{\star}\left(\frac{\frac{\sqrt{T\_{\rm 0}}}{\sqrt{T\_{\rm 1}}}\frac{\mu\_{\rm 1}}{\mu\_{\rm 0}}\frac{\rho\_{\rm 1}}{\rho\_{\rm 0}}}{\frac{L\_{\rm 1}}{L\_{\rm 0}}}\right)^{2/3}. |  | (9) |

### 5.3 Batch Size Scheduling

We now consider a training setting in which data arrives sequentially rather than being fully available upfront.
In this setting, a model is first trained on an initial corpus and subsequently updated as additional data becomes available, causing the effective token budget to grow over time.

This departs from standard pretraining assumptions and raises a practical question: how should hyperparameters such as batch size and sequence length be adapted as the available token budget increases? Naïvely reusing batch and sequence settings tuned for early stages can lead to suboptimal token efficiency and slower convergence.

In the following, we propose a principled and practically implementable pipeline for selecting and adapting batch size and sequence length in the delayed-data regime.

##### First stage (training with T(1)=T0T\_{(1)}=T\_{0} tokens).

Assume that in the beginning, we only have a smaller token budget T(1)=T0T\_{(1)}=T\_{0}, which is sufficient to train a smaller model efficiently, but insufficient to do so for a larger model.

The remaining tokens T(2)=T1−T0T\_{(2)}=T\_{1}-T\_{0} arrive at a later time. Based on ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), when training the large model using T(1)T\_{(1)} tokens,555We should use T(1)T\_{(1)} instead of T1T\_{1} in ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) when TPP is not fixed. our theory suggests choosing the batch size B1B\_{1} and sequence length S1S\_{1} such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | B1​S1=B0⋆​S0⋆​(μ1/μ0⋅ρ1/ρ0L1/L0)2/3​≈\Hy@raisedlink([a](#desca0 "\Hy@raisedlink ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​B0⋆​S0⋆,B\_{1}S\_{1}=B\_{0}^{\star}S\_{0}^{\star}\left(\frac{\nicefrac{{\mu\_{1}}}{{\mu\_{0}}}\cdot\nicefrac{{\rho\_{1}}}{{\rho\_{0}}}}{\nicefrac{{L\_{1}}}{{L\_{0}}}}\right)^{2/3}\overset{\text{\Hy@raisedlink{\hypertarget{a0}{}}{(\hyperlink{desca0}{a})}}}{\approx}B\_{0}^{\star}S\_{0}^{\star}, |  | (10) |

where \Hy@raisedlink([a](#a0 "\Hy@raisedlink ‣ Equation 10 ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) holds if problem constants do not change significantly, that is, the effective batch–sequence scale for the large model should closely match that of the small model when the problem-dependent constants do not vary substantially with the model size (see [Section˜6.4](#S6.SS4 "6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

From ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), the Frank–Wolfe stepsize should be chosen as

|  |  |  |  |
| --- | --- | --- | --- |
|  | β1​=\Hy@raisedlink([a](#desca1 "\Hy@raisedlink ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​β0⋆​B1​S1B0⋆​S0⋆=β0⋆​(μ1/μ0⋅ρ1/ρ0L1/L0)2/3​≈\Hy@raisedlink([b](#descb1 "\Hy@raisedlink ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​β0⋆,\beta\_{1}\overset{\text{\Hy@raisedlink{\hypertarget{a1}{}}{(\hyperlink{desca1}{a})}}}{=}\beta\_{0}^{\star}\frac{B\_{1}S\_{1}}{B\_{0}^{\star}S\_{0}^{\star}}=\beta\_{0}^{\star}\left(\frac{\nicefrac{{\mu\_{1}}}{{\mu\_{0}}}\cdot\nicefrac{{\rho\_{1}}}{{\rho\_{0}}}}{\nicefrac{{L\_{1}}}{{L\_{0}}}}\right)^{2/3}\overset{\text{\Hy@raisedlink{\hypertarget{b1}{}}{(\hyperlink{descb1}{b})}}}{\approx}\beta\_{0}^{\star}, |  | (11) |

where \Hy@raisedlink([a](#a1 "\Hy@raisedlink ‣ Equation 11 ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) holds since the first stage involves T0T\_{0} tokens to train a larger model; \Hy@raisedlink([b](#b1 "\Hy@raisedlink ‣ Equation 11 ‣ First stage (training with 𝑇₍₁₎=𝑇₀ tokens). ‣ 5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) holds when the problem constants do not change significantly.
Such a choice of the Frank–Wolfe stepsize is also recommended by the μ\muP literature, which advocates keeping the learning rate fixed when the token budget and batch configuration are unchanged.

##### Second stage (training with the full budget T(1)+T(2)T\_{(1)}+T\_{(2)}).

Next, we receive an additional T(2)T\_{(2)} tokens.
Eq. ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) suggests that we should expect the optimization error to improve from order T0−1/3T\_{0}^{-1/3} at the end of the first stage to order T1−1/3T\_{1}^{-1/3} at the end of the second stage.
To realize this improvement in practice, we switch to using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) during the second stage, with the full token budget T1=T(1)+T(2)T\_{1}=T\_{(1)}+T\_{(2)}.

Overall, this hyperparameter restart strategy for Scion suggests selecting the batch size, sequence length, and Frank–Wolfe stepsize based on the total number of tokens that will ultimately be available to the model.
If additional tokens arrive at later times, the same procedure can be repeated: the batch size and sequence length are increased accordingly, and the Frank–Wolfe stepsize is adjusted based on the final token budget that the larger model will observe.

### 5.4 Guidelines for Practitioners

We summarize all the details on how to adjust the optimizer’s parameters under the BST scaling rule below to facilitate its implementation in practice.

#### 5.4.1 Hyperparameter Scaling: From Small to Large Models

In this scenario, the model size changes. Therefore, we need to account for a change of optimization problem constants, such as L,μ,ρL,\mu,\rho. We summarize the resulting procedure below:

1. 1.

   Obtain optimal values of the batch size B0⋆B\_{0}^{\star} and sequence length S0⋆S\_{0}^{\star}, Frank–Wolfe stepsize β0⋆\beta\_{0}^{\star} by tuning a small model, while setting momentum parameter α\alpha and radii η\eta to default values.
2. 2.

   Estimate the problem constants L0,μ0,ρ0L\_{0},\mu\_{0},\rho\_{0} and L1,μ1,ρ1L\_{1},\mu\_{1},\rho\_{1} for small and large models, respectively, based on the fitted power laws ([15](#S6.E15 "Equation 15 ‣ Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).
3. 3.

   Choose batch size B1B\_{1}, sequence length S1S\_{1}, and Frank–Wolfe stepsize β1\beta\_{1} for larger model using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), namely

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | B1​S1=B0⋆​S0⋆​(T1T0​μ1μ0​ρ1ρ0L1L0)2/3,β1=β0⋆​(T0T1​μ1μ0​ρ1ρ0L1L0)2/3,B\_{1}S\_{1}=B\_{0}^{\star}S\_{0}^{\star}\left(\frac{\frac{T\_{1}}{T\_{0}}\frac{\mu\_{1}}{\mu\_{0}}\frac{\rho\_{1}}{\rho\_{0}}}{\frac{L\_{1}}{L\_{0}}}\right)^{2/3},\quad\beta\_{1}=\beta\_{\rm 0}^{\star}\left(\frac{\frac{\sqrt{T\_{\rm 0}}}{\sqrt{T\_{\rm 1}}}\frac{\mu\_{\rm 1}}{\mu\_{\rm 0}}\frac{\rho\_{\rm 1}}{\rho\_{\rm 0}}}{\frac{L\_{\rm 1}}{L\_{\rm 0}}}\right)^{2/3}, |  | (12) |

   while keeping radii η\eta and momentum α\alpha unchanged.
4. 4.

   Use new parameters to train a larger model (either from the beginning or after processing the token budget used for tuning a smaller model).

#### 5.4.2 Hyperparameter Scaling: From Small to Large Token Budget

Now assume that the model size remains the same, but the token budget increases. Therefore, the constants LL and μ\mu remain the same, while we need to account for a change of ρ\rho with batch size.

1. 1.

   Obtain optimal values of the batch size B0B\_{0} and sequence length S0S\_{0}, Frank–Wolfe stepsize β0\beta\_{0} by tuning a model for a smaller token budget, while setting momentum parameter α\alpha and radii η\eta to default values.
2. 2.

   Estimate the problem constants ρ0\rho\_{0} and ρ1\rho\_{1} for small and large token budgets, respectively, based on the fitted power laws ([15](#S6.E15 "Equation 15 ‣ Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).
3. 3.

   Choose batch size B1B\_{1}, sequence length S1S\_{1}, and Frank–Wolfe stepsize β1\beta\_{1} for a larger token budget T1T\_{1} using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), namely

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | B1​S1=B0⋆​S0⋆​(T1T0​ρ1ρ0)2/3,β1=β0⋆​(T0T1​ρ1ρ0)2/3,B\_{1}S\_{1}=B\_{0}^{\star}S\_{0}^{\star}\left(\frac{T\_{1}}{T\_{0}}\frac{\rho\_{1}}{\rho\_{0}}\right)^{2/3},\quad\beta\_{1}=\beta\_{\rm 0}^{\star}\left(\frac{\sqrt{T\_{\rm 0}}}{\sqrt{T\_{\rm 1}}}\frac{\rho\_{\rm 1}}{\rho\_{\rm 0}}\right)^{2/3}, |  | (13) |

   while keeping radii η\eta and momentum α\alpha unchanged.
4. 4.

   Use new parameters to train a model for a longer horizon T1T\_{1} (either from the beginning or after processing the token budget used for a smaller model).

## 6 Experiments

In this section, we empirically evaluate our theoretical results by training a modded NanoGPT model on the FineWeb dataset, following the experimental setup of Pethick et al. ([2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")) and based on the codebase of Jordan et al. ([2024a](#bib.bib91 "Moddednanogpt: speedrunning the nanogpt baseline")). Details are given in [Appendix˜A](#A1 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). For Scion, we adopt the recommended operator norms (Sign →\to Spectral →\to Sign): we choose the radius η=3000\eta=3000 for sign-updated layers and η=50\eta=50 for matrix-type layers. Concretely, this corresponds to using the polar factor of the gradient for matrix-valued parameters and the elementwise sign of the gradient for all other parameter types (cf., (Pethick et al., [2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs"))).

### 6.1 Verification of Assumption [3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")

First, we empirically test the validity of [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") when training a 124M base model with Scion for a fixed number of iterations K=5100K=5100.
To approximate the gradient variance as a function of the batch size BB, we sample mm mini-batch gradients of size BB such that m​B=32768mB=32768, and compute the empirical variance across the sampled mm mini-batch gradients.
We track the evolution of this empirical variance over training in [Figure˜B.1](#A2.F1 "In B.1 Verification of ˜3.4 ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") and observe that it stabilizes rapidly after a short initial transient phase.
In [Figure˜2](#S4.F2 "In 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we report the final empirical variance values measured at the end of training.
The fitted power-law relationships support σ2∼1B​S\sigma^{2}\sim\frac{1}{BS} as a reasonable working approximation in the regime B​S≪TBS\ll T.

### 6.2 Verification of Assumption [3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")

Second, we conduct experiments to assess the validity of [˜3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") in practice.
We use the same experimental setup as in the previous section and track both the dual norm of mini-batch gradients and the corresponding mini-batch training loss throughout training.
When using Scion, the primal and dual norms are defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖x‖=maxℓ∈[N]⁡‖xℓ‖ℓ,‖x‖∗=∑ℓ=1N‖xℓ‖∗,ℓ,\|x\|=\max\_{\ell\in[N]}\|x\_{\ell}\|\_{\ell},\quad\|x\|\_{\*}=\sum\_{\ell=1}^{N}\|x\_{\ell}\|\_{\*,\ell}, |  | (14) |

where ‖xℓ‖ℓ\|x\_{\ell}\|\_{\ell} and ‖xℓ‖∗,ℓ\|x\_{\ell}\|\_{\*,\ell} denote the primal and dual norms of the ℓ\ell-th layer of the network with NN layers, respectively.
Their precise definitions are provided in Table 2 (second and third columns) of Pethick et al. ([2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")). See also the recent work by Crawshaw et al. ([2025](#bib.bib132 "An exploration of non-euclidean gradient descent: muon and its many variants")).

We report the joint evolution of the dual gradient norm and the training loss over the course of training in [Figure˜1](#S3.F1 "In 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
We observe that, once the training loss falls below approximately 55, the data points closely follow a linear relationship, empirically supporting the use of [˜3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") in this setting.
To quantify this relationship, we estimate the slope using a robust linear regression model with Huber loss, which interpolates between least squares and absolute-error (ℓ1\ell\_{1}) regression and thereby reduces sensitivity to outliers.

Table 1: Final validation loss when training a 124M NanoGPT model varying the batch size while keeping the validation and train sequence lengths 10241024 under the token budget 1.31.3B (TPP 10.8). We report the average across 55 runs along with a standard deviation. Bold numbers indicate the best performance in the column. The runs in red indicate the best configuration of batch size, sequence length, and Frank–Wolfe stepsize across all runs for a given token budget.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T=1.3T=1.3B | Batch Size | | | | | | |
| 𝜷,×10−4\boldsymbol{\beta},\times 10^{-4} | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| 1.2 | 3.4258±0.0004 | 3.3889±0.0012 | 3.3857±0.0013 | 3.4074±0.0010 | 3.4587±0.0010 | 3.5598±0.0012 | 3.7715±0.0013 |
| 2.4 | 3.4394±0.0007 | 3.3880±0.0043 | 3.3706±0.0019 | 3.3801±0.0016 | 3.4144±0.0004 | 3.4940±0.0009 | 3.6694±0.0021 |
| 3.6 | 3.4554±0.0008 | 3.3945±0.0015 | 3.3717±0.0020 | 3.3765±0.0017 | 3.4065±0.0007 | 3.4799±0.0017 | 3.6472±0.0023 |
| 4.8 | 3.4766±0.0016 | 3.4072±0.0048 | 3.3790±0.0013 | 3.3807±0.0006 | 3.4115±0.0025 | 3.4945±0.0038 | 3.6611±0.0020 |
| 6.0 | 3.4967±0.049 | 3.4198±0.0002 | 3.3875±0.0019 | 3.3887±0.0024 | 3.4202±0.0022 | 3.5005±0.0038 | 3.7013±0.0160 |
| 7.2 | 3.5151±0.0007 | 3.4301±0.0001 | 3.3978±0.0026 | 3.3960±0.0022 | 3.4331±0.0025 | 3.5270±0.0071 | 3.7506±0.0271 |




Table 2: Final validation loss when training a 124M NanoGPT model varying the train sequence length while keeping the batch size 256256 under the token budget 1.31.3B. The validation sequence length is always 10241024. We report the average across 55 runs along with a standard deviation. ∗ indicates that not all runs had a stable decrease in validation loss. Bold numbers indicate the best performance in the column. The runs in red indicate the best configuration of batch size, sequence length, and Frank–Wolfe stepsize across all runs for a given token budget.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| T=1.3T=1.3B | Sequence Length | | | | |
| 𝜷,×10−4\boldsymbol{\beta},\times 10^{-4} | 256 | 512 | 1024 | 2048 | 4096 |
| 1.2 | 3.7076±0.0084 | 3.4647±0.0240 | 3.4587±0.0010 | 3.4126±0.0014 | 3.4811±0.0025 |
| 2.4 | 3.9622±0.0585 ∗ | 3.4633±0.0091 | 3.3706±0.0019 | 3.3834±0.0026 | 3.4299±0.0021 |
| 3.6 | 4.0441±0.2055 ∗ | 3.4796±0.0131 | 3.3717±0.0020 | 3.3792±0.0022 | 3.4216±0.0013 |
| 4.8 | 3.9292±0.0852 ∗ | 3.5004±0.0063 | 3.3790±0.0013 | 3.3829±0.0020 | 3.4243±0.0029 |
| 6.0 | 3.9901±0.0170 ∗ | 3.5134±0.0059 | 3.3875±0.0019 | 3.3910±0.0024 | 3.4374±0.0037 |
| 7.2 | 3.9819±0.1195 ∗ | 3.5269±0.0187 | 3.3960±0.0022 | 3.3987±0.0029 | 3.4537±0.0030 |

### 6.3 Ablations on Batch Size and Sequence Length

We conduct ablation studies by varying the batch size BB and sequence length SS to identify the optimal Frank–Wolfe stepsize β\beta for Scion when training a base 124M model with a fixed validation sequence length 10241024. We report results under a fixed token budget of 1.31.3B in Tables [1](#S6.T1 "Table 1 ‣ 6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") and [2](#S6.T2 "Table 2 ‣ 6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). This corresponds to TPP ratio of 10.810.8 (approximately 0.5×0.5\times the Chinchilla optimum).

We observe that once the batch size (or sequence length) is sufficiently large, the optimal Frank–Wolfe stepsize stabilizes at 3.6⋅10−43.6\cdot 10^{-4}.
Moreover, the results indicate that, for the base model, the optimal batch size and sequence length are approximately 256256 and 10241024, yielding the lowest validation loss. Additionally, for significantly short train sequence lengths of 256256, most runs were unstable and exhibited high standard deviations since the validation loss is 10241024. We also observe that the best performance between batch sizes 256256 and 512512 differs little, indicating that performance is almost batch-independent. This aligns with [Corollary˜4.1](#S4.Thmcorollary1 "Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), which shows that there exists a batch-independent regime.

### 6.4 Estimating Problem-Dependent Constants

In our next experiment, we estimate the problem-dependent constants LL, μ\mu, and ρ\rho across different model configurations in order to track how these quantities change with model size.
Specifically, we train models using a fixed Frank–Wolfe stepsize β=3.6⋅10−4\beta=3.6\cdot 10^{-4}, batch size B=512B=512, and sequence length S=1024S=1024 for 51005100 iterations, following the ablation study in [Section˜6.3](#S6.SS3 "6.3 Ablations on Batch Size and Sequence Length ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), while varying the number of layers n\_layer and the embedding dimension n\_embd. In this section, we ignore the change in the constants L,μ,ρL,\mu,\rho with the batch size, but later we account for this dependency in the hyperparameter transfer. The estimated values are reported in [Appendix˜B](#A2 "Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
The estimation procedure is carried out as follows.

##### Smoothness constant LL.

To estimate the smoothness constant, we measure the following ratio

|  |  |  |
| --- | --- | --- |
|  | ‖g​(xk;ξk)−g​(xk−1;ξk−1)‖∗‖xk−xk−1‖,\frac{\|g(x\_{k};\xi\_{k})-g(x\_{k-1};\xi\_{k-1})\|\_{\*}}{\|x\_{k}-x\_{k-1}\|}, |  |

where g​(xk;ξk)g(x\_{k};\xi\_{k}) and g​(xk−1;ξk−1)g(x\_{k-1};\xi\_{k-1}) denote the mini-batch gradients at two consecutive iterations, and the norms are defined as in the previous section.
This quantity has been used in prior work as a proxy for local curvature during training (Alimisis et al., [2025](#bib.bib82 "Why do we need warm-up? a theoretical perspective"); Zhang and Sennrich, [2019](#bib.bib53 "Root mean square layer normalization"); Riabinin et al., [2025](#bib.bib3 "Gluon: making Muon & Scion great again! (Bridging theory and practice of LMO-based optimizers for LLMs)")).
As a final estimate of LL, we average the measured ratio over the last 100100 iterations.

##### KL condition constant μ\mu.

The estimation of μ\mu follows the same procedure as in [Section˜6.2](#S6.SS2 "6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
In particular, we fit a robust linear regression model with Huber loss to the relationship between the dual gradient norm and the training loss, and use the resulting slope as an estimate of μ\mu.

##### Norm-equivalence constant ρ\rho.

In the proof of [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we apply [˜3.2](#S3.Thmassumption2 "Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") to bound terms of the form

|  |  |  |
| --- | --- | --- |
|  | ‖g​(xk;ξk)−∇f​(xk)‖∗≤ρ​‖g​(xk;ξk)−∇f​(xk)‖2.\|g(x\_{k};\xi\_{k})-\nabla f(x\_{k})\|\_{\*}\leq\rho\,\|g(x\_{k};\xi\_{k})-\nabla f(x\_{k})\|\_{2}. |  |

To approximate the full gradient ∇f​(xk)\nabla f(x\_{k}), we follow the same procedure described in [Section˜6.1](#S6.SS1 "6.1 Verification of Assumption 3.4 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
We track the ratio between the dual norm and the Euclidean norm throughout training, and report the average of this ratio over the last 100100 iterations as an estimate of ρ\rho.

We conduct the estimation procedure for several model configurations and fit a shifted power law666The choice of the fitting model is flexible, and alternative functional forms could also be considered. We leave the exploration of other functional dependencies to future work. for the problem constants L,μ,ρL,\mu,\rho of the form:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | μ​(n\_layer,n\_embd)=5.2​(n\_layer+1.7)−0.2;\displaystyle\mu(\texttt{n\\_layer},\texttt{n\\_embd})=5.2(\texttt{n\\_layer}+1.7)^{-0.2}; |  | (15) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | L​(n\_layer,n\_embd)=0.4​(n\_layer+0.7)0.2​(n\_embd+126)0.35;\displaystyle L(\texttt{n\\_layer},\texttt{n\\_embd})=0.4(\texttt{n\\_layer}+0.7)^{0.2}(\texttt{n\\_embd}+126)^{0.35}; |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ρ​(n\_layer,n\_embd,batch\_size)=4.1​(n\_layer−2.7)0.25​(n\_embd−250.8)0.3​(batch\_size−9.4)0.1.\displaystyle\rho(\texttt{n\\_layer},\texttt{n\\_embd},\texttt{batch\\_size})=4.1(\texttt{n\\_layer}-2.7)^{0.25}(\texttt{n\\_embd}-250.8)^{0.3}(\texttt{batch\\_size}-9.4)^{0.1}. |  |

Table 3: Estimated problem-dependent constants, assuming that they change with the number of layers n\_layer, embedding dimension n\_embd, and batch size batch\_size according to ([15](#S6.E15 "Equation 15 ‣ Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). The estimations of the change for β\beta and B​SBS are based on ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | 𝑳\boldsymbol{L} | 𝝁\boldsymbol{\mu} | 𝝆\boldsymbol{\rho} | |  | | --- | | How to change β\boldsymbol{\beta} | | w.r.t. 124M model? | | |  | | --- | | How to change B​S\boldsymbol{B}\boldsymbol{S} | | w.r.t. 124M model? | |
| 124M | 7.27.2 | 3.13.1 | 62.762.7 | 1×1\times | 1×1\times |
| 1B | 10.610.6 | 2.92.9 | 111.9111.9 | |  | | --- | | ↘0.5×(a)\searrow 0.5\times\penalty 10000\ {}^{(a)} | | ↘0.54×(b)\searrow 0.54\times\penalty 10000\ {}^{(b)} | | |  | | --- | | ↗4×(a)\nearrow 4\times\penalty 10000\ {}^{(a)} | | ↗4.37×(b)\nearrow 4.37\times\penalty 10000\ {}^{(b)} | |

* •

  (a) Taking into account the practical requirement that BB and SS should be powers of two. We increase the product B​SBS rounding to the closest power of two.
* •

  (b) Ignoring the practical requirement that BB and SS should be powers of two.

Interestingly, the constant μ\mu decreases with n\_layer, while it remains unchanged with n\_embd. In contrast, the constants LL and ρ\rho increase with both n\_layer and n\_emdb. Using the fitted power laws, we estimate the constants for 22 model configurations used in the experiments of size 124M and 1B.

We observe that the problem-dependent constants vary slowly with model size and remain relatively stable across the model configurations we consider. Although we use these estimates in subsequent experiments, neglecting this variation does not significantly affect the resulting derivations. The impact of these changes becomes more pronounced only in regimes where D1≫D0D\_{1}\gg D\_{0}. Using estimated constants, we characterize how the Frank–Wolfe stepsize β\beta and the product B​SBS should be set for the 1B model, knowing the optimal configuration (B=256,S=1024,β=3.6⋅10−4B=256,S=1024,\beta=3.6\cdot 10^{-4}) for the 124M model in [Table˜3](#S6.T3 "In Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") and using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Note that we provide two configurations for the 1B model: whether the practical requirement that the batch size and sequence length be powers of 22 should be taken into account.

### 6.5 Increasing Batch Size and Sequence Length during Training

Next, we evaluate the proposed strategy from [Section˜5.3](#S5.SS3 "5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
We use the 124M model as a base model, for which we previously identified batch size B0=256B\_{0}=256, sequence length S0=1024S\_{0}=1024, and Frank–Wolfe stepsize β0=3.6⋅10−4\beta\_{0}=3.6\cdot 10^{-4} as providing the best performance under a token budget T0=1.3T\_{0}=1.3B. We then consider training a larger 1B model under a total token budget of T1=10.8T\_{1}=10.8B (the same TPP) using the following strategies:

* •

  Restarted Scion.
  We follow the strategy described in [Section˜5.3](#S5.SS3 "5.3 Batch Size Scheduling ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
  During the first stage, corresponding to the initial T0T\_{0} tokens, we use batch size B0=256B\_{0}=256 and sequence length S0=1024S\_{0}=1024 with stepsize β0=3.6⋅10−4\beta\_{0}=3.6\cdot 10^{-4}.
  After processing T0T\_{0} tokens, we increase the product B​SBS four times and consider two restarted schemes: B1=512B\_{1}=512 and sequence length S1=2048S\_{1}=2048 (in yellow) and B2=1024B\_{2}=1024 and sequence length S2=1024S\_{2}=1024 (in gray), both with stepsize β1=β0/2=1.8⋅10−4\beta\_{1}=\beta\_{0}/2=1.8\cdot 10^{-4} for the remaining token budget,777The batch size, sequence length, and Frank–Wolfe stepsize used in the first and second training stages are determined using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and reported in [Table 3](#S6.T3 "In Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). following the derivations in ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), with estimates of the problem-dependent constants taken from [Section˜6.4](#S6.SS4 "6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* •

  Fixed *tuned*-batch Scion.
  We train the 1B model using the *tuned* batch size B0=256B\_{0}=256, which was obtained on a smaller 124M model. We set the sequence length S0=1024S\_{0}=1024 with Frank–Wolfe stepsize β0=3.6⋅10−4\beta\_{0}=3.6\cdot 10^{-4} over the entire horizon T1=10.8T\_{1}=10.8B (in light blue). This configuration is motivated by hyperparameter transfer results under the μ\muP framework, where the hyperparameters tuned for a smaller model are used when training a larger model.
* •

  Fixed *large*-batch Scion.
  We train the 1B model using a larger batch size–sequence-length product. In particular, we consider two settings. For B1=512,S1=2048,B\_{1}=512,S\_{1}=2048, we evaluate two baselines trained from the beginning over the full token budget T1=10.8T\_{1}=10.8B: one with Frank–Wolfe stepsize β0\beta\_{0} (in orange), suggested by the μ\muP framework, and one with stepsize β1\beta\_{1} (in pink), suggested by ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). For B2=1024andS2=1024,B\_{2}=1024\quad\text{and}\quad S\_{2}=1024, we again train from the beginning over the full token budget T1=10.8T\_{1}=10.8B, and consider two baselines: one with stepsize β0\beta\_{0} (in blue), suggested by the μ\muP framework, and one with stepsize β1\beta\_{1} (in green), suggested by ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

|  |
| --- |
| Refer to caption |

Figure 3: Comparison of batch size and sequence length scheduling strategies when training a 1B model. The restarting schemes (in yellow and gray) are compared against fixed schedules. The validation loss is evaluated with a smaller sequence length of 10241024. The values of batch sizes B0,1,2B\_{0,1,2}, sequence lengths S0,1S\_{0,1}, and Frank–Wolfe stepsizes β0,1\beta\_{0,1} are given in the legends. The notation (B0,1,2,S0,1,2,β0,1)(B\_{0,1,2},S\_{0,1,2},\beta\_{0,1}) characterizes which batch size, sequence length, and Frank–Wolfe stepsize are used for the particular setup. The notation (B0,S0,β0)→(B1,2,S1,2,β1)(B\_{0},S\_{0},\beta\_{0})\to(B\_{1,2},S\_{1,2},\beta\_{1}) characterizes how parameters of Scion change after restart (e.g., batch size increases from B0B\_{0} to B1,2B\_{1,2}), respectively. The notation μ\muP or BST indicates the rule used to select B,SB,S, and β\beta.

###### Remark 6.1.

We note that the choice of batch size BB and sequence length SS in this set of experiments is partially guided by practical considerations, as these values are typically selected as powers of 22. We follow this convention to evaluate the performance of the restarted and small- and large-batch baselines in a setting that more closely reflects real-world practice. However, in the later experiments aimed at demonstrating hyperparameter transfer, we select BB and SS strictly according to the BST scaling rule, ignoring the aforementioned practical constraints.

Based on the results in [Figure˜3](#S6.F3 "In 6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we can make the following claims.

1. 1.

   The μ\muP framework, where all parameters of the algorithm, i.e., Frank–Wolfe stepsize, batch size, and sequence length remain unchanged, achieves the worst performance. This result demonstrates the limitation of the μ\muP framework, which ignores changes of batch size and sequence length. Our BST scaling instead suggests increasing the product B​SBS that leads to enhanced performance.

   Finding 1. The training configuration suggested by the μ\muP framework becomes sub-optimal when the batch size and/or sequence length increase.
2. 2.

   Both restarting strategies for Scion demonstrate competitive performance compared to the other baselines. After the restart, both variants show accelerated improvement in training and validation loss relative to the baselines. Moreover, the training curves of the restarted Scion models remain consistently below those of the other methods from the restart point onward.

   Finding 2. Restarting strategies improve performance in comparison to other fixed large-batch baselines. Eventually, both restarted and fixed large-batch variants of Scion match in performance at the end of the training.
3. 3.

   In addition, we observe that quadrupling the batch size while keeping the sequence length fixed performs slightly better than doubling both the batch size and the sequence length. In this setting, the former strategy achieves approximately a 0.010.01 lower validation loss than the latter.

   Finding 3. Increasing batch size while keeping sequence length might be slightly more preferable than increasing both parameters simultaneously if context extension is not necessary. Otherwise, extending the sequence length (i.e., adding this capability) must be compensated for by the batch size for optimization efficiency.
4. 4.

   All large-batch baselines (with both values of Frank–Wolfe stepsize: β0\beta\_{0} suggested by μ\muP and β1\beta\_{1} suggested by BST rule) achieve similar performance. The best performance is achieved when we double the batch size and the sequence length with Frank-Wolfe stepsize set according to ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). The other three baselines are slightly worse and achieve the validation loss 0.005−0.010.005-0.01 higher.

   Finding 4. Large batch size 512512 is sub-optimal for a smaller model with TPP 10.610.6, but becomes preferable for a larger model with the same TPP, highlighting limitations of the μ\muP framework. Even if we tune the smaller model with the larger batch for μ\muP, the step-size configuration is suboptimal for the larger model. This issue is further confounded by the fact that we often do not know the optimal batch size for the larger model since it depends on the token horizon.

   ### 6.6 Increasing Batch Size Further Does Not Help

   Next, we investigate whether increasing the product B​SBS by 88 or 1616 times (when fixing a train sequence length 10241024, this results in batch sizes 20482048 and 40964096) yields additional benefits when training a larger 1B model. All experiments are conducted with fixed training and validation sequence lengths of 10241024. We report results using both Frank–Wolfe stepsizes suggested by the μ\muP framework and those determined by our BST scaling rule.

   The results are shown in [Figure˜4](#S6.F4 "In 6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). We observe that Scion with a batch size of 10241024 outperforms the baselines with batch sizes 20482048 and 40964096. This finding suggests that our BST scaling rule, which prescribes how to scale the product B​SBS, provides a reliable practical guideline. Increasing the product B​SBS beyond this recommendation does not yield further performance gains. In particular, the validation loss for Scion with batch size 20482048 worsens by approximately 0.0050.005-0.010.01, while for batch size 40964096 the degradation is more pronounced, around 0.030.03-0.040.04.

   Finding 5. Increasing the batch size beyond 10241024, suggested by the BST scaling rule, worsens performance gains.

|  |
| --- |
| Refer to caption |

Figure 4: Comparison of fixed large batch size strategies when training a 1B model. The validation loss is evaluated with a smaller sequence length 10241024. Scion with a batch size of 10241024 suggested by our BST scaling rule achieves the best performance compared to other baselines with batch sizes 20482048 and 40964096. The values of batch sizes B1,2,3B\_{1,2,3}, sequence lengths SS, and Frank–Wolfe stepsizes β0,1\beta\_{0,1} are given in the legends. The notation (B1,2,3,S,β0,1)(B\_{1,2,3},S,\beta\_{0,1}) characterizes which batch size, sequence length, and Frank–Wolfe stepsize are used for the particular setup, respectively. The notation BST indicates the rule used to select the B,SB,S, and β\beta.



|  |
| --- |
| Refer to caption |

Figure 5: The final performance of the 124M model when varying the Frank–Wolfe stepsize β\beta under different token budgets (left: 2.7B, center: 5.3B, right: 8.0B). We average the train loss over 3 random seeds and report the moving average in the window of size 500. We observe that the BST scaling rule predicts a good estimate for the optimal β\beta when increasing the token budget. Moreover, the difference in performance between BST and μ\muP baselines grows with a token budget.



|  |
| --- |
| Refer to caption |

Figure 6: The final performance of the 124M model when varying the momentum α\alpha under different token budgets (left: 1.3B, center left: 2.7B, center right: 5.3B, right: 8.0B). We average the train loss over 3 random seeds and report the moving average in the window of size 500. We observe that rule momentum parameter α\alpha transfers under BST scaling.



|  |
| --- |
| Refer to caption |

Figure 7: The final performance of the 1B model when varying the Frank–Wolfe stepsize β\beta (left) and momentum parameter (right) under different token budget 10.6 TPP. We report the final train loss, smoothed in the window of size 500500. We observe that the BST scaling rule predicts a good estimate for both optimal α\alpha and β\beta when transferring from a smaller 124M model to a larger 1B model.

### 6.7 Hyperparameter Transfer

From [Table˜1](#S6.T1 "In 6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we know that for a base 124M model, the optimal set of hyperparameters is B=256,S=1024,β=3.6⋅10−4B=256,S=1024,\beta=3.6\cdot 10^{-4} under token budget T=1.3T=1.3B (TPP 10.8). We want to use these parameters in BST rule to obtain them for a larger training horizon or model size. In this section, the reported train losses are averaged over 3 random seeds (only for 124M) and smoothed using running average in the window of size 500 (for both 124M and 1B models). We ignore the requirement for BB to be the powers of 22 in these set of experiments.

#### 6.7.1 Increasing Token Budget for 124M Model

In this section, we report the pretraining results for 124M model under increased token budgets (i)(i) T=2.7T=2.7B (TPP 21.6), (i​i)(ii) T=5.3T=5.3B (TPP 43.1), and (i​i​i)(iii) T=8.0T=8.0B (TPP 64.7). We set a batch size for longer horizons using ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")): B=416B=416 for T=2.7T=2.7B, B=672B=672 for T=5.3T=5.3B, and B=896B=896 for T=8.0T=8.0B, using estimates of problem-dependent constants from ([3](#S6.T3 "Table 3 ‣ Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Momentum and sequence length are set to α=0.1\alpha=0.1 and S=1024S=1024, respectively.

To demonstrate the predictive power of the BST scaling rule in finding optimal Frank–Wolfe stepsize β\beta in ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), we report the final train losses when varying β\beta under three token budgets. Momentum parameter is fixed to α=0.1\alpha=0.1. We expect the optimal Frank–Wolfe stepsize to be around (i)(i) β=3.0⋅10−4\beta=3.0\cdot 10^{-4}, (i​i)(ii) β=2.4⋅10−4\beta=2.4\cdot 10^{-4}, and (i​i​i)(iii) β=2.1⋅10−4\beta=2.1\cdot 10^{-4}. In [Figure˜5](#S6.F5 "In 6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we observe that the BST scaling rule predicts Frank–Wolfe stepsize close to the optimal one in all the cases. Moreover, we observe that μ\muP baseline (B=256,S=1024,β=3.6⋅10−4)(B=256,S=1024,\beta=3.6\cdot 10^{-4}) becomes more suboptimal when increasing the token budget, which demonstrates the limitations of the μ\muP framework even further.

Next, we switch to testing the BST rule for predicting the optimal value of the momentum parameter α\alpha. According to the BST rule, α\alpha should transfer. Empirical results in [Figure˜6](#S6.F6 "In 6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") support this claim. For all values of the token budget, the optimal α\alpha is close to 0.090.09.

#### 6.7.2 Increasing Model Size

Now we want to train a 1B model with batch size B=1120,S=1024B=1120,S=1024 under token budget 10.810.8B (TPP 10.8). In this setup, we test the predictive power of the BST scaling rule when we change the model size. The value of the batch size is set according to ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), using estimates from [Table˜3](#S6.T3 "In Norm-equivalence constant 𝜌. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). We expect the optimal Frank–Wolfe stepsize to be close to 1.95⋅10−41.95\cdot 10^{-4}, while the momentum parameter to be close to 0.090.09. We report the results in [Figure˜7](#S6.F7 "In 6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). We observe that the BST rule provides a good estimation for both the optimal momentum α\alpha and Frank–Wolfe stepsize β\beta, when increasing the model size.

## 7 Conclusion

We developed a token-budget–aware theory for scaling batch size, sequence length, and Frank–Wolfe stepsize in SCG methods under a μ\mu-KL condition.
Our analysis reveals a non-monotone dependence of optimization error on the effective batch–sequence scale and yields a principled BST-scaling rule that identifies when increasing batch size is beneficial and when it becomes suboptimal.
In contrast to hyperparameter transfer approaches that ensure local stability at initialization, our results characterize long-horizon, trajectory-level behavior and explain how hyperparameters should adapt as the token budget grows.
Empirically, we show that large batches are not inherently detrimental: when scaled according to our theory, jointly adapting (B,S,β)(B,S,\beta) improves both token efficiency and convergence in large-scale training.

## Limitations and Future Work

Note that in our experiments, the remaining hyperparameters, such as the radii η\eta and the variance initialization, were adopted directly from Pethick et al. ([2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")) without additional tuning. This configuration already yields strong empirical performance for Scion. However, for other model architectures, these hyperparameters may not be readily available. In such cases, we recommend selecting them based on prior literature or performing a small hyperparameter sweep. We expect their precise choice to be less critical for final performance than that of the batch size or the Frank–Wolfe stepsize.

More generally, substantially suboptimal choices of these hyperparameters may affect the predictive accuracy of our BST scaling rule. Determining the minimal set of hyperparameters that must be tuned on a small model to ensure that our theoretical predictions remain practically actionable remains an important open question, which we leave for future work.

Another important question that requires further investigation is the transfer of the momentum parameter. In our experiments, we observe that increasing the model size slightly shifts the range of near-optimal values of the momentum parameter α\alpha to lower values (0.060.06–0.080.08), compared to the range (0.080.08–0.10.1) for the base 124M model. This shift may be due to the power-law fits being based on an insufficient number of data points, suboptimal functional dependency, or because additional training parameters, such as the token budget, should be incorporated into the scaling analysis.

## Acknowledgement

Rustem Islamov and Aurelien Lucchi acknowledge the financial support of the Swiss National Science Foundation, SNSF grant No 207392. Volkan Cevher acknowledges the financial support of the Swiss National Science Foundation, SNSF grant No 240094. This work was also supported under project ID # 37 as part of the Swiss AI Initiative, through a grant from the ETH Domain and computational resources provided by the Swiss National Supercomputing Centre (CSCS) under the Alps infrastructure.

## References

* F. Alimisis, R. Islamov, and A. Lucchi (2025)
  Why do we need warm-up? a theoretical perspective.
  arXiv preprint arXiv:2510.03164.
  Cited by: [§6.4](#S6.SS4.SSS0.Px1.p1.4 "Smoothness constant 𝐿. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Bergsma, N. S. Dey, G. Gosal, G. Gray, D. Soboleva, and J. Hestness (2025)
  Power lines: scaling laws for weight decay and batch size in LLM pre-training.
  In The Thirty-ninth Annual Conference on Neural Information Processing Systems,
  Cited by: [§1](#S1.p3.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. (2024)
  Deepseek llm: scaling open-source language models with longtermism.
  arXiv preprint arXiv:2401.02954.
  Cited by: [§2](#S2.SS0.SSS0.Px4.p2.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Bolte, A. Daniilidis, and A. Lewis (2007)
  The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems.
  SIAM Journal on Optimization.
  Cited by: [§1](#S1.p6.3 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Bolte, S. Sabach, and M. Teboulle (2014)
  Proximal alternating linearized minimization for nonconvex and nonsmooth problems.
  Mathematical Programming 146 (1),  pp. 459–494.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* E. M. Compagnoni, R. Islamov, F. N. Proske, and A. Lucchi (2025a)
  Unbiased and sign compression in distributed learning: comparing noise resilience via sdes.
  arXiv preprint arXiv:2502.17009.
  Cited by: [§C.1](#A3.SS1.p1.18 "C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* E. M. Compagnoni, T. Liu, R. Islamov, F. N. Proske, A. Orvieto, and A. Lucchi (2025b)
  Adaptive methods through the lens of SDEs: theoretical insights on the role of noise.
  In The Thirteenth International Conference on Learning Representations,
  Cited by: [§C.1](#A3.SS1.p1.18 "C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* M. Crawshaw, C. Modi, M. Liu, and R. M. Gower (2025)
  An exploration of non-euclidean gradient descent: muon and its many variants.
  arXiv preprint arXiv:2510.09827.
  Cited by: [§6.2](#S6.SS2.p1.4 "6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* N. Dey, B. C. Zhang, L. Noci, M. Li, B. Bordelon, S. Bergsma, C. Pehlevan, B. Hanin, and J. Hestness (2025)
  Don’t be lazy: completep enables compute-efficient deep transformers.
  arXiv preprint arXiv:2505.01618.
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Works on Hyperparameter Transfer. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* I. Fatkhullin, J. Etesami, N. He, and N. Kiyavash (2022)
  Sharp analysis of stochastic optimization under global kurdyka-łojasiewicz inequality.
  Advances in Neural Information Processing Systems.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Ghadimi and G. Lan (2012)
  Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization i: a generic algorithmic framework.
  SIAM Journal on Optimization 22 (4),  pp. 1469–1492.
  Cited by: [§3](#S3.p7.1 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Ghadimi and G. Lan (2013)
  Stochastic first-and zeroth-order methods for nonconvex stochastic programming.
  SIAM journal on optimization 23 (4),  pp. 2341–2368.
  Cited by: [§3](#S3.p7.1 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He (2017)
  Accurate, large minibatch sgd: training imagenet in 1 hour.
  arXiv preprint arXiv:1706.02677.
  Cited by: [§1](#S1.p1.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§1](#S1.p10.1 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* C. Guille-Escuret, H. Naganuma, K. Fatras, and I. Mitliagkas (2023)
  No wrong turns: the simple geometry of neural networks optimization paths.
  arXiv preprint arXiv:2306.11922.
  Cited by: [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Guminov, A. Gasnikov, and I. Kuruzov (2017)
  Accelerated methods for α\alpha-weakly-quasi-convex problems.
  arXiv preprint arXiv:1710.00797.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* M. Hardt, T. Ma, and B. Recht (2018)
  Gradient descent learns linear dynamical systems.
  Journal of Machine Learning Research 19 (29),  pp. 1–44.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. (2022)
  Training compute-optimal large language models (2022).
  arXiv preprint arXiv:2203.15556.
  Cited by: [§2](#S2.SS0.SSS0.Px4.p2.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* R. Islamov, N. Ajroldi, A. Orvieto, and A. Lucchi (2024)
  Loss landscape characterization of neural networks without over-parametrization.
  Advances in Neural Information Processing Systems.
  Cited by: [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* K. Jordan, J. Bernstein, B. Rappazzo, Fernbear, Vlado, J. Yu, F. Cesista, B. Koszarsky, and Grad62304977 (2024a)
  Moddednanogpt: speedrunning the nanogpt baseline.
  Note: Version 2024a<https://github.com/KellerJordan/modded-nanogpt>
  Cited by: [§6](#S6.p1.4 "6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein (2024b)
  Muon: an optimizer for hidden layers in neural networks.
  Note: Blog postAccessed 2026-01-25
  External Links: [Link](https://kellerjordan.github.io/posts/muon/)
  Cited by: [Appendix A](#A1.p1.4 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§1](#S1.p5.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020)
  Scaling laws for neural language models.
  arXiv preprint arXiv:2001.08361.
  Cited by: [§2](#S2.SS0.SSS0.Px4.p2.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* H. Karimi, J. Nutini, and M. Schmidt (2016)
  Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition.
  In Joint European conference on machine learning and knowledge discovery in databases,
  Cited by: [§1](#S1.p6.3 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang (2017)
  On large-batch training for deep learning: generalization gap and sharp minima.
  In International Conference on Learning Representations (ICLR),
  Cited by: [§1](#S1.p1.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px4.p1.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* B. Kleinberg, Y. Li, and Y. Yuan (2018)
  An alternative view: when does sgd escape local minima?.
  In International conference on machine learning,
   pp. 2698–2707.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* D. Kovalev (2025)
  Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization.
  arXiv preprint arXiv:2503.12645.
  Cited by: [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Remark D.2](#A4.Thmremark2.p2.1.1 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Appendix D](#A4.p1.1 "Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px2.p3.2 "Assumptions in SCG methods: structured nonconvexity. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Remark 4.1](#S4.Thmremark1.p1.4.4 "Remark 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [footnote 1](#footnote1 "In Remark 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* T. Large, Y. Liu, M. Huh, H. Bahng, P. Isola, and J. Bernstein (2024)
  Scalable optimization in the modular norm.
  Advances in Neural Information Processing Systems 37,  pp. 73501–73548.
  Cited by: [Appendix A](#A1.p1.4 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* C. Liu, L. Zhu, and M. Belkin (2022)
  Loss landscapes and optimization in over-parameterized non-linear systems and neural networks.
  Applied and Computational Harmonic Analysis 59,  pp. 85–116.
  Cited by: [§3](#S3.p3.3 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* H. Liu and J. Tong (2024)
  New sample complexity bounds for sample average approximation in heavy-tailed stochastic programming.
  In Forty-first International Conference on Machine Learning,
  External Links: [Link](https://openreview.net/forum?id=2hWd4CVhXz)
  Cited by: [§C.1](#A3.SS1.p1.4 "C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Łojasiewicz (1963)
  A topological property of real analytic subsets.
  Coll. du CNRS, Les équations aux dérivées partielles 117 (87-89),  pp. 2.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p3.2 "Assumptions in SCG methods: structured nonconvexity. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§3](#S3.p3.3 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* I. Loshchilov and F. Hutter (2019)
  Decoupled weight decay regularization.
  In International Conference on Learning Representations (ICLR),
  Cited by: [§4](#S4.p6.2 "4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. McCandlish, J. Kaplan, D. Amodei, and O. D. Team (2018)
  An empirical model of large-batch training.
  arXiv preprint arXiv:1812.06162.
  Cited by: [§1](#S1.p10.1 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px4.p2.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* W. Merrill, S. Arora, D. Groeneveld, and H. Hajishirzi (2025)
  Critical batch size revisited: a simple empirical approach to large-batch language model training.
  In The Thirty-ninth Annual Conference on Neural Information Processing Systems,
  Cited by: [§1](#S1.p3.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* B. Mlodozeniec, P. Ablin, L. Béthune, D. Busbridge, M. Klein, J. Ramapuram, and M. Cuturi (2025)
  Completed hyperparameter transfer across modules, width, depth, batch and duration.
  arXiv preprint arXiv:2512.22382.
  Cited by: [§C.1](#A3.SS1.p1.18 "C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, et al. (2021)
  Efficient large-scale language model training on gpu clusters using megatron-lm.
  In Proceedings of the international conference for high performance computing, networking, storage and analysis,
   pp. 1–15.
  Cited by: [§4](#S4.p4.1 "4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* T. Pethick, W. Xie, K. Antonakopoulos, Z. Zhu, A. Silveti-Falls, and V. Cevher (2025a)
  Training deep learning models with norm-constrained LMOs.
  In Forty-second International Conference on Machine Learning,
  Cited by: [Appendix A](#A1.p1.4 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§1](#S1.p5.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Remark 4.1](#S4.Thmremark1.p1.4.4 "Remark 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§6.2](#S6.SS2.p1.4 "6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§6](#S6.p1.4 "6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [Limitations and Future Work](#Sx1.p1.1 "Limitations and Future Work ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* T. Pethick, W. Xie, M. Erdogan, K. Antonakopoulos, T. Silveti-Falls, and V. Cevher (2025b)
  Generalized gradient norm clipping & non-euclidean (L0,L1)({L}\_{0},{L}\_{1})-smoothness.
  arXiv preprint arXiv:2506.01913.
  External Links: [Link](https://arxiv.org/abs/2506.01913)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.2 "Assumptions in SCG methods: smoothness. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* B. T. Polyak (1963)
  Gradient methods for the minimisation of functionals.
  USSR Computational Mathematics and Mathematical Physics 3 (4),  pp. 864–878.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p3.2 "Assumptions in SCG methods: structured nonconvexity. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§3](#S3.p3.3 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Qiu, Z. Chen, H. Phan, Q. Lei, and A. G. Wilson (2025)
  Hyperparameter transfer enables consistent gains of matrix-preconditioned optimizers across scales.
  In Advances in Neural Information Processing Systems (NeurIPS) 2025,
  Cited by: [§4](#S4.p6.2 "4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* A. Riabinin, E. Shulgin, K. Gruntkowska, and P. Richtárik (2025)
  Gluon: making Muon & Scion great again! (Bridging theory and practice of LMO-based optimizers for LLMs).
  arXiv preprint arXiv:2505.13416.
  External Links: [Link](https://arxiv.org/abs/2505.13416)
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.2 "Assumptions in SCG methods: smoothness. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px2.p3.2 "Assumptions in SCG methods: structured nonconvexity. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§6.4](#S6.SS4.SSS0.Px1.p1.4 "Smoothness constant 𝐿. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* F. Schaipp, A. Hägele, A. Taylor, U. Simsekli, and F. Bach (2025)
  The surprising agreement between convex optimization theory and learning-rate scheduling for large model training.
  arXiv preprint arXiv:2501.18965.
  Cited by: [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan (2009)
  Stochastic convex optimization..
  In COLT,
  Vol. 2,  pp. 5.
  Cited by: [§C.1](#A3.SS1.p1.4 "C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, and G. E. Dahl (2019)
  Measuring the effects of data parallelism on neural network training.
  Journal of Machine Learning Research.
  Cited by: [§1](#S1.p1.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§1](#S1.p10.1 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* S. L. Smith, P. Kindermans, C. Ying, and Q. V. Le (2018)
  Don’t decay the learning rate, increase the batch size.
  In International Conference on Learning Representations (ICLR),
  Cited by: [§1](#S1.p1.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§1](#S1.p10.1 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px4.p1.1 "Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu (2021)
  RoFormer: enhanced transformer with rotary position embedding.
  arXiv preprint arXiv:2104.09864.
  Cited by: [Appendix A](#A1.p1.4 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* H. Tran, Q. Zhang, and A. Cutkosky (2024)
  Reevaluating theoretical analysis methods for optimization in deep learning.
  arXiv preprint arXiv:2407.01825.
  Cited by: [Remark D.2](#A4.Thmremark2.p1.5.5 "Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, et al. (2020)
  SciPy 1.0: fundamental algorithms for scientific computing in python.
  Nature Methods 17 (3),  pp. 261–272.
  Cited by: [§B.2.1](#A2.SS2.SSS1.p1.3 "B.2.1 Estimating the Smoothness Constant 𝐿 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* L. Xiao (2024)
  Rethinking conventional wisdom in machine learning: from generalization to scaling.
  arXiv preprint arXiv:2409.15156.
  Cited by: [§4](#S4.p6.2 "4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* G. Yang, E. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao (2021)
  Tuning large neural networks via zero-shot hyperparameter transfer.
  In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. W. Vaughan (Eds.),
  Cited by: [§1](#S1.p4.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px3.p1.1 "Works on Hyperparameter Transfer. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao (2022)
  Tensor programs v: tuning large neural networks via zero-shot hyperparameter transfer.
  arXiv preprint arXiv:2203.03466.
  Cited by: [§1](#S1.p4.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px3.p1.1 "Works on Hyperparameter Transfer. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* G. Yang and E. J. Hu (2020)
  Feature learning in infinite-width neural networks.
  arXiv preprint arXiv:2011.14522.
  Cited by: [§1](#S1.p4.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§2](#S2.SS0.SSS0.Px3.p1.1 "Works on Hyperparameter Transfer. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* G. Yang, D. Yu, C. Zhu, and S. Hayou (2023)
  Feature learning in infinite-depth neural networks.
  In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning,
  Cited by: [§2](#S2.SS0.SSS0.Px3.p1.1 "Works on Hyperparameter Transfer. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* Y. Yang, E. Tripp, Y. Sun, S. Zou, and Y. Zhou (2024)
  Adaptive gradient normalization and independent sampling for (stochastic) generalized-smooth optimization.
  arXiv preprint arXiv:2410.14054.
  Cited by: [§2](#S2.SS0.SSS0.Px2.p2.1 "Assumptions in SCG methods: structured nonconvexity. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* B. Zhang and R. Sennrich (2019)
  Root mean square layer normalization.
  Advances in neural information processing systems 32.
  Cited by: [Appendix A](#A1.p1.4 "Appendix A Description of the Experimental Setup ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"),
  [§6.4](#S6.SS4.SSS0.Px1.p1.4 "Smoothness constant 𝐿. ‣ 6.4 Estimating Problem-Dependent Constants ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, and S. M. Kakade (2025)
  How does critical batch size scale in pre-training?.
  In The Thirteenth International Conference on Learning Representations,
  Cited by: [§1](#S1.p3.2 "1 Introduction ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* J. Zhang, T. He, S. Sra, and A. Jadbabaie (2019)
  Why gradient clipping accelerates training: a theoretical justification for adaptivity.
  arXiv:1905.11881.
  Cited by: [§2](#S2.SS0.SSS0.Px1.p1.2 "Assumptions in SCG methods: smoothness. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").
* Y. Zhou, J. Yang, H. Zhang, Y. Liang, and V. Tarokh (2019)
  Sgd converges to global minimum in deep learning via star-convex path.
  arXiv preprint arXiv:1901.00451.
  Cited by: [§3](#S3.p4.13 "3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

Appendix

## Appendix A Description of the Experimental Setup

Our implementation uses Scaled ReLU2 from Large et al. [[2024](#bib.bib81 "Scalable optimization in the modular norm")] (see Appendix B.2), rotary embeddings [Su et al., [2021](#bib.bib80 "RoFormer: enhanced transformer with rotary position embedding")] in place of positional embeddings, RMSNorm [Zhang and Sennrich, [2019](#bib.bib53 "Root mean square layer normalization")] (without learnable parameters following Pethick et al. [[2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")]) instead of LayerNorm, and a linear learning rate decay schedule instead of cosine annealing. The choice of radius is taken from Pethick et al. [[2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs")]: η=50\eta=50 for matrix-type layers and η=3000\eta=3000 for the rest of the layers. To approximate the polar factor of the gradient, we use the Newton-Schulz method with 55 iterations, following [Jordan et al., [2024b](#bib.bib95 "Muon: an optimizer for hidden layers in neural networks")]. All other details are reported in [Table˜B.1](#A2.T1 "In B.1 Verification of ˜3.4 ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

Note that due to limited GPU availability, the 1B model is trained using checkpointing. This introduces slight fluctuations across runs. Although the random seed is fixed, some variability remains due to nondeterminism in the PyTorch implementation.

## Appendix B Empirical Verification of Assumptions

### B.1 Verification of [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")

In this section, we provide the evolution of the empirical variance throughout the training when varying the batch size and sequence length when training a base 124M model. When we vary the batch size, we keep the sequence length equal 10241024; when we vary the sequence length, we keep the batch size equal 512512. To approximate the full gradient, we sample a mini-batch gradient of size 3276832768. In [Figure˜B.1](#A2.F1 "In B.1 Verification of ˜3.4 ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we demonstrate that after a short initial phase (up to 1k iterations) the empirical variance stabilizes and fluctuates around the average, suggesting that the variance is fixed during most of the training.

Table B.1: The model configurations and training details used in [Section˜6](#S6 "6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | 124M Model | 775M Model | 1B Model |
| Layers | 12 | 36 | 18 |
| Heads | 6 | 20 | 16 |
| Embedding Size | 768 | 1280 | 2048 |
| Weight Tying | Yes | | |
| Activation Function | ReLU2 | | |
| Vocabulary Size | 50304 | | |
| Dataset | FineWeb | | |
| Warmdown | 28% of the total token budget | | |
| Stepsize Schedule | βk={γif ​k<n−mγ⋅n−kmotherwise\beta\_{k}=\begin{cases}\gamma&\text{if }k<n-m\\ \gamma\cdot\frac{n-k}{m}&\text{otherwise}\end{cases} | | |
| Gradient Clipping | No | | |
| Momentum Parameter | α=0.1\alpha=0.1 | | |
| lm\_head/embd Radii | 30003000 | | |
| Matrix Weights Radii | 5050 | | |
| Precision | bf16 | | |
| Device Batch Size | |  | | --- | | 32 | | if the opposite | | is not stated explicitly | |  | 16 |



|  |  |
| --- | --- |
| Refer to caption | Refer to caption |

Figure B.1: Evolution of the empirical gradient variance varying batch size BB with fixed sequence length S=1024S=1024 (left) and sequence length SS with fixed batch size B=512B=512 (right) when training a 124M NanoGPT model on the FineWeb dataset. We observe that the variance quickly stabilizes after a short initial phase.

### B.2 Verification of Assumptions [3.1](#S3.Thmassumption1 "Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")-[3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") when Varying Model Configuration

In this section, we provide the estimations of problem-dependent constants L,μ,ρL,\mu,\rho, varying n\_embd and n\_layer, while keeping n\_head=6. We measure the constants for several configurations with a goal to cover a broader number of configurations. Due to extensive requirements on memory and time resources, we do not provide the measurements for all possible configurations.

#### B.2.1 Estimating the Smoothness Constant LL

First, we measure the smoothness constant LL using the following estimation

|  |  |  |
| --- | --- | --- |
|  | ‖g​(xk;ξk)−g​(xk−1;ξk−1)‖∗‖xk−xk−1‖,\frac{\|g(x\_{k};\xi\_{k})-g(x\_{k-1};\xi\_{k-1})\|\_{\*}}{\|x\_{k}-x\_{k-1}\|}, |  |

where g​(xk;ξk),g​(xk−1;ξk−1)g(x\_{k};\xi\_{k}),g(x\_{k-1};\xi\_{k-1}) are mini-batch gradient at two consecutive iterations, while the norms are defined in ([14](#S6.E14 "Equation 14 ‣ 6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Based on the results in [Table˜B.2](#A2.T2 "In B.2.1 Estimating the Smoothness Constant 𝐿 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we fit a power law with shifts of the form

|  |  |  |
| --- | --- | --- |
|  | L​(n\_layer,n\_embd)=C​(n\_layer+a0)ν​(n\_embd+b0)γ.L(\texttt{n\\_layer},\texttt{n\\_embd})=C(\texttt{n\\_layer}+a\_{0})^{\nu}(\texttt{n\\_embd}+b\_{0})^{\gamma}. |  |

We fit the parameters of the power law in log-space, using least squares with soft\_l1 loss from scipy.optimize [Virtanen et al., [2020](#bib.bib83 "SciPy 1.0: fundamental algorithms for scientific computing in python")]. The fit provides the following approximations for the constants of the power law:

|  |  |  |
| --- | --- | --- |
|  | C=0.4,ν=0.2,γ=0.35,a0=0.7,b0=126.C=0.4,\quad\nu=0.2,\quad\gamma=0.35,\quad a\_{0}=0.7,\quad b\_{0}=126. |  |

Table B.2: Estimated LL constant for various model configurations.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n\_embd \\,\backslash\, n\_layer | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
| 384 | – | 5.3 | 6.2 | 6.5 | 6.2 | 18 | 7.6 | – | 27 | 7.84 |
| 576 | – | 6.4 | 7.7 | 7.0 | 7.4 | – | 7.5 | 8.5 | 7.9 | 8.2 |
| 768 | 6.1 | 6.5 | 6.9 | 7.6 | 8.3 | 18 | 9.9 | 8.5 | 8.7 | 10.0 |
| 1152 | – | – | – | 8.8 | 9.4 | 10.8 | 9.5 | – | – | – |
| 1536 | 3 | 7.9 | 9.2 | 12 | 9.9 | – | – | – | – | – |
| 2304 | – | 9.9 | – | – | – | – | – | – | – | – |

#### B.2.2 Estimating the Norm Equivalence Constant ρ\rho

Table B.3: Estimated ρ\rho constant for various model configurations.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n\_embd \\,\backslash\, n\_layer | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
| 384 | – | 35.5 | 41.3 | 48.2 | 48.7 | – | 50.7 | – | – | 58.0 |
| 576 | – | 42.1 | 53.8 | 61.1 | 62.9 | – | 64.2 | 64.6 | 66.2 | 68.6 |
| 768 | 31.2 | 52.6 | 64.1 | 67.1 | 72.4 | – | 80.2 | 87.0 | 86.3 | 89.2 |
| 1152 | – | – | – | 76.6 | 81.1 | 87.3 | — | – | – | – |
| 1536 | – | 67.5 | 77.4 | – | – | – | – | – | – | – |
| 2304 | – | – | – | – | – | – | – | – | – | – |

Now we measure the norm equivalence constant ρ\rho. We observed that the ρ\rho constant changes not only with the model size but also with the batch size and sequence length. To measure it, we run Scion with batch size 512512 and sequence length 10241024, and the Frank–Wolfe stepsize β=3.6⋅10−4\beta=3.6\cdot 10^{-4}. We estimate the ρ\rho constant as follows

|  |  |  |
| --- | --- | --- |
|  | ‖g​(xk;ξk)−G​(xk;Ξk)‖∗‖g​(xk;ξk)−G​(xk;Ξk)‖2,\frac{\|g(x\_{k};\xi\_{k})-G(x\_{k};\Xi\_{k})\|\_{\*}}{\|g(x\_{k};\xi\_{k})-G(x\_{k};\Xi\_{k})\|\_{2}}, |  |

where g​(xk;ξk)g(x\_{k};\xi\_{k}) is a mini-batch gradient of size 512512, while G​(xk;Ξk)G(x\_{k};\Xi\_{k}) is a mini-batch gradient of size 81928192, which serves as an approximation of the full gradient.

We also observe that the constant ρ\rho significantly changes with the batch size. Therefore, we measured how ρ\rho changes with batch size for a configuration with 66 layers and 768768 embedding dimension in [Table˜B.4](#A2.T4 "In B.2.2 Estimating the Norm Equivalence Constant 𝜌 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods").

Based on the results in [Table˜B.3](#A2.T3 "In B.2.2 Estimating the Norm Equivalence Constant 𝜌 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") and [Table˜B.4](#A2.T4 "In B.2.2 Estimating the Norm Equivalence Constant 𝜌 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we fit the parameters of the power law of the form

|  |  |  |
| --- | --- | --- |
|  | ρ​(n\_layer,n\_embd,batch\_size)=C​(n\_layer+a0)ν​(n\_embd+b0)γ​(batch\_size+c0)δ\rho(\texttt{n\\_layer},\texttt{n\\_embd},\texttt{batch\\_size})=C(\texttt{n\\_layer}+a\_{0})^{\nu}(\texttt{n\\_embd}+b\_{0})^{\gamma}(\texttt{batch\\_size}+c\_{0})^{\delta} |  |

in log-space, using least squares with soft\_l1 loss from scipy.optimize. The fit provides the following approximations for the constants of the power law:

|  |  |  |
| --- | --- | --- |
|  | C=4.1,a0=−2.7,ν=0.25,b0=−250.8,γ=0.3,c0=−9.4,δ=0.1.C=4.1,\quad a\_{0}=-2.7,\quad\nu=0.25,\quad b\_{0}=-250.8,\quad\gamma=0.3,\quad c\_{0}=-9.4,\quad\delta=0.1. |  |

Table B.4: Estimated ρ\rho constant for a configuration with 66 layers and 768768 embedding dimension when varying the batch size.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| batch\_size | 256 | 512 | 1024 | 2048 | 4096 |
| 𝝆\boldsymbol{\rho} | 48.9 | 52.6 | 55.0 | 56.4 | 57.9 |

#### B.2.3 Estimating Kurdyka–Łojasiewicz Constant 𝝁\boldsymbol{\mu}

Table B.5: Estimated Kurdyka–Łojasiewicz constant μ\mu for various model configurations.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n\_embd \\,\backslash\, n\_layer | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
| 384 | – | 3.4 | 3.1 | 3.1 | 3.0 | – | 2.9 | – | – | 2.7 |
| 576 | – | 3.3 | 3.2 | 3.0 | 2.9 | – | 2.7 | 2.6 | 2.5 | 2.4 |
| 768 | 3.7 | 3.2 | 3.0 | 2.9 | 2.8 | – | 2.6 | 2.5 | 2.3 | 2.4 |
| 1152 | – | – | – | 2.7 | 2.7 | 2.8 | 2.6 | – | 2.5 | – |
| 1536 | – | 3.2 | 2.9 | – | 2.9 | 3.0 | – | – | – | – |
| 2304 | – | 3.6 | – | – | – | – | – | – | – | – |

Finally, we measure the KL constant μ\mu by tracking the dual gradient norm and train loss (norms are defined in ([14](#S6.E14 "Equation 14 ‣ 6.2 Verification of Assumption 3.3 ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))). Then, we fit a linear regression with Huber loss, robust to outliers. The slope of the linear fit serves as an approximation of μ\mu constant. Based on the results in [Table˜B.5](#A2.T5 "In B.2.3 Estimating Kurdyka–Łojasiewicz Constant 𝝁 ‣ B.2 Verification of Assumptions 3.1-3.3 when Varying Model Configuration ‣ Appendix B Empirical Verification of Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we fit the parameters of the power law of the form

|  |  |  |
| --- | --- | --- |
|  | μ​(n\_layer,n\_embd)=C​(n\_layer+a0)ν​(n\_embd+b0)γ.\mu(\texttt{n\\_layer},\texttt{n\\_embd})=C(\texttt{n\\_layer}+a\_{0})^{\nu}(\texttt{n\\_embd}+b\_{0})^{\gamma}. |  |

in log-space, using least squares with soft\_l1 loss from scipy.optimize. The fit provides the following approximations for the constants of the power law:

|  |  |  |
| --- | --- | --- |
|  | C=5.2,ν=0.2,γ=0,a0=1.7,b0=−384.C=5.2,\quad\nu=0.2,\quad\gamma=0,\quad a\_{0}=1.7,\quad b\_{0}=-384. |  |

## Appendix C Additional Experiments

### C.1 Additional Baselines in Experiments from [Section˜6.5](#S6.SS5 "6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")

In this section, we add additional baselines to the setting from [Section˜6.5](#S6.SS5 "6.5 Increasing Batch Size and Sequence Length during Training ‣ 6 Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). The idea behind the two new baselines is the following. The literature on the learning theory suggests that the excess risk should decay as ∼1T\sim\frac{1}{\sqrt{T}} under standard convexity [Shalev-Shwartz et al., [2009](#bib.bib124 "Stochastic convex optimization."), Liu and Tong, [2024](#bib.bib131 "New sample complexity bounds for sample average approximation in heavy-tailed stochastic programming")], which is the closest setting to μ\mu-KL case due to the relation between μ\mu-KL condition and ζ\zeta-quasar convexity, described after [˜3.3](#S3.Thmassumption3 "Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"). This hypothesizes that we need to keep the optimization error close to the excess risk. In particular, if we find that for a small model the dominating term in ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) is the first one, then we control it as follows

|  |  |  |
| --- | --- | --- |
|  | L​B0​S0μ2​T0∼ε0T0.\frac{LB\_{0}S\_{0}}{\mu^{2}T\_{0}}\sim\frac{\varepsilon\_{0}}{\sqrt{T\_{0}}}. |  |

We want to choose parameters B1B\_{1} and S1S\_{1} such that the same approximation holds for a larger model. This gives another recipe on how to increase the batch size and sequence length:

|  |  |  |  |
| --- | --- | --- | --- |
|  | B0​S0/T0B1​S1/T1=1/T01/T1⇒B1​S1=B0​S0​T1T0.\displaystyle\frac{B\_{0}S\_{0}/T\_{0}}{B\_{1}S\_{1}/T\_{1}}=\frac{1/\sqrt{T\_{0}}}{1/\sqrt{T\_{1}}}\Rightarrow B\_{1}S\_{1}=B\_{0}S\_{0}\frac{\sqrt{T\_{1}}}{\sqrt{T\_{0}}}. |  | (16) |

From ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), we obtain that we should the Frank–Wolfe stepsize of the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | β1=β0​B1​S1B0​S0​T0T1=T0T1.\beta\_{1}=\beta\_{0}\frac{B\_{1}S\_{1}}{B\_{0}S\_{0}}\frac{T\_{0}}{T\_{1}}=\frac{\sqrt{T\_{0}}}{\sqrt{T\_{1}}}. |  | (17) |

For a 1B model, this means that we should increase the product B​SBS by a factor 8\sqrt{8}, while decreasing the Frank–Wolfe stepsize by a factor 1/81/\sqrt{8}. Performing all derivations, this gives the values of batch size 736736, sequence length 1024,1024, and the Frank–Wolfe stepsize 1.27⋅10−41.27\cdot 10^{-4}. In [Figure˜C.1](#A3.F1 "In C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we add two more baselines: one when we do a restart with the parameters 736736, 10241024, 1.27⋅10−41.27\cdot 10^{-4} after a 1.3B token budget, and another one where we use the parameters 736736, 10241024, 1.27⋅10−41.27\cdot 10^{-4} from the beginning. These ideas are closely aligned with prior work Compagnoni et al. [[2025b](#bib.bib130 "Adaptive methods through the lens of SDEs: theoretical insights on the role of noise"), [a](#bib.bib128 "Unbiased and sign compression in distributed learning: comparing noise resilience via sdes")], Mlodozeniec et al. [[2025](#bib.bib23 "Completed hyperparameter transfer across modules, width, depth, batch and duration")], which also propose to rescale the weight decay (equivalent of our Frank–Wolfe stepsize) following the square-root rule.

We observe that the new baselines are also competitive in practice. However, more aggressive BST scaling rules in ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([8](#S5.E8 "Equation 8 ‣ 5.2 Tuning the Frank–Wolfe Stepsize ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) provide slightly better results: the restarted version of BST baseline achieves the best validation loss, while a fixed batch BST baseline slightly outperforms square-root fixed batch baseline. This further supports that our theory-inspired BST scaling rule is predictive and can be used in practice. The notation Sqrt or BST indicates the rule used to select the Frank–Wolfe stepsize.

|  |
| --- |
| Refer to caption |

Figure C.1: Comparison of two strategies: BST scaling rule, where B​S∼T2/3BS\sim T^{2/3} (fixed 10241024 batch size with β1=1.8⋅10−4\beta\_{1}=1.8\cdot 10^{-4} in light green and restarted version in gray) and square-root rule ([16](#A3.E16 "Equation 16 ‣ C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), where B​S∼T1/2BS\sim T^{1/2} (fixed 736736 batch size with β2=1.27⋅10−4\beta\_{2}=1.27\cdot 10^{-4} in orange and restarted version in pink). The validation and train sequence lengths are fixed to 10241024. ( large batch size strategies when training a 1B model. Scion with a batch size of 10241024 (fixed from the beginning or after a restart) achieves slightly better performance compared to the baselines, where the batch size is set according to the square-root rule ([16](#A3.E16 "Equation 16 ‣ C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). The notation (B1,2,S,β0,1)(B\_{1,2},S,\beta\_{0,1}) characterizes which batch size, sequence length, and Frank–Wolfe stepsize are used for the particular setup, respectively. The notation (B0,S,β0)→(B1,2,S,β1,2)(B\_{0},S,\beta\_{0})\to(B\_{1,2},S,\beta\_{1,2}) characterizes how parameters of Scion change after restart (e.g., batch size increases from B0B\_{0} to B1,2B\_{1,2}), respectively.

### C.2 Effect of the Device Batch Size

In this section, we evaluate how the device batch size affects the final performance of the 124M model. Note that using a smaller device batch size within a fixed global batch size results in more gradient accumulation steps. In [Figure˜C.2](#A3.F2 "In C.2 Effect of the Device Batch Size ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we report the model’s final validation loss as we vary the device batch size and the momentum parameter. We observe that for device batch sizes 88 and 3232, the performance is closely aligned. The quantization errors become slightly visible for a device batch size of 22 when using extreme values of momentum far from the optimal value. However, around the optimal momentum parameter, the difference is within one standard deviation.

|  |
| --- |
| Refer to caption |

Figure C.2: Final performance of 124M model with B=256,S=1024,β=3.6⋅10−4B=256,S=1024,\beta=3.6\cdot 10^{-4} and varying the momentum parameter α\alpha and device batch size.

## Appendix D In-Expectation Convergence Proofs for SCG

The proof structure is inspired by the analysis of the first-order stochastic trust-region method with momentum under star-convexity by Kovalev [[2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization")].

###### Lemma 1.

Let assumptions ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) hold. Assume that x0x\_{0} and η\eta are chosen such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​‖x0‖≤η,β=cK,andK≥2​c.2\|x\_{0}\|\leq\eta,\quad\beta=\frac{c}{K},\quad\text{and}\quad K\geq 2c. |  | (18) |

Let {xk}\{x\_{k}\} be the iterates of [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") . Then, the following inequalities hold for all k∈{0,1​…,K−1}k\in\{0,1\ldots,K-1\}

|  |  |  |  |
| --- | --- | --- | --- |
|  | η−‖xk‖≥(1−β)k​η2,‖xk+1−xk‖≤2​β​η.\displaystyle\eta-\|x\_{k}\|\geq(1-\beta)^{k}\frac{\eta}{2},\quad\|x\_{k+1}-x\_{k}\|\leq 2\beta\eta. |  | (19) |

###### Proof.

We show by induction kk that

|  |  |  |
| --- | --- | --- |
|  | ‖xk‖≤(1−β)k​η2+η​(1−(1−β)k)\|x\_{k}\|\leq(1-\beta)^{k}\frac{\eta}{2}+\eta(1-(1-\beta)^{k}) |  |

holds. The base of induction is k=0k=0. Note that ‖x0‖≤η2\|x\_{0}\|\leq\frac{\eta}{2} holds by the choice of η\eta and x0x\_{0} in ([18](#A4.E18 "Equation 18 ‣ Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Assume that inequalities hold for some k∈{0,1,…​K−2}k\in\{0,1,\ldots K-2\}. We show that they also hold at iteration k+1k+1. Indeed, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖xk+1‖\displaystyle\|x\_{k+1}\| | ≤\Hy@raisedlink([a](#desca2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​‖(1−β)​xk+β​η​dk+1‖​≤\Hy@raisedlink([b](#descb2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−β)​‖xk‖+β​η​‖dk+1‖\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a2}{}}{(\hyperlink{desca2}{a})}}}{\leq}\|(1-\beta)x\_{k}+\beta\eta d\_{k+1}\|\overset{\text{\Hy@raisedlink{\hypertarget{b2}{}}{(\hyperlink{descb2}{b})}}}{\leq}(1-\beta)\|x\_{k}\|+\beta\eta\|d\_{k+1}\| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([c](#descc2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−β)​((1−β)k​η2+η​(1−(1−β)k))+β​η\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{c2}{}}{(\hyperlink{descc2}{c})}}}{\leq}(1-\beta)\left((1-\beta)^{k}\frac{\eta}{2}+\eta(1-(1-\beta)^{k})\right)+\beta\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−β)k+1​η2+η​((1−β)−(1−β)k+1+β)=(1−β)k+1​η2+η​(1−(1−β)k+1),\displaystyle=(1-\beta)^{k+1}\frac{\eta}{2}+\eta((1-\beta)-(1-\beta)^{k+1}+\beta)=(1-\beta)^{k+1}\frac{\eta}{2}+\eta(1-(1-\beta)^{k+1}), |  |

where \Hy@raisedlink([a](#a2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the update step; \Hy@raisedlink([b](#b2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the triangle inequality; \Hy@raisedlink([c](#c2 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the restriction on dk+1d\_{k+1} in and induction hypothesis. This concludes the induction step and proves the first inequality in ([19](#A4.E19 "Equation 19 ‣ Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) for all k∈{0,1,…,K}.k\in\{0,1,\ldots,K\}. We can lower bound (1−β)K(1-\beta)^{K} using the inequality log⁡(1−y)≥−y−y2\log(1-y)\geq-y-y^{2} for all y∈[0,0.5]y\in[0,0.5] and K≥2​cK\geq 2c as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | log⁡((1−β)K)=K​log⁡(1−β)≥K​(−β−β2)=−c−c2K≥−3​c2.\log((1-\beta)^{K})=K\log(1-\beta)\geq K(-\beta-\beta^{2})=-c-\frac{c^{2}}{K}\geq-\frac{3c}{2}. |  | (20) |

This implies that (1−β)K≥e−3​c/2(1-\beta)^{K}\geq e^{-\nicefrac{{3c}}{{2}}}. Using the obtained bound, we derive for all k∈{0,1,…,K}k\in\{0,1,\ldots,K\}

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ‖xk‖\displaystyle\|x\_{k}\| | ≤(1−β)K​η2+η​(1−(1−β)K)≤η−η2​e−3​c/2.\displaystyle\leq(1-\beta)^{K}\frac{\eta}{2}+\eta(1-(1-\beta)^{K})\leq\eta-\frac{\eta}{2}e^{-\nicefrac{{3c}}{{2}}}. |  | (21) |

Now we prove the last inequality in ([19](#A4.E19 "Equation 19 ‣ Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). We have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖xk+1−xk‖\displaystyle\|x\_{k+1}-x\_{k}\| | =\Hy@raisedlink([a](#desca3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​‖−β​xk+β​η​dk+1‖\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a3}{}}{(\hyperlink{desca3}{a})}}}{=}\|-\beta x\_{k}+\beta\eta d\_{k+1}\| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([b](#descb3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​β​‖xk‖+β​η​‖dk+1‖\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{b3}{}}{(\hyperlink{descb3}{b})}}}{\leq}\beta\|x\_{k}\|+\beta\eta\|d\_{k+1}\| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([c](#descc3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​β​((1−β)K​η2+η​(1−(1−β)K))+β​η\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{c3}{}}{(\hyperlink{descc3}{c})}}}{\leq}\beta((1-\beta)^{K}\frac{\eta}{2}+\eta(1-(1-\beta)^{K}))+\beta\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​η​((1−β)K/2+1−(1−β)K+1)\displaystyle=\beta\eta((1-\beta)^{K}/2+1-(1-\beta)^{K}+1) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β​η​(2−(1−β)K/2)≤2​β​η,\displaystyle=\beta\eta(2-(1-\beta)^{K}/2)\leq 2\beta\eta, |  |

where \Hy@raisedlink([a](#a3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the update rule; \Hy@raisedlink([b](#b3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the triangle inequality; \Hy@raisedlink([c](#c3 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the previous inequality and the restriction on dk+1d\_{k+1}.
∎

###### Lemma 2.

Let Assumptions ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([A2](#S3.Ex2 "Equation A2 ‣ Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([3.4](#S3.Ex5 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) hold. Let m0=g​(x0;ξ0)m\_{0}=g(x\_{0};\xi\_{0}), then the iterates of [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") satisfy the following inequality:

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖mk+1−∇f​(xk)‖∗]≤(1−α)k​ρ​σ+2​L​β​ηα+ρ​σ​α.\mathbb{E}[\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*}]\leq(1-\alpha)^{k}\rho\sigma+\frac{2L\beta\eta}{\alpha}+\rho\sigma\sqrt{\alpha}. |  |

###### Proof.

We can express mk+1−∇f​(xk)m\_{k+1}-\nabla f(x\_{k}) as follows using the definition of the momentum buffer in [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")

|  |  |  |  |
| --- | --- | --- | --- |
|  | mk+1−∇f​(xk)\displaystyle m\_{k+1}-\nabla f(x\_{k}) | =(1−α)​mk+α​g​(xk;ξk)−∇f​(xk)\displaystyle=(1-\alpha)m\_{k}+\alpha g(x\_{k};\xi\_{k})-\nabla f(x\_{k}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(1−α)​(mk−∇f​(xk−1))+α​(g​(xk;ξk)−∇f​(xk))\displaystyle=(1-\alpha)(m\_{k}-\nabla f(x\_{k-1}))+\alpha(g(x\_{k};\xi\_{k})-\nabla f(x\_{k})) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +(1−α)​(∇f​(xk−1)−∇f​(xk)).\displaystyle+(1-\alpha)(\nabla f(x\_{k-1})-\nabla f(x\_{k})). |  |

This implies the following for all k≥0k\geq 0:

|  |  |  |  |
| --- | --- | --- | --- |
|  | mk+1−∇f​(xk)\displaystyle m\_{k+1}-\nabla f(x\_{k}) | =(1−α)k​(m1−∇f​(x0))+∑i=0k−1(1−α)k−i​(∇f​(xi)−∇f​(xi+1))\displaystyle=(1-\alpha)^{k}(m\_{1}-\nabla f(x\_{0}))+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}(\nabla f(x\_{i})-\nabla f(x\_{i+1})) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi)).\displaystyle+\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i})). |  |

Using this decomposition, we can upper-bound ‖mk+1−∇f​(xk)‖∗\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*} as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖mk+1−∇f​(xk)‖∗\displaystyle\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*} | ≤\Hy@raisedlink([a](#desca4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−α)k​‖m1−∇f​(x0)‖∗+∑i=0k−1(1−α)k−i​‖∇f​(xi)−∇f​(xi+1)‖∗\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a4}{}}{(\hyperlink{desca4}{a})}}}{\leq}(1-\alpha)^{k}\|m\_{1}-\nabla f(x\_{0})\|\_{\*}+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}\|\nabla f(x\_{i})-\nabla f(x\_{i+1})\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +‖∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi))‖∗\displaystyle\quad+\|\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i}))\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([b](#descb4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−α)k​‖m1−∇f​(x0)‖∗\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{b4}{}}{(\hyperlink{descb4}{b})}}}{\leq}(1-\alpha)^{k}\|m\_{1}-\nabla f(x\_{0})\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=0k−1(1−α)k−i​2​L​β​η+‖∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi))‖∗\displaystyle\quad+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}2L\beta\eta+\|\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i}))\|\_{\*} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([c](#descc4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−α)k​ρ​‖m1−∇f​(x0)‖2+∑i=0k−1(1−α)k−i​2​L​β​η\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{c4}{}}{(\hyperlink{descc4}{c})}}}{\leq}(1-\alpha)^{k}\rho\|m\_{1}-\nabla f(x\_{0})\|\_{2}+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}2L\beta\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +ρ​‖∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi))‖2,\displaystyle\quad+\rho\|\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i}))\|\_{2}, |  |

where \Hy@raisedlink([a](#a4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the triangle inequality; \Hy@raisedlink([b](#b4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and [Lemma˜1](#Thmlemma1 "Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") with L,β,ηL,\beta,\eta; \Hy@raisedlink([c](#c4 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([A2](#S3.Ex2 "Equation A2 ‣ Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Next, we take the full expectation and get

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[‖mk+1−∇f​(xk)‖∗]\displaystyle\mathbb{E}\left[\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*}\right] | ≤(1−α)k​ρ​𝔼​[‖m1−∇f​(x0)‖2]+∑i=0k−1(1−α)k−i​2​L​β​η\displaystyle\leq(1-\alpha)^{k}\rho\mathbb{E}[\|m\_{1}-\nabla f(x\_{0})\|\_{2}]+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}2L\beta\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +ρ​𝔼​[‖∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi))‖2]\displaystyle\quad+\rho\mathbb{E}\left[\|\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i}))\|\_{2}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([a](#desca5 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−α)k​ρ​𝔼​[‖m1−∇f​(x0)‖22]+∑i=0k−1(1−α)k−i​2​L​β​η\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a5}{}}{(\hyperlink{desca5}{a})}}}{\leq}(1-\alpha)^{k}\rho\sqrt{\mathbb{E}[\|m\_{1}-\nabla f(x\_{0})\|\_{2}^{2}]}+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}2L\beta\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +ρ​𝔼​[‖∑i=1kα​(1−α)k−i​(g​(xi,ξi)−∇f​(xi))‖22]\displaystyle\quad+\rho\sqrt{\mathbb{E}\left[\|\sum\_{i=1}^{k}\alpha(1-\alpha)^{k-i}(g(x\_{i},\xi\_{i})-\nabla f(x\_{i}))\|\_{2}^{2}\right]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([b](#descb5 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​(1−α)k​ρ​σ+∑i=0k−1(1−α)k−i​2​L​β​η+α​ρ​σ​∑i=1k(1−α)2​(k−i)\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{b5}{}}{(\hyperlink{descb5}{b})}}}{\leq}(1-\alpha)^{k}\rho\sigma+\sum\_{i=0}^{k-1}(1-\alpha)^{k-i}2L\beta\eta+\alpha\rho\sigma\sqrt{\sum\limits\_{i=1}^{k}(1-\alpha)^{2(k-i)}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤(1−α)k​ρ​σ+2​L​β​ηα+α​ρ​σ,\displaystyle\leq(1-\alpha)^{k}\rho\sigma+\frac{2L\beta\eta}{\alpha}+\sqrt{\alpha}\rho\sigma, |  |

where \Hy@raisedlink([a](#a5 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses Jensen’s inequality; \Hy@raisedlink([b](#b5 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([3.4](#S3.Ex5 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and the fact that samples ξi∼𝒟\xi\_{i}\sim\mathcal{D} are independent.
∎

###### Theorem D.1 (Full statement of [Theorem˜4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

Let Assumption ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([A2](#S3.Ex2 "Equation A2 ‣ Assumption 3.2. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([3.4](#S3.Ex5 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) hold. Let m0=g​(x0;ξ0)m\_{0}=g(x\_{0};\xi\_{0})and c>0c>0. Let the parameters of [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") are chosen as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | β=cK,η=2​e3​c/2μ​c​log⁡(2​(f​(x0)−f⋆)ε),2​‖x0‖≤η,\displaystyle\beta=\frac{c}{K},\quad\eta=\frac{2e^{3c/2}}{\mu c}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right),\quad 2\|x\_{0}\|\leq\eta, |  | (22) |

and

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | α\displaystyle\alpha | =min⁡{1,(ε​μ)2(32​ρ​σ)2​e3​c},\displaystyle=\min\left\{1,\frac{(\varepsilon\mu)^{2}}{(32\rho\sigma)^{2}e^{3c}}\right\}, |  | (23) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | K\displaystyle K | =max⁡[2​c,max⁡{12,128​L​e3​cε​μ2,32​ρ​σ​e3​c/2ε​μ,128​L​e6​c​(32​ρ​σ)2μ​(ε​μ)3,(32​ρ​σ​e3​c/2)3(ε​μ)3}​log⁡(2​(f​(x0)−f⋆)ε)].\displaystyle=\max\left[2c,\max\left\{\frac{1}{2},\frac{128Le^{3c}}{\varepsilon\mu^{2}},\frac{32\rho\sigma e^{\nicefrac{{3c}}{{2}}}}{\varepsilon\mu},\frac{128Le^{6c}(32\rho\sigma)^{2}}{\mu(\varepsilon\mu)^{3}},\frac{(32\rho\sigma e^{\nicefrac{{3c}}{{2}}})^{3}}{(\varepsilon\mu)^{3}}\right\}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right)\right]. |  |

Then the output of [Algorithm˜1](#alg1 "In Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") after KK iterations satisfies 𝔼​[f​(xK)−f⋆]≤ε\mathbb{E}[f(x\_{K})-f^{\star}]\leq\varepsilon.

###### Remark D.1.

The choice of η∼log⁡(2​(f​(x0)−f⋆)ε)\eta\sim\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right) and 2​‖x0‖≤η2\|x\_{0}\|\leq\eta ensures a sufficient contraction factor in front of f​(xk)−f⋆f(x\_{k})-f^{\star} in the proof. We also note that all iterates produced by Algorithm [1](#alg1 "Algorithm 1 ‣ Batch Size Scheduling. ‣ 2 Related Works ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") have a bounded norm by η\eta. However, we do not make any explicit assumptions about arg⁡minx∈𝒳⁡f​(x)\arg\min\_{x\in\mathcal{X}}f(x), e.g., we do not assume its existence or boundedness of its norm by η\eta. Therefore, for a fixed choice of ε\varepsilon it is possible that an optimizer has norm larger than η\eta while 𝔼​[f​(xK)−f⋆]≤ε\mathbb{E}[f(x\_{K})-f^{\star}]\leq\varepsilon.

###### Proof.

Let uk=arg​minu∈𝒳⁡⟨∇f​(xk),u⟩u\_{k}={\rm arg}\min\_{u\in\mathcal{X}}\langle\nabla f(x\_{k}),u\rangle s.t. ‖u‖≤1\|u\|\leq 1. Then we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(xk+1)\displaystyle f(x\_{k+1}) | ≤\Hy@raisedlink([a](#desca6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)+⟨∇f​(xk),xk+1−xk⟩+12​L​‖xk+1−xk‖2\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a6}{}}{(\hyperlink{desca6}{a})}}}{\leq}f(x\_{k})+\langle\nabla f(x\_{k}),x\_{k+1}-x\_{k}\rangle+\frac{1}{2}L\|x\_{k+1}-x\_{k}\|^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =\Hy@raisedlink([b](#descb6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)+⟨∇f​(xk),−β​xk+β​η​dk+1⟩+2​L​β2​η2\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{b6}{}}{(\hyperlink{descb6}{b})}}}{=}f(x\_{k})+\langle\nabla f(x\_{k}),-\beta x\_{k}+\beta\eta d\_{k+1}\rangle+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(xk)−β​⟨∇f​(xk),xk⟩+β​η​⟨∇f​(xk)−mk+1,dk+1⟩+β​η​⟨mk+1,dk+1⟩+2​L​β2​η2\displaystyle=f(x\_{k})-\beta\langle\nabla f(x\_{k}),x\_{k}\rangle+\beta\eta\langle\nabla f(x\_{k})-m\_{k+1},d\_{k+1}\rangle+\beta\eta\langle m\_{k+1},d\_{k+1}\rangle+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([c](#descc6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)−β​⟨∇f​(xk),xk⟩+β​η​⟨∇f​(xk)−mk+1,dk+1⟩+β​η​⟨mk+1,uk⟩+2​L​β2​η2\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{c6}{}}{(\hyperlink{descc6}{c})}}}{\leq}f(x\_{k})-\beta\langle\nabla f(x\_{k}),x\_{k}\rangle+\beta\eta\langle\nabla f(x\_{k})-m\_{k+1},d\_{k+1}\rangle+\beta\eta\langle m\_{k+1},u\_{k}\rangle+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =\Hy@raisedlink([d](#descd6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)−β​⟨∇f​(xk),xk⟩+β​η​⟨∇f​(xk)−mk+1,dk+1−uk⟩−β​η​‖∇f​(xk)‖∗+2​L​β2​η2\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{d6}{}}{(\hyperlink{descd6}{d})}}}{=}f(x\_{k})-\beta\langle\nabla f(x\_{k}),x\_{k}\rangle+\beta\eta\langle\nabla f(x\_{k})-m\_{k+1},d\_{k+1}-u\_{k}\rangle-\beta\eta\|\nabla f(x\_{k})\|\_{\*}+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([e](#desce6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)+β​‖∇f​(xk)‖∗⋅‖xk‖+2​β​η​‖∇f​(xk)−mk+1‖∗−β​η​‖∇f​(xk)‖∗+2​L​β2​η2\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{e6}{}}{(\hyperlink{desce6}{e})}}}{\leq}f(x\_{k})+\beta\|\nabla f(x\_{k})\|\_{\*}\cdot\|x\_{k}\|+2\beta\eta\|\nabla f(x\_{k})-m\_{k+1}\|\_{\*}-\beta\eta\|\nabla f(x\_{k})\|\_{\*}+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =f​(xk)−β​‖∇f​(xk)‖∗​(η−‖xk‖)+2​β​η​‖mk+1−∇f​(xk)‖∗+2​L​β2​η2\displaystyle=f(x\_{k})-\beta\|\nabla f(x\_{k})\|\_{\*}(\eta-\|x\_{k}\|)+2\beta\eta\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*}+2L\beta^{2}\eta^{2} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≤\Hy@raisedlink([f](#descf6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​f​(xk)−β​η​μ2​e−3​c/2​(f​(xk)−f⋆)+2​β​η​‖mk+1−∇f​(xk)‖∗+2​L​β2​η2,\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{f6}{}}{(\hyperlink{descf6}{f})}}}{\leq}f(x\_{k})-\frac{\beta\eta\mu}{2}e^{-\nicefrac{{3c}}{{2}}}(f(x\_{k})-f^{\star})+2\beta\eta\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*}+2L\beta^{2}\eta^{2}, |  | (24) |

where \Hy@raisedlink([a](#a6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([A1](#S3.Ex1 "Equation A1 ‣ Assumption 3.1. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")); \Hy@raisedlink([b](#b6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the update step and [Lemma˜1](#Thmlemma1 "Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"); \Hy@raisedlink([c](#c6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the optimality of dk+1d\_{k+1}; \Hy@raisedlink([d](#d6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ⟨∇f​(xk),uk⟩=−‖∇f​(xk)‖∗\langle\nabla f(x\_{k}),u\_{k}\rangle=-\|\nabla f(x\_{k})\|\_{\*}; \Hy@raisedlink([e](#e6 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses Cauchy-Schwarz and ‖dk+1‖,‖uk‖≤1\|d\_{k+1}\|,\|u\_{k}\|\leq 1; \Hy@raisedlink([f](#f6 "\Hy@raisedlink ‣ Equation 24 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses [Lemma˜1](#Thmlemma1 "Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), ([A3](#S3.Ex3 "Equation A3 ‣ Assumption 3.3. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), and ([21](#A4.E21 "Equation 21 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). With the assumption that m0=g​(x0;ξ0)m\_{0}=g(x\_{0};\xi\_{0}), we have from [Lemma˜2](#Thmlemma2 "Lemma 2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") that

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[‖mk+1−∇f​(xk)‖∗]≤(1−α)k​ρ​σ+2​L​β​ηα+ρ​σ​α.\mathbb{E}[\|m\_{k+1}-\nabla f(x\_{k})\|\_{\*}]\leq(1-\alpha)^{k}\rho\sigma+\frac{2L\beta\eta}{\alpha}+\rho\sigma\sqrt{\alpha}. |  |

Taking the expectation from ([24](#A4.E24 "Equation 24 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and using this bound and [Lemma˜1](#Thmlemma1 "Lemma 1. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we derive

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[f​(xk+1)−f⋆]\displaystyle\mathbb{E}[f(x\_{k+1})-f^{\star}] | ≤(1−μ​β​η2​e3​c/2)​𝔼​[f​(xk)−f⋆]+(1−α)k​2​β​η​ρ​σ+4​L​β2​η2α+2​β​η​ρ​σ​α\displaystyle\leq\left(1-\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}\right)\mathbb{E}[f(x\_{k})-f^{\star}]+(1-\alpha)^{k}2\beta\eta\rho\sigma+\frac{4L\beta^{2}\eta^{2}}{\alpha}+2\beta\eta\rho\sigma\sqrt{\alpha} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +2​L​β2​η2.\displaystyle\hskip 284.52756pt+2L\beta^{2}\eta^{2}. |  | (25) |

The contraction factor 1−μ​β​η2​e3​c/2∈(0,1)1-\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}\in(0,1) by the choice of K≥12​log⁡(2​(f​(x0)−f∗)ε)K\geq\frac{1}{2}\log\left(\frac{2(f(x\_{0})-f^{\*})}{\varepsilon}\right). Unrolling this recursion for all iterations k∈{0,1,…,K−1}k\in\{0,1,\ldots,K-1\} and using the bound for the geometric series, we guarantee progress such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[f​(xK)−f​(x⋆)]\displaystyle\mathbb{E}[f(x\_{K})-f(x^{\star})] | ≤(1−μ​β​η2​e3​c/2)K​(f​(x0)−f​(x⋆))+2​β​η​ρ​σα+4​ρ​σ​αμ​e3​c/2\displaystyle\leq\left(1-\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}\right)^{K}(f(x\_{0})-f(x^{\star}))+\frac{2\beta\eta\rho\sigma}{\alpha}+\frac{4\rho\sigma\sqrt{\alpha}}{\mu}e^{\nicefrac{{3c}}{{2}}} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +4​L​β​ημ​e3​c/2+8​L​β​ηα​μ​e3​c/2.\displaystyle\hskip 227.62204pt+\frac{4L\beta\eta}{\mu}e^{\nicefrac{{3c}}{{2}}}+\frac{8L\beta\eta}{\alpha\mu}e^{\nicefrac{{3c}}{{2}}}. |  | (26) |

Now we need to bound each of the terms proportionally to ε\varepsilon using the choice of parameters η,α,β,K\eta,\alpha,\beta,K from ([22](#A4.E22 "Equation 22 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and ([23](#A4.E23 "Equation 23 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). First, we want

|  |  |  |
| --- | --- | --- |
|  | 4​ρ​σ​αμ​e3​c/2≤ε8⇒α≤(ε​μ)2(32​ρ​σ)2​e3​c.\displaystyle 4\rho\sigma\frac{\sqrt{\alpha}}{\mu}e^{\nicefrac{{3c}}{{2}}}\leq\frac{\varepsilon}{8}\Rightarrow\alpha\leq\frac{(\varepsilon\mu)^{2}}{(32\rho\sigma)^{2}e^{3c}}. |  |

We can satisfy the above bound with the choice of α\alpha such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | α=min⁡{1,(ε​μ)2(32​ρ​σ)2​e3​c},\displaystyle\alpha=\min\left\{1,\frac{(\varepsilon\mu)^{2}}{(32\rho\sigma)^{2}e^{3c}}\right\}, |  | (27) |

which is exactly the choice of α\alpha in ([23](#A4.E23 "Equation 23 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Next, we want

|  |  |  |
| --- | --- | --- |
|  | 8​L​e3​c/2μ​β​ηα≤ε8⇒β=cK≤ε​μ​α64​L​η​e3​c/2​≤\Hy@raisedlink([a](#desca7 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​min⁡{ε​μ64​L​η​e3​c/2,(ε​μ)364​L​η​e9​c/2​(32​ρ​σ)2},\displaystyle\frac{8Le^{\nicefrac{{3c}}{{2}}}}{\mu}\frac{\beta\eta}{\alpha}\leq\frac{\varepsilon}{8}\Rightarrow\beta=\frac{c}{K}\leq\frac{\varepsilon\mu\alpha}{64L\eta e^{\nicefrac{{3c}}{{2}}}}\overset{\text{\Hy@raisedlink{\hypertarget{a7}{}}{(\hyperlink{desca7}{a})}}}{\leq}\min\left\{\frac{\varepsilon\mu}{64L\eta e^{\nicefrac{{3c}}{{2}}}},\frac{(\varepsilon\mu)^{3}}{64L\eta e^{\nicefrac{{9c}}{{2}}}(32\rho\sigma)^{2}}\right\}, |  |

where \Hy@raisedlink([a](#a7 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([27](#A4.E27 "Equation 27 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). The above can be satisfied if we choose KK such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | K\displaystyle K | ≥max⁡{64​L​η​c​e3​c/2ε​μ,64​L​η​c​e9​c/2​(32​ρ​σ)2(ε​μ)3},\displaystyle\geq\max\left\{\frac{64L\eta ce^{\nicefrac{{3c}}{{2}}}}{\varepsilon\mu},\frac{64L\eta ce^{\nicefrac{{9c}}{{2}}}(32\rho\sigma)^{2}}{(\varepsilon\mu)^{3}}\right\}, |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =\Hy@raisedlink([a](#desca8 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​max⁡{128​L​e3​cε​μ2,128​L​e6​c​(32​ρ​σ)2μ​(ε​μ)3}⋅log⁡(2​(f​(x0)−f⋆)ε),\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a8}{}}{(\hyperlink{desca8}{a})}}}{=}\max\left\{\frac{128Le^{3c}}{\varepsilon\mu^{2}},\frac{128Le^{6c}(32\rho\sigma)^{2}}{\mu(\varepsilon\mu)^{3}}\right\}\cdot\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right), |  | (28) |

where \Hy@raisedlink([a](#a8 "\Hy@raisedlink ‣ Equation 28 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the value of η\eta. This choice of KK is satisfied by the choice in ([23](#A4.E23 "Equation 23 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Moving on, we want

|  |  |  |  |
| --- | --- | --- | --- |
|  | 4​L​β​η​e3​c/2μ≤ε8⇒β=cK≤ε​μ32​L​η​e3​c/2.\frac{4L\beta\eta e^{\nicefrac{{3c}}{{2}}}}{\mu}\leq\frac{\varepsilon}{8}\Rightarrow\beta=\frac{c}{K}\leq\frac{\varepsilon\mu}{32L\eta e^{\nicefrac{{3c}}{{2}}}}. |  | (29) |

We can satisfy the right inequality above if we choose KK such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | K≥32​L​η​c​e3​c/2ε​μ​=\Hy@raisedlink([a](#desca9 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​64​L​e3​cμ2​ε​log⁡(2​(f​(x0)−f⋆)ε),K\geq\frac{32L\eta ce^{\nicefrac{{3c}}{{2}}}}{\varepsilon\mu}\overset{\text{\Hy@raisedlink{\hypertarget{a9}{}}{(\hyperlink{desca9}{a})}}}{=}\frac{64Le^{3c}}{\mu^{2}\varepsilon}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right), |  | (30) |

where \Hy@raisedlink([a](#a9 "\Hy@raisedlink ‣ Equation 30 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the value of η\eta in ([22](#A4.E22 "Equation 22 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). This choice of KK is satisfied by the choice in ([23](#A4.E23 "Equation 23 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Finally, we want

|  |  |  |  |
| --- | --- | --- | --- |
|  | 2​ρ​σ​β​ηα≤ε8⇒β=cK\displaystyle 2\rho\sigma\frac{\beta\eta}{\alpha}\leq\frac{\varepsilon}{8}\Rightarrow\beta=\frac{c}{K} | ≤ε​α16​ρ​σ​η​≤\Hy@raisedlink([a](#desca10 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​min⁡{ε16​ρ​σ​η,ε​(ε​μ)216​ρ​σ​η​(32​ρ​σ)2​e3​c},\displaystyle\leq\frac{\varepsilon\alpha}{16\rho\sigma\eta}\overset{\text{\Hy@raisedlink{\hypertarget{a10}{}}{(\hyperlink{desca10}{a})}}}{\leq}\min\left\{\frac{\varepsilon}{16\rho\sigma\eta},\frac{\varepsilon(\varepsilon\mu)^{2}}{16\rho\sigma\eta(32\rho\sigma)^{2}e^{3c}}\right\}, |  |

where \Hy@raisedlink([a](#a10 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses ([27](#A4.E27 "Equation 27 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). The above inequality is satisfied with the choice of KK such that

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | K\displaystyle K | ≥max⁡{16​ρ​σ​η​cε,16​ρ​σ​η​c​(32​ρ​σ)2​e3​cε​(ε​μ)2}\displaystyle\geq\max\left\{\frac{16\rho\sigma\eta c}{\varepsilon},\frac{16\rho\sigma\eta c(32\rho\sigma)^{2}e^{3c}}{\varepsilon(\varepsilon\mu)^{2}}\right\} |  | (31) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =\Hy@raisedlink([a](#desca11 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))​max⁡{32​ρ​σ​e3​c/2ε​μ,(32​ρ​σ​e3​c/2)3(ε​μ)3}​log⁡(2​(f​(x0)−f⋆)ε),\displaystyle\overset{\text{\Hy@raisedlink{\hypertarget{a11}{}}{(\hyperlink{desca11}{a})}}}{=}\max\left\{\frac{32\rho\sigma e^{\nicefrac{{3c}}{{2}}}}{\varepsilon\mu},\frac{(32\rho\sigma e^{\nicefrac{{3c}}{{2}}})^{3}}{(\varepsilon\mu)^{3}}\right\}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right), |  |

where \Hy@raisedlink([a](#a11 "\Hy@raisedlink ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) uses the value of η\eta in ([22](#A4.E22 "Equation 22 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). This bound on KK is satisfied by the choice in ([22](#A4.E22 "Equation 22 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). A combination of ([28](#A4.E28 "Equation 28 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([30](#A4.E30 "Equation 30 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), ([31](#A4.E31 "Equation 31 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) gives the choice of KK in ([23](#A4.E23 "Equation 23 ‣ Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | K\displaystyle K | =max⁡{128​L​e3​cε​μ2,32​ρ​σ​e3​c/2ε​μ,128​L​e6​c​(32​ρ​σ)2μ​(ε​μ)3,(32​ρ​σ​e3​c/2)3(ε​μ)3}​log⁡(2​(f​(x0)−f⋆)ε).\displaystyle=\max\left\{\frac{128Le^{3c}}{\varepsilon\mu^{2}},\frac{32\rho\sigma e^{\nicefrac{{3c}}{{2}}}}{\varepsilon\mu},\frac{128Le^{6c}(32\rho\sigma)^{2}}{\mu(\varepsilon\mu)^{3}},\frac{(32\rho\sigma e^{\nicefrac{{3c}}{{2}}})^{3}}{(\varepsilon\mu)^{3}}\right\}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right). |  | (32) |

Now we show that the choice of KK, β\beta, and η\eta ensures that the first term in ([26](#A4.E26 "Equation 26 ‣ Proof. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) is smaller than ε/2\varepsilon/2. Let us show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | (1−μ​β​η2​e3​c/2)K​(f​(x0)−f⋆)≤e−μ​β​η​e−3​c/2​K/2​(f​(x0)−f⋆)≤ε2.\left(1-\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}\right)^{K}(f(x\_{0})-f^{\star})\leq e^{-\mu\beta\eta e^{-\nicefrac{{3c}}{{2}}}K/2}(f(x\_{0})-f^{\star})\leq\frac{\varepsilon}{2}. |  | (33) |

The last inequality is satisfied if the following condition holds:

|  |  |  |
| --- | --- | --- |
|  | μ​β​η2​e3​c/2​K≥log⁡(2​(f​(x0)−f⋆)ε).\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}K\geq\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right). |  |

Plugging in the choice of β=cK\beta=\frac{c}{K} and η=2​e3​c/2μ​c​log⁡(2​(f​(x0)−f⋆)ε)\eta=\frac{2e^{\nicefrac{{3c}}{{2}}}}{\mu c}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right), we obtain

|  |  |  |
| --- | --- | --- |
|  | μ​β​η2​e3​c/2​K=μ2​e3​c/2⋅cK⋅2​e3​c/2μ​c​log⁡(2​(f​(x0)−f⋆)ε)⋅K=log⁡(2​(f​(x0)−f⋆)ε).\frac{\mu\beta\eta}{2e^{\nicefrac{{3c}}{{2}}}}K=\frac{\mu}{2e^{\nicefrac{{3c}}{{2}}}}\cdot\frac{c}{K}\cdot\frac{2e^{\nicefrac{{3c}}{{2}}}}{\mu c}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right)\cdot K=\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right). |  |

Grouping the bounds together, we obtain that the choice of K,η,βK,\eta,\beta implies that

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[f​(xK)−f⋆]≤ε2+4⋅ε8=ε.\mathbb{E}[f(x\_{K})-f^{\star}]\leq\frac{\varepsilon}{2}+4\cdot\frac{\varepsilon}{8}=\varepsilon. |  |

∎

###### Corollary D.1 (Full statement of [Corollary˜4.1](#S4.Thmcorollary1 "Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")).

Under the setup of [Theorem˜D.1](#A4.Thmtheorem1 "Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), let the token budget be large enough: T≥max⁡{2​c​B​S,B​S2​log⁡(2(f(x0)−f∗ε)}T\geq\max\left\{2cBS,\frac{BS}{2}\log\left(\frac{2(f(x\_{0})-f^{\*}}{\varepsilon}\right)\right\}. Then, running the algorithm with parameters from [Theorem˜D.1](#A4.Thmtheorem1 "Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") for K=T/B​SK=\nicefrac{{T}}{{BS}} iterations, we achieve the optimization error

|  |  |  |  |
| --- | --- | --- | --- |
|  | ε=max⁡{128​L​B​S​e3​cμ2​T,(128​L​e6​c​(32​ρ​σ⋆)2μ4​T)1/3,32​e3​c/2​ρ​σ⋆μ​(T2​B​S)1/6}.\varepsilon=\max\left\{\frac{128LBSe^{3c}}{\mu^{2}T},\left(\frac{128Le^{6c}(32\rho\sigma\_{\star})^{2}}{\mu^{4}T}\right)^{1/3},\frac{32e^{\nicefrac{{3c}}{{2}}}\rho\sigma\_{\star}}{\mu(T^{2}BS)^{1/6}}\right\}. |  | (34) |

###### Proof.

From [Theorem˜D.1](#A4.Thmtheorem1 "Theorem D.1 (Full statement of Theorem˜4.1). ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), we have that to achieve the optimization error ε\varepsilon, we need to use KK iterations defined as

|  |  |  |
| --- | --- | --- |
|  | K=max⁡[128​L​e3​cε​μ2,32​e3​c/2​ρ​σε​μ,128​L​e6​c​(32​ρ​σ)2μ​(ε​μ)3,(32​ρ​σ​e3​c/2)3(ε​μ)3]​log⁡(2​(f​(x0)−f⋆)ε),K=\max\left[\frac{128Le^{3c}}{\varepsilon\mu^{2}},\frac{32e^{\nicefrac{{3c}}{{2}}}\rho\sigma}{\varepsilon\mu},\frac{128Le^{6c}(32\rho\sigma)^{2}}{\mu(\varepsilon\mu)^{3}},\frac{(32\rho\sigma e^{\nicefrac{{3c}}{{2}}})^{3}}{(\varepsilon\mu)^{3}}\right]\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right), |  |

ignoring the requirements K≥2​cK\geq 2c, K≥12​log⁡(2​(f​(x0)−f∗)ε)K\geq\frac{1}{2}\log\left(\frac{2(f(x\_{0})-f^{\*})}{\varepsilon}\right), which holds in practice (and also follows from the assumption on TT). Multiplying both sides of this expression by B​SBS, using [˜3.4](#S3.Thmassumption4 "Assumption 3.4. ‣ 3 Problem Formulation and Assumptions ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods") that says that σ2=σ⋆2B​S\sigma^{2}=\frac{\sigma\_{\star}^{2}}{BS}, and using the relation T=K​B​ST=KBS, we obtain

|  |  |  |
| --- | --- | --- |
|  | T=max⁡{128​L​e3​c​L​B​Sε​μ2,32​e3​c/2​ρ​σ⋆​B​Sε​μ,128​L​e6​c​(32​ρ​σ⋆)2μ​(ε​μ)3,(32​ρ​σ⋆​e3​c/2)3(ε​μ)3​B​S}​log⁡(2​(f​(x0)−f⋆)ε).T=\max\left\{\frac{128Le^{3c}LBS}{\varepsilon\mu^{2}},\frac{32e^{\nicefrac{{3c}}{{2}}}\rho\sigma\_{\star}\sqrt{BS}}{\varepsilon\mu},\frac{128Le^{6c}(32\rho\sigma\_{\star})^{2}}{\mu(\varepsilon\mu)^{3}},\frac{(32\rho\sigma\_{\star}e^{\nicefrac{{3c}}{{2}}})^{3}}{(\varepsilon\mu)^{3}\sqrt{BS}}\right\}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right). |  |

Since the token budget TT is fixed in the experiments, the expression above says that we cannot achieve an arbitrary optimization error ε\varepsilon:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ε\displaystyle\varepsilon | =max{128​e3​c​L​B​ST​μ2,32​e3​c/2​ρ​σ⋆​B​ST​μ,(128​L​e6​c​(32​ρ​σ⋆)2μ4​T)1/3,\displaystyle=\max\left\{\frac{128e^{3c}LBS}{T\mu^{2}},\frac{32e^{\nicefrac{{3c}}{{2}}}\rho\sigma\_{\star}\sqrt{BS}}{T\mu},\left(\frac{128Le^{6c}(32\rho\sigma\_{\star})^{2}}{\mu^{4}T}\right)^{1/3},\right. |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ((32​ρ​σ⋆​e3​c/2)3T​μ3​B​S)1/3}log(2​(f​(x0)−f⋆)ε).\displaystyle\hskip 256.0748pt\left.\left(\frac{(32\rho\sigma\_{\star}e^{\nicefrac{{3c}}{{2}}})^{3}}{T\mu^{3}\sqrt{BS}}\right)^{1/3}\right\}\log\left(\frac{2(f(x\_{0})-f^{\star})}{\varepsilon}\right). |  |

Now we compare the second and fourth terms in the expression above. We note that the second term is larger *iff*

|  |  |  |  |
| --- | --- | --- | --- |
|  | 32​e3​c/2​ρ​σ⋆​B​ST​μ≥32​ρ​σ⋆​e3​c/2T1/3​μ​(B​S)1/6⟺(B​S)2/3≥T2/3⟺B​S≥T.\frac{32e^{\nicefrac{{3c}}{{2}}}\rho\sigma\_{\star}\sqrt{BS}}{T\mu}\geq\frac{32\rho\sigma\_{\star}e^{\nicefrac{{3c}}{{2}}}}{T^{1/3}\mu(BS)^{1/6}}\Longleftrightarrow(BS)^{2/3}\geq T^{2/3}\Longleftrightarrow BS\geq T. |  | (35) |

In other words, the second term is smaller than or equal to the fourth term. Therefore, it can be ignored in the maximum. This finalizes the proof.
∎

Algorithm 2  Unconstrained Stochastic Conditional Gradient (uSCG)

Input: x0,m0∈𝒳x\_{0},m\_{0}\in\mathcal{X}, parameters α,η>0\alpha,\eta>0

for k=0,…,K−1k=0,\ldots,K-1 do

sample ξk∼𝒟\xi\_{k}\sim\mathcal{D}

compute mk+1=(1−α)​mk+α​g​(xk;ξk)m\_{k+1}=(1-\alpha)m\_{k}+\alpha g(x\_{k};\xi\_{k})

compute dk+1=arg​mind∈𝒳⁡⟨mk+1,d⟩d\_{k+1}={\rm arg}\min\_{d\in\mathcal{X}}\langle m\_{k+1},d\rangle s.t. ‖d‖≤1\|d\|\leq 1

compute xk+1=xk+η​dk+1x\_{k+1}=x\_{k}+\eta d\_{k+1}

end for

###### Remark D.2.

Our work is based on the convergence guarantees under the μ\mu-KL condition following prior work [Schaipp et al., [2025](#bib.bib129 "The surprising agreement between convex optimization theory and learning-rate scheduling for large model training"), Islamov et al., [2024](#bib.bib127 "Loss landscape characterization of neural networks without over-parametrization"), Tran et al., [2024](#bib.bib126 "Reevaluating theoretical analysis methods for optimization in deep learning"), Guille-Escuret et al., [2023](#bib.bib125 "No wrong turns: the simple geometry of neural networks optimization paths")] that provides evidence that the loss landscape of neural networks exhibits a convex-like structure. However, it is possible to extend the results to a standard non-convex setting under the smoothness assumption only. In such a case, one can consider Unconstrained SCG (Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) and the convergence metric changes from the function sub-optimality to a dual gradient norm, i.e., mink=0,1,…,K−1⁡𝔼​[‖∇f​(xk)‖∗]\min\_{k=0,1,\ldots,K-1}\mathbb{E}[\|\nabla f(x\_{k})\|\_{\*}] or 𝔼​[‖∇f​(x¯k)‖∗]\mathbb{E}[\|\nabla f(\overline{x}\_{k})\|\_{\*}] with x¯k\overline{x}\_{k} being selected uniformly at random from {x0,x1,…,xK−1}\{x\_{0},x\_{1},\ldots,x\_{K-1}\}; see [Pethick et al., [2025a](#bib.bib106 "Training deep learning models with norm-constrained LMOs"), Theorem 5.5] and [Kovalev, [2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization"), Corollary 2].

Under the setup of [Kovalev, [2025](#bib.bib96 "Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization"), Corollary 2] and a fixed token budget TT, we achieve the optimization error

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | ε\displaystyle\varepsilon | =\displaystyle= | 𝒪​(max⁡{L​Δ​B​ST,(L​Δ)1/4​ρ​σ⋆T1/4,ρ​σ⋆​B​ST,ρ​σ⋆T1/3​(B​S)1/6})\displaystyle\mathcal{O}\left(\max\left\{\frac{\sqrt{L\Delta BS}}{\sqrt{T}},\frac{(L\Delta)^{1/4}\sqrt{\rho\sigma\_{\star}}}{T^{1/4}},\frac{\rho\sigma\_{\star}\sqrt{BS}}{T},\frac{\rho\sigma\_{\star}}{T^{1/3}(BS)^{1/6}}\right\}\right) |  | (36) |
|  |  | =\displaystyle= | 𝒪​(max⁡{L​Δ​B​ST,(L​Δ)1/4​ρ​σ⋆T1/4,ρ​σ⋆T1/3​(B​S)1/6}),\displaystyle\mathcal{O}\left(\max\left\{\frac{\sqrt{L\Delta BS}}{\sqrt{T}},\frac{(L\Delta)^{1/4}\sqrt{\rho\sigma\_{\star}}}{T^{1/4}},\frac{\rho\sigma\_{\star}}{T^{1/3}(BS)^{1/6}}\right\}\right), |  |

where we used Δ=f​(x0)−f⋆\Delta=f(x\_{0})-f^{\star} and T≥B​ST\geq BS.
We observe that the
third term in ([36](#A4.E36 "Equation 36 ‣ Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) is identical to the third term in ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) (up to constant μ\mu, which is expected due to the change of convergence metric), while the first two are different. Besides, the middle term is also batch size and sequence length independent, but has a power T1/4T^{1/4} instead of T1/3T^{1/3} as in ([3](#S4.E3 "Equation 3 ‣ Corollary 4.1 (BST Scaling Rule). ‣ 4 Theoretical Analysis ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")). Following the approach of [Section˜5](#S5 "5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"), i.e., choosing BB and SS in the intersection of the first two terms in ([37](#A4.E37 "Equation 37 ‣ Remark D.2. ‣ Appendix D In-Expectation Convergence Proofs for SCG ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")), we derive the scaling rules similar to ([7](#S5.E7 "Equation 7 ‣ 5.1 Increasing Batch Size ‣ 5 Strategies for Hyperparameter Choice ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods"))

|  |  |  |  |
| --- | --- | --- | --- |
|  | B1​S1=B0​S0​D1D0​ρ12ρ02​L0L1,B\_{1}S\_{1}=B\_{0}S\_{0}\sqrt{\frac{D\_{1}}{D\_{0}}\frac{\rho\_{1}^{2}}{\rho\_{0}^{2}}\frac{L\_{0}}{L\_{1}}}, |  | (37) |

assuming that parameters Δ\Delta and σ⋆\sigma\_{\star} are independent of the model size. This approach is similar to ([16](#A3.E16 "Equation 16 ‣ C.1 Additional Baselines in Experiments from Section˜6.5 ‣ Appendix C Additional Experiments ‣ On the Role of Batch Size in Stochastic Conditional Gradient Methods")) up to problem-dependent constants ρ\rho and LL.

[◄](/html/2603.21189)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2603.21191)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2603.21191)
[View original  
on arXiv](https://arxiv.org/abs/2603.21191)[►](/html/2603.21192)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Apr 6 05:31:36 2026 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
