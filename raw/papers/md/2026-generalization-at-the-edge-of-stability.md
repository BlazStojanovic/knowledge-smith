---
arxiv: '2604.19740'
authors:
- Mario Tuci
- Caner Korkmaz
- Umut Şimşekli
- Tolga Birdal
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: Generalization at the Edge of Stability
url: https://arxiv.org/abs/2604.19740
year: 2026
---

# Generalization at the Edge of Stability

\nameMario Tuci \emailmario.tuci@inria.fr\addrINRIA, CNRS, Département d’Informatique de l’Ecole Normale Supérieure / PSL, France\addrDepartment of Computing, Imperial College London, United Kingdom
  
[0.5em]
\nameCaner Korkmaz \emailc.korkmaz23@imperial.ac.uk\addrDepartment of Computing, Imperial College London, United Kingdom
 Authors contributed equally.
  
[0.5em]
\nameUmut Şimşekli \emailumut.simsekli@inria.fr\addrINRIA, CNRS, Département d’Informatique de l’Ecole Normale Supérieure / PSL, France
  
[0.5em]
\nameTolga Birdal \emailtbirdal@imperial.ac.uk\addrDepartment of Computing, Imperial College London, United Kingdom

###### Abstract

Training modern neural networks often relies on large learning rates, operating at the *edge of stability*, where the optimization dynamics exhibit oscillatory and *chaotic behavior*. Empirically, this regime often yields improved generalization performance, yet the underlying mechanism remains poorly understood. In this work, we represent stochastic optimizers as *random dynamical systems*, which often converge to a *fractal attractor set* (rather than a point) with a smaller intrinsic dimension. Building on this connection and inspired by Lyapunov dimension theory, we introduce a novel notion of dimension, coined the ‘sharpness dimension’, and prove a generalization bound based on this dimension. Our results show that generalization in the chaotic regime depends on the *complete* Hessian spectrum and the structure of its *partial determinants*, highlighting a complexity that cannot be captured by the trace or spectral norm considered in prior work. Experiments across various MLPs and transformers validate our theory while also providing new insights into the recently observed phenomenon of grokking.

## 1 Introduction

Understanding why large, overparameterized neural networks trained by gradient-based methods generalize remains one of the central open problems in modern machine learning.

!(/html/2604.19740/assets/x1.png)

Figure 1: Generalization at the Edge of Stability (EoS). Modeling stochastic optimization as a random dynamical system (RDS), we show that at EoS the leading sharpness satisfies λ1>0\lambda\_{1}>0, implying expansion along at least one direction. The fundamental balance between expansion and contraction implies that the effective dimensionality of the dynamics, measured by our Sharpness Dimension (SD), is strictly smaller than the ambient parameter space: SD<d\mathrm{SD}<d. We prove that the worst-case generalization error is governed by SD\mathrm{SD} rather than the parameter count. Our results identify EoS as precisely the regime where generalization is controlled by a provably lower-dimensional attractor, providing a principled explanation for why overparameterized models can generalize beyond classical complexity measures.

Recent empirical evidence has revealed a phenomenon that challenges classical convex optimization theory. Cohen et al., ([2021](#bib.bib16)) observed that when training neural networks with gradient descent (GD) with a fixed learning rate η\eta, the largest eigenvalue of the loss Hessian often oscillates around, and frequently exceeds 2/η2/\eta, even as the training loss continues to decrease. This behavior, termed the *edge of stability* (EoS), has generated considerable interest (see Sec. [2](#S2 "2 Related Work ‣ Generalization at the Edge of Stability")), since the threshold 2/η2/\eta implies instability and divergence for quadratic objectives Ghosh et al., ([2025](#bib.bib29)).
Ly and Gong, ([2025](#bib.bib49)) demonstrated that exceeding the threshold of 2η\frac{2}{\eta} is sufficient to induce chaotic training dynamics. Additionally, in the chaotic regime the optimizer will not settle at a single point, rather it explores a bounded, typically fractal like set Singh Kalra et al., ([2023](#bib.bib67)).
This raises a fundamental question:

*How can generalization be explained in the regime that is locally unstable and potentially chaotic?*

A natural response to this puzzle is to examine the local geometry of the loss landscape. In particular, the Hessian, which encodes local curvature, has long been viewed as a key lens for understanding generalization (Keskar et al.,, [2016](#bib.bib44); [Jiang et al., 2019a,](#bib.bib38) ), motivating a large body of work based on *pointwise* notions of *sharpness* and *flatness* (see Sec. [2](#S2.SS0.SSS0.Px2 "Hessian and generalization ‣ 2 Related Work ‣ Generalization at the Edge of Stability")). Despite its appeal and empirical successes, this viewpoint has been challenged. It is now well understood that sharpness-based criteria (e.g., Hessian trace) are neither necessary nor sufficient for good generalization: there exist flat minima that generalize poorly and sharp that do well (Dinh et al.,, [2017](#bib.bib21); Kaur et al.,, [2023](#bib.bib42); Wen et al.,, [2023](#bib.bib72)).

Consequently, characterizing generalization at the edge of stability through pointwise analysis of individual solutions may be fundamentally inadequate. For practical learning rates, training dynamics are expected to exhibit chaotic behavior (Singh Kalra et al.,, [2023](#bib.bib67)), wherein training trajectories display sensitive dependence on initialization. In this regime, generalization performance should be attributed not to the properties of any single solution, but rather to the geometric and characteristics of the entire solution set explored by the optimizer in the long term.

#### Contributions

We introduce a framework for studying generalization at the edge of chaos where modern deep networks operate (Cai et al.,, [2026](#bib.bib10)). In particular, we contribute the following:

* •

  Attractor-Centric Framework: Modeling stochastic optimization as a random dynamical system (RDS), we shift the study of generalization at EoS from isolated parameter vectors to the geometric properties of the (random) attractor.
* •

  RDS Sharpness & Sharpness Dimension (SD\mathrm{SD}): We propose two new complexity measures, RDS Sharpness & SD\mathrm{SD}, derived not from the trajectory but from expansion and contraction rates that characterizes the attractor’s geometry.
* •

  Generalization Bound: We provide a new bound on the worst case generalization error, rigorously linking it to the fractal dimension of the Random Attractor measured as SD\mathrm{SD}.
* •

  Empirical Validation: We explain how to compute SD efficiently and deploy our findings in quantifying generalization across multilayer perceptrons and recent transformers (GPT-2 (Radford et al.,, [2019](#bib.bib61))) as well as to study the recently introduced paradigm of *grokking*, delayed and sudden generalization (Power et al.,, [2022](#bib.bib59); Prieto et al.,, [2025](#bib.bib60)).

As illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Generalization at the Edge of Stability"), our results identify EoS as precisely the regime where generalization is controlled by a provably lower-dimensional attractor, whose effective dimensionality is controlled by SD and is strictly smaller than the ambient parameter space. We prove that in the EoS, the worst-case generalization error is governed exactly by SD\mathrm{SD}. Our findings present a principled explanation for why overparameterized models can generalize beyond classical complexity measures. Our code will be publicly available under: <https://circle-group.github.io/research/GATES>.

## 2 Related Work

#### Generalization bounds

Recent work has established strong empirical and theoretical connections between generalization and geometric and topological complexity measures derived from the *optimization trajectory* (Simsekli et al.,, [2020](#bib.bib66); Birdal et al.,, [2021](#bib.bib8); Dupuis et al.,, [2023](#bib.bib22); Andreeva et al.,, [2024](#bib.bib4); Tuci et al.,, [2025](#bib.bib69)). In particular, it has been shown that weight iterates sampled *after* initial convergence encode critical information about generalization. Topological summaries such as the ‘α\alpha-weighted lifetime sum’ exhibit consistent correlations with the generalization gap across training runs (Andreeva et al.,, [2024](#bib.bib4); Tuci et al.,, [2025](#bib.bib69)). These findings suggest that the stochastic fluctuations observed during late-stage training are not merely noise, but reflect the optimizer exploring a structured geometric object. Differently, Camuto et al., ([2021](#bib.bib11)) showed that under contraction, stochastic GD (SGD) admits an invariant measure supported on a fractal set and linked its geometric complexity to generalization. However, this theory relies on contraction and therefore does not apply at the EoS.

#### Hessian and generalization

Since Hochreiter and Schmidhuber, ([1994](#bib.bib33)), it has been conjectured that flatter minima generalize better, motivating extensive work on flatness and sharpness (Keskar et al.,, [2016](#bib.bib44); Dinh et al.,, [2017](#bib.bib21); Neyshabur et al.,, [2017](#bib.bib55); Sagun et al.,, [2017](#bib.bib63); Yao et al.,, [2018](#bib.bib75); Chaudhari et al.,, [2019](#bib.bib12); Simsekli et al.,, [2019](#bib.bib65); Nguyen et al.,, [2019](#bib.bib56); Mulayoff and Michaeli,, [2020](#bib.bib53); Tsuzuku et al.,, [2020](#bib.bib68); [Ahn et al., 2023b,](#bib.bib2) ). Despite this effort, there is no universally accepted definition of flatness, and most practical surrogates rely on second-order cues such as the trace of the Hessian (e.g., Jastrzebski et al.,, [2020](#bib.bib37); Wen et al.,, [2023](#bib.bib72)).

In parallel, optimization methods explicitly designed to favor flat minima have shown strong empirical gains in generalization (Izmailov et al.,, [2018](#bib.bib36); Wu et al.,, [2020](#bib.bib73); Zheng et al.,, [2021](#bib.bib78); Kaddour et al.,, [2022](#bib.bib40); Foret et al.,, [2020](#bib.bib25)). Theoretically, minimizing the Hessian trace was shown to select the true solution in low-rank matrix recovery (Ding et al.,, [2024](#bib.bib20)), extended to deep networks (Gatmiry et al.,, [2023](#bib.bib27)), and supported empirically for large language models (Liu et al.,, [2023](#bib.bib47)), and linked to output stability (Ma and Ying,, [2021](#bib.bib50)). More recently, generalization bounds have been obtained in terms of sums of gradient norms (Haddouche et al.,, [2024](#bib.bib31); Clerico et al.,, [2022](#bib.bib15)), which reduce to the Hessian trace under suitable conditions. Despite these advances, recent results show that trace-based flatness alone does not guarantee good generalization (Wen et al.,, [2023](#bib.bib72)).

#### Edge of Stability (EoS) and chaos

The dynamics of deep network training have been widely studied, with early work documenting rapid changes in the local loss landscape during the initial phase of optimization (Keskar et al.,, [2016](#bib.bib44); Jastrzebski et al.,, [2020](#bib.bib37); Xing et al.,, [2018](#bib.bib74)), and later characterizing it for gradient descent by Cohen et al., ([2021](#bib.bib16)). A growing literature has since been interested in EoS phenomena Arora et al., ([2022](#bib.bib7)); [Ahn et al., 2023a](#bib.bib1) ; Ahn et al., ([2022](#bib.bib3)); Wang et al., ([2022](#bib.bib71)); Chen and Bruna, ([2023](#bib.bib14)); Damian et al., ([2022](#bib.bib19)); Zhu et al., ([2022](#bib.bib79)). In particular, Ahn et al., ([2022](#bib.bib3)); Ma and Ying, ([2021](#bib.bib50)) showed that the existence of a forward-invariant set prevents divergence, and that for tanh\tanh networks such a set exists, explaining stability even at large learning rates in the EoS regime. Related phenomena have also been observed for SGD via the notion of mini-batch sharpness (Andreyev and Beneventano,, [2024](#bib.bib5)). From a complementary perspective, stability analyses reveal connections to chaos (Sasdelli et al.,, [2021](#bib.bib64); Ly and Gong,, [2025](#bib.bib49)): in particular, Ly and Gong, ([2025](#bib.bib49)) show that sustained criticality of the top Hessian eigenvalue (EoS) is sufficient to induce chaos, and Chemnitz and Engel, ([2025](#bib.bib13)) study a cubic model to characterize the boundary between EoS and divergence.

Our work connects Edge of Stability, chaotic dynamics, Hessian-based generalization bounds via rigorous generalization guarantees through the lens of random dynamical systems.

## 3 Preliminaries

#### Learning setup

We consider supervised learning with parameter vector w∈ℝdw\in\mathbb{R}^{d} and population risk ℛ​(w):=𝔼Z∼μ​[ℓ​(w,Z)]\mathcal{R}(w):=\mathbb{E}\_{Z\sim\mu}[\ell(w,Z)],
where μ\mu is an unknown data distribution and ℓ:ℝd×𝒵\ell:\mathbb{R}^{d}\times\mathcal{Z} is the composed loss function. In practice, training proceeds by minimizing the empirical risk ℛ^S​(w):=1n​∑i=1nℓ​(w,Zi)\widehat{\mathcal{R}}\_{S}(w):=\frac{1}{n}\sum\_{i=1}^{n}\ell(w,Z\_{i}) over a dataset S={Zi}i=1n∼μz⊗nS=\{Z\_{i}\}\_{i=1}^{n}\sim\mu\_{z}^{\otimes n} using stochastic gradient methods. Our analysis focuses on the asymptotic behavior of the optimization dynamics rather than on a single training iterate. Therefore we are interested in the worst-case generalization gap

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒢​(𝒜S​(ω)):=supw∈𝒜S​(ω)ℛ​(w)−ℛ^S​(w).\displaystyle\mathcal{G}(\mathcal{A}\_{S}(\omega)):=\sup\_{w\in\mathcal{A}\_{S}(\omega)}\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w). |  | (1) |

Here, 𝒜S​(ω)\mathcal{A}\_{S}(\omega) denotes a dataset-dependent random set, where S∈𝒵nS\in\mathcal{Z}^{n} is the dataset and ω\omega is an independent random variable capturing algorithmic noise (e.g., minibatch sampling). In particular, 𝒜​(ω)\mathcal{A}(\omega) denotes the *random attractor* associated with the optimizer, to be precised in Dfn [3.2](#S3.Thmdefinition2 "Definition 3.2 (Pullback Random Attractor). ‣ Random Attractors ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability").

#### Random dynamical systems

To have a rigorous consideration of random attractors as sets, we represent stochastic optimization algorithms as discrete-time *random dynamical systems* (RDS) according to Arnold, ([2006](#bib.bib6)). While the definition is abstract, it will follow with a concrete example for SGD, represented in the RDS form. Note that a similar formalism was introduced to study the stability of SGD in Chemnitz and Engel, ([2025](#bib.bib13)).

###### Definition 3.1 (Random Dynamical System).

A *discrete-time random dynamical system (RDS)* on ℝd\mathbb{R}^{d} is a tuple (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) consisting of the following components:

1. 1.

   A metric dynamical system (Ω,ℱ,ℙ,θ)(\Omega,\mathcal{F},\mathbb{P},\theta), where (Ω,ℱ,ℙ)(\Omega,\mathcal{F},\mathbb{P}) is a probability space and θ:Ω→Ω\theta:\Omega\to\Omega is an invertible, measure-preserving, and ergodic transformation111A transformation θ\theta is measure-preserving if ℙ​(θ−1​A)=ℙ​(A)\mathbb{P}(\theta^{-1}A)=\mathbb{P}(A) for all A∈ℱA\in\mathcal{F}, and ergodic if any invariant set A=θ−1​AA=\theta^{-1}A satisfies ℙ​(A)∈{0,1}\mathbb{P}(A)\in\{0,1\}. such that the map (t,ω)↦θt​(ω)(t,\omega)\mapsto\theta^{t}(\omega) defines a ℤ\mathbb{Z}-action on Ω\Omega. That is, the family of maps {θt}t∈ℤ\{\theta^{t}\}\_{t\in\mathbb{Z}} satisfies

   |  |  |  |
   | --- | --- | --- |
   |  | θ0=idΩ,θt+s=θt∘θsfor all ​t,s∈ℤ,\theta^{0}=\mathrm{id}\_{\Omega},\qquad\theta^{t+s}=\theta^{t}\circ\theta^{s}\quad\text{for all }t,s\in\mathbb{Z}, |  |

   where we use the notation θ​ω:=θ​(ω)\theta\omega:=\theta(\omega).
2. 2.

   A measurable cocycle ϕ:ℕ0×Ω×ℝd→ℝd\phi:\mathbb{N}\_{0}\times\Omega\times\mathbb{R}^{d}\to\mathbb{R}^{d} over θ\theta, which is (ℬ​(ℕ0)⊗ℱ⊗ℬ​(ℝd),ℬ​(ℝd))(\mathcal{B}(\mathbb{N}\_{0})\otimes\mathcal{F}\otimes\mathcal{B}(\mathbb{R}^{d}),\mathcal{B}(\mathbb{R}^{d}))333Let ℬ​(ℝd)\mathcal{B}(\mathbb{R}^{d}) or ℬ​(ℕ0)\mathcal{B}(\mathbb{N}\_{0}) denote the corresponding Borel σ\sigma-algebras.
   -measurable and satisfies the *cocycle property*:

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | | | |
   |  | ϕ​(0,ω,⋅)=Idℝd,\displaystyle\phi(0,\omega,\cdot)=\text{Id}\_{\mathbb{R}^{d}}, |  | (2a) |
   |  | ϕ​(t+s,ω,w)=ϕ​(t,θs​ω,ϕ​(s,ω,w)),\displaystyle\phi(t+s,\omega,w)=\phi(t,\theta^{s}\omega,\phi(s,\omega,w)), |  | (2b) |

   for all t,s∈ℕ0t,s\in\mathbb{N}\_{0}, ω∈Ω\omega\in\Omega, and w∈ℝdw\in\mathbb{R}^{d}.

An RDS is said to be kk times *continuously differentiable* (or CkC^{k}) if the mapping w↦ϕ​(t,ω,w)w\mapsto\phi(t,\omega,w) is CkC^{k} for all t∈ℕ0t\in\mathbb{N}\_{0} and ω∈Ω\omega\in\Omega.

Let us provide a more intuitive explanation for the above definition by considering SGD as a special case.
Given a dataset S={Z1,…,Zn}∈𝒵nS=\{Z\_{1},\dots,Z\_{n}\}\in\mathcal{Z}^{n}, SGD is based on the following recursion:

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1=wt−η​(1b​∑i∈Ωt∇ℓ​(wt,Zi)),\displaystyle w\_{t+1}=w\_{t}-\eta\left(\frac{1}{b}\sum\_{i\in\Omega\_{t}}\nabla\ell(w\_{t},Z\_{i})\right), |  | (3) |

where Ωt⊂{1,…,n}\Omega\_{t}\subset\{1,\dots,n\} is the minibatch with |Ωt|=b|\Omega\_{t}|=b being the batch-size. Since there are only finitely many possible choices of minibatches, i.e., (nb)\binom{n}{b} many, we can enumerate all the minibatches such as {Ω(1),…,Ω((nb))}\{\Omega^{(1)},\dots,\Omega^{(\binom{n}{b})}\} s.t. Ωt=Ω(j)\Omega\_{t}=\Omega^{(j)} for some jj. Hence ([3](#S3.E3 "Equation 3 ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")) can be alternatively written as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | wt+1=wt−η​(1b​∑i∈Ω(jt)∇ℓ​(wt,Zi)),\displaystyle w\_{t+1}=w\_{t}-\eta\left(\frac{1}{b}\sum\_{i\in\Omega^{(j\_{t})}}\nabla\ell(w\_{t},Z\_{i})\right), |  | (4) |

where jtj\_{t} is randomly drawn from the set {1,…,(nb)}\{1,\dots,\binom{n}{b}\}.

Going back to Definition [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"), the random event ω\omega corresponds to the *algorithmic randomness*. In the case of SGD, the only source of algorithmic randomness is the choice of minibatches at every iteration. Hence, for SGD, ω={…,j−t,…​j−1,j0,j1,…,jt,…}\omega=\{\dots,j\_{-t},\dots j\_{-1},j\_{0},j\_{1},\dots,j\_{t},\dots\} will encapsulate the infinite sequence of minibatch indices that are drawn through optimization444For technical reasons, we need to consider a doubly infinite sequence that also takes into account for negative times −t-t.. In other words, a single event ω\omega will contain all the information about the randomness coming from the algorithm.

Given a sequence of minibatch indices ω\omega, we can define the *RDS map* ϕ\phi, which essentially corresponds to the algorithm update rule. For t=1t=1, we have the following form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(1,ω,w):=w−η​(1b​∑i∈Ω(j0)∇ℓ​(w,Zi)),\displaystyle\phi(1,\omega,w):=w-\eta\left(\frac{1}{b}\sum\_{i\in\Omega^{(j\_{0})}}\nabla\ell(w,Z\_{i})\right), |  | (5) |

which coincides with ([4](#S3.E4 "Equation 4 ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")), but only for t=1t=1.

At iteration tt, the stochastic gradient is computed over the minibatch with index jt−1j\_{t-1}. Hence, to define the RDS map for a general tt, i.e. ϕ​(t,ω,w)\phi(t,\omega,w), we additionally need to extract the minibatch index jt−1j\_{t-1} from the infinite sequence ω\omega. This operation will be done by the *metric dynamical system* θ\theta. For this example555For most of the stochastic optimizers one can use the same metric dynamical system as long as the algorithmic randomness is only coming from random minibatches., we set θ:ℤ×Ω→Ω\theta:\mathbb{Z}\times\Omega\to\Omega to the so-called the ‘left-shift operator’, (θ​ω)k:=jk+1(\theta\omega)\_{k}:=j\_{k+1}, which takes an infinite sequence ω\omega and shifts its elements by one coordinate, and returns the resulting infinite sequence. Iterating this operator tt times would shift the coordinates tt times: i.e., (θt​ω)k:=jk+t(\theta^{t}\omega)\_{k}:=j\_{k+t}.

Given all the ingredients, by ([2b](#S3.E2.2 "Equation 2b ‣ Equation 2 ‣ Item 2 ‣ Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")) we can define our RDS for a general tt:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(t,ω,w)=ϕ​(1,θt−1​ω,ϕ​(t−1,ω,w)),\displaystyle\phi(t,\omega,w)=\phi(1,\theta^{t-1}\omega,\phi(t-1,\omega,w)), |  | (6) |

where ϕ​(1,⋅,⋅)\phi(1,\cdot,\cdot) is defined in ([5](#S3.E5 "Equation 5 ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")). It is now easy to verify that ([6](#S3.E6 "Equation 6 ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")) exactly recovers the recursion given in ([4](#S3.E4 "Equation 4 ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")).
Finally, we observe that the system satisfies the cocycle property:

|  |  |  |
| --- | --- | --- |
|  | ϕ​(t,ω,w)=ϕ​(1,θt−1​ω,⋅)∘⋯∘ϕ​(1,ω,w),∀t∈ℕ0.\displaystyle\phi(t,\omega,w)=\phi(1,\theta^{t-1}\omega,\cdot)\circ\dots\circ\phi(1,\omega,w),\forall t\in\mathbb{N}\_{0}. |  |

This property serves as a fundamental consistency requirement. Intuitively, it ensures that evolving a state for t+st+s steps is equivalent to evolving it for ss steps and then resuming for tt more steps using the remaining noise history θs​ω\theta^{s}\omega: system’s evolution is chronologically coherent.

While denoting an optimization algorithm with such abstract notions might seem rather unorthodox, thanks to this formal connection, we will be able to access the rich toolbox of random dynamical systems theory.

#### Random Attractors

Dynamics driven by stochastic optimization with persistent noise (e.g. constant learning rate SGD) generally do not converge to a single location. We are interested in the set in which an RDS settles.

###### Definition 3.2 (Pullback Random Attractor).

Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a discrete-time RDS according to Dfn. [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"). A mapping ω↦𝒜​(ω)\omega\mapsto\mathcal{A}(\omega) from Ω\Omega into the space of non-empty compact subsets of ℝd\mathbb{R}^{d} (denoted comp⁡(ℝd)\operatorname{comp}(\mathbb{R}^{d}) ) is called a *pullback random attractor* if it is (ℱ,ℬ​(comp⁡(ℝd)))(\mathcal{F},\mathcal{B}(\operatorname{comp}(\mathbb{R}^{d})))-measurable and satisfies the following two properties for ℙ\mathbb{P}-almost all ω∈Ω\omega\in\Omega:

1. 1.

   Invariance: ϕ​(t,ω,𝒜​(ω))=𝒜​(θt​ω)\phi(t,\omega,\mathcal{A}(\omega))=\mathcal{A}(\theta^{t}\omega) for all t∈ℕ0t\in\mathbb{N}\_{0}.
2. 2.

   Pullback Attraction: For every deterministic bounded set B⊂ℝdB\subset\mathbb{R}^{d}:

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | limt→∞dist⁡(ϕ​(t,θ−t​ω,B),𝒜​(ω))=0,\lim\_{t\to\infty}\operatorname{dist}\bigl(\phi(t,\theta^{-t}\omega,B),\mathcal{A}(\omega)\bigr)=0, |  | (7) |

   where dist⁡(X,Y):=supx∈Xinfy∈Y‖x−y‖\operatorname{dist}(X,Y):=\sup\_{x\in X}\inf\_{y\in Y}\|x-y\| is the Hausdorff semi-distance. Further the image of a function is defined as follows. If f:X→Yf:X\to Y is a map and A⊂XA\subset X, then the image of AA under ff is defined as f​(A):={f​(x)∈Y:x∈A}f(A):=\{\,f(x)\in Y\;:\;x\in A\,\}.

In stochastic systems, attraction must be understood in a *pullback* rather than forward sense. Because fresh noise is continually injected, trajectories that come close can later separate, so pathwise forward convergence generally fails. The pullback viewpoint instead fixes a noise realization and examines
the state at time t=0t=0 obtained by initializing the system in the remote past. The resulting pullback
attractor is a noise-conditioned “snapshot” of the asymptotic state, representing the set to which all past histories converge at the present time. This is the central object of our analysis.

## 4 Theoretical Results

#### Sharpness

Recent work introduced the notion of sharpness in terms of the Hessian of the empirical risk Cohen et al., ([2021](#bib.bib16)). We introduce an alternative notion of sharpness that extends to general random dynamical systems and provides an intuitive understanding of how the two notions are related.

Definition 4.1 
(RDS Sharpness):  
Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a C1C^{1} random dynamical system on ℝd\mathbb{R}^{d}. For a fixed state w∈ℝdw\in\mathbb{R}^{d}, let D​ϕ​(1,ω,w)∈ℝd×dD\phi(1,\omega,w)\in\mathbb{R}^{d\times d} denote the Jacobian (Fréchet derivative)
of the map x↦ϕ​(1,ω,x)x\mapsto\phi(1,\omega,x) evaluated at x=wx=w. Let 𝒜​(ω)\mathcal{A}(\omega) be random compact set. We define the RDS Sharpness of Order kk as the expected log-variation of the kk-th singular value:

λk:=𝔼​[supw∈𝒜​(ω)ln⁡σk​(ω,w)]∀k∈{1,…,d},\lambda\_{k}:=\mathbb{E}\left[\sup\_{w\in\mathcal{A}(\omega)}\ln\sigma\_{k}(\omega,w)\right]\quad\forall k\in\{1,\dots,d\},

(8)
where σ1​(ω,w)≥⋯≥σd​(ω,w)\sigma\_{1}(\omega,w)\geq\dots\geq\sigma\_{d}(\omega,w) are the singular values of D​ϕ​(1,ω,w)D\phi(1,\omega,w), assuming integrability holds.

The next example, shows how Dfn. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") relates to the classical notion of sharpness introduced in Cohen et al., ([2021](#bib.bib16)), which is defined in terms of the largest eigenvalue
of the Hessian.

Example 4.1 
(GD Sharpness):
We can interpret GD as a deterministic instance of a random dynamical system where the algorithmic randomness ω\omega is constant. Let ℛ^S​(w)\widehat{\mathcal{R}}\_{S}(w) denote the empirical risk for a fixed dataset S∈𝒵nS\in\mathcal{Z}^{n} and w∈ℝdw\in\mathbb{R}^{d}; the discrete-time update map ϕ\phi is defined as:

ϕ​(1,w)=w−η​∇ℛ^S​(w).\phi(1,w)=w-\eta\nabla\widehat{\mathcal{R}}\_{S}(w).

(9)
The local stability of this system is governed by the Jacobian D​ϕ​(1,w)=I−η​∇2ℛ^S​(w)D\phi(1,w)=I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w). In the optimization literature (Cohen et al.,, [2021](#bib.bib16)), sharpness is traditionally defined as the largest eigenvalue the Hessian ∇2ℛ^S​(w)\nabla^{2}\widehat{\mathcal{R}}\_{S}(w).
If we interpret our sharpness definition (see Dfn. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")) locally over point (e.g. 𝒜​(ω)={w}\mathcal{A}(\omega)=\{w\}), we obtain, that the sharpness of order 1 corresponds to λ1=ln⁡(‖I−η​∇2ℛ^S​(w)‖)\lambda\_{1}=\ln(\|I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w)\|), which exhibits related information to the largest eigenvalue of the hessian. Indeed in the EoS regime, we have λ1≥0\lambda\_{1}\geq 0.

#### Edge of Stability and Chaos

To build intuition for the relationship between our notion of RDS-sharpness of order 11 (Dfn. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")), the classical sharpness of Cohen et al., ([2021](#bib.bib16)), and the onset of chaos, we consider the linearized framework in Ly and Gong, ([2025](#bib.bib49)), and examine GD in the EoS regime through its *sensitivity to initial conditions*. Let ϕ​(K,w)\phi(K,w) denote the KK-step GD cocycle (cf. ([9](#S4.E9 "Equation 9 ‣ Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"))) and consider a reference trajectory xt=ϕ​(t,w0)=ϕ​(1,wt−1)x\_{t}=\phi(t,w\_{0})=\phi(1,w\_{t-1}). To track the propagation of an infinitesimal perturbation δ​wt\delta w\_{t}, we linearize the dynamics:

|  |  |  |
| --- | --- | --- |
|  | δ​wt+1:=ϕ​(1,wt+δ​wt)−ϕ​(1,wt)≈D​ϕ​(1,wt)​δ​wt,\delta w\_{t+1}:=\phi(1,w\_{t}+\delta w\_{t})-\phi(1,w\_{t})\approx D\phi(1,w\_{t})\delta w\_{t}, |  |

where D​ϕ​(1,wt)=I−η​∇2ℛ^S​(wt)D\phi(1,w\_{t})=I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w\_{t}) is the *one-step Jacobian* (cf. Ex. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")). The local exponential growth rate in the direction vt=δ​wt/‖δ​wt‖v\_{t}=\delta w\_{t}/\|\delta w\_{t}\| is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | log⁡‖δ​wt+1‖‖δ​wt‖=log⁡‖(I−η​∇2ℛ^S​(wt))​vt‖.\log\frac{\|\delta w\_{t+1}\|}{\|\delta w\_{t}\|}=\log\|(I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w\_{t}))v\_{t}\|. |  | (10) |

Let {(αi​(t),ui​(t))}i=1d\{(\alpha\_{i}(t),u\_{i}(t))\}\_{i=1}^{d} denote the eigenpairs of the Hessian ∇2ℛ^S​(wt)\nabla^{2}\widehat{\mathcal{R}}\_{S}(w\_{t}). If the perturbation direction vtv\_{t} is aligned with an eigenvector ui​(t)u\_{i}(t), the one-step growth factor is |1−η​αi​(t)||1-\eta\alpha\_{i}(t)|. In the EoS regime, the maximum eigenvalue αmax​(t)\alpha\_{\max}(t) typically exceeds the stability threshold 2/η2/\eta. When αmax​(t)>2/η\alpha\_{\max}(t)>2/\eta, the growth factor satisfies |1−η​αmax​(t)|>1|1-\eta\alpha\_{\max}(t)|>1, or equivalently, ln⁡|1−η​αmax​(t)|>0\ln|1-\eta\alpha\_{\max}(t)|>0.
In the context of Ex. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") this would make the singular value σ1\sigma\_{1} in ([8](#S4.E8 "Equation 8 ‣ Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")) larger than 0, hence we obtain λ1>0\lambda\_{1}>0 for this system.

If this condition holds on average along the trajectory, the top Lyapunov exponent Λ1:=limT→∞1T​∑t=0T−1ln⁡‖D​ϕ​(1,xt)​vt‖\Lambda\_{1}:=\lim\_{T\to\infty}\frac{1}{T}\sum\_{t=0}^{T-1}\ln\|D\phi(1,x\_{t})v\_{t}\| see Arnold, ([2006](#bib.bib6), Lemma 3.2.2, p. 113) becomes positive . In the language of dynamical systems, Λ1>0\Lambda\_{1}>0 implies that trajectories diverge from each other exponentially (even though the system might not be divergent), providing a signature of *deterministic chaos* within the EoS oscillations. Hence, we argue that EoS emerges when the system is chaotic with λ1>0\lambda\_{1}>0 and our goal is to develop theoretical tools specifically designed for this challenging setup.

###### Remark 4.1 (Sharpness for an RDS).

Andreyev and Beneventano, ([2024](#bib.bib5)) observed that the *expected* mini-batch Hessian typically concentrates near the stability threshold 2/η2/\eta.

#### Existence of Random Attractor

Since our work focuses on the random pullback attractor to which the RDS settles, we briefly discuss the existence of the random attractor.

###### Proposition 4.1 (Existence of the Random Pullback Attractor (Crauel et al.,, [1997](#bib.bib17))).

Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a C0C^{0} random dynamical system on ℝd\mathbb{R}^{d}. Suppose there exists a bounded random set K​(ω)K(\omega) that is pullback absorbing for ℙ\mathbb{P}-almost every ω∈Ω\omega\in\Omega; that is, for every deterministic bounded set D⊂ℝdD\subset\mathbb{R}^{d}, there exists a time T​(D,ω)≥0T(D,\omega)\geq 0 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(t,θ−t​ω,D)⊂K​(ω)∀t≥T​(D,ω).\phi(t,\theta^{-t}\omega,D)\subset K(\omega)\quad\forall t\geq T(D,\omega). |  | (11) |

Then, there exists a unique, compact and measurable 666Measurable with respect to the past σ\sigma-algebra ℱ−∞0:=σ​{θt​ω:t≤0}\mathcal{F}\_{-\infty}^{0}:=\sigma\{\theta^{t}\omega:t\leq 0\}. random pullback attractor 𝒜​(ω)\mathcal{A}(\omega) satisfying Dfn. [3.2](#S3.Thmdefinition2 "Definition 3.2 (Pullback Random Attractor). ‣ Random Attractors ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability").

Prop. [4.1](#S4.Thmtheorem1 "Proposition 4.1 (Existence of the Random Pullback Attractor (Crauel et al.,, 1997)). ‣ Existence of Random Attractor ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") has a clear dynamical meaning.
The existence of a pullback absorbing set K​(ω)K(\omega) means that, for a fixed noise realization ω\omega, all trajectories eventually enter a bounded region of the state space, provided the system is evolved from the distant past to the present.
The random pullback attractor 𝒜​(ω)\mathcal{A}(\omega) is the smallest invariant set inside this region.
It contains exactly the states that can be reached asymptotically under the fixed noise realization ω\omega.

Indeed, this demonstrates that local instability, when coupled with global dissipativity, leads to the emergence of a compact pullback attractor. An intuitive explanation of this mechanism, for example in the case of neural networks using tanh\tanh as activation function, is given by Ahn et al., ([2022](#bib.bib3)).

#### Complexity meassure of Random Attractors

Given the existence of the random pullback attractor, a natural question arises: Can we quantify its complexity, particularly in the EoS regime, where we expect chaos.

Definition 4.2 
(Sharpness Dimension):  
Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a C1C^{1} discrete random dynamical system on ℝd\mathbb{R}^{d}. Further let 𝒜​(ω)⊂ℝd\mathcal{A}(\omega)\subset\mathbb{R}^{d} be an almost surely compact random set. As in Dfn. [4](#S4.SS0.SSS0.Px1 "Sharpness ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"), let λk\lambda\_{k} be the RDS sharpness of order kk, for k=1,…,dk=1,\dots,d.
Set

j∗:=max⁡{i∈{1,…,d}:∑k=1iλk≥0},j^{\*}:=\max\left\{i\in\{1,\dots,d\}:\sum\_{k=1}^{i}\lambda\_{k}\geq 0\right\},
with the convention j∗=0j^{\*}=0 if λ1<0\lambda\_{1}<0.
The Sharpness Dimension of  𝒜​(ω)\mathcal{A}(\omega) is defined as

dimS𝒜:={j∗+∑i=1j∗λi|λj∗+1|,if ​1≤j∗<d,d,if ​j∗=d,0,if ​λ1<0.\dim\_{\mathrm{S}}\mathcal{A}:=\begin{cases}j^{\*}+\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|},&\text{if }1\leq j^{\*}<d,\\
d,&\text{if }j^{\*}=d,\\
0,&\text{if }\lambda\_{1}<0.\end{cases}

Similar notions have been previously considered Kaplan and Yorke, ([2006](#bib.bib41)); Hunt, ([1996](#bib.bib35)); Feng and Simon, ([2022](#bib.bib24)).
The Sharpness Dimension SD measures the effective number of expanding directions of the dynamics on the attractor before global contraction dominates. It is defined from the ordered global sharpness indices λ1≥λ2≥⋯≥λd\lambda\_{1}\geq\lambda\_{2}\geq\dots\geq\lambda\_{d}, which quantify worst-case logarithmic stretching rates along principal directions on the attractor. Let j∗j^{\*} be the largest index such that ∑k=1j∗λk≥0\sum\_{k=1}^{j^{\*}}\lambda\_{k}\geq 0; then j∗j^{\*} is the maximal dimension in which volumes do not contract. Indeed, we observe that, in the case of a proper set, SD is strictly smaller than the ambient dimension in the EoS regime where λ1\lambda\_{1} is expected to be positive.

#### Generalization Bounds

We are now ready to formulate our main theorem and establish the connection between our novel complexity measure and the worst-case generalization gap over the random pullback attractor.
We start by stating our main assumptions, the first two of which are standard practice in the literature
(Simsekli et al.,, [2020](#bib.bib66); Andreeva et al.,, [2024](#bib.bib4); Dupuis et al.,, [2024](#bib.bib23); Birdal et al.,, [2021](#bib.bib8)):

###### Assumption 4.2 (Boundedness of Loss).

We assume the loss function ℓ:ℝd×𝒵→ℝ\ell:\mathbb{R}^{d}\times\mathcal{Z}\to\mathbb{R} to be bounded. That is, there exists a constant B>0B>0 such that for all weights w∈ℝdw\in\mathbb{R}^{d} and z∈𝒵z\in\mathcal{Z} holds: 0≤ℓ​(w,ω)≤B0\leq\ell(w,\omega)\leq B.

###### Assumption 4.3 (Lipschitz Continuity of Loss).

We assume that the loss function ℓ:ℝd×𝒵→ℝ\ell:\mathbb{R}^{d}\times\mathcal{Z}\to\mathbb{R} satisfies the following properties for all z∈𝒵z\in\mathcal{Z}:
The function w↦ℓ​(w,z)w\mapsto\ell(w,z) is LL-Lipschitz continuous for some L>0L>0. That is, for all w1,w2∈ℝdw\_{1},w\_{2}\in\mathbb{R}^{d}: |ℓ​(w1,z)−ℓ​(w2,z)|≤L​‖w1−w2‖|\ell(w\_{1},z)-\ell(w\_{2},z)|\leq L\|w\_{1}-w\_{2}\|.

###### Assumption 4.4 (Regular Random Dynamics).

For each dataset S∈𝒵nS\in\mathcal{Z}^{n}, let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a C2C^{2} discrete-time RDS according to Dfn. [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"), with a unique compact random pullback attractor 𝒜S​(ω)\mathcal{A}\_{S}(\omega). We assume:

1. 1.

   Non-Singularity: For ℙ\mathbb{P}-a.e. ω\omega we assume infx∈𝒜​(ω)σd(Dϕ(1,ω,x)>0\inf\_{x\in\mathcal{A}(\omega)}\sigma\_{d}(D\phi(1,\omega,x)>0
2. 2.

   *Integrability*:

   |  |  |  |
   | --- | --- | --- |
   |  | 𝔼​[supx∈𝒜S​(ω)ln⁡‖D​ϕS​(1,ω,x)‖]<∞and𝔼​[supx∈𝒜S​(ω)ln⁡‖D2​ϕS​(1,ω,x)‖]<∞.\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}\_{S}(\omega)}\ln\|D\phi\_{S}(1,\omega,x)\|\right]<\infty\qquad\text{and}\qquad\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}\_{S}(\omega)}\ln\|D^{2}\phi\_{S}(1,\omega,x)\|\right]<\infty. |  |
3. 3.

   *Transition Index*: There exists an integer j∗∈{1,…,d−1}j^{\*}\in\{1,\dots,d-1\} such that:

   |  |  |  |
   | --- | --- | --- |
   |  | ∑i=1j∗λi≥ 0>∑i=1j∗+1λi,\sum\_{i=1}^{j^{\*}}\lambda\_{i}\;\geq\;0\;>\;\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}, |  |

   where λi\lambda\_{i} denotes the global sharpness of order ii (Dfn. [4](#S4.SS0.SSS0.Px4 "Complexity meassure of Random Attractors ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")).
4. 4.

   Bounded Distortion: For A∈ℝd×dA\in\mathbb{R}^{d\times d} and j∈{1,…,d}j\in\{1,\dots,d\}, define

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ‖A‖j:=σ1​(A)​⋯​σj​(A),\|A\|\_{j}:=\sigma\_{1}(A)\cdots\sigma\_{j}(A), |  | (12) |

   where σ1​(A)≥⋯≥σd​(A)\sigma\_{1}(A)\geq\cdots\geq\sigma\_{d}(A) are the singular values of AA. Equivalently, ‖A‖j\|A\|\_{j} is the maximal expansion factor of AA on jj-dimensional volumes.

   We assume that the spatial variation of ‖D​ϕ​(m,ω,⋅)‖j\|D\phi(m,\omega,\cdot)\|\_{j} over the attractor is subexponential in mm: for each j∈{1,…,d}j\in\{1,\dots,d\},

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | limm→∞1m​𝔼​[supx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j−infx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j]=0.\lim\_{m\to\infty}\frac{1}{m}\,\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\;-\;\inf\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\right]=0. |  | (13) |

Intuitively, the jj-volume growth rates along different orbits in 𝒜​(ω)\mathcal{A}(\omega) may differ at sub-exponential order, but coincide to leading exponential order as m→∞m\to\infty. Conditions of this type, requiring the spatial variation of the cocycle to be subexponential, are commonly imposed in the smooth ergodic theory of random dynamical systems to obtain Lyapunov exponents that do not depend on the base point x∈𝒜​(ω)x\in\mathcal{A}(\omega); see, e.g., Arnold Arnold, ([2006](#bib.bib6)) for related formulations.
Under Assumption [4.4](#S4.Thmtheorem4 "Assumption 4.4 (Regular Random Dynamics). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"), we now present our main result.

Theorem 4.5 
(Generalization via Sharpness Dimension):
Let S={z1,…,zn}∼μz⊗nS=\{z\_{1},\dots,z\_{n}\}\sim\mu\_{z}^{\otimes n} be a dataset of size nn. Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a discrete-time RDS according to Dfn [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability") such that Assump. [4.4](#S4.Thmtheorem4 "Assumption 4.4 (Regular Random Dynamics). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") holds.
Under Assumps. [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Boundedness of Loss). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") and [4.3](#S4.Thmtheorem3 "Assumption 4.3 (Lipschitz Continuity of Loss). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"), there exists a constant C>0C>0 s.t. with probability at least 1−ζ−γ1-\zeta-\gamma over the joint draw (S,ω)∼μz⊗n⊗ℙ(S,\omega)\sim\mu\_{z}^{\otimes n}\otimes\mathbb{P}, there exists δn,γ>0\delta\_{n,\gamma}>0 such that for all 0<δ<δn,γ0<\delta<\delta\_{n,\gamma},

𝒢S​(𝒜​(ω))\displaystyle\mathcal{G}\_{S}(\mathcal{A}(\omega))
≤2​L​δ+2​B​4​dimS𝒜S​log⁡(1/δ)n\displaystyle\leq 2L\delta+2B\sqrt{\frac{4\,\dim\_{\mathrm{S}}\mathcal{A}\_{S}\>\log(1/\delta)}{n}}

+I∞​(𝒜S​(ω),S)+log⁡(1/ζ)n+C​B2n.\displaystyle\quad+\frac{I\_{\infty}(\mathcal{A}\_{S}(\omega),S)+\log(1/\zeta)}{\sqrt{n}}+\frac{CB^{2}}{\sqrt{n}}.
We recall that 𝒢S​(𝒜​(ω))\mathcal{G}\_{S}(\mathcal{A}(\omega)) denotes the worst-case generalization gap (see ([1](#S3.E1 "Equation 1 ‣ Learning setup ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"))) and I∞​(𝒜S​(ω),S)I\_{\infty}(\mathcal{A}\_{S}(\omega),S) (see Dfn. [A.5](#A1.Thmdefinition5 "Definition A.5 (Total mutual information). ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")) denotes the total mutual information between the random pullback attractor 𝒜S​(ω)\mathcal{A}\_{S}(\omega) and SS.

Thm [4](#S4.SS0.SSS0.Px5 "Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") rigorously links the generalization gap to the stability of training by showing that, at the EoS (where dimS𝒜S<d\dim\_{\mathrm{S}}\mathcal{A}\_{S}<d), the generalization gap is governed by the global geometry of the attractor rather than by any isolated solution. In particular, the observed expansion along at least one direction implies that the attractor is confined to a proper subset of the ambient parameter space, of dimension strictly smaller than dd. The sharpness dimension dimS𝒜S\dim\_{\mathrm{S}}\mathcal{A}\_{S} quantifies this effect by capturing the spectral balance between the expanding directions and the remaining contracting ones.

###### Proof sketch.

The main idea is to show that dimS𝒜S\dim\_{\mathrm{S}}\mathcal{A}\_{S} upper bounds another notion of fractal dimension called the Minkowski dimension (MD, see Dfn. [A.4](#A1.Thmdefinition4 "Definition A.4 (Minkowski dimensions). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")). As the MD is directly linked to covering numbers, we directly obtain a generalization bound by relying on existing results (Dupuis et al.,, [2024](#bib.bib23)). Proving the fact that dimS𝒜S\dim\_{\mathrm{S}}\mathcal{A}\_{S} is larger than MD is achieved by covering 𝒜S​(ω)\mathcal{A}\_{S}(\omega) by using ellipsoids whose principal axes are determined by λ1,…,λd\lambda\_{1},\dots,\lambda\_{d}, then computing the covering number for these ellipsoids. The proof is given in App. [B](#A2.SSx2.SSS0.Px13 "Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability").
∎

Finally, the mutual term I∞I\_{\infty}, wich is used to handle the dependece of the dataset SS and the random pullback attractor 𝒜S​(ω)\mathcal{A}\_{S}(\omega), can be avoided by using the recent ‘set stability’ framework of Tuci et al., ([2025](#bib.bib69)). Within this framework, we provide another generalization bound without the mutual information term in Thm. [B.5](#A2.Thmtheorem5 "Theorem B.5 (Stability 𝐷_𝑆 Bound). ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability").

## 5 Empirical Results

#### Numerical Implementation

Unlike trajectory-based approaches Birdal et al., ([2021](#bib.bib8)); Simsekli et al., ([2020](#bib.bib66)); Andreeva et al., ([2024](#bib.bib4)), our generalization bound neither requires access to the full training trajectory nor necessitates tools of topological data analysis. Instead, our computational bottleneck lies in estimating the eigenvalue spectrum of the Hessian of a parameter near convergence (see App. [D](#A4 "Appendix D Additional Results ‣ Generalization at the Edge of Stability") for a formal approximation result). For small-scale networks, this can be achieved via (potentially randomized) GPU-parallelized singular value decomposition (SVD) Halko et al., ([2011](#bib.bib32)). However, the quadratic memory complexity of the Hessian renders such approaches rapidly intractable as model size grows. This challenge is exacerbated by the fact that we require access not merely to the leading eigenvalues, but to the entire spectrum, which is dominated by a large mass of near-zero modes, precisely the regime where Krylov-based methods such as Lanczos iterations Lanczos, ([1950](#bib.bib45)) become ineffective. To address this, we adopt *stochastic Lanczos quadrature* (SLQ) Lin et al., ([2016](#bib.bib46)); Golub and Welsch, ([1969](#bib.bib30)), a scalable spectral estimation technique that has recently been analyzed and successfully applied in the context of (moderately sized) neural networks Ghorbani et al., ([2019](#bib.bib28)); Papyan, ([2018](#bib.bib57)). Unlike classical SLQ where all runs probe a fixed matrix–vector product operator, we vary the operator across runs by computing Hessian–vector products on independently sampled minibatches. The operator is held fixed within each Lanczos run but resampled across runs so as to estimate the SD directly in expectation rather than post-hoc averaging over a single global spectrum.
We present a complexity analysis in App. [D](#A4 "Appendix D Additional Results ‣ Generalization at the Edge of Stability").

#### Metrics

We assess the correlation between various notions of complexity and generalization error by using Kendall’s coefficients (KC) Kendall, ([1938](#bib.bib43)) as well as their “granulated” versions (GC) [Jiang et al., 2019b](#bib.bib39) . While the classical KC (denoted τ\tau) measures correlation between two quantities, it may fail to capture their causal relationship. Instead, one GC is defined for each hyperparameter (i.e., 𝝍LR\boldsymbol{\psi}\_{\mathrm{LR}} for η\eta and 𝝍BS\boldsymbol{\psi}\_{\mathrm{BS}} for bb); it measures correlation when only this hyperparameter is varying. Note that scaling the constant BB in Thm. [4](#S4.SS0.SSS0.Px5 "Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") does not affect the observed correlation between generalization and topological complexities.

As target metrics, we compute the *generalization gap* (Gen. Gap) and *loss gap* as the absolute difference in accuracy and loss, respectively, both between training and test data.

#### Notions of complexity

As a proxy to generalization, we use both Euclidean (∥⋅∥\|\cdot\|) and data-dependent (ρ\rho) persistent homology (PH) dimensions (PH-Dim) Birdal et al., ([2021](#bib.bib8)); Dupuis et al., ([2023](#bib.bib22)), and α\alpha-weighted lifetime sums Andreeva et al., ([2024](#bib.bib4)) (EαE\_{\alpha}) using a simulated trajectory of 5,000 iterations (PH-Dim-Fwd). We also consider the last 5,000 training iterations without trajectory simulation (PH-Dim-Bwd). We further include our *RDS sharpness* as λ1\lambda\_{1}, the leading singular value (SV) of I−η​∇2ℛ^S​(w)I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w) and our Sharpness Dimension (SD). In grokking settings, we also track *Hessian trace* (tr​(H)\mathrm{tr}(H)) and the magnitude of its largest eigenvalue (*sharpness*), obtained as the absolute value of the largest Hessian eigenvalue. When η⋅sharpness>1\eta\cdot\mathrm{sharpness}>1, the two quantities correspond; otherwise, RDS sharpness is determined by a non-positive Hessian eigenvalue.

### 5.1 Analysis

#### Generalization in 3-layer small MLPs

We begin by experimenting on classical learning scenarios with a small 3-layer MLP trained on MNIST that allows computation of the exact eigenvalue spectrum of Hessian matrices. We use a width of 16, ReLU activation, and no bias, for a total of 12,960 parameters. We train the networks using SGD, without momentum or weight decay (W​D{WD}), for 250 epochs using cross-entropy loss. We vary the learning rate (lr, η\eta) and batch size (bb) to define a 5×55\times 5 grid of hyperparameters. For each (η\eta, bb) pair, we estimate the expectation in our Sharpness Dimension dimS𝒜S\dim\_{\mathrm{S}}\mathcal{A}\_{S} (SD) by sampling random minibatches, computing parameter Hessians, and taking the SVD of the I−η​∇2ℛ^S​(w)I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w) matrices. RDS sharpness values λk\lambda\_{k} are then estimated as the average of the log SVs and SD is computed following Definition [4](#S4.SS0.SSS0.Px4 "Complexity meassure of Random Attractors ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability").

For our SLQ-based eigenvalue density estimator (SD-SLQ), we use 500 Lanczos runs of 100 steps each via random Rademacher initializations and full reorthogonalization. The expectation is estimated using a separate minibatch per Lanczos run, followed by Gaussian quadrature and kernel smoothing. We then compute SD by integrating the estimated spectral density and the eigenvalue-weighted spectral density using the trapezoidal rule.

!(/html/2604.19740/assets/x2.png)

Figure 2: 
Correlations between various generalization indices and the empirical generalization gap on our small 3-layer MLP trained on MNIST dataset. The region indicated in green shows that our proposed Sharpness Dimension (SD) better predicts the generalization and loss gaps.

Fig. [2](#S5.F2 "Figure 2 ‣ Generalization in 3-layer small MLPs ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability") shows correlations between various generalization measures and the empirical generalization and loss gaps. Both the Euclidean and data-dependent PH-Dims fail to capture the generalization behaviour. In addition, both Euclidean and data-dependent EαE\_{\alpha} show only weak positive correlation with the generalization gap. These results suggest that the trajectory-based topological indices may fail to accurately quantify generalization when networks operate at the edge of stability regime, where I−η​∇2ℛ^S​(w)I-\eta\nabla^{2}\widehat{\mathcal{R}}\_{S}(w) has singular values larger than 1, i.e., not contractive on average. The top RDS-Sharpness λ1\lambda\_{1} alone also shows a weak negative correlation, suggesting that the largest SV alone is not indicative of generalization in this setting. In contrast, both SD and the approximate SD-SLQ show higher correlations with the gap. The full Hessian-spectrum SD achieves the strongest result indicating that SD better quantifies generalization in this edge of stability regime.

!(/html/2604.19740/assets/x3.png)

Figure 3: Kendall coefficients and their average granulated variant for the same 5-layer MLP trained using various batch sizes and learning rates. Our estimate further uses the SLQ approximation and its density estimating variant, SD-KDE.

Generalization in larger 5-layer MLPs.
To evaluate performance on larger networks, we consider a 5-layer MLP on MNIST with a width of 200, corresponding to 278,800 parameters. We follow the same training setup as the 3-layer MLP above and use the same 5×55\times 5 (η,b)(\eta,b) hyperparameter grid, with networks trained for 100 epochs.
Here, we use the same SLQ setup to estimate eigenvalue densities, with the number of Lanczos runs varying with minibatch size, corresponding to two training epochs. We compute both the SLQ-based histogram and the Gaussian kernel-smoothed estimate, denoted as SD-SLQ and SD-KDE, respectively. We cannot use the exact SVD, and hence do not report SD.

For various complexity measures, Fig. [3](#S5.F3 "Figure 3 ‣ Generalization in 3-layer small MLPs ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability") indicates that backward PH-Dim, forward EαE\_{\alpha}’s, our SD-SLQ and SD-KDE achieve similar average GKC values, with backward PH-Dim performing best and SD-SLQ following closely. Yet, when comparing Kendall τ\tau values, SD-SLQ outperforms all other indicators. Overall, when learning rates and batch sizes are grouped separately, the indicators perform similarly. However, when the full hyperparameter grid is considered, our dimension most accurately captures generalization.

!(/html/2604.19740/assets/x4.png)

Figure 4: Grokking analysis for different learning rates (η\eta), weight decay (W​DWD) and seeds across two architectures: (top row) 2-layer MLP. (second row) 2-layer MLP. Note that the *suddenness*  of the grokking behavior is best captured in the complexity measures we introduce: RDS-Sharpness and Sharpness Dimension (SD).

#### Studying grokking

Next, we study the phenomenon of *grokking*, delayed and sudden generalization Power et al., ([2022](#bib.bib59)); Nanda et al., ([2023](#bib.bib54)); Prieto et al., ([2025](#bib.bib60)) through our EoS framework. We consider the task of *arithmetic modulo 97*, a family of supervised learning tasks where two one-hot encoded inputs representing integers a,b<pa,b<p are used to predict the target y=a∗bmodpy=a\*b\mod p. ∗\* is a binary operation and pp is a prime. In all our experiments, we use addition as the binary operation. The dataset size is defined as the percentage of the 97297^{2} possible input pairs used for training, with the remainder used for testing as in Nanda et al., ([2023](#bib.bib54)) and Power et al., ([2022](#bib.bib59)). We use a 40%/60% train/test split.

To induce and observe grokking, we use a 2-layer MLP with 32 hidden features and GELU activation, trained using SGD with momentum and WD for 20k epochs. In all evaluations presented in Fig. [4](#S5.F4 "Figure 4 ‣ Generalization in 3-layer small MLPs ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability"), training acc. exceeds 90% and reaches 100% early, while test acc. continues to increase, a.k.a. *grok*.
We report complexity measures for 100 uniformly spaced checkpoints as well as their intra-correlations and correlations with the Gen. Gap in Fig. [4](#S5.F4 "Figure 4 ‣ Generalization in 3-layer small MLPs ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability"). In App. [D](#A4 "Appendix D Additional Results ‣ Generalization at the Edge of Stability"), we also consider a 3-layer MLP with 32 hidden features and ReLU activation, trained with SGD using only WD.

Fig. [4](#S5.F4 "Figure 4 ‣ Generalization in 3-layer small MLPs ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability") shows that correlations our SD achieves with the generalization gap are comparable to or higher than those of EαE\_{\alpha}, tr​(H)\mathrm{tr}(H), and Sharpness, with our other measure, RDS sharpness, performing worse than SD for all the cases but the first. We observe that the Sharpness and tr​(H)\mathrm{tr}(H) are less stable for certain hyperparameter choices, performing worse in one of the configurations. EαE\_{\alpha} shows stable correlations when using the training trajectories, whereas SD performs similarly or better even with only the model weights from the current epoch. These results indicate that SD captures information about training dynamics and the underlying attractor without requiring access to trajectory information like EαE\_{\alpha}. In addition, during the grokking phase transition occurs Rubin et al., ([2024](#bib.bib62)) simultaneously with changes in the structure of the Hessian spectrum.

#### Generalization in modern transformers: the case of GPT-2

To assess whether the same behavior persists at a larger scale, we consider GPT-2 Radford et al., ([2019](#bib.bib61)) (124M parameters) trained from
scratch on WikiText-2 Merity et al., ([2016](#bib.bib51)). We evaluate SGD and SGD with momentum on a 5×55\times 5 hyperparameter grid over learning rate
and effective batch size, and AdamW Loshchilov and Hutter, ([2019](#bib.bib48)) on a 4×3×34\times 3\times 3 hyperparameter grid over learning rate, batch size, and
weight decay. We report the correlations between the resulting complexity measures and the empirical loss gap and 0–1
loss gap.
For this experiment we additionally report SD-PS, an equal-mass pseudo-spectrum estimator computed from the smoothed SLQ density; see App. [C](#A3.SS0.SSS0.Px2 "Pseudo-spectrum SD (SD-PS) ‣ Appendix C Implementation Details and Computational Complexity ‣ Generalization at the Edge of Stability") for details.
Fig. [5](#S5.F5 "Figure 5 ‣ Generalization in modern transformers: the case of GPT-2 ‣ 5.1 Analysis ‣ 5 Empirical Results ‣ Generalization at the Edge of Stability") indicates that the qualitative conclusions remain consistent with our smaller-scale
experiments, with the strongest distinction appearing for AdamW. In particular, classical sharpness does not provide a
reliable indicator of generalization, and under AdamW, it shows a strong negative correlation with the loss gaps. By
contrast, the RDS-based quantities remain informative overall, with the SD variants, especially the smoothed estimates
SD-KDE and SD-PS, showing the most consistent positive correlations across optimizers. PH-Dim and EαE\_{\alpha} also
provide useful signals in some settings, but their behavior is less uniform. Overall, these results suggest that, in this
transformer setting as well, generalization is better captured by SD and related RDS-based quantities than by classical
sharpness alone.

!(/html/2604.19740/assets/x5.png)

Figure 5: Correlation Matrices Corresponding to GPT-2 Trained on WikiText2-Dataset for different learning rates (LR), batch sizes (BS) and weight decay values (WD), across three different optimizers: SGD, SGD with momentum and AdamW.

## 6 Conclusion

Training neural networks at the edge of numerical stability challenges classical generalization theories that assume convergence to a single solution. In this work, we show that this regime is instead governed by the geometry of a random pullback attractor induced by stochastic optimization viewed as a random dynamical system. Building on this, we introduce the Sharpness Dimension (SD), a Lyapunov-inspired spectral complexity measure that captures the effective dimensionality of expanding directions on the attractor. We prove a worst-case generalization bound over the entire attractor, establishing SD as a principled quantity linking chaotic optimization dynamics to generalization. Empirically, SD reliably predicts generalization gaps across optimization regimes, correlates with grokking phase transitions, and remains computable at scale via stochastic Lanczos quadrature.

#### Limitations & future work

Currently, we are required to estimate the full Hessian spectrum, which, despite scalable approximations, remains computationally demanding for very large models.
Future work involves tightening the theory under weaker regularity assumptions, extending it to adaptive optimizers and large-scale architectures such as transformers Zhang et al., ([2024](#bib.bib77)).

#### Acknowledgments

The authors are grateful for the support of the Excellence Strategy of local and state governments.
T. B. was supported by a UKRI Future Leaders Fellowship (MR/Y018818/1). The authors acknowledge support from the UK AI Research Resource (AIRR Isambard AI) through grant 0251-4584-0945-1 - TopoFound. U.Ş. is partially supported by the French government under the management of Agence Nationale de la Recherche as part of the “Investissements d’avenir” program, reference ANR-19-P3IA-0001 (PRAIRIE 3IA Institute). M.T and U.Ş. are partially supported by the European Research Council Starting Grant DYNASTY – 101039676.

## References

* (1)

  Ahn, K., Bubeck, S., Chewi, S., Lee, Y. T., Suarez, F., and Zhang, Y. (2023a).
  Learning threshold neurons via edge of stability.
  Advances in Neural Information Processing Systems, 36:19540–19569.
* (2)

  Ahn, K., Jadbabaie, A., and Sra, S. (2023b).
  How to escape sharp minima with random perturbations.
  arXiv preprint arXiv:2305.15659.
* Ahn et al., (2022)

  Ahn, K., Zhang, J., and Sra, S. (2022).
  Understanding the unstable convergence of gradient descent.
  In International conference on machine learning. PMLR.
* Andreeva et al., (2024)

  Andreeva, R., Dupuis, B., Sarkar, R., Birdal, T., and Simsekli, U. (2024).
  Topological generalization bounds for discrete-time stochastic optimization algorithms.
  Advances in Neural Information Processing Systems, 37.
* Andreyev and Beneventano, (2024)

  Andreyev, A. and Beneventano, P. (2024).
  Edge of stochastic stability: Revisiting the edge of stability for sgd.
  arXiv preprint arXiv:2412.20553.
* Arnold, (2006)

  Arnold, L. (2006).
  Random dynamical systems.
  In Dynamical Systems: Lectures Given at the 2nd Session of the Centro Internazionale Matematico Estivo (CIME) held in Montecatini Terme, Italy, June 13–22, 1994. Springer.
* Arora et al., (2022)

  Arora, S., Li, Z., and Panigrahi, A. (2022).
  Understanding gradient descent on the edge of stability in deep learning.
  In International Conference on Machine Learning, pages 948–1024. PMLR.
* Birdal et al., (2021)

  Birdal, T., Lou, A., Guibas, L. J., and Simsekli, U. (2021).
  Intrinsic dimension, persistent homology and generalization in neural networks.
  Advances in Neural Information Processing Systems, 34:6776–6789.
* Bogachev, (2007)

  Bogachev, V. (2007).
  Measure theory.
  Springer.
* Cai et al., (2026)

  Cai, Y., Huang, H., Wen, H., Liu, D., Ma, Y., and Lyu, K. (2026).
  Does LLM pre-training typically occur at the edge of stability?
  In Workshop on Scientific Methods for Understanding Deep Learning.
* Camuto et al., (2021)

  Camuto, A., Deligiannidis, G., Erdogdu, M. A., Gurbuzbalaban, M., Simsekli, U., and Zhu, L. (2021).
  Fractal structure and generalization properties of stochastic optimization algorithms.
  Advances in neural information processing systems, 34:18774–18788.
* Chaudhari et al., (2019)

  Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., Chayes, J., Sagun, L., and Zecchina, R. (2019).
  Entropy-sgd: Biasing gradient descent into wide valleys.
  Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124018.
* Chemnitz and Engel, (2025)

  Chemnitz, D. and Engel, M. (2025).
  Characterizing dynamical stability of stochastic gradient descent in overparameterized learning.
  Journal of Machine Learning Research, 26(134):1–46.
* Chen and Bruna, (2023)

  Chen, L. and Bruna, J. (2023).
  Beyond the edge of stability via two-step gradient updates.
  In International Conference on Machine Learning, pages 4330–4391. PMLR.
* Clerico et al., (2022)

  Clerico, E., Farghly, T., Deligiannidis, G., Guedj, B., and Doucet, A. (2022).
  Generalisation under gradient descent via deterministic pac-bayes.
  arXiv preprint arXiv:2209.02525.
* Cohen et al., (2021)

  Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar, A. (2021).
  Gradient descent on neural networks typically occurs at the edge of stability.
  arXiv preprint arXiv:2103.00065.
* Crauel et al., (1997)

  Crauel, H., Debussche, A., and Flandoli, F. (1997).
  Random attractors.
  Journal of Dynamics and Differential Equations, 9(2):307–341.
* Crauel and Flandoli, (1994)

  Crauel, H. and Flandoli, F. (1994).
  Attractors for random dynamical systems.
  Probability Theory and Related Fields, 100(3):365–393.
* Damian et al., (2022)

  Damian, A., Nichani, E., and Lee, J. D. (2022).
  Self-stabilization: The implicit bias of gradient descent at the edge of stability.
  arXiv preprint arXiv:2209.15594.
* Ding et al., (2024)

  Ding, L., Drusvyatskiy, D., Fazel, M., and Harchaoui, Z. (2024).
  Flat minima generalize for low-rank matrix recovery.
  Information and Inference: A Journal of the IMA.
* Dinh et al., (2017)

  Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. (2017).
  Sharp minima can generalize for deep nets.
  In International Conference on Machine Learning, pages 1019–1028. PMLR.
* Dupuis et al., (2023)

  Dupuis, B., Deligiannidis, G., and Simsekli, U. (2023).
  Generalization bounds using data-dependent fractal dimensions.
  In International conference on machine learning, pages 8922–8968. PMLR.
* Dupuis et al., (2024)

  Dupuis, B., Viallard, P., Deligiannidis, G., and Simsekli, U. (2024).
  Uniform generalization bounds on data-dependent hypothesis sets via PAC-bayesian theory on random sets.
  Journal of Machine Learning Research, 25(409).
* Feng and Simon, (2022)

  Feng, D.-J. and Simon, K. (2022).
  Dimension estimates for iterated function systems and repellers. part ii.
  Ergodic Theory and Dynamical Systems, 42(11):3357–3392.
* Foret et al., (2020)

  Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. (2020).
  Sharpness-aware minimization for efficiently improving generalization.
  arXiv preprint arXiv:2010.01412.
* Foster et al., (2019)

  Foster, D. J., Greenberg, S., Kale, S., Luo, H., Mohri, M., and Sridharan, K. (2019).
  Hypothesis set stability and generalization.
  Advances in Neural Information Processing Systems, 32.
* Gatmiry et al., (2023)

  Gatmiry, K., Li, Z., Ma, T., Reddi, S., Jegelka, S., and Chuang, C.-Y. (2023).
  What is the inductive bias of flatness regularization? a study of deep matrix factorization models.
  Advances in Neural Information Processing Systems, 36:28040–28052.
* Ghorbani et al., (2019)

  Ghorbani, B., Krishnan, S., and Xiao, Y. (2019).
  An investigation into neural net optimization via hessian eigenvalue density.
  In International Conference on Machine Learning, pages 2232–2241. PMLR.
* Ghosh et al., (2025)

  Ghosh, A., Cong, B., Yokota, R., Ravishankar, S., Wang, R., Tao, M., Khan, M. E., and Möllenhoff, T. (2025).
  Variational learning finds flatter solutions at the edge of stability.
  arXiv preprint arXiv:2506.12903.
* Golub and Welsch, (1969)

  Golub, G. H. and Welsch, J. H. (1969).
  Calculation of gauss quadrature rules.
  Mathematics of computation, 23.
* Haddouche et al., (2024)

  Haddouche, M., Viallard, P., Simsekli, U., and Guedj, B. (2024).
  A pac-bayesian link between generalisation and flat minima.
  arXiv preprint arXiv:2402.08508.
* Halko et al., (2011)

  Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011).
  Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
  SIAM review, 53(2):217–288.
* Hochreiter and Schmidhuber, (1994)

  Hochreiter, S. and Schmidhuber, J. (1994).
  Simplifying neural nets by discovering flat minima.
  Advances in neural information processing systems, 7.
* Hodgkinson et al., (2022)

  Hodgkinson, L., Simsekli, U., Khanna, R., and Mahoney, M. (2022).
  Generalization bounds using lower tail exponents in stochastic optimizers.
  In International Conference on Machine Learning, pages 8774–8795. PMLR.
* Hunt, (1996)

  Hunt, B. R. (1996).
  Maximum local lyapunov dimension bounds the box dimension of chaotic attractors.
  Nonlinearity, 9(4):845.
* Izmailov et al., (2018)

  Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. (2018).
  Averaging weights leads to wider optima and better generalization.
  arXiv preprint arXiv:1803.05407.
* Jastrzebski et al., (2020)

  Jastrzebski, S., Szymczak, M., Fort, S., Arpit, D., Tabor, J., Cho, K., and Geras, K. (2020).
  The break-even point on optimization trajectories of deep neural networks.
  arXiv preprint arXiv:2002.09572.
* (38)

  Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., and Bengio, S. (2019a).
  Fantastic generalization measures and where to find them.
  arXiv preprint arXiv:1912.02178.
* (39)

  Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., and Bengio, S. (2019b).
  Fantastic Generalization Measures and Where to Find Them.
  ICLR 2020.
* Kaddour et al., (2022)

  Kaddour, J., Liu, L., Silva, R., and Kusner, M. J. (2022).
  When do flat minima optimizers work?
  Advances in Neural Information Processing Systems, 35:16577–16595.
* Kaplan and Yorke, (2006)

  Kaplan, J. L. and Yorke, J. A. (2006).
  Chaotic behavior of multidimensional difference equations.
  In Functional Differential Equations and Approximation of Fixed Points: Proceedings, Bonn, July 1978. Springer.
* Kaur et al., (2023)

  Kaur, S., Cohen, J., and Lipton, Z. C. (2023).
  On the maximum hessian eigenvalue and generalization.
  In Proceedings on, pages 51–65. PMLR.
* Kendall, (1938)

  Kendall, M. G. (1938).
  A new reasure of rank correlation.
  Biometrika.
* Keskar et al., (2016)

  Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. (2016).
  On large-batch training for deep learning: Generalization gap and sharp minima.
  arXiv preprint arXiv:1609.04836.
* Lanczos, (1950)

  Lanczos, C. (1950).
  An iteration method for the solution of the eigenvalue problem of linear differential and integral operators.
  Journal of research of the National Bureau of Standards, 45(4):255–282.
* Lin et al., (2016)

  Lin, L., Saad, Y., and Yang, C. (2016).
  Approximating spectral densities of large matrices.
  SIAM review, 58(1):34–65.
* Liu et al., (2023)

  Liu, H., Xie, S. M., Li, Z., and Ma, T. (2023).
  Same pre-training loss, better downstream: Implicit bias matters for language models.
  In International Conference on Machine Learning, pages 22188–22214. PMLR.
* Loshchilov and Hutter, (2019)

  Loshchilov, I. and Hutter, F. (2019).
  Decoupled weight decay regularization.
* Ly and Gong, (2025)

  Ly, A. and Gong, P. (2025).
  Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning.
  Nature Communications, 16(1):3252.
* Ma and Ying, (2021)

  Ma, C. and Ying, L. (2021).
  On linear stability of sgd and input-smoothness of neural networks.
  Advances in Neural Information Processing Systems, 34:16805–16817.
* Merity et al., (2016)

  Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016).
  Pointer sentinel mixture models.
* Molchanov, (2017)

  Molchanov, I. (2017).
  Theory of Random Sets.
  Number 87 in Probability Theory and Stochastic Modeling. Springer, second edition edition.
* Mulayoff and Michaeli, (2020)

  Mulayoff, R. and Michaeli, T. (2020).
  Unique properties of flat minima in deep networks.
  In International conference on machine learning, pages 7108–7118. PMLR.
* Nanda et al., (2023)

  Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023).
  Progress measures for grokking via mechanistic interpretability.
  In The Eleventh International Conference on Learning Representations.
* Neyshabur et al., (2017)

  Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. (2017).
  Exploring generalization in deep learning.
  Advances in neural information processing systems, 30.
* Nguyen et al., (2019)

  Nguyen, T. H., Simsekli, U., Gurbuzbalaban, M., and Richard, G. (2019).
  First exit time analysis of stochastic gradient descent under heavy-tailed gradient noise.
  Advances in neural information processing systems, 32.
* Papyan, (2018)

  Papyan, V. (2018).
  The full spectrum of deepnet hessians at scale: Dynamics with sgd training and sample size.
  arXiv preprint arXiv:1811.07062.
* Posch et al., (1986)

  Posch, H. A., Hoover, W. G., and Vesely, F. J. (1986).
  Canonical dynamics of the nosé oscillator: Stability, order, and chaos.
  Physical review A, 33(6):4253.
* Power et al., (2022)

  Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. (2022).
  Grokking: Generalization beyond overfitting on small algorithmic datasets.
  arXiv preprint arXiv:2201.02177.
* Prieto et al., (2025)

  Prieto, L., Barsbey, M., Mediano, P. A. M., and Birdal, T. (2025).
  Grokking at the edge of numerical stability.
  In The Thirteenth International Conference on Learning Representations.
* Radford et al., (2019)

  Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019).
  Language models are unsupervised multitask learners.
* Rubin et al., (2024)

  Rubin, N., Seroussi, I., and Ringel, Z. (2024).
  Grokking as a first order phase transition in two layer networks.
  In The Twelfth International Conference on Learning Representations.
* Sagun et al., (2017)

  Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., and Bottou, L. (2017).
  Empirical analysis of the hessian of over-parametrized neural networks.
  arXiv preprint arXiv:1706.04454.
* Sasdelli et al., (2021)

  Sasdelli, M., Ajanthan, T., Chin, T.-J., and Carneiro, G. (2021).
  A chaos theory approach to understand neural network optimization.
  In 2021 Digital Image Computing: Techniques and Applications (DICTA), pages 1–10. IEEE.
* Simsekli et al., (2019)

  Simsekli, U., Sagun, L., and Gurbuzbalaban, M. (2019).
  A tail-index analysis of stochastic gradient noise in deep neural networks.
  In International Conference on Machine Learning, pages 5827–5837. PMLR.
* Simsekli et al., (2020)

  Simsekli, U., Sener, O., Deligiannidis, G., and Erdogdu, M. A. (2020).
  Hausdorff dimension, heavy tails, and generalization in neural networks.
  Advances in Neural Information Processing Systems, 33:5138–5151.
* Singh Kalra et al., (2023)

  Singh Kalra, D., He, T., and Barkeshli, M. (2023).
  Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos.
  arXiv e-prints, pages arXiv–2311.
* Tsuzuku et al., (2020)

  Tsuzuku, Y., Sato, I., and Sugiyama, M. (2020).
  Normalized flat minima: Exploring scale invariant definition of flat minima for neural networks using pac-bayesian analysis.
  In International Conference on Machine Learning, pages 9636–9647. PMLR.
* Tuci et al., (2025)

  Tuci, M., Bastian, L., Dupuis, B., Navab, N., Birdal, T., and Şimşekli, U. (2025).
  Mutual information free topological generalization bounds via stability.
  arXiv preprint arXiv:2507.06775.
* Van Erven and Harremos, (2014)

  Van Erven, T. and Harremos, P. (2014).
  Rényi divergence and kullback-leibler divergence.
  IEEE Transactions on Information Theory, 60(7):3797–3820.
* Wang et al., (2022)

  Wang, Z., Li, Z., and Li, J. (2022).
  Analyzing sharpness along gd trajectory: Progressive sharpening and edge of stability.
  Advances in Neural Information Processing Systems, 35:9983–9994.
* Wen et al., (2023)

  Wen, K., Li, Z., and Ma, T. (2023).
  Sharpness minimization algorithms do not only minimize sharpness to achieve better generalization.
  Advances in Neural Information Processing Systems, 36:1024–1035.
* Wu et al., (2020)

  Wu, D., Xia, S.-T., and Wang, Y. (2020).
  Adversarial weight perturbation helps robust generalization.
  Advances in neural information processing systems, 33:2958–2969.
* Xing et al., (2018)

  Xing, C., Arpit, D., Tsirigotis, C., and Bengio, Y. (2018).
  A walk with sgd.
  arXiv preprint arXiv:1802.08770.
* Yao et al., (2018)

  Yao, Z., Gholami, A., Lei, Q., Keutzer, K., and Mahoney, M. W. (2018).
  Hessian-based analysis of large batch training and robustness to adversaries.
  Advances in Neural Information Processing Systems, 31.
* Yunis, (2017)

  Yunis, D. (2017).
  The birkhoff ergodic theorem with applications.
  The University of Chicago. Disponıvel em.
* Zhang et al., (2024)

  Zhang, Y., Chen, C., Ding, T., Li, Z., Sun, R., and Luo, Z. (2024).
  Why transformers need adam: A hessian perspective.
  Advances in neural information processing systems, 37:131786–131823.
* Zheng et al., (2021)

  Zheng, Y., Zhang, R., and Mao, Y. (2021).
  Regularizing neural networks via adversarial model perturbation.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8156–8165.
* Zhu et al., (2022)

  Zhu, X., Wang, Z., Wang, X., Zhou, M., and Ge, R. (2022).
  Understanding edge-of-stability training dynamics with a minimalist example.
  arXiv preprint arXiv:2210.03294.

## Appendix

We provide here the technical details and full proofs omitted from the main paper, along with supplementary experiments that further support our findings. The appendix is structured as follows:

* •

  Appendix A introduces additional background material from random dynamical systems, random attractors, and worst-case generalization bounds.
* •

  Appendix B presents the complete proofs of all theoretical results stated in the main text, as well as several supplementary theoretical developments.
* •

  Appendix C describes the experimental setups and implementation details required to reproduce our empirical results.
* •

  Appendix D We provide additional empirical results.

## Appendix A Theoretical Background

This section provides the theorems and technical background necessary for our main results.

### A.1 Random Attractor

The notion of an *attractor* is one of the fundamental concepts in dynamical systems theory. While this concept has been central in the study of deterministic systems for several decades, its extension to stochastic systems is more subtle.

In stochastic systems (like SGD), the "noise" never stops kicking the particle. Therefore, the attractor is likely not to be a single static point. Instead, it is a *random set*, a moving target that fluctuates in shape and position over time, driven by the specific realization of the noise. The theory of attractors has been thoroughly developed over the past decades (Crauel and Flandoli,, [1994](#bib.bib18); Arnold,, [2006](#bib.bib6)). Unlike the autonomous deterministic case, Random Dynamical Systems (RDS) are inherently non-autonomous and exhibit multiple notions of attraction, such as pullback and forward attraction. All definitions rely on the concept of a compact, invariant, random set.

###### Definition A.1 (Compact and Invariant Random Sets).

Let ϕ\phi be an RDS on a Polish space XX. A mapping A:Ω→𝒫​(X)A:\Omega\to\mathcal{P}(X) is called a compact random set if:

1. 1.

   Measurability: The indicator function ω↦𝟏A​(ω)​(x)\omega\mapsto\mathbf{1}\_{A(\omega)}(x) is measurable for each fixed x∈Xx\in X.
2. 2.

   Compactness: A​(ω)A(\omega) is compact (i.e., closed and bounded) for ℙ\mathbb{P}-almost every ω∈Ω\omega\in\Omega.

The set AA is said to be an invariant compact random set if additionally it satisfies ϕ\phi-Invariance:

|  |  |  |
| --- | --- | --- |
|  | ϕ​(t,ω,A​(ω))=A​(θt​ω),∀t∈𝕋+,ℙ​-a.s.\phi(t,\omega,A(\omega))=A(\theta^{t}\omega),\quad\forall t\in\mathbb{T}^{+},\quad\mathbb{P}\text{-a.s.} |  |

###### Definition A.2 (Modes of Random Attraction).

Let (θ,ϕ)(\theta,\phi) be an RDS on a Polish space XX, and let AA be an invariant compact random set according to Dfn. [A.1](#A1.Thmdefinition1 "Definition A.1 (Compact and Invariant Random Sets). ‣ A.1 Random Attractor ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability"). Let 𝒮\mathcal{S} be a collection of bounded subsets of XX (e.g., bounded initializations). We denote dist⁡(E,F):=supx∈Einfy∈Fd​(x,y)\operatorname{dist}(E,F):=\sup\_{x\in E}\inf\_{y\in F}d(x,y). AA is called:

1. 1.

   A Random Pullback Attractor if for all D∈𝒮D\in\mathcal{S}:

   |  |  |  |
   | --- | --- | --- |
   |  | limt→∞dist⁡(ϕ​(t,θ−t​ω)​D,A​(ω))=0a.s.\lim\_{t\to\infty}\operatorname{dist}\left(\phi(t,\theta^{-t}\omega)D,A(\omega)\right)=0\quad\text{a.s.} |  |

   Intuition: This asks: "If I started training infinitely long ago (t→−∞t\to-\infty) with noise θ−t​ω\theta^{-t}\omega, where would I be *right now* (t=0t=0)?"
2. 2.

   A Random Forward Attractor if for all D∈𝒮D\in\mathcal{S}:

   |  |  |  |
   | --- | --- | --- |
   |  | limt→∞dist⁡(ϕ​(t,ω)​D,A​(θt​ω))=0a.s.\lim\_{t\to\infty}\operatorname{dist}\left(\phi(t,\omega)D,A(\theta^{t}\omega)\right)=0\quad\text{a.s.} |  |

   Intuition: This asks: "If I start training *now* (t=0t=0), where will I be in the distant future?"
   SGD Context: This describes the physical trajectory of a specific training run.
3. 3.

   A Weak (Point) Attractor if the convergence holds in probability rather than almost surely, (for singleton sets D={y}D=\{y\}).

Fig. [6](#A1.F6 "Figure 6 ‣ A.1 Random Attractor ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") complements our definitions and results with an example. Indeed in the main write up we focus on the notion of random pullback attractor.

!(/html/2604.19740/assets/x6.png)

Figure 6: (a) Fractal Pullback Attractor at the Edge of Stability.
Visualization of the random snapshot attractor 𝒜​(ζ)\mathcal{A}(\zeta) generated by
Stochastic Gradient Descent (SGD) on a non-linear function
L​(x)=12​‖x‖2−A​∏i=12cos⁡(k​xi)L(x)=\frac{1}{2}\|x\|^{2}-A\prod\_{i=1}^{2}\cos(kx\_{i}) (A=2.0,k=4.0A=2.0,k=4.0).
The system is evolved for T=250T=250 steps with a learning rate η=0.15\eta=0.15
placing the dynamics at the edge of stability. The figure illustrates the
collapse of 6×1056\times 10^{5} particles onto a fractal skeleton under a shared
noise realization ζ={ξt}t=1T\zeta=\{\xi\_{t}\}\_{t=1}^{T}, where ξt∼𝒩​(0,σ2​I)\xi\_{t}\sim\mathcal{N}(0,\sigma^{2}I)
with σ=0.1\sigma=0.1. The intricate filaments emerge from the recursive
stretching and folding of the state space, characteristic of chaotic
synchronization in random dynamical systems (RDS).
(b) 3D chaotic attractor of an RDS. Illustration of a chaotic random pullback attractor using a stochastic generalized Nosé-Hoover system Posch et al., ([1986](#bib.bib58)).

### A.2 Fractal Dimensions

We start by introducing the notion of a covering, which plays a central role throughout this paper and is especially important in the proofs of our main results.

###### Definition A.3 (Covering of a set).

Let X⊆ℝdX\subseteq\mathbb{R}^{d} be a set. A δ\delta-cover of XX is a family (Ui)i(U\_{i})\_{i} of sets with diameter at most δ\delta such that

|  |  |  |
| --- | --- | --- |
|  | X⊆⋃iUi.X\subseteq\bigcup\_{i}U\_{i}. |  |

If XX is bounded, a δ\delta-cover of minimal cardinality is called a *minimal cover*, and its cardinality is called the *covering number*, denoted by 𝒩​(X,δ)\mathcal{N}(X,\delta).

Based on Dfn. [A.3](#A1.Thmdefinition3 "Definition A.3 (Covering of a set). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability"), we are now able to define the upper Minkowski dimension, which will later be bounded using our new notion of fractal dimension.

###### Definition A.4 (Minkowski dimensions).

Let X⊂ℝdX\subset\mathbb{R}^{d} be a bounded set. The lower and upper Minkowski (or box) dimensions of XX are defined as

|  |  |  |
| --- | --- | --- |
|  | dim¯B(X):=lim infδ→0log⁡𝒩​(X,δ)log⁡(1/δ),dim¯B(X):=lim supδ→0log⁡𝒩​(X,δ)log⁡(1/δ).\underline{\dim}\_{\mathrm{B}}(X):=\liminf\_{\delta\to 0}\frac{\log\mathcal{N}(X,\delta)}{\log(1/\delta)},\qquad\overline{\dim}\_{\mathrm{B}}(X):=\limsup\_{\delta\to 0}\frac{\log\mathcal{N}(X,\delta)}{\log(1/\delta)}. |  |

If the two quantities agree, the resulting limit is called the *Minkowski dimension* of XX, written dimB(X)\dim\_{\mathrm{B}}(X).

### A.3 Data-dependent worst-case generalization bounds

Before presenting the proofs of our main results, we state the following technical assumption, lemmata and preperation which are necessray to formulate our main result.

Recent advances in topological generalization bounds rely on data-dependent worst-case generalization bounds, leveraging PAC-Bayesian theory on random sets Dupuis et al., ([2024](#bib.bib23)) or the stability-based framework of Tuci et al., ([2025](#bib.bib69)).

###### Theorem A.1 (Egoroff’s theorem).

Let (Ω,ℱ,ℙ)(\Omega,\mathcal{F},\mathbb{P}) be a probability space, and let ff and (fn)n∈ℕ(f\_{n})\_{n\in\mathbb{N}} be measurable functions on Ω\Omega. Suppose that

|  |  |  |
| --- | --- | --- |
|  | fn​(x)⟶f​(x)for almost all ​x∈Ω.f\_{n}(x)\longrightarrow f(x)\quad\text{for almost all }x\in\Omega. |  |

Then, for any ε>0\varepsilon>0, there exists a measurable set Ωε∈ℱ\Omega\_{\varepsilon}\in\mathcal{F} such that

|  |  |  |
| --- | --- | --- |
|  | ℙ​(Ωε)≥1−ε,\mathbb{P}(\Omega\_{\varepsilon})\geq 1-\varepsilon, |  |

and the convergence of fnf\_{n} to ff is uniform on Ωε\Omega\_{\varepsilon}.

This theorem allows us to relate the notion of covering (see Dfn. [A.3](#A1.Thmdefinition3 "Definition A.3 (Covering of a set). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")) to the Minkowski dimension (see Dfn. [A.4](#A1.Thmdefinition4 "Definition A.4 (Minkowski dimensions). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")). Similar ideas have been used in previous work (Simsekli et al.,, [2020](#bib.bib66); Dupuis et al.,, [2023](#bib.bib22), [2024](#bib.bib23)).

###### Corollary A.2 (Uniform control of covering numbers).

Let (ΩU,ℱU,ℙU)(\Omega\_{U},\mathcal{F}\_{U},\mathbb{P}\_{U}) be the probability space supporting the random variable UU, and let (𝒵,ℱ)(\mathcal{Z},\mathcal{F}) denote the data space. Assume that the random set 𝒲S,U\mathcal{W}\_{S,U} is almost surely of finite diameter. Then, for any γ>0\gamma>0, there exists a measurable set Ωγ⊆ℱ⊗n⊗ℱU\Omega\_{\gamma}\subseteq\mathcal{F}^{\otimes n}\otimes\mathcal{F}\_{U} with

|  |  |  |
| --- | --- | --- |
|  | μ⊗n⊗ℙU​(Ωγ)≥1−γ,\mu^{\otimes n}\otimes\mathbb{P}\_{U}(\Omega\_{\gamma})\geq 1-\gamma, |  |

such that, on Ωγ\Omega\_{\gamma}, the following holds: there exists δn,γ>0\delta\_{n,\gamma}>0 for which, for all 0<δ<δn,γ0<\delta<\delta\_{n,\gamma},

|  |  |  |
| --- | --- | --- |
|  | log⁡|𝒩​(𝒲S,U,δ)|≤ 2​dim¯B​(𝒲S,U)​log⁡(1δ).\log\bigl|\mathcal{N}(\mathcal{W}\_{S,U},\delta)\bigr|\;\leq\;2\,\overline{\dim}\_{\mathrm{B}}(\mathcal{W}\_{S,U})\,\log\!\left(\tfrac{1}{\delta}\right). |  |

###### Proof.

Let (ΩU,ℱU,ℙU)(\Omega\_{U},\mathcal{F}\_{U},\mathbb{P}\_{U}) denote the probability space supporting the auxiliary random variable UU, and let ℱ\mathcal{F} denote the σ\sigma-algebra associated with the data space 𝒵\mathcal{Z}. Assume that the random set 𝒲S,U\mathcal{W}\_{S,U} is almost surely of finite diameter.

By definition of the upper Minkowski (box-counting) dimension, we have μ⊗n⊗ℙU\mu^{\otimes n}\otimes\mathbb{P}\_{U}-almost surely

|  |  |  |
| --- | --- | --- |
|  | dimB(𝒲S,U):=lim supδ→0log⁡|𝒩​(𝒲S,U,δ)|log⁡(1/δ)=limδ→0fδ​(𝒲S,U),\dim\_{\mathrm{B}}(\mathcal{W}\_{S,U}):=\limsup\_{\delta\to 0}\frac{\log\bigl|\mathcal{N}(\mathcal{W}\_{S,U},\delta)\bigr|}{\log(1/\delta)}=\lim\_{\delta\to 0}f\_{\delta}(\mathcal{W}\_{S,U}), |  |

where we define

|  |  |  |
| --- | --- | --- |
|  | fδ​(𝒲S,U):=sup0<r<δlog⁡|𝒩​(𝒲S,U,r)|log⁡(1/r).f\_{\delta}(\mathcal{W}\_{S,U}):=\sup\_{0<r<\delta}\frac{\log\bigl|\mathcal{N}(\mathcal{W}\_{S,U},r)\bigr|}{\log(1/r)}. |  |

Let (δk)k≥0(\delta\_{k})\_{k\geq 0} be a decreasing sequence of positive numbers in (0,1)(0,1) such that δk→0\delta\_{k}\to 0, and fix γ>0\gamma>0. By Egoroff’s theorem (see Bogachev,, [2007](#bib.bib9)), there exists a measurable set

|  |  |  |
| --- | --- | --- |
|  | Ωγ∈ℱ⊗n⊗ℱU\Omega\_{\gamma}\in\mathcal{F}^{\otimes n}\otimes\mathcal{F}\_{U} |  |

such that

|  |  |  |
| --- | --- | --- |
|  | μ⊗n⊗ℙU​(Ωγ)≥1−γ,\mu^{\otimes n}\otimes\mathbb{P}\_{U}(\Omega\_{\gamma})\geq 1-\gamma, |  |

and such that the convergence

|  |  |  |
| --- | --- | --- |
|  | fδk​(𝒲S,U)⟶dimB(𝒲S,U)f\_{\delta\_{k}}(\mathcal{W}\_{S,U})\longrightarrow\dim\_{\mathrm{B}}(\mathcal{W}\_{S,U}) |  |

is uniform on Ωγ\Omega\_{\gamma}.

Consequently, there exists an index kγ,n≥0k\_{\gamma,n}\geq 0 such that for all k≥kγ,nk\geq k\_{\gamma,n} and all (S,U)∈Ωγ(S,U)\in\Omega\_{\gamma},

|  |  |  |
| --- | --- | --- |
|  | fδk​(𝒲S,U)≤2​dimB(𝒲S,U).f\_{\delta\_{k}}(\mathcal{W}\_{S,U})\leq 2\,\dim\_{\mathrm{B}}(\mathcal{W}\_{S,U}). |  |

Therefore, for any 0<δ<δk0<\delta<\delta\_{k} and any (S,U)∈Ωγ(S,U)\in\Omega\_{\gamma}, we obtain

|  |  |  |
| --- | --- | --- |
|  | log⁡|𝒩​(𝒲S,U,δ)|≤2​dimB(𝒲S,U)​log⁡(1δ),\log\bigl|\mathcal{N}(\mathcal{W}\_{S,U},\delta)\bigr|\leq 2\,\dim\_{\mathrm{B}}(\mathcal{W}\_{S,U})\,\log\!\left(\tfrac{1}{\delta}\right), |  |

which concludes the proof.
∎

The framework of Dupuis et al., ([2024](#bib.bib23)) is based on information-theoretic quantities. In particular, we provide below a precise definition of the total mutual information term appearing in our main theoretical results; see (Van Erven and Harremos,, [2014](#bib.bib70); Hodgkinson et al.,, [2022](#bib.bib34)) for further background.

###### Definition A.5 (Total mutual information).

Let XX and YY be two random elements defined on a probability space (Ω,ℱ,ℙ)(\Omega,\mathcal{F},\mathbb{P}) (note that the codomains of XX and YY may be distinct). We define the total mutual information between XX and YY by

|  |  |  |
| --- | --- | --- |
|  | I∞​(X,Y)=log​supA∈𝒜X,YℙX,Y​(A)ℙX⊗ℙY​(A).I\_{\infty}(X,Y)=\log\sup\_{A\in\mathcal{A}\_{X,Y}}\frac{\mathbb{P}\_{X,Y}(A)}{\mathbb{P}\_{X}\otimes\mathbb{P}\_{Y}(A)}. |  |

#### Generalization Bounds via Mutual Information

Many studies have employed information-theoretic techniques, particularly within the “fractal-based” literature (Simsekli et al.,, [2020](#bib.bib66); Birdal et al.,, [2021](#bib.bib8)). A unifying perspective recently emerged with the PAC-Bayesian theory for random sets (Dupuis et al.,, [2024](#bib.bib23)), which was recently employed by Andreeva et al., ([2024](#bib.bib4)) to establish generalization bounds based on novel topological complexity measures. Informally, all these bounds are of the following form777We use the notation ≲\lesssim in informal statements where absolute constants have been omitted.:

|  |  |  |  |
| --- | --- | --- | --- |
|  | supw∈𝒲S,U(ℛ​(w)−ℛ^S​(w))≲𝐂​(𝒲S,U)+IT+log⁡(1/ζ)n,\displaystyle{\textstyle\sup\_{w\in\mathcal{W}\_{S,U}}}\big(\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w)\big)\lesssim\sqrt{\frac{\mathbf{C}(\mathcal{W}\_{S,U})+\mathrm{IT}+\log(1/\zeta)}{n}}, |  | (14) |

with probability at least 1−ζ1-\zeta. The term IT\mathrm{IT} is an *information-theoretic* (IT) term, typically the *total* mutual information between the dataset SS and the set 𝒲S,U\mathcal{W}\_{S,U}
The aforementioned bounds differ in the choice of complexity measure 𝐂​(𝒲S,U)\mathbf{C}(\mathcal{W}\_{S,U}), but all include an IT term.

By adapting Corollary 33 of Dupuis et al., ([2024](#bib.bib23)) to our framework, we derive the following Corollary

###### Corollary A.3.

Assume that ℓ​(w,z)\ell(w,z) is LL-Lipschitz in ww, bounded, and that 𝒲S,U\mathcal{W}\_{S,U} is almost surely bounded. Then there exists a constant C>0C>0 such that, for any λ,δ>0\lambda,\delta>0, with probability at least 1−ζ1-\zeta under the joint law of (S,U), we have

|  |  |  |
| --- | --- | --- |
|  | supw∈𝒲S,U(ℛ​(w)−ℛ^S​(w))≤2​L​δ+2​B​2​log⁡|𝒩​(𝒲S,U,δ)|n+I∞​(𝒲S,U,S)+log⁡1ζλ+C​λ​B2n\sup\_{w\in\mathcal{W}\_{S,U}}\left(\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w)\right)\leq 2L\delta+2B\sqrt{\frac{2\log|\mathcal{N}(\mathcal{W}\_{S,U},\delta)|}{n}}+\frac{I\_{\infty}(\mathcal{W}\_{S,U},S)+\log\frac{1}{\zeta}}{\lambda}+C\lambda\frac{B^{2}}{n} |  |

By combining Cor. [A.2](#A1.Thmtheorem2 "Corollary A.2 (Uniform control of covering numbers). ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") and Cor. [A.3](#A1.Thmtheorem3 "Corollary A.3. ‣ Generalization Bounds via Mutual Information ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") and applying a union bound, we can now state the following theorem, which serves as a key foundation for our subsequent analysis.

###### Corollary A.4.

Assume that the loss ℓ​(w,z)\ell(w,z) is LL-Lipschitz in ww, bounded by BB,
and that the random set 𝒲S,U\mathcal{W}\_{S,U} is almost surely of finite
diameter. Then there exists a constant C>0C>0 such that, for any
λ>0\lambda>0, with probability at least 1−ζ−γ1-\zeta-\gamma under the
joint law of (S,U)(S,U), there exists δn,γ>0\delta\_{n,\gamma}>0 such that
for all 0<δ<δn,γ0<\delta<\delta\_{n,\gamma},

|  |  |  |  |
| --- | --- | --- | --- |
|  | supw∈𝒲S,U(ℛ​(w)−ℛ^S​(w))≤2​L​δ+2​B​4​dim¯B​(𝒲S,U)​log⁡(1/δ)n+I∞​(𝒲S,U,S)+log⁡(1/ζ)λ+C​λ​B2n.\sup\_{w\in\mathcal{W}\_{S,U}}\bigl(\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w)\bigr)\leq 2L\delta+2B\sqrt{\frac{4\,\overline{\dim}\_{\mathrm{B}}(\mathcal{W}\_{S,U})\,\log(1/\delta)}{n}}\\ +\frac{I\_{\infty}(\mathcal{W}\_{S,U},S)+\log(1/\zeta)}{\lambda}+\frac{C\lambda B^{2}}{n}. |  | (15) |

###### Remark A.1.

The parameter δn\delta\_{n} appearing in [Theorem˜A.6](#A1.Thmtheorem6 "Theorem A.6 (Tuci et al., (2025) Theorem 4.3.). ‣ Generalization Bounds via Random Set Stability ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")
deserves a brief comment. As can be seen from the proof, it
quantifies the uniformity in nn of the limit defining the upper
box-counting dimension: if this convergence is uniform in nn, then
δ\delta becomes independent of nn. A similar parameter already
arises in (Dupuis et al.,, [2023](#bib.bib22)), and
(Dupuis et al.,, [2024](#bib.bib23)) shows that δ\delta can indeed be taken
independent of nn under a suitable convergence assumption on the
distributions of the random sets. We refer the reader to these
works for further details.

#### Generalization Bounds via Random Set Stability

A different perspective was taken by Tuci et al., ([2025](#bib.bib69)), who showed that various ’topological generalization’ bounds can be recovered within a novel stability framework. In our case, we make use of both frameworks. Before stating our main bound in terms of the stability parameter, we recall the following assumption on the random set, denoted by 𝒲S,U\mathcal{W}\_{S,U}, which, in our setting, is the random attractor.

###### Definition A.6.

A data-dependent selection of 𝒲S,U\mathcal{W}\_{S,U} is a deterministic mapping
ω:CL​(ℝd)×𝒵n→ℝd\omega:\mathrm{CL}(\mathbb{R}^{d})\times\mathcal{Z}^{n}\to\mathbb{R}^{d} such that ω​(𝒲S,U,S′)∈𝒲S,U\omega(\mathcal{W}\_{S,U},S^{\prime})\in\mathcal{W}\_{S,U}, almost surely. In particular, we assume the existence of a random variable ω0​(𝒲S,U,S′)\omega\_{0}(\mathcal{W}\_{S,U},S^{\prime}) such that, almost surely, ω0​(𝒲S,U,S′)∈arg​maxw∈𝒲S,U⁡GS′​(w)\omega\_{0}(\mathcal{W}\_{S,U},S^{\prime})\in\operatorname\*{arg\,max}\_{w\in\mathcal{W}\_{S,U}}G\_{S^{\prime}}(w).

###### Assumption A.5 (Random set stability by Tuci et al., ([2025](#bib.bib69))).

𝒲S,U\mathcal{W}\_{S,U} is βn\beta\_{n}-random set stable if for any J∈ℕ⋆J\in\mathbb{N}^{\star} and any data-dependent selection ω\omega of 𝒲S,U\mathcal{W}\_{S,U}, there exists a map ω′:CL​(ℝd)×ℝd→ℝd\omega^{\prime}:\mathrm{CL}(\mathbb{R}^{d})\times\mathbb{R}^{d}\to\mathbb{R}^{d} such that:

* •

  For any SS, UU and w∈ℝdw\in\mathbb{R}^{d}, ω′​(𝒲S,U,w)∈𝒲S,U\omega^{\prime}(\mathcal{W}\_{S,U},w)\in\mathcal{W}\_{S,U} .
* •

  For all z∈𝒵z\in\mathcal{Z} and two datasets S,S′∈𝒵nS,S^{\prime}\in\mathcal{Z}^{n} differing by JJ elements we have:

  |  |  |  |
  | --- | --- | --- |
  |  | 𝔼U​[|ℓ​(ω​(𝒲S,U,S),z)−ℓ​(ω′​(𝒲S′,U,ω​(𝒲S,U,S)),z)|]≤βn​J.\displaystyle\mathbb{E}\_{U}[|\ell(\omega(\mathcal{W}\_{S,U},S),z)-\ell(\omega^{\prime}(\mathcal{W}\_{S^{\prime},U},\omega(\mathcal{W}\_{S,U},S)),z)|]\leq\beta\_{n}J. |  |

As noted by Tuci et al., ([2025](#bib.bib69)), in the absence of algorithmic randomness (that is, when UU is constant), Assump. [A.5](#A1.Thmtheorem5 "Assumption A.5 (Random set stability by Tuci et al., (2025)). ‣ Generalization Bounds via Random Set Stability ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") reduces to a special case of the celebrated random set stability notion introduced by Foster et al., ([2019](#bib.bib26)). Stability assumptions are typically formulated for neighboring datasets; here, we instead consider a variant in which datasets differ by JJ elements. The two formulations are equivalent, and we adopt this version to streamline subsequent proofs and simplify notation.

###### Theorem A.6 (Tuci et al., ([2025](#bib.bib69)) Theorem 4.3.).

Suppose that the loss ℓ\ell is bounded by BB, LL-Lipschitz and Assump. [A.5](#A1.Thmtheorem5 "Assumption A.5 (Random set stability by Tuci et al., (2025)). ‣ Generalization Bounds via Random Set Stability ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") hold, and that 𝒲S,U\mathcal{W}\_{S,U} is a.s. of finite diameter. Without loss of generality, assume that βn−2/3\beta\_{n}^{-2/3} is an integer divisor of nn. There exists δn>0\delta\_{n}>0 such that for all δ<δn\delta<\delta\_{n}

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[supw∈𝒲S,U(ℛ​(w)−ℛ^S​(w))]≤2​𝔼​[Bn+δ​L+βn1/3​(1+B​4​dim¯B​(𝒲S,U)​log⁡1δ)],\displaystyle\mathbb{E}\bigg[\sup\_{w\in\mathcal{W}\_{S,U}}\left(\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w)\right)\bigg]\leq 2\mathbb{E}\bigg[\frac{B}{n}+\delta L+\beta\_{n}^{1/3}\left(1+B\sqrt{4\overline{\dim}\_{\mathrm{B}}(\mathcal{W}\_{S,U})\log\frac{1}{\delta}}\right)\bigg], |  |

where dim¯B​(𝒲S,U)\overline{\dim}\_{\mathrm{B}}(\mathcal{W}\_{S,U}) is the upper box-counting dimension (see Dfn. [A.4](#A1.Thmdefinition4 "Definition A.4 (Minkowski dimensions). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability"))

Indeed, we present a simplified version of the theorem here. Since we
assume that the loss ℓ\ell is globally LL-Lipschitz, this constant
appears directly in the bound. It would in fact suffice to assume
Lipschitz continuity only on the random set itself.

We will next provide the proofs omitted from the main paper.

## Appendix B Omitted Proofs

Before we present our main theorem. We recall some definition and basic facts, which will be used in the proof and improve readability.

### B.1 Preliminaries

### Important Sets

###### Definition B.1 (Minkowski Sum).

Let UU and VV be two non-empty subsets of a vector space (in this context, ℝd\mathbb{R}^{d}). The Minkowski sum, denoted by U⊕VU\oplus V, is the set formed by adding every element of UU to every element of VV:

|  |  |  |
| --- | --- | --- |
|  | U⊕V:={u+v∣u∈U,v∈V}U\oplus V:=\{u+v\mid u\in U,v\in V\} |  |

In the context of the proof, the Minkowski sum provides a rigorous way to define the “thickening” of a transformed set to account for approximation errors. Let U⊂ℝdU\subset\mathbb{R}^{d} be a compact set and let B​(0,ϵ)B(0,\epsilon) be the closed ball of radius ϵ>0\epsilon>0 centered at the origin. The Minkowski sum U⊕B​(0,ϵ)U\oplus B(0,\epsilon) corresponds exactly to the closed ϵ\epsilon-neighborhood of UU:

|  |  |  |  |
| --- | --- | --- | --- |
|  | U⊕B​(0,ϵ)={y∈ℝd∣∃u∈U,‖y−u‖≤ϵ}U\oplus B(0,\epsilon)=\{y\in\mathbb{R}^{d}\mid\exists u\in U,\|y-u\|\leq\epsilon\} |  | (16) |

Equivalently, this can be expressed using the distance function dist​(y,U)=infu∈U‖y−u‖\text{dist}(y,U)=\inf\_{u\in U}\|y-u\|:

|  |  |  |  |
| --- | --- | --- | --- |
|  | U⊕B​(0,ϵ)={y∈ℝd∣dist​(y,U)≤ϵ}U\oplus B(0,\epsilon)=\{y\in\mathbb{R}^{d}\mid\text{dist}(y,U)\leq\epsilon\} |  | (17) |

### Geometric Interpretation: Ellipsoids and Linear Images of Balls

Let L∈ℝd×dL\in\mathbb{R}^{d\times d} be a real matrix of rank

|  |  |  |
| --- | --- | --- |
|  | k:=rank​(L)≤d.k:=\mathrm{rank}(L)\leq d. |  |

We study the geometry of the image of the unit ball

|  |  |  |
| --- | --- | --- |
|  | B​(0,1):={ξ∈ℝd:‖ξ‖2≤1}B(0,1):=\{\xi\in\mathbb{R}^{d}:\|\xi\|\_{2}\leq 1\} |  |

under the linear map LL.

Define

|  |  |  |
| --- | --- | --- |
|  | E:=L​(B​(0,1))={L​ξ∈ℝd:‖ξ‖2≤1}.E:=L(B(0,1))=\{L\xi\in\mathbb{R}^{d}:\|\xi\|\_{2}\leq 1\}. |  |

Then EE is an ellipsoid if LL is invertible, and a *degenerate ellipsoid* (i.e. a flattened ellipsoid lying in a lower-dimensional subspace) if rank​(L)<d\mathrm{rank}(L)<d.

#### Canonical Form of the Ellipsoid

By the Singular Value Decomposition, there exist orthonormal bases

|  |  |  |
| --- | --- | --- |
|  | {u1,…,ud}⊂ℝd,{v1,…,vd}⊂ℝd\{u\_{1},\dots,u\_{d}\}\subset\mathbb{R}^{d},\qquad\{v\_{1},\dots,v\_{d}\}\subset\mathbb{R}^{d} |  |

and singular values

|  |  |  |
| --- | --- | --- |
|  | σ1≥σ2≥⋯≥σk>0,σk+1=⋯=σd=0\sigma\_{1}\geq\sigma\_{2}\geq\cdots\geq\sigma\_{k}>0,\qquad\sigma\_{k+1}=\cdots=\sigma\_{d}=0 |  |

such that

|  |  |  |
| --- | --- | --- |
|  | L=∑i=1kσi​vi​uiT.L=\sum\_{i=1}^{k}\sigma\_{i}\,v\_{i}u\_{i}^{T}. |  |

Equivalently,

|  |  |  |
| --- | --- | --- |
|  | L​ui=σi​vifor ​i=1,…,k,L​ui=0for ​i=k+1,…,d.Lu\_{i}=\sigma\_{i}v\_{i}\quad\text{for }i=1,\dots,k,\qquad Lu\_{i}=0\quad\text{for }i=k+1,\dots,d. |  |

The singular values satisfy

|  |  |  |
| --- | --- | --- |
|  | σi=λi​(LT​L),\sigma\_{i}=\sqrt{\lambda\_{i}(L^{T}L)}, |  |

where λi​(LT​L)\lambda\_{i}(L^{T}L) are the eigenvalues of LT​LL^{T}L ordered in decreasing order. Since {u1,…​ud}\{u\_{1},\dots u\_{d}\} an orthonormal basis we rewrite ξ∈ℝd\xi\in\mathbb{R}^{d} with ‖ξ‖≤1\|\xi\|\leq 1 as

|  |  |  |
| --- | --- | --- |
|  | ξ=∑i=1dai​ui,∑i=1dai2≤1.\xi=\sum\_{i=1}^{d}a\_{i}u\_{i},\qquad\sum\_{i=1}^{d}a\_{i}^{2}\leq 1. |  |

Then

|  |  |  |
| --- | --- | --- |
|  | L​ξ=∑i=1kσi​ai​vi.L\xi=\sum\_{i=1}^{k}\sigma\_{i}a\_{i}v\_{i}. |  |

Therefore for ti:=σi​ait\_{i}:=\sigma\_{i}a\_{i},

|  |  |  |
| --- | --- | --- |
|  | E={∑i=1kti​vi|∑i=1kti2σi2≤1}.E=\left\{\sum\_{i=1}^{k}t\_{i}v\_{i}\;\Bigg|\;\sum\_{i=1}^{k}\frac{t\_{i}^{2}}{\sigma\_{i}^{2}}\leq 1\right\}. |  |

Geometrically, EE lies in the kk-dimensional subspace

|  |  |  |
| --- | --- | --- |
|  | Im​(L)=span​{v1,…,vk},\mathrm{Im}(L)=\mathrm{span}\{v\_{1},\dots,v\_{k}\}, |  |

its principal axes are the directions v1,…,vkv\_{1},\dots,v\_{k}, and the length of the ii-th semi-axis is σi\sigma\_{i}.

#### Image of a Ball of Radius ρ\rho

Let ρ>0\rho>0. Since

|  |  |  |
| --- | --- | --- |
|  | B​(0,ρ)=ρ​B​(0,1),B(0,\rho)=\rho\,B(0,1), |  |

by linearity we have

|  |  |  |
| --- | --- | --- |
|  | L​(B​(0,ρ))=ρ​L​(B​(0,1))=ρ​E.L(B(0,\rho))=\rho\,L(B(0,1))=\rho\,E. |  |

Thus the scaled ellipsoid is

|  |  |  |
| --- | --- | --- |
|  | ρ​E={∑i=1kti​vi|∑i=1kti2(ρ​σi)2≤1},\rho E=\left\{\sum\_{i=1}^{k}t\_{i}v\_{i}\;\Bigg|\;\sum\_{i=1}^{k}\frac{t\_{i}^{2}}{(\rho\sigma\_{i})^{2}}\leq 1\right\}, |  |

with principal semi-axis lengths

|  |  |  |
| --- | --- | --- |
|  | ρ​σ1,ρ​σ2,…,ρ​σk.\rho\sigma\_{1},\;\rho\sigma\_{2},\;\dots,\;\rho\sigma\_{k}. |  |

#### Image of a Shifted Ball.

For any x∈ℝdx\in\mathbb{R}^{d},

|  |  |  |
| --- | --- | --- |
|  | B​(x,ρ)={x}⊕B​(0,ρ),B(x,\rho)=\{x\}\oplus B(0,\rho), |  |

and therefore

|  |  |  |
| --- | --- | --- |
|  | L​(B​(x,ρ))={L​x}⊕ρ​E.L(B(x,\rho))=\{Lx\}\oplus\rho E. |  |

#### Covering Estimates for Ellipsoid

###### Lemma B.1 (Ellipsoid covering).

Let E⊂ℝdE\subset\mathbb{R}^{d} be an ellipsoid with semi-axes

|  |  |  |
| --- | --- | --- |
|  | σ1≥σ2≥⋯≥σd>0.\sigma\_{1}\geq\sigma\_{2}\geq\cdots\geq\sigma\_{d}>0. |  |

Let ρ>0\rho>0, and let j∈{0,1,…,d}j\in\{0,1,\dots,d\} be such that

|  |  |  |
| --- | --- | --- |
|  | σj+1≤ρ(with ​σd+1:=0).\sigma\_{j+1}\leq\rho\quad(\text{with }\sigma\_{d+1}:=0). |  |

Then the minimal number of Euclidean balls of radius ρ\rho needed to cover EE satisfies

|  |  |  |
| --- | --- | --- |
|  | 𝒩​(E,ρ)≤ 3d​∏i=1jσiρ.\mathcal{N}(E,\rho)\;\leq\;3^{d}\prod\_{i=1}^{j}\frac{\sigma\_{i}}{\rho}. |  |

###### Proof.

Up to translation and rotation, we may assume that

|  |  |  |
| --- | --- | --- |
|  | E={x∈ℝd:∑i=1dxi2σi2≤1}.E=\left\{x\in\mathbb{R}^{d}:\sum\_{i=1}^{d}\frac{x\_{i}^{2}}{\sigma\_{i}^{2}}\leq 1\right\}. |  |

Then EE is contained in the axis-aligned box

|  |  |  |
| --- | --- | --- |
|  | Q:=∏i=1d[−σi,σi].Q:=\prod\_{i=1}^{d}[-\sigma\_{i},\sigma\_{i}]. |  |

Hence

|  |  |  |
| --- | --- | --- |
|  | 𝒩​(E,ρ)≤𝒩​(Q,ρ).\mathcal{N}(E,\rho)\leq\mathcal{N}(Q,\rho). |  |

We cover QQ by a grid of cubes of side length ρ\rho. Along the ii-th coordinate direction, the interval [−σi,σi][-\sigma\_{i},\sigma\_{i}] can be covered by at most

|  |  |  |
| --- | --- | --- |
|  | ⌈2​σiρ⌉≤ 1+2​σiρ\left\lceil\frac{2\sigma\_{i}}{\rho}\right\rceil\;\leq\;1+\frac{2\sigma\_{i}}{\rho} |  |

intervals of length ρ\rho. Therefore,

|  |  |  |
| --- | --- | --- |
|  | 𝒩​(Q,ρ)≤∏i=1d(1+2​σiρ).\mathcal{N}(Q,\rho)\leq\prod\_{i=1}^{d}\left(1+\frac{2\sigma\_{i}}{\rho}\right). |  |

Now assume that σj+1≤ρ\sigma\_{j+1}\leq\rho.

Step 1: Small axes.
For i≥j+1i\geq j+1, since σi≤ρ\sigma\_{i}\leq\rho, we have

|  |  |  |
| --- | --- | --- |
|  | 1+2​σiρ≤1+2=3.1+\frac{2\sigma\_{i}}{\rho}\leq 1+2=3. |  |

Step 2: Large axes.
For i≤ji\leq j, since σi≥ρ\sigma\_{i}\geq\rho, we have

|  |  |  |
| --- | --- | --- |
|  | 1+2​σiρ≤3​σiρ.1+\frac{2\sigma\_{i}}{\rho}\;\leq\;\frac{3\sigma\_{i}}{\rho}. |  |

Indeed, this is equivalent to 1≤σi/ρ1\leq\sigma\_{i}/\rho, which holds.

Step 3: Combine.
Therefore,

|  |  |  |
| --- | --- | --- |
|  | 𝒩​(E,ρ)≤∏i=1j3​σiρ⋅∏i=j+1d3= 3d​∏i=1jσiρ.\mathcal{N}(E,\rho)\;\leq\;\prod\_{i=1}^{j}\frac{3\sigma\_{i}}{\rho}\;\cdot\;\prod\_{i=j+1}^{d}3\;=\;3^{d}\prod\_{i=1}^{j}\frac{\sigma\_{i}}{\rho}. |  |

This completes the proof.
∎

#### Covering for Minkowski Sum

###### Lemma B.2 (Covering of Minkowski sums).

Let X,Y⊂ℝdX,Y\subset\mathbb{R}^{d} be bounded sets and let ε,δ>0\varepsilon,\delta>0. Assume that

|  |  |  |
| --- | --- | --- |
|  | X⊂⋃i=1NB​(xi,ε),Y⊂B​(0,δ),X\subset\bigcup\_{i=1}^{N}B(x\_{i},\varepsilon),\qquad Y\subset B(0,\delta), |  |

where N∈ℕN\in\mathbb{N} and xi∈ℝdx\_{i}\in\mathbb{R}^{d} for all i∈{1,…,N}i\in\{1,\dots,N\}.
Then the Minkowski sum A⊕BA\oplus B satisfies

|  |  |  |
| --- | --- | --- |
|  | X⊕Y⊂⋃i=1NB​(xi,ε+δ).X\oplus Y\subset\bigcup\_{i=1}^{N}B(x\_{i},\varepsilon+\delta). |  |

In particular, the covering numbers satisfy

|  |  |  |
| --- | --- | --- |
|  | 𝒩​(X⊕Y,ε+δ)≤𝒩​(X,ε).\mathcal{N}(X\oplus Y,\varepsilon+\delta)\leq\mathcal{N}(X,\varepsilon). |  |

###### Proof.

Let a∈X⊕Ya\in X\oplus Y. Then a=x+ya=x+y with x∈Xx\in X and y∈Yy\in Y. By assumption, there exists i∈{1,…,N}i\in\{1,\dots,N\} such that x∈B​(xi,ε)x\in B(x\_{i},\varepsilon), hence ‖x−xi‖≤ε\|x-x\_{i}\|\leq\varepsilon. Moreover, since y∈B​(0,δ)y\in B(0,\delta), we have ‖y‖≤δ\|y\|\leq\delta. Therefore,

|  |  |  |
| --- | --- | --- |
|  | ‖a−xi‖=‖x+y−xi‖≤‖x−xi‖+‖y‖≤ε+δ,\|a-x\_{i}\|=\|x+y-x\_{i}\|\leq\|x-x\_{i}\|+\|y\|\leq\varepsilon+\delta, |  |

which shows that a∈B​(xi,ε+δ)a\in B(x\_{i},\varepsilon+\delta). This proves the claim.
∎

We will now prove the main result of our work. We begin by bounding the Minkowski dimension (see Dfn. [A.4](#A1.Thmdefinition4 "Definition A.4 (Minkowski dimensions). ‣ A.2 Fractal Dimensions ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability"))

###### Theorem B.3 (Minkowski Dimension Bound).

Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a discrete-time
random dynamical system according to Dfn. [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"), such that for ℙ\mathbb{P}-a.e. ω\omega, the map x↦ϕ​(1,ω,x)x\mapsto\phi(1,\omega,x) is C2C^{2}.
Suppose the following hold:

1. 1.

   Non-Singularity: For ℙ\mathbb{P}-a.e. ω\omega we assume

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | infx∈𝒜​(ω)σd(Dϕ(1,ω,x)>0\displaystyle\inf\_{x\in\mathcal{A}(\omega)}\sigma\_{d}(D\phi(1,\omega,x)>0 |  | (18) |
2. 2.

   Random Invariant Set: There exists a random compact set 𝒜={𝒜​(ω)}ω∈Ω\mathcal{A}=\{\mathcal{A}(\omega)\}\_{\omega\in\Omega} in ℝd\mathbb{R}^{d} that is invariant under the cocycle ϕ\phi, i.e.,

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ϕ​(n,ω,𝒜​(ω))=𝒜​(θn​ω),ℙ​-a.s. for all ​n∈ℕ.\phi(n,\omega,\mathcal{A}(\omega))=\mathcal{A}(\theta^{n}\omega),\quad\mathbb{P}\text{-a.s. for all }n\in\mathbb{N}. |  | (19) |

   The mapping ω↦𝒜​(ω)\omega\mapsto\mathcal{A}(\omega) is measurable in the sense that the distance function ω↦dist​(x,𝒜​(ω))\omega\mapsto\text{dist}(x,\mathcal{A}(\omega)) is a random variable for every x∈ℝdx\in\mathbb{R}^{d}.
3. 3.

   Integrability: The logarithms of the linearized growth and the local curvature are integrable over the attractor. Specifically, we assume:

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | 𝔼​[supx∈𝒜​(ω)ln⁡‖D​ϕ​(1,ω,x)‖]<∞\mathbb{E}\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(1,\omega,x)\|\right]<\infty |  | (20) |

   and

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | 𝔼​[ln​supx∈𝒜​(ω)‖D2​ϕ​(1,ω,x)‖]<∞.\mathbb{E}\left[\ln\sup\_{x\in\mathcal{A}(\omega)}\|D^{2}\phi(1,\omega,x)\|\right]<\infty. |  | (21) |
4. 4.

   Transition Index: There exists an integer j∗∈{1,…,d−1}j^{\*}\in\{1,\dots,d-1\} such that:

   |  |  |  |
   | --- | --- | --- |
   |  | ∑i=1j⁣∗λi≥0and∑i=1j∗+1λi<0.\sum\_{i=1}^{j\*}\lambda\_{i}\geq 0\quad\text{and}\quad\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0. |  |

   .
5. 5.

   Bounded Distortion: For A∈ℝd×dA\in\mathbb{R}^{d\times d} and j∈{1,…,d}j\in\{1,\dots,d\}, define

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ‖A‖j:=σ1​(A)​⋯​σj​(A),\|A\|\_{j}:=\sigma\_{1}(A)\cdots\sigma\_{j}(A), |  | (22) |

   where σ1​(A)≥⋯≥σd​(A)\sigma\_{1}(A)\geq\cdots\geq\sigma\_{d}(A) are the singular values of AA. Equivalently, ‖A‖j\|A\|\_{j} is the maximal expansion factor of AA on jj-dimensional volumes.

   We assume that the spatial variation of ‖D​ϕ​(m,ω,⋅)‖j\|D\phi(m,\omega,\cdot)\|\_{j} over the attractor is subexponential in mm: for each j∈{1,…,d}j\in\{1,\dots,d\},

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | limm→∞1m​𝔼​[supx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j−infx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j]=0.\lim\_{m\to\infty}\frac{1}{m}\,\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\;-\;\inf\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\right]=0. |  | (23) |

   In other words, the maximal and minimal jj-volume growth rates over 𝒜​(ω)\mathcal{A}(\omega) agree at exponential scale.

Define λ1≥λ2≥⋯≥λd\lambda\_{1}\geq\lambda\_{2}\geq\dots\geq\lambda\_{d} be the one-step exponents associated with the maximal expansion on the set 𝒜​(ω)\mathcal{A}(\omega), defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | λk:=𝔼​[supw∈𝒜​(ω)ln⁡σk​(D​ϕ​(1,ω,w))],\displaystyle\lambda\_{k}:=\mathbb{E}\left[\sup\_{w\in\mathcal{A}(\omega)}\ln\sigma\_{k}(D\phi(1,\omega,w))\right], |  | (24) |

where σk​(A)\sigma\_{k}(A) denotes the kk-th singular value of a linear map AA. Note that by the integrability assumption, these values are finite.

Then, for ℙ\mathbb{P}-almost every ω∈Ω\omega\in\Omega, the *upper Minkowski dimension* of the set 𝒜​(ω)\mathcal{A}(\omega) is bounded by the Sharpness Dimension dimS𝒜\dim\_{\mathrm{S}}\mathcal{A}:

|  |  |  |
| --- | --- | --- |
|  | dim¯M​(𝒜​(ω))≤dimS𝒜.\overline{\dim}\_{M}(\mathcal{A}(\omega))\leq\dim\_{\mathrm{S}}\mathcal{A}. |  |

###### Proof.

The main idea behind the proof is the fact that the sets 𝒜​(ω)\mathcal{A}(\omega) and 𝒜​(θK​ω)\mathcal{A}(\theta^{K}\omega) have the same distribution for any KK. Hence, we will estimate the covering number of 𝒜​(θK​ω)\mathcal{A}(\theta^{K}\omega) which will enable us to link the covering number to the singular values of D​ϕD\phi.

#### Step 1: General Covering

Let (Ω,ℱ,ℙ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\phi) be a C2C^{2} random dynamical system. We begin by covering the set 𝒜​(ω)\mathcal{A}(\omega), which is almost surely compact and, therefore, bounded. Hence, for any R>0R>0 we have a finite integer N1​(ω,R)N\_{1}(\omega,R) such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜​(ω)⊂⋃i=1N1​(ω,R)B​(xi​(ω),R),\displaystyle\mathcal{A}(\omega)\subset\bigcup\_{i=1}^{N\_{1}(\omega,R)}B(x\_{i}(\omega),R), |  | (25) |

where {x1​(ω),…,xN1​(ω)}\{x\_{1}(\omega),\dots,x\_{N\_{1}}(\omega)\} denote the centers of a finite covering of 𝒜​(ω)\mathcal{A}(\omega) by balls of radius RR. These centers can be chosen in a measurable way Molchanov, ([2017](#bib.bib52)). We denote the corresponding covering number N1​(ω,R)N\_{1}(\omega,R) simply by N1N\_{1}, or, to make the dependence on ω\omega and RR explicit, by N1​(ω,R)N\_{1}(\omega,R).

Since by assumption 𝒜​(ω)\mathcal{A}(\omega) is ϕ\phi-invariant, we have
𝒜​(θ​ω)=ϕ​(1,ω,𝒜​(ω))\mathcal{A}(\theta\omega)=\phi(1,\omega,\mathcal{A}(\omega)) and thus by ([25](#A2.E25 "Equation 25 ‣ Step 1: General Covering ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability"))

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜​(θ​ω)⊂⋃i=1N1​(ω,R)ϕ​(1,ω,B​(xi,R)).\mathcal{A}(\theta\omega)\subset\bigcup\_{i=1}^{N\_{1}(\omega,R)}\phi(1,\omega,B(x\_{i},R)). |  | (26) |

#### Step 2: Local Approximation

In the second step, we approximate ϕ\phi via a second-order Taylor approximation.

By assumption, for every ω∈Ω\omega\in\Omega the map x↦ϕ​(1,ω,x)x\mapsto\phi(1,\omega,x) is twice continuously differentiable. We define the random variable

|  |  |  |  |
| --- | --- | --- | --- |
|  | C​(ω,R):=12​supx∈𝒜​(ω)R‖D2​ϕ​(1,ω,x)‖,\displaystyle C(\omega,R):=\frac{1}{2}\sup\_{x\in\mathcal{A}(\omega)\_{R}}\bigl\|D^{2}\phi(1,\omega,x)\bigr\|, |  | (27) |

where 𝒜​(ω)R\mathcal{A}(\omega)\_{R} denotes the closed RR-neighborhood of 𝒜​(ω)\mathcal{A}(\omega), cf. ([16](#A2.E16 "Equation 16 ‣ Definition B.1 (Minkowski Sum). ‣ Important Sets ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")).

Since 𝒜​(ω)\mathcal{A}(\omega) is bounded almost surely, its closed RR-neighborhood 𝒜​(ω)R\mathcal{A}(\omega)\_{R} is also compact almost surely. Because D2​ϕ​(1,ω,⋅)D^{2}\phi(1,\omega,\cdot) is continuous, the supremum in the definition of C​(ω,R)C(\omega,R) is finite for almost every ω\omega. Hence,

|  |  |  |
| --- | --- | --- |
|  | C​(ω,R)<∞for almost every ​ω∈Ω.C(\omega,R)<\infty\quad\text{for almost every }\omega\in\Omega. |  |

We now study the image of the random ball B​(xi​(ω),R)B(x\_{i}(\omega),R) under the map ϕ​(1,ω,⋅)\phi(1,\omega,\cdot). As argued above, the random constant C​(ω)C(\omega) is finite only almost surely and not uniformly in ω\omega. Therefore, we fix a set Ω0⊂Ω\Omega\_{0}\subset\Omega with full measure, i.e.

|  |  |  |
| --- | --- | --- |
|  | ℙ​(Ω0)=1,\mathbb{P}(\Omega\_{0})=1, |  |

such that the constant C​(ω,R)C(\omega,R) is finite for all ω∈Ω0\omega\in\Omega\_{0}. In what follows, all arguments are carried out pathwise for ω∈Ω0\omega\in\Omega\_{0}. Fix i∈{1,…,N1}i\in\{1,\dots,N\_{1}\} and we investigate the image of ϕ​(1,ω,B​(xi​(ω),R))\phi(1,\omega,B(x\_{i}(\omega),R)).

Fix ω∈Ω0\omega\in\Omega\_{0}. Let i:=i​(ω)∈{1,…,N1​(ω)}i:=i(\omega)\in\{1,\dots,N\_{1}(\omega)\} (for notational simplicity we suppress the dependence on ω\omega in the index). We consider the image of the ball B​(xi​(ω),R)B(x\_{i}(\omega),R) under ϕ​(1,ω,⋅)\phi(1,\omega,\cdot). Let y​(ω)∈B​(xi​(ω),R)y(\omega)\in B(x\_{i}(\omega),R) be arbitrary. Then there exists ξ​(ω)\xi(\omega) with ‖ξ​(ω)‖≤R\|\xi(\omega)\|\leq R such that

|  |  |  |
| --- | --- | --- |
|  | y​(ω)=xi​(ω)+ξ​(ω).y(\omega)=x\_{i}(\omega)+\xi(\omega). |  |

Since ϕ​(1,ω,⋅)\phi(1,\omega,\cdot) is C2C^{2}, Taylor’s theorem yields

|  |  |  |
| --- | --- | --- |
|  | ϕ​(1,ω,xi​(ω)+ξ​(ω))=ϕ​(1,ω,xi​(ω))+D​ϕ​(1,ω,xi​(ω))​ξ​(ω)+R​(ω,xi​(ω),ξ​(ω)),\displaystyle\phi(1,\omega,x\_{i}(\omega)+\xi(\omega))=\phi(1,\omega,x\_{i}(\omega))+D\phi(1,\omega,x\_{i}(\omega))\,\xi(\omega)+R(\omega,x\_{i}(\omega),\xi(\omega)), |  |

where the remainder term satisfies the uniform bound

|  |  |  |
| --- | --- | --- |
|  | ‖R​(ω,xi​(ω),ξ)‖≤C​(ω,R)​‖ξ‖2≤C​(ω,R)​R2.\|R(\omega,x\_{i}(\omega),\xi)\|\leq C(\omega,R)\,\|\xi\|^{2}\leq C(\omega,R)\,R^{2}. |  |

Since the point y​(ω)∈B​(xi​(ω),R)y(\omega)\in B(x\_{i}(\omega),R) was arbitrary, the above estimate holds uniformly for all y​(ω)∈B​(xi​(ω),R)y(\omega)\in B(x\_{i}(\omega),R). Therefore, the image of the ball B​(xi​(ω),R)B(x\_{i}(\omega),R) under ϕ​(1,ω,⋅)\phi(1,\omega,\cdot) satisfies the set inclusion

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(1,ω,B​(xi​(ω),R))⊂{ϕ​(1,ω,xi​(ω))}⊕D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2),\displaystyle\phi(1,\omega,B(x\_{i}(\omega),R))\subset\{\phi(1,\omega,x\_{i}(\omega))\}\;\oplus\;D\phi(1,\omega,x\_{i}(\omega))\,B(0,R)\;\oplus\;B\bigl(0,C(\omega,R)\,R^{2}\bigr), |  | (28) |

where ⊕\oplus denotes the Minkowski sum of sets. Since ii was arbitrary, this holds for each i∈{1,…​N1}i\in\{1,\dots N\_{1}\}.
In other words, we showed the set inclusion for almost all ω\omega.

#### Step 3: Covering Numbers

We recall we have now by ([26](#A2.E26 "Equation 26 ‣ Step 1: General Covering ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and ([28](#A2.E28 "Equation 28 ‣ Step 2: Local Approximation ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability"))

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜​(θ​ω)⊂⋃i=1N1{ϕ​(1,ω,xi​(ω))}⊕D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2).\displaystyle\mathcal{A}(\theta\omega)\subset\bigcup\_{i=1}^{N\_{1}}\{\phi(1,\omega,x\_{i}(\omega))\}\;\oplus\;D\phi(1,\omega,x\_{i}(\omega))\,B(0,R)\;\oplus\;B\bigl(0,C(\omega,R)\,R^{2}\bigr). |  | (29) |

Now, we will obtain an esimate on the covering number of 𝒜​(θ​ω)\mathcal{A}(\theta\omega) by using the fact that each element in the union

|  |  |  |
| --- | --- | --- |
|  | {ϕ​(1,ω,xi​(ω))}⊕D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2)\{\phi(1,\omega,x\_{i}(\omega))\}\;\oplus\;D\phi(1,\omega,x\_{i}(\omega))\,B(0,R)\;\oplus\;B\bigl(0,C(\omega,R)\,R^{2}\bigr) |  |

is a dilated ellipsoid, so that we can use our result for estimating covering numbers for ellipsoids.

More precisely, we have that for any ρ>0\rho>0

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θ​ω),ρ)≤∑i=1N1𝒩​({ϕ​(1,ω,xi​(ω))}⊕D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2),ρ)\displaystyle\mathcal{N}(\mathcal{A}(\theta\omega),\rho)\leq\sum\_{i=1}^{N\_{1}}\mathcal{N}\left(\{\phi(1,\omega,x\_{i}(\omega))\}\oplus D\phi(1,\omega,x\_{i}(\omega))B(0,R)\oplus B(0,C(\omega,R)R^{2}),\rho\right) |  | (30) |

We now apply Lemma [B.2](#A2.Thmtheorem2 "Lemma B.2 (Covering of Minkowski sums). ‣ Covering for Minkowski Sum ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability") with

|  |  |  |
| --- | --- | --- |
|  | X:=D​ϕ​(1,ω,xi​(ω))​B​(0,R),Y:=B​(0,C​(ω,R)​R2).X:=D\phi(1,\omega,x\_{i}(\omega))\,B(0,R),\qquad Y:=B\bigl(0,C(\omega,R)\,R^{2}\bigr). |  |

The translation by ϕ​(1,ω,xi​(ω))\phi(1,\omega,x\_{i}(\omega)) does not affect the covering number and can be ignored in the following estimates.

We further define as covering radius

|  |  |  |
| --- | --- | --- |
|  | ρ1​(ω):=R​supw∈𝒜​(ω)σj∗+1​(D​ϕ​(1,ω,w)).\rho\_{1}(\omega):=R\sup\_{w\in\mathcal{A}(\omega)}\sigma\_{j^{\*}+1}(D\phi(1,\omega,w)). |  |

By Lemma [B.2](#A2.Thmtheorem2 "Lemma B.2 (Covering of Minkowski sums). ‣ Covering for Minkowski Sum ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability") we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(X⊕Y,ρ1​(ω)+C​(ω,R)​R2)≤𝒩​(X,ρ1​(ω)).\displaystyle\mathcal{N}(X\oplus Y,\rho\_{1}(\omega)+C(\omega,R)R^{2})\leq\mathcal{N}(X,\rho\_{1}(\omega)). |  | (31) |

We recall that j∗j^{\*} is the transition index and σi​(D​ϕ​(1,ω,x))\sigma\_{i}(D\phi(1,\omega,x)) denotes the ii-th singular value of D​ϕ​(1,ω,x)D\phi(1,\omega,x) for x∈ℝdx\in\mathbb{R}^{d}.
Now we recall that X is an ellipsoid with semi axis lengths

|  |  |  |
| --- | --- | --- |
|  | R​σ1​(D​ϕ​(1,ω,xi​(ω))),…,R​σd​(D​ϕ​(1,ω,xi​(ω)))R\sigma\_{1}(D\phi(1,\omega,x\_{i}(\omega))),\dots,R\sigma\_{d}(D\phi(1,\omega,x\_{i}(\omega))) |  |

since

|  |  |  |
| --- | --- | --- |
|  | X=D​ϕ​(1,ω,xi​(ω))​B​(0,R)=R⋅D​ϕ​(1,ω,xi​(ω))​B​(0,1).X=D\phi(1,\omega,x\_{i}(\omega))\,B(0,R)=R\cdot D\phi(1,\omega,x\_{i}(\omega))\,B(0,1). |  |

Therefore, we are now ready to apply Lemma [B.1](#A2.Thmtheorem1 "Lemma B.1 (Ellipsoid covering). ‣ Covering Estimates for Ellipsoid ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability"). We note that

|  |  |  |
| --- | --- | --- |
|  | Rσj∗+1(Dϕ(1,ω,xi(ω)))≤Rsupw∈𝒜​(ω)σj∗+1(Dϕ(1,ω,w)=ρ1(ω).R\sigma\_{j^{\*}+1}(D\phi(1,\omega,x\_{i}(\omega)))\leq R\sup\_{w\in\mathcal{A}(\omega)}\sigma\_{j^{\*}+1}(D\phi(1,\omega,w)=\rho\_{1}(\omega). |  |

Hence, by Lemma [B.1](#A2.Thmtheorem1 "Lemma B.1 (Ellipsoid covering). ‣ Covering Estimates for Ellipsoid ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability") we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩(X,ρ1(ω))≤3d∏k=1j∗R​σk​(D​ϕ​(1,ω,xi​(ω)))ρ1​(ω)≤3d∏k=1j∗Rsupx∈𝒜​(ω)σk(Dϕ(1,ω,x)ρ1​(ω)=:3dμ(ω).\displaystyle\mathcal{N}(X,\rho\_{1}(\omega))\leq 3^{d}\prod\_{k=1}^{j^{\*}}\frac{R\sigma\_{k}(D\phi(1,\omega,x\_{i}(\omega)))}{\rho\_{1}(\omega)}\leq 3^{d}\prod\_{k=1}^{j^{\*}}\frac{R\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{k}(D\phi(1,\omega,x)}{\rho\_{1}(\omega)}=:3^{d}\mu(\omega). |  | (32) |

Notice that even though XX depends on the particular center xi​(ω)x\_{i}(\omega), μ​(ω)\mu(\omega) is uniform over all the centers and does not depend on ii.

Now, let us define

|  |  |  |
| --- | --- | --- |
|  | ρ​(ω):=ρ1​(ω)+C​(ω,R)​R2\rho(\omega):=\rho\_{1}(\omega)+C(\omega,R)R^{2} |  |

and observe that by combining ([30](#A2.E30 "Equation 30 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and ([31](#A2.E31 "Equation 31 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), we have that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θ​ω),ρ​(ω))≤\displaystyle\mathcal{N}(\mathcal{A}(\theta\omega),\rho(\omega))\leq | ∑i=1N1𝒩​({ϕ​(1,ω,xi​(ω))}⊕D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2),ρ​(ω))\displaystyle\sum\_{i=1}^{N\_{1}}\mathcal{N}\left(\{\phi(1,\omega,x\_{i}(\omega))\}\oplus D\phi(1,\omega,x\_{i}(\omega))B(0,R)\oplus B(0,C(\omega,R)R^{2}),\rho(\omega)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =\displaystyle= | ∑i=1N1𝒩​(D​ϕ​(1,ω,xi​(ω))​B​(0,R)⊕B​(0,C​(ω,R)​R2),ρ​(ω))\displaystyle\sum\_{i=1}^{N\_{1}}\mathcal{N}\left(D\phi(1,\omega,x\_{i}(\omega))B(0,R)\oplus B(0,C(\omega,R)R^{2}),\rho(\omega)\right) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ≤\displaystyle\leq | ∑i=1N1𝒩(Dϕ(1,ω,xi(ω))B(0,R),ρ1(ω)\displaystyle\sum\_{i=1}^{N\_{1}}\mathcal{N}(D\phi(1,\omega,x\_{i}(\omega))B(0,R),\rho\_{1}(\omega) |  | (33) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ≤\displaystyle\leq | N1​(ω,R)​3d​μ​(ω),\displaystyle N\_{1}(\omega,R)3^{d}\mu(\omega), |  | (34) |

where ([33](#A2.E33 "Equation 33 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) follows from ([31](#A2.E31 "Equation 31 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and ([34](#A2.E34 "Equation 34 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) follows from ([32](#A2.E32 "Equation 32 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")).

Now, we choose RR sufficiently small such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | C​(ω,R)​R2≤ρ1​(ω).\displaystyle C(\omega,R)R^{2}\leq\rho\_{1}(\omega). |  | (35) |

This is possible since when RR tends to 0, C​(ω,R)C(\omega,R) converges to a constant by definition (cf. ([27](#A2.E27 "Equation 27 ‣ Step 2: Local Approximation ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability"))) hence C​(ω,R)​R2C(\omega,R)R^{2} tends to 0 as RR tends to 0. On the other hand, since by assumption infx∈𝒜​(ω)σd​(D​ϕ​(1,ω,x))>0\inf\_{x\in\mathcal{A}(\omega)}\sigma\_{d}(D\phi(1,\omega,x))>0, that makes ρ1​(ω)>0\rho\_{1}(\omega)>0, which yields ([35](#A2.E35 "Equation 35 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")).

Therefore, for sufficiently small RR, we have that

|  |  |  |
| --- | --- | --- |
|  | ρ(ω)≤2ρ1(ω)=:ϵ1(ω).\rho(\omega)\leq 2\rho\_{1}(\omega)=:\epsilon\_{1}(\omega). |  |

By the monotonicity of covering numbers, a larger radius requires fewer balls.
Therefore, we have the following chain of inequalities:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θ​ω),ϵ1​(ω))≤𝒩​(𝒜​(θ​ω),ρ​(ω))≤N1​(ω,R)​3d​μ​(ω),\displaystyle\mathcal{N}(\mathcal{A}(\theta\omega),\epsilon\_{1}(\omega))\leq\mathcal{N}(\mathcal{A}(\theta\omega),\rho(\omega))\leq N\_{1}(\omega,R)3^{d}\mu(\omega), |  | (36) |

where the last step follows from ([34](#A2.E34 "Equation 34 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")).

At this stage we have estimated the covering number of 𝒜​(θ1​ω)\mathcal{A}(\theta^{1}\omega). In the next step, we will iterate this idea to get an estimate on the covering number of 𝒜​(θK​ω)\mathcal{A}(\theta^{K}\omega).

#### Step 4: Global Iteration and Recursive Covering

We now extend the one-step covering estimate to multiple time steps by induction over the discrete time index KK. Fix a sufficiently small initial radius R>0R>0 and suppose that 𝒜​(ω)\mathcal{A}(\omega) is covered by N1​(ω,R)N\_{1}(\omega,R) balls of radius RR. By the invariance of the random attractor,

|  |  |  |
| --- | --- | --- |
|  | 𝒜​(θK+1​ω)=ϕ​(1,θK​ω,𝒜​(θK​ω))for all ​K∈ℕ.\mathcal{A}(\theta^{K+1}\omega)=\phi(1,\theta^{K}\omega,\mathcal{A}(\theta^{K}\omega))\quad\text{for all }K\in\mathbb{N}. |  |

Define the sequence of radii {ϵK}K≥0\{\epsilon\_{K}\}\_{K\geq 0} by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵK+1​(ω):=2⋅ϵK​(ω)​supx∈𝒜​(θK​ω)σj∗+1​(D​ϕ​(1,θK​ω,x)),ϵ0:=R.\epsilon\_{K+1}(\omega):=2\cdot\epsilon\_{K}(\omega)\sup\_{x\in\mathcal{A}(\theta^{K}\omega)}\sigma\_{j^{\*}+1}(D\phi(1,\theta^{K}\omega,x)),\qquad\epsilon\_{0}:=R. |  | (37) |

Moreover, the one-step covering argument derived above applies to any sufficiently small covering radius. Consequently, if 𝒜​(θK​ω)\mathcal{A}(\theta^{K}\omega) is covered by balls of radius ϵK​(ω)\epsilon\_{K}(\omega), then applying the same argument with ω\omega replaced by θK​ω\theta^{K}\omega yields a cover of 𝒜​(θK+1​ω)\mathcal{A}(\theta^{K+1}\omega) by balls of radius ϵK+1​(ω)\epsilon\_{K+1}(\omega). In other words, all previous estimates remain valid after the replacements 𝒜​(ω)←𝒜​(θK​ω)\mathcal{A}(\omega)\leftarrow\mathcal{A}(\theta^{K}\omega) and 𝒜​(θ​ω)←𝒜​(θK+1​ω)\mathcal{A}(\theta\omega)\leftarrow\mathcal{A}(\theta^{K+1}\omega).

By the one-step covering estimate from ([36](#A2.E36 "Equation 36 ‣ Step 3: Covering Numbers ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), each ball of radius ϵK​(ω)\epsilon\_{K}(\omega) is mapped into a set that can be covered by at most μ​(θK​ω)\mu(\theta^{K}\omega) balls of radius ϵK+1​(ω)\epsilon\_{K+1}(\omega). Consequently,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θK​ω),ϵK​(ω))≤N1​(ω,R)​∏k=0K−13d⋅μ​(θk​ω).\mathcal{N}(\mathcal{A}(\theta^{K}\omega),\epsilon\_{K}(\omega))\;\leq\;N\_{1}(\omega,R)\prod\_{k=0}^{K-1}3^{d}\cdot\mu(\theta^{k}\omega). |  | (38) |

Iterating the radius recursion yields the explicit expression

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵK​(ω)=R​∏k=0K−1(2⋅supx∈𝒜​(θk​ω)σj∗+1​(D​ϕ​(1,θk​ω,x))).\epsilon\_{K}(\omega)=R\prod\_{k=0}^{K-1}\left(2\cdot\sup\_{x\in\mathcal{A}(\theta^{k}\omega)}\sigma\_{j^{\*}+1}(D\phi(1,\theta^{k}\omega,x))\right). |  | (39) |

#### Step 5: Ergodic Limits and Dimension Bound

To determine the asymptotic growth rate of the covering number, we begin by recalling the definition of the ii-th sharpness exponent:

|  |  |  |  |
| --- | --- | --- | --- |
|  | λi:=𝔼​[supx∈𝒜​(ω)ln⁡σi​(D​ϕ​(1,ω,x))].\lambda\_{i}:=\mathbb{E}\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{i}(D\phi(1,\omega,x))\right]. |  | (40) |

Taking logarithms in the covering estimate from ([38](#A2.E38 "Equation 38 ‣ Step 4: Global Iteration and Recursive Covering ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and normalizing by KK, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1K​ln⁡𝒩​(𝒜​(θK​ω),ϵK​(ω))≤1K​ln⁡N1​(ω,R)+1K​∑k=0K−1ln⁡μ​(θk​ω)+d​ln⁡3.\frac{1}{K}\ln\mathcal{N}(\mathcal{A}(\theta^{K}\omega),\epsilon\_{K}(\omega))\leq\frac{1}{K}\ln N\_{1}(\omega,R)+\frac{1}{K}\sum\_{k=0}^{K-1}\ln\mu(\theta^{k}\omega)+d\ln 3. |  | (41) |

Since N1​(ω,R)N\_{1}(\omega,R) is finite almost surely and independent of KK, we have 1K​ln⁡N1​(ω,R)→0\frac{1}{K}\ln N\_{1}(\omega,R)\to 0 as K→∞K\to\infty. By the Birkhoff Ergodic Theorem (see Yunis, ([2017](#bib.bib76), Theorem 3.10)) applied to the ergodic process ω↦ln⁡μ​(ω)\omega\mapsto\ln\mu(\omega)888By our integrability assumption μ\mu is also integrable, therefore the theorem can be applied., we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | lim supK→∞1n​ln⁡𝒩​(𝒜​(θK​ω),ϵK​(ω))≤𝔼​[ln⁡μ​(ω)]+d​ln⁡3\displaystyle\limsup\_{K\to\infty}\frac{1}{n}\ln\mathcal{N}(\mathcal{A}(\theta^{K}\omega),\epsilon\_{K}(\omega))\leq\mathbb{E}\left[\ln\mu(\omega)\right]+d\ln 3 |  | (42) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[∑i=1j∗supx∈𝒜​(ω)ln⁡σi​(D​ϕ​(1,ω,x))−j∗​supx∈𝒜​(ω)ln⁡σj∗+1​(D​ϕ​(1,ω,x))]+d​ln⁡3.\displaystyle=\mathbb{E}\left[\sum\_{i=1}^{j^{\*}}\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{i}(D\phi(1,\omega,x))-j^{\*}\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{j^{\*}+1}(D\phi(1,\omega,x))\right]+d\ln 3. |  | (43) |

#### Step 5: Asymptotic Scale Decay

By using ([39](#A2.E39 "Equation 39 ‣ Step 4: Global Iteration and Recursive Covering ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), taking logarithms and normalizing by KK, we compute

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 1K​ln⁡ϵK​(ω)\displaystyle\frac{1}{K}\ln\epsilon\_{K}(\omega) | =1K​∑k=0K−1(supx∈𝒜​(θk​ω)ln⁡σj∗+1​(D​ϕ​(1,θk​ω,x)))+ln⁡2+1K​ln⁡R.\displaystyle=\frac{1}{K}\sum\_{k=0}^{K-1}\left(\sup\_{x\in\mathcal{A}(\theta^{k}\omega)}\ln\sigma\_{j^{\*}+1}(D\phi(1,\theta^{k}\omega,x))\right)+\ln 2+\frac{1}{K}\ln R. |  | (44) |

Applying the Birkhoff Ergodic Theorem once more, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | limK→∞1K​ln⁡ϵK​(ω)=𝔼​[supx∈𝒜​(ω)ln⁡σj∗+1​(D​ϕ​(1,ω,x))]+ln⁡2=λj∗+1+ln⁡2.\lim\_{K\to\infty}\frac{1}{K}\ln\epsilon\_{K}(\omega)=\mathbb{E}\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{j^{\*}+1}(D\phi(1,\omega,x))\right]+\ln 2=\lambda\_{j^{\*}+1}+\ln 2. |  | (45) |

Our goal is to study the asymptotic growth of the covering numbers 𝒩​(𝒜​(θK​ω),ϵK​(ω))\mathcal{N}(\mathcal{A}(\theta^{K}\omega),\epsilon\_{K}(\omega)) by dividing their logarithm by KK and passing to the limit K→∞K\to\infty. However, the current bound involve several scale-independent multiplicative constants (such as 3d3^{d}) that obscure the leading exponential behavior and would not disappear when taking this limit. To resolve this, we refine the analysis by rewriting the recursive scales ϵK​(ω)\epsilon\_{K}(\omega) in a form that makes their multiplicative structure explicit.

#### Step 6: Computation of the Dimension Bound

We consider the mm-th iterate of the cocycle, ϕ​(m,ω,x)\phi(m,\omega,x), x∈ℝdx\in\mathbb{R}^{d}. By the cocycle property we have for m,k∈ℕm,k\in\mathbb{N} and x∈ℝdx\in\mathbb{R}^{d}

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϕ​(m+k,ω,x)=ϕ​(k,θm​ω,ϕ​(m,ω,x)),\phi(m+k,\omega,x)=\phi(k,\theta^{m}\omega,\phi(m,\omega,x)), |  | (46) |

and by applying the chain rule:

|  |  |  |  |
| --- | --- | --- | --- |
|  | D​ϕ​(m+k,ω,x)=D​ϕ​(k,θm​ω,ϕ​(m,ω,x))⋅D​ϕ​(m,ω,x).D\phi(m+k,\omega,x)=D\phi(k,\theta^{m}\omega,\phi(m,\omega,x))\cdot D\phi(m,\omega,x). |  | (47) |

For a linear map A∈ℝd×dA\in\mathbb{R}^{d\times d} and j∈{1,…,d}j\in\{1,\dots,d\}, define the *jj-th exterior-power norm*

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖A‖j:=σ1​(A)​σ2​(A)​⋯​σj​(A),\|A\|\_{j}:=\sigma\_{1}(A)\,\sigma\_{2}(A)\cdots\sigma\_{j}(A), |  | (48) |

where σ1​(A)≥σ2​(A)≥⋯≥σd​(A)≥0\sigma\_{1}(A)\geq\sigma\_{2}(A)\geq\dots\geq\sigma\_{d}(A)\geq 0 are the singular values of AA. We also set ‖A‖0:=1\|A\|\_{0}:=1 for convenience, so that σj​(A)=‖A‖j/‖A‖j−1\sigma\_{j}(A)=\|A\|\_{j}/\|A\|\_{j-1} for all j≥1j\geq 1.

By the functoriality of the exterior power, ⋀j(A​B)=(⋀jA)​(⋀jB)\bigwedge^{j}(AB)=(\bigwedge^{j}A)(\bigwedge^{j}B), and the submultiplicativity of the operator norm, we have the fundamental inequality

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖A​B‖j≤‖A‖j​‖B‖jfor all ​A,B∈ℝd×d.\|AB\|\_{j}\leq\|A\|\_{j}\,\|B\|\_{j}\qquad\text{for all }A,B\in\mathbb{R}^{d\times d}. |  | (49) |

6.1. Subadditivity and Fekete limits. Define the expected log-growth of ωj\omega\_{j} along the cocycle:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ωj(m):=𝔼​[ln​supw∈𝒜​(ω)‖(D​ϕ​(m,ω,w))‖j],m∈ℕ∗.\Omega\_{j}^{(m)}:=\mathbb{E}\!\left[\,\ln\sup\_{w\in\mathcal{A}(\omega)}\|\bigl(D\phi(m,\omega,w)\bigr)\|\_{j}\right],\qquad m\in\mathbb{N}^{\*}. |  | (50) |

Combining the cocycle identity ([47](#A2.E47 "Equation 47 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) with the submultiplicativity ([49](#A2.E49 "Equation 49 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) gives

|  |  |  |
| --- | --- | --- |
|  | ‖(D​ϕ​(m+k,ω,x))‖j≤‖(D​ϕ​(k,θm​ω,ϕ​(m,ω,x)))‖j⋅‖(D​ϕ​(m,ω,x))‖j.\|\bigl(D\phi(m+k,\omega,x)\bigr)\|\_{j}\leq\|\bigl(D\phi(k,\theta^{m}\omega,\phi(m,\omega,x))\bigr)\|\_{j}\cdot\|\bigl(D\phi(m,\omega,x)\bigr)\|\_{j}. |  |

Taking the supremum over x∈𝒜​(ω)x\in\mathcal{A}(\omega) and using the invariance ϕ​(m,ω,𝒜​(ω))=𝒜​(θm​ω)\phi(m,\omega,\mathcal{A}(\omega))=\mathcal{A}(\theta^{m}\omega):

|  |  |  |
| --- | --- | --- |
|  | supx∈𝒜​(ω)‖(D​ϕ​(m+k,ω,x))‖j≤supy∈𝒜​(θm​ω)‖(D​ϕ​(k,θm​ω,y))‖j⋅supx∈𝒜​(ω)‖(D​ϕ​(m,ω,x))‖j.\sup\_{x\in\mathcal{A}(\omega)}\|\bigl(D\phi(m+k,\omega,x)\bigr)\|\_{j}\leq\sup\_{y\in\mathcal{A}(\theta^{m}\omega)}\|\bigl(D\phi(k,\theta^{m}\omega,y)\bigr)\|\_{j}\;\cdot\;\sup\_{x\in\mathcal{A}(\omega)}\|\bigl(D\phi(m,\omega,x)\bigr)\|\_{j}. |  |

Taking logarithms, then expectations, and using the θm\theta^{m}-invariance of ℙ\mathbb{P}, we obtain the subadditivity

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ωj(m+k)≤Ωj(m)+Ωj(k)for all ​m,k∈ℕ∗.\Omega\_{j}^{(m+k)}\leq\Omega\_{j}^{(m)}+\Omega\_{j}^{(k)}\qquad\text{for all }m,k\in\mathbb{N}^{\*}. |  | (51) |

By the integrability assumption and Fekete’s lemma, the following limit exists for each j∈{1,…,d}j\in\{1,\dots,d\}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Λj:=limm→∞1m​Ωj(m)=infm≥11m​Ωj(m).\Lambda\_{j}:=\lim\_{m\to\infty}\frac{1}{m}\,\Omega\_{j}^{(m)}=\inf\_{m\geq 1}\frac{1}{m}\,\Omega\_{j}^{(m)}. |  | (52) |

We set Λ0:=0\Lambda\_{0}:=0. By Fekete’s lemma, Λj≤Ωj(1)\Lambda\_{j}\leq\Omega\_{j}^{(1)}. Moreover, since ln\ln is monotone increasing and ∏i=1jσi​(D​ϕ​(1,ω,x))>0\prod\_{i=1}^{j}\sigma\_{i}(D\phi(1,\omega,x))>0 by the non-singularity assumption,

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ωj(1)\displaystyle\Omega\_{j}^{(1)} | =𝔼​[ln​supx∈𝒜​(ω)∏i=1jσi​(D​ϕ​(1,ω,x))]=𝔼​[supx∈𝒜​(ω)∑i=1jln⁡σi​(D​ϕ​(1,ω,x))]\displaystyle=\mathbb{E}\!\left[\ln\sup\_{x\in\mathcal{A}(\omega)}\prod\_{i=1}^{j}\sigma\_{i}(D\phi(1,\omega,x))\right]=\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\sum\_{i=1}^{j}\ln\sigma\_{i}(D\phi(1,\omega,x))\right] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≤∑i=1j𝔼​[supx∈𝒜​(ω)ln⁡σi​(D​ϕ​(1,ω,x))]=∑i=1jλi,\displaystyle\leq\sum\_{i=1}^{j}\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{i}(D\phi(1,\omega,x))\right]=\sum\_{i=1}^{j}\lambda\_{i}, |  | (53) |

where the inequality uses supx(f1​(x)+⋯+fj​(x))≤supxf1​(x)+⋯+supxfj​(x)\sup\_{x}(f\_{1}(x)+\cdots+f\_{j}(x))\leq\sup\_{x}f\_{1}(x)+\cdots+\sup\_{x}f\_{j}(x). Hence Λj≤∑i=1jλi\Lambda\_{j}\leq\sum\_{i=1}^{j}\lambda\_{i}. In particular, if Λj≥0\Lambda\_{j}\geq 0 then ∑i=1jλi≥0\sum\_{i=1}^{j}\lambda\_{i}\geq 0, so the transition index j∗​(Λ)j^{\*}(\Lambda) (the largest j∈{0,…,d−1}j\in\{0,\dots,d-1\} with Λj≥0\Lambda\_{j}\geq 0) satisfies jΛ∗≤j∗j^{\*}\_{\Lambda}\leq j^{\*} the transition index defined in terms of the one-step exponents λi\lambda\_{i}.

Now define the mm-step sharpness exponents

|  |  |  |  |
| --- | --- | --- | --- |
|  | λi(m):=𝔼​[supx∈𝒜​(ω)ln⁡σi​(D​ϕ​(m,ω,x))],i=1,…,d.\lambda\_{i}^{(m)}:=\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{i}\bigl(D\phi(m,\omega,x)\bigr)\right],\qquad i=1,\dots,d. |  | (54) |

6.3. Covering bound for the mm-step map. We apply the covering argument from Steps 1–3 to the mm-step map
ϕ​(m,ω,⋅)\phi(m,\omega,\cdot) in place of ϕ​(1,ω,⋅)\phi(1,\omega,\cdot), with the
replacement θ​ω←θm​ω\theta\omega\leftarrow\theta^{m}\omega. Steps 1 and 2
(general covering and Taylor approximation) carry over without
modification. In Step 3, the ellipsoid covering lemma
(Lemma [B.1](#A2.Thmtheorem1 "Lemma B.1 (Ellipsoid covering). ‣ Covering Estimates for Ellipsoid ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) involves a free index jj at
which the singular values are split into expanding directions
σ1,…,σj\sigma\_{1},\dots,\sigma\_{j} and a contracting direction
σj+1\sigma\_{j+1} that determines the covering radius. We choose j=jΛ∗j=j^{\*}\_{\Lambda}, where jΛ∗j^{\*}\_{\Lambda} is the largest
integer in {0,…,d−1}\{0,\dots,d-1\} such that ΛjΛ∗≥0\Lambda\_{j^{\*}\_{\Lambda}}\geq 0.
We now verify that this choice is well-defined and that the resulting
covering argument is valid.

First, jΛ∗j^{\*}\_{\Lambda} is well-defined: by Assumption 4,
∑i=1j∗+1λi<0\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0. Since
Λj∗+1≤∑i=1j∗+1λi<0\Lambda\_{j^{\*}+1}\leq\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0, and
Λ0=0≥0\Lambda\_{0}=0\geq 0, there exists at least one jj with
Λj≥0\Lambda\_{j}\geq 0 and at least one with Λj<0\Lambda\_{j}<0, so
jΛ∗j^{\*}\_{\Lambda} is well-defined and satisfies
0≤jΛ∗≤j∗0\leq j^{\*}\_{\Lambda}\leq j^{\*}.

Second, ΛjΛ∗+1<0\Lambda\_{j^{\*}\_{\Lambda}+1}<0 by definition of jΛ∗j^{\*}\_{\Lambda}.
By Fekete’s lemma, 1m​ΩjΛ∗+1(m)→ΛjΛ∗+1<0\frac{1}{m}\,\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}\to\Lambda\_{j^{\*}\_{\Lambda}+1}<0. Since for any A∈ℝd×dA\in\mathbb{R}^{d\times d} it holds σ1​(A)≥σ2​(A)≥⋯≥σd​(A)\sigma\_{1}(A)\geq\sigma\_{2}(A)\geq\cdots\geq\sigma\_{d}(A), we
have σk​(A)≥σjΛ∗+1​(A)\sigma\_{k}(A)\geq\sigma\_{j^{\*}\_{\Lambda}+1}(A) for all
k≤jΛ∗+1k\leq j^{\*}\_{\Lambda}+1, and therefore

|  |  |  |
| --- | --- | --- |
|  | ‖A‖jΛ∗+1=∏k=1jΛ∗+1σk​(A)≥σjΛ∗+1​(A)jΛ∗+1.\|A\|\_{j^{\*}\_{\Lambda}+1}=\prod\_{k=1}^{j^{\*}\_{\Lambda}+1}\sigma\_{k}(A)\geq\sigma\_{j^{\*}\_{\Lambda}+1}(A)^{j^{\*}\_{\Lambda}+1}. |  |

Taking logarithms and dividing by jΛ∗+1j^{\*}\_{\Lambda}+1:

|  |  |  |
| --- | --- | --- |
|  | ln⁡σjΛ∗+1​(A)≤1jΛ∗+1​ln⁡‖A‖jΛ∗+1.\ln\sigma\_{j^{\*}\_{\Lambda}+1}(A)\leq\frac{1}{j^{\*}\_{\Lambda}+1}\,\ln\|A\|\_{j^{\*}\_{\Lambda}+1}. |  |

Applying this with A=D​ϕ​(m,ω,x)A=D\phi(m,\omega,x) and taking the supremum
over x∈𝒜​(ω)x\in\mathcal{A}(\omega):

|  |  |  |
| --- | --- | --- |
|  | supx∈𝒜​(ω)ln⁡σjΛ∗+1​(D​ϕ​(m,ω,x))≤1jΛ∗+1​ln​supx∈𝒜​(ω)‖(D​ϕ​(m,ω,x))‖jΛ∗+1,\sup\_{x\in\mathcal{A}(\omega)}\ln\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x))\;\leq\;\frac{1}{j^{\*}\_{\Lambda}+1}\,\ln\sup\_{x\in\mathcal{A}(\omega)}\|(D\phi(m,\omega,x))\|\_{j^{\*}\_{\Lambda}+1}, |  |

where we used the monotonicity of ln\ln and the fact that
t↦t1/(jΛ∗+1)t\mapsto t^{1/(j^{\*}\_{\Lambda}+1)} is monotone increasing, so the
supremum passes inside. Taking expectations and dividing by mm:

|  |  |  |
| --- | --- | --- |
|  | 1m​λjΛ∗+1(m)≤1jΛ∗+1⋅1m​ΩjΛ∗+1(m)→m→∞ΛjΛ∗+1jΛ∗+1<0,\frac{1}{m}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}\;\leq\;\frac{1}{j^{\*}\_{\Lambda}+1}\cdot\frac{1}{m}\,\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}\;\xrightarrow{m\to\infty}\;\frac{\Lambda\_{j^{\*}\_{\Lambda}+1}}{j^{\*}\_{\Lambda}+1}<0, |  |

by Fekete’s lemma. In particular,
λjΛ∗+1(m)→−∞\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}\to-\infty, which shows that
supx∈𝒜​(ω)σjΛ∗+1​(D​ϕ​(m,ω,x))\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x)) decays exponentially
in mm.
Therefore, the
covering radius
ρ1​(ω,m)=R​supxσjΛ∗+1​(D​ϕ​(m,ω,x))\rho\_{1}(\omega,m)=R\,\sup\_{x}\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x)) shrinks with each iteration of the mm-step map,
and the iterative covering argument of Steps 4–5 produces a finite
dimension bound.

With this choice, the covering number of each image ellipsoid satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(D​ϕ​(m,ω,xi)​B​(0,R),ρ1​(ω,m))≤ 3d​∏k=1jΛ∗σk​(D​ϕ​(m,ω,xi))(supx∈𝒜​(ω)σjΛ∗+1​(D​ϕ​(m,ω,x)))jΛ∗,\mathcal{N}\bigl(D\phi(m,\omega,x\_{i})\,B(0,R),\;\rho\_{1}(\omega,m)\bigr)\;\leq\;3^{d}\,\frac{\prod\_{k=1}^{j^{\*}\_{\Lambda}}\sigma\_{k}\bigl(D\phi(m,\omega,x\_{i})\bigr)}{\bigl(\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x))\bigr)^{j^{\*}\_{\Lambda}}}, |  | (55) |

where ρ1​(ω,m):=R​supx∈𝒜​(ω)σjΛ∗+1​(D​ϕ​(m,ω,x))\rho\_{1}(\omega,m):=R\,\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x)). To obtain a bound that is uniform over all centers xi∈𝒜​(ω)x\_{i}\in\mathcal{A}(\omega), we use

|  |  |  |
| --- | --- | --- |
|  | ∏k=1jΛ∗σk​(D​ϕ​(m,ω,xi))=‖(D​ϕ​(m,ω,xi))‖jΛ∗≤supx∈𝒜​(ω)‖(D​ϕ​(m,ω,x))‖jΛ∗.\prod\_{k=1}^{j^{\*}\_{\Lambda}}\sigma\_{k}\bigl(D\phi(m,\omega,x\_{i})\bigr)=\|\bigl(D\phi(m,\omega,x\_{i})\bigr)\|\_{j^{\*}\_{\Lambda}}\leq\sup\_{x\in\mathcal{A}(\omega)}\|\bigl(D\phi(m,\omega,x)\bigr)\|\_{j^{\*}\_{\Lambda}}. |  |

Define the multiplier

|  |  |  |  |
| --- | --- | --- | --- |
|  | μ(m)​(ω):=supx∈𝒜​(ω)‖(D​ϕ​(m,ω,x))‖jΛ∗(supx∈𝒜​(ω)σjΛ∗+1​(D​ϕ​(m,ω,x)))jΛ∗.\mu^{(m)}(\omega):=\frac{\sup\_{x\in\mathcal{A}(\omega)}\|\bigl(D\phi(m,\omega,x)\bigr)\|\_{j^{\*}\_{\Lambda}}}{\Bigl(\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}\bigl(D\phi(m,\omega,x)\bigr)\Bigr)^{j^{\*}\_{\Lambda}}}. |  | (56) |

Then the one-step covering estimate (applied to the mm-step map) gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θm​ω),ϵ1​(ω,m))≤N1​(ω,R)​ 3d​μ(m)​(ω),\mathcal{N}\bigl(\mathcal{A}(\theta^{m}\omega),\,\epsilon\_{1}(\omega,m)\bigr)\leq N\_{1}(\omega,R)\,3^{d}\,\mu^{(m)}(\omega), |  | (57) |

where ϵ1​(ω,m)=2​ρ1​(ω,m)=2​R​supx∈𝒜​(ω)σjΛ∗+1​(D​ϕ​(m,ω,x))\epsilon\_{1}(\omega,m)=2\rho\_{1}(\omega,m)=2R\,\sup\_{x\in\mathcal{A}(\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}(D\phi(m,\omega,x)).

6.4. Iterating the mm-step map. Define the sequence of radii for the KK-fold iteration of the mm-step map:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵK+1​(ω,m):=2​ϵK​(ω,m)​supx∈𝒜​(θm​K​ω)σjΛ∗+1​(D​ϕ​(m,θm​K​ω,x)),ϵ0:=R.\epsilon\_{K+1}(\omega,m):=2\,\epsilon\_{K}(\omega,m)\,\sup\_{x\in\mathcal{A}(\theta^{mK}\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}\bigl(D\phi(m,\theta^{mK}\omega,x)\bigr),\qquad\epsilon\_{0}:=R. |  | (58) |

By the inductive argument of Step 5 (with θ\theta replaced by θm\theta^{m}):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(θm​K​ω),ϵK​(ω,m))≤N1​(ω,R)​∏k=0K−13d​μ(m)​(θm​k​ω).\mathcal{N}\bigl(\mathcal{A}(\theta^{mK}\omega),\,\epsilon\_{K}(\omega,m)\bigr)\leq N\_{1}(\omega,R)\,\prod\_{k=0}^{K-1}3^{d}\,\mu^{(m)}(\theta^{mk}\omega). |  | (59) |

The radii iterate to

|  |  |  |  |
| --- | --- | --- | --- |
|  | ϵK​(ω,m)=R​∏k=0K−12​supx∈𝒜​(θm​k​ω)σjΛ∗+1​(D​ϕ​(m,θm​k​ω,x)).\epsilon\_{K}(\omega,m)=R\,\prod\_{k=0}^{K-1}2\,\sup\_{x\in\mathcal{A}(\theta^{mk}\omega)}\sigma\_{j^{\*}\_{\Lambda}+1}\bigl(D\phi(m,\theta^{mk}\omega,x)\bigr). |  | (60) |

6.5. Ergodic limits. Taking logarithms in ([59](#A2.E59 "Equation 59 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), dividing by KK, and applying the Birkhoff Ergodic Theorem to ω↦ln⁡μ(m)​(ω)\omega\mapsto\ln\mu^{(m)}(\omega) (with respect to θm\theta^{m}), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | lim supK→∞1K​ln⁡𝒩​(𝒜​(θm​K​ω),ϵK​(ω,m))≤𝔼​[ln⁡μ(m)​(ω)]+d​ln⁡3.\limsup\_{K\to\infty}\frac{1}{K}\ln\mathcal{N}\bigl(\mathcal{A}(\theta^{mK}\omega),\epsilon\_{K}(\omega,m)\bigr)\leq\mathbb{E}\bigl[\ln\mu^{(m)}(\omega)\bigr]+d\ln 3. |  | (61) |

From the definition of μ(m)\mu^{(m)}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[ln⁡μ(m)​(ω)]=ΩjΛ∗(m)−jΛ∗​λjΛ∗+1(m).\mathbb{E}\bigl[\ln\mu^{(m)}(\omega)\bigr]=\Omega\_{j^{\*}\_{\Lambda}}^{(m)}-j^{\*}\_{\Lambda}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}. |  | (62) |

Similarly, from ([60](#A2.E60 "Equation 60 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and the Birkhoff Ergodic Theorem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | limK→∞1K​ln⁡ϵK​(ω,m)=λjΛ∗+1(m)+ln⁡2.\lim\_{K\to\infty}\frac{1}{K}\ln\epsilon\_{K}(\omega,m)=\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+\ln 2. |  | (63) |

6.6. Sign condition and passage to the limit. We need λjΛ∗+1(m)+ln⁡2<0\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+\ln 2<0, i.e.,

|  |  |  |  |
| --- | --- | --- | --- |
|  | λjΛ∗+1(m)<−ln⁡2.\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}<-\ln 2. |  | (64) |

Since σjΛ∗+1​(A)jΛ∗+1≤ωjΛ∗+1​(A)\sigma\_{j^{\*}\_{\Lambda}+1}(A)^{j^{\*}\_{\Lambda}+1}\leq\omega\_{j^{\*}\_{\Lambda}+1}(A) for any A∈ℝd×dA\in\mathbb{R}^{d\times d}, we have the upper bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | λjΛ∗+1(m)≤1jΛ∗+1​ΩjΛ∗+1(m).\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}\leq\frac{1}{j^{\*}\_{\Lambda}+1}\,\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}. |  | (65) |

By Fekete’s lemma, 1m​ΩjΛ∗+1(m)→ΛjΛ∗+1<0\frac{1}{m}\,\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}\to\Lambda\_{j^{\*}\_{\Lambda}+1}<0, so ΩjΛ∗+1(m)→−∞\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}\to-\infty as m→∞m\to\infty. Combined with ([65](#A2.E65 "Equation 65 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), this gives λjΛ∗+1(m)→−∞\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}\to-\infty, and in particular ([64](#A2.E64 "Equation 64 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) holds for all mm sufficiently large.

For such mm, the box-counting dimension satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim¯M​𝒜​(ω)≤ΩjΛ∗(m)−jΛ∗​λjΛ∗+1(m)+d​ln⁡3−λjΛ∗+1(m)−ln⁡2.\overline{\dim}\_{M}\,\mathcal{A}(\omega)\leq\frac{\Omega\_{j^{\*}\_{\Lambda}}^{(m)}-j^{\*}\_{\Lambda}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+d\ln 3}{-\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}-\ln 2}. |  | (66) |

Dividing numerator and denominator by mm:

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim¯M​𝒜​(ω)≤1m​ΩjΛ∗(m)−jΛ∗​1m​λjΛ∗+1(m)+d​ln⁡3m−1m​λjΛ∗+1(m)−ln⁡2m.\overline{\dim}\_{M}\,\mathcal{A}(\omega)\leq\frac{\frac{1}{m}\,\Omega\_{j^{\*}\_{\Lambda}}^{(m)}-j^{\*}\_{\Lambda}\,\frac{1}{m}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+\frac{d\ln 3}{m}}{-\frac{1}{m}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}-\frac{\ln 2}{m}}. |  | (67) |

We now pass to the limit m→∞m\to\infty. By Fekete’s lemma, 1m​ΩjΛ∗(m)→ΛjΛ∗\frac{1}{m}\,\Omega\_{j^{\*}\_{\Lambda}}^{(m)}\to\Lambda\_{j^{\*}\_{\Lambda}}. The terms d​ln⁡3m\frac{d\ln 3}{m} and ln⁡2m\frac{\ln 2}{m} vanish. It remains to identify the limit of 1m​λjΛ∗+1(m)\frac{1}{m}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}.

Since σjΛ∗+1​(A)=ωjΛ∗+1​(A)/ωjΛ∗​(A)\sigma\_{j^{\*}\_{\Lambda}+1}(A)=\omega\_{j^{\*}\_{\Lambda}+1}(A)/\omega\_{j^{\*}\_{\Lambda}}(A), we have the two-sided bounds

|  |  |  |  |
| --- | --- | --- | --- |
|  | ΩjΛ∗+1(m)−ΩjΛ∗(m)≤λjΛ∗+1(m)≤ΩjΛ∗+1(m)−Ω^jΛ∗(m),\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}-\Omega\_{j^{\*}\_{\Lambda}}^{(m)}\;\leq\;\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}\;\leq\;\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}-\hat{\Omega}\_{j^{\*}\_{\Lambda}}^{(m)}, |  | (68) |

where the lower bound uses sup(f−g)≥supf−supg\sup(f-g)\geq\sup f-\sup g and the upper bound uses sup(f−g)≤supf−infg\sup(f-g)\leq\sup f-\inf g. By Fekete’s lemma, 1m​(ΩjΛ∗+1(m)−ΩjΛ∗(m))→ΛjΛ∗+1−ΛjΛ∗\frac{1}{m}(\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}-\Omega\_{j^{\*}\_{\Lambda}}^{(m)})\to\Lambda\_{j^{\*}\_{\Lambda}+1}-\Lambda\_{j^{\*}\_{\Lambda}}. By the Bounded Distortion assumption, 1m​(ΩjΛ∗(m)−Ω^jΛ∗(m))→0\frac{1}{m}(\Omega\_{j^{\*}\_{\Lambda}}^{(m)}-\hat{\Omega}\_{j^{\*}\_{\Lambda}}^{(m)})\to 0, and hence 1m​Ω^jΛ∗(m)→ΛjΛ∗\frac{1}{m}\,\hat{\Omega}\_{j^{\*}\_{\Lambda}}^{(m)}\to\Lambda\_{j^{\*}\_{\Lambda}}, so that 1m​(ΩjΛ∗+1(m)−Ω^jΛ∗(m))→ΛjΛ∗+1−ΛjΛ∗\frac{1}{m}(\Omega\_{j^{\*}\_{\Lambda}+1}^{(m)}-\hat{\Omega}\_{j^{\*}\_{\Lambda}}^{(m)})\to\Lambda\_{j^{\*}\_{\Lambda}+1}-\Lambda\_{j^{\*}\_{\Lambda}}. By the squeeze theorem,

|  |  |  |  |
| --- | --- | --- | --- |
|  | limm→∞1mλjΛ∗+1(m)=ΛjΛ∗+1−ΛjΛ∗=:λ~jΛ∗+1.\lim\_{m\to\infty}\frac{1}{m}\,\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}=\Lambda\_{j^{\*}\_{\Lambda}+1}-\Lambda\_{j^{\*}\_{\Lambda}}=:\tilde{\lambda}\_{j^{\*}\_{\Lambda}+1}. |  | (69) |

Substituting all limits into ([67](#A2.E67 "Equation 67 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim¯M𝒜(ω)≤ΛjΛ∗−jΛ∗​λ~jΛ∗+1−λ~jΛ∗+1=jΛ∗+ΛjΛ∗ΛjΛ∗−ΛjΛ∗+1=:Ds.\overline{\dim}\_{M}\,\mathcal{A}(\omega)\;\leq\;\frac{\Lambda\_{j^{\*}\_{\Lambda}}-j^{\*}\_{\Lambda}\,\tilde{\lambda}\_{j^{\*}\_{\Lambda}+1}}{-\tilde{\lambda}\_{j^{\*}\_{\Lambda}+1}}=j^{\*}\_{\Lambda}+\frac{\Lambda\_{j^{\*}\_{\Lambda}}}{\Lambda\_{j^{\*}\_{\Lambda}}-\Lambda\_{j^{\*}\_{\Lambda}+1}}=:D\_{s}. |  | (70) |

We now show that Ds≤dimS𝒜SD\_{s}\leq\dim\_{\mathrm{S}}\mathcal{A}\_{S}, where
dimS𝒜S:=j∗+∑i=1j∗λi|λj∗+1|\dim\_{\mathrm{S}}\mathcal{A}\_{S}:=j^{\*}+\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|}
denotes the bound with one-step exponents.
Define g​(a,b):=aa−bg(a,b):=\frac{a}{a-b} for a≥0>ba\geq 0>b, so that
Ds=jΛ∗+g​(ΛjΛ∗,ΛjΛ∗+1)D\_{s}=j^{\*}\_{\Lambda}+g(\Lambda\_{j^{\*}\_{\Lambda}},\Lambda\_{j^{\*}\_{\Lambda}+1}).
Since b<0b<0 and a≥0a\geq 0:

|  |  |  |
| --- | --- | --- |
|  | ∂g∂a=−b(a−b)2>0,∂g∂b=a(a−b)2≥0,\frac{\partial g}{\partial a}=\frac{-b}{(a-b)^{2}}>0,\qquad\frac{\partial g}{\partial b}=\frac{a}{(a-b)^{2}}\geq 0, |  |

so gg is monotonically increasing in both arguments, and
g​(a,b)<1g(a,b)<1 since a<a−ba<a-b (because b<0b<0).

We distinguish two cases.

Case 1: jΛ∗=j∗j^{\*}\_{\Lambda}=j^{\*}. Since Λj≤∑i=1jλi\Lambda\_{j}\leq\sum\_{i=1}^{j}\lambda\_{i} for all jj, the
monotonicity of gg in the first argument gives

|  |  |  |
| --- | --- | --- |
|  | g​(Λj∗,Λj∗+1)≤g​(∑i=1j∗λi,Λj∗+1).g(\Lambda\_{j^{\*}},\,\Lambda\_{j^{\*}+1})\;\leq\;g\!\left(\sum\_{i=1}^{j^{\*}}\lambda\_{i},\;\Lambda\_{j^{\*}+1}\right). |  |

By Assumption 4, ∑i=1j∗+1λi<0\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0, and since
Λj∗+1≤∑i=1j∗+1λi<0\Lambda\_{j^{\*}+1}\leq\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0, the
monotonicity of gg in the second argument yields

|  |  |  |
| --- | --- | --- |
|  | g​(∑i=1j∗λi,Λj∗+1)≤g​(∑i=1j∗λi,∑i=1j∗+1λi)=∑i=1j∗λi|λj∗+1|.g\!\left(\sum\_{i=1}^{j^{\*}}\lambda\_{i},\;\Lambda\_{j^{\*}+1}\right)\;\leq\;g\!\left(\sum\_{i=1}^{j^{\*}}\lambda\_{i},\;\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}\right)=\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|}. |  |

Therefore Ds=j∗+g​(Λj∗,Λj∗+1)≤j∗+∑i=1j∗λi|λj∗+1|=DλD\_{s}=j^{\*}+g(\Lambda\_{j^{\*}},\Lambda\_{j^{\*}+1})\leq j^{\*}+\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|}=D\_{\lambda}.

Case 2: jΛ∗<j∗j^{\*}\_{\Lambda}<j^{\*}. Since g​(ΛjΛ∗,ΛjΛ∗+1)<1g(\Lambda\_{j^{\*}\_{\Lambda}},\Lambda\_{j^{\*}\_{\Lambda}+1})<1,
we have Ds<jΛ∗+1≤j∗D\_{s}<j^{\*}\_{\Lambda}+1\leq j^{\*}. On the other hand,
Dλ=j∗+∑i=1j∗λi|λj∗+1|≥j∗D\_{\lambda}=j^{\*}+\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|}\geq j^{\*}, since ∑i=1j∗λi≥0\sum\_{i=1}^{j^{\*}}\lambda\_{i}\geq 0 by Assumption 4.
Therefore Ds<j∗≤DλD\_{s}<j^{\*}\leq D\_{\lambda}.

Combining both cases, we conclude that for
ℙ\mathbb{P}-almost every ω\omega:

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim¯M​(𝒜​(ω))≤Ds≤Dλ=j∗+∑i=1j∗λi|λj∗+1|<dimS𝒜.\overline{\dim}\_{M}(\mathcal{A}(\omega))\;\leq\;D\_{s}\;\leq\;D\_{\lambda}\;=\;j^{\*}+\frac{\sum\_{i=1}^{j^{\*}}\lambda\_{i}}{|\lambda\_{j^{\*}+1}|}\;<\;\dim\_{\mathrm{S}}\mathcal{A}. |  | (71) |

The bound DsD\_{s} is strictly sharper than dimS𝒜S\dim\_{\mathrm{S}}\mathcal{A}\_{S} whenever
Λj<∑i=1jλi\Lambda\_{j}<\sum\_{i=1}^{j}\lambda\_{i} for some
j∈{jΛ∗,jΛ∗+1}j\in\{j^{\*}\_{\Lambda},j^{\*}\_{\Lambda}+1\}, which occurs when
different points on the attractor maximize different singular
values σi\sigma\_{i}. This completes Step 6.

#### Step 7: Transition to Arbitrary Radii

The covering estimates obtained in Step 6 are formulated along the
discrete sequence of radii {ϵK​(ω,m)}K∈ℕ\{\epsilon\_{K}(\omega,m)\}\_{K\in\mathbb{N}}.
In this step, we show that the dimension bound extends to arbitrary
radii ϵ→0\epsilon\to 0, thereby establishing an upper bound on the
upper Minkowski dimension.

Fix m∈ℕm\in\mathbb{N} sufficiently large such that
λjΛ∗+1(m)+ln⁡2<0\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+\ln 2<0 (which is possible by
Step 6.6). Let {ϵK​(ω,m)}K∈ℕ\{\epsilon\_{K}(\omega,m)\}\_{K\in\mathbb{N}} be the
sequence of radii defined in ([58](#A2.E58 "Equation 58 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")). By
([63](#A2.E63 "Equation 63 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), the Birkhoff Ergodic Theorem gives

|  |  |  |  |
| --- | --- | --- | --- |
|  | limK→∞1KlnϵK(ω,m)=λjΛ∗+1(m)+ln2=:ℓ(m)<0ℙ-a.s.\lim\_{K\to\infty}\frac{1}{K}\ln\epsilon\_{K}(\omega,m)=\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}+\ln 2=:\ell(m)<0\qquad\mathbb{P}\text{-a.s.} |  | (72) |

In particular, ϵK​(ω,m)→0\epsilon\_{K}(\omega,m)\to 0 exponentially as
K→∞K\to\infty, and for ℙ\mathbb{P}-a.e. ω\omega the sequence
{ϵK​(ω,m)}K\{\epsilon\_{K}(\omega,m)\}\_{K} is eventually strictly decreasing.

We now work with a fixed realization ω\omega in the full-measure
set where ([72](#A2.E72 "Equation 72 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and the covering estimate
([59](#A2.E59 "Equation 59 ‣ Step 6: Computation of the Dimension Bound ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) both hold. The estimates in Step 6 bound
𝒩​(𝒜​(θm​K​ω),ϵK​(ω,m))\mathcal{N}(\mathcal{A}(\theta^{mK}\omega),\epsilon\_{K}(\omega,m)).
Since θm​K\theta^{mK} preserves ℙ\mathbb{P} and the entire argument
from Steps 1–6 applies with ω\omega replaced by any ω′\omega^{\prime} in
the underlying full-measure set, the same estimates hold with
θm​K​ω\theta^{mK}\omega replaced by ω\omega. That is, for
ℙ\mathbb{P}-a.e. ω\omega there exists a sequence of radii
ϵK​(ω,m)→0\epsilon\_{K}(\omega,m)\to 0 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | lim supK→∞1K​ln⁡𝒩​(𝒜​(ω),ϵK​(ω,m))≤𝔼​[ln⁡μ(m)​(ω)]+d​ln⁡3.\limsup\_{K\to\infty}\frac{1}{K}\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K}(\omega,m))\;\leq\;\mathbb{E}[\ln\mu^{(m)}(\omega)]+d\ln 3. |  | (73) |

For any ϵ>0\epsilon>0 sufficiently small, there exists
K=K​(ϵ)K=K(\epsilon) such that
ϵ∈[ϵK+1​(ω,m),ϵK​(ω,m))\epsilon\in[\epsilon\_{K+1}(\omega,m),\,\epsilon\_{K}(\omega,m)).
Since covering numbers are monotonically non-increasing in the
radius:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒩​(𝒜​(ω),ϵK​(ω,m))≤𝒩​(𝒜​(ω),ϵ)≤𝒩​(𝒜​(ω),ϵK+1​(ω,m)).\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K}(\omega,m))\;\leq\;\mathcal{N}(\mathcal{A}(\omega),\epsilon)\;\leq\;\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K+1}(\omega,m)). |  | (74) |

Taking logarithms and dividing by −ln⁡ϵ>0-\ln\epsilon>0:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ln⁡𝒩​(𝒜​(ω),ϵK​(ω,m))−ln⁡ϵK+1​(ω,m)≤ln⁡𝒩​(𝒜​(ω),ϵ)−ln⁡ϵ≤ln⁡𝒩​(𝒜​(ω),ϵK+1​(ω,m))−ln⁡ϵK​(ω,m),\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K}(\omega,m))}{-\ln\epsilon\_{K+1}(\omega,m)}\;\leq\;\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon)}{-\ln\epsilon}\;\leq\;\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K+1}(\omega,m))}{-\ln\epsilon\_{K}(\omega,m)}, |  | (75) |

where we used ϵK+1​(ω,m)≤ϵ<ϵK​(ω,m)\epsilon\_{K+1}(\omega,m)\leq\epsilon<\epsilon\_{K}(\omega,m) to bound −ln⁡ϵ-\ln\epsilon from below and above
in the denominators.

It remains to show that the lower and upper bounds in
([75](#A2.E75 "Equation 75 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) have the same lim sup\limsup. By
([72](#A2.E72 "Equation 72 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), 1K​ln⁡ϵK​(ω,m)→ℓ​(m)<0\frac{1}{K}\ln\epsilon\_{K}(\omega,m)\to\ell(m)<0, and therefore

|  |  |  |  |
| --- | --- | --- | --- |
|  | ln⁡ϵK+1​(ω,m)ln⁡ϵK​(ω,m)=1K+1​ln⁡ϵK+1​(ω,m)1K​ln⁡ϵK​(ω,m)⋅K+1K→K→∞ℓ​(m)ℓ​(m)⋅1=1.\frac{\ln\epsilon\_{K+1}(\omega,m)}{\ln\epsilon\_{K}(\omega,m)}=\frac{\frac{1}{K+1}\ln\epsilon\_{K+1}(\omega,m)}{\frac{1}{K}\ln\epsilon\_{K}(\omega,m)}\cdot\frac{K+1}{K}\;\xrightarrow{K\to\infty}\;\frac{\ell(m)}{\ell(m)}\cdot 1=1. |  | (76) |

Consequently, replacing −ln⁡ϵK+1-\ln\epsilon\_{K+1} by −ln⁡ϵK-\ln\epsilon\_{K}
(or vice versa) in the denominators of ([75](#A2.E75 "Equation 75 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) does
not affect the lim sup\limsup. Both the lower and upper bounds converge
to the same value, and we conclude

|  |  |  |  |
| --- | --- | --- | --- |
|  | lim supϵ→0ln⁡𝒩​(𝒜​(ω),ϵ)−ln⁡ϵ=lim supK→∞ln⁡𝒩​(𝒜​(ω),ϵK​(ω,m))−ln⁡ϵK​(ω,m).\limsup\_{\epsilon\to 0}\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon)}{-\ln\epsilon}\;=\;\limsup\_{K\to\infty}\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon\_{K}(\omega,m))}{-\ln\epsilon\_{K}(\omega,m)}. |  | (77) |

Combining ([77](#A2.E77 "Equation 77 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) with the covering estimate
([73](#A2.E73 "Equation 73 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")) and the radii asymptotics
([72](#A2.E72 "Equation 72 ‣ Step 7: Transition to Arbitrary Radii ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability")), we obtain for each fixed mm sufficiently
large:

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim¯M​(𝒜​(ω))=lim supϵ→0ln⁡𝒩​(𝒜​(ω),ϵ)−ln⁡ϵ≤𝔼​[ln⁡μ(m)​(ω)]+d​ln⁡3−λjΛ∗+1(m)−ln⁡2.\overline{\dim}\_{M}(\mathcal{A}(\omega))\;=\;\limsup\_{\epsilon\to 0}\frac{\ln\mathcal{N}(\mathcal{A}(\omega),\epsilon)}{-\ln\epsilon}\;\leq\;\frac{\mathbb{E}[\ln\mu^{(m)}(\omega)]+d\ln 3}{-\lambda\_{j^{\*}\_{\Lambda}+1}^{(m)}-\ln 2}. |  | (78) |

Passing to the limit m→∞m\to\infty as in Step 6.6 yields

|  |  |  |
| --- | --- | --- |
|  | dim¯M​(𝒜​(ω))≤dimS𝒜S\overline{\dim}\_{M}(\mathcal{A}(\omega))\;\leq\;\dim\_{\mathrm{S}}\mathcal{A}\_{S} |  |

for ℙ\mathbb{P}-almost every ω\omega. This completes the proof.
∎

###### Theorem B.4.

Generalization via Sharpness Dimension

Let S={z1,…,zn}∼μz⊗nS=\{z\_{1},\dots,z\_{n}\}\sim\mu\_{z}^{\otimes n} be a dataset of size nn. Let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a discrete-time RDS according to Dfn [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability") such that Assump. [4.4](#S4.Thmtheorem4 "Assumption 4.4 (Regular Random Dynamics). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") holds.
Under Assumps. [4.2](#S4.Thmtheorem2 "Assumption 4.2 (Boundedness of Loss). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability") and [4.3](#S4.Thmtheorem3 "Assumption 4.3 (Lipschitz Continuity of Loss). ‣ Generalization Bounds ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"), there exists a constant C>0C>0 s.t. with probability at least 1−ζ−γ1-\zeta-\gamma over the joint draw (S,ω)∼μz⊗n⊗ℙ(S,\omega)\sim\mu\_{z}^{\otimes n}\otimes\mathbb{P}, there exists δn,γ>0\delta\_{n,\gamma}>0 such that for all 0<δ<δn,γ0<\delta<\delta\_{n,\gamma},

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒢S​(𝒜​(ω))\displaystyle\mathcal{G}\_{S}(\mathcal{A}(\omega)) | ≤2​L​δ+2​B​4​dimS𝒜S​log⁡(1/δ)n\displaystyle\leq 2L\delta+2B\sqrt{\frac{4\,\dim\_{\mathrm{S}}\mathcal{A}\_{S}\>\log(1/\delta)}{n}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +I∞​(𝒜S​(ω),S)+log⁡(1/ζ)n+C​B2n.\displaystyle\quad+\frac{I\_{\infty}(\mathcal{A}\_{S}(\omega),S)+\log(1/\zeta)}{\sqrt{n}}+\frac{CB^{2}}{\sqrt{n}}. |  |

We recall that 𝒢S​(𝒜​(ω))\mathcal{G}\_{S}(\mathcal{A}(\omega)) denotes the worst-case generalization gap (see ([1](#S3.E1 "Equation 1 ‣ Learning setup ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability"))) and I∞​(𝒜S​(ω),S)I\_{\infty}(\mathcal{A}\_{S}(\omega),S) (see Dfn. [A.5](#A1.Thmdefinition5 "Definition A.5 (Total mutual information). ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability")) denotes the total mutual information between the random pullback attractor 𝒜S​(ω)\mathcal{A}\_{S}(\omega) and SS.

###### Proof.

The result is an immediate consequence of Cor. [A.4](#A1.Thmtheorem4 "Corollary A.4. ‣ Generalization Bounds via Mutual Information ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") and Thm. [B.3](#A2.Thmtheorem3 "Theorem B.3 (Minkowski Dimension Bound). ‣ Covering for Minkowski Sum ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability").
∎

###### Theorem B.5 (Stability DSD\_{S} Bound).

Suppose the loss function ℓ\ell is BB-bounded and LL-Lipschitz, and Assump. [A.5](#A1.Thmtheorem5 "Assumption A.5 (Random set stability by Tuci et al., (2025)). ‣ Generalization Bounds via Random Set Stability ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") holds. For each dataset S∈𝒵nS\in\mathcal{Z}^{n}, let (Ω,ℱ,ℙ,θ,ϕ)(\Omega,\mathcal{F},\mathbb{P},\theta,\phi) be a C2C^{2} discrete-time RDS (per Dfn. [3.1](#S3.Thmdefinition1 "Definition 3.1 (Random Dynamical System). ‣ Random dynamical systems ‣ 3 Preliminaries ‣ Generalization at the Edge of Stability")) possessing a unique compact random pullback attractor 𝒜S​(ω)\mathcal{A}\_{S}(\omega). We assume the following conditions hold:

1. (i)

   Non-Singularity: For ℙ\mathbb{P}-a.e. ω\omega, the Jacobian is non-singular on the attractor:

   |  |  |  |
   | --- | --- | --- |
   |  | infx∈𝒜S​(ω)σd​(D​ϕ​(1,ω,x))>0.\displaystyle\inf\_{x\in\mathcal{A}\_{S}(\omega)}\sigma\_{d}(D\phi(1,\omega,x))>0. |  |
2. (ii)

   Integrability: The first and second derivatives of the map satisfy:

   |  |  |  |
   | --- | --- | --- |
   |  | 𝔼​[supx∈𝒜S​(ω)ln+⁡‖D​ϕS​(1,ω,x)‖]<∞and𝔼​[supx∈𝒜S​(ω)ln+⁡‖D2​ϕS​(1,ω,x)‖]<∞,\displaystyle\mathbb{E}\left[\sup\_{x\in\mathcal{A}\_{S}(\omega)}\ln^{+}\|D\phi\_{S}(1,\omega,x)\|\right]<\infty\quad\text{and}\quad\mathbb{E}\left[\sup\_{x\in\mathcal{A}\_{S}(\omega)}\ln^{+}\|D^{2}\phi\_{S}(1,\omega,x)\|\right]<\infty, |  |

   where ln+⁡(x):=max⁡{0,ln⁡x}\ln^{+}(x):=\max\{0,\ln x\}.
3. (iii)

   Transition Index: There exists an integer j∗∈{1,…,d−1}j^{\*}\in\{1,\dots,d-1\} such that the global sharpness values λi\lambda\_{i} (see Dfn. [4](#S4.SS0.SSS0.Px4 "Complexity meassure of Random Attractors ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability")) satisfy:

   |  |  |  |
   | --- | --- | --- |
   |  | ∑i=1j∗λi≥0and∑i=1j∗+1λi<0.\sum\_{i=1}^{j^{\*}}\lambda\_{i}\geq 0\quad\text{and}\quad\sum\_{i=1}^{j^{\*}+1}\lambda\_{i}<0. |  |

   .
4. (iv)

   Bounded Distortion: For A∈ℝd×dA\in\mathbb{R}^{d\times d} and j∈{1,…,d}j\in\{1,\dots,d\}, define

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | ‖A‖j:=σ1​(A)​⋯​σj​(A),\|A\|\_{j}:=\sigma\_{1}(A)\cdots\sigma\_{j}(A), |  | (79) |

   where σ1​(A)≥⋯≥σd​(A)\sigma\_{1}(A)\geq\cdots\geq\sigma\_{d}(A) are the singular values of AA. Equivalently, ‖A‖j\|A\|\_{j} is the maximal expansion factor of AA on jj-dimensional volumes.

   We assume that the spatial variation of ‖D​ϕ​(m,ω,⋅)‖j\|D\phi(m,\omega,\cdot)\|\_{j} over the attractor is subexponential in mm: for each j∈{1,…,d}j\in\{1,\dots,d\},

   |  |  |  |  |
   | --- | --- | --- | --- |
   |  | limm→∞1m​𝔼​[supx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j−infx∈𝒜​(ω)ln⁡‖D​ϕ​(m,ω,x)‖j]=0.\lim\_{m\to\infty}\frac{1}{m}\,\mathbb{E}\!\left[\sup\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\;-\;\inf\_{x\in\mathcal{A}(\omega)}\ln\|D\phi(m,\omega,x)\|\_{j}\right]=0. |  | (80) |

   In other words, the maximal and minimal jj-volume growth rates over 𝒜​(ω)\mathcal{A}(\omega) agree at exponential scale.

Furthermore, assume βn−2/3\beta\_{n}^{-2/3} is an integer divisor of nn. Then, there exists δn>0\delta\_{n}>0 such that for all 0<δ<δn0<\delta<\delta\_{n}:

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[supw∈𝒲S,U(ℛ​(w)−ℛ^S​(w))]≤2​𝔼​[Bn+δ​L+βn1/3​(1+B​4​dimS𝒜S​log⁡1δ)].\displaystyle\mathbb{E}\bigg[\sup\_{w\in\mathcal{W}\_{S,U}}\left(\mathcal{R}(w)-\widehat{\mathcal{R}}\_{S}(w)\right)\bigg]\leq 2\mathbb{E}\bigg[\frac{B}{n}+\delta L+\beta\_{n}^{1/3}\left(1+B\sqrt{4\dim\_{\mathrm{S}}\mathcal{A}\_{S}\log\frac{1}{\delta}}\right)\bigg]. |  |

###### Proof.

The proof follows from Thm. [A.6](#A1.Thmtheorem6 "Theorem A.6 (Tuci et al., (2025) Theorem 4.3.). ‣ Generalization Bounds via Random Set Stability ‣ A.3 Data-dependent worst-case generalization bounds ‣ Appendix A Theoretical Background ‣ Generalization at the Edge of Stability") and Thm. [B.3](#A2.Thmtheorem3 "Theorem B.3 (Minkowski Dimension Bound). ‣ Covering for Minkowski Sum ‣ Geometric Interpretation: Ellipsoids and Linear Images of Balls ‣ Appendix B Omitted Proofs ‣ Generalization at the Edge of Stability").
∎

## Appendix C Implementation Details and Computational Complexity

#### Lanczos and SLQ

For all SLQ-based estimators, each Lanczos run uses one Rademacher probe vector and one minibatch Hessian operator. The
operator is fixed throughout the Lanczos iterations of that run, so the Krylov subspace and Gaussian quadrature
interpretation are well defined. We resample only between independent runs, average the resulting quadrature measures or
their histogram/smoothed density estimates, and then compute SD-SLQ, SD-KDE, or SD-PS from this averaged spectral
measure. This procedure estimates the expectation over minibatch Hessian operators and probe vectors without mixing
Hessian-vector products from different operators inside a single Lanczos run.

More explicitly, let NN be the number of parameters and let RSLQR\_{\mathrm{SLQ}} be the number of independent SLQ
runs. In run ii, we sample a minibatch ℬi\mathcal{B}\_{i} and define the corresponding Hessian operator

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hi:=∇w2ℓ​(w⋆;ℬi).H\_{i}:=\nabla\_{w}^{2}\ell(w^{\star};\mathcal{B}\_{i}). |  | (81) |

Given an independent Rademacher probe vi∈{±1}Nv\_{i}\in\{\pm 1\}^{N}, we set qi,1=vi/‖vi‖q\_{i,1}=v\_{i}/\|v\_{i}\| and run mm Lanczos
steps on the fixed operator HiH\_{i}. This produces a basis Qi=[qi,1,…,qi,m]Q\_{i}=[q\_{i,1},\ldots,q\_{i,m}] and a symmetric tridiagonal
matrix TiT\_{i}, with diagonal entries αi,1,…,αi,m\alpha\_{i,1},\ldots,\alpha\_{i,m} and off-diagonal entries βi,1,…,βi,m−1\beta\_{i,1},\ldots,\beta\_{i,m-1}, such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hi​Qi=Qi​Ti+βi,m​qi,m+1​em⊤.H\_{i}Q\_{i}=Q\_{i}T\_{i}+\beta\_{i,m}q\_{i,m+1}e\_{m}^{\top}. |  | (82) |

Here eme\_{m} denotes the mm-th standard basis vector in ℝm)\mathbb{R}^{m}).
Let

|  |  |  |  |
| --- | --- | --- | --- |
|  | Ti=Ui​diag⁡(α~i,1,…,α~i,m)​Ui⊤T\_{i}=U\_{i}\operatorname{diag}(\widetilde{\alpha}\_{i,1},\ldots,\widetilde{\alpha}\_{i,m})U\_{i}^{\top} |  | (83) |

be the eigendecomposition of this tridiagonal matrix. The eigenvalues α~i,ℓ\widetilde{\alpha}\_{i,\ell} of TiT\_{i} are the Ritz values, i.e. Lanczos approximations to the eigenvalues of HiH\_{i}. The associated columns of UiU\_{i} are the Ritz vectors, and the corresponding Gaussian quadrature weights are given by their squared first components:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ωi,ℓ=(Ui)1​ℓ2,ℓ=1,…,m.\omega\_{i,\ell}=(U\_{i})\_{1\ell}^{2},\qquad\ell=1,\ldots,m. |  | (84) |

Thus, for a test function ff,

|  |  |  |  |
| --- | --- | --- | --- |
|  | qi,1⊤​f​(Hi)​qi,1≈e1⊤​f​(Ti)​e1=∑ℓ=1mωi,ℓ​f​(α~i,ℓ).q\_{i,1}^{\top}f(H\_{i})q\_{i,1}\approx e\_{1}^{\top}f(T\_{i})e\_{1}=\sum\_{\ell=1}^{m}\omega\_{i,\ell}f(\widetilde{\alpha}\_{i,\ell}). |  | (85) |

Because qi,1q\_{i,1} is a normalized Rademacher probe, multiplying by NN gives an unbiased trace-scale estimate in
expectation over the probe. Averaging across the independently sampled minibatch Hessians and probes yields the
empirical spectral measure

|  |  |  |  |
| --- | --- | --- | --- |
|  | ν^H=NRSLQ​∑i=1RSLQ∑ℓ=1mωi,ℓ​δα~i,ℓ.\widehat{\nu}\_{H}=\frac{N}{R\_{\mathrm{SLQ}}}\sum\_{i=1}^{R\_{\mathrm{SLQ}}}\sum\_{\ell=1}^{m}\omega\_{i,\ell}\,\delta\_{\widetilde{\alpha}\_{i,\ell}}. |  | (86) |

Here δx\delta\_{x} denotes the Dirac measure at xx.
This is the object averaged by our SLQ procedure: not full spectra, but the quadrature measures induced by Ritz values
and weights from independent fixed-operator Lanczos runs.
The histogram estimator used for SD-SLQ is obtained by binning this measure. For a bin Ib=[ab,ab+1)I\_{b}=[a\_{b},a\_{b+1}) of width
Δb=ab+1−ab\Delta\_{b}=a\_{b+1}-a\_{b}, we set

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ^hist​(α)=NRSLQ​Δb​∑i=1RSLQ∑ℓ=1mωi,ℓ​𝟏​{α~i,ℓ∈Ib},α∈Ib.\widehat{\rho}\_{\mathrm{hist}}(\alpha)=\frac{N}{R\_{\mathrm{SLQ}}\,\Delta\_{b}}\sum\_{i=1}^{R\_{\mathrm{SLQ}}}\sum\_{\ell=1}^{m}\omega\_{i,\ell}\mathbf{1}\{\widetilde{\alpha}\_{i,\ell}\in I\_{b}\},\qquad\alpha\in I\_{b}. |  | (87) |

The smoothed estimator used for SD-KDE replaces the bin indicator by a kernel KhK\_{h} with bandwidth hh:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ρ^kde​(α)=NRSLQ​∑i=1RSLQ∑ℓ=1mωi,ℓ​Kh​(α−α~i,ℓ).\widehat{\rho}\_{\mathrm{kde}}(\alpha)=\frac{N}{R\_{\mathrm{SLQ}}}\sum\_{i=1}^{R\_{\mathrm{SLQ}}}\sum\_{\ell=1}^{m}\omega\_{i,\ell}K\_{h}(\alpha-\widetilde{\alpha}\_{i,\ell}). |  | (88) |

To compute the Sharpness Dimension estimators, we apply the one-step Jacobian map

|  |  |  |  |
| --- | --- | --- | --- |
|  | gη​(α)=log⁡(|1−η​α|+ε)g\_{\eta}(\alpha)=\log\!\left(|1-\eta\alpha|+\varepsilon\right) |  | (89) |

to the Ritz values and estimate the corresponding push-forward density. SD-SLQ and SD-KDE apply the Sharpness Dimension
formula to the histogram or smoothed density of these transformed values, respectively. SD-PS, described next, uses the same
smoothed-density viewpoint but first converts the density into equal-mass pseudo-eigenvalues before applying the
finite-dimensional weighted sharpness dimension formula.

#### Pseudo-spectrum SD (SD-PS)

We also report a pseudo-spectrum version of the SLQ estimator, denoted SD-PS, which discretizes the smoothed spectral density before applying the same sharpness-dimension formula as in Dfn. [4](#S4.SS0.SSS0.Px4 "Complexity meassure of Random Attractors ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"). Let ρ^​(α)\widehat{\rho}(\alpha) be the smoothed SLQ estimate of the Hessian spectral density for a model with NN parameters, normalized so that ∫ρ^​(α)​𝑑α=N\int\widehat{\rho}(\alpha)\,d\alpha=N. We form its cumulative mass function F^​(α)=∫−∞αρ^​(s)​𝑑s\widehat{F}(\alpha)=\int\_{-\infty}^{\alpha}\widehat{\rho}(s)\,ds and choose equal-mass quantiles

|  |  |  |  |
| --- | --- | --- | --- |
|  | α^r:=F^−1​(r+1/2M​N),r=0,…,M−1,\widehat{\alpha}\_{r}:=\widehat{F}^{-1}\!\left(\frac{r+1/2}{M}N\right),\qquad r=0,\ldots,M-1, |  | (90) |

where M≤NM\leq N is the number of pseudo-eigenvalues used in the discretization. Each pseudo-eigenvalue carries weight N/MN/M. We then map these values through the one-step GD Jacobian spectrum,

|  |  |  |  |
| --- | --- | --- | --- |
|  | z^r=log⁡(|1−η​α^r|+ε),\widehat{z}\_{r}=\log\!\left(|1-\eta\widehat{\alpha}\_{r}|+\varepsilon\right), |  | (91) |

sort the values and reindex them so that z^0≥⋯≥z^M−1\widehat{z}\_{0}\geq\cdots\geq\widehat{z}\_{M-1}, and compute the weighted Sharpness Dimension:
if jj is the largest index for which ∑r=0j(N/M)​z^r≥0\sum\_{r=0}^{j}(N/M)\widehat{z}\_{r}\geq 0, then

|  |  |  |  |
| --- | --- | --- | --- |
|  | dim^SPS=∑r=0jNM+∑r=0j(N/M)​z^r|z^j+1|,\widehat{\dim}\_{\mathrm{S}}^{\mathrm{PS}}=\sum\_{r=0}^{j}\frac{N}{M}+\frac{\sum\_{r=0}^{j}(N/M)\widehat{z}\_{r}}{|\widehat{z}\_{j+1}|}, |  | (92) |

with the usual truncation to [0,N][0,N] and the boundary conventions from Dfn. [4](#S4.SS0.SSS0.Px4 "Complexity meassure of Random Attractors ‣ 4 Theoretical Results ‣ Generalization at the Edge of Stability"). In our
GPT-2 experiments we use this estimator as an alternative discretization of the same smoothed SLQ measure used for
SD-KDE. Thus SD-PS differs from SD-KDE only in how the density is converted into the ordered log-singular-value
spectrum: SD-KDE integrates the density directly, whereas SD-PS first constructs equal-mass pseudo-eigenvalues and then
applies the finite-dimensional weighted Sharpness Dimension formula.

#### Complexity analysis

Regarding the SD computation, for a network with NN parameters and kk minibatches, assume the cost of one Hessian–vector product (HVP) is ChvpC\_{\text{hvp}}. Explicit Hessian computation has memory complexity Θ​(N2)\Theta(N^{2}) and worst-case runtime complexity 𝒪​(N⋅Chvp)\mathcal{O}(N\cdot C\_{\text{hvp}}). Singular value or eigenvalue decompositions have memory complexity Θ​(N2)\Theta(N^{2}) and runtime complexity Θ​(N3)\Theta(N^{3}). In this case, since SD computation requires Θ​(N)\Theta(N) memory and runtime, the overall algorithm has 𝒪​(k​N​Chvp+k​N3)\mathcal{O}(kNC\_{\text{hvp}}+kN^{3}) runtime complexity and Θ​(N2)\Theta(N^{2}) memory complexity. In the approximate SLQ variant, assuming mm iterations per Lanczos run, each run has runtime complexity Θ​(m⋅Chvp)\Theta(m\cdot C\_{\text{hvp}}) for HVPs and Θ​(m2​N)\Theta(m^{2}N) for reorthogonalization, with Θ​(m​N)\Theta(mN) memory complexity. Hence, the SLQ-based algorithm has total runtime complexity Θ​(k​m​Chvp+k​m2​N)\Theta(kmC\_{\text{hvp}}+km^{2}N) and memory complexity Θ​(m​N)\Theta(mN). For typical settings where m≪Nm\ll N, both runtime and memory complexity are significantly reduced.

## Appendix D Additional Results

#### Grokking for the 3-layer MLP

Below in Fig [7](#A4.F7 "Figure 7 ‣ Grokking for the 3-layer MLP ‣ Appendix D Additional Results ‣ Generalization at the Edge of Stability") we present another experiment, with different seeds and hyper-parameters in the grokking setting identical to the main paper but for a 3-layer MLP with 32 hidden features instead.
Similarly, we use 100 uniformly spaced checkpoints and ReLU activation, trained with SGD using only weight decay. In particular, we observe an interesting phenomenon in the first plot, where our dimension increases while test accuracy is increasing slowly, sharply decreases while the test accuracy is increasing sharply (grokking), and then increases again afterwards when test accuracy reaches 100%. In contrast, the other measures decrease monotonically. This behaviour aligns with the theoretical motivation of our dimension, which targets the edge-of-stability regime, and accurately reflects the grokking phase transitions. A similar phase transition is visible in Plot 3, while Plot 4 reflects the grokking experiment from the main paper. In Plot 2, focusing on the sharp grokking region reveals a similarly sharp decrease in SD.

!(/html/2604.19740/assets/x7.png)

Figure 7: Grokking analysis for different learning rates (η\eta), weight decay (W​DWD) and seeds 3-layer MLP with ReLU activation and no momentum. Note that the *suddenness*  of the grokking behavior is best captured in the complexity measures we introduce: RDS-Sharpness and Sharpness Dimension (SD).

#### Hessian Spectra & RDS Sharpness Spectrum.

Figure [8](#A4.F8 "Figure 8 ‣ Hessian Spectra & RDS Sharpness Spectrum. ‣ Appendix D Additional Results ‣ Generalization at the Edge of Stability") illustrates our spectral estimators for Hessian spectral density and the RDS sharpness spectrum on GPT-2 across SGD, SGD with momentum, and AdamW. The left image in each pair visualizes the expectation of minibatch Hessian spectral density over eigenvalues α\alpha. The corresponding right images show the push-forward of this density through the transformation α↦log⁡|1−η​α|\alpha\mapsto\log|1-\eta\alpha|, i.e., the transformed spectral density underlying the RDS Sharpnesses of Order kk, λk\lambda\_{k}. We observe that, across all three optimizers, a large fraction of the Hessian spectrum is concentrated near α=0\alpha=0. This mass near α=0\alpha=0 produces a peak near 0 in the RDS sharpness spectrum and corresponds to neutral or nearly neutral directions (i.e., expanding and contracting with almost 0 log-singular values). In contrast, the isolated spikes, positive tails, and negative curvature components of the Hessian spectrum produce the contractive and expansive directions of the RDS Sharpness spectrum. Positive eigenvalues less than 2/η2/\eta yield negative RDS sharpness values, corresponding to the contractive directions. Positive eigenvalues that are larger than 2/η2/\eta and all negative eigenvalues yield positive RDS sharpness values, corresponding to expansive directions instead. Hence, this visualization clarifies why the Sharpness Dimension depends on the full Hessian spectrum rather than only the top Hessian eigenvalue or a small part of it: SD is determined by the balance between the positive RDS sharpness directions, the near-neutral bulk, and the negative tail. Moreover, the estimated Hessian densities are consistent with prior observations on neural-network and transformer Hessian spectra Ghorbani et al., ([2019](#bib.bib28)); Zhang et al., ([2024](#bib.bib77)), providing additional evidence that our SLQ-based procedure captures the relevant spectral structure.

!(/html/2604.19740/assets/x8.png)

Figure 8: Hessian Spectra & RDS Sharpness Spectrum. For selected GPT-2 runs trained with (a) SGD, (b) SGD with momentum, and (c) AdamW, we show the SLQ-based histogram estimate (SD-SLQ) and the kernel-smoothed estimate (SD-KDE) for both the raw Hessian spectrum (left in each pair) and the transformed RDS sharpness spectrum log⁡|1−η​α|\log|1-\eta\alpha| (right in each pair). Together, the panels show how optimizer and hyperparameter choices affect both the Hessian spectrum and its induced RDS sharpness spectrum. These examples show that the SLQ histogram and the corresponding kernel-smoothed estimate provide consistent views of the underlying spectrum across a range of training configurations.
