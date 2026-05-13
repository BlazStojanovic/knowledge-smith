---
arxiv: '2505.10860'
authors:
- Huy Nguyen
- Thong T. Doan
- Quang Pham
- Nghi D. Q. Bui
- Nhat Ho
- Alessandro Rinaldo
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'On DeepSeekMoE: Statistical Benefits of Shared Experts and Normalized Sigmoid
  Gating'
url: https://arxiv.org/abs/2505.10860
year: 2025
---

[2505.10860] 1 Introduction














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



On DeepSeekMoE: Statistical Benefits of Shared Experts
and Normalized Sigmoid Gating

  

|  |  |  |
| --- | --- | --- |
| Huy Nguyen† | Thong T. Doan⋄ | Quang Pham‡ |
| Nghi D. Q. Bui⋄ | Nhat Ho†,⋆ | Alessandro Rinaldo†,⋆ |

  

|  |
| --- |
| †The University of Texas at Austin |
| ⋄FPT Software AI Center |
| ‡ Independent Researcher |

  

June 5, 2025

###### Abstract

Mixture of experts (MoE) methods
are a key component in most large language model architectures, including the recent series of DeepSeek models. Compared to other MoE implementations, DeepSeekMoE stands out because of two unique features: the deployment of a shared expert strategy and of the normalized sigmoid gating mechanism. Despite the prominent role of DeepSeekMoE in the success of the DeepSeek series of models, there have been only a few attempts to justify theoretically the value of the shared expert strategy, while its normalized sigmoid gating has remained unexplored.
To bridge this gap, we undertake a comprehensive theoretical study of these two features of DeepSeekMoE from a statistical perspective. We perform a convergence analysis of the expert estimation task to highlight the gains in sample efficiency for both the shared expert strategy and the normalized sigmoid gating, offering useful insights into the design of expert and gating structures. To verify empirically our theoretical findings, we carry out several experiments on both synthetic data and real-world datasets for (vision) language modeling tasks. Finally, we conduct an extensive empirical analysis of the router behaviors, ranging from router saturation, router change rate, to expert utilization.

††footnotetext: ⋆\star Co-last authors.

## 1 Introduction

The recent years have witnessed a dramatic increase in the use and success of of deep learning models, leading to remarkable advances in a variety of fields, namely natural language processing [[30](#bib.bib30), [18](#bib.bib18), [20](#bib.bib20), [38](#bib.bib38)], computer vision [[62](#bib.bib62), [41](#bib.bib41)], multimodal learning [[23](#bib.bib23), [79](#bib.bib79)], and reinforcement learning [[4](#bib.bib4), [10](#bib.bib10)]. However, this trend has also introduced several challenges in terms of computational efficiency. One common approach to tackle this challenge is to leverage Mixture-of-Experts (MoE) architecture, which allows to scale up the model capacity without a proportional increase in computation.

Originally proposed by Jacob et al. [[28](#bib.bib28)], MoE has been known as a form of ensemble learning that combines the power of several individual models through an adaptive gating network.
In particular, these individual models are termed experts and can be formulated as classifiers [[8](#bib.bib8), [52](#bib.bib52)], regression models [[19](#bib.bib19), [34](#bib.bib34)], or feed-forward networks (FFNs) [[64](#bib.bib64), [12](#bib.bib12)]. Meanwhile, the gating network is responsible for dynamically assigning input-dependent softmax weights to experts based on their specialization in the input domain.
Then, to improve the scalability of MoE, Shazeer et al. [[64](#bib.bib64)] have recently introduced a sparse version of MoE which activates only a subset of specialized experts per input, allowing to increase the number of trainable parameters while keeping the computation overhead nearly unchanged. As a result, there has been a surge of interest in employing the sparse MoE architecture in several large-scale applications, particularly language modeling [[15](#bib.bib15), [14](#bib.bib14), [22](#bib.bib22), [69](#bib.bib69), [60](#bib.bib60)].

Despite their widespread use in large language models, the sparse MoE architecture faces the challenge of knowledge redundancy, that is, multiple experts may end up acquiring overlapping knowledge, leading to the redundancy of expert parameters. In response to this issue, Dai et al. [[12](#bib.bib12)] have come up with a novel DeepSeekMoE framework that divides the set of experts into two disjoint subsets. Experts in the first subset are referred to as shared experts and are always activated to capture common knowledge across different domains. On the other hand, only few experts in the second subset, called routed experts, are activated, typically via a sparse softmax gating mechanism to learn specialized knowledge. This shared expert strategy helps enhance expert specialization by encouraging experts to specialize in distinctive aspects of the data, thereby alleviating the parameter redundancy problem. The new DeepSeekMoE architecture has been adopted as a vital component in the series of high-performing DeepSeek language models, most notably DeepSeek-V2 [[14](#bib.bib14)] which uses sparse softmax gating in the DeepSeekMoE framework, and DeepSeek-V3 [[15](#bib.bib15)], which employs a sparse normalized sigmoid gating. Surprisingly, the shared expert strategy has only been briefly investigated in [[12](#bib.bib12)] from the perspective of expert specialization, while there have been no studies on the benefits of the normalized sigmoid gating.

Contributions. The primary goal of this paper is to provide a comprehensive theoretical study of these two distinguishing features of DeepSeekMoE. Below we perform a convergence analysis of the task of parameter estimation in order to examine the sample efficiency of the shared expert strategy, that is the rate, as a function of the number of data points, at which each expert to specialize in some aspects of the data. Furthermore, we also compare the sample efficiency of the normalized sigmoid gating used in the DeepSeek-V3 model to that of the softmax gating used in the DeepSeek-V2 model.
Our contributions are threefold and can be summarized as follows.

*(1) Sample efficiency of the shared expert strategy.* Our analysis in Section [2](#S2 "2 On Shared Expert Strategy")
reveals that shared experts admit significantly faster convergence rates than routed experts and experts in MoE models without the shared expert strategy, whose rates depend in a complicated manner on the solvability of certain systems of polynomial equations as well as the number of fitted experts (see Table [1](#S1.T1 "Table 1 ‣ 1 Introduction")). As a result, a smaller amount of data are required to approximate shared experts compared to non-shared experts in DeepSeekMoE and standard MoE models to achieve the same level of statistical accuracy.

*(2) Sample efficiency of the normalized sigmoid gating.* Similarly, when using the normalized sigmoid gating instead of the softmax gating, the convergence rates of routed experts no longer hinge on the solvability of a system of polynomial equations and, therefore, are provably faster than those of shared experts, which remain unchanged in this setting (see also Table [1](#S1.T1 "Table 1 ‣ 1 Introduction")). Thus, the amount of data required to estimate routed experts within a given error decreases substantially, demonstrating the sample efficiency of the normalized sigmoid gating over the standard softmax gating. Due to space limitations and the technical nature of this analysis, we present these results in Appendix [A](#A1 "Appendix A On Normalized Sigmoid Gating").

*(3) Empirical validation.* To validate our theoretical findings, we conduct extensive numerical experiments on simulated and real-world data. The experimental results on synthetic data are in very close agreement with our theoretical findings about the convergence rates of the shared expert strategy and the normalized sigmoid gating; see Appendix [G](#A7 "Appendix G Additional Experiments") for detailed results. The experiments summarized in Section [3](#S3 "3 Experiments") on language modeling and vision-language modeling further demonstrate the applicability of our theoretical insights in real-world scenarios. Finally, we perform a detailed analysis of router behavior in Section [3.3](#S3.SS3 "3.3 Router Analysis ‣ 3 Experiments"), providing further insights into the contribution and dynamics of each component of the DeepSeekMoE architecture.

Notation. For any n∈ℕn\in\mathbb{N}, we let [n]={1,2,…,n}[n]=\{1,2,\ldots,n\}. For any vectors v:=(vi)i=1d∈ℝdv:=(v\_{i})\_{i=1}^{d}\in\mathbb{R}^{d} and α:=(αi)i=1d∈ℕd\alpha:=(\alpha\_{i})\_{i=1}^{d}\in\mathbb{N}^{d}, we denote vα:=∏i=1dviαiv^{\alpha}:=\prod\_{i=1}^{d}v\_{i}^{\alpha\_{i}}, |v|:=∑i=1dvi|v|:=\sum\_{i=1}^{d}v\_{i} and α!:=∏i=1dαi!\alpha!:=\prod\_{i=1}^{d}\alpha\_{i}!, while ‖v‖\|v\| represents the ℓ2\ell\_{2}-norm of vv. The cardinality of a set SS is denoted with |S||S|. Finally, for any two positive sequences (an)n≥1(a\_{n})\_{n\geq 1} and (bn)n≥1(b\_{n})\_{n\geq 1}, we write an=𝒪​(bn)a\_{n}=\mathcal{O}(b\_{n}) or an≲bna\_{n}\lesssim b\_{n} if an≤C​bna\_{n}\leq Cb\_{n} for all n∈ℕn\in\mathbb{N}, for some constant C>0C>0. For a sequence (An)n≥1(A\_{n})\_{n\geq 1} of positive random variables, the notation An=𝒪P​(bn)A\_{n}=\mathcal{O}\_{P}(b\_{n}) signifies An/bnA\_{n}/b\_{n} is stochastically bounded, that is, for any ϵ>0\epsilon>0, there exists an M>0M>0 such that ℙ​(An/bn>M)<ϵ\mathbb{P}(A\_{n}/b\_{n}>M)<\epsilon for all nn large enough. We further write An=𝒪~P​(bn)A\_{n}=\widetilde{\mathcal{O}}\_{P}(b\_{n}) when An=𝒪P​(bn​logc⁡(bn))A\_{n}=\mathcal{O}\_{P}(b\_{n}\log^{c}(b\_{n})), for some c>0c>0. Finally, for two Lebesgue probability densities on ℝd\mathbb{R}^{d}, f1f\_{1} and f2f\_{2}, V​(f1,f2):=12​∫|f1​(y)−f2​(y)|​𝑑yV(f\_{1},f\_{2}):=\frac{1}{2}\int|f\_{1}(y)-f\_{2}(y)|dy denotes their total variation distance.

Table 1: Summary of expert estimation rates in DeepSeek-V2’s MoE with softmax gating (Section [2](#S2 "2 On Shared Expert Strategy")) and DeepSeek-V3’s MoE with normalized sigmoid gating (Appendix [A](#A1 "Appendix A On Normalized Sigmoid Gating")). Below, the function r2r\_{2} stands for the solvability of certain systems of polynimial equations specified in Appendix [B](#A2 "Appendix B Systems of Polynomial Equations"), while the notation 𝒱2,j\mathcal{V}\_{2,j} denotes a Voronoi cell defined in equation ([2.1](#S2.Ex4 "2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy")). For the normalized sigmoid gating setting, we consider two complementary parameter settings, namely sparse regime and dense regime (see Appendix [A](#A1 "Appendix A On Normalized Sigmoid Gating") for further details).

|  |  |  |
| --- | --- | --- |
| DeepSeek-V2’s MoE | ReLU FFN Experts | Linear Experts |
| Shared Experts | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) |
| Routed Experts | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) | 𝒪~P​(n−1/r2​(|𝒱2,j|))\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2}(|\mathcal{V}\_{2,j}|)}) |



|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| DeepSeek-V3’s MoE | ReLU FFN Experts | | Linear Experts | |
| Sparse Regime | Dense Regime | Sparse Regime | Dense Regime |
| Shared Experts | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) | | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) | |
| Routed Experts | 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) | 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) | 𝒪~P​(n−1/r2​(|𝒱2,j|))\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2}(|\mathcal{V}\_{2,j}|)}) | 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) |

## 2 On Shared Expert Strategy

Below we derive convergence rates for the shared expert estimation problem in the DeepSeekMoE architecture. For ease of presentation, we will focus here on the dense DeepSeekMoE case, and analyze the less popular sparse DeepSeekMoE settings in Appendix [F](#A6 "Appendix F Extended Theoretical Results for Sparse Gating MoE"). After formally introducing the settings, we formulate a *strong identifiability* condition on the expert functions ensuring fast expert convergence rates. We then turn to linear experts, which violate the strong identifiability condition, and prove that, in fact, they exhibit slow rates of convergence.

Problem setting. Assume that (X1,Y1),(X2,Y2),…,(Xn,Yn)∈ℝd×ℝ(X\_{1},Y\_{1}),(X\_{2},Y\_{2}),\ldots,(X\_{n},Y\_{n})\in\mathbb{R}^{d}\times\mathbb{R} are i.i.d. samples drawn from a Gaussian DeepSeekMoE model, whose conditional density function fG1∗,G2∗​(y|x)f\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x) is given by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fG1∗,G2∗​(y|x):=\displaystyle f\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x):= | 12​∑i=1k1∗ωi∗​π​(Y|h1​(x,κi∗),τi∗)+12​∑i=1k2∗exp⁡((β1​i∗)⊤​x+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​x+β0​j∗)​π​(y|h2​(x,ηi∗),νi∗).\displaystyle\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|h\_{1}(x,\kappa^{\*}\_{i}),\tau^{\*}\_{i})+\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{2}}\frac{\exp((\beta\_{1i}^{\*})^{\top}x+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}x+\beta\_{0j}^{\*})}\pi(y|h\_{2}(x,\eta^{\*}\_{i}),\nu\_{i}^{\*}). |  | (1) |

Above, π(⋅|μ,ν)\pi(\cdot|\mu,\nu) denotes the Gaussian density function with mean μ\mu and variance ν\nu, h1​(⋅,κi∗)h\_{1}(\cdot,\kappa^{\*}\_{i}) and h2​(⋅,ηi∗)h\_{2}(\cdot,\eta^{\*}\_{i}) are real-valued functions on ℝd\mathbb{R}^{d} referred to as shared and routed experts, respectively.
The weight parameters ω1∗,ω2∗,…,ωk1∗∗\omega^{\*}\_{1},\omega^{\*}\_{2},\ldots,\omega^{\*}\_{k^{\*}\_{1}} are positive and satisfy ∑i=1k1∗ωi∗=1\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}=1.
We conveniently represent all the model parameters with the *mixing measures* G1∗:=∑i=1k1∗ωi∗​δ(κi∗,τi∗)G^{\*}\_{1}:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\delta\_{(\kappa^{\*}\_{i},\tau^{\*}\_{i})} and G2∗:=∑i=1k2∗exp⁡(β0​i∗)​δ(β1​i∗,ηi∗,νi∗)G^{\*}\_{2}:=\sum\_{i=1}^{k^{\*}\_{2}}\exp(\beta\_{0i}^{\*})\delta\_{(\beta\_{1i}^{\*},\eta\_{i}^{\*},\nu\_{i}^{\*})}, a combination of Dirac δ\delta-measures with mass on the unknown true parameters θ1​i∗:=(ωi∗,κi∗,τi∗)\theta^{\*}\_{1i}:=(\omega^{\*}\_{i},\kappa^{\*}\_{i},\tau^{\*}\_{i}) in Θ1⊆ℝ×ℝd1×ℝ+\Theta\_{1}\subseteq\mathbb{R}\times\mathbb{R}^{d\_{1}}\times\mathbb{R}\_{+} and θ2​i∗:=(β0​i∗,β1​i∗,ηi∗,νi∗)\theta^{\*}\_{2i}:=(\beta^{\*}\_{0i},\beta^{\*}\_{1i},\eta^{\*}\_{i},\nu^{\*}\_{i}) in Θ2⊆ℝ×ℝd×ℝd2×ℝ+\Theta\_{2}\subseteq\mathbb{R}\times\mathbb{R}^{d}\times\mathbb{R}^{d\_{2}}\times\mathbb{R}\_{+}, respectively.
Thus, our goal is to estimate the pair of ground-truth mixing measures (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}).

Maximum likelihood estimation (MLE). As the numbers k1∗k^{\*}\_{1} and k2∗k^{\*}\_{2} of shared and routed experts are unknown, we consider the ground-truth model ([1](#S2.E1 "In 2 On Shared Expert Strategy")) with up to k1>k1∗k\_{1}>k^{\*}\_{1} shared experts and k2>k2∗k\_{2}>k^{\*}\_{2} routed experts. Towards that goal, we let 𝒢k1,k2​(Θ):=𝒢k1​(Θ1)×𝒢k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):=\mathcal{G}\_{k\_{1}}(\Theta\_{1})\times\mathcal{G}\_{k\_{2}}(\Theta) stands for the set of mixing measure pairs (G1,G2)(G\_{1},G\_{2}) with at most k1k\_{1} and k2k\_{2} atoms, respectively; that is 𝒢k1​(Θ1):={G1=∑i=1k1′ωi​δ(κi,τi):1≤k1′≤k1}\mathcal{G}\_{k\_{1}}(\Theta\_{1}):=\Big{\{}G\_{1}=\sum\_{i=1}^{k^{\prime}\_{1}}\omega\_{i}\delta\_{(\kappa\_{i},\tau\_{i})}:1\leq k^{\prime}\_{1}\leq k\_{1}\Big{\}} and 𝒢k2​(Θ2):={G2=∑i=1k2′exp⁡(β0​i)​δ(β1​i,ηi∗,νi∗):1≤k2′≤k2}\mathcal{G}\_{k\_{2}}(\Theta\_{2}):=\Big{\{}G\_{2}=\sum\_{i=1}^{k^{\prime}\_{2}}\exp(\beta\_{0i})\delta\_{(\beta\_{1i},\eta^{\*}\_{i},\nu^{\*}\_{i})}:1\leq k^{\prime}\_{2}\leq k\_{2}\Big{\}}.
Our final estimator is the MLE over 𝒢k1,k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta), i.e.

|  |  |  |  |
| --- | --- | --- | --- |
|  | (G^1n,G^2n)∈arg​max(G1,G2)∈𝒢k1,k2​(Θ)⁡1n​∑i=1nlog⁡(fG1,G2​(Yi|Xi)),\displaystyle(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2})\in\operatorname\*{arg\,max}\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)}\frac{1}{n}\sum\_{i=1}^{n}\log(f\_{G\_{1},G\_{2}}(Y\_{i}|X\_{i})), |  | (2) |

Universal assumptions. For our theoretical analysis, we impose the following three mild assumptions on the ground-truth parameters throughout the paper.

*(A.1) The parameter space Θ\Theta is compact with fixed dimension, while the input space 𝒳\mathcal{X} is bounded.*

*(A.2) The last pair of gating parameters vanish, that is, β1​k2∗∗=0d\beta^{\*}\_{1k^{\*}\_{2}}=0\_{d} and β0​k2∗∗=0\beta^{\*}\_{0k^{\*}\_{2}}=0 (to avoid non-identifiability due to invariance to translation of the softmax gating function). In addition, at least one among parameters {β1​i∗,i∈[k2∗]}\{\beta^{\*}\_{1i},i\in[k^{\*}\_{2}]\}, is non-zero (to maintain the dependence of the gating on the input value).*

*(A.3) The expert parameters (κi∗)i=1k1∗(\kappa^{\*}\_{i})\_{i=1}^{k^{\*}\_{1}} and (ηi∗)i=1k2∗(\eta^{\*}\_{i})\_{i=1}^{k^{\*}\_{2}} are distinct. Meanwhile, the expert functions h1​(⋅,κ)h\_{1}(\cdot,\kappa) and h2​(⋅,η)h\_{2}(\cdot,\eta) are bounded and Lipschitz continuous w.r.t κ\kappa and η\eta.*

Equipped with these assumptions, we are now ready to give our first consistency result for the ground-truth conditional density fG1∗,G2∗f\_{G^{\*}\_{1},G^{\*}\_{2}}.

###### Proposition 1.

The maximum likelihood density estimator fG^1n,G^2n​(Y|X)f\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(Y|X) converges to the true density fG1∗,G2∗​(Y|X)f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) in total variation distance at the rate

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(fG^1n,G^2n(⋅|X),fG1∗,G2∗(⋅|X))]=𝒪P([log(n)/n]12).\displaystyle\mathbb{E}\_{X}[V(f\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

The above result, whose proof can be found in Appendix [E.1](#A5.SS1 "E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results"), shows that the true density function fG1∗,G2∗​(y|x)f\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x) can be estimated at a rate that is nearly parametric. Following a strategy used in the analysis of MoE models [[56](#bib.bib56)], if one can exhibit an appropriate loss function over the mixing measures, say 𝒟​((G1,G2),(G2∗,G2∗))\mathcal{D}((G\_{1},G\_{2}),(G^{\*}\_{2},G^{\*}\_{2})), that, up to constant, is a lower bound on 𝔼X[V(fG^1n,G^2n(⋅|X),fG1∗,G2∗(⋅|X))]\mathbb{E}\_{X}[V(f\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))], Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy") will then imply a near parametric rate also for the parameters and expert functions themselves. However, the derivation of this lower bound is challenging. Specifically, a key step in establishing the aforementioned lower bound is to decompose the difference fG^1n,G^2n​(Y|X)−fG1∗,G2∗​(Y|X)f\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) through a series of Taylor expansions of the functions x↦π​(Y|h1​(x,κ),τ)x\mapsto\pi(Y|h\_{1}(x,\kappa),\tau) and x↦F​(Y|X;β1,η,ν):=exp⁡(β1⊤​x)​π​(Y|h2​(x,η),ν)x\mapsto F(Y|X;\beta\_{1},\eta,\nu):=\exp(\beta\_{1}^{\top}x)\pi(Y|h\_{2}(x,\eta),\nu) w.r.t their parameters (κ,τ)(\kappa,\tau) and (β1,η,ν)(\beta\_{1},\eta,\nu), respectively. When the difference of the densities converges to zero (as ensured by Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy")), then one may expect the coefficients of this Taylor expansions, which correspond to the difference between the true and estimated parameters, will also vanish. However, this is true only provided that these functions and their partial derivatives arising from the Taylor expansions remain linearly independent. To ensure that this property holds, we formulate a new, non-trivial condition, called *strong identifiability* for the expert functions h1h\_{1} and h2h\_{2}.

###### Definition 1 (Strong Identifiability).

We say that the expert functions x↦h1​(x,κ)x\mapsto h\_{1}(x,\kappa) and x↦h2​(x,η)x\mapsto h\_{2}(x,\eta) are strongly identifiable if they are twice differentiable w.r.t their parameters κ\kappa and η\eta, and if for any k1,k2≥1k\_{1},k\_{2}\geq 1, κ1,…,κk1\kappa\_{1},\ldots,\kappa\_{k\_{1}} and η1,…,ηk2\eta\_{1},\ldots,\eta\_{k\_{2}}, each of the sets

|  |  |  |
| --- | --- | --- |
|  | {∂h1∂κ(u1)​(x,κi):i∈[k1],u1∈[d1]},{∂h1∂κ(u1)​(x,κi)​∂h1∂κ(v1)​(x,κi),1:i∈[k1],u1,v1∈[d1]},\displaystyle\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(x,\kappa\_{i}):i\in[k\_{1}],\ u\_{1}\in[d\_{1}]\Bigg{\}},\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(x,\kappa\_{i})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(x,\kappa\_{i}),1:i\in[k\_{1}],\ u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |
| --- | --- | --- |
|  | {∂h2∂η(u2)​(x,ηj),∂2h2∂η(u2)​∂η(v2)​(x,ηj),x(u)​∂h2∂η(v2)​(x,ηj):j∈[k2],u2,v2∈[d2],u∈[d]}\displaystyle\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(x,\eta\_{j}),\ \frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(x,\eta\_{j}),\ x^{(u)}\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(x,\eta\_{j}):j\in[k\_{2}],\ u\_{2},v\_{2}\in[d\_{2}],\ u\in[d]\Bigg{\}} |  |

consists of linearly independent functions (in xx).

Examples. Two-layer FFNs h1​(x,(κ2,κ1,κ0)):=κ2​ReLU​(κ1⊤​x+κ0)h\_{1}(x,(\kappa\_{2},\kappa\_{1},\kappa\_{0})):=\kappa\_{2}\mathrm{ReLU}(\kappa\_{1}^{\top}x+\kappa\_{0}) and h2​(x,(η2,η1)):=η2​GELU​(η1⊤​x)h\_{2}(x,(\eta\_{2},\eta\_{1})):=\eta\_{2}\mathrm{GELU}(\eta\_{1}^{\top}x) are strongly identifiable. The same claim holds when replacing the ReLU\mathrm{ReLU} function with other activation functions such as sigmoid\mathrm{sigmoid} and tanh\tanh. On the other hand, linear experts h1​(x,(κ1,κ0)):=κ1⊤​x+κ0h\_{1}(x,(\kappa\_{1},\kappa\_{0})):=\kappa\_{1}^{\top}x+\kappa\_{0} and h2​(x,(η1,η0)):=η1⊤​x+η0h\_{2}(x,(\eta\_{1},\eta\_{0})):=\eta\_{1}^{\top}x+\eta\_{0} fail to satisfy the strong identifiability condition because ∂h1∂κ0​∂h1∂κ0=1\frac{\partial h\_{1}}{\partial\kappa\_{0}}\frac{\partial h\_{1}}{\partial\kappa\_{0}}=1 and ∂h2∂η1=x​∂h2∂η0\frac{\partial h\_{2}}{\partial\eta\_{1}}=x\frac{\partial h\_{2}}{\partial\eta\_{0}} for all xx.

### 2.1 Strongly Identifiable Experts

Our next task is to construct a loss over pairs of mixing measures (G1,G2)(G\_{1},G\_{2}) and (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}). To this end, let us revisit the concepts of Voronoi cells and Voronoi loss function presented in [[49](#bib.bib49)].

Voronoi loss. For any pair of mixing measures (G1,G2)(G\_{1},G\_{2}) with k1′≤k1k^{\prime}\_{1}\leq k\_{1} and k2′≤k2k^{\prime}\_{2}\leq k\_{2} atoms, we distribute their atoms to the Voronoi cells 𝒱1,j1≡𝒱1,j1​(G)\mathcal{V}\_{1,j\_{1}}\equiv\mathcal{V}\_{1,j\_{1}}(G) and 𝒱2,j2≡𝒱2,j2​(G)\mathcal{V}\_{2,j\_{2}}\equiv\mathcal{V}\_{2,j\_{2}}(G), defined as

|  |  |  |
| --- | --- | --- |
|  | 𝒱1,j1:={i1∈[k1′]:‖ξi1−ξj1∗‖≤‖ξi1−ξℓ1∗‖,∀ℓ1≠j1},\displaystyle\mathcal{V}\_{1,j\_{1}}:=\{i\_{1}\in[k^{\prime}\_{1}]:\|\xi\_{i\_{1}}-\xi^{\*}\_{j\_{1}}\|\leq\|\xi\_{i\_{1}}-\xi^{\*}\_{\ell\_{1}}\|,\ \forall\ell\_{1}\neq j\_{1}\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒱2,j2:={i2∈[k2′]:‖ζi2−ζj2∗‖≤‖ζi2−ζℓ2∗‖,∀ℓ2≠j2},\displaystyle\mathcal{V}\_{2,j\_{2}}:=\{i\_{2}\in[k^{\prime}\_{2}]:\|\zeta\_{i\_{2}}-\zeta^{\*}\_{j\_{2}}\|\leq\|\zeta\_{i\_{2}}-\zeta^{\*}\_{\ell\_{2}}\|,\ \forall\ell\_{2}\neq j\_{2}\}, |  | (3) |

where we denote ξi1:=(κi1,τi1)\xi\_{i\_{1}}:=(\kappa\_{i\_{1}},\tau\_{i\_{1}}), ξj1∗:=(κj1∗,τj1∗)\xi^{\*}\_{j\_{1}}:=(\kappa^{\*}\_{j\_{1}},\tau^{\*}\_{j\_{1}}) for all j1∈[k1∗]j\_{1}\in[k^{\*}\_{1}], and ζi2:=(β1​i2,ηi2,νi2)\zeta\_{i\_{2}}:=(\beta\_{1i\_{2}},\eta\_{i\_{2}},\nu\_{i\_{2}}), ζj2∗:=(β1​j2∗,ηj2∗,νj2∗)\zeta^{\*}\_{j\_{2}}:=(\beta^{\*}\_{1j\_{2}},\eta^{\*}\_{j\_{2}},\nu^{\*}\_{j\_{2}}) for all j2∈[k2∗]j\_{2}\in[k^{\*}\_{2}]. Then, the proposed Voronoi loss over mixing measures is

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟1​((G1,G2),(G1∗,G2∗)):=∑j=1k1∗|∑i∈𝒱1,jωi−ωj∗|+∑j=1k2∗|∑i∈𝒱2,jexp⁡(β0​i)−exp⁡(β0​j∗)|\displaystyle\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})):=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}-\omega\_{j}^{\*}\Big{|}+\sum\_{j=1}^{k^{\*}\_{2}}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|=1∑i∈𝒱1,jωi​(‖Δ​κi​j‖+|Δ​τi​j|)+∑j∈[k2∗],|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​i)​(‖Δ​β1​i​j‖+‖Δ​ηi​j‖+|Δ​νi​j|)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|+|\Delta\tau\_{ij}|)+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}],\\ |\mathcal{V}\_{2,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})(\|\Delta\beta\_{1ij}\|+\|\Delta\eta\_{ij}\|+|\Delta\nu\_{ij}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|>1ωi​(‖Δ​κi​j‖2+|Δ​τi​j|2)+∑j∈[k2∗],|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​i)​(‖Δ​β1​i​j‖2+‖Δ​ηi​j‖2+|Δ​νi​j|2),\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\omega\_{i}(\|\Delta\kappa\_{ij}\|^{2}+|\Delta\tau\_{ij}|^{2})+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}],\\ |\mathcal{V}\_{2,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})(\|\Delta\beta\_{1ij}\|^{2}+\|\Delta\eta\_{ij}\|^{2}+|\Delta\nu\_{ij}|^{2}), |  | (4) |

where we let Δ​κi​j:=κi−κj∗\Delta\kappa\_{ij}:=\kappa\_{i}-\kappa\_{j}^{\*}, Δ​τi​j:=τi−τj∗\Delta\tau\_{ij}:=\tau\_{i}-\tau\_{j}^{\*}, Δ​β1​i​j:=β1​i−β1​j∗\Delta\beta\_{1ij}:=\beta\_{1i}-\beta\_{1j}^{\*}, Δ​ηi​j:=ηi−ηj∗\Delta\eta\_{ij}:=\eta\_{i}-\eta\_{j}^{\*}, and Δ​νi​j:=νi−νj∗\Delta\nu\_{ij}:=\nu\_{i}-\nu\_{j}^{\*}. It is clear that convergence of the mixing measures in the 𝒟1\mathcal{D}\_{1} loss is equivalent to convergence of their respective parameters. Thus, though not a metric over mixing measures, the 𝒟1\mathcal{D}\_{1} loss can be used to characterize parameter and expert estimation rates.

###### Theorem 1.

Assume that the expert functions h1h\_{1} and h2h\_{2} are strongly identifiable. Then, the lower bound 𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]≳𝒟1((G1,G2),(G1∗,G2∗))\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))
holds for all (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). As a consequence,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒟1​((G^1n,G^2n),(G1∗,G2∗))=𝒪P​([log⁡(n)/n]12).\displaystyle\mathcal{D}\_{1}((\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  | (5) |

The combination of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy"), whose proof is in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), and of the form of the loss 𝒟1\mathcal{D}\_{1} leads to various estimation rates. Below we say that a parameter is exactly-specified or over-specified depending on whether the associated Voronoi cell has one or more elements, respectively.

*(i) Shared experts.* For shared experts, we see that the estimation rate for exactly-specified parameters κj∗\kappa^{\*}\_{j}, τj∗\tau^{\*}\_{j}, is nearly parameteric, i.e. of order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}). On the other hand, over-specified parameters κj∗\kappa^{\*}\_{j}, τj∗\tau^{\*}\_{j}, admit slightly slower estimation rates, of order 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}). As for the expert estimation rates, since the shared expert function h1​(⋅,κ)h\_{1}(\cdot,\kappa) is Lipschitz continuous, we have that |h1​(x,κ^in)−h1​(x,κj∗)|≲‖κ^in−κj∗‖|h\_{1}(x,\hat{\kappa}^{n}\_{i})-h\_{1}(x,\kappa^{\*}\_{j})|\lesssim\|\hat{\kappa}^{n}\_{i}-\kappa^{\*}\_{j}\| for almost every xx.It then follows that the estimation rates for exactly-specified and over-specified shared experts h1​(x,κj∗)h\_{1}(x,\kappa^{\*}\_{j}) are also of order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) and 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}), respectively. Thus, polynomially many data points 𝒪​(ϵ−2)\mathcal{O}(\epsilon^{-2}) and 𝒪​(ϵ−4)\mathcal{O}(\epsilon^{-4}) are needed to estimate these experts within a error ϵ>0\epsilon>0.

*(ii) Routed experts.* Likewise, exactly-specified and over-specified parameters β1​j∗\beta^{\*}\_{1j}, ηj∗\eta^{\*}\_{j}, νj∗\nu^{\*}\_{j}, for j∈[k2∗]j\in[k^{\*}\_{2}], have estimation rates of order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) and 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}), respectively. As the routed expert function h2​(⋅,η)h\_{2}(\cdot,\eta) is Lipschitz continuous, we deduce that the rates for estimating routed experts h2​(x,ηj∗)h\_{2}(x,\eta^{\*}\_{j}) also vary between 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) and 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) depending on the cardinality of the corresponding Voronoi cell 𝒱2,j\mathcal{V}\_{2,j}. In summary, when both shared and routed expert functions are strongly identifiable, they enjoy the same estimation rates.

### 2.2 Linear Experts

In this section, we consider linear expert functions of the form h1​(X,(κ1,κ0)):=κ1⊤​X+κ0h\_{1}(X,(\kappa\_{1},\kappa\_{0})):=\kappa\_{1}^{\top}X+\kappa\_{0} and h2​(X,(η1,η0)):=η1⊤​X+η0h\_{2}(X,(\eta\_{1},\eta\_{0})):=\eta\_{1}^{\top}X+\eta\_{0}.
Then,
the pair of ground-truth mixing measures (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) become G1∗:=∑i=1k1∗ωi∗​δ(κ1​i∗,κ0​i∗,τi∗)G^{\*}\_{1}:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\delta\_{(\kappa^{\*}\_{1i},\kappa^{\*}\_{0i},\tau^{\*}\_{i})} and G2∗:=∑i=1k2∗exp⁡(β0​i∗)​δ(β1​i∗,η1​i∗,η0​i∗,νi∗)G^{\*}\_{2}:=\sum\_{i=1}^{k^{\*}\_{2}}\exp(\beta\_{0i}^{\*})\delta\_{(\beta\_{1i}^{\*},\eta\_{1i}^{\*},\eta^{\*}\_{0i},\nu\_{i}^{\*})}. As discussed above, linear experts violate the strong identifiability condition due to the PDEs ∂h1∂κ0​∂h1∂κ0=1\frac{\partial h\_{1}}{\partial\kappa\_{0}}\frac{\partial h\_{1}}{\partial\kappa\_{0}}=1 and ∂h2∂η1=x​∂h2∂η0\frac{\partial h\_{2}}{\partial\eta\_{1}}=x\frac{\partial h\_{2}}{\partial\eta\_{0}}. In turn, these PDEs lead to linear dependencies among the partial derivatives of the Gaussian p.d.f. π\pi and of the function FF defined below Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy"), given by ∂2π∂κ02=2​∂π∂τ\frac{\partial^{2}\pi}{\partial\kappa\_{0}^{2}}=2\frac{\partial\pi}{\partial\tau} and ∂F∂η1=∂2F∂β1​∂η0\frac{\partial F}{\partial\eta\_{1}}=\frac{\partial^{2}F}{\partial\beta\_{1}\partial\eta\_{0}}. These delicate relationships, which can be intuitively interpreted as interactions between the parameters κ0\kappa\_{0} and τ\tau, and among the parameters η1\eta\_{1}, β1\beta\_{1} and η0\eta\_{0},
affect the parameter and expert estimation rates. To overcome this issue, we consider instead a new Voronoi loss, given by

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟2​((G1,G2),(G1∗,G2∗)):=∑j=1k1∗|∑i∈𝒱1,jωi−ωj∗|+∑j=1k2∗|∑i∈𝒱2,jexp⁡(β0​i)−exp⁡(β0​j∗)|\displaystyle\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})):=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}-\omega\_{j}^{\*}\Big{|}+\sum\_{j=1}^{k^{\*}\_{2}}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|=1∑i∈𝒱1,jωi​(‖Δ​κ1​i​j‖+|Δ​κ0​i​j|+|Δ​τi​j|)+∑j∈[k1∗],|𝒱1,j|>1∑i∈𝒱1,jωi​(‖Δ​κi​j‖2+|Δ​κ0​i​j|r1,j+|Δ​τi​j|r1,j/2)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{1ij}\|+|\Delta\kappa\_{0ij}|+|\Delta\tau\_{ij}|)+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|^{2}+|\Delta\kappa\_{0ij}|^{r\_{1,j}}+|\Delta\tau\_{ij}|^{r\_{1,j}/2}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​i)​(‖Δ​β1​i​j‖+‖Δ​η1​i​j‖+|Δ​η0​i​j|+|Δ​νi​j|)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})(\|\Delta\beta\_{1ij}\|+\|\Delta\eta\_{1ij}\|+|\Delta\eta\_{0ij}|+|\Delta\nu\_{ij}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​i)​(‖Δ​β1​i​j‖r2,j+‖Δ​η1​i​j‖r2,j/2+|Δ​η0​i​j|r2,j+|Δ​νi​j|r2,j/2),\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i})(\|\Delta\beta\_{1ij}\|^{r\_{2,j}}+\|\Delta\eta\_{1ij}\|^{r\_{2,j}/2}+|\Delta\eta\_{0ij}|^{r\_{2,j}}+|\Delta\nu\_{ij}|^{r\_{2,j}/2}), |  | (6) |

where we denote Δ​κ1​i​j:=κ1​i−κ1​j∗\Delta\kappa\_{1ij}:=\kappa\_{1i}-\kappa^{\*}\_{1j}, Δ​κ0​i​j:=κ0​i−κ0​j∗\Delta\kappa\_{0ij}:=\kappa\_{0i}-\kappa^{\*}\_{0j}, Δ​η1​i​j:=η1​i−η1​j∗\Delta\eta\_{1ij}:=\eta\_{1i}-\eta^{\*}\_{1j} and Δ​η0​i​j:=η0​i−η0​j∗\Delta\eta\_{0ij}:=\eta\_{0i}-\eta^{\*}\_{0j}. In addition, we define r1,j:=r1​(|𝒱1,j|)r\_{1,j}:=r\_{1}(|\mathcal{V}\_{1,j}|) and r2,j:=r2​(|𝒱2,j|)r\_{2,j}:=r\_{2}(|\mathcal{V}\_{2,j}|), where the functions r1r\_{1} and r2r\_{2} stand for the solvability of polynomial equation systems specified in Appendix [B](#A2 "Appendix B Systems of Polynomial Equations"). In particular, we have r1​(2)=r2​(2)=4r\_{1}(2)=r\_{2}(2)=4, r1​(3)=r2​(3)=6r\_{1}(3)=r\_{2}(3)=6, and r1​(m),r2​(m)≥7r\_{1}(m),r\_{2}(m)\geq 7 for all m≥4m\geq 4.

###### Theorem 2.

Assume the expert functions h1h\_{1} and h2h\_{2} take linear forms. Then, the lower bound 𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]≳𝒟2((G1,G2),(G1∗,G2∗))\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})) holds for any (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). As a consequence, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒟2(G^1n,G^2n),(G1∗,G2∗))=𝒪P([log(n)/n]12).\displaystyle\mathcal{D}\_{2}(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  | (7) |

By comparing the Voronoi losses 𝒟1\mathcal{D}\_{1} and 𝒟2\mathcal{D}\_{2}, we see that the estimation rates for exactly-specified shared and routed experts remain of parametric order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}). By contrast, there are changes in the estimation rates for the over-specified experts.

*(i) Shared experts.* The estimation rates for over-specified parameters κ1​j∗\kappa^{\*}\_{1j}, κ0​j∗\kappa^{\*}\_{0j}, τj∗\tau^{\*}\_{j} are heterogeneous, of orders 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}), 𝒪~P​(n−1/2​r1,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/2r\_{1,j}}), 𝒪~P​(n−1/r1,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{1,j}}), respectively. Since the input space is bounded, we have |(κ^1​in)⊤​x+κ^0​in−(κ1​j∗)⊤​x−κ0​j∗|≲‖κ^1​in−κ1​j∗‖+|κ^0​in−κ0​j∗||(\hat{\kappa}^{n}\_{1i})^{\top}x+\hat{\kappa}^{n}\_{0i}-(\kappa^{\*}\_{1j})^{\top}x-\kappa^{\*}\_{0j}|\lesssim\|\hat{\kappa}^{n}\_{1i}-\kappa^{\*}\_{1j}\|+|\hat{\kappa}^{n}\_{0i}-\kappa^{\*}\_{0j}|. Then, it follows that the shared experts (κ1​j∗)⊤​x+κ0​j∗(\kappa^{\*}\_{1j})^{\top}x+\kappa^{\*}\_{0j} admit estimation rates of orders 𝒪~P​(n−1/2​r1,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/2r\_{1,j}}). However, note that the rates for estimating their input-dependent terms (κ1​j∗)⊤​x(\kappa^{\*}\_{1j})^{\top}x are much faster, of order 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}).

*(ii) Routed experts.* The estimation rates for over-specified parameters η1​j∗\eta^{\*}\_{1j}, νj∗\nu^{\*}\_{j} are of orders 𝒪~P​(n−1/r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2,j}}), while those for β1​j∗\beta^{\*}\_{1j}, η0​j∗\eta^{\*}\_{0j} are slower, of orders 𝒪~P​(n−1/2​r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/2r\_{2,j}}). By arguing similarly to the case of shared experts, the rates for estimating the routed experts (η1​j∗)⊤​x+η0​j∗(\eta^{\*}\_{1j})^{\top}x+\eta^{\*}\_{0j} and their input-dependent terms (η1​j∗)⊤​x(\eta^{\*}\_{1j})^{\top}x depend on the parameter r2r\_{2} (related to the solvability of a certain system of polynomial equations) and are of orders 𝒪~P​(n−1/2​r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/2r\_{2,j}}) and 𝒪~P​(n−1/r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2,j}}), respectively. Notably, these rates become increasingly slow with the cardinality of the corresponding Voronoi cell 𝒱2,j\mathcal{V}\_{2,j}. In particular, when |𝒱2,j|=3|\mathcal{V}\_{2,j}|=3, they become 𝒪~P​(n−1/12)\widetilde{\mathcal{O}}\_{P}(n^{-1/12}) and 𝒪~P​(n−1/6)\widetilde{\mathcal{O}}\_{P}(n^{-1/6}), respectively.

*(iii) Sample efficiency of the shared expert strategy.* From the above observations, we see that shared experts have faster estimation rates than routed experts, i.e.𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) compared to 𝒪~P​(n−1/r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2,j}}). Furthermore, the estimation rates for shared experts in DeepSeekMoE are also faster than those for experts in MoE models without the shared expert strategy [[56](#bib.bib56)], which are also of the order 𝒪~P​(n−1/r2,j)\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2,j}}). The punchline is that fewer data points are needed to estimate shared experts.

## 3 Experiments

In this section, we empirically validate the theoretical findings in the previous section. Using synthetic data, we demonstrate the convergence behavior of the maximum likelihood estimator (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) towards the true mixing measure (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}); we defer this experiment to Appendix [G.1](#A7.SS1 "G.1 Numerical Experiments ‣ Appendix G Additional Experiments"). In real-world scenarios, we evaluate our methodology on language modeling tasks using the SlimPajama corpus [[66](#bib.bib66)] (Section [3.1](#S3.SS1 "3.1 Language Modeling ‣ 3 Experiments")), and extend our evaluation to vision-language modeling benchmarks using the LLaVA architecture [[43](#bib.bib43)] integrated within the LIBMoE framework [[57](#bib.bib57)] (Section [3.2](#S3.SS2 "3.2 Vision-Language Modeling ‣ 3 Experiments")). Our empirical study compares four model configurations: Vanilla SMoE, DeepSeek-V3 (shared experts combined with normalized sigmoid gating), DeepSeek-V2 (shared experts with softmax routing), and SMoE Sigmoid Gating (normalized sigmoid gating without shared experts).

Table 2: Performance comparisons of different Sparse Mixture of Experts (SMoE) models on subsets of the SlimPajama dataset using a small-scale model with 158M parameters and large-scale model with 679M parameters. (SMoE-SG refers to SMoE Sigmoid Gating). PPL indicates the perplexity score.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Small Models (158M) | | | | Large Models (679M) | | | |
|  | SMoE | DeepSeek-V3 | DeepSeek-V2 | SMoE-SG | SMoE | DeepSeek-V3 | DeepSeek-V2 | SMoE-SG |
| PPL ↓\downarrow | 13.63 | 13.42 | 13.49 | 13.61 | 9.51 | 9.49 | 9.52 | 9.46 |
| LAMBADA | 25.27% | 25.49% | 25.29% | 25.43% | 37.13% | 36.88% | 37.11% | 37.56% |
| BLiMP | 77.71% | 77.20% | 77.37% | 77.38% | 80.47% | 81.28% | 80.98% | 81.08% |
| CBT | 84.18% | 84.40% | 84.33% | 84.23% | 89.83% | 89.65% | 89.93% | 89.57% |
| HellaSwag | 29.43% | 29.38% | 29.38% | 29.13% | 37.49% | 37.32% | 37.14% | 37.52% |
| PIQA | 57.94% | 59.14% | 60.17% | 58.92% | 64.36% | 65.72% | 64.36% | 64.91% |
| ARC-Challenge | 21.20% | 21.63% | 20.52% | 21.37% | 23.09% | 23.95% | 24.21% | 23.09% |
| RACE | 30.11% | 30.60% | 31.02% | 31.05% | 33.03% | 33.12% | 33.17% | 32.68% |
| SIQA | 35.62% | 35.57% | 34.90% | 34.90% | 37.41% | 38.59% | 36.95% | 37.67% |
| CommonSenseQA | 24.65% | 25.47% | 24.98% | 24.90% | 26.54% | 28.09% | 27.35% | 28.50% |
| Average | 42.90% | 43.21% | 43.11% | 43.04% | 47.71% | 48.29% | 47.91% | 48.06% |

![Refer to caption](/html/2505.10860/assets/x1.png)


Figure 1: Average performance (%) over training steps in language modeling tasks. Left: Model with 158M parameters; Right: Model with 679M parameters.

### 3.1 Language Modeling

Experimental Setup. We conduct the experiments on language modeling using subsets of the popular SLimPajama [[66](#bib.bib66)] dataset using Switch Transformer [[20](#bib.bib20)] baseline in two scales: small (158M parameters trained on 6.5B tokens) and large (679M parameters trained on 26.2B tokens). The models are configured with 66 total experts, utilizing top-8 expert routing in the baseline and a top-6 plus 2 shared experts routing scheme in the DeepSeek variants. We measure model performance in terms of perplexity and zero-shot accuracy across nine diverse downstream evaluation tasks [[58](#bib.bib58), [74](#bib.bib74), [24](#bib.bib24), [81](#bib.bib81), [3](#bib.bib3), [11](#bib.bib11), [35](#bib.bib35), [63](#bib.bib63), [68](#bib.bib68)]. Full experimental details are provided in Appendix [I.1](#A9.SS1 "I.1 Language Modeling ‣ Appendix I Experimental Details").

Zero-shot performance on downstream tasks. Table [2](#S3.T2 "Table 2 ‣ 3 Experiments") summarizes our primary experimental results for two model sizes trained on the SlimPajama dataset [[66](#bib.bib66)]. The results clearly demonstrate that both DeepSeek-V3 and DeepSeek-V2 consistently outperform the Vanilla SMoE baseline, achieving lower perplexity (PPL) scores and higher average accuracy across various downstream tasks for both model scales. Additionally, we integrated the normalized sigmoid router into the Vanilla SMoE architecture and observed that the SMoE Sigmoid Gating achieves superior performance compared to the Vanilla SMoE and, in some benchmarks, even surpasses the DeepSeek variants.

Convergence Rate. Figure [1](#S3.F1 "Figure 1 ‣ 3 Experiments") presents the average performance across various downstream tasks for DeepSeek-V3 and DeepSeek-V2 compared to the Vanilla SMoE. Across both model sizes, the DeepSeek variants demonstrate substantially faster convergence. Specifically, in both 158M and 679M parameter scales, DeepSeek-V3 and DeepSeek-V2 consistently reach the final performance of Vanilla SMoE using only 70-80% of the total training steps. Notably, DeepSeek-V3, which incorporates normalized sigmoid gating, demonstrates marginal improvements over DeepSeek-V2 in both convergence speed and final task performance. These results highlight the efficiency gains introduced by the shared expert and normalized sigmoid gating mechanisms and provide empirical support for our theoretical findings.

### 3.2 Vision-Language Modeling

Experimental Setup. We conduct experiments on the visual instruction tuning tasks [[42](#bib.bib42)] using the popular LLaVA architecture [[44](#bib.bib44)]. Building upon the LIBMoE framework [[57](#bib.bib57)], we adopt Phi3.5-mini [[1](#bib.bib1)] as the language model and SigLIP [[82](#bib.bib82)] as the vision encoder. Unlike LIBMoE, we sparse-upcycled [[32](#bib.bib32)] only the MLP Connector into 8 experts, employing a top-4 expert routing strategy, while the DeepSeek variants adopt a top-3 expert routing scheme with an additional shared expert, making our model approximately 4.4B parameters. To compare different SMoE algorithms, we use a subset of the LLaVA 1.5 dataset [[42](#bib.bib42)] (332K samples and 287M tokens) to train the models in the Visual Instruction Tuning (VIT) stage. Evaluation covers diverse benchmarks containing various vision-language capabilities, including perception, reasoning, OCR, instruction following, and more [[31](#bib.bib31), [7](#bib.bib7), [40](#bib.bib40), [47](#bib.bib47), [65](#bib.bib65), [27](#bib.bib27), [83](#bib.bib83), [78](#bib.bib78), [45](#bib.bib45)]. See Appendix [I.2](#A9.SS2 "I.2 Vision Language Modeling ‣ Appendix I Experimental Details").

Table 3: Vision-language model performance across benchmarks. (SMoE-SG refers to SMoE Sigmoid Gating)

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AI2D | MMStar | POPE | Science  QA | TextVQA | GQA | MME-RW  -Lite | MMMU  Pro-S | OCR  Bench | Average |
| SMoE | 64.90% | 41.66% | 85.67% | 81.61% | 40.92% | 60.19% | 31.79% | 25.61% | 30.90% | 51.47% |
| DeepSeek-V3 | 65.45% | 41.40% | 85.44% | 81.94% | 40.69% | 60.01% | 32.20% | 26.01% | 32.60% | 51.75% |
| DeepSeek-V2 | 64.70% | 41.55% | 85.80% | 82.20% | 40.51% | 60.15% | 31.11% | 25.72% | 31.00% | 51.41% |
| SMoE-SR | 64.64% | 41.51% | 85.87% | 82.17% | 40.54% | 60.07% | 31.68% | 25.95% | 31.00% | 51.49% |

![Refer to caption](/html/2505.10860/assets/x2.png)


Figure 2: Average performance (%) over training steps on vision-language pretraining tasks. Left: Vanilla SMoE vs. DeepSeek-V3; Center: Vanilla SMoE vs. DeepSeek-V2; Right: DeepSeek-V2 vs. DeepSeek-V3.

Performance. As summarized in Table [3](#S3.T3 "Table 3 ‣ 3.2 Vision-Language Modeling ‣ 3 Experiments"), DeepSeek-V3 achieves the highest average score (51.75%), outperforming the Vanilla SMoE (51.47%) and other model variants. Although DeepSeek-V2 shows slightly lower performance compared to other models, the difference remains marginal. Consistent with observations from language modeling experiments, additional evaluations conducted with Vanilla SMoE and the normalized sigmoid router show a similar pattern, confirming that the normalized sigmoid routing mechanism consistently enhances the performance of the standard SMoE architecture.

Convergence Rate. Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Vision-Language Modeling ‣ 3 Experiments") illustrates the performance progression over training steps, where both DeepSeek variants exhibit faster and more stable convergence compared to Vanilla SMoE. Notably, both DeepSeek-V2 and DeepSeek-V3 demonstrate accelerated convergence during the final stages of training. These results suggest that both shared expert integration and normalized routing significantly contribute to faster learning in vision-language pretraining.

### 3.3 Router Analysis

We now explore the router behavior by empirically examining the router saturation and change rate.

![Refer to caption](/html/2505.10860/assets/x3.png)


Figure 3: Evolution of router saturation (averaged across all layers) during training for language-modeling tasks with 158 M (left) and 679 M (right) parameter models. We compute saturation by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

Router Saturation.
Router Saturation, first introduced in OLMoE [[51](#bib.bib51)], quantifies the proportion of overlapping activated experts between the final checkpoint and an intermediary checkpoint at time tt. It serves as a measure of the router’s convergence over the course of training. A higher router saturation value indicates stronger alignment in expert selection, signifying that the router’s decisions become increasingly consistent with its final configuration. The formal definition and formula are defined in the Appendix [H.1](#A8.SS1 "H.1 Router Saturation ‣ Appendix H Additional Router Analysis").
Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Router Analysis ‣ 3 Experiments") shows that, after 5% of training, up to ~60% of router decisions have already saturated. This early saturation aligns with prior findings in OLMoE [[51](#bib.bib51)] and OpenMoE [[76](#bib.bib76)], supporting the validity of our experimental setup. When comparing model configurations, we observe that models equipped with normalized sigmoid gating achieve noticeably faster saturation than those using softmax gating. In particular, the SMoE Sigmoid Gating exhibits consistently steeper saturation curves compared to Vanilla SMoE, reflecting more rapid convergence in expert selection. A similar pattern is observed in the comparison between DeepSeek-V3 and DeepSeek-V2 under the shared expert configuration. These findings highlight the effectiveness of normalized sigmoid gating in accelerating router convergence, potentially reducing the training time required for convergence.

![Refer to caption](/html/2505.10860/assets/x4.png)


Figure 4: Router Change Rate (averaged across all layers) during training for language-modeling tasks with 158 M (left) and 679 M (right) parameter models. We compute router change rate by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

Router Change Rate.
To evaluate the stability of the routing mechanism in Mixture-of-Experts (MoE) models during training, we introduce the Router Change Rate metric. This metric quantifies the proportion of expert activation decisions that change between consecutive checkpoints. A lower router change rate implies greater consistency in routing decisions over time, reflecting a more stable training process. The formal definition and computation details are provided in Appendix [H.2](#A8.SS2 "H.2 Router Change Rate ‣ Appendix H Additional Router Analysis").
Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Router Analysis ‣ 3 Experiments") presents the router change rate comparison of different model configurations. We find that models employing normalized sigmoid gating have significantly lower change rates in both non-shared and shared expert settings. These findings underscore the efficiency of normalized sigmoid gating in stabilizing routing decisions throughout training. By reducing the routing fluctuation problem [[13](#bib.bib13)], this mechanism promotes a more consistent expert specialization, indicating that stable routing is critical in enhancing both optimization efficiency and final model performance.

## 4 Discussion

In this paper, we have presented an extensive study on the benefits of two fundamental ingredients of DeepSeekMoE architecture, namely the shared expert strategy and the normalized sigmoid gating mechanism. From the theoretical side, we perform a convergence analysis of expert estimation to investigate differences in sample efficiency. Our analysis reveals that the shared expert strategy leads to faster estimation rates for shared experts compared to routed experts and experts in the standard MoE. Furthermore, the estimation rates for routed experts become dramatically faster when replacing the softmax gating with the normalized sigmoid gating in DeepSeekMoE. Therefore, the incorporation of these two key factors into DeepSeekMoE significantly reduces the overall sample complexity for the estimation tasks.

From the empirical side, we validate our theoretical findings through extensive experiments and analysis on both synthetic and real-world datasets. Our results consistently demonstrate that both the shared experts strategy and the normalized sigmoid gating mechanism substantially affect the convergence rate and downstream performance in real-world scenarios. Moreover, these two ingredients also yield substantial gains in router convergence, routing stability, and expert utilization. Overall, our work provides both a principled understanding and robust empirical evidence for the effectiveness of these two components, offering valuable guidance for the design of future sparse mixture-of-experts.

Although our analysis confirms that using shared experts improves the sample complexity, it does not indicate how many shared experts should be employed to achieve the optimal configuration given a fixed computational budget. A potential approach to this problem is to derive a scaling law involving these quantities induced from extensive experiments as in [[48](#bib.bib48)]. However, since this direction goes beyond the scope of our work, we leave it for future development.

Appendices for
  
“On DeepSeekMoE: Statistical Benefits of Shared Experts
  
and Normalized Sigmoid Gating”

###### Contents

1. [1 Introduction](#S1)
2. [2 On Shared Expert Strategy](#S2)
   1. [2.1 Strongly Identifiable Experts](#S2.SS1 "In 2 On Shared Expert Strategy")
   2. [2.2 Linear Experts](#S2.SS2 "In 2 On Shared Expert Strategy")
3. [3 Experiments](#S3)
   1. [3.1 Language Modeling](#S3.SS1 "In 3 Experiments")
   2. [3.2 Vision-Language Modeling](#S3.SS2 "In 3 Experiments")
   3. [3.3 Router Analysis](#S3.SS3 "In 3 Experiments")
4. [4 Discussion](#S4)
5. [A On Normalized Sigmoid Gating](#A1)
   1. [A.1 Sparse Regime](#A1.SS1 "In Appendix A On Normalized Sigmoid Gating")
   2. [A.2 Dense Regime](#A1.SS2 "In Appendix A On Normalized Sigmoid Gating")
6. [B Systems of Polynomial Equations](#A2)
7. [C Related Works](#A3)
8. [D Proof of Main Results](#A4)
   1. [D.1 Proof of Theorem 1](#A4.SS1 "In Appendix D Proof of Main Results")
   2. [D.2 Proof of Theorem 2](#A4.SS2 "In Appendix D Proof of Main Results")
   3. [D.3 Proof of Theorem 3](#A4.SS3 "In Appendix D Proof of Main Results")
   4. [D.4 Proof of Theorem 4](#A4.SS4 "In Appendix D Proof of Main Results")
9. [E Proof of Auxiliary Results](#A5)
   1. [E.1 Proof of Proposition 1](#A5.SS1 "In Appendix E Proof of Auxiliary Results")
   2. [E.2 Identifiability of DeepSeekMoE](#A5.SS2 "In Appendix E Proof of Auxiliary Results")
10. [F Extended Theoretical Results for Sparse Gating MoE](#A6)
    1. [F.1 Proof of Lemma 3](#A6.SS1 "In Appendix F Extended Theoretical Results for Sparse Gating MoE")
    2. [F.2 Proof of Proposition 6](#A6.SS2 "In Appendix F Extended Theoretical Results for Sparse Gating MoE")
11. [G Additional Experiments](#A7)
    1. [G.1 Numerical Experiments](#A7.SS1 "In Appendix G Additional Experiments")
       1. [G.1.1 Experimental Setup](#A7.SS1.SSS1 "In G.1 Numerical Experiments ‣ Appendix G Additional Experiments")
       2. [G.1.2 Theorem 1](#A7.SS1.SSS2 "In G.1 Numerical Experiments ‣ Appendix G Additional Experiments")
       3. [G.1.3 Theorem 2](#A7.SS1.SSS3 "In G.1 Numerical Experiments ‣ Appendix G Additional Experiments")
       4. [G.1.4 Theorem 3](#A7.SS1.SSS4 "In G.1 Numerical Experiments ‣ Appendix G Additional Experiments")
       5. [G.1.5 Theorem 4](#A7.SS1.SSS5 "In G.1 Numerical Experiments ‣ Appendix G Additional Experiments")
    2. [G.2 Language Modeling](#A7.SS2 "In Appendix G Additional Experiments")
    3. [G.3 Vision-Language Modeling](#A7.SS3 "In Appendix G Additional Experiments")
12. [H Additional Router Analysis](#A8)
    1. [H.1 Router Saturation](#A8.SS1 "In Appendix H Additional Router Analysis")
    2. [H.2 Router Change Rate](#A8.SS2 "In Appendix H Additional Router Analysis")
    3. [H.3 Expert Utilization](#A8.SS3 "In Appendix H Additional Router Analysis")
13. [I Experimental Details](#A9)
    1. [I.1 Language Modeling](#A9.SS1 "In Appendix I Experimental Details")
       1. [I.1.1 Datasets](#A9.SS1.SSS1 "In I.1 Language Modeling ‣ Appendix I Experimental Details")
       2. [I.1.2 Model Settings, Training Settings and Evaluation](#A9.SS1.SSS2 "In I.1 Language Modeling ‣ Appendix I Experimental Details")
    2. [I.2 Vision Language Modeling](#A9.SS2 "In Appendix I Experimental Details")
       1. [I.2.1 Datasets](#A9.SS2.SSS1 "In I.2 Vision Language Modeling ‣ Appendix I Experimental Details")
       2. [I.2.2 Model Settings, Training Settings and Evaluation](#A9.SS2.SSS2 "In I.2 Vision Language Modeling ‣ Appendix I Experimental Details")
    3. [I.3 Training Time and Resource Allocation](#A9.SS3 "In Appendix I Experimental Details")

## Appendix A On Normalized Sigmoid Gating

In this appendix, we conduct a convergence analysis of expert estimation in DeepSeek-V3’s MoE to investigate the sample efficiency of the normalized sigmoid gating used in this architecture.

Problem setting. Assume that (X1,Y1),(X2,Y2),…,(Xn,Yn)∈ℝd×ℝ(X\_{1},Y\_{1}),(X\_{2},Y\_{2}),\ldots,(X\_{n},Y\_{n})\in\mathbb{R}^{d}\times\mathbb{R} are i.i.d. samples drawn from the Gaussian DeepSeek-V3’s MOE whose conditional density function gG∗​(y|x)g\_{G\_{\*}}(y|x) is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | gG1∗,G2∗​(y|x):=12\displaystyle g\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x):=\frac{1}{2} | ∑i=1k1∗ωi∗​π​(y|h1​(x,κi∗),τi∗)\displaystyle\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(y|h\_{1}(x,\kappa^{\*}\_{i}),\tau^{\*}\_{i}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +12​∑i=1k2∗σ​((β1​i∗)⊤​x+β0​i∗)∑j=1k2∗σ​((β1​j∗)⊤​x+β0​j∗)⋅π​(y|h2​(x,ηi∗),νi∗),\displaystyle+\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{2}}\frac{\sigma((\beta\_{1i}^{\*})^{\top}x+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}x+\beta\_{0j}^{\*})}\cdot\pi(y|h\_{2}(x,\eta^{\*}\_{i}),\nu\_{i}^{\*}), |  | (8) |

where σ:ℝ→(0,∞)\sigma:\mathbb{R}\to(0,\infty) stands for the sigmoid function, that is, σ​(z):=11+exp⁡(−z)\sigma(z):=\frac{1}{1+\exp(-z)}, for all z∈ℝz\in\mathbb{R}. By abuse of notations, we define the pair of ground-truth mixing measures (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) under this setting as G1∗:=∑i=1k1∗ωi∗​δ(κi∗,τi∗)G^{\*}\_{1}:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\delta\_{(\kappa^{\*}\_{i},\tau^{\*}\_{i})} and G2∗:=∑i=1k2∗σ​(β0​i∗)​δ(β1​i∗,ηi∗,νi∗)G^{\*}\_{2}:=\sum\_{i=1}^{k^{\*}\_{2}}\sigma(\beta\_{0i}^{\*})\delta\_{(\beta\_{1i}^{\*},\eta\_{i}^{\*},\nu\_{i}^{\*})}. Here, we still impose all the assumptions used for Section [2](#S2 "2 On Shared Expert Strategy") on this analysis.

Maximum likelihood estimation (MLE). Under the above setting, the MLE defined in equation ([2](#S2.E2 "In 2 On Shared Expert Strategy")) is rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  | (G~1n,G~2n)∈arg​max(G1,G2)∈𝒢k1,k2​(Θ)⁡1n​∑i=1nlog⁡(gG1,G2​(Yi|Xi)),\displaystyle(\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2})\in\operatorname\*{arg\,max}\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)}\frac{1}{n}\sum\_{i=1}^{n}\log(g\_{G\_{1},G\_{2}}(Y\_{i}|X\_{i})), |  | (9) |

where 𝒢k1,k2​(Θ):=𝒢k1​(Θ1)×𝒢k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):=\mathcal{G}\_{k\_{1}}(\Theta\_{1})\times\mathcal{G}\_{k\_{2}}(\Theta) denotes the set of mixing measure pairs (G1,G2)(G\_{1},G\_{2}) with at most k1k\_{1} and k2k\_{2} atoms, respectively, that is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒢k1​(Θ1)\displaystyle\mathcal{G}\_{k\_{1}}(\Theta\_{1}) | :={G1=∑i=1k1′ωi​δ(κi,τi):1≤k1′≤k1},\displaystyle:=\Big{\{}G\_{1}=\sum\_{i=1}^{k^{\prime}\_{1}}\omega\_{i}\delta\_{(\kappa\_{i},\tau\_{i})}:1\leq k^{\prime}\_{1}\leq k\_{1}\Big{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒢k2​(Θ2)\displaystyle\mathcal{G}\_{k\_{2}}(\Theta\_{2}) | :={G2=∑i=1k2′σ​(β0​i)​δ(β1​i,ηi∗,νi∗):1≤k2′≤k2}.\displaystyle:=\Big{\{}G\_{2}=\sum\_{i=1}^{k^{\prime}\_{2}}\sigma(\beta\_{0i})\delta\_{(\beta\_{1i},\eta^{\*}\_{i},\nu^{\*}\_{i})}:1\leq k^{\prime}\_{2}\leq k\_{2}\Big{\}}. |  |

Given the MLE (G~1n,G~2n)(\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}) in equation ([9](#A1.E9 "In Appendix A On Normalized Sigmoid Gating")), we proceed to establish the convergence rate of density estimation gG~1n,G~2ng\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}. However, there are some changes in the gating convergence behavior compared to that in DeepSeekMoE due to the structure of the sigmoid function.

The convergence of normalized sigmoid gating. Recall that we fit the ground-truth DeepSeek-V3’s MoE model ([A](#A1.Ex10 "Appendix A On Normalized Sigmoid Gating")) with a mixture of k1>k1∗k\_{1}>k^{\*}\_{1} shared experts and k2>k2∗k\_{2}>k^{\*}\_{2} routed experts. Then, there must be some gorund-truth routed experts approximated by more than one fitted routed experts. As a result, the sum of weights of these fitted routed experts is expected to converge to the weight of the ground-truth routed experts, for example,

|  |  |  |
| --- | --- | --- |
|  | ∑i=12σ​((β^1​in)⊤​x+β^0​in)∑j=1k2nσ​((β^1​jn)⊤​x+β^0​jn)→σ​((β11∗)⊤​x+β01∗)∑j=1k2∗σ​((β1​j∗)⊤​x+β0​j∗),\displaystyle\sum\_{i=1}^{2}\frac{\sigma((\hat{\beta}^{n}\_{1i})^{\top}x+\hat{\beta}^{n}\_{0i})}{\sum\_{j=1}^{k^{n}\_{2}}\sigma((\hat{\beta}^{n}\_{1j})^{\top}x+\hat{\beta}^{n}\_{0j})}\to\frac{\sigma((\beta^{\*}\_{11})^{\top}x+\beta^{\*}\_{01})}{\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta^{\*}\_{1j})^{\top}x+\beta^{\*}\_{0j})}, |  |

for almost every xx. Since the denominator ∑j=1k2nσ​((β^1​jn)⊤​x+β^0​jn)\sum\_{j=1}^{k^{n}\_{2}}\sigma((\hat{\beta}^{n}\_{1j})^{\top}x+\hat{\beta}^{n}\_{0j}) should converge to its counterpart ∑j=1k2∗σ​((β1​j∗)⊤​x+β0​j∗)\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta^{\*}\_{1j})^{\top}x+\beta^{\*}\_{0j}). Then, it must hold that

|  |  |  |
| --- | --- | --- |
|  | ∑i=12σ​((β^1​in)⊤​x+β^0​in)→σ​((β11∗)⊤​x+β01∗),\displaystyle\sum\_{i=1}^{2}\sigma((\hat{\beta}^{n}\_{1i})^{\top}x+\hat{\beta}^{n}\_{0i})\to\sigma((\beta^{\*}\_{11})^{\top}x+\beta^{\*}\_{01}), |  |

as n→∞n\to\infty, for almost every xx. This result occurs only if β11∗=0d\beta^{\*}\_{11}=0\_{d}. Therefore, we will divide our analysis into two complement regimes for the over-specified parameters β1​i∗\beta^{\*}\_{1i}:

Sparse regime. All over-specified parameters β1​i∗\beta^{\*}\_{1i} equal zero vector;

Dense regime. Not all over-specified parameters β1​i∗\beta^{\*}\_{1i} equal zero vector.

It is worth noting that the sparse regime of parameters rarely occurs in practice. However, for completeness, we will perform the convergence analysis of expert estimation under both the sparse and dense regimes in Appendix [A.1](#A1.SS1 "A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating") and Appendix [A.2](#A1.SS2 "A.2 Dense Regime ‣ Appendix A On Normalized Sigmoid Gating"), respectively.

### A.1 Sparse Regime

To begin with, let us derive the density estimation rate for the sparse regime in Proposition [2](#Thmproposition2 "Proposition 2. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating").

###### Proposition 2.

Under the sparse regime, the density estimation gG~1n,G~2n​(Y|X)g\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(Y|X) converges to the true density gG1∗,G2∗​(Y|X)g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) at the following rate:

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(gG~1n,G~2n(⋅|X),gG1∗,G2∗(⋅|X))]=𝒪P([log(n)/n]12).\displaystyle\mathbb{E}\_{X}[V(g\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

Since the sigmoid function is Lipschitz continuous, the proof of this proposition can be done similarly to that of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy"), which is provided in Appendix [E.1](#A5.SS1 "E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results"). The result of Proposition [2](#Thmproposition2 "Proposition 2. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating") indicates that the density estimation gG~1n,G~2ng\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}} converges to the ground-truth density gG1∗,G2∗g\_{G^{\*}\_{1},G^{\*}\_{2}} under the Total Variation distance at the parametric rate of order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}).

Voronoi loss. Next, we construct Voronoi loss tailored to the sparse regime as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟3​((G1,G2),(G1∗,G2∗)):=∑j=1k1∗|∑i∈𝒱1,jωi−ωj∗|+∑j∈[k2∗]:|𝒱2,j|>1|∑i∈𝒱2,jσ​(β0​i)−σ​(β0​j∗)|\displaystyle\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})):=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}-\omega\_{j}^{\*}\Big{|}+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i})-\sigma(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|=1∑i∈𝒱1,jωi​(‖Δ​κi​j‖+|Δ​τi​j|)+∑j∈[k2∗],|𝒱2,j|=1∑i∈𝒱2,j(‖Δ​β1​i​j‖+|Δ​β0​i​j|+‖Δ​ηi​j‖+|Δ​νi​j|)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|+|\Delta\tau\_{ij}|)+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}],\\ |\mathcal{V}\_{2,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}(\|\Delta\beta\_{1ij}\|+|\Delta\beta\_{0ij}|+\|\Delta\eta\_{ij}\|+|\Delta\nu\_{ij}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|>1∑i∈𝒱1,jωi​(‖Δ​κi​j‖2+|Δ​τi​j|2)+∑j∈[k2∗],|𝒱2,j|>1∑i∈𝒱2,j(‖Δ​β1​i​j‖2+‖Δ​ηi​j‖2+|Δ​νi​j|2),\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|^{2}+|\Delta\tau\_{ij}|^{2})+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{2}],\\ |\mathcal{V}\_{2,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}(\|\Delta\beta\_{1ij}\|^{2}+\|\Delta\eta\_{ij}\|^{2}+|\Delta\nu\_{ij}|^{2}), |  | (10) |

where we denote Δ​β0​i​j:=β0​i−β0​j∗\Delta\beta\_{0ij}:=\beta\_{0i}-\beta\_{0j}^{\*}. Given the above loss function, we are now able to capture parameter and expert estimation rates under the sparse regime in the following theorem.

###### Theorem 3.

Suppose that the expert functions h1h\_{1} and h2h\_{2} are strongly identifiable. Then, the lower bound 𝔼X[V(gG1,G2(⋅|X),gG1∗,G2∗(⋅|X))]≳𝒟3((G1,G2),(G1∗,G2∗))\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})) holds for any (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). As a consequence, we have

|  |  |  |
| --- | --- | --- |
|  | 𝒟3(G~1n,G~2n),(G1∗,G2∗))=𝒪P([log(n)/n]12).\displaystyle\mathcal{D}\_{3}(\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

The proof of Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating") is provided in Appendix [D.3](#A4.SS3 "D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results"). From the formulations of Voronoi losses 𝒟1\mathcal{D}\_{1} and 𝒟3\mathcal{D}\_{3} in equations ([2.1](#S2.Ex5 "2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy")) and ([A.1](#A1.Ex16 "A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating")), respectively, we observe that shared experts and routed experts which satisfy the strong identifiability condition admit the same estimation rates as those in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy"). In particular, the rates for estimating both types of experts are of orders 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) and 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) when they are exactly-specified and over-specified, respectively. In other words, the normalized sigmoid gating does not have clear advantages over the standard softmax gating under the sparse regime. However, it should be noted that the sparse regime is less likely to occur in practice than the dense regime. Thus, we continue the comparison of sample efficiency between the two gatings under the dense regime in the next section.

### A.2 Dense Regime

Next, under the dense regime, note that the ground-truth model is misspecified, that is, the density estimation gG~1n,G~2ng\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}} converges to the missepcified density function gG1∗,Gˇ2g\_{G^{\*}\_{1},\check{G}\_{2}} rather than the ground-truth density gG1∗,G2∗g\_{G^{\*}\_{1},G^{\*}\_{2}}, where Gˇ2∈𝒢¯k2​(Θ2):=arg​minG2∈𝒢k2​(Θ2)∖𝒢k2∗​(Θ2)⁡KL​(gG1∗,G2∥gG1∗,G2∗)\check{G}\_{2}\in\overline{\mathcal{G}}\_{k\_{2}}(\Theta\_{2}):=\operatorname\*{arg\,min}\_{G\_{2}\in\mathcal{G}\_{k\_{2}}(\Theta\_{2})\setminus\mathcal{G}\_{k^{\*}\_{2}}(\Theta\_{2})}\mathrm{KL}(g\_{G^{\*}\_{1},G\_{2}}\|g\_{G^{\*}\_{1},G^{\*}\_{2}}). Following the result of Proposition [2](#Thmproposition2 "Proposition 2. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating"), we are also able to establish the parametric density estimation rate under the dense regime in the following corollary.

###### Corollary 1.

Under the dense regime, the density estimation gG~1n,G~2ng\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}} converges to the density gG1∗,Gˇ2g\_{G^{\*}\_{1},\check{G}\_{2}} at the rate: infGˇ2∈𝒢¯k2​(Θ2)𝔼X[V(gG~1n,G~2n(⋅|X),gG1∗,Gˇ2(⋅|X))]=𝒪P([log(n)/n]12)\inf\_{\check{G}\_{2}\in\overline{\mathcal{G}}\_{k\_{2}}(\Theta\_{2})}\mathbb{E}\_{X}[V(g\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}).

Subsequently, we focus on characterizing parameter and expert estimation rates under the dense regime by establishing the Total Variation lower bound

|  |  |  |
| --- | --- | --- |
|  | inf(G1∗,Gˇ2)∈𝒢¯k1,k2​(Θ)𝔼X[V(gG1,G2(⋅|X),gG1∗,Gˇ2(⋅|X))]≳𝒟4((G1,G2),(G1∗,Gˇ2)),\inf\_{(G^{\*}\_{1},\check{G}\_{2})\in\overline{\mathcal{G}}\_{k\_{1},k\_{2}}(\Theta)}\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2})), |  |

where 𝒟4\mathcal{D}\_{4} is a Voronoi loss that will be defined later in equation ([A.2](#A1.Ex21 "A.2 Dense Regime ‣ Appendix A On Normalized Sigmoid Gating")). Recall that a key step in deriving this lower bound is to decompose the density difference gG~1n,G~2n​(Y|X)−gG1∗,Gˇ2​(Y|X)g\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},\check{G}\_{2}}(Y|X) into linearly independent terms using Taylor expansions to the functions x↦π​(Y|h1​(x,κ),τ)x\mapsto\pi(Y|h\_{1}(x,\kappa),\tau) and x↦σ​(β1⊤​x+β0)​π​(Y|h2​(x,η),ν)x\mapsto\sigma(\beta\_{1}^{\top}x+\beta\_{0})\pi(Y|h\_{2}(x,\eta),\nu) w.r.t their parameters (κ,τ)(\kappa,\tau) and (β1,β0,η,ν)(\beta\_{1},\beta\_{0},\eta,\nu), respectively. Due to the gating change, it is necessary to introduce a new condition on the routed expert function h2h\_{2} to ensure linear independence among terms in the Taylor expansions.

###### Definition 2 (Weak Identifiability).

We say that the expert function x↦h2​(x,η)x\mapsto h\_{2}(x,\eta) is weakly identifiable if it is differentiable w.r.t its parameter η\eta, and if for any k2≥1k\_{2}\geq 1 and η1,η2,…,ηk2\eta\_{1},\eta\_{2},\ldots,\eta\_{k\_{2}}, the following set is linearly independent w.r.t xx:

|  |  |  |
| --- | --- | --- |
|  | {∂h2∂η(u2)​(x,ηi):i∈[k2],u2∈[d2]}.\displaystyle\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(x,\eta\_{i}):i\in[k\_{2}],\ u\_{2}\in[d\_{2}]\Bigg{\}}. |  |

Examples.
It can be validated that even linear experts of the form h2​(x,(η1,η0)):=η1⊤​x+η0h\_{2}(x,(\eta\_{1},\eta\_{0})):=\eta\_{1}^{\top}x+\eta\_{0} satisfy the weak identifiability condition. Note that the strong identifiability condition in Definition [1](#Thmdefinition1 "Definition 1 (Strong Identifiability). ‣ 2 On Shared Expert Strategy") implies the weak identifiability condition. Therefore,
two-layer FFNs h2​(x,(η2,η1,η0)):=η2​ReLU​(η1⊤​x+η0)h\_{2}(x,(\eta\_{2},\eta\_{1},\eta\_{0})):=\eta\_{2}\mathrm{ReLU}(\eta\_{1}^{\top}x+\eta\_{0}) are also weakly identifiable. On the other hand, input-free experts h2​(x,η)=c​(η)h\_{2}(x,\eta)=c(\eta) does not meet the weak identifiability condition.

Voronoi loss. Now, we build a Voronoi loss to capture parameter estimation rates under the dense regime, which is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟4​((G1,G2),(G1∗,Gˇ2)):=∑j=1k1∗|∑i∈𝒱1,jωi−ωj∗|+∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωi​(‖Δ​κi​j‖+|Δ​τi​j|)\displaystyle\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2})):=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}-\omega\_{j}^{\*}\Big{|}+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|+|\Delta\tau\_{ij}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|>1∑i∈𝒱1,jωi(∥Δκi​j∥2+|Δτi​j|2)+∑j=1k2∗∑i∈𝒱2,j(∥β1​i−βˇ1​j∥+|β0​i−βˇ0​j|\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|^{2}+|\Delta\tau\_{ij}|^{2})+\sum\_{j=1}^{k^{\*}\_{2}}\sum\_{i\in\mathcal{V}\_{2,j}}(\|\beta\_{1i}-\check{\beta}\_{1j}\|+|\beta\_{0i}-\check{\beta}\_{0j}| |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∥ηi−ηˇj∥+|νi−νˇj|).\displaystyle\hskip 227.62204pt+\|\eta\_{i}-\check{\eta}\_{j}\|+|\nu\_{i}-\check{\nu}\_{j}|). |  | (11) |

Given the above loss, we are now ready to present results for the convergence rates of parameter estimation and expert estimation in Theorem [4](#Thmtheorem4 "Theorem 4. ‣ A.2 Dense Regime ‣ Appendix A On Normalized Sigmoid Gating"), whose proof can be found in Appendix [D.4](#A4.SS4 "D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results").

###### Theorem 4.

Suppose that the shared expert function h1h\_{1} is strongly identifiable, while the routed expert function h2h\_{2} is weakly identifiable. Then, the lower bound

|  |  |  |
| --- | --- | --- |
|  | inf(G1∗,Gˇ2)∈𝒢¯k1,k2​(Θ)𝔼X[V(gG1,G2(⋅|X),gG1∗,Gˇ2(⋅|X))]≳𝒟4((G1,G2),(G1∗,Gˇ2))\inf\_{(G^{\*}\_{1},\check{G}\_{2})\in\overline{\mathcal{G}}\_{k\_{1},k\_{2}}(\Theta)}\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2})) |  |

holds for any (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). As a consequence, we have

|  |  |  |
| --- | --- | --- |
|  | inf(G1∗,Gˇ2)∈𝒢¯k1,k2​(Θ)𝒟4(G~1n,G~2n),(G1∗,Gˇ2))=𝒪P([log(n)/n]12).\displaystyle\inf\_{(G^{\*}\_{1},\check{G}\_{2})\in\overline{\mathcal{G}}\_{k\_{1},k\_{2}}(\Theta)}\mathcal{D}\_{4}(\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}),(G^{\*}\_{1},\check{G}\_{2}))=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

It can be seen from the formulation of the Voronoi loss 𝒟4\mathcal{D}\_{4} that the estimation rates for shared experts remain unchanged compared to those in Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating"), which are of the orders 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}) for exactly-specified ones and 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) for over-specified ones. However, there are changes in the estimation rates for routed experts.

*(i) Routed experts:* In particular, the convergence rates of parameter estimation η~in\widetilde{\eta}^{n}\_{i} are of parametric order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}). Since the routed expert function h2​(x,η)h\_{2}(x,\eta) is Lipschitz continuous w.r.t its parameter η\eta, then the rates for estimating both exactly-specified and over-specified routed experts are of order 𝒪~P​(n−1/2)\widetilde{\mathcal{O}}\_{P}(n^{-1/2}). These rates are substantially faster than those when using the standard softmax gating in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy") and Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.2 Linear Experts ‣ 2 On Shared Expert Strategy"), which are of orders 𝒪~P​(n−1/4)\widetilde{\mathcal{O}}\_{P}(n^{-1/4}) and 𝒪~P​(n−1/r2​(|𝒱2,j|))\widetilde{\mathcal{O}}\_{P}(n^{-1/r\_{2}(|\mathcal{V}\_{2,j}|)}), respectively.

*(ii) Sample efficiency of the normalized sigmoid gating:* As a result, when using the normalized sigmoid gating, then we need only 𝒪​(ϵ−2)\mathcal{O}(\epsilon^{-2}) to approximate routed experts with a given error ϵ\epsilon, even if they are of linear form. On the other hand, when using the softmax gating, it requires 𝒪​(ϵ−4)\mathcal{O}(\epsilon^{-4}) data points to estimate strongly identifiable experts. Furthermore, if the routed experts are of linear form, then we need 𝒪​(ϵ−r2​(|𝒱2,j|))\mathcal{O}(\epsilon^{-r\_{2}(|\mathcal{V}\_{2,j}|)}) data points to estimate, which is equivalent to 𝒪​(ϵ−12)\mathcal{O}(\epsilon^{-12}) when these routed experts have three fitted experts, that is, |𝒱2,j|=3|\mathcal{V}\_{2,j}|=3. Hence, we claim that the normalized sigmoid gating helps improve the sample efficiency of DeepSeekMoE.

## Appendix B Systems of Polynomial Equations

In this appendix, we will provide a formal definition of the functions r1r\_{1} and r2r\_{2} involved in the Voronoi loss 𝒟2\mathcal{D}\_{2} defined in equation ([2.2](#S2.Ex7 "2.2 Linear Experts ‣ 2 On Shared Expert Strategy")).

Definition of the function r1r\_{1}. To capture estimation rates for shared expert parameters in Section [2.2](#S2.SS2 "2.2 Linear Experts ‣ 2 On Shared Expert Strategy"), it is necessary to consider the solvability of a system of polynomial equations previously studied in [[25](#bib.bib25)]. More specifically, for each m≥2m\geq 2, let r1​(m)r\_{1}(m) be the smallest natural number rr such that the system:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1m∑n1,n2∈ℕ:n1+2​n2=ℓs3​i2​s1​in1​s2​in2n1!​n2!=0,ℓ=1,2,…,r,\displaystyle\sum\_{i=1}^{m}\sum\_{\begin{subarray}{c}n\_{1},n\_{2}\in\mathbb{N}:\\ n\_{1}+2n\_{2}=\ell\end{subarray}}\dfrac{s^{2}\_{3i}~s^{n\_{1}}\_{1i}~s^{n\_{2}}\_{2i}}{n\_{1}!~n\_{2}!}=0,\quad\ell=1,2,\ldots,r, |  | (12) |

does not admit any non-trivial solutions for the unknown variables {s1​i,s2​i,s3​i}i=1m\{s\_{1i},s\_{2i},s\_{3i}\}\_{i=1}^{m}. Here, we call a solution non-trivial if all the values of s3​is\_{3i} are non-zero, whereas at least one among s1​is\_{1i} is different from zero. In the following proposition, we provide the values of the function r1r\_{1} at some specific points m∈ℕm\in\mathbb{N}.

###### Proposition 3 (Proposition 2.1, [[25](#bib.bib25)]).

For m=2m=2, we get r1​(m)=4r\_{1}(m)=4, while for m=3m=3, we have r1​(m)=6r\_{1}(m)=6. When m≥4m\geq 4, we have r1​(m)≥7r\_{1}(m)\geq 7.

The proof of Proposition [3](#Thmproposition3 "Proposition 3 (Proposition 2.1, [25]). ‣ Appendix B Systems of Polynomial Equations") can be found in [[25](#bib.bib25)].

Definition of the function r2r\_{2}. To characterize estimation rates for routed expert parameters in Section [2.2](#S2.SS2 "2.2 Linear Experts ‣ 2 On Shared Expert Strategy"), we need to take into account the solvability of another system of polynomial equations studied in [[56](#bib.bib56)], which is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i=1m∑α∈ℐℓ1,ℓ2t5​i2​t1​iα1​t2​iα2​t3​iα3​t4​iα4α1!​α2!​α3!​α4!=0,\displaystyle\sum\_{i=1}^{m}\sum\_{\alpha\in\mathcal{I}\_{\ell\_{1},\ell\_{2}}}\dfrac{t\_{5i}^{2}~t\_{1i}^{\alpha\_{1}}~t\_{2i}^{\alpha\_{2}}~t\_{3i}^{\alpha\_{3}}~t\_{4i}^{\alpha\_{4}}}{\alpha\_{1}!~\alpha\_{2}!~\alpha\_{3}!~\alpha\_{4}!}=0, |  | (13) |

for all ℓ1,ℓ2≥0\ell\_{1},\ell\_{2}\geq 0 satisfying 1≤ℓ1+ℓ2≤r1\leq\ell\_{1}+\ell\_{2}\leq r, where

|  |  |  |
| --- | --- | --- |
|  | ℐℓ1,ℓ2:={α=(αi)i=14∈ℕd×ℕd×ℕ×ℕ:α1+α2=ℓ1,α3+2​α4=ℓ2−|α2|}.\displaystyle\mathcal{I}\_{\ell\_{1},\ell\_{2}}:=\{\alpha=(\alpha\_{i})\_{i=1}^{4}\in\mathbb{N}^{d}\times\mathbb{N}^{d}\times\mathbb{N}\times\mathbb{N}:\alpha\_{1}+\alpha\_{2}=\ell\_{1},\alpha\_{3}+2\alpha\_{4}=\ell\_{2}-|\alpha\_{2}|\}. |  |

Then, we define r2​(m)r\_{2}(m) as the smallest natural number rr such that the system in equation ([13](#A2.E13 "In Appendix B Systems of Polynomial Equations")) has no non-trivial solutions for the unknown variables {t5​i,t1​i,t2​i,t3​i,t4​i}i=1m\{t\_{5i},t\_{1i},t\_{2i},t\_{3i},t\_{4i}\}\_{i=1}^{m}. Here, a solution is called non-trivial if all the values of t5​it\_{5i} are different from, while at least one among t4​it\_{4i} is non-zero. The following proposition provides a relation between the two functions r1r\_{1} and r2r\_{2} as well as specify the values of r2​(m)r\_{2}(m) at some points m∈ℕm\in\mathbb{N}.

###### Proposition 4 (Lemma 1, [[56](#bib.bib56)]).

The function r2r\_{2} is upper bounded by the function r1r\_{1}, that is, r2​(m)≤r1​(m)r\_{2}(m)\leq r\_{1}(m), for all m∈ℕm\in\mathbb{N}. In addition, we have r2​(2)=4r\_{2}(2)=4, r2​(3)=6r\_{2}(3)=6 and r​(m)≥7r(m)\geq 7 when m≥4m\geq 4.

The proof of Lemma [4](#Thmproposition4 "Proposition 4 (Lemma 1, [56]). ‣ Appendix B Systems of Polynomial Equations") can be found in [[56](#bib.bib56)].

## Appendix C Related Works

There have been two primary lines of works on understanding MoE models in the literature.

From a statistical perspective, Zeevi et al. [[80](#bib.bib80)] investigated the representation power of a mixture of generalized linear experts when using this model to approximate target functions belonging to a Sobolev class. Next, Mendes et al. [[50](#bib.bib50)] performed a convergence analysis of MLE under the MoE with
experts being polynomial regression models, offering an important insight for finding the optimal configuration of the number of experts and their sizes. After that, considering data generated from a Gaussian MoE with covariate-free gating, Ho et al. [[26](#bib.bib26)] established an *algebraic independence* condition on the location and scale functions of the Gaussian density to characterize which choices of this pair will lead to faster convergence rates of parameter estimation. Then, this analysis was extended to more practical yet challenging settings of dense and sparse softmax gating Gaussian MoE in [[56](#bib.bib56)] and [[54](#bib.bib54)], respectively. These works demonstrated that parameter and expert estimation rates hinged on the solvability of some systems of polynomial equations and became significantly slow as the number of experts increased. Lastly, Nguyen et al. [[55](#bib.bib55)] considered a MoE-based regression framework where the regression function took the form of MoE with standard softmax gating, dense-to-sparse gating, and hierarchical softmax gating, respectively. Their convergence analysis of least squares estimation provided critical implications on the design of expert structures. In particular, it indicated that feed-forward expert
networks equipped with the sigmoid function or the Gaussian linear error unit (GELU) activation function admitted estimation rates of polynomial orders, while experts of polynomial forms had much slower estimation rates, of exponential orders.

From a deep learning perspective, Chen et al. [[8](#bib.bib8)] took into account a classification problem with cluster structures using MoE models. In particular, they justified the ability of the gating network to learn the cluster-center features, enabling the model to separate a big complex problem into simpler ones, each of which will be handled by the corresponding specialized experts. Furthermore, theories for applications of MoE in continual learning [[39](#bib.bib39), [37](#bib.bib37)], domain adaptation [[53](#bib.bib53), [9](#bib.bib9)], and language modeling [[59](#bib.bib59), [17](#bib.bib17)] have also been extensively explored in the literature. Interestingly, self-attention mechanism in the Transformers architecture [[73](#bib.bib73)] has recently been shown to be represented by a mixture of linear experts with quadratic softmax gating [[2](#bib.bib2), [77](#bib.bib77)], leading to numerous advances in parameter-efficient fine-tuning methods [[71](#bib.bib71), [36](#bib.bib36)].

However, to the best of our knowledge, no prior work has been done to identify the theoretical properties of the DeepSeekMoE architecture.

## Appendix D Proof of Main Results

### D.1 Proof of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy")

Proof overview. Recall that our goal is to demonstrate that the following lower bound holds for any G∈𝒢k1,k2​(Θ)G\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]≳𝒟1((G1,G2),(G1∗,G2∗)).\displaystyle\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})). |  | (14) |

Our proof will be divided into two main parts. Firstly, we aim to establish the local part of the bound ([14](#A4.E14 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")), that is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  | (15) |

The above result implies that there exists a positive constant ε′\varepsilon^{\prime} such that

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))≤ε′𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  |

Then, we complete the proof by deriving the following global part of the bound ([14](#A4.E14 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  | (16) |

Proof for the local part ([15](#A4.E15 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")): Assume by contrary that the claim in equation ([15](#A4.E15 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")) does not hold. Then, we can find a sequence of mixing measure pairs (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) taking the form G1n:=∑i=1k1nωin​δ(κin,τin)G^{n}\_{1}:=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{i}^{n},\tau\_{i}^{n})}, G2n:=∑i=1k2nexp⁡(β0​in)​δ(β1​in,ηin,νin)G^{n}\_{2}:=\sum\_{i=1}^{k^{n}\_{2}}\exp(\beta\_{0i}^{n})\delta\_{(\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})} for n∈ℕn\in\mathbb{N} such that 𝒟1​n:=𝒟1​((G1n,G2n),(G1∗,G2∗))→0\mathcal{D}\_{1n}:=\mathcal{D}\_{1}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\to 0 and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(fG1n,G2n(⋅|X),fG1∗,G2∗(⋅|X))]/𝒟1​n→0,\displaystyle\mathbb{E}\_{X}[V(f\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{1n}\to 0, |  | (17) |

as n→∞n\to\infty. As our proof argument is asymptotic, we may assume that the number of shared and routed experts k1n,k2nk^{n}\_{1},k^{n}\_{2} do not vary with the sample size nn. In addition, we also assume that Voronoi cells are independent of nn, that is, 𝒱1,j1=𝒱1,j1​(G1n)\mathcal{V}\_{1,j\_{1}}=\mathcal{V}\_{1,j\_{1}}(G^{n}\_{1}) and 𝒱2,j2=𝒱2,j2​(G2n)\mathcal{V}\_{2,j\_{2}}=\mathcal{V}\_{2,j\_{2}}(G^{n}\_{2}), for all j1∈[k1∗]j\_{1}\in[k^{\*}\_{1}] and j2∈[k2∗]j\_{2}\in[k^{\*}\_{2}]. Then, we can represent the Voronoi loss 𝒟1​n\mathcal{D}\_{1n} as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟1​n=∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑j=1k2∗|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)|\displaystyle\mathcal{D}\_{1n}=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{j=1}^{k^{\*}\_{2}}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)+∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖+‖Δ​ηi​jn‖+|Δ​νi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|)+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|+\|\Delta\eta\_{ij}^{n}\|+|\Delta\nu\_{ij}^{n}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖2+|Δ​τi​jn|2)+∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖2+‖Δ​ηi​jn‖2+|Δ​νi​jn|2),\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2})+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|^{2}+\|\Delta\eta\_{ij}^{n}\|^{2}+|\Delta\nu\_{ij}^{n}|^{2}), |  | (18) |

where we denote Δ​κi​jn:=κin−κj∗\Delta\kappa\_{ij}^{n}:=\kappa^{n}\_{i}-\kappa\_{j}^{\*}, Δ​τi​jn:=τin−τj∗\Delta\tau\_{ij}^{n}:=\tau^{n}\_{i}-\tau\_{j}^{\*}, Δ​β1​i​jn:=β1​in−β1​j∗\Delta\beta\_{1ij}^{n}:=\beta^{n}\_{1i}-\beta\_{1j}^{\*}, Δ​ηi​jn:=ηin−ηj∗\Delta\eta\_{ij}^{n}:=\eta^{n}\_{i}-\eta\_{j}^{\*}, and Δ​νi​jn:=νin−νj∗\Delta\nu\_{ij}^{n}:=\nu^{n}\_{i}-\nu\_{j}^{\*}. Recall that 𝒟1​n→0\mathcal{D}\_{1n}\to 0 as n→∞n\to\infty, then it follows that ∑i∈𝒱1,jωin→ωj∗\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\to\omega\_{j}^{\*}, (κin,τin)→(κj∗,τj∗)(\kappa\_{i}^{n},\tau\_{i}^{n})\to(\kappa\_{j}^{\*},\tau\_{j}^{\*}) as n→∞n\to\infty for all i∈𝒱1,ji\in\mathcal{V}\_{1,j} and j∈[k1∗]j\in[k^{\*}\_{1}]. Furthermore, we also have ∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*}), (β1​in,ηin,νin)→(β1​j∗,ηj∗,νj∗)(\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})\to(\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) as n→∞n\to\infty for all i∈𝒱2,ji\in\mathcal{V}\_{2,j} and j∈[k2∗]j\in[k^{\*}\_{2}].

Subsequently, we partition the rest of this proof into three main stages:

Stage 1 - Density Decomposition: In this stage, we focus on decomposing the density difference fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X). For ease of presentation, let us denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X) | :=∑i=1k1nωin​π​(Y|h1​(X,κin),τin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{1}}\omega^{n}\_{i}\pi(Y|h\_{1}(X,\kappa^{n}\_{i}),\tau^{n}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1∗​(Y|X)\displaystyle q\_{G^{\*}\_{1}}(Y|X) | :=∑i=1k1∗ωi∗​π​(Y|h1​(X,κi∗),τi∗),\displaystyle:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2n​(Y|X)\displaystyle p\_{G^{n}\_{2}}(Y|X) | :=∑i=1k2nexp⁡((β1​in)⊤​X+β0​in)∑j=1k2nexp⁡((β1​jn)⊤​X+β0​jn)⋅π​(Y|h2​(X,ηin),νin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{2}}\frac{\exp((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})}{\sum\_{j=1}^{k^{n}\_{2}}\exp((\beta\_{1j}^{n})^{\top}X+\beta\_{0j}^{n})}\cdot\pi(Y|h\_{2}(X,\eta^{n}\_{i}),\nu\_{i}^{n}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2∗​(Y|X)\displaystyle p\_{G^{\*}\_{2}}(Y|X) | :=∑i=1k2∗exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)⋅π​(Y|h2​(X,ηi∗),νi∗).\displaystyle:=\sum\_{i=1}^{k^{\*}\_{2}}\frac{\exp((\beta\_{1i}^{\*})^{\top}X+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})}\cdot\pi(Y|h\_{2}(X,\eta^{\*}\_{i}),\nu\_{i}^{\*}). |  |

Then, we have

|  |  |  |
| --- | --- | --- |
|  | fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)=12​[(qG1n​(Y|X)−qG1∗​(Y|X))+(pG2n​(Y|X)−pG2∗​(Y|X))].\displaystyle f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)=\frac{1}{2}\left[(q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X))+(p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X))\right]. |  |

Stage 1.1: In this step, we decompose the term qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)−qG1∗​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k1∗(∑i∈𝒱1,jωin−ωj∗)​π​(Y|h1​(X,κj∗),τj∗)\displaystyle+\sum\_{j=1}^{k^{\*}\_{1}}\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=An,1​(Y|X)+An,2​(Y|X)+An,0​(Y|X).\displaystyle:=A\_{n,1}(Y|X)+A\_{n,2}(Y|X)+A\_{n,0}(Y|X). |  |

By applying the first-order and second-order Taylor expansions to the function π(Y|h1(X,κin),τin))\pi(Y|h\_{1}(X,\kappa^{n}\_{i}),\tau^{n}\_{i})) around the point (κj∗,τj∗)(\kappa^{\*}\_{j},\tau^{\*}\_{j}), respectively, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1​(Y|X)\displaystyle A\_{n,1}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​∑|α|=11α!​(Δ​κi​jn)α1​(Δ​τi​jn)α2⋅∂π∂κα1​∂τα2​(Y|h1​(X,κj∗),τj∗)+Rn,1​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{|\alpha|=1}\frac{1}{\alpha!}(\Delta\kappa\_{ij}^{n})^{\alpha\_{1}}(\Delta\tau\_{ij}^{n})^{\alpha\_{2}}\cdot\frac{\partial\pi}{\partial\kappa^{\alpha\_{1}}\partial\tau^{\alpha\_{2}}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,1}(Y|X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2​(Y|X)\displaystyle A\_{n,2}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​∑|α|=121α!​(Δ​κi​jn)α1​(Δ​τi​jn)α2⋅∂|α|π∂κα1​∂τα2​(Y|h1​(X,κj∗),τj∗)+Rn,2​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{|\alpha|=1}^{2}\frac{1}{\alpha!}(\Delta\kappa\_{ij}^{n})^{\alpha\_{1}}(\Delta\tau\_{ij}^{n})^{\alpha\_{2}}\cdot\frac{\partial^{|\alpha|}\pi}{\partial\kappa^{\alpha\_{1}}\partial\tau^{\alpha\_{2}}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,2}(Y|X), |  |

where Rn,1​(Y|X)R\_{n,1}(Y|X) and Rn,2​(Y|X)R\_{n,2}(Y|X) are the Taylor remainders such that Rn,1​(Y|X)/𝒟1​n→0R\_{n,1}(Y|X)/\mathcal{D}\_{1n}\to 0 as n→∞n\to\infty. By the chain rule, the first-order derivatives of the function π\pi with respect to its parameters κ\kappa and τ\tau are given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂π∂κ(u1)​(Y|h1​(X,κj∗),τj∗)\displaystyle\frac{\partial\pi}{\partial\kappa^{(u\_{1})}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) | =∂h1∂κ(u1)​(X,κj∗)​∂π∂h1​(Y|h1​(X,κj∗),τj∗),\displaystyle=\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂π∂τ​(Y|h1​(X,κj∗),τj∗)\displaystyle\frac{\partial\pi}{\partial\tau}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) | =12​∂2π∂h12​(Y|h1​(X,κj∗),τj∗),\displaystyle=\frac{1}{2}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), |  |

for all u1∈[d1]u\_{1}\in[d\_{1}]. Analogously, the second-order derivatives of the function π\pi w.r.t its parameters are calculated as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2π∂κ(u1)​∂κ(v1)​(Y|h1​(X,κj∗),τj∗)\displaystyle\frac{\partial^{2}\pi}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) | =∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\displaystyle=\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)​∂2π∂h12​(Y|h1​(X,κj∗),τj∗),\displaystyle\quad+\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2π∂τ2​(Y|h1​(X,κj∗),τj∗)\displaystyle\frac{\partial^{2}\pi}{\partial\tau^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) | =14​∂4π∂h14​(Y|h1​(X,κj∗),τj∗),\displaystyle=\frac{1}{4}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2π∂κ(u1)​∂τ​(Y|h1​(X,κj∗),τj∗)\displaystyle\frac{\partial^{2}\pi}{\partial\kappa^{(u\_{1})}\partial\tau}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) | =12​∂h1∂κ(u1)​(X,κj∗)​∂3π∂h13​(Y|h1​(X,κj∗),τj∗),\displaystyle=\frac{1}{2}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), |  |

for all u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}]. Combine the above results, we can rewrite An,1​(Y|X)A\_{n,1}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1​(Y|X)\displaystyle A\_{n,1}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1[An,1,1(j)​(X)​∂π∂h1​(Y|h1​(X,κj∗),τj∗)+An,1,2(j)​(X)​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)]+Rn,1​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\Big{[}A^{(j)}\_{n,1,1}(X)\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+A^{(j)}\_{n,1,2}(X)\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}+R\_{n,1}(Y|X), |  |

where we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,1(j)​(X)\displaystyle A^{(j)}\_{n,1,1}(X) | :=∑i∈𝒱1,jωin​∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,2(j)​(X)\displaystyle A^{(j)}\_{n,1,2}(X) | :=∑i∈𝒱1,jωin​12​(Δ​τi​jn),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{2}(\Delta\tau\_{ij}^{n}), |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|=1|\mathcal{V}\_{1,j}|=1. Similarly, the quantity An,2​(Y|X)A\_{n,2}(Y|X) can be represented as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2​(Y|X)\displaystyle A\_{n,2}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|>1[An,2,1(j)(X)∂π∂h1(Y|h1(X,κj∗),τj∗)+An,2,2(j)(X)∂2π∂h12(Y|h1(X,κj∗),τj∗)\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\Big{[}~A^{(j)}\_{n,2,1}(X)\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+A^{(j)}\_{n,2,2}(X)\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +An,2,3(j)(X)∂3π∂h13(Y|h1(X,κj∗),τj∗)+An,2,4(j)(X)∂4π∂h14(Y|h1(X,κj∗),τj∗)]+Rn,2(Y|X),\displaystyle+A^{(j)}\_{n,2,3}(X)\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+A^{(j)}\_{n,2,4}(X)\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}+R\_{n,2}(Y|X), |  |

where we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,1(j)​(X)\displaystyle A^{(j)}\_{n,2,1}(X) | :=∑i∈𝒱1,jωin​(∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,2(j)​(X)\displaystyle A^{(j)}\_{n,2,2}(X) | :=∑i∈𝒱1,jωin​(12​(Δ​τi​jn)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\frac{1}{2}(\Delta\tau\_{ij}^{n})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,3(j)​(X)\displaystyle A^{(j)}\_{n,2,3}(X) | :=∑i∈𝒱1,jωin​∑u1=1d112​(Δ​κi​jn)(u1)​(Δ​τi​jn)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}\frac{1}{2}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,4(j)​(X)\displaystyle A^{(j)}\_{n,2,4}(X) | :=∑i∈𝒱1,jωin​18​(Δ​τi​jn)2,\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{8}(\Delta\tau\_{ij}^{n})^{2}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|>1|\mathcal{V}\_{1,j}|>1.

Stage 1.2: In this step, we decompose the term Qn​(Y|X):=[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]⋅[pG2n​(Y|X)−pG2∗​(Y|X)]Q\_{n}(Y|X):=\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot[p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X)]. By denoting F​(Y|X;β1,η,ν):=exp⁡(β1⊤​X)​π​(Y|h2​(X,η),ν)F(Y|X;\beta\_{1},\eta,\nu):=\exp(\beta\_{1}^{\top}X)\pi(Y|h\_{2}(X,\eta),\nu) and H​(Y|X;β1):=exp⁡(β1⊤​X)​pG2​(Y|X)H(Y|X;\beta\_{1}):=\exp(\beta\_{1}^{\top}X)p\_{G\_{2}}(Y|X), we can represent Qn​(Y|X)Q\_{n}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Qn​(Y|X)\displaystyle Q\_{n}(Y|X) | =∑j=1k2∗∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,ηin,νin)−F​(Y|X;β1​j∗,ηj∗,νj∗)]\displaystyle=\sum\_{j=1}^{k^{\*}\_{2}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −∑j=1k2∗∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle-\sum\_{j=1}^{k^{\*}\_{2}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k2∗(∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗))​[F​(Y|X;β1​j∗,ηj∗,νj∗)−H​(Y|X;β1​j∗)]\displaystyle+\sum\_{j=1}^{k^{\*}\_{2}}\Big{(}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{)}[F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn​(Y|X)−Cn​(Y|X)+En​(Y|X).\displaystyle:=B\_{n}(Y|X)-C\_{n}(Y|X)+E\_{n}(Y|X). |  |

Stage 1.2.1: In this step, we decompose the term Bn​(Y|X)B\_{n}(Y|X):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn​(Y|X)\displaystyle B\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,ηin,νin)−F​(Y|X;β1​j∗,ηj∗,νj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,ηin,νin)−F​(Y|X;β1​j∗,ηj∗,νj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn,1​(Y|X)+Bn,2​(Y|X).\displaystyle:=B\_{n,1}(Y|X)+B\_{n,2}(Y|X). |  |

By applying the first-order and second-order Taylor expansions to the function F​(Y|X;β1​in,ηin,νin)F(Y|X;\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n}) around the point (β1​j∗,ηj∗,νj∗)(\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}), we have

|  |  |  |
| --- | --- | --- |
|  | Bn,1​(Y|X)=∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​∑|α|=11α!​(Δ​β1​i​jn)α1​(Δ​ηi​jn)α2​(Δ​νi​jn)α3\displaystyle B\_{n,1}(Y|X)=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{|\alpha|=1}\frac{1}{\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{ij}^{n})^{\alpha\_{2}}(\Delta\nu\_{ij}^{n})^{\alpha\_{3}} |  |
|  |  |  |
| --- | --- | --- |
|  | ×∂F∂β1α1​∂ηα2​∂να3​(Y|X;β1​j∗,ηj∗,νj∗)+Rn,3​(Y|X),\displaystyle\times\frac{\partial F}{\partial\beta\_{1}^{\alpha\_{1}}\partial\eta^{\alpha\_{2}}\partial\nu^{\alpha\_{3}}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})+R\_{n,3}(Y|X), |  |
|  |  |  |
| --- | --- | --- |
|  | Bn,2​(Y|X)=∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​∑|α|=121α!​(Δ​β1​i​jn)α1​(Δ​ηi​jn)α2​(Δ​νi​jn)α3\displaystyle B\_{n,2}(Y|X)=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{|\alpha|=1}^{2}\frac{1}{\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{ij}^{n})^{\alpha\_{2}}(\Delta\nu\_{ij}^{n})^{\alpha\_{3}} |  |
|  |  |  |
| --- | --- | --- |
|  | ×∂|α|F∂β1α1​∂ηα2​∂να3​(Y|X;β1​j∗,ηj∗,νj∗)+Rn,4​(Y|X),\displaystyle\times\frac{\partial^{|\alpha|}F}{\partial\beta\_{1}^{\alpha\_{1}}\partial\eta^{\alpha\_{2}}\partial\nu^{\alpha\_{3}}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})+R\_{n,4}(Y|X), |  |

where Rn,3​(Y|X)R\_{n,3}(Y|X) and Rn,4​(Y|X)R\_{n,4}(Y|X) are the Taylor remainders such that Rn,3​(Y|X)/𝒟1​n→0R\_{n,3}(Y|X)/\mathcal{D}\_{1n}\to 0 and Rn,4​(Y|X)/𝒟1​n→0R\_{n,4}(Y|X)/\mathcal{D}\_{1n}\to 0 as n→∞n\to\infty. By means of the chain rule, the first-order derivatives of the function FF w.r.t its parameters β1,η,ν\beta\_{1},\eta,\nu are given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂F∂β1(u)​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial F}{\partial\beta\_{1}^{(u)}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =X(u)​exp⁡((β1​j∗)⊤​X)​π​(Y|h2​(X,ηj∗),νj∗),\displaystyle=X^{(u)}\exp((\beta\_{1j}^{\*})^{\top}X)\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂F∂η(u2)​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial F}{\partial\eta^{(u\_{2})}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =∂h2∂η(u2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗),\displaystyle=\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂F∂ν​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial F}{\partial\nu}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =12​exp⁡((β1​j∗)⊤​X)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗),\displaystyle=\frac{1}{2}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |

for all u2∈[d2]u\_{2}\in[d\_{2}]. Similarly, we can derive the second-order derivatives of the function FF w.r.t its parameters as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂β1(u)​∂β1(v)​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =X(u)​X(v)​exp⁡((β1​j∗)⊤​X)​π​(Y|h2​(X,ηj∗),νj∗),\displaystyle=X^{(u)}X^{(v)}\exp((\beta\_{1j}^{\*})^{\top}X)\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂η(u2)​∂η(v2)​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =∂2h2∂η(u2)​∂η(v2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\displaystyle=\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∂h2∂η(u2)​(X,ηj∗)​∂h2∂η(v2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗),\displaystyle+\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂ν2​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\nu^{2}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =14​exp⁡((β1​j∗)⊤​X)​∂4π∂h24​(Y|h2​(X,ηj∗),νj∗),\displaystyle=\frac{1}{4}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |

and

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂β1(u)​∂η(v2)​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\beta\_{1}^{(u)}\partial\eta^{(v\_{2})}}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =X(u)​∂h2∂η(v2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗),\displaystyle=X^{(u)}\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂β1(u)​∂ν​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\beta\_{1}^{(u)}\partial\nu}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =12​X(u)​exp⁡((β1​j∗)⊤​X)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗),\displaystyle=\frac{1}{2}X^{(u)}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂2F∂η(u2)​∂ν​(Y|X;β1​j∗,ηj∗,νj∗)\displaystyle\frac{\partial^{2}F}{\partial\eta^{(u\_{2})}\partial\nu}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) | =12​∂h2∂η(u2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X)​∂3π∂h23​(Y|h2​(X,ηj∗),νj∗),\displaystyle=\frac{1}{2}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{3}\pi}{\partial h\_{2}^{3}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |

for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}]. Putting the above results together, we can rewrite Bn,1​(Y|X)B\_{n,1}(Y|X) as

|  |  |  |
| --- | --- | --- |
|  | Bn,1(Y|X)=∑j∈[k2∗]:|𝒱2,j|=1[Bn,1,0(j)(X)π(Y|h2(X,ηj∗),νj∗)+Bn,1,1(j)(X)∂π∂h2(Y|h2(X,ηj∗),νj∗)\displaystyle B\_{n,1}(Y|X)=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}B^{(j)}\_{n,1,0}(X)\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+B^{(j)}\_{n,1,1}(X)\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +Bn,1,2(j)(X)∂2π∂h22(Y|h2(X,ηj∗),νj∗)]+Rn,3(Y|X),\displaystyle+B^{(j)}\_{n,1,2}(X)\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]}+R\_{n,3}(Y|X), |  |

where we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,0(j)​(X)\displaystyle B^{(j)}\_{n,1,0}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​∑u=1d(Δ​β1​i​jn)(u)​X(u)​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}X^{(u)}\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,1(j)​(X)\displaystyle B^{(j)}\_{n,1,1}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​∑u2=1d2(Δ​ηi​jn)(u2)​∂h2∂η(u2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,2(j)​(X)\displaystyle B^{(j)}\_{n,1,2}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​12​(Δ​νi​jn)​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\frac{1}{2}(\Delta\nu\_{ij}^{n})\exp((\beta\_{1j}^{\*})^{\top}X), |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|=1|\mathcal{V}\_{2,j}|=1. Analogously, we can represent the term Bn,2​(Y|X)B\_{n,2}(Y|X) as

|  |  |  |
| --- | --- | --- |
|  | Bn,2​(Y|X)=∑j∈[k2∗]:|𝒱2,j|=1∑ρ=04Bn,2,ρ(j)​(X)​∂ρπ∂h2ρ​(Y|h2​(X,ηj∗),νj∗)+Rn,4​(Y|X),\displaystyle B\_{n,2}(Y|X)=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{\rho=0}^{4}B^{(j)}\_{n,2,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{2}^{\rho}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+R\_{n,4}(Y|X), |  |

where we define

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,0(j)​(X)\displaystyle B^{(j)}\_{n,2,0}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​[∑u=1d(Δ​β1​i​jn)(u)​X(u)+∑u,v=1d(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)1+1{u=v}​X(u)​X(v)]​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\Bigg{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}X^{(u)}+\sum\_{u,v=1}^{d}\frac{(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}}{1+1\_{\{u=v\}}}X^{(u)}X^{(v)}\Bigg{]}\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,1(j)​(X)\displaystyle B^{(j)}\_{n,2,1}(X) | :=∑i∈𝒱2,jexp(β0​in)[∑u2=1d2(Δηi​jn)(u2)∂h2∂η(u2)(X,ηj∗)+∑u2,v2=1d2(Δ​ηi​jn)(u2)​(Δ​ηi​jn)(v2)1+1{u2=v2}∂2h2∂η(u2)​∂η(v2)(X,ηj∗)\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\Bigg{[}\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑u=1d∑v2=1d2(Δβ1​i​jn)(u)(Δηi​jn)(v2)X(u)∂h2∂η(v2)(X,ηj∗)]exp((β1​j∗)⊤X),\displaystyle+\sum\_{u=1}^{d}\sum\_{v\_{2}=1}^{d\_{2}}(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\eta\_{ij}^{n})^{(v\_{2})}X^{(u)}\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\Bigg{]}\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,2(j)​(X)\displaystyle B^{(j)}\_{n,2,2}(X) | :=∑i∈𝒱2,jexp(β0​in)[12(Δνi​jn)+∑u2,v2=1d2(Δ​ηi​jn)(u2)​(Δ​ηi​jn)(v2)1+1{u2=v2}∂h2∂η(u2)(X,ηj∗)∂h2∂η(v2)(X,ηj∗)\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\Bigg{[}\frac{1}{2}(\Delta\nu\_{ij}^{n})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑u=1d12(Δβ1​i​jn)(u)(Δνi​jn)X(u)]exp((β1​j∗)⊤X),\displaystyle\hskip 113.81102pt+\sum\_{u=1}^{d}\frac{1}{2}(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\nu\_{ij}^{n})X^{(u)}\Bigg{]}\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,3(j)​(X)\displaystyle B^{(j)}\_{n,2,3}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​∑u2=1d212​(Δ​ηi​jn)(u2)​(Δ​νi​jn)​∂h2∂η(u2)​(X,ηj∗)​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{u\_{2}=1}^{d\_{2}}\frac{1}{2}(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\nu\_{ij}^{n})\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\exp((\beta\_{1j}^{\*})^{\top}X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,4(j)​(X)\displaystyle B^{(j)}\_{n,2,4}(X) | :=∑i∈𝒱2,jexp⁡(β0​in)​18​(Δ​νi​jn)2​exp⁡((β1​j∗)⊤​X),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\frac{1}{8}(\Delta\nu\_{ij}^{n})^{2}\exp((\beta\_{1j}^{\*})^{\top}X), |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|>1|\mathcal{V}\_{2,j}|>1.

Stage 1.2.2: In this step, we decompose the term Cn​(Y|X)C\_{n}(Y|X):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn​(Y|X)\displaystyle C\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Cn,1​(Y|X)+Cn,2​(Y|X).\displaystyle:=C\_{n,1}(Y|X)+C\_{n,2}(Y|X). |  |

By means of the first-order and second-order Taylor expansions to the function H​(Y|X;β1​in)H(Y|X;\beta\_{1i}^{n}) around the point β1​j∗\beta\_{1j}^{\*}, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,1​(Y|X)\displaystyle C\_{n,1}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​∑u=1d(Δ​β1​i​jn)(u)​X(u)​H​(Y|X;β1​j∗)+Rn,5​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}X^{(u)}H(Y|X;\beta\_{1j}^{\*})+R\_{n,5}(Y|X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,2​(Y|X)\displaystyle C\_{n,2}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp(β0​in)[∑u=1d(Δβ1​i​jn)(u)X(u)H(Y|X;β1​j∗)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\Bigg{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}X^{(u)}H(Y|X;\beta\_{1j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑u,v=1d(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)1+1{u=v}X(u)X(v)H(Y|X;β1​j∗)]+Rn,6(Y|X),\displaystyle\hskip 42.67912pt+\sum\_{u,v=1}^{d}\frac{(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}}{1+1\_{\{u=v\}}}X^{(u)}X^{(v)}H(Y|X;\beta\_{1j}^{\*})\Bigg{]}+R\_{n,6}(Y|X), |  |

where Rn,5​(Y|X)R\_{n,5}(Y|X) and Rn,6​(Y|X)R\_{n,6}(Y|X) are the Taylor remainders such that Rn,5​(Y|X)/𝒟1​n→0R\_{n,5}(Y|X)/\mathcal{D}\_{1n}\to 0 and Rn,6​(Y|X)/𝒟1​n→0R\_{n,6}(Y|X)/\mathcal{D}\_{1n}\to 0 as n→∞n\to\infty.

Putting the above decompositions together, we can view An,0​(Y|X)/𝒟1​nA\_{n,0}(Y|X)/\mathcal{D}\_{1n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟1​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{1n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟1​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{1n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟1​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{1n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟1​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{1n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟1​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{1n}, [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟1​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{1n}, and En​(Y|X)/𝒟1​nE\_{n}(Y|X)/\mathcal{D}\_{1n} as a combination of elements of the following sets

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮0,j\displaystyle\mathcal{S}\_{0,j} | :={π​(Y|h1​(X,κj∗),τj∗)},\displaystyle:=\{\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮1,j\displaystyle\mathcal{S}\_{1,j} | :={∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗),∂2h1∂κ(u1)​∂κ(v1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮2,j\displaystyle\mathcal{S}\_{2,j} | :={∂2π∂h12(Y|h1(X,κj∗),τj∗),∂h1∂κ(u1)(X,κj∗)∂h1∂κ(v1)(X,κj∗)∂2π∂h12(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮3,j\displaystyle\mathcal{S}\_{3,j} | :={∂h1∂κ(u1)(X,κj∗)∂3π∂h13(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮4,j\displaystyle\mathcal{S}\_{4,j} | :={∂4π∂h14(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}], and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯0,j\displaystyle\mathcal{T}\_{0,j} | :={F(Y|X;β1​j∗,ηj∗,νj∗),X(u2)F(Y|X;β1​j∗,ηj∗,νj∗),X(u2)X(v2)F(Y|X;β1​j∗,ηj∗,νj∗):u2,v2∈[d2]},\displaystyle:=\{F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}),\ X^{(u\_{2})}F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}),\ X^{(u\_{2})}X^{(v\_{2})}F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}):u\_{2},v\_{2}\in[d\_{2}]\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯1,j\displaystyle\mathcal{T}\_{1,j} | :={∂h2∂η(u2)(X,ηj∗)F1(Y|X;β1​j∗,ηj∗,νj∗),∂2h2∂η(u2)​∂η(v2)(X,ηj∗)F1(Y|X;β1​j∗,ηj∗,νj∗),\displaystyle:=\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})F\_{1}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}),\ \frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})F\_{1}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | X(u2)∂h2∂η(u2)(X,ηj∗)F1(Y|X;β1​j∗,ηj∗,νj∗):u2,v2∈[d2]},\displaystyle\hskip 28.45274ptX^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})F\_{1}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}):u\_{2},v\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯2,j\displaystyle\mathcal{T}\_{2,j} | :={F2(Y|X;β1​j∗,ηj∗,νj∗),∂h2∂η(u2)(X,ηj∗)∂h2∂η(v2)(X,ηj∗)F2(Y|X;β1​j∗,ηj∗,νj∗),\displaystyle:=\Bigg{\{}F\_{2}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}),\ \frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})F\_{2}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | X(u2)F2(Y|X;β1​j∗,ηj∗,νj∗):u2,v2∈[d2]},\displaystyle\hskip 28.45274ptX^{(u\_{2})}F\_{2}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}):u\_{2},v\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯3,j\displaystyle\mathcal{T}\_{3,j} | :={∂h2∂η(u2)(X,ηj∗)F3(Y|X;β1​j∗,ηj∗,νj∗):u2∈[d2]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})F\_{3}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}):u\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯4,j\displaystyle\mathcal{T}\_{4,j} | :={F4​(Y|X;β1​j∗,ηj∗,νj∗)},\displaystyle:=\{F\_{4}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*})\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯5,j\displaystyle\mathcal{T}\_{5,j} | :={H(Y|X;β1​j∗),X(u)H(Y|X;β1​j∗),X(u)X(v)H(Y|X;β1​j∗):u,v∈[d]},\displaystyle:=\{H(Y|X;\beta\_{1j}^{\*}),\ X^{(u)}H(Y|X;\beta\_{1j}^{\*}),\ X^{(u)}X^{(v)}H(Y|X;\beta\_{1j}^{\*}):u,v\in[d]\}, |  |

where we denote

|  |  |  |
| --- | --- | --- |
|  | Fρ​(Y|X;β1​j∗,ηj∗,νj∗):=exp⁡((β1​j∗)⊤​X)​∂ρπ∂h1ρ​(Y|h2​(X,ηj∗),νj∗),\displaystyle F\_{\rho}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}):=\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |

for all ρ∈[4]\rho\in[4] and j∈[k2∗]j\in[k^{\*}\_{2}].

Stage 2 - Non-vanishing coefficients: In this stage, we show by contradiction that not all the coefficients in the representations of An,0​(Y|X)/𝒟1​nA\_{n,0}(Y|X)/\mathcal{D}\_{1n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟1​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{1n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟1​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{1n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟1​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{1n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟1​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{1n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟1​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{1n}, [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟1​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{1n}, and En​(Y|X)/𝒟1​nE\_{n}(Y|X)/\mathcal{D}\_{1n} converge to zero as n→∞n\to\infty. In particular, we assume that all those coefficients go to zero. Then, by looking into the coefficients of the terms:

* •

  π​(Y|h1​(X,κj∗),τj∗)\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) for j∈[k1∗]j\in[k^{\*}\_{1}], we have 1𝒟1​n⋅∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|→0\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}\to 0;
* •

  ∂h1∂κ(u1)​(X,κj∗)​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) for j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1 and u1∈[d1]u\_{1}\in[d\_{1}], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​‖Δ​κi​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\|\Delta\kappa\_{ij}^{n}\|\to 0; |  |
* •

  ∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) for j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​|Δ​τi​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}|\Delta\tau\_{ij}^{n}|\to 0; |  |
* •

  [∂h1∂κ(u1)​(X,κj∗)]2​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\big{[}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\big{]}^{2}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) for j∈[k1∗]:|𝒱1,j|>1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1 and u1∈[d1]u\_{1}\in[d\_{1}], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​‖Δ​κi​jn‖2→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\|\Delta\kappa\_{ij}^{n}\|^{2}\to 0; |  |
* •

  ∂4π∂h14​(Y|h1​(X,κj∗),τj∗)\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) for j∈[k1∗]:|𝒱1,j|>1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1 and u1∈[d1]u\_{1}\in[d\_{1}], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​|Δ​τi​jn|2→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}|\Delta\tau\_{ij}^{n}|^{2}\to 0; |  |
* •

  F​(Y|X;β1​j∗,ηj∗,νj∗)F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]j\in[k^{\*}\_{2}], we have 1𝒟1​n⋅∑j=1k2∗|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β1​j∗)|→0\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j=1}^{k^{\*}\_{2}}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{1j}^{\*})\Big{|}\to 0;
* •

  X(u)​F​(Y|X;β1​j∗,ηj∗,νj∗)X^{(u)}F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1 and u∈[d]u\in[d], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​β1​i​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\beta\_{1ij}^{n}\|\to 0; |  |
* •

  ∂h2∂η(u2)​F1​(Y|X;β1​j∗,ηj∗,νj∗)\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}F\_{1}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1 and u2∈[d2]u\_{2}\in[d\_{2}], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​ηi​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\eta\_{ij}^{n}\|\to 0; |  |
* •

  F2​(Y|X;β1​j∗,ηj∗,νj∗)F\_{2}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​|Δ​νi​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})|\Delta\nu\_{ij}^{n}|\to 0; |  |
* •

  X(u)​X(v)​F​(Y|X;β1​j∗,ηj∗,νj∗)X^{(u)}X^{(v)}F(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1 and u,v∈[d]u,v\in[d], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​β1​i​jn‖2→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\beta\_{1ij}^{n}\|^{2}\to 0; |  |
* •

  [∂2h2∂η(u2)​(X,ηj∗)]2​F2​(Y|X;β1​j∗,ηj∗,νj∗)\big{[}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\big{]}^{2}F\_{2}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1 and u2∈[d2]u\_{2}\in[d\_{2}], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​ηi​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\eta\_{ij}^{n}\|\to 0; |  |
* •

  F4​(Y|X;β1​j∗,ηj∗,νj∗)F\_{4}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟1​n⋅∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​|Δ​νi​jn|2→0;\displaystyle\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})|\Delta\nu\_{ij}^{n}|^{2}\to 0; |  |

Taking the sum of the above limits, we deduce 1=1𝒟1​n⋅𝒟1​n→01=\frac{1}{\mathcal{D}\_{1n}}\cdot\mathcal{D}\_{1n}\to 0 as n→∞n\to\infty, which is a contradiction. Thus, at least one among the coefficients in the representations of An,0​(Y|X)/𝒟1​nA\_{n,0}(Y|X)/\mathcal{D}\_{1n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟1​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{1n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟1​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{1n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟1​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{1n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟1​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{1n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟1​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{1n}, [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟1​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{1n}, and En​(Y|X)/𝒟1​nE\_{n}(Y|X)/\mathcal{D}\_{1n} does not converge to zero.

Stage 3 - Fatou’s lemma contradiction: In this stage, we use the Fatou’s lemma to show a contradiction to the result of Stage 2. For that purpose, let us denote mnm\_{n} as the maximum of the absolute values of the coefficients in the representations of An,0​(Y|X)/𝒟1​nA\_{n,0}(Y|X)/\mathcal{D}\_{1n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟1​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{1n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟1​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{1n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟1​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{1n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟1​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{1n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟1​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{1n}, [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟1​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{1n}, and En​(Y|X)/𝒟1​nE\_{n}(Y|X)/\mathcal{D}\_{1n}. It follows from the result of Stage 2 that 1/mn↛∞1/m\_{n}\not\to\infty as n→∞n\to\infty. In addition, we also denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)→s1,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\to s^{(u\_{1})}\_{1,j}, | 1mn​𝒟1​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)→s2,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})\to s\_{2,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)→s3,j(u1​v1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}\to s^{(u\_{1}v\_{1})}\_{3,j}, | 1mn​𝒟1​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)2→s4,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})^{2}\to s\_{4,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​τi​jn)→s5,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\to s^{(u\_{1})}\_{5,j}, | 1mn​𝒟1​n⋅(∑i∈𝒱1,jωin−ωj∗)→s0,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\to s\_{0,j}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅(∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗))→t0,j,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\Big{(}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{)}\to t\_{0,j}, | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​β1​i​jn)(u)→t1,j(u),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\beta\_{1ij}^{n})^{(u)}\to t^{(u)}\_{1,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​ηi​jn)(u2)→t2,j(u2),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\eta\_{ij}^{n})^{(u\_{2})}\to t^{(u\_{2})}\_{2,j}, | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​νi​jn)→t3,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\nu\_{ij}^{n})\to t\_{3,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)→t4,j(u​v),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}\to t^{(uv)}\_{4,j}, | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​ηi​jn)(u2)​(Δ​ηi​jn)(v2)→t5,j(u2​v2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}\to t^{(u\_{2}v\_{2})}\_{5,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​νi​jn)2→t6,j,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\nu\_{ij}^{n})^{2}\to t\_{6,j}, | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​β1​i​jn)(u)​(Δ​ηi​jn)(v2)→t7,j(u​v2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\eta\_{ij}^{n})^{(v\_{2})}\to t^{(uv\_{2})}\_{7,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​β1​i​jn)(u)​(Δ​νi​jn)→t8,j(u),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\nu\_{ij}^{n})\to t^{(u)}\_{8,j}, | 1mn​𝒟1​n⋅∑i∈𝒱2,jexp⁡(β0​in)​(Δ​ηi​jn)(u2)​(Δ​νi​jn)→t9,j(u2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{1n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\nu\_{ij}^{n})\to t^{(u\_{2})}\_{9,j}, |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] as n→∞n\to\infty. Due to the result of Stage 2, at least one among the above limits is different from zero. Recall from equation ([17](#A4.E17 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")) that we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(fG1n,G2n(⋅|X),fG1∗,G2∗(⋅|X))]/𝒟1​n→0,\displaystyle\mathbb{E}\_{X}[V(f\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{1n}\to 0, |  |

Furthermore, according to the Fatou’s lemma, we get

|  |  |  |
| --- | --- | --- |
|  | limn→∞𝔼X[V(fG1n,G2n(⋅|X),fG1∗,G2∗(⋅|X))]mn​𝒟1​n≥∫lim infn→∞|fG1n,G2n(Y|X)−fG1∗,G2∗(Y|X)|2​mn​𝒟1​n​d​(X,Y).\displaystyle\lim\_{n\to\infty}\dfrac{\mathbb{E}\_{X}[V(f\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{m\_{n}\mathcal{D}\_{1n}}\geq\int\liminf\_{n\to\infty}\dfrac{|f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)|}{2m\_{n}\mathcal{D}\_{1n}}\mathrm{d}(X,Y). |  |

Then, we deduce [fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)]/[mn​𝒟1​n]→0[f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{1n}]\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). Since the input space is bounded and the parameter space is compact, the quantity ∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*}) is bounded. Thus, we also have

|  |  |  |
| --- | --- | --- |
|  | [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​[fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)]/[mn​𝒟1​n]→0,\displaystyle\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}[f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{1n}]\to 0, |  |

implying that

|  |  |  |
| --- | --- | --- |
|  | 12​[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]⋅qG1n​(Y|X)−qG1∗​(Y|X)mn​𝒟1​n+12​Qn​(Y|X)mn​𝒟1​n→0.\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot\dfrac{q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}+\frac{1}{2}\dfrac{Q\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}\to 0. |  |

as n→∞n\to\infty for almost surely (X,Y)(X,Y). From the decomposition of the terms qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) and Qn​(Y|X)Q\_{n}(Y|X) in Stage 1, we have

|  |  |  |
| --- | --- | --- |
|  | 12​[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]⋅An,2​(Y|X)+An,1​(Y|X)+An,0​(Y|X)mn​𝒟1​n\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot\dfrac{A\_{n,2}(Y|X)+A\_{n,1}(Y|X)+A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{1n}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | +12​Bn,1​(Y|X)+Bn,2​(Y|X)−Cn,1​(Y|X)−Cn,2​(Y|X)+En​(Y|X)mn​𝒟1​n→0.\displaystyle+\frac{1}{2}\dfrac{B\_{n,1}(Y|X)+B\_{n,2}(Y|X)-C\_{n,1}(Y|X)-C\_{n,2}(Y|X)+E\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}\to 0. |  | (19) |

Denote Fρ,j​(Y|X):=Fρ​(Y|X;β1​j∗,ηj∗,νj∗)F\_{\rho,j}(Y|X):=F\_{\rho}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) and Hj​(Y|X):=H​(Y|X,β1​j∗)H\_{j}(Y|X):=H(Y|X,\beta\_{1j}^{\*}), we have

|  |  |  |
| --- | --- | --- |
|  | limn→∞An,0​(Y|X)mn​𝒟1​n=∑j=1k1∗s0,j​π​(Y|h1​(X,κj∗),τj∗),\displaystyle\lim\_{n\to\infty}\frac{A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j=1}^{k^{\*}\_{1}}s\_{0,j}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,1​(Y|X)mn​𝒟1​n=∑j∈[k1∗]:|𝒱1,j|=1[∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗)\displaystyle\lim\_{n\to\infty}\frac{A\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\Big{[}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +12s2,j∂2π∂h12(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 284.52756pt+\frac{1}{2}s\_{2,j}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,2​(Y|X)mn​𝒟1​n=∑j∈[k1∗]:|𝒱1,j|>1[(∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}∂2h1∂κ(u1)​∂κ(v1)(X,κj∗))\displaystyle\lim\_{n\to\infty}\frac{A\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\Big{[}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)} |  |
|  |  |  |
| --- | --- | --- |
|  | ×∂π∂h1​(Y|h1​(X,κj∗),τj∗)+(12​s2,j+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗))​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\displaystyle\times\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\Big{(}\frac{1}{2}s\_{2,j}+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +(12∑u1=1d1s5,j(u1)∂h1∂κ(u1)(X,κj∗))∂3π∂h13(Y|h1(X,κj∗),τj∗)+18s4,j∂4π∂h14(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 85.35826pt+\Big{(}\frac{1}{2}\sum\_{u\_{1}=1}^{d\_{1}}s\_{5,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\frac{1}{8}s\_{4,j}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn,1​(Y|X)mn​𝒟1​n=∑j∈[k2∗]:|𝒱2,j|=1[∑u=1dt1,j(u)​X(u)​F0,j​(Y|X)+∑u2=1d2t2,j(u2)​∂h2∂η(u2)​(X,ηj∗)​F1,j​(Y|X)+12​t3,j​F2,j​(Y|X)],\displaystyle\lim\_{n\to\infty}\frac{B\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}F\_{0,j}(Y|X)+\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})F\_{1,j}(Y|X)+\frac{1}{2}t\_{3,j}F\_{2,j}(Y|X)\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn,2​(Y|X)mn​𝒟1​n=∑j∈[k2∗]:|𝒱2,j|>1[(∑u=1dt1,j(u)X(u)+∑u,v=1dt4,j(u​v)X(u)X(v))F0,j(Y|X)\displaystyle\lim\_{n\to\infty}\frac{B\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{[}\Big{(}\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}+\sum\_{u,v=1}^{d}t\_{4,j}^{(uv)}X^{(u)}X^{(v)}\Big{)}F\_{0,j}(Y|X) |  |
|  |  |  |
| --- | --- | --- |
|  | +(∑u2=1d2t2,j(u2)​∂h2∂η(u2)​(X,ηj∗)+∑u2,v2=1d2t5,j(u2​v2)​∂2h2∂η(u2)​∂η(v2)​(X,ηj∗)+∑u=1d∑v2=1d2t7,j(u​v2)​X(u)​∂h2∂η(v2)​(X,ηj∗))​F1,j​(Y|X)\displaystyle+\Big{(}\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j}^{(u\_{2}v\_{2})}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u=1}^{d}\sum\_{v\_{2}=1}^{d\_{2}}t\_{7,j}^{(uv\_{2})}X^{(u)}\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\Big{)}F\_{1,j}(Y|X) |  |
|  |  |  |
| --- | --- | --- |
|  | +(12​t3,j+∑u2,v2=1d2t5,j(u2​v2)​∂h2∂η(u2)​(X,ηj∗)​∂h2∂η(v2)​(X,ηj∗)+∑u=1d12​t8,j(u)​X(u))​F2,j​(Y|X)\displaystyle+\Big{(}\frac{1}{2}t\_{3,j}+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j}^{(u\_{2}v\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u=1}^{d}\frac{1}{2}t\_{8,j}^{(u)}X^{(u)}\Big{)}F\_{2,j}(Y|X) |  |
|  |  |  |
| --- | --- | --- |
|  | +(∑u2=1d212t9,j(u2)∂h2∂η(u2)(X,ηj∗))F3,j(Y|X)+18t6,jF4,j(Y|X)],\displaystyle+\Big{(}\sum\_{u\_{2}=1}^{d\_{2}}\frac{1}{2}t\_{9,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\Big{)}F\_{3,j}(Y|X)+\frac{1}{8}t\_{6,j}F\_{4,j}(Y|X)\Big{]}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn,1​(Y|X)mn​𝒟1​n=∑j∈[k2∗]:|𝒱2,j|=1∑u=1dt1,j(u)​X(u)​Hj​(Y|X),\displaystyle\lim\_{n\to\infty}\frac{C\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}H\_{j}(Y|X), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn,2​(Y|X)mn​𝒟1​n=∑j∈[k2∗]:|𝒱2,j|>1(∑u=1dt1,j(u)​X(u)+∑u,v=1dt4,j(u​v)​X(u)​X(v))​Hj​(Y|X),\displaystyle\lim\_{n\to\infty}\frac{C\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{(}\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}+\sum\_{u,v=1}^{d}t\_{4,j}^{(uv)}X^{(u)}X^{(v)}\Big{)}H\_{j}(Y|X), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞En​(Y|X)mn​𝒟1​n=∑j=1k2∗t0,j​[F0,j​(Y|X)−Hj​(Y|X)].\displaystyle\lim\_{n\to\infty}\frac{E\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{1n}}=\sum\_{j=1}^{k^{\*}\_{2}}t\_{0,j}[F\_{0,j}(Y|X)-H\_{j}(Y|X)]. |  |

It is worth noting that for almost every XX, the set

|  |  |  |
| --- | --- | --- |
|  | {[∑j=1k2∗exp((β1​j∗)⊤X+β0​j∗)]∂ρπ∂h1ρ(Y|h1(X,κj∗),τj∗):0≤ρ≤4,j∈[k1∗]}\displaystyle\Bigg{\{}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):0\leq\rho\leq 4,\ j\in[k^{\*}\_{1}]\Bigg{\}} |  |
|  |  |  |
| --- | --- | --- |
|  | ∪{Fρ(Y|X;β1​j∗,ηj∗,νj∗),H(Y|X;β1​j∗):0≤ρ≤4,j∈[k2∗]}\displaystyle\cup~\Big{\{}F\_{\rho}(Y|X;\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}),\ H(Y|X;\beta\_{1j}^{\*}):0\leq\rho\leq 4,\ j\in[k^{\*}\_{2}]\Big{\}} |  |

is linearly independent w.r.t YY. Therefore, it follows that the coefficients of those terms in the limit in equation ([19](#A4.E19 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")) become zero.

For j∈[k1∗]j\in[k^{\*}\_{1}], by looking at the coefficient of the term [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​π​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s0,j=0s\_{0,j}=0.

For j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|=1|\mathcal{V}\_{1,j}|=1, by considering the coefficients of

* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have ∑u1=1d1s1,j(u1)​∂h1∂κ(u1)​(X,κj∗)=0\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})=0 for almost every XX. Since the expert function h1h\_{1} is strongly identifiable, we get s1,j(u1)=0s\_{1,j}^{(u\_{1})}=0 for all u1∈[d1]u\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s2,j=0s\_{2,j}=0.

For j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|>1|\mathcal{V}\_{1,j}|>1, by considering the coefficients of

* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | ∑u1=1d1s1,j​∂h1∂κ(u1)​(X,κj∗)+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)=0,\displaystyle\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})=0, |  |

  for almost every XX. Since the expert function h1h\_{1} satisfies the strong identifiability condition, we get s1,j(u1)=s3,j(u1​v1)=0s\_{1,j}^{(u\_{1})}=s\_{3,j}^{(u\_{1}v\_{1})}=0 for all u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | 12​s2,j+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)=0,\displaystyle\frac{1}{2}s\_{2,j}+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})=0, |  |

  for almost every XX. Since s3,j(u1​v1)=0s\_{3,j}^{(u\_{1}v\_{1})}=0 for all u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}], we deduce s2,j=0s\_{2,j}=0;
* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂3π∂h13​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have 12​∑u1=1d1s5,j(u1)​∂h1∂κ(u1)​(X,κj∗)=0\frac{1}{2}\sum\_{u\_{1}=1}^{d\_{1}}s\_{5,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})=0, for almost every XX. As the expert function h1h\_{1} meets the strong identifiability condition, we get s5,j(u1)=0s\_{5,j}^{(u\_{1})}=0 for all u1∈[d1]u\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​∂4π∂h14​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s4,j=0s\_{4,j}=0.

For j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|=1|\mathcal{V}\_{2,j}|=1, by considering the coefficients of

* •

  F0,j​(Y|X)F\_{0,j}(Y|X), we have t0,j+∑u=1dt1,j(u)​X(u)=0t\_{0,j}+\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}=0, for almost every XX. Then, we deduce t0,j=t1,j(u)=0t\_{0,j}=t\_{1,j}^{(u)}=0 for all u∈[d]u\in[d];
* •

  F1,j​(Y|X)F\_{1,j}(Y|X), we have ∑u2=1d2t2,j(u2)​∂h2∂η(u2)​(X,ηj∗)\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*}), for almost every XX. As the expert function h2h\_{2} is strongly identifiable, we get t2,j(u2)=0t\_{2,j}^{(u\_{2})}=0 for all u2∈[d2]u\_{2}\in[d\_{2}];
* •

  F2,j​(Y|X)F\_{2,j}(Y|X), we have t3,j=0t\_{3,j}=0.

For j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|>1|\mathcal{V}\_{2,j}|>1, by considering the coefficients of

* •

  F0,j​(Y|X)F\_{0,j}(Y|X), we have t0,j+∑u=1dt1,j(u)​X(u)+∑u,v=1dt4,j(u​v)​X(u)​X(v)=0t\_{0,j}+\sum\_{u=1}^{d}t\_{1,j}^{(u)}X^{(u)}+\sum\_{u,v=1}^{d}t\_{4,j}^{(uv)}X^{(u)}X^{(v)}=0, for almost surely XX. Then, we get t0,j=t1,j(u)=t4,j(u​v)t\_{0,j}=t\_{1,j}^{(u)}=t\_{4,j}^{(uv)} for all u,v∈[d]u,v\in[d].
* •

  F1,j​(Y|X)F\_{1,j}(Y|X), we have

  |  |  |  |
  | --- | --- | --- |
  |  | ∑u2=1d2t2,j(u2)​∂h2∂η(u2)​(X,ηj∗)+∑u2,v2=1d2t5,j(u2​v2)​∂2h2∂η(u2)​∂η(v2)​(X,ηj∗)+∑u=1d∑v2=1d2t7,j(u​v2)​X(u)​∂h2∂η(v2)​(X,ηj∗)=0,\displaystyle\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j}^{(u\_{2}v\_{2})}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u=1}^{d}\sum\_{v\_{2}=1}^{d\_{2}}t\_{7,j}^{(uv\_{2})}X^{(u)}\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})=0, |  |

  for almost every XX. As the expert function h2h\_{2} meets the strong identifiability condition, we get t2,j(u2)=t5,j(u2​v2)=t7,j(u​v2)=0t\_{2,j}^{(u\_{2})}=t\_{5,j}^{(u\_{2}v\_{2})}=t\_{7,j}^{(uv\_{2})}=0 for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}] and u∈[d]u\in[d];
* •

  F2,j​(Y|X)F\_{2,j}(Y|X), we have

  |  |  |  |
  | --- | --- | --- |
  |  | 12​t3,j+∑u2,v2=1d2t5,j(u2​v2)​∂h2∂η(u2)​(X,ηj∗)​∂h2∂η(v2)​(X,ηj∗)+∑u=1d12​t8,j(u)​X(u)=0,\displaystyle\frac{1}{2}t\_{3,j}+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j}^{(u\_{2}v\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u=1}^{d}\frac{1}{2}t\_{8,j}^{(u)}X^{(u)}=0, |  |

  for almost every XX. Since t5,j(u2​v2)=0t\_{5,j}^{(u\_{2}v\_{2})}=0 for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}], we deduce 12​t3,j+∑u=1d12​t8,j(u)​X(u)=0\frac{1}{2}t\_{3,j}+\sum\_{u=1}^{d}\frac{1}{2}t\_{8,j}^{(u)}X^{(u)}=0, for almost every XX. Then, we get t3,j=t8,j(u)=0t\_{3,j}=t\_{8,j}^{(u)}=0 for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}] and u∈[d]u\in[d];
* •

  F3,j​(Y|X)F\_{3,j}(Y|X), we have ∑u2=1d212​t9,j(u2)​∂h2∂η(u2)​(X,ηj∗)=0\sum\_{u\_{2}=1}^{d\_{2}}\frac{1}{2}t\_{9,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})=0, for almost every XX. As the expert function h2h\_{2} is strongly identifiable, we get t9,j(u2)t\_{9,j}^{(u\_{2})} for all u2∈[d2]u\_{2}\in[d\_{2}];
* •

  F4,j​(Y|X)F\_{4,j}(Y|X), we have t6,j=0t\_{6,j}=0.

Putting the above results together, we have (i) s0,j=s1,j(u1)=s2,j=s3,j(u1​v1)=s4,j=s5,j(u1)=0s\_{0,j}=s\_{1,j}^{(u\_{1})}=s\_{2,j}=s\_{3,j}^{(u\_{1}v\_{1})}=s\_{4,j}=s\_{5,j}^{(u\_{1})}=0 for all j∈[k1∗]j\in[k^{\*}\_{1}] and u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}]; (ii) t0,j=t1,j(u)=t2,j(u2)=t3,j=t4,j(u​v)=t5,j(u2​v2)=t6,j=t7,ju​v2=t8,j(u)=t9,j(u2)=0t\_{0,j}=t\_{1,j}^{(u)}=t\_{2,j}^{(u\_{2})}=t\_{3,j}=t\_{4,j}^{(uv)}=t\_{5,j}^{(u\_{2}v\_{2})}=t\_{6,j}=t\_{7,j}^{uv\_{2}}=t\_{8,j}^{(u)}=t\_{9,j}^{(u\_{2})}=0 for all j∈[k2∗]j\in[k^{\*}\_{2}], u,v∈[d]u,v\in[d] and u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}]. This contradicts to the fact that at least one among them is non-zero. Consequently, we achieve the local part in equation ([15](#A4.E15 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")), that is,

|  |  |  |
| --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  |

The local part indicates that there exists a positive constant ε′\varepsilon^{\prime} such that

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  |

Proof for the global part ([16](#A4.E16 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")): Thus, it is sufficient to establish the global part

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟1​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  |

Suppose that the global part does not hold, then there exists a sequence of mixing measure pairs (G~1n,G~2n)(\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}) satisfying 𝒟1​((G~1n,G~2n),(G1∗,G2∗))>ε′\mathcal{D}\_{1}((\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime} and limn→∞𝔼X[V(fG~1n,G~2n(⋅|X),fG1∗,G2∗(⋅|X))]𝒟1​((G~1n,G~2n),(G1∗,G2∗))=0\lim\_{n\to\infty}\frac{\mathbb{E}\_{X}[V(f\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{1}((\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}=0. In other words, we have

|  |  |  |
| --- | --- | --- |
|  | limn→∞𝔼X[V(fG~1n,G~2n(⋅|X),fG1∗,G2∗(⋅|X))]=0.\displaystyle\lim\_{n\to\infty}\mathbb{E}\_{X}[V(f\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]=0. |  |

Recall that the parameter space Θ\Theta is compact, then we can replace the sequence (G~1n,G~2n)(\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}) by one of its subsequences which converges to some pair of mixing measures (G~1,G~2)(\tilde{G}\_{1},\tilde{G}\_{2}). Due to the fact that 𝒟1​((G~1n,G~2n),(G1∗,G2∗))>ε′\mathcal{D}\_{1}((\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}, we get 𝒟1​((G~1,G~2),(G1∗,G2∗))>ε′\mathcal{D}\_{1}((\tilde{G}\_{1},\tilde{G}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}. Next, by applying the Fatou’s lemma, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 0=limn→∞𝔼X[V(fG~1n,G~2n(⋅|X),fG1∗,G2∗(⋅|X))]\displaystyle 0=\lim\_{n\to\infty}\mathbb{E}\_{X}[V(f\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))] | ≥12∫lim infn→∞|fG~1n,G~2n(Y|X),fG1∗,G2∗(Y|X)|d(X,Y)\displaystyle\geq\frac{1}{2}\int\liminf\_{n\to\infty}\Big{|}f\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(Y|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)\Big{|}\mathrm{d}(X,Y) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =12∫|fG~1,G~2(Y|X)−fG1∗,G2∗(Y|X)|d(X,Y).\displaystyle=\frac{1}{2}\int\Big{|}f\_{\tilde{G}\_{1},\tilde{G}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)\Big{|}\mathrm{d}(X,Y). |  |

The above result implies that fG~1,G~2​(Y|X)=fG1∗,G2∗​(Y|X)f\_{\tilde{G}\_{1},\tilde{G}\_{2}}(Y|X)=f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) for almost surely (X,Y)(X,Y). According to Proposition [5](#Thmproposition5 "Proposition 5 (Identifiability). ‣ E.2 Identifiability of DeepSeekMoE ‣ Appendix E Proof of Auxiliary Results"), we deduce (G~1,G~2)≡(G1∗,G2∗)(\tilde{G}\_{1},\tilde{G}\_{2})\equiv(G^{\*}\_{1},G^{\*}\_{2}), indicating that 𝒟1​((G~1,G~2),(G1∗,G2∗))=0\mathcal{D}\_{1}((\tilde{G}\_{1},\tilde{G}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=0. This contradicts the fact that 𝒟1​((G~1,G~2),(G1∗,G2∗))>ε′>0\mathcal{D}\_{1}((\tilde{G}\_{1},\tilde{G}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}>0. Hence, we obtain the global part ([16](#A4.E16 "In D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results")) and complete the proof.

### D.2 Proof of Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.2 Linear Experts ‣ 2 On Shared Expert Strategy")

By employing arguments used in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), it is sufficient to establish the local part

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟2​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟2​((G1,G2),(G1∗,G2∗))>0,\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0, |  | (20) |

and the global part

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟2​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(fG1,G2(⋅|X),fG1∗,G2∗(⋅|X))]𝒟2​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(f\_{G\_{1},G\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{2}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  | (21) |

in this appendix. As the global part ([21](#A4.E21 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")) can be derived similarly to Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), we omit its proof here. Thus, we will focus on showing only the local part ([20](#A4.E20 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")). Assume by contrary that the local part is not true. Then, there exists a sequence of mixing measure pairs (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) taking the form G1n:=∑i=1k1nωin​δ(κ1​in,κ0​in,τin)G^{n}\_{1}:=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{1i}^{n},\kappa\_{0i}^{n},\tau\_{i}^{n})}, G2n:=∑i=1k2nexp⁡(β0​in)​δ(β1​in,η1​in,η0​in,νin)G^{n}\_{2}:=\sum\_{i=1}^{k^{n}\_{2}}\exp(\beta\_{0i}^{n})\delta\_{(\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})} for n∈ℕn\in\mathbb{N} such that 𝒟2​n:=𝒟2​((G1n,G2n),(G1∗,G2∗))→0\mathcal{D}\_{2n}:=\mathcal{D}\_{2}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\to 0 and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(fG1n,G2n(⋅|X),fG1∗,G2∗(⋅|X))]/𝒟2​n→0,\displaystyle\mathbb{E}\_{X}[V(f\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{2n}\to 0, |  | (22) |

as n→∞n\to\infty. Here, we may assume WLOG that the number of shared experts and routed experts k1nk^{n}\_{1}, k2nk^{n}\_{2} and Voronoi cells 𝒱1,j=𝒱1,j​(G1n)\mathcal{V}\_{1,j}=\mathcal{V}\_{1,j}(G^{n}\_{1}), 𝒱2,j=𝒱2,j​(G2n)\mathcal{V}\_{2,j}=\mathcal{V}\_{2,j}(G^{n}\_{2}) do not change with the sample size nn. Then, the Voronoi loss 𝒟2​n\mathcal{D}\_{2n} can be rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟2​n=∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑j∈[k2∗]:|𝒱2,j|>1|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)|\displaystyle\mathcal{D}\_{2n}=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κ1​i​jn‖+|Δ​κ0​i​jn|+|Δ​τi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{1ij}^{n}\|+|\Delta\kappa\_{0ij}^{n}|+|\Delta\tau\_{ij}^{n}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​(‖Δ​κ1​i​jn‖2+|Δ​κ0​i​jn|r1,j+|Δ​τi​jn|r1,j/2)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{1ij}^{n}\|^{2}+|\Delta\kappa\_{0ij}^{n}|^{r\_{1,j}}+|\Delta\tau\_{ij}^{n}|^{r\_{1,j}/2}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖+‖Δ​η1​i​jn‖+|Δ​η0​i​jn|+|Δ​νi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|+\|\Delta\eta\_{1ij}^{n}\|+|\Delta\eta\_{0ij}^{n}|+|\Delta\nu\_{ij}^{n}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖r2,j+‖Δ​η1​i​jn‖r2,j/2+|Δ​η0​i​jn|r2,j+|Δ​νi​jn|r2,j/2),\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|^{r\_{2,j}}+\|\Delta\eta\_{1ij}^{n}\|^{r\_{2,j}/2}+|\Delta\eta\_{0ij}^{n}|^{r\_{2,j}}+|\Delta\nu\_{ij}^{n}|^{r\_{2,j}/2}), |  | (23) |

where we denote Δ​κ1​i​jn:=κ1​in−κ1​j∗\Delta\kappa\_{1ij}^{n}:=\kappa\_{1i}^{n}-\kappa\_{1j}^{\*}, Δ​κ0​i​jn:=κ0​in−κ0​j∗\Delta\kappa\_{0ij}^{n}:=\kappa\_{0i}^{n}-\kappa\_{0j}^{\*}, Δ​η1​i​jn:=η1​in−η1​j∗\Delta\eta\_{1ij}^{n}:=\eta\_{1i}^{n}-\eta\_{1j}^{\*}, and Δ​η0​i​jn:=η0​in−η0​j∗\Delta\eta\_{0ij}^{n}:=\eta\_{0i}^{n}-\eta\_{0j}^{\*}. Since 𝒟2​n→0\mathcal{D}\_{2n}\to 0 as n→∞n\to\infty, then the above formulation indicates that as n→∞n\to\infty, we have
∑i∈𝒱1,jωin→ωj∗\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\to\omega\_{j}^{\*}, (κ1​in,κ0​in,τin)→(κ1​j∗,κ0​j∗,τj∗)(\kappa\_{1i}^{n},\kappa\_{0i}^{n},\tau\_{i}^{n})\to(\kappa\_{1j}^{\*},\kappa\_{0j}^{\*},\tau\_{j}^{\*}) as n→∞n\to\infty for all i∈𝒱1,ji\in\mathcal{V}\_{1,j} and j∈[k1∗]j\in[k^{\*}\_{1}]. Furthermore, we also have ∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*}), (β1​in,η1​in,η0​in,νin)→(β1​j∗,η1​j∗,η0​j∗,νj∗)(\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})\to(\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*}) as n→∞n\to\infty for all i∈𝒱2,ji\in\mathcal{V}\_{2,j} and j∈[k2∗]j\in[k^{\*}\_{2}].

Next, we divide the rest of this proof into three main steps:

Stage 1 - Density Decomposition: In this stage, we aim to decompose the density discrepancy fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X). For ease of presentation, we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X) | :=∑i=1k1nωin​π​(Y|(κ1​in)⊤​X+κ0​in,τin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{1}}\omega^{n}\_{i}\pi(Y|(\kappa\_{1i}^{n})^{\top}X+\kappa\_{0i}^{n},\tau^{n}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1∗​(Y|X)\displaystyle q\_{G^{\*}\_{1}}(Y|X) | :=∑i=1k1∗ωi∗​π​(Y|(κ1​j∗)⊤​X+κ0​j∗,τi∗),\displaystyle:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau^{\*}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2n​(Y|X)\displaystyle p\_{G^{n}\_{2}}(Y|X) | :=∑i=1k2nexp⁡((β1​in)⊤​X+β0​in)∑j=1k2nexp⁡((β1​jn)⊤​X+β0​jn)⋅π​(Y|(η1​in)⊤​X+η0​in,νin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{2}}\frac{\exp((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})}{\sum\_{j=1}^{k^{n}\_{2}}\exp((\beta\_{1j}^{n})^{\top}X+\beta\_{0j}^{n})}\cdot\pi(Y|(\eta\_{1i}^{n})^{\top}X+\eta\_{0i}^{n},\nu\_{i}^{n}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2∗​(Y|X)\displaystyle p\_{G^{\*}\_{2}}(Y|X) | :=∑i=1k2∗exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)⋅π​(Y|(η1​j∗)⊤​X+η0​j∗,νi∗).\displaystyle:=\sum\_{i=1}^{k^{\*}\_{2}}\frac{\exp((\beta\_{1i}^{\*})^{\top}X+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})}\cdot\pi(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{i}^{\*}). |  |

Given the above notations, we get

|  |  |  |
| --- | --- | --- |
|  | fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)=12​[(qG1n​(Y|X)−qG1∗​(Y|X))+(pG2n​(Y|X)−pG2∗​(Y|X))].\displaystyle f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)=\frac{1}{2}\left[(q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X))+(p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X))\right]. |  |

Stage 1.1: Firstly, we decompose the term qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)−qG1∗​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​[π​(Y|(κ1​in)⊤​X+κ0​in,τin)−π​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|(\kappa\_{1i}^{n})^{\top}X+\kappa\_{0i}^{n},\tau\_{i}^{n})-\pi(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​[π​(Y|(κ1​in)⊤​X+κ0​in,τin)−π​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|(\kappa\_{1i}^{n})^{\top}X+\kappa\_{0i}^{n},\tau\_{i}^{n})-\pi(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k1∗(∑i∈𝒱1,jωin−ωj∗)​π​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)\displaystyle+\sum\_{j=1}^{k^{\*}\_{1}}\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\pi(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau^{\*}\_{j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=An,1​(Y|X)+An,2​(Y|X)+An,0​(Y|X).\displaystyle:=A\_{n,1}(Y|X)+A\_{n,2}(Y|X)+A\_{n,0}(Y|X). |  |

By applying the first-order Taylor expansion to the function π​(Y|(κ1​in)⊤​X+κ0​in,τin)\pi(Y|(\kappa\_{1i}^{n})^{\top}X+\kappa\_{0i}^{n},\tau^{n}\_{i}) around the point (κ1​j∗,κ0​j∗,τj∗)(\kappa\_{1j}^{\*},\kappa\_{0j}^{\*},\tau\_{j}^{\*}), the term An,1​(Y|X)A\_{n,1}(Y|X) is rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1​(Y|X)\displaystyle A\_{n,1}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,j∑|α|=1ωinα!​(Δ​κ1​i​jn)α1​(Δ​κ0​i​jn)α2​(Δ​τi​jn)α3\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\sum\_{|\alpha|=1}\frac{\omega\_{i}^{n}}{\alpha!}(\Delta\kappa\_{1ij}^{n})^{\alpha\_{1}}(\Delta\kappa\_{0ij}^{n})^{\alpha\_{2}}(\Delta\tau\_{ij}^{n})^{\alpha\_{3}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×∂|α1|+α2+α3π∂κ1α1​∂κ2α2​∂τα3​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)+Rn,1​(Y|X)\displaystyle\hskip 113.81102pt\times\frac{\partial^{|\alpha\_{1}|+\alpha\_{2}+\alpha\_{3}}\pi}{\partial\kappa\_{1}^{\alpha\_{1}}\partial\kappa\_{2}^{\alpha\_{2}}\partial\tau^{\alpha\_{3}}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})+R\_{n,1}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,j∑|α|=1ωin2α3​α!​(Δ​κ1​i​jn)α1​(Δ​κ0​i​jn)α2​(Δ​τi​jn)α3\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\sum\_{|\alpha|=1}\frac{\omega\_{i}^{n}}{2^{\alpha\_{3}}\alpha!}(\Delta\kappa\_{1ij}^{n})^{\alpha\_{1}}(\Delta\kappa\_{0ij}^{n})^{\alpha\_{2}}(\Delta\tau\_{ij}^{n})^{\alpha\_{3}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×Xα1​∂|α1|+α2+2​α3π∂h1|α1|+α2+2​α3​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)+Rn,1​(Y|X)\displaystyle\hskip 113.81102pt\times X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\alpha\_{2}+2\alpha\_{3}}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\alpha\_{2}+2\alpha\_{3}}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})+R\_{n,1}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k1∗]:|𝒱1,j|=1∑|α1|=01∑ℓ=1{|α1|=0}2​(1−|α1|)An,α1,ℓ(j)⋅Xα1​∂|α1|+ℓπ∂h1|α1|+ℓ​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)+Rn,1​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{|\alpha\_{1}|=0}^{1}\sum\_{\ell=1\_{\{|\alpha\_{1}|=0\}}}^{2(1-|\alpha\_{1}|)}A^{(j)}\_{n,\alpha\_{1},\ell}\cdot X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\ell}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\ell}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})+R\_{n,1}(Y|X), |  |

where Rn,1​(Y|X)R\_{n,1}(Y|X) is a Taylor remainder such that Rn,1​(Y|X)/𝒟2​n→0R\_{n,1}(Y|X)/\mathcal{D}\_{2n}\to 0 as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,α1,ℓ(j)\displaystyle A^{(j)}\_{n,\alpha\_{1},\ell} | :=∑i∈𝒱1,j∑α2+2​α3=ℓωin2α3​α!​(Δ​κ1​i​jn)α1​(Δ​κ0​i​jn)α2​(Δ​τi​jn)α3,\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\sum\_{\alpha\_{2}+2\alpha\_{3}=\ell}\frac{\omega\_{i}^{n}}{2^{\alpha\_{3}}\alpha!}(\Delta\kappa\_{1ij}^{n})^{\alpha\_{1}}(\Delta\kappa\_{0ij}^{n})^{\alpha\_{2}}(\Delta\tau\_{ij}^{n})^{\alpha\_{3}}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}], α1∈ℕd\alpha\_{1}\in\mathbb{N}^{d} and ℓ∈ℕ\ell\in\mathbb{N} such that (α1,ℓ)≠(0d,0)(\alpha\_{1},\ell)\neq(0\_{d},0). Meanwhile, by applying the Taylor expansion of the order r1,j:=r1​(|𝒱1,j|)r\_{1,j}:=r\_{1}(|\mathcal{V}\_{1,j}|) to the function π​(Y|(κ1​in)⊤​X+κ0​in,τin)\pi(Y|(\kappa\_{1i}^{n})^{\top}X+\kappa\_{0i}^{n},\tau^{n}\_{i}) around the point (κ1​j∗,κ0​j∗,τj∗)(\kappa\_{1j}^{\*},\kappa\_{0j}^{\*},\tau\_{j}^{\*}), we rewrite the term An,2​(Y|X)A\_{n,2}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2​(Y|X)\displaystyle A\_{n,2}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|>1∑|α1|=0r1,j∑ℓ=1{|α1|=0}2​(r1,j−|α1|)An,α1,ℓ(j)⋅Xα1​∂|α1|+ℓπ∂h1|α1|+ℓ​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)+Rn,2​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{|\alpha\_{1}|=0}^{r\_{1,j}}\sum\_{\ell=1\_{\{|\alpha\_{1}|=0\}}}^{2(r\_{1,j}-|\alpha\_{1}|)}A^{(j)}\_{n,\alpha\_{1},\ell}\cdot X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\ell}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\ell}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*})+R\_{n,2}(Y|X), |  |

where Rn,2​(Y|X)R\_{n,2}(Y|X) is a Taylor remainder such that Rn,2​(Y|X)/𝒟2​n→R\_{n,2}(Y|X)/\mathcal{D}\_{2n}\to as n→∞n\to\infty.

Stage 1.2: Next, we attempt to decompose the term Qn​(Y|X):=[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]⋅[pG2n​(Y|X)−pG2∗​(Y|X)]Q\_{n}(Y|X):=\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot[p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X)]. By denoting F​(Y|X;β1,η1,η0,ν):=exp⁡(β1⊤​X)​π​(Y|(η1)⊤​X+η0,ν)F(Y|X;\beta\_{1},\eta\_{1},\eta\_{0},\nu):=\exp(\beta\_{1}^{\top}X)\pi(Y|(\eta\_{1})^{\top}X+\eta\_{0},\nu) and H​(Y|X;β1):=exp⁡(β1⊤​X)​pG2​(Y|X)H(Y|X;\beta\_{1}):=\exp(\beta\_{1}^{\top}X)p\_{G\_{2}}(Y|X), we can represent Qn​(Y|X)Q\_{n}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Qn​(Y|X)\displaystyle Q\_{n}(Y|X) | =∑j=1k2∗∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,η1​in,η0​in,νin)−F​(Y|X;β1​j∗,η1​j∗,η0​j∗,νj∗)]\displaystyle=\sum\_{j=1}^{k^{\*}\_{2}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −∑j=1k2∗∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle-\sum\_{j=1}^{k^{\*}\_{2}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k2∗(∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗))​[F​(Y|X;β1​j∗,η1​j∗,η0​j∗,νj∗)−H​(Y|X;β1​j∗)]\displaystyle+\sum\_{j=1}^{k^{\*}\_{2}}\Big{(}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{)}[F(Y|X;\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn​(Y|X)−Cn​(Y|X)+En​(Y|X).\displaystyle:=B\_{n}(Y|X)-C\_{n}(Y|X)+E\_{n}(Y|X). |  |

Stage 1.2.1: In this step, we decompose the term Bn​(Y|X)B\_{n}(Y|X):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn​(Y|X)\displaystyle B\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,η1​in,η0​in,νin)−F​(Y|X;β1​j∗,η1​j∗,η0​j∗,νj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​[F​(Y|X;β1​in,η1​in,η0​in,νin)−F​(Y|X;β1​j∗,η1​j∗,η0​j∗,νj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[F(Y|X;\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})-F(Y|X;\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn,1​(Y|X)+Bn,2​(Y|X).\displaystyle:=B\_{n,1}(Y|X)+B\_{n,2}(Y|X). |  |

By applying the first-order Taylor expansion to the function F​(Y|X;β1​in,η1​in,η0​in,νin)F(Y|X;\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n}) around the point (β1​j∗,η1​j∗,η0​j∗,νj∗)(\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*}), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1​(Y|X)\displaystyle B\_{n,1}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​∑|α|=11α!​(Δ​β1​i​jn)α1​(Δ​η1​i​jn)α2​(Δ​η0​i​jn)α3​(Δ​νi​jn)α4\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{|\alpha|=1}\frac{1}{\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{1ij}^{n})^{\alpha\_{2}}(\Delta\eta\_{0ij}^{n})^{\alpha\_{3}}(\Delta\nu\_{ij}^{n})^{\alpha\_{4}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×∂|α1|+|α2|+α3+α4F∂β1α1​∂η1α2​∂η0α3​∂να4​(Y|X;β1​j∗,η1​j∗,η0​j∗,νj∗)+Rn,3​(Y|X)\displaystyle\hskip 85.35826pt\times\frac{\partial^{|\alpha\_{1}|+|\alpha\_{2}|+\alpha\_{3}+\alpha\_{4}}F}{\partial\beta\_{1}^{\alpha\_{1}}\partial\eta\_{1}^{\alpha\_{2}}\partial\eta\_{0}^{\alpha\_{3}}\partial\nu^{\alpha\_{4}}}(Y|X;\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*})+R\_{n,3}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j∑|α|=1exp⁡(β0​in)2α4​α!​(Δ​β1​i​jn)α1​(Δ​η1​i​jn)α2​(Δ​η0​i​jn)α3​(Δ​νi​jn)α4\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{|\alpha|=1}\frac{\exp(\beta\_{0i}^{n})}{2^{\alpha\_{4}}\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{1ij}^{n})^{\alpha\_{2}}(\Delta\eta\_{0ij}^{n})^{\alpha\_{3}}(\Delta\nu\_{ij}^{n})^{\alpha\_{4}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×Xα1+α2​exp⁡((β1​j∗)⊤​X)​∂|α2|+α3+2​α4π∂h2|α2|+α3+2​α4​(Y|(η1​j∗)⊤​X+η0​j∗,νj∗)+Rn,3​(Y|X)\displaystyle\hskip 56.9055pt\times X^{\alpha\_{1}+\alpha\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{|\alpha\_{2}|+\alpha\_{3}+2\alpha\_{4}}\pi}{\partial h\_{2}^{|\alpha\_{2}|+\alpha\_{3}+2\alpha\_{4}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*})+R\_{n,3}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1∑|ℓ1|+ℓ2=12Bn,ℓ1,ℓ2(j)⋅Xℓ1​exp⁡((β1​j∗)⊤​X)​∂ℓ2π∂h2ℓ2​(Y|(η1​j∗)⊤​X+η0​j∗,νj∗)+Rn,3​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{|\ell\_{1}|+\ell\_{2}=1}^{2}B^{(j)}\_{n,\ell\_{1},\ell\_{2}}\cdot X^{\ell\_{1}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\ell\_{2}}\pi}{\partial h\_{2}^{\ell\_{2}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*})+R\_{n,3}(Y|X), |  |

where Rn,3​(Y|X)R\_{n,3}(Y|X) is the Taylor remainder such that Rn,3​(Y|X)/𝒟2​n→0R\_{n,3}(Y|X)/\mathcal{D}\_{2n}\to 0, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,ℓ1,ℓ2(j)\displaystyle B^{(j)}\_{n,\ell\_{1},\ell\_{2}} | :=∑i∈𝒱2,j∑α∈ℐℓ1,ℓ2exp⁡(β0​in)2α4​α!​(Δ​β1​i​jn)α1​(Δ​η1​i​jn)α2​(Δ​η0​i​jn)α3​(Δ​νi​jn)α4,\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{\alpha\in\mathcal{I}\_{\ell\_{1},\ell\_{2}}}\frac{\exp(\beta\_{0i}^{n})}{2^{\alpha\_{4}}\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{1ij}^{n})^{\alpha\_{2}}(\Delta\eta\_{0ij}^{n})^{\alpha\_{3}}(\Delta\nu\_{ij}^{n})^{\alpha\_{4}}, |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}], ℓ1∈ℕd\ell\_{1}\in\mathbb{N}^{d}, and ℓ2∈ℕ\ell\_{2}\in\mathbb{N} such that (ℓ1,ℓ2)≠(0d,0)(\ell\_{1},\ell\_{2})\neq(0\_{d},0), where we define

|  |  |  |
| --- | --- | --- |
|  | ℐℓ1,ℓ2:={α=(αi)i=14∈ℕd×ℕd×ℕ×ℕ:α1+α2=ℓ1,α3+2​α4=ℓ2−|α2|}.\displaystyle\mathcal{I}\_{\ell\_{1},\ell\_{2}}:=\{\alpha=(\alpha\_{i})\_{i=1}^{4}\in\mathbb{N}^{d}\times\mathbb{N}^{d}\times\mathbb{N}\times\mathbb{N}:\alpha\_{1}+\alpha\_{2}=\ell\_{1},\alpha\_{3}+2\alpha\_{4}=\ell\_{2}-|\alpha\_{2}|\}. |  |

By applying the Taylor expansion of the order r2,j:=r2​(|𝒱2,j|)r\_{2,j}:=r\_{2}(|\mathcal{V}\_{2,j}|) to the function F​(Y|X;β1​in,η1​in,η0​in,νin)F(Y|X;\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n}) around the point (β1​j∗,η1​j∗,η0​j∗,νj∗)(\beta\_{1j}^{\*},\eta\_{1j}^{\*},\eta\_{0j}^{\*},\nu\_{j}^{\*}), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2​(Y|X)\displaystyle B\_{n,2}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑|ℓ1|+ℓ2=12​r2,jBn,ℓ1,ℓ2(j)⋅Xℓ1​exp⁡((β1​j∗)⊤​X)​∂ℓ2π∂h2ℓ2​(Y|(η1​j∗)⊤​X+η0​j∗,νj∗)+Rn,4​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{|\ell\_{1}|+\ell\_{2}=1}^{2r\_{2,j}}B^{(j)}\_{n,\ell\_{1},\ell\_{2}}\cdot X^{\ell\_{1}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\ell\_{2}}\pi}{\partial h\_{2}^{\ell\_{2}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*})+R\_{n,4}(Y|X), |  |

where Rn,4​(Y|X)R\_{n,4}(Y|X) is the Taylor remainder such that Rn,4​(Y|X)/𝒟2​n→0R\_{n,4}(Y|X)/\mathcal{D}\_{2n}\to 0.

Stage 1.2.2: In this step, we decompose the term Cn​(Y|X)C\_{n}(Y|X):

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn​(Y|X)\displaystyle C\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​[H​(Y|X;β1​in)−H​(Y|X;β1​j∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})[H(Y|X;\beta\_{1i}^{n})-H(Y|X;\beta\_{1j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Cn,1​(Y|X)+Cn,2​(Y|X).\displaystyle:=C\_{n,1}(Y|X)+C\_{n,2}(Y|X). |  |

By means of the first-order and second-order Taylor expansions to the function H​(Y|X;β1​in)H(Y|X;\beta\_{1i}^{n}) around the point β1​j∗\beta\_{1j}^{\*}, the term Cn,1​(Y|X)C\_{n,1}(Y|X) can be represented as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,1​(Y|X)\displaystyle C\_{n,1}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​∑|γ|=11γ!​(Δ​β1​i​jn)γ​∂|γ|H∂β1γ​(Y|X;β1​j∗)+Rn,5​(Y|X)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\sum\_{|\gamma|=1}\frac{1}{\gamma!}(\Delta\beta\_{1ij}^{n})^{\gamma}\frac{\partial^{|\gamma|}H}{\partial\beta\_{1}^{\gamma}}(Y|X;\beta\_{1j}^{\*})+R\_{n,5}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j∑|γ|=1exp⁡(β0​in)γ!​(Δ​β1​i​jn)γ⋅Xγ​exp⁡((β1​j∗)⊤​X)​pG2n​(Y|X)+Rn,5​(Y|X)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{|\gamma|=1}\frac{\exp(\beta\_{0i}^{n})}{\gamma!}(\Delta\beta\_{1ij}^{n})^{\gamma}\cdot X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{n}\_{2}}(Y|X)+R\_{n,5}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1∑|γ|=1Cn,γ(j)⋅Xγ​exp⁡((β1​j∗)⊤​X)​pG2n​(Y|X)+Rn,5​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{|\gamma|=1}C^{(j)}\_{n,\gamma}\cdot X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{n}\_{2}}(Y|X)+R\_{n,5}(Y|X), |  |

where Rn,5​(Y|X)R\_{n,5}(Y|X) is the Taylor remainder such that Rn,5​(Y|X)/𝒟2​n→0R\_{n,5}(Y|X)/\mathcal{D}\_{2n}\to 0, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,γ(j)\displaystyle C^{(j)}\_{n,\gamma} | :=∑i∈𝒱2,jexp⁡(β0​in)γ!​(Δ​β1​i​jn)γ,\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\frac{\exp(\beta\_{0i}^{n})}{\gamma!}(\Delta\beta\_{1ij}^{n})^{\gamma}, |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] and γ∈ℕd∖{0d}\gamma\in\mathbb{N}^{d}\setminus\{0\_{d}\}. Analogously, by applying the second-order Taylor expansion to the function H​(Y|X;β1​in)H(Y|X;\beta\_{1i}^{n}) around the point β1​j∗\beta\_{1j}^{\*}, we represent the term Cn,2​(Y|X)C\_{n,2}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,2​(Y|X)\displaystyle C\_{n,2}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑|γ|=12Cn,γ(j)⋅Xγ​exp⁡((β1​j∗)⊤​X)​pG2n​(Y|X)+Rn,6​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{|\gamma|=1}^{2}C^{(j)}\_{n,\gamma}\cdot X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{n}\_{2}}(Y|X)+R\_{n,6}(Y|X), |  |

where Rn,6​(Y|X)R\_{n,6}(Y|X) is the Taylor remainder such that Rn,6​(Y|X)/𝒟2​n→0R\_{n,6}(Y|X)/\mathcal{D}\_{2n}\to 0.

Combining the above decompositions of An​(Y|X)A\_{n}(Y|X), Bn​(Y|X)B\_{n}(Y|X), and Cn​(Y|X)C\_{n}(Y|X) together, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]⋅[fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)]\displaystyle\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot[f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​12​∑j∈[k1∗]∑|α1|=0r1,j∑ℓ=02​(r1,j−|α1|)An,α1,ℓ(j)⋅Xα1​∂|α1|+ℓπ∂h1|α1|+ℓ​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)\displaystyle=\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{1}{2}\sum\_{j\in[k^{\*}\_{1}]}\sum\_{|\alpha\_{1}|=0}^{r\_{1,j}}\sum\_{\ell=0}^{2(r\_{1,j}-|\alpha\_{1}|)}A^{(j)}\_{n,\alpha\_{1},\ell}\cdot X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\ell}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\ell}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +12​∑j∈[k2∗]∑|ℓ1|+ℓ2=02​r2,jBn,ℓ1,ℓ2(j)⋅Xℓ1​exp⁡((β1​j∗)⊤​X)​∂ℓ2π∂h2ℓ2​(Y|(η1​j∗)⊤​X+η0​j∗,νj∗)\displaystyle+\frac{1}{2}\sum\_{j\in[k^{\*}\_{2}]}\sum\_{|\ell\_{1}|+\ell\_{2}=0}^{2r\_{2,j}}B^{(j)}\_{n,\ell\_{1},\ell\_{2}}\cdot X^{\ell\_{1}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\ell\_{2}}\pi}{\partial h\_{2}^{\ell\_{2}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −12​∑j∈[k2∗]∑|γ|=01+1{|𝒱2,j|>1}Cn,γ(j)⋅Xγ​exp⁡((β1​j∗)⊤​X)​pG2n​(Y|X)\displaystyle-\frac{1}{2}\sum\_{j\in[k^{\*}\_{2}]}\sum\_{|\gamma|=0}^{1+1\_{\{|\mathcal{V}\_{2,j}|>1\}}}C^{(j)}\_{n,\gamma}\cdot X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +12​[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​[Rn,1​(Y|X)+Rn,2​(Y|X)]\displaystyle+\frac{1}{2}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}[R\_{n,1}(Y|X)+R\_{n,2}(Y|X)] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +12​[Rn,3​(Y|X)+Rn,4​(Y|X)−Rn,5​(Y|X)−Rn,6​(Y|X)],\displaystyle+\frac{1}{2}[R\_{n,3}(Y|X)+R\_{n,4}(Y|X)-R\_{n,5}(Y|X)-R\_{n,6}(Y|X)], |  | (24) |

with a convention that r1,j=1r\_{1,j}=1 for j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1 and r2,j=1r\_{2,j}=1 for j∈[k2∗]:|𝒱2,j|j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|, where we define

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,0d,0(j)\displaystyle A^{(j)}\_{n,0\_{d},0} | :=∑i∈𝒱1,jωin−ωj∗,j∈[k1∗]\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*},\qquad j\in[k^{\*}\_{1}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,0d,0(j)\displaystyle B^{(j)}\_{n,0\_{d},0} | :=∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗),j∈[k2∗]\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*}),\qquad j\in[k^{\*}\_{2}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,0d(j)\displaystyle C^{(j)}\_{n,0\_{d}} | :=∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗),j∈[k2∗].\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*}),\qquad j\in[k^{\*}\_{2}]. |  |

Stage 2 - Non-vanishing coefficients: In this stage, we demonstrate that at least one among the terms An,α1,ℓ(j)/𝒟2​nA^{(j)}\_{n,\alpha\_{1},\ell}/\mathcal{D}\_{2n}, Bn,ℓ1,ℓ2(j)/𝒟2​nB^{(j)}\_{n,\ell\_{1},\ell\_{2}}/\mathcal{D}\_{2n}, and Cn,γ(j)/𝒟2​nC^{(j)}\_{n,\gamma}/\mathcal{D}\_{2n} does not converge to zero as n→∞n\to\infty. In particular, we assume that all these terms go to zero. Then, by looking at the terms An,α1,ℓ(j)A^{(j)}\_{n,\alpha\_{1},\ell},

* •

  For j∈[k1∗]j\in[k^{\*}\_{1}] and |α1|=ℓ=0|\alpha\_{1}|=\ell=0, we have 1𝒟1​n⋅∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|→0\frac{1}{\mathcal{D}\_{1n}}\cdot\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}\to 0;
* •

  For j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1, α1∈ℕd:|α1|=1\alpha\_{1}\in\mathbb{N}^{d}:|\alpha\_{1}|=1 and ℓ=0\ell=0, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​‖Δ​κ1​i​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\|\Delta\kappa\_{1ij}^{n}\|\to 0; |  |
* •

  For j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1, α1=0d\alpha\_{1}=0\_{d} and ℓ=1\ell=1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​|Δ​κ0​i​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}|\Delta\kappa\_{0ij}^{n}|\to 0; |  |
* •

  For j∈[k1∗]:|𝒱1,j|=1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1, α1∈ℕd:|α1|=1\alpha\_{1}\in\mathbb{N}^{d}:|\alpha\_{1}|=1 and ℓ=2\ell=2 we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​|Δ​τi​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}|\Delta\tau\_{ij}^{n}|\to 0; |  |
* •

  For j∈[k1∗]:|𝒱1,j|>1j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1, α1=2​eu\alpha\_{1}=2e\_{u}, where eu∈ℕde\_{u}\in\mathbb{N}^{d} is a one-hot vector with the uu-th entry being one while other entries being zero, for u∈[d]u\in[d], we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​‖Δ​κ1​i​jn‖2→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\|\Delta\kappa\_{1ij}^{n}\|^{2}\to 0; |  |

Next, by considering the terms Bn,ℓ1,ℓ2(j)B^{(j)}\_{n,\ell\_{1},\ell\_{2}}

* •

  For j∈[k2∗]j\in[k^{\*}\_{2}] and |ℓ1|=ℓ2=0|\ell\_{1}|=\ell\_{2}=0, we have 1𝒟2​n⋅∑j=1k2∗|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β1​j∗)|→0\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j=1}^{k^{\*}\_{2}}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{1j}^{\*})\Big{|}\to 0;
* •

  For j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, ℓ1=eu\ell\_{1}=e\_{u} for u∈[d]u\in[d], and ℓ2=0\ell\_{2}=0, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​β1​i​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\beta\_{1ij}^{n}\|\to 0; |  |
* •

  For j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, ℓ1=eu\ell\_{1}=e\_{u} for u∈[d]u\in[d], and ℓ2=1\ell\_{2}=1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​‖Δ​η1​i​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})\|\Delta\eta\_{1ij}^{n}\|\to 0; |  |
* •

  For j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, ℓ1=0d\ell\_{1}=0\_{d} and ℓ2=1\ell\_{2}=1, we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​|Δ​η0​i​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})|\Delta\eta\_{0ij}^{n}|\to 0; |  |
* •

  For j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, ℓ1=0d\ell\_{1}=0\_{d}, and ℓ2=2\ell\_{2}=2 we have

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟2​n⋅∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​|Δ​νi​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})|\Delta\nu\_{ij}^{n}|\to 0; |  |

Taking the sum of the above limits, we deduce

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1𝒟2​n⋅[\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\Bigg{[} | ∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑j∈[k2∗]:|𝒱2,j|>1|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)|\displaystyle\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κ1​i​jn‖+|Δ​κ0​i​jn|+|Δ​τi​jn|)+∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​‖Δ​κ1​i​jn‖2\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{1ij}^{n}\|+|\Delta\kappa\_{0ij}^{n}|+|\Delta\tau\_{ij}^{n}|)+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\|\Delta\kappa\_{1ij}^{n}\|^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,jexp(β0​in)(∥Δβ1​i​jn∥+∥Δη1​i​jn∥+|Δη0​i​jn|+|Δνi​jn|)]→0,\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|+\|\Delta\eta\_{1ij}^{n}\|+|\Delta\eta\_{0ij}^{n}|+|\Delta\nu\_{ij}^{n}|)\Bigg{]}\to 0, |  |

as n→∞n\to\infty. From the formulation of the Voronoi loss 𝒟2​n\mathcal{D}\_{2n} in equation ([D.2](#A4.Ex159 "D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")), it follows that

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 1𝒟2​n[∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin(|Δκ0​i​jn|r1,j+|Δτi​jn|r1,j/2)\displaystyle\frac{1}{\mathcal{D}\_{2n}}\Bigg{[}\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(|\Delta\kappa\_{0ij}^{n}|^{r\_{1,j}}+|\Delta\tau\_{ij}^{n}|^{r\_{1,j}/2}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp(β0​in)(∥Δβ1​i​jn∥r2,j+∥Δη1​i​jn∥r2,j/2+|Δη0​i​jn|r2,j+|Δνi​jn|r2,j/2)]↛0,\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|^{r\_{2,j}}+\|\Delta\eta\_{1ij}^{n}\|^{r\_{2,j}/2}+|\Delta\eta\_{0ij}^{n}|^{r\_{2,j}}+|\Delta\nu\_{ij}^{n}|^{r\_{2,j}/2})\Bigg{]}\not\to 0, |  | (25) |

as n→∞n\to\infty. Then, we consider two following cases:

Case I:
1𝒟2​n​∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​(|Δ​κ0​i​jn|r1,j+|Δ​τi​jn|r1,j/2)↛0\frac{1}{\mathcal{D}\_{2n}}\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(|\Delta\kappa\_{0ij}^{n}|^{r\_{1,j}}+|\Delta\tau\_{ij}^{n}|^{r\_{1,j}/2})\not\to 0 as n→∞n\to\infty.

In this case, there exists some index j′∈[k1∗]:|𝒱1,j′|>1j^{\prime}\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j^{\prime}}|>1 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1𝒟2​n⋅∑i∈𝒱1,j′ωin​(|Δ​κ0​i​j′n|r1,j′+|Δ​τi​j′n|r1,j′/2)↛0,\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j^{\prime}}}\omega\_{i}^{n}(|\Delta\kappa^{n}\_{0ij^{\prime}}|^{r\_{1,j^{\prime}}}+|\Delta\tau^{n}\_{ij^{\prime}}|^{r\_{1,j^{\prime}}/2})\not\to 0, |  | (26) |

as n→∞n\to\infty. WLOG, we may assume that j′=1j^{\prime}=1. Recall that the term An,α1,ℓ(j)/𝒟2​n→0A^{(j)}\_{n,\alpha\_{1},\ell}/\mathcal{D}\_{2n}\to 0 as n→∞n\to\infty for all 0≤|α1|≤r1,j0\leq|\alpha\_{1}|\leq r\_{1,j} and 0≤ℓ≤2​(r1,j−|α1|)0\leq\ell\leq 2(r\_{1,j}-|\alpha\_{1}|). Then, by dividing the ratio An,0d,ℓ(1)A^{(1)}\_{n,0\_{d},\ell} by the left hand side of equation ([26](#A4.E26 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")), we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i∈𝒱1,1∑α2+2​α3=ℓωin2α3​α2!​α3!​(Δ​κ1​i​1)α2​(Δ​τi​1)α3∑i∈𝒱1,1ωin​(|Δ​κ0​i​1n|r1,1+|Δ​τi​1n|r1,1/2)→0,\displaystyle\dfrac{\sum\_{i\in\mathcal{V}\_{1,1}}\sum\_{\alpha\_{2}+2\alpha\_{3}=\ell}\frac{\omega\_{i}^{n}}{2^{\alpha\_{3}}\alpha\_{2}!\alpha\_{3}!}(\Delta\kappa\_{1i1})^{\alpha\_{2}}(\Delta\tau\_{i1})^{\alpha\_{3}}}{\sum\_{i\in\mathcal{V}\_{1,1}}\omega\_{i}^{n}(|\Delta\kappa^{n}\_{0i1}|^{r\_{1,1}}+|\Delta\tau^{n}\_{i1}|^{r\_{1,1}/2})}\to 0, |  | (27) |

as n→∞n\to\infty for all 0≤ℓ≤2​r1,10\leq\ell\leq 2r\_{1,1}.

Let us denote Mn,1:=max⁡{|Δ​κ0​i​1n|,|Δ​τi​1n|:i∈𝒱1,1}M\_{n,1}:=\max\{|\Delta\kappa^{n}\_{0i1}|,|\Delta\tau^{n}\_{i1}|:i\in\mathcal{V}\_{1,1}\} and Wn,1:=max⁡{ωin:i∈𝒱1,1}W\_{n,1}:=\max\{\omega\_{i}^{n}:i\in\mathcal{V}\_{1,1}\}. Since the sequence (ωin/Wn,1)n(\omega\_{i}^{n}/W\_{n,1})\_{n} is bounded below, we can replace it by its subsequence that admits the limit s1​i2:=limn→∞ωin/Wn,1>0s\_{1i}^{2}:=\lim\_{n\to\infty}\omega\_{i}^{n}/W\_{n,1}>0. It should be noted that at least one among the terms s1​i2s^{2}\_{1i}, for i∈𝒱1,1i\in\mathcal{V}\_{1,1}, is equal to 1. Next, we denote (Δ​κ0​i​1n)/Mn,1→s2​i(\Delta\kappa^{n}\_{0i1})/M\_{n,1}\to s\_{2i} and (Δ​τi​1n)/[2​Mn,12]→s3​i(\Delta\tau^{n}\_{i1})/[2M\_{n,1}^{2}]\to s\_{3i} for all i∈𝒱1,1i\in\mathcal{V}\_{1,1}. Similarly, at least one of each of the s2​is\_{2i} and s3​is\_{3i} is equal to 1 or −1-1. Then, by dividing both the numerators and the denominators of the left hand side of equation ([27](#A4.E27 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")) by Wn,1​Mn,1ℓW\_{n,1}M\_{n,1}^{\ell}, we obtain the following system of polynomial equations:

|  |  |  |
| --- | --- | --- |
|  | ∑i∈𝒱1,1∑α2+2​α3=ℓs1​i2​s2​iα2​s3​iα3α2!​α3!=0,1≤ℓ≤r1,1.\displaystyle\sum\_{i\in\mathcal{V}\_{1,1}}\sum\_{\alpha\_{2}+2\alpha\_{3}=\ell}\frac{s^{2}\_{1i}~s\_{2i}^{\alpha\_{2}}~s\_{3i}^{\alpha\_{3}}}{\alpha\_{2}!\alpha\_{3}!}=0,\qquad 1\leq\ell\leq r\_{1,1}. |  |

According to the definition of the term r1,1r\_{1,1}, the above system does not admit any non-trivial solutions, which is a contradiction. Thus, Case I cannot occur.

Case II: 1𝒟2​n​∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖r2,j+‖Δ​η1​i​jn‖r2,j/2+|Δ​η0​i​jn|r2,j+|Δ​νi​jn|r2,j/2)↛0\frac{1}{\mathcal{D}\_{2n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|^{r\_{2,j}}+\|\Delta\eta\_{1ij}^{n}\|^{r\_{2,j}/2}+|\Delta\eta\_{0ij}^{n}|^{r\_{2,j}}+|\Delta\nu\_{ij}^{n}|^{r\_{2,j}/2})\not\to 0 as n→∞n\to\infty.

In this case, we can find some index j′∈[k2∗]:|𝒱2,j′|>1j^{\prime}\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j^{\prime}}|>1 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1𝒟2​n⋅∑i∈𝒱2,j′exp⁡(β0​in)​(‖Δ​β1​i​j′n‖r2,j′+‖Δ​η1​i​j′n‖r2,j′/2+|Δ​η0​i​j′n|r2,j′+|Δ​νi​j′n|r2,j′/2)↛0,\displaystyle\frac{1}{\mathcal{D}\_{2n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j^{\prime}}}\exp(\beta\_{0i}^{n})(\|\Delta\beta^{n}\_{1ij^{\prime}}\|^{r\_{2,j^{\prime}}}+\|\Delta\eta^{n}\_{1ij^{\prime}}\|^{r\_{2,j^{\prime}}/2}+|\Delta\eta^{n}\_{0ij^{\prime}}|^{r\_{2,j^{\prime}}}+|\Delta\nu^{n}\_{ij^{\prime}}|^{r\_{2,j^{\prime}}/2})\not\to 0, |  | (28) |

as n→∞n\to\infty. WLOG, we may assume that j′=1j^{\prime}=1. Recall that the term Bn,ℓ1,ℓ2(j)/𝒟2​n→0B^{(j)}\_{n,\ell\_{1},\ell\_{2}}/\mathcal{D}\_{2n}\to 0 as n→∞n\to\infty for all j∈[k2∗]j\in[k^{\*}\_{2}] and (ℓ1,ℓ2)∈ℕd×ℕ:0≤|ℓ1|+ℓ2≤2​r2,j(\ell\_{1},\ell\_{2})\in\mathbb{N}^{d}\times\mathbb{N}:0\leq|\ell\_{1}|+\ell\_{2}\leq 2r\_{2,j}. Then, by dividing the ratio Bn,ℓ1,ℓ2(1)B^{(1)}\_{n,\ell\_{1},\ell\_{2}} by the left hand side of equation ([28](#A4.E28 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")), we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑i∈𝒱2,1∑α∈ℐℓ1,ℓ2exp⁡(β0​in)2α4​α!​(Δ​β1​i​1n)α1​(Δ​η1​i​1n)α2​(Δ​η0​i​1n)α3​(Δ​νi​1n)α4∑i∈𝒱2,1exp⁡(β0​in)​(‖Δ​β1​i​1n‖r2,1+‖Δ​η1​i​1n‖r2,1/2+|Δ​η0​i​1n|r2,1+|Δ​νi​1n|r2,1/2)→0,\displaystyle\dfrac{\sum\_{i\in\mathcal{V}\_{2,1}}\sum\_{\alpha\in\mathcal{I}\_{\ell\_{1},\ell\_{2}}}\frac{\exp(\beta\_{0i}^{n})}{2^{\alpha\_{4}}\alpha!}(\Delta\beta^{n}\_{1i1})^{\alpha\_{1}}(\Delta\eta^{n}\_{1i1})^{\alpha\_{2}}(\Delta\eta^{n}\_{0i1})^{\alpha\_{3}}(\Delta\nu^{n}\_{i1})^{\alpha\_{4}}}{\sum\_{i\in\mathcal{V}\_{2,1}}\exp(\beta\_{0i}^{n})(\|\Delta\beta^{n}\_{1i1}\|^{r\_{2,1}}+\|\Delta\eta^{n}\_{1i1}\|^{r\_{2,1}/2}+|\Delta\eta^{n}\_{0i1}|^{r\_{2,1}}+|\Delta\nu^{n}\_{i1}|^{r\_{2,1}/2})}\to 0, |  | (29) |

as n→∞n\to\infty for all (ℓ1,ℓ2)∈ℕd×ℕ:0≤|ℓ1|+ℓ2≤2​r2,1(\ell\_{1},\ell\_{2})\in\mathbb{N}^{d}\times\mathbb{N}:0\leq|\ell\_{1}|+\ell\_{2}\leq 2r\_{2,1}.

Let us denote Mn,2:=max⁡{‖Δ​β1​i​1n‖,‖Δ​η1​i​1n‖,|Δ​η0​i​1n|,|Δ​νi​1n|:i∈𝒱2,1}M\_{n,2}:=\max\{\|\Delta\beta^{n}\_{1i1}\|,\|\Delta\eta^{n}\_{1i1}\|,|\Delta\eta^{n}\_{0i1}|,|\Delta\nu^{n}\_{i1}|:i\in\mathcal{V}\_{2,1}\} and Wn,2:=max⁡{exp⁡(β0​in):i∈𝒱2,1}W\_{n,2}:=\max\{\exp(\beta\_{0i}^{n}):i\in\mathcal{V}\_{2,1}\}. Since the sequence (exp⁡(β0​in)/Wn,2)n(\exp(\beta\_{0i}^{n})/W\_{n,2})\_{n} is bounded below, we can replace it by its subsequence that admits the limit t5​i2:=limn→∞exp⁡(β0​in)/Wn,2>0t\_{5i}^{2}:=\lim\_{n\to\infty}\exp(\beta\_{0i}^{n})/W\_{n,2}>0. It should be noted that at least one among the terms t5​i2t^{2}\_{5i}, for i∈𝒱2,1i\in\mathcal{V}\_{2,1}, is equal to 1. Next, we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | (Δ​β1​i​1n)/Mn,2→t1​i\displaystyle(\Delta\beta^{n}\_{1i1})/M\_{n,2}\to t\_{1i} | ,(Δη1​i​1n)/Mn,22→t2​i,\displaystyle,\qquad(\Delta\eta^{n}\_{1i1})/M\_{n,2}^{2}\to t\_{2i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (Δ​η0​i​1n)/Mn,2→t3​i\displaystyle(\Delta\eta^{n}\_{0i1})/M\_{n,2}\to t\_{3i} | ,(Δνi​1n)/[2Mn,22]→t4​i,\displaystyle,\qquad(\Delta\nu^{n}\_{i1})/[2M\_{n,2}^{2}]\to t\_{4i}, |  |

for all i∈𝒱2,1i\in\mathcal{V}\_{2,1}. Similarly, at least one of each of the t1​it\_{1i}, t2​it\_{2i}, t3​it\_{3i}, and t4​it\_{4i}, is equal to 1 or −1-1. Then, by dividing both the numerators and the denominators of the left hand side of equation ([29](#A4.E29 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")) by Wn,2​Mn,2|ℓ1|+ℓ2W\_{n,2}M\_{n,2}^{|\ell\_{1}|+\ell\_{2}}, we obtain the following system of polynomial equations:

|  |  |  |
| --- | --- | --- |
|  | ∑i∈𝒱2,1∑α∈ℐℓ1,ℓ21α!⋅t5​i2​t1​iα1​t2​iα2​t3​iα3​t4​iα4=0,1≤|ℓ1|+ℓ2≤r2,1.\displaystyle\sum\_{i\in\mathcal{V}\_{2,1}}\sum\_{\alpha\in\mathcal{I}\_{\ell\_{1},\ell\_{2}}}\frac{1}{\alpha!}\cdot t^{2}\_{5i}~t\_{1i}^{\alpha\_{1}}~t\_{2i}^{\alpha\_{2}}~t\_{3i}^{\alpha\_{3}}~t\_{4i}^{\alpha\_{4}}=0,\qquad 1\leq|\ell\_{1}|+\ell\_{2}\leq r\_{2,1}. |  |

According to the definition of the term r2,1r\_{2,1}, the above system does not admit any non-trivial solutions, which is a contradiction. Thus, Case II cannot occur.

The fact that both Case I and Case II cannot occur contradicts the result of equation ([D.2](#A4.Ex221 "D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")). Thus, not all the terms An,α1,ℓ(j)/𝒟2​nA^{(j)}\_{n,\alpha\_{1},\ell}/\mathcal{D}\_{2n}, Bn,ℓ1,ℓ2(j)/𝒟2​nB^{(j)}\_{n,\ell\_{1},\ell\_{2}}/\mathcal{D}\_{2n}, and Cn,γ(j)/𝒟2​nC^{(j)}\_{n,\gamma}/\mathcal{D}\_{2n} converge to zero as n→∞n\to\infty.

Stage 3 - Fatou’s lemma contradiction: We denote by mnm\_{n} the maximum of the absolute values of the ratios An,α1,ℓ(j)/𝒟2​nA^{(j)}\_{n,\alpha\_{1},\ell}/\mathcal{D}\_{2n}, Bn,ℓ1,ℓ2(j)/𝒟2​nB^{(j)}\_{n,\ell\_{1},\ell\_{2}}/\mathcal{D}\_{2n}, and Cn,γ(j)/𝒟2​nC^{(j)}\_{n,\gamma}/\mathcal{D}\_{2n}. It follows from the result of Stage that 1/mn↛∞1/m\_{n}\not\to\infty as n→∞n\to\infty. Then, by means of the Fatou’s lemma, we have

|  |  |  |
| --- | --- | --- |
|  | limn→∞𝔼X[V(fG1n,G2n(⋅|X),fG1∗,G2∗(⋅|X))]mn​𝒟2​n≥∫lim infn→∞|fG1n,G2n(Y|X)−fG1∗,G2∗(Y|X)|2​mn​𝒟2​n​d​(X,Y).\displaystyle\lim\_{n\to\infty}\dfrac{\mathbb{E}\_{X}[V(f\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{m\_{n}\mathcal{D}\_{2n}}\geq\int\liminf\_{n\to\infty}\dfrac{|f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)|}{2m\_{n}\mathcal{D}\_{2n}}\mathrm{d}(X,Y). |  |

Then, we deduce [fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)]/[mn​𝒟1​n]→0[f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{1n}]\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). Since the input space is bounded and the parameter space is compact, the quantity ∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*}) is bounded. Thus, we also have

|  |  |  |
| --- | --- | --- |
|  | 1mn​𝒟2​n​[∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​[fG1n,G2n​(Y|X)−fG1∗,G2∗​(Y|X)]→0,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{2n}}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}[f\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]\to 0, |  |

as n→∞n\to\infty for almost surely (X,Y)(X,Y). Let us denote

|  |  |  |
| --- | --- | --- |
|  | 1mn​𝒟2​n​An,α1,ℓ(j)→aα1,ℓ(j),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{2n}}A^{(j)}\_{n,\alpha\_{1},\ell}\to a^{(j)}\_{\alpha\_{1},\ell}, |  |
|  |  |  |
| --- | --- | --- |
|  | 1mn​𝒟2​n​Bn,ℓ1,ℓ2(j)→bℓ1,ℓ2(j),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{2n}}B^{(j)}\_{n,\ell\_{1},\ell\_{2}}\to b^{(j)}\_{\ell\_{1},\ell\_{2}}, |  |
|  |  |  |
| --- | --- | --- |
|  | 1mn​𝒟2​n​Cn,γ(j)→cγ(j),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{2n}}C^{(j)}\_{n,\gamma}\to c^{(j)}\_{\gamma}, |  |

as n→∞n\to\infty with a note that at least one among them is non-zero. From equation ([24](#A4.E24 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")), we deduce

|  |  |  |
| --- | --- | --- |
|  | [∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)]​12​∑j∈[k1∗]∑|α1|=0r1,j∑ℓ=02​(r1,j−|α1|)aα1,ℓ(j)⋅Xα1​∂|α1|+ℓπ∂h1|α1|+ℓ​(Y|(κ1​j∗)⊤​X+κ0​j∗,τj∗)\displaystyle\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{1}{2}\sum\_{j\in[k^{\*}\_{1}]}\sum\_{|\alpha\_{1}|=0}^{r\_{1,j}}\sum\_{\ell=0}^{2(r\_{1,j}-|\alpha\_{1}|)}a^{(j)}\_{\alpha\_{1},\ell}\cdot X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\ell}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\ell}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +12​∑j∈[k2∗]∑|ℓ1|+ℓ2=02​r2,jbℓ1,ℓ2(j)⋅Xℓ1​exp⁡((β1​j∗)⊤​X)​∂ℓ2π∂h2ℓ2​(Y|(η1​j∗)⊤​X+η0​j∗,νj∗)\displaystyle+\frac{1}{2}\sum\_{j\in[k^{\*}\_{2}]}\sum\_{|\ell\_{1}|+\ell\_{2}=0}^{2r\_{2,j}}b^{(j)}\_{\ell\_{1},\ell\_{2}}\cdot X^{\ell\_{1}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\ell\_{2}}\pi}{\partial h\_{2}^{\ell\_{2}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | −12∑j∈[k2∗]∑|γ|=01+1{|𝒱2,j|>1}cγ(j)⋅Xγexp((β1​j∗)⊤X)pG2∗(Y|X)]→0,\displaystyle-\frac{1}{2}\sum\_{j\in[k^{\*}\_{2}]}\sum\_{|\gamma|=0}^{1+1\_{\{|\mathcal{V}\_{2,j}|>1\}}}c^{(j)}\_{\gamma}\cdot X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{\*}\_{2}}(Y|X)\Big{]}\to 0, |  |

as n→∞n\to\infty for almost surely (X,Y)(X,Y). Since the set

|  |  |  |
| --- | --- | --- |
|  | {Xα1∂|α1|+ℓπ∂h1|α1|+ℓ(Y|(κ1​j∗)⊤X+κ0​j∗,τj∗):j∈[k1∗],0≤|α1|≤r1,j,0≤ℓ≤2(r1,j−|α1|)}\displaystyle\Bigg{\{}X^{\alpha\_{1}}\frac{\partial^{|\alpha\_{1}|+\ell}\pi}{\partial h\_{1}^{|\alpha\_{1}|+\ell}}(Y|(\kappa\_{1j}^{\*})^{\top}X+\kappa\_{0j}^{\*},\tau\_{j}^{\*}):j\in[k^{\*}\_{1}],0\leq|\alpha\_{1}|\leq r\_{1,j},0\leq\ell\leq 2(r\_{1,j}-|\alpha\_{1}|)\Bigg{\}} |  |
|  |  |  |
| --- | --- | --- |
|  | ∪{Xℓ1exp((β1​j∗)⊤X)∂ℓ2π∂h2ℓ2(Y|(η1​j∗)⊤X+η0​j∗,νj∗),Xγexp((β1​j∗)⊤X)pG2∗(Y|X):\displaystyle\cup\Bigg{\{}X^{\ell\_{1}}\exp((\beta\_{1j}^{\*})^{\top}X)\frac{\partial^{\ell\_{2}}\pi}{\partial h\_{2}^{\ell\_{2}}}(Y|(\eta\_{1j}^{\*})^{\top}X+\eta\_{0j}^{\*},\nu\_{j}^{\*}),\ X^{\gamma}\exp((\beta\_{1j}^{\*})^{\top}X)p\_{G^{\*}\_{2}}(Y|X): |  |
|  |  |  |
| --- | --- | --- |
|  | j∈[k2∗],0≤|ℓ1|+ℓ2≤2r2,j,0≤|γ|≤2}\displaystyle j\in[k^{\*}\_{2}],0\leq|\ell\_{1}|+\ell\_{2}\leq 2r\_{2,j},0\leq|\gamma|\leq 2\Bigg{\}} |  |

is linearly independent w.r.t …, we obtain aα1,ℓ(j)a^{(j)}\_{\alpha\_{1},\ell} for all j∈[k1∗]j\in[k^{\*}\_{1}], α1∈ℕd\alpha\_{1}\in\mathbb{N}^{d}, ℓ∈ℕ\ell\in\mathbb{N}, and bℓ1,ℓ2(j)=cγ(j)=0b^{(j)}\_{\ell\_{1},\ell\_{2}}=c^{(j)}\_{\gamma}=0 for all j∈[k2∗]j\in[k^{\*}\_{2}], (ℓ1,ℓ2)∈ℕd×ℕ(\ell\_{1},\ell\_{2})\in\mathbb{N}^{d}\times\mathbb{N}, γ∈ℕd\gamma\in\mathbb{N}^{d}. This result contradicts the fact that not all the terms aα1,ℓ(j)a^{(j)}\_{\alpha\_{1},\ell}, bℓ1,ℓ2(j)b^{(j)}\_{\ell\_{1},\ell\_{2}}, and cγ(j)c^{(j)}\_{\gamma} equal zero. Hence, we achieve the local part in equation ([20](#A4.E20 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")) and complete the proof.

### D.3 Proof of Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating")

By leveraging the proof framework in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), we also focus on demonstrating the local part

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟3​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(gG1,G2(⋅|X),gG1∗,G2∗(⋅|X))]𝒟3​((G1,G2),(G1∗,G2∗))>0,\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0, |  | (30) |

and the global part

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟3​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(gG1,G2(⋅|X),gG1∗,G2∗(⋅|X))]𝒟3​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{3}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  | (31) |

in this appendix. Note that since the global part ([31](#A4.E31 "In D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")) can be argued in a similar fashion to Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), its derivation is omitted here. Therefore, it is sufficient to establish the local part ([30](#A4.E30 "In D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")). Suppose that the local part does not hold. Then, there exists a sequence of mixing measure pairs (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) taking the form G1n:=∑i=1k1nωin​δ(κin,τin)G^{n}\_{1}:=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{i}^{n},\tau\_{i}^{n})}, G2n:=∑i=1k2nσ​(β0​in)​δ(β1​in,ηin,νin)G^{n}\_{2}:=\sum\_{i=1}^{k^{n}\_{2}}\sigma(\beta\_{0i}^{n})\delta\_{(\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})} for n∈ℕn\in\mathbb{N} such that 𝒟3​n:=𝒟3​((G1n,G2n),(G1∗,G2∗))→0\mathcal{D}\_{3n}:=\mathcal{D}\_{3}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\to 0 and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(gG1n,G2n(⋅|X),gG1∗,G2∗(⋅|X))]/𝒟3​n→0,\displaystyle\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{3n}\to 0, |  | (32) |

as n→∞n\to\infty. Here, we may assume WLOG that the number of shared experts and routed experts k1nk^{n}\_{1}, k2nk^{n}\_{2} and Voronoi cells 𝒱1,j=𝒱1,j​(G1n)\mathcal{V}\_{1,j}=\mathcal{V}\_{1,j}(G^{n}\_{1}), 𝒱2,j=𝒱2,j​(G2n)\mathcal{V}\_{2,j}=\mathcal{V}\_{2,j}(G^{n}\_{2}) do not change with the sample size nn. Then, the Voronoi loss 𝒟3​n\mathcal{D}\_{3n} can be rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟3​n=∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑j∈[k2∗]:|𝒱2,j|>1|∑i∈𝒱2,jσ​(β0​in)−σ​(β0​j∗)|\displaystyle\mathcal{D}\_{3n}=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i}^{n})-\sigma(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)+∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j(‖Δ​β1​i​jn‖+|Δ​β0​i​jn|+‖Δ​ηi​jn‖+|Δ​νi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|)+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}(\|\Delta\beta\_{1ij}^{n}\|+|\Delta\beta\_{0ij}^{n}|+\|\Delta\eta\_{ij}^{n}\|+|\Delta\nu\_{ij}^{n}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖2+|Δ​τi​jn|2)+∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j(‖Δ​β1​i​jn‖2+‖Δ​ηi​jn‖2+|Δ​νi​jn|2),\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2})+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}(\|\Delta\beta\_{1ij}^{n}\|^{2}+\|\Delta\eta\_{ij}^{n}\|^{2}+|\Delta\nu\_{ij}^{n}|^{2}), |  | (33) |

where we denote Δ​β0​i​jn:=β0​in−β0​j∗\Delta\beta\_{0ij}^{n}:=\beta^{n}\_{0i}-\beta\_{0j}^{\*}. Since 𝒟3​n→0\mathcal{D}\_{3n}\to 0 as n→∞n\to\infty, then the above formulation indicates that as n→∞n\to\infty, we have

* •

  For j∈[k1∗]j\in[k^{\*}\_{1}] and i∈𝒱1,ji\in\mathcal{V}\_{1,j}: ∑i∈𝒱1,jωin→ωj∗\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\to\omega\_{j}^{\*}, (κin,τin)→(κj∗,τj∗)(\kappa\_{i}^{n},\tau\_{i}^{n})\to(\kappa\_{j}^{\*},\tau\_{j}^{\*});
* •

  For j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1 and i∈𝒱2,ji\in\mathcal{V}\_{2,j}: (β1​in,β0​in,ηin,νin)→(β1​j∗,β0​j∗,ηj∗,νj∗)(\beta\_{1i}^{n},\beta\_{0i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})\to(\beta\_{1j}^{\*},\beta\_{0j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*});
* •

  For j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1 and i∈𝒱2,ji\in\mathcal{V}\_{2,j}: ∑i∈𝒱2,jσ​(β0​in)−σ​(β0​j∗)\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i}^{n})-\sigma(\beta\_{0j}^{\*}), (β1​in,ηin,νin)→(β1​j∗,ηj∗,νj∗)(\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})\to(\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}).

Now, we divide the proof into three main stages:

Stage 1 - Density Decomposition: In this stage, we aim to decompose the density discrepancy gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X). For ease of presentation, we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X) | :=∑i=1k1nωin​π​(Y|h1​(X,κin),τin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{1}}\omega^{n}\_{i}\pi(Y|h\_{1}(X,\kappa^{n}\_{i}),\tau^{n}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1∗​(Y|X)\displaystyle q\_{G^{\*}\_{1}}(Y|X) | :=∑i=1k1∗ωi∗​π​(Y|h1​(X,κi∗),τi∗),\displaystyle:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2n​(Y|X)\displaystyle p\_{G^{n}\_{2}}(Y|X) | :=∑i=1k2nσ​((β1​in)⊤​X+β0​in)∑j=1k2nσ​((β1​jn)⊤​X+β0​jn)⋅π​(Y|h2​(X,ηin),νin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{2}}\frac{\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})}{\sum\_{j=1}^{k^{n}\_{2}}\sigma((\beta\_{1j}^{n})^{\top}X+\beta\_{0j}^{n})}\cdot\pi(Y|h\_{2}(X,\eta^{n}\_{i}),\nu\_{i}^{n}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2∗​(Y|X)\displaystyle p\_{G^{\*}\_{2}}(Y|X) | :=∑i=1k2∗σ​((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)⋅π​(Y|h2​(X,ηi∗),νi∗).\displaystyle:=\sum\_{i=1}^{k^{\*}\_{2}}\frac{\sigma((\beta\_{1i}^{\*})^{\top}X+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})}\cdot\pi(Y|h\_{2}(X,\eta^{\*}\_{i}),\nu\_{i}^{\*}). |  |

Given the above notations, we get

|  |  |  |
| --- | --- | --- |
|  | gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)=12​[(qG1n​(Y|X)−qG1∗​(Y|X))+(pG2n​(Y|X)−pG2∗​(Y|X))].\displaystyle g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)=\frac{1}{2}\left[(q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X))+(p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X))\right]. |  |

Stage 1.1: Firstly, we decompose the term qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)−qG1∗​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k1∗(∑i∈𝒱1,jωin−ωj∗)​π​(Y|h1​(X,κj∗),τj∗)\displaystyle+\sum\_{j=1}^{k^{\*}\_{1}}\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=An,1​(Y|X)+An,2​(Y|X)+An,0​(Y|X).\displaystyle:=A\_{n,1}(Y|X)+A\_{n,2}(Y|X)+A\_{n,0}(Y|X). |  |

By using the same arguments as in Stage 1.1 in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), the term An,1​(Y|X)A\_{n,1}(Y|X) is rewritten as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1​(Y|X)\displaystyle A\_{n,1}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑ρ=12An,1,ρ(j)​(X)​∂ρπ∂h1ρ​(Y|h1​(X,κj∗),τj∗)+Rn,1​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{\rho=1}^{2}A^{(j)}\_{n,1,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,1}(Y|X), |  |

where Rn,1​(Y|X)R\_{n,1}(Y|X) is a Taylor remainder such that Rn,1​(Y|X)/𝒟3​n→R\_{n,1}(Y|X)/\mathcal{D}\_{3n}\to as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,1(j)​(X)\displaystyle A^{(j)}\_{n,1,1}(X) | :=∑i∈𝒱1,jωin​∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,2(j)​(X)\displaystyle A^{(j)}\_{n,1,2}(X) | :=∑i∈𝒱1,jωin​12​(Δ​τi​jn),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{2}(\Delta\tau\_{ij}^{n}), |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|=1|\mathcal{V}\_{1,j}|=1. Meanwhile, we can represent An,2​(Y|X)A\_{n,2}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2​(Y|X)\displaystyle A\_{n,2}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|>1∑ρ=14An,1,ρ(j)​(X)​∂ρπ∂h1ρ​(Y|h1​(X,κj∗),τj∗)+Rn,2​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{\rho=1}^{4}A^{(j)}\_{n,1,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,2}(Y|X), |  |

where Rn,2​(Y|X)R\_{n,2}(Y|X) is a Taylor remainder such that Rn,2​(Y|X)/𝒟3​n→R\_{n,2}(Y|X)/\mathcal{D}\_{3n}\to as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,1(j)​(X)\displaystyle A^{(j)}\_{n,2,1}(X) | :=∑i∈𝒱1,jωin​(∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,2(j)​(X)\displaystyle A^{(j)}\_{n,2,2}(X) | :=∑i∈𝒱1,jωin​(12​(Δ​τi​jn)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\frac{1}{2}(\Delta\tau\_{ij}^{n})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,3(j)​(X)\displaystyle A^{(j)}\_{n,2,3}(X) | :=∑i∈𝒱1,jωin​∑u1=1d112​(Δ​κi​jn)(u1)​(Δ​τi​jn)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}\frac{1}{2}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,4(j)​(X)\displaystyle A^{(j)}\_{n,2,4}(X) | :=∑i∈𝒱1,jωin​18​(Δ​τi​jn)2,\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{8}(\Delta\tau\_{ij}^{n})^{2}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|>1|\mathcal{V}\_{1,j}|>1.

Stage 1.2: Next, we attempt to decompose the term Qn​(Y|X):=[∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]⋅[pG2n​(Y|X)−pG2∗​(Y|X)]Q\_{n}(Y|X):=\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot[p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X)] as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Qn​(Y|X)\displaystyle Q\_{n}(Y|X) | =∑j=1k2∗[∑i∈𝒱2,jσ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​((β1​j∗)⊤​X+β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)]\displaystyle=\sum\_{j=1}^{k^{\*}\_{2}}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −∑j=1k2∗[∑i∈𝒱2,jσ​((β1​in)⊤​X+β0​in)−σ​((β1​j∗)⊤​X+β0​j∗)]​pG2n​(Y|X)\displaystyle-\sum\_{j=1}^{k^{\*}\_{2}}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})-\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn​(Y|X)−Cn​(Y|X).\displaystyle:=B\_{n}(Y|X)-C\_{n}(Y|X). |  |

Stage 1.2.1: In this step, we decompose the term Bn​(Y|X)B\_{n}(Y|X) with a note that β1​j∗=0d\beta\_{1j}^{\*}=0\_{d} for all j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn​(Y|X)\displaystyle B\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1[∑i∈𝒱2,jσ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​((β1​j∗)⊤​X+β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1[∑i∈𝒱2,jσ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​(β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma(\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1[∑i∈𝒱2,jσ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​((β1​j∗)⊤​X+β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j[σ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​(β0​in)​π​(Y|h2​(X,ηj∗),νj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma(\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1[∑i∈𝒱2,jσ​(β0​in)−σ​(β0​j∗)]​π​(Y|h2​(X,ηj∗),νj∗)\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i}^{n})-\sigma(\beta\_{0j}^{\*})\Big{]}\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn,1​(Y|X)+Bn,2​(Y|X)+Bn,0​(Y|X).\displaystyle:=B\_{n,1}(Y|X)+B\_{n,2}(Y|X)+B\_{n,0}(Y|X). |  |

Denote ψ​(X;β1,β0):=σ​(β1⊤​X+β0)\psi(X;\beta\_{1},\beta\_{0}):=\sigma(\beta\_{1}^{\top}X+\beta\_{0}). By applying the first-order Taylor expansion to the function ψ​(X,β1​in,β0​in)​π​(Y|h2​(X,ηin),νin)\psi(X,\beta\_{1i}^{n},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n}) around the point (β1​j∗,β0​j∗,ηin,νin)(\beta\_{1j}^{\*},\beta\_{0j}^{\*},\eta\_{i}^{n},\nu\_{i}^{n}), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1​(Y|X)\displaystyle B\_{n,1}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j∑|α|=11α!​(Δ​β1​i​jn)α1​(Δ​β0​i​jn)α2​(Δ​ηi​jn)α3​(Δ​νi​jn)α4\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{|\alpha|=1}\frac{1}{\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\beta\_{0ij}^{n})^{\alpha\_{2}}(\Delta\eta\_{ij}^{n})^{\alpha\_{3}}(\Delta\nu\_{ij}^{n})^{\alpha\_{4}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×∂|α1|+α2ψ∂β1α1​∂β0α2​(X;β1​j∗,β0​j∗)​∂|α3|+α4π∂ηα3​∂να4​(Y|h2​(X,ηj∗),νj∗)+Rn,3​(Y|X)\displaystyle\hskip 56.9055pt\times\frac{\partial^{|\alpha\_{1}|+\alpha\_{2}}\psi}{\partial\beta\_{1}^{\alpha\_{1}}\partial\beta\_{0}^{\alpha\_{2}}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial^{|\alpha\_{3}|+\alpha\_{4}}\pi}{\partial\eta^{\alpha\_{3}}\partial\nu^{\alpha\_{4}}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+R\_{n,3}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1∑ρ=02Bn,1,ρ(j)​(X)⋅∂ρπ∂h2ρ​(Y|h2​(X,ηj∗),νj∗)+Rn,3​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{\rho=0}^{2}B^{(j)}\_{n,1,\rho}(X)\cdot\frac{\partial^{\rho}\pi}{\partial h\_{2}^{\rho}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+R\_{n,3}(Y|X), |  |

where Rn,3​(Y|X)R\_{n,3}(Y|X) is a Taylor remainder such that Rn,3​(Y|X)/𝒟3​n→0R\_{n,3}(Y|X)/\mathcal{D}\_{3n}\to 0 as n→∞n\to\infty and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,0(j)​(X)\displaystyle B^{(j)}\_{n,1,0}(X) | :=∑i∈𝒱2,j[∑u=1d(Δ​β1​i​jn)(u)​∂ψ∂β1(u)​(X;β1​j∗,β0​j∗)+(Δ​β0​i​jn)​∂ψ∂β0​(X;β1​j∗,β0​j∗)],\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})+(\Delta\beta\_{0ij}^{n})\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,1(j)​(X)\displaystyle B^{(j)}\_{n,1,1}(X) | :=∑i∈𝒱2,j∑u2=1d2(Δ​ηi​jn)(u2)​∂h2∂η(u2)​(X,ηj∗)​ψ​(X;β1​j∗,β0​j∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1,2(j)​(X)\displaystyle B^{(j)}\_{n,1,2}(X) | :=∑i∈𝒱2,j12​(Δ​νi​jn)​ψ​(X;β1​j∗,β0​j∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\frac{1}{2}(\Delta\nu\_{ij}^{n})\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*}), |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|=1|\mathcal{V}\_{2,j}|=1. Next, by means of the second-order Taylor expansion to the function ψ​(X;β1​j∗,β0​in)​π​(Y|h2​(X,ηj∗),νj∗)\psi(X;\beta\_{1j}^{\*},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) around the point (β1​j∗,ηj∗,νj∗)(\beta\_{1j}^{\*},\eta\_{j}^{\*},\nu\_{j}^{\*}) with a note that β1​j∗=0d\beta\_{1j}^{\*}=0\_{d} for all j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we decompose the term Bn,2​(Y|X)B\_{n,2}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2​(Y|X)\displaystyle B\_{n,2}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j∑|α|=121α!​(Δ​β1​i​jn)α1​(Δ​ηi​jn)α2​(Δ​νi​jn)α3\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\sum\_{|\alpha|=1}^{2}\frac{1}{\alpha!}(\Delta\beta\_{1ij}^{n})^{\alpha\_{1}}(\Delta\eta\_{ij}^{n})^{\alpha\_{2}}(\Delta\nu\_{ij}^{n})^{\alpha\_{3}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×∂|α1|ψ∂β1α1​(X;0d,β0​in)​∂|α2|+α3π∂ηα2​∂να3​(Y|h2​(X,ηj∗),νj∗)+Rn,4​(Y|X)\displaystyle\hskip 56.9055pt\times\frac{\partial^{|\alpha\_{1}|}\psi}{\partial\beta\_{1}^{\alpha\_{1}}}(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{|\alpha\_{2}|+\alpha\_{3}}\pi}{\partial\eta^{\alpha\_{2}}\partial\nu^{\alpha\_{3}}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+R\_{n,4}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|>1∑ρ=04Bn,2,ρ(j)​(X)⋅∂ρπ∂h2ρ​(Y|h2​(X,ηj∗),νj∗)+Rn,4​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{\rho=0}^{4}B^{(j)}\_{n,2,\rho}(X)\cdot\frac{\partial^{\rho}\pi}{\partial h\_{2}^{\rho}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+R\_{n,4}(Y|X), |  |

where Rn,4​(Y|X)R\_{n,4}(Y|X) is a Taylor remainder such that Rn,4​(Y|X)/𝒟3​n→0R\_{n,4}(Y|X)/\mathcal{D}\_{3n}\to 0 as n→∞n\to\infty and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,0(j)​(X)\displaystyle B^{(j)}\_{n,2,0}(X) | :=∑i∈𝒱2,j[∑u=1d(Δ​β1​i​jn)(u)​∂ψ∂β1(u)​(X;0d,β0​in)+∑u,v=1d(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)1+1{u=v}​∂2ψ∂β1(u)​∂β1(v)​(X;0d,β0​in)],\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})+\sum\_{u,v=1}^{d}\frac{(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}}{1+1\_{\{u=v\}}}\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\beta\_{0i}^{n})\Big{]}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,1(j)​(X)\displaystyle B^{(j)}\_{n,2,1}(X) | :=∑i∈𝒱2,j[∑u2=1d2(Δηi​jn)(u2)∂h2∂η(u2)(X.ηj∗)ψ(X;0d,β0​in)+∑u2,v2=1d2(Δ​ηi​jn)(u2)​(Δ​ηi​jn)(v2)1+1{u2=v2}∂2h2∂η(u2)​∂η(v2)(X,ηj∗)\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×ψ(X;0d,β0​in)+∑u=1d∑u2=1d2(Δβ1​i​jn)(u)(Δηi​jn)(u2)∂h2∂η(u2)(X.ηj∗)∂ψ∂β1(u)(X;0d,β0​in)],\displaystyle\times\psi(X;0\_{d},\beta\_{0i}^{n})+\sum\_{u=1}^{d}\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})\Big{]}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,2(j)​(X)\displaystyle B^{(j)}\_{n,2,2}(X) | :=∑i∈𝒱2,j[12(Δνi​jn)ψ(X;0d,β0​in)+∑u=1d(Δβ1​i​jn)(u)12(Δνi​jn)∂ψ∂β1(u)(X;0d,β0​in)\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\frac{1}{2}(\Delta\nu\_{ij}^{n})\psi(X;0\_{d},\beta\_{0i}^{n})+\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}\frac{1}{2}(\Delta\nu\_{ij}^{n})\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑u2,v2=1d2(Δ​ηi​jn)(u2)​(Δ​ηi​jn)(v2)1+1{u2=v2}∂2h2∂η(u2)​∂η(v2)(X,ηj∗)ψ(X;0d,β0​in)],\displaystyle\hskip 56.9055pt+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\Big{]}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,3(j)​(X)\displaystyle B^{(j)}\_{n,2,3}(X) | :=∑i∈𝒱2,j[∑u2=1d2(Δ​ηi​jn)(u2)​12​(Δ​νi​jn)​∂h2∂η(u2)​(X,ηj∗)​ψ​(X;0d,β0​in)],\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\frac{1}{2}(\Delta\nu\_{ij}^{n})\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\Big{]}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2,4(j)​(X)\displaystyle B^{(j)}\_{n,2,4}(X) | :=∑i∈𝒱2,j18​(Δ​νi​jn)2​ψ​(X;0d,β0​in),\displaystyle:=\sum\_{i\in\mathcal{V}\_{2,j}}\frac{1}{8}(\Delta\nu\_{ij}^{n})^{2}\psi(X;0\_{d},\beta\_{0i}^{n}), |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|>1|\mathcal{V}\_{2,j}|>1.

Stage 1.2.2: In this step, we decompose the term Cn​(Y|X)C\_{n}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn​(Y|X)\displaystyle C\_{n}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1[∑i∈𝒱2,jψ​(X;β1​in,β0​in)−ψ​(X;β1​j∗,β0​j∗)]​pG2n​(Y|X)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})-\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1[∑i∈𝒱2,jψ​(X;β1​in,β0​in)−ψ​(X;β1​j∗,β0​j∗)]​pG2n​(Y|X)\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})-\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑j∈[k2∗]:|𝒱2,j|=1[∑i∈𝒱2,jψ​(X;β1​in,β0​in)−ψ​(X;β1​j∗,β0​j∗)]​pG2n​(Y|X)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})-\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j[ψ​(X;β1​in,β0​in)−ψ​(X;0d,β0​in)]​pG2n​(Y|X)\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})-\psi(X;0\_{d},\beta\_{0i}^{n})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k2∗]:|𝒱2,j|>1[∑i∈𝒱2,jψ​(X;0d,β0​in)−ψ​(X;0d,β0​j∗)]​pG2n​(Y|X)\displaystyle+\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{[}\sum\_{i\in\mathcal{V}\_{2,j}}\psi(X;0\_{d},\beta\_{0i}^{n})-\psi(X;0\_{d},\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Cn,1​(Y|X)+Cn,2​(Y|X)+Cn,0​(Y|X).\displaystyle:=C\_{n,1}(Y|X)+C\_{n,2}(Y|X)+C\_{n,0}(Y|X). |  |

By applying the first-order Taylor expansion to the function ψ​(X;β1​in,β0​in)\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n}) around the point (β1​j∗,β0​j∗)(\beta\_{1j}^{\*},\beta\_{0j}^{\*}), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,1​(Y|X)\displaystyle C\_{n,1}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j[∑u=1d(Δ​β1​i​jn)(u)​∂ψβ1(u)​(X;β1​j∗,β0​j∗)+(Δ​β0​i​jn)​∂ψ∂β0​(X;β1​j∗,β0​j∗)]​pG2n​(Y|X)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}\frac{\partial\psi}{\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})+(\Delta\beta\_{0ij}^{n})\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +Rn,5​(Y|X),\displaystyle\hskip 341.43306pt+R\_{n,5}(Y|X), |  |

where Rn,5​(Y|X)R\_{n,5}(Y|X) is a Taylor remainder such that Rn,5​(Y|X)/𝒟3​n→0R\_{n,5}(Y|X)/\mathcal{D}\_{3n}\to 0 as n→∞n\to\infty. Next, by means of the second-order Taylor expansion to the function ψ​(X;β1​in,β0​in)\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n}) around the point β1​j∗=0d\beta^{\*}\_{1j}=0\_{d} for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn,2​(Y|X)\displaystyle C\_{n,2}(Y|X) | =∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j[∑u=1d(Δβ1​i​jn)(u)∂ψ∂β1(u)(X;0d,β0​in)\displaystyle=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u=1}^{d}(\Delta\beta\_{1ij}^{n})^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑u,v=1d(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)1+1{u=v}∂2ψ∂β1(u)​∂β1(v)(X;0d,β0​in)]pG2n(Y|X)+Rn,6(Y|X),\displaystyle\hskip 28.45274pt+\sum\_{u,v=1}^{d}\frac{(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}}{1+1\_{\{u=v\}}}\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\beta\_{0i}^{n})\Big{]}p\_{G^{n}\_{2}}(Y|X)+R\_{n,6}(Y|X), |  |

where Rn,6​(Y|X)R\_{n,6}(Y|X) is a Taylor remainder such that Rn,6​(Y|X)/𝒟3​n→0R\_{n,6}(Y|X)/\mathcal{D}\_{3n}\to 0 as n→∞n\to\infty.

Combining the above decompositions, we can view An,0​(Y|X)/𝒟3​nA\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟3​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{3n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟3​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{3n}, Bn,0​(Y|X)/𝒟3​nB\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟3​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{3n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟3​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{3n}, Cn,0​(Y|X)/𝒟3​nC\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟3​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{3n} and [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟3​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{3n} as a combination of elements from the following sets

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮0,j\displaystyle\mathcal{S}\_{0,j} | :={π​(Y|h1​(X,κj∗),τj∗)},\displaystyle:=\{\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮1,j\displaystyle\mathcal{S}\_{1,j} | :={∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗),∂2h1∂κ(u1)​∂κ(v1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮2,j\displaystyle\mathcal{S}\_{2,j} | :={∂2π∂h12(Y|h1(X,κj∗),τj∗),∂h1∂κ(u1)(X,κj∗)∂h1∂κ(v1)(X,κj∗)∂2π∂h12(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮3,j\displaystyle\mathcal{S}\_{3,j} | :={∂h1∂κ(u1)(X,κj∗)∂3π∂h13(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮4,j\displaystyle\mathcal{S}\_{4,j} | :={∂4π∂h14(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}], and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯0,j\displaystyle\mathcal{T}\_{0,j} | :={π(Y|h2(X,ηj∗),νj∗),∂ψ∂β1(u)(X;0d,β0​in)π(Y|h2(X,ηj∗),νj∗),\displaystyle:=\Bigg{\{}\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂2ψ∂β1(u)​∂β1(v)(X;0d,β0​in)π(Y|h2(X,ηj∗),νj∗):u,v∈[d]},\displaystyle\hskip 113.81102pt\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}):u,v\in[d]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯1,j\displaystyle\mathcal{T}\_{1,j} | :={∂h2∂η(u2)(X.ηj∗)ψ(X;0d,β0​in)∂π∂h2(Y|h2(X,ηj∗),νj∗),\displaystyle:=\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂h2∂η(u2)(X.ηj∗)∂ψ∂β1(u)(X;0d,β0​in)∂π∂h2(Y|h2(X,ηj∗),νj∗):u∈[d],u2∈[d2]},\displaystyle\hskip 113.81102pt\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}):u\in[d],\ u\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯2,j\displaystyle\mathcal{T}\_{2,j} | :={ψ(X;0d,β0​in)∂2π∂h22(Y|h2(X,ηj∗),νj∗),∂ψ∂β1(u)(X;0d,β0​in)∂2π∂h22(Y|h2(X,ηj∗),νj∗),\displaystyle:=\Bigg{\{}\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂2h2∂η(u2)​∂η(v2)(X,ηj∗)ψ(X;0d,β0​in)∂2π∂h22(Y|h2(X,ηj∗),νj∗):u∈[d],u2,v2∈[d2]},\displaystyle\hskip 85.35826pt\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}):u\in[d],\ u\_{2},v\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯3,j\displaystyle\mathcal{T}\_{3,j} | :={∂h2∂η(u2)(X,ηj∗)ψ(X;0d,β0​in)∂3π∂h23(Y|h2(X,ηj∗),νj∗):u2∈[d2]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{3}\pi}{\partial h\_{2}^{3}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}):u\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯4,j\displaystyle\mathcal{T}\_{4,j} | :={ψ​(X;0d,β0​in)​∂4π∂h24​(Y|h2​(X,ηj∗),νj∗)},\displaystyle:=\Bigg{\{}\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯5,j\displaystyle\mathcal{T}\_{5,j} | :={∂ψβ1(u)(X;β1​j∗,β0​j∗)pG2n(Y|X),∂ψ∂β0(X;β1​j∗,β0​j∗)pG2n(Y|X),∂ψ∂β1(u)(X;0d,β0​in)pG2n(Y|X),\displaystyle:=\Bigg{\{}\frac{\partial\psi}{\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})p\_{G^{n}\_{2}}(Y|X),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})p\_{G^{n}\_{2}}(Y|X),\ \frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\beta\_{0i}^{n})p\_{G^{n}\_{2}}(Y|X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂2ψ∂β1(u)​∂β1(v)(X;0d,β0​in)pG2n(Y|X):u∈[d]},\displaystyle\hskip 113.81102pt\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\beta\_{0i}^{n})p\_{G^{n}\_{2}}(Y|X):u\in[d]\Bigg{\}}, |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}].

Stage 2 - Non-vanishing coefficients: In this stage, we demonstrate that not all the coefficients in the representations of An,0​(Y|X)/𝒟3​nA\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟3​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{3n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟3​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{3n}, Bn,0​(Y|X)/𝒟3​nB\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟3​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{3n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟3​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{3n}, Cn,0​(Y|X)/𝒟3​nC\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟3​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{3n} and [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟3​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{3n} go to zero when n→∞n\to\infty. Assume by contrary that all these coefficients converge to zero. By using the same arguments as in Stage 2 in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1𝒟3​n[∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|\displaystyle\frac{1}{\mathcal{D}\_{3n}}\Big{[}\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|} | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin(∥Δκi​jn∥2+|Δτi​jn|2)]→0,\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2})\Big{]}\to 0, |  |

as n→∞n\to\infty. Additionally, by considering the coefficients of the terms:

* •

  ∂ψ∂β1(u)​(X;β1​j∗,β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j‖Δ​β1​i​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\|\Delta\beta\_{1ij}^{n}\|\to 0; |  |
* •

  ∂ψ∂β0​(X;β1​j∗,β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j|Δ​β0​i​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}|\Delta\beta\_{0ij}^{n}|\to 0; |  |
* •

  ∂h2∂η(u2)​(X,ηj∗)​ψ​(X;β1​j∗,β0​j∗)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j‖Δ​ηi​jn‖→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}\|\Delta\eta\_{ij}^{n}\|\to 0; |  |
* •

  ψ​(X;β1​j∗,β0​j∗)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|=1∑i∈𝒱2,j|Δ​νi​jn|→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\sum\_{i\in\mathcal{V}\_{2,j}}|\Delta\nu\_{ij}^{n}|\to 0; |  |
* •

  π​(Y|h2​(X,ηj∗),νj∗)\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|>1|∑i∈𝒱2,jσ​(β0​in)−σ​(β0​j∗)|→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i}^{n})-\sigma(\beta\_{0j}^{\*})\Big{|}\to 0; |  |
* •

  ∂2ψ∂β1(u)​∂β1(v)​(X;0d,β0​in)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j‖Δ​β1​i​jn‖2→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\|\Delta\beta\_{1ij}^{n}\|^{2}\to 0; |  |
* •

  ∂2h2∂η(u2)​∂η(v2)​(X,ηj∗)​ψ​(X;0d,β0​in)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗)\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j‖Δ​ηi​jn‖2→0;\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\|\Delta\eta\_{ij}^{n}\|^{2}\to 0; |  |
* •

  ψ​(X;0d,β0​in)​∂4π∂h24​(Y|h2​(X,ηj∗),νj∗)\psi(X;0\_{d},\beta\_{0i}^{n})\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) for j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, we get

  |  |  |  |
  | --- | --- | --- |
  |  | 1𝒟3​n​∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j|Δ​νi​jn|2→0.\displaystyle\frac{1}{\mathcal{D}\_{3n}}\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}|\Delta\nu\_{ij}^{n}|^{2}\to 0. |  |

Putting the above limits together, we deduce 1=𝒟3​n𝒟3​n→01=\frac{\mathcal{D}\_{3n}}{\mathcal{D}\_{3n}}\to 0 as n→∞n\to\infty, which is a contradiction. Therefore, at least one among the coefficients in the representations of An,0​(Y|X)/𝒟3​nA\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟3​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{3n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟3​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{3n}, Bn,0​(Y|X)/𝒟3​nB\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟3​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{3n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟3​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{3n}, Cn,0​(Y|X)/𝒟3​nC\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟3​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{3n} and [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟3​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{3n} does not go to zero.

Stage 3 - Fatou’s lemma contradiction: In this stage, we use the Fatou’s lemma to show a contradiction to the result of Stage 2. For that purpose, let us denote mnm\_{n} as the maximum of the absolute values of the coefficients in the representations of An,0​(Y|X)/𝒟3​nA\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟3​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{3n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟3​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{3n}, Bn,0​(Y|X)/𝒟3​nB\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Bn,1​(Y|X)−Rn,3​(Y|X)]/𝒟3​n[B\_{n,1}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{3n}, [Bn,2​(Y|X)−Rn,4​(Y|X)]/𝒟3​n[B\_{n,2}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{3n}, Cn,0​(Y|X)/𝒟3​nC\_{n,0}(Y|X)/\mathcal{D}\_{3n}, [Cn,1​(Y|X)−Rn,5​(Y|X)]/𝒟3​n[C\_{n,1}(Y|X)-R\_{n,5}(Y|X)]/\mathcal{D}\_{3n} and [Cn,2​(Y|X)−Rn,6​(Y|X)]/𝒟3​n[C\_{n,2}(Y|X)-R\_{n,6}(Y|X)]/\mathcal{D}\_{3n}. It follows from the result of Stage 2 that 1/mn↛∞1/m\_{n}\not\to\infty as n→∞n\to\infty. In addition, we also denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)→s1,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\to s^{(u\_{1})}\_{1,j}, | 1mn​𝒟3​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)→s2,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})\to s\_{2,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)→s3,j(u1​v1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}\to s^{(u\_{1}v\_{1})}\_{3,j}, | 1mn​𝒟3​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)2→s4,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})^{2}\to s\_{4,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​τi​jn)→s5,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\to s^{(u\_{1})}\_{5,j}, | 1mn​𝒟3​n⋅(∑i∈𝒱1,jωin−ωj∗)→s0,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\to s\_{0,j}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅∑i∈𝒱2,j(Δ​β0​i​jn)→t0,j,j∈[k2∗]:|𝒱2,j|=1,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}(\Delta\beta\_{0ij}^{n})\to t\_{0,j},j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, | 1mn​𝒟3​n⋅∑i∈𝒱2,j(Δ​β1​i​jn)(u)→t1,j(u),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}(\Delta\beta\_{1ij}^{n})^{(u)}\to t^{(u)}\_{1,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅∑i∈𝒱2,j(Δ​ηi​jn)(u2)→t2,j(u2),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}(\Delta\eta\_{ij}^{n})^{(u\_{2})}\to t^{(u\_{2})}\_{2,j}, | 1mn​𝒟3​n⋅∑i∈𝒱2,j(Δ​νi​jn)→t3,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\sum\_{i\in\mathcal{V}\_{2,j}}(\Delta\nu\_{ij}^{n})\to t\_{3,j}, |  |

for all j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅(∑i∈𝒱2,jσ​(β0​in)−σ​(β0​j∗))→t0,j,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot\Big{(}\sum\_{i\in\mathcal{V}\_{2,j}}\sigma(\beta\_{0i}^{n})-\sigma(\beta\_{0j}^{\*})\Big{)}\to t\_{0,j}, | 1mn​𝒟3​n⋅(Δ​β1​i​jn)(u)→t1,j,i(u),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\beta\_{1ij}^{n})^{(u)}\to t^{(u)}\_{1,j,i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅(Δ​ηi​jn)(u2)→t2,j,i(u2),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\eta\_{ij}^{n})^{(u\_{2})}\to t^{(u\_{2})}\_{2,j,i}, | 1mn​𝒟3​n⋅(Δ​νi​jn)→t3,j,i,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\nu\_{ij}^{n})\to t\_{3,j,i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅(Δ​β1​i​jn)(u)​(Δ​β1​i​jn)(v)→t4,j,i(u​v),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\beta\_{1ij}^{n})^{(v)}\to t^{(uv)}\_{4,j,i}, | 1mn​𝒟3​n⋅=(Δηi​jn)(u2)(Δηi​jn)(v2)→t5,j,i(u2​v2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot=(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\eta\_{ij}^{n})^{(v\_{2})}\to t^{(u\_{2}v\_{2})}\_{5,j,i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅(Δ​νi​jn)2→t6,j,i,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\nu\_{ij}^{n})^{2}\to t\_{6,j,i}, | 1mn​𝒟3​n⋅(Δ​β1​i​jn)(u)​(Δ​ηi​jn)(v2)→t7,j,i(u​v2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\eta\_{ij}^{n})^{(v\_{2})}\to t^{(uv\_{2})}\_{7,j,i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟3​n⋅(Δ​β1​i​jn)(u)​(Δ​νi​jn)→t8,j,i(u),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\beta\_{1ij}^{n})^{(u)}(\Delta\nu\_{ij}^{n})\to t^{(u)}\_{8,j,i}, | 1mn​𝒟3​n⋅(Δ​ηi​jn)(u2)​(Δ​νi​jn)→t9,j,i(u2),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{3n}}\cdot(\Delta\eta\_{ij}^{n})^{(u\_{2})}(\Delta\nu\_{ij}^{n})\to t^{(u\_{2})}\_{9,j,i}, |  |

for all j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1 as n→∞n\to\infty.
Due to the result of Stage 2, at least one among the above limits is non-zero. Recall from equation ([32](#A4.E32 "In D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")) that we get

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(gG1n,G2n(⋅|X),gG1∗,G2∗(⋅|X))]/𝒟3​n→0,\displaystyle\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{3n}\to 0, |  |

Moreover, by means of the Fatou’s lemma, we have

|  |  |  |
| --- | --- | --- |
|  | limn→∞𝔼X[V(gG1n,G2n(⋅|X),gG1∗,G2∗(⋅|X))]mn​𝒟3​n≥∫lim infn→∞|gG1n,G2n(Y|X)−gG1∗,G2∗(Y|X)|2​mn​𝒟3​n​d​(X,Y).\displaystyle\lim\_{n\to\infty}\dfrac{\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{m\_{n}\mathcal{D}\_{3n}}\geq\int\liminf\_{n\to\infty}\dfrac{|g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)|}{2m\_{n}\mathcal{D}\_{3n}}\mathrm{d}(X,Y). |  |

Then, we deduce [gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)]/[mn​𝒟3​n]→0[g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{3n}]\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). Since the input space is bounded and the parameter space is compact, the quantity ∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*}) is bounded. Thus, we also have

|  |  |  |
| --- | --- | --- |
|  | [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​[gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)]/[mn​𝒟3​n]→0,\displaystyle\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}[g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{3n}]\to 0, |  |

implying that

|  |  |  |
| --- | --- | --- |
|  | 12​[∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]⋅qG1n​(Y|X)−qG1∗​(Y|X)mn​𝒟3​n+12​Qn​(Y|X)mn​𝒟3​n→0.\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot\dfrac{q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}+\frac{1}{2}\dfrac{Q\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}\to 0. |  |

as n→∞n\to\infty for almost surely (X,Y)(X,Y). From the decomposition of the terms qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) and Qn​(Y|X)Q\_{n}(Y|X) in Stage 1, we have

|  |  |  |
| --- | --- | --- |
|  | 12​[∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]⋅An,2​(Y|X)+An,1​(Y|X)+An,0​(Y|X)mn​𝒟3​n\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\cdot\dfrac{A\_{n,2}(Y|X)+A\_{n,1}(Y|X)+A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{3n}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | +12​Bn,1​(Y|X)+Bn,2​(Y|X)+Bn,3​(Y|X)−Cn,1​(Y|X)−Cn,2​(Y|X)−Cn,3​(Y|X)mn​𝒟3​n→0.\displaystyle+\frac{1}{2}\dfrac{B\_{n,1}(Y|X)+B\_{n,2}(Y|X)+B\_{n,3}(Y|X)-C\_{n,1}(Y|X)-C\_{n,2}(Y|X)-C\_{n,3}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}\to 0. |  | (34) |

We have

|  |  |  |
| --- | --- | --- |
|  | limn→∞An,0​(Y|X)mn​𝒟3​n=∑j=1k1∗s0,j​π​(Y|h1​(X,κj∗),τj∗),\displaystyle\lim\_{n\to\infty}\frac{A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j=1}^{k^{\*}\_{1}}s\_{0,j}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,1​(Y|X)mn​𝒟3​n=∑j∈[k1∗]:|𝒱1,j|=1[∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗)\displaystyle\lim\_{n\to\infty}\frac{A\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\Big{[}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +12s2,j∂2π∂h12(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 284.52756pt+\frac{1}{2}s\_{2,j}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,2​(Y|X)mn​𝒟3​n=∑j∈[k1∗]:|𝒱1,j|>1[(∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}∂2h1∂κ(u1)​∂κ(v1)(X,κj∗))\displaystyle\lim\_{n\to\infty}\frac{A\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\Big{[}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)} |  |
|  |  |  |
| --- | --- | --- |
|  | ×∂π∂h1​(Y|h1​(X,κj∗),τj∗)+(12​s2,j+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗))​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\displaystyle\times\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\Big{(}\frac{1}{2}s\_{2,j}+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +(12∑u1=1d1s5,j(u1)∂h1∂κ(u1)(X,κj∗))∂3π∂h13(Y|h1(X,κj∗),τj∗)+18s4,j∂4π∂h14(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 85.35826pt+\Big{(}\frac{1}{2}\sum\_{u\_{1}=1}^{d\_{1}}s\_{5,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\frac{1}{8}s\_{4,j}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn,0​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|>1t0,j​π​(Y|h2​(X,ηj∗),νj∗),\displaystyle\lim\_{n\to\infty}\frac{B\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}t\_{0,j}\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn,1​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|=1[(∑u=1dt1,j(u)∂ψ∂β1(u)(X;β1​j∗,β0​j∗)+t0,j∂ψ∂β0(X;β1​j∗,β0​j∗))π(Y|h2(X,ηj∗),νj∗)\displaystyle\lim\_{n\to\infty}\frac{B\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\Big{(}\sum\_{u=1}^{d}t\_{1,j}^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})+t\_{0,j}\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{)}\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑u2=1d2t2,j(u2)∂h2∂η(u2)(X,ηj∗)ψ(X;β1​j∗,β0​j∗)∂π∂h2(Y|h2(X,ηj∗),νj∗)+12t3,jψ(X;β1​j∗,β0​j∗)∂2π∂h22(Y|h2(X,ηj∗),νj∗)],\displaystyle+\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+\frac{1}{2}t\_{3,j}\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn,2​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j[(∑u,v=1dt4,j,i(u​v)1+1{u=v}∂2ψ∂β1(u)​∂β1(v)(X;0d,β¯0​i)\displaystyle\lim\_{n\to\infty}\frac{B\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\Big{(}\sum\_{u,v=1}^{d}\frac{t\_{4,j,i}^{(uv)}}{1+1\_{\{u=v\}}}\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\bar{\beta}\_{0i}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑u=1dt1,j,i(u)∂ψ∂β1(u)(X;0d,β¯0​i))π(Y|h2(X,ηj∗),νj∗)+(∑u=1d∑u2=1d2t7,j,i(u​u2)∂h2∂η(u2)(X.ηj∗)∂ψ∂β1(u)(X;0d,β¯0​i)\displaystyle+\sum\_{u=1}^{d}t\_{1,j,i}^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\Big{)}\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+\Big{(}\sum\_{u=1}^{d}\sum\_{u\_{2}=1}^{d\_{2}}t\_{7,j,i}^{(uu\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑u2=1d2t2,j,i(u2)∂h2∂η(u2)(X.ηj∗)ψ(X;0d,β¯0​i)+∑u2,v2=1d2t5,j,i(u2​v2)∂2h2∂η(u2)​∂ηv2)(X,ηj∗)ψ(X;0d,β¯0​i))∂π∂h2(Y|h2(X,ηj∗),νj∗)\displaystyle+\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X.\eta\_{j}^{\*})\psi(X;0\_{d},\bar{\beta}\_{0i})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j,i}^{(u\_{2}v\_{2})}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{v\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\bar{\beta}\_{0i})\Big{)}\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +(∑u=1d12t8,j,i(u)∂ψ∂β1(u)(X;0d,β¯0​i)+12t3,j,iψ(X;0d,β¯0​i)+∑u2,v2=1d2t5,j,i(u2​v2)1+1{u2=v2}∂h2∂η(u2)(X,ηj∗)∂h2∂η(v2)(X,ηj∗)\displaystyle+\Big{(}\sum\_{u=1}^{d}\frac{1}{2}t\_{8,j,i}^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})+\frac{1}{2}t\_{3,j,i}\psi(X;0\_{d},\bar{\beta}\_{0i})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{t\_{5,j,i}^{(u\_{2}v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | ×ψ(X;0d,β¯0​i))∂2π∂h22(Y|h2(X,ηj∗),νj∗)+∑u2=1d212t9,j,i(u2)∂h2∂η(u2)(X,ηj∗)ψ(X;0d,β¯0​i)∂3π∂h23(Y|h2(X,ηj∗),νj∗)\displaystyle\times\psi(X;0\_{d},\bar{\beta}\_{0i})\Big{)}\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})+\sum\_{u\_{2}=1}^{d\_{2}}\frac{1}{2}t\_{9,j,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{3}\pi}{\partial h\_{2}^{3}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +18t6,j,iψ(X;0d,β¯0​i)∂4π∂h24(Y|h2(X,ηj∗),νj∗)],\displaystyle\hskip 227.62204pt+\frac{1}{8}t\_{6,j,i}\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*})\Big{]}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn,0​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|>1t0,j​pG2∗​(Y|X),\displaystyle\lim\_{n\to\infty}\frac{C\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}t\_{0,j}p\_{G^{\*}\_{2}}(Y|X), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn,1​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|=1[∑u=1dt1,j(u)​∂ψβ1(u)​(X;β1​j∗,β0​j∗)+t0,j​∂ψ∂β0​(X;β1​j∗,β0​j∗)]​pG2∗​(Y|X),\displaystyle\lim\_{n\to\infty}\frac{C\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1}\Big{[}\sum\_{u=1}^{d}t\_{1,j}^{(u)}\frac{\partial\psi}{\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})+t\_{0,j}\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\Big{]}p\_{G^{\*}\_{2}}(Y|X), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn,2​(Y|X)mn​𝒟3​n=∑j∈[k2∗]:|𝒱2,j|>1∑i∈𝒱2,j[∑u=1dt1,j,i(u)∂ψβ1(u)(X;0d,β¯0​i)\displaystyle\lim\_{n\to\infty}\frac{C\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{3n}}=\sum\_{j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1}\sum\_{i\in\mathcal{V}\_{2,j}}\Big{[}\sum\_{u=1}^{d}t\_{1,j,i}^{(u)}\frac{\partial\psi}{\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑u,v=1dt4,j,i(u​v)1+1{u=v}∂2ψ∂β1(u)​∂β1(v)(X;0d,β¯0​i)]pG2∗(Y|X).\displaystyle\hskip 170.71652pt+\sum\_{u,v=1}^{d}\frac{t\_{4,j,i}^{(uv)}}{1+1\_{\{u=v\}}}\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\bar{\beta}\_{0i})\Big{]}p\_{G^{\*}\_{2}}(Y|X). |  |

Note that for almost every XX, the set

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | {[∑j=1k2∗σ((β1​j∗)⊤X+β0​j∗)]∂ρπ∂h1ρ(Y|h1(X,κj∗),τj∗):0≤ρ≤4,j∈[k1∗]}\displaystyle\Bigg{\{}\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):0\leq\rho\leq 4,\ j\in[k^{\*}\_{1}]\Bigg{\}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∪\displaystyle\cup~ | {∂ψ∂β1(u)(X;β1​j∗,β0​j∗)π(Y|h2(X,ηj∗),νj∗),∂ψ∂β0(X;β1​j∗,β0​j∗)π(Y|h2(X,ηj∗),νj∗),\displaystyle\Bigg{\{}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)​(X;β1​j∗,β0​j∗)​pG2∗​(Y|X),∂ψ∂β0​(X;β1​j∗,β0​j∗)​pG2∗​(Y|X),ψ​(X;β1​j∗,β0​j∗)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗),\displaystyle\quad\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})p\_{G^{\*}\_{2}}(Y|X),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})p\_{G^{\*}\_{2}}(Y|X),\ \psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ψ(X;β1​j∗,β0​j∗)∂2π∂h22(Y|h2(X,ηj∗),νj∗):u∈[d],j∈[k2∗]:|𝒱2,j|=1}\displaystyle\hskip 113.81102pt\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}):u\in[d],\ j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1\Bigg{\}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∪\displaystyle\cup~ | {∂ψ∂β1(u)(X;0d,β¯0​i)π(Y|h2(X,ηj∗),νj∗),∂2ψ∂β1(u)​∂β1(v)(X;0d,β¯0​i)π(Y|h2(X,ηj∗),νj∗),π(Y|h2(X,ηj∗),νj∗)\displaystyle\Bigg{\{}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\bar{\beta}\_{0i})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)​(X;0d,β¯0​i)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗),ψ​(X;0d,β¯0​i)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗),\displaystyle\quad\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)​(X;0d,β¯0​i)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗),ψ​(X;0d,β¯0​i)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗),\displaystyle\quad\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ψ​(X;0d,β¯0​i)​∂3π∂h23​(Y|h2​(X,ηj∗),νj∗),ψ​(X;0d,β¯0​i)​∂4π∂h24​(Y|h2​(X,ηj∗),νj∗),\displaystyle\quad\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{3}\pi}{\partial h\_{2}^{3}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}),\ \psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)(X;0d,β¯0​i)pG2∗(Y|X),∂2ψ∂β1(u)​∂β1(v)(X;0d,β¯0​i)pG2∗(Y|X):u,v∈[d],j∈[k2∗]:|𝒱2,j|>1,i∈𝒱2,j}\displaystyle\quad\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})p\_{G^{\*}\_{2}}(Y|X),\ \frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\bar{\beta}\_{0i})p\_{G^{\*}\_{2}}(Y|X):u,v\in[d],j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1,i\in\mathcal{V}\_{2,j}\Bigg{\}} |  |

is linearly independent w.r.t YY, implying that the coefficients of those terms in the limit in equation ([34](#A4.E34 "In D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")) are equal to zero.

For j∈[k1∗]j\in[k^{\*}\_{1}], by looking at the coefficient of the term [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​π​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s0,j=0s\_{0,j}=0.

For j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|=1|\mathcal{V}\_{1,j}|=1, by considering the coefficients of

* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have ∑u1=1d1s1,j(u1)​∂h1∂κ(u1)​(X,κj∗)=0\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})=0 for almost every XX. Since the expert function h1h\_{1} is strongly identifiable, we get s1,j(u1)=0s\_{1,j}^{(u\_{1})}=0 for all u1∈[d1]u\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s2,j=0s\_{2,j}=0.

For j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|>1|\mathcal{V}\_{1,j}|>1, by taking into account the coefficients of

* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂π∂h1​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | ∑u1=1d1s1,j​∂h1∂κ(u1)​(X,κj∗)+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)=0,\displaystyle\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})=0, |  |

  for almost every XX. Since the expert function h1h\_{1} satisfies the strong identifiability condition, we get s1,j(u1)=s3,j(u1​v1)=0s\_{1,j}^{(u\_{1})}=s\_{3,j}^{(u\_{1}v\_{1})}=0 for all u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | 12​s2,j+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)=0,\displaystyle\frac{1}{2}s\_{2,j}+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})=0, |  |

  for almost every XX. Since s3,j(u1​v1)=0s\_{3,j}^{(u\_{1}v\_{1})}=0 for all u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}], we deduce s2,j=0s\_{2,j}=0;
* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂3π∂h13​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have 12​∑u1=1d1s5,j(u1)​∂h1∂κ(u1)​(X,κj∗)=0\frac{1}{2}\sum\_{u\_{1}=1}^{d\_{1}}s\_{5,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})=0, for almost every XX. As the expert function h1h\_{1} meets the strong identifiability condition, we get s5,j(u1)=0s\_{5,j}^{(u\_{1})}=0 for all u1∈[d1]u\_{1}\in[d\_{1}];
* •

  [∑j=1k2∗σ​((β1​j∗)⊤​X+β0​j∗)]​∂4π∂h14​(Y|h1​(X,κj∗),τj∗)\Big{[}\sum\_{j=1}^{k^{\*}\_{2}}\sigma((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})\Big{]}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}), we have s4,j=0s\_{4,j}=0.

For j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|=1|\mathcal{V}\_{2,j}|=1, by considering the coefficients of

* •

  ∂ψ∂β1(u)​(X;β1​j∗,β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t1,j(u)=0t\_{1,j}^{(u)}=0 for all u∈[d]u\in[d];
* •

  ∂ψ∂β0​(X;β1​j∗,β0​j∗)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{0}}(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t0,j=0t\_{0,j}=0;
* •

  ψ​(X;β1​j∗,β0​j∗)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have ∑u2=1d2t2,j(u2)​∂h2∂η(u2)​(X,ηj∗)=0\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})=0. Since the expert function h2h\_{2} is strongly identifiable, we deduce t2,j(u2)=0t\_{2,j}^{(u\_{2})}=0 for all u2∈[d2]u\_{2}\in[d\_{2}];
* •

  ψ​(X;β1​j∗,β0​j∗)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗)\psi(X;\beta\_{1j}^{\*},\beta\_{0j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t3,j=0t\_{3,j}=0.

For j∈[k2∗]j\in[k^{\*}\_{2}] such that |𝒱2,j|>1|\mathcal{V}\_{2,j}|>1, by considering the coefficients of

* •

  π​(Y|h2​(X,ηj∗),νj∗)\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t0,j=0t\_{0,j}=0;
* •

  ∂ψ∂β1(u)​(X;0d,β¯0​i)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t1,j,i(u)=0t\_{1,j,i}^{(u)}=0 for all u∈[d]u\in[d] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ∂2ψ∂β1(u)​∂β1(v)​(X;0d,β¯0​i)​π​(Y|h2​(X,ηj∗),νj∗)\frac{\partial^{2}\psi}{\partial\beta\_{1}^{(u)}\partial\beta\_{1}^{(v)}}(X;0\_{d},\bar{\beta}\_{0i})\pi(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t4,j,i(u​v)=0t\_{4,j,i}^{(uv)}=0 for all u,v∈[d]u,v\in[d] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ψ​(X;0d,β¯0​i)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | ∑u2=1d2t2,j,i(u2)​∂h2∂η(u2)​(X,ηj∗)+∑u2,v2=1d2t5,j,i(u2​v2)​∂2h2∂η(u2)​∂η(v2)​(X,ηj∗)=0.\displaystyle\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,j,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}t\_{5,j,i}^{(u\_{2}v\_{2})}\frac{\partial^{2}h\_{2}}{\partial\eta^{(u\_{2})}\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})=0. |  |

  As the expert function h2h\_{2} satisfies the strong identifiability condition, we deduce t2,j,i(u2)=t5,j,i(u2​v2)=0t\_{2,j,i}^{(u\_{2})}=t\_{5,j,i}^{(u\_{2}v\_{2})}=0 for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ∂ψ∂β1(u)​(X;0d,β¯0​i)​∂π∂h2​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have ∑u2=1d2t7,j,i(u​u2)​∂h2∂η(u2)​(X,ηj∗)=0\sum\_{u\_{2}=1}^{d\_{2}}t\_{7,j,i}^{(uu\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})=0. Since the expert function h2h\_{2} is strongly identifiable, we deduce t7,j,i(u​u2)=0t\_{7,j,i}^{(uu\_{2})}=0 for all u∈[d]u\in[d], u2∈[d2]u\_{2}\in[d\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ψ​(X;0d,β¯0​i)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗)\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have

  |  |  |  |
  | --- | --- | --- |
  |  | 12​t3,j,i+∑u2,v2=1d2t5,j,i(u2​v2)1+1{u2=v2}​∂h2∂η(u2)​(X,ηj∗)​∂h2∂η(v2)​(X,ηj∗)=0.\displaystyle\frac{1}{2}t\_{3,j,i}+\sum\_{u\_{2},v\_{2}=1}^{d\_{2}}\frac{t\_{5,j,i}^{(u\_{2}v\_{2})}}{1+1\_{\{u\_{2}=v\_{2}\}}}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\frac{\partial h\_{2}}{\partial\eta^{(v\_{2})}}(X,\eta\_{j}^{\*})=0. |  |

  Note that t5,j,i(u2​v2)=0t\_{5,j,i}^{(u\_{2}v\_{2})}=0 for all u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j}, we deduce t3,j,i=0t\_{3,j,i}=0 for all i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ∂ψ∂β1(u)​(X;0d,β¯0​i)​∂2π∂h22​(Y|h2​(X,ηj∗),νj∗)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t8,j,i(u)=0t\_{8,j,i}^{(u)}=0 for all u∈[d]u\in[d] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ψ​(X;0d,β¯0​i)​∂3π∂h23​(Y|h2​(X,ηj∗),νj∗)\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{3}\pi}{\partial h\_{2}^{3}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have ∑u2=1d212​t9,j,i(u2)​∂h2∂η(u2)​(X,ηj∗)=0\sum\_{u\_{2}=1}^{d\_{2}}\frac{1}{2}t\_{9,j,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})=0. Since the expert function h2h\_{2} meets the strong identifiability, we deduce t9,j,i(u2)t\_{9,j,i}^{(u\_{2})} for all u2∈[d2]u\_{2}\in[d\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j};
* •

  ψ​(X;0d,β¯0​i)​∂4π∂h24​(Y|h2​(X,ηj∗),νj∗)\psi(X;0\_{d},\bar{\beta}\_{0i})\frac{\partial^{4}\pi}{\partial h\_{2}^{4}}(Y|h\_{2}(X,\eta\_{j}^{\*}),\nu\_{j}^{\*}), we have t6,j,i=0t\_{6,j,i}=0 for all i∈𝒱2,ji\in\mathcal{V}\_{2,j}.

Putting the above results together, we have (i) s0,j=s1,j(u1)=s2,j=s3,j(u1​v1)=s4,j=s5,j(u1)=0s\_{0,j}=s\_{1,j}^{(u\_{1})}=s\_{2,j}=s\_{3,j}^{(u\_{1}v\_{1})}=s\_{4,j}=s\_{5,j}^{(u\_{1})}=0 for all j∈[k1∗]j\in[k^{\*}\_{1}] and u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}]; (ii) t0,j=t1,j(u)=t2,j(u2)=t3,j=0t\_{0,j}=t\_{1,j}^{(u)}=t\_{2,j}^{(u\_{2})}=t\_{3,j}=0 for all j∈[k2∗]:|𝒱2,j|=1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|=1, u∈[d]u\in[d] and u2∈[d2]u\_{2}\in[d\_{2}]; (iii) t0,j=t1,j,i(u)=t2,j,i(u2)=t3,j,i=t4,j,i(u​v)=t5,j,i(u2​v2)=t6,j,i=t7,j,iu​v2=t8,j,i(u)=t9,j,i(u2)t\_{0,j}=t\_{1,j,i}^{(u)}=t\_{2,j,i}^{(u\_{2})}=t\_{3,j,i}=t\_{4,j,i}^{(uv)}=t\_{5,j,i}^{(u\_{2}v\_{2})}=t\_{6,j,i}=t\_{7,j,i}^{uv\_{2}}=t\_{8,j,i}^{(u)}=t\_{9,j,i}^{(u\_{2})} for all j∈[k2∗]:|𝒱2,j|>1j\in[k^{\*}\_{2}]:|\mathcal{V}\_{2,j}|>1, u,v∈[d]u,v\in[d] and u2,v2∈[d2]u\_{2},v\_{2}\in[d\_{2}]. This contradicts to the fact that at least one among them is non-zero. Consequently, we achieve the local part in equation ([30](#A4.E30 "In D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")) and complete the proof.

### D.4 Proof of Theorem [4](#Thmtheorem4 "Theorem 4. ‣ A.2 Dense Regime ‣ Appendix A On Normalized Sigmoid Gating")

Note that it is sufficient to demonstrate that

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ)𝔼X[V(gG1,G2(⋅|X),gG1∗,Gˇ2(⋅|X))]𝒟4​((G1,G2),(G1∗,Gˇ2))>0,\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)}\dfrac{\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]}{\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2}))}>0, |  |

for any pair of mixing measures (G1∗,Gˇ2)∈𝒢ˇk1∗,k2​(Θ)(G^{\*}\_{1},\check{G}\_{2})\in\check{\mathcal{G}}\_{k^{\*}\_{1},k\_{2}}(\Theta). For that purpose, given an arbitrary mixing measure Gˇ2:=∑i=1k2σ​(βˇ0​i)​δ(βˇ1​i,ηˇi,νˇi)\check{G}\_{2}:=\sum\_{i=1}^{k\_{2}}\sigma(\check{\beta}\_{0i})\delta\_{(\check{\beta}\_{1i},\check{\eta}\_{i},\check{\nu}\_{i})}, we need to establish its local part

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟4​((G1,G2),(G1∗,Gˇ2))≤ε𝔼X[V(gG1,G2(⋅|X),gG1∗,Gˇ2(⋅|X))]𝒟4​((G1,G2),(G1∗,Gˇ2))>0,\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]}{\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2}))}>0, |  | (35) |

and its global part

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟4​((G1,G2),(G1∗,Gˇ2))>ε′𝔼X[V(gG1,G2(⋅|X),gG1∗,Gˇ2(⋅|X))]𝒟4​((G1,G2),(G1∗,Gˇ2))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(g\_{G\_{1},G\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]}{\mathcal{D}\_{4}((G\_{1},G\_{2}),(G^{\*}\_{1},\check{G}\_{2}))}>0. |  | (36) |

Since the global part ([36](#A4.E36 "In D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")) can be demonstrated analogously to that in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), we will focus only on proving the local part ([35](#A4.E35 "In D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")) in this appendix. Assume by contrary that the above local part is not true. Then, we can find a sequence (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) of the form G1n:=∑i=1k1nωin​δ(κin,τin)G^{n}\_{1}:=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{i}^{n},\tau\_{i}^{n})}, G2n:=∑i=1k2nσ​(β0​in)​δ(β1​in,ηin,νin)G^{n}\_{2}:=\sum\_{i=1}^{k^{n}\_{2}}\sigma(\beta\_{0i}^{n})\delta\_{(\beta\_{1i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})} for n∈ℕn\in\mathbb{N} satisfying 𝒟4​n:=𝒟4​((G1n,G2n),(G1∗,Gˇ2))→0\mathcal{D}\_{4n}:=\mathcal{D}\_{4}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},\check{G}\_{2}))\to 0 and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(gG1n,G2n(⋅|X),gG1∗,Gˇ2(⋅|X))]/𝒟4​n→0,\displaystyle\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]/\mathcal{D}\_{4n}\to 0, |  | (37) |

as n→∞n\to\infty. Moreover, we may assume WLOG that the number of shared experts k1nk^{n}\_{1}, the number of routed experts k2nk^{n}\_{2}, and Voronoi cells 𝒱1,j=𝒱1,j​(G1n)\mathcal{V}\_{1,j}=\mathcal{V}\_{1,j}(G^{n}\_{1}), 𝒱2,j=𝒱2,j​(G2n)\mathcal{V}\_{2,j}=\mathcal{V}\_{2,j}(G^{n}\_{2}) are independent of the sample size nn. In addition, since G2nG^{n}\_{2} and Gˇ2\check{G}\_{2} have the same number of atoms k2k\_{2}, we may assume WLOG that the Voronoi cell 𝒱2,j\mathcal{V}\_{2,j} admits only one element, that is, 𝒱2,j={j}\mathcal{V}\_{2,j}=\{j\} for all j∈[k2]j\in[k\_{2}]. Thus, we can represent the Voronoi loss 𝒟4​n\mathcal{D}\_{4n} as

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟4​n=∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑i=1k2∗(‖Δ​βˇ1​in‖+|Δ​βˇ0​in|+‖Δ​ηˇin‖+|Δ​νˇin|)\displaystyle\mathcal{D}\_{4n}=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{i=1}^{k^{\*}\_{2}}(\|\Delta\check{\beta}\_{1i}^{n}\|+|\Delta\check{\beta}\_{0i}^{n}|+\|\Delta\check{\eta}\_{i}^{n}\|+|\Delta\check{\nu}\_{i}^{n}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)+∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖2+|Δ​τi​jn|2),\displaystyle\hskip 28.45274pt+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|)+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2}), |  | (38) |

where we denote Δ​βˇ1​in:=β1​in−βˇ1​i\Delta\check{\beta}\_{1i}^{n}:=\beta\_{1i}^{n}-\check{\beta}\_{1i}, Δ​βˇ0​in:=β0​in−βˇ0​i\Delta\check{\beta}\_{0i}^{n}:=\beta\_{0i}^{n}-\check{\beta}\_{0i}, Δ​ηˇin:=ηin−ηˇi\Delta\check{\eta}\_{i}^{n}:=\eta\_{i}^{n}-\check{\eta}\_{i}, and Δ​νˇin:=νin−νˇi\Delta\check{\nu}\_{i}^{n}:=\nu\_{i}^{n}-\check{\nu}\_{i} for all i∈[k2]i\in[k\_{2}].
Recall that 𝒟4​n→0\mathcal{D}\_{4n}\to 0 as n→∞n\to\infty, then equation ([D.4](#A4.Ex366 "D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")) implies that as n→∞n\to\infty, we have

* •

  For j∈[k1∗]j\in[k^{\*}\_{1}] and i∈𝒱1,ji\in\mathcal{V}\_{1,j}: ∑i∈𝒱1,jωin→ωj∗\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\to\omega\_{j}^{\*}, (κin,τin)→(κj∗,τj∗)(\kappa\_{i}^{n},\tau\_{i}^{n})\to(\kappa\_{j}^{\*},\tau\_{j}^{\*});
* •

  For i∈[k2∗]i\in[k^{\*}\_{2}]: (β1​in,β0​in,ηin,νin)→(βˇ1​i,βˇ0​i,ηˇi,νˇi)(\beta\_{1i}^{n},\beta\_{0i}^{n},\eta\_{i}^{n},\nu\_{i}^{n})\to(\check{\beta}\_{1i},\check{\beta}\_{0i},\check{\eta}\_{i},\check{\nu}\_{i}).

Now, we divide the proof into three main stages:

Stage 1 - Density Decomposition: In this step, we reuse the following decomposition of the density discrepancy gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) in Appendix [D.3](#A4.SS3 "D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results")

|  |  |  |
| --- | --- | --- |
|  | gG1n,G2n​(Y|X)−gG1∗,G2∗​(Y|X)=12​[(qG1n​(Y|X)−qG1∗​(Y|X))+(pG2n​(Y|X)−pG2∗​(Y|X))],\displaystyle g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)=\frac{1}{2}\left[(q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X))+(p\_{G^{n}\_{2}}(Y|X)-p\_{G^{\*}\_{2}}(Y|X))\right], |  |

where we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X) | :=∑i=1k1nωin​π​(Y|h1​(X,κin),τin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{1}}\omega^{n}\_{i}\pi(Y|h\_{1}(X,\kappa^{n}\_{i}),\tau^{n}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1∗​(Y|X)\displaystyle q\_{G^{\*}\_{1}}(Y|X) | :=∑i=1k1∗ωi∗​π​(Y|h1​(X,κi∗),τi∗),\displaystyle:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2n​(Y|X)\displaystyle p\_{G^{n}\_{2}}(Y|X) | :=∑i=1k2nσ​((β1​in)⊤​X+β0​in)∑j=1k2nσ​((β1​jn)⊤​X+β0​jn)⋅π​(Y|h2​(X,ηin),νin),\displaystyle:=\sum\_{i=1}^{k^{n}\_{2}}\frac{\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})}{\sum\_{j=1}^{k^{n}\_{2}}\sigma((\beta\_{1j}^{n})^{\top}X+\beta\_{0j}^{n})}\cdot\pi(Y|h\_{2}(X,\eta^{n}\_{i}),\nu\_{i}^{n}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pGˇ2​(Y|X)\displaystyle p\_{\check{G}\_{2}}(Y|X) | :=∑i=1k2σ​((βˇ1​i)⊤​X+βˇ0​i)∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)⋅π​(Y|h2​(X,ηˇi),νˇi).\displaystyle:=\sum\_{i=1}^{k\_{2}}\frac{\sigma((\check{\beta}\_{1i})^{\top}X+\check{\beta}\_{0i})}{\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})}\cdot\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}). |  |

Stage 1.1: We also utilize the decomposition of the term qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) in Appendix [D.3](#A4.SS3 "D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results") as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1n​(Y|X)−qG1∗​(Y|X)\displaystyle q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin​[π​(Y|h1​(X,κin),τin)−π​(Y|h1​(X,κj∗),τj∗)]\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}[\pi(Y|h\_{1}(X,\kappa\_{i}^{n}),\tau\_{i}^{n})-\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j=1k1∗(∑i∈𝒱1,jωin−ωj∗)​π​(Y|h1​(X,κj∗),τj∗)\displaystyle+\sum\_{j=1}^{k^{\*}\_{1}}\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=An,1​(Y|X)+An,2​(Y|X)+An,0​(Y|X).\displaystyle:=A\_{n,1}(Y|X)+A\_{n,2}(Y|X)+A\_{n,0}(Y|X). |  |

Above, the quantity An,1​(Y|X)A\_{n,1}(Y|X) is expanded as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1​(Y|X)\displaystyle A\_{n,1}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|=1∑ρ=12An,1,ρ(j)​(X)​∂ρπ∂h1ρ​(Y|h1​(X,κj∗),τj∗)+Rn,1​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{\rho=1}^{2}A^{(j)}\_{n,1,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,1}(Y|X), |  |

where Rn,1​(Y|X)R\_{n,1}(Y|X) is a Taylor remainder such that Rn,1​(Y|X)/𝒟4​n→R\_{n,1}(Y|X)/\mathcal{D}\_{4n}\to as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,1(j)​(X)\displaystyle A^{(j)}\_{n,1,1}(X) | :=∑i∈𝒱1,jωin​∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,1,2(j)​(X)\displaystyle A^{(j)}\_{n,1,2}(X) | :=∑i∈𝒱1,jωin​12​(Δ​τi​jn),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{2}(\Delta\tau\_{ij}^{n}), |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|=1|\mathcal{V}\_{1,j}|=1. In addition, we can rewrite An,2​(Y|X)A\_{n,2}(Y|X) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2​(Y|X)\displaystyle A\_{n,2}(Y|X) | =∑j∈[k1∗]:|𝒱1,j|>1∑ρ=14An,1,ρ(j)​(X)​∂ρπ∂h1ρ​(Y|h1​(X,κj∗),τj∗)+Rn,2​(Y|X),\displaystyle=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{\rho=1}^{4}A^{(j)}\_{n,1,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+R\_{n,2}(Y|X), |  |

where Rn,2​(Y|X)R\_{n,2}(Y|X) is a Taylor remainder such that Rn,2​(Y|X)/𝒟4​n→R\_{n,2}(Y|X)/\mathcal{D}\_{4n}\to as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,1(j)​(X)\displaystyle A^{(j)}\_{n,2,1}(X) | :=∑i∈𝒱1,jωin​(∑u1=1d1(Δ​κi​jn)(u1)​∂h1∂κ(u1)​(X,κj∗)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂2h1∂κ(u1)​∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,2(j)​(X)\displaystyle A^{(j)}\_{n,2,2}(X) | :=∑i∈𝒱1,jωin​(12​(Δ​τi​jn)+∑u1,v1=1d1(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗)),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\Big{(}\frac{1}{2}(\Delta\tau\_{ij}^{n})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,3(j)​(X)\displaystyle A^{(j)}\_{n,2,3}(X) | :=∑i∈𝒱1,jωin​∑u1=1d112​(Δ​κi​jn)(u1)​(Δ​τi​jn)​∂h1∂κ(u1)​(X,κj∗),\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\sum\_{u\_{1}=1}^{d\_{1}}\frac{1}{2}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | An,2,4(j)​(X)\displaystyle A^{(j)}\_{n,2,4}(X) | :=∑i∈𝒱1,jωin​18​(Δ​τi​jn)2,\displaystyle:=\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}\frac{1}{8}(\Delta\tau\_{ij}^{n})^{2}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] such that |𝒱1,j|>1|\mathcal{V}\_{1,j}|>1.

Stage 1.2: Next, we decompose the term Qn​(Y|X):=[∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)]⋅[pG2n​(Y|X)−pGˇ2​(Y|X)]Q\_{n}(Y|X):=\Big{[}\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})\Big{]}\cdot[p\_{G^{n}\_{2}}(Y|X)-p\_{\check{G}\_{2}}(Y|X)] as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Qn​(Y|X)\displaystyle Q\_{n}(Y|X) | =∑i=1k2[σ​((β1​in)⊤​X+β0​in)​π​(Y|h2​(X,ηin),νin)−σ​((βˇ1​i)⊤​X+βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)]\displaystyle=\sum\_{i=1}^{k\_{2}}\Big{[}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\sigma((\check{\beta}\_{1i})^{\top}X+\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −∑i=1k2[σ​((β1​in)⊤​X+β0​in)−σ​((βˇ1​i)⊤​X+βˇ0​i)]​pG2n​(Y|X)\displaystyle-\sum\_{i=1}^{k\_{2}}\Big{[}\sigma((\beta\_{1i}^{n})^{\top}X+\beta\_{0i}^{n})-\sigma((\check{\beta}\_{1i})^{\top}X+\check{\beta}\_{0i})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1k2[ψ​(X;β1​in,β0​in)​π​(Y|h2​(X,ηin),νin)−ψ​(X;βˇ1​i,βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)]\displaystyle=\sum\_{i=1}^{k\_{2}}\Big{[}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n})-\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −∑i=1k2[ψ​(X;β1​in,β0​in)−ψ​(X;βˇ1​i,βˇ0​i)]​pG2n​(Y|X)\displaystyle-\sum\_{i=1}^{k\_{2}}\Big{[}\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})-\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\Big{]}p\_{G^{n}\_{2}}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | :=Bn​(Y|X)−Cn​(Y|X),\displaystyle:=B\_{n}(Y|X)-C\_{n}(Y|X), |  |

where we denote ψ​(X;β1,β0):=σ​(β1⊤​X+β0)\psi(X;\beta\_{1},\beta\_{0}):=\sigma(\beta\_{1}^{\top}X+\beta\_{0}).

Stage 1.2.1: In this step, we decompose Bn​(Y|X)B\_{n}(Y|X) by applying the first-order Taylor expansion to the function ψ​(X;β1​in,β0​in)​π​(Y|h2​(X,ηin),νin)\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n})\pi(Y|h\_{2}(X,\eta\_{i}^{n}),\nu\_{i}^{n}) around the point (βˇ1​i,βˇ0​i,ηˇi,νˇi)(\check{\beta}\_{1i},\check{\beta}\_{0i},\check{\eta}\_{i},\check{\nu}\_{i}) as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn​(Y|X)\displaystyle B\_{n}(Y|X) | =∑i=1k2∑|α|=1(Δ​βˇ1​in)α1​(Δ​βˇ0​in)α2​(Δ​ηˇin)α3​(Δ​νˇin)α4\displaystyle=\sum\_{i=1}^{k\_{2}}\sum\_{|\alpha|=1}(\Delta\check{\beta}\_{1i}^{n})^{\alpha\_{1}}(\Delta\check{\beta}\_{0i}^{n})^{\alpha\_{2}}(\Delta\check{\eta}\_{i}^{n})^{\alpha\_{3}}(\Delta\check{\nu}\_{i}^{n})^{\alpha\_{4}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ×∂|α1|+α2ψ∂β1α1​∂β0α2​(X;βˇ1​i,βˇ0​i)​∂|α3|+α4π∂ηα3​∂να4​(Y|h2​(X,ηˇi),νˇi)+Rn,3​(Y|X)\displaystyle\hskip 85.35826pt\times\frac{\partial^{|\alpha\_{1}|+\alpha\_{2}}\psi}{\partial\beta\_{1}^{\alpha\_{1}}\partial\beta\_{0}^{\alpha\_{2}}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial^{|\alpha\_{3}|+\alpha\_{4}}\pi}{\partial\eta^{\alpha\_{3}}\partial\nu^{\alpha\_{4}}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})+R\_{n,3}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1k2∑ρ=02Bn,ρ(i)​(X)​∂ρπ∂h2ρ​(Y|h2​(X,ηˇi),νˇi)+Rn,3​(Y|X),\displaystyle=\sum\_{i=1}^{k\_{2}}\sum\_{\rho=0}^{2}B^{(i)}\_{n,\rho}(X)\frac{\partial^{\rho}\pi}{\partial h\_{2}^{\rho}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})+R\_{n,3}(Y|X), |  |

where Rn,3​(Y|X)R\_{n,3}(Y|X) is a Taylor remainder such that Rn,3​(Y|X)/𝒟4​n→R\_{n,3}(Y|X)/\mathcal{D}\_{4n}\to as n→∞n\to\infty, and

|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,0(i)\displaystyle B\_{n,0}^{(i)} | :=∑u=1d(Δ​βˇ1​in)(u)​∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)+(Δ​βˇ0​in)​∂ψ∂β0​(X;βˇ1​i,βˇ0​i),\displaystyle:=\sum\_{u=1}^{d}(\Delta\check{\beta}\_{1i}^{n})^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})+(\Delta\check{\beta}\_{0i}^{n})\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,1(i)\displaystyle B\_{n,1}^{(i)} | :=∑u2=1d2(Δ​ηˇin)(u2)​∂h2∂η(u2)​(X,ηˇi)​ψ​(X;βˇ1​i,βˇ0​i),\displaystyle:=\sum\_{u\_{2}=1}^{d\_{2}}(\Delta\check{\eta}\_{i}^{n})^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\check{\eta}\_{i})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Bn,2(i)\displaystyle B\_{n,2}^{(i)} | :=12​(Δ​νˇin)​ψ​(X;βˇ1​i,βˇ0​i),\displaystyle:=\frac{1}{2}(\Delta\check{\nu}\_{i}^{n})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i}), |  |

for all i∈[k2]i\in[k\_{2}].

Stage 1.2.2: Next, we proceed to decompose Cn​(Y|X)C\_{n}(Y|X) by applying the first-order Taylor expansion to the function ψ​(X;β1​in,β0​in)\psi(X;\beta\_{1i}^{n},\beta\_{0i}^{n}) around the point (βˇ1​i,βˇ0​i)(\check{\beta}\_{1i},\check{\beta}\_{0i}) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cn​(Y|X)\displaystyle C\_{n}(Y|X) | =∑i=1k2∑|α|=1(Δ​βˇ1​in)α1​(Δ​βˇ0​in)α2​∂|α1|+α2ψ∂β1α1​∂β0α2​(X;βˇ1​i,βˇ0​i)​pG2n​(Y|X)+Rn,4​(Y|X)\displaystyle=\sum\_{i=1}^{k\_{2}}\sum\_{|\alpha|=1}(\Delta\check{\beta}\_{1i}^{n})^{\alpha\_{1}}(\Delta\check{\beta}\_{0i}^{n})^{\alpha\_{2}}\frac{\partial^{|\alpha\_{1}|+\alpha\_{2}}\psi}{\partial\beta\_{1}^{\alpha\_{1}}\partial\beta\_{0}^{\alpha\_{2}}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})p\_{G^{n}\_{2}}(Y|X)+R\_{n,4}(Y|X) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑i=1k2[∑u=1d(Δ​βˇ1​in)(u)​∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)+(Δ​βˇ0​in)​∂ψ∂β0​(X;βˇ1​i,βˇ0​i)]​pG2n​(Y|X)+Rn,4​(Y|X),\displaystyle=\sum\_{i=1}^{k\_{2}}\Big{[}\sum\_{u=1}^{d}(\Delta\check{\beta}\_{1i}^{n})^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})+(\Delta\check{\beta}\_{0i}^{n})\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\Big{]}p\_{G^{n}\_{2}}(Y|X)+R\_{n,4}(Y|X), |  |

where Rn,4​(Y|X)R\_{n,4}(Y|X) is a Taylor remainder such that Rn,4​(Y|X)/𝒟4​n→R\_{n,4}(Y|X)/\mathcal{D}\_{4n}\to as n→∞n\to\infty.

Combining the above decompositions, we can view An,0​(Y|X)/𝒟4​nA\_{n,0}(Y|X)/\mathcal{D}\_{4n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟4​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{4n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟4​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{4n}, [Bn​(Y|X)−Rn,3​(Y|X)]/𝒟4​n[B\_{n}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{4n}, [Cn​(Y|X)−Rn,4​(Y|X)]/𝒟4​n[C\_{n}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{4n} as a combination of elements from the following sets

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮0,j\displaystyle\mathcal{S}\_{0,j} | :={π​(Y|h1​(X,κj∗),τj∗)},\displaystyle:=\{\pi(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮1,j\displaystyle\mathcal{S}\_{1,j} | :={∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗),∂2h1∂κ(u1)​∂κ(v1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮2,j\displaystyle\mathcal{S}\_{2,j} | :={∂2π∂h12(Y|h1(X,κj∗),τj∗),∂h1∂κ(u1)(X,κj∗)∂h1∂κ(v1)(X,κj∗)∂2π∂h12(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}),\ \frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮3,j\displaystyle\mathcal{S}\_{3,j} | :={∂h1∂κ(u1)(X,κj∗)∂3π∂h13(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒮4,j\displaystyle\mathcal{S}\_{4,j} | :={∂4π∂h14(Y|h1(X,κj∗),τj∗):u1,v1∈[d1]},\displaystyle:=\Bigg{\{}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):u\_{1},v\_{1}\in[d\_{1}]\Bigg{\}}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}], and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯0,j\displaystyle\mathcal{T}\_{0,j} | :={∂ψ∂β1(u)(X;βˇ1​i,βˇ0​i)π(Y|h2(X,ηˇi),νˇi),∂ψ∂β0(X;βˇ1​i,βˇ0​i)π(Y|h2(X,ηˇi),νˇi),\displaystyle:=\Bigg{\{}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)(X;βˇ1​i,βˇ0​i)pG2n(Y|X),∂ψ∂β0(X;βˇ1​i,βˇ0​i)pG2n(Y|X):u∈[d]},\displaystyle\hskip 56.9055pt\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})p\_{G^{n}\_{2}}(Y|X),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})p\_{G^{n}\_{2}}(Y|X):u\in[d]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯1,j\displaystyle\mathcal{T}\_{1,j} | :={∂h2∂η(u2)(X,ηj∗)ψ(X;βˇ1​i,βˇ0​i)∂π∂h2(Y|h2(X,ηˇi),νˇi):u∈[d],u2∈[d2]},\displaystyle:=\Bigg{\{}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\eta\_{j}^{\*})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}):u\in[d],\ u\_{2}\in[d\_{2}]\Bigg{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒯2,j\displaystyle\mathcal{T}\_{2,j} | :={ψ​(X;βˇ1​i,βˇ0​i)​∂2π∂h22​(Y|h2​(X,ηˇi),νˇi)},\displaystyle:=\Bigg{\{}\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})\Bigg{\}}, |  |

for all j∈[k2∗]j\in[k^{\*}\_{2}].

Stage 2 - Non-vanishing coefficients: In this stage, we show that at least one among the coefficients in the representations of An,0​(Y|X)/𝒟4​nA\_{n,0}(Y|X)/\mathcal{D}\_{4n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟4​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{4n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟4​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{4n}, [Bn​(Y|X)−Rn,3​(Y|X)]/𝒟4​n[B\_{n}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{4n}, [Cn​(Y|X)−Rn,4​(Y|X)]/𝒟4​n[C\_{n}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{4n} does not converge to zero when n→∞n\to\infty. Suppose that all these coefficients go to zero. By using the same arguments as in Stage 2 in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1𝒟4​n[∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|\displaystyle\frac{1}{\mathcal{D}\_{4n}}\Big{[}\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|} | +∑j∈[k1∗]:|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗]:|𝒱1,j|>1∑i∈𝒱1,jωin(∥Δκi​jn∥2+|Δτi​jn|2)]→0,\displaystyle+\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2})\Big{]}\to 0, |  |

as n→∞n\to\infty. Additionally, by considering the coefficients of the terms:

* •

  ∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) for i∈[k2]i\in[k\_{2}], we get 1𝒟4​n​∑i=1k2‖Δ​β1​i​jn‖→0\frac{1}{\mathcal{D}\_{4n}}\sum\_{i=1}^{k\_{2}}\|\Delta\beta\_{1ij}^{n}\|\to 0;
* •

  ∂ψ∂β0​(X;βˇ1​i,βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) for i∈[k2]i\in[k\_{2}], we get 1𝒟4​n​∑i=1k2|Δ​β0​i​jn|→0\frac{1}{\mathcal{D}\_{4n}}\sum\_{i=1}^{k\_{2}}|\Delta\beta\_{0ij}^{n}|\to 0;
* •

  ∂h2∂η(u2)​(X,ηˇi)​ψ​(X;βˇ1​i,βˇ0​i)​∂π∂h2​(Y|h2​(X,ηˇi),νˇi)\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\check{\eta}\_{i})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) for i∈[k2]i\in[k\_{2}], we get 1𝒟4​n​∑i=1k2‖Δ​ηi​jn‖→0\frac{1}{\mathcal{D}\_{4n}}\sum\_{i=1}^{k\_{2}}\|\Delta\eta\_{ij}^{n}\|\to 0;
* •

  ψ​(X;βˇ1​i,βˇ0​i)​∂π∂h2​(Y|h2​(X,ηˇi),νˇi)\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) for i∈[k2]i\in[k\_{2}], we get 1𝒟4​n​∑i=1k2|Δ​νi​jn|→0\frac{1}{\mathcal{D}\_{4n}}\sum\_{i=1}^{k\_{2}}|\Delta\nu\_{ij}^{n}|\to 0.

Taking the summation of the above limits, we deduce 1=𝒟4​n𝒟4​n→01=\frac{\mathcal{D}\_{4n}}{\mathcal{D}\_{4n}}\to 0 as n→∞n\to\infty, which is a contradiction. Thus, not all the coefficients in the representations of An,0​(Y|X)/𝒟4​nA\_{n,0}(Y|X)/\mathcal{D}\_{4n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟4​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{4n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟4​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{4n}, [Bn​(Y|X)−Rn,3​(Y|X)]/𝒟4​n[B\_{n}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{4n}, [Cn​(Y|X)−Rn,4​(Y|X)]/𝒟4​n[C\_{n}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{4n} converge to zero as n→∞n\to\infty.

Stage 3 - Fatou’s lemma contradiction: In this stage, we attempto to show a contradiction to the result of Stage 2 using the Fatou’s lemma. Firstly, we denote mnm\_{n} as the maximum of the absolute values of the coefficients in the representations of An,0​(Y|X)/𝒟4​nA\_{n,0}(Y|X)/\mathcal{D}\_{4n}, [An,1​(Y|X)−Rn,1​(Y|X)]/𝒟4​n[A\_{n,1}(Y|X)-R\_{n,1}(Y|X)]/\mathcal{D}\_{4n}, [An,2​(Y|X)−Rn,2​(Y|X)]/𝒟4​n[A\_{n,2}(Y|X)-R\_{n,2}(Y|X)]/\mathcal{D}\_{4n}, [Bn​(Y|X)−Rn,3​(Y|X)]/𝒟4​n[B\_{n}(Y|X)-R\_{n,3}(Y|X)]/\mathcal{D}\_{4n}, [Cn​(Y|X)−Rn,4​(Y|X)]/𝒟4​n[C\_{n}(Y|X)-R\_{n,4}(Y|X)]/\mathcal{D}\_{4n}. The result of Stage 2 implies that 1/mn↛∞1/m\_{n}\not\to\infty as n→∞n\to\infty. In addition, we also denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟4​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)→s1,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}\to s^{(u\_{1})}\_{1,j}, | 1mn​𝒟4​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)→s2,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})\to s\_{2,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟4​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​κi​jn)(v1)→s3,j(u1​v1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\kappa\_{ij}^{n})^{(v\_{1})}\to s^{(u\_{1}v\_{1})}\_{3,j}, | 1mn​𝒟4​n⋅∑i∈𝒱1,jωin​(Δ​τi​jn)2→s4,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\tau\_{ij}^{n})^{2}\to s\_{4,j}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟4​n⋅∑i∈𝒱1,jωin​(Δ​κi​jn)(u1)​(Δ​τi​jn)→s5,j(u1),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\Delta\kappa\_{ij}^{n})^{(u\_{1})}(\Delta\tau\_{ij}^{n})\to s^{(u\_{1})}\_{5,j}, | 1mn​𝒟4​n⋅(∑i∈𝒱1,jωin−ωj∗)→s0,j,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot\Big{(}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{)}\to s\_{0,j}, |  |

for all j∈[k1∗]j\in[k^{\*}\_{1}] and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟4​n⋅(Δ​βˇ0​in)→t0,i,\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot(\Delta\check{\beta}\_{0i}^{n})\to t\_{0,i}, | 1mn​𝒟4​n⋅(Δ​βˇ1​in)(u)→t1,i(u),\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot(\Delta\check{\beta}\_{1i}^{n})^{(u)}\to t^{(u)}\_{1,i}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 1mn​𝒟4​n⋅(Δ​ηˇin)(u2)→t2,i(u2),\displaystyle\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot(\Delta\check{\eta}\_{i}^{n})^{(u\_{2})}\to t^{(u\_{2})}\_{2,i}, | 1mn​𝒟4​n⋅(Δ​νˇin)→t3,i,\displaystyle\quad\frac{1}{m\_{n}\mathcal{D}\_{4n}}\cdot(\Delta\check{\nu}\_{i}^{n})\to t\_{3,i}, |  |

for all i∈[k2]i\in[k\_{2}]. Due to the result of Stage 2, at least one among the above limits is different from zero. Recall from equation ([37](#A4.E37 "In D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")) that we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(gG1n,G2n(⋅|X),gG1∗,Gˇ2(⋅|X))]/𝒟4​n→0,\displaystyle\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]/\mathcal{D}\_{4n}\to 0, |  |

Furthermore, according to the Fatou’s lemma, we get

|  |  |  |
| --- | --- | --- |
|  | limn→∞𝔼X[V(gG1n,G2n(⋅|X),gG1∗,Gˇ2(⋅|X))]mn​𝒟4​n≥∫lim infn→∞|gG1n,G2n(Y|X)−gG1∗,Gˇ2(Y|X)|2​mn​𝒟4​n​d​(X,Y).\displaystyle\lim\_{n\to\infty}\dfrac{\mathbb{E}\_{X}[V(g\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),g\_{G^{\*}\_{1},\check{G}\_{2}}(\cdot|X))]}{m\_{n}\mathcal{D}\_{4n}}\geq\int\liminf\_{n\to\infty}\dfrac{|g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},\check{G}\_{2}}(Y|X)|}{2m\_{n}\mathcal{D}\_{4n}}\mathrm{d}(X,Y). |  |

Then, it follows that [gG1n,G2n​(Y|X)−gG1∗,Gˇ2​(Y|X)]/[mn​𝒟4​n]→0[g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},\check{G}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{4n}]\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). As the input space is bounded and the parameter space is compact, the quantity ∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j}) is bounded. Therefore, we deduce

|  |  |  |
| --- | --- | --- |
|  | [∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)]​[gG1n,G2n​(Y|X)−gG1∗,Gˇ2​(Y|X)]/[mn​𝒟4​n]→0,\displaystyle\Big{[}\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})\Big{]}[g\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-g\_{G^{\*}\_{1},\check{G}\_{2}}(Y|X)]/[m\_{n}\mathcal{D}\_{4n}]\to 0, |  |

as n→∞n\to\infty. This result indicates

|  |  |  |
| --- | --- | --- |
|  | 12​[∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)]⋅qG1n​(Y|X)−qG1∗​(Y|X)mn​𝒟4​n+12​Qn​(Y|X)mn​𝒟4​n→0.\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})\Big{]}\cdot\dfrac{q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}+\frac{1}{2}\dfrac{Q\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}\to 0. |  |

as n→∞n\to\infty for almost surely (X,Y)(X,Y). From the decomposition of the terms qG1n​(Y|X)−qG1∗​(Y|X)q\_{G^{n}\_{1}}(Y|X)-q\_{G^{\*}\_{1}}(Y|X) and Qn​(Y|X)Q\_{n}(Y|X) in Stage 1, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​[∑j=1k2σ​((βˇ1​j)⊤​X+βˇ0​j)]⋅An,2​(Y|X)+An,1​(Y|X)+An,0​(Y|X)mn​𝒟4​n+12​Bn​(Y|X)−Cn​(Y|X)mn​𝒟4​n→0.\displaystyle\frac{1}{2}\Big{[}\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})\Big{]}\cdot\dfrac{A\_{n,2}(Y|X)+A\_{n,1}(Y|X)+A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}+\frac{1}{2}\dfrac{B\_{n}(Y|X)-C\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}\to 0. |  | (39) |

We have

|  |  |  |
| --- | --- | --- |
|  | limn→∞An,0​(Y|X)mn​𝒟4​n=∑j=1k1∗s0,j​π​(Y|h1​(X,κj∗),τj∗),\displaystyle\lim\_{n\to\infty}\frac{A\_{n,0}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}=\sum\_{j=1}^{k^{\*}\_{1}}s\_{0,j}\pi(Y|h\_{1}(X,\kappa^{\*}\_{j}),\tau^{\*}\_{j}), |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,1​(Y|X)mn​𝒟4​n=∑j∈[k1∗]:|𝒱1,j|=1[∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)∂π∂h1(Y|h1(X,κj∗),τj∗)\displaystyle\lim\_{n\to\infty}\frac{A\_{n,1}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|=1}\Big{[}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +12s2,j∂2π∂h12(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 284.52756pt+\frac{1}{2}s\_{2,j}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞An,2​(Y|X)mn​𝒟4​n=∑j∈[k1∗]:|𝒱1,j|>1[(∑u1=1d1s1,j(u1)∂h1∂κ(u1)(X,κj∗)+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}∂2h1∂κ(u1)​∂κ(v1)(X,κj∗))\displaystyle\lim\_{n\to\infty}\frac{A\_{n,2}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}=\sum\_{j\in[k^{\*}\_{1}]:|\mathcal{V}\_{1,j}|>1}\Big{[}\Big{(}\sum\_{u\_{1}=1}^{d\_{1}}s\_{1,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial^{2}h\_{1}}{\partial\kappa^{(u\_{1})}\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)} |  |
|  |  |  |
| --- | --- | --- |
|  | ×∂π∂h1​(Y|h1​(X,κj∗),τj∗)+(12​s2,j+∑u1,v1=1d1s3,j(u1​v1)1+1{u1=v1}​∂h1∂κ(u1)​(X,κj∗)​∂h1∂κ(v1)​(X,κj∗))​∂2π∂h12​(Y|h1​(X,κj∗),τj∗)\displaystyle\times\frac{\partial\pi}{\partial h\_{1}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\Big{(}\frac{1}{2}s\_{2,j}+\sum\_{u\_{1},v\_{1}=1}^{d\_{1}}\frac{s\_{3,j}^{(u\_{1}v\_{1})}}{1+1\_{\{u\_{1}=v\_{1}\}}}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\frac{\partial h\_{1}}{\partial\kappa^{(v\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{2}\pi}{\partial h\_{1}^{2}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}) |  |
|  |  |  |
| --- | --- | --- |
|  | +(12∑u1=1d1s5,j(u1)∂h1∂κ(u1)(X,κj∗))∂3π∂h13(Y|h1(X,κj∗),τj∗)+18s4,j∂4π∂h14(Y|h1(X,κj∗),τj∗)],\displaystyle\hskip 85.35826pt+\Big{(}\frac{1}{2}\sum\_{u\_{1}=1}^{d\_{1}}s\_{5,j}^{(u\_{1})}\frac{\partial h\_{1}}{\partial\kappa^{(u\_{1})}}(X,\kappa\_{j}^{\*})\Big{)}\frac{\partial^{3}\pi}{\partial h\_{1}^{3}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})+\frac{1}{8}s\_{4,j}\frac{\partial^{4}\pi}{\partial h\_{1}^{4}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*})\Big{]}, |  |

and

|  |  |  |
| --- | --- | --- |
|  | limn→∞Bn​(Y|X)mn​𝒟4​n=∑i=1k2[(∑u=1dt1,i(u)∂ψ∂β1(u)(X;βˇ1​i,βˇ0​i)+t0,i∂ψ∂β0(X;βˇ1​i,βˇ0​i))π(Y|h2(X,ηˇi),νˇi)\displaystyle\lim\_{n\to\infty}\frac{B\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}=\sum\_{i=1}^{k\_{2}}\Big{[}\Big{(}\sum\_{u=1}^{d}t\_{1,i}^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})+t\_{0,i}\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\Big{)}\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑u2=1d2t2,i(u2)​∂h2∂η(u2)​(X,ηˇi)​ψ​(X;βˇ1​i,βˇ0​i)​∂π∂h2​(Y|h2​(X,ηˇi),νˇi)\displaystyle\hskip 113.81102pt+\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\check{\eta}\_{i})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}) |  |
|  |  |  |
| --- | --- | --- |
|  | +12(Δνˇin)ψ(X;βˇ1​i,βˇ0​i)∂2π∂h22(Y|h2(X,ηˇi),νˇi)],\displaystyle\hskip 113.81102pt+\frac{1}{2}(\Delta\check{\nu}\_{i}^{n})\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i})\Big{]}, |  |
|  |  |  |
| --- | --- | --- |
|  | limn→∞Cn​(Y|X)mn​𝒟4​n=∑i=1k2[∑u=1dt1,i(u)​∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)+t0,i​∂ψ∂β0​(X;βˇ1​i,βˇ0​i)]​pGˇ2​(Y|X).\displaystyle\lim\_{n\to\infty}\frac{C\_{n}(Y|X)}{m\_{n}\mathcal{D}\_{4n}}=\sum\_{i=1}^{k\_{2}}\Big{[}\sum\_{u=1}^{d}t\_{1,i}^{(u)}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})+t\_{0,i}\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\Big{]}p\_{\check{G}\_{2}}(Y|X). |  |

Note that for almost every XX, the set

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | {[∑j=1k2σ((βˇ1​j)⊤X+βˇ0​j)]∂ρπ∂h1ρ(Y|h1(X,κj∗),τj∗):0≤ρ≤4,j∈[k1∗]}\displaystyle\Bigg{\{}\Big{[}\sum\_{j=1}^{k\_{2}}\sigma((\check{\beta}\_{1j})^{\top}X+\check{\beta}\_{0j})\Big{]}\frac{\partial^{\rho}\pi}{\partial h\_{1}^{\rho}}(Y|h\_{1}(X,\kappa\_{j}^{\*}),\tau\_{j}^{\*}):0\leq\rho\leq 4,\ j\in[k^{\*}\_{1}]\Bigg{\}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∪\displaystyle\cup~ | {∂ψ∂β1(u)(X;βˇ1​i,βˇ0​i)π(Y|h2(X,ηˇi),νˇi),∂ψ∂β0(X;βˇ1​i,βˇ0​i)π(Y|h2(X,ηˇi),νˇi),\displaystyle\Bigg{\{}\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)​pGˇ2​(Y|X),∂ψ∂β0​(X;βˇ1​i,βˇ0​i)​pGˇ2​(Y|X),\displaystyle\quad\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})p\_{\check{G}\_{2}}(Y|X),\ \frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})p\_{\check{G}\_{2}}(Y|X), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ψ(X;βˇ1​i,βˇ0​i)∂π∂h2(Y|h2(X,ηˇi),νˇi),ψ(X;βˇ1​i,βˇ0​i)∂2π∂h22(Y|h2(X,ηˇi),νˇi):u∈[d],i∈[k2]}\displaystyle\quad\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}),\ \psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}):u\in[d],\ i\in[k\_{2}]\Bigg{\}} |  |

is linearly independent w.r.t YY, implying that the coefficients of those terms in the limit in equation ([39](#A4.E39 "In D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")) are equal to zero.

Since the expert function h1h\_{1} is strongly identifiable, then by employing the same arguments as in the Stage 3 of Appendix [D.3](#A4.SS3 "D.3 Proof of Theorem 3 ‣ Appendix D Proof of Main Results"), we get s0,j=s1,j(u1)=s2,j=s3,j(u1​v1)=s4,j=s5,j(u1)=0s\_{0,j}=s\_{1,j}^{(u\_{1})}=s\_{2,j}=s\_{3,j}^{(u\_{1}v\_{1})}=s\_{4,j}=s\_{5,j}^{(u\_{1})}=0 for all j∈[k1∗]j\in[k^{\*}\_{1}] and u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}].
For i∈[k2]i\in[k\_{2}], by considering the coefficients of

* •

  ∂ψ∂β1(u)​(X;βˇ1​i,βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)\frac{\partial\psi}{\partial\beta\_{1}^{(u)}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), we get t1,i(u)=0t\_{1,i}^{(u)}=0 for all u∈[d]u\in[d];
* •

  ∂ψ∂β0​(X;βˇ1​i,βˇ0​i)​π​(Y|h2​(X,ηˇi),νˇi)\frac{\partial\psi}{\partial\beta\_{0}}(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\pi(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), we get t0,i=0t\_{0,i}=0;
* •

  ψ​(X;βˇ1​i,βˇ0​i)​∂π∂h2​(Y|h2​(X,ηˇi),νˇi)\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial\pi}{\partial h\_{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), we get ∑u2=1d2t2,i(u2)​∂h2∂η(u2)​(X,ηˇi)=0\sum\_{u\_{2}=1}^{d\_{2}}t\_{2,i}^{(u\_{2})}\frac{\partial h\_{2}}{\partial\eta^{(u\_{2})}}(X,\check{\eta}\_{i})=0. Since the expert function h2h\_{2} is weakly identifiable, we deduce t2,i(u2)=0t\_{2,i}^{(u\_{2})}=0 for all u2∈[d2]u\_{2}\in[d\_{2}];
* •

  ψ​(X;βˇ1​i,βˇ0​i)​∂2π∂h22​(Y|h2​(X,ηˇi),νˇi)\psi(X;\check{\beta}\_{1i},\check{\beta}\_{0i})\frac{\partial^{2}\pi}{\partial h\_{2}^{2}}(Y|h\_{2}(X,\check{\eta}\_{i}),\check{\nu}\_{i}), we get t3,i=0t\_{3,i}=0.

From the above results, it follows that (i) s0,j=s1,j(u1)=s2,j=s3,j(u1​v1)=s4,j=s5,j(u1)=0s\_{0,j}=s\_{1,j}^{(u\_{1})}=s\_{2,j}=s\_{3,j}^{(u\_{1}v\_{1})}=s\_{4,j}=s\_{5,j}^{(u\_{1})}=0 for all j∈[k1∗]j\in[k^{\*}\_{1}] and u1,v1∈[d1]u\_{1},v\_{1}\in[d\_{1}]; (ii) t0,i=t1,i(u)=t2,i(u2)=t3,i=0t\_{0,i}=t\_{1,i}^{(u)}=t\_{2,i}^{(u\_{2})}=t\_{3,i}=0 for all i∈[k2]i\in[k\_{2}], u∈[d]u\in[d] and u2∈[d2]u\_{2}\in[d\_{2}]. This contradicts to the fact that not all of them equal to zero. As a consequence, we obtain the local part in equation ([35](#A4.E35 "In D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results")). Hence, the proof is completed.

## Appendix E Proof of Auxiliary Results

### E.1 Proof of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy")

In this proof, we will leverage fundamental results on density estimation for M-estimators in [[72](#bib.bib72)]. Before streamlining our arguments, let us introduce some concepts from the empirical process theory adapted to the setting of the model ([1](#S2.E1 "In 2 On Shared Expert Strategy")).

Firstly, we denote by ℱk1,k2(Θ):={fG1,G2(Y|X):(G1,G2)∈𝒢k1,k2(Θ)}\mathcal{F}\_{k\_{1},k\_{2}}(\Theta):=\{f\_{G\_{1},G\_{2}}(Y|X):(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)\} the set of conditional density functions of interest. Furthermore, we also consider two variants of this set defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℱ~k1,k2​(Θ)\displaystyle\widetilde{\mathcal{F}}\_{k\_{1},k\_{2}}(\Theta) | :={12f(G1,G2)(Y|X)+12f(G1,G2)(Y|X):(G1∗,G2∗)∈𝒢k1,k2(Θ)},\displaystyle:=\Big{\{}\frac{1}{2}f\_{(G\_{1},G\_{2})}(Y|X)+\frac{1}{2}f\_{(G\_{1},G\_{2})}(Y|X):(G^{\*}\_{1},G^{\*}\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)\Big{\}}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ℱ~k1,k21/2​(Θ)\displaystyle\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta) | :={f~1/2:f~∈ℱ~k1,k2​(Θ)}.\displaystyle:=\{\tilde{f}^{1/2}:\tilde{f}\in\widetilde{\mathcal{F}}\_{k\_{1},k\_{2}}(\Theta)\}. |  |

For any δ>0\delta>0, the Hellinger ball centered around the the true density fG1∗,G2∗​(Y|X)f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) and intersected with ℱ~k1,k2​(Θ)\widetilde{\mathcal{F}}\_{k\_{1},k\_{2}}(\Theta) is defined as

|  |  |  |
| --- | --- | --- |
|  | ℱ~k1,k21/2​(Θ,δ):={p1/2∈ℱ~k1,k21/2​(Θ):h​(p,fG1∗,G2∗)≤δ}.\displaystyle\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,\delta):=\{p^{1/2}\in\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta):h(p,f\_{G^{\*}\_{1},G^{\*}\_{2}})\leq\delta\}. |  |

The size of the above Hellinger ball is determined by the quantity [[72](#bib.bib72)]

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥B(δ,ℱ~k1,k21/2(Θ,δ),∥⋅∥2):=∫δ2/213δHB1/2(t,ℱ~k1,k21/2(Θ,t),∥⋅∥2)dt∨δ,\displaystyle\mathcal{J}\_{B}(\delta,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,\delta),\|\cdot\|\_{2}):=\int\_{\delta^{2}/2^{13}}^{\delta}H\_{B}^{1/2}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t),\|\cdot\|\_{2})\mathrm{d}t\vee\delta, |  | (40) |

where HB(t,ℱ~k1,k21/2(Θ,t),∥⋅∥2)H\_{B}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t),\|\cdot\|\_{2}) stands for the bracketing entropy of ℱ~k1,k21/2​(Θ,t)\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t) under the L2​(m)L^{2}(m)-norm with mm being the Lebesgue measure, and t∨δ:=max⁡{t,δ}t\vee\delta:=\max\{t,\delta\}. Equipped with these notations, we are ready to present a standard result on density estimation for M-estimators in the following lemma:

###### Lemma 1 (Theorem 7.4, [[72](#bib.bib72)]).

Let δ∈(0,1)\delta\in(0,1) and take Ψ​(δ)≥𝒥B​(δ,ℱ~k1,k21/2​(Θ,δ))\Psi(\delta)\geq\mathcal{J}\_{B}(\delta,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,\delta)) such that Ψ​(δ)/δ2\Psi(\delta)/\delta^{2} is a non-increasing function of δ\delta. Then, for a universal constant cc and for some sequence (δn)(\delta\_{n}) satisfying n​δn2≥c​Ψ​(δn)\sqrt{n}\delta^{2}\_{n}\geq c\Psi(\delta\_{n}), the following holds for all δ≥δn\delta\geq\delta\_{n}:

|  |  |  |
| --- | --- | --- |
|  | ℙ(𝔼X[h(fG~1n,G~2n(⋅|X),fG1∗,G2∗(⋅|X))>δ])≤cexp(−n​δ2c2).\displaystyle\mathbb{P}\Big{(}\mathbb{E}\_{X}\Big{[}h(f\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))>\delta\Big{]}\Big{)}\leq c\exp\Big{(}-\frac{n\delta^{2}}{c^{2}}\Big{)}. |  |

Given the above result, we will provide below the proof for Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy").

###### Main proof of Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy").

Since ℱ~k1,k21/2​(Θ,t)⊂ℱ~k1,k21/2​(Θ)\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t)\subset\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta) for any t>0t>0, we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | HB(t,ℱ~k1,k21/2(Θ,t),∥⋅∥2)\displaystyle H\_{B}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t),\|\cdot\|\_{2}) | ≤HB(t,ℱ~k1,k21/2(Θ),∥⋅∥2)=HB(t/2,ℱ~k1,k2(Θ),h),\displaystyle\leq H\_{B}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{2})=H\_{B}(t/\sqrt{2},\widetilde{\mathcal{F}}\_{k\_{1},k\_{2}}(\Theta),h), |  | (41) |

where the last equality is due to the relationship between the Hellinger distance hh and the L2L^{2}-norm. Note that for any two mixing measure pairs (G1,G2)(G\_{1},G\_{2}) and (G1′,G2′)(G^{\prime}\_{1},G^{\prime}\_{2}), Lemma 4.2 in [[72](#bib.bib72)] shows that

|  |  |  |
| --- | --- | --- |
|  | h2​(12​fG1,G2+12​fG1∗,G2∗,12​fG1′,G2′+12​fG1∗,G2∗)≤12​h2​(fG1,G2,fG1′,G2′),\displaystyle h^{2}\Big{(}\frac{1}{2}f\_{G\_{1},G\_{2}}+\frac{1}{2}f\_{G^{\*}\_{1},G^{\*}\_{2}},\frac{1}{2}f\_{G^{\prime}\_{1},G^{\prime}\_{2}}+\frac{1}{2}f\_{G^{\*}\_{1},G^{\*}\_{2}}\Big{)}\leq\frac{1}{2}h^{2}(f\_{G\_{1},G\_{2}},f\_{G^{\prime}\_{1},G^{\prime}\_{2}}), |  |

which yields that HB​(t/2,ℱ~k1,k2​(Θ),h)≤HB​(t,ℱk1,k2​(Θ),h)H\_{B}(t/\sqrt{2},\widetilde{\mathcal{F}}\_{k\_{1},k\_{2}}(\Theta),h)\leq H\_{B}(t,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),h). This result together with equation ([41](#A5.E41 "In E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results")) implies that

|  |  |  |
| --- | --- | --- |
|  | HB(t,ℱ~k1,k21/2(Θ,t),∥⋅∥2)≤HB(t,ℱk1,k2(Θ),h).\displaystyle H\_{B}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t),\|\cdot\|\_{2})\leq H\_{B}(t,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),h). |  |

From the definition of the Hellinger ball size in equation ([40](#A5.E40 "In E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results")), we have that

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒥B(δ,ℱ~k1,k21/2(Θ,δ),∥⋅∥2)\displaystyle\mathcal{J}\_{B}(\delta,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,\delta),\|\cdot\|\_{2}) | =∫δ2/213δHB1/2(t,ℱ~k1,k21/2(Θ,t),∥⋅∥2)dt∨δ\displaystyle=\int\_{\delta^{2}/2^{13}}^{\delta}H\_{B}^{1/2}(t,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,t),\|\cdot\|\_{2})\mathrm{d}t\vee\delta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∫δ2/213δHB1/2​(t,ℱk1,k2​(Θ),h)​dt∨δ\displaystyle\leq\int\_{\delta^{2}/2^{13}}^{\delta}H\_{B}^{1/2}(t,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),h)\mathrm{d}t\vee\delta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≲∫δ2/213δ[log⁡(1/t)]1/2​dt∨δ,\displaystyle\lesssim\int\_{\delta^{2}/2^{13}}^{\delta}[\log(1/t)]^{1/2}\mathrm{d}t\vee\delta, |  |

where the last inequality is due to Lemma [2](#Thmlemma2 "Lemma 2. ‣ E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results") below. Let Ψ​(δ):=δ​log⁡(1/δ)\Psi(\delta):=\delta\sqrt{\log(1/\delta)}, it can be verified that Ψ​(δ)/δ2\Psi(\delta)/\delta^{2} is a non-increasing function of δ\delta. Furthermore, the above result indicates that Ψ(δ)≥𝒥B(δ,ℱ~k1,k21/2(Θ,δ),∥⋅∥2)\Psi(\delta)\geq\mathcal{J}\_{B}(\delta,\widetilde{\mathcal{F}}^{1/2}\_{k\_{1},k\_{2}}(\Theta,\delta),\|\cdot\|\_{2}). By considering the sequence (δn)(\delta\_{n}) defined as δn:=log⁡(n)/n\delta\_{n}:=\sqrt{\log(n)/n}, we have n​δn2≥c​Ψ​(δn)\sqrt{n}\delta\_{n}^{2}\geq c\Psi(\delta\_{n}) for some universal constant c>0c>0. Then, according to Lemma [1](#Thmlemma1 "Lemma 1 (Theorem 7.4, [72]). ‣ E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results"), we get

|  |  |  |
| --- | --- | --- |
|  | ℙ(𝔼X[h(fG~1n,G~2n(⋅|X),fG1∗,G2∗(⋅|X))>Clog⁡(n)/n])≲exp(−clog(n)),\displaystyle\mathbb{P}\Big{(}\mathbb{E}\_{X}\Big{[}h(f\_{\widetilde{G}^{n}\_{1},\widetilde{G}^{n}\_{2}}(\cdot|X),f\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))>C\sqrt{\log(n)/n}\Big{]}\Big{)}\lesssim\exp(-c\log(n)), |  |

for some universal constant CC depending on Θ\Theta.
∎

###### Lemma 2.

The following holds for any 0<ϵ<1/20<\epsilon<1/2:

|  |  |  |
| --- | --- | --- |
|  | HB​(ϵ,ℱk1,k2​(Θ),h)≲log⁡(1/ϵ).\displaystyle H\_{B}(\epsilon,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),h)\lesssim\log(1/\epsilon). |  |

###### Proof of Lemma [2](#Thmlemma2 "Lemma 2. ‣ E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results").

Recall that for any mixing measure pair (G1,G2)(G\_{1},G\_{2}), we have

|  |  |  |
| --- | --- | --- |
|  | fG1,G2​(Y|X)=12​∑i=1k1ωi​π​(Y|h1​(X,κi),τi)+12​∑i=1k2exp⁡((β1​i)⊤​X+β0​i)∑j=1k2exp⁡((β1​j)⊤​X+β0​j)⋅π​(Y|h2​(X,ηi),νi).\displaystyle f\_{G\_{1},G\_{2}}(Y|X)=\frac{1}{2}\sum\_{i=1}^{k\_{1}}\omega\_{i}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i})+\frac{1}{2}\sum\_{i=1}^{k\_{2}}\frac{\exp((\beta\_{1i})^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp((\beta\_{1j})^{\top}X+\beta\_{0j})}\cdot\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i}). |  |

Firstly, we will establish upper bounds for the Gaussian densities π​(Y|h1​(X,κ),τ)\pi(Y|h\_{1}(X,\kappa),\tau) and π​(Y|h2​(X,η),ν)\pi(Y|h\_{2}(X,\eta),\nu), respectively. Indeed, since the expert function h1h\_{1} is bounded and the parameter space is compact, we have |h1​(X,κ)|≤M1|h\_{1}(X,\kappa)|\leq M\_{1} for all X∈𝒳X\in\mathcal{X} for some constant M1>0M\_{1}>0, and ℓ1≤τ≤u1\ell\_{1}\leq\tau\leq u\_{1} for some ℓ1,u1>0\ell\_{1},u\_{1}>0. Therefore, for |Y|≥2​M1|Y|\geq 2M\_{1}, since (Y−h1​(X,κ))22​τ≥Y28​u1\frac{(Y-h\_{1}(X,\kappa))^{2}}{2\tau}\geq\frac{Y^{2}}{8u\_{1}} for all X∈𝒳X\in\mathcal{X}, we have

|  |  |  |
| --- | --- | --- |
|  | π​(Y|h1​(X,κ),τ)=12​𝝅​τ​exp⁡(−(Y−h1​(X,κ))22​τ)≤12​𝝅​ℓ1​exp⁡(−Y28​u1).\displaystyle\pi(Y|h\_{1}(X,\kappa),\tau)=\frac{1}{\sqrt{2\bm{\pi}\tau}}\exp\Big{(}-\frac{(Y-h\_{1}(X,\kappa))^{2}}{2\tau}\Big{)}\leq\frac{1}{\sqrt{2\bm{\pi}\ell\_{1}}}\exp\Big{(}-\frac{Y^{2}}{8u\_{1}}\Big{)}. |  |

Next, for |Y|<2​M1|Y|<2M\_{1}, it follows that

|  |  |  |
| --- | --- | --- |
|  | π​(Y|h1​(X,κ),τ)=12​𝝅​τ​exp⁡(−(Y−h1​(X,κ))22​τ)≤12​𝝅​τ≤12​𝝅​ℓ1.\displaystyle\pi(Y|h\_{1}(X,\kappa),\tau)=\frac{1}{\sqrt{2\bm{\pi}\tau}}\exp\Big{(}-\frac{(Y-h\_{1}(X,\kappa))^{2}}{2\tau}\Big{)}\leq\frac{1}{\sqrt{2\bm{\pi}\tau}}\leq\frac{1}{\sqrt{2\bm{\pi}\ell\_{1}}}. |  |

Combine the above results together, we deduce π​(Y|h1​(X,κ),τ)≤E1​(Y|X)\pi(Y|h\_{1}(X,\kappa),\tau)\leq E\_{1}(Y|X) for all (X,Y)(X,Y) where

|  |  |  |
| --- | --- | --- |
|  | E1​(Y|X):={12​𝝅​ℓ1​exp⁡(−Y28​u1),for ​|Y|≥2​M112​𝝅​ℓ1,for ​|Y|<2​M1.\displaystyle E\_{1}(Y|X):=\begin{cases}\frac{1}{\sqrt{2\bm{\pi}\ell\_{1}}}\exp\Big{(}-\frac{Y^{2}}{8u\_{1}}\Big{)},\quad\text{for }|Y|\geq 2M\_{1}\\ \frac{1}{\sqrt{2\bm{\pi}\ell\_{1}}},\hskip 73.97733pt\text{for }|Y|<2M\_{1}.\end{cases} |  |

By arguing in similar fashion based on the assumptions that |h2​(X,η)|≤M2|h\_{2}(X,\eta)|\leq M\_{2} for all X∈𝒳X\in\mathcal{X} for some constant M2>0M\_{2}>0, and ℓ2≤ν≤u2\ell\_{2}\leq\nu\leq u\_{2} for some ℓ2,u2>0\ell\_{2},u\_{2}>0, we also get π​(Y|h2​(X,η),ν)≤E2​(Y|X)\pi(Y|h\_{2}(X,\eta),\nu)\leq E\_{2}(Y|X), where

|  |  |  |
| --- | --- | --- |
|  | E2​(Y|X):={12​𝝅​ℓ2​exp⁡(−Y28​u2),for ​|Y|≥2​M212​𝝅​ℓ2,for ​|Y|<2​M2.\displaystyle E\_{2}(Y|X):=\begin{cases}\frac{1}{\sqrt{2\bm{\pi}\ell\_{2}}}\exp\Big{(}-\frac{Y^{2}}{8u\_{2}}\Big{)},\quad\text{for }|Y|\geq 2M\_{2}\\ \frac{1}{\sqrt{2\bm{\pi}\ell\_{2}}},\hskip 73.97733pt\text{for }|Y|<2M\_{2}.\end{cases} |  |

Now, let λ≤ϵ\lambda\leq\epsilon be some constant that we will choose later, we denote p1,p2,…,pNp\_{1},p\_{2},\ldots,p\_{N} as an λ\lambda-cover of the set ℱk1,k2​(Θ)\mathcal{F}\_{k\_{1},k\_{2}}(\Theta), where N:=N(λ,ℱk1,k2(Θ),∥⋅∥∞)N:=N(\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{\infty}) stands for the λ\lambda-covering number of the set ℱk1,k2​(Θ)\mathcal{F}\_{k\_{1},k\_{2}}(\Theta) under the L∞L^{\infty}-norm. Then, we take into account the brackets [piL,piU][p^{L}\_{i},p^{U}\_{i}] given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | piL​(Y|X)\displaystyle p^{L}\_{i}(Y|X) | :=max⁡{pi​(Y|X)−λ,0},\displaystyle:=\max\{p\_{i}(Y|X)-\lambda,0\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | piU​(Y|X)\displaystyle p^{U}\_{i}(Y|X) | :=max⁡{pi​(Y|X)+λ,E​(Y|X)},\displaystyle:=\max\{p\_{i}(Y|X)+\lambda,E(Y|X)\}, |  |

for all i∈[N]i\in[N], where E​(Y|X):=12​E1​(Y|X)+12​E2​(Y|X)E(Y|X):=\frac{1}{2}E\_{1}(Y|X)+\frac{1}{2}E\_{2}(Y|X). It can be justified that ℱk1,k2​(Θ)⊆∪i=1N[piL,piU]\mathcal{F}\_{k\_{1},k\_{2}}(\Theta)\subseteq\cup\_{i=1}^{N}[p^{L}\_{i},p^{U}\_{i}] and piU​(Y|X)−piL​(Y|X)≤min⁡{2​λ,E​(Y|X)}p^{U}\_{i}(Y|X)-p^{L}\_{i}(Y|X)\leq\min\{2\lambda,E(Y|X)\}. Furthermore, we have

|  |  |  |
| --- | --- | --- |
|  | ‖piU−piL‖2=(∫[piU​(Y|X)−piL​(Y|X)]2​d​(X,Y))1/2≤2​λ.\displaystyle\|p^{U}\_{i}-p^{L}\_{i}\|\_{2}=\Big{(}\int[p^{U}\_{i}(Y|X)-p^{L}\_{i}(Y|X)]^{2}\mathrm{d}(X,Y)\Big{)}^{1/2}\leq 2\lambda. |  |

By definition of the bracketing entropy, we get

|  |  |  |
| --- | --- | --- |
|  | HB(2λ,ℱk1,k2(Θ),∥⋅∥2)≤logN=logN(λ,ℱk1,k2(Θ),∥⋅∥∞).\displaystyle H\_{B}(2\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{2})\leq\log N=\log N(\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{\infty}). |  |

Thus, we need to derive an upper bound for the covering number N(λ,ℱk1,k2(Θ),∥⋅∥∞)N(\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{\infty}). Let us denote Δ:=Δ1×Δ2\Delta:=\Delta\_{1}\times\Delta\_{2} and Ω:=Ω1×Ω2\Omega:=\Omega\_{1}\times\Omega\_{2}, where

|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ1\displaystyle\Delta\_{1} | :={ωi∈ℝ+:(ω,κ,τ)∈Θ1},\displaystyle:=\{\omega\_{i}\in\mathbb{R}\_{+}:(\omega,\kappa,\tau)\in\Theta\_{1}\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Δ2\displaystyle\Delta\_{2} | :={(κ,τ)∈ℝd1×ℝ+:(ω,κ,τ)∈Θ1},\displaystyle:=\{(\kappa,\tau)\in\mathbb{R}^{d\_{1}}\times\mathbb{R}\_{+}:(\omega,\kappa,\tau)\in\Theta\_{1}\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Ω1\displaystyle\Omega\_{1} | :={(β0,β1)∈ℝ×ℝd:(β0,β1,η,ν)∈Θ2},\displaystyle:=\{(\beta\_{0},\beta\_{1})\in\mathbb{R}\times\mathbb{R}^{d}:(\beta\_{0},\beta\_{1},\eta,\nu)\in\Theta\_{2}\}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Ω2\displaystyle\Omega\_{2} | :={(η,ν)∈ℝd2×ℝ+:(β0,β1,η,ν)∈Θ2}.\displaystyle:=\{(\eta,\nu)\in\mathbb{R}^{d\_{2}}\times\mathbb{R}\_{+}:(\beta\_{0},\beta\_{1},\eta,\nu)\in\Theta\_{2}\}. |  |

Since Θ1\Theta\_{1} and Θ2\Theta\_{2} are compact, the sets Δ1,Δ2\Delta\_{1},\Delta\_{2} and Ω1,Ω2\Omega\_{1},\Omega\_{2} are also compact. Thus, there exist λ\lambda-covers Δ1,λ,Δ2,λ\Delta\_{1,\lambda},\Delta\_{2,\lambda} and Ω1,λ,Ω2,λ\Omega\_{1,\lambda},\Omega\_{2,\lambda} for those sets, respectively. Moreover, the cardinalities of those λ\lambda-covers are bounded as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | |Δ1,λ|≤𝒪​(λ−k1),\displaystyle|\Delta\_{1,\lambda}|\leq\mathcal{O}(\lambda^{-k\_{1}}), | |Δ2,λ|≤𝒪​(λ−(d1+1)​k1),\displaystyle\quad|\Delta\_{2,\lambda}|\leq\mathcal{O}(\lambda^{-(d\_{1}+1)k\_{1}}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | |Ω1,λ|≤𝒪​(λ−(d+1)​k2),\displaystyle|\Omega\_{1,\lambda}|\leq\mathcal{O}(\lambda^{-(d+1)k\_{2}}), | |Ω2,λ|≤𝒪​(λ−(d2+1)​k2).\displaystyle\quad|\Omega\_{2,\lambda}|\leq\mathcal{O}(\lambda^{-(d\_{2}+1)k\_{2}}). |  |

For each pair of mixing measure (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta), we consider two other mixing measure pairs (G1′,G2′)(G^{\prime}\_{1},G^{\prime}\_{2}) and (G¯1,G¯2)(\overline{G}\_{1},\overline{G}\_{2}) given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | G1′:=∑i=1k1ωi​δ(κ¯i,τ¯i),\displaystyle G^{\prime}\_{1}:=\sum\_{i=1}^{k\_{1}}\omega\_{i}\delta\_{(\bar{\kappa}\_{i},\bar{\tau}\_{i})}, | G2′:=∑i=1k2ω¯i​δ(κ¯i,τ¯i),\displaystyle\hskip 56.9055ptG^{\prime}\_{2}:=\sum\_{i=1}^{k\_{2}}\bar{\omega}\_{i}\delta\_{(\bar{\kappa}\_{i},\bar{\tau}\_{i})}, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | G¯1:=∑i=1k2exp⁡(β0​i)​δ(β1​i,η¯i,τ¯i),\displaystyle\overline{G}\_{1}:=\sum\_{i=1}^{k\_{2}}\exp(\beta\_{0i})\delta\_{(\beta\_{1i},\bar{\eta}\_{i},\bar{\tau}\_{i})}, | G¯2:=∑i=1k2exp⁡(β¯0​i)​δ(β¯1​i,η¯i,ν¯i).\displaystyle\hskip 56.9055pt\overline{G}\_{2}:=\sum\_{i=1}^{k\_{2}}\exp(\bar{\beta}\_{0i})\delta\_{(\bar{\beta}\_{1i},\bar{\eta}\_{i},\bar{\nu}\_{i})}. |  |

Above, ω¯i∈Δ1,λ\bar{\omega}\_{i}\in\Delta\_{1,\lambda} is the closest point to ωi\omega\_{i} in that set, (κ¯i,τ¯i)∈Δ2,λ(\bar{\kappa}\_{i},\bar{\tau}\_{i})\in\Delta\_{2,\lambda} is the closest point to (κi,τi)(\kappa\_{i},\tau\_{i}) in that set, (β¯0​i,β¯1​i)∈Ω1,λ(\bar{\beta}\_{0i},\bar{\beta}\_{1i})\in\Omega\_{1,\lambda} is the closest point to (β0​i,β1​i)(\beta\_{0i},\beta\_{1i}) in that set, (η¯i,ν¯i)∈Ω2,λ(\bar{\eta}\_{i},\bar{\nu}\_{i})\in\Omega\_{2,\lambda} is the closest point to (ηi,νi)(\eta\_{i},\nu\_{i}) in that set. Subsequently, we aim to upper bound the term ‖fG1,G2−fG¯1,G¯2‖∞\|f\_{G\_{1},G\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}. By the triangle inequality, we have

|  |  |  |
| --- | --- | --- |
|  | ‖fG1,G2−fG¯1,G¯2‖∞≤‖fG1,G2−fG1′,G2′‖∞+‖fG1′,G2′−fG¯1,G¯2‖∞.\displaystyle\|f\_{G\_{1},G\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}\leq\|f\_{G\_{1},G\_{2}}-f\_{G^{\prime}\_{1},G^{\prime}\_{2}}\|\_{\infty}+\|f\_{G^{\prime}\_{1},G^{\prime}\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}. |  |

We aim to upper bound the two terms in the above right hand sides, respectively. For ease of presentation, for any mixing measure pair (G1,G2)(G\_{1},G\_{2}), we denote

|  |  |  |  |
| --- | --- | --- | --- |
|  | qG1​(Y|X)\displaystyle q\_{G\_{1}}(Y|X) | :=∑i=1k1ωi​π​(Y|h1​(X,κi),τi),\displaystyle:=\sum\_{i=1}^{k\_{1}}\omega\_{i}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | pG2​(Y|X)\displaystyle p\_{G\_{2}}(Y|X) | :=∑i=1k2exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)​π​(Y|h2​(X,ηi),νi).\displaystyle:=\sum\_{i=1}^{k\_{2}}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i}). |  |

We start with bounding the term ‖fG1,G2−fG1′,G2′‖∞\|f\_{G\_{1},G\_{2}}-f\_{G^{\prime}\_{1},G^{\prime}\_{2}}\|\_{\infty} as follows:

|  |  |  |
| --- | --- | --- |
|  | ‖fG1,G2−fG1′,G2′‖∞≤12​‖qG1−qG1′‖∞+12​‖pG2−pG2′‖∞.\displaystyle\|f\_{G\_{1},G\_{2}}-f\_{G^{\prime}\_{1},G^{\prime}\_{2}}\|\_{\infty}\leq\frac{1}{2}\|q\_{G\_{1}}-q\_{G^{\prime}\_{1}}\|\_{\infty}+\frac{1}{2}\|p\_{G\_{2}}-p\_{G^{\prime}\_{2}}\|\_{\infty}. |  |

In particular, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖qG1−qG1′‖∞\displaystyle\|q\_{G\_{1}}-q\_{G^{\prime}\_{1}}\|\_{\infty} | =sup(X,Y)∈𝒳×𝒴|∑i=1k1ωi[π(Y|h1(X,κi),τi)−π(Y|h1(X,κ¯i),τ¯i)]|\displaystyle=\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}\Bigg{|}\sum\_{i=1}^{k\_{1}}\omega\_{i}\Big{[}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i})-\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})\Big{]}\Bigg{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑i=1k1ωisup(X,Y)∈𝒳×𝒴|π(Y|h1(X,κi),τi)−π(Y|h1(X,κ¯i),τ¯i)|\displaystyle\leq\sum\_{i=1}^{k\_{1}}\omega\_{i}\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~\Big{|}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i})-\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≲∑i=1k1ωi​(‖κi−κ¯i‖+|τi−τ¯i|)\displaystyle\lesssim\sum\_{i=1}^{k\_{1}}\omega\_{i}(\|\kappa\_{i}-\bar{\kappa}\_{i}\|+|\tau\_{i}-\bar{\tau}\_{i}|) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑i=1k1ωi​(λ+λ)=2​λ≲λ,\displaystyle\leq\sum\_{i=1}^{k\_{1}}\omega\_{i}(\lambda+\lambda)=2\lambda\lesssim\lambda, |  |

and

|  |  |  |
| --- | --- | --- |
|  | ‖pG2−pG2′‖∞\displaystyle\|p\_{G\_{2}}-p\_{G^{\prime}\_{2}}\|\_{\infty} |  |
|  |  |  |
| --- | --- | --- |
|  | =sup(X,Y)∈𝒳×𝒴|∑i=1k2exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)[π(Y|h2(X,ηi),νi)−π(Y|h1(X,η¯i),ν¯i)]|\displaystyle=\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}\Bigg{|}\sum\_{i=1}^{k\_{2}}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}\Big{[}\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i})-\pi(Y|h\_{1}(X,\bar{\eta}\_{i}),\bar{\nu}\_{i})\Big{]}\Bigg{|} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∑i=1k2sup(X,Y)∈𝒳×𝒴exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)|π(Y|h2(X,ηi),νi)−π(Y|h1(X,η¯i),ν¯i)|\displaystyle\leq\sum\_{i=1}^{k\_{2}}\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}\Big{|}\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i})-\pi(Y|h\_{1}(X,\bar{\eta}\_{i}),\bar{\nu}\_{i})\Big{|} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∑i=1k2sup(X,Y)∈𝒳×𝒴|π(Y|h2(X,ηi),νi)−π(Y|h1(X,η¯i),ν¯i)|\displaystyle\leq\sum\_{i=1}^{k\_{2}}\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~\Big{|}\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i})-\pi(Y|h\_{1}(X,\bar{\eta}\_{i}),\bar{\nu}\_{i})\Big{|} |  |
|  |  |  |
| --- | --- | --- |
|  | ≲∑i=1k2(‖ηi−η¯i‖+|νi−ν¯i|)≤∑i=1k2(λ+λ)≲λ,\displaystyle\lesssim\sum\_{i=1}^{k\_{2}}(\|\eta\_{i}-\bar{\eta}\_{i}\|+|\nu\_{i}-\bar{\nu}\_{i}|)\leq\sum\_{i=1}^{k\_{2}}(\lambda+\lambda)\lesssim\lambda, |  |

which implies that

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖fG1,G2−fG1′,G2′‖∞≲12​λ+12​λ=λ.\displaystyle\|f\_{G\_{1},G\_{2}}-f\_{G^{\prime}\_{1},G^{\prime}\_{2}}\|\_{\infty}\lesssim\frac{1}{2}\lambda+\frac{1}{2}\lambda=\lambda. |  | (42) |

Next, we continue with bounding the term ‖fG1′,G2′−fG¯1,G¯2‖∞\|f\_{G^{\prime}\_{1},G^{\prime}\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty} as

|  |  |  |
| --- | --- | --- |
|  | ‖fG1′,G2′−fG¯1,G¯2‖∞≤12​‖qG1′−qG¯1‖∞+12​‖pG2′−pG¯2‖∞.\displaystyle\|f\_{G^{\prime}\_{1},G^{\prime}\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}\leq\frac{1}{2}\|q\_{G^{\prime}\_{1}}-q\_{\overline{G}\_{1}}\|\_{\infty}+\frac{1}{2}\|p\_{G^{\prime}\_{2}}-p\_{\overline{G}\_{2}}\|\_{\infty}. |  |

By looking into each term in the above right hand side, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖qG1′−qG¯1‖∞\displaystyle\|q\_{G^{\prime}\_{1}}-q\_{\overline{G}\_{1}}\|\_{\infty} | =sup(X,Y)∈𝒳×𝒴|∑i=1k1[ωi−ω¯i]π(Y|h1(X,κ¯i),τ¯i)|\displaystyle=\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}\Bigg{|}\sum\_{i=1}^{k\_{1}}[\omega\_{i}-\bar{\omega}\_{i}]\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})\Bigg{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤∑i=1k1|ωi−ω¯i|sup(X,Y)∈𝒳×𝒴|π(Y|h1(X,κ¯i),τ¯i)|\displaystyle\leq\sum\_{i=1}^{k\_{1}}|\omega\_{i}-\bar{\omega}\_{i}|\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~|\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≲∑i=1k1|ωi−ω¯i|≤∑i=1k1λ≲λ,\displaystyle\lesssim\sum\_{i=1}^{k\_{1}}|\omega\_{i}-\bar{\omega}\_{i}|\leq\sum\_{i=1}^{k\_{1}}\lambda\lesssim\lambda, |  |

and

|  |  |  |
| --- | --- | --- |
|  | ‖pG2′−pG¯2‖∞\displaystyle\|p\_{G^{\prime}\_{2}}-p\_{\overline{G}\_{2}}\|\_{\infty} |  |
|  |  |  |
| --- | --- | --- |
|  | =sup(X,Y)∈𝒳×𝒴|∑i=1k2[exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)−exp⁡(β¯1​i⊤​X+β¯0​i)∑j=1k2exp⁡(β¯1​j⊤​X+β¯0​j)]π(Y|h1(X,κ¯i),τ¯i)|\displaystyle=\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}\Bigg{|}\sum\_{i=1}^{k\_{2}}\Bigg{[}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}-\frac{\exp(\bar{\beta}\_{1i}^{\top}X+\bar{\beta}\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\bar{\beta}\_{1j}^{\top}X+\bar{\beta}\_{0j})}\Bigg{]}\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})\Bigg{|} |  |
|  |  |  |
| --- | --- | --- |
|  | ≤∑i=1k2sup(X,Y)∈𝒳×𝒴|exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)−exp⁡(β¯1​i⊤​X+β¯0​i)∑j=1k2exp⁡(β¯1​j⊤​X+β¯0​j)|⋅|π(Y|h1(X,κ¯i),τ¯i)|\displaystyle\leq\sum\_{i=1}^{k\_{2}}\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~\Bigg{|}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}-\frac{\exp(\bar{\beta}\_{1i}^{\top}X+\bar{\beta}\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\bar{\beta}\_{1j}^{\top}X+\bar{\beta}\_{0j})}\Bigg{|}\cdot|\pi(Y|h\_{1}(X,\bar{\kappa}\_{i}),\bar{\tau}\_{i})| |  |
|  |  |  |
| --- | --- | --- |
|  | ≲∑i=1k2sup(X,Y)∈𝒳×𝒴|exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)−exp⁡(β¯1​i⊤​X+β¯0​i)∑j=1k2exp⁡(β¯1​j⊤​X+β¯0​j)|\displaystyle\lesssim\sum\_{i=1}^{k\_{2}}\sup\_{(X,Y)\in\mathcal{X}\times\mathcal{Y}}~\Bigg{|}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}-\frac{\exp(\bar{\beta}\_{1i}^{\top}X+\bar{\beta}\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\bar{\beta}\_{1j}^{\top}X+\bar{\beta}\_{0j})}\Bigg{|} |  |
|  |  |  |
| --- | --- | --- |
|  | ≲∑i=1k2supX∈𝒳(‖β1​i−β¯1​i‖⋅‖X‖+‖β0​i−β¯0​i‖)\displaystyle\lesssim\sum\_{i=1}^{k\_{2}}\sup\_{X\in\mathcal{X}}~\Big{(}\|\beta\_{1i}-\bar{\beta}\_{1i}\|\cdot\|X\|+\|\beta\_{0i}-\bar{\beta}\_{0i}\|\Big{)} |  |
|  |  |  |
| --- | --- | --- |
|  | ≲∑i=1k2(λ⋅supX∈𝒳‖X‖+λ)≲λ.\displaystyle\lesssim\sum\_{i=1}^{k\_{2}}\Big{(}\lambda\cdot\sup\_{X\in\mathcal{X}}~\|X\|+\lambda\Big{)}\lesssim\lambda. |  |

Putting these bounds together, we deduce

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖fG1′,G2′−fG¯1,G¯2‖∞≲12​λ+12​λ=λ.\displaystyle\|f\_{G^{\prime}\_{1},G^{\prime}\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}\lesssim\frac{1}{2}\lambda+\frac{1}{2}\lambda=\lambda. |  | (43) |

From equations ([42](#A5.E42 "In E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results")) and ([43](#A5.E43 "In E.1 Proof of Proposition 1 ‣ Appendix E Proof of Auxiliary Results")), we obtain

|  |  |  |
| --- | --- | --- |
|  | ‖fG1,G2−fG¯1,G¯2‖∞≤λ+λ≲λ.\displaystyle\|f\_{G\_{1},G\_{2}}-f\_{\overline{G}\_{1},\overline{G}\_{2}}\|\_{\infty}\leq\lambda+\lambda\lesssim\lambda. |  |

By definition of the covering number, we get

|  |  |  |  |
| --- | --- | --- | --- |
|  | N(λ,ℱk1,k2(Θ),∥⋅∥∞)\displaystyle N(\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{\infty}) | ≤|Δ1,λ|⋅|Δ2,λ|⋅|Ω1,λ|⋅|Ω2,λ|\displaystyle\leq|\Delta\_{1,\lambda}|\cdot|\Delta\_{2,\lambda}|\cdot|\Omega\_{1,\lambda}|\cdot|\Omega\_{2,\lambda}| |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤𝒪​(λ−k1)⋅𝒪​(λ−(d1+1)​k1)⋅𝒪​(λ−(d+1)​k2)⋅𝒪​(λ−(d2+1)​k2)\displaystyle\leq\mathcal{O}(\lambda^{-k\_{1}})\cdot\mathcal{O}(\lambda^{-(d\_{1}+1)k\_{1}})\cdot\mathcal{O}(\lambda^{-(d+1)k\_{2}})\cdot\mathcal{O}(\lambda^{-(d\_{2}+1)k\_{2}}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≤𝒪​(λ−(d1+2)​k1−(d2+d+2)​k2).\displaystyle\leq\mathcal{O}(\lambda^{-(d\_{1}+2)k\_{1}-(d\_{2}+d+2)k\_{2}}). |  |

As a result, we deduce

|  |  |  |
| --- | --- | --- |
|  | HB(2λ,ℱk1,k2(Θ),∥⋅∥2)≤logN(λ,ℱk1,k2(Θ),∥⋅∥∞)≲log(1/λ).\displaystyle H\_{B}(2\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{2})\leq\log N(\lambda,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{\infty})\lesssim\log(1/\lambda). |  |

Let λ=ϵ/2\lambda=\epsilon/2, we achieve the desired result that HB(ϵ,ℱk1,k2(Θ),∥⋅∥2)≲log(1/ϵ)H\_{B}(\epsilon,\mathcal{F}\_{k\_{1},k\_{2}}(\Theta),\|\cdot\|\_{2})\lesssim\log(1/\epsilon). Hence, the proof is completed.
∎

### E.2 Identifiability of DeepSeekMoE

###### Proposition 5 (Identifiability).

For any pair of mixing measures (G1,G2)(G\_{1},G\_{2}), if the equation fG1,G2​(Y|X)=fG1∗,G2∗​(Y|X)f\_{G\_{1},G\_{2}}(Y|X)=f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) holds for almost surely (X,Y)(X,Y), then we obtain (G1,G2)≡(G1∗,G2∗)(G\_{1},G\_{2})\equiv(G^{\*}\_{1},G^{\*}\_{2}).

###### Proof of Proposition [5](#Thmproposition5 "Proposition 5 (Identifiability). ‣ E.2 Identifiability of DeepSeekMoE ‣ Appendix E Proof of Auxiliary Results").

First of all, we expand the equation fG1,G2​(Y|X)=fG1∗,G2∗​(Y|X)f\_{G\_{1},G\_{2}}(Y|X)=f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) for almost surely (X,Y)(X,Y) as follows:

|  |  |  |
| --- | --- | --- |
|  | 12​∑i=1k1ωi​π​(Y|h1​(X,κi),τi)+12​∑i=1k2exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)​π​(Y|h2​(X,ηi),νi)\displaystyle\frac{1}{2}\sum\_{i=1}^{k\_{1}}\omega\_{i}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i})+\frac{1}{2}\sum\_{i=1}^{k\_{2}}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =12​∑i=1k1∗ωi∗​π​(Y|h1​(X,κi∗),τi∗)+12​∑i=1k2∗exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)⋅π​(Y|h2​(X,ηi∗),νi∗).\displaystyle=\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(Y|h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i})+\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{2}}\frac{\exp((\beta\_{1i}^{\*})^{\top}X+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*})}\cdot\pi(Y|h\_{2}(X,\eta^{\*}\_{i}),\nu\_{i}^{\*}). |  | (44) |

Since the location-scale Gaussian mixtures are identifiable [[70](#bib.bib70)], the above equation implies that k1+k2=k1∗+k2∗k\_{1}+k\_{2}=k^{\*}\_{1}+k^{\*}\_{2} and

|  |  |  |  |
| --- | --- | --- | --- |
|  | {ωi′,exp⁡(β1​i⊤​X+β0​i)∑j=1k2exp⁡(β1​j⊤​X+β0​j)\displaystyle\Bigg{\{}\omega\_{i^{\prime}},\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})} | :i′∈[k1],i∈[k2]}\displaystyle:i^{\prime}\in[k\_{1}],\ i\in[k\_{2}]\Bigg{\}} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ={ωi′∗,exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗):i′∈[k1∗],i∈[k2∗]},\displaystyle=\Bigg{\{}\omega^{\*}\_{i^{\prime}},\frac{\exp((\beta^{\*}\_{1i})^{\top}X+\beta^{\*}\_{0i})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta^{\*}\_{1j})^{\top}X+\beta^{\*}\_{0j})}:i^{\prime}\in[k^{\*}\_{1}],\ i\in[k^{\*}\_{2}]\Bigg{\}}, |  |

for almost surely XX. As the weights ωi′\omega\_{i^{\prime}} and ωi′∗\omega^{\*}\_{i^{\prime}} are independent of XX for all i′∈[k1∗]i^{\prime}\in[k^{\*}\_{1}], we deduce k1=k1∗k\_{1}=k^{\*}\_{1} and {ωi′:i′∈[k1∗]}={ωi′∗:i′∈[k1∗]}\{\omega\_{i^{\prime}}:i^{\prime}\in[k^{\*}\_{1}]\}=\{\omega^{\*}\_{i^{\prime}}:i^{\prime}\in[k^{\*}\_{1}]\}. For simplicity, we assume WLOG that ωi′=ωi′∗\omega\_{i^{\prime}}=\omega^{\*}\_{i^{\prime}} for all i′∈[k1∗]i^{\prime}\in[k^{\*}\_{1}]. Furthermore, we also get k2=k2∗k\_{2}=k^{\*}\_{2} and

|  |  |  |  |
| --- | --- | --- | --- |
|  | {exp⁡(β1​i⊤​X+β0​i)∑j=1k2∗exp⁡(β1​j⊤​X+β0​j)\displaystyle\Bigg{\{}\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})} | :i∈[k2∗]}={exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗):i∈[k2∗]},\displaystyle:i\in[k^{\*}\_{2}]\Bigg{\}}=\Bigg{\{}\frac{\exp((\beta^{\*}\_{1i})^{\top}X+\beta^{\*}\_{0i})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta^{\*}\_{1j})^{\top}X+\beta^{\*}\_{0j})}:i\in[k^{\*}\_{2}]\Bigg{\}}, |  |

for almost surely XX. Again, we assume WLOG that exp⁡(β1​i⊤​X+β0​i)∑j=1k2∗exp⁡(β1​j⊤​X+β0​j)=exp⁡((β1​i∗)⊤​X+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)\frac{\exp(\beta\_{1i}^{\top}X+\beta\_{0i})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp(\beta\_{1j}^{\top}X+\beta\_{0j})}=\frac{\exp((\beta^{\*}\_{1i})^{\top}X+\beta^{\*}\_{0i})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta^{\*}\_{1j})^{\top}X+\beta^{\*}\_{0j})} for almost surely XX for all i∈[k2∗]i\in[k^{\*}\_{2}]. Due to the invariance to translation of the softmax function, this result indicates that β1​i=β1​i∗+c1\beta\_{1i}=\beta^{\*}\_{1i}+c\_{1} and β0​i=β0​i∗+c0\beta\_{0i}=\beta^{\*}\_{0i}+c\_{0} for some c1∈ℝdc\_{1}\in\mathbb{R}^{d} and c0∈ℝc\_{0}\in\mathbb{R}. Then, it follows from the assumption β1​k2∗=β1​k2∗∗=0d\beta\_{1k^{\*}\_{2}}=\beta^{\*}\_{1k^{\*}\_{2}}=0\_{d} and β0​k2∗=β0​k2∗∗=0\beta\_{0k^{\*}\_{2}}=\beta^{\*}\_{0k^{\*}\_{2}}=0 that c1=0dc\_{1}=0\_{d} and c0=0c\_{0}=0. Therefore, we obtain β1​i=β1​i∗\beta\_{1i}=\beta^{\*}\_{1i} and β0​i=β0​i∗\beta\_{0i}=\beta^{\*}\_{0i} for all i∈[k2∗]i\in[k^{\*}\_{2}].

Subsequently, we partition the index set [k1∗][k^{\*}\_{1}] into disjoint subsets U1,U2,…,Um1U\_{1},U\_{2},\ldots,U\_{m\_{1}} such that for each ℓ∈[m1]\ell\in[m\_{1}], we have (i) ωi=ωi′∗\omega\_{i}=\omega^{\*}\_{i^{\prime}} for i,i′∈Uℓi,i^{\prime}\in U\_{\ell} and (ii) ωi≠ωi′∗\omega\_{i}\neq\omega^{\*}\_{i^{\prime}} if ii and i′i^{\prime} dot not belong to the same set UℓU\_{\ell}. Similarly, we also partition the index set [k2∗][k^{\*}\_{2}] into disjoint subsets V1,V2,…,Vm2V\_{1},V\_{2},\ldots,V\_{m\_{2}} such that for each ℓ∈[m2]\ell\in[m\_{2}], we have (i) exp⁡(β0​i)=exp⁡(β0​i′∗)\exp(\beta\_{0i})=\exp(\beta^{\*}\_{0i^{\prime}}) for i,i′∈Vℓi,i^{\prime}\in V\_{\ell} and (ii) exp⁡(β0​i)≠exp⁡(β0​i′∗)\exp(\beta\_{0i})\neq\exp(\beta^{\*}\_{0i^{\prime}}) if ii and i′i^{\prime} dot not belong to the same set VℓV\_{\ell}.
As a consequence, we can rewrite equation ([44](#A5.E44 "In E.2 Identifiability of DeepSeekMoE ‣ Appendix E Proof of Auxiliary Results")) as

|  |  |  |
| --- | --- | --- |
|  | 12​∑ℓ=1m1∑i∈Uℓωi​π​(Y|h1​(X,κi),τi)+12​S​∑ℓ=1m2∑i∈Vℓexp⁡(β0​i)​exp⁡(β1​i⊤​X)​π​(Y|h2​(X,ηi),νi)\displaystyle\frac{1}{2}\sum\_{\ell=1}^{m\_{1}}\sum\_{i\in U\_{\ell}}\omega\_{i}\pi(Y|h\_{1}(X,\kappa\_{i}),\tau\_{i})+\frac{1}{2S}\sum\_{\ell=1}^{m\_{2}}\sum\_{i\in V\_{\ell}}\exp(\beta\_{0i})\exp(\beta\_{1i}^{\top}X)\pi(Y|h\_{2}(X,\eta\_{i}),\nu\_{i}) |  |
|  |  |  |
| --- | --- | --- |
|  | =12​∑ℓ=1m1∑i∈Uℓωi∗​π​(Y|h1​(X,κi∗),τi∗)+12​S​∑ℓ=1m2∑i∈Vℓexp⁡(β0​i∗)​exp⁡((β1​i∗)⊤​X)​π​(Y|h2​(X,ηi∗),νi∗),\displaystyle=\frac{1}{2}\sum\_{\ell=1}^{m\_{1}}\sum\_{i\in U\_{\ell}}\omega^{\*}\_{i}\pi(Y|h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i})+\frac{1}{2S}\sum\_{\ell=1}^{m\_{2}}\sum\_{i\in V\_{\ell}}\exp(\beta\_{0i}^{\*})\exp((\beta\_{1i}^{\*})^{\top}X)\pi(Y|h\_{2}(X,\eta^{\*}\_{i}),\nu\_{i}^{\*}), |  |

for almost surely (X,Y)(X,Y), where we denote S:=∑j=1k2∗exp⁡((β1​j∗)⊤​X+β0​j∗)S:=\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*}). The above equation implies that

|  |  |  |  |
| --- | --- | --- | --- |
|  | {(h1​(X,κi),τi):i∈Uℓ}\displaystyle\{(h\_{1}(X,\kappa\_{i}),\tau\_{i}):i\in U\_{\ell}\} | ={(h1​(X,κi∗),τi∗):i∈Uℓ},∀ℓ∈[m1]\displaystyle=\{(h\_{1}(X,\kappa^{\*}\_{i}),\tau^{\*}\_{i}):i\in U\_{\ell}\},\quad\forall\ell\in[m\_{1}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | {(h2​(X,ηi),νi):i∈Vℓ}\displaystyle\{(h\_{2}(X,\eta\_{i}),\nu\_{i}):i\in V\_{\ell}\} | ={(h2​(X,ηi∗),νi∗):i∈Vℓ},∀ℓ∈[m2],\displaystyle=\{(h\_{2}(X,\eta^{\*}\_{i}),\nu^{\*}\_{i}):i\in V\_{\ell}\},\quad\forall\ell\in[m\_{2}], |  |

for almost surely XX. As the expert functions h1h\_{1} and h2h\_{2} are identifiable, we deduce

|  |  |  |  |
| --- | --- | --- | --- |
|  | {(κi,τi):i∈Uℓ}\displaystyle\{(\kappa\_{i},\tau\_{i}):i\in U\_{\ell}\} | ={(κi∗,τi∗):i∈Uℓ},∀ℓ∈[m1]\displaystyle=\{(\kappa^{\*}\_{i},\tau^{\*}\_{i}):i\in U\_{\ell}\},\quad\forall\ell\in[m\_{1}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | {(ηi,νi):i∈Vℓ}\displaystyle\{(\eta\_{i},\nu\_{i}):i\in V\_{\ell}\} | ={(ηi∗,νi∗):i∈Vℓ},∀ℓ∈[m2].\displaystyle=\{(\eta^{\*}\_{i},\nu^{\*}\_{i}):i\in V\_{\ell}\},\quad\forall\ell\in[m\_{2}]. |  |

Therefore, we obtain

|  |  |  |
| --- | --- | --- |
|  | G1=∑ℓ=1m1∑i∈Uℓωi​δ(κi,τi)=∑ℓ=1m1∑i∈Uℓωi∗​δ(κi∗,τi∗)=G1∗,\displaystyle G\_{1}=\sum\_{\ell=1}^{m\_{1}}\sum\_{i\in U\_{\ell}}\omega\_{i}\delta\_{(\kappa\_{i},\tau\_{i})}=\sum\_{\ell=1}^{m\_{1}}\sum\_{i\in U\_{\ell}}\omega^{\*}\_{i}\delta\_{(\kappa^{\*}\_{i},\tau^{\*}\_{i})}=G^{\*}\_{1}, |  |
|  |  |  |
| --- | --- | --- |
|  | G2=∑ℓ=1m2∑i∈Vℓexp⁡(β0​i)​δ(β1​i,ηi,νi)=∑ℓ=1m2∑i∈Vℓexp⁡(β0​i∗)​δ(β1​i∗,ηi∗,νi∗)=G2∗.\displaystyle G\_{2}=\sum\_{\ell=1}^{m\_{2}}\sum\_{i\in V\_{\ell}}\exp(\beta\_{0i})\delta\_{(\beta\_{1i},\eta\_{i},\nu\_{i})}=\sum\_{\ell=1}^{m\_{2}}\sum\_{i\in V\_{\ell}}\exp(\beta^{\*}\_{0i})\delta\_{(\beta^{\*}\_{1i},\eta^{\*}\_{i},\nu^{\*}\_{i})}=G^{\*}\_{2}. |  |

Hence, the proof is completed.
∎

## Appendix F Extended Theoretical Results for Sparse Gating MoE

In this appendix, we extend the convergence analysis of parameter and expert estimations presented in Theoreom [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy") to the setting of a Top-KK sparse gating function. Our main arguments rely on fundamental techniques for dealing with the sparse gating function proposed in [[54](#bib.bib54)]. Since the results of Theorems [2](#Thmtheorem2 "Theorem 2. ‣ 2.2 Linear Experts ‣ 2 On Shared Expert Strategy"), [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating"), and [4](#Thmtheorem4 "Theorem 4. ‣ A.2 Dense Regime ‣ Appendix A On Normalized Sigmoid Gating") can be extended in a similar fashion, we will omit their extension here.

Problem setting: Assume that (X1,Y1),…,(Xn,Yn)∈ℝd×ℝ(X\_{1},Y\_{1}),\ldots,(X\_{n},Y\_{n})\in\mathbb{R}^{d}\times\mathbb{R} are i.i.d. samples drawn from the softmax gating Gaussian mixture of experts of order k∗k\_{\*} whose conditional density function sG1∗,G2∗​(y|x)s\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x) is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | sG1∗,G2∗​(y|x):=12\displaystyle s\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x):=\frac{1}{2} | ∑i=1k1∗ωi∗​π​(y|h1​(x,κi∗),τi∗)\displaystyle\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(y|h\_{1}(x,\kappa^{\*}\_{i}),\tau^{\*}\_{i}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +12​∑i=1k2∗softmax​(TopK​((β1​i∗)⊤​x;β0​i∗))​π​(y|h2​(x,ηi∗),νi∗),\displaystyle+\frac{1}{2}\sum\_{i=1}^{k^{\*}\_{2}}\mathrm{softmax}(\mathrm{Top}\_{K}((\beta\_{1i}^{\*})^{\top}x;\beta\_{0i}^{\*}))\pi(y|h\_{2}(x,\eta^{\*}\_{i}),\nu\_{i}^{\*}), |  | (45) |

where the pair of ground-truth mixing measures (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) are given by G1∗:=∑i=1k1∗ωi∗​δ(κi∗,τi∗)G^{\*}\_{1}:=\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\delta\_{(\kappa^{\*}\_{i},\tau^{\*}\_{i})} and G2∗:=∑i=1k2∗exp⁡(β0​i∗)​δ(β1​i∗,ηi∗,νi∗)G^{\*}\_{2}:=\sum\_{i=1}^{k^{\*}\_{2}}\exp(\beta\_{0i}^{\*})\delta\_{(\beta\_{1i}^{\*},\eta\_{i}^{\*},\nu\_{i}^{\*})}. Additionally, for any natural number kk and vectors (vi)i=1k(v\_{i})\_{i=1}^{k} and (ui)i=1(u\_{i})\_{i=1} in ℝk\mathbb{R}^{k}, the TopK\mathrm{Top}\_{K} sparse function is defined as

|  |  |  |
| --- | --- | --- |
|  | TopK​(vi,K;ui):={vi+ui,if ​vi​ is in the top ​K​ elements of ​v;−∞,otherwise,\displaystyle\mathrm{Top}\_{K}(v\_{i},K;u\_{i}):=\begin{cases}v\_{i}+u\_{i},\hskip 23.33147pt\text{if }v\_{i}\text{ is in the top }K\text{ elements of }v;\\ -\infty,\hskip 34.14322pt\text{otherwise},\end{cases} |  |

while the softmax function is formulated as softmax​(vi):=exp⁡(vi)/∑j=1kexp⁡(vj)\mathrm{softmax}(v\_{i}):={\exp(v\_{i})}/{\sum\_{j=1}^{k}\exp(v\_{j})}.

In practice, since the number of shared experts k1∗k^{\*}\_{1} and routed experts k∗​2k^{\*}2 are typically unknown, we have to fit the ground-truth model ([F](#A6.Ex501 "Appendix F Extended Theoretical Results for Sparse Gating MoE")) with k1>k1∗k\_{1}>k^{\*}\_{1} shared experts and k2>k2∗k\_{2}>k^{\*}\_{2} routed experts. Thus, some ground-truth shared experts and routed experts will be fitted by more than one estimated expert. As a result, since there are KK routed experts activated per input in the ground-truth density sG1∗,G2∗s\_{G^{\*}\_{1},G^{\*}\_{2}}, it is necessary to activate K¯>K\bar{K}>K experts in the density estimation in order to ensure its convergence to the true density. For that purpose, let us introduce the formulation of the density estimation as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | s¯G1n,G2n​(Y|X):=12\displaystyle\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(Y|X):=\frac{1}{2} | ∑i=1k1nωin​π​(y|h1​(x,κin),τin)\displaystyle\sum\_{i=1}^{k^{n}\_{1}}\omega^{n}\_{i}\pi(y|h\_{1}(x,\kappa^{n}\_{i}),\tau^{n}\_{i}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +12​∑i=1k2nsoftmax​(TopK¯​((β1​in)⊤​x;β0​in))​π​(y|h2​(x,ηin),νin),\displaystyle+\frac{1}{2}\sum\_{i=1}^{k^{n}\_{2}}\mathrm{softmax}(\mathrm{Top}\_{\bar{K}}((\beta\_{1i}^{n})^{\top}x;\beta\_{0i}^{n}))\pi(y|h\_{2}(x,\eta^{n}\_{i}),\nu\_{i}^{n}), |  |

where K<K¯≤k2K<\bar{K}\leq k\_{2} and the pair of mixing measure estimations (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) are defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | (G^1n,G^2n)∈arg​max(G1,G2)∈𝒢k1,k2​(Θ)⁡1n​∑i=1nlog⁡(s¯G1,G2​(Yi|Xi)),\displaystyle(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2})\in\operatorname\*{arg\,max}\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)}\frac{1}{n}\sum\_{i=1}^{n}\log(\bar{s}\_{G\_{1},G\_{2}}(Y\_{i}|X\_{i})), |  | (46) |

where the set of mixing measures 𝒢k1,k2​(Θ):=𝒢k1​(Θ1)×𝒢k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):=\mathcal{G}\_{k\_{1}}(\Theta\_{1})\times\mathcal{G}\_{k\_{2}}(\Theta) is defined below equation ([2](#S2.E2 "In 2 On Shared Expert Strategy")).

Input space partition w.r.t the true density. In order that the density estimation sG1n,G2ns\_{G^{n}\_{1},G^{n}\_{2}} converges to the true density sG1∗,G2∗s\_{G^{\*}\_{1},G^{\*}\_{2}}, we must ensure that for each input, the K¯\bar{K} routed experts activated in the density estimation converge to the KK routed experts activated in the true density. Since the activated experts vary with the input value, we need to partition the input space 𝒳\mathcal{X} into M:=(k2∗K)M:=\binom{k^{\*}\_{2}}{K} regions 𝒳m∗\mathcal{X}^{\*}\_{m} corresponding to (k2∗K)\binom{k^{\*}\_{2}}{K} choices of activated experts in the true density. For each m∈[M]m\in[M], let us denote {m1,m2,…,mK}\{m\_{1},m\_{2},\ldots,m\_{K}\} as an KK-element subset of the index set [k2∗][k^{\*}\_{2}], and {mK+1,…,mk2∗}:=[k2∗]∖{m1,m2,…,mK}\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\}:=[k^{\*}\_{2}]\setminus\{m\_{1},m\_{2},\ldots,m\_{K}\}. Then, the mm-th region of the input space is defined as

|  |  |  |
| --- | --- | --- |
|  | 𝒳m∗:={x∈𝒳:(β1​i∗)⊤​x≥(β1​i′∗)⊤​x,∀i∈{m1,m2,…,mK},i′∈{mK+1,…,mk2∗}},\displaystyle\mathcal{X}^{\*}\_{m}:=\Big{\{}x\in\mathcal{X}:(\beta^{\*}\_{1i})^{\top}x\geq(\beta^{\*}\_{1i^{\prime}})^{\top}x,\ \forall i\in\{m\_{1},m\_{2},\ldots,m\_{K}\},i^{\prime}\in\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\}\Big{\}}, |  |

for any m∈[M]m\in[M]. For example, suppose that X∈𝒳m∗X\in\mathcal{X}^{\*}\_{m} where m∈[M]m\in[M] such that {m1,m2,…,mK}={1,2,…,K}\{m\_{1},m\_{2},\ldots,m\_{K}\}=\{1,2,\ldots,K\}. Then, it follows that

|  |  |  |
| --- | --- | --- |
|  | TopK​((β1​i∗)⊤​X;β0​i∗)=(β1​i∗)⊤​X+β0​i∗,\displaystyle\mathrm{Top}\_{K}((\beta\_{1i}^{\*})^{\top}X;\beta\_{0i}^{\*})=(\beta\_{1i}^{\*})^{\top}X+\beta\_{0i}^{\*}, |  |

for all i∈[K]i\in[K]. In other words, h2​(X,η1∗),h2​(X,η2∗),…,h2​(X,ηK∗)h\_{2}(X,\eta^{\*}\_{1}),h\_{2}(X,\eta^{\*}\_{2}),\ldots,h\_{2}(X,\eta^{\*}\_{K}) are the KK routed experts activated in the true density sG1∗,G2∗​(y|x)s\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x), which is reduced to

|  |  |  |  |
| --- | --- | --- | --- |
|  | sG1∗,G2∗​(y|x):=12\displaystyle s\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x):=\frac{1}{2} | ∑i=1k1∗ωi∗​π​(y|h1​(x,κi∗),τi∗)\displaystyle\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(y|h\_{1}(x,\kappa^{\*}\_{i}),\tau^{\*}\_{i}) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +12​∑i=1Kexp⁡((β1​i∗)⊤​x+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​x+β0​j∗)⋅π​(y|h2​(x,ηi∗),νi∗).\displaystyle+\frac{1}{2}\sum\_{i=1}^{K}\frac{\exp((\beta\_{1i}^{\*})^{\top}x+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}x+\beta\_{0j}^{\*})}\cdot\pi(y|h\_{2}(x,\eta^{\*}\_{i}),\nu\_{i}^{\*}). |  | (47) |

Input space partition w.r.t the density estimation.
Next, with the same input X∈𝒳m∗X\in\mathcal{X}^{\*}\_{m}, we need to guarantee that the routed expert estimations converging to the above KK routed experts activated in the true density sG1∗,G2∗​(Y|X)s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) are also activated in the density estimation sG1n,G2n​(Y|X)s\_{G^{n}\_{1},G^{n}\_{2}}(Y|X). For that purpose, it is necessary to partition the input space with respect to the density estimation. In particular, we partition the input space into M¯:=(k2K¯)\bar{M}:=\binom{k\_{2}}{\bar{K}} regions 𝒳¯m\bar{\mathcal{X}}\_{m} corresponding to (k2K¯)\binom{k\_{2}}{\bar{K}} choices of activated experts in the true density. For each m¯∈[M¯]\bar{m}\in[\bar{M}], we denote {m¯1,m¯2,…,m¯K¯}\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\} as an K¯\bar{K}-element subset of the index set [k2][k\_{2}], and {m¯K¯+1,…,m¯k2}:=[k2]∖{m¯1,m¯2,…,m¯K¯}\{\bar{m}\_{\bar{K}+1},\ldots,\bar{m}\_{k\_{2}}\}:=[k\_{2}]\setminus\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}.
Given these notations, we are ready to show that the input partition w.r.t the density estimation aligns with the input space partition w.r.t the true density in the following lemma whose proof will be provided in Appendix [F.1](#A6.SS1 "F.1 Proof of Lemma 3 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE"):

###### Lemma 3.

For any j∈[k2∗]j\in[k^{\*}\_{2}], i∈𝒱2,ji\in\mathcal{V}\_{2,j} and β1​i,β1​j∗∈ℝd\beta\_{1i},\beta^{\*}\_{1j}\in\mathbb{R}^{d}, assume that there exist sufficiently small εj>0\varepsilon\_{j}>0 satisfying ‖β1​i−β1​j∗‖≤εj\|\beta\_{1i}-\beta^{\*}\_{1j}\|\leq\varepsilon\_{j}. Moreover, suppose that there exist m∈[M]m\in[M] and m¯∈[M¯]\bar{m}\in[\bar{M}] such that {m¯1,m¯2,…,m¯K¯}=𝒱2,m1∪𝒱2,m2​…∪𝒱2,mK\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,m\_{1}}\cup\mathcal{V}\_{2,m\_{2}}\ldots\cup\mathcal{V}\_{2,m\_{K}}. Then, for any m∈[M]m\in[M], if the input region 𝒳m∗\mathcal{X}^{\*}\_{m} has non-zero measure, we have 𝒳m∗=𝒳¯m¯\mathcal{X}^{\*}\_{m}=\bar{\mathcal{X}}\_{\bar{m}},
where

|  |  |  |
| --- | --- | --- |
|  | 𝒳¯m¯:={x∈𝒳:(β1​i)⊤​x≥(β1​i′)⊤​x,∀i∈{m¯1,m¯2,…,m¯K¯},i′∈{m¯K+1,…,m¯k2}}.\displaystyle\bar{\mathcal{X}}\_{\bar{m}}:=\Big{\{}x\in\mathcal{X}:(\beta\_{1i})^{\top}x\geq(\beta\_{1i^{\prime}})^{\top}x,\ \forall i\in\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\},i^{\prime}\in\{\bar{m}\_{K+1},\ldots,\bar{m}\_{k\_{2}}\}\Big{\}}. |  |

Suppose that the expert estimation h​(X,η^in)h(X,\hat{\eta}^{n}\_{i}) converges to the ground-truth expert h​(X,ηj∗)h(X,\eta^{\*}\_{j}) for some j∈[k2∗]j\in[k^{\*}\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j}. Then, Lemma [3](#Thmlemma3 "Lemma 3. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE") reveals that for almost surely XX, if the expert h​(X,ηj∗)h(X,\eta^{\*}\_{j}) is activated in the true density, then the expert h​(X,η^in)h(X,\hat{\eta}^{n}\_{i}) is also activated in the density estimation. Mathematically, we have TopK​((β1​j∗)⊤​X;β0​j∗)=(β1​j∗)⊤​X+β0​j∗\mathrm{Top}\_{K}((\beta\_{1j}^{\*})^{\top}X;\beta\_{0j}^{\*})=(\beta\_{1j}^{\*})^{\top}X+\beta\_{0j}^{\*} occurs holds if and only if TopK¯​((β^1​in)⊤​X;β^0​in)=(β^1​in)⊤​X+β^0​in\mathrm{Top}\_{\bar{K}}((\hat{\beta}^{n}\_{1i})^{\top}X;\hat{\beta}^{n}\_{0i})=(\hat{\beta}^{n}\_{1i})^{\top}X+\hat{\beta}^{n}\_{0i}.

Density estimation convergence. Given the above input partition w.r.t the density estimation, we exhibit in Proposition [6](#Thmproposition6 "Proposition 6. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE") an interesting phenomenon that the density estimation s¯G^1n,G^2n\bar{s}\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}} converges to the true density sG1∗,G2∗s\_{G^{\*}\_{1},G^{\*}\_{2}} under the Total Variation distance only if the number of routed experts activated in the density estimation is bounded below as K¯≥max{m1,m2,…,mK}⊂[k2∗]​∑j=1K|𝒱2,mj|\bar{K}\geq\max\_{\{m\_{1},m\_{2},\ldots,m\_{K}\}\subset[k^{\*}\_{2}]}\sum\_{j=1}^{K}|\mathcal{V}\_{2,m\_{j}}|.

###### Proposition 6.

If K¯<max{m1,m2,…,mK}⊂[k2∗]​∑j=1K|𝒱2,mj|\bar{K}<\max\_{\{m\_{1},m\_{2},\ldots,m\_{K}\}\subset[k^{\*}\_{2}]}\sum\_{j=1}^{K}|\mathcal{V}\_{2,m\_{j}}|, then the following holds:

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ)𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta)}\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]>0. |  |

Proof of Proposition [6](#Thmproposition6 "Proposition 6. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE") will be provided in Appendix [F.2](#A6.SS2 "F.2 Proof of Proposition 6 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE"). Following from the result of Proposition [6](#Thmproposition6 "Proposition 6. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE"), we will assume max{m1,m2,…,mK}⊂[k2∗]​∑j=1K|𝒱2,mj|≤K¯≤k2\max\_{\{m\_{1},m\_{2},\ldots,m\_{K}\}\subset[k^{\*}\_{2}]}\sum\_{j=1}^{K}|\mathcal{V}\_{2,m\_{j}}|\leq\bar{K}\leq k\_{2} in the rest of this appendix unless stating otherwise to ensure the convergence of density estimation. Next, by combining the above results and the arguments used to prove Proposition [1](#Thmproposition1 "Proposition 1. ‣ 2 On Shared Expert Strategy"), we arrive at the following density estimation rate.

###### Proposition 7.

The density estimation s¯G^1n,G^2n​(Y|X)\bar{s}\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(Y|X) converges to the true density sG1∗,G2∗​(Y|X)s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) at the following rate:

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(s¯G^1n,G^2n(⋅|X),sG1∗,G2∗(⋅|X))]=𝒪P([log(n)/n]12).\displaystyle\mathbb{E}\_{X}[V(\bar{s}\_{\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

Voronoi loss. In align with the above input partition w.r.t the true density, we need to modify the formulation of the Voronoi loss previously defined in equation ([2.1](#S2.Ex5 "2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy")) as follows:

|  |  |  |
| --- | --- | --- |
|  | 𝒟5((G1,G2),(G1∗,G2∗)):=max{m1,…,mK}⊂[k2∗]{∑j=1k1∗|∑i∈𝒱1,jωi−ωj∗|+∑j=1K|∑i∈𝒱2,mjexp(β0​i)−exp(β0​j∗)|\displaystyle\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})):=\max\_{\{m\_{1},\ldots,m\_{K}\}\subset[k^{\*}\_{2}]}\Bigg{\{}\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}-\omega\_{j}^{\*}\Big{|}+\sum\_{j=1}^{K}\Big{|}\sum\_{i\in\mathcal{V}\_{2,m\_{j}}}\exp(\beta\_{0i})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |
| --- | --- | --- |
|  | +∑j∈[k1∗],|𝒱1,j|=1∑i∈𝒱1,jωi​(‖Δ​κi​j‖+|Δ​τi​j|)+∑j∈[K],|𝒱2,mj|=1∑i∈𝒱2,mjexp⁡(β0​i)​(‖Δ​β1​i​mj‖+‖Δ​ηi​mj‖+|Δ​νi​mj|)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|+|\Delta\tau\_{ij}|)+\sum\_{\begin{subarray}{c}j\in[K],\\ |\mathcal{V}\_{2,m\_{j}}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,m\_{j}}}\exp(\beta\_{0i})(\|\Delta\beta\_{1im\_{j}}\|+\|\Delta\eta\_{im\_{j}}\|+|\Delta\nu\_{im\_{j}}|) |  |
|  |  |  |
| --- | --- | --- |
|  | +∑j∈[k1∗],|𝒱1,j|>1∑i∈𝒱1,jωi(∥Δκi​j∥2+|Δτi​j|2)+∑j∈[K],|𝒱2,mj|>1∑i∈𝒱2,mjexp(β0​i)(∥Δβ1​i​mj∥2+∥Δηi​mj∥2+|Δνi​mj|2)}.\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}(\|\Delta\kappa\_{ij}\|^{2}+|\Delta\tau\_{ij}|^{2})+\sum\_{\begin{subarray}{c}j\in[K],\\ |\mathcal{V}\_{2,m\_{j}}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,m\_{j}}}\exp(\beta\_{0i})(\|\Delta\beta\_{1im\_{j}}\|^{2}+\|\Delta\eta\_{im\_{j}}\|^{2}+|\Delta\nu\_{im\_{j}}|^{2})\Bigg{\}}. |  |

The maximum operator in the above formulation helps capture the convergence behavior of the parameter estimation in different input regions partitioned w.r.t the true density. Given the loss function 𝒟5​(G,G∗)\mathcal{D}\_{5}(G,G\_{\*}), it is sufficient to establish parameter and expert estimation rates in the following theorem:

###### Theorem 5.

Suppose that the expert functions h1h\_{1} and h2h\_{2} are strongly identifiable. Then, the lower bound 𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]≳𝒟5((G1,G2),(G1∗,G2∗))\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\gtrsim\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2})) holds for any (G1,G2)∈𝒢k1,k2​(Θ)(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). As a consequence, we have

|  |  |  |
| --- | --- | --- |
|  | 𝒟5(G^1n,G^2n),(G1∗,G2∗))=𝒪P([log(n)/n]12).\displaystyle\mathcal{D}\_{5}(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=\mathcal{O}\_{P}([\log(n)/n]^{\frac{1}{2}}). |  |

###### Proof of Theorem [5](#Thmtheorem5 "Theorem 5. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE").

Analogous to Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), it suffices to derive the local part

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟5​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]𝒟5​((G1,G2),(G1∗,G2∗))>0,\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\dfrac{\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0, |  | (48) |

and the global part

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟5​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]𝒟5​((G1,G2),(G1∗,G2∗))>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\dfrac{\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]}{\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))}>0. |  | (49) |

in this appendix. However, since the global part ([21](#A4.E21 "In D.2 Proof of Theorem 2 ‣ Appendix D Proof of Main Results")) can be established in the same fashion as in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"), its proof is omitted here. Thus, we will focus on showing only the local part ([48](#A6.E48 "In Appendix F Extended Theoretical Results for Sparse Gating MoE")). Suppose that the local part does not hold. Then, we can find a sequence of mixing measure pairs (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) of the form G1n:=∑i=1k1nωin​δ(κ1​in,κ0​in,τin)G^{n}\_{1}:=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{1i}^{n},\kappa\_{0i}^{n},\tau\_{i}^{n})}, G2n:=∑i=1k2nexp⁡(β0​in)​δ(β1​in,η1​in,η0​in,νin)G^{n}\_{2}:=\sum\_{i=1}^{k^{n}\_{2}}\exp(\beta\_{0i}^{n})\delta\_{(\beta\_{1i}^{n},\eta\_{1i}^{n},\eta\_{0i}^{n},\nu\_{i}^{n})} for n∈ℕn\in\mathbb{N} satisfying 𝒟5​n:=𝒟5​((G1n,G2n),(G1∗,G2∗))→0\mathcal{D}\_{5n}:=\mathcal{D}\_{5}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\to 0 and

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X[V(s¯G1n,G2n(⋅|X),sG1∗,G2∗(⋅|X))]/𝒟5​n→0,\displaystyle\mathbb{E}\_{X}[V(\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{5n}\to 0, |  | (50) |

as n→∞n\to\infty. Here, we may assume WLOG that the number of shared experts and routed experts k1nk^{n}\_{1}, k2nk^{n}\_{2} and Voronoi cells 𝒱1,j=𝒱1,j​(G1n)\mathcal{V}\_{1,j}=\mathcal{V}\_{1,j}(G^{n}\_{1}), 𝒱2,j=𝒱2,j​(G2n)\mathcal{V}\_{2,j}=\mathcal{V}\_{2,j}(G^{n}\_{2}) do not change with the sample size nn. WLOG, we may assume that the Voronoi loss 𝒟5​n\mathcal{D}\_{5n} is reduced to

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝒟5​n=∑j=1k1∗|∑i∈𝒱1,jωin−ωj∗|+∑j=1K|∑i∈𝒱2,jexp⁡(β0​in)−exp⁡(β0​j∗)|\displaystyle\mathcal{D}\_{5n}=\sum\_{j=1}^{k^{\*}\_{1}}\Big{|}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}-\omega\_{j}^{\*}\Big{|}+\sum\_{j=1}^{K}\Big{|}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})-\exp(\beta\_{0j}^{\*})\Big{|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|=1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖+|Δ​τi​jn|)+∑j∈[K],|𝒱2,j|=1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖+‖Δ​ηi​jn‖+|Δ​νi​jn|)\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|+|\Delta\tau\_{ij}^{n}|)+\sum\_{\begin{subarray}{c}j\in[K],\\ |\mathcal{V}\_{2,j}|=1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|+\|\Delta\eta\_{ij}^{n}\|+|\Delta\nu\_{ij}^{n}|) |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑j∈[k1∗],|𝒱1,j|>1∑i∈𝒱1,jωin​(‖Δ​κi​jn‖2+|Δ​τi​jn|2)+∑j∈[K],|𝒱2,j|>1∑i∈𝒱2,jexp⁡(β0​in)​(‖Δ​β1​i​jn‖2+‖Δ​ηi​jn‖2+|Δ​νi​jn|2).\displaystyle+\sum\_{\begin{subarray}{c}j\in[k^{\*}\_{1}],\\ |\mathcal{V}\_{1,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{1,j}}\omega\_{i}^{n}(\|\Delta\kappa\_{ij}^{n}\|^{2}+|\Delta\tau\_{ij}^{n}|^{2})+\sum\_{\begin{subarray}{c}j\in[K],\\ |\mathcal{V}\_{2,j}|>1\end{subarray}}\sum\_{i\in\mathcal{V}\_{2,j}}\exp(\beta\_{0i}^{n})(\|\Delta\beta\_{1ij}^{n}\|^{2}+\|\Delta\eta\_{ij}^{n}\|^{2}+|\Delta\nu\_{ij}^{n}|^{2}). |  | (51) |

Recall that we partition the input space w.r.t the true density into M=(k2∗K)M=\binom{k^{\*}\_{2}}{K} regions. For each m∈[M]m\in[M], we denote {m1,m2,…,mK}\{m\_{1},m\_{2},\ldots,m\_{K}\} as a subset of the index set [k2∗][k^{\*}\_{2}] and {mK+1,…,mk2∗}=[k2∗]∖{m1,m2,…,mK}\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\}=[k^{\*}\_{2}]\setminus\{m\_{1},m\_{2},\ldots,m\_{K}\}. Then, the mm-th region is given by

|  |  |  |
| --- | --- | --- |
|  | 𝒳m∗:={x∈𝒳:(β1​i∗)⊤​x≥(β1​i′∗)⊤​x,∀i∈{m1,m2,…,mK},i′∈{mK+1,…,mk2∗}},\displaystyle\mathcal{X}^{\*}\_{m}:=\Big{\{}x\in\mathcal{X}:(\beta^{\*}\_{1i})^{\top}x\geq(\beta^{\*}\_{1i^{\prime}})^{\top}x,\ \forall i\in\{m\_{1},m\_{2},\ldots,m\_{K}\},i^{\prime}\in\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\}\Big{\}}, |  |

for any m∈[M]m\in[M]. Let K¯∈ℕ\bar{K}\in\mathbb{N} such that max⁡{∑j=1K|𝒱2,j|:{m1,…,mK}⊂[k2∗]}≤K¯≤k2\max\{\sum\_{j=1}^{K}|\mathcal{V}\_{2,j}|:\{m\_{1},\ldots,m\_{K}\}\subset[k^{\*}\_{2}]\}\leq\bar{K}\leq k\_{2} and let M¯:=(k2K¯)\bar{M}:=\binom{k\_{2}}{\bar{K}}. Next, for any m¯∈[M¯]\bar{m}\in[\bar{M}], we denote {m¯1,m¯2,…,m¯K¯}\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\} as a subset of the index set [k2][k\_{2}] and {m¯K¯+1,…,m¯k2}:=[k2]∖{m¯1,m¯2,…,m¯K¯}\{\bar{m}\_{\bar{K}+1},\ldots,\bar{m}\_{k\_{2}}\}:=[k\_{2}]\setminus\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}. Then, we partition the input space w.r.t the density estimation sG1n,G2n​(Y|X)s\_{G^{n}\_{1},G^{n}\_{2}}(Y|X) as 𝒳=∪m¯=1M¯𝒳m¯n\mathcal{X}=\cup\_{\bar{m}=1}^{\bar{M}}\mathcal{X}^{n}\_{\bar{m}}, where the m¯\bar{m}-th region is defined as

|  |  |  |
| --- | --- | --- |
|  | 𝒳m¯n:={x∈𝒳:(β1​in)⊤​x≥(β1​i′n)⊤​x,∀i∈{m¯1,m¯2,…,m¯K¯},i′∈{m¯K¯+1,…,m¯k2}}\displaystyle\mathcal{X}^{n}\_{\bar{m}}:=\Big{\{}x\in\mathcal{X}:(\beta^{n}\_{1i})^{\top}x\geq(\beta^{n}\_{1i^{\prime}})^{\top}x,\ \forall i\in\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\},i^{\prime}\in\{\bar{m}\_{\bar{K}+1},\ldots,\bar{m}\_{k\_{2}}\}\Big{\}} |  |

for any m¯∈[M¯]\bar{m}\in[\bar{M}]. Let X​𝒳m∗X\mathcal{X}^{\*}\_{m} for m∈[M]m\in[M] such that {m1,m2,…,mK}={1,2,…,K}\{m\_{1},m\_{2},\ldots,m\_{K}\}=\{1,2,\ldots,K\}. If there does not exist m¯∈[M¯]\bar{m}\in[\bar{M}] such that {m¯1,m¯2,…,m¯K¯}=𝒱2,1∪𝒱2,2∪…∪𝒱2,K\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,1}\cup\mathcal{V}\_{2,2}\cup\ldots\cup\mathcal{V}\_{2,K}, then the ratio 𝔼X[V(s¯G1n,G2n(⋅|X),sG1∗,G2∗(⋅|X))]/𝒟5​n\mathbb{E}\_{X}[V(\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]/\mathcal{D}\_{5n} does not converge to zero, which contradicts the result in equation ([50](#A6.E50 "In Appendix F Extended Theoretical Results for Sparse Gating MoE")). Thus, we can find m¯∈[M¯]\bar{m}\in[\bar{M}] such that {m¯1,m¯2,…,m¯K¯}=𝒱2,1∪𝒱2,2∪…∪𝒱2,K\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,1}\cup\mathcal{V}\_{2,2}\cup\ldots\cup\mathcal{V}\_{2,K}.

Since the Voronoi loss 𝒟5​n\mathcal{D}\_{5n} converges to zero, it follows that β1​in→β1​j∗\beta\_{1i}^{n}\to\beta\_{1j}^{\*} for all j∈[K]j\in[K] and i∈𝒱2,ji\in\mathcal{V}\_{2,j}. Then, by means of Lemma [3](#Thmlemma3 "Lemma 3. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE"), we deduce 𝒳m∗=𝒳m¯n\mathcal{X}^{\*}\_{m}=\mathcal{X}^{n}\_{\bar{m}} for sufficiently large nn, implying that X∈𝒳m¯nX\in\mathcal{X}^{n}\_{\bar{m}}. Therefore, we can represent the true density and the density estimation when the sample size nn is sufficiently large as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | sG1∗,G2∗​(y|x):=12\displaystyle s\_{G^{\*}\_{1},G^{\*}\_{2}}(y|x):=\frac{1}{2} | ∑i=1k1∗ωi∗​π​(y|h1​(x,κi∗),τi∗)+12​∑i=1Kexp⁡((β1​i∗)⊤​x+β0​i∗)∑j=1k2∗exp⁡((β1​j∗)⊤​x+β0​j∗)⋅π​(y|h2​(x,ηi∗),νi∗),\displaystyle\sum\_{i=1}^{k^{\*}\_{1}}\omega^{\*}\_{i}\pi(y|h\_{1}(x,\kappa^{\*}\_{i}),\tau^{\*}\_{i})+\frac{1}{2}\sum\_{i=1}^{K}\frac{\exp((\beta\_{1i}^{\*})^{\top}x+\beta\_{0i}^{\*})}{\sum\_{j=1}^{k^{\*}\_{2}}\exp((\beta\_{1j}^{\*})^{\top}x+\beta\_{0j}^{\*})}\cdot\pi(y|h\_{2}(x,\eta^{\*}\_{i}),\nu\_{i}^{\*}), |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | sG1n,G2n​(y|x):=12\displaystyle s\_{G^{n}\_{1},G^{n}\_{2}}(y|x):=\frac{1}{2} | ∑i=1k1nωi∗​π​(y|h1​(x,κin),τin)+12​∑i=1K¯exp⁡((β1​in)⊤​x+β0​in)∑j=1K¯exp⁡((β1​jn)⊤​x+β0​jn)⋅π​(y|h2​(x,ηin),νin).\displaystyle\sum\_{i=1}^{k^{n}\_{1}}\omega^{\*}\_{i}\pi(y|h\_{1}(x,\kappa^{n}\_{i}),\tau^{n}\_{i})+\frac{1}{2}\sum\_{i=1}^{\bar{K}}\frac{\exp((\beta\_{1i}^{n})^{\top}x+\beta\_{0i}^{n})}{\sum\_{j=1}^{\bar{K}}\exp((\beta\_{1j}^{n})^{\top}x+\beta\_{0j}^{n})}\cdot\pi(y|h\_{2}(x,\eta^{n}\_{i}),\nu\_{i}^{n}). |  |

Given the above formulations, we can achieve the local part ([48](#A6.E48 "In Appendix F Extended Theoretical Results for Sparse Gating MoE")) by employing the same arguments used in Appendix [D.1](#A4.SS1 "D.1 Proof of Theorem 1 ‣ Appendix D Proof of Main Results"). Hence, the proof is completed.
∎

### F.1 Proof of Lemma [3](#Thmlemma3 "Lemma 3. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")

Let us consider εj=Nj​η\varepsilon\_{j}=N\_{j}\eta, where η>0\eta>0 is some fixed constant, and Nj>0N\_{j}>0 will be chosen later. Since the input space 𝒳\mathcal{X} and the parameter space Θ\Theta are bounded, there exists a constant cm∗≥0c^{\*}\_{m}\geq 0 such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | minx,j,j′⁡[(β1​j∗)⊤​x−(β1​j′∗)⊤​x]=cm∗​η,\displaystyle\min\_{x,j,j^{\prime}}\Big{[}(\beta^{\*}\_{1j})^{\top}x-(\beta^{\*}\_{1j^{\prime}})^{\top}x\Big{]}=c^{\*}\_{m}\eta, |  | (52) |

where the above minimum is subject to x∈𝒳m∗,j∈{m1,m2,…,mK}x\in\mathcal{X}^{\*}\_{m},j\in\{m\_{1},m\_{2},\ldots,m\_{K}\} and j′∈{mK+1,…,mk2∗}j^{\prime}\in\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\}. We will show by contradiction that cm∗>0c^{\*}\_{m}>0. Suppose that cm∗=0c^{\*}\_{m}=0. For x∈𝒳m∗x\in\mathcal{X}^{\*}\_{m}, we may assume for any 1≤i<j≤k2∗1\leq i<j\leq k^{\*}\_{2} that

|  |  |  |
| --- | --- | --- |
|  | (β1​mi∗)⊤​x≥(β1​mj∗)⊤​x.\displaystyle(\beta^{\*}\_{1m\_{i}})^{\top}x\geq(\beta^{\*}\_{1m\_{j}})^{\top}x. |  |

As cm∗=0c^{\*}\_{m}=0, the result in equation ([52](#A6.E52 "In F.1 Proof of Lemma 3 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")) indicates that (β1​mK∗)⊤​x−(β1​mK+1∗)⊤​x=0(\beta^{\*}\_{1m\_{K}})^{\top}x-(\beta^{\*}\_{1m\_{K+1}})^{\top}x=0, or equivalently

|  |  |  |
| --- | --- | --- |
|  | (β1​mK∗−β1​mK+1∗)⊤​x=0.\displaystyle(\beta^{\*}\_{1m\_{K}}-\beta^{\*}\_{1m\_{K+1}})^{\top}x=0. |  |

In other words, 𝒳m∗\mathcal{X}^{\*}\_{m} is a subset of

|  |  |  |
| --- | --- | --- |
|  | 𝒩:={x∈𝒳:(β1​mK∗−β1​mK+1∗)⊤​x=0}.\displaystyle\mathcal{N}:=\{x\in\mathcal{X}:(\beta^{\*}\_{1m\_{K}}-\beta^{\*}\_{1m\_{K+1}})^{\top}x=0\}. |  |

Since the difference β1​mK−β1​mK+1\beta\_{1m\_{K}}-\beta\_{1m\_{K+1}} is non-zero and the input XX follows a continuous distribution, then the set 𝒩\mathcal{N} has measure zero. Furthermore, as 𝒳m∗⊆𝒩\mathcal{X}^{\*}\_{m}\subseteq\mathcal{N}, it follows that 𝒳m∗\mathcal{X}^{\*}\_{m} also has measure zero, which contradicts the fact that it has non-zero measure. Thus, we must have cm∗>0c^{\*}\_{m}>0.

Subsequently, let x∈𝒳m∗x\in\mathcal{X}^{\*}\_{m} and m¯∈[M¯]\bar{m}\in[\bar{M}] such that {m¯1,m¯2,…,m¯K¯}=𝒱2,m1∪𝒱2,m2∪…∪𝒱2,mK\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,m\_{1}}\cup\mathcal{V}\_{2,m\_{2}}\cup\ldots\cup\mathcal{V}\_{2,m\_{K}}. We will demonstrate that x∈𝒳¯m¯x\in\bar{\mathcal{X}}\_{\bar{m}}. Indeed, recall that the input space 𝒳\mathcal{X} is bounded, then we may assume that ‖x‖≤B\|x\|\leq B for any x∈𝒳x\in\mathcal{X}, where B>0B>0 is some constant. Then, for any i∈{m¯1,m¯2,…,m¯K¯}i\in\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\} and i′∈{m¯K¯+1,…,m¯k2}i^{\prime}\in\{\bar{m}\_{\bar{K}+1},\ldots,\bar{m}\_{k\_{2}}\}, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | β1​i⊤​x\displaystyle{\beta}\_{1i}^{\top}x | =(β1​i−β1​j∗)⊤​x+(β1​j∗)⊤​x\displaystyle=({\beta}\_{1i}-\beta^{\*}\_{1j})^{\top}x+(\beta^{\*}\_{1j})^{\top}x |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥−Nj​η​B+(β1​j′∗)⊤​x+cm∗​η\displaystyle\geq-N\_{j}\eta B+(\beta^{\*}\_{1j^{\prime}})^{\top}x+c^{\*}\_{m}\eta |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =−Nj​η​B+cm∗​η+(β1​j′∗−β1​i′)⊤​x+β1​i′⊤​x\displaystyle=-N\_{j}\eta B+c^{\*}\_{m}\eta+(\beta^{\*}\_{1j^{\prime}}-{\beta}\_{1i^{\prime}})^{\top}x+{\beta}\_{1i^{\prime}}^{\top}x |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ≥−2​Nj​η​B+cm∗​η+β1​i′⊤​x,\displaystyle\geq-2N\_{j}\eta B+c^{\*}\_{m}\eta+{\beta}\_{1i^{\prime}}^{\top}x, |  |

where j∈{m1,m2,…,mK}j\in\{m\_{1},m\_{2},\ldots,m\_{K}\} and j′∈{mK+1,…,mk2∗}j^{\prime}\in\{m\_{K+1},\ldots,m\_{k^{\*}\_{2}}\} such that i∈𝒱2,ji\in\mathcal{V}\_{2,j} and i′∈𝒱2,j′i^{\prime}\in\mathcal{V}\_{2,j^{\prime}}. Note that if Nj≤cm∗2​BN\_{j}\leq\dfrac{c^{\*}\_{m}}{2B}, then we obtain x∈𝒳m¯x\in{\mathcal{X}}\_{\bar{m}}, which implies that 𝒳m∗⊆𝒳¯m¯\mathcal{X}^{\*}\_{m}\subseteq\bar{\mathcal{X}}\_{\bar{m}}.

Analogously, assume that there exists some constant cm≥0c\_{m}\geq 0 such that

|  |  |  |
| --- | --- | --- |
|  | minx,j,j′⁡[(β1​j∗)⊤​x−(β1​j′∗)⊤​x]=cm∗​η,\displaystyle\min\_{x,j,j^{\prime}}\Big{[}(\beta^{\*}\_{1j})^{\top}x-(\beta^{\*}\_{1j^{\prime}})^{\top}x\Big{]}=c^{\*}\_{m}\eta, |  |

where the above minimum is subject to x∈𝒳¯m¯x\in\bar{\mathcal{X}}\_{\bar{m}}, i∈{m¯1,m¯2,…,m¯K¯}i\in\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\} and i′∈{m¯K¯+1,…,m¯k}i^{\prime}\in\{\bar{m}\_{\bar{K}+1},\ldots,\bar{m}\_{k}\}. Then, if Nj≤cm2​BN\_{j}\leq\dfrac{c\_{m}}{2B}, we have 𝒳¯m¯⊆𝒳m∗\bar{\mathcal{X}}\_{\bar{m}}\subseteq\mathcal{X}^{\*}\_{m}. Consequently, by setting Nj=12​B​min⁡{cm∗,cm}N\_{j}=\dfrac{1}{2B}\min\{c^{\*}\_{m},c\_{m}\}, we reach the conclusion that 𝒳¯m¯=𝒳m∗\bar{\mathcal{X}}\_{\bar{m}}=\mathcal{X}^{\*}\_{m}. Hence, the proof is completed.

### F.2 Proof of Proposition [6](#Thmproposition6 "Proposition 6. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")

To begin with, we show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | limε→0inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟5​((G1,G2),(G1∗,G2∗))≤ε𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]>0.\displaystyle\lim\_{\varepsilon\to 0}\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon}\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]>0. |  | (53) |

Suppose that the above inequality does not hold, then there exist a sequence of pairs of mixing measures (G1n,G2n)(G^{n}\_{1},G^{n}\_{2}) in 𝒢k1,k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta) given by G1n=∑i=1k1nωin​δ(κin,τin)G^{n}\_{1}=\sum\_{i=1}^{k^{n}\_{1}}\omega\_{i}^{n}\delta\_{(\kappa\_{i}^{n},\tau\_{i}^{n})} and G2n=∑i=1k2nexp⁡(β0​in)​δ(β1​in,ηin,νin)G^{n}\_{2}=\sum\_{i=1}^{k^{n}\_{2}}\exp(\beta^{n}\_{0i})\delta\_{(\beta^{n}\_{1i},\eta^{n}\_{i},\nu^{n}\_{i})} that satisfies 𝒟5​((G1n,G2n),(G1∗,G2∗))→0\mathcal{D}\_{5}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\to 0 and

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(s¯G1n,G2n(⋅|X),sG1∗,G2∗(⋅|X))]→0\displaystyle\mathbb{E}\_{X}[V(\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\to 0 |  |

as n→∞n\to\infty. According to the Fatou’s lemma, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 0\displaystyle 0 | =limn→∞𝔼X[V(s¯G1n,G2n(⋅|X),sG1∗,G2∗(⋅|X))]\displaystyle=\lim\_{n\to\infty}\mathbb{E}\_{X}[V(\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))] |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ≥12∫𝒳×𝒴lim infn→∞|s¯G1n,G2n(Y|X)−sG1∗,G2∗(Y|X)|d(X,Y),\displaystyle\geq\frac{1}{2}\int\_{\mathcal{X}\times\mathcal{Y}}\liminf\_{n\to\infty}|\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)|\mathrm{d}(X,Y), |  | (54) |

implying that s¯G1n,G2n​(Y|X)−sG1∗,G2∗​(Y|X)→0\bar{s}\_{G^{n}\_{1},G^{n}\_{2}}(Y|X)-s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). WLOG, we may assume that

|  |  |  |
| --- | --- | --- |
|  | max{m1,m2,…,mK}​∑j=1K|𝒱2,mj|=|𝒱2,1|+|𝒱2,2|+…+|𝒱2,K|.\displaystyle\max\_{\{m\_{1},m\_{2},\ldots,m\_{K}\}}\sum\_{j=1}^{K}|\mathcal{V}\_{2,m\_{j}}|=|\mathcal{V}\_{2,1}|+|\mathcal{V}\_{2,2}|+\ldots+|\mathcal{V}\_{2,K}|. |  |

Let X∈𝒳m∗X\in\mathcal{X}^{\*}\_{m}, where m∈[M]m\in[M] such that {m1,m2,…,mK}={1,2,…,K}\{m\_{1},m\_{2},\ldots,m\_{K}\}=\{1,2,\ldots,K\}. Since the Voronoi loss 𝒟5​((G1n,G2n),(G1∗,G2∗))\mathcal{D}\_{5}((G^{n}\_{1},G^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2})) goes to zero, it follows that β1​in→β1​j∗\beta^{n}\_{1i}\to\beta^{\*}\_{1j} as n→∞n\to\infty for any j∈[k2∗]j\in[k^{\*}\_{2}] and i∈𝒱2,ji\in\mathcal{V}\_{2,j}. By means of Lemma [3](#Thmlemma3 "Lemma 3. ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE"), we deduce X∈𝒳¯m¯X\in\bar{\mathcal{X}}\_{\bar{m}}, where m¯∈[q¯]\bar{m}\in[\bar{q}] such that {m¯1,m¯2,…,m¯K¯}=𝒱2,1∪𝒱2,2∪…∪𝒱2,K\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,1}\cup\mathcal{V}\_{2,2}\cup\ldots\cup\mathcal{V}\_{2,K}.
However, as K¯<∑j=1K|𝒱2,j|\bar{K}<\sum\_{j=1}^{K}|\mathcal{V}\_{2,j}|, the fact that {m¯1,m¯2,…,m¯K¯}=𝒱2,1∪𝒱2,2∪…∪𝒱2,K\{\bar{m}\_{1},\bar{m}\_{2},\ldots,\bar{m}\_{\bar{K}}\}=\mathcal{V}\_{2,1}\cup\mathcal{V}\_{2,2}\cup\ldots\cup\mathcal{V}\_{2,K} cannot occur. Thus, we obtain the result in equation ([53](#A6.E53 "In F.2 Proof of Proposition 6 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")). As a consequence, we can find a positive constant ε′\varepsilon^{\prime} such that

|  |  |  |
| --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟5​((G1,G2),(G1∗,G2∗))≤ε′𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))\leq\varepsilon^{\prime}}\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]>0. |  |

Given the above result, it is sufficient to show that

|  |  |  |  |
| --- | --- | --- | --- |
|  | inf(G1,G2)∈𝒢k1,k2​(Θ):𝒟5​((G1,G2),(G1∗,G2∗))>ε′𝔼X[V(s¯G1,G2(⋅|X),sG1∗,G2∗(⋅|X))]>0.\displaystyle\inf\_{(G\_{1},G\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta):\mathcal{D}\_{5}((G\_{1},G\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}}\mathbb{E}\_{X}[V(\bar{s}\_{G\_{1},G\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]>0. |  | (55) |

Assume by contrary that the inequality ([55](#A6.E55 "In F.2 Proof of Proposition 6 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")) does not hold, then we can find a sequence (G~1n,G~2n)∈𝒢k1,k2​(Θ)(\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2})\in\mathcal{G}\_{k\_{1},k\_{2}}(\Theta) such that 𝒟5​((G~1n,G~2n),(G1∗,G2∗))>ε′\mathcal{D}\_{5}((\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime} and

|  |  |  |
| --- | --- | --- |
|  | 𝔼X[V(s¯G~1n,G~2n(⋅|X),sG1∗,G2∗(⋅|X))]→0.\displaystyle\mathbb{E}\_{X}[V(\bar{s}\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(\cdot|X),s\_{G^{\*}\_{1},G^{\*}\_{2}}(\cdot|X))]\to 0. |  |

Again, by utilizing the Fatou’s lemma as in equation ([F.2](#A6.Ex530 "F.2 Proof of Proposition 6 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")), we get s¯G~1n,G~2n​(Y|X)−sG1∗,G2∗​(Y|X)→0\bar{s}\_{\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}}(Y|X)-s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X)\to 0 as n→∞n\to\infty for almost surely (X,Y)(X,Y). Since the parameter space Θ\Theta is compact, we can substitute the sequence (G~1n,G~2n)(\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}) with its subsequence which converges to some pair of mixing measures (G~1,G~2)(\tilde{G}\_{1},\tilde{G}\_{2}) in 𝒢k1,k2​(Θ)\mathcal{G}\_{k\_{1},k\_{2}}(\Theta). This result leads to s¯G~1,G~2​(Y|X)=sG1∗,G2∗​(Y|X)\bar{s}\_{\tilde{G}\_{1},\tilde{G}\_{2}}(Y|X)=s\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) for almost surely (X,Y)(X,Y). As the Top-KK sparse gating MoE is identifiable, we deduce (G~1,G~2)≡(G1∗,G2∗)(\tilde{G}\_{1},\tilde{G}\_{2})\equiv(G^{\*}\_{1},G^{\*}\_{2}), or equivalently, 𝒟5​((G~1,G~2),(G1∗,G2∗))=0\mathcal{D}\_{5}((\tilde{G}\_{1},\tilde{G}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))=0.
On the other hand, due to the fact that 𝒟5​((G~1n,G~2n),(G1∗,G2∗))>ε′\mathcal{D}\_{5}((\tilde{G}^{n}\_{1},\tilde{G}^{n}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime} for any n∈ℕn\in\mathbb{N}, we obtain 𝒟5​((G~1,G~2),(G1∗,G2∗))>ε′>0\mathcal{D}\_{5}((\tilde{G}\_{1},\tilde{G}\_{2}),(G^{\*}\_{1},G^{\*}\_{2}))>\varepsilon^{\prime}>0, which contradicts the previous result. Hence, we reach the result in equation ([55](#A6.E55 "In F.2 Proof of Proposition 6 ‣ Appendix F Extended Theoretical Results for Sparse Gating MoE")) and complete the proof.

## Appendix G Additional Experiments

In this appendix, we provide supplementary experimental results that reinforce and extend our theoretical analyses. In Appendix [G.1](#A7.SS1 "G.1 Numerical Experiments ‣ Appendix G Additional Experiments"), we illustrate the convergence properties of the maximum likelihood estimator (MLE) (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) towards the true mixing measure (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) using synthetic data, explicitly evaluating four theorem-based scenarios. Appendix [G.2](#A7.SS2 "G.2 Language Modeling ‣ Appendix G Additional Experiments") and Appendix [G.3](#A7.SS3 "G.3 Vision-Language Modeling ‣ Appendix G Additional Experiments") provide detailed training and validation performance curves during training of each model in language modeling and vision-language modeling, respectively.

### G.1 Numerical Experiments

![Refer to caption](/html/2505.10860/assets/x5.png)


(a) Theorem 1

![Refer to caption](/html/2505.10860/assets/x6.png)


(b) Theorem 2

![Refer to caption](/html/2505.10860/assets/x7.png)


(c) Theorem 3

![Refer to caption](/html/2505.10860/assets/x8.png)


(d) Theorem 4

Figure 5: Empirical illustration of the input - output relationship (X,Y)(X,Y) under synthetic conditions for each theoretical result. Each subplot corresponds to a different theoretical setting: (a) Theorem 1, (b) Theorem 2, (c) Theorem 3, and (d) Theorem 4.

#### G.1.1 Experimental Setup

Synthetic Data. For each sample size nn, we generate i.i.d samples {(Xi,Yi)}i=1n\{(X\_{i},Y\_{i})\}^{n}\_{i=1} by first sampling XiX\_{i}’s from the uniform distribution Uniform​[−3,3]\text{Uniform}[-3,3] and then sampling YiY\_{i}’s from the true conditional density fG1∗,G2∗​(Y|X)f\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) or gG1∗,G2∗​(Y|X)g\_{G^{\*}\_{1},G^{\*}\_{2}}(Y|X) of Gaussian mixture of experts (MoE) model setting of each theorem configuration. Figure [5](#A7.F5 "Figure 5 ‣ G.1 Numerical Experiments ‣ Appendix G Additional Experiments") shows the visualization of the relationship between XX and YY in each experiment.

Maximum Likelihood Estimation (MLE). A popular approach to determining the MLE (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) for each set of samples is to use the Expectation-Maximization (EM) algorithm [[16](#bib.bib16)]. However, since there are not any closed-form expressions for updating the gating parameters β0​i\beta\_{0i}, β1​i\beta\_{1i} in the maximization steps, we have to leverage an EM-based numerical scheme, which was previously used in [[5](#bib.bib5)]. We select the convergence criterion of ϵ=10−6\epsilon=10^{-6} and run a maximum of 1000 EM iterations.

Experiment Design.  Our empirical investigation systematically examines four experimental configurations, each precisely corresponding to the theoretical scenarios elaborated in our main paper. Each configuration includes 40 independent sample generations over a comprehensive range of sample sizes nn, specifically n∈[102,105]n\in[10^{2},10^{5}]. To ensure consistency and comparative clarity across experiments, we uniformly adopt an architecture consisting of one shared expert (k1∗=1k\_{1}^{\*}=1) complemented by two routed experts (k2∗=2k\_{2}^{\*}=2), where we fit two shared experts (k1=2k\_{1}=2) and three routed experts (k2=3k\_{2}=3) in our experiment settings.

#### G.1.2 Theorem 1

The problem setting is defined in Equation [1](#S2.E1 "In 2 On Shared Expert Strategy"), where we establish h1h\_{1} and h2h\_{2} to satisfy the identifiable experts condition, specifically h1​(x,(κ2,κ1,κ0)):=κ2​ReLU​(κ1⊤​x+κ0)h\_{1}(x,(\kappa\_{2},\kappa\_{1},\kappa\_{0})):=\kappa\_{2}\mathrm{ReLU}(\kappa\_{1}^{\top}x+\kappa\_{0}) and h2​(x,(η2,η1)):=η2​ReLU​(η1⊤​x)h\_{2}(x,(\eta\_{2},\eta\_{1})):=\eta\_{2}\mathrm{ReLU}(\eta\_{1}^{\top}x). The ground-truth parameters employed in our experiments are presented as follows:

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ω∗\displaystyle\omega^{\*} | =1.0,\displaystyle=1.0, | κ0∗\displaystyle\kappa\_{0}^{\*} | =0,\displaystyle=0, | κ1∗\displaystyle\kappa\_{1}^{\*} | =6,\displaystyle=6, | κ2∗\displaystyle\kappa\_{2}^{\*} | =−8,\displaystyle=-8, | τ∗\displaystyle\tau^{\*} | =0.25,\displaystyle=0.25, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β01∗\displaystyle\beta\_{01}^{\*} | =−0.5,\displaystyle=-0.5, | β11∗\displaystyle\beta\_{11}^{\*} | =5,\displaystyle=5, | η11∗\displaystyle\eta\_{11}^{\*} | =−12,\displaystyle=-12, | η21∗\displaystyle\eta\_{21}^{\*} | =4,\displaystyle=4, | ν1∗\displaystyle\nu\_{1}^{\*} | =0.4,\displaystyle=0.4, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β02∗\displaystyle\beta\_{02}^{\*} | =0.5,\displaystyle=0.5, | β12∗\displaystyle\beta\_{12}^{\*} | =5,\displaystyle=5, | η12∗\displaystyle\eta\_{12}^{\*} | =12,\displaystyle=12, | η22∗\displaystyle\eta\_{22}^{\*} | =4,\displaystyle=4, | ν2∗\displaystyle\nu\_{2}^{\*} | =0.4,\displaystyle=0.4, |  |

As illustrated in Figure [6(a)](#A7.F6.sf1 "In Figure 6 ‣ G.1.2 Theorem 1 ‣ G.1 Numerical Experiments ‣ Appendix G Additional Experiments"), the maximum likelihood estimator MLE (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) exhibits empirical convergence to the true mixing measure G∗G^{\*} under the Voronoi metric 𝒟1\mathcal{D}\_{1} (Equation [2.1](#S2.Ex5 "2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy")) at the rate of order 𝒪P​([log⁡(n)/n]0.451)\mathcal{O}\_{P}([\log(n)/n]^{0.451}). This empirically observed rate closely matches the theoretical parametric convergence rate 𝒪P​([log⁡(n)/n]1/2)\mathcal{O}\_{P}([\log(n)/n]^{1/2}) established in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy"), thereby validating the practical applicability of the theoretical result under strongly identifiable expert assumptions.

![Refer to caption](/html/2505.10860/assets/x9.png)


(a) Theorem 1

![Refer to caption](/html/2505.10860/assets/x10.png)


(b) Theorem 2

![Refer to caption](/html/2505.10860/assets/x11.png)


(c) Theorem 3

![Refer to caption](/html/2505.10860/assets/x12.png)


(d) Theorem 4

Figure 6: Log-log scaled plots illustrating simulation results with different model settings. The blue curves depict the mean discrepancy between the MLE (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) and the true mixing measure (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) accompanied by error bars representing the standard deviation over 40 times of experiments for each sample size nn. Additionally, an orange dash-dotted line represents the least-squares fitted linear regression line for these data points.

#### G.1.3 Theorem 2

In this experiment, we adopt the problem setting outlined in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy"). However, instead of using two-layer FFNs, we define h1h\_{1} and h2h\_{2} are linear experts as in Section [2.2](#S2.SS2 "2.2 Linear Experts ‣ 2 On Shared Expert Strategy"). Specifically, we set h1​(X,(κ1,κ0)):=κ1⊤​X+κ0h\_{1}(X,(\kappa\_{1},\kappa\_{0})):=\kappa\_{1}^{\top}X+\kappa\_{0} and h2​(X,(η1,η0)):=η1⊤​X+η0h\_{2}(X,(\eta\_{1},\eta\_{0})):=\eta\_{1}^{\top}X+\eta\_{0}, with the associated ground-truth parameters defined as follows:

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ω∗\displaystyle\omega^{\*} | =1.0,\displaystyle=1.0, | κ0∗\displaystyle\kappa\_{0}^{\*} | =0,\displaystyle=0, | κ1∗\displaystyle\kappa\_{1}^{\*} | =2,\displaystyle=2, | τ∗\displaystyle\tau^{\*} | =0.2,\displaystyle=0.2, |  | | |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β01∗\displaystyle\beta\_{01}^{\*} | =−0.5,\displaystyle=-0.5, | β11∗\displaystyle\beta\_{11}^{\*} | =5,\displaystyle=5, | η11∗\displaystyle\eta\_{11}^{\*} | =8,\displaystyle=8, | η01∗\displaystyle\eta\_{01}^{\*} | =2,\displaystyle=2, | ν1∗\displaystyle\nu\_{1}^{\*} | =0.4,\displaystyle=0.4, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β02∗\displaystyle\beta\_{02}^{\*} | =0.5,\displaystyle=0.5, | β12∗\displaystyle\beta\_{12}^{\*} | =5,\displaystyle=5, | η12∗\displaystyle\eta\_{12}^{\*} | =−6,\displaystyle=-6, | η02∗\displaystyle\eta\_{02}^{\*} | =1,\displaystyle=1, | ν2∗\displaystyle\nu\_{2}^{\*} | =0.4,\displaystyle=0.4, |  |

The result is shown in Figure [6(b)](#A7.F6.sf2 "In Figure 6 ‣ G.1.2 Theorem 1 ‣ G.1 Numerical Experiments ‣ Appendix G Additional Experiments"). Under linear experts settings and Voronoi metric 𝒟2\mathcal{D}\_{2} (Equation [2.2](#S2.Ex7 "2.2 Linear Experts ‣ 2 On Shared Expert Strategy")), the maximum likelihood estimator MLE has the convergence rate of 𝒪P​([log⁡(n)/n]1/2)\mathcal{O}\_{P}([\log(n)/n]^{1/2}). Notably, the linear expert settings make a perfect result with convergence rate of 𝒪P​([log⁡(n)/n]0.517)\mathcal{O}\_{P}([\log(n)/n]^{0.517}) where the noise in each sample size is minimal and uniform. This result strongly supports our theoretical result in Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 2.2 Linear Experts ‣ 2 On Shared Expert Strategy").

#### G.1.4 Theorem 3

This experiment is designed to empirically validate Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating") under the problem setting specified in Appendix [A](#A1 "Appendix A On Normalized Sigmoid Gating"), which employs normalized sigmoid gating. Under the sparse regime, we set all over-specified parameters β1​i∗\beta^{\*}\_{1i} equal to zero vectors. The expert functions follow the same structural assumptions as in Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.1 Strongly Identifiable Experts ‣ 2 On Shared Expert Strategy") where h1​(x,(κ2,κ1,κ0)):=κ2​ReLU​(κ1⊤​x+κ0)h\_{1}(x,(\kappa\_{2},\kappa\_{1},\kappa\_{0})):=\kappa\_{2}\mathrm{ReLU}(\kappa\_{1}^{\top}x+\kappa\_{0}) and h2​(x,(η2,η1)):=η2​ReLU​(η1⊤​x)h\_{2}(x,(\eta\_{2},\eta\_{1})):=\eta\_{2}\mathrm{ReLU}(\eta\_{1}^{\top}x). The complete set of ground-truth parameters used in this experiment is detailed below:

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ω∗\displaystyle\omega^{\*} | =1.0,\displaystyle=1.0, | κ0∗\displaystyle\kappa\_{0}^{\*} | =0,\displaystyle=0, | κ1∗\displaystyle\kappa\_{1}^{\*} | =6,\displaystyle=6, | κ2∗\displaystyle\kappa\_{2}^{\*} | =−8,\displaystyle=-8, | τ∗\displaystyle\tau^{\*} | =0.25,\displaystyle=0.25, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β01∗\displaystyle\beta\_{01}^{\*} | =−0.5,\displaystyle=-0.5, | β11∗\displaystyle\beta\_{11}^{\*} | =0,\displaystyle=0, | η11∗\displaystyle\eta\_{11}^{\*} | =−12,\displaystyle=-12, | η21∗\displaystyle\eta\_{21}^{\*} | =4,\displaystyle=4, | ν1∗\displaystyle\nu\_{1}^{\*} | =0.4,\displaystyle=0.4, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β02∗\displaystyle\beta\_{02}^{\*} | =0.5,\displaystyle=0.5, | β12∗\displaystyle\beta\_{12}^{\*} | =0,\displaystyle=0, | η12∗\displaystyle\eta\_{12}^{\*} | =12,\displaystyle=12, | η22∗\displaystyle\eta\_{22}^{\*} | =4,\displaystyle=4, | ν2∗\displaystyle\nu\_{2}^{\*} | =0.4,\displaystyle=0.4, |  |

Figure [6(c)](#A7.F6.sf3 "In Figure 6 ‣ G.1.2 Theorem 1 ‣ G.1 Numerical Experiments ‣ Appendix G Additional Experiments") presents the experimental results for the convergence analysis under the sparse regime utilizing normalized sigmoid gating. The maximum likelihood estimator (MLE) (G^1n,G^2n)(\widehat{G}^{n}\_{1},\widehat{G}^{n}\_{2}) empirically converges to the true mixing measure (G1∗,G2∗)(G^{\*}\_{1},G^{\*}\_{2}) at a rate of 𝒪P​([log⁡(n)/n]0.46)\mathcal{O}\_{P}([\log(n)/n]^{0.46}) under the Voronoi metric 𝒟3\mathcal{D}\_{3} (Equation [A.1](#A1.Ex16 "A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating")). This empirical convergence rate is closely aligned with the theoretical prediction articulated in Theorem [3](#Thmtheorem3 "Theorem 3. ‣ A.1 Sparse Regime ‣ Appendix A On Normalized Sigmoid Gating"). Consistent with the theorem’s implications, our experimental results suggest that under the sparse regime, normalized sigmoid gating does not exhibit significant advantages in terms of convergence speed compared to standard softmax gating mechanisms.

#### G.1.5 Theorem 4

In this experiment, we adopt the same problem setting with Theorem [D.4](#A4.SS4 "D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results") specified in Appendix [A](#A1 "Appendix A On Normalized Sigmoid Gating"). With sigmoid gating under the dense regime, we define shared expert function h1h\_{1} is strongly identifiable while the routed expert function h2h\_{2} is weakly identifiable. Specifically, h1h\_{1} is the two-layer FFNs function h1​(x,(κ2,κ1,κ0)):=κ2​ReLU​(κ1⊤​x+κ0)h\_{1}(x,(\kappa\_{2},\kappa\_{1},\kappa\_{0})):=\kappa\_{2}\mathrm{ReLU}(\kappa\_{1}^{\top}x+\kappa\_{0}) where h2h\_{2} is the linear experts h2​(X,(η1,η0)):=η1⊤​X+η0h\_{2}(X,(\eta\_{1},\eta\_{0})):=\eta\_{1}^{\top}X+\eta\_{0}. The complete set of ground-truth parameters used in this experiment is detailed below:

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ω∗\displaystyle\omega^{\*} | =1.0,\displaystyle=1.0, | κ0∗\displaystyle\kappa\_{0}^{\*} | =0,\displaystyle=0, | κ1∗\displaystyle\kappa\_{1}^{\*} | =6,\displaystyle=6, | κ2∗\displaystyle\kappa\_{2}^{\*} | =−8,\displaystyle=-8, | τ∗\displaystyle\tau^{\*} | =0.25,\displaystyle=0.25, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β01∗\displaystyle\beta\_{01}^{\*} | =−0.5,\displaystyle=-0.5, | β11∗\displaystyle\beta\_{11}^{\*} | =5,\displaystyle=5, | η11∗\displaystyle\eta\_{11}^{\*} | =8,\displaystyle=8, | η01∗\displaystyle\eta\_{01}^{\*} | =2,\displaystyle=2, | ν1∗\displaystyle\nu\_{1}^{\*} | =0.4,\displaystyle=0.4, |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | β02∗\displaystyle\beta\_{02}^{\*} | =0.5,\displaystyle=0.5, | β12∗\displaystyle\beta\_{12}^{\*} | =5,\displaystyle=5, | η12∗\displaystyle\eta\_{12}^{\*} | =−6,\displaystyle=-6, | η02∗\displaystyle\eta\_{02}^{\*} | =1,\displaystyle=1, | ν2∗\displaystyle\nu\_{2}^{\*} | =0.4,\displaystyle=0.4, |  |

Figure [6(d)](#A7.F6.sf4 "In Figure 6 ‣ G.1.2 Theorem 1 ‣ G.1 Numerical Experiments ‣ Appendix G Additional Experiments") presents the numerical results corresponding to Theorem [D.4](#A4.SS4 "D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results"). Under the dense regime, the Mixture-of-Experts (MoE) model achieves a convergence rate of 𝒪P​([log⁡(n)/n]0.552)\mathcal{O}\_{P}([\log(n)/n]^{0.552}), which closely aligns with the theoretical rate of 𝒪P​([log⁡(n)/n]1/2)\mathcal{O}\_{P}([\log(n)/n]^{1/2}). This empirical evidence substantiates Theorem [D.4](#A4.SS4 "D.4 Proof of Theorem 4 ‣ Appendix D Proof of Main Results"), suggesting that the use of normalized sigmoid gating contributes to improved sample efficiency in DeepSeekMoE.

### G.2 Language Modeling

![Refer to caption](/html/2505.10860/assets/x13.png)


Figure 7: Average performance (%) compared in pairs with Vanilla SMoE across three model settings over training steps on language modeling tasks. Left: Vanilla SMoE vs. DeepSeek-V3; Center: Vanilla SMoE vs. DeepSeek-V2; Right: Vanilla SMoE vs. SMoE Sigmoid Gating.

Figure [7](#A7.F7 "Figure 7 ‣ G.2 Language Modeling ‣ Appendix G Additional Experiments") presents a pairwise comparison between DeepSeek-V3, DeepSeek-V2, SMoE Sigmoid Gating, and the baseline Vanilla SMoE. Remarkably, across both model scales, by integrating normalized sigmoid gating into SMoE, SMoE Sigmoid Gating yields a substantial improvement in convergence rate compared to the softmax-gated baseline. Notably, in several training trajectories, SMoE Sigmoid Gating achieves a convergence rate comparable to that of DeepSeek-V2. For a more detailed examination, we provide the full training benchmark curves for both the 158M and 679M parameter language modeling settings in Figure [14](#A9.F14 "Figure 14 ‣ I.3 Training Time and Resource Allocation ‣ Appendix I Experimental Details") and Figure [15](#A9.F15 "Figure 15 ‣ I.3 Training Time and Resource Allocation ‣ Appendix I Experimental Details"), respectively.

### G.3 Vision-Language Modeling

![Refer to caption](/html/2505.10860/assets/x14.png)


Figure 8: Average performance (%) over training steps on vision-language pretraining tasks, comparing SMoE variants across three model configurations. Left: Full comparison among Vanilla SMoE, SMoE with Sigmoid Gating, DeepSeek-V2, and DeepSeek-V3; Right: Focused comparison between Vanilla SMoE and SMoE Sigmoid Gating.

Figure [7](#A7.F7 "Figure 7 ‣ G.2 Language Modeling ‣ Appendix G Additional Experiments") presents a pairwise comparison among DeepSeek-V3, DeepSeek-V2, SMoE with Sigmoid Gating, and the baseline Vanilla SMoE. On vision-language pretraining tasks, SMoE Sigmoid Gating exhibits a comparable convergence rate and final performance to the Vanilla SMoE. However, similar to the DeepSeek variants, it demonstrates faster convergence during the later stages of training and achieves greater training stability. To facilitate a finer-grained analysis, we provide benchmark-specific performance trajectories in Figure [16](#A9.F16 "Figure 16 ‣ I.3 Training Time and Resource Allocation ‣ Appendix I Experimental Details").

## Appendix H Additional Router Analysis

In this appendix, we provide further analyses regarding router behavior. Formal definitions, equations, and detailed discussions on router saturation and router change rate are provided in Appendix [H.1](#A8.SS1 "H.1 Router Saturation ‣ Appendix H Additional Router Analysis") and Appendix [H.2](#A8.SS2 "H.2 Router Change Rate ‣ Appendix H Additional Router Analysis"), respectively. Additionally, an in-depth analysis of expert utilization is included in Appendix [H.3](#A8.SS3 "H.3 Expert Utilization ‣ Appendix H Additional Router Analysis"). For consistency, all analyses utilize the same ordered set of the 6000 most frequent tokens from the validation dataset.

### H.1 Router Saturation

![Refer to caption](/html/2505.10860/assets/x15.png)


Figure 9: Router saturation across layers for 158M-parameter models in language modeling tasks. We compute saturation by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

In formal terms, router saturation is the proportion of expert activations at some intermediary checkpoint at time tt that matches the expert IDs activated at some final checkpoint TT over the same dataset:

|  |  |  |
| --- | --- | --- |
|  | Router Saturation​(t)=1N​∑i=1N|ℰi(t)∩ℰi(T)|k,\displaystyle\text{Router Saturation}(t)=\frac{1}{N}\sum\_{i=1}^{N}\frac{\left|\mathcal{E}\_{i}^{(t)}\cap\mathcal{E}\_{i}^{(T)}\right|}{k}, |  |

where:

* •

  NN: The total number of tokens in the dataset.
* •

  kk: The number of top-k experts activated per input token.
* •

  ℰi(t)\mathcal{E}\_{i}^{(t)}: The set of kk experts activated for the ii-th token at the tt-th checkpoint.
* •

  ℰi(T)\mathcal{E}\_{i}^{(T)}: The set of kk experts activated for the ii-th token at the final checkpoint TT.
* •

  |ℰi(t)∩ℰi(T)|\left|\mathcal{E}\_{i}^{(t)}\cap\mathcal{E}\_{i}^{(T)}\right|: The number of common experts activated for the ii-th token between the tt-th and final checkpoints.

Router saturation provides a quantitative measure of how early the routing decisions converge during training. A saturation value of 100% indicates that the router at an intermediate checkpoint routes to the same set of experts as at the final checkpoint. High saturation values at early checkpoints reflect early convergence in expert selection, indicating that the router has rapidly settled into a stable assignment pattern. In contrast, low saturation values suggest ongoing exploration or adaptation in expert allocations, signaling that the routing mechanism is still undergoing significant adjustments.

Figure [9](#A8.F9 "Figure 9 ‣ H.1 Router Saturation ‣ Appendix H Additional Router Analysis") and Figure [11](#A8.F11 "Figure 11 ‣ H.2 Router Change Rate ‣ Appendix H Additional Router Analysis") show the detailed router saturation for each layer with 158M and 679M parameters, respectively. The result shows that the later layer tends to saturate earlier during training, where layer 0 is an outlier and saturates significantly slower than the others. Additionally, we observe that in shared layer settings (DeepSeek-V2 and DeepSeek-V3), the gap between saturation of different layers is smaller than SMoE and SMoE Sigmoid Gating. When comparing the MoE model with normalized sigmoid gating and softmax gating, we can easily observe that the model with normalized sigmoid gating exhibits a more uniform saturation rate across layers compared to the model with softmax gating. This observation further highlights the effectiveness of normalized sigmoid gating in mixture-of-experts model.

### H.2 Router Change Rate

![Refer to caption](/html/2505.10860/assets/x16.png)


Figure 10: Router change rate across layers for 158M-parameter models in language modeling tasks. We compute router change rate by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

Router Change Rate is a metric that measures the stability of the gating of mixture of experts models. This metric directly quantifies the gating fluctuation between two consecutive checkpoints:

|  |  |  |
| --- | --- | --- |
|  | Router Change Rate​(t)=1N​∑i=1N|ℰi(t+1)\ℰi(i)|k,\displaystyle\text{Router Change Rate}(t)=\frac{1}{N}\sum\_{i=1}^{N}\frac{\left|\mathcal{E}\_{i}^{(t+1)}\backslash\mathcal{E}\_{i}^{(i)}\right|}{k}, |  |

Where:

* •

  NN: The total number of tokens in the dataset.
* •

  kk: The number of top-k experts activated per input token.
* •

  ℰi(t)\mathcal{E}\_{i}^{(t)}: The set of kk experts activated for the ii-th token at the tt-th checkpoint.
* •

  ℰi(T)\mathcal{E}\_{i}^{(T)}: The set of kk experts activated for the ii-th token at the (t+1)(t+1)-th checkpoint.
* •

  |ℰi(t)\ℰi(T)|\left|\mathcal{E}\_{i}^{(t)}\backslash\mathcal{E}\_{i}^{(T)}\right|: The number of non-intersecting experts activated for the ii-th token between the (t+1)(t+1)-th and the tt-th checkpoint

![Refer to caption](/html/2505.10860/assets/x17.png)


Figure 11: Router saturation across layers for 679M-parameter models in language modeling tasks. We compute saturation by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

![Refer to caption](/html/2505.10860/assets/x18.png)


Figure 12: Router change rate across layers for 679M-parameter models in language modeling tasks. We compute router change rate by comparing the routing to the top-8 experts with SMoE and SMoE Sigmoid Gating, and the top-6 experts with DeepSeek variants.

Router Change Rate is a quantitative metric to measure the stability of routing mechanism in Mixture-of-Experts (MoE) during training. Unlike router saturation, which assesses convergence towards a final routing decision, the router change rate evaluates fluctuations between consecutive checkpoints. A low router change rate indicates stable routing decisions across training intervals, implying that the gating mechanism has achieved consistent expert assignments, minimizing disruptions and promoting steady specialization of experts. Conversely, a high router change rate suggests volatility in routing decisions, reflecting ongoing exploration or adjustment, potentially introducing training inefficiencies and hindering expert specialization. Thus, monitoring the router change rate provides valuable insights into the dynamics of expert allocation stability, enabling deeper understanding and optimization of the routing strategy in MoE architectures.

Figure [9](#A8.F9 "Figure 9 ‣ H.1 Router Saturation ‣ Appendix H Additional Router Analysis") and Figure [11](#A8.F11 "Figure 11 ‣ H.2 Router Change Rate ‣ Appendix H Additional Router Analysis") show the detailed router change rate for each layer with 158M and 679M parameters, respectively. Similar to router saturation, later layers show more stability with lower router change rate. However, the router change rate between layers show more consistency compared to router saturation. Layer 0 still has some differences in router change rate, the difference with other layers is still not too large, which show that although layer 0 saturates significantly slower, it still keep the stability of during training. When comparing between different model settings, the model with normalized sigmoid gating (SMoE Sigmoid Gating and DeepSeek-V3) shows lower and more consistent router change rate compared with the model with traditional softmax gating (SMoE, DeepSeek-V2).

### H.3 Expert Utilization

![Refer to caption](/html/2505.10860/assets/x19.png)


Figure 13: Jain’s Fairness Index across MoE layers for language-modeling tasks with 158 M (left) and 679 M (right) parameter models.

To quantify the fairness of expert utilization in the Mixture-of-Experts (MoE) model, we apply Jain’s Fairness Index to the router’s resource allocation across nn experts. Let R=(r1,r2,…,rn)R=(r\_{1},r\_{2},\dots,r\_{n}) denote the utilization vector, where ri≥0r\_{i}\geq 0 represents the proportion of input tokens (or total routing weight) assigned to expert ii over a given evaluation window. The Jain’s Fairness Index J​(R)J(R) is computed as:

|  |  |  |
| --- | --- | --- |
|  | J​(R)=J​(r1,r2,…,rn)=(∑i=1nri)2n​∑i=1nri2,\displaystyle J(R)=J(r\_{1},r\_{2},...,r\_{n})=\frac{(\sum\_{i=1}^{n}r\_{i})^{2}}{n\sum\_{i=1}^{n}r\_{i}^{2}}, |  |

This index ranges from [1/n,1][1/n,1], where J​(R)=1J(R)=1 indicates perfectly uniform expert usage, (i.e., all experts are used equally), where J​(R)=1/nJ(R)=1/n signifies complete imbalance, with only one expert active. Thus, higher values of J​(R)J(R) correspond to fairer and more evenly distributed expert selection.

Figure [13](#A8.F13 "Figure 13 ‣ H.3 Expert Utilization ‣ Appendix H Additional Router Analysis") presents a comparison of Jain’s Fairness Index [[29](#bib.bib29)] across different Mixture-of-Experts (MoE) model configurations and scales. Across both 158M and 679M parameter models, all configurations exhibit a consistent pattern: fairness in expert utilization is highest in the initial layers and declines in subsequent layers, suggesting that earlier layers facilitate broader expert utilization. Notably, models employing normalized sigmoid gating (SMoE Sigmoid Gating and DeepSeek-V3) maintain a higher fairness index, especially in the later layers, indicating better expert utilization. These results highlight the efficacy of normalized sigmoid gating in promoting more balanced expert utilization throughout the network.

## Appendix I Experimental Details

### I.1 Language Modeling

#### I.1.1 Datasets

SlimPajama. The SlimPajama [[66](#bib.bib66)] dataset is a filtered and deduplicated corpus of the 1.2T token RedPajama dataset [[75](#bib.bib75)] designed for language model pretraining. It contains around 627B tokens across diverse sources.

LAMBADA. The LAMBADA [[58](#bib.bib58)] dataset evaluates a model’s ability to predict the final word of a passage, requiring understanding of broad discourse context. Each instance comprises a narrative where the target word is only predictable when considering the entire passage, challenging models to perform deep contextual comprehension beyond sentence-level cues

BLiMP. The Benchmark of Linguistic Minimal Pairs (BLiMP) [[74](#bib.bib74)] assesses language models’ grasp of English grammar through 67 sub-datasets, each containing 1,000 minimal pairs. These pairs differ subtly to test specific syntactic, morphological, or semantic phenomena, enabling fine-grained evaluation of linguistic competence

Children’s Book Test (CBT). CBT [[24](#bib.bib24)] measures a model’s ability to utilize wider linguistic context by providing passages from children’s books with a missing word to predict. The dataset distinguishes between predicting syntactic function words and semantically rich content words, emphasizing the importance of context in language understanding

HellaSwag. HellaSwag [[81](#bib.bib81)] challenges models with sentence completion tasks that require commonsense reasoning. Each instance presents a context and multiple plausible continuations, with only one being correct. The dataset is adversarially filtered to be trivial for humans but difficult for models, highlighting gaps in machine commonsense understanding.

PIQA. The Physical Interaction Question Answering (PIQA) [[3](#bib.bib3)] dataset tests models on physical commonsense reasoning. It comprises questions about everyday tasks, requiring knowledge of physical properties and affordances, challenging models to reason about the physical world without direct sensory experience.

ARC-Challenge. The AI2 Reasoning Challenge (ARC) [[11](#bib.bib11)] presents grade-school level multiple-choice science questions that necessitate reasoning and external knowledge. The Challenge set includes questions that are particularly difficult for models, serving as a benchmark for advanced question-answering capabilities .

OpenBookQA. OpenBookQA [[35](#bib.bib35)] consists of multiple-choice questions derived from a curated set of science facts, resembling open-book exams. Answering requires combining the provided facts with external commonsense knowledge, testing a model’s ability to integrate information from multiple sources.

RACE. The Reading Comprehension Dataset from Examinations (RACE) [[63](#bib.bib63)] contains passages and questions from English exams for Chinese middle and high school students. With nearly 100,000 questions, it evaluates a model’s reading comprehension and reasoning skills across diverse topics.

SIQA. Social IQa (SIQA) [[63](#bib.bib63)] focuses on social commonsense reasoning, presenting questions about everyday social interactions. Models must infer motivations, reactions, and social dynamics, challenging their understanding of human social behavior.

CommonSenseQA. CommonSenseQA [[68](#bib.bib68)] is a multiple-choice question-answering dataset that requires models to apply commonsense knowledge. Each question is designed to probe a specific aspect of commonsense reasoning, with distractor answers carefully crafted to be plausible yet incorrect.

#### I.1.2 Model Settings, Training Settings and Evaluation

Table 4: Comprehensive Model Configurations for Experimental Evaluation. SMoE refers to settings applied for both Vanilla SMoE and SMoE Sigmoid Gating, whereas DeepSeek corresponds to configurations used for DeepSeek-V2 and DeepSeek-V3 models.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |  | | --- | | Scale | | |  | | --- | | Model | | |  | | --- | | # params | | |  | | --- | | # act. | | params | | |  | | --- | | # trained | | tokens | | |  | | --- | | dmodeld\_{\mathrm{model}} | | |  | | --- | | H | | |  | | --- | | dheadd\_{\mathrm{head}} | | |  | | --- | | NEN\_{E} | | |  | | --- | | KrK\_{r} | | |  | | --- | | NsN\_{s} | | |  | | --- | | Expert | | dim | | |  | | --- | | NwarmupN\_{\mathrm{warmup}} | | |  | | --- | | κ\kappa | |
| Small | SMoE | 158M | 36M | 6.5B | 512 | 4 | 82 | 66 | 8 | 0 | 128 | 0 | 0.1 |
| DeepSeek | 158M | 36M | 6.5B | 512 | 4 | 82 | 64 | 6 | 2 | 128 | 0 | 0.1 |
| Large | SMoE | 679M | 131M | 26.2B | 1024 | 4 | 128 | 66 | 8 | 0 | 256 | 4000 | 0.25 |
| DeepSeek | 679M | 131M | 26.2B | 1024 | 4 | 128 | 64 | 6 | 2 | 256 | 4000 | 0.25 |

Training datasets.
We conduct the experiments on language modeling using the popular SLimPajama [[66](#bib.bib66)] dataset.
Due to the limited computational resource, we utilize only subsets of the SlimPajama [[66](#bib.bib66)] dataset containing 6.5B and 26.2B tokens to train our 158M and 679M parameter models, respectively.

Model Settings. Table [4](#A9.T4 "Table 4 ‣ I.1.2 Model Settings, Training Settings and Evaluation ‣ I.1 Language Modeling ‣ Appendix I Experimental Details") summarizes the comprehensive set of hyperparameters and configurations for both scales and the two model variants evaluated in our experiments. All models employ a total of Nr=66N\_{r}=66 experts. For routing schemes, the baseline SMoE utilizes a top-8 expert routing strategy (Kr=8K\_{r}=8), while the DeepSeek variants adopt a mixed routing approach comprising top-6 expert selection (Kr=6K\_{r}=6) plus Ns=2N\_{s}=2 shared experts. To align with the fine-grained expert segmentation proposed in DeepSeekMoE [[12](#bib.bib12)], we set the expert dimensionality to 1/4​dm​o​d​e​l1/4\ d\_{model} and increase the expert count to 66 instead of the common settings with 16 experts. Additionally, the number of attention heads is uniformly set to H=4H=4 across both model scales. All models leverage Rotary Positional Embedding (RoPE) [[67](#bib.bib67)], PyTorch’s optimized attention implementation, and employ pre-layernorm Transformers. To ensure balanced expert utilization, we use the standard load balancing loss defined in Switch Transformers [[21](#bib.bib21)].

Training Settings. All models are trained in PyTorch using a batch size of 6464, context length of 10241024, and a learning rate of 2.5​e−42.5e-4. We apply 40004000 linear warm-up steps specifically for the larger-scale models and utilize the AdamW optimizer [[46](#bib.bib46)] with its default hyperparameters and a weight decay of 0.010.01. Gradient clipping is performed with threshold κ\kappa, and the precise number of linear warm-up steps (Nw​a​r​m​u​pN\_{warmup}) per model variant is provided in Table [4](#A9.T4 "Table 4 ‣ I.1.2 Model Settings, Training Settings and Evaluation ‣ I.1 Language Modeling ‣ Appendix I Experimental Details"). We tokenize the input using SentencePiece [[33](#bib.bib33)], configured with a vocabulary size of 8000 tokens, which is trained on a representative subset of the SlimPajama dataset [[66](#bib.bib66)].

Evaluation. We evaluate our model with the Perplexity score (PPL) and zero-shot performance with nine different downstream tasks: LAMBADA [[58](#bib.bib58)], BLiMP [[74](#bib.bib74)], Children’s Book Test [[24](#bib.bib24)], HellaSwag [[81](#bib.bib81)], PIQA[[3](#bib.bib3)], ARC-Challenge [[11](#bib.bib11)], RACE [[35](#bib.bib35)], SIQA [[63](#bib.bib63)] and CommonSenseQA [[68](#bib.bib68)]. For LAMBADA, we use the detokenized version from OpenAI, and we evaluate the top-1 accuracy of the last word (it can span multiple tokens; here we use greedy decoding). For CBT, BLiMP, and RACE, we measure the accuracy of each task and report the average accuracy of the tasks.

Compute Resource. All models are trained and evaluated on a single node equipped with 4 NVIDIA A100 80GB CoWoS HBM2e PCIe 4.0 employing data-parallelism.

### I.2 Vision Language Modeling

#### I.2.1 Datasets

LLaVA-558K. The LLaVA 558K [[44](#bib.bib44)] dataset is a curated subset of 558,000 image-text pairs derived from the LAION/CC/SBU dataset. It is designed for the pretraining stage of visual instruction tuning, facilitating the alignment between visual and language modalities. This dataset includes BLIP-generated captions and synthetic multimodal conversations, serving as a foundational resource for training models like LLaVA towards enhanced vision-language capabilities.

ALLaVA. ALLaVA [[6](#bib.bib6)] is a large-scale synthetic dataset comprising approximately 1.3 million samples, generated using GPT-4V. It includes fine-grained image annotations and complex reasoning visual question-answering pairs. The dataset aims to bridge the performance gap between traditional large vision-language models and more resource-efficient lite versions by providing high-quality training data for visual instruction tuning.

LLaVA-665K. The LLaVA-665K [[42](#bib.bib42)] dataset is an expanded and refined version of the original LLaVA instruction tuning dataset, containing 665,000 multimodal instruction-following samples. It integrates diverse sources such as VQAv2, GQA, OCR-VQA, and RefCOCO, among others, to enhance the model’s performance across various vision-language tasks. This comprehensive dataset supports improved visual instruction tuning for models like LLaVA-1.5 [[42](#bib.bib42)].

AI2D. The AI2D (AI2 Diagrams) [[31](#bib.bib31)] dataset comprises over 5000 grade school science diagrams, annotated with more than 150,000 rich annotations and over 15000 corresponding multiple-choice questions. It serves as a resource for evaluating models’ abilities in diagram understanding and visual reasoning within educational contexts.

MMStar. MMStar [[7](#bib.bib7)] is a meticulously curated benchmark designed to evaluate large vision-language models (LVLMs) on vision-indispensable tasks. It includes 1,500 samples across six core capabilities and 18 detailed axes, ensuring each sample necessitates visual understanding and minimizes data leakage.

POPE. The POPE (Polling-based Object Probing Evaluation) [[40](#bib.bib40)] dataset is developed to assess object hallucination in LVLMs. It provides a systematic approach to evaluate the consistency of object descriptions generated by models, highlighting tendencies to generate objects not present in the input images.

ScienceQA. ScienceQA [[47](#bib.bib47)] is a large-scale multimodal dataset featuring science questions enriched with lectures and explanations. It spans diverse subjects, including natural science, language science, and social science, aiming to evaluate models’ abilities in multimodal reasoning and explanatory question answering.

TextVQA. The TextVQA [[65](#bib.bib65)] dataset focuses on visual question answering tasks that require reading and reasoning about text within images. It contains 45,336 questions over 28,408 images, challenging models to integrate textual and visual information effectively.

GQA. GQA (Graph Question Answering) [[27](#bib.bib27)] is a large-scale dataset designed for real-world visual reasoning and compositional question answering. It includes 22 million questions based on 113,000 images, each accompanied by scene graphs detailing objects, attributes, and relationships, facilitating structured reasoning evaluations.

MME-RealWorld-Lite. MME-RealWorld-Lite [[83](#bib.bib83)] is a streamlined version of the MME-RealWorld benchmark, offering a subset of 50 samples per task to accelerate inference. It maintains the benchmark’s focus on evaluating multimodal models in real-world scenarios with high-resolution images and complex tasks.

MMMU Pro. MMMU Pro [[78](#bib.bib78)] is an enhanced benchmark for assessing multimodal models’ understanding and reasoning across multiple disciplines. It filters out questions answerable by text-only models, augments candidate options, and introduces vision-only input settings, thereby rigorously evaluating models’ true multimodal capabilities.

OCRBench. OCRBench [[45](#bib.bib45)] is a comprehensive evaluation benchmark for optical character recognition (OCR) capabilities in large multimodal models. It encompasses 29 datasets covering tasks like text recognition, scene text-centric VQA, document-oriented VQA, key information extraction, and handwritten mathematical expression recognition, providing a thorough assessment of OCR performance.

#### I.2.2 Model Settings, Training Settings and Evaluation

Model Settings. We embrace the vision-language pre-training task [[42](#bib.bib42)], a challenging problem setting that enables effective model training with relatively limited data. We adopt the experiment setting in LIBMoE [[57](#bib.bib57)] with LLaVA architecture [[44](#bib.bib44)], which includes three modules: pre-trained Large Language Model, pre-trained visual encoder, and randomly initialized MLP connector. We employ the pre-trained SigLIP (Patch14-224) [[82](#bib.bib82)] as the vision encoder, pre-trained Phi-3.5-mini-instruct [[1](#bib.bib1)] as the LLM, and a randomly initialized MLP connector. In the Visual Instruction Tuning (VIT) stage, we adopt a sparse upcycling approach [[32](#bib.bib32)] and upcycle only the MLP Connector into 8 experts, employing a top-4 expert routing strategy, while the DeepSeek variants adopt a top-3 expert routing scheme with an additional shared expert. Thus, our model has approximately 4.4B parameters.

Training Settings. We follow LIBMoE [[57](#bib.bib57)] for the training settings. Specifically, our training recipe with three stages of training: pre-training, pre-finetuning, and Visual Instruction Tuning (VIT). In the first stage, we only pretrain the MLP connector for better alignment using LLaVA 558K dataset [[44](#bib.bib44)]. During the second pre-finetuning stage, we train all parameters using high-quality caption data with the ALLaVA [[6](#bib.bib6)] dataset with 708K samples, aiming to warm up the entire model. In the third stage, we upcycle the MLP Connector to MoE block and trained on visual instruction tuning data (a subset of LLaVA-665K [[42](#bib.bib42)] with 332K samples). The learning rate is set to 1​e−31e-3 for pre-training the MLP connector and reduced to 2​e−62e-6 for pre-finetuning and 4​e−64e-6 for the final stage. All models are trained in PyTorch using a batch size of 4 and AdamW optimizer [[46](#bib.bib46)] with its default hyperparameters. We use Zero Redundancy Optimizer (ZeRO) [[61](#bib.bib61)] for memory optimization with Zero2 for the first stage and Zero3 for both pre-finetuning and VIT stages.

Evaluation. Our model is evaluated under the zero-shot setting across a diverse set of benchmarks encompassing various vision-language capabilities, such as perception, reasoning, OCR, instruction following, etc. The benchmarks considered include AI2D [[31](#bib.bib31)], MMStar [[7](#bib.bib7)], POPE [[40](#bib.bib40)], ScienceQA [[47](#bib.bib47)], TextVQA [[65](#bib.bib65)], GQA [[27](#bib.bib27)], MME-RealWorld-Lite [[83](#bib.bib83)], MMMU Pro [[78](#bib.bib78)], OCRBench [[45](#bib.bib45)].

Compute Resource. All models are trained and evaluated on a single node equipped with 4 NVIDIA A100 80GB CoWoS HBM2e PCIe 4.0 employing data-parallelism.

### I.3 Training Time and Resource Allocation

Table [5](#A9.T5 "Table 5 ‣ I.3 Training Time and Resource Allocation ‣ Appendix I Experimental Details") summarizes the training time and resource utilization across all experimental settings.

Table 5: Training Time and GPU Resource Allocation Across All Experimental Settings.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | | | |  | | --- | | Training Time | | (hours) | | Resourse |
| |  | | --- | | Vision Language | | Modeling | | Pre-Training | | 5.5 | 4xA100 |
| Pre-FineTuning | | 18 | 4xA100 |
| |  | | --- | | Visual Instruction | | Tuning | | SMoE | 10 | 4xA100 |
| SMoE Sigmoid Gating | 10 | 4xA100 |
| DeepSeek-V2 | 10.5 | 4xA100 |
| DeepSeek-V3 | 10.5 | 4xA100 |
| |  | | --- | | Language | | Modeling | | 158M parametes | SMoE | 9.5 | 4xA100 |
| SMoE Sigmoid Gating | 10 | 4xA100 |
| DeepSeek-V2 | 10.5 | 4xA100 |
| DeepSeek-V3 | 10.5 | 4xA100 |
| 679M parametes | SMoE | 65 | 4xA100 |
| SMoE Sigmoid Gating | 65 | 4xA100 |
| DeepSeek-V2 | 71 | 4xA100 |
| DeepSeek-V3 | 71.5 | 4xA100 |

![Refer to caption](/html/2505.10860/assets/x20.png)


Figure 14: Benchmark curves during training in language modeling tasks for models with 158M parameters.

![Refer to caption](/html/2505.10860/assets/x21.png)


Figure 15: Benchmark curves during training in language modeling tasks for models with 679M parameters.

![Refer to caption](/html/2505.10860/assets/x22.png)


Figure 16: Benchmark curves during training in vision-language modeling tasks.

## References

* [1]

  M. Abdin, J. Aneja, H. Awadalla, A. Awadallah, A. A. Awan, N. Bach, A. Bahree,
  A. Bakhtiari, J. Bao, H. Behl, et al.
  Phi-3 technical report: A highly capable language model locally on
  your phone.
  arXiv preprint arXiv:2404.14219, 2024.
* [2]

  P. Akbarian, H. Nguyen, X. Han, and N. Ho.
  Quadratic gating functions in mixture of experts: A statistical
  insight.
  arXiv preprint arXiv:2410.11222, 2024.
* [3]

  Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al.
  Piqa: Reasoning about physical commonsense in natural language.
  In Proceedings of the AAAI conference on artificial
  intelligence, 2020.
* [4]

  J. S. O. Ceron, G. Sokar, T. Willi, C. Lyle, J. Farebrother, J. N. Foerster,
  G. K. Dziugaite, D. Precup, and P. S. Castro.
  Mixtures of experts unlock parameter scaling for deep RL.
  In Forty-first International Conference on Machine Learning,
  2024.
* [5]

  F. Chamroukhi, A. Samé, G. Govaert, and P. Aknin.
  Time series modeling by a regression approach based on a latent
  process.
  Neural Networks, 22(5-6):593–602, 2009.
* [6]

  G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li,
  X. Wan, and B. Wang.
  Allava: Harnessing gpt4v-synthesized data for lite vision-language
  models.
  arXiv preprint arXiv:2402.11684, 2024.
* [7]

  L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao,
  D. Lin, et al.
  Are we on the right way for evaluating large vision-language models?
  arXiv preprint arXiv:2403.20330, 2024.
* [8]

  Z. Chen, Y. Deng, Y. Wu, Q. Gu, and Y. Li.
  Towards understanding the mixture-of-experts layer in deep learning.
  In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh,
  editors, Advances in Neural Information Processing Systems, volume 35,
  pages 23049–23062. Curran Associates, Inc., 2022.
* [9]

  Z. Chi, L. Dong, S. Huang, D. Dai, S. Ma, B. Patra, S. Singhal, P. Bajaj,
  X. Song, X.-L. Mao, H. Huang, and F. Wei.
  On the representation collapse of sparse mixture of experts.
  In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022.
* [10]

  Y. Chow, A. Tulepbergenov, O. Nachum, D. Gupta, M. Ryu, M. Ghavamzadeh, and
  C. Boutilier.
  A Mixture-of-Expert Approach to RL-based Dialogue
  Management.
  In The Eleventh International Conference on Learning
  Representations, 2023.
* [11]

  P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and
  O. Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning
  challenge.
  arXiv preprint arXiv:1803.05457, 2018.
* [12]

  D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu,
  Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang.
  Deepseekmoe: Towards ultimate expert specialization in
  mixture-of-experts language models.
  arXiv preprint arXiv:2401.04088, 2024.
* [13]

  D. Dai, L. Dong, S. Ma, B. Zheng, Z. Sui, B. Chang, and F. Wei.
  Stablemoe: Stable routing strategy for mixture of experts.
  arXiv preprint arXiv:2204.08396, 2022.
* [14]

  DeepSeek-AI et al.
  Deepseek-v2: A strong, economical, and efficient mixture-of-experts
  language model.
  arXiv preprint arXiv:2405.04434, 2024.
* [15]

  DeepSeek-AI et al.
  Deepseek-v3 technical report.
  arXiv preprint arXiv:2412.19437, 2024.
* [16]

  A. P. Dempster, N. M. Laird, and D. B. Rubin.
  Maximum likelihood from incomplete data via the em algorithm.
  Journal of the royal statistical society: series B
  (methodological), 39(1):1–22, 1977.
* [17]

  N. T. Diep, H. Nguyen, C. Nguyen, M. Le, D. M. H. Nguyen, D. Sonntag,
  M. Niepert, and N. Ho.
  On zero-initialized attention: Optimal prompt and gating factor
  estimation.
  arXiv preprint arXiv:2502.03029, 2025.
* [18]

  N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou,
  A. Yu, O. Firat, B. Zoph, L. Fedus, M. Bosma, Z. Zhou, T. Wang, E. Wang,
  K. Webster, M. Pellat, K. Robinson, K. Meier-Hellstern, T. Duke, L. Dixon,
  K. Zhang, Q. Le, Y. Wu, Z. Chen, and C. Cui.
  Glam: Efficient scaling of language models with mixture-of-experts.
  In ICML, 2022.
* [19]

  S. Faria and G. Soromenho.
  Fitting mixtures of linear regressions.
  Journal of Statistical Computation and Simulation,
  80(2):201–225, 2010.
* [20]

  W. Fedus, B. Zoph, and N. Shazeer.
  Switch transformers: Scaling to trillion parameter models with simple
  and efficient sparsity.
  Journal of Machine Learning Research, 23(120):1–39, 2022.
* [21]

  W. Fedus, B. Zoph, and N. Shazeer.
  Switch transformers: Scaling to trillion parameter models with simple
  and efficient sparsity.
  Journal of Machine Learning Research, 23:1–39, 2022.
* [22]

  A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle,
  A. Letman, A. Mathur, et al.
  The llama 3 herd of models.
  arXiv preprint arXiv:2407.21783, 2024.
* [23]

  X. Han, H. Nguyen, C. Harris, N. Ho, and S. Saria.
  Fusemoe: Mixture-of-experts transformers for fleximodal fusion.
  In Advances in Neural Information Processing Systems, 2024.
* [24]

  F. Hill, A. Bordes, S. Chopra, and J. Weston.
  The goldilocks principle: Reading children’s books with explicit
  memory representations.
  arXiv preprint arXiv:1511.02301, 2015.
* [25]

  N. Ho and X. Nguyen.
  Convergence rates of parameter estimation for some weakly
  identifiable finite mixtures.
  The Annals of Statistics, 44(6):2726 – 2755, 2016.
  Publisher: Institute of Mathematical Statistics and Bernoulli
  Society.
* [26]

  N. Ho, C.-Y. Yang, and M. I. Jordan.
  Convergence rates for Gaussian mixtures of experts.
  Journal of Machine Learning Research, 23(323):1–81, 2022.
* [27]

  D. A. Hudson and C. D. Manning.
  Gqa: A new dataset for real-world visual reasoning and compositional
  question answering.
  In Proceedings of the IEEE/CVF conference on computer vision and
  pattern recognition, pages 6700–6709, 2019.
* [28]

  R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton.
  Adaptive mixtures of local experts.
  Neural Computation, 3, 1991.
* [29]

  R. K. Jain, D.-M. W. Chiu, W. R. Hawe, et al.
  A quantitative measure of fairness and discrimination.
  Eastern Research Laboratory, Digital Equipment Corporation,
  Hudson, MA, 21(1), 1984.
* [30]

  A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S.
  Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour,
  G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock,
  S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril,
  T. Wang, T. Lacroix, and W. E. Sayed.
  Mixtral of experts.
  arxiv preprint arxiv 2401.04088, 2024.
* [31]

  A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi.
  A diagram is worth a dozen images.
  In Computer Vision–ECCV 2016: 14th European Conference,
  Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14,
  pages 235–251. Springer, 2016.
* [32]

  A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. R. Ruiz, B. Mustafa,
  J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby.
  Sparse upcycling: Training mixture-of-experts from dense checkpoints.
  arXiv preprint arXiv:2212.05055, 2022.
* [33]

  T. Kudo and J. Richardson.
  Sentencepiece: A simple and language independent subword tokenizer
  and detokenizer for neural text processing.
  arXiv preprint arXiv:1808.06226, 2018.
* [34]

  J. Kwon and C. Caramanis.
  EM Converges for a Mixture of Many Linear Regressions.
  In S. Chiappa and R. Calandra, editors, Proceedings of the
  Twenty Third International Conference on Artificial Intelligence
  and Statistics, volume 108 of Proceedings of Machine Learning
  Research, pages 1727–1736. PMLR, Aug. 2020.
* [35]

  G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy.
  RACE: Large-scale ReAding comprehension dataset from
  examinations.
  In Proceedings of the 2017 Conference on Empirical Methods in
  Natural Language Processing, pages 785–794, Copenhagen, Denmark, Sept.
  2017. Association for Computational Linguistics.
* [36]

  M. Le, C. Nguyen, H. Nguyen, Q. Tran, T. Le, and N. Ho.
  Revisiting prefix-tuning: Statistical benefits of reparameterization
  among prompts.
  In The Thirteenth International Conference on Learning
  Representations, 2025.
* [37]

  M. Le, A. N. The, H. Nguyen, T. T. N. Vu, H. T. Pham, L. N. Van, and N. Ho.
  Mixture of experts meets prompt-based continual learning.
  In The Thirty-eighth Annual Conference on Neural Information
  Processing Systems, 2024.
* [38]

  D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer,
  and Z. Chen.
  GShard: Scaling Giant Models with Conditional Computation
  and Automatic Sharding.
  In International Conference on Learning Representations,
  2021.
* [39]

  H. Li, S. Lin, L. Duan, Y. Liang, and N. Shroff.
  Theory on mixture-of-experts in continual learning.
  In The Thirteenth International Conference on Learning
  Representations, 2025.
* [40]

  Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen.
  Evaluating object hallucination in large vision-language models.
  arXiv preprint arXiv:2305.10355, 2023.
* [41]

  H. Liang, Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao, and
  Z. Wang.
  M3ViT: Mixture-of-Experts Vision Transformer for
  Efficient Multi-task Learning with Model-Accelerator Co-design.
  In NeurIPS, 2022.
* [42]

  H. Liu, C. Li, Y. Li, and Y. J. Lee.
  Improved baselines with visual instruction tuning.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, pages 26296–26306, 2024.
* [43]

  H. Liu, C. Li, Q. Wu, and Y. J. Lee.
  Visual instruction tuning.
  In NeurIPS, 2023.
* [44]

  H. Liu, C. Li, Q. Wu, and Y. J. Lee.
  Visual instruction tuning.
  Advances in neural information processing systems,
  36:34892–34916, 2023.
* [45]

  Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L. Liu, L. Jin,
  and X. Bai.
  Ocrbench: on the hidden mystery of ocr in large multimodal models.
  Science China Information Sciences, 67(12):220102, 2024.
* [46]

  I. Loshchilov and F. Hutter.
  Decoupled weight decay regularization.
  arXiv preprint arXiv:1711.05101, 2017.
* [47]

  P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark,
  and A. Kalyan.
  Learn to explain: Multimodal reasoning via thought chains for science
  question answering.
  Advances in Neural Information Processing Systems,
  35:2507–2521, 2022.
* [48]

  J. Ludziejewski, J. Krajewski, K. Adamczewski, M. Pióro, M. Krutul,
  S. Antoniak, K. Ciebiera, K. Król, T. Odrzygóźdź,
  P. Sankowski, M. Cygan, and S. Jaszczur.
  Scaling laws for fine-grained mixture of experts.
  In ICLR 2024 Workshop on Mathematical and Empirical
  Understanding of Foundation Models, 2024.
* [49]

  T. Manole and N. Ho.
  Refined convergence rates for maximum likelihood estimation under
  finite mixture models.
  In Proceedings of the 39th International Conference on Machine
  Learning, volume 162 of Proceedings of Machine Learning Research,
  pages 14979–15006. PMLR, 17–23 Jul 2022.
* [50]

  E. F. Mendes and W. Jiang.
  Convergence rates for mixture-of-experts.
  arXiv preprint arxiv 1110.2058, 2011.
* [51]

  N. Muennighoff, L. Soldaini, D. Groeneveld, K. Lo, J. Morrison, S. Min, W. Shi,
  P. Walsh, O. Tafjord, N. Lambert, et al.
  Olmoe: Open mixture-of-experts language models.
  arXiv preprint arXiv:2409.02060, 2024.
* [52]

  H. Nguyen, P. Akbarian, T. Nguyen, and N. Ho.
  A general theory for softmax gating multinomial logistic mixture of
  experts.
  In Proceedings of the ICML, 2024.
* [53]

  H. Nguyen, P. Akbarian, T. Pham, T. Nguyen, S. Zhang, and N. Ho.
  Statistical advantages of perturbing cosine router in mixture of
  experts.
  In International Conference on Learning Representations, 2025.
* [54]

  H. Nguyen, P. Akbarian, F. Yan, and N. Ho.
  Statistical perspective of top-k sparse softmax gating mixture of
  experts.
  In International Conference on Learning Representations, 2024.
* [55]

  H. Nguyen, N. Ho, and A. Rinaldo.
  Convergence rates for softmax gating mixture of experts.
  arXiv preprint arXiv:2503.03213, 2025.
* [56]

  H. Nguyen, T. Nguyen, and N. Ho.
  Demystifying softmax gating function in Gaussian mixture of
  experts.
  In Advances in Neural Information Processing Systems, 2023.
* [57]

  N. V. Nguyen, T. T. Doan, L. Tran, V. Nguyen, and Q. Pham.
  Libmoe: A library for comprehensive benchmarking mixture of experts
  in large language models.
  arXiv preprint arXiv:2411.00918, 2024.
* [58]

  D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle,
  M. Baroni, G. Boleda, and R. Fernández.
  The lambada dataset: Word prediction requiring a broad discourse
  context.
  arXiv preprint arXiv:1606.06031, 2016.
* [59]

  Q. Pham, G. Do, H. Nguyen, T. Nguyen, C. Liu, M. Sartipi, B. T. Nguyen,
  S. Ramasamy, X. Li, S. Hoi, and N. Ho.
  Competesmoe – effective training of sparse mixture of experts via
  competition.
  arXiv preprint arXiv:2402.02526, 2024.
* [60]

  Qwen et al.
  Qwen2.5 technical report.
  arXiv preprint arXiv:2412.15115, 2025.
* [61]

  S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He.
  Zero: Memory optimizations toward training trillion parameter models.
  In SC20: International Conference for High Performance
  Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
* [62]

  C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pint,
  D. Keysers, and N. Houlsby.
  Scaling vision with sparse mixture of experts.
  In Advances in Neural Information Processing Systems,
  volume 34, pages 8583–8595. Curran Associates, Inc., 2021.
* [63]

  M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi.
  Socialiqa: Commonsense reasoning about social interactions.
  arXiv preprint arXiv:1904.09728, 2019.
* [64]

  N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean.
  Outrageously large neural networks: The sparsely-gated
  mixture-of-experts layer.
  In In International Conference on Learning Representations,
  2017.
* [65]

  A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and
  M. Rohrbach.
  Towards vqa models that can read.
  In Proceedings of the IEEE/CVF conference on computer vision and
  pattern recognition, pages 8317–8326, 2019.
* [66]

  D. Soboleva, F. Al-Khateeb, R. Myers, J. R. Steeves, J. Hestness, and N. Dey.
  SlimPajama: A 627B token cleaned and deduplicated version of
  RedPajama, 2023.
* [67]

  J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu.
  Roformer: Enhanced transformer with rotary position embedding.
  Neurocomputing, 568:127063, 2024.
* [68]

  A. Talmor, J. Herzig, N. Lourie, and J. Berant.
  Commonsenseqa: A question answering challenge targeting commonsense
  knowledge.
  arXiv preprint arXiv:1811.00937, 2018.
* [69]

  G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer,
  D. Vincent, Z. Pan, et al.
  Gemini 1.5: Unlocking multimodal understanding across millions of
  tokens of context.
  arXiv preprint arXiv:2403.05530, 2024.
* [70]

  H. Teicher.
  Identifiability of finite mixtures.
  Ann. Math. Statist., 32:1265–1269, 1963.
* [71]

  T. Truong, C. Nguyen, H. Nguyen, M. Le, T. Le, and N. Ho.
  Replora: Reparameterizing low-rank adaptation via the perspective of
  mixture of experts.
  arXiv preprint arXiv:2502.03044, 2025.
* [72]

  S. van de Geer.
  Empirical processes in M-estimation.
  Cambridge University Press, 2000.
* [73]

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u.
  Kaiser, and I. Polosukhin.
  Attention is all you need.
  In Advances in Neural Information Processing Systems,
  volume 30. Curran Associates, Inc., 2017.
* [74]

  A. Warstadt, A. Parrish, H. Liu, A. Mohananey, W. Peng, S.-F. Wang, and S. R.
  Bowman.
  Blimp: The benchmark of linguistic minimal pairs for english.
  Transactions of the Association for Computational Linguistics,
  8:377–392, 2020.
* [75]

  M. Weber, D. Fu, Q. Anthony, Y. Oren, S. Adams, A. Alexandrov, X. Lyu,
  H. Nguyen, X. Yao, V. Adams, et al.
  Redpajama: an open dataset for training large language models.
  Advances in neural information processing systems,
  37:116462–116492, 2024.
* [76]

  F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, and Y. You.
  Openmoe: An early effort on open mixture-of-experts language models.
  arXiv preprint arXiv:2402.01739, 2024.
* [77]

  F. Yan, H. Nguyen, P. Akbarian, N. Ho, and A. Rinaldo.
  Sigmoid self-attention is better than softmax self-attention: A
  mixture-of-experts perspective.
  arXiv preprint arXiv:2502.00281, 2025.
* [78]

  X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang,
  H. Sun, et al.
  Mmmu-pro: A more robust multi-discipline multimodal understanding
  benchmark.
  arXiv preprint arXiv:2409.02813, 2024.
* [79]

  S. Yun, I. Choi, J. Peng, Y. Wu, J. Bao, Q. Zhang, J. Xin, Q. Long, and
  T. Chen.
  Flex-moe: Modeling arbitrary modality combination via the flexible
  mixture-of-experts.
  In The Thirty-eighth Annual Conference on Neural Information
  Processing Systems, 2024.
* [80]

  A. Zeevi, R. Meir, and V. Maiorov.
  Error bounds for functional approximation and estimation using
  mixtures of experts.
  IEEE Transactions on Information Theory, 44(3):1010–1025,
  1998.
* [81]

  R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi.
  Hellaswag: Can a machine really finish your sentence?
  arXiv preprint arXiv:1905.07830, 2019.
* [82]

  X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer.
  Sigmoid loss for language image pre-training.
  In Proceedings of the IEEE/CVF international conference on
  computer vision, pages 11975–11986, 2023.
* [83]

  Y.-F. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang, Q. Wen,
  Z. Zhang, et al.
  Mme-realworld: Could your multimodal llm challenge high-resolution
  real-world scenarios that are difficult for humans?
  arXiv preprint arXiv:2408.13257, 2024.

[◄](/html/2505.10859)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2505.10860)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2505.10860)
[View original  
on arXiv](https://arxiv.org/abs/2505.10860)[►](/html/2505.10861)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Jun 5 21:12:51 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
