---
arxiv: '2007.01627'
authors:
- Marine Le Morvan
- Julie Josse
- Thomas Moreau
- Erwan Scornet
- Gaël Varoquaux
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'NeuMiss networks: differentiable programming for supervised learning with
  missing values'
url: http://arxiv.org/abs/2007.01627v4
year: 2020
---

[2007.01627] NeuMiss networks: differentiable programming for supervised learning with missing values














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



# NeuMiss networks: differentiable programming for supervised learning with missing values

Marine Le Morvan1,2  
Julie Josse1,3   
Thomas Moreau1  
Erwan Scornet3  
Gaël Varoquaux1, 4
  
1 Université Paris-Saclay, Inria, CEA, Palaiseau, 91120, France
  
2 Université Paris-Saclay, CNRS/IN2P3, IJCLab, 91405 Orsay, France
  
3 CMAP, UMR7641, Ecole Polytechnique, IP Paris, 91128 Palaiseau, France
  
4 Mila, McGill University, Montréal, Canada
  
{marine.le-morvan, julie.josse, thomas.moreau, gael.varoquaux}@inria.fr
  
erwan.scornet@polytechnique.edu

###### Abstract

The presence of missing values makes supervised learning much more challenging. Indeed, previous work has shown that even when the response is a linear function of the complete data, the optimal predictor is a complex function of the observed entries and the missingness indicator. As a result, the computational or sample complexities of consistent approaches depend on the number of missing patterns, which can be exponential in the number of dimensions. In this work, we derive the analytical form of the optimal predictor under a linearity assumption and various missing data mechanisms including Missing at Random (MAR) and self-masking (Missing Not At Random). Based on a Neumann-series approximation of the optimal predictor, we propose a new principled architecture, named NeuMiss networks. Their originality and strength come from the use of a new type of non-linearity: the multiplication by the missingness indicator. We provide an upper bound on the Bayes risk of NeuMiss networks, and show that they have good predictive accuracy with both a number of parameters and a computational complexity independent of the number of missing data patterns. As a result they scale well to problems with many features, and remain statistically efficient for medium-sized samples. Moreover, we show that, contrary to procedures using EM or imputation, they are robust to the missing data mechanism, including difficult MNAR settings such as self-masking.

## 1 Introduction

Increasingly complex data-collection pipelines, often assembling
multiple sources of information, lead to datasets with incomplete
observations and complex missing-values mechanisms. The pervasiveness of missing values
has triggered an abundant statistical literature on the subject
[[14](#bib.bib14), [31](#bib.bib31)]: a recent survey reviewed more than 150 implementations to handle missing data [[10](#bib.bib10)].
Nevertheless, most methods have been developed either for
inferential purposes, i.e. to estimate parameters of a probabilistic
model of the fully-observed data, or for imputation, completing missing entries as well as possible [[6](#bib.bib6)].
These methods often require strong assumptions on the missing-values
mechanism, i.e. either the missing at random (MAR) assumption [[27](#bib.bib27)] –
the probability of being missing only depends on observed values – or the more restrictive Missing Completely At Random assumption (MCAR) – the missingness is independent of the data.
In MAR or MCAR settings, good imputation is sufficient to fit statistical
models, or even train supervised-learning models
[[11](#bib.bib11)]. In particular, a precise knowledge of the
data-generating mechanism can be used to derive an Expectation
Maximization (EM) [[2](#bib.bib2)] formulation with the
minimum number of necessary parameters. Yet, as we will see, this is
intractable if the number of features is not small, as potentially 2dsuperscript2𝑑2^{d}
missing-value patterns must be modeled.

The last missing-value mechanism category,
Missing Not At Random (MNAR), covers cases where the
probability of being missing depends on the unobserved values.
This is a frequent situation in which
missingness cannot be ignored in the
statistical analysis [[12](#bib.bib12)].
Much of the work on MNAR data focuses on problems of identifiability, in both parametric and non-parametric settings [[29](#bib.bib29), [20](#bib.bib20), [21](#bib.bib21), [22](#bib.bib22)].
In MNAR settings, estimation strategies often require modeling
the missing-values mechanism [[9](#bib.bib9)]. This
complicates the inference task and is often limited to cases with few MNAR variables.
Other approaches need the masking matrix to be well approximated
with low-rank matrices [[18](#bib.bib18), [1](#bib.bib1), [7](#bib.bib7), [16](#bib.bib16), [32](#bib.bib32)].

Supervised learning with missing values has different goals than
probabilistic modeling [[11](#bib.bib11)] and has been less
studied. As the test set is also expected to have missing entries,
optimality on the fully-observed data is no longer a goal per se. Rather, the
goal of minimizing an expected risk lend itself well to
non-parametric models which can compensate from some oddities introduced
by missing values. Indeed, with a powerful learner capable of learning
any function, imputation by a constant is Bayes consistent
[[11](#bib.bib11)]. Yet, the complexity of this function that
must be approximated governs the success of this approach
outside of asymptotic regimes. In the simple case of a linear regression
with missing values, the optimal predictor has a combinatorial
expression: for d𝑑d features, there are 2dsuperscript2𝑑2^{d} possible missing-values
patterns requiring 2dsuperscript2𝑑2^{d} models [[13](#bib.bib13)].

Le Morvan et al. [[13](#bib.bib13)] showed that in this setting, a multilayer
perceptrons (MLP) can be consistent even in a pattern mixture
MNAR model, but assuming 2dsuperscript2𝑑2^{d} hidden units.
There have been many adaptations of neural networks to missing values,
often involving an imputation with 0’s and concatenating the mask (the
indicator matrix coding for missing values)
[[23](#bib.bib23), [19](#bib.bib19), [15](#bib.bib15), [34](#bib.bib34), [4](#bib.bib4)]. However there is no theory relating the
network architecture to the impact of the missing-value mechanism on the
prediction function. In particular, an important practical question is:
how complex should the architecture be to cater for a given mechanism?
Overly-complex architectures require a lot of data, but being too
restrictive will introduce bias for missing values.

The present paper addresses the challenge of supervised learning with missing values.
We propose a theoretically-grounded neural-network architecture which allows
to implicitly impute values as a function of the observed data, aiming at
the best prediction. More precisely,

* •

  We derive an analytical expression of the Bayes predictor for linear regression in the presence of missing values under various missing data mechanisms including MAR and self-masking MNAR.
* •

  We propose a new principled architecture, named NeuMiss network, based on a Neumann series approximation of the Bayes predictors, whose originality and strength is the use of ⊙Mdirect-productabsent𝑀\odot M nonlinearities, i.e. the elementwise multiplication by the missingness indicator.
* •

  We provide an upper bound on the Bayes risk of NeuMiss networks which highlights the benefits of depth and learning to approximate.
* •

  We provide an interpretation of a classical ReLU network as a shallow NeuMiss network. We further demonstrate empirically the crucial role of the ⊙direct-product\odot nonlinearities,
  by showing that increasing the capacity of NeuMiss networks improves predictions while it does not for classical networks.
* •

  We show that NeuMiss networks are suited
  medium-sized datasets: they require O​(d2)𝑂superscript𝑑2O(d^{2}) samples,
  contrary to O​(2d)𝑂superscript2𝑑O(2^{d}) for methods that do not share weights between missing data patterns.
* •

  We demonstrate the benefits of the proposed
  architecture over classical methods such as EM algorithms or iterative
  conditional imputation [[31](#bib.bib31)] both in terms of
  computational complexity –these methods scale in O​(2d​d2)𝑂superscript2𝑑superscript𝑑2O(2^{d}d^{2})
  [[28](#bib.bib28)] and O​(d3)𝑂superscript𝑑3O(d^{3}) respectively–, and in the ability to be robust to the missing data mechanism, including MNAR.

## 2 Optimal predictors in the presence of missing values

#### Notations

We consider a data set 𝒟n={(X1,Y1),…,(Xn,Yn)}subscript𝒟𝑛subscript𝑋1subscript𝑌1…subscript𝑋𝑛subscript𝑌𝑛\mathcal{D}\_{n}=\{(X\_{1},Y\_{1}),\ldots,(X\_{n},Y\_{n})\} of independent pairs (Xi,Yi)subscript𝑋𝑖subscript𝑌𝑖(X\_{i},Y\_{i}), distributed as the generic pair (X,Y)𝑋𝑌(X,Y), where X∈ℝd𝑋superscriptℝ𝑑X\in\mathbb{R}^{d} and Y∈ℝ𝑌ℝY\in\mathbb{R}. We introduce the indicator
vector M∈{0,1}d𝑀superscript01𝑑M\in\{0,1\}^{d} which satisfies, for all 1≤j≤d1𝑗𝑑1\leq j\leq d,
Mj=1subscript𝑀𝑗1M\_{j}=1 if and only if Xjsubscript𝑋𝑗X\_{j} is not observed. The random vector M𝑀M
acts as a mask on X𝑋X. We define the incomplete feature vector X~∈𝒳~=(ℝ∪{𝙽𝙰})d~𝑋~𝒳superscriptℝ𝙽𝙰𝑑\widetilde{X}\in\widetilde{\mathcal{X}}=(\mathbb{R}\cup\{\mathtt{NA}\})^{d} (see
[[27](#bib.bib27)], [[26](#bib.bib26), appendix B]) as X~j=𝙽𝙰subscript~𝑋𝑗𝙽𝙰\widetilde{X}\_{j}=\mathtt{NA} if Mj=1subscript𝑀𝑗1M\_{j}=1, and X~j=Xjsubscript~𝑋𝑗subscript𝑋𝑗\widetilde{X}\_{j}=X\_{j} otherwise.
As such, X~~𝑋\widetilde{X} is a mixed categorical and continuous variable. An
example of realization (lower-case letters) of the previous random
variables would be a vector x=(1.1,2.3,−3.1,8,5.27)𝑥1.12.33.185.27x=(1.1,2.3,-3.1,8,5.27) with the
missing pattern m=(0,1,0,0,1)𝑚01001m=(0,1,0,0,1), giving
x~=(1.1,NA,−3.1,8,NA).~𝑥1.1NA3.18NA\widetilde{x}=(1.1,~{}~{}\texttt{NA},~{}-3.1,~{}~{}8,~{}~{}\texttt{NA}).

For realizations m𝑚m of M𝑀M, we also denote by
o​b​s​(m)𝑜𝑏𝑠𝑚obs(m) (resp. m​i​s​(m)𝑚𝑖𝑠𝑚mis(m)) the indices of the zero entries of m𝑚m
(resp. non-zero). Following classic missing-value notations, we let
Xo​b​s​(M)subscript𝑋𝑜𝑏𝑠𝑀X\_{obs(M)} (resp. Xm​i​s​(M)subscript𝑋𝑚𝑖𝑠𝑀X\_{mis(M)})
be the observed (resp. missing) entries in X𝑋X. Pursuing
the above example, we have m​i​s​(m)={1,4}𝑚𝑖𝑠𝑚14mis(m)=\{1,4\}, o​b​s​(m)={0,2,3}𝑜𝑏𝑠𝑚023obs(m)=\{0,2,3\},
xo​b​s​(m)=(1.1,−3.1,8)subscript𝑥𝑜𝑏𝑠𝑚1.13.18x\_{obs(m)}=(1.1,-3.1,~{}~{}8), xm​i​s​(m)=(2.3,5.27)subscript𝑥𝑚𝑖𝑠𝑚2.35.27x\_{mis(m)}=(2.3,~{}~{}5.27).
To lighten notations, when there is no ambiguity, we remove the explicit dependence in m𝑚m and write, e.g.,
Xo​b​ssubscript𝑋𝑜𝑏𝑠X\_{obs}.

### 2.1 Problem statement: supervised learning with missing values

We consider a linear model of the complete data, such that the response Y𝑌Y satisfies:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Y𝑌\displaystyle Y | =β0⋆+⟨X,β⋆⟩+ε,for some​β0⋆∈ℝ,β⋆∈ℝd,and​ε∼𝒩​(0,σ2).formulae-sequenceabsentsuperscriptsubscript𝛽0⋆  𝑋superscript𝛽⋆ 𝜀formulae-sequencefor somesuperscriptsubscript𝛽0⋆ℝformulae-sequencesuperscript𝛽⋆superscriptℝ𝑑similar-toand𝜀𝒩0superscript𝜎2\displaystyle=\beta\_{0}^{\star}+\langle X,\beta^{\star}\rangle+\varepsilon,\qquad\text{for some}\;\beta\_{0}^{\star}\in\mathbb{R},\beta^{\star}\in\mathbb{R}^{d},\;\text{and}\;\varepsilon\sim\mathcal{N}(0,\sigma^{2}). |  | (1) |

Prediction with missing values departs from standard linear-model
settings: the aim is
to predict Y𝑌Y given X~~𝑋\widetilde{X}, as the complete input X𝑋X may be unavailable. The corresponding optimization problem is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | fX~⋆∈argminf:𝒳~→ℝ​𝔼​[(Y−f​(X~))2],subscriptsuperscript𝑓⋆~𝑋:𝑓→~𝒳ℝargmin𝔼delimited-[]superscript𝑌𝑓~𝑋2\displaystyle f^{\star}\_{\widetilde{X}}\in\underset{f:\widetilde{\mathcal{X}}\rightarrow\mathbb{R}}{\mathrm{argmin}}~{}\mathbb{E}[(Y-f(\widetilde{X}))^{2}], |  | (2) |

where fX~⋆subscriptsuperscript𝑓⋆~𝑋f^{\star}\_{\widetilde{X}} is the Bayes predictor for the squared loss, in the presence of missing values. The main difficulty of this problem comes from the half-discrete nature of the input space 𝒳~~𝒳\widetilde{\mathcal{X}}. Indeed, the Bayes predictor fX~⋆​(X~)=𝔼​[Y|X~]subscriptsuperscript𝑓⋆~𝑋~𝑋𝔼delimited-[]conditional𝑌~𝑋f^{\star}\_{\widetilde{X}}(\widetilde{X})=\mathbb{E}\big{[}Y~{}|~{}\widetilde{X}\big{]} can be rewritten as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | fX~⋆​(X~)=𝔼​[Y|M,Xo​b​s​(M)]=∑m∈{0,1}d𝔼​[Y|Xo​b​s​(m),M=m]​𝟙M=m,subscriptsuperscript𝑓⋆~𝑋~𝑋𝔼delimited-[]conditional𝑌  𝑀subscript𝑋𝑜𝑏𝑠𝑀subscript𝑚superscript01𝑑𝔼delimited-[]conditional𝑌  subscript𝑋𝑜𝑏𝑠𝑚𝑀𝑚subscript1𝑀𝑚f^{\star}\_{\widetilde{X}}(\widetilde{X})=\mathbb{E}\left[Y~{}|~{}M,X\_{obs(M)}\right]=\sum\_{m\in\{0,1\}^{d}}\mathbb{E}\left[Y|X\_{obs(m)},M=m\right]~{}\mathds{1}\_{M=m}, |  | (3) |

which highlights the combinatorial issue of solving
([2](#S2.E2 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")): one may need to
optimize 2dsuperscript2𝑑2^{d} submodels, for the different m𝑚m.
In the
following,
we write the Bayes predictor f⋆superscript𝑓⋆f^{\star} as a function of
(Xo​b​s​(M),M)subscript𝑋𝑜𝑏𝑠𝑀𝑀(X\_{obs(M)},M):

|  |  |  |
| --- | --- | --- |
|  | f⋆​(Xo​b​s​(M),M)=𝔼​[Y|Xo​b​s​(M),M].superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀𝑀𝔼delimited-[]conditional𝑌  subscript𝑋𝑜𝑏𝑠𝑀𝑀\displaystyle f^{\star}(X\_{obs(M)},M)=\mathbb{E}\left[Y|X\_{obs(M)},M\right]. |  |

### 2.2 Expression of the Bayes predictor under various missing-values mechanisms

There is no general closed-form expression for the Bayes predictor, as it depends on the data distribution and missingness mechanism. However, an exact expression can be derived for Gaussian data with various missingness mechanisms.

###### Assumption 1 (Gaussian data).

The distribution of X𝑋X is Gaussian, that is, X∼𝒩​(μ,Σ)similar-to𝑋𝒩𝜇ΣX\sim\mathcal{N}(\mathbf{\mu},\Sigma).

###### Assumption 2 (MCAR mechanism).

For all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, P​(M=m|X)=P​(M=m)𝑃𝑀conditional𝑚𝑋𝑃𝑀𝑚P(M=m|X)=P(M=m).

###### Assumption 3 (MAR mechanism).

For all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, P​(M=m|X)=P​(M=m|Xo​b​s​(m))𝑃𝑀conditional𝑚𝑋𝑃𝑀conditional𝑚subscript𝑋𝑜𝑏𝑠𝑚P(M=m|X)=P(M=m|X\_{obs(m)}).

###### Proposition 2.1 (MAR Bayes predictor).

Assume that the data are generated via the linear model defined in equation ([1](#S2.E1 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and satisfy Assumption [1](#Thmassumption1 "Assumption 1 (Gaussian data). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). Additionally, assume that either Assumption [2](#Thmassumption2 "Assumption 2 (MCAR mechanism). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") or Assumption [3](#Thmassumption3 "Assumption 3 (MAR mechanism). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") holds. Then the Bayes predictor f⋆superscript𝑓⋆f^{\star} takes the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆​(Xo​b​s,M)=β0⋆+⟨βo​b​s⋆,Xo​b​s⟩+⟨βm​i​s⋆,μm​i​s+Σm​i​s,o​b​s​(Σo​b​s)−1​(Xo​b​s−μo​b​s)⟩,superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀superscriptsubscript𝛽0⋆  superscriptsubscript𝛽𝑜𝑏𝑠⋆subscript𝑋𝑜𝑏𝑠  superscriptsubscript𝛽𝑚𝑖𝑠⋆subscript𝜇𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠f^{\star}(X\_{obs},M)=\beta\_{0}^{\star}+\langle\beta\_{obs}^{\star},X\_{obs}\rangle+\langle\beta\_{mis}^{\star},\mu\_{mis}+\Sigma\_{mis,obs}(\Sigma\_{obs})^{-1}(X\_{obs}-\mu\_{obs})\rangle, |  | (4) |

where we use o​b​s𝑜𝑏𝑠obs (resp. m​i​s𝑚𝑖𝑠mis) instead of o​b​s​(M)𝑜𝑏𝑠𝑀obs(M) (resp. m​i​s​(M)𝑚𝑖𝑠𝑀mis(M)) for lighter notations.

Obtaining the Bayes predictor expression turns out to be far more complicated for general MNAR settings but feasible for the Gaussian self-masking mechanism described below.

###### Assumption 4 (Gaussian self-masking).

The missing data mechanism is self-masked with P​(M|X)=∏k=1dP​(Mk|Xk)𝑃conditional𝑀𝑋superscriptsubscriptproduct𝑘1𝑑𝑃conditionalsubscript𝑀𝑘subscript𝑋𝑘P(M|X)=\prod\_{k=1}^{d}P(M\_{k}|X\_{k}) and ∀k∈⟦1,d⟧,for-all𝑘

1𝑑\forall k\in\left\llbracket 1,d\right\rrbracket,

|  |  |  |
| --- | --- | --- |
|  | P​(Mk=1|Xk)=Kk​exp⁡(−12​(Xk−μ~k)2σ~k2)with​ 0<Kk<1.formulae-sequence𝑃subscript𝑀𝑘conditional1subscript𝑋𝑘subscript𝐾𝑘12superscriptsubscript𝑋𝑘subscript~𝜇𝑘2superscriptsubscript~𝜎𝑘2with 0subscript𝐾𝑘1P(M\_{k}=1|X\_{k})=K\_{k}\exp\left(-\frac{1}{2}\frac{(X\_{k}-\widetilde{\mu}\_{k})^{2}}{\widetilde{\sigma}\_{k}^{2}}\right)\qquad\text{with}\;0<K\_{k}<1. |  |

###### Proposition 2.2 (Bayes predictor with Gaussian self-masking).

Assume that the data are generated via the linear model defined in equation ([1](#S2.E1 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and satisfy Assumption [1](#Thmassumption1 "Assumption 1 (Gaussian data). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and Assumption [4](#Thmassumption4 "Assumption 4 (Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). Let Σm​i​s|o​b​s=Σm​i​s,m​i​s−Σm​i​s,o​b​s​Σo​b​s−1​Σo​b​s,m​i​s,subscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠subscriptΣ

𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣ

𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ

𝑜𝑏𝑠𝑚𝑖𝑠\Sigma\_{mis|obs}=\Sigma\_{mis,mis}-\Sigma\_{mis,obs}\Sigma\_{obs}^{-1}\Sigma\_{obs,mis}, and let D𝐷D be the diagonal matrix such that diag​(D)=(σ~12,…,σ~d2)diag𝐷superscriptsubscript~𝜎12…superscriptsubscript~𝜎𝑑2\mathrm{diag}(D)=(\widetilde{\sigma}\_{1}^{2},\ldots,\widetilde{\sigma}\_{d}^{2}). Then the Bayes predictor writes

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆​(Xo​b​s,M)superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀\displaystyle f^{\star}(X\_{obs},M) | =β0⋆+⟨βo​b​s⋆,Xo​b​s⟩+⟨βm​i​s⋆,(Id+Dm​i​sΣm​i​s|o​b​s−1)−1\displaystyle=\beta\_{0}^{\star}+\langle\beta\_{obs}^{\star},X\_{obs}\rangle+\langle\beta\_{mis}^{\star},(Id+D\_{mis}\Sigma\_{mis|obs}^{-1})^{-1} |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ×(μ~m​i​s+Dm​i​sΣm​i​s|o​b​s−1(μm​i​s+Σm​i​s,o​b​s(Σo​b​s)−1(Xo​b​s−μo​b​s)))⟩\displaystyle\quad\times(\tilde{\mu}\_{mis}+D\_{mis}\Sigma\_{mis|obs}^{-1}(\mu\_{mis}+\Sigma\_{mis,obs}\left(\Sigma\_{obs}\right)^{-1}\left(X\_{obs}-\mu\_{obs}\right)))\rangle |  | (5) |

The proof of Propositions [2.1](#S2.Thmproposition1 "Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and [2.2](#S2.Thmproposition2 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") are
in the Supplementary Materials ([A.3](#A1.SS3 "A.3 Proof of Proposition 2.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and [A.4](#A1.SS4 "A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")). These are the first results establishing
exact expressions of the Bayes predictor in a MAR and specific MNAR
mechanisms.
Note that these propositions show that the Bayes predictor is linear by
pattern under the assumptions studied, i.e., each of the 2dsuperscript2𝑑2^{d} submodels
in equation [3](#S2.E3 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") are linear functions of Xo​b​ssubscript𝑋𝑜𝑏𝑠X\_{obs}. For
non-Gaussian data, the Bayes predictor may not be
linear by pattern [[13](#bib.bib13), Example 3.1].

#### Generality of the Gaussian self-masking model

For a self-masking mechanism where
the probability of being missing increases (or decreases) with the value
of the underlying variable, probit or logistic functions are often
used [[12](#bib.bib12)]. A Gaussian self-masking model is also
a suitable model: setting the mean of the Gaussian close to the
extreme values gives a similar behaviour. In addition, it covers cases where the
probability of being missing is centered around a given value.

## 3 NeuMiss networks: learning by approximating the Bayes predictors

### 3.1 Insight to build a network: sharing parameters across missing-value patterns

Computing the Bayes predictors in
equations ([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) or ([2.2](#S2.Ex3 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) requires to estimate
the inverse of each submatrix Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)} for each missing-data pattern
m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, *ie* one linear model per missing-data
pattern.
For a number of hidden units ∝2dproportional-toabsentsuperscript2𝑑\propto 2^{d}, a
MLP with ReLU non-linearities can fit these linear models independently
from one-another, and is shown to be consistent
[[13](#bib.bib13)]. But it is prohibitive when d𝑑d grows.
Such an architecture is largely over-parametrized as it does not
share information between similar missing-data patterns.
Indeed, the slopes of each of the linear regression per pattern given by the Bayes
predictor in equations ([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and ([2.2](#S2.Ex3 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) are
linked via the inverses of Σo​b​ssubscriptΣ𝑜𝑏𝑠\Sigma\_{obs}.

Thus, one approach is to
estimate only one vector μ𝜇\mu and one covariance matrix ΣΣ\Sigma via an
expectation maximization (EM) algorithm [[2](#bib.bib2)], and
then compute the inverses of Σo​b​ssubscriptΣ𝑜𝑏𝑠\Sigma\_{obs}. But the computational
complexity then scales linearly in the
number of missing-data patterns (which is in the worst case exponential in the
dimension d𝑑d), and is therefore also prohibitive when the dimension increases.

In what follows, we propose an in-between
solution, modeling the relationships between the slopes for
different missing-data patterns without directly estimating
the covariance matrix. Intuitively, observations from one pattern will be used to estimate the regression parameters of other patterns.

### 3.2 Differentiable approximations of the inverse covariances with Neumann series

The major challenge of equations ([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and
([2.2](#S2.Ex3 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) is the inversion
of the matrices Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)} for all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}. Indeed,
there is no simple relationship for the inverses of different submatrices
in general. As a result, the slope corresponding to a pattern m𝑚m cannot be
easily expressed as a function of ΣΣ\Sigma.

We therefore propose to approximate (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1\left(\Sigma\_{obs(m)}\right)^{-1} for all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d} recursively in the following way. First, we choose as a
starting point a d×d𝑑𝑑d\times d matrix S(0)superscript𝑆0S^{(0)}. So​b​s​(m)(0)subscriptsuperscript𝑆0𝑜𝑏𝑠𝑚S^{(0)}\_{obs(m)} is
then defined as the sub-matrix of S(0)superscript𝑆0S^{(0)} obtained by selecting the
columns and rows that are observed (components for which m=0𝑚0m=0) and is
our order-00 approximation of (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1\left(\Sigma\_{obs(m)}\right)^{-1}. Then, for all
m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, we define the order-ℓℓ\ell approximation So​b​s​(m)(ℓ)subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝑚S^{(\ell)}\_{obs(m)} of (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1\left(\Sigma\_{obs(m)}\right)^{-1} via the following iterative formula: for all ℓ≥1ℓ1\ell\geq 1,

|  |  |  |  |
| --- | --- | --- | --- |
|  | So​b​s​(m)(ℓ)=(I​d−Σo​b​s​(m))​So​b​s​(m)(ℓ−1)+I​d.subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝑚𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚subscriptsuperscript𝑆ℓ1𝑜𝑏𝑠𝑚𝐼𝑑S^{(\ell)}\_{obs(m)}=(Id-\Sigma\_{obs(m)})\,S^{(\ell-1)}\_{obs(m)}+Id. |  | (6) |

The iterates So​b​s​(m)(ℓ)subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝑚S^{(\ell)}\_{obs(m)} converge linearly to (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1(\Sigma\_{obs(m)})^{-1}([A.5](#A1.SS5 "A.5 Controlling the convergence of Neumann iterates ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") in the Supplementary
Materials), and are in fact Neumann series truncated to ℓℓ\ell terms if S(0)=I​dsuperscript𝑆0𝐼𝑑S^{(0)}=Id.

We now define the
order-ℓℓ\ell approximation of the Bayes predictor in MAR settings (equation
 ([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"))) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | fℓ⋆​(Xo​b​s,M)=⟨βo​b​s⋆,Xo​b​s⟩+⟨βm​i​s⋆,μm​i​s+Σm​i​s,o​b​s​So​b​s​(m)(ℓ)​(Xo​b​s−μo​b​s)⟩.subscriptsuperscript𝑓⋆ℓsubscript𝑋𝑜𝑏𝑠𝑀  subscriptsuperscript𝛽⋆𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠  subscriptsuperscript𝛽⋆𝑚𝑖𝑠subscript𝜇𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝑚subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠f^{\star}\_{\ell}(X\_{obs},M)=\langle\beta^{\star}\_{obs},X\_{obs}\rangle+\langle\beta^{\star}\_{mis},\mu\_{mis}+\Sigma\_{mis,obs}S^{(\ell)}\_{obs(m)}(X\_{obs}-\mu\_{obs})\rangle. |  | (7) |

The error between the Bayes predictor and its order-ℓℓ\ell approximation is provided in Proposition [3.1](#S3.Thmproposition1 "Proposition 3.1. ‣ 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

###### Proposition 3.1.

Let ν𝜈\nu be the smallest eigenvalue of ΣΣ\Sigma. Assume that the data are generated via a linear model defined in equation ([1](#S2.E1 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and satisfy Assumption [1](#Thmassumption1 "Assumption 1 (Gaussian data). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). Additionally, assume that either Assumption [2](#Thmassumption2 "Assumption 2 (MCAR mechanism). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") or Assumption [3](#Thmassumption3 "Assumption 3 (MAR mechanism). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") holds and that the spectral radius of ΣΣ\Sigma is strictly smaller than one. Then, for all ℓ≥1ℓ1\ell\geq 1,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[(fℓ⋆​(Xo​b​s,M)−f⋆​(Xo​b​s,M))2]≤(1−ν)2​ℓ​‖β⋆‖22ν​𝔼​[∥I​d−So​b​s​(M)(0)​Σo​b​s​(M)∥22]𝔼delimited-[]superscriptsuperscriptsubscript𝑓ℓ⋆subscript𝑋𝑜𝑏𝑠𝑀superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀2superscript1𝜈2ℓsuperscriptsubscriptnormsuperscript𝛽⋆22𝜈𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝐼𝑑subscriptsuperscript𝑆0𝑜𝑏𝑠𝑀subscriptΣ𝑜𝑏𝑠𝑀22\displaystyle\mathbb{E}\biggl{[}\bigl{(}f\_{\ell}^{\star}(X\_{obs},M)-f^{\star}(X\_{obs},M)\bigr{)}^{2}\biggr{]}\;\leq\;\frac{(1-\nu)^{2\ell}\|\beta^{\star}\|\_{2}^{2}}{\nu}\,\mathbb{E}\biggl{[}\bigl{\|}Id-S^{(0)}\_{obs(M)}\Sigma\_{obs(M)}\bigr{\|}\_{2}^{2}\biggr{]} |  | (8) |

The error of the order-ℓℓ\ell approximation decays exponentially fast with ℓℓ\ell.
More importantly, if the submatrices So​b​s(0)subscriptsuperscript𝑆0𝑜𝑏𝑠S^{(0)}\_{obs} of S(0)superscript𝑆0S^{(0)} are good approximations of (Σo​b​s)−1superscriptsubscriptΣ𝑜𝑏𝑠1(\Sigma\_{obs})^{-1} on average, that is if we choose S(0)superscript𝑆0S^{(0)} which minimizes the expectation in the right-hand side in inequality ([8](#S3.E8 "In Proposition 3.1. ‣ 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), then our model provides a good approximation of the Bayes predictor even with order ℓ=0ℓ0\ell=0.
This is the case for a diagonal covariance matrix, as taking S(0)=Σ−1superscript𝑆0superscriptΣ1S^{(0)}=\Sigma^{-1} has no approximation error as (Σ−1)o​b​s=(Σo​b​s)−1subscriptsuperscriptΣ1𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1(\Sigma^{-1})\_{obs}=(\Sigma\_{obs})^{-1}.

### 3.3 NeuMiss network architecture: multiplying by the mask

#### Network architecture

We propose a neural-network architecture to approximate the Bayes predictor, where the inverses (Σo​b​s)−1superscriptsubscriptΣ𝑜𝑏𝑠1(\Sigma\_{obs})^{-1} are computed using an unrolled version of the iterative algorithm. Figure [1](#S3.F1 "Figure 1 ‣ Multiplying by the mask ‣ 3.3 NeuMiss network architecture: multiplying by the mask ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") gives a diagram for such neural network using an order-3
approximation corresponding to a depth 4. x𝑥x is the input, with missing
values replaced by 0. μ𝜇\mu is a trainable parameter corresponding to the parameter μ𝜇\mu in equation ([7](#S3.E7 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")). To match the Bayes predictor exactly (equation ([7](#S3.E7 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"))), weight matrices should be simple transformations of the covariance matrix indicated in blue on Figure [1](#S3.F1 "Figure 1 ‣ Multiplying by the mask ‣ 3.3 NeuMiss network architecture: multiplying by the mask ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

Following strictly Neummann iterates would call for
a shared weight matrix across
all WN​e​u(k)superscriptsubscript𝑊𝑁𝑒𝑢𝑘W\_{Neu}^{(k)}. Rather, we learn each layer independently. This choice is motivated by works on iterative algorithm unrolling [[5](#bib.bib5)] where independent layers’ weights can improve a network’s approximation performance [[33](#bib.bib33)]. Note that [[3](#bib.bib3)] has also introduced a neural network architecture based on unrolling the Neumann series. However, their goal is to solve a linear inverse problem with a learned regularization, which is very different from ours.

#### Multiplying by the mask

Note that the observed indices change for each sample, leading to an implementation challenge. For a sample with missing data pattern m𝑚m, the weight matrices S(0)superscript𝑆0S^{(0)}, WN​e​u(1)superscriptsubscript𝑊𝑁𝑒𝑢1W\_{Neu}^{(1)} and
WN​e​u(2)superscriptsubscript𝑊𝑁𝑒𝑢2W\_{Neu}^{(2)} of Figure [1](#S3.F1 "Figure 1 ‣ Multiplying by the mask ‣ 3.3 NeuMiss network architecture: multiplying by the mask ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") should be masked such
that their rows and columns corresponding to the indices m​i​s​(m)𝑚𝑖𝑠𝑚mis(m) are
zeroed, and the rows of WM​i​xsubscript𝑊𝑀𝑖𝑥W\_{Mix} corresponding to o​b​s​(m)𝑜𝑏𝑠𝑚obs(m) as well as
the columns of WM​i​xsubscript𝑊𝑀𝑖𝑥W\_{Mix} corresponding to m​i​s​(m)𝑚𝑖𝑠𝑚mis(m) are zeroed.
Implementing efficiently a network in which the weight matrices are masked
differently for each sample can be challenging. We thus use the following trick.
Let W𝑊W be a weight matrix, v𝑣v a vector, and m¯=1−m¯𝑚1𝑚\bar{m}=1-m. Then (W⊙m¯​m¯⊤)​v=(W​(v⊙m¯))⊙m¯direct-product𝑊¯𝑚superscript¯𝑚top𝑣direct-product𝑊direct-product𝑣¯𝑚¯𝑚(W\odot\bar{m}\bar{m}^{\top})v=(W(v\odot\bar{m}))\odot\bar{m}, i.e, using a masked weight matrix is equivalent to masking the
input and output vector.
The network can then be seen as a classical network where the nonlinearities
are multiplications by the mask.

x⊙m¯direct-product𝑥¯𝑚x\odot\bar{m}−-μ⊙m¯direct-product𝜇¯𝑚\mu\odot\bar{m}S(0)superscript𝑆0S^{(0)}WN​e​u(1)superscriptsubscript𝑊𝑁𝑒𝑢1W\_{Neu}^{(1)}(I​d−Σo​b​s𝐼𝑑subscriptΣ𝑜𝑏𝑠Id-\Sigma\_{obs})++WN​e​u(2)superscriptsubscript𝑊𝑁𝑒𝑢2W\_{Neu}^{(2)}(I​d−Σo​b​s𝐼𝑑subscriptΣ𝑜𝑏𝑠Id-\Sigma\_{obs})++WM​i​x(3)superscriptsubscript𝑊𝑀𝑖𝑥3W\_{Mix}^{{\color[rgb]{0.93,0.93,0.93}\definecolor[named]{pgfstrokecolor}{rgb}{0.93,0.93,0.93}\pgfsys@color@gray@stroke{0.93}\pgfsys@color@gray@fill{0.93}(3)}}(Σm​i​s,o​b​ssubscriptΣ

𝑚𝑖𝑠𝑜𝑏𝑠\Sigma\_{mis,obs})++μ⊙mdirect-product𝜇𝑚\mu\odot mWβsubscript𝑊𝛽W\_{\beta}β𝛽\betaY𝑌Y⊙m¯direct-productabsent¯𝑚\odot\bar{m}⊙m¯direct-productabsent¯𝑚\odot\bar{m}⊙m¯direct-productabsent¯𝑚\odot\bar{m}⊙mdirect-productabsent𝑚\odot mNeumann iterationsNon-linearity

Figure 1: NeuMiss network architecture with a depth of 4 — m¯=1−m¯𝑚1𝑚\bar{m}=1-m. Each weight matrix W(k)superscript𝑊𝑘W^{(k)} corresponds to a simple transformation of the covariance matrix indicated in blue.

#### Approximation of the Gaussian self-masking Bayes predictor

Although our architecture is motivated by the expression of the Bayes predictor in MCAR and MAR settings, a similar architecture can be used to target the prediction function ([2.2](#S2.Ex3 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) for self-masking data.
To see why, let’s first assume that Dm​i​s​Σm​i​s|o​b​s−1≈I​dsubscript𝐷𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1𝐼𝑑D\_{mis}\Sigma\_{mis|obs}^{-1}\approx Id. Then, the self-masking Bayes predictor ([2.2](#S2.Ex3 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) becomes:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆​(Xo​b​s,M)superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀\displaystyle f^{\star}(X\_{obs},M) | ≈β0⋆+⟨βo​b​s⋆,Xo​b​s⟩absentsuperscriptsubscript𝛽0⋆  superscriptsubscript𝛽𝑜𝑏𝑠⋆subscript𝑋𝑜𝑏𝑠\displaystyle\approx\beta\_{0}^{\star}+\bigl{\langle}\beta\_{obs}^{\star},X\_{obs}\rangle |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +⟨βm​i​s⋆,12​(μ~m​i​s+μm​i​s)+12​Σm​i​s,o​b​s​(Σo​b​s)−1​(Xo​b​s−μo​b​s)⟩  superscriptsubscript𝛽𝑚𝑖𝑠⋆12subscript~𝜇𝑚𝑖𝑠subscript𝜇𝑚𝑖𝑠12subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠\displaystyle\quad+\langle\beta\_{mis}^{\star},\frac{1}{2}(\tilde{\mu}\_{mis}+\mu\_{mis})+\frac{1}{2}\Sigma\_{mis,obs}\left(\Sigma\_{obs}\right)^{-1}\left(X\_{obs}-\mu\_{obs}\right)\bigr{\rangle} |  | (9) |

i.e., its expression is the same as for the M(C)AR Bayes predictor ([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) except that μm​i​ssubscript𝜇𝑚𝑖𝑠\mu\_{mis} is replaced by 12​(μ~m​i​s+μm​i​s)12subscript~𝜇𝑚𝑖𝑠subscript𝜇𝑚𝑖𝑠\frac{1}{2}(\tilde{\mu}\_{mis}+\mu\_{mis}) and Σm​i​s,o​b​ssubscriptΣ

𝑚𝑖𝑠𝑜𝑏𝑠\Sigma\_{mis,obs} is scaled down by a factor 1212\frac{1}{2}. Thus, under this approximation, the self-masking Bayes predictor can be modeled by our proposed architecture (just as the M(C)AR Bayes predictor), the only difference being the targeted values for the parameters μ𝜇\mu and Wm​i​xsubscript𝑊𝑚𝑖𝑥W\_{mix} of the network. A less coarse approximation also works: Dm​i​s​Σm​i​s|o​b​s−1≈D^m​i​ssubscript𝐷𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1subscript^𝐷𝑚𝑖𝑠D\_{mis}\Sigma\_{mis|obs}^{-1}\approx\hat{D}\_{mis} where D^^𝐷\hat{D} is a diagonal matrix. In this case, the proposed architecture can perfectly model the self-masking Bayes predictor: the parameter μ𝜇\mu of the network should target (I​d+D^)−1​(μ~+D^​μ)superscript𝐼𝑑^𝐷1~𝜇^𝐷𝜇(Id+\hat{D})^{-1}(\tilde{\mu}+\hat{D}\mu) and Wm​i​xsubscript𝑊𝑚𝑖𝑥W\_{mix} should target (I​d+D^)−1​D^​Σsuperscript𝐼𝑑^𝐷1^𝐷Σ(Id+\hat{D})^{-1}\hat{D}\,\Sigma instead of simply ΣΣ\Sigma in the M(C)AR case. Consequently, our architecture can well approximate the self-masking Bayes predictor by adjusting the values learned for the parameters μ𝜇\mu and Wm​i​xsubscript𝑊𝑚𝑖𝑥W\_{mix} if Dm​i​s​Σm​i​s|o​b​s−1subscript𝐷𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1D\_{mis}\Sigma\_{mis|obs}^{-1} are close to diagonal matrices.

### 3.4 Link with the multilayer perceptron with ReLU activations

A common practice to handle missing values
is to consider as input the data concatenated with the mask
eg in [[13](#bib.bib13)]. The next proposition connects
this practice to Neumman networks.

###### Proposition 3.2 (equivalence MLP - depth-1 NeuMiss network).

Let [X⊙(1−M),M]∈[0,1]d×{0,1}ddirect-product𝑋1𝑀𝑀superscript01𝑑superscript01𝑑\left[X\odot(1-M),M\right]\in[0,1]^{d}\times\left\{0,1\right\}^{d} be an input X𝑋X imputed by 0 concatenated with the mask M𝑀M.

* •

  Let ℋR​e​L​U=(W∈ℝd×2​d,R​e​L​U)subscriptℋ𝑅𝑒𝐿𝑈𝑊
  superscriptℝ𝑑2𝑑𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU}=\left(W\in\mathbb{R}^{d\times 2d},ReLU\right) be a hidden layer which connects [X⊙(1−M),M]direct-product𝑋1𝑀𝑀\left[X\odot(1-M),M\right] to d𝑑d hidden units, and applies a ReLU nonlinearity to the activations.
* •

  Let ℋ⊙M=(W∈ℝd×d,μ,⊙M)\mathcal{H}\_{\odot M}=\left(W\in\mathbb{R}^{d\times d},\mu,\odot M\right) be a hidden layer that connects an input (X−μ)⊙(1−M)direct-product𝑋𝜇1𝑀(X-\mu)\odot(1-M) to d𝑑d hidden units, and applies a ⊙Mdirect-productabsent𝑀\odot M nonlinearity.

Denote by hkR​e​L​Usubscriptsuperscriptℎ𝑅𝑒𝐿𝑈𝑘h^{ReLU}\_{k} and hk⊙Msubscriptsuperscriptℎdirect-productabsent𝑀𝑘h^{\odot M}\_{k} the outputs of the kt​hsuperscript𝑘𝑡ℎk^{th} hidden unit of each layer. Then there exists a configuration of the weights of the hidden layer ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} such that ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M} and ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} have the same hidden units activated for any (Xo​b​s,M)subscript𝑋𝑜𝑏𝑠𝑀(X\_{obs},M), and activated hidden units are such that hkR​e​L​U​(Xo​b​s,M)=hk⊙M​(Xo​b​s,M)+cksubscriptsuperscriptℎ𝑅𝑒𝐿𝑈𝑘subscript𝑋𝑜𝑏𝑠𝑀subscriptsuperscriptℎdirect-productabsent𝑀𝑘subscript𝑋𝑜𝑏𝑠𝑀subscript𝑐𝑘h^{ReLU}\_{k}(X\_{obs},M)=h^{\odot M}\_{k}(X\_{obs},M)+c\_{k} where ck∈ℝsubscript𝑐𝑘ℝc\_{k}\in\mathbb{R}.

Proposition [3.2](#S3.Thmproposition2 "Proposition 3.2 (equivalence MLP - depth-1 NeuMiss network). ‣ 3.4 Link with the multilayer perceptron with ReLU activations ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") states that a hidden layer
ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} can be rewritten as a ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M} layer
up to a constant. Note that, as soon as another layer is stacked after
ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M} or ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU}, this additional
constant can be absorbed into the biases of this new layer.
Thus the weights of ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} can be
learned so as to mimic ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M}. In our case, this means
that a MLP with ReLU activations, one hidden layer of d𝑑d hidden units,
and which operates on the concatenated vector, is closely related to
the 111-depth NeuMiss network (see Figure [1](#S3.F1 "Figure 1 ‣ Multiplying by the mask ‣ 3.3 NeuMiss network architecture: multiplying by the mask ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")),
thereby providing theoretical support for the use of the latter MLP. This
theoretical link completes the results of
[[13](#bib.bib13)], who showed experimentally that in such a MLP
O​(d)𝑂𝑑O(d) units were enough to perform well on Gaussian data, but only provided theoretical results with 2dsuperscript2𝑑2^{d} hidden units.

## 4 Empirical results

### 4.1 The ⊙Mdirect-productabsent𝑀\odot M nonlinearity is crucial to the performance

The specificity of NeuMiss networks resides in the ⊙Mdirect-productabsent𝑀\odot M
nonlinearities, instead of more conventional choices such as ReLU.
Figure [2](#S4.F2.4 "Figure 2 ‣ 4.1 The ⊙𝑀 nonlinearity is crucial to the performance ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") shows how the choice of nonlinearity
impacts the performance as a function of the depth. We compare two
networks that take as input the data imputed by 0 concatenated with
the mask: MLP Deep which has 1 to 10 hidden layers of d𝑑d hidden units followed by ReLU nonlinearities and MLP Wide which has one hidden layer whose width is increased followed by a ReLU nonlinearity. This latter was shown to be consistent given 2dsuperscript2𝑑2^{d} hidden units [[13](#bib.bib13)].

Figure [2](#S4.F2.4 "Figure 2 ‣ 4.1 The ⊙𝑀 nonlinearity is crucial to the performance ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") shows that increasing the capacity (depth) of MLP Deep
fails to improve the performances, unlike with
NeuMiss networks. Similarly, it is also significantly more effective to increase the capacity of the NeuMiss network (depth) than to increase the capacity (width) of MLP Wide.
These results highlight the crucial role played by
the ⊙direct-product\odot nonlinearity.
Finally, the performance of MLP Wide with d𝑑d hidden units is close to
that of NeuMiss with a depth of 1, suggesting that it may rely on the weight
configuration established in Proposition [3.2](#S3.Thmproposition2 "Proposition 3.2 (equivalence MLP - depth-1 NeuMiss network). ‣ 3.4 Link with the multilayer perceptron with ReLU activations ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

Figure 2: Performance as a function of capacity across
architectures — Empirical evolution of the performance for a linear
generating mechanism in MCAR settings.
Data are generated under a linear model with Gaussian covariates in a
MCAR setting (50% missing values, n=105𝑛superscript105n=10^{5}, d=20𝑑20d=20).

![Refer to caption](/html/2007.01627/assets/x1.png)

### 4.2 Approximation learned by the NeuMiss network

The NeuMiss architecture was designed to approximate well the Bayes predictor
([4](#S2.E4 "In Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")). As shown in
Figure [1](#S3.F1 "Figure 1 ‣ Multiplying by the mask ‣ 3.3 NeuMiss network architecture: multiplying by the mask ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), its weights can be chosen so as to express
the Neumann approximation of the Bayes predictor
([7](#S3.E7 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) exactly. We will call this particular
instance of the network, with S(0)superscript𝑆0S^{(0)} set to identity, the analytic network. However, just like LISTA [[5](#bib.bib5)]
learns improved weights compared to the ISTA iterations, the NeuMiss network may learn improved weights compared to the Neumann iterations.
Comparing the performance of the analytic network to its learned
counterpart on simulated MCAR data, Figure [3](#S4.F3 "Figure 3 ‣ 4.3 NeuMiss networks require 𝑂⁢(𝑑²) samples ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") (left)
shows that the learned network requires a much smaller depth compared to
the analytic network to reach a given performance. Moreover, the depth-1
learned network largely outperforms the depth-1 analytic network, which
means that it is able to learn a good initialization S(0)superscript𝑆0S^{(0)}
for the iterates. Figure [3](#S4.F3 "Figure 3 ‣ 4.3 NeuMiss networks require 𝑂⁢(𝑑²) samples ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") also compares the
performance of the learned network with and without residual connections,
and shows that residual connections are not needed for good performance. This observation is another hint that the iterates learned by the network depart from the Neumann ones.

### 4.3 NeuMiss networks require O​(d2)𝑂superscript𝑑2O(d^{2}) samples

Figure [3](#S4.F3 "Figure 3 ‣ 4.3 NeuMiss networks require 𝑂⁢(𝑑²) samples ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") (right) studies the depth for which NeuMiss networks perform well for different number of samples n𝑛n and features
d𝑑d. It outlines that NeuMiss networks work well in regimes with more
than 10 samples available per model parameters, where the number of model
parameters scales as d2superscript𝑑2d^{2}. In general, even with many samples, depth of
more than 5 explore diminishing returns.
Supplementary figure [5](#A2.F5 "Figure 5 ‣ B.1 NeuMiss network scaling law in MNAR ‣ Appendix B Additional results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") shows the same
behavior in various MNAR settings.

![Refer to caption](/html/2007.01627/assets/x2.png)

![Refer to caption](/html/2007.01627/assets/x3.png)

MCAR

Figure 3: Left: learned versus analytic Neumann iterates — NeuMiss analytic is the NeuMiss architecture with weights set to represent ([6](#S3.E6 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), supposing we have access to the ground truth parameters, NeuMiss (resp. NeuMiss res) corresponds to the network without (resp. with) residual connections.
  Right: Required capacity in various settings —
Performance of NeuMiss networks varying the depth in simulations with
different number of samples n𝑛n and of features d𝑑d.

### 4.4 Prediction performance: NeuMiss networks are robust to the missing data mechanism

We now evaluate the performance of NeuMiss networks compared to other
methods under various missing values mechanisms. The data are generated
according to a multivariate Gaussian distribution, with a covariance matrix
Σ=U​U⊤+diag​(ϵ)Σ𝑈superscript𝑈topdiagitalic-ϵ\Sigma=UU^{\top}+\text{diag}(\epsilon), U∈ℝd×d2𝑈superscriptℝ𝑑𝑑2U\in\mathbb{R}^{d\times\frac{d}{2}}, and the entries of U𝑈U drawn from a standard normal
distribution. The noise ϵitalic-ϵ\epsilon is a vector of entries drawn uniformly in [10−2,10−1]superscript102superscript101\left[10^{-2},10^{-1}\right] to make ΣΣ\Sigma full rank. The mean is drawn from a
standard normal distribution. The response Y𝑌Y is generated as a linear
function of the complete data X𝑋X as in equation [1](#S2.E1 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). The
noise is chosen to obtain a signal-to-noise ratio of 10. 50% of entries
on each features are missing, with various missing data mechanisms: MCAR, MAR, Gaussian self-masking and Probit self-masking. The Gaussian self-masking is obtained according to Assumption  [4](#Thmassumption4 "Assumption 4 (Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), while the Probit self-masking is a similar setting where the probability for feature j𝑗j to be missing depends on its value Xjsubscript𝑋𝑗X\_{j} through an inverse probit function.
We compare the performances of the following methods:

* •

  EM: an Expectation-Maximisation algorithm
  [[30](#bib.bib30)] is run to estimate the parameters of the joint probability
  distribution of X𝑋X and Y𝑌Y –Gaussian– with missing values. Then based on this estimated distribution, the prediction is given by taking the expectation of Y𝑌Y given X𝑋X.
* •

  MICE + LR: the data is first imputed using conditional
  imputation as implemented in scikit-learn’s [[25](#bib.bib25)] IterativeImputer, which proceeds by iterative ridge regression. It adapts the well known MICE [[31](#bib.bib31)] algorithm to be able to impute a test set. A linear regression is then fit on the imputed data.
* •

  MLP: A multilayer perceptron as in
  [[13](#bib.bib13)], with one hidden layer followed by a ReLU
  nonlinearity, taking as input the data imputed by 0 concatenated with the
  mask. The width of the hidden layer is varied between d𝑑d and 100​d100𝑑100\,d
  hidden units, and chosen using a validation set. The MLP is trained using
  ADAM and a batch size of 200. The learning rate is initialized to 10−2dsuperscript102𝑑\frac{10^{-2}}{d} and decreased by a factor of 0.2 when the loss stops decreasing for 2 epochs. The training finishes when either the learning rate goes below 5×10−65superscript1065\times 10^{-6} or the maximum number of epochs is reached.
* •

  NeuMiss : The NeuMiss architecture, without residual
  connections, choosing the depth on a validation set.
  The architecture was implemented using PyTorch
  [[24](#bib.bib24)], and
  optimized using stochastic gradient descent and a batch size of 10. The learning rate schedule and stopping criterion are the same as for the MLP.

MCAR
Gaussian self-masking
            Probit self-masking

![Refer to caption](/html/2007.01627/assets/x4.png)

![Refer to caption](/html/2007.01627/assets/x5.png)

![Refer to caption](/html/2007.01627/assets/x6.png)

Figure 4: Predictive performances in various scenarios —
varying missing-value mechanisms, number of samples n𝑛n, and number
of features d𝑑d. All experiments are repeated 20 times. For self-masking
settings, the x-xaxis is in log scale, to accommodate
the large difference between methods.

For MCAR, MAR, and Gaussian self-masking settings, the performance is given
as the obtained R2 score minus the Bayes rate (the closer to 0 the
better), the best achievable R2 knowing the
underlying ground truth parameters. In our experiments, an estimation of the Bayes rate is obtained using the score of the Bayes predictor. For probit self-masking, as we lack an analytical expression for the Bayes predictor, the performance is given with respect to the best performance achieved across all methods. The code to reproduce the experiments is available in GitHub 111https://github.com/marineLM/NeuMiss.

In MCAR settings, figure [4](#S4.F4 "Figure 4 ‣ 4.4 Prediction performance: NeuMiss networks are robust to the missing data mechanism ‣ 4 Empirical results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") shows that, as
expected, EM gives the best results when tractable. Yet, we could
not run it for number of features d≥50𝑑50d\geq 50. NeuMiss is the best
performing method behind EM, in all cases except for n=2×104,d=50formulae-sequence𝑛2superscript104𝑑50n=2\times 10^{4},d=50, where depth of 1 or greater overfit due to the low ratio of
number of parameters to number of samples. In such situation, MLP has
the same expressive power and performs slightly better.
Note that for a high samples-to-parameters ratio
(n=1×105,d=10formulae-sequence𝑛1superscript105𝑑10n=1\times 10^{5},d=10), NeuMiss reaches an almost perfect R​2𝑅2R2 score, less than 1% below the Bayes rate. The results for the MAR setting are very similar to the MCAR results, and are given in supplementary figure [6](#A2.F6 "Figure 6 ‣ B.2 NeuMiss network performances in MAR ‣ Appendix B Additional results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

For the self-masking mechanisms, the NeuMiss network significantly
improves upon the competitors, followed by the MLP. This is even true
for the probit self-masking case for which we have no theoretical results.
The gap between the two
architectures widens as the number of samples increases, with the NeuMiss network benefiting from a large amount of data. These results emphasize
the robustness of NeuMiss and MLP to the missing data mechanism,
including MNAR settings in which EM or conditional imputation do not
enable statistical analysis.

## 5 Discussion and conclusion

Traditionally, statistical models are adapted to missing values
using EM or imputation. However, these require strong
assumptions on the missing values. Rather, we frame the problem as a risk
minimization with a flexible yet tractable function family.
We propose the NeuMiss network, a theoretically-grounded architecture
that handles missing values
using multiplication by the mask as nonlinearities.
It targets the Bayes predictor with
differentiable approximations of the inverses of the various covariance submatrices, thereby reducing complexity by sharing parameters across missing data patterns.
Strong connections between a shallow version of our architecture and
the common practice of inputing the mask to an MLP is established.

The NeuMiss architecture has clear practical
benefits. It is robust to the missing-values mechanism, often
unknown in practice. Moreover its sample and computational complexity
are independent of the number of missing-data patterns, which allows to
work with datasets of higher dimensionality and limited sample sizes.
This work opens many perspectives, in particular using
this network as a building block in larger architectures, *eg* to
tackle nonlinear problems.

## Broader Impact

In our work, we proposed theoretical foundations to justify the use of a
specific neural network architecture in the presence of missing-values.

Neural networks are known for their challenging black-box nature. We believe that such theory leads to a better understanding of the mechanisms at work in neural networks.

Our architecture is tailored for missing data. These are present in
many applications, in particular in social or health data. In these
fields, it is common for under-represented groups to exhibit a higher
percentage of missing values (MNAR mechanism). Dealing with these missing
values will definitely improve prediction for these groups, thereby
reducing potential bias against these exact same groups.

As any predictive algorithm, our proposal can be misused in a variety of
context, including in medical science, for which a proper assessment of
the specific characteristics of the algorithm output is required
(assessing bias in prediction, prevent false conclusion resulting from
misinterpreting outputs). Yet, by improving performance and
understanding of a fundamental challenge in many applications settings,
our work is not facilitating more unethical aspects of AI than ethical
applications. Rather, medical studies that suffer chronically from
limited sample sizes are mostly likely to benefit from the reduced sample
complexity that these advances provide.

## Acknowledgments and Disclosure of Funding

This work was funded by ANR-17-CE23-0018 - DirtyData - Intégration et nettoyage de données pour l’analyse statistique (2017) and the MissingBigData grant from DataIA.

## References

* Audibert et al. [2011]

  Jean-Yves Audibert, Olivier Catoni, and Others.
  Robust linear least squares regression.
  *The Annals of Statistics*, 39(5):2766–2794, 2011.
* Dempster et al. [1977]

  Arthur P Dempster, Nan M Laird, and Donald B Rubin.
  Maximum likelihood from incomplete data via the EM algorithm.
  *Journal of the royal statistical society. Series B
  (methodological)*, pages 1–38, 1977.
* Gilton et al. [2020]

  D. Gilton, G. Ongie, and R. Willett.
  Neumann networks for linear inverse problems in imaging.
  *IEEE Transactions on Computational Imaging*, 6:328–343, 2020.
* Gong et al. [2020]

  Yu Gong, Hossein Hajimirsadeghi, Jiawei He, Megha Nawhal, Thibaut Durand, and
  Greg Mori.
  Variational selective autoencoder.
  In Cheng Zhang, Francisco Ruiz, Thang Bui, Adji Bousso Dieng, and
  Dawen Liang, editors, *Proceedings of The 2nd Symposium on Advances in
  Approximate Bayesian Inference*, volume 118 of *Proceedings of Machine
  Learning Research*, pages 1–17. PMLR, 08 Dec 2020.
* Gregor and LeCun [2010]

  Karol Gregor and Yann LeCun.
  Learning fast approximations of sparse coding.
  In *Proceedings of the 27th International Conference on
  International Conference on Machine Learning*, pages 399–406, 2010.
* Hastie et al. [2015]

  Trevor Hastie, Rahul Mazumder, Jason D. Lee, and Reza Zadeh.
  Matrix completion and low-rank svd via fast alternating least
  squares.
  *J. Mach. Learn. Res.*, 16(1):3367–3402,
  January 2015.
  ISSN 1532-4435.
* Hernández-Lobato et al. [2014]

  José Miguel Hernández-Lobato, Neil Houlsby, and Zoubin Ghahramani.
  Probabilistic matrix factorization with non-random missing data.
  In *International Conference on Machine Learning*, pages
  1512–1520, 2014.
* Hwang [2004]

  Suk-Geun Hwang.
  Cauchy’s Interlace Theorem for Eigenvalues of Hermitian
  Matrices.
  *The American Mathematical Monthly*, 111(2):157, February 2004.
  ISSN 00029890.
  doi: 10.2307/4145217.
* Ibrahim et al. [1999]

  Joseph G Ibrahim, Stuart R Lipsitz, and M-H Chen.
  Missing covariates in generalized linear models when the missing data
  mechanism is non-ignorable.
  *Journal of the Royal Statistical Society: Series B (Statistical
  Methodology)*, 61(1):173–190, 1999.
* Imke Mayer Julie Josse and Vialaneix [2019]

  Nicholas Tierney Imke Mayer Julie Josse and Nathalie Vialaneix.
  R-miss-tastic: a unified platform for missing values methods and
  workflows, 2019.
* Josse et al. [2019]

  Julie Josse, Nicolas Prost, Erwan Scornet, and Gaël Varoquaux.
  On the consistency of supervised learning with missing values.
  *arXiv preprint arXiv:1902.06931*, 2019.
* Kim and Ying [2018]

  J K Kim and Z Ying.
  *Data Missing Not at Random, special issue*.
  Statistica Sinica. Institute of Statistical Science, Academia Sinica,
  2018.
* Le Morvan et al. [2020]

  Marine Le Morvan, Nicolas Prost, Julie Josse, Erwan Scornet, and Gaël
  Varoquaux.
  Linear predictor on linearly-generated data with missing values: non
  consistency and solutions.
  *arXiv preprint arXiv:2002.00658*, 2020.
* Little and Rubin [2019]

  Roderick J A Little and Donald B Rubin.
  *Statistical analysis with missing data*.
  John Wiley & Sons, 2019.
* Ma et al. [2018]

  Chao Ma, Sebastian Tschiatschek, Konstantina Palla, José Miguel
  Hernández-Lobato, Sebastian Nowozin, and Cheng Zhang.
  Eddi: Efficient dynamic discovery of high-value information with
  partial vae.
  *arXiv preprint arXiv:1809.11142*, 2018.
* Ma and Chen [2019]

  Wei Ma and George H Chen.
  Missing not at random in matrix completion: The effectiveness of
  estimating missingness probabilities under a low nuclear norm assumption.
  In *Advances in Neural Information Processing Systems*, pages
  14871–14880, 2019.
* Majumdar and Majumdar [2019]

  Rajeshwari Majumdar and Suman Majumdar.
  On the conditional distribution of a multivariate normal given a
  transformation–the linear case.
  *Heliyon*, 5(2):e01136, 2019.
* Marlin and Zemel [2009]

  Benjamin M Marlin and Richard S Zemel.
  Collaborative prediction and ranking with non-random missing data.
  In *Proceedings of the third ACM conference on Recommender
  systems*, pages 5–12. ACM, 2009.
* Mattei and Frellsen [2019]

  Pierre-Alexandre Mattei and Jes Frellsen.
  MIWAE: Deep generative modelling and imputation of incomplete data
  sets.
  In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
  *Proceedings of the 36th International Conference on Machine Learning*,
  volume 97 of *Proceedings of Machine Learning Research*, pages
  4413–4423, Long Beach, California, USA, 09–15 Jun 2019. PMLR.
* Miao et al. [2016]

  Wang Miao, Peng Ding, and Zhi Geng.
  Identifiability of normal and normal mixture models with nonignorable
  missing data.
  *Journal of the American Statistical Association*, 111(516):1673–1683, 2016.
* Mohan and Pearl [2019]

  K Mohan and J Pearl.
  Graphical Models for Processing Missing Data.
  Technical Report R-473-L, Department of Computer Science, University
  of California, Los Angeles, CA, 2019.
* Nabi et al. [2020]

  Razieh Nabi, Rohit Bhattacharya, and Ilya Shpitser.
  Full law identification in graphical models of missing data:
  Completeness results.
  *arXiv preprint arXiv:2004.04872*, 2020.
* Nazabal et al. [2018]

  Alfredo Nazabal, Pablo M Olmos, Zoubin Ghahramani, and Isabel Valera.
  Handling incomplete heterogeneous data using vaes.
  *arXiv preprint arXiv:1807.03653*, 2018.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
  Pytorch: An imperative style, high-performance deep learning library.
  In *Advances in Neural Information Processing Systems*, pages
  8024–8035, 2019.
* Pedregosa et al. [2011]

  F Pedregosa, G Varoquaux, A Gramfort, V Michel, B Thirion, O Grisel, M Blondel,
  P Prettenhofer, R Weiss, V Dubourg, J Vanderplas, A Passos, D Cournapeau,
  M Brucher, M Perrot, and E Duchesnay.
  Scikit-learn: Machine Learning in Python .
  *Journal of Machine Learning Research*, 12:2825–2830,
  2011.
* Rosenbaum and Rubin [1984]

  Paul R Rosenbaum and Donald B Rubin.
  Reducing bias in observational studies using subclassification on
  the propensity score.
  *Journal of the American Statistical Association*, 79(387):516–524, 1984.
  doi: 10.2307/2288398.
* Rubin [1976]

  Donald B Rubin.
  Inference and missing data.
  *Biometrika*, 63(3):581–592, 1976.
* Seber and Lee [2003]

  George AF Seber and Alan J Lee.
  Wiley series in probability and statistics.
  *Linear Regression Analysis*, pages 36–44, 2003.
* Tang et al. [2003]

  Gong Tang, Roderick JA Little, and Trivellore E Raghunathan.
  Analysis of multivariate missing data with nonignorable nonresponse.
  *Biometrika*, 90(4):747–764, 2003.
* to R by Alvaro A. Novo. Original by Joseph L.
  Schafer <jls@stat.psu.edu>. [2013]

  Ported to R by Alvaro A. Novo. Original by Joseph L.
  Schafer <jls@stat.psu.edu>.
  *norm: Analysis of multivariate normal datasets with missing
  values*, 2013.
  R package version 1.0-9.5.
* van Buuren [2018]

  S van Buuren.
  *Flexible Imputation of Missing Data*.
  Chapman and Hall/CRC, Boca Raton, FL, 2018.
* Wang et al. [2019]

  Xiaojie Wang, Rui Zhang, Yu Sun, and Jianzhong Qi.
  Doubly robust joint learning for recommendation on data missing not
  at random.
  In *International Conference on Machine Learning*, pages
  6638–6647, 2019.
* Xin et al. [2016]

  Bo Xin, Yizhou Wang, Wen Gao, and David Wipf.
  Maximal Sparsity with Deep Networks?
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 4340–4348, 2016.
* Yoon et al. [2018]

  Jinsung Yoon, James Jordon, and Mihaela Schaar.
  GAIN: Missing Data Imputation using Generative Adversarial Nets.
  In *International Conference on Machine Learning*, pages
  5675–5684, 2018.

Supplementary materials – NeuMiss networks: differentiable programming for supervised learning with missing values

## Appendix A Proofs

### A.1 Proof of Lemma [1](#Thmlemma1 "Lemma 1 (General expression of the Bayes predictor). ‣ A.1 Proof of Lemma 1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

###### Lemma 1 (General expression of the Bayes predictor).

Assume that the data are generated via the linear model defined in equation ([1](#S2.E1 "In 2.1 Problem statement: supervised learning with missing values ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), then the Bayes predictor takes the form

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆​(Xo​b​s​(M),M)=β0⋆+⟨βo​b​s​(M)⋆,Xo​b​s​(M)⟩+⟨βm​i​s​(M)⋆,𝔼​[Xm​i​s​(M)|M,Xo​b​s​(M)]⟩,superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀𝑀superscriptsubscript𝛽0⋆  superscriptsubscript𝛽𝑜𝑏𝑠𝑀⋆subscript𝑋𝑜𝑏𝑠𝑀  subscriptsuperscript𝛽⋆𝑚𝑖𝑠𝑀𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀  𝑀subscript𝑋𝑜𝑏𝑠𝑀f^{\star}(X\_{obs(M)},M)=\beta\_{0}^{\star}+\langle\beta\_{obs(M)}^{\star},X\_{obs(M)}\rangle+\langle\beta^{\star}\_{mis(M)},\mathbb{E}[X\_{mis(M)}|M,X\_{obs(M)}]\rangle, |  | (10) |

where
(βo​b​s​(M)⋆,βm​i​s​(M)⋆

superscriptsubscript𝛽𝑜𝑏𝑠𝑀⋆superscriptsubscript𝛽𝑚𝑖𝑠𝑀⋆\beta\_{obs(M)}^{\star},\beta\_{mis(M)}^{\star}) correspond to the decomposition of the regression coefficients in observed and missing elements.

###### Proof of Lemma [1](#Thmlemma1 "Lemma 1 (General expression of the Bayes predictor). ‣ A.1 Proof of Lemma 1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

By definition of the linear model, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | fX~⋆​(X~)subscriptsuperscript𝑓⋆~𝑋~𝑋\displaystyle f^{\star}\_{\widetilde{X}}(\widetilde{X}) | =𝔼​[Y|X~]absent𝔼delimited-[]conditional𝑌~𝑋\displaystyle=\mathbb{E}[Y|\widetilde{X}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[β0⋆+⟨β⋆,X⟩|M,Xo​b​s​(M)]absent𝔼delimited-[]subscriptsuperscript𝛽⋆0conditional  superscript𝛽⋆𝑋  𝑀subscript𝑋𝑜𝑏𝑠𝑀\displaystyle=\mathbb{E}[\beta^{\star}\_{0}+\langle\beta^{\star},X\rangle~{}|~{}M,X\_{obs(M)}] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =β0⋆+⟨βo​b​s​(M)⋆,Xo​b​s​(M)⟩+⟨βm​i​s​(M)⋆,𝔼​[Xm​i​s​(M)|M,Xo​b​s​(M)]⟩.absentsuperscriptsubscript𝛽0⋆  subscriptsuperscript𝛽⋆𝑜𝑏𝑠𝑀subscript𝑋𝑜𝑏𝑠𝑀  subscriptsuperscript𝛽⋆𝑚𝑖𝑠𝑀𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀  𝑀subscript𝑋𝑜𝑏𝑠𝑀\displaystyle=\beta\_{0}^{\star}+\langle\beta^{\star}\_{obs(M)},X\_{obs(M)}\rangle+\langle\beta^{\star}\_{mis(M)},\mathbb{E}[X\_{mis(M)}~{}|~{}M,X\_{obs(M)}]\rangle. |  |

∎

### A.2 Proof of Lemma [2](#Thmlemma2 "Lemma 2 (Product of two multivariate gaussians). ‣ A.2 Proof of Lemma 2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

###### Lemma 2 (Product of two multivariate gaussians).

Let f​(X)=exp⁡((X−a)⊤​A−1​(X−a))𝑓𝑋superscript𝑋𝑎topsuperscript𝐴1𝑋𝑎f(X)=\exp\left((X-a)^{\top}A^{-1}(X-a)\right) and g​(X)=exp⁡((X−b)⊤​B−1​(X−b))𝑔𝑋superscript𝑋𝑏topsuperscript𝐵1𝑋𝑏g(X)=\exp\left((X-b)^{\top}B^{-1}(X-b)\right) be two Gaussian functions, with A𝐴A and B𝐵B positive semidefinite matrices. Then the product f​(X)​g​(X)𝑓𝑋𝑔𝑋f(X)g(X) is another gaussian function given by:

|  |  |  |
| --- | --- | --- |
|  | f(X)g(X)=exp(−12(a−b)⊤(A+B)−1(a−b)))exp(−12(X−μp)⊤Σp−1(X−μp))f(X)g(X)=\exp\left(-\frac{1}{2}(a-b)^{\top}(A+B)^{-1}(a-b))\right)\exp\left(-\frac{1}{2}(X-\mu\_{p})^{\top}\Sigma\_{p}^{-1}(X-\mu\_{p})\right) |  |

where μpsubscript𝜇𝑝\mu\_{p} and ΣpsubscriptΣ𝑝\Sigma\_{p} depend on a𝑎a, A𝐴A, b𝑏b and B𝐵B.

###### Proof of Lemma [2](#Thmlemma2 "Lemma 2 (Product of two multivariate gaussians). ‣ A.2 Proof of Lemma 2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

Identifying the second and first order terms in X𝑋X we get:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Σp−1superscriptsubscriptΣ𝑝1\displaystyle\Sigma\_{p}^{-1} | =A−1+B−1absentsuperscript𝐴1superscript𝐵1\displaystyle=A^{-1}+B^{-1} |  | (11) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Σp−1​μpsuperscriptsubscriptΣ𝑝1subscript𝜇𝑝\displaystyle\Sigma\_{p}^{-1}\mu\_{p} | =A−1​a+B−1​babsentsuperscript𝐴1𝑎superscript𝐵1𝑏\displaystyle=A^{-1}a+B^{-1}b |  | (12) |

By completing the square, the product can be rewritten as:

|  |  |  |
| --- | --- | --- |
|  | f(X)g(X)=exp(−12(a⊤A−1a+b⊤B−1b−μp⊤Σp−1μp)exp(−12(X−μp)⊤Σp−1(X−μp))f(X)g(X)=\exp\left(-\frac{1}{2}(a^{\top}A^{-1}a+b^{\top}B^{-1}b-\mu\_{p}^{\top}\Sigma\_{p}^{-1}\mu\_{p}\right)\exp\left(-\frac{1}{2}(X-\mu\_{p})^{\top}\Sigma\_{p}^{-1}(X-\mu\_{p})\right) |  |

Let’s now simplify the scaling factor:

|  |  |  |  |
| --- | --- | --- | --- |
|  | c𝑐\displaystyle c | =a⊤​A−1​a+b⊤​B−1​b−μp⊤​Σp−1​μpabsentsuperscript𝑎topsuperscript𝐴1𝑎superscript𝑏topsuperscript𝐵1𝑏superscriptsubscript𝜇𝑝topsuperscriptsubscriptΣ𝑝1subscript𝜇𝑝\displaystyle=a^{\top}A^{-1}a+b^{\top}B^{-1}b-\mu\_{p}^{\top}\Sigma\_{p}^{-1}\mu\_{p} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =a⊤​A−1​a+b⊤​B−1​b−(a⊤​A−1​(A−1+B−1)−1+b⊤​B−1​(A−1+B−1)−1)​(A−1​a+B−1​b)absentsuperscript𝑎topsuperscript𝐴1𝑎superscript𝑏topsuperscript𝐵1𝑏superscript𝑎topsuperscript𝐴1superscriptsuperscript𝐴1superscript𝐵11superscript𝑏topsuperscript𝐵1superscriptsuperscript𝐴1superscript𝐵11superscript𝐴1𝑎superscript𝐵1𝑏\displaystyle=a^{\top}A^{-1}a+b^{\top}B^{-1}b-\left(a^{\top}A^{-1}(A^{-1}+B^{-1})^{-1}+b^{\top}B^{-1}(A^{-1}+B^{-1})^{-1}\right)\left(A^{-1}a+B^{-1}b\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =a⊤​(A−1−A−1​(A−1+B−1)−1​A−1)​a+b⊤​(B−1−B−1​(A−1+B−1)−1​B−1)​babsentsuperscript𝑎topsuperscript𝐴1superscript𝐴1superscriptsuperscript𝐴1superscript𝐵11superscript𝐴1𝑎superscript𝑏topsuperscript𝐵1superscript𝐵1superscriptsuperscript𝐴1superscript𝐵11superscript𝐵1𝑏\displaystyle=a^{\top}(A^{-1}-A^{-1}(A^{-1}+B^{-1})^{-1}A^{-1})a+b^{\top}(B^{-1}-B^{-1}(A^{-1}+B^{-1})^{-1}B^{-1})b |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | −2​a⊤​(A−1​(A−1+B−1)−1​B−1)​b2superscript𝑎topsuperscript𝐴1superscriptsuperscript𝐴1superscript𝐵11superscript𝐵1𝑏\displaystyle\quad-2a^{\top}(A^{-1}(A^{-1}+B^{-1})^{-1}B^{-1})b |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =a⊤​(A+B)−1​a+b⊤​(A+B)−1​b−2​a⊤​(A+B)−1​babsentsuperscript𝑎topsuperscript𝐴𝐵1𝑎superscript𝑏topsuperscript𝐴𝐵1𝑏2superscript𝑎topsuperscript𝐴𝐵1𝑏\displaystyle=a^{\top}(A+B)^{-1}a+b^{\top}(A+B)^{-1}b-2a^{\top}(A+B)^{-1}b |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(a−b)⊤​(A+B)−1​(a−b)absentsuperscript𝑎𝑏topsuperscript𝐴𝐵1𝑎𝑏\displaystyle=(a-b)^{\top}(A+B)^{-1}(a-b) |  |

The third equality is true because A𝐴A and B𝐵B are symmetric. The fourth equality uses the Woodbury identity and the fact that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (A−1​(A−1+B−1)−1​B−1)superscript𝐴1superscriptsuperscript𝐴1superscript𝐵11superscript𝐵1\displaystyle(A^{-1}(A^{-1}+B^{-1})^{-1}B^{-1}) | =(B​(A−1+B−1)​A)−1absentsuperscript𝐵superscript𝐴1superscript𝐵1𝐴1\displaystyle=\left(B(A^{-1}+B^{-1})A\right)^{-1} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(B​A−1​A+B​B−1​A)−1absentsuperscript𝐵superscript𝐴1𝐴𝐵superscript𝐵1𝐴1\displaystyle=\left(BA^{-1}A+BB^{-1}A\right)^{-1} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =(B+A)−1absentsuperscript𝐵𝐴1\displaystyle=\left(B+A\right)^{-1} |  |

The last equality allows to conclude the proof.
∎

### A.3 Proof of Proposition [2.1](#S2.Thmproposition1 "Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

See [2.1](#S2.Thmproposition1 "Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

Lemma [1](#Thmlemma1 "Lemma 1 (General expression of the Bayes predictor). ‣ A.1 Proof of Lemma 1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") gives the general expression of the Bayes predictor for any data distribution and missing data mechanism. From this expression, on can see that the crucial step to compute the Bayes predictor is computing 𝔼​[Xm​i​s|M,Xo​b​s]𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠

𝑀subscript𝑋𝑜𝑏𝑠\mathbb{E}[X\_{mis}|M,X\_{obs}], or in other words, 𝔼​[Xj|M,Xo​b​s]𝔼delimited-[]conditionalsubscript𝑋𝑗

𝑀subscript𝑋𝑜𝑏𝑠\mathbb{E}[X\_{j}|M,X\_{obs}] for all j∈m​i​s𝑗𝑚𝑖𝑠j\in mis. In order to compute this expectation, we will characterize the distribution P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗

𝑀subscript𝑋𝑜𝑏𝑠P(X\_{j}|M,X\_{obs}) for all j∈m​i​s𝑗𝑚𝑖𝑠j\in mis. Let m​i​s′​(M,j)=m​i​s​(M)∖{j}𝑚𝑖superscript𝑠′𝑀𝑗𝑚𝑖𝑠𝑀𝑗mis^{\prime}(M,j)=mis(M)\setminus\{j\}. For clarity, when there is no ambiguity we will just write m​i​s′𝑚𝑖superscript𝑠′mis^{\prime}. Using the sum and product rules of probability, we have:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗  𝑀subscript𝑋𝑜𝑏𝑠\displaystyle P(X\_{j}|M,X\_{obs}) | =P​(M,Xj,Xo​b​s)P​(M,Xo​b​s)absent𝑃𝑀subscript𝑋𝑗subscript𝑋𝑜𝑏𝑠𝑃𝑀subscript𝑋𝑜𝑏𝑠\displaystyle=\frac{P(M,X\_{j},X\_{obs})}{P(M,X\_{obs})} |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∫P​(M,Xj,Xo​b​s,Xm​i​s′)​dXm​i​s′∫∫P​(M,Xj,Xo​b​s,Xm​i​s′)​dXm​i​s′​dXjabsent𝑃𝑀subscript𝑋𝑗subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃𝑀subscript𝑋𝑗subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{\int P(M,X\_{j},X\_{obs},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}}{\int\int P(M,X\_{j},X\_{obs},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∫P​(M|Xo​b​s,Xj,Xm​i​s′)​P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′∫∫P​(M|Xo​b​s,Xj,Xm​i​s′)​P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′​dXjabsent𝑃conditional𝑀  subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃conditional𝑀  subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{\int P(M|X\_{obs},X\_{j},X\_{mis^{\prime}})P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}}{\int\int P(M|X\_{obs},X\_{j},X\_{mis^{\prime}})P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (15) |

In the MCAR case, for all m∈{0,1}d,ℙ​(M=m|X)=ℙ​(M=m)formulae-sequence𝑚superscript01𝑑ℙ𝑀conditional𝑚𝑋ℙ𝑀𝑚m\in\{0,1\}^{d},\mathds{P}(M=m|X)=\mathds{P}(M=m), thus we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗  𝑀subscript𝑋𝑜𝑏𝑠\displaystyle P(X\_{j}|M,X\_{obs}) | =P​(M)​∫P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′P​(M)​∫∫P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′​dXjabsent𝑃𝑀𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃𝑀𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{P(M)\int P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}}{P(M)\int\int P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =P​(Xo​b​s,Xj)P​(Xo​b​s)absent𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗𝑃subscript𝑋𝑜𝑏𝑠\displaystyle=\frac{P(X\_{obs},X\_{j})}{P(X\_{obs})} |  | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =P​(Xj|Xo​b​s)absent𝑃conditionalsubscript𝑋𝑗subscript𝑋𝑜𝑏𝑠\displaystyle=P(X\_{j}|X\_{obs}) |  | (18) |

On the other hand, assuming MAR mechanism, that is, for all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, P​(M=m|X)=P​(M=m|Xo​b​s​(m))𝑃𝑀conditional𝑚𝑋𝑃𝑀conditional𝑚subscript𝑋𝑜𝑏𝑠𝑚P(M=m|X)=P(M=m|X\_{obs(m)}), we have, given equation ([15](#A1.E15 "In A.3 Proof of Proposition 2.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")),

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗  𝑀subscript𝑋𝑜𝑏𝑠\displaystyle P(X\_{j}|M,X\_{obs}) | =P​(M|Xo​b​s)​∫P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′P​(M|Xo​b​s)​∫∫P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′​dXjabsent𝑃conditional𝑀subscript𝑋𝑜𝑏𝑠𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃conditional𝑀subscript𝑋𝑜𝑏𝑠𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{P(M|X\_{obs})\int P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}}{P(M|X\_{obs})\int\int P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =P​(Xo​b​s,Xj)P​(Xo​b​s)absent𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗𝑃subscript𝑋𝑜𝑏𝑠\displaystyle=\frac{P(X\_{obs},X\_{j})}{P(X\_{obs})} |  | (20) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =P​(Xj|Xo​b​s)absent𝑃conditionalsubscript𝑋𝑗subscript𝑋𝑜𝑏𝑠\displaystyle=P(X\_{j}|X\_{obs}) |  | (21) |

Therefore, if the missing data mechanism is MCAR or MAR, we have, according to equation ([18](#A1.E18 "In A.3 Proof of Proposition 2.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) and ([21](#A1.E21 "In A.3 Proof of Proposition 2.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")),

|  |  |  |
| --- | --- | --- |
|  | 𝔼​[Xm​i​s​(M)|M,Xo​b​s​(M)]=𝔼​[Xm​i​s​(M)|Xo​b​s​(M)].𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀  𝑀subscript𝑋𝑜𝑏𝑠𝑀𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\mathbb{E}[X\_{mis(M)}~{}|~{}M,X\_{obs(M)}]=\mathbb{E}[X\_{mis(M)}~{}|X\_{obs(M)}]. |  |

Since X𝑋X is a Gaussian vector distributed as 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma), we know that the conditional expectation 𝔼​[Xm​i​s​(M)|Xo​b​s​(M)]𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀subscript𝑋𝑜𝑏𝑠𝑀\mathbb{E}[X\_{mis(M)}~{}|X\_{obs(M)}] satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼[Xm​i​s​(m)|Xo​b​s​(m)]=μm​i​s​(m)+Σm​i​s​(m),o​b​s​(m)(Σo​b​s​(m))−1(Xo​b​s​(m)−μo​b​s​(m)),\displaystyle\mathbb{E}\left[X\_{mis(m)}~{}\middle|~{}X\_{obs(m)}\right]=\mu\_{mis(m)}+\Sigma\_{mis(m),obs(m)}\left(\Sigma\_{obs(m)}\right)^{-1}\left(X\_{obs(m)}-\mu\_{obs(m)}\right), |  | (22) |

[see, e.g., [17](#bib.bib17)]. This concludes the proof according to Lemma [1](#Thmlemma1 "Lemma 1 (General expression of the Bayes predictor). ‣ A.1 Proof of Lemma 1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

### A.4 Proof of Proposition [2.2](#S2.Thmproposition2 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

See [2.2](#S2.Thmproposition2 "Proposition 2.2 (Bayes predictor with Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

In the Gaussian self-masking case, according to Assumption [4](#Thmassumption4 "Assumption 4 (Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), the probability factorizes as P​(M=m|X)=P​(Mm​i​s​(m)=1|Xm​i​s​(m))​P​(Mo​b​s​(m)=0|Xo​b​s​(m))𝑃𝑀conditional𝑚𝑋𝑃subscript𝑀𝑚𝑖𝑠𝑚conditional1subscript𝑋𝑚𝑖𝑠𝑚𝑃subscript𝑀𝑜𝑏𝑠𝑚conditional0subscript𝑋𝑜𝑏𝑠𝑚P(M=m|X)=P(M\_{mis(m)}=1|X\_{mis(m)})P(M\_{obs(m)}=0|X\_{obs(m)}). Equation [15](#A1.E15 "In A.3 Proof of Proposition 2.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") can thus be rewritten as:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗  𝑀subscript𝑋𝑜𝑏𝑠\displaystyle P(X\_{j}|M,X\_{obs}) | =P​(Mo​b​s=0|Xo​b​s)​∫P​(Mm​i​s=1|Xm​i​s)​P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′P​(Mo​b​s=0|Xo​b​s)​∫∫P​(Mm​i​s=1|Xm​i​s)​P​(Xo​b​s,Xj,Xm​i​s′)​dXm​i​s′​dXjabsent𝑃subscript𝑀𝑜𝑏𝑠conditional0subscript𝑋𝑜𝑏𝑠𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃subscript𝑀𝑜𝑏𝑠conditional0subscript𝑋𝑜𝑏𝑠𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠𝑃subscript𝑋𝑜𝑏𝑠subscript𝑋𝑗subscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{P(M\_{obs}=0|X\_{obs})\int P(M\_{mis}=1|X\_{mis})P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}}{P(M\_{obs}=0|X\_{obs})\int\int P(M\_{mis}=1|X\_{mis})P(X\_{obs},X\_{j},X\_{mis^{\prime}})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (23) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∫P​(Mm​i​s=1|Xm​i​s)​P​(Xm​i​s|Xo​b​s)​dXm​i​s′∫∫P​(Mm​i​s=1|Xm​i​s)​P​(Xm​i​s|Xo​b​s)​dXm​i​s′​dXjabsent𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{\int P(M\_{mis}=1|X\_{mis})P(X\_{mis}|X\_{obs})\mathrm{d}X\_{mis^{\prime}}}{\int\int P(M\_{mis}=1|X\_{mis})P(X\_{mis}|X\_{obs})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (24) |

Let D𝐷D be the diagonal matrix such that diag​(D)=σ~2diag𝐷superscript~𝜎2\mathrm{diag}(D)=\widetilde{\sigma}^{2}, where σ~~𝜎\widetilde{\sigma} is defined in Assumption [4](#Thmassumption4 "Assumption 4 (Gaussian self-masking). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). Then the masking probability reads:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(Mm​i​s=1|Xm​i​s)=∏k∈m​i​sdKk​exp⁡(−12​(Xm​i​s−μ~m​i​s)​(Dm​i​s,m​i​s)−1​(Xm​i​s−μ~m​i​s))𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠superscriptsubscriptproduct𝑘𝑚𝑖𝑠𝑑subscript𝐾𝑘12subscript𝑋𝑚𝑖𝑠subscript~𝜇𝑚𝑖𝑠superscriptsubscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1subscript𝑋𝑚𝑖𝑠subscript~𝜇𝑚𝑖𝑠P(M\_{mis}=1|X\_{mis})=\prod\_{k\in mis}^{d}K\_{k}\exp\left(-\frac{1}{2}(X\_{mis}-\widetilde{\mu}\_{mis})(D\_{mis,mis})^{-1}(X\_{mis}-\widetilde{\mu}\_{mis})\right) |  | (25) |

Using the conditional Gaussian formula, we have P​(Xm​i​s|Xo​b​s)=𝒩​(Xm​i​s|μm​i​s|o​b​s,Σm​i​s|o​b​s)𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠𝒩conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠subscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠P(X\_{mis}|X\_{obs})=\mathcal{N}(X\_{mis}|\mu\_{mis|obs},\Sigma\_{mis|obs}) with

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μm​i​s|o​b​ssubscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\mu\_{mis|obs} | =μm​i​s+Σm​i​s,o​b​s​Σo​b​s,o​b​s−1​(Xo​b​s−μo​b​s)absentsubscript𝜇𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ  𝑜𝑏𝑠𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠\displaystyle=\mu\_{mis}+\Sigma\_{mis,obs}\Sigma\_{obs,obs}^{-1}\left(X\_{obs}-\mu\_{obs}\right) |  | (26) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Σm​i​s|o​b​ssubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\Sigma\_{mis|obs} | =Σm​i​s,m​i​s−Σm​i​s,o​b​s​Σo​b​s−1​Σo​b​s,m​i​sabsentsubscriptΣ  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠\displaystyle=\Sigma\_{mis,mis}-\Sigma\_{mis,obs}\Sigma\_{obs}^{-1}\Sigma\_{obs,mis} |  | (27) |

Thus, according to equation ([25](#A1.E25 "In A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), P​(Mm​i​s=1|Xm​i​s)𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠P(M\_{mis}=1|X\_{mis}) and P​(Xm​i​s|Xo​b​s)𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠P(X\_{mis}|X\_{obs}) are Gaussian functions of Xm​i​ssubscript𝑋𝑚𝑖𝑠X\_{mis}. By Lemma [2](#Thmlemma2 "Lemma 2 (Product of two multivariate gaussians). ‣ A.2 Proof of Lemma 2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), their product is also a Gaussian function given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(Mm​i​s=1|Xm​i​s)​P​(Xm​i​s|Xo​b​s)=K​exp⁡(−12​(Xm​i​s−aM)⊤​(AM)−1​(Xm​i​s−aM))𝑃subscript𝑀𝑚𝑖𝑠conditional1subscript𝑋𝑚𝑖𝑠𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠𝐾12superscriptsubscript𝑋𝑚𝑖𝑠subscript𝑎𝑀topsuperscriptsubscript𝐴𝑀1subscript𝑋𝑚𝑖𝑠subscript𝑎𝑀P(M\_{mis}=1|X\_{mis})P(X\_{mis}|X\_{obs})=K\exp\left(-\frac{1}{2}(X\_{mis}-a\_{M})^{\top}\left(A\_{M}\right)^{-1}(X\_{mis}-a\_{M})\right) |  | (28) |

where aMsubscript𝑎𝑀a\_{M} and AMsubscript𝐴𝑀A\_{M} depend on the missingness pattern and

|  |  |  |  |
| --- | --- | --- | --- |
|  | K=∏k∈m​i​sdKk(2​π)|m​i​s|​|Σm​i​s|o​b​s|​exp⁡(−12​(μ~m​i​s−μm​i​s|o​b​s)⊤​(Σm​i​s|o​b​s+Dm​i​s,m​i​s)−1​(μ~m​i​s−μm​i​s|o​b​s))𝐾superscriptsubscriptproduct𝑘𝑚𝑖𝑠𝑑subscript𝐾𝑘superscript2𝜋𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠12superscriptsubscript~𝜇𝑚𝑖𝑠subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠topsuperscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠subscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1subscript~𝜇𝑚𝑖𝑠subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle K=\prod\_{k\in mis}^{d}\frac{K\_{k}}{\sqrt{(2\pi)^{|mis|}|\Sigma\_{mis|obs}|}}\exp\left(-\frac{1}{2}(\widetilde{\mu}\_{mis}-\mu\_{mis|obs})^{\top}(\Sigma\_{mis|obs}+D\_{mis,mis})^{-1}(\widetilde{\mu}\_{mis}-\mu\_{mis|obs})\right) |  | (29) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (AM)−1=Dm​i​s,m​i​s−1+Σm​i​s|o​b​s−1superscriptsubscript𝐴𝑀1superscriptsubscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1\displaystyle\left(A\_{M}\right)^{-1}=D\_{mis,mis}^{-1}+\Sigma\_{mis|obs}^{-1} |  | (30) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (AM)−1​aM=Dm​i​s,m​i​s−1​μ~m​i​s+Σm​i​s|o​b​s−1​μm​i​s|o​b​ssuperscriptsubscript𝐴𝑀1subscript𝑎𝑀superscriptsubscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1subscript~𝜇𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\left(A\_{M}\right)^{-1}a\_{M}=D\_{mis,mis}^{-1}\widetilde{\mu}\_{mis}+\Sigma\_{mis|obs}^{-1}\mu\_{mis|obs} |  | (31) |

Because K𝐾K does not depend on Xm​i​ssubscript𝑋𝑚𝑖𝑠X\_{mis}, it simplifies from eq [24](#A1.E24 "In A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). As a result we get:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | P​(Xj|M,Xo​b​s)𝑃conditionalsubscript𝑋𝑗  𝑀subscript𝑋𝑜𝑏𝑠\displaystyle P(X\_{j}|M,X\_{obs}) | =∫𝒩​(Xm​i​s|aM,AM)​dXm​i​s′∫∫𝒩​(Xm​i​s|aM,AM)​dXm​i​s′​dXjabsent𝒩conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑎𝑀subscript𝐴𝑀differential-dsubscript𝑋𝑚𝑖superscript𝑠′𝒩conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑎𝑀subscript𝐴𝑀differential-dsubscript𝑋𝑚𝑖superscript𝑠′differential-dsubscript𝑋𝑗\displaystyle=\frac{\int\mathcal{N}(X\_{mis}|a\_{M},A\_{M})\mathrm{d}X\_{mis^{\prime}}}{\int\int\mathcal{N}(X\_{mis}|a\_{M},A\_{M})\mathrm{d}X\_{mis^{\prime}}\mathrm{d}X\_{j}} |  | (32) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =𝒩​(Xj|(aM)j,(AM)j,j)absent𝒩conditionalsubscript𝑋𝑗  subscriptsubscript𝑎𝑀𝑗subscriptsubscript𝐴𝑀  𝑗𝑗\displaystyle=\mathcal{N}(X\_{j}|(a\_{M})\_{j},(A\_{M})\_{j,j}) |  | (33) |

By definition of the Bayes predictor, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | fX~⋆​(X~)=β0⋆+⟨βo​b​s​(M)⋆,Xo​b​s​(M)⟩+⟨βm​i​s​(M)⋆,𝔼​[Xm​i​s​(M)|M,Xo​b​s​(M)]⟩,subscriptsuperscript𝑓⋆~𝑋~𝑋superscriptsubscript𝛽0⋆  superscriptsubscript𝛽𝑜𝑏𝑠𝑀⋆subscript𝑋𝑜𝑏𝑠𝑀  subscriptsuperscript𝛽⋆𝑚𝑖𝑠𝑀𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠𝑀  𝑀subscript𝑋𝑜𝑏𝑠𝑀\displaystyle f^{\star}\_{\widetilde{X}}(\widetilde{X})=\beta\_{0}^{\star}+\langle\beta\_{obs(M)}^{\star},X\_{obs(M)}\rangle+\langle\beta^{\star}\_{mis(M)},\mathbb{E}[X\_{mis(M)}|M,X\_{obs(M)}]\rangle, |  | (34) |

where

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[Xm​i​s|M,Xo​b​s]=(aM)m​i​s.𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  𝑀subscript𝑋𝑜𝑏𝑠subscriptsubscript𝑎𝑀𝑚𝑖𝑠\mathbb{E}[X\_{mis}|M,X\_{obs}]=(a\_{M})\_{mis}. |  | (35) |

Combining equations ([30](#A1.E30 "In A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), ([31](#A1.E31 "In A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), ([35](#A1.E35 "In A.4 Proof of Proposition 2.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")), we obtain

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[Xm​i​s|M,Xo​b​s]=𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  𝑀subscript𝑋𝑜𝑏𝑠absent\displaystyle\mathbb{E}[X\_{mis}|M,X\_{obs}]= | (I​d+Dm​i​s​Σm​i​s|o​b​s−1)−1superscript𝐼𝑑subscript𝐷𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠11\displaystyle\left(Id+D\_{mis}\Sigma\_{mis|obs}^{-1}\right)^{-1} |  | (36) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ×[μ~m​i​s+Dm​i​s​Σm​i​s|o​b​s−1​(μm​i​s+Σm​i​s,o​b​s​(Σo​b​s)−1​(Xo​b​s−μo​b​s))]absentdelimited-[]subscript~𝜇𝑚𝑖𝑠subscript𝐷𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1subscript𝜇𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠\displaystyle\times\left[\tilde{\mu}\_{mis}+D\_{mis}\Sigma\_{mis|obs}^{-1}\left(\mu\_{mis}+\Sigma\_{mis,obs}\left(\Sigma\_{obs}\right)^{-1}\left(X\_{obs}-\mu\_{obs}\right)\right)\right] |  | (37) |

### A.5 Controlling the convergence of Neumann iterates

Here we establish an auxiliary result, controlling the convergence of
Neumann iterates to the matrix inverse.

###### Proposition A.1 (Linear convergence of Neumann iterations).

Assume that the spectral radius of ΣΣ\Sigma is strictly less than 111. Therefore, for all missing data patterns m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, the iterates So​b​s​(m)(ℓ)subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝑚S^{(\ell)}\_{obs(m)} defined in equation ([6](#S3.E6 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) converge linearly towards (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1(\Sigma\_{obs(m)})^{-1} and satisfy, for all ℓ≥1ℓ1\ell\geq 1,

|  |  |  |
| --- | --- | --- |
|  | ‖I​d−Σo​b​s​(m)​So​b​s​(m)(ℓ)‖2≤(1−νo​b​s​(m))ℓ​‖I​d−Σo​b​s​(m)​So​b​s​(m)(0)‖2,subscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ2superscript1subscript𝜈𝑜𝑏𝑠𝑚ℓsubscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚subscriptsuperscript𝑆0𝑜𝑏𝑠𝑚2\|Id-\Sigma\_{obs(m)}S\_{obs(m)}^{(\ell)}\|\_{2}\leq(1-\nu\_{obs(m)})^{\ell}\|Id-\Sigma\_{obs(m)}S^{(0)}\_{obs(m)}\|\_{2}\enspace, |  |

where νo​b​s​(m)subscript𝜈𝑜𝑏𝑠𝑚\nu\_{obs(m)} is the smallest eigenvalue of Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)}.

Note that Proposition [A.1](#A1.Thmproposition1 "Proposition A.1 (Linear convergence of Neumann iterations). ‣ A.5 Controlling the convergence of Neumann iterates ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") can easily be
extended to the general case by working with Σ/ρ​(Σ)Σ𝜌Σ\Sigma/\rho(\Sigma) and
multiplying the resulting approximation by ρ​(Σ)𝜌Σ\rho(\Sigma), where
ρ​(Σ)𝜌Σ\rho(\Sigma) is the spectral radius of ΣΣ\Sigma.

###### Proof.

Since the spectral radius of ΣΣ\Sigma is strictly smaller than one, the spectral radius of each submatrix Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)} is also strictly smaller than one. This is a direct application of Cauchy Interlace Theorem [[8](#bib.bib8)] or it can be seen with the definition of the eigenvalues

|  |  |  |
| --- | --- | --- |
|  | ρ​(Σo​b​s​(m))=maxu∈ℝ|o​b​s​(m)|⁡u⊤​Σo​b​s​(m)​u=maxx∈ℝdxm​i​s=0⁡x⊤​Σ​x≤maxx∈ℝd⁡x⊤​Σ​x=ρ​(Σ).𝜌subscriptΣ𝑜𝑏𝑠𝑚subscript𝑢superscriptℝ𝑜𝑏𝑠𝑚superscript𝑢topsubscriptΣ𝑜𝑏𝑠𝑚𝑢subscript  𝑥superscriptℝ𝑑subscript𝑥𝑚𝑖𝑠0superscript𝑥topΣ𝑥subscript𝑥superscriptℝ𝑑superscript𝑥topΣ𝑥𝜌Σ\rho(\Sigma\_{obs(m)})=\max\_{u\in\mathbb{R}^{|obs(m)|}}u^{\top}\Sigma\_{obs(m)}u=\max\_{\begin{subarray}{c}x\in\mathbb{R}^{d}\\ x\_{mis}=0\end{subarray}}x^{\top}\Sigma x\leq\max\_{x\in\mathbb{R}^{d}}x^{\top}\Sigma x=\rho(\Sigma)\enspace. |  |

Note that So​b​s​(m)ℓ=∑k=0ℓ−1(I​d−Σo​b​s)k+(I​d−Σo​b​s)ℓ​So​b​s​(m)0superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓsuperscriptsubscript𝑘0ℓ1superscript𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑘superscript𝐼𝑑subscriptΣ𝑜𝑏𝑠ℓsubscriptsuperscript𝑆0𝑜𝑏𝑠𝑚S\_{obs(m)}^{\ell}=\sum\_{k=0}^{\ell-1}\left(Id-\Sigma\_{obs}\right)^{k}+\left(Id-\Sigma\_{obs}\right)^{\ell}S^{0}\_{obs(m)}
can be defined recursively via the iterative formula

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | So​b​s​(m)ℓsuperscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ\displaystyle S\_{obs(m)}^{\ell} | =(I​d−Σo​b​s​(m))​So​b​s​(m)ℓ−1+I​dabsent𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ1𝐼𝑑\displaystyle=(Id-\Sigma\_{obs(m)})S\_{obs(m)}^{\ell-1}+Id |  | (38) |

The matrix (Σo​b​s​(m))−1superscriptsubscriptΣ𝑜𝑏𝑠𝑚1(\Sigma\_{obs(m)})^{-1} is a fixed point of the Neumann iterations (equation ([38](#A1.E38 "In Proof. ‣ A.5 Controlling the convergence of Neumann iterates ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"))). It verifies the following equation

|  |  |  |  |
| --- | --- | --- | --- |
|  | (Σo​b​s​(m))−1=(I​d−Σo​b​s​(m))​(Σo​b​s​(m))−1+I​d.superscriptsubscriptΣ𝑜𝑏𝑠𝑚1𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscriptΣ𝑜𝑏𝑠𝑚1𝐼𝑑(\Sigma\_{obs(m)})^{-1}=(Id-\Sigma\_{obs(m)})(\Sigma\_{obs(m)})^{-1}+Id\enspace. |  | (39) |

By substracting [38](#A1.E38 "In Proof. ‣ A.5 Controlling the convergence of Neumann iterates ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") to this equation, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | (Σo​b​s​(m))−1−So​b​s​(m)ℓ=(I​d−Σo​b​s​(m))​((Σo​b​s​(m))−1−So​b​s​(m)ℓ−1).superscriptsubscriptΣ𝑜𝑏𝑠𝑚1superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscriptΣ𝑜𝑏𝑠𝑚1superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ1(\Sigma\_{obs(m)})^{-1}-S\_{obs(m)}^{\ell}=(Id-\Sigma\_{obs(m)})((\Sigma\_{obs(m)})^{-1}-S\_{obs(m)}^{\ell-1})\enspace. |  | (40) |

Multiplying both sides by Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)} yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | (I​d−Σo​b​s​(m)​So​b​s​(m)ℓ)=(I​d−Σo​b​s​(m))​(I​d−Σo​b​s​(m)​So​b​s​(m)ℓ−1).𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ1(Id-\Sigma\_{obs(m)}S\_{obs(m)}^{\ell})=(Id-\Sigma\_{obs(m)})(Id-\Sigma\_{obs(m)}S\_{obs(m)}^{\ell-1})\enspace. |  | (41) |

Taking the ℓ2subscriptℓ2\ell\_{2}-norm and using Cauchy-Schwartz inequality yields

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖I​d−Σo​b​s​(m)​So​b​s​(m)ℓ‖2≤‖I​d−Σo​b​s​(m)‖2​‖I​d−Σo​b​s​(m)​So​b​s​(m)ℓ−1‖2.subscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ2subscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚2subscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ12\|Id-\Sigma\_{obs(m)}S\_{obs(m)}^{\ell}\|\_{2}\leq\|Id-\Sigma\_{obs(m)}\|\_{2}\|Id-\Sigma\_{obs(m)}S\_{obs(m)}^{\ell-1}\|\_{2}\enspace. |  | (42) |

Let νo​b​s​(m)subscript𝜈𝑜𝑏𝑠𝑚\nu\_{obs(m)} be the smallest eigenvalue of Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)}, which is positive since ΣΣ\Sigma is invertible. Since the largest eigenvalue of Σo​b​s​(m)subscriptΣ𝑜𝑏𝑠𝑚\Sigma\_{obs(m)} is upper bounded by 111, we get that ‖I​d−Σ~‖2=(1−νo​b​s​(m))subscriptnorm𝐼𝑑~Σ21subscript𝜈𝑜𝑏𝑠𝑚\|Id-\widetilde{\Sigma}\|\_{2}=(1-\nu\_{obs(m)}) and by recursion we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖I​d−Σo​b​s​(m)​So​b​s​(m)ℓ‖2≤(1−νo​b​s​(m))ℓ​‖I​d−Σo​b​s​(m)​So​b​s​(m)0‖2.subscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚superscriptsubscript𝑆𝑜𝑏𝑠𝑚ℓ2superscript1subscript𝜈𝑜𝑏𝑠𝑚ℓsubscriptnorm𝐼𝑑subscriptΣ𝑜𝑏𝑠𝑚subscriptsuperscript𝑆0𝑜𝑏𝑠𝑚2\|Id-\Sigma\_{obs(m)}S\_{obs(m)}^{\ell}\|\_{2}\leq(1-\nu\_{obs(m)})^{\ell}\|Id-\Sigma\_{obs(m)}S^{0}\_{obs(m)}\|\_{2}\enspace. |  | (43) |

∎

### A.6 Proof of Proposition [3.1](#S3.Thmproposition1 "Proposition 3.1. ‣ 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

See [3.1](#S3.Thmproposition1 "Proposition 3.1. ‣ 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

According to Proposition [2.1](#S2.Thmproposition1 "Proposition 2.1 (MAR Bayes predictor). ‣ 2.2 Expression of the Bayes predictor under various missing-values mechanisms ‣ 2 Optimal predictors in the presence of missing values ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and the definition of the approximation of order p𝑝p of the Bayes predictor (see equations ([7](#S3.E7 "In 3.2 Differentiable approximations of the inverse covariances with Neumann series ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")))

|  |  |  |
| --- | --- | --- |
|  | fX~,ℓ⋆​(X~)=⟨βo​b​s⋆,Xo​b​s⟩+⟨βm​i​s⋆,μm​i​s+Σm​i​s,o​b​s​So​b​s(ℓ)​(Xo​b​s−μo​b​s)⟩,subscriptsuperscript𝑓⋆  ~𝑋ℓ~𝑋  subscriptsuperscript𝛽⋆𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠  subscriptsuperscript𝛽⋆𝑚𝑖𝑠subscript𝜇𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠f^{\star}\_{\widetilde{X},\ell}(\widetilde{X})=\langle\beta^{\star}\_{obs},X\_{obs}\rangle+\langle\beta^{\star}\_{mis},\mu\_{mis}+\Sigma\_{mis,obs}S^{(\ell)}\_{obs}\left(X\_{obs}-\mu\_{obs}\right)\rangle\enspace, |  |

Then

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[(fX~,ℓ⋆​(X~)−fX~⋆​(X~))2]𝔼delimited-[]superscriptsubscriptsuperscript𝑓⋆  ~𝑋ℓ~𝑋subscriptsuperscript𝑓⋆~𝑋~𝑋2\displaystyle\mathbb{E}[(f^{\star}\_{\widetilde{X},\ell}(\widetilde{X})-f^{\star}\_{\widetilde{X}}(\widetilde{X}))^{2}] |  | (44) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[⟨βm​i​s⋆,Σm​i​s,o​b​s​(So​b​sℓ−Σo​b​s−1)​(Xo​b​s−μo​b​s)⟩2]absent𝔼delimited-[]superscript  superscriptsubscript𝛽𝑚𝑖𝑠⋆subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠 2\displaystyle=\mathbb{E}\Big{[}\big{\langle}\beta\_{mis}^{\star}~{},~{}\Sigma\_{mis,obs}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})(X\_{obs}-\mu\_{obs})\big{\rangle}^{2}\Big{]} |  | (45) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[(βm​i​s⋆)⊤​Σm​i​s,o​b​s​(So​b​sℓ−Σo​b​s−1)​(Xo​b​s−μo​b​s)​(Xo​b​s−μo​b​s)⊤​(So​b​sℓ−Σo​b​s−1)​Σo​b​s,m​i​s​βm​i​s⋆]absent𝔼delimited-[]superscriptsuperscriptsubscript𝛽𝑚𝑖𝑠⋆topsubscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠superscriptsubscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠topsubscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠superscriptsubscript𝛽𝑚𝑖𝑠⋆\displaystyle=\mathbb{E}\Big{[}(\beta\_{mis}^{\star})^{\top}\Sigma\_{mis,obs}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})(X\_{obs}-\mu\_{obs})(X\_{obs}-\mu\_{obs})^{\top}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})\Sigma\_{obs,mis}\beta\_{mis}^{\star}\Big{]} |  | (46) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[(βm​i​s⋆)⊤​Σm​i​s,o​b​s​(So​b​sℓ−Σo​b​s−1)​𝔼​[(Xo​b​s−μo​b​s)​(Xo​b​s−μo​b​s)⊤|M]⏟Σo​b​s​(So​b​sℓ−Σo​b​s−1)​Σo​b​s,m​i​s​βm​i​s⋆]absent𝔼delimited-[]superscriptsuperscriptsubscript𝛽𝑚𝑖𝑠⋆topsubscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript⏟𝔼delimited-[]conditionalsubscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠superscriptsubscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠top𝑀subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠superscriptsubscript𝛽𝑚𝑖𝑠⋆\displaystyle=\mathbb{E}\Big{[}(\beta\_{mis}^{\star})^{\top}\Sigma\_{mis,obs}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})\underbrace{\mathbb{E}[(X\_{obs}-\mu\_{obs})(X\_{obs}-\mu\_{obs})^{\top}|M]}\_{\Sigma\_{obs}}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})\Sigma\_{obs,mis}\beta\_{mis}^{\star}\Big{]} |  | (47) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[(βm​i​s⋆)⊤​Σm​i​s,o​b​s​(So​b​sℓ−Σo​b​s−1)​Σo​b​s​(So​b​sℓ−Σo​b​s−1)​Σo​b​s,m​i​s​βm​i​s⋆]absent𝔼delimited-[]superscriptsuperscriptsubscript𝛽𝑚𝑖𝑠⋆topsubscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠superscriptsubscript𝛽𝑚𝑖𝑠⋆\displaystyle=\mathbb{E}\Big{[}(\beta\_{mis}^{\star})^{\top}\Sigma\_{mis,obs}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})\Sigma\_{obs}(S^{\ell}\_{obs}-\Sigma\_{obs}^{-1})\Sigma\_{obs,mis}\beta\_{mis}^{\star}\Big{]} |  | (48) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[‖(Σo​b​s)12​(Σo​b​s)−1​(Σo​b​s​So​b​sℓ−I​do​b​s)​Σo​b​s,m​i​s​βm​i​s⋆‖22]absent𝔼delimited-[]superscriptsubscriptnormsuperscriptsubscriptΣ𝑜𝑏𝑠12superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠𝐼subscript𝑑𝑜𝑏𝑠subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠superscriptsubscript𝛽𝑚𝑖𝑠⋆22\displaystyle=\mathbb{E}\Big{[}\big{\|}(\Sigma\_{obs})^{\frac{1}{2}}(\Sigma\_{obs})^{-1}(\Sigma\_{obs}S^{\ell}\_{obs}-Id\_{obs})\Sigma\_{obs,mis}\beta\_{mis}^{\star}\big{\|}\_{2}^{2}\Big{]} |  | (49) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | =𝔼​[‖(Σo​b​s)−12​(I​do​b​s−Σo​b​s​So​b​sℓ)​Σo​b​s,m​i​s​βm​i​s⋆‖22]absent𝔼delimited-[]superscriptsubscriptnormsuperscriptsubscriptΣ𝑜𝑏𝑠12𝐼subscript𝑑𝑜𝑏𝑠subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠superscriptsubscript𝛽𝑚𝑖𝑠⋆22\displaystyle=\mathbb{E}\Big{[}\big{\|}(\Sigma\_{obs})^{-\frac{1}{2}}(Id\_{obs}-\Sigma\_{obs}S^{\ell}\_{obs})\Sigma\_{obs,mis}\beta\_{mis}^{\star}\big{\|}\_{2}^{2}\Big{]} |  | (50) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ≤‖Σ−1‖2​‖Σ‖22​‖β⋆‖22​𝔼​[‖I​do​b​s−Σo​b​s​So​b​sℓ‖22]absentsubscriptnormsuperscriptΣ12superscriptsubscriptnormΣ22superscriptsubscriptnormsuperscript𝛽⋆22𝔼delimited-[]superscriptsubscriptnorm𝐼subscript𝑑𝑜𝑏𝑠subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆ℓ𝑜𝑏𝑠22\displaystyle\leq\|\Sigma^{-1}\|\_{2}\|\Sigma\|\_{2}^{2}\|\beta^{\star}\|\_{2}^{2}\mathbb{E}\big{[}\|Id\_{obs}-\Sigma\_{obs}S^{\ell}\_{obs}\|\_{2}^{2}\big{]} |  | (51) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ≤1ν​‖β⋆‖22​𝔼​[(1−νo​b​s)2​ℓ​‖I​do​b​s−Σo​b​s​So​b​s0‖22]absent1𝜈superscriptsubscriptnormsuperscript𝛽⋆22𝔼delimited-[]superscript1subscript𝜈𝑜𝑏𝑠2ℓsuperscriptsubscriptnorm𝐼subscript𝑑𝑜𝑏𝑠subscriptΣ𝑜𝑏𝑠subscriptsuperscript𝑆0𝑜𝑏𝑠22\displaystyle\leq\frac{1}{\nu}\|\beta^{\star}\|\_{2}^{2}\mathbb{E}\big{[}(1-\nu\_{obs})^{2\ell}\|Id\_{obs}-\Sigma\_{obs}S^{0}\_{obs}\|\_{2}^{2}\big{]} |  | (52) |

An important point for going from ([50](#A1.E50 "In A.6 Proof of Proposition 3.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) to ([51](#A1.E51 "In A.6 Proof of Proposition 3.1 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) is to notice that for any missing pattern, we have

|  |  |  |
| --- | --- | --- |
|  | ‖Σo​b​s,m​i​s‖2≤‖Σ‖2​ and ​‖Σo​b​s−1‖2≤‖Σ−1‖2.subscriptnormsubscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠2subscriptnormΣ2 and subscriptnormsuperscriptsubscriptΣ𝑜𝑏𝑠12subscriptnormsuperscriptΣ12\displaystyle\|\Sigma\_{obs,mis}\|\_{2}\leq\|\Sigma\|\_{2}\text{ and }\|\Sigma\_{obs}^{-1}\|\_{2}\leq\|\Sigma^{-1}\|\_{2}\enspace. |  |

The first inequality can be obtained by observing that computing the largest singular value of Σo​b​s,m​i​ssubscriptΣ

𝑜𝑏𝑠𝑚𝑖𝑠\Sigma\_{obs,mis} reduces to solving a constrained version of the maximization problem that defines the largest eigenvalue of ΣΣ\Sigma:

|  |  |  |
| --- | --- | --- |
|  | ‖Σo​b​s,m​i​s‖2=max‖xm​i​s‖2=1⁡‖Σo​b​s,m​i​s​xm​i​s‖2≤max‖x‖2=1xo​b​s=0⁡‖Σo​b​s,⋅​x‖2≤max‖x‖2=1xo​b​s=0⁡‖Σ​x‖2≤max‖x‖2=1⁡‖Σ​x‖22=‖Σ‖2.subscriptnormsubscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠2subscriptsubscriptnormsubscript𝑥𝑚𝑖𝑠21subscriptnormsubscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠subscript𝑥𝑚𝑖𝑠2subscript  subscriptnorm𝑥21subscript𝑥𝑜𝑏𝑠0subscriptnormsubscriptΣ  𝑜𝑏𝑠⋅𝑥2subscript  subscriptnorm𝑥21subscript𝑥𝑜𝑏𝑠0subscriptnormΣ𝑥2subscriptsubscriptnorm𝑥21superscriptsubscriptnormΣ𝑥22subscriptnormΣ2\displaystyle\|\Sigma\_{obs,mis}\|\_{2}=\max\_{\|x\_{mis}\|\_{2}=1}\|\Sigma\_{obs,mis}x\_{mis}\|\_{2}\leq\max\_{\begin{subarray}{c}\|x\|\_{2}=1\\ x\_{obs}=0\end{subarray}}\|\Sigma\_{obs,\cdot}x\|\_{2}\leq\max\_{\begin{subarray}{c}\|x\|\_{2}=1\\ x\_{obs}=0\end{subarray}}\|\Sigma x\|\_{2}\leq\max\_{\|x\|\_{2}=1}\|\Sigma x\|\_{2}^{2}=\|\Sigma\|\_{2}\enspace. |  |

where we used ‖Σo​b​s,⋅​x‖22=∑i∈o​b​s(Σi⊤​x)2≤∑i=1d(Σi⊤​x)2=‖Σ​x‖22superscriptsubscriptnormsubscriptΣ

𝑜𝑏𝑠⋅𝑥22subscript𝑖𝑜𝑏𝑠superscriptsuperscriptsubscriptΣ𝑖top𝑥2superscriptsubscript𝑖1𝑑superscriptsuperscriptsubscriptΣ𝑖top𝑥2superscriptsubscriptnormΣ𝑥22\|\Sigma\_{obs,\cdot}x\|\_{2}^{2}=\sum\_{i\in obs}(\Sigma\_{i}^{\top}x)^{2}\leq\sum\_{i=1}^{d}(\Sigma\_{i}^{\top}x)^{2}=\|\Sigma x\|\_{2}^{2}.
  
A similar observation can be done for computing the smallest eigenvalue
of ΣΣ\Sigma, λmin​(Σ)subscript𝜆Σ\lambda\_{\min}(\Sigma):

|  |  |  |
| --- | --- | --- |
|  | λmin​(Σ)=min‖x‖2=1⁡x⊤​Σ​x≤min‖x‖2=1xm​i​s=0⁡x⊤​Σ​x=min‖xo​b​s‖2=1⁡xo​b​s⊤​Σo​b​s​xo​b​s=λmin​(Σo​b​s).subscript𝜆Σsubscriptsubscriptnorm𝑥21superscript𝑥topΣ𝑥subscript  subscriptnorm𝑥21subscript𝑥𝑚𝑖𝑠0superscript𝑥topΣ𝑥subscriptsubscriptnormsubscript𝑥𝑜𝑏𝑠21subscriptsuperscript𝑥top𝑜𝑏𝑠subscriptΣ𝑜𝑏𝑠subscript𝑥𝑜𝑏𝑠subscript𝜆subscriptΣ𝑜𝑏𝑠\lambda\_{\min}(\Sigma)=\min\_{\|x\|\_{2}=1}x^{\top}\Sigma x\leq\min\_{\begin{subarray}{c}\|x\|\_{2}=1\\ x\_{mis}=0\end{subarray}}x^{\top}\Sigma x=\min\_{\|x\_{obs}\|\_{2}=1}x^{\top}\_{obs}\Sigma\_{obs}x\_{obs}=\lambda\_{\min}(\Sigma\_{obs})\enspace. |  |

and we can deduce the second inequality by noting that λmin​(Σ)=1‖Σ−1‖22subscript𝜆Σ1superscriptsubscriptnormsuperscriptΣ122\lambda\_{\min}(\Sigma)=\frac{1}{\|\Sigma^{-1}\|\_{2}^{2}} and λmin​(Σo​b​s)=1‖Σo​b​s−1‖22subscript𝜆subscriptΣ𝑜𝑏𝑠1superscriptsubscriptnormsuperscriptsubscriptΣ𝑜𝑏𝑠122\lambda\_{\min}(\Sigma\_{obs})=\frac{1}{\|\Sigma\_{obs}^{-1}\|\_{2}^{2}}.

### A.7 Proof of Proposition [3.2](#S3.Thmproposition2 "Proposition 3.2 (equivalence MLP - depth-1 NeuMiss network). ‣ 3.4 Link with the multilayer perceptron with ReLU activations ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

See [3.2](#S3.Thmproposition2 "Proposition 3.2 (equivalence MLP - depth-1 NeuMiss network). ‣ 3.4 Link with the multilayer perceptron with ReLU activations ‣ 3 NeuMiss networks: learning by approximating the Bayes predictors ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")

#### Obtaining a ⊙Mdirect-productabsent𝑀\odot M nonlinearity from a ReLU nonlinearity.

Let ℋR​e​L​U=([W(X),W(M)]∈ℝd×2​d,R​e​L​U)subscriptℋ𝑅𝑒𝐿𝑈superscript𝑊𝑋superscript𝑊𝑀

superscriptℝ𝑑2𝑑𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU}=\left(\left[W^{(X)},W^{(M)}\right]\in\mathbb{R}^{d\times 2d},ReLU\right) be a hidden layer which connects [X,M]𝑋𝑀\left[X,M\right] to d𝑑d hidden units, and applies a ReLU nonlinearity to the activations. We denote by b∈ℝd𝑏superscriptℝ𝑑b\in\mathbb{R}^{d} the bias corresponding to this layer. Let k∈⟦1,d⟧𝑘

1𝑑k\in\left\llbracket 1,d\right\rrbracket. Depending on the missing data pattern that is given as input, the kt​hsuperscript𝑘𝑡ℎk^{th} entry can correspond to either a missing or an observed entry. We now write the activation of the kt​hsuperscript𝑘𝑡ℎk^{th} hidden unit depending on whether entry k𝑘k is observed or missing. The activation of the kt​hsuperscript𝑘𝑡ℎk^{th} hidden unit is given by

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | aksubscript𝑎𝑘\displaystyle a\_{k} | =Wk,.(X)​X+Wk,.(M)​M+bk\displaystyle=W\_{k,.}^{(X)}X+W\_{k,.}^{(M)}M+b\_{k} |  | (53) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =Wk,o​b​s(X)​Xo​b​s+Wk,m​i​s(M)​𝟏m​i​s+bk.absentsuperscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑋subscript𝑋𝑜𝑏𝑠superscriptsubscript𝑊  𝑘𝑚𝑖𝑠𝑀subscript1𝑚𝑖𝑠subscript𝑏𝑘\displaystyle=W\_{k,obs}^{(X)}X\_{obs}+W\_{k,mis}^{(M)}\mathbf{1}\_{mis}+b\_{k}. |  | (54) |

Emphasizing the role of Wk,k(M)superscriptsubscript𝑊

𝑘𝑘𝑀W\_{k,k}^{(M)} and Wk,k(X)superscriptsubscript𝑊

𝑘𝑘𝑋W\_{k,k}^{(X)}, we can decompose equation ([54](#A1.E54 "In Obtaining a ⊙𝑀 nonlinearity from a ReLU nonlinearity. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values")) depending on whether the kt​hsuperscript𝑘𝑡ℎk^{th} entry is observed or missing

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈m​i​s,akIf 𝑘  𝑚𝑖𝑠subscript𝑎𝑘\displaystyle\text{If }k\in mis,\quad a\_{k} | =Wk,o​b​s(X)​Xo​b​s+Wk,k(M)+Wk,m​i​s∖{k}(M)​𝟏k,m​i​s∖{k}+bkabsentsuperscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑋subscript𝑋𝑜𝑏𝑠superscriptsubscript𝑊  𝑘𝑘𝑀subscriptsuperscript𝑊𝑀  𝑘𝑚𝑖𝑠𝑘subscript1  𝑘𝑚𝑖𝑠𝑘subscript𝑏𝑘\displaystyle=W\_{k,obs}^{(X)}X\_{obs}+W\_{k,k}^{(M)}+W^{(M)}\_{k,mis\setminus\{k\}}\mathbf{1}\_{k,mis\setminus\{k\}}+b\_{k} |  | (55) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈o​b​s,akIf 𝑘  𝑜𝑏𝑠subscript𝑎𝑘\displaystyle\text{If }k\in obs,\quad a\_{k} | =Wk,k(X)​Xk+Wk,o​b​s∖{k}(X)​Xo​b​s∖{k}+Wk,m​i​s(M)​𝟏m​i​s+bk.absentsuperscriptsubscript𝑊  𝑘𝑘𝑋subscript𝑋𝑘superscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑘𝑋subscript𝑋𝑜𝑏𝑠𝑘subscriptsuperscript𝑊𝑀  𝑘𝑚𝑖𝑠subscript1𝑚𝑖𝑠subscript𝑏𝑘\displaystyle=W\_{k,k}^{(X)}X\_{k}+W\_{k,obs\setminus\{k\}}^{(X)}X\_{obs\setminus\{k\}}+W^{(M)}\_{k,mis}\mathbf{1}\_{mis}+b\_{k}. |  | (56) |

Suppose that the weights W(X)superscript𝑊𝑋W^{(X)} as well as Wi,j(M),i≠j

subscriptsuperscript𝑊𝑀

𝑖𝑗𝑖
𝑗W^{(M)}\_{i,j},i\neq j are fixed. Then, under the assumption that the support of X𝑋X is finite, there exists a bias bk∗subscriptsuperscript𝑏𝑘b^{\*}\_{k} which verifies:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∀X,ak=Wk,k(X)​Xk+Wk,o​b​s∖{k}(X)​Xo​b​s∖{k}+Wk,m​i​s(M)​𝟏m​i​s+bk∗≤0  for-all𝑋subscript𝑎𝑘 superscriptsubscript𝑊  𝑘𝑘𝑋subscript𝑋𝑘superscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑘𝑋subscript𝑋𝑜𝑏𝑠𝑘subscriptsuperscript𝑊𝑀  𝑘𝑚𝑖𝑠subscript1𝑚𝑖𝑠subscriptsuperscript𝑏𝑘0\forall X,\quad a\_{k}=W\_{k,k}^{(X)}X\_{k}+W\_{k,obs\setminus\{k\}}^{(X)}X\_{obs\setminus\{k\}}+W^{(M)}\_{k,mis}\mathbf{1}\_{mis}+b^{\*}\_{k}\leq 0 |  | (57) |

i.e., there exists a bias bk∗subscriptsuperscript𝑏𝑘b^{\*}\_{k} such that the activation of the kt​hsuperscript𝑘𝑡ℎk^{th} hidden unit is always negative when k𝑘k is observed. Similarly, there exists Wk,k∗,(M)superscriptsubscript𝑊

𝑘𝑘

𝑀W\_{k,k}^{\*,(M)} such that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∀X,ak=Wk,o​b​s(X)​Xo​b​s+Wk,k∗,(M)+Wk,m​i​s∖{k}(M)​𝟏k,m​i​s∖{k}+bk∗≥0  for-all𝑋subscript𝑎𝑘 superscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑋subscript𝑋𝑜𝑏𝑠superscriptsubscript𝑊  𝑘𝑘  𝑀subscriptsuperscript𝑊𝑀  𝑘𝑚𝑖𝑠𝑘subscript1  𝑘𝑚𝑖𝑠𝑘subscriptsuperscript𝑏𝑘0\forall X,\quad a\_{k}=W\_{k,obs}^{(X)}X\_{obs}+W\_{k,k}^{\*,(M)}+W^{(M)}\_{k,mis\setminus\{k\}}\mathbf{1}\_{k,mis\setminus\{k\}}+b^{\*}\_{k}\geq 0 |  | (58) |

i.e., there exists a weight Wk,k∗,(M)superscriptsubscript𝑊

𝑘𝑘

𝑀W\_{k,k}^{\*,(M)} such that the activation of the kt​hsuperscript𝑘𝑡ℎk^{th} hidden unit is always positive when k𝑘k is missing. Note that these results hold because the weight Wk,k(M)superscriptsubscript𝑊

𝑘𝑘𝑀W\_{k,k}^{(M)} only appears in the expression of aksubscript𝑎𝑘a\_{k} when entry k𝑘k is missing. Let hk=R​e​L​U​(ak)subscriptℎ𝑘𝑅𝑒𝐿𝑈subscript𝑎𝑘h\_{k}=ReLU(a\_{k}). By choosing bk=bk∗subscript𝑏𝑘subscriptsuperscript𝑏𝑘b\_{k}=b^{\*}\_{k} and Wk,k(M)=Wk,k∗,(M)superscriptsubscript𝑊

𝑘𝑘𝑀superscriptsubscript𝑊

𝑘𝑘

𝑀W\_{k,k}^{(M)}=W\_{k,k}^{\*,(M)}, we have that:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈m​i​s,hkIf 𝑘  𝑚𝑖𝑠subscriptℎ𝑘\displaystyle\text{If }k\in mis,\quad h\_{k} | =akabsentsubscript𝑎𝑘\displaystyle=a\_{k} |  | (59) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈o​b​s,hkIf 𝑘  𝑜𝑏𝑠subscriptℎ𝑘\displaystyle\text{If }k\in obs,\quad h\_{k} | =0absent0\displaystyle=0 |  | (60) |

As a result, the output of the hidden layer ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} can be rewritten as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | hk=ak⊙Msubscriptℎ𝑘direct-productsubscript𝑎𝑘𝑀h\_{k}=a\_{k}\odot M |  | (61) |

i.e., a ⊙Mdirect-productabsent𝑀\odot M nonlinearity is applied to the activations.

#### Equating the slopes and biases of ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} and ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M}.

Let ℋ⊙M=(W∈ℝd×d,μ,⊙M)\mathcal{H}\_{\odot M}=\left(W\in\mathbb{R}^{d\times d},\mu,\odot M\right) be the layer that connect (X−μ)⊙(1−M)direct-product𝑋𝜇1𝑀(X-\mu)\odot(1-M) to d𝑑d hidden units via the weight matrix W𝑊W, and applies a ⊙Mdirect-productabsent𝑀\odot M nonlinearity to the activations. We will denote by c∈ℝd𝑐superscriptℝ𝑑c\in\mathbb{R}^{d} the bias corresponding to this layer.

The activations for this layer are given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | aksubscript𝑎𝑘\displaystyle a\_{k} | =Wk,o​b​s​(Xo​b​s−μo​b​s)+ckabsentsubscript𝑊  𝑘𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠subscript𝑐𝑘\displaystyle=W\_{k,obs}(X\_{obs}-\mu\_{obs})+c\_{k} |  | (62) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =Wk,o​b​s​Xo​b​s−Wk,o​b​s​μo​b​s+ckabsentsubscript𝑊  𝑘𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠subscript𝑊  𝑘𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠subscript𝑐𝑘\displaystyle=W\_{k,obs}X\_{obs}-W\_{k,obs}\mu\_{obs}+c\_{k} |  | (63) |

Then by applying the non-linearity we obtain the output of the hidden layer:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈m​i​s,hkIf 𝑘  𝑚𝑖𝑠subscriptℎ𝑘\displaystyle\text{If }k\in mis,\quad h\_{k} | =akabsentsubscript𝑎𝑘\displaystyle=a\_{k} |  | (64) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | If ​k∈o​b​s,hkIf 𝑘  𝑜𝑏𝑠subscriptℎ𝑘\displaystyle\text{If }k\in obs,\quad h\_{k} | =0absent0\displaystyle=0 |  | (65) |

It is straigthforward to see that with the choice of bk=bk∗subscript𝑏𝑘subscriptsuperscript𝑏𝑘b\_{k}=b^{\*}\_{k} and Wk,k(M)=Wk,k∗,(M)superscriptsubscript𝑊

𝑘𝑘𝑀superscriptsubscript𝑊

𝑘𝑘

𝑀W\_{k,k}^{(M)}=W\_{k,k}^{\*,(M)} for ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU}, both hidden layers have the same output hk=0subscriptℎ𝑘0h\_{k}=0 when entry k𝑘k is observed. It remains to be shown that there exists a configuration of the weights of ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} such that the activations aksubscript𝑎𝑘a\_{k} when entry k𝑘k is missing are equal to those of ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M}. To avoid confusions, we will now denote by ak(N)subscriptsuperscript𝑎𝑁𝑘a^{(N)}\_{k} the activations of ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M} and by ak(R)subscriptsuperscript𝑎𝑅𝑘a^{(R)}\_{k} the activations of ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU}. We recall here the activations for both layers as derived in [63](#A1.E63 "In Equating the slopes and biases of ℋ_{𝑅⁢𝑒⁢𝐿⁢𝑈} and ℋ_{⊙𝑀}. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and [55](#A1.E55 "In Obtaining a ⊙𝑀 nonlinearity from a ReLU nonlinearity. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values").

|  |  |  |  |
| --- | --- | --- | --- |
|  | If ​k∈m​i​s,{ak(N)=Wk,o​b​s​Xo​b​s−Wk,o​b​s​μo​b​s+ckak(R)=Wk,o​b​s(X)​Xo​b​s+Wk,k∗,(M)+Wk,m​i​s∖{k}(M)​𝟏k,m​i​s∖{k}+bk∗If 𝑘  𝑚𝑖𝑠casessuperscriptsubscript𝑎𝑘𝑁subscript𝑊  𝑘𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠subscript𝑊  𝑘𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠subscript𝑐𝑘otherwisesuperscriptsubscript𝑎𝑘𝑅superscriptsubscript𝑊  𝑘𝑜𝑏𝑠𝑋subscript𝑋𝑜𝑏𝑠superscriptsubscript𝑊  𝑘𝑘  𝑀subscriptsuperscript𝑊𝑀  𝑘𝑚𝑖𝑠𝑘subscript1  𝑘𝑚𝑖𝑠𝑘superscriptsubscript𝑏𝑘otherwise\text{If }k\in mis,\begin{cases}a\_{k}^{(N)}=W\_{k,obs}X\_{obs}-W\_{k,obs}\mu\_{obs}+c\_{k}\\ a\_{k}^{(R)}=W\_{k,obs}^{(X)}X\_{obs}+W\_{k,k}^{\*,(M)}+W^{(M)}\_{k,mis\setminus\{k\}}\mathbf{1}\_{k,mis\setminus\{k\}}+b\_{k}^{\*}\end{cases} |  | (66) |

By setting Wk,.(X)=Wk,.W\_{k,.}^{(X)}=W\_{k,.}, we obtain that both activations have the same slopes with regards to X𝑋X. We now turn to the biases. We have that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Wk,k∗,(M)+Wk,m​i​s∖{k}(M)​𝟏k,m​i​s∖{k}+bk∗=Wk,.(M)​𝟏−Wk,o​b​s(M)​𝟏+bk∗\displaystyle W\_{k,k}^{\*,(M)}+W^{(M)}\_{k,mis\setminus\{k\}}\mathbf{1}\_{k,mis\setminus\{k\}}+b\_{k}^{\*}=W\_{k,.}^{(M)}\mathbf{1}-W^{(M)}\_{k,obs}\mathbf{1}+b\_{k}^{\*} |  | (67) |

We now set:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∀j∈o​b​s,for-all𝑗𝑜𝑏𝑠\displaystyle\forall j\in obs,\quad | Wk​j(M)=Wk​j​μjsubscriptsuperscript𝑊𝑀𝑘𝑗subscript𝑊𝑘𝑗subscript𝜇𝑗\displaystyle W^{(M)}\_{kj}=W\_{kj}\mu\_{j} |  | (68) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Wk.(M)​𝟏+bk∗=cksuperscriptsubscript𝑊  𝑘𝑀1superscriptsubscript𝑏𝑘subscript𝑐𝑘\displaystyle W\_{k.}^{(M)}\mathbf{1}+b\_{k}^{\*}=c\_{k} |  | (69) |

to obtain that both activations have the same biases. Note that [68](#A1.E68 "In Equating the slopes and biases of ℋ_{𝑅⁢𝑒⁢𝐿⁢𝑈} and ℋ_{⊙𝑀}. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") sets the weights Wk,jsubscript𝑊

𝑘𝑗W\_{k,j} for all j≠k𝑗𝑘j\neq k (since o​b​s𝑜𝑏𝑠obs can contain any entries except k𝑘k). As a consequence, equation [69](#A1.E69 "In Equating the slopes and biases of ℋ_{𝑅⁢𝑒⁢𝐿⁢𝑈} and ℋ_{⊙𝑀}. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") implies an equation invloving Wk​k∗,(M)superscriptsubscript𝑊𝑘𝑘

𝑀W\_{kk}^{\*,(M)} and bk∗superscriptsubscript𝑏𝑘b\_{k}^{\*} where all other parameters have already been set. Since Wk​k∗,(M)superscriptsubscript𝑊𝑘𝑘

𝑀W\_{kk}^{\*,(M)} and bk∗superscriptsubscript𝑏𝑘b\_{k}^{\*} are also chosen to satisfy the inequalities [57](#A1.E57 "In Obtaining a ⊙𝑀 nonlinearity from a ReLU nonlinearity. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values") and [58](#A1.E58 "In Obtaining a ⊙𝑀 nonlinearity from a ReLU nonlinearity. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), it may not be possible to choose them so as to also satify equation [69](#A1.E69 "In Equating the slopes and biases of ℋ_{𝑅⁢𝑒⁢𝐿⁢𝑈} and ℋ_{⊙𝑀}. ‣ A.7 Proof of Proposition 3.2 ‣ Appendix A Proofs ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"). As a result, the functions computed by the activated hidden units of ℋR​e​L​Usubscriptℋ𝑅𝑒𝐿𝑈\mathcal{H}\_{ReLU} can be equal to those computed by ℋ⊙Msubscriptℋdirect-productabsent𝑀\mathcal{H}\_{\odot M} up to a constant.

## Appendix B Additional results

### B.1 NeuMiss network scaling law in MNAR

![Refer to caption](/html/2007.01627/assets/x7.png)

Gaussian self-masking

![Refer to caption](/html/2007.01627/assets/x8.png)

Probit self-masking

Figure 5: 
Required capacity in various MNAR settings —
Top: Gaussian self-masking, bottom: probit self-masking.
Performance of NeuMiss networks varying the depth in simulations with
different number of samples n𝑛n and of features d𝑑d.

### B.2 NeuMiss network performances in MAR

The MAR data was generated as follows: first, a subset of variables with *no* missing values is randomly selected (10%). The remaining variables have missing values according to a logistic model with random weights, but whose intercept is chosen so as to attain the desired proportion of missing values on those variables (50%). As can be seen from figure [6](#A2.F6 "Figure 6 ‣ B.2 NeuMiss network performances in MAR ‣ Appendix B Additional results ‣ NeuMiss networks: differentiable programming for supervised learning with missing values"), the trends observed for MAR are the same as those for MCAR.

MAR
  
![Refer to caption](/html/2007.01627/assets/x9.png)

Figure 6: Predictive performances in MAR scenario —
varying number of samples n𝑛n, and number of features d𝑑d. All experiments are repeated 20 times.

[◄](/html/2007.01626)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2007.01627)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2007.01627)
[View original  
on arXiv](https://arxiv.org/abs/2007.01627)[►](/html/2007.01628)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Mar 18 17:47:22 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
