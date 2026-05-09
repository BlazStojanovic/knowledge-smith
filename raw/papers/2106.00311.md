---
arxiv: '2106.00311'
authors:
- Marine Le Morvan 1,2 Julie Josse 4 Erwan Scornet 3 Gaël Varoquaux 1 1 Université
  Paris-Saclay, Inria, CEA, Palaiseau, 91120, France 2 Université Paris-Saclay, CNRS/IN2P3,
  IJCLab, 91405 Orsay, France 3 CMAP, UMR7641, Ecole Polytechnique, IP Paris, 91128
  Palaiseau, France 4 Inria Sophia-Antipolis, Montpellier, France {marine.le-morvan,
  julie.josse, gael.varoquaux}@inria.fr erwan.scornet@polytechnique.edu
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: What's a good imputation to predict with missing values?
url: https://arxiv.org/abs/2106.00311
year: 2021
---

[2106.00311] What’s a good imputation to predict with missing values?














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



# What’s a good imputation to predict with missing values?

Marine Le Morvan1,2  
Julie Josse4   
Erwan Scornet3  
Gaël Varoquaux1
  
1 Université Paris-Saclay, Inria, CEA, Palaiseau, 91120, France
  
2 Université Paris-Saclay, CNRS/IN2P3, IJCLab, 91405 Orsay, France
  
3 CMAP, UMR7641, Ecole Polytechnique, IP Paris, 91128 Palaiseau, France
  
4 Inria Sophia-Antipolis, Montpellier, France
  
  
{marine.le-morvan, julie.josse, gael.varoquaux}@inria.fr
  
erwan.scornet@polytechnique.edu

###### Abstract

How to learn a good predictor on data with missing values? Most
efforts focus on first imputing as well as possible
and second learning on the completed data to predict the outcome.
Yet, this widespread practice has no theoretical grounding.
Here we show that for almost all imputation functions, an
impute-then-regress procedure with a powerful learner is Bayes optimal. This result holds for all missing-values mechanisms, in
contrast with the classic statistical results that require
missing-at-random settings to use imputation in probabilistic
modeling.
Moreover, it implies that perfect conditional imputation is not needed for good prediction asymptotically. In fact, we show that on perfectly imputed data the best regression function will generally be discontinuous, which makes it hard to learn.
Crafting instead the
imputation so as to leave the regression function unchanged
simply shifts the problem to learning discontinuous imputations.
Rather, we suggest that it is easier to learn imputation and
regression jointly. We propose such a procedure, adapting NeuMiss, a
neural network capturing the conditional links across observed and
unobserved variables whatever the missing-value pattern.
Experiments confirm
that joint imputation and regression through NeuMiss is better than various
two step procedures in our experiments with finite number of samples.

## 1 Introduction

Data with missing values are ubiquitous in many applications, as in health or business: some observations come with missing features. There is a rich statistical literature on imputation as well as inference with missing values (Rubin, [1976](#bib.bib20); Little and Rubin, [1987, 2002, 2019](#bib.bib16)). Most of the theory and practices build upon the *Missing At Random* (MAR) assumption that allows to maximize the likelihood of observed data while ignoring the missing-values mechanism, for instance using expectation maximization (Dempster et al., [1977](#bib.bib6)). On the contrary, Missing Not At Random settings, where missingness depends on the unobserved values, may not be identifiable and require dedicated methods with a model of the missing-values mechanism.

Learning predictive models with missing values poses distinct challenges compared to inference tasks (Josse et al., [2019](#bib.bib13)). Indeed, when the input is an arbitrary subset of variables in dimension d𝑑d, there are 2dsuperscript2𝑑2^{d} potential missing data patterns and as many sub-models to learn. Consequently, even simple data-generating mechanisms lead to complex decision rules (Le Morvan et al., [2020b](#bib.bib15)). To date, there are few supervised-learning models natively suited for partially-observed data. A notable exception is found with tree-based models (Twala et al., [2008](#bib.bib21); Chen and Guestrin, [2016](#bib.bib5)), widely used in data-science practice.

The most common practice however remains by far to use off-the-shelf methods first for imputation of missing values and second for supervised-learning on the resulting completed data. Such a procedure may benefit from progress in missing-value imputation with machine learning [van Buuren [2018](#bib.bib22), Yoon et al. [2018](#bib.bib23), Mattei and Frellsen [2019](#bib.bib17)].
However, there is a lack of learning theory to support such Impute-then-Regress procedures: Under what conditions are they Bayes consistent? Which aspects of the imputation are important?

There is empirical realization that the choice of imputation matters for predictive performance. The NADIA R package ([Borowski and Fic,](#bib.bib4) ) can select an imputation method to minimize a prediction error on a test set.
Auto-ML is used to optimize full pipelines, including imputation (eg Jarrett et al., [2021](#bib.bib12)).
Ipsen et al. ([2020](#bib.bib11)) optimize a constant imputation for supervised learning. However, the imputation is only weakly guided by the target in these approaches, it is set either from a family of black-box methods using gradient-free model selection, or from trivial imputation functions. In addition, there is a lack of insight on what drives a good imputation for prediction.

We contribute a systematic analysis of Impute-the-Regress procedures in a general setting: non-linear response function and any missingness mechanism (no MAR assumptions).
We show that:

* •

  Impute-then-Regress procedures are Bayes optimal for *all missing data mechanisms* and for *almost all imputation functions*, whatever the number of variables that may be missing. This very general result gives theoretical grounding to such widespread procedures.
* •

  We study “natural” choices of imputation and regression functions: the oracle imputation by the conditional expectation and oracle regression function on the complete data. We show that chaining these oracles is not Bayes optimal in general and quantify its excess risk. We show that in both cases, choosing an oracle for one step, imputation or regression, imposes discontinuities on the other step, thus making it harder to learn.
* •

  As these results suggest that imputation and regression should be adapted to one another, we contribute a method that jointly optimizes imputation and regression, using NeuMiss networks (Le Morvan et al., [2020a](#bib.bib14)) as a differentiable imputation procedure.
* •

  We compare empirically a number of Impute-then-Regress procedures on simulated non-linear regression tasks. Joint optimization of both steps provides the best performance.

## 2 Problem setting

#### Notations

We consider a dataset of i.i.d. realizations of the random variable (X,M,Y)∈ℝd×{0,1}d×ℝ𝑋𝑀𝑌superscriptℝ𝑑superscript01𝑑ℝ(X,M,Y)\in\mathbb{R}^{d}\times\left\{0,1\right\}^{d}\times\mathbb{R} where X𝑋X are the complete covariates, M𝑀M a missingness indicator, and Y𝑌Y a response of interest. For each realization (x,m,y)𝑥𝑚𝑦(x,m,y), mj=1subscript𝑚𝑗1m\_{j}=1 indicates that xjsubscript𝑥𝑗x\_{j} is missing, and mj=0subscript𝑚𝑗0m\_{j}=0 that it is observed. We denote by m​i​s​(m)⊂⟦1,d⟧𝑚𝑖𝑠𝑚

1𝑑mis(m)\subset\left\llbracket 1,d\right\rrbracket the indices corresponding to the missing covariates (and similarly o​b​s​(m)𝑜𝑏𝑠𝑚obs(m) the observed indices), so that xo​b​s​(m)subscript𝑥𝑜𝑏𝑠𝑚x\_{obs(m)} corresponds to the entries actually observed. We define the incomplete covariate vector X~∈(ℝ∪(𝙽𝙰))d~𝑋superscriptℝ𝙽𝙰𝑑\widetilde{X}\in\left(\mathbb{R}\cup\left(\mathtt{NA}\right)\right)^{d} as X~j=Xjsubscript~𝑋𝑗subscript𝑋𝑗\widetilde{X}\_{j}=X\_{j} if Mj=0subscript𝑀𝑗0M\_{j}=0 and X~j=𝙽𝙰subscript~𝑋𝑗𝙽𝙰\widetilde{X}\_{j}=\mathtt{NA} otherwise, where 𝙽𝙰𝙽𝙰\mathtt{NA} represents a missing value.

#### Assumptions

We assume that X𝑋X admits a density on ℝdsuperscriptℝ𝑑\mathbb{R}^{d} and that, for all j∈⟦1,d⟧,𝑗

1𝑑j\in\left\llbracket 1,d\right\rrbracket, each component Xjsubscript𝑋𝑗X\_{j} has finite expectation and variance, that is 𝔼​[Xj2]<∞𝔼delimited-[]superscriptsubscript𝑋𝑗2\mathbb{E}\left[X\_{j}^{2}\right]<\infty. Moreover, we assume that the response Y𝑌Y is generated according to:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Y=f⋆​(X)+ϵ,with​𝔼​[ϵ|Xo​b​s​(M),M]=0and​𝔼​[Y2]<∞.formulae-sequence𝑌superscript𝑓⋆𝑋italic-ϵformulae-sequencewith𝔼delimited-[]conditionalitalic-ϵ  subscript𝑋𝑜𝑏𝑠𝑀𝑀0and𝔼delimited-[]superscript𝑌2Y=f^{\star}(X)+\epsilon,\qquad\textrm{with}~{}~{}\mathbb{E}\left[\epsilon|X\_{obs(M)},M\right]=0\quad\textrm{and}~{}~{}\mathbb{E}\left[Y^{2}\right]<\infty. |  | (1) |

where f⋆:ℝd→ℝ:superscript𝑓⋆→superscriptℝ𝑑ℝf^{\star}:\mathbb{R}^{d}\to\mathbb{R} is a function of the complete input data X𝑋X, ϵ∈ℝitalic-ϵℝ\epsilon\in\mathbb{R} is a random noise variable.

### 2.1 Supervised learning with missing values

#### Optimization problem

In practice, in the presence of missing values, we do not have access to the complete data (X,M,Y)𝑋𝑀𝑌(X,M,Y) but only to the subset of it that is observed, i.e, (Xo​b​s​(M),M,Y)subscript𝑋𝑜𝑏𝑠𝑀𝑀𝑌(X\_{obs(M)},M,Y). Thus instead of learning a mapping from ℝdsuperscriptℝ𝑑\mathbb{R}^{d} to ℝℝ\mathbb{R}, we need to learn a mapping from (ℝ∪(𝙽𝙰))dsuperscriptℝ𝙽𝙰𝑑\left(\mathbb{R}\cup\left(\mathtt{NA}\right)\right)^{d} to ℝℝ\mathbb{R}, where the set of observed covariates can be any subset of ⟦1,d⟧

1𝑑\left\llbracket 1,d\right\rrbracket. It is this unusual input space, partly discrete, that makes supervised learning with missing values challenging and different from classical supervised learning problems. Formally, the optimization problem we wish to solve is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minf:(ℝ∪(𝙽𝙰))d↦ℝ⁡ℛ​(f):=𝔼​[(Y−f​(X~))2]assignsubscript:𝑓maps-tosuperscriptℝ𝙽𝙰𝑑ℝℛ𝑓𝔼delimited-[]superscript𝑌𝑓~𝑋2\min\_{f:\left(\mathbb{R}\cup\left(\mathtt{NA}\right)\right)^{d}\mapsto\mathbb{R}}\mathcal{R}(f):=\mathbb{E}\left[\left(Y-f(\widetilde{X})\right)^{2}\right] |  | (2) |

#### Bayes predictor

The function which minimizes ([2](#S2.E2 "In Optimization problem ‣ 2.1 Supervised learning with missing values ‣ 2 Problem setting ‣ What’s a good imputation to predict with missing values?")), called the *Bayes predictor*, is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~⋆​(X~)=𝔼​[Y|Xo​b​s​(M),M]=𝔼​[f⋆​(X)|Xo​b​s​(M),M].superscript~𝑓⋆~𝑋𝔼delimited-[]conditional𝑌  subscript𝑋𝑜𝑏𝑠𝑀𝑀𝔼delimited-[]conditionalsuperscript𝑓⋆𝑋  subscript𝑋𝑜𝑏𝑠𝑀𝑀\tilde{f}^{\star}(\widetilde{X})=\mathbb{E}\left[Y|X\_{obs(M)},M\right]=\mathbb{E}\left[f^{\star}(X)|X\_{obs(M)},M\right]. |  | (3) |

As X~~𝑋\widetilde{X} is a function of Xo​b​ssubscript𝑋𝑜𝑏𝑠X\_{obs} and M𝑀M, we will sometimes slightly abuse notations and write f~⋆​(X~)=f~⋆​(Xo​b​s,M)superscript~𝑓⋆~𝑋superscript~𝑓⋆subscript𝑋𝑜𝑏𝑠𝑀\tilde{f}^{\star}(\widetilde{X})=\tilde{f}^{\star}(X\_{obs},M). The risk of the Bayes predictor is called the *Bayes risk*, which we denote as ℛ⋆superscriptℛ⋆\mathcal{R}^{\star}. It is the lowest achievable risk for a given supervised learning problem.

###### Definition 1 (Bayes optimality).

A *Bayes optimal* function f𝑓f achieves the Bayes rate, i.e, ℛ​(f)=ℛ⋆ℛ𝑓superscriptℛ⋆\mathcal{R}(f)=\mathcal{R}^{\star}.

As can be seen from ([3](#S2.E3 "In Bayes predictor ‣ 2.1 Supervised learning with missing values ‣ 2 Problem setting ‣ What’s a good imputation to predict with missing values?")), the Bayes predictor is a function of M𝑀M, a discrete random variable that can take one of 2dsuperscript2𝑑2^{d} values since M∈{0,1}d𝑀superscript01𝑑M\in\left\{0,1\right\}^{d}. The function f~⋆superscript~𝑓⋆\tilde{f}^{\star} can thus be viewed as 2dsuperscript2𝑑2^{d} different functions, one for each possible subset of variables. This view raises questions that are central to this paper: How should we parametrize functions on such input domains? And which function families should we consider to approximate f~⋆superscript~𝑓⋆\tilde{f}^{\star}? These questions have been studied in the case where f⋆superscript𝑓⋆f^{\star} is assumed to be a linear function, and X𝑋X follows a Gaussian distribution. Indeed, under these assumptions, Le Morvan et al. ([2020b](#bib.bib15), [a](#bib.bib14)) have derived analytical expressions for the Bayes predictor and deduced appropriate parametric estimators. However, aside from specific cases, it is impossible to derive an analytical expression for the Bayes predictor, and novel arguments are needed to understand which classes of functions should be considered in general.

## 3 Asymptotic analysis of Impute-then-regress procedures

### 3.1 Impute-then-regress procedures

Let |m​i​s​(m)|𝑚𝑖𝑠𝑚|mis(m)| (resp. |o​b​s​(m)|𝑜𝑏𝑠𝑚|obs(m)|) be the number of missing entries (resp. observed) for any missing data pattern m𝑚m. For each m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}, we define an *imputation function* ϕ(m):ℝ|o​b​s​(m)|→ℝ|m​i​s​(m)|:superscriptitalic-ϕ𝑚→superscriptℝ𝑜𝑏𝑠𝑚superscriptℝ𝑚𝑖𝑠𝑚\phi^{(m)}:\mathbb{R}^{|obs(m)|}\to\mathbb{R}^{|mis(m)|} which outputs values for the missing entries based on the observed ones. We denote by ϕj(m):ℝ|o​b​s​(m)|→ℝ:superscriptsubscriptitalic-ϕ𝑗𝑚→superscriptℝ𝑜𝑏𝑠𝑚ℝ\phi\_{j}^{(m)}:\mathbb{R}^{|obs(m)|}\to\mathbb{R} the component function of ϕ(m)superscriptitalic-ϕ𝑚\phi^{(m)} that imputes the j𝑗j-th component in X𝑋X if it is missing. Classical choices of imputation functions include constant functions or linear functions. Finally, we introduce the family of functions ℱIsuperscriptℱ𝐼\mathcal{F}^{I} that transform an incomplete vector into a complete one, precisely:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℱI={Φ:(ℝ∪{𝙽𝙰})d→ℝd:∀j∈⟦1,d⟧,Φj​(X~)={Xjif​Mj=0ϕj(M)​(Xo​b​s​(M))if​Mj=1}.superscriptℱ𝐼conditional-setΦ:→superscriptℝ𝙽𝙰𝑑superscriptℝ𝑑formulae-sequencefor-all𝑗  1𝑑subscriptΦ𝑗~𝑋casessubscript𝑋𝑗ifsubscript𝑀𝑗0subscriptsuperscriptitalic-ϕ𝑀𝑗subscript𝑋𝑜𝑏𝑠𝑀ifsubscript𝑀𝑗1\mathcal{F}^{I}=\left\{\Phi:\left(\mathbb{R}\cup\left\{\mathtt{NA}\right\}\right)^{d}\to\mathbb{R}^{d}:\forall j\in\left\llbracket 1,d\right\rrbracket,\>\Phi\_{j}(\widetilde{X})=\begin{cases}X\_{j}\>&\text{if}\>M\_{j}=0\\ \phi^{(M)}\_{j}(X\_{obs(M)})\>&\text{if}\>M\_{j}=1\end{cases}\right\}. |  | (4) |

Let us define ℱ∞Isubscriptsuperscriptℱ𝐼\mathcal{F}^{I}\_{\infty} in the exact same way but for imputation functions ϕ(m)∈𝒞∞superscriptitalic-ϕ𝑚superscript𝒞\phi^{(m)}\in\mathcal{C}^{\infty}, for all m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}. Here we study *Impute-then-regress procedures*, which we define as two-step procedures where the data is first imputed using a function Φ∈ℱIΦsuperscriptℱ𝐼\Phi\in\mathcal{F}^{I}, and then a regression is performed on the imputed data. Such a procedure is quite natural to deal with arbitrary subsets of inputs variables. It embeds the data into ℝdsuperscriptℝ𝑑\mathbb{R}^{d} to reduce the problem to a classical one. In practice, impute-then-regress procedures are widely used. However, the choice of function class is so far mostly ad-hoc and raises multiple questions: How close to the Bayes rate can functions obtained via such procedures be? Should we prefer some choices of imputation functions over others? What happens when the missing data mechanism is missing not at random? In this section, we will give answers to these questions.

Below, we write o​b​s𝑜𝑏𝑠obs (resp. m​i​s𝑚𝑖𝑠mis) instead of o​b​s​(M)𝑜𝑏𝑠𝑀obs(M) (resp. m​i​s​(M)𝑚𝑖𝑠𝑀mis(M)) to lighten notations.

### 3.2 Impute-then-regress procedures are Bayes optimal

###### Definition 2 (Universal consistency).

An estimator fnsubscript𝑓𝑛f\_{n} is *Bayes consistent* if limn→∞ℛ​(fn)=ℛ⋆subscript→𝑛ℛsubscript𝑓𝑛superscriptℛ⋆\lim\_{n\to\infty}\mathcal{R}(f\_{n})=\mathcal{R}^{\star}. It is said to be *universally consistent* if the previous statement holds for all distributions of (X,Y)𝑋𝑌(X,Y).

The following theorem shows that Impute-then-regress procedures are Bayes optimal for almost all imputation functions. In other words, it means that a universal learner trained on imputed data provides optimal performances asymptotically for almost all imputation functions. Let us now define, for all imputation functions Φ∈ℱIΦsuperscriptℱ𝐼\Phi\in\mathcal{F}^{I}, the function gΦ⋆∈argming:ℝd↦ℝ𝔼​[(Y−g∘Φ​(X~))2]subscriptsuperscript𝑔⋆Φ

:𝑔maps-tosuperscriptℝ𝑑ℝargmin𝔼delimited-[]superscript𝑌𝑔Φ~𝑋2g^{\star}\_{\Phi}\in\underset{g:\mathbb{R}^{d}\mapsto\mathbb{R}}{\text{argmin}}\quad\mathbb{E}\left[\left(Y-g\circ\Phi(\widetilde{X})\right)^{2}\right].

###### Theorem 3.1 (Bayes consistency of Impute-then-regress procedures).

Assume the data is generated according to ([1](#S2.E1 "In Assumptions ‣ 2 Problem setting ‣ What’s a good imputation to predict with missing values?")).
Then, for almost all imputation function Φ∈ℱ∞IΦsubscriptsuperscriptℱ𝐼\Phi\in\mathcal{F}^{I}\_{\infty}, the function gΦ⋆∘Φsubscriptsuperscript𝑔⋆ΦΦg^{\star}\_{\Phi}\circ\Phi is Bayes optimal.
In other words, for almost all imputation functions Φ∈ℱ∞IΦsubscriptsuperscriptℱ𝐼\Phi\in\mathcal{F}^{I}\_{\infty}, a universally consistent algorithm trained on the imputed data Φ​(X~)Φ~𝑋\Phi(\widetilde{X}) is Bayes consistent.

Appendix [A.3](#A1.SS3 "A.3 Proof of Theorem 3.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") gives the proof. Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") states a very general result: Impute-then-regress procedures are Bayes consistent for all missing data mechanisms, almost all imputation functions, regardless of the distribution of (X,Y)𝑋𝑌(X,Y) and the number of missing covariates. Since Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") holds for almost all imputation functions, it implies that good imputations are not required to obtain good predictive performances, at least asymptotically. Note that here, the notion of *almost all* is to be understood in its topological sense, and not in its measure theory sense. Moreover, this theorem does not make any assumption on the missing data mechanism, and is therefore valid for Missing Not At Random (MNAR) data. This contrasts with most methods for inference and imputation with missing values, valid only for MAR data. Finally, the theorem remains valid for any configuration of variables that may contain missing values, including the case in which all variables may contain missing values.
Bayes consistency of Impute-the-Regress procedures has already been
studied, but in much more restricted settings.
Josse et al. ([2019](#bib.bib13)) proved that such procedures are Bayes consistent under the MAR assumption, for constant imputations functions and for only one potentially missing variable.
Bertsimas et al. ([2021](#bib.bib2)) refined this result to almost surely continuous imputation functions. While these two prior works build on very similar proofs, we use here very different arguments summarized in the next paragraph.

Figure 1: Example - Imputation manifolds in three dimensions — 3-dimensional Gaussian data after imputation. Data points are colored according to their missing data pattern prior to imputation. Red, brown and purple (resp. orange, blue, and green) correspond to missing data patterns with two (resp. one) missing value(s). Completely observed points are not represented to ease the visualization of manifolds.

![Refer to caption](/html/2106.00311/assets/manifolds_corrected_points.png)

The first key idea of the proof is that, after imputation, all data points with a given missing data pattern m𝑚m are mapped to a manifold ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} of dimension |o​b​s​(m)|𝑜𝑏𝑠𝑚|obs(m)|. For example in 3D, data points are mapped to ℝ3superscriptℝ3\mathbb{R}^{3} when completely observed, to 2D manifolds when they have one value missing, to 1D manifolds when they have two values missing, and to one point when all values are missing (see Figure [1](#S3.F1.fig1 "Figure 1 ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?")). Thus, Impute-then-Regress procedures first map data points to various manifolds depending on their missing data patterns and then apply a prediction function defined on the whole space including manifolds. The second key idea of the proof is to ensure that the original missing data patterns of imputed points can almost surely be identified. For this, the proof requires that all manifolds of the same dimension are pairwise transverse. This assumption is sufficient, though not necessary, to ensure that the intersection of two manifolds of dimension |o​b​s​(m)|𝑜𝑏𝑠𝑚|obs(m)| cannot itself be of dimension |o​b​s​(m)|𝑜𝑏𝑠𝑚|obs(m)|. Transversality is a weak assumption. In fact, Thom’s transversality theorem, (which we rely on in our proof) says that it is a generic property: it holds for “typical examples”, i.e *almost all* imputation functions will lead to transverse manifolds. To clarify this concept, we provide a particular case in 2D where 1D manifolds are not transverse in Appendix [A.4](#A1.SS4 "A.4 Examples of transverse and nontransverse manifolds in 2D. ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?").

The proof is constructive and exhibits a function gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} which achieves the Bayes rate for a given set of imputation functions. For each manifold ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)}, ordered from smallest dimension to largest, we require that gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} on ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} equals the Bayes predictor for missing data pattern m𝑚m except on points for which gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} has already been defined, i.e, the points where ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} intersects with the manifolds ranked before it. Thus, we obtain a function gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} that does not depend on m𝑚m, and which for each manifold, equals the Bayes predictor except on subsets of measure zero under the assumption that manifolds of the same dimension are pairwise transverse. Refer to appendix [A.3](#A1.SS3 "A.3 Proof of Theorem 3.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") for more details.

While this theorem is a very general result, it does not say what the optimal function associated to a given imputation looks like. In fact, depending on the imputation function it may be non-continuous, vary widely, and require a very large number of samples to be approximated.

#### Note on Impute-then-classify procedures -

Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") applies to regression problems. However, it can easily be shown that a similar result holds in binary classification settings. Indeed, in a binary classification setting, the Bayes predictor predicts class 1 if P​(Y=1|X)>0.5𝑃𝑌conditional1𝑋0.5P(Y=1|X)>0.5 and -1 otherwise. Thus, it suffices to consider that the function of interest f⋆​(X)superscript𝑓⋆𝑋f^{\star}(X) is the posterior probability P​(Y=1|X)𝑃𝑌conditional1𝑋P(Y=1|X). Then the same arguments as those used to prove Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") can be used to show that Impute-then-classify procedures are Bayes optimal for almost all imputation functions.

## 4 Imputation versus regression: choosing one may break the other

Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") gives a theoretical grounding to Impute-then-regress procedures. As it holds for almost any imputation function, one could very well choose simple and cheap imputations such as imputing by a constant. However, the difficulty of the ensuing learning problem will depend on the choice of imputation function. Indeed, the function gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} that achieves the Bayes rate depends on the imputation function ΦΦ\Phi. In general, it may not be continuous or smooth. Thus gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} can be more or less difficult to approximate by machine learning algorithms depending on the chosen imputation function.

Le Morvan et al. ([2020b](#bib.bib15)) showed that even if Y𝑌Y is a linear function of X𝑋X, imputing by a constant leads to a complicated Bayes predictor: piecewise affine but with 2dsuperscript2𝑑2^{d} regions. This result highlights how imputations neglecting the structure of covariates can result in additional complexity for the regression function gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi}. Rather, another common practice is to impute by the conditional expectation: it minimizes the mean squared error between the imputed matrix and the complete one and is the target of most imputation methods. One hope may be that if f⋆superscript𝑓⋆f^{\star} has desirable properties, such as smoothness, conditional imputation will lead to a function gΦ⋆subscriptsuperscript𝑔⋆Φg^{\star}\_{\Phi} which inherits from these properties.

In this section we first show that replacing missing values by their conditional expectation in the oracle regression function f⋆superscript𝑓⋆f^{\star} gives a small but non-zero risk. Characterizing the optimal function on the conditionally-imputed data, we find that it suffers from discontinuities and thus forms a difficult estimation problem. Rather, we study whether the imputation can be corrected for f⋆superscript𝑓⋆f^{\star} to form the Bayes predictor on partially-observed data.

### 4.1 Applying f⋆superscript𝑓⋆f^{\star} on conditional imputations: chaining oracles isn’t without risks.

The conditional imputation function ΦC​I:(ℝ∪{𝙽𝙰})d→ℝd:superscriptΦ𝐶𝐼→superscriptℝ𝙽𝙰𝑑superscriptℝ𝑑\Phi^{CI}:\left(\mathbb{R}\cup\left\{\mathtt{NA}\right\}\right)^{d}\to\mathbb{R}^{d} is defined as follows:

|  |  |  |
| --- | --- | --- |
|  | ∀j∈⟦1,d⟧,ΦjC​I​(X~)={Xjif​Mj=0𝔼​[Xj|Xo​b​s,M]if​Mj=1formulae-sequencefor-all𝑗  1𝑑subscriptsuperscriptΦ𝐶𝐼𝑗~𝑋casessubscript𝑋𝑗ifsubscript𝑀𝑗0𝔼delimited-[]conditionalsubscript𝑋𝑗  subscript𝑋𝑜𝑏𝑠𝑀ifsubscript𝑀𝑗1\forall j\in\left\llbracket 1,d\right\rrbracket,\>\Phi^{CI}\_{j}(\widetilde{X})=\begin{cases}X\_{j}\>&\text{if}\>M\_{j}=0\\ \mathbb{E}\left[X\_{j}|X\_{obs},M\right]\>&\text{if}\>M\_{j}=1\end{cases} |  |

Note that ΦC​I∈ℱIsuperscriptΦ𝐶𝐼superscriptℱ𝐼\Phi^{CI}\in\mathcal{F}^{I}. To lighten notations, we will write XC​I:=ΦC​I​(X~)assignsuperscript𝑋𝐶𝐼superscriptΦ𝐶𝐼~𝑋X^{CI}:=\Phi^{CI}(\widetilde{X}) to denote the conditionally imputed data.

###### Lemma 4.1 (First order approximation).

Assume that the data is generated according to ([1](#S2.E1 "In Assumptions ‣ 2 Problem setting ‣ What’s a good imputation to predict with missing values?")). Moreover assume that (i) f⋆∈𝒞2​(𝒮,ℝ)superscript𝑓⋆superscript𝒞2𝒮ℝf^{\star}\in\mathcal{C}^{2}(\mathcal{S},\mathbb{R}) where 𝒮⊂ℝd𝒮superscriptℝ𝑑\mathcal{S}\subset\mathbb{R}^{d} is the support of the data, and that (ii) there exists positive semidefnite matrices H¯+∈Pd+superscript¯𝐻superscriptsubscript𝑃𝑑\bar{H}^{+}\in P\_{d}^{+} and H¯−∈Pd+superscript¯𝐻superscriptsubscript𝑃𝑑\bar{H}^{-}\in P\_{d}^{+} such that for all X𝑋X in 𝒮𝒮\mathcal{S}, H¯−≼H​(X)≼H¯+precedes-or-equalssuperscript¯𝐻𝐻𝑋precedes-or-equalssuperscript¯𝐻\bar{H}^{-}\preccurlyeq H(X)\preccurlyeq\bar{H}^{+} with H​(X)𝐻𝑋H(X) the Hessian of f⋆superscript𝑓⋆f^{\star} at X𝑋X. Then for all X𝑋X in 𝒮𝒮\mathcal{S} and for all missing data patterns:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​tr​(H¯m​i​s,m​i​s−​Σm​i​s|o​b​s,M)≤f~⋆​(X~)−f⋆​(XC​I)≤12​tr​(H¯m​i​s,m​i​s+​Σm​i​s|o​b​s,M)12trsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼12trsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀\frac{1}{2}\text{tr}\left(\bar{H}^{-}\_{mis,mis}\Sigma\_{mis|obs,M}\right)\leq\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\leq\frac{1}{2}\text{tr}\left(\bar{H}^{+}\_{mis,mis}\Sigma\_{mis|obs,M}\right) |  | (5) |

where Σm​i​s|o​b​s,MsubscriptΣconditional𝑚𝑖𝑠

𝑜𝑏𝑠𝑀\Sigma\_{mis|obs,M} is the covariance matrix of the distribution of Xm​i​ssubscript𝑋𝑚𝑖𝑠X\_{mis} given Xo​b​ssubscript𝑋𝑜𝑏𝑠X\_{obs} and M𝑀M.

Appendix [A.6](#A1.SS6 "A.6 Proof of Lemma 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") gives the proof. The assumption that H¯−≼H​(X)≼H¯+precedes-or-equalssuperscript¯𝐻𝐻𝑋precedes-or-equalssuperscript¯𝐻\bar{H}^{-}\preccurlyeq H(X)\preccurlyeq\bar{H}^{+} for any X𝑋X means that the minimum and maximum curvatures of f⋆superscript𝑓⋆f^{\star} in any direction are uniformly bounded over the entire space. Lemma [4.1](#S4.Thmlemma1 "Lemma 4.1 (First order approximation). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") shows that applying f⋆superscript𝑓⋆f^{\star} to the conditionally imputed (CI) data is a good approximation of the Bayes predictor when there is no direction in which both the curvature of f⋆superscript𝑓⋆f^{\star} and the conditional variance of the missing data given the observed one are high. Intuitively, if a low quality imputation is compensated by a flat function, or conversely, if a fast varying function is compensated by a high quality imputation, then f⋆superscript𝑓⋆f^{\star} applied to the CI data approximates well the Bayes predictor.

###### Proposition 4.1 ((Non-)Consistency of chaining oracles).

The function f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI} is Bayes optimal if and only if the function f⋆superscript𝑓⋆f^{\star} and the imputed data XC​Isuperscript𝑋𝐶𝐼X^{CI} satisfy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∀M​s.t.​P​(M)>0,𝔼​[f⋆​(X)|Xo​b​s,M]=f⋆​(XC​I)almost everywhere.formulae-sequencefor-all𝑀s.t.𝑃𝑀0𝔼delimited-[]conditionalsuperscript𝑓⋆𝑋  subscript𝑋𝑜𝑏𝑠𝑀  superscript𝑓⋆superscript𝑋𝐶𝐼almost everywhere\forall M\;\text{s.t.}\;P(M)>0,\quad\mathbb{E}\left[f^{\star}(X)|X\_{obs},M\right]=f^{\star}(X^{CI})\quad\text{almost everywhere}. |  | (6) |

Besides,
under the assumptions of Lemma [4.1](#S4.Thmlemma1 "Lemma 4.1 (First order approximation). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?"), the excess risk of chaining oracles compared to the Bayes risk ℛ⋆superscriptℛ⋆\mathcal{R}^{\star} is upper-bounded by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℛ​(f⋆∘ΦC​I)−ℛ⋆≤14​𝔼M​[max⁡(t​r​(H¯m​i​s,m​i​s−​Σm​i​s|o​b​s,M)2,t​r​(H¯m​i​s,m​i​s+​Σm​i​s|o​b​s,M)2)]ℛsuperscript𝑓⋆superscriptΦ𝐶𝐼superscriptℛ⋆14subscript𝔼𝑀delimited-[]𝑡𝑟superscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2𝑡𝑟superscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2\mathcal{R}(f^{\star}\circ\Phi^{CI})-\mathcal{R}^{\star}\leq\frac{1}{4}\mathbb{E}\_{M}\left[\max\left(tr\left(\bar{H}^{-}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2},tr\left(\bar{H}^{+}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2}\right)\right] |  | (7) |

Appendix [A.7](#A1.SS7 "A.7 Proof of Proposition 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") gives the proof.
Condition ([6](#S4.E6 "In Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")) for Bayes optimality is clearly stringent.
Indeed, if one variable is missing, condition ([6](#S4.E6 "In Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")) says that the expectation of the regression function should be equal to the regression function applied at the expected entry. Although such functions are difficult to characterize precisely, it is clear that condition ([6](#S4.E6 "In Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")) is difficult to fulfill for generic regression functions (linear functions are among the few examples that do satisfy it). Therefore, for most functions f⋆superscript𝑓⋆f^{\star}, f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI} is not Bayes optimal. Proposition [4.1](#S4.Thmproposition1 "Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") also gives an upper bound for the excess risk of the predictor f⋆​(XC​I)superscript𝑓⋆superscript𝑋𝐶𝐼f^{\star}(X^{CI}) compared to the Bayes rate, showing here again that if there is no direction in which both the curvature and the variance of the missing data given the observed one are high, the excess risk is small.

*The special case of linear regression:* When f⋆superscript𝑓⋆f^{\star} is a linear function, the curvature is 0, hence eq. ([7](#S4.E7 "In Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")) implies no excess risk. This is also visible from the expression of the Bayes predictor ([3](#S2.E3 "In Bayes predictor ‣ 2.1 Supervised learning with missing values ‣ 2 Problem setting ‣ What’s a good imputation to predict with missing values?")), where the expectation on unobserved data can be pushed inside f⋆superscript𝑓⋆f^{\star} as it is linear. The Bayes predictor can thus be exactly written as f⋆superscript𝑓⋆f^{\star} applied to conditionally-imputed data.

### 4.2 Regressing on conditional imputations, a good idea?

###### Proposition 4.2 (Regression function discontinuities).

Suppose that f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI} is not Bayes optimal, and that the probability of observing all variables is strictly positive, i.e., for all x𝑥x, P​(M=(0,…,0),X=x)>0𝑃formulae-sequence𝑀0…0𝑋𝑥0P(M=(0,\dots,0),X=x)>0. Then there is no continuous function g𝑔g such that g∘ΦC​I𝑔superscriptΦ𝐶𝐼g\circ\Phi^{CI} is Bayes optimal.

In other words, when conditional imputation is used, the optimal regression function experiences discontinuities unless it is f⋆superscript𝑓⋆f^{\star}. The proof is given in appendix [A.8](#A1.SS8 "A.8 Proof of Proposition 4.2 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?"). From a finite-sample learning standpoint, discontinuous functions are in general harder to learn: in the general case, non-parametric regression requires more samples to achieve a given error on functions without specific regularities as opposed to functions with a form of smoothness
(see e.g., Györfi et al., [2006](#bib.bib10), chap 3). Hence, while regression on conditional imputation may be consistent ([Theorem 3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?")), it can require an inordinate number of samples.

### 4.3 Fasten your seat belt: corrected imputations may experience discontinuities.

![Refer to caption](/html/2106.00311/assets/x1.png)

Bowl                                                    
![Refer to caption](/html/2106.00311/assets/x2.png)Wave

Figure 2: Left: corrected imputation The regression function is f⋆​(x1,x2)↦x12+x22maps-tosuperscript𝑓⋆subscript𝑥1subscript𝑥2superscriptsubscript𝑥12superscriptsubscript𝑥22f^{\star}(x\_{1},x\_{2})\mapsto x\_{1}^{2}+x\_{2}^{2}. When x2subscript𝑥2x\_{2} is missing, chaining perfect conditional imputation with the regression function (f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI}) gives a biased predictor, shown in red, as the unexplained variance in x2subscript𝑥2x\_{2} is turned into bias. However, using as an imputation Φ​(x1)=ρ2​x12+(1−ρ2)Φsubscript𝑥1superscript𝜌2superscriptsubscript𝑥121superscript𝜌2\Phi(x\_{1})=\sqrt{\rho^{2}x\_{1}^{2}+(1-\rho^{2})} corrects this bias, with ρ𝜌\rho the correlation between x1subscript𝑥1x\_{1} and x2subscript𝑥2x\_{2}.
Right: no continuous corrected imputation exists. The
function is defined as f⋆​(x1,x2)↦x22−3​x2maps-tosuperscript𝑓⋆subscript𝑥1subscript𝑥2superscriptsubscript𝑥223subscript𝑥2f^{\star}(x\_{1},x\_{2})\mapsto x\_{2}^{2}-3\,x\_{2}. No continuous corrected imputation is possible because the Bayes
predictor on the partially-observed data 𝔼​[Y|X1]𝔼delimited-[]conditional𝑌subscript𝑋1\mathbb{E}[Y|X\_{1}] is
monotonous, while the regression function f⋆superscript𝑓⋆f^{\star} is not.

Another possible route is to find *corrected imputations* which we
define as imputation functions ΦΦ\Phi such that, if f⋆superscript𝑓⋆f^{\star} is used as
regression function, the impute-then-regress procedure f⋆∘Φsuperscript𝑓⋆Φf^{\star}\circ\Phi is Bayes optimal. Intuitively, given a
fixed regression function f⋆superscript𝑓⋆f^{\star}, the question is: can we "correct" an
imputation function and thus the manifold that it describes so that
f⋆superscript𝑓⋆f^{\star} restricted to this manifold is equal to the Bayes
predictor?
Assuming f⋆superscript𝑓⋆f^{\star} is continuous, the intermediate value theorem gives a first answer to this question by ensuring the existence of imputations functions satisfying

|  |  |  |
| --- | --- | --- |
|  | f⋆∘Φ​(Xo​b​s​(M),M)=𝔼​[f⋆​(X)|Xo​b​s​(M),M].superscript𝑓⋆Φsubscript𝑋𝑜𝑏𝑠𝑀𝑀𝔼delimited-[]conditionalsuperscript𝑓⋆𝑋  subscript𝑋𝑜𝑏𝑠𝑀𝑀\displaystyle f^{\star}\circ\Phi(X\_{obs(M)},M)=\mathbb{E}\left[f^{\star}(X)|X\_{obs(M)},M\right]. |  |

For the same reasons as above, determining that such imputations not only exist but are *continuous* is important from a practical perspective. Indeed, assuming f⋆superscript𝑓⋆f^{\star} is continuous, the Bayes predictor with missing values could then be tackled as the composition of two continuous functions, with an Impute-then-Regress strategy. Intuitively in 2D, the existence of a continuous corrected imputation can be seen as the existence of a continuous path in the 2D plane whose value by f⋆superscript𝑓⋆f^{\star} equals the Bayes predictor. Figure [2](#S4.F2 "Figure 2 ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") (left) gives a simple example in 2D for which a continuous corrected imputation exists. Here if one chooses the imputation function of X2subscript𝑋2X\_{2} given X1subscript𝑋1X\_{1} as the black function denoted as Φc​o​r​r​e​c​t​e​dsubscriptΦ𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑\Phi\_{corrected}, then its composition by the green paraboloid f⋆superscript𝑓⋆f^{\star} gives the Bayes Predictor depicted on the right in black. By contrast, if one imputes by the conditional expectation, then its composition with f⋆superscript𝑓⋆f^{\star} gives the red curve which is different from the Bayes Predictor. Note that the manifolds in Figure [1](#S3.F1.fig1 "Figure 1 ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") were obtained using (continuous) corrected imputations functions for the same setting as Figure [2](#S4.F2 "Figure 2 ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") (left) but with 3-dimensional data. However, as illustrated in Figure [2](#S4.F2 "Figure 2 ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") (right), *continuous* corrected imputations do not always exist. Indeed, on this example the Bayes predictor is non-decreasing but there is no continuous path in the 2D plane on which f⋆superscript𝑓⋆f^{\star} is non-decreasing and maps at some point to both the ’purple’ and ’yellow values’ (proof in Appendix [A.9](#A1.SS9 "A.9 Example of a case where no continuous corrected imputation exists. ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")). It is thus of interest to clarify when continuous corrected imputations exist. Proposition [4.3](#S4.Thmproposition3 "Proposition 4.3 (Existence of continuous corrected imputations). ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") establishes such conditions.

###### Proposition 4.3 (Existence of continuous corrected imputations).

Assume that f⋆superscript𝑓⋆f^{\star} is uniformly continuous, twice continuously differentiable and that, for all missing patterns m𝑚m and all xo​b​ssubscript𝑥𝑜𝑏𝑠x\_{obs}, the support of Xm​i​s|Xo​b​s=xo​b​s,M=mformulae-sequenceconditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠subscript𝑥𝑜𝑏𝑠𝑀𝑚X\_{mis}|X\_{obs}=x\_{obs},M=m is connected.
Additionally, assume that for all missing patterns m𝑚m, and all (xo​b​s,xm​i​s)subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠(x\_{obs},x\_{mis}), the gradient of f⋆superscript𝑓⋆f^{\star} with respect to the missing coordinates is nonzero:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∇xm​i​sf⋆​(xo​b​s,xm​i​s)≠0.subscript∇subscript𝑥𝑚𝑖𝑠superscript𝑓⋆subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠0\nabla\_{x\_{mis}}f^{\star}(x\_{obs},x\_{mis})\neq 0. |  | (8) |

Then, for all m𝑚m, theres exist continuous imputation functions ϕ(m):ℝ|o​b​s​(m)|→ℝ|m​i​s​(m)|:superscriptitalic-ϕ𝑚→superscriptℝ𝑜𝑏𝑠𝑚superscriptℝ𝑚𝑖𝑠𝑚\phi^{(m)}:\mathbb{R}^{|obs(m)|}\to\mathbb{R}^{|mis(m)|} such that f⋆∘Φsuperscript𝑓⋆Φf^{\star}\circ\Phi is Bayes optimal.

Appendix [A.10](#A1.SS10 "A.10 Proof of Proposition 4.3 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") gives a proof based on a global implicit function theorem. Assumption [8](#S4.E8 "In Proposition 4.3 (Existence of continuous corrected imputations). ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") is restrictive: it is for instance not met for our example in Figure [2](#S4.F2 "Figure 2 ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?") (left), which still admits continuous corrected imputations. This highlights the fact that continuous corrected imputations also exist under weaker conditions, but it is difficult to conclude on “how often” it is the case.

## 5 Jointly optimizing an impute-n-regress procedure: NeuMiss+MLP

The above suggests that it is beneficial to adapt the regression function to the imputation procedure and vice versa. Hence, we introduce a method for the joint optimization of these two steps by chaining a NeuMiss network with an MLP (multi-layer perceptron).

NeuMiss (Le Morvan et al., [2020a](#bib.bib14)) is a neural-network architecture originally designed to approximate the Bayes predictor for linear models with missing values. It contains a Neumann block that has the particularity of using element-wise multiplications by the missingness indicator as non-linearities. Here we reuse this block to play the role of an imputation layer. This choice is motivated by two key reasons. First, the Neumann block is a theoretically grounded layer for missing values: it can approximate the conditional expectation of the missing values given the observed ones with an error that decays exponentially fast with its depth. As explained in Prop. 4.1, this property is desirable in some cases. Second, it is a differentiable block, which allows it to be chained with a MLP and learned jointly with the regression function. The resulting architecture can thus be seen as an Impute-then-Regress architecture, but that can be jointly optimized.

We performed one minor improvement on the NeuMiss architecture compared to the original paper. Though the theory behind NeuMiss points to using shared weights in the Neumann block as well as residual connections going from the input to each hidden layer of the Neumann block, Le Morvan et al. ([2020a](#bib.bib14)) used neither. We found empirically that shared weights in the Neumann block as well as residual connections improved performance. Therefore, we used both in all our experiments. For clarity, the (non-linear) NeuMiss architecture is described in detail in Appendix [B.1](#A2.SS1 "B.1 NeuMiss+MLP architecture ‣ Appendix B Additional results ‣ What’s a good imputation to predict with missing values?").

## 6 Empirical study of impute-n-regress procedures

### 6.1 Experimental setup

#### Data generation

The data X∈ℝn×d𝑋superscriptℝ𝑛𝑑X\in\mathbb{R}^{n\times d} are generated according to a multivariate Gaussian distribution 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) where the mean is drawn from a standard Gaussian and the covariance is generated as Σ=B​B⊤+DΣ𝐵superscript𝐵top𝐷\Sigma=BB^{\top}+D. B∈ℝd×q𝐵superscriptℝ𝑑𝑞B\in\mathbb{R}^{d\times q} is a matrix with entries drawn from a standard normal Gaussian distribution, and D𝐷D is a diagonal matrix with small entries that ensures that the covariance matrix is full rank. We study two correlation settings called *high* and *low* corresponding respectively to q=int​(0.3∗d)𝑞int0.3𝑑q=\texttt{int}(0.3\*d) and q=int​(0.7∗d)𝑞int0.7𝑑q=\texttt{int}(0.7\*d). The experiments are run with d=50𝑑50d=50.

#### Choice of f⋆superscript𝑓⋆f^{\star}

The response Y𝑌Y is generated according to Y=f⋆​(X)+ϵ𝑌superscript𝑓⋆𝑋italic-ϵY=f^{\star}(X)+\epsilon with three choices of f⋆superscript𝑓⋆f^{\star} named *bowl*, *wave*, and *break* depicted in Figure [3](#S6.F3.2 "Figure 3 ‣ Choice of 𝑓^⋆ ‣ 6.1 Experimental setup ‣ 6 Empirical study of impute-n-regress procedures ‣ What’s a good imputation to predict with missing values?") (exact expression in appendix [B.2](#A2.SS2 "B.2 Expressions of 𝑓^⋆_{𝑏⁢𝑜⁢𝑤⁢𝑙}, 𝑓^⋆_{𝑤⁢𝑎⁢𝑣⁢𝑒} and 𝑓^⋆_{𝑏⁢𝑟⁢𝑒⁢𝑎⁢𝑘} and the corresponding Bayes predictors. ‣ Appendix B Additional results ‣ What’s a good imputation to predict with missing values?")). β𝛽\beta is a vector of ones normalized such that the quantity z=β⊤​X+β0𝑧superscript𝛽top𝑋subscript𝛽0z=\beta^{\top}X+\beta\_{0} follows a Gaussian distribution centered on 1 with variance 1. Note that fb​o​w​l⋆subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙f^{\star}\_{bowl}, fw​a​v​e⋆subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒f^{\star}\_{wave} and fb​r​e​a​k⋆subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘f^{\star}\_{break} were designed so that the desired variations occur over the support of the data. The noise ϵitalic-ϵ\epsilon is chosen so as to have a signal-to-noise ratio of 10.

Figure 3: Bowl, wave and break functions used for f⋆superscript𝑓⋆f^{\star} in the empirical study.

![Refer to caption](/html/2106.00311/assets/x3.png)

![Refer to caption](/html/2106.00311/assets/x4.png)

![Refer to caption](/html/2106.00311/assets/x5.png)

#### Missing values

50% of the entries of X𝑋X were deleted according to one of two missing data mechanisms: Missing Completely At Random (MCAR) or Gaussian self-masking (GSM, see Le Morvan et al., [2020a](#bib.bib14)). Gaussian self-masking is a Missing Not At Random (MNAR) mechanism, where the probability that a variable j𝑗j is missing depends on Xjsubscript𝑋𝑗X\_{j} via a Gaussian function.

#### Baseline methods benchmarked

For each level of correlation (*low* or *high*), for each function f⋆superscript𝑓⋆f^{\star} (*bowl*, *wave* or *break*), and each missing data mechanism (MCAR or GSM), we compare a number of methods. First, for reference, we compute various oracle predictors:

* •

  Bayes predictor: This is the function that achieves the lowest achievable risk. In general cases, its expression cannot be derived analytically. However, we show that it can be derived for ridge functions, i.e. functions of the form x↦g​(β⊤​x)maps-to𝑥𝑔superscript𝛽top𝑥x\mapsto g(\beta^{\top}x), for some choices of g𝑔g including polynomials, the Gaussian cdf and piecewise constant functions. We thus built fb​o​w​l⋆subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙f^{\star}\_{bowl}, fw​a​v​e⋆subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒f^{\star}\_{wave} and fb​r​e​a​ksubscript𝑓𝑏𝑟𝑒𝑎𝑘f\_{break} as combination of these base functions which allows us to compute their corresponding Bayes predictors. Appendix [B.2](#A2.SS2 "B.2 Expressions of 𝑓^⋆_{𝑏⁢𝑜⁢𝑤⁢𝑙}, 𝑓^⋆_{𝑤⁢𝑎⁢𝑣⁢𝑒} and 𝑓^⋆_{𝑏⁢𝑟⁢𝑒⁢𝑎⁢𝑘} and the corresponding Bayes predictors. ‣ Appendix B Additional results ‣ What’s a good imputation to predict with missing values?") gives their expressions.
* •

  Chained oracles: f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI} consists in imputing by the conditional expectation and then applying f⋆superscript𝑓⋆f^{\star}. The analytical expression of ΦC​IsuperscriptΦ𝐶𝐼\Phi^{CI} can be derived analytically for both MCAR and GSM, and we thus use this analytical expression to impute the missing values.
* •

  Oracle + MLP: The data is imputed using the analytical expression of the conditional expectation, and then a MLP is fitted to the completed data.

These three predictors all use ground truth information (parameters μ𝜇\mu, ΣΣ\Sigma of the data distribution, expression of f⋆superscript𝑓⋆f^{\star} or of the missing data mechanism) which are unavailable in practice. They are mainly useful as reference points. We then compare the NeurMiss+MLP architecture and a number of classic Impute-then-Regress methods as well as gradient boosted regression trees:

* •

  Mean + MLP The data is imputed by the mean, and a multilayer perceptron (MLP) is fitted to the completed data.
* •

  MICE + MLP The data is imputed using Scikit-learn’s (Pedregosa et al., [2012](#bib.bib19), BSD licensed) conditional imputer IterativeImputer that adapts the popular Multivariate Imputation by Chained Equations (MICE, van Buuren, [2018](#bib.bib22)) to be able to impute a test set. A multilayer perceptron (MLP) is then fitted to the completed data.
* •

  GBRT: Gradient boosted regression trees (Scikit-learn’s HistGradientBoostingRegressor with default parameters). This predictor readily supports missing values: during training, missing values on the decision variable for a given split are sent to the left or right child depending on which provides the largest gain. This is know as the Missing Incorporated Attribute strategy (Twala et al., [2008](#bib.bib21)).

Finally, we also run Mean + mask + MLP as well as MICE + mask + MLP in which the mask is concatenated to the imputed data before fitting a MLP. Concatenating the mask is a widespread pratice to account for MNAR data.

All MLPs are implemented with PyTorch (Paszke et al., [2019](#bib.bib18)). A validation set is used to choose MLPs’ depth (1, 2 or 5), width (1​d1𝑑1d, 5​d5𝑑5d or 10​d10𝑑10d), initial learning rate (ranging from 5.10−4superscript5.1045.10^{-4} to 10−2superscript10210^{-2}) and weight decay (ranging from 10−6superscript10610^{-6} to 10−3superscript10310^{-3}). Adam is used with an adaptive learning rate: the learning rate is divided by 5 each time 10 consecutive epochs fail to decrease the training loss by at least 1e-4. Early stopping is triggered when the validation score does not improve by at least 1e-4 for 12 consecutive epochs. The batch size is set to 100, and ReLUs are used as activation functions. Finally for NeuMiss the depth is set to 20. Note that since the weights of NeuMiss are shared, increasing its depth does not increase its number of parameters. For gradient boosted regression trees, several hyperparameters are chosen using the validation set including the maximum number of leaves for each tree (from 50 to 600), the maximum number of iterations for the boosting process (from 100 to 300), as well as the minimum number of samples per leaf (from 10 to 50).

The experiments use training sets of size n=100 000𝑛100000n=100\,000 and validation and test sets of size n=10 000𝑛10000n=10\,000. The code for all experiments is available at <https://github.com/marineLM/Impute_then_Regress>.

### 6.2 Experimental results

The results are presented in Figure [4](#S6.F4 "Figure 4 ‣ 6.2 Experimental results ‣ 6 Empirical study of impute-n-regress procedures ‣ What’s a good imputation to predict with missing values?") as well as in Figure [8](#A2.F8 "Figure 8 ‣ B.3 Supplementary experiments with 𝑓^∗_{𝑏⁢𝑟⁢𝑒⁢𝑎⁢𝑘}. ‣ Appendix B Additional results ‣ What’s a good imputation to predict with missing values?") (Appendix [B.3](#A2.SS3 "B.3 Supplementary experiments with 𝑓^∗_{𝑏⁢𝑟⁢𝑒⁢𝑎⁢𝑘}. ‣ Appendix B Additional results ‣ What’s a good imputation to predict with missing values?")).

![Refer to caption](/html/2106.00311/assets/x6.png)


Figure 4: Performances (R2 score on a test set) compared to that of the Bayes predictor across 10 repeated experiments.

#### Chaining oracles fails when both curvature is high and correlation is low.

The chained oracle has a performance close to that of the Bayes predictor in all cases except when the wave or break functions are applied to low correlation data. This observation illustrates well Proposition [4.1](#S4.Thmproposition1 "Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?"). Intuitively, the Bayes predictor for each missing data pattern is a smoothed version of f⋆superscript𝑓⋆f^{\star}, and it is all the more smoothed that there is uncertainty around the likely values of the missing data. In the low correlation setting, the uncertainty is such that f⋆superscript𝑓⋆f^{\star} is not a good proxy anymore for the Bayes predictor.

#### Regressing on oracle conditional imputation provide excellent performances.

Contrary to the chained oracles, *Oracle + MLP* is close to the Bayes rate in all cases. This result should be put into perspective with Proposition [4.2](#S4.Thmproposition2 "Proposition 4.2 (Regression function discontinuities). ‣ 4.2 Regressing on conditional imputations, a good idea? ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?"), which states that there is no *continuous* regression function g𝑔g such that g∘ΦC​I𝑔superscriptΦ𝐶𝐼g\circ\Phi^{CI} is Bayes optimal unless it is f⋆superscript𝑓⋆f^{\star}. Indeed, as the MLP can only learn continuous functions, it shows that there are continuous functions g𝑔g such that g∘ΦC​I𝑔superscriptΦ𝐶𝐼g\circ\Phi^{CI}, even though it is not Bayes optimal, performs very well.

#### Adding the mask is critical in MNAR settings with *mean* and *MICE* imputations

In MNAR settings, missingness carries information that can be useful for prediction. However, both the mean and iterative conditional imputation output an imputed dataset in which the missingness information is more difficult to retrieve. For this reason, it is common practice to concatenate the mask with the imputed data to expose the missingness information to the predictor. Our experiments show that under self-masking (MNAR), adding the mask to the mean or iteratively imputed data markedly improves performances. Note that NeuMiss does not require adding the mask as an input since the missingness information is already incorporated via the non-linearities.

#### NeuMiss+MLP performs best among Impute-then-Regress predictors.

In *all*
settings, NeuMiss performs best. GBRT performs poorly here possibly because they are not well adapted to approximate smooth functions. Finally, note that when the difficulty of the problem increases, for example with a lower correlation, then (i) the performance of the Bayes predictor decreases and (ii) the differences in performance among methods is reduced, as in the lower right panel.

## 7 Conclusion

Impute-then-regress procedures assemble standard statistical routines to build predictors suited for data with missing values. However, we have shown that seeking the best prediction of the outcome leads to different tradeoffs compared to inferential purposes. Given a powerful learner, *almost all imputations* lead asymptotically to the optimal prediction, *whatever the missingness mechanism*. A good choice of imputation can however reduce the complexity of the function to learn.
Though conditional expectation can lead to discontinuous optimal regression functions, our experiments show that it still leads to easier learning problems compared to simpler imputations.
In order to adapt the imputation to the regression function, we proposed to jointly learn these two steps by chaining a trainable imputation via the NeuMiss networks and a classical MLP. An empirical study of non-linear regression shows that it outperforms impute-then-regress procedures built on standard imputation methods as well as gradient-boosted trees with incorporated handling of missing values. In further work, it would be useful to theoretically characterize the learning behaviors of Impute-then-Regress methods in finite sample regimes.

## Acknowledgments and Disclosure of Funding

MLM, JJ, and GV acknowledge funding via DataIA MissingBigData. MlM and GV acknowledge funding via ANR-17-CE23-0018 DirtyData and GV acknowledges funding via ANR-20-CHIA-0026 LearnI.
JJ acknowledges funding via ANR-16-IDEX-0006.

## References

* Arutyunov and Zhukovskiy [2019]

  AV Arutyunov and SE Zhukovskiy.
  Application of methods of ordinary differential equations to global
  inverse function theorems.
  *Differential Equations*, 55(4):437–448,
  2019.
* Bertsimas et al. [2021]

  Dimitris Bertsimas, Arthur Delarue, and Jean Pauphilet.
  Prediction with Missing Data, 2021.
* Bishop [2006]

  Christopher M. Bishop.
  *Pattern Recognition and Machine Learning (Information Science
  and Statistics)*.
  Springer-Verlag, Berlin, Heidelberg, 2006.
  ISBN 0387310738.
* [4]

  Jan Borowski and Piotr Fic.
  NADIA r package.
  <https://cran.r-project.org/web/packages/NADIA/index.html>.
  Accessed: 2021-05-26.
* Chen and Guestrin [2016]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *Proceedings of the 22nd acm sigkdd international conference
  on knowledge discovery and data mining*, pages 785–794, 2016.
* Dempster et al. [1977]

  Arthur P Dempster, Nan M Laird, and Donald B Rubin.
  Maximum likelihood from incomplete data via the em algorithm.
  *Journal of the Royal Statistical Society: Series B
  (Methodological)*, 39:1, 1977.
* Folland [2002]

  G B Folland.
  *Advanced Calculus*.
  Featured Titles for Advanced Calculus Series. Prentice Hall, 2002.
  ISBN 9780130652652.
* Golubitsky [1973]

  Martin Golubitsky.
  *Stable mappings and their singularities*.
  Graduate texts in mathematics. Springer-Verlag, 1973.
  ISBN 0-387-90072-1.
* Guillemin and Pollack [1974]

  Victor W. Guillemin and Alan Pollack.
  *Differential topology*.
  Prentice-Hall Englewood Cliffs, N.J, 1974.
  ISBN 0132126052.
* Györfi et al. [2006]

  László Györfi, Michael Kohler, Adam Krzyzak, and Harro Walk.
  *A distribution-free theory of nonparametric regression*.
  Springer Science & Business Media, 2006.
* Ipsen et al. [2020]

  Niels Ipsen, Pierre-Alexandre Mattei, and Jes Frellsen.
  How to deal with missing data in supervised deep learning?
  In *ICML Workshop on the Art of Learning with Missing Values
  (Artemiss)*, 2020.
* Jarrett et al. [2021]

  Daniel Jarrett, Jinsung Yoon, Ioana Bica, Zhaozhi Qian, Ari Ercole, and Mihaela
  van der Schaar.
  Clairvoyance: A pipeline toolkit for medical time series.
  In *International Conference on Learning Representations*, 2021.
* Josse et al. [2019]

  Julie Josse, Nicolas Prost, Erwan Scornet, and Gaël Varoquaux.
  On the consistency of supervised learning with missing values.
  *arXiv preprint arXiv:1902.06931*, 2019.
* Le Morvan et al. [2020a]

  Marine Le Morvan, Julie Josse, Thomas Moreau, Erwan Scornet, and Gaël
  Varoquaux.
  NeuMiss networks: differentiable programming for supervised learning
  with missing values.
  *Advances in Neural Information Processing Systems*, 33,
  2020a.
* Le Morvan et al. [2020b]

  Marine Le Morvan, Nicolas Prost, Julie Josse, Erwan Scornet, and Gael
  Varoquaux.
  Linear predictor on linearly-generated data with missing values: non
  consistency and solutions.
  In Silvia Chiappa and Roberto Calandra, editors, *Proceedings of
  the Twenty Third International Conference on Artificial Intelligence and
  Statistics*, volume 108 of *Proceedings of Machine Learning Research*,
  pages 3165–3174. PMLR, 2020b.
* Little and Rubin [1987, 2002, 2019]

  Roderick JA Little and Donald B Rubin.
  *Statistical analysis with missing data*.
  John Wiley & Sons, 1987, 2002, 2019.
* Mattei and Frellsen [2019]

  Pierre-Alexandre Mattei and Jes Frellsen.
  MIWAE: Deep generative modelling and imputation of incomplete data
  sets.
  In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
  *Proceedings of the 36th International Conference on Machine Learning*,
  volume 97 of *Proceedings of Machine Learning Research*, pages
  4413–4423. PMLR, 09–15 Jun 2019.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
  Pytorch: An imperative style, high-performance deep learning library.
  *Advances in Neural Information Processing Systems*,
  32:8026–8037, 2019.
* Pedregosa et al. [2012]

  Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
  Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron
  Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau,
  Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay.
  Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research*, 12(oct):2825–2830, 2012.
* Rubin [1976]

  Donald B Rubin.
  Inference and missing data.
  *Biometrika*, 63(3):581–592, 1976.
* Twala et al. [2008]

  B. E. T. H. Twala, M. C. Jones, and D. J. Hand.
  Good methods for coping with missing data in decision trees.
  *Pattern Recogn. Lett.*, 29:950–956, 2008.
  ISSN 0167-8655.
* van Buuren [2018]

  Stef van Buuren.
  *Flexible Imputation of Missing Data, Second Edition*.
  2018.
* Yoon et al. [2018]

  Jinsung Yoon, James Jordon, and Mihaela Schaar.
  Gain: Missing data imputation using generative adversarial nets.
  In *International Conference on Machine Learning*, page 5689.
  PMLR, 2018.

Supplementary materials – What’s a good imputation to predict with missing values?

## Appendix A Proofs

### A.1 Proof of Lemma [A.1](#A1.Thmlemma1 "Lemma A.1. ‣ A.1 Proof of Lemma A.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")

###### Lemma A.1.

Let ϕ(m)∈𝒞∞​(ℝ|o​b​s​(m)|,ℝ|m​i​s​(m)|)superscriptitalic-ϕ𝑚superscript𝒞superscriptℝ𝑜𝑏𝑠𝑚superscriptℝ𝑚𝑖𝑠𝑚\phi^{(m)}\in\mathcal{C}^{\infty}\left(\mathbb{R}^{|obs(m)|},\mathbb{R}^{|mis(m)|}\right) be the imputation function for missing data pattern m𝑚m, and let ℳ(m)={x∈ℝd:xm​i​s=ϕ(m)​(xo​b​s)}superscriptℳ𝑚conditional-set𝑥superscriptℝ𝑑subscript𝑥𝑚𝑖𝑠superscriptitalic-ϕ𝑚subscript𝑥𝑜𝑏𝑠\mathcal{M}^{(m)}=\left\{x\in\mathbb{R}^{d}:x\_{mis}=\phi^{(m)}(x\_{obs})\right\}. For all m𝑚m, ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} is an |o​b​s|−limit-from𝑜𝑏𝑠|obs|-dimensional manifold.

###### Proof.

Let:

|  |  |  |  |
| --- | --- | --- | --- |
|  | h(m):ℝd:superscriptℎ𝑚superscriptℝ𝑑\displaystyle h^{(m)}:\mathbb{R}^{d} | →ℝ|m​i​s|→absentsuperscriptℝ𝑚𝑖𝑠\displaystyle\to\mathbb{R}^{|mis|} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | x𝑥\displaystyle x | ↦xm​i​s−ϕ(m)​(xo​b​s)maps-toabsentsubscript𝑥𝑚𝑖𝑠superscriptitalic-ϕ𝑚subscript𝑥𝑜𝑏𝑠\displaystyle\mapsto x\_{mis}-\phi^{(m)}(x\_{obs}) |  |

Regular value: We will show that 𝟎m​i​ssubscript0𝑚𝑖𝑠\bm{0}\_{mis} is a regular value of h(m)superscriptℎ𝑚h^{(m)}. By definition [see p21 in Guillemin and Pollack, [1974](#bib.bib9)], a point y∈ℝ|m​i​s|𝑦superscriptℝ𝑚𝑖𝑠y\in\mathbb{R}^{|mis|} is a regular value of h(m)superscriptℎ𝑚h^{(m)} if d​hx(m)𝑑subscriptsuperscriptℎ𝑚𝑥dh^{(m)}\_{x} is surjective at every point x𝑥x such that h(m)​(x)=ysuperscriptℎ𝑚𝑥𝑦h^{(m)}(x)=y. The mapping d​hx(m)𝑑subscriptsuperscriptℎ𝑚𝑥dh^{(m)}\_{x} is linear and can be represented by the Jacobian of h(m)superscriptℎ𝑚h^{(m)} at x𝑥x:

|  |  |  |
| --- | --- | --- |
|  | Jh(m)​(x)=(AI​d),A∈ℝ|m​i​s|×|o​b​s|,I​d∈ℝ|m​i​s|×|m​i​s|.formulae-sequencesubscript𝐽superscriptℎ𝑚𝑥matrixmissing-subexpression𝐴missing-subexpression𝐼𝑑missing-subexpressionformulae-sequence𝐴superscriptℝ𝑚𝑖𝑠𝑜𝑏𝑠𝐼𝑑superscriptℝ𝑚𝑖𝑠𝑚𝑖𝑠J\_{h^{(m)}}(x)=\left(\begin{matrix}&\vrule width=0.0pt,height=17.0pt,depth=15.0ptA&\vline&Id&\end{matrix}\right),\quad A\in\mathbb{R}^{|mis|\times|obs|},\,Id\in\mathbb{R}^{|mis|\times|mis|}. |  |

Given the structure of Jh(m)​(x)subscript𝐽superscriptℎ𝑚𝑥J\_{h^{(m)}}(x), it is obviously of rank |m​i​s|𝑚𝑖𝑠|mis| at every point x𝑥x. Thus d​hx(m)𝑑subscriptsuperscriptℎ𝑚𝑥dh^{(m)}\_{x} is surjective at every point x𝑥x, and it is true in particular for the points x𝑥x such that h(m)​(x)=𝟎superscriptℎ𝑚𝑥0h^{(m)}(x)=\bm{0}. We conclude that by definition, 𝟎m​i​ssubscript0𝑚𝑖𝑠\bm{0}\_{mis} is a regular value of h(m)superscriptℎ𝑚h^{(m)}.

Preimage theorem: By the Preimage theorem ([Guillemin and Pollack, [1974](#bib.bib9)], p.21), since 𝟎∈ℝm​i​s0subscriptℝ𝑚𝑖𝑠\bm{0}\in\mathbb{R}\_{mis} is a regular value of h(m):ℝd→ℝ|m​i​s|:superscriptℎ𝑚→superscriptℝ𝑑superscriptℝ𝑚𝑖𝑠h^{(m)}:\mathbb{R}^{d}\to\mathbb{R}^{|mis|}, then the the preimage (h(m))−1​(𝟎)superscriptsuperscriptℎ𝑚10\left(h^{(m)}\right)^{-1}(\bm{0}) is a submanifold of ℝdsuperscriptℝ𝑑\mathbb{R}^{d} of dimension d−|m​i​s|=|o​b​s|𝑑𝑚𝑖𝑠𝑜𝑏𝑠d-|mis|=|obs|.

Since by definition, (h(m))−1​(𝟎)=ℳ(m)superscriptsuperscriptℎ𝑚10superscriptℳ𝑚\left(h^{(m)}\right)^{-1}(\bm{0})=\mathcal{M}^{(m)}, we have that ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} is a |o​b​s|−limit-from𝑜𝑏𝑠|obs|-dimensional mainfold.
∎

### A.2 Proof of Lemma [A.2](#A1.Thmlemma2 "Lemma A.2. ‣ A.2 Proof of Lemma A.2 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")

###### Lemma A.2.

Let m𝑚m and m′superscript𝑚′m^{\prime} be two distinct missing data patterns with the same number of missing values |m​i​s|𝑚𝑖𝑠|mis|. Let ϕ(m)∈𝒞∞​(ℝ|o​b​s​(m)|,ℝ|m​i​s​(m)|)superscriptitalic-ϕ𝑚superscript𝒞superscriptℝ𝑜𝑏𝑠𝑚superscriptℝ𝑚𝑖𝑠𝑚\phi^{(m)}\in\mathcal{C}^{\infty}\left(\mathbb{R}^{|obs(m)|},\mathbb{R}^{|mis(m)|}\right) be the imputation function for missing data pattern m𝑚m, and let ℳ(m)={x∈ℝd:xm​i​s=ϕ(m)​(xo​b​s)}superscriptℳ𝑚conditional-set𝑥superscriptℝ𝑑subscript𝑥𝑚𝑖𝑠superscriptitalic-ϕ𝑚subscript𝑥𝑜𝑏𝑠\mathcal{M}^{(m)}=\left\{x\in\mathbb{R}^{d}:x\_{mis}=\phi^{(m)}(x\_{obs})\right\}. We define similarly ϕ(m′)superscriptitalic-ϕsuperscript𝑚′\phi^{(m^{\prime})} and ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})}.
For almost all imputation functions ϕ(m)superscriptitalic-ϕ𝑚\phi^{(m)} and ϕ(m′)superscriptitalic-ϕsuperscript𝑚′\phi^{(m^{\prime})},

|  |  |  |  |
| --- | --- | --- | --- |
|  | d​i​m​(ℳ(m)∩ℳ(m′))={0i​f​|m​i​s|>d2d−2​|m​i​s|o​t​h​e​r​w​i​s​e.𝑑𝑖𝑚superscriptℳ𝑚superscriptℳsuperscript𝑚′cases0𝑖𝑓𝑚𝑖𝑠𝑑2𝑑2𝑚𝑖𝑠𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒dim\left(\mathcal{M}^{(m)}\cap\mathcal{M}^{(m^{\prime})}\right)=\begin{cases}0\quad&if\;|mis|>\frac{d}{2}\\ d-2|mis|\quad&otherwise.\end{cases} |  | (9) |

###### Proof.

According to Thom Transversality theorem ([Golubitsky, [1973](#bib.bib8)], p.54) with:

* •

  W=ℳ(m′)𝑊superscriptℳsuperscript𝑚′W=\mathcal{M}^{(m^{\prime})},
* •

  f=ϕ(m)𝑓superscriptitalic-ϕ𝑚f=\phi^{(m)},
* •

  k=0𝑘0k=0 (note that as stated p.37, J0​(X,Y)=X×Ysuperscript𝐽0𝑋𝑌𝑋𝑌J^{0}(X,Y)=X\times Y and j0​f​(x)=graph​(f)superscript𝑗0𝑓𝑥graph𝑓j^{0}f(x)=\text{graph}(f)),

we have that {ϕ(m)∈𝒞∞​(ℝ|o​b​s|,ℝ|m​i​s|)|graph​(ϕ(m))⋔ℳ(m′)}conditional-setsuperscriptitalic-ϕ𝑚superscript𝒞superscriptℝ𝑜𝑏𝑠superscriptℝ𝑚𝑖𝑠proper-intersectiongraphsuperscriptitalic-ϕ𝑚superscriptℳsuperscript𝑚′\left\{\phi^{(m)}\in\mathcal{C}^{\infty}(\mathbb{R}^{|obs|},\mathbb{R}^{|mis|})\,|\,\text{graph}(\phi^{(m)})\pitchfork\mathcal{M}^{(m^{\prime})}\right\} is a residual subset of 𝒞∞​(ℝ|o​b​s|,ℝ|m​i​s|)superscript𝒞superscriptℝ𝑜𝑏𝑠superscriptℝ𝑚𝑖𝑠\mathcal{C}^{\infty}(\mathbb{R}^{|obs|},\mathbb{R}^{|mis|}) in the 𝒞∞superscript𝒞\mathcal{C}^{\infty} topology. In other words, the fact that graph​(ϕ(m))graphsuperscriptitalic-ϕ𝑚\text{graph}(\phi^{(m)}) is transverse to ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})} is a generic property. Put differently, almost all functions ϕ(m)superscriptitalic-ϕ𝑚\phi^{(m)} have their graph transverse to ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})}. Note that here the notion of *almost all* has to be understood in its topological sense, and not in its measure theory sense.

Suppose that |o​b​s|<d2𝑜𝑏𝑠𝑑2|obs|<\frac{d}{2}. According to Lemma [A.1](#A1.Thmlemma1 "Lemma A.1. ‣ A.1 Proof of Lemma A.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?"), ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})} is a |o​b​s|−limit-from𝑜𝑏𝑠|obs|-dimensional manifold. Moreover we just showed that for almost all ϕ(m)superscriptitalic-ϕ𝑚\phi^{(m)}, graph​(ϕ(m))⋔ℳ(m′)proper-intersectiongraphsuperscriptitalic-ϕ𝑚superscriptℳsuperscript𝑚′\text{graph}(\phi^{(m)})\pitchfork\mathcal{M}^{(m^{\prime})}. Applying Proposition 4.2 of [Golubitsky, [1973](#bib.bib8)] (p.51) with W=ℳ(m′)𝑊superscriptℳsuperscript𝑚′W=\mathcal{M}^{(m^{\prime})} and f=graph​(ϕ(m))𝑓graphsuperscriptitalic-ϕ𝑚f=\text{graph}(\phi^{(m)}), we obtain that ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} and ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})} are disjoint, since, by definition, ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} is the image of graph​(ϕ(m))graphsuperscriptitalic-ϕ𝑚\text{graph}(\phi^{(m)}). Consequently, the dimension of their intersection is 0.

Suppose that |o​b​s|≥d2𝑜𝑏𝑠𝑑2|obs|\geq\frac{d}{2}. According to the theorem p.30 of [Guillemin and Pollack, [1974](#bib.bib9)], since ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} and ℳ(m′)superscriptℳsuperscript𝑚′\mathcal{M}^{(m^{\prime})} are transverse submanifolds of ℝdsuperscriptℝ𝑑\mathbb{R}^{d}, their intersection is again a manifold with codim​(ℳ(m)∩ℳ(m′))=codim​(ℳ(m))+codim​(ℳ(m′))codimsuperscriptℳ𝑚superscriptℳsuperscript𝑚′codimsuperscriptℳ𝑚codimsuperscriptℳsuperscript𝑚′\text{codim}(\mathcal{M}^{(m)}\cap\mathcal{M}^{(m^{\prime})})=\text{codim}(\mathcal{M}^{(m)})+\text{codim}(\mathcal{M}^{(m^{\prime})}). This implies that dim​(ℳ(m)∩ℳ(m′))=2​|o​b​s|−ddimsuperscriptℳ𝑚superscriptℳsuperscript𝑚′2𝑜𝑏𝑠𝑑\text{dim}(\mathcal{M}^{(m)}\cap\mathcal{M}^{(m^{\prime})})=2|obs|-d.
∎

### A.3 Proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?")

See [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?")

###### Proof.

Let ϕ(m)∈𝒞∞​(ℝ|o​b​s​(m)|,ℝ|m​i​s​(m)|)superscriptitalic-ϕ𝑚superscript𝒞superscriptℝ𝑜𝑏𝑠𝑚superscriptℝ𝑚𝑖𝑠𝑚\phi^{(m)}\in\mathcal{C}^{\infty}\left(\mathbb{R}^{|obs(m)|},\mathbb{R}^{|mis(m)|}\right) be the imputation function for missing data pattern m𝑚m, and let ℳ(m)={x∈ℝd:xm​i​s=ϕ(m)​(xo​b​s)}superscriptℳ𝑚conditional-set𝑥superscriptℝ𝑑subscript𝑥𝑚𝑖𝑠superscriptitalic-ϕ𝑚subscript𝑥𝑜𝑏𝑠\mathcal{M}^{(m)}=\left\{x\in\mathbb{R}^{d}:x\_{mis}=\phi^{(m)}(x\_{obs})\right\}. According to Lemma [A.1](#A1.Thmlemma1 "Lemma A.1. ‣ A.1 Proof of Lemma A.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?"), for all m𝑚m, ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} is an |o​b​s|−limit-from𝑜𝑏𝑠|obs|-dimensional manifold. ℳ(m)superscriptℳ𝑚\mathcal{M}^{(m)} corresponds to the subspace where all points with missing data pattern m𝑚m are mapped after imputation.

Let us order missing data patterns according to their number of missing values, with the pattern of all missing entries ranked first and the pattern of all observed entries ranked last. Two patterns with the same number of missing values are ordered arbitrarily. We use m​(i)𝑚𝑖m(i) to refer to the missing data pattern ranked in it​hsuperscript𝑖𝑡ℎi^{th} position.

Let g⋆superscript𝑔⋆g^{\star} be the function defined as follows: for all i𝑖i,

|  |  |  |
| --- | --- | --- |
|  | ∀Z=Φ​(X~)∈ℳ(m​(i))∖⋃m​(k)<m​(i)ℳ(m​(k)),g⋆​(Z)=f~⋆​(X~).formulae-sequencefor-all𝑍Φ~𝑋superscriptℳ𝑚𝑖subscript𝑚𝑘𝑚𝑖superscriptℳ𝑚𝑘superscript𝑔⋆𝑍superscript~𝑓⋆~𝑋\forall Z=\Phi(\widetilde{X})\in\mathcal{M}^{(m(i))}\setminus\bigcup\limits\_{m(k)<m(i)}\mathcal{M}^{(m(k))},\qquad g^{\star}(Z)=\tilde{f}^{\star}(\widetilde{X}). |  |

For a given missing data pattern m​(i)𝑚𝑖m(i), by distributivity of intersections across unions, we have:

|  |  |  |
| --- | --- | --- |
|  | ℳ(m​(i))​⋂(⋃m​(k)<m​(i)ℳ(m​(k)))=⋃m​(k)<m​(i)​(ℳ(m​(i))​⋂ℳ(m​(k)))superscriptℳ𝑚𝑖subscript𝑚𝑘𝑚𝑖superscriptℳ𝑚𝑘𝑚𝑘𝑚𝑖superscriptℳ𝑚𝑖superscriptℳ𝑚𝑘\mathcal{M}^{(m(i))}\bigcap\left(\bigcup\limits\_{m(k)<m(i)}\mathcal{M}^{(m(k))}\right)=\underset{m(k)<m(i)}{\bigcup}\left(\mathcal{M}^{(m(i))}\bigcap\mathcal{M}^{(m(k))}\right) |  |

If m​(k)𝑚𝑘m(k) has strictly more missing values than m​(i)𝑚𝑖m(i), then by Lemma [A.1](#A1.Thmlemma1 "Lemma A.1. ‣ A.1 Proof of Lemma A.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") dim​(ℳ(m​(k)))<dim​(ℳ(m​(i)))dimsuperscriptℳ𝑚𝑘dimsuperscriptℳ𝑚𝑖\text{dim}(\mathcal{M}^{(m(k))})<\text{dim}(\mathcal{M}^{(m(i))}), and thus dim​(ℳ(m​(k))∩ℳ(m​(i)))<dim​(ℳ(m​(i)))dimsuperscriptℳ𝑚𝑘superscriptℳ𝑚𝑖dimsuperscriptℳ𝑚𝑖\text{dim}(\mathcal{M}^{(m(k))}\cap\mathcal{M}^{(m(i))})<\text{dim}(\mathcal{M}^{(m(i))}). Moreover, If m​(k)𝑚𝑘m(k) has the same number of missing values as m​(i)𝑚𝑖m(i), then by Lemma [A.2](#A1.Thmlemma2 "Lemma A.2. ‣ A.2 Proof of Lemma A.2 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?"), for almost all imputation functions ϕ(m​(k))superscriptitalic-ϕ𝑚𝑘\phi^{(m(k))} and ϕ(m​(i))superscriptitalic-ϕ𝑚𝑖\phi^{(m(i))}, dim​(ℳ(m​(k))∩ℳ(m​(i)))<dim​(ℳ(m​(i)))dimsuperscriptℳ𝑚𝑘superscriptℳ𝑚𝑖dimsuperscriptℳ𝑚𝑖\text{dim}(\mathcal{M}^{(m(k))}\cap\mathcal{M}^{(m(i))})<\text{dim}(\mathcal{M}^{(m(i))}). We conclude that for all m​(k)<m​(i)𝑚𝑘𝑚𝑖m(k)<m(i), ℳ(m​(k))∩ℳ(m​(i))superscriptℳ𝑚𝑘superscriptℳ𝑚𝑖\mathcal{M}^{(m(k))}\cap\mathcal{M}^{(m(i))} is a subset of measure zero in ℳ(m​(i))superscriptℳ𝑚𝑖\mathcal{M}^{(m(i))}. Finally, since a countable union of sets of measure zero has measure zero, we obtain that ∪m​(k)<m​(i)​(ℳ(m​(i))∩ℳ(m​(k)))𝑚𝑘𝑚𝑖superscriptℳ𝑚𝑖superscriptℳ𝑚𝑘\underset{m(k)<m(i)}{\cup}\left(\mathcal{M}^{(m(i))}\cap\mathcal{M}^{(m(k))}\right) has measure zero in ℳ(m​(i))superscriptℳ𝑚𝑖\mathcal{M}^{(m(i))}.

Let’s now compute the risk of g⋆∘Φsuperscript𝑔⋆Φg^{\star}\circ\Phi:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℛ​(g⋆∘Φ)=∑M=mP​(M=m)​∫Xo​b​sP​(Xo​b​s|M=m)​(f~⋆​(X~)−g⋆∘Φ​(X~))2ℛsuperscript𝑔⋆Φsubscript𝑀𝑚𝑃𝑀𝑚subscriptsubscript𝑋𝑜𝑏𝑠𝑃conditionalsubscript𝑋𝑜𝑏𝑠𝑀𝑚superscriptsuperscript~𝑓⋆~𝑋superscript𝑔⋆Φ~𝑋2\mathcal{R}(g^{\star}\circ\Phi)=\sum\_{M=m}P(M=m)\int\_{X\_{obs}}P(X\_{obs}|M=m)\left(\tilde{f}^{\star}(\widetilde{X})-g^{\star}\circ\Phi(\widetilde{X})\right)^{2} |  | (10) |

For a given missing data pattern m𝑚m, Φ​(X~)∈ℳ(m)Φ~𝑋superscriptℳ𝑚\Phi(\widetilde{X})\in\mathcal{M}^{(m)}. Moreover, we constructed g⋆superscript𝑔⋆g^{\star} such that g⋆∘Φ​(X~)=f~⋆​(X~)superscript𝑔⋆Φ~𝑋superscript~𝑓⋆~𝑋g^{\star}\circ\Phi(\widetilde{X})=\tilde{f}^{\star}(\widetilde{X}) for all Φ​(X~)∈ℳ(m)Φ~𝑋superscriptℳ𝑚\Phi(\widetilde{X})\in\mathcal{M}^{(m)} except on a set that we just showed to be of measure zero for almost all imputation functions. As a result, the function Xo​b​s↦f~⋆​(X~)−g⋆∘Φ​(X~)maps-tosubscript𝑋𝑜𝑏𝑠superscript~𝑓⋆~𝑋superscript𝑔⋆Φ~𝑋X\_{obs}\mapsto\tilde{f}^{\star}(\widetilde{X})-g^{\star}\circ\Phi(\widetilde{X}) is zero almost everywhere for a given m𝑚m, and the function Xo​b​s↦P​(Xo​b​s|M=m)​(f~⋆​(X~)−g⋆∘Φ​(X~))2maps-tosubscript𝑋𝑜𝑏𝑠𝑃conditionalsubscript𝑋𝑜𝑏𝑠𝑀𝑚superscriptsuperscript~𝑓⋆~𝑋superscript𝑔⋆Φ~𝑋2X\_{obs}\mapsto P(X\_{obs}|M=m)\left(\tilde{f}^{\star}(\widetilde{X})-g^{\star}\circ\Phi(\widetilde{X})\right)^{2} is also zero almost everywhere. Since the integral of a function that vanishes almost everywhere is equal to 0, we conclude that ℛ​(g⋆∘Φ)=0ℛsuperscript𝑔⋆Φ0\mathcal{R}(g^{\star}\circ\Phi)=0. Since the risk cannot be negative, g⋆∘Φsuperscript𝑔⋆Φg^{\star}\circ\Phi is a minimizer of the risk and thus it is Bayes optimal.
∎

### A.4 Examples of transverse and nontransverse manifolds in 2D.

Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") is true for *almost all* imputation functions and not *all* of them. Thus, we can construct examples with particular choices of imputation functions that lead to nontransverse manifolds, and consequently for which Impute-then-Regress procedures are not Bayes optimal. We provide such an example below.

Consider a dataset with points x∈ℝ2𝑥superscriptℝ2x\in\mathbb{R}^{2}, and let a∈ℝ𝑎ℝa\in\mathbb{R}. Let Φ2(0,1)​(x1)=a∗x1subscriptsuperscriptΦ012subscript𝑥1𝑎subscript𝑥1\Phi^{(0,1)}\_{2}(x\_{1})=a\*x\_{1} be the imputation function for x2subscript𝑥2x\_{2} when only x1subscript𝑥1x\_{1} is observed. And let Φ1(1,0)​(x2)=1a​x2subscriptsuperscriptΦ101subscript𝑥21𝑎subscript𝑥2\Phi^{(1,0)}\_{1}(x\_{2})=\frac{1}{a}x\_{2} be the imputation function for x1subscript𝑥1x\_{1} when only x2subscript𝑥2x\_{2} is observed. In this particular case shown in Figure [5](#A1.F5.10 "Figure 5 ‣ A.4 Examples of transverse and nontransverse manifolds in 2D. ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") (bottom), the manifolds on which the data with either x1subscript𝑥1x\_{1} missing or x2subscript𝑥2x\_{2} missing are projected are exactly the same (the same line in the 2D space). Thus they are nontransverse and consequently Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Bayes consistency of Impute-then-regress procedures). ‣ 3.2 Impute-then-regress procedures are Bayes optimal ‣ 3 Asymptotic analysis of Impute-then-regress procedures ‣ What’s a good imputation to predict with missing values?") does not hold.

However according to the Thom transversality theorem, almost all imputation functions will lead to transverse manifolds.

ℳ(0,1)superscriptℳ01\mathcal{M}^{(0,1)}ℳ(1,0)superscriptℳ10\mathcal{M}^{(1,0)}ℳ(1,1)superscriptℳ11\mathcal{M}^{(1,1)}ℳ(0,0)superscriptℳ00\mathcal{M}^{(0,0)}X1subscript𝑋1X\_{1}X2subscript𝑋2X\_{2}

ℳ(0,1)superscriptℳ01\mathcal{M}^{(0,1)}ℳ(1,0)superscriptℳ10\mathcal{M}^{(1,0)}ℳ(1,1)superscriptℳ11\mathcal{M}^{(1,1)}ℳ(0,0)superscriptℳ00\mathcal{M}^{(0,0)}X1subscript𝑋1X\_{1}X2subscript𝑋2X\_{2}

Figure 5: Example - Linear imputation manifolds in two dimensions Manifolds represented for linear imputation functions. ℳ(0,0)superscriptℳ00\mathcal{M}^{(0,0)} is the whole plane. Note that ℳ(1,1)superscriptℳ11\mathcal{M}^{(1,1)} need not be at the intersection of the two lines, it depends on the imputation function chosen. With linear imputation functions, ℳ(0,1)superscriptℳ01\mathcal{M}^{(0,1)} and ℳ(1,0)superscriptℳ10\mathcal{M}^{(1,0)} are transverse if and only if the two lines are not coincident.Top: Transverse manifolds. Bottom: Nontransverse manifolds.

### A.5 Proof of Lemma [A.3](#A1.Thmlemma3 "Lemma A.3. ‣ A.5 Proof of Lemma A.3 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")

###### Lemma A.3.

|  |  |  |
| --- | --- | --- |
|  | ∀X∈ℝp,∀m​i​s⊆⟦1,p⟧,H​(X)≼H¯+⟹Hm​i​s,m​i​s​(X)≼H¯m​i​s,m​i​s+formulae-sequencefor-all𝑋superscriptℝ𝑝formulae-sequencefor-all𝑚𝑖𝑠  1𝑝precedes-or-equals𝐻𝑋superscript¯𝐻subscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠𝑋precedes-or-equalssubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠\forall X\in\mathbb{R}^{p},\,\forall mis\subseteq\left\llbracket 1,p\right\rrbracket,\,H(X)\preccurlyeq\bar{H}^{+}\implies H\_{mis,mis}(X)\preccurlyeq\bar{H}^{+}\_{mis,mis} |  |

###### Proof.

Let X∈ℝp𝑋superscriptℝ𝑝X\in\mathbb{R}^{p}, and let m𝑚m be a missing data pattern with observed (resp. missing) indices o​b​s𝑜𝑏𝑠obs (resp. m​i​s𝑚𝑖𝑠mis).
H​(X)≼H¯+precedes-or-equals𝐻𝑋superscript¯𝐻H(X)\preccurlyeq\bar{H}^{+} is equivalent to:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∀u∈ℝp,u⊤​(H¯+−H​(X))​u≥0.formulae-sequencefor-all𝑢superscriptℝ𝑝superscript𝑢topsuperscript¯𝐻𝐻𝑋𝑢0\forall u\in\mathbb{R}^{p},\,u^{\top}\left(\bar{H}^{+}-H(X)\right)u\geq 0. |  | (11) |

Let 𝒱⊆ℝp𝒱superscriptℝ𝑝\mathcal{V}\subseteq\mathbb{R}^{p} be a subspace such that for any v𝑣v in 𝒱𝒱\mathcal{V}, vo​b​s=0subscript𝑣𝑜𝑏𝑠0v\_{obs}=0. Since 𝒱⊆ℝp𝒱superscriptℝ𝑝\mathcal{V}\subseteq\mathbb{R}^{p}, ([11](#A1.E11 "In Proof. ‣ A.5 Proof of Lemma A.3 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) implies:

|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∀v∈𝒱,v⊤​(H¯+−H​(X))​v≥0formulae-sequencefor-all𝑣𝒱superscript𝑣topsuperscript¯𝐻𝐻𝑋𝑣0\displaystyle\forall v\in\mathcal{V},\,v^{\top}\left(\bar{H}^{+}-H(X)\right)v\geq 0 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔iff\displaystyle\iff | ∀vm​i​s∈ℝ|m​i​s|,vm​i​s⊤​(H¯m​i​s,m​i​s+−Hm​i​s,m​i​s​(X))​vm​i​s≥0formulae-sequencefor-allsubscript𝑣𝑚𝑖𝑠superscriptℝ𝑚𝑖𝑠superscriptsubscript𝑣𝑚𝑖𝑠topsuperscriptsubscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠𝑋subscript𝑣𝑚𝑖𝑠0\displaystyle\forall v\_{mis}\in\mathbb{R}^{\left|\,mis\,\right|},\,v\_{mis}^{\top}\left(\bar{H}\_{mis,mis}^{+}-H\_{mis,mis}(X)\right)v\_{mis}\geq 0 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ⇔iff\displaystyle\iff | Hm​i​s,m​i​s​(X)≼H¯m​i​s,m​i​s+precedes-or-equalssubscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠𝑋superscriptsubscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠\displaystyle H\_{mis,mis}(X)\preccurlyeq\bar{H}\_{mis,mis}^{+} |  |

∎

### A.6 Proof of Lemma [4.1](#S4.Thmlemma1 "Lemma 4.1 (First order approximation). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

See [4.1](#S4.Thmlemma1 "Lemma 4.1 (First order approximation). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

###### Proof.

Without loss of generality, suppose that we reorder variables such that we can write X=(Xo​b​s,Xm​i​s)𝑋subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠X=(X\_{obs},X\_{mis}). Consider the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | fm​i​s⋆:ℝ|m​i​s|:subscriptsuperscript𝑓⋆𝑚𝑖𝑠superscriptℝ𝑚𝑖𝑠\displaystyle f^{\star}\_{mis}:\mathbb{R}^{|mis|} | →ℝ→absentℝ\displaystyle\to\mathbb{R} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Xm​i​ssubscript𝑋𝑚𝑖𝑠\displaystyle X\_{mis} | ↦f⋆​(Xo​b​s,Xm​i​s)maps-toabsentsuperscript𝑓⋆subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠\displaystyle\mapsto f^{\star}(X\_{obs},X\_{mis}) |  |

Since f⋆∈𝒞2​(ℝd,ℝ)superscript𝑓⋆superscript𝒞2superscriptℝ𝑑ℝf^{\star}\in\mathcal{C}^{2}\left(\mathbb{R}^{d},\mathbb{R}\right), we have fm​i​s⋆∈𝒞2​(ℝ|m​i​s|,ℝ)subscriptsuperscript𝑓⋆𝑚𝑖𝑠superscript𝒞2superscriptℝ𝑚𝑖𝑠ℝf^{\star}\_{mis}\in\mathcal{C}^{2}\left(\mathbb{R}^{|mis|},\mathbb{R}\right). Therefore, we can write the first order Taylor expansion (see Theorem 2.68 in Folland [[2002](#bib.bib7)]) of fm​i​s⋆subscriptsuperscript𝑓⋆𝑚𝑖𝑠f^{\star}\_{mis} around E​[Xm​i​s|Xo​b​s,M]𝐸delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀E\left[X\_{mis}|X\_{obs},M\right]:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fm​i​s⋆​(Xm​i​s)=f⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])+∇fm​i​s⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])⊤​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])+R​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M]),subscriptsuperscript𝑓⋆𝑚𝑖𝑠subscript𝑋𝑚𝑖𝑠superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀∇subscriptsuperscript𝑓⋆𝑚𝑖𝑠superscriptsubscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀topsubscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀𝑅subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\begin{split}f^{\star}\_{mis}(X\_{mis})=&f^{\star}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right])\\ &+\nabla f^{\star}\_{mis}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right])^{\top}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)\\ &+R\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right),\end{split} | |  | (12) |

where R𝑅R is the Lagrange remainder satisfying

|  |  |  |
| --- | --- | --- |
|  | R​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])=𝑅subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀absent\displaystyle R\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)= |  |
|  |  |  |
| --- | --- | --- |
|  | 12​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])⊤​Hm​i​s,m​i​s​(c)​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M]),12superscriptsubscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀topsubscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠𝑐subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\hskip 40.00006pt\frac{1}{2}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)^{\top}H\_{mis,mis}(c)\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right), |  |

for some c𝑐c in the ball ℬ(𝔼[Xm​i​s|Xo​b​s,M],∥Xm​i​s−𝔼[Xm​i​s|Xo​b​s,M]∥2)\mathcal{B}\left(\mathbb{E}\left[X\_{mis}|X\_{obs},M\right],\left\|\,X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\,\right\|\_{2}\right). By assumption, for all X𝑋X, H​(X)≼H¯+precedes-or-equals𝐻𝑋superscript¯𝐻H(X)\preccurlyeq\bar{H}^{+}. Therefore, according to Lemma [A.3](#A1.Thmlemma3 "Lemma A.3. ‣ A.5 Proof of Lemma A.3 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?"), we have Hm​i​s,m​i​s​(X)≼H¯m​i​s,m​i​s+precedes-or-equalssubscript𝐻

𝑚𝑖𝑠𝑚𝑖𝑠𝑋subscriptsuperscript¯𝐻

𝑚𝑖𝑠𝑚𝑖𝑠H\_{mis,mis}(X)\preccurlyeq\bar{H}^{+}\_{mis,mis} for any missing data pattern, which leads to:

|  |  |  |
| --- | --- | --- |
|  | R​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])≤𝑅subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀absent\displaystyle R\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)\leq |  |
|  |  |  |
| --- | --- | --- |
|  | 12​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])⊤​H¯m​i​s,m​i​s+​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M]).12superscriptsubscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀topsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\hskip 60.00009pt\frac{1}{2}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)^{\top}\bar{H}^{+}\_{mis,mis}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right). |  |

Using equality ([12](#A1.E12 "In Proof. ‣ A.6 Proof of Lemma 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")), we get:

|  |  |  |
| --- | --- | --- |
|  | f⋆​(Xo​b​s,Xm​i​s)−f⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])superscript𝑓⋆subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle f^{\star}(X\_{obs},X\_{mis})-f^{\star}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]) |  |
|  |  |  |
| --- | --- | --- |
|  | −∇fm​i​s⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])⊤​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])∇subscriptsuperscript𝑓⋆𝑚𝑖𝑠superscriptsubscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀topsubscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\hskip 40.00006pt-\nabla f^{\star}\_{mis}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right])^{\top}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right) |  |
|  |  |  |
| --- | --- | --- |
|  | ≤12​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])⊤​H¯m​i​s,m​i​s+​(Xm​i​s−𝔼​[Xm​i​s|Xo​b​s,M])absent12superscriptsubscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀topsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscript𝑋𝑚𝑖𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\hskip 100.00015pt\leq\frac{1}{2}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right)^{\top}\bar{H}^{+}\_{mis,mis}\left(X\_{mis}-\mathbb{E}\left[X\_{mis}|X\_{obs},M\right]\right) |  |

Finally, taking the expectation with regards to P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M) on both sides, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[f⋆​(Xo​b​s,Xm​i​s)|Xo​b​s,M]−f⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])≤12​t​r​(Hm​i​s,m​i​s+⊤​Σm​i​s|o​b​s,M),𝔼delimited-[]conditionalsuperscript𝑓⋆subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀12𝑡𝑟superscriptsubscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠absenttopsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀\mathbb{E}\left[f^{\star}(X\_{obs},X\_{mis})|X\_{obs},M\right]-f^{\star}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right])\leq\frac{1}{2}tr(H\_{mis,mis}^{+\top}\Sigma\_{mis|obs,M}), |  | (13) |

where we have used the fact that, for any vector X∈ℝd𝑋superscriptℝ𝑑X\in\mathbb{R}^{d} and for any H∈Pd+𝐻superscriptsubscript𝑃𝑑H\in P\_{d}^{+},

|  |  |  |
| --- | --- | --- |
|  | X⊤​H​X=t​r​(X⊤​H​X)=t​r​(H​X​X⊤).superscript𝑋top𝐻𝑋𝑡𝑟superscript𝑋top𝐻𝑋𝑡𝑟𝐻𝑋superscript𝑋top\displaystyle X^{\top}HX=tr(X^{\top}HX)=tr(HXX^{\top}). |  |

Following a similar reasoning, we can show that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[f⋆​(Xo​b​s,Xm​i​s)|Xo​b​s,M]−f⋆​(Xo​b​s,𝔼​[Xm​i​s|Xo​b​s,M])≥12​t​r​(Hm​i​s,m​i​s−⊤​Σm​i​s|o​b​s,M)𝔼delimited-[]conditionalsuperscript𝑓⋆subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀superscript𝑓⋆subscript𝑋𝑜𝑏𝑠𝔼delimited-[]conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀12𝑡𝑟superscriptsubscript𝐻  𝑚𝑖𝑠𝑚𝑖𝑠absenttopsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀\mathbb{E}\left[f^{\star}(X\_{obs},X\_{mis})|X\_{obs},M\right]-f^{\star}(X\_{obs},\mathbb{E}\left[X\_{mis}|X\_{obs},M\right])\geq\frac{1}{2}tr(H\_{mis,mis}^{-\top}\Sigma\_{mis|obs,M}) |  | (14) |

Together, inequalities ([13](#A1.E13 "In Proof. ‣ A.6 Proof of Lemma 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) and ([14](#A1.E14 "In Proof. ‣ A.6 Proof of Lemma 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) conclude the proof.
∎

### A.7 Proof of Proposition [4.1](#S4.Thmproposition1 "Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

See [4.1](#S4.Thmproposition1 "Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

###### Proof.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Y−f⋆​(XC​I)=𝑌superscript𝑓⋆superscript𝑋𝐶𝐼absent\displaystyle Y-f^{\star}(X^{CI})= | (Y−f~⋆​(X~))+(f~⋆​(X~)−f⋆​(XC​I))𝑌superscript~𝑓⋆~𝑋superscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼\displaystyle(Y-\tilde{f}^{\star}(\widetilde{X}))+(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})) |  | (15) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (Y−f​(XC​I))2=superscript𝑌𝑓superscript𝑋𝐶𝐼2absent\displaystyle\left(Y-f(X^{CI})\right)^{2}= | (Y−f~⋆​(X~))2+(f~⋆​(X~)−f⋆​(XC​I))2superscript𝑌superscript~𝑓⋆~𝑋2superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼2\displaystyle(Y-\tilde{f}^{\star}(\widetilde{X}))^{2}+(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI}))^{2} |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +2(Y−f~⋆(X~))(f~⋆(X~)−f⋆(XC​I)\displaystyle+2(Y-\tilde{f}^{\star}(\widetilde{X}))(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI}) |  | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | =\displaystyle= | (Y−f~⋆​(X~))2+(f~⋆​(X~)−f⋆​(XC​I))2superscript𝑌superscript~𝑓⋆~𝑋2superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼2\displaystyle(Y-\tilde{f}^{\star}(\widetilde{X}))^{2}+(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI}))^{2} |  | (18) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +2​(f⋆​(X)−f~⋆​(X~))​(f~⋆​(X~)−f⋆​(XC​I))2superscript𝑓⋆𝑋superscript~𝑓⋆~𝑋superscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼\displaystyle+2(f^{\star}(X)-\tilde{f}^{\star}(\widetilde{X}))(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})) |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +2​ϵ​(f~⋆​(X~)−f⋆​(XC​I))2italic-ϵsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼\displaystyle+2\epsilon(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})) |  | (20) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[(Y−f⋆​(XC​I))2]=𝔼delimited-[]superscript𝑌superscript𝑓⋆superscript𝑋𝐶𝐼2absent\displaystyle\mathbb{E}\left[\left(Y-f^{\star}(X^{CI})\right)^{2}\right]= | ℛ⋆+𝔼​[(f~⋆​(X~)−f⋆​(XC​I))2]superscriptℛ⋆𝔼delimited-[]superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼2\displaystyle\mathcal{R}^{\star}+\mathbb{E}\left[\left(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\right)^{2}\right] |  | (21) |

where we used the definition of the Bayes rate. Moreover, term ([20](#A1.E20 "In Proof. ‣ A.7 Proof of Proposition 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) vanishes when taking the expectation w.r.t ϵitalic-ϵ\epsilon because 𝔼​[ϵ|Xo​b​s,M]=0𝔼delimited-[]conditionalitalic-ϵ

subscript𝑋𝑜𝑏𝑠𝑀0\mathbb{E}\left[\epsilon|X\_{obs},M\right]=0 and ϵitalic-ϵ\epsilon in uncorrelated with X𝑋X or M𝑀M, and term ([19](#A1.E19 "In Proof. ‣ A.7 Proof of Proposition 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) vanishes when taking the expectation w.r.t Xm​i​s|Xo​b​s,Mconditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀X\_{mis}|X\_{obs},M because by definition 𝔼Xm​i​s|Xo​b​s,M​[f⋆​(Xo​b​s,Xm​i​s)]=f~⋆​(X~)subscript𝔼conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀delimited-[]superscript𝑓⋆subscript𝑋𝑜𝑏𝑠subscript𝑋𝑚𝑖𝑠superscript~𝑓⋆~𝑋\mathbb{E}\_{X\_{mis}|X\_{obs},M}\left[f^{\star}(X\_{obs},X\_{mis})\right]=\tilde{f}^{\star}(\widetilde{X}).

Clearly, f⋆⊙ΦC​Idirect-productsuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\odot\Phi^{CI} is Bayes optimal if ans only if:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼​[(f~⋆​(X~)−f⋆​(XC​I))2]=0𝔼delimited-[]superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼20\displaystyle\quad\mathbb{E}\left[\left(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\right)^{2}\right]=0 |  | (22) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ⇔iff\displaystyle\iff | ∑M∫P​(Xo​b​s,M)​(f~⋆​(X~)−f⋆​(XC​I))2​𝑑Xo​b​s=0subscript𝑀𝑃subscript𝑋𝑜𝑏𝑠𝑀superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼2differential-dsubscript𝑋𝑜𝑏𝑠0\displaystyle\quad\sum\_{M}\int P(X\_{obs},M)\left(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\right)^{2}dX\_{obs}=0 |  | (23) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ⇔iff\displaystyle\iff | ∀M,Xo​b​s:P​(Xo​b​s,M)>0,f~⋆​(X~)=f⋆​(XC​I)almost everywhere.:  for-all𝑀subscript𝑋𝑜𝑏𝑠 formulae-sequence𝑃subscript𝑋𝑜𝑏𝑠𝑀0superscript~𝑓⋆~𝑋  superscript𝑓⋆superscript𝑋𝐶𝐼almost everywhere\displaystyle\quad\forall M,X\_{obs}:P(X\_{obs},M)>0,\;\tilde{f}^{\star}(\widetilde{X})=f^{\star}(X^{CI})\quad\text{almost everywhere}. |  | (24) |

where equality [24](#A1.E24 "In Proof. ‣ A.7 Proof of Proposition 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?") is true since all terms are positive.

Besides, by Lemma [4.1](#S4.Thmlemma1 "Lemma 4.1 (First order approximation). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?"), we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 12​tr​(H¯m​i​s,m​i​s−​Σm​i​s|o​b​s,M)≤f~⋆​(X~)−f⋆​(XC​I)≤12​tr​(H¯m​i​s,m​i​s+​Σm​i​s|o​b​s,M).12trsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼12trsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀\frac{1}{2}\text{tr}\left(\bar{H}^{-}\_{mis,mis}\Sigma\_{mis|obs,M}\right)\leq\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\leq\frac{1}{2}\text{tr}\left(\bar{H}^{+}\_{mis,mis}\Sigma\_{mis|obs,M}\right). |  | (25) |

By convexity of the square function, it follows that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | (f~⋆​(X~)−f⋆​(XC​I))2≤12​max⁡(tr​(H¯m​i​s,m​i​s−​Σm​i​s|o​b​s,M)2,tr​(H¯m​i​s,m​i​s+​Σm​i​s|o​b​s,M)2).superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼212trsuperscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2trsuperscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2\left(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\right)^{2}\leq\frac{1}{2}\max\left(\text{tr}\left(\bar{H}^{-}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2},\text{tr}\left(\bar{H}^{+}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2}\right). |  | (26) |

Finally, by taking the expectation on both sides:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[(f~⋆​(X~)−f⋆​(XC​I))2]≤12​𝔼M​[max⁡(tr​(H¯m​i​s,m​i​s−​Σm​i​s|o​b​s,M)2,tr​(H¯m​i​s,m​i​s+​Σm​i​s|o​b​s,M)2)].𝔼delimited-[]superscriptsuperscript~𝑓⋆~𝑋superscript𝑓⋆superscript𝑋𝐶𝐼212subscript𝔼𝑀delimited-[]trsuperscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2trsuperscriptsubscriptsuperscript¯𝐻  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2\displaystyle\begin{split}&\mathbb{E}\left[\left(\tilde{f}^{\star}(\widetilde{X})-f^{\star}(X^{CI})\right)^{2}\right]\leq\\ &\hskip 40.00006pt\frac{1}{2}\mathbb{E}\_{M}\left[\max\left(\text{tr}\left(\bar{H}^{-}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2},\text{tr}\left(\bar{H}^{+}\_{mis,mis}\Sigma\_{mis|obs,M}\right)^{2}\right)\right].\end{split} | |  | (27) |

Combining equation ([21](#A1.E21 "In Proof. ‣ A.7 Proof of Proposition 4.1 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")) with inequality (LABEL:eq:cofimp2) concludes the proof.
∎

### A.8 Proof of Proposition [4.2](#S4.Thmproposition2 "Proposition 4.2 (Regression function discontinuities). ‣ 4.2 Regressing on conditional imputations, a good idea? ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

See [4.2](#S4.Thmproposition2 "Proposition 4.2 (Regression function discontinuities). ‣ 4.2 Regressing on conditional imputations, a good idea? ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

###### Proof.

We will prove this result by contradiction. Suppose that (i) f⋆∘ΦC​Isuperscript𝑓⋆superscriptΦ𝐶𝐼f^{\star}\circ\Phi^{CI} is not Bayes optimal, (ii) the probability of observing all variables is strictly positive, (iii) there exists a function g𝑔g continuous such that g∘ΦC​I𝑔superscriptΦ𝐶𝐼g\circ\Phi^{CI} is Bayes optimal.

Following a reasoning similar to the one in the proof of proposition [4.1](#S4.Thmproposition1 "Proposition 4.1 ((Non-)Consistency of chaining oracles). ‣ 4.1 Applying 𝑓^⋆ on conditional imputations: chaining oracles isn’t without risks. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?"), we can show that g∘ΦC​I𝑔superscriptΦ𝐶𝐼g\circ\Phi^{CI} is Bayes optimal if and only if:

|  |  |  |
| --- | --- | --- |
|  | ∀M,Xo​b​s:P​(Xo​b​s,M)>0,𝔼​[f⋆​(X)|Xo​b​s,M]=g​(XC​I)almost everywhere.:  for-all𝑀subscript𝑋𝑜𝑏𝑠 formulae-sequence𝑃subscript𝑋𝑜𝑏𝑠𝑀0𝔼delimited-[]conditionalsuperscript𝑓⋆𝑋  subscript𝑋𝑜𝑏𝑠𝑀  𝑔superscript𝑋𝐶𝐼almost everywhere\forall M,X\_{obs}:P(X\_{obs},M)>0,\quad\mathbb{E}\left[f^{\star}(X)|X\_{obs},M\right]=g(X^{CI})\quad\text{almost everywhere}. |  |

In particular since for all x𝑥x, the joint probability P​(M=(0,…,0),X=x)𝑃formulae-sequence𝑀0…0𝑋𝑥P(M=(0,\dots,0),X=x) of observing all variables is strictly positive, g𝑔g should satisfy this equality for M=(0,…,0)𝑀0…0M=(0,\dots,0), i.e.:

|  |  |  |
| --- | --- | --- |
|  | f⋆​(X)=g​(X)almost everywhere.superscript𝑓⋆𝑋  𝑔𝑋almost everywheref^{\star}(X)=g(X)\quad\text{almost everywhere}. |  |

Since g𝑔g is continuous, it implies g=f⋆𝑔superscript𝑓⋆g=f^{\star}.
Since by assumption, f⋆superscript𝑓⋆f^{\star} is not Bayes optimal, then g𝑔g is not either, which is a contradiction.
∎

### A.9 Example of a case where no continuous corrected imputation exists.

Let:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆:ℝ2:superscript𝑓⋆superscriptℝ2\displaystyle f^{\star}:\mathbb{R}^{2} | →ℝ→absentℝ\displaystyle\to\mathbb{R} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | (X1,X2)subscript𝑋1subscript𝑋2\displaystyle(X\_{1},X\_{2}) | ↦X23−3​X2maps-toabsentsuperscriptsubscript𝑋233subscript𝑋2\displaystyle\mapsto X\_{2}^{3}-3X\_{2} |  |

and let:

|  |  |  |  |
| --- | --- | --- | --- |
|  | X2=X1+ϵwithsubscript𝑋2  subscript𝑋1italic-ϵwith\displaystyle X\_{2}=X\_{1}+\epsilon\quad\text{with}\quad | 𝔼​[ϵ|X1,M=(0,1)]=0𝔼delimited-[]conditionalitalic-ϵ  subscript𝑋1𝑀010\displaystyle\mathbb{E}\left[\epsilon|X\_{1},M=(0,1)\right]=0 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝔼​[ϵ2|X1,M=(0,1)]=σ2,σ2>1formulae-sequence𝔼delimited-[]conditionalsuperscriptitalic-ϵ2  subscript𝑋1𝑀01superscript𝜎2superscript𝜎21\displaystyle\mathbb{E}\left[\epsilon^{2}|X\_{1},M=(0,1)\right]=\sigma^{2},\,\sigma^{2}>1 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝔼​[ϵ3|X1,M=(0,1)]=0𝔼delimited-[]conditionalsuperscriptitalic-ϵ3  subscript𝑋1𝑀010\displaystyle\mathbb{E}\left[\epsilon^{3}|X\_{1},M=(0,1)\right]=0 |  |

Suppose that X2subscript𝑋2X\_{2} is missing. Then the Bayes predictor is given by:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f~⋆​(X1,M=(0,1))superscript~𝑓⋆  subscript𝑋1𝑀 01\displaystyle\tilde{f}^{\star}(X\_{1},M=(0,1)) | =𝔼​[f⋆​(X)|X1,M=(0,1)]absent𝔼delimited-[]conditionalsuperscript𝑓⋆𝑋  subscript𝑋1𝑀01\displaystyle=\mathbb{E}\left[f^{\star}(X)|X\_{1},M=(0,1)\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[X23−3​X2|X1,M=(0,1)]absent𝔼delimited-[]superscriptsubscript𝑋23conditional3subscript𝑋2  subscript𝑋1𝑀01\displaystyle=\mathbb{E}\left[X\_{2}^{3}-3X\_{2}|X\_{1},M=(0,1)\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[(X1+ϵ)3−3​(X1+ϵ)|X1,M=(0,1)]absent𝔼delimited-[]superscriptsubscript𝑋1italic-ϵ3conditional3subscript𝑋1italic-ϵ  subscript𝑋1𝑀01\displaystyle=\mathbb{E}\left[\left(X\_{1}+\epsilon\right)^{3}-3\left(X\_{1}+\epsilon\right)|X\_{1},M=(0,1)\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼[X13+ϵ3+3X1ϵ2+3X12ϵ−3X1−3ϵ)|X1,M=(0,1)]\displaystyle=\mathbb{E}\left[X\_{1}^{3}+\epsilon^{3}+3X\_{1}\epsilon^{2}+3X\_{1}^{2}\epsilon-3X\_{1}-3\epsilon)|X\_{1},M=(0,1)\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =X13+3​X1​(σ2−1)absentsuperscriptsubscript𝑋133subscript𝑋1superscript𝜎21\displaystyle=X\_{1}^{3}+3X\_{1}(\sigma^{2}-1) |  |

Clearly, the Bayes predictor for M=(0,1)𝑀01M=(0,1) is:

* •

  continuous,
* •

  non-decreasing since σ2>1superscript𝜎21\sigma^{2}>1,
* •

  limX​1→+∞​f~⋆​(X1,M=(0,1))=+∞→𝑋1superscript~𝑓⋆
  subscript𝑋1𝑀01\underset{X1\to+\infty}{\lim}\tilde{f}^{\star}(X\_{1},M=(0,1))=+\infty and limX​1→−∞​f~⋆​(X1,M=(0,1))=−∞→𝑋1superscript~𝑓⋆
  subscript𝑋1𝑀01\underset{X1\to-\infty}{\lim}\tilde{f}^{\star}(X\_{1},M=(0,1))=-\infty.

Proof by contradiction: Suppose that there exists a function Φ:ℝ→ℝ:Φ→ℝℝ\Phi:\mathbb{R}\to\mathbb{R} (i) continuous and (ii) such that for all X1subscript𝑋1X\_{1}, f⋆​(X1,Φ​(X1))=f~⋆​(X1,M=(0,1))superscript𝑓⋆subscript𝑋1Φsubscript𝑋1superscript~𝑓⋆

subscript𝑋1𝑀
01f^{\star}(X\_{1},\Phi(X\_{1}))=\tilde{f}^{\star}(X\_{1},M=(0,1)).

![Refer to caption](/html/2106.00311/assets/x7.png)


Figure 6: Graph of X2↦f⋆​(X1,X2)maps-tosubscript𝑋2superscript𝑓⋆subscript𝑋1subscript𝑋2X\_{2}\mapsto f^{\star}(X\_{1},X\_{2})

Let x1+∈ℝsuperscriptsubscript𝑥1ℝx\_{1}^{+}\in\mathbb{R} such that f~⋆​(X1=x1+,M=(0,1))>2superscript~𝑓⋆formulae-sequencesubscript𝑋1superscriptsubscript𝑥1𝑀012\tilde{f}^{\star}(X\_{1}=x\_{1}^{+},M=(0,1))>2. x1+superscriptsubscript𝑥1x\_{1}^{+} exists since limX​1→+∞​f~⋆​(X1,M=(0,1))=+∞→𝑋1superscript~𝑓⋆

subscript𝑋1𝑀
01\underset{X1\to+\infty}{\lim}\tilde{f}^{\star}(X\_{1},M=(0,1))=+\infty. Clearly,

|  |  |  |
| --- | --- | --- |
|  | f⋆​(x1+,X2)=f~⋆​(x1+,M=(0,1))⇔X2=x2+withx2+>2.iffsuperscript𝑓⋆superscriptsubscript𝑥1subscript𝑋2superscript~𝑓⋆  superscriptsubscript𝑥1𝑀 01formulae-sequencesubscript𝑋2  superscriptsubscript𝑥2withsuperscriptsubscript𝑥22f^{\star}(x\_{1}^{+},X\_{2})=\tilde{f}^{\star}(x\_{1}^{+},M=(0,1))\iff X\_{2}=x\_{2}^{+}\quad\text{with}\quad x\_{2}^{+}>2. |  |

Similarly, let x1−∈ℝsuperscriptsubscript𝑥1ℝx\_{1}^{-}\in\mathbb{R} such that f~⋆​(X1=x1−,M=(0,1))<−2superscript~𝑓⋆formulae-sequencesubscript𝑋1superscriptsubscript𝑥1𝑀012\tilde{f}^{\star}(X\_{1}=x\_{1}^{-},M=(0,1))<-2. x1−superscriptsubscript𝑥1x\_{1}^{-} exists since limX​1→−∞​f~⋆​(X1,M=(0,1))=−∞→𝑋1superscript~𝑓⋆

subscript𝑋1𝑀
01\underset{X1\to-\infty}{\lim}\tilde{f}^{\star}(X\_{1},M=(0,1))=-\infty. Clearly,

|  |  |  |
| --- | --- | --- |
|  | f⋆​(x1−,X2)=f~⋆​(x1−,M=(0,1))⇔X2=x2−withx2−<−2.iffsuperscript𝑓⋆superscriptsubscript𝑥1subscript𝑋2superscript~𝑓⋆  superscriptsubscript𝑥1𝑀 01formulae-sequencesubscript𝑋2  superscriptsubscript𝑥2withsuperscriptsubscript𝑥22f^{\star}(x\_{1}^{-},X\_{2})=\tilde{f}^{\star}(x\_{1}^{-},M=(0,1))\iff X\_{2}=x\_{2}^{-}\quad\text{with}\quad x\_{2}^{-}<-2. |  |

So ΦΦ\Phi must satisfy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Φ​(x1−)Φsuperscriptsubscript𝑥1\displaystyle\Phi(x\_{1}^{-}) | =x2−<−2absentsuperscriptsubscript𝑥22\displaystyle=x\_{2}^{-}<-2 |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Φ​(x1+)Φsuperscriptsubscript𝑥1\displaystyle\Phi(x\_{1}^{+}) | =x2+>2absentsuperscriptsubscript𝑥22\displaystyle=x\_{2}^{+}>2 |  |

Note that since the Bayes predictor is non-decreasing, we have x1−<x1+superscriptsubscript𝑥1superscriptsubscript𝑥1x\_{1}^{-}<x\_{1}^{+}. Since ΦΦ\Phi is continuous, there exists xˇ1∈[x1−,x1+]subscriptˇ𝑥1superscriptsubscript𝑥1superscriptsubscript𝑥1\check{x}\_{1}\in\left[x\_{1}^{-},x\_{1}^{+}\right] and x^1∈[x1−,x1+]subscript^𝑥1superscriptsubscript𝑥1superscriptsubscript𝑥1\hat{x}\_{1}\in\left[x\_{1}^{-},x\_{1}^{+}\right] such that xˇ1<x^1subscriptˇ𝑥1subscript^𝑥1\check{x}\_{1}<\hat{x}\_{1}
and Φ​(xˇ1)=−1Φsubscriptˇ𝑥11\Phi(\check{x}\_{1})=-1 and Φ​(x^1)=1Φsubscript^𝑥11\Phi(\hat{x}\_{1})=1. It implies that:

|  |  |  |
| --- | --- | --- |
|  | f⋆​(xˇ1,Φ​(xˇ1))=f⋆​(xˇ1,−1)=2>−2=f⋆​(x^1,1)=f⋆​(x^1,Φ​(x^1)).superscript𝑓⋆subscriptˇ𝑥1Φsubscriptˇ𝑥1superscript𝑓⋆subscriptˇ𝑥1122superscript𝑓⋆subscript^𝑥11superscript𝑓⋆subscript^𝑥1Φsubscript^𝑥1f^{\star}(\check{x}\_{1},\Phi(\check{x}\_{1}))=f^{\star}(\check{x}\_{1},-1)=2>-2=f^{\star}(\hat{x}\_{1},1)=f^{\star}(\hat{x}\_{1},\Phi(\hat{x}\_{1})). |  |

This implies that the function X1↦f⋆​(X1,Φ​(X1))maps-tosubscript𝑋1superscript𝑓⋆subscript𝑋1Φsubscript𝑋1X\_{1}\mapsto f^{\star}(X\_{1},\Phi(X\_{1})) cannot be non-decreasing. Since the Bayes predictor is non-decreasing, the two cannot be equal. CONTRADICTION.

### A.10 Proof of Proposition [4.3](#S4.Thmproposition3 "Proposition 4.3 (Existence of continuous corrected imputations). ‣ 4.3 Fasten your seat belt: corrected imputations may experience discontinuities. ‣ 4 Imputation versus regression: choosing one may break the other ‣ What’s a good imputation to predict with missing values?")

We start by proving the result for a given missing pattern m∈{0,1}d𝑚superscript01𝑑m\in\{0,1\}^{d}. Take r∈{1,…,d−1}𝑟1…𝑑1r\in\{1,\ldots,d-1\} and consider a missing pattern m𝑚m such that |o​b​s​(m)|=r𝑜𝑏𝑠𝑚𝑟|obs(m)|=r.
We let F:ℝr×ℝd−r→ℝ:𝐹→superscriptℝ𝑟superscriptℝ𝑑𝑟ℝF:\mathbb{R}^{r}\times\mathbb{R}^{d-r}\to\mathbb{R} defined, for all (xo​b​s,xm​i​s)subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠(x\_{obs},x\_{mis}) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | F​(xo​b​s,xm​i​s)=f⋆​(xo​b​s,xm​i​s)−f~⋆​(xo​b​s,m).𝐹subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠superscript𝑓⋆subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠superscript~𝑓⋆subscript𝑥𝑜𝑏𝑠𝑚\displaystyle F(x\_{obs},x\_{mis})=f^{\star}(x\_{obs},x\_{mis})-\tilde{f}^{\star}(x\_{obs},m). |  | (28) |

Our aim is to find, for all xo​b​ssubscript𝑥𝑜𝑏𝑠x\_{obs}, a value xm​i​ssubscript𝑥𝑚𝑖𝑠x\_{mis} (depending continuously on xo​b​ssubscript𝑥𝑜𝑏𝑠x\_{obs}) satisfying

|  |  |  |  |
| --- | --- | --- | --- |
|  | F​(xo​b​s,xm​i​s)=0.𝐹subscript𝑥𝑜𝑏𝑠subscript𝑥𝑚𝑖𝑠0\displaystyle F(x\_{obs},x\_{mis})=0. |  | (29) |

To this aim, we check the assumptions of Theorem 6 in Arutyunov and Zhukovskiy [[2019](#bib.bib1)] for the function F𝐹F. The desired conclusion will follow.

Since f⋆superscript𝑓⋆f^{\star} is uniformly continuous and twice continuously differentiable, condition 1−3131-3 of Theorem 6 in Arutyunov and Zhukovskiy [[2019](#bib.bib1)] are satisfied.
To verify the next condition, we have to prove that there exists (xo​b​s,0,xm​i​s,0)subscript𝑥

𝑜𝑏𝑠0subscript𝑥

𝑚𝑖𝑠0(x\_{obs,0},x\_{mis,0}) such that F​(xo​b​s,0,xm​i​s,0)=0𝐹subscript𝑥

𝑜𝑏𝑠0subscript𝑥

𝑚𝑖𝑠00F(x\_{obs,0},x\_{mis,0})=0.
Note that this is equivalent to finding (xo​b​s,0,xm​i​s,0)subscript𝑥

𝑜𝑏𝑠0subscript𝑥

𝑚𝑖𝑠0(x\_{obs,0},x\_{mis,0}) satisfying

|  |  |  |  |
| --- | --- | --- | --- |
|  | f⋆​(xo​b​s,0,xm​i​s,0)=f~⋆​(xo​b​s,0,m)=𝔼​[f⋆​(X)|Xo​b​s=xo​b​s,0,M=m],superscript𝑓⋆subscript𝑥  𝑜𝑏𝑠0subscript𝑥  𝑚𝑖𝑠0superscript~𝑓⋆subscript𝑥  𝑜𝑏𝑠0𝑚𝔼delimited-[]formulae-sequenceconditionalsuperscript𝑓⋆𝑋subscript𝑋𝑜𝑏𝑠subscript𝑥  𝑜𝑏𝑠0𝑀𝑚\displaystyle f^{\star}(x\_{obs,0},x\_{mis,0})=\tilde{f}^{\star}(x\_{obs,0},m)=\mathbb{E}\left[f^{\star}(X)|X\_{obs}=x\_{obs,0},M=m\right], |  | (30) |

by definition of the regression function f~⋆superscript~𝑓⋆\tilde{f}^{\star}. By assumption, the support of Xm​i​s|Xo​b​s=xo​b​s,0,M=mformulae-sequenceconditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠subscript𝑥

𝑜𝑏𝑠0𝑀𝑚X\_{mis}|X\_{obs}=x\_{obs,0},M=m is connected. Therefore, the intermediate value theorem can be applied and proves the existence of a pair (xo​b​s,0,xm​i​s,0)subscript𝑥

𝑜𝑏𝑠0subscript𝑥

𝑚𝑖𝑠0(x\_{obs,0},x\_{mis,0}) satisfying equation ([30](#A1.E30 "In A.10 Proof of Proposition 4.3 ‣ Appendix A Proofs ‣ What’s a good imputation to predict with missing values?")). Finally, by assumption, the regularity condition (GR1) in Arutyunov and Zhukovskiy [[2019](#bib.bib1)] is satisfied. This proves that there exists a continuous mapping ϕ(m):ℝr→ℝd−r:superscriptitalic-ϕ𝑚→superscriptℝ𝑟superscriptℝ𝑑𝑟\phi^{(m)}:\mathbb{R}^{r}\to\mathbb{R}^{d-r} such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | F​(xo​b​s,ϕ(m)​(xo​b​s))=0.𝐹subscript𝑥𝑜𝑏𝑠superscriptitalic-ϕ𝑚subscript𝑥𝑜𝑏𝑠0\displaystyle F(x\_{obs},\phi^{(m)}(x\_{obs}))=0. |  | (31) |

The previous reasoning holds for all missing patterns m𝑚m, such that |m​i​s​(m)|≥1𝑚𝑖𝑠𝑚1|mis(m)|\geq 1. Besides the result is clear for r=0𝑟0r=0 since the imputation function is reduced to a constant in this case (no components of X𝑋X are observed). On the contrary, in the case where all covariates are observed (r=d𝑟𝑑r=d), no imputation function is needed. Therefore, the result holds for all 0≤r≤d0𝑟𝑑0\leq r\leq d, which concludes the proof.

## Appendix B Additional results

### B.1 NeuMiss+MLP architecture

x⊙m¯direct-product𝑥¯𝑚x\odot\bar{m}−-μ⊙m¯direct-product𝜇¯𝑚\mu\odot\bar{m}W(0)superscript𝑊0W^{(0)}++W(0)superscript𝑊0W^{(0)}++W(0)superscript𝑊0W^{(0)}++MLPY𝑌Y⊙m¯direct-productabsent¯𝑚\odot\bar{m}⊙m¯direct-productabsent¯𝑚\odot\bar{m}⊙m¯direct-productabsent¯𝑚\odot\bar{m}Neumann blockNon-linearity

Figure 7: (Non-linear) NeuMiss+MLP network architecture with a Neumann block of depth 3 — m¯=1−m¯𝑚1𝑚\bar{m}=1-m. MLP stands for a standard multi-layer perceptron with ReLU activations.

### B.2 Expressions of fb​o​w​l⋆subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙f^{\star}\_{bowl}, fw​a​v​e⋆subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒f^{\star}\_{wave} and fb​r​e​a​k⋆subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘f^{\star}\_{break} and the corresponding Bayes predictors.

#### Expressions of fb​o​w​l⋆subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙f^{\star}\_{bowl}, fw​a​v​e⋆subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒f^{\star}\_{wave} and fb​r​e​a​k⋆subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘f^{\star}\_{break}.

The functions f⋆superscript𝑓⋆f^{\star} used in the experimental study are defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | fb​o​w​l⋆​(X)subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙𝑋\displaystyle f^{\star}\_{bowl}(X) | =(β⊤​X+β0−1)2absentsuperscriptsuperscript𝛽top𝑋subscript𝛽012\displaystyle=\left(\beta^{\top}X+\beta\_{0}-1\right)^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | fw​a​v​e⋆​(X)subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒𝑋\displaystyle f^{\star}\_{wave}(X) | =(β⊤​X+β0−1)+∑(ai,bi)∈Sai​Φ​(γ​(β⊤​X+β0+bi))absentsuperscript𝛽top𝑋subscript𝛽01subscriptsubscript𝑎𝑖subscript𝑏𝑖𝑆subscript𝑎𝑖Φ𝛾superscript𝛽top𝑋subscript𝛽0subscript𝑏𝑖\displaystyle=(\beta^{\top}X+\beta\_{0}-1)+\sum\_{(a\_{i},b\_{i})\in S}a\_{i}\,\Phi\left(\gamma\left(\beta^{\top}X+\beta\_{0}+b\_{i}\right)\right) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | fb​r​e​a​k⋆​(X)subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘𝑋\displaystyle f^{\star}\_{break}(X) | =(β⊤​X+β0)+3×𝟙β⊤​X+β0>1absentsuperscript𝛽top𝑋subscript𝛽03subscript1superscript𝛽top𝑋subscript𝛽01\displaystyle=\left(\beta^{\top}X+\beta\_{0}\right)+3\times\mathds{1}\_{\beta^{\top}X+\beta\_{0}>1} |  |

where ΦΦ\Phi the standard Gaussian cdf, γ=20​π8𝛾20𝜋8\gamma=20\sqrt{\frac{\pi}{8}} and S={(2,−0.8),(−4,−1),(2,−1.2)}𝑆20.84121.2S=\left\{(2,-0.8),(-4,-1),(2,-1.2)\right\}. β𝛽\beta is chosen as a vector of ones rescaled so that var​(β⊤​X)=1varsuperscript𝛽top𝑋1\text{var}(\beta^{\top}X)=1. These functions are depicted in Figure [3](#S6.F3.2 "Figure 3 ‣ Choice of 𝑓^⋆ ‣ 6.1 Experimental setup ‣ 6 Empirical study of impute-n-regress procedures ‣ What’s a good imputation to predict with missing values?").

#### Expressions of the Bayes predictors.

The expressions of the corresponding Bayes predictors are given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f~b​o​w​l⋆​(X~)subscriptsuperscript~𝑓⋆𝑏𝑜𝑤𝑙~𝑋\displaystyle\tilde{f}^{\star}\_{bowl}(\widetilde{X}) | =𝔼​[fb​o​w​l⋆​(X)|Xo​b​s,M]absent𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙𝑋  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle=\mathbb{E}\left[f^{\star}\_{bowl}(X)|X\_{obs},M\right] |  | (32) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1)2+βm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​sabsentsuperscriptsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽012superscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\displaystyle=\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1\right)^{2}+\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis} |  | (33) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f~w​a​v​e⋆​(X~)subscriptsuperscript~𝑓⋆𝑤𝑎𝑣𝑒~𝑋\displaystyle\tilde{f}^{\star}\_{wave}(\widetilde{X}) | =𝔼​[fw​a​v​e⋆​(X)|Xo​b​s,M]absent𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒𝑋  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle=\mathbb{E}\left[f^{\star}\_{wave}(X)|X\_{obs},M\right] |  | (34) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽01\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1 |  | (35) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑(ai,bi)∈Sai​Φ​(βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0+bi1/γ2+βm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​s)subscriptsubscript𝑎𝑖subscript𝑏𝑖𝑆subscript𝑎𝑖Φsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0subscript𝑏𝑖1superscript𝛾2superscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\displaystyle\quad+\sum\_{(a\_{i},b\_{i})\in S}a\_{i}\,\Phi\left(\frac{\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}+b\_{i}}{\sqrt{1/\gamma^{2}+\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis}}}\right) |  | (36) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f~b​r​e​a​k⋆​(X~)subscriptsuperscript~𝑓⋆𝑏𝑟𝑒𝑎𝑘~𝑋\displaystyle\tilde{f}^{\star}\_{break}(\widetilde{X}) | =𝔼​[fb​r​e​a​k⋆​(X)|Xo​b​s,M]absent𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘𝑋  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle=\mathbb{E}\left[f^{\star}\_{break}(X)|X\_{obs},M\right] |  | (37) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0+3​(1−Φ​(1−μm​i​s|o​b​s,Mβm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​s))absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽031Φ1subscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}+3\left(1-\Phi\left(\frac{1-\mu\_{mis|obs,M}}{\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis}}\right)\right) |  | (38) |

with μm​i​s|o​b​s,Msubscript𝜇conditional𝑚𝑖𝑠

𝑜𝑏𝑠𝑀\mu\_{mis|obs,M} and Σm​i​s|o​b​s,MsubscriptΣconditional𝑚𝑖𝑠

𝑜𝑏𝑠𝑀\Sigma\_{mis|obs,M} the mean and covariance matrix of the conditional distribution P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M). Below, we give the expression of these parameters for the MCAR and Gaussian self-masking missing data mechanisms. Let μm​i​s|o​b​ssubscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\mu\_{mis|obs} and Σm​i​s|o​b​ssubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠\Sigma\_{mis|obs} the mean and covariance matrix of the conditional distribution P​(Xm​i​s|Xo​b​s)𝑃conditionalsubscript𝑋𝑚𝑖𝑠subscript𝑋𝑜𝑏𝑠P(X\_{mis}|X\_{obs}). Since the data is generated according to a multivariate Gaussian distribution 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}\left(\mu,\Sigma\right), we have:

|  |  |  |  |
| --- | --- | --- | --- |
|  | μm​i​s|o​b​ssubscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\mu\_{mis|obs} | =μm​i​s+Σm​i​s|o​b​s​Σo​b​s−1​(Xo​b​s−μo​b​s)absentsubscript𝜇𝑚𝑖𝑠subscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscript𝑋𝑜𝑏𝑠subscript𝜇𝑜𝑏𝑠\displaystyle=\mu\_{mis}+\Sigma\_{mis|obs}\Sigma\_{obs}^{-1}(X\_{obs}-\mu\_{obs}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | Σm​i​s|o​b​ssubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\Sigma\_{mis|obs} | =Σm​i​s,m​i​s−Σm​i​s,o​b​s​Σo​b​s−1​Σo​b​s,m​i​sabsentsubscriptΣ  𝑚𝑖𝑠𝑚𝑖𝑠subscriptΣ  𝑚𝑖𝑠𝑜𝑏𝑠superscriptsubscriptΣ𝑜𝑏𝑠1subscriptΣ  𝑜𝑏𝑠𝑚𝑖𝑠\displaystyle=\Sigma\_{mis,mis}-\Sigma\_{mis,obs}\Sigma\_{obs}^{-1}\Sigma\_{obs,mis} |  |

In the MCAR case, we simply have Σm​i​s|o​b​s,M=Σm​i​s|o​b​ssubscriptΣconditional𝑚𝑖𝑠

𝑜𝑏𝑠𝑀subscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠\Sigma\_{mis|obs,M}=\Sigma\_{mis|obs} and μm​i​s|o​b​s,M=μm​i​s|o​b​ssubscript𝜇conditional𝑚𝑖𝑠

𝑜𝑏𝑠𝑀subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\mu\_{mis|obs,M}=\mu\_{mis|obs}. In the Gaussian self-masking case, it has been shown in Le Morvan et al. [[2020a](#bib.bib14)] that P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M) is again Gaussian but with parameters:

|  |  |  |
| --- | --- | --- |
|  | Σm​i​s|o​b​s,M=(Dm​i​s,m​i​s−1+Σm​i​s|o​b​s−1)−1subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscriptsuperscriptsubscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠11\displaystyle\Sigma\_{mis|obs,M}=\left(D\_{mis,mis}^{-1}+\Sigma\_{mis|obs}^{-1}\right)^{-1} |  |
|  |  |  |
| --- | --- | --- |
|  | μm​i​s|o​b​s,M=Σm​i​s|o​b​s,M​(Dm​i​s,m​i​s−1​μ~m​i​s+Σm​i​s|o​b​s−1​μm​i​s|o​b​s)subscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscriptsubscript𝐷  𝑚𝑖𝑠𝑚𝑖𝑠1subscript~𝜇𝑚𝑖𝑠superscriptsubscriptΣconditional𝑚𝑖𝑠𝑜𝑏𝑠1subscript𝜇conditional𝑚𝑖𝑠𝑜𝑏𝑠\displaystyle\mu\_{mis|obs,M}=\Sigma\_{mis|obs,M}\left(D\_{mis,mis}^{-1}\widetilde{\mu}\_{mis}+\Sigma\_{mis|obs}^{-1}\mu\_{mis|obs}\right) |  |

where μ~~𝜇\tilde{\mu} and D𝐷D are parameters of the Gaussian self-masking missing data mechanism. Finally, we detail below the derivations to obtain the expression of the Bayes predictors.

#### Derivation of the Bayes predictor for fb​o​w​l⋆subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙f^{\star}\_{bowl}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fb​o​w​l⋆​(X)subscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙𝑋\displaystyle f^{\star}\_{bowl}(X) | =(β⊤​X+β0−1)2absentsuperscriptsuperscript𝛽top𝑋subscript𝛽012\displaystyle=\left(\beta^{\top}X+\beta\_{0}-1\right)^{2} |  | (39) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(βo​b​s⊤​Xo​b​s+βm​i​s⊤​Xm​i​s+β0−1)2absentsuperscriptsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝛽012\displaystyle=\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}X\_{mis}+\beta\_{0}-1\right)^{2} |  | (40) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(βo​b​s⊤​Xo​b​s+βm​i​s⊤​(Xm​i​s−μm​i​s|o​b​s,M)+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1)2absentsuperscriptsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽012\displaystyle=\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}(X\_{mis}-\mu\_{mis|obs,M})+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1\right)^{2} |  | (41) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1)2+(βm​i​s⊤​(Xm​i​s−μm​i​s|o​b​s,M))2absentsuperscriptsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽012superscriptsuperscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀2\displaystyle=\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1\right)^{2}+\left(\beta\_{mis}^{\top}(X\_{mis}-\mu\_{mis|obs,M})\right)^{2} |  | (42) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +2​βm​i​s⊤​(Xm​i​s−μm​i​s|o​b​s,M)​(βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1)2superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀superscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽01\displaystyle\quad+2\beta\_{mis}^{\top}(X\_{mis}-\mu\_{mis|obs,M})\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1\right) |  | (43) |

Now taking the expectation with regards to P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M), the last term vanishes and we get:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼​[fb​o​w​l⋆​(X)|Xo​b​s,M]=(βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0−1)2+βm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​s𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑏𝑜𝑤𝑙𝑋  subscript𝑋𝑜𝑏𝑠𝑀superscriptsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽012superscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\mathbb{E}\left[f^{\star}\_{bowl}(X)|X\_{obs},M\right]=\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0}-1\right)^{2}+\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis} |  | (44) |

#### Derivation of the Bayes predictor for fw​a​v​e⋆subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒f^{\star}\_{wave}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fw​a​v​e⋆​(X)subscriptsuperscript𝑓⋆𝑤𝑎𝑣𝑒𝑋\displaystyle f^{\star}\_{wave}(X) | =(β⊤​X+β0−1)+∑(ai,bi)∈Sai​Φ​(γ​(β⊤​X+β0+bi))absentsuperscript𝛽top𝑋subscript𝛽01subscriptsubscript𝑎𝑖subscript𝑏𝑖𝑆subscript𝑎𝑖Φ𝛾superscript𝛽top𝑋subscript𝛽0subscript𝑏𝑖\displaystyle=(\beta^{\top}X+\beta\_{0}-1)+\sum\_{(a\_{i},b\_{i})\in S}a\_{i}\,\Phi\left(\gamma\left(\beta^{\top}X+\beta\_{0}+b\_{i}\right)\right) |  | (45) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =(βo​b​s⊤​Xo​b​s+βm​i​s⊤​Xm​i​s+β0−1)absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝛽01\displaystyle=(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}X\_{mis}+\beta\_{0}-1) |  | (46) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +∑(ai,bi)∈Sai​Φ​(γ​(βo​b​s⊤​Xo​b​s+βm​i​s⊤​Xm​i​s+β0+bi))subscriptsubscript𝑎𝑖subscript𝑏𝑖𝑆subscript𝑎𝑖Φ𝛾superscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝛽0subscript𝑏𝑖\displaystyle\quad+\sum\_{(a\_{i},b\_{i})\in S}a\_{i}\,\Phi\left(\gamma\left(\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}X\_{mis}+\beta\_{0}+b\_{i}\right)\right) |  | (47) |

Define T(m)=βm​i​s⊤​Xm​i​ssuperscript𝑇𝑚superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠T^{(m)}=\beta\_{mis}^{\top}X\_{mis}. Since P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M) is Gaussian in both the MCAR and Gaussian self-masking cases, P​(T(m)|Xo​b​s,M)𝑃conditionalsuperscript𝑇𝑚

subscript𝑋𝑜𝑏𝑠𝑀P(T^{(m)}|X\_{obs},M) is also Gaussian with mean and variance given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μT(m)|Xo​b​s,Msubscript𝜇superscript𝑇conditional𝑚subscript𝑋  𝑜𝑏𝑠𝑀\displaystyle\mu\_{T^{(m)|X\_{obs,M}}} | =βm​i​s⊤​μm​i​s|o​b​s,Mabsentsuperscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀\displaystyle=\beta\_{mis}^{\top}\mu\_{mis|obs,M} |  | (48) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | σT(m)|Xo​b​s,M2subscriptsuperscript𝜎2superscript𝑇conditional𝑚subscript𝑋  𝑜𝑏𝑠𝑀\displaystyle\sigma^{2}\_{T^{(m)|X\_{obs,M}}} | =βm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​sabsentsuperscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\displaystyle=\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis} |  | (49) |

To compute the Bayes predictor, we now need to compute the quantity:

|  |  |  |
| --- | --- | --- |
|  | 𝔼T(m)|Xo​b​s,M​[Φ​(γ​(βo​b​s⊤​Xo​b​s+T(m)+β0+bi))]subscript𝔼conditionalsuperscript𝑇𝑚subscript𝑋  𝑜𝑏𝑠𝑀delimited-[]Φ𝛾superscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscript𝑇𝑚subscript𝛽0subscript𝑏𝑖\mathbb{E}\_{T^{(m)}|X\_{obs,M}}\left[\Phi\left(\gamma\left(\beta\_{obs}^{\top}X\_{obs}+T^{(m)}+\beta\_{0}+b\_{i}\right)\right)\right] |  |

This expectation can then be computed following [Bishop, [2006](#bib.bib3)] (section 4.5.2) which gives the result.

#### Derivation of the Bayes predictor for fb​r​e​a​k⋆subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘f^{\star}\_{break}.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | fb​r​e​a​k⋆​(X)subscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘𝑋\displaystyle f^{\star}\_{break}(X) | =(β⊤​X+β0)+3×𝟙β⊤​X+β0>1absentsuperscript𝛽top𝑋subscript𝛽03subscript1superscript𝛽top𝑋subscript𝛽01\displaystyle=\left(\beta^{\top}X+\beta\_{0}\right)+3\times\mathds{1}\_{\beta^{\top}X+\beta\_{0}>1} |  | (50) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[fb​r​e​a​k⋆​(X)|Xo​b​s,M]𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘𝑋  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\mathbb{E}\left[f^{\star}\_{break}(X)|X\_{obs},M\right] | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0} |  | (51) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +3×∫P​(Xm​i​s|Xo​b​s,M)​𝟙βo​b​s⊤​Xo​b​s+βm​i​s⊤​Xm​i​s+β0>1​𝑑Xm​i​s3𝑃conditionalsubscript𝑋𝑚𝑖𝑠  subscript𝑋𝑜𝑏𝑠𝑀subscript1superscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝑋𝑚𝑖𝑠subscript𝛽01differential-dsubscript𝑋𝑚𝑖𝑠\displaystyle\quad+3\times\int P(X\_{mis}|X\_{obs},M)\mathds{1}\_{\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}X\_{mis}+\beta\_{0}>1}dX\_{mis} |  | (52) |

Let U(m)=βo​b​s​Xo​b​s+βm​i​s​Xm​i​s+β0superscript𝑈𝑚subscript𝛽𝑜𝑏𝑠subscript𝑋𝑜𝑏𝑠subscript𝛽𝑚𝑖𝑠subscript𝑋𝑚𝑖𝑠subscript𝛽0U^{(m)}=\beta\_{obs}X\_{obs}+\beta\_{mis}X\_{mis}+\beta\_{0}. Since P​(Xm​i​s|Xo​b​s,M)𝑃conditionalsubscript𝑋𝑚𝑖𝑠

subscript𝑋𝑜𝑏𝑠𝑀P(X\_{mis}|X\_{obs},M) is Gaussian in both the MCAR and Gaussian self-masking cases, P​(U(m)|Xo​b​s,M)𝑃conditionalsuperscript𝑈𝑚

subscript𝑋𝑜𝑏𝑠𝑀P(U^{(m)}|X\_{obs},M) is also Gaussian with mean and variance given by:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | μU(m)|Xo​b​s,Msubscript𝜇superscript𝑈conditional𝑚subscript𝑋  𝑜𝑏𝑠𝑀\displaystyle\mu\_{U^{(m)|X\_{obs,M}}} | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0} |  | (53) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | σU(m)|Xo​b​s,M2subscriptsuperscript𝜎2superscript𝑈conditional𝑚subscript𝑋  𝑜𝑏𝑠𝑀\displaystyle\sigma^{2}\_{U^{(m)|X\_{obs,M}}} | =βm​i​s⊤​Σm​i​s|o​b​s,M​βm​i​sabsentsuperscriptsubscript𝛽𝑚𝑖𝑠topsubscriptΣconditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽𝑚𝑖𝑠\displaystyle=\beta\_{mis}^{\top}\Sigma\_{mis|obs,M}\beta\_{mis} |  | (54) |

Using the law of the unconscious statistician, we get:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼​[fb​r​e​a​k⋆​(X)|Xo​b​s,M]𝔼delimited-[]conditionalsubscriptsuperscript𝑓⋆𝑏𝑟𝑒𝑎𝑘𝑋  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\mathbb{E}\left[f^{\star}\_{break}(X)|X\_{obs},M\right] | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0} |  | (55) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +3×∫P​(U(m)|Xo​b​s,M)​𝟙U(m)>1​𝑑U(m)3𝑃conditionalsuperscript𝑈𝑚  subscript𝑋𝑜𝑏𝑠𝑀subscript1superscript𝑈𝑚1differential-dsuperscript𝑈𝑚\displaystyle\quad+3\times\int P(U^{(m)}|X\_{obs},M)\mathds{1}\_{U^{(m)}>1}dU^{(m)} |  | (56) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0} |  | (57) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +3×[1−ℙ​(U(m)≤1|Xo​b​s,M)]3delimited-[]1ℙsuperscript𝑈𝑚conditional1  subscript𝑋𝑜𝑏𝑠𝑀\displaystyle\quad+3\times\left[1-\mathbb{P}\left(U^{(m)}\leq 1|X\_{obs},M\right)\right] |  | (58) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =βo​b​s⊤​Xo​b​s+βm​i​s⊤​μm​i​s|o​b​s,M+β0absentsuperscriptsubscript𝛽𝑜𝑏𝑠topsubscript𝑋𝑜𝑏𝑠superscriptsubscript𝛽𝑚𝑖𝑠topsubscript𝜇conditional𝑚𝑖𝑠  𝑜𝑏𝑠𝑀subscript𝛽0\displaystyle=\beta\_{obs}^{\top}X\_{obs}+\beta\_{mis}^{\top}\mu\_{mis|obs,M}+\beta\_{0} |  | (59) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | +3×[1−ΦU(m)|Xo​b​s,M​(1)]3delimited-[]1subscriptΦconditionalsuperscript𝑈𝑚  subscript𝑋𝑜𝑏𝑠𝑀1\displaystyle\quad+3\times\left[1-\Phi\_{U^{(m)}|X\_{obs},M}(1)\right] |  | (60) |

### B.3 Supplementary experiments with fb​r​e​a​k∗subscriptsuperscript𝑓𝑏𝑟𝑒𝑎𝑘f^{\*}\_{break}.

![Refer to caption](/html/2106.00311/assets/x8.png)


Figure 8: Performances (R2 score on a test set) compared to that of the Bayes predictor across 10 repeated experiments.

[◄](/html/2106.00310)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2106.00311)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2106.00311)
[View original  
on arXiv](https://arxiv.org/abs/2106.00311)[►](/html/2106.00312)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 1 14:53:41 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
