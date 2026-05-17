---
arxiv: '1802.07814'
authors:
- Jianbo Chen
- Le Song
- Martin J. Wainwright
- Michael I. Jordan
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Learning to Explain: An Information-Theoretic Perspective on Model Interpretation'
url: http://arxiv.org/abs/1802.07814v2
year: 2018
---

# Learning to Explain: An Information-Theoretic Perspective on Model Interpretation

Jianbo Chen
  
Le Song
  
Martin J. Wainwright
  
Michael I. Jordan

###### Abstract

We introduce *instancewise feature selection* as a methodology for model interpretation. Our method is based on learning a function to extract a subset of features that are most informative for each given example. This feature selector is trained to maximize the mutual information between selected features and the response
variable, where the conditional distribution of the response variable given the input is the model to be explained. We develop an efficient variational approximation to the mutual information, and show the effectiveness of our method on a variety of synthetic and real data sets using both quantitative metrics and human evaluation.

Model Interpretation

## 1 Introduction

Interpretability is an extremely important criterion when a machine
learning model is applied in areas such as medicine, financial
markets, and criminal justice (e.g., see the discussion paper by
Lipton ((Lipton, [2016](#bib.bib16))), as well as references therein). Many
complex models, such as random forests, kernel methods, and deep
neural networks, have been developed and employed to optimize
prediction accuracy, which can compromise their ease of
interpretation.

In this paper, we focus on instancewise feature selection as
a specific approach for model interpretation. Given a machine learning
model, instancewise feature selection asks for the importance scores
of each feature on the prediction of a given instance, and the
relative importance of each feature are allowed to vary across
instances. Thus, the importance scores can act as an explanation for
the specific instance, indicating which features are the key for the
model to make its prediction on that instance. A related concept in
machine learning is feature selection, which selects a subset of
features that are useful to build a good predictor for a specified
response variable (Guyon & Elisseeff, [2003](#bib.bib9)). While feature
selection produces a global importance of features with respect to the
entire labeled data set, instancewise feature selection measures
feature importance locally for each instance labeled by the model.

Existing work on interpreting models approach the problem from two
directions. The first line of work computes the gradient of the output
of the correct class with respect to the input vector for the given
model, and uses it as a saliency map for masking the
input (Simonyan et al., [2013](#bib.bib26); Springenberg et al., [2014](#bib.bib27)). The gradient
is computed using a Parzen window approximation of the original
classifier if the original one is not
available (Baehrens et al., [2010](#bib.bib3)).
Another line of research approximates the model to be interpreted via
a locally additive model in order to explain the difference between
the model output and some “reference” output in terms of the
difference between the input and some “reference”
input (Bach et al., [2015](#bib.bib2); Kindermans et al., [2016](#bib.bib13); Ribeiro et al., [2016](#bib.bib24); Lundberg & Lee, [2017](#bib.bib17); Shrikumar et al., [2017](#bib.bib25)). Ribeiro et al. ([2016](#bib.bib24))
proposed the LIME, methods which randomly draws instances from a
density centered at the sample to be explained, and fits a sparse
linear model to predict the model outputs for these instances.
Shrikumar et al. ([2017](#bib.bib25)) presented DeepLIFT, a method designed
specifically for neural networks, which decomposes the output of a
neural network on a specific input by backpropagating the contribution
back to every feature of the input. Lundberg & Lee ([2017](#bib.bib17)) used
Shapley values to quantify the importance of features of a given
input, and proposed a sampling based method “kernel SHAP” for
approximating Shapley values. Essentially, the two directions both
approximate the model locally via an additive model, with different
definitions of locality. While the first one considers infinitesimal
regions on the decision surface and takes the first-order term in the
Taylor expansion as the additive model, the second one considers the
finite difference between an input vector and a reference vector.

In this paper, our approach to instancewise feature selection is via
mutual information, a conceptually different perspective from existing
approaches. We define an “explainer,” or instancewise feature
selector, as a model which returns a distribution over the subset of
features given the input vector.
For a given instance, an ideal explainer should assign the highest
probability to the subset of features that are most informative for
the associated model response. This motivates us to maximize the
mutual information between the selected subset of features and the
response variable with respect to the instancewise feature
selector. Direct estimation of mutual information and discrete feature
subset sampling are intractable; accordingly, we derive a tractable
method by first applying a variational lower bound for mutual
information, and then developing a continuous reparametrization of the
sampling distribution.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Training | Efficiency | Additive | Model-agnostic |
| LIME (Ribeiro et al., [2016](#bib.bib24)) | No | Low | Yes | Yes |
| Kernel SHAP (Lundberg & Lee, [2017](#bib.bib17)) | No | Low | Yes | Yes |
| DeepLIFT (Shrikumar et al., [2017](#bib.bib25)) | No | High | Yes | No |
| Salient map (Simonyan et al., [2013](#bib.bib26)) | No | High | Yes | No |
| Parzen (Baehrens et al., [2010](#bib.bib3)) | Yes | High | Yes | Yes |
| LRP (Bach et al., [2015](#bib.bib2)) | No | High | Yes | No |
| L2X | Yes | High | No | Yes |

Table 1: Summary of the properties of different
methods. “Training” indicates whether a method
requires training on an unlabeled data set. “Efficiency”
qualitatively evaluates the computational time during single
interpretation. “Additive” indicates whether a method is locally
additive. “Model-agnostic” indicates whether a method is generic
to black-box models.

At a high level, the primary differences between our approach and past
work are the following. First, our framework globally learns
a local explainer, and therefore takes the distribution of
inputs into consideration. Second, our framework removes the
constraint of local feature additivity on an explainer. These
distinctions enable our framework to yield a more efficient, flexible,
and natural approach for instancewise feature selection. In summary,
our contributions in this work are as follows (see also
Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") for systematic comparisons):

* •

  We propose an information-based framework for
  instancewise feature selection.
* •

  We introduce a learning-based method for instancewise
  feature selection, which is both efficient and model-agnostic.

Furthermore, we show that the effectiveness of our method on a variety of synthetic and real data sets using both quantitative metric and human evaluation on Amazon Mechanical
Turk.

## 2 A framework

We now lay out the primary ingredients of our general approach. While
our framework is generic and can be applied to both classification and
regression models, the current discussion is restricted to
classification models. We assume one has access to the output of a
model as a conditional distribution, ℙm(⋅∣x)\mathbb{P}\_{m}(\cdot\mid x), of
the response variable Y𝑌Y given the realization of the input random
variable X=x∈dX=x\in{}^{d}.

!(/html/1802.07814/assets/x1.png)

Figure 1: The graphical model of obtaining XSsubscript𝑋𝑆X\_{S} from X𝑋X.

### 2.1 Mutual information

Our method is derived from considering the mutual information between
a particular pair of random vectors, so we begin by providing some
basic background. Given two random vectors X𝑋X and Y𝑌Y, the
mutual information I​(X;Y)𝐼

𝑋𝑌I(X;Y) is a measure of dependence
between them; intuitively, it corresponds to how much knowledge of one
random vector reduces the uncertainty about the other. More precisely,
the mutual information is given by the Kullback-Leibler divergence of
the product of marginal distributions of X𝑋X and Y𝑌Y from the joint
distribution of X𝑋X and Y𝑌Y (Cover & Thomas, [2012](#bib.bib7)); it takes the
form

|  |  |  |
| --- | --- | --- |
|  | I​(X;Y)=𝔼X,Y​[log⁡pX​Y​(X,Y)pX​(X)​pY​(Y)],𝐼  𝑋𝑌subscript𝔼  𝑋𝑌delimited-[]subscript𝑝𝑋𝑌𝑋𝑌subscript𝑝𝑋𝑋subscript𝑝𝑌𝑌\displaystyle I(X;Y)=\mathbb{E}\_{X,Y}\left[\log\frac{p\_{XY}(X,Y)}{p\_{X}(X)p\_{Y}(Y)}\right], |  |

where pX​Ysubscript𝑝𝑋𝑌p\_{XY} and pX,pY

subscript𝑝𝑋subscript𝑝𝑌p\_{X},p\_{Y} are the joint and marginal probability
densities if X,Y

𝑋𝑌X,Y are continuous, or the joint and marginal
probability mass functions if they are discrete. The expectation is
taken with respect to the joint distribution of X𝑋X and Y𝑌Y. One can
show the mutual information is nonnegative and symmetric in two random
variables. The mutual information has been a popular criteria in
feature selection, where one selects the subset of features that
approximately maximizes the mutual information between the response
variable and the selected
features (Gao et al., [2016](#bib.bib8); Peng et al., [2005](#bib.bib22)). Here we propose to
use mutual information as a criteria for instancewise feature
selection.

### 2.2 How to construct explanations

We now describe how to construct explanations using mutual
information. In our specific setting, the pair (X,Y)𝑋𝑌(X,Y) are
characterized by the marginal distribution X∼ℙX​(⋅)similar-to𝑋subscriptℙ𝑋⋅X\sim\mathbb{P}\_{X}(\cdot), and a family of conditional distributions of the
form (Y∣x)∼ℙm(⋅∣x)(Y\mid x)\sim\mathbb{P}\_{m}(\cdot\mid x). For a given
positive integer k𝑘k, let ℘k={S⊂2d∣|S|=k}subscriptWeierstrass-p𝑘conditional-set𝑆superscript2𝑑𝑆𝑘\raisebox{1.79993pt}{\Large$\wp$}\_{k}=\{S\subset 2^{d}\,\mid\,|S|=k\} be the set of all subsets of size k𝑘k. An *explainer*
ℰℰ\mathcal{E} of size k𝑘k is a mapping from the feature space d
to the power set ℘ksubscriptWeierstrass-p𝑘\raisebox{1.79993pt}{\Large$\wp$}\_{k}; we allow the mapping to be randomized,
meaning that we can also think of ℰℰ\mathcal{E} as mapping x𝑥x to a
conditional distribution ℙ​(S∣x)ℙconditional𝑆𝑥\mathbb{P}(S\mid x) over S∈℘k𝑆subscriptWeierstrass-p𝑘S\in\raisebox{1.79993pt}{\Large$\wp$}\_{k}.
Given the chosen subset S=ℰ​(x)𝑆ℰ𝑥S=\mathcal{E}(x), we use xSsubscript𝑥𝑆x\_{S} to denote
the sub-vector formed by the chosen features. We view the choice of
the number of explaining features k𝑘k as best left in the hands of the
user, but it can also be tuned as a hyper-parameter.

We have thus defined a new random vector XS∈kX\_{S}\in{}^{k}; see
Figure [1](#S2.F1 "Figure 1 ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") for a probabilistic graphical model
representing its construction. We formulate instancewise feature
selection as seeking explainer that optimizes the criterion

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxℰ⁡I​(XS;Y)subject toS∼ℰ​(X).similar-to  subscriptℰ𝐼  subscript𝑋𝑆𝑌subject to𝑆 ℰ𝑋\displaystyle\max\_{\mathcal{E}}I(X\_{S};Y)\quad\text{subject to}\qquad S\sim\mathcal{E}(X). |  | (1) |

In words, we aim to maximize the mutual information between the
response variable from the model and the selected features, as a
function of the choice of selection rule.

It turns out that a global optimum of Problem ([1](#S2.E1 "In 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) has a
natural information-theoretic interpretation: it corresponds to the
minimization of the expected length of encoded message for the model
ℙm​(Y∣x)subscriptℙ𝑚conditional𝑌𝑥\mathbb{P}\_{m}(Y\mid x) using ℙm​(Y|xS)subscriptℙ𝑚conditional𝑌subscript𝑥𝑆\mathbb{P}\_{m}(Y|x\_{S}), where the latter
corresponds to the conditional distribution of Y𝑌Y upon observing the
selected sub-vector. More concretely, we have the following:

###### Theorem 1.

Letting 𝔼m[⋅∣x]\mathbb{E}\_{m}[\cdot\mid x] denote the expectation over
ℙm(⋅∣x)\mathbb{P}\_{m}(\cdot\mid x), define

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℰ∗​(x)superscriptℰ𝑥\displaystyle\mathcal{E}^{\*}(x) | :=argminS𝔼m[log1ℙm​(Y∣xS)|x].\displaystyle:\,=\arg\min\_{S}\;\mathbb{E}\_{m}\left[\log\frac{1}{\mathbb{P}\_{m}(Y\mid x\_{S})}\;\Big{|}\;x\right]. |  | (2) |

Then ℰ∗superscriptℰ\mathcal{E}^{\*} is a global optimum of
Problem ([1](#S2.E1 "In 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")). Conversely, any global optimum of
Problem ([1](#S2.E1 "In 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) degenerates to ℰ∗superscriptℰ\mathcal{E}^{\*} almost surely over
the marginal distribution ℙXsubscriptℙ𝑋\mathbb{P}\_{X}.

The proof of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") is left to Appendix.
In practice, the above global optimum is
obtained only if the explanation family ℰℰ\mathcal{E} is sufficiently
large. In the case when ℙm​(Y|xS)subscriptℙ𝑚conditional𝑌subscript𝑥𝑆\mathbb{P}\_{m}(Y|x\_{S}) is unknown or
computationally expensive to estimate accurately, we can choose to
restrict ℰℰ\mathcal{E} to suitably controlled families so as to prevent
overfitting.

## 3 Proposed method

A direct solution to Problem ([1](#S2.E1 "In 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) is not possible, so that
we need to approach it by a variational approximation. In particular,
we derive a lower bound on the mutual information, and we approximate
the model conditional distribution ℙmsubscriptℙ𝑚\mathbb{P}\_{m} by a suitably rich
family of functions.

### 3.1 Obtaining a tractable variational formulation

We now describe the steps taken to obtain a tractable variational
formulation.

##### A variational lower bound:

Mutual information between XSsubscript𝑋𝑆X\_{S} and Y𝑌Y can be expressed in terms of
the conditional distribution of Y𝑌Y given XSsubscript𝑋𝑆X\_{S}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | I​(XS,Y)𝐼subscript𝑋𝑆𝑌\displaystyle I(X\_{S},Y) | =𝔼​[log⁡ℙm​(XS,Y)ℙ​(XS)​ℙm​(Y)]=𝔼​[log⁡ℙm​(Y|XS)ℙm​(Y)]absent𝔼delimited-[]subscriptℙ𝑚subscript𝑋𝑆𝑌ℙsubscript𝑋𝑆subscriptℙ𝑚𝑌𝔼delimited-[]subscriptℙ𝑚conditional𝑌subscript𝑋𝑆subscriptℙ𝑚𝑌\displaystyle=\mathbb{E}\Big{[}\log\frac{\mathbb{P}\_{m}(X\_{S},Y)}{\mathbb{P}(X\_{S})\mathbb{P}\_{m}(Y)}\Big{]}=\mathbb{E}\Big{[}\log\frac{\mathbb{P}\_{m}(Y|X\_{S})}{\mathbb{P}\_{m}(Y)}\Big{]} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼​[log⁡ℙm​(Y|XS)]+Const.absent𝔼delimited-[]subscriptℙ𝑚conditional𝑌subscript𝑋𝑆Const.\displaystyle=\mathbb{E}\Big{[}\log\mathbb{P}\_{m}(Y|X\_{S})\Big{]}+\text{Const.} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼X​𝔼S|X​𝔼Y|XS​[log⁡ℙm​(Y|XS)]+Const.absentsubscript𝔼𝑋subscript𝔼conditional𝑆𝑋subscript𝔼conditional𝑌subscript𝑋𝑆delimited-[]subscriptℙ𝑚conditional𝑌subscript𝑋𝑆Const.\displaystyle=\mathbb{E}\_{X}\mathbb{E}\_{S|X}\mathbb{E}\_{Y|X\_{S}}\Big{[}\log\mathbb{P}\_{m}(Y|X\_{S})\Big{]}+\text{Const.} |  |

For a generic model, it is impossible to compute expectations under
the conditional distribution ℙm(⋅∣xs)\mathbb{P}\_{m}(\cdot\mid x\_{s}). Hence we
introduce a variational family for approximation:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒬𝒬\displaystyle\mathcal{Q} | :={ℚ∣ℚ={xS→ℚS(Y|xS),S∈℘k}}.\displaystyle:\,=\Big{\{}\mathbb{Q}\mid\mathbb{Q}=\{x\_{S}\to\mathbb{Q}\_{S}(Y|x\_{S}),S\in\raisebox{1.79993pt}{\Large$\wp$}\_{k}\}\Big{\}}. |  | (3) |

Note each member ℚℚ\mathbb{Q} of the family 𝒬𝒬\mathcal{Q} is a collection of
conditional distributions ℚS​(Y|xS)subscriptℚ𝑆conditional𝑌subscript𝑥𝑆\mathbb{Q}\_{S}(Y|x\_{S}), one for each choice of
k𝑘k-sized feature subset S𝑆S. For any ℚℚ\mathbb{Q}, an application of
Jensen’s inequality yields the lower bound

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼Y|XS​[log⁡ℙm​(Y|XS)]subscript𝔼conditional𝑌subscript𝑋𝑆delimited-[]subscriptℙ𝑚conditional𝑌subscript𝑋𝑆\displaystyle\mathbb{E}\_{Y|X\_{S}}[\log\mathbb{P}\_{m}(Y|X\_{S})] | ≥∫ℙm​(Y|XS)​log⁡ℚS​(Y|XS)absentsubscriptℙ𝑚conditional𝑌subscript𝑋𝑆subscriptℚ𝑆conditional𝑌subscript𝑋𝑆\displaystyle\geq\int\mathbb{P}\_{m}(Y|X\_{S})\log\mathbb{Q}\_{S}(Y|X\_{S}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =𝔼Y|XS​[log⁡ℚS​(Y|XS)],absentsubscript𝔼conditional𝑌subscript𝑋𝑆delimited-[]subscriptℚ𝑆conditional𝑌subscript𝑋𝑆\displaystyle=\mathbb{E}\_{Y|X\_{S}}[\log\mathbb{Q}\_{S}(Y|X\_{S})], |  |

where equality holds if and only if ℙm​(Y∣XS)subscriptℙ𝑚conditional𝑌subscript𝑋𝑆\mathbb{P}\_{m}(Y\mid X\_{S}) and
ℚS​(Y|XS)subscriptℚ𝑆conditional𝑌subscript𝑋𝑆\mathbb{Q}\_{S}(Y|X\_{S}) are equal in distribution. We have thus obtained a
variational lower bound of the mutual information I​(XS;Y)𝐼

subscript𝑋𝑆𝑌I(X\_{S};Y).
Problem ([1](#S2.E1 "In 2.2 How to construct explanations ‣ 2 A framework ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) can thus be relaxed as maximizing the
variational lower bound, over both the explanation ℰℰ\mathcal{E} and the
conditional distribution ℚℚ\mathbb{Q}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxℰ,ℚ⁡𝔼​[log⁡ℚS​(Y∣XS)]such that S∼ℰ​(X).  subscript  ℰℚ𝔼delimited-[]subscriptℚ𝑆conditional𝑌subscript𝑋𝑆such that S∼ℰ(X).\displaystyle\max\_{\mathcal{E},\mathbb{Q}}\mathbb{E}\Big{[}\log\mathbb{Q}\_{S}(Y\mid X\_{S})\Big{]}\qquad\mbox{such that $S\sim\mathcal{\mathcal{E}}(X)$.} |  | (4) |

For generic choices ℚℚ\mathbb{Q} and ℰℰ\mathcal{E}, it is still difficult to
solve the variational approximation ([4](#S3.E4 "In A variational lower bound: ‣ 3.1 Obtaining a tractable variational formulation ‣ 3 Proposed method ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")). In order
to obtain a tractable method, we need to restrict both ℚℚ\mathbb{Q} and ℰℰ\mathcal{E} to suitable families over which it is efficient to perform
optimization.

##### A single neural network for parametrizing ℚℚ\mathbb{Q}:

Recall that ℚ={ℚS(⋅∣xS),S∈℘k}\mathbb{Q}=\{\mathbb{Q}\_{S}(\cdot\mid x\_{S}),\;S\in\raisebox{1.79993pt}{\Large$\wp$}\_{k}\} is a collection of conditional distributions with
cardinality |ℚ|=(dk)ℚbinomial𝑑𝑘|\mathbb{Q}|={d\choose k}. We assume X𝑋X is a continuous
random vector, and ℙm​(Y∣x)subscriptℙ𝑚conditional𝑌𝑥\mathbb{P}\_{m}(Y\mid x) is continuous with respect
to x𝑥x. Then we introduce a single neural network function gα:→dΔc−1g\_{\alpha}:{}^{d}\to\Delta\_{c-1} for parametrizing ℚℚ\mathbb{Q}, where the
codomain is a (c−1)𝑐1(c-1)-simplex Δc−1={y∈[0,1]c:0≤yi≤1,∑i=1cyi=1}subscriptΔ𝑐1conditional-set𝑦superscript01𝑐formulae-sequence0subscript𝑦𝑖1superscriptsubscript𝑖1𝑐subscript𝑦𝑖1\Delta\_{c-1}=\{y\in[0,1]^{c}:0\leq y\_{i}\leq 1,\sum\_{i=1}^{c}y\_{i}=1\} for the class distribution, and
α𝛼\alpha denotes the learnable parameters. We define ℚS(Y|xS):=gα(x~S)\mathbb{Q}\_{S}(Y|x\_{S}):\,=g\_{\alpha}(\tilde{x}\_{S}), where x~S∈d\tilde{x}\_{S}\in{}^{d} is
transformed from x𝑥x with entries not in S𝑆S replaced by zeros:

|  |  |  |
| --- | --- | --- |
|  | (x~S)i={xi,i∈S,0,i∉S.subscriptsubscript~𝑥𝑆𝑖cases  subscript𝑥𝑖𝑖 𝑆otherwise  0𝑖 𝑆otherwise\displaystyle(\tilde{x}\_{S})\_{i}=\begin{cases}x\_{i},i\in S,\\ 0,i\notin S.\end{cases} |  |

When X𝑋X contains discrete features, we embed each discrete feature
with a vector, and the vector representing a specific feature is set
to zero simultaneously when the corresponding feature is not in S𝑆S.

### 3.2 Continuous relaxation of subset sampling

Direct estimation of the objective function in
equation ([4](#S3.E4 "In A variational lower bound: ‣ 3.1 Obtaining a tractable variational formulation ‣ 3 Proposed method ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) requires summing over (dk)binomial𝑑𝑘{d\choose k}
combinations of feature subsets after the variational
approximation. Several tricks exist for tackling this issue, like
REINFORCE-type Algorithms (Williams, [1992](#bib.bib29)), or weighted sum
of features parametrized by deterministic functions of X𝑋X. (A similar
concept to the second trick is the “soft attention” structure in
vision (Ba et al., [2014](#bib.bib1)) and NLP (Bahdanau et al., [2014](#bib.bib4)) where
the weight of each feature is parametrized by a function of the
respective feature itself.) We employ an alternative approach
generalized from Concrete Relaxation (Gumbel-softmax
trick) (Jang et al., [2017](#bib.bib11); Maddison et al., [2014](#bib.bib19), [2016](#bib.bib20)),
which empirically has a lower variance than REINFORCE and encourages
discreteness (Raffel et al., [2017](#bib.bib23)).

The Gumbel-softmax trick uses the concrete distribution as a
continuous differentiable approximation to a categorical distribution.
In particular, suppose we want to approximate a categorical random
variable represented as a one-hot vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} with
category probability p1,p2,…,pd

subscript𝑝1subscript𝑝2…subscript𝑝𝑑p\_{1},p\_{2},\dots,p\_{d}. The random perturbation for
each category is independently generated from a Gumbel(0,1)01(0,1)
distribution:

|  |  |  |
| --- | --- | --- |
|  | Gi=−log⁡(−log⁡ui),ui∼Uniform​(0,1).formulae-sequencesubscript𝐺𝑖subscript𝑢𝑖similar-tosubscript𝑢𝑖Uniform01\displaystyle G\_{i}=-\log(-\log u\_{i}),u\_{i}\sim\text{Uniform}(0,1). |  |

We add the random perturbation to the log probability of each category
and take a temperature-dependent softmax over the d𝑑d-dimensional
vector:

|  |  |  |
| --- | --- | --- |
|  | Ci=exp⁡{(log⁡pi+Gi)/τ}∑j=1dexp⁡{(log⁡pj+Gj)/τ}.subscript𝐶𝑖subscript𝑝𝑖subscript𝐺𝑖𝜏superscriptsubscript𝑗1𝑑subscript𝑝𝑗subscript𝐺𝑗𝜏\displaystyle C\_{i}=\frac{\exp\{(\log p\_{i}+G\_{i})/\tau\}}{\sum\_{j=1}^{d}\exp\{(\log p\_{j}+G\_{j})/\tau\}}. |  |

The resulting random vector C=(C1,…,Cd)𝐶subscript𝐶1…subscript𝐶𝑑C=(C\_{1},\dots,C\_{d}) is called a Concrete
random vector, which we denote by

|  |  |  |
| --- | --- | --- |
|  | C∼Concrete​(log⁡p1,…,log⁡pd).similar-to𝐶Concretesubscript𝑝1…subscript𝑝𝑑\displaystyle C\sim\text{Concrete}(\log p\_{1},\dots,\log p\_{d}). |  |

We apply the Gumbel-softmax trick to approximate weighted subset
sampling. We would like to sample a subset S𝑆S of k𝑘k distinct
features out of the d𝑑d dimensions. The sampling scheme for S𝑆S can be
equivalently viewed as sampling a k𝑘k-hot random vector Z𝑍Z from
Dkd:={z∈{0,1}d∣∑zi=k}D^{d}\_{k}:\,=\{z\in\{0,1\}^{d}\mid\sum z\_{i}=k\}, with each entry of
z𝑧z being one if it is in the selected subset S𝑆S and being zero
otherwise. An importance score which depends on the input vector is
assigned for each feature. Concretely, we define wθ:→ddw\_{\theta}\colon{}^{d}\to{}^{d} that maps the input to a d𝑑d-dimensional vector,
with the i𝑖ith entry of wθ​(X)subscript𝑤𝜃𝑋w\_{\theta}(X) representing the importance
score of the i𝑖ith feature.

We start with approximating sampling k𝑘k distinct features out of d𝑑d
features by the sampling scheme below: Sample a single feature out of
d𝑑d features independently for k𝑘k times. Discard the overlapping
features and keep the rest. Such a scheme samples at most k𝑘k
features, and is easier to approximate by a continuous relaxation. We
further approximate the above scheme by independently sampling k𝑘k
independent Concrete random vectors, and then we define a
d𝑑d-dimensional random vector V𝑉V that is the elementwise maximum of
C1,C2,…,Ck

superscript𝐶1superscript𝐶2…superscript𝐶𝑘C^{1},C^{2},\dots,C^{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cjsuperscript𝐶𝑗\displaystyle C^{j} | ∼Concrete​(wθ​(X))​ i.i.d. for ​j=1,2,…,k,formulae-sequencesimilar-toabsentConcretesubscript𝑤𝜃𝑋 i.i.d. for 𝑗1  2…𝑘\displaystyle\sim\text{Concrete}(w\_{\theta}(X))\text{ i.i.d. for }j=1,2,\dots,k, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | V𝑉\displaystyle V | =(V1,V2,…,Vd),Vi=maxj⁡Cij.formulae-sequenceabsentsubscript𝑉1subscript𝑉2…subscript𝑉𝑑subscript𝑉𝑖subscript𝑗superscriptsubscript𝐶𝑖𝑗\displaystyle=(V\_{1},V\_{2},\dots,V\_{d}),\quad V\_{i}=\max\_{j}C\_{i}^{j}. |  |

The random vector V𝑉V is then used to approximate the k𝑘k-hot random
vector Z𝑍Z during training.

We write V=V​(θ,ζ)𝑉𝑉𝜃𝜁V=V(\theta,\zeta) as V𝑉V is a function of θ𝜃\theta and a
collection of auxiliary random variables ζ𝜁\zeta sampled independently
from the Gumbel distribution. Then we use the elementwise product
V​(θ,ζ)⊙Xdirect-product𝑉𝜃𝜁𝑋V(\theta,\zeta)\odot X between V𝑉V and X𝑋X as an approximation of
X~Ssubscript~𝑋𝑆\tilde{X}\_{S}.

### 3.3 The final objective and its optimization

After having applied the continuous approximation of feature subset
sampling, we have reduced Problem ([4](#S3.E4 "In A variational lower bound: ‣ 3.1 Obtaining a tractable variational formulation ‣ 3 Proposed method ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")) to the
following:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxθ,α⁡𝔼X,Y,ζ​[log⁡gα​(V​(θ,ζ)⊙X,Y)],subscript  𝜃𝛼subscript𝔼  𝑋𝑌𝜁delimited-[]subscript𝑔𝛼direct-product𝑉𝜃𝜁𝑋𝑌\displaystyle\max\_{\theta,\alpha}\mathbb{E}\_{X,Y,\zeta}\Big{[}\log g\_{\alpha}(V(\theta,\zeta)\odot X,Y)\Big{]}, |  | (5) |

where gαsubscript𝑔𝛼g\_{\alpha} denotes the neural network used to approximate the
model conditional distribution, and the quantity θ𝜃\theta is used to
parametrize the explainer. In the case of classification with c𝑐c
classes, we can write

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼X,ζ[∑y=1c[ℙm(y∣X)loggα(V(θ,ζ)⊙X,y)].\displaystyle\mathbb{E}\_{X,\zeta}\Big{[}\sum\_{y=1}^{c}[\mathbb{P}\_{m}(y\mid X)\log g\_{\alpha}(V(\theta,\zeta)\odot X,y)\Big{]}. |  | (6) |

Note that the expectation operator 𝔼X,ζsubscript𝔼

𝑋𝜁\mathbb{E}\_{X,\zeta} does not depend
on the parameters (α,θ)𝛼𝜃(\alpha,\theta), so that during the training
stage, we can apply stochastic gradient methods to jointly optimize
the pair (α,θ)𝛼𝜃(\alpha,\theta). In each update, we sample a mini-batch
of unlabeled data with their class distributions from the model to be
explained, and the auxiliary random variables ζ𝜁\zeta, and we then
compute a Monte Carlo estimate of the gradient of the objective
function ([6](#S3.E6 "In 3.3 The final objective and its optimization ‣ 3 Proposed method ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation")).

### 3.4 The explaining stage

During the explaining stage, the learned explainer maps each sample
X𝑋X to a weight vector wθ​(X)subscript𝑤𝜃𝑋w\_{\theta}(X) of dimension d𝑑d, each entry
representing the importance of the corresponding feature for the
specific sample X𝑋X. In order to provide a deterministic explanation
for a given sample, we rank features according to the weight vector,
and the k𝑘k features with the largest weights are picked as the
explaining features.

For each sample, only a single forward pass through the neural network
parametrizing the explainer is required to yield explanation. Thus our
algorithm is much more efficient in the explaining stage compared to
other model-agnostic explainers like LIME or Kernel SHAP which require
thousands of evaluations of the original model per sample.

## 4 Experiments

We carry out experiments on both synthetic and real data sets.
For all experiments, we use RMSprop (Maddison et al., [2016](#bib.bib20)) with the default hyperparameters for
optimization. We also fix the step size to be 0.0010.0010.001 across
experiments. The temperature for Gumbel-softmax approximation is fixed
to be 0.10.10.1. Codes for reproducing the key results are available online at <https://github.com/Jianbo-Lab/L2X>.

!(/html/1802.07814/assets/x2.png)

Figure 2: The clock time (in log scale) of explaining 10,000

1000010,000 samples
for each method. The training time of L2X is shown in translucent
bars.

### 4.1 Synthetic Data

We begin with experiments on four synthetic data sets:

* •

  222-dimensional XOR as binary classification. The input vector X𝑋X is generated from a 101010-dimensional standard Gaussian. The response variable Y𝑌Y is generated from P​(Y=1|X)∝exp⁡{X1​X2}proportional-to𝑃𝑌conditional1𝑋subscript𝑋1subscript𝑋2P(Y=1|X)\propto\exp\{X\_{1}X\_{2}\}.
* •

  Orange Skin. The input vector X𝑋X is generated from a 101010-dimensional standard Gaussian. The response variable Y𝑌Y is generated from P​(Y=1|X)∝exp⁡{∑i=14Xi2−4}proportional-to𝑃𝑌conditional1𝑋superscriptsubscript𝑖14superscriptsubscript𝑋𝑖24P(Y=1|X)\propto\exp\{\sum\_{i=1}^{4}X\_{i}^{2}-4\}.
* •

  Nonlinear additive model. Generate X𝑋X from a 10-dimensional standard Gaussian. The response variable Y𝑌Y is generated from P​(Y=1|X)∝exp⁡{−100​sin⁡(2​X1)+2​|X2|+X3+exp⁡{−X4}}proportional-to𝑃𝑌conditional1𝑋1002subscript𝑋12subscript𝑋2subscript𝑋3subscript𝑋4P(Y=1|X)\propto\exp\{-100\sin(2X\_{1})+2|X\_{2}|+X\_{3}+\exp\{-X\_{4}\}\}.
* •

  Switch feature. Generate X1subscript𝑋1X\_{1} from a mixture of two Gaussians centered at ±3plus-or-minus3\pm 3 respectively with equal probability. If X1subscript𝑋1X\_{1} is generated from the Gaussian centered at 333, the 2−5252-5th dimensions are used to generate Y𝑌Y like the orange skin model. Otherwise, the 6−9​t​h69𝑡ℎ6-9th dimensions are used to generate Y𝑌Y from the nonlinear additive model.

!(/html/1802.07814/assets/x3.png)

!(/html/1802.07814/assets/x4.png)

!(/html/1802.07814/assets/x5.png)

!(/html/1802.07814/assets/x6.png)

Figure 3: The box plots for the median ranks of the influential features by each sample, over 10,000

1000010,000 samples for each data set. The red line and the dotted blue line on each box is the median and the mean respectively. Lower median ranks are better. The dotted green lines indicate the optimal median rank.

| Truth | Model | Key words |
| --- | --- | --- |
| positive | positive | Ray Liotta and Tom Hulce shine in this sterling example of brotherly love and commitment. Hulce plays Dominick, (nicky) a mildly mentally handicapped young man who is putting his 12 minutes younger, twin brother, Liotta, who plays Eugene, through medical school. It is set in Baltimore and deals with the issues of sibling rivalry, the unbreakable bond of twins, child abuse and good always winning out over evil. It is captivating, and filled with laughter and tears. If you have not yet seen this film, please rent it, I promise, you’ll be amazed at how such a wonderful film could go unnoticed. |
| negative | negative | Sorry to go against the flow but I thought this film was unrealistic, boring and way too long. I got tired of watching Gena Rowlands long arduous battle with herself and the crisis she was experiencing. Maybe the film has some cinematic value or represented an important step for the director but for pure entertainment value. I wish I would have skipped it. |
| negative | positive | This movie is chilling reminder of Bollywood being just a parasite of Hollywood. Bollywood also tends to feed on past blockbusters for furthering its industry. Vidhu Vinod Chopra made this movie with the reasoning that a cocktail mix of deewar and on the waterfront will bring home an oscar. It turned out to be rookie mistake. Even the idea of the title is inspired from the Elia Kazan classic. In the original, Brando is shown as raising doves as symbolism of peace. Bollywood must move out of Hollywoods shadow if it needs to be taken seriously. |
| positive | negative | When a small town is threatened by a child killer, a lady police officer goes after him by pretending to be his friend. As she becomes more and more emotionally involved with the murderer her psyche begins to take a beating causing her to lose focus on the job of catching the criminal. Not a film of high voltage excitement, but solid police work and a good depiction of the faulty mind of a psychotic loser. |

Table 2: True labels and labels predicted by the model are in the first two columns. Key words picked by L2X are highlighted in yellow.

The first three data sets are modified from commonly used data sets in the feature selection literature (Chen et al., [2017](#bib.bib5)). The fourth data set is designed specifically for instancewise feature selection. Every sample in the first data set has the first two dimensions as true features, where each dimension itself is independent of the response variable Y𝑌Y but the combination of them has a joint effect on Y𝑌Y. In the second data set, the samples with positive labels centered around a sphere in a four-dimensional space. The sufficient statistic is formed by an additive model of the first four features. The response variable in the third data set is generated from a nonlinear additive model using the first four features. The last data set switches important features (roughly) based on the sign of the first feature. The 1−5151-5 features are true for samples with X1subscript𝑋1X\_{1} generated from the Gaussian centered at −33-3, and the 1,6−9

1691,6-9 features are true otherwise.

We compare our method L2X (for “Learning to Explain”) with several
strong existing algorithms for instancewise feature selection,
including Saliency (Simonyan et al., [2013](#bib.bib26)), DeepLIFT (Shrikumar et al., [2017](#bib.bib25)), SHAP (Lundberg & Lee, [2017](#bib.bib17)), LIME (Ribeiro et al., [2016](#bib.bib24)). Saliency refers to the method that computes
the gradient of the selected class with respect to the input feature
and uses the absolute values as importance scores. SHAP refers to
Kernel SHAP. The number of samples used for explaining each instance
for LIME and SHAP is set as default for all experiments. We also
compare with a method that ranks features by the
input feature times the gradient of the selected class with respect to
the input feature. Shrikumar et al. ([2017](#bib.bib25)) showed it is equivalent to
LRP (Bach et al., [2015](#bib.bib2)) when activations are piecewise linear, and
used it in Shrikumar et al. ([2017](#bib.bib25)) as a strong baseline. We call it
“Taylor” as it is the first-order Taylor approximation of the model.

Our experimental setup is as follows. For each data set, we train a neural network model with three hidden dense layers. We can safely assume the neural network has successfully captured the important features, and ignored noise features, based on its error rate. Then we use Taylor, Saliency, DeepLIFT, SHAP, LIME, and L2X for instancewise feature selection on the trained neural network models. For L2X, the explainer is a neural network composed of two hidden layers. The variational family is composed of three hidden layers. All layers are linear with dimension 200200200. The number of desired features k𝑘k is set to the number of true features.

The underlying true features are known for each sample, and hence the
median ranks of selected features for each sample in a validation data
set are reported as a performance metric, the box plots of which have
been plotted in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"). We observe that
L2X outperforms all other methods on nonlinear additive and feature
switching data sets. On the XOR model, DeepLIFT, SHAP and L2X achieve
the best performance. On the orange skin model, all algorithms have
near optimal performance, with L2X and LIME achieving the most stable
performance across samples.

We also report the clock time of each method in Figure [2](#S4.F2 "Figure 2 ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"),
where all experiments were performed on a single NVidia Tesla k80 GPU,
coded in TensorFlow. Across all the four data sets, SHAP and LIME are
the least efficient as they require multiple evaluations of the
model. DeepLIFT, Taylor and Saliency requires a backward pass of the
model. DeepLIFT is the slowest among the three, probably due to the
fact that backpropagation of gradients for Taylor and Saliency are
built-in operations of TensorFlow, while backpropagation in DeepLIFT
is implemented with high-level operations in TensorFlow. Our method
L2X is the most efficient in the explanation stage as it only requires
a forward pass of the subset sampler. It is much more efficient
compared to SHAP and LIME even after the training time has been taken
into consideration, when a moderate number of samples (10,000) need to
be explained. As the scale of the data to be explained increases, the
training of L2X accounts for a smaller proportion of the over-all
time. Thus the relative efficiency of L2X to other algorithms
increases with the size of a data set.

| Truth | Predicted | Key sentence |
| --- | --- | --- |
| positive | positive | There are few really hilarious films about science fiction but this one will knock your sox off. The lead Martians Jack Nicholson take-off is side-splitting. The plot has a very clever twist that has be seen to be enjoyed. This is a movie with heart and excellent acting by all. Make some popcorn and have a great evening. |
| negative | negative | You get 5 writers together, have each write a different story with a different genre, and then you try to make one movie out of it. Its action, its adventure, its sci-fi, its western, its a mess. Sorry, but this movie absolutely stinks. 4.5 is giving it an awefully high rating. That said, its movies like this that make me think I could write movies, and I can barely write. |
| negative | positive | This movie is not the same as the 1954 version with Judy garland and James mason, and that is a shame because the 1954 version is, in my opinion, much better. I am not denying Barbra Streisand’s talent at all. She is a good actress and brilliant singer. I am not acquainted with Kris Kristofferson’s other work and therefore I can’t pass judgment on it. However, this movie leaves much to be desired. It is paced slowly, it has gratuitous nudity and foul language, and can be very difficult to sit through. However, I am not a big fan of rock music, so its only natural that I would like the judy garland version better. See the 1976 film with Barbra and Kris, and judge for yourself. |
| positive | negative | The first time you see the second renaissance it may look boring. Look at it at least twice and definitely watch part 2. it will change your view of the matrix. Are the human people the ones who started the war? Is ai a bad thing? |

Table 3: True labels and labels from the model are shown in the first two columns. Key sentences picked by L2X highlighted in yellow.

### 4.2 IMDB

The Large Movie Review Dataset (IMDB) is a dataset of movie reviews for sentiment classification (Maas et al., [2011](#bib.bib18)). It contains 50,000

5000050,000 labeled movie reviews, with a split of 25,000

2500025,000 for training and 25,000

2500025,000 for testing. The average document length is 231231231 words, and 10.710.710.7 sentences.
We use L2X to study two popular classes of models for sentiment analysis on the IMDB data set.

#### 4.2.1 Explaining a CNN model with key words

Convolutional neural networks (CNN) have shown excellent performance for sentiment analysis (Kim, [2014](#bib.bib12); Zhang & Wallace, [2015](#bib.bib32)).
We use a simple CNN model on Keras (Chollet et al., [2015](#bib.bib6)) for the
IMDB data set, which is composed of a word embedding of dimension
505050, a 111-D convolutional layer of kernel size 333 with 250250250
filters, a max-pooling layer and a dense layer of dimension 250250250 as
hidden layers. Both the convolutional and the dense layers are
followed by ReLU as nonlinearity, and
Dropout (Srivastava et al., [2014](#bib.bib28)) as regularization. Each review is
padded/cut to 400400400 words. The CNN model achieves 90%percent9090\% accuracy on
the test data, close to the state-of-the-art performance (around
94%percent9494\%). We would like to find out which k𝑘k words make the most
influence on the decision of the model in a specific review. The
number of key words is fixed to be k=10𝑘10k=10 for all the experiments.

The explainer of L2X is composed of a global component and a local component (See Figure 2 in Yang et al. ([2018](#bib.bib31))). The input is initially fed into a common embedding layer followed by a convolutional layer with 100100100 filters. Then the local component processes the common output using two convolutional layers with 505050 filters, and the global component processes the common output using a max-pooling layer followed by a 100100100-dimensional dense layer. Then we concatenate the global and local outputs corresponding to each feature, and process them through one convolutional layer with 505050 filters, followed by a Dropout layer (Srivastava et al., [2014](#bib.bib28)). Finally a convolutional network with kernel size 111 is used to yield the output. All previous convolutional layers are of kernel size 3, and ReLU is used as nonlinearity.
The variational family is composed of an word embedding layer
of the same size, followed by an average pooling and a
250250250-dimensional dense layer. Each entry of the output vector V𝑉V
from the explainer is multiplied with the embedding of the respective
word in the variational family. We use both automatic metrics and human annotators to validate the effectiveness of L2X.

##### Post-hoc accuracy.

We introduce post-hoc accuracy for quantitatively validating the effectiveness of our method. Each model explainer outputs a subset of features XSsubscript𝑋𝑆X\_{S}
for each specific sample X𝑋X. We use ℙm​(y|X~S)subscriptℙ𝑚conditional𝑦subscript~𝑋𝑆\mathbb{P}\_{m}(y|\tilde{X}\_{S}) to
approximate ℙm​(y|XS)subscriptℙ𝑚conditional𝑦subscript𝑋𝑆\mathbb{P}\_{m}(y|X\_{S}). That is, we feed in the sample X𝑋X to
the model with unselected words masked by zero paddings. Then we
compute the accuracy of using ℙm​(y|X~S)subscriptℙ𝑚conditional𝑦subscript~𝑋𝑆\mathbb{P}\_{m}(y|\tilde{X}\_{S}) to predict
samples in the test data set labeled by ℙm​(y|X)subscriptℙ𝑚conditional𝑦𝑋\mathbb{P}\_{m}(y|X), which we
call post-hoc accuracy as it is computed after instancewise
feature selection.

##### Human accuracy.

When designing human experiments, we assume
that the key words convey an attitude toward a movie, and can thus be
used by a human to infer the review sentiment. This assumption has
been partially validated given the aligned outcomes provided by
post-hoc accuracy and by human judges, because the alignment implies
the consistency between the sentiment judgement based on selected
words from the original model and that from humans. Based on this
assumption, we ask humans on Amazon Mechanical Turk (AMT) to infer the
sentiment of a review given the ten key words selected by each
explainer. The words adjacent to each other, like “not good at all,”
keep their adjacency on the AMT interface if they are selected
simultaneously. The reviews from different explainers have been mixed
randomly, and the final sentiment of each review is averaged over the
results of multiple human annotators. We measure whether the labels
from human based on selected words align with the labels provided by
the model, in terms of the average accuracy over 500500500 reviews in the
test data set. Some reviews are labeled as “neutral” based on
selected words, which is because the selected key words do not contain
sentiment, or the selected key words contain comparable numbers of
positive and negative words. Thus these reviews are neither put in the
positive nor in the negative class when we compute accuracy. We call
this metric human accuracy.

The result is reported in Table [4](#S4.T4 "Table 4 ‣ 4.2.2 Explaining hierarchical LSTM ‣ 4.2 IMDB ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"). We observe that the model prediction based on only ten words selected by L2X align with the original prediction for over 90%percent9090\% of the data. The human judgement given ten words also aligns with the model prediction for 84.4%percent84.484.4\% of the data. The human accuracy is even higher than that based on the original review, which is 83.3%percent83.383.3\% (Yang et al., [2018](#bib.bib31)). This indicates the selected words by L2X can serve as key words for human to understand the model behavior. Table [2](#S4.T2 "Table 2 ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") shows the results of our model on four examples.

#### 4.2.2 Explaining hierarchical LSTM

Another competitive class of models in sentiment analysis uses
hierarchical LSTM (Hochreiter & Schmidhuber, [1997](#bib.bib10); Li et al., [2015](#bib.bib15)). We
build a simple hierarchical LSTM by putting one layer of LSTM on top
of word embeddings, which yields a representation vector for each
sentence, and then using another LSTM to encoder all sentence
vectors. The output representation vector by the second LSTM is passed
to the class distribution via a linear layer. Both the two LSTMs and
the word embedding are of dimension 100100100. The word embedding is
pretrained on a large corpus (Mikolov et al., [2013](#bib.bib21)). Each
review is padded to contain 151515 sentences. The hierarchical LSTM
model gets around 90% accuracy on the test data. We take each
sentence as a single feature group, and study which sentence is the
most important in each review for the model.

The explainer of L2X is composed of a 100100100-dimensional word embedding followed by a convolutional layer and a max pooling layer to encode each sentence. The encoded sentence vectors are fed through three convolutional layers and a dense layer to get sampling weights for each sentence. The variational family also encodes each sentence with a convolutional layer and a max pooling layer. The encoding vectors are weighted by the output of the subset sampler, and passed through an average pooling layer and a dense layer to the class probability. All convolutional layers are of filter size 150150150 and kernel size 333. In this setting, L2X can be interpreted as a hard attention model (Xu et al., [2015](#bib.bib30)) that employs the Gumbel-softmax trick.

Comparison is carried out with the same metrics. For human accuracy, one selected sentence for each review is shown to human annotators. The other experimental setups are kept the same as above. We observe that post-hoc accuracy reaches 84.4%percent84.484.4\% with one sentence selected by L2X, and human judgements using one sentence align with the original model prediction for 77.4%percent77.477.4\% of data.
Table [3](#S4.T3 "Table 3 ‣ 4.1 Synthetic Data ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") shows the explanations from our model on four examples.

!(/html/1802.07814/assets/figs/both_0.png)

!(/html/1802.07814/assets/figs/both_2.png)

!(/html/1802.07814/assets/figs/both_3.png)

!(/html/1802.07814/assets/figs/both_5.png)

!(/html/1802.07814/assets/figs/both_6.png)

!(/html/1802.07814/assets/figs/both_8.png)

!(/html/1802.07814/assets/figs/both_9.png)

!(/html/1802.07814/assets/figs/both_13.png)

!(/html/1802.07814/assets/figs/both_15.png)

!(/html/1802.07814/assets/figs/both_16.png)

!(/html/1802.07814/assets/figs/explanation_0.png)

!(/html/1802.07814/assets/figs/explanation_2.png)

!(/html/1802.07814/assets/figs/explanation_3.png)

!(/html/1802.07814/assets/figs/explanation_5.png)

!(/html/1802.07814/assets/figs/explanation_6.png)

!(/html/1802.07814/assets/figs/explanation_8.png)

!(/html/1802.07814/assets/figs/explanation_9.png)

!(/html/1802.07814/assets/figs/explanation_13.png)

!(/html/1802.07814/assets/figs/explanation_15.png)

!(/html/1802.07814/assets/figs/explanation_16.png)

Figure 4: The above figure shows ten randomly selected figures of 333 and 888 in the validation set. The first line include the original digits while the second line does not. The selected patches are colored with red if the pixel is activated (white) and blue otherwise.

|  | IMDB-Word | IMDB-Sent | MNIST |
| --- | --- | --- | --- |
| Post-hoc accuracy | 0.90.8 | 0.849 | 0.958 |
| Human accuracy | 0.844 | 0.774 | NA |

Table 4: Post-hoc accuracy and human accuracy of L2X on three models: a word-based CNN model on IMDB, a hierarchical LSTM model on IMDB, and a CNN model on MNIST.

### 4.3 MNIST

The MNIST data set contains 28×28282828\times 28 images of handwritten digits (LeCun et al., [1998](#bib.bib14)). We form a subset of the MNIST data set by choosing images of digits 333 and 888, with 11,982

1198211,982 images for training and 1,984

19841,984 images for testing.
Then we train a simple neural network for binary classification over the subset, which achieves accuracy 99.7%percent99.799.7\% on the test data set. The neural network is composed of two convolutional layers of kernel size 555 and a dense linear layer at last. The two convolutional layers contains 888 and 161616 filters respectively, and both are followed by a max pooling layer of pool size 222. We try to explain each sample image with k=4𝑘4k=4 image patches on the neural network model, where each patch contains 4×4444\times 4 pixels, obtained by dividing each 28×28282828\times 28 image into 7×7777\times 7 patches. We use patches instead of raw pixels as features for better
visualization.

We parametrize the explainer and the variational family with three-layer and two-layer convolutional networks respectively, with max pooling added after each hidden layer. The 7×7777\times 7 vector sampled from the explainer is upsampled (with repetition) to size 28×28282828\times 28 and multiplied with the input raw pixels.

We use only the post-hoc accuracy for experiment, with results shown in Table [4](#S4.T4 "Table 4 ‣ 4.2.2 Explaining hierarchical LSTM ‣ 4.2 IMDB ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"). The predictions based on 4 patches selected by L2X out of 49 align with those from original images for 95.8%percent95.895.8\% of data. Randomly selected examples with explanations are shown in Figure [4](#S4.F4 "Figure 4 ‣ 4.2.2 Explaining hierarchical LSTM ‣ 4.2 IMDB ‣ 4 Experiments ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"). We observe that L2X captures most of the informative patches, in particular those containing patterns that can distinguish 3 and 8.

## 5 Conclusion

We have proposed a framework for instancewise feature selection via mutual information, and a method L2X which seeks a variational approximation of the mutual information, and makes use of a
Gumbel-softmax relaxation of discrete subset sampling during training. To our best knowledge, L2X is the first method to realize real-time interpretation of a black-box model. We have shown the efficiency and the capacity of L2X for instancewise feature selection on both synthetic and real data sets.

## Acknowledgements

L.S. was also supported in part by NSF IIS-1218749, NIH BIGDATA 1R01GM108341, NSF CAREER IIS-1350983, NSF IIS-1639792 EAGER, NSF CNS-1704701, ONR N00014-15-1-2340, Intel ISTC, NVIDIA and Amazon AWS. We thank Nilesh Tripuraneni for comments about the Gumbel trick.

## Appendix A Proof of Theorem 1

##### Forward direction:

Any explanation is represented as a conditional distribution of the feature subset over the input vector. Given the definition of S∗superscript𝑆S^{\*}, we have for any X𝑋X, and any explanation ℰ:S|X:ℰconditional𝑆𝑋\mathcal{E}:S|X,

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼S|X𝔼m[\displaystyle\mathbb{E}\_{S|X}\mathbb{E}\_{m}[ | logPm(Y|XS)|X]≤\displaystyle\log{P\_{m}(Y|X\_{S})}|X]\leq |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 𝔼m​[log⁡Pm​(Y|XS∗​(X))|X].subscript𝔼𝑚delimited-[]conditionalsubscript𝑃𝑚conditional𝑌subscript𝑋superscript𝑆𝑋𝑋\displaystyle\mathbb{E}\_{m}[\log{P\_{m}(Y|X\_{S^{\*}(X)})}|X]. |  |

In the case when S∗​(X)superscript𝑆𝑋S^{\*}(X) is a set instead of a singleton, we identify S∗​(X)superscript𝑆𝑋S^{\*}(X) with any distribution that assigns arbitrary probability to each elements in S∗​(X)superscript𝑆𝑋S^{\*}(X) with zero probability outside S∗​(X)superscript𝑆𝑋S^{\*}(X). With abuse of notation, S∗superscript𝑆S^{\*} indicates both the set function that maps every X𝑋X to a set S∗​(X)superscript𝑆𝑋S^{\*}(X) and any real-valued function that maps X𝑋X to an element in S∗​(X)superscript𝑆𝑋S^{\*}(X).

Taking expectation over the distribution of X𝑋X, and adding 𝔼​log⁡Pm​(Y)𝔼subscript𝑃𝑚𝑌\mathbb{E}\log P\_{m}(Y) at both sides, we have

|  |  |  |
| --- | --- | --- |
|  | I​(XS;Y)≤I​(XS∗;Y)𝐼  subscript𝑋𝑆𝑌𝐼  subscript𝑋superscript𝑆𝑌I(X\_{S};Y)\leq I(X\_{S^{\*}};Y) |  |

for any explanation ℰ:S|X:ℰconditional𝑆𝑋\mathcal{E}:S|X.

##### Reverse direction:

The reverse direction is proved by contradiction. Assume the optimal explanation P​(S|X)𝑃conditional𝑆𝑋P(S|X) is such that there exists a set M𝑀M of nonzero probability, over which P​(S|X)𝑃conditional𝑆𝑋P(S|X) does not degenerates to an element in S∗​(X)superscript𝑆𝑋S^{\*}(X). Concretely, we define M𝑀M as

|  |  |  |
| --- | --- | --- |
|  | M={x:P​(S∉S∗​(x)|X=x)>0}.𝑀conditional-set𝑥𝑃𝑆conditionalsuperscript𝑆𝑥𝑋𝑥0M=\{x:P(S\notin S^{\*}(x)|X=x)>0\}. |  |

For any x∈M𝑥𝑀x\in M, we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼S|X𝔼m[\displaystyle\mathbb{E}\_{S|X}\mathbb{E}\_{m}[ | logPm(Y|XS)|X=x]<\displaystyle\log{P\_{m}(Y|X\_{S})}|X=x]< |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼m​[log⁡Pm​(Y|XS∗​(x))|X=x],subscript𝔼𝑚delimited-[]conditionalsubscript𝑃𝑚conditional𝑌subscript𝑋superscript𝑆𝑥𝑋𝑥\displaystyle\mathbb{E}\_{m}[\log{P\_{m}(Y|X\_{S^{\*}(x)})}|X=x], |  | (7) |

where S∗​(x)superscript𝑆𝑥S^{\*}(x) is a deterministic function in the set of distributions that assign arbitrary probability to each elements in S∗​(x)superscript𝑆𝑥S^{\*}(x) with zero probability outside S∗​(x)superscript𝑆𝑥S^{\*}(x). Outside M𝑀M, we always have

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼S|X𝔼m[\displaystyle\mathbb{E}\_{S|X}\mathbb{E}\_{m}[ | logPm(Y|XS)|X=x]≤\displaystyle\log{P\_{m}(Y|X\_{S})}|X=x]\leq |  |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼m​[log⁡Pm​(Y|XS∗​(x))|X=x]subscript𝔼𝑚delimited-[]conditionalsubscript𝑃𝑚conditional𝑌subscript𝑋superscript𝑆𝑥𝑋𝑥\displaystyle\mathbb{E}\_{m}[\log{P\_{m}(Y|X\_{S^{\*}(x)})}|X=x] |  | (8) |

from the definition of S∗superscript𝑆S^{\*}. As M𝑀M is of nonzero size over P​(X)𝑃𝑋P(X), combining Equation [A](#A1.Ex17 "Reverse direction: ‣ Appendix A Proof of Theorem 1 ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") and Equation [A](#A1.Ex18 "Reverse direction: ‣ Appendix A Proof of Theorem 1 ‣ Learning to Explain: An Information-Theoretic Perspective on Model Interpretation") and taking expectation with respect to P​(X)𝑃𝑋P(X), we have

|  |  |  |  |
| --- | --- | --- | --- |
|  | I​(XS;Y)<I​(XS∗;Y),𝐼  subscript𝑋𝑆𝑌𝐼  subscript𝑋superscript𝑆𝑌I(X\_{S};Y)<I(X\_{S^{\*}};Y), |  | (9) |

which is a contradiction.

## References

* Ba et al. (2014)

  Ba, J., Mnih, V., and Kavukcuoglu, K.
  Multiple object recognition with visual attention.
  *arXiv preprint arXiv:1412.7755*, 2014.
* Bach et al. (2015)

  Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., and
  Samek, W.
  On pixel-wise explanations for non-linear classifier decisions by
  layer-wise relevance propagation.
  *PloS one*, 10(7):e0130140, 2015.
* Baehrens et al. (2010)

  Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., and
  MÃžller, K.-R.
  How to explain individual classification decisions.
  *Journal of Machine Learning Research*, 11(Jun):1803–1831, 2010.
* Bahdanau et al. (2014)

  Bahdanau, D., Cho, K., and Bengio, Y.
  Neural machine translation by jointly learning to align and
  translate.
  *arXiv e-prints*, abs/1409.0473, September 2014.
* Chen et al. (2017)

  Chen, J., Stern, M., Wainwright, M. J., and Jordan, M. I.
  Kernel feature selection via conditional covariance minimization.
  In *Advances in Neural Information Processing Systems 30*, pp. 6949–6958. 2017.
* Chollet et al. (2015)

  Chollet, F. et al.
  Keras.
  <https://github.com/keras-team/keras>, 2015.
* Cover & Thomas (2012)

  Cover, T. M. and Thomas, J. A.
  *Elements of information theory*.
  John Wiley & Sons, 2012.
* Gao et al. (2016)

  Gao, S., Ver Steeg, G., and Galstyan, A.
  Variational information maximization for feature selection.
  In *Advances in Neural Information Processing Systems*, pp. 487–495, 2016.
* Guyon & Elisseeff (2003)

  Guyon, I. and Elisseeff, A.
  An introduction to variable and feature selection.
  *Journal of machine learning research*, 3(Mar):1157–1182, 2003.
* Hochreiter & Schmidhuber (1997)

  Hochreiter, S. and Schmidhuber, J.
  Long short-term memory.
  *Neural computation*, 9(8):1735–1780, 1997.
* Jang et al. (2017)

  Jang, E., Gu, S., and Poole, B.
  Categorical reparameterization with gumbel-softmax.
  *stat*, 1050:1, 2017.
* Kim (2014)

  Kim, Y.
  Convolutional neural networks for sentence classification.
  *arXiv preprint arXiv:1408.5882*, 2014.
* Kindermans et al. (2016)

  Kindermans, P.-J., Schütt, K., Müller, K.-R., and Dähne, S.
  Investigating the influence of noise and distractors on the
  interpretation of neural networks.
  *arXiv preprint arXiv:1611.07270*, 2016.
* LeCun et al. (1998)

  LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
* Li et al. (2015)

  Li, J., Luong, M.-T., and Jurafsky, D.
  A hierarchical neural autoencoder for paragraphs and documents.
  *arXiv preprint arXiv:1506.01057*, 2015.
* Lipton (2016)

  Lipton, Z. C.
  The mythos of model interpretability.
  *arXiv preprint arXiv:1606.03490*, 2016.
* Lundberg & Lee (2017)

  Lundberg, S. M. and Lee, S.-I.
  A unified approach to interpreting model predictions.
  pp.  4768–4777, 2017.
* Maas et al. (2011)

  Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C.
  Learning word vectors for sentiment analysis.
  In *Proceedings of the 49th Annual Meeting of the Association
  for Computational Linguistics: Human Language Technologies-Volume 1*, pp. 142–150. Association for Computational Linguistics, 2011.
* Maddison et al. (2014)

  Maddison, C. J., Tarlow, D., and Minka, T.
  A\* sampling.
  In *Advances in Neural Information Processing Systems*, pp. 3086–3094, 2014.
* Maddison et al. (2016)

  Maddison, C. J., Mnih, A., and Teh, Y. W.
  The concrete distribution: A continuous relaxation of discrete random
  variables.
  *arXiv preprint arXiv:1611.00712*, 2016.
* Mikolov et al. (2013)

  Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J.
  Distributed representations of words and phrases and their
  compositionality.
  In *Advances in neural information processing systems*, pp. 3111–3119, 2013.
* Peng et al. (2005)

  Peng, H., Long, F., and Ding, C.
  Feature selection based on mutual information criteria of
  max-dependency, max-relevance, and min-redundancy.
  *IEEE Transactions on pattern analysis and machine
  intelligence*, 27(8):1226–1238, 2005.
* Raffel et al. (2017)

  Raffel, C., Luong, T., Liu, P. J., Weiss, R. J., and Eck, D.
  Online and linear-time attention by enforcing monotonic alignments.
  *arXiv preprint arXiv:1704.00784*, 2017.
* Ribeiro et al. (2016)

  Ribeiro, M. T., Singh, S., and Guestrin, C.
  Why should i trust you?: Explaining the predictions of any
  classifier.
  In *Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, pp.  1135–1144. ACM, 2016.
* Shrikumar et al. (2017)

  Shrikumar, A., Greenside, P., and Kundaje, A.
  Learning important features through propagating activation
  differences.
  In *ICML*, volume 70 of *Proceedings of Machine Learning
  Research*, pp.  3145–3153. PMLR, 06–11 Aug 2017.
* Simonyan et al. (2013)

  Simonyan, K., Vedaldi, A., and Zisserman, A.
  Deep inside convolutional networks: Visualising image classification
  models and saliency maps.
  *arXiv preprint arXiv:1312.6034*, 2013.
* Springenberg et al. (2014)

  Springenberg, J. T., Dosovitskiy, A., Brox, T., and Riedmiller, M.
  Striving for simplicity: The all convolutional net.
  *arXiv preprint arXiv:1412.6806*, 2014.
* Srivastava et al. (2014)

  Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov,
  R.
  Dropout: A simple way to prevent neural networks from overfitting.
  *The Journal of Machine Learning Research*, 15(1):1929–1958, 2014.
* Williams (1992)

  Williams, R. J.
  Simple statistical gradient-following algorithms for connectionist
  reinforcement learning.
  *Machine learning*, 8(3-4):229–256, 1992.
* Xu et al. (2015)

  Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R.,
  and Bengio, Y.
  Show, attend and tell: Neural image caption generation with visual
  attention.
  In *International Conference on Machine Learning*, pp. 2048–2057, 2015.
* Yang et al. (2018)

  Yang, P., Chen, J., Hsieh, C.-J., Wang, J.-L., and Jordan, M. I.
  Greedy attack and gumbel attack: Generating adversarial examples for
  discrete data.
  *arXiv preprint arXiv:1805.12316*, 2018.
* Zhang & Wallace (2015)

  Zhang, Y. and Wallace, B.
  A sensitivity analysis of (and practitioners’ guide to) convolutional
  neural networks for sentence classification.
  *arXiv preprint arXiv:1510.03820*, 2015.
