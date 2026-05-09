---
arxiv: '2106.02584'
authors:
- Jannik Kossen
- Neil Band
- Clare Lyle
- Aidan N. Gomez
- Tom Rainforth
- Yarin Gal
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs
  in Deep Learning'
url: http://arxiv.org/abs/2106.02584v2
year: 2021
---

[2106.02584] Untitled Document














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



\doparttoc\faketableofcontents

## 

Jannik Kossen1
Neil Band1∗
  
Clare Lyle1
Aidan N. Gomez1,3
Tom Rainforth2
Yarin Gal1
  
1 OATML, Department of Computer Science, University of Oxford
  
2 Department of Statistics, University of Oxford
  
3 Cohere
Equal Contribution. Correspondence to {jannik.kossen, neil.band}@cs.ox.ac.uk.

## Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning

Jannik Kossen1
Neil Band1∗
  
Clare Lyle1
Aidan N. Gomez1,3
Tom Rainforth2
Yarin Gal1
  
1 OATML, Department of Computer Science, University of Oxford
  
2 Department of Statistics, University of Oxford
  
3 Cohere
Equal Contribution. Correspondence to {jannik.kossen, neil.band}@cs.ox.ac.uk.

###### Abstract

We challenge a common assumption underlying most supervised *deep learning*: that a model makes a prediction depending only on its parameters and the features of a *single input*.
To this end, we introduce a general-purpose deep learning architecture that takes as input the *entire dataset* instead of processing one datapoint at a time.
Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms.
However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction.
Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models.
We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

### 1 Introduction

From CNNs [[57](#bib.bib57)] to Transformers [[90](#bib.bib90)], most of supervised deep learning relies on *parametric* modeling:
models learn parameters 𝜽𝜽\bm{\theta} from a set of training data 𝒟train={(𝒙1,𝒚1),…,(𝒙n,𝒚n)}subscript𝒟trainsubscript𝒙1subscript𝒚1…subscript𝒙𝑛subscript𝒚𝑛\mathcal{D}\_{\textup{train}}=\{(\bm{x}\_{1},\bm{y}\_{1}),\dots,(\bm{x}\_{n},\bm{y}\_{n})\} to maximize training likelihoods p​(𝒚∣𝒙;𝜽)𝑝conditional𝒚

𝒙𝜽p(\bm{y}\mid\bm{x};\bm{\theta}) mapping from features 𝒙∈𝓧𝒙𝓧\bm{x}\in\mathcal{\bm{X}} to target values 𝒚∈𝓨𝒚𝓨\bm{y}\in\mathcal{\bm{Y}}.
At test time, they then make a prediction p​(𝒚∗∣𝒙∗;𝜽)𝑝conditionalsuperscript𝒚

superscript𝒙𝜽p(\bm{y}^{\*}\mid\bm{x}^{\*};\bm{\theta}) that depends only on those parameters 𝜽𝜽\bm{\theta} and the test input 𝒙∗superscript𝒙\bm{x}^{\*}.
That is, parametric models do not consider direct dependencies between datapoints.

This paper challenges parametric modeling as the dominant paradigm in deep learning.
Based on the same end-to-end learning motivations that underpin deep learning itself, we consider giving models the *additional flexibility* of using training data *directly* when making predictions p​(𝒚∗∣𝒙∗,𝒟train;𝜽)𝑝conditionalsuperscript𝒚

superscript𝒙subscript𝒟train𝜽p(\bm{y}^{\*}\mid\bm{x}^{\*},\mathcal{D}\_{\textup{train}};\bm{\theta}).

Concretely, we introduce Non-Parametric Transformers (NPTs): a general deep learning architecture that takes the entire dataset as input and predicts by explicitly *learning* interactions between datapoints ([Fig. 1](#S1.F1 "In Background. ‣ 1 Introduction")).
NPTs leverage both parametric and *non*-parametric predictive mechanisms, with the use of end-to-end training allowing the model to naturally learn from the data how to balance the two.
Namely, instead of just learning predictive functions from the features to the targets of independent datapoints, NPTs can also learn to reason about general relationships *between* inputs.
We use multi-head self-attention [[4](#bib.bib4), [90](#bib.bib90), [59](#bib.bib59)] to model relationships between datapoints and construct a training objective for NPTs with a stochastic masking mechanism inspired by self-supervised reconstruction tasks in natural language processing [[24](#bib.bib24)].
We show that these models *learn* to look up information from other datapoints and capture the causal mechanism generating the data in semi-synthetic settings.
However, unlike conventional non-parametric models, NPTs are not forced to *only* make predictions in this way: they can also use the power of ordinary parametric deep learning.

###### Background.

While questioning parametric modeling assumptions is unconventional in deep learning, in statistics, so-called *non-parametric* models are a well-known and long-established field of study.
Non-parametric models make predictions in explicit dependence of the training data p​(𝒚∗∣𝒙∗,𝒟train)𝑝conditionalsuperscript𝒚

superscript𝒙subscript𝒟trainp(\bm{y}^{\*}\mid\bm{x}^{\*},\mathcal{D}\_{\textup{train}}).
The most popular example of such models in the machine learning community are perhaps Gaussian Processes [[74](#bib.bib74)].
Non-parametric models typically do not require any training of parameters, and instead often directly interpolate between training points according to a fixed procedure, e.g., [[74](#bib.bib74), p.17].
The interactions between inputs are fully defined by architectural choices and a small set of hyperparameters that must be carefully chosen.
Conventional non-parametric models cannot *learn* – in the sense familiar to deep learning practitioners – interactions from the data, limiting the flexibility these models have in adapting to the data at hand.
Approaches such as Deep Gaussian Processes [[22](#bib.bib22)], Deep Kernel Learning [[95](#bib.bib95)], and Neural Processes [[37](#bib.bib37), [36](#bib.bib36), [49](#bib.bib49)] have all sought to apply ideas from deep neural networks to non-parametrics.
Compared to NPTs, these approaches rely heavily on motivations from stochastic processes.
This leads to them being either less flexible than NPTs or requiring strong assumptions on the data, making them *inapplicable* to the practical scenarios considered in this paper (cf. §[3](#S3 "3 Related Work")).
Unlike previous work, NPTs explicitly learn to predict from interactions between datapoints, and they can be applied to general supervised machine learning tasks.
We refer to §[3](#S3 "3 Related Work") for an overview of these and other related approaches.

A key contribution of this paper is opening the door to a more general treatment of how deep learning models can make use of dependencies between datapoints for predictions.
Our results demonstrate that NPTs make use of interactions between datapoints in practice, and we show highly competitive performance on several established tabular datasets as well as early image classification results.
Additionally, we show that NPTs can solve complex reasoning tasks by combining representation learning and cross-datapoint lookup; something that is impossible for conventional deep learning or non-parametric models due to their inability to *learn* relations *between* datapoints.

We next discuss the specifics of our model (§[2](#S2 "2 Non-Parametric Transformers")), before moving on to related work (§[3](#S3 "3 Related Work")), empirical results (§[4](#S4 "4 Experiments")), and finally, limitations, future work, and conclusions (§[5](#S5 "5 Limitations, Future Work, and Conclusions")).

![Refer to caption](/html/2106.02584/assets/x1.png)


Figure 1: NPTs learn direct interactions between datapoints.
(a) Input data: predict masked target entry [?] for datapoint 𝑿isubscript𝑿𝑖\bm{X}\_{i}.
(b) Notation from §[2](#S2 "2 Non-Parametric Transformers").
(c) Parametric models predict only from the features of the given input.
(d) NPTs predict by modeling relationships between all points in the dataset.

### 2 Non-Parametric Transformers

Non-Parametric Transformers (NPTs) explicitly *learn* relationships between datapoints to improve predictions.
To accomplish this, they rely on three main ingredients:
(1) We provide the model with the entire dataset – all datapoints – as input.
We approximate this with minibatches where necessary for large data.
At test time, both training and test data are input to the model; during training, the model learns to predict targets from the training data (§[2.6](#S2.SS6 "2.6 Masking and Optimization ‣ 2 Non-Parametric Transformers")).
(2) We use self-attention between datapoints to explicitly model relationships between datapoints.
For example, at test time, the attention mechanism models relationships amongst training points, amongst test points, and between the two.
(3) NPT’s training objective is to reconstruct a corrupted version of the input dataset.
Similar to BERT [[24](#bib.bib24)], we apply stochastic masking to inputs and minimize a loss on predictions at entries masked out in the input.
Next, we introduce the three components in detail.

#### 2.1 Datasets as Inputs

NPTs take as input the entire dataset 𝑿∈ℝn×d𝑿superscriptℝ𝑛𝑑\bm{X}\in\mathbb{R}^{n\times d}.
The datapoints are stacked as the rows of this matrix {𝑿i,:∈ℝd∣i∈1​…​n}conditional-setsubscript𝑿

𝑖:superscriptℝ𝑑𝑖1…𝑛\{\bm{X}\_{i,:}\in\mathbb{R}^{d}\mid i\in 1\dots n\}, and we refer to the columns as attributes {𝑿:,j∈ℝn∣j∈1​…​d}conditional-setsubscript𝑿

:𝑗superscriptℝ𝑛𝑗1…𝑑\{\bm{X}\_{:,j}\in\mathbb{R}^{n}\mid j\in 1\dots d\}.
Each attribute is assumed to share a semantic meaning among all datapoints.
In single-target classification and regression, we assume that the targets (labels) are the final attribute 𝑿:,dsubscript𝑿

:𝑑\bm{X}\_{:,d}, and the other attributes {𝑿:,j∣j≠d}conditional-setsubscript𝑿

:𝑗𝑗𝑑\{\bm{X}\_{:,j}\mid j\neq d\} are input features, e.g., the pixels of an image.
Each 𝑿i,jsubscript𝑿

𝑖𝑗\bm{X}\_{i,j} is an entry or value.
In addition to tabular data, many modalities such as images, graphs, or timeseries can be reshaped to fit this format.
Note that this is a departure from common notation for supervised learning as introduced in §[1](#S1 "1 Introduction"), as the input 𝑿𝑿\bm{X} now includes both features and targets (collectively, attributes).

In masked language modeling [[24](#bib.bib24)], mask tokens denote which words in a sentence are unknown and where, at training time, model predictions will have a loss backpropagated.
Analogously, we use a binary matrix 𝑴∈ℝn×d𝑴superscriptℝ𝑛𝑑\bm{M}\in\mathbb{R}^{n\times d} to specify which entries are *masked* in the input 𝑿𝑿\bm{X}.
This matrix is also passed to NPT as input.
The task is to predict the masked values 𝑿M={𝑿i,j∣𝑴i,j=1}superscript𝑿𝑀conditional-setsubscript𝑿

𝑖𝑗subscript𝑴

𝑖𝑗1\bm{X}^{M}=\{\bm{X}\_{i,j}\mid\bm{M}\_{i,j}=1\} from the observed values 𝑿O={𝑿i,j∣𝑴i,j=0}superscript𝑿𝑂conditional-setsubscript𝑿

𝑖𝑗subscript𝑴

𝑖𝑗0\bm{X}^{O}=\{\bm{X}\_{i,j}\mid\bm{M}\_{i,j}=0\}, i.e., to predict p​(𝑿M∣𝑿O)𝑝conditionalsuperscript𝑿𝑀superscript𝑿𝑂p(\bm{X}^{M}\mid\bm{X}^{O}).

In summary, NPT takes as input the entire dataset and masking matrix (𝑿,𝑴)𝑿𝑴(\bm{X},\bm{M}), and makes predictions 𝑿^∈ℝn×d^𝑿superscriptℝ𝑛𝑑\smash{\hat{\bm{X}}\in\mathbb{R}^{n\times d}} for values masked at input.
This general setup accommodates many machine learning settings simply by adjusting the placement of the binary masks in 𝑴𝑴\bm{M}.
We focus on single-target classification and regression – corresponding to a masking matrix 𝑴𝑴\bm{M} with 111s at all entries of the label column 𝑿:,dsubscript𝑿

:𝑑\bm{X}\_{:,d} – but outline multi-target settings, imputation, self-supervision using input features, and semi-supervision in [Section C.4](#A3.SS4 "C.4 NPT Masking ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").
Next, we describe the NPT architecture.

#### 2.2 NPT Architecture

![Refer to caption](/html/2106.02584/assets/x2.png)


Figure 2: Overview of the Non-Parametric Transformer.
(a) The input dataset and mask matrix are stacked and (b) linearly embedded for all datapoints independently.
NPT then applies (c) Attention Between Datapoints (ABD, §[2.4](#S2.SS4 "2.4 Attention Between Datapoints (ABD) ‣ 2 Non-Parametric Transformers")) across all n𝑛n samples of hidden dimension h=d⋅eℎ⋅𝑑𝑒h=d\cdot e.
(d) Attention Between Attributes (ABA, §[2.5](#S2.SS5 "2.5 Attention Between Attributes (ABA) ‣ 2 Non-Parametric Transformers")) then attends between the attributes for each datapoint independently.
We repeat steps (c) and (d) and obtain a final prediction from a separate linear projection (not shown).

An overview of the Non-Parametric Transformer (NPT) is depicted in [Fig. 2](#S2.F2 "In 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers").
NPT receives the dataset and masking matrix (𝑿,𝑴)𝑿𝑴(\bm{X},\bm{M}) as input ([Fig. 2](#S2.F2 "In 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers")a).
We stack these and apply an identical linear embedding to each of n𝑛n datapoints, obtaining an input representation 𝑯(0)∈ℝn×d×esuperscript𝑯0superscriptℝ𝑛𝑑𝑒\bm{H}^{(0)}\in\mathbb{R}^{n\times d\times e} ([Fig. 2](#S2.F2 "In 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers")b).
Next, we apply a sequence of multi-head self-attention layers [[90](#bib.bib90), [24](#bib.bib24), [4](#bib.bib4)].
Crucially, we alternatingly apply attention between *datapoints* and attention between *attributes* of individual datapoints (Figs. [2](#S2.F2 "Figure 2 ‣ 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers")c-d).

These operations allow our model to learn both relationships between datapoints as well as transformations of individual datapoints.
Finally, an output embedding gives the prediction 𝑿^∈ℝn×d^𝑿superscriptℝ𝑛𝑑\smash{\hat{\bm{X}}\in\mathbb{R}^{n\times d}}, which now has predicted values at entries that were masked at input.
We refer to [Section C.3](#A3.SS3 "C.3 Input and Output Embdedings ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix") for details, such as treatment of categorical and continuous variables. Importantly:

###### Property 1.

NPTs are equivariant to a permutation of the datapoints. (cf. [Appendix A](#A1 "Appendix A Proof – NPT Is Equivariant over Datapoints ‣ Appendix") for proof.)

In other words, if the set of input datapoints is shuffled, NPTs produce the same prediction but shuffled in an analogous manner.
This explicitly encodes the assumption that the learned relations between datapoints should not depend on their ordering.
At a high level, permutation-equivariance holds because all components of NPTs are permutation-equivariant, and the composition of permutation-equivariant functions is itself permutation-equivariant.
We now briefly recap multi-head self-attention which plays an important role throughout the NPT architecture.

#### 2.3 Multi-Head Self-Attention

Multi-head self-attention (MHSA) is a powerful mechanism for learning complex interactions between elements in an input sequence.
Popularized in natural language processing [[90](#bib.bib90), [24](#bib.bib24), [4](#bib.bib4)], MHSA-based models have since been successfully applied to many areas of machine learning (cf. §[3](#S3 "3 Related Work")).

*Dot-product attention* computes attention weights by comparing queries {𝑸i∈ℝ1×hk∣i∈1​…​n}conditional-setsubscript𝑸𝑖superscriptℝ1subscriptℎ𝑘𝑖1…𝑛\{\bm{Q}\_{i}\in\mathbb{R}^{1\times h\_{k}}\mid i\in 1\dots n\} with keys {𝑲i∈ℝ1×hk∣i∈1​…​m}conditional-setsubscript𝑲𝑖superscriptℝ1subscriptℎ𝑘𝑖1…𝑚\{\bm{K}\_{i}\in\mathbb{R}^{1\times h\_{k}}\mid i\in 1\dots m\}, ultimately updating the representation of the queries by aggregating over values {𝑽i∈ℝ1×hv∣i∈1​…​m}conditional-setsubscript𝑽𝑖superscriptℝ1subscriptℎ𝑣𝑖1…𝑚\{\bm{V}\_{i}\in\mathbb{R}^{1\times h\_{v}}\mid i\in 1\dots m\} via the attention weights. We stack the queries, keys, and values into matrices 𝑸∈ℝn×hk𝑸superscriptℝ𝑛subscriptℎ𝑘\bm{Q}\in\mathbb{R}^{n\times h\_{k}}, 𝑲∈ℝm×hk𝑲superscriptℝ𝑚subscriptℎ𝑘\bm{K}\in\mathbb{R}^{m\times h\_{k}}, and 𝑽∈ℝm×hv𝑽superscriptℝ𝑚subscriptℎ𝑣\bm{V}\in\mathbb{R}^{m\times h\_{v}} and, as is commonly done for convenience, assume hk=hv=hsubscriptℎ𝑘subscriptℎ𝑣ℎh\_{k}=h\_{v}=h. Then, we compute dot-product attention as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Att​(𝑸,𝑲,𝑽)Att𝑸𝑲𝑽\displaystyle\text{Att}(\bm{Q},\bm{K},\bm{V}) | =softmax​(𝑸​𝑲T/h)​𝑽.absentsoftmax𝑸superscript𝑲𝑇ℎ𝑽\displaystyle=\text{softmax}(\bm{Q}\bm{K}^{T}/\sqrt{h})\bm{V}. |  | (1) |

*Multi-head* dot-product attention concatenates a series of k𝑘k independent *attention heads*

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | MHAtt​(𝑸,𝑲,𝑽)MHAtt𝑸𝑲𝑽\displaystyle\text{MHAtt}(\bm{Q},\bm{K},\bm{V}) | =concataxis=h​(𝑶1,…,𝑶k)​𝑾O, whereabsent  axisℎconcatsubscript𝑶1…subscript𝑶𝑘superscript𝑾𝑂 where\displaystyle=\underset{\text{axis}=h}{\text{concat}}(\bm{O}\_{1},\dots,\bm{O}\_{k})\bm{W}^{O},\text{\ where \ } |  | (2) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑶jsubscript𝑶𝑗\displaystyle\bm{O}\_{j} | =Att​(𝑸​𝑾jQ,𝑲​𝑾jK,𝑽​𝑾jV).absentAtt𝑸subscriptsuperscript𝑾𝑄𝑗𝑲subscriptsuperscript𝑾𝐾𝑗𝑽subscriptsuperscript𝑾𝑉𝑗\displaystyle=\text{Att}(\bm{Q}\bm{W}^{Q}\_{j},\bm{K}\bm{W}^{K}\_{j},\bm{V}\bm{W}^{V}\_{j}). |  | (3) |

We learn embedding matrices 𝑾jQ,𝑾jK,𝑾jV∈ℝh×h/k,j∈{1,…,k}formulae-sequence

subscriptsuperscript𝑾𝑄𝑗subscriptsuperscript𝑾𝐾𝑗subscriptsuperscript𝑾𝑉𝑗
superscriptℝℎℎ𝑘𝑗1…𝑘\smash{\bm{W}^{Q}\_{j},\bm{W}^{K}\_{j},\bm{W}^{V}\_{j}\in\mathbb{R}^{h\times h/k},j\in\{1,\dots,k\}} for each head j𝑗j, and 𝑾O∈ℝh×hsuperscript𝑾𝑂superscriptℝℎℎ\smash{\bm{W}^{O}\in\mathbb{R}^{h\times h}} mixes outputs from different heads.
Here, we focus on multi-head *self*-attention, MHSelfAtt​(𝑯)=MHAtt​(𝑸=𝑯,𝑲=𝑯,𝑽=𝑯)MHSelfAtt𝑯MHAttformulae-sequence𝑸𝑯formulae-sequence𝑲𝑯𝑽𝑯\text{MHSelfAtt}(\bm{H})=\text{MHAtt}(\bm{Q}=\bm{H},\bm{K}=\bm{H},\bm{V}=\bm{H}), which uses the *same* inputs for queries, keys, and values.
Following Transformer best practices to improve performance [[90](#bib.bib90), [24](#bib.bib24), [59](#bib.bib59), [16](#bib.bib16), [66](#bib.bib66)], we first add a residual branch and apply Layer Normalization (LN) [[3](#bib.bib3)] followed by MHSelfAtt​(⋅)MHSelfAtt⋅\text{MHSelfAtt}(\cdot),

|  |  |  |  |
| --- | --- | --- | --- |
|  | Res​(𝑯)=𝑯​𝑾res+MHSelfAtt​(LN​(𝑯)),Res𝑯𝑯superscript𝑾resMHSelfAttLN𝑯\displaystyle\text{Res}(\bm{H})=\bm{H}\bm{W}^{\text{res}}+\text{MHSelfAtt}(\text{LN}(\bm{H})), |  | (4) |

with learnable weight matrix 𝑾res∈ℝh×hsuperscript𝑾ressuperscriptℝℎℎ\bm{W}^{\text{res}}\in\mathbb{R}^{h\times h}.
Then, we add another residual branch with LN and a row-wise feed-forward network (rFF), finally giving the full multi-head self-attention layer as

|  |  |  |  |
| --- | --- | --- | --- |
|  | MHSA(𝑯)=Res(𝑯)+rFF(LN(Res(𝑯))∈ℝn×h.\displaystyle\text{MHSA}(\bm{H})=\text{Res}(\bm{H})+\text{rFF}(\text{LN}(\text{Res}(\bm{H}))\in\mathbb{R}^{n\times h}. |  | (5) |

#### 2.4 Attention Between Datapoints (ABD)

The Attention Between Datapoints (ABD) layer is a key operation for NPT.
It explicitly transforms data by reasoning about pairwise relationships between all datapoints, see [Fig. 2](#S2.F2 "In 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers")c.
As input to ABD, we flatten the output of the previous layer 𝑯(ℓ)superscript𝑯ℓ\bm{H}^{(\ell)} from ℝn×d×esuperscriptℝ𝑛𝑑𝑒\mathbb{R}^{n\times d\times e} to ℝn×hsuperscriptℝ𝑛ℎ\mathbb{R}^{n\times h} with h=d⋅eℎ⋅𝑑𝑒h=d\cdot e.
Then, we apply MHSA​(⋅)MHSA⋅\text{MHSA}(\cdot) between the intermediate datapoint representations {𝑯i(ℓ)∈ℝ1×h∣i∈1​…​n}conditional-setsubscriptsuperscript𝑯ℓ𝑖superscriptℝ1ℎ𝑖1…𝑛\{\bm{H}^{(\ell)}\_{i}\in\mathbb{R}^{1\times h}\mid i\in 1\dots n\} as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ABD​(𝑯(ℓ))=MHSA​(𝑯(ℓ))=𝑯(ℓ+1)∈ℝn×h.ABDsuperscript𝑯ℓMHSAsuperscript𝑯ℓsuperscript𝑯ℓ1superscriptℝ𝑛ℎ\displaystyle\text{ABD}(\bm{H}^{(\ell)})=\text{MHSA}(\bm{H}^{(\ell)})=\bm{H}^{(\ell+1)}\in\mathbb{R}^{n\times h}. |  | (6) |

At the first ABD layer, we input 𝑯(0)∈ℝn×d×esuperscript𝑯0superscriptℝ𝑛𝑑𝑒\bm{H}^{(0)}\in\mathbb{R}^{n\times d\times e}, the linearly embedded input data.
After applying ABD, we reshape the output again, from ℝn×hsuperscriptℝ𝑛ℎ\mathbb{R}^{n\times h} to ℝn×d×esuperscriptℝ𝑛𝑑𝑒\mathbb{R}^{n\times d\times e}.
Here, the rFF of each ABD layer is an MLP that is applied independently to each of the n𝑛n datapoints.

Note that this is distinct from how MHSA​(⋅)MHSA⋅\text{MHSA}(\cdot) is usually applied in the literature, as we compute attention between *different datapoints* and not between the *features of a single datapoint* [[90](#bib.bib90), [24](#bib.bib24), [25](#bib.bib25), [46](#bib.bib46)].
For example, in natural language processing, attention is usually applied between the tokens (attributes) of a sentence (datapoint) but not between different sentences.
For example, NPT could learn to attend between two datapoints with indices i𝑖i and i′superscript𝑖′i^{\prime} by embedding 𝑸isubscript𝑸𝑖\bm{Q}\_{i} and 𝑲i′subscript𝑲superscript𝑖′\bm{K}\_{i^{\prime}} in close proximity.
Following ([1](#S2.E1 "Equation 1 ‣ 2.3 Multi-Head Self-Attention ‣ 2 Non-Parametric Transformers")), datapoint i𝑖i will then attend more closely to i′superscript𝑖′i^{\prime} because 𝑸i​𝑲i′Tsubscript𝑸𝑖superscriptsubscript𝑲superscript𝑖′𝑇\bm{Q}\_{i}\bm{K}\_{i^{\prime}}^{T} will be large.
By stacking many ABD layers, NPT can learn higher-order interactions between datapoints [[90](#bib.bib90), [24](#bib.bib24)].

#### 2.5 Attention Between Attributes (ABA)

We now introduce Attention Between Attributes (ABA), which we by default perform after each ABD layer.
ABA layers can help the model learn better per-datapoint representations for the between-datapoint interactions, see [Fig. 2](#S2.F2 "In 2.2 NPT Architecture ‣ 2 Non-Parametric Transformers")d.
For ABA, we apply MHSA​(⋅)MHSA⋅\text{MHSA}(\cdot) independently to each row (corresponding to a single datapoint) in the input 𝑯i(ℓ)∈ℝd×esuperscriptsubscript𝑯𝑖ℓsuperscriptℝ𝑑𝑒\bm{H}\_{i}^{(\ell)}\in\mathbb{R}^{d\times e}, i∈{1,…,n}𝑖1…𝑛i\in\{1,\dots,n\}, giving

|  |  |  |  |
| --- | --- | --- | --- |
|  | ABA​(𝑯(ℓ))=stackaxis=n​(MHSA​(𝑯1(ℓ)),…,MHSA​(𝑯n(ℓ)))=𝑯(ℓ+1)∈ℝn×d×e.ABAsuperscript𝑯ℓaxis𝑛stackMHSAsubscriptsuperscript𝑯ℓ1…MHSAsubscriptsuperscript𝑯ℓ𝑛superscript𝑯ℓ1superscriptℝ𝑛𝑑𝑒\displaystyle\text{ABA}(\bm{H}^{(\ell)})=\underset{\text{axis}=n}{\text{stack}}(\text{MHSA}(\bm{H}^{(\ell)}\_{1}),\dots,\text{MHSA}(\bm{H}^{(\ell)}\_{n}))=\bm{H}^{(\ell+1)}\in\mathbb{R}^{n\times d\times e}. |  | (7) |

Just like in standard Transformers [[90](#bib.bib90), [24](#bib.bib24), [25](#bib.bib25), [46](#bib.bib46)], ABA is used to transform attribute representations of single datapoints independently.
We batch over the n𝑛n dimension to compute ABA efficiently.
By alternating between attention between datapoints (ABD) and attributes (ABA), NPTs can model both complex dependencies between points as well as learn suitable transformations of datapoints individually.
Next, we describe the use of masking mechanisms during NPT training and evaluation.

#### 2.6 Masking and Optimization

Masking. Much like in masked language modeling [[24](#bib.bib24)], we use masks to indicate which values NPT is expected to predict, and to prevent the model from accessing ground truth values.
Recall that NPT needs to predict p​(𝑿M∣𝑿O)𝑝conditionalsuperscript𝑿𝑀superscript𝑿𝑂p(\bm{X}^{M}\mid\bm{X}^{O}), with masked values 𝑿M={𝑿i,j∣𝑴i,j=1}superscript𝑿𝑀conditional-setsubscript𝑿

𝑖𝑗subscript𝑴

𝑖𝑗1\bm{X}^{M}=\{\bm{X}\_{i,j}\mid\bm{M}\_{i,j}=1\} and observed values 𝑿O={𝑿i,j∣𝑴i,j=0}superscript𝑿𝑂conditional-setsubscript𝑿

𝑖𝑗subscript𝑴

𝑖𝑗0\bm{X}^{O}=\{\bm{X}\_{i,j}\mid\bm{M}\_{i,j}=0\}.
Masked values can be either features or targets.
Canonically, masked language modeling is used to perform self-supervised learning on a sequence of tokens in a sentence [[24](#bib.bib24)].
We use such *stochastic feature masking* to mask feature values 𝑿i,j,j≠d

subscript𝑿

𝑖𝑗𝑗
𝑑\bm{X}\_{i,j},j\neq d, with probability pfeaturesubscript𝑝featurep\_{\text{feature}} during training.
We also apply stochastic masking to the targets of the training set 𝑿:,dsubscript𝑿

:𝑑\bm{X}\_{:,d} with probability ptargetsubscript𝑝targetp\_{\text{target}}.
We call this *stochastic target masking*.
Note that we take great care to avoid test set leakage and *never* reveal targets of the test set to NPT.
We refer to [Section C.4](#A3.SS4 "C.4 NPT Masking ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix") for full details of our masking procedure in a variety of settings.

NPT Objective. During training, we compute the negative log-likelihood loss at training targets ℒTargetssuperscriptℒTargets\mathcal{L}^{\textup{Targets}} as well as the auxiliary loss from masked-out features ℒFeaturessuperscriptℒFeatures\mathcal{L}^{\textup{Features}}.
We write the NPT training objective as ℒNPT=(1−λ)​ℒTargets+λ​ℒFeaturessuperscriptℒNPT1𝜆superscriptℒTargets𝜆superscriptℒFeatures\mathcal{L}^{\textup{NPT}}=(1-\lambda)\mathcal{L}^{\textup{Targets}}+\lambda\mathcal{L}^{\textup{Features}}, where λ𝜆\lambda is a hyperparameter.
At test time, we only mask and compute a loss over the targets of test points.
See [Section C.5](#A3.SS5 "C.5 NPT Optimization ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix") for optimization details.

This objective has a few notable elements.
Feature masking requires NPTs to make predictions over all attributes, encouraging the models to learn a representation of the entire dataset.
This increases the difficulty of the task and adds more supervision, which we find tends to have a beneficial regularizing effect.
Interestingly, stochastic *target* masking means that many training targets are *unmasked* to the model at training time.
This allows NPTs to learn to predict the masked targets of certain training datapoints using the *targets of other training datapoints* in addition to all input features.111A concern here could be that the model will memorize training targets and fail to generalize. In practice, we do not observe generalization issues, likely because (i) a loss is never backpropagated on an unmasked value, and (ii) BERT-style masking [[24](#bib.bib24)] uses token randomization to prevent memorization.
See [Section C.4](#A3.SS4 "C.4 NPT Masking ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").
NPTs no longer have to memorize a mapping between training inputs and outputs in their parameters 𝜽𝜽\bm{\theta}, and can instead use their representational capacity to learn functions using other *training features and targets as input*.
For example, NPTs could learn to assign test datapoints to clusters of training datapoints, and predict on those points using interpolation of the training targets in their respective cluster.
We explore the ability of NPTs to solve such tasks in §[4.2](#S4.SS2 "4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments").
Further, we study more complex extensions to these tasks, which cannot be solved by simple interpolative models, in [Section B.1.2](#A2.SS1.SSS2 "B.1.2 Modified Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix").

Handling Large Datasets.
Due to the poor 𝒪​(n2)𝒪superscript𝑛2\mathcal{O}(n^{2}) time and space complexity of self-attention, we resort to approximations once the data grows too large.
For example, we reach 242424 GB of GPU memory for standard NPT model sizes at about 80008000\mathrm{8}\mathrm{0}\mathrm{0}\mathrm{0} datapoints.
We find that processing the data in random subsets for model training and prediction, i.e., *minibatching*, is a simple and effective solution.
We construct minibatches such that, at test time, training and test data are both present in the same batch, to allow NPTs to attend to training datapoints.
In §[4.3](#S4.SS3 "4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments"), we show that NPTs make use of attention between datapoints with minibatching enabled.
See §[5](#S5 "5 Limitations, Future Work, and Conclusions") for further discussion and ideas for future work.

### 3 Related Work

Deep Non-Parametric Models.
Deep Gaussian Processes [[22](#bib.bib22)] and Deep Kernel Learning (DKL) [[95](#bib.bib95)] extend ideas from Gaussian Processes [[74](#bib.bib74)] to representation learning.
Deep GPs stack standard GPs with the aim to learn more expressive relationships between input points, sharing motivation with NPTs.
However, unlike NPTs, deep GPs are difficult to work with in practice, requiring complex approximate inference schemes [[21](#bib.bib21), [13](#bib.bib13), [77](#bib.bib77)].
DKL applies a neural network to each datapoint *independently* before passing points on to a standard Gaussian Process, making predictions based directly on similarity in embedding space instead of *learning* the interactions themselves.

Neural Processes.
Similar to GPs, Neural Processes (NPs) [[37](#bib.bib37), [36](#bib.bib36)] define a distribution over functions.
They use a latent variable model parametrized by neural networks, fulfilling specific architectural constraints to approximately preserve consistency of finite-dimensional marginals.
Attentive Neural Processes (ANPs) [[49](#bib.bib49)] extend Neural Processes to allow for direct attention between a context set and targets.
However, as the authors themselves stress, “NPs and GPs have different training regimes” [[49](#bib.bib49)].
While a GP can be trained on a single dataset, *(A)NPs require multiple realizations of the dataset*.
The authors further note that *“a direct comparison between the two is usually not plausible”* [[49](#bib.bib49)], which is why we cannot compare (A)NPs to NPTs on our standard tasks.

Attention.
NPTs are part of a line of recent work that explores the use of Transformer-based architectures outside of natural language processing, e.g., Transformers in computer vision [[67](#bib.bib67), [25](#bib.bib25), [46](#bib.bib46)] or architectures exploiting desirable invariances or equivariances [[59](#bib.bib59), [61](#bib.bib61), [33](#bib.bib33), [44](#bib.bib44)].
Like NPTs, Set Transformer [[59](#bib.bib59)] attends to a set of input points.
However, unlike NPTs, Set Transformer relies on the existence of multiple independent sets for training and makes only a single prediction for each set.
Like NPTs, Axial Transformers [[42](#bib.bib42)] and MSA Transformers [[73](#bib.bib73)] attend to multiple dimensions of matrix-shaped input.
However, Axial Transformers process single images as input, i.e., no attention across datapoints is performed.
MSA Transformers use attention within individual protein sequences and across an aligned protein family for contact prediction, but do not consider a more general setting.
Recent works have improved neural network performance on tabular data using attention.
AutoInt [[80](#bib.bib80)] is a direct application of multi-head attention to tabular data, and TabNet [[2](#bib.bib2)] sequentially attends to sparse subsets of the features inspired by tree-based models.
Both approaches do not reason about interactions between datapoints, a key contribution that we introduce with NPT in this work.

Few-Shot Learning, Meta-Learning, and Prompting.
In §[4.2](#S4.SS2 "4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments"), we apply NPTs to tasks that require learning of relational structure between datapoints on training data to achieve good generalization performance on novel test inputs.
This setup shares motivations with meta-learning [[8](#bib.bib8), [6](#bib.bib6), [56](#bib.bib56), [29](#bib.bib29)], in which a model is pre-trained on a variety of tasks, such that it can then learn new tasks using only a small number of additional training points from the new task.
However, we consider evaluation without any additional gradient updates, unlike recent meta-learning methods [[29](#bib.bib29), [97](#bib.bib97)] which are therefore inapplicable to this setting.
Recent works on few-shot learning with text prompting [[72](#bib.bib72), [12](#bib.bib12)] provide a trained Transformer-based language model with a few examples of a novel relationship in a prompt at prediction time, where they observe strong generalization on the task.
Similarly, we consider attention between a “context” of datapoints.
While ground-truth input-output pairs are provided for prompting, we consider settings in which no ground-truth is given at prediction time (cf. [Section B.1.2](#A2.SS1.SSS2 "B.1.2 Modified Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix")), but the model can solve the task if it has learned the underlying relational structure.

Semi-Supervised Learning and Graph Neural Networks.
NPTs relate to work on semi-supervised learning [[15](#bib.bib15), [27](#bib.bib27), [51](#bib.bib51)] and transductive learning [[89](#bib.bib89)], which both make use of unlabeled inputs during training.
NPTs natively support this by simply including any unlabeled datapoints with masked-out targets in the input matrix at training time.
This body of related work includes semi-supervised and transductive learning on graphs using graph neural networks (GNNs), e.g., [[52](#bib.bib52), [34](#bib.bib34), [91](#bib.bib91), [53](#bib.bib53), [96](#bib.bib96)].
NPTs can be seen as a generalization of GNNs in which a set of dependencies (edges) between datapoints is not known a priori and is instead learned from data using self-attention.
Like NPTs, Neural Relational Inference (NRI) [[53](#bib.bib53)] attempts to discover relations amongst datapoints.
However, NRI lacks scalability because it requires that embeddings be stored for each potential graph edge.

Metric Learning. (Deep) Metric Learning aims to learn distance functions such that the (semantic) similarity and dissimilarity between input points is meaningfully captured, e.g., [[92](#bib.bib92), [76](#bib.bib76), [93](#bib.bib93), [65](#bib.bib65), [94](#bib.bib94), [79](#bib.bib79)].
Similarly, retrieval models in NLP learn to look up relevant training instances for prediction [[38](#bib.bib38), [41](#bib.bib41), [39](#bib.bib39)].
The attention between datapoints in NPTs can be seen as implicitly learning exactly such (dis-)similarity.
Usually, metric learning embeds inputs by applying the same embedding function independently to each datapoint.
This is in contrast to NPTs, which leverage a learned self-attention mechanism between test inputs and training datapoints (including their labels) at prediction time.

### 4 Experiments

We seek to answer the following set of questions in our evaluation222
We release code for NPTs at [github.com/OATML/Non-Parametric-Transformers](https://github.com/OATML/Non-Parametric-Transformers).
 of NPTs:
(Q1) How do NPTs perform on standard benchmarks for supervised machine learning?
(Q2) Can NPTs successfully model interactions between datapoints in idealized settings?
(Q3) Do NPTs actually learn to rely on interactions between datapoints for prediction on real-world datasets?
(Q4) If so, what is the nature of these interactions, e.g., which other datapoints are relevant for prediction?

#### 4.1 NPTs Perform Competitively on Established Benchmarks

Table 1: 
Average rank order of various methods (±standard error)plus-or-minusstandard error(\pm\ \text{standard error}) on UCI benchmarks, across binary classification, multi-class classification, and regression tasks. We determine rank using the test area under the receiver operating characteristic (AUROC) curve on binary classification (4 of 10 datasets), accuracy on multi-class classification (2 of 10), and root mean squared error (RMSE) on regression (4 of 10), and sort methods by ascending rank for each metric. See [Section B.7](#A2.SS7 "B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix") for the full results.

|  |  |
| --- | --- |
| Method | AUROC |
| NPT | 2.50±0.87uncertain2.500.872.50\pm 0.87 |
| CatBoost | 2.75±0.85uncertain2.750.852.75\pm 0.85 |
| LightGBM | 3.50±1.55uncertain3.501.553.50\pm 1.55 |
| XGBoost | 4.75±1.25uncertain4.751.254.75\pm 1.25 |
| Gradient Boosting | 5.00±0.71uncertain5.000.715.00\pm 0.71 |
| MLP | 5.75±1.49uncertain5.751.495.75\pm 1.49 |
| Random Forest | 6.00±0.71uncertain6.000.716.00\pm 0.71 |
| TabNet | 6.50±1.32uncertain6.501.326.50\pm 1.32 |
| k-NN | 8.25±0.48uncertain8.250.488.25\pm 0.48 |

|  |  |
| --- | --- |
| Method | Accuracy |
| NPT | 2.50±0.50uncertain2.500.502.50\pm 0.50 |
| XGBoost | 2.50±1.50uncertain2.501.502.50\pm 1.50 |
| MLP | 3.00±2.00uncertain3.002.003.00\pm 2.00 |
| CatBoost | 3.50±0.50uncertain3.500.503.50\pm 0.50 |
| Gradient Boosting | 3.50±1.50uncertain3.501.503.50\pm 1.50 |
| Random Forest | 6.50±0.50uncertain6.500.506.50\pm 0.50 |
| TabNet | 7.50±0.50uncertain7.500.507.50\pm 0.50 |
| LightGBM | 7.50±1.50uncertain7.501.507.50\pm 1.50 |
| k-NN | 8.50±0.50uncertain8.500.508.50\pm 0.50 |

|  |  |
| --- | --- |
| Method | RMSE |
| CatBoost | 3.00±0.91uncertain3.000.913.00\pm 0.91 |
| XGBoost | 3.25±0.63uncertain3.250.633.25\pm 0.63 |
| NPT | 3.25±1.31uncertain3.251.313.25\pm 1.31 |
| Gradient Boosting | 4.00±1.08uncertain4.001.084.00\pm 1.08 |
| Random Forest | 4.50±0.87uncertain4.500.874.50\pm 0.87 |
| MLP | 5.00±1.22uncertain5.001.225.00\pm 1.22 |
| LightGBM | 6.50±1.55uncertain6.501.556.50\pm 1.55 |
| TabNet | 6.75±0.95uncertain6.750.956.75\pm 0.95 |
| k-NN | 8.75±0.25uncertain8.750.258.75\pm 0.25 |

To answer (Q1), we evaluate NPTs on tabular data from the UCI Repository [[26](#bib.bib26)] as well as the CIFAR-10 [[55](#bib.bib55)] and MNIST [[58](#bib.bib58)] image classification datasets.
Tabular data is ubiquitous in real-world machine learning [[20](#bib.bib20)] but notoriously challenging for general purpose deep neural networks, which are rarely used in practice here because they are consistently outperformed by boosting models [[78](#bib.bib78)].333We conduct an informal survey of all Kaggle [[45](#bib.bib45)] competitions using tabular data completed in 202020202020 with a public leaderboard.
In 111111 out of a total of 131313 cases, the winning entries relied on some form of boosting.

Tabular Datasets, Setup, and Baselines.
We evaluate NPTs over 10 datasets varying across the number of datapoints, number of features, composition (categorical or continuous) of features, and task.
4 of the 10 are binary classification, 2 are multi-class classification, and 4 are regression.
We compare NPT against a wide set of standard or state-of-the-art baselines: Random Forests [[10](#bib.bib10)], Gradient Boosting Trees [[32](#bib.bib32)], XGBoost [[17](#bib.bib17)], CatBoost [[71](#bib.bib71)], LightGBM [[48](#bib.bib48)], MLPs, k-NN [[30](#bib.bib30), [1](#bib.bib1)], and TabNet [[2](#bib.bib2)].
For additional background on tree-based models, see [Section D.1](#A4.SS1 "D.1 Tree-Based Baselines ‣ Appendix D Related Work – Continued ‣ Appendix").
We tune the parameters of all models on validation sets and use 10-fold cross-validation whenever computationally feasible.
Note that while we perform an extensive grid search for the baselines, we only search over a small set of configurations for NPTs.
We refer the reader to [Appendix E](#A5 "Appendix E Classification and Regression Benchmark Details ‣ Appendix") for further details on the setup for datasets and baselines, and [Section C.1](#A3.SS1 "C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix") for NPT hyperparameters.

Tabular Data Results.
We report the average rank order for NPT and various tree-based and deep learning baselines in [Table 1](#S4.T1 "In 4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments").
NPT achieves the highest average ranking on binary and multi-class classification tasks, outperforming CatBoost and XGBoost, two popular state-of-the-art boosting methods designed specifically for tabular data.
On regression tasks, NPT ties in average rank with XGBoost, and is outperformed only by CatBoost.
In addition to its strong rank-wise performance, NPT achieves best performance on 4 of the 10 benchmark datasets – more than any other method.
We find that these are remarkable results for a general purpose model that does not include tabular-specific design, supporting our hypothesis that attention between datapoints is a useful architectural inductive bias for prediction.
For all metrics across all datasets, i.e., NLL for classification, AUROC/accuracy for binary/multi-class classification, and (R)MSE for regression, we refer the reader to [Section B.7](#A2.SS7 "B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix").
In the appendix, we present ablations which suggest that the performance of NPT is robust across a wide range of hyperparameter choices ([Section B.4](#A2.SS4 "B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix")) and that both the introduction of the ABA layer and the stochastic feature masking contribute positively to the performance of NPTs ([Section B.5](#A2.SS5 "B.5 Ablation Study 2: NPT without ABA and NPT without Feature Masking ‣ Appendix B Additional Results ‣ Appendix")).

Image Data Results. On CIFAR-10, we replace our linear encoder with a CNN followed by ABD layers on the CNN encodings, achieving a test accuracy of 93.7%percent93.7$93.7$\%.
We achieve 98.3%percent98.3$98.3$\% accuracy on MNIST using linear patching [[25](#bib.bib25)].
Crucially, we show in §[4.3](#S4.SS3 "4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments") that NPTs learn to make use of interactions between images on both the CIFAR-10 and MNIST datasets, supporting the claim that attention between datapoints is useful beyond tabular data.
We also explore linear patching on CIFAR-10. See
[Section B.8](#A2.SS8 "B.8 Image Classification Results ‣ Appendix B Additional Results ‣ Appendix") for these results along with setup details and further discussion.

#### 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints

![Refer to caption](/html/2106.02584/assets/x3.png)


Figure 3: 
Demonstrating NPT’s ability to predict from Attention Between Datapoints (ABD).
(a) We append to the original data with masked targets [?] a copy of the same data with all masked values revealed, such that perfect prediction via lookup is possible.
(b) Attention weights indicate that the ideal lookup behavior is learned by NPT.
Shown are actual values learned by NPT at head 00 and depth 444 for the first 333 datapoints.
(c) NPT predictions closely match the ideal values.
(d) Additionally, we intervene on the values of individual targets, (e) finding that NPT predictions adjust accordingly.

To determine if NPTs can successfully learn to exploit interactions between datapoints (Q2), we introduce a task with strong input correlations for which we know ground-truth interactions.
Concretely, we use the UCI Protein regression dataset (cf. §[4.1](#S4.SS1 "4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments")) to construct the following semi-synthetic task:
for each batch, we input the original data with masked target values as well as a *copy* of the original data where all target values have been revealed, i.e., no masking is applied ([Fig. 3](#S4.F3 "In 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")a).
NPTs can use attention between datapoints to achieve arbitrarily good performance by *learning* to look up the target values in the matching duplicate row.
At test time, we input novel semi-synthetic test data to ensure that NPT has learned the correct relational mechanism and not just memorized target values.

NPTs successfully learn to perform this lookup between original and duplicate datapoints.
The ABD attention weights, visualized for the first three datapoints in [Fig. 3](#S4.F3 "In 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")b, clearly show the model correctly attending to the duplicates.
As a result, NPT predictions are Pearson-correlated with the duplicate targets at r=99.9%𝑟percent99.9r=99.9\% ([Fig. 3](#S4.F3 "In 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")c).
This equals an RMSE of only 0.440.440.44, about a magnitude lower than the error on the original Protein dataset ([Table 11](#A2.T11 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix")).
We conclude that NPTs learn to predict by looking up the target values from matching points.
Further discussion and attention maps are in [Section B.1.1](#A2.SS1.SSS1 "B.1.1 Attention Maps for the Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix").

Purely parametric models cannot exploit information from other datapoints, limiting their performance.
For example, MLPs achieve an RMSE of 3.623.623.62 on this task.
Non-parametric approaches also cannot solve this task in its original form, because unlike NPTs they must be told which datapoints are the originals (training data) and which the duplicates (test data) as well as which columns contain features and which target values.
We demonstrate in [Section B.1.2](#A2.SS1.SSS2 "B.1.2 Modified Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix") that even when we make these concessions, we can easily adapt the task such that both k-Nearest Neighbors and Deep Kernel Learning fail to solve it.
In fact, we are not aware of any other model that can solve the adapted task.

Additionally, we perform an *interventional* experiment to investigate the extent to which NPTs have actually learned the causal mechanism underlying the lookup task.
As illustrated in [Fig. 3](#S4.F3 "In 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")d, we now intervene on individual duplicate datapoints at test time by varying their target value across a wide range.
We stress that we perform these experiments without retraining the model, using exactly the same NPT from Figs. [3](#S4.F3 "Figure 3 ‣ 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")a-c.
The model is now confronted with target values associated with features that are highly unlikely under the training data.
This label distribution shift [[35](#bib.bib35)] is a challenging setting for neural networks.
However, NPT predictions follow the intervened target values with near-perfect correlation, [Fig. 3](#S4.F3 "In 4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments")e, continuing to predict by correctly looking up targets.

We now confidently conclude that NPTs robustly learn the causal data-generating mechanism underlying the semi-synthetic dataset.
This requires NPTs to *learn* a non-trivial sequence of compuational steps.
They must learn to match rows based on similarity of relevant features; to look up the target value of the duplicated datapoint; and, to copy that value into the target of the masked datapoint.

#### 4.3 NPTs Learn to Use Attention Between Datapoints on Real Data

Table 2: 
Drop in NPT performance after destroying information from other datapoints.
Shown are changes in test set performance, where negative values indicate worse performance after corruption.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ΔΔ\Delta Accuracy | CIFAR-10 | Poker | Income | Higgs | MNIST | Forest | Kick | Breast Cancer |
|  | −1.2-1.2-1.2 | −1.1-1.1-1.1 | −1.1-1.1-1.1 | −0.5-0.5-0.5 | −0.4-0.4-0.4 | −0.1-0.1-0.1 | −0.1-0.1-0.1 | 0.00.00.0 |
| Δ​RMSE/RMSE(%)\nicefrac{{\Delta\text{{RMSE}}}}{{\text{{RMSE}}}}\ (\%) | Yacht | Protein | Boston | Concrete |  |  |  |  |
|  | −52-52-52% | −21-21-21% | −20-20-20% | −7-7-7% |  |  |  |  |

We next consider (Q3):
do NPTs actually learn to use attention between datapoints for prediction on real data?
We design a test that allows us to quantify the extent to which the predictions of an NPT trained in standard fashion on one of our benchmark datasets depend on relationships between datapoints at test time.
Concretely, for each target value in the input we randomize the data for all *other* datapoints by independently shuffling each of their attributes across the rows.
We then evaluate the loss on the prediction at the target entry and repeat this procedure for all test datapoints.
This completely corrupts the information from all datapoints except the one for which we evaluate.
Hence, a model that relies meaningfully on attention between datapoints will show deteriorating performance.
We give an algorithm for the corruption procedure as well as further discussion in [Section B.2.1](#A2.SS2.SSS1 "B.2.1 Corruption Experiments ‣ B.2 Attention Between Datapoints on Real Data ‣ Appendix B Additional Results ‣ Appendix").

We report the resulting change in performance after corruption in [Table 2](#S4.T2 "In 4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments") for all datasets from §[4.1](#S4.SS1 "4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments").
We find that for most datasets, the corruption of other rows at test time significantly decreases the performance of the trained NPT models.
This indicates that the NPTs have successfully learned to make predictions supported by attention between datapoints.
For some datasets, the corruption experiment deteriorates performance completely.
For example, for the Protein regression dataset NPT achieves state-of-the-art performance, but corrupting the input at test time leads to NPT performing worse than all of the baselines considered in §[4.1](#S4.SS1 "4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments").
We note that minor differences in performance are often still significant, as differences between competing models in §[4.1](#S4.SS1 "4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments") are often likewise small.

Interestingly, on certain datasets such as Forest Cover, Kick, and Breast Cancer, corrupted inputs do not significantly affect performance.
It appears that when NPTs do not find it advantageous to rely on attention between datapoints during training, they can learn to completely ignore other inputs, essentially collapsing into a standard parametric model.
This supports our earlier claims that NPTs can learn end-to-end from data the extent to which they rely on other datapoints for prediction.
We think this is extremely interesting behavior and are unaware of prior work reporting similar results.
However, we stress that these results reflect inductive biases of the NPT architecture and do not lend themselves to general statements about the performance of parametric versus non-parametric models.

#### 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data

![Refer to caption](/html/2106.02584/assets/x4.png)


Figure 4: Fig. 4: Attention weights.

So far, we have presented convincing evidence that NPTs (sometimes strongly) depend on attention between datapoints.
However, we do not know what kind of interactions are learned in practice on real data (Q4).
As an initial step towards understanding this, we now present two experiments investigating *to which* other datapoints NPT attends.

Qualitative Evidence.
[Figure 4](#S4.F4 "In 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data ‣ 4 Experiments") shows an attention map for attention between datapoints (ABD) of NPT on a batch of the Protein regression dataset.
We sort the input data with respect to their input space distance such that similar datapoints are now close to each other.
The diagonal pattern in [Fig. 4](#S4.F4 "In 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data ‣ 4 Experiments") indicates that NPT attends more strongly to datapoints that are similar in feature space.
[Section B.3.1](#A2.SS3.SSS1 "B.3.1 Attention Maps on Real Data ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") discusses this further and gives additional attention maps.

Quantitative Evidence.
Seeking a quantitative measure for this hypothesis, the *data deletion* experiment repeats the following procedure for all test set points:
iteratively delete other datapoints from the input if they do not significantly affect the prediction.
We stop if less than 2%percent2$2$\% of the original datapoints remain, or if the total change in prediction for the target (relative to the original prediction with all data) exceeds 10%percent10$10$\%.
We investigate the average input feature space distances between the test point and the *kept* datapoints, as well as the distances between the test point and the *deleted* datapoints.
“Input features” here refer to all attributes of the input datapoints that are not labels.

We find that kept datapoints have a significantly lower average feature space distance to the test point than those deleted.
This indicates that two datapoints i,i′

𝑖superscript𝑖′i,i^{\prime} that are similar in input feature space, such that ∑j<d(Xi,j−Xi′,j)2subscript𝑗𝑑superscriptsubscript𝑋

𝑖𝑗subscript𝑋

superscript𝑖′𝑗2\sum\_{j<d}(X\_{i,j}-X\_{i^{\prime},j})^{2} is low, have a larger effect on the predictions of one another.
A Wilcoxon signed-rank test is significant at p≈8.77⋅10−130𝑝⋅8.77superscript10130\smash{p\approx 8.77\cdot 10^{-130}}.
We give full details on this in [Section B.3.2](#A2.SS3.SSS2 "B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix").

Both experiments support the hypothesis that NPTs rely on similar datapoints for prediction in real data settings.
One possible explanation is that similar datapoints might have different realizations of observation noise which NPTs could learn to average out.
Altogether, we conclude that NPTs can and do learn representations which rely on interactions between datapoints for prediction.

### 5 Limitations, Future Work, and Conclusions

###### Limitations.

NPTs share scaling limitations with all naïvely non-parametric approaches [[74](#bib.bib74)] and GNNs [[52](#bib.bib52)].
We demonstrate this in a preliminary analysis of the computational cost of NPTs and the baseline methods – including training time and CPU/GPU memory requirements – in [Section B.6](#A2.SS6 "B.6 Computational Cost of Non-Parametric Transformers ‣ Appendix B Additional Results ‣ Appendix").
While we have seen success with random minibatching (§[2.6](#S2.SS6 "2.6 Masking and Optimization ‣ 2 Non-Parametric Transformers")), future work might consider applying principled attention approximations, such as learning representative input points [[59](#bib.bib59)], kernelization [[47](#bib.bib47), [19](#bib.bib19)], or other sparsity-inducing methods [[84](#bib.bib84), [18](#bib.bib18), [5](#bib.bib5)], to improve the scalability of NPTs.

###### Future Work.

We believe that the unique predictive mechanism of NPTs makes them an interesting object of study for other tasks including continual learning, multi-task learning, few-shot generalization, and domain adaptation.
For example, when predicting under distribution shift, general relations between datapoints and attributes may remain valid and allow NPTs to accommodate such scenarios better.
Additionally, future work could explore the connections to stochastic processes, e.g., by extending NPTs to be approximately consistent, similar to Neural Processes [[37](#bib.bib37), [36](#bib.bib36), [49](#bib.bib49)].

###### Conclusions.

We have introduced Non-Parametric Transformers (NPTs), a novel deep learning architecture that takes the entire dataset as input and uses self-attention to model complex relationships *between* datapoints.
NPTs challenge and naturally extend parametric modeling as the dominant paradigm of deep learning.
They have the additional flexibility to learn to predict by directly attending to other datapoints.
Notably, NPTs learn this end-to-end from the data at hand.
Empirically, NPTs achieve highly competitive performance on a variety of benchmarks, and additional experiments demonstrate their ability to solve complex reasoning tasks over datapoints.
Further, we show that on real data, NPTs learn to rely on attention between datapoints for prediction.
We believe that the characteristics of NPTs will make them an exciting object of further study.

### Acknowledgments and Disclosure of Funding

We acknowledge funding from the New College Yeotown Scholarship (JK), the Rhodes Trust (NB), and the Open Philanthropy AI Fellowship (CL).
We thank Lewis Smith, Pascal Notin, Uri Shalit, Joost van Amersfoort, Sören Mindermann, Lood van Niekerk, and the anonymous reviewers for helpful feedback and interesting discussions that have led to numerous improvements of the paper.

### References

* Altman [1992]

  Naomi S Altman.
  An introduction to kernel and nearest-neighbor nonparametric
  regression.
  *The American Statistician*, 46, 1992.
* Arik and Pfister [2019]

  Sercan O Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  *arXiv:1908.07442*, 2019.
* Ba et al. [2016]

  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.
  Layer normalization.
  *arXiv:1607.06450*, 2016.
* Bahdanau et al. [2015]

  Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.
  Neural machine translation by jointly learning to align and
  translate.
  In *International Conference on Learning Representations*, 2015.
* Beltagy et al. [2020]

  Iz Beltagy, Matthew E. Peters, and Arman Cohan.
  Longformer: The long-document transformer.
  *arXiv:2004.05150*, 2020.
* Bengio et al. [1991]

  Y. Bengio, S. Bengio, and J. Cloutier.
  Learning a synaptic learning rule.
  In *International Joint Conference on Neural Networks*,
  volume 2, 1991.
* Bentley [1975]

  J. L. Bentley.
  Multidimensional binary search trees used for associative searching.
  In *Communications of the ACM*, volume 18, 1975.
* Biggs [1985]

  John B Biggs.
  The role of metalearning in study processes.
  *British journal of educational psychology*, 55, 1985.
* Breiman [1996]

  Leo Breiman.
  Bagging predictors.
  *Machine learning*, 24, 1996.
* Breiman [2001]

  Leo Breiman.
  Random forests.
  *Machine learning*, 45, 2001.
* Breiman et al. [1984]

  Leo Breiman, Jerome Friedman, Charles J Stone, and Richard A Olshen.
  *Classification and regression trees*.
  CRC press, 1984.
* Brown et al. [2020]

  Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
  Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
  et al.
  Language models are few-shot learners.
  *arXiv:2005.14165*, 2020.
* Bui et al. [2016]

  Thang Bui, Daniel Hernández-Lobato, Jose Hernandez-Lobato, Yingzhen Li, and
  Richard Turner.
  Deep gaussian processes for regression using approximate expectation
  propagation.
  In *International Conference on Machine Learning*, 2016.
* Carlini et al. [2020]

  Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel
  Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar
  Erlingsson, et al.
  Extracting training data from large language models.
  *arXiv:2012.07805*, 2020.
* Chapelle et al. [2009]

  Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien.
  Semi-supervised learning.
  *IEEE Transactions on Neural Networks*, 20(3):542–542, 2009.
* Chen et al. [2018]

  Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey,
  George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish
  Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and
  Macduff Hughes.
  The best of both worlds: Combining recent advances in neural machine
  translation.
  In *Annual Meeting of the Association for Computational
  Linguistics*, volume 56, 2018.
* Chen and Guestrin [2016]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *Knowledge Discovery and Data Mining*, volume 22, 2016.
* Child et al. [2019]

  Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.
  Generating long sequences with sparse transformers.
  *arXiv:1904.10509*, 2019.
* Choromanski et al. [2021]

  Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
  Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz
  Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian
  Weller.
  Rethinking attention with performers.
  In *International Conference on Learning Representations*, 2021.
* Chui et al. [2018]

  Michael Chui, James Manyika, Mehdi Miremadi, Nicolaus Henke, Rita Chung, Pieter
  Nel, and Sankalp Malhotra.
  Notes from the AI frontier: Insights from hundreds of use cases,
  2018.
* Dai et al. [2016]

  Zhenwen Dai, Andreas Damianou, Javier González, and Neil Lawrence.
  Variational auto-encoded deep gaussian processes.
  In *International Conference on Learning Representations*, 2016.
* Damianou and Lawrence [2013]

  Andreas Damianou and Neil D Lawrence.
  Deep gaussian processes.
  In *International Conference on Artificial Intelligence and
  Statistics*, volume 16, 2013.
* Deng et al. [2009]

  Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.
  Imagenet: A large-scale hierarchical image database.
  In *Conference on Computer Vision and Pattern Recognition*,
  2009.
* Devlin et al. [2018]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  *arXiv:1810.04805*, 2018.
* Dosovitskiy et al. [2021]

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
  Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
  Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *International Conference on Learning Representations*, 2021.
* Dua and Graff [2017]

  Dheeru Dua and Casey Graff.
  UCI machine learning repository, 2017.
  URL <http://archive.ics.uci.edu/ml>.
* Erhan et al. [2010]

  Dumitru Erhan, Aaron Courville, Yoshua Bengio, and Pascal Vincent.
  Why does unsupervised pre-training help deep learning?
  In *International Conference on Artificial Intelligence and
  Statistics*, volume 13, pages 201–208, 2010.
* Filos et al. [2019]

  Angelos Filos, Sebastian Farquhar, Aidan N Gomez, Tim GJ Rudner, Zachary
  Kenton, Lewis Smith, Milad Alizadeh, Arnoud De Kroon, and Yarin Gal.
  A systematic comparison of bayesian deep learning robustness in
  diabetic retinopathy tasks.
  In *NeurIPS Workshop on Bayesian Deep Learning*, 2019.
* Finn et al. [2017]

  Chelsea Finn, Pieter Abbeel, and Sergey Levine.
  Model-agnostic meta-learning for fast adaptation of deep networks.
  In *International Conference on Machine Learning*, volume 34,
  2017.
* Fix [1985]

  Evelyn Fix.
  *Discriminatory analysis: nonparametric discrimination,
  consistency properties*, volume 1.
  USAF school of Aviation Medicine, 1985.
* Freund and Schapire [1997]

  Yoav Freund and Robert E Schapire.
  A decision-theoretic generalization of on-line learning and an
  application to boosting.
  *Journal of computer and system sciences*, 55, 1997.
* Friedman [2001]

  Jerome H Friedman.
  Greedy function approximation: a gradient boosting machine.
  *Annals of statistics*, 2001.
* Fuchs et al. [2020]

  Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling.
  Se(3)-transformers: 3d roto-translation equivariant attention
  networks.
  In *Advances in Neural Information Processing Systems*,
  volume 33, 2020.
* Garcia and Bruna [2018]

  Victor Garcia and Joan Bruna.
  Few-shot learning with graph neural networks.
  In *International Conference on Learning Representations*, 2018.
* Garg et al. [2020]

  Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, and Zachary Lipton.
  A unified view of label shift estimation.
  In *Advances in Neural Information Processing Systems*,
  volume 33, 2020.
* Garnelo et al. [2018a]

  Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David
  Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami.
  Conditional neural processes.
  In *International Conference on Machine Learning*, volume 35,
  2018a.
* Garnelo et al. [2018b]

  Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende,
  SM Eslami, and Yee Whye Teh.
  Neural processes.
  *arXiv:1807.01622*, 2018b.
* Guu et al. [2018]

  Kelvin Guu, Tatsunori B Hashimoto, Yonatan Oren, and Percy Liang.
  Generating sentences by editing prototypes.
  *Transactions of the Association for Computational Linguistics*,
  6:437–450, 2018.
* Guu et al. [2020]

  Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
  Realm: Retrieval-augmented language model pre-training.
  *arXiv:2002.08909*, 2020.
* Harris et al. [2020]

  Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf
  Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor,
  Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer,
  Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández
  del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant,
  Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph
  Gohlke, and Travis E. Oliphant.
  Array programming with NumPy.
  *Nature*, 585, 2020.
* Hashimoto et al. [2018]

  Tatsunori B Hashimoto, Kelvin Guu, Yonatan Oren, and Percy Liang.
  A retrieve-and-edit framework for predicting structured outputs.
  In *Advances in neural information processing systems*, 2018.
* Ho et al. [2019]

  Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans.
  Axial attention in multidimensional transformers.
  *arXiv:1912.12180*, 2019.
* Honaker and King [2010]

  James Honaker and Gary King.
  What to do about missing values in time series cross-section data.
  *American Journal of Political Science*, 2010.
* Hutchinson et al. [2020]

  Michael Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye
  Teh, and Hyunjik Kim.
  Lietransformer: Equivariant self-attention for lie groups.
  *arXiv:2012.10885*, 2020.
* Inc. [2021]

  Google Inc.
  Kaggle.
  *https://www.kaggle.com/*, 2021.
* Jaegle et al. [2021]

  Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and
  Joao Carreira.
  Perceiver: General perception with iterative attention.
  *arXiv:2103.03206*, 2021.
* Katharopoulos et al. [2020]

  Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
  Fleuret.
  Transformers are RNNs: Fast autoregressive transformers with linear
  attention.
  In *International Conference on Machine Learning*, volume 37,
  2020.
* Ke et al. [2017]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei
  Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In *Advances in neural information processing systems*,
  volume 30, 2017.
* Kim et al. [2019]

  Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan
  Rosenbaum, Oriol Vinyals, and Yee Whye Teh.
  Attentive neural processes.
  In *International Conference on Learning Representations*, 2019.
* King et al. [2001]

  Gary King, James Honaker, Anne Joseph, and Kenneth Scheve.
  Analyzing incomplete political science data: An alternative algorithm
  for multiple imputation.
  *American Political Science Review*, 2001.
* Kingma et al. [2014]

  Diederik P Kingma, Danilo J Rezende, Shakir Mohamed, and Max Welling.
  Semi-supervised learning with deep generative models.
  *arXiv:1406.5298*, 2014.
* Kipf and Welling [2017]

  Thomas Kipf and Max Welling.
  Semi-supervised classification with graph convolutional networks.
  In *International Conference on Learning Representations*, 2017.
* Kipf et al. [2018]

  Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel.
  Neural relational inference for interacting systems.
  In *International Conference on Machine Learning*, volume 35,
  2018.
* Kolesnikov et al. [2020]

  Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung,
  Sylvain Gelly, and Neil Houlsby.
  Big transfer (bit): General visual representation learning.
  In *European Conference on Computer Vision*, 2020.
* Krizhevsky et al. [2009]

  Alex Krizhevsky, Geoffrey Hinton, et al.
  Learning multiple layers of features from tiny images, 2009.
* Lake et al. [2015]

  Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum.
  Human-level concept learning through probabilistic program induction.
  *Science*, 350, 2015.
* LeCun et al. [1998]

  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86, 1998.
* LeCun et al. [2010]

  Yann LeCun, Corinna Cortes, and CJ Burges.
  Mnist handwritten digit database.
  *ATT Labs [Online]*, 2, 2010.
* Lee et al. [2019]

  Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye
  Teh.
  Set transformer: A framework for attention-based
  permutation-invariant neural networks.
  In *International Conference on Machine Learning*, volume 36,
  2019.
* Liu et al. [2006]

  T. Liu, A. Moore, and A. Gray.
  New algorithms for efficient high-dimensional nonparametric
  classification.
  In *Journal of Machine Learning Research*, volume 7, 2006.
* Locatello et al. [2020]

  Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran,
  Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf.
  Object-centric learning with slot attention.
  In *Advances in Neural Information Processing Systems*,
  volume 33, 2020.
* Loh [2014]

  Wei-Yin Loh.
  Fifty years of classification and regression trees.
  *International Statistical Review*, 82, 2014.
* Michelmore et al. [2018]

  Rhiannon Michelmore, Marta Kwiatkowska, and Yarin Gal.
  Evaluating uncertainty quantification in end-to-end autonomous
  driving control.
  *arXiv:1811.06817*, 2018.
* Morgan and Sonquist [1963]

  James N Morgan and John A Sonquist.
  Problems in the analysis of survey data, and a proposal.
  *Journal of the American statistical association*, 58, 1963.
* Movshovitz-Attias et al. [2017]

  Yair Movshovitz-Attias, Alexander Toshev, Thomas K Leung, Sergey Ioffe, and
  Saurabh Singh.
  No fuss distance metric learning using proxies.
  In *International Conference on Computer Vision*, pages
  360–368, 2017.
* Narang et al. [2021]

  Sharan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Févry,
  Michael Matena, Karishma Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan,
  Yanqi Zhou, Wei Li, Nan Ding, Jake Marcus, Adam Roberts, and Colin Raffel.
  Do transformer modifications transfer across implementations and
  applications?
  *arXiv:2102.11972*, 2021.
* Parmar et al. [2018]

  Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer,
  Alexander Ku, and Dustin Tran.
  Image transformer.
  In *International Conference on Machine Learning*, volume 35,
  2018.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
  Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan
  Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
  Chintala.
  Pytorch: An imperative style, high-performance deep learning library.
  In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural
  Information Processing Systems*, volume 32, 2019.
* Pedregosa et al. [2011]

  Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
  Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron
  Weiss, Vincent Dubourg, et al.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12, 2011.
* Platform [2021]

  Google Cloud AI Platform.
  Getting started with the built-in tabnet algorithm, 2021.
  URL
  <cloud.google.com/ai-platform/training/docs/algorithms/tab-net-start>.
* Prokhorenkova et al. [2018]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
  and R. Garnett, editors, *Advances in Neural Information Processing
  Systems*, volume 31, 2018.
* Radford et al. [2019]

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
  Sutskever.
  Language models are unsupervised multitask learners.
  *OpenAI blog*, 2019.
* Rao et al. [2021]

  Roshan Rao, Jason Liu, Robert Verkuil, Joshua Meier, John F Canny, Pieter
  Abbeel, Tom Sercu, and Alexander Rives.
  Msa transformer.
  *bioRxiv*, 2021.
* Rasmussen [2003]

  Carl Edward Rasmussen.
  Gaussian processes in machine learning.
  In *Summer school on machine learning*, 2003.
* Ridnik et al. [2021]

  Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor.
  Imagenet-21k pretraining for the masses.
  *arXiv:2104.10972*, 2021.
* Roweis et al. [2004]

  Sam Roweis, Geoffrey Hinton, and Ruslan Salakhutdinov.
  Neighbourhood component analysis.
  In *Advances in Neural Information Processing Systems*,
  volume 17, page 4, 2004.
* Salimbeni and Deisenroth [2017]

  Hugh Salimbeni and Marc Deisenroth.
  Doubly stochastic variational inference for deep gaussian processes.
  In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett, editors, *Advances in Neural
  Information Processing Systems*, volume 30, 2017.
* Schapire [1990]

  Robert E Schapire.
  The strength of weak learnability.
  *Machine learning*, 5, 1990.
* Seidenschwarz et al. [2021]

  Jenny Seidenschwarz, Ismail Elezi, and Laura Leal-Taixé.
  Learning intra-batch connections for deep metric learning.
  In *International Conference on Machine Learning*, 2021.
* Song et al. [2019]

  Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and
  Jian Tang.
  Autoint: Automatic feature interaction learning via self-attentive
  neural networks.
  In *Proceedings of the 28th ACM International Conference on
  Information and Knowledge Management*, 2019.
* Stekhoven and Buehlmann [2012]

  D.J. Stekhoven and P. Buehlmann.
  Missforest - nonparametric missing value imputation for mixed-type
  data.
  *Bioinformatics*, 2012.
* Su et al. [2012]

  Yu-Sung Su, Andrew E. Gelman, Jennifer Hill, and Masanao Yajima.
  Multiple imputation with diagnostics (mi) in R: Opening windows
  into the black box.
  *Journal of Statistical Software*, 2012.
* Sun et al. [2017]

  C. Sun, A. Shrivastava, S. Singh, and A. Gupta.
  Revisiting unreasonable effectiveness of data in deep learning era.
  In *2017 IEEE International Conference on Computer Vision
  (ICCV)*, 2017.
* Tay et al. [2020]

  Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler.
  Efficient transformers: A survey.
  *arXiv:2009.06732*, 2020.
* Touvron et al. [2019]

  Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou.
  Fixing the train-test resolution discrepancy.
  In *Advances in Neural Information Processing Systems*,
  volume 32, 2019.
* Touvron et al. [2020]

  Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre
  Sablayrolles, and Hervé Jégou.
  Training data-efficient image transformers & distillation through
  attention.
  *arXiv:2012.12877*, 2020.
* Van Aken et al. [2018]

  Betty Van Aken, Julian Risch, Ralf Krestel, and Alexander Löser.
  Challenges for toxic comment classification: An in-depth error
  analysis.
  *arXiv preprint arXiv:1809.07572*, 2018.
* van Buuren and Groothuis-Oudshoorn [2011]

  Stef van Buuren and Karin Groothuis-Oudshoorn.
  mice: Multivariate imputation by chained equations in r.
  *Journal of Statistical Software*, 2011.
* Vapnik [2006]

  Vladimir Vapnik.
  *Estimation of dependences based on empirical data*.
  Springer Science & Business Media, 2006.
* Vaswani et al. [2017]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *Advances in Neural Information Processing Systems*,
  volume 30, 2017.
* Veličković et al. [2018]

  Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
  Pietro Lio, and Yoshua Bengio.
  Graph attention networks.
  In *International Conference on Learning Representations*, 2018.
* Vijayakumar and Schaal [1997]

  Sethu Vijayakumar and Stefan Schaal.
  Local dimensionality reduction for locally weighted learning.
  In *International Symposium on Computational Intelligence in
  Robotics and Automation*, pages 220–225. IEEE, 1997.
* Vinyals et al. [2016]

  Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al.
  Matching networks for one shot learning.
  In *Advances in neural information processing systems*,
  volume 29, pages 3630–3638, 2016.
* Wang et al. [2019]

  Xinshao Wang, Yang Hua, Elyor Kodirov, Guosheng Hu, Romain Garnier, and Neil M
  Robertson.
  Ranked list loss for deep metric learning.
  In *Conference on Computer Vision and Pattern Recognition*,
  pages 5207–5216, 2019.
* Wilson et al. [2016]

  Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing.
  Deep kernel learning.
  In *International Conference on Artificial Intelligence and
  Statistics*, volume 19, 2016.
* Xu et al. [2019]

  Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.
  How powerful are graph neural networks?
  In *International Conference on Learning Representations*, 2019.
* Yoon et al. [2018]

  Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin
  Ahn.
  Bayesian model-agnostic meta-learning.
  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
  and R. Garnett, editors, *Advances in Neural Information Processing
  Systems*, volume 31, 2018.
* You et al. [2020]

  Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh
  Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh.
  Large batch optimization for deep learning: Training bert in 76
  minutes.
  In *International Conference on Learning Representations*, 2020.
* Zhang et al. [2019]

  Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton.
  Lookahead optimizer: k steps forward, 1 step back.
  In *Advances in Neural Information Processing Systems*,
  volume 32, 2019.

Self-Attention Between Datapoints: Going Beyond
  
 Individual Input-Output Pairs in Deep Learning

## Appendix

\parttoc

### Appendix A Proof – NPT Is Equivariant over Datapoints

We here provide proof that NPT is equivariant to a permutation of the datapoints.
This requires, among other things, showing that multi-head self-attention is equivariant.
We were unable to find this proof in the existing literature, e.g., Set Transformer [[59](#bib.bib59)] relies heavily on equivariance of self-attention but does not provide proof.
In the following, we will refer to datapoints as the *rows* of our input, see e.g., [Fig. 1](#S1.F1 "In Background. ‣ 1 Introduction").

###### Definition 1.

A function f:𝒳n→𝒳n:𝑓→superscript𝒳𝑛superscript𝒳𝑛f:\mathcal{X}^{n}\rightarrow\mathcal{X}^{n} is row-equivariant if for any permutation σ:[1,…,n]→[1,…,n]:𝜎→

1…𝑛

1…𝑛\sigma:[1,\dots,n]\rightarrow[1,\dots,n] applied to the dimensions of 𝒳nsuperscript𝒳𝑛\mathcal{X}^{n}, we have for all i𝑖i, f​(X1,…,Xn)​[i]=f​(Xσ−1​(1),…,Xσ−1​(n))​[σ​(i)]𝑓subscript𝑋1…subscript𝑋𝑛delimited-[]𝑖𝑓subscript𝑋superscript𝜎11…subscript𝑋superscript𝜎1𝑛delimited-[]𝜎𝑖f(X\_{1},\dots,X\_{n})[i]=f(X\_{\sigma^{-1}(1)},\dots,X\_{\sigma^{-1}(n)})[\sigma(i)].

###### Lemma 1.

Any function of the form f​(X1,…,Xn)=(g​(X1),…,g​(Xn))𝑓subscript𝑋1…subscript𝑋𝑛𝑔subscript𝑋1…𝑔subscript𝑋𝑛f(X\_{1},\dots,X\_{n})=(g(X\_{1}),\dots,g(X\_{n})) for some g𝑔g is row-equivariant. These functions are denoted as ‘row-wise operations’, as they consist of the same function applied to each of the rows of the input.

###### Proof.

Follows immediately from the structure of f𝑓f.
∎

###### Lemma 2.

The composition of row-equivariant functions is row-equivariant.

###### Proof.

This result is widely known, but a proof here is included for completeness. Let f𝑓f and g𝑔g be row-equivariant.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | f∘g​(σ​X)𝑓𝑔𝜎𝑋\displaystyle f\circ g(\sigma X) | =f​(g​(σ​X))=f​(σ​g​(X))=σ​f​(g​(X)).absent𝑓𝑔𝜎𝑋𝑓𝜎𝑔𝑋𝜎𝑓𝑔𝑋\displaystyle=f(g(\sigma X))=f(\sigma g(X))=\sigma f(g(X)). |  | (8) |

∎

###### Lemma 3.

Let W∈ℝn×m1𝑊superscriptℝ𝑛subscript𝑚1W\in\mathbb{R}^{n\times m\_{1}} and X∈ℝm2×n𝑋superscriptℝsubscript𝑚2𝑛X\in\mathbb{R}^{m\_{2}\times n}. The function X↦X​Wmaps-to𝑋𝑋𝑊X\mapsto XW is row-equivariant.

###### Proof.

Let σ​X𝜎𝑋\sigma X be a permutation of the rows of X𝑋X. Then we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | (σ​X)​W​[i,j]𝜎𝑋𝑊𝑖𝑗\displaystyle(\sigma X)W[i,j] | =∑σ​X​[i,k]​W​[k,j]absent𝜎𝑋𝑖𝑘𝑊𝑘𝑗\displaystyle=\sum\sigma X[i,k]W[k,j] |  | (9) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑X​[σ−1​(i),k]​W​[k,j]=X​W​[σ−1​(i),j]=σ​(X​W)​[i,j].absent𝑋superscript𝜎1𝑖𝑘𝑊𝑘𝑗𝑋𝑊superscript𝜎1𝑖𝑗𝜎𝑋𝑊𝑖𝑗\displaystyle=\sum X[\sigma^{-1}(i),k]W[k,j]=XW[\sigma^{-1}(i),j]=\sigma(XW)[i,j]. |  | (10) |

∎

###### Lemma 4.

The function X↦A​t​t​(X​WQ,X​WK,X​WV)maps-to𝑋𝐴𝑡𝑡𝑋superscript𝑊𝑄𝑋superscript𝑊𝐾𝑋superscript𝑊𝑉X\mapsto Att(XW^{Q},XW^{K},XW^{V}) is row-equivariant.

###### Proof.

Let the row-wise softmax function be denoted ω​(⋅)𝜔⋅\omega(\cdot). Then we have

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | A​t​t​(X​WQ,X​WK,X​WV)𝐴𝑡𝑡𝑋superscript𝑊𝑄𝑋superscript𝑊𝐾𝑋superscript𝑊𝑉\displaystyle Att(XW^{Q},XW^{K},XW^{V}) | =ω​(X​WQ​(X​WK)⊤/h)​X​WV,absent𝜔𝑋superscript𝑊𝑄superscript𝑋superscript𝑊𝐾topℎ𝑋superscript𝑊𝑉\displaystyle=\omega(XW^{Q}(XW^{K})^{\top}/\sqrt{h})XW^{V}, |  | (11) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| where | | | | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | σ​X​WQ​(σ​X​WK)⊤​[i,j]𝜎𝑋superscript𝑊𝑄superscript𝜎𝑋superscript𝑊𝐾top𝑖𝑗\displaystyle\sigma XW^{Q}(\sigma XW^{K})^{\top}[i,j] | =σ​(X​WQ)​σ​(X​WK)⊤​[i,j]absent𝜎𝑋superscript𝑊𝑄𝜎superscript𝑋superscript𝑊𝐾top𝑖𝑗\displaystyle=\sigma(XW^{Q})\sigma(XW^{K})^{\top}[i,j] |  | (12) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑σ​(X​WQ)​[i,k]​σ​(X​WK)​[j,k]absent𝜎𝑋superscript𝑊𝑄𝑖𝑘𝜎𝑋superscript𝑊𝐾𝑗𝑘\displaystyle=\sum\sigma(XW^{Q})[i,k]\sigma(XW^{K})[j,k] |  | (13) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑X​WQ​[σ−1​(i),k]​X​WK​[σ−1​(j),k]absent𝑋superscript𝑊𝑄superscript𝜎1𝑖𝑘𝑋superscript𝑊𝐾superscript𝜎1𝑗𝑘\displaystyle=\sum XW^{Q}[\sigma^{-1}(i),k]XW^{K}[\sigma^{-1}(j),k] |  | (14) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =X​WQ​(X​WK)⊤​[σ−1​(i),σ−1​(j)]absent𝑋superscript𝑊𝑄superscript𝑋superscript𝑊𝐾topsuperscript𝜎1𝑖superscript𝜎1𝑗\displaystyle=XW^{Q}(XW^{K})^{\top}[\sigma^{-1}(i),\sigma^{-1}(j)] |  | (15) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =:A.\displaystyle=:A. |  | (16) |

Note that the above result states that the function X​WQ​(X​WK)⊤𝑋superscript𝑊𝑄superscript𝑋superscript𝑊𝐾topXW^{Q}(XW^{K})^{\top} is not row-equivariant because of the additional permutation of the columns. Let σ𝜎\sigma denote a permutation operator on matrices. Then straightforwardly we have the following:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ω(σA/h))\displaystyle\omega(\sigma A/\sqrt{h})) | =σ​ω​(A/h).absent𝜎𝜔𝐴ℎ\displaystyle=\sigma\omega(A/\sqrt{h})\,. |  | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Finally, it remains to show that the final matrix multiplication step restores the row-equivariance property we seek. | | | | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | σ​ω​(X​WQ​(X​WK)⊤/h)⏟=⁣:M​(σ​X​WV)​[i,j]𝜎subscript⏟𝜔𝑋superscript𝑊𝑄superscript𝑋superscript𝑊𝐾topℎ  :absent𝑀𝜎𝑋superscript𝑊𝑉𝑖𝑗\displaystyle\sigma\underbrace{\omega(XW^{Q}(XW^{K})^{\top}/\sqrt{h})}\_{=:M}(\sigma XW^{V})[i,j] | =σ​(M)​(σ​X​WV)​[i,j]absent𝜎𝑀𝜎𝑋superscript𝑊𝑉𝑖𝑗\displaystyle=\sigma(M)(\sigma XW^{V})[i,j] |  | (18) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =σ​(M)​σ​(X​WV)​[i,j]absent𝜎𝑀𝜎𝑋superscript𝑊𝑉𝑖𝑗\displaystyle=\sigma(M)\sigma(XW^{V})[i,j] |  | (19) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =∑M​[σ−1​(i),σ−1​(k)]​(X​WV)​[σ−1​(k),j]absent𝑀superscript𝜎1𝑖superscript𝜎1𝑘𝑋superscript𝑊𝑉superscript𝜎1𝑘𝑗\displaystyle=\sum M[\sigma^{-1}(i),\sigma^{-1}(k)](XW^{V})[\sigma^{-1}(k),j] |  | (20) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | =M​(X​WV)​[σ−1​(i),j].absent𝑀𝑋superscript𝑊𝑉superscript𝜎1𝑖𝑗\displaystyle=M(XW^{V})[\sigma^{-1}(i),j]. |  | (21) |

Which shows that self-attention is row-equivariant.
∎

###### Lemma 5.

The following hold:

1. 1.

   Multihead self-attention is equivariant.
2. 2.

   If f𝑓f and g𝑔g are row-equivariant, then the function x↦g​(x)+f​(x)maps-to𝑥𝑔𝑥𝑓𝑥x\mapsto g(x)+f(x) is also row-equivariant.
3. 3.

   Res(H) is row-equivariant.
4. 4.

   MHSA(H) is row-equivariant.
5. 5.

   ABD is row-equivariant.
6. 6.

   ABA is row-equivariant.

###### Proof.

We show each item.

1. 1.

   We know that X↦Oimaps-to𝑋subscript𝑂𝑖X\mapsto O\_{i} is equivariant from the previous lemma, and this trivially implies that X↦concat​(O1,…,Ok)maps-to𝑋concatsubscript𝑂1…subscript𝑂𝑘X\mapsto\text{concat}(O\_{1},\dots,O\_{k}) will also be row-equivariant. Finally, because σ​A​B=σ​(A​B)𝜎𝐴𝐵𝜎𝐴𝐵\sigma AB=\sigma(AB), get that MHSelfAtt(H) is row-equivariant.
2. 2.

   Straightforward.
3. 3.

   Because LayerNorm is row-equivariant (being a function applied row-wise to the matrix), Res(H) is a sum of two row-equivariant functions and so by a previous result will also be row-equivariant.
4. 4.

   Because rFF is again a row-wise operation and so trivially row-equivariant, the previous results on sums and compositions of row-equivariant functions directly yield row-equivariance of MHSA.
5. 5.

   ABD is by definition an application of MHSA(H), and therefore is row-equivariant by the above result.
6. 6.

   ABA is a row-wise operation and is therefore trivially row-equivariant.

∎

###### Property A.0.1.

NPT is row-equivariant.

###### Proof.

Each layer of NPT has been shown to be row-equivariant. Because NPT is a composition of such row-equivariant functions, it is therefore row-equivariant.
∎

### Appendix B Additional Results

#### B.1 Semi-Synthetic Experiments

##### B.1.1 Attention Maps for the Semi-Synthetic Experiments

We here display additional results for the semi-synthetic experiments of [Section 4.2](#S4.SS2 "4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments").
In [Fig. B.1](#A2.F1 "In B.1.1 Attention Maps for the Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix"), we display attention weights for Attention Between Datapoints (ABD) for all depths and a subset of heads of the architecture.
We see that some, but not all, attention heads display the desired diagonal lookup pattern.
Note that, in this case, one head would suffice to implement lookup and perfectly solve the task.

A brief comment on the attention maps with the “double diagonal” structure (e.g., depth 4, head 0):
we see that (a) original datapoints attend to the duplicate points and (b) duplicates also attend to duplicate datapoints.
Behavior (a) makes sense: NPT needs to attend to the duplicates from the originals to look up the target values.
This behavior in turn minimizes loss.
Behavior (b) is irrelevant to loss, because NPT does not need to predict anything for the duplicates, and no loss is computed.
However, (b) suggests that the query embeddings learned by the self-attention *ignore* the masked out label column in the input.
Hence, the resulting queries for the originals and the duplicates would be identical – both leading to high attention values for the keys of the duplicates – and ultimately resulting in the double diagonals in [Fig. B.1](#A2.F1 "In B.1.1 Attention Maps for the Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix").

![Refer to caption](/html/2106.02584/assets/x5.png)


Figure B.1: 
Visualizations of NPT attention maps for Attention Between Datapoints (ABD) for the semi-synthetic experiment at all model depths, a selection of heads, and a single batch of input data.
Evidently, not all attention maps need to perform a “lookup” for the model to solve the task.
In fact, some heads appear to learn almost query-independent behavior (e.g., heads 0, 1, and 2 at depth 0).

##### B.1.2 Modified Semi-Synthetic Experiments

Table 3: 
Variations of the semi-synthetic dataset that require learning of between-datapoint interactions more complex than simple lookups.
While NPTs can learn complex interactions between datapoints, conventional non-parametric approaches lack flexibility and fail.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Test RMSE ↓↓\downarrow | Original Synthetic | Random Feats. | Add One | Random Feats. + Add One |
| 1-NN | 0.000.000.00 | 7.197.197.19 | 6.116.116.11 | 7.807.807.80 |
| k-NN | 0.000.000.00 | 5.425.425.42 | 5.185.185.18 | 5.645.645.64 |
| DKL | 0.000.000.00 | 5.945.945.94 | 6.316.316.31 | 6.366.366.36 |
| NPT | 0.340.340.34 | 0.240.240.24 | 0.460.460.46 | 0.750.750.75 |

###### Setup.

In [Section 4.2](#S4.SS2 "4.2 NPTs Can Learn to Predict Using Attention Between Datapoints ‣ 4 Experiments"), we mention that with some concessions the original lookup task can also be solved by standard non-parametric models.
However, we also mention that simple modifications to the task make it, again, unsolvable for any model of which we are aware other than NPT.
We here demonstrate these hypotheses for two non-parametric models: k-Nearest Neighbors (k-NN) and Deep Kernel Learning (DKL).

First, we apply k-NN and DKL to the original duplication tasks.
As mentioned in the main text, this already requires us to make some concessions:
we now need to explicitly split the input data into a global training set (all duplicated datapoints) as well as a test set (all original datapoints).
That is, if all duplicate datapoints make up the training set, then non-parametric models are able to predict perfectly on the original datapoints, because most non-parametric models rely on distances in some manner, and here, distances in input feature space are sufficient to successfully match entries.
This is trivially true for k-NN but also for DKL, where the RBF kernel of the GP will lead to the desired “matching behavior” as long as the learned neural network embedding does not collapse distances.

In other words, NPTs would ideally learn a k-NN-style prediction for the semi-synthetic dataset.
Crucially, while non-parametric models predict based on distances because of fixed design choices, NPTs *learn* this behavior and can just as well learn other more complicated relations between datapoints.

We now present two modifications to the semi-synthetic dataset; NPT can accommodate them because the model learns the nature of interactions, but they significantly affect the performance of the fixed kernel methods.

* •

  Random Features: A subset of the features are randomized across both original and duplicate datapoints independently.
  Specifically, we overwrite the entries of the last three features with noise drawn independently from a Gaussian distribution 𝒩​(1,1)𝒩11\mathcal{N}(1,1).
  To solve the task, matches between datapoints must now be computed using the subset of non-randomized features only.
* •

  Add One: We add 111 to all target regression values *only* for the duplicate datapoints.
  Matches can still be made based on all features, but now a 111 must be subtracted from the lookup value to solve the task.

As in the original setting, we train the models on the modified semi-synthetic datasets and check with novel test data whether they have learnt the correct relational mechanism underlying the experiment.

Note that the Random Features and Add One settings also distinguish our setup from prompting in natural language processing literature [[72](#bib.bib72), [12](#bib.bib12)] because the original datapoints are no longer “correct” input-output pairs; the model must use an underlying relational structure instead of memorization to solve the task.

###### Results.

[Table 3](#A2.T3 "In B.1.2 Modified Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix") presents RMSE values obtained by the models when trained on the original duplication task, the two modifications separately, as well as both modifications applied.

Evidently, for NPTs, the different scenarios do not lead to a large difference in performance; in all instances, they achieve near-perfect loss because their predictions leverage attention between datapoints.
Careful optimization of NPT training convergence would likely lead to a further reduction in loss.
Nevertheless, the achieved losses by NPT are more than a magnitude lower than those on the original data and correspond to a near-perfect Pearson-correlation with the target values of r>99.9%𝑟percent99.9r>99.9\%.
We conclude that NPTs successfully learn to attend to the correct subset of features, to subtract 111 from the lookup target values, or to do both at the same time.

Next, we consider the non-parametric models.
First, we confirm in *Original Synthetic* that the non-parametric models can indeed solve the original lookup task. However, we find that neither DKL nor k-NN can accommodate any of the modifications, reverting to an RMSE that is worse than the performance of all baselines on the original Protein dataset, see [Table 11](#A2.T11 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix").444
In fact, the RMSEs are about equal to the standard deviations of the target values in the Protein dataset, 6.116.116.11, such that the values obtained by the models on the modified setups amount to random guessing.
We further note that we apply all modifications to the standardized input data, such that the Add One setting adds a full standard deviation for the final evaluation in [Table 3](#A2.T3 "In B.1.2 Modified Semi-Synthetic Experiments ‣ B.1 Semi-Synthetic Experiments ‣ Appendix B Additional Results ‣ Appendix").

For k𝑘k-Nearest Neighbor, k=1𝑘1k=1 is clearly optimal in the original semi-synthetic setup.
However, k-NN cannot learn to ignore certain attributes (Random Features) and or to modify looked-up values.
Setting k>1𝑘1k>1 actually improves prediction because it considers other matching points in addition to the (now misleading) duplicates for prediction.
However, even with k>1𝑘1k>1, k-NN does not achieve much better than guessing performance on the modified tasks.

DKL also fails to accommodate any of the presented task modifications.
We suspect that DKL, in theory, should be able to solve the Random Features task.
That is, DKL should be able to use the neural network to learn a representation that discards any information from the randomized columns.
We were unable to achieve this, but it may be possible with additional adaptations to the model.
Ideally, we would condition the GP on new “test data” (the duplicates) in each minibatch during training.
This was not easily possible with the GPyTorch codebase.555
Gardner, Jacob R., et al. ”Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration.” NeurIPS 2018.
 At test time however, we did directly reconstruct an exact GP using embedded inputs and RBF scale parameters learned during training.

In any case, DKL can never solve the Add One scenario because, after independently transforming features with a neural network, DKL simply applies a GP in embedding space.
This means that it will always naively interpolate target values between training data (duplicates) and test data (features) in embedding space, and cannot *learn* interactions between points, such as subtracting 1 from all duplicate targets.

Even further, there is another easy option of how to construct this experiment such that only NPT will be able to solve it: we could *randomly sample the attribute* for which we mask out the entry, i.e., all columns can now be target columns.
All non-parametric models presented here rely on a fixed set of features as input to predict for a fixed target column.
They are not compatible with this style of “imputation” problem, i.e., there is no way to even take as input data like this in such models.
NPTs, however, take both features and targets as input, only using the masking mechanism to distinguish between features and targets as well as train and test data.
Hence, they can easily adapt to this scenario.

The bad results for the non-parametric models also highlight that these models must predict non-parametrically, unlike NPT, which could always fall back to parametric prediction if it cannot learn the interactions required for a task.

(k)-NN Hyperparameter details. We use the scikit-learn [[69](#bib.bib69)] implementation of (k)-Nearest Neighbors, where we exhaustively search for neighbors by setting algorithm=brute and otherwise use default parameters.
For 111-NN, we set k=1𝑘1k=1, for k𝑘k-NN we sweep over k∈[1,…,10]𝑘

1…10k\in[1,\dots,10] and report results for the k𝑘k that achieved the best performance.

DKL Hyperparameter details. We use the GPyTorch implementation of Deep Kernel Learning.
We perform a non-exhaustive random sweep over a selection of hyperparameters and select those with best validation performance.
This results in the following changes from the default hyperparameter values:
for the Original Synthetic and Add One scenario we disable dropout, use hidden layers [100,100]100100[100,100], a learning rate of 0.00010.00010.0001, train for a maximum of 300003000030000 epochs, with 256256256 inducing points, 888 features, batch size of 128128128, and early stopping patience on the validation loss of 202020 epochs.
For the Random Features and the Random Features + Add One scenarios, we arrive at the same configuration, except that we train with 646464 inducing points.

#### B.2 Attention Between Datapoints on Real Data

##### B.2.1 Corruption Experiments

In our Data Corruption experiments in [Section 4.3](#S4.SS3 "4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments"), we make use of [Algorithm 1](#algorithm1 "In B.2.1 Corruption Experiments ‣ B.2 Attention Between Datapoints on Real Data ‣ Appendix B Additional Results ‣ Appendix") below.
When predicting for a datapoint k𝑘k, this algorithm completely destroys information from all other datapoints i≠k𝑖𝑘i\neq k in the batch b𝑏b by randomly permuting attribute values across all other datapoints.
Therefore, if NPT’s loss increases after corruption, it must meaningfully rely on attention between datapoints for prediction.

Input: list of masked minibatches ℬ=[𝑿(b)∈ℝK×d∣b∈1​…​B]ℬdelimited-[]superscript𝑿𝑏conditionalsuperscriptℝ𝐾𝑑𝑏1…𝐵\mathcal{B}=[\bm{X}^{(b)}\in\mathbb{R}^{K\times d}\mid b\in 1\dots B], unmasked label column 𝑿:,dsubscript𝑿

:𝑑\bm{X}\_{:,d}, trained model f:𝑿(b)→𝑿(b):𝑓→superscript𝑿𝑏superscript𝑿𝑏f:\bm{X}^{(b)}\rightarrow\bm{X}^{(b)}, batch size K𝐾K, loss function ℒℒ\mathcal{L}, number of attributes (including features and target) d𝑑d

Returns: test loss under data corruption ℒcorrsuperscriptℒcorr\mathcal{L}^{\text{corr}}

ℒcorr←0←superscriptℒcorr0\mathcal{L}^{\text{corr}}\leftarrow 0

for *𝐗(b)superscript𝐗𝑏\bm{X}^{(b)} in ℬℬ\mathcal{B}* do

for *k𝑘k in 1​…​K1…𝐾1\dots K* do

𝑿(b,k)←𝑿(b)←superscript𝑿𝑏𝑘superscript𝑿𝑏\bm{X}^{(b,k)}\leftarrow\bm{X}^{(b)}

// initialize batch to be corrupted

for *j𝑗j in 1​…​d1…𝑑1\dots d* do

𝑿i≠k,j(b,k)←permuteaxis=i​(𝑿i≠k,j(b,k))←subscriptsuperscript𝑿𝑏𝑘𝑖

𝑘𝑗subscriptpermuteaxis𝑖subscriptsuperscript𝑿𝑏𝑘𝑖

𝑘𝑗\bm{X}^{(b,k)}\_{i\neq k,j}\leftarrow\texttt{permute}\_{\text{axis}=i}(\bm{X}^{(b,k)}\_{i\neq k,j})

// permute each attr. column indep.

end for

ℒcorr+=ℒ(f(𝑿(b,k))k,d,𝑿k,d)\mathcal{L}^{\text{corr}}\mathrel{+}=\mathcal{L}(f(\bm{X}^{(b,k)})\_{k,d},\bm{X}\_{k,d})

// compute loss w/ unmasked label column

end for

end for

return ℒcorrsuperscriptℒcorr\mathcal{L}^{\text{corr}}

Algorithm 1 Data Corruption

Alternatively, we could also input datapoints *individually*, i.e., decrease the minibatch size to 1, to test if NPT depends on attention between datapoints.
Indeed, we find that performance also deteriorates in this scenario.
However, we believe that the Data Corruption experiment provides stronger evidence because it preserves batch statistics across attributes.
This makes sure that performance deterioration is not caused by spurious factors, such as a decreased batch size that was not encountered in training.
While NPT is generally compatible with varying batch sizes, we leave a thorough investigation of this for future work.

#### B.3 Real Data – *To Which* Other Points Does NPT Attend?

##### B.3.1 Attention Maps on Real Data

In [Fig. B.2](#A2.F2 "In B.3.1 Attention Maps on Real Data ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix"), we display ABD attention maps of NPT for the Protein regression dataset in addition to the one shown in [Fig. 4](#S4.F4 "In 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data ‣ 4 Experiments").
For visualization purposes, we sort the input datapoints with respect to their feature space distance to an arbitrary test datapoint.
This is to ensure that the global structure of the attention maps in [Fig. B.2](#A2.F2 "In B.3.1 Attention Maps on Real Data ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") has meaning.
Specifically, nearby entries in the attention maps belong to input datapoints that are close in input space.
With this transformation, the diagonal patterns appearing in [Fig. B.2](#A2.F2 "In B.3.1 Attention Maps on Real Data ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") clearly suggest that our model is attending more strongly between datapoints that are similar in input space.
Similar to the semi-synthetic experiments, some but not all attention heads display this pattern of interest.

![Refer to caption](/html/2106.02584/assets/x6.png)


Figure B.2: 
Visualizations of the Attention Between Datapoints (ABD) attention maps for real data – here, the Protein regression dataset – for all depths and a selection of heads.
Input to the model is sorted such that datapoints that are similar in input space have nearby indices.
The diagonal pattern (e.g., depth 2 and head 1) indicates that the model attends to similar inputs more strongly.
For illustration purposes, we here plot the log of the attention values.

##### B.3.2 Data Deletion Experiment

1
Input: Masked data 𝑿∈ℝn×d𝑿superscriptℝ𝑛𝑑\bm{X}\in\mathbb{R}^{n\times d}, active sample index i∗superscript𝑖i^{\*}.

2
y^←NPT​(𝑿)i∗,d←^𝑦NPTsubscript𝑿

superscript𝑖𝑑\hat{y}\leftarrow\text{NPT}(\bm{X})\_{i^{\*},d}

// original NPT prediction at active datapoint

3
Δmax←0.1←subscriptΔmax0.1\Delta\_{\text{max}}\leftarrow 0.1

// maximum allowed change in prediction

4
Δit←0.01←subscriptΔit0.01\Delta\_{\text{it}}\leftarrow 0.01

// initialize maximum change per deleted datapoint

5
Nmax-retry←50←subscript𝑁max-retry50N\_{\textup{max-retry}}\leftarrow 50

// maximum number of retries before increasing ΔitsubscriptΔit\Delta\_{\textup{it}}

6
ϵ←0.02←italic-ϵ0.02\epsilon\leftarrow 0.02

// fraction of points remaining at which we break

7
ℛ←{1,…,n}∖{i∗}←ℛ1…𝑛superscript𝑖\mathcal{R}\leftarrow\{1,\dots,n\}\setminus\{i^{\*}\}

// initialize remaining set

8
Nretry←0←subscript𝑁retry0N\_{\textup{retry}}\leftarrow 0

// initialize no. of retries

9
while *True* do

10
c=𝑐absentc=random\_choice(R)𝑅(R)

// random proposal for data deletion

11
y^proposal=NPT​(𝑿(ℛ∖{c})∪{i∗})i∗,dsubscript^𝑦proposalNPTsubscriptsubscript𝑿ℛ𝑐superscript𝑖

superscript𝑖𝑑\hat{y}\_{\text{proposal}}=\text{NPT}(\bm{X}\_{(\mathcal{R}\setminus\{c\})\cup\{i^{\*}\}})\_{i^{\*},d}

// predict without proposed datapoint

12
Δproposal=|y^proposal−y^|y^subscriptΔproposalsubscript^𝑦proposal^𝑦^𝑦\Delta\_{\textup{proposal}}=\frac{\lvert\hat{y}\_{\text{proposal}}-\hat{y}\rvert}{\hat{y}}

// change in pred. when deleting proposal

13
if *Δ*proposal*<Δ*it*subscriptΔ*proposal*subscriptΔ*it*\Delta\_{\textup{proposal}}<\Delta\_{\text{it}}* then

14
if *Δ*proposal*<Δ*max*subscriptΔ*proposal*subscriptΔ*max*\Delta\_{\textup{proposal}}<\Delta\_{\text{max}}* then

15
ℛ←ℛ∖{c}←ℛℛ𝑐\mathcal{R}\leftarrow\mathcal{R}\setminus\{c\}

// delete datapoint from input

16
Nretry←0←subscript𝑁retry0N\_{\textup{retry}}\leftarrow 0

17
else

18
break

// exceeded maximum change

19
else

20
Nretry←Nretry+1←subscript𝑁retrysubscript𝑁retry1N\_{\textup{retry}}\leftarrow N\_{\textup{retry}}+1

// candidate change was too large, try again

21
if *N*retry*≥N*max-retry*subscript𝑁*retry*subscript𝑁*max-retry*N\_{\textup{retry}}\geq N\_{\textup{max-retry}}* then

22
Δit←1.1⋅Δit←subscriptΔit⋅1.1subscriptΔit\Delta\_{\textup{it}}\leftarrow 1.1\cdot\Delta\_{\textup{it}}

// increase allowed change per iteration

23
Nretry←0←subscript𝑁retry0N\_{\textup{retry}}\leftarrow 0

24if *|ℛ|<ϵ⋅n\rvert\mathcal{R}\lvert<\epsilon\cdot n* then

25
break

// less than ϵ%percentitalic-ϵ\epsilon\% of original datapoints remaining

end while

26return ℛℛ\mathcal{R}

Algorithm 2 Data Deletion

![Refer to caption](/html/2106.02584/assets/x7.png)


Figure B.3: 
When predicting for any given datapoint, NPT prefers to keep similar datapoints around.
Displayed are average feature space differences and their standard errors between the active datapoint and the sets of kept, random, and deleted datapoints for a single batch.

We here give full details on the Data Deletion experiment presented in [Fig. 4](#S4.F4 "In 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data ‣ 4 Experiments").
To recap, we consider the prediction of NPT for a single test sample i∗superscript𝑖i^{\*}.
We then iteratively delete other datapoints from the input if they do not significantly change the prediction of NPT on i∗superscript𝑖i^{\*}.
[Algorithm 2](#algorithm2 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") describes this in detail.
We are then interested in differences between the deleted and the kept datapoints.
Specifically, we compare the average feature space distance in input space between the active datapoint i∗superscript𝑖i^{\*} and either the kept datapoints ℛℛ\mathcal{R} or deleted datapoints {1,…,n}∖({i∗}∪ℛ)1…𝑛superscript𝑖ℛ\{1,\dots,n\}\setminus(\{i^{\*}\}\cup\mathcal{R}), obtaining average distances Di∗,keptsubscript𝐷

superscript𝑖keptD\_{i^{\*},\textup{kept}}, Di∗,deletedsubscript𝐷

superscript𝑖deletedD\_{i^{\*},\textup{deleted}}.
We break out of the deletion algorithm if less than ϵ%percentitalic-ϵ\epsilon\% of the original points remain, to reduce variance in our estimates of the kept statistic.
We repeat [Algorithm 2](#algorithm2 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") for all 556755675567 test points i∗∈𝒟testsuperscript𝑖subscript𝒟testi^{\*}\in\mathcal{D}\_{\textup{test}} in the Protein regression dataset.

We perform a Wilcoxon signed-rank test on the pairs {Di∗,kept,Di∗,deleted}i∗∈𝒟testsubscriptsubscript𝐷

superscript𝑖keptsubscript𝐷

superscript𝑖deletedsuperscript𝑖subscript𝒟test\left\{D\_{i^{\*},\textup{kept}},D\_{i^{\*},\textup{deleted}}\right\}\_{i^{\*}\in\mathcal{D}\_{\textup{test}}} to determine if the median of the kept datapoints is less than the median of the deleted ones.
The test is highly significant at p≈0𝑝0p\approx 0, i.e., smaller than the floating point precision of SciPy Stats allows.
The raw Wilcoxon statistic is 3125889.53125889.53125889.5.

To make sure the difference is not an effect of sample size, we also construct a set of average differences to a set of randomly drawn datapoints.666There are many fewer kept than deleted datapoints.
Further, there are outliers in the dataset, and these affect the deleted datapoints more often than the kept datapoints.
We find that the average distance between a *random* subset and the *deleted* (not the kept!) datapoints also becomes statistically significantly smaller at large sample sizes.
Hence, we compare the *deleted* datapoints to a *random* subset to control for size effects.
That is, instead of using [Algorithm 2](#algorithm2 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix") for *targeted* deletion, we *randomly* construct ℛℛ\mathcal{R}, essentially only applying lines 101010 and 151515 of [Algorithm 2](#algorithm2 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix").
For each active test row i∗superscript𝑖i^{\*}, we randomly delete as many datapoints as were deleted in targeted fashion.
A Wilcoxon signed-rank test between the distances for the random and kept subset is likewise significant at p≈8.77⋅10−130𝑝⋅8.77superscript10130p\approx 8.77\cdot 10^{-130}.
This is the value we report in the main body.

We also run a computationally more demanding version of the algorithm with Δit←0.005←subscriptΔit0.005\Delta\_{\textup{it}}\leftarrow 0.005, ϵ←0.01←italic-ϵ0.01\epsilon\leftarrow 0.01 to see how many points we can successfully delete.
This version of the algorithm requires more computation which is why we limit execution to the test datapoints of a single batch.
The results are statistically significant at 5.26⋅10−49⋅5.26superscript10495.26\cdot 10^{-49} for kept << deleted and 8.38⋅10−39⋅8.38superscript10398.38\cdot 10^{-39} for kept << random for a Wilcoxon signed-rank test.
We illustrate the differences between the distances in [Fig. B.3](#A2.F3 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix").
We further note that using [Algorithm 2](#algorithm2 "In B.3.2 Data Deletion Experiment ‣ B.3 Real Data – To Which Other Points Does NPT Attend? ‣ Appendix B Additional Results ‣ Appendix"), we are able to reduce the set of datapoints present in the input to 111% of the original n𝑛n for 79.579.579.5% of active test datapoints and to 101010% in 99.599.599.5% of cases.
Percentages refer to n=2048𝑛2048n=2048 datapoints in total, of which 398398398 were test datapoints.

All in all, these experiments strongly suggest that NPT relies on interactions between similar datapoints for prediction.

#### B.4 Ablation Study 1: NPT Hyperparameters

We conduct an ablation study on the Protein and Boston Housing datasets (Table [4](#A2.T4 "Table 4 ‣ B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix")).
For Protein, the same 0.7/0.1/0.2 train/validation/test split is used for all model configurations.
Boston Housing uses a 0.7/0.2/0.1 train/validation/test split with 10-fold cross-validation.

Despite the significant difference in dataset sizes between Boston Housing (n=506)𝑛506(n=$506$) and Protein (n=45730)𝑛45730(n=45730), and the fact that Boston Housing includes both categorical and continuous variables, the base models used for each dataset are nearly identical.

On both datasets, we use an NPT model with 8 layers, 8 heads, per-attribute hidden dimension e=128𝑒128e=128, feature and target masking with p=0.15𝑝0.15p=$0.15$ for each, a cosine annealing schedule for the loss tradeoff λ𝜆\lambda, the LAMB [[98](#bib.bib98)] optimizer with Lookahead [[99](#bib.bib99)], a flat-then-anneal learning rate schedule with cosine decay and base learning rate 0.0010.0010.001, dropout with rate 0.10.10.1 on the attention weights and after linear layers, and gradient clipping at 111.
This configuration is essentially the same as the NPT-Base configuration described in [Section C.1](#A3.SS1 "C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix"), which we use with minimal per-dataset modifications for all other results in this work.

Different in our base models between the two datasets are the following settings.
The Boston Housing model takes as input the full dataset (i.e., batch size =507absent507=$507$) and Protein uses minibatching with batch size =2048absent2048=$2048$.
Boston Housing trains for 200002000020000 steps, and Protein for 400000400000400000.
The learning rate is constant for the first 70 %times70percentabsent70\text{\,}\frac{\mathrm{\char 37\relax}}{} of steps for Protein, but only for the first 50 %times50percentabsent50\text{\,}\frac{\mathrm{\char 37\relax}}{} of steps for Boston, starting the learning rate annealing earlier to defend against overfitting on the small dataset.These changes directly result from the different dataset sizes.

Table 4: NPT ablation study: test root mean-squared error (RMSE) on the Protein and Boston Housing regression datasets.

|  |  |  |
| --- | --- | --- |
| Test RMSE (±plus-or-minus\pm Std Err) ↓↓\downarrow | Protein | Boston |
| Base NPT | 3.413.413.41 | 3.00±0.23uncertain3.000.233.00\pm 0.23 |
| No Semi-Supervision | 3.383.383.38 | 3.38±0.46uncertain3.380.463.38\pm 0.46 |
| No Target Masking | 3.323.323.32 | 2.93±0.18uncertain2.930.182.93\pm 0.18 |
| No Feature Masking | 3.563.563.56 | 2.95±0.21uncertain2.950.212.95\pm 0.21 |
| No Feature Masking, No Target Masking | 3.583.583.58 | 3.20±0.26uncertain3.200.263.20\pm 0.26 |
| Feature Mask p=0.15→p=0.5𝑝0.15→𝑝0.5p=0.15\rightarrow p=0.5 | 3.873.873.87 | 3.39±0.23uncertain3.390.233.39\pm 0.23 |
| Target Mask p=0.15→p=0.5𝑝0.15→𝑝0.5p=0.15\rightarrow p=0.5 | 3.373.373.37 | 3.11±0.28uncertain3.110.283.11\pm 0.28 |
| 888 →→\rightarrow 444 Layers | 3.433.433.43 | 3.30±0.41uncertain3.300.413.30\pm 0.41 |
| 888 →→\rightarrow 161616 Layers | 3.363.363.36 | 3.05±0.24uncertain3.050.243.05\pm 0.24 |
| 888 →→\rightarrow 444 Heads | 3.423.423.42 | 3.25±0.30uncertain3.250.303.25\pm 0.30 |
| 888 →→\rightarrow 161616 Heads | 3.373.373.37 | 3.20±0.39uncertain3.200.393.20\pm 0.39 |
| Tradeoff λ=0.5𝜆0.5\lambda\ =0.5 | 3.503.503.50 | 2.96±0.25uncertain2.960.252.96\pm 0.25 |

As [Table 4](#A2.T4 "In B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix") shows, the performance of NPT is robust to a variety of significant hyperparameter choices.
This illustrates that practitioners will likely *not need to spend much time tuning hyperparameters* when applying NPT to novel datasets.
We now give results for the ablation study on the Protein and Boston datasets separately.

###### Protein Dataset.

See [Table 4](#A2.T4 "In B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix") for results and performed ablations.
It is computationally too expensive for us to perform full cross-validation over all ablations for the Protein regression dataset.
Instead, we report the results of a single 5-fold cross-validation for the Base NPT configuration on Protein (also varying the model random state).
This results in an RMSE of 3.40±0.05plus-or-minus3.400.053.40\pm 0.05 (σ𝜎\sigma).
The standard deviation of the 5-fold cross-validation allows us to roughly gauge which ablations have significant effect.
Given the results in [Table 4](#A2.T4 "In B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix"), we find that the majority of ablations do not lead to meaningful changes in performance.
Only the somewhat dramatic changes to the optimization of NPT result in its performance falling from the top rank on the Protein Dataset (second rank CatBoost has RMSE=3.51RMSE3.51\text{RMSE}=$3.51$):
removing stochastic feature masking (pfeature=0subscript𝑝feature0p\_{\text{feature}}=0), removing both stochastic feature masking (pfeature=0subscript𝑝feature0p\_{\text{feature}}=0) and stochastic target masking (ptarget=1subscript𝑝target1p\_{\text{target}}=1, training targets are always masked out at training time and NPT therefore cannot learn to attend to training targets at test time), or changing pfeaturesubscript𝑝featurep\_{\text{feature}} to 0.5 (meaning that 50% of all input features are masked out).
NPT appears to be particularly robust to changes in model complexity, e.g., depth and number of heads, although the results suggest that we could have further increased the size of Base NPT to achieve slightly higher performance.

###### Boston Dataset.

See [Table 4](#A2.T4 "In B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix") for results and performed ablations.
For the Boston dataset, we repeat ablations over all 10 CV splits.
Similarly, ablations on the Boston dataset are largely inconsequential;
none of them result in a statistically significant change in performance from the base model.
The second rank performer on Boston is MLP, at RMSE=3.32RMSE3.32\text{RMSE}=3.32.
Only ablation of semi-supervision or changing pfeaturesubscript𝑝featurep\_{\text{feature}} to 0.5 result in a change in the top ranking of NPT among the baselines.

Altogether, the ablation study supports the claim that NPT can be applied successfully with very little tuning to datasets of vastly different sizes and feature types.
Changes in model depth and number of heads do not appear significant, but using a reasonably low feature masking probability (e.g., 15%, as has been commonly used in the literature [[24](#bib.bib24)]) may be important to stable training.

Supported by these ablations, we sweep over only a small selection of configurations for our main benchmark comparison in [Section 4.1](#S4.SS1 "4.1 NPTs Perform Competitively on Established Benchmarks ‣ 4 Experiments").
And indeed, it seems that NPT is robust to hyperparameter changes, given that these configurations perform well across vastly different settings (binary and multi-class classification, datasets with millions of datapoints, etc.) than those explored in the ablations.
See [Appendix E](#A5 "Appendix E Classification and Regression Benchmark Details ‣ Appendix") for details.

We speculate that NPT’s robustness stems from (a) being a relatively overparametrized architecture that is powerful enough to model a wide variety of datasets and (b) from the effective regularization introduced by the feature masking mechanism.
Finally, we emphasize that the aim of this work is to introduce the NPT architecture and examine its properties, not to spend significant effort and compute resources on achieving top performance across all benchmarks.

#### B.5 Ablation Study 2: NPT without ABA and NPT without Feature Masking

We next present an additional ablation study targeting two core components of NPTs across all datasets: the Attention Between Attributes (ABA) layer and the stochastic feature masking.

ABA Layer. First, we perform an ablation to test if ABA layers are beneficial in practice.
For this, we simply leave out the ABA layers, such that the MLP at the end of the ABD layers (see “rFF” in [Eq. 5](#S2.E5 "In 2.3 Multi-Head Self-Attention ‣ 2 Non-Parametric Transformers")) is now the only way for the model to independently transform the features of input datapoints.

Our results, given in [Table 5](#A2.T5 "In B.5 Ablation Study 2: NPT without ABA and NPT without Feature Masking ‣ Appendix B Additional Results ‣ Appendix"), show that, generally, ABA is a useful component of the NPT architecture.
Leaving out ABA increases performance only for 3/10 datasets.
Interestingly, all three of these datasets are regression tasks, which may warrant further investigation.
We observe the largest difference for the Poker Hands dataset, which requires complex reasoning between input features: in the same number of training steps, the ablation only achieves 57.4% accuracy compared to 99.3% for full NPT.
These results support our hypothesis that ABA is useful when the dataset requires complex transformations of the features.
Our most general recommendation would be to default to using NPTs with ABA layers, as they boost performance on the majority of datasets we examine.
However, if practitioners can spend the extra compute, exploring NPTs without ABA can be worthwhile.

Stochastic Feature Masking.
We perform an ablation to test if the stochastic feature masking objective (cf. §[2.6](#S2.SS6 "2.6 Masking and Optimization ‣ 2 Non-Parametric Transformers")) is beneficial in practice.
For this, we simply disable all stochastic masking of input features by setting pfeatures=0subscript𝑝features0p\_{\textup{features}}=0.

Our results, also in [Table 5](#A2.T5 "In B.5 Ablation Study 2: NPT without ABA and NPT without Feature Masking ‣ Appendix B Additional Results ‣ Appendix"), show that for 9/10 datasets, enabling feature masking yields at least a small improvement in performance.
Disabling feature masking is detrimental to the performance on the Poker Hands dataset, leading to a 30% drop in accuracy.
Again, our general recommendation would be to use NPTs with feature masking by default, as it rarely seems to decrease performance and sometimes helps significantly, but to explore NPTs without feature masking if feasible.

Table 5: Additional ablation studies.
We study ablations of NPT (a) without ABA layers and (b) without stochastic feature masking.
In both cases, performance tends to decrease.
These results suggest that both ABA layers and stochastic feature masking contribute positively to the performance of NPTs.
For the small datasets, we report mean values and standard errors over 101010 CV splits.

|  |  |  |  |
| --- | --- | --- | --- |
|  | NPT without ABA | NPT without Feature Masking | Default NPT |
| Classification |  |  |  |
| Poker Hand (Acc. ↑↑\uparrow) | 57.457.457.4 | 69.769.769.7 | 99.399.399.3 |
| Forest Cover (Acc. ↑↑\uparrow) | 95.595.595.5 | 96.096.096.0 | 96.796.796.7 |
| Higgs Boson (AUC ↑↑\uparrow) | 0.8590.8590.859 | 0.8710.8710.871 | 0.8920.8920.892 |
| Income (AUC ↑↑\uparrow) | 0.9520.9520.952 | 0.9520.9520.952 | 0.9520.9520.952 |
| Kick (AUC ↑↑\uparrow) | 0.7670.7670.767 | 0.7660.7660.766 | 0.7700.7700.770 |
| Breast Cancer (AUC ↑↑\uparrow) | 0.992±0.008uncertain0.9920.0080.992\pm 0.008 | 0.996±0.006uncertain0.9960.0060.996\pm 0.006 | 0.997±0.001uncertain0.9970.0010.997\pm 0.001 |
| Regression |  |  |  |
| Boston Housing (RMSE ↓↓\downarrow) | 3.22±0.25uncertain3.220.253.22\pm 0.25 | 3.18±0.35uncertain3.180.353.18\pm 0.35 | 2.92±0.15uncertain2.920.152.92\pm 0.15 |
| Yacht (RMSE ↓↓\downarrow) | 1.15±0.11uncertain1.150.111.15\pm 0.11 | 0.50±0.06uncertain0.500.060.50\pm 0.06 | 1.27±0.15uncertain1.270.151.27\pm 0.15 |
| Concrete (RMSE ↓↓\downarrow) | 4.79±0.12uncertain4.790.124.79\pm 0.12 | 5.37±0.20uncertain5.370.205.37\pm 0.20 | 5.21±0.20uncertain5.210.205.21\pm 0.20 |
| Protein (RMSE ↓↓\downarrow) | 3.293.293.29 | 3.593.593.59 | 3.413.413.41 |

#### B.6 Computational Cost of Non-Parametric Transformers

We next compare the computational requirements of NPT against the various baselines.
More specifically, we compare experiment runtimes and maximum memory usage on the Protein and Higgs datasets.
We choose these datasets because they are representative of medium and large datasets in terms of computational requirements, with 457304573045730 and 110000001100000011000000 datapoints respectively.
Note that, while we re-use hyperparameter configurations across datasets for NPTs, the baselines require a novel hyperparameter search to be performed for each dataset (cf. Appendices [C](#A3 "Appendix C Additional Details on the NPT Architecture ‣ Appendix") and [E](#A5 "Appendix E Classification and Regression Benchmark Details ‣ Appendix")).
Below, we include the cost of hyperparameter optimization for the baselines.

Note that these numbers only provide a rough ordering of the compute and memory costs of the various methods.
We did *not* optimize the baselines or NPT for memory usage, training time, or prediction speed.
Additionally, while NPTs rely on GPU-accelerated PyTorch code, many of the baselines are CPU-only: therefore, the results depend on our particular CPU and GPU choices.

We also give the number of CPUs used in each experiment for each baseline.
Here, we maximize the number of CPUs used in parallel execution in order to speed up training.
This is mainly limited by the memory used per process: e.g., if we list # CPUs as 1, this does not mean that we used a machine with only 1 CPU, but rather that each process used a significant amount of the total available memory and hence we could not increase the number of CPUs used in parallel.
Note that, additionally, for the CPU baselines, we made use of high-memory instances when this was necessary to avoid out-of-memory issues.

In summary, the numbers we give are a rough indication of the computational cost that a practitioner should expect to require in order to reproduce our results.
It is likely that by tuning aspects of our setup, both for NPTs and the baselines, memory usage and/or runtimes could be improved.

Table 6: Protein dataset (45,730 datapoints): compute and memory requirements of hyperparameter tuning for baselines and training time of the selected hyperparameter configuration for NPTs. We report the number of CPUs used in execution, execution time, and peak memory usage, where the relevant bottleneck is main memory usage for CPU-based methods and GPU memory usage for GPU-based methods (i.e., TabNet and NPT).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Metric | # CPUs | Execution Time | Peak Main Memory (GB) | Peak GPU Memory (GB) |
| Random Forest | 888 | 13h 33m 58s | 7.82 | – |
| Gradient Boosting | 111 | 47m 51s | 11.17 | – |
| XGBoost | 888 | 10m 31s | 2.94 | – |
| CatBoost | 111 | 8m 33s | 11.27 | – |
| LightGBM | 888 | 21s | 1.65 | – |
| MLP | 646464 | 42m 14s | 8.96 | – |
| k-NN | 888 | 1m 8s | 40.47 | – |
| TabNet | 111 | 1h 33m 35s | 16.00 | 3.72 |
| NPT | 444 | 11h 51m 25s | 4.42 | 6.17 |




Table 7: Higgs dataset (11,000,000 datapoints): compute and memory requirements of hyperparameter tuning for baselines and training time of the selected hyperparameter configuration for NPTs. We report the number of CPUs used in execution, execution time, and peak memory usage, where the relevant bottleneck is main memory usage for CPU-based methods and GPU memory usage for GPU-based methods (i.e., TabNet and NPT).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Metric | # CPUs | Execution Time | Peak Main Memory (GB) | Peak GPU Memory (GB) |
| Random Forest | 111 | 13d 13h 5m 6s | 189.18 | – |
| Gradient Boosting | 111 | 3d 19h 45m 56s | 26.65 | – |
| XGBoost | 888 | 23h 26m 17s | 108.54 | – |
| CatBoost | 888 | 2h 6m 35s | 78.34 | – |
| LightGBM | 888 | 55m 57s | 35.13 | – |
| MLP | 666 | 12h 54m 7s | 34.41 | – |
| k-NN | 111 | 4d 22h 12m 20s | 16.26 | – |
| TabNet | 111 | 2d 5h 2m 43s | 16.00 | 1.18 |
| NPT | 444 | 5d 22h 12m 7s | 37.79 | 19.18 |

We display the observed computational costs in [Tables 6](#A2.T6 "In B.6 Computational Cost of Non-Parametric Transformers ‣ Appendix B Additional Results ‣ Appendix") and [7](#A2.T7 "Table 7 ‣ B.6 Computational Cost of Non-Parametric Transformers ‣ Appendix B Additional Results ‣ Appendix") for the Protein and Higgs datasets.
As of now, NPTs do generally require longer training times than the non-neural baselines.
For example, for the Protein dataset, the selected hyperparameter configuration of NPT trains in 11 hours, while all boosting methods finish their runs in less than 1 hour, including the hyperparameter tuning.
The exception to this rule is given by some of the baselines, e.g., Random Forests, which do not scale well to large datasets such as Higgs.
On Higgs, the NPT run takes 5d 22h compared to 13d 13h for Random Forests.

With NPTs, we want to store as much data as possible in addition to the network weights; recall that this is done to improve the quality of the minibatch approximation of the full dataset.
Therefore, as expected, NPT is much more GPU-memory intensive during training than TabNet, the only other baseline with a GPU-based implementation, for which maximizing minibatch size is not desirable.
In particular, the peak GPU memory usage on Higgs for NPTs is 19.1819.1819.18 GB and
1.181.181.18 GB for TabNet.
However, we note that other methods are often also memory-intensive on larger datasets.
For example, Random Forest with 1 process uses 189.18189.18189.18 GB peak CPU memory.

We next give a rough indication of prediction time behavior of NPT and the baselines.
For the same reason as above, NPT is expected to have high memory usage at prediction time.
In terms of prediction speed, we suspect that our ability to scale NPT to large batch sizes, e.g., 409640964096 on the Higgs dataset, might give us an advantage in comparison to those baselines that cannot be parallelized well and/or lack GPU support.
We leave a detailed investigation of prediction time behavior to future work.

Finally, as discussed in §[5](#S5 "5 Limitations, Future Work, and Conclusions"), we note that by incorporating recent tools for sparse and efficient attention [[47](#bib.bib47), [19](#bib.bib19), [84](#bib.bib84), [18](#bib.bib18), [5](#bib.bib5)], future research could significantly improve the scalability of NPTs.

#### B.7 Extended Results for Tabular Data Benchmarks

See [Table 8](#A2.T8 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix") ([Table 9](#A2.T9 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix")) for test accuracies (negative log-likelihood scores) on the UCI classification datasets and additionally [Table 10](#A2.T10 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix") for AUROC results on the binary classification datasets.
For the regression datasets, see [Table 11](#A2.T11 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix") for RMSE scores and [Table 12](#A2.T12 "In B.7 Extended Results for Tabular Data Benchmarks ‣ Appendix B Additional Results ‣ Appendix") for MSE scores.

Table 8: UCI classification datasets: test accuracy. Standard error reported for datasets with multiple cross-validation splits.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Test Accuracy ↑↑\uparrow | Higgs Boson | Poker Hand | Forest Cover | Income | Kick | Breast Cancer |
| Random Forest | 76.276.276.2 | 71.571.571.5 | 94.894.894.8 | 95.495.495.4 | 90.190.190.1 | 94.20±0.70uncertain94.200.7094.20\pm 0.70 |
| Gradient Boosting | 76.576.576.5 | 94.194.194.1 | 96.796.796.7 | 95.895.895.8 | 90.290.290.2 | 94.03±0.90uncertain94.030.9094.03\pm 0.90 |
| XGBoost | 77.077.077.0 | 95.995.995.9 | 97.197.197.1 | 95.695.695.6 | 90.390.390.3 | 94.91±0.68uncertain94.910.6894.91\pm 0.68 |
| CatBoost | 76.676.676.6 | 99.299.299.2 | 95.795.795.7 | 95.895.895.8 | 90.190.190.1 | 95.61±0.75uncertain95.610.7595.61\pm 0.75 |
| LightGBM | 75.975.975.9 | 92.892.892.8 | 85.085.085.0 | 95.895.895.8 | 90.390.390.3 | 95.26±0.82uncertain95.260.8295.26\pm 0.82 |
| MLP | 78.378.378.3 | 99.599.599.5 | 95.295.295.2 | 95.495.495.4 | 90.090.090.0 | 94.73±0.89uncertain94.730.8994.73\pm 0.89 |
| k-NN777Out-of-memory on the Higgs Boson dataset when attempting approximate 3-NN on an Azure D64 v3 instance with 256 GB RAM. | — | 50.450.450.4 | 90.790.790.7 | 94.894.894.8 | 87.787.787.7 | 95.26±0.79uncertain95.260.7995.26\pm 0.79 |
| TabNet888TabNet had notably lower accuracy in our setup on the Poker Hand dataset (which has a fixed test set) than that the 99.2% reported in the original work [[2](#bib.bib2)]. We are in communication with the authors, attempting to improve these results. However, our results on Higgs Boson match the reported performance more closely (78.44%percent78.4478.44\% (theirs) vs 77.1%percent77.177.1\% (ours)). Further, we note that our other baselines achieve significantly better performance on the same datasets than those reported in [[2](#bib.bib2)]; e.g., our MLP achieves 99.5%percent99.599.5\% accuracy on Poker Hand dataset while they report 50.0%percent50.050.0\%; our XGBoost achieves 97.1%percent97.197.1\% on Forest Cover while they report 89.34%percent89.3489.34\%. However, we note that some of the datasets – such as Forest Cover – do not have fixed test sets. Therefore, we cannot exclude the possibility that the performance differences are due to differently chosen train-test splits. | 77.177.177.1 | 53.353.353.3 | 94.294.294.2 | 95.595.595.5 | 89.589.589.5 | 94.91±0.76uncertain94.910.7694.91\pm 0.76 |
| NPT | 80.780.780.7 | 99.399.399.3 | 96.796.796.7 | 95.695.695.6 | 90.090.090.0 | 94.73±0.69uncertain94.730.6994.73\pm 0.69 |




Table 9: UCI classification datasets: negative log-likelihood (NLL). Standard error reported for datasets with multiple cross-validation splits.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Test NLL ↓↓\downarrow | Higgs Boson | Poker Hand | Forest Cover | Income | Kick | Breast Cancer |
| Random Forest | 0.4890.4890.489 | 0.8430.8430.843 | 0.1910.1910.191 | 0.1260.1260.126 | 0.3050.3050.305 | 0.142±0.012uncertain0.1420.0120.142\pm 0.012 |
| Gradient Boosting | 0.4770.4770.477 | 0.3790.3790.379 | 0.1090.1090.109 | 0.1110.1110.111 | 0.2960.2960.296 | 0.185±0.024uncertain0.1850.0240.185\pm 0.024 |
| XGBoost | 0.4710.4710.471 | 0.1780.1780.178 | 0.0800.0800.080 | 0.1470.1470.147 | 0.2930.2930.293 | 0.143±0.025uncertain0.1430.0250.143\pm 0.025 |
| CatBoost | 0.4760.4760.476 | 0.0650.0650.065 | 0.1200.1200.120 | 0.1090.1090.109 | 0.2960.2960.296 | 0.124±0.024uncertain0.1240.0240.124\pm 0.024 |
| LightGBM | 0.4860.4860.486 | 0.4200.4200.420 | 0.3610.3610.361 | 0.1090.1090.109 | 0.2940.2940.294 | 0.163±0.034uncertain0.1630.0340.163\pm 0.034 |
| MLP | 0.4520.4520.452 | 0.0280.0280.028 | 0.1310.1310.131 | 0.1180.1180.118 | 0.3330.3330.333 | 0.545±0.254uncertain0.5450.2540.545\pm 0.254 |
| k-NN999See above note on out-of-memory. | — | 0.9750.9750.975 | 0.2740.2740.274 | 0.1390.1390.139 | 0.3330.3330.333 | 0.466±0.167uncertain0.4660.1670.466\pm 0.167 |
| TabNet | 0.4690.4690.469 | 0.9730.9730.973 | 0.1510.1510.151 | 0.1190.1190.119 | 0.3140.3140.314 | 0.233±0.036uncertain0.2330.0360.233\pm 0.036 |
| NPT | 0.4120.4120.412 | 0.1190.1190.119 | 0.0870.0870.087 | 0.1150.1150.115 | 0.2990.2990.299 | 0.137±0.026uncertain0.1370.0260.137\pm 0.026 |
|  |  |  |  |  |  |  |




Table 10: UCI classification datasets: test area under the receiver operating characteristic curve (AUROC) on binary classification tasks. Standard error reported for datasets with multiple cross-validation splits.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Test AUROC ↑↑\uparrow | Higgs Boson | Income | Kick | Breast Cancer |
| Random Forest | 0.8470.8470.847 | 0.9470.9470.947 | 0.7590.7590.759 | 0.989±0.003uncertain0.9890.0030.989\pm 0.003 |
| Gradient Boosting | 0.8500.8500.850 | 0.9550.9550.955 | 0.7690.7690.769 | 0.987±0.004uncertain0.9870.0040.987\pm 0.004 |
| XGBoost | 0.8540.8540.854 | 0.9460.9460.946 | 0.7750.7750.775 | 0.989±0.003uncertain0.9890.0030.989\pm 0.003 |
| CatBoost | 0.8510.8510.851 | 0.9560.9560.956 | 0.7730.7730.773 | 0.992±0.003uncertain0.9920.0030.992\pm 0.003 |
| LightGBM | 0.8430.8430.843 | 0.9560.9560.956 | 0.7760.7760.776 | 0.992±0.003uncertain0.9920.0030.992\pm 0.003 |
| MLP | 0.8670.8670.867 | 0.9490.9490.949 | 0.7390.7390.739 | 0.982±0.007uncertain0.9820.0070.982\pm 0.007 |
| k-NN101010See above note on out-of-memory. | — | 0.9320.9320.932 | 0.7470.7470.747 | 0.980±0.005uncertain0.9800.0050.980\pm 0.005 |
| TabNet | 0.8570.8570.857 | 0.9480.9480.948 | 0.7450.7450.745 | 0.978±0.005uncertain0.9780.0050.978\pm 0.005 |
| NPT | 0.8920.8920.892 | 0.9520.9520.952 | 0.7700.7700.770 | 0.997±0.001uncertain0.9970.0010.997\pm 0.001 |




Table 11: UCI regression datasets: test root mean-squared error (RMSE). Standard error reported for datasets with multiple cross-validation splits.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Test RMSE ↓↓\downarrow | Protein | Concrete | Boston Housing | Yacht |
| Random Forest | 3.573.573.57 | 5.48±0.18uncertain5.480.185.48\pm 0.18 | 3.78±0.33uncertain3.780.333.78\pm 0.33 | 0.91±0.13uncertain0.910.130.91\pm 0.13 |
| Gradient Boosting | 3.613.613.61 | 4.7±0.18uncertain4.70.184.7\pm 0.18 | 3.44±0.22uncertain3.440.223.44\pm 0.22 | 0.85±0.12uncertain0.850.120.85\pm 0.12 |
| XGBoost | 3.603.603.60 | 4.68±0.15uncertain4.680.154.68\pm 0.15 | 3.39±0.29uncertain3.390.293.39\pm 0.29 | 0.88±0.13uncertain0.880.130.88\pm 0.13 |
| CatBoost | 3.513.513.51 | 4.28±0.16uncertain4.280.164.28\pm 0.16 | 3.44±0.34uncertain3.440.343.44\pm 0.34 | 1.05±0.16uncertain1.050.161.05\pm 0.16 |
| LightGBM | 3.653.653.65 | 4.64±0.18uncertain4.640.184.64\pm 0.18 | 3.86±0.27uncertain3.860.273.86\pm 0.27 | 13.6±0.73uncertain13.60.7313.6\pm 0.73 |
| MLP | 3.623.623.62 | 5.53±0.2uncertain5.530.25.53\pm 0.2 | 3.32±0.39uncertain3.320.393.32\pm 0.39 | 0.91±0.13uncertain0.910.130.91\pm 0.13 |
| k-NN | 3.773.773.77 | 8.51±0.3uncertain8.510.38.51\pm 0.3 | 4.27±0.37uncertain4.270.374.27\pm 0.37 | 12.02±0.65uncertain12.020.6512.02\pm 0.65 |
| TabNet | 3.593.593.59 | 5.85±0.15uncertain5.850.155.85\pm 0.15 | 3.88±0.34uncertain3.880.343.88\pm 0.34 | 3.41±1.12uncertain3.411.123.41\pm 1.12 |
| NPT | 3.413.413.41 | 5.21±0.20uncertain5.210.205.21\pm 0.20 | 2.92±0.15uncertain2.920.152.92\pm 0.15 | 1.27±0.15uncertain1.270.151.27\pm 0.15 |




Table 12: UCI regression datasets: test mean-squared error (MSE). Standard deviation reported for datasets with multiple cross-validation splits.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Test MSE (±(\pm Std Dev) ↓↓\downarrow | Protein | Concrete | Boston | Yacht |
| Random Forest | 12.812.812.8 | 30.4±6.4uncertain30.46.430.4\pm 6.4 | 15.4±9.5uncertain15.49.515.4\pm 9.5 | 0.986±0.818uncertain0.9860.8180.986\pm 0.818 |
| Gradient Boosting | 13.013.013.0 | 22.4±5.2uncertain22.45.222.4\pm 5.2 | 12.3±4.9uncertain12.34.912.3\pm 4.9 | 0.867±0.779uncertain0.8670.7790.867\pm 0.779 |
| XGBoost | 13.013.013.0 | 22.1±4.2uncertain22.14.222.1\pm 4.2 | 12.3±7.6uncertain12.37.612.3\pm 7.6 | 0.939±0.881uncertain0.9390.8810.939\pm 0.881 |
| CatBoost | 12.312.312.3 | 18.6±4.3uncertain18.64.318.6\pm 4.3 | 13±9.8uncertain139.813\pm 9.8 | 1.36±1.12uncertain1.361.121.36\pm 1.12 |
| LightGBM | 13.313.313.3 | 21.9±5.3uncertain21.95.321.9\pm 5.3 | 15.6±7.6uncertain15.67.615.6\pm 7.6 | 190±65.1uncertain19065.1190\pm 65.1 |
| MLP | 13.113.113.1 | 31±6.9uncertain316.931\pm 6.9 | 12.6±11uncertain12.61112.6\pm 11 | 0.994±0.937uncertain0.9940.9370.994\pm 0.937 |
| k-NN | 14.214.214.2 | 73.3±16uncertain73.31673.3\pm 16 | 19.6±11uncertain19.61119.6\pm 11 | 149±52.6uncertain14952.6149\pm 52.6 |
| TabNet | 12.912.912.9 | 34.4±5.8uncertain34.45.834.4\pm 5.8 | 16.2±11uncertain16.21116.2\pm 11 | 24.1±54.3uncertain24.154.324.1\pm 54.3 |
| NPT | 11.611.611.6 | 27.6±7.6uncertain27.67.627.6\pm 7.6 | 8.77±2.6uncertain8.772.68.77\pm 2.6 | 1.8±1.49uncertain1.81.491.8\pm 1.49 |

#### B.8 Image Classification Results

We explore two different setups for applying NPTs to high-dimensional image data: (1) using a CNN encoder based on the ResNet-18 architecture, followed by ABD layers, and
(2) using a linear patching encoder that is then followed by ABD and ABA layers.
We present results using (1) for CIFAR-10 and (2) for MNIST in the main paper, and additionally provide results using (2) on CIFAR-10 below.

Note that the aim of our image classification experiments is not to match the performance of a pretrained Transformer image classifier.
Rather, we hope to demonstrate that NPTs can readily learn interactions between datapoints on a wide variety of data modalities and tasks, including image classification, while achieving reasonable performance.

###### (1) CNN Encoder.

In this setup, we replace our linear encoder with a CNN, which is then folowed by several rounds of Attention Between Datapoints (ABD) on the CNN encodings.
We apply this setup to CIFAR-10.

In detail, we use a ResNet-18 encoder followed by 4 blocks of ABD (as we have in a default 8 layer NPT, cf. [Section C.1.1](#A3.SS1.SSS1 "C.1.1 NPT-Base Architecture ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix")) with 8 heads.
Because we do not use Attention Between Attributes (ABA) the output of the encoder corresponds to the dimensions h=d⋅e=128ℎ⋅𝑑𝑒128h=d\cdot e=128.
We train in a supervised manner (without test inputs available at training time) with a training batch size of 128 and evaluation batch size of 480, for a fixed 100 epochs.

As reported in the main text, we achieve a test accuracy of 93.7% on CIFAR-10 with this architecture.
We find that the data corruption test (cf. [Section 4.3](#S4.SS3 "4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments")) decreases accuracy by 1.2%, which suggests that NPT meaningfully relies on other datapoints for prediction on CIFAR-10.
The ResNet-18 alone achieves a test accuracy of 93.9%.

We further note that with a ResNet-18 encoder pretrained on ImageNet, our ResNet + NPT architecture achieves a test accuracy of 94.7% on CIFAR-10, and loses 0.7% in the data corruption experiment, whereas the pretrained ResNet-18 alone achieves a lower 94.2% accuracy.
We believe that an exploration of how pretraining might affect the performance of NPT and the extent to which predictions rely on other datapoints is interesting future work.

###### (2) Linear Patching Encoder.

We additionally consider an image classification setup using a linear patching encoder, which we apply to both MNIST and CIFAR-10.

In detail, we append the mask dimension as an extra channel and apply image patching with linear embeddings as in [[25](#bib.bib25)].
Further following [[25](#bib.bib25)], we use a learned position embedding for each patch and the class token.
We use 7×7=4977497\times 7=49 patches on MNIST and 8×8=6488648\times 8=64 patches on CIFAR-10.
On both datasets, for this linear patching setup, we begin with the NPT-Base architecture described in [C.1.1](#A3.SS1.SSS1 "C.1.1 NPT-Base Architecture ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").
On MNIST, we use batch size 512, train for 500,000 steps, use hidden dimensions e=16𝑒16e=16, ptarget=0.15subscript𝑝target0.15p\_{\text{target}}=0.15, and use 7×7=4977497\times 7=49 patches.
On CIFAR-10, we use batch size 512, train for 1,000,000 steps, use random crops and horizontal flips for data augmentation, use 8×8=6488648\times 8=64 patches of each image, and do not use target masking due to constraints on compute time.

With this setup, NPT achieves 98.3%percent98.3$98.3$\% accuracy on MNIST and 68.2%percent68.2$68.2$\% accuracy on CIFAR-10.
We additionally find in the data corruption experiment (detailed in [Section 4.3](#S4.SS3 "4.3 NPTs Learn to Use Attention Between Datapoints on Real Data ‣ 4 Experiments")) that after destroying information from other datapoints, the change in accuracy is -0.4% on MNIST and -5.1% on CIFAR-10, demonstrating that NPTs learn to make use of interactions between images.

However, we did not find that this sufficiently demonstrated that NPTs make use of datapoint interactions in achieving reasonable performance on CIFAR-10, and hence conducted the experiment on CIFAR-10 using the CNN encoder setup above.

We expect that the relatively low performance in the linear patching setup on CIFAR-10 was due to a number of differences between our setup and other works, which report state-of-the-art results on image classification using Transformers and linear patching.
Most importantly, previous works [[25](#bib.bib25), [46](#bib.bib46)] either consider only, or pretrain on, large or huge datasets; for example, ImageNet [[23](#bib.bib23), [46](#bib.bib46), [86](#bib.bib86)], ImageNet-21k [[75](#bib.bib75)], or JFT-300M, with over 375 million labeled datapoints [[83](#bib.bib83), [25](#bib.bib25)].
We perform no pretraining, and therefore a direct comparison of these results to this line of work is inappropriate.
Additionally, previous works use significantly more patches (e.g., 256256256 in [[25](#bib.bib25)]) and use higher resolutions, including during fine-tuning by upscaling from 32×32323232\times 32 to 224×224224224224\times 224 resolution [[85](#bib.bib85), [54](#bib.bib54), [25](#bib.bib25), [46](#bib.bib46)].

### Appendix C Additional Details on the NPT Architecture

#### C.1 NPT Training and Hyperparameters

##### C.1.1 NPT-Base Architecture

Below, we outline the NPT-Base model configuration.
The final configurations used for each dataset are essentially the same as NPT-Base, with minor alterations in parameters such as hidden dimension size, learning rate warmup, batch size, and number of training steps.
Given our limited memory and compute time budget, these changes directly result from differences in number of datapoints/attributes between the datasets.
We divide the NPT-Base configuration into architectural details and optimization details.

###### NPT-Base Architecture

* •

  8 layers, alternating Attention Between Datapoints and Attention Between Attributes.
* •

  8 heads.
* •

  Row-wise feed-forward (rFF) networks with one hidden layer, 4x expansion factor, and GeLU activation (standard in Transformer literature [[90](#bib.bib90), [66](#bib.bib66)]).
* •

  Attention weight and hidden layer dropout with p=0.1𝑝0.1p=0.1 (cf. [Section C.2.1](#A3.SS2.SSS1 "C.2.1 Dropout ‣ C.2 Further Details on ABD and ABA Layers ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix")).
* •

  Per-attribute hidden dimension e=64𝑒64e=64.

###### NPT-Base Optimization

* •

  LAMB [[98](#bib.bib98)] optimizer with β=(0.9,0.999)𝛽0.90.999\beta=(0.9,0.999) and ϵ=1​e−6italic-ϵ1𝑒6\epsilon=1e-6, and a Lookahead [[99](#bib.bib99)] wrapper with slow update rate α=0.5𝛼0.5\alpha=0.5 and k=6𝑘6k=6 steps between updates.
* •

  Stochastic feature masking probability pfeature=0.15subscript𝑝feature0.15p\_{\text{feature}}=0.15.
* •

  Anneal the tradeoff λ𝜆\lambda between feature and target loss with a cosine schedule, starting at 1 (all feature loss) to 0 (all target loss) over the course of training.
* •

  Flat-then-anneal learning rate schedule: flat at the base learning rate for 70% of steps, and then anneals following a cosine schedule to 0 by the end of training.
* •

  Base learning rate 1e-3.
* •

  Gradient clipping at 1.

On all datasets with minibatching, we approximately maintain relative train, validation, and test datapoint proportions in each batch.
We train NPT in semi-supervised mode (cf. [Section C.4.2](#A3.SS4.SSS2 "C.4.2 Masking Encompasses Many Common Machine Learning Settings ‣ C.4 NPT Masking ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix")) but have found that this does not consistently improve performance compared to conventional training because the amount of unlabeled test data is usually comparatively small.

##### C.1.2 NPT Training on Small Data

Here we describe the hyperparameter sweep details for small datasets – Breast Cancer, Boston, Concrete, and Yacht.

###### Base Hyperparameter Configurations.

Across these small datasets, we make a few minor adjustments to the NPT-Base architecture and optimization to obtain the NPT-Small configuration: we increase the default number of hidden dimensions to e=128𝑒128e=128, fix the flat-then-anneal schedule to be flat for 50% instead of 70% of steps, and train with the entire dataset as input, i.e., no minibatching.
We set stochastic target masking probability to ptarget=1subscript𝑝target1p\_{\text{target}}=1 by default, i.e., deterministically mask out train labels as would be done in a normal supervised setting, and then introduce modifications in our sweep.

Note that the vast majority of hyperparameters such as the number of layers and heads, optimizer, pfeaturesubscript𝑝featurep\_{\text{feature}}, tradeoff annealing schedule, learning rate schedule, and gradient clipping are exactly the same between NPT-Base and NPT-Small.

We would like to keep the base configuration for each of the small datasets exactly the same.
However, we need to slightly vary the learning rate and number of epochs per dataset to optimize loss convergence across datasets.
We use a base learning rate 5e-4 on Breast Cancer and 1e-3 on the other small datasets.
We train for 200020002000 epochs on Breast Cancer and Boston, and 100001000010000 epochs on Yacht and Concrete.
On Breast Cancer, we additionally drop e=32𝑒32e=32 due to memory constraints (it has more attributes than other small datasets).

###### Small Data Sweep.

Based on these configurations, we sweep over the following 8 configurations of the model on each dataset.111111
Note that we do not search a 28superscript282^{8} grid over these modifications.
We only try out these 888 distinct models.

* •

  Vanilla NPT-Small model for given dataset.
* •

  Increase number of layers 8→16→8168\rightarrow 16.
* •

  Increase number of heads 8→16→8168\rightarrow 16.
* •

  Increase number of layers 8→16→8168\rightarrow 16, and number of heads 8→16→8168\rightarrow 16.
* •

  Stochastic target masking with probability ptarget=0.1subscript𝑝target0.1p\_{\text{target}}=0.1.
* •

  Stochastic target masking with probability ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5.
* •

  Increase stochastic feature masking probability from 0.15 to pfeature=0.2subscript𝑝feature0.2p\_{\text{feature}}=0.2.
* •

  Use a cosine cyclic learning rate scheduler with two cycles, initial learning rate 1e-7, final learning rate 1e-7, and max learning rate given by the base model learning rate.

For the stochastic target masking variants, we proportionally increase the number of epochs (e.g., with ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5, half as many targets are observed in a given epoch, so we double the total number of epochs).

###### Small Data Variant Rank Orders.

Table 13: 
Average rank order of variants of NPT-Small (±standard error)plus-or-minusstandard error(\pm\ \text{standard error}) across 10 cross-validation splits on each small dataset.
We determine rank using negative log-likelihood and sort methods by ascending rank for each metric.

|  |  |
| --- | --- |
| Dataset | Boston |
| ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5 | 2.50±0.73uncertain2.500.732.50\pm 0.73 |
| ptarget=0.1subscript𝑝target0.1p\_{\text{target}}=0.1 | 2.50±0.83uncertain2.500.832.50\pm 0.83 |
| 888 →→\rightarrow\  161616 Layers, 888 →→\rightarrow\  161616 Heads | 2.60±0.65uncertain2.600.652.60\pm 0.65 |
| Cosine Cyclic LR Schedule | 3.10±0.75uncertain3.100.753.10\pm 0.75 |
| Base NPT-Small | 3.70±0.84uncertain3.700.843.70\pm 0.84 |
| 888 →→\rightarrow\  161616 Layers | 4.3±0.67uncertain4.30.674.3\pm 0.67 |
| 888 →→\rightarrow\  161616 Heads | 4.40±0.60uncertain4.400.604.40\pm 0.60 |
| pfeature=0.2subscript𝑝feature0.2p\_{\text{feature}}=0.2 | 4.90±0.46uncertain4.900.464.90\pm 0.46 |

|  |  |
| --- | --- |
| Dataset | Breast Cancer |
| ptarget=0.1subscript𝑝target0.1p\_{\text{target}}=0.1 | 2.60±0.92uncertain2.600.922.60\pm 0.92 |
| Base NPT-Small | 2.70±0.65uncertain2.700.652.70\pm 0.65 |
| 888 →→\rightarrow\  161616 Heads | 3.00±0.49uncertain3.000.493.00\pm 0.49 |
| pfeature=0.2subscript𝑝feature0.2p\_{\text{feature}}=0.2 | 3.20±0.68uncertain3.200.683.20\pm 0.68 |
| ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5 | 3.50±0.56uncertain3.500.563.50\pm 0.56 |
| Cosine Cyclic LR Schedule | 4.10±0.89uncertain4.100.894.10\pm 0.89 |
| 888 →→\rightarrow\  161616 Layers, 888 →→\rightarrow\  161616 Heads | 4.40±0.70uncertain4.400.704.40\pm 0.70 |
| 888 →→\rightarrow\  161616 Layers | 4.50±0.81uncertain4.500.814.50\pm 0.81 |

|  |  |
| --- | --- |
| Dataset | Concrete |
| ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5 | 2.30±0.76uncertain2.300.762.30\pm 0.76 |
| Cosine Cyclic LR Schedule | 2.50±0.69uncertain2.500.692.50\pm 0.69 |
| 888 →→\rightarrow 161616 Heads | 2.60±0.62uncertain2.600.622.60\pm 0.62 |
| Base NPT-Small | 2.70±0.52uncertain2.700.522.70\pm 0.52 |
| ptarget=0.1subscript𝑝target0.1p\_{\text{target}}=0.1 | 3.10±0.64uncertain3.100.643.10\pm 0.64 |
| 888 →→\rightarrow 161616 Layers | 3.90±0.80uncertain3.900.803.90\pm 0.80 |
| 888 →→\rightarrow 161616 Layers, 888 →→\rightarrow\  161616 Heads | 5.10±0.66uncertain5.100.665.10\pm 0.66 |
| pfeature=0.2subscript𝑝feature0.2p\_{\text{feature}}=0.2 | 5.8±0.39uncertain5.80.395.8\pm 0.39 |

|  |  |
| --- | --- |
| Dataset | Yacht |
| ptarget=0.1subscript𝑝target0.1p\_{\text{target}}=0.1 | 1.20±0.53uncertain1.200.531.20\pm 0.53 |
| ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5 | 2.70±0.52uncertain2.700.522.70\pm 0.52 |
| 888 →→\rightarrow\  161616 Heads | 2.80±0.66uncertain2.800.662.80\pm 0.66 |
| Cosine Cyclic LR Schedule | 3.10±0.69uncertain3.100.693.10\pm 0.69 |
| Base NPT-Small | 3.60±0.54uncertain3.600.543.60\pm 0.54 |
| 888 →→\rightarrow 161616 Layers | 4.10±0.74uncertain4.100.744.10\pm 0.74 |
| pfeature=0.2subscript𝑝feature0.2p\_{\text{feature}}=0.2 | 5.20±0.47uncertain5.200.475.20\pm 0.47 |
| 888 →→\rightarrow 161616 Layers, 888 →→\rightarrow\  161616 Heads | 5.30±0.83uncertain5.300.835.30\pm 0.83 |

We report the rank order (±plus-or-minus\pm standard error) of these variants in [Table 13](#A3.T13 "In Small Data Variant Rank Orders. ‣ C.1.2 NPT Training on Small Data ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").
A notable trend is that the *target masking configurations perform particularly well*.
One of the two configurations with target masking is the top performer on each of the four datasets.
This could be attributed to some combination of the representational advantage of label masking (cf. [Section 2.6](#S2.SS6 "2.6 Masking and Optimization ‣ 2 Non-Parametric Transformers")), an additional regularization effect akin to dropout, or stabler convergence over a greater number of epochs.

Other configurations did not display similarly obvious trends in performance. This is in concordance with the ablation study ([Section B.4](#A2.SS4 "B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix")) and supports the claim that NPT is robust to changes in hyperparameters.

##### C.1.3 NPT Training on Medium and Large Data

For the medium and large datasets, we again adopt the NPT-Base architecture and optimization hyperparameters, and make minor manual changes on a per-dataset basis to account for differences in number of datapoints and attributes across the datasets.
No more than 333 manual iterations are performed to find these adaptations.
We generally attempt to maximize batch size given a fixed memory budget.
Given the rank order results on small data (cf. [Table 13](#A3.T13 "In Small Data Variant Rank Orders. ‣ C.1.2 NPT Training on Small Data ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix")) we use target masking on the medium and large datasets whenever computationally feasible.121212Training is slower as only a ptargetsubscript𝑝targetp\_{\text{target}} proportion of training labels are used for backpropagation in each epoch.
Therefore, target masking may increase training time beyond our budget.
.
These per-dataset alterations are reported below.

###### UCI Datasets.

We report results for Protein using the Base NPT configuration in the ablation study (cf. [Table 4](#A2.T4 "In B.4 Ablation Study 1: NPT Hyperparameters ‣ Appendix B Additional Results ‣ Appendix")).
On Kick, we use batch size 409640964096, train for 250000250000250000 steps, and use ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5.
On Income, we use batch size 204820482048, train for 200000020000002000000 steps, use no feature masking (and correspondingly fix the tradeoff parameter λ=0𝜆0\lambda=0), and use ptarget=0.15subscript𝑝target0.15p\_{\text{target}}=0.15.
On Poker Hand, we use batch size 409640964096, train for 200000200000200000 steps, use ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5, and stratify by class (i.e., compose training datapoints in each minibatch proportionally to the empirical label distribution of the training set to account for significant class imbalance).
On Forest Cover, we use batch size 180018001800, train for 800000800000800000 steps, use a polynomial decay learning rate scheduler with warmup over the first 1%percent11\% of steps, use base learning rate 0.0050.0050.005, ptarget=0.5subscript𝑝target0.5p\_{\text{target}}=0.5, and class balancing as above.
The changes to learning rate scheduling were made to speed up training and hence save compute resources.
On Higgs, we use batch size 409640964096, train for 500000500000500000 steps, and do not use target masking due to constraints on compute time.

###### Image Data (CIFAR-10 and MNIST).

See [Section B.8](#A2.SS8 "B.8 Image Classification Results ‣ Appendix B Additional Results ‣ Appendix") for details on the image data architecture and setup.

Again, we stress that the vast majority of hyperparameters used on all datasets (small, medium, and large benchmarks from UCI as well as the image benchmarks) are identical; configurations follow NPT-Base (cf. [Section C.1.1](#A3.SS1.SSS1 "C.1.1 NPT-Base Architecture ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix")) very closely and changes usually affect NPT optimization rather than architecture.

#### C.2 Further Details on ABD and ABA Layers

##### C.2.1 Dropout

In practice, we apply elementwise dropout on the attention scores exp​(𝑸​𝑲⊤/h)exp𝑸superscript𝑲topℎ\text{exp}(\bm{Q}\bm{K}^{\top}/\sqrt{h}), as well as on the input/output embeddings and the output of the MHSelfAtt​(⋅)MHSelfAtt⋅\text{MHSelfAtt}(\cdot) function (often referred to as attention and hidden dropout).

#### C.3 Input and Output Embdedings

##### C.3.1 Input Embedding

At a high-level, we embed inputs by encoding categorical attributes as one-hot vectors and standardizing continuous attributes, followed by a learned linear embedding for each attribute to obtain InputEmbed​(𝑿)=𝑯(0)∈ℝn×d×eInputEmbed𝑿superscript𝑯0superscriptℝ𝑛𝑑𝑒\text{InputEmbed}(\bm{X})=\bm{H}^{(0)}\in\mathbb{R}^{n\times d\times e}.

More specifically, we perform the following sequence of steps:
Attributes 𝑿:,j,j∈{1,…,d}

subscript𝑿

:𝑗𝑗
1…𝑑\bm{X}\_{:,j},j\in\{1,\dots,d\} of the input matrix can be either continuous or categorical.
We first apply a function Encode​(⋅)Encode⋅\text{Encode}(\cdot) to each attribute 𝑿:,jsubscript𝑿

:𝑗\bm{X}\_{:,j}. This “encodes” categorical attributes with a one-hot representation and standardizes continuous attributes to zero mean and unit standard deviation.
Each encoded attribute j𝑗j has (potentially unique) dimensions n×ej𝑛subscript𝑒𝑗n\times e\_{j}.
Then, we concatenate this encoded attribute with its respective column of the masking matrix 𝑴:,jsubscript𝑴

:𝑗\bm{M}\_{:,j} along the second dimension to produce a column encoding of dimensions n×(ej+1)𝑛subscript𝑒𝑗1n\times(e\_{j}+1).
We learn separate embedding weights for each attribute 𝑾jin∈ℝ(ej+1)×esubscriptsuperscript𝑾in𝑗superscriptℝsubscript𝑒𝑗1𝑒\bm{W}^{\text{in}}\_{j}\in\mathbb{R}^{(e\_{j}+1)\times e} that embed all attributes to a common hidden dimension e𝑒e.
Altogether, we can state the embedding of a single attribute column 𝑿:,jsubscript𝑿

:𝑗\bm{X}\_{:,j} as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑯:,j(0)=concataxis=e​(Encode​(𝑿:,j),𝑴:,j)​𝑾jin+𝑯:,jIndex+𝑯:,jType,subscriptsuperscript𝑯0  :𝑗axis𝑒concatEncodesubscript𝑿  :𝑗subscript𝑴  :𝑗subscriptsuperscript𝑾in𝑗subscriptsuperscript𝑯Index  :𝑗subscriptsuperscript𝑯Type  :𝑗\displaystyle\bm{H}^{(0)}\_{:,j}=\underset{\text{axis}=e}{\text{concat}}(\text{Encode}(\bm{X}\_{:,j}),\bm{M}\_{:,j})\bm{W}^{\text{in}}\_{j}+\bm{H}^{\text{Index}}\_{:,j}+\bm{H}^{\text{Type}}\_{:,j}, |  | (22) |

where 𝑯:,jIndex∈ℝn×esubscriptsuperscript𝑯Index

:𝑗superscriptℝ𝑛𝑒\bm{H}^{\text{Index}}\_{:,j}\in\mathbb{R}^{n\times e} is a learnt embedding for the index and 𝑯:,jType∈ℝn×esubscriptsuperscript𝑯Type

:𝑗superscriptℝ𝑛𝑒\bm{H}^{\text{Type}}\_{:,j}\in\mathbb{R}^{n\times e} for the type (either continuous or categorical) of attribute j𝑗j.

Finally, we write the full NPT input embedding layer as

|  |  |  |  |
| --- | --- | --- | --- |
|  | InputEmbed​(𝑿)=stackaxis=d​(𝑯:,1(0),…,𝑯:,d(0))=𝑯(0)∈ℝn×d×e.InputEmbed𝑿axis𝑑stacksubscriptsuperscript𝑯0  :1…subscriptsuperscript𝑯0  :𝑑superscript𝑯0superscriptℝ𝑛𝑑𝑒\displaystyle\text{InputEmbed}(\bm{X})=\underset{\text{axis}=d}{\text{stack}}(\bm{H}^{(0)}\_{:,1},\dots,\bm{H}^{(0)}\_{:,d})=\bm{H}^{(0)}\in\mathbb{R}^{n\times d\times e}. |  | (23) |

The stack operation constructs 𝑯(0)∈ℝn×d×esuperscript𝑯0superscriptℝ𝑛𝑑𝑒\bm{H}^{(0)}\in\mathbb{R}^{n\times d\times e} from d𝑑d attribute embeddings 𝑯:,j(0)∈ℝn×e,j∈{1,…​d}formulae-sequencesubscriptsuperscript𝑯0

:𝑗superscriptℝ𝑛𝑒𝑗1…𝑑\bm{H}^{(0)}\_{:,j}\in\mathbb{R}^{n\times e},j\in\{1,\dots d\}.

##### C.3.2 Output Embedding

For an NPT with L𝐿L layers, we obtain an output prediction by applying a learnt linear output embedding (that closely mirrors the process of the input embedding) to the output of the last attention layer 𝑯(L)superscript𝑯𝐿\bm{H}^{(L)}.
We write the output embedding layer as

|  |  |  |  |
| --- | --- | --- | --- |
|  | OutputEmbed​(𝑯(L))=[𝒁:,1,…,𝒁:,d]=𝒁,OutputEmbedsuperscript𝑯𝐿  subscript𝒁  :1…subscript𝒁  :𝑑𝒁\displaystyle\text{OutputEmbed}(\bm{H}^{(L)})=[\bm{Z}\_{:,1},\dots,\bm{Z}\_{:,d}]=\bm{Z}, |  | (24) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | where ​𝒁:,j=𝑯:,j,:(L)​𝑾jout.where subscript𝒁  :𝑗subscriptsuperscript𝑯𝐿  :𝑗:subscriptsuperscript𝑾out𝑗\displaystyle\text{where\ }\bm{Z}\_{:,j}=\bm{H}^{(L)}\_{:,j,:}\bm{W}^{\text{out}}\_{j}. |  | (25) |

Our prediction 𝒁𝒁\bm{Z} is a list of d𝑑d attribute predictions 𝒁j∈ℝn×ejsubscript𝒁𝑗superscriptℝ𝑛subscript𝑒𝑗\bm{Z}\_{j}\in\mathbb{R}^{n\times e\_{j}}.
We learn output embedding weights 𝑾jout∈ℝe×ejsubscriptsuperscript𝑾out𝑗superscriptℝ𝑒subscript𝑒𝑗\bm{W}^{\text{out}}\_{j}\in\mathbb{R}^{e\times e\_{j}} which are applied on attribute slices 𝑯:,j,:(L)∈ℝn×esubscriptsuperscript𝑯𝐿

:𝑗:superscriptℝ𝑛𝑒\bm{H}^{(L)}\_{:,j,:}\in\mathbb{R}^{n\times e} of the output of the L𝐿Lth layer 𝑯(L)∈ℝn×d×esuperscript𝑯𝐿superscriptℝ𝑛𝑑𝑒\bm{H}^{(L)}\in\mathbb{R}^{n\times d\times e}.
Note that the second dimension of each attribute prediction 𝒁jsubscript𝒁𝑗\bm{Z}\_{j} is determined by the encoding size (i.e., ej=1subscript𝑒𝑗1e\_{j}=1 for continuous attributes, ejsubscript𝑒𝑗e\_{j} is the number of categories for a categorical attribute) as in the input embedding.
Note also that we do not predict a mask value (i.e., we do not predict to dimensions n×(ej+1)𝑛subscript𝑒𝑗1n\times(e\_{j}+1) for each attribute).
To obtain the final prediction matrix 𝑿^∈ℝn×d^𝑿superscriptℝ𝑛𝑑\hat{\bm{X}}\in\mathbb{R}^{n\times d} we take the arg⁡max\arg\max over the categorical predictions.

#### C.4 NPT Masking

##### C.4.1 Handling Missing Values

Real-world data – particularly tabular data – often contains *missing entries*.
Many popular models for supervised prediction on tabular data cannot accommodate missing values as input.
Instead they require that missing features are *imputed*, i.e., an additional model predicts a surrogate value for what the missing values could have been, such that the supervised model then receives a “clean” dataset as input which no longer overtly contains missing values.

For example, all scikit-learn [[69](#bib.bib69)] predictors, including Gradient Boosting and Random Forests, require an explicit imputation step before training.
Often, extremely simple imputation methods are used in practice.
For example, TabNet [[2](#bib.bib2)] drops datapoints with >10% missing entries and otherwise applies univariate mean imputation as part of a Google AI Platform pipeline [[70](#bib.bib70)]; and CatBoost [[71](#bib.bib71)] treats a missing continuous entry as the minimum or maximum of that feature (univariate min/max imputation), or raises an error.
While more complex imputation methods could in theory be applied as pre-processing [[88](#bib.bib88), [50](#bib.bib50), [43](#bib.bib43), [81](#bib.bib81), [82](#bib.bib82)], there will always remain a separation between the imputation step and the prediction model.
Additionally, more complex imputation methods often require training and hyperparameter selection, such that the combined imputation and prediction process becomes cumbersome.
Both for practical as well as performance reasons, it is desirable to have a single model that can *directly* handle missing data, learn complex internal imputation operations from the data, and at the same time learn the desired predictive function from features to target.

This is exactly what NPTs achieve.
They are able to accommodate inputs with missing values gracefully without requiring any imputation pre-processing steps, therefore modeling data with missing values end-to-end.
We can explicitly indicate that a value 𝑿i,jsubscript𝑿

𝑖𝑗\bm{X}\_{i,j} is missing by simply setting the mask token 𝑴i,j=1subscript𝑴

𝑖𝑗1\bm{M}\_{i,j}=1.
Already in standard NPTs, the stochastic feature masking during training teaches NPTs to predict values for which 𝑴i,j=1subscript𝑴

𝑖𝑗1\bm{M}\_{i,j}=1 while ignoring the value of their entry 𝑿i,jsubscript𝑿

𝑖𝑗\bm{X}\_{i,j} at input.
Further, no choice of fixed imputation algorithm has to be made with NPTs.
Instead, NPTs learn directly from the data how to make predictions given missing values.
Attention between datapoints might be particularly useful for learning a general mechanism of how to impute missing values by attending to other datapoints.
We therefore suspect that NPTs could be a strong contender for predicting on data with missing values.
Further, unlike common imputation pre-processing, NPTs do not discard the information of *which* attributes were missing.
Future work could also explore the ability of NPT to model arbitrary correlations underlying the pattern of which data is missing, i.e., datasets where values are not missing at random.

##### C.4.2 Masking Encompasses Many Common Machine Learning Settings

The flexible masking mechanism of NPTs can be used to accommodate a variety of common machine learning settings.

Multi-Target Prediction.
In *multi*-target classification or regression, more than one column of the dataset contains targets.
Standard supervised models often do not support multi-output settings and must resort to training multiple models, one for each target.
NPTs can accommodate multi-target prediction trivially, since they learn to make predictions at any masked input entry.
For prediction in a multi-target setting, we simply apply target masking on all columns with targets.

Self-Supervision.
In self-supervised learning, we are often interested in learning a generative model or useful encoding from unlabeled data.
The reconstruction of corrupted input features as part of stochastic feature masking can already be seen as self-supervised learning.
The stochastic masking mechanism allows NPTs to learn to predict masked out values anywhere in the input.
In theory, NPTs should be able to learn a fully generative model of the dataset in this manner.

Semi-Supervision.
In semi-supervised learning, we hope to use large quantities of unlabeled data to aid in learning a predictive function on a small set of labeled data.
Often, this involves a two-step process, such as learning a powerful autoencoder from all data and then training a predictor using the learnt encoder and the small set of labeled data.
NPTs can accommodate semi-supervised learning without changes to the architecture.
Specifically, we can include large amounts of unlabeled data by simply appending those feature values to the labeled input dataset.
We indicate that no labels are available for all unlabeled datapoints i′superscript𝑖′i^{\prime} by setting their mask token at the target column 𝑿i′,d=1subscript𝑿

superscript𝑖′𝑑1\bm{X}\_{i^{\prime},d}=1.
NPTs can use attention between datapoints to make use of information from the features of the unlabeled datapoints.

Imputation.
With imputation, we refer to scenarios where the main task is to predict missing values for arbitrary attributes and datapoints.
Similar to self-supervision, NPTs already learn how to do this from the stochastic masking mechanism that is enabled by default.
(Unlike for the self-supervision category, the imputation scenario assumes that there are actually some missing values that we would like to predict.)

##### C.4.3 Stochastic Masking: Details

For stochastic masking, a specified proportion of training entries (we default to 15% following [[24](#bib.bib24)]) are selected for masking at the start of each epoch.
Among those entries chosen, we mask out the value with 90% probability and randomize it with 10% probability.
“Masking out” means that the original value 𝑿i,jsubscript𝑿

𝑖𝑗\bm{X}\_{i,j} is overwritten with zeros and the mask token is set to 1.
Randomization is done for categorical targets by sampling a new class uniformly at random.
Continuous targets are sampled from a standard Normal 𝒩​(0,1)𝒩01\mathcal{N}(0,1).

This sampling scheme is applied for both stochastic feature masking and stochastic target masking, where we allow for different masking proportions between the two (pfeaturesubscript𝑝featurep\_{\text{feature}} and ptargetsubscript𝑝targetp\_{\text{target}}).
During training, a loss is backpropagated on the masked entries.

#### C.5 NPT Optimization

Each of the losses ℒFeaturessuperscriptℒFeatures\mathcal{L}^{\text{Features}} (feature loss) and ℒTargetssuperscriptℒTargets\mathcal{L}^{\text{Targets}} (target loss) is normalized by the number of entries on which it is evaluated.

As described in [Section C.1.1](#A3.SS1.SSS1 "C.1.1 NPT-Base Architecture ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix"): we anneal the λ𝜆\lambda parameter in the NPT objective using a cosine schedule, i.e., starting with full weight on the feature loss term at epoch 0 and annealing to full weight on the target loss term by the end of training.
We use LAMB [[98](#bib.bib98)] with Lookahead [[99](#bib.bib99)] for optimization, which we find to perform well with large minibatches.
We use a flat-then-anneal learning rate schedule with cosine decay, notable as Transformer works [[90](#bib.bib90), [24](#bib.bib24)] often report that a linear learning rate warmup is necessary for training stability.
Our placement of Layer Normalization before self-attention (“pre-LayerNorm” [[3](#bib.bib3), [16](#bib.bib16)]) may contribute to our not needing this.

### Appendix D Related Work – Continued

#### D.1 Tree-Based Baselines

Tree-based approaches in machine learning have been popular for over half a century [[62](#bib.bib62), [64](#bib.bib64), [11](#bib.bib11)].
Each node of a tree splits the data into smaller subsets, and predictions are made at each of the leaves.
The splits are learned from a set of training data by minimizing some objective function.
Many established methods combine predictions of multiple trees through bagging [[9](#bib.bib9)] and/or boosting [[78](#bib.bib78)].
Bagging uses an ensemble of trees, each learned by training on a random subsample of the data. This approach is most popularly used in Random Forests [[10](#bib.bib10)].
Boosting learns a sequence of trees, conditioning the learning of each additional model on the predictions of previous models, with the aim of reducing overall prediction error.

Popular examples of tree-based boosting models include AdaBoost [[31](#bib.bib31)], XGBoost [[17](#bib.bib17)], CatBoost [[71](#bib.bib71)], and LightGBM [[48](#bib.bib48)].
To date, boosting arguably comprises the most popular approach for tabular data prediction.
These models often rely on careful tuning of a large variety of hyperparameters.
However, training cost is often cheap compared to neural network architectures, and therefore, so is hyperparameter optimization.
This balance is slightly offset for NPTs, which seem largely robust to hyperparameter tuning.
Hence, the training of a single NPT is often competitive to a grid search over hyperparameters for a tree-based model.

### Appendix E Classification and Regression Benchmark Details

#### E.1 General Setup

For certain datasets we use a canonical fixed test set.
Otherwise, we default to 10-fold cross validation with 0.7/0.2/0.1 splits on smaller datasets and a single 0.7/0.1/0.2 split on larger datasets, where the exact split indices are always consistent across baselines.
The full details on all UCI benchmark datasets are given in [Tables 14](#A5.T14 "In E.1 General Setup ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix") and [15](#A5.T15 "Table 15 ‣ E.1 General Setup ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix").
Note the variety of the datasets across number of instances, number of features, composition (categorical or continuous) of features, and task (multi-class classification, binary classification, and regression).

Table 14: UCI classification dataset statistics and experimental setup details.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset | Higgs Boson | Poker Hand | Forest Cover | Income | Kick | Breast Cancer |
| # Instances | 11,000,0001100000011,000,000 | 1,025,01010250101,025,010 | 581,012581012581,012 | 299,285299285299,285 | 72,9837298372,983 | 569569569 |
| # Features | 282828 | 101010 | 545454 | 424242 | 323232 | 313131 |
| # Categorical Features | 00 | 101010 | 444444 | 363636 | 181818 | 00 |
| # Continuous Features | 282828 | 00 | 101010 | 666 | 141414 | 313131 |
| # Classes | 222 | 101010 | 777 | 222 | 222 | 222 |
| Train/Val/Test Split | 0.840.840.84/0.120.120.12/0.050.050.05 | 0.0170.0170.017/0.0030.0030.003/0.980.980.98 | 0.70.70.7/0.10.10.1/0.20.20.2 | 0.570.570.57/0.10.10.1/0.330.330.33 | 0.70.70.7/0.10.10.1/0.20.20.2 | 0.70.70.7/0.20.20.2/0.10.10.1 |
| Fixed Test Set | Yes | Yes | No | Yes | No | No (10-Fold CV) |
| Uses Minibatching | Yes | Yes | Yes | Yes | Yes | No |




Table 15: UCI regression dataset statistics and experimental setup details.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Dataset | Protein | Concrete | Boston | Yacht |
| # Instances | 45,7304573045,730 | 103010301030 | 506506506 | 308308308 |
| # Features | 999 | 999 | 131313 | 666 |
| # Categorical Features | 00 | 00 | 222 | 555 |
| # Continuous Features | 999 | 999 | 111111 | 111 |
| Train/Val/Test Split | 0.70.70.7/0.10.10.1/0.20.20.2 | 0.70.70.7/0.20.20.2/0.10.10.1 | 0.70.70.7/0.20.20.2/0.10.10.1 | 0.70.70.7/0.20.20.2/0.10.10.1 |
| Fixed Test Set | No | No (10-Fold CV) | No (10-Fold CV) | No (10-Fold CV) |
| Uses Minibatching | Yes | No | No | No |

#### E.2 Hyperparameter Tuning

##### E.2.1 Overview

Table 16: 
Number of unique hyperparameter configurations swept over for each model class and dataset. Here we shorten Boston Housing to BH, Breast Cancer to BC, Poker Hand to PH, Forest Cover to FC, and Higgs Boson to HB. Datasets are ordered by increasing number of datapoints (n𝑛n) from left to right.
  
\* TabNet on Protein, Kick, and Income is tuned by sweeping over all 6 configurations listed in the original paper [[2](#bib.bib2)] in addition to the default configuration.
Note that these configs include one tuned on Income.
  
††\dagger TabNet on Poker Hand, Forest Cover, and Higgs Boson use precisely the configuration specified for those datasets in the original paper [[2](#bib.bib2)].
  
‡‡\ddagger For some of these, we manually optimized convergence of the validation loss by adjusting non-architectural parameters such as learning rate (schedule), batch size, or number of steps in at most 333 iterations.
See [C.1.3](#A3.SS1.SSS3 "C.1.3 NPT Training on Medium and Large Data ‣ C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Yacht | BH | BC | Concrete | Protein | Kick | Income | PH | FC | HB |
| Random Forest | 242424 | 242424 | 242424 | 242424 | 242424 | 242424 | 242424 | 242424 | 242424 | 242424 |
| Gradient Boosting | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 |
| XGBoost | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 |
| CatBoost | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 |
| LightGBM | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 | 484848 |
| MLP | 11,3401134011,340 | 11,3401134011,340 | 11,3401134011,340 | 11,3401134011,340 | 270270270 | 270270270 | 270270270 | 270270270 | 270270270 | 666 |
| k-NN | 480480480 | 480480480 | 480480480 | 480480480 | 404040 | 404040 | 404040 | 404040 | 404040 | - |
| TabNet | 484848 | 484848 | 484848 | 484848 | 777\* | 777\* | 777\* | 1†superscript1†$1$^{\dagger} | 1†superscript1†$1$^{\dagger} | 1†superscript1†$1$^{\dagger} |
| NPT | 888 | 888 | 888 | 888 | 1‡superscript1‡$1$^{\ddagger} | 1‡superscript1‡$1$^{\ddagger} | 1‡superscript1‡$1$^{\ddagger} | 1‡superscript1‡$1$^{\ddagger} | 1‡superscript1‡$1$^{\ddagger} | 1‡superscript1‡$1$^{\ddagger} |

Table [16](#A5.T16 "Table 16 ‣ E.2.1 Overview ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix") lists the number of unique hyperparameter configurations swept over for each baseline and classification/regression dataset.

All details on the NPT hyperparameter setup are given in [Section C.1](#A3.SS1 "C.1 NPT Training and Hyperparameters ‣ Appendix C Additional Details on the NPT Architecture ‣ Appendix").
Note that for any given dataset, NPT is tuned over fewer configurations than the baselines:
we fix a base model configuration with minimal data-dependent tuning of hyperparameters such as learning rate, scheduler, number of steps, and target masking percentage pfeaturesubscript𝑝featurep\_{\text{feature}}, and choose the largest batch size viable for our hardware.
On small datasets, we then sweep over 8 variants, and on medium and large datasets (including image data) use only the fixed variant with minor modifications.

In the case of TabNet, the configurations used for Poker Hand, Forest Cover, and Higgs Boson are those reported by the original authors for these datasets [[2](#bib.bib2)]; for Income, we performed a sweep over configurations including one reported for that dataset in the original publication.
All deep learning approaches (MLP, TabNet, and NPT) use early stopping on the validation target loss.

##### E.2.2 Baseline Sweep Details

We report hyperparameter sweep details for baselines below.
The associated tables for each baseline give the bounds of the search space for numerical hyperparameters and all values for categorical hyperparameters.
We clarify specific hyperparameters and provide context where helpful.

###### Random Forest (Tables [17](#A5.T17 "Table 17 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix"), [18](#A5.T18 "Table 18 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix")).

criterion refers to the split criterion. max\_features is the number of features to consider when looking for the best split.

###### Gradient Boosting, XGBoost, LightGBM, and CatBoost (Table [19](#A5.T19 "Table 19 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix")).

See [D.1](#A4.SS1 "D.1 Tree-Based Baselines ‣ Appendix D Related Work – Continued ‣ Appendix") for background on tree-based baselines.

###### MLP (Tables [20](#A5.T20 "Table 20 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix"), [21](#A5.T21 "Table 21 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix"), [22](#A5.T22 "Table 22 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix")).

The invscaling learning\_rate scheduler scales with αt=α0/t0.5subscript𝛼𝑡subscript𝛼0superscript𝑡0.5\alpha\_{t}=\alpha\_{0}/t^{0.5} where t𝑡t is the step, α0subscript𝛼0\alpha\_{0} the initial learning rate, and αtsubscript𝛼𝑡\alpha\_{t} the learning rate at step t𝑡t.
The adaptive learning\_rate divides the current learning rate by 5 when two consecutive epochs fail to decrease training or validation log loss by a tolerance 1e-4.
Due to compute constraints, we decreased the size of the search space as the dataset size increased by focusing on 3-layer networks, lower L2 penalties, and higher batch sizes.

###### k-NN (Tables [23](#A5.T23 "Table 23 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix"), [24](#A5.T24 "Table 24 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix"), [25](#A5.T25 "Table 25 ‣ k-NN (Tables 23, 24, 25). ‣ E.2.2 Baseline Sweep Details ‣ E.2 Hyperparameter Tuning ‣ Appendix E Classification and Regression Benchmark Details ‣ Appendix")).

weights describes the weight function applied to the neighborhood, i.e., “distance” means that closer neighbors of a query point have greater influence than those further away.
algorithm specifies the underlying k-NN algorithm, where KD Tree [[7](#bib.bib7)] and Ball Tree [[60](#bib.bib60)] are approximations of brute-force search.
The “auto” setting determines an appropriate algorithm based on the input data [[69](#bib.bib69)].
leaf\_size is a hyperparameter of KD Tree and Ball Tree.
p is the power parameter for the distance metric, i.e., p=1p1\texttt{p}=1 yields Manhattan and p=2p2\texttt{p}=2 Euclidean distance.
It was computationally infeasible for us to obtain reasonable results on the 11M instance Higgs Boson dataset.
Even when attempting approximate 3-NN on an Azure D64 v3 instance with 256 GB RAM, we encountered an out-of-memory error.

Table 17: 
Random Forest classification hyperparameters.

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | criterion | n\_estimators | max\_features |
| Setting | gini, entropy | [50, 1000] | auto, sqrt, log2 |




Table 18: 
Random Forest regression hyperparameters.

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | criterion | n\_estimators | max\_features |
| Setting | mae, mse | [50, 1000] | auto, sqrt, log2 |




Table 19: 
Gradient Boosting, XGBoost, LightGBM, and CatBoost hyperparameters (for both regression and classification).

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | learning\_rate | max\_depth | n\_estimators |
| Setting | [1e-3, 0.3] | [3, 10] | [50, 1000] |




Table 20: 
MLP hyperparameters for small datasets (Boston Housing, Breast Cancer, Concrete, and Yacht).

|  |  |  |
| --- | --- | --- |
| Hyperparameter | hidden\_layer\_sizes | l2\_penalty |
| Setting | [(25)-(500), (25,25)-(500,500), (25,25,25)-(500,500,500)] | [0, 1] |

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | batch\_size | learning\_rate | learning\_rate\_init |
| Setting | [32, 256] | constant, invscaling, adaptive | [1e-5, 1e-1] |




Table 21: 
MLP hyperparameters for medium and large datasets other than Higgs Boson (Protein, Kick, Income, Poker Hand, Forest Cover).

|  |  |  |
| --- | --- | --- |
| Hyperparameter | hidden\_layer\_sizes | l2\_penalty |
| Setting | [(25,25,25)-(500,500,500)] | [0, 1e-2] |

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | batch\_size | learning\_rate | learning\_rate\_init |
| Setting | [128, 256] | constant, invscaling, adaptive | [1e-5, 1e-1] |




Table 22: 
MLP hyperparameters for the Higgs Boson dataset.

|  |  |  |
| --- | --- | --- |
| Hyperparameter | hidden\_layer\_sizes | l2\_penalty |
| Setting | (500,500,500) | 0 |

|  |  |  |  |
| --- | --- | --- | --- |
| Hyperparameter | batch\_size | learning\_rate | learning\_rate\_init |
| Setting | [512, 1024] | constant | [1e-4, 1e-2] |




Table 23: 
k-NN hyperparameters for small datasets (Boston Housing, Breast Cancer, Concrete, and Yacht).

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Hyperparameter | n\_neighbors | weights | algorithm | leaf\_size | p |
| Setting | [2, 100] | uniform, distance | ball\_tree, kd\_tree, brute | [10, 100] | 1, 2 |




Table 24: 
k-NN hyperparameters for medium-large datasets (Protein, Kick, Income, Poker Hand).

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Hyperparameter | n\_neighbors | weights | algorithm | leaf\_size | p |
| Setting | [2, 1000] | distance | auto | [10, 100] | 2 |




Table 25: 
k-NN hyperparameters for Forest Cover.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Hyperparameter | n\_neighbors | weights | algorithm | leaf\_size | p |
| Setting | [2, 25] | distance | auto | [10, 100] | 2 |

### Appendix F Societal Impacts of NPT

We have introduced Non-Parametric Transformers, a novel deep learning architecture that predicts by including learned interactions between points of the dataset.
In this work, we take first steps towards exploring NPTs and their properties.
We do not recommend that NPTs are carelessly applied in production settings, because we do not yet know enough about them.
We now list common concerns in applying machine learning models, discuss how they may apply to NPTs, and how to potentially mitigate them.

Many countries of the world, such as the US, UK, and the countries of the EU, are implementing “Right to Explanation”-schemes that grant those affected by autonomous decisionmaking the right to an explanation of why and how decisions were made.
In general, Transformer-based architectures such as NPT have been shown to be amenable to explanations, see e.g., [[90](#bib.bib90)].
One could argue that our experiments in §[4](#S4.F4 "Figure 4 ‣ 4.4 NPTs Rely on Similar Datapoints for Predictions on Real Data ‣ 4 Experiments") move in an explanatory direction.
However, we have not sufficiently investigated the explanations of individual NPT decisions, and believe this to be exciting future work.

Machine learning models are increasingly used in autonomous decision making that affects human beings in some capacity, e.g., clinical diagnosis, autonomous driving, and detection of toxic comments online.131313For example, see [[28](#bib.bib28), [63](#bib.bib63), [87](#bib.bib87)].
It is of great importance that those decisions are *fair*, i.e., that they do not discriminate against underrepresented groups in some manner.
We have not yet investigated how NPTs respond to common techniques of calibrating machine learning models to fulfil some definition of fairness.
We believe that their special predictive behavior from similar datapoints likely poses both challenges and opportunities in this domain.
For example, instead of needing to retrain the model to elicit changes in prediction – which could be infeasible in a real-world deployment – NPT could be “prompted” with a different set of context datapoints to modify its predictive behavior towards a more socially desirable response.

In large architectures based on Transformers, the memorization of training data is a common concern.
If the model memorizes training data, adversarial attacks can be used to extract training data from the model weights, see e.g., [[14](#bib.bib14)].
This can lead to violations of privacy if, for example, a publicly available model was trained on data that must remain private.
This can also cause more subtle problems; for example, if training data “lives on” in the model but must be deleted at some point in time to comply with privacy regulations.
As NPT directly relies on training data as input for prediction, NPT is not a “private” model per definition.
However, we can imagine future work tackling this question; for example, by learning to predict from a set of anonymous representative points instead of the training data directly.

At the model sizes presented in the paper, the environmental impact of training and using NPT is relatively small compared to some of the large architectures currently in fashion, see e.g., [[12](#bib.bib12)].
However, NPT could be scaled up to larger sizes at which point the energy used for training and prediction would become a serious concern.
When considering tabular data, training a *single* NPT model is expensive compared to training a *single* one of our tree-based baselines such as XGBoost.
However, we find that such baselines are often more sensitive to correctly tuned hyperparameters than NPT, such that the total compute including hyperparameter tuning of NPT and the baselines is actually often similar, particularly on larger datasets.
Sparse approximations as referenced in [Section 5](#S5 "5 Limitations, Future Work, and Conclusions") may further reduce the computational impact of NPT.

NPT is a new – and exciting – architecture.
Therefore, in applications where explanations, fairness, or privacy are desired or legally required, we do not recommend that NPT be used at this stage.

### Appendix G Code, Computational Resources, and License

###### Code.

We release code for NPTs at [github.com/OATML/Non-Parametric-Transformers](https://github.com/OATML/Non-Parametric-Transformers).
The codebase relies on PyTorch [[68](#bib.bib68)] and NumPy [[40](#bib.bib40)], and we use Scikit-Learn [[69](#bib.bib69)] for many of the baseline experiments.

###### Computational Resources.

For the experiments we mainly rely on a shared internal cluster that has both NVIDIA Titan RTX GPUs (24 GB memory) as well as NVIDIA GeForce RTX 2080 Tis (12 GB memory).
For tuning baselines, which are often compute-heavy workloads, we use Azure D-series compute-optimized VMs.
For small datasets (<1000absent1000<1000 datapoints) such as Breast Cancer, training and evaluation of a single NPT model takes about 10 minutes.
For larger datasets such as Protein (<100000absent100000<$100000$ datapoints), training and evaluation of NPT takes about 10 hours.
For the largest datasets, e.g., Higgs Boson with 11 million datapoints, training and evaluation of NPT takes about 5 days.
We did not optimize NPT for efficiency or training speed in this paper and suspect that convergence could be drastically improved with relatively little effort.
The total amount of compute used for this paper is given by all NPT and baseline runs with repetitions for cross-validation, which amounts to more than 30 GPU days.

###### License Agreements.

123
  
License agreement of the CIFAR-10 dataset:
CIFAR-10 is published under MIT license.

License agreement of the MNIST dataset:
License: Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

UCI Machine Learning Repository:
Licenses for all datasets can be found at [archive.ics.uci.edu/ml/](https://archive.ics.uci.edu/ml/index.php).

[◄](/html/2106.02583)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2106.02584)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2106.02584)
[View original  
on arXiv](https://arxiv.org/abs/2106.02584)[►](/html/2106.02585)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Mar 19 10:09:26 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
