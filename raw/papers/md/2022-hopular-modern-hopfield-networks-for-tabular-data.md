---
arxiv: '2206.00664'
authors:
- 'Bernhard Schäfl 2 2 footnotemark: 2 , , ~{}^{,} Lukas Gruber 2 2 footnotemark:
  2 Angela Bitto-Nemling 2 2 footnotemark: 2 , , ~{}^{,} 3 3 footnotemark: 3 Sepp
  Hochreiter 2 2 footnotemark: 2 , , ~{}^{,} 3 3 footnotemark: 3 2 2 footnotemark:
  2 ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler
  University Linz, Austria 3 3 footnotemark: 3 Institute of Advanced Research in Artificial
  Intelligence (IARAI)'
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Hopular: Modern Hopfield Networks for Tabular Data'
url: https://arxiv.org/abs/2206.00664
year: 2022
---

[2206.00664] Hopular: Modern Hopfield Networks for Tabular Data














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



# Hopular: Modern Hopfield Networks for Tabular Data

Bernhard Schäfl22footnotemark: 2 ,,~{}^{,}  Lukas Gruber22footnotemark: 2  Angela Bitto-Nemling22footnotemark: 2 ,,~{}^{,}33footnotemark: 3  Sepp Hochreiter22footnotemark: 2 ,,~{}^{,}33footnotemark: 3
  
  
22footnotemark: 2  ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning,
  
  Johannes Kepler University Linz, Austria
  
33footnotemark: 3  Institute of Advanced Research in
Artificial Intelligence (IARAI)
Corresponding author: Bernhard Schäfl <[schaefl@ml.jku.at](mailto:schaefl@ml.jku.at)>

###### Abstract

While Deep Learning excels in structured data as encountered in vision and
natural language processing, it failed to meet its expectations on tabular data.
For tabular data, Support Vector Machines (SVMs), Random Forests,
and Gradient Boosting are the best performing techniques with Gradient Boosting in the lead.
Recently, we saw a surge of Deep Learning methods that were tailored to tabular data
but still underperform compared to Gradient Boosting on small-sized datasets.
We suggest “Hopular”, a novel Deep Learning architecture for medium- and
small-sized datasets,
where each layer is
equipped with continuous modern Hopfield networks. The modern Hopfield networks
use stored data to identify feature-feature,
feature-target, and sample-sample dependencies.
Hopular’s novelty is that every layer can directly access
the original input as well as the whole training set
via stored data in the Hopfield networks.
Therefore,
Hopular can step-wise update its current model and the resulting
prediction at every layer
like standard iterative learning algorithms.
In experiments on small-sized tabular datasets with less than 1,000 samples,
Hopular surpasses Gradient Boosting, Random Forests, SVMs,
and in particular several Deep Learning methods.
In experiments on medium-sized tabular data with about 10,000 samples,
Hopular outperforms XGBoost, CatBoost, LightGBM and
a state-of-the art Deep Learning method designed for tabular data.
Thus, Hopular is a strong alternative to these methods on tabular data.

## 1 Introduction

Deep Learning has led to tremendous success
in vision and natural language processing, where it
excelled on large image and text corpora (LeCun et al., [2015](#bib.bib27); Schmidhuber, [2015](#bib.bib35)).
While it yielded competitive results on large tabular datasets Avati et al. ([2018](#bib.bib3)); Simm et al. ([2018](#bib.bib40)); Zhang et al. ([2019b](#bib.bib52)); Mayr et al. ([2018](#bib.bib28)),
so far it could not convince on small tabular data.
However, in real-world settings,
small tabular datasets with less than 10,000 samples are ubiquitous.
They are found in life sciences, when building a model for a certain disease
with a limited number of patients,
for bio-assays in drug design, or for the effect of environmental soil contamination.
The same situation appears in most industrial applications, when
a company wants to predict customer behavior, to control processes,
to optimize its logistics, to market new products, or to employ predictive maintenance.
The omnipresence of small tabular datasets can also be witnessed at Kaggle challenges.
On small-sized and medium-sized tabular datasets with less than 10,000 samples,
Support Vector Machines (SVMs) (Boser et al., [1992](#bib.bib5); Cortes & Vapnik, [1995](#bib.bib8); Schölkopf & Smola, [2002](#bib.bib36)),
Random Forests (Ho, [1995](#bib.bib20); Breiman, [2001](#bib.bib6)) and, in particular,
Gradient Boosting (Friedman, [2001](#bib.bib16))
typically outperform Deep Learning methods with
Gradient Boosting having the edge.
In real world applications,
the best performing and most prevalent Gradient Boosting variants are
XGBoost (Chen & Guestrin, [2016](#bib.bib7)),
CatBoost (Dorogush et al., [2017](#bib.bib11); Prokhorenkova et al., [2018](#bib.bib32)), and
LightGBM (Ke et al., [2017](#bib.bib23)).

Recently, research on extending Deep Learning
methods to tabular data has been intensified.
Some approaches to tabular data
are only remotely related to Deep Learning.
AutoGluon-Tabular stacks small neural networks for tabular
data (Erickson et al., [2020](#bib.bib13)).
Neural Oblivious Decision Ensembles (NODE) generalizes
ensembles of oblivious decision trees by hierarchical representation
learning (Popov et al., [2019](#bib.bib31)).
NODE is a hybrid of
differentiable decision trees and neural networks.
DNF-Net builds neural structures corresponding to
logical Boolean formulas in disjunctive normal forms,
which enable localized decisions using small subsets of the features (Abutbul et al., [2020](#bib.bib1)).

However, most research focused on adapting established Deep Learning techniques to
tabular data.
Modifications to deep neural networks
like introducing leaky gates or skip connections
can improve their performance on tabular data (Fiedler, [2021](#bib.bib15)).
Even plain MLPs that are well-regularized work well on tabular data (Kadra et al., [2021](#bib.bib22)).
Different regularization coefficients to each weight improve
the performance of Deep Learning architectures on tabular data (Shavitt & Segal, [2018](#bib.bib37)).
TabularNet consists of three modules (Du et al., [2021](#bib.bib12)).
First, it uses handcrafted cell-level feature extraction
with a language model for textual data.
Secondly, it uses both row and column-wise
pooling via bidirectional gated recurrent units.
Thirdly, a graph convolutional network captures
dependencies between cells of the table.

Many approaches that adapt Deep Learning methods to tabular data
use attention mechanisms from transformers (Vaswani et al., [2017](#bib.bib43)) and BERT (Devlin et al., [2019](#bib.bib10)).
The TabTransformer learns contextual embeddings of categorical
features (Huang et al., [2020](#bib.bib21)).
However, continuous features are not covered, therefore the feature-feature
interaction is limited.
The FT-Transformer maps
features to tokens that are fed into a transformer (Gorishniy et al., [2021](#bib.bib17)).
The FT-Transformer performs well on tabular data but
all considered datasets have more than 10,000 samples.
TabNet uses an attentive transformer for sequential
attention to predict masked features (Arik & Pfister, [2021](#bib.bib2)).
Therefore, TabNet does instance-wise feature selection, that is,
can select the relevant features for each input differently.
TabNet also utilizes feature masking for pre-training, which was very successful
in natural language processing when pre-training the BERT model.
Also semi-supervised learning has been proposed for tabular
data using projections of the
features and contrastive learning (Darabi et al., [2021](#bib.bib9)).
The contrastive loss is low if pairs of the same class have
high similarity.
Value Imputation and Mask Estimation (VIME) uses
self- and semi-supervised learning of deep architectures for tabular
data (Yoon et al., [2020](#bib.bib49)).
Like BERT, the network has
to predict the values of the masked feature vectors, where the
target is always masked.
The success of BERT feature masking confirms that Deep Learning techniques
must employ strong regularization to be
successful on tabular data (Kadra et al., [2021](#bib.bib22)).
A multi-head self-attentive neural network for modeling feature-feature interactions
was also used in AutoInt (Song et al., [2019](#bib.bib42)).
So far we mentioned work, where attention mechanisms extract
feature-feature and feature-target relations.
However, also inter-sample attention can be implemented, if the whole training
set is given at the input.
TabGNN uses a graph neural network for tabular data
to model inter-sample relations (Guo et al., [2021](#bib.bib19)).
However, the authors focus on large tabular datasets with more than
40,000 samples.
SAINT contains both self-attention and inter-sample attention
and embeds both categorical and continuous features
before feeding them into transformer modules (Somepalli et al., [2021](#bib.bib41)).
SAINT uses self-supervised pre-training with a contrastive loss to
minimize the difference between original and mixed samples.
Non-Parametric Transformers (NPTs) also use feature self-attention
and inter-sample attention (Kossen et al., [2021](#bib.bib26)).
The feature self-attention identifies dependencies between
features, while inter-sample attention detects relations
between samples.
As in previous approaches, BERT masking is used
during training, where
the masked feature values and the target have to be predicted.

We suggest Hopular to learn with modern Hopfield networks from tabular data.
Hopular is a Deep Learning architecture, where each layer is
equipped with continuous modern Hopfield networks
(Ramsauer et al., [2021](#bib.bib34); Widrich et al., [2020](#bib.bib47)).
Continuous modern Hopfield networks can store two types of data:
(i) the whole training set or
(ii) the feature embedding vectors of the original input.
Like SAINT and NPT, Hopular can detect feature-feature, feature-target,
sample-sample, and sample-target dependencies via modern Hopfield networks.
Hopular’s novelty is that every layer can directly access
the original input as well as the whole training set
via stored data in the Hopfield networks.
In each layer, the stored training set enables
similarity-, prototype-, or quantization-based learning methods like
nearest neighbor.
In each layer, the stored original input enables
the identification of dependencies between the features and the target.
Consequently, the current model and its prediction
can be step-wise improved at every
layer via direct access to both the training set and the original input.
Therefore, a pass through a Hopular model is similar to standard
learning algorithms, which iteratively improve the current model and its prediction
by re-accessing the training set. The number of iterations
is fixed by the number of layers in the Hopular architecture.
As previous methods, Hopular uses a feature embedding and
BERT masking, where masked features have to be predicted.
Hopular is most closely related to SAINT (Somepalli et al., [2021](#bib.bib41))
and Non-Parametric Transformers (NPTs) (Kossen et al., [2021](#bib.bib26)),
but in contrast to SAINT and NPTs, the whole training set and the original input
are provided via Hopfield networks at every layer and
not only at the input.

Recently, it was reported that Random Forests
still outperform standard Deep Learning techniques on tabular datasets with up to 10,000
samples (Xu et al., [2021](#bib.bib48)).
In (Shwartz-Ziv & Armon, [2021](#bib.bib39)), the authors show that XGBoost
outperforms various Deep Learning methods that are designed for tabular data on
datasets that did not appear in the original papers.
Therefore, we test Hopular on exactly those datasets to see whether
it performs as well as XGBoost.
Furthermore, we test Hopular on
UCI datasets (Ramsauer et al., [2021](#bib.bib34); Klambauer et al., [2017](#bib.bib24); Wainberg et al., [2016](#bib.bib44); Fernández-Delgado et al., [2014](#bib.bib14)).
Hopular surpasses Gradient Boosting, Random Forests,
and SVMs but also state-of-the-art Deep Learning approaches
to tabular data like NPTs.

## 2 Brief Review of Modern Hopfield Networks

We briefly review
continuous modern Hopfield networks.
Their main properties are that they retrieve
stored patterns with only one update
and that they have exponential storage capacity
(Ramsauer et al., [2021](#bib.bib34)).

We assume a set of patterns {𝒙1,…,𝒙N}⊂ℝdsubscript𝒙1…subscript𝒙𝑁superscriptℝ𝑑\{\bm{x}\_{1},\ldots,\bm{x}\_{N}\}\subset\mathbb{R}^{d}
that are stacked as columns to
the matrix 𝑿=(𝒙1,…,𝒙N)𝑿subscript𝒙1…subscript𝒙𝑁\bm{X}=\left(\bm{x}\_{1},\ldots,\bm{x}\_{N}\right) and a
state pattern (query) 𝝃∈ℝd𝝃superscriptℝ𝑑\bm{\xi}\in\mathbb{R}^{d} that represents the current state.
The largest norm of a stored pattern is
M=maxi⁡‖𝒙i‖𝑀subscript𝑖normsubscript𝒙𝑖M=\max\_{i}{{\left\|\bm{x}\_{i}\right\|}}.
Continuous modern Hopfield networks with state 𝝃𝝃\bm{\xi}
have the energy

|  |  |  |  |
| --- | --- | --- | --- |
|  | E=−β−1​log⁡(∑i=1Nexp⁡(β​𝒙iT​𝝃))+β−1​log⁡N+12​𝝃T​𝝃+12​M2.Esuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽superscriptsubscript𝒙𝑖𝑇𝝃superscript𝛽1𝑁12superscript𝝃𝑇𝝃12superscript𝑀2\mathrm{E}\ =\ -\ \beta^{-1}\ \log\left(\sum\_{i=1}^{N}\exp(\beta\bm{x}\_{i}^{T}\bm{\xi})\right)+\ \beta^{-1}\log N\ +\ \frac{1}{2}\ \bm{\xi}^{T}\bm{\xi}\ +\ \frac{1}{2}\ M^{2}\ . |  | (1) |

For energy EE\mathrm{E} and state 𝝃𝝃\bm{\xi}, the update rule

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝝃new=f​(𝝃;𝑿,β)=𝑿​𝒑=𝑿​softmax​(β​𝑿T​𝝃)superscript𝝃new𝑓  𝝃𝑿𝛽𝑿𝒑𝑿softmax𝛽superscript𝑿𝑇𝝃\bm{\xi}^{\mathrm{new}}\ =\ f(\bm{\xi};\bm{X},\beta)=\ \bm{X}\ \bm{p}=\ \bm{X}\ \mathrm{softmax}(\beta\bm{X}^{T}\bm{\xi}) |  | (2) |

has been proven to converge globally
to stationary points of the energy EE\mathrm{E}, which are almost always local minima
(Ramsauer et al., [2021](#bib.bib34)).
The update rule Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data"))
is also the formula of the well-known transformer attention mechanism
(Vaswani et al., [2017](#bib.bib43); Ramsauer et al., [2021](#bib.bib34)), therefore Hopfield retrieval and
transformer attention coincide.

The separation ΔisubscriptΔ𝑖\Delta\_{i} of a
pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} is defined as its minimal dot product difference to any of the other
patterns:
Δi=minj,j≠i⁡(𝒙iT​𝒙i−𝒙iT​𝒙j)subscriptΔ𝑖subscript

𝑗𝑗
𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑖superscriptsubscript𝒙𝑖𝑇subscript𝒙𝑗\Delta\_{i}=\min\_{j,j\not=i}\left(\bm{x}\_{i}^{T}\bm{x}\_{i}-\bm{x}\_{i}^{T}\bm{x}\_{j}\right).
A pattern is well-separated from the data if Δi≥2/β​N+1/β​log⁡(2​(N−1)​N​β​M2)subscriptΔ𝑖2𝛽𝑁1𝛽2𝑁1𝑁𝛽superscript𝑀2\Delta\_{i}\geq\nicefrac{{2}}{{\beta N}}+\nicefrac{{1}}{{\beta}}\log\left(2(N-1)N\beta M^{2}\right).
If the patterns 𝒙isubscript𝒙𝑖\bm{x}\_{i} are well separated, the iterate Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data"))
converges to a fixed point close to a stored pattern.
If some patterns are similar to one another and, therefore, not well separated,
the update rule Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")) converges to
a fixed point close to the mean of the similar patterns.
This fixed point is a metastable state of the energy function
and averages over similar patterns.

The next theorem states that the update rule Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")) typically converges after
one update if the patterns are well separated. Furthermore, it states
that the retrieval error is
exponentially small in the separation ΔisubscriptΔ𝑖\Delta\_{i} (for the proof see (Ramsauer et al., [2021](#bib.bib34))):

###### Theorem 2.1.

With query 𝛏𝛏\bm{\xi}, after one update the distance of the new point f​(𝛏)𝑓𝛏f(\bm{\xi})
to the fixed point 𝐱i∗superscriptsubscript𝐱𝑖\bm{x}\_{i}^{\*} is exponentially small in the separation ΔisubscriptΔ𝑖\Delta\_{i}.
The precise bounds using the Jacobian J=∂f​(𝛏)/∂𝛏J𝑓𝛏𝛏\mathrm{J}=\nicefrac{{\partial f(\bm{\xi})}}{{\partial\bm{\xi}}} and its value JmsuperscriptJ𝑚\mathrm{J}^{m} in the mean value
theorem are:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i∗‖≤‖Jm‖2​‖𝝃−𝒙i∗‖,norm𝑓𝝃superscriptsubscript𝒙𝑖subscriptnormsuperscriptJ𝑚2norm𝝃superscriptsubscript𝒙𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}\ \leq\ {{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ {{\left\|\bm{\xi}\ -\ \bm{x}\_{i}^{\*}\right\|}}\ , |  | (3) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖Jm‖2≤ 2​β​N​M2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M)).subscriptnormsuperscriptJ𝑚22𝛽𝑁superscript𝑀2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀{{\left\|\mathrm{J}^{m}\right\|}}\_{2}\ \leq\ 2\ \beta\ N\ M^{2}\ (N-1)\ \exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ . |  | (4) |

For given ϵitalic-ϵ\epsilon and
sufficiently large ΔisubscriptΔ𝑖\Delta\_{i}, we have ‖f​(𝛏)−𝐱i∗‖<ϵnorm𝑓𝛏superscriptsubscript𝐱𝑖italic-ϵ{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}^{\*}\right\|}}<\epsilon,
that is, retrieval with one update.
The retrieval error ‖f​(𝛏)−𝐱i‖norm𝑓𝛏subscript𝐱𝑖{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}} of pattern 𝐱isubscript𝐱𝑖\bm{x}\_{i}
is bounded by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ‖f​(𝝃)−𝒙i‖≤ 2​(N−1)​exp⁡(−β​(Δi− 2​max⁡{‖𝝃−𝒙i‖,‖𝒙i∗−𝒙i‖}​M))​M.norm𝑓𝝃subscript𝒙𝑖2𝑁1𝛽subscriptΔ𝑖2norm𝝃subscript𝒙𝑖normsuperscriptsubscript𝒙𝑖subscript𝒙𝑖𝑀𝑀{{\left\|f(\bm{\xi})\ -\ \bm{x}\_{i}\right\|}}\ \leq\ 2\ (N-1)\ \exp(-\ \beta\ (\Delta\_{i}\ -\ 2\ \max\{{{\left\|\bm{\xi}\ -\ \bm{x}\_{i}\right\|}},{{\left\|\bm{x}\_{i}^{\*}\ -\ \bm{x}\_{i}\right\|}}\}\ M))\ M\ . |  | (5) |

The main requirement to modern Hopfield networks to
be suited for tabular data is that they can store and retrieve enough patterns.
We want to store a potentially large training set in every layer
of a Deep Learning architecture.
We first define what we mean by storing and retrieving patterns
from a modern Hopfield network.

###### Definition 2.2 (Pattern Stored and Retrieved).

We assume that around every pattern 𝒙isubscript𝒙𝑖\bm{x}\_{i} a sphere SisubscriptS𝑖\mathrm{S}\_{i} is given.
We say 𝒙isubscript𝒙𝑖\bm{x}\_{i} is stored if there is a single fixed point 𝒙i∗∈Sisuperscriptsubscript𝒙𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i} to
which all points 𝝃∈Si𝝃subscriptS𝑖\bm{\xi}\in\mathrm{S}\_{i} converge,
and Si∩Sj=∅subscriptS𝑖subscriptS𝑗\mathrm{S}\_{i}\cap\mathrm{S}\_{j}=\emptyset for i≠j𝑖𝑗i\not=j.
We say 𝒙isubscript𝒙𝑖\bm{x}\_{i} is retrieved for a given ϵitalic-ϵ\epsilon if
iteration (update rule) Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")) gives
a point 𝒙~isubscript~𝒙𝑖\tilde{\bm{x}}\_{i} that is at least
ϵitalic-ϵ\epsilon-close to the single fixed point 𝒙i∗∈Sisuperscriptsubscript𝒙𝑖subscriptS𝑖\bm{x}\_{i}^{\*}\in\mathrm{S}\_{i}.
The retrieval error is ‖𝒙~i−𝒙i‖normsubscript~𝒙𝑖subscript𝒙𝑖{{\left\|\tilde{\bm{x}}\_{i}-\bm{x}\_{i}\right\|}}.

![Refer to caption](/html/2206.00664/assets/x1.png)


Figure 1: Architecture overview of Hopular. Hopular consists of three different types of layers or blocks. (I) Embedding Layer—each attribute of an original input sample is represented in an e𝑒e-dimensional space. The original input sample itself is then represented by the concatenation of all of its attribute representations. (II) Hopular Block—the input representation is then refined by L𝐿L consecutive Hopular blocks. This is achieved by applying the two Hopfield modules Hssubscript𝐻𝑠H\_{s} and Hfsubscript𝐻𝑓H\_{f} in an alternating way. (III) Summarization Layer—lastly, this refined current prediction is summarized by an attribute-wise mapping, leading to the final prediction.

As with classical Hopfield networks, we consider patterns on the sphere,
i.e. patterns with a fixed norm.
For randomly chosen patterns, the number of patterns that can be stored
is exponential in the dimension d𝑑d of the space of the patterns
(for the proof see (Ramsauer et al., [2021](#bib.bib34))):

###### Theorem 2.3.

We assume a failure probability 0<p≤10𝑝10<p\leq 1 and randomly chosen patterns
on the sphere with radius M:=K​d−1assign𝑀𝐾𝑑1M:=K\sqrt{d-1}.
We define a:=2/d−1​(1+ln⁡(2​β​K2​p​(d−1)))assign𝑎2𝑑112𝛽superscript𝐾2𝑝𝑑1a:=\nicefrac{{2}}{{d-1}}(1+\ln(2\beta K^{2}p(d-1))),
b:=2​K2​β/5assign𝑏2superscript𝐾2𝛽5b:=\nicefrac{{2K^{2}\beta}}{{5}},
and c:=b/W0(exp(a+ln(b))c:=\nicefrac{{b}}{{W\_{0}(\exp(a+\ln(b))}},
where W0subscript𝑊0W\_{0} is the upper branch of the Lambert W𝑊W function (Olver et al., [2010](#bib.bib30), [(4.13)](http://dlmf.nist.gov/4.13)),
and ensure c≥(2/p)4/d−1𝑐superscript2𝑝4𝑑1c\geq\left(\nicefrac{{2}}{{\sqrt{p}}}\right)^{\nicefrac{{4}}{{d-1}}}.
Then with probability 1−p1𝑝1-p, the number of random patterns
that can be stored is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | N𝑁\displaystyle N\ | ≥p​cd−14.absent𝑝superscript𝑐𝑑14\displaystyle\geq\ \sqrt{p}\ c^{\frac{d-1}{4}}\ . |  | (6) |

Therefore it is proven for c≥3.1546𝑐3.1546c\geq 3.1546 with
β=1𝛽1\beta=1, K=3𝐾3K=3, d=20𝑑20d=20 and p=0.001𝑝0.001p=0.001 (a+ln⁡(b)>1.27𝑎𝑏1.27a+\ln(b)>1.27)
and proven for c≥1.3718𝑐1.3718c\geq 1.3718 with β=1𝛽1\beta=1, K=1𝐾1K=1, d=75𝑑75d=75, and p=0.001𝑝0.001p=0.001
(a+ln⁡(b)<−0.94𝑎𝑏0.94a+\ln(b)<{-0.94}).

This theorem motivates to use continuous modern Hopfield networks
for tabular data, where we want to store the training set in each
layer of a Deep Learning architecture.
Even for hundreds of thousands of training samples, the
continuous modern Hopfield network is able to store the training set
if the dimension of the pattern is large enough.

![Refer to caption](/html/2206.00664/assets/x2.png)


Figure 2: A Hopular Block. The first Hopfield module stores the whole training set and identifies sample-sample relations. The second Hopfield module stores the embedded input features and extracts feature-feature and feature-target relations. The Hopfield
modules refine the current prediction by combining the aggregated retrievals
of the M𝑀M Hopfield networks with their respective input.

## 3 Hopular: Modern Hopfield Networks for Tabular Data

Hopular architecture.
The Hopular architecture consists of an Embedding layer, several stacked
Hopular blocks, and a Summarization layer as depicted in Figure [1](#S2.F1 "Figure 1 ‣ 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data"). As Hopular operates on features as well as on targets, we more generally refer to them as *attributes*.

(i) The input to the Embedding Layer is an original input sample with d𝑑d attributes,
including a masked target.
Categorical attributes are encoded as one-hot vectors,
whereas continuous attributes are normalized to zero mean
and unit variance.
Then a mapping to an e𝑒e-dimensional embedding space is applied.
The index of an attribute w.r.t. the position inside
the sample as well as the attribute type are conserved
by separate e𝑒e-dimensional learnable embeddings.
All three embedding vectors are element-wise summed and
serve as the final representation of an input attribute.
The original input sample is then represented by the concatenation of all
attribute representations. This concatenation also initializes the current
prediction vector 𝝃∈ℝd⋅e𝝃superscriptℝ⋅𝑑𝑒\bm{\xi}\in\mathbb{R}^{d\cdot e} – see
Figure [A.3](#A1.F3 "Figure A.3 ‣ A.1 Architecture ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix.

(ii) The current prediction vector serves as input to a Hopular Block.
A Hopular block consecutively applies two different Hopfield modules.
Each of these Hopfield modules refines the current prediction vector by updating
the current predictions for all attributes and combining it with its
input via a residual connection.
Thus, in addition to the target, also the features of the original input sample must be predicted during training.
Figure [2](#S2.F2 "Figure 2 ‣ 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data") illustrates the forward-pass
of a single original input sample with the masked target
indicated by the question mark (?).
All current attribute predictions are refined.
The masked target is transformed by the Hopular block
to a corresponding prediction as indicated by a check mark (✓).
Also feature representations can be masked as with
BERT pre-training.

(iii) The Summarization Layer summarizes the refined current prediction vector resulting from the stacked Hopular blocks.
The current prediction vector is mapped to the final prediction vector by separately mapping each current feature prediction to the corresponding final prediction as well as mapping the current target prediction to the final target prediction – see Figure [A.4](#A1.F4 "Figure A.4 ‣ A.1 Architecture ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix. In the following we describe the components (I)–(II) of a Hopular Block.

(I) Hopfield Module Hssubscript𝐻𝑠H\_{s}.
The first Hopfield module Hssubscript𝐻𝑠H\_{s} implements a modern Hopfield network for Deep Learning architectures
similar to HopfieldLayer (Ramsauer et al., [2021](#bib.bib34), [2020](#bib.bib33))
with the training set as fixed stored patterns.
The current input 𝝃𝝃\bm{\xi} (which is also the current prediction from the previous
layer) to Hopfield module Hssubscript𝐻𝑠H\_{s}
is interacting with the whole training data
as described in Eq. ([7](#S3.E7 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
This is the update rule of
continuous modern Hopfield networks as given in Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
Hence, the Hopfield module Hssubscript𝐻𝑠H\_{s} identifies sample-sample relations
and can perform similarity searches like a nearest-neighbor search
in the whole training data.
Hssubscript𝐻𝑠H\_{s} can also average over training data that
are similar to a mapping of the current prediction vector 𝝃𝝃\bm{\xi}.

Next, we describe Hopfield Module Hssubscript𝐻𝑠H\_{s} in more detail.
Let d𝑑d be the number of attributes,
e𝑒e the embedding dimension of each single attribute,
hℎh the dimension of the Hopfield embedding space, and n𝑛n the number of samples in the training set.
The forward-pass for module Hssubscript𝐻𝑠H\_{s} with one Hopfield network and
current prediction vector 𝝃∈ℝd⋅e𝝃superscriptℝ⋅𝑑𝑒\bm{\xi}\in\mathbb{R}^{d\cdot e},
learned weight matrices 𝑾𝝃,𝑾𝑿∈ℝh×(d⋅e)

subscript𝑾𝝃subscript𝑾𝑿
superscriptℝℎ⋅𝑑𝑒\bm{W}\_{\bm{\xi}},\bm{W}\_{\bm{X}}\in\mathbb{R}^{h\times(d\cdot e)},  𝑾𝑺∈ℝ(d⋅e)×hsubscript𝑾𝑺superscriptℝ⋅𝑑𝑒ℎ\bm{W}\_{\bm{S}}\in\mathbb{R}^{(d\cdot e)\times h}, the stored training set 𝑿∈ℝ(d⋅e)×n𝑿superscriptℝ⋅𝑑𝑒𝑛\bm{X}\in\mathbb{R}^{(d\cdot e)\times n}, and a fixed scaling parameter β𝛽\beta
is given as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hs​(𝝃)=𝑾𝑺​𝑾𝑿​𝑿​softmax​(β​𝑿T​𝑾𝑿T​𝑾𝝃​𝝃).subscript𝐻𝑠𝝃subscript𝑾𝑺subscript𝑾𝑿𝑿softmax𝛽superscript𝑿𝑇superscriptsubscript𝑾𝑿𝑇subscript𝑾𝝃𝝃H\_{s}\left(\bm{\xi}\right)\ =\ \bm{W}\_{\bm{S}}\ \bm{W}\_{\bm{X}}\ \bm{X}\mathrm{softmax}(\beta\ \bm{X}^{T}\ \bm{W}\_{\bm{X}}^{T}\ \bm{W}\_{\bm{\xi}}\ \bm{\xi})\ . |  | (7) |

The hyperparameter β𝛽\beta allows to steer the type of fixed point
the update rule Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")) converges to,
hence it may further amplify the nearest-neighbor-lookup of the sample-sample Hopfield module Hssubscript𝐻𝑠H\_{s}. Hssubscript𝐻𝑠H\_{s} may contain more than one
continuous modern Hopfield network.
In this case, the respective results are combined and projected,
serving as the modules final output.
We have M𝑀M separate Hopfield networks Hsisuperscriptsubscript𝐻𝑠𝑖H\_{s}^{i}, where the module output is defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Hs​(𝝃)subscript𝐻𝑠𝝃\displaystyle H\_{s}\left(\bm{\xi}\right)\ | =𝑾G​(Hs1​(𝝃)T,…,HsM​(𝝃)T)T,absentsubscript𝑾𝐺superscriptsuperscriptsubscript𝐻𝑠1superscript𝝃𝑇…superscriptsubscript𝐻𝑠𝑀superscript𝝃𝑇𝑇\displaystyle=\ \bm{W}\_{G}\ \left(H\_{s}^{1}\left(\bm{\xi}\right)^{T},\ldots,\ H\_{s}^{M}\left(\bm{\xi}\right)^{T}\right)^{T}\ , |  | (8) |

with vector (Hs1​(𝝃)T,…,HsM​(𝝃)T)Tsuperscriptsuperscriptsubscript𝐻𝑠1superscript𝝃𝑇…superscriptsubscript𝐻𝑠𝑀superscript𝝃𝑇𝑇\left(H\_{s}^{1}\left(\bm{\xi}\right)^{T},\ldots,\ H\_{s}^{M}\left(\bm{\xi}\right)^{T}\right)^{T}
and a learnable weight matrix 𝑾G∈ℝ(d⋅e)×(M⋅d⋅e)subscript𝑾𝐺superscriptℝ⋅𝑑𝑒⋅𝑀𝑑𝑒\bm{W}\_{G}\in\mathbb{R}^{(d\cdot{}e)\times{}(M\cdot{}d\cdot{}e)}.

(II) Hopfield Module Hfsubscript𝐻𝑓H\_{f}.
The second Hopfield module Hfsubscript𝐻𝑓H\_{f} implements a modern Hopfield network for Deep Learning architectures
via the layer Hopfield (Ramsauer et al., [2021](#bib.bib34), [2020](#bib.bib33))
with the embedded features of the original input sample as stored patterns.
The refined prediction vector from the previous layer
is reshaped and transposed
to the matrix 𝚵𝚵\bm{\Xi}, which serves as input to
the Hopfield module Hfsubscript𝐻𝑓H\_{f}.
𝚵𝚵\bm{\Xi} interacts with the embedded features
of the original input sample
as described in Eq. ([9](#S3.E9 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
Again, this is the update rule of
continuous modern Hopfield networks as given in Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
Therefore, the Hopfield module Hfsubscript𝐻𝑓H\_{f} extracts and models
feature-feature and feature-target relations.
Current feature and target predictions are adjusted and refined after they are associated with the original input sample feature representations.

Next, we describe Hopfield Module Hfsubscript𝐻𝑓H\_{f} in more detail.
The matrix 𝚵∈ℝe×d𝚵superscriptℝ𝑒𝑑\bm{\Xi}\in\mathbb{R}^{e\times d} is a transposed and reshaped version
of current prediction vector 𝝃𝝃\bm{\xi} with respect to the embedding dimension e𝑒e.
Using the learned weight matrices 𝑾𝚵,𝑾𝒀∈ℝh×e

subscript𝑾𝚵subscript𝑾𝒀
superscriptℝℎ𝑒\bm{W}\_{\bm{\Xi}},\bm{W}\_{\bm{Y}}\in\mathbb{R}^{h\times e},  𝑾𝑭∈ℝe×hsubscript𝑾𝑭superscriptℝ𝑒ℎ\bm{W}\_{\bm{F}}\in\mathbb{R}^{e\times h},
the embedded original input sample 𝒀∈ℝe×d𝒀superscriptℝ𝑒𝑑\bm{Y}\in\mathbb{R}^{e\times d}, and a fixed scaling parameter β𝛽\beta
the forward-pass is

|  |  |  |  |
| --- | --- | --- | --- |
|  | Hf​(𝚵)=𝑾𝑭​𝑾𝒀​𝒀​softmax​(β​𝒀T​𝑾𝒀T​𝑾𝚵​𝚵).subscript𝐻𝑓𝚵subscript𝑾𝑭subscript𝑾𝒀𝒀softmax𝛽superscript𝒀𝑇superscriptsubscript𝑾𝒀𝑇subscript𝑾𝚵𝚵H\_{f}\left(\bm{\Xi}\right)\ =\ \bm{W}\_{\bm{F}}\ \bm{W}\_{\bm{Y}}\ \bm{Y}\mathrm{softmax}\left(\beta\ \bm{Y}^{T}\ \bm{W}\_{\bm{Y}}^{T}\ \bm{W}\_{\bm{\Xi}}\ \bm{\Xi}\right). |  | (9) |

Hfsubscript𝐻𝑓H\_{f} may contain more than one continuous modern Hopfield network,
which leads to an analog equation as Eq. ([8](#S3.E8 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data")) for Hssubscript𝐻𝑠H\_{s}.

Hopular architecture and Modern Hopfield Networks.
Deep Learning could not convince so far on small tabular datasets,
on the other hand iterative learning algorithms,
like Gradient Boosting methods, are the best-performing methods in this domain.
Therefore, we introduce a DL architecture that is able to mimic and
extend these iterative algorithms by reaccessing the whole training set and
refining the current prediction in each layer.
Modern Hopfield Networks directly access an external memory
in a content-based fashion as depicted in Eq. ([2](#S2.E2 "In 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
Hopular populates this external memory in two different ways: (a) Hopular uses the training set as an external memory, and (b) Hopular uses the embedded feature representations of the original input sample as external memory.
During training, retrieval from the respective memory is learned whereas
the type of fixed point of the modern Hopfield network,
as described in Section [2](#S2 "2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data"), specifies the type of retrieved pattern.
Additionally, modern Hopfield networks
can retrieve patterns with only one update – see Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1. ‣ 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data").

Furthermore, their exponential storage capacity (Theorem [2.3](#S2.Thmtheorem3 "Theorem 2.3. ‣ 2 Brief Review of Modern Hopfield Networks ‣ Hopular: Modern Hopfield Networks for Tabular Data")) makes it possible to
retrieve patterns from external memories with even hundreds of thousands
instances.
Because of these properties Hopular can mimic iterative learning algorithms e.g. such based on gradient descent, boosting, or feature selection that refine the current prediction by re-accessing the training set in contrast to other Deep Learning methods for tabular data. Both NPTs and SAINT consider feature-feature and sample-sample interactions via their respective attention mechanisms which solely use the result of the previous layer. In contrast, Hopular not only uses the result of the previous layer but also the original input sample and the whole training set.
For example, our method can implement gradient boosting with a boosting step at each layer.
The ability to mimic iterative learning algorithms that are known to perform specifically well on tabular data makes modern Hopfield networks a promising
approach for processing tabular data.
For the instantiation variant that we use for our experiments
the Hopfield module Hssubscript𝐻𝑠H\_{s} identifies sample-sample relations
and can perform similarity searches like a nearest-neighbor search
in the whole training data.
In the Appendix in Section [A.6](#A1.SS6 "A.6 Hopular Intuition: Mimicking Iterative Learning ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data")
we give further intuition of how Hopular
can mimic iterative learning algorithms on the basis of two examples.

Hopular’s Objective and Training Method.
Hopular’s objective is a weighted sum of
the self-supervised loss for predicting masked features and
the standard supervised target loss.
In the following we explain the feature masking as well as the objective in more detail.

Feature Masking.
We follow state-of-the-art Deep Learning methods
like SAINT (Somepalli et al., [2021](#bib.bib41))
and Non-Parametric Transformers (NPTs) (Kossen et al., [2021](#bib.bib26))
that are tailored to tabular data and
use BERT masking (Devlin et al., [2019](#bib.bib10)) of the input features.
Masked input features must be predicted during training.
Feature masking is an especially beneficial self-supervised approach when
handling small datasets as it exerts a
strong regularizing effect on the training procedure.
The amount of masked features during training is
determined by the masking probability, which is a hyperparameter of the model.
In Hopular, both features and targets can be masked during training,
while for inference only the target is masked.

Objective.
Hopular’s objective is a weighted sum of the masked
feature loss LfsubscriptL𝑓\mathrm{L}\_{f}
and the supervised target loss LtsubscriptL𝑡\mathrm{L}\_{t}.
The overall loss LL\mathrm{L} is

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | LL\displaystyle\mathrm{L}\ | =γ​Lf+(1−γ)​Lt,absent𝛾subscriptL𝑓1𝛾subscriptL𝑡\displaystyle=\ \gamma\ \mathrm{L}\_{f}\ +\ (1\ -\ \gamma)\mathrm{L}\_{t}\ , |  | (10) |

where LtsubscriptL𝑡\mathrm{L}\_{t} and LfsubscriptL𝑓\mathrm{L}\_{f} are the negative logloss
in case of discrete attributes and the mean squared error
in case of continuous attributes with γ𝛾\gamma as a hyperparameter.
In our default hyperparameter setting γ𝛾\gamma is
annealed using a cosine scheduler starting at 111 with a final value of 00.
Another essential hyperparameter for Hopular is β𝛽\beta
in Eq. ([7](#S3.E7 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data")) and Eq. ([9](#S3.E9 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data")).
A small β𝛽\beta retrieves a pattern close to the mean of
the stored patterns, while a large β𝛽\beta
retrieves the stored pattern
that is closest to the initial state pattern (Ramsauer et al., [2021](#bib.bib34)).
For module Hssubscript𝐻𝑠H\_{s} a large β𝛽\beta value
emphasizes a nearest-neighbor lookup mechanics.
For module Hfsubscript𝐻𝑓H\_{f} a large β𝛽\beta value leads to less
diluted features.
Thus, large β𝛽\beta values seem to be beneficial for Hopular.
Experiments confirm this assumption
(see Section [4](#S4 "4 Experiments ‣ Hopular: Modern Hopfield Networks for Tabular Data")).

Hopular Pseudocode. Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data") shows the forward pass of Hopular for
an original input sample 𝒙𝒙\bm{x}.

Algorithm 1  Forward pass of Hopular

1:Hopfield modules Hssubscript𝐻𝑠H\_{s} and Hfsubscript𝐻𝑓H\_{f}, embedding layer E𝐸E,
summarization layer S𝑆S, number of features d𝑑d,
number of Hopular blocks L𝐿L
and original input sample 𝒙∈ℝd𝒙superscriptℝ𝑑\bm{x}\in\mathbb{R}^{d}

2:𝒙←Mask​(𝒙)←𝒙Mask𝒙\bm{x}\leftarrow\text{Mask}(\bm{x})

3:𝝃←E​(𝒙)←𝝃𝐸𝒙\bm{\xi}\leftarrow E(\bm{x})

4:for i=1𝑖1i=1 to L𝐿L do

5:     𝝃←𝝃+Hs​(𝝃)←𝝃𝝃subscript𝐻𝑠𝝃\bm{\xi}\leftarrow\bm{\xi}+H\_{s}(\bm{\xi})

6:     𝚵←Reshape​(𝝃T)←𝚵Reshapesuperscript𝝃𝑇\bm{\Xi}\leftarrow\text{Reshape}(\bm{\xi}^{T})

7:     𝚵←𝚵+Hf​(𝚵)←𝚵𝚵subscript𝐻𝑓𝚵\bm{\Xi}\leftarrow\bm{\Xi}+H\_{f}(\bm{\Xi})

8:     𝝃←Reshape​(𝚵)T←𝝃Reshapesuperscript𝚵𝑇\bm{\xi}\leftarrow\text{Reshape}(\bm{\Xi})^{T}

9:end for

10:𝝃←S​(𝝃)←𝝃𝑆𝝃\bm{\xi}\leftarrow S(\bm{\xi})

## 4 Experiments

Since Deep Learning methods have already been successfully applied to larger tabular datasets (Avati et al., [2018](#bib.bib3); Simm et al., [2018](#bib.bib40); Zhang et al., [2019b](#bib.bib52); Mayr et al., [2018](#bib.bib28))
we want to know whether Hopular
is competitive on small tabular datasets.
In particular, we compare Hopular to XGBoost, CatBoost, LightGBM, and NPTs (Kossen et al., [2021](#bib.bib26)).
Gradient Boosting has the lead on tabular data when excluding Deep Learning methods.
NPTs represent state-of-the-art Deep Learning methods
for tabular data,
as NPTs yielded very good results on small tabular datasets.

### 4.1 Small-Sized Tabular Datasets

In these experiments, we compare Hopular to other Deep Learning methods, XGBoost, CatBoost, and LightGBM on small-sized tabular datasets.

Methods Compared.
We compare Hopular, XGBoost, CatBoost, LightGBM, NPTs, and other 24 machine learning
methods as described in (Klambauer et al., [2017](#bib.bib24)).
The compared methods include 10 Deep Learning (DL) approaches.
Following (Klambauer et al., [2017](#bib.bib24); Wainberg et al., [2016](#bib.bib44)),
17 methods are selected from their respective method group
as the model with the median performance over all datasets within each method group.
NPTs are used in a non-transductive setting for a fair comparison.

Hyperparameter Selection.
All hyperparameters are selected on seperate validation sets. For NPTs we perform hyperparameter search as in Table [A.5](#A1.T5 "Table A.5 ‣ A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data"). This includes the hyperparameters that have already been successfully used in (Kossen et al., [2021](#bib.bib26)) on small- and medium-sized tabular datasets.
This selection also serves as a constraint on the computational resources invested for Hopular.
For XGBoost, CatBoost, and LightGBM, we apply the same Bayesian hyperparameter optimization
procedure as described in (Shwartz-Ziv & Armon, [2021](#bib.bib39)). For LightGBM we use the default hyperparameter ranges as specified by hyperopt-sklearn (Komer et al., [2014](#bib.bib25)).
Section [A.3](#A1.SS3 "A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix describes the hyperparameter selection in more detail.

Datasets.
Following (Klambauer et al., [2017](#bib.bib24)),
we consider UCI machine learning repository datasets
with less than or equal to 1,000 samples as being small.
We select 21 of these datasets and give an overview in Table [A.3](#A1.T3 "Table A.3 ‣ A.2.2 Small-Sized Dataset Description ‣ A.2 Datasets ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data").
The datasets themselves as well as the train/test splits are taken from (Fernández-Delgado et al., [2014](#bib.bib14)).
A detailed explanation of the dataset selection process as well as a description of the datasets can be found in Section [A.2](#A1.SS2 "A.2 Datasets ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix.

Table 1: Median rank of compared methods across
the datasets of the UCI machine learning repository.
Methods are ranked for each dataset according to the accuracy on the respective test set.
Hopular achieves the lowest median rank of 7.57.57.5, therefore is the best
performing method across the considered UCI datasets. The complete list can be seen in Table [A.7](#A1.T7 "Table A.7 ‣ A.4 Results ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix.

|  |  |  |  |
| --- | --- | --- | --- |
| Method | Rank | Method | Rank |
| Hopular (DL) | 7.57.57.5 | CatBoost | 14.014.014.0 |
| ⋮⋮\vdots{} | ⋮⋮\vdots{} | LightGBM | 14.514.514.5 |
| ⋮⋮\vdots{} | ⋮⋮\vdots{} |
| Non-Parametric Transformers (DL) | 11.011.011.0 |
| XGBoost | 12.012.012.0 | Stacking (Wolpert) | 28.028.028.0 |

Results. Table [1](#S4.T1 "Table 1 ‣ 4.1 Small-Sized Tabular Datasets ‣ 4 Experiments ‣ Hopular: Modern Hopfield Networks for Tabular Data") shows the median rank of all compared methods across
the datasets of the UCI machine learning repository
(see Table [A.7](#A1.T7 "Table A.7 ‣ A.4 Results ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix for the complete list).
Methods are ranked for each dataset according to the accuracy on the respective test set.
17 method groups have been compared previously (Wainberg et al., [2016](#bib.bib44)), to which
we add XGBoost (Chen & Guestrin, [2016](#bib.bib7)), CatBoost (Dorogush et al., [2017](#bib.bib11); Prokhorenkova et al., [2018](#bib.bib32)), LightGBM (Ke et al., [2017](#bib.bib23)),
NPTs (Kossen et al., [2021](#bib.bib26)), Self-Normalizing Networks (Klambauer et al., [2017](#bib.bib24)), and our Hopular.
Deep Learning methods are indicated by “(DL)” and are not grouped.
Hopular has a median rank of 7.57.57.5, followed by Support Vector Machines with 9.59.59.5,
while NPTs, XGBoost, CatBoost, and LightGBM
have a median rank of 111111, 121212, 141414, and 14.514.514.5 respectively.
Hopular with modern Hopfield networks as memory performs better than
other Deep Learning methods
and in particular better than the closely-related NPTs.
Across the considered UCI datasets,
Hopular is the best performing method.

### 4.2 Medium-Sized Tabular Datasets

In these experiments, we compare Hopular to other Deep Learning methods,
XGBoost, CatBoost, and LightGBM on medium-sized tabular datasets.
In (Shwartz-Ziv & Armon, [2021](#bib.bib39)), the authors show that XGBoost outperforms various
Deep Learning methods that are designed for tabular data on
datasets that did not appear in the original papers.
We want to know whether XGBoost still has the lead on
these medium-sized datasets.

Methods Compared.
We compare Hopular, NPTs, XGBoost, CatBoost, and LightGBM.
NPTs are used in a non-transductive setting for a fair comparison.

Hyperparameter Selection.
All hyperparameters are selected on seperate validation sets. For NPTs we perform hyperparameter search as in Table [A.5](#A1.T5 "Table A.5 ‣ A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data"). This includes the hyperparameters that have already been successfully used in (Kossen et al., [2021](#bib.bib26)) on small- and medium-sized tabular datasets.
This selection also serves as a constraint on the computational resources invested for Hopular.
For XGBoost, CatBoost, and LightGBM, we apply the same Bayesian hyperparameter optimization
procedure as described in (Shwartz-Ziv & Armon, [2021](#bib.bib39)). For LightGBM we use the default hyperparameter ranges as specified by hyperopt-sklearn (Komer et al., [2014](#bib.bib25)).
Section [A.3](#A1.SS3 "A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix describes the hyperparameter selection in more detail.

Datasets.
We select the datasets and dataset splits of (Shwartz-Ziv & Armon, [2021](#bib.bib39)),
where XGBoost performs better than Deep Learning methods that have been
designed for tabular data.
We extend this selection by two datasets for regression: (a) colleges was already
used for other Deep Learning methods for tabular data (Somepalli et al., [2021](#bib.bib41)), and
(b) sulfur is publicly available and fits with its 10,082 instances well into
the existing collection of medium-sized datasets.
Table [A.4](#A1.T4 "Table A.4 ‣ A.2.3 Medium-Sized Dataset Description ‣ A.2 Datasets ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") gives an overview of the medium-sized datasets.
A detailed description of the datasets can be found in Section [A.2](#A1.SS2 "A.2 Datasets ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") of the Appendix.

Table 2: Results of all compared methods on the subset of medium-sized tabular datasets (Shwartz-Ziv & Armon, [2021](#bib.bib39)). For classification tasks (C), the accuracy is reported. For regression tasks (R), the mean squared error multiplied by a factor of 100010001000 is reported. The reported deviations are the corresponding standard error of the mean. All values are computed on the respective test sets, averaged over three replicates.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | Hopular | NPTs | XGBoost | CatBoost | LightGBM |
| sulfur (R) | 1.04±0.02uncertain1.040.021.04\pm 0.02 | 1.24±0.02uncertain1.240.021.24\pm 0.02 | 1.23±0.00uncertain1.230.001.23\pm 0.00 | 1.06±0.01uncertain1.060.011.06\pm 0.01 | 1.16±0.01uncertain1.160.011.16\pm 0.01 |
| colleges (R) | 21.18±0.09uncertain21.180.0921.18\pm 0.09 | 25.67±0.23uncertain25.670.2325.67\pm 0.23 | 30.47±0.00uncertain30.470.0030.47\pm 0.00 | 26.40±0.09uncertain26.400.0926.40\pm 0.09 | 25.64±0.09uncertain25.640.0925.64\pm 0.09 |
| eye (C) | 53.56±0.48uncertain53.560.4853.56\pm 0.48 | 53.21±0.12uncertain53.210.1253.21\pm 0.12 | 57.43±0.00uncertain57.430.0057.43\pm 0.00 | 56.35±0.05uncertain56.350.0556.35\pm 0.05 | 57.34±0.28uncertain57.340.2857.34\pm 0.28 |
| gesture (C) | 71.20±0.19uncertain71.200.1971.20\pm 0.19 | 67.83±0.06uncertain67.830.0667.83\pm 0.06 | 68.05±0.00uncertain68.050.0068.05\pm 0.00 | 68.86±0.21uncertain68.860.2168.86\pm 0.21 | 69.01±0.09uncertain69.010.0969.01\pm 0.09 |
| blastchar (C) | 80.05±0.11uncertain80.050.1180.05\pm 0.11 | 79.98±0.11uncertain79.980.1179.98\pm 0.11 | 76.78±0.00uncertain76.780.0076.78\pm 0.00 | 80.13±0.12uncertain80.130.1280.13\pm 0.12 | 79.92±0.21uncertain79.920.2179.92\pm 0.21 |
| shrutime (C) | 86.12±0.09uncertain86.120.0986.12\pm 0.09 | 85.62±0.07uncertain85.620.0785.62\pm 0.07 | 84.58±0.00uncertain84.580.0084.58\pm 0.00 | 86.39±0.04uncertain86.390.0486.39\pm 0.04 | 86.18±0.02uncertain86.180.0286.18\pm 0.02 |

Results. Table [2](#S4.T2 "Table 2 ‣ 4.2 Medium-Sized Tabular Datasets ‣ 4 Experiments ‣ Hopular: Modern Hopfield Networks for Tabular Data") reports the results
of Hopular, NPTs, XGBoost, CatBoost, and LightGBM on the medium-sized datasets.
The evaluation procedure is from (Shwartz-Ziv & Armon, [2021](#bib.bib39)).
Hopular is the best performing method on 3 out of the 6 datasets.
The runner-up method, CatBoost, is twice the best method, whereas XGBoost once.
The biggest performance difference is achieved by Hopular on the two regression datasets,
where the capabilities of an external memory really shine.
Directly deriving the underlying function for regression datasets may be a difficult task,
especially in absence of abundant data.
Hopular is able to mitigate this shortcoming
by incorporating local neighbourhood information and
iteratively refining its current prediction by memory lookups.
Over the 6 datasets, NPTs and XGBoost have a median rank of 4.5,
CatBoost and LightGBM of 2.5 and 2, respectively,
and Hopular has a median rank of 1.5.
On average over all 6 datasets, Hopular performs better than
NPTs, XGBoost, CatBoost, and LightGBM.
We also found that our method needs only a fraction of the memory compared to NPTs which can be seen in Table [A.8](#A1.T8 "Table A.8 ‣ A.5 Memory footprint and runtime estimates ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data"). We also added runtime estimates in Table [A.9](#A1.T9 "Table A.9 ‣ A.5 Memory footprint and runtime estimates ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data").

## 5 Conclusion

Hopular is a novel Deep Learning architecture where every layer is equipped
with an external memory. This enables Hopular to mimic standard iterative learning
algorithms that refine the current prediction by re-accessing the training set.
We validated the usefulness of this property both on small- and
medium-sized tabular datasets. Hopular is the best performing method
across a broad selection of specifically challenging small-sized UCI
datasets. Additionally, Hopular is the best-performing method on
medium-sized tabular datasets among which CatBoost and LightGBM achieved very competitive
results.
This makes Hopular a strong contender to current state-of-the-art methods like Gradient Boosting and other Deep Learning methods specialized in small- and medium-sized datasets.

## Acknowledgments

The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria. IARAI is supported by Here Technologies. We thank the projects AI-MOTION (LIT-2018-6-YOU-212), AI-SNN (LIT-2018-6-YOU-214), DeepFlood (LIT-2019-8-YOU-213), Medical Cognitive Computing Center (MC3), INCONTROL-RL (FFG-881064), PRIMAL (FFG-873979), S3AI (FFG-872172), DL for GranularFlow (FFG-871302), AIRI FG 9-N (FWF-36284, FWF-36235), ELISE (H2020-ICT-2019-3 ID: 951847). We thank Audi.JKU Deep Learning Center, TGW LOGISTICS GROUP GMBH, Silicon Austria Labs (SAL), FILL Gesellschaft mbH, Anyline GmbH, Google, ZF Friedrichshafen AG, Robert Bosch GmbH, UCB Biopharma SRL, Merck Healthcare KGaA, Verbund AG, Software Competence Center Hagenberg GmbH, TÜV Austria, Frauscher Sensonic and the NVIDIA Corporation.

## References

* Abutbul et al. (2020)

  Abutbul, A., Elidan, G., Katzir, L., and El-Yaniv, R.
  DNF-Net: A neural architecture for tabular data.
  *ArXiv*, 2006.06465, 2020.
  URL <https://openreview.net/forum?id=73WTGs96kho>.
  9th International Conference on Learning Representations (ICLR).
* Arik & Pfister (2021)

  Arik, S. Ö. and Pfister, T.
  TabNet: Attentive interpretable tabular learning.
  *Proceedings of the AAAI Conference on Artificial Intelligence*,
  35(8):6679–6687, 2021.
* Avati et al. (2018)

  Avati, A., Jung, K., Harman, S., Downing, L., Ng, A., and Shah, N.
  Improving palliative care with deep learning.
  *BMC Medical Informatics and Decision Making*, 122, 2018.
  doi: 10.1186/s12911-018-0677-8.
* Benedetti (1977)

  Benedetti, J. K.
  On the nonparametric estimation of regression functions.
  *Journal of the Royal Statistical Society*, 39:248–253, 1977.
* Boser et al. (1992)

  Boser, B. E., Guyon, I. M., and Vapnik, V. N.
  A training algorithm for optimal margin classifiers.
  In *Proceedings of the 5th Annual ACM Workshop on Computational
  Learning Theory*, pp.  144–152. ACM Press, Pittsburgh, PA, 1992.
* Breiman (2001)

  Breiman, L.
  Random forests.
  *Machine Learning*, 45(1):5–32, 2001.
  doi: 10.1023/A:1010933404324.
* Chen & Guestrin (2016)

  Chen, T. and Guestrin, C.
  XGBoost: A scalable tree boosting system.
  In *Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, KDD ’16, pp.  785–794, New York,
  NY, USA, 2016. Association for Computing Machinery.
  doi: 10.1145/2939672.2939785.
* Cortes & Vapnik (1995)

  Cortes, C. and Vapnik, V.
  Support-vector networks.
  *Machine learning*, 20(3):273–297, 1995.
* Darabi et al. (2021)

  Darabi, S., Fazeli, S., Pazoki, A., Sankararaman, S., and Sarrafzadeh, M.
  Contrastive Mixup: self- and semi-supervised learning for tabular
  domain.
  *ArXiv*, 2108.12296, 2021.
* Devlin et al. (2019)

  Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.
  BERT: pre-training of deep bidirectional transformers for language
  understanding.
  In *Proceedings of the 2019 Conference of the North American
  Chapter of the Association for Computational Linguistics: Human Language
  Technologies, Volume 1 (Long and Short Papers)*, pp.  4171–4186.
  Association for Computational Linguistics, 2019.
  doi: 10.18653/v1/N19-1423.
* Dorogush et al. (2017)

  Dorogush, A. V., Gulin, A., Gusev, G., Kazeev, N., Prokhorenkova, L. O., and
  Vorobev, A.
  CatBoost: unbiased boosting with categorical features.
  *ArXiv*, 1706.09516, 2017.
* Du et al. (2021)

  Du, L., Gao, F., Chen, X., Jia, R., Wang, J., Zhang, J., Han, S., and Zhang, D.
  TabularNet: A neural network architecture for understanding
  semantic structures of tabular data.
  In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge
  Discovery & Data Mining*, KDD ’21, pp.  322–331, New York, NY, USA, 2021.
  Association for Computing Machinery.
  doi: 10.1145/3447548.3467228.
* Erickson et al. (2020)

  Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and
  Smola, A.
  AutoGluon-Tabular: Robust and accurate AutoML for structured
  data.
  *ArXiv*, 2003.06505, 2020.
* Fernández-Delgado et al. (2014)

  Fernández-Delgado, M., Cernadas, E., Barro, S., and Amorim, D.
  Do we need hundreds of classifiers to solve real world classification
  problems?
  *The Journal of Machine Learning Research*, 15(1):3133–3181, 2014.
* Fiedler (2021)

  Fiedler, J.
  Simple modifications to improve tabular neural networks.
  *ArXiv*, 2108.03214, 2021.
* Friedman (2001)

  Friedman, J. H.
  Greedy function approximation: A gradient boosting machine.
  *The Annals of Statistics*, 29(5):1189–1232, 2001.
  doi: 10.1214/aos/1013203451.
* Gorishniy et al. (2021)

  Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A.
  Revisiting deep learning models for tabular data.
  *ArXiv*, 2106.11959, 2021.
* Grill et al. (2020)

  Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P. H.,
  Buchatskaya, E., Doersch, C., Pires, B. Á., Guo, Z. D., Azar, M. G.,
  Piot, B., Kavukcuoglu, K., Munos, R., and Valko, M.
  Bootstrap your own latent - a new approach to self-supervised
  learning.
  In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin,
  H. (eds.), *Advances in Neural Information Processing Systems*,
  volume 33, pp.  21271–21284. Curran Associates, Inc., 2020.
* Guo et al. (2021)

  Guo, X., Quan, Y., Zhao, H., Yao, Q., Li, Y., and Tu, W.
  TabGNN: Multiplex graph neural network for tabular data
  prediction.
  *ArXiv*, 2108.09127, 2021.
* Ho (1995)

  Ho, T. K.
  Random decision forests.
  In *Proceedings of 3rd International Conference on Document
  Analysis and Recognition*, volume 1, pp.  278–282, 1995.
  doi: 10.1109/ICDAR.1995.598994.
* Huang et al. (2020)

  Huang, X., Khetan, A., Cvitkovic, M., and Karnin, Z.
  TabTransformer: Tabular data modeling using contextual
  embeddings.
  *ArXiv*, 2012.06678, 2020.
* Kadra et al. (2021)

  Kadra, A., Lindauer, M., Hutter, F., and Grabocka, J.
  Regularization is all you need: Simple neural nets can excel on
  tabular data.
  *ArXiv*, 2106.11189, 2021.
* Ke et al. (2017)

  Ke, G., Meng, A., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu,
  T.-Y.
  LightGBM: A highly efficient gradient boosting decision tree.
  In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
  Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural
  Information Processing Systems*, volume 30. Curran Associates, Inc., 2017.
* Klambauer et al. (2017)

  Klambauer, G., Unterthiner, T., Mayr, A., and Hochreiter, S.
  Self-normalizing neural networks.
  In *Advances in Neural Information Processing Systems*, pp. 971–980, 2017.
* Komer et al. (2014)

  Komer, B., Bergstra, J., and Eliasmith, C.
  Hyperopt-sklearn: automatic hyperparameter configuration for
  scikit-learn.
  In *ICML workshop on AutoML*, volume 9, pp.  50. Citeseer,
  2014.
* Kossen et al. (2021)

  Kossen, J., Band, N., Lyle, C., Gomez, A. N., Rainforth, T., and Gal, Y.
  Self-attention between datapoints: Going beyond individual
  input-output pairs in deep learning.
  *ArXiv*, 2106.02584, 2021.
* LeCun et al. (2015)

  LeCun, Y., Bengio, Y., and Hinton, G.
  Deep learning.
  *Nature*, 521:436–444, 2015.
* Mayr et al. (2018)

  Mayr, A., Klambauer, G., Unterthiner, T., Steijaert, M., Wegner, J., Ceulemans,
  H., Clevert, D., and Hochreiter, S.
  Large-scale comparison of machine learning methods for drug target
  prediction on chembl.
  *Chemical Science*, 9:5441–5451, 2018.
  doi: 10.1039/C8SC00148K.
* Nadaraya (1964)

  Nadaraya, E. A.
  On estimating regression.
  *Theory of Probability & Its Applications*, 9(1):141–142, 1964.
  doi: 10.1137/1109020.
* Olver et al. (2010)

  Olver, F. W. J., Lozier, D. W., Boisvert, R. F., and Clark, C. W.
  *NIST handbook of mathematical functions*.
  Cambridge University Press, 1 pap/cdr edition, 2010.
  ISBN 9780521192255.
* Popov et al. (2019)

  Popov, S., Morozov, S., and Babenko, A.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  *ArXiv*, 1909.06312, 2019.
  URL <https://openreview.net/forum?id=r1eiu2VtwH>.
  8th International Conference on Learning Representations (ICLR).
* Prokhorenkova et al. (2018)

  Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A.
  CatBoost: unbiased boosting with categorical features.
  In Bengio, S., Wallach, H., Larochelle, H., Grauman, K.,
  Cesa-Bianchi, N., and Garnett, R. (eds.), *Advances in Neural
  Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.
* Ramsauer et al. (2020)

  Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L.,
  Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D.,
  Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S.
  Hopfield networks is all you need.
  *ArXiv*, 2008.02217, 2020.
* Ramsauer et al. (2021)

  Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L.,
  Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D.,
  Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S.
  Hopfield networks is all you need.
  In *9th International Conference on Learning Representations
  (ICLR)*, 2021.
  URL <https://openreview.net/forum?id=tL89RnzIiCd>.
* Schmidhuber (2015)

  Schmidhuber, J.
  Deep learning in neural networks: An overview.
  *Neural Networks*, 61:85–117, 2015.
  doi: 10.1016/j.neunet.2014.09.003.
* Schölkopf & Smola (2002)

  Schölkopf, B. and Smola, A. J.
  *Learning with kernels - Support Vector Machines,
  Regularization, Optimization, and Beyond*.
  MIT Press, Cambridge, 2002.
* Shavitt & Segal (2018)

  Shavitt, I. and Segal, E.
  Regularization learning networks: Deep learning for tabular datasets.
  In Bengio, S., Wallach, H., Larochelle, H., Grauman, K.,
  Cesa-Bianchi, N., and Garnett, R. (eds.), *Advances in Neural
  Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.
* Shen & Li (2010)

  Shen, C. and Li, H.
  On the dual formulation of boosting algorithms.
  *IEEE transactions on pattern analysis and machine
  intelligence*, 32:2216–2231, 2010.
  doi: 10.1109/TPAMI.2010.47.
* Shwartz-Ziv & Armon (2021)

  Shwartz-Ziv, R. and Armon, A.
  Tabular Data: Deep learning is not all you need.
  *ArXiv*, 2106.03253, 2021.
  URL <https://openreview.net/forum?id=vdgtepS1pV>.
  AutoML Workshop of International Conference on Machine Learning
  (ICML).
* Simm et al. (2018)

  Simm, J., Klambauer, G., Arany, A., Steijaert, M., Wegner, J., Gustin, E.,
  Chupakhin, V., Chong, Y., Vialard, J., Bujinsters, P., Velter, I., Vapirev,
  A., Singh, S., Carpenter, A., Wuyts, R., Hochreiter, S., Moreau, Y., and
  Ceulemans, H.
  Crepurposing high-throughput image assays enables biological activity
  prediction for drug discovery.
  *Cell Chemical Biology*, 25:611–618, 2018.
  doi: 10.1016/j.chembiol.2018.01.015.
* Somepalli et al. (2021)

  Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., and Goldstein, T.
  SAINT: Improved neural networks for tabular data via row
  attention and contrastive pre-training.
  *ArXiv*, 2106.01342, 2021.
* Song et al. (2019)

  Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., and Tang, J.
  AutoInt: Automatic feature interaction learning via
  self-attentive neural networks.
  In *Proceedings of the 28th ACM International Conference on
  Information and Knowledge Management*, CIKM ’19, pp.  1161–1170, New York,
  NY, USA, 2019. Association for Computing Machinery.
  doi: 10.1145/3357384.3357925.
* Vaswani et al. (2017)

  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., and Polosukhin, I.
  Attention is all you need.
  In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,
  Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural
  Information Processing Systems 30*, pp.  5998–6008. Curran Associates,
  Inc., 2017.
* Wainberg et al. (2016)

  Wainberg, M., Alipanahi, B., and Frey, B. J.
  Are random forests truly the best classifiers?
  *The Journal of Machine Learning Research*, 17(1):3837–3841, 2016.
* Watson (1964)

  Watson, G. S.
  Smooth regression analysis.
  *Sankhya: The Indian Journal of Statistics, Series A
  (1961-2002)*, 26(4):359–372, 1964.
* Weinberger & Tesauro (2007)

  Weinberger, K. Q. and Tesauro, G.
  Metric learning for kernel regression.
  In Meila, M. and Shen, X. (eds.), *Proceedings of the Eleventh
  International Conference on Artificial Intelligence and Statistics*, volume 2
  of *Proceedings of Machine Learning Research*, pp.  612–619, San Juan,
  Puerto Rico, 2007. PMLR.
* Widrich et al. (2020)

  Widrich, M., Schäfl, B., Pavlović, M., Ramsauer, H., Gruber, L.,
  Holzleitner, M., Brandstetter, J., Sandve, G. K., Greiff, V., Hochreiter, S.,
  and Klambauer, G.
  Modern Hopfield networks and attention for immune repertoire
  classification.
  In *Advances in Neural Information Processing Systems*. Curran
  Associates, Inc., 2020.
* Xu et al. (2021)

  Xu, H., Ainsworth, M., Peng, Y.-C., Kusmanov, M., Panda, S., and Vogelstein,
  J. T.
  When are deep networks really better than random forests at small
  sample sizes?
  *ArXiv*, 2108.13637, 2021.
* Yoon et al. (2020)

  Yoon, J., Zhang, Y., Jordon, J., and vanDerSchaar, M.
  VIME: Extending the success of self- and semi-supervised learning
  to tabular domain.
  In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin,
  H. (eds.), *Advances in Neural Information Processing Systems*,
  volume 33, pp.  11033–11043. Curran Associates, Inc., 2020.
* You et al. (2020)

  You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X.,
  Demmel, J., Keutzer, K., and Hsieh, C.
  Large batch optimization for deep learning: Training bert in 76
  minutes.
  In *International Conference on Learning Representations*, 2020.
  ArXiv 1904.00962.
* Zhang et al. (2019a)

  Zhang, M., Lucas, J., Ba, J., and Hinton, G. E.
  Lookahead optimizer: k steps forward, 1 step back.
  In *Advances in Neural Information Processing Systems 32*,
  2019a.
  ArXiv 1907.08610.
* Zhang et al. (2019b)

  Zhang, X., Tang, Z., Hou, J., and Hao, Y.
  3d human pose estimation via human structure-aware fully connected
  network.
  *Pattern Recognition Letters*, 125:404–410,
  2019b.
  doi: 10.1016/j.patrec.2019.05.020.

## Appendix A Appendix

### A.1 Architecture

![Refer to caption](/html/2206.00664/assets/x3.png)


Figure A.3: Embedding Layer. All attributes of an original input sample
are mapped to an e𝑒e-dimensional embedding space. The position of an
attribute within a sample and the attribute type are conserved by separate e𝑒e-dimensional embeddings. All
three embedding vectors are summed and serve as the final
representation of an input attribute. The input sample is represented
by the concatenation of all its attribute representations.

![Refer to caption](/html/2206.00664/assets/x4.png)


Figure A.4: Summarization Layer. The current prediction vector on the right is mapped to the final prediction vector on the left by separately mapping each current attribute
prediction to its respective final prediction. This final prediction vector lives
in the same space as the original input sample and is used for the
computation of the respective losses.

### A.2 Datasets

#### A.2.1 UCI Dataset Selection

To assess the performance of Hopular and other Deep Learning methods
on small datasets,
we select a subset of 21 datasets from (Klambauer et al., [2017](#bib.bib24)).
The sizes of these datasets range from 200 to 1,000 samples.
We put the focus on smaller sizes, therefore we select
13 datasets with 500 samples or less.
Additionally, we select four datasets with 500 to 750 samples and
four dataset with 750 to 1,000 samples.
Small datasets typically have small test sets,
which introduce a high variance in their evaluations.
This is especially true if they are overly small or unbalanced.
Furthermore, some test sets seem to be not sampled iid from the whole population.
Thus, the method evaluation may be highly dependent on the chosen train/test split and
performance estimates may be skewed.
Problematic datasets in (Klambauer et al., [2017](#bib.bib24)) are characterized
by having a range of accuracy values across well established methods of greater or equal 0.50.50.5
We exclude the problematic datasets
seeds, spectf, libras, dermatology, arrythmia,
and conn-bench-vowel-deterding.
The dataset spect is excluded as its description in (Fernández-Delgado et al., [2014](#bib.bib14))
is in conflict with the available UCI version regarding the number of attributes and samples.
The dataset
heart-hungarian is excluded as the dataset description
is insufficient to distinguish between categorical and continuous attributes,
which is required by some methods.
Since breast-cancer-wisc is practically solved (0.98590.98590.9859 accuracy), it is excluded as
it does not allow to distinguish the performances of the compared methods.
We drop heart-va,
since the best reported method has only a low accuracy of 0.40.40.4.

#### A.2.2 Small-Sized Dataset Description

Table A.3: Overview of small-sized datasets with their number of instances, number
of continuous features, and number of categorical features. All small-sized datasets are classification tasks.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Size  (N𝑁N) | # cont.  features | # cat.  features |
| conn-bench | 208208208 | 606060 | 00 |
| glass | 214214214 | 999 | 00 |
| statlog-heart | 270270270 | 666 | 777 |
| breast-cancer | 286286286 | 00 | 999 |
| heart-cleveland | 303303303 | 666 | 999 |
| haberman-survival | 306306306 | 333 | 00 |
| vertebral-column2 | 310310310 | 666 | 00 |
| vertebral-column3 | 310310310 | 666 | 00 |
| primary-tumor | 330330330 | 00 | 171717 |
| ecoli | 336336336 | 555 | 00 |
| horse-colic | 368368368 | 888 | 191919 |
| congressional-voting | 435435435 | 00 | 161616 |
| cylinder-bands | 512512512 | 202020 | 191919 |
| monks-2 | 601601601 | 666 | 00 |
| statlog-australian-credit | 690690690 | 555 | 999 |
| credit-approval | 690690690 | 666 | 999 |
| blood-transfusion | 748748748 | 444 | 111 |
| energy-y2 | 768768768 | 777 | 00 |
| mammographic | 961961961 | 111 | 555 |
| led-display | 1,00010001,000 | 00 | 666 |
| statlog-german-credit | 1,00010001,000 | 232323 | 00 |

Below we give more precise descriptions of the datasets used in our small-sized experiments:

:   conn-bench-sonar-mines-rocks or Connectionist Bench (Sonar, Mines vs. Rocks): A classification setting of 208 instances with 60 continuous features per instance. The task is to discriminate between sonar sounds from metal vs. rocks.
:   glass or Glass Identification: A classification setting of 214 instances with 9 continuous features per instance. The task is to discriminate between 6 types of glass.
:   statlog-heart: A classification setting of 270 instances with 6 continuous and 7 categorical features per instance. The task is to predict the presence or absence of a heart disease.
:   breast-cancer: A classification setting of 286 instances with 9 categorical features per instance. The task is to predict the presence or absence of breast cancer.
:   heart-cleveland or Heart Disease: A classification setting of 303 instances with 6 continuous and 7 categorical features per instance. The task is to predict the presence or absence of a heart disease.
:   haberman-survival: A classification setting of 306 instances with 3 continuous features per instance. The task is to predict whether patients survived longer than 5 years or not.
:   vertebral-column2, vertebral-column3 or Vertebral Column Dataset: Two classification settings of 310 instances each with 6 continuous features per instance. The task is to classify patients into either 2 or 3 classes.
:   primary-tumor: A classification setting of 330 instances with 17 categorical features per instance. The task is to predict the class of primary tumors.
:   ecoli: A classification setting of 336 instances with 5 continuous and 2 categorical features per instance. The tasks is to classify proteins into 8 classes.
:   horse-colic: A classification setting of 368 instances with 8 continuous and 19 categorical features per instance. The task is to predict the survival or death of a horse.
:   congressional-voting: A classification setting of 435 instances with 16 categorical features per instance. The task is to predict political affiliation.
:   cylinder-bands: A classification setting of 512 instances with 20 continuous and 19 categorical features per instance. The task is to classify the band type.
:   credit-approval: A classification setting of 690 instances with 6 continuous and 9 categorical features per instance. The task is to determine positive or negative feedback for credit card applications.
:   blood-transfusion or Blood Transfusion Service Center: A classification setting of 748 instances with 4 continuous and 1 categorical feature per instance. The task is to predict whether a person donated blood or not.
:   statlog-german-credit: A classification setting of 1,000 instances with 23
    continuous features per instance. The goal is to determine credit-worthiness of customers.
:   mammographic or Mammographic Mass: A classification setting of 961 instances with 1 continuous and 5 categorical features per instance. The task is to discriminate between benign and malignant mammographic masses.
:   led-display: A classification setting of 1,000 instances with 6 categorical features
    per instance. The task is to classify decimal digits from light-emiting diodes with noise.
:   statlog-australian-credit: A classification setting of 690 instances
    with 5 continuous and 9 categorical features. The task to grant customers
    credit-approval or not.
:   energy-y2 or Energy efficiency Data Set:
    A classification setting of 768 instances with 7 continuous
    features per instance. The task is to predict the cooling load for a given building.
:   monks-2 It is part of the Monk’s Problems Data Set. A classification
    task for 601 instances with 6 categorical features. The task is to discriminate
    between two classes.

#### A.2.3 Medium-Sized Dataset Description

Table A.4: Medium-sized datasets with their number of instances, number
of continuous features, and number of categorical features. Classification tasks are marked with (C), whereas regression tasks are marked with (R).

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset | Size  (N𝑁N) | # cont.  features | # cat.  features |
| blastchar (C) | 7,04870487,048 | 333 | 171717 |
| colleges (R) | 7,06470647,064 | 333333 | 121212 |
| gesture-phase (C) | 9,87398739,873 | 313131 | 00 |
| shrutime (C) | 10,0001000010,000 | 222 | 999 |
| sulfur (R) | 10,0821008210,082 | 555 | 00 |
| eye-movements (C) | 10,9361093610,936 | 191919 | 333 |

Below we give more precise descriptions of the datasets used in our medium-sized experiments:

:   shrutime: A classification setting of 10,000 instances with 2 continuous and 9 categorical features per instance. The task is to predict whether a bank account is closed or not.
:   blastchar: A classification setting of 7,048 instances with 3 continuous and 17 categorical features per instance. The task is to predict customer behavior.
:   gesture or gesture-phase or Gesture Phase Segmentation: A classification setting of 9,873 instances with 31 continuous features per instance. The task is to classify gesture phases.
:   eye or eye-movements: A classification setting of 10,936 instances with 19 continuous and 3 categorical features per instance. The task is to discriminate between correct, irrelevant or relevant answers.
:   colleges: A regression setting of 7,064 instances with 33 continuous and
    12 categorical features per instance. The task is to predict pell grant percentages
    for colleges in the USA.
:   sulfur: A regression setting of 10,082 instances with 5 continuous features
    per instance. The task is to predict H2S concentration in a factory module.

### A.3 Hyperparameter selection process

Table A.5: Complete listing of all evaluated hyperparameter settings for NPTs. For all experiments a learning rate of 0.0010.0010.001 as well as a dropout probability of 0.10.10.1 is used. Settings marked with an asterisk (\*) are not performed on conn-bench-sonar-mines-rocks due to out-of-memory issues.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataset  group | # netw.  layers | # att.  heads | label mask.  prob. | feature mask.  prob. | learn. rate  scheduler | emb.  dim. |  |
| small and  medium | 888 | 888 | 1.01.01.0 | 0.150.150.15 | cosine | 323232 |  |
| 161616 | 888 | 1.01.01.0 | 0.150.150.15 | cosine | 323232 |  |
| 888 | 161616 | 1.01.01.0 | 0.150.150.15 | cosine | 323232 |  |
| 161616 | 161616 | 1.01.01.0 | 0.150.150.15 | cosine | 323232 |  |
| 888 | 888 | 0.10.10.1 | 0.150.150.15 | cosine | 323232 |  |
| 888 | 888 | 0.50.50.5 | 0.150.150.15 | cosine | 323232 |  |
| 888 | 888 | 1.01.01.0 | 0.200.200.20 | cosine | 323232 |  |
| 888 | 888 | 1.01.01.0 | 0.150.150.15 | cosine cyclic | 323232 |  |
| small | 888 | 888 | 1.01.01.0 | 0.150.150.15 | cosine | 128128128 |  |
| 161616 | 888 | 1.01.01.0 | 0.150.150.15 | cosine | 128128128 | \* |
| 888 | 161616 | 1.01.01.0 | 0.150.150.15 | cosine | 128128128 |  |
| 161616 | 161616 | 1.01.01.0 | 0.150.150.15 | cosine | 128128128 | \* |
| 888 | 888 | 0.10.10.1 | 0.150.150.15 | cosine | 128128128 |  |
| 888 | 888 | 0.50.50.5 | 0.150.150.15 | cosine | 128128128 |  |
| 888 | 888 | 1.01.01.0 | 0.200.200.20 | cosine | 128128128 |  |
| 888 | 888 | 1.01.01.0 | 0.150.150.15 | cosine cyclic | 128128128 |  |

For the hyperparameter selection process for NPTs we follow (Kossen et al., [2021](#bib.bib26)) and take exactly
the same hyperparameter settings that were successfully used among several
datasets. We use these hyperparameter settings for experiments on small- and medium-sized
datasets. For small-sized datasets we additionally use these settings with an increased
embedding dimension of 128128128. Especially for such datasets the discrimination among similar samples can be a challenging task. This problem can be mitigated
by mapping to a higher-dimensional embedding space where the samples have greater
distances between each other.
NPTs follow a masking procedure similar to (Devlin et al., [2019](#bib.bib10))
which is realized by feature and label masking probabilities.
Following the strategy in (Kossen et al., [2021](#bib.bib26)) we use the LAMB (You et al., [2020](#bib.bib50)) optimizer for all NPT experiments, extended by a Lookahead (Zhang et al., [2019a](#bib.bib51)) wrapper with fixed values.
For LAMB we use βL=(0.9,0.999)subscript𝛽𝐿0.90.999\beta\_{L}=(0.9,0.999), ϵ=1​e−6italic-ϵ1𝑒6\epsilon=1e{-6} and for Lookahead α=0.5𝛼0.5\alpha=0.5, k=6𝑘6k=6.
The hyperparameter settings for NPTs are shown in Table [A.5](#A1.T5 "Table A.5 ‣ A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data").

Table A.6: Complete listing of all evaluated hyperparameter settings for Hopular. For all experiments a learning rate of 0.0010.0010.001 was used. The dropout probabilities pisubscript𝑝𝑖p\_{i}, phsubscript𝑝ℎp\_{h} and posubscript𝑝𝑜p\_{o} refer to the embedding layer, Hopular Block and summarization layer, respectively. The three settings of the second group (medium-sized) were performed in a non-exhaustive way w.r.t. to all medium-sized datasets.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset  group | # Hop.  blocks | # Hop.  nets | β𝛽\beta-scaling  factor | mask  prob. | replace  prob. | weight  decay | dropout | | |
| pisubscript𝑝𝑖p\_{i} | phsubscript𝑝ℎp\_{h} | posubscript𝑝𝑜p\_{o} |
| small and  medium | 444 | 888 | 10{0,2,3}superscript1002310^{\left\{0,2,3\right\}} | 0.0250.0250.025 | 0.1750.1750.175 | 0.10.10.1 | 0.10.10.1 | 0.10.10.1 | 0.010.010.01 |
| 888 | 888 | 10{0,2,3}superscript1002310^{\left\{0,2,3\right\}} | 0.0250.0250.025 | 0.1750.1750.175 | 0.10.10.1 | 0.10.10.1 | 0.10.10.1 | 0.010.010.01 |
| 444 | 161616 | 10{0,2,3}superscript1002310^{\left\{0,2,3\right\}} | 0.0250.0250.025 | 0.1750.1750.175 | 0.10.10.1 | 0.10.10.1 | 0.10.10.1 | 0.010.010.01 |
| 888 | 161616 | 10{0,2,3}superscript1002310^{\left\{0,2,3\right\}} | 0.0250.0250.025 | 0.1750.1750.175 | 0.10.10.1 | 0.10.10.1 | 0.10.10.1 | 0.010.010.01 |
| medium | 888 | 161616 | 10{0}superscript10010^{\left\{0\right\}} | 0.0000.0000.000 | 0.0000.0000.000 | 0.00.00.0 | 0.00.00.0 | 0.00.00.0 | 0.000.000.00 |

For a fair comparison we upper bound Hopular’s capacity by the capacity of NPTs which results
in the settings shown in Table [A.6](#A1.T6 "Table A.6 ‣ A.3 Hyperparameter selection process ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data").
As Hopular provides an additional adjustable scaling factor for β𝛽\beta, we also test scaling factors of 100100100 and 100010001000 to further emphasize nearest-neighbor search.
In our default setting the weighting term γ𝛾\gamma for our objective in Eq. ([10](#S3.E10 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data"))
is annealed using a cosine scheduler starting at 111 with a final value of 00.
For medium-sized datasets we also perform experiments with an initial γ𝛾\gamma value of 0.50.50.5.
We use the original BERT masking as in (Devlin et al., [2019](#bib.bib10)).
Since we store the training data in Hssubscript𝐻𝑠H\_{s} we have to make sure that
the model does not just learn to retrieve the original input sample
from the training set (like a database query). This is why we
independently of BERT masking always
mask the corresponding sample in the training set.
We use default values for masking and dropout.
For the medium-sized datasets we also test two different settings of weight decay, and of dropout probabilities in the
Embedding layer, Hopular block and Summarization layer.
In contrast to NPTs, we always mask all labels.
In our experiments the Hopfield dimension hℎh (as described in Section [3](#S3 "3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data"))
is fixed by the embedding size e𝑒e, the number of features d𝑑d and
the number of Hopfield networks M𝑀M such that h=d⋅e/Mℎ⋅𝑑𝑒𝑀h=d\cdot e/M.
The LAMB (You et al., [2020](#bib.bib50)) optimizer is used for all Hopular experiments, extended by a method similar to Lookahead (Zhang et al., [2019a](#bib.bib51)) but without synchronization of
fast and slow weights. This is analogous to the exponential moving average used in
(Grill et al., [2020](#bib.bib18)).
For LAMB we use βL=(0.9,0.999)subscript𝛽𝐿0.90.999\beta\_{L}=(0.9,0.999), ϵ=1​e−6italic-ϵ1𝑒6\epsilon=1e{-6} and for Lookahead α=0.005𝛼0.005\alpha=0.005, k=1𝑘1k=1.
NPTs and Hopular are both trained for 10,000 epochs with
early stopping.

For XGBoost and CatBoost we use the package hyperopt
and apply the same Bayesian hyperparameter optimization procedure
as described in Shwartz-Ziv & Armon ([2021](#bib.bib39)). For all Boosting methods we
thereby evaluate 1,000 different hpyerparameter settings. More precisely,
the hyperparameters and their search spaces for XGBoost are defined in the following.

* •

  Learning rate: Log-Uniform distribution [−7,0]70[-7,0]
* •

  Max depth: Discrete uniform distribution [1,10]110[1,10]
* •

  Subsample: Uniform distribution [0.2,1]0.21[0.2,1]
* •

  Colsample bytree: Uniform distribution [0.2,1]0.21[0.2,1]
* •

  Colsample bylevel: Uniform distribution [0.2,1]0.21[0.2,1]
* •

  Min child weight: Log-Uniform distribution [−16,2]162[-16,2]
* •

  Alpha: Uniform choice {0,Log-Uniform ​[−16,2]}0Log-Uniform 162\{0,\text{Log-Uniform }[-16,2]\}
* •

  Lambda: Uniform choice {0,Log-Uniform ​[−16,2]}0Log-Uniform 162\{0,\text{Log-Uniform }[-16,2]\}
* •

  Gamma: Uniform choice {0,Log-Uniform ​[−16,2]}0Log-Uniform 162\{0,\text{Log-Uniform }[-16,2]\}
* •

  Number of estimators: 100010001000

It is important to mention that the package hyperopt
defines the Log-Uniform distribution by the
exponents of the respective interval boundaries – e.g.
Log-Uniform​[−7,0]Log-Uniform70\text{Log-Uniform}[-7,0] is defined on [e−7,e0]superscript𝑒7superscript𝑒0[e^{-7},e^{0}].
The hyperparameters and their search spaces for CatBoost are defined in the following.

* •

  Learning rate: Log-Uniform distribution [−5,0]50[-5,0]
* •

  Random strength: Discrete uniform distribution [1,20]120[1,20]
* •

  Max size: Discrete uniform distribution [0,25]025[0,25]
* •

  L2 leaf regularization: Log-Uniform distribution [log⁡1,log⁡10]110[\log 1,\log 10]
* •

  Bagging temperature: Uniform distribution [0,1]01[0,1]
* •

  Leaf estimation iterations: Discrete uniform distribution [1,20]120[1,20]
* •

  Number of estimators: 100010001000

For LightGBM we use the default hyperparameter ranges as specified by hyperopt-sklearn (Komer et al., [2014](#bib.bib25)).

* •

  Learning rate: Log-Uniform distribution [log⁡0.0001,log⁡0.5]−0.00010.00010.50.0001[\log 0.0001,\log 0.5]-0.0001
* •

  Max depth: Discrete uniform distribution [1,11]111[1,11]
* •

  Number of leaves: Discrete uniform distribution [2,121]2121[2,121]
* •

  Gamma: Log-Uniform distribution [log⁡0.001,log⁡5]−0.00010.00150.0001[\log 0.001,\log 5]-0.0001
* •

  Min child weight: Log-Uniform distribution [log⁡1,log⁡100]1100[\log 1,\log 100]
* •

  Subsample: Uniform distribution [0.5,1]0.51[0.5,1]
* •

  Colsample bytree: Uniform distribution [0.5,1]0.51[0.5,1]
* •

  Colsample bylevel: Uniform distribution [0.5,1]0.51[0.5,1]
* •

  Alpha: Log-Uniform distribution [log⁡0.0001,log⁡1]0.00011[\log 0.0001,\log 1]
* •

  Lambda: Log-Uniform distribution [log⁡1,log⁡4]14[\log 1,\log 4]
* •

  Boosting type: Uniform choice {gbdt, dart, goss}gbdt, dart, goss\{\text{gbdt, dart, goss}\}
* •

  Number of estimators: 100010001000

### A.4 Results

In Table [A.7](#A1.T7 "Table A.7 ‣ A.4 Results ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") we show the median rank across
all 21 selected UCI datasets. Methods are ranked for each dataset
according to their accuracy on the respective test set.

Table A.7: Median rank of compared methods across
the datasets of the UCI machine learning repository.
Methods are ranked for each dataset according to the accuracy on the respective test set.
Hopular achieves the lowest median rank of 7.57.57.5, therefore is the best
performing method across the considered UCI datasets.

|  |  |  |  |
| --- | --- | --- | --- |
| Method | Rank | Method | Rank |
| Hopular (DL) | 7.57.57.5 | Rule-Based Methods | 15.015.015.0 |
| Support Vector Machines | 9.59.59.5 | Other Ensembles | 15.015.015.0 |
| Logistic and Multinomial Regression | 10.010.010.0 | BatchNorm (DL) | 15.015.015.0 |
| Random Forest | 11.011.011.0 | Boosting Methods | 15.015.015.0 |
| Self-Normalizing Networks (DL) | 11.011.011.0 | Generalized Linear Models | 15.515.515.5 |
| Non-Parametric Transformers (DL) | 11.011.011.0 | WeightNorm (DL) | 15.515.515.5 |
| Neural Networks (DL) | 11.511.511.5 | Discriminant Analysis | 16.016.016.0 |
| XGBoost | 12.012.012.0 | Other Methods | 17.517.517.5 |
| Multivariate Adaptive Reg. Splines | 12.012.012.0 | ResNet (DL) | 19.019.019.0 |
| Decision Trees | 13.513.513.5 | LayerNorm (DL) | 19.019.019.0 |
| MSRAinit (DL) | 14.014.014.0 | Partial Least Squares | 19.519.519.5 |
| Bagging Methods | 14.014.014.0 | Bayesian Methods | 20.020.020.0 |
| CatBoost | 14.014.014.0 | Nearest Neighbour | 24.024.024.0 |
| LightGBM | 14.514.514.5 | Stacking (Wolpert) | 28.028.028.0 |
| Highway Networks (DL) | 14.514.514.5 |  |  |

### A.5 Memory footprint and runtime estimates

In table [A.8](#A1.T8 "Table A.8 ‣ A.5 Memory footprint and runtime estimates ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") we show the memory footprint of Hopular and NPTs for all medium-sized datasets ranging from the smallest to the largest model. In all cases the whole training set is stored in the memory of module Hssubscript𝐻𝑠H\_{s}. Even in the full batch setting where all the data is used as model input there is no prohibitive memory increase. In contrast, NPTs have a much higher memory memory consumption in the full batch setting. There, for 3 datasets the larger models even run out of memory on an Nvidia A100 GPU.

Table A.8: Memory footprint of Hopular and NPTs in *gibibytes (GiB)* for medium-sized datasets ranging from our smallest to largest model. Settings with a memory footprint of 80.00​+80.00+80.00\text{{\raisebox{0.86108pt}{+}}} are not performed due to out-of-memory issues.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Hopular | | | | | | NPTs | | | | | |
| single sample | | | full batch | | | single sample | | | full batch | | |
| blastchar (C) | 2.382.382.38 | to | 2.752.752.75 | 4.834.834.83 | to | 7.617.617.61 | 1.971.971.97 | to | 2.382.382.38 | 20.4920.4920.49 | to | 56.1756.1756.17 |
| colleges (R) | 3.133.133.13 | to | 3.903.903.90 | 6.586.586.58 | to | 11.6211.6211.62 | 3.983.983.98 | to | 6.096.096.09 | 27.1327.1327.13 | to | 74.5674.5674.56 |
| gesture-phase (C) | 2.772.772.77 | to | 3.413.413.41 | 8.928.928.92 | to | 15.6115.6115.61 | 2.732.732.73 | to | 3.903.903.90 | 40.9540.9540.95 | to | 80.0080.0080.00+ |
| shrutime (C) | 2.602.602.60 | to | 3.233.233.23 | 7.537.537.53 | to | 13.0513.0513.05 | 1.661.661.66 | to | 1.791.791.79 | 36.3036.3036.30 | to | 78.7578.7578.75 |
| sulfur (R) | 2.552.552.55 | to | 3.183.183.18 | 7.547.547.54 | to | 13.1413.1413.14 | 1.551.551.55 | to | 1.591.591.59 | 35.9535.9535.95 | to | 80.0080.0080.00+ |
| eye-movements (C) | 2.682.682.68 | to | 3.283.283.28 | 10.1910.1910.19 | to | 18.2118.2118.21 | 2.112.112.11 | to | 2.672.672.67 | 45.9245.9245.92 | to | 80.0080.0080.00+ |

In table [A.9](#A1.T9 "Table A.9 ‣ A.5 Memory footprint and runtime estimates ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data") we perform measurements on training and inference times.
We show the step time for medium-sized datasets during training. Inference times are assumed to be much lower, as no gradient computation and parameter updates need to be performed.

Table A.9: Step time of Hopular and NPTs in *milliseconds (ms)* during training.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Dataset | Hopular | | NPTs | |
| single sample | full batch | single sample | full batch |
| blastchar (C) | 73.69±0.02uncertain73.690.0273.69\pm 0.02 | 503.45±0.08uncertain503.450.08503.45\pm 0.08 | 81.74±0.11uncertain81.740.1181.74\pm 0.11 | 167.26±0.25uncertain167.260.25167.26\pm 0.25 |
| colleges (R) | 120.15±0.09uncertain120.150.09120.15\pm 0.09 | 824.34±0.17uncertain824.340.17824.34\pm 0.17 | 118.13±0.13uncertain118.130.13118.13\pm 0.13 | 321.32±0.25uncertain321.320.25321.32\pm 0.25 |
| gesture-phase (C) | 95.40±0.03uncertain95.400.0395.40\pm 0.03 | 1,155.47±0.06uncertain1155.470.061,155.47\pm 0.06 | 99.38±0.08uncertain99.380.0899.38\pm 0.08 | 384.58±0.16uncertain384.580.16384.58\pm 0.16 |
| shrutime (C) | 61.90±0.02uncertain61.900.0261.90\pm 0.02 | 652.81±0.04uncertain652.810.04652.81\pm 0.04 | 68.18±0.08uncertain68.180.0868.18\pm 0.08 | 182.11±0.16uncertain182.110.16182.11\pm 0.16 |
| sulfur (R) | 52.71±0.02uncertain52.710.0252.71\pm 0.02 | 629.55±0.04uncertain629.550.04629.55\pm 0.04 | 59.44±0.08uncertain59.440.0859.44\pm 0.08 | 159.86±0.28uncertain159.860.28159.86\pm 0.28 |
| eye-movements (C) | 76.94±0.02uncertain76.940.0276.94\pm 0.02 | 1,141.37±0.03uncertain1141.370.031,141.37\pm 0.03 | 84.21±0.08uncertain84.210.0884.21\pm 0.08 | 338.53±0.18uncertain338.530.18338.53\pm 0.18 |

### A.6 Hopular Intuition: Mimicking Iterative Learning

In our first example we consider Nadaraya-Watson kernel regression
(Watson, [1964](#bib.bib45); Nadaraya, [1964](#bib.bib29); Benedetti, [1977](#bib.bib4); Weinberger & Tesauro, [2007](#bib.bib46)).
The training set is
{(𝒛1,𝒚1),…,(𝒛N,𝒚N)}subscript𝒛1subscript𝒚1…subscript𝒛𝑁subscript𝒚𝑁\{(\bm{z}\_{1},\bm{y}\_{1}),\ldots,(\bm{z}\_{N},\bm{y}\_{N})\}
with inputs 𝒛isubscript𝒛𝑖\bm{z}\_{i} summarized by the input
matrix 𝒁=(𝒛1,…,𝒛N)𝒁subscript𝒛1…subscript𝒛𝑁\bm{Z}=(\bm{z}\_{1},\ldots,\bm{z}\_{N}) and labels 𝒚isubscript𝒚𝑖\bm{y}\_{i} summarized
in the label matrix 𝒀=(𝒚1,…,𝒚N)𝒀subscript𝒚1…subscript𝒚𝑁\bm{Y}=(\bm{y}\_{1},\ldots,\bm{y}\_{N}). The kernel function
is k​(𝒛i,𝒛)𝑘subscript𝒛𝑖𝒛k(\bm{z}\_{i},\bm{z}).
The estimator 𝒈𝒈\bm{g} for 𝒚𝒚\bm{y} given 𝒛𝒛\bm{z} is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒈​(𝒛)𝒈𝒛\displaystyle\bm{g}(\bm{z})\ | =∑i=1N𝒚i​k​(𝒛i,𝒛)∑i=1Nk​(𝒛i,𝒛).absentsuperscriptsubscript𝑖1𝑁subscript𝒚𝑖𝑘subscript𝒛𝑖𝒛superscriptsubscript𝑖1𝑁𝑘subscript𝒛𝑖𝒛\displaystyle=\ \sum\_{i=1}^{N}\bm{y}\_{i}\ \frac{k(\bm{z}\_{i},\bm{z})}{\sum\_{i=1}^{N}k(\bm{z}\_{i},\bm{z})}\ . |  | (11) |

By using the RBF kernel we get:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | k​(𝒛i,𝒛j)𝑘subscript𝒛𝑖subscript𝒛𝑗\displaystyle k(\bm{z}\_{i},\bm{z}\_{j})\ | =exp⁡(−β/2​‖𝒛i−𝒛j‖2)=exp⁡(−β/2​(𝒛iT​𝒛i− 2​𝒛iT​𝒛j+𝒛jT​𝒛j)).absent𝛽2superscriptnormsubscript𝒛𝑖subscript𝒛𝑗2𝛽2superscriptsubscript𝒛𝑖𝑇subscript𝒛𝑖2superscriptsubscript𝒛𝑖𝑇subscript𝒛𝑗superscriptsubscript𝒛𝑗𝑇subscript𝒛𝑗\displaystyle=\ \exp(-\ \beta/2\ {{\left\|\bm{z}\_{i}\ -\ \bm{z}\_{j}\right\|}}^{2})\ =\ \exp(-\ \beta/2\ (\bm{z}\_{i}^{T}\bm{z}\_{i}\ -\ 2\ \bm{z}\_{i}^{T}\bm{z}\_{j}\ +\ \bm{z}\_{j}^{T}\bm{z}\_{j}))\ . |  | (12) |

For normalized vector 𝒛isubscript𝒛𝑖\bm{z}\_{i} we have 𝒛iT​𝒛i=‖𝒛i‖2=1superscriptsubscript𝒛𝑖𝑇subscript𝒛𝑖superscriptnormsubscript𝒛𝑖21\bm{z}\_{i}^{T}\bm{z}\_{i}={{\left\|\bm{z}\_{i}\right\|}}^{2}=1, therefore

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | k​(𝒛i,𝒛j)𝑘subscript𝒛𝑖subscript𝒛𝑗\displaystyle k(\bm{z}\_{i},\bm{z}\_{j})\ | =exp⁡(−β​(1−𝒛iT​𝒛j))=c​exp⁡(β​𝒛iT​𝒛j).absent𝛽1superscriptsubscript𝒛𝑖𝑇subscript𝒛𝑗𝑐𝛽superscriptsubscript𝒛𝑖𝑇subscript𝒛𝑗\displaystyle=\ \exp(-\ \beta\ (1\ -\ \bm{z}\_{i}^{T}\bm{z}\_{j}))\ =\ c\ \exp(\beta\ \bm{z}\_{i}^{T}\bm{z}\_{j})\ . |  | (13) |

We obtain for Nadaraya–Watson kernel regression with the RBF kernel and normalized inputs:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒈​(𝒛)𝒈𝒛\displaystyle\bm{g}(\bm{z})\ | =𝒀​softmax​(β​𝒁T​𝒛).absent𝒀softmax𝛽superscript𝒁𝑇𝒛\displaystyle=\ \bm{Y}\ \mathrm{softmax}(\beta\ \bm{Z}^{T}\ \bm{z})\ . |  | (14) |

Metric learning for kernel regression learns the kernel k𝑘k
which is the distance function (Weinberger & Tesauro, [2007](#bib.bib46)). A Hopular Block can
do the same in Eq. [7](#S3.E7 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data") via learning the weight matrices 𝑾𝑿subscript𝑾𝑿\bm{W}\_{\bm{X}}
and 𝑾𝝃subscript𝑾𝝃\bm{W}\_{\bm{\xi}}. If we set in Eq. [14](#A1.E14 "In A.6 Hopular Intuition: Mimicking Iterative Learning ‣ Appendix A Appendix ‣ Hopular: Modern Hopfield Networks for Tabular Data"):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒁T=𝑿T​𝑾𝑿T,𝒛=𝑾𝝃​𝝃,𝒀=𝑾𝑺​𝑾𝑿​𝑿formulae-sequencesuperscript𝒁𝑇superscript𝑿𝑇superscriptsubscript𝑾𝑿𝑇formulae-sequence𝒛subscript𝑾𝝃𝝃𝒀subscript𝑾𝑺subscript𝑾𝑿𝑿\displaystyle\bm{Z}^{T}=\bm{X}^{T}\ \bm{W}\_{\bm{X}}^{T},\ \ \ \bm{z}=\bm{W}\_{\bm{\xi}}\ \bm{\xi},\ \ \ \bm{Y}=\bm{W}\_{\bm{S}}\ \bm{W}\_{\bm{X}}\ \bm{X} |  | (15) |

then we obtain Eq. [7](#S3.E7 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data"), with the fixed label matrix 𝒀𝒀\bm{Y}.

In the second example we show how Hopular can realize a linear model
with the AdaBoost Objective. The AdaBoost objective for classification
with a binary target y∈{−1,+1}𝑦11y\in\{-1,+1\} can be written as follows – see Eq. 3 and Eq. 4 in (Shen & Li, [2010](#bib.bib38)):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | LL\displaystyle\mathrm{L}\ | =ln​∑i=1Nexp⁡(−yi​g​(𝒛i)).absentsuperscriptsubscript𝑖1𝑁subscript𝑦𝑖𝑔subscript𝒛𝑖\displaystyle=\ \ln\sum\_{i=1}^{N}\exp(-\ y\_{i}\ g(\bm{z}\_{i}))\ . |  | (16) |

We use this objective for learning the linear model:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | g​(𝒛i)𝑔subscript𝒛𝑖\displaystyle g(\bm{z}\_{i})\ | =β​𝝃T​𝒛i.absent𝛽superscript𝝃𝑇subscript𝒛𝑖\displaystyle=\ \beta\ \bm{\xi}^{T}\bm{z}\_{i}\ . |  | (17) |

The objective multiplied by β−1superscript𝛽1\beta^{-1} with 𝒀𝒀\bm{Y} as the diagonal matrix
of the targets {𝒚1,⋯,𝒚N}subscript𝒚1⋯subscript𝒚𝑁\{\bm{y}\_{1},\cdots,\bm{y}\_{N}\} becomes:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | LL\displaystyle\mathrm{L}\ | =β−1​ln​∑i=1Nexp⁡(−β​yi​𝝃T​𝒛i)=lse​(β,−𝒀​𝒁T​𝝃),absentsuperscript𝛽1superscriptsubscript𝑖1𝑁𝛽subscript𝑦𝑖superscript𝝃𝑇subscript𝒛𝑖lse𝛽𝒀superscript𝒁𝑇𝝃\displaystyle=\ \beta^{-1}\ \ln\sum\_{i=1}^{N}\exp(-\ \beta\ y\_{i}\ \bm{\xi}^{T}\bm{z}\_{i})\ =\ \mathrm{lse}(\beta\ ,\ -\ \bm{Y}\ \bm{Z}^{T}\ \bm{\xi})\ , |  | (18) |

where lselse\mathrm{lse} is the log-sum-exponential function.
The gradient of this objective is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ∂L∂𝝃L𝝃\displaystyle\frac{\partial\mathrm{L}}{\partial\bm{\xi}}\ | =−𝒁​𝒀​softmax​(−β​𝒀​𝒁T​𝝃).absent𝒁𝒀softmax𝛽𝒀superscript𝒁𝑇𝝃\displaystyle=\ -\ \bm{Z}\ \bm{Y}\ \mathrm{softmax}(-\ \beta\ \bm{Y}\ \bm{Z}^{T}\ \bm{\xi})\ . |  | (19) |

This is Eq. [7](#S3.E7 "In 3 Hopular: Modern Hopfield Networks for Tabular Data ‣ Hopular: Modern Hopfield Networks for Tabular Data") with:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒀​𝒁T=𝑿T​𝑾𝑿T,𝑾𝝃=𝑰,𝑾𝑺=𝑰formulae-sequence𝒀superscript𝒁𝑇superscript𝑿𝑇superscriptsubscript𝑾𝑿𝑇formulae-sequencesubscript𝑾𝝃𝑰subscript𝑾𝑺𝑰\displaystyle\bm{Y}\ \bm{Z}^{T}=\bm{X}^{T}\ \bm{W}\_{\bm{X}}^{T},\ \ \ \bm{W}\_{\bm{\xi}}=\bm{I},\ \ \ \bm{W}\_{\bm{S}}=\bm{I} |  | (20) |

Thus, a Hopular Block can implement a gradient descent update rule for a linear
classification model using the AdaBoost objective function. The current
prediction 𝝃𝝃\bm{\xi} comes from the previous layer.

These are two additional examples among the standard iterative learning
algorithms which Hopular can mimic.

### A.7 Source code

Source code is available at: <https://github.com/ml-jku/hopular>

[◄](/html/2206.00663)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2206.00664)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2206.00664)
[View original  
on arXiv](https://arxiv.org/abs/2206.00664)[►](/html/2206.00665)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Mar 11 19:41:03 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
