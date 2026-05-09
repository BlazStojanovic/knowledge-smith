---
arxiv: '1805.06440'
authors:
- Ira Shavitt
- Eran Segal
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Regularization Learning Networks: Deep Learning for Tabular Datasets'
url: http://arxiv.org/abs/1805.06440v3
year: 2018
---

[1805.06440] Regularization Learning Networks: Deep Learning for Tabular Datasets














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



# Regularization Learning Networks: Deep Learning for Tabular Datasets

Ira Shavitt
  
Weizmann Institute of Science
  
irashavitt@gmail.com
  
&Eran Segal
  
Weizmann Institute of Science
  
eran.segal@weizmann.ac.il

###### Abstract

Despite their impressive performance, Deep Neural Networks
(DNNs) typically underperform Gradient Boosting Trees (GBTs)
on many tabular-dataset learning tasks. We propose that applying a
different regularization coefficient to each weight might boost the
performance of DNNs by allowing them to make more use of the more
relevant inputs. However, this will lead to an intractable number
of hyperparameters. Here, we introduce Regularization Learning
Networks (RLNs), which overcome this challenge by introducing an
efficient hyperparameter tuning scheme which minimizes a new Counterfactual
Loss. Our results show that RLNs significantly improve DNNs on tabular
datasets, and achieve comparable results to GBTs, with the best performance
achieved with an ensemble that combines GBTs and RLNs. RLNs produce
extremely sparse networks, eliminating up to 99.8%percent99.899.8\% of the network
edges and 82%percent8282\% of the input features, thus providing more interpretable
models and reveal the importance that the network assigns to different
inputs. RLNs could efficiently learn a single network in datasets
that comprise both tabular and unstructured data, such as in the setting
of medical imaging accompanied by electronic health records. An open
source implementation of RLN can be found at <https://github.com/irashavitt/regularization_learning_networks>.

## 1 Introduction

Despite their impressive achievements on various prediction tasks
on datasets with distributed representation [[14](#bib.bib14), [4](#bib.bib4), [5](#bib.bib5)]
such as images [[19](#bib.bib19)], speech [[9](#bib.bib9)], and
text [[18](#bib.bib18)], there are many tasks in which
Deep Neural Networks (DNNs) underperform compared to other
models such as Gradient Boosting Trees (GBTs). This is evident
in various Kaggle [[1](#bib.bib1), [2](#bib.bib2)], or KDD
Cup [[7](#bib.bib7), [16](#bib.bib16), [27](#bib.bib27)] competitions, which are
typically won by GBT-based approaches and specifically by its XGBoost
[[8](#bib.bib8)] implementation, either when run alone or within a combination
of several different types of models.

The datasets in which neural networks are inferior to GBTs typically
have different statistical properties. Consider the task of image
recognition as compared to the task of predicting the life expectancy
of patients based on electronic health records. One key difference
is that in image classification, many pixels need to change in order
for the image to depict a different object [[25](#bib.bib25)].111This is not contradictory to the existence of adversarial examples
[[12](#bib.bib12)], which are able to fool DNNs by changing a small
number of input features, but do not actually depict a different object,
and generally are not able to fool humans. In contrast, the relative contribution of the input features in the
electronic health records example can vary greatly: Changing a single
input such as the age of the patient can profoundly impact the life
expectancy of the patient, while changes in other input features,
such as the time that passed since the last test was taken, may have
smaller effects.

We hypothesized that this potentially large variability in the relative
importance of different input features may partly explain the lower
performance of DNNs on such tabular datasets [[11](#bib.bib11)].
One way to overcome this limitation could be to assign a different
regularization coefficient to every weight, which might allow the
network to accommodate the non-distributed representation and the
variability in relative importance found in tabular datasets.

This will require tuning a large number of hyperparameters. The default
approach to hyperparameter tuning is using derivative-free optimization
of the validation loss, i.e., a loss of a subset of the training set
which is not used to fit the model. This approach becomes computationally
intractable very quickly.

Here, we present a new hyperparameter tuning technique, in which we
optimize the regularization coefficients using a newly introduced
loss function, which we term the Counterfactual Loss, orℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF}.
We term the networks that apply this technique Regularization
Learning Networks (RLNs). In RLNs, the regularization coefficients
are optimized together with learning the network weight parameters.
We show that RLNs significantly and substantially outperform DNNs
with other regularization schemes, and achieve comparable results
to GBTs. When used in an ensemble with GBTs, RLNs achieves state of
the art results on several prediction tasks on a tabular dataset with
varying relative importance for different features.

## 2 Related work

Applying different regularization coefficients to different parts
of the network is a common practice. The idea of applying different
regularization coefficients to every weight was introduced [[23](#bib.bib23)],
but it was only applied to images with a toy model to demonstrate
the ability to optimize many hyperparameters.

Our work is also related to the rich literature of works on hyperparameter
optimization [[29](#bib.bib29)]. These works
mainly focus on derivative-free optimization [[30](#bib.bib30), [6](#bib.bib6), [17](#bib.bib17)].
Derivative-based hyperparameter optimization is introduced in [[3](#bib.bib3)]
for linear models and in [[23](#bib.bib23)] for neural networks. In
these works, the hyperparameters are optimized using the gradients
of the validation loss. Practically, this means that every optimization
step of the hyperparameters requires training the whole network and
back propagating the loss to the hyperparameters. [[21](#bib.bib21)]
showed a more efficient derivative based way for hyperparameter optimization,
which still required a substantial amount of additional parameters.
[[22](#bib.bib22)] introduce an optimization technique similar
to the one introduced in this paper, however, the optimization technique
in [[22](#bib.bib22)] requires a validation set, and only
optimizes a single regularization coefficient for each layer, and
at most 10-20 hyperparameters in any network. In comparison, training
RLNs doesn’t require a validation set, assigns a different regularization
coefficient for every weight, which results in up to millions of hyperparameters,
optimized efficiently. Additionally, RLNs optimize the coefficients
in the log space and adds a projection after every update to counter
the vanishing of the coefficients. Most importantly, the efficient
optimization of the hyperparameters was applied to images and not
to dataset with non-distributed representation like tabular datasets.

DNNs have been successfully applied to tabular datasets like electronic
health records, in [[26](#bib.bib26), [24](#bib.bib24)]. The use of RLN
is complementary to these works, and might improve their results and
allow the use of deeper networks on smaller datasets.

To the best of our knowledge, our work is the first to illustrate
the statistical difference in distributed and non-distributed representations,
to hypothesize that addition of hyperparameters could enable neural
networks to achieve good results on datasets with non-distributed
representations such as tabular datasets, and to efficiently train
such networks on a real-world problems to significantly and substantially
outperform networks with other regularization schemes.

## 3 Regularization Learning

Generally, when using regularization, we minimize ℒ~​(Z,W,λ)=ℒ​(Z,W)+exp⁡(λ)⋅∑i=1n‖wi‖~ℒ𝑍𝑊𝜆ℒ𝑍𝑊⋅𝜆superscriptsubscript𝑖1𝑛normsubscript𝑤𝑖\tilde{\mathcal{L}}\left(Z,W,\lambda\right)=\mathcal{L}\left(Z,W\right)+\exp\left(\lambda\right)\cdot\sum\_{i=1}^{n}\left\|w\_{i}\right\|,
where Z={(xm,ym)}m=1M𝑍superscriptsubscriptsubscript𝑥𝑚subscript𝑦𝑚𝑚1𝑀Z=\left\{\left(x\_{m},y\_{m}\right)\right\}\_{m=1}^{M} are
the training samples, ℒℒ\mathcal{L} is the loss function, W={wi}i=1n𝑊superscriptsubscriptsubscript𝑤𝑖𝑖1𝑛W=\left\{w\_{i}\right\}\_{i=1}^{n}
are the weights of the model, ∥⋅∥\left\|\cdot\right\| is some
norm, and λ𝜆\lambda is the regularization coefficient,222The notation for the regularization term is typically λ⋅∑i=1n‖wi‖⋅𝜆superscriptsubscript𝑖1𝑛normsubscript𝑤𝑖\lambda\cdot\sum\_{i=1}^{n}\left\|w\_{i}\right\|.
We use the notation exp⁡(λ)⋅∑i=1n‖wi‖⋅𝜆superscriptsubscript𝑖1𝑛normsubscript𝑤𝑖\exp\left(\lambda\right)\cdot\sum\_{i=1}^{n}\left\|w\_{i}\right\|
to force the coefficients to be positive, to accelerate their optimization
and to simplify the calculations shown. a hyperparameter of the network. Hyperparameters of the network,
like λ𝜆\lambda, are usually obtained using cross-validation, which
is the application of derivative-free optimization on ℒC​V​(Zt,Zv,λ)subscriptℒ𝐶𝑉subscript𝑍𝑡subscript𝑍𝑣𝜆\mathcal{L}\_{CV}\left(Z\_{t},Z\_{v},\lambda\right)
with respect to λ𝜆\lambda where ℒC​V​(Zt,Zv,λ)=ℒ​(Zv,arg⁡minW⁡ℒ~​(Zt,W,λ))subscriptℒ𝐶𝑉subscript𝑍𝑡subscript𝑍𝑣𝜆ℒsubscript𝑍𝑣subscript𝑊~ℒsubscript𝑍𝑡𝑊𝜆\mathcal{L}\_{CV}\left(Z\_{t},Z\_{v},\lambda\right)=\mathcal{L}\left(Z\_{v},\arg\min\_{W}\tilde{\mathcal{L}}\left(Z\_{t},W,\lambda\right)\right)
and (Zt,Zv)subscript𝑍𝑡subscript𝑍𝑣\left(Z\_{t},Z\_{v}\right) is some partition of Z𝑍Z into train
and validation sets, respectively.

If a different regularization coefficient is assigned to each weight
in the network, our learning loss becomes ℒ†​(Z,W,Λ)=ℒ​(Z,W)+∑i=1nexp⁡(λi)⋅‖wi‖superscriptℒ†𝑍𝑊Λℒ𝑍𝑊superscriptsubscript𝑖1𝑛⋅subscript𝜆𝑖normsubscript𝑤𝑖\mathcal{L}^{\dagger}\left(Z,W,\Lambda\right)=\mathcal{L}\left(Z,W\right)+\sum\_{i=1}^{n}\exp\left(\lambda\_{i}\right)\cdot\left\|w\_{i}\right\|,
where Λ={λi}i=1nΛsuperscriptsubscriptsubscript𝜆𝑖𝑖1𝑛\Lambda=\left\{\lambda\_{i}\right\}\_{i=1}^{n} are the regularization
coefficients. Using ℒ†superscriptℒ†\mathcal{L}^{\dagger} will require n𝑛n hyperparameters,
one for every network parameter, which makes tuning with cross-validation
intractable, even for very small networks. We would like to keep using
ℒ†superscriptℒ†\mathcal{L}^{\dagger} to update the weights, but to find a more
efficient way to tune ΛΛ\Lambda. One way to do so is through SGD,
but it is unclear which loss to minimize: ℒℒ\mathcal{L} doesn’t have
a derivative with respect to ΛΛ\Lambda, while ℒ†superscriptℒ†\mathcal{L}^{\dagger}
has trivial optimal values, arg⁡minΛ⁡ℒ†​(Z,W,Λ)={−∞}i=1nsubscriptΛsuperscriptℒ†𝑍𝑊Λsuperscriptsubscript𝑖1𝑛\arg\min\_{\Lambda}\mathcal{L}^{\dagger}\left(Z,W,\Lambda\right)=\left\{-\infty\right\}\_{i=1}^{n}.
ℒC​Vsubscriptℒ𝐶𝑉\mathcal{L}\_{CV} has a non-trivial dependency on ΛΛ\Lambda, but
it is very hard to evaluate ∂ℒC​V∂Λsubscriptℒ𝐶𝑉Λ\frac{\partial\mathcal{L}\_{CV}}{\partial\Lambda}.

We introduce a new loss function, called the Counterfactual
Loss ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF}, which has a non-trivial dependency on ΛΛ\Lambda
and can be evaluated efficiently. For every time-step t𝑡t during
the training, let Wtsubscript𝑊𝑡W\_{t} and ΛtsubscriptΛ𝑡\Lambda\_{t} be the weights and regularization
coefficients of the network, respectively, and let wt,i∈Wtsubscript𝑤

𝑡𝑖subscript𝑊𝑡w\_{t,i}\in W\_{t}
and λt,i∈Λtsubscript𝜆

𝑡𝑖subscriptΛ𝑡\lambda\_{t,i}\in\Lambda\_{t} be the weight and the regularization
coefficient of the same edge i𝑖i in the network. When optimizing
using SGD, the value of this weight in the next time-step will be
wt+1,i=wt,i−η⋅∂ℒ†​(Zt,Wt,Λt)∂wt,isubscript𝑤

𝑡1𝑖subscript𝑤

𝑡𝑖⋅𝜂superscriptℒ†subscript𝑍𝑡subscript𝑊𝑡subscriptΛ𝑡subscript𝑤

𝑡𝑖w\_{t+1,i}=w\_{t,i}-\eta\cdot\frac{\partial\mathcal{L}^{\dagger}\left(Z\_{t},W\_{t},\Lambda\_{t}\right)}{\partial w\_{t,i}},
where η𝜂\eta is the learning rate, and Ztsubscript𝑍𝑡Z\_{t} is the training batch
at time t𝑡t.333We assume vanilla SGD is used in this analysis for brevity, but the
analysis holds for any derivative-based optimization method. We can split the gradient into two parts:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | wt+1,isubscript𝑤  𝑡1𝑖\displaystyle w\_{t+1,i} | =wt,i−η⋅(gt,i+rt,i)absentsubscript𝑤  𝑡𝑖⋅𝜂subscript𝑔  𝑡𝑖subscript𝑟  𝑡𝑖\displaystyle=w\_{t,i}-\eta\cdot\left(g\_{t,i}+r\_{t,i}\right) |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | gt,isubscript𝑔  𝑡𝑖\displaystyle g\_{t,i} | =∂ℒ​(Zt,Wt)∂wt,iabsentℒsubscript𝑍𝑡subscript𝑊𝑡subscript𝑤  𝑡𝑖\displaystyle=\frac{\partial\mathcal{L}\left(Z\_{t},W\_{t}\right)}{\partial w\_{t,i}} |  | (2) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | rt,isubscript𝑟  𝑡𝑖\displaystyle r\_{t,i} | =∂∂wt,i​(∑j=1nexp⁡(λt,j)⋅‖wt,j‖)=exp⁡(λt,i)⋅∂‖wt,i‖∂wt,iabsentsubscript𝑤  𝑡𝑖superscriptsubscript𝑗1𝑛⋅subscript𝜆  𝑡𝑗normsubscript𝑤  𝑡𝑗⋅subscript𝜆  𝑡𝑖normsubscript𝑤  𝑡𝑖subscript𝑤  𝑡𝑖\displaystyle=\frac{\partial}{\partial w\_{t,i}}\left(\sum\_{j=1}^{n}\exp\left(\lambda\_{t,j}\right)\cdot\left\|w\_{t,j}\right\|\right)=\exp\left(\lambda\_{t,i}\right)\cdot\frac{\partial\left\|w\_{t,i}\right\|}{\partial w\_{t,i}} |  | (3) |

We call gt,isubscript𝑔

𝑡𝑖g\_{t,i} the gradient of the empirical loss ℒℒ\mathcal{L}
and rt,isubscript𝑟

𝑡𝑖r\_{t,i} the gradient of the regularization term. All but one
of the addends of rt,isubscript𝑟

𝑡𝑖r\_{t,i} vanished since ∂∂wt,i​(exp⁡(λt,j)⋅‖wt,j‖)=0subscript𝑤

𝑡𝑖⋅subscript𝜆

𝑡𝑗normsubscript𝑤

𝑡𝑗0\frac{\partial}{\partial w\_{t,i}}\left(\exp\left(\lambda\_{t,j}\right)\cdot\left\|w\_{t,j}\right\|\right)=0
for every j≠i𝑗𝑖j\neq i. Denote by Wt+1={wt+1,i}i=1nsubscript𝑊𝑡1superscriptsubscriptsubscript𝑤

𝑡1𝑖𝑖1𝑛W\_{t+1}=\left\{w\_{t+1,i}\right\}\_{i=1}^{n}
the weights in the next time-step, which depend on Ztsubscript𝑍𝑡Z\_{t}, Wtsubscript𝑊𝑡W\_{t},
ΛtsubscriptΛ𝑡\Lambda\_{t}, and η𝜂\eta, as shown in Equation [1](#S3.E1 "In 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"),
and define the Counterfactual Loss to be

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | ℒC​F​(Zt,Zt+1,Wt,Λt,η)subscriptℒ𝐶𝐹subscript𝑍𝑡subscript𝑍𝑡1subscript𝑊𝑡subscriptΛ𝑡𝜂\displaystyle\mathcal{L}\_{CF}\left(Z\_{t},Z\_{t+1},W\_{t},\Lambda\_{t},\eta\right) | =ℒ​(Zt+1,Wt+1)absentℒsubscript𝑍𝑡1subscript𝑊𝑡1\displaystyle=\mathcal{L}\left(Z\_{t+1},W\_{t+1}\right) |  | (4) |

ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF} is the empirical loss ℒℒ\mathcal{L}, where the
weights have already been updated using SGD over the regularized loss
ℒ†superscriptℒ†\mathcal{L}^{\dagger}. We call this the Counterfactual Loss since
we are asking a counterfactual question: What would have been
the loss of the network had we updated the weights with respect to
ℒ†superscriptℒ†\mathcal{L}^{\dagger}? We will use ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF} to optimize
the regularization coefficients using SGD while learning the
weights of the network simultaneously using ℒ†superscriptℒ†\mathcal{L}^{\dagger}.
We call this technique Regularization Learning, and networks that
employ it Regularization Learning Networks (RLNs).

###### Theorem 1.

The gradient of the Counterfactual loss
with respect to the regularization coefficient is ∂ℒC​F∂λt,i=−η⋅gt+1,i⋅rt,isubscriptℒ𝐶𝐹subscript𝜆

𝑡𝑖⋅𝜂subscript𝑔

𝑡1𝑖subscript𝑟

𝑡𝑖\frac{\partial\mathcal{L}\_{CF}}{\partial\lambda\_{t,i}}=-\eta\cdot g\_{t+1,i}\cdot r\_{t,i}

###### Proof.

ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF} only depends on λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i} through wt+1,isubscript𝑤

𝑡1𝑖w\_{t+1,i},
allowing us to use the chain rule ∂ℒC​F∂λt,i=∂ℒC​F∂wt+1,i⋅∂wt+1,i∂λt,isubscriptℒ𝐶𝐹subscript𝜆

𝑡𝑖⋅subscriptℒ𝐶𝐹subscript𝑤

𝑡1𝑖subscript𝑤

𝑡1𝑖subscript𝜆

𝑡𝑖\frac{\partial\mathcal{L}\_{CF}}{\partial\lambda\_{t,i}}=\frac{\partial\mathcal{L}\_{CF}}{\partial w\_{t+1,i}}\cdot\frac{\partial w\_{t+1,i}}{\partial\lambda\_{t,i}}.
The first multiplier is the gradient gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i}. Regarding the second
multiplier, from Equation [1](#S3.E1 "In 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets") we see that only rt,isubscript𝑟

𝑡𝑖r\_{t,i}
depends on λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i}. Combining with Equation [3](#S3.E3 "In 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets") leaves
us with:

|  |  |  |
| --- | --- | --- |
|  | ∂wt+1,i∂λt,i=∂∂λt,i​(wt,i−η⋅(gt,i+rt,i))=−η⋅∂rt,i∂λt,i=subscript𝑤  𝑡1𝑖subscript𝜆  𝑡𝑖subscript𝜆  𝑡𝑖subscript𝑤  𝑡𝑖⋅𝜂subscript𝑔  𝑡𝑖subscript𝑟  𝑡𝑖⋅𝜂subscript𝑟  𝑡𝑖subscript𝜆  𝑡𝑖absent\displaystyle\frac{\partial w\_{t+1,i}}{\partial\lambda\_{t,i}}=\frac{\partial}{\partial\lambda\_{t,i}}\left(w\_{t,i}-\eta\cdot\left(g\_{t,i}+r\_{t,i}\right)\right)=-\eta\cdot\frac{\partial r\_{t,i}}{\partial\lambda\_{t,i}}= |  |
|  |  |  |
| --- | --- | --- |
|  | =−η⋅∂∂λt,i​(exp⁡(λt,i)⋅∂‖wt,i‖∂wt,i)=−η⋅exp⁡(λt,i)⋅∂‖wt,i‖∂wt,i=−η⋅rt,iabsent⋅𝜂subscript𝜆  𝑡𝑖⋅subscript𝜆  𝑡𝑖normsubscript𝑤  𝑡𝑖subscript𝑤  𝑡𝑖⋅𝜂subscript𝜆  𝑡𝑖normsubscript𝑤  𝑡𝑖subscript𝑤  𝑡𝑖⋅𝜂subscript𝑟  𝑡𝑖\displaystyle=-\eta\cdot\frac{\partial}{\partial\lambda\_{t,i}}\left(\exp\left(\lambda\_{t,i}\right)\cdot\frac{\partial\left\|w\_{t,i}\right\|}{\partial w\_{t,i}}\right)=-\eta\cdot\exp\left(\lambda\_{t,i}\right)\cdot\frac{\partial\left\|w\_{t,i}\right\|}{\partial w\_{t,i}}=-\eta\cdot r\_{t,i} |  |

∎

![Refer to caption](/html/1805.06440/assets/x1.png)


Figure 1: The input features, sorted by their R2superscript𝑅2R^{2} correlation to the label.
We display the microbiome dataset, with the covariates marked, in
comparison the MNIST dataset[[20](#bib.bib20)].

Theorem [1](#Thmthm1 "Theorem 1. ‣ 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets") gives us the update rule λt+1,i=λt,i−ν⋅∂ℒC​F∂λt,i=λt,i+ν⋅η⋅gt+1,i⋅rt,isubscript𝜆

𝑡1𝑖subscript𝜆

𝑡𝑖⋅𝜈subscriptℒ𝐶𝐹subscript𝜆

𝑡𝑖subscript𝜆

𝑡𝑖⋅𝜈𝜂subscript𝑔

𝑡1𝑖subscript𝑟

𝑡𝑖\lambda\_{t+1,i}=\lambda\_{t,i}-\nu\cdot\frac{\partial\mathcal{L}\_{CF}}{\partial\lambda\_{t,i}}=\lambda\_{t,i}+\nu\cdot\eta\cdot g\_{t+1,i}\cdot r\_{t,i},
where ν𝜈\nu is the learning rate of the regularization coefficients.

Intuitively, the gradient of the Counterfactual Loss has an opposite
sign to the product of gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i} and rt,isubscript𝑟

𝑡𝑖r\_{t,i}. Comparing this
result with Equation [1](#S3.E1 "In 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"), this means that when gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i}
and rt,isubscript𝑟

𝑡𝑖r\_{t,i} agree in sign, the regularization helps reduce the
loss, and we can strengthen it by increasing λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i}. When
they disagree, this means that the regularization hurts the performance
of the network, and we should relax it for this weight.

The size of the Counterfactual gradient is proportional to the product
of the sizes of gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i} and rt,isubscript𝑟

𝑡𝑖r\_{t,i}. When gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i} is
small, wt+1,isubscript𝑤

𝑡1𝑖w\_{t+1,i} does not affect the loss ℒℒ\mathcal{L} much,
and when rt,isubscript𝑟

𝑡𝑖r\_{t,i} is small, λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i} does not affect wt+1,isubscript𝑤

𝑡1𝑖w\_{t+1,i}
much. In both cases, λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i} has a small effect on ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF}.
Only when both rt,isubscript𝑟

𝑡𝑖r\_{t,i} is large (meaning that λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i}
affects wt+1subscript𝑤𝑡1w\_{t+1}), and gt+1,isubscript𝑔

𝑡1𝑖g\_{t+1,i} is large (meaning that wt+1subscript𝑤𝑡1w\_{t+1}
affects ℒℒ\mathcal{L}), λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i} has a large effect on ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF},
and we get a large gradient ∂ℒC​F∂λt,isubscriptℒ𝐶𝐹subscript𝜆

𝑡𝑖\frac{\partial\mathcal{L}\_{CF}}{\partial\lambda\_{t,i}}.

At the limit of many training iterations, λt,isubscript𝜆

𝑡𝑖\lambda\_{t,i} tends to
continuously decrease. We try to give some insight to this dynamics
in the supplementary material. To address this issue, we project the
regularization coefficients onto a simplex after updating them:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | λ~t+1,isubscript~𝜆  𝑡1𝑖\displaystyle\widetilde{\lambda}\_{t+1,i} | =λt,i+ν⋅η⋅gt+1,i⋅rt,iabsentsubscript𝜆  𝑡𝑖⋅𝜈𝜂subscript𝑔  𝑡1𝑖subscript𝑟  𝑡𝑖\displaystyle=\lambda\_{t,i}+\nu\cdot\eta\cdot g\_{t+1,i}\cdot r\_{t,i} |  | (5) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | λt+1,isubscript𝜆  𝑡1𝑖\displaystyle\lambda\_{t+1,i} | =λ~t+1,i+(θ−∑j=1nλ~t+1,jn)absentsubscript~𝜆  𝑡1𝑖𝜃superscriptsubscript𝑗1𝑛subscript~𝜆  𝑡1𝑗𝑛\displaystyle=\widetilde{\lambda}\_{t+1,i}+\left(\theta-\frac{\sum\_{j=1}^{n}\widetilde{\lambda}\_{t+1,j}}{n}\right) |  | (6) |

where θ𝜃\theta is the normalization factor of the regularization
coefficients, a hyperparameter of the network tuned using cross-validation.
This results in a zero-sum game behavior in the regularization, where
a relaxation in one edge allows us to strengthen the regularization
in other parts of the network. This could lead the network to assign
a modular regularization profile, where uninformative connections
are heavily regularized and informative connection get a very relaxed
regularization, which might boost performance on datasets with non-distributed
representation such as tabular datasets. The full algorithm is described
in the supplementary material.

![Refer to caption](/html/1805.06440/assets/x2.png)


Figure 2: Prediction of traits using microbiome data and covariates, given as
the overall explained variance (R2superscript𝑅2R^{2}).

## 4 Experiments

We demonstrate the performance of our method on the problem of predicting
human traits from gut microbiome data and basic covariates (age, gender,
BMI). The human gut microbiome is the collection of microorganisms
found in the human gut and is composed of trillions of cells including
bacteria, eukaryotes, and viruses. In recent years, there have been
major advances in our understanding of the microbiome and its connection
to human health. Microbiome composition is determined by DNA sequencing
human stool samples that results in short (75-100 basepairs) DNA reads.
By mapping these short reads to databases of known bacterial species,
we can deduce both the source species and gene from which each short
read originated. Thus, upon mapping a collection of different samples,
we obtain a matrix of estimated relative species abundances for each
person and a matrix of the estimated relative gene abundances for
each person. Since these features have varying relative importance
(Figure [1](#S3.F1 "Figure 1 ‣ 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")), we expected GBTs to
outperform DNNs on these tasks.

We sampled 2,574 healthy participants for which we measured, in addition
to the gut microbiome, a collection of different traits, including
important disease risk factors such as cholesterol levels and BMI.
Finding associations between these disease risk factors and the microbiome
composition is of

![Refer to caption](/html/1805.06440/assets/x3.png)


Figure 3: For each model type and trait, we took the 10 best performing models,
based on their validation performance, and calculated the average
variance of the predicted test samples, and plotted it against the
improvement in R2superscript𝑅2R^{2} obtained when training ensembles of these
models. Note that models that have a high variance in their prediction
benefit more from the use of ensembles. As expected, DNNs gain the
most from ensembling.

great scientific interest, and can raise novel hypotheses about the
role of the microbiome in disease. We tested 4 types of models: RLN,
GBT, DNN, and Linear Models (LM). The full list of hyperparameters,
the setting of the training of the models and the ensembles, as well
as the description of all the input features and the measured traits,
can be found in the supplementary material.

![Refer to caption](/html/1805.06440/assets/x4.png)


Figure 4: Ensembles of different predictors.

## 5 Results

![Refer to caption](/html/1805.06440/assets/x5.png)


Figure 5: Results of various ensembles that are each composed of different types
of models.

When running each model separately, GBTs achieve the best results
on all of the tested traits, but it is only significant on 333 of
them (Figure [2](#S3.F2 "Figure 2 ‣ 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")). DNNs achieve the worst results,
with 15%±1%plus-or-minuspercent15percent115\%\pm 1\% less explained variance than GBTs on average. RLNs
significantly and substantially improve this by a factor of 2.57±0.05plus-or-minus2.570.05\boldsymbol{2.57\pm 0.05},
and achieve only 2%±2%plus-or-minuspercent2percent22\%\pm 2\% less explained variance than GBTs on
average.

| Trait | RLN + GBT | LM + GBT | GBT | RLN | Max |
| --- | --- | --- | --- | --- | --- |
| Age | 31.9%±0.2%¯bold-¯plus-or-minuspercent31.9percent0.2\boldsymbol{\underline{31.9\%\pm 0.2\%}} | 30.5%±0.5%plus-or-minuspercent30.5percent0.530.5\%\pm 0.5\% | 30.9%±0.1%plus-or-minuspercent30.9percent0.130.9\%\pm 0.1\% | 29.1%±0.2%plus-or-minuspercent29.1percent0.229.1\%\pm 0.2\% | 31.9%percent31.931.9\% |
| HbA1c | 30.5%±0.2%plus-or-minuspercent30.5percent0.2\boldsymbol{30.5\%\pm 0.2\%} | 30.2%±0.3%plus-or-minuspercent30.2percent0.330.2\%\pm 0.3\% | 30.5%±0.04%plus-or-minuspercent30.5percent0.0430.5\%\pm 0.04\% | 28.4%±0.1%plus-or-minuspercent28.4percent0.128.4\%\pm 0.1\% | 30.5%percent30.530.5\% |
| HDL cholesterol | 28.8%±0.2%¯bold-¯plus-or-minuspercent28.8percent0.2\boldsymbol{\underline{28.8\%\pm 0.2\%}} | 27.7%±0.2%plus-or-minuspercent27.7percent0.227.7\%\pm 0.2\% | 27.2%±0.04%plus-or-minuspercent27.2percent0.0427.2\%\pm 0.04\% | 27.9%±0.1%plus-or-minuspercent27.9percent0.127.9\%\pm 0.1\% | 28.8%percent28.828.8\% |
| Median glucose | 26.2%±0.1%plus-or-minuspercent26.2percent0.1\boldsymbol{26.2\%\pm 0.1\%} | 26.1%±0.1%plus-or-minuspercent26.1percent0.126.1\%\pm 0.1\% | 25.2%±0.04%plus-or-minuspercent25.2percent0.0425.2\%\pm 0.04\% | 25.5%±0.1%plus-or-minuspercent25.5percent0.125.5\%\pm 0.1\% | 26.2%percent26.226.2\% |
| Max glucose | 25.2%±0.3%plus-or-minuspercent25.2percent0.3\boldsymbol{25.2\%\pm 0.3\%} | 25.0%±0.1%plus-or-minuspercent25.0percent0.125.0\%\pm 0.1\% | 24.6%±0.03%plus-or-minuspercent24.6percent0.0324.6\%\pm 0.03\% | 23.7%±0.4%plus-or-minuspercent23.7percent0.423.7\%\pm 0.4\% | 25.2%percent25.225.2\% |
| CRP | 24.0%±0.3%plus-or-minuspercent24.0percent0.3\boldsymbol{24.0\%\pm 0.3\%} | 23.7%±0.2%plus-or-minuspercent23.7percent0.223.7\%\pm 0.2\% | 22.4%±0.1%plus-or-minuspercent22.4percent0.122.4\%\pm 0.1\% | 22.8%±0.4%plus-or-minuspercent22.8percent0.422.8\%\pm 0.4\% | 24.0%percent24.024.0\% |
| Gender | 17.9%±0.4%plus-or-minuspercent17.9percent0.417.9\%\pm 0.4\% | 16.9%±0.6%plus-or-minuspercent16.9percent0.616.9\%\pm 0.6\% | 18.7%±0.03%plus-or-minuspercent18.7percent0.03\boldsymbol{18.7\%\pm 0.03\%} | 11.9%±0.4%plus-or-minuspercent11.9percent0.411.9\%\pm 0.4\% | 18.7%percent18.718.7\% |
| BMI | 17.6%±0.1%¯bold-¯plus-or-minuspercent17.6percent0.1\boldsymbol{\underline{17.6\%\pm 0.1\%}} | 17.2%±0.2%plus-or-minuspercent17.2percent0.217.2\%\pm 0.2\% | 16.9%±0.04%plus-or-minuspercent16.9percent0.0416.9\%\pm 0.04\% | 16.0%±0.1%plus-or-minuspercent16.0percent0.116.0\%\pm 0.1\% | 17.6%percent17.617.6\% |
| Cholesterol | 7.8%±0.3%plus-or-minuspercent7.8percent0.3\boldsymbol{7.8\%\pm 0.3\%} | 7.6%±0.3%plus-or-minuspercent7.6percent0.37.6\%\pm 0.3\% | 7.8%±0.1%plus-or-minuspercent7.8percent0.17.8\%\pm 0.1\% | 5.8%±0.2%plus-or-minuspercent5.8percent0.25.8\%\pm 0.2\% | 7.8%percent7.87.8\% |

Table 1: Explained variance (R2superscript𝑅2R^{2}) of various ensembles with different
types of models. Only the 4 ensembles that achieved the best results
are shown. The best result for each trait is highlighted, and underlined
if it outperforms significantly all other ensembles.

Constructing an ensemble of models is a powerful technique for improving
performance, especially for models which have high variance, like
neural networks in our task. As seen in Figure [3](#S4.F3 "Figure 3 ‣ 4 Experiments ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"),
the average variance of predictions of the top 10 models of RLN and
DNN is 1.3%±0.6%plus-or-minuspercent1.3percent0.61.3\%\pm 0.6\% and 14%±3%plus-or-minuspercent14percent314\%\pm 3\% respectively, while the
variance of predictions of the top 10 models of LM and GBT is only
0.13%±0.05%plus-or-minuspercent0.13percent0.050.13\%\pm 0.05\% and 0.26%±0.02%plus-or-minuspercent0.26percent0.020.26\%\pm 0.02\%, respectively. As expected,
the high variance of RLN and DNN models allows ensembles of these
models to improve the performance over a single model by 1.5%±0.7%plus-or-minuspercent1.5percent0.71.5\%\pm 0.7\%
and 4%±1%plus-or-minuspercent4percent14\%\pm 1\% respectively, while LM and GBT only improve by 0.2%±0.3%plus-or-minuspercent0.2percent0.30.2\%\pm 0.3\%
and 0.3%±0.4%plus-or-minuspercent0.3percent0.40.3\%\pm 0.4\%, respectively. Despite the improvement, DNN ensembles
still achieve the worst results on all of the traits except for Gender
and achieve results 9%±1%plus-or-minuspercent9percent19\%\pm 1\% lower than GBT ensembles (Figure
[4](#S4.F4 "Figure 4 ‣ 4 Experiments ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")). In comparison, this improvement
allows RLN ensembles to outperform GBT ensembles on HDL cholesterol,
Median glucose, and CRP, and to obtain results 8%±1%plus-or-minuspercent8percent18\%\pm 1\%
higher than DNN ensembles and only 1.4%±0.1%plus-or-minuspercent1.4percent0.11.4\%\pm 0.1\% lower than GBT
ensembles.

Using ensemble of different types of models could be even more effective
because their errors are likely to be even more uncorrelated than
ensembles from one type of model. Indeed, as shown in Figure [5](#S5.F5 "Figure 5 ‣ 5 Results ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"),
the best performance is obtained with an ensemble of RLN and GBT,
which achieves the best results on all traits except Gender,
and outperforms all other ensembles significantly on Age,
BMI, and HDL cholesterol (Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"))

## 6 Analysis

![Refer to caption](/html/1805.06440/assets/x6.png)


(a)

![Refer to caption](/html/1805.06440/assets/x7.png)


(b)

Figure 6: a) Each line represents an input feature in a model. The values of
each line are the absolute values of its outgoing weights, sorted
from greatest to smallest. Noticeably, only 12%percent1212\% of the input features
have any non-zero outgoing edge in the RLN model. b) The cumulative
distribution of non-zero outgoing weights for the input features for
different models. Remarkably, the distribution of non-zero weights
is quite similar for the two models.

We next sought to examine the effect that our new type of regularization
has on the learned networks. Strikingly, we found that RLNs are extremely
sparse, even compared to L1subscript𝐿1L\_{1} regulated networks. To demonstrate
this, we took the hyperparameter setting that achieved the best results
on the HbA1c task for the DNN and RLN models and trained
a single network on the entire dataset. Both models achieved their
best hyperparameter setting when using L1subscript𝐿1L\_{1} regularization. Remarkably,
82%percent8282\% of the input features in the RLN do not have any non-zero
outgoing edges, while all of the input features have at least one
non-zero outgoing edge in the DNN (Figure [6a](#S6.F6.sf1 "In Figure 6 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")).
A possible explanation could be that the RLN was simply trained using
a stronger regularization coefficients, and increasing the value of
λ𝜆\lambda for the DNN model would result in a similar behavior for
the DNN, but in fact the RLN was obtained with an average regularization
coefficient of θ=−6.6𝜃6.6\theta=-6.6 while the DNN model was trained using
a regularization coefficient of λ=−4.4𝜆4.4\lambda=-4.4. Despite this extreme
sparsity, the non zero weights are not particularly small and have
a similar distribution as the weights of the DNN (Figure [6b](#S6.F6.sf2 "In Figure 6 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")).

We suspect that the combination of a sparse network with large weights
allows RLNs to achieve their improved performance, as our dataset
includes features with varying relative importance. To show this,
we re-optimized the hyperparameters of the DNN and RLN models after
removing the covariates from the datasets. The covariates are very
important features (Figure [1](#S3.F1 "Figure 1 ‣ 3 Regularization Learning ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")),
and removing them would reduce the variability in relative importance.
As can be seen in Figure [7a](#S6.F7.sf1 "In Figure 7 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"), even without
the covariates, the RLN and GBT ensembles still achieve the best results
on 555 out of the 999 traits. However, this improvement is less
significant than when adding the covariates, where RLN and GBT ensembles
achieve the best results on 888 out of the 999 traits. RLNs still
significantly outperform DNNs, achieving explained variance higher
by 2%±1%plus-or-minuspercent2percent12\%\pm 1\%, but this is significantly smaller than the 9%±2%plus-or-minuspercent9percent29\%\pm 2\%
improvement obtained when adding the covariates (Figure [7b](#S6.F7.sf2 "In Figure 7 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")).
We speculate that this is because RLNs particularly shine when features
have very different relative importances.

![Refer to caption](/html/1805.06440/assets/x8.png)


(a)

![Refer to caption](/html/1805.06440/assets/x9.png)


(b)

Figure 7: a) Training our models without adding the covariates. b) The relative
improvement RLN achieves compared to DNN for different input features.

![Refer to caption](/html/1805.06440/assets/x10.png)


Figure 8: On the left axis, shown is the traversal of edges of the first layer
that finished the training with a non-zero weight in the w𝑤w, λ𝜆\lambda
space. Each colored line represents an edge, its color represents
its regularization, with yellow lines having strong regularization.
On the right axis, the black line plots the percent of zero weight
edges in the first layer during training.

To understand what causes this interesting structure, we next explored
how the weights in RLNs change during training. During training, each
edge performs a traversal in the w𝑤w, λ𝜆\lambda space. We expect
that when λ𝜆\lambda decreases and the regularization is relaxed,
the absolute value of w𝑤w should increase, and vice versa. In Figure
[8](#S6.F8 "Figure 8 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets"), we can see that 99.9%¯bold-¯percent99.9\boldsymbol{\underline{99.9\%}}
of the edges of the first layer finish the training with a zero value.
There are still 434434434 non-zero edges in the first layer due to the
large size of the network. This is not unique to the first layer,
and in fact, 99.8%¯bold-¯percent99.8\boldsymbol{\underline{99.8\%}} of the weights of
the entire network have a zero value by the end of the training. The
edges of the first layer that end up with a non-zero weight are decreasing
rapidly at the beginning of the training because of the regularization,
but during the first 10-20 epochs, the network quickly learns better
regularization coefficients for its edges. The regularization coefficients
are normalized after every update, hence by applying stronger regularization
on some edges, the network is allowed to have a more relaxed regularization
on other edges and consequently a larger weight. By epoch 20, the
edges of the first layer that end up with a non-zero weight have an
average regularization coefficient of −9.49.4-9.4, which is significantly
smaller than their initial value θ=−6.6𝜃6.6\theta=-6.6. These low values
pose effectively no regularization, and their weights are updated
primarily to minimize the empirical loss component of the loss function,
ℒℒ\mathcal{L}.

Finally, we reasoned that since RLNs assign non-zero weights to a
relatively small number of inputs, they may be used to provide insights
into the inputs that the model found to be more important for generating
its predictions using Garson’s algorithm [[10](#bib.bib10)].
There has been important progress in recent years in sample-aware
model interpretability techniques in DNNs [[28](#bib.bib28), [31](#bib.bib31)],
but tools to produce sample-agnostic model interpretations are lacking
[[15](#bib.bib15)].444The sparsity of RLNs could be beneficial for sample-aware model interpretability
techniques such as [[28](#bib.bib28), [31](#bib.bib31)]. This was not examined
in this paper. Model interpretability is particularly important in our problem for
obtaining insights into which bacterial species contribute to predicting
each trait.

Evaluating feature importance is difficult, especially in domains
in which little is known such as the gut microbiome. One possibility
is to examine the information it supplies. In Figure [9a](#S6.F9.sf1 "In Figure 9 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")
we show the feature importance achieved through this technique using
RLNs and DNNs. While the importance in DNNs is almost constant and
does not give any meaningful information about the specific importance
of the features, the importance in RLNs is much more meaningful, with
entropy of the 4.64.64.6 bits for the RLN importance, compared to more
than twice for the DNN importance, 9.59.59.5 bits.

Another possibility is to evaluate its consistency across different
instantiations of the model. We expect that a good feature importance
technique will give similar importance distributions regardless of
instantiation. We trained 10 instantiations for each model and phenotype
and evaluated their feature importance distributions, for which we
calculated the Jensen-Shannon divergence. In Figure [9b](#S6.F9.sf2 "In Figure 9 ‣ 6 Analysis ‣ Regularization Learning Networks: Deep Learning for Tabular Datasets")
we see that RLNs have divergence values 48%±1%plus-or-minuspercent48percent148\%\pm 1\% and 54%±2%plus-or-minuspercent54percent254\%\pm 2\%
lower than DNNs and LMs respectively. This is an indication that Garson’s
algorithm results in meaningful feature importances in RLNs. We list
of the 5 most important bacterial species for different traits in
the supplementary material.

![Refer to caption](/html/1805.06440/assets/x11.png)


(a)

![Refer to caption](/html/1805.06440/assets/x12.png)


(b)

Figure 9: a) The input features, sorted by their importance, in a DNN and RLN
models. b) The Jensen-Shannon divergence between the feature importance
of different instantiations of a model.

## 7 Conclusion

In this paper, we explore the learning of datasets with non-distributed
representation, such as tabular datasets. We hypothesize that modular
regularization could boost the performance of DNNs on such tabular
datasets. We introduce the Counterfactual Loss, *ℒC​Fsubscriptℒ𝐶𝐹\mathcal{L}\_{CF}*,
and Regularization Learning Networks (RLNs) which use the
Counterfactual Loss to tune its regularization hyperparameters efficiently
during learning together with the learning of the weights of the network.

We test our method on the task of predicting human traits from covariates
and microbiome data and show that RLNs significantly and substantially
improve the performance over classical DNNs, achieving an increased
explained variance by a factor of 2.75±0.05plus-or-minus2.750.052.75\pm 0.05 and comparable results
with GBTs. The use of ensembles further improves the performance of
RLNs, and ensembles of RLN and GBT achieve the best results on all
but one of the traits, and outperform significantly any other ensemble
not incorporating RLNs on 333 of the traits.

We further explore RLN structure and dynamics and show that RLNs learn
extremely sparse networks, eliminating 99.8%percent99.899.8\% of the network edges
and 82%percent8282\% of the input features. In our setting, this was achieved
in the first 10-20 epochs of training, in which the network learns
its regularization. Because of the modularity of the regularization,
the remaining edges are virtually not regulated at all, achieving
a similar distribution to a DNN. The modular structure of the network
is especially beneficial for datasets with high variability in the
relative importance of the input features, where RLNs particularly
shine compared to DNNs. The sparse structure of RLNs lends itself
naturally to model interpretability, which gives meaningful insights
into the relation between features and the labels, and may itself
serve as a feature selection technique that can have many uses on
its own [[13](#bib.bib13)].

Besides improving performance on tabular datasets, another important
application of RLNs could be learning tasks where there are multiple
data sources, one that includes features with high variability in
the relative importance, and one which does not. To illustrate this
point, consider the problem of detecting pathologies from medical
imaging. DNNs achieve impressive results on this task [[32](#bib.bib32)],
but in real life, the imaging is usually accompanied by a great deal
of tabular metadata in the form of the electronic health records of
the patient. We would like to use both datasets for prediction, but
different models achieve the best results on each part of the data.
Currently, there is no simple way to jointly train and combine the
models. Having a DNN architecture such as RLN that performs well on
tabular data will thus allow us to jointly train a network on both
of the datasets natively, and may improve the overall performance.

### Acknowledgments

We would like to thank Ron Sender, Eran Kotler, Smadar Shilo, Nitzan
Artzi, Daniel Greenfeld, Gal Yona, Tomer Levy, Dror Kaufmann, Aviv
Netanyahu, Hagai Rossman, Yochai Edlitz, Amir Globerson and Uri Shalit
for useful discussions.

## References

* [1]

  David Beam and Mark Schramm.
  Rossmann Store Sales.
  2015.
* [2]

  Kamil Belkhayat, Abou Omar, Gino Bruner, Yuyi Wang, and Roger Wattenhofer.
  XGBoost and LGBM for Porto Seguro’s Kaggle challenge: A comparison
  Semester Project.
  2018.
* [3]

  Yoshua Bengio.
  Gradient-Based Optimization of 1 Introduction.
  pages 1–18, 1999.
* [4]

  Yoshua Bengio, Aaron Courville, and Pascal Vincent.
  Representation Learning: A Review and New Perspectives.
* [5]

  Yoshua Bengio and Yann LeCun.
  Scaling Learning Algorithms towards AI.
  2007.
* [6]

  James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs
  Kégl.
  Algorithms for Hyper-Parameter Optimization.
  Advances in Neural Information Processing Systems (NIPS), pages
  2546–2554, 2011.
* [7]

  Hengxing Cai, Runxing Zhong, Chaohe Wang, Kejie Zhou, Hongyun Lee, Renxin
  Zhong, Yao Zhou, Da Li, Nan Jiang, Xu Cheng, and Jiawei Shen.
  KDD CUP 2018 Travel Time Prediction.
* [8]

  Tianqi Chen and Carlos Guestrin.
  XGBoost: A Scalable Tree Boosting System.
* [9]

  Chung-Cheng Chiu, Tara N Sainath, Yonghui Wu, Rohit Prabhavalkar, Patrick
  Nguyen, Zhifeng Chen, Anjuli Kannan, Ron J Weiss, Kanishka Rao, Ekaterina
  Gonina, Navdeep Jaitly, Bo Li, Jan Chorowski, and Michiel Bacchiani Google.
  State-Of-The-Art Speech Recognition with Sequence-To-Sequence
  Models.
* [10]

  G D Garson.
  Interpreting neural network connection weights.
  AI Expert, 6(4):47–51, apr 1991.
* [11]

  Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
  Deep Learning.
  MIT Press, 2016.
  <http://www.deeplearningbook.org>.
* [12]

  Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
  Explaining And Harnessing Adversarial Examples.
* [13]

  Bryce Goodman and Seth Flaxman.
  European Union regulations on algorithmic decision-making and a "
  right to explanation ".
* [14]

  GE HINTON, JL MCCLELLAND, and DE RUMELHART.
  Distributed representations.
* [15]

  Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim.
  Evaluating Feature Importance Estimates.
* [16]

  Yide Huang.
  Highway Tollgates Traffic Flow Prediction Task 1. Travel Time
  Prediction.
* [17]

  Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown.
  Sequential Model - Based Optimization for General Algorithm
  Configuration.
  Lecture Notes in Computer Science, 5:507–223, 2011.
* [18]

  Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng
  Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado,
  Macduff Hughes, and Jeffrey Dean.
  Google’s Multilingual Neural Machine Translation System: Enabling
  Zero-Shot Translation.
  2016.
* [19]

  Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
  ImageNet Classification with Deep Convolutional Neural Networks.
* [20]

  Yann LeCun.
  The mnist database of handwritten digits.
  http://yann. lecun. com/exdb/mnist/.
* [21]

  Jonathan Lorraine and David Duvenaud.
  Stochastic Hyperparameter Optimization through Hypernetworks.
  2018.
* [22]

  Jelena Luketina, Jelena Luketina@aalto Fi, Mathias Berglund,
  Mathias Berglund@aalto Fi, Klaus Greff, Klaus@idsia Ch, Tapani Raiko, and
  Tapani Raiko@aalto Fi.
  Scalable Gradient-Based Tuning of Continuous Regularization
  Hyperparameters.
* [23]

  Dougal Maclaurin, David Duvenaud, and Ryan P Adams.
  Gradient-based Hyperparameter Optimization through Reversible
  Learning.
* [24]

  Riccardo Miotto, Li Li, Brian A Kidd, and Joel T Dudley.
  Deep Patient: An Unsupervised Representation to Predict the Future
  of Patients from the Electronic Health Records.
  Nature Publishing Group, 2016.
* [25]

  Nicolas Papernot, Patrick Mcdaniel, Somesh Jha, Matt Fredrikson, Z Berkay
  Celik, and Ananthram Swami.
  The Limitations of Deep Learning in Adversarial Settings.
* [26]

  Alvin Rajkomar, Eyal Oren, Kai Chen, Andrew M Dai, Nissan Hajaj, Michaela
  Hardt, Peter J Liu, Xiaobing Liu, Jake Marcus, Mimi Sun, Patrik Sundberg,
  Hector Yee, Kun Zhang, Yi Zhang, Gerardo Flores, Gavin E Duggan, Jamie
  Irvine, Quoc Le, Kurt Litsch, Alexander Mossin, Justin Tansuwan, De Wang,
  James Wexler, Jimbo Wilson, Dana Ludwig, Samuel L Volchenboum, Katherine
  Chou, Michael Pearson, Srinivasan Madabushi, Nigam H Shah, Atul J Butte,
  Michael D Howell, Claire Cui, Greg S Corrado, and Jeffrey Dean.
  Scalable and accurate deep learning with electronic health records.
  npj Digital Medicine, 1, 2018.
* [27]

  Vlad Sandulescu, Adform Copenhagen, and Denmark Mihai Chiru.
  Predicting the future relevance of research institutions - The
  winning solution of the KDD Cup 2016.
* [28]

  Avanti Shrikumar, Peyton Greenside, and Anna Y Shcherbina.
  Not Just A Black Box: Learning Important Features Through
  Propagating Activation Differences.
  (3).
* [29]

  Leslie N Smith.
  A disciplined approach to neural network hyper-parameters: Part 1 -
  learning rate, batch size, momentum, and weight decay.
* [30]

  Jasper Snoek, Hugo Larochelle, and Ryan P. Adams.
  Practical Bayesian Optimization of Machine Learning Algorithms.
  pages 1–12, 2012.
* [31]

  Mukund Sundararajan, Ankur Taly, and Qiqi Yan.
  Gradients of Counterfactuals.
* [32]

  Kenji Suzuki.
  Overview of deep learning in medical imaging.
  Radiological Physics and Technology, 10.

[◄](/html/1805.06439)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1805.06440)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1805.06440)
[View original  
on arXiv](https://arxiv.org/abs/1805.06440)[►](/html/1805.06441)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sun Mar 3 01:28:41 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
