---
arxiv: '2407.03257'
authors:
- Han-Jia Ye
- Huai-Hong Yin
- De-Chuan Zhan
- Wei-Lun Chao
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two
  Decades Later'
url: http://arxiv.org/abs/2407.03257v2
year: 2024
---

[2407.03257] Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later














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



# Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later

Han-Jia Ye      Huai-Hong Yin       De-Chuan Zhan
  
School of Artificial Intelligence, Nanjing University, China
  
National Key Laboratory for Novel Software Technology,
Nanjing University
  
{yehj,yinhh,zhandc}@lamda.nju.edu.cn

###### Abstract

The growing success of deep learning in various domains has prompted investigations into its application to tabular data, where deep models have shown promising results compared to traditional tree-based methods. In this paper, we revisit Neighborhood Component Analysis (NCA), a classic tabular prediction method introduced in 2004, designed to learn a linear projection that captures semantic similarities between instances.
We find that minor modifications, such as adjustments to the learning objectives and the integration of deep learning architectures, significantly enhance NCA’s performance, enabling it to surpass most modern deep tabular models. Additionally, we introduce a stochastic neighbor sampling strategy that improves both the efficiency and predictive accuracy of our proposed ModernNCA— sampling only a subset of neighbors during training, while utilizing the entire neighborhood during inference.
Extensive experiments demonstrate that our ModernNCA achieves state-of-the-art results in both classification and regression tasks across various tabular datasets, outperforming both tree-based and other deep tabular models, while also reducing training time and model size.

## 1 Introduction

Tabular data, characterized by its highly structured format of rows and columns representing individual examples and specific features, is prevalent in domains such as healthcare [[26](#bib.bib26)] and e-commerce [[39](#bib.bib39)]. Inspired by the success of deep neural networks in fields like vision and language [[50](#bib.bib50), [57](#bib.bib57), [17](#bib.bib17)], numerous deep models have been developed to handle tabular data, capturing complex feature interactions and achieving promising results [[13](#bib.bib13), [24](#bib.bib24), [42](#bib.bib42), [4](#bib.bib4), [20](#bib.bib20), [31](#bib.bib31), [10](#bib.bib10), [11](#bib.bib11), [27](#bib.bib27)].

In contrast to parametric approaches like MLPs that make predictions directly, non-parametric methods such as nearest neighbors utilize the entire training dataset and leverage relationships between instances for tabular prediction [[38](#bib.bib38)]. Metric learning, rather than relying on raw features, aims to learn a distance metric corresponding to an embedding space where semantically similar instances are close together and dissimilar ones are far apart [[62](#bib.bib62), [14](#bib.bib14), [35](#bib.bib35), [7](#bib.bib7)]. These improved distance metrics facilitate the discerning process of non-parametric/neighborhood-based methods and are particularly beneficial when dealing with a large number of classes [[61](#bib.bib61)].
classes [[61](#bib.bib61)].

Although metric learning approaches perform well compared to vanilla nearest neighbor methods, they often struggle to match the predictive power of tree-based nonlinear approaches like Gradient Boosting Decision Trees (GBDT) [[12](#bib.bib12), [43](#bib.bib43)] on tabular data. However, the recent successes of deep metric learning in diverse domains such as image recognition[[47](#bib.bib47), [51](#bib.bib51), [52](#bib.bib52), [34](#bib.bib34)], person re-identification [[69](#bib.bib69), [64](#bib.bib64)], and recommendation systems [[28](#bib.bib28), [60](#bib.bib60)] suggest its untapped potential for tabular datasets. Moreover, specially designed MLP architectures and weight regularization strategies have enabled simple parametric models like MLPs to excel on many tabular datasets [[20](#bib.bib20), [30](#bib.bib30)], indicating that similar improvements could also benefit neighborhood-based methods.

![Refer to caption](/html/2407.03257/assets/x1.png)

(a) Classification

![Refer to caption](/html/2407.03257/assets/x2.png)

(b) Regression

Figure 1: Performance-Efficiency-Size comparison between ModernNCA on classification (a) and regression (b) datasets.
Representative tabular prediction methods, including the classical methods (in green), the parametric deep methods (in blue), and the non-parametric/neighborhood-based deep methods (in red), are investigated, based on their records over all datasets in [Table 1](#S5.T1 "Table 1 ‣ 5.1 Setups ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") and [Table 2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). The average rank among these eight methods is used as the performance measure. We calculate the average training time (in seconds) and the size of the model (denoted by the radius of the circles, where the larger the circle, the bigger the model). ModernNCA achieves high training speed compared to other deep tabular models and has a lower model size.
TabCon is another proposed baseline. ModernNCA outperforms other methods in two cases.

This leads us to a pivotal question: “Can we improve metric learning with modern deep learning techniques to achieve competitive performance in tabular prediction tasks?”

In this paper, we revisit the classical metric learning approach, Neighborhood Component Analysis (NCA), originally proposed by Goldberger et al. [[19](#bib.bib19)] in 2004. NCA optimizes the prediction accuracy of the target instance based on its neighbors in a linearly projected space.
With simple modifications to NCA, our enhanced method, ModernNCA, effectively addresses tabular prediction tasks.

In particular, our ModernNCA introduces a learning objective tailored for both classification and regression tasks. We further explore a deep neural network architecture that projects input instances into a latent embedding space in a nonlinear manner. Given that NCA’s prediction relies on the set of neighbors, which increases the batch size and computational burden during training, we employ a Stochastic Neighborhood Sampling (SNS) strategy.
SNS randomly selects a subset of the training set as neighbors during training while utilizing the entire training set in the test phase. This strategy improves not only the training efficiency but also the generalization capability of the model. Our experiments validate that ModernNCA outperforms most existing tabular prediction methods, including both tree-based and deep learning approaches, across various datasets. [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") demonstrates that ModernNCA balances training efficiency (with lower training time compared to other deep tabular models), generalization ability (with higher average accuracy), and model size.
The contributions of our paper are:

* •

  We revisit the classical metric learning approach NCA and explore ways to improve the model’s performance with modern deep learning techniques.
* •

  Our proposed ModernNCA is capable of both classification and regression tasks. An additional sampling strategy improves the training efficiency and the generalization ability of the model.
* •

  Extensive experiments across multiple datasets demonstrate that ModernNCA serves as a strong deep tabular baseline for tabular prediction tasks.

## 2 Related Work

Learning with Tabular Data.
Tabular data is a common format across various applications such as click-through rate prediction [[45](#bib.bib45)] and time-series forecasting [[1](#bib.bib1)]. Tree-based methods like XGBoost [[12](#bib.bib12)], LightGBM [[32](#bib.bib32)], and CatBoost [[43](#bib.bib43)] have proven effective at capturing feature interactions and are widely used in real-world applications.
Since changes in model families and hyper-parameters can significantly affect a model’s generalization ability [[16](#bib.bib16)], some strategies focus on automated model selection and hyper-parameter tuning to optimize performance across different tasks [[18](#bib.bib18), [25](#bib.bib25)].

Deep Tabular Data Learning.
Recognizing the ability of deep neural networks to learn feature representations from raw data and make nonlinear predictions, recent methods have applied deep learning techniques to tabular models [[13](#bib.bib13), [24](#bib.bib24), [42](#bib.bib42), [9](#bib.bib9), [4](#bib.bib4), [30](#bib.bib30), [31](#bib.bib31), [11](#bib.bib11)]. For instance, deep architectures such as residual networks and transformers have been adapted for tabular prediction [[20](#bib.bib20), [27](#bib.bib27)]. Moreover, data augmentation strategies have been introduced to mitigate overfitting in deep models [[54](#bib.bib54), [6](#bib.bib6), [46](#bib.bib46)].
Deep tabular models have demonstrated competitive performance across a wide range of applications, and pre-trained deep tabular predictors have shown the ability to generalize to downstream tabular tasks [[36](#bib.bib36), [27](#bib.bib27), [48](#bib.bib48), [41](#bib.bib41), [71](#bib.bib71), [70](#bib.bib70), [67](#bib.bib67)]. However, researchers have observed that deep models still face challenges in capturing high-order feature interactions as effectively as tree-based models [[23](#bib.bib23), [37](#bib.bib37)].

Metric Learning.
Methods like K Nearest Neighbors (KNN) make predictions based on the entire training set. The choice of distance metric critically determines the set of neighbors and affects the discriminative ability of the model [[62](#bib.bib62), [19](#bib.bib19)].
Metric learning learns a projection to bring similar instances closer together and push dissimilar ones farther apart, which further yields improved classification and regression results of KNN [[14](#bib.bib14), [61](#bib.bib61), [35](#bib.bib35), [7](#bib.bib7)].
In addition to linear metrics, nonlinear metrics are often implemented using multiple metrics [[44](#bib.bib44), [65](#bib.bib65), [66](#bib.bib66)], local composition [[59](#bib.bib59), [49](#bib.bib49), [3](#bib.bib3), [40](#bib.bib40)], or kernel methods [[33](#bib.bib33), [63](#bib.bib63)]. Initially focused on tabular data, metric learning has become a valuable tool when integrated with deep learning techniques across various domains such as image recognition [[47](#bib.bib47), [51](#bib.bib51), [52](#bib.bib52), [34](#bib.bib34)], person re-identification [[69](#bib.bib69), [64](#bib.bib64)], and recommendation systems [[28](#bib.bib28), [60](#bib.bib60)].
Recently, TabR [[22](#bib.bib22)] incorporated metric learning with transformers [[57](#bib.bib57)] to enhance tabular predictions through neighborhood-based methods. Despite promising results, the extensive computational burden of neighborhood selection and the complicated architecture limit TabR’s practicality. Based on the classical method NCA [[19](#bib.bib19)], we propose a simple deep tabular baseline that maintains efficient training speeds without sacrificing performance.

## 3 Preliminary

In this section, we first introduce the task learning with tabular data. We then provide a brief overview of metric learning approaches for tabular data, specifically on the NCA method.

### 3.1 Learning with Tabular Data

A labeled tabular dataset is formatted as N𝑁N examples (rows in the table) and d𝑑d features/attributes (columns in the table). An instance 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} is depicted by its d𝑑d feature values. There are two kinds of features: the numerical (continuous) ones and categorical (discrete) ones.
Given xi,jsubscript𝑥

𝑖𝑗x\_{i,j} as the j𝑗j-th feature of instance 𝒙isubscript𝒙𝑖{\bm{x}}\_{i}, we use xi,jnum∈ℝsuperscriptsubscript𝑥

𝑖𝑗numℝx\_{i,j}^{\textit{\rm num}}\in\mathbb{R} and 𝒙i,jcatsuperscriptsubscript𝒙

𝑖𝑗cat{\bm{x}}\_{i,j}^{\textit{\rm cat}} to denote numerical (*e.g*., the height of a person) and categorical (*e.g*., the gender of a person) feature values of an instance, respectively.
The categorical features are usually transformed in a one-hot manner, *i.e*., 𝒙i,jcat∈{0,1}Kjsuperscriptsubscript𝒙

𝑖𝑗catsuperscript01subscript𝐾𝑗{\bm{x}}\_{i,j}^{\textit{\rm cat}}\in\{0,1\}^{K\_{j}} and the index of value 1 indicates one of the Kjsubscript𝐾𝑗K\_{j} categorical values. We assume the instance 𝒙i∈ℝdsubscript𝒙𝑖superscriptℝ𝑑{\bm{x}}\_{i}\in\mathbb{R}^{d} w.l.o.g. and will explore other encoding strategies later.
Each instance is associated with a label yisubscript𝑦𝑖y\_{i}, where yi∈[C]={1,…,C}subscript𝑦𝑖delimited-[]𝐶1…𝐶y\_{i}\in[C]=\{1,\ldots,C\} in a multi-class classification task and yi∈ℝsubscript𝑦𝑖ℝy\_{i}\in\mathbb{R} in a regression task.

Given a tabular dataset 𝒟={(𝒙i,yi)}i=1N𝒟superscriptsubscriptsubscript𝒙𝑖subscript𝑦𝑖𝑖1𝑁\mathcal{D}=\{({\bm{x}}\_{i},y\_{i})\}\_{i=1}^{N}, we aim to learn a model f𝑓f on 𝒟𝒟\mathcal{D} that maps 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} to its label yisubscript𝑦𝑖y\_{i}. We measure the quality of f𝑓f by the joint likelihood on 𝒟𝒟\mathcal{D}, *i.e*., maxf​∏(𝒙i,yi)∈𝒟Pr⁡(yi∣f​(𝒙i))subscript𝑓subscriptproductsubscript𝒙𝑖subscript𝑦𝑖𝒟Prconditionalsubscript𝑦𝑖𝑓subscript𝒙𝑖\max\_{f}\prod\_{({\bm{x}}\_{i},y\_{i})\in\mathcal{D}}\Pr(y\_{i}\mid f({\bm{x}}\_{i})). The objective could be reformulated in the form of negative log-likelihood of predicted labels:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minf​∑(𝒙i,yi)∈𝒟−log⁡Pr⁡(yi∣f​(𝒙i))=∑(𝒙i,yi)∈𝒟ℓ​(y,y^i=f​(𝒙i)).subscript𝑓subscriptsubscript𝒙𝑖subscript𝑦𝑖𝒟Prconditionalsubscript𝑦𝑖𝑓subscript𝒙𝑖subscriptsubscript𝒙𝑖subscript𝑦𝑖𝒟ℓ  𝑦subscript^𝑦𝑖 𝑓subscript𝒙𝑖\min\_{f}\;\sum\_{({\bm{x}}\_{i},y\_{i})\in\mathcal{D}}-\log\Pr(y\_{i}\mid f({\bm{x}}\_{i}))=\sum\_{({\bm{x}}\_{i},y\_{i})\in\mathcal{D}}\ell(y,\;\hat{y}\_{i}=f({\bm{x}}\_{i}))\;. |  | (1) |

The discrepancy between the predicted label y^isubscript^𝑦𝑖\hat{y}\_{i} and the true label yisubscript𝑦𝑖y\_{i} could be measured by the loss ℓ​(⋅,⋅)ℓ⋅⋅\ell(\cdot,\cdot), *e.g*., cross-entropy. We expect the learned model f𝑓f is able to extend its ability to unseen instances sampled from the same distribution as 𝒟𝒟\mathcal{D}. f𝑓f could be implemented with classical methods such as SVM, MLP, or tree-based approaches.

K Nearest Neighbors (KNN) is one of the most representative non-parametric tabular models for classification and regression — making predictions based on the labels of the nearest neighbors [[8](#bib.bib8), [38](#bib.bib38)]. In other words, the model f𝑓f makes predictions via f​(𝒙i,𝒟)𝑓subscript𝒙𝑖𝒟f({\bm{x}}\_{i},\mathcal{D}).
Given an instance 𝒙isubscript𝒙𝑖{\bm{x}}\_{i}, KNN calculates the distance between 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} and other instances in 𝒟𝒟\mathcal{D}. Assume the K𝐾K nearest neighbors are 𝒩​(𝒙i;𝒟)={(𝒙1,y1),…,(𝒙K,yK)}𝒩

subscript𝒙𝑖𝒟subscript𝒙1subscript𝑦1…subscript𝒙𝐾subscript𝑦𝐾\mathcal{N}({\bm{x}}\_{i};\mathcal{D})=\{({\bm{x}}\_{1},y\_{1}),\ldots,({\bm{x}}\_{K},y\_{K})\}.
then, the label yisubscript𝑦𝑖y\_{i} of 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} is predicted based on those labels in the neighbor set 𝒩​(𝒙i;𝒟)𝒩

subscript𝒙𝑖𝒟\mathcal{N}({\bm{x}}\_{i};\mathcal{D}). For classification task y^isubscript^𝑦𝑖\hat{y}\_{i} is the majority voting of labels in 𝒩​(𝒙i;𝒟)𝒩

subscript𝒙𝑖𝒟\mathcal{N}({\bm{x}}\_{i};\mathcal{D}) while is the average of those labels in regression tasks.

### 3.2 Metric Learning on Tabular Data

The distance dist​(𝒙i,𝒙j)distsubscript𝒙𝑖subscript𝒙𝑗{\rm dist}({\bm{x}}\_{i},{\bm{x}}\_{j}) in KNN determines the set of nearest neighbors 𝒩​(𝒙i;𝒟)𝒩

subscript𝒙𝑖𝒟\mathcal{N}({\bm{x}}\_{i};\mathcal{D}), which is one of its key factors. Instead of using the Euclidean distance, the Mahalanobis distance between a pair (𝒙i,𝒙j)subscript𝒙𝑖subscript𝒙𝑗({\bm{x}}\_{i},{\bm{x}}\_{j}) with metric 𝑴𝑴\bm{M} is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | dist𝑴​(𝒙i,𝒙j)=(𝒙i−𝒙j)⊤​𝑴​(𝒙i−𝒙j).subscriptdist𝑴subscript𝒙𝑖subscript𝒙𝑗superscriptsubscript𝒙𝑖subscript𝒙𝑗top𝑴subscript𝒙𝑖subscript𝒙𝑗{\rm dist}\_{\bm{M}}({\bm{x}}\_{i},{\bm{x}}\_{j})=\sqrt{({\bm{x}}\_{i}-{\bm{x}}\_{j})^{\top}\bm{M}({\bm{x}}\_{i}-{\bm{x}}\_{j})}. |  | (2) |

The matrix 𝑴𝑴\bm{M} is assumed to be a semi-definite one.
dist𝑴2subscriptsuperscriptdist2𝑴{\rm dist}^{2}\_{\bm{M}} becomes the Euclidean distance (denoted by dist​(𝒙i,𝒙j)distsubscript𝒙𝑖subscript𝒙𝑗{\rm dist}({\bm{x}}\_{i},{\bm{x}}\_{j})) when we set the metric 𝑴𝑴\bm{M} as the identity matrix.
𝑴𝑴\bm{M} can be decomposed into projections 𝑴=𝑳​𝑳⊤𝑴𝑳superscript𝑳top\bm{M}=\bm{L}\bm{L}^{\top} with 𝑳∈ℝd×d′𝑳superscriptℝ𝑑superscript𝑑′\bm{L}\in\mathbb{R}^{d\times d^{\prime}} so that the Mahalanobis distance becomes the Euclidean distance in the space projected by 𝑳𝑳\bm{L}.
Most distance metric learning methods learn the metric 𝑴𝑴\bm{M} or 𝑳𝑳\bm{L} in two stages. Side information in the form of pairs or triplets is constructed first based on labels, guiding the similarity and comparison relationship between instances [[62](#bib.bib62), [14](#bib.bib14), [61](#bib.bib61), [7](#bib.bib7)]. With the help of the learned metric, KNN then makes more accurate final predictions.

Neighborhood Component Analysis (NCA).
NCA is a one-stage approach that directly links the projection 𝑳𝑳\bm{L} with the posterior probability that an instance 𝒙𝒙{\bm{x}} is classified as the class y𝑦y [[19](#bib.bib19)]:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Pr⁡(y^=y∣𝒙,𝒟,𝑳)=∑𝒙i∈𝒟∧yi=yexp⁡(−dist2⁡(𝑳⊤​𝒙,𝑳⊤​𝒙i))∑𝒙j∈𝒟,𝒙j≠𝒙iexp⁡(−dist2⁡(𝑳⊤​𝒙,𝑳⊤​𝒙j)).Pr^𝑦conditional𝑦  𝒙𝒟𝑳subscriptsubscript𝒙𝑖𝒟subscript𝑦𝑖𝑦superscriptdist2superscript𝑳top𝒙superscript𝑳topsubscript𝒙𝑖subscriptformulae-sequencesubscript𝒙𝑗𝒟subscript𝒙𝑗subscript𝒙𝑖superscriptdist2superscript𝑳top𝒙superscript𝑳topsubscript𝒙𝑗\Pr(\hat{y}=y\mid{\bm{x}},\mathcal{D},\bm{L})=\sum\_{{{\bm{x}}\_{i}\in\mathcal{D}\land y\_{i}=y}}\frac{\exp\left(-\operatorname{dist}^{2}(\bm{L}^{\top}{\bm{x}},\;\bm{L}^{\top}{\bm{x}}\_{i})\right)}{\sum\_{{\bm{x}}\_{j}\in\mathcal{D},{\bm{x}}\_{j}\neq{\bm{x}}\_{i}}\exp\left(-\operatorname{dist}^{2}(\bm{L}^{\top}{\bm{x}},\;\bm{L}^{\top}{\bm{x}}\_{j})\right)}\;. |  | (3) |

Therefore, the posterior that an instance 𝒙𝒙{\bm{x}} belongs to the class y𝑦y depends on its similarity (measured by the negative squared Euclidean distance in the space projected by 𝑳𝑳\bm{L}) between its neighbors from class y𝑦y in 𝒟𝒟\mathcal{D}.
Instead of considering all instances in the neighborhood equally, the prediction in [Equation 3](#S3.E3 "Equation 3 ‣ 3.2 Metric Learning on Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") mimics a soft version of KNN, where all instances in the training set are weighted (nearer neighbors have more weight) for the final decision. Similar strategies have also been explored in domains like few-shot learning [[58](#bib.bib58)].

## 4 ModernNCA

In this section, we describe the learning objective, the architecture, and the training strategy of our improved version of NCA, *i.e*., ModernNCA (abbreviated as M-NCA).

Learning Objective.
We use a nonlinear transformation ϕitalic-ϕ\phi, which maps the input 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} into a latent space with dimensionality d′superscript𝑑′d^{\prime}. Assume the label yisubscript𝑦𝑖y\_{i} is a continuous value in a regression task and in a one-hot form in a classification task. Then we modify the [Equation 3](#S3.E3 "Equation 3 ‣ 3.2 Metric Learning on Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") as

|  |  |  |  |
| --- | --- | --- | --- |
|  | y^i=∑𝒙i∈𝒟exp⁡(−dist⁡(ϕ​(𝒙),ϕ​(𝒙i)))∑𝒙j∈𝒟,𝒙j≠𝒙iexp⁡(−dist⁡(ϕ​(𝒙),ϕ​(𝒙j)))​yi.subscript^𝑦𝑖subscriptsubscript𝒙𝑖𝒟distitalic-ϕ𝒙italic-ϕsubscript𝒙𝑖subscriptformulae-sequencesubscript𝒙𝑗𝒟subscript𝒙𝑗subscript𝒙𝑖distitalic-ϕ𝒙italic-ϕsubscript𝒙𝑗subscript𝑦𝑖\hat{y}\_{i}=\sum\_{{{\bm{x}}\_{i}\in\mathcal{D}}}\frac{\exp\left(-\operatorname{dist}(\phi({\bm{x}}),\;\phi({\bm{x}}\_{i}))\right)}{\sum\_{{\bm{x}}\_{j}\in\mathcal{D},{\bm{x}}\_{j}\neq{\bm{x}}\_{i}}\exp\left(-\operatorname{dist}(\phi({\bm{x}}),\;\phi({\bm{x}}\_{j}))\right)}y\_{i}\;. |  | (4) |

[Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") makes predictions that similar instances (based on their distance in the embedding space mapped by ϕitalic-ϕ\phi) will have close outputs.
In the classification scenario, [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") generalizes [Equation 3](#S3.E3 "Equation 3 ‣ 3.2 Metric Learning on Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") and predicts the label of the target instance with a weighted average of its neighbors in each one of the C𝐶C classes. y^isubscript^𝑦𝑖\hat{y}\_{i} is a probability vector which can be used as {Pr⁡(y^=y∣𝒙,𝒟,ϕ)}y∈[C]subscriptPr^𝑦conditional𝑦

𝒙𝒟italic-ϕ𝑦delimited-[]𝐶\{\Pr(\hat{y}=y\mid{\bm{x}},\mathcal{D},\phi)\}\_{y\in[C]} in this case.
In the regression scenario, the prediction of an instance is the weighted sum over the scalar labels of its neighborhood.
By integrating
[Equation 3](#S3.E3 "Equation 3 ‣ 3.2 Metric Learning on Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") with [Equation 1](#S3.E1 "Equation 1 ‣ 3.1 Learning with Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), we set the loss ℓℓ\ell in [Equation 1](#S3.E1 "Equation 1 ‣ 3.1 Learning with Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") as the negative log-likelihood loss for classification, while the mean square error for regression problems.
By minimizing [Equation 1](#S3.E1 "Equation 1 ‣ 3.1 Learning with Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), the embedding ϕitalic-ϕ\phi pulls the same-class instances in the training set 𝒟𝒟\mathcal{D} close and pushes other instances away. In the test phase, neighbors of the test instance in the whole training set 𝒟𝒟\mathcal{D} are utilized to make predictions.

Architectures.
We set the mapping ϕitalic-ϕ\phi as a linear layer by default, *i.e*., ϕ​(𝒙i)=Linear​(𝒙i)italic-ϕsubscript𝒙𝑖Linearsubscript𝒙𝑖\phi({\bm{x}}\_{i})=\text{Linear}({\bm{x}}\_{i}), containing a linear projection and a bias. We further improve the ability of 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} with multiple nonlinear layers.

In particular, we define the one-layer nonlinear mapping as a sequential of multiple operators [[20](#bib.bib20)], namely, the one-dimensional batch normalization [[29](#bib.bib29)], linear layer, ReLU activation, dropout [[53](#bib.bib53)], and another linear layer. In other words, the input 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} will be transformed by

|  |  |  |  |
| --- | --- | --- | --- |
|  | g​(𝒙i)=Linear​(Dropout​((ReLU​(Linear​(BatchNorm​(𝒙i)))))).𝑔subscript𝒙𝑖LinearDropoutReLULinearBatchNormsubscript𝒙𝑖g({\bm{x}}\_{i})=\text{Linear}\left(\text{Dropout}\left(\left(\text{ReLU}\left(\text{Linear}\left(\text{BatchNorm}\left({\bm{x}}\_{i}\right)\right)\right)\right)\right)\right)\;. |  | (5) |

Zero or more layers of blocks are applied in addition to the original linear layer to implement the final mapping.
The final ϕitalic-ϕ\phi applies an additional batch normalization to calibrate the embedding. Empirical results indicate that batch normalization outperforms other normalization strategies such as layer normalization [[5](#bib.bib5)] in learning a latent embedding space.

Stochastic Neighborhood Sampling.
Stochastic gradient descent is usually applied to optimize deep neural networks — a batch of instances is sampled, and the batch-wise loss is calculated for back-propagation.
However, pairwise distances need to be calculated during the training process, where the distances between the instance in the batch and the entire training set 𝒟𝒟\mathcal{D} are considered to estimate the loss function [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), introducing a high computational burden.

To accelerate the training speed of ModernNCA, we propose a Stochastic Neighborhood Sampling (SNS) strategy, where a subset 𝒟^^𝒟\hat{\mathcal{D}} of the training set 𝒟𝒟\mathcal{D} is randomly sampled as the candidate instances and distances with which are calculated. In other words, 𝒟^^𝒟\hat{\mathcal{D}} substitutes 𝒟𝒟\mathcal{D} in [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") and only the labels in 𝒟^^𝒟\hat{\mathcal{D}} can be utilized to predict the label of a given instance. The model searches neighbors in the entire training set 𝒟𝒟\mathcal{D} during the inference stage.

We empirically observed that the SNS not only increases the training efficiency of ModernNCA since fewer examples are utilized for back-propagation but also SNS improves the generalization ability of the learned metric. The improvements mainly come from the fact that the mapping ϕitalic-ϕ\phi is learned on more difficult prediction tasks during the training, so the ϕitalic-ϕ\phi adapts to the noisy and unstable neighborhoods in the test scenario. The influence of sampling ratio and other sampling strategies are investigated in the experiments.

Remark1: Relationship with other deep tabular methods. A recent deep tabular model TabR [[22](#bib.bib22)] also takes advantage of the nearest neighbor and predicts in a neighborhood-based manner. However, there are three main differences between our ModernNCA and TabR.
First, our approach is much more concise than TabR, where the latter employs both a deep neural network and a Transformer-like architecture to search the entire training set. Our ModernNCA validates that with the prediction strategy in [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), only a linear layer can achieve competitive results in most cases.
Second, ModernNCA is more efficient than TabR, requiring less training time and memory. This efficiency stems from both its simpler architecture and the stochastic sampling strategy.
In addition, instead of using an instance-specific prediction as in TabR, ModernNCA learns an embedding space and makes predictions in a more effective manner.
[Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") demonstrates the advantages of ModernNCA in terms of performance, efficiency, and model size when compared to TabR.

Remark 2: Another way to learn the deep tabular transformation. Our ModernNCA leverages NCA to implement the objective function in [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), where the quality of the learned tabular embedding ϕ​(𝒙)italic-ϕ𝒙\phi({\bm{x}}) directly influences the classification/regression performance over 𝒙𝒙{\bm{x}}.
As mentioned in [subsection 3.2](#S3.SS2 "3.2 Metric Learning on Tabular Data ‣ 3 Preliminary ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), another common strategy for learning ϕitalic-ϕ\phi involves a two-stage process: first, ϕitalic-ϕ\phi is optimized to match the comparison relationships between mapped instances, such as triplets, and then simple classifiers like KNN and LR are applied based on ϕitalic-ϕ\phi.
Since a mini-batch ℬℬ\mathcal{B} of examples is sampled at a time during the training process, we investigate the Supervised Contrastive loss [[51](#bib.bib51), [34](#bib.bib34)], where supervision is generated within a mini-batch. The objective to learn the mapping ϕitalic-ϕ\phi is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℓ​(ϕ∣ℬ)=−∑(𝒙i,yi)∈ℬ∑𝒙j∈P​(𝒙i)log⁡exp⁡(ϕ​(𝒙i)⋅ϕ​(𝒙j)/τ)∑𝒙k∈𝒟,𝒙k≠𝒙iexp⁡(ϕ​(𝒙i)⋅ϕ​(𝒙k)/τ),ℓconditionalitalic-ϕℬsubscriptsubscript𝒙𝑖subscript𝑦𝑖ℬsubscriptsubscript𝒙𝑗𝑃subscript𝒙𝑖⋅italic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗𝜏subscriptformulae-sequencesubscript𝒙𝑘𝒟subscript𝒙𝑘subscript𝒙𝑖⋅italic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑘𝜏\displaystyle\ell(\phi\mid\mathcal{B})=-\sum\_{({\bm{x}}\_{i},y\_{i})\in\mathcal{B}}\sum\_{{\bm{x}}\_{j}\in P({\bm{x}}\_{i})}\log\frac{\exp\left(\phi({\bm{x}}\_{i})\cdot\phi({\bm{x}}\_{j})/\tau\right)}{\sum\_{{\bm{x}}\_{k}\in\mathcal{D},{\bm{x}}\_{k}\neq{\bm{x}}\_{i}}\exp\left(\phi({\bm{x}}\_{i})\cdot\phi({\bm{x}}\_{k})/\tau\right)}\;, |  | (6) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | P​(𝒙i)={𝒙j∣∀(𝒙j,yj)∈ℬ∧yj=yi}.𝑃subscript𝒙𝑖conditional-setsubscript𝒙𝑗for-allsubscript𝒙𝑗subscript𝑦𝑗ℬsubscript𝑦𝑗subscript𝑦𝑖\displaystyle P({\bm{x}}\_{i})=\{{\bm{x}}\_{j}\mid\forall({\bm{x}}\_{j},y\_{j})\in\mathcal{B}\wedge y\_{j}=y\_{i}\}\;. |  | (7) |

P​(𝒙i)𝑃subscript𝒙𝑖P({\bm{x}}\_{i}) is the set of instances in the mini-batch that have the same label as 𝒙isubscript𝒙𝑖{\bm{x}}\_{i}. τ𝜏\tau is the temperature parameter used to calibrate the loss function. By minimizing [Equation 6](#S4.E6 "Equation 6 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), ϕitalic-ϕ\phi also pulls same-class instances together and pushes dissimilar ones apart. With the learned ϕitalic-ϕ\phi, we apply KNN for classification/regression for inference. We discretize the label values in the regression scenario and denote this baseline method as Tabular Contrastive (TabCon).

## 5 Experiments

We validate the performance of ModernNCA (M-NCA) in standard tabular classification and regression tasks. Then, we analyze the properties of ModernNCA with ablation studies.

### 5.1 Setups

Datasets. We validate our ModernNCA on 12 classification and 10 regression datasets collected from OpenML [[56](#bib.bib56)] and Kaggle. The detailed statistics of datasets are listed in [Table 5](#A1.T5 "Table 5 ‣ A.2 Dataset Information ‣ Appendix A Datasets ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") in the appendix. Four out of the 12 classification datasets have more than two classes.

Evaluation. We mainly follow the setups in [[20](#bib.bib20), [22](#bib.bib22)] to evaluate all methods. We randomly split each dataset into three partitions for training/validation/test with proportions 64%/16%/20%, respectively. For each dataset, we train a given model with 15 random seeds, and the average performance on the test set is reported. We use accuracy for classification (the higher, the better) and Root Mean Square Error (RMSE) for regression (the lower, the better). We also follow [[16](#bib.bib16), [37](#bib.bib37)] and report the average performance rank among all methods and datasets (the lower, the better).

Comparison Methods. We compare ModernNCA with three different threads of approaches.
First, we compare classical parametric tabular prediction methods, including the tree-based methods XGBoost[[12](#bib.bib12)] and CatBoost [[43](#bib.bib43)].
Then, we compare parametric deep tabular models. We follow the architecture in [[20](#bib.bib20)] to implement MLP, and MLPPLR [[22](#bib.bib22)] is a variant of MLP that leverages the piecewise linear encoding on numerical features [[21](#bib.bib21)].
FT-Transformer (FT-T) [[20](#bib.bib20)] learns tokens for each attribute and then makes predictions with transformer. PTaRL is a prototype-based tabular method based on space calibration [[68](#bib.bib68)].
For neighborhood-based tabular methods, we consider K𝐾KNN,
metric-learning methods like LMNN [[61](#bib.bib61)] and NCA [[19](#bib.bib19)], TabR [[22](#bib.bib22)], and our tabular contrastive baseline TabCon.111The code of our ModernNCA is available at <https://github.com/qile2000/LAMDA-TALENT>.

Implementation Details.
We pre-process all datasets following [[20](#bib.bib20)].
For all deep methods, we set the batch size as 1024.
The hyper-parameters of all methods are searched based on the training and validation set via Optuna [[2](#bib.bib2)] following [[20](#bib.bib20), [22](#bib.bib22)] over 100 trials. The best-performed hyper-parameters are fixed during the final 15 seeds.
Since the sampling rate of SNS effectively enhances the performance and reduces the training overhead, we treat it as a hyper-parameter and search within the range of [0.05, 0.6].
The metric-learning methods like LMNN are implemented based on the metric-learn library [[15](#bib.bib15)].
We use one-hot encoding for categorical features, and PLR (lite) embedding, a simplified version of PLR embedding(the combination of periodic embeddings, a linear layer, and ReLU) [[21](#bib.bib21)] proposed in [[22](#bib.bib22)], for numerical features. For a fair comparison, the same encoding strategy is also applied in MLPPLR and TabR.

Table 1: The average accuracy (the higher, the better) of all methods on 12 classification datasets. The best results on each dataset are in bold, and the second-best ones are underlined. We also report the average rank of different methods among all datasets (the lower, the better).

| ↑↑\uparrow | M-NCA | TabCon | XGBoost | CatBoost | MLP | FT-T | TabR | KNN | MLPPLR | PTaRL | LMNN | NCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD | 87.29 | 86.97 | 87.21 | 87.48 | 85.80 | 85.89 | 86.86 | 84.61 | 86.97 | 85.53 | 83.21 | 83.87 |
| CU | 79.99 | 77.72 | 81.73 | 81.03 | 75.54 | 80.65 | 79.61 | 70.08 | 80.75 | 75.89 | 72.12 | 73.78 |
| EL | 96.12 | 87.44 | 92.06 | 91.69 | 85.49 | 87.50 | 96.46 | 84.27 | 86.93 | 85.10 | 83.56 | 85.15 |
| EY | 99.41 | 82.88 | 71.02 | 71.88 | 61.29 | 71.49 | 98.20 | 59.46 | 74.27 | 60.13 | 54.98 | 66.50 |
| HE | 39.84 | 38.32 | 37.86 | 38.26 | 38.34 | 38.49 | 40.80 | 32.91 | 38.95 | 37.64 | 29.36 | 31.30 |
| HI | 73.09 | 72.36 | 72.73 | 72.66 | 72.29 | 72.92 | 72.85 | 66.78 | 72.64 | 72.48 | 61.44 | 66.94 |
| IP | 87.96 | 87.65 | 88.64 | 87.74 | 84.86 | 84.95 | 87.23 | 83.72 | 87.84 | 84.83 | 83.81 | 84.10 |
| JA | 74.08 | 72.17 | 72.55 | 72.39 | 71.97 | 72.38 | 73.44 | 65.67 | 72.10 | 71.52 | 59.12 | 64.64 |
| MA | 87.90 | 87.22 | 87.67 | 87.87 | 87.08 | 87.73 | 87.91 | 84.65 | 87.17 | 86.94 | 83.68 | 84.73 |
| OT | 82.37 | 78.94 | 82.45 | 82.21 | 81.47 | 80.82 | 82.38 | 78.15 | 81.24 | 81.20 | 77.37 | 78.85 |
| PH | 89.29 | 87.88 | 87.66 | 87.97 | 86.88 | 87.86 | 89.10 | 86.86 | 87.13 | 85.54 | 85.75 | 87.05 |
| WI | 74.74 | 73.72 | 74.39 | 74.59 | 72.50 | 72.64 | 74.22 | 77.10 | 72.50 | 72.16 | 73.58 | 72.60 |
| rank | 2.000 | 5.917 | 3.833 | 3.750 | 8.000 | 5.417 | 3.083 | 10.000 | 5.667 | 9.250 | 11.333 | 9.750 |

### 5.2 Main Results

The comparison results between ModernNCA and other methods for classification and regression tasks are reported in [Table 1](#S5.T1 "Table 1 ‣ 5.1 Setups ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") and [Table 2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), respectively, where ModernNCA achieves the best average rank in both cases.

For classification results in [Table 1](#S5.T1 "Table 1 ‣ 5.1 Setups ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), ModernNCA achieves better results than tree-based methods like XGBoost in most cases, which indicates that utilizing deep neural network ModernNCA has a stronger ability to make nonlinear predictions. It is clearly observed that although classical metric learning methods such as LMNN and NCA can improve KNN in some cases, they are difficult to achieve competitive performance as tree-based and deep methods. When compared with deep tabular methods like FT-T and MLPPLR, ModernNCA still keeps its superiority. The most related comparison method is TabR, which extracts features of all neighbors in the training set and makes predictions with a variant of Transformer. Our ModernNCA outperforms TabR with simpler architecture and shorter training time. The results indicate that ModernNCA acts as an effective tabular deep baseline.
Our TabCon baseline also demonstrates the effectiveness of its learned mapping, outperforming classical metric learning methods in all cases. TabCon is competitive with FT-T and even surpasses XGBoost in some cases, such as “HE” and “PH”. The superiority of ModernNCA over TabCon may result from the direct linkage between the learned mapping and prediction performance in ModernNCA.
Similar phenomena exist in regression cases in [Table 2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). ModernNCA is the only deep model that consistently outperforms tree-based methods on regression datasets overall. Classical metric-learning methods cannot be applied to regression directly. Our ModernNCA shows its superiority over two competitive methods MLPPLR and TabR.
[Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") indicates the comparison between representative classification and regression methods w.r.t. the performance, training time, and model size over all datasets. Although some models, such as TabR, obtain strong performance, they require much longer training time. Our ModernNCA makes a good balance among various model evaluation criteria.

Table 2: The average RMSE (the lower, the better) of all methods on 10 regression datasets. The best results on each dataset are in bold, and the second-best ones are underlined.
The scientific notation beside the dataset name indicates the scale of the results, *e.g*., ×10absent10\times 10 means the final value of all results should be multiple by 101010.
We also report the average rank among all datasets (the lower, the better).

| ↓↓\downarrow | M-NCA | TabCon | XGBoost | CatBoost | MLP | FT-T | TabR | KNN | MLPPLR | PTaRL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI×10−3absentsuperscript103\times 10^{-3} | .1532 | .1522 | .1527 | .1465 | .1558 | .1565 | .1546 | .2431 | .1522 | .1563 |
| BI×102absentsuperscript102\times 10^{2} | .7057 | .7352 | .7180 | .7264 | .7773 | .7466 | .6657 | 1.087 | .7155 | .7193 |
| CA | .4212 | .4581 | .4328 | .4360 | .5074 | .4701 | .4157 | .5843 | .4690 | .5086 |
| CM×103absentsuperscript103\times 10^{3} | .4618 | .5011 | .4809 | .4929 | .5131 | .5187 | .5091 | .8754 | .4655 | .5211 |
| CP×10absent10\times 10 | .2381 | .2473 | .2404 | .2365 | .2490 | .2310 | .2337 | .8825 | .2468 | .2487 |
| FO×10absent10\times 10 | .7272 | .7436 | .7384 | .7393 | .7900 | .7885 | .7398 | .8140 | .7428 | .7896 |
| HO×105absentsuperscript105\times 10^{5} | .3041 | .3156 | .3026 | .3051 | .3161 | .3074 | .3133 | .3663 | .3093 | .3149 |
| HOU×105absentsuperscript105\times 10^{5} | .4291 | .4963 | .4797 | .4546 | .5221 | .4791 | .4149 | .6889 | .4921 | .5118 |
| LA×103absentsuperscript103\times 10^{3} | .4135 | .4529 | .4448 | .4506 | .4789 | .4609 | .4606 | .4968 | .4753 | .4777 |
| MIA×105absentsuperscript105\times 10^{5} | .8284 | .9322 | .8740 | .8034 | .8973 | .8717 | .8585 | 1.508 | .8562 | .9302 |
| rank | 2.200 | 6.100 | 3.500 | 3.100 | 8.300 | 5.900 | 3.600 | 10.000 | 4.500 | 7.800 |




Table 3: Results of comparison methods with different architectures. NCA cannot be applied for regression tasks.

|  | TabR | NCA | Linear | LayerNorm | Residual | M-NCA |
| --- | --- | --- | --- | --- | --- | --- |
| CA ↓↓\downarrow | .4157 | - | .4212 | .4210 | .4237 | .4212 |
| AD ↑↑\uparrow | 86.86 | 83.87 | 87.18 | 87.20 | 87.25 | 87.29 |
| MIA×105absentsuperscript105\times 10^{5}↓↓\downarrow | .8585 | - | .8325 | .9070 | .8418 | .8284 |
| PH ↑↑\uparrow | 89.10 | 87.05 | 89.09 | 88.70 | 89.78 | 89.29 |

### 5.3 Ablation Studies

We analyze different properties of the proposed ModernNCA based on three datasets, with CA/MIA for regression and AD/PH for classification.

Linear vs. Deep Architectures.
We first investigate the design of architectures ϕitalic-ϕ\phi in ModernNCA, where we add one or more layers of blocks g​(⋅)𝑔⋅g(\cdot) based on a linear projection. The number of layers is set as a hyper-parameter and could be determined based on the validation set. We consider three choices. First, we set ϕitalic-ϕ\phi as the linear projection, where the dimensionality of the projected space is the hyper-parameter. We also replace batch normalization in the block with layer normalization. Finally, we equip ϕitalic-ϕ\phi with another residual link from the block’s input to its output.

The results are listed in [Table 3](#S5.T3 "Table 3 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), and we also include the results of TabR/NCA for reference. Although NCA uses a linear projection, we find the linear version of ModernNCA (the fourth column) achieves much better performance on two classification datasets. For example, on AD, the linear version of ModernNCA outperforms NCA by around 4%. The comparison indicates that the objective and training strategy of ModernNCA is essential. When compared with the last column where ModernNCA utilizes the deep neural network, we find the usage of nonlinear deep architecture is necessary on datasets such as MIA. The linear version can achieve competitive or even better results since the projected dimension is searched in a smaller hyper-parameter space, and for ModernNCA the number of additional layers is searched based on the validation set from zero to more layers.

The variant of the fifth column utilizes another normalization strategy, layer normalization, in the blocks instead of batch normalization in ModernNCA. We empirically find that batch normalization performs better on average, especially in classification tasks. When comparing residual and MLP blocks in the last two columns, we find using multiple nonlinear layers directly performs a bit better, so we choose the MLP implementation in [Equation 5](#S4.E5 "Equation 5 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") for ModernNCA.

Table 4: Results of ModernNCA variants with different distance functions to obtain neighborhood relationship in the learned embedding space.

|  | Euclid | Euclid2 | ℓ1subscriptℓ1\ell\_{1}-Norm | Cosine | Inner Product |
| --- | --- | --- | --- | --- | --- |
| CA ↓↓\downarrow | .4212 | .4271 | .4184 | .4264 | .4528 |
| AD ↑↑\uparrow | 87.29 | 87.04 | 87.25 | 87.18 | 87.07 |
| MIA×105absentsuperscript105\times 10^{5}↓↓\downarrow | .8284 | .8637 | .8624 | .8627 | .8497 |
| PH ↑↑\uparrow | 89.29 | 89.38 | 89.08 | 89.10 | 88.23 |

The Influence of Distance Functions. The predicted label of a target instance 𝒙isubscript𝒙𝑖{\bm{x}}\_{i} is determined by the label of its neighbors in the learned embedding space projected by ϕitalic-ϕ\phi. The distance function is the key to determining the pairwise relationship between instances in the embedding space and influences the weights in [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").
In ModernNCA, we choose Euclidean distance distEUC​(ϕ​(𝒙i),ϕ​(𝒙j))=(ϕ​(𝒙i)−ϕ​(𝒙j))⊤​(ϕ​(𝒙i)−ϕ​(𝒙j))=‖ϕ​(𝒙i)−ϕ​(𝒙j)‖2subscriptdistEUCitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗superscriptitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗topitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗subscriptnormitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗2{\rm dist}\_{\rm EUC}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j}))=\sqrt{(\phi({\bm{x}}\_{i})-\phi({\bm{x}}\_{j}))^{\top}(\phi({\bm{x}}\_{i})-\phi({\bm{x}}\_{j}))}=\|\phi({\bm{x}}\_{i})-\phi({\bm{x}}\_{j})\|\_{2}. We also utilize other distance functions, *e.g*., the squared Euclidean distance distEUC2​(ϕ​(𝒙i),ϕ​(𝒙j))superscriptsubscriptdistEUC2italic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗{\rm dist}\_{\rm EUC}^{2}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j})), the ℓ1subscriptℓ1\ell\_{1}-norm distance dist​(ϕ​(𝒙i),ϕ​(𝒙j))=‖ϕ​(𝒙i)−ϕ​(𝒙j)‖1distitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗subscriptnormitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗1{\rm dist}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j}))=\|\phi({\bm{x}}\_{i})-\phi({\bm{x}}\_{j})\|\_{1}, the (negative) cosine similarity dist​(ϕ​(𝒙i),ϕ​(𝒙j))=−(𝒙i⊤​𝒙j)/(‖𝒙i‖2​‖𝒙j‖2)distitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗superscriptsubscript𝒙𝑖topsubscript𝒙𝑗subscriptnormsubscript𝒙𝑖2subscriptnormsubscript𝒙𝑗2{\rm dist}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j}))=-({\bm{x}}\_{i}^{\top}{\bm{x}}\_{j})/(\|{\bm{x}}\_{i}\|\_{2}\|{\bm{x}}\_{j}\|\_{2}), and the (negative) inner product dist​(ϕ​(𝒙i),ϕ​(𝒙j))=−𝒙i⊤​𝒙jdistitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗superscriptsubscript𝒙𝑖topsubscript𝒙𝑗{\rm dist}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j}))=-{\bm{x}}\_{i}^{\top}{\bm{x}}\_{j}.
The results using different distance functions are listed in [Table 4](#S5.T4 "Table 4 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), where Euclidean distance achieves good results on average. Although ℓ1subscriptℓ1\ell\_{1}-norm also performs well, it will introduce larger computational costs than the Euclidean distance.

![Refer to caption](/html/2407.03257/assets/x3.png)

(a) classification

![Refer to caption](/html/2407.03257/assets/x4.png)

(b) regression

Figure 2: The change of performance with different sampling rates among {10%, 30%, 50%, 80%, 100%} in stochastic neighborhood sampling strategy.

The Influence of Sampling Ratios.
Due to the huge computational cost of calculating distance in the learned embedding space, our ModernNCA utilizes a Stochastic Neighborhood Sampling (SNS) strategy, where only a proportion of training set data are randomly sampled for each mini-batch. The whole training set is utilized in the test phase. Due to the usage of a subset of sampling data, the training time cost is reduced a lot.
We vary the proportion and evaluate the corresponding test performance of the model in [Figure 2](#S5.F2 "Figure 2 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").
We observe from [Figure 2](#S5.F2 "Figure 2 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") that 30%-50% ratios of the training set help more for ModernNCA than using the full training set. SNS not only improves the training efficiency but also the model’s generalization ability.
The plots demonstrate that the ratio of sampled training data matters, which is a hyper-parameter tuned for ModernNCA.

The Influence of Sampling Strategy.
As mentioned before, SNS randomly samples a subset of training data for each mini-batch when calculating the loss of [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). We also investigate whether we could further improve the classification/regression ability of the model when we incorporate richer information during the sampling process, *e.g*., the label of the instances.

We consider another two sampling strategies in addition to the fully random one we used before. First is class-wise random sampling, which means that given a proportion, we sample from each class in the training set and combine them together. This strategy takes advantage of the training label information and keeps the instances from all classes that will exist in the sampled subset. Besides, we also consider the sampling strategy based on the pairwise distances between instances. Since the neighbors of an instance may contribute more (with larger weights) in [Equation 4](#S4.E4 "Equation 4 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), so given a mini-batch, we first calculate the Euclidean distance between instances in the batch and all the training set with the embedding function ϕitalic-ϕ\phi in the current epoch. Then we sample the training set based on the reciprocal of the pairwise distance value. In detail, given an instance 𝒙isubscript𝒙𝑖{\bm{x}}\_{i}, we provide instance-specific neighborhood candidates and 𝒙jsubscript𝒙𝑗{\bm{x}}\_{j} in the training set is sampled based on the probability ∼1/(dist​(ϕ​(𝒙i),ϕ​(𝒙j)))τsimilar-toabsent1superscriptdistitalic-ϕsubscript𝒙𝑖italic-ϕsubscript𝒙𝑗𝜏\sim 1/({\rm dist}(\phi({\bm{x}}\_{i}),\phi({\bm{x}}\_{j})))^{\tau}. τ𝜏\tau is a non-negative hyper-parameter to calibrate the distribution.
The distance calculation requires forward passes of the model ϕitalic-ϕ\phi over all the training instances, and the instance-specific neighborhood makes the loss related to a wide range of the training data. Therefore, the distance-based sampling strategy has a low training speed and high computational burden.

![Refer to caption](/html/2407.03257/assets/x5.png)


Figure 3: The influence of different sampling strategies, namely, the random strategy, the label-based strategy, and the distance-based strategy.

The comparison results between different sampling strategies on two classification datasets are listed in [Figure 3](#S5.F3 "Figure 3 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). We find the label-based sampling strategy cannot provide further improvements. Although the distance-based strategy helps in certain cases, the improvements are limited. Taking a holistic consideration of the performance and efficiency, we choose to use the vanilla random sampling in ModernNCA.

![Refer to caption](/html/2407.03257/assets/x6.png)

(a) CA ↓↓\downarrow

![Refer to caption](/html/2407.03257/assets/x7.png)

(b) AD ↑↑\uparrow

![Refer to caption](/html/2407.03257/assets/x8.png)

(c) MIA ↓↓\downarrow

![Refer to caption](/html/2407.03257/assets/x9.png)

(d) PH ↑↑\uparrow

Figure 4: The change of performance given few-shot training examples among the different percentages of the training set examples. Since the PH dataset has only one sample at 0.05% shot percent, we only show model performance for shot percents from 0.1% to 1%.

Learning with Few-Shot Data.
We investigate the change in performance when the number of training examples is limited. The results are shown in [Figure 4](#S5.F4 "Figure 4 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").
Given a dataset, we synthesize the few-shot learning scenario by randomly sampling a subset of the training data with the radio changes from {0.05%,0.1%,0.2%,0.5%,1%}percent0.05percent0.1percent0.2percent0.5percent1\{0.05\%,0.1\%,0.2\%,0.5\%,1\%\}. Due to the small-size of the training set, we learn all methods with the default hyper-parameters. The results are repeated 10 times and the average of them are reported.
Based on the few-shot performance shown in [Figure 4](#S5.F4 "Figure 4 ‣ 5.3 Ablation Studies ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), we find our ModernNCA is competitive when compared with other tabular prediction methods.

## 6 Conclusion

Making predictions by leveraging relationships within a learned embedding space is a classical metric learning notion in machine learning. While traditional metric learning methods often underperform compared to tree-based approaches on tabular datasets, our study revisits and enhances one of the most representative methods, NCA. The improved ModernNCA establishes itself as a strong baseline for deep tabular prediction tasks, often outperforming other methods in terms of classification and regression capabilities, while also reducing model size and training time. We believe our research will encourage further investigation into classical methods within the field of tabular data. By making minor yet effective modifications to their core concepts, these methods can be revitalized, offering significant improvements and novel solutions.

## References

* Ahmed et al. [2010]

  Nesreen K Ahmed, Amir F Atiya, Neamat El Gayar, and Hisham El-Shishiny.
  An empirical comparison of machine learning models for time series forecasting.
  *Econometric reviews*, 29(5-6):594–621, 2010.
* Akiba et al. [2019]

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, 2019.
* Amand and Huan [2017]

  Joseph St. Amand and Jun Huan.
  Sparse compositional local metric learning.
  In *SIGKDD*, 2017.
* Arik and Pfister [2021]

  Sercan Ö. Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  In *AAAI*, 2021.
* Ba et al. [2016]

  Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.
  Layer normalization.
  *CoRR*, abs/1607.06450, 2016.
* Bahri et al. [2022]

  Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler.
  Scarf: Self-supervised contrastive learning using random feature corruption.
  In *ICLR*, 2022.
* Bellet et al. [2015]

  Aurélien Bellet, Amaury Habrard, and Marc Sebban.
  *Metric Learning*.
  Morgan & Claypool Publishers, 2015.
* Bishop [2006]

  Christopher Bishop.
  *Pattern recognition and machine learning*.
  Springer, 2006.
* Borisov et al. [2022]

  Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci.
  Deep neural networks and tabular data: A survey.
  *IEEE Transactions on Neural Networks and Learning Systems*, abs/2110.01889:1–21, 2022.
* Chang et al. [2022]

  Chun-Hao Chang, Rich Caruana, and Anna Goldenberg.
  NODE-GAM: neural generalized additive model for interpretable deep learning.
  In *ICLR*, 2022.
* Chen et al. [2022]

  Jintai Chen, Kuanlun Liao, Yao Wan, Danny Z. Chen, and Jian Wu.
  Danets: Deep abstract networks for tabular data classification and regression.
  In *AAAI*, 2022.
* Chen and Guestrin [2016]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *KDD*, 2016.
* Cheng et al. [2016]

  Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah.
  Wide & deep learning for recommender systems.
  In *DLRS*, 2016.
* Davis et al. [2007]

  Jason V. Davis, Brian Kulis, Prateek Jain, Suvrit Sra, and Inderjit S. Dhillon.
  Information-theoretic metric learning.
  In *ICML*, 2007.
* de Vazelhes et al. [2020]

  William de Vazelhes, CJ Carey, Yuan Tang, Nathalie Vauquier, and Aurélien Bellet.
  metric-learn: Metric Learning Algorithms in Python.
  *Journal of Machine Learning Research*, 21(138):1–6, 2020.
* Delgado et al. [2014]

  Manuel Fernández Delgado, Eva Cernadas, Senén Barro, and Dinani Gomes Amorim.
  Do we need hundreds of classifiers to solve real world classification problems?
  *Journal of Machine Learning Research*, 15(1):3133–3181, 2014.
* Devlin et al. [2019]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  BERT: pre-training of deep bidirectional transformers for language understanding.
  In *NAACL-HLT*, 2019.
* Feurer et al. [2015]

  Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Tobias Springenberg, Manuel Blum, and Frank Hutter.
  Efficient and robust automated machine learning.
  In *NIPS*, 2015.
* Goldberger et al. [2004]

  Jacob Goldberger, Sam T. Roweis, Geoffrey E. Hinton, and Ruslan Salakhutdinov.
  Neighbourhood components analysis.
  In *NIPS*, 2004.
* Gorishniy et al. [2021]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, 2021.
* Gorishniy et al. [2022]

  Yury Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  In *NeurIPS*, 2022.
* Gorishniy et al. [2024]

  Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko.
  Tabr: Tabular deep learning meets nearest neighbors in 2023.
  In *ICLR*, 2024.
* Grinsztajn et al. [2022]

  Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux.
  Why do tree-based models still outperform deep learning on typical tabular data?
  In *NeurIPS*, 2022.
* Guo et al. [2017]

  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He.
  Deepfm: A factorization-machine based neural network for CTR prediction.
  In *IJCAI*, 2017.
* Guyon et al. [2019]

  Isabelle Guyon, Lisheng Sun-Hosoya, Marc Boullé, Hugo Jair Escalante, Sergio Escalera, Zhengying Liu, Damir Jajetic, Bisakha Ray, Mehreen Saeed, Michèle Sebag, et al.
  Analysis of the automl challenge series.
  *Automated Machine Learning*, 177:177–219, 2019.
* Hassan et al. [2020]

  Md. Rafiul Hassan, Sadiq Al-Insaif, Muhammad Imtiaz Hossain, and Joarder Kamruzzaman.
  A machine learning approach for prediction of pregnancy outcome following IVF treatment.
  *Neural Computing and Applications*, 32(7):2283–2297, 2020.
* Hollmann et al. [2023]

  Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.
  Tabpfn: A transformer that solves small tabular classification problems in a second.
  In *ICLR*, 2023.
* Hsieh et al. [2017]

  Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge J. Belongie, and Deborah Estrin.
  Collaborative metric learning.
  In *WWW*, 2017.
* Ioffe and Szegedy [2015]

  Sergey Ioffe and Christian Szegedy.
  Batch normalization: Accelerating deep network training by reducing internal covariate shift.
  In *ICML*, 2015.
* Kadra et al. [2021]

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In *NeurIPS*, pages 23928–23941, 2021.
* Katzir et al. [2021]

  Liran Katzir, Gal Elidan, and Ran El-Yaniv.
  Net-dnf: Effective deep modeling of tabular data.
  In *ICLR*, 2021.
* Ke et al. [2017]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In *NIPS*, 2017.
* Kedem et al. [2012]

  Dor Kedem, Stephen Tyree, Kilian Q. Weinberger, Fei Sha, and Gert R. G. Lanckriet.
  Non-linear metric learning.
  In *NIPS*, 2012.
* Khosla et al. [2020]

  Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan.
  Supervised contrastive learning.
  In *NeurIPS*, 2020.
* Kulis [2013]

  Brian Kulis.
  Metric learning: A survey.
  *Foundations and Trends in Machine Learning*, 5(4), 2013.
* Levin et al. [2023]

  Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C. Bayan Bruss, Tom Goldstein, Andrew Gordon Wilson, and Micah Goldblum.
  Transfer learning with deep tabular models.
  In *ICLR*, 2023.
* McElfresh et al. [2023]

  Duncan C. McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C., Ganesh Ramakrishnan, Micah Goldblum, and Colin White.
  When do neural nets outperform boosted trees on tabular data?
  In *NeurIPS*, 2023.
* Mohri et al. [2012]

  Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar.
  *Foundations of Machine Learning*.
  MIT Press, 2012.
* Nederstigt et al. [2014]

  Lennart J Nederstigt, Steven S Aanen, Damir Vandic, and Flavius Frasincar.
  Floppies: a framework for large-scale ontology population of product information from tabular data in e-commerce stores.
  *Decision Support Systems*, 59:296–311, 2014.
* Noh et al. [2018]

  Yung-Kyun Noh, Byoung-Tak Zhang, and Daniel D. Lee.
  Generative local metric learning for nearest neighbor classification.
  *IEEE Transactions on pattern analysis and machine intelligence*, 40(1):106–118, 2018.
* Onishi et al. [2023]

  Soma Onishi, Kenta Oono, and Kohei Hayashi.
  Tabret: Pre-training transformer-based tabular models for unseen columns.
  *CoRR*, abs/2303.15747, 2023.
* Popov et al. [2020]

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular data.
  In *ICLR*, 2020.
* Prokhorenkova et al. [2018]

  Liudmila Ostroumova Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Qian et al. [2015]

  Qi Qian, Rong Jin, Shenghuo Zhu, and Yuanqing Lin.
  Fine-grained visual categorization via multi-stage metric learning.
  In *CVPR*, 2015.
* Richardson et al. [2007]

  Matthew Richardson, Ewa Dominowska, and Robert Ragno.
  Predicting clicks: estimating the click-through rate for new ads.
  In *WWW*, 2007.
* Rubachev et al. [2022]

  Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko.
  Revisiting pretraining objectives for tabular deep learning.
  *CoRR*, abs/2207.03208, 2022.
* Schroff et al. [2015]

  Florian Schroff, Dmitry Kalenichenko, and James Philbin.
  Facenet: A unified embedding for face recognition and clustering.
  In *CVPR*, 2015.
* Shen et al. [2023]

  Junhong Shen, Liam Li, Lucio M Dery, Corey Staten, Mikhail Khodak, Graham Neubig, and Ameet Talwalkar.
  Cross-modal fine-tuning: Align then refine.
  In *ICML*, 2023.
* Shi et al. [2014]

  Yuan Shi, Aurélien Bellet, and Fei Sha.
  Sparse compositional metric learning.
  In *AAAI*, 2014.
* Simonyan and Zisserman [2015]

  Karen Simonyan and Andrew Zisserman.
  Very deep convolutional networks for large-scale image recognition.
  In *ICLR*, 2015.
* Sohn [2016]

  Kihyuk Sohn.
  Improved deep metric learning with multi-class n-pair loss objective.
  In *NIPS*, 2016.
* Song et al. [2016]

  Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese.
  Deep metric learning via lifted structured feature embedding.
  In *CVPR*, 2016.
* Srivastava et al. [2014]

  Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
  Dropout: a simple way to prevent neural networks from overfitting.
  *Journal of Machine Learning Research*, 15(1):1929–1958, 2014.
* Ucar et al. [2021]

  Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards.
  Subtab: Subsetting features of tabular data for self-supervised representation learning.
  In *NeurIPS*, pages 18853–18865, 2021.
* Van der Maaten and Hinton [2008]

  Laurens Van der Maaten and Geoffrey Hinton.
  Visualizing data using t-sne.
  *Journal of machine learning research*, 9(11), 2008.
* Vanschoren et al. [2014]

  Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo.
  Openml: networked science in machine learning.
  *ACM SIGKDD Explorations Newsletter*, 15(2):49–60, 2014.
* Vaswani et al. [2017]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *NIPS*, 2017.
* Vinyals et al. [2016]

  Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
  Matching networks for one shot learning.
  In *NIPS*, 2016.
* Wang et al. [2012]

  Jun Wang, Alexandros Kalousis, and Adam Woznica.
  Parametric local metric learning for nearest neighbor classification.
  In *NIPS*, 2012.
* Wei et al. [2023]

  Tianjun Wei, Jianghong Ma, and Tommy W. S. Chow.
  Collaborative residual metric learning.
  In *SIGIR*, 2023.
* Weinberger and Saul [2009]

  Kilian Q. Weinberger and Lawrence K. Saul.
  Distance metric learning for large margin nearest neighbor classification.
  *Journal of Machine Learning Research*, 10:207–244, 2009.
* Xing et al. [2002]

  Eric P. Xing, Andrew Y. Ng, Michael I. Jordan, and Stuart Russell.
  Distance metric learning with application to clustering with side-information.
  In *NIPS*, 2002.
* Xu et al. [2012]

  Zhixiang Eddie Xu, Kilian Q. Weinberger, and Olivier Chapelle.
  Distance metric learning for kernel machines.
  *CoRR*, abs/1208.3422, 2012.
* Yang et al. [2018]

  Xun Yang, Meng Wang, and Dacheng Tao.
  Person re-identification with metric learning using privileged information.
  *IEEE Transactions on Image Processing*, 27(2):791–805, 2018.
* Ye et al. [2016]

  Han-Jia Ye, De-Chuan Zhan, Xue-Min Si, Yuan Jiang, and Zhi-Hua Zhou.
  What makes objects similar: A unified multi-metric learning approach.
  In *NIPS*, 2016.
* Ye et al. [2020]

  Han-Jia Ye, De-Chuan Zhan, Nan Li, and Yuan Jiang.
  Learning multiple local metrics: Global consideration helps.
  *IEEE Transactions on pattern analysis and machine intelligence*, 42(7):1698–1712, 2020.
* Ye et al. [2023]

  Han-Jia Ye, Qi-Le Zhou, and De-Chuan Zhan.
  Training-free generalization on heterogeneous tabular data via meta-representation.
  *CoRR*, abs/2311.00055, 2023.
* Ye et al. [2024]

  Hangting Ye, Wei Fan, Xiaozhuang Song, Shun Zheng, He Zhao, Dan dan Guo, and Yi Chang.
  Ptarl: Prototype-based tabular representation learning via space calibration.
  In *ICLR*, 2024.
* Yi et al. [2014]

  Dong Yi, Zhen Lei, Shengcai Liao, and Stan Z. Li.
  Deep metric learning for person re-identification.
  In *ICME*, 2014.
* Zhou et al. [2023]

  Qi-Le Zhou, Han-Jia Ye, Leye Wang, and De-Chuan Zhan.
  Unlocking the transferability of tokens in deep models for tabular data.
  *CoRR*, abs/2310.15149, 2023.
* Zhu et al. [2023]

  Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, and Mahsa Shoaran.
  Xtab: Cross-table pretraining for tabular transformers.
  In *ICML*, 2023.

The Appendix consists of five sections:

* •

  [Appendix A](#A1 "Appendix A Datasets ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"): Datasets and implementation details.
* •

  [Appendix B](#A2 "Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"): Additional experimental results.

## Appendix A Datasets

In this section, we present the preprocessing steps applied to the dataset before training, along with a detailed description of the dataset we used.

### A.1 Data Pre-processing

We follow [[20](#bib.bib20)] and pre-process datasets for all methods.
For example, we use standardization, involving mean subtraction and scaling, to normalize each numerical dataset. We also apply one-hot encoding on all categorical features.

### A.2 Dataset Information

The datasets come from different domains, including [OpenML](https://www.openml.org/) [[56](#bib.bib56)] and [Kaggle](https://www.kaggle.com/). These datasets are commonly used in the tabular prediction fields [[23](#bib.bib23), [20](#bib.bib20)].
The descriptions are shown in [Table 5](#A1.T5 "Table 5 ‣ A.2 Dataset Information ‣ Appendix A Datasets ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").
To reduce randomness, we used datasets with sample sizes greater than 1000.

Table 5: Descriptions of all datasets. There are 8 binary classification datasets, 4 multi-class classification datasets, and 10 regression datasets.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Abbreviation | Task Type | Data Size | Num. | Cat. | Class Num |
| Adult | AD | classification | 48842 | 6 | 8 | 2 |
| Ailerons | AI | regression | 13750 | 40 | 0 | - |
| California Housing | CA | regression | 20640 | 8 | 0 | - |
| CPMP-2015 | CM | regression | 2108 | 22 | 2 | - |
| cpu\_act | CP | regression | 8192 | 21 | 0 | - |
| electricity | EL | classification | 45312 | 7 | 1 | 2 |
| eye\_movements | EY | classification | 10936 | 24 | 3 | 3 |
| Food\_Delivery\_Time | FO | regression | 45593 | 6 | 2 | - |
| Helena | HE | classification | 65196 | 27 | 0 | 100 |
| Higgs Small | HI | classification | 98050 | 28 | 0 | 2 |
| houses | HOU | regression | 20640 | 8 | 0 | - |
| House\_16H | HO | regression | 22784 | 16 | 0 | - |
| Jannis | JA | classification | 83733 | 54 | 0 | 4 |
| bike\_sharing\_demand\_challange | BI | regression | 10886 | 3 | 6 | - |
| KDDCup09\_upselling | CU | classification | 5128 | 34 | 15 | 2 |
| kdd\_ipums\_la\_97-small | IP | classification | 5188 | 20 | 0 | 2 |
| Laptop\_Prices\_Dataset | LA | regression | 4441 | 9 | 0 | - |
| MagicTelescope | MA | classification | 19020 | 9 | 0 | 2 |
| MiamiHousing2016 | MIA | regression | 13932 | 16 | 0 | - |
| Otto-Group-Products | OT | classification | 61878 | 93 | 0 | 9 |
| phoneme | PH | classification | 5404 | 5 | 0 | 2 |
| wine | WI | classification | 2554 | 4 | 0 | 2 |

We randomly sample 20% instances to construct the test set. The remaining 80% instances are used for splitting. In the training set, we randomly hold out 20% of instances as the validation set.
The validation set is used for tuning hyper-parameters and carrying out early stopping.
We selected the model that performed best on the validation set with tuned hyper-parameters for evaluation.

### A.3 Hardware

The majority of experiments were conducted on the Tesla V100 GPU, including those related to time and memory overhead calculation.

### A.4 Implementations of TabCon

TabCon is trained using Supervised Contrastive loss in a two-stage process. In the training stage, a mapping ϕitalic-ϕ\phi, defined the same as ModernNCA for simplicity, is trained to project the original features into a latent space as described in [Equation 7](#S4.E7 "Equation 7 ‣ 4 ModernNCA ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). Before calculating the Supervised Contrastive loss, each mapping ϕ​(𝒙)italic-ϕ𝒙\phi({\bm{x}}) is normalized to unit length. In the validation stage, a simple predictor (*e.g*., KNN and LR) is further trained based on the mappings learned in the training stage. Early stopping is applied during the validation stage based on the performance of the simple predictor. We use KNN with Euclidean distance as the default predictor.

Since Supervised Contrastive loss was originally proposed for classification, we discretize the labels for regression tasks before calculating the loss. Specifically, the samples are divided into different bins according to their values, and we use quantiles as the boundaries of the bins to ensure that each bin contains a similar number of samples. The number of bins is set as a hyper-parameter.

## Appendix B Additional Experiments

### B.1 Additional Ablation Studies

The Influence of PLR Embedding.
The results of ModernNCA in [Table 1](#S5.T1 "Table 1 ‣ 5.1 Setups ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") and [Table 2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later") leverage the PLR embedding for numerical features [[21](#bib.bib21)] (where all categorical features are processed in the one-hot form). We compare ModernNCA as well as MLP with or without PLR in [Table 6](#A2.T6 "Table 6 ‣ B.1 Additional Ablation Studies ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").

The results demonstrate that our ModernNCA can also utilize the advantages of PLR to achieve further improvements.
However, different from MLP, the vanilla variant of ModernNCA without PLR is still competitive. For example, in regression dataset CA, ModernNCA without PLR can outperform MLP as well as MLPPLR. The results validate that the ability of ModernNCA comes from its objective, architecture, and training strategy, not mainly from the PLR encoding strategy.

Table 6: Results of MLP and our ModernNCA with or without the PLR encoding on numerical attributes.

|  | MLP w/o PLR | MLP w/ PLR | M-NCA w/o PLR | M-NCA w/ PLR |
| --- | --- | --- | --- | --- |
| CA ↓↓\downarrow | .5074 | .4690 | .4266 | .4212 |
| AD ↑↑\uparrow | 85.80 | 86.97 | 86.77 | 87.29 |
| MIA×105absentsuperscript105\times 10^{5}↓↓\downarrow | .8973 | .8562 | .8759 | .8284 |
| PH ↑↑\uparrow | 86.88 | 87.13 | 88.41 | 89.29 |

Other Implementations of TabCon.
As mentioned in [subsection A.4](#A1.SS4 "A.4 Implementations of TabCon ‣ Appendix A Datasets ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), there are different choices for the predictors constructed based on the learned embeddings ϕ​(𝒙)italic-ϕ𝒙\phi({\bm{x}}) using Supervised Contrastive loss. We consider two variants here: the first is KNN, and the second is Logistic Regression for classification and Linear Regression for regression. We denote these two prediction methods as TabCon and TabCon (LR), respectively. We compare the results of these two variants in [Table 7](#A2.T7 "Table 7 ‣ B.1 Additional Ablation Studies ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). The results show that while different methods perform variably across datasets, their overall performance is comparable. Given that KNN has a lower training overhead, we use it as the default configuration.

Table 7: Results of TabCon variants with different ways to make predictions. The default choice of TabCon utilizes KNN based on the learned embeddings.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | CA↓↓\downarrow | AD↑↑\uparrow | MIA×105absentsuperscript105\times 10^{5}↓↓\downarrow | PH↑↑\uparrow |
| TabCon | .4581 | 86.97 | .9322 | 87.88 |
| TabCon (LR) | .4724 | 87.09 | .9017 | 86.69 |

Visualization Results.
To better analyze the properties of ModernNCA, we visualize the learned embeddings ϕ​(𝒙)italic-ϕ𝒙\phi({\bm{x}}) of ModernNCA, TabCon, and TabR using TSNE [[55](#bib.bib55)]. As shown in [Figure 5](#A2.F5 "Figure 5 ‣ B.2 Run-time and Memory Usage Estimation ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"), all deep tabular methods transform the embedding spaces to be more helpful for classification or regression compared to the raw features.
The embedding space learned by TabCon clusters samples of the same class together and separates samples of different classes, often clustering same-class instances into a single cluster. However, it still struggles with some hard-to-distinguish samples.
TabR and ModernNCA, on the other hand, divide samples of the same class into multiple clusters, ensuring that similar samples are positioned closer to each other. This strategy aligns with the prediction mechanism of KNN, where good performance is achieved by clustering instances with similar neighbors together rather than into a single cluster.
The embedding space learned by ModernNCA is more discriminative than that learned by TabR. The main reason is that TabR leverages a transformer-like architecture to modify the embedding for each instance before making predictions, making the learned embedding space less discriminative compared to ModernNCA.

### B.2 Run-time and Memory Usage Estimation

We make a run-time and memory usage comparison in  [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). Here are the steps that we take to perform the estimation. First, we tuned all models on the validation set for 100 iterations, saving the optimal parameters ever found. Next, we ran the models for 15 iterations with the tuned parameters and saved the best checkpoint on the validation set. The run-time for the models was estimated using the average time taken by the tuned model to run one seed in the training and validation stage. The model size was estimated using the size of the saved checkpoint. Here we present the average results of run-time and memory usage estimation across our benchmark datasets in [Table 8](#A2.T8 "Table 8 ‣ B.2 Run-time and Memory Usage Estimation ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").

Table 8: Training time and memory usage estimation for different tuned models on the benchmark datasets. The average rank represents the mean performance ranking of these models based on the performance metrics (RMSE for regression and accuracy for classification).

| Model | M-NCA | TabCon | MLP | MLPPLR | FT-T | TabR | XGBoost | CatBoost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training Time (s) | 87.5 | 53.16 | 30.36 | 38.94 | 111.91 | 173.34 | 4.53 | 20.48 |
| Size (MB) | 3.36 | 8.54 | 2.92 | 9.53 | 3.19 | 9.13 | 7.44 | 6.92 |
| Average Rank | 2.41 | 5.80 | 7.63 | 4.83 | 5.32 | 3.37 | 3.62 | 3.02 |



![Refer to caption](/html/2407.03257/assets/x10.png)

(a) AD ↑↑\uparrow Raw Feature

![Refer to caption](/html/2407.03257/assets/x11.png)

(b) AD ↑↑\uparrow TabR

![Refer to caption](/html/2407.03257/assets/x12.png)

(c) AD ↑↑\uparrow ModernNCA

![Refer to caption](/html/2407.03257/assets/x13.png)

(c) AD ↑↑\uparrow TabCon

![Refer to caption](/html/2407.03257/assets/x14.png)

(a) PH ↑↑\uparrow Raw Feature

![Refer to caption](/html/2407.03257/assets/x15.png)

(b) PH ↑↑\uparrow TabR

![Refer to caption]()

(c) PH ↑↑\uparrow ModernNCA

![Refer to caption](/html/2407.03257/assets/x17.png)

(c) PH ↑↑\uparrow TabCon

![Refer to caption](/html/2407.03257/assets/x18.png)

(a) CA↓↓\downarrow Raw Feature

![Refer to caption](/html/2407.03257/assets/x19.png)

(b) CA↓↓\downarrow TabR

![Refer to caption](/html/2407.03257/assets/x20.png)

(c) CA↓↓\downarrow ModernNCA

![Refer to caption](/html/2407.03257/assets/x21.png)

(c) CA↓↓\downarrow TabCon

![Refer to caption](/html/2407.03257/assets/x22.png)

(a) MIA↓↓\downarrow Raw Feature

![Refer to caption](/html/2407.03257/assets/x23.png)

(b) MIA↓↓\downarrow TabR

![Refer to caption](/html/2407.03257/assets/x24.png)

(c) MIA↓↓\downarrow ModernNCA

![Refer to caption](/html/2407.03257/assets/x25.png)

(c) MIA↓↓\downarrow TabCon

Figure 5: Visualization of the embedding space of different methods.

### B.3 Full Results with Standard Deviations

In this section, we present the whole result with standard deviation for the datasets described in [Table 5](#A1.T5 "Table 5 ‣ A.2 Dataset Information ‣ Appendix A Datasets ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). The results for the classification tasks are reported in [Table 9](#A2.T9 "Table 9 ‣ B.3 Full Results with Standard Deviations ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later"). The results for regression tasks are shown in [Table 10](#A2.T10 "Table 10 ‣ B.3 Full Results with Standard Deviations ‣ Appendix B Additional Experiments ‣ Modern Neighborhood Components Analysis: A Deep Tabular Baseline Two Decades Later").

Table 9: Test accuracy and standard deviations for 12 classification datasets across 15 seeds. The best results are highlighted in bold.

| ↑↑\uparrow | M-NCA | TabCon | XGBoost | CatBoost | MLP | FT-T | TabR | KNN | MLPPLR | PTaRL | LMNN | NCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD | 87.29 | 86.97 | 87.21 | 87.48 | 85.80 | 85.89 | 86.86 | 84.61 | 86.97 | 85.53 | 83.21 | 83.87 |
|  | ±plus-or-minus\pm 0.07 | ±plus-or-minus\pm 0.11 | ±plus-or-minus\pm 0.05 | ±plus-or-minus\pm 0.06 | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.16 | ±plus-or-minus\pm 0.10 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.21 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| CU | 79.99 | 77.72 | 81.73 | 81.03 | 75.54 | 80.65 | 79.61 | 70.08 | 80.75 | 75.89 | 72.12 | 73.78 |
|  | ±plus-or-minus\pm 0.55 | ±plus-or-minus\pm 1.85 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.38 | ±plus-or-minus\pm 0.57 | ±plus-or-minus\pm 0.61 | ±plus-or-minus\pm 1.18 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.67 | ±plus-or-minus\pm 0.62 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| EL | 96.12 | 87.44 | 92.06 | 91.69 | 85.49 | 87.50 | 96.46 | 84.27 | 86.93 | 85.10 | 83.56 | 85.15 |
|  | ±plus-or-minus\pm 0.11 | ±plus-or-minus\pm 0.23 | ±plus-or-minus\pm 0.21 | ±plus-or-minus\pm 0.18 | ±plus-or-minus\pm 0.28 | ±plus-or-minus\pm 0.25 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.61 | ±plus-or-minus\pm 0.39 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| EY | 99.41 | 82.88 | 71.02 | 71.88 | 61.29 | 71.49 | 98.20 | 59.46 | 74.27 | 60.13 | 54.98 | 66.50 |
|  | ±plus-or-minus\pm 0.33 | ±plus-or-minus\pm 9.49 | ±plus-or-minus\pm 0.42 | ±plus-or-minus\pm 0.34 | ±plus-or-minus\pm 0.98 | ±plus-or-minus\pm 0.85 | ±plus-or-minus\pm 0.54 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 2.28 | ±plus-or-minus\pm 1.20 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| HE | 39.84 | 38.32 | 37.86 | 38.26 | 38.34 | 38.49 | 40.80 | 32.91 | 38.95 | 37.64 | 29.36 | 31.30 |
|  | ±plus-or-minus\pm 0.24 | ±plus-or-minus\pm 0.23 | ±plus-or-minus\pm 0.09 | ±plus-or-minus\pm 0.11 | ±plus-or-minus\pm 0.25 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.18 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.27 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| HI | 73.09 | 72.36 | 72.73 | 72.66 | 72.29 | 72.92 | 72.85 | 66.78 | 72.64 | 72.48 | 61.44 | 66.94 |
|  | ±plus-or-minus\pm 0.13 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.08 | ±plus-or-minus\pm 0.11 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.13 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.21 | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| IP | 87.96 | 87.65 | 88.64 | 87.74 | 84.86 | 84.95 | 87.23 | 83.72 | 87.84 | 84.83 | 83.81 | 84.10 |
|  | ±plus-or-minus\pm 0.28 | ±plus-or-minus\pm 0.56 | ±plus-or-minus\pm 0.20 | ±plus-or-minus\pm 0.08 | ±plus-or-minus\pm 0.42 | ±plus-or-minus\pm 0.34 | ±plus-or-minus\pm 0.76 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.33 | ±plus-or-minus\pm 0.28 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| JA | 74.08 | 72.17 | 72.55 | 72.39 | 71.97 | 72.38 | 73.44 | 65.67 | 72.10 | 71.52 | 59.12 | 64.64 |
|  | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.37 | ±plus-or-minus\pm 0.08 | ±plus-or-minus\pm 0.09 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.16 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.19 | ±plus-or-minus\pm 0.27 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| MA | 87.90 | 87.22 | 87.67 | 87.87 | 87.08 | 87.73 | 87.91 | 84.65 | 87.17 | 86.94 | 83.68 | 84.73 |
|  | ±plus-or-minus\pm 0.24 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.32 | ±plus-or-minus\pm 0.28 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.22 | ±plus-or-minus\pm 0.17 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| OT | 82.37 | 78.94 | 82.45 | 82.21 | 81.47 | 80.82 | 82.38 | 78.15 | 81.24 | 81.20 | 77.37 | 78.85 |
|  | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.49 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.12 | ±plus-or-minus\pm 0.15 | ±plus-or-minus\pm 0.28 | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.14 | ±plus-or-minus\pm 0.21 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| PH | 89.29 | 87.88 | 87.66 | 87.97 | 86.88 | 87.86 | 89.10 | 86.86 | 87.13 | 85.54 | 85.75 | 87.05 |
|  | ±plus-or-minus\pm 0.63 | ±plus-or-minus\pm 0.45 | ±plus-or-minus\pm 0.53 | ±plus-or-minus\pm 0.49 | ±plus-or-minus\pm 0.72 | ±plus-or-minus\pm 0.40 | ±plus-or-minus\pm 0.49 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.49 | ±plus-or-minus\pm 0.63 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| WI | 74.74 | 73.72 | 74.39 | 74.59 | 72.50 | 72.64 | 74.22 | 77.10 | 72.50 | 72.16 | 73.58 | 72.60 |
|  | ±plus-or-minus\pm 1.55 | ±plus-or-minus\pm 1.55 | ±plus-or-minus\pm 0.85 | ±plus-or-minus\pm 0.91 | ±plus-or-minus\pm 0.50 | ±plus-or-minus\pm 0.80 | ±plus-or-minus\pm 1.29 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 1.35 | ±plus-or-minus\pm 0.82 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm 0.00 |
| rank | 2.000 | 5.917 | 3.833 | 3.750 | 8.000 | 5.417 | 3.083 | 10.000 | 5.667 | 9.250 | 11.333 | 9.750 |




Table 10: Test RMSE and standard deviations for 10 regression datasets across 15 seeds. The best results are highlighted in bold.

| ↓↓\downarrow | M-NCA | TabCon | XGBoost | CatBoost | MLP | FT-T | TabR | KNN | MLPPLR | PTaRL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI×10−3absentsuperscript103\times 10^{-3} | .1532 | .1522 | .1527 | .1465 | .1558 | .1565 | .1546 | .2431 | .1522 | .1563 |
|  | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .002 |
| BI×102absentsuperscript102\times 10^{2} | .7057 | .7352 | .7180 | .7264 | .7773 | .7466 | .6657 | 1.087 | .7155 | .7193 |
|  | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .011 | ±plus-or-minus\pm .006 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .011 | ±plus-or-minus\pm .006 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .005 |
| CA | .4212 | .4581 | .4328 | .4360 | .5074 | .4701 | .4157 | .5843 | .4690 | .5086 |
|  | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .010 |
| CM×103absentsuperscript103\times 10^{3} | .4618 | .5011 | .4809 | .4929 | .5131 | .5187 | .5091 | .8754 | .4655 | .5211 |
|  | ±plus-or-minus\pm .010 | ±plus-or-minus\pm .012 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .013 | ±plus-or-minus\pm .006 | ±plus-or-minus\pm .015 | ±plus-or-minus\pm .023 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .020 | ±plus-or-minus\pm .004 |
| CP×10absent10\times 10 | .2381 | .2473 | .2404 | .2365 | .2490 | .2310 | .2337 | .8825 | .2468 | .2487 |
|  | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .006 | ±plus-or-minus\pm .003 |
| FO×10absent10\times 10 | .7272 | .7436 | .7384 | .7393 | .7900 | .7885 | .7398 | .8140 | .7428 | .7896 |
|  | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .013 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .001 |
| HO×105absentsuperscript105\times 10^{5} | .3041 | .3156 | .3026 | .3051 | .3161 | .3074 | .3133 | .3663 | .3093 | .3149 |
|  | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .001 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .006 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .003 |
| HOU×105absentsuperscript105\times 10^{5} | .4291 | .4963 | .4797 | .4546 | .5221 | .4791 | .4149 | .6889 | .4921 | .5118 |
|  | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .020 | ±plus-or-minus\pm .008 | ±plus-or-minus\pm .002 | ±plus-or-minus\pm .009 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .004 | ±plus-or-minus\pm .004 |
| LA×103absentsuperscript103\times 10^{3} | .4135 | .4529 | .4448 | .4506 | .4789 | .4609 | .4606 | .4968 | .4753 | .4777 |
|  | ±plus-or-minus\pm .014 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .003 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .005 | ±plus-or-minus\pm .007 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .013 | ±plus-or-minus\pm .005 |
| MIA×105absentsuperscript105\times 10^{5} | .8284 | .9322 | .8740 | .8034 | .8973 | .8717 | .8585 | 1.508 | .8562 | .9302 |
|  | ±plus-or-minus\pm .017 | ±plus-or-minus\pm .014 | ±plus-or-minus\pm .016 | ±plus-or-minus\pm .008 | ±plus-or-minus\pm .011 | ±plus-or-minus\pm .023 | ±plus-or-minus\pm .012 | ±plus-or-minus\pm 0.00 | ±plus-or-minus\pm .012 | ±plus-or-minus\pm .017 |
| rank | 2.200 | 6.100 | 3.500 | 3.100 | 8.300 | 5.900 | 3.600 | 10.000 | 4.500 | 7.800 |

[◄](/html/2407.03256)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2407.03257)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2407.03257)
[View original  
on arXiv](https://arxiv.org/abs/2407.03257)[►](/html/2407.03258)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Aug 5 17:25:15 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
