---
arxiv: '2012.06678'
authors:
- Xin Huang
- Ashish Khetan
- Milan Cvitkovic
- Zohar Karnin
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TabTransformer: Tabular Data Modeling Using Contextual Embeddings'
url: http://arxiv.org/abs/2012.06678v1
year: 2020
---

[2012.06678] TabTransformer: Tabular Data Modeling Using Contextual Embeddings














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



# TabTransformer: Tabular Data Modeling Using Contextual Embeddings

Xin Huang,1
Ashish Khetan, 1
Milan Cvitkovic 2
Zohar Karnin 1

###### Abstract

We propose TabTransformer, a novel deep tabular data modeling architecture for supervised and semi-supervised learning. The TabTransformer is built upon self-attention based Transformers.
The Transformer layers transform the embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy.
Through extensive experiments on fifteen publicly available datasets,
we show that
the TabTransformer outperforms the
state-of-the-art deep learning methods for tabular data
by at least 1.0%percent1.01.0\% on mean AUC, and matches the performance of tree-based ensemble models.
Furthermore,
we demonstrate that the contextual embeddings learned from TabTransformer are highly robust against both missing and noisy data features, and provide better interpretability.
Lastly, for the semi-supervised setting we develop an unsupervised pre-training procedure to learn data-driven contextual embeddings, resulting in an average 2.1%percent2.12.1\% AUC lift over the state-of-the-art methods.

## 1 Introduction

Tabular data is the most common data type in many real-world applications such as recommender systems (Cheng et al. [2016](#bib.bib10)), online advertising (Song et al. [2019](#bib.bib46)), and portfolio optimization (Ban, El Karoui, and Lim [2018](#bib.bib4)). Many machine learning competitions such as Kaggle and KDD Cup are primarily designed to solve problems in tabular domain.

The state-of-the-art for modeling tabular data is tree-based ensemble methods such as the gradient boosted decision trees (GBDT) (Chen and Guestrin [2016](#bib.bib9); Prokhorenkova et al. [2018](#bib.bib42)). This is in contrast to modeling image and text data where
all the existing competitive models are based on deep learning (Sandler et al. [2018](#bib.bib45); Devlin et al. [2019](#bib.bib15)).
The tree-based ensemble models can achieve competitive prediction accuracy, are fast to train and easy to interpret. These benefits make them highly favourable among machine learning practitioners. However,
the tree-based models have several limitations in comparison to deep learning models.
(a) They are not suitable for continual training from streaming data, and do not allow efficient end-to-end learning of image/text encoders in presence of multi-modality along with tabular data.
(b) In their basic form they are not suitable for state-of-the-art semi-supervised learning methods. This is due to the fact that the basic decision tree learner does not produce reliable probability estimation to its predictions (Tanha, Someren, and Afsarmanesh [2017](#bib.bib49)).
(c) The state-of-the-art deep learning methods (Devlin et al. [2019](#bib.bib15)) to handle missing and noisy data features do not apply to them. Also, robustness of tree-based models has not been studied much in literature.

A classical and popular model that is trained using gradient descent and hence allows end-to-end learning of image/text encoders is multi-layer perceptron (MLP). The MLPs usually learn parametric embeddings to encode categorical data features. But due to their shallow architecture and context-free embeddings, they have the following limitations:
(a) neither the model nor the learned embeddings are interpretable; (b) it is not robust against missing and noisy data (Section [3.2](#S3.SS2 "3.2 The Robustness of TabTransformer ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"));
(c) for semi-supervised learning, they do not achieve competitive performance (Section [3.4](#S3.SS4 "3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings")). Most importantly, MLPs do not match the performance of tree-based models such as GBDT on most of the datasets (Arik and Pfister [2019](#bib.bib3)). To bridge this performance gap between MLP and GBDT, researchers have proposed various deep learning models (Song et al. [2019](#bib.bib46); Cheng et al. [2016](#bib.bib10); Arik and Pfister [2019](#bib.bib3); Guo et al. [2018](#bib.bib19)). Although these deep learning models achieve comparable prediction accuracy,
they do not address all the limitations of GBDT and MLP.
Furthermore, their comparisons are done in a limited setting of a handful of datasets. In particular, in Section [3.3](#S3.SS3 "3.3 Supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") we show that when compared to standard GBDT on a large collection of datasets, GBDT perform significantly better than these recent models.

In this paper, we propose TabTransformer to address the limitations of MLPs and existing deep learning models, while bridging the performance gap between MLP and GBDT. We establish performance gain of TabTransformer through extensive experiments on fifteen publicly available datasets.

The TabTransformer is built upon Transformers (Vaswani et al. [2017](#bib.bib51)) to learn efficient contextual embeddings of categorical features.
Different from tabular domain, the application of embeddings has been studied extensively in NLP.
The use of embeddings to encode words in a dense low dimensional space is prevalent in natural language processing. Beginning from Word2Vec (Rong [2014](#bib.bib43)) with the context-free word embeddings to BERT (Devlin et al. [2019](#bib.bib15)) which provides the contextual word-token embeddings, embeddings have been widely studied and applied in practice in NLP. In comparison to context-free embeddings, the contextual embedding based models (Mikolov et al. [2011](#bib.bib36); Huang, Xu, and Yu [2015](#bib.bib22); Devlin et al. [2019](#bib.bib15)) have achieved tremendous success. In particular, self-attention based Transformers (Vaswani et al. [2017](#bib.bib51)) have become a standard component of NLP models to achieve state-of-the-art performance. The effectiveness and interpretability of contextual embeddings generated by Transformers have been also well studied (Coenen et al. [2019](#bib.bib12); Brunner et al. [2019](#bib.bib6)).

Motivated by the successful applications of Transformers in NLP,
we adapt them in tabular domain.
In particular, TabTransformer applies a sequence of multi-head attention-based Transformer layers on parametric embeddings to transform them into contextual embeddings, bridging the performance gap between baseline MLP and GBDT models. We investigate the effectiveness and interpretability of the resulting contextual embeddings generated by the Transformers. We find that highly correlated features (including feature pairs in the same column and cross column) result in embedding vectors that are close together in Euclidean distance,
whereas no such pattern exists in context-free embeddings learned in a baseline MLP model. We also study the robustness of the TabTransformer against random missing and noisy data. The contextual embeddings make them highly robust in comparison to MLPs.

Furthermore, many existing deep learning models for tabular data are designed for supervised learning scenario but few are for semi-supervised leanring (SSL). Unfortunately, the state-of-art SSL models developed in computer vision (Voulodimos et al. [2018](#bib.bib52); Kendall and Gal [2017](#bib.bib30)) and NLP (Vaswani et al. [2017](#bib.bib51); Devlin et al. [2019](#bib.bib15)) cannot be easily extended to tabular domain. Motivated by such challenges, we exploit pre-training methodologies from the language models and propose a semi-supervised learning approach for pre-training Transformers of our TabTransformer model using unlabeled data.

One of the key benefits of our proposed method for semi-supervised learning is the two independent training phases:
a costly pre-training phase on unlabeled data and a lightweight fine-tuning phase on labeled data.
This differs from many state-of-the-art semi-supervised methods (Chapelle, Scholkopf, and Zien [2009](#bib.bib7); Oliver et al. [2018](#bib.bib40); Stretcu et al. [2019](#bib.bib47)) that require a single training job including both the labeled and unlabeled data.
The separated training procedure benefits the scenario where the model needs to be pretrained once but fine-tuned multiple times for multiple target variables.
This scenario is in fact quite common in the industrial setting as companies tend to have one large dataset (e.g. describing customers/products) and are interested in applying multiple analyses on this data. To summarize, we provide the following contributions:

1. 1.

   We propose TabTransformer, an architecture that provides and exploits contextual embeddings of categorical features. We provide extensive empirical evidence showing TabTransformer
   is superior to both a baseline MLP and recent deep networks for tabular data while matching the performance of tree-based ensemble models (GBDT).
2. 2.

   We investigate the resulting contextual embeddings and highlight their interpretability, contrasted to parametric context-free embeddings achieved by existing art.
3. 3.

   We demonstrate the robustness of TabTransformer against noisy and missing data.
4. 4.

   We provide and extensively study a two-phase pre-training then fine-tune procedure for tabular data, beating the state-of-the-art performance of semi-supervised learning methods.

## 2 The TabTransformer

The TabTransformer architecture comprises a column embedding layer, a stack of N𝑁N Transformer layers, and a multi-layer perceptron. Each Transformer layer (Vaswani et al. [2017](#bib.bib51)) consists of a multi-head self-attention layer followed by a position-wise feed-forward layer. The architecture of TabTransformer is shown below in Figure [1](#S2.F1 "Figure 1 ‣ 2 The TabTransformer ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

![Refer to caption](/html/2012.06678/assets/TabIllustration2.png)


Figure 1: The architecture of TabTransformer.

Let (𝒙,y)𝒙𝑦(\bm{x},y) denote a feature-target pair, where 𝒙≡{𝒙cat,𝒙cont}𝒙subscript𝒙catsubscript𝒙cont\bm{x}\equiv\{\bm{x}\_{\text{cat}},\bm{x}\_{\text{cont}}\}. The 𝒙catsubscript𝒙cat\bm{x}\_{\text{cat}} denotes all the categorical features and 𝒙cont∈ℝcsubscript𝒙contsuperscriptℝ𝑐\bm{x}\_{\text{cont}}\in\mathbb{R}^{c} denotes all of the c𝑐c continuous features. Let 𝒙cat≡{x1,x2,⋯,xm}subscript𝒙catsubscript𝑥1subscript𝑥2⋯subscript𝑥𝑚\bm{x}\_{\text{cat}}\equiv\{x\_{1},x\_{2},\cdots,x\_{m}\} with each xisubscript𝑥𝑖x\_{i} being a categorical feature, for i∈{1,⋯,m}𝑖1⋯𝑚i\in\{1,\cdots,m\}.

We embed each of the xisubscript𝑥𝑖x\_{i} categorical features into a parametric embedding of dimension d𝑑d using Column embedding, which is explained below in detail. Let 𝒆ϕi​(xi)∈ℝdsubscript𝒆subscriptitalic-ϕ𝑖subscript𝑥𝑖superscriptℝ𝑑\bm{e}\_{{\phi}\_{i}}(x\_{i})\in\mathbb{R}^{d} for i∈{1,⋯,m}𝑖1⋯𝑚i\in\{1,\cdots,m\} be the embedding of the xisubscript𝑥𝑖x\_{i} feature, and 𝑬ϕ​(𝒙cat)={𝒆ϕ1​(x1),⋯,𝒆ϕm​(xm)}subscript𝑬italic-ϕsubscript𝒙catsubscript𝒆subscriptitalic-ϕ1subscript𝑥1⋯subscript𝒆subscriptitalic-ϕ𝑚subscript𝑥𝑚\bm{E}\_{\phi}(\bm{x}\_{\text{cat}})=\{\bm{e}\_{{\phi}\_{1}}(x\_{1}),\cdots,\bm{e}\_{{\phi}\_{m}}(x\_{m})\} be the set of embeddings for all the categorical features.

Next, these parametric embeddings 𝑬ϕ​(𝒙cat)subscript𝑬italic-ϕsubscript𝒙cat\bm{E}\_{\phi}(\bm{x}\_{\text{cat}}) are inputted to the first Transformer layer. The output of the first Transformer layer is inputted to the second layer Transformer, and so forth.
Each parametric embedding is transformed into contextual embedding when outputted from the top layer Transformer, through successive aggregation of context from other embeddings. We denote the sequence of Transformer layers as a function fθsubscript𝑓𝜃f\_{\theta}. The function fθsubscript𝑓𝜃f\_{\theta} operates on parametric embeddings {𝒆ϕ1​(x1),⋯,𝒆ϕm​(xm)}subscript𝒆subscriptitalic-ϕ1subscript𝑥1⋯subscript𝒆subscriptitalic-ϕ𝑚subscript𝑥𝑚\{\bm{e}\_{{\phi}\_{1}}(x\_{1}),\cdots,\bm{e}\_{{\phi}\_{m}}(x\_{m})\} and returns the corresponding contextual embeddings {𝒉1,⋯,𝒉m}subscript𝒉1⋯subscript𝒉𝑚\{\bm{h}\_{1},\cdots,\bm{h}\_{m}\} where 𝒉i∈ℝdsubscript𝒉𝑖superscriptℝ𝑑\bm{h}\_{i}\in\mathbb{R}^{d} for i∈{1,⋯,m}𝑖1⋯𝑚i\in\{1,\cdots,m\}.

The contextual embeddings {𝒉1,⋯,𝒉m}subscript𝒉1⋯subscript𝒉𝑚\{\bm{h}\_{1},\cdots,\bm{h}\_{m}\} are concatenated along with the continuous features 𝒙contsubscript𝒙cont\bm{x}\_{\text{cont}} to form a vector of dimension (d×m+c)𝑑𝑚𝑐(d\times m+c). This vector is inputted to an MLP, denoted by g𝝍subscript𝑔𝝍g\_{\bm{\psi}}, to predict the target y𝑦y. Let H𝐻H be the cross-entropy for classification tasks and mean square error for regression tasks. We minimize the following loss function ℒ​(𝒙,y)ℒ𝒙𝑦\mathcal{L}(\bm{x},y) to learn all the TabTransformer parameters in an end-to-end learning by the first-order gradient methods. The TabTransformer parameters include ϕbold-italic-ϕ\bm{\phi} for column embedding, 𝜽𝜽\bm{\theta} for Transformer layers, and 𝝍𝝍\bm{\psi} for the top MLP layer.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝒙,y)≡H​(g𝝍​(f𝜽​(𝑬ϕ​(𝒙cat)),𝒙cont),y).ℒ𝒙𝑦𝐻subscript𝑔𝝍subscript𝑓𝜽subscript𝑬italic-ϕsubscript𝒙catsubscript𝒙cont𝑦\displaystyle\mathcal{L}(\bm{x},y)\equiv H(g\_{\bm{\psi}}(f\_{\bm{\bm{\theta}}}(\bm{E}\_{\phi}(\bm{x}\_{\text{cat}})),\bm{x}\_{\text{cont}}),y)\,. |  | (1) |

Below, we explain the Transformer layers and column embedding.

#### Transformer.

A Transformer (Vaswani et al. [2017](#bib.bib51)) consists of a multi-head self-attention layer followed by a position-wise feed-forward layer, with element-wise addition and layer-normalization being done after each layer.
A self-attention layer comprises three parametric matrices - Key, Query and Value. Each input embedding is projected on to these matrices, to generate their key, query and value vectors.
Formally, let K∈ℝm×k𝐾superscriptℝ𝑚𝑘K\in\mathbb{R}^{m\times k}, Q∈ℝm×k𝑄superscriptℝ𝑚𝑘Q\in\mathbb{R}^{m\times k} and V∈ℝm×v𝑉superscriptℝ𝑚𝑣V\in\mathbb{R}^{m\times v} be the matrices comprising key, query and value vectors of all the embeddings, respectively, and m𝑚m be the number of embeddings inputted to the Transformer, k𝑘k and v𝑣v be the dimensions of the key and value vectors, respectively. Every input embedding attends to all other embeddings through a Attention head, which is computed as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Attention​(K,Q,V)=A⋅V,Attention𝐾𝑄𝑉⋅𝐴𝑉\displaystyle\text{Attention}(K,Q,V)=A\cdot V, |  | (2) |

where A=softmax​((Q​KT)/k)𝐴softmax𝑄superscript𝐾𝑇𝑘A=\text{softmax}({(QK^{T})}/{\sqrt{k}}). For each embedding, the attention matrix A∈ℝm×m𝐴superscriptℝ𝑚𝑚A\in\mathbb{R}^{m\times m} calculates how much it attends to other embeddings, thus transforming the embedding into contextual one. The output of the attention head of dimension v𝑣v is projected back to the embedding of dimension d𝑑d through a fully connected layer, which in turn is passed through two position-wise feed-forward layers. The first layer expands the embedding to four times its size and the second layer projects it back to its original size.

#### Column embedding.

For each categorical feature (column) i𝑖i, we have an embedding lookup table 𝒆ϕi(.)\bm{e}\_{{\phi}\_{i}}(.), for i∈{1,2,…,m}𝑖12…𝑚i\in\{1,2,...,m\}. For i𝑖ith feature with disubscript𝑑𝑖d\_{i} classes, the embedding table 𝒆ϕi(.)\bm{e}\_{{\phi}\_{i}}(.) has (di+1)subscript𝑑𝑖1(d\_{i}+1) embeddings where the additional embedding corresponds to a missing value.
The embedding for the encoded value xi=j∈[0,1,2,..,di]x\_{i}=j\in[0,1,2,..,d\_{i}] is 𝒆ϕi​(j)=[𝒄ϕi,𝒘ϕi​j]subscript𝒆subscriptitalic-ϕ𝑖𝑗subscript𝒄subscriptitalic-ϕ𝑖subscript𝒘subscriptitalic-ϕ𝑖𝑗\bm{e}\_{{\phi}\_{i}}(j)=[\bm{c}\_{\phi\_{i}},\bm{w}\_{\phi\_{ij}}], where 𝒄ϕi∈ℝℓ,𝒘ϕi​j∈ℝd−ℓformulae-sequencesubscript𝒄subscriptitalic-ϕ𝑖superscriptℝℓsubscript𝒘subscriptitalic-ϕ𝑖𝑗superscriptℝ𝑑ℓ\bm{c}\_{\phi\_{i}}\in\mathbb{R}^{\ell},\bm{w}\_{\phi\_{ij}}\in\mathbb{R}^{d-\ell}.
The dimension of 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}}, ℓℓ\ell, is a hyper-parameter. The unique identifier 𝒄ϕi∈ℝℓsubscript𝒄subscriptitalic-ϕ𝑖superscriptℝℓ\bm{c}\_{\phi\_{i}}\in\mathbb{R}^{\ell} distinguishes the classes in column i𝑖i from those in the other columns.

The use of unique identifier is new and is particularly designed for tabular data.
Rather in language modeling, embeddings are element-wisely added with the positional encoding of the word in the sentence. Since, in tabular data, there is no ordering of the features, we do not use positional encodings.
An ablation study on different embedding strategies is given in Appendix [A](#A1 "Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). The strategies include both different choices for ℓ,d

ℓ𝑑\ell,d and element-wise adding the unique identifier and feature-value specific embeddings rather than concatenating them.

#### Pre-training the Embeddings.

The contextual embeddings explained above are learned in end-to-end supervised training using labeled examples. For a scenario, when there are a few labeled examples and a large number of unlabeled examples,
we introduce a pre-training procedure to train the Transformer layers using unlabeled data. This is followed by fine-tuning of the pre-trained Transformer layers along with the top MLP layer using the labeled data. For fine-tuning, we use the supervised loss defined in Equation ([1](#S2.E1 "In 2 The TabTransformer ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings")).

We explore two different types of pre-training procedures, the masked language modeling (MLM) (Devlin et al. [2019](#bib.bib15)) and the replaced token detection (RTD) (Clark et al. [2020](#bib.bib11)). Given an input 𝒙cat={x1,x2,…,xm}subscript𝒙catsubscript𝑥1subscript𝑥2…subscript𝑥𝑚\bm{x}\_{\text{cat}}=\{x\_{1},x\_{2},...,x\_{m}\}, MLM randomly selects k%percent𝑘k\% features from index 111 to m𝑚m and masks them as missing. The Transformer layers along with the column embeddings are trained by minimizing cross-entropy loss of a multi-class classifier that tries to predict the original features of the masked features, from the contextual embedding outputted from the top-layer Transformer.

Instead of masking features, RTD replaces the original feature by a random value of that feature. Here, the loss is minimized for a binary classifier that tries to predict whether or not the feature has been replaced.
The RTD procedure as proposed in (Clark et al. [2020](#bib.bib11)) uses auxiliary generator for sampling a subset of features that a feature should be replaced with. The reason they used an auxiliary encoder network as the generator is that there are tens of thousands of tokens in language data and a uniformly random token is too easy to detect. In contrast, (a) the number of classes within each categorical feature is typically limited; (b) a different binary classifier is defined for each column rather than a shared one, as each column has its own embedding lookup table.
We name the two pre-training methods as TabTransformer-MLM and TabTransformer-RTD. In our experiments, the replacement value k𝑘k is set to 303030. An ablation study on k𝑘k is given in Appendix [A](#A1 "Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

## 3 Experiments

#### Data.

We evaluate TabTransformer and baseline models on 151515 publicly available binary classification datasets from the UCI repository (Dua and Graff [2017](#bib.bib16)), the AutoML Challenge (Guyon et al. [2019](#bib.bib20)), and Kaggle (Kaggle, Inc. [2017](#bib.bib27)) for both supervised and semi-supervised learning. Each dataset is divided into five cross-validation splits. The training/validation/testing proportion of the data for each split are 65/15/20%6515percent2065/15/20\%. The number of categorical features across dataset ranges from 222 to 136136136. In the semi-supervised experiments, for each dataset and split, the first p𝑝p observations in the training data are marked as the labeled data and the remaining training data as the unlabeled set. The value of p𝑝p is chosen as 505050, 200200200, and 500500500, corresponding to 333 different scenarios.
In the supervised experiments, each training dataset is fully labeled. Summary statistics of the all the datasets are provided in Table [8](#A3.T8 "Table 8 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), [9](#A3.T9 "Table 9 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") in Appendix [C](#A3 "Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

#### Setup.

For the TabTransformer, the hidden (embedding) dimension, the number of layers and the number of attention heads are fixed to 323232, 666, and 888 respectively. The MLP layer sizes are set to {4×l,2×l}4𝑙2𝑙\{4\times l,2\times l\}, where l𝑙l is the size of its input.
For hyperparameter optimization (HPO), each model is given 202020 HPO rounds for each cross-validation split.
For evaluation metrics, we use the Area under the curve (AUC) (Bradley [1997](#bib.bib5)).
Note, the pre-training is only applied in semi-supervised scenario. We do not find much benefit in using it when the entire data is labeled. Its benefit is evident when there is a large number of unlabeled examples and a few labeled examples. Since in this scenario the pre-training provides a representation of the data that could not have been learned based only on the labeled examples.

The experiment section is organized as follows. In Section [3.1](#S3.SS1 "3.1 The Effectiveness of the Transformer Layers ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), we first demonstrate the effectiveness of the attention-based Transformer by comparing our model with the one without the Transformers (equivalently an MLP model). In Section [3.2](#S3.SS2 "3.2 The Robustness of TabTransformer ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), we illustrate the robustness of TabTransformer against noisy and missing data. Finally, extensive evaluation on various methods are conducted in Section [3.3](#S3.SS3 "3.3 Supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") for supervised learning, and in Section [3.4](#S3.SS4 "3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") for semi-supervised learning.

### 3.1 The Effectiveness of the Transformer Layers

First, a comparison between TabTransformers and the baseline MLP is conducted in a supervised learning scenario. We remove the Transformer layers f𝜽subscript𝑓𝜽f\_{\bm{\theta}} from the architecture, fix the rest of the components, and compare it with the original TabTransformer. The model without the attention-based Transformer layers is equivalently an MLP. The dimension of embeddings d𝑑d for categorical features is set as 323232 for both models. The comparison results over 151515 datasets are presented in Table [1](#S3.T1 "Table 1 ‣ 3.1 The Effectiveness of the Transformer Layers ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). The TabTransformer with the Transformer layers outperforms the baseline MLP on 141414 out of 151515 datasets with an average 1.0%percent1.01.0\% gain in AUC.

Table 1: Comparison between TabTransfomers and the baseline MLP. The evaluation metric is AUC in percentage.

| Dataset | Baseline MLP | TabTransformer | Gain (%) |
| --- | --- | --- | --- |
| albert | 74.0 | 75.7 | 1.7 |
| 1995\_income | 90.5 | 90.6 | 0.1 |
| dota2games | 63.1 | 63.3 | 0.2 |
| hcdr\_main | 74.3 | 75.1 | 0.8 |
| adult | 72.5 | 73.7 | 1.2 |
| bank\_marketing | 92.9 | 93.4 | 0.5 |
| blastchar | 83.9 | 83.5 | -0.4 |
| insurance\_co | 69.7 | 74.4 | 4.7 |
| jasmine | 85.1 | 85.3 | 0.2 |
| online\_shoppers | 91.9 | 92.7 | 0.8 |
| philippine | 82.1 | 83.4 | 1.3 |
| qsar\_bio | 91.0 | 91.8 | 0.8 |
| seismicbumps | 73.5 | 75.1 | 1.6 |
| shrutime | 84.6 | 85.6 | 1.0 |
| spambase | 98.4 | 98.5 | 0.1 |

Next, we take contextual embeddings from different layers of the Transformer and compute a t-SNE plot (Maaten and Hinton [2008](#bib.bib35)) to visualize their similarity in function space. More precisely, for each dataset we
take its test data,
pass their categorical features into a trained TabTransformer, and extract all contextual embeddings (across all columns) from a certain layer of the Transformer. The t-SNE algorithm is then used to reduce each embedding to a 2D point in the t-SNE plot. Figure [2](#S3.F2 "Figure 2 ‣ 3.1 The Effectiveness of the Transformer Layers ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") (left) shows the 2D visualization of embeddings from the last layer of the Transformer for dataset bank\_marketing. Each marker in the plot represents an average of 2D points over the test data points for a certain class. We can see that semantically similar classes are close with each other and form clusters in the embedding space. Each cluster is annotated by a set of labels. For example, we find that all of the client-based features (color markers) such as job, education level and martial status stay close in the center and non-client based features (gray markers) such as month (last contact month of the year), day (last contact day of the week) lie outside the central area; in the bottom cluster the embedding of owning a housing loan stays close with that of being default; over the left cluster, embeddings of being a student, martial status as single, not having a housing loan, and education level as tertiary get together; and in the right cluster, education levels are closely associated with the occupation types (Torpey and Watson [2014](#bib.bib50)).
In Figure [2](#S3.F2 "Figure 2 ‣ 3.1 The Effectiveness of the Transformer Layers ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), the center and right plots are t-SNE plots of embeddings before being passed through the Transformer and the context-free embeddings from MLP, respectively. For the embeddings before being passed into the Transformer, it starts to distinguish the non-client based features (gray markers) from the client-based features (color markers). For the embeddings from MLP, we do not observe such pattern and many categorical features which are not semantically similar are grouped together, as indicated by the annotation in the plot.

![Refer to caption](/html/2012.06678/assets/tsne_embedding_2_submission_night.png)


Figure 2: t-SNE plots of learned embeddings for categorical features on dataset BankMarketing. Left: TabTransformer-the embeddings generated from the last layer of the attention-based Transformer. Center: TabTransformer-the embeddings before being passed into the attention-based Transformer. Right: The embeddings learned from MLP.

In addition to prove the effectiveness of Transformer layers, on the test data we take all of the contextual embeddings from each Transformer layer of a trained TabTransformer, use the embeddings from each layer along with the continuous variables as features, and separately fit a linear model with target y𝑦y. Since all of the experimental datasets are for binary classification, the linear model is logistic regression.
The motivation for this evaluation is defining the success of a simple linear model as a measure of quality for the learned embeddings.

For each dataset and each layer, an average of CV-score in AUC on the test data is computed. The evaluation is conducted on the entire test data with number of data points over 9000. Figure [3](#S3.F3 "Figure 3 ‣ 3.1 The Effectiveness of the Transformer Layers ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") presents results for dataset
BankMarketing, Adult, and QSAR\_Bio. For each line, each prediction score is normalized by the “best score” from an end-to-end trained TabTransformer for the corresponding dataset. We also explore the average and maximum pooling strategy (Howard and Ruder [2018](#bib.bib21)) rather than concatenation of embeddings as the features for the linear model. The upward pattern clearly shows that embeddings becomes more effective as the Transformer layer progresses. In contrast, the embeddings from MLP (the single black markers) perform worse with a linear model. Furthermore, the last value in each line close to 1.01.01.0 indicates that a linear model with the last layer of embeddings as features can achieve reliable accuracy, which confirms our assumption.

![Refer to caption](/html/2012.06678/assets/quant-trend-3-mlp.png)


Figure 3: Predictions of liner models using features as the embeddings extracted from different Transformer layers in TabTransformer. Layer 00 corresponds to the embeddings before being passed into the Transformer layers. For each dataset, each prediction score is normalized by the “best score” from an end-to-end trained TabTransformer.

### 3.2 The Robustness of TabTransformer

We further demonstrate the robustness of TabTransformer on the noisy data and data with missing values, against the baseline MLP. We consider these two scenarios only on categorical features to specifically prove the robustness of contextual embeddings from the Transformer layers.

#### Noisy Data.

On the test examples, we firstly contaminate the data by replacing a certain number of values by randomly generated ones from the corresponding columns (features). Next, the noisy data are passed into a trained TabTransformer to compute a prediction AUC score. Results on a set of 3 different dataets are presented in Figure [4](#S3.F4 "Figure 4 ‣ Data with Missing Values. ‣ 3.2 The Robustness of TabTransformer ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). As the noisy rate increases, TabTransformer performs better in prediction accuracy and thus is more robust than MLP. In particular notice the *Blastchar* dataset where the performance is near identical with no noise, yet as the noise increases, TabTransformer becomes significantly more performant compared to the baseline.
We conjecture that the robustness comes from the contextual property of the embeddings. Despite a feature being noisy, it draws information from the correct features allowing for a certain amount of correction.

#### Data with Missing Values.

Similarly, on the test data we artificially select a number of values to be missing and send the data with missing values to a trained TabTransformer to compute the prediction score. There are two options to handle the embeddings of missing values: (1) Use the average learned embeddings over all classes in the corresponding column; (2) the embedding for the class of missing value, the additional embedding for each column mentioned in Section  [2](#S2 "2 The TabTransformer ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). Since the benchmark datasets do not contain enough missing values to effectively train the embedding in option (2), we use the average embedding in (1) for imputation. Results on the same 3 datasets are presented in Figure [5](#S3.F5 "Figure 5 ‣ Data with Missing Values. ‣ 3.2 The Robustness of TabTransformer ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). We can see the same patterns of the noisy data case, i.e. that the TabTransformer shows better stability than MLP in handling missing values.

![Refer to caption](/html/2012.06678/assets/corruption-plot.png)


Figure 4: Performance of TabTransformer and MLP with noisy data. For each dataset, each prediction score is normalized by the score of TabTransformer at 00 noise.

![Refer to caption](/html/2012.06678/assets/missing-data-plot.png)


Figure 5: Performance of TabTransformer and MLP under missing data scenario. For each dataset, each prediction score is normalized by the score of TabTransformer trained without missing values.

### 3.3 Supervised Learning

Here we compare the performance of TabTransformer against following four categories of methods: (a) Logistic regression and GBDT (b) MLP and a sparse MLP following (Morcos et al. [2019](#bib.bib37)) (c) TabNet model of Arik and Pfister ([2019](#bib.bib3)) (d) and the Variational Information Bottleneck model (VIB) of Alemi et al. ([2017](#bib.bib2)).

Results are summarized in Table [2](#S3.T2 "Table 2 ‣ 3.3 Supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). TabTransformer, MLP, and GBDT are the top 3 performers. The TabTransformer outperforms the baseline MLP with an average 1.0% gain and perform comparable with the GBDT. Furthermore, the TabTransformer is significantly better than TabNet and VIB, the recent deep networks for tabular data. For experiment and model details, see Appendix [B](#A2 "Appendix B Appendix: Experiment and Model Details ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). The models’ performances on each individual dataset are presented in Table [16](#A3.T16 "Table 16 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") and [17](#A3.T17 "Table 17 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") in Appendix [C](#A3 "Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

Table 2: Model performance in supervised learning. The evaluation metric is mean ±plus-or-minus\pm standard deviation of AUC score over the 15 datasets for each model.
Larger the number, better the result. The top 2 numbers are bold.

| Model Name | Mean AUC (%) |
| --- | --- |
| TabTransformer | 82.8±0.4plus-or-minus82.80.4\mathbf{82.8}\pm 0.4 |
| MLP | 81.8±0.4plus-or-minus81.80.481.8\pm 0.4 |
| GBDT | 82.9±0.4plus-or-minus82.90.4\mathbf{82.9}\pm 0.4 |
| Sparse MLP | 81.4±0.4plus-or-minus81.40.481.4\pm 0.4 |
| Logistic Regression | 80.4±0.4plus-or-minus80.40.480.4\pm 0.4 |
| TabNet | 77.1±0.5plus-or-minus77.10.577.1\pm 0.5 |
| VIB | 80.5±0.4plus-or-minus80.50.480.5\pm 0.4 |

### 3.4 Semi-supervised Learning

Lastly, we evaluate the TabTransformer under the semi-supervised learning scenario where few labeled training examples are available together with a significant number of unlabeled samples. Specifically, we compare our pretrained and then fine-tuned TabTransformer-RTD/MLM against following semi-supervised models: (a) Entropy Regularization (ER) (Grandvalet and Bengio [2006](#bib.bib18)) combined with MLP and TabTransformer (b) Pseudo Labeling (PL) (Lee [2013](#bib.bib32)) combined with MLP, TabTransformer, and GBDT (Jain [2017](#bib.bib26)) (c) MLP (DAE): an unsupervised pre-training method designed for deep models on tabular data: the swap noise Denoising AutoEncoder (Jahrer [2018](#bib.bib25)).

The pre-training models TabTransformer-MLM, TabTransformer-RTD and MLP (DAE)
are firstly pretrained on the entire unlabeled training data and then fine-tuned on labeled data. The semi-supervised learning methods, Pseudo Labeling and Entropy Regularization, are trained on the mix of labeled and unlabeled training data.
To better present results, we split the set of 151515 datasets into two subsets. The first set includes 666 datasets with more than 303030K data points and the second set includes remaining 999 datasets.

The results are presented in Table [3](#S3.T3 "Table 3 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") and Table [4](#S3.T4 "Table 4 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). When the number of unlabeled data is large, Table [3](#S3.T3 "Table 3 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") shows that our TabTransformer-RTD and TabTransformer-MLM significantly outperform all the other competitors. Particularly, TabTransformer-RTD/MLM improves over all the other competitors by at least 1.2%percent1.21.2\%, 2.0%percent2.02.0\% and 2.1%percent2.12.1\% on mean AUC for the scenario of 505050, 200200200, and 500500500 labeled data points respectively. The Transformer-based semi-supervised learning methods TabTransformer (ER) and TabTransformer (PL) and the tree-based semi-supervised learning method GBDT (PL) perform worse than the average of all the models. When the number of unlabeled data becomes smaller, as shown in Table [4](#S3.T4 "Table 4 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), TabTransformer-RTD still outperforms most of its competitors but with a marginal improvement.

Furthermore, we observe that when the number of unlabeled data is small as shown in Table [4](#S3.T4 "Table 4 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), TabTransformer-RTD performs better than TabTransformer-MLM, thanks to its easier pre-training task (a binary classification) than that of MLM (a multi-class classification).
This is consistent with the finding of the ELECTRA paper (Clark et al. [2020](#bib.bib11)). In Table [4](#S3.T4 "Table 4 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), with only 505050 labeled data points, MLP (ER) and MLP (PL) beat our TabTransformer-RTD/MLM. This can be attributed to the fact that there is room for improvement in our fine-tuning procedure. In particular, our approach allows to obtain informative embeddings but does not allow the weights of the classifier itself to be trained with unlabelled data. Since this issue does not occur for ER and PL, they obtain an advantage in extremely small labelled set. We point out however that this only means that the methods are complementary and mention that a possible follow up could combine the best of all approaches.

Both evaluation results, Table [3](#S3.T3 "Table 3 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") and Table [4](#S3.T4 "Table 4 ‣ 3.4 Semi-supervised Learning ‣ 3 Experiments ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), show that our
TabTransformer-RTD and Transformers-MLM models are promising in extracting useful information from unlabeled data to help supervised training, and are particularly useful when the size of unlabeled data is large. For model performance on each individual dataset see Table [10](#A3.T10 "Table 10 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), [11](#A3.T11 "Table 11 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"),
[12](#A3.T12 "Table 12 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), [13](#A3.T13 "Table 13 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"),
[14](#A3.T14 "Table 14 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"), [15](#A3.T15 "Table 15 ‣ Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings")
in Appendix [C](#A3 "Appendix C Appendix: Benchmark Dataset Information and Experiment Results ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

Table 3: Semi-supervised learning results for 888 datasets each with more than 3030{30}K data points, for different number of labeled data points. Evaluation metrics are mean AUC in percentage. Larger the number, better the result.

| # Labeled data | 505050 | 200200200 | 500500500 |
| --- | --- | --- | --- |
| TabTransformer-RTD | 66.6±0.6plus-or-minus66.60.666.6\pm 0.6 | 70.9±0.6plus-or-minus70.90.670.9\pm 0.6 | 73.1±0.6plus-or-minus73.10.6\mathbf{73.1}\pm 0.6 |
| TabTransformer-MLM | 66.8±0.6plus-or-minus66.80.6\mathbf{66.8}\pm 0.6 | 71.0±0.6plus-or-minus71.00.6\mathbf{71.0}\pm 0.6 | 72.9±0.6plus-or-minus72.90.672.9\pm 0.6 |
| MLP (ER) | 65.6±0.6plus-or-minus65.60.665.6\pm 0.6 | 69.0±0.6plus-or-minus69.00.669.0\pm 0.6 | 71.0±0.6plus-or-minus71.00.671.0\pm 0.6 |
| MLP (PL) | 65.4±0.6plus-or-minus65.40.665.4\pm 0.6 | 68.8±0.6plus-or-minus68.80.668.8\pm 0.6 | 71.0±0.6plus-or-minus71.00.671.0\pm 0.6 |
| TabTransformer (ER) | 62.7±0.6plus-or-minus62.70.662.7\pm 0.6 | 67.1±0.6plus-or-minus67.10.667.1\pm 0.6 | 69.3±0.6plus-or-minus69.30.669.3\pm 0.6 |
| TabTransformer (PL) | 63.6±0.6plus-or-minus63.60.663.6\pm 0.6 | 67.3±0.7plus-or-minus67.30.767.3\pm 0.7 | 69.3±0.6plus-or-minus69.30.669.3\pm 0.6 |
| MLP (DAE) | 65.2±0.5plus-or-minus65.20.565.2\pm 0.5 | 68.5±0.6plus-or-minus68.50.668.5\pm 0.6 | 71.0±0.6plus-or-minus71.00.671.0\pm 0.6 |
| GBDT (PL) | 56.5±0.5plus-or-minus56.50.556.5\pm 0.5 | 63.1±0.6plus-or-minus63.10.663.1\pm 0.6 | 66.5±0.7plus-or-minus66.50.766.5\pm 0.7 |




Table 4: Semi-supervised learning results for 121212 datasets each with less than 3030{30}K data points, for different number of labeled data points. Evaluation metrics are mean AUC in percentage. Larger the number, better the result.

| # Labeled data | 505050 | 200200200 | 500500500 |
| --- | --- | --- | --- |
| TabTransformer-RTD | 78.6±0.6plus-or-minus78.60.678.6\pm 0.6 | 81.6±0.5plus-or-minus81.60.5\mathbf{81.6}\pm 0.5 | 83.4±0.5plus-or-minus83.40.5\mathbf{83.4}\pm 0.5 |
| TabTransformer-MLM | 78.5±0.6plus-or-minus78.50.678.5\pm 0.6 | 81.0±0.6plus-or-minus81.00.681.0\pm 0.6 | 82.4±0.5plus-or-minus82.40.582.4\pm 0.5 |
| MLP (ER) | 79.4±0.6plus-or-minus79.40.6\mathbf{79.4}\pm 0.6 | 81.1±0.6plus-or-minus81.10.681.1\pm 0.6 | 82.3±0.6plus-or-minus82.30.682.3\pm 0.6 |
| MLP (PL) | 79.1±0.6plus-or-minus79.10.679.1\pm 0.6 | 81.1±0.6plus-or-minus81.10.681.1\pm 0.6 | 82.0±0.6plus-or-minus82.00.682.0\pm 0.6 |
| TabTransformer (ER) | 77.9±0.6plus-or-minus77.90.677.9\pm 0.6 | 81.2±0.6plus-or-minus81.20.681.2\pm 0.6 | 82.1±0.6plus-or-minus82.10.682.1\pm 0.6 |
| TabTransformer (PL) | 77.8±0.6plus-or-minus77.80.677.8\pm 0.6 | 81.0±0.6plus-or-minus81.00.681.0\pm 0.6 | 82.1±0.6plus-or-minus82.10.682.1\pm 0.6 |
| MLP (DAE) | 78.5±0.7plus-or-minus78.50.778.5\pm 0.7 | 80.7±0.6plus-or-minus80.70.680.7\pm 0.6 | 82.2±0.6plus-or-minus82.20.682.2\pm 0.6 |
| GBDT (PL) | 73.4±0.7plus-or-minus73.40.773.4\pm 0.7 | 78.8±0.6plus-or-minus78.80.678.8\pm 0.6 | 81.3±0.6plus-or-minus81.30.681.3\pm 0.6 |

## 4 Related Work

Supervised learning. Standard MLPs have been applied to tabular data for many years (De Brébisson et al. [2015](#bib.bib14)).
For deep models designed specifically for tabular data, there are deep versions of factorization machines (Guo et al. [2018](#bib.bib19); Xiao et al. [2017](#bib.bib54)), Transformers-based methods (Song et al. [2019](#bib.bib46); Li et al. [2020](#bib.bib33); Sun et al. [2019](#bib.bib48)), and deep versions of decision-tree-based algorithms (Ke et al. [2019](#bib.bib29); Yang, Morillo, and Hospedales [2018](#bib.bib55)). In particular, (Song et al. [2019](#bib.bib46)) applies one layer of multi-head attention on embeddings to learn higher order features. The higher order features are concatenated and inputted to a fully connected layer to make the final prediction. (Li et al. [2020](#bib.bib33)) use self-attention layers and track the attention scores to obtain feature importance scores. (Sun et al. [2019](#bib.bib48)) combine the Factorization Machine model with transformer mechanism. All 3 papers are focused on recommendation systems making it hard to have a clear comparison with this paper.
Other models have been designed around the purported properties of tabular data such as low-order and sparse feature interactions. These include Deep & Cross Networks (Wang et al. [2017](#bib.bib53)),
Wide & Deep Networks (Cheng et al. [2016](#bib.bib10)), TabNets (Arik and Pfister [2019](#bib.bib3)), and AdaNet (Cortes et al. [2016](#bib.bib13)).

Semi-supervised learning. (Izmailov et al. [2019](#bib.bib24)) give a semi-supervised method based on density estimation and evaluate their approach on tabular data. Pseudo labeling (Lee [2013](#bib.bib32)) is a simple, efficient and popular baseline method.
The Pseudo labeling uses the current
network to infer pseudo-labels of unlabeled examples, by choosing the most confident class. These pseudo-labels are treated like human-provided labels in the cross entropy loss.
Label propagation
(Zhu and Ghahramani [2002](#bib.bib56)),
(Iscen et al. [2019](#bib.bib23))
is a similar approach where a node’s labels propagate to all nodes according to their proximity,
and are used by the training model as if they were the true labels. Another standard method in semi-supervised learning is entropy regularization
(Grandvalet and Bengio [2005](#bib.bib17); Sajjadi, Javanmardi, and Tasdizen [2016](#bib.bib44)). It adds average per-sample entropy for the unlabeled examples to the original loss function for the labeled
examples. Another classical approach of semi-supervised learning is co-training (Nigam and Ghani [2000](#bib.bib38)). However, the recent approaches - entropy regularization and pseudo labeling - are typically better and more popular. A succinct review of semi-supervised learning methods in general can be found in (Oliver et al. [2019](#bib.bib39); Chappelle, Schölkopf, and Zien [2010](#bib.bib8)).

## 5 Conclusion

We proposed TabTransformer, a novel deep tabular data modeling architecture for supervised and semi-supervised learning. We provide extensive empirical evidence showing TabTransformer significantly outperforms MLP and recent deep networks for tabular data while matching the performance of tree-based ensemble models (GBDT).
We provide and extensively study a two-phase pre-training then fine-tune procedure for tabular data, beating the state-of-the-art performance of semi-supervised learning methods.
TabTransformer shows promising results for robustness against noisy and missing data, and interpretability of the contextual embeddings. For future work, it would be interesting to investigate them in detail.

> ## References
>
> * Alemi, Fischer, and Dillon (2018)
>
>   Alemi, A. A.; Fischer, I.; and Dillon, J. V. 2018.
>   Uncertainty in the Variational Information Bottleneck.
>   *arXiv:1807.00906 [cs, stat]*
>   URL http://arxiv.org/abs/1807.00906.
>   ArXiv: 1807.00906.
> * Alemi et al. (2017)
>
>   Alemi, A. A.; Fischer, I.; Dillon, J. V.; and Murphy, K. 2017.
>   Deep Variational Information Bottleneck.
>   *International Conference on Learning Representations*
>   abs/1612.00410.
>   URL https://arxiv.org/abs/1612.00410.
> * Arik and Pfister (2019)
>
>   Arik, S. O.; and Pfister, T. 2019.
>   TabNet: Attentive Interpretable Tabular Learning.
>   *arXiv preprint arXiv:1908.07442*
>   URL https://arxiv.org/abs/1908.07442.
> * Ban, El Karoui, and Lim (2018)
>
>   Ban, G.-Y.; El Karoui, N.; and Lim, A. E. 2018.
>   Machine learning and portfolio optimization.
>   *Management Science* 64(3): 1136–1154.
> * Bradley (1997)
>
>   Bradley, A. P. 1997.
>   The use of the area under the ROC curve in the evaluation of machine
>   learning algorithms.
>   *Pattern recognition* 30(7): 1145–1159.
> * Brunner et al. (2019)
>
>   Brunner, G.; Liu, Y.; Pascual, D.; Richter, O.; and Wattenhofer, R. 2019.
>   On the validity of self-attention as explanation in transformer
>   models.
>   *arXiv preprint arXiv:1908.04211* .
> * Chapelle, Scholkopf, and Zien (2009)
>
>   Chapelle, O.; Scholkopf, B.; and Zien, A. 2009.
>   Semi-supervised learning).
>   *IEEE Transactions on Neural Networks* 20(3): 542–542.
> * Chappelle, Schölkopf, and Zien (2010)
>
>   Chappelle, O.; Schölkopf, B.; and Zien, A. 2010.
>   Semi-supervised learning. Adaptive Computation and Machine Learning.
> * Chen and Guestrin (2016)
>
>   Chen, T.; and Guestrin, C. 2016.
>   Xgboost: A scalable tree boosting system.
>   In *Proceedings of the 22nd acm sigkdd international conference
>   on knowledge discovery and data mining*, 785–794.
> * Cheng et al. (2016)
>
>   Cheng, H.-T.; Koc, L.; Harmsen, J.; Shaked, T.; Chandra, T.; Aradhye, H.;
>   Anderson, G.; Corrado, G.; Chai, W.; Ispir, M.; et al. 2016.
>   Wide & deep learning for recommender systems.
>   In *Proceedings of the 1st workshop on deep learning for
>   recommender systems*, 7–10.
> * Clark et al. (2020)
>
>   Clark, K.; Luong, M.-T.; Le, Q. V.; and Manning, C. D. 2020.
>   ELECTRA: Pre-training Text Encoders as Discriminators
>   Rather Than Generators.
>   In *International Conference on Learning Representations*.
>   URL https://openreview.net/forum?id=r1xMH1BtvB.
> * Coenen et al. (2019)
>
>   Coenen, A.; Reif, E.; Yuan, A.; Kim, B.; Pearce, A.; Viégas, F.; and
>   Wattenberg, M. 2019.
>   Visualizing and measuring the geometry of bert.
>   *arXiv preprint arXiv:1906.02715* .
> * Cortes et al. (2016)
>
>   Cortes, C.; Gonzalvo, X.; Kuznetsov, V.; Mohri, M.; and Yang, S. 2016.
>   AdaNet: Adaptive Structural Learning of Artificial Neural Networks.
> * De Brébisson et al. (2015)
>
>   De Brébisson, A.; Simon, E.; Auvolat, A.; Vincent, P.; and Bengio, Y. 2015.
>   Artificial Neural Networks Applied to Taxi Destination Prediction.
>   In *Proceedings of the 2015th International Conference on ECML
>   PKDD Discovery Challenge - Volume 1526*, ECMLPKDDDC’15, 40–51. Aachen,
>   DEU: CEUR-WS.org.
> * Devlin et al. (2019)
>
>   Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019.
>   BERT: Pre-training of Deep Bidirectional Transformers for Language
>   Understanding.
>   In *NAACL-HLT*.
> * Dua and Graff (2017)
>
>   Dua, D.; and Graff, C. 2017.
>   UCI Machine Learning Repository.
>   URL http://archive.ics.uci.edu/ml.
> * Grandvalet and Bengio (2005)
>
>   Grandvalet, Y.; and Bengio, Y. 2005.
>   Semi-supervised learning by entropy minimization.
>   In *Advances in neural information processing systems*,
>   529–536.
> * Grandvalet and Bengio (2006)
>
>   Grandvalet, Y.; and Bengio, Y. 2006.
>   Entropy regularization.
>   *Semi-supervised learning* 151–168.
> * Guo et al. (2018)
>
>   Guo, H.; Tang, R.; Ye, Y.; Li, Z.; He, X.; and Dong, Z. 2018.
>   DeepFM: An End-to-End Wide & Deep Learning Framework
>   for CTR Prediction.
>   *arXiv:1804.04950 [cs, stat]*
>   URL http://arxiv.org/abs/1804.04950.
>   ArXiv: 1804.04950.
> * Guyon et al. (2019)
>
>   Guyon, I.; Sun-Hosoya, L.; Boullé, M.; Escalante, H. J.; Escalera, S.; Liu,
>   Z.; Jajetic, D.; Ray, B.; Saeed, M.; Sebag, M.; Statnikov, A.; Tu, W.; and
>   Viegas, E. 2019.
>   Analysis of the AutoML Challenge series 2015-2018.
>   In *AutoML*, Springer series on Challenges in Machine Learning.
>   URL https://www.automl.org/wp-content/uploads/2018/09/chapter10-challenge.pdf.
> * Howard and Ruder (2018)
>
>   Howard, J.; and Ruder, S. 2018.
>   Universal language model fine-tuning for text classification.
>   *arXiv preprint arXiv:1801.06146* .
> * Huang, Xu, and Yu (2015)
>
>   Huang, Z.; Xu, W.; and Yu, K. 2015.
>   Bidirectional LSTM-CRF models for sequence tagging.
>   *arXiv preprint arXiv:1508.01991* .
> * Iscen et al. (2019)
>
>   Iscen, A.; Tolias, G.; Avrithis, Y.; and Chum, O. 2019.
>   Label propagation for deep semi-supervised learning.
>   In *Proceedings of the IEEE Conference on Computer Vision and
>   Pattern Recognition*, 5070–5079.
> * Izmailov et al. (2019)
>
>   Izmailov, P.; Kirichenko, P.; Finzi, M.; and Wilson, A. G. 2019.
>   Semi-Supervised Learning with Normalizing Flows.
>   *arXiv:1912.13025 [cs, stat]*
>   URL http://arxiv.org/abs/1912.13025.
>   ArXiv: 1912.13025.
> * Jahrer (2018)
>
>   Jahrer, M. 2018.
>   Porto Seguro’s Safe Driver Prediction.
>   URL https://kaggle.com/c/porto-seguro-safe-driver-prediction.
> * Jain (2017)
>
>   Jain, S. 2017.
>   Introduction to Pseudo-Labelling : A Semi-Supervised learning
>   technique.
>   https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/.
> * Kaggle, Inc. (2017)
>
>   Kaggle, Inc. 2017.
>   The State of ML and Data Science 2017.
>   URL https://www.kaggle.com/surveys/2017.
> * Ke et al. (2017)
>
>   Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; and Liu,
>   T.-Y. 2017.
>   LightGBM: A highly efficient gradient boosting decision tree.
>   In *Advances in Neural Information Processing Systems*,
>   3146–3154.
>   URL https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf.
> * Ke et al. (2019)
>
>   Ke, G.; Zhang, J.; Xu, Z.; Bian, J.; and Liu, T.-Y. 2019.
>   TabNN: A Universal Neural Network Solution for Tabular Data.
>   URL https://openreview.net/forum?id=r1eJssCqY7.
> * Kendall and Gal (2017)
>
>   Kendall, A.; and Gal, Y. 2017.
>   What uncertainties do we need in bayesian deep learning for computer
>   vision?
>   In *Advances in neural information processing systems*,
>   5574–5584.
> * Klambauer et al. (2017)
>
>   Klambauer, G.; Unterthiner, T.; Mayr, A.; and Hochreiter, S. 2017.
>   Self-normalizing neural networks.
>   In *Advances in neural information processing systems*,
>   971–980.
> * Lee (2013)
>
>   Lee, D.-H. 2013.
>   Pseudo-label: The simple and efficient semi-supervised learning
>   method for deep neural networks.
>   In *Workshop on challenges in representation learning, ICML*,
>   volume 3, 2.
> * Li et al. (2020)
>
>   Li, Z.; Cheng, W.; Chen, Y.; Chen, H.; and Wang, W. 2020.
>   Interpretable Click-Through Rate Prediction through
>   Hierarchical Attention.
>   In *Proceedings of the 13th International Conference on
>   Web Search and Data Mining*, 313–321. Houston TX USA: ACM.
>   ISBN 978-1-4503-6822-3.
>   doi:10.1145/3336191.3371785.
>   URL http://dl.acm.org/doi/10.1145/3336191.3371785.
> * Loshchilov and Hutter (2017)
>
>   Loshchilov, I.; and Hutter, F. 2017.
>   Decoupled Weight Decay Regularization.
>   In *International Conference on Learning Representations*.
>   URL https://arxiv.org/abs/1711.05101.
> * Maaten and Hinton (2008)
>
>   Maaten, L. v. d.; and Hinton, G. 2008.
>   Visualizing data using t-SNE.
>   *Journal of machine learning research* 9(Nov): 2579–2605.
> * Mikolov et al. (2011)
>
>   Mikolov, T.; Kombrink, S.; Burget, L.; Černockỳ, J.; and Khudanpur,
>   S. 2011.
>   Extensions of recurrent neural network language model.
>   In *2011 IEEE international conference on acoustics, speech and
>   signal processing (ICASSP)*, 5528–5531. IEEE.
> * Morcos et al. (2019)
>
>   Morcos, A. S.; Yu, H.; Paganini, M.; and Tian, Y. 2019.
>   One ticket to win them all: generalizing lottery ticket
>   initializations across datasets and optimizers.
>   *arXiv:1906.02773 [cs, stat]*
>   URL http://arxiv.org/abs/1906.02773.
>   ArXiv: 1906.02773.
> * Nigam and Ghani (2000)
>
>   Nigam, K.; and Ghani, R. 2000.
>   Analyzing the effectiveness and applicability of co-training.
>   In *Proceedings of the ninth international conference on
>   Information and knowledge management*, 86–93.
> * Oliver et al. (2019)
>
>   Oliver, A.; Odena, A.; Raffel, C.; Cubuk, E. D.; and Goodfellow, I. J. 2019.
>   Realistic Evaluation of Deep Semi-Supervised Learning
>   Algorithms.
>   *arXiv:1804.09170 [cs, stat]*
>   URL http://arxiv.org/abs/1804.09170.
>   ArXiv: 1804.09170.
> * Oliver et al. (2018)
>
>   Oliver, A.; Odena, A.; Raffel, C. A.; Cubuk, E. D.; and Goodfellow, I. 2018.
>   Realistic evaluation of deep semi-supervised learning algorithms.
>   In *Advances in Neural Information Processing Systems*,
>   3235–3246.
> * Paszke et al. (2019)
>
>   Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen,
>   T.; Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.;
>   DeVito, Z.; Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.;
>   Bai, J.; and Chintala, S. 2019.
>   PyTorch: An Imperative Style, High-Performance Deep Learning Library.
>   In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d’Alché Buc, F.;
>   Fox, E.; and Garnett, R., eds., *Advances in Neural Information
>   Processing Systems 32*, 8024–8035. Curran Associates, Inc.
>   URL http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.
> * Prokhorenkova et al. (2018)
>
>   Prokhorenkova, L.; Gusev, G.; Vorobev, A.; Dorogush, A. V.; and Gulin, A. 2018.
>   CatBoost: unbiased boosting with categorical features.
>   In *Advances in neural information processing systems*,
>   6638–6648.
> * Rong (2014)
>
>   Rong, X. 2014.
>   word2vec parameter learning explained.
>   *arXiv preprint arXiv:1411.2738* .
> * Sajjadi, Javanmardi, and Tasdizen (2016)
>
>   Sajjadi, M.; Javanmardi, M.; and Tasdizen, T. 2016.
>   Regularization with stochastic transformations and perturbations for
>   deep semi-supervised learning.
>   In *Advances in neural information processing systems*,
>   1163–1171.
> * Sandler et al. (2018)
>
>   Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; and Chen, L.-C. 2018.
>   Mobilenetv2: Inverted residuals and linear bottlenecks.
>   In *Proceedings of the IEEE conference on computer vision and
>   pattern recognition*, 4510–4520.
> * Song et al. (2019)
>
>   Song, W.; Shi, C.; Xiao, Z.; Duan, Z.; Xu, Y.; Zhang, M.; and Tang, J. 2019.
>   AutoInt: Automatic Feature Interaction Learning via
>   Self-Attentive Neural Networks.
>   *Proceedings of the 28th ACM International Conference on
>   Information and Knowledge Management - CIKM ’19* 1161–1170.
>   doi:10.1145/3357384.3357925.
>   URL http://arxiv.org/abs/1810.11921.
>   ArXiv: 1810.11921.
> * Stretcu et al. (2019)
>
>   Stretcu, O.; Viswanathan, K.; Movshovitz-Attias, D.; Platanios, E.; Ravi, S.;
>   and Tomkins, A. 2019.
>   Graph Agreement Models for Semi-Supervised Learning.
>   In *Advances in Neural Information Processing Systems
>   32*, 8713–8723. Curran Associates, Inc.
>   URL http://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning.pdf.
> * Sun et al. (2019)
>
>   Sun, Q.; Cheng, Z.; Fu, Y.; Wang, W.; Jiang, Y.-G.; and Xue, X. 2019.
>   DeepEnFM: Deep neural networks with Encoder enhanced
>   Factorization Machine
>   URL https://openreview.net/forum?id=SJlyta4YPS.
> * Tanha, Someren, and Afsarmanesh (2017)
>
>   Tanha, J.; Someren, M.; and Afsarmanesh, H. 2017.
>   Semi-supervised self-training for decision tree classifiers.
>   *International Journal of Machine Learning and Cybernetics* 8:
>   355–370.
> * Torpey and Watson (2014)
>
>   Torpey, E.; and Watson, A. 2014.
>   *Education level and jobs: Opportunities by state*.
>   URL https://www.bls.gov/careeroutlook/2014/article/education-level-and-jobs.htm.
> * Vaswani et al. (2017)
>
>   Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.;
>   Kaiser, Ł.; and Polosukhin, I. 2017.
>   Attention is all you need.
>   In *Advances in neural information processing systems*,
>   5998–6008.
> * Voulodimos et al. (2018)
>
>   Voulodimos, A.; Doulamis, N.; Doulamis, A.; and Protopapadakis, E. 2018.
>   Deep learning for computer vision: A brief review.
>   *Computational intelligence and neuroscience* 2018.
> * Wang et al. (2017)
>
>   Wang, R.; Fu, B.; Fu, G.; and Wang, M. 2017.
>   Deep & Cross Network for Ad Click Predictions.
>   In *ADKDD@KDD*.
> * Xiao et al. (2017)
>
>   Xiao, J.; Ye, H.; He, X.; Zhang, H.; Wu, F.; and Chua, T.-S. 2017.
>   Attentional Factorization Machines: Learning the Weight of
>   Feature Interactions via Attention Networks.
>   In *Proceedings of the Twenty-Sixth International Joint
>   Conference on Artificial Intelligence*, 3119–3125. Melbourne,
>   Australia: International Joint Conferences on Artificial Intelligence
>   Organization.
>   ISBN 978-0-9992411-0-3.
>   doi:10.24963/ijcai.2017/435.
>   URL https://www.ijcai.org/proceedings/2017/435.
> * Yang, Morillo, and Hospedales (2018)
>
>   Yang, Y.; Morillo, I. G.; and Hospedales, T. M. 2018.
>   Deep neural decision trees.
>   *arXiv preprint arXiv:1806.06988* .
> * Zhu and Ghahramani (2002)
>
>   Zhu, X.; and Ghahramani, Z. 2002.
>   Learning from labeled and unlabeled data with label propagation .

## Appendix A Appendix: Ablation Studies

We perform a number of ablation studies on various architectural choices and pre-training approaches for our TabTransformer. The first ablation study is on the choice of column embedding. The second and third ablation studies focus on the pre-training approach. Specifically, they are on the replacement value k𝑘k and dynamic versus static replacement strategy. For the pre-training approach, we use TabTransformer-RTD as our model. That is, the loss in the pre-training is RTD loss. For TabTransformer, the hidden (embedding) dimension, the number of layers and the number of attention heads in the Transformer are set to 323232, 666, and 888 respectively. The MLP layer sizes are set to {4×l,2×l}4𝑙2𝑙\{4\times l,2\times l\}, where l𝑙l is the size of its input. To better present the result, we introduce an additional evaluation metric, the relative AUC. More precisely, for each dataset and cross-validation split, the relative AUC for a model is the relative change of its AUC against the mean AUC over all competing models.

#### Column Embedding.

The first study is on the choice of column embedding – shared parameters 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}} across the embeddings of multiple classes in column i𝑖i for i∈{1,2,…,m}𝑖12…𝑚i\in\{1,2,...,m\}. In particular, we study the optimal dimension of 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}}, ℓℓ\ell. An alternative choice is to element-wisely add the unique identifier 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}} and feature-value specific embeddings 𝒘ϕi​jsubscript𝒘subscriptitalic-ϕ𝑖𝑗\bm{w}\_{\phi\_{ij}} rather than concatenating them. In that case, both the dimension of 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}} and 𝒘ϕi​jsubscript𝒘subscriptitalic-ϕ𝑖𝑗\bm{w}\_{\phi\_{ij}} are equal to the dimension of embedding d𝑑d. The goal of having column embedding is to enable the model to distinguish the classes in one column from those in the other columns. A baseline approach is to not have any shared embedding. Results are presented in Table [5](#A1.T5 "Table 5 ‣ Column Embedding. ‣ Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") where “Col Embed-Concat-1/X𝑋X” indicates that the dimension ℓℓ\ell is set as d/X𝑑𝑋d/X. The relative AUC score is calculated over all the models that appear in the rows and columns in the table, which explains why negative scores appear in some of the entries. Results show that not having the shared column embedding performs worst and our concatenation column embedding gives an average better performance.

Table 5: Performance of TabTransformer with no column embedding, concatenation column embedding, and addition column embedding. The evaluation metric is mean ±plus-or-minus\pm standard deviation of relative AUCs (in percentage) over all 15 datasets. Larger value means better performance. The best model is bold for each row.

| # of Transformers Layers | No Col Embed | Col Embed-Concat-1/4 | Col Embed-Concat-1/8 | Col Embed-Add |
| --- | --- | --- | --- | --- |
| 1 | -0.59 ±plus-or-minus\pm 0.33 | -2.01 ±plus-or-minus\pm 1.33 | -0.27 ±plus-or-minus\pm 0.21 | -1.11 ±plus-or-minus\pm 0.77 |
| 2 | -0.59 ±plus-or-minus\pm 0.22 | -0.37 ±plus-or-minus\pm 0.20 | -0.14 ±plus-or-minus\pm 0.19 | 0.34 ±plus-or-minus\pm 0.27 |
| 3 | -0.37 ±plus-or-minus\pm 0.19 | 0.04 ±plus-or-minus\pm 0.18 | -0.02 ±plus-or-minus\pm 0.21 | 0.21 ±plus-or-minus\pm 0.23 |
| 6 | 0.54 ±plus-or-minus\pm 0.22 | 0.53 ±plus-or-minus\pm 0.24 | 0.70 ±plus-or-minus\pm 0.17 | 0.25 ±plus-or-minus\pm 0.23 |
| 12 | 0.66 ±plus-or-minus\pm 0.21 | 1.05 ±plus-or-minus\pm 0.31 | 0.73 ±plus-or-minus\pm 0.58 | 0.42 ±plus-or-minus\pm 0.39 |

#### The replacement value k𝑘k.

The second ablation study is on the replacement value k𝑘k in pre-training approach. We run experiments for three different choices of k𝑘k – {15,30,50}153050\{15,30,50\} on three different datasets, namely – Adult, BankMarketing, and 1995\_income. The TabTransformer is firstly pre-trained with a value of k𝑘k on unlabeled data and then fine-tuned on labeled data. The number of labeled data is set as 256. The final fine-tuning accuracy is not much sensitive to the value of k𝑘k, as shown in Table [6](#A1.T6 "Table 6 ‣ The replacement value 𝑘. ‣ Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). The pre-training curves of training and validation accuracy for the three different replacement value k𝑘k is shown in Figure [6](#A1.F6 "Figure 6 ‣ Dynamic versus Static Replacement. ‣ Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). Note, that a constant prediction model would achieve 85%percent8585\% accuracy for the 15%percent1515\% replacement value.

Table 6: Fine-tuning performance of TabTransformer-RTD for different pre-training replacement value k𝑘k. The number of labeled data points is 256256256. The evaluation metrics are mean ±plus-or-minus\pm standard deviation of (1) AUC score over 555 cross-validation splits for each dataset (in percentage); (2) relative AUCs over the 333 datasets (in percentage). Larger value means better performance. The best model is bold for each column.

| Replacement value k%percent𝑘k\% | Adult | BankMarketing | 1995\_income | relative AUC (%) |
| --- | --- | --- | --- | --- |
| 15 | 58.1 ±plus-or-minus\pm 3.52 | 85.9 ±plus-or-minus\pm 1.62 | 86.8 ±plus-or-minus\pm 1.35 | 0.02 ±plus-or-minus\pm 0.10 |
| 30 | 58.1 ±plus-or-minus\pm 3.15 | 86.1 ±plus-or-minus\pm 1.58 | 86.7 ±plus-or-minus\pm 1.41 | 0.08 ±plus-or-minus\pm 0.10 |
| 50 | 57.9 ±plus-or-minus\pm 3.21 | 85.7 ±plus-or-minus\pm 1.93 | 86.7 ±plus-or-minus\pm 1.38 | -0.10 ±plus-or-minus\pm 0.11 |

#### Dynamic versus Static Replacement.

The third ablation study is on dynamic vs static replacement in the pre-training approach. In dynamic replacement, we randomly replace feature values during pre-training over the epochs. That is the replacement is different in each epoch. Whereas in static replacement, the random replacement is chosen once, and then the same replacement is used in all the epochs. We combine this study with another ablation on shared RTD binary classifier (predictor) vs. different classifiers for different columns. Results in Table [7](#A1.T7 "Table 7 ‣ Dynamic versus Static Replacement. ‣ Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") show that our choice of dynamic replacement and un-shared RTD classifiers perform better than static replacement and shared RTD classifiers. Figure [7](#A1.F7 "Figure 7 ‣ Dynamic versus Static Replacement. ‣ Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings") shows the pre-training curves of training and validation accuracy for the three choices – dynamic replacement, static replacement, and static replacement with a shared RTD classifier.

Table 7: Fine-tuning performance of TabTransformer-RTD for dynamic replacement, static replacement, and static replacement with a shared classifier. The number of labeled data points is 256256256. The evaluation metrics are mean ±plus-or-minus\pm standard deviation of (1) AUC score over 555 cross-validation splits for each dataset (in percentage) ; (2) relative AUCs over the 333 datasets (in percentage). Larger value means better performance. The best model is bold for each column.

|  | Adult | BankMarketing | 1995\_income | relative AUC (%) |
| --- | --- | --- | --- | --- |
| Dynamic Replacement (Un-shared RTD classifiers) | 58.1 ±plus-or-minus\pm 3.52 | 85.9 ±plus-or-minus\pm 1.62 | 86.8 ±plus-or-minus\pm 1.35 | 0.81 ±plus-or-minus\pm 0.19 |
| Static Replacement (Un-shared RTD classifiers) | 57.9 ±plus-or-minus\pm 2.93 | 83.9 ±plus-or-minus\pm 1.18 | 85.9 ±plus-or-minus\pm 1.60 | -0.33 ±plus-or-minus\pm 0.15 |
| Static Replacement (Shared RTD Classifiers) | 57.5 ±plus-or-minus\pm 2.74 | 84.2 ±plus-or-minus\pm 1.46 | 86.0 ±plus-or-minus\pm 1.69 | -0.49 ±plus-or-minus\pm 0.11 |

![Refer to caption](/html/2012.06678/assets/pretraining_acc.png)


Figure 6: The pre-training curves of training and validation accuracy for the three different replacement value k𝑘k for dataset Adult, BankMarketing, and 1995\_income.

![Refer to caption](/html/2012.06678/assets/dynamic_static_plot_log.png)


Figure 7: The pre-training curves of training and validation accuracy for dynamic mask, static mask, and static mask with a shared predictor (classifier) for dataset Adult, BankMarketing, and 1995\_income.

## Appendix B Appendix: Experiment and Model Details

In this section, we discuss the experiments and model details. First, we go through the experiments details and hyper parameters search space for HPO in Section [B.1](#A2.SS1 "B.1 Experiments Details and Hyper Parameters ‣ Appendix B Appendix: Experiment and Model Details ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings"). Next, we discuss the feature engineering in Section [B.2](#A2.SS2 "B.2 Feature Engineering ‣ Appendix B Appendix: Experiment and Model Details ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

### B.1 Experiments Details and Hyper Parameters

#### Setup.

All experiments were run on an Ubuntu Linux machine with 8 CPUs and 60GB memory, with all models using a single NVIDIA V100 Tensor Core GPU. For the competing models mentioned in the experiment, we re-implemented all of them for consistency of pre-processing. In cases where there exist published results for a model, our tested results are close to the published records. The GBDT model is implemented using the LightGBM library (Ke et al. [2017](#bib.bib28)). All the other models are implemented using the PyTorch library (Paszke et al. [2019](#bib.bib41)). To reproduce our experiment results, the models’ implementations and the exact values for all hyper-parameters can be found in another supplemental material, Code and Data Appendix.

For each dataset, all of the cross-validation splits, labeled, and unlabeled training data are obtained with a fixed random seed such that every model tested receives exactly the same training and testing conditions.

As all the datasets are for binary classification, the cross entropy loss was used for both supervised and semi-supervised training (for pre-training, the problem is binary classification in RTD and multi-class classification in MLM). For all deep models, the AdamW optimizer (Loshchilov and Hutter [2017](#bib.bib34)) was used to update the model parameters, and a constant learning rate was applied throughout each training job. All models used early stopping based on the performance on the validation set and the early stopping patience (the number of epochs) is set as 15.

#### Hyper-parameters Search Space.

The hyper-parameters tuned for the GBDT model were the number of leaves in the trees with a search space {x∈ℤ|5≤x≤50}conditional-set𝑥ℤ5𝑥50\{x\in\mathbb{Z}|5\leq x\leq 50\}, the minimum number of datapoints required to split a leaf in the trees with a search space {x∈ℤ|1≤x≤100}conditional-set𝑥ℤ1𝑥100\{x\in\mathbb{Z}|1\leq x\leq 100\}, the boosting learning rate with a search space {x=5⋅10u,u∈𝕌|−3≤x≤−1}conditional-setformulae-sequence𝑥⋅5superscript10𝑢𝑢𝕌3𝑥1\{x=5\cdot 10^{u},u\in\mathbb{U}|-3\leq x\leq-1\}, and the number of trees used for boosting with a search space {x∈ℤ|10≤x≤1000}conditional-set𝑥ℤ10𝑥1000\{x\in\mathbb{Z}|10\leq x\leq 1000\}.

For all of the deep models, the common hyper-parameters include the weight decay factor with a search space {x=10u,u∈𝕌|−6≤u≤−1}conditional-setformulae-sequence𝑥superscript10𝑢𝑢𝕌6𝑢1\{x=10^{u},u\in\mathbb{U}|-6\leq u\leq-1\}, the learning rate with a search space {x=10u,u∈𝕌|−6≤u≤−3}conditional-setformulae-sequence𝑥superscript10𝑢𝑢𝕌6𝑢3\{x=10^{u},u\in\mathbb{U}|-6\leq u\leq-3\}, the dropout probability with a search space {0,0.1,0.2,…​0.5}00.10.2…0.5\{0,0.1,0.2,...0.5\}, and whether to one-hot encode categorical variables or train learnable embeddings.

For MLPs, they all used SELU activations (Klambauer et al. [2017](#bib.bib31)) followed by batch normalization in each layer, and set the number of hidden layers as 2. The model-specific hyper-parameters tuned were the first hidden layer with a search space {x=m∗l,m∈ℤ|1≤m≤8}conditional-setformulae-sequence𝑥𝑚𝑙𝑚ℤ1𝑚8\{x=m\*l,m\in\mathbb{Z}|1\leq m\leq 8\} where l𝑙l is the input size, and the second hidden layer with a search space {x=m∗l,m∈ℤ|1≤m≤3}conditional-setformulae-sequence𝑥𝑚𝑙𝑚ℤ1𝑚3\{x=m\*l,m\in\mathbb{Z}|1\leq m\leq 3\}.

For TabTransformer, the hidden (embedding) dimension, the number of layers and the number of attention heads in the Transformer were fixed to 323232, 666, and 888 respectively during the experiments. The MLP layer sizes were fixed to {4×l,2×l}4𝑙2𝑙\{4\times l,2\times l\}, where l𝑙l was the size of its input. However, these parameters were optimally selected based on 50 rounds of HPO run on 5 datasets. The search spaces were the number of attention heads {2,4,8}248\{2,4,8\}, the hidden dimension {32,64,128,256}3264128256\{32,64,128,256\}, and the number of layers {1,2,3,6,12}123612\{1,2,3,6,12\}. The search spaces of the first and second hidden layer in MLP are exactly the same as those in MLP model setting. The dimension of 𝒄ϕisubscript𝒄subscriptitalic-ϕ𝑖\bm{c}\_{\phi\_{i}}, ℓℓ\ell was chosen as d/8𝑑8d/8 based on the ablation study in Appendix [A](#A1 "Appendix A Appendix: Ablation Studies ‣ TabTransformer: Tabular Data Modeling Using Contextual Embeddings").

For Sparse MLP (Prune), its implementation was the same as the MLP except that at every k𝑘k epochs during training the fraction p𝑝p of weights with the smallest magnitude were permanently set to zero.
The model-specific hyper-parameters tuned were the fraction p𝑝p with a search space {x=5⋅10u,u∈𝕌|−2≤u≤−1}conditional-setformulae-sequence𝑥⋅5superscript10𝑢𝑢𝕌2𝑢1\{x=5\cdot 10^{u},u\in\mathbb{U}|-2\leq u\leq-1\}. The number of layers and layer sizes are exactly the same as the setting in MLP. The parameter k𝑘k is set as 10.

For TabNet model, we implemented exactly as described in Arik and Pfister ([2019](#bib.bib3)), though we also added the option to use a softmax attention instead of a sparsemax attention, and did not include the sparsification term in the loss function.
The model-specific hyper-parameters tuned were the number of layers with a search space {x∈ℤ|3≤x≤10}conditional-set𝑥ℤ3𝑥10\{x\in\mathbb{Z}|3\leq x\leq 10\} , the hidden dimension {x∈ℤ|8≤x≤128}conditional-set𝑥ℤ8𝑥128\{x\in\mathbb{Z}|8\leq x\leq 128\}, and the sparse coefficient with a search space {x=10u,u∈𝕌|−6≤u≤−2}conditional-setformulae-sequence𝑥superscript10𝑢𝑢𝕌6𝑢2\{x=10^{u},u\in\mathbb{U}|-6\leq u\leq-2\}.

For VIB model, we implemented it as described in Alemi, Fischer, and Dillon ([2018](#bib.bib1)). We used a diagonal covariance, with 10 samples from the variational distribution during training and 20 during testing.
The model-specific hyper-parameters tuned were the number of hidden layers and layer sizes, with exactly the same search spaces as MLP, and the number of mixture components in the mixture of gaussians used in the marginal distribution with a search space {x∈ℤ|3≤x≤10}conditional-set𝑥ℤ3𝑥10\{x\in\mathbb{Z}|3\leq x\leq 10\}.

For MLP (DAE), its pre-training used swap noise as described in Jahrer ([2018](#bib.bib25)). The model-specific hyper-parameters were exactly the same as MLP.

For Pseudo Labeling (Lee [2013](#bib.bib32)), since this method was combined with deep models such as MLP, TabTransformer and GBDT, the model-specific hyper-parameters were exactly the same as the corresponding deep models mentioned above. The unsupervised coefficient α𝛼\alpha is chosen as αf=3,T1=30,T2=70formulae-sequencesubscript𝛼𝑓3formulae-sequencesubscript𝑇130subscript𝑇270\alpha\_{f}=3,T\_{1}=30,T\_{2}=70.

For Entropy Regularization (Grandvalet and Bengio [2006](#bib.bib18)), it is the same as Pseudo Labeling. The additional model-specific hyper-parameter was the positive Lagrange multiplier λ𝜆\lambda with a search space {0.1,0.2,…,0.9}0.10.2…0.9\{0.1,0.2,...,0.9\}.

### B.2 Feature Engineering

For categorical variables, the processing options include whether to one-hot encode versus learn a parametric embedding, what embedding dimension to use, and how to apply dropout regularization (whether to drop vector elements or whole embeddings).
In our experiments we found that learned embeddings nearly always improved performance as long as the cardinality of the categorical variable is significantly less than the number of data points, otherwise the feature is merely a means for the model to overfit.

For scalar variables, the processing options include how to re-scale the variable (via quantiles, normalization, or log scaling) or whether to quantize the feature and treat it like a categorical variable.
While we have not explored this idea fully, the best strategy is likely to use all the different types of encoding in parallel, turning each scalar feature into three re-scaled features and one categorical feature. Unlike learning embeddings for high-cardinality categorical features, adding potentially-redundant encodings for scalar variables should not lead to overfitting, but can make the difference between a feature being useful or not.

For text variables, we simply encodes the number of words and character in the text.

## Appendix C Appendix: Benchmark Dataset Information and Experiment Results

Table 8: Benchmark datasets. All datasets are binary classification tasks. Positive Class% is the fraction of data points that belongs to the positive class.

| Dataset Name | N Datapoints | N Features | Positive Class% |
| --- | --- | --- | --- |
| 1995\_income | 32561 | 14 | 24.1 |
| adult | 34190 | 25 | 85.4 |
| albert | 425240 | 79 | 50.0 |
| bank\_marketing | 45211 | 16 | 11.7 |
| blastchar | 7043 | 20 | 26.5 |
| dota2games | 92650 | 117 | 52.7 |
| fabert | 8237 | 801 | 11.3 |
| hcdr\_main | 307511 | 120 | 8.1 |
| htru2 | 17898 | 8 | 9.2 |
| insurance\_co | 5822 | 85 | 6.0 |
| jannis | 83733 | 55 | 2.0 |
| jasmine | 2984 | 145 | 50.0 |
| online\_shoppers | 12330 | 17 | 15.5 |
| philippine | 5832 | 309 | 50.0 |
| qsar\_bio | 1055 | 41 | 33.7 |
| seismicbumps | 2583 | 18 | 6.6 |
| shrutime | 10000 | 11 | 20.4 |
| spambase | 4601 | 57 | 39.4 |
| sylvine | 5124 | 20 | 50.0 |
| volkert | 58310 | 181 | 12.7 |




Table 9: Benchmark Dataset Links.

| Dataset Name | URL |
| --- | --- |
| 1995\_income | https://www.kaggle.com/lodetomasi1995/income-classification |
| adult | http://automl.chalearn.org/data |
| albert | http://automl.chalearn.org/data |
| bank\_marketing | https://archive.ics.uci.edu/ml/datasets/bank+marketing |
| blastchar | https://www.kaggle.com/blastchar/telco-customer-churn |
| dota2games | https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results |
| fabert | http://automl.chalearn.org/data |
| hcdr\_main | https://www.kaggle.com/c/home-credit-default-risk |
| htru2 | https://archive.ics.uci.edu/ml/datasets/HTRU2 |
| insurance\_co | https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+“%28COIL+2000“%29 |
| jannis | http://automl.chalearn.org/data |
| jasmine | http://automl.chalearn.org/data |
| online\_shoppers | https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset |
| philippine | http://automl.chalearn.org/data |
| qsar\_bio | https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation |
| seismicbumps | https://archive.ics.uci.edu/ml/datasets/seismic-bumps |
| shrutime | https://www.kaggle.com/shrutimechlearn/churn-modelling |
| spambase | https://archive.ics.uci.edu/ml/datasets/Spambase |
| sylvine | http://automl.chalearn.org/data |
| volkert | http://automl.chalearn.org/data |




Table 10: AUC score for semi-supervised learning models on all datasets with 50 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | N Datapoints | N Features | Positive Class% | Best Model | TabTransformer-RTD | TabTransformer-MLM | MLP (ER) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| albert | 425240 | 79 | 50.0 | TabTransformer-MLM | 0.644 ±plus-or-minus\pm 0.015 | 0.647 ±plus-or-minus\pm 0.019 | 0.612 ±plus-or-minus\pm 0.017 |
| hcdr\_main | 307511 | 120 | 8.1 | MLP (DAE) | 0.592 ±plus-or-minus\pm 0.047 | 0.596 ±plus-or-minus\pm 0.047 | 0.602 ±plus-or-minus\pm 0.033 |
| dota2games | 92650 | 117 | 52.7 | TabTransformer-MLM | 0.526 ±plus-or-minus\pm 0.009 | 0.538 ±plus-or-minus\pm 0.011 | 0.519 ±plus-or-minus\pm 0.007 |
| jannis | 83733 | 55 | 2.0 | TabTransformer-RTD | 0.684 ±plus-or-minus\pm 0.055 | 0.665 ±plus-or-minus\pm 0.056 | 0.621 ±plus-or-minus\pm 0.022 |
| volkert | 58310 | 181 | 1.0 | TabTransformer-RTD | 0.693 ±plus-or-minus\pm 0.046 | 0.689 ±plus-or-minus\pm 0.042 | 0.657 ±plus-or-minus\pm 0.028 |
| bank\_marketing | 45211 | 16 | 11.7 | MLP (PL) | 0.771 ±plus-or-minus\pm 0.046 | 0.735 ±plus-or-minus\pm 0.040 | 0.792 ±plus-or-minus\pm 0.039 |
| adult | 34190 | 25 | 85.4 | MLP (DAE) | 0.580 ±plus-or-minus\pm 0.012 | 0.613 ±plus-or-minus\pm 0.014 | 0.609 ±plus-or-minus\pm 0.005 |
| 1995\_income | 32561 | 14 | 24.1 | TabTransformer-MLM | 0.840 ±plus-or-minus\pm 0.029 | 0.862 ±plus-or-minus\pm 0.018 | 0.839 ±plus-or-minus\pm 0.034 |
| htru2 | 17898 | 8 | 9.2 | MLP (DAE) | 0.956 ±plus-or-minus\pm 0.007 | 0.958 ±plus-or-minus\pm 0.009 | 0.969 ±plus-or-minus\pm 0.012 |
| online\_shoppers | 12330 | 17 | 15.5 | MLP (DAE) | 0.790 ±plus-or-minus\pm 0.013 | 0.780 ±plus-or-minus\pm 0.024 | 0.855 ±plus-or-minus\pm 0.019 |
| shrutime | 10000 | 11 | 20.4 | TabTransformer-RTD | 0.752 ±plus-or-minus\pm 0.019 | 0.741 ±plus-or-minus\pm 0.019 | 0.725 ±plus-or-minus\pm 0.032 |
| fabert | 8237 | 801 | 11.3 | MLP (PL) | 0.535 ±plus-or-minus\pm 0.027 | 0.525 ±plus-or-minus\pm 0.019 | 0.572 ±plus-or-minus\pm 0.019 |
| blastchar | 7043 | 20 | 26.5 | TabTransformer-MLM | 0.806 ±plus-or-minus\pm 0.018 | 0.822 ±plus-or-minus\pm 0.009 | 0.803 ±plus-or-minus\pm 0.021 |
| philippine | 5832 | 309 | 50.0 | TabTransformer-RTD | 0.739 ±plus-or-minus\pm 0.027 | 0.729 ±plus-or-minus\pm 0.035 | 0.722 ±plus-or-minus\pm 0.031 |
| insurance\_co | 5822 | 85 | 6.0 | MLP (PL) | 0.601 ±plus-or-minus\pm 0.056 | 0.573 ±plus-or-minus\pm 0.077 | 0.575 ±plus-or-minus\pm 0.063 |
| sylvine | 5124 | 20 | 50.0 | MLP (PL) | 0.872 ±plus-or-minus\pm 0.031 | 0.898 ±plus-or-minus\pm 0.030 | 0.930 ±plus-or-minus\pm 0.015 |
| spambase | 4601 | 57 | 39.4 | MLP (ER) | 0.949 ±plus-or-minus\pm 0.005 | 0.945 ±plus-or-minus\pm 0.011 | 0.957 ±plus-or-minus\pm 0.008 |
| jasmine | 2984 | 145 | 50.0 | TabTransformer-MLM | 0.821 ±plus-or-minus\pm 0.019 | 0.837 ±plus-or-minus\pm 0.019 | 0.830 ±plus-or-minus\pm 0.022 |
| seismicbumps | 2583 | 18 | 6.6 | TabTransformer (ER) | 0.740 ±plus-or-minus\pm 0.088 | 0.738 ±plus-or-minus\pm 0.068 | 0.712 ±plus-or-minus\pm 0.074 |
| qsar\_bio | 1055 | 41 | 33.7 | MLP (DAE) | 0.875 ±plus-or-minus\pm 0.028 | 0.869 ±plus-or-minus\pm 0.036 | 0.880 ±plus-or-minus\pm 0.022 |




Table 11: (Continued) AUC score for semi-supervised learning models on all datasets with 50 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | MLP (PL) | TabTransformer (ER) | TabTransformer (PL) | MLP (DAE) | GBDT (PL) |
| --- | --- | --- | --- | --- | --- |
| albert | 0.607 ±plus-or-minus\pm 0.013 | 0.580 ±plus-or-minus\pm 0.017 | 0.587 ±plus-or-minus\pm 0.012 | 0.612 ±plus-or-minus\pm 0.014 | 0.547 ±plus-or-minus\pm 0.032 |
| hcdr\_main | 0.599 ±plus-or-minus\pm 0.038 | 0.581 ±plus-or-minus\pm 0.023 | 0.570 ±plus-or-minus\pm 0.031 | 0.620 ±plus-or-minus\pm 0.028 | 0.531 ±plus-or-minus\pm 0.024 |
| dota2games | 0.520 ±plus-or-minus\pm 0.006 | 0.516 ±plus-or-minus\pm 0.009 | 0.519 ±plus-or-minus\pm 0.008 | 0.516 ±plus-or-minus\pm 0.004 | 0.505 ±plus-or-minus\pm 0.008 |
| jannis | 0.623 ±plus-or-minus\pm 0.035 | 0.582 ±plus-or-minus\pm 0.035 | 0.604 ±plus-or-minus\pm 0.013 | 0.626 ±plus-or-minus\pm 0.023 | 0.519 ±plus-or-minus\pm 0.047 |
| volkert | 0.653 ±plus-or-minus\pm 0.035 | 0.635 ±plus-or-minus\pm 0.024 | 0.639 ±plus-or-minus\pm 0.040 | 0.629 ±plus-or-minus\pm 0.019 | 0.525 ±plus-or-minus\pm 0.018 |
| bank\_marketing | 0.805 ±plus-or-minus\pm 0.036 | 0.744 ±plus-or-minus\pm 0.063 | 0.767 ±plus-or-minus\pm 0.058 | 0.786 ±plus-or-minus\pm 0.055 | 0.688 ±plus-or-minus\pm 0.057 |
| adult | 0.605 ±plus-or-minus\pm 0.021 | 0.568 ±plus-or-minus\pm 0.012 | 0.582 ±plus-or-minus\pm 0.024 | 0.616 ±plus-or-minus\pm 0.010 | 0.519 ±plus-or-minus\pm 0.024 |
| 1995\_income | 0.819 ±plus-or-minus\pm 0.042 | 0.813 ±plus-or-minus\pm 0.045 | 0.822 ±plus-or-minus\pm 0.048 | 0.811 ±plus-or-minus\pm 0.042 | 0.685 ±plus-or-minus\pm 0.084 |
| htru2 | 0.970 ±plus-or-minus\pm 0.012 | 0.955 ±plus-or-minus\pm 0.007 | 0.951 ±plus-or-minus\pm 0.009 | 0.973 ±plus-or-minus\pm 0.003 | 0.919 ±plus-or-minus\pm 0.021 |
| online\_shoppers | 0.848 ±plus-or-minus\pm 0.021 | 0.816 ±plus-or-minus\pm 0.036 | 0.818 ±plus-or-minus\pm 0.028 | 0.858 ±plus-or-minus\pm 0.019 | 0.818 ±plus-or-minus\pm 0.032 |
| shrutime | 0.715 ±plus-or-minus\pm 0.044 | 0.748 ±plus-or-minus\pm 0.035 | 0.739 ±plus-or-minus\pm 0.034 | 0.683 ±plus-or-minus\pm 0.055 | 0.651 ±plus-or-minus\pm 0.093 |
| fabert | 0.577 ±plus-or-minus\pm 0.027 | 0.504 ±plus-or-minus\pm 0.020 | 0.516 ±plus-or-minus\pm 0.020 | 0.552 ±plus-or-minus\pm 0.013 | 0.534 ±plus-or-minus\pm 0.016 |
| blastchar | 0.799 ±plus-or-minus\pm 0.025 | 0.799 ±plus-or-minus\pm 0.013 | 0.792 ±plus-or-minus\pm 0.025 | 0.817 ±plus-or-minus\pm 0.016 | 0.729 ±plus-or-minus\pm 0.053 |
| philippine | 0.725 ±plus-or-minus\pm 0.022 | 0.689 ±plus-or-minus\pm 0.046 | 0.703 ±plus-or-minus\pm 0.050 | 0.717 ±plus-or-minus\pm 0.022 | 0.628 ±plus-or-minus\pm 0.085 |
| insurance\_co | 0.601 ±plus-or-minus\pm 0.057 | 0.575 ±plus-or-minus\pm 0.066 | 0.592 ±plus-or-minus\pm 0.080 | 0.522 ±plus-or-minus\pm 0.052 | 0.560 ±plus-or-minus\pm 0.081 |
| sylvine | 0.939 ±plus-or-minus\pm 0.013 | 0.891 ±plus-or-minus\pm 0.022 | 0.904 ±plus-or-minus\pm 0.027 | 0.925 ±plus-or-minus\pm 0.010 | 0.914 ±plus-or-minus\pm 0.021 |
| spambase | 0.951 ±plus-or-minus\pm 0.010 | 0.947 ±plus-or-minus\pm 0.006 | 0.948 ±plus-or-minus\pm 0.006 | 0.949 ±plus-or-minus\pm 0.012 | 0.899 ±plus-or-minus\pm 0.039 |
| jasmine | 0.819 ±plus-or-minus\pm 0.021 | 0.825 ±plus-or-minus\pm 0.024 | 0.819 ±plus-or-minus\pm 0.018 | 0.812 ±plus-or-minus\pm 0.029 | 0.755 ±plus-or-minus\pm 0.016 |
| seismicbumps | 0.678 ±plus-or-minus\pm 0.106 | 0.745 ±plus-or-minus\pm 0.080 | 0.713 ±plus-or-minus\pm 0.090 | 0.724 ±plus-or-minus\pm 0.049 | 0.601 ±plus-or-minus\pm 0.071 |
| qsar\_bio | 0.875 ±plus-or-minus\pm 0.015 | 0.851 ±plus-or-minus\pm 0.041 | 0.835 ±plus-or-minus\pm 0.053 | 0.888 ±plus-or-minus\pm 0.022 | 0.804 ±plus-or-minus\pm 0.057 |




Table 12: AUC score for semi-supervised learning models on all datasets with 200 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | N Datapoints | N Features | Positive Class% | Best Model | TabTransformer-RTD | TabTransformer-MLM | MLP (ER) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| albert | 425240 | 79 | 50.0 | TabTransformer-MLM | 0.699 ±plus-or-minus\pm 0.011 | 0.701 ±plus-or-minus\pm 0.014 | 0.642 ±plus-or-minus\pm 0.020 |
| hcdr\_main | 307511 | 120 | 8.1 | TabTransformer-MLM | 0.655 ±plus-or-minus\pm 0.040 | 0.668 ±plus-or-minus\pm 0.028 | 0.639 ±plus-or-minus\pm 0.027 |
| dota2games | 92650 | 117 | 52.7 | TabTransformer-MLM | 0.536 ±plus-or-minus\pm 0.012 | 0.549 ±plus-or-minus\pm 0.008 | 0.527 ±plus-or-minus\pm 0.012 |
| jannis | 83733 | 55 | 2.0 | TabTransformer-RTD | 0.713 ±plus-or-minus\pm 0.037 | 0.692 ±plus-or-minus\pm 0.024 | 0.665 ±plus-or-minus\pm 0.024 |
| volkert | 58310 | 181 | 12.7 | TabTransformer-RTD | 0.753 ±plus-or-minus\pm 0.022 | 0.742 ±plus-or-minus\pm 0.023 | 0.696 ±plus-or-minus\pm 0.033 |
| bank\_marketing | 45211 | 16 | 11.7 | MLP (PL) | 0.854 ±plus-or-minus\pm 0.020 | 0.838 ±plus-or-minus\pm 0.010 | 0.860 ±plus-or-minus\pm 0.008 |
| adult | 34190 | 25 | 85.4 | MLP (ER) | 0.596 ±plus-or-minus\pm 0.023 | 0.614 ±plus-or-minus\pm 0.012 | 0.623 ±plus-or-minus\pm 0.017 |
| 1995\_income | 32561 | 14 | 24.1 | TabTransformer-MLM | 0.866 ±plus-or-minus\pm 0.014 | 0.875 ±plus-or-minus\pm 0.011 | 0.868 ±plus-or-minus\pm 0.007 |
| htru2 | 17898 | 8 | 9.2 | MLP (DAE) | 0.961 ±plus-or-minus\pm 0.008 | 0.963 ±plus-or-minus\pm 0.009 | 0.974 ±plus-or-minus\pm 0.007 |
| online\_shoppers | 12330 | 17 | 15.5 | MLP (ER) | 0.834 ±plus-or-minus\pm 0.015 | 0.838 ±plus-or-minus\pm 0.024 | 0.876 ±plus-or-minus\pm 0.019 |
| shrutime | 10000 | 11 | 20.4 | TabTransformer-RTD | 0.805 ±plus-or-minus\pm 0.017 | 0.783 ±plus-or-minus\pm 0.024 | 0.773 ±plus-or-minus\pm 0.013 |
| fabert | 8237 | 801 | 11.3 | MLP (ER) | 0.556 ±plus-or-minus\pm 0.023 | 0.561 ±plus-or-minus\pm 0.028 | 0.600 ±plus-or-minus\pm 0.046 |
| blastchar | 7043 | 20 | 26.5 | TabTransformer-MLM | 0.831 ±plus-or-minus\pm 0.010 | 0.841 ±plus-or-minus\pm 0.014 | 0.829 ±plus-or-minus\pm 0.010 |
| philippine | 5832 | 309 | 50.0 | TabTransformer-RTD | 0.757 ±plus-or-minus\pm 0.017 | 0.754 ±plus-or-minus\pm 0.016 | 0.732 ±plus-or-minus\pm 0.024 |
| insurance\_co | 5822 | 85 | 6.0 | TabTransformer (ER) | 0.667 ±plus-or-minus\pm 0.062 | 0.640 ±plus-or-minus\pm 0.043 | 0.601 ±plus-or-minus\pm 0.059 |
| sylvine | 5124 | 20 | 50.0 | MLP (PL) | 0.939 ±plus-or-minus\pm 0.008 | 0.948 ±plus-or-minus\pm 0.006 | 0.957 ±plus-or-minus\pm 0.008 |
| spambase | 4601 | 57 | 39.4 | MLP (ER) | 0.957 ±plus-or-minus\pm 0.006 | 0.955 ±plus-or-minus\pm 0.010 | 0.968 ±plus-or-minus\pm 0.009 |
| jasmine | 2984 | 145 | 50.0 | TabTransformer-RTD | 0.843 ±plus-or-minus\pm 0.016 | 0.843 ±plus-or-minus\pm 0.028 | 0.831 ±plus-or-minus\pm 0.019 |
| seismicbumps | 2583 | 18 | 6.6 | TabTransformer-RTD | 0.738 ±plus-or-minus\pm 0.063 | 0.708 ±plus-or-minus\pm 0.083 | 0.694 ±plus-or-minus\pm 0.088 |
| qsar\_bio | 1055 | 41 | 33.7 | TabTransformer-RTD | 0.896 ±plus-or-minus\pm 0.018 | 0.889 ±plus-or-minus\pm 0.030 | 0.895 ±plus-or-minus\pm 0.026 |




Table 13: (Continued) AUC score for semi-supervised learning models on all datasets with 200 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | MLP (PL) | TabTransformer (ER) | TabTransformer (PL) | MLP (DAE) | GBDT (PL) |
| --- | --- | --- | --- | --- | --- |
| albert | 0.638 ±plus-or-minus\pm 0.024 | 0.630 ±plus-or-minus\pm 0.025 | 0.630 ±plus-or-minus\pm 0.021 | 0.646 ±plus-or-minus\pm 0.023 | 0.628 ±plus-or-minus\pm 0.015 |
| hcdr\_main | 0.631 ±plus-or-minus\pm 0.019 | 0.611 ±plus-or-minus\pm 0.030 | 0.605 ±plus-or-minus\pm 0.021 | 0.636 ±plus-or-minus\pm 0.027 | 0.579 ±plus-or-minus\pm 0.039 |
| dota2games | 0.527 ±plus-or-minus\pm 0.014 | 0.528 ±plus-or-minus\pm 0.017 | 0.525 ±plus-or-minus\pm 0.011 | 0.528 ±plus-or-minus\pm 0.012 | 0.506 ±plus-or-minus\pm 0.008 |
| jannis | 0.667 ±plus-or-minus\pm 0.036 | 0.619 ±plus-or-minus\pm 0.024 | 0.637 ±plus-or-minus\pm 0.026 | 0.659 ±plus-or-minus\pm 0.020 | 0.525 ±plus-or-minus\pm 0.030 |
| volkert | 0.693 ±plus-or-minus\pm 0.028 | 0.694 ±plus-or-minus\pm 0.002 | 0.689 ±plus-or-minus\pm 0.015 | 0.672 ±plus-or-minus\pm 0.015 | 0.612 ±plus-or-minus\pm 0.042 |
| bank\_marketing | 0.866 ±plus-or-minus\pm 0.008 | 0.853 ±plus-or-minus\pm 0.016 | 0.858 ±plus-or-minus\pm 0.009 | 0.863 ±plus-or-minus\pm 0.009 | 0.802 ±plus-or-minus\pm 0.012 |
| adult | 0.616 ±plus-or-minus\pm 0.014 | 0.582 ±plus-or-minus\pm 0.026 | 0.584 ±plus-or-minus\pm 0.017 | 0.611 ±plus-or-minus\pm 0.027 | 0.572 ±plus-or-minus\pm 0.040 |
| 1995\_income | 0.869 ±plus-or-minus\pm 0.009 | 0.848 ±plus-or-minus\pm 0.024 | 0.852 ±plus-or-minus\pm 0.015 | 0.865 ±plus-or-minus\pm 0.011 | 0.822 ±plus-or-minus\pm 0.020 |
| htru2 | 0.974 ±plus-or-minus\pm 0.007 | 0.955 ±plus-or-minus\pm 0.007 | 0.954 ±plus-or-minus\pm 0.007 | 0.974 ±plus-or-minus\pm 0.010 | 0.946 ±plus-or-minus\pm 0.022 |
| online\_shoppers | 0.873 ±plus-or-minus\pm 0.030 | 0.857 ±plus-or-minus\pm 0.014 | 0.853 ±plus-or-minus\pm 0.017 | 0.873 ±plus-or-minus\pm 0.021 | 0.846 ±plus-or-minus\pm 0.019 |
| shrutime | 0.774 ±plus-or-minus\pm 0.018 | 0.803 ±plus-or-minus\pm 0.022 | 0.803 ±plus-or-minus\pm 0.024 | 0.763 ±plus-or-minus\pm 0.018 | 0.750 ±plus-or-minus\pm 0.050 |
| fabert | 0.595 ±plus-or-minus\pm 0.048 | 0.530 ±plus-or-minus\pm 0.027 | 0.522 ±plus-or-minus\pm 0.024 | 0.580 ±plus-or-minus\pm 0.020 | 0.573 ±plus-or-minus\pm 0.026 |
| blastchar | 0.829 ±plus-or-minus\pm 0.011 | 0.823 ±plus-or-minus\pm 0.011 | 0.823 ±plus-or-minus\pm 0.011 | 0.832 ±plus-or-minus\pm 0.013 | 0.783 ±plus-or-minus\pm 0.017 |
| philippine | 0.733 ±plus-or-minus\pm 0.018 | 0.736 ±plus-or-minus\pm 0.018 | 0.739 ±plus-or-minus\pm 0.024 | 0.720 ±plus-or-minus\pm 0.020 | 0.729 ±plus-or-minus\pm 0.024 |
| insurance\_co | 0.616 ±plus-or-minus\pm 0.045 | 0.715 ±plus-or-minus\pm 0.038 | 0.680 ±plus-or-minus\pm 0.034 | 0.612 ±plus-or-minus\pm 0.024 | 0.630 ±plus-or-minus\pm 0.087 |
| sylvine | 0.961 ±plus-or-minus\pm 0.004 | 0.951 ±plus-or-minus\pm 0.009 | 0.950 ±plus-or-minus\pm 0.010 | 0.955 ±plus-or-minus\pm 0.009 | 0.957 ±plus-or-minus\pm 0.005 |
| spambase | 0.965 ±plus-or-minus\pm 0.008 | 0.962 ±plus-or-minus\pm 0.006 | 0.960 ±plus-or-minus\pm 0.008 | 0.964 ±plus-or-minus\pm 0.009 | 0.957 ±plus-or-minus\pm 0.013 |
| jasmine | 0.839 ±plus-or-minus\pm 0.013 | 0.824 ±plus-or-minus\pm 0.024 | 0.841 ±plus-or-minus\pm 0.016 | 0.842 ±plus-or-minus\pm 0.014 | 0.826 ±plus-or-minus\pm 0.013 |
| seismicbumps | 0.684 ±plus-or-minus\pm 0.071 | 0.723 ±plus-or-minus\pm 0.080 | 0.727 ±plus-or-minus\pm 0.081 | 0.673 ±plus-or-minus\pm 0.070 | 0.603 ±plus-or-minus\pm 0.023 |
| qsar\_bio | 0.892 ±plus-or-minus\pm 0.033 | 0.871 ±plus-or-minus\pm 0.036 | 0.876 ±plus-or-minus\pm 0.032 | 0.891 ±plus-or-minus\pm 0.018 | 0.855 ±plus-or-minus\pm 0.035 |




Table 14: AUC score for semi-supervised learning models on all datasets with 500 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | N Datapoints | N Features | Positive Class% | Best Model | TabTransformer-RTD | TabTransformer-MLM | MLP (ER) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| albert | 425240 | 79 | 50.0 | TabTransformer-RTD | 0.711 ±plus-or-minus\pm 0.004 | 0.707 ±plus-or-minus\pm 0.006 | 0.666 ±plus-or-minus\pm 0.008 |
| hcdr\_main | 307511 | 120 | 8.1 | TabTransformer-MLM | 0.690 ±plus-or-minus\pm 0.038 | 0.698 ±plus-or-minus\pm 0.033 | 0.653 ±plus-or-minus\pm 0.019 |
| dota2games | 92650 | 117 | 52.7 | TabTransformer-MLM | 0.548 ±plus-or-minus\pm 0.008 | 0.557 ±plus-or-minus\pm 0.003 | 0.543 ±plus-or-minus\pm 0.008 |
| jannis | 83733 | 55 | 2.0 | TabTransformer-RTD | 0.747 ±plus-or-minus\pm 0.015 | 0.720 ±plus-or-minus\pm 0.018 | 0.707 ±plus-or-minus\pm 0.036 |
| volkert | 58310 | 181 | 12.7 | TabTransformer-RTD | 0.771 ±plus-or-minus\pm 0.016 | 0.760 ±plus-or-minus\pm 0.015 | 0.723 ±plus-or-minus\pm 0.016 |
| bank\_marketing | 45211 | 16 | 11.7 | TabTransformer-RTD | 0.879 ±plus-or-minus\pm 0.012 | 0.866 ±plus-or-minus\pm 0.016 | 0.869 ±plus-or-minus\pm 0.012 |
| adult | 34190 | 25 | 85.4 | MLP (PL) | 0.625 ±plus-or-minus\pm 0.011 | 0.647 ±plus-or-minus\pm 0.008 | 0.644 ±plus-or-minus\pm 0.015 |
| 1995\_income | 32561 | 14 | 24.1 | MLP (DAE) | 0.874 ±plus-or-minus\pm 0.008 | 0.880 ±plus-or-minus\pm 0.007 | 0.878 ±plus-or-minus\pm 0.002 |
| htru2 | 17898 | 8 | 9.2 | MLP (DAE) | 0.964 ±plus-or-minus\pm 0.009 | 0.966 ±plus-or-minus\pm 0.009 | 0.973 ±plus-or-minus\pm 0.010 |
| online\_shoppers | 12330 | 17 | 15.5 | MLP (ER) | 0.859 ±plus-or-minus\pm 0.009 | 0.861 ±plus-or-minus\pm 0.014 | 0.888 ±plus-or-minus\pm 0.012 |
| shrutime | 10000 | 11 | 20.4 | TabTransformer-RTD | 0.831 ±plus-or-minus\pm 0.017 | 0.815 ±plus-or-minus\pm 0.004 | 0.793 ±plus-or-minus\pm 0.017 |
| fabert | 8237 | 801 | 11.3 | MLP (ER) | 0.618 ±plus-or-minus\pm 0.014 | 0.609 ±plus-or-minus\pm 0.019 | 0.621 ±plus-or-minus\pm 0.032 |
| blastchar | 7043 | 20 | 26.5 | TabTransformer-RTD | 0.840 ±plus-or-minus\pm 0.013 | 0.839 ±plus-or-minus\pm 0.015 | 0.829 ±plus-or-minus\pm 0.013 |
| philippine | 5832 | 309 | 50.0 | TabTransformer-MLM | 0.769 ±plus-or-minus\pm 0.028 | 0.772 ±plus-or-minus\pm 0.017 | 0.734 ±plus-or-minus\pm 0.024 |
| insurance\_co | 5822 | 85 | 6.0 | TabTransformer (ER) | 0.688 ±plus-or-minus\pm 0.039 | 0.642 ±plus-or-minus\pm 0.029 | 0.659 ±plus-or-minus\pm 0.023 |
| sylvine | 5124 | 20 | 50.0 | MLP (PL) | 0.955 ±plus-or-minus\pm 0.007 | 0.959 ±plus-or-minus\pm 0.006 | 0.967 ±plus-or-minus\pm 0.003 |
| spambase | 4601 | 57 | 39.4 | MLP (ER) | 0.966 ±plus-or-minus\pm 0.007 | 0.968 ±plus-or-minus\pm 0.008 | 0.975 ±plus-or-minus\pm 0.004 |
| jasmine | 2984 | 145 | 50.0 | TabTransformer-RTD | 0.847 ±plus-or-minus\pm 0.016 | 0.844 ±plus-or-minus\pm 0.011 | 0.837 ±plus-or-minus\pm 0.019 |
| seismicbumps | 2583 | 18 | 6.6 | TabTransformer-RTD | 0.758 ±plus-or-minus\pm 0.081 | 0.729 ±plus-or-minus\pm 0.069 | 0.682 ±plus-or-minus\pm 0.123 |
| qsar\_bio | 1055 | 41 | 33.7 | MLP (DAE) | 0.909 ±plus-or-minus\pm 0.024 | 0.889 ±plus-or-minus\pm 0.038 | 0.918 ±plus-or-minus\pm 0.023 |




Table 15: (Continued) AUC score for semi-supervised learning models on all datasets with 500 fine-tune data points. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | MLP (PL) | TabTransformer (ER) | TabTransformer (PL) | MLP (DAE) | GBDT (PL) |
| --- | --- | --- | --- | --- | --- |
| albert | 0.662 ±plus-or-minus\pm 0.007 | 0.664 ±plus-or-minus\pm 0.011 | 0.643 ±plus-or-minus\pm 0.029 | 0.666 ±plus-or-minus\pm 0.006 | 0.653 ±plus-or-minus\pm 0.011 |
| hcdr\_main | 0.645 ±plus-or-minus\pm 0.022 | 0.623 ±plus-or-minus\pm 0.036 | 0.636 ±plus-or-minus\pm 0.031 | 0.657 ±plus-or-minus\pm 0.033 | 0.607 ±plus-or-minus\pm 0.035 |
| dota2games | 0.544 ±plus-or-minus\pm 0.010 | 0.538 ±plus-or-minus\pm 0.009 | 0.541 ±plus-or-minus\pm 0.010 | 0.542 ±plus-or-minus\pm 0.012 | 0.505 ±plus-or-minus\pm 0.005 |
| jannis | 0.698 ±plus-or-minus\pm 0.033 | 0.662 ±plus-or-minus\pm 0.007 | 0.660 ±plus-or-minus\pm 0.024 | 0.693 ±plus-or-minus\pm 0.024 | 0.521 ±plus-or-minus\pm 0.045 |
| volkert | 0.722 ±plus-or-minus\pm 0.012 | 0.712 ±plus-or-minus\pm 0.016 | 0.705 ±plus-or-minus\pm 0.021 | 0.712 ±plus-or-minus\pm 0.016 | 0.705 ±plus-or-minus\pm 0.016 |
| bank\_marketing | 0.876 ±plus-or-minus\pm 0.017 | 0.863 ±plus-or-minus\pm 0.008 | 0.868 ±plus-or-minus\pm 0.016 | 0.874 ±plus-or-minus\pm 0.012 | 0.838 ±plus-or-minus\pm 0.019 |
| adult | 0.651 ±plus-or-minus\pm 0.012 | 0.618 ±plus-or-minus\pm 0.023 | 0.618 ±plus-or-minus\pm 0.021 | 0.654 ±plus-or-minus\pm 0.016 | 0.647 ±plus-or-minus\pm 0.030 |
| 1995\_income | 0.880 ±plus-or-minus\pm 0.003 | 0.868 ±plus-or-minus\pm 0.008 | 0.869 ±plus-or-minus\pm 0.007 | 0.882 ±plus-or-minus\pm 0.001 | 0.839 ±plus-or-minus\pm 0.013 |
| htru2 | 0.974 ±plus-or-minus\pm 0.007 | 0.960 ±plus-or-minus\pm 0.010 | 0.960 ±plus-or-minus\pm 0.008 | 0.976 ±plus-or-minus\pm 0.006 | 0.949 ±plus-or-minus\pm 0.007 |
| online\_shoppers | 0.885 ±plus-or-minus\pm 0.021 | 0.861 ±plus-or-minus\pm 0.011 | 0.860 ±plus-or-minus\pm 0.013 | 0.885 ±plus-or-minus\pm 0.019 | 0.865 ±plus-or-minus\pm 0.011 |
| shrutime | 0.800 ±plus-or-minus\pm 0.015 | 0.825 ±plus-or-minus\pm 0.013 | 0.822 ±plus-or-minus\pm 0.016 | 0.804 ±plus-or-minus\pm 0.015 | 0.788 ±plus-or-minus\pm 0.019 |
| fabert | 0.596 ±plus-or-minus\pm 0.046 | 0.573 ±plus-or-minus\pm 0.048 | 0.578 ±plus-or-minus\pm 0.033 | 0.617 ±plus-or-minus\pm 0.042 | 0.585 ±plus-or-minus\pm 0.025 |
| blastchar | 0.833 ±plus-or-minus\pm 0.013 | 0.834 ±plus-or-minus\pm 0.013 | 0.832 ±plus-or-minus\pm 0.011 | 0.833 ±plus-or-minus\pm 0.012 | 0.795 ±plus-or-minus\pm 0.021 |
| philippine | 0.740 ±plus-or-minus\pm 0.023 | 0.746 ±plus-or-minus\pm 0.020 | 0.735 ±plus-or-minus\pm 0.015 | 0.739 ±plus-or-minus\pm 0.017 | 0.749 ±plus-or-minus\pm 0.026 |
| insurance\_co | 0.646 ±plus-or-minus\pm 0.048 | 0.710 ±plus-or-minus\pm 0.040 | 0.666 ±plus-or-minus\pm 0.060 | 0.612 ±plus-or-minus\pm 0.013 | 0.672 ±plus-or-minus\pm 0.037 |
| sylvine | 0.968 ±plus-or-minus\pm 0.003 | 0.958 ±plus-or-minus\pm 0.005 | 0.958 ±plus-or-minus\pm 0.003 | 0.967 ±plus-or-minus\pm 0.003 | 0.967 ±plus-or-minus\pm 0.006 |
| spambase | 0.973 ±plus-or-minus\pm 0.005 | 0.968 ±plus-or-minus\pm 0.007 | 0.967 ±plus-or-minus\pm 0.006 | 0.972 ±plus-or-minus\pm 0.006 | 0.972 ±plus-or-minus\pm 0.005 |
| jasmine | 0.833 ±plus-or-minus\pm 0.009 | 0.833 ±plus-or-minus\pm 0.021 | 0.838 ±plus-or-minus\pm 0.018 | 0.842 ±plus-or-minus\pm 0.011 | 0.838 ±plus-or-minus\pm 0.022 |
| seismicbumps | 0.677 ±plus-or-minus\pm 0.103 | 0.687 ±plus-or-minus\pm 0.100 | 0.735 ±plus-or-minus\pm 0.081 | 0.696 ±plus-or-minus\pm 0.112 | 0.666 ±plus-or-minus\pm 0.063 |
| qsar\_bio | 0.914 ±plus-or-minus\pm 0.032 | 0.894 ±plus-or-minus\pm 0.036 | 0.895 ±plus-or-minus\pm 0.035 | 0.925 ±plus-or-minus\pm 0.034 | 0.908 ±plus-or-minus\pm 0.024 |




Table 16: AUC score for supervised learning models on all datasets. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

| Dataset | N Datapoints | N Features | Positive Class% | Best Model | Logistic Regression | GBDT |
| --- | --- | --- | --- | --- | --- | --- |
| ds\_name |  |  |  |  |  |  |
| albert | 425240 | 79 | 50.0 | GBDT | 0.726 ±plus-or-minus\pm 0.001 | 0.763 ±plus-or-minus\pm 0.001 |
| hcdr\_main | 307511 | 120 | 8.1 | GBDT | 0.747 ±plus-or-minus\pm 0.004 | 0.756 ±plus-or-minus\pm 0.004 |
| dota2games | 92650 | 117 | 52.7 | Logistic Regression | 0.634 ±plus-or-minus\pm 0.003 | 0.621 ±plus-or-minus\pm 0.004 |
| bank\_marketing | 45211 | 16 | 11.7 | TabTransformer | 0.911 ±plus-or-minus\pm 0.005 | 0.933 ±plus-or-minus\pm 0.003 |
| adult | 34190 | 25 | 85.4 | GBDT | 0.721 ±plus-or-minus\pm 0.010 | 0.756 ±plus-or-minus\pm 0.011 |
| 1995\_income | 32561 | 14 | 24.1 | TabTransformer | 0.899 ±plus-or-minus\pm 0.002 | 0.906 ±plus-or-minus\pm 0.002 |
| online\_shoppers | 12330 | 17 | 15.5 | GBDT | 0.908 ±plus-or-minus\pm 0.015 | 0.930 ±plus-or-minus\pm 0.008 |
| shrutime | 10000 | 11 | 20.4 | GBDT | 0.828 ±plus-or-minus\pm 0.013 | 0.859 ±plus-or-minus\pm 0.009 |
| blastchar | 7043 | 20 | 26.5 | GBDT | 0.844 ±plus-or-minus\pm 0.010 | 0.847 ±plus-or-minus\pm 0.016 |
| philippine | 5832 | 309 | 50.0 | TabTransformer | 0.725 ±plus-or-minus\pm 0.022 | 0.812 ±plus-or-minus\pm 0.013 |
| insurance\_co | 5822 | 85 | 6.0 | TabTransformer | 0.736 ±plus-or-minus\pm 0.023 | 0.732 ±plus-or-minus\pm 0.022 |
| spambase | 4601 | 57 | 39.4 | GBDT | 0.947 ±plus-or-minus\pm 0.008 | 0.987 ±plus-or-minus\pm 0.005 |
| jasmine | 2984 | 145 | 50.0 | GBDT | 0.846 ±plus-or-minus\pm 0.017 | 0.862 ±plus-or-minus\pm 0.008 |
| seismicbumps | 2583 | 18 | 6.6 | GBDT | 0.749 ±plus-or-minus\pm 0.068 | 0.756 ±plus-or-minus\pm 0.084 |
| qsar\_bio | 1055 | 41 | 33.7 | TabTransformer | 0.847 ±plus-or-minus\pm 0.037 | 0.913 ±plus-or-minus\pm 0.031 |




Table 17: (Continued) AUC score for supervised learning models on all datasets. Values are the mean over 5 cross-validation splits, plus or minus the standard deviation. Larger values means better result.

|  | MLP | Sparse MLP | TabTransformer | TabNet | VIB |
| --- | --- | --- | --- | --- | --- |
| ds\_name |  |  |  |  |  |
| albert | 0.740 ±plus-or-minus\pm 0.001 | 0.741 ±plus-or-minus\pm 0.001 | 0.757 ±plus-or-minus\pm 0.002 | 0.705 ±plus-or-minus\pm 0.005 | 0.737 ±plus-or-minus\pm 0.001 |
| hcdr\_main | 0.743 ±plus-or-minus\pm 0.004 | 0.753 ±plus-or-minus\pm 0.004 | 0.751 ±plus-or-minus\pm 0.004 | 0.711 ±plus-or-minus\pm 0.006 | 0.745 ±plus-or-minus\pm 0.005 |
| dota2games | 0.631 ±plus-or-minus\pm 0.002 | 0.633 ±plus-or-minus\pm 0.004 | 0.633 ±plus-or-minus\pm 0.002 | 0.529 ±plus-or-minus\pm 0.025 | 0.628 ±plus-or-minus\pm 0.003 |
| bank\_marketing | 0.929 ±plus-or-minus\pm 0.003 | 0.926 ±plus-or-minus\pm 0.007 | 0.934 ±plus-or-minus\pm 0.004 | 0.885 ±plus-or-minus\pm 0.017 | 0.920 ±plus-or-minus\pm 0.005 |
| adult | 0.725 ±plus-or-minus\pm 0.010 | 0.740 ±plus-or-minus\pm 0.007 | 0.737 ±plus-or-minus\pm 0.009 | 0.663 ±plus-or-minus\pm 0.016 | 0.733 ±plus-or-minus\pm 0.009 |
| 1995\_income | 0.905 ±plus-or-minus\pm 0.003 | 0.904 ±plus-or-minus\pm 0.004 | 0.906 ±plus-or-minus\pm 0.003 | 0.875 ±plus-or-minus\pm 0.006 | 0.904 ±plus-or-minus\pm 0.003 |
| online\_shoppers | 0.919 ±plus-or-minus\pm 0.010 | 0.922 ±plus-or-minus\pm 0.011 | 0.927 ±plus-or-minus\pm 0.010 | 0.888 ±plus-or-minus\pm 0.020 | 0.907 ±plus-or-minus\pm 0.012 |
| shrutime | 0.846 ±plus-or-minus\pm 0.013 | 0.828 ±plus-or-minus\pm 0.007 | 0.856 ±plus-or-minus\pm 0.005 | 0.785 ±plus-or-minus\pm 0.024 | 0.833 ±plus-or-minus\pm 0.011 |
| blastchar | 0.839 ±plus-or-minus\pm 0.010 | 0.842 ±plus-or-minus\pm 0.015 | 0.835 ±plus-or-minus\pm 0.014 | 0.816 ±plus-or-minus\pm 0.014 | 0.842 ±plus-or-minus\pm 0.012 |
| philippine | 0.821 ±plus-or-minus\pm 0.020 | 0.764 ±plus-or-minus\pm 0.018 | 0.834 ±plus-or-minus\pm 0.018 | 0.721 ±plus-or-minus\pm 0.008 | 0.757 ±plus-or-minus\pm 0.018 |
| insurance\_co | 0.697 ±plus-or-minus\pm 0.027 | 0.705 ±plus-or-minus\pm 0.054 | 0.744 ±plus-or-minus\pm 0.009 | 0.630 ±plus-or-minus\pm 0.061 | 0.647 ±plus-or-minus\pm 0.028 |
| spambase | 0.984 ±plus-or-minus\pm 0.004 | 0.980 ±plus-or-minus\pm 0.009 | 0.985 ±plus-or-minus\pm 0.005 | 0.975 ±plus-or-minus\pm 0.008 | 0.983 ±plus-or-minus\pm 0.004 |
| jasmine | 0.851 ±plus-or-minus\pm 0.015 | 0.856 ±plus-or-minus\pm 0.013 | 0.853 ±plus-or-minus\pm 0.015 | 0.816 ±plus-or-minus\pm 0.017 | 0.847 ±plus-or-minus\pm 0.017 |
| seismicbumps | 0.735 ±plus-or-minus\pm 0.028 | 0.699 ±plus-or-minus\pm 0.074 | 0.751 ±plus-or-minus\pm 0.096 | 0.701 ±plus-or-minus\pm 0.051 | 0.681 ±plus-or-minus\pm 0.084 |
| qsar\_bio | 0.910 ±plus-or-minus\pm 0.037 | 0.916 ±plus-or-minus\pm 0.036 | 0.918 ±plus-or-minus\pm 0.038 | 0.860 ±plus-or-minus\pm 0.038 | 0.914 ±plus-or-minus\pm 0.028 |

[◄](/html/2012.06676)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2012.06678)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2012.06678)
[View original  
on arXiv](https://arxiv.org/abs/2012.06678)[►](/html/2012.06679)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sun Mar 3 17:15:19 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
