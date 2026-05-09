---
arxiv: '1810.11921'
authors:
- Weiping Song
- Chence Shi
- Zhiping Xiao
- Zhijian Duan
- Yewen Xu
- Ming Zhang
- Jian Tang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural
  Networks'
url: http://arxiv.org/abs/1810.11921v2
year: 2018
---

[1810.11921] AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks















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



# AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

Weiping Song
Department of Computer Science, School of EECS, Peking University
[weiping.song@pku.edu.cn](mailto:weiping.song@pku.edu.cn)
, 
Chence Shi
Department of Computer Science, School of EECS, Peking University
[chenceshi@pku.edu.cn](mailto:chenceshi@pku.edu.cn)
, 
Zhiping Xiao
Department of Computer Science, University of California, Los Angeles
[patriciaxiao@g.ucla.edu](mailto:patriciaxiao@g.ucla.edu)
, 
Zhijian Duan, Yewen Xu
Department of Computer Science, School of EECS, Peking University
[zjduan,xuyewen@pku.edu.cn](mailto:zjduan,xuyewen@pku.edu.cn)
, 
Ming Zhang
Department of Computer Science, School of EECS, Peking University
[mzhang˙cs@pku.edu.cn](mailto:mzhang%CB%99cs@pku.edu.cn)
 and 
Jian Tang
Mila-Quebec AI Institute,
  
HEC Montreal & CIFAR AI Chair
[jian.tang@hec.ca](mailto:jian.tang@hec.ca)

(2018; 2019)

###### Abstract.

Click-through rate (CTR) prediction, which aims to predict the probability of a user clicking on an ad or an item, is critical to many online applications such as online advertising and recommender systems. The problem is very challenging since (1) the input features (e.g., the user id, user age, item id, item category) are usually sparse and high-dimensional, and (2) an effective prediction relies on high-order combinatorial features (a.k.a. cross features), which are very time-consuming to hand-craft by domain experts and are impossible to be enumerated. Therefore, there have been efforts in finding low-dimensional representations of the sparse and high-dimensional raw features and their meaningful combinations.

In this paper, we propose an effective and efficient method called the *AutoInt* to automatically learn the high-order feature interactions of input features. Our proposed algorithm is very general, which can be applied to both numerical and categorical input features. Specifically, we map both the numerical and categorical features into the same low-dimensional space. Afterwards, a multi-head self-attentive neural network with residual connections is proposed to explicitly model the feature interactions in the low-dimensional space. With different layers of the multi-head self-attentive neural networks, different orders of feature combinations of input features can be modeled.
The whole model can be efficiently fit on large-scale raw data in an end-to-end fashion. Experimental results on four real-world datasets show that our proposed approach not only outperforms existing state-of-the-art approaches for prediction but also offers good explainability.
Code is available at: <https://github.com/DeepGraphLearning/RecommenderSystems>.

High-order feature interactions, Self attention, CTR prediction, Explainable recommendation

††journalyear: 2018††copyright: acmlicensed††conference: Woodstock ’18: ACM Symposium on Neural Gaze Detection; June 03–05, 2018; Woodstock, NY††price: 15.00††doi: 10.1145/1122445.1122456††isbn: 978-1-4503-9999-9/18/06††copyright: acmcopyright††ccs: Information systems Recommender systems††ccs: Computing methodologies Neural networks††ccs: Computing methodologies Learning latent representations††journalyear: 2019††conference: The 28th ACM International Conference on Information and Knowledge Management; November 3–7, 2019; Beijing, China††booktitle: The 28th ACM International Conference on Information and Knowledge Management (CIKM ’19), November 3–7, 2019, Beijing, China††price: 15.00††doi: 10.1145/3357384.3357925††isbn: 978-1-4503-6976-3/19/11

## 1. Introduction

Predicting the probabilities of users clicking on ads or items (a.k.a., click-through rate prediction) is a critical problem for many applications such as online advertising and recommender systems (Graepel et al., [2010](#bib.bib11); He et al., [2014](#bib.bib16); Cheng et al., [2016](#bib.bib9)).
The performance of the prediction has a direct impact on the final revenue of the business providers. Due to its importance, it has attracted growing interest in both academia and industry communities.

Machine learning has been playing a key role in click-through rate prediction,
which is usually formulated as supervised learning with user profiles and item attributes as input features. The problem is very challenging for several reasons. First, the input features are extremely sparse and high-dimensional (McMahan et al., [2013](#bib.bib22); Shan
et al., [2016a](#bib.bib33); He and Chua, [2017](#bib.bib14); Cheng et al., [2016](#bib.bib9); Guo
et al., [2017](#bib.bib12)). In real-world applications, a considerable percentage of user’s demographics and item’s attributes are usually discrete and/or categorical. To make supervised learning methods applicable, these features are first converted to a one-hot encoding vector, which can easily result in features with millions of dimensions. Taking the well-known CTR prediction data Criteo111http://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/ as an example, the feature dimension is approximately 30 million with sparsity over 99.99%. With such sparse and high-dimensional input features, the machine learning models are easily overfitted.
Second, as shown in extensive literature (Cheng et al., [2016](#bib.bib9); Guo
et al., [2017](#bib.bib12); Lian
et al., [2018](#bib.bib20); Shan
et al., [2016a](#bib.bib33)), high-order feature interactions222In this paper, we will use “combinatorial feature” and “feature interaction” interchangeably as they are both used in the literature (Shan
et al., [2016a](#bib.bib33); Lian
et al., [2018](#bib.bib20); Guo
et al., [2017](#bib.bib12)) . are crucial for a good performance. For example, it is reasonable to recommend Mario Bros., a famous video game, to David, who is a ten-year-old boy. In this case, the third-order combinatorial feature ¡Gender=Male, Age=10, ProductCategory=VideoGame¿ is very informative for prediction. However, finding such meaningful high-order combinatorial features heavily relies on domain experts. Moreover, it is almost impossible to hand-craft all the meaningful combinations (Rendle, [2010](#bib.bib27); Cheng et al., [2016](#bib.bib9)). One may ask that we can enumerate all the possible high-order features and let machine learning models select the meaningful ones. However, enumerating all the possible high-order features will exponentially increase the dimension and sparsity of the input features, leading to a more serious problem of model overfitting. Therefore, there has been extensive efforts in the communities in finding low-dimensional representations of the sparse and high-dimensional input features and meanwhile modeling different orders of feature combinations.

For example, Factorization Machines (FM) (Rendle, [2010](#bib.bib27)), which combine polynomial regression models with factorization
techniques, are developed to model feature interactions and have been proved effective for various tasks (Rendle et al., [2011](#bib.bib29), [2010](#bib.bib28)). However, limited by its polynomial fitting time, it is only effective for modeling low-order feature interactions and impractical to capture high-order feature interactions. Recently, many works (He and Chua, [2017](#bib.bib14); Cheng et al., [2016](#bib.bib9); Guo
et al., [2017](#bib.bib12); Wang
et al., [2017](#bib.bib39)) based on deep neural networks have been proposed to model the high-order feature interactions. Specifically, multiple layers of non-linear neural networks are usually used to capture the high-order feature interactions. However, such kinds of methods suffer from two limitations. First, fully-connected neural networks have been shown inefficient in learning multiplicative feature interactions (Beutel et al., [2018](#bib.bib5)).
Second, since these models learn the feature interactions in an implicit way, they lack good explanation on which feature combinations are meaningful. Therefore, we are looking for an approach that is able to explicitly model different orders of feature combinations, represent the entire features into low-dimensional spaces, and meanwhile offer good model explainability.

In this paper, we propose such an approach based on the multi-head self-attention mechanism (Vaswani et al., [2017](#bib.bib37)). Our proposed approach learns effective low-dimensional representations of the sparse and high-dimensional input features and is applicable to both the categorical and/or numerical input features. Specifically, both the categorical and numerical features are first embedded into low-dimensional spaces, which reduces the dimension of the input features and meanwhile allows different types of features to interact with each other via vector arithmetic (e.g., summation and inner product).
Afterwards, we propose a novel interacting layer to promote the interactions between different features. Within each interacting layer, each feature is allowed to interact with all the other features and is able to automatically identify relevant features to form meaningful higher-order features via the multi-head attention mechanism (Vaswani et al., [2017](#bib.bib37)). Moreover, the multi-head mechanism projects a feature into multiple subspaces, and hence it can capture different feature interactions in different subspaces. Such an interacting layer models the one-step interaction between the features. By stacking multiple interacting layers, we are able to model different orders of feature interactions. In practice, the residual connection (He
et al., [2016](#bib.bib13)) is added to the interacting layer, which allows combining different orders of feature combinations. We use the attention mechanism for measuring the correlations between features, which offers good model explainability.

To summarize, in this paper we make the following contributions:

* •

  We propose to study the problem of explicitly learning high-order feature interactions and meanwhile finding models with good explainability for the problem.
* •

  We propose a novel approach based on self-attentive neural network, which can automatically learn high-order feature interactions and efficiently handle large-scale high-dimensional sparse data.
* •

  We conducted extensive experiments on several real-world data sets. Experimental results on the task of CTR prediction show that our proposed approach not only outperforms existing state-of-the-art approaches for prediction but also offers good model explainability.

Our work is organized as follows. In Section 2, we summarize the related work. Section 3 formally defines our problem. Section 4 presents the proposed approach to learn feature interactions. In Section 5, we present the experimental results and detailed analysis. We conclude this paper and point out the future work in Section 6.

## 2. Related work

Our work is relevant to three lines of work: 1) Click-through rate prediction in recommender systems and online advertising, 2) techniques for learning feature interactions, and 3) self-attention mechanism and residual networks in the literature of deep learning.

### 2.1. Click-through Rate Prediction

Predicting click-through rates is important to many Internet companies, and various systems have been developed by different companies (Richardson
et al., [2007](#bib.bib30); Graepel et al., [2010](#bib.bib11); McMahan et al., [2013](#bib.bib22); He et al., [2014](#bib.bib16); Cheng et al., [2016](#bib.bib9); Covington
et al., [2016](#bib.bib10); Zhou et al., [2018](#bib.bib44)). For example, Google developed the Wide&Deep(Cheng et al., [2016](#bib.bib9)) learning system for recommender systems, which combines the advantages of both the linear shallow models and deep models. The system achieves remarkable performance in APP recommendation. The problem also receives a lot of attention in the academic communities. For example, Shan
et al. ([2016b](#bib.bib32)) proposed a context-aware CTR prediction method which factorized three-way ¡user, ad, context¿ tensor. Oentaryo
et al. ([2014](#bib.bib25)) developed hierarchical importance-aware factorization machine to model dynamic impacts of ads.

### 2.2. Learning Feature Interactions

Learning feature interactions is a fundamental problem and therefore extensively studied in the literature. A well-known example is Factorization Machines (FM) (Rendle, [2010](#bib.bib27)), which were proposed to mainly capture the first- and second-order feature interactions and have been proved effective for many tasks in recommender systems (Rendle et al., [2010](#bib.bib28), [2011](#bib.bib29)). Afterwards, different variants of factorization machines have been proposed. For example, Field-aware Factorization Machines (FFM) (Juan
et al., [2016](#bib.bib17)) modeled fine-grained interactions between features from different fields. GBFM (Cheng
et al., [2014](#bib.bib8)) and AFM (Xiao
et al., [2017](#bib.bib41)) considered the importance of different second-order feature interactions. However, all these approaches focus on modeling low-order feature interactions.

There are some recent works that model high-order feature interactions. For example, NFM (He and Chua, [2017](#bib.bib14)) stacked deep neural networks on top of the output of the second-order feature interactions to model higher-order features. Similarly, PNN (Qu
et al., [2016](#bib.bib26)), FNN (Zhang
et al., [2016](#bib.bib42)), DeepCrossing (Shan
et al., [2016a](#bib.bib33)), Wide&Deep (Cheng et al., [2016](#bib.bib9)) and DeepFM (Guo
et al., [2017](#bib.bib12)) utilized feed-forward neural networks to model high-order feature interactions. However, all these approaches learn the high-order feature interactions in an implicit way and therefore lack good model explainability.
On the contrary, there are three lines of works that learn feature interactions in an explicit fashion. First, Deep&Cross (Wang
et al., [2017](#bib.bib39)) and xDeepFM (Lian
et al., [2018](#bib.bib20)) took outer product of features at the bit- and vector-wise level respectively. Although they perform explicit feature interactions, it is not trivial to explain which combinations are useful.
Second, some tree-based methods (Zhu
et al., [2017](#bib.bib45); Zhao
et al., [2017](#bib.bib43); Wang
et al., [2018](#bib.bib40)) combined the power of embedding-based models and tree-based models but had to break training procedure into multiple stages. Third, HOFM (Blondel
et al., [2016a](#bib.bib6)) proposed efficient training algorithms for high-order factorization machines. However, HOFM requires too many parameters and only its low-order (usually less than 5) form can be practically used.
Different from existing work, we explicitly model feature interactions with attention mechanism in an end-to-end manner, and probe the learned feature combinations via visualization.

### 2.3. Attention and Residual Networks

Our proposed model makes use of the latest techniques in the literature of deep learning: attention (Bahdanau
et al., [2015](#bib.bib3)) and residual networks (He
et al., [2016](#bib.bib13)). Attention is first proposed in the context of neural machine translation (Bahdanau
et al., [2015](#bib.bib3))
and has been proved effective in a variety of tasks such as question answering (Sukhbaatar et al., [2015](#bib.bib36)), text summarization (Rush
et al., [2015](#bib.bib31)), and recommender systems (Zhou et al., [2018](#bib.bib44); Song et al., [2019](#bib.bib34); He
et al., [2018](#bib.bib15)). Vaswani et al. ([2017](#bib.bib37)) further proposed multi-head self-attention to model complicated dependencies between words in machine translation.

Residual networks (He
et al., [2016](#bib.bib13)) achieved state-of-the-art performance in the ImageNet contest. Since the residual connection, which can be simply formalized as y=F​(x)+x𝑦𝐹𝑥𝑥y=F(x)+x, encourages gradient flow through interval layers, it becomes a popular network structure for training very deep neural networks.

## 3. Problem Definition

We first formally define the problem of click-through rate (CTR) prediction as follows:

DEFINITION 1. (CTR Prediction) Let 𝐱∈ℝn𝐱superscriptℝ𝑛\mathbf{x}\in\mathbb{R}^{n} denotes the concatenation of user u𝑢u’s features and item v𝑣v’s features, where categorical features are represented with one-hot encoding, and n𝑛n is the dimension of concatenated features. The problem of click-through rate prediction aims to predict the probability of user u𝑢u clicking on item v𝑣v according to the feature vector 𝐱𝐱\mathbf{x}.

A straightforward solution for CTR prediction is to treat 𝐱𝐱\mathbf{x} as the input features and deploy the off-the-shelf classifiers such as logistic regression. However, since the original feature vector 𝐱𝐱\mathbf{x} is very sparse and high-dimensional, the model will be easily overfitted. Therefore, it is desirable to represent the raw input features in low-dimensional continuous spaces. Moreover, as shown in existing literature, it is crucial to utilize the higher-order combinatorial features to yield good prediction performance (Rendle, [2010](#bib.bib27); Cheng et al., [2016](#bib.bib9); Shan
et al., [2016a](#bib.bib33); Novikov
et al., [2016](#bib.bib24); Guo
et al., [2017](#bib.bib12); Blondel
et al., [2016b](#bib.bib7)). Specifically, we define the high-order combinatorial features as follows:

DEFINITION 2. (p-order Combinatorial Feature) Given input feature vector 𝐱∈ℝn𝐱superscriptℝ𝑛\mathbf{x}\in\mathbb{R}^{n}, a p-order combinatorial feature is defined as
g​(xi1,…,xip)𝑔subscript𝑥subscript𝑖1…subscript𝑥subscript𝑖𝑝g(x\_{i\_{1}},...,x\_{i\_{p}})
, where each feature comes from a distinct field, p𝑝p is the number of involved feature fields, and g​(⋅)𝑔⋅g(\cdot) is a non-additive combination function, such as multiplication (Rendle, [2010](#bib.bib27)) and outer product (Lian
et al., [2018](#bib.bib20); Wang
et al., [2017](#bib.bib39)). For example, xi1×xi2subscript𝑥subscript𝑖1subscript𝑥subscript𝑖2x\_{i\_{1}}\times x\_{i\_{2}} is a second-order combinatorial feature involving xi1subscript𝑥subscript𝑖1x\_{i\_{1}} and xi2subscript𝑥subscript𝑖2x\_{i\_{2}}.

Traditionally, meaningful high-order combinatorial features are hand-crafted by domain experts. However, this is very time-consuming and hard to generalize to other domains. Besides, it is almost impossible to hand-craft all meaningful high-order features. Therefore, we aim to develop an approach that is able to automatically discover the meaningful high-order combinatorial features and meanwhile map all these features into low-dimensional continuous spaces. Formally, we define our problem as follows:

DEFINITION 3. (Problem Definition) Given an input feature vector 𝐱∈ℝn𝐱superscriptℝ𝑛\mathbf{x}\in\mathbb{R}^{n} for click-through rate prediction, our goal is to learn a low-dimensional representation of 𝐱𝐱\mathbf{x}, which models the high-order combinatorial features.

## 4. AutoInt: Automatic Feature Interaction Learning

In this section, we first give an overview of the proposed approach AutoInt, which can automatically learn feature interactions for CTR prediction. Next, we present a comprehensive description of how to learn a low-dimensional representation that models high-order combinatorial features without manual feature engineering.

![Refer to caption](/html/1810.11921/assets/x1.png)


Figure 1. Overview of our proposed model AutoInt. The details of embedding layer and interacting layer are illustrated in Figure [2](#S4.F2 "Figure 2 ‣ 4.2. Input Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") and Figure [3](#S4.F3 "Figure 3 ‣ 4.4. Interacting Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") respectively.

### 4.1. Overview

The goal of our approach is to map the original sparse and high-dimensional feature vector into low-dimensional spaces and meanwhile model the high-order feature interactions.
As shown in Figure [1](#S4.F1 "Figure 1 ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"), our proposed method takes the sparse feature vector 𝐱𝐱\mathbf{x} as input, followed by an embedding layer that projects all features (i.e., both categorical and numerical features) into the same low-dimensional space. Next, we feed embeddings of all fields into a novel interacting layer, which is implemented as a multi-head self-attentive neural network. For each interacting layer, high-order features are combined through the attention mechanism, and different kinds of combinations can be evaluated with the multi-head mechanisms, which map the features into different subspaces. By stacking multiple interacting layers, different orders of combinatorial features can be modeled.

The output of the final interacting layer is the low-dimensional representation of the input feature, which models the high-order combinatorial features and is further used for estimating the click-through rate through a sigmoid function. Next, we introduce the details of our proposed method.

### 4.2. Input Layer

We first represent user’s profiles and item’s attributes as a sparse vector, which is the concatenation of all fields. Specifically,

|  |  |  |  |
| --- | --- | --- | --- |
| (1) |  | 𝐱=[𝐱𝟏;𝐱𝟐;…;𝐱𝐌],𝐱  subscript𝐱1subscript𝐱2…subscript𝐱𝐌\mathbf{x}=[\mathbf{x\_{1}};\mathbf{x\_{2}};...;\mathbf{x\_{M}}], |  |

where M𝑀M is the number of total feature fields, and 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is the feature representation of the i𝑖i-th field. 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is a one-hot vector if the i𝑖i-th field is categorical (e.g., 𝐱𝟏subscript𝐱1\mathbf{x\_{1}} in Figure [2](#S4.F2 "Figure 2 ‣ 4.2. Input Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks")). 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is a scalar value if the i𝑖i-th field is numerical (e.g., 𝐱𝐌subscript𝐱𝐌\mathbf{x\_{M}} in Figure [2](#S4.F2 "Figure 2 ‣ 4.2. Input Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks")).

![Refer to caption](/html/1810.11921/assets/x2.png)


Figure 2. Illustration of input and embedding layer, where both categorical and numerical fields are represented by low-dimensional dense vectors.

### 4.3. Embedding Layer

Since the feature representations of the categorical features are very sparse and high-dimensional, a common way is to represent them into low-dimensional spaces (e.g., word embeddings). Specifically, we represent each categorical feature with a low-dimensional vector, i.e.,

|  |  |  |  |
| --- | --- | --- | --- |
| (2) |  | 𝐞𝐢=𝐕𝐢​𝐱𝐢,subscript𝐞𝐢subscript𝐕𝐢subscript𝐱𝐢\mathbf{e\_{i}}=\mathbf{V\_{i}}\mathbf{x\_{i}}, |  |

where 𝐕𝐢subscript𝐕𝐢\mathbf{V\_{i}} is an embedding matrix for field i𝑖i, and 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is an one-hot vector. Often times categorical features can be multi-valued, i.e., 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is a multi-hot vector. Take movie watching prediction as an example, there could be a feature field *Genre* which describes the types of a movie and it may be multi-valued (e.g., Drama and Romance for movie “Titanic”). To be compatible with multi-valued inputs, we further modify the Equation [2](#S4.E2 "In 4.3. Embedding Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") and represent the multi-valued feature field as the average of corresponding feature embedding vectors:

|  |  |  |  |
| --- | --- | --- | --- |
| (3) |  | 𝐞𝐢=1q​𝐕𝐢​𝐱𝐢,subscript𝐞𝐢1𝑞subscript𝐕𝐢subscript𝐱𝐢\mathbf{e\_{i}}=\frac{1}{q}\mathbf{V\_{i}}\mathbf{x\_{i}}, |  |

where q𝑞q is the number of values that a sample has for i𝑖i-th field and 𝐱𝐢subscript𝐱𝐢\mathbf{x\_{i}} is the multi-hot vector representation for this field.

To allow the interaction between categorical and numerical features, we also represent the numerical features in the same low-dimensional feature space. Specifically, we represent the numerical feature as

|  |  |  |  |
| --- | --- | --- | --- |
| (4) |  | 𝐞𝐦=𝐯𝐦​xm,subscript𝐞𝐦subscript𝐯𝐦subscript𝑥𝑚\mathbf{e\_{m}}=\mathbf{v\_{m}}x\_{m}, |  |

where 𝐯𝐦subscript𝐯𝐦\mathbf{v\_{m}} is an embedding vector for field m𝑚m, and xmsubscript𝑥𝑚x\_{m} is a scalar value.

By doing this, the output of the embedding layer would be a concatenation of multiple embedding vectors, as presented in Figure [2](#S4.F2 "Figure 2 ‣ 4.2. Input Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").

### 4.4. Interacting Layer

Once the numerical and categorical features live in the same low-dimensional space, we move to model high-order combinatorial features in the space. The key problem is to determine which features should be combined to form meaningful high-order features. Traditionally, this is accomplished by domain experts who create meaningful combinations based on their knowledge. In this paper, we tackle this problem with a novel method, the multi-head self-attention mechanism (Vaswani et al., [2017](#bib.bib37)).

Multi-head self-attentive network (Vaswani et al., [2017](#bib.bib37)) has recently achieved remarkable performance in modeling complicated relations. For example, it shows superiority for modeling arbitrary word dependency in machine translation (Vaswani et al., [2017](#bib.bib37)) and sentence embedding (Lin et al., [2017](#bib.bib21)), and has been successfully applied to capturing node similarities in graph embedding (Velickovic et al., [2018](#bib.bib38)). Here we extend this latest technique to model the correlations between different feature fields.

Specifically, we adopt the key-value attention mechanism (Miller et al., [2016](#bib.bib23)) to determine which feature combinations are meaningful. Taking the feature m𝑚m as an example, next we explain how to identify multiple meaningful high-order features involving feature m𝑚m. We first define the correlation between feature m𝑚m and feature k𝑘k under a specific attention head hℎh as follows:

|  |  |  |  |
| --- | --- | --- | --- |
| (5) |  | α𝐦,𝐤(𝐡)=exp⁡(ψ(h)​(𝐞𝐦,𝐞𝐤))∑l=1Mexp⁡(ψ(h)​(𝐞𝐦,𝐞𝐥)),ψ(h)​(𝐞𝐦,𝐞𝐤)=⟨𝐖𝐐𝐮𝐞𝐫𝐲(𝐡)​𝐞𝐦,𝐖𝐊𝐞𝐲(𝐡)​𝐞𝐤⟩,formulae-sequencesuperscriptsubscript𝛼  𝐦𝐤𝐡superscript𝜓ℎsubscript𝐞𝐦subscript𝐞𝐤superscriptsubscript𝑙1𝑀superscript𝜓ℎsubscript𝐞𝐦subscript𝐞𝐥superscript𝜓ℎsubscript𝐞𝐦subscript𝐞𝐤  subscriptsuperscript𝐖𝐡𝐐𝐮𝐞𝐫𝐲subscript𝐞𝐦subscriptsuperscript𝐖𝐡𝐊𝐞𝐲subscript𝐞𝐤\begin{gathered}\mathbf{\alpha\_{m,k}^{(h)}}=\frac{\exp(\psi^{(h)}(\mathbf{e\_{m}},\mathbf{e\_{k}}))}{\sum\_{l=1}^{M}\exp(\psi^{(h)}(\mathbf{e\_{m}},\mathbf{e\_{l}}))},\\ \psi^{(h)}(\mathbf{e\_{m}},\mathbf{e\_{k}})=\langle\mathbf{W^{(h)}\_{Query}}\mathbf{e\_{m}},\mathbf{W^{(h)}\_{Key}}\mathbf{e\_{k}}\rangle,\end{gathered} |  |

where ψ(h)​(⋅,⋅)superscript𝜓ℎ⋅⋅\psi^{(h)}(\cdot,\cdot) is an attention function which defines the similarity between the feature m𝑚m and k𝑘k. It can be defined as a neural network or as simple as inner product, i.e., ⟨⋅,⋅⟩

⋅⋅\langle\cdot,\cdot\rangle. In this work, we use inner product due to its simplicity and effectiveness. 𝐖𝐐𝐮𝐞𝐫𝐲(𝐡)subscriptsuperscript𝐖𝐡𝐐𝐮𝐞𝐫𝐲\mathbf{W^{(h)}\_{Query}}, 𝐖𝐊𝐞𝐲(𝐡)∈ℝd′×dsubscriptsuperscript𝐖𝐡𝐊𝐞𝐲superscriptℝsuperscript𝑑′𝑑\mathbf{W^{(h)}\_{Key}}\in\mathbb{R}^{d^{\prime}\times d} in Equation [5](#S4.E5 "In 4.4. Interacting Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") are transformation matrices which map the original embedding space ℝdsuperscriptℝ𝑑\mathbb{R}^{d} into a new space ℝd′superscriptℝsuperscript𝑑′\mathbb{R}^{d^{\prime}}. Next, we update the representation of feature m𝑚m in subspace hℎh via combining all relevant features guided by coefficients α𝐦,𝐤(𝐡)superscriptsubscript𝛼

𝐦𝐤𝐡\mathbf{\alpha\_{m,k}^{(h)}}:

|  |  |  |  |
| --- | --- | --- | --- |
| (6) |  | 𝐞~𝐦(𝐡)=∑k=1Mα𝐦,𝐤(𝐡)​(𝐖𝐕𝐚𝐥𝐮𝐞(𝐡)​𝐞𝐤),superscriptsubscript~𝐞𝐦𝐡superscriptsubscript𝑘1𝑀superscriptsubscript𝛼  𝐦𝐤𝐡subscriptsuperscript𝐖𝐡𝐕𝐚𝐥𝐮𝐞subscript𝐞𝐤\mathbf{\widetilde{e}\_{m}^{(h)}}=\sum\_{k=1}^{M}\mathbf{\alpha\_{m,k}^{(h)}}(\mathbf{W^{(h)}\_{Value}}\mathbf{e\_{k}}), |  |

where 𝐖𝐕𝐚𝐥𝐮𝐞(𝐡)∈ℝd′×dsubscriptsuperscript𝐖𝐡𝐕𝐚𝐥𝐮𝐞superscriptℝsuperscript𝑑′𝑑\mathbf{W^{(h)}\_{Value}}\in\mathbb{R}^{d^{\prime}\times d}.

Since 𝐞~𝐦(𝐡)∈ℝd′superscriptsubscript~𝐞𝐦𝐡superscriptℝsuperscript𝑑′\mathbf{\widetilde{e}\_{m}^{(h)}}\in\mathbb{R}^{d^{\prime}} is a combination of feature m𝑚m and its relevant features (under head hℎh), it represents a new combinatorial feature learned by our method. Furthermore, a feature is also likely to be involved in different combinatorial features, and we achieve this by using multiple heads, which create different subspaces and learn distinct feature interactions separately. We collect combinatorial features learned in all subspaces as follows:

![Refer to caption](/html/1810.11921/assets/x3.png)


Figure 3. The architecture of interacting layer. Combinatorial features are conditioned on attention weights, i.e., α𝐦(𝐡)superscriptsubscript𝛼𝐦𝐡\mathbf{\alpha\_{m}^{(h)}}.

|  |  |  |  |
| --- | --- | --- | --- |
| (7) |  | 𝐞~𝐦=𝐞~𝐦(𝟏)⊕𝐞~𝐦(𝟐)⊕⋯⊕𝐞~𝐦(𝐇),subscript~𝐞𝐦direct-sumsuperscriptsubscript~𝐞𝐦1superscriptsubscript~𝐞𝐦2⋯superscriptsubscript~𝐞𝐦𝐇\mathbf{\widetilde{e}\_{m}}=\mathbf{\widetilde{e}\_{m}^{(1)}}\oplus\mathbf{\widetilde{e}\_{m}^{(2)}}\oplus\cdot\cdot\cdot\oplus\mathbf{\widetilde{e}\_{m}^{(H)}}, |  |

where ⊕direct-sum\oplus is the concatenation operator, and H is the number of total heads.

To preserve previously learned combinatorial features, including raw individual (i.e., first-order) features, we add standard residual connections in our network.
Formally,

|  |  |  |  |
| --- | --- | --- | --- |
| (8) |  | 𝐞𝐦𝐑𝐞𝐬=ReLU​(𝐞~𝐦+𝐖𝐑𝐞𝐬​𝐞𝐦),superscriptsubscript𝐞𝐦𝐑𝐞𝐬ReLUsubscript~𝐞𝐦subscript𝐖𝐑𝐞𝐬subscript𝐞𝐦\mathbf{e\_{m}^{Res}}=\textit{ReLU}(\mathbf{\widetilde{e}\_{m}}+\mathbf{W\_{Res}}\mathbf{e\_{m}}), |  |

where 𝐖𝐑𝐞𝐬∈ℝd′​H×dsubscript𝐖𝐑𝐞𝐬superscriptℝsuperscript𝑑′𝐻𝑑\mathbf{W\_{Res}}\in\mathbb{R}^{d^{\prime}H\times d} is the projection matrix in case of dimension mismatching (He
et al., [2016](#bib.bib13)), and ReLU​(z)=m​a​x​(0,z)ReLU𝑧𝑚𝑎𝑥0𝑧\textit{ReLU}(z)=max(0,z) is a non-linear activation function.

With such an interacting layer, the representation of each feature 𝐞𝐦subscript𝐞𝐦\mathbf{e\_{m}} will be updated into a new feature representation 𝐞𝐦𝐑𝐞𝐬superscriptsubscript𝐞𝐦𝐑𝐞𝐬\mathbf{e\_{m}^{Res}}, which is a representation of high-order features. We can stack multiple such layers with the output of the previous interacting layer as the input of the next interacting layer. By doing this, we can model arbitrary-order combinatorial features.

### 4.5. Output Layer

The output of the interacting layer is a set of feature vectors {𝐞𝐦𝐑𝐞𝐬}m=1Msuperscriptsubscriptsuperscriptsubscript𝐞𝐦𝐑𝐞𝐬𝑚1𝑀\{\mathbf{e\_{m}^{Res}}\}\_{m=1}^{M}, which includes raw individual features reserved by residual block and combinatorial features learned via the multi-head self-attention mechanism. For final CTR prediction, we simply concatenate all of them and then apply a non-linear projection as follows:

|  |  |  |  |
| --- | --- | --- | --- |
| (9) |  | y^=σ​(𝐰𝐓​(𝐞𝟏𝐑𝐞𝐬⊕𝐞𝟐𝐑𝐞𝐬⊕⋯⊕𝐞𝐌𝐑𝐞𝐬)+b),^𝑦𝜎superscript𝐰𝐓direct-sumsuperscriptsubscript𝐞1𝐑𝐞𝐬superscriptsubscript𝐞2𝐑𝐞𝐬⋯superscriptsubscript𝐞𝐌𝐑𝐞𝐬𝑏\hat{y}=\sigma(\mathbf{w^{T}}(\mathbf{e\_{1}^{Res}}\oplus\mathbf{e\_{2}^{Res}}\oplus\cdot\cdot\cdot\oplus\mathbf{e\_{M}^{Res})}+b), |  |

where 𝐰∈ℝd′​H​M𝐰superscriptℝsuperscript𝑑′𝐻𝑀\mathbf{w}\in\mathbb{R}^{d^{\prime}HM} is a column projection vector which linearly combines concatenated features, b𝑏b is the bias, and σ​(x)=1/(1+e−x)𝜎𝑥11superscript𝑒𝑥\sigma(x)=1/(1+e^{-x}) transforms the values to users clicking probabilities.

### 4.6. Training

Our loss function is Log loss,
which is defined as follows:

|  |  |  |  |
| --- | --- | --- | --- |
| (10) |  | L​o​g​l​o​s​s=−1N​∑j=1N(yj​log⁡(y^j)+(1−yj)​log⁡(1−y^j)),𝐿𝑜𝑔𝑙𝑜𝑠𝑠1𝑁subscriptsuperscript𝑁𝑗1subscript𝑦𝑗subscript^𝑦𝑗1subscript𝑦𝑗1subscript^𝑦𝑗Logloss=-\frac{1}{N}\sum^{N}\_{j=1}(y\_{j}\log(\hat{y}\_{j})+(1-y\_{j})\log(1-\hat{y}\_{j})), |  |

where yjsubscript𝑦𝑗y\_{j} and y^jsubscript^𝑦𝑗\hat{y}\_{j} are ground truth of user clicks and estimated CTR respectively, j𝑗j indexes the training samples, and N𝑁N is the total number of training samples. The parameters to learn in our model are {𝐕𝐢subscript𝐕𝐢\mathbf{V\_{i}}, 𝐯𝐦subscript𝐯𝐦\mathbf{v\_{m}}, 𝐖𝐐𝐮𝐞𝐫𝐲(𝐡)subscriptsuperscript𝐖𝐡𝐐𝐮𝐞𝐫𝐲\mathbf{W^{(h)}\_{Query}}, 𝐖𝐊𝐞𝐲(𝐡)subscriptsuperscript𝐖𝐡𝐊𝐞𝐲\mathbf{W^{(h)}\_{Key}}, 𝐖𝐕𝐚𝐥𝐮𝐞(𝐡)subscriptsuperscript𝐖𝐡𝐕𝐚𝐥𝐮𝐞\mathbf{W^{(h)}\_{Value}}, 𝐖𝐑𝐞𝐬subscript𝐖𝐑𝐞𝐬\mathbf{W\_{Res}}, 𝐰𝐰\mathbf{w}, b𝑏b}, which are updated via minimizing the total Logloss using gradient descent.

### 4.7. Analysis Of AutoInt

Modeling Arbitrary Order Combinatorial Features. Given feature interaction operator defined by Equation [5](#S4.E5 "In 4.4. Interacting Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") - [8](#S4.E8 "In 4.4. Interacting Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"), we now analyze how low-order and high-order combinatorial features are modeled in our proposed model.

For simplicity, let’s assume there are four feature fields (i.e., M𝑀M=4) denoted by x1subscript𝑥1x\_{1}, x2subscript𝑥2x\_{2}, x3subscript𝑥3x\_{3} and x4subscript𝑥4x\_{4} respectively. Within the first interacting layer, each individual feature interacts with any other features through attention mechanism (i.e. Equation [5](#S4.E5 "In 4.4. Interacting Layer ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks")) and therefore a set of second-order feature combinations such as g​(x1,x2)𝑔subscript𝑥1subscript𝑥2g(x\_{1},x\_{2}), g​(x2,x3)𝑔subscript𝑥2subscript𝑥3g(x\_{2},x\_{3}) and g​(x3,x4)𝑔subscript𝑥3subscript𝑥4g(x\_{3},x\_{4}) are captured with distinct correlation weights, where the non-additive property of interaction function g​(⋅)𝑔⋅g(\cdot) (in DEFINITION 2) can be ensured by the non-linearity of activation function ReLU​(⋅)ReLU⋅\textit{ReLU}(\cdot). Ideally, combinatorial features that involve x1subscript𝑥1x\_{1} can be encoded into the updated representation of the first feature field 𝐞𝟏𝐑𝐞𝐬superscriptsubscript𝐞1𝐑𝐞𝐬\mathbf{e\_{1}^{Res}}.
As the same can be derived for other feature fields, all second-order feature interactions can be encoded in the output of the first interacting layer, where attention weights distill useful feature combinations.

Next, we prove that higher-order feature interactions can be modeled within the second interacting layer. Given the representation of the first feature field 𝐞𝟏𝐑𝐞𝐬superscriptsubscript𝐞1𝐑𝐞𝐬\mathbf{e\_{1}^{Res}} and the representation of the third feature field 𝐞𝟑𝐑𝐞𝐬superscriptsubscript𝐞3𝐑𝐞𝐬\mathbf{e\_{3}^{Res}} generated by the first interacting layer, third-order combinatorial features that involve x1subscript𝑥1x\_{1}, x2subscript𝑥2x\_{2} and x3subscript𝑥3x\_{3} can be modeled by allowing 𝐞𝟏𝐑𝐞𝐬superscriptsubscript𝐞1𝐑𝐞𝐬\mathbf{e\_{1}^{Res}} to attend on 𝐞𝟑𝐑𝐞𝐬superscriptsubscript𝐞3𝐑𝐞𝐬\mathbf{e\_{3}^{Res}} because 𝐞𝟏𝐑𝐞𝐬superscriptsubscript𝐞1𝐑𝐞𝐬\mathbf{e\_{1}^{Res}} contains the interaction g​(x1,x2)𝑔subscript𝑥1subscript𝑥2g(x\_{1},x\_{2}) and 𝐞𝟑𝐑𝐞𝐬superscriptsubscript𝐞3𝐑𝐞𝐬\mathbf{e\_{3}^{Res}} contains the individual feature x3subscript𝑥3x\_{3} (from residual connection). Moreover, the maximum order of combinatorial features grows exponentially with respect to the number of interacting layers. For example, fourth-order feature interaction g​(x1,x2,x3,x4)𝑔subscript𝑥1subscript𝑥2subscript𝑥3subscript𝑥4g(x\_{1},x\_{2},x\_{3},x\_{4}) can be captured by the combination of 𝐞𝟏𝐑𝐞𝐬superscriptsubscript𝐞1𝐑𝐞𝐬\mathbf{e\_{1}^{Res}} and 𝐞𝟑𝐑𝐞𝐬superscriptsubscript𝐞3𝐑𝐞𝐬\mathbf{e\_{3}^{Res}}, which contain the second-order interactions g​(x1,x2)𝑔subscript𝑥1subscript𝑥2g(x\_{1},x\_{2}) and g​(x3,x4)𝑔subscript𝑥3subscript𝑥4g(x\_{3},x\_{4}) respectively.
Therefore a few interacting layers will suffice to model high-order feature interactions.

Based on above analysis, we can see that AutoInt learns feature interactions with attention mechanism in a hierarchical manner, i.e., from low-order to high-order, and all low-order feature interactions are carried by residual connections. This is promising and reasonable because learning hierarchical representation has proven quite effective in computer vision and speech processing with deep neural networks (Lee
et al., [2011](#bib.bib19); Bengio
et al., [2013](#bib.bib4)).

Space Complexity.
The embedding layer, which is a shared component in neural network-based methods (Lian
et al., [2018](#bib.bib20); Guo
et al., [2017](#bib.bib12); Shan
et al., [2016a](#bib.bib33)), contains n​d𝑛𝑑nd parameters, where n𝑛n is the dimension of sparse representation of input feature and d𝑑d is the embedding size.
As an interacting layer contains following weight matrices: {𝐖𝐐𝐮𝐞𝐫𝐲(𝐡),𝐖𝐊𝐞𝐲(𝐡),𝐖𝐕𝐚𝐥𝐮𝐞(𝐡),𝐖𝐑𝐞𝐬

subscriptsuperscript𝐖𝐡𝐐𝐮𝐞𝐫𝐲subscriptsuperscript𝐖𝐡𝐊𝐞𝐲subscriptsuperscript𝐖𝐡𝐕𝐚𝐥𝐮𝐞subscript𝐖𝐑𝐞𝐬\mathbf{W^{(h)}\_{Query}},\mathbf{W^{(h)}\_{Key}},\mathbf{W^{(h)}\_{Value}},\mathbf{W\_{Res}}}, the number of parameters in an L𝐿L-layer network is L×(3​d​d′+d′​H​d)𝐿3𝑑superscript𝑑′superscript𝑑′𝐻𝑑L\times(3dd^{\prime}+d^{\prime}Hd), which is independent of the number of feature fields M𝑀M. Finally, there are d′​H​M+1superscript𝑑′𝐻𝑀1d^{\prime}HM+1 parameters in the output layer. As far as interacting layers are concerned, the space complexity is O​(L​d​d′​H)𝑂𝐿𝑑superscript𝑑′𝐻O(Ldd^{\prime}H). Note that H𝐻H and d′superscript𝑑′d^{\prime} are usually small (e.g., H=2​ and ​d′=32𝐻2 and superscript𝑑′32H=2\text{ and }d^{\prime}=32 in our experiments), which makes the interacting layer memory-efficient.

Time Complexity. Within each interacting layer, the computation cost is two-fold. First, calculating attention weights for one head takes O​(M​d​d′+M2​d′)𝑂𝑀𝑑superscript𝑑′superscript𝑀2superscript𝑑′O(Mdd^{\prime}+M^{2}d^{\prime}) time. Afterwards, forming combinatorial features under one head also takes O​(M​d​d′+M2​d′)𝑂𝑀𝑑superscript𝑑′superscript𝑀2superscript𝑑′O(Mdd^{\prime}+M^{2}d^{\prime}) time. Because we have H𝐻H heads, it takes O​(M​H​d′​(M+d))𝑂𝑀𝐻superscript𝑑′𝑀𝑑O(MHd^{\prime}(M+d)) time altogether. It is therefore efficient because H,d

𝐻𝑑H,d and d′superscript𝑑′d^{\prime} are usually small. We provide running time of AutoInt in Section [5.2](#S5.SS2 "5.2. Quantitative Results (RQ1) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").

## 5. Experiment

In this section, we move forward to evaluate the effectiveness of our proposed approach. We aim to answer the following questions:

* RQ1

  How does our proposed AutoInt perform on the problem of CTR prediction? Is it efficient for large-scale sparse and high-dimensional data?
* RQ2

  What are the influences of different model configurations?
* RQ3

  What are the dependency structures between different features? Is our proposed model explainable?
* RQ4

  Will integrating implicit feature interactions further improve the performance?

We first describe the experimental settings before answering these questions.

### 5.1. Experiment Setup

#### 5.1.1. Data Sets

We use four public real-world data sets. The statistics of the data sets are summarized in Table [1](#S5.T1 "Table 1 ‣ 5.1.1. Data Sets ‣ 5.1. Experiment Setup ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").
Criteo333https://www.kaggle.com/c/criteo-display-ad-challenge This is a benchmark dataset for CTR prediction, which has 45 million users’ clicking records on displayed ads. It contains 26 categorical feature fields and 13 numerical feature fields. Avazu444https://www.kaggle.com/c/avazu-ctr-prediction This dataset contains users’ mobile behaviors including whether a displayed mobile ad is clicked by a user or not. It has 23 feature fields spanning from user/device features to ad attributes.
KDD12555https://www.kaggle.com/c/kddcup2012-track2 This data set was released by KDDCup 2012, which originally aimed to predict the number of clicks. Since our work focuses on CTR prediction rather than the exact number of clicks, we treat this problem as a binary classification problem (1 for clicks¿0, 0 for without click), which is similar to FFM (Juan
et al., [2016](#bib.bib17)).
MovieLens-1M666https://grouplens.org/datasets/movielens/ This dataset contains users’ ratings on movies. During binarization, we treat samples with a rating less than 3 as negative samples because a low score indicates that the user does not like the movie. We treat samples with a rating greater than 3 as positive samples and remove neutral samples, i.e., a rating equal to 3.

Data Preparation First, we remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature “¡unknown¿”, where threshold is set to {10, 5, 10} for Criteo, Avazu and KDD12 data sets respectively.
Second, since numerical features may have large variance and hurt machine learning algorithms, we normalize numerical values by transforming a value z𝑧z to l​o​g2​(z)𝑙𝑜superscript𝑔2𝑧log^{2}(z) if z>2𝑧2z>2, which is proposed by the winner of Criteo Competition777<https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf>. Third, we randomly select 80% of all samples for training and randomly split the rest into validation and test sets of equal size.

Table 1. Statistics of evaluation data sets.

| Data | #Samples | #Fields | #Features (Sparse) |
| --- | --- | --- | --- |
| Criteo | 45,840,617 | 39 | 998,960 |
| Avazu | 40,428,967 | 23 | 1,544,488 |
| KDD12 | 149,639,105 | 13 | 6,019,086 |
| MovieLens-1M | 739,012 | 7 | 3,529 |




Table 2. Effectiveness Comparison of Different Algorithms. We highlight that our proposed model almost outperforms all baselines across four data sets and both metrics. Further analysis is provided in Section [5.2](#S5.SS2 "5.2. Quantitative Results (RQ1) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model Class | Model | Criteo | | Avazu | | KDD12 | | MovieLens-1M | |
| AUC | Logloss | AUC | Logloss | AUC | Logloss | AUC | Logloss |
| First-order | LR | 0.7820 | 0.4695 | 0.7560 | 0.3964 | 0.7361 | 0.1684 | 0.7716 | 0.4424 |
| Second-order | FM (Rendle, [2010](#bib.bib27)) | 0.7836 | 0.4700 | 0.7706 | 0.3856 | 0.7759 | 0.1573 | 0.8252 | 0.3998 |
| AFM(Xiao et al., [2017](#bib.bib41)) | 0.7938 | 0.4584 | 0.7718 | 0.3854 | 0.7659 | 0.1591 | 0.8227 | 0.4048 |
| High-order | DeepCrossing (Shan et al., [2016a](#bib.bib33)) | 0.8009 | 0.4513 | 0.7643 | 0.3889 | 0.7715 | 0.1591 | 0.8448 | 0.3814 |
| NFM (He and Chua, [2017](#bib.bib14)) | 0.7957 | 0.4562 | 0.7708 | 0.3864 | 0.7515 | 0.1631 | 0.8357 | 0.3883 |
| CrossNet (Wang et al., [2017](#bib.bib39)) | 0.7907 | 0.4591 | 0.7667 | 0.3868 | 0.7773 | 0.1572 | 0.7968 | 0.4266 |
| CIN (Lian et al., [2018](#bib.bib20)) | 0.8009 | 0.4517 | 0.7758 | 0.3829 | 0.7799 | 0.1566 | 0.8286 | 0.4108 |
| HOFM (Blondel et al., [2016a](#bib.bib6)) | 0.8005 | 0.4508 | 0.7701 | 0.3854 | 0.7707 | 0.1586 | 0.8304 | 0.4013 |
| AutoInt (ours) | 0.8061\*\* | 0.4455\*\* | 0.7752 | 0.3824 | 0.7883\*\* | 0.1546\*\* | 0.8456\* | 0.3797\*\* |

* •

  AutoInt outperforms the strongest baseline w.r.t. Criteo, KDD12 and MovieLens-1M data at the: \*\* 0.01 and \* 0.05 level, unpaired t-test.

#### 5.1.2. Evaluation Metrics

We use two popular metrics to evaluate the performance of all methods.

AUC Area Under the ROC Curve (AUC) measures the probability that a CTR predictor will assign a higher score to a randomly chosen positive item than a randomly chosen negative item. A higher AUC indicates a better performance.

Logloss Since all models attempt to minimize the Logloss defined by Equation [10](#S4.E10 "In 4.6. Training ‣ 4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"), we use it as a straightforward metric.

It is noticeable that a slightly higher AUC or lower Logloss at 0.001-level is regarded significant for CTR prediction task, which has also been pointed out in existing works (Cheng et al., [2016](#bib.bib9); Guo
et al., [2017](#bib.bib12); Wang
et al., [2017](#bib.bib39)).

#### 5.1.3. Competing Models

We compare the proposed approach with three classes of previous models. (A) the linear approach that only uses individual features. (B) factorization machines-based methods that take into account second-order combinatorial features. (C) techniques that can capture high-order feature interactions. We associate the model classes with model names accordingly.

LR (A). LR only models the linear combination of raw features.

FM (Rendle, [2010](#bib.bib27)) (B). FM uses factorization techniques to model second-order feature interactions.

AFM (Xiao
et al., [2017](#bib.bib41)) (B). AFM is one of the state-of-the-art models that capture second-order feature interactions. It extends FM by using attention mechanism to distinguish the different importance of second-order combinatorial features.

DeepCrossing (Shan
et al., [2016a](#bib.bib33)) (C). DeepCrossing utilizes deep fully-connected neural networks with residual connections to learn non-linear feature interactions in an implicit fashion.

NFM (He and Chua, [2017](#bib.bib14)) (C). NFM stacks deep neural networks on top of second-order feature interaction layer. High-order feature interactions are implicitly captured by the nonlinearity of neural networks.

CrossNet (Wang
et al., [2017](#bib.bib39)) (C). Cross Network, which is the core of Deep&Cross model, takes outer product of concatenated feature vector at the bit-wise level to model feature interactions explicitly.

CIN (Lian
et al., [2018](#bib.bib20)) (C). Compressed Interaction Network, which is the core of xDeepFM model, takes outer product of stacked feature matrix at vector-wise level.

HOFM (Blondel
et al., [2016a](#bib.bib6)) (C). HOFM proposes efficient kernel-based algorithms for training high-order factorization machines. Follow settings in [Blondel
et al.](#bib.bib6) (Blondel
et al., [2016a](#bib.bib6)) and [He and Chua](#bib.bib14) (He and Chua, [2017](#bib.bib14)), we build a third-order factorization machine using public implementation.

We will compare with the full models of CrossNet and CIN, i.e., Deep&Cross and xDeepFM, under the setting of joint training with plain DNN later (i.e., Section [5.5](#S5.SS5 "5.5. Integrating Implicit Interactions (RQ4) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks")).

![Refer to caption](/html/1810.11921/assets/x4.png)


(a) Criteo

![Refer to caption](/html/1810.11921/assets/x5.png)


(b) Avazu

![Refer to caption](/html/1810.11921/assets/x6.png)


(c) KDD12

![Refer to caption](/html/1810.11921/assets/x7.png)


(d) MovieLens-1M

Figure 4. Efficiency Comparison of Different Algorithms in terms of Run Time. “DC” and “CN” are DeepCrossing and CrossNet for short, respectively. Since HOFM cannot be fit on one GPU card for the KDD12 dataset, extra communication cost makes it most time-consuming. Further analysis is presented in Section [5.2](#S5.SS2 "5.2. Quantitative Results (RQ1) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").

#### 5.1.4. Implementation Details

All methods are implemented in TensorFlow(Abadi et al., [2016](#bib.bib2)). For AutoInt and all baseline methods, we empirically set embedding dimension d𝑑d to 16 and batch size to 1024. AutoInt has three interacting layers and the number of hidden units d′superscript𝑑′d^{\prime} is 32 in default setting. Within each interacting layer, the number of attention head is two888We also tried different number of attention heads. The performance of using one head is inferior to that of two heads, and the improvement of further increasing head number is not significant.. To prevent overfitting, we use grid search to select dropout rate (Srivastava et al., [2014](#bib.bib35)) from {0.1 - 0.9} for MovieLens-1M data set, and we found dropout is not necessary for other three large data sets.
For baseline methods, we use one hidden layer of size 200 on top of Bi-Interaction layer for NFM as recommended by their paper. For CN and CIN, we use three interaction layers following AutoInt. DeepCrossing has four feed-forward layers and the number of hidden units is 100, because it performs poorly when using three neural layers. Once all network structures are fixed, we also apply grid search to baseline methods for optimal hype-parameters.
Finally, we use Adam (Kingma and Ba, [2015](#bib.bib18)) to optimize all deep neural network-based models.

### 5.2. Quantitative Results (RQ1)

Evaluation of Effectiveness
  
We summarize the results averaged over 10 different runs into Table [2](#S5.T2 "Table 2 ‣ 5.1.1. Data Sets ‣ 5.1. Experiment Setup ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks").
We have the following observations:
(1) FM and AFM, which explore second-order feature interactions, consistently outperform LR by a large margin on all datasets, which indicates that individual features are insufficient in CTR prediction.
(2) An interesting observation is the inferiority of some models which capture high-order feature interactions. For example, although DeepCrossing and NFM use the deep neural network as a core component to learning high-order feature interactions, they do not guarantee improvement over FM and AFM. This may attribute to the fact that they learn feature interactions in an implicit fashion. On the contrary, CIN does it explicitly and outperforms low-order models consistently.
(3) HOFM significantly outperforms FM on Criteo and MovieLens-1M datasets, which indicates that modeling third-order feature interactions can be beneficial to prediction performance.
(4) AutoInt achieves the best performance overall baseline methods on three of four real-world data sets. On Avazu data set, CIN performs a little better than AutoInt in AUC evaluation, but we get lower Logloss. Note that our proposed AutoInt shares the same structures as DeepCrossing except the feature interacting layer, which indicates using the attention mechanism to learn explicit combinatorial features is crucial.

Evaluation of Model Efficiency
  
We present the runtime results of different algorithms on four data sets in Figure [4](#S5.F4 "Figure 4 ‣ 5.1.3. Competing Models ‣ 5.1. Experiment Setup ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"). Unsurprisingly, LR is the most efficient algorithm due to its simplicity. FM and NFM perform similarly in terms of runtime because NFM only stacks a single feed-forward hidden layer on top of the second-order interaction layer. Among all listed methods, CIN, which achieves the best performance for prediction among all the baselines, is much more time-consuming due to its complicated crossing layer. This may make it impractical in the industrial scenarios. Note that AutoInt is sufficiently efficient, which is comparable to the efficient algorithms DeepCrossing and NFM.

We also compare the sizes of different models (i.e., the number of parameters) as another criterion for efficiency evaluation. As shown in Table [3](#S5.T3 "Table 3 ‣ 5.2. Quantitative Results (RQ1) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"), comparing to the best model CIN in the baseline models, the number of parameters in AutoInt is much smaller.

To summarize, our proposed AutoInt achieves the best performance among all the compared models. Compared to the most competitive baseline model CIN, AutoInt requires much fewer parameters and is much more efficient during online inference.

Table 3. Efficiency Comparison of Different Algorithms in terms of Model Size on Criteo data set. “DC” and “CN” are DeepCrossing and CrossNet for short, respectively. The counted parameters exclude the embedding layer.

| Model | DC | CN | CIN | NFM | AutoInt |
| --- | --- | --- | --- | --- | --- |
| #Params | 1.6×1051.6superscript1051.6\times 10^{5} | 3×1033superscript1033\times 10^{3} | 1.9×1061.9superscript1061.9\times 10^{6} | 4×1034superscript1034\times 10^{3} | 3.9×1043.9superscript1043.9\times 10^{4} |




Table 4. Ablation study comparing the performance of AutoInt with and without residual connections. AutoIntw/w/{}\_{\text{{w/}}} is the complete model while the AutoIntw/ow/o{}\_{\text{{w/o}}} is the model without residual connection.

| Data Sets | Models | AUC | Logloss |
| --- | --- | --- | --- |
| Criteo | AutoIntw/w/{}\_{\text{{w/}}} | 0.8061 | 0.4454 |
| AutoIntw/ow/o{}\_{\text{{w/o}}} | 0.8033 | 0.4478 |
| Avazu | AutoIntw/w/{}\_{\text{{w/}}} | 0.7752 | 0.3823 |
| AutoIntw/ow/o{}\_{\text{{w/o}}} | 0.7729 | 0.3836 |
| KDD12 | AutoIntw/w/{}\_{\text{{w/}}} | 0.7888 | 0.1545 |
| AutoIntw/ow/o{}\_{\text{{w/o}}} | 0.7831 | 0.1557 |
| MovieLens-1M | AutoIntw/w/{}\_{\text{{w/}}} | 0.8460 | 0.3784 |
| AutoIntw/ow/o{}\_{\text{{w/o}}} | 0.8299 | 0.3959 |

### 5.3. Analysis (RQ2)

To further validate and gain deep insights into the proposed model, we conduct ablation study and compare several variants of AutoInt.

#### 5.3.1. Influence of Residual Structure

The standard AutoInt utilizes residual connections, which carry through all learned combinatorial features and therefore allow modeling very high-order combinations. To justify the contribution of residual units, we tease apart them from our standard model and keep other structures as they are. As presented in Table [4](#S5.T4 "Table 4 ‣ 5.2. Quantitative Results (RQ1) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"), we observe that the performance decrease on all datasets if residual connections are removed. Specifically, the full model outperforms the variant by a large margin on the KDD12 and MovieLens-1M data, which indicates residual connections are crucial to model high-order feature interactions in our proposed method.

#### 5.3.2. Influence of Network Depths

Our model learns high-order feature combinations by stacking multiple interacting layers (introduced in Section [4](#S4 "4. AutoInt: Automatic Feature Interaction Learning ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks")). Therefore, we are interested in how the performance change w.r.t. the number of interacting layers, i.e., the order of combinatorial features. Note that when there is no interacting layer (i.e., Number of layers equals zero), our model takes the weighted sum of raw individual features as input, i.e., no combinatorial features are considered.

The results are summarized in Figure [5](#S5.F5 "Figure 5 ‣ 5.3.2. Influence of Network Depths ‣ 5.3. Analysis (RQ2) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"). We can see that if one interacting layer is used, i.e., feature interactions are taken into account, the performance increase dramatically on both data sets, showing that combinatorial features are very informative for prediction. As the number of interacting layers further increases, i.e., higher-order combinatorial features are taken into account, the performance of the model further increases. When the number of layers reaches three, the performance becomes stable, showing that adding extremely high-order features are not informative for prediction.

Table 5. Results of Integrating Implicit Feature Interactions. We indicate the base model behind each method. The last two columns are average changes of AUC and Logloss compared to corresponding base models (“+”: increase, “-”: decrease).

| Model | Criteo | | Avazu | | KDD12 | | MovieLens-1M | | Avg. Changes | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUC | Logloss | AUC | Logloss | AUC | Logloss | AUC | Logloss | AUC | Logloss |
| Wide&Deep (LR) | 0.8026 | 0.4494 | 0.7749 | 0.3824 | 0.7549 | 0.1619 | 0.8300 | 0.3976 | +0.0292 | -0.0213 |
| DeepFM (FM) | 0.8066 | 0.4449 | 0.7751 | 0.3829 | 0.7867 | 0.1549 | 0.8437 | 0.3846 | +0.0142 | -0.0113 |
| Deep&Cross (CN) | 0.8067 | 0.4447 | 0.7731 | 0.3836 | 0.7872 | 0.1549 | 0.8446 | 0.3809 | +0.0200 | -0.0164 |
| xDeepFM (CIN) | 0.8070 | 0.4447 | 0.7770 | 0.3823 | 0.7820 | 0.1560 | 0.8463 | 0.3808 | +0.0068 | -0.0096 |
| AutoInt+ (ours) | 0.8083\*\* | 0.4434\*\* | 0.7774\* | 0.3811\*\* | 0.7898\*\* | 0.1543\*\* | 0.8488\*\* | 0.3753\*\* | +0.0023 | -0.0020 |

* •

  AutoInt+ outperforms the strongest baseline w.r.t. each data at the: \*\* 0.01 and \* 0.05 level, unpaired t-test.



![Refer to caption](/html/1810.11921/assets/x8.png)


(a) AUC

![Refer to caption](/html/1810.11921/assets/x9.png)


(b) Logloss

Figure 5. Performance w.r.t. the number of interacting layers. Results on Criteo and Avazu data sets are similar and hence omitted.



![Refer to caption](/html/1810.11921/assets/x10.png)


(a) AUC

![Refer to caption](/html/1810.11921/assets/x11.png)


(b) Logloss

Figure 6. Performance w.r.t. number of embedding dimensions. Results on Criteo and Avazu data sets are similar and hence omitted.

#### 5.3.3. Influence of Different Dimensions

Next, we investigate the performance w.r.t. the parameter d𝑑d, which is the output dimension of the embedding layer. On the KDD12 dataset, we can see that the performance continuously increase as we increase the dimension size since larger models are used for prediction. The results are different on the MovieLens-1M dataset. When the dimension size reaches 24, the performance begins to decrease. The reason is that this data set is small, and the model is overfitted when too many parameters are used.

![Refer to caption](/html/1810.11921/assets/x12.png)


(a) Label=1, Predicted CTR=0.89

![Refer to caption](/html/1810.11921/assets/x13.png)


(b) Overall feature interactions

Figure 7. Heat maps of attention weights for both case- and global-level feature interactions on MovieLens-1M. The axises represent feature fields ¡Gender, Age, Occupation, Zipcode, RequestTime, RealeaseTime, Genre¿. We highlight some learned combinatorial features in rectangles.

### 5.4. Explainable Recommendations (RQ3)

A good recommender system can not only provide good recommendations but also offer good explainability. Therefore, in this part, we present how our AutoInt is able to explain the recommendation results.
We take the MovieLens-1M dataset as an example.

Let’s look at a recommendation result suggested by our algorithm, i.e., a user likes an item. Figure [7](#S5.F7 "Figure 7 ‣ 5.3.3. Influence of Different Dimensions ‣ 5.3. Analysis (RQ2) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") (a) presents the correlations between different fields of input features, which are obtained by the attention score. We can see that AutoInt is able to identify the meaningful combinatorial feature ¡Gender=Male, Age=[18-24), MovieGenre=Action&Triller¿ (i.e., red dotted rectangle). This is very reasonable since young men are very likely to prefer action&triller movies.

We are also interested in what the correlations between different feature fields in the data are. Therefore, we measure the correlations between the feature fields according to their average attention score in the entire data. The correlations between different fields are summarized into Figure [7](#S5.F7 "Figure 7 ‣ 5.3.3. Influence of Different Dimensions ‣ 5.3. Analysis (RQ2) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") (b). We can see that ¡Gender, Genre¿, ¡Age, Genre¿, ¡RequestTime, ReleaseTime¿ and ¡Gender, Age, Genre¿ (i.e., solid green region) are strongly correlated, which are the explainable rules for recommendation in this domain.

### 5.5. Integrating Implicit Interactions (RQ4)

Feed-forward neural networks are capable of modeling implicit feature interactions and have been widely integrated into existing CTR prediction methods (Cheng et al., [2016](#bib.bib9); Guo
et al., [2017](#bib.bib12); Lian
et al., [2018](#bib.bib20)). To investigate whether integrating implicit feature interactions further improves the performance, we combine AutoInt with a two-layer feed-forward neural network by joint training.
We name the joint model AutoInt+ and compare it with the following algorithms:

* •

  Wide&Deep (Cheng et al., [2016](#bib.bib9)). Wide&Deep integrates the outputs of logistic regression and feed-forward neural networks.
* •

  DeepFM (Guo
  et al., [2017](#bib.bib12)). DeepFM combines trainditional second-order factorization machines and feed-forward neural network, with a shared embedding layer.
* •

  Deep&Cross (Wang
  et al., [2017](#bib.bib39)). Deep&Cross is the extension of CrossNet by integrating feed-forward neural networks.
* •

  xDeepFM (Lian
  et al., [2018](#bib.bib20)). xDeepFM is the extension of CIN by integrating feed-forward neural networks.

Table [5](#S5.T5 "Table 5 ‣ 5.3.2. Influence of Network Depths ‣ 5.3. Analysis (RQ2) ‣ 5. Experiment ‣ AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks") presents the averaged results (over 10 runs) of joint-training models. We have the following observations: 1) The performance of our method improves by joint training with feed-forward neural networks on all datasets. This indicates that integrating implicit feature interactions indeed boosts the predictive ability of our proposed model. However, as can be seen from last two columns, the magnitude of performance improvement is fairly small compared to other models, showing that our individual model AutoInt is quite powerful. 2) After integrating implicit feature interactions, AutoInt+ outperforms all competitive methods, and achieves new state-of-the-art performances on used CTR prediction data sets.

## 6. Conclusion and future work

In this work, we propose a novel CTR prediction model based on self-attention mechanism, which can automatically learn high-order feature interactions in an explicit fashion. The key to our method is the newly-introduced interacting layer, which allows each feature to interact with the others and to determine the relevance through learning. Experimental results on four real-world data sets demonstrate the effectiveness and efficiency of our proposed model. Besides, we provide good model explainability via visualizing the learned combinatorial features. When integrating with implicit feature interactions captured by feed-forward neural networks, we achieve better offline AUC and Logloss scores compared to the previous state-of-the-art methods.

For future work , we are interested in incorporating contextual information into our method and improving its performance for online recommender systems. Besides, we also plan to extend AutoInt for general machine learning tasks, such as regression, classification and ranking.

## 7. Acknowledgement

The authors would like to thank all the anonymous reviewers for their insightful comments. We thank Xiao Xiao and Jianbo Dong for the discussion on recommendation mechanism in China University MOOC platform.
We also thank Meng Qu for reviewing the initial version of this paper.
Weiping Song and Ming Zhang are supported by National Key Research and Development Program of China with Grant No. SQ2018AAA010010, Beijing Municipal Commission of Science and Technology under Grant No. Z181100008918005 as well as the National Natural Science Foundation of China (NSFC Grant Nos.61772039 and 91646202). Weiping Song is also supported by Chinese Scholarship Council. Jian Tang is supported by the Natural Sciences and Engineering Research Council of Canada, as well as the Canada CIFAR AI Chair Program.

## References

* (1)
* Abadi et al. (2016)

  Martín Abadi, Paul
  Barham, Jianmin Chen, Zhifeng Chen,
  Andy Davis, Jeffrey Dean,
  Matthieu Devin, Sanjay Ghemawat,
  Geoffrey Irving, et al.
  2016.
  TensorFlow: A System for Large-Scale Machine
  Learning.. In *OSDI*, Vol. 16.
  265–283.
* Bahdanau
  et al. (2015)

  Dzmitry Bahdanau,
  Kyunghyun Cho, and Yoshua Bengio.
  2015.
  Neural machine translation by jointly learning to
  align and translate. In *International Conference
  on Learning Representations*.
* Bengio
  et al. (2013)

  Yoshua Bengio, Aaron
  Courville, and Pascal Vincent.
  2013.
  Representation learning: A review and new
  perspectives.
  *IEEE transactions on pattern analysis and
  machine intelligence* 35, 8
  (2013), 1798–1828.
* Beutel et al. (2018)

  Alex Beutel, Paul
  Covington, Sagar Jain, Can Xu,
  Jia Li, Vince Gatto, and
  Ed H Chi. 2018.
  Latent Cross: Making Use of Context in Recurrent
  Recommender Systems. In *Proceedings of the
  Eleventh ACM International Conference on Web Search and Data Mining*. ACM,
  46–54.
* Blondel
  et al. (2016a)

  Mathieu Blondel, Akinori
  Fujino, Naonori Ueda, and Masakazu
  Ishihata. 2016a.
  Higher-order factorization machines. In
  *Advances in Neural Information Processing
  Systems*. 3351–3359.
* Blondel
  et al. (2016b)

  Mathieu Blondel, Masakazu
  Ishihata, Akinori Fujino, and Naonori
  Ueda. 2016b.
  Polynomial Networks and Factorization Machines: New
  Insights and Efficient Training Algorithms. In
  *International Conference on Machine Learning*.
  850–858.
* Cheng
  et al. (2014)

  Chen Cheng, Fen Xia,
  Tong Zhang, Irwin King, and
  Michael R Lyu. 2014.
  Gradient boosting factorization machines. In
  *Proceedings of the 8th ACM Conference on
  Recommender systems*. ACM, 265–272.
* Cheng et al. (2016)

  Heng-Tze Cheng, Levent
  Koc, Jeremiah Harmsen, Tal Shaked,
  Tushar Chandra, Hrishi Aradhye,
  Glen Anderson, Greg Corrado,
  Wei Chai, Mustafa Ispir, et al.
  2016.
  Wide & deep learning for recommender systems. In
  *Proceedings of the 1st Workshop on Deep Learning
  for Recommender Systems*. ACM, 7–10.
* Covington
  et al. (2016)

  Paul Covington, Jay
  Adams, and Emre Sargin.
  2016.
  Deep neural networks for youtube recommendations.
  In *Proceedings of the 10th ACM Conference on
  Recommender Systems*. ACM, 191–198.
* Graepel et al. (2010)

  Thore Graepel, Joaquin
  Quiñonero Candela, Thomas Borchert, and
  Ralf Herbrich. 2010.
  Web-scale Bayesian Click-through Rate Prediction
  for Sponsored Search Advertising in Microsoft’s Bing Search Engine. In
  *Proceedings of the 27th International Conference on
  International Conference on Machine Learning*. 13–20.
* Guo
  et al. (2017)

  Huifeng Guo, Ruiming
  Tang, Yunming Ye, Zhenguo Li, and
  Xiuqiang He. 2017.
  DeepFM: A Factorization-machine Based Neural
  Network for CTR Prediction. In *Proceedings of the
  26th International Joint Conference on Artificial Intelligence*.
  AAAI Press, 1725–1731.
* He
  et al. (2016)

  Kaiming He, Xiangyu
  Zhang, Shaoqing Ren, and Jian Sun.
  2016.
  Deep residual learning for image recognition. In
  *Proceedings of the IEEE conference on computer
  vision and pattern recognition*. 770–778.
* He and Chua (2017)

  Xiangnan He and Tat-Seng
  Chua. 2017.
  Neural factorization machines for sparse predictive
  analytics. In *Proceedings of the 40th
  International ACM SIGIR conference on Research and Development in Information
  Retrieval*. ACM, 355–364.
* He
  et al. (2018)

  Xiangnan He, Zhankui He,
  Jingkuan Song, Zhenguang Liu,
  Yu-Gang Jiang, and Tat-Seng Chua.
  2018.
  NAIS: Neural attentive item similarity model for
  recommendation.
  *IEEE Transactions on Knowledge and Data
  Engineering* 30, 12
  (2018), 2354–2366.
* He et al. (2014)

  Xinran He, Junfeng Pan,
  Ou Jin, Tianbing Xu, Bo
  Liu, Tao Xu, Yanxin Shi,
  Antoine Atallah, Ralf Herbrich,
  Stuart Bowers, et al.
  2014.
  Practical lessons from predicting clicks on ads at
  facebook. In *Proceedings of the Eighth
  International Workshop on Data Mining for Online Advertising*. ACM,
  1–9.
* Juan
  et al. (2016)

  Yuchin Juan, Yong Zhuang,
  Wei-Sheng Chin, and Chih-Jen Lin.
  2016.
  Field-aware factorization machines for CTR
  prediction. In *Proceedings of the 10th ACM
  Conference on Recommender Systems*. ACM, 43–50.
* Kingma and Ba (2015)

  Diederick P Kingma and
  Jimmy Ba. 2015.
  Adam: A method for stochastic optimization. In
  *International Conference on Learning
  Representations*.
* Lee
  et al. (2011)

  Honglak Lee, Roger
  Grosse, Rajesh Ranganath, and Andrew Y
  Ng. 2011.
  Unsupervised learning of hierarchical
  representations with convolutional deep belief networks.
  *Commun. ACM* 54,
  10 (2011), 95–103.
* Lian
  et al. (2018)

  Jianxun Lian, Xiaohuan
  Zhou, Fuzheng Zhang, Zhongxia Chen,
  Xing Xie, and Guangzhong Sun.
  2018.
  xDeepFM: Combining Explicit and Implicit Feature
  Interactions for Recommender Systems. In
  *Proceedings of the 24th ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining*.
  ACM, 1754–1763.
* Lin et al. (2017)

  Zhouhan Lin, Minwei Feng,
  Cicero Nogueira dos Santos, Mo Yu,
  Bing Xiang, Bowen Zhou, and
  Yoshua Bengio. 2017.
  A structured self-attentive sentence embedding. In
  *International Conference on Learning
  Representations*.
* McMahan et al. (2013)

  H. Brendan McMahan, Gary
  Holt, D. Sculley, Michael Young,
  Dietmar Ebner, Julian Grady,
  Lan Nie, Todd Phillips, et al.
  2013.
  Ad Click Prediction: A View from the Trenches. In
  *Proceedings of the 19th ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining*.
  ACM, 1222–1230.
* Miller et al. (2016)

  Alexander Miller, Adam
  Fisch, Jesse Dodge, Amir-Hossein Karimi,
  Antoine Bordes, and Jason Weston.
  2016.
  Key-Value Memory Networks for Directly Reading
  Documents. In *Proceedings of the 2016 Conference
  on Empirical Methods in Natural Language Processing*.
  Association for Computational Linguistics,
  1400–1409.
* Novikov
  et al. (2016)

  Alexander Novikov, Mikhail
  Trofimov, and Ivan Oseledets.
  2016.
  Exponential machines.
  *arXiv preprint arXiv:1605.03795*
  (2016).
* Oentaryo
  et al. (2014)

  Richard J Oentaryo,
  Ee-Peng Lim, Jia-Wei Low,
  David Lo, and Michael Finegold.
  2014.
  Predicting response in mobile advertising with
  hierarchical importance-aware factorization machine. In
  *Proceedings of the 7th ACM international conference
  on Web search and data mining*. ACM, 123–132.
* Qu
  et al. (2016)

  Yanru Qu, Han Cai,
  Kan Ren, Weinan Zhang,
  Yong Yu, Ying Wen, and
  Jun Wang. 2016.
  Product-based neural networks for user response
  prediction. In *Data Mining (ICDM), 2016 IEEE 16th
  International Conference on*. IEEE, 1149–1154.
* Rendle (2010)

  Steffen Rendle.
  2010.
  Factorization machines. In
  *Data Mining (ICDM), 2010 IEEE 10th International
  Conference on*. IEEE, 995–1000.
* Rendle et al. (2010)

  Steffen Rendle, Christoph
  Freudenthaler, and Lars Schmidt-Thieme.
  2010.
  Factorizing personalized markov chains for
  next-basket recommendation. In *Proceedings of the
  19th international conference on World wide web*. ACM,
  811–820.
* Rendle et al. (2011)

  Steffen Rendle, Zeno
  Gantner, Christoph Freudenthaler, and
  Lars Schmidt-Thieme. 2011.
  Fast context-aware recommendations with
  factorization machines. In *Proceedings of the 34th
  international ACM SIGIR conference on Research and development in Information
  Retrieval*. ACM, 635–644.
* Richardson
  et al. (2007)

  Matthew Richardson, Ewa
  Dominowska, and Robert Ragno.
  2007.
  Predicting clicks: estimating the click-through
  rate for new ads. In *Proceedings of the 16th
  international conference on World Wide Web*. ACM, 521–530.
* Rush
  et al. (2015)

  Alexander M. Rush, Sumit
  Chopra, and Jason Weston.
  2015.
  A Neural Attention Model for Abstractive Sentence
  Summarization. In *Proceedings of the 2015
  Conference on Empirical Methods in Natural Language Processing*.
  Association for Computational Linguistics,
  379–389.
* Shan
  et al. (2016b)

  Lili Shan, Lei Lin,
  Chengjie Sun, and Xiaolong Wang.
  2016b.
  Predicting ad click-through rates via feature-based
  fully coupled interaction tensor factorization.
  *Electronic Commerce Research and
  Applications* 16 (2016),
  30–42.
* Shan
  et al. (2016a)

  Ying Shan, T Ryan Hoens,
  Jian Jiao, Haijing Wang,
  Dong Yu, and JC Mao.
  2016a.
  Deep crossing: Web-scale modeling without manually
  crafted combinatorial features. In *Proceedings of
  the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
  Mining*. ACM, 255–262.
* Song et al. (2019)

  Weiping Song, Zhiping
  Xiao, Yifan Wang, Laurent Charlin,
  Ming Zhang, and Jian Tang.
  2019.
  Session-based Social Recommendation via Dynamic
  Graph Attention Networks. In *Proceedings of the
  Twelfth ACM International Conference on Web Search and Data Mining*. ACM,
  555–563.
* Srivastava et al. (2014)

  Nitish Srivastava,
  Geoffrey Hinton, Alex Krizhevsky,
  Ilya Sutskever, and Ruslan
  Salakhutdinov. 2014.
  Dropout: A simple way to prevent neural networks
  from overfitting.
  *The Journal of Machine Learning Research*
  15, 1 (2014),
  1929–1958.
* Sukhbaatar et al. (2015)

  Sainbayar Sukhbaatar,
  Jason Weston, Rob Fergus,
  et al. 2015.
  End-to-end memory networks. In
  *Advances in neural information processing
  systems*. 2440–2448.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam
  Shazeer, Niki Parmar, Jakob Uszkoreit,
  Llion Jones, Aidan N Gomez,
  Łukasz Kaiser, and Illia
  Polosukhin. 2017.
  Attention is all you need. In
  *Advances in Neural Information Processing
  Systems*. 6000–6010.
* Velickovic et al. (2018)

  Petar Velickovic, Guillem
  Cucurull, Arantxa Casanova, Adriana
  Romero, Pietro Lio, and Yoshua
  Bengio. 2018.
  Graph Attention Networks. In
  *International Conference on Learning
  Representations*.
* Wang
  et al. (2017)

  Ruoxi Wang, Bin Fu,
  Gang Fu, and Mingliang Wang.
  2017.
  Deep & Cross Network for Ad Click Predictions. In
  *Proceedings of the ADKDD’17*.
  ACM, 12:1–12:7.
* Wang
  et al. (2018)

  Xiang Wang, Xiangnan He,
  Fuli Feng, Liqiang Nie, and
  Tat-Seng Chua. 2018.
  TEM: Tree-enhanced Embedding Model for Explainable
  Recommendation. In *Proceedings of the 2018 World
  Wide Web Conference on World Wide Web*. International World Wide Web
  Conferences Steering Committee, 1543–1552.
* Xiao
  et al. (2017)

  Jun Xiao, Hao Ye,
  Xiangnan He, Hanwang Zhang,
  Fei Wu, and Tat-Seng Chua.
  2017.
  Attentional factorization machines: learning the
  weight of feature interactions via attention networks. In
  *Proceedings of the 26th International Joint
  Conference on Artificial Intelligence*. AAAI Press,
  3119–3125.
* Zhang
  et al. (2016)

  Weinan Zhang, Tianming
  Du, and Jun Wang. 2016.
  Deep learning over multi-field categorical data.
  In *European conference on information retrieval*.
  Springer, 45–57.
* Zhao
  et al. (2017)

  Qian Zhao, Yue Shi, and
  Liangjie Hong. 2017.
  GB-CENT: Gradient Boosted Categorical Embedding and
  Numerical Trees. In *Proceedings of the 26th
  International Conference on World Wide Web*. International World Wide Web
  Conferences Steering Committee, 1311–1319.
* Zhou et al. (2018)

  Guorui Zhou, Xiaoqiang
  Zhu, Chenru Song, Ying Fan,
  Han Zhu, Xiao Ma,
  Yanghui Yan, Junqi Jin,
  Han Li, and Kun Gai.
  2018.
  Deep Interest Network for Click-Through Rate
  Prediction. In *Proceedings of the 24th ACM SIGKDD
  International Conference on Knowledge Discovery and Data Mining*.
  ACM, 1059–1068.
* Zhu
  et al. (2017)

  Jie Zhu, Ying Shan,
  JC Mao, Dong Yu, Holakou
  Rahmanian, and Yi Zhang.
  2017.
  Deep embedding forest: Forest-based serving with
  deep embedding features. In *Proceedings of the
  23rd ACM SIGKDD International Conference on Knowledge Discovery and Data
  Mining*. ACM, 1703–1711.

[◄](/html/1810.11920)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1810.11921)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1810.11921)
[View original  
on arXiv](https://arxiv.org/abs/1810.11921)[►](/html/1810.11922)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 2 20:11:50 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
