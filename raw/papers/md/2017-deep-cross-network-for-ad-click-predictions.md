---
arxiv: '1708.05123'
authors:
- Ruoxi Wang
- Bin Fu
- Gang Fu
- Mingliang Wang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Deep & Cross Network for Ad Click Predictions
url: http://arxiv.org/abs/1708.05123v1
year: 2017
---

# Deep & Cross Network for Ad Click Predictions

Ruoxi Wang
Stanford UniversityStanfordCA
[ruoxi@stanford.edu](mailto:ruoxi@stanford.edu)
, 
Bin Fu
Google Inc.New YorkNY
[binfu@google.com](mailto:binfu@google.com)
, 
Gang Fu
Google Inc.New YorkNY
[thomasfu@google.com](mailto:thomasfu@google.com)
 and 
Mingliang Wang
Google Inc.New YorkNY
[mlwang@google.com](mailto:mlwang@google.com)

###### Abstract.

Feature engineering has been the key to the success of many prediction models. However, the process is nontrivial and often requires manual feature engineering or exhaustive searching. DNNs are able to automatically learn feature interactions; however, they generate all the interactions implicitly, and are not necessarily efficient in learning all types of cross features. In this paper, we propose the Deep & Cross Network (DCN) which keeps the benefits of a DNN model, and beyond that, it introduces a novel cross network that is more efficient in learning certain bounded-degree feature interactions. In particular, DCN explicitly applies feature crossing at each layer, requires no manual feature engineering, and adds negligible extra complexity to the DNN model. Our experimental results have demonstrated its superiority over the state-of-art algorithms on the CTR prediction dataset and dense classification dataset, in terms of both model accuracy and memory usage.

## 1. Introduction

Click-through rate (CTR) prediction is a large-scale problem that is essential to multi-billion dollar online advertising industry. In the advertising industry, advertisers pay publishers to display their ads on publishers’ sites. One popular payment model is the cost-per-click (CPC) model, where advertisers are charged only when a click occurs. As a consequence, a publisher’s revenue relies heavily on the ability to predict CTR accurately.

Identifying frequently predictive features and at the same time exploring unseen or rare cross features is the key to making good predictions. However, data for Web-scale recommender systems is mostly discrete and categorical, leading to a large and sparse feature space that is challenging for feature exploration. This has limited most large-scale systems to linear models such as logistic regression.

Linear models (Chapelle
et al., [2015](#bib.bib4)) are simple, interpretable and easy to scale; however, they are limited in their expressive power. Cross features, on the other hand, have been shown to be significant in improving the models’ expressiveness. Unfortunately, it often requires manual feature engineering or exhaustive search to identify such features; moreover, generalizing to unseen feature interactions is difficult.

In this paper, we aim to avoid task-specific feature engineering by introducing a novel neural network structure – a *cross network* – that explicitly applies feature crossing in an automatic fashion. The cross network consists of multiple layers, where the highest-degree of interactions are provably determined by layer depth. Each layer produces higher-order interactions based on existing ones, and keeps the interactions from previous layers.
We train the cross network jointly with a deep neural network (DNN) (LeCun
et al., [2015](#bib.bib11); Schmidhuber, [2015](#bib.bib15)). DNN has the promise to capture very complex interactions across features; however, compared to our cross network it requires nearly an order of magnitude more parameters, is unable to form cross features explicitly, and may fail to efficiently learn some types of feature interactions. Jointly training the cross and DNN components together, however, efficiently captures predictive feature interactions, and delivers state-of-the-art performance on the Criteo CTR dataset.

### 1.1. Related Work

Due to the dramatic increase in size and dimensionality of datasets, a number of methods have been proposed to avoid extensive task-specific feature engineering, mostly based on embedding techniques and neural networks.

Factorization machines (FMs) (Rendle, [2010](#bib.bib12), [2012](#bib.bib13)) project sparse features onto low-dimensional dense vectors and learn feature interactions from vector inner products. Field-aware factorization machines (FFMs) (Juan
et al., [2016](#bib.bib9), [2017](#bib.bib8)) further allow each feature to learn several vectors where each vector is associated with a field. Regrettably, the shallow structures of FMs and FFMs limit their representative power. There have been work extending FMs to higher orders (Blondel
et al., [2016](#bib.bib2); Yang and Gittens, [2015](#bib.bib19)), but one downside lies in their large number of parameters which yields undesirable computational cost. Deep neural networks (DNN) are able to learn non-trivial high-degree feature interactions due to embedding vectors and nonlinear activation functions. The recent success of the Residual Network (He
et al., [2015](#bib.bib6)) has enabled training of very deep networks. Deep Crossing (Shan
et al., [2016](#bib.bib16)) extends residual networks and achieves automatic feature learning by stacking all types of inputs.

The remarkable success of deep learning has elicited theoretical analyses on its representative power. There has been research (Valiant, [2014](#bib.bib17); Veit
et al., [2016](#bib.bib18)) showing that DNNs are able to approximate an arbitrary function under certain smoothness assumptions to an arbitrary accuracy, given sufficiently many hidden units or hidden layers. Moreover, in practice, it has been found that DNNs work well with a feasible number of parameters. One key reason is that most functions of practical interest are not arbitrary.

Yet one remaining question is whether DNNs are indeed the most efficient ones in representing such functions of practical interest. In the Kaggle111https://www.kaggle.com/ competition, the manually crafted features in many winning solutions are low-degree, in an explicit format and effective. The features learned by DNNs, on the other hand, are implicit and highly nonlinear. This has shed light on designing a model that is able to learn bounded-degree feature interactions more efficiently and explicitly than a universal DNN.

The wide-and-deep (Cheng et al., [2016](#bib.bib5)) is a model in this spirit. It takes cross features as inputs to a linear model, and jointly trains the linear model with a DNN model. However, the success of wide-and-deep hinges on a proper choice of cross features, an exponential problem for which there is yet no clear efficient method.

### 1.2. Main Contributions

In this paper, we propose the Deep & Cross Network (DCN) model that enables Web-scale automatic feature learning with both sparse and dense inputs. DCN efficiently captures effective feature interactions of bounded degrees, learns highly nonlinear interactions, requires no manual feature engineering or exhaustive searching, and has low computational cost.

The main contributions of the paper include:

* •

  We propose a novel cross network that explicitly applies feature crossing at each layer, efficiently learns predictive cross features of bounded degrees, and requires no manual feature engineering or exhaustive searching.
* •

  The cross network is simple yet effective. By design, the highest polynomial degree increases at each layer and is determined by layer depth. The network consists of all the cross terms of degree up to the highest, with their coefficients all different.
* •

  The cross network is memory efficient, and easy to implement.
* •

  Our experimental results have demonstrated that with a cross network, DCN has lower logloss than a DNN with nearly an order of magnitude fewer number of parameters.

The paper is organized as follows: [Section 2](#S2 "2. Deep & Cross Network (DCN) ‣ Deep & Cross Network for Ad Click Predictions") describes the architecture of the Deep & Cross Network. [Section 3](#S3 "3. Cross Network Analysis ‣ Deep & Cross Network for Ad Click Predictions") analyzes the cross network in detail. [Section 4](#S4 "4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions") shows the experimental results.

## 2. Deep & Cross Network (DCN)

In this section we describe the architecture of Deep & Cross Network (DCN) models.
A DCN model starts with an *embedding and stacking layer*, followed by a *cross network* and a *deep network* in parallel. These in turn are followed by a final *combination layer* which combines the outputs from the two networks. The complete DCN model is depicted in [Figure 1](#S2.F1 "Figure 1 ‣ 2. Deep & Cross Network (DCN) ‣ Deep & Cross Network for Ad Click Predictions").

!(/html/1708.05123/assets/x1.png)

Figure 1. The Deep & Cross Network

### 2.1. Embedding and Stacking Layer

We consider input data with sparse and dense features. In Web-scale recommender systems such as CTR prediction, the inputs are mostly categorical features, *e.g.* "country=usa". Such features are often encoded as one-hot vectors *e.g.* "[0,1,0]"; however, this often leads to excessively high-dimensional feature spaces for large vocabularies.

To reduce the dimensionality, we employ an embedding procedure to transform these binary features into dense vectors of real values (commonly called embedding vectors):

|  |  |  |  |
| --- | --- | --- | --- |
| (1) |  | 𝐱embed,i=Wembed,i​𝐱i,subscript𝐱  embed𝑖subscript𝑊  embed𝑖subscript𝐱𝑖{\bf x}\_{\text{embed},i}=W\_{\text{embed},i}{\bf x}\_{i}, |  |

where 𝐱embed,isubscript𝐱

embed𝑖{\bf x}\_{\text{embed},i} is the embedding vector, 𝐱isubscript𝐱𝑖{\bf x}\_{i} is the binary input in the i𝑖i-th category, and Wembed,i∈ℝne×nvsubscript𝑊

embed𝑖superscriptℝsubscript𝑛𝑒subscript𝑛𝑣W\_{\text{embed},i}\in\mathbb{R}^{n\_{e}\times n\_{v}} is the corresponding embedding matrix that will be optimized together with other parameters in the network, and ne,nv

subscript𝑛𝑒subscript𝑛𝑣n\_{e},n\_{v} are the embedding size and vocabulary size, respectively.

In the end, we stack the embedding vectors, along with the normalized dense features 𝐱densesubscript𝐱dense{\bf x}\_{\text{dense}}, into one vector:

|  |  |  |  |
| --- | --- | --- | --- |
| (2) |  | 𝐱0=[𝐱embed,1T,…,𝐱embed,kT,𝐱denseT],subscript𝐱0  superscriptsubscript𝐱  embed1𝑇…superscriptsubscript𝐱  embed𝑘𝑇superscriptsubscript𝐱dense𝑇{\bf x}\_{0}=\left[{\bf x}\_{\text{embed},1}^{T},\ldots,{\bf x}\_{\text{embed},k}^{T},{\bf x}\_{\text{dense}}^{T}\right], |  |

and feed 𝐱0subscript𝐱0{\bf x}\_{0} to the network.

### 2.2. Cross Network

The key idea of our novel cross network is to apply explicit feature crossing in an efficient way. The cross network is composed of cross layers, with each layer having the following formula:

|  |  |  |  |
| --- | --- | --- | --- |
| (3) |  | 𝐱l+1=𝐱0​𝐱lT​𝐰l+𝐛l+𝐱l=f​(𝐱l,𝐰l,𝐛l)+𝐱l,subscript𝐱𝑙1subscript𝐱0superscriptsubscript𝐱𝑙𝑇subscript𝐰𝑙subscript𝐛𝑙subscript𝐱𝑙𝑓subscript𝐱𝑙subscript𝐰𝑙subscript𝐛𝑙subscript𝐱𝑙{\bf x}\_{l+1}={\bf x}\_{0}{\bf x}\_{l}^{T}{\bf w}\_{l}+{\bf b}\_{l}+{\bf x}\_{l}=f({\bf x}\_{l},{\bf w}\_{l},{\bf b}\_{l})+{\bf x}\_{l}, |  |

where 𝐱l,𝐱l+1∈ℝd

subscript𝐱𝑙subscript𝐱𝑙1
superscriptℝ𝑑{\bf x}\_{l},{\bf x}\_{l+1}\in\mathbb{R}^{d} are column vectors denoting the outputs from the l𝑙l-th and (l+1)𝑙1(l+1)-th cross layers, respectively; 𝐰l,𝐛l∈ℝd

subscript𝐰𝑙subscript𝐛𝑙
superscriptℝ𝑑{\bf w}\_{l},{\bf b}\_{l}\in\mathbb{R}^{d} are the weight and bias parameters of the l𝑙l-th layer. Each cross layer adds back its input after a feature crossing f𝑓f, and the mapping function f:ℝd↦ℝd:𝑓maps-tosuperscriptℝ𝑑superscriptℝ𝑑f:\mathbb{R}^{d}\mapsto\mathbb{R}^{d} fits the residual of 𝐱l+1−𝐱lsubscript𝐱𝑙1subscript𝐱𝑙{\bf x}\_{l+1}-{\bf x}\_{l}. A visualization of one cross layer is shown in [Figure 2](#S2.F2 "Figure 2 ‣ 2.2. Cross Network ‣ 2. Deep & Cross Network (DCN) ‣ Deep & Cross Network for Ad Click Predictions").

!(/html/1708.05123/assets/x2.png)

Figure 2. Visualization of a cross layer.

High-degree Interaction Across Features.
The special structure of the cross network causes the degree of cross features to grow with layer depth. The highest polynomial degree (in terms of input 𝐱0subscript𝐱0{\bf x}\_{0}) for an l𝑙l-layer cross network is l+1𝑙1l+1. In fact, the cross network comprises all the cross terms x1α1​x2α2​…​xdαdsuperscriptsubscript𝑥1subscript𝛼1superscriptsubscript𝑥2subscript𝛼2…superscriptsubscript𝑥𝑑subscript𝛼𝑑x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\ldots x\_{d}^{\alpha\_{d}} of degree from 1 to l+1𝑙1l+1. Detailed analysis is in [Section 3](#S3 "3. Cross Network Analysis ‣ Deep & Cross Network for Ad Click Predictions").

Complexity Analysis.
Let Lcsubscript𝐿𝑐L\_{c} denote the number of cross layers, and d𝑑d denote the input dimension. Then, the number of parameters involved in the cross network is

|  |  |  |
| --- | --- | --- |
|  | d×Lc×2.𝑑subscript𝐿𝑐2\displaystyle d\times L\_{c}\times 2. |  |

The time and space complexity of a cross network are linear in input dimension.
Therefore, a cross network introduces negligible complexity compared to its deep counterpart, keeping the overall complexity for DCN at the same level as that of a traditional DNN.
This efficiency benefits from the rank-one property of 𝐱0​𝐱lTsubscript𝐱0superscriptsubscript𝐱𝑙𝑇{\bf x}\_{0}{\bf x}\_{l}^{T}, which enables us to generate all cross terms without computing or storing the entire matrix.

The small number of parameters of the cross network has limited the model capacity. To capture highly nonlinear interactions, we introduce a deep network in parallel.

### 2.3. Deep Network

The deep network is a fully-connected feed-forward neural network, with each deep layer having the following formula:

|  |  |  |  |
| --- | --- | --- | --- |
| (4) |  | 𝐡l+1=f​(Wl​𝐡l+𝐛l),subscript𝐡𝑙1𝑓subscript𝑊𝑙subscript𝐡𝑙subscript𝐛𝑙{\bf h}\_{l+1}=f(W\_{l}{\bf h}\_{l}+{\bf b}\_{l}), |  |

where 𝐡l∈ℝnl,𝐡l+1∈ℝnl+1formulae-sequencesubscript𝐡𝑙superscriptℝsubscript𝑛𝑙subscript𝐡𝑙1superscriptℝsubscript𝑛𝑙1{\bf h}\_{l}\in\mathbb{R}^{n\_{l}},{\bf h}\_{l+1}\in\mathbb{R}^{n\_{l+1}} are the l𝑙l-th and (l+1)𝑙1(l+1)-th hidden layer, respectively; Wl∈ℝnl+1×nl,𝐛l∈ℝnl+1formulae-sequencesubscript𝑊𝑙superscriptℝsubscript𝑛𝑙1subscript𝑛𝑙subscript𝐛𝑙superscriptℝsubscript𝑛𝑙1W\_{l}\in\mathbb{R}^{n\_{l+1}\times n\_{l}},{\bf b}\_{l}\in\mathbb{R}^{n\_{l+1}} are parameters for the l𝑙l-th deep layer; and f​(⋅)𝑓⋅f(\cdot) is the ReLU function.

Complexity Analysis.
For simplicity, we assume all the deep layers are of equal size. Let Ldsubscript𝐿𝑑L\_{d} denote the number of deep layers and m𝑚m denote the deep layer size. Then, the number of parameters in the deep network is

|  |  |  |
| --- | --- | --- |
|  | d×m+m+(m2+m)×(Ld−1).𝑑𝑚𝑚superscript𝑚2𝑚subscript𝐿𝑑1\displaystyle d\times m+m+(m^{2}+m)\times(L\_{d}-1). |  |

### 2.4. Combination Layer

The combination layer concatenates the outputs from two networks and feed the concatenated vector into a standard logits layer.

The following is the formula for a two-class classification problem:

|  |  |  |  |
| --- | --- | --- | --- |
| (5) |  | p=σ​([𝐱L1T,𝐡L2T]​𝐰logits),𝑝𝜎superscriptsubscript𝐱subscript𝐿1𝑇superscriptsubscript𝐡subscript𝐿2𝑇subscript𝐰logitsp=\sigma\left([{\bf x}\_{L\_{1}}^{T},{\bf h}\_{L\_{2}}^{T}]{\bf w}\_{\text{logits}}\right), |  |

where 𝐱L1∈ℝd,𝐡L2∈ℝmformulae-sequencesubscript𝐱subscript𝐿1superscriptℝ𝑑subscript𝐡subscript𝐿2superscriptℝ𝑚{\bf x}\_{L\_{1}}\in\mathbb{R}^{d},{\bf h}\_{L\_{2}}\in\mathbb{R}^{m} are the outputs from the cross network and deep network, respectively, 𝐰logits∈ℝ(d+m)subscript𝐰logitssuperscriptℝ𝑑𝑚{\bf w}\_{\text{logits}}\in\mathbb{R}^{(d+m)} is the weight vector for the combination layer, and σ​(x)=1/(1+exp⁡(−x))𝜎𝑥11𝑥\sigma(x)=1/(1+\exp(-x)).

The loss function is the log loss along with a regularization term,

|  |  |  |  |
| --- | --- | --- | --- |
| (6) |  | loss=−1N​∑i=1Nyi​log⁡(pi)+(1−yi)​log⁡(1−pi)+λ​∑l‖𝐰l‖2,loss1𝑁superscriptsubscript𝑖1𝑁subscript𝑦𝑖subscript𝑝𝑖1subscript𝑦𝑖1subscript𝑝𝑖𝜆subscript𝑙superscriptdelimited-∥∥subscript𝐰𝑙2\begin{split}\text{loss}=&-\frac{1}{N}\sum\_{i=1}^{N}y\_{i}\log(p\_{i})+(1-y\_{i})\log(1-p\_{i})+\lambda\sum\_{l}\|{\bf w}\_{l}\|^{2},\end{split} |  |

where pisubscript𝑝𝑖p\_{i}’s are the probabilities computed from [Equation 5](#S2.E5 "5 ‣ 2.4. Combination Layer ‣ 2. Deep & Cross Network (DCN) ‣ Deep & Cross Network for Ad Click Predictions"), yisubscript𝑦𝑖y\_{i}’s are the true labels, N𝑁N is the total number of inputs, and λ𝜆\lambda is the L2subscript𝐿2L\_{2} regularization parameter.

We jointly train both networks, as this allows each individual network to be aware of the others during the training.

## 3. Cross Network Analysis

In this section, we analyze the cross network of DCN for the purpose of understanding its effectiveness. We offer three perspectives: polynomial approximation, generalization to FMs, and efficient projection. For simplicity, we assume 𝐛i=0subscript𝐛𝑖0{\bf b}\_{i}=0.

*Notations.* Let the i𝑖i-th element in 𝐰jsubscript𝐰𝑗{\bf w}\_{j} be wj(i)superscriptsubscript𝑤𝑗𝑖w\_{j}^{(i)}. For multi-index 𝜶=[α1,⋯,αd]∈ℕd𝜶

subscript𝛼1⋯subscript𝛼𝑑superscriptℕ𝑑{\bm{\alpha}}=[\alpha\_{1},\cdots,\alpha\_{d}]\in\mathbb{N}^{d} and 𝐱=[x1,⋯,xd]∈ℝd𝐱

subscript𝑥1⋯subscript𝑥𝑑superscriptℝ𝑑{\bf x}=[x\_{1},\cdots,x\_{d}]\in\mathbb{R}^{d}, we define |𝜶|=∑i=1dαi𝜶superscriptsubscript𝑖1𝑑subscript𝛼𝑖|{\bm{\alpha}}|=\sum\_{i=1}^{d}\alpha\_{i}.

*Terminology.* The degree of a cross term (monomial) x1α1​x2α2​⋯​xdαdsuperscriptsubscript𝑥1subscript𝛼1superscriptsubscript𝑥2subscript𝛼2⋯superscriptsubscript𝑥𝑑subscript𝛼𝑑x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\cdots x\_{d}^{\alpha\_{d}} is defined by |𝜶|𝜶|{\bm{\alpha}}|. The degree of a polynomial is defined by the highest degree of its terms.

### 3.1. Polynomial Approximation

By the Weierstrass approximation theorem (Rudin
et al., [1964](#bib.bib14)), any function under certain smoothness assumption can be approximated by a polynomial to an arbitrary accuracy. Therefore, we analyze the cross network from the perspective of polynomial approximation.
In particular, the cross network approximates the polynomial class of the same degree in a way that is efficient, expressive and generalizes better to real-world datasets.

We study in detail the approximation of a cross network to the polynomial class of the same degree. Let us denote by Pn​(𝐱)subscript𝑃𝑛𝐱P\_{n}({\bf x}) the multivariate polynomial class of degree n𝑛n:

|  |  |  |  |
| --- | --- | --- | --- |
| (7) |  | Pn(𝐱)={∑𝜶w𝜶x1α1x2α2…xdαd|0≤|𝜶|≤n,𝜶∈ℕd}.P\_{n}({\bf x})=\biggl{\{}\sum\_{{\bm{\alpha}}}w\_{{\bm{\alpha}}}x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\ldots x\_{d}^{\alpha\_{d}}\mathrel{\bigg{|}}0\leq|{\bm{\alpha}}|\leq n,{\bm{\alpha}}\in\mathbb{N}^{d}\biggr{\}}. |  |

Each polynomial in this class has O​(dn)𝑂superscript𝑑𝑛O(d^{n}) coefficients. We show that, with only O​(d)𝑂𝑑O(d) parameters, the cross network contains all the cross terms occurring in the polynomial of the same degree, with each term’s coefficient distinct from each other.

###### Theorem 3.1.

Consider an l𝑙l-layer cross network with the i+1𝑖1i+1-th layer defined as 𝐱i+1=𝐱0​𝐱iT​𝐰i+𝐱isubscript𝐱𝑖1subscript𝐱0superscriptsubscript𝐱𝑖𝑇subscript𝐰𝑖subscript𝐱𝑖{\bf x}\_{i+1}={\bf x}\_{0}{\bf x}\_{i}^{T}{\bf w}\_{i}+{\bf x}\_{i}. Let the input to the network be 𝐱0=[x1,x2,…,xd]Tsubscript𝐱0superscript

subscript𝑥1subscript𝑥2…subscript𝑥𝑑
𝑇{\bf x}\_{0}=[x\_{1},x\_{2},\ldots,x\_{d}]^{T}, the output be gl​(𝐱0)=𝐱lT​𝐰lsubscript𝑔𝑙subscript𝐱0superscriptsubscript𝐱𝑙𝑇subscript𝐰𝑙g\_{l}({\bf x}\_{0})={\bf x}\_{l}^{T}{\bf w}\_{l}, and the parameters be 𝐰i,𝐛i∈ℝd

subscript𝐰𝑖subscript𝐛𝑖
superscriptℝ𝑑{\bf w}\_{i},{\bf b}\_{i}\in\mathbb{R}^{d}. Then, the multivariate polynomial gl​(𝐱0)subscript𝑔𝑙subscript𝐱0g\_{l}({\bf x}\_{0}) reproduces polynomials in the following class:

|  |  |  |
| --- | --- | --- |
|  | {∑𝜶c𝜶​(𝐰0,…,𝐰l)​x1α1​x2α2​…​xdαd|0≤|𝜶|≤l+1,𝜶∈ℕd},formulae-sequence|subscript𝜶subscript𝑐𝜶subscript𝐰0…subscript𝐰𝑙superscriptsubscript𝑥1subscript𝛼1superscriptsubscript𝑥2subscript𝛼2…superscriptsubscript𝑥𝑑subscript𝛼𝑑0𝜶𝑙1𝜶superscriptℕ𝑑\biggl{\{}\sum\_{{\bm{\alpha}}}c\_{{\bm{\alpha}}}({\bf w}\_{0},\ldots,{\bf w}\_{l})x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\ldots x\_{d}^{\alpha\_{d}}\mathrel{\bigg{|}}0\leq|{\bm{\alpha}}|\leq l+1,{\bm{\alpha}}\in\mathbb{N}^{d}\biggr{\}}, |  |

where c𝛂=M𝛂​∑𝐢∈B𝛂∑𝐣∈P𝛂∏k=1|𝛂|wik(jk)subscript𝑐𝛂subscript𝑀𝛂subscript𝐢subscript𝐵𝛂subscript𝐣subscript𝑃𝛂superscriptsubscriptproduct𝑘1𝛂superscriptsubscript𝑤subscript𝑖𝑘subscript𝑗𝑘c\_{{\bm{\alpha}}}=M\_{{\bm{\alpha}}}\sum\_{{\bf i}\in B\_{\bm{\alpha}}}\sum\_{{\bf j}\in P\_{\bm{\alpha}}}\prod\_{k=1}^{|{\bm{\alpha}}|}w\_{i\_{k}}^{(j\_{k})}, M𝛂subscript𝑀𝛂M\_{\bm{\alpha}} is a constant independent of 𝐰isubscript𝐰𝑖{\bf w}\_{i}’s, 𝐢=[i1,…,i|𝛂|]𝐢

subscript𝑖1…subscript𝑖𝛂{\bf i}=[i\_{1},\ldots,i\_{|{\bm{\alpha}}|}] and 𝐣=[j1,…,j|𝛂|]𝐣

subscript𝑗1…subscript𝑗𝛂{\bf j}=[j\_{1},\ldots,j\_{|{\bm{\alpha}}|}] are multi-indices, B𝛂={𝐲∈{0,1,⋯,l}|𝛂||yi<yj∧y|𝛂|=l}subscript𝐵𝛂𝐲superscript01⋯𝑙𝛂|subscript𝑦𝑖subscript𝑦𝑗subscript𝑦𝛂𝑙B\_{{\bm{\alpha}}}=\bigl{\{}{\bf y}\in\{0,1,\cdots,l\}^{|{\bm{\alpha}}|}\mathrel{\big{|}}y\_{i}<y\_{j}\wedge y\_{|{\bm{\alpha}}|}=l\bigr{\}}, and P𝛂subscript𝑃𝛂P\_{\bm{\alpha}} is the set of all the permutations of the indices (1,⋯,1⏟α1​times​⋯​d,⋯,d⏟αd​times)subscript⏟

1⋯1subscript𝛼1times⋯subscript⏟

𝑑⋯𝑑subscript𝛼𝑑times(\underbrace{1,\cdots,1}\_{\alpha\_{1}\,\text{times}}\cdots\underbrace{d,\cdots,d}\_{\alpha\_{d}\,\text{times}}).

The proof of [Theorem 3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1. Polynomial Approximation ‣ 3. Cross Network Analysis ‣ Deep & Cross Network for Ad Click Predictions") is in the Appendix.
Let us give an example. Consider the coefficient c𝜶subscript𝑐𝜶c\_{{\bm{\alpha}}} for x1​x2​x3subscript𝑥1subscript𝑥2subscript𝑥3x\_{1}x\_{2}x\_{3} with 𝜶=(1,1,1,0,…,0)𝜶1110…0{\bm{\alpha}}=(1,1,1,0,\ldots,0). Up to some constant, when l=2𝑙2l=2,
c𝜶=∑i,j,k∈P𝜶w0(i)​w1(j)​w2(k)subscript𝑐𝜶subscript

𝑖𝑗𝑘
subscript𝑃𝜶superscriptsubscript𝑤0𝑖superscriptsubscript𝑤1𝑗superscriptsubscript𝑤2𝑘c\_{\bm{\alpha}}=\sum\_{i,j,k\in P\_{\bm{\alpha}}}w\_{0}^{(i)}w\_{1}^{(j)}w\_{2}^{(k)}; when l=3𝑙3l=3, c𝜶=∑i,j,k∈P𝜶w0(i)​w1(j)​w3(k)+w0(i)​w2(j)​w3(k)+w1(i)​w2(j)​w3(k)subscript𝑐𝜶subscript

𝑖𝑗𝑘
subscript𝑃𝜶superscriptsubscript𝑤0𝑖superscriptsubscript𝑤1𝑗superscriptsubscript𝑤3𝑘superscriptsubscript𝑤0𝑖superscriptsubscript𝑤2𝑗superscriptsubscript𝑤3𝑘superscriptsubscript𝑤1𝑖superscriptsubscript𝑤2𝑗superscriptsubscript𝑤3𝑘c\_{\bm{\alpha}}=\sum\_{i,j,k\in P\_{\bm{\alpha}}}w\_{0}^{(i)}w\_{1}^{(j)}w\_{3}^{(k)}+w\_{0}^{(i)}w\_{2}^{(j)}w\_{3}^{(k)}+w\_{1}^{(i)}w\_{2}^{(j)}w\_{3}^{(k)}.

### 3.2. Generalization of FMs

The cross network shares the spirit of parameter sharing as the FM model and further extends it to a deeper structure.

In a FM model, feature xisubscript𝑥𝑖x\_{i} is associated with a weight vector 𝐯isubscript𝐯𝑖{\bf v}\_{i}, and the weight of cross term xi​xjsubscript𝑥𝑖subscript𝑥𝑗x\_{i}x\_{j} is computed by ⟨𝐯i,𝐯j⟩

subscript𝐯𝑖subscript𝐯𝑗\langle{\bf v}\_{i},{\bf v}\_{j}\rangle. In DCN, xisubscript𝑥𝑖x\_{i} is associated with scalars {wk(i)}k=1lsuperscriptsubscriptsuperscriptsubscript𝑤𝑘𝑖𝑘1𝑙\{w\_{k}^{(i)}\}\_{k=1}^{l}, and the weight of xi​xjsubscript𝑥𝑖subscript𝑥𝑗x\_{i}x\_{j} is the multiplications of parameters from the sets {wk(i)}k=0lsuperscriptsubscriptsuperscriptsubscript𝑤𝑘𝑖𝑘0𝑙\{w\_{k}^{(i)}\}\_{k=0}^{l} and {wk(j)}k=0lsuperscriptsubscriptsuperscriptsubscript𝑤𝑘𝑗𝑘0𝑙\{w\_{k}^{(j)}\}\_{k=0}^{l}. Both models have each feature learned some parameters independent from other features, and the weight of a cross term is a certain combination of corresponding parameters.

Parameter sharing not only makes the model more efficient, but also enables the model to generalize to unseen feature interactions and be more robust to noise. For example, take datasets with sparse features. If two binary features xisubscript𝑥𝑖x\_{i} and xjsubscript𝑥𝑗x\_{j} rarely or never co-occur in the training data, *i.e.*, xi≠0∧xj≠0subscript𝑥𝑖0subscript𝑥𝑗0x\_{i}\neq 0\wedge x\_{j}\neq 0, then the learned weight of xi​xjsubscript𝑥𝑖subscript𝑥𝑗x\_{i}x\_{j} would carry no meaningful information for prediction.

The FM is a shallow structure and is limited to representing cross terms of degree 2. DCN, in contrast, is able to construct all the cross terms x1α1​x2α2​…​xdαdsuperscriptsubscript𝑥1subscript𝛼1superscriptsubscript𝑥2subscript𝛼2…superscriptsubscript𝑥𝑑subscript𝛼𝑑x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\ldots x\_{d}^{\alpha\_{d}} with degree |𝜶|𝜶|{\bm{\alpha}}| bounded by some constant determined by layer depth, as claimed in [Theorem 3.1](#S3.Thmtheorem1 "Theorem 3.1. ‣ 3.1. Polynomial Approximation ‣ 3. Cross Network Analysis ‣ Deep & Cross Network for Ad Click Predictions"). Therefore, the cross network extends the idea of parameter sharing from a single layer to multiple layers and high-degree cross-terms. Note that different from the higher-order FMs, the number of parameters in a cross network only grows linearly with the input dimension.

### 3.3. Efficient Projection

Each cross layer projects all the pairwise interactions between 𝐱0subscript𝐱0{\bf x}\_{0} and 𝐱lsubscript𝐱𝑙{\bf x}\_{l}, in an efficient manner, back to the input’s dimension.

Consider 𝐱~∈ℝd~𝐱superscriptℝ𝑑\tilde{\bf x}\in\mathbb{R}^{d} as the input to a cross layer. The cross layer first implicitly constructs d2superscript𝑑2d^{2} pairwise interactions xi​x~jsubscript𝑥𝑖subscript~𝑥𝑗x\_{i}\tilde{x}\_{j}, and then implicitly projects them back to dimension d𝑑d in a memory-efficient way. A direct approach, however, comes with a cubic cost.

Our cross layer provides an efficient solution to reduce the cost to linear in dimension d𝑑d. Consider 𝐱p=𝐱0​𝐱~T​𝐰subscript𝐱𝑝subscript𝐱0superscript~𝐱𝑇𝐰{\bf x}\_{p}={\bf x}\_{0}\tilde{\bf x}^{T}{\bf w}. This is in fact equivalent to

|  |  |  |  |
| --- | --- | --- | --- |
| (8) |  | 𝐱pT=[x1​x~1​…​x1​x~d…xd​x~1​…​xd​x~d]​[∣𝐰∣𝟎…𝟎𝟎∣𝐰∣…𝟎⋮⋮⋱⋮𝟎𝟎…∣𝐰∣]superscriptsubscript𝐱𝑝𝑇matrixsubscript𝑥1subscript~𝑥1…subscript𝑥1subscript~𝑥𝑑…subscript𝑥𝑑subscript~𝑥1…subscript𝑥𝑑subscript~𝑥𝑑delimited-[]∣𝐰∣0…00∣𝐰∣…0⋮⋮⋱⋮00…∣𝐰∣\begin{split}{\bf x}\_{p}^{T}=\begin{bmatrix}x\_{1}\tilde{x}\_{1}\ldots x\_{1}\tilde{x}\_{d}&\ldots&x\_{d}\tilde{x}\_{1}\ldots x\_{d}\tilde{x}\_{d}\end{bmatrix}\left[\begin{smallmatrix}\begin{smallmatrix}\mid\\ {\bf w}\\ \mid\end{smallmatrix}&{\bf 0}&\ldots&{\bf 0}\\ {\bf 0}&\begin{smallmatrix}\mid\\ {\bf w}\\ \mid\end{smallmatrix}&\ldots&{\bf 0}\\ \vdots&\vdots&\ddots&\vdots\\ {\bf 0}&{\bf 0}&\ldots&\begin{smallmatrix}\mid\\ {\bf w}\\ \mid\end{smallmatrix}\end{smallmatrix}\right]\end{split} |  |

where the row vector contains all d2superscript𝑑2d^{2} pairwise interactions xi​x~jsubscript𝑥𝑖subscript~𝑥𝑗x\_{i}\tilde{x}\_{j}’s, the projection matrix has a block diagonal structure with 𝐰∈ℝd𝐰superscriptℝ𝑑{\bf w}\in\mathbb{R}^{d} being a column vector.

## 4. Experimental Results

In this section, we evaluate the performance of DCN on some popular classification datasets.

### 4.1. Criteo Display Ads Data

The Criteo Display Ads222https://www.kaggle.com/c/criteo-display-ad-challenge dataset is for the purpose of predicting ads click-through rate. It has 13 integer features and 26 categorical features where each category has a high cardinality. For this dataset, an improvement of 0.001 in logloss is considered as practically significant. When considering a large user base, a small improvement in prediction accuracy can potentially lead to a large increase in a company’s revenue.
The data contains 11 GB user logs from a period of 7 days (∼similar-to\sim41 million records). We used the data of the first 6 days for training, and randomly split day 7 data into validation and test sets of equal size.

### 4.2. Implementation Details

DCN is implemented on TensorFlow, we briefly discuss some implementation details for training with DCN.

* *Data processing and embedding.* Real-valued features are normalized by applying a log transform. For categorical features, we embed the features in dense vectors of dimension 6×(category cardinality)1/4.6superscriptcategory cardinality146\times(\text{category cardinality})^{1/4}. Concatenating all embeddings results in a vector of dimension 1026.
* *Optimization.* We applied mini-batch stochastic optimization with Adam optimizer (Kingma and Ba, [2014](#bib.bib10)). The batch size is set at 512. Batch normalization (Ioffe and Szegedy, [2015](#bib.bib7)) was applied to the deep network and gradient clip norm was set at 100.
* *Regularization.* We used early stopping, as we did not find L2subscript𝐿2L\_{2} regularization or dropout to be effective.
* *Hyperparameters.*
  We report results based on a grid search over the number of hidden layers, hidden layer size, initial learning rate and number of cross layers. The number of hidden layers ranged from 2 to 5, with hidden layer sizes from 32 to 1024. For DCN, the number of cross layers333More cross layers did not lead to significant improvement, so we restrict ourselves in a small range for finer tuning. is from 1 to 6. The initial learning rate444Experimentally we observe that for the Criteo dataset, a learning rate larger than 0.001 usually degrades the performance. was tuned from 0.0001 to 0.001 with increments of 0.0001. All experiments applied early stopping at training step 150,000, beyond which overfitting started to occur.

### 4.3. Models for Comparisons

We compare DCN with five models: the DCN model with no cross network (DNN), logistic regression (LR), Factorization Machines (FMs), Wide and Deep Model (W&D), and Deep Crossing (DC).

* *DNN*. The embedding layer, the output layer, and the hyperparameter tuning process are the same as DCN. The only change from the DCN model was that there are no cross layers.
* *LR*. We used Sibyl (Canini, [2012](#bib.bib3))—a large-scale machine-learning system for distributed logistic regression. The integer features were discretized on a log scale. The cross features were selected by a sophisticated feature selection tool. All of the single features were used.
* *FM*. We used an FM-based model with proprietary details.
* *W&D*. Different than DCN, its wide component takes as input raw sparse features, and relies on exhaustive searching and domain knowledge to select predictive cross features. We skipped the comparison as no good method is known to select cross features.
* *DC*. Compared to DCN, DC does not form explicit cross features. It mainly relies on stacking and residual units to create implicit crossings. We applied the same embedding (stacking) layer as DCN, followed by another ReLu layer to generate input to a sequence of residual units. The number of residual units was tuned form 1 to 5, with input dimension and cross dimension from 100 to 1026.

### 4.4. Model Performance

In this section, we first list the best performance of different models in logloss, then we compare DCN with DNN in detail, that is, we investigate further into the effects introduced by the cross network.

Performance of different models.
The best test logloss of different models are listed in [Table 1](#S4.T1 "Table 1 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions").
The optimal hyperparameter settings were 2 deep layers of size 1024 and 6 cross layers for the DCN model, 5 deep layers of size 1024 for the DNN, 5 residual units with input dimension 424 and cross dimension 537 for the DC, and 42 cross features for the LR model. That the best performance was found with the deepest cross architecture suggests that the higher-order feature interactions from the cross network are valuable. As we can see, DCN outperforms all the other models by a large amount. In particular, it outperforms the state-of-art DNN model but uses only 40% of the memory consumed in DNN.

Table 1. Best test logloss from different models. “DC” is deep crossing, “DNN” is DCN with no cross layer, “FM” is Factorization Machine based model, “LR” is logistic regression.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | DCN | DC | DNN | FM | LR |
| Logloss | 0.4419 | 0.4425 | 0.4428 | 0.4464 | 0.4474 |

For the optimal hyperparameter setting of each model, we also report the mean and standard deviation of the test logloss out of 10 independent runs:
DCN: 0.4422±𝟗×𝟏𝟎−𝟓plus-or-minus0.44229superscript105{\bf 0.4422\pm 9\times 10^{-5}}, DNN: 0.4430±3.7×10−4plus-or-minus0.44303.7superscript1040.4430\pm 3.7\times 10^{-4}, DC: 0.4430±4.3×10−4plus-or-minus0.44304.3superscript1040.4430\pm 4.3\times 10^{-4}. As can be seen, DCN consistently outperforms other models by a large amount.

Comparisons Between DCN and DNN.
Considering that the cross network only introduces O​(d)𝑂𝑑O(d) extra parameters, we compare DCN to its deep network—a traditional DNN, and present the experimental results while varying memory budget and loss tolerance.

In the following, the loss for a certain number of parameters is reported as the best validation loss among all the learning rates and model structures. The number of parameters in the embedding layer was omitted in our calculation as it is identical to both models.

[Table 2](#S4.T2 "Table 2 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions") reports the minimal number of parameters needed to achieve a desired logloss threshold. From [Table 2](#S4.T2 "Table 2 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions"), we see that DCN is nearly an order of magnitude more memory efficient than a single DNN, thanks to the cross network which is able to learn bounded-degree feature interactions more efficiently.

Table 2. #parameters needed to achieve a desired logloss.

| Logloss | 0.4430 | 0.4460 | 0.4470 | 0.4480 |
| --- | --- | --- | --- | --- |
| DNN | 3.2×1063.2superscript1063.2\times 10^{6} | 1.5×1051.5superscript1051.5\times 10^{5} | 1.5×1051.5superscript1051.5\times 10^{5} | 7.8×1047.8superscript1047.8\times 10^{4} |
| DCN | 7.9×𝟏𝟎𝟓7.9superscript105{\bf 7.9\times 10^{5}} | 7.3×𝟏𝟎𝟒7.3superscript104{\bf 7.3\times 10^{4}} | 3.7×𝟏𝟎𝟒3.7superscript104{\bf 3.7\times 10^{4}} | 3.7×𝟏𝟎𝟒3.7superscript104{\bf 3.7\times 10^{4}} |

[Table 3](#S4.T3 "Table 3 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions") compares performance of the neural models subject to fixed memory budgets. As we can see, DCN consistently outperforms DNN. In the small-parameter regime, the number of parameters in the cross network is comparable to that in the deep network, and the clear improvement indicates that the cross network is more efficient in learning effective feature interactions. In the large-parameter regime, the DNN closes some of the gap; however, DCN still outperforms DNN by a large amount, suggesting that it can efficiently learn some types of meaningful feature interactions that even a huge DNN model cannot.

Table 3. Best logloss achieved with various memory budgets.

| #Params | 5×1045superscript1045\times 10^{4} | 1×1051superscript1051\times 10^{5} | 4×1054superscript1054\times 10^{5} | 1.1×1061.1superscript1061.1\times 10^{6} | 2.5×1062.5superscript1062.5\times 10^{6} |
| --- | --- | --- | --- | --- | --- |
| DNN | 0.4480 | 0.4471 | 0.4439 | 0.4433 | 0.4431 |
| DCN | 0.4465 | 0.4453 | 0.4432 | 0.4426 | 0.4423 |

We analyze DCN in finer detail by illustrating the effect from introducing a cross network to a given DNN model. We first compare the best performance of DNN with that of DCN under the same number of layers and layer size, and then for each setting, we show how the validation logloss changes as more cross layers are added. [Table 4](#S4.T4 "Table 4 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions") shows the differences between the DCN and DNN model in logloss. Under the same experimental setting, the best logloss from the DCN model consistently outperforms that from a single DNN model of the same structure. That the improvement is consistent for all the hyperparameters has mitigated the randomness effect from the initialization and stochastic optimization.

Table 4. Differences in the validation logloss (×10−2absentsuperscript102\times 10^{-2}) between DCN and DNN. The DNN model is the DCN model with the number of cross layers set to 0. Negative values mean that the DCN outperforms DNN.

| #Layers  #Nodes | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | -0.28 | -0.10 | -0.16 | -0.06 | -0.05 | -0.08 |
| 3 | -0.19 | -0.10 | -0.13 | -0.18 | -0.07 | -0.05 |
| 4 | -0.12 | -0.10 | -0.06 | -0.09 | -0.09 | -0.21 |
| 5 | -0.21 | -0.11 | -0.13 | -0.00 | -0.06 | -0.02 |

[Figure 3](#S4.F3 "Figure 3 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions") shows the improvement as we increase the number of cross layers on randomly selected settings. For the deep networks in [Figure 3](#S4.F3 "Figure 3 ‣ 4.4. Model Performance ‣ 4. Experimental Results ‣ Deep & Cross Network for Ad Click Predictions"), there is a clear improvement when 1 cross layer is added to the model. As more cross layers are introduced, for some settings the logloss continues to decrease, indicating the introduced cross terms are effective in the prediction; whereas for others the logloss starts to fluctuate and even slightly increase, which indicates the higher-degree feature interactions introduced are not helpful.

!(/html/1708.05123/assets/x3.png)

Figure 3. Improvement in the validation logloss with the growth of cross layer depth. The case with 0 cross layers is equivalent to a single DNN model. In the legend, “layers” is hidden layers, “nodes” is hidden nodes. Different symbols represent different hyperparameters for the deep network.

### 4.5. Non-CTR datasets

We show that DCN performs well on non-CTR prediction problems. We used the forest covertype (581012 samples and 54 features) and Higgs (11M samples and 28 features) datasets from the UCI repository. The datasets were randomly split into training (90%) and testing (10%) set. A grid search over the hyperparameters was performed. The number of deep layers ranged from 1 to 10 with layer size from 50 to 300. The number of cross layers ranged from 4 to 10. The number of residual units ranged from 1 to 5 with their input dimension and cross dimension from 50 to 300. For DCN, the input vector was fed to the cross network directly.

For the forest covertype data, DCN achieved the best test accuracy 0.9740 with the least memory consumption. Both DNN and DC achieved 0.9737. The optimal hyperparameter settings were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN, 7 deep layers of size 292 for DNN, and 4 residual units with input dimension 271 and cross dimension 287 for DC.

For the Higgs data, DCN achieved the best test logloss 0.4494, whereas DNN achieved 0.4506.
The optimal hyperparameter settings were 4 cross layers of size 28 and 4 deep layers of size 209 for DCN, and 10 deep layers of size 196 for DNN. DCN outperforms DNN with half of the memory used in DNN.

## 5. Conclusion and Future Directions

Identifying effective feature interactions has been the key to the success of many prediction models. Regrettably, the process often requires manual feature crafting and exhaustive searching. DNNs are popular for automatic feature learning; however, the features learned are implicit and highly nonlinear, and the network could be unnecessarily large and inefficient in learning certain features. The Deep & Cross Network proposed in this paper can handle a large set of sparse and dense features, and learns explicit cross features of bounded degree jointly with traditional deep representations. The degree of cross features increases by one at each cross layer. Our experimental results have demonstrated its superiority over the state-of-art algorithms on both sparse and dense datasets, in terms of both model accuracy and memory usage.

We would like to further explore using cross layers as building blocks in other models, enable effective training for deeper cross networks, investigate the efficiency of the cross network in polynomial approximation, and better understand its interaction with deep networks during optimization.

## References

* (1)
* Blondel
  et al. (2016)

  Mathieu Blondel, Akinori
  Fujino, Naonori Ueda, and Masakazu
  Ishihata. 2016.
  Higher-Order Factorization Machines. In
  Advances in Neural Information Processing Systems.
  3351–3359.
* Canini (2012)

  K. Canini.
  2012.
  Sibyl: A system for large scale supervised machine
  learning.
  Technical Talk (2012).
* Chapelle
  et al. (2015)

  Olivier Chapelle, Eren
  Manavoglu, and Romer Rosales.
  2015.
  Simple and scalable response prediction for display
  advertising.
  ACM Transactions on Intelligent Systems and
  Technology (TIST) 5, 4
  (2015), 61.
* Cheng et al. (2016)

  Heng-Tze Cheng, Levent
  Koc, Jeremiah Harmsen, Tal Shaked,
  Tushar Chandra, Hrishi Aradhye,
  Glen Anderson, Greg Corrado,
  Wei Chai, Mustafa Ispir, and
  others. 2016.
  Wide & Deep Learning for Recommender Systems.
  arXiv preprint arXiv:1606.07792
  (2016).
* He
  et al. (2015)

  Kaiming He, Xiangyu
  Zhang, Shaoqing Ren, and Jian Sun.
  2015.
  Deep residual learning for image recognition.
  arXiv preprint arXiv:1512.03385
  (2015).
* Ioffe and Szegedy (2015)

  Sergey Ioffe and
  Christian Szegedy. 2015.
  Batch normalization: Accelerating deep network
  training by reducing internal covariate shift.
  arXiv preprint arXiv:1502.03167
  (2015).
* Juan
  et al. (2017)

  Yuchin Juan, Damien
  Lefortier, and Olivier Chapelle.
  2017.
  Field-aware factorization machines in a real-world
  online advertising system. In Proceedings of the
  26th International Conference on World Wide Web Companion. International
  World Wide Web Conferences Steering Committee, 680–688.
* Juan
  et al. (2016)

  Yuchin Juan, Yong Zhuang,
  Wei-Sheng Chin, and Chih-Jen Lin.
  2016.
  Field-aware factorization machines for CTR
  prediction. In Proceedings of the 10th ACM
  Conference on Recommender Systems. ACM, 43–50.
* Kingma and Ba (2014)

  Diederik Kingma and
  Jimmy Ba. 2014.
  Adam: A method for stochastic optimization.
  arXiv preprint arXiv:1412.6980
  (2014).
* LeCun
  et al. (2015)

  Yann LeCun, Yoshua
  Bengio, and Geoffrey Hinton.
  2015.
  Deep learning.
  Nature 521,
  7553 (2015), 436–444.
* Rendle (2010)

  Steffen Rendle.
  2010.
  Factorization machines. In
  2010 IEEE International Conference on Data Mining.
  IEEE, 995–1000.
* Rendle (2012)

  Steffen Rendle.
  2012.
  Factorization Machines with libFM.
  ACM Trans. Intell. Syst. Technol.
  3, 3, Article 57
  (May 2012), 22 pages.
* Rudin
  et al. (1964)

  Walter Rudin and
  others. 1964.
  Principles of mathematical analysis.
  Vol. 3.
  McGraw-Hill New York.
* Schmidhuber (2015)

  Jürgen Schmidhuber.
  2015.
  Deep learning in neural networks: An overview.
  Neural networks 61
  (2015), 85–117.
* Shan
  et al. (2016)

  Ying Shan, T Ryan Hoens,
  Jian Jiao, Haijing Wang,
  Dong Yu, and JC Mao.
  2016.
  Deep Crossing: Web-Scale Modeling without Manually
  Crafted Combinatorial Features. In Proceedings of
  the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
  Mining. ACM, 255–262.
* Valiant (2014)

  Gregory Valiant.
  2014.
  Learning polynomials with neural networks.
  (2014).
* Veit
  et al. (2016)

  Andreas Veit, Michael J
  Wilber, and Serge Belongie.
  2016.
  Residual Networks Behave Like Ensembles of
  Relatively Shallow Networks.
  In Advances in Neural Information Processing
  Systems 29, D. D. Lee,
  M. Sugiyama, U. V. Luxburg,
  I. Guyon, and R. Garnett (Eds.).
  Curran Associates, Inc., 550–558.
* Yang and Gittens (2015)

  Jiyan Yang and Alex
  Gittens. 2015.
  Tensor machines for learning target-specific
  polynomial features.
  arXiv preprint arXiv:1504.01697
  (2015).

Appendix: Proof of Theorem 3.1

###### Proof.

*Notations.* Let 𝐢𝐢{\bf i} be a multi-index vector of 0’s and 1’s with its last entry fixed at 1. For multi-index 𝜶=[α1,⋯,αd]∈ℕd𝜶

subscript𝛼1⋯subscript𝛼𝑑superscriptℕ𝑑{\bm{\alpha}}=[\alpha\_{1},\cdots,\alpha\_{d}]\in\mathbb{N}^{d} and 𝐱=[x1,⋯,xd]T𝐱superscript

subscript𝑥1⋯subscript𝑥𝑑
𝑇{\bf x}=[x\_{1},\cdots,x\_{d}]^{T}, we define |𝜶|=∑i=1dαi𝜶superscriptsubscript𝑖1𝑑subscript𝛼𝑖|{\bm{\alpha}}|=\sum\_{i=1}^{d}\alpha\_{i}, and 𝐱𝜶=x1α1​x2α2​⋯​xdαdsuperscript𝐱𝜶superscriptsubscript𝑥1subscript𝛼1superscriptsubscript𝑥2subscript𝛼2⋯superscriptsubscript𝑥𝑑subscript𝛼𝑑{\bf x}^{{\bm{\alpha}}}=x\_{1}^{\alpha\_{1}}x\_{2}^{\alpha\_{2}}\cdots x\_{d}^{\alpha\_{d}}.

We first proof by induction that

|  |  |  |  |
| --- | --- | --- | --- |
| (9) |  | gl​(𝐱0)=𝐱lT​𝐰l=∑p=1l+1∑|𝐢|=p∏j=0l(𝐱0T​𝐰j)ij,subscript𝑔𝑙subscript𝐱0superscriptsubscript𝐱𝑙𝑇subscript𝐰𝑙superscriptsubscript𝑝1𝑙1subscript  𝐢𝑝superscriptsubscriptproduct𝑗0𝑙superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗g\_{l}({\bf x}\_{0})={\bf x}\_{l}^{T}{\bf w}\_{l}=\sum\_{p=1}^{l+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}\prod\_{j=0}^{l}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}, |  |

and then we rewrite the above form to obtain the desired claim.

* Base case. When l=0𝑙0l=0, g0​(𝐱0)=𝐱0T​𝐰0subscript𝑔0subscript𝐱0superscriptsubscript𝐱0𝑇subscript𝐰0g\_{0}({\bf x}\_{0})={\bf x}\_{0}^{T}{\bf w}\_{0}. Clearly [Equation 9](#S5.E9 "9 ‣ Proof. ‣ Deep & Cross Network for Ad Click Predictions") holds.
* Induction step. We assume that when l=k𝑙𝑘l=k,

  |  |  |  |
  | --- | --- | --- |
  |  | gk​(𝐱0)=𝐱kT​𝐰k=∑p=1k+1∑|𝐢|=p∏j=0k(𝐱0T​𝐰j)ij.subscript𝑔𝑘subscript𝐱0superscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘superscriptsubscript𝑝1𝑘1subscript 𝐢𝑝superscriptsubscriptproduct𝑗0𝑘superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗g\_{k}({\bf x}\_{0})={\bf x}\_{k}^{T}{\bf w}\_{k}=\sum\_{p=1}^{k+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}\prod\_{j=0}^{k}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}. |  |

  When l=k+1𝑙𝑘1l=k+1,

  |  |  |  |  |
  | --- | --- | --- | --- |
  | (10) |  | 𝐱k+1T​𝐰k+1=(𝐱kT​𝐰k)​(𝐱0T​𝐰k+1)+𝐱kT​𝐰k+1superscriptsubscript𝐱𝑘1𝑇subscript𝐰𝑘1superscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘superscriptsubscript𝐱0𝑇subscript𝐰𝑘1superscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘1\begin{split}{\bf x}\_{k+1}^{T}{\bf w}\_{k+1}=({\bf x}\_{k}^{T}{\bf w}\_{k})({\bf x}\_{0}^{T}{\bf w}\_{k+1})+{\bf x}\_{k}^{T}{\bf w}\_{k+1}\end{split} |  |

  Because 𝐱ksubscript𝐱𝑘{\bf x}\_{k} only contains 𝐰0,…,𝐰k−1
  subscript𝐰0…subscript𝐰𝑘1{\bf w}\_{0},\ldots,{\bf w}\_{k-1}, it follows that the formula of 𝐱kT​𝐰k+1superscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘1{\bf x}\_{k}^{T}{\bf w}\_{k+1} can be obtained from that of 𝐱kT​𝐰ksuperscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘{\bf x}\_{k}^{T}{\bf w}\_{k} by replacing all the 𝐰ksubscript𝐰𝑘{\bf w}\_{k}’s occurred in 𝐱kT​𝐰ksuperscriptsubscript𝐱𝑘𝑇subscript𝐰𝑘{\bf x}\_{k}^{T}{\bf w}\_{k} to 𝐰k+1subscript𝐰𝑘1{\bf w}\_{k+1}. Then

  |  |  |  |  |
  | --- | --- | --- | --- |
  | (11) |  | 𝐱k+1T​𝐰k+1=∑p=1k+1∑|𝐢|=p(𝐱0T​𝐰k+1)​∏j=0k(𝐱0T​𝐰j)ij+∑p=1k+1∑|𝐢|=p(𝐱0T​𝐰k+1)ik​∏j=0k−1(𝐱0T​𝐰j)ij=∑p=2k+2∑|𝐢|=pik=1∏j=0k+1(𝐱0T​𝐰j)ij+∑p=1k+1∑|𝐢|=pik=0∏j=0k+1(𝐱0T​𝐰j)ij=∑p=2k+1∑|𝐢|=p∏j=0k+1(𝐱0T​𝐰j)ij+(𝐱0T​𝐰k+1)+∏j=0k+1(𝐱0T​𝐰j)=∑p=1k+2∑|𝐢|=p∏j=0k+1(𝐱0T​𝐰j)ij.superscriptsubscript𝐱𝑘1𝑇subscript𝐰𝑘1superscriptsubscript𝑝1𝑘1subscript 𝐢𝑝superscriptsubscript𝐱0𝑇subscript𝐰𝑘1superscriptsubscriptproduct𝑗0𝑘superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗superscriptsubscript𝑝1𝑘1subscript 𝐢𝑝superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑘1subscript𝑖𝑘superscriptsubscriptproduct𝑗0𝑘1superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗superscriptsubscript𝑝2𝑘2subscript 𝐢𝑝subscript𝑖𝑘1superscriptsubscriptproduct𝑗0𝑘1superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗superscriptsubscript𝑝1𝑘1subscript 𝐢𝑝subscript𝑖𝑘0superscriptsubscriptproduct𝑗0𝑘1superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗superscriptsubscript𝑝2𝑘1subscript 𝐢𝑝superscriptsubscriptproduct𝑗0𝑘1superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗superscriptsubscript𝐱0𝑇subscript𝐰𝑘1superscriptsubscriptproduct𝑗0𝑘1superscriptsubscript𝐱0𝑇subscript𝐰𝑗superscriptsubscript𝑝1𝑘2subscript 𝐢𝑝superscriptsubscriptproduct𝑗0𝑘1superscriptsuperscriptsubscript𝐱0𝑇subscript𝐰𝑗subscript𝑖𝑗\begin{split}&{\bf x}\_{k+1}^{T}{\bf w}\_{k+1}=\\ &\sum\_{p=1}^{k+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}({\bf x}\_{0}^{T}{\bf w}\_{k+1})\prod\_{j=0}^{k}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}+\sum\_{p=1}^{k+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}({\bf x}\_{0}^{T}{\bf w}\_{k+1})^{i\_{k}}\prod\_{j=0}^{k-1}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}\\ =&\sum\_{p=2}^{k+2}\sum\_{\begin{subarray}{c}|{\bf i}|=p\\ i\_{k}=1\end{subarray}}\prod\_{j=0}^{k+1}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}+\sum\_{p=1}^{k+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\\ i\_{k}=0\end{subarray}}\prod\_{j=0}^{k+1}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}\\ =&\sum\_{p=2}^{k+1}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}\prod\_{j=0}^{k+1}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}+({\bf x}\_{0}^{T}{\bf w}\_{k+1})+\prod\_{j=0}^{k+1}({\bf x}\_{0}^{T}{\bf w}\_{j})\\ =&\sum\_{p=1}^{k+2}\sum\_{\begin{subarray}{c}|{\bf i}|=p\end{subarray}}\prod\_{j=0}^{k+1}({\bf x}\_{0}^{T}{\bf w}\_{j})^{i\_{j}}.\end{split} |  |

  The first equality is a result of increasing the size of 𝐢𝐢{\bf i} from k+1𝑘1k+1 to k+2𝑘2k+2.
  The second equality used the fact that the last entry of 𝐢𝐢{\bf i} is always 1 by definition, and the same was applied to the last equality.
  By induction hypothesis, [Equation 9](#S5.E9 "9 ‣ Proof. ‣ Deep & Cross Network for Ad Click Predictions") holds for all l∈ℤ𝑙ℤl\in\mathbb{Z}.

Next, we compute c𝜶​(𝐰0,⋯,𝐰l)subscript𝑐𝜶subscript𝐰0⋯subscript𝐰𝑙c\_{\bm{\alpha}}({\bf w}\_{0},\cdots,{\bf w}\_{l}), the coefficient of 𝐱𝜶superscript𝐱𝜶{\bf x}^{\bm{\alpha}}, by rearranging the terms in [Equation 9](#S5.E9 "9 ‣ Proof. ‣ Deep & Cross Network for Ad Click Predictions"). Note that all the different permutations of x1​⋯​x1⏟α1​⋯​xd​⋯​xd⏟αdsubscript⏟subscript𝑥1⋯subscript𝑥1subscript𝛼1⋯subscript⏟subscript𝑥𝑑⋯subscript𝑥𝑑subscript𝛼𝑑\underbrace{x\_{1}\cdots x\_{1}}\_{\alpha\_{1}}\cdots\underbrace{x\_{d}\cdots x\_{d}}\_{\alpha\_{d}} are in the form of 𝐱𝜶superscript𝐱𝜶{\bf x}^{\bm{\alpha}}. Therefore, c𝜶subscript𝑐𝜶c\_{\bm{\alpha}} is the summation of all the weights associated with each permutation occurred in [Equation 9](#S5.E9 "9 ‣ Proof. ‣ Deep & Cross Network for Ad Click Predictions"). The weight for permutation xj1​xj2​⋯​xjpsubscript𝑥subscript𝑗1subscript𝑥subscript𝑗2⋯subscript𝑥subscript𝑗𝑝x\_{j\_{1}}x\_{j\_{2}}\cdots x\_{j\_{p}} is

|  |  |  |
| --- | --- | --- |
|  | ∑i1,⋯,ipwi1(j1)​wi2(j2)​⋯​wip(jp),subscript  subscript𝑖1⋯subscript𝑖𝑝superscriptsubscript𝑤subscript𝑖1subscript𝑗1superscriptsubscript𝑤subscript𝑖2subscript𝑗2⋯superscriptsubscript𝑤subscript𝑖𝑝subscript𝑗𝑝\sum\_{i\_{1},\cdots,i\_{p}}w\_{i\_{1}}^{(j\_{1})}w\_{i\_{2}}^{(j\_{2})}\cdots w\_{i\_{p}}^{(j\_{p})}, |  |

where (i1,⋯,ip)subscript𝑖1⋯subscript𝑖𝑝(i\_{1},\cdots,i\_{p}) belongs to the set of all the corresponding active indices for |𝐢|=p𝐢𝑝|{\bf i}|=p, specifically,

|  |  |  |
| --- | --- | --- |
|  | (i1,⋯,ip)∈Bp=:{𝐲∈{0,1,⋯,l}p|yi<yj∧yp=l}.(i\_{1},\cdots,i\_{p})\in B\_{p}=:\bigl{\{}{\bf y}\in\{0,1,\cdots,l\}^{p}\mathrel{\big{|}}y\_{i}<y\_{j}\wedge y\_{p}=l\bigr{\}}. |  |

Therefore, if we denote P𝜶subscript𝑃𝜶P\_{\bm{\alpha}} to be the set of all the permutations of (1​⋯​1⏟α1​⋯​d​⋯​d⏟αd)subscript⏟1⋯1subscript𝛼1⋯subscript⏟𝑑⋯𝑑subscript𝛼𝑑(\underbrace{1\cdots 1}\_{\alpha\_{1}}\cdots\underbrace{d\cdots d}\_{\alpha\_{d}}), then we arrive at our claim

|  |  |  |  |
| --- | --- | --- | --- |
| (12) |  | c𝜶=∑j1,⋯,jp∈Pp∑i1,⋯,ip∈Bp∏k=1pwik(jk).subscript𝑐𝜶subscript  subscript𝑗1⋯subscript𝑗𝑝 subscript𝑃𝑝subscript  subscript𝑖1⋯subscript𝑖𝑝 subscript𝐵𝑝superscriptsubscriptproduct𝑘1𝑝superscriptsubscript𝑤subscript𝑖𝑘subscript𝑗𝑘c\_{\bm{\alpha}}=\sum\_{j\_{1},\cdots,j\_{p}\in P\_{p}}\sum\_{i\_{1},\cdots,i\_{p}\in B\_{p}}\prod\_{k=1}^{p}w\_{i\_{k}}^{(j\_{k})}. |  |

∎
