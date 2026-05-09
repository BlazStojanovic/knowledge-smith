---
arxiv: '1803.05170'
authors:
- Jianxun Lian
- Xiaohuan Zhou
- Fuzheng Zhang
- Zhongxia Chen
- Xing Xie
- Guangzhong Sun
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender
  Systems'
url: http://arxiv.org/abs/1803.05170v3
year: 2018
---

[1803.05170] xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems















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



# xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

Jianxun Lian
University of Science and Technology of China
[jianxun.lian@outlook.com](mailto:jianxun.lian@outlook.com)
, 
Xiaohuan Zhou
Beijing University of Posts and Telecommunications
[maggione@bupt.edu.cn](mailto:maggione@bupt.edu.cn)
, 
Fuzheng Zhang
Microsoft Research
[fuzzhang@microsoft.com](mailto:fuzzhang@microsoft.com)
, 
Zhongxia Chen
University of Science and Technology of China
[czx87@mail.ustc.edu.cn](mailto:czx87@mail.ustc.edu.cn)
, 
Xing Xie
Microsoft Research
[xingx@microsoft.com](mailto:xingx@microsoft.com)
 and 
Guangzhong Sun
University of Science and Technology of China
[gzsun@ustc.edu.cn](mailto:gzsun@ustc.edu.cn)

(2018)

###### Abstract.

Combinatorial features are essential for the success of many commercial models. Manually crafting these features usually comes with high cost due to the variety, volume and velocity of raw data in web-scale systems. Factorization based models, which measure interactions in terms of vector product, can learn patterns of combinatorial features automatically and generalize to unseen features as well. With the great success of deep neural networks (DNNs) in various fields, recently researchers have proposed several DNN-based factorization model to learn both low- and high-order feature interactions. Despite the powerful ability of learning an arbitrary function from data, plain DNNs generate feature interactions implicitly and at the bit-wise level. In this paper, we propose a novel Compressed Interaction Network (CIN), which aims to generate feature interactions in an explicit fashion and at the vector-wise level. We show that the CIN share some functionalities with convolutional neural networks (CNNs) and recurrent neural networks (RNNs). We further combine a CIN and a classical DNN into one unified model, and named this new model eXtreme Deep Factorization Machine (xDeepFM). On one hand, the xDeepFM is able to learn certain bounded-degree feature interactions explicitly; on the other hand, it can learn arbitrary low- and high-order feature interactions implicitly. We conduct comprehensive experiments on three real-world datasets. Our results demonstrate that xDeepFM outperforms state-of-the-art models. We have released the source code of xDeepFM at <https://github.com/Leavingseason/xDeepFM>.

Factorization machines, neural network, recommender systems, deep learning, feature interactions

††ccs: Information systems Personalization††ccs: Computing methodologies Neural networks††ccs: Computing methodologies Factorization methods††journalyear: 2018††copyright: acmcopyright††conference: The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; August 19–23, 2018; London, United Kingdom††booktitle: KDD ’18: The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 19–23, 2018, London, United Kingdom††price: 15.00††doi: 10.1145/3219819.3220023††isbn: 978-1-4503-5552-0/18/08

## 1. introduction

Features play a central role in the success of many predictive systems. Because using raw features can rarely lead to optimal results, data scientists usually spend a lot of work on the transformation of raw features in order to generate best predictive systems (He et al., [2014](#bib.bib15); Lian
et al., [2017c](#bib.bib25)) or to win data mining games (Liu et al., [2016](#bib.bib27); Lian
et al., [2017a](#bib.bib23); Lian and Xie, [2016](#bib.bib22)). One major type of feature transformation is the cross-product transformation over categorical features (Cheng et al., [2016](#bib.bib6)). These features are called cross features or multi-way features, they measure the interactions of multiple raw features. For instance, a 3-way feature AND(user\_organization=msra, item\_category=deeplearning, time=monday) has value 1 if the user works at Microsoft Research Asia and is shown a technical article about deep learning on a Monday.
  
There are three major downsides for traditional cross feature engineering. First, obtaining high-quality features comes with a high cost. Because right features are usually task-specific, data scientists need spend a lot of time exploring the potential patterns from the product data before they become domain experts and extract meaningful cross features. Second, in large-scale predictive systems such as web-scale recommender systems, the huge number of raw features makes it infeasible to extract all cross features manually. Third, hand-crafted cross features do not generalize to unseen interactions in the training data. Therefore, learning to interact features without manual engineering is a meaningful task.
  
Factorization Machines (FM) (Rendle, [2010](#bib.bib33)) embed each feature i𝑖i to a latent factor vector 𝐯i=[vi​1,vi​2,…,vi​D]subscript𝐯𝑖

subscript𝑣𝑖1subscript𝑣𝑖2…subscript𝑣𝑖𝐷\mathbf{v}\_{i}=[v\_{i1},v\_{i2},...,v\_{iD}], and pairwise feature interactions are modeled as the inner product of latent vectors: f(2)​(i,j)=⟨𝐯i,𝐯j⟩​xi​xjsuperscript𝑓2𝑖𝑗

subscript𝐯𝑖subscript𝐯𝑗
subscript𝑥𝑖subscript𝑥𝑗f^{(2)}(i,j)=\langle\mathbf{v}\_{i},\mathbf{v}\_{j}\rangle x\_{i}x\_{j}. In this paper we use the term bit to denote a element (such as vi​1subscript𝑣𝑖1v\_{i1}) in latent vectors. The classical FM can be extended to arbitrary higher-order feature interactions (Blondel
et al., [2016](#bib.bib3)), but one major downside is that, (Blondel
et al., [2016](#bib.bib3)) proposes to model all feature interactions, including both useful and useless combinations. As revealed in (Xiao
et al., [2017](#bib.bib44)), the interactions with useless features may introduce noises and degrade the performance. In recent years, deep neural networks (DNNs) have become successful in computer vision, speech recognition, and natural language processing with their great power of feature representation learning. It is promising to exploit DNNs to learn sophisticated and selective feature interactions. (Zhang
et al., [2016a](#bib.bib47)) proposes a Factorisation-machine supported Neural Network (FNN) to learn high-order feature interactions. It uses the pre-trained factorization machines for field embedding before applying DNN. (Qu
et al., [2016](#bib.bib32)) further proposes a Product-based Neural Network (PNN), which introduces a product layer between embedding layer and DNN layer, and does not rely on pre-trained FM. The major downside of FNN and PNN is that they focus more on high-order feature interactions while capture little low-order interactions. The Wide&Deep (Cheng et al., [2016](#bib.bib6)) and DeepFM (Guo
et al., [2017](#bib.bib10)) models overcome this problem by introducing hybrid architectures, which contain a shallow component and a deep component with the purpose of learning both memorization and generalization. Therefore they can jointly learn low-order and high-order feature interactions.
  
All the abovementioned models leverage DNNs for learning high-order feature interactions. However, DNNs model high-order feature interactions in an implicit fashion. The final function learned by DNNs can be arbitrary, and there is no theoretical conclusion on what the maximum degree of feature interactions is. In addition, DNNs model feature interactions at the bit-wise level, which is different from the traditional FM framework which models feature interactions at the vector-wise level. Thus, in the field of recommender systems, whether DNNs are indeed the most effective model in representing high-order feature interactions remains an open question. In this paper, we propose a neural network-based model to learn feature interactions in an explicit, vector-wise fashion. Our approach is based on the Deep & Cross Network (DCN) (Wang
et al., [2017](#bib.bib41)), which aims to efficiently capture feature interactions of bounded degrees. However, we will argue in Section [2.3](#S2.SS3 "2.3. Explicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") that DCN will lead to a special format of interactions. We thus design a novel compressed interaction network (CIN) to replace the cross network in the DCN. CIN learns feature interactions explicitly, and the degree of interactions grows with the depth of the network. Following the spirit of the Wide&Deep and DeepFM models, we combine the explicit high-order interaction module with implicit interaction module and traditional FM module, and name the joint model eXtreme Deep Factorization Machine (xDeepFM). The new model requires no manual feature engineering and release data scientists from tedious feature searching work. To summarize, we make the following contributions:

* •

  We propose a novel model, named eXtreme Deep Factorization Machine (xDeepFM), that jointly learns explicit and implicit high-order feature interactions effectively and requires no manual feature engineering.
* •

  We design a compressed interaction network (CIN) in xDeepFM that learns high-order feature interactions explicitly. We show that the degree of feature interactions increases at each layer, and features interact at the vector-wise level rather than the bit-wise level.
* •

  We conduct extensive experiments on three real-world dataset, and the results demonstrate that our xDeepFM outperforms several state-of-the-art models significantly.

The rest of this paper is organized as follows. Section [2](#S2 "2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") provides some preliminary knowledge which is necessary for understanding deep learning-based recommender systems. Section [3](#S3 "3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") introduces our proposed CIN and xDeepFM model in detail. We will present experimental explorations on multiple datasets in Section [4](#S4 "4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). Related works are discussed in Section [5](#S5 "5. related work ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). Section [6](#S6 "6. Conclusions ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") concludes this paper.

## 2. Preliminaries

### 2.1. Embedding Layer

In computer vision or natural language understanding, the input data are usually images or textual signals, which are known to be spatially and/or temporally correlated, so DNNs can be applied directly on the raw feature with dense structures. However, in web-scale recommender systems, the input features are sparse, of huge dimension, and present no clear spatial or temporal correlation. Therefore, multi-field categorical form is widely used by related works (Guo
et al., [2017](#bib.bib10); Shan
et al., [2016](#bib.bib38); Wang
et al., [2017](#bib.bib41); Qu
et al., [2016](#bib.bib32); Zhang
et al., [2016a](#bib.bib47)). For example, one input instance [user\_id=s02,gender=male,
  
organization=msra,interests=comedy&rock] is normally transformed into a high-dimensional sparse features via field-aware one-hot encoding:

|  |  |  |
| --- | --- | --- |
|  | [0,1,0,0,…,0⏟u​s​e​r​i​d]​[1,0⏟g​e​n​d​e​r]​[0,1,0,0,…,0⏟o​r​g​a​n​i​z​a​t​i​o​n]​[0,1,0,1,…,0⏟i​n​t​e​r​e​s​t​s]delimited-[]subscript⏟  0100…0𝑢𝑠𝑒𝑟𝑖𝑑delimited-[]subscript⏟  10𝑔𝑒𝑛𝑑𝑒𝑟delimited-[]subscript⏟  0100…0𝑜𝑟𝑔𝑎𝑛𝑖𝑧𝑎𝑡𝑖𝑜𝑛delimited-[]subscript⏟  0101…0𝑖𝑛𝑡𝑒𝑟𝑒𝑠𝑡𝑠[\underbrace{0,1,0,0,...,0}\_{userid}]\ [\underbrace{1,0}\_{gender}]\ [\underbrace{0,1,0,0,...,0}\_{organization}]\ [\underbrace{0,1,0,1,...,0}\_{interests}] |  |

An embedding layer is applied upon the raw feature input to compress it to a low dimensional, dense real-value vector. If the field is univalent, the feature embedding is used as the field embedding. Take the above instance as an example, the embedding of feature male is taken as the embedding of field gender. If the field is multivalent, the sum of feature embedding is used as the field embedding. The embedding layer is illustrated in Figure [1](#S2.F1 "Figure 1 ‣ 2.1. Embedding Layer ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). The result of embedding layer is a wide concatenated vector:

|  |  |  |
| --- | --- | --- |
|  | 𝐞=[𝐞1,𝐞2,…,𝐞m]𝐞  subscript𝐞1subscript𝐞2…subscript𝐞𝑚\mathbf{e}=[\mathbf{e}\_{1},\mathbf{e}\_{2},...,\mathbf{e}\_{m}] |  |

where m𝑚m denotes the number of fields, and 𝐞𝐢∈ℝDsubscript𝐞𝐢superscriptℝ𝐷\mathbf{e\_{i}}\in\mathbb{R}^{D} denotes the embedding of one field. Although the feature lengths of instances can be various, their embeddings are of the same length m×D𝑚𝐷m\times D, where D𝐷D is the dimension of field embedding.

![Refer to caption](/html/1803.05170/assets/x1.png)


Figure 1. The field embedding layer. The dimension of embedding in this example is 4.

### 2.2. Implicit High-order Interactions

FNN (Zhang
et al., [2016a](#bib.bib47)), Deep Crossing (Shan
et al., [2016](#bib.bib38)), and the deep part in Wide&Deep (Cheng et al., [2016](#bib.bib6)) exploit a feed-forward neural network on the field embedding vector 𝐞𝐞\mathbf{e} to learn high-order feature interactions. The forward process is :

|  |  |  |  |
| --- | --- | --- | --- |
| (1) |  | 𝐱1=σ​(𝐖(1)​𝐞+𝐛1)superscript𝐱1𝜎superscript𝐖1𝐞superscript𝐛1\mathbf{x}^{1}=\sigma(\mathbf{W}^{(1)}\mathbf{e}+\mathbf{b}^{1}) |  |

|  |  |  |  |
| --- | --- | --- | --- |
| (2) |  | 𝐱k=σ​(𝐖(k)​𝐱(k−1)+𝐛k)superscript𝐱𝑘𝜎superscript𝐖𝑘superscript𝐱𝑘1superscript𝐛𝑘\mathbf{x}^{k}=\sigma(\mathbf{W}^{(k)}\mathbf{x}^{(k-1)}+\mathbf{b}^{k}) |  |

where k𝑘k is the layer depth, σ𝜎\sigma is an activation function, and 𝐱ksuperscript𝐱𝑘\mathbf{x}^{k} is the output of the k𝑘k-th layer. The visual structure is very similar to what is shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.2. Implicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), except that they do not include the FM or Product layer. This architecture models the interaction in a bit-wise fashion. That is to say, even the elements within the same field embedding vector will influence each other.
  
PNN (Qu
et al., [2016](#bib.bib32)) and DeepFM (Guo
et al., [2017](#bib.bib10)) modify the above architecture slightly. Besides applying DNNs on the embedding vector 𝐞𝐞\mathbf{e}, they add a two-way interaction layer in the architecture. Therefore, both bit-wise and vector-wise interaction is included in their model. The major difference between PNN and DeepFM, is that PNN connects the outputs of product layer to the DNNs, whereas DeepFM connects the FM layer directly to the output unit (refer to Figure [2](#S2.F2 "Figure 2 ‣ 2.2. Implicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems")).

![Refer to caption](/html/1803.05170/assets/x2.png)


Figure 2. The architecture of DeepFM (with linear part omitted) and PNN. We re-use the symbols in (Guo
et al., [2017](#bib.bib10)), where red edges represent weight-1 connections (no parameters) and gray edges represent normal connections (network parameters).

### 2.3. Explicit High-order Interactions

(Wang
et al., [2017](#bib.bib41)) proposes the Cross Network (CrossNet) whose architecture is shown in Figure [3](#S2.F3 "Figure 3 ‣ 2.3. Explicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). It aims to explicitly model the high-order feature interactions. Unlike the classical fully-connected feed-forward network, the hidden layers are calculated by the following cross operation:

|  |  |  |  |
| --- | --- | --- | --- |
| (3) |  | 𝐱k=𝐱0​𝐱k−1T​𝐰k+𝐛k+𝐱k−1subscript𝐱𝑘subscript𝐱0superscriptsubscript𝐱𝑘1𝑇subscript𝐰𝑘subscript𝐛𝑘subscript𝐱𝑘1\mathbf{x}\_{k}=\mathbf{x}\_{0}\mathbf{x}\_{k-1}^{T}\mathbf{w}\_{k}+\mathbf{b}\_{k}+\mathbf{x}\_{k-1} |  |

where 𝐰k,𝐛k,𝐱k∈ℝm​D

subscript𝐰𝑘subscript𝐛𝑘subscript𝐱𝑘
superscriptℝ𝑚𝐷\mathbf{w}\_{k},\mathbf{b}\_{k},\mathbf{x}\_{k}\in\mathbb{R}^{mD} are weights, bias and output of the k𝑘k-th layer, respectively. We argue that the CrossNet learns a special type of high-order feature interactions, where each hidden layer in the CrossNet is a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}.

![Refer to caption](/html/1803.05170/assets/x3.png)


Figure 3. The architecture of the Cross Network.

###### Theorem 2.1.

Consider a k𝑘k-layer cross network with the (i+1)-th layer defined as 𝐱i+1=𝐱0​𝐱iT​𝐰i+1+𝐱isubscript𝐱𝑖1subscript𝐱0superscriptsubscript𝐱𝑖𝑇subscript𝐰𝑖1subscript𝐱𝑖\mathbf{x}\_{i+1}=\mathbf{x}\_{0}\mathbf{x}\_{i}^{T}\mathbf{w}\_{i+1}+\mathbf{x}\_{i}. Then, the output of the cross network 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} is a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}.

###### Proof.

When k𝑘k=1, according to the associative law and distributive law for matrix multiplication, we have:

|  |  |  |  |
| --- | --- | --- | --- |
| (4) |  | 𝐱1=𝐱0​(𝐱0T​𝐰1)+𝐱0=𝐱0​(𝐱0T​𝐰1+1)=α1​𝐱0subscript𝐱1subscript𝐱0superscriptsubscript𝐱0𝑇subscript𝐰1subscript𝐱0subscript𝐱0superscriptsubscript𝐱0𝑇subscript𝐰11superscript𝛼1subscript𝐱0\begin{split}\mathbf{x}\_{1}&=\mathbf{x}\_{0}(\mathbf{x}\_{0}^{T}\mathbf{w}\_{1})+\mathbf{x}\_{0}\\ &=\mathbf{x}\_{0}(\mathbf{x}\_{0}^{T}\mathbf{w}\_{1}+1)\\ &=\alpha^{1}\mathbf{x}\_{0}\end{split} |  |

where the scalar α1=𝐱0T​𝐰1+1superscript𝛼1superscriptsubscript𝐱0𝑇subscript𝐰11\alpha^{1}=\mathbf{x}\_{0}^{T}\mathbf{w}\_{1}+1 is actually a linear regression of 𝐱0subscript𝐱0\mathbf{x}\_{0}. Thus, 𝐱1subscript𝐱1\mathbf{x}\_{1} is a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}. Suppose the scalar multiple statement holds for k𝑘k=i𝑖i. For k𝑘k=i+1𝑖1i+1, we have :

|  |  |  |  |
| --- | --- | --- | --- |
| (5) |  | 𝐱i+1=𝐱0​𝐱iT​𝐰i+1+𝐱i=𝐱0​((αi​𝐱0)T​𝐰i+1)+αi​𝐱0=αi+1​𝐱0subscript𝐱𝑖1subscript𝐱0superscriptsubscript𝐱𝑖𝑇subscript𝐰𝑖1subscript𝐱𝑖subscript𝐱0superscriptsuperscript𝛼𝑖subscript𝐱0𝑇subscript𝐰𝑖1superscript𝛼𝑖subscript𝐱0superscript𝛼𝑖1subscript𝐱0\begin{split}\mathbf{x}\_{i+1}&=\mathbf{x}\_{0}\mathbf{x}\_{i}^{T}\mathbf{w}\_{i+1}+\mathbf{x}\_{i}\\ &=\mathbf{x}\_{0}((\alpha^{i}\mathbf{x}\_{0})^{T}\mathbf{w}\_{i+1})+\alpha^{i}\mathbf{x}\_{0}\\ &=\alpha^{i+1}\mathbf{x}\_{0}\end{split} |  |

where, αi+1=αi​(𝐱0T​𝐰i+1+1)superscript𝛼𝑖1superscript𝛼𝑖superscriptsubscript𝐱0𝑇subscript𝐰𝑖11\alpha^{i+1}=\alpha^{i}(\mathbf{x}\_{0}^{T}\mathbf{w}\_{i+1}+1) is a scalar. Thus 𝐱i+1subscript𝐱𝑖1\mathbf{x}\_{i+1} is still a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}. By induction hypothesis, the output of cross network 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} is a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}.
∎

Note that the scalar multiple does not mean 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} is linear with 𝐱0subscript𝐱0\mathbf{x}\_{0}. The coefficient αi+1superscript𝛼𝑖1\alpha^{i+1} is sensitive with 𝐱0subscript𝐱0\mathbf{x}\_{0}. The CrossNet can learn feature interactions very efficiently (the complexity is negligible compared with a DNN model), however the downsides are: (1) the output of CrossNet is limited in a special form, with each hidden layer is a scalar multiple of 𝐱0subscript𝐱0\mathbf{x}\_{0}; (2) interactions come in a bit-wise fashion.

## 3. Our proposed model

![Refer to caption](/html/1803.05170/assets/x4.png)


(a) Outer products along each dimension for feature interactions. The tensor 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1} is an intermediate result for further learning.

![Refer to caption](/html/1803.05170/assets/x5.png)


(b) The k𝑘k-th layer of CIN. It compresses the intermediate tensor 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1} to Hk+1subscript𝐻𝑘1H\_{k+1} embedding vectors (aslo known as feature maps).

![Refer to caption](/html/1803.05170/assets/x6.png)


(c) An overview of the CIN architecture.

Figure 4. Components and architecture of the Compressed Interaction Network (CIN).

### 3.1. Compressed Interaction Network

We design a new cross network, named Compressed Interaction Network (CIN), with the following considerations: (1) interactions are applied at vector-wise level, not at bit-wise level; (2) high-order feature interactions is measured explicitly; (3) the complexity of network will not grow exponentially with the degree of interactions.
  
Since an embedding vector is regarded as a unit for vector-wise interactions, hereafter we formulate the output of field embedding as a matrix 𝐗0∈ℝm×Dsuperscript𝐗0superscriptℝ𝑚𝐷\mathbf{X}^{0}\in\mathbb{R}^{m\times D}, where the i𝑖i-th row in 𝐗0superscript𝐗0\mathbf{X}^{0} is the embedding vector of the i𝑖i-th field: 𝐗i,∗0=𝐞isubscriptsuperscript𝐗0

𝑖subscript𝐞𝑖\mathbf{X}^{0}\_{i,\*}=\mathbf{e}\_{i}, and D𝐷D is the dimension of the field embedding. The output of the k𝑘k-th layer in CIN is also a matrix 𝐗k∈ℝHk×Dsuperscript𝐗𝑘superscriptℝsubscript𝐻𝑘𝐷\mathbf{X}^{k}\in\mathbb{R}^{H\_{k}\times D}, where Hksubscript𝐻𝑘H\_{k} denotes the number of (embedding) feature vectors in the k𝑘k-th layer and we let H0=msubscript𝐻0𝑚H\_{0}=m. For each layer, 𝐗ksuperscript𝐗𝑘\mathbf{X}^{k} are calculated via:

|  |  |  |  |
| --- | --- | --- | --- |
| (6) |  | 𝐗h,∗k=∑i=1Hk−1∑j=1m𝐖i​jk,h​(𝐗i,∗k−1∘𝐗j,∗0)subscriptsuperscript𝐗𝑘  ℎsuperscriptsubscript𝑖1subscript𝐻𝑘1superscriptsubscript𝑗1𝑚subscriptsuperscript𝐖  𝑘ℎ𝑖𝑗subscriptsuperscript𝐗𝑘1  𝑖subscriptsuperscript𝐗0  𝑗\mathbf{X}^{k}\_{h,\*}=\sum\_{i=1}^{H\_{k-1}}\sum\_{j=1}^{m}\mathbf{W}^{k,h}\_{ij}(\mathbf{X}^{k-1}\_{i,\*}\circ\mathbf{X}^{0}\_{j,\*}) |  |

where 1≤h≤Hk1ℎsubscript𝐻𝑘1\leq h\leq H\_{k}, 𝐖k,h∈ℝHk−1×msuperscript𝐖

𝑘ℎsuperscriptℝsubscript𝐻𝑘1𝑚\mathbf{W}^{k,h}\in\mathbb{R}^{H\_{k-1}\times m} is the parameter matrix for the hℎh-th feature vector, and ∘\circ denotes the Hadamard product, for example, ⟨a1,a2,a3⟩∘⟨b1,b2,b3⟩=⟨a1​b1,a2​b2,a3​b3⟩

subscript𝑎1subscript𝑎2subscript𝑎3

subscript𝑏1subscript𝑏2subscript𝑏3

subscript𝑎1subscript𝑏1subscript𝑎2subscript𝑏2subscript𝑎3subscript𝑏3\langle a\_{1},a\_{2},a\_{3}\rangle\circ\langle b\_{1},b\_{2},b\_{3}\rangle=\langle a\_{1}b\_{1},a\_{2}b\_{2},a\_{3}b\_{3}\rangle. Note that 𝐗ksuperscript𝐗𝑘\mathbf{X}^{k} is derived via the interactions between 𝐗k−1superscript𝐗𝑘1\mathbf{X}^{k-1} and 𝐗0superscript𝐗0\mathbf{X}^{0}, thus feature interactions are measured explicitly and the degree of interactions increases with the layer depth. The structure of CIN is very similar to the Recurrent Neural Network (RNN), where the outputs of the next hidden layer are dependent on the last hidden layer and an additional input. We hold the structure of embedding vectors at all layers, thus the interactions are applied at the vector-wise level.
  
It is interesting to point out that Equation [6](#S3.E6 "In 3.1. Compressed Interaction Network ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") has strong connections with the well-known Convolutional Neural Networks (CNNs) in computer vision. As shown in Figure [4(a)](#S3.F4.sf1 "In Figure 4 ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), we introduce an intermediate tensor 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1}, which is the outer products (along each embedding dimension) of hidden layer 𝐗ksuperscript𝐗𝑘\mathbf{X}^{k} and original feature matrix 𝐗0superscript𝐗0\mathbf{X}^{0}. Then 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1} can be regarded as a special type of image and 𝐖k,hsuperscript𝐖

𝑘ℎ\mathbf{W}^{k,h} is a filter. We slide the filter across 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1} along the embedding dimension (D) as shown in Figure [4(b)](#S3.F4.sf2 "In Figure 4 ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), and get an hidden vector 𝐗i,∗k+1subscriptsuperscript𝐗𝑘1

𝑖\mathbf{X}^{k+1}\_{i,\*}, which is usually called a feature map in computer vision. Therefore, 𝐗ksuperscript𝐗𝑘\mathbf{X}^{k} is a collection of Hksubscript𝐻𝑘H\_{k} different feature maps. The term “compressed” in the name of CIN indicates that the k𝑘k-th hidden layer compress the potential space of Hk−1×msubscript𝐻𝑘1𝑚H\_{k-1}\times m vectors down to Hksubscript𝐻𝑘H\_{k} vectors.
  
Figure [4(c)](#S3.F4.sf3 "In Figure 4 ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") provides an overview of the architecture of CIN. Let T denotes the depth of the network. Every hidden layer 𝐗k,k∈[1,T]

superscript𝐗𝑘𝑘
1𝑇\mathbf{X}^{k},k\in[1,T] has a connection with output units. We first apply sum pooling on each feature map of the hidden layer:

|  |  |  |  |
| --- | --- | --- | --- |
| (7) |  | pik=∑j=1D𝐗i,jksubscriptsuperscript𝑝𝑘𝑖superscriptsubscript𝑗1𝐷subscriptsuperscript𝐗𝑘  𝑖𝑗{p}^{k}\_{i}=\sum\_{j=1}^{D}\mathbf{X}^{k}\_{i,j} |  |

for i∈[1,Hk]𝑖1subscript𝐻𝑘i\in[1,H\_{k}]. Thus, we have a pooling vector 𝐩k=[p1k,p2k,…,pHkk]superscript𝐩𝑘

subscriptsuperscript𝑝𝑘1subscriptsuperscript𝑝𝑘2…subscriptsuperscript𝑝𝑘subscript𝐻𝑘\mathbf{p}^{k}=[p^{k}\_{1},p^{k}\_{2},...,p^{k}\_{H\_{k}}] with length Hksubscript𝐻𝑘H\_{k} for the k𝑘k-th hidden layer. All pooling vectors from hidden layers are concatenated before connected to output units: 𝐩+=[𝐩1,𝐩2,…,𝐩T]∈ℝ∑i=1THisuperscript𝐩

superscript𝐩1superscript𝐩2…superscript𝐩𝑇superscriptℝsuperscriptsubscript𝑖1𝑇subscript𝐻𝑖\mathbf{p}^{+}=[\mathbf{p}^{1},\mathbf{p}^{2},...,\mathbf{p}^{T}]\in\mathbb{R}^{\sum\_{i=1}^{T}H\_{i}}. If we use CIN directly for binary classification, the output unit is a sigmoid node on 𝐩+superscript𝐩\mathbf{p}^{+}:

|  |  |  |  |
| --- | --- | --- | --- |
| (8) |  | y=11+e​x​p​(𝐩+T​𝐰o)𝑦11𝑒𝑥𝑝superscriptsuperscript𝐩𝑇superscript𝐰𝑜y=\frac{1}{1+exp(\mathbf{p^{+}}^{T}\mathbf{w}^{o})} |  |

where 𝐰osuperscript𝐰𝑜\mathbf{w}^{o} are the regression parameters.

### 3.2. CIN Analysis

We analyze the proposed CIN to study the model complexity and the potential effectiveness.

#### 3.2.1. Space Complexity

The hℎh-th feature map at the k𝑘k-th layer contains Hk−1×msubscript𝐻𝑘1𝑚H\_{k-1}\times m parameters, which is exactly the size of 𝐖k,hsuperscript𝐖

𝑘ℎ\mathbf{W}^{k,h}. Thus, there are Hk×Hk−1×msubscript𝐻𝑘subscript𝐻𝑘1𝑚H\_{k}\times H\_{k-1}\times m parameters at the k𝑘k-th layer. Considering the last regression layer for the output unit, which has ∑k=1THksuperscriptsubscript𝑘1𝑇subscript𝐻𝑘\sum\_{k=1}^{T}H\_{k} parameters, the total number of parameters for CIN is ∑k=1THk×(1+Hk−1×m)superscriptsubscript𝑘1𝑇subscript𝐻𝑘1subscript𝐻𝑘1𝑚\sum\_{k=1}^{T}H\_{k}\times(1+H\_{k-1}\times m). Note that CIN is independent of the embedding dimension D𝐷D. In contrast, a plain T𝑇T-layers DNN contains m×D×H1+HT+∑k=2THk×Hk−1𝑚𝐷subscript𝐻1subscript𝐻𝑇superscriptsubscript𝑘2𝑇subscript𝐻𝑘subscript𝐻𝑘1m\times D\times H\_{1}+H\_{T}+\sum\_{k=2}^{T}H\_{k}\times H\_{k-1} parameters, and the number of parameters will increase with the embedding dimension D𝐷D.
  
Usually m𝑚m and Hksubscript𝐻𝑘H\_{k} will not be very large, so the scale of 𝐖k,hsuperscript𝐖

𝑘ℎ\mathbf{W}^{k,h} is acceptable. When necessary, we can exploit a L𝐿L-order decomposition and replace 𝐖k,hsuperscript𝐖

𝑘ℎ\mathbf{W}^{k,h} with two smaller matrices 𝐔k,h∈ℝHk−1×Lsuperscript𝐔

𝑘ℎsuperscriptℝsubscript𝐻𝑘1𝐿\mathbf{U}^{k,h}\in\mathbb{R}^{H\_{k-1}\times L} and 𝐕k,h∈ℝm×Lsuperscript𝐕

𝑘ℎsuperscriptℝ𝑚𝐿\mathbf{V}^{k,h}\in\mathbb{R}^{m\times L}:

|  |  |  |  |
| --- | --- | --- | --- |
| (9) |  | 𝐖k,h=𝐔k,h​(𝐕k,h)Tsuperscript𝐖  𝑘ℎsuperscript𝐔  𝑘ℎsuperscriptsuperscript𝐕  𝑘ℎ𝑇\mathbf{W}^{k,h}=\mathbf{U}^{k,h}(\mathbf{V}^{k,h})^{T} |  |

where L≪Hmuch-less-than𝐿𝐻L\ll H and L≪mmuch-less-than𝐿𝑚L\ll m. Hereafter we assume that each hidden layer has the same number (which is H𝐻H) of feature maps for simplicity. Through the L𝐿L-order decomposition, the space complexity of CIN is reduced from O​(m​T​H2)𝑂𝑚𝑇superscript𝐻2O(mTH^{2}) to O​(m​T​H​L+T​H2​L)𝑂𝑚𝑇𝐻𝐿𝑇superscript𝐻2𝐿O(mTHL+TH^{2}L). In contrast, the space complexity of the plain DNN is O​(m​D​H+T​H2)𝑂𝑚𝐷𝐻𝑇superscript𝐻2O(mDH+TH^{2}), which is sensitive to the dimension (D) of field embedding.

#### 3.2.2. Time Complexity

The cost of computing tensor 𝐙k+1superscript𝐙𝑘1\mathbf{Z}^{k+1} (as shown in Figure [4(a)](#S3.F4.sf1 "In Figure 4 ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems")) is O​(m​H​D)𝑂𝑚𝐻𝐷O(mHD) time. Because we have H𝐻H feature maps in one hidden layer, computing a T𝑇T-layers CIN takes O​(m​H2​D​T)𝑂𝑚superscript𝐻2𝐷𝑇O(mH^{2}DT) time. A T𝑇T-layers plain DNN, by contrast, takes O​(m​H​D+H2​T)𝑂𝑚𝐻𝐷superscript𝐻2𝑇O(mHD+H^{2}T) time. Therefore, the major downside of CIN lies in the time complexity.

#### 3.2.3. Polynomial Approximation

Next we examine the high-order interaction properties of CIN. For simplicity, we assume that numbers of feature maps at hidden layers are all equal to the number of fields m𝑚m. Let [m]delimited-[]𝑚[m] denote the set of positive integers that are less than or equal to m𝑚m. The hℎh-th feature map at the first layer, denoted as 𝐱h1∈ℝDsubscriptsuperscript𝐱1ℎsuperscriptℝ𝐷\mathbf{x}^{1}\_{h}\ \in\mathbb{R}^{D}, is calculated via:

|  |  |  |  |
| --- | --- | --- | --- |
| (10) |  | 𝐱h1=∑i∈[m]j∈[m]𝐖i,j1,h​(𝐱i0∘𝐱j0)subscriptsuperscript𝐱1ℎsubscript  𝑖delimited-[]𝑚𝑗delimited-[]𝑚subscriptsuperscript𝐖  1ℎ  𝑖𝑗subscriptsuperscript𝐱0𝑖subscriptsuperscript𝐱0𝑗\mathbf{x}^{1}\_{h}=\sum\_{\begin{subarray}{c}i\in[m]\\ j\in[m]\end{subarray}}\mathbf{W}^{1,h}\_{i,j}(\mathbf{x}^{0}\_{i}\circ\mathbf{x}^{0}\_{j}) |  |

Therefore, each feature map at the first layer models pair-wise interactions with O​(m2)𝑂superscript𝑚2O(m^{2}) coefficients. Similarly, the hℎh-th feature map at the second layer is:

|  |  |  |  |
| --- | --- | --- | --- |
| (11) |  | 𝐱h2=∑i∈[m]j∈[m]𝐖i,j2,h​(𝐱i1∘𝐱j0)=∑i∈[m]j∈[m]∑l∈[m]k∈[m]𝐖i,j2,h​𝐖l,k1,i​(𝐱j0∘𝐱k0∘𝐱l0)subscriptsuperscript𝐱2ℎsubscript  𝑖delimited-[]𝑚𝑗delimited-[]𝑚subscriptsuperscript𝐖  2ℎ  𝑖𝑗subscriptsuperscript𝐱1𝑖subscriptsuperscript𝐱0𝑗subscript  𝑖delimited-[]𝑚𝑗delimited-[]𝑚subscript  𝑙delimited-[]𝑚𝑘delimited-[]𝑚subscriptsuperscript𝐖  2ℎ  𝑖𝑗subscriptsuperscript𝐖  1𝑖  𝑙𝑘subscriptsuperscript𝐱0𝑗subscriptsuperscript𝐱0𝑘subscriptsuperscript𝐱0𝑙\begin{split}\mathbf{x}^{2}\_{h}&=\sum\_{\begin{subarray}{c}i\in[m]\\ j\in[m]\end{subarray}}\mathbf{W}^{2,h}\_{i,j}(\mathbf{x}^{1}\_{i}\circ\mathbf{x}^{0}\_{j})\\ &=\sum\_{\begin{subarray}{c}i\in[m]\\ j\in[m]\end{subarray}}\sum\_{\begin{subarray}{c}l\in[m]\\ k\in[m]\end{subarray}}\mathbf{W}^{2,h}\_{i,j}\mathbf{W}^{1,i}\_{l,k}(\mathbf{x}^{0}\_{j}\circ\mathbf{x}^{0}\_{k}\circ\mathbf{x}^{0}\_{l})\end{split} |  |

Note that all calculations related to the subscript l𝑙l and k𝑘k is already finished at the previous hidden layer. We expand the factors in Equation [11](#S3.E11 "In 3.2.3. Polynomial Approximation ‣ 3.2. CIN Analysis ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") just for clarity. We can observe that each feature map at the second layer models 3-way interactions with O​(m2)𝑂superscript𝑚2O(m^{2}) new parameters.
  
A classical k𝑘k-order polynomial has O​(mk)𝑂superscript𝑚𝑘O(m^{k}) coefficients. We show that CIN approximate this class of polynomial with only O​(k​m3)𝑂𝑘superscript𝑚3O(km^{3}) parameters in terms of a chain of feature maps. By induction hypothesis, we can prove that the hℎh-th feature map at the k𝑘k-th layer is:

|  |  |  |  |
| --- | --- | --- | --- |
| (12) |  | 𝐱hk=∑i∈[m]j∈[m]𝐖i,jk,h​(𝐱ik−1∘𝐱j0)=∑i∈[m]j∈[m]…​∑r∈[m]t∈[m]∑l∈[m]s∈[m]𝐖i,jk,h​…​𝐖l,s1,r​(𝐱j0∘…∘𝐱s0∘𝐱l0⏟k​v​e​c​t​o​r​s)subscriptsuperscript𝐱𝑘ℎsubscript  𝑖delimited-[]𝑚𝑗delimited-[]𝑚subscriptsuperscript𝐖  𝑘ℎ  𝑖𝑗subscriptsuperscript𝐱𝑘1𝑖subscriptsuperscript𝐱0𝑗subscript  𝑖delimited-[]𝑚𝑗delimited-[]𝑚…subscript  𝑟delimited-[]𝑚𝑡delimited-[]𝑚subscript  𝑙delimited-[]𝑚𝑠delimited-[]𝑚subscriptsuperscript𝐖  𝑘ℎ  𝑖𝑗…subscriptsuperscript𝐖  1𝑟  𝑙𝑠subscript⏟subscriptsuperscript𝐱0𝑗…subscriptsuperscript𝐱0𝑠subscriptsuperscript𝐱0𝑙𝑘𝑣𝑒𝑐𝑡𝑜𝑟𝑠\begin{split}\mathbf{x}^{k}\_{h}&=\sum\_{\begin{subarray}{c}i\in[m]\\ j\in[m]\end{subarray}}\mathbf{W}^{k,h}\_{i,j}(\mathbf{x}^{k-1}\_{i}\circ\mathbf{x}^{0}\_{j})\\ &=\sum\_{\begin{subarray}{c}i\in[m]\\ j\in[m]\end{subarray}}...\sum\_{\begin{subarray}{c}r\in[m]\\ t\in[m]\end{subarray}}\sum\_{\begin{subarray}{c}l\in[m]\\ s\in[m]\end{subarray}}\mathbf{W}^{k,h}\_{i,j}...\mathbf{W}^{1,r}\_{l,s}(\underbrace{\mathbf{x}^{0}\_{j}\circ...\circ\mathbf{x}^{0}\_{s}\circ\mathbf{x}^{0}\_{l}}\_{k\ vectors})\end{split} |  |

For better illustration, here we borrow the notations from (Wang
et al., [2017](#bib.bib41)). Let 𝜶=[α1,…,αm]∈ℕd𝜶

subscript𝛼1…subscript𝛼𝑚superscriptℕ𝑑\boldsymbol{\alpha}=[\alpha\_{1},...,\alpha\_{m}]\in\mathbb{N}^{d} denote a multi-index, and |𝜶|=∑i=1mαi𝜶superscriptsubscript𝑖1𝑚subscript𝛼𝑖|\boldsymbol{\alpha}|=\sum\_{i=1}^{m}\alpha\_{i}. We omit the original superscript from 𝐱i0subscriptsuperscript𝐱0𝑖\mathbf{x}^{0}\_{i}, and use 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} to denote it since we only we the feature maps from the 00-th layer (which is exactly the field embeddings) for the final expanded expression (refer to Eq. [12](#S3.E12 "In 3.2.3. Polynomial Approximation ‣ 3.2. CIN Analysis ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems")). Now a superscript is used to denote the vector operation, such as 𝐱i3=𝐱i∘𝐱i∘𝐱isubscriptsuperscript𝐱3𝑖subscript𝐱𝑖subscript𝐱𝑖subscript𝐱𝑖\mathbf{x}^{3}\_{i}=\mathbf{x}\_{i}\circ\mathbf{x}\_{i}\circ\mathbf{x}\_{i}. Let V​Pk​(𝐗)𝑉subscript𝑃𝑘𝐗VP\_{k}(\mathbf{X}) denote a multi-vector polynomial of degree k𝑘k:

|  |  |  |  |
| --- | --- | --- | --- |
| (13) |  | V​Pk​(𝐗)={∑𝜶w𝜶​𝐱1α1∘𝐱2α2∘…∘𝐱mαm|2⩽|𝜶|⩽k}𝑉subscript𝑃𝑘𝐗conditional-setsubscript𝜶subscript𝑤𝜶superscriptsubscript𝐱1subscript𝛼1superscriptsubscript𝐱2subscript𝛼2…superscriptsubscript𝐱𝑚subscript𝛼𝑚2𝜶𝑘VP\_{k}(\mathbf{X})=\left\{\left.\sum\_{\boldsymbol{\alpha}}w\_{\boldsymbol{\alpha}}\mathbf{x}\_{1}^{\alpha\_{1}}\circ\mathbf{x}\_{2}^{\alpha\_{2}}\circ...\circ\mathbf{x}\_{m}^{\alpha\_{m}}\right|2\leqslant|\boldsymbol{\alpha}|\leqslant k\right\} |  |

Each vector polylnomial in this class has O​(mk)𝑂superscript𝑚𝑘O(m^{k}) coefficients. Then, our CIN approaches the coefficient w𝜶subscript𝑤𝜶w\_{\boldsymbol{\alpha}} with:

|  |  |  |  |
| --- | --- | --- | --- |
| (14) |  | w^𝜶=∑i=1m∑j=1m∑B∈P𝜶∏t=2|𝜶|𝐖i,Btt,jsubscript^𝑤𝜶superscriptsubscript𝑖1𝑚superscriptsubscript𝑗1𝑚subscript𝐵subscript𝑃𝜶superscriptsubscriptproduct𝑡2𝜶subscriptsuperscript𝐖  𝑡𝑗  𝑖subscript𝐵𝑡\hat{w}\_{\boldsymbol{\alpha}}=\sum\_{i=1}^{m}\sum\_{j=1}^{m}\sum\_{B\in P\_{\boldsymbol{\alpha}}}\prod\_{t=2}^{|\boldsymbol{\alpha}|}\mathbf{W}^{t,j}\_{i,B\_{t}} |  |

where, B=[B1,B2,…,B|𝜶|]𝐵

subscript𝐵1subscript𝐵2…subscript𝐵𝜶B=[B\_{1},B\_{2},...,B\_{|\boldsymbol{\alpha}|}] is a multi-index, and P𝜶subscript𝑃𝜶P\_{\boldsymbol{\alpha}} is the set of all the permutations of the indices (1,…​1⏟α1​t​i​m​e​s,…,m,…,m⏟αm​t​i​m​e​s)subscript⏟

1…1subscript𝛼1𝑡𝑖𝑚𝑒𝑠…subscript⏟

𝑚…𝑚subscript𝛼𝑚𝑡𝑖𝑚𝑒𝑠(\underbrace{1,...1}\_{\alpha\_{1}\ times},...,\underbrace{m,...,m}\_{\alpha\_{m}\ times}).

### 3.3. Combination with Implicit Networks

As discussed in Section [2.2](#S2.SS2 "2.2. Implicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), plain DNNs learn implicit high-order feature interactions. Since CIN and plain DNNs can complement each other, an intuitive way to make the model stronger is to combine these two structures. The resulting model is very similar to the Wide&Deep or DeepFM model. The architecture is shown in Figure [5](#S3.F5 "Figure 5 ‣ 3.3. Combination with Implicit Networks ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). We name the new model eXtreme Deep Factorization Machine (xDeepFM), considering that on one hand, it includes both low-order and high-order feature interactions; on the other hand, it includes both implicit feature interactions and explicit feature interactions. Its resulting output unit becomes:

|  |  |  |  |
| --- | --- | --- | --- |
| (15) |  | y^=σ​(𝐰l​i​n​e​a​rT​𝐚+𝐰d​n​nT​𝐱d​n​nk+𝐰c​i​nT​𝐩++b)^𝑦𝜎superscriptsubscript𝐰𝑙𝑖𝑛𝑒𝑎𝑟𝑇𝐚subscriptsuperscript𝐰𝑇𝑑𝑛𝑛subscriptsuperscript𝐱𝑘𝑑𝑛𝑛subscriptsuperscript𝐰𝑇𝑐𝑖𝑛superscript𝐩𝑏\hat{y}=\sigma(\mathbf{w}\_{linear}^{T}\mathbf{a}+\mathbf{w}^{T}\_{dnn}\mathbf{x}^{k}\_{dnn}+\mathbf{w}^{T}\_{cin}\mathbf{p}^{+}+b) |  |

where σ𝜎\sigma is the sigmoid function, 𝐚𝐚\mathbf{a} is the raw features. 𝐱d​n​nk,𝐩+

subscriptsuperscript𝐱𝑘𝑑𝑛𝑛superscript𝐩\mathbf{x}^{k}\_{dnn},\mathbf{p}^{+} are the outputs of the plain DNN and CIN, respectively. 𝐰∗subscript𝐰\mathbf{w}\_{\*} and b𝑏b are learnable parameters. For binary classifications, the loss function is the log loss:

|  |  |  |  |
| --- | --- | --- | --- |
| (16) |  | ℒ=−1N​∑i=1Nyi​l​o​g​y^i+(1−yi)​l​o​g​(1−y^i)ℒ1𝑁superscriptsubscript𝑖1𝑁subscript𝑦𝑖𝑙𝑜𝑔subscript^𝑦𝑖1subscript𝑦𝑖𝑙𝑜𝑔1subscript^𝑦𝑖\mathcal{L}=-\frac{1}{N}\sum\_{i=1}^{N}y\_{i}log\hat{y}\_{i}+(1-y\_{i})log(1-\hat{y}\_{i}) |  |

where N𝑁N is the total number of training instances. The optimization process is to minimize the following objective function:

|  |  |  |  |
| --- | --- | --- | --- |
| (17) |  | 𝒥=ℒ+λ∗​‖Θ‖𝒥ℒsubscript𝜆normΘ\mathcal{J}=\mathcal{L}+\lambda\_{\*}||\Theta|| |  |

where λ∗subscript𝜆\lambda\_{\*} denotes the regularization term and ΘΘ\Theta denotes the set of parameters, including these in linear part, CIN part, and DNN part.

![Refer to caption](/html/1803.05170/assets/x7.png)


Figure 5. The architecture of xDeepFM.

#### 3.3.1. Relationship with FM and DeepFM

Suppose all fields are univalent. It’s not hard to observe from Figure [5](#S3.F5 "Figure 5 ‣ 3.3. Combination with Implicit Networks ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") that, when the depth and feature maps of the CIN part are both set to 1, xDeepFM is a generalization of DeepFM by learning the linear regression weights for the FM layer (note that in DeepFM, units of FM layer are directly linked to the output unit without any coefficients). When we further remove the DNN part, and at the same time use a constant sum filter (which simply takes the sum of inputs without any parameter learning) for the feature map, then xDeepFM is downgraded to the traditional FM model.

## 4. Experiments

In this section, we conduct extensive experiments to answer the following questions:

* •

  (Q1) How does our proposed CIN perform in high-order feature interactions learning?
* •

  (Q2) Is it necessary to combine explicit and implicit high-order feature interactions for recommender systems?
* •

  (Q3) How does the settings of networks influence the performance of xDeepFM?

We will answer these questions after presenting some fundamental experimental settings.

### 4.1. Experiment Setup

#### 4.1.1. Datasets.

We evaluate our proposed models on the following three datasets:
  
1. Criteo Dataset. It is a famous industry benchmarking dataset for developing models predicting ad click-through rate, and is publicly accessible111http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/. Given a user and the page he is visiting, the goal is to predict the probability that he will clik on a given ad.
  
2. Dianping Dataset. Dianping.com is the largest consumer review site in China. It provides diverse functions such as reviews, check-ins, and shops’ meta information (including geographical messages and shop attributes). We collect 6 months’ users check-in activities for restaurant recommendation experiments. Given a user’s profile, a restaurant’s attributes and the user’s last three visited POIs (point of interest), we want to predict the probability that he will visit the restaurant. For each restaurant in a user’s check-in instance, we sample four restaurants which are within 3 kilometers as negative instances by POI popularity.
  
3. Bing News Dataset. Bing News222https://www.bing.com/news is part of Microsoft’s Bing search engine. In order to evaluate the performance of our model in a real commercial dataset, we collect five consecutive days’ impression logs on news reading service. We use the first three days’ data for training and validation, and the next two days for testing.
  
For the Criteo dataset and the Dianping dataset, we randomly split instances by 8:1:1 for training , validation and test. The characteristics of the three datasets are summarized in Table [1](#S4.T1 "Table 1 ‣ 4.1.1. Datasets. ‣ 4.1. Experiment Setup ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems").

Table 1. Statistics of the evaluation datasets. M indicates million and K indicates thousand.

| Datasest | #instances | #fields | #features (sparse) |
| --- | --- | --- | --- |
| Criteo | 45M | 39 | 2.3M |
| Dianping | 1.2M | 18 | 230K |
| Bing News | 5M | 45 | 17K |

#### 4.1.2. Evaluation Metrics.

We use two metrics for model evaluation: AUC (Area Under the ROC curve) and Logloss (cross entropy). These two metrics evaluate the performance from two different angels: AUC measures the probability that a positive instance will be ranked higher than a randomly chosen negative one. It only takes into account the order of predicted instances and is insensitive to class imbalance problem. Logloss, in contrast, measures the distance between the predicted score and the true label for each instance. Sometimes we rely more on Logloss because we need to use the predicted probability to estimate the benefit of a ranking strategy (which is usually adjusted as CTR ×\times bid).

#### 4.1.3. Baselines.

We compare our xDeepFM with LR(logistic regression), FM, DNN (plain deep neural network), PNN (choose the better one from iPNN and oPNN) (Qu
et al., [2016](#bib.bib32)), Wide & Deep (Cheng et al., [2016](#bib.bib6)), DCN (Deep & Cross Network) (Wang
et al., [2017](#bib.bib41)) and DeepFM (Guo
et al., [2017](#bib.bib10)). As introduced and discussed in Section [2](#S2 "2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), these models are highly related to our xDeepFM and some of them are state-of-the-art models for recommender systems. Note that the focus of this paper is to learn feature interactions automatically, so we do not include any hand-crafted cross features.

#### 4.1.4. Reproducibility

We implement our method using Tensorflow333https://www.tensorflow.org/. Hyper-parameters of each model are tuned by grid-searching on the validation set, and the best settings for each model will be shown in corresponding sections. Learning rate is set to 0.001. For optimization method, we use the Adam (Kingma and Ba, [2014](#bib.bib17)) with a mini-batch size of 4096. We use a L2 regularization with λ=0.0001𝜆0.0001\lambda=0.0001 for DNN, DCN, Wide&Deep, DeepFM and xDeepFM, and use dropout 0.5 for PNN. The default setting for number of neurons per layer is: (1) 400 for DNN layers; (2) 200 for CIN layers on Criteo dataset, and 100 for CIN layers on Dianping and Bing News datasets. Since we focus on neural networks structures in this paper, we make the dimension of field embedding for all models be a fixed value of 10. We conduct experiments of different settings in parallel with 5 Tesla K80 GPUs.
The source code is available at <https://github.com/Leavingseason/xDeepFM>.

Table 2. Performance of individual models on the Criteo, Dianping, and Bing News datasets. Column Depth indicates the best network depth for each model.

|  |  |  |  |
| --- | --- | --- | --- |
| Model name | AUC | Logloss | Depth |
| Criteo | | | |
| FM | 0.7900 | 0.4592 | - |
| DNN | 0.7993 | 0.4491 | 2 |
| CrossNet | 0.7961 | 0.4508 | 3 |
| CIN | 0.8012 | 0.4493 | 3 |
| Dianping | | | |
| FM | 0.8165 | 0.3558 | - |
| DNN | 0.8318 | 0.3382 | 3 |
| CrossNet | 0.8283 | 0.3404 | 2 |
| CIN | 0.8576 | 0.3225 | 2 |
| Bing News | | | |
| FM | 0.8223 | 0.2779 | - |
| DNN | 0.8366 | 0.273 | 2 |
| CrossNet | 0.8304 | 0.2765 | 6 |
| CIN | 0.8377 | 0.2662 | 5 |




Table 3. Overall performance of different models on Criteo, Dianping and Bing News datasets. The column Depth presents the best setting for network depth with a format of (cross layers, DNN layers).

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Criteo | | | Dianping | | | Bing News | | |
| Model name | AUC | Logloss | Depth | AUC | Logloss | Depth | AUC | Logloss | Depth |
| LR | 0.7577 | 0.4854 | -,- | 0.8018 | 0.3608 | -,- | 0.7988 | 0.2950 | -,- |
| FM | 0.7900 | 0.4592 | -,- | 0.8165 | 0.3558 | -,- | 0.8223 | 0.2779 | -,- |
| DNN | 0.7993 | 0.4491 | -,2 | 0.8318 | 0.3382 | -,3 | 0.8366 | 0.2730 | -,2 |
| DCN | 0.8026 | 0.4467 | 2,2 | 0.8391 | 0.3379 | 4,3 | 0.8379 | 0.2677 | 2,2 |
| Wide&Deep | 0.8000 | 0.4490 | -,3 | 0.8361 | 0.3364 | -,2 | 0.8377 | 0.2668 | -,2 |
| PNN | 0.8038 | 0.4927 | -,2 | 0.8445 | 0.3424 | -,3 | 0.8321 | 0.2775 | -,3 |
| DeepFM | 0.8025 | 0.4468 | -,2 | 0.8481 | 0.3333 | -,2 | 0.8376 | 0.2671 | -,3 |
| xDeepFM | 0.8052 | 0.4418 | 3,2 | 0.8639 | 0.3156 | 3,3 | 0.8400 | 0.2649 | 3,2 |

### 4.2. Performance Comparison among Individual Neural Components (Q1)

We want to know how CIN performs individually. Note that FM measures 2-order feature interactions explicitly, DNN model high-order feature interactions implicitly, CrossNet tries to model high-order feature interactions with a small number of parameters (which is proven not effective in Section [2.3](#S2.SS3 "2.3. Explicit High-order Interactions ‣ 2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems")), and CIN models high-order feature interactions explicitly. There is no theoretic guarantee of the superiority of one individual model over the others, due to that it really depends on the dataset. For example, if the practical dataset does not require high-order feature interactions, FM may be the best individual model. Thus we do not have any expectation for which model will perform the best in this experiment.

Table [2](#S4.T2 "Table 2 ‣ 4.1.4. Reproducibility ‣ 4.1. Experiment Setup ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") shows the results of individual models on the three practical datasets. Surprisingly, our CIN outperform the other models consistently. On one hand, the results indicate that for practical datasets, higher-order interactions over sparse features are necessary, and this can be verified through the fact that DNN, CrossNet and CIN outperform FM significantly on all the three datasets. On the other hand, CIN is the best individual model, which demonstrates the effectiveness of CIN on modeling explicit high-order feature interactions. Note that a k𝑘k-layer CIN can model k𝑘k-degree feature interactions. It is also interesting to see that it take 5 layers for CIN to yield the best result ON the Bing News dataset.

### 4.3. Performance of Integrated Models (Q2)

xDeepFM integrates CIN and DNN into an end-to-end model. While CIN and DNN covers two distinct properties in learning feature interactions, we are interested to know whether it is indeed necessary and effective to combine them together for jointly explicit and implicit learning. Here we compare several strong baselines which are not limited to individual models, and the results are shown in Table [3](#S4.T3 "Table 3 ‣ 4.1.4. Reproducibility ‣ 4.1. Experiment Setup ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). We observe that LR is far worse than all the rest models, which demonstrates that factorization-based models are essential for measuring sparse features. Wide&Deep, DCN, DeepFM and xDeepFM are significantly better than DNN, which directly reflects that, despite their simplicity, incorporating hybrid components are important for boosting the accuracy of predictive systems. Our proposed xDeepFM achieves the best performance on all datasets, which demonstrates that combining explicit and implicit high-order feature interaction is necessary, and xDeepFM is effective in learning this class of combination. Another interesting observation is that, all the neural-based models do not require a very deep network structure for the best performance. Typical settings for the depth hyper-parameter are 2 and 3, and the best depth setting for xDeepFM is 3, which indicates that the interactions we learned are at most 4-order.

### 4.4. Hyper-Parameter Study (Q3)

We study the impact of hyper-parameters on xDeepFM in this section, including (1) the number of hidden layers; (2) the number of neurons per layer; and (3) activation functions. We conduct experiments via holding the best settings for the DNN part while varying the settings for the CIN part.

![Refer to caption](/html/1803.05170/assets/x8.png)


(a) Number of layers.

![Refer to caption](/html/1803.05170/assets/x9.png)


(b) Number of neurons per layer.

![Refer to caption](/html/1803.05170/assets/x10.png)


(c) Activation functions

Figure 6. Impact of network hyper-parameters on AUC performance.



![Refer to caption](/html/1803.05170/assets/x11.png)


(a) Number of layers.

![Refer to caption](/html/1803.05170/assets/x12.png)


(b) Number of neurons per layer.

![Refer to caption](/html/1803.05170/assets/x13.png)


(c) Activation functions

Figure 7. Impact of network hyper-parameters on Logloss performance.

Depth of Network. Figure [6(a)](#S4.F6.sf1 "In Figure 6 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") and [7(a)](#S4.F7.sf1 "In Figure 7 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") demonstrate the impact of number of hidden layers. We can observe that the performance of xDeepFM increases with the depth of network at the beginning. However, model performance degrades when the depth of network is set greater than 3. It is caused by overfitting evidenced by that we notice that the loss of training data still keeps decreasing when we add more hidden layers.
  
Number of Neurons per Layer. Adding the number of neurons per layer indicates increasing the number of feature maps in CIN. As shown in Figure [6(b)](#S4.F6.sf2 "In Figure 6 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") and [7(b)](#S4.F7.sf2 "In Figure 7 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), model performance on Bing News dataset increases steadily when we increase the number of neurons from 202020 to 200200200, while on Dianping dataset, 100100100 is a more suitable setting for the number of neurons per layer. In this experiment we fix the depth of network at 3.
  
Activation Function. Note that we exploit the identity as activation function on neurons of CIN, as shown in Eq. [6](#S3.E6 "In 3.1. Compressed Interaction Network ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"). A common practice in deep learning literature is to employ non-linear activation functions on hidden neurons. We thus compare the results of different activation functions on CIN (for neurons in DNN, we keep the activation function with relu). As shown in Figure [6(c)](#S4.F6.sf3 "In Figure 6 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") and [7(c)](#S4.F7.sf3 "In Figure 7 ‣ 4.4. Hyper-Parameter Study (Q3) ‣ 4. Experiments ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), identify function is indeed the most suitable one for neurons in CIN.

## 5. related work

### 5.1. Classical Recommender Systems

#### 5.1.1. Non-factorization Models

For web-scale recommender systems (RSs), the input features are usually sparse, categorical-continuous-mixed, and high-dimensional. Linear models, such as logistic regression with FTRL (McMahan
et al., [2013](#bib.bib28)), are widely adopted as they are easy to manage, maintain, and deploy. Because linear models lack the ability of learning feature interactions, data scientists have to spend a lot of work on engineering cross features in order to achieve better performance (Richardson
et al., [2007](#bib.bib36); Lian
et al., [2017a](#bib.bib23)). Considering that some hidden features are hard to design manually, some researchers exploit boosting decision trees to help build feature transformations (He et al., [2014](#bib.bib15); Ling
et al., [2017](#bib.bib26)).

#### 5.1.2. Factorization Models

A major downside of the aforementioned models is that they can not generalize to unseen feature interactions in the training set. Factorization Machines (Rendle, [2010](#bib.bib33)) overcome this problem via embedding each feature into a low dimension latent vector. Matrix factorization (MF) (Koren
et al., [2009](#bib.bib19)), which only considers IDs as features, can be regarded as a special kind of FM. Recommendations are made via the product of two latent vectors, thus it does not require the co-occurrence of user and item in the training set. MF is the most popular model-based collaborative filtering method in the RS literature (Srebro
et al., [2005](#bib.bib39); Koren, [2008](#bib.bib18); Lee
et al., [2013](#bib.bib21); Pan et al., [2008](#bib.bib31)). (Chen
et al., [2012](#bib.bib5); Menon and Elkan, [2010](#bib.bib29)) extend MF to leveraging side information, in which both a linear model and a MF model are included. On the other hand, for many recommender systems, only implicit feedback datasets such as users’ watching history and browsing activities are available. Thus researchers extend the factorization models to a Bayesian Personalized Ranking (BPR) framework (Rendle et al., [2009](#bib.bib34); Rendle and
Schmidt-Thieme, [2010](#bib.bib35); He and McAuley, [2016](#bib.bib12); Yuan
et al., [2016](#bib.bib45)) for implicit feedback.

### 5.2. Recommender Systems with Deep Learning

Deep learning techniques have achieved great success in computer vision (Krizhevsky
et al., [2012](#bib.bib20); He
et al., [2016](#bib.bib11)), speech recognition (Hinton
et al., [2012](#bib.bib16); Amodei
et al., [2016](#bib.bib2)) and natural language understanding (Mikolov et al., [2010](#bib.bib30); Cho et al., [2014](#bib.bib7)). As a result, an increasing number of researchers are interested in employing DNNs for recommender systems.

#### 5.2.1. Deep Learning for High-Order Interactions

To avoid manually building up high-order cross features, researchers apply DNNs on field embedding, thus patterns from categorical feature interactions can be learned automatically. Representative models include FNN (Zhang
et al., [2016a](#bib.bib47)), PNN (Qu
et al., [2016](#bib.bib32)), DeepCross (Shan
et al., [2016](#bib.bib38)), NFM (He and Chua, [2017](#bib.bib13)), DCN (Wang
et al., [2017](#bib.bib41)), Wide&Deep (Cheng et al., [2016](#bib.bib6)), and DeepFM (Guo
et al., [2017](#bib.bib10)). These models are highly related to our proposed xDeepFM. Since we have reviewed them in Section [1](#S1 "1. introduction ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems") and Section [2](#S2 "2. Preliminaries ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), we do not further discuss them in detail in this section. We have demonstrated that our proposed xDeepFM has two special properties in comparison with these models: (1) xDeepFM learns high-order feature interactions in both explicit and implicit fashions; (2) xDeepFM learns feature interactions at the vector-wise level rather than at the bit-wise level.

#### 5.2.2. Deep Learning for Elaborate Representation Learning

We include some other deep learning-based RSs in this section due to that they are less focused on learning feature interactions. Some early work employs deep learning mainly to model auxiliary information, such as visual data (He and McAuley, [2016](#bib.bib12)) and audio data (Wang and Wang, [2014](#bib.bib42)). Recently, deep neural networks are used to model the collaborative filtering (CF) in RSs. (He
et al., [2017](#bib.bib14)) proposes a Neural Collaborative Filtering (NCF) so that the inner product in MF can be replaced with an arbitrary function via a neural architecture. (Sedhain
et al., [2015](#bib.bib37); Wu
et al., [2016](#bib.bib43)) model CF base on the autoencoder paradigm, and they have empirically demonstrated that autoencoder-based CF outperforms several classical MF models. Autoencoders can be further employed for jointly modeling CF and side information with the purpose of generating better latent factors (Dong
et al., [2017](#bib.bib8); Wang
et al., [2015](#bib.bib40); Zhang
et al., [2016b](#bib.bib46)). (Elkahky
et al., [2015](#bib.bib9); Lian
et al., [2017b](#bib.bib24)) employ neural networks to jointly train multiple domains’ latent factors. (Chen
et al., [2017](#bib.bib4)) proposes the Attentive Collaborative Filtering (ACF) to learn more elaborate preference at both item-level and component-level. (Zhou et al., [2017](#bib.bib48)) shows tha traditional RSs can not capture interest diversity and local activation effectively, so they introduce a Deep Interest Network (DIN) to represent users’ diverse interests with an attentive activation mechanism.

## 6. Conclusions

In this paper, we propose a novel network named Compressed Interaction Network (CIN), which aims to learn high-order feature interactions explicitly. CIN has two special virtues: (1) it can learn certain bounded-degree feature interactions effectively; (2) it learns feature interactions at a vector-wise level. Following the spirit of several popular models, we incorporate a CIN and a DNN in an end-to-end framework, and named the resulting model eXtreme Deep Factorization Machine (xDeepFM). Thus xDeepFM can automatically learn high-order feature interactions in both explicit and implicit fashions, which is of great significance to reducing manual feature engineering work. We conduct comprehensive experiments and the results demonstrate that our xDeepFM outperforms state-of-the-art models consistently on three real-world datasets.
  
There are two directions for future work. First, currently we simply employ a sum pooling for embedding multivalent fields. We can explore the usage of the DIN mechanism (Zhou et al., [2017](#bib.bib48)) to capture the related activation according to the candidate item. Second, as discussed in section [3.2.2](#S3.SS2.SSS2 "3.2.2. Time Complexity ‣ 3.2. CIN Analysis ‣ 3. Our proposed model ‣ xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"), the time complexity of the CIN module is high. We are interested in developing a distributed version of xDeepFM which can be trained efficiently on a GPU cluster.

## Acknowledgements

The authors would like to thank the anonymous reviewers for their insightful reviews, which are very helpful on the revision of this paper. This work is supported in part by Youth Innovation Promotion Association of CAS.

## References

* (1)
* Amodei
  et al. (2016)

  Dario Amodei, Sundaram
  Ananthanarayanan, Rishita Anubhai,
  Jingliang Bai, Eric Battenberg,
  Carl Case, Jared Casper,
  Bryan Catanzaro, Qiang Cheng,
  Guoliang Chen, et al.
  2016.
  Deep speech 2: End-to-end speech recognition in
  english and mandarin. In *International Conference
  on Machine Learning*. 173–182.
* Blondel
  et al. (2016)

  Mathieu Blondel, Akinori
  Fujino, Naonori Ueda, and Masakazu
  Ishihata. 2016.
  Higher-order factorization machines. In
  *Advances in Neural Information Processing
  Systems*. 3351–3359.
* Chen
  et al. (2017)

  Jingyuan Chen, Hanwang
  Zhang, Xiangnan He, Liqiang Nie,
  Wei Liu, and Tat-Seng Chua.
  2017.
  Attentive collaborative filtering: Multimedia
  recommendation with item-and component-level attention. In
  *Proceedings of the 40th International ACM SIGIR
  conference on Research and Development in Information Retrieval*. ACM,
  335–344.
* Chen
  et al. (2012)

  Tianqi Chen, Weinan
  Zhang, Qiuxia Lu, Kailong Chen,
  Zhao Zheng, and Yong Yu.
  2012.
  SVDFeature: a toolkit for feature-based
  collaborative filtering.
  *Journal of Machine Learning Research*
  13, Dec (2012),
  3619–3622.
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
* Cho et al. (2014)

  Kyunghyun Cho, Bart
  Van Merriënboer, Caglar Gulcehre,
  Dzmitry Bahdanau, Fethi Bougares,
  Holger Schwenk, and Yoshua Bengio.
  2014.
  Learning phrase representations using RNN
  encoder-decoder for statistical machine translation.
  *arXiv preprint arXiv:1406.1078*
  (2014).
* Dong
  et al. (2017)

  Xin Dong, Lei Yu,
  Zhonghuo Wu, Yuxia Sun,
  Lingfeng Yuan, and Fangxi Zhang.
  2017.
  A Hybrid Collaborative Filtering Model with Deep
  Structure for Recommender Systems. In *AAAI*.
  1309–1315.
* Elkahky
  et al. (2015)

  Ali Mamdouh Elkahky, Yang
  Song, and Xiaodong He. 2015.
  A multi-view deep learning approach for cross
  domain user modeling in recommendation systems. In
  *Proceedings of the 24th International Conference on
  World Wide Web*. International World Wide Web Conferences Steering
  Committee, 278–288.
* Guo
  et al. (2017)

  Huifeng Guo, Ruiming
  Tang, Yunming Ye, Zhenguo Li, and
  Xiuqiang He. 2017.
  Deepfm: A factorization-machine based neural
  network for CTR prediction.
  *arXiv preprint arXiv:1703.04247*
  (2017).
* He
  et al. (2016)

  Kaiming He, Xiangyu
  Zhang, Shaoqing Ren, and Jian Sun.
  2016.
  Deep residual learning for image recognition. In
  *Proceedings of the IEEE conference on computer
  vision and pattern recognition*. 770–778.
* He and McAuley (2016)

  Ruining He and Julian
  McAuley. 2016.
  VBPR: Visual Bayesian Personalized Ranking from
  Implicit Feedback. In *AAAI*.
  144–150.
* He and Chua (2017)

  Xiangnan He and Tat-Seng
  Chua. 2017.
  Neural factorization machines for sparse predictive
  analytics. In *Proceedings of the 40th
  International ACM SIGIR conference on Research and Development in Information
  Retrieval*. ACM, 355–364.
* He
  et al. (2017)

  Xiangnan He, Lizi Liao,
  Hanwang Zhang, Liqiang Nie,
  Xia Hu, and Tat-Seng Chua.
  2017.
  Neural collaborative filtering. In
  *Proceedings of the 26th International Conference on
  World Wide Web*. International World Wide Web Conferences Steering
  Committee, 173–182.
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
* Hinton
  et al. (2012)

  Geoffrey Hinton, Li Deng,
  Dong Yu, George E Dahl,
  Abdel-rahman Mohamed, Navdeep Jaitly,
  Andrew Senior, Vincent Vanhoucke,
  Patrick Nguyen, Tara N Sainath,
  et al. 2012.
  Deep neural networks for acoustic modeling in
  speech recognition: The shared views of four research groups.
  *IEEE Signal Processing Magazine*
  29, 6 (2012),
  82–97.
* Kingma and Ba (2014)

  Diederik P Kingma and
  Jimmy Ba. 2014.
  Adam: A method for stochastic optimization.
  *arXiv preprint arXiv:1412.6980*
  (2014).
* Koren (2008)

  Yehuda Koren.
  2008.
  Factorization meets the neighborhood: a
  multifaceted collaborative filtering model. In
  *Proceedings of the 14th ACM SIGKDD international
  conference on Knowledge discovery and data mining*. ACM,
  426–434.
* Koren
  et al. (2009)

  Yehuda Koren, Robert
  Bell, and Chris Volinsky.
  2009.
  Matrix factorization techniques for recommender
  systems.
  *Computer* 42,
  8 (2009).
* Krizhevsky
  et al. (2012)

  Alex Krizhevsky, Ilya
  Sutskever, and Geoffrey E Hinton.
  2012.
  Imagenet classification with deep convolutional
  neural networks. In *Advances in neural information
  processing systems*. 1097–1105.
* Lee
  et al. (2013)

  Joonseok Lee, Seungyeon
  Kim, Guy Lebanon, and Yoram Singer.
  2013.
  Local low-rank matrix approximation. In
  *International Conference on Machine Learning*.
  82–90.
* Lian and Xie (2016)

  Jianxun Lian and Xing
  Xie. 2016.
  Cross-Device User Matching Based on Massive Browse
  Logs: The Runner-Up Solution for the 2016 CIKM Cup.
  *arXiv preprint arXiv:1610.03928*
  (2016).
* Lian
  et al. (2017a)

  Jianxun Lian, Fuzheng
  Zhang, Min Hou, Hongwei Wang,
  Xing Xie, and Guangzhong Sun.
  2017a.
  Practical Lessons for Job Recommendations in the
  Cold-Start Scenario. In *Proceedings of the
  Recommender Systems Challenge 2017* *(RecSys Challenge
  ’17)*. ACM, New York, NY, USA,
  Article 4, 6 pages.

  <https://doi.org/10.1145/3124791.3124794>
* Lian
  et al. (2017b)

  Jianxun Lian, Fuzheng
  Zhang, Xing Xie, and Guangzhong Sun.
  2017b.
  CCCFNet: a content-boosted collaborative filtering
  neural network for cross domain recommender systems. In
  *Proceedings of the 26th International Conference on
  World Wide Web Companion*. International World Wide Web Conferences Steering
  Committee, 817–818.
* Lian
  et al. (2017c)

  Jianxun Lian, Fuzheng
  Zhang, Xing Xie, and Guangzhong Sun.
  2017c.
  Restaurant Survival Analysis with Heterogeneous
  Information. In *Proceedings of the 26th
  International Conference on World Wide Web Companion*. International World
  Wide Web Conferences Steering Committee, 993–1002.
* Ling
  et al. (2017)

  Xiaoliang Ling, Weiwei
  Deng, Chen Gu, Hucheng Zhou,
  Cui Li, and Feng Sun.
  2017.
  Model Ensemble for Click Prediction in Bing Search
  Ads. In *Proceedings of the 26th International
  Conference on World Wide Web Companion*. International World Wide Web
  Conferences Steering Committee, 689–698.
* Liu et al. (2016)

  Guimei Liu, Tam T Nguyen,
  Gang Zhao, Wei Zha,
  Jianbo Yang, Jianneng Cao,
  Min Wu, Peilin Zhao, and
  Wei Chen. 2016.
  Repeat buyer prediction for e-commerce. In
  *Proceedings of the 22nd ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining*. ACM,
  155–164.
* McMahan
  et al. (2013)

  H Brendan McMahan, Gary
  Holt, David Sculley, Michael Young,
  Dietmar Ebner, Julian Grady,
  Lan Nie, Todd Phillips,
  Eugene Davydov, Daniel Golovin,
  et al. 2013.
  Ad click prediction: a view from the trenches. In
  *Proceedings of the 19th ACM SIGKDD international
  conference on Knowledge discovery and data mining*. ACM,
  1222–1230.
* Menon and Elkan (2010)

  Aditya Krishna Menon and
  Charles Elkan. 2010.
  A log-linear model with latent features for dyadic
  prediction. In *Data Mining (ICDM), 2010 IEEE 10th
  International Conference on*. IEEE, 364–373.
* Mikolov et al. (2010)

  Tomáš Mikolov,
  Martin Karafiát, Lukáš
  Burget, Jan Černockỳ, and
  Sanjeev Khudanpur. 2010.
  Recurrent neural network based language model. In
  *Eleventh Annual Conference of the International
  Speech Communication Association*.
* Pan et al. (2008)

  Rong Pan, Yunhong Zhou,
  Bin Cao, Nathan N Liu,
  Rajan Lukose, Martin Scholz, and
  Qiang Yang. 2008.
  One-class collaborative filtering. In
  *Data Mining, 2008. ICDM’08. Eighth IEEE
  International Conference on*. IEEE, 502–511.
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
* Rendle et al. (2009)

  Steffen Rendle, Christoph
  Freudenthaler, Zeno Gantner, and Lars
  Schmidt-Thieme. 2009.
  BPR: Bayesian personalized ranking from implicit
  feedback. In *Proceedings of the twenty-fifth
  conference on uncertainty in artificial intelligence*. AUAI Press,
  452–461.
* Rendle and
  Schmidt-Thieme (2010)

  Steffen Rendle and Lars
  Schmidt-Thieme. 2010.
  Pairwise interaction tensor factorization for
  personalized tag recommendation. In *Proceedings of
  the third ACM international conference on Web search and data mining*. ACM,
  81–90.
* Richardson
  et al. (2007)

  Matthew Richardson, Ewa
  Dominowska, and Robert Ragno.
  2007.
  Predicting clicks: estimating the click-through
  rate for new ads. In *Proceedings of the 16th
  international conference on World Wide Web*. ACM, 521–530.
* Sedhain
  et al. (2015)

  Suvash Sedhain,
  Aditya Krishna Menon, Scott Sanner, and
  Lexing Xie. 2015.
  Autorec: Autoencoders meet collaborative
  filtering. In *Proceedings of the 24th
  International Conference on World Wide Web*. ACM, 111–112.
* Shan
  et al. (2016)

  Ying Shan, T Ryan Hoens,
  Jian Jiao, Haijing Wang,
  Dong Yu, and JC Mao.
  2016.
  Deep crossing: Web-scale modeling without manually
  crafted combinatorial features. In *Proceedings of
  the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
  Mining*. ACM, 255–262.
* Srebro
  et al. (2005)

  Nathan Srebro, Jason
  Rennie, and Tommi S Jaakkola.
  2005.
  Maximum-margin matrix factorization. In
  *Advances in neural information processing
  systems*. 1329–1336.
* Wang
  et al. (2015)

  Hao Wang, Naiyan Wang,
  and Dit-Yan Yeung. 2015.
  Collaborative deep learning for recommender
  systems. In *Proceedings of the 21th ACM SIGKDD
  International Conference on Knowledge Discovery and Data Mining*. ACM,
  1235–1244.
* Wang
  et al. (2017)

  Ruoxi Wang, Bin Fu,
  Gang Fu, and Mingliang Wang.
  2017.
  Deep & Cross Network for Ad Click Predictions.
  *arXiv preprint arXiv:1708.05123*
  (2017).
* Wang and Wang (2014)

  Xinxi Wang and Ye
  Wang. 2014.
  Improving content-based and hybrid music
  recommendation using deep learning. In *Proceedings
  of the 22nd ACM international conference on Multimedia*. ACM,
  627–636.
* Wu
  et al. (2016)

  Yao Wu, Christopher
  DuBois, Alice X Zheng, and Martin
  Ester. 2016.
  Collaborative denoising auto-encoders for top-n
  recommender systems. In *Proceedings of the Ninth
  ACM International Conference on Web Search and Data Mining*. ACM,
  153–162.
* Xiao
  et al. (2017)

  Jun Xiao, Hao Ye,
  Xiangnan He, Hanwang Zhang,
  Fei Wu, and Tat-Seng Chua.
  2017.
  Attentional Factorization Machines: Learning the
  Weight of Feature Interactions via Attention Networks. In
  *Proceedings of the Twenty-Sixth International Joint
  Conference on Artificial Intelligence, IJCAI 2017, Melbourne, Australia,
  August 19-25, 2017*. 3119–3125.

  <https://doi.org/10.24963/ijcai.2017/435>
* Yuan
  et al. (2016)

  Fajie Yuan, Guibing Guo,
  Joemon M Jose, Long Chen,
  Haitao Yu, and Weinan Zhang.
  2016.
  Lambdafm: learning optimal ranking with
  factorization machines using lambda surrogates. In
  *Proceedings of the 25th ACM International on
  Conference on Information and Knowledge Management*. ACM,
  227–236.
* Zhang
  et al. (2016b)

  Fuzheng Zhang,
  Nicholas Jing Yuan, Defu Lian,
  Xing Xie, and Wei-Ying Ma.
  2016b.
  Collaborative knowledge base embedding for
  recommender systems. In *Proceedings of the 22nd
  ACM SIGKDD international conference on knowledge discovery and data mining*.
  ACM, 353–362.
* Zhang
  et al. (2016a)

  Weinan Zhang, Tianming
  Du, and Jun Wang. 2016a.
  Deep learning over multi-field categorical data.
  In *European conference on information retrieval*.
  Springer, 45–57.
* Zhou et al. (2017)

  Guorui Zhou, Chengru
  Song, Xiaoqiang Zhu, Xiao Ma,
  Yanghui Yan, Xingya Dai,
  Han Zhu, Junqi Jin, Han
  Li, and Kun Gai. 2017.
  Deep interest network for click-through rate
  prediction.
  *arXiv preprint arXiv:1706.06978*
  (2017).

[◄](/html/1803.05169)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1803.05170)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1803.05170)
[View original  
on arXiv](https://arxiv.org/abs/1803.05170)[►](/html/1803.05171)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 16 13:43:34 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
