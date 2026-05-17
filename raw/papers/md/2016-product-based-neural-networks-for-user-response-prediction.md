---
arxiv: '1611.00144'
authors:
- Yanru Qu
- Han Cai
- Kan Ren
- Weinan Zhang
- Yong Yu
- Ying Wen
- Jun Wang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Product-based Neural Networks for User Response Prediction
url: http://arxiv.org/abs/1611.00144v1
year: 2016
---

# Product-based Neural Networks for User Response Prediction

Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu
Shanghai Jiao Tong University
  
{kevinqu, hcai, kren, wnzhang, yyu}@apex.sjtu.edu.cn
  
Ying Wen, Jun Wang
University College London
  
{ying.wen, j.wang}@cs.ucl.ac.uk

###### Abstract

Predicting user responses, such as clicks and conversions, is of great importance and has found its usage in many Web applications including recommender systems, web search and online advertising. The data in those applications is mostly categorical and contains multiple fields; a typical representation is to transform it into a high-dimensional sparse binary feature representation via one-hot encoding.
Facing with the extreme sparsity, traditional models may limit their capacity of mining shallow patterns from the data, i.e. low-order feature combinations. Deep models like deep neural networks, on the other hand, cannot be directly applied for the high-dimensional input because of the huge feature space.
In this paper, we propose a Product-based Neural Networks (PNN) with an embedding layer to learn a distributed representation of the categorical data, a product layer to capture interactive patterns between inter-field categories, and further fully connected layers to explore high-order feature interactions.
Our experimental results on two large-scale real-world ad click datasets demonstrate that PNNs consistently outperform the state-of-the-art models on various metrics.

## I Introduction

Learning and predicting user response now plays a crucial role in many personalization tasks in information retrieval (IR), such as recommender systems, web search and online advertising. The goal of user response prediction is to estimate the probability that the user will provide a predefined positive response, e.g. clicks, purchases etc., in a given context
[[1](#bib.bib1)].
This predicted probability indicates the user’s interest on the specific item such as a news article, a commercial item or an advertising post, which influences the subsequent decision making such as document ranking [[2](#bib.bib2)] and ad bidding [[3](#bib.bib3)].

The data collection in these IR tasks is mostly in a multi-field categorical form, for example, [Weekday=Tuesday, Gender=Male, City=London], which is normally transformed into high-dimensional sparse binary features via one-hot encoding [[4](#bib.bib4)].
For example, the three field vectors with one-hot encoding are concatenated as

|  |  |  |
| --- | --- | --- |
|  | [0,1,0,0,0,0,0]⏟Weekday=Tuesday​[0,1]⏟Gender=Male​[0,0,1,0,…,0,0]⏟City=London.subscript⏟  0100000Weekday=Tuesdaysubscript⏟01Gender=Malesubscript⏟  0010…00City=London\underbrace{[0,1,0,0,0,0,0]}\_{\texttt{Weekday=Tuesday}}\underbrace{[0,1]}\_{\texttt{Gender=Male}}\underbrace{[0,0,1,0,\ldots,0,0]}\_{\texttt{City=London}}. |  |

Many machine learning models, including linear logistic regression [[5](#bib.bib5)], non-linear gradient boosting decision trees [[4](#bib.bib4)] and factorization machines [[6](#bib.bib6)], have been proposed to work on such high-dimensional sparse binary features and produce high quality user response predictions.
However, these models highly depend on feature engineering in order to capture high-order latent patterns [[7](#bib.bib7)].

Recently, deep neural networks (DNNs) [[8](#bib.bib8)] have shown great capability in classification and regression tasks, including computer vision [[9](#bib.bib9)], speech recognition [[10](#bib.bib10)] and natural language processing [[11](#bib.bib11)]. It is promising to adopt DNNs in user response prediction since DNNs could automatically learn more expressive feature representations and deliver better prediction performance.
In order to improve the multi-field categorical data interaction, [[12](#bib.bib12)] presented an embedding methodology based on pre-training of a factorization machine. Based on the concatenated embedding vectors, multi-layer perceptrons (MLPs) were built to explore feature interactions. However, the quality of embedding initialization is largely limited by the factorization machine.
More importantly, the “add” operations of the perceptron layer might not be useful to explore the interactions of categorical data in multiple fields. Previous work [[1](#bib.bib1), [6](#bib.bib6)] has shown that local dependencies between features from different fields can be effectively explored by feature vector “product” operations instead of “add” operations.

To utilize the learning ability of neural networks and mine the latent patterns of data in a more effective way than MLPs, in this paper we propose Product-based Neural Network (PNN) which (i) starts from an embedding layer without pre-training as used in [[12](#bib.bib12)], and (ii) builds a product layer based on the embedded feature vectors to model the inter-field feature interactions, and (iii) further distills the high-order feature patterns with fully connected MLPs.
We present two types of PNNs, with inner and outer product operations in the product layer, to efficiently model the interactive patterns.

We take CTR estimation in online advertising as the working example to explore the learning ability of our PNN model. The extensive experimental results on two large-scale real-world datasets demonstrate the consistent superiority of our model over state-of-the-art user response prediction models [[6](#bib.bib6), [13](#bib.bib13), [12](#bib.bib12)] on various metrics.

## II Related Work

The response prediction problem is normally formulated as a binary classification problem with prediction likelihood or cross entropy as the training objective [[14](#bib.bib14)].
Area under ROC Curve (AUC) and Relative Information Gain (RIG) are common evaluation metrics for response prediction accuracy [[15](#bib.bib15)].
From the modeling perspective, linear logistic regression (LR) [[5](#bib.bib5), [16](#bib.bib16)] and non-linear gradient boosting decision trees (GBDT) [[4](#bib.bib4)] and factorization machines (FM) [[6](#bib.bib6)] are widely used in industrial applications. However, these models are limited in mining high-order latent patterns or learning quality feature representations.

Deep learning is able to explore high-order latent patterns as well as generalizing expressive data representations [[11](#bib.bib11)].
The input data of DNNs are usually dense real vectors, while the solution of multi-field categorical data has not been well studied. Factorization-machine supported neural networks (FNN) was proposed in [[12](#bib.bib12)] to learn embedding vectors of categorical data via pre-trained FM.
Convolutional Click Prediction Model (CCPM) was proposed in [[13](#bib.bib13)] to predict ad click by convolutional neural networks (CNN). However, in CCPM the convolutions are only performed on the neighbor fields in a certain alignment, which fails to model the full interactions among non-neighbor features.
Recurrent neural networks (RNN) was leveraged to model the user queries as a series of user context to predict the ad click behavior [[17](#bib.bib17)].
Product unit neural network (PUNN) [[18](#bib.bib18)] was proposed to build high-order combinations of the inputs. However, neither can PUNN learn local dependencies, nor produce bounded outputs to fit the response rate.

In this paper, we demonstrate the way our PNN models learn local dependencies and high-order feature interactions.

## III Deep Learning for CTR Estimation

We take CTR estimation in online advertising [[14](#bib.bib14)] as a working example to formulate our model and explore the performance on various metrics. The task is to build a prediction model to estimate the probability of a user clicking a specific ad in a given context.

Each data sample consists of multiple fields of categorical data such as user information (City, Hour, etc.), publisher information (Domain, Ad slot, etc.) and ad information (Ad creative ID, Campaign ID, etc.) [[19](#bib.bib19)]. All the information is represented as a multi-field categorical feature vector, where each field (e.g. City) is one-hot encoded as discussed in Section [I](#S1 "I Introduction ‣ Product-based Neural Networks for User Response Prediction").
Such a field-wise one-hot encoding representation results in curse of dimensionality and enormous sparsity [[12](#bib.bib12)]. Besides, there exist local dependencies and hierarchical structures among fields [[1](#bib.bib1)].

Thus we are seeking a DNN model to capture high-order latent patterns in multi-field categorical data. And we come up with the idea of product layers to explore feature interactions automatically. In FM, feature interaction is defined as the inner product of two feature vectors [[20](#bib.bib20)].

The proposed deep learning model is named as Product-based Neural Network (PNN). In this section, we present PNN model in detail and discuss two variants of this model, namely Inner Product-based Neural Network (IPNN), which has an inner product layer, and Outer Product-based Neural Network (OPNN) which uses an outer product expression.

### III-A Product-based Neural Network

!(/html/1611.00144/assets/x1.png)

Figure 1: Product-based Neural Network Architecture.

The architecture of the PNN model is illustrated in Figure [1](#S3.F1 "Figure 1 ‣ III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction").
From a top-down perspective, the output of PNN is a real number y^∈(0,1)^𝑦01\hat{y}\in(0,1) as the predicted CTR:

|  |  |  |  |
| --- | --- | --- | --- |
|  | y^=σ​(𝑾3​𝒍2+b3),^𝑦𝜎subscript𝑾3subscript𝒍2subscript𝑏3\hat{y}=\sigma(\bm{W}\_{3}\bm{l}\_{2}+b\_{3}), |  | (1) |

where 𝑾3∈ℝ1×D2subscript𝑾3superscriptℝ1subscript𝐷2\bm{W}\_{3}\in\mathbb{R}^{1\times D\_{2}} and b3∈ℝsubscript𝑏3ℝb\_{3}\in\mathbb{R} are the parameters of the output layer, 𝒍2∈ℝD2subscript𝒍2superscriptℝsubscript𝐷2\bm{l}\_{2}\in\mathbb{R}^{D\_{2}} is the output of the second hidden layer, and σ​(x)𝜎𝑥\sigma(x) is the sigmoid activation function: σ​(x)=1/(1+e−x)𝜎𝑥11superscript𝑒𝑥\sigma(x)=1/(1+e^{-x}). And we use Disubscript𝐷𝑖D\_{i} to represent the dimension of the i𝑖i-th hidden layer.

The output 𝒍2subscript𝒍2\bm{l}\_{2} of the second hidden layer is constructed as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒍2=relu​(𝑾2​𝒍1+𝒃2),subscript𝒍2relusubscript𝑾2subscript𝒍1subscript𝒃2\bm{l}\_{2}=\text{relu}(\bm{W}\_{2}\bm{l}\_{1}+\bm{b}\_{2}), |  | (2) |

where 𝒍1∈ℝD1subscript𝒍1superscriptℝsubscript𝐷1\bm{l}\_{1}\in\mathbb{R}^{D\_{1}} is the output of the first hidden layer. The rectified linear unit (relu), defined as relu​(x)=max⁡(0,x)relu𝑥0𝑥\text{relu}(x)=\max(0,x), is chosen as the activation function for hidden layer output since it has outstanding performance and efficient computation.

The first hidden layer is fully connected with the product layer. The inputs to it consist of linear signals 𝒍zsubscript𝒍𝑧\bm{l}\_{z} and quadratic signals 𝒍psubscript𝒍𝑝\bm{l}\_{p}. With respect to 𝒍zsubscript𝒍𝑧\bm{l}\_{z} and 𝒍psubscript𝒍𝑝\bm{l}\_{p} inputs, separately, the formulation of 𝒍1subscript𝒍1\bm{l}\_{1} is:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒍1subscript𝒍1\displaystyle\bm{l}\_{1} | =relu​(𝒍z+𝒍p+𝒃1),absentrelusubscript𝒍𝑧subscript𝒍𝑝subscript𝒃1\displaystyle=\text{relu}(\bm{l}\_{z}+\bm{l}\_{p}+\bm{b}\_{1}), |  | (3) |

where all 𝒍zsubscript𝒍𝑧\bm{l}\_{z}, 𝒍psubscript𝒍𝑝\bm{l}\_{p} and the bias vector 𝒃1∈ℝD1subscript𝒃1superscriptℝsubscript𝐷1\bm{b}\_{1}\in\mathbb{R}^{D\_{1}}.

Then, let us define the operation of tensor inner product:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝑨𝑨\displaystyle\bm{A} | ⊙𝑩≜∑i,j𝑨i,j𝑩i,j,\displaystyle\odot\bm{B}\triangleq\sum\_{i,j}\bm{A}\_{i,j}\bm{B}\_{i,j}, |  | (4) |

where firstly element-wise multiplication is applied to 𝑨,𝑩

𝑨𝑩\bm{A},\bm{B}, then the multiplication result is summed up to a scalar. After that,
𝒍zsubscript𝒍𝑧\bm{l}\_{z} and 𝒍psubscript𝒍𝑝\bm{l}\_{p} are calculated through 𝒛𝒛\bm{z} and 𝒑𝒑\bm{p}, respectively:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒍zsubscript𝒍𝑧\displaystyle\bm{l}\_{z} | =(lz1,lz2,…,lzn,…,lzD1),lzn=𝑾zn⊙𝒛formulae-sequenceabsentmatrix  superscriptsubscript𝑙𝑧1superscriptsubscript𝑙𝑧2…superscriptsubscript𝑙𝑧𝑛…superscriptsubscript𝑙𝑧subscript𝐷1superscriptsubscript𝑙𝑧𝑛direct-productsuperscriptsubscript𝑾𝑧𝑛𝒛\displaystyle=\begin{pmatrix}l\_{z}^{1},l\_{z}^{2},\ldots,l\_{z}^{n},\ldots,l\_{z}^{D\_{1}}\end{pmatrix},\qquad l\_{z}^{n}=\bm{W}\_{z}^{n}\odot\bm{z} |  | (5) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒍psubscript𝒍𝑝\displaystyle\bm{l}\_{p} | =(lp1,lp2,…,lpn,…,lpD1),lpn=𝑾pn⊙𝒑formulae-sequenceabsentmatrix  superscriptsubscript𝑙𝑝1superscriptsubscript𝑙𝑝2…superscriptsubscript𝑙𝑝𝑛…superscriptsubscript𝑙𝑝subscript𝐷1superscriptsubscript𝑙𝑝𝑛direct-productsuperscriptsubscript𝑾𝑝𝑛𝒑\displaystyle=\begin{pmatrix}l\_{p}^{1},l\_{p}^{2},\ldots,l\_{p}^{n},\ldots,l\_{p}^{D\_{1}}\end{pmatrix},\qquad l\_{p}^{n}=\bm{W}\_{p}^{n}\odot\bm{p} |  |

where 𝑾znsuperscriptsubscript𝑾𝑧𝑛\bm{W}\_{z}^{n} and 𝑾pnsuperscriptsubscript𝑾𝑝𝑛\bm{W}\_{p}^{n} are the weights in the product layer, and their shapes are determined by 𝒛𝒛\bm{z} and 𝒑𝒑\bm{p} respectively.

By introducing a “1” constant signal, the product layer can not only generate the quadratic signals 𝒑𝒑\bm{p}, but also maintaining the linear signals 𝒛𝒛\bm{z}, as illustrated in Figure [1](#S3.F1 "Figure 1 ‣ III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction"). More specifically,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒛𝒛\displaystyle\bm{z} | =(𝒛1,𝒛2,…,𝒛N)≜(𝒇1,𝒇2,…,𝒇N),absentmatrix  subscript𝒛1subscript𝒛2…subscript𝒛𝑁≜matrix  subscript𝒇1subscript𝒇2…subscript𝒇𝑁\displaystyle=\begin{pmatrix}\bm{z}\_{1},\bm{z}\_{2},\ldots,\bm{z}\_{N}\end{pmatrix}\triangleq\begin{pmatrix}\bm{f}\_{1},\bm{f}\_{2},\ldots,\bm{f}\_{N}\end{pmatrix}, |  | (6) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒑𝒑\displaystyle\bm{p} | ={𝒑i,j},i=1​…​N,j=1​…​N,formulae-sequenceabsentsubscript𝒑  𝑖𝑗formulae-sequence𝑖1…𝑁𝑗1…𝑁\displaystyle=\{\bm{p}\_{i,j}\},i=1...N,j=1...N, |  | (7) |

where 𝒇i∈ℝMsubscript𝒇𝑖superscriptℝ𝑀\bm{f}\_{i}\in\mathbb{R}^{M} is the embedding vector for field i𝑖i. 𝒑i,j=g​(𝒇i,𝒇j)subscript𝒑

𝑖𝑗𝑔subscript𝒇𝑖subscript𝒇𝑗\bm{p}\_{i,j}=g(\bm{f}\_{i},\bm{f}\_{j}) defines the pairwise feature interaction.
Our PNN model can have different implementations by designing different operation for g𝑔g. In this paper, we propose two variants of PNN, namely IPNN and OPNN, as will be discussed later.

The embedding vector 𝒇isubscript𝒇𝑖\bm{f}\_{i} of field i𝑖i, is the output of the embedding layer:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒇i=𝑾0i𝒙[starti:endi],\bm{f}\_{i}=\bm{W}\_{0}^{i}~{}\bm{x}[\text{start}\_{i}:\text{end}\_{i}], |  | (8) |

where 𝒙𝒙\bm{x} is the input feature vector containing multiple fields, and 𝒙[starti:endi]\bm{x}[\text{start}\_{i}:\text{end}\_{i}] represents the one-hot encoded vector for field i𝑖i. 𝑾0subscript𝑾0\bm{W}\_{0} represents the parameters of the embedding layer, and 𝑾0i∈ℝM×(endi−starti+1)superscriptsubscript𝑾0𝑖superscriptℝ𝑀subscriptend𝑖subscriptstart𝑖1\bm{W}\_{0}^{i}\in\mathbb{R}^{M\times(\text{end}\_{i}-\text{start}\_{i}+1)} is fully connected with field i𝑖i.

Finally, supervised training is applied to minimize the log loss, which is a widely used objective function capturing divergence between two probability distributions:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(y,y^)=−y​log⁡y^−(1−y)​log⁡(1−y^),𝐿𝑦^𝑦𝑦^𝑦1𝑦1^𝑦L(y,\hat{y})=-y\log\hat{y}-(1-y)\log(1-\hat{y}), |  | (9) |

where y𝑦y is the ground truth (1 for click, 0 for non-click), and y^^𝑦\hat{y} is the predicted CTR of our model as in Eq. ([1](#S3.E1 "In III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction")).

### III-B Inner Product-based Neural Network

In this section, we demonstrate the Inner Product-based Neural Network (IPNN). In IPNN, we firstly define the pairwise feature interaction as vector inner product
: g​(𝒇i,𝒇j)=⟨𝒇i,𝒇j⟩𝑔subscript𝒇𝑖subscript𝒇𝑗

subscript𝒇𝑖subscript𝒇𝑗g(\bm{f}\_{i},\bm{f}\_{j})=\langle\bm{f}\_{i},\bm{f}\_{j}\rangle.

With the constant signal “1”, the linear information 𝒛𝒛\bm{z} is preserved as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒍zn=𝑾zn⊙𝒛=∑i=1N∑j=1M(𝑾zn)i,j​𝒛i,j.superscriptsubscript𝒍𝑧𝑛direct-productsuperscriptsubscript𝑾𝑧𝑛𝒛superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑀subscriptsuperscriptsubscript𝑾𝑧𝑛  𝑖𝑗subscript𝒛  𝑖𝑗\bm{l}\_{z}^{n}=\bm{W}\_{z}^{n}\odot\bm{z}=\sum\_{i=1}^{N}{\sum\_{j=1}^{M}{(\bm{W}\_{z}^{n})\_{i,j}\bm{z}\_{i,j}}}. |  | (10) |

As for the quadratic information 𝒑𝒑\bm{p}, the pairwise inner product terms of g​(𝒇i,𝒇j)𝑔subscript𝒇𝑖subscript𝒇𝑗g(\bm{f}\_{i},\bm{f}\_{j}) form a square matrix 𝒑∈ℝN×N𝒑superscriptℝ𝑁𝑁\bm{p}\in\mathbb{R}^{N\times N}.
Recalling the definition of 𝒍psubscript𝒍𝑝\bm{l}\_{p} in Eq. ([5](#S3.E5 "In III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction")), lpn=∑i=1N∑j=1N(𝑾pn)i,j​𝒑i,jsuperscriptsubscript𝑙𝑝𝑛superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑁subscriptsuperscriptsubscript𝑾𝑝𝑛

𝑖𝑗subscript𝒑

𝑖𝑗l\_{p}^{n}=\sum\_{i=1}^{N}\sum\_{j=1}^{N}(\bm{W}\_{p}^{n})\_{i,j}\bm{p}\_{i,j} and the commutative law in vector inner product, 𝒑𝒑\bm{p} and 𝑾pnsuperscriptsubscript𝑾𝑝𝑛\bm{W}\_{p}^{n} should be symmetric.

Such pairwise connection expands the capacity of the neural network, but also enormously increases the complexity.
In this case, the formulation of 𝒍1subscript𝒍1\bm{l}\_{1}, described in Eq. ([3](#S3.E3 "In III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction")), has the space complexity of O​(D1​N​(M+N))𝑂subscript𝐷1𝑁𝑀𝑁O(D\_{1}N(M+N)), and the time complexity of O​(N2​(D1+M))𝑂superscript𝑁2subscript𝐷1𝑀O(N^{2}(D\_{1}+M)), where D1subscript𝐷1D\_{1} and M𝑀M are the hyper-parameters about network architecture, N𝑁N is the number of input fields.
Inspired by FM [[20](#bib.bib20)], we come up with the idea of matrix factorization to reduce complexity.

By introducing the assumption that 𝑾pn=𝜽n​𝜽nTsuperscriptsubscript𝑾𝑝𝑛superscript𝜽𝑛superscriptsuperscript𝜽𝑛𝑇\bm{W}\_{p}^{n}=\bm{\theta}^{n}{\bm{\theta}^{n}}^{T}, where 𝜽n∈ℝNsuperscript𝜽𝑛superscriptℝ𝑁\bm{\theta}^{n}\in\mathbb{R}^{N}, we can simplify 𝒍1subscript𝒍1\bm{l}\_{1}’s formulation as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑾pn⊙𝒑=∑i=1N∑j=1Nθin​θjn​⟨𝒇i,𝒇j⟩=⟨∑i=1N𝜹in,∑i=1N𝜹in⟩direct-productsuperscriptsubscript𝑾𝑝𝑛𝒑superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑁subscriptsuperscript𝜃𝑛𝑖subscriptsuperscript𝜃𝑛𝑗  subscript𝒇𝑖subscript𝒇𝑗  superscriptsubscript𝑖1𝑁subscriptsuperscript𝜹𝑛𝑖superscriptsubscript𝑖1𝑁subscriptsuperscript𝜹𝑛𝑖\bm{W}\_{p}^{n}\odot\bm{p}=\sum\_{i=1}^{N}{\sum\_{j=1}^{N}\theta^{n}\_{i}\theta^{n}\_{j}\langle\bm{f}\_{i},\bm{f}\_{j}\rangle}=\langle\sum\_{i=1}^{N}\bm{\delta}^{n}\_{i},\sum\_{i=1}^{N}\bm{\delta}^{n}\_{i}\rangle |  | (11) |

where, for convenience, we use 𝜹in∈ℝMsubscriptsuperscript𝜹𝑛𝑖superscriptℝ𝑀\bm{\delta}^{n}\_{i}\in\mathbb{R}^{M} to denote a feature vector 𝒇isubscript𝒇𝑖\bm{f}\_{i} weighted by θinsubscriptsuperscript𝜃𝑛𝑖\theta^{n}\_{i}, i.e. 𝜹in=θin​𝒇isubscriptsuperscript𝜹𝑛𝑖subscriptsuperscript𝜃𝑛𝑖subscript𝒇𝑖\bm{\delta}^{n}\_{i}=\theta^{n}\_{i}\bm{f}\_{i}. And we also have 𝜹n=(𝜹1n,𝜹2n,…,𝜹in,…,𝜹Nn)∈ℝN×Msuperscript𝜹𝑛matrix

subscriptsuperscript𝜹𝑛1subscriptsuperscript𝜹𝑛2…subscriptsuperscript𝜹𝑛𝑖…subscriptsuperscript𝜹𝑛𝑁superscriptℝ𝑁𝑀\bm{\delta}^{n}=\begin{pmatrix}\bm{\delta}^{n}\_{1},\bm{\delta}^{n}\_{2},\ldots,\bm{\delta}^{n}\_{i},\ldots,\bm{\delta}^{n}\_{N}\end{pmatrix}\in\mathbb{R}^{N\times M}.

With the first order decomposition on n𝑛n-th single node, we give the 𝒍psubscript𝒍𝑝\bm{l}\_{p} complete form:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒍p=(‖∑i𝜹i1‖,…,‖∑i𝜹in‖,…,‖∑i𝜹iD1‖).subscript𝒍𝑝delimited-∥∥subscript𝑖superscriptsubscript𝜹𝑖1…delimited-∥∥subscript𝑖superscriptsubscript𝜹𝑖𝑛…delimited-∥∥subscript𝑖superscriptsubscript𝜹𝑖subscript𝐷1\begin{split}\bm{l}\_{p}=\Big{(}\|\sum\_{i}\bm{\delta}\_{i}^{1}\|,\ldots,\|\sum\_{i}\bm{\delta}\_{i}^{n}\|,\ldots,\|\sum\_{i}\bm{\delta}\_{i}^{D\_{1}}\|\Big{)}.\end{split} |  | (12) |

By reduction of 𝒍psubscript𝒍𝑝\bm{l}\_{p} in Eq. ([12](#S3.E12 "In III-B Inner Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction")), the space complexity of 𝒍1subscript𝒍1\bm{l}\_{1} becomes O​(D1​M​N)𝑂subscript𝐷1𝑀𝑁O(D\_{1}MN), and the time complexity is also O​(D1​M​N)𝑂subscript𝐷1𝑀𝑁O(D\_{1}MN). In general, 𝒍1subscript𝒍1\bm{l}\_{1} complexity is reduced from quadratic to linear with respect to N𝑁N. This well-formed equation makes reusable for some intermediate results. Moreover, matrix operations are easily accelerated in practice with GPUs.

More generally, we discuss K𝐾K-order decomposition of 𝑾pnsuperscriptsubscript𝑾𝑝𝑛\bm{W}\_{p}^{n} at the end of this section. We should point out that 𝑾pn=𝜽n​𝜽nTsuperscriptsubscript𝑾𝑝𝑛subscript𝜽𝑛superscriptsubscript𝜽𝑛𝑇\bm{W}\_{p}^{n}=\bm{\theta}\_{n}\bm{\theta}\_{n}^{T} is only the first order decomposition with a strong assumption. The general matrix decomposition method can be derived that:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝑾pn⊙𝒑=∑i=1N∑j=1N⟨𝜽ni,𝜽nj⟩​⟨𝒇i,𝒇j⟩.direct-productsuperscriptsubscript𝑾𝑝𝑛𝒑superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑁  superscriptsubscript𝜽𝑛𝑖superscriptsubscript𝜽𝑛𝑗  subscript𝒇𝑖subscript𝒇𝑗\bm{W}\_{p}^{n}\odot\bm{p}=\sum\_{i=1}^{N}{\sum\_{j=1}^{N}\langle\bm{\theta}\_{n}^{i},\bm{\theta}\_{n}^{j}\rangle\langle\bm{f}\_{i},\bm{f}\_{j}\rangle}. |  | (13) |

In this case, 𝜽ni∈ℝKsuperscriptsubscript𝜽𝑛𝑖superscriptℝ𝐾\bm{\theta}\_{n}^{i}\in\mathbb{R}^{K}. This general decomposition is more expressive with weaker assumptions, but also leading to K𝐾K times model complexity.

### III-C Outer Product-based Neural Network

Vector inner product takes a pair of vectors as input and outputs a scalar. Different from that, vector outer product takes a pair of vectors and produces a matrix. IPNN defines feature interaction by vector inner product, while in this section, we discuss the Outer Product-based Neural Network (OPNN).

The only difference between IPNN and OPNN is the quadratic term 𝒑𝒑\bm{p}.
In OPNN, we define feature interaction as g​(𝒇i,𝒇j)=𝒇i​𝒇jT𝑔subscript𝒇𝑖subscript𝒇𝑗subscript𝒇𝑖superscriptsubscript𝒇𝑗𝑇g(\bm{f}\_{i},\bm{f}\_{j})=\bm{f}\_{i}\bm{f}\_{j}^{T}. Thus for every element in 𝒑𝒑\bm{p}, 𝒑i,j∈ℝM×Msubscript𝒑

𝑖𝑗superscriptℝ𝑀𝑀\bm{p}\_{i,j}\in\mathbb{R}^{M\times M} is a square matrix.

For calculating 𝒍1subscript𝒍1\bm{l}\_{1}, the space complexity is O​(D1​M2​N2)𝑂subscript𝐷1superscript𝑀2superscript𝑁2O(D\_{1}M^{2}N^{2}) , and the time complexity is also O​(D1​M2​N2)𝑂subscript𝐷1superscript𝑀2superscript𝑁2O(D\_{1}M^{2}N^{2}). Recall that D1subscript𝐷1D\_{1} and M𝑀M are the hyper-parameters of the network architecture, and N𝑁N is the number of the input fields, this implementation is expensive in practice. To reduce the complexity, we propose the idea of *superposition*.

By element-wise superposition, we can reduce the complexity by a large step. Specifically, we re-define 𝒑𝒑\bm{p} formulation as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒑=∑i=1N∑j=1N𝒇i​𝒇jT=𝒇Σ​(𝒇Σ)T,𝒇Σ=∑i=1N𝒇i,formulae-sequence𝒑superscriptsubscript𝑖1𝑁superscriptsubscript𝑗1𝑁subscript𝒇𝑖superscriptsubscript𝒇𝑗𝑇subscript𝒇Σsuperscriptsubscript𝒇Σ𝑇subscript𝒇Σsuperscriptsubscript𝑖1𝑁subscript𝒇𝑖\bm{p}=\sum\_{i=1}^{N}{\sum\_{j=1}^{N}{\bm{f}\_{i}\bm{f}\_{j}^{T}}}=\bm{f}\_{\Sigma}(\bm{f}\_{\Sigma})^{T},\quad\bm{f}\_{\Sigma}=\sum\_{i=1}^{N}{\bm{f}\_{i}}, |  | (14) |

where 𝒑∈ℝM×M𝒑superscriptℝ𝑀𝑀\bm{p}\in\mathbb{R}^{M\times M} becomes symmetric, thus 𝑾pnsuperscriptsubscript𝑾𝑝𝑛\bm{W}\_{p}^{n} should also be symmetric.
Recall Eq. ([5](#S3.E5 "In III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction")) that 𝑾p∈ℝD1×M×Msubscript𝑾𝑝superscriptℝsubscript𝐷1𝑀𝑀\bm{W}\_{p}\in\mathbb{R}^{D\_{1}\times M\times M}. In this case, the space complexity of 𝒍1subscript𝒍1\bm{l}\_{1} becomes O​(D1​M​(M+N))𝑂subscript𝐷1𝑀𝑀𝑁O(D\_{1}M(M+N)), and the time complexity is also O​(D1​M​(M+N))𝑂subscript𝐷1𝑀𝑀𝑁O(D\_{1}M(M+N)).

### III-D Discussions

Compared with FNN [[12](#bib.bib12)], PNN has a product layer. If removing 𝒍𝒑subscript𝒍𝒑\bm{l}\_{\bm{p}} part of the product layer, PNN is identical to FNN. With the inner product operator, PNN is quite similar with FM [[20](#bib.bib20)]: if there is no hidden layer and the output layer is simply summing up with uniform weight, PNN is identical to FM. Inspired by Net2Net [[21](#bib.bib21)], we can firstly train a part of PNN (e.g., the FNN or FM part) as the initialization, and then start to let the back propagation go over the whole net. The resulted PNN should at least be as good as FNN or FM.

In general, PNN uses product layers to explore feature interactions. Vector products can be viewed as a series of addition/multiplication operations. Inner product and outer product are just two implementations. In fact, we can define more general or complicated product layers, gaining PNN better capability in exploration of feature interactions.

Analogous to electronic circuit, addition acts like “OR” gate while multiplication acting like “AND” gate, and the product layer seems to learn rules other than features. Reviewing the scenario of computer vision, while pixels in images are real-world raw features, categorical data in web applications are artificial features with high levels and rich meanings. Logic is a powerful tool in dealing with concepts, domains and relationships. Thus we believe that introducing product operations in neural networks will improve networks’ ability for modeling multi-field categorical data.

## IV Experiments

In this section, we present our experiments in detail, including datasets, data processing, experimental setup, model comparison, and the corresponding analysis111We release the repeatable experiment code on GitHub: https://github.com/Atomu2014/product-nets. In our experiments, PNN models outperform major state-of-the-art models in the CTR estimation task on two real-world datasets.

### IV-A Datasets

#### IV-A1 Criteo

Criteo 1TB click log222Criteo terabyte dataset download link: http://labs.criteo.com/downloads/download-terabyte-click-logs/. is a famous ad tech industry benchmarking dataset.
We select 7 consecutive days of samples for training, and the next 1 day for evaluation. Because of the enormous data volume and high bias, we apply negative down-sampling on this dataset.
Define the down-sampling ratio as w𝑤w, the predicted CTR as p𝑝p, the re-calibrated CTR q𝑞q should be q=p/(p+1−pw)𝑞𝑝𝑝1𝑝𝑤q=p/(p+\frac{1-p}{w}) [[4](#bib.bib4)].
After down-sampling and feature mapping, we get a dataset, which comprises 79.38M instances with 1.64M feature dimensions.

#### IV-A2 iPinYou

The iPinYou dataset333iPinYou dataset download link: http://data.computational-advertising.org. We only use the data from season 2 and 3 because of the same data schema. is another real-world dataset for ad click logs over 10 days.
After one-hot encoding, we get a dataset containing 19.50M instances with 937.67K input dimensions.
We keep the original train/test splitting scheme, where for each advertiser the last 3-day data are used as the test dataset while the rest as the training dataset.

### IV-B Model Comparison

We compare 7 models in our experiments, which are implemented with TensorFlow444TensorFlow: https://www.tensorflow.org/, and trained with Stochastic Gradient Descent (SGD).

LR: LR is the most widely used linear model in industrial applications [[22](#bib.bib22)]. It is easy to implement and fast to train, however, unable to capture non-linear information.

FM: FM has many successful applications in recommender systems and user response prediction tasks [[20](#bib.bib20)]. FM explores feature interactions, which is effective on sparse data.

FNN: FNN is proposed in [[12](#bib.bib12)], being able to capture high-order latent patterns of multi-field categorical data.

CCPM: CCPM is a convolutional model for click prediction [[13](#bib.bib13)]. This model learns local-global features efficiently. However, CCPM highly relies on feature alignment, and is lack of interpretation.

IPNN: PNN with inner product layer [III-B](#S3.SS2 "III-B Inner Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction").

OPNN: PNN with outer product layer [III-C](#S3.SS3 "III-C Outer Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction").

PNN\*: This model has a product layer, which is a concatenation of inner product and outer product.

Additionally, in order to prevent over-fitting, the popular L2 regularization term is added to the loss function L​(y,y^)𝐿𝑦^𝑦L(y,\hat{y}) when training LR and FM.
And we also employ dropout as a regularization method to prevent over-fitting when training neural networks.

### IV-C Evaluation Metrics

Four evaluation metrics are tested in our experiments. The two major metrics are:

AUC: Area under ROC curve is a widely used metric in evaluating classification problems. Besides, some work validates AUC as a good measurement in CTR estimation [[15](#bib.bib15)].

RIG: Relative Information Gain, R​I​G=1−N​E𝑅𝐼𝐺1𝑁𝐸RIG=1-NE, where NE is the Normalized Cross Entropy [[4](#bib.bib4)].

Besides, we also employ Log Loss (Eq. ([9](#S3.E9 "In III-A Product-based Neural Network ‣ III Deep Learning for CTR Estimation ‣ Product-based Neural Networks for User Response Prediction"))) and root mean square error (RMSE) as our additional evaluation metrics.

### IV-D Performance Comparison

TABLE I: Overall Performance on the Criteo Dataset.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | AUC | Log Loss | RMSE | RIG |
| LR | 71.48% | 0.1334 | 9.362e-4 | 6.680e-2 |
| FM | 72.20% | 0.1324 | 9.284e-4 | 7.436e-2 |
| FNN | 75.66% | 0.1283 | 9.030e-4 | 1.024e-1 |
| CCPM | 76.71% | 0.1269 | 8.938e-4 | 1.124e-1 |
| IPNN | 77.79% | 0.1252 | 8.803e-4 | 1.243e-1 |
| OPNN | 77.54% | 0.1257 | 8.846e-4 | 1.211e-1 |
| PNN\* | 77.00% | 0.1270 | 8.988e-4 | 1.118e-1 |

TABLE II: Overall Performance on the iPinYou Dataset.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | AUC | Log Loss | RMSE | RIG |
| LR | 73.43% | 5.581e-3 | 5.350e-07 | 7.353e-2 |
| FM | 75.52% | 5.504e-3 | 5.343e-07 | 8.635e-2 |
| FNN | 76.19% | 5.443e-3 | 5.285e-07 | 9.635e-2 |
| CCPM | 76.38% | 5.522e-3 | 5.343e-07 | 8.335e-2 |
| IPNN | 79.14% | 5.195e-3 | 4.851e-07 | 1.376e-1 |
| OPNN | 81.74% | 5.211e-3 | 5.293e-07 | 1.349e-1 |
| PNN\* | 76.61% | 4.975e-3 | 4.819e-07 | 1.740e-1 |

TABLE III: P-values under the Log Loss Metric.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | LR | FM | FNN | CCPM |
| IPNN | <10−6absentsuperscript106<10^{-6} | <10−6absentsuperscript106<10^{-6} | <10−6absentsuperscript106<10^{-6} | <10−6absentsuperscript106<10^{-6} |
| OPNN | <10−6absentsuperscript106<10^{-6} | <10−5absentsuperscript105<10^{-5} | <10−6absentsuperscript106<10^{-6} | <10−6absentsuperscript106<10^{-6} |

Table [III](#S4.T3 "TABLE III ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") and [III](#S4.T3 "TABLE III ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") show the overall performance on Criteo and iPinYou datasets, respectively.
In FM, we employ 10-order factorization and correspondingly, we employ 10-order embedding in network models. CCPM has 1 embedding layer, 2 convolution layers (with max pooling) and 1 hidden layer (5 layers in total). FNN has 1 embedding layer and 3 hidden layers (4 layers in total).
Every PNN has 1 embedding layer, 1 product layer and 3 hidden layers (5 layers in total). The impact of network depth will be discussed later.

The LR and FM models are trained with L2 norm regularization, while FNN, CCPM and PNNs are trained with dropout.
By default, we set dropout rate at 0.5 on network hidden layers, which is proved effective in Figure [2](#S4.F2 "Figure 2 ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction").
Further discussions about the network architecture will be provided in Section [IV-E](#S4.SS5 "IV-E Ablation Study on Network Architecture ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction").

!(/html/1611.00144/assets/x2.png)

Figure 2: AUC Comparison of Dropout (OPNN).

Firstly, we focus on the AUC performance. The overall results in Table [III](#S4.T3 "TABLE III ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") and [III](#S4.T3 "TABLE III ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") illustrate that (i) FM outperforms LR, demonstrating the effectiveness of feature interactions; (ii) Neural networks outperform LR and FM, which validates the importance of high-order latent patterns; (iii) PNNs perform the best on both Criteo and iPinYou datasets. As for log loss, RMSE and RIG, the results are similar.

We also conduct t-test between our proposed PNNs and the other compared models. Table [III](#S4.T3 "TABLE III ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") shows the calculated p-values under log loss metric on both datasets. The results verify that our models significantly improve the performance of user response prediction against the baseline models.

We also find that PNN\*, which is the combination of IPNN and OPNN, has no obvious advantages over IPNN and OPNN on AUC performance.
We consider that IPNN and OPNN are sufficient to capture the feature interactions in multi-field categorical data.

!(/html/1611.00144/assets/x3.png)

Figure 3: Learning Curves on the iPinYou Dataset.

Figure [3](#S4.F3 "Figure 3 ‣ IV-D Performance Comparison ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") shows the AUC performance with respect to the training iterations on iPinYou dataset. We find that network models converge more quickly than LR and FM. We also observe that our two proposed PNNs have better convergence than other network models.

### IV-E Ablation Study on Network Architecture

In this section, we discuss the impact of neural network architecture.
For IPNN and OPNN,
we take three hyper-parameters (or settings) into consideration:
(i) embedding layer size, (ii) network depth and (iii) activation function. Since CCPM shares few similarities with other neural networks and PNN\* is just a combination of IPNN and OPNN, we only compare FNN, IPNN and OPNN in this section.

#### IV-E1 Embedding Layer

The embedding layer is to convert sparse binary inputs to dense real-value vectors.
Take word embedding as an example [[11](#bib.bib11)],
an embedding vector contains the information of the word and its context, and indicates the relationships between words.

We take the idea of embedding layer from [[12](#bib.bib12)]. In this paper, the latent vectors learned by FM are explained as node representations, and the authors use a pre-trained FM to initialize the embedding layers in FNN. Thus the factorization order of FM keeps consistent with the embedding order.

The input units are fully connected with the embedding layer within each field.
We compare different orders, like 2, 10, 50 and 100. However, when the order grows larger, it is harder to fit the parameters in memory, and the models are much easier to over-fit.
In our experiments, we take 10-order embedding in neural networks.

#### IV-E2 Network Depth

!(/html/1611.00144/assets/x4.png)

Figure 4: Performance Comparison over Network Depths.

We also explore the impact of network depth by adjusting the number of hidden layers in FNN and PNNs.
We compare different number of hidden layers: 1, 3, 5 and 7. Figure [4](#S4.F4 "Figure 4 ‣ IV-E2 Network Depth ‣ IV-E Ablation Study on Network Architecture ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") shows the performance as network depth grows. Generally speaking, the networks with 3 hidden layers have better generalization on the test set.

For convenience, we call convolution layers and product layers as representation layers.
These layers can capture complex feature patterns using fewer parameters, thus are efficient in training, and generalize better on the test set.

#### IV-E3 Activation Function

We compare three mainstream activation functions: sigmoid​(x)=11+e−xsigmoid𝑥11superscript𝑒𝑥\text{sigmoid}(x)=\frac{1}{1+e^{-x}}, tanh⁡(x)=1−e−2​x1+e−2​x𝑥1superscript𝑒2𝑥1superscript𝑒2𝑥\tanh(x)=\frac{1-e^{-2x}}{1+e^{-2x}}, and relu​(x)=max⁡(0,x)relu𝑥0𝑥\text{relu}(x)=\max(0,x).
Compared with the sigmoidal family, relu function has the advantages of sparsity and efficient gradient, which is possible to gain more benefits on multi-field categorical data.

!(/html/1611.00144/assets/x5.png)

Figure 5: AUC Comparison over Various Activation Functions.

Figure [5](#S4.F5 "Figure 5 ‣ IV-E3 Activation Function ‣ IV-E Ablation Study on Network Architecture ‣ IV Experiments ‣ Product-based Neural Networks for User Response Prediction") compares these activation functions on FNN, IPNN and OPNN. From this figure, we find that tanh has better performance than sigmoid. This is supported by [[12](#bib.bib12)].
Besides tanh, we find relu function also has good performance. Possible reasons include: (i) Sparse activation, nodes with negative outputs will not be activated; (ii) Efficient gradient propagation, no vanishing gradient problem or exploding effect; (iii) Efficient computation, only comparison, addition and multiplication.

## V Conclusion and Future Work

In this paper, we proposed a deep neural network model with novel architecture, namely Product-based Neural Network, to improve the prediction performance of DNN working on categorical data. And we chose CTR estimation as our working example. By exploration of feature interactions, PNN
is promising to learn high-order latent patterns on multi-field categorical data.
We designed two types of PNN: IPNN based on inner product and OPNN based on outer product. We also discussed solutions to reduce complexity, making PNN efficient and scalable. Our experimental results demonstrated that PNN outperformed the other state-of-the-art models in 4 metrics on 2 datasets. To sum up, we obtain the following conclusions: (i)
By investigating feature interactions, PNN gains better capacity on multi-field categorical data. (ii) Being both efficient and effective, PNN outperforms major state-of-the-art models. (iii)
Analogous to “AND”/“OR” gates, the product/add operations in PNN
provide a potential strategy for data representation, more specifically, rule representation.

In the future work, we will explore PNN with more general and complicated product layers. Besides, we are interested in explaining and visualizing the feature vectors learned by our models. We will investigate their properties, and further apply these node representations to other tasks.

## References

* [1]

  A. K. Menon, K.-P. Chitrapura, S. Garg *et al.*, “Response prediction
  using collaborative filtering with hierarchies and side-information,” in
  *SIGKDD*.   ACM, 2011, pp.
  141–149.
* [2]

  G.-R. Xue, H.-J. Zeng, Z. Chen, Y. Yu, W.-Y. Ma, W. Xi, and W. Fan,
  “Optimizing web search using web click-through data,” in *CIKM*, 2004.
* [3]

  W. Zhang, S. Yuan, and J. Wang, “Optimal real-time bidding for display
  advertising,” in *SIGKDD*.   ACM,
  2014, pp. 1077–1086.
* [4]

  X. He, J. Pan, O. Jin *et al.*, “Practical lessons from predicting clicks
  on ads at facebook,” in *Proceedings of the Eighth International
  Workshop on Data Mining for Online Advertising*.   ACM, 2014, pp. 1–9.
* [5]

  K.-c. Lee, B. Orten, A. Dasdan *et al.*, “Estimating conversion rate in
  display advertising from past erformance data,” in *SIGKDD*.   ACM, 2012, pp. 768–776.
* [6]

  A.-P. Ta, “Factorization machines with follow-the-regularized-leader for ctr
  prediction in display advertising,” in *IEEE BigData*.   IEEE, 2015, pp. 2889–2891.
* [7]

  Y. Cui, R. Zhang, W. Li *et al.*, “Bid landscape forecasting in online ad
  exchange marketplace,” in *SIGKDD*.   ACM, 2011, pp. 265–273.
* [8]

  Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, 2015.
* [9]

  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with
  deep convolutional neural networks,” in *NIPS*, 2012, pp. 1097–1105.
* [10]

  A. Graves, A.-r. Mohamed, and G. Hinton, “Speech recognition with deep
  recurrent neural networks,” in *ICASSP*.   IEEE, 2013, pp. 6645–6649.
* [11]

  T. Mikolov, I. Sutskever, K. Chen *et al.*, “Distributed representations
  of words and phrases and their compositionality,” in *NIPS*, 2013, pp.
  3111–3119.
* [12]

  W. Zhang, T. Du, and J. Wang, “Deep learning over multi-field categorical
  data: A case study on user response prediction,” *ECIR*, 2016.
* [13]

  Q. Liu, F. Yu, S. Wu *et al.*, “A convolutional click prediction model,”
  in *CIKM*.   ACM, 2015, pp.
  1743–1746.
* [14]

  M. Richardson, E. Dominowska, and R. Ragno, “Predicting clicks: estimating the
  click-through rate for new ads,” in *WWW*.   ACM, 2007, pp. 521–530.
* [15]

  T. Graepel, J. Q. Candela, T. Borchert *et al.*, “Web-scale bayesian
  click-through rate prediction for sponsored search advertising in microsoft’s
  bing search engine,” in *ICML*, 2010, pp. 13–20.
* [16]

  K. Ren, W. Zhang, Y. Rong, H. Zhang, Y. Yu, and J. Wang, “User response
  learning for directly optimizing campaign performance in display
  advertising,” in *CIKM*, 2016.
* [17]

  Y. Zhang, H. Dai, C. Xu *et al.*, “Sequential click prediction for
  sponsored search with recurrent neural networks,” *arXiv preprint
  arXiv:1404.5772*, 2014.
* [18]

  A. P. Engelbrecht, A. Engelbrecht, and A. Ismail, “Training product unit
  neural networks,” 1999.
* [19]

  W. Zhang, S. Yuan, and J. Wang, “Real-time bidding benchmarking with ipinyou
  dataset,” *arXiv:1407.7073*, 2014.
* [20]

  S. Rendle, “Factorization machines,” in *ICDM*.   IEEE, 2010, pp. 995–1000.
* [21]

  T. Chen, I. Goodfellow, and J. Shlens, “Net2net: Accelerating learning via
  knowledge transfer,” in *ICLR*, 2016.
* [22]

  H. B. McMahan, G. Holt, D. Sculley *et al.*, “Ad click prediction: a view
  from the trenches,” in *SIGKDD*.   ACM, 2013, pp. 1222–1230.
