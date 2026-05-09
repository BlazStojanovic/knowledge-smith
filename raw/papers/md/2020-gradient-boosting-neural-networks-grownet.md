---
arxiv: '2002.07971'
authors:
- Sarkhan Badirli
- Xuanqing Liu
- Zhengming Xing
- Avradeep Bhowmik
- Khoa Doan
- Sathiya S. Keerthi
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Gradient Boosting Neural Networks: GrowNet'
url: http://arxiv.org/abs/2002.07971v2
year: 2020
---

[2002.07971] Gradient Boosting Neural Networks: GrowNet














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



# Gradient Boosting Neural Networks: GrowNet

Sarkhan Badirli
  
Department of Computer Science
  
Purdue University, Indiana, USA
  
sbadirli@purdue.edu
  
&Xuanqing Liu
  
Department of Computer Science
  
UCLA, California, USA
  
xqliu@cs.ucla.edu
&Zhengming Xing
  
Linkedin, California, USA
  
zhxing@linkedin.com
&Avradeep Bhowmik
  
Amazon, California, USA
  
avradeep.1@gmail.com
  
&Khoa Doan
  
Department of Computer Science
  
Virginia Tech, Virginia, USA
  
textttdoankhoadang@gmail.com
  
&Sathiya S. Keerthi
  
Linkedin, California, USA
  
keselvaraj@linkedin.com
  
This work was started when authors were in Criteo AI Lab

###### Abstract

A novel gradient boosting framework is proposed where shallow neural networks are employed as “weak learners”. General loss functions are considered under this unified framework with specific examples presented for classification, regression and learning to rank. A fully corrective step is incorporated to remedy the pitfall of greedy function approximation of classic gradient boosting decision tree. The proposed model rendered outperforming results against state-of-the-art boosting methods in all three tasks on multiple datasets. An ablation study is performed to shed light on the effect of each model components and model hyperparameters.

## 1 Introduction

AI and machine learning pervade every aspect of modern life from email spam filtering and e-commerce, to financial security and medical diagnostics [[18](#bib.bib18), [25](#bib.bib25)]. Deep learning in particular has been one of the key innovations that has truly pushed the boundary of science beyond what was considered feasible [[14](#bib.bib14), [12](#bib.bib12)].

However, in spite of its seemingly limitless possibilities, both in theory as well as demonstrated practice, developing tailor-made deep neural networks for new application areas remains notoriously difficult because of its inherent complexity. Designing architectures for any given application requires immense dexterity and often a lot of luck. The lack of an established paradigm for creating an application-specific DNN presents significant challenges to practitioners, and often results in resorting to heuristics or even hacks.

In this paper, we attempt to rectify this situation by introducing a novel paradigm that builds neural networks from the ground up layer by layer. Specifically, we use the idea of gradient boosting [[11](#bib.bib11)] which has a formidable reputation in machine learning for its capacity to incrementally build sophisticated models out of simpler components, that can successfully be applied to the most complex learning tasks. Popular GBDT frameworks like XGBoost [[6](#bib.bib6)], LightGBM [[17](#bib.bib17)] and CatBoost [[22](#bib.bib22)] use decision trees as weak learners, and combine them using a gradient boosting framework to build complex models that are widely used in both academia and industry as a reliable workhorse for common tasks in a wide variety of domains.

However, while useful in their own right, decision trees are not universally applicable, and there are many domains– especially involving structured data– where deep neural networks perform much better [[32](#bib.bib32), [29](#bib.bib29), [1](#bib.bib1)]. In this paper, we combine the power of gradient boosting with the flexibility and versatility of neural networks and introduce a new modelling paradigm called GrowNet that can build up a DNN layer by layer. Instead of decision trees, we use shallow neural networks as our weak learners in a general gradient boosting framework that can be applied to a wide variety of tasks spanning classification, regression and ranking. We introduce further innovations like adding second order statistics to the training process, and also including a global corrective step that has been shown, both in theory [[31](#bib.bib31)] and in empirical evaluation, to provide performance lift and precise fine-tuning to the specific task at hand.

Our specific contributions are summarised below:

* •

  We propose a novel approach to combine the power of gradient boosting to incrementally build complex deep neural networks out of shallow components. We introduce a versatile framework that can readily be adapted for a diverse range of machine learning tasks in a wide variety of domains.
* •

  We develop an off-the-shelf optimization algorithm that is faster and easier to train than traditional deep neural networks. We introduce training innovations including second order statistics and global corrective steps that improve stability and allow finer-grained tuning of our models for specific tasks.
* •

  We demonstrate the efficacy of our techniques using experimental evaluation, and show superior results on multiple real datasets in three different ML tasks: classification, regression and learning-to-rank.

## 2 Related Work

In this section, we briefly summarize the gradient boosting algorithms with decision trees and general boosting/ensemble methods for training neural nets.

Gradient Boosting Algorithms.
Gradient Boosting Machine [[11](#bib.bib11)] is a function estimation method using numerical optimization in the function space. Unlike parameter estimation, function approximation cannot be solved by traditional optimization methods in Euclidean space. Decision Trees are the most common functions (predictive learners) that are used in Gradient Boosting framework. In his seminal paper, [[11](#bib.bib11)] proposed Gradient Boosting Decision Trees (GBDT) where decision trees are trained in sequence and each tree is modeled by fitting negative gradients. In recent years, there have been many implementations of GBDT in machine learning literature. Among these, [[27](#bib.bib27)] used GBDT to perform learning to rank, [[10](#bib.bib10)] did classification and [[6](#bib.bib6), [17](#bib.bib17)] generalized GBDT for multi-tasking purposes. In particular, scalable framework of [[6](#bib.bib6)] made it possible for data scientists to achieve state-of-the-art results on various industry related machine learning problems. For that reason, we take XGBoost [[6](#bib.bib6)] as our baseline. Unlike these GBDT methods, we propose gradient boosting neural network where we train gradient boosting with shallow neural nets. Using neural nets as base learners also gives our method an edge over GBDT models, where we can correct each previous model after adding the new one, referred to as “corrective step”, in addition to the ability to propagate information from the previous predictors to the next ones.

Boosted Neural Nets.
Although weak learners, like decision trees, are popular in boosting and ensemble methods, there have been a substantial work done on combining neural nets with boosting/ensemble methods for better performance over single large/deep neural networks.
The idea of considering shallow neural nets as weak learners and constructively combining them started with [[8](#bib.bib8)]. In their pioneering work, fully connected, multi-layer perceptrons are trained in a layer-by-layer fashion and added to get a cascade-structured neural net. Their model is not exactly a boosting model as the final model is a single, multi-layer neural network.

In 1990’s, ensemble of neural networks got popular as ensemble methods helped to significantly improve the generalization ability of neural nets. Nevertheless, these methods were simply either majority voting [[13](#bib.bib13)] for classification tasks, simple averaging [[20](#bib.bib20)] or weighted averaging [[21](#bib.bib21)] for regression tasks. After the introduction of adaptive boosting (A​d​a​b​o​o​s​t𝐴𝑑𝑎𝑏𝑜𝑜𝑠𝑡Adaboost) algorithm [[9](#bib.bib9)], [[23](#bib.bib23)] investigated boosting with multi-layer neural networks for a character recognition task and achieved a remarkable performance improvement. They extended the work to traditional machine learning tasks with variations of Adaboost methods where different weighting schemes are explored [[24](#bib.bib24)]. The adaptive boosting can be seen as a specific version of the gradient boosting algorithm where a simple exponential loss function is used [[10](#bib.bib10)].

In early 2000’s, [[15](#bib.bib15)]
introduced greedy layer-wise unsupervised training for Deep Belief Nets (DBN). DBN is built upon a layer at a time by utilizing Gibbs sampling to obtain the estimator of the gradient on the log-likelihood of Restricted Boltzmann Machines (RBM) in each layer. The authors of [[3](#bib.bib3)] expounded this work for continuous inputs and explained its success on attaining high quality features from image data. They concluded that unsupervised training helped model training by initializing RBM weights in a region close to a good local minimum.

Most recently, AdaNet [[7](#bib.bib7)] was proposed to adaptively built Neural Network (NN) layer by layer from a singe layer NN to perform image classification task. Beside learning network weights, AdaNet adjusts the network structure and its growth procedure is reinforced by a theoretical justification. AdaNet optimizes over a generalization bound that consists of empirical risk and complexity of the architecture. Coordinate descend approach is applied to the objective function, and heuristic search (weak learning algorithm) is performed to obtain δ−o​p​t​i​m​a​l𝛿𝑜𝑝𝑡𝑖𝑚𝑎𝑙\delta-optimal coordinates. Although the learning process is boosting-style, the final model is a single NN whose final output layer is connected to all lower layers. Unlike AdaNet, we train each weak learner in a gradient boosting style, resulting in less entangled training. The final prediction is the weighted sum of all weak learners’ output. Our method also renders a unified platform to perform various ML tasks.

In recent years, a few work have been done to explain the success of deep residual neural networks [[14](#bib.bib14)] with hundreds of layers by showing that they can be decomposed into a collection of many subnetworks.
The work in [[16](#bib.bib16)] extends AdaNet to specifically focus on ResNet architecture [[14](#bib.bib14)] to provide a new training algorithm for ResNet. The authors of [[28](#bib.bib28)], meanwhile, argue that these deeper layers might serve as a bagging mechanism in a similar spirit to random forest classifier. These studies challenge the common belief that neural networks are too strong to serve as weak learners for boosting methods.

![Refer to caption](/html/2002.07971/assets/x1.png)


Figure 1: GrowNet architecture. After the first weak learner, each predictor is trained on combined features from original input and penultimate layer features from previous weak learner. The final output is the weighted sum of outputs from all predictors, ∑k=1k=Kαk​fk​(x)superscriptsubscript𝑘1𝑘𝐾subscript𝛼𝑘subscript𝑓𝑘𝑥\sum\_{k=1}^{k=K}\alpha\_{k}f\_{k}(x). Here Model K means weak learner K.

## 3 Model

In this section, we first describe the basic framework of GrowNet for general loss functions, and then we show how the corrective step is incorporated.
The key idea in gradient boosting is to take simple, lower-order models as weak learners and use them as fundamental building blocks to build a powerful, higher-order model by sequential boosting using first or second order gradient statistics. We use shallow neural networks (e.g., with one or two hidden layers) as weak learners in this paper. As each boosting step, we augment the original input features with the output from the penultimate layer of the current iteration (see Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Gradient Boosting Neural Networks: GrowNet")). This augmented feature-set is then fed as input to train the next weak learner via a boosting mechanism using the current residuals. The final output of the model is a weighted combination of scores from all these sequentially trained models.

### 3.1 Gradient Boosting Neural Network: GrowNet

Let us assume a dataset with n𝑛n samples in d𝑑d dimensional feature space 𝒟={(𝒙i,yi)i=1n|𝒙i∈ℝd,yi∈ℝ}𝒟conditional-setsuperscriptsubscriptsubscript𝒙𝑖subscript𝑦𝑖𝑖1𝑛formulae-sequencesubscript𝒙𝑖superscriptℝ𝑑subscript𝑦𝑖ℝ\mathcal{D}=\{(\boldsymbol{x}\_{i},y\_{i})\_{i=1}^{n}|\boldsymbol{x}\_{i}\in\mathbb{R}^{d},y\_{i}\in\mathbb{R}\}. GrowNet uses K𝐾K additive functions to predict the output,

|  |  |  |  |
| --- | --- | --- | --- |
|  | y^i=ℰ​(𝒙i)=∑k=0Kαk​fk​(𝒙i),fk∈ℱformulae-sequencesubscript^𝑦𝑖ℰsubscript𝒙𝑖superscriptsubscript𝑘0𝐾subscript𝛼𝑘subscript𝑓𝑘subscript𝒙𝑖subscript𝑓𝑘ℱ\hat{y}\_{i}=\mathcal{E}(\boldsymbol{x}\_{i})=\sum\_{k=0}^{K}\alpha\_{k}f\_{k}(\boldsymbol{x}\_{i}),f\_{k}\in\mathcal{F} |  | (1) |

where ℱℱ\mathcal{F} is the space of multilayer perceptrons and αksubscript𝛼𝑘\alpha\_{k} is the step size (boost rate). Each function fksubscript𝑓𝑘f\_{k} represents an independent, shallow neural network with a linear layer as an output layer. For a given sample 𝒙𝒙\boldsymbol{x}, the model calculates the prediction as a weighted sum of fksubscript𝑓𝑘f\_{k}’s in GrowNet.

Let l𝑙l be any differentiable convex loss function. Our objective is to learn a set of functions (shallow neural networks) that minimize the following equation: ℒ​(ℰ)=∑i=0nl​(yi,y^i).ℒℰsuperscriptsubscript𝑖0𝑛𝑙subscript𝑦𝑖subscript^𝑦𝑖\mathcal{L}(\mathcal{E})=\sum\_{i=0}^{n}l(y\_{i},\hat{y}\_{i}).

We may further add regularization terms to penalize the model complexity but it is omitted for simplicity in this work. As the objective we are optimizing is over the functions and not on the parameters, traditional optimization techniques will not work here. Analogous to GBDT [[11](#bib.bib11)], the model is trained in an additive manner.

Let y^i(t−1)=∑k=0t−1αk​fk​(𝒙i)subscriptsuperscript^𝑦𝑡1𝑖superscriptsubscript𝑘0𝑡1subscript𝛼𝑘subscript𝑓𝑘subscript𝒙𝑖\hat{y}^{(t-1)}\_{i}=\sum\_{k=0}^{t-1}\alpha\_{k}f\_{k}(\boldsymbol{x}\_{i}) be the output of GrowNet at stage t−1𝑡1t-1 for the sample 𝒙isubscript𝒙𝑖\boldsymbol{x}\_{i}.
We greedily seek the next weak learner ft​(x)subscript𝑓𝑡xf\_{t}(\textbf{x}) that will minimize the loss at stage t𝑡t which can be summized as,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ(t)=∑i=0nl​(yi,y^i(t−1)+αt​ft​(xi))superscriptℒ𝑡superscriptsubscript𝑖0𝑛𝑙subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡1subscript𝛼𝑡subscript𝑓𝑡subscriptx𝑖\mathcal{L}^{(t)}=\sum\_{i=0}^{n}l(y\_{i},\hat{y}\_{i}^{(t-1)}+\alpha\_{t}f\_{t}(\textbf{x}\_{i})) |  | (2) |

In addition, Taylor expansion of the loss function l𝑙l was adopted to ease the computational complexity. As second-order optimization techniques are proven to be superior to first-order ones and require less steps to converge, we train models with Newton-Raphson steps. Consequently, regardless of the ML task, individual model parameters are optimized by running regression on the second order gradients of the GrowNet’s outputs. Objective function for the weak learner ftsubscript𝑓𝑡f\_{t} can be simplified as follows,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ(t)=∑i=0nhi​(y~i−αt​ft​(𝒙i))2superscriptℒ𝑡superscriptsubscript𝑖0𝑛subscriptℎ𝑖superscriptsubscript~𝑦𝑖subscript𝛼𝑡subscript𝑓𝑡subscript𝒙𝑖2\mathcal{L}^{(t)}=\sum\_{i=0}^{n}h\_{i}(\tilde{y}\_{i}-\alpha\_{t}f\_{t}(\boldsymbol{x}\_{i}))^{2} |  | (3) |

where y~i=−gi/hisubscript~𝑦𝑖subscript𝑔𝑖subscriptℎ𝑖\tilde{y}\_{i}=-g\_{i}/h\_{i}, and gisubscript𝑔𝑖g\_{i} & hisubscriptℎ𝑖h\_{i} are the first and second order gradients of the objective function l𝑙l at 𝒙isubscript𝒙𝑖\boldsymbol{x}\_{i}, w.r.t. y^i(t−1)superscriptsubscript^𝑦𝑖𝑡1\hat{y}\_{i}^{(t-1)}.
(See pseudo-code in part 1 of Algorithm 1 from supplementary material.)

### 3.2 Corrective Step (C/S)

In a traditional boosting framework, each weak learner is greedily learned. This means that only the parameters of tt​hsuperscript𝑡𝑡ℎt^{th} weak learner are updated at boosting step t𝑡t where all the parameters of previous t−1𝑡1t-1 weak learners remain unchanged. The myopic learning procedures may cause the model to get stuck in a local minima, and a fixed boosting rate αksubscript𝛼𝑘\alpha\_{k} aggravates the issue [[11](#bib.bib11)]. Therefore, we implemented a corrective step to address this problem. In the corrective step, instead of fixing the previous t−1𝑡1t-1 weak learners, we allow update of the parameters of the previous t−1𝑡1t-1 weak learners through back-propagation. Moreover, we incorporated the boosting rate αksubscript𝛼𝑘\alpha\_{k} into parameters of the model and it is automatically updated through the corrective step. Beyond getting better performance, this move saves us from tuning a delicate parameter. C/S can also be interpreted as a regularizer to mitigate the correlation among weak learners, as during corrective step, the main training objective becomes task specific loss function on just original inputs. The usefulness of this step is empirically and theoretically investigated in [[31](#bib.bib31)] for gradient boosting decision tree models. Our experiments in the ablation study [6.2](#S6.SS2 "6.2 Analyzing corrective step ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") further validate the necessity of c/s in our model as well. The corrective step is summarized in the second part of Algorithm 1 in the supplementary material.

## 4 Applications

In this section, we show how GrowNet can be
adapted for regression, classification and learning to rank problems.

GrowNet for Regression.
We employ mean squared error (MSE) loss function for the regression task. Let us assume l𝑙l is the MSE loss; then we can easily obtain y~isubscript~𝑦𝑖\tilde{y}\_{i}, first order, and second order statistics at stage t𝑡t, as follows:

|  |  |  |
| --- | --- | --- |
|  | gi=2​(y^i(t−1)−yi),hi=2⟹y~i=yi−y^i(t−1)formulae-sequencesubscript𝑔𝑖2superscriptsubscript^𝑦𝑖𝑡1subscript𝑦𝑖subscriptℎ𝑖2subscript~𝑦𝑖subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡1\displaystyle g\_{i}=2(\hat{y}\_{i}^{(t-1)}-y\_{i}),\quad h\_{i}=2\implies\tilde{y}\_{i}=y\_{i}-\hat{y}\_{i}^{(t-1)} |  |

We train next weak learner ftsubscript𝑓𝑡f\_{t} by least square regression on {𝒙i,y~i}subscript𝒙𝑖subscript~𝑦𝑖\{\boldsymbol{x}\_{i},\tilde{y}\_{i}\} for i=1,2,…,n𝑖

12…𝑛i=1,2,...,n. In the corrective step, all model parameters in GrowNet are updated again using the MSE loss.

GrowNet for Classification.
For the illustration purposes, let us consider the binary cross entropy loss function; however, note that any differentiable loss function can be used. Choosing labels yi∈{−1,+1}subscript𝑦𝑖11y\_{i}\in\{-1,+1\} (this notation has an advantage of yi2=1superscriptsubscript𝑦𝑖21y\_{i}^{2}=1, which will be used in the derivation), the first and second order gradients, gisubscript𝑔𝑖g\_{i} and hisubscriptℎ𝑖h\_{i} , respectively, at stage t𝑡t can be written as follows,

|  |  |  |
| --- | --- | --- |
|  | gi=−2​yi1+e2​yi​y^i(t−1),hi=4​yi2​e2​yi​y^i(t−1)(1+e2​yi​y^i(t−1))2⟹y~i=−gi/hi=yi​(1+e−2​yi​y^i(t−1))/2formulae-sequencesubscript𝑔𝑖2subscript𝑦𝑖1superscript𝑒2subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡1subscriptℎ𝑖4superscriptsubscript𝑦𝑖2superscript𝑒2subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡1superscript1superscript𝑒2subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡12subscript~𝑦𝑖subscript𝑔𝑖subscriptℎ𝑖subscript𝑦𝑖1superscript𝑒2subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑡12\displaystyle g\_{i}=\frac{-2y\_{i}}{1+e^{2y\_{i}\hat{y}\_{i}^{(t-1)}}},\quad h\_{i}=\frac{4y\_{i}^{2}e^{2y\_{i}\hat{y}\_{i}^{(t-1)}}}{(1+e^{2y\_{i}\hat{y}\_{i}^{(t-1)}})^{2}}\implies\tilde{y}\_{i}=-g\_{i}/h\_{i}=y\_{i}(1+e^{-2y\_{i}\hat{y}\_{i}^{(t-1)}})/2 |  |

The next weak learner ftsubscript𝑓𝑡f\_{t} is fitted by least square regression using second order gradient statistics on {xi,y~i}subscript𝑥𝑖subscript~𝑦𝑖\{x\_{i},\tilde{y}\_{i}\}. In the corrective step, parameters of all the added predictive functions are updated by retraining the whole model using the binary cross entropy loss. This step slightly corrects the weights according to the main objective function of the task at hand, i.e. classification in this case.

GrowNet for Learning to Rank.
In this part, we demonstrate how the model is adapted to learning to rank (L2R) with a pairwise loss. In the L2R framework, there are queries and documents associated with each query. A document can be associated with many different queries. Then for each query, the associated documents have relevance scores. Assume for a given query, a pair of documents Uisubscript𝑈𝑖U\_{i} and Ujsubscript𝑈𝑗U\_{j} is chosen. Assume further that we have a feature vector for these documents, 𝒙isubscript𝒙𝑖\boldsymbol{x}\_{i} and 𝒙jsubscript𝒙𝑗\boldsymbol{x}\_{j}. Let y^isubscript^𝑦𝑖\hat{y}\_{i} and y^jsubscript^𝑦𝑗\hat{y}\_{j} denote the output of the model for samples 𝒙isubscript𝒙𝑖\boldsymbol{x}\_{i} and 𝒙jsubscript𝒙𝑗\boldsymbol{x}\_{j} respectively.
According to [[4](#bib.bib4)], a common pairwise loss for a given query can be formulated as follows,

|  |  |  |
| --- | --- | --- |
|  | l​(y^i,y^j)=12​(1−Si​j)​σ0​(y^i−y^j)+l​o​g​(1+e−σ0​(y^i−y^j))𝑙subscript^𝑦𝑖subscript^𝑦𝑗121subscript𝑆𝑖𝑗subscript𝜎0subscript^𝑦𝑖subscript^𝑦𝑗𝑙𝑜𝑔1superscript𝑒subscript𝜎0subscript^𝑦𝑖subscript^𝑦𝑗l(\hat{y}\_{i},\hat{y}\_{j})=\frac{1}{2}(1-S\_{ij})\sigma\_{0}(\hat{y}\_{i}-\hat{y}\_{j})+log(1+e^{-\sigma\_{0}(\hat{y}\_{i}-\hat{y}\_{j})}) |  |

where Si​j∈{0,−1,+1}subscript𝑆𝑖𝑗011S\_{ij}\in\{0,-1,+1\} denotes the documents’ relevance difference; it is 111 if the Uisubscript𝑈𝑖U\_{i} has a relevance score greater than Ujsubscript𝑈𝑗U\_{j}, −11-1 vice-versa and 00 if both document have been labeled with the same relevance score. σ0subscript𝜎0\sigma\_{0} is the sigmoid function.
Note that the cost function l𝑙l is symmetric and its gradients can be easily computed as follows (for the details, readers can refer to [[4](#bib.bib4)]),

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂y^il​(y^i,y^j)subscriptsubscript^𝑦𝑖𝑙subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle\partial\_{\hat{y}\_{i}}l(\hat{y}\_{i},\hat{y}\_{j}) | =σ0​(12​(1−Si​j)−11+eσ0​(y^i−y^j))=−∂y^jl​(y^i,y^j)absentsubscript𝜎0121subscript𝑆𝑖𝑗11superscript𝑒subscript𝜎0subscript^𝑦𝑖subscript^𝑦𝑗subscriptsubscript^𝑦𝑗𝑙subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle=\sigma\_{0}(\frac{1}{2}(1-S\_{ij})-\frac{1}{1+e^{\sigma\_{0}(\hat{y}\_{i}-\hat{y}\_{j})}})=-\partial\_{\hat{y}\_{j}}l(\hat{y}\_{i},\hat{y}\_{j}) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | ∂y^i2l​(y^i,y^j)superscriptsubscriptsubscript^𝑦𝑖2𝑙subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle\partial\_{\hat{y}\_{i}}^{2}l(\hat{y}\_{i},\hat{y}\_{j}) | =σ02​(11+eσ0​(y^i−y^j))​(1−11+eσ0​(y^i−y^j))absentsuperscriptsubscript𝜎0211superscript𝑒subscript𝜎0subscript^𝑦𝑖subscript^𝑦𝑗111superscript𝑒subscript𝜎0subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle=\sigma\_{0}^{2}(\frac{1}{1+e^{\sigma\_{0}(\hat{y}\_{i}-\hat{y}\_{j})}})(1-\frac{1}{1+e^{\sigma\_{0}(\hat{y}\_{i}-\hat{y}\_{j})}}) |  |

where I𝐼I denotes the set of pairs of indices {i,j}𝑖𝑗\{i,j\}, for which Uisubscript𝑈𝑖U\_{i} is desired to be ranked differently from Ujsubscript𝑈𝑗U\_{j} for a given query.
Then for a particular document Uisubscript𝑈𝑖U\_{i}, the loss function and its first and second order statistics can be derived as follows,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | l𝑙\displaystyle l | =∑j:{i,j}∈Il​(y^i,y^j)+∑j:{j,i}∈Il​(y^i,y^j)absentsubscript:𝑗𝑖𝑗𝐼𝑙subscript^𝑦𝑖subscript^𝑦𝑗subscript:𝑗𝑗𝑖𝐼𝑙subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle=\sum\_{j:\{i,j\}\in I}l(\hat{y}\_{i},\hat{y}\_{j})+\sum\_{j:\{j,i\}\in I}l(\hat{y}\_{i},\hat{y}\_{j}) |  | |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | gisubscript𝑔𝑖\displaystyle g\_{i} | =∑j:{i,j}∈I∂y^il​(y^i,y^j)−∑j:{j,i}∈I∂y^il​(y^i,y^j),hiabsent  subscript:𝑗𝑖𝑗𝐼subscriptsubscript^𝑦𝑖𝑙subscript^𝑦𝑖subscript^𝑦𝑗subscript:𝑗𝑗𝑖𝐼subscriptsubscript^𝑦𝑖𝑙subscript^𝑦𝑖subscript^𝑦𝑗subscriptℎ𝑖\displaystyle=\sum\_{j:\{i,j\}\in I}\partial\_{\hat{y}\_{i}}l(\hat{y}\_{i},\hat{y}\_{j})-\sum\_{j:\{j,i\}\in I}\partial\_{\hat{y}\_{i}}l(\hat{y}\_{i},\hat{y}\_{j}),\quad h\_{i} | =∑j:{i,j}∈I∂y^i2l​(y^i,y^j)−∑j:{j,i}∈I∂y^i2l​(y^i,y^j)absentsubscript:𝑗𝑖𝑗𝐼superscriptsubscriptsubscript^𝑦𝑖2𝑙subscript^𝑦𝑖subscript^𝑦𝑗subscript:𝑗𝑗𝑖𝐼superscriptsubscriptsubscript^𝑦𝑖2𝑙subscript^𝑦𝑖subscript^𝑦𝑗\displaystyle=\sum\_{j:\{i,j\}\in I}\partial\_{\hat{y}\_{i}}^{2}l(\hat{y}\_{i},\hat{y}\_{j})-\sum\_{j:\{j,i\}\in I}\partial\_{\hat{y}\_{i}}^{2}l(\hat{y}\_{i},\hat{y}\_{j}) |  |

## 5 Experiments

Experiment Setup. All predictive functions added to the model are multilayer perceptrons with two hidden layers. We generally set the number of hidden layer units to roughly half of or equal to the input feature dimension. More hidden layers degraded the performance as the model starts overfitting. 404040 additive functions were employed in the experiments for all three tasks and the number of weak learners in test time is chosen by the validation results. Boosting rate is initially set to 111 and automatically adjusted during the corrective step.

We trained each predictive function for just 111 epoch and the entire model is also trained for 111 epoch during the corrective step by stochastic gradient descent with Adam optimizer. The number of epochs is increased to 2 for the ranking task. We also employed 2​D2𝐷2D batch normalization on the hidden layers. We compared the model performance with XGBoost since similar results are obtain with LightGBM or CatBoost and with AdaNet. Tuning and model details of all 3 methods are provided in supplementary material.

Datasets. We evaluate our model on 5 datasets from 3 different tasks.
Higgs Bozon dataset is used for classification. Higgs data is created using Monte Carlo simulations on high energy physics events.

To perform regression, 2 datasets from UCI machine learning repository are selected. The first one is Computed Tomography (CT) slice localization data where the aim is to retrieve the location of CT slices on axial axis. The second regression dataset is YearPredictionMSD, a subset of Million Song dataset. The goal is to predict the release year of a song from its audio features.

We choose Yahoo LTR dataset [[5](#bib.bib5)] for the learning to rank task as it is a well-known benchmark dataset and also used in XGBoost’s paper. The dataset has 20​K20𝐾20K queries each associated with approximately 222222 documents. Train-test split from the original paper is preserved. The second benchmark ranking dataset we used is MSLR-WEB 10​K10𝐾10K in which there are 10​K10𝐾10K queries, each corresponding to list of 100−200100200100-200 documents. Detailed statistics of each dataset can be found in the supplementary material.

Table 1: L2R results in Normalized Discounted Cumulative Gain for top 5 and 10 queries (NDCG@5 & 10), on Microsoft Learning to Rank with 10K queries and Yahoo LTR datasets. GrowNet results are average of 5 iterations and the values in the parenthesis represents the standard deviation.

|  | MSLR-WEB 10K | | Yahoo LTR | |
| --- | --- | --- | --- | --- |
|  | NDCG@5 | NDCG@10 | NDCG@5 | NDCG@10 |
| XGBoost | 0.4677​(0.0287)0.46770.02870.4677(0.0287) | 0.4858​(0.0245)0.48580.02450.4858(0.0245) | 0.76180.76180.7618 | 0.79130.79130.7913 |
| GrowNet (pairwise loss) | 0.5106​(0.0011)0.51060.0011\boldsymbol{0.5106}(0.0011) | 0.5203​(0.0015)0.52030.0015\boldsymbol{0.5203}(0.0015) | 0.7726​(0.0006)0.77260.0006\boldsymbol{0.7726}(0.0006) | 0.8101​(0.0003)0.81010.0003\boldsymbol{0.8101}(0.0003) |
| GrowNet (Gen. I div. loss) | 0.5044​(0.0072)0.50440.00720.5044(0.0072) | 0.5137​(0.0070)0.51370.00700.5137(0.0070) | 0.7713​(0.0006)0.77130.00060.7713(0.0006) | 0.8088​(0.0005)0.80880.00050.8088(0.0005) |

### 5.1 Results

Table 2: Regression results in root mean square error (RMSE). GrowNet results are average of 5 iterations and the values in the parenthesis represent standard deviation.

|  | Music Year Pred. | Slice Localz. |
| --- | --- | --- |
| XGBoost | 8.93018.93018.9301 | 6.67446.67446.6744 |
| AdaNet | 12.177812.177812.1778 | 5.38245.38245.3824 |
| GrowNet | 8.81568.8156\boldsymbol{8.8156} (0.0061) | 5.31125.3112\boldsymbol{5.3112} (0.3512) |

Table 3: Classification results, in AUC, on Higgs bozon dataset. For our model, we preset 3 different results: using all the data, 10%percent1010\% of the data (1​M1𝑀1M), and 1%percent11\% of the data (100​K100𝐾100K).

| XGBoost | 0.83040.83040.8304 |
| --- | --- |
| GrowNet (all data) | 0.85100.8510\boldsymbol{0.8510} |
| GrowNet (data sampling=10%absentpercent10=10\%) | 0.84390.84390.8439 |
| GrowNet (data sampling=1%absentpercent1=1\%) | 0.81800.81800.8180 |

Regression. Table [3](#S5.T3 "Table 3 ‣ 5.1 Results ‣ 5 Experiments ‣ Gradient Boosting Neural Networks: GrowNet") reports regression performance on two UCI datasets. GrowNet outperforms both methods on Music dataset where AdaNet delivers the worse result. On CT slice localization dataset, our model obtains on par results with Adanet and displays 21%percent2121\% decrease in RMSE compared to XGBoost.

Classification. To make a fair comparison with XGBoost, we tested our model on Higgs bozon dataset as it is used in XGBoost’s paper [[6](#bib.bib6)]. Classification results are presented in the Table [3](#S5.T3 "Table 3 ‣ 5.1 Results ‣ 5 Experiments ‣ Gradient Boosting Neural Networks: GrowNet"). GrowNet clearly outperforms XGBoost using all the data. Subsampling 10%percent1010\% of the data for training each weak learner also renders better performance. We used 30 weak learners (multilayer perceptrons with two hidden layers of 161616 units) and the number of weak learners to be used at test time is chosen by validation results. In all 3 experiments, this number was 30.

Learning to Rank. Ranking experiment results on Yahoo and MSLR datasets are presented in Table [1](#S5.T1 "Table 1 ‣ 5 Experiments ‣ Gradient Boosting Neural Networks: GrowNet"). We evaluated GrowNet with 2 different loss functions, namely pairwise loss and generalized I-divergence loss. In both scenarios, GrowNet outperforms XGBoost on both datasets; in particular, it delivers 6−8%6percent86-8\% increase on Microsoft data in NDCG@5 and NDCG@10. For our model to achieve these results, 30 weak learners were enough.

## 6 Ablation study

![Refer to caption](/html/2002.07971/assets/x2.png)


Figure 2: Classification training losses

We investigated different components of GrowNet. We picked 2 datasets for these experiments: Higgs and Microsoft. For Higgs dataset, we randomly selected 1​M1𝑀1M points for training and 5%percent55\% of the remaining as the validation set. The original test data was used as the test set. For Microsoft dataset, we used Fold 1 and the original split was preserved. In all upcoming experiments, only the component that is being analyzed, is altered while the rest of the parameters remain unchanged. All ablation experiments are reported in Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet"), and the third column (GrowNet) represents the results from final version of our model on these datasets.

Table 4: Ablation study experiment results on Higgs 1​M1𝑀1M and Microsoft (Fold 1) datasets. All models have two-layer shallow networks as weak learners. Hidden layer dimension is 16 for classification and 64 for ranking task. The third column is the final GrowNet model that all different versions are compared against. Reported results are AUC scores for classification and NDCG for ranking.

| Datasets | Eval. metric | GrowNet | 1s​tsuperscript1𝑠𝑡1^{st} order grad. | Constant αtsubscript𝛼𝑡\alpha\_{t} | Simple version | No C/S | C/S in every 5 stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Higgs 1​M1𝑀1M | AUC | 0.84010.8401\boldsymbol{0.8401} | 0.83630.83630.8363 | 0.83970.83970.8397 | 0.83260.83260.8326 | 0.80930.80930.8093 | 0.83150.83150.8315 |
| MSLR Fold1 | NDCG@5 | 0.51060.5106\boldsymbol{0.5106} | 0.50010.50010.5001 | 0.50200.50200.5020 | 0.48360.48360.4836 | 0.47430.47430.4743 | 0.48810.48810.4881 |
| NDCG@10 | 0.51950.5195\boldsymbol{0.5195} | 0.51040.51040.5104 | 0.51150.51150.5115 | 0.49720.49720.4972 | 0.48720.48720.4872 | 0.49980.49980.4998 |

### 6.1 Stacked versus simple version

As seen in Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Gradient Boosting Neural Networks: GrowNet"), every weak learner except the first one is trained on the combined features of the original input and penultimate layer’s features from previous predictive function. It is worth to note that the input dimension does not grow by iteration; indeed, it is always the dimension of hidden layer plus the dimension of of original input. This idea of stacked features has a weak resemblance to auto-context [[26](#bib.bib26)] in literature, where the authors utilized the direct output of the classifier, along with the original inputs, to boost the image segmentation performance. The work in [[2](#bib.bib2)] extended this idea to not only use the output of the classifier, probabilities, but also the raw prediction image itself. Our model is significantly different from these methods, as we do not simply use the previous model’s output but more expressive representation at the penultimate layer. These features leverage our model by propagating more complex information from previous model to the new one. To test the advantage of this stacked model, we compared the proposed model against its simpler version in which the original input features are used for all learners. The sixth column in Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") presents the results from the simpler version. In both tasks, the stacked model outperforms the simpler version; especially, the difference is noteworthy in the ranking task. Training loss in Figure [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") also supports the information gain while the stacked version is utilized. Unlike tree boosting methods, our model makes this architecture possible through its flexible weak learners.

### 6.2 Analyzing corrective step

Among all components of the model, the corrective step is presumably the most vital one. In this step, the parameters of all weak learners, that are added to the model, are updated by training the whole model on the original inputs without the penultimate layer features. The loss function used in this step is a task specific one. This procedure allows the model to rectify the parameters to specifically better accommodate the task at hand rather than fitting negative gradients. C/S also alleviates the potential correlation among weak learners. Moreover, within this step, we incorporated the boosting rate αtsubscript𝛼𝑡\alpha\_{t} and it is automatically adjusted without requiring any tuning. The last two columns of Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") present the classification and learning to rank results from GrowNet without using any corrective step and using a corrective step in every 5 stages, respectively. The performance severely degraded in the former one, and the model hardly learned any information after a couple of predictive functions added. The flat training loss in Figure [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") confirms this phenomenon as well. Running the corrective step in every 5 steps rendered much better performance, yet was not as good as GrowNet’s results. The stair-like loss curve in the Figure [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") evidently displays the influence of the corrective step on model training.

![Refer to caption](/html/2002.07971/assets/x3.png)


Figure 3: Boosting rate evolution

Dynamic boost rate. Within the corrective step, we are able to dynamically update the boost rate αtsubscript𝛼𝑡\alpha\_{t} (at stage t𝑡t). Taking this measure saved us from tuning one more parameter as well as yielded a mild performance increase in all tasks. Moreover, the model obtained better training loss convergence, compared to the fixed boost rate version (see Fig. [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet")). In our setup, starting with α0=1subscript𝛼01\alpha\_{0}=1, the boost rate is automatically updated each time the corrective step is executed (see Fig. [3](#S6.F3 "Figure 3 ‣ 6.2 Analyzing corrective step ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet")) . Results of the best constant boost rate (αt=0.1subscript𝛼𝑡0.1\alpha\_{t}=0.1), coarsely tuned in a set of {0.01,0.1,1}0.010.11\{0.01,0.1,1\}, are reported in fifth column of Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet").

### 6.3 First order statistics vs second order statistics

In this experiment, we explored the impact of first and second order statistics on model performance as well as on the convergence of training loss. As the forth column of Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") displays, using the second order (third column in the Table [4](#S6.T4 "Table 4 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet")) renders a slight performance boost over the first order in classification and almost 2%percent22\% increase in learning to rank task. Figure [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") displays the effects of first and second order statistics on training loss. The final model (with second order statistics) again shows slightly better convergence on classification yet the difference is more apparent on ranking. As the learning rate is decreased by a rate of 1/2121/2 per 151515 weak learners, sudden drops are observed in classification loss curve at 15t​hsuperscript15𝑡ℎ15^{th} and 30t​hsuperscript30𝑡ℎ30^{th} stages in Figure [2](#S6.F2 "Figure 2 ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet").

### 6.4 Analyzing the effect of hidden layers

As the literature suggests, boosting algorithms work best with weak learners, thus we utilized a shallow neural network with two hidden layers as a weak predictor for our model. While adding more hidden layers yields stronger predictors, they are not weak learners anymore. To explore this weak learner limit on the number of hidden layers, we assayed weak learners with 1, 2, 3, and 4 hidden layers.
Each hidden layer had 16 units. Although weak learners with more hidden layers render better training loss convergence as expected, the overall model starts saturating on performance and overfitting. Weak learners with 1 and 2 hidden layers attain the best scores, yet the latter one outperforms the former. The worst test AUC score is from the model with 4 hidden layers (See Fig 6 in Supp. material).

![Refer to caption](/html/2002.07971/assets/x4.png)


Figure 4: Effect of # neurons on classification performance

Altering the number of hidden units has a lesser effect on performance. To illustrate the impact of hidden layer dimensions, we tested the final model (weak learner with two hidden layers) with various hidden units. Higgs data has 28 features and we tested the model with 2, 4, 8, 16, 32, 64, 128 and 256 hidden units. The smaller the hidden layer dimension is, the less information propagation the weak learners get. On the other hand, having a large number of units also leads to overfitting after a certain point. Figure [4](#S6.F4 "Figure 4 ‣ 6.4 Analyzing the effect of hidden layers ‣ 6 Ablation study ‣ Gradient Boosting Neural Networks: GrowNet") displays test AUC scores from this experiment on Higgs 1M data. The highest AUC of 0.84780.84780.8478 is achieved with 128 units, yet the performance suffers when the number is increased to 256.

### 6.5 GrowNet versus DNN

One might ask what would happen if we just combine all these shallow networks into one deep neural network. There are a couple of issues with this approach: (1) it is very time-consuming to tune the parameters of the DNN, such as the number of hidden layers, the number of units in each hidden layer, the overall architecture, batch normalization, dropout level, and etc., (2) DNNs require a huge computational power and in general run slower. We compared our model (with 30 weak learners) against DNN with 5, 10 , 20, and 30 hidden-layer configurations. The best DNN (with 10 hidden layers) produced 0.83420.83420.8342 on Higgs 1M data in 1000 epochs, and each epoch took 111111 seconds. The DNN achieved this score (its best) at epoch 900. GrowNet rendered 0.84010.84010.8401 AUC on the same configuration with 30 weak learners. The average stage training time, including the corrective step, took 50 seconds. Both models are run on the same machine with NVIDIA Tesla V100 (16GB) GPU. We find that GrowNet has a clear advantage over stacked DNNs on all these aspects.

Further details and illustrations from the ablation study and the code are provided in supplementary material.

## 7 Conclusion

In this work, we propose GrowNet, a novel approach to leverage shallow neural networks as “weak learners" in a gradient boosting framework. This flexible network structure allows us to perform multiple machine learning tasks under a unified framework while incorporating second order statistics, corrective step and dynamic boost rate to remedy the pitfalls of traditional gradient boosting decision tree. Ablation study is conducted to explore the limits of neural networks as weak learners in the boosting paradigm and analyze the effects of each GrowNet component on the model performance and convergence. We show that the proposed model achieves better performance in regression, classification and learning to rank on multiple datasets, compared to state-of-the-art boosting methods. We further demonstrate that GrowNet is a better alternative to DNNs in these tasks as it yields better performance, requires less training time and is much easier to tune.

## Broader Impact

GrowNet describes a novel boosting framework, which could be applied to a wide range of tasks and application domains, not limited to classification, regression and learning to rank, which are discussed in this paper. Our research could help improve the performance of these tasks and applications in practice.

While there are general impacts of our work, similar to those of the popular boosting methods [[6](#bib.bib6), [7](#bib.bib7)], we focus on the impact of the ease of employing boosted neural networks in practice. The potential benefits are improved predictions in a lesser amount of time (instead of a longer time to comprise application-specific algorithms) in a wide range of applications. For example, in healthcare, better predictions result in better diagnosis, which could save lives; in social domains, better predictions improve the quality of our lives; the list of tasks and applications, where our method could be adapted to, goes on. While there are undoubtedly benefits, there are many concerns and risks of irresponsible uses. This is even more important in today’s environment, where the issues such as bias, discrimination, privacy, etc… increasingly become more serious. Better predictions should not result in bias and discriminate decisions and invasion of privacy.

To mitigate these risks and concerns, we encourage the research community and policy makers to understand and evaluate the specific impacts of more and more powerful, ready-made Artificial Intelligent algorithms. Here, we need to understand the risks and derive the appropriate policies but should not hinder research activities when there are clearly beneficial implications.

## References

* Bao et al. [2019]

  Bao, W., Lai, W.-S., Ma, C., Zhang, X., Gao, Z., and Yang, M.-H.
  Depth-aware video frame interpolation.
  *2019 IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR)*, pp.  3698–3707, 2019.
* Becker et al. [2013]

  Becker, C. J., Rigamonti, R., Lepetit, V., and Fua, P.
  Kernelboost: Supervised learning of image features for
  classification.
  In *Technical Report*, 2013.
* Bengio et al. [2007]

  Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H.
  Greedy layer-wise training of deep networks.
  In *Proceedings of the 19th International Conference on Neural
  Information Processing Systems*, 2007.
* Burges [2010]

  Burges, C. J. C.
  From ranknet to lambdarank to lambdamart: An overview.
  In *Microsoft Research Technical Report*, 2010.
* Chapelle & Chang [2011]

  Chapelle, O. and Chang, Y.
  Yahoo! learning to rank challenge overview.
  *Journal of Machine Learning Research - W & CP*, 14:1–24, 2011.
* Chen & Guestrin [2016]

  Chen, T. and Guestrin, C.
  Xgboost: A scalable tree boosting system.
  In *Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, pp.  785–794. ACM, 2016.
* Cortes et al. [2017]

  Cortes, C., Gonzalvo, X., Kuznetsov, V., Mohri, M., and Yang, S.
  AdaNet: Adaptive structural learning of artificial neural
  networks.
  In *Proceedings of the 34th International Conference on Machine
  Learning*, 2017.
* Fahlman & Lebiere [1990]

  Fahlman, S. E. and Lebiere, C.
  The cascade-correlation learning architecture.
  In *NIPS*, 1990.
* Freund [1995]

  Freund, Y.
  Boosting a weak learning algorithm by majority.
  *Information and Computation*, pp.  256–285, 1995.
* Friedman et al. [2000]

  Friedman, J., Hastie, T., and Tibshirani, R.
  Additive logistic regression: A statistical view of boosting.
  *The Annals of Statistics*, 28:337–407, 2000.
* Friedman [2001]

  Friedman, J. H.
  Greedy function approximation: a gradient boosting machine.
  *Annals of statistics*, pp.  1189–1232, 2001.
* Goodfellow et al. [2014]

  Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair,
  S., Courville, A., and Bengio, Y.
  Generative adversarial nets.
  In *Advances in neural information processing systems*, pp. 2672–2680, 2014.
* Hansen & Salamon [1990]

  Hansen, L. and Salamon, P.
  Neural network ensembles.
  *TPAMI*, 12:993–1001, 1990.
* He et al. [2016]

  He, K., Zhang, X., Ren, S., and Sun, J.
  Deep residual learning for image recognition.
  In *CVPR*, pp.  770–778, 2016.
* Hinton et al. [2006]

  Hinton, G. E., Osindero, S., and Teh, Y.
  A fast learning algorithm for deep belief nets.
  *Neural Computation*, 18:1527–1554, 2006.
* Huang et al. [2018]

  Huang, F., Ash, J., Langford, J., and Schapire, R.
  Learning deep ResNet blocks sequentially using boosting theory.
  In *Proceedings of the 35th International Conference on Machine
  Learning*, 2018.
* Ke et al. [2017]

  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu,
  T. Y.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In *NIPS*, 2017.
* McKinney et al. [2020]

  McKinney, S., Sieniek, M., and et. al, V. G.
  International evaluation of an ai system for breast cancer screening.
  *Nature*, 577:89–94, 2020.
* Moghimi et al. [2016]

  Moghimi, M., Belongie, S. J., Saberian, M. J., Yang, J., Vasconcelos, N., and
  Li, L.-J.
  Boosted convolutional neural networks.
  In *BMVC*, 2016.
* Opitz & Shavlik [1996]

  Opitz, D. and Shavlik, J.
  Actively searching for an effective neural network ensemble.
  *Connection Science*, 8:337–353, 1996.
* Perrone & Cooper [1993]

  Perrone, M. P. and Cooper, L. N.
  When networks disagree: Ensemble methods for hybrid neural networks.
  pp.  126–142. Chapman and Hall, 1993.
* Prokhorenkova et al. [2018]

  Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A., and Gulin, A.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Schwenk & Bengio [1997]

  Schwenk, H. and Bengio, Y.
  Training methods for adaptive boosting of neural networks for
  character recognition.
  In *NIPS*, 1997.
* Schwenk & Bengio [2000]

  Schwenk, H. and Bengio, Y.
  Boosting neural networks.
  *Neural Computation*, 12:1869–1887, 2000.
* Simeon et al. [2019]

  Simeon, S., David, M., Marco, D., and et. al, D. C.
  A multimodality test to guide the management of patients with a
  pancreatic cyst.
  *Science Translational Medicine*, 11(501), 2019.
  doi: 10.1126/scitranslmed.aav4772.
* Tu & Bai [2010]

  Tu, Z. and Bai, X.
  Auto-context and its application to high-level vision tasks and 3d
  brain image segmentation.
  *IEEE Transactions on Pattern Analysis and Machine
  Intelligence*, 32(10):1744–1757, 2010.
* Tyree et al. [2011]

  Tyree, S., Weinberger, K., Agrawal, K., and Paykin, J.
  Parallel boosted regression trees for web search ranking.
  In *20th International Conference on World Wide Web*, 2011.
* Veit et al. [2016]

  Veit, A., Wilber, M., and Belongie, S.
  Residual networks behave like ensembles of relatively shallow
  networks.
  In *NIPS*, 2016.
* Yang et al. [2019]

  Yang, Z., Dai, Z., Yang, Y., Carbonell, J. G., Salakhutdinov, R., and Le, Q. V.
  Xlnet: Generalized autoregressive pretraining for language
  understanding.
  In *NeurIPS*, 2019.
* Zhang et al. [2016]

  Zhang, F., Du, B., and Zhang, L.
  Scene classification via a gradient boosting random convolutional
  network framework.
  *IEEE Transactions on Geoscience and Remote Sensing*, 54, 2016.
* Zhang & Johnson [2014]

  Zhang, T. and Johnson, R.
  Learning nonlinear functions using regularized greedy forest.
  *IEEE Transactions on Pattern Analysis and Machine
  Intelligence*, 2014.
* Zoph et al. [2019]

  Zoph, B., Cubuk, E. D., Ghiasi, G., Lin, T.-Y., Shlens, J., and Le, Q. V.
  Learning data augmentation strategies for object detection.
  *ArXiv*, abs/1906.11172, 2019.

## Appendix A Additional Related Work

A few works [[19](#bib.bib19), [30](#bib.bib30)] have also been proposed to directly combine Gradient Boosting with Convolutional Neural Nets (CNN). The authors of [[30](#bib.bib30)] propose to train gradient boosting machine with CNN as a base learner by introducing a custom multi-class softmax loss function for a specific scene classification task in the remote sensing domain. The work in [[19](#bib.bib19)], on the other hand, focuses on training each CNN sequentially on the mistakes of the previous networks, similar to Adaboost to perform on solely image classification task. Our method is different from [[30](#bib.bib30), [19](#bib.bib19)] as it is a unified framework to perform various machine learning tasks, such as classification, regression and even learning to rank. Moreover, unlike those two methods, we leveraged a corrective step to update the previously added predictor parameters and achieved a significant performance boost.

## Appendix B Algorithm Pseudocode

The pseudo-code of training the weak learner is explained in Individual model training part (1) of algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Algorithm Pseudocode ‣ Gradient Boosting Neural Networks: GrowNet"). The second part of the algorithm describes the corrective step. The code is available at GitHub page: <https://github.com/sbadirli/GrowNet>.

Algorithm 1  Full GrowNet training

Input: f0​(x)=l​o​g​(n+n−)subscript𝑓0𝑥𝑙𝑜𝑔subscript𝑛subscript𝑛f\_{0}(x)=log(\frac{n\_{+}}{n\_{-}}), α0subscript𝛼0\alpha\_{0}, Training data 𝒟t​rsubscript𝒟𝑡𝑟\mathcal{D}\_{tr}

Output: GrowNet ℰℰ\mathcal{E}

for k=1𝑘1k=1 to M𝑀M do

# Part 1 - Individual model training

Initialize model fk​(x)subscript𝑓𝑘𝑥f\_{k}(x)

Calculate 1s​tsuperscript1𝑠𝑡1^{st} order grad.: gi=∂y^i(k−1)l​(yi,y^i(k−1)),subscript𝑔𝑖subscriptsuperscriptsubscript^𝑦𝑖𝑘1𝑙subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑘1g\_{i}=\partial\_{\hat{y}\_{i}^{(k-1)}}l(y\_{i},\hat{y}\_{i}^{(k-1)}), ∀xi∈𝒟t​rfor-allsubscript𝑥𝑖subscript𝒟𝑡𝑟\forall x\_{i}\in\mathcal{D}\_{tr}

Calculate 2n​dsuperscript2𝑛𝑑2^{nd} order grad.: hi=∂y^i(k−1)2l​(yi,y^i(k−1)),subscriptℎ𝑖subscriptsuperscript2superscriptsubscript^𝑦𝑖𝑘1𝑙subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑘1h\_{i}=\partial^{2}\_{\hat{y}\_{i}^{(k-1)}}l(y\_{i},\hat{y}\_{i}^{(k-1)}), ∀xi∈𝒟t​rfor-allsubscript𝑥𝑖subscript𝒟𝑡𝑟\forall x\_{i}\in\mathcal{D}\_{tr}

Train fk​(⋅)subscript𝑓𝑘⋅f\_{k}(\cdot) by least square regression on {xi,−gi/hi}subscript𝑥𝑖subscript𝑔𝑖subscriptℎ𝑖\{x\_{i},-g\_{i}/h\_{i}\}

Add the model fk​(x)subscript𝑓𝑘𝑥f\_{k}(x) into the GrowNet ℰℰ\mathcal{E}

# Part 2 - Corrective step

for e​p​o​c​h=1𝑒𝑝𝑜𝑐ℎ1epoch=1 to T𝑇T do

Calculate GrowNet output: y^i(k)=∑m=0kαm​fm​(xi)superscriptsubscript^𝑦𝑖𝑘superscriptsubscript𝑚0𝑘subscript𝛼𝑚subscript𝑓𝑚subscript𝑥𝑖\hat{y}\_{i}^{(k)}=\sum\_{m=0}^{k}\alpha\_{m}f\_{m}(x\_{i}), ∀xi∈𝒟t​rfor-allsubscript𝑥𝑖subscript𝒟𝑡𝑟\forall x\_{i}\in\mathcal{D}\_{tr}

Calculate Loss from GrowNet: ℒ=1n​∑inl​(yi,y^i(k))ℒ1𝑛superscriptsubscript𝑖𝑛𝑙subscript𝑦𝑖superscriptsubscript^𝑦𝑖𝑘\mathcal{L}=\frac{1}{n}\sum\_{i}^{n}l(y\_{i},\hat{y}\_{i}^{(k)})

Update model fmsubscript𝑓𝑚f\_{m} parameters through back-propagation ∀m∈{1,…​k}for-all𝑚1…𝑘\forall m\in\{1,...k\}

Update step size αksubscript𝛼𝑘\alpha\_{k} through back-propagation

end for

end for

## Appendix C Additional Dataset Statistics

We evaluate our model on 5 datasets from 3 different tasks. A brief description of these datasets are presented in Table [5](#A3.T5 "Table 5 ‣ Appendix C Additional Dataset Statistics ‣ Gradient Boosting Neural Networks: GrowNet").

We used Higgs Bozon dataset111<https://archive.ics.uci.edu/ml/datasets/HIGGS> for classification. Higgs data is created using Monte Carlo simulations on high energy physics events. It is a binary event classification data with 28 attributes.

For the regression task, 2 datasets from the UCI machine learning repository are selected. The first one is Computed Tomography (CT) slice localization data222<https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis> where the aim is to retrieve the location of CT slices on the axial axis. The data was constructed from a set of 53,500

5350053,500 CT images that were taken from 74 different patients (43 male, 31 female).

The second regression dataset is YearPredictionMSD333<https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD> data, a subset of Million Song dataset, from the UCI repository. The goal is to predict the release year of a song from its audio features. The songs are mostly western, commercial tracks ranging from 1922 to 2011, with a peak in the year 2000s.

We choose Yahoo LTRC dataset444<https://webscope.sandbox.yahoo.com/catalog.php?datatype=c> [[5](#bib.bib5)] for the learning to rank task as it is a well-know benchmark dataset and also is used in the XGBoost paper. This dataset has 20​K20𝐾20K queries, each associated with approximately 222222 documents. Train-test split from the original paper is preserved. The second benchmark ranking dataset we used is MSLR-WEB 10​K10𝐾10K555<http://research.microsoft.com/en-us/projects/mslr/>. The dataset contains 10​K10𝐾10K queries, each of which corresponds to a list of 100−200100200100-200 documents.

Table 5: Datasets used in the experiments and their brief description. The second and third columns marked as N, M represent number of samples and feature dimension of the dataset, respectively.

| Dataset | N | M | Task |
| --- | --- | --- | --- |
| Higgs Bozon | 10​M10𝑀10M | 282828 | Binary classification |
| Slice localization | 53​K53𝐾53K | 384384384 | Regression |
| Year prediction | 515​K515𝐾515K | 909090 | Regression |
| Yahoo LTRC | 473​K473𝐾473K | 700700700 | Learning to rank |
| MSLR-WEB 10K | 1.2​M1.2𝑀1.2M | 136136136 | Learning to rank |

## Appendix D Hyperparameters of GrowNet

Experiment Setup. All predictive functions added to the model are multilayer perceptrons with two hidden layers. More hidden layers degraded the performance as the model starts overfitting. We generally set the number of hidden layer units to roughly a third of, a half of or equal to the input feature dimension. 404040 additive functions were employed in the experiments for all three tasks, and number of weak predictors in test time is chosen by the validation results. From all the experiments, we observe that 30 weak learners are more than enough to get the best results before the model performance saturates. Early stopping or other heuristics can also be incorporated into the model to terminate the training before the model begin to overfit.

The boosting rate is initially set to 111 and automatically adjusted during corrective step. Depending on the dataset and the task at hand, it may be initially set to a lower number such as 0.10.10.1. In our experiments, we did not tune or alter the boost rate.

We trained each predictive function for just 111 epoch, and the entire model is also trained for 111 epoch during the corrective step using stochastic gradient descent with the Adam optimizer. The Adam optimizer is run with l2subscript𝑙2l\_{2} regularization at a rate of 0.0010.0010.001. Epoch numbers are increased to 2 for the ranking task as we used larger batch sizes. Increasing the epoch number does not contribute to the performance, and higher numbers cause overfitting. We also performed 2​D2𝐷2D batch normalization for the hidden layers. The batch size for classification was set to 2048 and the learning rate was set to 0.0050.0050.005. ReLU was used as the activation function for the penultimate layer, whereas Leaky ReLU was used for the hidden layers. For the ranking task, we replaced ReLU with ReLU6.

The source code is uploaded in a separate file.

## Appendix E XGBoost and AdaNet Tuning

### E.1 XGBoost Tuning

For XGBoost, we tuned the main parameters, including the number of trees, learning rate, maximum leaves and ℓ2subscriptℓ2\ell\_{2} regularization in the following range:

* •

  Number of trees: {64,128,256,512,1000}641282565121000\{64,128,256,512,1000\}
* •

  Learning rate: {0.05,0.1}0.050.1\{0.05,0.1\}
* •

  Maximum number of leaves: {128,256,512}128256512\{128,256,512\}
* •

  ℓ2subscriptℓ2\ell\_{2} regularization (lambda): {0,0.2}00.2\{0,0.2\}

We did not tune XGBoost on Yahoo LTR (ranking task) and Higgs (classification task) datasets as we used the results reported in the original XGBoost paper [[6](#bib.bib6)] as is.

### E.2 AdaNet Tuning

We tuned 3 main parameters for AdaNet: the learning rate, the number of sub-networks and the complexity regularization parameter (λ𝜆\lambda) within the following ranges:

* •

  Learning rate: {0.01,0.001,0.0001}0.010.0010.0001\{0.01,0.001,0.0001\}
* •

  AdaNet iterations (# subnetworks): {2,3,4}234\{2,3,4\}
* •

  Complexity regularizer λ𝜆\lambda: {0.01,0.001,0.0001}0.010.0010.0001\{0.01,0.001,0.0001\}

The model was first tuned with default mixture weights and λ=0𝜆0\lambda=0, as suggested in the authors’ Github page666<https://github.com/tensorflow/adanet/blob/master/adanet/examples/tutorials/adanet_objective.ipynb>. From this experiment, we got the optimal learning rate and number of sub-networks. Then using the learned parameters from the previous setting, AdaNet is again tuned to learn the mixture weights and regularization complexity parameter λ𝜆\lambda.

The model is trained for 30,000

3000030,000 epochs, and the number of neurons in layers is set to 512, following the results from AdaNet paper [[7](#bib.bib7)]. We also observed that the model with 512 neurons generally renders better performance.

### E.3 Classification on Higgs-1M

Following the same data split on Higgs data from the XGBoost paper [[6](#bib.bib6)], we created Higgs-1M data. Table [6](#A5.T6 "Table 6 ‣ E.3 Classification on Higgs-1M ‣ Appendix E XGBoost and AdaNet Tuning ‣ Gradient Boosting Neural Networks: GrowNet") reports the AUC scores on HIggs-1M data from GrowNet, AdaNet and XGBoost. GrowNet renders favorable results compared to XGBoost and 3%percent33\% increase over AdaNet result.

|  | GrowNet | AdaNet | XGBoost |
| --- | --- | --- | --- |
| AUC | 0.84010.8401\boldsymbol{0.8401} | 0.81430.81430.8143 | 0.83040.83040.8304 |

Table 6: Classification results on Higgs-1M data. The scores are in AUC-ROC.

## Appendix F Additional Illustrations for Ablation Study

Analogous to Figure 2 from the main text, Figure [5](#A6.F5 "Figure 5 ‣ Appendix F Additional Illustrations for Ablation Study ‣ Gradient Boosting Neural Networks: GrowNet") presents pairwise losses on Microsoft dataset from the ranking task.

![Refer to caption](/html/2002.07971/assets/x5.png)


Figure 5: Training loss visualization for the learning to rank task on MSLR dataset. We used pairwise loss.

Effect of hidden layers. Table [7](#A6.T7 "Table 7 ‣ Appendix F Additional Illustrations for Ablation Study ‣ Gradient Boosting Neural Networks: GrowNet") reports the results from the hidden-layer experiment. GrowNet final, employing weak learners with 2 hidden layers, got the best performance (AUC score is 0.84010.84010.8401). The model with a shallow network of 1 hidden layer as a weak learner obtains better performance (AUC of 0.83360.83360.8336) once the number of hidden units is increased from 16 to 32. The inverse effect on the model with weak learners of 3 or 4 hidden layers did not work as expected. That is, decreasing the number of neurons in the hidden layers for these predictive functions did not improve much the classification performance.

|  | GrowNet 1HL | GrowNet 2HL | GrowNet 3HL | GrowNet 4HL |
| --- | --- | --- | --- | --- |
| AUC | 0.82880.82880.8288 | 0.84010.84010.8401 | 0.81460.81460.8146 | 0.78010.78010.7801 |

Table 7: Results from hidden layer experiment.



![Refer to caption](/html/2002.07971/assets/x6.png)


(a) Classification training loss

![Refer to caption](/html/2002.07971/assets/x7.png)


(b) Classification test loss

Figure 6: Effect of hidden layers on model training and classification performance (AUC).

Details on DNN versus GrowNet
Both Deep Neural Network (DNN) models and Grownet are run on the same machine with NVIDIA Tesla V100 (16GB) GPU.

Unlike GrowNet, DNN performed better with SELU activation functions. We also applied batch normalization on the hidden layers of DNN. Each of DNN models run for 1000 epochs. The results are reported in Table [8](#A6.T8 "Table 8 ‣ Appendix F Additional Illustrations for Ablation Study ‣ Gradient Boosting Neural Networks: GrowNet"). The best performing DNN model has 10 hidden layers, and each epoch took approximately 12 seconds. The model reaches its best performance after epoch 900. GrowNet shows a clear advantage on both classification performance and training time.

Both methods, DNN and GrowNet are not fully optimized, thus their training time can slightly be improved. Figure [7](#A6.F7 "Figure 7 ‣ Appendix F Additional Illustrations for Ablation Study ‣ Gradient Boosting Neural Networks: GrowNet") displays the training time of GrowNet while adding new weak learners.
DNN with 30 hidden layers are implemented with Dropout(0.3), as without Dropout the model started to overfit immediately after a few epochs. That also explains very close training times of DNN with 20 and 30 layers.

| Models | DNN-5 | DNN-10 | DNN-20 | DNN-30 | GrowNet |
| --- | --- | --- | --- | --- | --- |
| Training time (sec) | 10.210.210.2 | 11.611.611.6 | 15.215.215.2 | 15.015.015.0 | 50.150.150.1 |
| AUC | 0.82880.82880.8288 | 0.83420.83420.8342 | 0.83380.83380.8338 | 0.83010.83010.8301 | 0.84010.84010.8401 |

Table 8: Training time and performance comparison between DNN and GrowNet on Higg-1M data. Training time for DNNs are average seconds per epoch and for GrowNet average seconds per stages.

![Refer to caption](/html/2002.07971/assets/x8.png)


Figure 7: Training time over iterations. As observed, training time is linearly correlated with number of weak learners.

[◄](/html/2002.07970)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2002.07971)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2002.07971)
[View original  
on arXiv](https://arxiv.org/abs/2002.07971)[►](/html/2002.07972)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 1 18:04:16 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
