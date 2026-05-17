---
arxiv: '1706.09516'
authors:
- Liudmila Prokhorenkova
- Gleb Gusev
- Aleksandr Vorobev
- Anna Veronika Dorogush
- Andrey Gulin
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'CatBoost: unbiased boosting with categorical features'
url: http://arxiv.org/abs/1706.09516v5
year: 2017
---

# CatBoost: unbiased boosting with categorical features

Liudmila Prokhorenkova1,2, Gleb Gusev1,2, Aleksandr Vorobev1,
  
Anna Veronika Dorogush1, Andrey Gulin1
  
1Yandex, Moscow, Russia
  
2Moscow Institute of Physics and Technology, Dolgoprudny, Russia
  
{ostroumova-la, gleb57, alvor88, annaveronika, gulin}@yandex-team.ru

###### Abstract

This paper presents the key algorithmic techniques behind CatBoost, a new gradient boosting toolkit. Their combination leads to CatBoost outperforming other publicly available boosting implementations in terms of quality on a variety of datasets. Two critical algorithmic advances introduced in CatBoost are the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features. Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing implementations of gradient boosting algorithms. In this paper, we provide a detailed analysis of this problem and demonstrate that proposed algorithms solve it effectively, leading to excellent empirical results.

## 1 Introduction

Gradient boosting is a powerful machine-learning technique that achieves state-of-the-art results in a variety of practical tasks. For many years, it has remained the primary method for learning problems with heterogeneous features, noisy data, and complex dependencies: web search, recommendation systems, weather forecasting, and many others [[5](#bib.bib5), [26](#bib.bib26), [29](#bib.bib29), [32](#bib.bib32)]. Gradient boosting is essentially a process of constructing an ensemble predictor by performing gradient descent in a functional space. It is backed by solid theoretical results that explain how strong predictors can be built by iteratively combining weaker models (base predictors) in a greedy manner [[17](#bib.bib17)].

We show in this paper that all existing implementations of gradient boosting face the following statistical issue. A prediction model F𝐹F obtained after several steps of boosting relies on the targets of all training examples. We demonstrate that this actually leads to a shift of the distribution of F​(𝐱k)∣𝐱kconditional𝐹subscript𝐱𝑘subscript𝐱𝑘F(\mathbf{x}\_{k})\mid\mathbf{x}\_{k} for a training example 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} from the distribution of F​(𝐱)∣𝐱conditional𝐹𝐱𝐱F(\mathbf{x})\mid\mathbf{x} for a test example 𝐱𝐱\mathbf{x}. This finally leads to a prediction shift of the learned model. We identify this problem as a special kind of target leakage in Section [4](#S4 "4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features"). Further, there is a similar issue in standard algorithms of preprocessing categorical features. One of the most effective ways [[6](#bib.bib6), [25](#bib.bib25)] to use them in gradient boosting is converting categories to their target statistics. A target statistic is a simple statistical model itself, and it can also cause target leakage and a prediction shift. We analyze this in Section [3](#S3 "3 Categorical features ‣ CatBoost: unbiased boosting with categorical features").

In this paper, we propose ordering principle to solve both problems. Relying on it, we derive ordered boosting, a modification of standard gradient boosting algorithm, which avoids target leakage (Section [4](#S4 "4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")), and a new algorithm for processing categorical features (Section [3](#S3 "3 Categorical features ‣ CatBoost: unbiased boosting with categorical features")). Their combination is implemented as an open-source library111<https://github.com/catboost/catboost> called CatBoost (for “Categorical Boosting”), which outperforms the existing state-of-the-art implementations of gradient boosted decision trees — XGBoost [[8](#bib.bib8)] and LightGBM [[16](#bib.bib16)] — on a diverse set of popular machine learning tasks (see Section [6](#S6 "6 Experiments ‣ CatBoost: unbiased boosting with categorical features")).

## 2 Background

Assume we observe a dataset of examples 𝒟={(𝐱k,yk)}k=1..n\mathcal{D}=\{(\mathbf{x}\_{k},y\_{k})\}\_{k=1..n}, where 𝐱k=(xk1,…,xkm)subscript𝐱𝑘superscriptsubscript𝑥𝑘1…superscriptsubscript𝑥𝑘𝑚\mathbf{x}\_{k}=(x\_{k}^{1},\ldots,x\_{k}^{m}) is a random vector of m𝑚m features and yk∈ℝsubscript𝑦𝑘ℝy\_{k}\in\mathbb{R} is a target, which can be either binary or a numerical response. Examples (𝐱k,yk)subscript𝐱𝑘subscript𝑦𝑘(\mathbf{x}\_{k},y\_{k}) are independent and identically distributed according to some unknown distribution P​(⋅,⋅)𝑃⋅⋅P(\cdot,\cdot). The goal of a learning task is to train a function F:ℝm→ℝ:𝐹→superscriptℝ𝑚ℝF\colon\mathbb{R}^{m}\to\mathbb{R} which minimizes the expected loss ℒ​(F):=𝔼​L​(y,F​(𝐱))assignℒ𝐹𝔼𝐿𝑦𝐹𝐱\mathcal{L}(F):=\mathbb{E}L(y,F(\mathbf{x})). Here L​(⋅,⋅)𝐿⋅⋅L(\cdot,\cdot) is a smooth loss function and (𝐱,y)𝐱𝑦(\mathbf{x},y) is a test example sampled from P𝑃P independently of the training set 𝒟𝒟\mathcal{D}.

A gradient boosting procedure [[12](#bib.bib12)] builds iteratively a sequence of approximations Ft:ℝm→ℝ:superscript𝐹𝑡→superscriptℝ𝑚ℝF^{t}\colon\mathbb{R}^{m}\to\mathbb{R}, t=0,1,…𝑡

01…t=0,1,\ldots in a greedy fashion. Namely, Ftsuperscript𝐹𝑡F^{t} is obtained from the previous approximation Ft−1superscript𝐹𝑡1F^{t-1} in an additive manner: Ft=Ft−1+α​htsuperscript𝐹𝑡superscript𝐹𝑡1𝛼superscriptℎ𝑡F^{t}=F^{t-1}+\alpha h^{t}, where α𝛼\alpha is a step size and function ht:ℝm→ℝ:superscriptℎ𝑡→superscriptℝ𝑚ℝh^{t}\colon\mathbb{R}^{m}\to\mathbb{R} (a base predictor) is chosen from a family of functions H𝐻H in order to minimize the expected loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht=arg​minh∈H⁡ℒ​(Ft−1+h)=arg​minh∈H⁡𝔼​L​(y,Ft−1​(𝐱)+h​(𝐱)).superscriptℎ𝑡subscriptargminℎ𝐻ℒsuperscript𝐹𝑡1ℎsubscriptargminℎ𝐻𝔼𝐿𝑦superscript𝐹𝑡1𝐱ℎ𝐱h^{t}=\operatorname\*{arg\,min}\_{h\in H}\mathcal{L}(F^{t-1}+h)=\operatorname\*{arg\,min}\_{h\in H}\mathbb{E}L(y,F^{t-1}(\mathbf{x})+h(\mathbf{x})). |  | (1) |

The minimization problem is usually approached by the Newton method using a second–order approximation of ℒ​(Ft−1+ht)ℒsuperscript𝐹𝑡1superscriptℎ𝑡\mathcal{L}(F^{t-1}+h^{t}) at Ft−1superscript𝐹𝑡1F^{t-1} or by taking a (negative) gradient step. Both methods are kinds of functional gradient descent [[10](#bib.bib10), [24](#bib.bib24)]. In particular, the gradient step htsuperscriptℎ𝑡h^{t} is chosen in such a way that ht​(𝐱)superscriptℎ𝑡𝐱h^{t}(\mathbf{x}) approximates −gt​(𝐱,y)superscript𝑔𝑡𝐱𝑦-g^{t}(\mathbf{x},y), where gt​(𝐱,y):=∂L​(y,s)∂s|s=Ft−1​(𝐱)assignsuperscript𝑔𝑡𝐱𝑦evaluated-at𝐿𝑦𝑠𝑠𝑠superscript𝐹𝑡1𝐱g^{t}(\mathbf{x},y):=\frac{\partial L(y,s)}{\partial s}\big{|}\_{s=F^{t-1}(\mathbf{x})}. Usually, the least-squares approximation is used:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht=arg​minh∈H⁡𝔼​(−gt​(𝐱,y)−h​(𝐱))2.superscriptℎ𝑡subscriptargminℎ𝐻𝔼superscriptsuperscript𝑔𝑡𝐱𝑦ℎ𝐱2h^{t}=\operatorname\*{arg\,min}\_{h\in H}\mathbb{E}\left(-g^{t}(\mathbf{x},y)-h(\mathbf{x})\right)^{2}. |  | (2) |

CatBoost is an implementation of gradient boosting, which uses binary decision trees as base predictors. A decision tree [[4](#bib.bib4), [10](#bib.bib10), [27](#bib.bib27)] is a model built by a recursive partition of the feature space ℝmsuperscriptℝ𝑚\mathbb{R}^{m} into several disjoint regions (tree nodes) according to the values of some splitting attributes a𝑎a. Attributes are usually binary variables that identify that some feature xksuperscript𝑥𝑘x^{k} exceeds some threshold t𝑡t, that is, a=𝟙{xk>t}𝑎subscript1superscript𝑥𝑘𝑡a=\mathbbm{1}\_{\{x^{k}>t\}}, where xksuperscript𝑥𝑘x^{k} is either numerical or binary feature, in the latter case t=0.5𝑡0.5t=0.5.222Alternatively, non-binary splits can be used, e.g., a region can be split according to all values of a categorical feature. However, such splits, compared to binary ones, would lead to either shallow trees (unable to capture complex dependencies) or to very complex trees with exponential number of terminal nodes (having weaker target statistics in each of them). According to [[4](#bib.bib4)], the tree complexity has a crucial effect on the accuracy of the model and less complex trees are less prone to overfitting. Each final region (leaf of the tree) is assigned to a value, which is an estimate of the response y𝑦y in the region for the regression task or the predicted class label in the case of classification problem.333In a regression task, splitting attributes and leaf values are usually chosen by the least–squares criterion. Note that, in gradient boosting, a tree is constructed to approximate the negative gradient (see Equation ([2](#S2.E2 "In 2 Background ‣ CatBoost: unbiased boosting with categorical features"))), so it solves a regression problem. In this way, a decision tree hℎh can be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | h​(𝐱)=∑j=1Jbj​𝟙{𝐱∈Rj},ℎ𝐱superscriptsubscript𝑗1𝐽subscript𝑏𝑗subscript1𝐱subscript𝑅𝑗h(\mathbf{x})=\sum\_{j=1}^{J}b\_{j}\mathbbm{1}\_{\{\mathbf{x}\in R\_{j}\}}, |  | (3) |

where Rjsubscript𝑅𝑗R\_{j} are the disjoint regions corresponding to the leaves of the tree.

## 3 Categorical features

### 3.1 Related work on categorical features

A categorical feature is one with a discrete set of values called categories that are not comparable to each other. One popular technique for dealing with categorical features in boosted trees is one-hot encoding [[7](#bib.bib7), [25](#bib.bib25)], i.e., for each category, adding a new binary feature indicating it. However, in the case of high cardinality features (like, e.g., “user ID” feature), such technique leads to infeasibly large number of new features. To address this issue, one can group categories into a limited number of clusters and then apply one-hot encoding. A popular method is to group categories by target statistics (TS) that estimate expected target value in each category. Micci-Barreca [[25](#bib.bib25)] proposed to consider TS as a new numerical feature instead. Importantly, among all possible partitions of categories into two sets, an optimal split on the training data in terms of logloss, Gini index, MSE can be found among thresholds for the numerical TS feature [[4](#bib.bib4), Section 4.2.2] [[11](#bib.bib11), Section 9.2.4]. In LightGBM [[20](#bib.bib20)], categorical features are converted to gradient statistics at each step of gradient boosting. Though providing important information for building a tree, this approach can dramatically increase (i) computation time, since it calculates statistics for each categorical value at each step, and (ii) memory consumption to store which category belongs to which node for each split based on a categorical feature. To overcome this issue, LightGBM groups tail categories into one cluster [[21](#bib.bib21)] and thus looses part of information. Besides, the authors claim that it is still better to convert categorical features with high cardinality to numerical features [[19](#bib.bib19)]. Note that TS features require calculating and storing only one number per one category.

Thus, using TS as new numerical features seems to be the most efficient method of handling categorical features with minimum information loss. TS are widely-used, e.g., in the click prediction task (click-through rates) [[1](#bib.bib1), [15](#bib.bib15), [18](#bib.bib18), [22](#bib.bib22)], where such categorical features as user, region, ad, publisher play a crucial role. We further focus on ways to calculate TS and leave one-hot encoding and gradient statistics out of the scope of the current paper. At the same time, we believe that the ordering principle proposed in this paper is also effective for gradient statistics.

### 3.2 Target statistics

As discussed in Section [3.1](#S3.SS1 "3.1 Related work on categorical features ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features"), an effective and efficient way to deal with a categorical feature i𝑖i is to substitute the category xkisuperscriptsubscript𝑥𝑘𝑖x\_{k}^{i} of k𝑘k-th training example with one numeric feature equal to some target statistic (TS) x^kisuperscriptsubscript^𝑥𝑘𝑖\hat{x}\_{k}^{i}. Commonly, it estimates the expected target y𝑦y conditioned by the category: x^ki≈𝔼​(y∣xi=xki)superscriptsubscript^𝑥𝑘𝑖𝔼conditional𝑦superscript𝑥𝑖superscriptsubscript𝑥𝑘𝑖\hat{x}\_{k}^{i}\approx\mathbb{E}(y\mid x^{i}=x\_{k}^{i}).

##### Greedy TS

A straightforward approach is to estimate 𝔼​(y∣xi=xki)𝔼conditional𝑦superscript𝑥𝑖superscriptsubscript𝑥𝑘𝑖\mathbb{E}(y\mid x^{i}=x\_{k}^{i}) as the average value of y𝑦y over the training examples with the same category xkisuperscriptsubscript𝑥𝑘𝑖x\_{k}^{i} [[25](#bib.bib25)]. This estimate is noisy for low-frequency categories, and one usually smoothes it by some prior p𝑝p:

|  |  |  |  |
| --- | --- | --- | --- |
|  | x^ki=∑j=1n𝟙{xji=xki}⋅yj+a​p∑j=1n𝟙{xji=xki}+a,superscriptsubscript^𝑥𝑘𝑖superscriptsubscript𝑗1𝑛⋅subscript1superscriptsubscript𝑥𝑗𝑖superscriptsubscript𝑥𝑘𝑖subscript𝑦𝑗𝑎𝑝superscriptsubscript𝑗1𝑛subscript1superscriptsubscript𝑥𝑗𝑖superscriptsubscript𝑥𝑘𝑖𝑎\hat{x}\_{k}^{i}=\frac{\sum\_{j=1}^{n}{\mathbbm{1}\_{\{x\_{j}^{i}=x\_{k}^{i}\}}\cdot y\_{j}}+a\,p}{\sum\_{j=1}^{n}\mathbbm{1}\_{\{x\_{j}^{i}=x\_{k}^{i}\}}+a}\,, |  | (4) |

where a>0𝑎0a>0 is a parameter. A common setting for p𝑝p is the average target value in the dataset [[25](#bib.bib25)].

The problem of such greedy approach is target leakage: feature x^kisuperscriptsubscript^𝑥𝑘𝑖\hat{x}\_{k}^{i} is computed using yksubscript𝑦𝑘y\_{k}, the target of 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k}. This leads to a conditional shift [[30](#bib.bib30)]: the distribution of x^i|yconditionalsuperscript^𝑥𝑖𝑦\hat{x}^{i}|y differs for training and test examples. The following extreme example illustrates how dramatically this may affect the generalization error of the learned model. Assume i𝑖i-th feature is categorical, all its values are unique, and for each category A𝐴A, we have P​(y=1∣xi=A)=0.5P𝑦conditional1superscript𝑥𝑖𝐴0.5\mathrm{P}(y=1\mid x^{i}=A)=0.5 for a classification task. Then, in the training dataset, x^ki=yk+a​p1+asuperscriptsubscript^𝑥𝑘𝑖subscript𝑦𝑘𝑎𝑝1𝑎\hat{x}\_{k}^{i}=\frac{y\_{k}+ap}{1+a}, so it is sufficient to make only one split with threshold t=0.5+a​p1+a𝑡0.5𝑎𝑝1𝑎t=\frac{0.5+ap}{1+a} to perfectly classify all training examples. However, for all test examples, the value of the greedy TS is p𝑝p, and the obtained model predicts 00 for all of them if p<t𝑝𝑡p<t and predicts 111 otherwise, thus having accuracy 0.50.50.5 in both cases. To this end, we formulate the following desired property for TS:

* P1

  𝔼​(x^i∣y=v)𝔼conditionalsuperscript^𝑥𝑖𝑦𝑣\mathbb{E}(\hat{x}^{i}\mid y=v) = 𝔼​(x^ki∣yk=v)𝔼conditionalsuperscriptsubscript^𝑥𝑘𝑖subscript𝑦𝑘𝑣\mathbb{E}(\hat{x}\_{k}^{i}\mid y\_{k}=v), where (𝐱k,yk)subscript𝐱𝑘subscript𝑦𝑘(\mathbf{x}\_{k},y\_{k}) is the k𝑘k-th training example.

In our example above, 𝔼​(x^ki∣yk)=yk+a​p1+a𝔼conditionalsuperscriptsubscript^𝑥𝑘𝑖subscript𝑦𝑘subscript𝑦𝑘𝑎𝑝1𝑎\mathbb{E}(\hat{x}\_{k}^{i}\mid y\_{k})=\frac{y\_{k}+ap}{1+a} and 𝔼​(x^i∣y)=p𝔼conditionalsuperscript^𝑥𝑖𝑦𝑝\mathbb{E}(\hat{x}^{i}\mid y)=p are different.

There are several ways to avoid this conditional shift. Their general idea is to compute the TS for 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} on a subset of examples 𝒟k⊂𝒟∖{𝐱k}subscript𝒟𝑘𝒟subscript𝐱𝑘\mathcal{D}\_{k}\subset\mathcal{D}\setminus\{\mathbf{x}\_{k}\} excluding 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | x^ki=∑𝐱j∈𝒟k𝟙{xji=xki}⋅yj+a​p∑𝐱j∈𝒟k𝟙{xji=xki}+a.superscriptsubscript^𝑥𝑘𝑖subscriptsubscript𝐱𝑗subscript𝒟𝑘⋅subscript1superscriptsubscript𝑥𝑗𝑖superscriptsubscript𝑥𝑘𝑖subscript𝑦𝑗𝑎𝑝subscriptsubscript𝐱𝑗subscript𝒟𝑘subscript1superscriptsubscript𝑥𝑗𝑖superscriptsubscript𝑥𝑘𝑖𝑎\hat{x}\_{k}^{i}=\frac{\sum\_{\mathbf{x}\_{j}\in\mathcal{D}\_{k}}{\mathbbm{1}\_{\{x\_{j}^{i}=x\_{k}^{i}\}}\cdot y\_{j}}+a\,p}{\sum\_{\mathbf{x}\_{j}\in\mathcal{D}\_{k}}\mathbbm{1}\_{\{x\_{j}^{i}=x\_{k}^{i}\}}+a}\,. |  | (5) |

##### Holdout TS

One way is to partition the training dataset into two parts 𝒟=𝒟^0⊔𝒟^1𝒟square-unionsubscript^𝒟0subscript^𝒟1\mathcal{D}=\hat{\mathcal{D}}\_{0}\sqcup\hat{\mathcal{D}}\_{1} and use 𝒟k=𝒟^0subscript𝒟𝑘subscript^𝒟0\mathcal{D}\_{k}=\hat{\mathcal{D}}\_{0} for calculating the TS according to ([5](#S3.E5 "In Greedy TS ‣ 3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features")) and 𝒟^1subscript^𝒟1\hat{\mathcal{D}}\_{1} for training (e.g., applied in [[8](#bib.bib8)] for Criteo dataset). Though such holdout TS satisfies P1, this approach significantly reduces the amount of data used both for training the model and calculating the TS. So, it violates the following desired property:

* P2

  Effective usage of all training data for calculating TS features and for learning a model.

##### Leave-one-out TS

At first glance, a leave-one-out technique might work well: take 𝒟k=𝒟∖𝐱ksubscript𝒟𝑘𝒟subscript𝐱𝑘\mathcal{D}\_{k}=\mathcal{D}\setminus\mathbf{x}\_{k} for training examples 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} and 𝒟k=𝒟subscript𝒟𝑘𝒟\mathcal{D}\_{k}=\mathcal{D} for test ones [[31](#bib.bib31)]. Surprisingly, it does not prevent target leakage. Indeed, consider a constant categorical feature: xki=Asuperscriptsubscript𝑥𝑘𝑖𝐴x\_{k}^{i}=A for all examples. Let n+superscript𝑛n^{+} be the number of examples with y=1𝑦1y=1, then x^ki=n+−yk+a​pn−1+asuperscriptsubscript^𝑥𝑘𝑖superscript𝑛subscript𝑦𝑘𝑎𝑝𝑛1𝑎\hat{x}\_{k}^{i}=\frac{n^{+}-y\_{k}+a\,p}{n-1+a} and one can perfectly classify the training dataset by making a split with threshold t=n+−0.5+a​pn−1+a𝑡superscript𝑛0.5𝑎𝑝𝑛1𝑎t=\frac{n^{+}-0.5+a\,p}{n-1+a}.

##### Ordered TS

CatBoost uses a more effective strategy. It relies on the ordering principle, the central idea of the paper, and is inspired by online learning algorithms which get training examples sequentially in time [[1](#bib.bib1), [15](#bib.bib15), [18](#bib.bib18), [22](#bib.bib22)]). Clearly, the values of TS for each example rely only on the observed history. To adapt this idea to standard offline setting, we introduce an artificial “time”, i.e., a random permutation σ𝜎\sigma of the training examples. Then, for each example, we use all the available “history” to compute its TS, i.e., take 𝒟k={𝐱j:σ(j)<σ(k)\mathcal{D}\_{k}=\{\mathbf{x}\_{j}:\sigma(j)<\sigma(k)} in Equation ([5](#S3.E5 "In Greedy TS ‣ 3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features")) for a training example and 𝒟k=𝒟subscript𝒟𝑘𝒟\mathcal{D}\_{k}=\mathcal{D} for a test one. The obtained ordered TS satisfies the requirement P1 and allows to use all training data for learning the model (P2). Note that, if we use only one random permutation, then preceding examples have TS with much higher variance than subsequent ones. To this end, CatBoost uses different permutations for different steps of gradient boosting, see details in Section [5](#S5 "5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features").

## 4 Prediction shift and ordered boosting

### 4.1 Prediction shift

In this section, we reveal the problem of prediction shift in gradient boosting, which was neither recognized nor previously addressed. Like in case of TS, prediction shift is caused by a special kind of target leakage. Our solution is called ordered boosting and resembles the ordered TS method.

Let us go back to the gradient boosting procedure described in Section [2](#S2 "2 Background ‣ CatBoost: unbiased boosting with categorical features"). In practice, the expectation in ([2](#S2.E2 "In 2 Background ‣ CatBoost: unbiased boosting with categorical features")) is unknown and is usually approximated using the same dataset 𝒟𝒟\mathcal{D}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ht=arg​minh∈H⁡1n​∑k=1n(−gt​(𝐱k,yk)−h​(𝐱k))2.superscriptℎ𝑡subscriptargminℎ𝐻1𝑛superscriptsubscript𝑘1𝑛superscriptsuperscript𝑔𝑡subscript𝐱𝑘subscript𝑦𝑘ℎsubscript𝐱𝑘2h^{t}=\operatorname\*{arg\,min}\_{h\in H}\frac{1}{n}\sum\_{k=1}^{n}\left(-g^{t}(\mathbf{x}\_{k},y\_{k})-h(\mathbf{x}\_{k})\right)^{2}. |  | (6) |

Now we describe and analyze the following chain of shifts:

1. 1.

   the conditional distribution of the gradient gt​(𝐱k,yk)∣𝐱kconditionalsuperscript𝑔𝑡subscript𝐱𝑘subscript𝑦𝑘subscript𝐱𝑘g^{t}(\mathbf{x}\_{k},y\_{k})\mid\mathbf{x}\_{k} (accounting for randomness of 𝒟∖{𝐱k}𝒟subscript𝐱𝑘\mathcal{D}\setminus\{\mathbf{x}\_{k}\}) is shifted from that distribution on a test example gt​(𝐱,y)∣𝐱conditionalsuperscript𝑔𝑡𝐱𝑦𝐱g^{t}(\mathbf{x},y)\mid\mathbf{x};
2. 2.

   in turn, base predictor htsuperscriptℎ𝑡h^{t} defined by Equation ([6](#S4.E6 "In 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")) is biased from the solution of Equation ([2](#S2.E2 "In 2 Background ‣ CatBoost: unbiased boosting with categorical features"));
3. 3.

   this, finally, affects the generalization ability of the trained model Ftsuperscript𝐹𝑡F^{t}.

As in the case of TS, these problems are caused by the target leakage. Indeed, gradients used at each step are estimated using the target values of the same data points the current model Ft−1superscript𝐹𝑡1F^{t-1} was built on. However, the conditional distribution Ft−1​(𝐱k)∣𝐱kconditionalsuperscript𝐹𝑡1subscript𝐱𝑘subscript𝐱𝑘F^{t-1}(\mathbf{x}\_{k})\mid\mathbf{x}\_{k} for a training example 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} is shifted, in general, from the distribution Ft−1​(𝐱)∣𝐱conditionalsuperscript𝐹𝑡1𝐱𝐱F^{t-1}(\mathbf{x})\mid\mathbf{x} for a test example 𝐱𝐱\mathbf{x}. We call this a prediction shift.

##### Related work on prediction shift

The shift of gradient conditional distribution gt​(𝐱k,yk)∣𝐱kconditionalsuperscript𝑔𝑡subscript𝐱𝑘subscript𝑦𝑘subscript𝐱𝑘g^{t}(\mathbf{x}\_{k},y\_{k})\mid\mathbf{x}\_{k} was previously mentioned in papers on boosting [[3](#bib.bib3), [13](#bib.bib13)] but was not formally defined. Moreover, even the existence of non-zero shift was not proved theoretically. Based on the out-of-bag estimation [[2](#bib.bib2)], Breiman proposed iterated bagging [[3](#bib.bib3)] which constructs a bagged weak learner at each iteration on the basis of “out-of-bag” residual estimates. However, as we formally show in Appendix [E](#A5 "Appendix E Analysis of iterated bagging ‣ CatBoost: unbiased boosting with categorical features"), such residual estimates are still shifted. Besides, the bagging scheme increases learning time by factor of the number of data buckets. Subsampling of the dataset at each iteration proposed by Friedman [[13](#bib.bib13)] addresses the problem much more heuristically and also only alleviates it.

##### Analysis of prediction shift

We formally analyze the problem of prediction shift in a simple case of a regression task with the quadratic loss function L​(y,y^)=(y−y^)2𝐿𝑦^𝑦superscript𝑦^𝑦2L(y,\hat{y})=(y-\hat{y})^{2}.444We restrict the rest of Section [4](#S4 "4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") to this case, but the approaches of Section [4.2](#S4.SS2 "4.2 Ordered boosting ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") are applicable to other tasks. In this case, the negative gradient −gt​(𝐱k,yk)superscript𝑔𝑡subscript𝐱𝑘subscript𝑦𝑘-g^{t}(\mathbf{x}\_{k},y\_{k}) in Equation ([6](#S4.E6 "In 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")) can be substituted by the residual function rt−1​(𝐱k,yk):=yk−Ft−1​(𝐱k)assignsuperscript𝑟𝑡1subscript𝐱𝑘subscript𝑦𝑘subscript𝑦𝑘superscript𝐹𝑡1subscript𝐱𝑘r^{t-1}(\mathbf{x}\_{k},y\_{k}):=y\_{k}-F^{t-1}(\mathbf{x}\_{k}).555Here we removed the multiplier 2, what does not matter for further analysis. Assume we have m=2𝑚2m=2 features x1,x2

superscript𝑥1superscript𝑥2x^{1},x^{2} that are i.i.d. Bernoulli random variables with p=1/2𝑝12p=1/2 and y=f∗​(𝐱)=c1​x1+c2​x2𝑦superscript𝑓𝐱subscript𝑐1superscript𝑥1subscript𝑐2superscript𝑥2y=f^{\*}(\mathbf{x})=c\_{1}x^{1}+c\_{2}x^{2}. Assume we make N=2𝑁2N=2 steps of gradient boosting with decision stumps (trees of depth 1) and step size α=1𝛼1\alpha=1. We obtain a model F=F2=h1+h2𝐹superscript𝐹2superscriptℎ1superscriptℎ2F=F^{2}=h^{1}+h^{2}. W.l.o.g., we assume that h1superscriptℎ1h^{1} is based on x1superscript𝑥1x^{1} and h2superscriptℎ2h^{2} is based on x2superscript𝑥2x^{2}, what is typical for |c1|>|c2|subscript𝑐1subscript𝑐2|c\_{1}|>|c\_{2}| (here we set some asymmetry between x1superscript𝑥1x^{1} and x2superscript𝑥2x^{2}).

###### Theorem 1

1. If two independent samples 𝒟1subscript𝒟1\mathcal{D}\_{1} and 𝒟2subscript𝒟2\mathcal{D}\_{2} of size n𝑛n are used to estimate h1superscriptℎ1h^{1} and h2superscriptℎ2h^{2}, respectively, using Equation ([6](#S4.E6 "In 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")), then 𝔼𝒟1,𝒟2​F2​(𝐱)=f∗​(𝐱)+O​(1/2n)subscript𝔼

subscript𝒟1subscript𝒟2superscript𝐹2𝐱superscript𝑓𝐱𝑂1superscript2𝑛\mathbb{E}\_{\mathcal{D}\_{1},\mathcal{D}\_{2}}F^{2}(\mathbf{x})=f^{\*}(\mathbf{x})+O(1/2^{n}) for any 𝐱∈{0,1}2𝐱superscript012\mathbf{x}\in\{0,1\}^{2}.
  
2. If the same dataset 𝒟=𝒟1=𝒟2𝒟subscript𝒟1subscript𝒟2\mathcal{D}=\mathcal{D}\_{1}=\mathcal{D}\_{2} is used in Equation ([6](#S4.E6 "In 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")) for both h1superscriptℎ1h^{1} and h2superscriptℎ2h^{2}, then 𝔼𝒟​F2​(𝐱)=f∗​(𝐱)−1n−1​c2​(x2−12)+O​(1/2n)subscript𝔼𝒟superscript𝐹2𝐱superscript𝑓𝐱1𝑛1subscript𝑐2superscript𝑥212𝑂1superscript2𝑛\mathbb{E}\_{\mathcal{D}}F^{2}(\mathbf{x})=f^{\*}(\mathbf{x})-\frac{1}{n-1}c\_{2}(x^{2}-\frac{1}{2})+O(1/2^{n}).

This theorem means that the trained model is an unbiased estimate of the true dependence y=f∗​(𝐱)𝑦superscript𝑓𝐱y=f^{\*}(\mathbf{x}), when we use independent datasets at each gradient step.666Up to an exponentially small term, which occurs for a technical reason. On the other hand, if we use the same dataset at each step, we suffer from a bias −1n−1​c2​(x2−12)1𝑛1subscript𝑐2superscript𝑥212-\frac{1}{n-1}c\_{2}(x^{2}-\frac{1}{2}), which is inversely proportional to the data size n𝑛n. Also, the value of the bias can depend on the relation f∗superscript𝑓f^{\*}: in our example, it is proportional to c2subscript𝑐2c\_{2}. We track the chain of shifts for the second part of Theorem [1](#Thmthm1 "Theorem 1 ‣ Analysis of prediction shift ‣ 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") in a sketch of the proof below, while the full proof of Theorem [1](#Thmthm1 "Theorem 1 ‣ Analysis of prediction shift ‣ 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") is available in Appendix [A](#A1 "Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features").

Sketch of the proof.
Denote by ξs​tsubscript𝜉𝑠𝑡\xi\_{st}, s,t∈{0,1}

𝑠𝑡
01s,t\in\{0,1\}, the number of examples (𝐱k,yk)∈𝒟subscript𝐱𝑘subscript𝑦𝑘𝒟(\mathbf{x}\_{k},y\_{k})\in\mathcal{D} with 𝐱k=(s,t)subscript𝐱𝑘𝑠𝑡\mathbf{x}\_{k}=(s,t). We have h1​(s,t)=c1​s+c2​ξs​1ξs​0+ξs​1superscriptℎ1𝑠𝑡subscript𝑐1𝑠subscript𝑐2subscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1h^{1}(s,t)=c\_{1}s+\frac{c\_{2}\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}. Its expectation 𝔼​(h1​(𝐱))𝔼superscriptℎ1𝐱\mathbb{E}(h^{1}(\mathbf{x})) on a test example 𝐱𝐱\mathbf{x} equals c1​x1+c22subscript𝑐1superscript𝑥1subscript𝑐22c\_{1}x^{1}+\frac{c\_{2}}{2}. At the same time, the expectation 𝔼​(h1​(𝐱k))𝔼superscriptℎ1subscript𝐱𝑘\mathbb{E}(h^{1}(\mathbf{x}\_{k})) on a training example 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k} is different and equals (c1​x1+c22)−c2​(2​x2−1n)+O​(2−n)subscript𝑐1superscript𝑥1subscript𝑐22subscript𝑐22superscript𝑥21𝑛𝑂superscript2𝑛(c\_{1}x^{1}+\frac{c\_{2}}{2})-c\_{2}(\frac{2x^{2}-1}{n})+O(2^{-n}). That is, we experience a prediction shift of h1superscriptℎ1h^{1}. As a consequence, the expected value of h2​(𝐱)superscriptℎ2𝐱h^{2}(\mathbf{x}) is 𝔼​(h2​(𝐱))=c2​(x2−12)​(1−1n−1)+O​(2−n)𝔼superscriptℎ2𝐱subscript𝑐2superscript𝑥21211𝑛1𝑂superscript2𝑛\mathbb{E}(h^{2}(\mathbf{x}))=c\_{2}(x^{2}-\frac{1}{2})(1-\frac{1}{n-1})+O(2^{-n}) on a test example 𝐱𝐱\mathbf{x} and 𝔼​(h1​(𝐱)+h2​(𝐱))=f∗​(𝐱)−1n−1​c2​(x2−12)+O​(1/2n)𝔼superscriptℎ1𝐱superscriptℎ2𝐱superscript𝑓𝐱1𝑛1subscript𝑐2superscript𝑥212𝑂1superscript2𝑛\mathbb{E}(h^{1}(\mathbf{x})+h^{2}(\mathbf{x}))=f^{\*}(\mathbf{x})-\frac{1}{n-1}c\_{2}(x^{2}-\frac{1}{2})+O(1/2^{n}).
□□\Box

Finally, recall that greedy TS x^isuperscript^𝑥𝑖\hat{x}^{i} can be considered as a simple statistical model predicting the target y𝑦y and it suffers from a similar problem, conditional shift of x^ki∣ykconditionalsubscriptsuperscript^𝑥𝑖𝑘subscript𝑦𝑘\hat{x}^{i}\_{k}\mid y\_{k}, caused by the target leakage, i.e., using yksubscript𝑦𝑘y\_{k} to compute x^kisubscriptsuperscript^𝑥𝑖𝑘\hat{x}^{i}\_{k}.

### 4.2 Ordered boosting

Here we propose a boosting algorithm which does not suffer from the prediction shift problem described in Section [4.1](#S4.SS1 "4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features"). Assuming access to an unlimited amount of training data, we can easily construct such an algorithm. At each step of boosting, we sample a new dataset 𝒟tsubscript𝒟𝑡\mathcal{D}\_{t} independently and obtain unshifted residuals by applying the current model to new training examples. In practice, however, labeled data is limited. Assume that we learn a model with I𝐼I trees. To make the residual rI−1​(𝐱k,yk)superscript𝑟𝐼1subscript𝐱𝑘subscript𝑦𝑘r^{I-1}(\mathbf{x}\_{k},y\_{k}) unshifted, we need to have FI−1superscript𝐹𝐼1F^{I-1} trained without the example 𝐱ksubscript𝐱𝑘\mathbf{x}\_{k}. Since we need unbiased residuals for all training examples, no examples may be used for training FI−1superscript𝐹𝐼1F^{I-1}, which at first glance makes the training process impossible. However, it is possible to maintain a set of models differing by examples used for their training. Then, for calculating the residual on an example, we use a model trained without it. In order to construct such a set of models, we can use the ordering principle previously applied to TS in Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features"). To illustrate the idea, assume that we take one random permutation σ𝜎\sigma of the training examples and maintain n𝑛n different supporting models M1,…,Mn

subscript𝑀1…subscript𝑀𝑛M\_{1},\ldots,M\_{n} such that the model Misubscript𝑀𝑖M\_{i} is learned using only the first i𝑖i examples in the permutation. At each step, in order to obtain the residual for j𝑗j-th sample, we use the model Mj−1subscript𝑀𝑗1M\_{j-1} (see Figure [1](#S5.F1 "Figure 1 ‣ 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features")). The resulting Algorithm [1](#algorithm1 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features") is called ordered boosting below. Unfortunately, this algorithm is not feasible in most practical tasks due to the need of training n𝑛n different models, what increase the complexity and memory requirements by n𝑛n times. In CatBoost, we implemented a modification of this algorithm on the basis of the gradient boosting algorithm with decision trees as base predictors (GBDT) described in Section [5](#S5 "5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features").

##### Ordered boosting with categorical features

In Sections [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features") and [4.2](#S4.SS2 "4.2 Ordered boosting ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") we proposed to use random permutations σc​a​tsubscript𝜎𝑐𝑎𝑡\sigma\_{cat} and σb​o​o​s​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{boost} of training examples for the TS calculation and for ordered boosting, respectively. Combining them in one algorithm, we should take σc​a​t=σb​o​o​s​tsubscript𝜎𝑐𝑎𝑡subscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{cat}=\sigma\_{boost} to avoid prediction shift. This guarantees that target yisubscript𝑦𝑖y\_{i} is not used for training Misubscript𝑀𝑖M\_{i} (neither for the TS calculation, nor for the gradient estimation). See Appendix [F](#A6 "Appendix F Ordered boosting with categorical features ‣ CatBoost: unbiased boosting with categorical features") for theoretical guarantees. Empirical results confirming the importance of having σc​a​t=σb​o​o​s​tsubscript𝜎𝑐𝑎𝑡subscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{cat}=\sigma\_{boost} are presented in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features").

## 5 Practical implementation of ordered boosting

CatBoost has two boosting modes, Ordered and Plain. The latter mode is the standard GBDT algorithm with inbuilt ordered TS. The former mode presents an efficient modification of Algorithm [1](#algorithm1 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features"). A formal description of the algorithm is included in Appendix [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features"). In this section, we overview the most important implementation details.

Figure 1: Ordered boosting principle,
  
examples are ordered according to σ𝜎\sigma.

input :   {(𝐱k,yk)}k=1nsuperscriptsubscriptsubscript𝐱𝑘subscript𝑦𝑘𝑘1𝑛\{(\mathbf{x}\_{k},y\_{k})\}\_{k=1}^{n}, I𝐼I;

σ←←𝜎absent\sigma\leftarrow random permutation of [1,n]1𝑛[1,n] ;

Mi←0←subscript𝑀𝑖0M\_{i}\leftarrow 0 for i=1..ni=1..n;

for *t←1←𝑡1t\leftarrow 1 to I𝐼I* do

for *i←1←𝑖1i\leftarrow 1 to n𝑛n* do

ri←yi−Mσ​(i)−1​(𝐱i)←subscript𝑟𝑖subscript𝑦𝑖subscript𝑀𝜎𝑖1subscript𝐱𝑖r\_{i}\leftarrow y\_{i}-M\_{\sigma(i)-1}(\mathbf{x}\_{i});

for *i←1←𝑖1i\leftarrow 1 to n𝑛n* do

ΔM←LearnModel((𝐱j,rj):σ(j)≤i)\Delta M\leftarrow LearnModel((\mathbf{x}\_{j},r\_{j}):\sigma(j)\leq i);

Mi←Mi+Δ​M←subscript𝑀𝑖subscript𝑀𝑖Δ𝑀M\_{i}\leftarrow M\_{i}+\Delta M ;

return *Mnsubscript𝑀𝑛M\_{n}*

Algorithm 1 Ordered boosting

input :   M𝑀M, {(𝐱i,yi)}i=1nsuperscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n}, α𝛼\alpha, L𝐿L, {σi}i=1ssuperscriptsubscriptsubscript𝜎𝑖𝑖1𝑠\{\sigma\_{i}\}\_{i=1}^{s}, M​o​d​e𝑀𝑜𝑑𝑒Mode

g​r​a​d←C​a​l​c​G​r​a​d​i​e​n​t​(L,M,y)←𝑔𝑟𝑎𝑑𝐶𝑎𝑙𝑐𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡𝐿𝑀𝑦grad\leftarrow CalcGradient(L,M,y);

r←r​a​n​d​o​m​(1,s)←𝑟𝑟𝑎𝑛𝑑𝑜𝑚1𝑠r\leftarrow random(1,s);

if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

G←(gradr(i) for i=1..n)G\leftarrow(grad\_{r}(i)\mbox{ for }i=1..n);

if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

G←(gradr,σr​(i)−1(i) for i=1..n)G\leftarrow(grad\_{r,\sigma\_{r}(i)-1}(i)\mbox{ for }i=1..n);

T←←𝑇absentT\leftarrow empty tree;

foreach *step of top-down procedure* do

foreach *candidate split c𝑐c*  do

Tc←←subscript𝑇𝑐absentT\_{c}\leftarrow add split c𝑐c to T𝑇T;

if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

Δ(i)←avg(gradr(p)\Delta(i)\leftarrow\mathrm{avg}(grad\_{r}(p) for p:leafr(p)=leafr(i))p:\ leaf\_{r}(p)=leaf\_{r}(i))  for i=1..ni=1..n;

if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

Δ(i)←avg(gradr,σr​(i)−1(p)\Delta(i)\leftarrow\mathrm{avg}(grad\_{r,\sigma\_{r}(i)-1}(p) for p:leafr(p)=leafr(i),σr(p)<σr(i))p:\ leaf\_{r}(p)=leaf\_{r}(i),\sigma\_{r}(p)<\sigma\_{r}(i))  for i=1..ni=1..n;

l​o​s​s​(Tc)←cos⁡(Δ,G)←𝑙𝑜𝑠𝑠subscript𝑇𝑐Δ𝐺loss(T\_{c})\leftarrow\cos(\Delta,G)

T←arg​minTc⁡(l​o​s​s​(Tc))←𝑇subscriptargminsubscript𝑇𝑐𝑙𝑜𝑠𝑠subscript𝑇𝑐T\leftarrow\operatorname\*{arg\,min}\_{T\_{c}}(loss(T\_{c}))

if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

Mr′(i)←Mr′(i)−αavg(gradr′(p)M\_{r^{\prime}}(i)\leftarrow M\_{r^{\prime}}(i)-\alpha\,\mathrm{avg}(grad\_{r^{\prime}}(p) for p:leafr′(p)=leafr′(i))p:\ leaf\_{r^{\prime}}(p)=leaf\_{r^{\prime}}(i)) for r′=1..sr^{\prime}=1..s, i=1..ni=1..n;

if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

Mr′,j(i)←Mr′,j(i)−αavg(gradr′,j(p)M\_{r^{\prime},j}(i)\leftarrow M\_{r^{\prime},j}(i)-\alpha\,\mathrm{avg}(grad\_{r^{\prime},j}(p) for p:leafr′(p)=leafr′(i),σr′(p)≤j)p:\ leaf\_{r^{\prime}}(p)=leaf\_{r^{\prime}}(i),\sigma\_{r^{\prime}}(p)\leq j) for r′=1..sr^{\prime}=1..s, i=1..ni=1..n, j≥σr′​(i)−1𝑗subscript𝜎superscript𝑟′𝑖1j\geq\sigma\_{r^{\prime}}(i)-1;

return *T,M

𝑇𝑀T,M*

Algorithm 2 Building a tree in CatBoost

At the start, CatBoost generates s+1𝑠1s+1 independent random permutations of the training dataset. The permutations σ1,…,σs

subscript𝜎1…subscript𝜎𝑠\sigma\_{1},\ldots,\sigma\_{s} are used for evaluation of splits that define tree structures (i.e., the internal nodes), while σ0subscript𝜎0\sigma\_{0} serves for choosing the leaf values bjsubscript𝑏𝑗b\_{j} of the obtained trees (see Equation ([3](#S2.E3 "In 2 Background ‣ CatBoost: unbiased boosting with categorical features"))). For examples with short history in a given permutation, both TS and predictions used by ordered boosting (Mσ​(i)−1​(𝐱i)subscript𝑀𝜎𝑖1subscript𝐱𝑖M\_{\sigma(i)-1}(\mathbf{x}\_{i}) in Algorithm [1](#algorithm1 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features")) have a high variance. Therefore, using only one permutation may increase the variance of the final model predictions, while several permutations allow us to reduce this effect in a way we further describe. The advantage of several permutations is confirmed by our experiments in Section [6](#S6.SS0.SSS0.Px5 "Number of permutations ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features").

##### Building a tree

In CatBoost, base predictors are oblivious decision trees [[9](#bib.bib9), [14](#bib.bib14)] also called decision tables [[23](#bib.bib23)]. Term oblivious means that the same splitting criterion is used across an entire level of the tree. Such trees are balanced, less prone to overfitting, and allow speeding up execution at testing time significantly. The procedure of building a tree in CatBoost is described in Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features").

In the Ordered boosting mode, during the learning process, we maintain the supporting models Mr,jsubscript𝑀

𝑟𝑗M\_{r,j}, where Mr,j​(i)subscript𝑀

𝑟𝑗𝑖M\_{r,j}(i) is the current prediction for the i𝑖i-th example based on the first j𝑗j examples in the permutation σrsubscript𝜎𝑟\sigma\_{r}. At each iteration t𝑡t of the algorithm, we sample a random permutation σrsubscript𝜎𝑟\sigma\_{r} from {σ1,…,σs}subscript𝜎1…subscript𝜎𝑠\{\sigma\_{1},\ldots,\sigma\_{s}\} and construct a tree Ttsubscript𝑇𝑡T\_{t} on the basis of it. First, for categorical features, all TS are computed according to this permutation. Second, the permutation affects the tree learning procedure. Namely, based on Mr,j​(i)subscript𝑀

𝑟𝑗𝑖M\_{r,j}(i), we compute the corresponding gradients g​r​a​dr,j​(i)=∂L​(yi,s)∂s|s=Mr,j​(i)𝑔𝑟𝑎subscript𝑑

𝑟𝑗𝑖evaluated-at𝐿subscript𝑦𝑖𝑠𝑠𝑠subscript𝑀

𝑟𝑗𝑖grad\_{r,j}(i)=\frac{\partial L(y\_{i},s)}{\partial s}\big{|}\_{s=M\_{r,j}(i)}. Then, while constructing a tree, we approximate the gradient G𝐺G in terms of the cosine similarity
cos⁡(⋅,⋅)⋅⋅\cos(\cdot,\cdot), where, for each example i𝑖i, we take the gradient g​r​a​dr,σ​(i)−1​(i)𝑔𝑟𝑎subscript𝑑

𝑟𝜎𝑖1𝑖grad\_{r,\sigma(i)-1}(i) (it is based only on the previous examples in σrsubscript𝜎𝑟\sigma\_{r}). At the candidate splits evaluation step, the leaf value Δ​(i)Δ𝑖\Delta(i) for example i𝑖i is obtained individually by averaging the gradients g​r​a​dr,σr​(i)−1𝑔𝑟𝑎subscript𝑑

𝑟subscript𝜎𝑟𝑖1grad\_{r,\sigma\_{r}(i)-1} of the preceding examples p𝑝p lying in the same leaf l​e​a​fr​(i)𝑙𝑒𝑎subscript𝑓𝑟𝑖leaf\_{r}(i) the example i𝑖i belongs to. Note that l​e​a​fr​(i)𝑙𝑒𝑎subscript𝑓𝑟𝑖leaf\_{r}(i) depends on the chosen permutation σrsubscript𝜎𝑟\sigma\_{r}, because σrsubscript𝜎𝑟\sigma\_{r} can influence the values of ordered TS for example i𝑖i. When the tree structure Ttsubscript𝑇𝑡T\_{t} (i.e., the sequence of splitting attributes) is built, we use it to boost all the models Mr′,jsubscript𝑀

superscript𝑟′𝑗M\_{r^{\prime},j}. Let us stress that one common tree structure Ttsubscript𝑇𝑡T\_{t} is used for all the models, but this tree is added to different Mr′,jsubscript𝑀

superscript𝑟′𝑗M\_{r^{\prime},j} with different sets of leaf values depending on r′superscript𝑟′r^{\prime} and j𝑗j, as described in Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features").

The Plain boosting mode works similarly to a standard GBDT procedure, but, if categorical features are present, it maintains s𝑠s supporting models Mrsubscript𝑀𝑟M\_{r} corresponding to TS based on σ1,…,σs

subscript𝜎1…subscript𝜎𝑠\sigma\_{1},\ldots,\sigma\_{s}.

##### Choosing leaf values

Given all the trees constructed, the leaf values of the final model F𝐹F are calculated by the standard gradient boosting procedure equally for both modes. Training examples i𝑖i are matched to leaves l​e​a​f0​(i)𝑙𝑒𝑎subscript𝑓0𝑖leaf\_{0}(i), i.e., we use permutation σ0subscript𝜎0\sigma\_{0} to calculate TS here. When the final model F𝐹F is applied to a new example at testing time, we use TS calculated on the whole training data according to Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features").

##### Complexity

In our practical implementation, we use one important trick, which significantly reduces the computational complexity of the algorithm. Namely, in the Ordered mode, instead of O​(s​n2)𝑂𝑠superscript𝑛2O(s\,n^{2}) values Mr,j​(i)subscript𝑀

𝑟𝑗𝑖M\_{r,j}(i), we store and update only the values Mr,j′​(i):=Mr,2j​(i)assignsubscriptsuperscript𝑀′

𝑟𝑗𝑖subscript𝑀

𝑟superscript2𝑗𝑖M^{\prime}\_{r,j}(i):=M\_{r,2^{j}}(i) for j=1,…,⌈log2⁡n⌉𝑗

1…subscript2𝑛j=1,\ldots,\lceil\log\_{2}n\rceil and all i𝑖i with σr​(i)≤2j+1subscript𝜎𝑟𝑖superscript2𝑗1\sigma\_{r}(i)\leq 2^{j+1}, what reduces the number of maintained supporting predictions to O​(s​n)𝑂𝑠𝑛O(s\,n).
See Appendix [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features") for the pseudocode of this modification of Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features").

In Table [1](#S5.T1 "Table 1 ‣ Complexity ‣ 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features"), we present the computational complexity of different components of both CatBoost modes per one iteration (see Appendix [C.1](#A3.SS1 "C.1 Theoretical analysis ‣ Appendix C Time complexity analysis ‣ CatBoost: unbiased boosting with categorical features") for the proof). Here NT​S,tsubscript𝑁

𝑇𝑆𝑡N\_{TS,t} is the number of TS to be calculated at the iteration t𝑡t and C𝐶C is the set of candidate splits to be considered at the given iteration. It follows that our implementation of ordered boosting with decision trees has the same asymptotic complexity as the standard GBDT with ordered TS. In comparison with other types of TS (Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features")), ordered TS slow down by s𝑠s times the procedures C​a​l​c​G​r​a​d​i​e​n​t𝐶𝑎𝑙𝑐𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡CalcGradient, updating supporting models M𝑀M, and computation of TS.

Table 1: Computational complexity.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Procedure | CalcGradient | Build T𝑇T | Calc all bjtsubscriptsuperscript𝑏𝑡𝑗b^{t}\_{j} | Update M𝑀M | Calc ordered TS |
| Complexity for iteration t𝑡t | O​(s⋅n)𝑂⋅𝑠𝑛O(s\cdot n) | O​(|C|⋅n)𝑂⋅𝐶𝑛O(|C|\cdot n) | O​(n)𝑂𝑛O(n) | O​(s⋅n)𝑂⋅𝑠𝑛O(s\cdot n) | O​(NT​S,t⋅n)𝑂⋅subscript𝑁  𝑇𝑆𝑡𝑛O(N\_{TS,t}\cdot n) |

##### Feature combinations

Another important detail of CatBoost is using combinations of categorical features as additional categorical features which capture high-order dependencies like joint information of user ID and ad topic in the task of ad click prediction. The number of possible combinations grows exponentially with the number of categorical features in the dataset, and it is infeasible to process all of them. CatBoost constructs combinations in a greedy way. Namely, for each split of a tree, CatBoost combines (concatenates) all categorical features (and their combinations) already used for previous splits in the current tree with all categorical features in the dataset. Combinations are converted to TS on the fly.

##### Other important details

Finally, let us discuss two options of the CatBoost algorithm not covered above. The first one is subsampling of the dataset at each iteration of boosting procedure, as proposed by Friedman [[13](#bib.bib13)]. We claimed earlier in Section [4.1](#S4.SS1 "4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") that this approach alone cannot fully avoid the problem of prediction shift. However, since it has proved effective, we implemented it in both modes of CatBoost as a Bayesian bootstrap procedure. Specifically, before training a tree according to Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features"), we assign a weight wi=aitsubscript𝑤𝑖superscriptsubscript𝑎𝑖𝑡w\_{i}=a\_{i}^{t} to each example i𝑖i, where aitsuperscriptsubscript𝑎𝑖𝑡a\_{i}^{t} are generated according to the Bayesian bootstrap procedure (see [[28](#bib.bib28), Section 2]). These weights are used as multipliers for gradients g​r​a​dr​(i)𝑔𝑟𝑎subscript𝑑𝑟𝑖grad\_{r}(i) and g​r​a​dr,j​(i)𝑔𝑟𝑎subscript𝑑

𝑟𝑗𝑖grad\_{r,j}(i), when we calculate Δ​(i)Δ𝑖\Delta(i) and the components of the vector Δ−GΔ𝐺\Delta-G to define l​o​s​s​(Tc)𝑙𝑜𝑠𝑠subscript𝑇𝑐loss(T\_{c}).

The second option deals with first several examples in a permutation. For examples i𝑖i with small values σr​(i)subscript𝜎𝑟𝑖\sigma\_{r}(i), the variance of g​r​a​dr,σr​(i)−1​(i)𝑔𝑟𝑎subscript𝑑

𝑟subscript𝜎𝑟𝑖1𝑖grad\_{r,\sigma\_{r}(i)-1}(i) can be high. Therefore, we discard Δ​(i)Δ𝑖\Delta(i) from the beginning of the permutation, when we calculate l​o​s​s​(Tc)𝑙𝑜𝑠𝑠subscript𝑇𝑐loss(T\_{c}) in Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features"). Particularly, we eliminate the corresponding components of vectors G𝐺G and ΔΔ\Delta when calculating the cosine similarity between them.

## 6 Experiments

##### Comparison with baselines

We compare our algorithm with the most popular open-source libraries — XGBoost and LightGBM — on several well-known machine learning tasks. The detailed description of the experimental setup together with dataset descriptions is available in Appendix [D](#A4 "Appendix D Experimental setup ‣ CatBoost: unbiased boosting with categorical features"). The source code of the experiment is available, and the results can be reproduced.777<https://github.com/catboost/benchmarks/tree/master/quality_benchmarks> For all learning algorithms, we preprocess categorical features using the ordered TS method described in Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features"). The parameter tuning and training were performed on 4/5 of the data and the testing was performed on the remaining 1/5.888For Epsilon, we use default parameters instead of parameter tuning due to large running time for all algorithms. We tune only the number of trees to avoid overfitting. The results measured by logloss and zero-one loss are presented in Table [2](#S6.T2 "Table 2 ‣ Comparison with baselines ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features") (the absolute values for the baselines are in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features")). For CatBoost, we used Ordered boosting mode in this experiment.999The numbers for CatBoost in Table [2](#S6.T2 "Table 2 ‣ Comparison with baselines ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features") may slightly differ from the corresponding numbers in our GitHub repository, since we use another version of CatBoost with all the discussed features implemented. One can see that CatBoost outperforms other algorithms on all the considered datasets. We also measured statistical significance of improvements presented in Table [2](#S6.T2 "Table 2 ‣ Comparison with baselines ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features"): except three datasets (Appetency, Churn and Upselling) the improvements are statistically significant with p-value ≪0.01much-less-thanabsent0.01\ll 0.01 measured by the paired one-tailed t-test.

To demonstrate that our implementation of plain boosting is an appropriate baseline for our research, we show that a raw setting of CatBoost provides state-of-the-art quality. Particularly, we take a setting of CatBoost, which is close to classical GBDT [[12](#bib.bib12)], and compare it with the baseline boosting implementations in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"). Experiments show that this raw setting differs from the baselines insignificantly.

Table 2: Comparison with baselines: logloss / zero-one loss (relative increase for baselines).

|  |  |  |  |
| --- | --- | --- | --- |
|  | CatBoost | LightGBM | XGBoost |
| Adult | 0.270 / 0.127 | +2.4% / +1.9% | +2.2% / +1.0% |
| Amazon | 0.139 / 0.044 | +17% / +21% | +17% / +21% |
| Click | 0.392 / 0.156 | +1.2% / +1.2% | +1.2% / +1.2% |
| Epsilon | 0.265 / 0.109 | +1.5% / +4.1% | +11% / +12% |
| Appetency | 0.072 / 0.018 | +0.4% / +0.2% | +0.4% / +0.7% |
| Churn | 0.232 / 0.072 | +0.1% / +0.6% | +0.5% / +1.6% |
| Internet | 0.209 / 0.094 | +6.8% / +8.6% | +7.9% / +8.0% |
| Upselling | 0.166 / 0.049 | +0.3% / +0.1% | +0.04% / +0.3% |
| Kick | 0.286 / 0.095 | +3.5% / +4.4% | +3.2% / +4.1% |

Table 3: Plain boosting mode: logloss, zero-one loss and their change relative to Ordered boosting mode.

|  |  |  |
| --- | --- | --- |
|  | Logloss | Zero-one loss |
| Adult | 0.272 (+1.1%) | 0.127 (-0.1%) |
| Amazon | 0.139 (-0.6%) | 0.044 (-1.5%) |
| Click | 0.392 (-0.05%) | 0.156 (+0.19%) |
| Epsilon | 0.266 (+0.6%) | 0.110 (+0.9%) |
| Appetency | 0.072 (+0.5%) | 0.018 (+1.5%) |
| Churn | 0.232 (-0.06%) | 0.072 (-0.17%) |
| Internet | 0.217 (+3.9%) | 0.099 (+5.4%) |
| Upselling | 0.166 (+0.1%) | 0.049 (+0.4%) |
| Kick | 0.285 (-0.2%) | 0.095 (-0.1%) |

We also empirically analyzed the running times of the algorithms on Epsilon dataset. The details of the comparison can be found in Appendix [C.2](#A3.SS2 "C.2 Empirical analysis ‣ Appendix C Time complexity analysis ‣ CatBoost: unbiased boosting with categorical features"). To summarize, we obtained that CatBoost Plain and LightGBM are the fastest ones followed by Ordered mode, which is about 1.7 times slower.

##### Ordered and Plain modes

In this section, we compare two essential boosting modes of CatBoost: Plain and Ordered. First, we compared their performance on all the considered datasets, the results are presented in Table [3](#S6.T3 "Table 3 ‣ Comparison with baselines ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features"). It can be clearly seen that Ordered mode is particularly useful on small datasets. Indeed, the largest benefit from Ordered is observed on Adult and Internet datasets, which are relatively small (less than 40K training examples), which supports our hypothesis that a higher bias negatively affects the performance. Indeed, according to Theorem [1](#Thmthm1 "Theorem 1 ‣ Analysis of prediction shift ‣ 4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features") and our reasoning in Section [4.1](#S4.SS1 "4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features"), bias is expected to be larger for smaller datasets (however, it can also depend on other properties of the dataset, e.g., on the dependency between features and target). In order to further validate this hypothesis, we make the following experiment: we train CatBoost in Ordered and Plain modes on randomly filtered datasets and compare the obtained losses, see Figure [2](#S6.F2 "Figure 2 ‣ Ordered and Plain modes ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features"). As we expected, for smaller datasets the relative performance of Plain mode becomes worse. To save space, here we present the results only for logloss; the figure for zero-one loss is similar.

We also compare Ordered and Plain modes in the above-mentioned raw setting of CatBoost in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features") and conclude that the advantage of Ordered mode is not caused by interaction with specific CatBoost options.

Table 4: Comparison of target statistics, relative change in logloss / zero-one loss compared to ordered TS.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Greedy | Holdout | Leave-one-out |
| Adult | +1.1% / +0.8% | +2.1% / +2.0% | +5.5% / +3.7% |
| Amazon | +40% / +32% | +8.3% / +8.3% | +4.5% / +5.6% |
| Click | +13% / +6.7% | +1.5% / +0.5% | +2.7% / +0.9% |
| Appetency | +24% / +0.7% | +1.6% / -0.5% | +8.5% / +0.7% |
| Churn | +12% / +2.1% | +0.9% / +1.3% | +1.6% / +1.8% |
| Internet | +33% / +22% | +2.6% / +1.8% | +27% / +19% |
| Upselling | +57% / +50% | +1.6% / +0.9% | +3.9% / +2.9% |
| Kick | +22% / +28% | +1.3% / +0.32% | +3.7% / +3.3% |

Figure 2: Relative error of Plain boosting mode compared to Ordered boosting mode depending on the fraction of the dataset.

##### Analysis of target statistics

We compare different TSs introduced in Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features") as options of CatBoost in Ordered boosting mode keeping all other algorithmic details the same; the results can be found in Table [4](#S6.T4 "Table 4 ‣ Ordered and Plain modes ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features"). Here, to save space, we present only relative increase in loss functions for each algorithm compared to CatBoost with ordered TS. Note that the ordered TS used in CatBoost significantly outperform all other approaches. Also, among the baselines, the holdout TS is the best for most of the datasets since it does not suffer from conditional shift discussed in Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features") (P1); still, it is worse than CatBoost due to less effective usage of training data (P2). Leave-one-out is usually better than the greedy TS, but it can be much worse on some datasets, e.g., on Adult. The reason is that the greedy TS suffer from low-frequency categories, while the leave-one-out TS suffer also from high-frequency ones, and on Adult all the features have high frequency.

Finally, let us note that in Table [4](#S6.T4 "Table 4 ‣ Ordered and Plain modes ‣ 6 Experiments ‣ CatBoost: unbiased boosting with categorical features") we combine Ordered mode of CatBoost with different TSs. To generalize these results, we also made a similar experiment by combining different TS with Plain mode, used in standard gradient boosting. The obtained results and conclusions turned out to be very similar to the ones discussed above.

##### Feature combinations

The effect of feature combinations discussed in Section [5](#S5 "5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features") is demonstrated in Figure [3](#A7.F3 "Figure 3 ‣ Feature combinations ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features") in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"). In average, changing the number cm​a​xsubscript𝑐𝑚𝑎𝑥c\_{max} of features allowed to be combined from 1 to 2 provides an outstanding improvement of logloss by 1.86%percent1.861.86\% (reaching 11.3%percent11.311.3\%), changing from 1 to 3 yields 2.04%percent2.042.04\%, and further increase of cm​a​xsubscript𝑐𝑚𝑎𝑥c\_{max} does not influence the performance significantly.

##### Number of permutations

The effect of the number s𝑠s of permutations on the performance of CatBoost is presented in Figure [4](#A7.F4 "Figure 4 ‣ Number of permutations ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features") in Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"). In average, increasing s𝑠s slightly decreases logloss, e.g., by 0.19%percent0.190.19\% for s=3𝑠3s=3 and by 0.38%percent0.380.38\% for s=9𝑠9s=9 compared to s=1𝑠1s=1.

## 7 Conclusion

In this paper, we identify and analyze the problem of prediction shifts present in all existing implementations of gradient boosting. We propose a general solution, ordered boosting with ordered TS, which solves the problem. This idea is implemented in CatBoost, which is a new gradient boosting library. Empirical results demonstrate that CatBoost outperforms leading GBDT packages and leads to new state-of-the-art results on common benchmarks.

#### Acknowledgments

We are very grateful to Mikhail Bilenko for important references and advices that lead to theoretical analysis of this paper, as well as suggestions on the presentation. We also thank Pavel Serdyukov for many helpful discussions and valuable links, Nikita Kazeev, Nikita Dmitriev, Stanislav Kirillov and Victor Omelyanenko for help with experiments.

## References

* [1]

  L. Bottou and Y. L. Cun.
  Large scale online learning.
  In Advances in neural information processing systems, pages
  217–224, 2004.
* [2]

  L. Breiman.
  Out-of-bag estimation, 1996.
* [3]

  L. Breiman.
  Using iterated bagging to debias regressions.
  Machine Learning, 45(3):261–277, 2001.
* [4]

  L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen.
  Classification and regression trees.
  CRC press, 1984.
* [5]

  R. Caruana and A. Niculescu-Mizil.
  An empirical comparison of supervised learning algorithms.
  In Proceedings of the 23rd international conference on Machine
  learning, pages 161–168. ACM, 2006.
* [6]

  B. Cestnik et al.
  Estimating probabilities: a crucial task in machine learning.
  In ECAI, volume 90, pages 147–149, 1990.
* [7]

  O. Chapelle, E. Manavoglu, and R. Rosales.
  Simple and scalable response prediction for display advertising.
  ACM Transactions on Intelligent Systems and Technology (TIST),
  5(4):61, 2015.
* [8]

  T. Chen and C. Guestrin.
  Xgboost: A scalable tree boosting system.
  In Proceedings of the 22Nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016.
* [9]

  M. Ferov and M. Modrỳ.
  Enhancing lambdamart using oblivious trees.
  arXiv preprint arXiv:1609.05610, 2016.
* [10]

  J. Friedman, T. Hastie, and R. Tibshirani.
  Additive logistic regression: a statistical view of boosting.
  The annals of statistics, 28(2):337–407, 2000.
* [11]

  J. Friedman, T. Hastie, and R. Tibshirani.
  The elements of statistical learning, volume 1.
  Springer series in statistics New York, 2001.
* [12]

  J. H. Friedman.
  Greedy function approximation: a gradient boosting machine.
  Annals of statistics, pages 1189–1232, 2001.
* [13]

  J. H. Friedman.
  Stochastic gradient boosting.
  Computational Statistics & Data Analysis, 38(4):367–378,
  2002.
* [14]

  A. Gulin, I. Kuralenok, and D. Pavlov.
  Winning the transfer learning track of yahoo!’s learning to rank
  challenge with yetirank.
  In Yahoo! Learning to Rank Challenge, pages 63–76, 2011.
* [15]

  X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich,
  S. Bowers, et al.
  Practical lessons from predicting clicks on ads at facebook.
  In Proceedings of the Eighth International Workshop on Data
  Mining for Online Advertising, pages 1–9. ACM, 2014.
* [16]

  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In Advances in Neural Information Processing Systems, pages
  3149–3157, 2017.
* [17]

  M. Kearns and L. Valiant.
  Cryptographic limitations on learning boolean formulae and finite
  automata.
  Journal of the ACM (JACM), 41(1):67–95, 1994.
* [18]

  J. Langford, L. Li, and T. Zhang.
  Sparse online learning via truncated gradient.
  Journal of Machine Learning Research, 10(Mar):777–801, 2009.
* [19]

  LightGBM.
  Categorical feature support.
  <http://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support>,
  2017.
* [20]

  LightGBM.
  Optimal split for categorical features.
  <http://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features>,
  2017.
* [21]

  LightGBM.
  feature\_histogram.cpp.
  <https://github.com/Microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp>,
  2018.
* [22]

  X. Ling, W. Deng, C. Gu, H. Zhou, C. Li, and F. Sun.
  Model ensemble for click prediction in bing search ads.
  In Proceedings of the 26th International Conference on World
  Wide Web Companion, pages 689–698. International World Wide Web Conferences
  Steering Committee, 2017.
* [23]

  Y. Lou and M. Obukhov.
  Bdt: Gradient boosted decision tables for high accuracy and scoring
  efficiency.
  In Proceedings of the 23rd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining, pages 1893–1901. ACM, 2017.
* [24]

  L. Mason, J. Baxter, P. L. Bartlett, and M. R. Frean.
  Boosting algorithms as gradient descent.
  In Advances in neural information processing systems, pages
  512–518, 2000.
* [25]

  D. Micci-Barreca.
  A preprocessing scheme for high-cardinality categorical attributes in
  classification and prediction problems.
  ACM SIGKDD Explorations Newsletter, 3(1):27–32, 2001.
* [26]

  B. P. Roe, H.-J. Yang, J. Zhu, Y. Liu, I. Stancu, and G. McGregor.
  Boosted decision trees as an alternative to artificial neural
  networks for particle identification.
  Nuclear Instruments and Methods in Physics Research Section A:
  Accelerators, Spectrometers, Detectors and Associated Equipment,
  543(2):577–584, 2005.
* [27]

  L. Rokach and O. Maimon.
  Top–down induction of decision trees classifiers — a survey.
  IEEE Transactions on Systems, Man, and Cybernetics, Part C
  (Applications and Reviews), 35(4):476–487, 2005.
* [28]

  D. B. Rubin.
  The bayesian bootstrap.
  The annals of statistics, pages 130–134, 1981.
* [29]

  Q. Wu, C. J. Burges, K. M. Svore, and J. Gao.
  Adapting boosting for information retrieval measures.
  Information Retrieval, 13(3):254–270, 2010.
* [30]

  K. Zhang, B. Schölkopf, K. Muandet, and Z. Wang.
  Domain adaptation under target and conditional shift.
  In International Conference on Machine Learning, pages
  819–827, 2013.
* [31]

  O. Zhang.
  Winning data science competitions.
  <https://www.slideshare.net/ShangxuanZhang/winning-data-science-competitions-presented-by-owen-zhang>,
  2015.
* [32]

  Y. Zhang and A. Haghani.
  A gradient boosting method to improve travel time prediction.
  Transportation Research Part C: Emerging Technologies,
  58:308–324, 2015.

## Appendix A Proof of Theorem 1

### A.1 Proof for the case 𝒟1=𝒟2subscript𝒟1subscript𝒟2\mathcal{D}\_{1}=\mathcal{D}\_{2}

Let us denote by A𝐴A the event that each leaf in both stumps h1superscriptℎ1h^{1} and h2superscriptℎ2h^{2} contains at least one example, i.e., there exists at least one 𝐱∈𝒟𝐱𝒟\mathbf{x}\in\mathcal{D} with 𝐱i=ssuperscript𝐱𝑖𝑠\mathbf{x}^{i}=s for all i∈{1,2}𝑖12i\in\{1,2\}, s∈{0,1}𝑠01s\in\{0,1\}. All further reasonings are given conditioning on A𝐴A. Note that the probability of A𝐴A is 1−O​(2−n)1𝑂superscript2𝑛1-O\left(2^{-n}\right), therefore we can assign an arbitrary value to any empty leaf during the learning process, and the choice of the value will affect all expectations we calculate below by O​(2−n)𝑂superscript2𝑛O\left(2^{-n}\right).

Denote by ξs​tsubscript𝜉𝑠𝑡\xi\_{st}, s,t∈{0,1}

𝑠𝑡
01s,t\in\{0,1\}, the number of examples 𝐱k∈𝒟subscript𝐱𝑘𝒟\mathbf{x}\_{k}\in\mathcal{D} with 𝐱k=(s,t)subscript𝐱𝑘𝑠𝑡\mathbf{x}\_{k}=(s,t). The value of the first stump h1superscriptℎ1h^{1} in the region {x1=s}superscript𝑥1𝑠\{x^{1}=s\} is the average value of yksubscript𝑦𝑘y\_{k} over examples from 𝒟𝒟\mathcal{D} belonging to this region. That is,

|  |  |  |
| --- | --- | --- |
|  | h1​(0,t)=∑j=1nc2​𝟙{xj=(0,1)}∑j=1n𝟙{xj1=0}=c2​ξ01ξ00+ξ01,superscriptℎ10𝑡superscriptsubscript𝑗1𝑛subscript𝑐2subscript1subscript𝑥𝑗01superscriptsubscript𝑗1𝑛subscript1superscriptsubscript𝑥𝑗10subscript𝑐2subscript𝜉01subscript𝜉00subscript𝜉01h^{1}(0,t)=\frac{\sum\_{j=1}^{n}c\_{2}\mathbbm{1}\_{\{x\_{j}=(0,1)\}}}{\sum\_{j=1}^{n}\mathbbm{1}\_{\{x\_{j}^{1}=0\}}}=\frac{c\_{2}\xi\_{01}}{\xi\_{00}+\xi\_{01}}\,, |  |

|  |  |  |
| --- | --- | --- |
|  | h1​(1,t)=∑j=1nc1​𝟙{xj1=1}+c2​𝟙{xj=(1,1)}∑j=1n𝟙{xj1=1}=c1+c2​ξ11ξ10+ξ11.superscriptℎ11𝑡superscriptsubscript𝑗1𝑛subscript𝑐1subscript1superscriptsubscript𝑥𝑗11subscript𝑐2subscript1subscript𝑥𝑗11superscriptsubscript𝑗1𝑛subscript1superscriptsubscript𝑥𝑗11subscript𝑐1subscript𝑐2subscript𝜉11subscript𝜉10subscript𝜉11h^{1}(1,t)=\frac{\sum\_{j=1}^{n}c\_{1}\mathbbm{1}\_{\{x\_{j}^{1}=1\}}+c\_{2}\mathbbm{1}\_{\{x\_{j}=(1,1)\}}}{\sum\_{j=1}^{n}\mathbbm{1}\_{\{x\_{j}^{1}=1\}}}=c\_{1}+\frac{c\_{2}\xi\_{11}}{\xi\_{10}+\xi\_{11}}\,. |  |

Summarizing, we obtain

|  |  |  |  |
| --- | --- | --- | --- |
|  | h1​(s,t)=c1​s+c2​ξs​1ξs​0+ξs​1.superscriptℎ1𝑠𝑡subscript𝑐1𝑠subscript𝑐2subscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1h^{1}(s,t)=c\_{1}s+\frac{c\_{2}\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}. |  | (7) |

Note that, by conditioning on A𝐴A, we guarantee that the denominator ξs​0+ξs​1subscript𝜉𝑠0subscript𝜉𝑠1\xi\_{s0}+\xi\_{s1} is not equal to zero.

Now we derive the expectation 𝔼​(h1​(𝐱))𝔼superscriptℎ1𝐱\mathbb{E}(h^{1}(\mathbf{x})) of prediction h1superscriptℎ1h^{1} for a test example 𝐱=(s,t)𝐱𝑠𝑡\mathbf{x}=(s,t).

It is easy to show that
𝔼​(ξs​1ξs​0+ξs​1∣A)=12𝔼conditionalsubscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1𝐴12\mathbb{E}\left(\frac{\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}\mid A\right)=\frac{1}{2}. Indeed, due to the symmetry we have 𝔼​(ξs​1ξs​0+ξs​1∣A)=𝔼​(ξs​0ξs​0+ξs​1∣A)𝔼conditionalsubscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1𝐴𝔼conditionalsubscript𝜉𝑠0subscript𝜉𝑠0subscript𝜉𝑠1𝐴\mathbb{E}\left(\frac{\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}\mid A\right)=\mathbb{E}\left(\frac{\xi\_{s0}}{\xi\_{s0}+\xi\_{s1}}\mid A\right) and the sum of these expectations is 𝔼​(ξs​0+ξs​1ξs​0+ξs​1∣A)=1𝔼conditionalsubscript𝜉𝑠0subscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1𝐴1\mathbb{E}\left(\frac{\xi\_{s0}+\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}\mid A\right)=1. So, by taking the expectation of ([7](#A1.E7 "In A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features")), we obtain the following proposition.

###### Proposition 1

We have 𝔼​(h1​(s,t)∣A)=c1​s+c22𝔼conditionalsuperscriptℎ1𝑠𝑡𝐴subscript𝑐1𝑠subscript𝑐22\mathbb{E}(h^{1}(s,t)\mid A)=c\_{1}s+\frac{c\_{2}}{2}.

It means that the conditional expectation 𝔼​(h1​(𝐱)∣𝐱=(s,t),A)𝔼conditionalsuperscriptℎ1𝐱𝐱

𝑠𝑡𝐴\mathbb{E}(h^{1}(\mathbf{x})\mid\mathbf{x}=(s,t),A) on a test example 𝐱𝐱\mathbf{x} equals c1​s+c22subscript𝑐1𝑠subscript𝑐22c\_{1}s+\frac{c\_{2}}{2}, since 𝐱𝐱\mathbf{x} and h1superscriptℎ1h^{1} are independent.

##### Prediction shift of h1superscriptℎ1h^{1}

In this paragraph, we show that the conditional expectation 𝔼​(h1​(𝐱l)∣𝐱l=(s,t),A)𝔼conditionalsuperscriptℎ1subscript𝐱𝑙subscript𝐱𝑙

𝑠𝑡𝐴\mathbb{E}(h^{1}(\mathbf{x}\_{l})\mid\mathbf{x}\_{l}=(s,t),A) on a training example 𝐱lsubscript𝐱𝑙\mathbf{x}\_{l} is shifted for any l=1,…,n𝑙

1…𝑛l=1,\ldots,n, because the model h1superscriptℎ1h^{1} is fitted to 𝐱lsubscript𝐱𝑙\mathbf{x}\_{l}. This is an auxiliary result, which is not used directly for proving the theorem, but helps to track the chain of obtained shifts.

###### Proposition 2

The conditional expectation is

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h1​(𝐱l)∣𝐱l=(s,t),A)=c1​s+c22−c2​(2​t−1n)+O​(2−n).𝔼conditionalsuperscriptℎ1subscript𝐱𝑙subscript𝐱𝑙  𝑠𝑡𝐴subscript𝑐1𝑠subscript𝑐22subscript𝑐22𝑡1𝑛𝑂superscript2𝑛\mathbb{E}(h^{1}(\mathbf{x}\_{l})\mid\mathbf{x}\_{l}=(s,t),A)=c\_{1}s+\frac{c\_{2}}{2}-c\_{2}\left(\frac{2t-1}{n}\right)+O(2^{-n})\,. |  |

Proof.
Let us introduce the following notation

|  |  |  |
| --- | --- | --- |
|  | αs​k=𝟙{xk=(s,1)}ξs​0+ξs​1.subscript𝛼𝑠𝑘subscript1subscript𝑥𝑘𝑠1subscript𝜉𝑠0subscript𝜉𝑠1\alpha\_{sk}=\frac{\mathbbm{1}\_{\{x\_{k}=(s,1)\}}}{\xi\_{s0}+\xi\_{s1}}\,. |  |

Then, we can rewrite the conditional expectation as

|  |  |  |
| --- | --- | --- |
|  | c1​s+c2​∑k=1n𝔼​(αs​k∣𝐱l=(s,t),A).subscript𝑐1𝑠subscript𝑐2superscriptsubscript𝑘1𝑛𝔼conditionalsubscript𝛼𝑠𝑘subscript𝐱𝑙  𝑠𝑡𝐴c\_{1}s+c\_{2}\sum\_{k=1}^{n}\mathbb{E}(\alpha\_{sk}\mid\mathbf{x}\_{l}=(s,t),A)\,. |  |

Lemma [1](#Thmlem1 "Lemma 1 ‣ Prediction shift of ℎ¹ ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features") below implies that
𝔼​(αs​l∣𝐱l=(s,t),A)=2​tn𝔼conditionalsubscript𝛼𝑠𝑙subscript𝐱𝑙

𝑠𝑡𝐴2𝑡𝑛\mathbb{E}(\alpha\_{sl}\mid\mathbf{x}\_{l}=(s,t),A)=\frac{2t}{n}.
For k≠l𝑘𝑙k\neq l, we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(αs​k∣𝐱l=(s,t),A)=14​𝔼​(1ξs​0+ξs​1∣𝐱l=(s,t),𝐱k=(s,1),A)=12​n​(1−1n−1+n−2(2n−1−2)​(n−1))𝔼conditionalsubscript𝛼𝑠𝑘subscript𝐱𝑙  𝑠𝑡𝐴14𝔼formulae-sequenceconditional1subscript𝜉𝑠0subscript𝜉𝑠1subscript𝐱𝑙𝑠𝑡subscript𝐱𝑘  𝑠1𝐴12𝑛11𝑛1𝑛2superscript2𝑛12𝑛1\mathbb{E}(\alpha\_{sk}\mid\mathbf{x}\_{l}=(s,t),A)=\frac{1}{4}\,\mathbb{E}\left(\frac{1}{\xi\_{s0}+\xi\_{s1}}\mid\mathbf{x}\_{l}=(s,t),\mathbf{x}\_{k}=(s,1),A\right)\\ =\frac{1}{2n}\left(1-\frac{1}{n-1}+\frac{n-2}{\left(2^{n-1}-2\right)(n-1)}\right) |  |

due to Lemma [2](#Thmlem2 "Lemma 2 ‣ Prediction shift of ℎ¹ ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features") below. Finally, we obtain

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h1​(𝐱l)∣𝐱l=(s,t))=c1​s+c2​(2​tn+(n−1)​12​n​(1−1n−1))+O​(2−n)=c1​s+c22−c2​(2​t−1n)+O​(2−n).𝔼conditionalsuperscriptℎ1subscript𝐱𝑙subscript𝐱𝑙𝑠𝑡subscript𝑐1𝑠subscript𝑐22𝑡𝑛𝑛112𝑛11𝑛1𝑂superscript2𝑛subscript𝑐1𝑠subscript𝑐22subscript𝑐22𝑡1𝑛𝑂superscript2𝑛\mathbb{E}(h^{1}(\mathbf{x}\_{l})\mid\mathbf{x}\_{l}=(s,t))=c\_{1}s+c\_{2}\left(\frac{2t}{n}+(n-1)\frac{1}{2n}\left(1-\frac{1}{n-1}\right)\right)\\ +O\left(2^{-n}\right)=c\_{1}s+\frac{c\_{2}}{2}-c\_{2}\left(\frac{2t-1}{n}\right)+O(2^{-n}). |  |

□□\Box

###### Lemma 1

𝔼​(1ξs​0+ξs​1∣𝐱1=(s,t),A)=2n𝔼conditional1subscript𝜉𝑠0subscript𝜉𝑠1subscript𝐱1

𝑠𝑡𝐴2𝑛\mathbb{E}\left(\frac{1}{\xi\_{s0}+\xi\_{s1}}\mid\mathbf{x}\_{1}=(s,t),A\right)=\frac{2}{n} .

Proof.
Note that given 𝐱1=(s,t)subscript𝐱1𝑠𝑡\mathbf{x}\_{1}=(s,t), A𝐴A corresponds to the event that there is an example with x1=1−ssuperscript𝑥11𝑠x^{1}=1-s and (possibly another) example with x2=1−tsuperscript𝑥21𝑡x^{2}=1-t among 𝐱2,…,𝐱n

subscript𝐱2…subscript𝐱𝑛\mathbf{x}\_{2},\ldots,\mathbf{x}\_{n}.

Note that ξs​0+ξs​1=∑j=1n𝟙{xj1=s}subscript𝜉𝑠0subscript𝜉𝑠1superscriptsubscript𝑗1𝑛subscript1superscriptsubscript𝑥𝑗1𝑠\xi\_{s0}+\xi\_{s1}=\sum\_{j=1}^{n}\mathbbm{1}\_{\{x\_{j}^{1}=s\}}.
For k=1,…,n−1𝑘

1…𝑛1k=1,\ldots,n-1, we have

|  |  |  |
| --- | --- | --- |
|  | P(ξs​0+ξs​1=k∣𝐱1=(s,t),A)=P​(ξs​0+ξs​1=k,A∣𝐱1=(s,t))P​(A∣𝐱1=(s,t))=(n−1k−1)2n−1​(1−2−(n−1)),\mathrm{P}(\xi\_{s0}+\xi\_{s1}=k\mid\mathbf{x}\_{1}=(s,t),A)=\frac{\mathrm{P}(\xi\_{s0}+\xi\_{s1}=k,A\mid\mathbf{x}\_{1}=(s,t))}{\mathrm{P}(A\mid\mathbf{x}\_{1}=(s,t))}=\frac{\binom{n-1}{k-1}}{2^{n-1}\left(1-2^{-(n-1)}\right)}, |  |

since 𝟙{x11=s}=1subscript1superscriptsubscript𝑥11𝑠1\mathbbm{1}\_{\{x\_{1}^{1}=s\}}=1 when 𝐱1=(s,t)subscript𝐱1𝑠𝑡\mathbf{x}\_{1}=(s,t) with probability 1, ∑j=2n𝟙{xj1=s}superscriptsubscript𝑗2𝑛subscript1superscriptsubscript𝑥𝑗1𝑠\sum\_{j=2}^{n}\mathbbm{1}\_{\{x\_{j}^{1}=s\}} is a binomial variable independent of 𝐱1subscript𝐱1\mathbf{x}\_{1}, and an example with x1=1−ssuperscript𝑥11𝑠x^{1}=1-s exists whenever ξs​0+ξs​1=k<nsubscript𝜉𝑠0subscript𝜉𝑠1𝑘𝑛\xi\_{s0}+\xi\_{s1}=k<n and 𝐱1=(s,t)subscript𝐱1𝑠𝑡\mathbf{x}\_{1}=(s,t) (while the existence of one with x2=1−tsuperscript𝑥21𝑡x^{2}=1-t is an independent event). Therefore, we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(1ξs​0+ξs​1∣𝐱1=(s,t),A)=∑k=1n−11k​(n−1k−1)2n−1−1=1n​(2n−1−1)​∑k=1n−1(nk)=2n.𝔼conditional1subscript𝜉𝑠0subscript𝜉𝑠1subscript𝐱1  𝑠𝑡𝐴superscriptsubscript𝑘1𝑛11𝑘binomial𝑛1𝑘1superscript2𝑛111𝑛superscript2𝑛11superscriptsubscript𝑘1𝑛1binomial𝑛𝑘2𝑛\mathbb{E}\left(\frac{1}{\xi\_{s0}+\xi\_{s1}}\mid\mathbf{x}\_{1}=(s,t),A\right)=\sum\_{k=1}^{n-1}\frac{1}{k}\frac{\binom{n-1}{k-1}}{2^{n-1}-1}=\frac{1}{n\left(2^{n-1}-1\right)}\sum\_{k=1}^{n-1}\binom{n}{k}=\frac{2}{n}\,. |  |

□□\Box

###### Lemma 2

We have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(1ξs​0+ξs​1∣𝐱1=(s,t1),𝐱2=(s,t2),A)=2n​(1−1n−1+n−2(2n−1−2)​(n−1)).𝔼formulae-sequenceconditional1subscript𝜉𝑠0subscript𝜉𝑠1subscript𝐱1𝑠subscript𝑡1subscript𝐱2  𝑠subscript𝑡2𝐴2𝑛11𝑛1𝑛2superscript2𝑛12𝑛1\mathbb{E}\left(\frac{1}{\xi\_{s0}+\xi\_{s1}}\mid\mathbf{x}\_{1}=(s,t\_{1}),\mathbf{x}\_{2}=(s,t\_{2}),A\right)=\frac{2}{n}\left(1-\frac{1}{n-1}+\frac{n-2}{\left(2^{n-1}-2\right)(n-1)}\right). |  |

Proof.
Similarly to the previous proof, for k=2,…,n−1𝑘

2…𝑛1k=2,\ldots,n-1, we have

|  |  |  |
| --- | --- | --- |
|  | P(ξs​0+ξs​1=k∣𝐱1=(s,t1),𝐱2=(s,t2),A)=(n−2k−2)2n−2​(1−2−(n−2)).\mathrm{P}\left(\xi\_{s0}+\xi\_{s1}=k\mid\mathbf{x}\_{1}=(s,t\_{1}),\mathbf{x}\_{2}=(s,t\_{2}),A\right)=\frac{\binom{n-2}{k-2}}{2^{n-2}\left(1-2^{-(n-2)}\right)}\,. |  |

Therefore,

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(1ξs​0+ξs​1∣𝐱1=(s,t1),𝐱2=(s,t2),A)=12n−2​(1−2−(n−1))​∑k=2n−1(n−2k−2)k=12n−2−1​∑k=2n−1(n−2k−2)​(1k−1−1(k−1)​k)=12n−2−1​∑k=2n−1(1n−1​(n−1k−1)−1n​(n−1)​(nk))==12n−2−1​(1n−1​(2n−1−2)−1n​(n−1)​(2n−n−2))==2n​(1−1n−1+n−2(2n−1−2)​(n−1)).𝔼formulae-sequenceconditional1subscript𝜉𝑠0subscript𝜉𝑠1subscript𝐱1𝑠subscript𝑡1subscript𝐱2  𝑠subscript𝑡2𝐴1superscript2𝑛21superscript2𝑛1superscriptsubscript𝑘2𝑛1binomial𝑛2𝑘2𝑘1superscript2𝑛21superscriptsubscript𝑘2𝑛1binomial𝑛2𝑘21𝑘11𝑘1𝑘1superscript2𝑛21superscriptsubscript𝑘2𝑛11𝑛1binomial𝑛1𝑘11𝑛𝑛1binomial𝑛𝑘1superscript2𝑛211𝑛1superscript2𝑛121𝑛𝑛1superscript2𝑛𝑛22𝑛11𝑛1𝑛2superscript2𝑛12𝑛1\mathbb{E}\left(\frac{1}{\xi\_{s0}+\xi\_{s1}}\mid\mathbf{x}\_{1}=(s,t\_{1}),\mathbf{x}\_{2}=(s,t\_{2}),A\right)=\frac{1}{2^{n-2}\left(1-2^{-(n-1)}\right)}\sum\_{k=2}^{n-1}\frac{\binom{n-2}{k-2}}{k}\\ =\frac{1}{2^{n-2}-1}\sum\_{k=2}^{n-1}\binom{n-2}{k-2}\left(\frac{1}{k-1}-\frac{1}{(k-1)k}\right)\\ =\frac{1}{2^{n-2}-1}\sum\_{k=2}^{n-1}\left(\frac{1}{n-1}\binom{n-1}{k-1}-\frac{1}{n(n-1)}\binom{n}{k}\right)=\\ =\frac{1}{2^{n-2}-1}\left(\frac{1}{n-1}(2^{n-1}-2)-\frac{1}{n(n-1)}(2^{n}-n-2)\right)=\\ =\frac{2}{n}\left(1-\frac{1}{n-1}+\frac{n-2}{\left(2^{n-1}-2\right)(n-1)}\right)\,. |  |

□□\Box

##### Bias of the model h1+h2superscriptℎ1superscriptℎ2h^{1}+h^{2}

Proposition [2](#Thmprop2 "Proposition 2 ‣ Prediction shift of ℎ¹ ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features") shows that the values of the model h1superscriptℎ1h^{1} on training examples are shifted with respect to the ones on test examples. The next step is to show how this can lead to a bias of the trained model, if we use the same dataset for building both h1superscriptℎ1h^{1} and h2superscriptℎ2h^{2}. Namely, we derive the expected value of h1​(s,t)+h2​(s,t)superscriptℎ1𝑠𝑡superscriptℎ2𝑠𝑡h^{1}(s,t)+h^{2}(s,t) and obtain a bias according to the following result.

###### Proposition 3

If both h1superscriptℎ1h^{1} and h2superscriptℎ2h^{2} are built using the same dataset 𝒟𝒟\mathcal{D}, then

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h1​(s,t)+h2​(s,t)∣A)=f∗​(s,t)−1n−1​c2​(t−12)+O​(1/2n).𝔼superscriptℎ1𝑠𝑡conditionalsuperscriptℎ2𝑠𝑡𝐴superscript𝑓𝑠𝑡1𝑛1subscript𝑐2𝑡12𝑂1superscript2𝑛\mathbb{E}\left(h^{1}(s,t)+h^{2}(s,t)\mid A\right)=f^{\*}(s,t)-\frac{1}{n-1}c\_{2}\left(t-\frac{1}{2}\right)+O(1/2^{n})\,. |  |

Proof.
The residual after the first step is

|  |  |  |
| --- | --- | --- |
|  | f∗​(s,t)−h1​(s,t)=c2​(t−ξs​1ξs​0+ξs​1).superscript𝑓𝑠𝑡superscriptℎ1𝑠𝑡subscript𝑐2𝑡subscript𝜉𝑠1subscript𝜉𝑠0subscript𝜉𝑠1f^{\*}(s,t)-h^{1}(s,t)=c\_{2}\left(t-\frac{\xi\_{s1}}{\xi\_{s0}+\xi\_{s1}}\right)\,. |  |

Therefore, we get

|  |  |  |
| --- | --- | --- |
|  | h2​(s,t)=c2ξ0​t+ξ1​t​((t−ξ01ξ00+ξ01)​ξ0​t+(t−ξ11ξ10+ξ11)​ξ1​t),superscriptℎ2𝑠𝑡subscript𝑐2subscript𝜉0𝑡subscript𝜉1𝑡𝑡subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉0𝑡𝑡subscript𝜉11subscript𝜉10subscript𝜉11subscript𝜉1𝑡h^{2}(s,t)=\frac{c\_{2}}{\xi\_{0t}+\xi\_{1t}}\left(\left(t-\frac{\xi\_{01}}{\xi\_{00}+\xi\_{01}}\right)\xi\_{0t}+\left(t-\frac{\xi\_{11}}{\xi\_{10}+\xi\_{11}}\right)\xi\_{1t}\right)\,, |  |

which is equal to

|  |  |  |
| --- | --- | --- |
|  | −c2​(ξ00​ξ01(ξ00+ξ01)​(ξ00+ξ10)+ξ10​ξ11(ξ10+ξ11)​(ξ00+ξ10))subscript𝑐2subscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉10subscript𝜉10subscript𝜉11subscript𝜉10subscript𝜉11subscript𝜉00subscript𝜉10-c\_{2}\left(\frac{\xi\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi\_{00}+\xi\_{10})}+\frac{\xi\_{10}\xi\_{11}}{(\xi\_{10}+\xi\_{11})(\xi\_{00}+\xi\_{10})}\right) |  |

for t=0𝑡0t=0 and to

|  |  |  |
| --- | --- | --- |
|  | c2​(ξ00​ξ01(ξ00+ξ01)​(ξ01+ξ11)+ξ10​ξ11(ξ10+ξ11)​(ξ01+ξ11))subscript𝑐2subscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11subscript𝜉10subscript𝜉11subscript𝜉10subscript𝜉11subscript𝜉01subscript𝜉11c\_{2}\left(\frac{\xi\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}+\frac{\xi\_{10}\xi\_{11}}{(\xi\_{10}+\xi\_{11})(\xi\_{01}+\xi\_{11})}\right) |  |

for t=1𝑡1t=1. The expected values of all four ratios are equal due to symmetries, and they are equal to 14​(1−1n−1)+O​(2−n)1411𝑛1𝑂superscript2𝑛\frac{1}{4}\left(1-\frac{1}{n-1}\right)+O(2^{-n}) according to Lemma [3](#Thmlem3 "Lemma 3 ‣ Bias of the model ℎ¹+ℎ² ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features") below.
So, we obtain

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h2​(s,t)∣A)=(2​t−1)​c22​(1−1n−1)+O​(2−n)𝔼conditionalsuperscriptℎ2𝑠𝑡𝐴2𝑡1subscript𝑐2211𝑛1𝑂superscript2𝑛\mathbb{E}(h^{2}(s,t)\mid A)=(2t-1)\frac{c\_{2}}{2}\left(1-\frac{1}{n-1}\right)+O(2^{-n}) |  |

and

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h1​(s,t)+h2​(s,t)∣A)=f∗​(s,t)−c2​1n−1​(t−12)+O​(2−n).𝔼superscriptℎ1𝑠𝑡conditionalsuperscriptℎ2𝑠𝑡𝐴superscript𝑓𝑠𝑡subscript𝑐21𝑛1𝑡12𝑂superscript2𝑛\mathbb{E}(h^{1}(s,t)+h^{2}(s,t)\mid A)=f^{\*}(s,t)-c\_{2}\frac{1}{n-1}\left(t-\frac{1}{2}\right)+O(2^{-n})\,. |  |

□□\Box

###### Lemma 3

We have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(ξ00​ξ01(ξ00+ξ01)​(ξ01+ξ11)∣A)=14​(1−1n−1)+O​(2−n).𝔼conditionalsubscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11𝐴1411𝑛1𝑂superscript2𝑛\mathbb{E}\left(\frac{\xi\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}\mid A\right)=\frac{1}{4}\left(1-\frac{1}{n-1}\right)+O(2^{-n})\,. |  |

Proof.
First, linearity implies

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(ξ00​ξ01(ξ00+ξ01)​(ξ01+ξ11)∣A)=∑i,j𝔼​(𝟙𝐱i=(0,0),𝐱j=(0,1)(ξ00+ξ01)​(ξ01+ξ11)∣A).𝔼conditionalsubscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11𝐴subscript  𝑖𝑗𝔼conditionalsubscript1formulae-sequencesubscript𝐱𝑖00subscript𝐱𝑗01subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11𝐴\mathbb{E}\left(\frac{\xi\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}\mid A\right)=\sum\_{i,j}\mathbb{E}\left(\frac{\mathbbm{1}\_{\mathbf{x}\_{i}=(0,0),\mathbf{x}\_{j}=(0,1)}}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}\mid A\right)\,. |  |

Taking into account that all terms are equal, the expectation can be written as n​(n−1)42​a𝑛𝑛1superscript42𝑎\frac{n(n-1)}{4^{2}}a, where

|  |  |  |
| --- | --- | --- |
|  | a=𝔼​(1(ξ00+ξ01)​(ξ01+ξ11)∣𝐱1=(0,0),𝐱2=(0,1),A).𝑎𝔼formulae-sequenceconditional1subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11subscript𝐱100subscript𝐱2  01𝐴a=\mathbb{E}\left(\frac{1}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1),A\right)\,. |  |

A key observation is that ξ00+ξ01subscript𝜉00subscript𝜉01\xi\_{00}+\xi\_{01} and ξ01+ξ11subscript𝜉01subscript𝜉11\xi\_{01}+\xi\_{11} are two independent binomial variables: the former one is the number of k𝑘k such that xk1=0superscriptsubscript𝑥𝑘10{x\_{k}^{1}=0} and the latter one is the number of k𝑘k such that xk2=1superscriptsubscript𝑥𝑘21x\_{k}^{2}=1. Moreover, they (and also their inverses) are also conditionally independent given that first two observations of the Bernoulli scheme are known (𝐱1=(0,0),𝐱2=(0,1)formulae-sequencesubscript𝐱100subscript𝐱201\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1)) and given A𝐴A. This conditional independence implies that a𝑎a is the product of 𝔼​(1ξ00+ξ01∣𝐱1=(0,0),𝐱2=(0,1),A)𝔼formulae-sequenceconditional1subscript𝜉00subscript𝜉01subscript𝐱100subscript𝐱2

01𝐴\mathbb{E}\left(\frac{1}{\xi\_{00}+\xi\_{01}}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1),A\right)
and
𝔼​(1ξ01+ξ11∣𝐱1=(0,0),𝐱2=(0,1),A)𝔼formulae-sequenceconditional1subscript𝜉01subscript𝜉11subscript𝐱100subscript𝐱2

01𝐴\mathbb{E}\left(\frac{1}{\xi\_{01}+\xi\_{11}}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1),A\right).
The first factor equals
2n​(1−1n−1+O​(2−n))2𝑛11𝑛1𝑂superscript2𝑛\frac{2}{n}\left(1-\frac{1}{n-1}+O(2^{-n})\right)
according to Lemma [2](#Thmlem2 "Lemma 2 ‣ Prediction shift of ℎ¹ ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features"). The second one is equal to
𝔼​(1ξ01+ξ11∣𝐱1=(0,0),𝐱2=(0,1))𝔼formulae-sequenceconditional1subscript𝜉01subscript𝜉11subscript𝐱100subscript𝐱201\mathbb{E}\left(\frac{1}{\xi\_{01}+\xi\_{11}}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1)\right)
since A𝐴A does not bring any new information about the number of k𝑘k with xk2=1superscriptsubscript𝑥𝑘21x\_{k}^{2}=1 given 𝐱1=(0,0),𝐱2=(0,1)formulae-sequencesubscript𝐱100subscript𝐱201\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1). So, according to Lemma [4](#Thmlem4 "Lemma 4 ‣ Bias of the model ℎ¹+ℎ² ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features") below, the second factor equals 2n−1​(1+O​(2−n))2𝑛11𝑂superscript2𝑛\frac{2}{n-1}(1+O(2^{-n})). Finally, we obtain

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(ξ00​ξ01(ξ00+ξ01)​(ξ01+ξ11))=n​(n−1)42​4n​(n−1)​(1−1n−1)+O​(2−n)=14​(1−1n−1)+O​(2−n).𝔼subscript𝜉00subscript𝜉01subscript𝜉00subscript𝜉01subscript𝜉01subscript𝜉11𝑛𝑛1superscript424𝑛𝑛111𝑛1𝑂superscript2𝑛1411𝑛1𝑂superscript2𝑛\mathbb{E}\left(\frac{\xi\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi\_{01}+\xi\_{11})}\right)\\ =\frac{n(n-1)}{4^{2}}\frac{4}{n(n-1)}\left(1-\frac{1}{n-1}\right)+O(2^{-n})=\frac{1}{4}\left(1-\frac{1}{n-1}\right)+O(2^{-n}). |  |

□□\Box

###### Lemma 4

𝔼​(1ξ01+ξ11∣𝐱1=(0,0),𝐱2=(0,1))=2n−1−12n−2​(n−1)𝔼formulae-sequenceconditional1subscript𝜉01subscript𝜉11subscript𝐱100subscript𝐱2012𝑛11superscript2𝑛2𝑛1\mathbb{E}\left(\frac{1}{\xi\_{01}+\xi\_{11}}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1)\right)=\frac{2}{n-1}-\frac{1}{2^{n-2}(n-1)} .

Proof.
Similarly to the proof of Lemma [2](#Thmprop2 "Proposition 2 ‣ Prediction shift of ℎ¹ ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features"), we have

|  |  |  |
| --- | --- | --- |
|  | P(ξ01+ξ11=k∣𝐱1=(0,0),𝐱2=(0,1))=(n−2k−1)2−(n−2).\mathrm{P}(\xi\_{01}+\xi\_{11}=k\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1))=\binom{n-2}{k-1}2^{-(n-2)}\,. |  |

Therefore, we get

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(1ξ01+ξ11∣𝐱1=(0,0),𝐱2=(0,1))=∑k=1n−11k​(n−2k−1)​2−(n−2)=2−(n−2)n−1​∑k=1n−1(n−1k)=2n−1−12n−2​(n−1).𝔼formulae-sequenceconditional1subscript𝜉01subscript𝜉11subscript𝐱100subscript𝐱201superscriptsubscript𝑘1𝑛11𝑘binomial𝑛2𝑘1superscript2𝑛2superscript2𝑛2𝑛1superscriptsubscript𝑘1𝑛1binomial𝑛1𝑘2𝑛11superscript2𝑛2𝑛1\mathbb{E}\left(\frac{1}{\xi\_{01}+\xi\_{11}}\mid\mathbf{x}\_{1}=(0,0),\mathbf{x}\_{2}=(0,1)\right)=\sum\_{k=1}^{n-1}\frac{1}{k}\binom{n-2}{k-1}2^{-(n-2)}\\ =\frac{2^{-(n-2)}}{n-1}\sum\_{k=1}^{n-1}\binom{n-1}{k}=\frac{2}{n-1}-\frac{1}{2^{n-2}(n-1)}\,. |  |

□□\Box

### A.2 Proof for independently sampled 𝒟1subscript𝒟1\mathcal{D}\_{1} and 𝒟2subscript𝒟2\mathcal{D}\_{2}

Assume that we have an additional sample 𝒟2={𝐱n+k}k=1..n\mathcal{D}\_{2}=\{\mathbf{x}\_{n+k}\}\_{k=1..n} for building h2superscriptℎ2h^{2}. Now A𝐴A denotes the event that each leaf in h1superscriptℎ1h^{1} contains at least one example from 𝒟1subscript𝒟1\mathcal{D}\_{1} and each leaf in h2superscriptℎ2h^{2} contains at least one example from 𝒟2subscript𝒟2\mathcal{D}\_{2}.

###### Proposition 4

If h2superscriptℎ2h^{2} is built using dataset 𝒟2subscript𝒟2\mathcal{D}\_{2}, then

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(h1​(s,t)+h2​(s,t)∣A)=f∗​(s,t).𝔼superscriptℎ1𝑠𝑡conditionalsuperscriptℎ2𝑠𝑡𝐴superscript𝑓𝑠𝑡\mathbb{E}(h^{1}(s,t)+h^{2}(s,t)\mid A)=f^{\*}(s,t)\,. |  |

Proof.

Let us denote by ξs​t′subscriptsuperscript𝜉′𝑠𝑡\xi^{\prime}\_{st} the number of examples 𝐱n+ksubscript𝐱𝑛𝑘\mathbf{x}\_{n+k} that are equal to (s,t)𝑠𝑡(s,t), k=1,…,n𝑘

1…𝑛k=1,\ldots,n.

First, we need to derive the expectation 𝔼​(h2​(s,t))𝔼superscriptℎ2𝑠𝑡\mathbb{E}(h^{2}(s,t)) of h2superscriptℎ2h^{2} on a test example 𝐱=(s,t)𝐱𝑠𝑡\mathbf{x}=(s,t). Similarly to the proof of Proposition [3](#Thmprop3 "Proposition 3 ‣ Bias of the model ℎ¹+ℎ² ‣ A.1 Proof for the case 𝒟₁=𝒟₂ ‣ Appendix A Proof of Theorem 1 ‣ CatBoost: unbiased boosting with categorical features"), we get

|  |  |  |
| --- | --- | --- |
|  | h2​(s,0)=−c2​(ξ00′​ξ01(ξ00+ξ01)​(ξ00′+ξ10′)+ξ10′​ξ11(ξ10+ξ11)​(ξ00′+ξ10′)),superscriptℎ2𝑠0subscript𝑐2subscriptsuperscript𝜉′00subscript𝜉01subscript𝜉00subscript𝜉01subscriptsuperscript𝜉′00subscriptsuperscript𝜉′10subscriptsuperscript𝜉′10subscript𝜉11subscript𝜉10subscript𝜉11subscriptsuperscript𝜉′00subscriptsuperscript𝜉′10h^{2}(s,0)=-c\_{2}\left(\frac{\xi^{\prime}\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi^{\prime}\_{00}+\xi^{\prime}\_{10})}+\frac{\xi^{\prime}\_{10}\xi\_{11}}{(\xi\_{10}+\xi\_{11})(\xi^{\prime}\_{00}+\xi^{\prime}\_{10})}\right)\,, |  |

|  |  |  |
| --- | --- | --- |
|  | h2​(s,1)=c2​(ξ00​ξ01′(ξ00+ξ01)​(ξ01′+ξ11′)+ξ10​ξ11′(ξ10+ξ11)​(ξ01′+ξ11′)).superscriptℎ2𝑠1subscript𝑐2subscript𝜉00subscriptsuperscript𝜉′01subscript𝜉00subscript𝜉01subscriptsuperscript𝜉′01subscriptsuperscript𝜉′11subscript𝜉10subscriptsuperscript𝜉′11subscript𝜉10subscript𝜉11subscriptsuperscript𝜉′01subscriptsuperscript𝜉′11h^{2}(s,1)=c\_{2}\left(\frac{\xi\_{00}\xi^{\prime}\_{01}}{(\xi\_{00}+\xi\_{01})(\xi^{\prime}\_{01}+\xi^{\prime}\_{11})}+\frac{\xi\_{10}\xi^{\prime}\_{11}}{(\xi\_{10}+\xi\_{11})(\xi^{\prime}\_{01}+\xi^{\prime}\_{11})}\right)\,. |  |

Due to the symmetries, the expected values of all four fractions above are equal. Also, due to the independence of ξi​jsubscript𝜉𝑖𝑗\xi\_{ij} and ξk​l′subscriptsuperscript𝜉′𝑘𝑙\xi^{\prime}\_{kl}, we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼​(ξ00′​ξ01(ξ00+ξ01)​(ξ00′+ξ10′)∣A)=𝔼​(ξ01ξ00+ξ01∣A)​𝔼​(ξ00′ξ00′+ξ10′∣A)=14.𝔼conditionalsubscriptsuperscript𝜉′00subscript𝜉01subscript𝜉00subscript𝜉01subscriptsuperscript𝜉′00subscriptsuperscript𝜉′10𝐴𝔼conditionalsubscript𝜉01subscript𝜉00subscript𝜉01𝐴𝔼conditionalsubscriptsuperscript𝜉′00subscriptsuperscript𝜉′00subscriptsuperscript𝜉′10𝐴14\mathbb{E}\left(\frac{\xi^{\prime}\_{00}\xi\_{01}}{(\xi\_{00}+\xi\_{01})(\xi^{\prime}\_{00}+\xi^{\prime}\_{10})}\mid A\right)=\mathbb{E}\left(\frac{\xi\_{01}}{\xi\_{00}+\xi\_{01}}\mid A\right)\mathbb{E}\left(\frac{\xi^{\prime}\_{00}}{\xi^{\prime}\_{00}+\xi^{\prime}\_{10}}\mid A\right)=\frac{1}{4}\,. |  |

Therefore, 𝔼​(h2​(s,0)∣A)=−c22𝔼conditionalsuperscriptℎ2𝑠0𝐴subscript𝑐22\mathbb{E}(h^{2}(s,0)\mid A)=-\frac{c\_{2}}{2} and 𝔼​(h2​(s,1)∣A)=c22𝔼conditionalsuperscriptℎ2𝑠1𝐴subscript𝑐22\mathbb{E}(h^{2}(s,1)\mid A)=\frac{c\_{2}}{2}.

Summing up, 𝔼​(h2​(s,t)∣A)=c2​t−c22𝔼conditionalsuperscriptℎ2𝑠𝑡𝐴subscript𝑐2𝑡subscript𝑐22\mathbb{E}(h^{2}(s,t)\mid A)=c\_{2}t-\frac{c\_{2}}{2} and 𝔼​(h1​(s,t)+h2​(s,t)∣A)=c1​s+c2​t𝔼superscriptℎ1𝑠𝑡conditionalsuperscriptℎ2𝑠𝑡𝐴subscript𝑐1𝑠subscript𝑐2𝑡\mathbb{E}(h^{1}(s,t)+h^{2}(s,t)\mid A)=c\_{1}s+c\_{2}t.
□□\Box

## Appendix B Formal description of CatBoost algorithm

In this section, we formally describe the CatBoost algorithm introduced in Section [5](#S5 "5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features"). In Algorithm [3](#algorithm3 "In Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features"), we provide more information on particular details including the speeding up trick introduced in paragraph “Complexity”. The key step of the CatBoost algorithm is the procedure of building a tree described in detail in Function [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features"). To obtain the formal description of the CatBoost algorithm without the speeding up trick, one should replace ⌈log2⁡n⌉subscript2𝑛\lceil\log\_{2}n\rceil by n𝑛n in line 6 of Algorithm [3](#algorithm3 "In Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features") and use Algorithm [2](#algorithm2 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features") instead of Function [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features").

We use Function G​e​t​L​e​a​f​(𝐱,T,σr)𝐺𝑒𝑡𝐿𝑒𝑎𝑓𝐱𝑇subscript𝜎𝑟GetLeaf(\mathbf{x},T,\sigma\_{r}) to describe how examples are matched to leaves l​e​a​fr​(i)𝑙𝑒𝑎subscript𝑓𝑟𝑖leaf\_{r}(i). Given an example with features 𝐱𝐱\mathbf{x}, we calculate ordered TS on the basis of the permutation σrsubscript𝜎𝑟\sigma\_{r} and then choose the leaf of tree T𝑇T corresponding to features 𝐱𝐱\mathbf{x} enriched by the obtained ordered TS. Using A​p​p​l​y​M​o​d​e𝐴𝑝𝑝𝑙𝑦𝑀𝑜𝑑𝑒ApplyMode instead of a permutation in function G​e​t​L​e​a​f𝐺𝑒𝑡𝐿𝑒𝑎𝑓GetLeaf in line 15 of Algorithm [3](#algorithm3 "In Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features") means that we use TS calculated on the whole training data to apply the trained model on a new example.

input :   {(𝐱i,yi)}i=1nsuperscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n}, I𝐼I, α𝛼\alpha, L𝐿L, s𝑠s, M​o​d​e𝑀𝑜𝑑𝑒Mode

0
σr←←subscript𝜎𝑟absent\sigma\_{r}\leftarrow random permutation of [1,n]1𝑛[1,n] for r=0..sr=0..s;

1
M0​(i)←0←subscript𝑀0𝑖0M\_{0}(i)\leftarrow 0 for i=1..ni=1..n;

2
if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

3
Mr​(i)←0←subscript𝑀𝑟𝑖0M\_{r}(i)\leftarrow 0 for r=1..sr=1..s, i:σr​(i)≤2j+1:𝑖subscript𝜎𝑟𝑖superscript2𝑗1i:\sigma\_{r}(i)\leq 2^{j+1};

4if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

5
for *j←1←𝑗1j\leftarrow 1 to ⌈log2⁡n⌉subscript2𝑛\lceil\log\_{2}n\rceil* do

6
Mr,j​(i)←0←subscript𝑀

𝑟𝑗𝑖0M\_{r,j}(i)\leftarrow 0 for r=1..sr=1..s,
i=1..2j+1𝑖superscript1..2𝑗1i=1..2^{j+1};

8for *t𝑡t ←1←absent1\leftarrow 1 to I𝐼I* do

9
Tt,{Mr}r=1s←B​u​i​l​d​T​r​e​e​({Mr}r=1s,{(𝐱i,yi)}i=1n,α,L,{σi}i=1s,M​o​d​e)←

subscript𝑇𝑡superscriptsubscriptsubscript𝑀𝑟𝑟1𝑠
𝐵𝑢𝑖𝑙𝑑𝑇𝑟𝑒𝑒superscriptsubscriptsubscript𝑀𝑟𝑟1𝑠superscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛𝛼𝐿superscriptsubscriptsubscript𝜎𝑖𝑖1𝑠𝑀𝑜𝑑𝑒T\_{t},\ \{M\_{r}\}\_{r=1}^{s}\leftarrow BuildTree(\{M\_{r}\}\_{r=1}^{s},\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n},\alpha,L,\{\sigma\_{i}\}\_{i=1}^{s},Mode);

10
l​e​a​f0​(i)←G​e​t​L​e​a​f​(𝐱i,Tt,σ0)←𝑙𝑒𝑎subscript𝑓0𝑖𝐺𝑒𝑡𝐿𝑒𝑎𝑓subscript𝐱𝑖subscript𝑇𝑡subscript𝜎0leaf\_{0}(i)\leftarrow GetLeaf(\mathbf{x}\_{i},T\_{t},\sigma\_{0}) for i=1..ni=1..n;

11
g​r​a​d0←C​a​l​c​G​r​a​d​i​e​n​t​(L,M0,y)←𝑔𝑟𝑎subscript𝑑0𝐶𝑎𝑙𝑐𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡𝐿subscript𝑀0𝑦grad\_{0}\leftarrow CalcGradient(L,M\_{0},y);

12
foreach *leaf j𝑗j in Ttsubscript𝑇𝑡T\_{t}* do

13
bjt←−avg(grad0(i)b\_{j}^{t}\leftarrow-\mathrm{avg}(grad\_{0}(i) for i:leaf0(i)=j)i:\ leaf\_{0}(i)=j);

15M0​(i)←M0​(i)+α​bl​e​a​f0​(i)t←subscript𝑀0𝑖subscript𝑀0𝑖𝛼superscriptsubscript𝑏𝑙𝑒𝑎subscript𝑓0𝑖𝑡M\_{0}(i)\leftarrow M\_{0}(i)+\alpha b\_{leaf\_{0}(i)}^{t} for i=1..ni=1..n;

17return F​(𝐱)=∑t=1I∑jα​bjt​𝟙{G​e​t​L​e​a​f​(𝐱,Tt,A​p​p​l​y​M​o​d​e)=j}𝐹𝐱superscriptsubscript𝑡1𝐼subscript𝑗𝛼superscriptsubscript𝑏𝑗𝑡subscript1𝐺𝑒𝑡𝐿𝑒𝑎𝑓𝐱subscript𝑇𝑡𝐴𝑝𝑝𝑙𝑦𝑀𝑜𝑑𝑒𝑗F(\mathbf{x})=\sum\_{t=1}^{I}\sum\_{j}\alpha\,b\_{j}^{t}\mathbbm{1}\_{\{GetLeaf(\mathbf{x},T\_{t},ApplyMode)=j\}};

Algorithm 3 CatBoost

[htbp]
  

 

input :
M𝑀M,{(𝐱i,yi)}i=1nsuperscriptsubscriptsubscript𝐱𝑖subscript𝑦𝑖𝑖1𝑛\{(\mathbf{x}\_{i},y\_{i})\}\_{i=1}^{n}, α𝛼\alpha, L𝐿L, {σi}i=1ssuperscriptsubscriptsubscript𝜎𝑖𝑖1𝑠\{\sigma\_{i}\}\_{i=1}^{s}, M​o​d​e𝑀𝑜𝑑𝑒Mode

g​r​a​d←C​a​l​c​G​r​a​d​i​e​n​t​(L,M,y)←𝑔𝑟𝑎𝑑𝐶𝑎𝑙𝑐𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡𝐿𝑀𝑦grad\leftarrow CalcGradient(L,M,y);

1
r←r​a​n​d​o​m​(1,s)←𝑟𝑟𝑎𝑛𝑑𝑜𝑚1𝑠r\leftarrow random(1,s);

2
if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

3
G←(gradr(i) for i=1..n)G\leftarrow(grad\_{r}(i)\mbox{ for }i=1..n);

4if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

5
G←(gradr,⌊log2⁡(σr​(i)−1)⌋(i) for i=1..n)G\leftarrow(grad\_{r,\lfloor\log\_{2}(\sigma\_{r}(i)-1)\rfloor}(i)\mbox{ for }i=1..n);

6T←←𝑇absentT\leftarrow empty tree;

7
foreach *step of top-down procedure* do

8
foreach *candidate split c𝑐c*  do

9
Tc←←subscript𝑇𝑐absentT\_{c}\leftarrow add split c𝑐c to T𝑇T;

10
l​e​a​fr​(i)←G​e​t​L​e​a​f​(𝐱i,Tc,σr)←𝑙𝑒𝑎subscript𝑓𝑟𝑖𝐺𝑒𝑡𝐿𝑒𝑎𝑓subscript𝐱𝑖subscript𝑇𝑐subscript𝜎𝑟leaf\_{r}(i)\leftarrow GetLeaf(\mathbf{x}\_{i},T\_{c},\sigma\_{r}) for i=1..ni=1..n;

11
if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

12
Δ(i)←avg(gradr(p)\Delta(i)\leftarrow\mathrm{avg}(grad\_{r}(p) for p:leafr(p)=leafr(i))p:\ leaf\_{r}(p)=leaf\_{r}(i))  for i=1..ni=1..n;

14if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

15
Δ(i)←avg(gradr,⌊log2⁡(σr​(i)−1)⌋(p)\Delta(i)\leftarrow\mathrm{avg}(grad\_{r,\lfloor\log\_{2}(\sigma\_{r}(i)-1)\rfloor}(p) for p:leafr(p)=leafr(i),σr(p)<σr(i))p:\ leaf\_{r}(p)=leaf\_{r}(i),\sigma\_{r}(p)<\sigma\_{r}(i))  for i=1..ni=1..n;

16l​o​s​s​(Tc)←cos⁡(Δ,G)←𝑙𝑜𝑠𝑠subscript𝑇𝑐Δ𝐺loss(T\_{c})\leftarrow\cos(\Delta,G)

17T←arg​minTc⁡(l​o​s​s​(Tc))←𝑇subscriptargminsubscript𝑇𝑐𝑙𝑜𝑠𝑠subscript𝑇𝑐T\leftarrow\operatorname\*{arg\,min}\_{T\_{c}}(loss(T\_{c}))

18l​e​a​fr′​(i)←G​e​t​L​e​a​f​(𝐱i,T,σr′)←𝑙𝑒𝑎subscript𝑓superscript𝑟′𝑖𝐺𝑒𝑡𝐿𝑒𝑎𝑓subscript𝐱𝑖𝑇subscript𝜎superscript𝑟′leaf\_{r^{\prime}}(i)\leftarrow GetLeaf(\mathbf{x}\_{i},T,\sigma\_{r^{\prime}}) for r′=1..sr^{\prime}=1..s, i=1..ni=1..n;

19
if *M​o​d​e=P​l​a​i​n𝑀𝑜𝑑𝑒𝑃𝑙𝑎𝑖𝑛Mode=Plain* then

20
Mr′(i)←Mr′(i)−αavg(gradr′(p)M\_{r^{\prime}}(i)\leftarrow M\_{r^{\prime}}(i)-\alpha\,\mathrm{avg}(grad\_{r^{\prime}}(p) for p:leafr′(p)=leafr′(i))p:\ leaf\_{r^{\prime}}(p)=leaf\_{r^{\prime}}(i)) for r′=1..sr^{\prime}=1..s, i=1..ni=1..n;

22if *M​o​d​e=O​r​d​e​r​e​d𝑀𝑜𝑑𝑒𝑂𝑟𝑑𝑒𝑟𝑒𝑑Mode=Ordered* then

23
for *j←1←𝑗1j\leftarrow 1 to ⌈log2⁡n⌉subscript2𝑛\lceil\log\_{2}n\rceil* do

24
Mr′,j(i)←Mr′,j(i)−αavg(gradr′,j(p)M\_{r^{\prime},j}(i)\leftarrow M\_{r^{\prime},j}(i)-\alpha\,\mathrm{avg}(grad\_{r^{\prime},j}(p) for p:leafr′(p)=leafr′(i),σr′(p)≤2j)p:\ leaf\_{r^{\prime}}(p)=leaf\_{r^{\prime}}(i),\sigma\_{r^{\prime}}(p)\leq 2^{j}) for r′=1..sr^{\prime}=1..s,  i:σr′​(i)≤2j+1:𝑖subscript𝜎superscript𝑟′𝑖superscript2𝑗1i:\sigma\_{r^{\prime}}(i)\leq 2^{j+1};

26return *T,M

𝑇𝑀T,M*
B​u​i​l​d​T​r​e​e𝐵𝑢𝑖𝑙𝑑𝑇𝑟𝑒𝑒BuildTree()

## Appendix C Time complexity analysis

### C.1 Theoretical analysis

We present the computational complexity of different components of any of the two modes of CatBoost per one iteration in Table [5](#A3.T5 "Table 5 ‣ C.1 Theoretical analysis ‣ Appendix C Time complexity analysis ‣ CatBoost: unbiased boosting with categorical features").

Table 5: Computational complexity.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Procedure | CalcGradient | Build T𝑇T | Calc values bjtsubscriptsuperscript𝑏𝑡𝑗b^{t}\_{j} | Update M𝑀M | Calc ordered TS |
| Complexity  for iteration t𝑡t | O​(s⋅n)𝑂⋅𝑠𝑛O(s\cdot n) | O​(|C|⋅n)𝑂⋅𝐶𝑛O(|C|\cdot n) | O​(n)𝑂𝑛O(n) | O​(s⋅n)𝑂⋅𝑠𝑛O(s\cdot n) | O​(NT​S,t⋅n)𝑂⋅subscript𝑁  𝑇𝑆𝑡𝑛O(N\_{TS,t}\cdot n) |

We first prove these asymptotics for the Ordered mode. For this purpose, we estimate the number Np​r​e​dsubscript𝑁𝑝𝑟𝑒𝑑N\_{pred} of predictions Mr,j​(i)subscript𝑀

𝑟𝑗𝑖M\_{r,j}(i) to be maintained:

|  |  |  |
| --- | --- | --- |
|  | Np​r​e​d=(s+1)⋅∑j=1⌈log2⁡n⌉2j+1<(s+1)⋅2log2⁡n+3=8​(s+1)​n.subscript𝑁𝑝𝑟𝑒𝑑⋅𝑠1superscriptsubscript𝑗1subscript2𝑛superscript2𝑗1⋅𝑠1superscript2subscript2𝑛38𝑠1𝑛N\_{pred}=(s+1)\cdot\sum\_{j=1}^{\lceil\log\_{2}n\rceil}2^{j+1}<(s+1)\cdot 2^{\log\_{2}n+3}=8(s+1)n\,. |  |

Then, obviously, the complexity of CalcGradient is O​(Np​r​e​d)=O​(s⋅n)𝑂subscript𝑁𝑝𝑟𝑒𝑑𝑂⋅𝑠𝑛O(N\_{pred})=O(s\cdot n). The complexity of leaf values calculation is O​(n)𝑂𝑛O(n), since each example i𝑖i is included only in averaging operation in leaf l​e​a​f0​(i)𝑙𝑒𝑎subscript𝑓0𝑖leaf\_{0}(i).

Calculation of the ordered TS for one categorical feature can be performed sequentially in the order of the permutation by n𝑛n additive operations for calculation of n𝑛n partial sums and n𝑛n division operations. Thus, the overall complexity of the procedure is O​(NT​S,t⋅n)𝑂⋅subscript𝑁

𝑇𝑆𝑡𝑛O(N\_{TS,t}\cdot n), where NT​S,tsubscript𝑁

𝑇𝑆𝑡N\_{TS,t} is the number of TS which were not calculated on the previous iterations. Since the leaf values Δ​(i)Δ𝑖\Delta(i) calculated in line 15 of Function [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features") can be considered as ordered TS, where gradients play the role of targets, the complexity of building a tree T𝑇T is O​(|C|⋅n)𝑂⋅𝐶𝑛O(|C|\cdot n), where C𝐶C is the set of candidate splits to be considered at the given iteration.
Finally, for updating the supporting models (lines 22-23 in Function [B](#A2 "Appendix B Formal description of CatBoost algorithm ‣ CatBoost: unbiased boosting with categorical features")), we need to perform one averaging operation for each j=1,…,⌈log2⁡n⌉𝑗

1…subscript2𝑛j=1,\ldots,\lceil\log\_{2}n\rceil, and each maintained gradient g​r​a​dr′,j​(p)𝑔𝑟𝑎subscript𝑑

superscript𝑟′𝑗𝑝grad\_{r^{\prime},j}(p) is included in one averaging operation. Thus, the number of operations is bounded by the number of the maintained gradients g​r​a​dr′,j​(p)𝑔𝑟𝑎subscript𝑑

superscript𝑟′𝑗𝑝grad\_{r^{\prime},j}(p), which is equal to Np​r​e​d=O​(s⋅n)subscript𝑁𝑝𝑟𝑒𝑑𝑂⋅𝑠𝑛N\_{pred}=O(s\cdot n).

To finish the proof, note that any component of the Plain mode is not less efficient than the same one of the Ordered mode but, at the same time, cannot be more efficient than corresponding asymptotics from Table [5](#A3.T5 "Table 5 ‣ C.1 Theoretical analysis ‣ Appendix C Time complexity analysis ‣ CatBoost: unbiased boosting with categorical features").

### C.2 Empirical analysis

It is quite hard to compare different boosting libraries in terms of training speed. Every algorithm has a vast number of parameters which affect training speed, quality and model size in a non-obvious way. Different libraries have their unique quality/training speed trade-off’s and they cannot be compared without domain knowledge (e.g., is 0.5%percent0.50.5\% of quality metric worth it to train a model 3-4 times slower?). Plus for each library it is possible to obtain almost the same quality with different ensemble sizes and parameters. As a result, one cannot compare libraries by time needed to obtain a certain level of quality. As a result, we could give only some insights of how fast our implementation could train a model of a fixed size. We use Epsilon dataset and we measure mean tree construction time one can achieve without using feature subsampling and/or bagging by CatBoost (both Ordered and Plain modes), XGBoost (we use histogram-based version, which is faster) and LightGBM. For XGBoost and CatBoost we use the default tree depth equal to 6, for LightGBM we set leaves count to 64 to have comparable results.
We run all experiments on the same machine with Intel Xeon E3-12xx 2.6GHz, 16 cores, 64GB RAM and run all algorithms with 16 threads.

We set such learning rate that algorithms start to overfit approximately after constructing about 7000 trees and measure the average time to train ensembles of 8000 trees. Mean tree construction time is presented in Table [6](#A3.T6 "Table 6 ‣ C.2 Empirical analysis ‣ Appendix C Time complexity analysis ‣ CatBoost: unbiased boosting with categorical features"). Note that CatBoost Plain and LightGBM are the fastest ones followed by Ordered mode, which is about 1.7 times slower, which is expected.

Table 6: Comparison of running times on Epsilon

|  |  |
| --- | --- |
|  | time per tree |
| CatBoost Plain | 1.1 s |
| CatBoost Ordered | 1.9 s |
| XGBoost | 3.9 s |
| LightGBM | 1.1 s |

Finally, let us note that CatBoost has a highly efficient GPU implementation. The detailed description and comparison of the running times are beyond the scope of the current article, but these experiments can be found on the corresponding GitHub page.101010<https://github.com/catboost/benchmarks/tree/master/gpu_training>

## Appendix D Experimental setup

### D.1 Description of the datasets

The datasets used in our experiments are described in Table [7](#A4.T7 "Table 7 ‣ D.1 Description of the datasets ‣ Appendix D Experimental setup ‣ CatBoost: unbiased boosting with categorical features").

Table 7: Description of the datasets.

|  |  |  |  |
| --- | --- | --- | --- |
| Dataset name | Instances | Features | Description |
| Adult111111<https://archive.ics.uci.edu/ml/datasets/Adult> | 48842 | 15 | Prediction task is to determine whether a person makes over 50K a year. Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: (AAGE>16) and (AGI>100) and (AFNLWGT>1) and (HRSWK>0) |
| Amazon121212<https://www.kaggle.com/c/amazon-employee-access-challenge> | 32769 | 10 | Data from the Kaggle Amazon Employee challenge. |
| Click Prediction131313<http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2> | 399482 | 12 | This data is derived from the 2012 KDD Cup. The data is subsampled to 1% of the original number of instances, downsampling the majority class (click=0) so that the target feature is reasonably balanced (5 to 1). The data is about advertisements shown alongside search results in a search engine, and whether or not people clicked on these ads. The task is to build the best possible model to predict whether a user will click on a given ad. |
| Epsilon141414<https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html> | 400000 | 2000 | PASCAL Challenge 2008. |
| KDD appetency151515<http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data> | 50000 | 231 | Small version of KDD 2009 Cup data. |
| KDD churn161616<http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data> | 50000 | 231 | Small version of KDD 2009 Cup data. |
| KDD Internet171717<https://kdd.ics.uci.edu/databases/internet_usage/internet_usage.html> | 10108 | 69 | Binarized version of the original dataset. The multi-class target feature is converted to a two-class nominal target feature by re-labeling the majority class as positive (‘P’) and all others as negative (‘N’). Originally converted by Quan Sun. |
| KDD upselling181818<http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data> | 50000 | 231 | Small version of KDD 2009 Cup data. |
| Kick prediction191919<https://www.kaggle.com/c/DontGetKicked> | 72983 | 36 | Data from “Don’t Get Kicked!” Kaggle challenge. |

### D.2 Experimental settings

In our experiments, we evaluate different modifications of CatBoost and two popular gradient boosting libraries: LightGBM and XGBoost. All the code needed for reproducing our experiments is published on our GitHub202020<https://github.com/catboost/benchmarks/tree/master/quality_benchmarks>.

##### Train-test splits

Each dataset was randomly split into training set (80%) and test set (20%). We denote them as Df​u​l​l​\_​t​r​a​i​nsubscript𝐷𝑓𝑢𝑙𝑙\_𝑡𝑟𝑎𝑖𝑛D\_{full\\_train} and Dt​e​s​tsubscript𝐷𝑡𝑒𝑠𝑡D\_{test}.

We use 5-fold cross-validation to tune parameters of each model on the training set. Accordingly, Df​u​l​l​\_​t​r​a​i​nsubscript𝐷𝑓𝑢𝑙𝑙\_𝑡𝑟𝑎𝑖𝑛D\_{full\\_train} is randomly split into 5 equally sized parts D1,…,D5

subscript𝐷1…subscript𝐷5D\_{1},\dots,D\_{5} (sampling is stratified by classes). These parts are used to construct 5 training and validation sets: Dit​r​a​i​n=∪j≠iDjsuperscriptsubscript𝐷𝑖𝑡𝑟𝑎𝑖𝑛subscript𝑗𝑖subscript𝐷𝑗D\_{i}^{train}=\cup\_{j\neq i}D\_{j} and Div​a​l=Disuperscriptsubscript𝐷𝑖𝑣𝑎𝑙subscript𝐷𝑖D\_{i}^{val}=D\_{i} for 1≤i≤51𝑖51\leq i\leq 5.

##### Preprocessing

We applied the following steps to datasets with missing values:

* •

  For categorical variables, missing values are replaced with a special value, i.e., we treat missing values as a special category;
* •

  For numerical variables, missing values are replaced with zeros, and a binary dummy feature for each imputed feature is added.

For XGBoost, LightGBM and the raw setting of CatBoost (see Appendix [G](#A7 "Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features")), we perform the following preprocessing of categorical features. For each pair of datasets (Dit​r​a​i​n,Div​a​l)subscriptsuperscript𝐷𝑡𝑟𝑎𝑖𝑛𝑖subscriptsuperscript𝐷𝑣𝑎𝑙𝑖(D^{train}\_{i},\ D^{val}\_{i}), i=1,…,5𝑖

1…5i=1,\ldots,5, and (Df​u​l​l​\_​t​r​a​i​n,Dt​e​s​t)subscript𝐷𝑓𝑢𝑙𝑙\_𝑡𝑟𝑎𝑖𝑛subscript𝐷𝑡𝑒𝑠𝑡(D\_{full\\_train},D\_{test}), we preprocess the categorical features by calculating ordered TS (described in Section [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features")) on the basis of a random permutation of the examples of the first (training) dataset. All the permutations are generated independently. The resulting values of TS are considered as numerical features by any algorithm to be evaluated.

##### Parameter Tuning

We tune all the key parameters of each algorithm by 50 steps of the sequential optimization algorithm Tree Parzen Estimator implemented in Hyperopt library212121<https://github.com/hyperopt/hyperopt> (mode algo=tpe.suggest) by minimizing logloss. Below is the list of the tuned parameters and their distributions the optimization algorithm started from:

XGBoost:

* •

  ‘eta’: Log-uniform distribution [e−7,1]superscript𝑒71[e^{-7},1]
* •

  ‘max\_depth’: Discrete uniform distribution [2,10]210[2,10]
* •

  ‘subsample’: Uniform [0.5,1]0.51[0.5,1]
* •

  ‘colsample\_bytree’: Uniform [0.5,1]0.51[0.5,1]
* •

  ‘colsample\_bylevel’: Uniform [0.5,1]0.51[0.5,1]
* •

  ‘min\_child\_weight’: Log-uniform distribution [e−16,e5]superscript𝑒16superscript𝑒5[e^{-16},e^{5}]
* •

  ‘alpha’: Mixed: 0.5⋅0.5\,\cdot Degenerate at 0 + 0.5⋅0.5\,\cdot Log-uniform distribution [e−16,e2]superscript𝑒16superscript𝑒2[e^{-16},e^{2}]
* •

  ‘lambda’: Mixed: 0.5⋅0.5\,\cdot Degenerate at 0 + 0.5⋅0.5\,\cdot Log-uniform distribution [e−16,e2]superscript𝑒16superscript𝑒2[e^{-16},e^{2}]
* •

  ‘gamma’: Mixed: 0.5⋅0.5\,\cdot Degenerate at 0 + 0.5⋅0.5\,\cdot Log-uniform distribution [e−16,e2]superscript𝑒16superscript𝑒2[e^{-16},e^{2}]

LightGBM:

* •

  ‘learning\_rate’: Log-uniform distribution [e−7,1]superscript𝑒71[e^{-7},1]
* •

  ‘num\_leaves’ : Discrete log-uniform distribution [1,e7]1superscript𝑒7[1,e^{7}]
* •

  ‘feature\_fraction’: Uniform [0.5,1]0.51[0.5,1]
* •

  ‘bagging\_fraction’: Uniform [0.5,1]0.51[0.5,1]
* •

  ‘min\_sum\_hessian\_in\_leaf’: Log-uniform distribution [e−16,e5]superscript𝑒16superscript𝑒5[e^{-16},e^{5}]
* •

  ‘min\_data\_in\_leaf’: Discrete log-uniform distribution [1,e6]1superscript𝑒6[1,e^{6}]
* •

  ‘lambda\_l1’: Mixed: 0.5⋅0.5\,\cdot Degenerate at 0 + 0.5⋅0.5\,\cdot Log-uniform distribution [e−16,e2]superscript𝑒16superscript𝑒2[e^{-16},e^{2}]
* •

  ‘lambda\_l2’: Mixed: 0.5⋅0.5\,\cdot Degenerate at 0 + 0.5⋅0.5\,\cdot Log-uniform distribution [e−16,e2]superscript𝑒16superscript𝑒2[e^{-16},e^{2}]

CatBoost:

* •

  ‘learning\_rate’: Log-uniform distribution [e−7,1]superscript𝑒71[e^{-7},1]
* •

  ‘random\_strength’: Discrete uniform distribution over a set {1,20}120\{1,20\}
* •

  ‘one\_hot\_max\_size’: Discrete uniform distribution over a set {0,25}025\{0,25\}
* •

  ‘l2\_leaf\_reg’: Log-uniform distribution [1,10]110[1,10]
* •

  ‘bagging\_temperature’: Uniform [0,1]01[0,1]
* •

  ‘gradient\_iterations’ : Discrete uniform distribution over a set {1,10}110\{1,10\}

Next, having fixed all other parameters, we perform exhaustive search for the number of trees in the interval [1,5000]15000[1,5000]. We collect logloss value for each training iteration from 1 to 5000 for each of the 5 folds. Then we choose the iteration with minimum logloss averaged over 5 folds.

For evaluation, each algorithm was run on the preprocessed training data Df​u​l​l​\_​t​r​a​i​nsubscript𝐷𝑓𝑢𝑙𝑙\_𝑡𝑟𝑎𝑖𝑛D\_{full\\_train} with the tuned parameters. The resulting model was evaluated on the preprocessed test set Dt​e​s​tsubscript𝐷𝑡𝑒𝑠𝑡D\_{test}.

##### Versions of the libraries

* •

  catboost (0.3)
* •

  xgboost (0.6)
* •

  scikit-learn (0.18.1)
* •

  scipy (0.19.0)
* •

  pandas (0.19.2)
* •

  numpy (1.12.1)
* •

  lightgbm (0.1)
* •

  hyperopt (0.0.2)
* •

  h2o (3.10.4.6)
* •

  R (3.3.3)

## Appendix E Analysis of iterated bagging

Based on the out-of-bag estimation [[2](#bib.bib2)], Breiman proposed iterated bagging [[3](#bib.bib3)] which simultaneously constructs K𝐾K models Fisubscript𝐹𝑖F\_{i}, i=1,…,K𝑖

1…𝐾i=1,\ldots,K, associated with K𝐾K independently bootstrapped subsamples 𝒟isubscript𝒟𝑖\mathcal{D}\_{i}. At t𝑡t-th step of the process, models Fitsubscriptsuperscript𝐹𝑡𝑖F^{t}\_{i} are grown from their predecessors Fit−1subscriptsuperscript𝐹𝑡1𝑖F^{t-1}\_{i} as follows. The current estimate Mjtsubscriptsuperscript𝑀𝑡𝑗M^{t}\_{j} at example j𝑗j is obtained as the average of the outputs of all models Fkt−1subscriptsuperscript𝐹𝑡1𝑘F^{t-1}\_{k} such that j∉𝒟k𝑗subscript𝒟𝑘j\notin\mathcal{D}\_{k}. The term hitsubscriptsuperscriptℎ𝑡𝑖h^{t}\_{i} is built as a predictor of the residuals rjt:=yj−Mjtassignsubscriptsuperscript𝑟𝑡𝑗subscript𝑦𝑗subscriptsuperscript𝑀𝑡𝑗r^{t}\_{j}:=y\_{j}-M^{t}\_{j} (targets minus current estimates) on 𝒟isubscript𝒟𝑖\mathcal{D}\_{i}. Finally, the models are updated: Fit:=Fit−1+hitassignsubscriptsuperscript𝐹𝑡𝑖subscriptsuperscript𝐹𝑡1𝑖subscriptsuperscriptℎ𝑡𝑖F^{t}\_{i}:=F^{t-1}\_{i}+h^{t}\_{i}. Unfortunately, the residuals rjtsubscriptsuperscript𝑟𝑡𝑗r^{t}\_{j} used in this procedure are not unshifted (in terms of Section [4.1](#S4.SS1 "4.1 Prediction shift ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features")), or unbiased (in terms of iterated bagging), because each model Fitsubscriptsuperscript𝐹𝑡𝑖F^{t}\_{i} depends on each observation (𝐱j,yj)subscript𝐱𝑗subscript𝑦𝑗(\mathbf{x}\_{j},y\_{j}) by construction. Indeed, although hktsubscriptsuperscriptℎ𝑡𝑘h^{t}\_{k} does not use yjsubscript𝑦𝑗y\_{j} directly, if j∉𝒟k𝑗subscript𝒟𝑘j\notin\mathcal{D}\_{k}, it still uses Mj′t−1subscriptsuperscript𝑀𝑡1superscript𝑗′M^{t-1}\_{j^{\prime}} for j′∈𝒟ksuperscript𝑗′subscript𝒟𝑘j^{\prime}\in\mathcal{D}\_{k}, which, in turn, can depend on (𝐱j,yj)subscript𝐱𝑗subscript𝑦𝑗(\mathbf{x}\_{j},y\_{j}).

Also note that computational complexity of this algorithm exceeds one of classic GBDT by factor of K𝐾K.

## Appendix F Ordered boosting with categorical features

In Sections [3.2](#S3.SS2 "3.2 Target statistics ‣ 3 Categorical features ‣ CatBoost: unbiased boosting with categorical features") and [4.2](#S4.SS2 "4.2 Ordered boosting ‣ 4 Prediction shift and ordered boosting ‣ CatBoost: unbiased boosting with categorical features"), we proposed to use some random permutations σc​a​tsubscript𝜎𝑐𝑎𝑡\sigma\_{cat} and σb​o​o​s​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{boost} of training examples for the TS calculation and for ordered boosting, respectively. Now, being combined in one algorithm, should these two permutations be somehow dependent? We argue that they should coincide. Otherwise, there exist examples 𝐱isubscript𝐱𝑖\mathbf{x}\_{i} and 𝐱jsubscript𝐱𝑗\mathbf{x}\_{j} such that σb​o​o​s​t​(i)<σb​o​o​s​t​(j)subscript𝜎𝑏𝑜𝑜𝑠𝑡𝑖subscript𝜎𝑏𝑜𝑜𝑠𝑡𝑗\sigma\_{boost}(i)<\sigma\_{boost}(j) and σc​a​t​(i)>σc​a​t​(j)subscript𝜎𝑐𝑎𝑡𝑖subscript𝜎𝑐𝑎𝑡𝑗\sigma\_{cat}(i)>\sigma\_{cat}(j). Then, the model Mσb​o​o​s​t​(j)subscript𝑀subscript𝜎𝑏𝑜𝑜𝑠𝑡𝑗M\_{\sigma\_{boost}(j)} is trained using TS features of, in particular, example 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}, which are calculated using yjsubscript𝑦𝑗y\_{j}. In general, it may shift the prediction Mσb​o​o​s​t​(j)​(𝐱j)subscript𝑀subscript𝜎𝑏𝑜𝑜𝑠𝑡𝑗subscript𝐱𝑗M\_{\sigma\_{boost}(j)}(\mathbf{x}\_{j}). To avoid such a shift, we set σc​a​t=σb​o​o​s​tsubscript𝜎𝑐𝑎𝑡subscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{cat}=\sigma\_{boost} in CatBoost. In the case of the ordered boosting (Algorithm [1](#algorithm1 "In 5 Practical implementation of ordered boosting ‣ CatBoost: unbiased boosting with categorical features")) with sliding window TS222222Ordered TS calculated on the basis of a fixed number of preceding examples (both for training and test examples).
it guarantees that the prediction Mσ​(i)−1​(𝐱i)subscript𝑀𝜎𝑖1subscript𝐱𝑖M\_{\sigma(i)-1}(\mathbf{x}\_{i}) is not shifted for i=1,…,n𝑖

1…𝑛i=1,\ldots,n, since, first, the target yisubscript𝑦𝑖y\_{i} was not used for training Mσ​(i)−1subscript𝑀𝜎𝑖1M\_{\sigma(i)-1} (neither for the TS calculation, nor for the gradient estimation) and, second, the distribution of TS x^isuperscript^𝑥𝑖\hat{x}^{i} conditioned by the target value is the same for a training example and a test example with the same value of feature xisuperscript𝑥𝑖x^{i}.

## Appendix G Experimental results

##### Comparison with baselines

In Section [6](#S6 "6 Experiments ‣ CatBoost: unbiased boosting with categorical features") we demonstrated that the strong setting of CatBoost, including ordered TS, Ordered mode and feature combinations, outperforms the baselines. Detailed experimental results of that comparison are presented in Table [8](#A7.T8 "Table 8 ‣ Comparison with baselines ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features").

Table 8: Comparison with baselines: logloss / zero-one loss, relative increase is presented in the brackets.

|  |  |  |  |
| --- | --- | --- | --- |
|  | CatBoost | LightGBM | XGBoost |
| Adult | 0.2695 / 0.1267 | 0.2760 (+2.4%) / 0.1291 (+1.9%) | 0.2754 (+2.2%) / 0.1280 (+1.0%) |
| Amazon | 0.1394 / 0.0442 | 0.1636 (+17%) / 0.0533 (+21%) | 0.1633 (+17%) / 0.0532 (+21%) |
| Click | 0.3917 / 0.1561 | 0.3963 (+1.2%) / 0.1580 (+1.2%) | 0.3962 (+1.2%) / 0.1581 (+1.2%) |
| Epsilon | 0.2647 / 0.1086 | 0.2703 (+1.5%) / 0.114 (+4.1%) | 0.2993 (+11%) / 0.1276 (+12%) |
| Appetency | 0.0715 / 0.01768 | 0.0718 (+0.4%) / 0.01772 (+0.2%) | 0.0718 (+0.4%) / 0.01780 (+0.7%) |
| Churn | 0.2319 / 0.0719 | 0.2320 (+0.1%) / 0.0723 (+0.6%) | 0.2331 (+0.5%) / 0.0730 (+1.6%) |
| Internet | 0.2089 / 0.0937 | 0.2231 (+6.8%) / 0.1017 (+8.6%) | 0.2253 (+7.9%) / 0.1012 (+8.0%) |
| Upselling | 0.1662 / 0.0490 | 0.1668 (+0.3%) / 0.0491 (+0.1%) | 0.1663 (+0.04%) / 0.0492 (+0.3%) |
| Kick | 0.2855 / 0.0949 | 0.2957 (+3.5%) / 0.0991 (+4.4%) | 0.2946 (+3.2%) / 0.0988 (+4.1%) |

In this section, we empirically show that our implementation of GBDT provides state-of-the-art quality and thus is an appropriate basis for building CatBoost by adding different improving options including the above-mentioned ones. For this purpose, we compare with baselines a raw setting of CatBoost which is as close to classical GBDT [[12](#bib.bib12)] as possible. Namely, we use CatBoost in GPU mode with the following parameters: – – boosting–type Plain  
– – border–count 255  
– – dev–bootstrap–type DiscreteUniform   
– – gradient–iterations 1  
– – random–strength 0  
– – depth 6.
Besides, we tune the parameters dev–sample–rate, learning–rate, l2–leaf–reg instead of the parameters described in paragraph “Parameter tuning” of Appendix [D.2](#A4.SS2 "D.2 Experimental settings ‣ Appendix D Experimental setup ‣ CatBoost: unbiased boosting with categorical features") by 50 steps of the optimization algorithm. Further, for all the algorithms, all categorical features are transformed to ordered TS on the basis of a random permutation (the same for all algorithms) of training examples at the preprocessing step. The resulting TS are used as numerical features in the training process. Thus, no CatBoost options dealing with categorical features are used. As a result, the main difference of the raw setting of CatBoost compared with XGBoost and LightGBM is using oblivious trees as base predictors.

Table 9: Comparison with baselines: logloss / zero-one loss (relative increase for baselines).

|  |  |  |  |
| --- | --- | --- | --- |
|  | Raw setting of CatBoost | LightGBM | XGBoost |
| Adult | 0.2800 / 0.1288 | -1.4% / +0.2% | -1.7% / -0.6% |
| Amazon | 0.1631 / 0.0533 | +0.3% / 0% | +0.1% / -0.2% |
| Click | 0.3961 / 0.1581 | +0.1% / -0.1% | 0% / 0% |
| Appetency | 0.0724 / 0.0179 | -0.8% / -1.0% | -0.8% / -0.4% |
| Churn | 0.2316 / 0.0718 | +0.2% / +0.7% | +0.6% / +1.6% |
| Internet | 0.2223 / 0.0993 | +0.4% / +2.4% | +1.4% / +1.9% |
| Upselling | 0.1679 / 0.0493 | -0.7% / -0.4% | -1.0% / -0.2% |
| Kick | 0.2955 / 0.0993 | +0.1% / -0.4% | -0.3% / -0.2% |
| Average |  | -0.2% / +0.2% | -0.2% / +0.2% |

For the baselines, we take the same results as in Table [8](#A7.T8 "Table 8 ‣ Comparison with baselines ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"). As we can see from Table [9](#A7.T9 "Table 9 ‣ Comparison with baselines ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"), in average, the difference between all the algorithms is rather small: the raw setting of CatBoost outperforms the baselines in terms of zero-one loss by 0.2% while they are better in terms of logloss by 0.2%. Thus, taking into account that a GBDT model with oblivious trees can significantly speed up execution at testing time [[23](#bib.bib23)], our implementation of GBDT is very reasonable choice to build CatBoost on.

##### Ordered and Plain modes

In Section [6](#S6 "6 Experiments ‣ CatBoost: unbiased boosting with categorical features") we showed experimentally that Ordered mode of CatBoost significantly outperforms Plain mode in the strong setting of CatBoost, including ordered TS and feature combinations. In this section, we verify that this advantage is not caused by interaction with these and other specific CatBoost options. For this purpose, we compare Ordered and Plain modes in the raw setting of CatBoost described in the previous paragraph.

In Table [10](#A7.T10 "Table 10 ‣ Ordered and Plain modes ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"), we present relative results w.r.t. Plain mode for two modifications of Ordered mode. The first one uses one random permutation σb​o​o​s​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{boost} for Ordered mode generated independently from the permutation σc​a​tsubscript𝜎𝑐𝑎𝑡\sigma\_{cat} used for ordered TS. Clearly, discrepancy between the two permutations provides target leakage, which should be avoided. However, even in this setting Ordered mode considerably outperforms Plain one by 0.5% in terms of logloss and by 0.2% in terms of zero-one loss in average. Thus, advantage of Ordered mode remains strong in the raw setting of CatBoost.

Table 10: Ordered vs Plain modes in raw setting: change of logloss / zero-one loss relative to Plain mode.

|  |  |  |
| --- | --- | --- |
|  | Ordered, σb​o​o​s​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{boost} independent of σc​a​tsubscript𝜎𝑐𝑎𝑡\sigma\_{cat} | Ordered, σb​o​o​s​t=σc​a​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡subscript𝜎𝑐𝑎𝑡\sigma\_{boost}=\sigma\_{cat} |
| Adult | -1.1% / +0.2% | -2.1% / -1.2% |
| Amazon | +0.9% / +0.9% | +0.8% / -2.2% |
| Click | 0% / 0% | 0.1% / 0% |
| Appetency | -0.2% / 0.2% | -0.5% / -0.3% |
| Churn | +0.2% / -0.1% | +0.3% / +0.4% |
| Internet | -3.5% / -3.2% | -2.8% / -3.5% |
| Upselling | -0.4% / +0.3% | -0.3% / -0.1% |
| Kick | -0.2% / -0.1% | -0.2% / -0.3% |
| Average | -0.5% / -0.2% | -0.6% / -0.9% |

In the second modification, we set σb​o​o​s​t=σc​a​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡subscript𝜎𝑐𝑎𝑡\sigma\_{boost}=\sigma\_{cat}, which remarkably improves both metrics: the relative difference with Plain becomes (in average) 0.6% for logloss and 0.9% for zero-one loss. This result empirically confirms the importance of the correspondence between permutations σb​o​o​s​tsubscript𝜎𝑏𝑜𝑜𝑠𝑡\sigma\_{boost} and σc​a​tsubscript𝜎𝑐𝑎𝑡\sigma\_{cat}, which was theoretically motivated in Appendix [F](#A6 "Appendix F Ordered boosting with categorical features ‣ CatBoost: unbiased boosting with categorical features").

##### Feature combinations

To demonstrate the effect of feature combinations, in Figure [3](#A7.F3 "Figure 3 ‣ Feature combinations ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features") we present the relative change in logloss for different numbers cm​a​xsubscript𝑐𝑚𝑎𝑥c\_{max} of features allowed to be combined (compared to cm​a​x=1subscript𝑐𝑚𝑎𝑥1c\_{max}=1, where combinations are absent). In average, changing cm​a​xsubscript𝑐𝑚𝑎𝑥c\_{max} from 1 to 2 provides an outstanding improvement of 1.86%percent1.861.86\% (reaching 11.3%percent11.311.3\%), changing from 1 to 3 yields 2.04%percent2.042.04\%, and further increase of cm​a​xsubscript𝑐𝑚𝑎𝑥c\_{max} does not influences the performance significantly.

!(/html/1706.09516/assets/x3.png)

Figure 3: Relative change in logloss for a given allowed complexity compared to the absence of feature combinations.

##### Number of permutations

The effect of the number s𝑠s of permutations on the performance of CatBoost is presented in Figure [4](#A7.F4 "Figure 4 ‣ Number of permutations ‣ Appendix G Experimental results ‣ CatBoost: unbiased boosting with categorical features"). In average, increasing s𝑠s slightly decreases logloss, e.g., by 0.19%percent0.190.19\% for s=3𝑠3s=3 and by 0.38%percent0.380.38\% for s=9𝑠9s=9 compared to s=1𝑠1s=1.

!(/html/1706.09516/assets/x4.png)

Figure 4: Relative change in logloss for a given number of permutations s𝑠s compared to s=1𝑠1s=1,
