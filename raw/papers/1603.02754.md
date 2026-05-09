---
arxiv: '1603.02754'
authors:
- Tianqi Chen
- Carlos Guestrin
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'XGBoost: A Scalable Tree Boosting System'
url: http://arxiv.org/abs/1603.02754v3
year: 2016
---

[1603.02754] XGBoost: A Scalable Tree Boosting System















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



\setcopyright

rightsretained
\isbn
\acmPrice

# XGBoost: A Scalable Tree Boosting System

Tianqi Chen
  
  
Carlos Guestrin
  
  
University of Washington
[tqchen@cs.washington.edu](mailto:tqchen@cs.washington.edu)
University of Washington
[guestrin@cs.washington.edu](mailto:guestrin@cs.washington.edu)

(2016; 29 Jan. 2016)

###### Abstract

Tree boosting is a highly effective and widely used machine learning method.
In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning.
More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system.
By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

###### keywords:

Large-scale Machine Learning

††conference: KDD ’16, August 13-17, 2016, San Francisco, CA, USA

{CCSXML}

<ccs2012>
<concept\_id>10010147.10010257</concept\_id>
<concept\_desc>Computing methodologies Machine learning</concept\_desc>
<concept\_significance>500</concept\_significance>
</concept>
<concept>
<concept>
<concept\_id>10002951.10003227.10003351</concept\_id>
<concept\_desc>Information systems Data mining</concept\_desc>
<concept\_significance>500</concept\_significance>
</concept>
<concept>
</ccs2012>

\printccsdesc

## 1 Introduction

Machine learning and data-driven approaches are becoming very important in many areas. Smart spam classifiers protect our email by learning from massive amounts of spam data and user feedback;
advertising systems learn to match the right ads with the right context;
fraud detection systems protect banks from malicious attackers; anomaly event detection systems help experimental physicists to find events that lead to new physics. There are two important factors that drive these successful applications: usage of effective (statistical) models that capture the complex data dependencies and scalable learning systems that learn the model of interest from large datasets.

Among the machine learning methods used in practice, gradient tree boosting [[10](#bib.bib10)]111Gradient tree boosting is also known as gradient boosting machine (GBM) or gradient boosted regression tree (GBRT) is one technique that shines in many applications.
Tree boosting has been shown to give state-of-the-art results on many standard classification benchmarks [[16](#bib.bib16)].
LambdaMART [[5](#bib.bib5)], a variant of tree boosting for ranking, achieves state-of-the-art result for ranking problems. Besides being used as a stand-alone predictor, it is also incorporated into real-world production pipelines for ad click through rate prediction [[15](#bib.bib15)]. Finally, it is the de-facto choice of ensemble method and is used in challenges such as the Netflix prize [[3](#bib.bib3)].

In this paper, we describe XGBoost, a scalable machine learning system for tree boosting. The system is available as an open source package222<https://github.com/dmlc/xgboost>.
The impact of the system has been widely recognized in a number of machine learning and data mining challenges.
Take the challenges hosted by the machine learning competition site Kaggle for example. Among the 29 challenge winning solutions 333Solutions come from of top-3 teams of each competitions. published at Kaggle’s blog during 2015, 17 solutions used XGBoost. Among these solutions, eight solely used XGBoost to train the model, while most others combined XGBoost with neural nets in ensembles.
For comparison, the second most popular method, deep neural nets, was used in 11 solutions.
The success of the system was also witnessed in KDDCup 2015, where XGBoost was used by every winning team in the top-10.
Moreover, the winning teams reported that ensemble methods outperform a well-configured XGBoost by only a small amount  [[1](#bib.bib1)].

These results demonstrate that our system gives state-of-the-art results on a wide range of problems. Examples of the problems in these winning solutions include: store sales prediction; high energy physics event classification; web text classification; customer behavior prediction; motion detection; ad click through rate prediction; malware classification; product categorization; hazard risk prediction; massive online course dropout rate prediction.
While domain dependent data analysis and feature engineering play an important role in these solutions, the fact that XGBoost is the consensus choice of learner shows the impact and importance of our system and tree boosting.

The most important factor behind the success of XGBoost is its scalability in all scenarios.
The system runs more than ten times faster than existing popular solutions on a single machine and scales to billions of examples in distributed or memory-limited settings.
The scalability of XGBoost is due to several important systems and algorithmic optimizations. These innovations include:
a novel tree learning algorithm is for handling *sparse data*; a theoretically justified weighted quantile sketch procedure enables handling instance weights in
approximate tree learning.
Parallel and distributed computing makes learning faster which enables quicker model exploration.
More importantly, XGBoost exploits out-of-core computation and enables data scientists to process hundred millions of examples on a desktop.
Finally, it is even more exciting to combine these techniques to make an end-to-end system that scales to even larger data with the least amount of cluster resources.
The major contributions of this paper is listed as follows:

* •

  We design and build a highly scalable end-to-end tree boosting system.
* •

  We propose a theoretically justified weighted quantile sketch for efficient proposal calculation.
* •

  We introduce a novel sparsity-aware algorithm for parallel tree learning.
* •

  We propose an effective cache-aware block structure for out-of-core tree learning.

While there are some existing works on parallel tree boosting [[22](#bib.bib22), [23](#bib.bib23), [19](#bib.bib19)],
the directions such as out-of-core computation, cache-aware and sparsity-aware learning have not been explored.
More importantly, an end-to-end system that combines all of these aspects gives a novel solution for real-world use-cases. This enables data scientists as well as researchers to build powerful variants of tree boosting algorithms [[7](#bib.bib7), [8](#bib.bib8)].
Besides these major contributions, we also make additional improvements in proposing a regularized learning objective, which we will include for completeness.

The remainder of the paper is organized as follows. We will first review tree boosting and introduce a regularized objective in Sec. [2](#S2 "2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System"). We then describe the split finding methods in Sec. [3](#S3 "3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System") as well as the system design in Sec. [4](#S4 "4 System Design ‣ XGBoost: A Scalable Tree Boosting System"), including experimental results when relevant to provide quantitative support for each optimization we describe. Related work is discussed in Sec. [5](#S5 "5 Related Works ‣ XGBoost: A Scalable Tree Boosting System").
Detailed end-to-end evaluations are included in Sec. [6](#S6 "6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
Finally we conclude the paper in Sec. [7](#S7 "7 Conclusion ‣ XGBoost: A Scalable Tree Boosting System").

## 2 Tree Boosting in a NutShell

We review gradient tree boosting algorithms in this section.
The derivation follows from the same idea in existing literatures in gradient boosting.
Specicially the second order method is originated from Friedman et al. [[12](#bib.bib12)].
We make minor improvements in the reguralized objective,
which were found helpful in practice.

### 2.1 Regularized Learning Objective

For a given data set with n𝑛n examples and m𝑚m features 𝒟={(𝐱i,yi)}𝒟subscript𝐱𝑖subscript𝑦𝑖\mathcal{D}=\{(\mathbf{x}\_{i},y\_{i})\} (|𝒟|=n,𝐱i∈ℝm,yi∈ℝformulae-sequence𝒟𝑛formulae-sequencesubscript𝐱𝑖superscriptℝ𝑚subscript𝑦𝑖ℝ|\mathcal{D}|=n,\mathbf{x}\_{i}\in\mathbb{R}^{m},y\_{i}\in\mathbb{R}), a tree ensemble model (shown in Fig. [1](#S2.F1 "Figure 1 ‣ 2.1 Regularized Learning Objective ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")) uses K𝐾K additive functions to predict the output.

|  |  |  |  |
| --- | --- | --- | --- |
|  | y^i=ϕ​(𝐱i)=∑k=1Kfk​(𝐱i),fk∈ℱ,formulae-sequencesubscript^𝑦𝑖italic-ϕsubscript𝐱𝑖subscriptsuperscript𝐾𝑘1subscript𝑓𝑘subscript𝐱𝑖subscript𝑓𝑘ℱ\hat{y}\_{i}=\phi(\mathbf{x}\_{i})=\sum^{K}\_{k=1}f\_{k}(\mathbf{x}\_{i}),\ \ f\_{k}\in\mathcal{F}, |  | (1) |

where ℱ={f(𝐱)=wq​(𝐱)}(q:ℝm→T,w∈ℝT)\mathcal{F}=\{f(\mathbf{x})=w\_{q(\mathbf{x})}\}(q:\mathbb{R}^{m}\rightarrow T,w\in\mathbb{R}^{T}) is the space of regression trees (also known as CART).
Here q𝑞q represents the structure of each tree that maps an example to the corresponding leaf index. T𝑇T is the number of leaves in the tree.
Each fksubscript𝑓𝑘f\_{k} corresponds to an independent tree structure q𝑞q and leaf weights w𝑤w.
Unlike decision trees, each regression tree contains a continuous score on each of the leaf, we use wisubscript𝑤𝑖w\_{i} to represent score on i𝑖i-th leaf.
For a given example, we will use the decision rules in the trees (given by q𝑞q) to classify it into the leaves and calculate the final prediction by summing up the score in the corresponding leaves (given by w𝑤w).
To learn the set of functions used in the model, we minimize the following *regularized* objective.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(ϕ)=∑il​(y^i,yi)+∑kΩ​(fk) where Ω​(f)=γ​T+12​λ​‖w‖2ℒitalic-ϕsubscript𝑖𝑙subscript^𝑦𝑖subscript𝑦𝑖subscript𝑘Ωsubscript𝑓𝑘 where Ω𝑓𝛾𝑇12𝜆superscriptdelimited-∥∥𝑤2\begin{split}\mathcal{L}(\phi)=&\sum\_{i}l(\hat{y}\_{i},y\_{i})+\sum\_{k}\Omega(f\_{k})\\ \mbox{ where }&\Omega(f)=\gamma T+\frac{1}{2}\lambda\|w\|^{2}\end{split} |  | (2) |

Here l𝑙l is a differentiable convex loss function that measures the difference between the prediction y^isubscript^𝑦𝑖\hat{y}\_{i} and the target yisubscript𝑦𝑖y\_{i}. The second term ΩΩ\Omega penalizes the complexity of the model (i.e., the regression tree functions).
The additional regularization term helps to smooth the final learnt weights to avoid over-fitting.
Intuitively, the regularized objective will tend to select a model employing simple and predictive functions.
A similar regularization technique has been used in Regularized greedy forest (RGF) [[25](#bib.bib25)] model.
Our objective and the corresponding learning algorithm is simpler than RGF and easier to parallelize.
When the regularization parameter is set to zero, the objective falls back to the traditional gradient tree boosting.

![Refer to caption](/html/1603.02754/assets/tree_model.png)


Figure 1: Tree Ensemble Model. The final prediction for a given example is the sum of predictions from each tree.

### 2.2 Gradient Tree Boosting

The tree ensemble model in Eq. ([2](#S2.E2 "In 2.1 Regularized Learning Objective ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")) includes functions as parameters and cannot be optimized using traditional optimization methods in Euclidean space.
Instead, the model is trained in an additive manner.
Formally, let y^i(t)superscriptsubscript^𝑦𝑖𝑡\hat{y}\_{i}^{(t)} be the prediction of the i𝑖i-th instance at the t𝑡t-th iteration, we will need to add ftsubscript𝑓𝑡f\_{t} to minimize the following objective.

|  |  |  |
| --- | --- | --- |
|  | ℒ(t)=∑i=1nl​(yi,yi^(t−1)+ft​(𝐱i))+Ω​(ft)superscriptℒ𝑡superscriptsubscript𝑖1𝑛𝑙subscript𝑦𝑖superscript^subscript𝑦𝑖𝑡1subscript𝑓𝑡subscript𝐱𝑖Ωsubscript𝑓𝑡\mathcal{L}^{(t)}=\sum\_{i=1}^{n}l(y\_{i},\hat{y\_{i}}^{(t-1)}+f\_{t}(\mathbf{x}\_{i}))+\Omega(f\_{t}) |  |

This means we greedily add the ftsubscript𝑓𝑡f\_{t} that most improves our model according to Eq. ([2](#S2.E2 "In 2.1 Regularized Learning Objective ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")).
Second-order approximation can be used to quickly optimize the objective in the general setting [[12](#bib.bib12)].

|  |  |  |
| --- | --- | --- |
|  | ℒ(t)≃∑i=1n[l​(yi,y^(t−1))+gi​ft​(𝐱i)+12​hi​ft2​(𝐱i)]+Ω​(ft)similar-to-or-equalssuperscriptℒ𝑡superscriptsubscript𝑖1𝑛delimited-[]𝑙subscript𝑦𝑖superscript^𝑦𝑡1subscript𝑔𝑖subscript𝑓𝑡subscript𝐱𝑖12subscriptℎ𝑖superscriptsubscript𝑓𝑡2subscript𝐱𝑖Ωsubscript𝑓𝑡\mathcal{L}^{(t)}\simeq\sum\_{i=1}^{n}[l(y\_{i},\hat{y}^{(t-1)})+g\_{i}f\_{t}(\mathbf{x}\_{i})+\frac{1}{2}h\_{i}f\_{t}^{2}(\mathbf{x}\_{i})]+\Omega(f\_{t}) |  |

where gi=∂y^(t−1)l​(yi,y^(t−1))subscript𝑔𝑖subscriptsuperscript^𝑦𝑡1𝑙subscript𝑦𝑖superscript^𝑦𝑡1g\_{i}=\partial\_{\hat{y}^{(t-1)}}l(y\_{i},\hat{y}^{(t-1)}) and hi=∂y^(t−1)2l​(yi,y^(t−1))subscriptℎ𝑖subscriptsuperscript2superscript^𝑦𝑡1𝑙subscript𝑦𝑖superscript^𝑦𝑡1h\_{i}=\partial^{2}\_{\hat{y}^{(t-1)}}l(y\_{i},\hat{y}^{(t-1)})
are first and second order gradient statistics on the loss function.
We can remove the constant terms to obtain the following simplified objective at step t𝑡t.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ~(t)=∑i=1n[gi​ft​(𝐱i)+12​hi​ft2​(𝐱i)]+Ω​(ft)superscript~ℒ𝑡superscriptsubscript𝑖1𝑛delimited-[]subscript𝑔𝑖subscript𝑓𝑡subscript𝐱𝑖12subscriptℎ𝑖superscriptsubscript𝑓𝑡2subscript𝐱𝑖Ωsubscript𝑓𝑡\tilde{\mathcal{L}}^{(t)}=\sum\_{i=1}^{n}[g\_{i}f\_{t}(\mathbf{x}\_{i})+\frac{1}{2}h\_{i}f\_{t}^{2}(\mathbf{x}\_{i})]+\Omega(f\_{t}) |  | (3) |

Define Ij={i|q​(𝐱i)=j}subscript𝐼𝑗conditional-set𝑖𝑞subscript𝐱𝑖𝑗I\_{j}=\{i|q(\mathbf{x}\_{i})=j\} as the instance set of leaf j𝑗j.
We can rewrite Eq ([3](#S2.E3 "In 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")) by expanding ΩΩ\Omega as follows

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ~(t)=∑i=1n[gi​ft​(𝐱i)+12​hi​ft2​(𝐱i)]+γ​T+12​λ​∑j=1Twj2=∑j=1T[(∑i∈Ijgi)​wj+12​(∑i∈Ijhi+λ)​wj2]+γ​Tsuperscript~ℒ𝑡subscriptsuperscript𝑛𝑖1delimited-[]subscript𝑔𝑖subscript𝑓𝑡subscript𝐱𝑖12subscriptℎ𝑖superscriptsubscript𝑓𝑡2subscript𝐱𝑖𝛾𝑇12𝜆subscriptsuperscript𝑇𝑗1superscriptsubscript𝑤𝑗2subscriptsuperscript𝑇𝑗1delimited-[]subscript𝑖subscript𝐼𝑗subscript𝑔𝑖subscript𝑤𝑗12subscript𝑖subscript𝐼𝑗subscriptℎ𝑖𝜆superscriptsubscript𝑤𝑗2𝛾𝑇\begin{split}\tilde{\mathcal{L}}^{(t)}&=\sum^{n}\_{i=1}[g\_{i}f\_{t}(\mathbf{x}\_{i})+\frac{1}{2}h\_{i}f\_{t}^{2}(\mathbf{x}\_{i})]+\gamma T+\frac{1}{2}\lambda\sum^{T}\_{j=1}w\_{j}^{2}\\ &=\sum^{T}\_{j=1}[(\sum\_{i\in I\_{j}}g\_{i})w\_{j}+\frac{1}{2}(\sum\_{i\in I\_{j}}h\_{i}+\lambda)w\_{j}^{2}]+\gamma T\end{split} |  | (4) |

For a fixed structure q​(𝐱)𝑞𝐱q(\mathbf{x}), we can compute the optimal weight wj∗superscriptsubscript𝑤𝑗w\_{j}^{\*} of leaf j𝑗j by

|  |  |  |  |
| --- | --- | --- | --- |
|  | wj∗=−∑i∈Ijgi∑i∈Ijhi+λ,subscriptsuperscript𝑤𝑗subscript𝑖subscript𝐼𝑗subscript𝑔𝑖subscript𝑖subscript𝐼𝑗subscriptℎ𝑖𝜆w^{\*}\_{j}=-\frac{\sum\_{i\in I\_{j}}g\_{i}}{\sum\_{i\in I\_{j}}h\_{i}+\lambda}, |  | (5) |

and calculate the corresponding optimal value by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ~(t)​(q)=−12​∑j=1T(∑i∈Ijgi)2∑i∈Ijhi+λ+γ​T.superscript~ℒ𝑡𝑞12subscriptsuperscript𝑇𝑗1superscriptsubscript𝑖subscript𝐼𝑗subscript𝑔𝑖2subscript𝑖subscript𝐼𝑗subscriptℎ𝑖𝜆𝛾𝑇\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2}\sum^{T}\_{j=1}\frac{(\sum\_{i\in I\_{j}}g\_{i})^{2}}{\sum\_{i\in I\_{j}}h\_{i}+\lambda}+\gamma T. |  | (6) |

![Refer to caption](/html/1603.02754/assets/struct_score.png)


Figure 2: Structure Score Calculation. We only need to sum up the gradient and second order gradient statistics on each leaf, then apply the scoring formula to get the quality score.

Eq ([6](#S2.E6 "In 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")) can be used as a scoring function to measure the quality of a tree structure q𝑞q. This score is like the impurity score for evaluating decision trees, except that it is derived for a wider range of objective functions.
Fig. [2](#S2.F2 "Figure 2 ‣ 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System") illustrates how this score can be calculated.

Normally it is impossible to enumerate all the possible tree structures q𝑞q.
A greedy algorithm that starts from a single leaf and iteratively adds branches to the tree is used instead.
Assume that ILsubscript𝐼𝐿I\_{L} and IRsubscript𝐼𝑅I\_{R} are the instance sets of left and right nodes after the split.
Lettting I=IL∪IR𝐼subscript𝐼𝐿subscript𝐼𝑅I=I\_{L}\cup I\_{R}, then the loss reduction after the split is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒs​p​l​i​t=12​[(∑i∈ILgi)2∑i∈ILhi+λ+(∑i∈IRgi)2∑i∈IRhi+λ−(∑i∈Igi)2∑i∈Ihi+λ]−γsubscriptℒ𝑠𝑝𝑙𝑖𝑡12delimited-[]superscriptsubscript𝑖subscript𝐼𝐿subscript𝑔𝑖2subscript𝑖subscript𝐼𝐿subscriptℎ𝑖𝜆superscriptsubscript𝑖subscript𝐼𝑅subscript𝑔𝑖2subscript𝑖subscript𝐼𝑅subscriptℎ𝑖𝜆superscriptsubscript𝑖𝐼subscript𝑔𝑖2subscript𝑖𝐼subscriptℎ𝑖𝜆𝛾\mathcal{L}\_{split}=\frac{1}{2}\left[\frac{(\sum\_{i\in I\_{L}}g\_{i})^{2}}{\sum\_{i\in I\_{L}}h\_{i}+\lambda}+\frac{(\sum\_{i\in I\_{R}}g\_{i})^{2}}{\sum\_{i\in I\_{R}}h\_{i}+\lambda}-\frac{(\sum\_{i\in I}g\_{i})^{2}}{\sum\_{i\in I}h\_{i}+\lambda}\right]-\gamma |  | (7) |

This formula is usually used in practice for evaluating the split candidates.

### 2.3 Shrinkage and Column Subsampling

Besides the regularized objective mentioned in Sec. [2.1](#S2.SS1 "2.1 Regularized Learning Objective ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System"), two additional techniques are used to further prevent over-fitting.
The first technique is shrinkage introduced by Friedman [[11](#bib.bib11)].
Shrinkage scales newly added weights by a factor η𝜂\eta after each step of tree boosting.
Similar to a learning rate in tochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model.
The second technique is column (feature) subsampling. This technique is used in RandomForest [[4](#bib.bib4), [13](#bib.bib13)],
It is implemented in a commercial software TreeNet 444https://www.salford-systems.com/products/treenet for gradient boosting,
but is not implemented in existing opensource packages.
According to user feedback, using column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling (which is also supported). The usage of column sub-samples also speeds up computations of the parallel algorithm described later.

## 3 Split Finding Algorithms

### 3.1 Basic Exact Greedy Algorithm

Input: I𝐼I, instance set of current node

Input: d𝑑d, feature dimension

g​a​i​n←0←𝑔𝑎𝑖𝑛0gain\leftarrow 0

G←∑i∈Igi←𝐺subscript𝑖𝐼subscript𝑔𝑖G\leftarrow\sum\_{i\in I}g\_{i}, H←∑i∈Ihi←𝐻subscript𝑖𝐼subscriptℎ𝑖H\leftarrow\sum\_{i\in I}h\_{i}

for *k=1𝑘1k=1 to m𝑚m*  do

GL←0,HL←0formulae-sequence←subscript𝐺𝐿0←subscript𝐻𝐿0G\_{L}\leftarrow 0,\ H\_{L}\leftarrow 0

for *j𝑗j in sorted(I𝐼I, by 𝐱j​ksubscript𝐱𝑗𝑘\mathbf{x}\_{jk})* do

GL←GL+gj,HL←HL+hjformulae-sequence←subscript𝐺𝐿subscript𝐺𝐿subscript𝑔𝑗←subscript𝐻𝐿subscript𝐻𝐿subscriptℎ𝑗G\_{L}\leftarrow G\_{L}+g\_{j},\ H\_{L}\leftarrow H\_{L}+h\_{j}

GR←G−GL,HR←H−HLformulae-sequence←subscript𝐺𝑅𝐺subscript𝐺𝐿←subscript𝐻𝑅𝐻subscript𝐻𝐿G\_{R}\leftarrow G-G\_{L},\ H\_{R}\leftarrow H-H\_{L}

s​c​o​r​e←max⁡(s​c​o​r​e,GL2HL+λ+GR2HR+λ−G2H+λ)←𝑠𝑐𝑜𝑟𝑒𝑠𝑐𝑜𝑟𝑒superscriptsubscript𝐺𝐿2subscript𝐻𝐿𝜆superscriptsubscript𝐺𝑅2subscript𝐻𝑅𝜆superscript𝐺2𝐻𝜆score\leftarrow\max(score,\frac{G\_{L}^{2}}{H\_{L}+\lambda}+\frac{G\_{R}^{2}}{H\_{R}+\lambda}-\frac{G^{2}}{H+\lambda})

end for

end for

Output: Split with max score

Algorithm 1 Exact Greedy Algorithm for Split Finding

One of the key problems in tree learning is to find the best split as indicated by Eq ([7](#S2.E7 "In 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")).
In order to do so, a split finding algorithm enumerates over all the possible splits on all the features. We call this the *exact greedy algorithm*.
Most existing single machine tree boosting implementations, such as scikit-learn [[20](#bib.bib20)], R’s gbm [[21](#bib.bib21)] as well as the single machine version of XGBoost support the exact greedy algorithm.
The exact greedy algorithm is shown in Alg. [1](#alg1 "In 3.1 Basic Exact Greedy Algorithm ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System").
It is computationally demanding to enumerate all the possible splits for continuous features.
In order to do so efficiently, the algorithm must first sort the data according to feature values and visit the data in sorted order to accumulate the gradient statistics
for the structure score in Eq ([7](#S2.E7 "In 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")).

### 3.2 Approximate Algorithm

for *k=1𝑘1k=1 to m𝑚m*  do

Propose Sk={sk​1,sk​2,⋯​sk​l}subscript𝑆𝑘subscript𝑠𝑘1subscript𝑠𝑘2⋯subscript𝑠𝑘𝑙S\_{k}=\{s\_{k1},s\_{k2},\cdots s\_{kl}\} by percentiles on feature k𝑘k.

Proposal can be done per tree (global), or per split(local).

end for

for *k=1𝑘1k=1 to m𝑚m*  do

Gk​v←=∑j∈{j|sk,v≥𝐱j​k>sk,v−1}gjG\_{kv}\leftarrow=\sum\_{j\in\{j|s\_{k,v}\geq\mathbf{x}\_{jk}>s\_{k,v-1}\}}g\_{j}
Hk​v←=∑j∈{j|sk,v≥𝐱j​k>sk,v−1}hjH\_{kv}\leftarrow=\sum\_{j\in\{j|s\_{k,v}\geq\mathbf{x}\_{jk}>s\_{k,v-1}\}}h\_{j}

end for

Follow same step as in previous section to find max score only among proposed splits.

Algorithm 2 Approximate Algorithm for Split Finding

The exact greedy algorithm is very powerful since it enumerates over all possible splitting points greedily.
However, it is impossible to efficiently do so when the data does not fit entirely into memory.
Same problem also arises in the distributed setting.
To support effective gradient tree boosting in these two settings, an approximate algorithm is needed.

We summarize an approximate framework, which resembles the ideas proposed in past literatures [[17](#bib.bib17), [2](#bib.bib2), [22](#bib.bib22)],
in Alg. [2](#alg2 "In 3.2 Approximate Algorithm ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System").
To summarize, the algorithm first proposes candidate splitting points according to percentiles of feature distribution (a specific criteria will be given in Sec. [3.3](#S3.SS3 "3.3 Weighted Quantile Sketch ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System")).
The algorithm then maps the continuous features into buckets split by these candidate points, aggregates the statistics and finds the best solution among proposals based on the aggregated statistics.

![Refer to caption](/html/1603.02754/assets/x1.png)


Figure 3: Comparison of test AUC convergence on Higgs 10M dataset.
The eps parameter corresponds to the accuracy of the approximate sketch. This roughly translates to 1 / eps buckets in the proposal.
We find that local proposals require fewer buckets, because it refine split candidates.

There are two variants of the algorithm, depending on when the proposal is given.
The global variant proposes all the candidate splits during the initial phase of tree construction,
and uses the same proposals for split finding at all levels. The local variant re-proposes after each split.
The global method requires less proposal steps than the local method. However, usually more candidate points are needed for the global proposal
because candidates are not refined after each split.
The local proposal refines the candidates after splits, and can potentially be more appropriate for deeper trees.
A comparison of different algorithms on a Higgs boson dataset is given by Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Approximate Algorithm ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System").
We find that the local proposal indeed requires fewer candidates. The global proposal can be as accurate as the local one given enough candidates.

Most existing approximate algorithms for distributed tree learning also follow this framework.
Notably, it is also possible to directly construct approximate histograms of gradient statistics [[22](#bib.bib22)].
It is also possible to use other variants of binning strategies instead of quantile [[17](#bib.bib17)].
Quantile strategy benefit from being distributable and recomputable, which we will detail in next subsection.
From Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Approximate Algorithm ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System"), we also find that the quantile strategy can get the same accuracy as exact greedy given reasonable approximation level.

Our system efficiently supports exact greedy for the single machine setting, as well as approximate algorithm with both local and global proposal methods for all settings.
Users can freely choose between the methods according to their needs.

### 3.3 Weighted Quantile Sketch

One important step in the approximate algorithm is to propose candidate split points. Usually percentiles of a feature are used to make candidates distribute evenly on the data.
Formally, let multi-set
𝒟k={(x1​k,h1),(x2​k,h2)​⋯​(xn​k,hn)}subscript𝒟𝑘subscript𝑥1𝑘subscriptℎ1subscript𝑥2𝑘subscriptℎ2⋯subscript𝑥𝑛𝑘subscriptℎ𝑛\mathcal{D}\_{k}=\{(x\_{1k},h\_{1}),(x\_{2k},h\_{2})\cdots(x\_{nk},h\_{n})\}
represent the k𝑘k-th feature values and second order gradient statistics of each training instances.
We can define a rank functions rk:ℝ→[0,+∞):subscript𝑟𝑘→ℝ0r\_{k}:\mathbb{R}\rightarrow[0,+\infty) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | rk​(z)=1∑(x,h)∈𝒟kh​∑(x,h)∈𝒟k,x<zh,subscript𝑟𝑘𝑧1subscript𝑥ℎsubscript𝒟𝑘ℎsubscriptformulae-sequence𝑥ℎsubscript𝒟𝑘𝑥𝑧ℎr\_{k}(z)=\frac{1}{\sum\_{(x,h)\in\mathcal{D}\_{k}}h}\sum\_{(x,h)\in\mathcal{D}\_{k},x<z}h, |  | (8) |

which represents the proportion of instances whose feature value k𝑘k is smaller than z𝑧z.
The goal is to find candidate split points {sk​1,sk​2,⋯​sk​l}subscript𝑠𝑘1subscript𝑠𝑘2⋯subscript𝑠𝑘𝑙\{s\_{k1},s\_{k2},\cdots s\_{kl}\}, such that

|  |  |  |  |
| --- | --- | --- | --- |
|  | |rk​(sk,j)−rk​(sk,j+1)|<ϵ,sk​1=mini⁡𝐱i​k,sk​l=maxi⁡𝐱i​k.formulae-sequencesubscript𝑟𝑘subscript𝑠  𝑘𝑗subscript𝑟𝑘subscript𝑠  𝑘𝑗1italic-ϵformulae-sequencesubscript𝑠𝑘1subscript𝑖subscript𝐱𝑖𝑘subscript𝑠𝑘𝑙subscript𝑖subscript𝐱𝑖𝑘|r\_{k}(s\_{k,j})-r\_{k}(s\_{k,j+1})|<\epsilon,\ \ s\_{k1}=\min\_{i}\mathbf{x}\_{ik},s\_{kl}=\max\_{i}\mathbf{x}\_{ik}. |  | (9) |

Here ϵitalic-ϵ\epsilon is an approximation factor.
Intuitively, this means that there is roughly 1/ϵ1italic-ϵ1/\epsilon candidate points.
Here each data point is weighted by hisubscriptℎ𝑖h\_{i}.
To see why hisubscriptℎ𝑖h\_{i} represents the weight, we can rewrite Eq ([3](#S2.E3 "In 2.2 Gradient Tree Boosting ‣ 2 Tree Boosting in a NutShell ‣ XGBoost: A Scalable Tree Boosting System")) as

|  |  |  |
| --- | --- | --- |
|  | ∑i=1n12​hi​(ft​(𝐱i)−gi/hi)2+Ω​(ft)+c​o​n​s​t​a​n​t,superscriptsubscript𝑖1𝑛12subscriptℎ𝑖superscriptsubscript𝑓𝑡subscript𝐱𝑖subscript𝑔𝑖subscriptℎ𝑖2Ωsubscript𝑓𝑡𝑐𝑜𝑛𝑠𝑡𝑎𝑛𝑡\sum\_{i=1}^{n}\frac{1}{2}h\_{i}(f\_{t}(\mathbf{x}\_{i})-g\_{i}/h\_{i})^{2}+\Omega(f\_{t})+constant, |  |

which is exactly weighted squared loss with labels gi/hisubscript𝑔𝑖subscriptℎ𝑖g\_{i}/h\_{i} and weights hisubscriptℎ𝑖h\_{i}.
For large datasets, it is non-trivial to find candidate splits that satisfy the criteria.
When every instance has equal weights, an existing algorithm called quantile sketch [[14](#bib.bib14), [24](#bib.bib24)]
solves the problem. However, there is no existing quantile sketch for the weighted datasets.
Therefore, most existing approximate algorithms either resorted to sorting on a random subset of data which have a chance of failure or heuristics that do not have theoretical guarantee.

To solve this problem, we introduced a novel distributed weighted quantile sketch algorithm that can handle weighted data with a
*provable theoretical guarantee*. The general idea is to propose a data structure that supports *merge* and *prune*
operations, with each operation proven to maintain a certain accuracy level.
A detailed description of the algorithm as well as proofs are given in the appendix.

### 3.4 Sparsity-aware Split Finding

Input: I𝐼I, instance set of current node

Input: Ik={i∈I|xi​k≠missing}subscript𝐼𝑘conditional-set𝑖𝐼subscript𝑥𝑖𝑘missingI\_{k}=\{i\in I|x\_{ik}\neq\mbox{missing}\}

Input: d𝑑d, feature dimension

*Also applies to the approximate setting, only collect statistics of non-missing entries into buckets*

g​a​i​n←0←𝑔𝑎𝑖𝑛0gain\leftarrow 0

G←∑i∈I,gi←𝐺

subscript𝑖𝐼subscript𝑔𝑖G\leftarrow\sum\_{i\in I},g\_{i},H←∑i∈Ihi←𝐻subscript𝑖𝐼subscriptℎ𝑖H\leftarrow\sum\_{i\in I}h\_{i}

for *k=1𝑘1k=1 to m𝑚m* do

*// enumerate missing value goto right*

GL←0,HL←0formulae-sequence←subscript𝐺𝐿0←subscript𝐻𝐿0G\_{L}\leftarrow 0,\ H\_{L}\leftarrow 0

for *j𝑗j in sorted(Iksubscript𝐼𝑘I\_{k}, ascent order by 𝐱j​ksubscript𝐱𝑗𝑘\mathbf{x}\_{jk})* do

GL←GL+gj,HL←HL+hjformulae-sequence←subscript𝐺𝐿subscript𝐺𝐿subscript𝑔𝑗←subscript𝐻𝐿subscript𝐻𝐿subscriptℎ𝑗G\_{L}\leftarrow G\_{L}+g\_{j},\ H\_{L}\leftarrow H\_{L}+h\_{j}

GR←G−GL,HR←H−HLformulae-sequence←subscript𝐺𝑅𝐺subscript𝐺𝐿←subscript𝐻𝑅𝐻subscript𝐻𝐿G\_{R}\leftarrow G-G\_{L},\ H\_{R}\leftarrow H-H\_{L}

s​c​o​r​e←max⁡(s​c​o​r​e,GL2HL+λ+GR2HR+λ−G2H+λ)←𝑠𝑐𝑜𝑟𝑒𝑠𝑐𝑜𝑟𝑒superscriptsubscript𝐺𝐿2subscript𝐻𝐿𝜆superscriptsubscript𝐺𝑅2subscript𝐻𝑅𝜆superscript𝐺2𝐻𝜆score\leftarrow\max(score,\frac{G\_{L}^{2}}{H\_{L}+\lambda}+\frac{G\_{R}^{2}}{H\_{R}+\lambda}-\frac{G^{2}}{H+\lambda})

end for

*// enumerate missing value goto left*

GR←0,HR←0formulae-sequence←subscript𝐺𝑅0←subscript𝐻𝑅0G\_{R}\leftarrow 0,\ H\_{R}\leftarrow 0

for *j𝑗j in sorted(Iksubscript𝐼𝑘I\_{k}, descent order by 𝐱j​ksubscript𝐱𝑗𝑘\mathbf{x}\_{jk})* do

GR←GR+gj,HR←HR+hjformulae-sequence←subscript𝐺𝑅subscript𝐺𝑅subscript𝑔𝑗←subscript𝐻𝑅subscript𝐻𝑅subscriptℎ𝑗G\_{R}\leftarrow G\_{R}+g\_{j},\ H\_{R}\leftarrow H\_{R}+h\_{j}

GL←G−GR,HL←H−HRformulae-sequence←subscript𝐺𝐿𝐺subscript𝐺𝑅←subscript𝐻𝐿𝐻subscript𝐻𝑅G\_{L}\leftarrow G-G\_{R},\ H\_{L}\leftarrow H-H\_{R}

s​c​o​r​e←max⁡(s​c​o​r​e,GL2HL+λ+GR2HR+λ−G2H+λ)←𝑠𝑐𝑜𝑟𝑒𝑠𝑐𝑜𝑟𝑒superscriptsubscript𝐺𝐿2subscript𝐻𝐿𝜆superscriptsubscript𝐺𝑅2subscript𝐻𝑅𝜆superscript𝐺2𝐻𝜆score\leftarrow\max(score,\frac{G\_{L}^{2}}{H\_{L}+\lambda}+\frac{G\_{R}^{2}}{H\_{R}+\lambda}-\frac{G^{2}}{H+\lambda})

end for

end for

Output: Split and default directions with max gain

Algorithm 3 Sparsity-aware Split Finding

![Refer to caption](/html/1603.02754/assets/tree_default.png)


Figure 4: Tree structure with default directions.
An example will be classified into the default direction when the feature needed for the split is missing.

![Refer to caption](/html/1603.02754/assets/x2.png)


Figure 5: Impact of the sparsity aware algorithm on Allstate-10K.
The dataset is sparse mainly due to one-hot encoding. The sparsity aware algorithm is more than 50 times faster than the naive version that does not take sparsity into consideration.

![Refer to caption](/html/1603.02754/assets/data_layout.png)


Figure 6: Block structure for parallel learning. Each column in a block is sorted by the corresponding feature value.
A linear scan over one column in the block is sufficient to enumerate all the split points.

In many real-world problems, it is quite common for the input 𝐱𝐱\mathbf{x} to be sparse.
There are multiple possible causes for sparsity: 1) presence of missing values in the data;
2) frequent zero entries in the statistics; and, 3) artifacts of feature engineering such as one-hot encoding.
It is important to make the algorithm aware of the sparsity pattern in the data.
In order to do so, we propose to add a default direction in each tree node, which is shown in Fig. [4](#S3.F4 "Figure 4 ‣ 3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System").
When a value is missing in the sparse matrix 𝐱𝐱\mathbf{x}, the instance is classified into the default direction.
There are two choices of default direction in each branch. The optimal default directions are learnt from the data.
The algorithm is shown in Alg. [3](#alg3 "In 3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System"). The key improvement is to only visit the non-missing entries Iksubscript𝐼𝑘I\_{k}. The presented algorithm treats the non-presence as a missing value and learns the best direction to handle missing values.
The same algorithm can also be applied when the non-presence corresponds to a user specified value by limiting the enumeration only to consistent solutions.

To the best of our knowledge, most existing tree learning algorithms are either only optimized for dense data, or need specific procedures to handle limited cases such as categorical encoding.
XGBoost handles all sparsity patterns in a unified way.
More importantly, our method exploits the sparsity to make computation complexity linear to number of non-missing entries in the input.
Fig. [5](#S3.F5 "Figure 5 ‣ 3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System") shows the comparison of sparsity aware and a naive implementation on an Allstate-10K dataset (description of dataset given in Sec. [6](#S6 "6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System")). We find that the sparsity aware algorithm runs 50 times faster than the naive version.
This confirms the importance of the sparsity aware algorithm.

![Refer to caption](/html/1603.02754/assets/x3.png)


(a) Allstate 10M

![Refer to caption](/html/1603.02754/assets/x4.png)


(b) Higgs 10M

![Refer to caption](/html/1603.02754/assets/x5.png)


(c) Allstate 1M

![Refer to caption](/html/1603.02754/assets/x6.png)


(d) Higgs 1M

Figure 7: Impact of cache-aware prefetching in exact greedy algorithm.
We find that the cache-miss effect impacts the performance on the large datasets (10 million instances).
Using cache aware prefetching improves the performance by factor of two when the dataset is large.

## 4 System Design

### 4.1 Column Block for Parallel Learning

The most time consuming part of tree learning is to get the data into sorted order.
In order to reduce the cost of sorting, we propose to store the data in in-memory units, which we called *block*.
Data in each block is stored in the compressed column (CSC) format, with each column sorted by the corresponding feature value.
This input data layout only needs to be computed once before training, and can be reused in later iterations.

In the exact greedy algorithm, we store the entire dataset in a single block and run the split search algorithm by linearly scanning over the pre-sorted entries. We do the split finding of all leaves collectively, so one scan over the block will collect the statistics of the split candidates in all leaf branches. Fig. [6](#S3.F6 "Figure 6 ‣ 3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System") shows how we transform a dataset into the format and find the optimal split using the block structure.

The block structure also helps when using the approximate algorithms.
Multiple blocks can be used in this case, with each block corresponding to subset of rows in the dataset.
Different blocks can be distributed across machines, or stored on disk in the out-of-core setting.
Using the sorted structure, the quantile finding step becomes a *linear scan* over the sorted columns.
This is especially valuable for local proposal algorithms, where candidates are generated frequently at each branch.
The binary search in histogram aggregation also becomes a linear time merge style algorithm.

Collecting statistics for each column can be *parallelized*, giving us a parallel algorithm for split finding.
Importantly, the column block structure also supports column subsampling, as it is easy to select a subset of columns in a block.

Time Complexity Analysis
Let d𝑑d be the maximum depth of the tree and K𝐾K be total number of trees.
For the exact greedy algorithm, the time complexity of original spase aware algorithm is O​(K​d​‖𝐱‖0​log⁡n)𝑂𝐾𝑑subscriptnorm𝐱0𝑛O(Kd\|\mathbf{x}\|\_{0}\log n).
Here we use ‖𝐱‖0subscriptnorm𝐱0\|\mathbf{x}\|\_{0} to denote number of non-missing entries in the training data.
On the other hand, tree boosting on the block structure only cost O​(K​d​‖𝐱‖0+‖𝐱‖0​log⁡n)𝑂𝐾𝑑subscriptnorm𝐱0subscriptnorm𝐱0𝑛O(Kd\|\mathbf{x}\|\_{0}+\|\mathbf{x}\|\_{0}\log n).
Here O​(‖𝐱‖0​log⁡n)𝑂subscriptnorm𝐱0𝑛O(\|\mathbf{x}\|\_{0}\log n) is the one time preprocessing cost that can be amortized.
This analysis shows that the block structure helps to save an additional log⁡n𝑛\log n factor, which is significant when n𝑛n is large.
For the approximate algorithm, the time complexity of original algorithm with binary search is O​(K​d​‖𝐱‖0​log⁡q)𝑂𝐾𝑑subscriptnorm𝐱0𝑞O(Kd\|\mathbf{x}\|\_{0}\log q).
Here q𝑞q is the number of proposal candidates in the dataset. While q𝑞q is usually between 32 and 100, the log factor still introduces overhead.
Using the block structure, we can reduce the time to O​(K​d​‖𝐱‖0+‖𝐱‖0​log⁡B)𝑂𝐾𝑑subscriptnorm𝐱0subscriptnorm𝐱0𝐵O(Kd\|\mathbf{x}\|\_{0}+\|\mathbf{x}\|\_{0}\log B), where B𝐵B is the maximum number of rows in each block.
Again we can save the additional log⁡q𝑞\log q factor in computation.

### 4.2 Cache-aware Access

![Refer to caption](/html/1603.02754/assets/cache-miss.png)


Figure 8: 
Short range data dependency pattern that can cause stall due to cache miss.



![Refer to caption](/html/1603.02754/assets/x7.png)


(a) Allstate 10M

![Refer to caption](/html/1603.02754/assets/x8.png)


(b) Higgs 10M

Figure 9: 
The impact of block size in the approximate algorithm.
We find that overly small blocks results in inefficient parallelization, while overly large blocks also slows down training due to cache misses.

While the proposed block structure helps optimize the computation complexity of split finding, the new algorithm requires indirect fetches of gradient statistics by row index, since these values are accessed in order of feature. This is a non-continuous memory access.
A naive implementation of split enumeration introduces immediate read/write dependency between the accumulation and the non-continuous memory fetch operation (see Fig. [8](#S4.F8 "Figure 8 ‣ 4.2 Cache-aware Access ‣ 4 System Design ‣ XGBoost: A Scalable Tree Boosting System")).
This slows down split finding when the gradient statistics do not fit into CPU cache and cache miss occur.

Table 1: Comparison of major tree boosting systems.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| System | exact greedy | approximate global | approximate local | out-of-core | sparsity aware | parallel |
| XGBoost | yes | yes | yes | yes | yes | yes |
| pGBRT | no | no | yes | no | no | yes |
| Spark MLLib | no | yes | no | no | partially | yes |
| H2O | no | yes | no | no | partially | yes |
| scikit-learn | yes | no | no | no | no | no |
| R GBM | yes | no | no | no | partially | no |

For the exact greedy algorithm, we can alleviate the problem by a cache-aware prefetching algorithm.
Specifically, we allocate an internal buffer in each thread, fetch the gradient statistics into it, and then perform accumulation in a mini-batch manner.
This prefetching changes the direct read/write dependency to a longer dependency and helps to reduce the runtime overhead when number of rows in the is large.
Figure [7](#S3.F7 "Figure 7 ‣ 3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System") gives the comparison of cache-aware vs. non cache-aware algorithm on the the Higgs and the Allstate dataset.
We find that cache-aware implementation of the exact greedy algorithm runs twice as fast as the naive version when the dataset is large.

For approximate algorithms, we solve the problem by choosing a correct block size. We define the block size to be maximum number of examples in contained in a block, as this reflects the cache storage cost of gradient statistics.
Choosing an overly small block size results in small workload for each thread and leads to inefficient parallelization. On the other hand, overly large blocks result in cache misses, as the gradient statistics do not fit into the CPU cache.
A good choice of block size balances these two factors.
We compared various choices of block size on two data sets.
The results are given in Fig. [9](#S4.F9 "Figure 9 ‣ 4.2 Cache-aware Access ‣ 4 System Design ‣ XGBoost: A Scalable Tree Boosting System").
This result validates our discussion and shows that choosing 216superscript2162^{16} examples per block balances the cache property and parallelization.

### 4.3 Blocks for Out-of-core Computation

One goal of our system is to fully utilize a machine’s resources to achieve scalable learning.
Besides processors and memory, it is important to utilize disk space to handle data that does not fit into main memory. To enable out-of-core computation, we divide the data into multiple blocks and store each block on disk.
During computation, it is important to use an independent thread to pre-fetch the block into a main memory buffer, so computation can happen in concurrence with disk reading.
However, this does not entirely solve the problem since the disk reading takes most of the computation time.
It is important to reduce the overhead and increase the throughput of disk IO.
We mainly use two techniques to improve the out-of-core computation.

Block Compression
The first technique we use is block compression.
The block is compressed by columns, and decompressed on the fly by an independent thread when loading into main memory.
This helps to trade some of the computation in decompression with the disk reading cost.
We use a general purpose compression algorithm for compressing the features values.
For the row index, we substract the row index by the begining index of the block and use a 16bit integer to store each offset.
This requires 216superscript2162^{16} examples per block, which is confirmed to be a good setting.
In most of the dataset we tested, we achieve roughly a 26% to 29% compression ratio.

Block Sharding
The second technique is to shard the data onto multiple disks in an alternative manner.
A pre-fetcher thread is assigned to each disk and fetches the data into an in-memory buffer. The training thread then alternatively reads the data from each buffer. This helps to increase the throughput of disk reading when multiple disks are available.

## 5 Related Works

Our system implements gradient boosting [[10](#bib.bib10)], which performs additive optimization in functional space.
Gradient tree boosting has been successfully used in classification [[12](#bib.bib12)], learning to rank [[5](#bib.bib5)],
structured prediction [[8](#bib.bib8)] as well as other fields. XGBoost incorporates a regularized model to prevent overfitting. This this resembles previous work on regularized greedy forest [[25](#bib.bib25)], but simplifies the objective and algorithm for parallelization.
Column sampling is a simple but effective technique borrowed from RandomForest [[4](#bib.bib4)].
While sparsity-aware learning is essential in other types of models such as linear models [[9](#bib.bib9)], few works on tree learning have considered this topic in a principled way. The algorithm proposed in this paper is the first unified approach to handle all kinds of sparsity patterns.

There are several existing works on parallelizing tree learning [[22](#bib.bib22), [19](#bib.bib19)].
Most of these algorithms fall into the approximate framework described in this paper.
Notably, it is also possible to partition data by columns [[23](#bib.bib23)] and apply the exact greedy algorithm. This is also supported in our framework, and the techniques such as cache-aware pre-fecthing can be used to benefit this type of algorithm.
While most existing works focus on the algorithmic aspect of parallelization,
our work improves in two unexplored system directions: out-of-core computation and cache-aware learning.
This gives us insights on how the system and the algorithm can be jointly optimized and provides an end-to-end system that can handle large scale problems
with very limited computing resources.
We also summarize the comparison between our system and existing opensource implementations in Table [1](#S4.T1 "Table 1 ‣ 4.2 Cache-aware Access ‣ 4 System Design ‣ XGBoost: A Scalable Tree Boosting System").

Quantile summary (without weights) is a classical problem in the database community [[14](#bib.bib14), [24](#bib.bib24)].
However, the approximate tree boosting algorithm reveals a more general problem – finding quantiles on weighted data.
To the best of our knowledge, the weighted quantile sketch proposed in this paper is the first method to solve this problem.
The weighted quantile summary is also not specific to the tree learning and can benefit other applications in data science and machine learning in the future.

## 6 End to End Evaluations

### 6.1 System Implementation

We implemented XGBoost as an open source package555<https://github.com/dmlc/xgboost>.
The package is portable and reusable.
It supports various weighted classification and rank objective functions, as well as user defined objective function.
It is available in popular languages such as python, R, Julia and integrates naturally with language native data science pipelines such as scikit-learn.
The distributed version is built on top of the rabit library666https://github.com/dmlc/rabit for allreduce.
The portability of XGBoost makes it available in many ecosystems, instead of only being tied to a specific platform.
The distributed XGBoost runs natively on Hadoop, MPI Sun Grid engine.
Recently, we also enable distributed XGBoost on jvm bigdata stacks such as Flink and Spark.
The distributed version has also been integrated into cloud platform
Tianchi777https://tianchi.aliyun.com of Alibaba.
We believe that there will be more integrations in the future.

### 6.2 Dataset and Setup

Table 2: Dataset used in the Experiments.

| Dataset | n𝑛n | m𝑚m | Task |
| --- | --- | --- | --- |
| Allstate | 10 M | 4227 | Insurance claim classification |
| Higgs Boson | 10 M | 28 | Event classification |
| Yahoo LTRC | 473K | 700 | Learning to Rank |
| Criteo | 1.7 B | 67 | Click through rate prediction |

We used four datasets in our experiments. A summary of these datasets is given in Table [2](#S6.T2 "Table 2 ‣ 6.2 Dataset and Setup ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
In some of the experiments, we use a randomly selected subset of the data either due to slow baselines or to demonstrate the performance of the algorithm with varying dataset size.
We use a suffix to denote the size in these cases. For example
Allstate-10K means a subset of the Allstate dataset with 10K instances.

The first dataset we use is the Allstate insurance claim dataset888https://www.kaggle.com/c/ClaimPredictionChallenge.
The task is to predict the likelihood and cost of an insurance claim given different risk factors.
In the experiment, we simplified the task to only predict the likelihood of an insurance claim.
This dataset is used to evaluate the impact of sparsity-aware algorithm in Sec. [3.4](#S3.SS4 "3.4 Sparsity-aware Split Finding ‣ 3 Split Finding Algorithms ‣ XGBoost: A Scalable Tree Boosting System").
Most of the sparse features in this data come from one-hot encoding.
We randomly select 10M instances as training set and use the rest as evaluation set.

The second dataset is the Higgs boson dataset999https://archive.ics.uci.edu/ml/datasets/HIGGS from high energy physics.
The data was produced using Monte Carlo simulations of physics events.
It contains 21 kinematic properties measured by the particle detectors in the accelerator.
It also contains seven additional derived physics quantities of the particles.
The task is to classify whether an event corresponds to the Higgs boson.
We randomly select 10M instances as training set and use the rest as evaluation set.

The third dataset is the Yahoo! learning to rank challenge dataset [[6](#bib.bib6)], which is one of the most commonly used benchmarks in learning to rank algorithms.
The dataset contains 20K web search queries, with each query corresponding to a list of around 22 documents.
The task is to rank the documents according to relevance of the query.
We use the official train test split in our experiment.

The last dataset is the criteo terabyte click log dataset101010http://labs.criteo.com/downloads/download-terabyte-click-logs/.
We use this dataset to evaluate the scaling property of the system in the out-of-core and the distributed settings.
The data contains 13 integer features and 26 ID features of user, item and advertiser information.
Since a tree based model is better at handling continuous features,
we preprocess the data by calculating the statistics of average CTR and count of ID features on the first ten days, replacing the ID features by the corresponding count statistics during the next ten days for training.
The training set after preprocessing contains 1.7 billion instances with 67 features (13 integer, 26 average CTR statistics and 26 counts).
The entire dataset is more than one terabyte in LibSVM format.

We use the first three datasets for the single machine parallel setting, and the last dataset for the distributed and out-of-core settings.
All the single machine experiments are conducted on a Dell PowerEdge R420 with two eight-core Intel Xeon (E5-2470) (2.3GHz) and 64GB of memory.
If not specified, all the experiments are run using all the available cores in the machine.
The machine settings of the distributed and the out-of-core experiments will be described in the corresponding section.
In all the experiments, we boost trees with a common setting of maximum depth equals 8, shrinkage equals 0.1
and no column subsampling unless explicitly specified.
We can find similar results when we use other settings of maximum depth.

### 6.3 Classification

Table 3: Comparison of Exact Greedy Methods with 500 trees on Higgs-1M data.

| Method | Time per Tree (sec) | Test AUC |
| --- | --- | --- |
| XGBoost | 0.6841 | 0.8304 |
| XGBoost (colsample=0.5) | 0.6401 | 0.8245 |
| scikit-learn | 28.51 | 0.8302 |
| R.gbm | 1.032 | 0.6224 |

In this section, we evaluate the performance of XGBoost on a single machine using the exact greedy algorithm on Higgs-1M data, by comparing it against two other commonly used exact greedy tree boosting implementations.
Since scikit-learn only handles non-sparse input, we choose the dense Higgs dataset for a fair comparison.
We use the 1M subset to make scikit-learn finish running in reasonable time.
Among the methods in comparison, R’s GBM uses a greedy approach that only expands one branch of a tree, which makes
it faster but can result in lower accuracy, while both scikit-learn and XGBoost learn a full tree.
The results are shown in Table [3](#S6.T3 "Table 3 ‣ 6.3 Classification ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
Both XGBoost and scikit-learn give better performance than R’s GBM, while XGBoost runs more than 10x faster than scikit-learn.
In this experiment, we also find column subsamples gives slightly worse performance than using all the features. This could due to the fact that there are few important features in this dataset and we can benefit from greedily select from all the features.

### 6.4 Learning to Rank

![Refer to caption](/html/1603.02754/assets/x9.png)


Figure 10: Comparison between XGBoost and pGBRT on Yahoo LTRC dataset.




Table 4: Comparison of Learning to Rank with 500 trees on Yahoo! LTRC Dataset

| Method | Time per Tree (sec) | NDCG@10 |
| --- | --- | --- |
| XGBoost | 0.826 | 0.7892 |
| XGBoost (colsample=0.5) | 0.506 | 0.7913 |
| pGBRT [[22](#bib.bib22)] | 2.576 | 0.7915 |

![Refer to caption](/html/1603.02754/assets/x10.png)


Figure 11: Comparison of out-of-core methods on different subsets of criteo data.
The missing data points are due to out of disk space.
We can find that basic algorithm can only handle 200M examples.
Adding compression gives 3x speedup, and sharding into two disks gives another 2x speedup.
The system runs out of file cache start from 400M examples. The algorithm really has to rely on disk after this point.
The compression+shard method has a less dramatic slowdown when running out of file cache, and exhibits a linear trend afterwards.



![Refer to caption](/html/1603.02754/assets/x11.png)


(a) End-to-end time cost include data loading

![Refer to caption](/html/1603.02754/assets/x12.png)


(b) Per iteration cost exclude data loading

Figure 12: 
Comparison of different distributed systems on 32 EC2 nodes for 10 iterations on different subset of criteo data.
XGBoost runs more 10x than spark per iteration and 2.2x as H2O’s optimized version
 (However, H2O is slow in loading the data, getting worse end-to-end time).
Note that spark suffers from drastic slow down when running out of memory.
XGBoost runs faster and scales smoothly to the full 1.7 billion examples with given resources by utilizing out-of-core computation.

![Refer to caption](/html/1603.02754/assets/x13.png)


Figure 13: 
Scaling of XGBoost with different number of machines on criteo full 1.7 billion dataset.
Using more machines results in more file cache and makes the system run faster,
causing the trend to be slightly super linear.
XGBoost can process the entire dataset using as little as four machines, and scales smoothly by utilizing more available resources.

We next evaluate the performance of XGBoost on the learning to rank problem.
We compare against pGBRT [[22](#bib.bib22)], the best previously pubished system on this task.
XGBoost runs exact greedy algorithm, while pGBRT only support an approximate algorithm.
The results are shown in Table [4](#S6.T4 "Table 4 ‣ 6.4 Learning to Rank ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System") and Fig. [10](#S6.F10 "Figure 10 ‣ 6.4 Learning to Rank ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
We find that XGBoost runs faster.
Interestingly, subsampling columns not only reduces running time, and but also gives a bit higher performance for this problem.
This could due to the fact that the subsampling helps prevent overfitting, which is observed by many of the users.

### 6.5 Out-of-core Experiment

We also evaluate our system in the out-of-core setting on the criteo data.
We conducted the experiment on one AWS c3.8xlarge machine (32 vcores, two 320 GB SSD, 60 GB RAM).
The results are shown in Figure [11](#S6.F11 "Figure 11 ‣ 6.4 Learning to Rank ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
We can find that compression helps to speed up computation by factor of three, and sharding into two disks further gives 2x speedup.
For this type of experiment, it is important to use a very large dataset to drain the system file cache for a real out-of-core setting.
This is indeed our setup. We can observe a transition point when the system runs out of file cache.
Note that the transition in the final method is less dramatic. This is due to larger disk throughput and better utilization of computation resources.
Our final method is able to process 1.7 billion examples on a single machine.

### 6.6 Distributed Experiment

Finally, we evaluate the system in the distributed setting. We set up a YARN cluster on EC2 with m3.2xlarge machines, which is a very common choice for clusters.
Each machine contains 8 virtual cores, 30GB of RAM and two 80GB SSD local disks. The dataset is stored on AWS S3 instead of HDFS to avoid purchasing persistent storage.

We first compare our system against two production-level distributed systems: Spark MLLib [[18](#bib.bib18)] and H2O 111111www.h2o.ai.
We use 32 m3.2xlarge machines and test the performance of the systems with various input size.
Both of the baseline systems are in-memory analytics frameworks that need to store the data in RAM, while XGBoost can switch to out-of-core setting when it runs out of memory. The results are shown in Fig. [12](#S6.F12 "Figure 12 ‣ 6.4 Learning to Rank ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
We can find that XGBoost runs faster than the baseline systems.
More importantly, it is able to take advantage of out-of-core computing and smoothly scale to all 1.7 billion examples with the given limited computing resources.
The baseline systems are only able to handle subset of the data with the given resources.
This experiment shows the advantage to bring all the system improvement together and solve a real-world scale problem.
We also evaluate the scaling property of XGBoost by varying the number of machines. The results are shown in Fig. [13](#S6.F13 "Figure 13 ‣ 6.4 Learning to Rank ‣ 6 End to End Evaluations ‣ XGBoost: A Scalable Tree Boosting System").
We can find XGBoost’s performance scales linearly as we add more machines. Importantly, XGBoost is able to handle the entire 1.7 billion
data with only four machines. This shows the system’s potential to handle even larger data.

## 7 Conclusion

In this paper, we described the lessons we learnt when building XGBoost, a scalable tree boosting system that is widely used by data scientists and provides state-of-the-art results on many problems.
We proposed a novel sparsity aware algorithm for handling sparse data and a theoretically justified weighted quantile sketch for approximate learning. Our experience shows that cache access patterns, data compression and sharding are essential elements for building a scalable end-to-end system for tree boosting. These lessons can be applied to other machine learning systems as well.
By combining these insights, XGBoost is able to solve real-world scale problems using a minimal amount of resources.

## Acknowledgments

We would like to thank Tyler B. Johnson, Marco Tulio Ribeiro, Sameer Singh, Arvind Krishnamurthy for their valuable feedback.
We also sincerely thank Tong He, Bing Xu, Michael Benesty, Yuan Tang, Hongliang Liu, Qiang Kou, Nan Zhu and all other contributors in the XGBoost community.
This work was supported in part by ONR (PECASE) N000141010672, NSF IIS 1258741 and the TerraSwarm Research
Center sponsored by MARCO and DARPA.

## References

* [1]

  R. Bekkerman.
  The present and the future of the kdd cup competition: an outsider’s
  perspective.
* [2]

  R. Bekkerman, M. Bilenko, and J. Langford.
  Scaling Up Machine Learning: Parallel and Distributed
  Approaches.
  Cambridge University Press, New York, NY, USA, 2011.
* [3]

  J. Bennett and S. Lanning.
  The netflix prize.
  In Proceedings of the KDD Cup Workshop 2007, pages 3–6, New
  York, Aug. 2007.
* [4]

  L. Breiman.
  Random forests.
  Maching Learning, 45(1):5–32, Oct. 2001.
* [5]

  C. Burges.
  From ranknet to lambdarank to lambdamart: An overview.
  Learning, 11:23–581, 2010.
* [6]

  O. Chapelle and Y. Chang.
  Yahoo! Learning to Rank Challenge Overview.
  Journal of Machine Learning Research - W & CP, 14:1–24, 2011.
* [7]

  T. Chen, H. Li, Q. Yang, and Y. Yu.
  General functional matrix factorization using gradient boosting.
  In Proceeding of 30th International Conference on Machine
  Learning (ICML’13), volume 1, pages 436–444, 2013.
* [8]

  T. Chen, S. Singh, B. Taskar, and C. Guestrin.
  Efficient second-order gradient boosting for conditional random
  fields.
  In Proceeding of 18th Artificial Intelligence and Statistics
  Conference (AISTATS’15), volume 1, 2015.
* [9]

  R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.
  LIBLINEAR: A library for large linear classification.
  Journal of Machine Learning Research, 9:1871–1874, 2008.
* [10]

  J. Friedman.
  Greedy function approximation: a gradient boosting machine.
  Annals of Statistics, 29(5):1189–1232, 2001.
* [11]

  J. Friedman.
  Stochastic gradient boosting.
  Computational Statistics & Data Analysis, 38(4):367–378,
  2002.
* [12]

  J. Friedman, T. Hastie, and R. Tibshirani.
  Additive logistic regression: a statistical view of boosting.
  Annals of Statistics, 28(2):337–407, 2000.
* [13]

  J. H. Friedman and B. E. Popescu.
  Importance sampled learning ensembles, 2003.
* [14]

  M. Greenwald and S. Khanna.
  Space-efficient online computation of quantile summaries.
  In Proceedings of the 2001 ACM SIGMOD International Conference
  on Management of Data, pages 58–66, 2001.
* [15]

  X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich,
  S. Bowers, and J. Q. n. Candela.
  Practical lessons from predicting clicks on ads at facebook.
  In Proceedings of the Eighth International Workshop on Data
  Mining for Online Advertising, ADKDD’14, 2014.
* [16]

  P. Li.
  Robust Logitboost and adaptive base class (ABC) Logitboost.
  In Proceedings of the Twenty-Sixth Conference Annual Conference
  on Uncertainty in Artificial Intelligence (UAI’10), pages 302–311, 2010.
* [17]

  P. Li, Q. Wu, and C. J. Burges.
  Mcrank: Learning to rank using multiple classification and gradient
  boosting.
  In Advances in Neural Information Processing Systems 20, pages
  897–904. 2008.
* [18]

  X. Meng, J. Bradley, B. Yavuz, E. Sparks, S. Venkataraman, D. Liu, J. Freeman,
  D. Tsai, M. Amde, S. Owen, D. Xin, R. Xin, M. J. Franklin, R. Zadeh,
  M. Zaharia, and A. Talwalkar.
  MLlib: Machine learning in apache spark.
  Journal of Machine Learning Research, 17(34):1–7, 2016.
* [19]

  B. Panda, J. S. Herbach, S. Basu, and R. J. Bayardo.
  Planet: Massively parallel learning of tree ensembles with mapreduce.
  Proceeding of VLDB Endowment, 2(2):1426–1437, Aug. 2009.
* [20]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  Journal of Machine Learning Research, 12:2825–2830, 2011.
* [21]

  G. Ridgeway.
  Generalized Boosted Models: A guide to the gbm package.
* [22]

  S. Tyree, K. Weinberger, K. Agrawal, and J. Paykin.
  Parallel boosted regression trees for web search ranking.
  In Proceedings of the 20th international conference on World
  wide web, pages 387–396. ACM, 2011.
* [23]

  J. Ye, J.-H. Chow, J. Chen, and Z. Zheng.
  Stochastic gradient boosted distributed decision trees.
  In Proceedings of the 18th ACM Conference on Information and
  Knowledge Management, CIKM ’09.
* [24]

  Q. Zhang and W. Wang.
  A fast algorithm for approximate quantiles in high speed data
  streams.
  In Proceedings of the 19th International Conference on
  Scientific and Statistical Database Management, 2007.
* [25]

  T. Zhang and R. Johnson.
  Learning nonlinear functions using regularized greedy forest.
  IEEE Transactions on Pattern Analysis and Machine Intelligence,
  36(5), 2014.

## Appendix A Weighted Quantile Sketch

In this section, we introduce the weighted quantile sketch algorithm.
Approximate answer of quantile queries is for many real-world applications.
One classical approach to this problem is GK algorithm [[14](#bib.bib14)] and extensions based on the GK framework [[24](#bib.bib24)].
The main component of these algorithms is a data structure called quantile summary, that is able to answer quantile queries with relative accuracy of ϵitalic-ϵ\epsilon.
Two operations are defined for a quantile summary:

* •

  A merge operation that combines two summaries with approximation error ϵ1subscriptitalic-ϵ1\epsilon\_{1} and ϵ2subscriptitalic-ϵ2\epsilon\_{2} together and create a merged summary with approximation error max⁡(ϵ1,ϵ2)subscriptitalic-ϵ1subscriptitalic-ϵ2\max(\epsilon\_{1},\epsilon\_{2}).
* •

  A prune operation that reduces the number of elements in the summary to b+1𝑏1b+1 and changes approximation error from ϵitalic-ϵ\epsilon to ϵ+1bitalic-ϵ1𝑏\epsilon+\frac{1}{b}.

A quantile summary with merge and prune operations forms basic building blocks of the distributed and streaming quantile computing algorithms [[24](#bib.bib24)].

In order to use quantile computation for approximate tree boosting, we need to find quantiles on weighted data. This more general problem is not supported
by any of the existing algorithm. In this section, we describe a non-trivial weighted quantile summary structure to solve this problem.
Importantly, the new algorithm contains merge and prune operations with *the same guarantee* as GK summary.
This allows our summary to be plugged into all the frameworks used GK summary as building block and answer quantile queries over weighted data efficiently.

### A.1 Formalization and Definitions

Given an input multi-set 𝒟={(x1,w1),(x2,w2)​⋯​(xn,wn)}𝒟subscript𝑥1subscript𝑤1subscript𝑥2subscript𝑤2⋯subscript𝑥𝑛subscript𝑤𝑛\mathcal{D}=\{(x\_{1},w\_{1}),(x\_{2},w\_{2})\cdots(x\_{n},w\_{n})\} such that wi∈[0,+∞),xi∈𝒳formulae-sequencesubscript𝑤𝑖0subscript𝑥𝑖𝒳w\_{i}\in[0,+\infty),x\_{i}\in\mathcal{X}.
Each xisubscript𝑥𝑖x\_{i} corresponds to a position of the point and wisubscript𝑤𝑖w\_{i} is the weight of the point. Assume we have a total order << defined on 𝒳𝒳\mathcal{X}.
Let us define two rank functions r𝒟−,r𝒟+:𝒳→[0,+∞):

subscriptsuperscript𝑟𝒟subscriptsuperscript𝑟𝒟
→𝒳0r^{-}\_{\mathcal{D}},r^{+}\_{\mathcal{D}}:\mathcal{X}\rightarrow[0,+\infty)

|  |  |  |  |
| --- | --- | --- | --- |
|  | r𝒟−​(y)=∑(x,w)∈𝒟,x<ywsubscriptsuperscript𝑟𝒟𝑦subscriptformulae-sequence𝑥𝑤𝒟𝑥𝑦𝑤r^{-}\_{\mathcal{D}}(y)=\sum\_{(x,w)\in\mathcal{D},x<y}w |  | (10) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | r𝒟+​(y)=∑(x,w)∈𝒟,x≤ywsubscriptsuperscript𝑟𝒟𝑦subscriptformulae-sequence𝑥𝑤𝒟𝑥𝑦𝑤r^{+}\_{\mathcal{D}}(y)=\sum\_{(x,w)\in\mathcal{D},x\leq y}w |  | (11) |

We should note that since 𝒟𝒟\mathcal{D} is defined to be a *multiset* of the points. It can contain multiple record with exactly same position x𝑥x and weight w𝑤w.
We also define another weight function ω𝒟:𝒳→[0,+∞):subscript𝜔𝒟→𝒳0\omega\_{\mathcal{D}}:\mathcal{X}\rightarrow[0,+\infty) as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ω𝒟​(y)=r𝒟+​(y)−r𝒟−​(y)=∑(x,w)∈𝒟,x=yw.subscript𝜔𝒟𝑦subscriptsuperscript𝑟𝒟𝑦subscriptsuperscript𝑟𝒟𝑦subscriptformulae-sequence𝑥𝑤𝒟𝑥𝑦𝑤\omega\_{\mathcal{D}}(y)=r^{+}\_{\mathcal{D}}(y)-r^{-}\_{\mathcal{D}}(y)=\sum\_{(x,w)\in\mathcal{D},x=y}w. |  | (12) |

Finally, we also define the weight of multi-set 𝒟𝒟\mathcal{D} to be the sum of weights of all the points in the set

|  |  |  |  |
| --- | --- | --- | --- |
|  | ω​(𝒟)=∑(x,w)∈𝒟w𝜔𝒟subscript𝑥𝑤𝒟𝑤\omega(\mathcal{D})=\sum\_{(x,w)\in\mathcal{D}}w |  | (13) |

Our task is given a series of input 𝒟𝒟\mathcal{D}, to estimate r+​(y)superscript𝑟𝑦r^{+}(y) and r−​(y)superscript𝑟𝑦r^{-}(y) for y∈𝒳𝑦𝒳y\in\mathcal{X} as well as finding points
with specific rank.
Given these notations, we define quantile summary of weighted examples as follows:

###### Definition A.1

Quantile Summary of Weighted Data
  
A quantile summary for 𝒟𝒟\mathcal{D} is defined to be tuple Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}), where
S={x1,x2,⋯,xk}𝑆subscript𝑥1subscript𝑥2⋯subscript𝑥𝑘S=\{x\_{1},x\_{2},\cdots,x\_{k}\} is selected from the points in 𝒟𝒟\mathcal{D} (i.e. xi∈{x|(x,w)∈𝒟}subscript𝑥𝑖conditional-set𝑥𝑥𝑤𝒟x\_{i}\in\{x|(x,w)\in\mathcal{D}\}) with the following properties:

1) xi<xi+1​ for all ​isubscript𝑥𝑖subscript𝑥𝑖1 for all 𝑖x\_{i}<x\_{i+1}\mbox{ for all }i, and x1subscript𝑥1x\_{1} and xksubscript𝑥𝑘x\_{k} are minimum and maximum point in 𝒟𝒟\mathcal{D}:

|  |  |  |
| --- | --- | --- |
|  | x1=min(x,w)∈𝒟⁡x,xk=max(x,w)∈𝒟⁡xformulae-sequencesubscript𝑥1subscript𝑥𝑤𝒟𝑥subscript𝑥𝑘subscript𝑥𝑤𝒟𝑥x\_{1}=\min\_{(x,w)\in\mathcal{D}}x,\ \ x\_{k}=\max\_{(x,w)\in\mathcal{D}}x |  |

2) r~𝒟+superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{+}, r~𝒟−superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{-} and ω~𝒟subscript~𝜔𝒟\tilde{\omega}\_{\mathcal{D}} are functions in S→[0,+∞)→𝑆0S\rightarrow[0,+\infty), that satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(xi)≤r𝒟−​(xi),r~𝒟+​(xi)≥r𝒟+​(xi),ω~𝒟​(xi)≤ω𝒟​(xi),formulae-sequencesuperscriptsubscript~𝑟𝒟subscript𝑥𝑖subscriptsuperscript𝑟𝒟subscript𝑥𝑖formulae-sequencesuperscriptsubscript~𝑟𝒟subscript𝑥𝑖subscriptsuperscript𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖subscript𝜔𝒟subscript𝑥𝑖\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})\leq r^{-}\_{\mathcal{D}}(x\_{i}),\ \ \tilde{r}\_{\mathcal{D}}^{+}(x\_{i})\geq r^{+}\_{\mathcal{D}}(x\_{i}),\ \ \tilde{\omega}\_{\mathcal{D}}(x\_{i})\leq\omega\_{\mathcal{D}}(x\_{i}), |  | (14) |

the equality sign holds for maximum and minimum point (
r~𝒟−​(xi)=r𝒟−​(xi)superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscriptsuperscript𝑟𝒟subscript𝑥𝑖\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})=r^{-}\_{\mathcal{D}}(x\_{i}), r~𝒟+​(xi)=r𝒟+​(xi)superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscriptsuperscript𝑟𝒟subscript𝑥𝑖\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})=r^{+}\_{\mathcal{D}}(x\_{i}) and ω~𝒟​(xi)=ω𝒟​(xi)subscript~𝜔𝒟subscript𝑥𝑖subscript𝜔𝒟subscript𝑥𝑖\tilde{\omega}\_{\mathcal{D}}(x\_{i})=\omega\_{\mathcal{D}}(x\_{i}) for i∈{1,k}𝑖1𝑘i\in\{1,k\}).
  
Finally, the function value must also satisfy the following constraints

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(xi)+ω~𝒟​(xi)≤r~𝒟−​(xi+1),r~𝒟+​(xi)≤r~𝒟+​(xi+1)−ω~𝒟​(xi+1)formulae-sequencesuperscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})\leq\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1}),\ \ \tilde{r}\_{\mathcal{D}}^{+}(x\_{i})\leq\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1}) |  | (15) |

Since these functions are only defined on S𝑆S, it is suffice to use 4​k4𝑘4k record to store the summary. Specifically, we need to remember each xisubscript𝑥𝑖x\_{i} and the corresponding function values of each xisubscript𝑥𝑖x\_{i}.

###### Definition A.2

Extension of Function Domains
  
Given a quantile summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) defined in Definition [A.1](#A1.Thmthmdef1 "Definition A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"), the domain of r~𝒟+superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{+}, r~𝒟−superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{-} and ω~𝒟subscript~𝜔𝒟\tilde{\omega}\_{\mathcal{D}} were defined only in S𝑆S. We extend the definition of these functions to 𝒳→[0,+∞)→𝒳0\mathcal{X}\rightarrow[0,+\infty) as follows
  
When y<x1𝑦subscript𝑥1y<x\_{1}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)=0,r~𝒟+​(y)=0,ω~𝒟​(y)=0formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦0formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦0subscript~𝜔𝒟𝑦0\tilde{r}\_{\mathcal{D}}^{-}(y)=0,\ \tilde{r}\_{\mathcal{D}}^{+}(y)=0,\ \tilde{\omega}\_{\mathcal{D}}(y)=0 |  | (16) |

When y>xk𝑦subscript𝑥𝑘y>x\_{k}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)=r~𝒟+​(xk),r~𝒟+​(y)=r~𝒟+​(xk),ω~𝒟​(y)=0formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑘formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑘subscript~𝜔𝒟𝑦0\tilde{r}\_{\mathcal{D}}^{-}(y)=\tilde{r}\_{\mathcal{D}}^{+}(x\_{k}),\ \tilde{r}\_{\mathcal{D}}^{+}(y)=\tilde{r}\_{\mathcal{D}}^{+}(x\_{k}),\ \tilde{\omega}\_{\mathcal{D}}(y)=0 |  | (17) |

When y∈(xi,xi+1)𝑦subscript𝑥𝑖subscript𝑥𝑖1y\in(x\_{i},x\_{i+1}) for some i𝑖i:

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)=r~𝒟−​(xi)+ω~𝒟​(xi),r~𝒟+​(y)=r~𝒟+​(xi+1)−ω~𝒟​(xi+1),ω~𝒟​(y)=0formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1subscript~𝜔𝒟𝑦0\begin{split}\tilde{r}\_{\mathcal{D}}^{-}(y)&=\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i}),\\ \tilde{r}\_{\mathcal{D}}^{+}(y)&=\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1}),\\ \tilde{\omega}\_{\mathcal{D}}(y)&=0\end{split} |  | (18) |

###### Lemma A.1

Extended Constraint
  
The extended definition of r~𝒟−superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{-}, r~𝒟+superscriptsubscript~𝑟𝒟\tilde{r}\_{\mathcal{D}}^{+}, ω~𝒟subscript~𝜔𝒟\tilde{\omega}\_{\mathcal{D}} satisfies the following constraints

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)≤r𝒟−​(y),r~𝒟+​(y)≥r𝒟+​(y),ω~𝒟​(y)≤ω𝒟​(y)formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦subscriptsuperscript𝑟𝒟𝑦formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦subscriptsuperscript𝑟𝒟𝑦subscript~𝜔𝒟𝑦subscript𝜔𝒟𝑦\tilde{r}\_{\mathcal{D}}^{-}(y)\leq r^{-}\_{\mathcal{D}}(y),\ \ \tilde{r}\_{\mathcal{D}}^{+}(y)\geq r^{+}\_{\mathcal{D}}(y),\ \ \tilde{\omega}\_{\mathcal{D}}(y)\leq\omega\_{\mathcal{D}}(y) |  | (19) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)+ω~𝒟​(y)≤r~𝒟−​(x),r~𝒟+​(y)≤r~𝒟+​(x)−ω~𝒟​(x), for all ​y<xformulae-sequencesuperscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦superscriptsubscript~𝑟𝒟𝑥formulae-sequencesuperscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟𝑥subscript~𝜔𝒟𝑥 for all 𝑦𝑥\tilde{r}\_{\mathcal{D}}^{-}(y)+\tilde{\omega}\_{\mathcal{D}}(y)\leq\tilde{r}\_{\mathcal{D}}^{-}(x),\ \ \tilde{r}\_{\mathcal{D}}^{+}(y)\leq\tilde{r}\_{\mathcal{D}}^{+}(x)-\tilde{\omega}\_{\mathcal{D}}(x),\mbox{ for all }y<x |  | (20) |

###### Proof A.1.

The only non-trivial part is to prove the case when y∈(xi,xi+1)𝑦subscript𝑥𝑖subscript𝑥𝑖1y\in(x\_{i},x\_{i+1}):

|  |  |  |
| --- | --- | --- |
|  | r~𝒟−​(y)=r~𝒟−​(xi)+ω~𝒟​(xi)≤r𝒟−​(xi)+ω𝒟​(xi)≤r𝒟−​(y)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript𝑟𝒟subscript𝑥𝑖subscript𝜔𝒟subscript𝑥𝑖superscriptsubscript𝑟𝒟𝑦\tilde{r}\_{\mathcal{D}}^{-}(y)=\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})\leq r\_{\mathcal{D}}^{-}(x\_{i})+\omega\_{\mathcal{D}}(x\_{i})\leq r\_{\mathcal{D}}^{-}(y) |  |

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(y)=r~𝒟+​(xi+1)−ω~𝒟​(xi+1)≥r𝒟+​(xi+1)−ω𝒟​(xi+1)≥r𝒟+​(y)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1superscriptsubscript𝑟𝒟subscript𝑥𝑖1subscript𝜔𝒟subscript𝑥𝑖1superscriptsubscript𝑟𝒟𝑦\tilde{r}\_{\mathcal{D}}^{+}(y)=\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})\geq r\_{\mathcal{D}}^{+}(x\_{i+1})-\omega\_{\mathcal{D}}(x\_{i+1})\geq r\_{\mathcal{D}}^{+}(y) |  |

This proves Eq. ([19](#A1.E19 "In Lemma A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System")).
Furthermore, we can verify that

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(xi)≤r~𝒟+​(xi+1)−ω~𝒟​(xi+1)=r~𝒟+​(y)−ω~𝒟​(y)superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})\leq\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})=\tilde{r}\_{\mathcal{D}}^{+}(y)-\tilde{\omega}\_{\mathcal{D}}(y) |  |

|  |  |  |
| --- | --- | --- |
|  | r~𝒟−​(y)+ω~𝒟​(y)=r~𝒟−​(xi)+ω~𝒟​(xi)+0≤r~𝒟−​(xi+1)superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖0superscriptsubscript~𝑟𝒟subscript𝑥𝑖1\tilde{r}\_{\mathcal{D}}^{-}(y)+\tilde{\omega}\_{\mathcal{D}}(y)=\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})+0\leq\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1}) |  |

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(y)=r~𝒟+​(xi+1)−ω~𝒟​(xi+1)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1\tilde{r}\_{\mathcal{D}}^{+}(y)=\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1}) |  |

Using these facts and transitivity of << relation, we can prove Eq. ([20](#A1.E20 "In Lemma A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"))

We should note that the extension is based on the ground case defined in S𝑆S, and we do not require extra space to store the summary in order to use the extended definition. We are now ready to introduce the definition of ϵitalic-ϵ\epsilon-approximate quantile summary.

###### Definition A.3

ϵitalic-ϵ\epsilon-Approximate Quantile Summary
  
Given a quantile summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}), we call it is ϵitalic-ϵ\epsilon-approximate summary if for any y∈𝒳𝑦𝒳y\in\mathcal{X}

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(y)−r~𝒟−​(y)−ω~𝒟​(y)≤ϵ​ω​(𝒟)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦italic-ϵ𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+}(y)-\tilde{r}\_{\mathcal{D}}^{-}(y)-\tilde{\omega}\_{\mathcal{D}}(y)\leq\epsilon\omega(\mathcal{D}) |  | (21) |

We use this definition since we know that r−​(y)∈[r~𝒟−​(y),r~𝒟+​(y)−ω~𝒟​(y)]superscript𝑟𝑦superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦r^{-}(y)\in[\tilde{r}\_{\mathcal{D}}^{-}(y),\tilde{r}\_{\mathcal{D}}^{+}(y)-\tilde{\omega}\_{\mathcal{D}}(y)] and r+​(y)∈[r~𝒟−​(y)+ω~𝒟​(y),r~𝒟+​(y)]superscript𝑟𝑦superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦superscriptsubscript~𝑟𝒟𝑦r^{+}(y)\in[\tilde{r}\_{\mathcal{D}}^{-}(y)+\tilde{\omega}\_{\mathcal{D}}(y),\tilde{r}\_{\mathcal{D}}^{+}(y)]. Eq. ([21](#A1.E21 "In Definition A.3 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System")) means the we can get estimation of r+​(y)superscript𝑟𝑦r^{+}(y) and r−​(y)superscript𝑟𝑦r^{-}(y) by error of at most ϵ​ω​(𝒟)italic-ϵ𝜔𝒟\epsilon\omega(\mathcal{D}).

###### Lemma A.2

Quantile summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) is an ϵitalic-ϵ\epsilon-approximate summary if and only if the following two condition holds

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(xi)−r~𝒟−​(xi)−ω~𝒟​(xi)≤ϵ​ω​(𝒟)superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖italic-ϵ𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})\leq\epsilon\omega(\mathcal{D}) |  | (22) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(xi+1)−r~𝒟−​(xi)−ω~𝒟​(xi+1)−ω~𝒟​(xi)≤ϵ​ω​(𝒟)superscriptsubscript~𝑟𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖italic-ϵ𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})\leq\epsilon\omega(\mathcal{D}) |  | (23) |

###### Proof A.2.

The key is again consider y∈(xi,xi+1)𝑦subscript𝑥𝑖subscript𝑥𝑖1y\in(x\_{i},x\_{i+1})

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(y)−r~𝒟−​(y)−ω~𝒟​(y)=[r~𝒟+​(xi+1)−ω~𝒟​(xi+1)]−[r~𝒟+​(xi)+ω~𝒟​(xi)]−0superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖0\tilde{r}\_{\mathcal{D}}^{+}(y)-\tilde{r}\_{\mathcal{D}}^{-}(y)-\tilde{\omega}\_{\mathcal{D}}(y)=[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})]-[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})]-0 |  |

This means the condition in Eq. ([23](#A1.E23 "In Lemma A.2 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System")) plus Eq.([22](#A1.E22 "In Lemma A.2 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System")) can give us Eq. ([21](#A1.E21 "In Definition A.3 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"))

Property of Extended Function In this section, we have introduced the extension of function r~𝒟+,r~𝒟−,ω~𝒟

superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}} to 𝒳→[0,+∞)→𝒳0\mathcal{X}\rightarrow[0,+\infty).
The key theme discussed in this section is the relation of *constraints on the original function and constraints on the extended function*.
Lemma [A.1](#A1.Thmthmlemma1 "Lemma A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System") and  [A.2](#A1.Thmthmlemma2 "Lemma A.2 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System") show that the constraints on the original function can lead to in more general constraints on the extended function.
This is a very useful property which will be used in the proofs in later sections.

### A.2 Construction of Initial Summary

Given a small multi-set 𝒟={(x1,w1),(x2,w2),⋯,(xn,wn)}𝒟subscript𝑥1subscript𝑤1subscript𝑥2subscript𝑤2⋯subscript𝑥𝑛subscript𝑤𝑛\mathcal{D}=\{(x\_{1},w\_{1}),(x\_{2},w\_{2}),\cdots,(x\_{n},w\_{n})\}, we can construct initial summary Q​(𝒟)={S,r~𝒟+,r~𝒟−,ω~𝒟}𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=\{S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}\}, with S𝑆S to the set of all values in 𝒟𝒟\mathcal{D} (S={x|(x,w)∈𝒟}𝑆conditional-set𝑥𝑥𝑤𝒟S=\{x|(x,w)\in\mathcal{D}\}), and r~𝒟+,r~𝒟−,ω~𝒟

superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}} defined to be

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(x)=r𝒟+​(x),r~𝒟−​(x)=r𝒟−​(x),ω~𝒟​(x)=ω𝒟​(x)​ for ​x∈Sformulae-sequencesuperscriptsubscript~𝑟𝒟𝑥subscriptsuperscript𝑟𝒟𝑥formulae-sequencesuperscriptsubscript~𝑟𝒟𝑥subscriptsuperscript𝑟𝒟𝑥subscript~𝜔𝒟𝑥subscript𝜔𝒟𝑥 for 𝑥𝑆\tilde{r}\_{\mathcal{D}}^{+}(x)=r^{+}\_{\mathcal{D}}(x),\ \ \tilde{r}\_{\mathcal{D}}^{-}(x)=r^{-}\_{\mathcal{D}}(x),\ \ \tilde{\omega}\_{\mathcal{D}}(x)=\omega\_{\mathcal{D}}(x)\mbox{ for }x\in S |  | (24) |

The constructed summary is 00-approximate summary, since it can answer all the queries accurately. The constructed summary can be feed into future operations described in the latter sections.

### A.3 Merge Operation

In this section, we define how we can merge the two summaries together. Assume we have Q​(𝒟1)=(S1,r~𝒟1+,r~𝒟1−,ω~𝒟1)𝑄subscript𝒟1subscript𝑆1superscriptsubscript~𝑟subscript𝒟1superscriptsubscript~𝑟subscript𝒟1subscript~𝜔subscript𝒟1Q(\mathcal{D}\_{1})=(S\_{1},\tilde{r}\_{\mathcal{D}\_{1}}^{+},\tilde{r}\_{\mathcal{D}\_{1}}^{-},\tilde{\omega}\_{\mathcal{D}\_{1}}) and Q​(𝒟2)=(S2,r~𝒟1+,r~𝒟2−,ω~𝒟2)𝑄subscript𝒟2subscript𝑆2superscriptsubscript~𝑟subscript𝒟1superscriptsubscript~𝑟subscript𝒟2subscript~𝜔subscript𝒟2Q(\mathcal{D}\_{2})=(S\_{2},\tilde{r}\_{\mathcal{D}\_{1}}^{+},\tilde{r}\_{\mathcal{D}\_{2}}^{-},\tilde{\omega}\_{\mathcal{D}\_{2}}) quantile summary of two dataset 𝒟1subscript𝒟1\mathcal{D}\_{1} and 𝒟2subscript𝒟2\mathcal{D}\_{2}. Let 𝒟=𝒟1∪𝒟2𝒟subscript𝒟1subscript𝒟2\mathcal{D}=\mathcal{D}\_{1}\cup\mathcal{D}\_{2}, and define the merged summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) as follows.

|  |  |  |  |
| --- | --- | --- | --- |
|  | S={x1,x2​⋯,xk},xi∈S1​ or ​xi∈S2formulae-sequence𝑆subscript𝑥1subscript𝑥2⋯subscript𝑥𝑘subscript𝑥𝑖subscript𝑆1 or subscript𝑥𝑖subscript𝑆2S=\{x\_{1},x\_{2}\cdots,x\_{k}\},x\_{i}\in S\_{1}\mbox{ or }x\_{i}\in S\_{2} |  | (25) |

The points in S𝑆S are combination of points in S1subscript𝑆1S\_{1} and S2subscript𝑆2S\_{2}. And the function r~𝒟+,r~𝒟−,ω~𝒟

superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}} are defined to be

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(xi)=r~𝒟1−​(xi)+r~𝒟2−​(xi)superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟subscript𝒟1subscript𝑥𝑖superscriptsubscript~𝑟subscript𝒟2subscript𝑥𝑖\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})=\tilde{r}\_{\mathcal{D}\_{1}}^{-}(x\_{i})+\tilde{r}\_{\mathcal{D}\_{2}}^{-}(x\_{i}) |  | (26) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(xi)=r~𝒟1+​(xi)+r~𝒟2+​(xi)superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟subscript𝒟1subscript𝑥𝑖superscriptsubscript~𝑟subscript𝒟2subscript𝑥𝑖\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})=\tilde{r}\_{\mathcal{D}\_{1}}^{+}(x\_{i})+\tilde{r}\_{\mathcal{D}\_{2}}^{+}(x\_{i}) |  | (27) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | ω~𝒟​(xi)=ω~𝒟1​(xi)+ω~𝒟2​(xi)subscript~𝜔𝒟subscript𝑥𝑖subscript~𝜔subscript𝒟1subscript𝑥𝑖subscript~𝜔subscript𝒟2subscript𝑥𝑖\tilde{\omega}\_{\mathcal{D}}(x\_{i})=\tilde{\omega}\_{\mathcal{D}\_{1}}(x\_{i})+\tilde{\omega}\_{\mathcal{D}\_{2}}(x\_{i}) |  | (28) |

Here we use functions defined on S→[0,+∞)→𝑆0S\rightarrow[0,+\infty) on the left sides of equalities
and use the extended function definitions on the right sides.

Due to additive nature of r+superscript𝑟r^{+}, r−superscript𝑟r^{-} and ω𝜔\omega, which can be formally written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | r𝒟−​(y)=r𝒟1−​(y)+r𝒟2−​(y),r𝒟+​(y)=r𝒟1+​(y)+r𝒟2+​(y),ω𝒟​(y)=ω𝒟1​(y)+ω𝒟2​(y),formulae-sequencesubscriptsuperscript𝑟𝒟𝑦subscriptsuperscript𝑟subscript𝒟1𝑦subscriptsuperscript𝑟subscript𝒟2𝑦formulae-sequencesubscriptsuperscript𝑟𝒟𝑦subscriptsuperscript𝑟subscript𝒟1𝑦subscriptsuperscript𝑟subscript𝒟2𝑦subscript𝜔𝒟𝑦subscript𝜔subscript𝒟1𝑦subscript𝜔subscript𝒟2𝑦\begin{split}r^{-}\_{\mathcal{D}}(y)=&r^{-}\_{\mathcal{D}\_{1}}(y)+r^{-}\_{\mathcal{D}\_{2}}(y),\\ r^{+}\_{\mathcal{D}}(y)=&r^{+}\_{\mathcal{D}\_{1}}(y)+r^{+}\_{\mathcal{D}\_{2}}(y),\\ \omega\_{\mathcal{D}}(y)=&\omega\_{\mathcal{D}\_{1}}(y)+\omega\_{\mathcal{D}\_{2}}(y),\end{split} |  | (29) |

and the extended constraint property in Lemma [A.1](#A1.Thmthmlemma1 "Lemma A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"), we can verify that
Q​(𝒟)𝑄𝒟Q(\mathcal{D}) satisfies all the constraints in Definition [A.1](#A1.Thmthmdef1 "Definition A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"). Therefore it is a valid quantile summary.

###### Lemma A.3

The combined quantile summary satisfies

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟−​(y)=r~𝒟1−​(y)+r~𝒟2−​(y)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟subscript𝒟1𝑦superscriptsubscript~𝑟subscript𝒟2𝑦\tilde{r}\_{\mathcal{D}}^{-}(y)=\tilde{r}\_{\mathcal{D}\_{1}}^{-}(y)+\tilde{r}\_{\mathcal{D}\_{2}}^{-}(y) |  | (30) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | r~𝒟+​(y)=r~𝒟1+​(y)+r~𝒟2+​(y)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟subscript𝒟1𝑦superscriptsubscript~𝑟subscript𝒟2𝑦\tilde{r}\_{\mathcal{D}}^{+}(y)=\tilde{r}\_{\mathcal{D}\_{1}}^{+}(y)+\tilde{r}\_{\mathcal{D}\_{2}}^{+}(y) |  | (31) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | ω~𝒟​(y)=ω~𝒟1​(y)+ω~𝒟2​(y)subscript~𝜔𝒟𝑦subscript~𝜔subscript𝒟1𝑦subscript~𝜔subscript𝒟2𝑦\tilde{\omega}\_{\mathcal{D}}(y)=\tilde{\omega}\_{\mathcal{D}\_{1}}(y)+\tilde{\omega}\_{\mathcal{D}\_{2}}(y) |  | (32) |

for all y∈𝒳𝑦𝒳y\in\mathcal{X}

This can be obtained by straight-forward application of Definition [A.2](#A1.Thmthmdef2 "Definition A.2 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System").

###### Theorem A.1

If Q​(𝒟1)𝑄subscript𝒟1Q(\mathcal{D}\_{1}) is ϵ1subscriptitalic-ϵ1\epsilon\_{1}-approximate summary, and Q​(𝒟2)𝑄subscript𝒟2Q(\mathcal{D}\_{2}) is ϵ2subscriptitalic-ϵ2\epsilon\_{2}-approximate summary. Then the merged summary Q​(𝒟)𝑄𝒟Q(\mathcal{D}) is max⁡(ϵ1,ϵ2)subscriptitalic-ϵ1subscriptitalic-ϵ2\max(\epsilon\_{1},\epsilon\_{2})-approximate summary.

###### Proof A.3.

For any y∈𝒳𝑦𝒳y\in\mathcal{X}, we have

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(y)−r~𝒟−​(y)−ω~𝒟​(y)=[r~𝒟1+​(y)+r~𝒟2+​(y)]−[r~𝒟1−​(y)+r~𝒟2−​(y)]−[ω~𝒟1​(y)+ω~𝒟2​(y)]≤ϵ1​ω​(𝒟1)+ϵ2​ω​(𝒟2)≤max⁡(ϵ1,ϵ2)​ω​(𝒟1∪𝒟2)superscriptsubscript~𝑟𝒟𝑦superscriptsubscript~𝑟𝒟𝑦subscript~𝜔𝒟𝑦delimited-[]superscriptsubscript~𝑟subscript𝒟1𝑦superscriptsubscript~𝑟subscript𝒟2𝑦delimited-[]superscriptsubscript~𝑟subscript𝒟1𝑦superscriptsubscript~𝑟subscript𝒟2𝑦delimited-[]subscript~𝜔subscript𝒟1𝑦subscript~𝜔subscript𝒟2𝑦subscriptitalic-ϵ1𝜔subscript𝒟1subscriptitalic-ϵ2𝜔subscript𝒟2subscriptitalic-ϵ1subscriptitalic-ϵ2𝜔subscript𝒟1subscript𝒟2\begin{split}&\tilde{r}\_{\mathcal{D}}^{+}(y)-\tilde{r}\_{\mathcal{D}}^{-}(y)-\tilde{\omega}\_{\mathcal{D}}(y)\\ =&[\tilde{r}\_{\mathcal{D}\_{1}}^{+}(y)+\tilde{r}\_{\mathcal{D}\_{2}}^{+}(y)]-[\tilde{r}\_{\mathcal{D}\_{1}}^{-}(y)+\tilde{r}\_{\mathcal{D}\_{2}}^{-}(y)]-[\tilde{\omega}\_{\mathcal{D}\_{1}}(y)+\tilde{\omega}\_{\mathcal{D}\_{2}}(y)]\\ \leq&\epsilon\_{1}\omega(\mathcal{D}\_{1})+\epsilon\_{2}\omega(\mathcal{D}\_{2})\leq\max(\epsilon\_{1},\epsilon\_{2})\omega(\mathcal{D}\_{1}\cup\mathcal{D}\_{2})\end{split} |  |

Here the first inequality is due to Lemma [A.3](#A1.Thmthmlemma3 "Lemma A.3 ‣ A.3 Merge Operation ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System").

### A.4 Prune Operation

Input: d𝑑d: 0≤d≤ω​(𝒟)0𝑑𝜔𝒟0\leq d\leq\omega(\mathcal{D})

Input: Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) where S=x1,x2,⋯,xk𝑆

subscript𝑥1subscript𝑥2⋯subscript𝑥𝑘S={x\_{1},x\_{2},\cdots,x\_{k}}

if *d<12​[r~𝒟−​(x1)+r~𝒟+​(x1)]𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1d<\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})]* then
return x1subscript𝑥1x\_{1}
;

if *d≥12​[r~𝒟−​(xk)+r~𝒟+​(xk)]𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘d\geq\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{k})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})]* then
return xksubscript𝑥𝑘x\_{k}
;

Find i𝑖i such that 12​[r~𝒟−​(xi)+r~𝒟+​(xi)]≤d<12​[r~𝒟−​(xi+1)+r~𝒟+​(xi+1)]12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖1\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})]\leq d<\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})]

if *2​d<r~𝒟−​(xi)+ω~𝒟​(xi)+r~𝒟+​(xi+1)−ω~𝒟​(xi+1)2𝑑superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖12d<\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})* then

return xisubscript𝑥𝑖x\_{i}

else

return xi+1subscript𝑥𝑖1x\_{i+1}

end if

Algorithm 4 Query Function g​(Q,d)𝑔𝑄𝑑g(Q,d)

Before we start discussing the prune operation, we first introduce a query function g​(Q,d)𝑔𝑄𝑑g(Q,d). The definition of function is shown in Algorithm [4](#alg4 "In A.4 Prune Operation ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System").
For a given rank d𝑑d, the function returns a x𝑥x whose rank is close to d𝑑d. This property is formally described in the following Lemma.

###### Lemma A.4

For a given ϵitalic-ϵ\epsilon-approximate summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}), x∗=g​(Q,d)superscript𝑥𝑔𝑄𝑑x^{\*}=g(Q,d) satisfies the following property

|  |  |  |  |
| --- | --- | --- | --- |
|  | d≥r~𝒟+​(x∗)−ω~𝒟​(x∗)−ϵ2​ω​(𝒟)d≤r~𝒟−​(x∗)+ω~𝒟​(x∗)+ϵ2​ω​(𝒟)𝑑superscriptsubscript~𝑟𝒟superscript𝑥subscript~𝜔𝒟superscript𝑥italic-ϵ2𝜔𝒟𝑑superscriptsubscript~𝑟𝒟superscript𝑥subscript~𝜔𝒟superscript𝑥italic-ϵ2𝜔𝒟\begin{split}d&\geq\tilde{r}\_{\mathcal{D}}^{+}(x^{\*})-\tilde{\omega}\_{\mathcal{D}}(x^{\*})-\frac{\epsilon}{2}\omega(\mathcal{D})\\ d&\leq\tilde{r}\_{\mathcal{D}}^{-}(x^{\*})+\tilde{\omega}\_{\mathcal{D}}(x^{\*})+\frac{\epsilon}{2}\omega(\mathcal{D})\end{split} |  | (33) |

###### Proof A.4.

We need to discuss four possible cases

* •

  d<12​[r~𝒟−​(x1)+r~𝒟+​(x1)]𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1d<\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})] and x∗=x1superscript𝑥subscript𝑥1x^{\*}=x\_{1}. Note that the rank information for x1subscript𝑥1x\_{1} is accurate
  (ω~𝒟​(x1)=r~𝒟+​(x1)=ω​(x1)subscript~𝜔𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1𝜔subscript𝑥1\tilde{\omega}\_{\mathcal{D}}(x\_{1})=\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})=\omega(x\_{1}), r~𝒟−​(x1)=0superscriptsubscript~𝑟𝒟subscript𝑥10\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})=0), we have

  |  |  |  |
  | --- | --- | --- |
  |  | d≥0−ϵ2​ω​(𝒟)=r~𝒟+​(x1)−ω~𝒟​(x1)−ϵ2​ω​(𝒟)d<12​[r~𝒟−​(x1)+r~𝒟+​(x1)]≤r~𝒟−​(x1)+r~𝒟+​(x1)=r~𝒟−​(x1)+ω~𝒟+​(x1)𝑑0italic-ϵ2𝜔𝒟superscriptsubscript~𝑟𝒟subscript𝑥1subscript~𝜔𝒟subscript𝑥1italic-ϵ2𝜔𝒟𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝑟𝒟subscript𝑥1superscriptsubscript~𝜔𝒟subscript𝑥1\begin{split}d&\geq 0-\frac{\epsilon}{2}\omega(\mathcal{D})=\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})-\tilde{\omega}\_{\mathcal{D}}(x\_{1})-\frac{\epsilon}{2}\omega(\mathcal{D})\\ d&<\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})]\\ &\leq\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{1})\\ &=\tilde{r}\_{\mathcal{D}}^{-}(x\_{1})+\tilde{\omega}\_{\mathcal{D}}^{+}(x\_{1})\end{split} |  |
* •

  d≥12​[r~𝒟−​(xk)+r~𝒟+​(xk)]𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘d\geq\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{k})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})] and x∗=xksuperscript𝑥subscript𝑥𝑘x^{\*}=x\_{k}, then

  |  |  |  |
  | --- | --- | --- |
  |  | d≥12​[r~𝒟−​(xk)+r~𝒟+​(xk)]=r~𝒟+​(xk)−12​[r~𝒟+​(xk)−r~𝒟−​(xk)]=r~𝒟+​(xk)−12​ω~𝒟​(xk)d<ω​(𝒟)+ϵ2​ω​(𝒟)=r~𝒟−​(xk)+ω~𝒟​(xk)+ϵ2​ω​(𝒟)𝑑12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘superscriptsubscript~𝑟𝒟subscript𝑥𝑘12subscript~𝜔𝒟subscript𝑥𝑘𝑑𝜔𝒟italic-ϵ2𝜔𝒟superscriptsubscript~𝑟𝒟subscript𝑥𝑘subscript~𝜔𝒟subscript𝑥𝑘italic-ϵ2𝜔𝒟\begin{split}d&\geq\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{-}(x\_{k})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})]\\ &=\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})-\frac{1}{2}[\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{k})]\\ &=\tilde{r}\_{\mathcal{D}}^{+}(x\_{k})-\frac{1}{2}\tilde{\omega}\_{\mathcal{D}}(x\_{k})\\ d&<\omega(\mathcal{D})+\frac{\epsilon}{2}\omega(\mathcal{D})=\tilde{r}\_{\mathcal{D}}^{-}(x\_{k})+\tilde{\omega}\_{\mathcal{D}}(x\_{k})+\frac{\epsilon}{2}\omega(\mathcal{D})\end{split} |  |
* •

  x∗=xisuperscript𝑥subscript𝑥𝑖x^{\*}=x\_{i} in the general case, then

  |  |  |  |
  | --- | --- | --- |
  |  | 2​d<r~𝒟−​(xi)+ω~𝒟​(xi)+r~𝒟+​(xi+1)−ω~𝒟​(xi+1)=2​[r~𝒟−​(xi)+ω~𝒟​(xi)]+[r~𝒟+​(xi+1)−ω~𝒟​(xi+1)−r~𝒟−​(xi)−ω~𝒟​(xi)]≤2​[r~𝒟−​(xi)+ω~𝒟​(xi)]+ϵ​ω​(𝒟)2​d≥r~𝒟−​(xi)+r~𝒟+​(xi)=2​[r~𝒟+​(xi)−ω~𝒟​(xi)]−[r~𝒟+​(xi)−ω~𝒟​(xi)−r~𝒟−​(xi)]+ω~𝒟​(xi)≥2​[r~𝒟+​(xi)−ω~𝒟​(xi)]−ϵ​ω​(𝒟)+02𝑑superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖2delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖italic-ϵ𝜔𝒟2𝑑superscriptsubscript~𝑟𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖2delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖2delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖italic-ϵ𝜔𝒟0\begin{split}2d&<\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})\\ &=2[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})]+[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})]\\ &\leq 2[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})]+\epsilon\omega(\mathcal{D})\\ 2d&\geq\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})\\ &=2[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})]-[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})]+\tilde{\omega}\_{\mathcal{D}}(x\_{i})\\ &\geq 2[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})]-\epsilon\omega(\mathcal{D})+0\end{split} |  |
* •

  x∗=xi+1superscript𝑥subscript𝑥𝑖1x^{\*}=x\_{i+1} in the general case

  |  |  |  |
  | --- | --- | --- |
  |  | 2​d≥r~𝒟−​(xi)+ω~𝒟​(xi)+r~𝒟+​(xi+1)−ω~𝒟​(xi+1)=2​[r~𝒟+​(xi+1)−ω~𝒟​(xi+1)]−[r~𝒟+​(xi+1)−ω~𝒟​(xi+1)−r~𝒟−​(xi)−ω~𝒟​(xi)]≥2​[r~𝒟+​(xi+1)+ω~𝒟​(xi+1)]−ϵ​ω​(𝒟)2​d≤r~𝒟−​(xi+1)+r~𝒟+​(xi+1)=2​[r~𝒟−​(xi+1)+ω~𝒟​(xi+1)]+[r~𝒟+​(xi+1)−ω~𝒟​(xi+1)−r~𝒟−​(xi+1)]−ω~𝒟​(xi+1)≤2​[r~𝒟−​(xi+1)+ω~𝒟​(xi+1)]+ϵ​ω​(𝒟)−0formulae-sequence2𝑑superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖subscript~𝜔𝒟subscript𝑥𝑖2delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1italic-ϵ𝜔𝒟2𝑑superscriptsubscript~𝑟𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖12delimited-[]superscriptsubscript~𝑟𝒟subscript𝑥𝑖1subscript~𝜔𝒟subscript𝑥𝑖1italic-ϵ𝜔𝒟0\begin{split}2d&\geq\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})+\tilde{\omega}\_{\mathcal{D}}(x\_{i})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})\\ &=2[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})]\\ &\ \ \ \ -[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i})-\tilde{\omega}\_{\mathcal{D}}(x\_{i})]\\ &\geq 2[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})+\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})]-\epsilon\omega(\mathcal{D})\\ 2d&\leq\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1})+\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})\\ &=2[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1})+\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})]\\ &\ \ \ \ +[\tilde{r}\_{\mathcal{D}}^{+}(x\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})-\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1})]-\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})\\ &\leq 2[\tilde{r}\_{\mathcal{D}}^{-}(x\_{i+1})+\tilde{\omega}\_{\mathcal{D}}(x\_{i+1})]+\epsilon\omega(\mathcal{D})-0\end{split} |  |

Now we are ready to introduce the prune operation. Given a quantile summary Q​(𝒟)=(S,r~𝒟+,r~𝒟−,ω~𝒟)𝑄𝒟𝑆superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q(\mathcal{D})=(S,\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) with S={x1,x2,⋯,xk}𝑆subscript𝑥1subscript𝑥2⋯subscript𝑥𝑘S=\{x\_{1},x\_{2},\cdots,x\_{k}\} elements, and a memory budget b𝑏b. The prune operation creates another summary Q′​(𝒟)=(S′,r~𝒟+,r~𝒟−,ω~𝒟)superscript𝑄′𝒟superscript𝑆′superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟Q^{\prime}(\mathcal{D})=(S^{\prime},\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}}) with S′={x1′,x2′,⋯,xb+1′}superscript𝑆′subscriptsuperscript𝑥′1subscriptsuperscript𝑥′2⋯subscriptsuperscript𝑥′𝑏1S^{\prime}=\{x^{\prime}\_{1},x^{\prime}\_{2},\cdots,x^{\prime}\_{b+1}\}, where xi′subscriptsuperscript𝑥′𝑖x^{\prime}\_{i} are selected by query the original summary such that

|  |  |  |
| --- | --- | --- |
|  | xi′=g​(Q,i−1b​ω​(𝒟)).subscriptsuperscript𝑥′𝑖𝑔𝑄𝑖1𝑏𝜔𝒟x^{\prime}\_{i}=g\left(Q,\frac{i-1}{b}\omega(\mathcal{D})\right). |  |

The definition of r~𝒟+,r~𝒟−,ω~𝒟

superscriptsubscript~𝑟𝒟superscriptsubscript~𝑟𝒟subscript~𝜔𝒟\tilde{r}\_{\mathcal{D}}^{+},\tilde{r}\_{\mathcal{D}}^{-},\tilde{\omega}\_{\mathcal{D}} in Q′superscript𝑄′Q^{\prime} is copied from original summary Q𝑄Q, by restricting input domain from S𝑆S to S′superscript𝑆′S^{\prime}.
There could be duplicated entries in the S′superscript𝑆′S^{\prime}. These duplicated entries can be safely removed to further reduce the memory cost.
Since all the elements in Q′superscript𝑄′Q^{\prime} comes from Q𝑄Q, we can verify that Q′superscript𝑄′Q^{\prime} satisfies all the constraints in Definition [A.1](#A1.Thmthmdef1 "Definition A.1 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System") and is a valid quantile summary.

###### Theorem A.2

Let Q′​(𝒟)superscript𝑄′𝒟Q^{\prime}(\mathcal{D}) be the summary pruned from an ϵitalic-ϵ\epsilon-approximate quantile summary Q​(𝒟)𝑄𝒟Q(\mathcal{D}) with b𝑏b memory budget.
Then Q′​(𝒟)superscript𝑄′𝒟Q^{\prime}(\mathcal{D}) is a (ϵ+1b)italic-ϵ1𝑏(\epsilon+\frac{1}{b})-approximate summary.

###### Proof A.5.

We only need to prove the property in Eq. ([23](#A1.E23 "In Lemma A.2 ‣ A.1 Formalization and Definitions ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System")) for Q′superscript𝑄′Q^{\prime}.
Using Lemma [A.4](#A1.Thmthmlemma4 "Lemma A.4 ‣ A.4 Prune Operation ‣ Appendix A Weighted Quantile Sketch ‣ XGBoost: A Scalable Tree Boosting System"), we have

|  |  |  |
| --- | --- | --- |
|  | i−1b​ω​(𝒟)+ϵ2​ω​(𝒟)≥r~𝒟+​(xi′)−ω~𝒟​(xi′)i−1b​ω​(𝒟)−ϵ2​ω​(𝒟)≤r~𝒟−​(xi′)+ω~𝒟​(xi′)𝑖1𝑏𝜔𝒟italic-ϵ2𝜔𝒟superscriptsubscript~𝑟𝒟subscriptsuperscript𝑥′𝑖subscript~𝜔𝒟subscriptsuperscript𝑥′𝑖𝑖1𝑏𝜔𝒟italic-ϵ2𝜔𝒟superscriptsubscript~𝑟𝒟subscriptsuperscript𝑥′𝑖subscript~𝜔𝒟subscriptsuperscript𝑥′𝑖\begin{split}\frac{i-1}{b}\omega(\mathcal{D})+\frac{\epsilon}{2}\omega(\mathcal{D})&\geq\tilde{r}\_{\mathcal{D}}^{+}(x^{\prime}\_{i})-\tilde{\omega}\_{\mathcal{D}}(x^{\prime}\_{i})\\ \frac{i-1}{b}\omega(\mathcal{D})-\frac{\epsilon}{2}\omega(\mathcal{D})&\leq\tilde{r}\_{\mathcal{D}}^{-}(x^{\prime}\_{i})+\tilde{\omega}\_{\mathcal{D}}(x^{\prime}\_{i})\end{split} |  |

Combining these inequalities gives

|  |  |  |
| --- | --- | --- |
|  | r~𝒟+​(xi+1′)−ω~𝒟​(xi+1′)−r~𝒟−​(xi′)−ω~𝒟​(xi′)≤[ib​ω​(𝒟)+ϵ2​ω​(𝒟)]−[i−1b​ω​(𝒟)−ϵ2​ω​(𝒟)]=(1b+ϵ)​ω​(𝒟)superscriptsubscript~𝑟𝒟subscriptsuperscript𝑥′𝑖1subscript~𝜔𝒟subscriptsuperscript𝑥′𝑖1superscriptsubscript~𝑟𝒟subscriptsuperscript𝑥′𝑖subscript~𝜔𝒟subscriptsuperscript𝑥′𝑖delimited-[]𝑖𝑏𝜔𝒟italic-ϵ2𝜔𝒟delimited-[]𝑖1𝑏𝜔𝒟italic-ϵ2𝜔𝒟1𝑏italic-ϵ𝜔𝒟\begin{split}&\tilde{r}\_{\mathcal{D}}^{+}(x^{\prime}\_{i+1})-\tilde{\omega}\_{\mathcal{D}}(x^{\prime}\_{i+1})-\tilde{r}\_{\mathcal{D}}^{-}(x^{\prime}\_{i})-\tilde{\omega}\_{\mathcal{D}}(x^{\prime}\_{i})\\ \leq&[\frac{i}{b}\omega(\mathcal{D})+\frac{\epsilon}{2}\omega(\mathcal{D})]-[\frac{i-1}{b}\omega(\mathcal{D})-\frac{\epsilon}{2}\omega(\mathcal{D})]=(\frac{1}{b}+\epsilon)\omega(\mathcal{D})\end{split} |  |

[◄](/html/1603.02753)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1603.02754)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1603.02754)
[View original  
on arXiv](https://arxiv.org/abs/1603.02754)[►](/html/1603.02755)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 8 16:17:03 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
