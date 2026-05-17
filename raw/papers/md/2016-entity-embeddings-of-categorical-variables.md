---
arxiv: '1604.06737'
authors:
- Cheng Guo
- Felix Berkhahn
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Entity Embeddings of Categorical Variables
url: http://arxiv.org/abs/1604.06737v1
year: 2016
---

# Entity Embeddings of Categorical Variables

Cheng Guo
[cheng.guo.work@gmail.com](mailto:cheng.guo.work@gmail.com)
  
Felix Berkhahn
[felix.berkhahn@gmail.com](mailto:felix.berkhahn@gmail.com)
Neokami Inc.

###### Abstract

We map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables.
The mapping is learned by a neural network during the standard supervised training process.
Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding,
but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables.
We applied it successfully in a recent Kaggle competition111<https://www.kaggle.com/c/rossmann-store-sales> and were able to reach the third position with relative simple features.
We further demonstrate in this paper that entity embedding helps the neural network to generalize better when the data is sparse and statistics is unknown.
Thus it is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit.
We also demonstrate that the embeddings obtained from the trained neural network boost the performance of all tested machine learning methods considerably when used as the input features instead. As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.

## I Introduction

Many advances have been achieved in the past 15 years in the field of neural networks due to a combination of faster computers, more data and better methods LeCun *et al.* ([2015](#bib.bib1)).
Neural networks revolutionized computer visionKrizhevsky *et al.* ([2012](#bib.bib2)); Zeiler and Fergus ([2014](#bib.bib3)); Simonyan and Zisserman ([2014](#bib.bib4)); Sermanet *et al.* ([2013](#bib.bib5)); Szegedy *et al.* ([2015](#bib.bib6)), speech recognitionHinton *et al.* ([2012](#bib.bib7)); Sainath *et al.* ([2013](#bib.bib8)) and natural language processingBengio *et al.* ([2003](#bib.bib9)); Mikolov *et al.* ([2011](#bib.bib10), ); Kim ([2014a](#bib.bib12)) and have replaced or are replacing the long dominating methods in each field.

Unlike in the above fields where data is unstructured, neural networks are not as prominent when dealing with machine learning problems with structured data.
This can be easily seen by the fact that the top teams in many online machine learning competitions like those hosted on Kaggle use tree based methods more often than neural networksChen and Guestrin ([2016](#bib.bib13)).

To understand this, we compared neural network and decision tree’s approach to the general machine learning problem, which is to approximate the function

|  |  |  |  |
| --- | --- | --- | --- |
|  | y=f​(x1,x2,…,xn).𝑦𝑓subscript𝑥1subscript𝑥2…subscript𝑥𝑛y=f(x\_{1},x\_{2},...,x\_{n}). |  | (1) |

Given a set of input values (x1,x2,…,xn)subscript𝑥1subscript𝑥2…subscript𝑥𝑛(x\_{1},x\_{2},...,x\_{n}) it generates the target output value y𝑦y.

In principle a neural network can approximate any continuous function[Cybenko](#bib.bib14) ; Nielsen ([2015](#bib.bib15)) and piece wise continuous function Llanas *et al.* ([2008](#bib.bib16)).
However, it is not suitable to approximate arbitrary non-continuous functions as it assumes certain level of continuity in its general form. During the training phase the continuity of the data guarantees the convergence of the optimization, and during the prediction phase it ensures that slightly changing the values of the input keeps the output stable.
On the other hand decision trees do not assume any continuity of the feature variables and can divide the states of a variable as fine as necessary.

Interestingly the problems we usually face in nature are often continuous if we use the right representation of data.
Whenever we find a better way to reveal the continuity of the data we increase the power of neural networks to learn the data.
For example, convolutional neural networks LeCun *et al.* ([1998](#bib.bib17)) group pixels in the same neighborhood together. This increases the continuity of the data
compared to simply representing the image as a flattened vector of all the pixel values of the images.
The rise of neural networks in natural language processing is based on the word embedding Bengio *et al.* ([2003](#bib.bib9)); [Mikolov *et al.*](#bib.bib11) ; Pennington *et al.* ([2014](#bib.bib18)) which puts words with similar meaning closer to each other in a word space thus increasing the continuity of the words compared to using one-hot encoding of words.

Unlike unstructured data found in nature, structured data with categorical features may not have continuity at all and even if it has it may not be so obvious. The continuous nature of neural networks limits their applicability to categorical variables.
Therefore, naively applying neural networks on structured data with integer representation for category variables does not work well.
A common way to circumvent this problem is to use one-hot encoding, but it has two shortcomings: First when we have many high cardinality features one-hot encoding often results in an unrealistic amount of computational resource requirement.
Second, it treats different values of categorical variables completely independent of each other and often ignores the informative relations between them.

In this paper we show how to use the entity embedding method to automatically learn the representation of categorical features in multi-dimensional spaces which puts values with similar effect in the function approximation problem Eq. ([1](#S1.E1 "In I Introduction ‣ Entity Embeddings of Categorical Variables")) close to each other, and thereby reveals the intrinsic continuity of the data and helps neural networks as well as other common machine learning algorithms to solve the problem.

Distributed representation of entities has been used in many contexts before[Hinton](#bib.bib19) ; Bengio and Bengio ([1999](#bib.bib20)); Hinton ([2002](#bib.bib21)).
Our main contributions are: First we explored this idea in the general function approximation problem and demonstrated its power in a large machine learning competition.
Second we studied the properties of the learned embeddings and showed how the embeddings can be used to understand and visualize categorical data.

## II Related Work

As far as we know the first domain where the entity embedding method in the context of neural networks has been explored is the representation of relational data[Hinton](#bib.bib19) . More recently, knowledge base which is a large collection of complex relational data is seeing lots of works using entity embeddingJenatton *et al.* ([2012](#bib.bib22)); Yang *et al.* ([2014](#bib.bib23)); Wu *et al.* ([2015](#bib.bib24)).
The basic data structure of relational data is triplets (h,r,t)ℎ𝑟𝑡(h,r,t), where hℎh and t𝑡t are two entities and r𝑟r is the relation.
The entities are mapped to vectors and relations are sometimes mapped to
a matrix(e.g. Linear Relation Embedding [Paccanaro and Hinton](#bib.bib25) ) or two matrices(e.g. Structured EmbeddingsBordes *et al.* ([2011](#bib.bib26)))
or a vector in the same embedding space as the entitiesBordes *et al.* ([2014](#bib.bib27)) etc.
Various kind of score function can be defined (see Table. 1 of He *et al.* ([2015](#bib.bib28))) to measure the likelihood of such a triplet, and the score function is used as the objective function for learning the embeddings.

In natural language processing, Word embeddings have been used to map words and phrases Bengio *et al.* ([2003](#bib.bib9)) into a continuous distributed vector in a semantic space.
In this space similar words are closer.
What is even more interesting is that not only the distance between words are meaningful but also the direction of the difference vectors. For example, it has been observed [Mikolov *et al.*](#bib.bib11)  that the learned word vectors have relations such as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐊𝐢𝐧𝐠−𝐌𝐚𝐧≈𝐐𝐮𝐞𝐞𝐧−𝐖𝐨𝐦𝐚𝐧𝐊𝐢𝐧𝐠𝐌𝐚𝐧𝐐𝐮𝐞𝐞𝐧𝐖𝐨𝐦𝐚𝐧\displaystyle\mathbf{King}-\mathbf{Man}\approx\mathbf{Queen}-\mathbf{Woman} |  | (2) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐏𝐚𝐫𝐢𝐬−𝐅𝐫𝐚𝐧𝐜𝐞≈𝐑𝐨𝐦𝐞−𝐈𝐭𝐚𝐥𝐲𝐏𝐚𝐫𝐢𝐬𝐅𝐫𝐚𝐧𝐜𝐞𝐑𝐨𝐦𝐞𝐈𝐭𝐚𝐥𝐲\displaystyle\mathbf{Paris}-\mathbf{France}\approx\mathbf{Rome}-\mathbf{Italy} |  | (3) |

There are many ways Bengio *et al.* ([2003](#bib.bib9)); [Mikolov *et al.*](#bib.bib11) ; Pennington *et al.* ([2014](#bib.bib18)); Levy and Goldberg ([2014](#bib.bib29)); Kim ([2014b](#bib.bib30)) to learn word embeddings. A very fast way Mikolov *et al.* ([2013](#bib.bib31)) is to use the word context with the aim to maximize

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(wc|w)=exp⁡(𝐰⋅𝐰𝐜)∑iexp⁡(𝐰⋅𝐰𝐢),𝑝conditionalsubscript𝑤𝑐𝑤⋅𝐰subscript𝐰𝐜subscript𝑖⋅𝐰subscript𝐰𝐢p(w\_{c}|w)=\frac{\exp(\mathbf{w}\cdot\mathbf{w\_{c}})}{\sum\_{i}\exp(\mathbf{w}\cdot\mathbf{w\_{i}})}, |  | (4) |

where 𝐰𝐰\mathbf{w} and 𝐰𝐜subscript𝐰𝐜\mathbf{w\_{c}} are the vector representation of a word w𝑤w and its neighbor word wcsubscript𝑤𝑐w\_{c} inside the context window while p​(wc|w)𝑝conditionalsubscript𝑤𝑐𝑤p(w\_{c}|w) is the probability to have wcsubscript𝑤𝑐w\_{c} in the context of w𝑤w.
The sum is over the whole vocabulary. Word embeddings can also be learned with supervised methods.
For example in Ref. Kim ([2014b](#bib.bib30)) the embeddings can be learned using text with labeled sentiment.
This approach is very close to the approach we use in this paper but in a different context.

## III Tree based methods

As tree based methods are the most widely used method for structured data and they are the main methods we are comparing to, we will briefly review them here.
Random Forests and in particular Gradient Boosted Trees have proven their capabilities in numerous recent Kaggle competitions Chen and Guestrin ([2016](#bib.bib13)).
In the following, we will briefly describe the process of growing a single decision tree used for regression, as well as two popular tree ensemble methods: random forests and gradient tree boosting.

### III.1 Single decision tree

Decision trees partition the feature space X𝑋X into M𝑀M different sub-spaces R1,R2,…​RM

subscript𝑅1subscript𝑅2…subscript𝑅𝑀R\_{1},R\_{2},\dots R\_{M}. The function f𝑓f in equation ([1](#S1.E1 "In I Introduction ‣ Entity Embeddings of Categorical Variables")) is thus modeled as

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=∑m=1Mcm​I​(x∈Rm)𝑓𝑥superscriptsubscript𝑚1𝑀subscript𝑐𝑚𝐼𝑥subscript𝑅𝑚f(x)=\sum\_{m=1}^{M}c\_{m}I(x\in R\_{m}) |  | (5) |

with I𝐼I being the indicator function
  
I​(x∈Rm)={1if ​x∈Rm0else𝐼𝑥subscript𝑅𝑚cases1if 𝑥subscript𝑅𝑚0elseI(x\in R\_{m})=\begin{cases}1&\text{if }x\in R\_{m}\\
0&\text{else}\end{cases}.
Using the common sum of squares

|  |  |  |  |
| --- | --- | --- | --- |
|  | L=∑i(yi−f​(xi))2𝐿subscript𝑖superscriptsubscript𝑦𝑖𝑓subscript𝑥𝑖2L=\sum\_{i}\left(y\_{i}-f(x\_{i})\right)^{2} |  | (6) |

as loss function, it follows from standard linear regression theory that, for given Rmsubscript𝑅𝑚R\_{m}, the optimal choices for the parameters cmsubscript𝑐𝑚c\_{m} are just the averages

|  |  |  |  |
| --- | --- | --- | --- |
|  | c^m=1|Rm|​∑xi∈Rmyisubscript^𝑐𝑚1subscript𝑅𝑚subscriptsubscript𝑥𝑖subscript𝑅𝑚subscript𝑦𝑖\hat{c}\_{m}=\frac{1}{|R\_{m}|}\sum\_{x\_{i}\in R\_{m}}y\_{i} |  | (7) |

with |Rm|subscript𝑅𝑚|R\_{m}| the number of elements in the set Rmsubscript𝑅𝑚R\_{m}. Ideally, we would try to find the optimal partition {Rm}subscript𝑅𝑚\{R\_{m}\} such as to minimize the loss function ([6](#S3.E6 "In III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")). However, this is not computationally feasible, as the number of possible partitions grows exponentially with the size of the feature space X𝑋X. Instead, a greedy algorithm is applied, that tries to find subsequent splits of X𝑋X that try to minimize ([6](#S3.E6 "In III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")) locally at each split. To start with, given a splitting variable j𝑗j and a split point s𝑠s, we define the pair of half-planes

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | R1​(j,s)subscript𝑅1𝑗𝑠\displaystyle R\_{1}(j,s) | =\displaystyle= | {X|Xj≤s}conditional-set𝑋subscript𝑋𝑗𝑠\displaystyle\{X|X\_{j}\leq s\} |  | (8) |
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | R2​(j,s)subscript𝑅2𝑗𝑠\displaystyle R\_{2}(j,s) | =\displaystyle= | {X|Xj>s}conditional-set𝑋subscript𝑋𝑗𝑠\displaystyle\{X|X\_{j}>s\} |  | (9) |

and optimize ([6](#S3.E6 "In III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")) for j𝑗j and s𝑠s:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minj,s⁡[∑xi∈R1​(j,s)(yi−c^1)2+∑xi∈R2​(j,s)(yi−c^2)2]subscript  𝑗𝑠subscriptsubscript𝑥𝑖subscript𝑅1𝑗𝑠superscriptsubscript𝑦𝑖subscript^𝑐12subscriptsubscript𝑥𝑖subscript𝑅2𝑗𝑠superscriptsubscript𝑦𝑖subscript^𝑐22\min\_{j,s}\left[\sum\_{x\_{i}\in R\_{1}(j,s)}(y\_{i}-\hat{c}\_{1})^{2}+\sum\_{x\_{i}\in R\_{2}(j,s)}(y\_{i}-\hat{c}\_{2})^{2}\right] |  | (10) |

The optimal choices for the parameters c^1subscript^𝑐1\hat{c}\_{1} and c^2subscript^𝑐2\hat{c}\_{2} follow directly from ([7](#S3.E7 "In III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")).

After ([10](#S3.E10 "In III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")) is solved for j𝑗j and s𝑠s, the same algorithm is applied recursively on the two half-planes R1subscript𝑅1R\_{1} and R2subscript𝑅2R\_{2} until the tree is fully grown.

The size up to which the tree is grown governs the complexity of the model and thus implies a bias-variance tradeoff: A very large tree likely overfits the training data, while a very small tree likely is not complex enough to capture the important dependencies in the data. There are several strategies and measures available to control the tree size. A very popular strategy is pruning, where first large trees are grown until they reach a minimal tree size (like minimum number of nodes or minimal height), and then internal nodes are collapsed (i.e. pruned) to minimize a cost-complexity measure Cαsubscript𝐶𝛼C\_{\alpha} such as

|  |  |  |  |
| --- | --- | --- | --- |
|  | Cα=∑i(yi−f​(xi))2+α​|T|subscript𝐶𝛼subscript𝑖superscriptsubscript𝑦𝑖𝑓subscript𝑥𝑖2𝛼𝑇C\_{\alpha}=\sum\_{i}(y\_{i}-f(x\_{i}))^{2}+\alpha|T| |  | (11) |

where |T|𝑇|T| is the number of terminal nodes in the tree T𝑇T and α𝛼\alpha is a free parameter to control the complexity of the model.

### III.2 Random forests

A single decision tree is a highly non-linear classifier with typically low bias but high variance. Random forests address the problem of high variance by establishing a committee (i.e. average) of identically distributed single decision trees.

To be precise, random forests contain N𝑁N single decision trees grown by the following algorithm:

1. 1.

   Draw a bootstrap sample from the training data, that is, select n𝑛n random records from the training data.
2. 2.

   Grow a single decision tree Tisubscript𝑇𝑖T\_{i} as described in section [III.1](#S3.SS1 "III.1 Single decision tree ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables"), with the only difference that at each split-node m𝑚m features are randomly picked that are considered for the best split at the split-node.
3. 3.

   Output the ensemble of all decision trees {Ti}i=1​…​Nsubscriptsubscript𝑇𝑖𝑖1…𝑁\{T\_{i}\}\_{i=1\dots N}.

For regression, an unseen sample is then predicted as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=1N​∑i=1NTi​(x)𝑓𝑥1𝑁superscriptsubscript𝑖1𝑁subscript𝑇𝑖𝑥f(x)=\frac{1}{N}\sum\_{i=1}^{N}T\_{i}(x) |  | (12) |

As all Tisubscript𝑇𝑖T\_{i} are identically distributed, the linear average of ([12](#S3.E12 "In III.2 Random forests ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")) preserves the presumably low bias of a single decision tree. However, averaging will reduce the variance of the single decision trees.

### III.3 Gradient boosted trees

Gradient tree boosting is another ensemble tree based method, that is we try to approximate f​(x)𝑓𝑥f(x) by a sum of trees Tisubscript𝑇𝑖T\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | f​(x)=∑k=1NTk​(x)𝑓𝑥superscriptsubscript𝑘1𝑁subscript𝑇𝑘𝑥f(x)=\sum\_{k=1}^{N}T\_{k}(x) |  | (13) |

For a generic loss function L𝐿L (not necessarily quadratic), the n𝑛n-th tree is grown on the quantity ri​nsubscript𝑟𝑖𝑛r\_{in}

|  |  |  |  |
| --- | --- | --- | --- |
|  | ri​n=−∂L​(yi,f​(xi))∂f​(xi)|f=fn−1subscript𝑟𝑖𝑛evaluated-at𝐿subscript𝑦𝑖𝑓subscript𝑥𝑖𝑓subscript𝑥𝑖𝑓subscript𝑓𝑛1r\_{in}=-\frac{\partial L(y\_{i},f(x\_{i}))}{\partial f(x\_{i})}\Bigr{|}\_{f=f\_{n-1}} |  | (14) |

computed using its n−1𝑛1n-1 predecessor trees. Here, the yisubscript𝑦𝑖y\_{i} are the target labels, xisubscript𝑥𝑖x\_{i} are the sample features and fn−1subscript𝑓𝑛1f\_{n-1} is the sum of the first n−1𝑛1n-1 trees

|  |  |  |  |
| --- | --- | --- | --- |
|  | fn−1​(x)=∑k=1n−1Tk​(x)subscript𝑓𝑛1𝑥superscriptsubscript𝑘1𝑛1subscript𝑇𝑘𝑥f\_{n-1}(x)=\sum\_{k=1}^{n-1}T\_{k}(x) |  | (15) |

In case of a squared error loss L=∑i(yi−f​(xi))2𝐿subscript𝑖superscriptsubscript𝑦𝑖𝑓subscript𝑥𝑖2L=\sum\_{i}\left(y\_{i}-f(x\_{i})\right)^{2} this amounts to fitting the n𝑛n-th tree on the residuals yi−fn−1​(xi)subscript𝑦𝑖subscript𝑓𝑛1subscript𝑥𝑖y\_{i}-f\_{n-1}(x\_{i}) of its n−1𝑛1n-1 predecessor trees. Hence, equation ([14](#S3.E14 "In III.3 Gradient boosted trees ‣ III Tree based methods ‣ Entity Embeddings of Categorical Variables")) generalizes to a generic loss function by minimizing the loss function L𝐿L iteratively at each step along the gradient descent direction in the space spanned by all possible trees Tnsubscript𝑇𝑛T\_{n}. This is where the name gradient boosted trees comes from.

As for every boosting algorithm, the next iterative classifier Tnsubscript𝑇𝑛T\_{n} tries to correct its Tn−1subscript𝑇𝑛1T\_{n-1} predecessors. Hence, in contrast to random forests, gradient tree boosting also aims to minimize the bias of the ensemble and not only the variance.

## IV Structured data

By structured data we mean data collected and organized in a table format with columns representing different features (variables) or target values and rows representing different samples. We focus on this type of data in this paper.

The most common variable types in structured data are continuous variables and discrete variables.
Continuous variables such as temperature, price, weight can be represented by real numbers.
Discrete variables such as age, color, bus line number can be represented by integers.
Often the integers are just used for convenience to label the different states and have no information in themselves.
For example if we use 1, 2, 3 to represent red, blue and yellow, one can not assume that ”blue is bigger than red” or ”the average of red and yellow are blue” or anything that introduces additional information based on the properties of integers.
These integers are called nominal numbers.
Other times there is an intrinsic ordering in the integer index such as age or month of the year.
These integers are called cardinal number or ordinal numbers. Note that the meaning or order may not be more useful for the problem than only considering the integer as nominal numbers.
For example the month ordering has nothing to do with number of days in a month (January is closer to Jun than February regarding number of days it has). Therefore we will treat both types of discrete variables in the same way. The task of entity embedding is to map discrete values to a multi-dimensional space where values with similar function output are close to each other.

## V Entity embedding

To learn the approximation of the function Eq. ([1](#S1.E1 "In I Introduction ‣ Entity Embeddings of Categorical Variables"))
we map each state of a discrete variable to a vector as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ei:xi↦𝐱i:subscript𝑒𝑖maps-tosubscript𝑥𝑖subscript𝐱𝑖e\_{i}:x\_{i}\mapsto\mathbf{x}\_{i} |  | (16) |

This mapping is equivalent to an extra layer of linear neurons on top of the one-hot encoded input as shown in Fig. [1](#S5.F1 "Figure 1 ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables").
To show this we represent one-hot encoding of xisubscript𝑥𝑖x\_{i} as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ui:xi↦δxi​α,:subscript𝑢𝑖maps-tosubscript𝑥𝑖subscript𝛿subscript𝑥𝑖𝛼u\_{i}:x\_{i}\mapsto\delta\_{x\_{i}\alpha}, |  | (17) |

where δxi​αsubscript𝛿subscript𝑥𝑖𝛼\delta\_{x\_{i}\alpha} is Kronecker delta and the possible values for α𝛼\alpha are the same as xisubscript𝑥𝑖x\_{i}.
If misubscript𝑚𝑖m\_{i} is the number of values for the categorical variable xisubscript𝑥𝑖x\_{i}, then δxi​αsubscript𝛿subscript𝑥𝑖𝛼\delta\_{x\_{i}\alpha} is a vector of length misubscript𝑚𝑖m\_{i}, where the element is only non-zero when α=xi𝛼subscript𝑥𝑖\alpha=x\_{i}.

The output of the extra layer of linear neurons given the input xisubscript𝑥𝑖x\_{i} is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐱i≡∑αwα​β​δxi​α=wxi​βsubscript𝐱𝑖subscript𝛼subscript𝑤𝛼𝛽subscript𝛿subscript𝑥𝑖𝛼subscript𝑤subscript𝑥𝑖𝛽\mathbf{x}\_{i}\equiv\sum\_{\alpha}w\_{\alpha\beta}\delta\_{x\_{i}\alpha}=w\_{x\_{i}\beta} |  | (18) |

where wα​βsubscript𝑤𝛼𝛽w\_{\alpha\beta} is the weight connecting the one-hot encoding layer to the embedding layer and
β𝛽\beta is the index of the embedding layer.
Now we can see that the mapped embeddings are just the weights of this layer and can be learned in the same way as the parameters of other neural network layers.

!(/html/1604.06737/assets/x1.png)

Figure 1: Illustration that entity embedding layers are equivalent to extra layers on top of each one-hot encoded input.

After we use entity embeddings to represent all categorical variables, all embedding layers and the input of all continuous variables (if any) are concatenated. The merged layer is treated like a normal input layer in neural networks and other layers can be build on top of it. The whole network can be trained with the standard back-propagation method. In this way, the entity embedding layer learns about the intrinsic properties of each category, while the deeper layers form complex combinations of them.

The dimensions of the embedding layers Disubscript𝐷𝑖D\_{i} are hyper-parameters that need to be pre-defined.
The bound of the dimensions of entity embeddings are between 1 and mi−1subscript𝑚𝑖1m\_{i}-1 where misubscript𝑚𝑖m\_{i} is the number of values for the categorical variable xisubscript𝑥𝑖x\_{i}. In practice we chose the dimensions based on experiments. The following empirical guidelines are used during this process: First, the more complex the more dimensions. We roughly estimated how many features/aspects one might need to describe the entities and used that as the dimension to start with. Second, if we had no clue about the first guideline, then we started with mi−1subscript𝑚𝑖1m\_{i}-1.

It would be good to have more theoretical guidelines on how to choose Disubscript𝐷𝑖D\_{i}. We think this probably relates to the problem of embedding of finite metric space, and that is what we want to explore next.

### V.1 Relation with embedding of finite metric space

With entity embedding we want to put similar values of a categorical variable closer to each
other in the embedding space.
If we use a real number to define similarity of the values then entity embedding is closely related to the
embedding of finite metric space problem in topology.

We define a finite metric space (Mi,di)subscript𝑀𝑖subscript𝑑𝑖(\mathit{M\_{i}},d\_{i}) associated with each categorical variable xisubscript𝑥𝑖x\_{i} in the function approximation problem Eq. ([1](#S1.E1 "In I Introduction ‣ Entity Embeddings of Categorical Variables")),
where Misubscript𝑀𝑖\mathit{M\_{i}} is the set of all possible values of xisubscript𝑥𝑖x\_{i}. disubscript𝑑𝑖d\_{i} is the metric on Misubscript𝑀𝑖\mathit{M\_{i}}, which is the distance function between any two pairs of values (xip,xiq)superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞(x\_{i}^{p},x\_{i}^{q}) of xisubscript𝑥𝑖x\_{i}.
We want disubscript𝑑𝑖d\_{i} to represent the similarity of (xip,xiq)superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞(x\_{i}^{p},x\_{i}^{q}). There are many ways to define it, one simple and natural way is

|  |  |  |  |
| --- | --- | --- | --- |
|  | di​(xip,xiq)=⟨|f​(xip,𝐱¯𝐢)−f​(xiq,𝐱¯𝐢)|⟩𝐱¯𝐢subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞subscriptdelimited-⟨⟩𝑓superscriptsubscript𝑥𝑖𝑝subscript¯𝐱𝐢𝑓superscriptsubscript𝑥𝑖𝑞subscript¯𝐱𝐢subscript¯𝐱𝐢d\_{i}(x\_{i}^{p},x\_{i}^{q})=\langle|f(x\_{i}^{p},\mathbf{\bar{x}\_{i}})-f(x\_{i}^{q},\mathbf{\bar{x}\_{i}})|\rangle\_{\mathbf{\bar{x}\_{i}}} |  | (19) |

where ⟨…⟩𝐱¯𝐢subscriptdelimited-⟨⟩…subscript¯𝐱𝐢\langle\dots\rangle\_{\mathbf{\bar{x}\_{i}}} is the average over all values of the parameters of f𝑓f other than xisubscript𝑥𝑖x\_{i}. 𝐱¯𝐢subscript¯𝐱𝐢\mathbf{\bar{x}\_{i}} is shorter notation for (x1,x2,…,xi−1,xi+1,…)subscript𝑥1subscript𝑥2…subscript𝑥𝑖1subscript𝑥𝑖1…(x\_{1},x\_{2},\dots,x\_{i-1},x\_{i+1},\dots).
It can be verified that the following conditions hold for the metric Eq. ([19](#S5.E19 "In V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | di​(xip,xiq)=0⇔xip=xiq⇔subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞0superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞\displaystyle d\_{i}(x\_{i}^{p},x\_{i}^{q})=0\Leftrightarrow x\_{i}^{p}=x\_{i}^{q} |  | (20) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | di​(xip,xiq)=di​(xiq,xip)subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑞superscriptsubscript𝑥𝑖𝑝\displaystyle d\_{i}(x\_{i}^{p},x\_{i}^{q})=d\_{i}(x\_{i}^{q},x\_{i}^{p}) |  | (21) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | di​(xip,xir)≤di​(xip,xiq)+di​(xiq,xir)subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑟subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑝superscriptsubscript𝑥𝑖𝑞subscript𝑑𝑖superscriptsubscript𝑥𝑖𝑞superscriptsubscript𝑥𝑖𝑟\displaystyle d\_{i}(x\_{i}^{p},x\_{i}^{r})\leq d\_{i}(x\_{i}^{p},x\_{i}^{q})+d\_{i}(x\_{i}^{q},x\_{i}^{r}) |  | (22) |

Eq. ([20](#S5.E20 "In V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables")) may not automatically hold in a real problem when two different values always generate the same output. However, this also means one value is redundant, and it is easy to simply merge these two values into one by redefining the categorical variable to make Eq. ([20](#S5.E20 "In V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables")) hold.

Ref. Schoenberg ([1938](#bib.bib32)) proved sufficient and necessary conditions to isometrically embed a generic metric space in an euclidean metric space. Applied on the metric Eq. ([19](#S5.E19 "In V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables")), it would require that the matrix

|  |  |  |  |
| --- | --- | --- | --- |
|  | (𝑴𝒊)p​q=e−λ​⟨|f​(xip,𝐱¯𝐢)−f​(xiq,𝐱¯𝐢)|⟩𝐱¯𝐢subscriptsubscript𝑴𝒊𝑝𝑞superscript𝑒𝜆subscriptdelimited-⟨⟩𝑓superscriptsubscript𝑥𝑖𝑝subscript¯𝐱𝐢𝑓superscriptsubscript𝑥𝑖𝑞subscript¯𝐱𝐢subscript¯𝐱𝐢(\bm{M\_{i}})\_{pq}=e^{-\lambda\langle|f(x\_{i}^{p},\mathbf{\bar{x}\_{i}})-f(x\_{i}^{q},\mathbf{\bar{x}\_{i}})|\rangle\_{\mathbf{\bar{x}\_{i}}}} |  | (23) |

is positive definite. We took the store feature (see Table [1](#S6.T1 "Table 1 ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables")) as an example and verified this numerically and found that it is not true. Therefore the store metric space as we defined cannot be isometrically embedded in an Euclidean space.

What is the relation of the learned embeddings of a categorical variable to this metric space? To answer this question we plot in Fig. [2](#S5.F2 "Figure 2 ‣ V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables") the distance between 10000 random store pairs in the learned store embedding space and in the metric space as defined in Eq. ([19](#S5.E19 "In V.1 Relation with embedding of finite metric space ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables")).
It is not an isometric embedding obviously. We can also see from the figure that there is a linear relation with well defined upper and lower boundary.
Why are there clear boundaries and what does the shape mean?
Is this related to some theorems regarding the distorted mapping of metric spaceIttai Abraham ([2006](#bib.bib33)); Matou ek ([1996](#bib.bib34))?
How is the distortion related to the embedding dimension Disubscript𝐷𝑖D\_{i}?
If we apply multidimensional scalingKruskal ([1964](#bib.bib35)) directly on the metric disubscript𝑑𝑖d\_{i} how is the result different to the learned entity embeddings of the neural network?
Due to time limit we will leave these interesting questions for future investigations.

!(/html/1604.06737/assets/x2.png)

Figure 2: Distance in the store embedding space versus distance in the metric space for 10000 random pair of stores.

## VI Experiments

In this paper we will use the dataset from the Kaggle Rossmann Sale Prediction competition as an example. The goal of the competition is to predict the daily sales of each store of Dirk Rossmann GmbH (abbreviated as ’Rossmann’ in the following) as accurate as possible.
The dataset published by the Rossmann hosts222<https://www.kaggle.com/c/rossmann-store-sales/data> has two parts: the first part is train.csv which comprises about 2.52.52.5 years of daily sales data for 111511151115 different Rossmann stores, resulting in a total number of 101721010172101017210 records; the second part is store.csv which describes some further details about each of these 1115 stores.

Besides the data published by the host, external data was also allowed as long as it was shared on the competition forum. Many features had been proposed by participants of this competition.
For example the Kaggle user dune\_dweller smartly figured out the German state each store belongs to by correlating the store open variable with the state holiday and school holiday calendar of the German states (state and school holidays differ in Germany from state to state)333<https://www.kaggle.com/c/rossmann-store-sales/forums/t/17048/putting-stores-on-the-map>. Other popular external data was weather data, Google Trends data and even sport events dates.

In our winning solution we used most of the above data, but in this paper the aim is to compare different machine learning methods and not to obtain the very best result. Therefore, to simplify, we use only a small subset of the features (see Table [1](#S6.T1 "Table 1 ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables")) and we do not apply any feature engineering.

| feature | data type | number of values | EE dimension |
| --- | --- | --- | --- |
| store | nominal | 1115 | 10 |
| day of week | ordinal | 7 | 6 |
| day | ordinal | 31 | 10 |
| month | ordinal | 12 | 6 |
| year | ordinal | 3 (2013-2015) | 2 |
| promotion | binary | 2 | 1 |
| state | nominal | 12 | 6 |

Table 1: Features we used from the Kaggle Rossmann competition dataset. promotion signals whether or not the store was issuing a promotion on the observation date. state corresponds to the German state where the store resides. The last column describes the dimension we used for each entity embedding (EE).

The dataset is divided into a 909090% portion for training, and a 101010% portion for testing. We consider both a split leaving the temporal structure of the data intact (i.e., using the first 909090% days for training), as well as a random shuffling of the dataset before the training-test split was applied.
For shuffled data, the test data shares the same statistical distribution as the training data.
More specifically, as the Rossmann dataset has relatively few features compared to the number of samples, the distribution of the test data in the feature space is well represented by the distribution of the training data.
The shuffled data is useful for us to benchmark model performance with respect to the pure statistical prediction accuracy.
For the time based splitting (i.e. unshuffled data), the test data is of a future time compared to the training data and the statistical distribution of the test data with respect to time is not exactly sampled by the training data. Therefore, it can measure the model’s generalization ability based on what it has learned from the training data.

The code used for this experiment can be found in this github repository444<https://github.com/entron/entity-embedding-rossmann>.

### VI.1 Neural networks

In this experiment we use both one-hot encoding and entity embedding to represent input features of neural networks. We use two fully connected layers (1000 and 500 neurons respectively) on top of either the embedding layer or directly on top of the one-hot encoding layer. The fully connected layer uses ReLU activation function. The output layer contains one neuron with sigmoid activation function. No dropout is used as we found that it did not improve the result.
We also experimented with a neural network where the entity embedding layer was replaced with an extra fully connected layer (on top of the one-hot encoding layer) of the same size as the sum of all entity embedding components but the result is worse than without this layer.
We use the deep learning framework Keras555<https://github.com/fchollet/keras>
to implement the neural network.

As S​a​l​e​s𝑆𝑎𝑙𝑒𝑠Sales in the data set spans 4 orders of magnitude, we used log⁡(S​a​l​e)𝑆𝑎𝑙𝑒\log(Sale) and rescaled it to the same range as the neural network output with log⁡(S​a​l​e)/log⁡(S​a​l​em​a​x)𝑆𝑎𝑙𝑒𝑆𝑎𝑙subscript𝑒𝑚𝑎𝑥\log(Sale)/\log(Sale\_{max}). Adam optimization methodKingma and Ba ([2014](#bib.bib36)) is used to optimize the networks. Each network is trained for 10 epochs. For prediction we use the average result of 5 neural networks, as an individual neural network showed notable variance.

### VI.2 Comparison of different methods

We compared k-nearest neighbors (KNN), random forests and gradient boosted trees with neural networks. KNN and random forests are tested using the scikit-learn library of python Pedregosa *et al.* ([2011](#bib.bib37)), while we use the xgboost implementation of gradient boosted trees Chen and Guestrin ([2016](#bib.bib13)). The used model parameters can be found in Table [2](#S6.T2 "Table 2 ‣ VI.2 Comparison of different methods ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables"). They were empirically found by optimizing the results of the validation set. For the input variables, KNN is fed with one-hot-encoded features, while random forests and gradient boosted trees use the integer coded categorical variables directly. We use log⁡(S​a​l​e​s)𝑆𝑎𝑙𝑒𝑠\log(Sales) as the target value for all machine learning methods.

As we are using relatively small number of features (7) compared to available training samples (about 1 million) the dataset is not sparse enough for our purpose. Therefore, we sparsified the training data by randomly sampling 200,000 samples out of the training set for benchmarking the models.

Instead of root mean square percentage error (RMSPE) used in the competition we use mean absolute percentage error (MAPE) as the criterion:

|  |  |  |  |
| --- | --- | --- | --- |
|  | M​A​P​E=⟨|S​a​l​e​s−S​a​l​e​sp​r​e​d​i​c​tS​a​l​e​s|⟩𝑀𝐴𝑃𝐸delimited-⟨⟩𝑆𝑎𝑙𝑒𝑠𝑆𝑎𝑙𝑒subscript𝑠𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑆𝑎𝑙𝑒𝑠MAPE=\left\langle\left\lvert\frac{Sales-Sales\_{predict}}{Sales}\right\rvert\right\rangle |  | (24) |

The reason is that we find MAPE is more stable with outliners, which may be caused by factors not included as features in the Rossmann dataset.

| xgboost | |
| --- | --- |
| max\_depth | 101010 |
| eta | 0.020.020.02 |
| objective | reg:linear |
| colsample\_bytree | 0.70.70.7 |
| subsample | 0.70.70.7 |
| num\_round | 300030003000 |
| random forest | |
| n\_estimators | 200200200 |
| max\_depth | 353535 |
| min\_samples\_split | 222 |
| min\_samples\_leaf | 111 |
| KNN | |
| n\_neighbors | 101010 |
| weights | distance |
| p | 111 |

Table 2: Parameters of models used to compare with neural networks. If a parameter is not specified, the default choice of scikit-learn (for random forests and KNN) and xgboost was taken.

The results that we obtained can be found in Table [3](#S6.T3 "Table 3 ‣ VI.2 Comparison of different methods ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables") and [4](#S6.T4 "Table 4 ‣ VI.2 Comparison of different methods ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables").
We can see that neural networks give the best results for non-shuffled data. For shuffled data, gradient boosted trees with entity embedding (see below for an explanation) and neural networks give comparable good results.
Neural networks with one-hot encoding give slightly better results than entity embedding for the shuffled data while entity embedding is clearly better than one-hot encoding for the non-shuffled data.
The explanation is that entity embedding, by restricting the network in a much smaller parameter space in a meaningful way, reduces the chance that the network converges to local minimums far from the global minimum.
More intuitively, entity embeddings force the network to learn the intrinsic properties of each of the feature as well as the sales distribution in the feature space.
One-hot encoding, on the other hand, only learns about the sales distribution. A better understanding of the intrinsic properties of the components (features) will give the model an advantage when facing a new combination of the components not seen during training. We expect this effect will be stronger when we add more features, for both shuffled and unshuffled data.

We also used the entity embeddings learned from a neural network as the input for other machine learning methods, that is, we feed the embedded features into other machine learning methods. This significantly improves all the methods tested here as shown in the right columns of the tables.

| method | MAPE | MAPE (with EE) |
| --- | --- | --- |
| KNN | 0.315 | 0.099 |
| random forest | 0.167 | 0.089 |
| gradient boosted trees | 0.122 | 0.071 |
| neural network | 0.070 | 0.070 |

Table 3: Comparison of different methods on the Kaggle Rossmann dataset with 10% shuffled data used for testing and 200,000 random samples from the remaining 90% for training.

| method | MAPE | MAPE (with EE) |
| --- | --- | --- |
| KNN | 0.290 | 0.116 |
| random forest | 0.158 | 0.108 |
| gradient boosted trees | 0.152 | 0.115 |
| neural network | 0.101 | 0.093 |

Table 4: Same as Table [4](#S6.T4 "Table 4 ‣ VI.2 Comparison of different methods ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables") except the data is not shuffled and the test data is the latest 10% of the data. This result shows the models generalization ability based on what they have learned from the training data.

### VI.3 Distribution in the embedding space

The main goal of entity embedding is to map similar categories close to each other in the embedding space. A natural question is thus how the embedding space and the distribution of the data within it look like. For the following analyses, we used a store embedding matrix of dimension 505050 and trained the network on the full first 90%percent9090\% of data, i.e. we did not apply data sparsification.

!(/html/1604.06737/assets/x3.png)

Figure 3: The learned German state embedding is mapped to a 2D space with t-SNE. The relative positions of German states here resemble that on the real German map surprisingly well.

To visualize the high dimensional embeddings we used t-SNEVan der Maaten and Hinton ([2008](#bib.bib38)) to map the embeddings to a 2D space.
Fig [3](#S6.F3 "Figure 3 ‣ VI.3 Distribution in the embedding space ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables") shows the result for the German state embeddings.
Though the algorithm does not know anything about German geography and society, the relative positions of the learned embedding of German states resemble that on the German map surprisingly well! The reason is that the embedding maps states with similar distribution of features, i.e. similar economical and cultural environments, close to each other, while at the same time two geographically neighboring states are likely sharing similar economy and culture. Especially, the three states on the right cluster, namely Sachsen, Thueringen and Sachsen Anhalt are all from eastern Germany while states in the left cluster are from western Germany. This shows the effectiveness of entity embedding for abductive reasoning. It also shows that entity embedding can be used to cluster categorical data. This is a consequence of entity embedding putting similar values close to each other in an euclidean space equipped with distance measure, on which any known clustering algorithm can be applied.

Regarding the sales distribution in entity embeddings, we take entity embedding of the store as an example. Figure [4](#S6.F4 "Figure 4 ‣ VI.3 Distribution in the embedding space ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables") shows the sales distribution in the store embedding along its first two principal components and along two random directions. It is apparent from the plot that the sales follows a continuous functional relationship along the first principal component. This allows the neural network to understand the impact of the store index, as stores with similar sales are mapped close to each other. Although the other directions in the subspace have no direct correlation with sales, they are encoding probably other properties of the store and when combined with other features in the deeper layers of the network they could have an impact on the final sales prediction.

!(/html/1604.06737/assets/pca_store_index_all_1.png)

!(/html/1604.06737/assets/pca_store_index_all_2.png)

!(/html/1604.06737/assets/random_store_index_all.png)

!(/html/1604.06737/assets/random_store_index_all_2.png)

Figure 4: Sales distribution along first principal component (upper left) and second principal component (upper right) of embedded store indices and along two random directions (lower left and right). All 111511151115 stores contributed to the plot.

The density distribution of store embedding is visualized in Fig. [5](#S6.F5 "Figure 5 ‣ VI.3 Distribution in the embedding space ‣ VI Experiments ‣ Entity Embeddings of Categorical Variables"), which shows the distribution along the first four principal components. Interestingly, the univariate density along the first principal components is approximately gaussian distributed. However, their joint distribution is not multivariate gaussian, as the Mardia test Mardia ([1970](#bib.bib39)) reveals.

!(/html/1604.06737/assets/pca_store_distribution_1.png)

!(/html/1604.06737/assets/pca_store_distribution_2.png)

!(/html/1604.06737/assets/pca_store_distribution_3.png)

!(/html/1604.06737/assets/pca_store_distribution_4.png)

Figure 5: Density distribution of embedded store indices along the first four principal components (from upper left to lower right). The red line corresponds to a gaussian fit. The p-values of the D’Agostino’s K2superscript𝐾2K^{2} normality test are all statistically significant, i.e. below 0.050.050.05.

As can be seen in Fig [1](#S5.F1 "Figure 1 ‣ V Entity embedding ‣ Entity Embeddings of Categorical Variables"), the neural network is fed with the direct product of all the entity embedding subspaces. We also investigated the statistical properties of this concatenated space. We found that there is no strong correlation between the individual subspaces. It is thus sufficient to consider them independently, as we did in this section.

## VII Future Work

Due to the limitation of time we leave the following points for future explorations:

First of all, entity embedding should be tested with more datasets, in particular datasets with many high cardinality features, where the data is getting sparse and entity embedding is supposed to show its full strength compared with other methods. For some datasets and entity embeddings it could also be interesting to explore the meaning of the directions in the embeddings like those in Eq. ([2](#S2.E2 "In II Related Work ‣ Entity Embeddings of Categorical Variables")) and Eq. ([3](#S2.E3 "In II Related Work ‣ Entity Embeddings of Categorical Variables")).

Second, we only touched the surface of the relation of entity embedding with the finite metric spaces. A deeper understanding of this relation might also help to find the optimal dimension of the embedding space and how neural networks work in general.

Third, similar methods may be applied to improve the approximation of continuous (i.e. non-categorical), but non-monotone functions. One way to achieve this is by discretizing the continuous variables and transform them into categorical variables as discussed in this paper.

Last, it might be interesting to systematically compare different activation functions of the entity embedding layer.

## VIII Acknowledge

We thank Dirk Rossmann GmbH to allow us to use their data for the publication. We thank Kaggle Inc. for hosting such an interesting competition.
We thank Gert Jacobusse for helpful discussions regarding xgboost.
We thank Neokami Inc. co-founders Ozel Christo and Andrei Ciobotar for their support joining the competition and writing this paper.

## References

* LeCun *et al.* (2015)
  Yann LeCun, Yoshua Bengio,
   and Geoffrey Hinton, “Deep learning,” [Nature 521, 436–444
  (2015)](http://dx.doi.org/10.1038/nature14539).
* Krizhevsky *et al.* (2012)
  Alex Krizhevsky, Ilya Sutskever,  and Geoffrey E. Hinton, “Imagenet classification with deep convolutional neural networks,” in [*Advances in neural information processing systems*](http://papers.nips.cc/paper/4824-imagenet-classification-w) (2012) pp. 1097–1105.
* Zeiler and Fergus (2014)
  Matthew D. Zeiler and Rob Fergus, “Visualizing and understanding convolutional networks,” in [*Computer Vision?ECCV 2014*](http://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) (Springer, 2014) pp. 818–833.
* Simonyan and Zisserman (2014)
  Karen Simonyan and Andrew Zisserman, “Very deep
  convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556  (2014).
* Sermanet *et al.* (2013)
  Pierre Sermanet, David Eigen,
  Xiang Zhang, Michaël Mathieu, Rob Fergus,  and Yann LeCun, “Overfeat: Integrated recognition, localization
  and detection using convolutional networks,” arXiv preprint arXiv:1312.6229  (2013).
* Szegedy *et al.* (2015)
  Christian Szegedy, Wei Liu, Yangqing Jia,
  Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke,  and Andrew Rabinovich, “Going deeper with convolutions,” in *Proceedings of the IEEE
  Conference on Computer Vision and Pattern Recognition* (2015) pp. 1–9.
* Hinton *et al.* (2012)
  Geoffrey Hinton, Li Deng, Dong Yu,
  George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, *et al.*, “Deep neural networks for
  acoustic modeling in speech recognition: The shared views of four research
  groups,” Signal
  Processing Magazine, IEEE 29, 82–97 (2012).
* Sainath *et al.* (2013)
  Tara N Sainath, Abdel-rahman Mohamed, Brian Kingsbury,  and Bhuvana Ramabhadran, “Deep
  convolutional neural networks for lvcsr,” in *Acoustics, Speech and Signal Processing (ICASSP), 2013
  IEEE International Conference on* (IEEE, 2013) pp. 8614–8618.
* Bengio *et al.* (2003)
  Yoshua Bengio, R jean Ducharme, Pascal Vincent,  and Christian Janvin, “A neural
  probabilistic language model,” [The Journal of Machine Learning Research 3, 1137–1155 (2003)](http://dl.acm.org/citation.cfm?id=944966).
* Mikolov *et al.* (2011)
  Tomas Mikolov, Anoop Deoras,
  Stefan Kombrink, Lukas Burget,  and Jan Cernockỳ, “Empirical evaluation and combination of
  advanced language modeling techniques.” in *INTERSPEECH*, s 1 (2011) pp. 605–608.
* (11)
  Tomas Mikolov, Kai Chen,
  Greg Corrado,  and Jeffrey Dean, “Efficient estimation of word
  representations in vector space,” .
* Kim (2014a)
  Yoon Kim, “Convolutional
  neural networks for sentence classification,” [arXiv preprint arXiv:1408.5882  (2014a)](http://arxiv.org/abs/1408.5882).
* Chen and Guestrin (2016)
  Tianqi Chen and Carlos Guestrin, “Xgboost: A scalable
  tree boosting system,”  (2016), [arXiv:1603.02754](http://arxiv.org/abs/arXiv:1603.02754) .
* (14)
  George Cybenko, “Approximation
  by superpositions of a sigmoidal function,” [2, 303–314](http://link.springer.com/article/10.1007/BF02551274).
* Nielsen (2015)
  Michael Nielsen, “Neural networks and deep
  learning,”  (Determination Press, 2015) Chap. 4.
* Llanas *et al.* (2008)
  Bernardo Llanas, Sagrario Lantarón,  and Francisco J Sáinz, “Constructive approximation of discontinuous functions by
  neural networks,” Neural Processing Letters 27, 209–226 (2008).
* LeCun *et al.* (1998)
  Yann LeCun, Léon Bottou, Yoshua Bengio,
   and Patrick Haffner, “Gradient-based learning
  applied to document recognition,” Proceedings of the IEEE 86, 2278–2324 (1998).
* Pennington *et al.* (2014)
  Jeffrey Pennington, Richard Socher,  and Christopher D. Manning, “Glove: Global vectors for word representation,” in [*Empirical Methods in Natural Language Processing (EMNLP)*](http://www.aclweb.org/anthology/D14-1162) (2014) pp. 1532–1543.
* (19)
  Geoffrey E. Hinton, “Learning distributed representations of concepts,” in [*Proceedings of the eighth annual conference of
  the cognitive science society*](http://www.cogsci.ucsd.edu/~ajyu/Teaching/Cogs202_sp13/Readings/hinton86.pdf), Vol. 1 (Amherst, MA) p. 12.
* Bengio and Bengio (1999)
  Yoshua Bengio and Samy Bengio, “Modeling
  high-dimensional discrete data with multi-layer neural networks.” in *NIPS*, Vol. 99 (1999) pp. 400–406.
* Hinton (2002)
  Alberto Paccanaro
  Geoffrey E Hinton, “Learning hierarchical structures with linear relational
  embedding,” in *Advances in
  Neural Information Processing Systems 14: Proceedings of the 2001
  Conference*, Vol. 2 (MIT
  Press, 2002) p. 857.
* Jenatton *et al.* (2012)
  Rodolphe Jenatton, Nicolas L Roux, Antoine Bordes,  and Guillaume R Obozinski, “A latent factor model for highly multi-relational data,” in *Advances in Neural
  Information Processing Systems* (2012) pp. 3167–3175.
* Yang *et al.* (2014)
  Bishan Yang, Wen-tau Yih,
  Xiaodong He, Jianfeng Gao,  and Li Deng, “Embedding entities and relations for learning
  and inference in knowledge bases,” arXiv preprint arXiv:1412.6575  (2014).
* Wu *et al.* (2015)
  Fei Wu, Jun Song, Yi Yang, Xi Li, Zhongfei Zhang,  and Yueting Zhuang, [“Structured embedding via pairwise relations and long-range
  interactions in knowledge base,”](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9342) (2015).
* (25)
  Alberto Paccanaro and Geoffrey E. Hinton, “Extracting distributed representations of concepts and relations from
  positive and negative propositions,” in [*Neural Networks, 2000. IJCNN 2000, Proceedings of
  the IEEE-INNS-ENNS International Joint Conference on*](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=857906), Vol. 2 (IEEE) pp. 259–264.
* Bordes *et al.* (2011)
  Antoine Bordes, Jason Weston,
  Ronan Collobert,  and Yoshua Bengio, “Learning structured
  embeddings of knowledge bases,” in *Conference on Artificial Intelligence*, EPFL-CONF-192344 (2011).
* Bordes *et al.* (2014)
  Antoine Bordes, Xavier Glorot,
  Jason Weston,  and Yoshua Bengio, “A semantic matching energy
  function for learning with multi-relational data,” Machine Learning 94, 233–259 (2014).
* He *et al.* (2015)
  Shizhu He, Kang Liu, Guoliang Ji,  and Jun Zhao, “Learning to represent knowledge graphs with
  gaussian embedding,” in *Proceedings of the 24th ACM International on Conference on Information and
  Knowledge Management* (ACM, 2015) pp. 623–632.
* Levy and Goldberg (2014)
  Omer Levy and Yoav Goldberg, “Neural word
  embedding as implicit matrix factorization,” in *Advances in Neural Information Processing Systems* (2014) pp. 2177–2185.
* Kim (2014b)
  Yoon Kim, “Convolutional
  neural networks for sentence classification,” arXiv preprint arXiv:1408.5882  (2014b).
* Mikolov *et al.* (2013)
  Tomas Mikolov, Ilya Sutskever, Kai Chen,
  Greg S. Corrado,  and Jeff Dean, “Distributed representations of words and
  phrases and their compositionality,” in [*Advances
  in neural information processing systems*](http://papers.nips.cc/paper/5021-di) (2013) pp. 3111–3119.
* Schoenberg (1938)
  Schoenberg, “Metric spaces and positive
  definite functions,” American Mathematical Society  (1938).
* Ittai Abraham (2006)
  Ofer Neiman Ittai Abraham, Yair Bartal, “On embedding of finite metric spaces
  into hilbert space,” Leibniz Center for Research in Computer Science  (2006).
* Matou ek (1996)
  Ji?  Matou ek, “On the
  distortion required for embedding finite metric spaces into normed spaces,” Isreal Journal of
  Mathematics , 333–344 (1996).
* Kruskal (1964)
  Joseph B Kruskal, “Multidimensional scaling by optimizing goodness of fit to a nonmetric
  hypothesis,” Psychometrika 29, 1–27
  (1964).
* Kingma and Ba (2014)
  Diederik P. Kingma and Jimmy Ba, “Adam:
  A method for stochastic optimization,” [CoRR abs/1412.6980 (2014)](http://arxiv.org/abs/1412.6980).
* Pedregosa *et al.* (2011)
  F. Pedregosa, G. Varoquaux, A. Gramfort,
  V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot,  and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning
  Research 12, 2825–2830
  (2011).
* Van der Maaten and Hinton (2008)
  Laurens Van der Maaten and Geoffrey Hinton, “Visualizing data using t-sne,” Journal of Machine Learning Research 9, 85 (2008).
* Mardia (1970)
  K.V. Mardia, “Measures of
  multivariate skewness and kurtosis with applications,” Biometrika 57, 519–530 (1970).
