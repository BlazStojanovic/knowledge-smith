---
arxiv: '2203.05556'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Artem Babenko
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: On Embeddings for Numerical Features in Tabular Deep Learning
url: http://arxiv.org/abs/2203.05556v4
year: 2022
---

# On Embeddings for Numerical Features in Tabular Deep Learning

Yury Gorishniy
  
Yandex
&Ivan Rubachev
  
HSE, Yandex
&Artem Babenko
  
Yandex
The first author: firstnamelastname@gmail.com

###### Abstract

Recently, Transformer-like deep architectures have shown strong performance on tabular data problems. Unlike traditional models, e.g., MLP, these architectures map scalar values of numerical features to high-dimensional embeddings before mixing them in the main backbone. In this work, we argue that embeddings for numerical features are an underexplored degree of freedom in tabular DL, which allows constructing more powerful DL models and competing with gradient boosted decision trees (GBDT) on some GBDT-friendly benchmarks (that is, where GBDT outperforms conventional DL models). We start by describing two conceptually different approaches to building embedding modules: the first one is based on a piecewise linear encoding of scalar values, and the second one utilizes periodic activations. Then, we empirically demonstrate that these two approaches can lead to significant performance boosts compared to the embeddings based on conventional blocks such as linear layers and ReLU activations. Importantly, we also show that embedding numerical features is beneficial for many backbones, not only for Transformers. Specifically, after proper embeddings, simple MLP-like models can perform on par with the attention-based architectures. Overall, we highlight embeddings for numerical features as an important design aspect with good potential for further improvements in tabular DL. The source code is available at <https://github.com/yandex-research/tabular-dl-num-embeddings>.

## 1 Introduction

Tabular data problems are currently a final frontier for deep learning (DL) research.
While the most recent breakthroughs in NLP, vision, and speech are achieved by deep models [[12](#bib.bib12)], their success in the tabular domain is not convincing yet.
Despite a large number of proposed architectures for tabular DL [[21](#bib.bib21), [31](#bib.bib31), [2](#bib.bib2), [40](#bib.bib40), [3](#bib.bib3), [17](#bib.bib17), [39](#bib.bib39), [13](#bib.bib13), [24](#bib.bib24)], the performance gap between them and the “shallow” ensembles of decision trees, like GBDT, often remains significant [[13](#bib.bib13), [36](#bib.bib36)].

The recent line of works [[13](#bib.bib13), [39](#bib.bib39), [24](#bib.bib24)] reduce this performance gap by successfully adapting the Transformer architecture [[45](#bib.bib45)] for the tabular domain.
Compared to traditional models, like MLP or ResNet, the proposed Transformer-like architectures have a specific way to handle numerical features of the data.
Namely, they map scalar values of numerical features to high-dimensional embedding vectors, which are then mixed by the self-attention modules.
Beyond transformers, mapping numerical features to vectors was also employed in different forms in the click-through rate (CTR) prediction problems [[8](#bib.bib8), [40](#bib.bib40), [14](#bib.bib14)].
Nevertheless, the literature is mostly focused on developing more powerful backbones while keeping the design of embedding modules relatively simple.
In particular, the existing architectures [[13](#bib.bib13), [39](#bib.bib39), [24](#bib.bib24), [40](#bib.bib40), [14](#bib.bib14)] construct embeddings for numerical features using quite restrictive parametric mappings, e.g., linear functions, which can lead to suboptimal performance.
In this work, we demonstrate that the embedding step has a substantial impact on the model effectiveness, and its proper design can significantly improve tabular DL models.

Specifically, we describe two different building blocks suitable for constructing embeddings for numerical features.
The first one is a piecewise linear encoding that produces alternative initial representations for the original scalar values and is based on feature binning, a long-existing preprocessing technique [[11](#bib.bib11)].
The second one relies on periodic activation functions, which is inspired by their usage in implicit neural representations [[28](#bib.bib28), [42](#bib.bib42), [38](#bib.bib38)], NLP [[45](#bib.bib45), [41](#bib.bib41)] and CV tasks [[25](#bib.bib25)].
The first approach is simple, interpretable and non-differentiable, while the second demonstrates better results on average.
We observe that DL models equipped with our embedding schemes successfully compete with GBDT on GBDT-friendly benchmarks and achieve the new state-of-the-art on tabular DL.

As another important finding, we demonstrate that the step of embedding the numerical features is universally beneficial for different deep architectures, not only for Transformer-like ones.
In particular, we show, that after proper embeddings, simple MLP-like architectures often provide the performance comparable to the state-of-the-art attention-based models.
Overall, our work demonstrates the large impact of the embeddings of numerical features on the tabular DL performance and shows the potential of investigating more advanced embedding schemes in future research.

To sum up, our contributions are as follows:

1. 1.

   We demonstrate that embedding schemes for numerical features are an underexplored research question in tabular DL. Namely, we show that more expressive embedding schemes can provide substantial performance improvements over prior models.
2. 2.

   We show that the profit from embedding numerical features is not specific for Transformer-like architectures, and proper embedding schemes benefit traditional models as well.
3. 3.

   On a number of public benchmarks, we achieve the new state-of-the-art on tabular DL.

## 2 Related work

Tabular deep learning.
During several recent years, the community has proposed a large number of deep models for tabular data [[21](#bib.bib21), [31](#bib.bib31), [2](#bib.bib2), [40](#bib.bib40), [46](#bib.bib46), [3](#bib.bib3), [15](#bib.bib15), [17](#bib.bib17), [39](#bib.bib39), [13](#bib.bib13), [24](#bib.bib24)].
However, when systematically evaluated, these models do not consistently outperform the ensembles of decision trees, such as GBDT (Gradient Boosting Decision Tree) [[7](#bib.bib7), [32](#bib.bib32), [19](#bib.bib19)], which are typically the top-choice in various ML competitions [[13](#bib.bib13), [36](#bib.bib36)].
Moreover, several recent works have shown that the proposed sophisticated architectures are not superior to properly tuned simple models, like MLP and ResNet [[13](#bib.bib13), [18](#bib.bib18)].
In this work, unlike the prior literature, we do not aim to propose a new backbone architecture.
Instead, we focus on more accurate ways to handle numerical features, and our developments can be potentially combined with any model, including traditional MLPs and more recent Transformer-like ones.

Transformers in tabular DL.
Due to the tremendous success of Transformers for different domains [[45](#bib.bib45), [10](#bib.bib10)], several recent works adapt their self-attention design for tabular DL as well [[17](#bib.bib17), [13](#bib.bib13), [39](#bib.bib39), [24](#bib.bib24)].
Compared to existing alternatives, applying self-attention modules to the numerical features of tabular data requires mapping the scalar values of these features to high-dimensional embedding vectors.
So far, the existing architectures perform this “scalar” →→\rightarrow “vector” mapping by relatively simple computational blocks, which, in practice, can limit the model expressiveness.
For instance, the recent FT-Transformer architecture [[13](#bib.bib13)] employs only a single linear layer.
In our experiments, we demonstrate that such embedding schemes can provide suboptimal performance, and more advanced schemes often lead to substantial profit.

CTR Prediction.
In CTR prediction problems, objects are represented by numerical and categorical features, which makes this field highly relevant to tabular data problems.
In several works, numerical features are handled in some non-trivial way while not being the central part of the research [[8](#bib.bib8), [40](#bib.bib40)].
Recently, however, a more advanced scheme has been proposed in Guo et al. [[14](#bib.bib14)].
Nevertheless, it is still based on linear layers and conventional activation functions, which we found to be suboptimal in our evaluation.

Feature binning.
Binning is a discretization technique that converts numerical features to categorical features.
Namely, for a given feature, its value range is split into bins (intervals), after which the original feature values are replaced with discrete descriptors (e.g. bin indices or one-hot vectors) of the corresponding bins.
We point to the work by Dougherty et al. [[11](#bib.bib11)], which performs an overview of some classic approaches to binning and can serve as an entry point to the relevant literature on the topic.
In our work, however, we utilize bins in a different way.
Specifically, we use their edges to construct lossless piecewise linear representations of the original scalar values.
It turns out that this simple and interpretable representations can provide substantial benefit to deep models on several tabular problems.

Periodic activations.
Recently, periodic activation functions have become a key component in processing coordinates-like inputs, which is required in many applications.
Examples include NLP [[45](#bib.bib45)], CV [[25](#bib.bib25)], implicit neural representations [[28](#bib.bib28), [42](#bib.bib42), [38](#bib.bib38)].
In our work, we show that periodic activations can be used to construct powerful embedding modules for numerical features in tabular data problems.
Contrary to some of the aforementioned papers, where components of the multidimensional coordinates are mixed (e.g. with linear layers) before passing them to periodic functions [[38](#bib.bib38), [42](#bib.bib42)], we find it crucial to embed each feature separately before mixing them in the main backbone.

## 3 Embeddings for numerical features

In this section, we describe the general framework for what we call "embeddings for numerical features" and the main building blocks used in the experimental comparison in [section 4](#S4 "4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

Notation. For a given supervised learning problem on tabular data, we denote the dataset as {(xj,yj)}j=1nsuperscriptsubscriptsuperscript𝑥𝑗superscript𝑦𝑗𝑗1𝑛\left\{\left(x^{j},\ y^{j}\right)\right\}\_{j=1}^{n} where yj∈𝕐superscript𝑦𝑗𝕐y^{j}\in\mathbb{Y} represents the object’s label and xj=(xj​(n​u​m),xj​(c​a​t))∈𝕏superscript𝑥𝑗superscript𝑥𝑗𝑛𝑢𝑚superscript𝑥𝑗𝑐𝑎𝑡𝕏x^{j}{=}\left(x^{j(num)},\ x^{j(cat)}\right)\in\mathbb{X} represents the object’s features (numerical and categorical).
xij​(n​u​m)superscriptsubscript𝑥𝑖𝑗𝑛𝑢𝑚x\_{i}^{j(num)}, in turn, denotes the i𝑖i-th numerical feature of the j𝑗j-th object.
Depending on the context, the j𝑗j index can be omitted.
The dataset is split into three disjoint parts: 1,n¯=Jt​r​a​i​n∪Jv​a​l∪Jt​e​s​t¯

1𝑛subscript𝐽𝑡𝑟𝑎𝑖𝑛subscript𝐽𝑣𝑎𝑙subscript𝐽𝑡𝑒𝑠𝑡\overline{1,n}=J\_{train}\cup J\_{val}\cup J\_{test}, where the “train” part is used for training, the “validation” part is used for early stopping and hyperparameter tuning, and the “test” part is used for the final evaluation.

### 3.1 General framework

We formalize the notion of "embeddings for numerical features" as zi=fi((xi(n​u​m))∈ℝdiz\_{i}=f\_{i}((x\_{i}^{(num)})\in\mathbb{R}^{d\_{i}}, where fi​(x)subscript𝑓𝑖𝑥f\_{i}(x) is the embedding function for the i𝑖i-th numerical feature, zisubscript𝑧𝑖z\_{i} is the embedding of the i𝑖i-th numerical feature and disubscript𝑑𝑖d\_{i} is the dimensionality of the embedding.
Importantly, the proposed framework implies that embeddings for all features are computed independently of each other.
Note that the function fisubscript𝑓𝑖f\_{i} can depend on parameters that are trained as a part of the whole model or in some other fashion (e.g. before the main optimization).
In this work, we consider only embedding schemes where the embedding functions for all features are of the same functional form.
We never share parameters of embedding functions of different features.

The subsequent use of the embeddings depends on the model backbone.
For MLP-like architectures, they are concatenated into one flat vector (see [Appendix A](#A1 "Appendix A MLP with embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") for illustrations).
For Transformer-based architectures, no extra step is performed and the embeddings are passed as is, so the usage is defined by the original architectures.

### 3.2 Piecewise linear encoding

While vanilla MLP is known to be a universal approximator [[9](#bib.bib9), [16](#bib.bib16)], in practice, due to optimization peculiarities, it has limitations in its learning capabilities [[34](#bib.bib34)].
However, the recent work by Tancik et al. [[42](#bib.bib42)] uncovers the case where changing the input space alleviates the above issue.
This observation motivates us to check if changing the representations of the original scalar values of numerical features can improve the learning capabilities of tabular DL models.

At this point, we try to start simple and turn to "classical" machine learning techniques. Namely, we take inspiration from the one-hot encoding algorithm that is widely and successfully used for representing discrete entities such as categorical features in tabular data problems or tokens in NLP.
We note that the one-hot representation can be seen as an opposite solution to the scalar representation in terms of the trade-off between parameter efficiency and expressivity.
To check whether the one-hot-like approach can be beneficial for tabular DL models, we design a continuous alternative to the one-hot encoding (since the vanilla one-hot encoding is barely applicable to numerical features).

Formally, for the i𝑖i-th numerical feature, we split its value range into the disjoint set of Tisuperscript𝑇𝑖T^{i} intervals B1i,…,BTi

superscriptsubscript𝐵1𝑖…superscriptsubscript𝐵𝑇𝑖B\_{1}^{i},\ \dots,\ B\_{T}^{i}, which we call bins: Bti=[bt−1i,bti)superscriptsubscript𝐵𝑡𝑖superscriptsubscript𝑏𝑡1𝑖superscriptsubscript𝑏𝑡𝑖B\_{t}^{i}=[b\_{t-1}^{i},b\_{t}^{i}).
The splitting algorithm is an important implementation detail that we discuss later.
From now on, we omit the feature index i𝑖i for simplicity.
Once the bins are determined, we define the encoding scheme as in [Equation 1](#S3.E1 "1 ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | PLE​(x)=[e1,…,eT]∈ℝTet={0,x​<bt−1​AND​t>​11,x≥bt​AND​t<Tx−bt−1bt−bt−1,otherwisePLE𝑥  subscript𝑒1…subscript𝑒𝑇superscriptℝ𝑇subscript𝑒𝑡cases0𝑥expectationsubscript𝑏𝑡1AND𝑡11𝑥subscript𝑏𝑡AND𝑡𝑇𝑥subscript𝑏𝑡1subscript𝑏𝑡subscript𝑏𝑡1otherwise\displaystyle\begin{split}&\texttt{PLE}(x)=[e\_{1},\ \dots,\ e\_{T}]\in\mathbb{R}^{T}\\ &e\_{t}=\begin{cases}0,&x<b\_{t-1}\ \texttt{AND}\ t>1\\ 1,&x\geq b\_{t}\ \texttt{AND}\ t<T\\ \frac{x-b\_{t-1}}{b\_{t}-b\_{t-1}},&\text{otherwise}\end{cases}\end{split} | |  | (1) |

where PLE stands for “peicewise linear encoding”. We provide the visualization in [Figure 1](#S3.F1 "Figure 1 ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

Figure 1: The piecewise linear encoding (PLE) in action for T=4𝑇4T=4 (see [Equation 1](#S3.E1 "1 ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).

Note that:

* •

  PLE produces alternative initial representations for the numerical features and can be viewed as a preprocessing strategy. These representations are computed once and then used instead of the original scalar values during the main optimization.
* •

  For T=1𝑇1T=1, the PLE-representation is effectively equivalent to the scalar representation.
* •

  Contrary to categorical features, numerical features are ordered; we express that by setting to 111 the components corresponding to bins with the right boundaries lower than the given feature value (this approach resembles how labels are encoded in ordinal regression problems).
* •

  The cases (x<b0)𝑥subscript𝑏0(x<b\_{0}) and (x≥bT)𝑥subscript𝑏𝑇(x\geq b\_{T}) are also covered by [Equation 1](#S3.E1 "1 ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") (which leads to (e1≤0)subscript𝑒10(e\_{1}\leq 0) and (eT≥1)subscript𝑒𝑇1(e\_{T}\geq 1) respectively).
* •

  The choice to make the representation piecewise linear is itself a subject for discussion. We analyze some alternatives in [subsection 5.2](#S5.SS2 "5.2 Ablation study ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
* •

  PLE can be viewed as feature preprocessing, which is additionally discussed in [subsection 5.3](#S5.SS3 "5.3 Piecewise linear encoding as a feature preprocessing technique ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

A note on attention-based models.
While the described PLE-representations can be passed to MLP-like models as is, attention-based models are inherently invariant to the order of input embeddings, so one additional step is required to add the information about feature indices to the obtained encodings.
Technically, we observe that it is enough to place one linear layer after PLE(without sharing weights between features).
Conceptually, however, this solution has a clear semantic interpretation.
Namely, it is equivalent to allocating one trainable embedding vt∈ℝdsubscript𝑣𝑡superscriptℝ𝑑v\_{t}\in\mathbb{R}^{d} for each bin Btsubscript𝐵𝑡B\_{t} and obtaining the final feature embedding by aggregating the embeddings of its bins with etsubscript𝑒𝑡e\_{t} as weights, plus bias v0subscript𝑣0v\_{0}. Formally: fi​(x)=v0+∑t=1Tet⋅vt=Linear​(PLE​(x))subscript𝑓𝑖𝑥subscript𝑣0superscriptsubscript𝑡1𝑇⋅subscript𝑒𝑡subscript𝑣𝑡LinearPLE𝑥f\_{i}\left(x\right)=v\_{0}+\sum\_{t=1}^{T}e\_{t}\cdot v\_{t}=\texttt{Linear}\left(\texttt{PLE}\left(x\right)\right).

In the following two sections, we describe two simple algorithms for building bins suitable for PLE.
Namely, we rely on the classic binning algorithms [[11](#bib.bib11)] and one of the two algorithms is unsupervised, while another one utilizes labels for constructing bins.

#### 3.2.1 Obtaining bins from quantiles

A natural baseline way to construct the bins for PLE is by splitting value ranges according to the uniformly chosen empirical quantiles of the corresponding individual feature distributions.
Formally, for the i𝑖i-th feature: bt=QtT​({xij​(n​u​m)}j∈Jt​r​a​i​n)subscript𝑏𝑡subscriptQ𝑡𝑇subscriptsuperscriptsubscript𝑥𝑖𝑗𝑛𝑢𝑚𝑗subscript𝐽𝑡𝑟𝑎𝑖𝑛b\_{t}=\texttt{Q}\_{\frac{t}{T}}\left(\{x\_{i}^{j(num)}\}\_{j\in J\_{train}}\right), where Q is the empirical quantile function.
Trivial bins of zero size are removed. In [subsection D.1](#A4.SS1 "D.1 Testing quantile-based PLE on the synthetic GBDT-friendly dataset ‣ Appendix D Additional analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), we demonstrate the usefulness of the proposed scheme on the synthetic GBDT-friendly dataset described in section 5.1 in Gorishniy et al. [[13](#bib.bib13)].

#### 3.2.2 Building target-aware bins

In fact, there are also supervised approaches that employ training labels for constructing bins [[11](#bib.bib11)].
Intuitively, such target-aware algorithms aim to produce bins that correspond to relatively narrow ranges of possible target values.
The supervised approach used in our work is identical in its spirit to the "C4.5 Discretization" algorithm from Kohavi and Sahami [[23](#bib.bib23)].
In a nutshell, for each feature, we recursively split its value range in a greedy manner using target as guidance, which is equivalent to building a decision tree (which uses for growing only this one feature and the target) and treating the regions corresponding to its leaves as the bins for PLE (see the illustration in [Figure 4](#A2.F4 "Figure 4 ‣ Appendix B Target-aware piecewise linear encoding ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
Additionally, we define b0i=minj∈Jt​r​a​i​n⁡xijsuperscriptsubscript𝑏0𝑖subscript𝑗subscript𝐽𝑡𝑟𝑎𝑖𝑛superscriptsubscript𝑥𝑖𝑗b\_{0}^{i}=\min\_{j\in J\_{train}}x\_{i}^{j} and bTi=maxj∈Jt​r​a​i​n⁡xijsuperscriptsubscript𝑏𝑇𝑖subscript𝑗subscript𝐽𝑡𝑟𝑎𝑖𝑛superscriptsubscript𝑥𝑖𝑗b\_{T}^{i}=\max\_{j\in J\_{train}}x\_{i}^{j}.

### 3.3 Periodic activation functions

Recall that in [subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") the work by Tancik et al. [[42](#bib.bib42)] was used as a starting point of our motivation for developing PLE.
Thus, we also try to adapt the original work itself for tabular data problems.
Our variation differs in two aspects.
First, we take into account the fact the embedding framework described in [subsection 3.1](#S3.SS1 "3.1 General framework ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") forbids mixing features during the embedding process (see [subsection D.2](#A4.SS2 "D.2 Fourier features ‣ Appendix D Additional analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning") for additional discussion).
Second, we train the pre-activation coefficients instead of keeping them fixed.
As a result, our approach is rather close to Li et al. [[25](#bib.bib25)] with the number of “groups” equal to the number of numerical features.
We formalize the described scheme in [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning"),

|  |  |  |  |
| --- | --- | --- | --- |
|  | fi​(x)=Periodic​(x)=concat​[sin⁡(v),cos⁡(v)],v=[2​π​c1​x,…, 2​π​ck​x]formulae-sequencesubscript𝑓𝑖𝑥Periodic𝑥concat𝑣𝑣𝑣  2𝜋subscript𝑐1𝑥…2𝜋subscript𝑐𝑘𝑥f\_{i}(x)=\texttt{Periodic}(x)=\texttt{concat}[\sin(v),\ \cos(v)],\qquad v=[2\pi c\_{1}x,\ \dots,\ 2\pi c\_{k}x] |  | (2) |

where cisubscript𝑐𝑖c\_{i} are trainable parameters initialized from 𝒩​(0,σ)𝒩0𝜎\mathcal{N}(0,\sigma).
We observe that σ𝜎\sigma is an important hyperparameter. Both σ𝜎\sigma and k𝑘k are tuned using validation sets.

### 3.4 Simple differentiable layers

In the context of Deep Learning, embedding numerical features with conventional differentiable layers (e.g. linear layers, ReLU activation, etc.) is a natural approach.
In fact, this technique is already used on its own in the recently proposed attention-based architectures [[13](#bib.bib13), [24](#bib.bib24), [39](#bib.bib39)] and in some models for CTR prediction problems [[40](#bib.bib40), [14](#bib.bib14)].
However, we also note that such conventional modules can be used on top of the components described in [subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 3.3](#S3.SS3 "3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
In [section 4](#S4 "4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), we find that such combinations often lead to better results.

## 4 Experiments

In this section, we empirically evaluate the techniques discussed in [section 3](#S3 "3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and compare them with Gradient Boosted Decision Trees to check the status quo of the “DL vs GBDT” competition.

### 4.1 Datasets

Table 1: Dataset properties. “RMSE” denotes root-mean-square error, “Acc.” denotes accuracy.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE | CH | CA | HO | AD | OT | HI | FB | SA | CO | MI |
| #objects | 9873 | 10000 | 20640 | 22784 | 48842 | 61878 | 98049 | 197080 | 200000 | 581012 | 1200192 |
| #num. features | 32 | 10 | 8 | 16 | 6 | 93 | 28 | 50 | 200 | 54 | 136 |
| #cat. features | 0 | 1 | 0 | 0 | 8 | 0 | 0 | 1 | 0 | 0 | 0 |
| metric | Acc. | Acc. | RMSE | RMSE | Acc. | Acc. | Acc. | RMSE | Acc. | Acc. | RMSE |
| #classes | 5 | 2 | – | – | 2 | 9 | 2 | – | 2 | 7 | – |
| majority class | 29% | 79% | – | – | 76% | 26% | 52% | – | 89% | 48% | – |

We use eleven public datasets mostly from the previous works on tabular DL and Kaggle competitions.
Importantly, we focus on the middle and large scale tasks, and our benchmark is biased towards GBDT-friendly problems, since, as of now, closing the gap with GBDT models on such tasks is one of the main challenges for tabular DL.
The main dataset properties are summarized in [Table 1](#S4.T1 "Table 1 ‣ 4.1 Datasets ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and the used sources and additional details are provided in [Appendix C](#A3 "Appendix C Additional details on datasets ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

### 4.2 Implementation details

We mostly follow Gorishniy et al. [[13](#bib.bib13)] in terms of the hyperparameter tuning, training and evaluation protocols.
Nevertheless, for completeness, we list all the details in [Appendix E](#A5 "Appendix E Implementation details ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
In the next paragraph, we describe the implementation details specific to embeddings for numerical features.

Embeddings for numerical features.
If linear layers are used, we tune their output dimensions.
The PLE hyperparameters are the same for all features.
For quantile-based PLE, we tune the number of quantiles.
For target-aware PLE, we tune the following parameters for decision trees: the maximum number of leaves, the minimum number of items per leaf, and the minimum information gain required for making a split when growing the tree.
For the Periodic module (see [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")), we tune σ𝜎\sigma and k𝑘k (these hyperparameters are the same for all features).

### 4.3 Model names

In the experiments, we consider different combinations of backbones and embeddings.
For convenience, we use the “Backbone-Embedding” pattern to name the models, where “Backbone” denotes the backbone (e.g. MLP, ResNet, Transformer) and “Embedding” denotes the embedding type.
See [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") for all considered embedding modules.
Note that:

* •

  Periodic is defined in [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
* •

  PLEqsubscriptPLEq\texttt{PLE}\_{\texttt{q}} denotes the quantile-based PLE. PLEtsubscriptPLEt\texttt{PLE}\_{\texttt{t}} denotes the target-aware PLE.
* •

  Linear−subscriptLinear\texttt{Linear}\_{-} denotes bias-free linear layer. LReLU denotes leaky ReLU. AutoDis was proposed in Guo et al. [[14](#bib.bib14)]
* •

  “Transformer-L” is equivalent to FT-Transformer [[13](#bib.bib13)].

Table 2: Embedding names. See [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning")

|  |  |
| --- | --- |
| Name | Embedding function (fisubscript𝑓𝑖f\_{i}) |
| L | Linear |
| LR | ReLU∘LinearReLULinear\texttt{ReLU}\circ\texttt{Linear} |
| LRLR | ReLU∘Linear∘ReLU∘LinearReLULinearReLULinear\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{ReLU}\circ\texttt{Linear} |
| Q | PLEqsubscriptPLEq\texttt{PLE}\_{\texttt{q}} |
| Q-L | Linear∘PLEqLinearsubscriptPLEq\texttt{Linear}\circ\texttt{PLE}\_{\texttt{q}} |
| Q-LR | ReLU∘Linear∘PLEqReLULinearsubscriptPLEq\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{PLE}\_{\texttt{q}} |
| Q-LRLR | ReLU∘Linear∘ReLU∘Linear∘PLEqReLULinearReLULinearsubscriptPLEq\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{PLE}\_{\texttt{q}} |
| T | PLEtsubscriptPLEt\texttt{PLE}\_{\texttt{t}} |
| T-L | Linear∘PLEtLinearsubscriptPLEt\texttt{Linear}\circ\texttt{PLE}\_{\texttt{t}} |
| T-LR | ReLU∘Linear∘PLEtReLULinearsubscriptPLEt\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{PLE}\_{\texttt{t}} |
| T-LRLR | ReLU∘Linear∘ReLU∘Linear∘PLEtReLULinearReLULinearsubscriptPLEt\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{PLE}\_{\texttt{t}} |
| P | Periodic |
| PL | Linear∘PeriodicLinearPeriodic\texttt{Linear}\circ\texttt{Periodic} |
| PLR | ReLU∘Linear∘PeriodicReLULinearPeriodic\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{Periodic} |
| PLRLR | ReLU∘Linear∘ReLU∘Linear∘PeriodicReLULinearReLULinearPeriodic\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{ReLU}\circ\texttt{Linear}\circ\texttt{Periodic} |
| AutoDis | Linear∘SoftMax∘Linear−∘LReLU∘Linear−LinearSoftMaxsubscriptLinearLReLUsubscriptLinear\texttt{Linear}\circ\texttt{SoftMax}\circ\texttt{Linear}\_{-}\circ\texttt{LReLU}\circ\texttt{Linear}\_{-} |

### 4.4 Simple differentiable embedding modules

Table 3: Results for MLP equipped with simple embedding modules (see [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
The metric values averaged over 15 random seeds are reported.
The standard deviations are provided in [Appendix F](#A6 "Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
We consider one result to be better than another if its mean score is better and its standard deviation is less than the difference.
For each dataset, top results are in bold.
Notation: ↓ corresponds to RMSE, ↑ corresponds to accuracy

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.6320.632\mathbf{0.632} | 0.8560.8560.856 | 0.4950.4950.495 | 3.2043.2043.204 | 0.8540.8540.854 | 0.8180.8180.818 | 0.7200.7200.720 | 5.6865.6865.686 | 0.9120.9120.912 | 0.9640.964\mathbf{0.964} | 0.7470.7470.747 |
| MLP-L | 0.6390.639\mathbf{0.639} | 0.8610.861\mathbf{0.861} | 0.4750.4750.475 | 3.1233.1233.123 | 0.8560.856\mathbf{0.856} | 0.8200.820\mathbf{0.820} | 0.7230.7230.723 | 5.6845.6845.684 | 0.9160.9160.916 | 0.9630.9630.963 | 0.7480.7480.748 |
| MLP-LR | 0.6420.642\mathbf{0.642} | 0.8600.860\mathbf{0.860} | 0.4710.471\mathbf{0.471} | 3.0843.084\mathbf{3.084} | 0.8570.857\mathbf{0.857} | 0.8190.819\mathbf{0.819} | 0.7260.726\mathbf{0.726} | 5.6255.625\mathbf{5.625} | 0.9230.923\mathbf{0.923} | 0.9630.9630.963 | 0.7460.746\mathbf{0.746} |

We start by evaluating embedding modules consisting of “conventional” differentiable layers (linear layers, ReLU activations, etc.).
The results are summarized in [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
  
The main takeaways:

* •

  first and foremost, the results indicate that MLP can benefit from embedding modules. Thus, we conclude that this backbone is worth attention when it comes to evaluating embedding modules.
* •

  the simple LR module leads to modest, but consistent improvements when applied to MLP.

Interestingly, the “redundant” MLP-L configuration also tends to outperform the vanilla MLP.
Although the improvements are not dramatic, the special property of this architecture is that the linear embedding module can be fused together with the first linear layer of MLP after training, which completely removes the overhead.
As for LRLR and AutoDis, we observe that these heavy modules do not justify the extra costs (see the results in [Appendix F](#A6 "Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).

### 4.5 Piecewise linear encoding

In this section, we evaluate the encoding scheme described in [subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The results are summarized in [Table 4](#S4.T4 "Table 4 ‣ 4.5 Piecewise linear encoding ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
  
The main takeaways:

* •

  The piecewise linear encoding is often beneficial for both types of architectures (MLP and Transformer) and the profit can be significant (for example, see the CA and AD datasets).
* •

  Adding differentiable components on top of the PLE can improve the performance. Though, the most expensive modifications such as Q-LRLR and T-LRLR are not worth it (see [Appendix F](#A6 "Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).

Note that the benchmark is biased towards GBDT-friendly problems, so the typical superiority of tree-based bins over quantile-based bins, which can be observed in [Table 4](#S4.T4 "Table 4 ‣ 4.5 Piecewise linear encoding ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), may not generalize to more DL-friendly datasets.
Thus, we do not make any general claims about the relative advantages of the two schemes here.

Table 4: 
Results for MLP and Transformer with embedding modules based on the piecewise linear encoding ([subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
Notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The best results are defined separately for the MLP and Transformer backbones.

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.6320.6320.632 | 0.8560.8560.856 | 0.4950.4950.495 | 3.2043.2043.204 | 0.8540.8540.854 | 0.8180.8180.818 | 0.7200.7200.720 | 5.6865.6865.686 | 0.9120.9120.912 | 0.9640.9640.964 | 0.7470.747\mathbf{0.747} |
| MLP-Q | 0.6530.653\mathbf{0.653} | 0.8540.8540.854 | 0.4640.4640.464 | 3.1633.163\mathbf{3.163} | 0.8590.8590.859 | 0.8160.8160.816 | 0.7210.7210.721 | 5.7665.7665.766 | 0.9220.9220.922 | 0.9680.9680.968 | 0.7500.7500.750 |
| MLP-T | 0.6470.647\mathbf{0.647} | 0.8610.861\mathbf{0.861} | 0.4470.4470.447 | 3.1493.149\mathbf{3.149} | 0.8640.8640.864 | 0.8210.821\mathbf{0.821} | 0.7200.7200.720 | 5.5775.5775.577 | 0.9230.9230.923 | 0.9670.9670.967 | 0.7490.7490.749 |
| MLP-Q-LR | 0.6460.646\mathbf{0.646} | 0.8570.8570.857 | 0.4550.4550.455 | 3.1843.184\mathbf{3.184} | 0.8630.8630.863 | 0.8110.8110.811 | 0.7200.7200.720 | 5.3945.394\mathbf{5.394} | 0.9230.9230.923 | 0.9690.969\mathbf{0.969} | 0.7470.747\mathbf{0.747} |
| MLP-T-LR | 0.6400.6400.640 | 0.8610.861\mathbf{0.861} | 0.4390.439\mathbf{0.439} | 3.2073.2073.207 | 0.8680.868\mathbf{0.868} | 0.8180.8180.818 | 0.7240.724\mathbf{0.724} | 5.5085.508\mathbf{5.508} | 0.9240.924\mathbf{0.924} | 0.9680.968\mathbf{0.968} | 0.7470.7470.747 |
| Transformer-L | 0.6320.6320.632 | 0.8600.8600.860 | 0.4650.4650.465 | 3.2393.2393.239 | 0.8580.8580.858 | 0.8170.817\mathbf{0.817} | 0.7250.7250.725 | 5.6025.602\mathbf{5.602} | 0.9240.9240.924 | 0.9710.9710.971 | 0.7460.746\mathbf{0.746} |
| Transformer-Q-L | 0.6590.659\mathbf{0.659} | 0.8560.8560.856 | 0.4510.4510.451 | 3.3193.3193.319 | 0.8670.8670.867 | 0.8120.8120.812 | 0.7290.729\mathbf{0.729} | 5.7415.7415.741 | 0.9240.924\mathbf{0.924} | 0.9730.973\mathbf{0.973} | 0.7470.7470.747 |
| Transformer-T-L | 0.6630.663\mathbf{0.663} | 0.8610.861\mathbf{0.861} | 0.4540.4540.454 | 3.1973.197\mathbf{3.197} | 0.8710.871\mathbf{0.871} | 0.8170.817\mathbf{0.817} | 0.7260.7260.726 | 5.8035.8035.803 | 0.9240.924\mathbf{0.924} | 0.9740.974\mathbf{0.974} | 0.7470.7470.747 |
| Transformer-Q-LR | 0.6590.659\mathbf{0.659} | 0.8570.8570.857 | 0.4480.4480.448 | 3.2703.2703.270 | 0.8670.8670.867 | 0.8120.8120.812 | 0.7230.7230.723 | 5.6835.6835.683 | 0.9230.9230.923 | 0.9720.9720.972 | 0.7480.7480.748 |
| Transformer-T-LR | 0.6650.665\mathbf{0.665} | 0.8600.8600.860 | 0.4420.442\mathbf{0.442} | 3.2193.219\mathbf{3.219} | 0.8700.8700.870 | 0.8180.818\mathbf{0.818} | 0.7290.729\mathbf{0.729} | 5.6995.6995.699 | 0.9240.924\mathbf{0.924} | 0.9730.9730.973 | 0.7470.7470.747 |

### 4.6 Periodic activation functions

Table 5: 
Results for MLP and Transformer with embedding modules based on periodic activations ([subsection 3.3](#S3.SS3 "3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
Notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The best results are defined separately for the MLP and Transformer backbones.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| MLP | 0.6320.6320.632 | 0.8560.8560.856 | 0.4950.4950.495 | 3.2043.2043.204 | 0.8540.8540.854 | 0.8180.818\mathbf{0.818} | 0.7200.7200.720 | 5.6865.6865.686 | 0.9120.9120.912 | 0.9640.9640.964 | 0.7470.7470.747 |
| MLP-P | 0.6310.6310.631 | 0.8600.860\mathbf{0.860} | 0.4890.4890.489 | 3.1293.1293.129 | 0.8690.8690.869 | 0.8070.8070.807 | 0.7230.7230.723 | 5.8455.8455.845 | 0.9230.9230.923 | 0.9680.9680.968 | 0.7470.7470.747 |
| MLP-PL | 0.6410.6410.641 | 0.8590.859\mathbf{0.859} | 0.4670.467\mathbf{0.467} | 3.1133.1133.113 | 0.8680.8680.868 | 0.8190.819\mathbf{0.819} | 0.7270.7270.727 | 5.5305.530\mathbf{5.530} | 0.9240.9240.924 | 0.9690.969\mathbf{0.969} | 0.7460.7460.746 |
| MLP-PLR | 0.6740.674\mathbf{0.674} | 0.8570.857\mathbf{0.857} | 0.4670.467\mathbf{0.467} | 3.0503.050\mathbf{3.050} | 0.8700.870\mathbf{0.870} | 0.8190.819\mathbf{0.819} | 0.7280.728\mathbf{0.728} | 5.5255.525\mathbf{5.525} | 0.9240.924\mathbf{0.924} | 0.9700.970\mathbf{0.970} | 0.7460.746\mathbf{0.746} |
| Transformer-L | 0.6320.632\mathbf{0.632} | 0.8600.8600.860 | 0.4650.465\mathbf{0.465} | 3.2393.2393.239 | 0.8580.8580.858 | 0.8170.817\mathbf{0.817} | 0.7250.7250.725 | 5.6025.602\mathbf{5.602} | 0.9240.924\mathbf{0.924} | 0.9710.971\mathbf{0.971} | 0.7460.746\mathbf{0.746} |
| Transformer-PLR | 0.6460.646\mathbf{0.646} | 0.8630.863\mathbf{0.863} | 0.4640.464\mathbf{0.464} | 3.1623.162\mathbf{3.162} | 0.8700.870\mathbf{0.870} | 0.8140.8140.814 | 0.7300.730\mathbf{0.730} | 5.7605.7605.760 | 0.9240.924\mathbf{0.924} | 0.9720.972\mathbf{0.972} | 0.7460.746\mathbf{0.746} |

In this section, we evaluate embedding modules based on periodic activation functions as described in [subsection 3.3](#S3.SS3 "3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The results are reported in [Table 5](#S4.T5 "Table 5 ‣ 4.6 Periodic activation functions ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
  
The main takeaway: on average, MLP-P is superior to the vanilla MLP. However, adding a differentiable component on top of the Periodic module should be the default strategy (which is in line with Li et al. [[25](#bib.bib25)]).
Indeed, MLP-PLR and MLP-PL provide meaningful improvements over MLP-P (e.g. see GE, CA, HO) and even “fix” MLP-P where it is inferior to MLP (OT, FB).

Although MLP-PLR is usually superior to MLP-PL, we note that in the latter case the last linear layer of the embedding module is “redundant” in terms of expressivity and can be fused with the first linear layer of the backbone after training, which, in theory, can lead to a more lightweight model.
Finally, we observe that MLP-PLRLR and MLP-PLR do not differ significantly enough to justify the extra cost of the PLRLR module (see [Appendix F](#A6 "Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).

### 4.7 Comparing DL models and GBDT

In this section, we perform a big comparison of different approaches to identify the best embedding modules and backbones, as well as to check if embeddings for numerical features allow DL models to compete with GBDT on more tasks than before.
Importantly, we compare ensembles of DL models against ensembles of GBDT, since Gradient Boosting is essentially an ensembling technique, so such comparison will be fairer.
Note that we focus only on the best metric values without taking efficiency into account, so we only check if DL models are conceptually ready to compete with GBDT.

We consider three backbones: MLP, ResNet, and Transformer, since they are reported to be representative of what baseline DL backbones are currently capable of [[13](#bib.bib13), [18](#bib.bib18), [39](#bib.bib39), [24](#bib.bib24)].
Note that we do not include the attention-based models that also apply attention on the level of objects [[39](#bib.bib39), [24](#bib.bib24), [35](#bib.bib35)], since this non-parametric component is orthogonal to the central topic of our work.
The results are summarized in [Table 6](#S4.T6 "Table 6 ‣ 4.7 Comparing DL models and GBDT ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

Table 6: Results for ensembles of GBDT, the baseline DL models and their modifications using different types of embeddings for numerical features. Notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning"). Due to the limited precision, some different values are represented with the same figures.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ | Avg. Rank |
| CatBoost | 0.6920.6920.692 | 0.8610.8610.861 | 0.4300.4300.430 | 3.0933.0933.093 | 0.8730.8730.873 | 0.8250.8250.825 | 0.7270.7270.727 | 5.2265.2265.226 | 0.9240.9240.924 | 0.9670.9670.967 | 0.7410.741\mathbf{0.741} | 3.6±2.9plus-or-minus3.62.93.6\pm 2.9 |
| XGBoost | 0.6830.6830.683 | 0.8590.8590.859 | 0.4340.4340.434 | 3.1523.1523.152 | 0.8750.875\mathbf{0.875} | 0.8270.8270.827 | 0.7260.7260.726 | 5.3385.3385.338 | 0.9190.9190.919 | 0.9690.9690.969 | 0.7420.7420.742 | 4.6±2.7plus-or-minus4.62.74.6\pm 2.7 |
| MLP | 0.6650.6650.665 | 0.8560.8560.856 | 0.4860.4860.486 | 3.1093.1093.109 | 0.8560.8560.856 | 0.8220.8220.822 | 0.7270.7270.727 | 5.6165.6165.616 | 0.9130.9130.913 | 0.9680.9680.968 | 0.7460.7460.746 | 8.5±2.6plus-or-minus8.52.68.5\pm 2.6 |
| MLP-LR | 0.6790.6790.679 | 0.8610.8610.861 | 0.4630.4630.463 | 3.0123.0123.012 | 0.8590.8590.859 | 0.8260.8260.826 | 0.7310.7310.731 | 5.4775.4775.477 | 0.9240.9240.924 | 0.9720.9720.972 | 0.7440.7440.744 | 5.5±2.7plus-or-minus5.52.75.5\pm 2.7 |
| MLP-Q-LR | 0.6820.6820.682 | 0.8590.8590.859 | 0.4330.4330.433 | 3.0803.0803.080 | 0.8670.8670.867 | 0.8180.8180.818 | 0.7240.7240.724 | 5.1445.144\mathbf{5.144} | 0.9240.9240.924 | 0.9740.9740.974 | 0.7450.7450.745 | 5.1±1.9plus-or-minus5.11.95.1\pm 1.9 |
| MLP-T-LR | 0.6730.6730.673 | 0.8610.8610.861 | 0.4350.4350.435 | 3.0993.0993.099 | 0.8700.8700.870 | 0.8210.8210.821 | 0.7270.7270.727 | 5.4095.4095.409 | 0.9240.9240.924 | 0.9730.9730.973 | 0.7460.7460.746 | 5.1±1.7plus-or-minus5.11.75.1\pm 1.7 |
| MLP-PLR | 0.7000.700\mathbf{0.700} | 0.8580.8580.858 | 0.4530.4530.453 | 2.9752.975\mathbf{2.975} | 0.8740.8740.874 | 0.8300.830\mathbf{0.830} | 0.7340.734\mathbf{0.734} | 5.3885.3885.388 | 0.9240.924\mathbf{0.924} | 0.9750.9750.975 | 0.7430.7430.743 | 3.0±2.4plus-or-minus3.02.43.0\pm 2.4 |
| ResNet | 0.6900.6900.690 | 0.8610.8610.861 | 0.4830.4830.483 | 3.0813.0813.081 | 0.8560.8560.856 | 0.8210.8210.821 | 0.7340.7340.734 | 5.4825.4825.482 | 0.9180.9180.918 | 0.9680.9680.968 | 0.7450.7450.745 | 6.7±3.3plus-or-minus6.73.36.7\pm 3.3 |
| ResNet-LR | 0.6720.6720.672 | 0.8620.8620.862 | 0.4500.4500.450 | 2.9922.9922.992 | 0.8590.8590.859 | 0.8220.8220.822 | 0.7330.7330.733 | 5.4155.4155.415 | 0.9230.9230.923 | 0.9710.9710.971 | 0.7430.7430.743 | 5.6±2.7plus-or-minus5.62.75.6\pm 2.7 |
| ResNet-Q-LR | 0.6740.6740.674 | 0.8590.8590.859 | 0.4270.4270.427 | 3.0663.0663.066 | 0.8680.8680.868 | 0.8150.8150.815 | 0.7290.7290.729 | 5.3095.3095.309 | 0.9230.9230.923 | 0.9760.9760.976 | 0.7460.7460.746 | 4.7±2.0plus-or-minus4.72.04.7\pm 2.0 |
| ResNet-T-LR | 0.6830.6830.683 | 0.8620.8620.862 | 0.4250.425\mathbf{0.425} | 3.0303.0303.030 | 0.8720.8720.872 | 0.8220.8220.822 | 0.7310.7310.731 | 5.4715.4715.471 | 0.9230.9230.923 | 0.9750.9750.975 | 0.7440.7440.744 | 4.1±1.9plus-or-minus4.11.94.1\pm 1.9 |
| ResNet-PLR | 0.6910.6910.691 | 0.8610.8610.861 | 0.4430.4430.443 | 3.0403.0403.040 | 0.8740.874\mathbf{0.874} | 0.8250.8250.825 | 0.7340.7340.734 | 5.4005.4005.400 | 0.9240.9240.924 | 0.9750.9750.975 | 0.7430.7430.743 | 3.2±1.3plus-or-minus3.21.33.2\pm 1.3 |
| Transformer-L | 0.6680.6680.668 | 0.8610.8610.861 | 0.4550.4550.455 | 3.1883.1883.188 | 0.8600.8600.860 | 0.8240.8240.824 | 0.7270.7270.727 | 5.4345.4345.434 | 0.9240.9240.924 | 0.9730.9730.973 | 0.7430.7430.743 | 5.9±2.2plus-or-minus5.92.25.9\pm 2.2 |
| Transformer-LR | 0.6660.6660.666 | 0.8610.8610.861 | 0.4460.4460.446 | 3.1933.1933.193 | 0.8610.8610.861 | 0.8240.8240.824 | 0.7330.7330.733 | 5.4305.4305.430 | 0.9240.9240.924 | 0.9730.9730.973 | 0.7430.7430.743 | 5.2±2.2plus-or-minus5.22.25.2\pm 2.2 |
| Transformer-Q-LR | 0.6900.6900.690 | 0.8570.8570.857 | 0.4250.425\mathbf{0.425} | 3.1433.1433.143 | 0.8680.8680.868 | 0.8180.8180.818 | 0.7260.7260.726 | 5.4715.4715.471 | 0.9240.924\mathbf{0.924} | 0.9750.9750.975 | 0.7440.7440.744 | 4.4±2.2plus-or-minus4.42.24.4\pm 2.2 |
| Transformer-T-LR | 0.6860.6860.686 | 0.8620.8620.862 | 0.4230.423\mathbf{0.423} | 3.1493.1493.149 | 0.8710.8710.871 | 0.8230.8230.823 | 0.7330.7330.733 | 5.5155.5155.515 | 0.9240.9240.924 | 0.9760.976\mathbf{0.976} | 0.7440.7440.744 | 3.7±2.2plus-or-minus3.72.23.7\pm 2.2 |
| Transformer-PLR | 0.6860.6860.686 | 0.8640.864\mathbf{0.864} | 0.4490.4490.449 | 3.0913.0913.091 | 0.8730.8730.873 | 0.8230.8230.823 | 0.7340.7340.734 | 5.5815.5815.581 | 0.9240.924\mathbf{0.924} | 0.9750.9750.975 | 0.7430.7430.743 | 3.9±2.5plus-or-minus3.92.53.9\pm 2.5 |

The main takeaways for DL models:

* •

  For most datasets, embeddings for numerical features can provide noticeable improvements for three different backbones. Although the average rank is not a good metric for making subtle conclusions, we highlight the impressive difference in average ranks between the MLP and MLP-PLR models.
* •

  The simplest LR embedding is a good baseline solution: although the performance gains are not dramatic, its main advantage is consistency (e.g. see MLP vs MLP-LR).
* •

  The PLR module provides the best average performance. Empirically, we observe σ𝜎\sigma (see [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")) to be an important hyperparameter that should be tuned.
* •

  Piecewise linear encoding (PLE) allows building well performing embeddings (e.g. T-LR, Q-LR). In addition to that, PLE itself is worth attention because of its simplicity, interpretability and efficiency (no computationally expensive periodic functions).
* •

  Importantly, after the MLP-like architectures are coupled with embeddings for numerical features, they perform on par with the Transformer-based models.

The main takeaway for the “DL vs GBDT” competition: embeddings for numerical features is a significant design aspect that has a great potential for improving DL models and closing the gap with GBDT on GBDT-friendly tasks.
Let us illustrate this claim with several observations:

* •

  The benchmark is initially biased to GBDT-friendly problems, which can be observed by comparing GBDT solutions with the vanilla DL models (MLP, ResNet, Transformer-L).
* •

  However, for the vast majority of the “backbone & dataset” pairs, proper embeddings are the only thing needed to close the gap with GBDT. Exceptions (rather formal) include the MI dataset and the following pairs: “ResNet & GE”, “Transformer & FB”, “Transformer & GE”, “Transformer & OT”.
* •

  Additionally, to the best of our knowledge, it is the first time when DL models perform on par with GBDT on the well-known California Housing and Adult datasets.

That said, compared to GBDT models, efficiency can still be an issue for the considered DL architectures.
In any case, the trade-off completely depends on the specific use case and requirements.

## 5 Analysis

### 5.1 Comparing model sizes

To quantify the effect of embeddings for numerical features on model sizes, we report the parameter counts in [Table 7](#S5.T7 "Table 7 ‣ 5.1 Comparing model sizes ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
Overall, introducing embeddings for numerical features can cause non-negligible overhead in terms of model size.
Importantly, the overhead in terms of size does not translate to the same overhead in terms of training times and throughput.
For example, the almost 200020002000-fold increase in the parameter count for MLP-LR on the CH dataset results in only 1.51.51.5-fold increase in training times.
Finally, in practice, we observe that coupling MLP and ResNet with embedding modules leads to architectures that are still faster than Transformer-based models.

Table 7: Parameter counts for MLP with different embedding modules. All the models are tuned and the corresponding backbones are not identical in their sizes, so we take into account the fact that different approaches require a different number of parameters to realize their full potential.

|  | GE | CH | CA | HO | AD | OT | HI | FB | SA | CO | MI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 2.02.02.0M | 1.51.51.5K | 43.543.543.5K | 3.63.63.6M | 5.35.35.3M | 479.9479.9479.9K | 25.825.825.8K | 937.3937.3937.3K | 5.85.85.8M | 3.23.23.2M | 276.5276.5276.5K |
| MLP-LR | ×2.52absent2.52\times 2.52 | ×1931.03absent1931.03\times 1931.03 | ×25.05absent25.05\times 25.05 | ×1.28absent1.28\times 1.28 | ×0.35absent0.35\times 0.35 | ×12.53absent12.53\times 12.53 | ×68.16absent68.16\times 68.16 | ×4.76absent4.76\times 4.76 | ×1.58absent1.58\times 1.58 | ×0.72absent0.72\times 0.72 | ×15.79absent15.79\times 15.79 |
| MLP-T | ×1.58absent1.58\times 1.58 | ×14.13absent14.13\times 14.13 | ×7.97absent7.97\times 7.97 | ×0.43absent0.43\times 0.43 | ×0.04absent0.04\times 0.04 | ×2.27absent2.27\times 2.27 | ×5.85absent5.85\times 5.85 | ×0.47absent0.47\times 0.47 | ×0.59absent0.59\times 0.59 | ×0.74absent0.74\times 0.74 | ×3.85absent3.85\times 3.85 |
| MLP-T-LR | ×1.61absent1.61\times 1.61 | ×463.55absent463.55\times 463.55 | ×6.80absent6.80\times 6.80 | ×0.23absent0.23\times 0.23 | ×0.16absent0.16\times 0.16 | ×2.52absent2.52\times 2.52 | ×113.22absent113.22\times 113.22 | ×3.43absent3.43\times 3.43 | ×0.41absent0.41\times 0.41 | ×0.35absent0.35\times 0.35 | ×8.47absent8.47\times 8.47 |
| MLP-PLR | ×1.73absent1.73\times 1.73 | ×250.24absent250.24\times 250.24 | ×12.94absent12.94\times 12.94 | ×1.07absent1.07\times 1.07 | ×0.66absent0.66\times 0.66 | ×8.05absent8.05\times 8.05 | ×110.57absent110.57\times 110.57 | ×4.93absent4.93\times 4.93 | ×0.64absent0.64\times 0.64 | ×0.44absent0.44\times 0.44 | ×9.57absent9.57\times 9.57 |

### 5.2 Ablation study

Table 8: Comparing piecewise linear encoding (PLE) with the two variations described in [subsection 5.2](#S5.SS2 "5.2 Ablation study ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning"). Notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-Q (piecewise linear) | 0.6530.653\mathbf{0.653} | 0.8540.854\mathbf{0.854} | 0.4640.4640.464 | 3.1633.163\mathbf{3.163} | 0.8590.859\mathbf{0.859} | 0.8160.816\mathbf{0.816} | 0.7210.721\mathbf{0.721} | 5.7665.7665.766 |
| MLP-Q (binary) | 0.6520.652\mathbf{0.652} | 0.8150.8150.815 | 0.4620.4620.462 | 3.2003.2003.200 | 0.8600.860\mathbf{0.860} | 0.8100.8100.810 | 0.7200.720\mathbf{0.720} | 5.7485.7485.748 |
| MLP-Q (one-blob) | 0.6130.6130.613 | 0.8510.8510.851 | 0.4610.461\mathbf{0.461} | 3.1873.187\mathbf{3.187} | 0.8570.8570.857 | 0.8080.8080.808 | 0.7190.7190.719 | 5.6455.645\mathbf{5.645} |
| MLP-T (piecewise linear) | 0.6470.647\mathbf{0.647} | 0.8610.861\mathbf{0.861} | 0.4470.447\mathbf{0.447} | 3.1493.149\mathbf{3.149} | 0.8640.8640.864 | 0.8210.821\mathbf{0.821} | 0.7200.7200.720 | 5.5775.5775.577 |
| MLP-T (binary) | 0.6390.6390.639 | 0.8550.8550.855 | 0.4640.4640.464 | 3.1633.163\mathbf{3.163} | 0.8690.869\mathbf{0.869} | 0.8130.8130.813 | 0.7180.7180.718 | 5.5725.5725.572 |
| MLP-T (one-blob) | 0.6220.6220.622 | 0.8580.8580.858 | 0.4640.4640.464 | 3.1583.158\mathbf{3.158} | 0.8700.870\mathbf{0.870} | 0.8090.8090.809 | 0.7240.724\mathbf{0.724} | 5.4755.475\mathbf{5.475} |

In this section, we compare two alternative binning-based encoding schemes with PLE (see [subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
The first one ("thermometer" [[6](#bib.bib6)]) sets the value 111 instead of the piecewise linear term (see [Equation 1](#S3.E1 "1 ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
The second one is a generalized version of the one-blob encoding [[29](#bib.bib29)] (see [subsection E.1](#A5.SS1 "E.1 One-blob encoding ‣ Appendix E Implementation details ‣ On Embeddings for Numerical Features in Tabular Deep Learning") for details).
The tuning and evaluation protocols are the same as in [subsection 4.2](#S4.SS2 "4.2 Implementation details ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The results in table [Table 8](#S5.T8 "Table 8 ‣ 5.2 Ablation study ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning") indicate that making the binning-based encoding piecewise linear is a good default strategy.

### 5.3 Piecewise linear encoding as a feature preprocessing technique

Table 9: Results for MLP and MLP with PLE for different types of data preprocessing. Solutions using PLE are significantly less sensitive to data preprocessing. Notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.4 Simple differentiable embedding modules ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP (none) | 0.5650.5650.565 | 0.7960.7960.796 | 1.1181.1181.118 | 5.3285.3285.328 | 0.8080.8080.808 | 0.7070.7070.707 | 13.12513.12513.125 | 0.9110.9110.911 | 0.9480.9480.948 | 0.8440.8440.844 |
| MLP (standard) | 0.6290.6290.629 | 0.8550.8550.855 | 0.5090.5090.509 | 3.3033.3033.303 | 0.8550.8550.855 | 0.7210.7210.721 | 5.9195.9195.919 | 0.9120.9120.912 | 0.9630.9630.963 | 0.7540.7540.754 |
| MLP (quantile) | 0.6320.6320.632 | 0.8560.8560.856 | 0.4950.4950.495 | 3.2043.2043.204 | 0.8540.8540.854 | 0.7200.7200.720 | 5.6865.6865.686 | 0.9120.9120.912 | 0.9640.9640.964 | 0.7470.7470.747 |
| MLP-Q (none) | 0.6540.6540.654 | 0.8510.8510.851 | 0.4630.4630.463 | 3.1623.1623.162 | 0.8600.8600.860 | 0.7210.7210.721 | 5.8895.8895.889 | 0.9220.9220.922 | 0.9680.9680.968 | 0.7540.7540.754 |
| MLP-Q (quantile) | 0.6530.6530.653 | 0.8540.8540.854 | 0.4640.4640.464 | 3.1633.1633.163 | 0.8590.8590.859 | 0.7210.7210.721 | 5.7665.7665.766 | 0.9220.9220.922 | 0.9680.9680.968 | 0.7500.7500.750 |
| MLP-T (none) | 0.6440.6440.644 | 0.8600.8600.860 | 0.4470.4470.447 | 3.1753.1753.175 | 0.8650.8650.865 | 0.7210.7210.721 | 5.5985.5985.598 | 0.9230.9230.923 | 0.9680.9680.968 | 0.7490.7490.749 |
| MLP-T (quantile) | 0.6470.6470.647 | 0.8610.8610.861 | 0.4470.4470.447 | 3.1493.1493.149 | 0.8640.8640.864 | 0.7200.7200.720 | 5.5775.5775.577 | 0.9230.9230.923 | 0.9670.9670.967 | 0.7490.7490.749 |

It is known that data preprocessing, such as standardization or quantile transformation, is often crucial for DL models for achieving competitive performance.
Moreover, the performance can significantly vary between different types of preprocessing.
At the same time, PLE-representations contain only values from [0, 1]01[0,\ 1] and they are invariant to shifting and scaling, which makes PLE itself a general feature preprocessing technique potentially suitable for DL models without the need to use traditional preprocessing first.

To illustrate that, for datasets where the quantile transformation was used in [section 4](#S4 "4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), we reevaluate the tuned configurations of MLP, MLP-Q, and MLP-T with different preprocessing policies and report the results in [Table 9](#S5.T9 "Table 9 ‣ 5.3 Piecewise linear encoding as a feature preprocessing technique ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning") (note that standardization is equivalent to no preprocessing for models with PLE).
  
First, the vanilla MLP often becomes unusable without preprocessing.
Second, for the vanilla MLP, it can be important to choose one specific type of preprocessing (CA, HO, FB, MI), which is less pronounced for MLP-Q and not the case for MLP-T (though, this specific observation can be the property of the benchmarks, not of MLP-T).
Overall, the results indicate that models using PLE are less sensitive to the initial preprocessing compared to the vanilla MLP.
This is an additional benefit of PLE-representations for practitioners since the aspect of preprocessing becomes less critical with PLE.

### 5.4 The “feature engineering” perspective

Table 10: The comparison of the effects of Periodic-based modules for XGBoost and MLP

|  | CA ↓ | HO ↓ | HI ↑ |
| --- | --- | --- | --- |
| XGBoost | 0.436 | 3.160 | 0.724 |
| XGBoost with Periodic | 0.441 | 3.184 | 0.724 |
| MLP | 0.495 | 3.204 | 0.720 |
| MLP-PL | 0.467 | 3.113 | 0.727 |

At first sight, feature embeddings may resemble feature engineering and should be suitable for all kinds of models.
However, the proposed embedding schemes are motivated by DL-specific aspects of training (see the motivational parts of [subsection 3.2](#S3.SS2 "3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [subsection 3.3](#S3.SS3 "3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")).
While our methods are likely to transfer well to models with similar training properties (e.g. to linear models since those are a special case of deep models), it is not the case in general.
To illustrate that, we try adopting the Periodic module for XGBoost by fixing the random coefficients from [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
We also keep the original features instead of dropping them.
The tuning and evaluation protocols are the same as in [subsection 4.2](#S4.SS2 "4.2 Implementation details ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
The results in [Table 10](#S5.T10 "Table 10 ‣ 5.4 The “feature engineering” perspective ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning") show that this technique, while being useful for DL models, does not provide any benefits for XGBoost.

## 6 Conclusion & Future work

In this work, we have demonstrated that embeddings for numerical features are an important design aspect of tabular DL architectures.
Namely, it allows existing DL backbones to achieve noticeably better results and significantly reduce the gap with Gradient Boosted Decision Trees.
We have described two approaches illustrating this phenomenon, one using the piecewise linear encoding of original scalar values, and another using periodic functions.
We have also shown that traditional MLP-like models coupled with embeddings can perform on par with attention-based models.

Nevertheless, we have only scratched the surface of the new direction. For example, it is still to be explained how exactly the discussed embedding modules help optimization on the fundamental level. Additionally, we have considered only schemes where the same functional transformation was applied to all features, which may be a suboptimal choice.

## References

* Akiba et al. [2019]

  T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, 2019.
* Arik and Pfister [2020]

  S. O. Arik and T. Pfister.
  Tabnet: Attentive interpretable tabular learning.
  *arXiv*, 1908.07442v5, 2020.
* Badirli et al. [2020]

  S. Badirli, X. Liu, Z. Xing, A. Bhowmik, K. Doan, and S. S. Keerthi.
  Gradient boosting neural networks: Grownet.
  *arXiv*, 2002.07971v2, 2020.
* Baldi et al. [2014]

  P. Baldi, P. Sadowski, and D. Whiteson.
  Searching for exotic particles in high-energy physics with deep
  learning.
  *Nature Communications*, 5, 2014.
* Blackard and Dean. [2000]

  J. A. Blackard and D. J. Dean.
  Comparative accuracies of artificial neural networks and discriminant
  analysis in predicting forest cover types from cartographic variables.
  *Computers and Electronics in Agriculture*, 24(3):131–151, 2000.
* Buckman et al. [2018]

  J. Buckman, A. Roy, C. Raffel, and I. J. Goodfellow.
  Thermometer encoding: One hot way to resist adversarial examples.
  In *International Conference on Learning Representations*, 2018.
* Chen and Guestrin [2016]

  T. Chen and C. Guestrin.
  Xgboost: A scalable tree boosting system.
  In *SIGKDD*, 2016.
* Covington et al. [2016]

  P. Covington, J. Adams, and E. Sargin.
  Deep neural networks for youtube recommendations.
  In *RecSys*, 2016.
* Cybenko [1989]

  G. Cybenko.
  Approximation by superpositions of a sigmoidal function.
  *Math. Control. Signals Syst.*, 2(4), 1989.
* Dosovitskiy et al. [2021]

  A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
  T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  In *ICLR*, 2021.
* Dougherty et al. [1995]

  J. Dougherty, R. Kohavi, and M. Sahami.
  Supervised and unsupervised discretization of continuous features.
  In *ICML*, 1995.
* Goodfellow et al. [2016]

  I. Goodfellow, Y. Bengio, and A. Courville.
  *Deep learning*.
  MIT press, 2016.
* Gorishniy et al. [2021]

  Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, 2021.
* Guo et al. [2021]

  H. Guo, B. Chen, R. Tang, W. Zhang, Z. Li, and X. He.
  An embedding learning framework for numerical features in CTR
  prediction.
  In *KDD*, 2021.
* Hazimeh et al. [2020]

  H. Hazimeh, N. Ponomareva, P. Mol, Z. Tan, and R. Mazumder.
  The tree ensemble layer: Differentiability meets conditional
  computation.
  In *ICML*, 2020.
* Hornik [1991]

  K. Hornik.
  Approximation capabilities of multilayer feedforward networks.
  *Neural Networks*, 4(2), 1991.
* Huang et al. [2020]

  X. Huang, A. Khetan, M. Cvitkovic, and Z. Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  *arXiv*, 2012.06678v1, 2020.
* Kadra et al. [2021]

  A. Kadra, M. Lindauer, F. Hutter, and J. Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In *NeurIPS*, 2021.
* Ke et al. [2017]

  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*,
  30:3146–3154, 2017.
* Kelley Pace and Barry [1997]

  R. Kelley Pace and R. Barry.
  Sparse spatial autoregressions.
  *Statistics & Probability Letters*, 33(3):291–297, 1997.
* Klambauer et al. [2017]

  G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter.
  Self-normalizing neural networks.
  In *NIPS*, 2017.
* Kohavi [1996]

  R. Kohavi.
  Scaling up the accuracy of naive-bayes classifiers: a decision-tree
  hybrid.
  In *KDD*, 1996.
* Kohavi and Sahami [1996]

  R. Kohavi and M. Sahami.
  Error-based and entropy-based discretization of continuous features.
  In *KDD*, pages 114–119. AAAI Press, 1996.
* Kossen et al. [2021]

  J. Kossen, N. Band, C. Lyle, A. N. Gomez, T. Rainforth, and Y. Gal.
  Self-attention between datapoints: Going beyond individual
  input-output pairs in deep learning.
  In *NeurIPS*, 2021.
* Li et al. [2021]

  Y. Li, S. Si, G. Li, C. Hsieh, and S. Bengio.
  Learnable fourier features for multi-dimensional spatial positional
  encoding.
  In *NeurIPS*, 2021.
* Loshchilov and Hutter [2019]

  I. Loshchilov and F. Hutter.
  Decoupled weight decay regularization.
  In *ICLR*, 2019.
* Madeo et al. [2013]

  R. C. B. Madeo, C. A. M. Lima, and S. M. Peres.
  Gesture unit segmentation using support vector machines: segmenting
  gestures from rest positions.
  In *Proceedings of the 28th Annual ACM Symposium on Applied
  Computing, SAC*, 2013.
* Mildenhall et al. [2020]

  B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and
  R. Ng.
  Nerf: Representing scenes as neural radiance fields for view
  synthesis.
  In *ECCV*, 2020.
* Müller et al. [2019]

  T. Müller, B. McWilliams, F. Rousselle, M. Gross, and J. Novák.
  Neural importance sampling.
  *ACM Trans. Graph.*, 38(5), 2019.
* Pedregosa et al. [2011]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12:2825–2830,
  2011.
* Popov et al. [2020]

  S. Popov, S. Morozov, and A. Babenko.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  In *ICLR*, 2020.
* Prokhorenkova et al. [2018]

  L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Qin and Liu [2013]

  T. Qin and T. Liu.
  Introducing LETOR 4.0 datasets.
  *arXiv*, 1306.2597v1, 2013.
* Rahaman et al. [2019]

  N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. A. Hamprecht,
  Y. Bengio, and A. C. Courville.
  On the spectral bias of neural networks.
  In *ICML*, 2019.
* Ramsauer et al. [2021]

  H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber,
  M. Holzleitner, T. Adler, D. P. Kreil, M. K. Kopp, G. Klambauer,
  J. Brandstetter, and S. Hochreiter.
  Hopfield networks is all you need.
  In *ICLR*, 2021.
* Shwartz-Ziv and Armon [2021]

  R. Shwartz-Ziv and A. Armon.
  Tabular data: Deep learning is not all you need.
  *arXiv*, 2106.03253v1, 2021.
* Singh et al. [2015]

  K. Singh, R. K. Sandhu, and D. Kumar.
  Comment volume prediction using neural networks and decision trees.
  In *IEEE UKSim-AMSS 17th International Conference on Computer
  Modelling and Simulation, UKSim*, 2015.
* Sitzmann et al. [2020]

  V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein.
  Implicit neural representations with periodic activation functions.
  In *NeurIPS*, 2020.
* Somepalli et al. [2021]

  G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss, and T. Goldstein.
  SAINT: improved neural networks for tabular data via row attention
  and contrastive pre-training.
  *arXiv*, 2106.01342v1, 2021.
* Song et al. [2019]

  W. Song, C. Shi, Z. Xiao, Z. Duan, Y. Xu, M. Zhang, and J. Tang.
  Autoint: Automatic feature interaction learning via self-attentive
  neural networks.
  In *CIKM*, 2019.
* Sundararaman et al. [2020]

  D. Sundararaman, S. Si, V. Subramanian, G. Wang, D. Hazarika, and L. Carin.
  Methods for numeracy-preserving word embeddings.
  In *Proceedings of the 2020 Conference on Empirical Methods in
  Natural Language Processing*, 2020.
* Tancik et al. [2020]

  M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan,
  U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng.
  Fourier features let networks learn high frequency functions in low
  dimensional domains.
  In *NeurIPS*, 2020.
* Turner et al. [2021]

  R. Turner, D. Eriksson, M. McCourt, J. Kiili, E. Laaksonen, Z. Xu, and
  I. Guyon.
  Bayesian optimization is superior to random search for machine
  learning hyperparameter tuning: Analysis of the black-box optimization
  challenge 2020.
  *arXiv*, https://arxiv.org/abs/2104.10201v1, 2021.
* Vanschoren et al. [2014]

  J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo.
  Openml: networked science in machine learning.
  *arXiv*, 1407.7722v1, 2014.
* Vaswani et al. [2017]

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  L. Kaiser, and I. Polosukhin.
  Attention is all you need.
  In *NIPS*, 2017.
* Wang et al. [2017]

  R. Wang, B. Fu, G. Fu, and M. Wang.
  Deep & cross network for ad click predictions.
  In *ADKDD*, 2017.

## Checklist

1. 1.

   For all authors…

   1. (a)

      Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?
      [Yes]
   2. (b)

      Did you describe the limitations of your work?
      [Yes] See the analysis in [subsection 5.1](#S5.SS1 "5.1 Comparing model sizes ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
   3. (c)

      Did you discuss any potential negative societal impacts of your work?
      [N/A] The work focuses on a generic aspect of deep learning models.
   4. (d)

      Have you read the ethics review guidelines and ensured that your paper conforms to them?
      [Yes]
2. 2.

   If you are including theoretical results…

   1. (a)

      Did you state the full set of assumptions of all theoretical results?
      [N/A] We do not include theoretical results.
   2. (b)

      Did you include complete proofs of all theoretical results?
      [N/A] We do not include theoretical results.
3. 3.

   If you ran experiments…

   1. (a)

      Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)?
      [Yes] See the supplementary material.
   2. (b)

      Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?
      [Yes] The supplementary material includes the script used to create data splits. The hyperparameters are either explicitly described in [subsection 4.2](#S4.SS2 "4.2 Implementation details ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and supplementary material, or tuned as described in [subsection 4.2](#S4.SS2 "4.2 Implementation details ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
   3. (c)

      Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)?
      [Yes] We provide standard deviations in the supplementary material, see [Table 18](#A6.T18 "Table 18 ‣ Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and see [Table 19](#A6.T19 "Table 19 ‣ Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning")
   4. (d)

      Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
      [Yes] The experiment reports included in the supplementary material provide the information about the used hardware and execution times.
4. 4.

   If you are using existing assets (e.g., code, data, models) or curating/releasing new assets…

   1. (a)

      If your work uses existing assets, did you cite the creators?
      [Yes] See [Appendix C](#A3 "Appendix C Additional details on datasets ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
   2. (b)

      Did you mention the license of the assets?
      [Yes] In the README.md file in the supplementary material, we refer to the original licenses of the used datasets.
   3. (c)

      Did you include any new assets either in the supplemental material or as a URL?
      [N/A] We do not provide new datasets.
   4. (d)

      Did you discuss whether and how consent was obtained from people whose data you’re using/curating?
      [N/A] We use publicly available datasets.
   5. (e)

      Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content?
      [N/A] We use publicly available datasets.
5. 5.

   If you used crowdsourcing or conducted research with human subjects…

   1. (a)

      Did you include the full text of instructions given to participants and screenshots, if applicable?
      [N/A] We did not use crowdsourcing. We did not conduct research with human subjects.
   2. (b)

      Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable?
      [N/A] We did not use crowdsourcing. We did not conduct research with human subjects.
   3. (c)

      Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation?
      [N/A] We did not use crowdsourcing. We did not conduct research with human subjects.

## Supplementary material

## Appendix A MLP with embeddings for numerical features

We provide visual explanation of how embeddings are passed to MLP in [Figure 2](#A1.F2 "Figure 2 ‣ Appendix A MLP with embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [Figure 3](#A1.F3 "Figure 3 ‣ Appendix A MLP with embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
Also, we provide the formal explanation in [Equation 3](#A1.E3 "3 ‣ Appendix A MLP with embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") (categorical features are omitted for simplicity).

Figure 2: The vanilla MLP. The model takes two numerical features as input.

Figure 3: The same MLP as in [Figure 2](#A1.F2 "Figure 2 ‣ Appendix A MLP with embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), but now with embeddings for numerical features.

|  |  |  |  |
| --- | --- | --- | --- |
|  | MLP​(z1,…,zk)=MLP​(concat​[z1,…,zk])concat​[z1,…,zk]∈ℝd1+…+dkformulae-sequenceMLPsubscript𝑧1…subscript𝑧𝑘MLPconcat  subscript𝑧1…subscript𝑧𝑘concat  subscript𝑧1…subscript𝑧𝑘superscriptℝsubscript𝑑1…subscript𝑑𝑘\texttt{MLP}(z\_{1},\ \dots,\ z\_{k})=\texttt{MLP}\left(\texttt{concat}[z\_{1},\ \dots,\ z\_{k}]\right)\qquad\texttt{concat}[z\_{1},\ \dots,\ z\_{k}]\in\mathbb{R}^{d\_{1}+\dots\ +d\_{k}} |  | (3) |

## Appendix B Target-aware piecewise linear encoding

We provide visualisation of target-aware PLE ([subsubsection 3.2.2](#S3.SS2.SSS2 "3.2.2 Building target-aware bins ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")) in [Figure 4](#A2.F4 "Figure 4 ‣ Appendix B Target-aware piecewise linear encoding ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

!(/html/2203.05556/assets/x2.png)

Figure 4: Obtaining bins for PLE from decision trees.

## Appendix C Additional details on datasets

Table 11: Details on datasets, used for experiments

| Abbr | Name | # Train | # Validation | # Test | # Num | # Cat | Task type | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GE | Gesture Phase | 631863186318 | 158015801580 | 197519751975 | 323232 | 00 | Multiclass | 128 |
| CH | Churn Modelling | 640064006400 | 160016001600 | 200020002000 | 101010 | 111 | Binclass | 128 |
| CA | California Housing | 132091320913209 | 330333033303 | 412841284128 | 888 | 00 | Regression | 256 |
| HO | House 16H | 145811458114581 | 364636463646 | 455745574557 | 161616 | 00 | Regression | 256 |
| AD | Adult | 260482604826048 | 651365136513 | 162811628116281 | 666 | 888 | Binclass | 256 |
| OT | Otto Group Products | 396013960139601 | 990199019901 | 123761237612376 | 939393 | 00 | Multiclass | 512 |
| HI | Higgs Small | 627516275162751 | 156881568815688 | 196101961019610 | 282828 | 00 | Binclass | 512 |
| FB | Facebook Comments Volume | 157638157638157638 | 197221972219722 | 197201972019720 | 505050 | 111 | Regression | 512 |
| SA | Santander Customer Transactions | 128000128000128000 | 320003200032000 | 400004000040000 | 200200200 | 00 | Binclass | 1024 |
| CO | Covertype | 371847371847371847 | 929629296292962 | 116203116203116203 | 545454 | 00 | Multiclass | 1024 |
| MI | MSLR-WEB10K (Fold 1) | 723412723412723412 | 235259235259235259 | 241521241521241521 | 136136136 | 00 | Regression | 1024 |

We used the following datasets:

* •

  Gesture Phase Prediction (Madeo et al. [[27](#bib.bib27)])
* •

  Churn Modeling111https://www.kaggle.com/shrutimechlearn/churn-modelling
* •

  California Housing (real estate data, Kelley Pace and Barry [[20](#bib.bib20)])
* •

  House 16H222https://www.openml.org/d/574
* •

  Adult (income estimation, Kohavi [[22](#bib.bib22)])
* •

  Otto Group Product Classification333https://www.kaggle.com/c/otto-group-product-classification-challenge/data
* •

  Higgs (simulated physical particles, Baldi et al. [[4](#bib.bib4)]; we use the version with 98K samples available in the OpenML repository [[44](#bib.bib44)])
* •

  Santander Customer Transaction Prediction444https://www.kaggle.com/c/santander-customer-transaction-prediction
* •

  Facebook Comments (Singh et al. [[37](#bib.bib37)])
* •

  Covertype (forest characteristics, Blackard and Dean. [[5](#bib.bib5)])
* •

  Microsoft (search queries, Qin and Liu [[33](#bib.bib33)]). We follow the pointwise approach to learning-to-rank and treat this ranking problem as a regression problem.

## Appendix D Additional analysis

### D.1 Testing quantile-based PLE on the synthetic GBDT-friendly dataset

!(/html/2203.05556/assets/x3.png)

Figure 5: RMSE (averaged over five random seeds) of different approaches on the same synthetic GBDT-friendly task.
Using PLE-representations (“-Q”) instead of scalar values improves the performance of MLP and Transformer.
Note that in practice, increasing the number of bins does not always lead to better results.

In this section, we apply the quantile-based piecewise linear encoding (described in [subsubsection 3.2.1](#S3.SS2.SSS1 "3.2.1 Obtaining bins from quantiles ‣ 3.2 Piecewise linear encoding ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning") to MLP and Transformer on the synthetic GBDT-friendly dataset described in section 5.1 in Gorishniy et al. [[13](#bib.bib13)].
In a nutshell, features of this dataset are sampled randomly from 𝒩​(0,1)𝒩01\mathcal{N}(0,1), and the target is produced by an ensemble of randomly constructed decision trees applied to the sampled features.
This task turns out to be easy for GBDT, but hard for traditional DL models [[13](#bib.bib13)].
The results are visualized in [Figure 5](#A4.F5 "Figure 5 ‣ D.1 Testing quantile-based PLE on the synthetic GBDT-friendly dataset ‣ Appendix D Additional analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
As the plot shows, PLE-representations can be helpful for both MLP and Transformer backbones. In the considered synthetic setup, increasing the number of bins leads to better results, however, in practice, using too many bins can lead to overfitting; therefore, we recommend tuning the number of bins based on a validation set.

Technical details. Our dataset has 10,000

1000010,000 objects, 888 features and the target was produced by 161616 decision trees of depth 666.
CatBoost is trained with the default hyperparameters.
Transformer is trained with the default hyperparameters of FT-Transformer.
The MLP backbone has four layers of size 256 each.
Importantly, the task GBDT-friendly, which can be illustrated by the performance of the tuned MLP: 0.2229±0.0055plus-or-minus0.22290.00550.2229\pm 0.0055 (it is still worse than the performance of CatBoost).
The remaining details can be found in the source code.

### D.2 Fourier features

In this section, we test Fourier features implemented exactly as in Tancik et al. [[42](#bib.bib42)], i.e. pre-activation coefficients are not trained and features are mixed right from the start.
Importantly, the latter means that this approach is not covered by the embedding framework described in [subsection 3.1](#S3.SS1 "3.1 General framework ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").
As reported in [Table 12](#A4.T12 "Table 12 ‣ D.2 Fourier features ‣ Appendix D Additional analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), MLP equipped with the original Fourier features does not perform well even compared to the vanilla MLP.
So, it seems to be important to embed each feature separately as described in [subsection 3.1](#S3.SS1 "3.1 General framework ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning").

Table 12: Results for the vanilla MLP and MLP equipped with Fourier features [[42](#bib.bib42)]. Notation
follows Table 3 and Table 2.

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.6320.632\mathbf{0.632} | 0.8560.856\mathbf{0.856} | 0.4950.495\mathbf{0.495} | 3.2043.204\mathbf{3.204} | 0.8540.8540.854 | 0.8180.818\mathbf{0.818} | 0.7200.720\mathbf{0.720} | 5.6865.686\mathbf{5.686} | 0.9120.9120.912 | 0.9640.964\mathbf{0.964} | 0.7470.747\mathbf{0.747} |
| MLP (Fourier features) | 0.6120.6120.612 | 0.8450.8450.845 | 0.4950.495\mathbf{0.495} | 3.2673.2673.267 | 0.8580.858\mathbf{0.858} | 0.8100.8100.810 | 0.7110.7110.711 | 5.7675.7675.767 | 0.9150.915\mathbf{0.915} | 0.9610.9610.961 | 0.7490.7490.749 |

## Appendix E Implementation details

We mostly follow Gorishniy et al. [[13](#bib.bib13)] in terms of the tuning, training and evaluation protocols.

Data preprocessing.
Preliminary data preprocessing is known to be crucial for the optimization of tabular DL models.
For each dataset, the same preprocessing was used for all deep models for a fair comparison.
For all datasets except for Otto Group Product Classification, we use the quantile transformation from the Scikit-learn library [[30](#bib.bib30)].
For Otto Group Product Classification, we do not apply any feature preprocessing.
We also apply standardization to regression targets for all algorithms.

Tuning.
For every dataset, we carefully tune each model’s hyperparameters.
The best hyperparameters are the ones that perform best on the validation set, so the test set is never used for tuning.
For most algorithms, we use the Optuna library [[1](#bib.bib1)] to run Bayesian optimization (the Tree-Structured Parzen Estimator algorithm), which is reported to be superior to random search [[43](#bib.bib43)]. The search spaces for all hyperparameters are reported in the appendix.

Evaluation. For each tuned configuration, we run 15 experiments with different random seeds and report the average performance on the test set.

Ensembles. For each model-dataset pair, we obtain three ensembles by splitting the 15 single models into three disjoint groups of equal size and averaging predictions of single models within each group.

Neural networks.
The implementations of the MLP, ResNet, and Transformer backbones are taken from Gorishniy et al. [[13](#bib.bib13)].
We minimize cross-entropy for classification problems and mean squared error for regression problems.
We use the AdamW optimizer [[26](#bib.bib26)].
We do not apply learning rate schedules.
For each dataset, we use a predefined batch size (see [Appendix C](#A3 "Appendix C Additional details on datasets ‣ On Embeddings for Numerical Features in Tabular Deep Learning") for the specific values).
We continue training until there are patience+1patience1\texttt{patience}+1 consecutive epochs without improvements on the validation set; we set patience=16patience16\texttt{patience}=16 for all models.

Categorical features.
For CatBoost, we employ the built-in support for categorical features. For all other algorithms, we use the one-hot encoding.

### E.1 One-blob encoding

In [subsection 5.2](#S5.SS2 "5.2 Ablation study ‣ 5 Analysis ‣ On Embeddings for Numerical Features in Tabular Deep Learning"), we used a slightly generalized version of the original one-blob encoding [[29](#bib.bib29)].
Namely, while the original sets the width of the kernel to T−1superscript𝑇1T^{-1} (T𝑇T is the number of bins), we set it to T−γsuperscript𝑇𝛾T^{-\gamma} and tune γ𝛾\gamma.

### E.2 Hyperparameter tuning configurations

### E.3 CatBoost

We fix and do not tune the following hyperparameters:

* •

  early-stopping-rounds=50early-stopping-rounds50\texttt{early-stopping-rounds}=50
* •

  od-pval=0.001od-pval0.001\texttt{od-pval}=0.001
* •

  iterations=2000iterations2000\texttt{iterations}=2000

For tuning on the MI and CO datasets, we set the task\_type parameter to “GPU”. In all other cases (including the evaluation on these two datasets), we set this parameter to “CPU”.

Table 13: CatBoost hyperparameter space

| Parameter | Distribution |
| --- | --- |
| Max depth | UniformInt​[1,10]UniformInt110\mathrm{UniformInt[1,10]} |
| Learning rate | LogUniform​[0.001,1]LogUniform0.0011\mathrm{LogUniform}[0.001,1] |
| Bagging temperature | Uniform​[0,1]Uniform01\mathrm{Uniform}[0,1] |
| L2 leaf reg | LogUniform​[1,10]LogUniform110\mathrm{LogUniform}[1,10] |
| Leaf estimation iterations | UniformInt​[1,10]UniformInt110\mathrm{UniformInt}[1,10] |
| # Iterations | 100 |

### E.4 XGBoost

We fix and do not tune the following hyperparameters:

* •

  booster="gbtree"booster"gbtree"\texttt{booster}=\text{"gbtree"}
* •

  early-stopping-rounds=50early-stopping-rounds50\texttt{early-stopping-rounds}=50
* •

  n-estimators=2000n-estimators2000\texttt{n-estimators}=2000

Table 14: XGBoost hyperparameter space.

| Parameter | Distribution |
| --- | --- |
| Max depth | UniformInt​[3,10]UniformInt310\mathrm{UniformInt[3,10]} |
| Min child weight | LogUniform​[0.0001,100]LogUniform0.0001100\mathrm{LogUniform}[0.0001,100] |
| Subsample | Uniform​[0.5,1]Uniform0.51\mathrm{Uniform}[0.5,1] |
| Learning rate | LogUniform​[0.001,1]LogUniform0.0011\mathrm{LogUniform}[0.001,1] |
| Col sample by tree | Uniform​[0.5,1]Uniform0.51\mathrm{Uniform}[0.5,1] |
| Gamma | {0,LogUniform​[0.001,100]}0LogUniform0.001100\{0,\mathrm{LogUniform}[0.001,100]\} |
| Lambda | {0,LogUniform​[0.1,10]}0LogUniform0.110\{0,\mathrm{LogUniform}[0.1,10]\} |
| # Iterations | 100 |

### E.5 MLP

Table 15: MLP hyperparameter space.

| Parameter | Distribution |
| --- | --- |
| # Layers | UniformInt​[1,16]UniformInt116\mathrm{UniformInt}[1,16] |
| Layer size | UniformInt​[1,1024]UniformInt11024\mathrm{UniformInt}[1,1024] |
| Dropout | {0,Uniform​[0,0.5]}0Uniform00.5\{0,\mathrm{Uniform}[0,0.5]\} |
| Learning rate | LogUniform​[5​e​-​5,0.005]LogUniform5𝑒-50.005\mathrm{LogUniform}[5e\text{-}5,0.005] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| # Iterations | 100 |

### E.6 ResNet

Table 16: ResNet hyperparameter space.

| Parameter | Distribution |
| --- | --- |
| # Layers | UniformInt​[1,8]UniformInt18\mathrm{UniformInt}[1,8] |
| Layer size | UniformInt​[32,512]UniformInt32512\mathrm{UniformInt}[32,512] |
| Hidden factor | Uniform​[1,4]Uniform14\mathrm{Uniform}[1,4] |
| Hidden dropout | Uniform​[0,0.5]Uniform00.5\mathrm{Uniform}[0,0.5] |
| Residual dropout | {0,Uniform​[0,0.5]}0Uniform00.5\{0,\mathrm{Uniform}[0,0.5]\} |
| Learning rate | LogUniform​[5​e​-​5,0.005]LogUniform5𝑒-50.005\mathrm{LogUniform}[5e\text{-}5,0.005] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| # Iterations | 100 |

### E.7 Transformer

Table 17: Transformer hyperparameter space. Here (A) = {SA, CO, MI} and (B) = the rest

| Parameter | (Datasets) Distribution |
| --- | --- |
| # Layers | (A) UniformInt​[2,4]UniformInt24\mathrm{UniformInt}[2,4], (B) UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| Embedding size | (A) UniformInt​[192,512]UniformInt192512\mathrm{UniformInt}[192,512], (B) UniformInt​[96,512]UniformInt96512\mathrm{UniformInt}[96,512] |
| Residual dropout | (A) Const​(0.0)Const0.0\mathrm{Const}(0.0), (B) {0,Uniform​[0,0.2]}0Uniform00.2\{0,\mathrm{Uniform}[0,0.2]\} |
| Attention dropout | (A,B) Uniform​[0,0.5]Uniform00.5\mathrm{Uniform}[0,0.5] |
| FFN dropout | (A,B) Uniform​[0,0.5]Uniform00.5\mathrm{Uniform}[0,0.5] |
| FFN factor | (A,B) Uniform​[2/3,8/3]Uniform2383\mathrm{Uniform}[\nicefrac{{2}}{{3}},\nicefrac{{8}}{{3}}] |
| Learning rate | (A) LogUniform​[1​e​-​5,3​e​-​4]LogUniform1𝑒-53𝑒-4\mathrm{LogUniform}[1e\text{-}5,3e\text{-}4], (B) LogUniform​[1​e​-​5,1​e​-​3]LogUniform1𝑒-51𝑒-3\mathrm{LogUniform}[1e\text{-}5,1e\text{-}3] |
| Weight decay | (A) Const​(1​e​-​5)Const1𝑒-5\mathrm{Const}(1e\text{-}5), (B) LogUniform​[1​e​-​6,1​e​-​4]LogUniform1𝑒-61𝑒-4\mathrm{LogUniform}[1e\text{-}6,1e\text{-}4] |
| # Iterations | (A) 50, (B) 100 |

### E.8 Embedding hyperparameters

The distribution for the output dimensions of linear layers is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128].

PLE. We share the same hyperparameter space for PLE across all datasets and models.
For the quantile-based PLE, the distribution for the number of quantiles is UniformInt​[2,256]UniformInt2256\mathrm{UniformInt}[2,256].
For the target-aware (tree-based) PLE, the distribution for the number of leaves is UniformInt​[2,256]UniformInt2256\mathrm{UniformInt}[2,256], the distribution for the minimum number of items per leaf is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128] and the distribution for the minimum information gain required for making a split is LogUniform​[1​e​-​9,0.01]LogUniform1𝑒-90.01\mathrm{LogUniform}[1e\text{-}9,0.01].

Periodic. The distribution for k𝑘k (see [Equation 2](#S3.E2 "2 ‣ 3.3 Periodic activation functions ‣ 3 Embeddings for numerical features ‣ On Embeddings for Numerical Features in Tabular Deep Learning")) is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128].

## Appendix F Extended tables with experimental results

The scores with standard deviations for single models and ensembles are provided in [Table 18](#A6.T18 "Table 18 ‣ Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning") and [Table 19](#A6.T19 "Table 19 ‣ Appendix F Extended tables with experimental results ‣ On Embeddings for Numerical Features in Tabular Deep Learning") respectively.
Please, refer to [subsection 4.3](#S4.SS3 "4.3 Model names ‣ 4 Experiments ‣ On Embeddings for Numerical Features in Tabular Deep Learning") to learn about the model names.

Additionally, we include the results for the DICE embeddings [[41](#bib.bib41)], which is a general way to represent numbers with vectors introduced in the context of NLP.
The results though demonstrate that it is a suboptimal approach in tabular data problems.

Table 18: Extended results for single models

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| CatBoost | 0.683±4.7​e​-​3plus-or-minus0.6834.7𝑒-30.683\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3 | 0.861±3.5​e​-​3plus-or-minus0.8613.5𝑒-30.861\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3 | 0.433±1.8​e​-​3plus-or-minus0.4331.8𝑒-30.433\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 3.115±1.9​e​-​2plus-or-minus3.1151.9𝑒-23.115\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2 | 0.872±9.0​e​-​4plus-or-minus0.8729.0𝑒-40.872\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}4 | 0.824±1.1​e​-​3plus-or-minus0.8241.1𝑒-30.824\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.726±1.0​e​-​3plus-or-minus0.7261.0𝑒-30.726\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 5.324±4.1​e​-​2plus-or-minus5.3244.1𝑒-25.324\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}2 | 0.923±3.6​e​-​4plus-or-minus0.9233.6𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4 | 0.966±3.3​e​-​4plus-or-minus0.9663.3𝑒-40.966\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4 | 0.743±3.1​e​-​4plus-or-minus0.7433.1𝑒-40.743\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4 |
| XGBoost | 0.678±4.9​e​-​3plus-or-minus0.6784.9𝑒-30.678\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3 | 0.858±2.2​e​-​3plus-or-minus0.8582.2𝑒-30.858\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.436±2.5​e​-​3plus-or-minus0.4362.5𝑒-30.436\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 3.160±6.9​e​-​3plus-or-minus3.1606.9𝑒-33.160\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3 | 0.874±8.2​e​-​4plus-or-minus0.8748.2𝑒-40.874\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4 | 0.825±2.3​e​-​3plus-or-minus0.8252.3𝑒-30.825\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 0.724±1.0​e​-​3plus-or-minus0.7241.0𝑒-30.724\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 5.383±2.9​e​-​2plus-or-minus5.3832.9𝑒-25.383\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2 | 0.918±5.0​e​-​3plus-or-minus0.9185.0𝑒-30.918\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}3 | 0.969±6.1​e​-​4plus-or-minus0.9696.1𝑒-40.969\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4 | 0.742±1.6​e​-​4plus-or-minus0.7421.6𝑒-40.742\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4 |
| MLP | 0.632±1.4​e​-​2plus-or-minus0.6321.4𝑒-20.632\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2 | 0.856±2.8​e​-​3plus-or-minus0.8562.8𝑒-30.856\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}3 | 0.495±4.3​e​-​3plus-or-minus0.4954.3𝑒-30.495\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3 | 3.204±4.0​e​-​2plus-or-minus3.2044.0𝑒-23.204\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}2 | 0.854±1.6​e​-​3plus-or-minus0.8541.6𝑒-30.854\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.818±3.1​e​-​3plus-or-minus0.8183.1𝑒-30.818\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.720±2.3​e​-​3plus-or-minus0.7202.3𝑒-30.720\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 5.686±4.7​e​-​2plus-or-minus5.6864.7𝑒-25.686\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}2 | 0.912±4.3​e​-​4plus-or-minus0.9124.3𝑒-40.912\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4 | 0.964±8.6​e​-​4plus-or-minus0.9648.6𝑒-40.964\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}4 | 0.747±2.5​e​-​4plus-or-minus0.7472.5𝑒-40.747\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4 |
| MLP-L | 0.639±1.3​e​-​2plus-or-minus0.6391.3𝑒-20.639\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.861±2.1​e​-​3plus-or-minus0.8612.1𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.475±5.4​e​-​3plus-or-minus0.4755.4𝑒-30.475\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}3 | 3.123±4.5​e​-​2plus-or-minus3.1234.5𝑒-23.123\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2 | 0.856±1.6​e​-​3plus-or-minus0.8561.6𝑒-30.856\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.820±1.5​e​-​3plus-or-minus0.8201.5𝑒-30.820\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.723±1.6​e​-​3plus-or-minus0.7231.6𝑒-30.723\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.684±4.5​e​-​2plus-or-minus5.6844.5𝑒-25.684\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2 | 0.916±3.5​e​-​4plus-or-minus0.9163.5𝑒-40.916\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4 | 0.963±9.3​e​-​4plus-or-minus0.9639.3𝑒-40.963\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}4 | 0.748±4.1​e​-​4plus-or-minus0.7484.1𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 |
| MLP-LR | 0.642±1.5​e​-​2plus-or-minus0.6421.5𝑒-20.642\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2 | 0.860±3.0​e​-​3plus-or-minus0.8603.0𝑒-30.860\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3 | 0.471±2.6​e​-​3plus-or-minus0.4712.6𝑒-30.471\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 3.084±2.7​e​-​2plus-or-minus3.0842.7𝑒-23.084\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}2 | 0.857±1.9​e​-​3plus-or-minus0.8571.9𝑒-30.857\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 0.819±1.6​e​-​3plus-or-minus0.8191.6𝑒-30.819\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.726±1.9​e​-​3plus-or-minus0.7261.9𝑒-30.726\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 5.625±5.6​e​-​2plus-or-minus5.6255.6𝑒-25.625\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}2 | 0.923±3.1​e​-​4plus-or-minus0.9233.1𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4 | 0.963±1.4​e​-​3plus-or-minus0.9631.4𝑒-30.963\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.746±3.9​e​-​4plus-or-minus0.7463.9𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4 |
| MLP-LRLR | 0.654±1.7​e​-​2plus-or-minus0.6541.7𝑒-20.654\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}2 | 0.861±2.6​e​-​3plus-or-minus0.8612.6𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.460±3.8​e​-​3plus-or-minus0.4603.8𝑒-30.460\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}3 | 3.070±3.6​e​-​2plus-or-minus3.0703.6𝑒-23.070\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2 | 0.857±1.4​e​-​3plus-or-minus0.8571.4𝑒-30.857\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.819±2.2​e​-​3plus-or-minus0.8192.2𝑒-30.819\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.725±1.2​e​-​3plus-or-minus0.7251.2𝑒-30.725\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 5.551±4.6​e​-​2plus-or-minus5.5514.6𝑒-25.551\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2 | 0.923±3.0​e​-​4plus-or-minus0.9233.0𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 | 0.963±1.4​e​-​3plus-or-minus0.9631.4𝑒-30.963\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.746±3.0​e​-​4plus-or-minus0.7463.0𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 |
| MLP-Q | 0.653±8.9​e​-​3plus-or-minus0.6538.9𝑒-30.653\scriptscriptstyle\pm\scriptstyle 8.9e\text{-}3 | 0.854±3.0​e​-​3plus-or-minus0.8543.0𝑒-30.854\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3 | 0.464±3.1​e​-​3plus-or-minus0.4643.1𝑒-30.464\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 3.163±3.1​e​-​2plus-or-minus3.1633.1𝑒-23.163\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2 | 0.859±1.6​e​-​3plus-or-minus0.8591.6𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.816±2.6​e​-​3plus-or-minus0.8162.6𝑒-30.816\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.721±1.0​e​-​3plus-or-minus0.7211.0𝑒-30.721\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 5.766±5.3​e​-​2plus-or-minus5.7665.3𝑒-25.766\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}2 | 0.922±6.3​e​-​4plus-or-minus0.9226.3𝑒-40.922\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 | 0.968±6.9​e​-​4plus-or-minus0.9686.9𝑒-40.968\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}4 | 0.750±3.7​e​-​4plus-or-minus0.7503.7𝑒-40.750\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4 |
| MLP-Q-LR | 0.646±6.3​e​-​3plus-or-minus0.6466.3𝑒-30.646\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}3 | 0.857±2.6​e​-​3plus-or-minus0.8572.6𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.455±3.4​e​-​3plus-or-minus0.4553.4𝑒-30.455\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3 | 3.184±3.1​e​-​2plus-or-minus3.1843.1𝑒-23.184\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2 | 0.863±1.7​e​-​3plus-or-minus0.8631.7𝑒-30.863\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.811±1.8​e​-​3plus-or-minus0.8111.8𝑒-30.811\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.720±1.5​e​-​3plus-or-minus0.7201.5𝑒-30.720\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 5.394±1.5​e​-​1plus-or-minus5.3941.5𝑒-15.394\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}1 | 0.923±6.1​e​-​4plus-or-minus0.9236.1𝑒-40.923\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4 | 0.969±4.8​e​-​4plus-or-minus0.9694.8𝑒-40.969\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.747±3.9​e​-​4plus-or-minus0.7473.9𝑒-40.747\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4 |
| MLP-Q-LRLR | 0.644±6.2​e​-​3plus-or-minus0.6446.2𝑒-30.644\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}3 | 0.859±2.2​e​-​3plus-or-minus0.8592.2𝑒-30.859\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.452±4.3​e​-​3plus-or-minus0.4524.3𝑒-30.452\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3 | 3.118±4.6​e​-​2plus-or-minus3.1184.6𝑒-23.118\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2 | 0.869±1.5​e​-​3plus-or-minus0.8691.5𝑒-30.869\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.812±2.5​e​-​3plus-or-minus0.8122.5𝑒-30.812\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.724±1.3​e​-​3plus-or-minus0.7241.3𝑒-30.724\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 5.618±2.0​e​-​1plus-or-minus5.6182.0𝑒-15.618\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}1 | 0.924±4.5​e​-​4plus-or-minus0.9244.5𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}4 | 0.969±1.2​e​-​3plus-or-minus0.9691.2𝑒-30.969\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 0.748±4.6​e​-​4plus-or-minus0.7484.6𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4 |
| MLP-T | 0.647±5.7​e​-​3plus-or-minus0.6475.7𝑒-30.647\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}3 | 0.861±1.1​e​-​3plus-or-minus0.8611.1𝑒-30.861\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.447±2.0​e​-​3plus-or-minus0.4472.0𝑒-30.447\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 3.149±5.2​e​-​2plus-or-minus3.1495.2𝑒-23.149\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}2 | 0.864±6.3​e​-​4plus-or-minus0.8646.3𝑒-40.864\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 | 0.821±1.8​e​-​3plus-or-minus0.8211.8𝑒-30.821\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.720±1.9​e​-​3plus-or-minus0.7201.9𝑒-30.720\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 5.577±3.7​e​-​2plus-or-minus5.5773.7𝑒-25.577\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}2 | 0.923±3.0​e​-​4plus-or-minus0.9233.0𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 | 0.967±1.1​e​-​3plus-or-minus0.9671.1𝑒-30.967\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.749±4.4​e​-​4plus-or-minus0.7494.4𝑒-40.749\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 |
| MLP-T-LR | 0.640±6.9​e​-​3plus-or-minus0.6406.9𝑒-30.640\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3 | 0.861±2.0​e​-​3plus-or-minus0.8612.0𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.439±3.7​e​-​3plus-or-minus0.4393.7𝑒-30.439\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3 | 3.207±5.2​e​-​2plus-or-minus3.2075.2𝑒-23.207\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}2 | 0.868±1.1​e​-​3plus-or-minus0.8681.1𝑒-30.868\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.818±1.3​e​-​3plus-or-minus0.8181.3𝑒-30.818\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.724±1.7​e​-​3plus-or-minus0.7241.7𝑒-30.724\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 5.508±3.0​e​-​2plus-or-minus5.5083.0𝑒-25.508\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}2 | 0.924±2.4​e​-​4plus-or-minus0.9242.4𝑒-40.924\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 | 0.968±7.2​e​-​4plus-or-minus0.9687.2𝑒-40.968\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}4 | 0.747±5.7​e​-​4plus-or-minus0.7475.7𝑒-40.747\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}4 |
| MLP-T-LRLR | 0.629±1.0​e​-​2plus-or-minus0.6291.0𝑒-20.629\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2 | 0.857±2.4​e​-​3plus-or-minus0.8572.4𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.446±3.6​e​-​3plus-or-minus0.4463.6𝑒-30.446\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}3 | 3.153±4.0​e​-​2plus-or-minus3.1534.0𝑒-23.153\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}2 | 0.870±9.9​e​-​4plus-or-minus0.8709.9𝑒-40.870\scriptscriptstyle\pm\scriptstyle 9.9e\text{-}4 | 0.818±2.1​e​-​3plus-or-minus0.8182.1𝑒-30.818\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.725±1.3​e​-​3plus-or-minus0.7251.3𝑒-30.725\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 5.553±2.4​e​-​2plus-or-minus5.5532.4𝑒-25.553\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2 | 0.924±3.6​e​-​4plus-or-minus0.9243.6𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4 | 0.967±8.5​e​-​4plus-or-minus0.9678.5𝑒-40.967\scriptscriptstyle\pm\scriptstyle 8.5e\text{-}4 | 0.748±5.8​e​-​4plus-or-minus0.7485.8𝑒-40.748\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 |
| MLP-P | 0.631±1.7​e​-​2plus-or-minus0.6311.7𝑒-20.631\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}2 | 0.860±3.1​e​-​3plus-or-minus0.8603.1𝑒-30.860\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.489±2.4​e​-​3plus-or-minus0.4892.4𝑒-30.489\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 3.129±4.3​e​-​2plus-or-minus3.1294.3𝑒-23.129\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}2 | 0.869±1.5​e​-​3plus-or-minus0.8691.5𝑒-30.869\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.807±4.3​e​-​3plus-or-minus0.8074.3𝑒-30.807\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3 | 0.723±1.5​e​-​3plus-or-minus0.7231.5𝑒-30.723\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 5.845±6.4​e​-​2plus-or-minus5.8456.4𝑒-25.845\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}2 | 0.923±4.3​e​-​4plus-or-minus0.9234.3𝑒-40.923\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4 | 0.968±9.0​e​-​4plus-or-minus0.9689.0𝑒-40.968\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}4 | 0.747±3.1​e​-​4plus-or-minus0.7473.1𝑒-40.747\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4 |
| MLP-PL | 0.641±1.0​e​-​2plus-or-minus0.6411.0𝑒-20.641\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2 | 0.859±2.4​e​-​3plus-or-minus0.8592.4𝑒-30.859\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.467±2.9​e​-​3plus-or-minus0.4672.9𝑒-30.467\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 3.113±3.1​e​-​2plus-or-minus3.1133.1𝑒-23.113\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2 | 0.868±1.1​e​-​3plus-or-minus0.8681.1𝑒-30.868\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.819±1.7​e​-​3plus-or-minus0.8191.7𝑒-30.819\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.727±1.7​e​-​3plus-or-minus0.7271.7𝑒-30.727\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 5.530±9.5​e​-​2plus-or-minus5.5309.5𝑒-25.530\scriptscriptstyle\pm\scriptstyle 9.5e\text{-}2 | 0.924±4.0​e​-​4plus-or-minus0.9244.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 | 0.969±5.0​e​-​4plus-or-minus0.9695.0𝑒-40.969\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 | 0.746±2.6​e​-​4plus-or-minus0.7462.6𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}4 |
| MLP-PLR | 0.674±1.0​e​-​2plus-or-minus0.6741.0𝑒-20.674\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2 | 0.857±2.4​e​-​3plus-or-minus0.8572.4𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.467±5.8​e​-​3plus-or-minus0.4675.8𝑒-30.467\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}3 | 3.050±3.4​e​-​2plus-or-minus3.0503.4𝑒-23.050\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}2 | 0.870±1.0​e​-​3plus-or-minus0.8701.0𝑒-30.870\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 0.819±2.0​e​-​3plus-or-minus0.8192.0𝑒-30.819\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.728±1.6​e​-​3plus-or-minus0.7281.6𝑒-30.728\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.525±3.5​e​-​2plus-or-minus5.5253.5𝑒-25.525\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}2 | 0.924±4.0​e​-​4plus-or-minus0.9244.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 | 0.970±9.5​e​-​4plus-or-minus0.9709.5𝑒-40.970\scriptscriptstyle\pm\scriptstyle 9.5e\text{-}4 | 0.746±3.0​e​-​4plus-or-minus0.7463.0𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 |
| MLP-PLRLR | 0.676±1.6​e​-​2plus-or-minus0.6761.6𝑒-20.676\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2 | 0.863±3.1​e​-​3plus-or-minus0.8633.1𝑒-30.863\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.456±3.7​e​-​3plus-or-minus0.4563.7𝑒-30.456\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3 | 3.038±2.3​e​-​2plus-or-minus3.0382.3𝑒-23.038\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2 | 0.871±1.4​e​-​3plus-or-minus0.8711.4𝑒-30.871\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.818±1.7​e​-​3plus-or-minus0.8181.7𝑒-30.818\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.725±1.6​e​-​3plus-or-minus0.7251.6𝑒-30.725\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.606±8.9​e​-​2plus-or-minus5.6068.9𝑒-25.606\scriptscriptstyle\pm\scriptstyle 8.9e\text{-}2 | 0.924±2.8​e​-​4plus-or-minus0.9242.8𝑒-40.924\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 | 0.968±2.0​e​-​3plus-or-minus0.9682.0𝑒-30.968\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.744±2.8​e​-​4plus-or-minus0.7442.8𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 |
| MLP-AutoDis | 0.649±1.2​e​-​2plus-or-minus0.6491.2𝑒-20.649\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2 | 0.857±3.2​e​-​3plus-or-minus0.8573.2𝑒-30.857\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3 | 0.474±5.1​e​-​3plus-or-minus0.4745.1𝑒-30.474\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3 | 3.165±1.8​e​-​2plus-or-minus3.1651.8𝑒-23.165\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2 | 0.859±1.3​e​-​3plus-or-minus0.8591.3𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.807±2.4​e​-​3plus-or-minus0.8072.4𝑒-30.807\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.725±1.9​e​-​3plus-or-minus0.7251.9𝑒-30.725\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 5.670±6.1​e​-​2plus-or-minus5.6706.1𝑒-25.670\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}2 | 0.924±3.0​e​-​4plus-or-minus0.9243.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 | 0.963±8.7​e​-​4plus-or-minus0.9638.7𝑒-40.963\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}4 | – |
| MLP-DICE | 0.610±1.2​e​-​2plus-or-minus0.6101.2𝑒-20.610\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2 | 0.858±2.9​e​-​3plus-or-minus0.8582.9𝑒-30.858\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 0.491±3.0​e​-​3plus-or-minus0.4913.0𝑒-30.491\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3 | 3.146±3.5​e​-​2plus-or-minus3.1463.5𝑒-23.146\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}2 | 0.860±1.4​e​-​3plus-or-minus0.8601.4𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.778±4.9​e​-​3plus-or-minus0.7784.9𝑒-30.778\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3 | 0.720±9.8​e​-​4plus-or-minus0.7209.8𝑒-40.720\scriptscriptstyle\pm\scriptstyle 9.8e\text{-}4 | 5.726±3.6​e​-​2plus-or-minus5.7263.6𝑒-25.726\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2 | 0.920±4.8​e​-​4plus-or-minus0.9204.8𝑒-40.920\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.964±1.1​e​-​3plus-or-minus0.9641.1𝑒-30.964\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.748±2.9​e​-​4plus-or-minus0.7482.9𝑒-40.748\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4 |
| ResNet | 0.655±2.0​e​-​2plus-or-minus0.6552.0𝑒-20.655\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.858±3.1​e​-​3plus-or-minus0.8583.1𝑒-30.858\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.490±5.0​e​-​3plus-or-minus0.4905.0𝑒-30.490\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}3 | 3.153±3.6​e​-​2plus-or-minus3.1533.6𝑒-23.153\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2 | 0.855±8.9​e​-​4plus-or-minus0.8558.9𝑒-40.855\scriptscriptstyle\pm\scriptstyle 8.9e\text{-}4 | 0.817±3.4​e​-​3plus-or-minus0.8173.4𝑒-30.817\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3 | 0.729±2.1​e​-​3plus-or-minus0.7292.1𝑒-30.729\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 5.681±5.3​e​-​2plus-or-minus5.6815.3𝑒-25.681\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}2 | 0.916±5.0​e​-​4plus-or-minus0.9165.0𝑒-40.916\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 | 0.965±8.3​e​-​4plus-or-minus0.9658.3𝑒-40.965\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4 | 0.747±4.1​e​-​4plus-or-minus0.7474.1𝑒-40.747\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 |
| ResNet-L | 0.644±1.9​e​-​2plus-or-minus0.6441.9𝑒-20.644\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2 | 0.859±1.8​e​-​3plus-or-minus0.8591.8𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.490±6.6​e​-​3plus-or-minus0.4906.6𝑒-30.490\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}3 | 3.126±5.6​e​-​2plus-or-minus3.1265.6𝑒-23.126\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}2 | 0.855±1.4​e​-​3plus-or-minus0.8551.4𝑒-30.855\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.813±2.2​e​-​3plus-or-minus0.8132.2𝑒-30.813\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.730±9.7​e​-​4plus-or-minus0.7309.7𝑒-40.730\scriptscriptstyle\pm\scriptstyle 9.7e\text{-}4 | 5.758±8.0​e​-​2plus-or-minus5.7588.0𝑒-25.758\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}2 | 0.915±4.3​e​-​4plus-or-minus0.9154.3𝑒-40.915\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4 | 0.964±1.8​e​-​3plus-or-minus0.9641.8𝑒-30.964\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.747±4.7​e​-​4plus-or-minus0.7474.7𝑒-40.747\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 |
| ResNet-LR | 0.635±2.3​e​-​2plus-or-minus0.6352.3𝑒-20.635\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2 | 0.861±2.2​e​-​3plus-or-minus0.8612.2𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.465±3.5​e​-​3plus-or-minus0.4653.5𝑒-30.465\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3 | 3.096±5.8​e​-​2plus-or-minus3.0965.8𝑒-23.096\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}2 | 0.856±1.6​e​-​3plus-or-minus0.8561.6𝑒-30.856\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.815±3.3​e​-​3plus-or-minus0.8153.3𝑒-30.815\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3 | 0.729±1.3​e​-​3plus-or-minus0.7291.3𝑒-30.729\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 5.574±7.4​e​-​2plus-or-minus5.5747.4𝑒-25.574\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}2 | 0.922±4.4​e​-​4plus-or-minus0.9224.4𝑒-40.922\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 | 0.967±8.8​e​-​4plus-or-minus0.9678.8𝑒-40.967\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}4 | 0.746±4.4​e​-​4plus-or-minus0.7464.4𝑒-40.746\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 |
| ResNet-Q | 0.658±8.0​e​-​3plus-or-minus0.6588.0𝑒-30.658\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}3 | 0.858±2.4​e​-​3plus-or-minus0.8582.4𝑒-30.858\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.454±3.6​e​-​3plus-or-minus0.4543.6𝑒-30.454\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}3 | 3.251±3.8​e​-​2plus-or-minus3.2513.8𝑒-23.251\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}2 | 0.860±1.3​e​-​3plus-or-minus0.8601.3𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.811±1.6​e​-​3plus-or-minus0.8111.6𝑒-30.811\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.718±1.0​e​-​3plus-or-minus0.7181.0𝑒-30.718\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 5.828±9.3​e​-​2plus-or-minus5.8289.3𝑒-25.828\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}2 | 0.921±9.1​e​-​4plus-or-minus0.9219.1𝑒-40.921\scriptscriptstyle\pm\scriptstyle 9.1e\text{-}4 | 0.970±5.7​e​-​4plus-or-minus0.9705.7𝑒-40.970\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}4 | 0.749±2.9​e​-​4plus-or-minus0.7492.9𝑒-40.749\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4 |
| ResNet-Q-LR | 0.650±9.2​e​-​3plus-or-minus0.6509.2𝑒-30.650\scriptscriptstyle\pm\scriptstyle 9.2e\text{-}3 | 0.854±4.2​e​-​3plus-or-minus0.8544.2𝑒-30.854\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3 | 0.446±5.1​e​-​3plus-or-minus0.4465.1𝑒-30.446\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3 | 3.217±5.2​e​-​2plus-or-minus3.2175.2𝑒-23.217\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}2 | 0.865±2.2​e​-​3plus-or-minus0.8652.2𝑒-30.865\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.808±2.6​e​-​3plus-or-minus0.8082.6𝑒-30.808\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.722±1.9​e​-​3plus-or-minus0.7221.9𝑒-30.722\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 5.514±6.0​e​-​2plus-or-minus5.5146.0𝑒-25.514\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}2 | 0.922±5.9​e​-​4plus-or-minus0.9225.9𝑒-40.922\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4 | 0.972±3.7​e​-​4plus-or-minus0.9723.7𝑒-40.972\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4 | 0.748±5.0​e​-​4plus-or-minus0.7485.0𝑒-40.748\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 |
| ResNet-T | 0.657±9.0​e​-​3plus-or-minus0.6579.0𝑒-30.657\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}3 | 0.859±2.9​e​-​3plus-or-minus0.8592.9𝑒-30.859\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 0.441±3.2​e​-​3plus-or-minus0.4413.2𝑒-30.441\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3 | 3.151±5.9​e​-​2plus-or-minus3.1515.9𝑒-23.151\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}2 | 0.866±1.8​e​-​3plus-or-minus0.8661.8𝑒-30.866\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.817±1.7​e​-​3plus-or-minus0.8171.7𝑒-30.817\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.724±2.0​e​-​3plus-or-minus0.7242.0𝑒-30.724\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 5.781±4.1​e​-​2plus-or-minus5.7814.1𝑒-25.781\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}2 | 0.923±6.0​e​-​4plus-or-minus0.9236.0𝑒-40.923\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4 | 0.970±1.1​e​-​3plus-or-minus0.9701.1𝑒-30.970\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.749±7.8​e​-​4plus-or-minus0.7497.8𝑒-40.749\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4 |
| ResNet-T-LR | 0.650±1.2​e​-​2plus-or-minus0.6501.2𝑒-20.650\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2 | 0.861±2.0​e​-​3plus-or-minus0.8612.0𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.438±2.9​e​-​3plus-or-minus0.4382.9𝑒-30.438\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 3.163±6.1​e​-​2plus-or-minus3.1636.1𝑒-23.163\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}2 | 0.870±1.5​e​-​3plus-or-minus0.8701.5𝑒-30.870\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.813±2.5​e​-​3plus-or-minus0.8132.5𝑒-30.813\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.725±1.6​e​-​3plus-or-minus0.7251.6𝑒-30.725\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.687±5.9​e​-​2plus-or-minus5.6875.9𝑒-25.687\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}2 | 0.922±8.1​e​-​4plus-or-minus0.9228.1𝑒-40.922\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}4 | 0.972±3.7​e​-​4plus-or-minus0.9723.7𝑒-40.972\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4 | 0.748±6.1​e​-​4plus-or-minus0.7486.1𝑒-40.748\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4 |
| ResNet-P | 0.630±1.8​e​-​2plus-or-minus0.6301.8𝑒-20.630\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2 | 0.858±3.1​e​-​3plus-or-minus0.8583.1𝑒-30.858\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.471±6.5​e​-​3plus-or-minus0.4716.5𝑒-30.471\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}3 | 3.147±2.9​e​-​2plus-or-minus3.1472.9𝑒-23.147\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2 | 0.866±1.7​e​-​3plus-or-minus0.8661.7𝑒-30.866\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.812±1.6​e​-​3plus-or-minus0.8121.6𝑒-30.812\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.729±7.0​e​-​4plus-or-minus0.7297.0𝑒-40.729\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4 | 5.566±7.5​e​-​2plus-or-minus5.5667.5𝑒-25.566\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}2 | 0.922±6.7​e​-​4plus-or-minus0.9226.7𝑒-40.922\scriptscriptstyle\pm\scriptstyle 6.7e\text{-}4 | 0.968±7.7​e​-​4plus-or-minus0.9687.7𝑒-40.968\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}4 | 0.747±6.3​e​-​4plus-or-minus0.7476.3𝑒-40.747\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 |
| ResNet-PLR | 0.651±1.3​e​-​2plus-or-minus0.6511.3𝑒-20.651\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.859±3.7​e​-​3plus-or-minus0.8593.7𝑒-30.859\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3 | 0.461±4.2​e​-​3plus-or-minus0.4614.2𝑒-30.461\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3 | 3.188±7.3​e​-​2plus-or-minus3.1887.3𝑒-23.188\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}2 | 0.869±1.7​e​-​3plus-or-minus0.8691.7𝑒-30.869\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.816±2.5​e​-​3plus-or-minus0.8162.5𝑒-30.816\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.728±1.8​e​-​3plus-or-minus0.7281.8𝑒-30.728\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 5.582±4.9​e​-​2plus-or-minus5.5824.9𝑒-25.582\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}2 | 0.923±5.9​e​-​4plus-or-minus0.9235.9𝑒-40.923\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4 | 0.972±5.1​e​-​4plus-or-minus0.9725.1𝑒-40.972\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4 | 0.747±6.4​e​-​4plus-or-minus0.7476.4𝑒-40.747\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}4 |
| Transformer-L | 0.632±2.0​e​-​2plus-or-minus0.6322.0𝑒-20.632\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.860±3.0​e​-​3plus-or-minus0.8603.0𝑒-30.860\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3 | 0.465±4.8​e​-​3plus-or-minus0.4654.8𝑒-30.465\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3 | 3.239±3.2​e​-​2plus-or-minus3.2393.2𝑒-23.239\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}2 | 0.858±1.3​e​-​3plus-or-minus0.8581.3𝑒-30.858\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.817±2.3​e​-​3plus-or-minus0.8172.3𝑒-30.817\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 0.725±3.2​e​-​3plus-or-minus0.7253.2𝑒-30.725\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3 | 5.602±4.8​e​-​2plus-or-minus5.6024.8𝑒-25.602\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}2 | 0.924±4.4​e​-​4plus-or-minus0.9244.4𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 | 0.971±6.8​e​-​4plus-or-minus0.9716.8𝑒-40.971\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}4 | 0.746±5.7​e​-​4plus-or-minus0.7465.7𝑒-40.746\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}4 |
| Transformer-LR | 0.614±4.5​e​-​2plus-or-minus0.6144.5𝑒-20.614\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2 | 0.860±2.2​e​-​3plus-or-minus0.8602.2𝑒-30.860\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.456±3.7​e​-​3plus-or-minus0.4563.7𝑒-30.456\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3 | 3.261±5.6​e​-​2plus-or-minus3.2615.6𝑒-23.261\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}2 | 0.858±1.6​e​-​3plus-or-minus0.8581.6𝑒-30.858\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.817±2.2​e​-​3plus-or-minus0.8172.2𝑒-30.817\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 0.729±1.5​e​-​3plus-or-minus0.7291.5𝑒-30.729\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 5.644±5.5​e​-​2plus-or-minus5.6445.5𝑒-25.644\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}2 | 0.924±3.9​e​-​4plus-or-minus0.9243.9𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4 | 0.971±7.6​e​-​4plus-or-minus0.9717.6𝑒-40.971\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}4 | 0.746±5.8​e​-​4plus-or-minus0.7465.8𝑒-40.746\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 |
| Transformer-Q-L | 0.659±8.7​e​-​3plus-or-minus0.6598.7𝑒-30.659\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}3 | 0.856±5.9​e​-​3plus-or-minus0.8565.9𝑒-30.856\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}3 | 0.451±5.4​e​-​3plus-or-minus0.4515.4𝑒-30.451\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}3 | 3.319±4.2​e​-​2plus-or-minus3.3194.2𝑒-23.319\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}2 | 0.867±1.6​e​-​3plus-or-minus0.8671.6𝑒-30.867\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.812±2.6​e​-​3plus-or-minus0.8122.6𝑒-30.812\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.729±2.9​e​-​3plus-or-minus0.7292.9𝑒-30.729\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 5.741±4.5​e​-​2plus-or-minus5.7414.5𝑒-25.741\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2 | 0.924±3.8​e​-​4plus-or-minus0.9243.8𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}4 | 0.973±6.1​e​-​4plus-or-minus0.9736.1𝑒-40.973\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4 | 0.747±7.9​e​-​4plus-or-minus0.7477.9𝑒-40.747\scriptscriptstyle\pm\scriptstyle 7.9e\text{-}4 |
| Transformer-Q-LR | 0.659±1.2​e​-​2plus-or-minus0.6591.2𝑒-20.659\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2 | 0.857±2.0​e​-​3plus-or-minus0.8572.0𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.448±6.1​e​-​3plus-or-minus0.4486.1𝑒-30.448\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}3 | 3.270±4.6​e​-​2plus-or-minus3.2704.6𝑒-23.270\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2 | 0.867±1.1​e​-​3plus-or-minus0.8671.1𝑒-30.867\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.812±2.5​e​-​3plus-or-minus0.8122.5𝑒-30.812\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.723±3.3​e​-​3plus-or-minus0.7233.3𝑒-30.723\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3 | 5.683±4.8​e​-​2plus-or-minus5.6834.8𝑒-25.683\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}2 | 0.923±5.8​e​-​4plus-or-minus0.9235.8𝑒-40.923\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 | 0.972±4.2​e​-​4plus-or-minus0.9724.2𝑒-40.972\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}4 | 0.748±7.7​e​-​4plus-or-minus0.7487.7𝑒-40.748\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}4 |
| Transformer-T-L | 0.663±7.4​e​-​3plus-or-minus0.6637.4𝑒-30.663\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}3 | 0.861±1.4​e​-​3plus-or-minus0.8611.4𝑒-30.861\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.454±4.7​e​-​3plus-or-minus0.4544.7𝑒-30.454\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3 | 3.197±2.9​e​-​2plus-or-minus3.1972.9𝑒-23.197\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2 | 0.871±1.4​e​-​3plus-or-minus0.8711.4𝑒-30.871\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.817±2.6​e​-​3plus-or-minus0.8172.6𝑒-30.817\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.726±1.7​e​-​3plus-or-minus0.7261.7𝑒-30.726\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 5.803±6.5​e​-​2plus-or-minus5.8036.5𝑒-25.803\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}2 | 0.924±3.3​e​-​4plus-or-minus0.9243.3𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4 | 0.974±4.5​e​-​4plus-or-minus0.9744.5𝑒-40.974\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}4 | 0.747±6.5​e​-​4plus-or-minus0.7476.5𝑒-40.747\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}4 |
| Transformer-T-LR | 0.665±6.6​e​-​3plus-or-minus0.6656.6𝑒-30.665\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}3 | 0.860±3.4​e​-​3plus-or-minus0.8603.4𝑒-30.860\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3 | 0.442±5.3​e​-​3plus-or-minus0.4425.3𝑒-30.442\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}3 | 3.219±3.2​e​-​2plus-or-minus3.2193.2𝑒-23.219\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}2 | 0.870±1.5​e​-​3plus-or-minus0.8701.5𝑒-30.870\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.818±2.6​e​-​3plus-or-minus0.8182.6𝑒-30.818\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.729±1.4​e​-​3plus-or-minus0.7291.4𝑒-30.729\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 5.699±6.7​e​-​2plus-or-minus5.6996.7𝑒-25.699\scriptscriptstyle\pm\scriptstyle 6.7e\text{-}2 | 0.924±4.4​e​-​4plus-or-minus0.9244.4𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 | 0.973±5.6​e​-​4plus-or-minus0.9735.6𝑒-40.973\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4 | 0.747±8.4​e​-​4plus-or-minus0.7478.4𝑒-40.747\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}4 |
| Transformer-PLR | 0.646±2.0​e​-​2plus-or-minus0.6462.0𝑒-20.646\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.863±2.7​e​-​3plus-or-minus0.8632.7𝑒-30.863\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3 | 0.464±2.8​e​-​3plus-or-minus0.4642.8𝑒-30.464\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}3 | 3.162±4.2​e​-​2plus-or-minus3.1624.2𝑒-23.162\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}2 | 0.870±1.5​e​-​3plus-or-minus0.8701.5𝑒-30.870\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.814±2.1​e​-​3plus-or-minus0.8142.1𝑒-30.814\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.730±1.9​e​-​3plus-or-minus0.7301.9𝑒-30.730\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 5.760±1.1​e​-​1plus-or-minus5.7601.1𝑒-15.760\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}1 | 0.924±5.2​e​-​4plus-or-minus0.9245.2𝑒-40.924\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4 | 0.972±1.1​e​-​3plus-or-minus0.9721.1𝑒-30.972\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.746±5.9​e​-​4plus-or-minus0.7465.9𝑒-40.746\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4 |

Table 19: Extended results for ensembles

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | AD ↑ | OT ↑ | HI ↑ | FB ↓ | SA ↑ | CO ↑ | MI ↓ |
| CatBoost | 0.692±1.9​e​-​3plus-or-minus0.6921.9𝑒-30.692\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 0.861±2.4​e​-​4plus-or-minus0.8612.4𝑒-40.861\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 | 0.430±1.1​e​-​3plus-or-minus0.4301.1𝑒-30.430\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 3.093±5.1​e​-​3plus-or-minus3.0935.1𝑒-33.093\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3 | 0.873±5.1​e​-​4plus-or-minus0.8735.1𝑒-40.873\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4 | 0.825±4.7​e​-​4plus-or-minus0.8254.7𝑒-40.825\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 | 0.727±3.6​e​-​4plus-or-minus0.7273.6𝑒-40.727\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4 | 5.226±1.3​e​-​2plus-or-minus5.2261.3𝑒-25.226\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.924±1.0​e​-​4plus-or-minus0.9241.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4 | 0.967±1.4​e​-​4plus-or-minus0.9671.4𝑒-40.967\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4 | 0.741±1.4​e​-​4plus-or-minus0.7411.4𝑒-40.741\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4 |
| XGBoost | 0.683±1.3​e​-​3plus-or-minus0.6831.3𝑒-30.683\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.859±2.4​e​-​4plus-or-minus0.8592.4𝑒-40.859\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 | 0.434±7.1​e​-​4plus-or-minus0.4347.1𝑒-40.434\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}4 | 3.152±1.2​e​-​3plus-or-minus3.1521.2𝑒-33.152\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 0.875±5.5​e​-​4plus-or-minus0.8755.5𝑒-40.875\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}4 | 0.827±8.4​e​-​4plus-or-minus0.8278.4𝑒-40.827\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}4 | 0.726±8.1​e​-​4plus-or-minus0.7268.1𝑒-40.726\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}4 | 5.338±1.9​e​-​2plus-or-minus5.3381.9𝑒-25.338\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2 | 0.919±4.8​e​-​4plus-or-minus0.9194.8𝑒-40.919\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.969±8.8​e​-​5plus-or-minus0.9698.8𝑒-50.969\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}5 | 0.742±5.3​e​-​5plus-or-minus0.7425.3𝑒-50.742\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}5 |
| MLP | 0.665±2.7​e​-​3plus-or-minus0.6652.7𝑒-30.665\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3 | 0.856±1.2​e​-​3plus-or-minus0.8561.2𝑒-30.856\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 0.486±7.8​e​-​4plus-or-minus0.4867.8𝑒-40.486\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4 | 3.109±1.0​e​-​2plus-or-minus3.1091.0𝑒-23.109\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2 | 0.856±4.6​e​-​4plus-or-minus0.8564.6𝑒-40.856\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4 | 0.822±8.0​e​-​4plus-or-minus0.8228.0𝑒-40.822\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4 | 0.727±1.7​e​-​3plus-or-minus0.7271.7𝑒-30.727\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 5.616±7.6​e​-​3plus-or-minus5.6167.6𝑒-35.616\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}3 | 0.913±8.2​e​-​5plus-or-minus0.9138.2𝑒-50.913\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}5 | 0.968±4.8​e​-​4plus-or-minus0.9684.8𝑒-40.968\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.746±1.1​e​-​4plus-or-minus0.7461.1𝑒-40.746\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4 |
| MLP-L | 0.670±3.2​e​-​3plus-or-minus0.6703.2𝑒-30.670\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3 | 0.862±1.5​e​-​3plus-or-minus0.8621.5𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.471±4.7​e​-​4plus-or-minus0.4714.7𝑒-40.471\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 | 3.021±1.1​e​-​2plus-or-minus3.0211.1𝑒-23.021\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2 | 0.857±5.9​e​-​4plus-or-minus0.8575.9𝑒-40.857\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4 | 0.824±1.1​e​-​3plus-or-minus0.8241.1𝑒-30.824\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.728±2.7​e​-​4plus-or-minus0.7282.7𝑒-40.728\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4 | 5.508±2.1​e​-​2plus-or-minus5.5082.1𝑒-25.508\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2 | 0.916±1.5​e​-​4plus-or-minus0.9161.5𝑒-40.916\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4 | 0.971±6.8​e​-​5plus-or-minus0.9716.8𝑒-50.971\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}5 | 0.746±2.3​e​-​4plus-or-minus0.7462.3𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4 |
| MLP-LR | 0.679±4.9​e​-​3plus-or-minus0.6794.9𝑒-30.679\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3 | 0.861±9.4​e​-​4plus-or-minus0.8619.4𝑒-40.861\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4 | 0.463±1.9​e​-​3plus-or-minus0.4631.9𝑒-30.463\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 3.012±1.8​e​-​3plus-or-minus3.0121.8𝑒-33.012\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.859±8.0​e​-​4plus-or-minus0.8598.0𝑒-40.859\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4 | 0.826±1.6​e​-​3plus-or-minus0.8261.6𝑒-30.826\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.731±1.1​e​-​3plus-or-minus0.7311.1𝑒-30.731\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 5.477±3.6​e​-​2plus-or-minus5.4773.6𝑒-25.477\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2 | 0.924±7.1​e​-​5plus-or-minus0.9247.1𝑒-50.924\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}5 | 0.972±7.6​e​-​5plus-or-minus0.9727.6𝑒-50.972\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}5 | 0.744±1.6​e​-​4plus-or-minus0.7441.6𝑒-40.744\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4 |
| MLP-LRLR | 0.676±4.8​e​-​3plus-or-minus0.6764.8𝑒-30.676\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3 | 0.863±1.4​e​-​3plus-or-minus0.8631.4𝑒-30.863\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.453±1.1​e​-​3plus-or-minus0.4531.1𝑒-30.453\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 3.017±1.1​e​-​2plus-or-minus3.0171.1𝑒-23.017\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2 | 0.858±1.6​e​-​4plus-or-minus0.8581.6𝑒-40.858\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4 | 0.828±1.3​e​-​3plus-or-minus0.8281.3𝑒-30.828\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.725±5.9​e​-​4plus-or-minus0.7255.9𝑒-40.725\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4 | 5.427±2.1​e​-​2plus-or-minus5.4272.1𝑒-25.427\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2 | 0.924±1.2​e​-​4plus-or-minus0.9241.2𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4 | 0.973±1.8​e​-​4plus-or-minus0.9731.8𝑒-40.973\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 | 0.744±1.8​e​-​4plus-or-minus0.7441.8𝑒-40.744\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 |
| MLP-Q | 0.677±4.8​e​-​3plus-or-minus0.6774.8𝑒-30.677\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3 | 0.856±1.4​e​-​3plus-or-minus0.8561.4𝑒-30.856\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.458±1.7​e​-​4plus-or-minus0.4581.7𝑒-40.458\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4 | 3.080±1.5​e​-​2plus-or-minus3.0801.5𝑒-23.080\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2 | 0.862±4.0​e​-​4plus-or-minus0.8624.0𝑒-40.862\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 | 0.822±1.7​e​-​3plus-or-minus0.8221.7𝑒-30.822\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.723±5.6​e​-​4plus-or-minus0.7235.6𝑒-40.723\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4 | 5.706±1.9​e​-​2plus-or-minus5.7061.9𝑒-25.706\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2 | 0.922±1.7​e​-​4plus-or-minus0.9221.7𝑒-40.922\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4 | 0.973±2.1​e​-​4plus-or-minus0.9732.1𝑒-40.973\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 0.748±2.2​e​-​4plus-or-minus0.7482.2𝑒-40.748\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4 |
| MLP-Q-LR | 0.682±3.9​e​-​3plus-or-minus0.6823.9𝑒-30.682\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}3 | 0.859±4.7​e​-​4plus-or-minus0.8594.7𝑒-40.859\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 | 0.433±1.9​e​-​3plus-or-minus0.4331.9𝑒-30.433\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 3.080±9.7​e​-​3plus-or-minus3.0809.7𝑒-33.080\scriptscriptstyle\pm\scriptstyle 9.7e\text{-}3 | 0.867±4.2​e​-​4plus-or-minus0.8674.2𝑒-40.867\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}4 | 0.818±1.4​e​-​3plus-or-minus0.8181.4𝑒-30.818\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.724±3.2​e​-​4plus-or-minus0.7243.2𝑒-40.724\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}4 | 5.144±1.4​e​-​2plus-or-minus5.1441.4𝑒-25.144\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2 | 0.924±3.7​e​-​4plus-or-minus0.9243.7𝑒-40.924\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4 | 0.974±1.5​e​-​4plus-or-minus0.9741.5𝑒-40.974\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4 | 0.745±2.8​e​-​4plus-or-minus0.7452.8𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 |
| MLP-Q-LRLR | 0.674±2.9​e​-​3plus-or-minus0.6742.9𝑒-30.674\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3 | 0.862±1.5​e​-​3plus-or-minus0.8621.5𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.438±2.1​e​-​3plus-or-minus0.4382.1𝑒-30.438\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 3.066±9.9​e​-​3plus-or-minus3.0669.9𝑒-33.066\scriptscriptstyle\pm\scriptstyle 9.9e\text{-}3 | 0.870±4.1​e​-​4plus-or-minus0.8704.1𝑒-40.870\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.817±2.4​e​-​3plus-or-minus0.8172.4𝑒-30.817\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.727±2.1​e​-​4plus-or-minus0.7272.1𝑒-40.727\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 5.268±7.5​e​-​2plus-or-minus5.2687.5𝑒-25.268\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}2 | 0.924±3.1​e​-​5plus-or-minus0.9243.1𝑒-50.924\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}5 | 0.973±2.8​e​-​4plus-or-minus0.9732.8𝑒-40.973\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 | 0.745±1.7​e​-​4plus-or-minus0.7451.7𝑒-40.745\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4 |
| MLP-T | 0.669±4.3​e​-​3plus-or-minus0.6694.3𝑒-30.669\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3 | 0.861±1.0​e​-​3plus-or-minus0.8611.0𝑒-30.861\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 0.439±2.1​e​-​4plus-or-minus0.4392.1𝑒-40.439\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 3.058±1.4​e​-​2plus-or-minus3.0581.4𝑒-23.058\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2 | 0.865±5.3​e​-​4plus-or-minus0.8655.3𝑒-40.865\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4 | 0.822±6.3​e​-​4plus-or-minus0.8226.3𝑒-40.822\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 | 0.724±7.2​e​-​4plus-or-minus0.7247.2𝑒-40.724\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}4 | 5.507±2.0​e​-​2plus-or-minus5.5072.0𝑒-25.507\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.923±8.5​e​-​5plus-or-minus0.9238.5𝑒-50.923\scriptscriptstyle\pm\scriptstyle 8.5e\text{-}5 | 0.972±2.7​e​-​4plus-or-minus0.9722.7𝑒-40.972\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4 | 0.747±4.1​e​-​5plus-or-minus0.7474.1𝑒-50.747\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}5 |
| MLP-T-LR | 0.673±8.3​e​-​4plus-or-minus0.6738.3𝑒-40.673\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4 | 0.861±8.5​e​-​4plus-or-minus0.8618.5𝑒-40.861\scriptscriptstyle\pm\scriptstyle 8.5e\text{-}4 | 0.435±1.1​e​-​3plus-or-minus0.4351.1𝑒-30.435\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 3.099±2.4​e​-​2plus-or-minus3.0992.4𝑒-23.099\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2 | 0.870±6.6​e​-​4plus-or-minus0.8706.6𝑒-40.870\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}4 | 0.821±2.6​e​-​4plus-or-minus0.8212.6𝑒-40.821\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}4 | 0.727±7.2​e​-​4plus-or-minus0.7277.2𝑒-40.727\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}4 | 5.409±6.2​e​-​3plus-or-minus5.4096.2𝑒-35.409\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}3 | 0.924±1.3​e​-​4plus-or-minus0.9241.3𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4 | 0.973±1.3​e​-​4plus-or-minus0.9731.3𝑒-40.973\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4 | 0.746±1.6​e​-​4plus-or-minus0.7461.6𝑒-40.746\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4 |
| MLP-T-LRLR | 0.670±4.1​e​-​4plus-or-minus0.6704.1𝑒-40.670\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.860±2.5​e​-​3plus-or-minus0.8602.5𝑒-30.860\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.431±6.0​e​-​4plus-or-minus0.4316.0𝑒-40.431\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4 | 3.056±2.2​e​-​2plus-or-minus3.0562.2𝑒-23.056\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}2 | 0.870±2.6​e​-​4plus-or-minus0.8702.6𝑒-40.870\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}4 | 0.826±5.0​e​-​4plus-or-minus0.8265.0𝑒-40.826\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 | 0.725±7.4​e​-​4plus-or-minus0.7257.4𝑒-40.725\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}4 | 5.440±1.8​e​-​3plus-or-minus5.4401.8𝑒-35.440\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.925±6.1​e​-​5plus-or-minus0.9256.1𝑒-50.925\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}5 | 0.973±2.2​e​-​4plus-or-minus0.9732.2𝑒-40.973\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4 | 0.745±4.7​e​-​4plus-or-minus0.7454.7𝑒-40.745\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 |
| MLP-P | 0.661±6.0​e​-​3plus-or-minus0.6616.0𝑒-30.661\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}3 | 0.861±6.2​e​-​4plus-or-minus0.8616.2𝑒-40.861\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4 | 0.473±1.1​e​-​3plus-or-minus0.4731.1𝑒-30.473\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 3.042±1.0​e​-​2plus-or-minus3.0421.0𝑒-23.042\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2 | 0.871±1.1​e​-​3plus-or-minus0.8711.1𝑒-30.871\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.812±1.7​e​-​3plus-or-minus0.8121.7𝑒-30.812\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.725±6.2​e​-​4plus-or-minus0.7256.2𝑒-40.725\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4 | 5.508±3.1​e​-​2plus-or-minus5.5083.1𝑒-25.508\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2 | 0.924±5.4​e​-​5plus-or-minus0.9245.4𝑒-50.924\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}5 | 0.973±3.0​e​-​4plus-or-minus0.9733.0𝑒-40.973\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 | 0.745±2.1​e​-​4plus-or-minus0.7452.1𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 |
| MLP-PL | 0.671±6.2​e​-​3plus-or-minus0.6716.2𝑒-30.671\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}3 | 0.860±1.2​e​-​3plus-or-minus0.8601.2𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 0.456±1.3​e​-​3plus-or-minus0.4561.3𝑒-30.456\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 3.065±8.1​e​-​3plus-or-minus3.0658.1𝑒-33.065\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}3 | 0.872±6.3​e​-​4plus-or-minus0.8726.3𝑒-40.872\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 | 0.825±4.1​e​-​4plus-or-minus0.8254.1𝑒-40.825\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.730±3.5​e​-​4plus-or-minus0.7303.5𝑒-40.730\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4 | 5.216±2.0​e​-​2plus-or-minus5.2162.0𝑒-25.216\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.924±1.2​e​-​4plus-or-minus0.9241.2𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4 | 0.974±1.9​e​-​4plus-or-minus0.9741.9𝑒-40.974\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}4 | 0.744±2.1​e​-​4plus-or-minus0.7442.1𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 |
| MLP-PLR | 0.700±2.1​e​-​3plus-or-minus0.7002.1𝑒-30.700\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.858±1.6​e​-​3plus-or-minus0.8581.6𝑒-30.858\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.453±5.8​e​-​4plus-or-minus0.4535.8𝑒-40.453\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 | 2.975±6.6​e​-​3plus-or-minus2.9756.6𝑒-32.975\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}3 | 0.874±9.0​e​-​4plus-or-minus0.8749.0𝑒-40.874\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}4 | 0.830±2.4​e​-​3plus-or-minus0.8302.4𝑒-30.830\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.734±3.5​e​-​4plus-or-minus0.7343.5𝑒-40.734\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4 | 5.388±1.6​e​-​2plus-or-minus5.3881.6𝑒-25.388\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2 | 0.924±5.4​e​-​5plus-or-minus0.9245.4𝑒-50.924\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}5 | 0.975±4.8​e​-​4plus-or-minus0.9754.8𝑒-40.975\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.743±1.0​e​-​4plus-or-minus0.7431.0𝑒-40.743\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4 |
| MLP-PLRLR | 0.699±9.3​e​-​3plus-or-minus0.6999.3𝑒-30.699\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}3 | 0.867±1.8​e​-​3plus-or-minus0.8671.8𝑒-30.867\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.448±8.3​e​-​4plus-or-minus0.4488.3𝑒-40.448\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4 | 2.993±6.5​e​-​3plus-or-minus2.9936.5𝑒-32.993\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}3 | 0.873±4.1​e​-​4plus-or-minus0.8734.1𝑒-40.873\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.823±8.3​e​-​4plus-or-minus0.8238.3𝑒-40.823\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4 | 0.729±9.1​e​-​4plus-or-minus0.7299.1𝑒-40.729\scriptscriptstyle\pm\scriptstyle 9.1e\text{-}4 | 5.346±4.8​e​-​2plus-or-minus5.3464.8𝑒-25.346\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}2 | 0.924±2.6​e​-​4plus-or-minus0.9242.6𝑒-40.924\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}4 | 0.972±8.2​e​-​4plus-or-minus0.9728.2𝑒-40.972\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4 | 0.743±9.9​e​-​5plus-or-minus0.7439.9𝑒-50.743\scriptscriptstyle\pm\scriptstyle 9.9e\text{-}5 |
| MLP-AutoDis | 0.676±7.6​e​-​3plus-or-minus0.6767.6𝑒-30.676\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}3 | 0.860±1.7​e​-​3plus-or-minus0.8601.7𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.464±1.6​e​-​3plus-or-minus0.4641.6𝑒-30.464\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 3.132±5.7​e​-​3plus-or-minus3.1325.7𝑒-33.132\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}3 | 0.860±2.8​e​-​4plus-or-minus0.8602.8𝑒-40.860\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 | 0.817±2.1​e​-​3plus-or-minus0.8172.1𝑒-30.817\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.730±2.5​e​-​4plus-or-minus0.7302.5𝑒-40.730\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4 | 5.580±2.2​e​-​2plus-or-minus5.5802.2𝑒-25.580\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}2 | 0.924±1.1​e​-​4plus-or-minus0.9241.1𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4 | 0.970±3.2​e​-​4plus-or-minus0.9703.2𝑒-40.970\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}4 | – |
| MLP-DICE | 0.636±2.6​e​-​3plus-or-minus0.6362.6𝑒-30.636\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3 | 0.859±2.0​e​-​3plus-or-minus0.8592.0𝑒-30.859\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3 | 0.486±1.4​e​-​3plus-or-minus0.4861.4𝑒-30.486\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 3.092±1.3​e​-​2plus-or-minus3.0921.3𝑒-23.092\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.862±4.8​e​-​4plus-or-minus0.8624.8𝑒-40.862\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4 | 0.784±2.3​e​-​3plus-or-minus0.7842.3𝑒-30.784\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 0.723±6.1​e​-​4plus-or-minus0.7236.1𝑒-40.723\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4 | 5.615±8.9​e​-​3plus-or-minus5.6158.9𝑒-35.615\scriptscriptstyle\pm\scriptstyle 8.9e\text{-}3 | 0.920±2.0​e​-​4plus-or-minus0.9202.0𝑒-40.920\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 | 0.969±1.4​e​-​4plus-or-minus0.9691.4𝑒-40.969\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4 | 0.746±2.0​e​-​4plus-or-minus0.7462.0𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 |
| ResNet | 0.690±5.9​e​-​3plus-or-minus0.6905.9𝑒-30.690\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}3 | 0.861±1.6​e​-​3plus-or-minus0.8611.6𝑒-30.861\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.483±1.8​e​-​3plus-or-minus0.4831.8𝑒-30.483\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 3.081±7.8​e​-​3plus-or-minus3.0817.8𝑒-33.081\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}3 | 0.856±3.4​e​-​4plus-or-minus0.8563.4𝑒-40.856\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4 | 0.821±1.8​e​-​3plus-or-minus0.8211.8𝑒-30.821\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.734±1.1​e​-​3plus-or-minus0.7341.1𝑒-30.734\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 5.482±1.1​e​-​2plus-or-minus5.4821.1𝑒-25.482\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2 | 0.918±5.3​e​-​4plus-or-minus0.9185.3𝑒-40.918\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4 | 0.968±4.4​e​-​4plus-or-minus0.9684.4𝑒-40.968\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4 | 0.745±6.5​e​-​5plus-or-minus0.7456.5𝑒-50.745\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}5 |
| ResNet-L | 0.674±5.2​e​-​3plus-or-minus0.6745.2𝑒-30.674\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}3 | 0.859±6.2​e​-​4plus-or-minus0.8596.2𝑒-40.859\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4 | 0.481±2.5​e​-​3plus-or-minus0.4812.5𝑒-30.481\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 3.025±1.8​e​-​2plus-or-minus3.0251.8𝑒-23.025\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2 | 0.857±2.9​e​-​4plus-or-minus0.8572.9𝑒-40.857\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4 | 0.819±1.3​e​-​3plus-or-minus0.8191.3𝑒-30.819\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.735±5.2​e​-​4plus-or-minus0.7355.2𝑒-40.735\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4 | 5.522±2.4​e​-​2plus-or-minus5.5222.4𝑒-25.522\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2 | 0.917±2.2​e​-​4plus-or-minus0.9172.2𝑒-40.917\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4 | 0.966±5.1​e​-​4plus-or-minus0.9665.1𝑒-40.966\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4 | 0.744±3.0​e​-​4plus-or-minus0.7443.0𝑒-40.744\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4 |
| ResNet-LR | 0.672±6.0​e​-​3plus-or-minus0.6726.0𝑒-30.672\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}3 | 0.862±1.7​e​-​3plus-or-minus0.8621.7𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.450±2.2​e​-​3plus-or-minus0.4502.2𝑒-30.450\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3 | 2.992±2.4​e​-​2plus-or-minus2.9922.4𝑒-22.992\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2 | 0.859±4.7​e​-​4plus-or-minus0.8594.7𝑒-40.859\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4 | 0.822±9.2​e​-​4plus-or-minus0.8229.2𝑒-40.822\scriptscriptstyle\pm\scriptstyle 9.2e\text{-}4 | 0.733±4.2​e​-​5plus-or-minus0.7334.2𝑒-50.733\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}5 | 5.415±9.5​e​-​5plus-or-minus5.4159.5𝑒-55.415\scriptscriptstyle\pm\scriptstyle 9.5e\text{-}5 | 0.923±7.7​e​-​5plus-or-minus0.9237.7𝑒-50.923\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}5 | 0.971±1.5​e​-​4plus-or-minus0.9711.5𝑒-40.971\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4 | 0.743±2.1​e​-​4plus-or-minus0.7432.1𝑒-40.743\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 |
| ResNet-Q | 0.671±1.7​e​-​3plus-or-minus0.6711.7𝑒-30.671\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.862±8.2​e​-​4plus-or-minus0.8628.2𝑒-40.862\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4 | 0.442±8.0​e​-​4plus-or-minus0.4428.0𝑒-40.442\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4 | 3.128±9.0​e​-​3plus-or-minus3.1289.0𝑒-33.128\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}3 | 0.862±5.8​e​-​4plus-or-minus0.8625.8𝑒-40.862\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 | 0.816±9.4​e​-​4plus-or-minus0.8169.4𝑒-40.816\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4 | 0.722±7.1​e​-​4plus-or-minus0.7227.1𝑒-40.722\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}4 | 5.402±3.3​e​-​2plus-or-minus5.4023.3𝑒-25.402\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}2 | 0.923±4.6​e​-​4plus-or-minus0.9234.6𝑒-40.923\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4 | 0.974±6.3​e​-​5plus-or-minus0.9746.3𝑒-50.974\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}5 | 0.746±2.4​e​-​4plus-or-minus0.7462.4𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 |
| ResNet-Q-LR | 0.674±2.5​e​-​3plus-or-minus0.6742.5𝑒-30.674\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3 | 0.859±1.8​e​-​3plus-or-minus0.8591.8𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3 | 0.427±2.3​e​-​3plus-or-minus0.4272.3𝑒-30.427\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 3.066±2.2​e​-​2plus-or-minus3.0662.2𝑒-23.066\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}2 | 0.868±1.1​e​-​3plus-or-minus0.8681.1𝑒-30.868\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.815±7.1​e​-​4plus-or-minus0.8157.1𝑒-40.815\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}4 | 0.729±1.6​e​-​3plus-or-minus0.7291.6𝑒-30.729\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.309±4.9​e​-​2plus-or-minus5.3094.9𝑒-25.309\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}2 | 0.923±3.9​e​-​4plus-or-minus0.9233.9𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4 | 0.976±1.2​e​-​4plus-or-minus0.9761.2𝑒-40.976\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4 | 0.746±1.8​e​-​4plus-or-minus0.7461.8𝑒-40.746\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 |
| ResNet-T | 0.681±1.3​e​-​3plus-or-minus0.6811.3𝑒-30.681\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3 | 0.861±2.1​e​-​3plus-or-minus0.8612.1𝑒-30.861\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3 | 0.428±8.0​e​-​4plus-or-minus0.4288.0𝑒-40.428\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4 | 3.064±3.6​e​-​2plus-or-minus3.0643.6𝑒-23.064\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2 | 0.868±8.3​e​-​4plus-or-minus0.8688.3𝑒-40.868\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4 | 0.823±4.0​e​-​4plus-or-minus0.8234.0𝑒-40.823\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 | 0.725±9.5​e​-​4plus-or-minus0.7259.5𝑒-40.725\scriptscriptstyle\pm\scriptstyle 9.5e\text{-}4 | 5.657±1.5​e​-​2plus-or-minus5.6571.5𝑒-25.657\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2 | 0.923±1.0​e​-​4plus-or-minus0.9231.0𝑒-40.923\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4 | 0.973±6.0​e​-​4plus-or-minus0.9736.0𝑒-40.973\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4 | 0.746±6.0​e​-​4plus-or-minus0.7466.0𝑒-40.746\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4 |
| ResNet-T-LR | 0.683±6.1​e​-​3plus-or-minus0.6836.1𝑒-30.683\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}3 | 0.862±0.0​e+00plus-or-minus0.8620.0𝑒000.862\scriptscriptstyle\pm\scriptstyle 0.0e+00 | 0.425±7.4​e​-​4plus-or-minus0.4257.4𝑒-40.425\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}4 | 3.030±3.4​e​-​2plus-or-minus3.0303.4𝑒-23.030\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}2 | 0.872±7.3​e​-​4plus-or-minus0.8727.3𝑒-40.872\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}4 | 0.822±5.5​e​-​4plus-or-minus0.8225.5𝑒-40.822\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}4 | 0.731±1.1​e​-​3plus-or-minus0.7311.1𝑒-30.731\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 5.471±9.2​e​-​3plus-or-minus5.4719.2𝑒-35.471\scriptscriptstyle\pm\scriptstyle 9.2e\text{-}3 | 0.923±5.8​e​-​4plus-or-minus0.9235.8𝑒-40.923\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4 | 0.975±1.0​e​-​4plus-or-minus0.9751.0𝑒-40.975\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4 | 0.744±3.3​e​-​4plus-or-minus0.7443.3𝑒-40.744\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4 |
| ResNet-P | 0.675±4.2​e​-​3plus-or-minus0.6754.2𝑒-30.675\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3 | 0.860±6.2​e​-​4plus-or-minus0.8606.2𝑒-40.860\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4 | 0.453±3.1​e​-​3plus-or-minus0.4533.1𝑒-30.453\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 3.041±1.7​e​-​2plus-or-minus3.0411.7𝑒-23.041\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}2 | 0.872±1.4​e​-​3plus-or-minus0.8721.4𝑒-30.872\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 0.820±2.0​e​-​4plus-or-minus0.8202.0𝑒-40.820\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 | 0.733±5.0​e​-​4plus-or-minus0.7335.0𝑒-40.733\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 | 5.305±2.3​e​-​2plus-or-minus5.3052.3𝑒-25.305\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2 | 0.923±3.6​e​-​4plus-or-minus0.9233.6𝑒-40.923\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4 | 0.972±2.1​e​-​4plus-or-minus0.9722.1𝑒-40.972\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 0.744±1.7​e​-​4plus-or-minus0.7441.7𝑒-40.744\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4 |
| ResNet-PLR | 0.691±6.3​e​-​3plus-or-minus0.6916.3𝑒-30.691\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}3 | 0.861±4.1​e​-​4plus-or-minus0.8614.1𝑒-40.861\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.443±1.4​e​-​3plus-or-minus0.4431.4𝑒-30.443\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 3.040±2.1​e​-​2plus-or-minus3.0402.1𝑒-23.040\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2 | 0.874±5.0​e​-​4plus-or-minus0.8745.0𝑒-40.874\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4 | 0.825±1.1​e​-​3plus-or-minus0.8251.1𝑒-30.825\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.734±6.3​e​-​4plus-or-minus0.7346.3𝑒-40.734\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4 | 5.400±2.6​e​-​2plus-or-minus5.4002.6𝑒-25.400\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}2 | 0.924±2.9​e​-​4plus-or-minus0.9242.9𝑒-40.924\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4 | 0.975±9.1​e​-​5plus-or-minus0.9759.1𝑒-50.975\scriptscriptstyle\pm\scriptstyle 9.1e\text{-}5 | 0.743±4.0​e​-​4plus-or-minus0.7434.0𝑒-40.743\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 |
| Transformer-L | 0.668±1.3​e​-​2plus-or-minus0.6681.3𝑒-20.668\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.861±6.2​e​-​4plus-or-minus0.8616.2𝑒-40.861\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4 | 0.455±1.4​e​-​3plus-or-minus0.4551.4𝑒-30.455\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3 | 3.188±8.8​e​-​3plus-or-minus3.1888.8𝑒-33.188\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}3 | 0.860±6.5​e​-​4plus-or-minus0.8606.5𝑒-40.860\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}4 | 0.824±4.6​e​-​4plus-or-minus0.8244.6𝑒-40.824\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4 | 0.727±1.1​e​-​3plus-or-minus0.7271.1𝑒-30.727\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 5.434±2.3​e​-​2plus-or-minus5.4342.3𝑒-25.434\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2 | 0.924±1.1​e​-​4plus-or-minus0.9241.1𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4 | 0.973±2.0​e​-​4plus-or-minus0.9732.0𝑒-40.973\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 | 0.743±2.7​e​-​4plus-or-minus0.7432.7𝑒-40.743\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4 |
| Transformer-LR | 0.666±1.0​e​-​3plus-or-minus0.6661.0𝑒-30.666\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 0.861±4.1​e​-​4plus-or-minus0.8614.1𝑒-40.861\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4 | 0.446±1.1​e​-​3plus-or-minus0.4461.1𝑒-30.446\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 3.193±1.6​e​-​2plus-or-minus3.1931.6𝑒-23.193\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2 | 0.861±2.0​e​-​4plus-or-minus0.8612.0𝑒-40.861\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 | 0.824±1.6​e​-​3plus-or-minus0.8241.6𝑒-30.824\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 0.733±7.8​e​-​4plus-or-minus0.7337.8𝑒-40.733\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4 | 5.430±3.0​e​-​2plus-or-minus5.4303.0𝑒-25.430\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}2 | 0.924±1.8​e​-​4plus-or-minus0.9241.8𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 | 0.973±1.0​e​-​4plus-or-minus0.9731.0𝑒-40.973\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4 | 0.743±1.8​e​-​4plus-or-minus0.7431.8𝑒-40.743\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 |
| Transformer-Q-L | 0.704±1.5​e​-​3plus-or-minus0.7041.5𝑒-30.704\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.861±1.1​e​-​3plus-or-minus0.8611.1𝑒-30.861\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.426±1.6​e​-​3plus-or-minus0.4261.6𝑒-30.426\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 3.183±2.5​e​-​2plus-or-minus3.1832.5𝑒-23.183\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}2 | 0.869±2.7​e​-​4plus-or-minus0.8692.7𝑒-40.869\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4 | 0.820±3.1​e​-​3plus-or-minus0.8203.1𝑒-30.820\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3 | 0.735±1.5​e​-​3plus-or-minus0.7351.5𝑒-30.735\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 5.553±1.5​e​-​2plus-or-minus5.5531.5𝑒-25.553\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2 | 0.925±2.8​e​-​4plus-or-minus0.9252.8𝑒-40.925\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4 | 0.976±5.9​e​-​5plus-or-minus0.9765.9𝑒-50.976\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}5 | 0.744±2.0​e​-​4plus-or-minus0.7442.0𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 |
| Transformer-Q-LR | 0.690±1.9​e​-​3plus-or-minus0.6901.9𝑒-30.690\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3 | 0.857±2.4​e​-​4plus-or-minus0.8572.4𝑒-40.857\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 | 0.425±1.2​e​-​3plus-or-minus0.4251.2𝑒-30.425\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 3.143±1.6​e​-​2plus-or-minus3.1431.6𝑒-23.143\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2 | 0.868±4.9​e​-​4plus-or-minus0.8684.9𝑒-40.868\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}4 | 0.818±2.3​e​-​3plus-or-minus0.8182.3𝑒-30.818\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 0.726±1.2​e​-​3plus-or-minus0.7261.2𝑒-30.726\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 5.471±1.5​e​-​2plus-or-minus5.4711.5𝑒-25.471\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2 | 0.924±2.0​e​-​4plus-or-minus0.9242.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4 | 0.975±1.9​e​-​4plus-or-minus0.9751.9𝑒-40.975\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}4 | 0.744±3.5​e​-​4plus-or-minus0.7443.5𝑒-40.744\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4 |
| Transformer-T-L | 0.693±6.8​e​-​3plus-or-minus0.6936.8𝑒-30.693\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}3 | 0.862±2.4​e​-​4plus-or-minus0.8622.4𝑒-40.862\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 | 0.439±1.0​e​-​3plus-or-minus0.4391.0𝑒-30.439\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3 | 3.136±3.5​e​-​3plus-or-minus3.1363.5𝑒-33.136\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3 | 0.872±1.3​e​-​4plus-or-minus0.8721.3𝑒-40.872\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4 | 0.826±2.3​e​-​3plus-or-minus0.8262.3𝑒-30.826\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3 | 0.731±1.6​e​-​3plus-or-minus0.7311.6𝑒-30.731\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3 | 5.579±5.2​e​-​2plus-or-minus5.5795.2𝑒-25.579\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}2 | 0.924±4.0​e​-​4plus-or-minus0.9244.0𝑒-40.924\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4 | 0.977±2.1​e​-​4plus-or-minus0.9772.1𝑒-40.977\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 0.743±2.3​e​-​4plus-or-minus0.7432.3𝑒-40.743\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4 |
| Transformer-T-LR | 0.686±4.1​e​-​3plus-or-minus0.6864.1𝑒-30.686\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}3 | 0.862±1.1​e​-​3plus-or-minus0.8621.1𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3 | 0.423±3.4​e​-​3plus-or-minus0.4233.4𝑒-30.423\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3 | 3.149±1.4​e​-​2plus-or-minus3.1491.4𝑒-23.149\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2 | 0.871±8.0​e​-​4plus-or-minus0.8718.0𝑒-40.871\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4 | 0.823±2.4​e​-​3plus-or-minus0.8232.4𝑒-30.823\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3 | 0.733±9.4​e​-​4plus-or-minus0.7339.4𝑒-40.733\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4 | 5.515±2.0​e​-​2plus-or-minus5.5152.0𝑒-25.515\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2 | 0.924±6.1​e​-​5plus-or-minus0.9246.1𝑒-50.924\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}5 | 0.976±9.2​e​-​5plus-or-minus0.9769.2𝑒-50.976\scriptscriptstyle\pm\scriptstyle 9.2e\text{-}5 | 0.744±2.9​e​-​4plus-or-minus0.7442.9𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4 |
| Transformer-PLR | 0.686±6.2​e​-​3plus-or-minus0.6866.2𝑒-30.686\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}3 | 0.864±9.4​e​-​4plus-or-minus0.8649.4𝑒-40.864\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4 | 0.449±1.2​e​-​3plus-or-minus0.4491.2𝑒-30.449\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3 | 3.091±1.3​e​-​2plus-or-minus3.0911.3𝑒-23.091\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2 | 0.873±1.5​e​-​3plus-or-minus0.8731.5𝑒-30.873\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3 | 0.823±1.7​e​-​3plus-or-minus0.8231.7𝑒-30.823\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3 | 0.734±2.1​e​-​4plus-or-minus0.7342.1𝑒-40.734\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4 | 5.581±6.4​e​-​2plus-or-minus5.5816.4𝑒-25.581\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}2 | 0.924±1.8​e​-​4plus-or-minus0.9241.8𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4 | 0.975±2.2​e​-​4plus-or-minus0.9752.2𝑒-40.975\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4 | 0.743±2.4​e​-​4plus-or-minus0.7432.4𝑒-40.743\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4 |
