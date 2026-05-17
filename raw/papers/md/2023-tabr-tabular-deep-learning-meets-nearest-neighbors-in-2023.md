---
arxiv: '2307.14338'
authors:
- Yury Gorishniy
- Ivan Rubachev
- Nikolay Kartashev
- Daniil Shlenskii
- Akim Kotelnikov
- Artem Babenko
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023'
url: http://arxiv.org/abs/2307.14338v2
year: 2023
---

# TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023

Yury Gorishniy††{\dagger}
    Ivan Rubachev‡⁣†

‡†{\ddagger}{\dagger}
    Nikolay Kartashev‡⁣†

‡†{\ddagger}{\dagger}
  
     Daniil Shlenskii††{\dagger}
    Akim Kotelnikov‡⁣†

‡†{\ddagger}{\dagger}
    Artem Babenko†⁣‡

†‡{\dagger}{\ddagger}
The first author: firstnamelastname@gmail.com
    ††{\dagger}Yandex
  ‡‡{\ddagger}HSE

###### Abstract

Deep learning (DL) models for tabular data problems (e.g. classification, regression) are currently receiving increasingly more attention from researchers.
However, despite the recent efforts, the non-DL algorithms based on gradient-boosted decision trees (GBDT) remain a strong go-to solution for these problems.
One of the research directions aimed at improving the position of tabular DL involves designing so-called retrieval-augmented models.
For a target object, such models retrieve other objects (e.g. the nearest neighbors) from the available training data and use their features and labels to make a better prediction.

In this work, we present TabR – essentially, a feed-forward network with a custom k-Nearest-Neighbors-like component in the middle.
On a set of public benchmarks with datasets up to several million objects, TabR marks a big step forward for tabular DL: it demonstrates the best average performance among tabular DL models, becomes the new state-of-the-art on several datasets, and even outperforms GBDT models on the recently proposed “GBDT-friendly” benchmark (see [Figure 1](#S0.F1 "Figure 1 ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
Among the important findings and technical details powering TabR, the main ones lie in the attention-like mechanism that is responsible for retrieving the nearest neighbors and extracting valuable signal from them.
In addition to the much higher performance, TabR is simple and significantly more efficient compared to prior retrieval-based tabular DL models.
The source code is published: [link](https://github.com/yandex-research/tabular-dl-tabr).

TabR (Ours, 2023)MLP-PLR (Gorishniy et al., [2022](#bib.bib12))FT-Transformer (Gorishniy et al., [2021](#bib.bib11))MLP (< 2021)23137111517717196928DL winsTiesXGBoost wins

Figure 1: 
Comparing DL models with XGBoost (Chen and Guestrin, [2016](#bib.bib7)) on 43 regression and classification tasks of middle scale (≤50​Kabsent50𝐾\leq 50K objects) from “Why do tree-based models still outperform deep learning on typical tabular data?” by Grinsztajn et al. ([2022](#bib.bib13)).
TabR marks a significant step forward compared to prior tabular DL models and continues the positive trend for the field.

## 1 Introduction

Machine learning (ML) problems on tabular data, where objects are described by a set of heterogeneous features, are ubiquitous in industrial applications in medicine, finance, manufacturing, and other fields.
Historically, for these tasks, the models based on gradient-boosted decision trees (GBDT) have been a go-to solution for a long time.
However, lately, tabular deep learning (DL) models have been receiving increasingly more attention, and they are becoming more competitive (Klambauer et al., [2017](#bib.bib26); Popov et al., [2020](#bib.bib35); Wang et al., [2020](#bib.bib46); Hazimeh et al., [2020](#bib.bib15); Huang et al., [2020](#bib.bib16); Gorishniy et al., [2021](#bib.bib11); Somepalli et al., [2021](#bib.bib42); Kossen et al., [2021](#bib.bib28); Gorishniy et al., [2022](#bib.bib12)).

In particular, several attempts to design a retrieval-augmented tabular DL model have been recently made (Somepalli et al., [2021](#bib.bib42); Qin et al., [2021](#bib.bib38); Kossen et al., [2021](#bib.bib28)).
For a target object, a retrieval-augmented model retrieves additional objects from the training set (e.g. the target object’s nearest neighbors, or even the whole training set) and uses them to improve the prediction for the target object.
In fact, the retrieval technique is widely popular in other domains, including natural language processing (Das et al., [2021](#bib.bib8); Wang et al., [2022](#bib.bib47); Izacard et al., [2022](#bib.bib18)), computer vision (Jia et al., [2021](#bib.bib20); Iscen et al., [2022](#bib.bib17); Long et al., [2022](#bib.bib30)), CTR prediction (Qin et al., [2020](#bib.bib37); [2021](#bib.bib38); Du et al., [2022](#bib.bib9)) and others.
Compared to purely parametric (i.e. retrieval-free) models, the retrieval-based ones can achieve higher performance and also exhibit several practically important properties, such as the ability for incremental learning and better robustness (Das et al., [2021](#bib.bib8); Jia et al., [2021](#bib.bib20)).

While multiple retrieval-augmented models for tabular data problems exist, in our experiments, we show that they provide if only minor benefits over the properly tuned multilayer perceptron (MLP; the simplest parametric model), while being significantly more complex and costly.
Nevertheless, in this work, we show that, with certain previously overlooked design aspects in mind, it is possible to obtain a retrieval-based tabular architecture that is powerful, simple and substantially more efficient than prior retrieval-based models.
We summarize our main contributions as follows:

1. 1.

   We design TabR – a simple retrieval-augmented tabular DL model which, on a set of public benchmarks, demonstrates the best average performance among DL models, achieves the new state-of-the-art on several datasets and is significantly more efficient than prior deep retrieval-based tabular models.
2. 2.

   In particular, TabR achieves a notable milestone for tabular DL by outperforming GBDT on the recently proposed benchmark with middle-scale tasks (Grinsztajn et al., [2022](#bib.bib13)), which was originally used to illustrate the superiority of decision-tree-based models over DL models. Tree-based models, in turn, remain a cheaper solution.
3. 3.

   We highlight the important degrees of freedom of the attention mechanism (the often used module in retrieval-based models) that allow designing better retrieval-based tabular models.

## 2 Related work

Gradient boosted decision trees (GBDT).
GBDT-based ML models are non-DL solutions for supervised problems on tabular data that are popular within the community due to their strong performance and high efficiency.
By employing the modern DL building blocks and, in particular, the retrieval technique, our new model successfully competes with GBDT and, in particular, demonstrates that DL models can be superior on non-big data by outperforming GBDT on the recently proposed benchmark with small-to-middle scale tasks (Grinsztajn et al., [2022](#bib.bib13)).

Parametric deep learning models.
Parametric tabular DL is a rapidly developing research direction aimed at bringing the benefits of deep learning to the world of tabular data while achieving competitive performance (Klambauer et al., [2017](#bib.bib26); Popov et al., [2020](#bib.bib35); Wang et al., [2020](#bib.bib46); Hazimeh et al., [2020](#bib.bib15); Huang et al., [2020](#bib.bib16); Gorishniy et al., [2021](#bib.bib11); [2022](#bib.bib12)).
The recent studies reveal that MLP-like backbones are still competitive (Gorishniy et al., [2021](#bib.bib11); Kadra et al., [2021](#bib.bib21); Gorishniy et al., [2022](#bib.bib12)), and that embeddings for continuous features (Gorishniy et al., [2022](#bib.bib12)) significantly reduce the gap between tabular DL and GBDT.
In this work, we show that a properly designed retrieval component can boost the performance of tabular DL even further.

Retrieval-augmented models in general.
Usually, the retrieval-based models are designed as follows.
For an input object, first, they retrieve relevant samples from available (training) data.
Then, they process the input object together with the retrieved instances to produce the final prediction for the input object.
One of the common motivations for designing retrieval-based schemes is the local learning paradigm (Bottou and Vapnik, [1992](#bib.bib6)), and the simplest possible example of such a model is the k𝑘k-nearest neighbors (kNN) algorithm (James et al., [2013](#bib.bib19)).
The promise of retrieval-based approaches was demonstrated across various domains, such as natural language processing (Lewis et al., [2020](#bib.bib29); Guu et al., [2020](#bib.bib14); Khandelwal et al., [2020](#bib.bib24); Izacard et al., [2022](#bib.bib18); Borgeaud et al., [2022](#bib.bib5)), computer vision (Iscen et al., [2022](#bib.bib17); Long et al., [2022](#bib.bib30)), CTR prediction (Qin et al., [2020](#bib.bib37); [2021](#bib.bib38); Du et al., [2022](#bib.bib9)) and others.
Additionally, retrieval-augmented models often have useful properties such as better interpretability (Wang and Sabuncu, [2023](#bib.bib45)), robustness (Zhao and Cho, [2018](#bib.bib49)) and others.

Retrieval-augmented models for tabular data problems.
The classic example of non-deep retrieval-based tabular models are the neighbor-based and kernel methods (James et al., [2013](#bib.bib19); Nader et al., [2022](#bib.bib33)).
There are also deep retrieval-based models applicable to (or directly designed for) tabular data problems (Wilson et al., [2016](#bib.bib48); Kim et al., [2019](#bib.bib25); Ramsauer et al., [2021](#bib.bib40); Kossen et al., [2021](#bib.bib28); Somepalli et al., [2021](#bib.bib42)).
Notably, some of them omit the retrieval step and use all training data points as the “retrieved” instances (Somepalli et al., [2021](#bib.bib42); Kossen et al., [2021](#bib.bib28); Schäfl et al., [2022](#bib.bib41)).
However, we show that the existing retrieval-based tabular DL models are only marginally better than simple parametric DL models, and that often comes with a cost of using heavy Transformer-like architectures.
Compared to prior work, where several layers with multi-head attention between objects and features are often used (Ramsauer et al., [2021](#bib.bib40); Kossen et al., [2021](#bib.bib28); Somepalli et al., [2021](#bib.bib42)), our model TabR implements its retrieval component with just one single-head attention-like module.
Importantly, the single attention-like module of TabR is customized in a way that makes it better suited for tabular data problems.
As a result, TabR substantially outperforms the existing retrieval-based DL models while being significantly more efficient.

## 3 TabR

In this section, we design a new retrieval-augmented deep learning model for tabular data problems.

### 3.1 Preliminaries

Notation.
For a given supervised learning problem on tabular data, we denote the dataset as {(xi,yi)}i=1nsuperscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑛\left\{\left(x\_{i},y\_{i}\right)\right\}\_{i=1}^{n} where xi∈𝕏subscript𝑥𝑖𝕏x\_{i}\in\mathbb{X} represents the i𝑖i-th object’s features and yi∈𝕐subscript𝑦𝑖𝕐y\_{i}\in\mathbb{Y} represents the i𝑖i-th object’s label.
Depending on the context, the i𝑖i index can be omitted.
We consider three types of tasks: binary classification 𝕐={0,1}𝕐01\mathbb{Y}=\{0,1\},
multiclass classification 𝕐={1,…,C}𝕐1…𝐶\mathbb{Y}=\{1,...,C\} and regression 𝕐=ℝ𝕐ℝ\mathbb{Y}=\mathbb{R}.
For simplicity, in most places, we will assume that xisubscript𝑥𝑖x\_{i} contains only continuous (i.e., continuous) features, and we will give additional comments on binary and categorical features when necessary.
The dataset is split into three disjoint parts: 1,n¯=It​r​a​i​n∪Iv​a​l∪It​e​s​t¯

1𝑛subscript𝐼𝑡𝑟𝑎𝑖𝑛subscript𝐼𝑣𝑎𝑙subscript𝐼𝑡𝑒𝑠𝑡\overline{1,n}=I\_{train}\cup I\_{val}\cup I\_{test}, where the “train” part is used for training, the “validation” part is used for early stopping and hyperparameter tuning, and the “test” part is used for the final evaluation.
An input object for which a given model makes a prediction is referred to as “input object” or “target object”.

When the retrieval technique is used for a given target object, the retrieval is performed within the set of “context candidates” or simply “candidates”: Ic​a​n​d⊆It​r​a​i​nsubscript𝐼𝑐𝑎𝑛𝑑subscript𝐼𝑡𝑟𝑎𝑖𝑛I\_{cand}\subseteq I\_{train}.
The retrieved objects, in turn, are called “context objects” or simply “context”.
Optionally, the target object can be included in its own context.
In this work, we use the same set of candidates for all input objects.

Experiment setup.
We extensively describe our tuning and evaluation protocols in [subsection D.6](#A4.SS6 "D.6 Experiment setup ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The most important points are that, for any given algorithm, on each dataset, following Gorishniy et al. ([2022](#bib.bib12)), (1) we perform hyperparameter tuning and early stopping using the validation set; (2) for the best hyperparameters, in the main text, we report the metric on the test set averaged over 15 random seeds, and provide standard deviations in [Appendix E](#A5 "Appendix E Extended results with standard deviations ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"); (3) when comparing any two algorithms, we take the standard deviations into account as described in [subsection D.6](#A4.SS6 "D.6 Experiment setup ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"); (4) to obtain ensembles of models of the same type, we split the 15 random seeds into three disjoint groups (i.e., into three ensembles) each consisting of five models, average predictions within each group, and report the average performance of the obtained three ensembles.

In this work, we mostly use the datasets from prior literature and provide their summary in [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") (sometimes, we refer to this set of datasets as “the default benchmark”).
Additionally, in [subsection 4.2](#S4.SS2 "4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we use the recently introduced benchmark with middle-scale tasks (≤50​Kabsent50𝐾\leq 50K objects) (Grinsztajn et al., [2022](#bib.bib13)) where GBDT was reported to be superior to DL solutions.

Table 1: Dataset properties. “RMSE” denotes root-mean-square error, “Acc.” denotes accuracy.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CH | CA | HO | AD | DI | OT | HI | BL | WE | CO | MI |
| #objects | 10000 | 20640 | 22784 | 48842 | 53940 | 61878 | 98049 | 166821 | 397099 | 581012 | 1200192 |
| #num.features | 7 | 8 | 16 | 6 | 6 | 93 | 28 | 4 | 118 | 10 | 131 |
| #bin.features | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 44 | 5 |
| #cat.features | 1 | 0 | 0 | 7 | 3 | 0 | 0 | 4 | 0 | 0 | 0 |
| metric | Acc. | RMSE | RMSE | Acc. | RMSE | Acc. | Acc. | RMSE | RMSE | Acc. | RMSE |
| #classes | 2 | – | – | 2 | – | 9 | 2 | – | – | 7 | – |
| majority class | 79% | – | – | 76% | – | 26% | 52% | – | – | 48% | – |

### 3.2 Architecture

To build a retrieval-based tabular DL model, we choose an incremental approach, where we start from a simple retrieval-free architecture, and, step by step, add and improve a retrieval component.

Let’s consider a generic feed-forward retrieval-free network f​(x)=P​(E​(x))𝑓𝑥𝑃𝐸𝑥f(x)=P(E(x)) informally partitioned into two parts: encoder E:𝕏→ℝd:𝐸→𝕏superscriptℝ𝑑E:\mathbb{X}\rightarrow\mathbb{R}^{d} and predictor P:ℝd→𝕐^:𝑃→superscriptℝ𝑑^𝕐P:\mathbb{R}^{d}\rightarrow\hat{\mathbb{Y}}.
To incrementally make it retrieval-based, we add retrieval module R𝑅R in a residual branch after E𝐸E as illustrated in [Figure 2](#S3.F2 "Figure 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), where x~∈ℝd~𝑥superscriptℝ𝑑\tilde{x}\in\mathbb{R}^{d} is the intermediate representation of the target object, {x~i}i∈Ic​a​n​d⊂ℝdsubscriptsubscript~𝑥𝑖𝑖subscript𝐼𝑐𝑎𝑛𝑑superscriptℝ𝑑\{\tilde{x}\_{i}\}\_{i\in I\_{cand}}\subset\mathbb{R}^{d} are the intermediate representations of the candidates and {yi}i∈Ic​a​n​d⊂𝕐subscriptsubscript𝑦𝑖𝑖subscript𝐼𝑐𝑎𝑛𝑑𝕐\{y\_{i}\}\_{i\in I\_{cand}}\subset\mathbb{Y} are the labels of the candidates.

!(/html/2307.14338/assets/x1.png)

Figure 2: 
The generic retrieval-based architecture introduced in [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and used to build TabR.
First, a target object and its candidates for retrieval are encoded with the same encoder E𝐸E.
Then, the retrieval module R𝑅R enriches the target object’s representation by retrieving and processing relevant objects from the candidates.
Finally, predictor P𝑃P makes a prediction.
The bold path highlights the structure of the feed-forward retrieval-free model before the addition of the retrieval module R𝑅R.

Encoder and predictor.
The encoder E𝐸E and predictor P𝑃P modules ([Figure 2](#S3.F2 "Figure 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) are not the focus of this work, so we keep them simple as illustrated in [Figure 3](#S3.F3 "Figure 3 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

!(/html/2307.14338/assets/x2.png)

Figure 3: 
Encoder E𝐸E and predictor P𝑃P introduced in [Figure 2](#S3.F2 "Figure 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
NEsubscript𝑁𝐸N\_{E} and NPsubscript𝑁𝑃N\_{P} denote the number of Block modules in E𝐸E and P𝑃P, respectively.
The Input Module encapsulates the input processing routines (feature normalization, one-hot encoding, etc.) and assembles a vector input for the subsequent linear layer.
In particular, Input Module can contain embeddings for continuous features (Gorishniy et al., [2022](#bib.bib12)).
(∗ LayerNorm is omitted in the first Block of E𝐸E.)

!(/html/2307.14338/assets/x3.png)

Figure 4: 
Simplified illustration of the retrieval module R𝑅R introduced in [Figure 2](#S3.F2 "Figure 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") (the omitted details are provided in the main text).
For the target object’s representation x~~𝑥\tilde{x}, the module takes the m𝑚m nearest neighbors among the candidates {x~i}subscript~𝑥𝑖\{\tilde{x}\_{i}\} according to the similarity module S:(ℝd,ℝd)→ℝ:𝑆→superscriptℝ𝑑superscriptℝ𝑑ℝS:(\mathbb{R}^{d},\mathbb{R}^{d})\rightarrow\mathbb{R} and aggregates
their values produced by the value module 𝒱:(ℝd,ℝd,𝕐)→ℝd:𝒱→superscriptℝ𝑑superscriptℝ𝑑𝕐superscriptℝ𝑑\mathcal{V}:(\mathbb{R}^{d},\mathbb{R}^{d},\mathbb{Y})\rightarrow\mathbb{R}^{d}.

Retrieval module.
We define the retrieval module R𝑅R in the spirit of k𝑘k-nearest neighbors as illustrated in [Figure 4](#S3.F4 "Figure 4 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
In the figure, the following formal details are omitted for clarity:

1. 1.

   If the encoder E𝐸E contains at least one Block (i.e. NE>0subscript𝑁𝐸0N\_{E}>0), then, before being passed to R𝑅R, x~~𝑥\tilde{x} and all x~isubscript~𝑥𝑖\tilde{x}\_{i} are normalized with a shared layer normalization (Ba et al., [2016](#bib.bib2)).
2. 2.

   Optionally, the target object itself can be unconditionally (i.e. ignoring the top-m𝑚m operation) added as the (m+1)𝑚1(m+1)-th object to its set of context objects with the similarity score 𝒮​(x~,x~)𝒮~𝑥~𝑥\mathcal{S}(\tilde{x},\tilde{x}).
3. 3.

   Dropout is applied to the weights produced by the softmax function.
4. 4.

   Throughout the paper, we use m=96𝑚96m=96 and Ic​a​n​d=It​r​a​i​nsubscript𝐼𝑐𝑎𝑛𝑑subscript𝐼𝑡𝑟𝑎𝑖𝑛I\_{cand}=I\_{train}.

Now, we iterate over possible designs of the similarity module 𝒮𝒮\mathcal{S} and the value module 𝒱𝒱\mathcal{V} (introduced in [Figure 4](#S3.F4 "Figure 4 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
During this process, we do not use embeddings for numerical features (Gorishniy et al., [2022](#bib.bib12)) in the Input Module of the encoder E𝐸E and set NEsubscript𝑁𝐸N\_{E} = 0, NPsubscript𝑁𝑃N\_{P} = 1 (see [Figure 3](#S3.F3 "Figure 3 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).

Step-0. The vanilla-attention-like baseline.
The self-attention operation (Vaswani et al., [2017](#bib.bib44)) was often used in prior work to model the interaction between a target object and candidate/context objects (Somepalli et al., [2021](#bib.bib42); Kossen et al., [2021](#bib.bib28); Schäfl et al., [2022](#bib.bib41)).
Then, instantiating retrieval module R𝑅R as the vanilla self-attention (modulo the top-m𝑚m operation) is a reasonable baseline:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒮​(x~,x~i)=WQ​(x~)T​WK​(xi~)⋅d−1/2𝒱​(x~,x~i,yi)=WV​(x~i)formulae-sequence𝒮~𝑥subscript~𝑥𝑖⋅subscript𝑊𝑄superscript~𝑥𝑇subscript𝑊𝐾~subscript𝑥𝑖superscript𝑑12𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑉subscript~𝑥𝑖\displaystyle\begin{split}\mathcal{S}(\tilde{x},\tilde{x}\_{i})=W\_{Q}(\tilde{x})^{T}W\_{K}(\tilde{x\_{i}})\cdot d^{-\nicefrac{{1}}{{2}}}\qquad\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{V}(\tilde{x}\_{i})\end{split} | |  | (1) |

where WQsubscript𝑊𝑄W\_{Q}, WKsubscript𝑊𝐾W\_{K}, and WVsubscript𝑊𝑉W\_{V} are linear layers, and the target object is added as the (m+1)𝑚1(m+1)-th object to its own context (i.e., ignoring the top-m𝑚m operation).
As reported in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), the Step-0 configuration performs similarly to MLP, which means that using the vanilla self-attention is a suboptimal strategy.

Step-1. Adding context labels.
A natural attempt to improve the Step-0 configuration is to utilize labels of the context objects, for example, by incorporating them into the value module as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒮​(x~,x~i)=WQ​(x~)T​WK​(xi~)⋅d−1/2𝒱​(x~,x~i,yi)=WY​(yi)+¯​WV​(x~i)formulae-sequence𝒮~𝑥subscript~𝑥𝑖⋅subscript𝑊𝑄superscript~𝑥𝑇subscript𝑊𝐾~subscript𝑥𝑖superscript𝑑12𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖¯limit-fromsubscript𝑊𝑌subscript𝑦𝑖subscript𝑊𝑉subscript~𝑥𝑖\displaystyle\begin{split}\mathcal{S}(\tilde{x},\tilde{x}\_{i})=W\_{Q}(\tilde{x})^{T}W\_{K}(\tilde{x\_{i}})\cdot d^{-\nicefrac{{1}}{{2}}}\qquad\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=\underline{W\_{Y}(y\_{i})+}W\_{V}(\tilde{x}\_{i})\end{split} | |  | (2) |

where the difference with [Equation 1](#S3.E1 "1 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") is the underlined addition of WY:𝕐→ℝd:subscript𝑊𝑌→𝕐superscriptℝ𝑑W\_{Y}:\mathbb{Y}\rightarrow\mathbb{R}^{d}, which is an embedding table for classification tasks and a linear layer for regression tasks.
[Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") shows no improvements from using labels, which is counter-intuitive.
Perhaps, the similarity module 𝒮𝒮\mathcal{S} taken from the vanilla attention does not allow benefiting from such a valuable signal as labels.

Step-2. Improving the similarity module 𝒮𝒮\mathcal{S}.
Empirically, we observed that removing the notion of queries (i.e. removing WQsubscript𝑊𝑄W\_{Q}) and using the L2subscript𝐿2L\_{2} distance instead of the dot product significantly improves performance on several datasets in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"):

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒮​(x~,x~i)=−‖WK​(x~)−WK​(x~i)‖2¯⋅d−1/2𝒱​(x~,x~i,yi)=WY​(yi)+WV​(x~i)formulae-sequence𝒮~𝑥subscript~𝑥𝑖⋅¯superscriptnormsubscript𝑊𝐾~𝑥subscript𝑊𝐾subscript~𝑥𝑖2superscript𝑑12𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑌subscript𝑦𝑖subscript𝑊𝑉subscript~𝑥𝑖\displaystyle\begin{split}\mathcal{S}(\tilde{x},\tilde{x}\_{i})=\underline{-\|W\_{K}(\tilde{x})-W\_{K}(\tilde{x}\_{i})\|^{2}}\cdot d^{-\nicefrac{{1}}{{2}}}\qquad\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{Y}(y\_{i})+W\_{V}(\tilde{x}\_{i})\end{split} | |  | (3) |

where the difference with [Equation 2](#S3.E2 "2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") is underlined.
This change is a turning point in our story, which was overlooked in prior work.
Crucially, in [subsection A.3](#A1.SS3 "A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we show that removing any of the three ingredients (context labels, key-only representation, L2subscript𝐿2L\_{2} distance) results in a performance drop back to the level of MLP.
While the L2subscript𝐿2L\_{2} distance is unlikely to be the universally best choice for problems (even within the tabular domain), it seems to be a reasonable default choice for tabular data problems.

Table 2: 
The performance of the implementations of the retrieval module R𝑅R, described in [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
If a number is underlined, then it is better than the corresponding number from the previous step at least by the standard deviation.
Noticeable improvements over MLP start at Step-2.
Notation: ↓ corresponds to RMSE, ↑ corresponds to accuracy.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CH ↑ | CA ↓ | HO ↓ | AD ↑ | DI ↓ | OT ↑ | HI ↑ | BL ↓ | WE ↓ | CO ↑ |
| MLP | 0.8540.8540.854 | 0.4990.4990.499 | 3.1123.1123.112 | 0.8530.8530.853 | 0.1400.1400.140 | 0.8160.8160.816 | 0.7190.7190.719 | 0.6970.6970.697 | 1.9051.9051.905 | 0.9630.9630.963 |
| (Step-0) The vanilla attention baseline | 0.8550.8550.855 | 0.484¯¯0.484\underline{0.484} | 3.2343.2343.234 | 0.857¯¯0.857\underline{0.857} | 0.1420.1420.142 | 0.8140.8140.814 | 0.7190.7190.719 | 0.6990.6990.699 | 1.9031.9031.903 | 0.9570.9570.957 |
| (Step-1) + Context labels | 0.8550.8550.855 | 0.4890.4890.489 | 3.2053.2053.205 | 0.8570.8570.857 | 0.1420.1420.142 | 0.8140.8140.814 | 0.7190.7190.719 | 0.6980.6980.698 | 1.9061.9061.906 | 0.960¯¯0.960\underline{0.960} |
| (Step-2) + New similarity module 𝒮𝒮\mathcal{S} | 0.860¯¯0.860\underline{0.860} | 0.418¯¯0.418\underline{0.418} | 3.153¯¯3.153\underline{3.153} | 0.8580.8580.858 | 0.140¯¯0.140\underline{0.140} | 0.8130.8130.813 | 0.7200.7200.720 | 0.692¯¯0.692\underline{0.692} | 1.804¯¯1.804\underline{1.804} | 0.972¯¯0.972\underline{0.972} |
| (Step-3) + New value module 𝒱𝒱\mathcal{V} | 0.8590.8590.859 | 0.408¯¯0.408\underline{0.408} | 3.1583.1583.158 | 0.863¯¯0.863\underline{0.863} | 0.135¯¯0.135\underline{0.135} | 0.8100.8100.810 | 0.7220.7220.722 | 0.6920.6920.692 | 1.8141.8141.814 | 0.975¯¯0.975\underline{0.975} |
| (Step-4) + Technical tweaks = TabR | 0.8600.8600.860 | 0.403¯¯0.403\underline{0.403} | 3.067¯¯3.067\underline{3.067} | 0.8650.8650.865 | 0.133¯¯0.133\underline{0.133} | 0.818¯¯0.818\underline{0.818} | 0.7220.7220.722 | 0.690¯¯0.690\underline{0.690} | 1.747¯¯1.747\underline{1.747} | 0.9730.9730.973 |

Step-3. Improving the value module 𝒱𝒱\mathcal{V}.
Now, we take inspiration from DNNR (Nader et al., [2022](#bib.bib33)) – the recently proposed generalization of the kNN algorithm for regression problems.
Namely, we make the value module 𝒱𝒱\mathcal{V} more expressive by taking the target object’s representation x~~𝑥\tilde{x} into account:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒮​(x~,x~i)=−‖WK​(x~)−WK​(x~i)‖2⋅d−1/2𝒱​(x~,x~i,yi)=WY​(yi)+T​(WK​(x~)−WK​(x~i))¯T​(⋅)=LinearWithoutBias​(Dropout​(ReLU​(Linear​(⋅))))formulae-sequence𝒮~𝑥subscript~𝑥𝑖⋅superscriptdelimited-∥∥subscript𝑊𝐾~𝑥subscript𝑊𝐾subscript~𝑥𝑖2superscript𝑑12𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑌subscript𝑦𝑖¯𝑇subscript𝑊𝐾~𝑥subscript𝑊𝐾subscript~𝑥𝑖𝑇⋅LinearWithoutBiasDropoutReLULinear⋅\displaystyle\begin{split}&\mathcal{S}(\tilde{x},\tilde{x}\_{i})=-\|W\_{K}(\tilde{x})-W\_{K}(\tilde{x}\_{i})\|^{2}\cdot d^{-\nicefrac{{1}}{{2}}}\ \ \mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{Y}(y\_{i})+\underline{T(W\_{K}(\tilde{x})-W\_{K}(\tilde{x}\_{i}))}\\ &T(\cdot)=\texttt{LinearWithoutBias}(\texttt{Dropout}(\texttt{ReLU}(\texttt{Linear}(\cdot))))\end{split} | |  | (4) |

where the difference with [Equation 3](#S3.E3 "3 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") is underlined.
[Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") shows that the new value module further improves the performance on several datasets.
Intuitively, the term WY​(yi)subscript𝑊𝑌subscript𝑦𝑖W\_{Y}(y\_{i}) (the embedding of the context object’s label) can be seen as the “raw” contribution of the i𝑖i-th context object.
The term T​(WK​(x~)−WK​(x~i))𝑇subscript𝑊𝐾~𝑥subscript𝑊𝐾subscript~𝑥𝑖T(W\_{K}(\tilde{x})-W\_{K}(\tilde{x}\_{i})) can be seen as the “correction” term, where the module T𝑇T translates the differences in the key space into the differences in the label embedding space.

Step-4. TabR.
Finally, empirically, we observed that omitting the scaling term d−1/2superscript𝑑12d^{-\nicefrac{{1}}{{2}}} in the similarity module and not including the target object to its own context leads to better results on average as reported in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
Both aspects can be considered hyperparameters, and the above notes can be seen as our default recommendations.
We call the obtained model “TabR” (Tab ∼similar-to\sim tabular, R ∼similar-to\sim retrieval).
The formal complete description of how TabR implements the retrieval module R𝑅R is as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | k=WK​(x~),ki=WK​(x~i)𝒮​(x~,x~i)=−‖k−ki‖2𝒱​(x~,x~i,yi)=WY​(yi)+T​(k−ki)formulae-sequence𝑘subscript𝑊𝐾~𝑥formulae-sequencesubscript𝑘𝑖subscript𝑊𝐾subscript~𝑥𝑖formulae-sequence𝒮~𝑥subscript~𝑥𝑖superscriptdelimited-∥∥𝑘subscript𝑘𝑖2𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑌subscript𝑦𝑖𝑇𝑘subscript𝑘𝑖\displaystyle\begin{split}k=W\_{K}(\tilde{x}),\ k\_{i}=W\_{K}(\tilde{x}\_{i})\quad\mathcal{S}(\tilde{x},\tilde{x}\_{i})=-\|k-k\_{i}\|^{2}\quad\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{Y}(y\_{i})+T(k-k\_{i})\end{split} | |  | (5) |

where WKsubscript𝑊𝐾W\_{K} is a linear layer, WYsubscript𝑊𝑌W\_{Y} is an embedding table for classification tasks and a linear layer for regression tasks, (by default) a target object is not included in its own context, (by default) the similarity scores are not scaled, and T​(⋅)=LinearWithoutBias​(Dropout​(ReLU​(Linear​(⋅))))𝑇⋅LinearWithoutBiasDropoutReLULinear⋅T(\cdot)=\texttt{LinearWithoutBias}(\texttt{Dropout}(\texttt{ReLU}(\texttt{Linear}(\cdot)))).

Limitations.
TabR has standard limitations of retrieval-augmented models, which we describe in [Appendix B](#A2 "Appendix B Limitations & Practical considerations ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
We encourage practitioners to review the limitations before using TabR in practice.

## 4 Experiments on public benchmarks

In this section, we compare TabR (introduced in [section 3](#S3 "3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) with existing retrieval-based solutions and state-of-the-art parametric models.
In addition to the fully-fledged configuration of TabR (with all degrees of freedom available for E𝐸E and P𝑃P as described in [Figure 3](#S3.F3 "Figure 3 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")), we also use TabR-S (“S” stands for “simple”) – a simple configuration, which does not use feature embeddings (Gorishniy et al., [2022](#bib.bib12)), has a linear encoder (NE=0subscript𝑁𝐸0N\_{E}=0) and a one-block predictor (NP=1subscript𝑁𝑃1N\_{P}=1).
We specify when TabR-S is used only in tables, figures, and captions but not in the text.
For other details on TabR, including hyperparameter tuning, see [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

### 4.1 Evaluating retrieval-augmented deep learning models for tabular data

In this section, we compare TabR ([section 3](#S3 "3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) and the existing retrieval-augmented solutions with fully parametric DL models (see [Appendix D](#A4 "Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for implementation details for all algorithms). [Table 3](#S4.T3 "Table 3 ‣ 4.1 Evaluating retrieval-augmented deep learning models for tabular data ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") indicates that TabR is the only retrieval-based model that provides a significant performance boost over MLP on many datasets.
In particular, the full variation of TabR outperforms MLP-PLR (the modern parametric DL model with the highest average rank from Gorishniy et al. ([2022](#bib.bib12))) on several datasets (CA, OT, BL, WE, CO), and performs on par with it on the rest except for the MI dataset.
Regarding the prior retrieval-based solutions, we faced various technical limitations, such as incompatibility with classification problems and scaling issues (e.g., as we show in [subsection A.4](#A1.SS4 "A.4 Comparing training times ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), it takes dramatically less time to train TabR than NPT (Kossen et al., [2021](#bib.bib28)) – the closest retrieval-based competitor from [Table 3](#S4.T3 "Table 3 ‣ 4.1 Evaluating retrieval-augmented deep learning models for tabular data ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
Notably, the retrieval component is not universally beneficial for all datasets.

Table 3: 
Comparing TabR with existing retrieval-augmented tabular models and parametric DL models. The notation follows [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"). The bold entries are the best-performing algorithms, which are defined with standard deviations taken into account as described in [subsection D.6](#A4.SS6 "D.6 Experiment setup ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CH ↑ | CA ↓ | HO ↓ | AD ↑ | DI ↓ | OT ↑ | HI ↑ | BL ↓ | WE ↓ | CO ↑ | MI ↓ | Avg. Rank |
| kNN | 0.8370.8370.837 | 0.5880.5880.588 | 3.7443.7443.744 | 0.8340.8340.834 | 0.2560.2560.256 | 0.7740.7740.774 | 0.6650.6650.665 | 0.7120.7120.712 | 2.2962.2962.296 | 0.9270.9270.927 | 0.7640.7640.764 | 6.0±1.7plus-or-minus6.01.76.0\pm 1.7 |
| DNNR (Nader et al., [2022](#bib.bib33)) | – | 0.4300.4300.430 | 3.2103.2103.210 | – | 0.1450.1450.145 | – | – | 0.7040.7040.704 | 1.9131.9131.913 | – | 0.7650.7650.765 | 4.8±1.9plus-or-minus4.81.94.8\pm 1.9 |
| DKL (Wilson et al., [2016](#bib.bib48)) | – | 0.5210.5210.521 | 3.4233.4233.423 | – | 0.1470.1470.147 | – | – | 0.6990.6990.699 | – | – | – | 6.2±0.5plus-or-minus6.20.56.2\pm 0.5 |
| ANP (Kim et al., [2019](#bib.bib25)) | – | 0.4720.4720.472 | 3.1623.1623.162 | – | 0.1400.1400.140 | – | – | 0.7050.7050.705 | 1.9021.9021.902 | – | – | 4.6±2.5plus-or-minus4.62.54.6\pm 2.5 |
| SAINT (Somepalli et al., [2021](#bib.bib42)) | 0.8600.860\mathbf{0.860} | 0.4680.4680.468 | 3.2423.2423.242 | 0.8600.8600.860 | 0.1370.1370.137 | 0.8120.8120.812 | 0.7240.7240.724 | 0.6930.6930.693 | 1.9331.9331.933 | 0.9640.9640.964 | 0.7630.7630.763 | 3.8±1.5plus-or-minus3.81.53.8\pm 1.5 |
| NPT (Kossen et al., [2021](#bib.bib28)) | 0.8580.8580.858 | 0.4740.4740.474 | 3.1753.1753.175 | 0.8530.8530.853 | 0.1380.1380.138 | 0.8150.8150.815 | 0.7210.7210.721 | 0.6920.6920.692 | 1.9471.9471.947 | 0.9660.9660.966 | 0.7530.7530.753 | 3.6±1.0plus-or-minus3.61.03.6\pm 1.0 |
| MLP | 0.8540.8540.854 | 0.4990.4990.499 | 3.1123.1123.112 | 0.8530.8530.853 | 0.1400.1400.140 | 0.8160.8160.816 | 0.7190.7190.719 | 0.6970.6970.697 | 1.9051.9051.905 | 0.9630.9630.963 | 0.7480.7480.748 | 3.7±1.3plus-or-minus3.71.33.7\pm 1.3 |
| MLP-PLR | 0.8600.8600.860 | 0.4760.4760.476 | 3.0563.056\mathbf{3.056} | 0.8700.870\mathbf{0.870} | 0.1340.1340.134 | 0.8190.8190.819 | 0.7290.729\mathbf{0.729} | 0.6870.6870.687 | 1.8601.8601.860 | 0.9700.9700.970 | 0.7440.744\mathbf{0.744} | 2.0±1.0plus-or-minus2.01.02.0\pm 1.0 |
| TabR-S | 0.8600.8600.860 | 0.4030.403\mathbf{0.403} | 3.0673.067\mathbf{3.067} | 0.8650.8650.865 | 0.1330.133\mathbf{0.133} | 0.8180.8180.818 | 0.7220.7220.722 | 0.6900.6900.690 | 1.7471.7471.747 | 0.9730.9730.973 | 0.7500.7500.750 | 1.9±0.7plus-or-minus1.90.71.9\pm 0.7 |
| TabR | 0.8620.862\mathbf{0.862} | 0.4000.400\mathbf{0.400} | 3.1053.1053.105 | 0.8700.870\mathbf{0.870} | 0.1330.133\mathbf{0.133} | 0.8250.825\mathbf{0.825} | 0.7290.729\mathbf{0.729} | 0.6760.676\mathbf{0.676} | 1.6901.690\mathbf{1.690} | 0.9760.976\mathbf{0.976} | 0.7500.7500.750 | 1.3±0.6plus-or-minus1.30.61.3\pm 0.6 |

The obtained results highlight the retrieval technique and embeddings for numerical features (Gorishniy et al., [2022](#bib.bib12)) (used in MLP-PLR and TabR) as two powerful architectural elements that improve the optimization properties of tabular DL models.
Interestingly, the two techniques are not fully orthogonal, but none of them can recover the full power of the other, and it depends on a given dataset whether one should prefer the retrieval, the embeddings, or a combination of both.

The main takeaway.
TabR becomes a new strong deep learning solution for tabular data problems and demonstrates a good potential of the retrieval-based approach.
TabR demonstrates strong average performance and achieves the new state-of-the-art on several datasets.

### 4.2 Comparing TabR with gradient-boosted decision trees

Table 4: 
Comparing ensembles of TabR with ensembles of GBDT models.
See [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") to learn how the “default” TabR-S was obtained.
The notation follows [Table 3](#S4.T3 "Table 3 ‣ 4.1 Evaluating retrieval-augmented deep learning models for tabular data ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CH ↑ | CA ↓ | HO ↓ | AD ↑ | DI ↓ | OT ↑ | HI ↑ | BL ↓ | WE ↓ | CO ↑ | MI ↓ | Avg. Rank |
| Tuned hyperparameters | | | | | | | | | | | | |
| XGBoost | 0.8610.8610.861 | 0.4320.4320.432 | 3.1643.1643.164 | 0.8720.872\mathbf{0.872} | 0.1360.1360.136 | 0.8320.832\mathbf{0.832} | 0.7260.7260.726 | 0.6800.6800.680 | 1.7691.7691.769 | 0.9710.9710.971 | 0.7410.7410.741 | 2.5±0.9plus-or-minus2.50.92.5\pm 0.9 |
| CatBoost | 0.8590.8590.859 | 0.4260.4260.426 | 3.1063.1063.106 | 0.8720.872\mathbf{0.872} | 0.1330.1330.133 | 0.8270.8270.827 | 0.7270.7270.727 | 0.6810.6810.681 | 1.7731.7731.773 | 0.9690.9690.969 | 0.7410.741\mathbf{0.741} | 2.5±1.1plus-or-minus2.51.12.5\pm 1.1 |
| LightGBM | 0.8600.8600.860 | 0.4340.4340.434 | 3.1673.1673.167 | 0.8720.872\mathbf{0.872} | 0.1360.1360.136 | 0.8320.832\mathbf{0.832} | 0.7260.7260.726 | 0.6790.6790.679 | 1.7611.7611.761 | 0.9710.9710.971 | 0.7410.7410.741 | 2.4±0.9plus-or-minus2.40.92.4\pm 0.9 |
| TabR | 0.8650.865\mathbf{0.865} | 0.3910.391\mathbf{0.391} | 3.0253.025\mathbf{3.025} | 0.8720.872\mathbf{0.872} | 0.1310.131\mathbf{0.131} | 0.8310.831\mathbf{0.831} | 0.7330.733\mathbf{0.733} | 0.6740.674\mathbf{0.674} | 1.6611.661\mathbf{1.661} | 0.9770.977\mathbf{0.977} | 0.7480.7480.748 | 1.3±0.9plus-or-minus1.30.91.3\pm 0.9 |
| Default hyperparameters | | | | | | | | | | | | |
| XGBoost | 0.8560.8560.856 | 0.4710.4710.471 | 3.3683.3683.368 | 0.8710.8710.871 | 0.1430.1430.143 | 0.8170.8170.817 | 0.7160.7160.716 | 0.6830.6830.683 | 1.9201.9201.920 | 0.9660.9660.966 | 0.7500.7500.750 | 3.4±0.9plus-or-minus3.40.93.4\pm 0.9 |
| CatBoost | 0.8610.8610.861 | 0.4320.4320.432 | 3.1083.1083.108 | 0.8740.874\mathbf{0.874} | 0.1320.1320.132 | 0.8220.8220.822 | 0.7260.726\mathbf{0.726} | 0.6840.6840.684 | 1.8861.8861.886 | 0.9240.9240.924 | 0.7440.744\mathbf{0.744} | 2.1±0.8plus-or-minus2.10.82.1\pm 0.8 |
| LightGBM | 0.8560.8560.856 | 0.4490.4490.449 | 3.2223.2223.222 | 0.8690.8690.869 | 0.1370.1370.137 | 0.8260.826\mathbf{0.826} | 0.7200.7200.720 | 0.6810.6810.681 | 1.8171.8171.817 | 0.8990.8990.899 | 0.7440.7440.744 | 2.5±0.9plus-or-minus2.50.92.5\pm 0.9 |
| TabR-S | 0.8640.864\mathbf{0.864} | 0.3980.398\mathbf{0.398} | 2.9712.971\mathbf{2.971} | 0.8590.8590.859 | 0.1310.131\mathbf{0.131} | 0.8240.8240.824 | 0.7240.7240.724 | 0.6880.6880.688 | 1.7211.721\mathbf{1.721} | 0.9740.974\mathbf{0.974} | 0.7520.7520.752 | 2.0±1.3plus-or-minus2.01.32.0\pm 1.3 |

In this section, we compare TabR with models based on gradient-boosted decision trees (GBDT): XGBoost (Chen and Guestrin, [2016](#bib.bib7)), LightGBM (Ke et al., [2017](#bib.bib22)) and CatBoost (Prokhorenkova et al., [2018](#bib.bib36)).
Specifically, we compare ensembles (e.g. an ensemble of TabRs vs. an ensemble of XGBoosts) for a fair comparison since gradient boosting is already an ensembling technique.

The default benchmark.
[Table 4](#S4.T4 "Table 4 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") shows that, on the default benchmark, the tuned TabR provides noticeable improvements over tuned GBDT on several datasets (CH, CA, HO, HI, WE, CO), while being competitive on the rest, except for the MI dataset.
The table also demonstrates that TabR has a competitive default configuration (defined in [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).

The benchmark from Grinsztajn et al. ([2022](#bib.bib13)).
Now, we go further and use the recently proposed benchmark with small-to-middle-scale tasks Grinsztajn et al. ([2022](#bib.bib13)).
Importantly, this benchmark was originally used to illustrate the superiority of GBDT over parametric DL models on datasets with ≤50​Kabsent50𝐾\leq 50K objects, which makes it an interesting challenge for TabR.
We adjust the benchmark to our tuning and evaluation protocols (see [subsection C.2](#A3.SS2 "C.2 The benchmark from Grinsztajn et al. [2022] ‣ Appendix C Benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for details) and report the results in [Table 5](#S4.T5 "Table 5 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
While MLP-PLR (one of the best parametric DL models) indeed is slightly inferior to GBDT on this set of tasks, TabR makes a significant step forward and outperforms GBDT on average.

In the appendix, we provide more analysis: in [subsection A.5](#A1.SS5 "A.5 Augmenting XGBoost with a retrieval component ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we try augmenting XGBoost with a retrieval component; in [subsection A.4](#A1.SS4 "A.4 Comparing training times ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we compare training times of TabR and GBDT models.

Table 5: 
Comparing ensembles of DL models with ensembles of GBDT models on the benchmark from Grinsztajn et al. ([2022](#bib.bib13)) (e.g., an ensemble of MLPs vs ensemble of XGBoosts; note that in [Figure 1](#S0.F1 "Figure 1 ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we compare single models, hence the different numbers).
See [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for the details on the “default” TabR-S.
The default configuration of TabR-S is compared against the default configurations of GBDT models.
The comparison is performed in a pairwise manner with standard deviations taken into account as described in [subsection D.6](#A4.SS6 "D.6 Experiment setup ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | vs. XGBoost | | | vs. CatBoost | | | vs. LightGBM | | |
|  | Win / | Tie / | Loss | Win / | Tie / | Loss | Win / | Tie / | Loss |
|  | Tuned hyperparameters | | | | | | | | |
| MLP | 666 | 111111 | 262626 | 666 | 888 | 292929 | 555 | 111111 | 272727 |
| MLP-PLR | 121212 | 171717 | 141414 | 101010 | 111111 | 222222 | 141414 | 151515 | 141414 |
| TabR-S | 212121 | 131313 | 999 | 171717 | 111111 | 151515 | 212121 | 151515 | 777 |
| TabR | 262626 | 141414 | 333 | 232323 | 131313 | 777 | 262626 | 141414 | 333 |
|  | Default hyperparameters | | | | | | | | |
| TabR-S | 282828 | 101010 | 555 | 171717 | 161616 | 101010 | 252525 | 999 | 999 |

The main takeaway. After the comparison with GBDT, TabR confirms its status of a new strong solution for tabular data problems: it provides strong average performance and can provide a noticeable improvement over GBDT on some datasets.

## 5 Analysis

### 5.1 Freezing contexts for faster training of TabR

In the vanilla formulation of TabR ([section 3](#S3 "3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")), for each training batch, the most up-to-date contexts are mined by encoding all the candidates and computing similarities with all of them, which can be prohibitively slow on large datasets.
For example, it takes more than 18 hours to train a single TabR on the full “Weather prediction” dataset (Malinin et al., [2021](#bib.bib32)) (3M+ objects; with the default hyperparameters from [Table 4](#S4.T4 "Table 4 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
However, as we show in [Figure 5](#S5.F5 "Figure 5 ‣ 5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), for an average training object, its context (i.e. the top-m𝑚m candidates and the distribution over them according to the similarity module 𝒮𝒮\mathcal{S}) gradually “stabilizes” during the course of training, which gives an opportunity for simple optimization.
Namely, after a fixed number of epochs, we can perform “context freeze”: i.e., compute the up-to-date contexts for all training (but not validation and test) objects for the one last time and then reuse these contexts for the rest of the training.
[Table 6](#S5.T6 "Table 6 ‣ 5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") indicates that, on some datasets, this simple technique allows accelerating training of TabR without much loss in metrics, with more noticeable speedups on larger datasets.
In particular, on the full “Weather prediction” dataset, we achieve nearly sevenfold speedup (from 18h9min to 3h15min) while maintaining competitive RMSE.
See [subsection D.2](#A4.SS2 "D.2 Implementation details of subsection 5.1 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for implementation details.

Table 6: 
The performance of TabR-S with the “context freeze” as described in [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
TabR-S (CF-N𝑁N) denotes TabR-S with the context freeze applied after N𝑁N epochs.
In parentheses, we provide the fraction of time spent on training compared to the training without freezing (the last row).

|  | CA ↓ | DI ↓ | HI ↑ | BL ↓ | WE ↓ | CO ↑ | WE (full) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TabR-S (CF-1) | 0.4140.4140.414 (0.720.720.72) | 0.1370.1370.137 (0.470.470.47) | 0.7180.718\mathbf{0.718} (0.800.800.80) | 0.6920.6920.692 (0.610.610.61) | 1.7701.7701.770 (0.570.570.57) | 0.9730.973\mathbf{0.973} (0.490.490.49) | 1.3251.3251.325 (0.130.130.13) |
| TabR-S (CF-4) | 0.4090.409\mathbf{0.409} (0.710.710.71) | 0.1360.1360.136 (0.510.510.51) | 0.7170.7170.717 (0.730.730.73) | 0.6910.691\mathbf{0.691} (0.620.620.62) | 1.7631.7631.763 (0.560.560.56) | 0.9730.973\mathbf{0.973} (0.590.590.59) | – |
| TabR-S | 0.4060.406\mathbf{0.406} (1.001.001.00) | 0.1330.133\mathbf{0.133} (1.001.001.00) | 0.7190.719\mathbf{0.719} (1.001.001.00) | 0.6910.691\mathbf{0.691} (1.001.001.00) | 1.7551.755\mathbf{1.755} (1.001.001.00) | 0.9730.973\mathbf{0.973} (1.001.001.00) | 1.3151.315\mathbf{1.315} (1.001.001.00) |

### 5.2 Updating TabR with new training data without retraining

Getting access to new unseen training data after training a machine learning model (e.g., after collecting yet another portion of daily logs of an application) is a common practical scenario.
Technically, TabR allows utilizing the new data without retraining by adding the new data to the set of candidates for retrieval.
We test this approach on the full “Weather prediction” dataset (Malinin et al., [2021](#bib.bib32)) (3M+ objects).
[Figure 6](#S5.F6 "Figure 6 ‣ 5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") indicates that such “online updates” may be a viable solution for incorporating new data into an already trained TabR.
Additionally, this approach can be used to scale TabR to large datasets by training the model on a subset of data and retrieving from the full data.
Overall, we consider the conducted experiment as a preliminary exploration and leave a systematic study of continual updates for future work.
See [subsection D.3](#A4.SS3 "D.3 Implementation details of subsection 5.2 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for implementation details.

Figure 5: 
ΔΔ\Delta-context (explained below) averaged over training objects until the early stopping while training TabR-S.
On a given epoch, for a given object, ΔΔ\Delta-context shows the portion of its context (the top-m𝑚m candidates and their weights) changed compared to the previous epoch (i.e., the lower the value, the smaller the change; see [subsection D.2](#A4.SS2 "D.2 Implementation details of subsection 5.1 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") for formal details).
The plot shows that context updates become less intensive during the course of training, which motivates the optimization described in [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Figure 6: 
Training TabR-S on various portions of the training data of the full “Weather prediction” dataset and gradually adding the remaining unseen training data to the set of candidates without retraining as described in [subsection 5.2](#S5.SS2 "5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
For each curve, the leftmost point corresponds to not adding any new data to the set of candidates after the training, and the rightmost point corresponds to adding all unseen training data to the set of candidates.

### 5.3 Further analysis

In the appendix, we provide a more insightful analysis. A non-exhaustive list of examples:

* •

  in [subsection A.1](#A1.SS1 "A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we analyze the key-only L2subscript𝐿2L\_{2}-based similarity module 𝒮𝒮\mathcal{S} introduced on Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), which was a turning point in our story.
  We provide intuition behind this specific implementation of 𝒮𝒮\mathcal{S} and perform an in-depth comparison with the similarity module of the vanilla attention (the dot product between queries and keys).
* •

  in [subsection A.2](#A1.SS2 "A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we analyze the value module 𝒱𝒱\mathcal{V} introduced on Step-3 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
  On regression problems, we confirm the correction semantics of the module T𝑇T from [Equation 4](#S3.E4 "4 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
* •

  in [subsection A.4](#A1.SS4 "A.4 Comparing training times ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we compare training times of TabR with training times of all the baselines.
  We show that compared to prior retrieval-based tabular models, TabR makes a big step forward in terms of efficiency.
  While TabR is relatively slower than simple retrieval-free models, within the considered scope of dataset sizes, the absolute training times of TabR are affordable for most practical scenarios.
* •

  in [subsection A.7](#A1.SS7 "A.7 Additional technical notes on TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we highlight additional technical properties of TabR.

## 6 Conclusion & Future work

In this work, we have demonstrated that retrieval-based deep learning models have great potential in supervised machine learning problems on tabular data.
Namely, we have designed TabR – a retrieval-augmented tabular DL architecture that provides strong average performance and achieves the new state-of-the-art on several datasets.
Importantly, we have highlighted similarity and value modules as the important details of the attention mechanism which have a significant impact on the performance of attention-based retrieval components.

An important direction for future work is improving the efficiency of retrieval-augmented models to make them faster in general and in particular applicable to tens and hundreds of millions of data points.
Also, in this paper, we focused more on the aspect of task performance, so some other properties of TabR remain underexplored.
For example, the retrieval nature of TabR provides new opportunities for interpreting the model’s predictions through the influence of context objects.
Also, TabR may enable better support for continual learning (we scratched the surface of this direction in [subsection 5.2](#S5.SS2 "5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
Regarding architecture details, possible directions are improving similarity and value modules, as well as performing multiple rounds of retrieval and interactions with the retrieved instances.

## Reproducibility Statement

To make the results and models reproducible and verifiable, we provide our full codebase, all the results, and step-by-step usage instructions: [link](https://github.com/yandex-research/tabular-dl-tabr).
In particular, (1) the results and hyperparameters reported the paper is just a summary of the results available at the provided URL (with minor exceptions); (2) implementations of TabR and all the baselines (except for NPT) are available; (3) the hyperparameter tuning, training and evaluation pipelines are available; (4) the hyperparameters are available; (5) the used datasets and splits are available; (6) hyperparameter tuning and training times are available; (7) the used hardware is available; (8) within a fixed environment (i.e. fixed hardware and software versions), most of the results are bitwise reproducible.

## References

* Akiba et al. [2019]

  T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, 2019.
* Ba et al. [2016]

  J. L. Ba, J. R. Kiros, and G. E. Hinton.
  Layer normalization.
  *arXiv*, 1607.06450v1, 2016.
* Baldi et al. [2014]

  P. Baldi, P. Sadowski, and D. Whiteson.
  Searching for exotic particles in high-energy physics with deep learning.
  *Nature Communications*, 5, 2014.
* Blackard and Dean. [2000]

  J. A. Blackard and D. J. Dean.
  Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables.
  *Computers and Electronics in Agriculture*, 24(3):131–151, 2000.
* Borgeaud et al. [2022]

  S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J. Lespiau, B. Damoc, A. Clark, D. de Las Casas, A. Guy, J. Menick, R. Ring, T. Hennigan, S. Huang, L. Maggiore, C. Jones, A. Cassirer, A. Brock, M. Paganini, G. Irving, O. Vinyals, S. Osindero, K. Simonyan, J. W. Rae, E. Elsen, and L. Sifre.
  Improving language models by retrieving from trillions of tokens.
  In *ICML*, 2022.
* Bottou and Vapnik [1992]

  L. Bottou and V. Vapnik.
  Local learning algorithms.
  *Neural Computation*, 4, 1992.
* Chen and Guestrin [2016]

  T. Chen and C. Guestrin.
  Xgboost: A scalable tree boosting system.
  In *SIGKDD*, 2016.
* Das et al. [2021]

  R. Das, M. Zaheer, D. Thai, A. Godbole, E. Perez, J. Y. Lee, L. Tan, L. Polymenakos, and A. McCallum.
  Case-based reasoning for natural language queries over knowledge bases.
  In *EMNLP*, 2021.
* Du et al. [2022]

  K. Du, W. Zhang, R. Zhou, Y. Wang, X. Zhao, J. Jin, Q. Gan, Z. Zhang, and D. P. Wipf.
  Learning enhanced representation for tabular data via neighborhood propagation.
  In *NeurIPS*, 2022.
* Gardner et al. [2018]

  J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, and A. G. Wilson.
  Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration.
  In *Advances in Neural Information Processing Systems*, 2018.
* Gorishniy et al. [2021]

  Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, 2021.
* Gorishniy et al. [2022]

  Y. Gorishniy, I. Rubachev, and A. Babenko.
  On embeddings for numerical features in tabular deep learning.
  In *NeurIPS*, 2022.
* Grinsztajn et al. [2022]

  L. Grinsztajn, E. Oyallon, and G. Varoquaux.
  Why do tree-based models still outperform deep learning on typical tabular data?
  In *NeurIPS, the "Datasets and Benchmarks" track*, 2022.
* Guu et al. [2020]

  K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang.
  Retrieval augmented language model pre-training.
  In *ICML*, 2020.
* Hazimeh et al. [2020]

  H. Hazimeh, N. Ponomareva, P. Mol, Z. Tan, and R. Mazumder.
  The tree ensemble layer: Differentiability meets conditional computation.
  In *ICML*, 2020.
* Huang et al. [2020]

  X. Huang, A. Khetan, M. Cvitkovic, and Z. Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  *arXiv*, 2012.06678v1, 2020.
* Iscen et al. [2022]

  A. Iscen, T. Bird, M. Caron, A. Fathi, and C. Schmid.
  A memory transformer network for incremental learning.
  *arXiv*, abs/2210.04485v1, 2022.
* Izacard et al. [2022]

  G. Izacard, P. S. H. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave.
  Few-shot learning with retrieval augmented language models.
  *arXiv*, abs/2208.03299v3, 2022.
* James et al. [2013]

  G. James, D. Witten, T. Hastie, and R. Tibshirani.
  *An Introduction to Statistical Learning*.
  Springer, 2013.
  <https://www.statlearning.com/>.
* Jia et al. [2021]

  M. Jia, B.-C. Chen, Z. Wu, C. Cardie, S. Belongie, and S.-N. Lim.
  Rethinking nearest neighbors for visual classification.
  *arXiv preprint arXiv:2112.08459*, 2021.
* Kadra et al. [2021]

  A. Kadra, M. Lindauer, F. Hutter, and J. Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In *NeurIPS*, 2021.
* Ke et al. [2017]

  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30:3146–3154, 2017.
* Kelley Pace and Barry [1997]

  R. Kelley Pace and R. Barry.
  Sparse spatial autoregressions.
  *Statistics & Probability Letters*, 33(3):291–297, 1997.
* Khandelwal et al. [2020]

  U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, and M. Lewis.
  Generalization through memorization: Nearest neighbor language models.
  In *ICLR*, 2020.
* Kim et al. [2019]

  H. Kim, A. Mnih, J. Schwarz, M. Garnelo, S. M. A. Eslami, D. Rosenbaum, O. Vinyals, and Y. W. Teh.
  Attentive neural processes.
  In *ICLR*, 2019.
* Klambauer et al. [2017]

  G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter.
  Self-normalizing neural networks.
  In *NIPS*, 2017.
* Kohavi [1996]

  R. Kohavi.
  Scaling up the accuracy of naive-bayes classifiers: a decision-tree hybrid.
  In *KDD*, 1996.
* Kossen et al. [2021]

  J. Kossen, N. Band, C. Lyle, A. N. Gomez, T. Rainforth, and Y. Gal.
  Self-attention between datapoints: Going beyond individual input-output pairs in deep learning.
  In *NeurIPS*, 2021.
* Lewis et al. [2020]

  P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela.
  Retrieval-augmented generation for knowledge-intensive NLP tasks.
  In *NeurIPS*, 2020.
* Long et al. [2022]

  A. Long, W. Yin, T. Ajanthan, V. Nguyen, P. Purkait, R. Garg, A. Blair, C. Shen, and A. van den Hengel.
  Retrieval augmented classification for long-tail visual recognition.
  In *CVPR*, 2022.
* Loshchilov and Hutter [2019]

  I. Loshchilov and F. Hutter.
  Decoupled weight decay regularization.
  In *ICLR*, 2019.
* Malinin et al. [2021]

  A. Malinin, N. Band, G. Chesnokov, Y. Gal, M. J. F. Gales, A. Noskov, A. Ploskonosov, L. Prokhorenkova, I. Provilkov, V. Raina, V. Raina, M. Shmatova, P. Tigas, and B. Yangel.
  Shifts: A dataset of real distributional shift across multiple large-scale tasks.
  *ArXiv*, abs/2107.07455v3, 2021.
* Nader et al. [2022]

  Y. Nader, L. Sixt, and T. Landgraf.
  Dnnr: Differential nearest neighbors regression.
  In *ICML*, 2022.
* Pedregosa et al. [2011]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12:2825–2830, 2011.
* Popov et al. [2020]

  S. Popov, S. Morozov, and A. Babenko.
  Neural oblivious decision ensembles for deep learning on tabular data.
  In *ICLR*, 2020.
* Prokhorenkova et al. [2018]

  L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Qin et al. [2020]

  J. Qin, W. Zhang, X. Wu, J. Jin, Y. Fang, and Y. Yu.
  User behavior retrieval for click-through rate prediction.
  In *SIGIR*, 2020.
* Qin et al. [2021]

  J. Qin, W. Zhang, R. Su, Z. Liu, W. Liu, R. Tang, X. He, and Y. Yu.
  Retrieval & interaction machine for tabular data prediction.
  In *KDD*, 2021.
* Qin and Liu [2013]

  T. Qin and T. Liu.
  Introducing LETOR 4.0 datasets.
  *arXiv*, 1306.2597v1, 2013.
* Ramsauer et al. [2021]

  H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, T. Adler, D. P. Kreil, M. K. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter.
  Hopfield networks is all you need.
  In *ICLR*, 2021.
* Schäfl et al. [2022]

  B. Schäfl, L. Gruber, A. Bitto-Nemling, and S. Hochreiter.
  Hopular: Modern hopfield networks for tabular data.
  *arXiv*, abs/2206.00664, 2022.
* Somepalli et al. [2021]

  G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss, and T. Goldstein.
  SAINT: improved neural networks for tabular data via row attention and contrastive pre-training.
  *arXiv*, 2106.01342v1, 2021.
* Vanschoren et al. [2014]

  J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo.
  Openml: networked science in machine learning.
  *arXiv*, 1407.7722v1, 2014.
* Vaswani et al. [2017]

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin.
  Attention is all you need.
  In *NIPS*, 2017.
* Wang and Sabuncu [2023]

  A. Q. Wang and M. R. Sabuncu.
  A flexible nadaraya-watson head can offer explainable and calibrated classification.
  In *TMLR*, 2023.
* Wang et al. [2020]

  R. Wang, R. Shivanna, D. Z. Cheng, S. Jain, D. Lin, L. Hong, and E. H. Chi.
  Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.
  *arXiv*, 2008.13535v2, 2020.
* Wang et al. [2022]

  S. Wang, Y. Xu, Y. Fang, Y. Liu, S. Sun, R. Xu, C. Zhu, and M. Zeng.
  Training data is more valuable than you think: A simple and effective method by retrieving from training data.
  *arXiv preprint arXiv:2203.08773*, 2022.
* Wilson et al. [2016]

  A. G. Wilson, Z. Hu, R. Salakhutdinov, and E. P. Xing.
  Deep kernel learning.
  In *AISTATS*, 2016.
* Zhao and Cho [2018]

  J. Zhao and K. Cho.
  Retrieval-augmented convolutional neural networks for improved robustness against adversarial examples.
  *arXiv preprint arXiv:1802.09502*, 2018.

## Supplementary material

## Appendix A Additional analysis

### A.1 Similarity module of TabR

#### A.1.1 Intuitive motivation

Recall that in Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), the change from the similarity module of the vanilla attention to the new key-only L2subscript𝐿2L\_{2}-driven similarity module was a turning point in our story, where a retrieval-based model started showing noticeable improvements over MLP on several datasets.
In fact, in addition to the empirical results (in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and [subsection A.3](#A1.SS3 "A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")), this specific similarity module has a reasonable intuitive motivation, which we now provide.

* •

  First, aligning two (query and key) representations of target and candidate objects is an additional challenge for the optimization process, and there is no clear motivation for introducing this challenge in our case.
  And, as demonstrated in [subsection A.3](#A1.SS3 "A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), avoiding this challenge is not just beneficial, but rather necessary.
* •

  Second, during the design process in [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), the similarity module 𝒮𝒮\mathcal{S} operates over linear transformations of the input (because, at that point, encoder E𝐸E is just a linear layer, since we fixed NE=0subscript𝑁𝐸0N\_{E}=0).
  Then, a reasonable similarity measure in the original feature space may remain reasonable in the transformed feature space.
  And, for tabular data, L2subscript𝐿2L\_{2} is usually a better similarity measure than the dot product in the original feature space.
  Note that the case of shallow/linear encoder is a specific, but very important case: since E𝐸E is applied to many candidates on each training step, E𝐸E is better to be lightweight to maintain adequate efficiency.

Combined, the above two points motivate removing query representations and switching to the L2subscript𝐿2L\_{2} distance, which leads to the similarity module introduced in Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

#### A.1.2 Analyzing attention patterns over candidates

In this section, we analyze the similarity module 𝒮𝒮\mathcal{S} introduced in Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), which greatly improved the performance on several datasets in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Formally, for a given input object, the similarity module defines a distribution over candidates (“weights” in [Figure 4](#S3.F4 "Figure 4 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) with exactly m+1𝑚1m+1 non-zero entries (m𝑚m is the context size; +11+1 comes from adding the target object to its own context in Step-2).
Intuitively, the less diverse such distributions are on average, the more frequently different input objects are augmented with similar contexts.
In [Table 7](#A1.T7 "Table 7 ‣ A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we demonstrate that such distributions are more diverse on average with the new similarity module compared to the one from the vanilla attention.
The implementation details are provided in [subsection D.4](#A4.SS4 "D.4 Implementation details of subsubsection A.1.2 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Table 7: 
Entropy of the average distribution over candidates (the averaging is performed over individual distributions for test objects).
The distributions are produced by the similarity module as explained in [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The trained Step-1 and Step-2 models are taken directly from [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The similarity module introduced at Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") produces more diverse contexts.

|  | CH | CA | HO | AD | DI | OT | HI | WE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step-1 | 6.66.66.6 | 6.16.16.1 | 7.07.07.0 | 7.17.17.1 | 5.85.85.8 | 5.35.35.3 | 8.58.58.5 | 8.98.98.9 |
| Step-2 | 8.48.48.4 | 9.09.09.0 | 9.39.39.3 | 9.79.79.7 | 10.310.310.3 | 10.110.110.1 | 10.510.510.5 | 9.59.59.5 |
| Uniform | 8.88.88.8 | 9.59.59.5 | 9.69.69.6 | 10.210.210.2 | 10.410.410.4 | 10.610.610.6 | 11.011.011.0 | 12.612.612.6 |

#### A.1.3 Case studies

In this section, we consider three datasets where the transition to the key-only L2subscript𝐿2L\_{2} similarity module from the vanilla dot-product-between-queries-and-keys demonstrated the most impressive performance.
Formally, this is the transition from “Step-1” to “Step-2” in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
For each of the three datasets, first, we notice that, for a given input object, there is a domain-specific notion of “good neighbors”, i.e., such neighbors that, from a human perspective, are very relevant to the input object and provide strong hints for making a better prediction for the input object.
Then, we show that the new similarity module allows finding and exploiting those natural hints.

California housing (CA).
On this dataset, the transition from the “vanilla” dot-product-between-queries-and-keys similarity module to the key-only L2subscript𝐿2L\_{2} similarity module resulted in a substantial performance boost, as indicated by the difference between “Step-1” and “Step-2” in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
On this dataset, the task is to estimate the prices of houses in California.
Intuitively, for a given house from the test set, the prices of the training houses in the geographical neighborhood should be a strong hint for solving the task.
Moreover, there are coordinates (longitude and latitude) among the features, which should simplify finding good neighbors.
And the “Step-2” model successfully does that, which is not true for the “Step-1” model.
Specifically, for an average test object, the “Step-2” model concentrates approximately 7% of the attention mass on the object itself (recall that “Step-2” includes the target object in the context objects) and approximately 77% on the context objects within the 10km radius. The corresponding numbers of the “Step-1” model are 0.07% and 1%.

Weather prediction (WE).
Here, the story is seemingly similar to the one with the CA dataset analyzed in the previous paragraph, but in fact has a major difference.
Again, here, for a given test data point, the dataset contains natural hints in the form of geographical neighbors from the training set which allow making a better weather forecast for a test query; and the “Step-2” model ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) successfully exploits that, while the “Step-1” model cannot pay any meaningful attention to those hints.
Specifically, for an average object, the “Step-2” model concentrates approximately 29% of the attention mass on the object itself (recall that “Step-2” includes the target object in the context objects) and approximately 25% on the context objects within the 200km radius. The corresponding numbers of the “Step-1” model are 0.25% and 0.5%.
However, there is a crucial distinction from the CA case: in the version of the dataset WE that we used, the features did not contain the coordinates.
In other words, to perform the analysis, after the training, we restored the original coordinates for each row from the original dataset and observed that the model learned the “correct” notion of “good neighbors” from other features.

Facebook comments volume (FB).
In this paper, this is the first time when we mention this dataset, which was used in prior work [Gorishniy et al., [2022](#bib.bib12)] and which we also used for some time in this project.
Notably, on this dataset, TabR was demonstrating unthinkable improvements over competitors (including GBDT and the best-in-class parametric DL models).
Then we noticed a strange pattern: often, for a given input, TabR concentrated an abnormally high percentage of its attention mass on just one context object (a different one for each input object).
This is how we discovered that the dataset split that we inherited from Gorishniy et al. [[2022](#bib.bib12)] contained a “leak”: roughly speaking, for many objects, it was possible to find their almost exact copies in the training set, and the task was dramatically simpler with this kind of hint.
In practice, it was dramatically simpler for the TabR, but not for other models.
Specifically, for an average object, the “Step-2” model concentrates approximately 20% of the attention mass on the object itself (recall that “Step-2” includes the target object in the context objects) and approximately 35% on its leaked almost-copies. The corresponding numbers of the “Step-1” model are 0.5% and 0.09%.

### A.2 Analyzing the value module of TabR

In this section, we analyze the value module 𝒱𝒱\mathcal{V} of TabR (see [Equation 5](#S3.E5 "5 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")):

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒱​(x~,x~i,yi)=WY​(yi)+T​(k−ki)=WY​(yi)+T​(Δ​ki)𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑌subscript𝑦𝑖𝑇𝑘subscript𝑘𝑖subscript𝑊𝑌subscript𝑦𝑖𝑇Δsubscript𝑘𝑖\displaystyle\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{Y}(y\_{i})+T(k-k\_{i})=W\_{Y}(y\_{i})+T(\Delta k\_{i}) |  | (6) |

Intuitively, for a given context object, its label yisubscript𝑦𝑖y\_{i} can be an important part of its contribution to the prediction.
Let’s consider regression problems, where, in [Equation 6](#A1.E6 "6 ‣ A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), yi∈ℝsubscript𝑦𝑖ℝy\_{i}\in\mathbb{R} is embedded by WYsubscript𝑊𝑌W\_{Y} to 𝕐~⊂ℝd~𝕐superscriptℝ𝑑\tilde{\mathbb{Y}}\subset\mathbb{R}^{d}.
Since WYsubscript𝑊𝑌W\_{Y} is a linear layer, 𝕐~~𝕐\tilde{\mathbb{Y}} is just a line, and each point on this line can be mapped back to the corresponding label from ℝℝ\mathbb{R}.
Then, the projection of the correction term T​(Δ​ki)𝑇Δsubscript𝑘𝑖T(\Delta k\_{i}) on 𝕐~~𝕐\tilde{\mathbb{Y}} can be translated to the correction of the context label yisubscript𝑦𝑖y\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒱​(x~,x~i,yi)=WY​(yi)+proj𝕐~​T​(Δ​ki)¯+proj𝕐~⟂​T​(Δ​ki)=WY​(yi+Δ​yi¯)+proj𝕐~⟂​T​(Δ​ki)𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖subscript𝑊𝑌subscript𝑦𝑖¯subscriptproj~𝕐𝑇Δsubscript𝑘𝑖subscriptprojsuperscript~𝕐perpendicular-to𝑇Δsubscript𝑘𝑖subscript𝑊𝑌subscript𝑦𝑖¯Δsubscript𝑦𝑖subscriptprojsuperscript~𝕐perpendicular-to𝑇Δsubscript𝑘𝑖\displaystyle\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i})=W\_{Y}(y\_{i})+\underline{\text{proj}\_{\tilde{\mathbb{Y}}}T(\Delta k\_{i})}+\text{proj}\_{\tilde{\mathbb{Y}}^{\perp}}T(\Delta k\_{i})=W\_{Y}(y\_{i}+\underline{\Delta y\_{i}})+\text{proj}\_{\tilde{\mathbb{Y}}^{\perp}}T(\Delta k\_{i}) |  | (7) |

To check whether the underlined correction term proj𝕐~​T​(Δ​ki)subscriptproj~𝕐𝑇Δsubscript𝑘𝑖\text{proj}\_{\tilde{\mathbb{Y}}}T(\Delta k\_{i}) (or Δ​yiΔsubscript𝑦𝑖\Delta y\_{i}) is important, we take a trained TabR, and reevaluate it without retraining while ignoring this projection (which is equivalent to setting Δ​yi=0Δsubscript𝑦𝑖0\Delta y\_{i}=0).
As a baseline, we also try ignoring the projection of T​(Δ​ki)𝑇Δsubscript𝑘𝑖T(\Delta k\_{i}) on a random one-dimensional subspace instead of 𝕐~~𝕐\tilde{\mathbb{Y}}.
[Table 8](#A1.T8 "Table 8 ‣ A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") indicates that the correction along 𝕐~~𝕐\tilde{\mathbb{Y}} plays a vital role for the model.
The implementation details are provided in [subsection D.5](#A4.SS5 "D.5 Implementation details of subsection A.2 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Table 8: 
Evaluating RMSE of trained TabR-S while ignoring projections of T​(Δ​ki)𝑇Δsubscript𝑘𝑖T(\Delta k\_{i}) on different one-dimensional subspaces as described in [subsection A.2](#A1.SS2 "A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The first column shows the projection on which one-dimensional subspace is removed from T​(Δ​ki)𝑇Δsubscript𝑘𝑖T(\Delta k\_{i}).
The first row corresponds to not removing any projections (i.e., the unmodified TabR-S).
Ignoring the projection on 𝕐~~𝕐\tilde{\mathbb{Y}} (the label embedding space) breaks the model while ignoring a random projection does not have much effect.

|  | CA ↓ | HO ↓ | DI ↓ | BL ↓ | WE ↓ |
| --- | --- | --- | --- | --- | --- |
| – | 0.4030.4030.403 | 3.0673.0673.067 | 0.1330.1330.133 | 0.6900.6900.690 | 1.7471.7471.747 |
| random | 0.4030.4030.403 | 3.0713.0713.071 | 0.1330.1330.133 | 0.6900.6900.690 | 1.7541.7541.754 |
| 𝕐~~𝕐\tilde{\mathbb{Y}} | 0.4650.4650.465 | 3.6493.6493.649 | 0.3640.3640.364 | 0.6950.6950.695 | 2.0032.0032.003 |

For classification problems, we tested similar hypotheses but did not obtain any interesting results.
Perhaps, the value module 𝒱𝒱\mathcal{V} and specifically the T𝑇T module should be designed differently to better model the nature of classification problems.

### A.3 Ablation study

Recall that on Step-2 of [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we mentioned that it was crucial that all changes from Step-2 compared to Step-0 (using labels + not using queries + using the L2subscript𝐿2L\_{2} distance instead of the dot product) are important to provide noticeable improvements over MLP on several datasets.
Note that not using queries is equivalent to sharing weights of WQsubscript𝑊𝑄W\_{Q} and WKsubscript𝑊𝐾W\_{K}: WQ=WKsubscript𝑊𝑄subscript𝑊𝐾W\_{Q}=W\_{K}.
[Table 9](#A1.T9 "Table 9 ‣ A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") contains the results of the corresponding experiment and indeed demonstrates that the Step-2 configuration cannot be trivially simplified without loss in metrics (see the CH, CA, BL, WE datasets).

Overall, we hypothesize that both things are important: how valuable the additional signal is (Step-1) and how well we measure the distance from the target object to the source of that valuable signal (Step-2).

Table 9: 
The ablation study as described in [subsection A.3](#A1.SS3 "A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
WQ=WKsubscript𝑊𝑄subscript𝑊𝐾W\_{Q}=W\_{K} means using only keys and not using queries.
Step-2 is the only variation providing noticeable improvements over MLP on the CH, CA, BL, WE datasets.

|  | L2,subscript𝐿2L\_{2}, | WQ=WK,subscript𝑊𝑄subscript𝑊𝐾W\_{Q}=W\_{K}, | WYsubscript𝑊𝑌W\_{Y} | CH | CA | HO | AD | DI | OT | HI | BL | WE | Avg. Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP |  |  |  | 0.8540.8540.854 | 0.4990.4990.499 | 3.1123.1123.112 | 0.8530.8530.853 | 0.1400.1400.140 | 0.8160.8160.816 | 0.7190.7190.719 | 0.6970.6970.697 | 1.9051.9051.905 | 2.4±1.4plus-or-minus2.41.42.4\pm 1.4 |
| Step-0 | ✗ | ✗ | ✗ | 0.8550.8550.855 | 0.4840.4840.484 | 3.2343.2343.234 | 0.8570.8570.857 | 0.1420.1420.142 | 0.8140.8140.814 | 0.7190.7190.719 | 0.6990.6990.699 | 1.9031.9031.903 | 2.4±0.9plus-or-minus2.40.92.4\pm 0.9 |
| Step-1 | ✗ | ✗ | ✓ | 0.8550.8550.855 | 0.4890.4890.489 | 3.2053.2053.205 | 0.8570.8570.857 | 0.1420.1420.142 | 0.8140.8140.814 | 0.7190.7190.719 | 0.6980.6980.698 | 1.9061.9061.906 | 2.4±1.2plus-or-minus2.41.22.4\pm 1.2 |
|  | ✗ | ✓ | ✗ | 0.8530.8530.853 | 0.4950.4950.495 | 3.1783.1783.178 | 0.8570.8570.857 | 0.1430.1430.143 | 0.8080.8080.808 | 0.7190.7190.719 | 0.6980.6980.698 | 1.9031.9031.903 | 2.9±0.8plus-or-minus2.90.82.9\pm 0.8 |
|  | ✗ | ✓ | ✓ | 0.8570.8570.857 | 0.4950.4950.495 | 3.2173.2173.217 | 0.8570.8570.857 | 0.1410.1410.141 | 0.8080.8080.808 | 0.7170.7170.717 | 0.6980.6980.698 | 1.8811.8811.881 | 2.7±0.7plus-or-minus2.70.72.7\pm 0.7 |
|  | ✓ | ✗ | ✗ | 0.8550.8550.855 | 0.4880.4880.488 | 3.1703.1703.170 | 0.8570.8570.857 | 0.1430.1430.143 | 0.8130.8130.813 | 0.7190.7190.719 | 0.6980.6980.698 | 1.9011.9011.901 | 2.3±1.0plus-or-minus2.31.02.3\pm 1.0 |
|  | ✓ | ✗ | ✓ | 0.8560.8560.856 | 0.4980.4980.498 | 3.2063.2063.206 | 0.8580.8580.858 | 0.1420.1420.142 | 0.8120.8120.812 | 0.7210.7210.721 | 0.6990.6990.699 | 1.9001.9001.900 | 2.4±1.1plus-or-minus2.41.12.4\pm 1.1 |
|  | ✓ | ✓ | ✗ | 0.8560.8560.856 | 0.4420.4420.442 | 3.1543.1543.154 | 0.8560.8560.856 | 0.1410.1410.141 | 0.8110.8110.811 | 0.7220.7220.722 | 0.6980.6980.698 | 1.8961.8961.896 | 2.0±0.7plus-or-minus2.00.72.0\pm 0.7 |
| Step-2 | ✓ | ✓ | ✓ | 0.8600.8600.860 | 0.4180.4180.418 | 3.1533.1533.153 | 0.8580.8580.858 | 0.1400.1400.140 | 0.8130.8130.813 | 0.7200.7200.720 | 0.6920.6920.692 | 1.8041.8041.804 | 1.2±0.4plus-or-minus1.20.41.2\pm 0.4 |

### A.4 Comparing training times

While TabR demonstrates strong performance, these benefits do not come for free, since, as with all retrieval-augmented models, the retrieval component of TabR brings additional overhead.
In this section, we aim to quantify this overhead by comparing the training times of TabR with those of all the baselines.
[Table 10](#A1.T10 "Table 10 ‣ A.4 Comparing training times ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") shows two important things:

* •

  first, TabR is significantly more efficient (i.e. provides a significantly better trade-off between the downstream performance and training times) than prior retrieval-augmented tabular models.
  In particular, TabR is significantly (and, sometimes, dramatically) more efficient than NPT [Kossen et al., [2021](#bib.bib28)] – the closest retrieval-based competitor according to [Table 3](#S4.T3 "Table 3 ‣ 4.1 Evaluating retrieval-augmented deep learning models for tabular data ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
* •

  second, within the considered scope of dataset sizes, the absolute training times of TabR will be affordable in practice.
  Moreover, the reported execution times are achieved with our naive implementation which lacks even some of the basic optimizations.

To sum up, compared to prior work on retrieval-based tabular DL, TabR makes a big step forward in terms of efficiency.
TabR is relatively slower than simple models (GBDT, parametric DL models), and improving its efficiency is an important research direction.
However, given the room for technical optimizations and techniques similar to context freeze ([subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")), the future of retrieval-based tabular DL looks positive.

Table 10: 
Training times of the tuned models (from [Table 3](#S4.T3 "Table 3 ‣ 4.1 Evaluating retrieval-augmented deep learning models for tabular data ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), [Table 4](#S4.T4 "Table 4 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and [Table 6](#S5.T6 "Table 6 ‣ 5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) averaged over the random seeds.
The format is hh:mm:ss.
TabR-S (CF-4) is TabR-S with the context freeze ([subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")) applied after four epochs.
Colors describe to the following informal tiers:
  
 ■■\blacksquare <5 minutes    ■■\blacksquare <30 minutes    ■■\blacksquare <2 hours    ■■\blacksquare <10 hours    ■■\blacksquare >10 hours

|  | CH | CA | HO | AD | DI | OT | HI | BL | WE | CO | MI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0:00:01 | 0:00:20 | 0:00:05 | 0:00:05 | 0:00:02 | 0:00:35 | 0:00:15 | 0:00:08 | 0:02:02 | 0:01:55 | 0:03:43 |
| LightGBM | 0:00:00 | 0:00:04 | 0:00:01 | 0:00:01 | 0:00:03 | 0:00:34 | 0:00:10 | 0:00:07 | 0:06:40 | 0:06:22 | 0:06:45 |
| MLP | 0:00:02 | 0:00:18 | 0:00:09 | 0:00:17 | 0:00:15 | 0:00:31 | 0:00:24 | 0:01:38 | 0:00:29 | 0:04:01 | 0:02:09 |
| MLP-PLR | 0:00:03 | 0:00:43 | 0:00:14 | 0:00:24 | 0:00:25 | 0:02:09 | 0:00:17 | 0:00:52 | 0:20:01 | 0:03:32 | 0:30:30 |
| Retrieval-augmented models | | | | | | | | | | | |
| TabR-S (CF-4) | 0:00:08 | 0:00:25 | 0:00:30 | 0:00:34 | 0:00:43 | 0:00:57 | 0:01:02 | 0:03:08 | 0:09:08 | 0:23:13 | – |
| TabR-S | 0:00:20 | 0:01:20 | 0:01:23 | 0:03:04 | 0:01:44 | 0:01:17 | 0:02:09 | 0:11:22 | 0:12:11 | 0:49:59 | 0:55:04 |
| TabR | 0:00:16 | 0:00:40 | 0:00:55 | 0:01:30 | 0:01:24 | 0:01:47 | 0:06:22 | 0:04:14 | 1:03:18 | 0:37:03 | 1:46:07 |
| DKL | – | 0:06:15 | 0:03:55 | – | 0:21:59 | – | – | 1:04:10 | – | – | – |
| ANP | – | 0:37:40 | 0:42:16 | – | 2:14:38 | – | – | 1:32:27 | 6:00:11 | – | – |
| SAINT | 0:00:23 | 0:06:04 | 0:01:44 | 0:00:58 | 0:01:55 | 0:05:37 | 0:03:47 | 0:06:22 | 2:55:51 | 6:17:20 | 5:39:37 |
| NPT | 0:08:44 | 0:06:58 | 0:12:21 | 0:11:22 | 0:54:55 | 10:45:42 | 3:26:47 | 0:55:04 | 5:28:56 | 12:05:28 | 8:07:36 |

### A.5 Augmenting XGBoost with a retrieval component

After the successful results of TabR reported in [subsection 4.2](#S4.SS2 "4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we tried augmenting XGBoost with a simple retrieval component to ensure that we do not miss this opportunity to improve the baselines.
Namely, for a given input object, we find m=96𝑚96m=96 (equal to the context size of TabR) nearest training objects in the original feature space, average their features and labels (the label as-is for regression problems, the one-hot encoding representations for classification problems), concatenate the target object’s features with the “average neighbor’s” features and label, and the obtained vector is used as the input for XGBoost.
The results in [Table 11](#A1.T11 "Table 11 ‣ A.5 Augmenting XGBoost with a retrieval component ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") indicate that this strategy does not lead to any noticeable profit for XGBoost.
We tried to vary the number of neighbors but did not achieve any significant improvements.

Table 11: Results for ensembles of tuned models. “XGBoost + retrieval” stands for XGBoost augmented with the “average neighbor’s” features and label as described in [subsection A.5](#A1.SS5 "A.5 Augmenting XGBoost with a retrieval component ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

|  | CH ↑ | CA ↓ | HO ↓ | AD ↑ | DI ↓ | OT ↑ | HI ↑ | BL ↓ | WE ↓ | CO ↑ | MI ↓ | Avg. Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.8610.8610.861 | 0.4320.4320.432 | 3.1643.1643.164 | 0.8720.872\mathbf{0.872} | 0.1360.1360.136 | 0.8320.832\mathbf{0.832} | 0.7260.7260.726 | 0.6800.6800.680 | 1.7691.7691.769 | 0.9710.9710.971 | 0.7410.741\mathbf{0.741} | 1.9±0.7plus-or-minus1.90.71.9\pm 0.7 |
| XGBoost + retrieval | 0.8550.8550.855 | 0.4360.4360.436 | 3.1343.1343.134 | 0.8710.8710.871 | 0.1330.1330.133 | 0.8150.8150.815 | 0.7240.7240.724 | 0.6870.6870.687 | 1.7881.7881.788 | 0.9620.9620.962 | 0.7430.7430.743 | 2.5±0.5plus-or-minus2.50.52.5\pm 0.5 |
| TabR | 0.8650.865\mathbf{0.865} | 0.3910.391\mathbf{0.391} | 3.0253.025\mathbf{3.025} | 0.8720.872\mathbf{0.872} | 0.1310.131\mathbf{0.131} | 0.8310.831\mathbf{0.831} | 0.7330.733\mathbf{0.733} | 0.6740.674\mathbf{0.674} | 1.6611.661\mathbf{1.661} | 0.9770.977\mathbf{0.977} | 0.7480.7480.748 | 1.2±0.6plus-or-minus1.20.61.2\pm 0.6 |

### A.6 Additional results for the “context freeze” technique

We report the extended results for [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") in [Figure 7](#A1.F7 "Figure 7 ‣ A.6 Additional results for the “context freeze” technique ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), [Table 12](#A1.T12 "Table 12 ‣ A.6 Additional results for the “context freeze” technique ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and [Table 13](#A1.T13 "Table 13 ‣ A.6 Additional results for the “context freeze” technique ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
For the formal definition of the ΔΔ\Delta-context metric, see [subsection D.2](#A4.SS2 "D.2 Implementation details of subsection 5.1 ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

!(/html/2307.14338/assets/x6.png)

Figure 7: The extended version of [Figure 5](#S5.F5 "Figure 5 ‣ 5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") with more datasets.

Table 12: 
The extended version of [Table 6](#S5.T6 "Table 6 ‣ 5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
Freezing after 0 epochs means freezing with a randomly initialized model.
The speedups are provided in [Table 13](#A1.T13 "Table 13 ‣ A.6 Additional results for the “context freeze” technique ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")

|  | CH ↑ | CA ↓ | HO ↓ | AD ↑ | DI ↓ | OT ↑ | HI ↑ | BL ↓ | WE ↓ | CO ↑ | WE (full) ↓ | Avg. Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.8540.8540.854 | 0.4990.4990.499 | 3.1123.1123.112 | 0.8530.8530.853 | 0.1400.1400.140 | 0.8160.816\mathbf{0.816} | 0.7190.719\mathbf{0.719} | 0.6970.6970.697 | 1.9051.9051.905 | 0.9630.9630.963 | – | 2.9±1.5plus-or-minus2.91.52.9\pm 1.5 |
| TabR-S (CF-0) | 0.8570.857\mathbf{0.857} | 0.4240.4240.424 | 3.0753.075\mathbf{3.075} | 0.8570.857\mathbf{0.857} | 0.1370.1370.137 | 0.8160.816\mathbf{0.816} | 0.7180.718\mathbf{0.718} | 0.7000.7000.700 | 1.7871.7871.787 | 0.9690.9690.969 | 1.3871.3871.387 | 2.3±1.4plus-or-minus2.31.42.3\pm 1.4 |
| TabR-S (CF-1) | 0.8560.856\mathbf{0.856} | 0.4140.4140.414 | 3.0653.065\mathbf{3.065} | 0.8560.8560.856 | 0.1370.1370.137 | 0.8160.816\mathbf{0.816} | 0.7180.718\mathbf{0.718} | 0.6920.6920.692 | 1.7701.7701.770 | 0.9730.973\mathbf{0.973} | 1.3251.3251.325 | 1.8±1.0plus-or-minus1.81.01.8\pm 1.0 |
| TabR-S (CF-2) | 0.8560.856\mathbf{0.856} | 0.4110.4110.411 | 3.0743.074\mathbf{3.074} | 0.8560.8560.856 | 0.1370.1370.137 | 0.8160.816\mathbf{0.816} | 0.7180.7180.718 | 0.6910.691\mathbf{0.691} | 1.7671.7671.767 | 0.9730.973\mathbf{0.973} | – | 1.7±0.8plus-or-minus1.70.81.7\pm 0.8 |
| TabR-S (CF-4) | 0.8580.858\mathbf{0.858} | 0.4090.409\mathbf{0.409} | 3.0873.087\mathbf{3.087} | 0.8570.857\mathbf{0.857} | 0.1360.1360.136 | 0.8160.816\mathbf{0.816} | 0.7170.7170.717 | 0.6910.691\mathbf{0.691} | 1.7631.7631.763 | 0.9730.973\mathbf{0.973} | – | 1.3±0.5plus-or-minus1.30.51.3\pm 0.5 |
| TabR-S (CF-8) | 0.8580.858\mathbf{0.858} | 0.4070.407\mathbf{0.407} | 3.1183.1183.118 | 0.8570.857\mathbf{0.857} | 0.1350.1350.135 | 0.8170.817\mathbf{0.817} | 0.7190.719\mathbf{0.719} | 0.6910.691\mathbf{0.691} | 1.7611.7611.761 | 0.9730.973\mathbf{0.973} | – | 1.3±0.5plus-or-minus1.30.51.3\pm 0.5 |
| TabR-S | 0.8590.859\mathbf{0.859} | 0.4060.406\mathbf{0.406} | 3.0933.093\mathbf{3.093} | 0.8580.858\mathbf{0.858} | 0.1330.133\mathbf{0.133} | 0.8160.816\mathbf{0.816} | 0.7190.719\mathbf{0.719} | 0.6910.691\mathbf{0.691} | 1.7551.755\mathbf{1.755} | 0.9730.973\mathbf{0.973} | 1.3151.315\mathbf{1.315} | 1.0±0.0plus-or-minus1.00.01.0\pm 0.0 |

Table 13: Fraction of time spent on training in [Table 12](#A1.T12 "Table 12 ‣ A.6 Additional results for the “context freeze” technique ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), relative to the training time without the context freeze (the last row; the format is hours:minutes:seconds).

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CH | CA | HO | AD | DI | OT | HI | BL | WE | CO | WE (full) |
| TabR-S (CF-0) | 0.960.960.96 | 0.780.780.78 | 0.790.790.79 | 0.830.830.83 | 0.750.750.75 | 0.870.870.87 | 1.031.031.03 | 0.640.640.64 | 0.530.530.53 | 0.520.520.52 | 0.130.130.13 |
| TabR-S (CF-1) | 0.880.880.88 | 0.720.720.72 | 0.890.890.89 | 0.890.890.89 | 0.470.470.47 | 0.800.800.80 | 0.800.800.80 | 0.610.610.61 | 0.570.570.57 | 0.490.490.49 | 0.130.130.13 |
| TabR-S (CF-2) | 0.940.940.94 | 0.650.650.65 | 0.780.780.78 | 0.830.830.83 | 0.470.470.47 | 0.860.860.86 | 0.820.820.82 | 0.630.630.63 | 0.570.570.57 | 0.600.600.60 | – |
| TabR-S (CF-4) | 1.011.011.01 | 0.710.710.71 | 0.730.730.73 | 0.730.730.73 | 0.510.510.51 | 0.970.970.97 | 0.730.730.73 | 0.620.620.62 | 0.560.560.56 | 0.590.590.59 | – |
| TabR-S (CF-8) | 1.031.031.03 | 0.760.760.76 | 0.710.710.71 | 0.820.820.82 | 0.610.610.61 | 0.900.900.90 | 0.780.780.78 | 0.670.670.67 | 0.590.590.59 | 0.590.590.59 | – |
| TabR-S | 0:00:08 | 0:00:36 | 0:00:42 | 0:00:46 | 0:01:25 | 0:00:58 | 0:01:24 | 0:05:03 | 0:16:19 | 0:39:13 | 18:08:39 |

### A.7 Additional technical notes on TabR

We highlight the following technical aspects of TabR:

1. 1.

   Because of the changes introduced in the Step-3 in [subsection 3.2](#S3.SS2 "3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), the value representations 𝒱​(x~,x~i,yi)𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i}) of the candidates cannot be precomputed for a trained model, since they depend on the target object.
   This implies roughly twice less memory usage when deploying the model to production (since only the key representations and labels have to be deployed for training objects), but 𝒱​(x~,x~i,yi)𝒱~𝑥subscript~𝑥𝑖subscript𝑦𝑖\mathcal{V}(\tilde{x},\tilde{x}\_{i},y\_{i}) has to be computed in runtime.
2. 2.

   Despite the attention-like nature of the retrieval module R𝑅R, contrary to prior work, TabR does not suffer from the quadratic complexity w.r.t. the number of candidates, because it computes attention only for the target object, but not for the context objects.

## Appendix B Limitations & Practical considerations

The following limitations and practical considerations are applicable to retrieval-augmented models in general.
TabR itself does not add anything new to this list.

First, for a given application, one should carefully evaluate from various perspectives (business logic, legal considerations, ethical aspects, etc.) whether using real training objects for making predictions is reasonable.

Second, depending on an application, for a given target object, one may want to retrieve only from a subset of the available data, where the subset is dynamically formed for the target object based on application-specific filters.
In terms of [subsection 3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), it means Ic​a​n​d=Ic​a​n​d​(x)⊂It​r​a​i​nsubscript𝐼𝑐𝑎𝑛𝑑subscript𝐼𝑐𝑎𝑛𝑑𝑥subscript𝐼𝑡𝑟𝑎𝑖𝑛I\_{cand}=I\_{cand}(x)\subset I\_{train}.

Third, ideally, retrieval during training should simulate retrieval during deployment, otherwise, a retrieval-based model can lead to (highly) suboptimal performance.
Examples:

* •

  For time series, during training, TabR must be allowed to retrieve only from the past.
  Moreover, perhaps, this “past” should also be limited to prevent the retrieval from too old data and too recent data.
  The decision should be made based on the domain expertise and business logic.
* •

  Let’s consider a task where, among all training objects, there are some “related objects”.
  For example, when solving a ranking problem as a point-wise regression, such “related objects” can be obtained as query-document pairs corresponding to the same query, but different documents.
  In some cases, during training, for a given target object, retrieving from “related objects” can be unfair, because the same will not be possible in production for new objects that do not have “related objects” in the available data.
  Again, this design decision should be made based on the domain expertise and business logic.

Lastly, while TabR is significantly more efficient than prior retrieval-based tabular DL models, the retrieval module R𝑅R still causes overhead compared to purely parametric models, so TabR may not scale to truly large datasets as-is.
We showcase a simple trick to scale TabR to larger datasets in [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
We discuss the efficiency aspect in more detail in [subsection A.4](#A1.SS4 "A.4 Comparing training times ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

## Appendix C Benchmarks

### C.1 The default benchmark

In [Table 14](#A3.T14 "Table 14 ‣ C.1 The default benchmark ‣ Appendix C Benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we provide more information on the datasets from [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The datasets include:

* •

  Churn Modeling111https://www.kaggle.com/shrutimechlearn/churn-modelling
* •

  California Housing (real estate data, [Kelley Pace and Barry, [1997](#bib.bib23)])
* •

  House 16H222https://www.openml.org/d/574
* •

  Adult (income estimation, [Kohavi, [1996](#bib.bib27)])
* •

  Diamond333https://www.openml.org/d/42225
* •

  Otto Group Product Classification444https://www.kaggle.com/c/otto-group-product-classification-challenge/data
* •

  Higgs (simulated physical particles, [Baldi et al., [2014](#bib.bib3)]; we use the version with 98K samples available in the OpenML repository [Vanschoren et al., [2014](#bib.bib43)])
* •

  Black Friday555https://www.openml.org/d/41540
* •

  Weather (temperature, [Malinin et al., [2021](#bib.bib32)]). We take 10% of the dataset for our experiments due to its large size.
* •

  Weather (full) (temperature, [Malinin et al., [2021](#bib.bib32)]). Original splits from the paper.
* •

  Covertype (forest characteristics, [Blackard and Dean., [2000](#bib.bib4)])
* •

  Microsoft (search queries, [Qin and Liu, [2013](#bib.bib39)]). We follow the pointwise approach to learning to rank and treat this ranking problem as a regression problem.

Table 14: Details on datasets from the main benchmark. “# Num”, “# Bin”, and “# Cat” denote the number of numerical, binary, and categorical features, respectively. The “Batch size” is the default batch size used to train DL-based models.

| Abbr | Name | # Train | # Validation | # Test | # Num | # Bin | # Cat | Task type | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH | Churn Modelling | 640064006400 | 160016001600 | 200020002000 | 101010 | 333 | 111 | Binclass | 128 |
| CA | California Housing | 132091320913209 | 330333033303 | 412841284128 | 888 | 00 | 00 | Regression | 256 |
| HO | House 16H | 145811458114581 | 364636463646 | 455745574557 | 161616 | 00 | 00 | Regression | 256 |
| AD | Adult | 260482604826048 | 651365136513 | 162811628116281 | 666 | 111 | 888 | Binclass | 256 |
| DI | Diamond | 345213452134521 | 863186318631 | 107881078810788 | 666 | 00 | 333 | Regression | 512 |
| OT | Otto Group Products | 396013960139601 | 990199019901 | 123761237612376 | 939393 | 00 | 00 | Multiclass | 512 |
| HI | Higgs Small | 627516275162751 | 156881568815688 | 196101961019610 | 282828 | 00 | 00 | Binclass | 512 |
| BL | Black Friday | 106764106764106764 | 266922669226692 | 333653336533365 | 444 | 111 | 444 | Regression | 512 |
| WE | Shifts Weather (subset) | 296554296554296554 | 473734737347373 | 531725317253172 | 118118118 | 111 | 00 | Regression | 1024 |
| CO | Covertype | 371847371847371847 | 929629296292962 | 116203116203116203 | 545454 | 444444 | 00 | Multiclass | 1024 |
| WE (full) | Shifts Weather (full) | 296554229655422965542 | 473734737347373 | 531720531720531720 | 118118118 | 111 | 00 | Regression | 1024 |

### C.2 The benchmark from Grinsztajn et al. [[2022](#bib.bib13)]

In this section, we describe how exactly we used the benchmark proposed in Grinsztajn et al. [[2022](#bib.bib13)].

* •

  We use the same train-val-test splits.
* •

  When there are several splits for one dataset (i.e., when the n-fold-cross-validation was performed in Grinsztajn et al. [[2022](#bib.bib13)]), we first treat each of them as separate datasets while tuning and evaluating algorithms as described in [Appendix D](#A4 "Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), but then, we average the metrics over the splits to obtain the final numbers for the dataset. For example, if there are five splits for a given dataset, then we tune and evaluate a given algorithm five times, each of the five tuned configurations is evaluated under 15 random seeds on the corresponding splits, and the reported metric value is the average over 5∗15=75515755\*15=75 runs.
* •

  When there are multiple versions of one dataset (e.g., the original regression task and the same dataset but converted to the binary classification task or the same dataset, but with the categorical features removed, etc.), we keep only one original dataset.
* •

  We removed the “Eye movements” dataset because there is a leak in that dataset.
* •

  We use the tuning and evaluation protocols as described in [Appendix D](#A4 "Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), which was also used in prior works on tabular DL [Gorishniy et al., [2021](#bib.bib11), [2022](#bib.bib12)]. Crucially, we tune hyperparameters of the GBDT models more extensively than most (if not all) prior work in terms of both budget (20 warmup iterations of random sampling followed by 180 iterations of the tree-structured Parzen estimator algorithm) and hyperparameter spaces (see the corresponding sections in [Appendix D](#A4 "Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).

## Appendix D Implementation details

### D.1 Hardware

We report the used hardware in the results published along with the source code.
In a nutshell, the vast majority of experiments on GPU were performed on one NVidia A100 GPU, the remaining small part of GPU experiments was performed on one Nvidia 2080 Ti GPU, and there was also a small portion of runs performed on CPU (e.g. all the experiments on LightGBM).

### D.2 Implementation details of [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")

In [subsection 5.1](#S5.SS1 "5.1 Freezing contexts for faster training of TabR ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we used TabR-S with the default hyperparameters (see [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).
To compute ΔΔ\Delta-context, we collect context distributions for training objects between training epochs.
That is, after the i𝑖i-th training epoch, we pause the training, collect the context distributions for all training objects, and then start the next (i+1)𝑖1(i+1)-th training epoch.

ΔΔ\Delta-context.
Intuitively, this heuristic metric describes in a single number how much, for a given input object, the context attention mass was updated compared to the previous epoch.
Namely, it is a sum of two terms:

1. 1.

   the novel attention mass, i.e. the attention mass coming from the context objects presented on the current epoch, but not presented on the previous epoch
2. 2.

   the increased attention mass, i.e. we take the intersection of the current and the previous context objects and compute the increase of their total attention mass. We set it to 0.0 if actually decreased.

Now, we formally define this metric.
For a given input object, let a∈ℝ|It​r​a​i​n|𝑎superscriptℝsubscript𝐼𝑡𝑟𝑎𝑖𝑛a\in\mathbb{R}^{|I\_{train}|} and b∈ℝ|It​r​a​i​n|𝑏superscriptℝsubscript𝐼𝑡𝑟𝑎𝑖𝑛b\in\mathbb{R}^{|I\_{train}|} denote the two distributions over the candidates from the previous and the current epochs, respectively.
Let denote the sets of non-zero entries as A={i:ai>0}𝐴conditional-set𝑖subscript𝑎𝑖0A=\{i:a\_{i}>0\} and B={i:ai>0}𝐵conditional-set𝑖subscript𝑎𝑖0B=\{i:a\_{i}>0\}.
Note that |A|=|B|=m=96𝐴𝐵𝑚96|A|=|B|=m=96.
In other words, A𝐴A and B𝐵B are the contexts from the two epochs.
Then:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Δ​-contextΔ-context\displaystyle\Delta\text{-context} | =novel+increasedabsentnovelincreased\displaystyle=\texttt{novel}+\texttt{increased} |  | (8) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | novel | =∑i∈B∖Abiabsentsubscript𝑖𝐵𝐴subscript𝑏𝑖\displaystyle=\sum\_{i\in B\setminus A}b\_{i} |  | (9) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | increased | =max⁡(∑i∈B∩Abi−∑i∈B∩Aai,0.0)absentsubscript𝑖𝐵𝐴subscript𝑏𝑖subscript𝑖𝐵𝐴subscript𝑎𝑖0.0\displaystyle=\max\left(\sum\_{i\in B\cap A}b\_{i}-\sum\_{i\in B\cap A}a\_{i},0.0\right) |  | (10) |

### D.3 Implementation details of [subsection 5.2](#S5.SS2 "5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")

In [subsection 5.2](#S5.SS2 "5.2 Updating TabR with new training data without retraining ‣ 5 Analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we used TabR-S with the default hyperparameters (see [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).

### D.4 Implementation details of [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")

In [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we performed the analysis over exactly the same model checkpoints that we used to assemble the rows “Step-1” and “Step-2” in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

To reiterate, this is how the entropy in [Table 7](#A1.T7 "Table 7 ‣ A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") is computed:

1. 1.

   First, we obtain individual distributions over candidates for all test objects. One such distribution contains exactly (m+1𝑚1m+1) non-zero entries.
2. 2.

   Then, we average all individual distributions and obtain the average distribution.
3. 3.

   [Table 7](#A1.T7 "Table 7 ‣ A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") reports the entropy of the average distribution.

Note that, when obtaining the distribution over candidates, the top-m𝑚m operation is taken into account.
Without that, if the distribution is always uniform regardless of the input object, then the average distribution will also be uniform and with the highest possible entropy, which would be misleading in the context of the story in [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Lastly, recall that in the Step-1 and Step-2 models, an input object is added to its own context.
Then, the edge case when all input objects pay 100% attention only to themselves would lead to the highest possible entropy, which would be misleading for the story in [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
In other words, for the story in [subsubsection A.1.2](#A1.SS1.SSS2 "A.1.2 Analyzing attention patterns over candidates ‣ A.1 Similarity module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we should treat the “paying attention to self” behavior similarly for all objects.
To achieve that, on the first step of the above recipe, we reassign the attention mass from “self” to a new virtual context object, which is the same for all input objects.

### D.5 Implementation details of [subsection A.2](#A1.SS2 "A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")

To build [Table 8](#A1.T8 "Table 8 ‣ A.2 Analyzing the value module of TabR ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), we used TabR-S with the default hyperparameters (see [subsection D.8](#A4.SS8 "D.8 TabR ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")).

### D.6 Experiment setup

For the most part, we simply follow Gorishniy et al. [[2022](#bib.bib12)], but we provide all the details for completeness.
Note that some of the prior work may differ from the common protocol that we describe below, but we provide the algorithm-specific implementation details further in this section.

Data preprocessing.
For each dataset, for all DL-based solutions, the same preprocessing was used for fair comparison.
For numerical features, by default, we used the quantile normalization from the Scikit-learn package [Pedregosa et al., [2011](#bib.bib34)], with rare exceptions when it turned out to be detrimental (for such datasets, we used the standard normalization or no normalization).
For categorical features, we used one-hot encoding.
Binary features (i.e. the ones that take only two distinct values) are mapped to {0,1}01\{0,1\} without any further preprocessing.

Training neural networks.
For DL-based algorithms, we minimize cross-entropy for classification problems and mean squared error for regression problems.
We use the AdamW optimizer [Loshchilov and Hutter, [2019](#bib.bib31)].
We do not apply learning rate schedules.
We do not use data augmentations.
For each dataset, we used a predefined dataset-specific batch size.
We continue training until there are patience+1patience1\texttt{patience}+1 consecutive epochs without improvements on the validation set; we set patience=16patience16\texttt{patience}=16 for the DL models.

How we compare algorithms.
For a given dataset, first, we define the “preliminary best” algorithm as the algorithm with the best mean score.
Then, we define a set of the best algorithms (i.e. their results are in bold in tables) as follows: a given algorithm is included in the best algorithms if its mean score differs from the mean score of the preliminary best algorithm by no more than the standard deviation of the preliminary best algorithm.

### D.7 Embeddings for numerical features

Figure 8: (Copied from Gorishniy et al. [[2022](#bib.bib12)]) The vanilla MLP. The model takes two numerical features as input.

Figure 9: (Copied from Gorishniy et al. [[2022](#bib.bib12)]) The same MLP as in [Figure 8](#A4.F8 "Figure 8 ‣ D.7 Embeddings for numerical features ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"), but now with embeddings for numerical features.

In this work, we actively used embeddings for numerical features from [Gorishniy et al., [2022](#bib.bib12)] (see [Figure 8](#A4.F8 "Figure 8 ‣ D.7 Embeddings for numerical features ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and [Figure 9](#A4.F9 "Figure 9 ‣ D.7 Embeddings for numerical features ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023")), the technique which was reported to universally improve DL models.
In a nutshell, for a given scalar numerical feature, an embedding module is a trainable module that maps this scalar feature to a vector.
Then, the embeddings of all numerical features are concatenated into one flat vector which is passed to further layers.
Following the original paper, when we use embeddings for numerical features, the same embedding architecture is used for all numerical features.

In this work, we used the LR (the combination of a linear layer and ReLU) and PLR (the combination of periodic embeddings, a linear layer, and ReLU) embeddings from the original paper.
Also, we introduce the PLR(lite) embedding, a simplified version of the PLR embedding where the linear layer is shared across all features.
We observed it to be significantly more lightweight without critical performance loss.

Hyperparameters tuning.
For the LR embeddings, we tune the embedding dimension in Uniform​[16,96]Uniform1696\mathrm{Uniform}[16,96].
For the PLR and PLR(lite) embeddings, we tune the number of frequencies in Uniform​[16,96]Uniform1696\mathrm{Uniform}[16,96] (in Uniform​[8,96]Uniform896\mathrm{Uniform}[8,96] for TabR on the datasets from Grinsztajn et al. [[2022](#bib.bib13)]), the frequency initialization scale in LogUniform​[0.01,100.0]LogUniform0.01100.0\mathrm{LogUniform}[0.01,100.0] and the embedding dimension in Uniform​[16,64]Uniform1664\mathrm{Uniform}[16,64] (in Uniform​[4,64]Uniform464\mathrm{Uniform}[4,64] for TabR on the datasets from Grinsztajn et al. [[2022](#bib.bib13)]).

### D.8 TabR

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

Embeddings for numerical features. (see [subsection D.7](#A4.SS7 "D.7 Embeddings for numerical features ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))
For the non-simple configurations of TabR, on datasets CH, CA, HO, AD, DI, OT, HI, BL, and on all the datasets from Grinsztajn et al. [[2022](#bib.bib13)], we used the PLR(lite) embeddings as defined in [subsection D.7](#A4.SS7 "D.7 Embeddings for numerical features ‣ Appendix D Implementation details ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
For other datasets, we used the LR embeddings.

Other details.
We observed that initializing the WYsubscript𝑊𝑌W\_{Y} module properly may be important for good performance.
Please, see the source code.

Default TabR-S.
The default hyperparameters for TabR-S were obtained at some point in the project by literally averaging the tuned hyperparameters over multiple datasets.
The specific set of datasets for averaging included all datasets from [Table 14](#A3.T14 "Table 14 ‣ C.1 The default benchmark ‣ Appendix C Benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") plus two datasets that used to be a part of the default benchmark, but were excluded later.
So, in total, 13 datasets contributed to the default hyperparameters.

Formally, this is not 100% fair to evaluate the obtained default TabR-S on the datasets which contributed to this default hyperparameters as in [Table 4](#S4.T4 "Table 4 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
However, we tested the fair leave-one-out approach as well (i.e. for a given dataset, averaging tuned hyperparameters over all datasets except for this one dataset) and did not observe any meaningful changes, so we decided to keep things simple and to have one common set of default hyperparameters for all datasets.
Plus, the obtained default TabR-S demonstrates decent performance in [Table 5](#S4.T5 "Table 5 ‣ 4.2 Comparing TabR with gradient-boosted decision trees ‣ 4 Experiments on public benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") as well, which illustrates that the obtained default configuration is not strongly “overfitted” to the datasets from [Table 14](#A3.T14 "Table 14 ‣ C.1 The default benchmark ‣ Appendix C Benchmarks ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").
The specific default hyperparameter values of TabR-S are as follows:

* •

  d=265𝑑265d=265
* •

  Attention dropout rate =0.38920071545944357absent0.38920071545944357=0.38920071545944357
* •

  Dropout rate in FFN =0.38852797479169876absent0.38852797479169876=0.38852797479169876
* •

  Learning rate =0.0003121273641315169absent0.0003121273641315169=0.0003121273641315169
* •

  Weight decay =0.0000012260352006404615absent0.0000012260352006404615=0.0000012260352006404615

Hyperparameters.
The output size of the first linear layer of FFN and of T𝑇T is 2​d2𝑑2d.
We performed tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.
The same protocol and hyperparameter spaces were used when tuning models in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023") and [Table 9](#A1.T9 "Table 9 ‣ A.3 Ablation study ‣ Appendix A Additional analysis ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023").

Table 15: 
The hyperparameter tuning space for TabR.
Here (A) = {CH, CA, HO, AD, DI, OT, HI, BL}, (B) = {WE, CO, MI}.
For the datasets from Grinsztajn et al. [[2022](#bib.bib13)], the tuning space is identical to (A) with the only difference that d𝑑d is tuned in UniformInt​[16,384]UniformInt16384\mathrm{UniformInt}[16,384].

| Parameter | (Datasets) Distribution | Comment |
| --- | --- | --- |
| Width d𝑑d | (A,B) UniformInt​[96,384]UniformInt96384\mathrm{UniformInt}[96,384] |  |
| Attention dropout rate | (A,B) Uniform​[0.0,0.6]Uniform0.00.6\mathrm{Uniform}[0.0,0.6] |  |
| Dropout rate in FFN | (A,B) Uniform​[0.0,0.6]Uniform0.00.6\mathrm{Uniform}[0.0,0.6] |  |
| Learning rate | (A,B) LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |  |
| Weight decay | (A,B) {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |  |
| NEsubscript𝑁𝐸N\_{E} | (A,B) UniformInt​[0,1]UniformInt01\mathrm{UniformInt}[0,1] | Const​[0]Constdelimited-[]0\mathrm{Const}[0] for TabR-S |
| NPsubscript𝑁𝑃N\_{P} | (A,B) UniformInt​[1,2]UniformInt12\mathrm{UniformInt}[1,2] | Const​[1]Constdelimited-[]1\mathrm{Const}[1] for TabR-S |
| # Tuning iterations | (A) 100 (B) 50 |  |

### D.9 MLP

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We used the implementation from Gorishniy et al. [[2022](#bib.bib12)].

Hyperparameters.
We use the same hidden dimension throughout the whole network.
We performed tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.

Table 16: The hyperparameter tuning space for MLP

| Parameter | Distribution |
| --- | --- |
| # layers | UniformInt​[1,6]UniformInt16\mathrm{UniformInt}[1,6] |
| Width (hidden size) | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| # Tuning iterations | 100 |

### D.10 FT-Transformer

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We used the implementation from the "rtdl" Python package (version 0.0.13).

Hyperparameters.
We use the rtdl.FTTransformer.make\_baseline method to create FT-Transformer, so most of hyperparameters is inherited from this method’s signature, and the rest is tuned as shown in the corresponding table.

Table 17: The hyperparameter tuning space for FT-Transformer

| Parameter | Distribution |
| --- | --- |
| # blocks | UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | UniformInt​[16,384]UniformInt16384\mathrm{UniformInt}[16,384] |
| Attention dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| FFN hidden dimension expansion rate | Uniform​[2/3,8/3]Uniform2383\mathrm{Uniform}[\nicefrac{{2}}{{3}},\nicefrac{{8}}{{3}}] |
| FFN dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Residual dropout rate | {0.0,Uniform​[0.0,0.2]}0.0Uniform0.00.2\{0.0,\mathrm{Uniform}[0.0,0.2]\} |
| Learning rate | LogUniform​[1​e​-​5,1​e​-​3]LogUniform1𝑒-51𝑒-3\mathrm{LogUniform}[1e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​4]}0LogUniform1𝑒-61𝑒-4\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}4]\} |
| # Tuning iterations | 100 |

### D.11 kNN

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

The features are preprocessed in the same way as for DL models.
The only hyperparameter is the number of neighbors which we tune in UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128].

### D.12 DNNR

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We’ve used the official implementation 666<https://github.com/younader/dnnr>, but to evaluate DNNR on larger datasets with greater hyperparameters variability, we have rewritten parts of the source code to make it more efficient: enabling GPU usage, batched data processing, multiprocessing, where possible.
Crucially, we leave the underlying method unchanged.
We provide our efficiency-improved DNNR in the source code.
There is no support for classification problems, so we evaluate DNNR only on regression problems.

Hyperparameters. We performed a grid-search over the main DNNR hyperparameters on all datasets, falling back to defaults (suggested by the authors) due to scaling issues on WE and MI.

Table 18: The hyperparameter grid used for DNNR. Here (A) = {CA, HO}; (B) = {DI, BL, WE, MI}. Notation: Nfsubscript𝑁𝑓N\_{f} – number of features for the dataset.

|  |  |  |
| --- | --- | --- |
| Parameter | (Datasets) Parameter grid | Comment |
| # neighbors k𝑘k | (A,B) [1,2,3,…,128]  123…128[1,2,3,\ldots,128] |  |
| Learned scaling | (A,B) [No scaling, Trained scaling] |  |
| # neighbors used in scaling | (A,B) [8⋅Nf,2,3,4,8,16,32,64,128  ⋅8subscript𝑁𝑓23481632641288\cdot N\_{f},2,3,4,8,16,32,64,128] | 8⋅Nf⋅8subscript𝑁𝑓8\cdot N\_{f} on WE, MI |
| # epochs used in scaling | 101010 |  |
| Cat. feature encoding | [one-hot, leave-one-out] |  |
| # neighbors for derivative k′superscript𝑘′k^{\prime} | (A) LinSpace​[2⋅Nf,18⋅Nf,20]LinSpace  ⋅2subscript𝑁𝑓⋅18subscript𝑁𝑓20\mathrm{LinSpace}[2\cdot N\_{f},18\cdot N\_{f},20] |  |
|  | (B) LinSpace​[2⋅Nf,12⋅Nf,14]LinSpace  ⋅2subscript𝑁𝑓⋅12subscript𝑁𝑓14\mathrm{LinSpace}[2\cdot N\_{f},12\cdot N\_{f},14] |  |

### D.13 DKL

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We used DKL implementation from GPyTorch [Gardner et al., [2018](#bib.bib10)]. We do not evaluate DKL on WE and MI datasets due to scaling issues (tuning alone takes 1 day and 17 hours, compared to 3 hours for TabR on the medium DI dataset, for example). There is no support for classification problems, thus we evaluate DKL only on regression problems.

Hyperparameters. As with MLP we use the same hidden dimension throughout the whole network. And perform tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [1] library.

Table 19: The hyperparameter tuning space for DKL

| Parameter | Distribution |
| --- | --- |
| Kernel | {rbf,sm}rbfsm\{\mathrm{rbf},\mathrm{sm}\} |
| # layers | UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| Width (hidden size) | UniformInt​[64,768]UniformInt64768\mathrm{UniformInt}[64,768] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[1​e​-​5,1​e​-​2]LogUniform1𝑒-51𝑒-2\mathrm{LogUniform}[1e\text{-}5,1e\text{-}2] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| # Tuning iterations | 100 |

### D.14 ANP

While the original paper introducing ANP did not focus on the tabular data, conceptually, it is very relevant to prior work on retrieval-based tabular DL, so we consider it as one of the baselines.

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We used the Pytorch implementation from an unofficial repository777<https://github.com/soobinseo/Attentive-Neural-Process> and modified it with respect to the official implementation from Kim et al. [[2019](#bib.bib25)]. Specifically, we reimplemented Decoder class exactly as it was done in Kim et al. [[2019](#bib.bib25)] and changed a binary cross-entropy loss with a Gaussian negative log-likelihood loss in LatentModel class since it matches with the official implementation.

We do not evaluate ANP on the MI dataset due to scaling issues. Tuning alone on the smaller WE dataset took more than four days for 20(!) iterations (instead of 50-100 used for other algorithms). Also, there is no support for classification problems, thus we evaluate ANP only on regression problems.

We used 100 tuning iterations on CA and HO, 50 on DI, and 20 on BL and WE.

Table 20: The hyperparameter tuning space for ANP

| Parameter | Distribution |
| --- | --- |
| # decoder layers | UniformInt​[1,3]UniformInt13\mathrm{UniformInt}[1,3] |
| # cross-attention layers | UniformInt​[1,2]UniformInt12\mathrm{UniformInt}[1,2] |
| # self-attention layers | UniformInt​[1,2]UniformInt12\mathrm{UniformInt}[1,2] |
| Width (hidden size) | UniformInt​[64,384]UniformInt64384\mathrm{UniformInt}[64,384] |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​4]}0LogUniform1𝑒-61𝑒-4\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}4]\} |

### D.15 NPT

We use the official NPT [Kossen et al., [2021](#bib.bib28)] implementation 888<https://github.com/OATML/non-parametric-transformers>.
We leave the model and training code unchanged and only adjust the datasets and their preprocessing according to our protocols.

We evaluate the NPT-Base configuration of the model and follow both NPT-Base architecture and optimization hyperparameters.
We train NPT for 200020002000 epochs on CH, CA, AD, HO, 100001000010000 epochs on OT, WE, MI, 150001500015000 epochs on DI, BL, HI and 300003000030000 epochs on CO.
For all datasets that don’t fit into the A100 80GB GPU, we use batch size 409640964096 (as suggested in the NPT paper).
We also decrease the hidden dim to 323232 on WE and MI to avoid the OOM error.

Note that NPT is conceptually equivalent to other transformer-based non-parametric tabular DL solutions: [Somepalli et al., [2021](#bib.bib42), Schäfl et al., [2022](#bib.bib41)].
All three methods use dot-product-based self-attention modules alternating between self-attention between object features and self-attention between objects (for the whole training dataset or its random subset).

### D.16 SAINT

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

We use the official implementation of SAINT 999<https://github.com/somepago/saint> with one important fix.
Recall that, in SAINT, a target object interacts with its context objects with intersample attention.
In the official implementation of SAINT, context objects are taken from the same dataset part, as a target object: for training objects, context objects are taken from the training set, for validation objects – from the validation set, for test objects – from the test set.
This is different from the approach described in this paper, where context objects are always taken from the training set.
Taking context objects from different dataset parts, as in the official implementation of SAINT, may be unwanted because of the following reasons:

1. 1.

   model can have suboptimal validation and test performance because it is trained to operate when context objects are taken from the training set, but evaluated when context objects are taken from other dataset parts.
2. 2.

   for a given validation/test object, the prediction depends on other validation/test objects. This is not in line with other retrieval-based models, which may result in inconsistent comparisons. Also, in many real-world scenarios, during deployment/test time, input objects should be processed independently, which is not the case for the official implementation of SAINT.

For the above reasons, we slightly modify SAINT such that each individual sample attends only to itself and to context samples from the training set, both during training and evaluation.
See the source code for details.

On small datasets (CH, CA, HO, AD, DI, OT, HI, BL) we fix the number of attention heads at 888 and performed hyperparameter tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.

Table 21: The hyperparameter tuning space for SAINT

| Parameter | Distribution |
| --- | --- |
| Depth | UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| Width | UniformInt​[4,32,4]UniformInt  4324\mathrm{UniformInt}[4,32,4] |
| Feed forward multiplier | Uniform​[2/3,8/3]Uniform2383\mathrm{Uniform}[\nicefrac{{2}}{{3}},\nicefrac{{8}}{{3}}] |
| Attention dropout | Uniform​[0,0.5]Uniform00.5\mathrm{Uniform}[0,0.5] |
| Feed forward dropout | Uniform​[0,0.5]Uniform00.5\mathrm{Uniform}[0,0.5] |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​4]}0LogUniform1𝑒-61𝑒-4\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}4]\} |

On larger datasets (WE, CO, MI) we use slightly modified (for optimizing memory consumption) default configuration from the paper with following fixed hyperparameters:

* •

  depth = 4
* •

  n\_heads = 8
* •

  dim = 32
* •

  ffn\_mult = 4
* •

  attn\_head\_dim = 48
* •

  attn\_dropout = 0.1
* •

  ff\_dropout = 0.8
* •

  learning\_rate = 0.0001
* •

  weight\_decay = 0.01

### D.17 XGBoost

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

In this work, we made our best to tune GBDT models as good as possible to make sure that the comparison is fair, and the conclusions are reliable.
Compared to prior work [Gorishniy et al., [2021](#bib.bib11), [2022](#bib.bib12)], where GBDT is already extensively tuned, we doubled the number of tuning iterations, doubled the number of trees, increased the maximum depth and increased the number of early stopping rounds by 4x.

The following hyperparameters are fixed and not tuned:

* •

  booster = “gbtree”
* •

  n\_estimators = 4000
* •

  tree\_method = “gpu\_hist”
* •

  early\_stopping\_rounds = 200

We performed tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.

Table 22: The hyperparameter tuning space for XGBoost

| Parameter | Distribution |
| --- | --- |
| colsample\_bytree | Uniform​[0.5,1.0]Uniform0.51.0\mathrm{Uniform}[0.5,1.0] |
| gamma | {0.0,LogUniform​[0.001,100.0]}0.0LogUniform0.001100.0\{0.0,\mathrm{LogUniform}[0.001,100.0]\} |
| lambda | {0.0,LogUniform​[0.1,10.0]}0.0LogUniform0.110.0\{0.0,\mathrm{LogUniform}[0.1,10.0]\} |
| learning\_rate | LogUniform​[0.001,1.0]LogUniform0.0011.0\mathrm{LogUniform}[0.001,1.0] |
| max\_depth | UniformInt​[3,14]UniformInt314\mathrm{UniformInt}[3,14] |
| min\_child\_weight | LogUniform​[0.0001,100.0]LogUniform0.0001100.0\mathrm{LogUniform}[0.0001,100.0] |
| subsample | Uniform[0.5,1.0\mathrm{Uniform}[0.5,1.0 |
| # Tuning iterations | 200 |

### D.18 LightGBM

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

In this work, we made our best to tune GBDT models as good as possible to make sure that the comparison is fair, and the conclusions are reliable.
Compared to prior work [Gorishniy et al., [2021](#bib.bib11), [2022](#bib.bib12)], where GBDT is already extensively tuned, we doubled the number of tuning iterations, doubled the number of trees, increased the maximum depth and increased the number of early stopping rounds by 4x.

The following hyperparameters are fixed and not tuned:

* •

  n\_estimators = 4000
* •

  early\_stopping\_rounds = 200

We performed tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.

Table 23: The hyperparameter tuning space for LightGBM

| Parameter | Distribution |
| --- | --- |
| feature\_fraction | Uniform​[0.5,1.0]Uniform0.51.0\mathrm{Uniform}[0.5,1.0] |
| lambda\_l2 | {0.0,LogUniform​[0.1,10.0]}0.0LogUniform0.110.0\{0.0,\mathrm{LogUniform}[0.1,10.0]\} |
| learning\_rate | LogUniform​[0.001,1.0]LogUniform0.0011.0\mathrm{LogUniform}[0.001,1.0] |
| num\_leaves | UniformInt​[4,768]UniformInt4768\mathrm{UniformInt}[4,768] |
| min\_sum\_hessian\_in\_leaf | LogUniform​[0.0001,100.0]LogUniform0.0001100.0\mathrm{LogUniform}[0.0001,100.0] |
| bagging\_fraction | Uniform​[0.5,1.0]Uniform0.51.0\mathrm{Uniform}[0.5,1.0] |
| # Tuning iterations | 200 |

### D.19 CatBoost

The implementation, tuning hyperparameters, evaluation hyperparameters, metrics, execution times, hardware and other details are available in the source code.
Here, we summarize some of the details for convenience.

In this work, we made our best to tune GBDT models as good as possible to make sure that the comparison is fair, and the conclusions are reliable.
Compared to prior work [Gorishniy et al., [2021](#bib.bib11), [2022](#bib.bib12)], where GBDT is already extensively tuned, we doubled the number of tuning iterations, doubled the number of trees, increased the maximum depth and increased the number of early stopping rounds by 4x.

The following hyperparameters are fixed and not tuned:

* •

  n\_estimators = 4000
* •

  early\_stopping\_rounds = 200
* •

  od\_pval = 0.001

We performed tuning using the tree-structured Parzen Estimator algorithm from the Akiba et al. [[2019](#bib.bib1)] library.

Table 24: The hyperparameter tuning space for CatBoost

| Parameter | Distribution |
| --- | --- |
| bagging\_temperature | Uniform​[0.0,1.0]Uniform0.01.0\mathrm{Uniform}[0.0,1.0] |
| depth | UniformInt​[3,14]UniformInt314\mathrm{UniformInt}[3,14] |
| l2\_leaf\_reg | Uniform​[0.1,10.0]Uniform0.110.0\mathrm{Uniform}[0.1,10.0] |
| leaf\_estimation\_iterations | Uniform​[1,10]Uniform110\mathrm{Uniform}[1,10] |
| learning\_rate | LogUniform​[0.001,1.0]LogUniform0.0011.0\mathrm{LogUniform}[0.001,1.0] |
| # Tuning iterations | 200 |

## Appendix E Extended results with standard deviations

In this section, we provide the extended results with standard deviations for the main results reported in the main text.
The results for the default benchmark are in the LABEL:A:tab:extended-results-ours.
The results for the benchmark from Grinsztajn et al. [[2022](#bib.bib13)] are in the LABEL:A:tab:extended-results-why.

Table 25: Extended results for the default benchmark. Results are grouped by datasets and span multiple pages below. Notation: ↓ corresponds to RMSE, ↑ corresponds to accuracy.

|  |  |
| --- | --- |
| CH ↑  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.837±0.000plus-or-minus0.8370.0000.837\pm 0.000 –  DNNR – –  DKL – –  ANP – –  NPT 0.858±0.003plus-or-minus0.8580.0030.858\pm 0.003 –  SAINT 0.860±0.003plus-or-minus0.8600.0030.860\pm 0.003 –  MLP 0.854±0.003plus-or-minus0.8540.0030.854\pm 0.003 –  MLP-PLR 0.860±0.002plus-or-minus0.8600.0020.860\pm 0.002 0.860±0.001plus-or-minus0.8600.0010.860\pm 0.001  TabR-S 0.860±0.002plus-or-minus0.8600.0020.860\pm 0.002 0.862±0.002plus-or-minus0.8620.0020.862\pm 0.002  TabR 0.862±0.002plus-or-minus0.8620.0020.862\pm 0.002 0.865±0.001plus-or-minus0.8650.0010.865\pm 0.001  CatBoost 0.858±0.002plus-or-minus0.8580.0020.858\pm 0.002 0.859±0.001plus-or-minus0.8590.0010.859\pm 0.001  XGBoost 0.861±0.002plus-or-minus0.8610.0020.861\pm 0.002 0.861±0.001plus-or-minus0.8610.0010.861\pm 0.001  LightGBM 0.860±0.001plus-or-minus0.8600.0010.860\pm 0.001 0.860±0.000plus-or-minus0.8600.0000.860\pm 0.000  Default hyperparameters  CatBoost 0.860±0.002plus-or-minus0.8600.0020.860\pm 0.002 0.861±0.001plus-or-minus0.8610.0010.861\pm 0.001  XGBoost 0.855±0.000plus-or-minus0.8550.0000.855\pm 0.000 0.856±0.000plus-or-minus0.8560.0000.856\pm 0.000  LightGBM 0.856±0.000plus-or-minus0.8560.0000.856\pm 0.000 0.856±0.000plus-or-minus0.8560.0000.856\pm 0.000  TabR-S 0.859±0.003plus-or-minus0.8590.0030.859\pm 0.003 0.864±0.001plus-or-minus0.8640.0010.864\pm 0.001  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.855±0.003plus-or-minus0.8550.0030.855\pm 0.003 0.857±0.002plus-or-minus0.8570.0020.857\pm 0.002  step-1 0.855±0.003plus-or-minus0.8550.0030.855\pm 0.003 0.858±0.002plus-or-minus0.8580.0020.858\pm 0.002  step-2 0.860±0.002plus-or-minus0.8600.0020.860\pm 0.002 0.862±0.003plus-or-minus0.8620.0030.862\pm 0.003  step-3 0.859±0.002plus-or-minus0.8590.0020.859\pm 0.002 0.862±0.002plus-or-minus0.8620.0020.862\pm 0.002 | CA ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.588±0.000plus-or-minus0.5880.0000.588\pm 0.000 –  DNNR 0.430±0.000plus-or-minus0.4300.0000.430\pm 0.000 –  DKL 0.521±0.055plus-or-minus0.5210.0550.521\pm 0.055 –  ANP 0.472±0.007plus-or-minus0.4720.0070.472\pm 0.007 –  NPT 0.474±0.003plus-or-minus0.4740.0030.474\pm 0.003 –  SAINT 0.468±0.005plus-or-minus0.4680.0050.468\pm 0.005 –  MLP 0.499±0.004plus-or-minus0.4990.0040.499\pm 0.004 –  MLP-PLR 0.476±0.004plus-or-minus0.4760.0040.476\pm 0.004 0.470±0.001plus-or-minus0.4700.0010.470\pm 0.001  TabR-S 0.403±0.002plus-or-minus0.4030.0020.403\pm 0.002 0.396±0.001plus-or-minus0.3960.0010.396\pm 0.001  TabR 0.400±0.003plus-or-minus0.4000.0030.400\pm 0.003 0.391±0.002plus-or-minus0.3910.0020.391\pm 0.002  CatBoost 0.429±0.001plus-or-minus0.4290.0010.429\pm 0.001 0.426±0.000plus-or-minus0.4260.0000.426\pm 0.000  XGBoost 0.433±0.002plus-or-minus0.4330.0020.433\pm 0.002 0.432±0.001plus-or-minus0.4320.0010.432\pm 0.001  LightGBM 0.435±0.002plus-or-minus0.4350.0020.435\pm 0.002 0.434±0.001plus-or-minus0.4340.0010.434\pm 0.001  Default hyperparameters  CatBoost 0.433±0.001plus-or-minus0.4330.0010.433\pm 0.001 0.432±0.001plus-or-minus0.4320.0010.432\pm 0.001  XGBoost 0.471±0.000plus-or-minus0.4710.0000.471\pm 0.000 0.471±0.000plus-or-minus0.4710.0000.471\pm 0.000  LightGBM 0.449±0.000plus-or-minus0.4490.0000.449\pm 0.000 0.449±0.000plus-or-minus0.4490.0000.449\pm 0.000  TabR-S 0.406±0.003plus-or-minus0.4060.0030.406\pm 0.003 0.398±0.001plus-or-minus0.3980.0010.398\pm 0.001  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.484±0.006plus-or-minus0.4840.0060.484\pm 0.006 0.470±0.005plus-or-minus0.4700.0050.470\pm 0.005  step-1 0.489±0.007plus-or-minus0.4890.0070.489\pm 0.007 0.474±0.005plus-or-minus0.4740.0050.474\pm 0.005  step-2 0.418±0.002plus-or-minus0.4180.0020.418\pm 0.002 0.411±0.000plus-or-minus0.4110.0000.411\pm 0.000  step-3 0.408±0.003plus-or-minus0.4080.0030.408\pm 0.003 0.399±0.002plus-or-minus0.3990.0020.399\pm 0.002 |
| HO ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 3.744±0.000plus-or-minus3.7440.0003.744\pm 0.000 –  DNNR 3.210±0.000plus-or-minus3.2100.0003.210\pm 0.000 –  DKL 3.423±0.393plus-or-minus3.4230.3933.423\pm 0.393 –  ANP 3.162±0.028plus-or-minus3.1620.0283.162\pm 0.028 –  NPT 3.175±0.032plus-or-minus3.1750.0323.175\pm 0.032 –  SAINT 3.242±0.059plus-or-minus3.2420.0593.242\pm 0.059 –  MLP 3.112±0.036plus-or-minus3.1120.0363.112\pm 0.036 –  MLP-PLR 3.056±0.021plus-or-minus3.0560.0213.056\pm 0.021 2.993±0.019plus-or-minus2.9930.0192.993\pm 0.019  TabR-S 3.067±0.040plus-or-minus3.0670.0403.067\pm 0.040 2.996±0.027plus-or-minus2.9960.0272.996\pm 0.027  TabR 3.105±0.041plus-or-minus3.1050.0413.105\pm 0.041 3.025±0.010plus-or-minus3.0250.0103.025\pm 0.010  CatBoost 3.117±0.013plus-or-minus3.1170.0133.117\pm 0.013 3.106±0.002plus-or-minus3.1060.0023.106\pm 0.002  XGBoost 3.177±0.010plus-or-minus3.1770.0103.177\pm 0.010 3.164±0.007plus-or-minus3.1640.0073.164\pm 0.007  LightGBM 3.177±0.009plus-or-minus3.1770.0093.177\pm 0.009 3.167±0.005plus-or-minus3.1670.0053.167\pm 0.005  Default hyperparameters  CatBoost 3.122±0.011plus-or-minus3.1220.0113.122\pm 0.011 3.108±0.002plus-or-minus3.1080.0023.108\pm 0.002  XGBoost 3.368±0.000plus-or-minus3.3680.0003.368\pm 0.000 3.368±0.000plus-or-minus3.3680.0003.368\pm 0.000  LightGBM 3.222±0.000plus-or-minus3.2220.0003.222\pm 0.000 3.222±0.000plus-or-minus3.2220.0003.222\pm 0.000  TabR-S 3.093±0.060plus-or-minus3.0930.0603.093\pm 0.060 2.971±0.017plus-or-minus2.9710.0172.971\pm 0.017  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 3.234±0.053plus-or-minus3.2340.0533.234\pm 0.053 3.144±0.034plus-or-minus3.1440.0343.144\pm 0.034  step-1 3.205±0.056plus-or-minus3.2050.0563.205\pm 0.056 3.104±0.043plus-or-minus3.1040.0433.104\pm 0.043  step-2 3.153±0.031plus-or-minus3.1530.0313.153\pm 0.031 3.117±0.012plus-or-minus3.1170.0123.117\pm 0.012  step-3 3.158±0.017plus-or-minus3.1580.0173.158\pm 0.017 3.117±0.006plus-or-minus3.1170.0063.117\pm 0.006 | AD ↑  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.834±0.000plus-or-minus0.8340.0000.834\pm 0.000 –  DNNR – –  DKL – –  ANP – –  NPT 0.853±0.010plus-or-minus0.8530.0100.853\pm 0.010 –  SAINT 0.860±0.002plus-or-minus0.8600.0020.860\pm 0.002 –  MLP 0.853±0.001plus-or-minus0.8530.0010.853\pm 0.001 –  MLP-PLR 0.870±0.002plus-or-minus0.8700.0020.870\pm 0.002 0.873±0.001plus-or-minus0.8730.0010.873\pm 0.001  TabR-S 0.865±0.002plus-or-minus0.8650.0020.865\pm 0.002 0.868±0.002plus-or-minus0.8680.0020.868\pm 0.002  TabR 0.870±0.001plus-or-minus0.8700.0010.870\pm 0.001 0.872±0.001plus-or-minus0.8720.0010.872\pm 0.001  CatBoost 0.871±0.001plus-or-minus0.8710.0010.871\pm 0.001 0.872±0.001plus-or-minus0.8720.0010.872\pm 0.001  XGBoost 0.872±0.001plus-or-minus0.8720.0010.872\pm 0.001 0.872±0.000plus-or-minus0.8720.0000.872\pm 0.000  LightGBM 0.871±0.001plus-or-minus0.8710.0010.871\pm 0.001 0.872±0.000plus-or-minus0.8720.0000.872\pm 0.000  Default hyperparameters  CatBoost 0.873±0.001plus-or-minus0.8730.0010.873\pm 0.001 0.874±0.001plus-or-minus0.8740.0010.874\pm 0.001  XGBoost 0.871±0.000plus-or-minus0.8710.0000.871\pm 0.000 0.871±0.000plus-or-minus0.8710.0000.871\pm 0.000  LightGBM 0.869±0.000plus-or-minus0.8690.0000.869\pm 0.000 0.869±0.000plus-or-minus0.8690.0000.869\pm 0.000  TabR-S 0.858±0.001plus-or-minus0.8580.0010.858\pm 0.001 0.859±0.000plus-or-minus0.8590.0000.859\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.857±0.002plus-or-minus0.8570.0020.857\pm 0.002 0.858±0.000plus-or-minus0.8580.0000.858\pm 0.000  step-1 0.857±0.002plus-or-minus0.8570.0020.857\pm 0.002 0.860±0.000plus-or-minus0.8600.0000.860\pm 0.000  step-2 0.858±0.002plus-or-minus0.8580.0020.858\pm 0.002 0.862±0.001plus-or-minus0.8620.0010.862\pm 0.001  step-3 0.863±0.002plus-or-minus0.8630.0020.863\pm 0.002 0.866±0.001plus-or-minus0.8660.0010.866\pm 0.001 |
| DI ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.256±0.000plus-or-minus0.2560.0000.256\pm 0.000 –  DNNR 0.145±0.000plus-or-minus0.1450.0000.145\pm 0.000 –  DKL 0.147±0.005plus-or-minus0.1470.0050.147\pm 0.005 –  ANP 0.140±0.001plus-or-minus0.1400.0010.140\pm 0.001 –  NPT 0.138±0.001plus-or-minus0.1380.0010.138\pm 0.001 –  SAINT 0.137±0.002plus-or-minus0.1370.0020.137\pm 0.002 –  MLP 0.140±0.001plus-or-minus0.1400.0010.140\pm 0.001 –  MLP-PLR 0.134±0.001plus-or-minus0.1340.0010.134\pm 0.001 0.133±0.000plus-or-minus0.1330.0000.133\pm 0.000  TabR-S 0.133±0.001plus-or-minus0.1330.0010.133\pm 0.001 0.131±0.000plus-or-minus0.1310.0000.131\pm 0.000  TabR 0.133±0.001plus-or-minus0.1330.0010.133\pm 0.001 0.131±0.000plus-or-minus0.1310.0000.131\pm 0.000  CatBoost 0.134±0.001plus-or-minus0.1340.0010.134\pm 0.001 0.133±0.000plus-or-minus0.1330.0000.133\pm 0.000  XGBoost 0.137±0.000plus-or-minus0.1370.0000.137\pm 0.000 0.136±0.000plus-or-minus0.1360.0000.136\pm 0.000  LightGBM 0.136±0.000plus-or-minus0.1360.0000.136\pm 0.000 0.136±0.000plus-or-minus0.1360.0000.136\pm 0.000  Default hyperparameters  CatBoost 0.133±0.000plus-or-minus0.1330.0000.133\pm 0.000 0.132±0.000plus-or-minus0.1320.0000.132\pm 0.000  XGBoost 0.143±0.000plus-or-minus0.1430.0000.143\pm 0.000 0.143±0.000plus-or-minus0.1430.0000.143\pm 0.000  LightGBM 0.137±0.000plus-or-minus0.1370.0000.137\pm 0.000 0.137±0.000plus-or-minus0.1370.0000.137\pm 0.000  TabR-S 0.133±0.001plus-or-minus0.1330.0010.133\pm 0.001 0.131±0.000plus-or-minus0.1310.0000.131\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.142±0.001plus-or-minus0.1420.0010.142\pm 0.001 0.139±0.001plus-or-minus0.1390.0010.139\pm 0.001  step-1 0.142±0.002plus-or-minus0.1420.0020.142\pm 0.002 0.138±0.000plus-or-minus0.1380.0000.138\pm 0.000  step-2 0.140±0.001plus-or-minus0.1400.0010.140\pm 0.001 0.139±0.001plus-or-minus0.1390.0010.139\pm 0.001  step-3 0.135±0.001plus-or-minus0.1350.0010.135\pm 0.001 0.133±0.001plus-or-minus0.1330.0010.133\pm 0.001 | OT ↑  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.774±0.000plus-or-minus0.7740.0000.774\pm 0.000 –  DNNR – –  DKL – –  ANP – –  NPT 0.815±0.002plus-or-minus0.8150.0020.815\pm 0.002 –  SAINT 0.812±0.002plus-or-minus0.8120.0020.812\pm 0.002 –  MLP 0.816±0.003plus-or-minus0.8160.0030.816\pm 0.003 –  MLP-PLR 0.819±0.002plus-or-minus0.8190.0020.819\pm 0.002 0.822±0.002plus-or-minus0.8220.0020.822\pm 0.002  TabR-S 0.818±0.002plus-or-minus0.8180.0020.818\pm 0.002 0.824±0.001plus-or-minus0.8240.0010.824\pm 0.001  TabR 0.825±0.002plus-or-minus0.8250.0020.825\pm 0.002 0.831±0.001plus-or-minus0.8310.0010.831\pm 0.001  CatBoost 0.825±0.001plus-or-minus0.8250.0010.825\pm 0.001 0.827±0.000plus-or-minus0.8270.0000.827\pm 0.000  XGBoost 0.830±0.001plus-or-minus0.8300.0010.830\pm 0.001 0.832±0.001plus-or-minus0.8320.0010.832\pm 0.001  LightGBM 0.830±0.001plus-or-minus0.8300.0010.830\pm 0.001 0.832±0.001plus-or-minus0.8320.0010.832\pm 0.001  Default hyperparameters  CatBoost 0.820±0.001plus-or-minus0.8200.0010.820\pm 0.001 0.822±0.001plus-or-minus0.8220.0010.822\pm 0.001  XGBoost 0.817±0.000plus-or-minus0.8170.0000.817\pm 0.000 0.817±0.000plus-or-minus0.8170.0000.817\pm 0.000  LightGBM 0.826±0.000plus-or-minus0.8260.0000.826\pm 0.000 0.826±0.000plus-or-minus0.8260.0000.826\pm 0.000  TabR-S 0.816±0.002plus-or-minus0.8160.0020.816\pm 0.002 0.824±0.000plus-or-minus0.8240.0000.824\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.814±0.002plus-or-minus0.8140.0020.814\pm 0.002 0.823±0.002plus-or-minus0.8230.0020.823\pm 0.002  step-1 0.814±0.002plus-or-minus0.8140.0020.814\pm 0.002 0.824±0.001plus-or-minus0.8240.0010.824\pm 0.001  step-2 0.813±0.002plus-or-minus0.8130.0020.813\pm 0.002 0.818±0.001plus-or-minus0.8180.0010.818\pm 0.001  step-3 0.810±0.002plus-or-minus0.8100.0020.810\pm 0.002 0.814±0.001plus-or-minus0.8140.0010.814\pm 0.001 |
| HI ↑  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.665±0.000plus-or-minus0.6650.0000.665\pm 0.000 –  DNNR – –  DKL – –  ANP – –  NPT 0.721±0.003plus-or-minus0.7210.0030.721\pm 0.003 –  SAINT 0.724±0.002plus-or-minus0.7240.0020.724\pm 0.002 –  MLP 0.719±0.002plus-or-minus0.7190.0020.719\pm 0.002 –  MLP-PLR 0.729±0.002plus-or-minus0.7290.0020.729\pm 0.002 0.735±0.000plus-or-minus0.7350.0000.735\pm 0.000  TabR-S 0.722±0.001plus-or-minus0.7220.0010.722\pm 0.001 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001  TabR 0.729±0.001plus-or-minus0.7290.0010.729\pm 0.001 0.733±0.001plus-or-minus0.7330.0010.733\pm 0.001  CatBoost 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001 0.727±0.001plus-or-minus0.7270.0010.727\pm 0.001  XGBoost 0.725±0.002plus-or-minus0.7250.0020.725\pm 0.002 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001  LightGBM 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001  Default hyperparameters  CatBoost 0.725±0.001plus-or-minus0.7250.0010.725\pm 0.001 0.726±0.001plus-or-minus0.7260.0010.726\pm 0.001  XGBoost 0.716±0.000plus-or-minus0.7160.0000.716\pm 0.000 0.716±0.000plus-or-minus0.7160.0000.716\pm 0.000  LightGBM 0.720±0.000plus-or-minus0.7200.0000.720\pm 0.000 0.720±0.000plus-or-minus0.7200.0000.720\pm 0.000  TabR-S 0.719±0.002plus-or-minus0.7190.0020.719\pm 0.002 0.724±0.000plus-or-minus0.7240.0000.724\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.719±0.002plus-or-minus0.7190.0020.719\pm 0.002 0.727±0.000plus-or-minus0.7270.0000.727\pm 0.000  step-1 0.719±0.002plus-or-minus0.7190.0020.719\pm 0.002 0.724±0.001plus-or-minus0.7240.0010.724\pm 0.001  step-2 0.720±0.002plus-or-minus0.7200.0020.720\pm 0.002 0.723±0.001plus-or-minus0.7230.0010.723\pm 0.001  step-3 0.722±0.002plus-or-minus0.7220.0020.722\pm 0.002 0.724±0.000plus-or-minus0.7240.0000.724\pm 0.000 | BL ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.712±0.000plus-or-minus0.7120.0000.712\pm 0.000 –  DNNR 0.704±0.000plus-or-minus0.7040.0000.704\pm 0.000 –  DKL 0.699±0.001plus-or-minus0.6990.0010.699\pm 0.001 –  ANP 0.705±0.005plus-or-minus0.7050.0050.705\pm 0.005 –  NPT 0.692±0.001plus-or-minus0.6920.0010.692\pm 0.001 –  SAINT 0.693±0.001plus-or-minus0.6930.0010.693\pm 0.001 –  MLP 0.697±0.001plus-or-minus0.6970.0010.697\pm 0.001 –  MLP-PLR 0.687±0.000plus-or-minus0.6870.0000.687\pm 0.000 0.684±0.000plus-or-minus0.6840.0000.684\pm 0.000  TabR-S 0.690±0.000plus-or-minus0.6900.0000.690\pm 0.000 0.688±0.000plus-or-minus0.6880.0000.688\pm 0.000  TabR 0.676±0.001plus-or-minus0.6760.0010.676\pm 0.001 0.674±0.001plus-or-minus0.6740.0010.674\pm 0.001  CatBoost 0.682±0.000plus-or-minus0.6820.0000.682\pm 0.000 0.681±0.000plus-or-minus0.6810.0000.681\pm 0.000  XGBoost 0.681±0.000plus-or-minus0.6810.0000.681\pm 0.000 0.680±0.000plus-or-minus0.6800.0000.680\pm 0.000  LightGBM 0.680±0.000plus-or-minus0.6800.0000.680\pm 0.000 0.679±0.000plus-or-minus0.6790.0000.679\pm 0.000  Default hyperparameters  CatBoost 0.685±0.000plus-or-minus0.6850.0000.685\pm 0.000 0.684±0.000plus-or-minus0.6840.0000.684\pm 0.000  XGBoost 0.683±0.000plus-or-minus0.6830.0000.683\pm 0.000 0.683±0.000plus-or-minus0.6830.0000.683\pm 0.000  LightGBM 0.681±0.000plus-or-minus0.6810.0000.681\pm 0.000 0.681±0.000plus-or-minus0.6810.0000.681\pm 0.000  TabR-S 0.691±0.000plus-or-minus0.6910.0000.691\pm 0.000 0.688±0.000plus-or-minus0.6880.0000.688\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.699±0.001plus-or-minus0.6990.0010.699\pm 0.001 0.694±0.001plus-or-minus0.6940.0010.694\pm 0.001  step-1 0.698±0.001plus-or-minus0.6980.0010.698\pm 0.001 0.693±0.001plus-or-minus0.6930.0010.693\pm 0.001  step-2 0.692±0.001plus-or-minus0.6920.0010.692\pm 0.001 0.690±0.000plus-or-minus0.6900.0000.690\pm 0.000  step-3 0.692±0.001plus-or-minus0.6920.0010.692\pm 0.001 0.688±0.000plus-or-minus0.6880.0000.688\pm 0.000 |
| WE ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 2.296±0.000plus-or-minus2.2960.0002.296\pm 0.000 –  DNNR 1.913±0.000plus-or-minus1.9130.0001.913\pm 0.000 –  DKL – –  ANP 1.902±0.009plus-or-minus1.9020.0091.902\pm 0.009 –  NPT 1.947±0.006plus-or-minus1.9470.0061.947\pm 0.006 –  SAINT 1.933±0.028plus-or-minus1.9330.0281.933\pm 0.028 –  MLP 1.905±0.005plus-or-minus1.9050.0051.905\pm 0.005 –  MLP-PLR 1.860±0.002plus-or-minus1.8600.0021.860\pm 0.002 1.833±0.002plus-or-minus1.8330.0021.833\pm 0.002  TabR-S 1.747±0.002plus-or-minus1.7470.0021.747\pm 0.002 1.718±0.001plus-or-minus1.7180.0011.718\pm 0.001  TabR 1.690±0.003plus-or-minus1.6900.0031.690\pm 0.003 1.661±0.002plus-or-minus1.6610.0021.661\pm 0.002  CatBoost 1.807±0.002plus-or-minus1.8070.0021.807\pm 0.002 1.773±0.001plus-or-minus1.7730.0011.773\pm 0.001  XGBoost 1.784±0.001plus-or-minus1.7840.0011.784\pm 0.001 1.769±0.001plus-or-minus1.7690.0011.769\pm 0.001  LightGBM 1.771±0.001plus-or-minus1.7710.0011.771\pm 0.001 1.761±0.001plus-or-minus1.7610.0011.761\pm 0.001  Default hyperparameters  CatBoost 1.895±0.001plus-or-minus1.8950.0011.895\pm 0.001 1.886±0.000plus-or-minus1.8860.0001.886\pm 0.000  XGBoost 1.920±0.000plus-or-minus1.9200.0001.920\pm 0.000 1.920±0.000plus-or-minus1.9200.0001.920\pm 0.000  LightGBM 1.845±0.003plus-or-minus1.8450.0031.845\pm 0.003 1.817±0.001plus-or-minus1.8170.0011.817\pm 0.001  TabR-S 1.755±0.002plus-or-minus1.7550.0021.755\pm 0.002 1.721±0.002plus-or-minus1.7210.0021.721\pm 0.002  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 1.903±0.004plus-or-minus1.9030.0041.903\pm 0.004 1.835±0.004plus-or-minus1.8350.0041.835\pm 0.004  step-1 1.906±0.003plus-or-minus1.9060.0031.906\pm 0.003 1.845±0.001plus-or-minus1.8450.0011.845\pm 0.001  step-2 1.804±0.003plus-or-minus1.8040.0031.804\pm 0.003 1.754±0.001plus-or-minus1.7540.0011.754\pm 0.001  step-3 1.814±0.003plus-or-minus1.8140.0031.814\pm 0.003 1.765±0.001plus-or-minus1.7650.0011.765\pm 0.001 | CO ↑  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.927±0.000plus-or-minus0.9270.0000.927\pm 0.000 –  DNNR – –  DKL – –  ANP – –  NPT 0.966±0.001plus-or-minus0.9660.0010.966\pm 0.001 –  SAINT 0.964±0.010plus-or-minus0.9640.0100.964\pm 0.010 –  MLP 0.963±0.001plus-or-minus0.9630.0010.963\pm 0.001 –  MLP-PLR 0.970±0.001plus-or-minus0.9700.0010.970\pm 0.001 0.974±0.000plus-or-minus0.9740.0000.974\pm 0.000  TabR-S 0.973±0.000plus-or-minus0.9730.0000.973\pm 0.000 0.974±0.000plus-or-minus0.9740.0000.974\pm 0.000  TabR 0.976±0.000plus-or-minus0.9760.0000.976\pm 0.000 0.977±0.000plus-or-minus0.9770.0000.977\pm 0.000  CatBoost 0.968±0.000plus-or-minus0.9680.0000.968\pm 0.000 0.969±0.000plus-or-minus0.9690.0000.969\pm 0.000  XGBoost 0.971±0.000plus-or-minus0.9710.0000.971\pm 0.000 0.971±0.000plus-or-minus0.9710.0000.971\pm 0.000  LightGBM 0.971±0.000plus-or-minus0.9710.0000.971\pm 0.000 0.971±0.000plus-or-minus0.9710.0000.971\pm 0.000  Default hyperparameters  CatBoost 0.923±0.000plus-or-minus0.9230.0000.923\pm 0.000 0.924±0.000plus-or-minus0.9240.0000.924\pm 0.000  XGBoost 0.966±0.000plus-or-minus0.9660.0000.966\pm 0.000 0.966±0.000plus-or-minus0.9660.0000.966\pm 0.000  LightGBM 0.884±0.016plus-or-minus0.8840.0160.884\pm 0.016 0.899±0.005plus-or-minus0.8990.0050.899\pm 0.005  TabR-S 0.973±0.001plus-or-minus0.9730.0010.973\pm 0.001 0.974±0.000plus-or-minus0.9740.0000.974\pm 0.000  Tuned hyperparameters ([Table 2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 TabR ‣ TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023"))  step-0 0.957±0.002plus-or-minus0.9570.0020.957\pm 0.002 0.965±0.001plus-or-minus0.9650.0010.965\pm 0.001  step-1 0.960±0.002plus-or-minus0.9600.0020.960\pm 0.002 0.967±0.001plus-or-minus0.9670.0010.967\pm 0.001  step-2 0.972±0.000plus-or-minus0.9720.0000.972\pm 0.000 0.973±0.000plus-or-minus0.9730.0000.973\pm 0.000  step-3 0.975±0.001plus-or-minus0.9750.0010.975\pm 0.001 0.976±0.000plus-or-minus0.9760.0000.976\pm 0.000 |
| MI ↓  Method Single model Ensemble  Tuned Hyperparameters  kNN 0.764±0.000plus-or-minus0.7640.0000.764\pm 0.000 –  DNNR 0.765±0.000plus-or-minus0.7650.0000.765\pm 0.000 –  DKL – –  ANP – –  NPT 0.753±0.001plus-or-minus0.7530.0010.753\pm 0.001 –  SAINT 0.763±0.007plus-or-minus0.7630.0070.763\pm 0.007 –  MLP 0.748±0.000plus-or-minus0.7480.0000.748\pm 0.000 –  MLP-PLR 0.744±0.000plus-or-minus0.7440.0000.744\pm 0.000 0.743±0.000plus-or-minus0.7430.0000.743\pm 0.000  TabR-S 0.750±0.001plus-or-minus0.7500.0010.750\pm 0.001 0.749±0.000plus-or-minus0.7490.0000.749\pm 0.000  TabR 0.750±0.001plus-or-minus0.7500.0010.750\pm 0.001 0.748±0.000plus-or-minus0.7480.0000.748\pm 0.000  CatBoost 0.741±0.000plus-or-minus0.7410.0000.741\pm 0.000 0.741±0.000plus-or-minus0.7410.0000.741\pm 0.000  XGBoost 0.741±0.000plus-or-minus0.7410.0000.741\pm 0.000 0.741±0.000plus-or-minus0.7410.0000.741\pm 0.000  LightGBM 0.742±0.000plus-or-minus0.7420.0000.742\pm 0.000 0.741±0.000plus-or-minus0.7410.0000.741\pm 0.000  Default hyperparameters  CatBoost 0.745±0.000plus-or-minus0.7450.0000.745\pm 0.000 0.744±0.000plus-or-minus0.7440.0000.744\pm 0.000  XGBoost 0.750±0.000plus-or-minus0.7500.0000.750\pm 0.000 0.750±0.000plus-or-minus0.7500.0000.750\pm 0.000  LightGBM 0.747±0.000plus-or-minus0.7470.0000.747\pm 0.000 0.744±0.000plus-or-minus0.7440.0000.744\pm 0.000  TabR-S 0.757±0.001plus-or-minus0.7570.0010.757\pm 0.001 0.752±0.001plus-or-minus0.7520.0010.752\pm 0.001 |  |

Table 26: Extended results for Grinsztajn et al. [[2022](#bib.bib13)] benchmark. Results are grouped by datasets and span multiple pages below. Notation: ↓ corresponds to RMSE, ↑ corresponds to accuracy.

|  |  |
| --- | --- |
| Ailerons ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 1.624±0.035plus-or-minus1.6240.0351.624\pm 0.035 1.620±0.037plus-or-minus1.6200.0371.620\pm 0.037  MLP-PLR 1.591±0.021plus-or-minus1.5910.0211.591\pm 0.021 1.582±0.019plus-or-minus1.5820.0191.582\pm 0.019  TabR-S 1.620±0.030plus-or-minus1.6200.0301.620\pm 0.030 1.595±0.022plus-or-minus1.5950.0221.595\pm 0.022  TabR 1.615±0.035plus-or-minus1.6150.0351.615\pm 0.035 1.585±0.042plus-or-minus1.5850.0421.585\pm 0.042  CatBoost 1.533±0.034plus-or-minus1.5330.0341.533\pm 0.034 1.527±0.037plus-or-minus1.5270.0371.527\pm 0.037  XGBoost 1.571±0.041plus-or-minus1.5710.0411.571\pm 0.041 1.565±0.040plus-or-minus1.5650.0401.565\pm 0.040  LightGBM 1.581±0.038plus-or-minus1.5810.0381.581\pm 0.038 1.577±0.040plus-or-minus1.5770.0401.577\pm 0.040  Default hyperparameters  TabR-S 1.615±0.029plus-or-minus1.6150.0291.615\pm 0.029 1.599±0.029plus-or-minus1.5990.0291.599\pm 0.029  CatBoost 1.542±0.041plus-or-minus1.5420.0411.542\pm 0.041 1.538±0.043plus-or-minus1.5380.0431.538\pm 0.043  XGBoost 1.644±0.046plus-or-minus1.6440.0461.644\pm 0.046 1.644±0.048plus-or-minus1.6440.0481.644\pm 0.048  LightGBM 1.594±0.051plus-or-minus1.5940.0511.594\pm 0.051 1.594±0.053plus-or-minus1.5940.0531.594\pm 0.053 | Bike Sharing Demand ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 45.702±0.756plus-or-minus45.7020.75645.702\pm 0.756 43.203±0.132plus-or-minus43.2030.13243.203\pm 0.132  MLP-PLR 42.615±0.415plus-or-minus42.6150.41542.615\pm 0.415 41.470±0.324plus-or-minus41.4700.32441.470\pm 0.324  TabR-S 43.637±0.681plus-or-minus43.6370.68143.637\pm 0.681 42.339±0.415plus-or-minus42.3390.41542.339\pm 0.415  TabR 42.649±0.939plus-or-minus42.6490.93942.649\pm 0.939 41.227±0.615plus-or-minus41.2270.61541.227\pm 0.615  CatBoost 40.927±0.232plus-or-minus40.9270.23240.927\pm 0.232 40.552±0.090plus-or-minus40.5520.09040.552\pm 0.090  XGBoost 42.766±0.126plus-or-minus42.7660.12642.766\pm 0.126 42.606±0.039plus-or-minus42.6060.03942.606\pm 0.039  LightGBM 42.503±0.190plus-or-minus42.5030.19042.503\pm 0.190 42.342±0.149plus-or-minus42.3420.14942.342\pm 0.149  Default hyperparameters  TabR-S 43.486±0.573plus-or-minus43.4860.57343.486\pm 0.573 42.369±0.354plus-or-minus42.3690.35442.369\pm 0.354  CatBoost 42.848±0.256plus-or-minus42.8480.25642.848\pm 0.256 42.626±0.243plus-or-minus42.6260.24342.626\pm 0.243  XGBoost 45.100±0.381plus-or-minus45.1000.38145.100\pm 0.381 45.100±0.410plus-or-minus45.1000.41045.100\pm 0.410  LightGBM 43.089±0.103plus-or-minus43.0890.10343.089\pm 0.103 43.089±0.111plus-or-minus43.0890.11143.089\pm 0.111 |
| Brazilian houses ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.049±0.018plus-or-minus0.0490.0180.049\pm 0.018 0.046±0.021plus-or-minus0.0460.0210.046\pm 0.021  MLP-PLR 0.043±0.019plus-or-minus0.0430.0190.043\pm 0.019 0.040±0.022plus-or-minus0.0400.0220.040\pm 0.022  TabR-S 0.049±0.015plus-or-minus0.0490.0150.049\pm 0.015 0.045±0.017plus-or-minus0.0450.0170.045\pm 0.017  TabR 0.045±0.016plus-or-minus0.0450.0160.045\pm 0.016 0.041±0.017plus-or-minus0.0410.0170.041\pm 0.017  CatBoost 0.047±0.031plus-or-minus0.0470.0310.047\pm 0.031 0.046±0.033plus-or-minus0.0460.0330.046\pm 0.033  XGBoost 0.054±0.027plus-or-minus0.0540.0270.054\pm 0.027 0.053±0.029plus-or-minus0.0530.0290.053\pm 0.029  LightGBM 0.060±0.025plus-or-minus0.0600.0250.060\pm 0.025 0.059±0.027plus-or-minus0.0590.0270.059\pm 0.027  Default hyperparameters  TabR-S 0.052±0.016plus-or-minus0.0520.0160.052\pm 0.016 0.048±0.018plus-or-minus0.0480.0180.048\pm 0.018  CatBoost 0.043±0.027plus-or-minus0.0430.0270.043\pm 0.027 0.042±0.029plus-or-minus0.0420.0290.042\pm 0.029  XGBoost 0.052±0.025plus-or-minus0.0520.0250.052\pm 0.025 0.052±0.027plus-or-minus0.0520.0270.052\pm 0.027  LightGBM 0.071±0.021plus-or-minus0.0710.0210.071\pm 0.021 0.071±0.022plus-or-minus0.0710.0220.071\pm 0.022 | Higgs ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.723±0.002plus-or-minus0.7230.0020.723\pm 0.002 0.725±0.001plus-or-minus0.7250.0010.725\pm 0.001  MLP-PLR 0.728±0.001plus-or-minus0.7280.0010.728\pm 0.001 0.730±0.001plus-or-minus0.7300.0010.730\pm 0.001  TabR-S 0.725±0.001plus-or-minus0.7250.0010.725\pm 0.001 0.728±0.000plus-or-minus0.7280.0000.728\pm 0.000  TabR 0.730±0.001plus-or-minus0.7300.0010.730\pm 0.001 0.733±0.000plus-or-minus0.7330.0000.733\pm 0.000  CatBoost 0.729±0.000plus-or-minus0.7290.0000.729\pm 0.000 0.730±0.000plus-or-minus0.7300.0000.730\pm 0.000  XGBoost 0.729±0.001plus-or-minus0.7290.0010.729\pm 0.001 0.730±0.000plus-or-minus0.7300.0000.730\pm 0.000  LightGBM 0.727±0.001plus-or-minus0.7270.0010.727\pm 0.001 0.728±0.000plus-or-minus0.7280.0000.728\pm 0.000  Default hyperparameters  TabR-S 0.722±0.001plus-or-minus0.7220.0010.722\pm 0.001 0.727±0.001plus-or-minus0.7270.0010.727\pm 0.001  CatBoost 0.727±0.001plus-or-minus0.7270.0010.727\pm 0.001 0.728±0.001plus-or-minus0.7280.0010.728\pm 0.001  XGBoost 0.718±0.000plus-or-minus0.7180.0000.718\pm 0.000 0.718±0.000plus-or-minus0.7180.0000.718\pm 0.000  LightGBM 0.721±0.000plus-or-minus0.7210.0000.721\pm 0.000 0.721±0.000plus-or-minus0.7210.0000.721\pm 0.000 |
| KDDCup09 upselling ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.776±0.011plus-or-minus0.7760.0110.776\pm 0.011 0.782±0.009plus-or-minus0.7820.0090.782\pm 0.009  MLP-PLR 0.797±0.009plus-or-minus0.7970.0090.797\pm 0.009 0.802±0.010plus-or-minus0.8020.0100.802\pm 0.010  TabR-S 0.784±0.014plus-or-minus0.7840.0140.784\pm 0.014 0.786±0.017plus-or-minus0.7860.0170.786\pm 0.017  TabR 0.791±0.012plus-or-minus0.7910.0120.791\pm 0.012 0.803±0.008plus-or-minus0.8030.0080.803\pm 0.008  CatBoost 0.799±0.012plus-or-minus0.7990.0120.799\pm 0.012 0.801±0.012plus-or-minus0.8010.0120.801\pm 0.012  XGBoost 0.793±0.011plus-or-minus0.7930.0110.793\pm 0.011 0.795±0.010plus-or-minus0.7950.0100.795\pm 0.010  LightGBM 0.793±0.012plus-or-minus0.7930.0120.793\pm 0.012 0.797±0.011plus-or-minus0.7970.0110.797\pm 0.011  Default hyperparameters  TabR-S 0.772±0.013plus-or-minus0.7720.0130.772\pm 0.013 0.781±0.013plus-or-minus0.7810.0130.781\pm 0.013  CatBoost 0.804±0.008plus-or-minus0.8040.0080.804\pm 0.008 0.804±0.006plus-or-minus0.8040.0060.804\pm 0.006  XGBoost 0.794±0.008plus-or-minus0.7940.0080.794\pm 0.008 0.794±0.009plus-or-minus0.7940.0090.794\pm 0.009  LightGBM 0.789±0.007plus-or-minus0.7890.0070.789\pm 0.007 0.789±0.007plus-or-minus0.7890.0070.789\pm 0.007 | MagicTelescope ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.853±0.006plus-or-minus0.8530.0060.853\pm 0.006 0.857±0.004plus-or-minus0.8570.0040.857\pm 0.004  MLP-PLR 0.860±0.007plus-or-minus0.8600.0070.860\pm 0.007 0.863±0.007plus-or-minus0.8630.0070.863\pm 0.007  TabR-S 0.868±0.006plus-or-minus0.8680.0060.868\pm 0.006 0.873±0.004plus-or-minus0.8730.0040.873\pm 0.004  TabR 0.864±0.005plus-or-minus0.8640.0050.864\pm 0.005 0.868±0.002plus-or-minus0.8680.0020.868\pm 0.002  CatBoost 0.859±0.007plus-or-minus0.8590.0070.859\pm 0.007 0.859±0.008plus-or-minus0.8590.0080.859\pm 0.008  XGBoost 0.855±0.009plus-or-minus0.8550.0090.855\pm 0.009 0.859±0.011plus-or-minus0.8590.0110.859\pm 0.011  LightGBM 0.855±0.008plus-or-minus0.8550.0080.855\pm 0.008 0.856±0.009plus-or-minus0.8560.0090.856\pm 0.009  Default hyperparameters  TabR-S 0.868±0.006plus-or-minus0.8680.0060.868\pm 0.006 0.871±0.005plus-or-minus0.8710.0050.871\pm 0.005  CatBoost 0.860±0.007plus-or-minus0.8600.0070.860\pm 0.007 0.860±0.008plus-or-minus0.8600.0080.860\pm 0.008  XGBoost 0.856±0.011plus-or-minus0.8560.0110.856\pm 0.011 0.856±0.012plus-or-minus0.8560.0120.856\pm 0.012  LightGBM 0.859±0.009plus-or-minus0.8590.0090.859\pm 0.009 0.859±0.010plus-or-minus0.8590.0100.859\pm 0.010 |
| Mercedes Benz Greener Manufacturing ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 8.383±0.854plus-or-minus8.3830.8548.383\pm 0.854 8.336±0.888plus-or-minus8.3360.8888.336\pm 0.888  MLP-PLR 8.383±0.854plus-or-minus8.3830.8548.383\pm 0.854 8.336±0.888plus-or-minus8.3360.8888.336\pm 0.888  TabR-S 8.351±0.815plus-or-minus8.3510.8158.351\pm 0.815 8.269±0.840plus-or-minus8.2690.8408.269\pm 0.840  TabR 8.319±0.819plus-or-minus8.3190.8198.319\pm 0.819 8.244±0.844plus-or-minus8.2440.8448.244\pm 0.844  CatBoost 8.163±0.819plus-or-minus8.1630.8198.163\pm 0.819 8.155±0.844plus-or-minus8.1550.8448.155\pm 0.844  XGBoost 8.218±0.817plus-or-minus8.2180.8178.218\pm 0.817 8.209±0.846plus-or-minus8.2090.8468.209\pm 0.846  LightGBM 8.208±0.823plus-or-minus8.2080.8238.208\pm 0.823 8.162±0.857plus-or-minus8.1620.8578.162\pm 0.857  Default hyperparameters  TabR-S 8.290±0.838plus-or-minus8.2900.8388.290\pm 0.838 8.223±0.865plus-or-minus8.2230.8658.223\pm 0.865  CatBoost 8.167±0.825plus-or-minus8.1670.8258.167\pm 0.825 8.164±0.848plus-or-minus8.1640.8488.164\pm 0.848  XGBoost 8.371±0.787plus-or-minus8.3710.7878.371\pm 0.787 8.371±0.810plus-or-minus8.3710.8108.371\pm 0.810  LightGBM 8.280±0.845plus-or-minus8.2800.8458.280\pm 0.845 8.280±0.869plus-or-minus8.2800.8698.280\pm 0.869 | MiamiHousing2016 ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.161±0.003plus-or-minus0.1610.0030.161\pm 0.003 0.157±0.003plus-or-minus0.1570.0030.157\pm 0.003  MLP-PLR 0.150±0.002plus-or-minus0.1500.0020.150\pm 0.002 0.147±0.002plus-or-minus0.1470.0020.147\pm 0.002  TabR-S 0.142±0.002plus-or-minus0.1420.0020.142\pm 0.002 0.139±0.002plus-or-minus0.1390.0020.139\pm 0.002  TabR 0.139±0.002plus-or-minus0.1390.0020.139\pm 0.002 0.136±0.002plus-or-minus0.1360.0020.136\pm 0.002  CatBoost 0.142±0.002plus-or-minus0.1420.0020.142\pm 0.002 0.141±0.003plus-or-minus0.1410.0030.141\pm 0.003  XGBoost 0.144±0.003plus-or-minus0.1440.0030.144\pm 0.003 0.143±0.003plus-or-minus0.1430.0030.143\pm 0.003  LightGBM 0.146±0.002plus-or-minus0.1460.0020.146\pm 0.002 0.145±0.003plus-or-minus0.1450.0030.145\pm 0.003  Default hyperparameters  TabR-S 0.141±0.002plus-or-minus0.1410.0020.141\pm 0.002 0.139±0.002plus-or-minus0.1390.0020.139\pm 0.002  CatBoost 0.142±0.003plus-or-minus0.1420.0030.142\pm 0.003 0.141±0.003plus-or-minus0.1410.0030.141\pm 0.003  XGBoost 0.160±0.003plus-or-minus0.1600.0030.160\pm 0.003 0.160±0.003plus-or-minus0.1600.0030.160\pm 0.003  LightGBM 0.152±0.004plus-or-minus0.1520.0040.152\pm 0.004 0.152±0.004plus-or-minus0.1520.0040.152\pm 0.004 |
| MiniBooNE ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.947±0.001plus-or-minus0.9470.0010.947\pm 0.001 0.948±0.001plus-or-minus0.9480.0010.948\pm 0.001  MLP-PLR 0.947±0.001plus-or-minus0.9470.0010.947\pm 0.001 0.949±0.000plus-or-minus0.9490.0000.949\pm 0.000  TabR-S 0.949±0.001plus-or-minus0.9490.0010.949\pm 0.001 0.950±0.000plus-or-minus0.9500.0000.950\pm 0.000  TabR 0.948±0.001plus-or-minus0.9480.0010.948\pm 0.001 0.949±0.000plus-or-minus0.9490.0000.949\pm 0.000  CatBoost 0.945±0.001plus-or-minus0.9450.0010.945\pm 0.001 0.946±0.001plus-or-minus0.9460.0010.946\pm 0.001  XGBoost 0.944±0.001plus-or-minus0.9440.0010.944\pm 0.001 0.945±0.000plus-or-minus0.9450.0000.945\pm 0.000  LightGBM 0.942±0.001plus-or-minus0.9420.0010.942\pm 0.001 0.943±0.000plus-or-minus0.9430.0000.943\pm 0.000  Default hyperparameters  TabR-S 0.947±0.001plus-or-minus0.9470.0010.947\pm 0.001 0.950±0.001plus-or-minus0.9500.0010.950\pm 0.001  CatBoost 0.945±0.001plus-or-minus0.9450.0010.945\pm 0.001 0.945±0.000plus-or-minus0.9450.0000.945\pm 0.000  XGBoost 0.942±0.000plus-or-minus0.9420.0000.942\pm 0.000 0.942±0.000plus-or-minus0.9420.0000.942\pm 0.000  LightGBM 0.944±0.000plus-or-minus0.9440.0000.944\pm 0.000 0.944±0.000plus-or-minus0.9440.0000.944\pm 0.000 | OnlineNewsPopularity ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.862±0.001plus-or-minus0.8620.0010.862\pm 0.001 0.860±0.000plus-or-minus0.8600.0000.860\pm 0.000  MLP-PLR 0.862±0.001plus-or-minus0.8620.0010.862\pm 0.001 0.860±0.000plus-or-minus0.8600.0000.860\pm 0.000  TabR-S 0.868±0.001plus-or-minus0.8680.0010.868\pm 0.001 0.863±0.001plus-or-minus0.8630.0010.863\pm 0.001  TabR 0.862±0.001plus-or-minus0.8620.0010.862\pm 0.001 0.859±0.000plus-or-minus0.8590.0000.859\pm 0.000  CatBoost 0.853±0.000plus-or-minus0.8530.0000.853\pm 0.000 0.853±0.000plus-or-minus0.8530.0000.853\pm 0.000  XGBoost 0.854±0.000plus-or-minus0.8540.0000.854\pm 0.000 0.854±0.000plus-or-minus0.8540.0000.854\pm 0.000  LightGBM 0.855±0.000plus-or-minus0.8550.0000.855\pm 0.000 0.854±0.000plus-or-minus0.8540.0000.854\pm 0.000  Default hyperparameters  TabR-S 0.870±0.001plus-or-minus0.8700.0010.870\pm 0.001 0.864±0.000plus-or-minus0.8640.0000.864\pm 0.000  CatBoost 0.855±0.000plus-or-minus0.8550.0000.855\pm 0.000 0.854±0.000plus-or-minus0.8540.0000.854\pm 0.000  XGBoost 0.874±0.000plus-or-minus0.8740.0000.874\pm 0.000 0.874±0.000plus-or-minus0.8740.0000.874\pm 0.000  LightGBM 0.862±0.000plus-or-minus0.8620.0000.862\pm 0.000 0.862±0.000plus-or-minus0.8620.0000.862\pm 0.000 |
| SGEMM GPU kernel performance ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.016±0.000plus-or-minus0.0160.0000.016\pm 0.000 0.016±0.000plus-or-minus0.0160.0000.016\pm 0.000  MLP-PLR 0.015±0.000plus-or-minus0.0150.0000.015\pm 0.000 0.015±0.000plus-or-minus0.0150.0000.015\pm 0.000  TabR-S 0.017±0.001plus-or-minus0.0170.0010.017\pm 0.001 0.016±0.000plus-or-minus0.0160.0000.016\pm 0.000  TabR 0.015±0.000plus-or-minus0.0150.0000.015\pm 0.000 0.015±0.000plus-or-minus0.0150.0000.015\pm 0.000  CatBoost 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000  XGBoost 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000  LightGBM 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000  Default hyperparameters  TabR-S 0.017±0.001plus-or-minus0.0170.0010.017\pm 0.001 0.016±0.000plus-or-minus0.0160.0000.016\pm 0.000  CatBoost 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000  XGBoost 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000  LightGBM 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 0.017±0.000plus-or-minus0.0170.0000.017\pm 0.000 | analcatdata supreme ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.078±0.009plus-or-minus0.0780.0090.078\pm 0.009 0.077±0.010plus-or-minus0.0770.0100.077\pm 0.010  MLP-PLR 0.079±0.008plus-or-minus0.0790.0080.079\pm 0.008 0.077±0.008plus-or-minus0.0770.0080.077\pm 0.008  TabR-S 0.080±0.007plus-or-minus0.0800.0070.080\pm 0.007 0.076±0.005plus-or-minus0.0760.0050.076\pm 0.005  TabR 0.081±0.009plus-or-minus0.0810.0090.081\pm 0.009 0.075±0.005plus-or-minus0.0750.0050.075\pm 0.005  CatBoost 0.078±0.007plus-or-minus0.0780.0070.078\pm 0.007 0.073±0.002plus-or-minus0.0730.0020.073\pm 0.002  XGBoost 0.080±0.013plus-or-minus0.0800.0130.080\pm 0.013 0.077±0.011plus-or-minus0.0770.0110.077\pm 0.011  LightGBM 0.078±0.012plus-or-minus0.0780.0120.078\pm 0.012 0.077±0.011plus-or-minus0.0770.0110.077\pm 0.011  Default hyperparameters  TabR-S 0.077±0.007plus-or-minus0.0770.0070.077\pm 0.007 0.074±0.007plus-or-minus0.0740.0070.074\pm 0.007  CatBoost 0.071±0.004plus-or-minus0.0710.0040.071\pm 0.004 0.071±0.004plus-or-minus0.0710.0040.071\pm 0.004  XGBoost 0.076±0.006plus-or-minus0.0760.0060.076\pm 0.006 0.076±0.006plus-or-minus0.0760.0060.076\pm 0.006  LightGBM 0.073±0.006plus-or-minus0.0730.0060.073\pm 0.006 0.073±0.006plus-or-minus0.0730.0060.073\pm 0.006 |
| bank-marketing ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.786±0.006plus-or-minus0.7860.0060.786\pm 0.006 0.790±0.004plus-or-minus0.7900.0040.790\pm 0.004  MLP-PLR 0.795±0.005plus-or-minus0.7950.0050.795\pm 0.005 0.798±0.004plus-or-minus0.7980.0040.798\pm 0.004  TabR-S 0.800±0.005plus-or-minus0.8000.0050.800\pm 0.005 0.802±0.004plus-or-minus0.8020.0040.802\pm 0.004  TabR 0.802±0.009plus-or-minus0.8020.0090.802\pm 0.009 0.804±0.010plus-or-minus0.8040.0100.804\pm 0.010  CatBoost 0.803±0.007plus-or-minus0.8030.0070.803\pm 0.007 0.806±0.008plus-or-minus0.8060.0080.806\pm 0.008  XGBoost 0.801±0.008plus-or-minus0.8010.0080.801\pm 0.008 0.803±0.008plus-or-minus0.8030.0080.803\pm 0.008  LightGBM 0.801±0.008plus-or-minus0.8010.0080.801\pm 0.008 0.801±0.007plus-or-minus0.8010.0070.801\pm 0.007  Default hyperparameters  TabR-S 0.800±0.006plus-or-minus0.8000.0060.800\pm 0.006 0.801±0.005plus-or-minus0.8010.0050.801\pm 0.005  CatBoost 0.803±0.009plus-or-minus0.8030.0090.803\pm 0.009 0.803±0.009plus-or-minus0.8030.0090.803\pm 0.009  XGBoost 0.800±0.009plus-or-minus0.8000.0090.800\pm 0.009 0.800±0.009plus-or-minus0.8000.0090.800\pm 0.009  LightGBM 0.803±0.004plus-or-minus0.8030.0040.803\pm 0.004 0.803±0.004plus-or-minus0.8030.0040.803\pm 0.004 | black friday ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.369±0.000plus-or-minus0.3690.0000.369\pm 0.000 0.367±0.000plus-or-minus0.3670.0000.367\pm 0.000  MLP-PLR 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000  TabR-S 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000  TabR 0.362±0.002plus-or-minus0.3620.0020.362\pm 0.002 0.359±0.001plus-or-minus0.3590.0010.359\pm 0.001  CatBoost 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000  XGBoost 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000  LightGBM 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000  Default hyperparameters  TabR-S 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000  CatBoost 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000  XGBoost 0.362±0.000plus-or-minus0.3620.0000.362\pm 0.000 0.362±0.000plus-or-minus0.3620.0000.362\pm 0.000  LightGBM 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000 |
| california ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.149±0.002plus-or-minus0.1490.0020.149\pm 0.002 0.146±0.001plus-or-minus0.1460.0010.146\pm 0.001  MLP-PLR 0.138±0.001plus-or-minus0.1380.0010.138\pm 0.001 0.135±0.000plus-or-minus0.1350.0000.135\pm 0.000  TabR-S 0.124±0.001plus-or-minus0.1240.0010.124\pm 0.001 0.121±0.000plus-or-minus0.1210.0000.121\pm 0.000  TabR 0.122±0.001plus-or-minus0.1220.0010.122\pm 0.001 0.120±0.000plus-or-minus0.1200.0000.120\pm 0.000  CatBoost 0.129±0.000plus-or-minus0.1290.0000.129\pm 0.000 0.128±0.000plus-or-minus0.1280.0000.128\pm 0.000  XGBoost 0.131±0.001plus-or-minus0.1310.0010.131\pm 0.001 0.130±0.000plus-or-minus0.1300.0000.130\pm 0.000  LightGBM 0.131±0.001plus-or-minus0.1310.0010.131\pm 0.001 0.130±0.000plus-or-minus0.1300.0000.130\pm 0.000  Default hyperparameters  TabR-S 0.124±0.001plus-or-minus0.1240.0010.124\pm 0.001 0.122±0.000plus-or-minus0.1220.0000.122\pm 0.000  CatBoost 0.129±0.000plus-or-minus0.1290.0000.129\pm 0.000 0.129±0.000plus-or-minus0.1290.0000.129\pm 0.000  XGBoost 0.141±0.000plus-or-minus0.1410.0000.141\pm 0.000 0.141±0.000plus-or-minus0.1410.0000.141\pm 0.000  LightGBM 0.135±0.000plus-or-minus0.1350.0000.135\pm 0.000 0.135±0.000plus-or-minus0.1350.0000.135\pm 0.000 | compass ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.768±0.005plus-or-minus0.7680.0050.768\pm 0.005 0.776±0.006plus-or-minus0.7760.0060.776\pm 0.006  MLP-PLR 0.783±0.007plus-or-minus0.7830.0070.783\pm 0.007 0.796±0.006plus-or-minus0.7960.0060.796\pm 0.006  TabR-S 0.863±0.003plus-or-minus0.8630.0030.863\pm 0.003 0.870±0.003plus-or-minus0.8700.0030.870\pm 0.003  TabR 0.871±0.003plus-or-minus0.8710.0030.871\pm 0.003 0.879±0.001plus-or-minus0.8790.0010.879\pm 0.001  CatBoost 0.771±0.004plus-or-minus0.7710.0040.771\pm 0.004 0.775±0.003plus-or-minus0.7750.0030.775\pm 0.003  XGBoost 0.819±0.005plus-or-minus0.8190.0050.819\pm 0.005 0.822±0.003plus-or-minus0.8220.0030.822\pm 0.003  LightGBM 0.771±0.003plus-or-minus0.7710.0030.771\pm 0.003 0.773±0.003plus-or-minus0.7730.0030.773\pm 0.003  Default hyperparameters  TabR-S 0.865±0.004plus-or-minus0.8650.0040.865\pm 0.004 0.870±0.001plus-or-minus0.8700.0010.870\pm 0.001  CatBoost 0.758±0.002plus-or-minus0.7580.0020.758\pm 0.002 0.760±0.001plus-or-minus0.7600.0010.760\pm 0.001  XGBoost 0.751±0.000plus-or-minus0.7510.0000.751\pm 0.000 0.751±0.000plus-or-minus0.7510.0000.751\pm 0.000  LightGBM 0.762±0.004plus-or-minus0.7620.0040.762\pm 0.004 0.762±0.004plus-or-minus0.7620.0040.762\pm 0.004 |
| covertype ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.929±0.001plus-or-minus0.9290.0010.929\pm 0.001 0.934±0.001plus-or-minus0.9340.0010.934\pm 0.001  MLP-PLR 0.944±0.002plus-or-minus0.9440.0020.944\pm 0.002 0.950±0.001plus-or-minus0.9500.0010.950\pm 0.001  TabR-S 0.953±0.000plus-or-minus0.9530.0000.953\pm 0.000 0.954±0.000plus-or-minus0.9540.0000.954\pm 0.000  TabR 0.957±0.000plus-or-minus0.9570.0000.957\pm 0.000 0.958±0.000plus-or-minus0.9580.0000.958\pm 0.000  CatBoost 0.938±0.000plus-or-minus0.9380.0000.938\pm 0.000 0.939±0.000plus-or-minus0.9390.0000.939\pm 0.000  XGBoost 0.940±0.000plus-or-minus0.9400.0000.940\pm 0.000 0.940±0.000plus-or-minus0.9400.0000.940\pm 0.000  LightGBM 0.939±0.000plus-or-minus0.9390.0000.939\pm 0.000 0.939±0.000plus-or-minus0.9390.0000.939\pm 0.000  Default hyperparameters  TabR-S 0.952±0.000plus-or-minus0.9520.0000.952\pm 0.000 0.953±0.000plus-or-minus0.9530.0000.953\pm 0.000  CatBoost 0.912±0.000plus-or-minus0.9120.0000.912\pm 0.000 0.913±0.000plus-or-minus0.9130.0000.913\pm 0.000  XGBoost 0.927±0.000plus-or-minus0.9270.0000.927\pm 0.000 0.927±0.000plus-or-minus0.9270.0000.927\pm 0.000  LightGBM 0.936±0.000plus-or-minus0.9360.0000.936\pm 0.000 0.936±0.000plus-or-minus0.9360.0000.936\pm 0.000 | cpu act ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 2.712±0.207plus-or-minus2.7120.2072.712\pm 0.207 2.544±0.052plus-or-minus2.5440.0522.544\pm 0.052  MLP-PLR 2.270±0.048plus-or-minus2.2700.0482.270\pm 0.048 2.214±0.059plus-or-minus2.2140.0592.214\pm 0.059  TabR-S 2.298±0.053plus-or-minus2.2980.0532.298\pm 0.053 2.223±0.050plus-or-minus2.2230.0502.223\pm 0.050  TabR 2.128±0.078plus-or-minus2.1280.0782.128\pm 0.078 2.063±0.050plus-or-minus2.0630.0502.063\pm 0.050  CatBoost 2.124±0.049plus-or-minus2.1240.0492.124\pm 0.049 2.109±0.050plus-or-minus2.1090.0502.109\pm 0.050  XGBoost 2.524±0.353plus-or-minus2.5240.3532.524\pm 0.353 2.472±0.379plus-or-minus2.4720.3792.472\pm 0.379  LightGBM 2.222±0.089plus-or-minus2.2220.0892.222\pm 0.089 2.207±0.092plus-or-minus2.2070.0922.207\pm 0.092  Default hyperparameters  TabR-S 2.285±0.045plus-or-minus2.2850.0452.285\pm 0.045 2.214±0.032plus-or-minus2.2140.0322.214\pm 0.032  CatBoost 2.185±0.088plus-or-minus2.1850.0882.185\pm 0.088 2.162±0.091plus-or-minus2.1620.0912.162\pm 0.091  XGBoost 2.910±0.463plus-or-minus2.9100.4632.910\pm 0.463 2.910±0.486plus-or-minus2.9100.4862.910\pm 0.486  LightGBM 2.274±0.128plus-or-minus2.2740.1282.274\pm 0.128 2.274±0.135plus-or-minus2.2740.1352.274\pm 0.135 |
| credit ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.772±0.004plus-or-minus0.7720.0040.772\pm 0.004 0.774±0.003plus-or-minus0.7740.0030.774\pm 0.003  MLP-PLR 0.774±0.004plus-or-minus0.7740.0040.774\pm 0.004 0.775±0.006plus-or-minus0.7750.0060.775\pm 0.006  TabR-S 0.773±0.004plus-or-minus0.7730.0040.773\pm 0.004 0.774±0.004plus-or-minus0.7740.0040.774\pm 0.004  TabR 0.772±0.004plus-or-minus0.7720.0040.772\pm 0.004 0.775±0.003plus-or-minus0.7750.0030.775\pm 0.003  CatBoost 0.773±0.003plus-or-minus0.7730.0030.773\pm 0.003 0.775±0.004plus-or-minus0.7750.0040.775\pm 0.004  XGBoost 0.770±0.003plus-or-minus0.7700.0030.770\pm 0.003 0.771±0.003plus-or-minus0.7710.0030.771\pm 0.003  LightGBM 0.769±0.003plus-or-minus0.7690.0030.769\pm 0.003 0.773±0.003plus-or-minus0.7730.0030.773\pm 0.003  Default hyperparameters  TabR-S 0.772±0.005plus-or-minus0.7720.0050.772\pm 0.005 0.774±0.005plus-or-minus0.7740.0050.774\pm 0.005  CatBoost 0.771±0.005plus-or-minus0.7710.0050.771\pm 0.005 0.773±0.002plus-or-minus0.7730.0020.773\pm 0.002  XGBoost 0.772±0.002plus-or-minus0.7720.0020.772\pm 0.002 0.772±0.002plus-or-minus0.7720.0020.772\pm 0.002  LightGBM 0.771±0.003plus-or-minus0.7710.0030.771\pm 0.003 0.771±0.003plus-or-minus0.7710.0030.771\pm 0.003 | diamonds ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.091±0.002plus-or-minus0.0910.0020.091\pm 0.002 0.086±0.000plus-or-minus0.0860.0000.086\pm 0.000  MLP-PLR 0.087±0.001plus-or-minus0.0870.0010.087\pm 0.001 0.084±0.001plus-or-minus0.0840.0010.084\pm 0.001  TabR-S 0.083±0.001plus-or-minus0.0830.0010.083\pm 0.001 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000  TabR 0.083±0.001plus-or-minus0.0830.0010.083\pm 0.001 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  CatBoost 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000 0.083±0.000plus-or-minus0.0830.0000.083\pm 0.000  XGBoost 0.085±0.000plus-or-minus0.0850.0000.085\pm 0.000 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000  LightGBM 0.085±0.000plus-or-minus0.0850.0000.085\pm 0.000 0.085±0.000plus-or-minus0.0850.0000.085\pm 0.000  Default hyperparameters  TabR-S 0.084±0.001plus-or-minus0.0840.0010.084\pm 0.001 0.082±0.001plus-or-minus0.0820.0010.082\pm 0.001  CatBoost 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000  XGBoost 0.088±0.000plus-or-minus0.0880.0000.088\pm 0.000 0.088±0.000plus-or-minus0.0880.0000.088\pm 0.000  LightGBM 0.086±0.000plus-or-minus0.0860.0000.086\pm 0.000 0.086±0.000plus-or-minus0.0860.0000.086\pm 0.000 |
| electricity ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.832±0.004plus-or-minus0.8320.0040.832\pm 0.004 0.841±0.002plus-or-minus0.8410.0020.841\pm 0.002  MLP-PLR 0.841±0.004plus-or-minus0.8410.0040.841\pm 0.004 0.849±0.000plus-or-minus0.8490.0000.849\pm 0.000  TabR-S 0.924±0.003plus-or-minus0.9240.0030.924\pm 0.003 0.929±0.001plus-or-minus0.9290.0010.929\pm 0.001  TabR 0.937±0.002plus-or-minus0.9370.0020.937\pm 0.002 0.942±0.000plus-or-minus0.9420.0000.942\pm 0.000  CatBoost 0.880±0.002plus-or-minus0.8800.0020.880\pm 0.002 0.882±0.001plus-or-minus0.8820.0010.882\pm 0.001  XGBoost 0.890±0.001plus-or-minus0.8900.0010.890\pm 0.001 0.891±0.001plus-or-minus0.8910.0010.891\pm 0.001  LightGBM 0.887±0.001plus-or-minus0.8870.0010.887\pm 0.001 0.887±0.001plus-or-minus0.8870.0010.887\pm 0.001  Default hyperparameters  TabR-S 0.887±0.004plus-or-minus0.8870.0040.887\pm 0.004 0.893±0.002plus-or-minus0.8930.0020.893\pm 0.002  CatBoost 0.875±0.001plus-or-minus0.8750.0010.875\pm 0.001 0.877±0.000plus-or-minus0.8770.0000.877\pm 0.000  XGBoost 0.882±0.000plus-or-minus0.8820.0000.882\pm 0.000 0.882±0.000plus-or-minus0.8820.0000.882\pm 0.000  LightGBM 0.890±0.000plus-or-minus0.8900.0000.890\pm 0.000 0.890±0.000plus-or-minus0.8900.0000.890\pm 0.000 | elevators ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000  MLP-PLR 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  TabR-S 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000  TabR 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  CatBoost 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  XGBoost 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  LightGBM 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  Default hyperparameters  TabR-S 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000 0.005±0.000plus-or-minus0.0050.0000.005\pm 0.000  CatBoost 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  XGBoost 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000  LightGBM 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 0.002±0.000plus-or-minus0.0020.0000.002\pm 0.000 |
| fifa ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.803±0.013plus-or-minus0.8030.0130.803\pm 0.013 0.801±0.015plus-or-minus0.8010.0150.801\pm 0.015  MLP-PLR 0.794±0.011plus-or-minus0.7940.0110.794\pm 0.011 0.792±0.012plus-or-minus0.7920.0120.792\pm 0.012  TabR-S 0.790±0.012plus-or-minus0.7900.0120.790\pm 0.012 0.786±0.012plus-or-minus0.7860.0120.786\pm 0.012  TabR 0.791±0.014plus-or-minus0.7910.0140.791\pm 0.014 0.787±0.016plus-or-minus0.7870.0160.787\pm 0.016  CatBoost 0.783±0.012plus-or-minus0.7830.0120.783\pm 0.012 0.782±0.011plus-or-minus0.7820.0110.782\pm 0.011  XGBoost 0.780±0.011plus-or-minus0.7800.0110.780\pm 0.011 0.780±0.011plus-or-minus0.7800.0110.780\pm 0.011  LightGBM 0.781±0.012plus-or-minus0.7810.0120.781\pm 0.012 0.779±0.012plus-or-minus0.7790.0120.779\pm 0.012  Default hyperparameters  TabR-S 0.790±0.013plus-or-minus0.7900.0130.790\pm 0.013 0.786±0.012plus-or-minus0.7860.0120.786\pm 0.012  CatBoost 0.782±0.012plus-or-minus0.7820.0120.782\pm 0.012 0.781±0.013plus-or-minus0.7810.0130.781\pm 0.013  XGBoost 0.790±0.012plus-or-minus0.7900.0120.790\pm 0.012 0.790±0.013plus-or-minus0.7900.0130.790\pm 0.013  LightGBM 0.780±0.011plus-or-minus0.7800.0110.780\pm 0.011 0.780±0.011plus-or-minus0.7800.0110.780\pm 0.011 | house 16H ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.598±0.012plus-or-minus0.5980.0120.598\pm 0.012 0.587±0.004plus-or-minus0.5870.0040.587\pm 0.004  MLP-PLR 0.594±0.003plus-or-minus0.5940.0030.594\pm 0.003 0.589±0.002plus-or-minus0.5890.0020.589\pm 0.002  TabR-S 0.608±0.016plus-or-minus0.6080.0160.608\pm 0.016 0.590±0.006plus-or-minus0.5900.0060.590\pm 0.006  TabR 0.629±0.024plus-or-minus0.6290.0240.629\pm 0.024 0.599±0.000plus-or-minus0.5990.0000.599\pm 0.000  CatBoost 0.599±0.005plus-or-minus0.5990.0050.599\pm 0.005 0.596±0.003plus-or-minus0.5960.0030.596\pm 0.003  XGBoost 0.591±0.007plus-or-minus0.5910.0070.591\pm 0.007 0.585±0.004plus-or-minus0.5850.0040.585\pm 0.004  LightGBM 0.575±0.002plus-or-minus0.5750.0020.575\pm 0.002 0.573±0.001plus-or-minus0.5730.0010.573\pm 0.001  Default hyperparameters  TabR-S 0.603±0.015plus-or-minus0.6030.0150.603\pm 0.015 0.583±0.003plus-or-minus0.5830.0030.583\pm 0.003  CatBoost 0.591±0.002plus-or-minus0.5910.0020.591\pm 0.002 0.590±0.001plus-or-minus0.5900.0010.590\pm 0.001  XGBoost 0.589±0.000plus-or-minus0.5890.0000.589\pm 0.000 0.589±0.000plus-or-minus0.5890.0000.589\pm 0.000  LightGBM 0.593±0.000plus-or-minus0.5930.0000.593\pm 0.000 0.593±0.000plus-or-minus0.5930.0000.593\pm 0.000 |
| house sales ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.181±0.001plus-or-minus0.1810.0010.181\pm 0.001 0.178±0.000plus-or-minus0.1780.0000.178\pm 0.000  MLP-PLR 0.169±0.001plus-or-minus0.1690.0010.169\pm 0.001 0.168±0.000plus-or-minus0.1680.0000.168\pm 0.000  TabR-S 0.169±0.001plus-or-minus0.1690.0010.169\pm 0.001 0.166±0.000plus-or-minus0.1660.0000.166\pm 0.000  TabR 0.164±0.001plus-or-minus0.1640.0010.164\pm 0.001 0.161±0.000plus-or-minus0.1610.0000.161\pm 0.000  CatBoost 0.167±0.000plus-or-minus0.1670.0000.167\pm 0.000 0.167±0.000plus-or-minus0.1670.0000.167\pm 0.000  XGBoost 0.169±0.000plus-or-minus0.1690.0000.169\pm 0.000 0.169±0.000plus-or-minus0.1690.0000.169\pm 0.000  LightGBM 0.169±0.000plus-or-minus0.1690.0000.169\pm 0.000 0.169±0.000plus-or-minus0.1690.0000.169\pm 0.000  Default hyperparameters  TabR-S 0.169±0.001plus-or-minus0.1690.0010.169\pm 0.001 0.167±0.000plus-or-minus0.1670.0000.167\pm 0.000  CatBoost 0.167±0.000plus-or-minus0.1670.0000.167\pm 0.000 0.167±0.000plus-or-minus0.1670.0000.167\pm 0.000  XGBoost 0.179±0.000plus-or-minus0.1790.0000.179\pm 0.000 0.179±0.000plus-or-minus0.1790.0000.179\pm 0.000  LightGBM 0.173±0.000plus-or-minus0.1730.0000.173\pm 0.000 0.173±0.000plus-or-minus0.1730.0000.173\pm 0.000 | houses ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.233±0.002plus-or-minus0.2330.0020.233\pm 0.002 0.227±0.001plus-or-minus0.2270.0010.227\pm 0.001  MLP-PLR 0.228±0.002plus-or-minus0.2280.0020.228\pm 0.002 0.224±0.000plus-or-minus0.2240.0000.224\pm 0.000  TabR-S 0.199±0.001plus-or-minus0.1990.0010.199\pm 0.001 0.196±0.000plus-or-minus0.1960.0000.196\pm 0.000  TabR 0.201±0.002plus-or-minus0.2010.0020.201\pm 0.002 0.197±0.000plus-or-minus0.1970.0000.197\pm 0.000  CatBoost 0.216±0.001plus-or-minus0.2160.0010.216\pm 0.001 0.214±0.000plus-or-minus0.2140.0000.214\pm 0.000  XGBoost 0.219±0.001plus-or-minus0.2190.0010.219\pm 0.001 0.217±0.000plus-or-minus0.2170.0000.217\pm 0.000  LightGBM 0.219±0.001plus-or-minus0.2190.0010.219\pm 0.001 0.217±0.000plus-or-minus0.2170.0000.217\pm 0.000  Default hyperparameters  TabR-S 0.200±0.001plus-or-minus0.2000.0010.200\pm 0.001 0.197±0.001plus-or-minus0.1970.0010.197\pm 0.001  CatBoost 0.216±0.000plus-or-minus0.2160.0000.216\pm 0.000 0.216±0.000plus-or-minus0.2160.0000.216\pm 0.000  XGBoost 0.234±0.000plus-or-minus0.2340.0000.234\pm 0.000 0.234±0.000plus-or-minus0.2340.0000.234\pm 0.000  LightGBM 0.226±0.000plus-or-minus0.2260.0000.226\pm 0.000 0.226±0.000plus-or-minus0.2260.0000.226\pm 0.000 |
| isolet ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 2.223±0.189plus-or-minus2.2230.1892.223\pm 0.189 2.037±0.106plus-or-minus2.0370.1062.037\pm 0.106  MLP-PLR 2.224±0.156plus-or-minus2.2240.1562.224\pm 0.156 2.030±0.103plus-or-minus2.0300.1032.030\pm 0.103  TabR-S 1.976±0.174plus-or-minus1.9760.1741.976\pm 0.174 1.763±0.152plus-or-minus1.7630.1521.763\pm 0.152  TabR 1.992±0.181plus-or-minus1.9920.1811.992\pm 0.181 1.748±0.143plus-or-minus1.7480.1431.748\pm 0.143  CatBoost 2.867±0.014plus-or-minus2.8670.0142.867\pm 0.014 2.848±0.002plus-or-minus2.8480.0022.848\pm 0.002  XGBoost 2.757±0.047plus-or-minus2.7570.0472.757\pm 0.047 2.729±0.037plus-or-minus2.7290.0372.729\pm 0.037  LightGBM 2.701±0.030plus-or-minus2.7010.0302.701\pm 0.030 2.690±0.029plus-or-minus2.6900.0292.690\pm 0.029  Default hyperparameters  TabR-S 1.995±0.156plus-or-minus1.9950.1561.995\pm 0.156 1.754±0.106plus-or-minus1.7540.1061.754\pm 0.106  CatBoost 2.895±0.020plus-or-minus2.8950.0202.895\pm 0.020 2.863±0.013plus-or-minus2.8630.0132.863\pm 0.013  XGBoost 3.368±0.010plus-or-minus3.3680.0103.368\pm 0.010 3.368±0.011plus-or-minus3.3680.0113.368\pm 0.011  LightGBM 2.953±0.056plus-or-minus2.9530.0562.953\pm 0.056 2.953±0.058plus-or-minus2.9530.0582.953\pm 0.058 | jannis ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.785±0.003plus-or-minus0.7850.0030.785\pm 0.003 0.787±0.002plus-or-minus0.7870.0020.787\pm 0.002  MLP-PLR 0.799±0.003plus-or-minus0.7990.0030.799\pm 0.003 0.804±0.001plus-or-minus0.8040.0010.804\pm 0.001  TabR-S 0.798±0.002plus-or-minus0.7980.0020.798\pm 0.002 0.802±0.002plus-or-minus0.8020.0020.802\pm 0.002  TabR 0.805±0.002plus-or-minus0.8050.0020.805\pm 0.002 0.811±0.001plus-or-minus0.8110.0010.811\pm 0.001  CatBoost 0.798±0.002plus-or-minus0.7980.0020.798\pm 0.002 0.801±0.001plus-or-minus0.8010.0010.801\pm 0.001  XGBoost 0.797±0.002plus-or-minus0.7970.0020.797\pm 0.002 0.800±0.001plus-or-minus0.8000.0010.800\pm 0.001  LightGBM 0.796±0.002plus-or-minus0.7960.0020.796\pm 0.002 0.797±0.001plus-or-minus0.7970.0010.797\pm 0.001  Default hyperparameters  TabR-S 0.795±0.002plus-or-minus0.7950.0020.795\pm 0.002 0.800±0.001plus-or-minus0.8000.0010.800\pm 0.001  CatBoost 0.795±0.001plus-or-minus0.7950.0010.795\pm 0.001 0.797±0.000plus-or-minus0.7970.0000.797\pm 0.000  XGBoost 0.783±0.000plus-or-minus0.7830.0000.783\pm 0.000 0.783±0.000plus-or-minus0.7830.0000.783\pm 0.000  LightGBM 0.794±0.000plus-or-minus0.7940.0000.794\pm 0.000 0.794±0.000plus-or-minus0.7940.0000.794\pm 0.000 |
| kdd ipums la 97-small ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.880±0.007plus-or-minus0.8800.0070.880\pm 0.007 0.880±0.006plus-or-minus0.8800.0060.880\pm 0.006  MLP-PLR 0.883±0.005plus-or-minus0.8830.0050.883\pm 0.005 0.883±0.005plus-or-minus0.8830.0050.883\pm 0.005  TabR-S 0.880±0.008plus-or-minus0.8800.0080.880\pm 0.008 0.882±0.008plus-or-minus0.8820.0080.882\pm 0.008  TabR 0.883±0.005plus-or-minus0.8830.0050.883\pm 0.005 0.884±0.005plus-or-minus0.8840.0050.884\pm 0.005  CatBoost 0.879±0.009plus-or-minus0.8790.0090.879\pm 0.009 0.880±0.010plus-or-minus0.8800.0100.880\pm 0.010  XGBoost 0.883±0.009plus-or-minus0.8830.0090.883\pm 0.009 0.883±0.008plus-or-minus0.8830.0080.883\pm 0.008  LightGBM 0.879±0.007plus-or-minus0.8790.0070.879\pm 0.007 0.880±0.007plus-or-minus0.8800.0070.880\pm 0.007  Default hyperparameters  TabR-S 0.877±0.006plus-or-minus0.8770.0060.877\pm 0.006 0.878±0.007plus-or-minus0.8780.0070.878\pm 0.007  CatBoost 0.879±0.007plus-or-minus0.8790.0070.879\pm 0.007 0.881±0.007plus-or-minus0.8810.0070.881\pm 0.007  XGBoost 0.883±0.010plus-or-minus0.8830.0100.883\pm 0.010 0.883±0.011plus-or-minus0.8830.0110.883\pm 0.011  LightGBM 0.884±0.005plus-or-minus0.8840.0050.884\pm 0.005 0.884±0.005plus-or-minus0.8840.0050.884\pm 0.005 | medical charges ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  MLP-PLR 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  TabR-S 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  TabR 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  CatBoost 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000  XGBoost 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000  LightGBM 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000  Default hyperparameters  TabR-S 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.081±0.000plus-or-minus0.0810.0000.081\pm 0.000  CatBoost 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000 0.082±0.000plus-or-minus0.0820.0000.082\pm 0.000  XGBoost 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000 0.084±0.000plus-or-minus0.0840.0000.084\pm 0.000  LightGBM 0.083±0.000plus-or-minus0.0830.0000.083\pm 0.000 0.083±0.000plus-or-minus0.0830.0000.083\pm 0.000 |
| nyc-taxi-green-dec-2016 ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.397±0.001plus-or-minus0.3970.0010.397\pm 0.001 0.391±0.001plus-or-minus0.3910.0010.391\pm 0.001  MLP-PLR 0.368±0.002plus-or-minus0.3680.0020.368\pm 0.002 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000  TabR-S 0.358±0.022plus-or-minus0.3580.0220.358\pm 0.022 0.338±0.003plus-or-minus0.3380.0030.338\pm 0.003  TabR 0.372±0.009plus-or-minus0.3720.0090.372\pm 0.009 0.350±0.003plus-or-minus0.3500.0030.350\pm 0.003  CatBoost 0.365±0.001plus-or-minus0.3650.0010.365\pm 0.001 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000  XGBoost 0.379±0.000plus-or-minus0.3790.0000.379\pm 0.000 0.379±0.000plus-or-minus0.3790.0000.379\pm 0.000  LightGBM 0.369±0.000plus-or-minus0.3690.0000.369\pm 0.000 0.368±0.000plus-or-minus0.3680.0000.368\pm 0.000  Default hyperparameters  TabR-S 0.389±0.001plus-or-minus0.3890.0010.389\pm 0.001 0.385±0.000plus-or-minus0.3850.0000.385\pm 0.000  CatBoost 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000  XGBoost 0.386±0.000plus-or-minus0.3860.0000.386\pm 0.000 0.386±0.000plus-or-minus0.3860.0000.386\pm 0.000  LightGBM 0.372±0.000plus-or-minus0.3720.0000.372\pm 0.000 0.372±0.000plus-or-minus0.3720.0000.372\pm 0.000 | particulate-matter-ukair-2017 ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.377±0.001plus-or-minus0.3770.0010.377\pm 0.001 0.374±0.000plus-or-minus0.3740.0000.374\pm 0.000  MLP-PLR 0.367±0.001plus-or-minus0.3670.0010.367\pm 0.001 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000  TabR-S 0.361±0.000plus-or-minus0.3610.0000.361\pm 0.000 0.359±0.000plus-or-minus0.3590.0000.359\pm 0.000  TabR 0.360±0.000plus-or-minus0.3600.0000.360\pm 0.000 0.358±0.000plus-or-minus0.3580.0000.358\pm 0.000  CatBoost 0.365±0.000plus-or-minus0.3650.0000.365\pm 0.000 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000  XGBoost 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000  LightGBM 0.364±0.000plus-or-minus0.3640.0000.364\pm 0.000 0.363±0.000plus-or-minus0.3630.0000.363\pm 0.000  Default hyperparameters  TabR-S 0.361±0.001plus-or-minus0.3610.0010.361\pm 0.001 0.359±0.000plus-or-minus0.3590.0000.359\pm 0.000  CatBoost 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000  XGBoost 0.368±0.000plus-or-minus0.3680.0000.368\pm 0.000 0.368±0.000plus-or-minus0.3680.0000.368\pm 0.000  LightGBM 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000 0.366±0.000plus-or-minus0.3660.0000.366\pm 0.000 |
| phoneme ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.851±0.014plus-or-minus0.8510.0140.851\pm 0.014 0.861±0.013plus-or-minus0.8610.0130.861\pm 0.013  MLP-PLR 0.866±0.012plus-or-minus0.8660.0120.866\pm 0.012 0.875±0.012plus-or-minus0.8750.0120.875\pm 0.012  TabR-S 0.878±0.010plus-or-minus0.8780.0100.878\pm 0.010 0.884±0.005plus-or-minus0.8840.0050.884\pm 0.005  TabR 0.877±0.009plus-or-minus0.8770.0090.877\pm 0.009 0.885±0.007plus-or-minus0.8850.0070.885\pm 0.007  CatBoost 0.883±0.012plus-or-minus0.8830.0120.883\pm 0.012 0.890±0.005plus-or-minus0.8900.0050.890\pm 0.005  XGBoost 0.868±0.017plus-or-minus0.8680.0170.868\pm 0.017 0.877±0.016plus-or-minus0.8770.0160.877\pm 0.016  LightGBM 0.870±0.013plus-or-minus0.8700.0130.870\pm 0.013 0.873±0.013plus-or-minus0.8730.0130.873\pm 0.013  Default hyperparameters  TabR-S 0.877±0.007plus-or-minus0.8770.0070.877\pm 0.007 0.880±0.003plus-or-minus0.8800.0030.880\pm 0.003  CatBoost 0.879±0.011plus-or-minus0.8790.0110.879\pm 0.011 0.881±0.012plus-or-minus0.8810.0120.881\pm 0.012  XGBoost 0.870±0.016plus-or-minus0.8700.0160.870\pm 0.016 0.870±0.016plus-or-minus0.8700.0160.870\pm 0.016  LightGBM 0.874±0.007plus-or-minus0.8740.0070.874\pm 0.007 0.874±0.007plus-or-minus0.8740.0070.874\pm 0.007 | pol ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 5.659±0.543plus-or-minus5.6590.5435.659\pm 0.543 5.143±0.579plus-or-minus5.1430.5795.143\pm 0.579  MLP-PLR 2.615±0.137plus-or-minus2.6150.1372.615\pm 0.137 2.445±0.073plus-or-minus2.4450.0732.445\pm 0.073  TabR-S 6.071±0.537plus-or-minus6.0710.5376.071\pm 0.537 5.558±0.404plus-or-minus5.5580.4045.558\pm 0.404  TabR 2.577±0.169plus-or-minus2.5770.1692.577\pm 0.169 2.326±0.058plus-or-minus2.3260.0582.326\pm 0.058  CatBoost 3.632±0.101plus-or-minus3.6320.1013.632\pm 0.101 3.551±0.090plus-or-minus3.5510.0903.551\pm 0.090  XGBoost 4.296±0.064plus-or-minus4.2960.0644.296\pm 0.064 4.255±0.049plus-or-minus4.2550.0494.255\pm 0.049  LightGBM 4.232±0.337plus-or-minus4.2320.3374.232\pm 0.337 4.188±0.311plus-or-minus4.1880.3114.188\pm 0.311  Default hyperparameters  TabR-S 6.200±0.396plus-or-minus6.2000.3966.200\pm 0.396 5.804±0.248plus-or-minus5.8040.2485.804\pm 0.248  CatBoost 4.479±0.051plus-or-minus4.4790.0514.479\pm 0.051 4.400±0.039plus-or-minus4.4000.0394.400\pm 0.039  XGBoost 5.249±0.183plus-or-minus5.2490.1835.249\pm 0.183 5.249±0.197plus-or-minus5.2490.1975.249\pm 0.197  LightGBM 4.382±0.195plus-or-minus4.3820.1954.382\pm 0.195 4.382±0.210plus-or-minus4.3820.2104.382\pm 0.210 |
| rl ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.671±0.013plus-or-minus0.6710.0130.671\pm 0.013 0.677±0.013plus-or-minus0.6770.0130.677\pm 0.013  MLP-PLR 0.744±0.019plus-or-minus0.7440.0190.744\pm 0.019 0.767±0.027plus-or-minus0.7670.0270.767\pm 0.027  TabR-S 0.874±0.008plus-or-minus0.8740.0080.874\pm 0.008 0.880±0.006plus-or-minus0.8800.0060.880\pm 0.006  TabR 0.884±0.016plus-or-minus0.8840.0160.884\pm 0.016 0.891±0.013plus-or-minus0.8910.0130.891\pm 0.013  CatBoost 0.790±0.007plus-or-minus0.7900.0070.790\pm 0.007 0.793±0.005plus-or-minus0.7930.0050.793\pm 0.005  XGBoost 0.797±0.012plus-or-minus0.7970.0120.797\pm 0.012 0.799±0.012plus-or-minus0.7990.0120.799\pm 0.012  LightGBM 0.781±0.010plus-or-minus0.7810.0100.781\pm 0.010 0.787±0.007plus-or-minus0.7870.0070.787\pm 0.007  Default hyperparameters  TabR-S 0.871±0.008plus-or-minus0.8710.0080.871\pm 0.008 0.876±0.007plus-or-minus0.8760.0070.876\pm 0.007  CatBoost 0.785±0.010plus-or-minus0.7850.0100.785\pm 0.010 0.790±0.004plus-or-minus0.7900.0040.790\pm 0.004  XGBoost 0.775±0.003plus-or-minus0.7750.0030.775\pm 0.003 0.775±0.003plus-or-minus0.7750.0030.775\pm 0.003  LightGBM 0.778±0.003plus-or-minus0.7780.0030.778\pm 0.003 0.778±0.003plus-or-minus0.7780.0030.778\pm 0.003 | road-safety ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.786±0.001plus-or-minus0.7860.0010.786\pm 0.001 0.789±0.000plus-or-minus0.7890.0000.789\pm 0.000  MLP-PLR 0.785±0.002plus-or-minus0.7850.0020.785\pm 0.002 0.789±0.001plus-or-minus0.7890.0010.789\pm 0.001  TabR-S 0.840±0.001plus-or-minus0.8400.0010.840\pm 0.001 0.844±0.000plus-or-minus0.8440.0000.844\pm 0.000  TabR 0.837±0.001plus-or-minus0.8370.0010.837\pm 0.001 0.843±0.000plus-or-minus0.8430.0000.843\pm 0.000  CatBoost 0.801±0.001plus-or-minus0.8010.0010.801\pm 0.001 0.802±0.000plus-or-minus0.8020.0000.802\pm 0.000  XGBoost 0.810±0.002plus-or-minus0.8100.0020.810\pm 0.002 0.813±0.000plus-or-minus0.8130.0000.813\pm 0.000  LightGBM 0.798±0.001plus-or-minus0.7980.0010.798\pm 0.001 0.800±0.000plus-or-minus0.8000.0000.800\pm 0.000  Default hyperparameters  TabR-S 0.791±0.003plus-or-minus0.7910.0030.791\pm 0.003 0.796±0.003plus-or-minus0.7960.0030.796\pm 0.003  CatBoost 0.792±0.001plus-or-minus0.7920.0010.792\pm 0.001 0.793±0.000plus-or-minus0.7930.0000.793\pm 0.000  XGBoost 0.796±0.000plus-or-minus0.7960.0000.796\pm 0.000 0.796±0.000plus-or-minus0.7960.0000.796\pm 0.000  LightGBM 0.803±0.000plus-or-minus0.8030.0000.803\pm 0.000 0.803±0.000plus-or-minus0.8030.0000.803\pm 0.000 |
| sulfur ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.022±0.002plus-or-minus0.0220.0020.022\pm 0.002 0.021±0.002plus-or-minus0.0210.0020.021\pm 0.002  MLP-PLR 0.020±0.002plus-or-minus0.0200.0020.020\pm 0.002 0.019±0.003plus-or-minus0.0190.0030.019\pm 0.003  TabR-S 0.022±0.002plus-or-minus0.0220.0020.022\pm 0.002 0.021±0.002plus-or-minus0.0210.0020.021\pm 0.002  TabR 0.022±0.003plus-or-minus0.0220.0030.022\pm 0.003 0.020±0.003plus-or-minus0.0200.0030.020\pm 0.003  CatBoost 0.019±0.002plus-or-minus0.0190.0020.019\pm 0.002 0.019±0.002plus-or-minus0.0190.0020.019\pm 0.002  XGBoost 0.020±0.002plus-or-minus0.0200.0020.020\pm 0.002 0.020±0.002plus-or-minus0.0200.0020.020\pm 0.002  LightGBM 0.020±0.002plus-or-minus0.0200.0020.020\pm 0.002 0.020±0.002plus-or-minus0.0200.0020.020\pm 0.002  Default hyperparameters  TabR-S 0.021±0.003plus-or-minus0.0210.0030.021\pm 0.003 0.021±0.002plus-or-minus0.0210.0020.021\pm 0.002  CatBoost 0.019±0.002plus-or-minus0.0190.0020.019\pm 0.002 0.019±0.003plus-or-minus0.0190.0030.019\pm 0.003  XGBoost 0.022±0.002plus-or-minus0.0220.0020.022\pm 0.002 0.022±0.002plus-or-minus0.0220.0020.022\pm 0.002  LightGBM 0.021±0.001plus-or-minus0.0210.0010.021\pm 0.001 0.021±0.001plus-or-minus0.0210.0010.021\pm 0.001 | superconduct ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 10.724±0.062plus-or-minus10.7240.06210.724\pm 0.062 10.455±0.005plus-or-minus10.4550.00510.455\pm 0.005  MLP-PLR 10.566±0.058plus-or-minus10.5660.05810.566\pm 0.058 10.334±0.028plus-or-minus10.3340.02810.334\pm 0.028  TabR-S 10.884±0.107plus-or-minus10.8840.10710.884\pm 0.107 10.480±0.028plus-or-minus10.4800.02810.480\pm 0.028  TabR 10.384±0.056plus-or-minus10.3840.05610.384\pm 0.056 10.137±0.023plus-or-minus10.1370.02310.137\pm 0.023  CatBoost 10.242±0.022plus-or-minus10.2420.02210.242\pm 0.022 10.212±0.006plus-or-minus10.2120.00610.212\pm 0.006  XGBoost 10.161±0.020plus-or-minus10.1610.02010.161\pm 0.020 10.141±0.002plus-or-minus10.1410.00210.141\pm 0.002  LightGBM 10.163±0.012plus-or-minus10.1630.01210.163\pm 0.012 10.155±0.005plus-or-minus10.1550.00510.155\pm 0.005  Default hyperparameters  TabR-S 10.812±0.110plus-or-minus10.8120.11010.812\pm 0.110 10.423±0.046plus-or-minus10.4230.04610.423\pm 0.046  CatBoost 10.263±0.028plus-or-minus10.2630.02810.263\pm 0.028 10.222±0.006plus-or-minus10.2220.00610.222\pm 0.006  XGBoost 10.736±0.000plus-or-minus10.7360.00010.736\pm 0.000 10.736±0.000plus-or-minus10.7360.00010.736\pm 0.000  LightGBM 10.471±0.000plus-or-minus10.4710.00010.471\pm 0.000 10.471±0.000plus-or-minus10.4710.00010.471\pm 0.000 |
| visualizing soil ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.138±0.012plus-or-minus0.1380.0120.138\pm 0.012 0.132±0.010plus-or-minus0.1320.0100.132\pm 0.010  MLP-PLR 0.158±0.067plus-or-minus0.1580.0670.158\pm 0.067 0.144±0.060plus-or-minus0.1440.0600.144\pm 0.060  TabR-S 0.398±0.352plus-or-minus0.3980.3520.398\pm 0.352 0.387±0.375plus-or-minus0.3870.3750.387\pm 0.375  TabR 0.227±0.264plus-or-minus0.2270.2640.227\pm 0.264 0.202±0.147plus-or-minus0.2020.1470.202\pm 0.147  CatBoost 0.055±0.006plus-or-minus0.0550.0060.055\pm 0.006 0.047±0.006plus-or-minus0.0470.0060.047\pm 0.006  XGBoost 0.176±0.071plus-or-minus0.1760.0710.176\pm 0.071 0.154±0.054plus-or-minus0.1540.0540.154\pm 0.054  LightGBM 0.062±0.016plus-or-minus0.0620.0160.062\pm 0.016 0.062±0.017plus-or-minus0.0620.0170.062\pm 0.017  Default hyperparameters  TabR-S 0.327±0.254plus-or-minus0.3270.2540.327\pm 0.254 0.310±0.257plus-or-minus0.3100.2570.310\pm 0.257  CatBoost 0.064±0.005plus-or-minus0.0640.0050.064\pm 0.005 0.058±0.005plus-or-minus0.0580.0050.058\pm 0.005  XGBoost 0.066±0.009plus-or-minus0.0660.0090.066\pm 0.009 0.066±0.010plus-or-minus0.0660.0100.066\pm 0.010  LightGBM 0.061±0.013plus-or-minus0.0610.0130.061\pm 0.013 0.061±0.014plus-or-minus0.0610.0140.061\pm 0.014 | wine ↑  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.769±0.015plus-or-minus0.7690.0150.769\pm 0.015 0.784±0.010plus-or-minus0.7840.0100.784\pm 0.010  MLP-PLR 0.771±0.016plus-or-minus0.7710.0160.771\pm 0.016 0.783±0.014plus-or-minus0.7830.0140.783\pm 0.014  TabR-S 0.794±0.011plus-or-minus0.7940.0110.794\pm 0.011 0.805±0.006plus-or-minus0.8050.0060.805\pm 0.006  TabR 0.780±0.015plus-or-minus0.7800.0150.780\pm 0.015 0.795±0.012plus-or-minus0.7950.0120.795\pm 0.012  CatBoost 0.799±0.013plus-or-minus0.7990.0130.799\pm 0.013 0.806±0.010plus-or-minus0.8060.0100.806\pm 0.010  XGBoost 0.795±0.018plus-or-minus0.7950.0180.795\pm 0.018 0.801±0.019plus-or-minus0.8010.0190.801\pm 0.019  LightGBM 0.789±0.016plus-or-minus0.7890.0160.789\pm 0.016 0.793±0.011plus-or-minus0.7930.0110.793\pm 0.011  Default hyperparameters  TabR-S 0.791±0.012plus-or-minus0.7910.0120.791\pm 0.012 0.800±0.008plus-or-minus0.8000.0080.800\pm 0.008  CatBoost 0.796±0.010plus-or-minus0.7960.0100.796\pm 0.010 0.799±0.010plus-or-minus0.7990.0100.799\pm 0.010  XGBoost 0.796±0.010plus-or-minus0.7960.0100.796\pm 0.010 0.796±0.010plus-or-minus0.7960.0100.796\pm 0.010  LightGBM 0.798±0.004plus-or-minus0.7980.0040.798\pm 0.004 0.798±0.004plus-or-minus0.7980.0040.798\pm 0.004 |
| wine quality ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.672±0.015plus-or-minus0.6720.0150.672\pm 0.015 0.659±0.016plus-or-minus0.6590.0160.659\pm 0.016  MLP-PLR 0.654±0.018plus-or-minus0.6540.0180.654\pm 0.018 0.634±0.018plus-or-minus0.6340.0180.634\pm 0.018  TabR-S 0.632±0.010plus-or-minus0.6320.0100.632\pm 0.010 0.620±0.010plus-or-minus0.6200.0100.620\pm 0.010  TabR 0.641±0.011plus-or-minus0.6410.0110.641\pm 0.011 0.620±0.007plus-or-minus0.6200.0070.620\pm 0.007  CatBoost 0.609±0.013plus-or-minus0.6090.0130.609\pm 0.013 0.606±0.014plus-or-minus0.6060.0140.606\pm 0.014  XGBoost 0.604±0.013plus-or-minus0.6040.0130.604\pm 0.013 0.602±0.014plus-or-minus0.6020.0140.602\pm 0.014  LightGBM 0.613±0.014plus-or-minus0.6130.0140.613\pm 0.014 0.612±0.014plus-or-minus0.6120.0140.612\pm 0.014  Default hyperparameters  TabR-S 0.628±0.015plus-or-minus0.6280.0150.628\pm 0.015 0.614±0.015plus-or-minus0.6140.0150.614\pm 0.015  CatBoost 0.628±0.012plus-or-minus0.6280.0120.628\pm 0.012 0.626±0.012plus-or-minus0.6260.0120.626\pm 0.012  XGBoost 0.648±0.008plus-or-minus0.6480.0080.648\pm 0.008 0.648±0.008plus-or-minus0.6480.0080.648\pm 0.008  LightGBM 0.641±0.011plus-or-minus0.6410.0110.641\pm 0.011 0.641±0.012plus-or-minus0.6410.0120.641\pm 0.012 | year ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 8.964±0.018plus-or-minus8.9640.0188.964\pm 0.018 8.901±0.003plus-or-minus8.9010.0038.901\pm 0.003  MLP-PLR 8.927±0.013plus-or-minus8.9270.0138.927\pm 0.013 8.901±0.006plus-or-minus8.9010.0068.901\pm 0.006  TabR-S 9.007±0.015plus-or-minus9.0070.0159.007\pm 0.015 8.913±0.009plus-or-minus8.9130.0098.913\pm 0.009  TabR 8.972±0.010plus-or-minus8.9720.0108.972\pm 0.010 8.917±0.003plus-or-minus8.9170.0038.917\pm 0.003  CatBoost 9.037±0.007plus-or-minus9.0370.0079.037\pm 0.007 9.005±0.003plus-or-minus9.0050.0039.005\pm 0.003  XGBoost 9.031±0.003plus-or-minus9.0310.0039.031\pm 0.003 9.024±0.001plus-or-minus9.0240.0019.024\pm 0.001  LightGBM 9.020±0.002plus-or-minus9.0200.0029.020\pm 0.002 9.013±0.001plus-or-minus9.0130.0019.013\pm 0.001  Default hyperparameters  TabR-S 9.067±0.022plus-or-minus9.0670.0229.067\pm 0.022 8.893±0.008plus-or-minus8.8930.0088.893\pm 0.008  CatBoost 9.073±0.008plus-or-minus9.0730.0089.073\pm 0.008 9.046±0.001plus-or-minus9.0460.0019.046\pm 0.001  XGBoost 9.376±0.000plus-or-minus9.3760.0009.376\pm 0.000 9.376±0.000plus-or-minus9.3760.0009.376\pm 0.000  LightGBM 9.214±0.000plus-or-minus9.2140.0009.214\pm 0.000 9.214±0.000plus-or-minus9.2140.0009.214\pm 0.000 |
| yprop 4 1 ↓  Method Single model Ensemble  Tuned Hyperparameters  MLP 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  MLP-PLR 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  TabR-S 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  TabR 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000  CatBoost 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  XGBoost 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  LightGBM 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000  Default hyperparameters  TabR-S 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  CatBoost 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000  XGBoost 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001 0.027±0.001plus-or-minus0.0270.0010.027\pm 0.001  LightGBM 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 0.027±0.000plus-or-minus0.0270.0000.027\pm 0.000 |  |
