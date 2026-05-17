---
arxiv: '1908.07442'
authors:
- Sercan O. Arik
- Tomas Pfister
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TabNet: Attentive Interpretable Tabular Learning'
url: http://arxiv.org/abs/1908.07442v5
year: 2019
---

# TabNet: Attentive Interpretable Tabular Learning

Sercan Ö. Arık,
Tomas Pfister

###### Abstract

We propose a novel high-performance and interpretable canonical deep tabular data learning architecture, TabNet.
TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning as the learning capacity is used for the most salient features.
We demonstrate that TabNet outperforms other variants on a wide range of non-performance-saturated tabular datasets and yields interpretable feature attributions plus insights into its global behavior.
Finally, we demonstrate self-supervised learning for tabular data, significantly improving performance when unlabeled data is abundant.

## Introduction

Deep neural networks (DNNs) have shown notable success with images (He et al. [2015](#bib.bib18)), text (Lai et al. [2015](#bib.bib31)) and audio (Amodei et al. [2015](#bib.bib1)).
For these, canonical architectures that efficiently encode the raw data into meaningful representations, fuel the rapid progress.
One data type that has yet to see such success with a canonical architecture is tabular data.

!(/html/1908.07442/assets/x1.png)

Figure 1: TabNet’s sparse feature selection exemplified for Adult Census Income prediction (Dua and Graff [2017](#bib.bib11)).
Sparse feature selection enables interpretability and better learning as the capacity is used for the most salient features.
TabNet employs multiple decision blocks that focus on processing a subset of input features for reasoning.
Two decision blocks shown as examples process features that are related to professional occupation and investments, respectively, in order to predict the income level.

Despite being the most common data type in real-world AI (as it is comprised of any categorical and numerical features), (Chui et al. [2018](#bib.bib6)), deep learning for tabular data remains under-explored, with variants of ensemble decision trees (DTs) still dominating most applications (Kaggle [2019a](#bib.bib25)).
Why?
First, because DT-based approaches have certain benefits:
(i) they are representionally efficient for decision manifolds with approximately hyperplane boundaries which are common in tabular data; and
(ii) they are highly interpretable in their basic form (e.g. by tracking decision nodes) and there are popular post-hoc explainability methods for their ensemble form, e.g. (Lundberg, Erion, and Lee [2018](#bib.bib32)) – this is an important concern in many real-world applications;
(iii) they are fast to train.
Second, because previously-proposed DNN architectures are not well-suited for tabular data: e.g. stacked convolutional layers or multi-layer perceptrons (MLPs) are vastly overparametrized – the lack of appropriate inductive bias often causes them to fail to find optimal solutions for tabular decision manifolds (Goodfellow, Bengio, and Courville [2016](#bib.bib14); Shavitt and Segal [2018](#bib.bib43); Xu et al. [2019](#bib.bib52)).

Why is deep learning worth exploring for tabular data?
One obvious motivation is expected performance improvements particularly for large datasets (Hestness et al. [2017](#bib.bib19)).
In addition, unlike tree learning,
DNNs enable gradient descent-based end-to-end learning for tabular data which can have a multitude of benefits:
(i) efficiently encoding multiple data types like images along with tabular data;
(ii) alleviating the need for feature engineering, which is currently a key aspect in tree-based tabular data learning methods;
(iii) learning from streaming data
and perhaps most importantly
(iv) end-to-end models allow representation learning which enables many valuable application scenarios including data-efficient domain adaptation (Goodfellow, Bengio, and Courville [2016](#bib.bib14)), generative modeling (Radford, Metz, and Chintala [2015](#bib.bib40)) and semi-supervised learning (Dai et al. [2017](#bib.bib8)).

!(/html/1908.07442/assets/x2.png)

Figure 2: Self-supervised tabular learning. Real-world tabular datasets have interdependent feature columns, e.g., the education level can be guessed from the occupation, or the gender can be guessed from the relationship. Unsupervised representation learning by masked self-supervised learning results in an improved encoder model for the supervised learning task.

We propose a new canonical DNN architecture for tabular data, TabNet.
The main contributions are summarized as:

1. 1.

   *TabNet inputs raw tabular data without any preprocessing* and is *trained using gradient descent-based optimization*, enabling flexible integration into end-to-end learning.
2. 2.

   *TabNet uses sequential attention to choose which features to reason from at each decision step*, enabling interpretability and better learning as the learning capacity is used for the most salient features (see Fig. [1](#Sx1.F1 "Figure 1 ‣ Introduction ‣ TabNet: Attentive Interpretable Tabular Learning")).
   This feature selection is *instance-wise*, e.g. it can be different for each input, and unlike other instance-wise feature selection methods like (Chen et al. [2018](#bib.bib4)) or (Yoon, Jordon, and van der Schaar [2019](#bib.bib54)), TabNet employs a *single deep learning architecture for feature selection and reasoning*.
3. 3.

   Above design choices lead to two valuable properties: (i) *TabNet outperforms or is on par with other tabular learning models* on various datasets for classification and regression problems from different domains; and (ii) *TabNet enables two kinds of interpretability*: local interpretability that visualizes the importance of features and how they are combined, and global interpretability which quantifies the contribution of each feature to the trained model.
4. 4.

   Finally, *for the first time for tabular data*, we show significant performance improvements by using unsupervised pre-training to predict masked features (see Fig. [2](#Sx1.F2 "Figure 2 ‣ Introduction ‣ TabNet: Attentive Interpretable Tabular Learning")).

## Related Work

Feature selection: Feature selection broadly refers to judiciously picking a subset of features based on their usefulness for prediction.
Commonly-used techniques such as forward selection and Lasso regularization (Guyon and Elisseeff [2003](#bib.bib17)) attribute feature importance based on the entire training data, and are referred as *global* methods.
*Instance-wise* feature selection refers to picking features individually for each input, studied in (Chen et al. [2018](#bib.bib4)) with an explainer model to maximize the mutual information between the selected features and the response variable, and in (Yoon, Jordon, and van der Schaar [2019](#bib.bib54)) by using an actor-critic framework to mimic a baseline while optimizing the selection.
Unlike these, TabNet employs *soft feature selection with controllable sparsity in end-to-end learning* – a single model jointly performs feature selection and output mapping, resulting in superior performance with compact representations.

!(/html/1908.07442/assets/x3.png)

Figure 3: 
Illustration of DT-like classification using conventional DNN blocks (left) and the corresponding decision manifold (right).
Relevant features are selected by using multiplicative sparse masks on inputs.
The selected features are linearly transformed, and after a bias addition (to represent boundaries) ReLU performs region selection by zeroing the regions.
Aggregation of multiple regions is based on addition.
As C1subscript𝐶1C\_{1} and C2subscript𝐶2C\_{2} get larger, the decision boundary gets sharper.

Tree-based learning: DTs are commonly-used for tabular data learning. Their prominent strength is efficient picking of global features with the most statistical information gain (Grabczewski and Jankowski [2005](#bib.bib15)).
To improve the performance of standard DTs, one common approach is ensembling to reduce variance.
Among ensembling methods, random forests (Ho [1998](#bib.bib20)) use random subsets of data with randomly selected features to grow many trees. XGBoost (Chen and Guestrin [2016](#bib.bib5)) and LightGBM (Ke et al. [2017](#bib.bib27)) are the two recent ensemble DT approaches that dominate most of the recent data science competitions.
Our experimental results for various datasets show that tree-based models can be outperformed when the representation capacity is improved with deep learning while retaining their feature selecting property.
  
Integration of DNNs into DTs:
Representing DTs with DNN building blocks as in (Humbird, Peterson, and McClarren [2018](#bib.bib23)) yields redundancy in representation and inefficient learning.
Soft (neural) DTs (Wang, Aggarwal, and Liu [2017](#bib.bib50); Kontschieder et al. [2015](#bib.bib30)) use differentiable decision functions, instead of non-differentiable axis-aligned splits.
However, losing automatic feature selection often degrades performance.
In (Yang, Morillo, and Hospedales [2018](#bib.bib53)), a soft binning function is proposed to simulate DTs in DNNs, by inefficiently enumerating of all possible decisions.
(Ke et al. [2019](#bib.bib28)) proposes a DNN architecture by explicitly leveraging expressive feature combinations, however, learning is based on transferring knowledge from gradient-boosted DT.
(Tanno et al. [2018](#bib.bib46)) proposes a DNN architecture by adaptively growing from primitive blocks while representation learning into edges, routing functions and leaf nodes.
TabNet differs from these as it embeds soft feature selection with controllable sparsity via sequential attention.
  
Self-supervised learning:
Unsupervised representation learning improves supervised learning especially in small data regime (Raina et al. [2007](#bib.bib41)).
Recent work for text (Devlin et al. [2018](#bib.bib10)) and image (Trinh, Luong, and Le [2019](#bib.bib48)) data has shown significant advances – driven by the judicious choice of the unsupervised learning objective (masked input prediction) and attention-based deep learning.

## TabNet for Tabular Learning

DTs are successful for learning from real-world tabular datasets.
With a specific design, conventional DNN building blocks can be used to implement DT-like output manifold, e.g. see Fig. [3](#Sx2.F3 "Figure 3 ‣ Related Work ‣ TabNet: Attentive Interpretable Tabular Learning")).
In such a design, individual feature selection is key to obtain decision boundaries in hyperplane form, which can be generalized to a linear combination of features where coefficients determine the proportion of each feature.
TabNet is based on such functionality and it outperforms DTs while reaping their benefits by careful design which:
(i) uses sparse instance-wise feature selection learned from data;
(ii) constructs a sequential multi-step architecture, where each step contributes to a portion of the decision based on the selected features;
(iii) improves the learning capacity via non-linear processing of the selected features; and
(iv) mimics ensembling via higher dimensions and more steps.

!(/html/1908.07442/assets/x4.png)

(a) TabNet encoder architecture

!(/html/1908.07442/assets/x5.png)

(b) TabNet decoder architecture

!(/html/1908.07442/assets/x6.png)

(c)

!(/html/1908.07442/assets/x7.png)

(d)

Figure 4: (a) TabNet encoder, composed of a feature transformer, an attentive transformer and feature masking.
A split block divides the processed representation to be used by the attentive transformer of the subsequent step as well as for the overall output.
For each step, the feature selection mask provides interpretable information about the model’s functionality, and the masks can be aggregated to obtain global feature important attribution.
(b) TabNet decoder, composed of a feature transformer block at each step.
(c) A feature transformer block example – 4-layer network is shown, where 2 are shared across all decision steps and 2 are decision step-dependent.
Each layer is composed of a fully-connected (FC) layer, BN and GLU nonlinearity.
(d) An attentive transformer block example – a single layer mapping is modulated with a prior scale information which aggregates how much each feature has been used before the current decision step.
sparsemax (Martins and Astudillo [2016](#bib.bib33)) is used for normalization of the coefficients, resulting in sparse selection of the salient features.

Fig. [4](#Sx3.F4 "Figure 4 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning") shows the TabNet architecture for encoding tabular data.
We use the raw numerical features and consider mapping of categorical features with trainable embeddings.
We do not consider any global feature normalization, but merely apply batch normalization (BN).
We pass the same D𝐷D-dimensional features 𝐟∈ℜB×D𝐟superscript𝐵𝐷\mathbf{f}\in\Re^{B\times D} to each decision step, where B𝐵B is the batch size.
TabNet’s encoding is based on sequential multi-step processing with Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} decision steps.
The it​hsuperscript𝑖𝑡ℎi^{th} step inputs the processed information from the (i−1)t​hsuperscript𝑖1𝑡ℎ(i-1)^{th} step to decide which features to use and outputs the processed feature representation to be aggregated into the overall decision.
The idea of top-down attention in the sequential form is inspired by its applications in processing visual and text data (Hudson and Manning [2018](#bib.bib22)) and reinforcement learning (Mott et al. [2019](#bib.bib36)) while searching for a small subset of relevant information in high dimensional input.
  
Feature selection:
We employ a learnable mask 𝐌​[𝐢]∈ℜB×D𝐌delimited-[]𝐢superscript𝐵𝐷\mathbf{M[i]}\in\Re^{B\times D} for soft selection of the salient features. Through sparse selection of the most salient features, the learning capacity of a decision step is not wasted on irrelevant ones, and thus the model becomes more parameter efficient. The masking is multiplicative, 𝐌​[𝐢]⋅𝐟⋅𝐌delimited-[]𝐢𝐟\mathbf{M[i]}\cdot\mathbf{f}. We use an attentive transformer (see Fig. [4](#Sx3.F4 "Figure 4 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning")) to obtain the masks using the processed features from the preceding step, 𝐚​[𝐢−𝟏]𝐚delimited-[]𝐢1\mathbf{a[i-1]}:
𝐌​[𝐢]=sparsemax​(𝐏​[𝐢−𝟏]⋅hi​(𝐚​[𝐢−𝟏])).𝐌delimited-[]𝐢sparsemax⋅𝐏delimited-[]𝐢1subscripth𝑖𝐚delimited-[]𝐢1\mathbf{M[i]}=\text{sparsemax}(\mathbf{P[i-1]}\cdot\text{h}\_{i}(\mathbf{a[i-1]})).
Sparsemax normalization (Martins and Astudillo [2016](#bib.bib33)) encourages sparsity by mapping the Euclidean projection onto the probabilistic simplex, which is observed to be superior in performance and aligned with the goal of sparse feature selection for explainability. Note that ∑j=1D𝐌​[𝐢]𝐛,𝐣=1superscriptsubscript𝑗1𝐷𝐌subscriptdelimited-[]𝐢

𝐛𝐣1\sum\nolimits\_{j=1}^{D}\mathbf{M[i]\_{b,j}}=1. hisubscripth𝑖\text{h}\_{i} is a trainable function, shown in Fig. [4](#Sx3.F4 "Figure 4 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning") using a FC layer, followed by BN. 𝐏​[𝐢]𝐏delimited-[]𝐢\mathbf{P[i]} is the prior scale term, denoting how much a particular feature has been used previously:
𝐏​[𝐢]=∏j=1i(γ−𝐌​[𝐣]),𝐏delimited-[]𝐢superscriptsubscriptproduct𝑗1𝑖𝛾𝐌delimited-[]𝐣\mathbf{P[i]}=\prod\nolimits\_{j=1}^{i}(\gamma-\mathbf{M[j]}),
where γ𝛾\gamma is a relaxation parameter – when γ=1𝛾1\gamma=1, a feature is enforced to be used only at one decision step and as γ𝛾\gamma increases, more flexibility is provided to use a feature at multiple decision steps. 𝐏​[𝟎]𝐏delimited-[]0\mathbf{P[0]} is initialized as all ones, 𝟏B×Dsuperscript1𝐵𝐷\mathbf{1}^{B\times D}, without any prior on the masked features. If some features are unused (as in self-supervised learning), corresponding 𝐏​[𝟎]𝐏delimited-[]0\mathbf{P[0]} entries are made 0 to help model’s learning.
To further control the sparsity of the selected features, we propose sparsity regularization in the form of entropy (Grandvalet and Bengio [2004](#bib.bib16)),
Ls​p​a​r​s​e=∑i=1Ns​t​e​p​s∑b=1B∑j=1D−𝐌𝐛,𝐣​[𝐢]​log⁡(𝐌𝐛,𝐣​[𝐢]+ϵ)Ns​t​e​p​s⋅B,subscript𝐿𝑠𝑝𝑎𝑟𝑠𝑒superscriptsubscript𝑖1subscript𝑁𝑠𝑡𝑒𝑝𝑠superscriptsubscript𝑏1𝐵superscriptsubscript𝑗1𝐷subscript𝐌

𝐛𝐣delimited-[]𝐢subscript𝐌

𝐛𝐣delimited-[]𝐢italic-ϵ⋅subscript𝑁𝑠𝑡𝑒𝑝𝑠𝐵L\_{sparse}=\sum\nolimits\_{i=1}^{N\_{steps}}\sum\nolimits\_{b=1}^{B}\sum\nolimits\_{j=1}^{D}\frac{-\mathbf{M\_{b,j}[i]}\log(\mathbf{M\_{b,j}[i]}\!+\!\epsilon)}{N\_{steps}\cdot B},
where ϵitalic-ϵ\epsilon is a small number for numerical stability. We add the sparsity regularization to the overall loss, with a coefficient λs​p​a​r​s​esubscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒\lambda\_{sparse}. Sparsity provides a favorable inductive bias for datasets where most features are redundant.
  
Feature processing:
We process the filtered features using a feature transformer (see Fig. [4](#Sx3.F4 "Figure 4 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning")) and then split for the decision step output and information for the subsequent step,
[𝐝​[𝐢],𝐚​[𝐢]]=fi​(𝐌​[𝐢]⋅𝐟)𝐝delimited-[]𝐢𝐚delimited-[]𝐢subscriptf𝑖⋅𝐌delimited-[]𝐢𝐟[\mathbf{d[i]},\mathbf{a[i]}]=\text{f}\_{i}(\mathbf{M[i]}\cdot\mathbf{f}),
where 𝐝​[𝐢]∈ℜB×Nd𝐝delimited-[]𝐢superscript𝐵subscript𝑁𝑑\mathbf{d[i]}\in\Re^{B\times N\_{d}} and 𝐚​[𝐢]∈ℜB×Na𝐚delimited-[]𝐢superscript𝐵subscript𝑁𝑎\mathbf{a[i]}\in\Re^{B\times N\_{a}}. For parameter-efficient and robust learning with high capacity, a feature transformer should comprise layers that are shared across all decision steps (as the same features are input across different decision steps), as well as decision step-dependent layers. Fig. [4](#Sx3.F4 "Figure 4 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning") shows the implementation as concatenation of two shared layers and two decision step-dependent layers. Each FC layer is followed by BN and gated linear unit (GLU) nonlinearity (Dauphin et al. [2016](#bib.bib9)),
eventually connected to a normalized residual connection with normalization. Normalization with 0.50.5\sqrt{0.5} helps to stabilize learning by ensuring that the variance throughout the network does not change dramatically (Gehring et al. [2017](#bib.bib12)). For faster training, we use large batch sizes with BN. Thus, except the one applied to the input features, we use ghost BN (Hoffer, Hubara, and Soudry [2017](#bib.bib21)) form, using a virtual batch size BVsubscript𝐵𝑉B\_{V} and momentum mBsubscript𝑚𝐵m\_{B}. For the input features, we observe the benefit of low-variance averaging and hence avoid ghost BN.
Finally, inspired by decision-tree like aggregation as in Fig. [3](#Sx2.F3 "Figure 3 ‣ Related Work ‣ TabNet: Attentive Interpretable Tabular Learning"), we construct the overall decision embedding as 𝐝𝐨𝐮𝐭=∑i=1Ns​t​e​p​sReLU​(𝐝​[𝐢])subscript𝐝𝐨𝐮𝐭superscriptsubscript𝑖1subscript𝑁𝑠𝑡𝑒𝑝𝑠ReLU𝐝delimited-[]𝐢\mathbf{d\_{out}}=\sum\nolimits\_{i=1}^{N\_{steps}}\text{ReLU}(\mathbf{d[i]}).
We apply a linear mapping 𝐖𝐟𝐢𝐧𝐚𝐥​𝐝𝐨𝐮𝐭subscript𝐖𝐟𝐢𝐧𝐚𝐥subscript𝐝𝐨𝐮𝐭\mathbf{W\_{final}}\mathbf{d\_{out}} to get the output mapping.111For discrete outputs, we additionally employ softmax during training (and argmax during inference).
  
Interpretability:
TabNet’s feature selection masks can shed light on the selected features at each step.
If 𝐌𝐛,𝐣​[𝐢]=0subscript𝐌

𝐛𝐣delimited-[]𝐢0\mathbf{M\_{b,j}[i]}=0, then jt​hsuperscript𝑗𝑡ℎj^{th} feature of the bt​hsuperscript𝑏𝑡ℎb^{th} sample should have no contribution to the decision.
If fisubscriptf𝑖\text{f}\_{i} were a linear function, the coefficient 𝐌𝐛,𝐣​[𝐢]subscript𝐌

𝐛𝐣delimited-[]𝐢\mathbf{M\_{b,j}[i]} would correspond to the feature importance of 𝐟𝐛,𝐣subscript𝐟

𝐛𝐣\mathbf{f\_{b,j}}.
Although each decision step employs non-linear processing, their outputs are combined later in a linear way.
We aim to quantify an aggregate feature importance in addition to analysis of each step.
Combining the masks at different steps requires a coefficient that can weigh the relative importance of each step in the decision.
We simply propose η𝐛​[𝐢]=∑c=1NdReLU​(𝐝𝐛,𝐜​[𝐢])subscript𝜂𝐛delimited-[]𝐢superscriptsubscript𝑐1subscript𝑁𝑑ReLUsubscript𝐝

𝐛𝐜delimited-[]𝐢\mathbf{\eta\_{b}[i]}=\sum\_{c=1}^{N\_{d}}\text{ReLU}(\mathbf{d\_{b,c}[i]}) to denote the aggregate decision contribution at it​hsuperscript𝑖𝑡ℎi^{th} decision step for the bt​hsuperscript𝑏𝑡ℎb^{th} sample.
Intuitively, if 𝐝𝐛,𝐜​[𝐢]<0subscript𝐝

𝐛𝐜delimited-[]𝐢0\mathbf{d\_{b,c}[i]}<0, then all features at it​hsuperscript𝑖𝑡ℎi^{th} decision step should have 0 contribution to the overall decision.
As its value increases, it plays a higher role in the overall linear combination. Scaling the decision mask at each decision step with η𝐛​[𝐢]subscript𝜂𝐛delimited-[]𝐢\mathbf{\eta\_{b}[i]}, we propose the aggregate feature importance mask,
𝐌𝐚𝐠𝐠−𝐛,𝐣=∑i=1Ns​t​e​p​sη𝐛​[𝐢]​𝐌𝐛,𝐣​[𝐢]/∑j=1D∑i=1Ns​t​e​p​sη𝐛​[𝐢]​𝐌𝐛,𝐣​[𝐢].subscript𝐌

𝐚𝐠𝐠𝐛𝐣superscriptsubscript𝑖1subscript𝑁𝑠𝑡𝑒𝑝𝑠subscript𝜂𝐛delimited-[]𝐢subscript𝐌

𝐛𝐣delimited-[]𝐢superscriptsubscript𝑗1𝐷superscriptsubscript𝑖1subscript𝑁𝑠𝑡𝑒𝑝𝑠subscript𝜂𝐛delimited-[]𝐢subscript𝐌

𝐛𝐣delimited-[]𝐢\mathbf{M\_{agg-b,j}}\!=\!\sum\nolimits\_{i=1}^{N\_{steps}}\!\mathbf{\eta\_{b}[i]}\mathbf{M\_{b,j}[i]}\Big{/}\sum\nolimits\_{j=1}^{D}\!\sum\nolimits\_{i=1}^{N\_{steps}}\!\mathbf{\eta\_{b}[i]}\mathbf{M\_{b,j}[i]}.222Normalization is used to ensure ∑j=1D𝐌𝐚𝐠𝐠−𝐛,𝐣=1superscriptsubscript𝑗1𝐷subscript𝐌

𝐚𝐠𝐠𝐛𝐣1\sum\nolimits\_{j=1}^{D}\mathbf{M\_{agg-b,j}}=1.

Table 1: Mean and std. of test area under the receiving operating characteristic curve (AUC) on 6 synthetic datasets from (Chen et al. [2018](#bib.bib4)), for TabNet vs. other feature selection-based DNN models: No sel.: using all features without any feature selection, Global: using only globally-salient features, Tree Ensembles (Geurts, Ernst, and Wehenkel [2006](#bib.bib13)), Lasso-regularized model, L2X (Chen et al. [2018](#bib.bib4)) and INVASE (Yoon, Jordon, and van der Schaar [2019](#bib.bib54)). Bold numbers denote the best for each dataset.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Test AUC | | | | | |
| Syn1 | Syn2 | Syn3 | Syn4 | Syn5 | Syn6 |
| No selection | .578 ±plus-or-minus\pm .004 | .789 ±plus-or-minus\pm .003 | .854 ±plus-or-minus\pm .004 | .558 ±plus-or-minus\pm .021 | .662 ±plus-or-minus\pm .013 | .692 ±plus-or-minus\pm .015 |
| Tree | .574 ±plus-or-minus\pm .101 | .872 ±plus-or-minus\pm .003 | .899 ±plus-or-minus\pm .001 | .684 ±plus-or-minus\pm .017 | .741 ±plus-or-minus\pm .004 | .771 ±plus-or-minus\pm .031 |
| Lasso-regularized | .498 ±plus-or-minus\pm .006 | .555 ±plus-or-minus\pm .061 | .886 ±plus-or-minus\pm .003 | .512 ±plus-or-minus\pm .031 | .691 ±plus-or-minus\pm .024 | .727 ±plus-or-minus\pm .025 |
| L2X | .498 ±plus-or-minus\pm .005 | .823 ±plus-or-minus\pm .029 | .862 ±plus-or-minus\pm .009 | .678 ±plus-or-minus\pm .024 | .709 ±plus-or-minus\pm .008 | .827 ±plus-or-minus\pm .017 |
| INVASE | .690 ±plus-or-minus\pm .006 | .877 ±plus-or-minus\pm .003 | .902 ±plus-or-minus\pm .003 | .787 ±plus-or-minus\pm .004 | .784 ±plus-or-minus\pm .005 | .877 ±plus-or-minus\pm .003 |
| Global | .686 ±plus-or-minus\pm .005 | .873 ±plus-or-minus\pm .003 | .900 ±plus-or-minus\pm .003 | .774 ±plus-or-minus\pm .006 | .784 ±plus-or-minus\pm .005 | .858 ±plus-or-minus\pm .004 |
| TabNet | .682 ±plus-or-minus\pm .005 | .892 ±plus-or-minus\pm .004 | .897 ±plus-or-minus\pm .003 | .776 ±plus-or-minus\pm .017 | .789 ±plus-or-minus\pm .009 | .878 ±plus-or-minus\pm .004 |

Tabular self-supervised learning:
We propose a decoder architecture to reconstruct tabular features from the TabNet encoded representations. The decoder is composed of feature transformers, followed by FC layers at each decision step. The outputs are summed to obtain the reconstructed features.
We propose the task of prediction of missing feature columns from the others. Consider a binary mask 𝐒∈{0,1}B×D𝐒superscript01𝐵𝐷\mathbf{S}\in\{0,1\}^{B\times D}.
The TabNet encoder inputs (𝟏−𝐒)⋅𝐟^⋅1𝐒^𝐟(\mathbf{1}-\mathbf{S})\cdot\mathbf{\hat{f}} and the decoder outputs the reconstructed features, 𝐒⋅𝐟^⋅𝐒^𝐟\mathbf{S}\cdot\mathbf{\hat{f}}.
We initialize 𝐏​[𝟎]=(𝟏−𝐒)𝐏delimited-[]01𝐒\mathbf{P[0]}=(\mathbf{1}-\mathbf{S}) in encoder so that the model emphasizes merely on the known features, and the decoder’s last FC layer is multiplied with 𝐒𝐒\mathbf{S} to output the unknown features.
We consider the reconstruction loss in self-supervised phase:
∑b=1B∑j=1D|(𝐟^𝐛,𝐣−𝐟𝐛,𝐣)⋅𝐒𝐛,𝐣∑b=1B(𝐟𝐛,𝐣−1/B​∑b=1B𝐟𝐛,𝐣)2|2.superscriptsubscript𝑏1𝐵superscriptsubscript𝑗1𝐷superscript⋅subscript^𝐟

𝐛𝐣subscript𝐟

𝐛𝐣subscript𝐒

𝐛𝐣superscriptsubscript𝑏1𝐵superscriptsubscript𝐟

𝐛𝐣1𝐵superscriptsubscript𝑏1𝐵subscript𝐟

𝐛𝐣22\sum\_{b=1}^{B}\sum\_{j=1}^{D}\left|\frac{({\mathbf{\hat{f}\_{b,j}}}-\mathbf{f\_{b,j}})\cdot\mathbf{\mathbf{S\_{b,j}}}}{\sqrt{\sum\_{b=1}^{B}(\mathbf{f\_{b,j}}-{1/B}\sum\_{b=1}^{B}\mathbf{f\_{b,j}})^{2}}}\right|^{2}.
Normalization with the population standard deviation of the ground truth is beneficial, as the features may have different ranges. We sample 𝐒𝐛,𝐣subscript𝐒

𝐛𝐣\mathbf{S\_{b,j}} independently from a Bernoulli distribution with parameter pssubscript𝑝𝑠p\_{s}, at each iteration.

## Experiments

We study TabNet in wide range of problems, that contain regression or classification tasks, *particularly with published benchmarks*.
For all datasets, categorical inputs are mapped to a single-dimensional trainable scalar with a learnable embedding333In some cases, higher dimensional embeddings may slightly improve the performance, but interpretation of individual dimensions may become challenging. and numerical columns are input without and preprocessing.444Specially-designed feature engineering, e.g. logarithmic transformation of variables highly-skewed distributions, may further improve the results but we leave it out of the scope of this paper.
We use standard classification (softmax cross entropy) and regression (mean squared error) loss functions and we train until convergence. Hyperparameters of the TabNet models are optimized on a validation set and listed in Appendix. TabNet performance is not very sensitive to most hyperparameters as shown with ablation studies in Appendix. In Appendix, we also present ablation studies on various design and guidelines on selection of the key hyperparameters.
For all experiments we cite, we use the same training, validation and testing data split with the original work. Adam optimization algorithm (Kingma and Ba [2014](#bib.bib29)) and Glorot uniform initialization are used for training of all models.555An open-source implementation will be released.

### Instance-wise feature selection

Selection of the salient features is crucial for high performance, especially for small datasets. We consider 6 tabular datasets from (Chen et al. [2018](#bib.bib4)) (consisting 10k training samples). The datasets are constructed in such a way that only a subset of the features determine the output. For Syn1-Syn3, salient features are same for all instances (e.g., the output of Syn2 depends on features X3subscript𝑋3X\_{3}-X6subscript𝑋6X\_{6}), and global feature selection, as if the salient features were known, would give high performance. For Syn4-Syn6, salient features are instance dependent (e.g., for Syn4, the output depends on either X1subscript𝑋1X\_{1}-X2subscript𝑋2X\_{2} or X3subscript𝑋3X\_{3}-X6subscript𝑋6X\_{6} depending on the value of X11subscript𝑋11X\_{11}), which makes global feature selection suboptimal.
Table [1](#Sx3.T1 "Table 1 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning") shows that TabNet outperforms others (Tree Ensembles (Geurts, Ernst, and Wehenkel [2006](#bib.bib13)), LASSO regularization, L2X (Chen et al. [2018](#bib.bib4))) and is on par with INVASE (Yoon, Jordon, and van der Schaar [2019](#bib.bib54)). For Syn1-Syn3, TabNet performance is close to global feature selection - *it can figure out what features are globally important*. For Syn4-Syn6, eliminating instance-wise redundant features, TabNet improves global feature selection.
All other methods utilize a predictive model with 43k parameters, and the total number of parameters is 101k for INVASE due to the two other models in the actor-critic framework. TabNet is a single architecture, and its size is 26k for Syn1-Syn3 and 31k for Syn4-Syn6. The compact representation is one of TabNet’s valuable properties.

### Performance on real-world datasets

Table 2: Performance for Forest Cover Type dataset.

| Model | Test accuracy (%) |
| --- | --- |
| XGBoost | 89.34 |
| LightGBM | 89.28 |
| CatBoost | 85.14 |
| AutoML Tables | 94.95 |
| TabNet | 96.99 |

Forest Cover Type (Dua and Graff [2017](#bib.bib11)): The task is classification of forest cover type from cartographic variables. Table [2](#Sx4.T2 "Table 2 ‣ Performance on real-world datasets ‣ Experiments ‣ TabNet: Attentive Interpretable Tabular Learning") shows that TabNet outperforms ensemble tree based approaches that are known to achieve solid performance (Mitchell et al. [2018](#bib.bib34)).
We also consider AutoML Tables (AutoML [2019](#bib.bib2)), an automated search framework based on ensemble of models including DNN, gradient boosted DT, AdaNet (Cortes et al. [2016](#bib.bib7)) and ensembles (AutoML [2019](#bib.bib2)) with very thorough hyperparameter search. A single TabNet without fine-grained hyperparameter search outperforms it.

Table 3: Performance for Poker Hand induction dataset.

| Model | Test accuracy (%) |
| --- | --- |
| DT | 50.0 |
| MLP | 50.0 |
| Deep neural DT | 65.1 |
| XGBoost | 71.1 |
| LightGBM | 70.0 |
| CatBoost | 66.6 |
| TabNet | 99.2 |
| Rule-based | 100.0 |

Poker Hand (Dua and Graff [2017](#bib.bib11)): The task is classification of the poker hand from the raw suit and rank attributes of the cards. The input-output relationship is deterministic and hand-crafted rules can get 100% accuracy. Yet, conventional DNNs, DTs, and even their hybrid variant of deep neural DTs (Yang, Morillo, and Hospedales [2018](#bib.bib53)) severely suffer from the imbalanced data and cannot learn the required sorting and ranking operations (Yang, Morillo, and Hospedales [2018](#bib.bib53)). Tuned XGBoost, CatBoost, and LightGBM show very slight improvements over them. TabNet outperforms other methods, as it can perform highly-nonlinear processing with its depth, without overfitting thanks to instance-wise feature selection.

Table 4: Performance on Sarcos dataset. Three TabNet models of different sizes are considered.

| Model | Test MSE | Model size |
| --- | --- | --- |
| Random forest | 2.39 | 16.7K |
| Stochastic DT | 2.11 | 28K |
| MLP | 2.13 | 0.14M |
| Adaptive neural tree | 1.23 | 0.60M |
| Gradient boosted tree | 1.44 | 0.99M |
| TabNet-S | 1.25 | 6.3K |
| TabNet-M | 0.28 | 0.59M |
| TabNet-L | 0.14 | 1.75M |

Sarcos (Vijayakumar and Schaal [2000](#bib.bib49)): The task is regressing inverse dynamics of an anthropomorphic robot arm.
(Tanno et al. [2018](#bib.bib46)) shows that decent performance with a very small model is possible with a random forest.
In the very small model size regime, TabNet’s performance is on par with the best model from (Tanno et al. [2018](#bib.bib46)) with  100x more parameters.
When the model size is not constrained, TabNet achieves almost an order of magnitude lower test MSE.

Table 5: Performance on Higgs Boson dataset. Two TabNet models are denoted with -S and -M.

| Model | Test acc. (%) | Model size |
| --- | --- | --- |
| Sparse evolutionary MLP | 78.47 | 81K |
| Gradient boosted tree-S | 74.22 | 0.12M |
| Gradient boosted tree-M | 75.97 | 0.69M |
| MLP | 78.44 | 2.04M |
| Gradient boosted tree-L | 76.98 | 6.96M |
| TabNet-S | 78.25 | 81K |
| TabNet-M | 78.84 | 0.66M |

Higgs Boson (Dua and Graff [2017](#bib.bib11)): The task is distinguishing between a Higgs bosons process vs. background.
Due to its much larger size (10.5M instances), DNNs outperform DT variants even with very large ensembles.
TabNet outperforms MLPs with more compact representations.
We also compare to the state-of-the-art evolutionary sparsification algorithm (Mocanu et al. [2018](#bib.bib35)) that integrates non-structured sparsity into training.
With its compact representation, TabNet yields almost similar performance to sparse evolutionary training for the same number of parameters.
Unlike sparse evolutionary training, the sparsity of TabNet is structured – it does not degrade the operational intensity (Wen et al. [2016](#bib.bib51)) and can efficiently utilize modern multi-core processors.

Table 6: Performance for Rossmann Store Sales dataset.

| Model | Test MSE |
| --- | --- |
| MLP | 512.62 |
| XGBoost | 490.83 |
| LightGBM | 504.76 |
| CatBoost | 489.75 |
| TabNet | 485.12 |

Rossmann Store Sales (Kaggle [2019b](#bib.bib26)): The task is forecasting the store sales from static and time-varying features. We observe that TabNet outperforms commonly-used methods.
The time features (e.g. day) obtain high importance, and the benefit of instance-wise feature selection is observed for cases like holidays where the sales dynamics are different.

### Interpretability

!(/html/1908.07442/assets/x8.png)

Figure 5: Feature importance masks 𝐌​[𝐢]𝐌delimited-[]𝐢\mathbf{M[i]} (that indicate feature selection at it​hsuperscript𝑖𝑡ℎi^{th} step) and the aggregate feature importance mask 𝐌𝐚𝐠𝐠subscript𝐌𝐚𝐠𝐠\mathbf{M\_{agg}} showing the global instance-wise feature selection, on Syn2 and Syn4 (Chen et al. [2018](#bib.bib4)). Brighter colors show a higher value. E.g. for Syn2, only X3subscript𝑋3X\_{3}-X6subscript𝑋6X\_{6} are used.

.

Synthetic datasets: Fig. [5](#Sx4.F5 "Figure 5 ‣ Interpretability ‣ Experiments ‣ TabNet: Attentive Interpretable Tabular Learning") shows the aggregate feature importance masks for the synthetic datasets from Table [1](#Sx3.T1 "Table 1 ‣ TabNet for Tabular Learning ‣ TabNet: Attentive Interpretable Tabular Learning").666For better illustration here, the models are trained with 10M samples rather than 10K as we obtain sharper selection masks. The output on Syn2 only depends on X3subscript𝑋3X\_{3}-X6subscript𝑋6X\_{6} and we observe that the aggregate masks are almost all zero for irrelevant features and TabNet merely focuses on the relevant ones.
For Syn4, the output depends on either X1subscript𝑋1X\_{1}-X2subscript𝑋2X\_{2} or X3subscript𝑋3X\_{3}-X6subscript𝑋6X\_{6} depending on the value of X11subscript𝑋11X\_{11}.
TabNet yields accurate instance-wise feature selection – it allocates a mask to focus on the indicator X11subscript𝑋11X\_{11}, and assigns almost all-zero weights to irrelevant features (the ones other than two feature groups).

!(/html/1908.07442/assets/tsne_income.png)

Figure 6: First two dimensions of the T-SNE of the decision manifold for Adult and the impact of the top feature ‘Age’.

Real-world datasets: We first consider the simple task of mushroom edibility prediction (Dua and Graff [2017](#bib.bib11)). TabNet achieves 100% test accuracy on this dataset. It is indeed known (Dua and Graff [2017](#bib.bib11)) that “Odor” is the most discriminative feature – with “Odor” only, a model can get >98.5%absentpercent98.5>98.5\% test accuracy (Dua and Graff [2017](#bib.bib11)). Thus, a high feature importance is expected for it. TabNet assigns an importance score ratio of 43% for it, while other methods like LIME (Ribeiro, Singh, and Guestrin [2016](#bib.bib42)), Integrated Gradients (Sundararajan, Taly, and Yan [2017](#bib.bib45)) and DeepLift (Shrikumar, Greenside, and Kundaje [2017](#bib.bib44)) assign less than 30% (Ibrahim et al. [2019](#bib.bib24)).
Next, we consider Adult Census Income.
TabNet yields feature importance rankings consistent with the well-known (Lundberg, Erion, and Lee [2018](#bib.bib32); Nbviewer [2019](#bib.bib37)) (see Appendix)
For the same problem, Fig. [6](#Sx4.F6 "Figure 6 ‣ Interpretability ‣ Experiments ‣ TabNet: Attentive Interpretable Tabular Learning") shows the clear separation between age groups, as suggested by “Age” being the most important feature by TabNet.

!(/html/1908.07442/assets/convergence_ssl.png)

Figure 7: Training curves on Higgs dataset with 10k samples.

### Self-supervised learning

Table 7: Mean and std. of accuracy (over 15 runs) on Higgs with Tabnet-M model, varying the size of the training dataset for supervised fine-tuning.

| Training | Test accuracy (%) | |
| --- | --- | --- |
| dataset size | Supervised | With pre-training |
| 1k | 57.47 ±plus-or-minus\pm 1.78 | 61.37 ±plus-or-minus\pm 0.88 |
| 10k | 66.66 ±plus-or-minus\pm 0.88 | 68.06 ±plus-or-minus\pm 0.39 |
| 100k | 72.92 ±plus-or-minus\pm 0.21 | 73.19 ±plus-or-minus\pm 0.15 |

Table [7](#Sx4.T7 "Table 7 ‣ Self-supervised learning ‣ Experiments ‣ TabNet: Attentive Interpretable Tabular Learning") shows that unsupervised pre-training significantly improves performance on the supervised classification task, especially in the regime where the unlabeled dataset is much larger than the labeled dataset.
As exemplified in Fig. [7](#Sx4.F7 "Figure 7 ‣ Interpretability ‣ Experiments ‣ TabNet: Attentive Interpretable Tabular Learning") the model convergence is much faster with unsupervised pre-training. Very fast convergence can be useful for continual learning and domain adaptation.

## Conclusions

We have proposed TabNet, a novel deep learning architecture for tabular learning.
TabNet uses a sequential attention mechanism to choose a subset of semantically meaningful features to process at each decision step.
Instance-wise feature selection enables efficient learning as the model capacity is fully used for the most salient features, and also yields more interpretable decision making via visualization of selection masks.
We demonstrate that TabNet outperforms previous work across tabular datasets from different domains.
Lastly, we demonstrate significant benefits of unsupervised pre-training for fast adaptation and improved performance.

## Acknowledgements

Discussions with Jinsung Yoon, Kihyuk Sohn, Long T. Le, Ariel Kleiner, Zizhao Zhang, Andrei Kouznetsov, Chen Xing, Ryan Takasugi and Andrew Moore are gratefully acknowledged.

## Appendix A Performance on KDD datasets

Table 8: Performance on KDD datasets.

| Model | Test accuracy (%) | | | |
| --- | --- | --- | --- | --- |
|  | Appetency | Churn | Upselling | Census |
| XGBoost | 98.2 | 92.7 | 95.1 | 95.8 |
| CatBoost | 98.2 | 92.8 | 95.1 | 95.7 |
| TabNet | 98.2 | 92.7 | 95.0 | 95.5 |

Appetency, Churn and Upselling datasets are classification tasks for customer relationship management, and KDD Census Income (Dua and Graff [2017](#bib.bib11)) dataset is for income prediction from demographic and employment related variables.
These datasets show saturated behavior in performance (even simple models yield similar results).
Table [8](#A1.T8 "Table 8 ‣ Appendix A Performance on KDD datasets ‣ TabNet: Attentive Interpretable Tabular Learning") shows that TabNet achieves very similar or slightly worse performance than XGBoost and CatBoost, that are known to be robust as they contain high amount of ensembles.

## Appendix B Comparison of feature importance ranking of TabNet

Table 9: Importance ranking of features for Adult Census Income. TabNet yields feature importance rankings consistent with the well-known methods.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| *Feature* | *SHAP* | *Skater* | *XGBoost* | *TabNet* |
| Age | 1 | 1 | 1 | 1 |
| Capital gain | 3 | 3 | 4 | 6 |
| Capital loss | 9 | 9 | 6 | 4 |
| Education | 5 | 2 | 3 | 2 |
| Gender | 8 | 10 | 12 | 8 |
| Hours per week | 7 | 7 | 2 | 7 |
| Marital status | 2 | 8 | 10 | 9 |
| Native country | 11 | 11 | 9 | 12 |
| Occupation | 6 | 5 | 5 | 3 |
| Race | 12 | 12 | 11 | 11 |
| Relationship | 4 | 4 | 8 | 5 |
| Work class | 10 | 8 | 7 | 10 |

We observe the commonality of the most important features (“Age”, “Capital gain/loss”, “Education number”, “Relationship”) and the least important features (“Native country”, “Race”, “Gender”, “Work class”).

## Appendix C Self-supervised learning on Forest Cover Type

Table 10: Self-supervised tabular learning results. Mean and std. of accuracy (over 15 runs) on Forest Cover Type, varying the size of the training dataset for supervised fine-tuning.

| Training | Test accuracy (%) | |
| --- | --- | --- |
| dataset size | Supervised | With pre-training |
| 1k | 65.91 ±plus-or-minus\pm 1.02 | 67.86 ±plus-or-minus\pm 0.63 |
| 10k | 78.85 ±plus-or-minus\pm 1.24 | 79.22 ±plus-or-minus\pm 0.78 |

## Appendix D Experiment hyperparameters

For all datasets, we use a pre-defined hyperparameter search space. Ndsubscript𝑁𝑑N\_{d} and Nasubscript𝑁𝑎N\_{a} are chosen from {8,16,24,32,64,128}816243264128\{8,16,24,32,64,128\}, Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} is chosen from {3,4,5,6,7,8,9,10}345678910\{3,4,5,6,7,8,9,10\}, γ𝛾\gamma is chosen from {1.0,1.2,1.5,2.0}1.01.21.52.0\{1.0,1.2,1.5,2.0\}, λs​p​a​r​s​esubscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒\lambda\_{sparse} is chosen from {0,0.000001,0.0001,0.001,0.01,0.1}00.0000010.00010.0010.010.1\{0,0.000001,0.0001,0.001,0.01,0.1\}, B𝐵B is chosen from {256,512,1024,2048,4096,8192,16384,32768}25651210242048409681921638432768\{256,512,1024,2048,4096,8192,16384,32768\}, BVsubscript𝐵𝑉B\_{V} is chosen from {256,512,1024,2048,4096}256512102420484096\{256,512,1024,2048,4096\}, the learning rate is chosen from {0.005,0.01.0.02,0.025}0.0050.01.0.020.025\{0.005,0.01.0.02,0.025\}, the decay rate is chosen from {0.4,0.8,0.9,0.95}0.40.80.90.95\{0.4,0.8,0.9,0.95\} and the decay iterations is chosen from {0.5​k,2​k,8​k,10​k,20​k}0.5𝑘2𝑘8𝑘10𝑘20𝑘\{0.5k,2k,8k,10k,20k\}, and mBsubscript𝑚𝐵m\_{B} is chosen from {0.6,0.7,0.8,0.9,0.95,0.98}0.60.70.80.90.950.98\{0.6,0.7,0.8,0.9,0.95,0.98\}. If the model size is not under the desired cutoff, we decrease the value to satisfy the size constraint. For all the comparison models, we run a hyperparameter tuning with the same number of search steps.
  
Synthetic:
All TabNet models use Nd=Na=16subscript𝑁𝑑subscript𝑁𝑎16N\_{d}{=}N\_{a}{=}16, B=3000𝐵3000B{=}3000, BV=100subscript𝐵𝑉100B\_{V}{=}100, mB=0.7subscript𝑚𝐵0.7m\_{B}{=}0.7. For Syn1 we use λs​p​a​r​s​e=0.02subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.02\lambda\_{sparse}{=}0.02, Ns​t​e​p​s=4subscript𝑁𝑠𝑡𝑒𝑝𝑠4N\_{steps}{=}4 and γ=2.0𝛾2.0\gamma{=}2.0; for Syn2 and Syn3 we use λs​p​a​r​s​e=0.01subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.01\lambda\_{sparse}{=}0.01, Ns​t​e​p​s=4subscript𝑁𝑠𝑡𝑒𝑝𝑠4N\_{steps}{=}4 and γ=2.0𝛾2.0\gamma{=}2.0; and for Syn4, Syn5 and Syn6 we use λs​p​a​r​s​e=0.005subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.005\lambda\_{sparse}{=}0.005, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. All models use Adam with a learning rate of 0.02 (decayed 0.7 every 200 iterations with an exponential decay) for 4k iterations.
For visualizations, we also train TabNet models with datasets of size 10M samples. For this case, we choose Nd=Na=32subscript𝑁𝑑subscript𝑁𝑎32N\_{d}=N\_{a}=32, λs​p​a​r​s​e=0.001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.001\lambda\_{sparse}{=}0.001, B=10000𝐵10000B{=}10000, BV=100subscript𝐵𝑉100B\_{V}{=}100, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9. Adam is used with a learning rate of 0.02 (decayed 0.9 every 2k iterations with an exponential decay) for 15k iterations. For Syn2 and Syn3, Ns​t​e​p​s=4subscript𝑁𝑠𝑡𝑒𝑝𝑠4N\_{steps}{=}4 and γ=2𝛾2\gamma{=}2. For Syn4 and Syn6, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5.
  
Forest Cover Type:
The dataset partition details, and the hyperparameters of XGBoost, LigthGBM, and CatBoost are from (Mitchell et al. [2018](#bib.bib34)). We re-optimize AutoInt hyperparameters.
TabNet model uses Nd=Na=64subscript𝑁𝑑subscript𝑁𝑎64N\_{d}{=}N\_{a}{=}64, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}{=}0.0001, B=16384𝐵16384B{=}16384, BV=512subscript𝐵𝑉512B\_{V}{=}512, mB=0.7subscript𝑚𝐵0.7m\_{B}{=}0.7, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.02 (decayed 0.95 every 0.5k iterations with an exponential decay) for 130k iterations.
For unsupervised pre-training, the decoder model uses Nd=Na=64subscript𝑁𝑑subscript𝑁𝑎64N\_{d}{=}N\_{a}{=}64, B=16384𝐵16384B{=}16384, BV=512subscript𝐵𝑉512B\_{V}{=}512, mB=0.7subscript𝑚𝐵0.7m\_{B}{=}0.7, and Ns​t​e​p​s=10subscript𝑁𝑠𝑡𝑒𝑝𝑠10N\_{steps}{=}10. For supervised fine-tuning, we use the batch size B=BV𝐵subscript𝐵𝑉B{=}B\_{V} as the training datasets are small.
  
Poker Hands:
We split 6k samples for validation from the training dataset, and after optimization of the hyperparameters, we retrain with the entire training dataset. DT, MLP and deep neural DT models follow the same hyperparameters with (Yang, Morillo, and Hospedales [2018](#bib.bib53)). We tune the hyperparameters of XGBoost, LigthGBM, and CatBoost.
TabNet uses Nd=Na=16subscript𝑁𝑑subscript𝑁𝑎16N\_{d}{=}N\_{a}{=}16, λs​p​a​r​s​e=0.000001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.000001\lambda\_{sparse}{=}0.000001, B=4096𝐵4096B{=}4096, BV=1024subscript𝐵𝑉1024B\_{V}{=}1024, mB=0.95subscript𝑚𝐵0.95m\_{B}=0.95, Ns​t​e​p​s=4subscript𝑁𝑠𝑡𝑒𝑝𝑠4N\_{steps}{=}4 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.01 (decayed 0.95 every 500 iterations with an exponential decay) for 50k iterations.
  
Sarcos:
We split 4.5k samples for validation from the training dataset, and after optimization of the hyperparameters, we retrain with the entire training dataset. All comparison models follow the hyperparameters from (Tanno et al. [2018](#bib.bib46)).
TabNet-S model uses Nd=Na=8subscript𝑁𝑑subscript𝑁𝑎8N\_{d}{=}N\_{a}{=}8, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}{=}0.0001, B=4096𝐵4096B{=}4096, BV=256subscript𝐵𝑉256B\_{V}{=}256, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, Ns​t​e​p​s=3subscript𝑁𝑠𝑡𝑒𝑝𝑠3N\_{steps}{=}3 and γ=1.2𝛾1.2\gamma{=}1.2. Each feature transformer block uses one shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.01 (decayed 0.95 every 8k iterations with an exponential decay) for 600k iterations.
TabNet-M model uses Nd=Na=64subscript𝑁𝑑subscript𝑁𝑎64N\_{d}{=}N\_{a}{=}64, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}{=}0.0001, B=4096𝐵4096B{=}4096, BV=128subscript𝐵𝑉128B\_{V}{=}128, mB=0.8subscript𝑚𝐵0.8m\_{B}{=}0.8, Ns​t​e​p​s=7subscript𝑁𝑠𝑡𝑒𝑝𝑠7N\_{steps}{=}7 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.01 (decayed 0.95 every 8k iterations with an exponential decay) for 600k iterations.
The TabNet-L model uses Nd=Na=128subscript𝑁𝑑subscript𝑁𝑎128N\_{d}{=}N\_{a}{=}128, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}{=}0.0001, B=4096𝐵4096B{=}4096, BV=128subscript𝐵𝑉128B\_{V}{=}128, mB=0.8subscript𝑚𝐵0.8m\_{B}{=}0.8, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.02 (decayed 0.9 every 8k iterations with an exponential decay) for 600k iterations.
  
Higgs:
We split 500k samples for validation from the training dataset, and after optimization of the hyperparameters, we retrain with the entire training dataset. MLP models are from (Mocanu et al. [2018](#bib.bib35)). For gradient boosted trees (Tensorflow [2019](#bib.bib47)), we tune the learning rate and depth – the gradient boosted tree-S, -M, and -L models use 50, 300 and 3000 trees respectively.
TabNet-S model uses Nd=24subscript𝑁𝑑24N\_{d}{=}24, Na=26subscript𝑁𝑎26N\_{a}{=}26, λs​p​a​r​s​e=0.000001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.000001\lambda\_{sparse}{=}0.000001, B=16384𝐵16384B{=}16384, BV=512subscript𝐵𝑉512B\_{V}{=}512, mB=0.6subscript𝑚𝐵0.6m\_{B}{=}0.6, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.02 (decayed 0.9 every 20k iterations with an exponential decay) for 870k iterations.
TabNet-M model uses Nd=96subscript𝑁𝑑96N\_{d}{=}96, Na=32subscript𝑁𝑎32N\_{a}{=}32, λs​p​a​r​s​e=0.000001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.000001\lambda\_{sparse}{=}0.000001, B=8192𝐵8192B{=}8192, BV=256subscript𝐵𝑉256B\_{V}{=}256, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, Ns​t​e​p​s=8subscript𝑁𝑠𝑡𝑒𝑝𝑠8N\_{steps}{=}8 and γ=2.0𝛾2.0\gamma{=}2.0. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.025 (decayed 0.9 every 10k iterations with an exponential decay) for 370k iterations.
For unsupervised pre-training, the decoder model uses Nd=Na=128subscript𝑁𝑑subscript𝑁𝑎128N\_{d}{=}N\_{a}{=}128, B=8192𝐵8192B{=}8192, BV=256subscript𝐵𝑉256B\_{V}{=}256, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, and Ns​t​e​p​s=20subscript𝑁𝑠𝑡𝑒𝑝𝑠20N\_{steps}{=}20. For supervised fine-tuning, we use the batch size B=BV𝐵subscript𝐵𝑉B{=}B\_{V} as the training datasets are small.
  
Rossmann:
We use the same preprocessing and data split with (Catboost [2019](#bib.bib3)) – data from 2014 is used for training and validation, whereas 2015 is used for testing. We split 100k samples for validation from the training dataset, and after optimization of the hyperparameters, we retrain with the entire training dataset. The performance of the comparison models are from (Catboost [2019](#bib.bib3)). Obtained with hyperparameter tuning, the MLP is composed of 5 layers of FC (with a hidden unit size of 128), followed by BN and ReLU nonlinearity, trained with a batch size of 512 and a learning rate of 0.001.
TabNet model uses Nd=Na=32subscript𝑁𝑑subscript𝑁𝑎32N\_{d}{=}N\_{a}{=}32, λs​p​a​r​s​e=0.001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.001\lambda\_{sparse}{=}0.001, B=4096𝐵4096B{=}4096, BV=512subscript𝐵𝑉512B\_{V}{=}512, mB=0.8subscript𝑚𝐵0.8m\_{B}{=}0.8, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.2𝛾1.2\gamma{=}1.2. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.002 (decayed 0.95 every 2000 iterations with an exponential decay) for 15k iterations.
  
KDD:
For Appetency, Churn and Upselling datasets, we apply the similar preprocessing and split as (Prokhorenkova et al. [2018](#bib.bib39)). The performance of the comparison models are from (Prokhorenkova et al. [2018](#bib.bib39)). TabNet models use Nd=Na=32subscript𝑁𝑑subscript𝑁𝑎32N\_{d}{=}N\_{a}{=}32, λs​p​a​r​s​e=0.001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.001\lambda\_{sparse}{=}0.001, B=8192𝐵8192B{=}8192, BV=256subscript𝐵𝑉256B\_{V}{=}256, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, Ns​t​e​p​s=7subscript𝑁𝑠𝑡𝑒𝑝𝑠7N\_{steps}{=}7 and γ=1.2𝛾1.2\gamma{=}1.2. Each feature transformer block uses two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.01 (decayed 0.9 every 1000 iterations with an exponential decay) for 10k iterations.
For Census Income, the dataset and comparison model specifications follow (Oza [2005](#bib.bib38)). TabNet model uses Nd=Na=48subscript𝑁𝑑subscript𝑁𝑎48N\_{d}{=}N\_{a}{=}48, λs​p​a​r​s​e=0.001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.001\lambda\_{sparse}{=}0.001, B=8192𝐵8192B{=}8192, BV=256subscript𝐵𝑉256B\_{V}{=}256, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.02 (decayed 0.7 every 2000 iterations with an exponential decay) for 4k iterations.
  
Mushroom edibility:
TabNet model uses Nd=Na=8subscript𝑁𝑑subscript𝑁𝑎8N\_{d}{=}N\_{a}{=}8, λs​p​a​r​s​e=0.001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.001\lambda\_{sparse}{=}0.001, B=2048𝐵2048B{=}2048, BV=128subscript𝐵𝑉128B\_{V}{=}128, mB=0.9subscript𝑚𝐵0.9m\_{B}{=}0.9, Ns​t​e​p​s=3subscript𝑁𝑠𝑡𝑒𝑝𝑠3N\_{steps}{=}3 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent FC layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.01 (decayed 0.8 every 400 iterations with an exponential decay) for 10k iterations.
  
Adult Census Income:
TabNet model uses Nd=Na=16subscript𝑁𝑑subscript𝑁𝑎16N\_{d}{=}N\_{a}{=}16, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}=0.0001, B=4096𝐵4096B{=}4096, BV=128subscript𝐵𝑉128B\_{V}{=}128, mB=0.98subscript𝑚𝐵0.98m\_{B}{=}0.98, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}{=}5 and γ=1.5𝛾1.5\gamma{=}1.5. Feature transformers use two shared and two decision step-dependent layer, ghost BN and GLU blocks. Adam is used with a learning rate of 0.02 (decayed 0.4 every 2.5k iterations with an exponential decay) for 7.7k iterations. 85.7% test accuracy is achieved.

## Appendix E Ablation studies

Table 11: Ablation studies for the TabNet encoder model for the forest cover type dataset.

| Ablation cases | Test accuracy % (difference) | Model size |
| --- | --- | --- |
| Base (Nd=Na=64subscript𝑁𝑑subscript𝑁𝑎64N\_{d}=N\_{a}=64, γ=1.5𝛾1.5\gamma=1.5, Ns​t​e​p​s=5subscript𝑁𝑠𝑡𝑒𝑝𝑠5N\_{steps}=5, λs​p​a​r​s​e=0.0001subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.0001\lambda\_{sparse}=0.0001, feature transformer block composed of two shared and two decision step-dependent layers, B=16384𝐵16384B=16384) | 96.99 | 470k |
| Decreasing capacity via number of units (with Nd=Na=32subscript𝑁𝑑subscript𝑁𝑎32N\_{d}=N\_{a}=32) | 94.99 (-2.00) | 129k |
| Decreasing capacity via number of decision steps (with Ns​t​e​p​s=3subscript𝑁𝑠𝑡𝑒𝑝𝑠3N\_{steps}=3) | 96.22 (-0.77) | 328k |
| Increasing capacity via number of decision steps (with Ns​t​e​p​s=9subscript𝑁𝑠𝑡𝑒𝑝𝑠9N\_{steps}=9) | 95.48 (-1.51) | 755k |
| Decreasing capacity via all-shared feature transformer blocks | 96.74 (-0.25) | 143k |
| Increasing capacity via decision step-dependent feature transformer blocks | 96.76 (-0.23) | 703k |
| Feature transformer block as a single shared layer | 95.32 (-1.67) | 35k |
| Feature transformer block as a single shared layer, with ReLU instead of GLU | 93.92 (-3.07) | 27k |
| Feature transformer block as two shared layers | 96.34 (-0.66) | 71k |
| Feature transformer block as two shared layers and 1 decision step-dependent layer | 96.54 (-0.45) | 271k |
| Feature transformer block as a single decision-step dependent layer | 94.71 (-0.28) | 105k |
| Feature transformer block as a single decision-step dependent layer, with Nd=Na=128subscript𝑁𝑑subscript𝑁𝑎128N\_{d}{=}N\_{a}{=}128 | 96.24 (-0.75) | 208k |
| Feature transformer block as a single decision-step dependent layer, with Nd=Na=128subscript𝑁𝑑subscript𝑁𝑎128N\_{d}{=}N\_{a}{=}128 and replacing GLU with ReLU | 95.67 (-1.32) | 139k |
| Feature transformer block as a single decision-step dependent layer, with Nd=Na=256subscript𝑁𝑑subscript𝑁𝑎256N\_{d}{=}N\_{a}{=}256 and replacing GLU with ReLU | 96.41 (-0.58) | 278k |
| Reducing the impact of prior scale (with γ=3.0𝛾3.0\gamma=3.0) | 96.49 (-0.50) | 470k |
| Increasing the impact of prior scale (with γ=1.0𝛾1.0\gamma=1.0) | 96.67 (-0.32) | 470k |
| No sparsity regularization (with λs​p​a​r​s​e=0subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0\lambda\_{sparse}=0) | 96.50 (-0.49) | 470k |
| High sparsity regularization (with λs​p​a​r​s​e=0.01subscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒0.01\lambda\_{sparse}=0.01) | 93.87 (-3.12) | 470k |
| Small batch size (B=4096𝐵4096B=4096) | 96.42 (-0.57) | 470k |

Table [11](#A5.T11 "Table 11 ‣ Appendix E Ablation studies ‣ TabNet: Attentive Interpretable Tabular Learning") shows the impact of ablation cases. For all cases, the number of iterations is optimized on the validation set.

Obtaining high performance necessitates appropriately-adjusted model capacity based on the characteristics of the dataset. Decreasing the number of units Ndsubscript𝑁𝑑N\_{d}, Nasubscript𝑁𝑎N\_{a} or the number of decision steps Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} are efficient ways of gradually decreasing the capacity without significant degradation in performance. On the other hand, increasing these parameters beyond some value causes optimization issues and do not yield performance benefits.
Replacing the feature transformer block with a simpler alternative, such as a single shared layer, can still give strong performance while yielding a very compact model architecture. This shows the importance of the inductive bias introduced with feature selection and sequential attention.
To push the performance, increasing the depth of the feature transformer is an effective approach. While increasing the depth, parameter sharing between feature transformer blocks across decision steps is an efficient way to decrease model size without degradation in performance. We indeed observe the benefit of partial parameter sharing, compared to fully decision step-dependent blocks or fully shared blocks. We also observe the empirical benefit of GLU, compared to conventional nonlinearities like ReLU.

The strength of sparse feature selection depends on the two parameters we introduce: γ𝛾\gamma and λs​p​a​r​s​esubscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒\lambda\_{sparse}. We show that optimal choice of these two is important for performance. A γ𝛾\gamma close to 1, or a high λs​p​a​r​s​esubscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒\lambda\_{sparse} may yield too tight constraints on the strength of sparsity and may hurt performance. On the other hand, there is still the benefit of a sufficient low γ𝛾\gamma and sufficiently high λs​p​a​r​s​esubscript𝜆𝑠𝑝𝑎𝑟𝑠𝑒\lambda\_{sparse}, to aid learning of the model via a favorable inductive bias.

Lastly, given the fixed model architecture, we show the benefit of large-batch training, enabled by ghost BN (Hoffer, Hubara, and Soudry [2017](#bib.bib21)). The optimal batch size for TabNet seems considerably higher than the conventional batch sizes used for other data types, such as images or speech.

## Appendix F Guidelines for hyperparameters

We consider datasets ranging from ∼similar-to\sim10K to ∼similar-to\sim10M samples, with varying degrees of fitting difficulty. TabNet obtains high performance on all with a few general principles on hyperparameters:

* •

  For most datasets, Ns​t​e​p​s∈[3,10]subscript𝑁𝑠𝑡𝑒𝑝𝑠310N\_{steps}\in[3,10] is optimal. Typically, when there are more information-bearing features, the optimal value of Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} tends to be higher. On the other hand, increasing it beyond some value may adversely affect training dynamics as some paths in the network becomes deeper and there are more potentially-problematic ill-conditioned matrices. A very high value of Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} may suffer from overfitting and yield poor generalization.
* •

  Adjustment of Ndsubscript𝑁𝑑N\_{d} and Nasubscript𝑁𝑎N\_{a} is an efficient way of obtaining a trade-off between performance and complexity. Nd=Nasubscript𝑁𝑑subscript𝑁𝑎N\_{d}=N\_{a} is a reasonable choice for most datasets. A very high value of Ndsubscript𝑁𝑑N\_{d} and Nasubscript𝑁𝑎N\_{a} may suffer from overfitting and yield poor generalization.
* •

  An optimal choice of γ𝛾\gamma can have a major role on the performance. Typically a larger Ns​t​e​p​ssubscript𝑁𝑠𝑡𝑒𝑝𝑠N\_{steps} value favors for a larger γ𝛾\gamma.
* •

  A large batch size is beneficial – if the memory constraints permit, as large as 1-10 % of the total training dataset size can help performance. The virtual batch size is typically much smaller.
* •

  Initially large learning rate is important, which should be gradually decayed until convergence.

## References

* Amodei et al. (2015)

  Amodei, D.; Anubhai, R.; Battenberg, E.; Case, C.; Casper, J.; et al. 2015.
  Deep Speech 2: End-to-End Speech Recognition in English and Mandarin.
  *arXiv:1512.02595* .
* AutoML (2019)

  AutoML. 2019.
  AutoML Tables – Google Cloud.
  URL https://cloud.google.com/automl-tables/.
* Catboost (2019)

  Catboost. 2019.
  Benchmarks.
  https://github.com/catboost/benchmarks.
  Accessed: 2019-11-10.
* Chen et al. (2018)

  Chen, J.; Song, L.; Wainwright, M. J.; and Jordan, M. I. 2018.
  Learning to Explain: An Information-Theoretic Perspective on Model
  Interpretation.
  *arXiv:1802.07814* .
* Chen and Guestrin (2016)

  Chen, T.; and Guestrin, C. 2016.
  XGBoost: A Scalable Tree Boosting System.
  In *KDD*.
* Chui et al. (2018)

  Chui, M.; Manyika, J.; Miremadi, M.; Henke, N.; Chung, R.; et al. 2018.
  Notes from the AI Frontier.
  *McKinsey Global Institute* .
* Cortes et al. (2016)

  Cortes, C.; Gonzalvo, X.; Kuznetsov, V.; Mohri, M.; and Yang, S. 2016.
  AdaNet: Adaptive Structural Learning of Artificial Neural Networks.
  *arXiv:1607.01097* .
* Dai et al. (2017)

  Dai, Z.; Yang, Z.; Yang, F.; Cohen, W. W.; and Salakhutdinov, R. 2017.
  Good Semi-supervised Learning that Requires a Bad GAN.
  *arxiv:1705.09783* .
* Dauphin et al. (2016)

  Dauphin, Y. N.; Fan, A.; Auli, M.; and Grangier, D. 2016.
  Language Modeling with Gated Convolutional Networks.
  *arXiv:1612.08083* .
* Devlin et al. (2018)

  Devlin, J.; Chang, M.; Lee, K.; and Toutanova, K. 2018.
  BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding.
  *arXiv:1810.04805* .
* Dua and Graff (2017)

  Dua, D.; and Graff, C. 2017.
  UCI Machine Learning Repository.
  URL http://archive.ics.uci.edu/ml.
* Gehring et al. (2017)

  Gehring, J.; Auli, M.; Grangier, D.; Yarats, D.; and Dauphin, Y. N. 2017.
  Convolutional Sequence to Sequence Learning.
  *arXiv:1705.03122* .
* Geurts, Ernst, and Wehenkel (2006)

  Geurts, P.; Ernst, D.; and Wehenkel, L. 2006.
  Extremely randomized trees.
  *Machine Learning* 63(1): 3–42.
  ISSN 1573-0565.
* Goodfellow, Bengio, and Courville (2016)

  Goodfellow, I.; Bengio, Y.; and Courville, A. 2016.
  *Deep Learning*.
  MIT Press.
* Grabczewski and Jankowski (2005)

  Grabczewski, K.; and Jankowski, N. 2005.
  Feature selection with decision tree criterion.
  In *HIS*.
* Grandvalet and Bengio (2004)

  Grandvalet, Y.; and Bengio, Y. 2004.
  Semi-supervised Learning by Entropy Minimization.
  In *NIPS*.
* Guyon and Elisseeff (2003)

  Guyon, I.; and Elisseeff, A. 2003.
  An Introduction to Variable and Feature Selection.
  *JMLR* 3: 1157–1182.
* He et al. (2015)

  He, K.; Zhang, X.; Ren, S.; and Sun, J. 2015.
  Deep Residual Learning for Image Recognition.
  *arXiv:1512.03385* .
* Hestness et al. (2017)

  Hestness, J.; Narang, S.; Ardalani, N.; Diamos, G. F.; Jun, H.; Kianinejad, H.;
  Patwary, M. M. A.; Yang, Y.; and Zhou, Y. 2017.
  Deep Learning Scaling is Predictable, Empirically.
  *arXiv:1712.00409* .
* Ho (1998)

  Ho, T. K. 1998.
  The random subspace method for constructing decision forests.
  *PAMI* 20(8): 832–844.
* Hoffer, Hubara, and Soudry (2017)

  Hoffer, E.; Hubara, I.; and Soudry, D. 2017.
  Train longer, generalize better: closing the generalization gap in
  large batch training of neural networks.
  *arXiv:1705.08741* .
* Hudson and Manning (2018)

  Hudson, D. A.; and Manning, C. D. 2018.
  Compositional Attention Networks for Machine Reasoning.
  *arXiv:1803.03067* .
* Humbird, Peterson, and McClarren (2018)

  Humbird, K. D.; Peterson, J. L.; and McClarren, R. G. 2018.
  Deep Neural Network Initialization With Decision Trees.
  *IEEE Trans Neural Networks and Learning Systems* .
* Ibrahim et al. (2019)

  Ibrahim, M.; Louie, M.; Modarres, C.; and Paisley, J. W. 2019.
  Global Explanations of Neural Networks: Mapping the Landscape of
  Predictions.
  *arxiv:1902.02384* .
* Kaggle (2019a)

  Kaggle. 2019a.
  Historical Data Science Trends on Kaggle.
  https://www.kaggle.com/shivamb/data-science-trends-on-kaggle.
  Accessed: 2019-04-20.
* Kaggle (2019b)

  Kaggle. 2019b.
  Rossmann Store Sales.
  https://www.kaggle.com/c/rossmann-store-sales.
  Accessed: 2019-11-10.
* Ke et al. (2017)

  Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; et al. 2017.
  LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  In *NIPS*.
* Ke et al. (2019)

  Ke, G.; Zhang, J.; Xu, Z.; Bian, J.; and Liu, T.-Y. 2019.
  TabNN: A Universal Neural Network Solution for Tabular Data.
  URL https://openreview.net/forum?id=r1eJssCqY7.
* Kingma and Ba (2014)

  Kingma, D. P.; and Ba, J. 2014.
  Adam: A Method for Stochastic Optimization.
  In *ICLR*.
* Kontschieder et al. (2015)

  Kontschieder, P.; Fiterau, M.; Criminisi, A.; and Bulò, S. R. 2015.
  Deep Neural Decision Forests.
  In *ICCV*.
* Lai et al. (2015)

  Lai, S.; Xu, L.; Liu, K.; and Zhao, J. 2015.
  Recurrent Convolutional Neural Networks for Text Classification.
  In *AAAI*.
* Lundberg, Erion, and Lee (2018)

  Lundberg, S. M.; Erion, G. G.; and Lee, S. 2018.
  Consistent Individualized Feature Attribution for Tree Ensembles.
  *arXiv:1802.03888* .
* Martins and Astudillo (2016)

  Martins, A. F. T.; and Astudillo, R. F. 2016.
  From Softmax to Sparsemax: A Sparse Model of Attention and
  Multi-Label Classification.
  *arXiv:1602.02068* .
* Mitchell et al. (2018)

  Mitchell, R.; Adinets, A.; Rao, T.; and Frank, E. 2018.
  XGBoost: Scalable GPU Accelerated Learning.
  *arXiv:1806.11248* .
* Mocanu et al. (2018)

  Mocanu, D.; Mocanu, E.; Stone, P.; Nguyen, P.; Gibescu, M.; and Liotta, A.
  2018.
  Scalable training of artificial neural networks with adaptive sparse
  connectivity inspired by network science.
  *Nature Communications* 9.
* Mott et al. (2019)

  Mott, A.; Zoran, D.; Chrzanowski, M.; Wierstra, D.; and Rezende, D. J. 2019.
  S3TA: A Soft, Spatial, Sequential, Top-Down Attention Model.
  URL https://openreview.net/forum?id=B1gJOoRcYQ.
* Nbviewer (2019)

  Nbviewer. 2019.
  Notebook on Nbviewer.
  URL https://nbviewer.jupyter.org/github/dipanjanS/data˙science˙for˙all/blob/master/tds˙model˙interpretation˙xai/Human-interpretableMachineLearning-DS.ipynb#.
* Oza (2005)

  Oza, N. C. 2005.
  Online bagging and boosting.
  In *IEEE Trans Conference on Systems, Man and Cybernetics*.
* Prokhorenkova et al. (2018)

  Prokhorenkova, L.; Gusev, G.; Vorobev, A.; Dorogush, A. V.; and Gulin, A. 2018.
  CatBoost: unbiased boosting with categorical features.
  In *NIPS*.
* Radford, Metz, and Chintala (2015)

  Radford, A.; Metz, L.; and Chintala, S. 2015.
  Unsupervised Representation Learning with Deep Convolutional
  Generative Adversarial Networks.
  *arXiv:1511.06434* .
* Raina et al. (2007)

  Raina, R.; Battle, A.; Lee, H.; Packer, B.; and Ng, A. Y. 2007.
  Self-Taught Learning: Transfer Learning from Unlabeled Data.
  In *ICML*.
* Ribeiro, Singh, and Guestrin (2016)

  Ribeiro, M.; Singh, S.; and Guestrin, C. 2016.
  “Why Should I Trust You?”: Explaining the Predictions of Any
  Classifier.
  In *KDD*.
* Shavitt and Segal (2018)

  Shavitt, I.; and Segal, E. 2018.
  Regularization Learning Networks: Deep Learning for Tabular Datasets.
* Shrikumar, Greenside, and Kundaje (2017)

  Shrikumar, A.; Greenside, P.; and Kundaje, A. 2017.
  Learning Important Features Through Propagating Activation
  Differences.
  *arXiv:1704.02685* .
* Sundararajan, Taly, and Yan (2017)

  Sundararajan, M.; Taly, A.; and Yan, Q. 2017.
  Axiomatic Attribution for Deep Networks.
  *arXiv:1703.01365* .
* Tanno et al. (2018)

  Tanno, R.; Arulkumaran, K.; Alexander, D. C.; Criminisi, A.; and Nori, A. V.
  2018.
  Adaptive Neural Trees.
  *arXiv:1807.06699* .
* Tensorflow (2019)

  Tensorflow. 2019.
  Classifying Higgs boson processes in the HIGGS Data Set.
  URL https://github.com/tensorflow/models/tree/master/official/boosted˙trees.
* Trinh, Luong, and Le (2019)

  Trinh, T. H.; Luong, M.; and Le, Q. V. 2019.
  Selfie: Self-supervised Pretraining for Image Embedding.
  *arXiv:1906.02940* .
* Vijayakumar and Schaal (2000)

  Vijayakumar, S.; and Schaal, S. 2000.
  Locally Weighted Projection Regression: An O(n) Algorithm for
  Incremental Real Time Learning in High Dimensional Space.
  In *ICML*.
* Wang, Aggarwal, and Liu (2017)

  Wang, S.; Aggarwal, C.; and Liu, H. 2017.
  Using a random forest to inspire a neural network and improving on
  it.
  In *SDM*.
* Wen et al. (2016)

  Wen, W.; Wu, C.; Wang, Y.; Chen, Y.; and Li, H. 2016.
  Learning Structured Sparsity in Deep Neural Networks.
  *arXiv:1608.03665* .
* Xu et al. (2019)

  Xu, L.; Skoularidou, M.; Cuesta-Infante, A.; and Veeramachaneni, K. 2019.
  Modeling Tabular data using Conditional GAN.
  *arXiv:1907.00503* .
* Yang, Morillo, and Hospedales (2018)

  Yang, Y.; Morillo, I. G.; and Hospedales, T. M. 2018.
  Deep Neural Decision Trees.
  *arXiv:1806.06988* .
* Yoon, Jordon, and van der Schaar (2019)

  Yoon, J.; Jordon, J.; and van der Schaar, M. 2019.
  INVASE: Instance-wise Variable Selection using Neural Networks.
  In *ICLR*.
