---
arxiv: '2309.17130'
authors:
- Sascha Marton
- Stefan Lüdtke
- Christian Bartelt
- Heiner Stuckenschmidt
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'GRANDE: Gradient-Based Decision Tree Ensembles for Tabular Data'
url: http://arxiv.org/abs/2309.17130v3
year: 2023
---

[2309.17130] GRANDE: Gradient-Based Decision Tree Ensembles














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



# GRANDE: Gradient-Based Decision Tree Ensembles

Sascha Marton
  
University of Mannheim
  
68131 Mannheim
  
sascha.marton@uni-mannheim.de
  
&Stefan Lüdtke
  
University of Rostock
  
18059 Rostock
  
stefan.luedtke@uni-rostock.de
  
&Christian Bartelt
  
University of Mannheim
  
68131 Mannheim
  
christian.bartelt@uni-mannheim.de
  
&Heiner Stuckenschmidt
  
University of Mannheim
  
68131 Mannheim
  
heiner.stuckenschmidt@uni-mannheim.de

###### Abstract

Despite the success of deep learning for text and image data, tree-based ensemble models are still state-of-the-art for machine learning with heterogeneous tabular data. However, there is a significant need for tabular-specific gradient-based methods due to their high flexibility. In this paper, we propose GRANDE, GRAdieNt-Based Decision Tree Ensembles, a novel approach for learning hard, axis-aligned decision tree ensembles using end-to-end gradient descent. GRANDE is based on a dense representation of tree ensembles, which affords to use backpropagation with a straight-through operator to jointly optimize all model parameters. Our method combines axis-aligned splits, which is a useful inductive bias for tabular data, with the flexibility of gradient-based optimization. Furthermore, we introduce an advanced instance-wise weighting that facilitates learning representations for both, simple and complex relations, within a single model. We conducted an extensive evaluation on a predefined benchmark with 19 classification datasets and demonstrate that our method outperforms existing gradient-boosting and deep learning frameworks on most datasets. The method is available under: <https://github.com/s-marton/GRANDE>

## 1 Introduction

Heterogeneous tabular data is the most frequently used form of data (Chui et al., [2018](#bib.bib11); Shwartz-Ziv & Armon, [2022](#bib.bib42)) and is indispensable in a wide range of applications such as medical diagnosis (Ulmer et al., [2020](#bib.bib45); Somani et al., [2021](#bib.bib43)), estimation of creditworthiness (Clements et al., [2020](#bib.bib12)) and fraud detection (Cartella et al., [2021](#bib.bib7)).
Therefore, enhancing the predictive performance and robustness of models can bring significant advantages to users and companies (Borisov et al., [2022](#bib.bib5)).
However, tabular data comes with considerable challenges like noise, missing values, class imbalance, and a combination of different feature types, especially categorical and numerical data.
Despite the success of deep learning in various domains, recent studies indicate that tabular data still poses a major challenge and tree-based models like XGBoost and CatBoost outperform them in most cases (Borisov et al., [2022](#bib.bib5); Grinsztajn et al., [2022](#bib.bib16); Shwartz-Ziv & Armon, [2022](#bib.bib42)).
At the same time, employing end-to-end gradient-based training provides several advantages over traditional machine learning methods (Borisov et al., [2022](#bib.bib5)). They offer a high level of flexibility by allowing an easy integration of arbitrary, differentiable loss functions tailored towards specific problems and support iterative training (Sahoo et al., [2017](#bib.bib41)). Moreover, gradient-based methods can be incorporated easily into multimodal learning, with tabular data being one of several input types (Lichtenwalter et al., [2021](#bib.bib28); Pölsterl et al., [2021](#bib.bib34)). Therefore, creating tabular-specific, gradient-based methods is a very active field of research and the need for well-performing methods is intense (Grinsztajn et al., [2022](#bib.bib16)).

Recently, Marton et al. ([2023](#bib.bib32)) introduced GradTree, a novel approach that uses gradient descent to learn hard, axis-aligned decision trees (DTs). This is achieved by reformulating DTs to a dense representation and jointly optimizing all tree parameters using backpropagation with a straight-through (ST) operator. Learning hard, axis-aligned DTs with gradient descent allows combining the advantageous inductive bias of tree-based methods with the flexibility of a gradient-based optimization.
In this paper, we propose GRANDE, GRAdieNt-Based Decision Tree Ensembles, a novel approach for learning decision tree ensembles using end-to-end gradient descent. Similar to Marton et al. ([2023](#bib.bib32)), we use a dense representation for split nodes and the ST operator to deal with the non-differentiable nature of DTs.
We build upon their approach, transitioning from individual trees to a weighted tree ensemble, while maintaining an efficient computation.
As a result, GRANDE holds a significant advantage over existing gradient-based methods. Typically, deep learning methods are biased towards smooth solutions (Rahaman et al., [2019](#bib.bib37)). As the target function in tabular datasets is usually not smooth, deep learning methods struggle to find these irregular functions. In contrast, models that are based on hard, axis aligned DTs learn piece-wise constant functions and therefore do not show such a bias (Grinsztajn et al., [2022](#bib.bib16)). This important advantage is one inherent aspect of GRANDE, as it utilizes hard, axis-aligned DTs.
This is a major difference to existing deep learning methods for hierarchical representations like NODE, where soft and oblique splits are used (Popov et al., [2019](#bib.bib35)).
Furthermore, we introduce instance-wise weighting in GRANDE. This allows learning appropriate representations for simple and complex rules within a single model, which increases the performance of the ensemble. Furthermore, we show that our instance-wise weighting has a positive impact on the local interpretability relative to other state-of-the-art methods.

More specifically, our contributions are as follows:

* •

  We extend GradTree (Marton et al., [2023](#bib.bib32)) from individual trees to an end-to-end gradient-based tree ensemble, maintaining efficient computation (Section [3.1](#S3.SS1 "3.1 From Decision Trees to Weighted Tree Ensembles ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).
* •

  We introduce softsign as a differentiable split function and show the advantage over commonly used alternatives (Section [3.2](#S3.SS2 "3.2 Differentiable Split Functions ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).
* •

  We propose a novel weighting technique that emphasizes instance-wise estimator importance (Section [3.3](#S3.SS3 "3.3 Instance-Wise Estimator Weights ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).

We conduct an extensive evaluation on 19 binary classification tasks (Section [4](#S4 "4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles")) based on the predefined tabular benchmark proposed by Bischl et al. ([2021](#bib.bib4)). GRANDE outperforms existing methods for both, default and optimized hyperparameters. The performance difference to other methods is substantial on several datasets, making GRANDE an important extension to the existing repertoire of tabular data methods.

## 2 Background: Gradient-Based Decision Trees

GRANDE builds on gradient-based decision trees (GradTree) at the level of individual trees in the ensemble. Hence, we summarize the relevant aspects and notation of GradTree in this section and refer to Marton et al. ([2023](#bib.bib32)) for a complete overview.

Traditionally, DTs involve nested concatenation of rules. In GradTree, DTs are formulated as arithmetic functions based on addition and multiplication to facilitate gradient-based learning. Thereby both, GradTree and GRANDE focus on learning fully-grown (i.e., complete, full) DTs which can be pruned post-hoc.
A DT of depth d𝑑d is formulated with respect to its parameters as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | t​(𝒙|𝝀,𝝉,𝜾)=∑l=02d−1λl​𝕃​(𝒙|l,𝝉,𝜾)𝑡conditional𝒙  𝝀𝝉𝜾superscriptsubscript𝑙0superscript2𝑑1subscript𝜆𝑙𝕃conditional𝒙  𝑙𝝉𝜾t(\bm{x}|\bm{\lambda},\bm{\tau},\bm{\iota})=\sum\_{l=0}^{2^{d}-1}\lambda\_{l}\,\mathbb{L}(\bm{x}|l,\bm{\tau},\bm{\iota}) |  | (1) |

where 𝕃𝕃\mathbb{L} is a function that indicates whether a sample 𝒙∈ℝn𝒙superscriptℝ𝑛\bm{x}\in\mathbb{R}^{n} belongs to a leaf l𝑙l, 𝝀∈𝒞2d𝝀superscript𝒞superscript2𝑑\bm{\lambda}\in\mathcal{C}^{2^{d}} denotes class membership for each leaf node, 𝝉∈ℝ2d−1𝝉superscriptℝsuperscript2𝑑1\bm{\tau}\in\mathbb{R}^{2^{d}-1} represents split thresholds and 𝜾∈ℕ2d−1𝜾superscriptℕsuperscript2𝑑1\bm{\iota}\in\mathbb{N}^{2^{d}-1} the feature index for each internal node.

To support a gradient-based optimization and ensure an efficient computation via matrix operations, a novel dense DT representation is introduced in GradTree. Traditionally, the feature index vector 𝜾𝜾\bm{\iota} is one-dimensional, but GradTree expands it into a matrix form. Specifically, this representation one-hot encodes the feature index, converting 𝜾∈ℝ2d−1𝜾superscriptℝsuperscript2𝑑1\bm{\iota}\in\mathbb{R}^{2^{d}-1} into a matrix 𝑰∈ℝ2d−1×ℝn𝑰superscriptℝsuperscript2𝑑1superscriptℝ𝑛\bm{I}\in\mathbb{R}^{2^{d}-1}\times\mathbb{R}^{n}. Similarly, for split thresholds, instead of a single value for all features, individual values for each feature are stored, leading to a matrix representation 𝑻∈ℝ2d−1×ℝn𝑻superscriptℝsuperscript2𝑑1superscriptℝ𝑛\bm{T}\in\mathbb{R}^{2^{d}-1}\times\mathbb{R}^{n}.
By enumerating the internal nodes in breadth-first order, we can redefine the indicator function 𝕃𝕃\mathbb{L} for a leaf l𝑙l, resulting in

|  |  |  |  |
| --- | --- | --- | --- |
|  | g​(𝒙|𝝀,T,I)=∑l=02d−1λl​𝕃​(𝒙|l,𝑻,𝑰)𝑔conditional𝒙  𝝀𝑇𝐼superscriptsubscript𝑙0superscript2𝑑1subscript𝜆𝑙𝕃conditional𝒙  𝑙𝑻𝑰g(\bm{x}|\bm{\lambda},T,I)=\sum\_{l=0}^{2^{d}-1}\lambda\_{l}\,\mathbb{L}(\bm{x}|l,\bm{T},\bm{I}) |  | (2) |

|  |  |  |  |
| --- | --- | --- | --- |
|  | where𝕃​(𝒙|l,𝑻,𝑰)=∏j=1d(1−𝔭​(l,j))​𝕊​(𝒙|𝑰𝔦​(l,j),𝑻𝔦​(l,j))+𝔭​(l,j)​(1−𝕊​(𝒙|𝑰𝔦​(l,j),𝑻𝔦​(l,j)))  where𝕃conditional𝒙  𝑙𝑻𝑰 subscriptsuperscriptproduct𝑑𝑗11𝔭𝑙𝑗𝕊conditional𝒙  subscript𝑰𝔦𝑙𝑗subscript𝑻𝔦𝑙𝑗𝔭𝑙𝑗1𝕊conditional𝒙  subscript𝑰𝔦𝑙𝑗subscript𝑻𝔦𝑙𝑗\text{where}\quad\mathbb{L}(\bm{x}|l,\bm{T},\bm{I})=\prod^{d}\_{j=1}\left(1-\mathfrak{p}(l,j)\right)\,\mathbb{S}(\bm{x}|\bm{I}\_{\mathfrak{i}(l,j)},\bm{T}\_{\mathfrak{i}(l,j)})+\mathfrak{p}(l,j)\,\left(1-\mathbb{S}(\bm{x}|\bm{I}\_{\mathfrak{i}(l,j)},\bm{T}\_{\mathfrak{i}(l,j)})\right) |  | (3) |

Here, 𝔦𝔦\mathfrak{i} is the index of the internal node preceding a leaf node l𝑙l at a certain depth j𝑗j and 𝔭𝔭\mathfrak{p} indicates whether the left (𝔭=0𝔭0\mathfrak{p}=0) or the right branch (𝔭=1𝔭1\mathfrak{p}=1) was taken.

Typically, DTs use the Heaviside step function for splitting, which is non-differentiable. GradTree reformulates the split function to account for reasonable gradients:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝕊(𝒙|𝜾,𝝉)=⌊S(𝜾⋅𝒙−𝜾⋅𝝉)⌉\mathbb{S}(\bm{x}|\bm{\iota},\bm{\tau})=\left\lfloor S\left(\bm{\iota}\cdot\bm{x}-\bm{\iota}\cdot\bm{\tau}\right)\right\rceil |  | (4) |

Where S​(z)=11+e−z𝑆𝑧11superscript𝑒𝑧S(z)=\frac{1}{1+e^{-z}} represents the logistic function, ⌊z⌉delimited-⌊⌉𝑧\left\lfloor z\right\rceil stands for rounding a real number z𝑧z to the nearest integer and 𝒂⋅𝒃⋅𝒂𝒃\bm{a}\cdot\bm{b} denotes the dot product between two vectors 𝒂𝒂\bm{a} and 𝒃𝒃\bm{b}. We further need to ensure that 𝜾𝜾\bm{\iota} is a one-hot encoded vector to account for axis-aligned splits. This is achieved by applying a hardmax transformation before calculating 𝕊𝕊\mathbb{S}.
Both rounding and hardmax operations are non-differentiable. To overcome this, GradTree employs the straight-through (ST) operator during backpropagation. This allows the model to use non-differentiable operations in the forward pass while ensuring gradient propagation in the backward pass.

## 3 GRANDE: Gradient-Based Decision Tree Ensembles

One core contribution of this paper is the extension of GradTree to tree ensembles (Section [3.1](#S3.SS1 "3.1 From Decision Trees to Weighted Tree Ensembles ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles")). In Section [3.2](#S3.SS2 "3.2 Differentiable Split Functions ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles") we propose softsign as a differentiable split function to propagate more reasonable gradients. Furthermore, we introduce an instance-wise weighting in Section [3.3](#S3.SS3 "3.3 Instance-Wise Estimator Weights ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles") and regularization techniques in Section [3.4](#S3.SS4 "3.4 Regularization: Feature Subset, Data Subset and Dropout ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles").
As a result, GRANDE can be learned end-to-end with gradient descent, leveraging the potential and flexibility of a gradient-based optimization.

### 3.1 From Decision Trees to Weighted Tree Ensembles

One advantage of GRANDE over existing gradient-based methods is the inductive bias of axis-aligned splits for tabular data.
Combining this property with an end-to-end gradient-based optimization is at the core of GRANDE.
This is also a major difference to existing deep learning methods for hierarchical representations like NODE, where soft, oblique splits are used (Popov et al., [2019](#bib.bib35)).
Therefore, we can define GRANDE as

|  |  |  |  |
| --- | --- | --- | --- |
|  | G​(𝒙|𝝎,𝑳,𝑻,𝑰)=∑e=0Eωe​g​(𝒙|𝑳e,𝑻e,𝑰e)𝐺conditional𝒙  𝝎𝑳𝑻𝑰superscriptsubscript𝑒0𝐸subscript𝜔𝑒𝑔conditional𝒙  subscript𝑳𝑒subscript𝑻𝑒subscript𝑰𝑒G(\bm{x}|\bm{\omega},\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}})=\sum\_{e=0}^{E}\omega\_{e}\,g(\bm{x}|\bm{L}\_{e},\bm{\mathsfit{T}}\_{e},\bm{\mathsfit{I}}\_{e}) |  | (5) |

where E𝐸E is the number of estimators in the ensemble and 𝝎𝝎\bm{\omega} is a weight vector. By extending 𝑳𝑳\bm{L} to a matrix and 𝑻𝑻\bm{\mathsfit{T}}, 𝑰𝑰\bm{\mathsfit{I}} to tensors for the complete ensemble instead of defining them individually for each tree, we can leverage parallel computation for an efficient training.

As GRANDE can be learned end-to-end with gradient descent, we keep an important advantage over existing, non-gradient-based tree methods like XGBoost and CatBoost. Both, the sequential induction of the individual trees and the sequential combination of individual trees via boosting are greedy. This results in constraints on the search space and can favor overfitting, as highlighted by Marton et al. ([2023](#bib.bib32)).
In contrast, GRANDE learns all parameters of the ensemble jointly and overcomes these limitations.

### 3.2 Differentiable Split Functions

The Heaviside step function, which is commonly used as split function in DTs, is non-differentiable. To address this challenge, various studies have proposed the employment of differentiable split functions. A predominant approach is the adoption of the sigmoid function, which facilitates soft decisions (Jordan & Jacobs, [1994](#bib.bib21); Irsoy et al., [2012](#bib.bib19); Frosst & Hinton, [2017](#bib.bib14)). A more recent development in this field originated with the introduction of the entmax transformation (Peters et al., [2019](#bib.bib33)). Researchers utilized a two-class entmax (entmoid) function to turn the decisions more sparse (Popov et al., [2019](#bib.bib35)). Further, Chang et al. ([2021](#bib.bib8)) proposed a temperature annealing procedure to gradually turn the decisions hard.
Marton et al. ([2023](#bib.bib32)) introduced an alternative method for generating hard splits by using a straight-through (ST) operator after a sigmoid split function to generate hard splits. While this allows using hard splits for calculating the function values, it also introduces a mismatch between the forward and backward pass. However, we can utilize this to incorporate additional information: By using a sigmoid function, the distance between a feature value and the threshold is used as additional information during gradient computation.
Accordingly, the gradient behavior plays a pivotal role in ensuring effective differentiation, especially in scenarios where input values are close to the decision threshold. The traditional sigmoid function can be suboptimal due to its smooth gradient decline. Entmoid, although addressing certain limitations of sigmoid, still displays an undesirable gradient behavior. Specifically, its gradient drops to zero when the difference in values is too pronounced. This can hinder the model’s ability to accommodate samples that exhibit substantial variances from the threshold.
Therefore, we propose using a softsign function, scaled to (0,1)01(0,1), as a differentiable split function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Sss​(z)=12​(z1+|z|+1)subscript𝑆ss𝑧12𝑧1𝑧1S\_{\text{ss}}(z)=\frac{1}{2}\left(\frac{z}{1+|z|}+1\right) |  | (6) |

The distinct gradient characteristics of the softsign, which are pronounced if samples are close to the threshold, reduce sharply but maintain responsive gradients if there is a large difference between the feature value and the threshold. These characteristics make it superior for differentiable splitting. This concept is visualized in Figure [1](#S3.F1 "Figure 1 ‣ 3.2 Differentiable Split Functions ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles"). Besides the intuitive advantage of using a softsign split function, we also show empirically that this is the superior choice (Table [4](#S4.T4 "Table 4 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).

![Refer to caption](/html/2309.17130/assets/x1.png)


(a) Sigmoid

![Refer to caption](/html/2309.17130/assets/x2.png)


(b) Entmoid

![Refer to caption](/html/2309.17130/assets/x3.png)


(c) Adjusted Softsign

Figure 1: Differentiable Split Functions. The sigmoid function’s gradient declines smoothly, while entmoid’s gradient decays more rapidly but becomes zero for large values. The scaled softsign activation has high gradients for small values but maintains a responsive gradient for large values, offering greater sensitivity.

### 3.3 Instance-Wise Estimator Weights

One challenge of ensemble methods is learning a good weighting scheme of the individual estimators. The flexibility of an end-to-end gradient-based optimization allows including learnable weight parameters to the optimization. A simple solution would be learning one weight for each estimator and using for instance a softmax over all weights, resulting in a weighted average.
However, this forces a very homogeneous ensemble, in which each tree aims to make equally good predictions for all samples. In contrast, it would be beneficial if individual trees can account for different areas of the target function, and are not required to make confident predictions for each sample.

![Refer to caption](/html/2309.17130/assets/x4.png)


Figure 2: GRANDE Architecture. This figure visualizes the structure and weighting of GRANDE for an exemplary ensemble with two trees of depth two.
For each tree in the ensemble, and for every sample, we determine the weight of the leaf which the sample is assigned to.
Subsequently, a softmax is applied on these chosen weights. Multiplying the post-softmax weights by the predictions equates a weighted average of the individual estimators.

To address this, we propose an advanced weighting scheme that allows calculating instance-wise weights that can be learned within the gradient-based optimization. Instead of using one weight per *estimator*, we use one weight for each *leaf* of the estimator as visualized in Figure [2](#S3.F2 "Figure 2 ‣ 3.3 Instance-Wise Estimator Weights ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles") and thus define the weights as 𝑾∈ℝE×ℝ2d𝑾superscriptℝ𝐸superscriptℝsuperscript2𝑑\bm{W}\in\mathbb{R}^{E}\times\mathbb{R}^{2^{d}} instead of 𝝎∈ℝE𝝎superscriptℝ𝐸\bm{\omega}\in\mathbb{R}^{E}.
We define p​(𝒙|𝑳,𝑻,𝑰):ℝn→ℝE:𝑝conditional𝒙

𝑳𝑻𝑰→superscriptℝ𝑛superscriptℝ𝐸p(\bm{x}|\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}}):\mathbb{R}^{n}\rightarrow\mathbb{R}^{E} as a function to calculate a vector comprising the individual prediction of each tree. Further, we define a function w​(𝒙|𝑾,𝑳,𝑻,𝑰):ℝn→ℝE:𝑤conditional𝒙

𝑾𝑳𝑻𝑰→superscriptℝ𝑛superscriptℝ𝐸w(\bm{x}|\bm{W},\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}}):\mathbb{R}^{n}\rightarrow\mathbb{R}^{E} to calculate a weight vector with one weight for each tree based on the leaf which the current sample is assigned to. Subsequently, a softmax is applied on these chosen weights for each sample. The process of multiplying the post-softmax weights by the predicted values from each tree equates to computing a weighted average.
This results in

|  |  |  |  |
| --- | --- | --- | --- |
|  | G​(𝒙|𝑾,𝑳,𝑻,𝑰)=σ​(w​(𝒙|𝑾,𝑳,𝑻,𝑰))⋅p​(𝒙|𝑳,𝑻,𝑰)𝐺conditional𝒙  𝑾𝑳𝑻𝑰⋅𝜎𝑤conditional𝒙  𝑾𝑳𝑻𝑰𝑝conditional𝒙  𝑳𝑻𝑰G(\bm{x}|\bm{W},\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}})=\sigma\left(w(\bm{x}|\bm{W},\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}})\right)\cdot p(\bm{x}|\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}}) |  | (7) |

|  |  |  |
| --- | --- | --- |
|  | wherew​(𝒙|𝑾,𝑳,𝑻,𝑰)=[∑l2d−1𝑾0,l​𝕃​(𝒙|𝑳0,l,𝑻0,𝑰0)∑l2d−1𝑾1,l​𝕃​(𝒙|𝑳1,l,𝑻1,𝑰1)⋮∑l2d−1𝑾E,l​𝕃​(𝒙|𝑳E,l,𝑻E,𝑰E)]​,p​(𝒙|𝑳,𝑻,𝑰)=[g​(𝒙|𝑳0,𝑻0,𝑰0)g​(𝒙|𝑳1,𝑻1,𝑰1)⋮g​(𝒙|𝑳E,𝑻E,𝑰E)]formulae-sequence  where𝑤conditional𝒙  𝑾𝑳𝑻𝑰 matrixsuperscriptsubscript𝑙superscript2𝑑1subscript𝑾  0𝑙𝕃conditional𝒙  subscript𝑳  0𝑙subscript𝑻0subscript𝑰0superscriptsubscript𝑙superscript2𝑑1subscript𝑾  1𝑙𝕃conditional𝒙  subscript𝑳  1𝑙subscript𝑻1subscript𝑰1⋮superscriptsubscript𝑙superscript2𝑑1subscript𝑾  𝐸𝑙𝕃conditional𝒙  subscript𝑳  𝐸𝑙subscript𝑻𝐸subscript𝑰𝐸,𝑝conditional𝒙  𝑳𝑻𝑰matrix𝑔conditional𝒙  subscript𝑳0subscript𝑻0subscript𝑰0𝑔conditional𝒙  subscript𝑳1subscript𝑻1subscript𝑰1⋮𝑔conditional𝒙  subscript𝑳𝐸subscript𝑻𝐸subscript𝑰𝐸\text{where}\quad w(\bm{x}|\bm{W},\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}})=\begin{bmatrix}\sum\_{l}^{2^{d-1}}\bm{W}\_{0,l}\,\mathbb{L}(\bm{x}|\bm{L}\_{0,l},\bm{\mathsfit{T}}\_{0},\bm{\mathsfit{I}}\_{0})\\ \sum\_{l}^{2^{d-1}}\bm{W}\_{1,l}\,\mathbb{L}(\bm{x}|\bm{L}\_{1,l},\bm{\mathsfit{T}}\_{1},\bm{\mathsfit{I}}\_{1})\\ \vdots\\ \sum\_{l}^{2^{d-1}}\bm{W}\_{E,l}\,\mathbb{L}(\bm{x}|\bm{L}\_{E,l},\bm{\mathsfit{T}}\_{E},\bm{\mathsfit{I}}\_{E})\\ \end{bmatrix}\,\text{,}\quad p(\bm{x}|\bm{L},\bm{\mathsfit{T}},\bm{\mathsfit{I}})=\begin{bmatrix}g(\bm{x}|\bm{L}\_{0},\bm{\mathsfit{T}}\_{0},\bm{\mathsfit{I}}\_{0})\\ g(\bm{x}|\bm{L}\_{1},\bm{\mathsfit{T}}\_{1},\bm{\mathsfit{I}}\_{1})\\ \vdots\\ g(\bm{x}|\bm{L}\_{E},\bm{\mathsfit{T}}\_{E},\bm{\mathsfit{I}}\_{E})\\ \end{bmatrix} |  |

and σ​(𝐳)𝜎𝐳\sigma(\mathbf{z}) is the softmax function.
It is important to note that when calculating 𝕃𝕃\mathbb{L} (see Equation [3](#S2.E3 "In 2 Background: Gradient-Based Decision Trees ‣ GRANDE: Gradient-Based Decision Tree Ensembles")), only the value for the leaf to which the sample is assigned in a given tree is non-zero.

We want to note that our weighting scheme permits calculating instance-wise weights even for unseen samples. Furthermore, our weighting allows GRANDE to learn representations for simple and complex rules withing one model.
In our evaluation, we demonstrate that instance-wise weights significantly enhance the performance of GRANDE and emphasize local interpretability.

### 3.4 Regularization: Feature Subset, Data Subset and Dropout

The combination of tree-based methods with a gradient-based optimization opens the door for the application of numerous regularization techniques.
For each tree in the ensemble, we select a feature subset. Therefore, we can regularize our model and simultaneously, we solve the poor scalability of GradTree with an increasing number of features. Similarly, we select a subset of the samples for each estimator. Furthermore, we implemented dropout by randomly deactivating a predefined fraction of the estimators in the ensemble and rescaling the weights accordingly.

## 4 Experimental Evaluation

As pointed out by Grinsztajn et al. ([2022](#bib.bib16)), most papers presenting a new method for tabular data have a highly varying evaluation methodology, with a small number of datasets that might be biased towards the authors’ model. As a result, recent surveys showed that tree boosting methods like XGBoost and CatBoost are still state-of-the-art and outperform new architectures for tabular data on most datasets (Grinsztajn et al., [2022](#bib.bib16); Shwartz-Ziv & Armon, [2022](#bib.bib42); Borisov et al., [2022](#bib.bib5)).
This highlights the necessity for an extensive and unbiased evaluation, as we will carry out in the following, to accurately assess the performance of a new method and draw valid conclusions.
We want to emphasize that recent surveys and evaluation on predefined benchmarks indicate that there is no “one-size-fits-all” solution for all tabular datasets. Consequently, we should view new methods as an extension to the existing repertoire and set our expectations in line with this perspective.

### 4.1 Experimental Setup

Datasets and Preprocessing   
For our evaluation, we used a predefined collection of datasets that was selected based on objective criteria from OpenML Benchmark Suites and comprises a total of 19 binary classification datasets (see Table [5](#A1.T5 "Table 5 ‣ Appendix A Benchmark Dataset Selction ‣ GRANDE: Gradient-Based Decision Tree Ensembles") for details). The selection process was adopted from Bischl et al. ([2021](#bib.bib4)) and therefore is not biased towards our method. A more detailed discussion on the selection of the benchmark can be found in Appendix [A](#A1 "Appendix A Benchmark Dataset Selction ‣ GRANDE: Gradient-Based Decision Tree Ensembles").
We one-hot encoded low-cardinality categorical features and used leave-one-out encoding for high-cardinality categorical features (more than 10 categories). To make them suitable for a gradient-based optimization, we gaussianized features using a quantile transformation, as it is common practice (Grinsztajn et al., [2022](#bib.bib16)).
In line with Borisov et al. ([2022](#bib.bib5)), we report the mean and standard deviation of the test performance over a 5-fold cross-validation to ensure reliable results.

Table 1: Categorization of Approaches

|  |  |  |
| --- | --- | --- |
|  | Standard DTs | Oblivious DTs |
| Tree-based | XGBoost | CatBoost |
| Gradient-based | GRANDE | NODE |

Methods   
We compare our approach to XGBoost and CatBoost, which achieved superior results according to recent studies, and NODE, which is most related to our approach. With this setup, we have one state-of-the-art tree-based and one gradient-based approach for each tree type (see Table [1](#S4.T1 "Table 1 ‣ 4.1 Experimental Setup ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles")). For a more extensive comparison of tree-based and gradient-based approaches, we refer to Borisov et al. ([2022](#bib.bib5)), Grinsztajn et al. ([2022](#bib.bib16)) and Shwartz-Ziv & Armon ([2022](#bib.bib42)). Our method is available under <https://github.com/s-marton/GRANDE>.

Hyperparameters   
We optimized the hyperparameters using Optuna (Akiba et al., [2019](#bib.bib2)) with 250 trials and selected the search space as well as the default parameters for related work in accordance with Borisov et al. ([2022](#bib.bib5)). The best parameters were selected based on a 5x2 cross-validation as suggested by Raschka ([2018](#bib.bib38)) where the test data of each fold was held out of the HPO to get unbiased results. To deal with class imbalance, we further included class weights. Additional information along with the hyperparameters for each approach are in Appendix [C](#A3 "Appendix C Hyperparameters ‣ GRANDE: Gradient-Based Decision Tree Ensembles").

### 4.2 Results

Table 2: Performance Comparison. We report the test macro F1-score (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GRANDE | XGB | CatBoost | NODE |
| dresses-sales | 0.612 ±plus-or-minus\pm 0.049 (1) | 0.581 ±plus-or-minus\pm 0.059 (3) | 0.588 ±plus-or-minus\pm 0.036 (2) | 0.564 ±plus-or-minus\pm 0.051 (4) |
| climate-simulation-crashes | 0.853 ±plus-or-minus\pm 0.070 (1) | 0.763 ±plus-or-minus\pm 0.064 (4) | 0.778 ±plus-or-minus\pm 0.050 (3) | 0.802 ±plus-or-minus\pm 0.035 (2) |
| cylinder-bands | 0.819 ±plus-or-minus\pm 0.032 (1) | 0.773 ±plus-or-minus\pm 0.042 (3) | 0.801 ±plus-or-minus\pm 0.043 (2) | 0.754 ±plus-or-minus\pm 0.040 (4) |
| wdbc | 0.975 ±plus-or-minus\pm 0.010 (1) | 0.953 ±plus-or-minus\pm 0.030 (4) | 0.963 ±plus-or-minus\pm 0.023 (3) | 0.966 ±plus-or-minus\pm 0.016 (2) |
| ilpd | 0.657 ±plus-or-minus\pm 0.042 (1) | 0.632 ±plus-or-minus\pm 0.043 (3) | 0.643 ±plus-or-minus\pm 0.053 (2) | 0.526 ±plus-or-minus\pm 0.069 (4) |
| tokyo1 | 0.921 ±plus-or-minus\pm 0.004 (3) | 0.915 ±plus-or-minus\pm 0.011 (4) | 0.927 ±plus-or-minus\pm 0.013 (1) | 0.921 ±plus-or-minus\pm 0.010 (2) |
| qsar-biodeg | 0.854 ±plus-or-minus\pm 0.022 (1) | 0.853 ±plus-or-minus\pm 0.020 (2) | 0.844 ±plus-or-minus\pm 0.023 (3) | 0.836 ±plus-or-minus\pm 0.028 (4) |
| ozone-level-8hr | 0.726 ±plus-or-minus\pm 0.020 (1) | 0.688 ±plus-or-minus\pm 0.021 (4) | 0.721 ±plus-or-minus\pm 0.027 (2) | 0.703 ±plus-or-minus\pm 0.029 (3) |
| madelon | 0.803 ±plus-or-minus\pm 0.010 (3) | 0.833 ±plus-or-minus\pm 0.018 (2) | 0.861 ±plus-or-minus\pm 0.012 (1) | 0.571 ±plus-or-minus\pm 0.022 (4) |
| Bioresponse | 0.794 ±plus-or-minus\pm 0.008 (3) | 0.799 ±plus-or-minus\pm 0.011 (2) | 0.801 ±plus-or-minus\pm 0.014 (1) | 0.780 ±plus-or-minus\pm 0.011 (4) |
| wilt | 0.936 ±plus-or-minus\pm 0.015 (2) | 0.911 ±plus-or-minus\pm 0.010 (4) | 0.919 ±plus-or-minus\pm 0.007 (3) | 0.937 ±plus-or-minus\pm 0.017 (1) |
| churn | 0.914 ±plus-or-minus\pm 0.017 (2) | 0.900 ±plus-or-minus\pm 0.017 (3) | 0.869 ±plus-or-minus\pm 0.021 (4) | 0.930 ±plus-or-minus\pm 0.011 (1) |
| phoneme | 0.846 ±plus-or-minus\pm 0.008 (4) | 0.872 ±plus-or-minus\pm 0.007 (2) | 0.876 ±plus-or-minus\pm 0.005 (1) | 0.862 ±plus-or-minus\pm 0.013 (3) |
| SpeedDating | 0.723 ±plus-or-minus\pm 0.013 (1) | 0.704 ±plus-or-minus\pm 0.015 (4) | 0.718 ±plus-or-minus\pm 0.014 (2) | 0.707 ±plus-or-minus\pm 0.015 (3) |
| PhishingWebsites | 0.969 ±plus-or-minus\pm 0.006 (1) | 0.968 ±plus-or-minus\pm 0.006 (2) | 0.965 ±plus-or-minus\pm 0.003 (4) | 0.968 ±plus-or-minus\pm 0.006 (3) |
| Amazon\_employee\_access | 0.665 ±plus-or-minus\pm 0.009 (2) | 0.621 ±plus-or-minus\pm 0.008 (4) | 0.671 ±plus-or-minus\pm 0.011 (1) | 0.649 ±plus-or-minus\pm 0.009 (3) |
| nomao | 0.958 ±plus-or-minus\pm 0.002 (3) | 0.965 ±plus-or-minus\pm 0.003 (1) | 0.964 ±plus-or-minus\pm 0.002 (2) | 0.956 ±plus-or-minus\pm 0.001 (4) |
| adult | 0.790 ±plus-or-minus\pm 0.006 (4) | 0.798 ±plus-or-minus\pm 0.004 (1) | 0.796 ±plus-or-minus\pm 0.004 (2) | 0.794 ±plus-or-minus\pm 0.004 (3) |
| numerai28.6 | 0.519 ±plus-or-minus\pm 0.003 (1) | 0.518 ±plus-or-minus\pm 0.001 (3) | 0.519 ±plus-or-minus\pm 0.002 (2) | 0.503 ±plus-or-minus\pm 0.010 (4) |
| Mean ↑↑\uparrow | 0.807 (1) | 0.792 (3) | 0.801 (2) | 0.775 (4) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.702 (1) | 0.417 (3) | 0.570 (2) | 0.395 (4) |

GRANDE outperforms existing methods on most datasets   
We evaluated the performance with optimized hyperparameters based on the macro F1-Score in Table [2](#S4.T2 "Table 2 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles") to account for class imbalance. Additionally, we report the accuracy and ROC-AUC score in the Appendix [B](#A2 "Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles"), which are consistent with the results presented in the following.
GRANDE outperformed existing methods and achieved the highest mean reciprocal rank (MRR) of 0.702 and the highest average performance of 0.807. CatBoost yielded the second-best results (MRR of 0.570 and mean of 0.801) followed by XGBoost (MRR of 0.417 and mean of 0.792) and NODE (MRR of 0.395 and mean of 0.775). Yet, our findings are in line with existing work, indicating that there is no universal method for tabular data.
However, on several datasets such as *climate-simulation-crashes* and *cylinder-bands* the performance difference to other methods was substantial, which highlights the importance of GRANDE as an extension to the existing repertoire. Furthermore, as the datasets are sorted by their size, we can observe that the results of GRANDE are especially good for small datasets, which is an interesting research direction for future work.

GRANDE is computationally efficient for large and high-dimensional datasets   
GRANDE averaged 45 seconds across all datasets, with a maximum runtime of 107 seconds. Thereby, the runtime of GRANDE is robust to high-dimensional (37 seconds for *Bioresponse* with 1,776 features) and larger datasets (39 seconds for *numerai28.6* with ≈\approx100,000 samples). GRANDE achieved a significantly lower runtime compared to our gradient-based benchmark NODE, which has an approximately three times higher average runtime of 130 seconds. However, it is important to note that GBDT frameworks, especially XGBoost, are highly efficient when executed on the GPU and achieve significantly lower runtimes compared to gradient-based methods. The complete runtimes are listed in the appendix (Table [9](#A2.T9 "Table 9 ‣ Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).

Table 3: Default Hyperparameter Performance Summary. The results are based on the test macro f1-score with the default setting. Complete results are listed in Table [8](#A2.T8 "Table 8 ‣ Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles").

|  |  |  |
| --- | --- | --- |
|  | Mean ↑↑\uparrow | Mean Reciprocal Rank (MRR) ↑↑\uparrow |
| GRANDE | 0.7931 (1) | 0.6404 (1) |
| XGB | 0.7877 (3) | 0.5175 (3) |
| CatBoost | 0.7925 (2) | 0.5219 (2) |
| NODE | 0.7663 (4) | 0.4035 (4) |

GRANDE outperforms existing methods with default hyperparameters   
Many machine learning methods, especially deep learning methods, are heavily reliant on a proper hyperparameter optimization. Yet, it is a desirable property that a method achieves good results even with their default setting. GRANDE achieves superior results with default hyperparameters, and significantly outperforms existing methods on most datasets. More specifically, GRANDE has the highest average performance (0.7931) and the highest MRR (0.6404) as summarized in Table [3](#S4.T3 "Table 3 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles").

Softsign improves performance   
As discussed in Section [3.2](#S3.SS2 "3.2 Differentiable Split Functions ‣ 3 GRANDE: Gradient-Based Decision Tree Ensembles ‣ GRANDE: Gradient-Based Decision Tree Ensembles"), we argue that employing softsign as split index activation propagates informative gradients beneficial for the optimization. In Table [4](#S4.T4 "Table 4 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles") we support these claims by showing a superior performance of GRANDE with a softsign activation compared to sigmoid as the default choice as well as an entmoid function which is commonly used in related work (Popov et al., [2019](#bib.bib35); Chang et al., [2021](#bib.bib8)).

Table 4: Ablation Study Summary. Left: Comparison of different options for differentiable split functions (complete results in Table [10](#A2.T10 "Table 10 ‣ Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles")). Right: Comparison of our instance-wise weighting based on leaf weights with a single weight for each estimator (complete results in Table [11](#A2.T11 "Table 11 ‣ Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles")).

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Differentiable Split Function | | | Weighting Technique | |
|  | Softsign | Entmoid | Sigmoid | Leaf Weights | Estimator Weights |
| Mean ↑↑\uparrow | 0.8071 (1) | 0.7990 (2) | 0.7959 (3) | 0.8071 (1) | 0.7857 (2) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.8246 (1) | 0.5526 (2) | 0.4561 (3) | 0.9211 (1) | 0.5789 (2) |

Instance-wise weighting increases model performance   
GRANDE uses instance-wise weighting to assign varying weights to estimators for each sample based on selected leaves. This promotes ensemble diversity and encourages estimators to capture unique local interactions. We argue that the ability to learn and represent simple, local rules with individual estimators in our ensemble can have a positive impact on the overall performance as it simplifies the task that has to be solved by the remaining estimators. As a result, GRANDE can efficiently learn compact representations for simple rules, where complex models usually tend to learn overly complex representations. In the following case study, we demonstrate the ability of GRANDE to learn compact representations for simple rules within a complex ensemble:

The *PhishingWebsites* dataset is concerned with identifying malicious websites based on metadata and additional observable characteristics. Although the task is challenging (i.e., it is not possible to solve it sufficiently well with a simple model, as shown in Table [12](#A2.T12 "Table 12 ‣ Appendix B Additional Results ‣ GRANDE: Gradient-Based Decision Tree Ensembles")), there exist several clear indicators for phishing websites.
Thus, some instances can be categorized using simple rules, while assigning other instances is more difficult.
Ideally, if an instance can be easily categorized, the model should follow simple rules to make a prediction.

![Refer to caption](/html/2309.17130/assets/x5.png)


Figure 3: Highest-Weighted Estimator. This figure visualizes the DT from GRANDE (1024 total estimators) which has the highest weight for an exemplary instance.

One example of such a rule, which holds universally in the given dataset, is that an instance can be classified as *phishing* if a prefix or suffix was added to the domain name. By assessing the weights for an exemplary instance fulfilling this rule, we can observe that the DT visualized in Figure [3](#S4.F3.1 "Figure 3 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles") accounts for
94% of the prediction. Accordingly, GRANDE has learned a very simple representation and the classification is derived by applying an easily comprehensible rule.
Notably, for the other methods, it is not possible to assess the importance of individual estimators out-of-the-box in a similar way, as the prediction is either derived by either sequentially summing up the predictions of all trees (e.g. XGBoost and CatBoost) or equally weighting all estimators (e.g. for random forests).
Furthermore, the results in Table [4](#S4.T4 "Table 4 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles") show that this has a significant positive impact on the average performance of GRANDE compared to using one single weight for each estimator.

Instance-wise weighting can be beneficial for local interpretability   
In addition to the performance increase, our instance-wise weighting has a notable impact on the local interpretability of GRANDE. For each instance, we can assess the weights of individual estimators and inspect the estimators with the highest importance to understand which rules have the greatest impact on the prediction. For the given example, we only need to observe a single tree of depth two (Figure [3](#S4.F3.1 "Figure 3 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles")) to understand why the given instance was classified as *phishing*, even though the complete model is very complex.
In contrast, existing ensemble methods require a global interpretation of the model and do not provide simple, local explanations out-of-the-box.

![Refer to caption](/html/2309.17130/assets/x6.png)


Figure 4: Anchors Explanations. This figure shows the local explanations generated by Anchors for the given instance. The explanation for GRANDE only comprises a single rule. In contrast, the corresponding explanations for the other methods have significantly higher complexity, which indicates that these methods are not able to learn simple representations within a complex model.

However, similar explanations can be extracted using Anchors (Ribeiro et al., [2018](#bib.bib40)). Anchors, as an extension to LIME (Ribeiro et al., [2016](#bib.bib39)), provides model-agnostic explanations by identifying conditions (called ”anchors”) which, when satisfied, guarantee a certain prediction with a high probability (noted as precision). These anchors are interpretable, rules-based conditions derived from input features that consistently lead to the same model prediction. Figure [4](#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles") shows the extracted rules for each approach. We can clearly see that the anchor extracted for GRANDE matches the rule we have identified based on the instance-wise weights in Figure [3](#S4.F3.1 "Figure 3 ‣ 4.2 Results ‣ 4 Experimental Evaluation ‣ GRANDE: Gradient-Based Decision Tree Ensembles"). Furthermore, it is evident that the prediction derived by GRANDE is much simpler compared to any other approach, as it only comprises a single rule. Notably, this comes without suffering a loss in the precision, which is 1.00 for all methods. Furthermore, the rule learned by GRANDE has a significantly higher coverage, which means that the rule applied by GRANDE is more broadly representative.
The corresponding experiment with additional details can be found in the supplementary material.

## 5 Related Work

Tabular data is the most frequently used type of data, and learning methods for tabular data are a field of very active research. Existing work can be divided into tree-based, deep learning and hybrid methods.
In the following, we categorize the most prominent methods based on these three categories and differentiate our approach from existing work.
For a more comprehensive review, we refer to Borisov et al. ([2022](#bib.bib5)), Shwartz-Ziv & Armon ([2022](#bib.bib42)) and Grinsztajn et al. ([2022](#bib.bib16)).

Tree-Based Methods   
Tree-based methods have been widely used for tabular data due to their interpretability and ability to capture non-linear relationships. While individual trees usually offer a higher interpretability, (gradient-boosted) tree ensembles are commonly used to achieve superior performance (Friedman, [2001](#bib.bib13)). The most prominent tree-based methods for tabular data improve the gradient boosting algorithm by for instance introducing advanced regularization (XGBoost (Chen & Guestrin, [2016](#bib.bib9))), a special handling for categorical variables (CatBoost (Prokhorenkova et al., [2018](#bib.bib36))) or a leaf-wise growth strategy (LightGBM (Ke et al., [2017](#bib.bib23))).
Regarding the structure, GRANDE is similar to existing tree-based models. The main difference is the end-to-end gradient-based training procedure, which offers additional flexibility, and the instance-wise weighting.

Deep Learning Methods   
With the success of deep learning in various domains, researchers have started to adjust deep learning architectures, mostly transformers, to tabular data (Gorishniy et al., [2021](#bib.bib15); Arik & Pfister, [2021](#bib.bib3); Huang et al., [2020](#bib.bib18); Cai et al., [2021](#bib.bib6); Kossen et al., [2021](#bib.bib27)).
According to recent studies, Self-Attention and Intersample Attention Transformer (SAINT) is the superior deep learning method for tabular data using attention over both, rows and columns (Somepalli et al., [2021](#bib.bib44)).
Although GRANDE, similar to deep learning methods, uses gradient descent for training, it has a shallow, hierarchical structure comprising hard, axis-aligned splits.

Hybrid Methods   
Hybrid methods aim to combine the strengths of a gradient-based optimization with other algorithms, most commonly tree-based methods (Abutbul et al., [2020](#bib.bib1); Hehn et al., [2020](#bib.bib17); Chen, [2020](#bib.bib10); Ke et al., [2019](#bib.bib25); [2018](#bib.bib24); Katzir et al., [2020](#bib.bib22)). One prominent way to achieve this is using soft DTs to apply gradient descent by replacing hard decisions with soft ones, and axis-aligned with oblique splits (Frosst & Hinton, [2017](#bib.bib14); Kontschieder et al., [2015](#bib.bib26); Luo et al., [2021](#bib.bib31)). Neural Oblivious Decision Ensembles (NODE) is one prominent hybrid method which learns ensembles of oblivious DTs with gradient descent and is therefore closely related to our work (Popov et al., [2019](#bib.bib35)). Oblivious DTs use the same splitting feature and threshold for each internal node at the same depth, which allows an efficient, parallel computation and makes them suitable as weak learners. In contrast, GRANDE uses standard DTs as weak learners. GRANDE can also be categorized as a hybrid method. The main difference to existing methods is the use of hard, axis-aligned splits, which prevents overly smooth solution typically inherent in soft, oblique trees.

Recent studies indicate that, despite huge effort in finding high-performant deep learning methods, tree-based models still outperform deep learning for tabular data (Grinsztajn et al., [2022](#bib.bib16); Borisov et al., [2022](#bib.bib5); Shwartz-Ziv & Armon, [2022](#bib.bib42)). However, they also highlight the need for gradient-based methods due to their flexibility.
One main reason for the superior performance of tree-based methods lies in the use of axis-aligned splits that are not biased towards overly smooth solutions (Grinsztajn et al., [2022](#bib.bib16)). Therefore, GRANDE aligns with this argument and utilizes hard, axis-aligned splits, while incorporating the benefits and flexibility of a gradient-based optimization.

## 6 Conclusion and Future Work

In this paper, we introduced GRANDE, a new method for learning hard, axis-aligned tree ensembles with gradient-descent.
GRANDE combines the advantageous inductive bias of axis-aligned splits with the flexibility offered by gradient descent optimization.
In an extensive evaluation on a predefined benchmark, we demonstrated that GRANDE achieved superior results. Both with optimized and default parameters, it outperformed existing state-of-the-art methods on most datasets.
Furthermore, we showed that the instance-wise weighting of GRANDE emphasizes learning representations for simple and complex relations within a single model, which increases the local interpretability compared to existing methods.

Currently, the proposed architecture is a shallow ensemble and already achieves state-of-the-art performance. However, the flexibility of a gradient-based optimization holds potential e.g., by including categorical embeddings, stacking of tree layers and an incorporation of tree layers to deep learning frameworks, which is subject to future work.

## References

* Abutbul et al. (2020)

  Ami Abutbul, Gal Elidan, Liran Katzir, and Ran El-Yaniv.
  Dnf-net: A neural architecture for tabular data.
  *arXiv preprint arXiv:2006.06465*, 2020.
* Akiba et al. (2019)

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*, pp.  2623–2631, 2019.
* Arik & Pfister (2021)

  Sercan Ö Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  In *Proceedings of the AAAI conference on artificial intelligence*, volume 35, pp.  6679–6687, 2021.
* Bischl et al. (2021)

  Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael Gomes Mantovani, Jan N van Rijn, and Joaquin Vanschoren.
  Openml benchmarking suites.
  In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.
* Borisov et al. (2022)

  Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci.
  Deep neural networks and tabular data: A survey.
  *IEEE Transactions on Neural Networks and Learning Systems*, 2022.
* Cai et al. (2021)

  Shaofeng Cai, Kaiping Zheng, Gang Chen, HV Jagadish, Beng Chin Ooi, and Meihui Zhang.
  Arm-net: Adaptive relation modeling network for structured data.
  In *Proceedings of the 2021 International Conference on Management of Data*, pp.  207–220, 2021.
* Cartella et al. (2021)

  Francesco Cartella, Orlando Anunciacao, Yuki Funabiki, Daisuke Yamaguchi, Toru Akishita, and Olivier Elshocht.
  Adversarial attacks for tabular data: Application to fraud detection and imbalanced data.
  *arXiv preprint arXiv:2101.08030*, 2021.
* Chang et al. (2021)

  Chun-Hao Chang, Rich Caruana, and Anna Goldenberg.
  Node-gam: Neural generalized additive model for interpretable deep learning.
  In *International Conference on Learning Representations*, 2021.
* Chen & Guestrin (2016)

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*, pp.  785–794, 2016.
* Chen (2020)

  Yingshi Chen.
  Attention augmented differentiable forest for tabular data.
  *arXiv preprint arXiv:2010.02921*, 2020.
* Chui et al. (2018)

  Michael Chui, James Manyika, Mehdi Miremadi, Nicolaus Henke, Rita Chung, Pieter Nel, and Sankalp Malhotra.
  Notes from the ai frontier: Insights from hundreds of use cases.
  *McKinsey Global Institute*, 2, 2018.
* Clements et al. (2020)

  Jillian M Clements, Di Xu, Nooshin Yousefi, and Dmitry Efimov.
  Sequential deep learning for credit risk monitoring with tabular financial data.
  *arXiv preprint arXiv:2012.15330*, 2020.
* Friedman (2001)

  Jerome H Friedman.
  Greedy function approximation: a gradient boosting machine.
  *Annals of statistics*, pp.  1189–1232, 2001.
* Frosst & Hinton (2017)

  Nicholas Frosst and Geoffrey Hinton.
  Distilling a neural network into a soft decision tree.
  *arXiv preprint arXiv:1711.09784*, 2017.
* Gorishniy et al. (2021)

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  *Advances in Neural Information Processing Systems*, 34:18932–18943, 2021.
* Grinsztajn et al. (2022)

  Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux.
  Why do tree-based models still outperform deep learning on typical tabular data?
  In *Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2022.
* Hehn et al. (2020)

  Thomas M Hehn, Julian FP Kooij, and Fred A Hamprecht.
  End-to-end learning of decision trees and forests.
  *International Journal of Computer Vision*, 128(4):997–1011, 2020.
* Huang et al. (2020)

  Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  *arXiv preprint arXiv:2012.06678*, 2020.
* Irsoy et al. (2012)

  Ozan Irsoy, Olcay Taner Yıldız, and Ethem Alpaydın.
  Soft decision trees.
  In *Proceedings of the 21st international conference on pattern recognition (ICPR2012)*, pp.  1819–1822. IEEE, 2012.
* Izmailov et al. (2018)

  Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson.
  Averaging weights leads to wider optima and better generalization.
  *arXiv preprint arXiv:1803.05407*, 2018.
* Jordan & Jacobs (1994)

  Michael I Jordan and Robert A Jacobs.
  Hierarchical mixtures of experts and the em algorithm.
  *Neural computation*, 6(2):181–214, 1994.
* Katzir et al. (2020)

  Liran Katzir, Gal Elidan, and Ran El-Yaniv.
  Net-dnf: Effective deep modeling of tabular data.
  In *International conference on learning representations*, 2020.
* Ke et al. (2017)

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30, 2017.
* Ke et al. (2018)

  Guolin Ke, Jia Zhang, Zhenhui Xu, Jiang Bian, and Tie-Yan Liu.
  Tabnn: A universal neural network solution for tabular data.
  *openreview preprint*, 2018.
* Ke et al. (2019)

  Guolin Ke, Zhenhui Xu, Jia Zhang, Jiang Bian, and Tie-Yan Liu.
  Deepgbm: A deep learning framework distilled by gbdt for online prediction tasks.
  In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp.  384–394, 2019.
* Kontschieder et al. (2015)

  Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, and Samuel Rota Bulo.
  Deep neural decision forests.
  In *Proceedings of the IEEE international conference on computer vision*, pp.  1467–1475, 2015.
* Kossen et al. (2021)

  Jannik Kossen, Neil Band, Clare Lyle, Aidan N Gomez, Thomas Rainforth, and Yarin Gal.
  Self-attention between datapoints: Going beyond individual input-output pairs in deep learning.
  *Advances in Neural Information Processing Systems*, 34:28742–28756, 2021.
* Lichtenwalter et al. (2021)

  David Lichtenwalter, Peter Burggräf, Johannes Wagner, and Tim Weißer.
  Deep multimodal learning for manufacturing problem solving.
  *Procedia CIRP*, 99:615–620, 2021.
* Lin et al. (2017)

  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.
  Focal loss for dense object detection.
  In *Proceedings of the IEEE international conference on computer vision*, pp.  2980–2988, 2017.
* Loshchilov & Hutter (2016)

  Ilya Loshchilov and Frank Hutter.
  Sgdr: Stochastic gradient descent with warm restarts.
  *arXiv preprint arXiv:1608.03983*, 2016.
* Luo et al. (2021)

  Haoran Luo, Fan Cheng, Heng Yu, and Yuqi Yi.
  Sdtr: Soft decision tree regressor for tabular data.
  *IEEE Access*, 9:55999–56011, 2021.
* Marton et al. (2023)

  Sascha Marton, Stefan Lüdtke, Christian Bartelt, and Heiner Stuckenschmidt.
  Learning decision trees with gradient descent.
  *arXiv preprint arXiv:2305.03515*, 2023.
* Peters et al. (2019)

  Ben Peters, Vlad Niculae, and André FT Martins.
  Sparse sequence-to-sequence models.
  *arXiv preprint arXiv:1905.05702*, 2019.
* Pölsterl et al. (2021)

  Sebastian Pölsterl, Tom Nuno Wolf, and Christian Wachinger.
  Combining 3d image and tabular data via the dynamic affine feature map transform.
  In *Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part V 24*, pp.  688–698. Springer, 2021.
* Popov et al. (2019)

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular data.
  *arXiv preprint arXiv:1909.06312*, 2019.
* Prokhorenkova et al. (2018)

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  *Advances in neural information processing systems*, 31, 2018.
* Rahaman et al. (2019)

  Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville.
  On the spectral bias of neural networks.
  In *International Conference on Machine Learning*, pp.  5301–5310. PMLR, 2019.
* Raschka (2018)

  Sebastian Raschka.
  Model evaluation, model selection, and algorithm selection in machine learning.
  *arXiv preprint arXiv:1811.12808*, 2018.
* Ribeiro et al. (2016)

  Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin.
  ”why should I trust you?”: Explaining the predictions of any classifier.
  In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016*, pp.  1135–1144, 2016.
* Ribeiro et al. (2018)

  Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin.
  Anchors: High-precision model-agnostic explanations.
  In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018.
* Sahoo et al. (2017)

  Doyen Sahoo, Quang Pham, Jing Lu, and Steven CH Hoi.
  Online deep learning: Learning deep neural networks on the fly.
  *arXiv preprint arXiv:1711.03705*, 2017.
* Shwartz-Ziv & Armon (2022)

  Ravid Shwartz-Ziv and Amitai Armon.
  Tabular data: Deep learning is not all you need.
  *Information Fusion*, 81:84–90, 2022.
* Somani et al. (2021)

  Sulaiman Somani, Adam J Russak, Felix Richter, Shan Zhao, Akhil Vaid, Fayzan Chaudhry, Jessica K De Freitas, Nidhi Naik, Riccardo Miotto, Girish N Nadkarni, et al.
  Deep learning and the electrocardiogram: review of the current state-of-the-art.
  *EP Europace*, 23(8):1179–1191, 2021.
* Somepalli et al. (2021)

  Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C Bayan Bruss, and Tom Goldstein.
  Saint: Improved neural networks for tabular data via row attention and contrastive pre-training.
  *arXiv preprint arXiv:2106.01342*, 2021.
* Ulmer et al. (2020)

  Dennis Ulmer, Lotta Meijerink, and Giovanni Cinà.
  Trust issues: Uncertainty estimation does not enable reliable ood detection on medical tabular data.
  In *Machine Learning for Health*, pp.  341–354. PMLR, 2020.

## Appendix A Benchmark Dataset Selction

We decided against using the original CC18 benchmark, as the number of datasets (727272) is extremely high and as reported by Grinsztajn et al. ([2022](#bib.bib16)), the selection process was not strict enough, as there are many simple datasets contained. Similarly, we decided not to use the tabular benchmark presented by Grinsztajn et al. ([2022](#bib.bib16)), as the datasets were adjusted to be extremely homogenous by removing all side-aspects (e.g. class imbalance, high-dimensionality, dataset size). We argue that this also removes some main challenges when dealing with tabular data.
As a result, we decided to use the benchmark proposed by Bischl et al. ([2021](#bib.bib4))111The notebook for dataset selection can be accessed under <https://github.com/openml/benchmark-suites/blob/master/OpenML%20Benchmark%20generator.ipynb>., which has a more strict selection process than CC18. The benchmark originally includes both, binary and multi-class tasks. For this paper, due to the limited scope, we focused only on binary classification tasks. Yet, our benchmark has a large overlap with CC18 as 16/19161916/19 datasets are also contained in CC18. The overlap with Grinsztajn et al. ([2022](#bib.bib16)) in contrast is rather small. This is mainly caused by the fact that most datasets in their tabular benchmark are binarized versions of multi-class or regression datasets, which was not allowed during the selection of our benchmark. Table [5](#A1.T5 "Table 5 ‣ Appendix A Benchmark Dataset Selction ‣ GRANDE: Gradient-Based Decision Tree Ensembles") lists the used datasets, along with relevant statistics and the source based on the OpenML-ID.

Table 5: Datasets

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Samples | Features | Categorical  Features | Features  (preprocessed) | Minority  Class | OpenML ID |
| dresses-sales | 500 | 12 | 11 | 37 | 42.00% | 23381 |
| climate-simulation-crashes | 540 | 18 | 0 | 18 | 8.52% | 40994 |
| cylinder-bands | 540 | 37 | 19 | 82 | 42.22% | 6332 |
| wdbc | 569 | 30 | 0 | 30 | 37.26% | 1510 |
| ilpd | 583 | 10 | 1 | 10 | 28.64% | 1480 |
| tokyo1 | 959 | 44 | 2 | 44 | 36.08% | 40705 |
| qsar-biodeg | 1,055 | 41 | 0 | 41 | 33.74% | 1494 |
| ozone-level-8hr | 2,534 | 72 | 0 | 72 | 6.31% | 1487 |
| madelon | 2,600 | 500 | 0 | 500 | 50.00% | 1485 |
| Bioresponse | 3,751 | 1,776 | 0 | 1,776 | 45.77% | 4134 |
| wilt | 4,839 | 5 | 0 | 5 | 5.39% | 40983 |
| churn | 5,000 | 20 | 4 | 22 | 14.14% | 40701 |
| phoneme | 5,404 | 5 | 0 | 5 | 29.35% | 1489 |
| SpeedDating | 8,378 | 120 | 61 | 241 | 16.47% | 40536 |
| PhishingWebsites | 11,055 | 30 | 30 | 46 | 44.31% | 4534 |
| Amazon\_employee\_access | 32,769 | 9 | 9 | 9 | 5.79% | 4135 |
| nomao | 34,465 | 118 | 29 | 172 | 28.56% | 1486 |
| adult | 48,842 | 14 | 8 | 37 | 23.93% | 1590 |
| numerai28.6 | 96,320 | 21 | 0 | 21 | 49.43% | 23517 |

## Appendix B Additional Results

Table 6: ROC-AUC Performance Comparison. We report the test ROC-AUC (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GRANDE | XGB | CatBoost | NODE |
| dresses-sales | 0.623 ±plus-or-minus\pm 0.049 (2) | 0.589 ±plus-or-minus\pm 0.060 (4) | 0.607 ±plus-or-minus\pm 0.040 (3) | 0.632 ±plus-or-minus\pm 0.041 (1) |
| climate-simulation-crashes | 0.958 ±plus-or-minus\pm 0.031 (1) | 0.923 ±plus-or-minus\pm 0.031 (4) | 0.938 ±plus-or-minus\pm 0.036 (2) | 0.933 ±plus-or-minus\pm 0.030 (3) |
| cylinder-bands | 0.896 ±plus-or-minus\pm 0.032 (1) | 0.872 ±plus-or-minus\pm 0.019 (3) | 0.879 ±plus-or-minus\pm 0.046 (2) | 0.837 ±plus-or-minus\pm 0.037 (4) |
| wdbc | 0.993 ±plus-or-minus\pm 0.006 (1) | 0.990 ±plus-or-minus\pm 0.010 (4) | 0.990 ±plus-or-minus\pm 0.010 (3) | 0.993 ±plus-or-minus\pm 0.007 (2) |
| ilpd | 0.748 ±plus-or-minus\pm 0.046 (1) | 0.721 ±plus-or-minus\pm 0.047 (4) | 0.728 ±plus-or-minus\pm 0.055 (3) | 0.745 ±plus-or-minus\pm 0.048 (2) |
| tokyo1 | 0.983 ±plus-or-minus\pm 0.005 (2) | 0.982 ±plus-or-minus\pm 0.006 (3) | 0.984 ±plus-or-minus\pm 0.005 (1) | 0.980 ±plus-or-minus\pm 0.005 (4) |
| qsar-biodeg | 0.934 ±plus-or-minus\pm 0.008 (1) | 0.925 ±plus-or-minus\pm 0.008 (3) | 0.933 ±plus-or-minus\pm 0.011 (2) | 0.920 ±plus-or-minus\pm 0.009 (4) |
| ozone-level-8hr | 0.925 ±plus-or-minus\pm 0.013 (1) | 0.879 ±plus-or-minus\pm 0.012 (4) | 0.910 ±plus-or-minus\pm 0.011 (2) | 0.906 ±plus-or-minus\pm 0.021 (3) |
| madelon | 0.875 ±plus-or-minus\pm 0.008 (3) | 0.904 ±plus-or-minus\pm 0.014 (2) | 0.928 ±plus-or-minus\pm 0.012 (1) | 0.612 ±plus-or-minus\pm 0.016 (4) |
| Bioresponse | 0.872 ±plus-or-minus\pm 0.003 (3) | 0.873 ±plus-or-minus\pm 0.007 (1) | 0.873 ±plus-or-minus\pm 0.002 (2) | 0.859 ±plus-or-minus\pm 0.008 (4) |
| wilt | 0.994 ±plus-or-minus\pm 0.007 (2) | 0.981 ±plus-or-minus\pm 0.015 (4) | 0.991 ±plus-or-minus\pm 0.009 (3) | 0.996 ±plus-or-minus\pm 0.003 (1) |
| churn | 0.928 ±plus-or-minus\pm 0.014 (1) | 0.919 ±plus-or-minus\pm 0.018 (4) | 0.920 ±plus-or-minus\pm 0.013 (3) | 0.927 ±plus-or-minus\pm 0.014 (2) |
| phoneme | 0.939 ±plus-or-minus\pm 0.006 (3) | 0.955 ±plus-or-minus\pm 0.007 (2) | 0.959 ±plus-or-minus\pm 0.005 (1) | 0.934 ±plus-or-minus\pm 0.010 (4) |
| SpeedDating | 0.859 ±plus-or-minus\pm 0.012 (1) | 0.827 ±plus-or-minus\pm 0.017 (4) | 0.856 ±plus-or-minus\pm 0.014 (2) | 0.853 ±plus-or-minus\pm 0.014 (3) |
| PhishingWebsites | 0.996 ±plus-or-minus\pm 0.001 (2) | 0.996 ±plus-or-minus\pm 0.001 (1) | 0.996 ±plus-or-minus\pm 0.001 (4) | 0.996 ±plus-or-minus\pm 0.001 (3) |
| Amazon\_employee\_access | 0.830 ±plus-or-minus\pm 0.010 (3) | 0.778 ±plus-or-minus\pm 0.015 (4) | 0.842 ±plus-or-minus\pm 0.014 (1) | 0.841 ±plus-or-minus\pm 0.009 (2) |
| nomao | 0.994 ±plus-or-minus\pm 0.001 (3) | 0.996 ±plus-or-minus\pm 0.001 (1) | 0.995 ±plus-or-minus\pm 0.001 (2) | 0.993 ±plus-or-minus\pm 0.001 (4) |
| adult | 0.910 ±plus-or-minus\pm 0.005 (4) | 0.927 ±plus-or-minus\pm 0.002 (1) | 0.925 ±plus-or-minus\pm 0.003 (2) | 0.915 ±plus-or-minus\pm 0.003 (3) |
| numerai28.6 | 0.529 ±plus-or-minus\pm 0.003 (3) | 0.529 ±plus-or-minus\pm 0.002 (2) | 0.529 ±plus-or-minus\pm 0.002 (1) | 0.529 ±plus-or-minus\pm 0.003 (4) |
| Mean ↑↑\uparrow | 0.883 (1) | 0.872 (3) | 0.883 (2) | 0.863 (4) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.645 (1) | 0.461 (3) | 0.575 (2) | 0.404 (4) |




Table 7: Accuracy Performance Comparison. We report the test balanced accuracy (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GRANDE | XGB | CatBoost | NODE |
| dresses-sales | 0.613 ±plus-or-minus\pm 0.048 (1) | 0.585 ±plus-or-minus\pm 0.060 (3) | 0.589 ±plus-or-minus\pm 0.037 (2) | 0.581 ±plus-or-minus\pm 0.037 (4) |
| climate-simulation-crashes | 0.875 ±plus-or-minus\pm 0.079 (1) | 0.762 ±plus-or-minus\pm 0.060 (4) | 0.832 ±plus-or-minus\pm 0.035 (2) | 0.773 ±plus-or-minus\pm 0.052 (3) |
| cylinder-bands | 0.817 ±plus-or-minus\pm 0.032 (1) | 0.777 ±plus-or-minus\pm 0.045 (3) | 0.798 ±plus-or-minus\pm 0.042 (2) | 0.751 ±plus-or-minus\pm 0.040 (4) |
| wdbc | 0.973 ±plus-or-minus\pm 0.010 (1) | 0.952 ±plus-or-minus\pm 0.029 (4) | 0.963 ±plus-or-minus\pm 0.023 (3) | 0.963 ±plus-or-minus\pm 0.017 (2) |
| ilpd | 0.709 ±plus-or-minus\pm 0.046 (1) | 0.673 ±plus-or-minus\pm 0.042 (3) | 0.689 ±plus-or-minus\pm 0.062 (2) | 0.548 ±plus-or-minus\pm 0.054 (4) |
| tokyo1 | 0.925 ±plus-or-minus\pm 0.002 (2) | 0.918 ±plus-or-minus\pm 0.012 (4) | 0.932 ±plus-or-minus\pm 0.010 (1) | 0.920 ±plus-or-minus\pm 0.007 (3) |
| qsar-biodeg | 0.853 ±plus-or-minus\pm 0.024 (2) | 0.856 ±plus-or-minus\pm 0.022 (1) | 0.847 ±plus-or-minus\pm 0.028 (3) | 0.831 ±plus-or-minus\pm 0.032 (4) |
| ozone-level-8hr | 0.774 ±plus-or-minus\pm 0.016 (1) | 0.733 ±plus-or-minus\pm 0.021 (3) | 0.735 ±plus-or-minus\pm 0.034 (2) | 0.669 ±plus-or-minus\pm 0.033 (4) |
| madelon | 0.803 ±plus-or-minus\pm 0.010 (3) | 0.833 ±plus-or-minus\pm 0.018 (2) | 0.861 ±plus-or-minus\pm 0.012 (1) | 0.571 ±plus-or-minus\pm 0.022 (4) |
| Bioresponse | 0.795 ±plus-or-minus\pm 0.009 (3) | 0.799 ±plus-or-minus\pm 0.011 (2) | 0.801 ±plus-or-minus\pm 0.014 (1) | 0.780 ±plus-or-minus\pm 0.011 (4) |
| wilt | 0.962 ±plus-or-minus\pm 0.026 (1) | 0.941 ±plus-or-minus\pm 0.012 (4) | 0.955 ±plus-or-minus\pm 0.021 (2) | 0.948 ±plus-or-minus\pm 0.024 (3) |
| churn | 0.909 ±plus-or-minus\pm 0.014 (1) | 0.895 ±plus-or-minus\pm 0.022 (3) | 0.894 ±plus-or-minus\pm 0.014 (4) | 0.904 ±plus-or-minus\pm 0.021 (2) |
| phoneme | 0.859 ±plus-or-minus\pm 0.011 (4) | 0.882 ±plus-or-minus\pm 0.005 (2) | 0.886 ±plus-or-minus\pm 0.006 (1) | 0.859 ±plus-or-minus\pm 0.013 (3) |
| SpeedDating | 0.752 ±plus-or-minus\pm 0.020 (2) | 0.740 ±plus-or-minus\pm 0.017 (3) | 0.758 ±plus-or-minus\pm 0.012 (1) | 0.694 ±plus-or-minus\pm 0.015 (4) |
| PhishingWebsites | 0.969 ±plus-or-minus\pm 0.006 (1) | 0.968 ±plus-or-minus\pm 0.006 (2) | 0.965 ±plus-or-minus\pm 0.003 (4) | 0.968 ±plus-or-minus\pm 0.006 (3) |
| Amazon\_employee\_access | 0.707 ±plus-or-minus\pm 0.010 (2) | 0.701 ±plus-or-minus\pm 0.015 (3) | 0.775 ±plus-or-minus\pm 0.009 (1) | 0.617 ±plus-or-minus\pm 0.007 (4) |
| nomao | 0.961 ±plus-or-minus\pm 0.001 (3) | 0.969 ±plus-or-minus\pm 0.002 (2) | 0.969 ±plus-or-minus\pm 0.001 (1) | 0.956 ±plus-or-minus\pm 0.001 (4) |
| adult | 0.817 ±plus-or-minus\pm 0.008 (3) | 0.841 ±plus-or-minus\pm 0.004 (2) | 0.841 ±plus-or-minus\pm 0.004 (1) | 0.778 ±plus-or-minus\pm 0.008 (4) |
| numerai28.6 | 0.520 ±plus-or-minus\pm 0.004 (2) | 0.520 ±plus-or-minus\pm 0.000 (3) | 0.521 ±plus-or-minus\pm 0.002 (1) | 0.519 ±plus-or-minus\pm 0.003 (4) |
| Mean ↑↑\uparrow | 0.821 (2) | 0.808 (3) | 0.822 (1) | 0.770 (4) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.689 (2) | 0.404 (3) | 0.693 (1) | 0.298 (4) |




Table 8: Default Parameter Performance Comparison. We report the test macro f1-score (mean ±plus-or-minus\pm stdev over 10 trials) with default parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GRANDE | XGB | CatBoost | NODE |
| dresses-sales | 0.596 ±plus-or-minus\pm 0.014 (1) | 0.570 ±plus-or-minus\pm 0.056 (3) | 0.573 ±plus-or-minus\pm 0.031 (2) | 0.559 ±plus-or-minus\pm 0.045 (4) |
| climate-simulation-crashes | 0.758 ±plus-or-minus\pm 0.065 (4) | 0.781 ±plus-or-minus\pm 0.060 (1) | 0.781 ±plus-or-minus\pm 0.050 (2) | 0.766 ±plus-or-minus\pm 0.088 (3) |
| cylinder-bands | 0.813 ±plus-or-minus\pm 0.023 (1) | 0.770 ±plus-or-minus\pm 0.010 (3) | 0.795 ±plus-or-minus\pm 0.051 (2) | 0.696 ±plus-or-minus\pm 0.028 (4) |
| wdbc | 0.962 ±plus-or-minus\pm 0.008 (3) | 0.966 ±plus-or-minus\pm 0.023 (1) | 0.955 ±plus-or-minus\pm 0.029 (4) | 0.964 ±plus-or-minus\pm 0.017 (2) |
| ilpd | 0.646 ±plus-or-minus\pm 0.021 (1) | 0.629 ±plus-or-minus\pm 0.052 (3) | 0.643 ±plus-or-minus\pm 0.042 (2) | 0.501 ±plus-or-minus\pm 0.085 (4) |
| tokyo1 | 0.922 ±plus-or-minus\pm 0.014 (1) | 0.917 ±plus-or-minus\pm 0.016 (4) | 0.917 ±plus-or-minus\pm 0.013 (3) | 0.921 ±plus-or-minus\pm 0.011 (2) |
| qsar-biodeg | 0.851 ±plus-or-minus\pm 0.032 (1) | 0.844 ±plus-or-minus\pm 0.021 (2) | 0.843 ±plus-or-minus\pm 0.017 (3) | 0.838 ±plus-or-minus\pm 0.027 (4) |
| ozone-level-8hr | 0.735 ±plus-or-minus\pm 0.011 (1) | 0.686 ±plus-or-minus\pm 0.034 (3) | 0.702 ±plus-or-minus\pm 0.029 (2) | 0.662 ±plus-or-minus\pm 0.019 (4) |
| madelon | 0.768 ±plus-or-minus\pm 0.022 (3) | 0.811 ±plus-or-minus\pm 0.016 (2) | 0.851 ±plus-or-minus\pm 0.015 (1) | 0.650 ±plus-or-minus\pm 0.017 (4) |
| Bioresponse | 0.789 ±plus-or-minus\pm 0.014 (2) | 0.789 ±plus-or-minus\pm 0.013 (3) | 0.792 ±plus-or-minus\pm 0.004 (1) | 0.786 ±plus-or-minus\pm 0.010 (4) |
| wilt | 0.933 ±plus-or-minus\pm 0.021 (1) | 0.903 ±plus-or-minus\pm 0.011 (3) | 0.898 ±plus-or-minus\pm 0.011 (4) | 0.904 ±plus-or-minus\pm 0.026 (2) |
| churn | 0.896 ±plus-or-minus\pm 0.007 (3) | 0.897 ±plus-or-minus\pm 0.022 (2) | 0.862 ±plus-or-minus\pm 0.015 (4) | 0.925 ±plus-or-minus\pm 0.025 (1) |
| phoneme | 0.860 ±plus-or-minus\pm 0.008 (3) | 0.864 ±plus-or-minus\pm 0.003 (1) | 0.861 ±plus-or-minus\pm 0.008 (2) | 0.842 ±plus-or-minus\pm 0.005 (4) |
| SpeedDating | 0.725 ±plus-or-minus\pm 0.007 (1) | 0.686 ±plus-or-minus\pm 0.010 (4) | 0.693 ±plus-or-minus\pm 0.013 (3) | 0.703 ±plus-or-minus\pm 0.013 (2) |
| PhishingWebsites | 0.969 ±plus-or-minus\pm 0.006 (1) | 0.969 ±plus-or-minus\pm 0.007 (2) | 0.963 ±plus-or-minus\pm 0.005 (3) | 0.961 ±plus-or-minus\pm 0.004 (4) |
| Amazon\_employee\_access | 0.602 ±plus-or-minus\pm 0.006 (4) | 0.608 ±plus-or-minus\pm 0.016 (3) | 0.652 ±plus-or-minus\pm 0.006 (1) | 0.621 ±plus-or-minus\pm 0.010 (2) |
| nomao | 0.955 ±plus-or-minus\pm 0.004 (3) | 0.965 ±plus-or-minus\pm 0.003 (1) | 0.962 ±plus-or-minus\pm 0.003 (2) | 0.955 ±plus-or-minus\pm 0.002 (4) |
| adult | 0.785 ±plus-or-minus\pm 0.008 (4) | 0.796 ±plus-or-minus\pm 0.003 (2) | 0.796 ±plus-or-minus\pm 0.005 (3) | 0.799 ±plus-or-minus\pm 0.003 (1) |
| numerai28.6 | 0.503 ±plus-or-minus\pm 0.003 (4) | 0.516 ±plus-or-minus\pm 0.002 (2) | 0.519 ±plus-or-minus\pm 0.001 (1) | 0.506 ±plus-or-minus\pm 0.009 (3) |
| Mean ↑↑\uparrow | 0.793 (1) | 0.788 (3) | 0.793 (2) | 0.766 (4) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.640 (1) | 0.518 (3) | 0.522 (2) | 0.404 (4) |




Table 9: Runtime Performance Comparison. We report the runtime (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size. For all methods, we used a single NVIDIA RTX A6000.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | GRANDE | XGB | CatBoost | NODE |
| dresses-sales | 011.121 ±plus-or-minus\pm 01.0 (3) | 0.052 ±plus-or-minus\pm 0.0 (1) | 04.340 ±plus-or-minus\pm 2.0 (2) | 031.371 ±plus-or-minus\pm 017.0 (4) |
| climate-simulation-crashes | 016.838 ±plus-or-minus\pm 03.0 (3) | 0.077 ±plus-or-minus\pm 0.0 (1) | 08.740 ±plus-or-minus\pm 9.0 (2) | 097.693 ±plus-or-minus\pm 010.0 (4) |
| cylinder-bands | 020.554 ±plus-or-minus\pm 02.0 (3) | 0.093 ±plus-or-minus\pm 0.0 (1) | 04.887 ±plus-or-minus\pm 2.0 (2) | 038.983 ±plus-or-minus\pm 012.0 (4) |
| wdbc | 029.704 ±plus-or-minus\pm 04.0 (3) | 0.151 ±plus-or-minus\pm 0.0 (1) | 01.046 ±plus-or-minus\pm 0.0 (2) | 083.548 ±plus-or-minus\pm 016.0 (4) |
| ilpd | 011.424 ±plus-or-minus\pm 01.0 (3) | 0.049 ±plus-or-minus\pm 0.0 (1) | 03.486 ±plus-or-minus\pm 3.0 (2) | 059.085 ±plus-or-minus\pm 024.0 (4) |
| tokyo1 | 021.483 ±plus-or-minus\pm 03.0 (3) | 0.078 ±plus-or-minus\pm 0.0 (1) | 01.485 ±plus-or-minus\pm 1.0 (2) | 084.895 ±plus-or-minus\pm 005.0 (4) |
| qsar-biodeg | 021.565 ±plus-or-minus\pm 02.0 (3) | 0.087 ±plus-or-minus\pm 0.0 (1) | 01.195 ±plus-or-minus\pm 0.0 (2) | 096.204 ±plus-or-minus\pm 020.0 (4) |
| ozone-level-8hr | 056.889 ±plus-or-minus\pm 06.0 (3) | 0.092 ±plus-or-minus\pm 0.0 (1) | 00.851 ±plus-or-minus\pm 0.0 (2) | 137.910 ±plus-or-minus\pm 027.0 (4) |
| madelon | 044.783 ±plus-or-minus\pm 24.0 (3) | 0.360 ±plus-or-minus\pm 0.0 (1) | 01.247 ±plus-or-minus\pm 0.0 (2) | 090.529 ±plus-or-minus\pm 013.0 (4) |
| Bioresponse | 037.224 ±plus-or-minus\pm 02.0 (3) | 0.865 ±plus-or-minus\pm 0.0 (1) | 02.136 ±plus-or-minus\pm 0.0 (2) | 309.178 ±plus-or-minus\pm 054.0 (4) |
| wilt | 044.476 ±plus-or-minus\pm 07.0 (3) | 0.127 ±plus-or-minus\pm 0.0 (1) | 01.090 ±plus-or-minus\pm 0.0 (2) | 199.653 ±plus-or-minus\pm 020.0 (4) |
| churn | 049.096 ±plus-or-minus\pm 05.0 (3) | 0.099 ±plus-or-minus\pm 0.0 (1) | 04.117 ±plus-or-minus\pm 3.0 (2) | 150.088 ±plus-or-minus\pm 033.0 (4) |
| phoneme | 059.286 ±plus-or-minus\pm 07.0 (3) | 0.201 ±plus-or-minus\pm 0.0 (1) | 01.793 ±plus-or-minus\pm 1.0 (2) | 240.607 ±plus-or-minus\pm 033.0 (4) |
| SpeedDating | 083.458 ±plus-or-minus\pm 24.0 (4) | 0.207 ±plus-or-minus\pm 0.0 (1) | 07.033 ±plus-or-minus\pm 1.0 (2) | 066.560 ±plus-or-minus\pm 022.0 (3) |
| PhishingWebsites | 107.101 ±plus-or-minus\pm 37.0 (3) | 0.271 ±plus-or-minus\pm 0.0 (1) | 08.527 ±plus-or-minus\pm 2.0 (2) | 340.660 ±plus-or-minus\pm 102.0 (4) |
| Amazon\_employee\_access | 037.190 ±plus-or-minus\pm 01.0 (4) | 0.047 ±plus-or-minus\pm 0.0 (1) | 02.021 ±plus-or-minus\pm 0.0 (2) | 030.309 ±plus-or-minus\pm 004.0 (3) |
| nomao | 095.775 ±plus-or-minus\pm 11.0 (3) | 0.268 ±plus-or-minus\pm 0.0 (1) | 10.911 ±plus-or-minus\pm 2.0 (2) | 208.682 ±plus-or-minus\pm 034.0 (4) |
| adult | 096.737 ±plus-or-minus\pm 06.0 (3) | 0.125 ±plus-or-minus\pm 0.0 (1) | 03.373 ±plus-or-minus\pm 1.0 (2) | 171.783 ±plus-or-minus\pm 034.0 (4) |
| numerai28.6 | 039.031 ±plus-or-minus\pm 03.0 (3) | 0.083 ±plus-or-minus\pm 0.0 (1) | 01.323 ±plus-or-minus\pm 1.0 (2) | 047.520 ±plus-or-minus\pm 038.0 (4) |
| Mean ↓↓\downarrow | 46.512 (3) | 0.175 (1) | 3.66 (2) | 130.80 (4) |




Table 10: Ablation Study Split Activation. We report the test macro F1-Score (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Softsign | Entmoid | Sigmoid |
| dresses-sales | 0.612 ±plus-or-minus\pm 0.049 (1) | 0.580 ±plus-or-minus\pm 0.047 (2) | 0.568 ±plus-or-minus\pm 0.052 (3) |
| climate-simulation-crashes | 0.853 ±plus-or-minus\pm 0.070 (1) | 0.840 ±plus-or-minus\pm 0.038 (2) | 0.838 ±plus-or-minus\pm 0.041 (3) |
| cylinder-bands | 0.819 ±plus-or-minus\pm 0.032 (1) | 0.802 ±plus-or-minus\pm 0.028 (3) | 0.807 ±plus-or-minus\pm 0.020 (2) |
| wdbc | 0.975 ±plus-or-minus\pm 0.010 (1) | 0.970 ±plus-or-minus\pm 0.007 (3) | 0.972 ±plus-or-minus\pm 0.008 (2) |
| ilpd | 0.657 ±plus-or-minus\pm 0.042 (2) | 0.652 ±plus-or-minus\pm 0.047 (3) | 0.663 ±plus-or-minus\pm 0.047 (1) |
| tokyo1 | 0.921 ±plus-or-minus\pm 0.004 (1) | 0.920 ±plus-or-minus\pm 0.010 (3) | 0.921 ±plus-or-minus\pm 0.008 (2) |
| qsar-biodeg | 0.854 ±plus-or-minus\pm 0.022 (3) | 0.855 ±plus-or-minus\pm 0.018 (2) | 0.845 ±plus-or-minus\pm 0.018 (1) |
| ozone-level-8hr | 0.726 ±plus-or-minus\pm 0.020 (1) | 0.710 ±plus-or-minus\pm 0.024 (3) | 0.707 ±plus-or-minus\pm 0.031 (2) |
| madelon | 0.803 ±plus-or-minus\pm 0.010 (1) | 0.773 ±plus-or-minus\pm 0.009 (2) | 0.747 ±plus-or-minus\pm 0.009 (3) |
| Bioresponse | 0.794 ±plus-or-minus\pm 0.008 (2) | 0.795 ±plus-or-minus\pm 0.012 (1) | 0.792 ±plus-or-minus\pm 0.010 (3) |
| wilt | 0.936 ±plus-or-minus\pm 0.015 (1) | 0.932 ±plus-or-minus\pm 0.014 (2) | 0.929 ±plus-or-minus\pm 0.012 (3) |
| churn | 0.914 ±plus-or-minus\pm 0.017 (1) | 0.899 ±plus-or-minus\pm 0.010 (2) | 0.887 ±plus-or-minus\pm 0.015 (3) |
| phoneme | 0.846 ±plus-or-minus\pm 0.008 (1) | 0.829 ±plus-or-minus\pm 0.002 (2) | 0.828 ±plus-or-minus\pm 0.010 (3) |
| SpeedDating | 0.723 ±plus-or-minus\pm 0.013 (3) | 0.725 ±plus-or-minus\pm 0.012 (1) | 0.725 ±plus-or-minus\pm 0.012 (2) |
| PhishingWebsites | 0.969 ±plus-or-minus\pm 0.006 (1) | 0.968 ±plus-or-minus\pm 0.006 (2) | 0.967 ±plus-or-minus\pm 0.006 (3) |
| Amazon\_employee\_access | 0.665 ±plus-or-minus\pm 0.009 (1) | 0.663 ±plus-or-minus\pm 0.010 (3) | 0.664 ±plus-or-minus\pm 0.016 (2) |
| nomao | 0.958 ±plus-or-minus\pm 0.002 (1) | 0.956 ±plus-or-minus\pm 0.002 (2) | 0.954 ±plus-or-minus\pm 0.003 (3) |
| adult | 0.790 ±plus-or-minus\pm 0.006 (2) | 0.793 ±plus-or-minus\pm 0.005 (1) | 0.790 ±plus-or-minus\pm 0.006 (3) |
| numerai28.6 | 0.519 ±plus-or-minus\pm 0.003 (2) | 0.520 ±plus-or-minus\pm 0.004 (1) | 0.519 ±plus-or-minus\pm 0.003 (3) |
| Mean ↑↑\uparrow | 0.807 (1) | 0.799 (2) | 0.796 (3) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.825 (1) | 0.553 (2) | 0.456 (3) |




Table 11: Ablation Study Weighting. We report the test macro F1-Score (mean ±plus-or-minus\pm stdev for a 5-fold CV) with optimized parameters and the ranking of each approach in parentheses. The datasets are sorted based on the data size.

|  |  |  |
| --- | --- | --- |
|  | Leaf Weights | Estimator Weights |
| dresses-sales | 0.612 ±plus-or-minus\pm 0.049 (1) | 0.605 ±plus-or-minus\pm 0.050 (2) |
| climate-simulation-crashes | 0.853 ±plus-or-minus\pm 0.070 (1) | 0.801 ±plus-or-minus\pm 0.040 (2) |
| cylinder-bands | 0.819 ±plus-or-minus\pm 0.032 (1) | 0.787 ±plus-or-minus\pm 0.051 (2) |
| wdbc | 0.975 ±plus-or-minus\pm 0.010 (1) | 0.970 ±plus-or-minus\pm 0.009 (2) |
| ilpd | 0.657 ±plus-or-minus\pm 0.042 (1) | 0.612 ±plus-or-minus\pm 0.064 (2) |
| tokyo1 | 0.921 ±plus-or-minus\pm 0.004 (2) | 0.922 ±plus-or-minus\pm 0.016 (1) |
| qsar-biodeg | 0.854 ±plus-or-minus\pm 0.022 (1) | 0.850 ±plus-or-minus\pm 0.019 (2) |
| ozone-level-8hr | 0.726 ±plus-or-minus\pm 0.020 (1) | 0.711 ±plus-or-minus\pm 0.020 (2) |
| madelon | 0.803 ±plus-or-minus\pm 0.010 (1) | 0.606 ±plus-or-minus\pm 0.028 (2) |
| Bioresponse | 0.794 ±plus-or-minus\pm 0.008 (1) | 0.784 ±plus-or-minus\pm 0.010 (2) |
| wilt | 0.936 ±plus-or-minus\pm 0.015 (1) | 0.930 ±plus-or-minus\pm 0.019 (2) |
| churn | 0.914 ±plus-or-minus\pm 0.017 (1) | 0.873 ±plus-or-minus\pm 0.018 (2) |
| phoneme | 0.846 ±plus-or-minus\pm 0.008 (1) | 0.845 ±plus-or-minus\pm 0.011 (2) |
| SpeedDating | 0.723 ±plus-or-minus\pm 0.013 (2) | 0.728 ±plus-or-minus\pm 0.009 (1) |
| PhishingWebsites | 0.969 ±plus-or-minus\pm 0.006 (1) | 0.965 ±plus-or-minus\pm 0.006 (2) |
| Amazon\_employee\_access | 0.665 ±plus-or-minus\pm 0.009 (2) | 0.675 ±plus-or-minus\pm 0.008 (1) |
| nomao | 0.958 ±plus-or-minus\pm 0.002 (1) | 0.954 ±plus-or-minus\pm 0.002 (2) |
| adult | 0.790 ±plus-or-minus\pm 0.0060 (1) | 0.790 ±plus-or-minus\pm 0.005 (2) |
| numerai28.6 | 0.519 ±plus-or-minus\pm 0.003 (1) | 0.519 ±plus-or-minus\pm 0.003 (2) |
| Mean ↑↑\uparrow | 0.807 (1) | 0.786 (2) |
| Mean Reciprocal Rank (MRR) ↑↑\uparrow | 0.921 (1) | 0.579 (2) |




Table 12: Pairwise Confusion Matrix PhishingWebsites We compare the predictions of each approach with the predictions of a CART DT. It becomes evident, that a simple model is not sufficient to solve the task well, as CART makes more than twice as many mistakes as state-of-the-art models.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Correct DT | Incorrect DT | Total |
| Correct GRANDE | 2012 | 128 | 2140 |
| Incorrect GRANDE | 15 | 56 | 71 |
| Total | 2027 | 184 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Correct DT | Incorrect DT | Total |
| Correct XGBoost | 2018 | 115 | 2133 |
| Incorrect XGBoost | 9 | 69 | 78 |
| Total | 2027 | 184 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Correct DT | Incorrect DT | Total |
| Correct CatBoost | 2019 | 105 | 2124 |
| Incorrect CatBoost | 8 | 79 | 87 |
| Total | 2027 | 184 |  |

|  |  |  |  |
| --- | --- | --- | --- |
|  | Correct DT | Incorrect DT | Total |
| Correct NODE | 2012 | 112 | 2124 |
| Incorrect NODE | 15 | 72 | 87 |
| Total | 2027 | 184 |  |

## Appendix C Hyperparameters

We optimized the hyperparameters using Optuna (Akiba et al., [2019](#bib.bib2)) with 250 trials and selected the search space and default parameters for related work in accordance with Borisov et al. ([2022](#bib.bib5)). The best parameters were selected based on a 5x2 cross-validation as suggested by Raschka ([2018](#bib.bib38)) where the test data of each fold was held out of the HPO to get unbiased results. To deal with class imbalance, we further included class weights. In line with Borisov et al. ([2022](#bib.bib5)), we did not tune the number of estimators for XGBoost and CatBoost, but used early stopping.

For GRANDE, we used a batch size of 64 and early stopping after 25 epochs.
Similar to NODE Popov et al. ([2019](#bib.bib35)), GRANDE uses an Adam optimizer with stochastic weight averaging over 5 checkpoints (Izmailov et al., [2018](#bib.bib20)) and a learning rate schedule that uses a cosine decay with optional warmup (Loshchilov & Hutter, [2016](#bib.bib30)). Furthermore, GRANDE allows using a focal factor (Lin et al., [2017](#bib.bib29)), similar to GradTree Marton et al. ([2023](#bib.bib32)).
In the supplementary material, we provide the notebook used for the optimization along with the search space for each approach.

Table 13: Hyperparameters GRANDE (Part 1).

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | depth | n\_estimators | lr\_weights | lr\_index | lr\_values | lr\_leaf |
| dresses-sales | 4 | 512 | 0.0015 | 0.0278 | 0.1966 | 0.0111 |
| climate-simulation-crashes | 4 | 2048 | 0.0007 | 0.0243 | 0.0156 | 0.0134 |
| cylinder-bands | 6 | 2048 | 0.0009 | 0.0084 | 0.0086 | 0.0474 |
| wdbc | 4 | 1024 | 0.0151 | 0.0140 | 0.1127 | 0.1758 |
| ilpd | 4 | 512 | 0.0007 | 0.0059 | 0.0532 | 0.0094 |
| tokyo1 | 6 | 1024 | 0.0029 | 0.1254 | 0.0056 | 0.0734 |
| qsar-biodeg | 6 | 2048 | 0.0595 | 0.0074 | 0.0263 | 0.0414 |
| ozone-level-8hr | 4 | 2048 | 0.0022 | 0.0465 | 0.0342 | 0.0503 |
| madelon | 4 | 2048 | 0.0003 | 0.0575 | 0.0177 | 0.0065 |
| Bioresponse | 6 | 2048 | 0.0304 | 0.0253 | 0.0073 | 0.0784 |
| wilt | 6 | 2048 | 0.0377 | 0.1471 | 0.0396 | 0.1718 |
| churn | 6 | 2048 | 0.0293 | 0.0716 | 0.0179 | 0.0225 |
| phoneme | 6 | 2048 | 0.0472 | 0.0166 | 0.0445 | 0.1107 |
| SpeedDating | 6 | 2048 | 0.0148 | 0.0130 | 0.0095 | 0.0647 |
| PhishingWebsites | 6 | 2048 | 0.0040 | 0.0118 | 0.0104 | 0.1850 |
| Amazon\_employee\_access | 6 | 2048 | 0.0036 | 0.0056 | 0.1959 | 0.1992 |
| nomao | 6 | 2048 | 0.0059 | 0.0224 | 0.0072 | 0.0402 |
| adult | 6 | 1024 | 0.0015 | 0.0087 | 0.0553 | 0.1482 |
| numerai28.6 | 4 | 512 | 0.0001 | 0.0737 | 0.0513 | 0.0371 |




Table 14: Hyperparameters GRANDE (Part 2).

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | dropout | selected\_variables | data\_fraction | focal\_factor | cosine\_decay\_steps |
| dresses-sales | 0.25 | 0.7996 | 0.9779 | 0 | 0.0 |
| climate-simulation-crashes | 0.00 | 0.6103 | 0.8956 | 0 | 1000.0 |
| cylinder-bands | 0.25 | 0.5309 | 0.8825 | 0 | 1000.0 |
| wdbc | 0.50 | 0.8941 | 0.8480 | 0 | 0.0 |
| ilpd | 0.50 | 0.6839 | 0.9315 | 3 | 1000.0 |
| tokyo1 | 0.50 | 0.5849 | 0.9009 | 0 | 1000.0 |
| qsar-biodeg | 0.00 | 0.5892 | 0.8098 | 0 | 0.0 |
| ozone-level-8hr | 0.25 | 0.7373 | 0.8531 | 0 | 1000.0 |
| madelon | 0.25 | 0.9865 | 0.9885 | 0 | 100.0 |
| Bioresponse | 0.50 | 0.5646 | 0.8398 | 0 | 0.0 |
| wilt | 0.25 | 0.9234 | 0.8299 | 0 | 0.0 |
| churn | 0.00 | 0.6920 | 0.8174 | 0 | 1000.0 |
| phoneme | 0.00 | 0.7665 | 0.8694 | 3 | 1000.0 |
| SpeedDating | 0.00 | 0.8746 | 0.8229 | 3 | 0.1 |
| PhishingWebsites | 0.00 | 0.9792 | 0.9588 | 0 | 0.1 |
| Amazon\_employee\_access | 0.50 | 0.9614 | 0.9196 | 3 | 0.0 |
| nomao | 0.00 | 0.8659 | 0.8136 | 0 | 100.0 |
| adult | 0.50 | 0.5149 | 0.8448 | 3 | 100.0 |
| numerai28.6 | 0.50 | 0.7355 | 0.8998 | 0 | 0.1 |




Table 15: Hyperparameters XGBoost.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | learning\_rate | max\_depth | reg\_alpha | reg\_lambda |
| dresses-sales | 0.1032 | 3 | 0.0000 | 0.0000 |
| climate-simulation-crashes | 0.0356 | 11 | 0.5605 | 0.0000 |
| cylinder-bands | 0.2172 | 11 | 0.0002 | 0.0057 |
| wdbc | 0.2640 | 2 | 0.0007 | 0.0000 |
| ilpd | 0.0251 | 4 | 0.3198 | 0.0000 |
| tokyo1 | 0.0293 | 3 | 0.2910 | 0.3194 |
| qsar-biodeg | 0.0965 | 5 | 0.0000 | 0.0000 |
| ozone-level-8hr | 0.0262 | 9 | 0.0000 | 0.6151 |
| madelon | 0.0259 | 6 | 0.0000 | 0.9635 |
| Bioresponse | 0.0468 | 5 | 0.9185 | 0.0000 |
| wilt | 0.1305 | 8 | 0.0000 | 0.0003 |
| churn | 0.0473 | 6 | 0.0000 | 0.3132 |
| phoneme | 0.0737 | 11 | 0.9459 | 0.2236 |
| SpeedDating | 0.0277 | 9 | 0.0000 | 0.9637 |
| PhishingWebsites | 0.1243 | 11 | 0.0017 | 0.3710 |
| Amazon\_employee\_access | 0.0758 | 11 | 0.9785 | 0.0042 |
| nomao | 0.1230 | 5 | 0.0000 | 0.0008 |
| adult | 0.0502 | 11 | 0.0000 | 0.7464 |
| numerai28.6 | 0.1179 | 2 | 0.0001 | 0.0262 |




Table 16: Hyperparameters CatBoost.

|  |  |  |  |
| --- | --- | --- | --- |
|  | learning\_rate | max\_depth | l2\_leaf\_reg |
| dresses-sales | 0.0675 | 3 | 19.8219 |
| climate-simulation-crashes | 0.0141 | 2 | 19.6955 |
| cylinder-bands | 0.0716 | 11 | 19.6932 |
| wdbc | 0.1339 | 3 | 0.7173 |
| ilpd | 0.0351 | 4 | 5.0922 |
| tokyo1 | 0.0228 | 5 | 0.5016 |
| qsar-biodeg | 0.0152 | 11 | 0.7771 |
| ozone-level-8hr | 0.0118 | 11 | 3.0447 |
| madelon | 0.0102 | 10 | 9.0338 |
| Bioresponse | 0.0195 | 11 | 8.1005 |
| wilt | 0.0192 | 11 | 1.1095 |
| churn | 0.0248 | 9 | 7.0362 |
| phoneme | 0.0564 | 11 | 0.6744 |
| SpeedDating | 0.0169 | 11 | 1.5494 |
| PhishingWebsites | 0.0239 | 8 | 1.6860 |
| Amazon\_employee\_access | 0.0123 | 11 | 1.6544 |
| nomao | 0.0392 | 8 | 2.6583 |
| adult | 0.1518 | 11 | 29.3098 |
| numerai28.6 | 0.0272 | 4 | 18.6675 |




Table 17: Hyperparameters NODE.

|  |  |  |  |
| --- | --- | --- | --- |
|  | num\_layers | total\_tree\_count | tree\_depth |
| dresses-sales | 2 | 2048 | 6 |
| climate-simulation-crashes | 2 | 1024 | 8 |
| cylinder-bands | 2 | 1024 | 8 |
| wdbc | 2 | 2048 | 6 |
| ilpd | 4 | 2048 | 8 |
| tokyo1 | 4 | 1024 | 8 |
| qsar-biodeg | 2 | 2048 | 6 |
| ozone-level-8hr | 4 | 1024 | 6 |
| madelon | 2 | 2048 | 6 |
| Bioresponse | 2 | 1024 | 8 |
| wilt | 4 | 1024 | 6 |
| churn | 2 | 1024 | 8 |
| phoneme | 4 | 2048 | 8 |
| SpeedDating | 4 | 1024 | 6 |
| PhishingWebsites | 2 | 1024 | 8 |
| Amazon\_employee\_access | 2 | 2048 | 6 |
| nomao | 2 | 2048 | 6 |
| adult | 4 | 1024 | 8 |
| numerai28.6 | 2 | 1024 | 8 |

[◄](/html/2309.17129)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2309.17130)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2309.17130)
[View original  
on arXiv](https://arxiv.org/abs/2309.17130)[►](/html/2309.17131)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Feb 28 03:54:06 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
