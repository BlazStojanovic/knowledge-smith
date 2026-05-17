---
arxiv: '2006.06465'
authors:
- Ami Abutbul amramabutbul@cs.technion.ac.il
- Gal Elidan elidan@google.com
- Liran Katzir lirank@google.com
- Ran El-Yaniv rani@cs.technion.ac.il
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'DNF-Net: A Neural Architecture for Tabular Data'
url: https://arxiv.org/abs/2006.06465
year: 2020
---

# DNF-Net: A Neural Architecture for Tabular Data

Ami Abutbul
  
amramabutbul@cs.technion.ac.il
  
Gal Elidan
  
elidan@google.com
  
Liran Katzir
  
lirank@google.com
  
Ran El-Yaniv
  
rani@cs.technion.ac.il

###### Abstract

A challenging open question in deep learning is how to handle tabular data. Unlike domains such as image and natural language processing, where deep architectures prevail, there is still no widely accepted neural architecture that dominates tabular data.
As a step toward bridging this gap, we present DNF-Net a novel generic architecture whose inductive bias elicits models whose structure corresponds to logical Boolean formulas in disjunctive normal form (DNF) over affine soft-threshold decision terms. In addition, DNF-Net promotes localized decisions that are taken over small subsets of the features. We present an extensive empirical study showing that DNF-Nets significantly and consistently outperform FCNs over tabular data.
With relatively few hyperparameters, DNF-Nets open the door to practical end-to-end handling of tabular data using neural networks.
We present ablation studies, which justify the design choices of DNF-Net including the three inductive bias elements, namely, Boolean formulation, locality, and feature selection.

## 1 Introduction

A key point in successfully applying deep neural models is the construction of architecture families that contain inductive bias relevant to the application domain.
Architectures such as CNNs and RNNs have become the preeminent favorites for modeling images and sequential data, respectively.
For example, the inductive bias of CNNs favors locality, as well as
translation and scale invariances. With these properties, CNNs work extremely well
on image data, and are capable of generating problem-dependent representations that almost completely overcome the need for expert knowledge.
Similarly, the inductive bias promoted by RNNs and LSTMs (and more recent models such as transformers) favors both locality and temporal stationarity.

When considering *tabular data*, however, neural networks are not the hypothesis class of choice. Most often, the winning class in learning problems involving *tabular data* is decision forests. In Kaggle competitions, for example, gradient boosting of decision trees (GBDTs) [[6](#bib.bib6), [9](#bib.bib9), [19](#bib.bib19), [14](#bib.bib14)] are generally the superior model.
While it is quite practical to use GBDTs for medium size datasets, it is extremely hard to scale these methods to very large datasets (e.g., Google or Facebook scale).
The most significant computational disadvantage of GBDTs is the need to store (almost) the entire dataset in memory111This disadvantage is shared among popular GBDT implementations: XGBoost, LightGBM, and CatBoost..
Moreover, handling multi-modal data, which involves both tabular and spatial data (e.g., images), is problematic. Thus, since GBDTs and neural networks cannot be organically optimized, such multi-modal tasks are left with sub-optimal solutions. Creating a purely neural model for tabular data, which can be trained with SGD end-to-end, is therefore a prime objective.

A few works have aimed at constructing neural models for tabular data (see Section [5](#S5 "5 Related Work ‣ DNF-Net: A Neural Architecture for Tabular Data")).
Currently, however, there is still no widely accepted end-to-end neural architecture that can handle
tabular data and consistently replace fully-connected architectures, or better yet, replace GBDTs.
Here we present DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s, a family of neural network architectures whose primary inductive bias is an ensemble comprising a disjunctive normal form (DNF) formulas over linear separators.
This family also promotes (input) feature selection and spatial localization of ensemble members.
These inductive biases have been included by design to promote
conceptually similar elements that are inherent GBDTs and random forests.
Appealingly, the DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} architecture can be trained end-to-end using standard gradient-based optimization. Importantly, it consistently and significantly outperforms FCNs on tabular data, and can sometime even outperform GBDTs.

The choice of appropriate inductive bias for specialized hypothesis classes for tabular data is challenging since, clearly, there are many different kinds of such data. Nevertheless,
the “universality” of forest methods in handling a wide variety of tabular data suggests that it might be beneficial to emulate, using neural networks, the important elements that are part of the tree ensemble representation and algorithms. Concretely, every decision tree is equivalent to some DNF formula over axis-aligned linear separators (see details in Section [3](#S3 "3 DNFs and Trees – A VC Analysis ‣ DNF-Net: A Neural Architecture for Tabular Data")). This makes DNFs an essential element in any such construction. Secondly, all contemporary forest ensemble methods rely heavily on feature selection. This feature selection is manifested both during the induction of each individual tree, where features are sequentially and greedily selected using information gain or other related heuristics, and by uniform sampling features for each ensemble member.
Finally, forest methods include an important localization element – GBDTs with their sequential construction within a boosting approach, where each tree re-weights the instance domain differently – and random forests with their reliance on bootstrap sampling.
DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s are designed to include precisely these three elements.

After introducing DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}, we include a Vapnik-Chervonenkins (VC) comparative analysis of DNFs and trees showing that DNFs potentially have advantage over trees when the input dimension is large and vice versa. We then present an extensive empirical study. First, we present an ablation study over three real-life tabular data prediction tasks that convincingly demonstrates the importance of all three elements included in the DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} design. Second, we analyze our novel feature selection component over controlled synthetic experiments, which indicate that this component is of independent interest.
Finally, we compare DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s to FCNs and GBDTs over several large classification tasks, including two past Kaggle competitions. Our results indicate that DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s consistently outperform FCNs, and can sometime even outperform GBDTs.

## 2 Disjunctive Normal Form Networks (DNF-Nets)

In this section we introduce the DNF-Net architecture, which consists
of three elements. The main component is a block of layers emulating a DNF formula.
This block will be referred to as a *Disjunctive Normal Neural Form* (DNNF). The second and third components, respectively, are a feature selection module, and a localization one.
In the remainder of this section we describe each component in detail. Throughout our description we denote by 𝐱∈ℝd𝐱superscriptℝ𝑑\mathbf{x}\in\mathbb{R}^{d} a column of input feature vectors, by 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}, its i𝑖ith entry, and by σ​(⋅)𝜎⋅\sigma(\cdot) the sigmoid function.

### 2.1 A Disjunctive Normal Neural Form (DNNF) Block

A *disjunctive normal neural form* (DNNF) block is assembled
using a two-hidden-layer network. The first layer creates
affine “literals” (features) and is trainable. The second layer implements a number of soft conjunctions over the literals, and the third output layer is a neural OR gate. Importantly, only the first layer is trainable, while the two other are binary and fixed.

We begin by describing the neural AND and OR gates. For an input vector 𝐱𝐱\mathbf{x},
we define soft, differentiable versions of such gates as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | OR​(𝐱)OR𝐱\displaystyle{\rm{OR}}(\mathbf{x}) | ≜≜\displaystyle\triangleq | tanh⁡(∑i=1d𝐱i+d−1.5),AND​(𝐱)≜tanh⁡(∑i=1d𝐱i−d+1.5).≜  superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1.5AND𝐱 superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1.5\displaystyle\tanh\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}+d-1.5\right),\hskip 50.0pt{\rm{AND}}(\mathbf{x})\triangleq\tanh\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}-d+1.5\right). |  |

These definitions are straightforwardly motivated by
the precise neural implementation of the corresponding binary gates.
Notice that by replacing tanh\tanh by a binary activation and changing the bias constant
from 1.5 to 1, we obtain an exact implementation of the corresponding logical gates
for binary input vectors [[2](#bib.bib2), [22](#bib.bib22)];
see a proof of this statement in Appendix [A](#A1 "Appendix A OR and AND Gates ‣ DNF-Net: A Neural Architecture for Tabular Data").
Notably, each unit does not have any trainable parameters.
We now define the AND gate in a vector form to project the logical operation over a subset of variables.
The projection is controlled by an indicator column vector (a mask)
𝐮∈{0,1}d𝐮superscript01𝑑\mathbf{u}\in\{0,1\}^{d}. With respect to such a projection vector 𝐮𝐮\mathbf{u}, we define the corresponding *projected* gate as AND𝐮​(𝐱)≜tanh⁡(𝐮T​𝐱−‖𝐮‖1+1.5).≜subscriptAND𝐮𝐱superscript𝐮𝑇𝐱subscriptnorm𝐮11.5{\rm{AND}}\_{\mathbf{u}}(\mathbf{x})\triangleq\tanh\kern-2.15277pt\left(\mathbf{u}^{T}\mathbf{x}-||\mathbf{u}||\_{1}+1.5\right).

Equipped with these definitions, a DNNF​(𝐱):ℝd→ℝ:DNNF𝐱→superscriptℝ𝑑ℝ{\rm{DNNF}}(\mathbf{x}):\mathbb{R}^{d}\rightarrow\mathbb{R} with k𝑘k conjunctions over m𝑚m literals is,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(𝐱)𝐿𝐱\displaystyle L(\mathbf{x}) | ≜tanh⁡(𝐱T​W+𝐛)∈ℝm≜absentsuperscript𝐱𝑇𝑊𝐛superscriptℝ𝑚\displaystyle\triangleq\tanh\kern-2.15277pt\left(\mathbf{x}^{T}W+\mathbf{b}\right)\in\mathbb{R}^{m} |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | DNNF​(𝐱)DNNF𝐱\displaystyle{\rm{DNNF}}(\mathbf{x}) | ≜OR​([AND𝐜1​(L​(𝐱)),AND𝐜2​(L​(𝐱)),…,AND𝐜k​(L​(𝐱))]).≜absentOR  subscriptANDsuperscript𝐜1𝐿𝐱subscriptANDsuperscript𝐜2𝐿𝐱…subscriptANDsuperscript𝐜𝑘𝐿𝐱\displaystyle\triangleq{\rm{OR}}\kern-2.15277pt\left([{\rm{AND}}\_{\mathbf{c}^{1}}(L(\mathbf{x})),{\rm{AND}}\_{\mathbf{c}^{2}}(L(\mathbf{x})),\dots,{\rm{AND}}\_{\mathbf{c}^{k}}(L(\mathbf{x}))]\right). |  | (2) |

Equation ([1](#S2.E1 "In 2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")) defines L​(𝐱)𝐿𝐱L(\mathbf{x}) that generates
m𝑚m “neural literals”, each of which is the result of a tanh\tanh-activation of a (trainable) affine transformation.
The (trainable) matrix W∈ℝd×m𝑊superscriptℝ𝑑𝑚W\in\mathbb{R}^{d\times m}, as well as the row vector bias term 𝐛∈ℝm𝐛superscriptℝ𝑚\mathbf{b}\in\mathbb{R}^{m},
determine the affine transformations for each literal such that each of its columns corresponds to one literal.
Equation ([2](#S2.E2 "In 2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")) defines a DNNF. In this equation, the vectors 𝐜i∈{0,1}msuperscript𝐜𝑖superscript01𝑚\mathbf{c}^{i}\in\{0,1\}^{m}, 1≤i≤k1𝑖𝑘1\leq i\leq k, are binary indicators such that 𝐜ji=1subscriptsuperscript𝐜𝑖𝑗1\mathbf{c}^{i}\_{j}=1 iff the j𝑗jth literal belongs to the i𝑖ith conjunction. In our design, each literal belongs to a single conjunction. These indicator vectors are defined and fixed according to the number and length of the conjunctions (See Appendix [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")).

### 2.2 DNF-Nets

The embedding layer of a DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} with n𝑛n DNNF blocks is a simple concatenation

|  |  |  |  |
| --- | --- | --- | --- |
|  | E​(𝐱)≜[DNNF1​(𝐱),DNNF2​(𝐱),…,DNNFn​(𝐱)].≜𝐸𝐱  subscriptDNNF1𝐱subscriptDNNF2𝐱…subscriptDNNF𝑛𝐱E(\mathbf{x})\triangleq[{\rm{DNNF}}\_{1}(\mathbf{x}),{\rm{DNNF}}\_{2}(\mathbf{x}),\ldots,{\rm{DNNF}}\_{n}(\mathbf{x})]. |  | (3) |

Depending on the application, the final
DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} is a composition of an output layer over E​(𝐱)𝐸𝐱E(\mathbf{x}). For example, for binary classification (logistic output layer),
DNF​-​Net​(x):ℝd→(0,1):DNF-Net𝑥→superscriptℝ𝑑01{\rm{DNF\text{-}Net}}(x):\mathbb{R}^{d}\rightarrow(0,1) has the following form,

|  |  |  |  |
| --- | --- | --- | --- |
|  | DNF​-​Net​(𝐱)≜σ​(∑i=1nwi​DNNFi​(𝐱)+bi).≜DNF-Net𝐱𝜎superscriptsubscript𝑖1𝑛subscript𝑤𝑖subscriptDNNF𝑖𝐱subscript𝑏𝑖{\rm{DNF\text{-}Net}}(\mathbf{x})\triangleq\sigma\left(\sum\_{i=1}^{n}w\_{i}{\rm{DNNF}}\_{i}(\mathbf{x})+b\_{i}\right). |  | (4) |

To summarize, a DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} is always a four-layer network (including the output layer),
and only the first and last layers are learned.
Each DNNF block has two parameters: the number of conjunctions k𝑘k and the length m𝑚m of these conjunctions, allowing for a variety of DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} architectures.
In all our experiments we considered a single DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} architecture that has a
fixed diversity of DNNF blocks which includes a number of different DNNF groups with different k𝑘k, each of which has a number of conjunction sizes m𝑚m (see details in Appendix [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")).
The number n𝑛n of DNNFs was treated as a hyperparameter, and selected based on a validation set as described on Appendix [D.1](#A4.SS1 "D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data").

### 2.3 Feature Selection

One key strategy in decision tree training is greedy feature selection,
which is performed hierarchically at any split, and allows decision trees to exclude irrelevant features. Additionally,
decision tree ensemble algorithms apply random sampling to select a subset of the features, which is used to promote diversity, and prevent different trees focusing on the same set of dominant features in their greedy selection.
In line with these strategies, we include in our DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s
conceptually similar feature selection elements:
(1) a subset of features uniformly and randomly sampled for each DNNF;
(2) a trainable mechanism for feature selection, applied on the resulting random subset. These two elements are combined and implemented in the affine literal generation layer described in Equation ([1](#S2.E1 "In 2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")), and applied independently for each DNNF.
We now describe these techniques in detail.

Recalling that d𝑑d is the input dimension, the random selection is made by generating a stochastic binary mask, 𝐦s∈{0,1}dsubscript𝐦𝑠superscript01𝑑\mathbf{m}\_{s}\in\{0,1\}^{d}, such that the probability of any entry being 1 is p𝑝p (see Appendix [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data") for details on setting this parameter). For a given mask 𝐦ssubscript𝐦𝑠\mathbf{m}\_{s}, this selection can be applied over affine literals using
a simple product diag(𝐦s)⁡Wdiagsubscript𝐦𝑠𝑊\operatorname\*{diag}(\mathbf{m}\_{s})W, where W𝑊W is the matrix of Equation ([1](#S2.E1 "In 2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")).
We then construct a *trainable* mask 𝐦t∈ℝdsubscript𝐦𝑡superscriptℝ𝑑\mathbf{m}\_{t}\in\mathbb{R}^{d}, which will be applied on the features that are kept by 𝐦ssubscript𝐦𝑠\mathbf{m}\_{s}.
We introduce a novel trainable feature selection component that combines binary quantization of the mask together with modified elastic-net regularization.
To train a binarized
vector we resort to the straight-through estimator [[10](#bib.bib10), [11](#bib.bib11)], which can be used effectively to train
non-differentiable step functions such as a threshold or sign. The trick is to compute
the step function *exactly* in the forward pass, and utilize a differentiable proxy in the backward pass.
We use a version of the straight-through estimator for the sign function [[3](#bib.bib3)],

|  |  |  |
| --- | --- | --- |
|  | Φ​(x)≜{sign(x),forward pass;tanh⁡(x),backward pass.≜Φ𝑥casessign𝑥forward pass𝑥backward pass\Phi(x)\triangleq\begin{cases}\operatorname\*{sign}(x),&\text{forward pass};\\ \tanh(x),&\text{backward pass}.\\ \end{cases} |  |

Using the estimator Φ​(x)Φ𝑥\Phi(x), we define a differentiable binary threshold
function T​(x)=12​Φ​(|x|−ϵ)+12𝑇𝑥12Φ𝑥italic-ϵ12T(x)=\frac{1}{2}\Phi(|x|-\epsilon)+\frac{1}{2}, where ϵ∈ℝitalic-ϵℝ\epsilon\in\mathbb{R} defines an epsilon neighborhood around zero for which the output of T​(x)𝑇𝑥T(x) is zero, and one outside of this neighborhood (in all our experiments, we set ϵ=1italic-ϵ1\epsilon=1 and initialize the entries of 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t} above this threshold). We then apply this selection by diag(T​(𝐦t))⁡Wdiag𝑇subscript𝐦𝑡𝑊\operatorname\*{diag}(T(\mathbf{m}\_{t}))W.

Given a fixed stochastic selection 𝐦ssubscript𝐦𝑠\mathbf{m}\_{s}, to train the binarized selection 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t}
we employ regularization. Specifically, we consider a modified version of the elastic net regularization, R​(𝐦t,𝐦s)𝑅subscript𝐦𝑡subscript𝐦𝑠R(\mathbf{m}\_{t},\mathbf{m}\_{s}), which is tailored to our task. The modifications are reflected in two parts. First, the balancing between the L1subscript𝐿1L\_{1} and L2subscript𝐿2L\_{2} regularization is controlled by a trainable parameter α∈ℝ𝛼ℝ\alpha\in\mathbb{R}. Second, the expressions of the L1subscript𝐿1L\_{1} and L2subscript𝐿2L\_{2} regularization are replaced by R1​(𝐦t,𝐦s),R2​(𝐦t,𝐦s)

subscript𝑅1subscript𝐦𝑡subscript𝐦𝑠subscript𝑅2subscript𝐦𝑡subscript𝐦𝑠R\_{1}(\mathbf{m}\_{t},\mathbf{m}\_{s}),R\_{2}(\mathbf{m}\_{t},\mathbf{m}\_{s}), respectively (defined below). Moreover, since we want to take into account only features that were selected by the random component, the regularization is applied on the vector 𝐦t​s=𝐦t⊙𝐦ssubscript𝐦𝑡𝑠direct-productsubscript𝐦𝑡subscript𝐦𝑠\mathbf{m}\_{ts}=\mathbf{m}\_{t}\odot\mathbf{m}\_{s}, where ⊙direct-product\odot is element-wise multiplication.
The functional form of the modified elastic net regularization is as follows,

|  |  |  |
| --- | --- | --- |
|  | R2​(𝐦t,𝐦s)≜|‖𝐦t​s‖22‖𝐦s‖1−β​ϵ2|,R1​(𝐦t,𝐦s)≜|‖𝐦t​s‖1‖𝐦s‖1−β​ϵ|formulae-sequence≜subscript𝑅2subscript𝐦𝑡subscript𝐦𝑠superscriptsubscriptnormsubscript𝐦𝑡𝑠22subscriptnormsubscript𝐦𝑠1𝛽superscriptitalic-ϵ2≜subscript𝑅1subscript𝐦𝑡subscript𝐦𝑠subscriptnormsubscript𝐦𝑡𝑠1subscriptnormsubscript𝐦𝑠1𝛽italic-ϵ\displaystyle R\_{2}(\mathbf{m}\_{t},\mathbf{m}\_{s})\triangleq\left\lvert\frac{||\mathbf{m}\_{ts}||\_{2}^{2}}{||\mathbf{m}\_{s}||\_{1}}-\beta\epsilon^{2}\right\rvert,\hskip 50.0ptR\_{1}(\mathbf{m}\_{t},\mathbf{m}\_{s})\triangleq\left\lvert\frac{||\mathbf{m}\_{ts}||\_{1}}{||\mathbf{m}\_{s}||\_{1}}-\beta\epsilon\right\rvert |  |
|  |  |  |
| --- | --- | --- |
|  | R​(𝐦t,𝐦s)≜1−σ​(α)2​R2​(𝐦t,𝐦s)+σ​(α)​R1​(𝐦t,𝐦s).≜𝑅subscript𝐦𝑡subscript𝐦𝑠1𝜎𝛼2subscript𝑅2subscript𝐦𝑡subscript𝐦𝑠𝜎𝛼subscript𝑅1subscript𝐦𝑡subscript𝐦𝑠\displaystyle R(\mathbf{m}\_{t},\mathbf{m}\_{s})\triangleq\frac{1-\sigma(\alpha)}{2}R\_{2}(\mathbf{m}\_{t},\mathbf{m}\_{s})+\sigma(\alpha)R\_{1}(\mathbf{m}\_{t},\mathbf{m}\_{s}). |  |

The above formulation of R2​(⋅)subscript𝑅2⋅R\_{2}(\cdot) and R1​(⋅)subscript𝑅1⋅R\_{1}(\cdot) is motivated as follows.
First, we normalize both norms by dividing with the effective input dimension, ‖𝐦s‖1subscriptnormsubscript𝐦𝑠1||\mathbf{m}\_{s}||\_{1},
which is done to be invariant to the (effective) input size. Second, we define R2subscript𝑅2R\_{2} and R1subscript𝑅1R\_{1} as
absolute errors, which encourages each entry to be, on average, approximately equal to the threshold ϵitalic-ϵ\epsilon.
The reason is that the vector 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t} passes through a binary threshold, and though the exact values of its entries are irrelevant. What is relevant is whether these values are within epsilon neighborhood of zero or not. Thus, when the values are roughly equal to the threshold, it is more likely to converge to a balanced point where the regularization term is low and the relevant features were selected. The threshold term is controlled by β𝛽\beta (a hyperparameter), which controls the cardinality of 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t}, where smaller values of β𝛽\beta lead to sparser 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t}.

Finally, the functional form of a DNNF block with the feature selection component is obtained by plugging the masks into Equation ([2](#S2.E2 "In 2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")), L=tanh⁡(𝐱T​diag(T​(𝐦t))​diag(𝐦s)⁡W+𝐛)∈ℝm𝐿superscript𝐱𝑇diag𝑇subscript𝐦𝑡diagsubscript𝐦𝑠𝑊𝐛superscriptℝ𝑚L=\tanh\kern-2.15277pt\left(\mathbf{x}^{T}\operatorname\*{diag}(T(\mathbf{m}\_{t}))\operatorname\*{diag}(\mathbf{m}\_{s})W+\mathbf{b}\right)\in\mathbb{R}^{m}.
Additionally, the mean over R​(𝐦t,𝐦s)𝑅subscript𝐦𝑡subscript𝐦𝑠R(\mathbf{m}\_{t},\mathbf{m}\_{s}) in all DNNFs is added to the loss function as a regularizer.

### 2.4 Spacial Localization

The last element we incorporate in the DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} construction
is *spatial localization*. This element encourages
each DNNF unit in a DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} ensemble
to specialize in some focused proximity of the input domain.
Localization is a well-known technique in classical machine learning, with various implementations and applications (see, e.g., [[13](#bib.bib13), [17](#bib.bib17)]).
On the one hand, localization allows construction of low-bias experts. On the other hand, it helps promote diversity, and reduction of the correlation between experts, which can improve the performance of an ensemble [[12](#bib.bib12), [7](#bib.bib7)].

We incorporate spatial localization by associating a Gaussian kernel loc(𝐱|μ,𝚺)i\operatorname\*{loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma})\_{i}
with a trainable mean vector μisubscript𝜇𝑖\mathbf{\mu}\_{i} and a trainable diagonal covariance matrix 𝚺isubscript𝚺𝑖\mathbf{\Sigma}\_{i} for
the i𝑖ith DNNF. Given a DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} with n𝑛n DNNF blocks, the functional form of its embedding layer (Equation [3](#S2.E3 "In 2.2 DNF-Nets ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")), with the spatial localization, is

|  |  |  |  |
| --- | --- | --- | --- |
|  | loc(𝐱|μ,𝚺)locconditional𝐱  𝜇𝚺\displaystyle\operatorname\*{loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma}) | ≜[e−‖𝚺1​(𝐱−μ1)‖2,e−‖𝚺2​(𝐱−μ2)‖2,…,e−‖𝚺n​(𝐱−μn)‖2]∈ℝn≜absent  superscript𝑒subscriptnormsubscript𝚺1𝐱subscript𝜇12superscript𝑒subscriptnormsubscript𝚺2𝐱subscript𝜇22…superscript𝑒subscriptnormsubscript𝚺𝑛𝐱subscript𝜇𝑛2superscriptℝ𝑛\displaystyle\triangleq[e^{-||\mathbf{\Sigma}\_{1}(\mathbf{x}-\mathbf{\mu}\_{1})||\_{2}},e^{-||\mathbf{\Sigma}\_{2}(\mathbf{x}-\mathbf{\mu}\_{2})||\_{2}},\dots,e^{-||\mathbf{\Sigma}\_{n}(\mathbf{x}-\mathbf{\mu}\_{n})||\_{2}}]\in\mathbb{R}^{n} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | sm−loc⁡(𝐱|μ,𝚺)smlocconditional𝐱  𝜇𝚺\displaystyle\operatorname\*{sm-loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma}) | ≜Softmax{loc(𝐱|μ,𝚺)⋅σ​(τ)}∈(0,1)n≜absentSoftmax⋅locconditional𝐱  𝜇𝚺𝜎𝜏superscript01𝑛\displaystyle\triangleq\operatorname\*{Softmax}\left\{\operatorname\*{loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma})\cdot\sigma(\tau)\right\}\in(0,1)^{n} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | E​(𝐱)𝐸𝐱\displaystyle E(\mathbf{x}) | ≜[sm−loc(𝐱|μ,𝚺)1⋅DNNF1(𝐱),…,sm−loc(𝐱|μ,𝚺)n⋅DNNFn(𝐱)],\displaystyle\triangleq[\operatorname\*{sm-loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma})\_{1}\cdot{\rm{DNNF}}\_{1}(\mathbf{x}),\ldots,\operatorname\*{sm-loc}(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma})\_{n}\cdot{\rm{DNNF}}\_{n}(\mathbf{x})], |  |

where τ∈ℝ𝜏ℝ\tau\in\mathbb{R} is a trainable parameter such that σ​(τ)𝜎𝜏\sigma(\tau)
serves as the trainable temperature in the softmax.
The inclusion of an adaptive temperature in this localization mechanism
facilitates a data-dependent degree of exclusivity: at high temperatures,
only a few DNNFs will handle an input instance whereas at low temperatures,
more DNNFs will effectively participate in the ensemble.
Observe that our localization mechanism is fully trainable and does not add any
hyperparameters.

## 3 DNFs and Trees – A VC Analysis

The basic unit in our construction is a (soft) DNF formula instead of a tree. Here we provide a theoretical perspective on this design choice. Specifically, we analyze the VC-dimension of
Boolean DNF formulas and compare it to that of decision trees. With this analysis we gain some insight into the generalization ability of formulas
and trees, and argue numerically that the generalization of a DNF can be superior to a tree when the input dimension is not small (and vice versa).

Throughout this discussion, we consider binary classification problems
whose instances are Boolean vectors in {0,1}nsuperscript01𝑛\{0,1\}^{n}.
The first simple observation is that
every decision tree has an equivalent DNF formula. Simply,
each tree path from the root to a positively labeled leaf can be expressed
by a conjunction of the conditions over the features appearing along the path to the leaf, and the whole tree can be represented by a disjunction of the resulting conjunctions.
However, DNFs and decision trees are not equivalent,
and we demonstrate that in the lense of VC-dimension.
Simon et al. [[24](#bib.bib24)] presented an exact expression for the VC-dimension of decision trees as a function of the tree *rank*.

###### Definition 1 (Rank).

Consider a binary tree T𝑇T.
If T𝑇T consists of a single node, its rank is defined as 0.
If T𝑇T consists of a root, a left subtree T0subscript𝑇0T\_{0} of rank r0subscript𝑟0r\_{0}, and a right subtree T1subscript𝑇1T\_{1} of rank r1subscript𝑟1r\_{1}, then

|  |  |  |
| --- | --- | --- |
|  | r​a​n​k​(T)={1+r0if r0 = r1max⁡{r0,r1}else𝑟𝑎𝑛𝑘𝑇cases1subscript𝑟0if r0 = r1subscript𝑟0subscript𝑟1elserank(T)=\begin{cases}1+r\_{0}&\text{if $r\_{0}$ = $r\_{1}$}\\ \max\{r\_{0},r\_{1}\}&\text{else}\end{cases} |  |

!(/html/2006.06465/assets/VCDim.png)

Figure 1: V​C​D​i​m​(D​Tnr)𝑉𝐶𝐷𝑖𝑚𝐷superscriptsubscript𝑇𝑛𝑟VCDim(DT\_{n}^{r}) and the upper bound on V​C​D​i​m​(D​N​Fnk)𝑉𝐶𝐷𝑖𝑚𝐷𝑁superscriptsubscript𝐹𝑛𝑘VCDim(DNF\_{n}^{k}) (log scale) as a function of the input dimension

Clearly, for any decision tree T𝑇T over n𝑛n
variables, 1≤r​a​n​k​(T)≤n1𝑟𝑎𝑛𝑘𝑇𝑛1\leq rank(T)\leq n.
Also, it is not hard to see that a binary tree T𝑇T has a rank greater than r𝑟r iff the complete binary tree of depth r+1𝑟1r+1 can be embedded into T𝑇T.

###### Theorem 1 (Simon, [[24](#bib.bib24)]).

Let D​Tnr𝐷superscriptsubscript𝑇𝑛𝑟DT\_{n}^{r} denote the class of decision trees of rank at most r𝑟r on n𝑛n Boolean variables. Then it holds that
V​C​D​i​m​(D​Tnr)=∑i=0r(ni).𝑉𝐶𝐷𝑖𝑚𝐷superscriptsubscript𝑇𝑛𝑟superscriptsubscript𝑖0𝑟binomial𝑛𝑖VCDim(DT\_{n}^{r})=\sum\_{i=0}^{r}\binom{n}{i}.

The following theorem, whose proof appears in Appendix [B](#A2 "Appendix B Proof of Theorem 2 ‣ DNF-Net: A Neural Architecture for Tabular Data"),
upper bounds the VC-dimension of a Boolean DNF formula.

###### Theorem 2 (DNF VC-dimension bound).

Let D​N​Fnk𝐷𝑁superscriptsubscript𝐹𝑛𝑘DNF\_{n}^{k} be the class of DNF formulas with k𝑘k conjunctions on n𝑛n Boolean variables. Then it holds that
V​C​D​i​m​(D​N​Fnk)≤2​(n+1)​k​log⁡(3​k).𝑉𝐶𝐷𝑖𝑚𝐷𝑁superscriptsubscript𝐹𝑛𝑘2𝑛1𝑘3𝑘VCDim(DNF\_{n}^{k})\leq 2(n+1)k\log(3k).

It is evident that in the case of DNF formulas the upper bound on
the VC-dimension grows linearly with the input dimension, whereas in the case of decision trees, if the rank is greater than 1, the VC-dimension
grows polynomially (with degree at least 2) with the input dimension. In the worst case, this growth is exponential.
A direct comparison of these dimensions is not trivial because there is a complex dependency between the rank r𝑟r of a decision tree, and the number k𝑘k of the conjunctions of an equivalent DNF formula. Even if we compare
large-k𝑘k DNF formulas to small-rank trees, it is
clear that the VC-dimension of the trees can be significantly larger. For example, in Figure [1](#S3.F1 "Figure 1 ‣ 3 DNFs and Trees – A VC Analysis ‣ DNF-Net: A Neural Architecture for Tabular Data"), we plot the upper bounds on the VC-dimension of large formulas (solid
curves), and the exact VC-dimensions of small-rank trees
(dashed curves). With the exception of rank-2 trees,
the VC-dimension of decision trees dominates the dimension of DNFs,
when the input dimension exceeds 100.
Trees, however, may have an advantage over DNF formulas for low-dimensional inputs.

Since the VC-dimension is a qualitative proxy of the sample complexity of a hypothesis class, the above analysis provides theoretical motivation for expressing trees
using DNF formulas when the input dimension is not small.
Having said that, the disclaimer is that in the present discussion we have only considered binary problems. Moreover, the final hypothesis classes of both DNF-Nets
and GBDTs are more complex in structure.

## 4 Empirical Study

In this section, we present an empirical study that substantiates the design of DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s and
convincingly shows its significant advantage over FCN architectures.
The datasets used in this study are from Kaggle competitions
and OpenML
[[25](#bib.bib25)]. A summary of these datasets appears in Appendix [C](#A3 "Appendix C Tabular Dataset Description ‣ DNF-Net: A Neural Architecture for Tabular Data").
All results presented in this work were obtained using a massive grid search for optimizing each model’s hyperparameters. A detailed description of the grid search process with additional details can be found in Appendices [D.1](#A4.SS1 "D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data"), [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data"). We present the scores for each dataset according to the score function defined in the Kaggle competition we used, log-loss and area under ROC curve (AUC ROC) for multiclass datasets and binary datasets, respectively. All results are the mean of the test scores over five different partitions, and the standard error of the mean is reported.

The merit of the different DNF-Net components.
We start with two different ablation studies, where we evaluate the contributions
of the three DNF-Net components.
In the first study, we start with a vanilla three-hidden-layer FCN and gradually add each component separately. In the second study, we start each experiment with the complete DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} and leave one component out each time. In each study, we present the results on three
real-world datasets, where all results are test log-loss scores (lower is better), out-of-memory (OOM) entries mean that the network was too
large to execute on our machine (see Appendix [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")).
More technical details can be found in Appendix [D.4](#A4.SS4 "D.4 Ablation Study ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data").

Table 1: Gradual study (test log-loss scores)

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Eye Movements | | | Gesture Phase | | | Gas Concentrations | | |
| # formulas | 128 | 512 | 2048 | 128 | 512 | 2048 | 128 | 512 | 2048 |
| Exp 1: Fully  trained FCN | 0.9864  (±plus-or-minus\pm0.0038) | 1.0138  (±plus-or-minus\pm0.0083) | OOM | 1.3139  (±plus-or-minus\pm0.0067) | 1.3368  (±plus-or-minus\pm0.0084) | OOM | 0.2423  (±plus-or-minus\pm0.0598) | 0.3862  (±plus-or-minus\pm0.0594) | OOM |
| Exp 2: Adding  DNF structure | 0.9034  (±plus-or-minus\pm0.0058) | 0.9336  (±plus-or-minus\pm0.0058) | 1.3011  (±plus-or-minus\pm0.0431) | 1.1391  (±plus-or-minus\pm0.0059) | 1.1812  (±plus-or-minus\pm0.0117) | 1.8633  (±plus-or-minus\pm0.1026) | 0.0351  (±plus-or-minus\pm0.0048) | 0.0421  (±plus-or-minus\pm0.0046) | 0.0778  (±plus-or-minus\pm0.0080) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp 3: Adding  feature selection | 0.8134  (±plus-or-minus\pm0.0142) | 0.8163  (±plus-or-minus\pm0.0096) | 0.9652  (±plus-or-minus\pm0.0143) | 1.1411  (±plus-or-minus\pm0.0093) | 1.1320  (±plus-or-minus\pm0.0083) | 1.3015  (±plus-or-minus\pm0.0317) | 0.0227  (±plus-or-minus\pm0.0019) | 0.0265  (±plus-or-minus\pm0.0012) | 0.0516  (±plus-or-minus\pm0.0061) |
| Exp 4: Adding  localization | 0.7621  (±plus-or-minus\pm0.0079) | 0.7125  (±plus-or-minus\pm0.0077) | 0.6903  (±plus-or-minus\pm0.0049) | 0.9742  (±plus-or-minus\pm0.0079) | 0.9120  (±plus-or-minus\pm0.0123) | 0.8770  (±plus-or-minus\pm0.0088) | 0.0162  (±plus-or-minus\pm0.0013) | 0.0149  (±plus-or-minus\pm0.0008) | 0.0145  (±plus-or-minus\pm0.0011) |

Consider Table [1](#S4.T1 "Table 1 ‣ 4 Empirical Study ‣ DNF-Net: A Neural Architecture for Tabular Data"). In Exp 1 we start with a vanilla three-hidden-layer FCN with a tanh\tanh activation. To make a fair comparison, we defined the widths of the layers according to the widths in the DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} with the corresponding formulas. In Exp 2, we added the DNF structure to the networks from Exp 1 (see Section [2.1](#S2.SS1 "2.1 A Disjunctive Normal Neural Form (DNNF) Block ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")). In Exp 3 we added the feature selection component (Section [2.3](#S2.SS3 "2.3 Feature Selection ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")).
In is evident that performance is monotonically improving, where the best results are clearly obtained on the complete DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} (Exp 4).
A subtle but important observation is that in all of the first three experiments, for all datasets, the trend is that the lower the number of formulas, the better the score. This trend is reversed in Exp 4, where the localization component (Section [2.4](#S2.SS4 "2.4 Spacial Localization ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")) is added, highlighting the importance of using all components of the DNF-Net representation in concert.

Table 2: Leave one out study (test log-loss scores)

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Eye Movements | | | Gesture Phase | | | Gas Concentrations | | |
| # formulas | 128 | 512 | 2048 | 128 | 512 | 2048 | 128 | 512 | 2048 |
| Exp 4: Complete  DNF-Net | 0.7621  (±plus-or-minus\pm0.0079) | 0.7125  (±plus-or-minus\pm0.0077) | 0.6903  (±plus-or-minus\pm0.0049) | 0.9742  (±plus-or-minus\pm0.0079) | 0.9120  (±plus-or-minus\pm0.0123) | 0.8770  (±plus-or-minus\pm0.0088) | 0.0162  (±plus-or-minus\pm0.0013) | 0.0149  (±plus-or-minus\pm0.0008) | 0.0145  (±plus-or-minus\pm0.0011) |
| Exp 5: Leave  feature selection out | 0.8150  (±plus-or-minus\pm0.0046) | 0.8031  (±plus-or-minus\pm0.0046) | 0.7969  (±plus-or-minus\pm0.0054) | 0.9732  (±plus-or-minus\pm0.0082) | 0.9479  (±plus-or-minus\pm0.0081) | 0.9438  (±plus-or-minus\pm0.0111) | 0.0222  (±plus-or-minus\pm0.0018) | 0.0205  (±plus-or-minus\pm0.0021) | 0.0200  (±plus-or-minus\pm0.0022) |
| Exp 6: Leave  localization out | 0.8134  (±plus-or-minus\pm0.0142) | 0.8163  (±plus-or-minus\pm0.0096) | 0.9652  (±plus-or-minus\pm0.0143) | 1.1411  (±plus-or-minus\pm0.0093) | 1.1320  (±plus-or-minus\pm0.0083) | 1.3015  (±plus-or-minus\pm0.0317) | 0.0227  (±plus-or-minus\pm0.0019) | 0.0265  (±plus-or-minus\pm0.0012) | 0.0516  (±plus-or-minus\pm0.0061) |
| Exp 7: Leave DNF  structure out | 0.8403  (±plus-or-minus\pm0.0068) | 0.8128  (±plus-or-minus\pm0.0077) | OOM | 1.1265  (±plus-or-minus\pm0.0066) | 1.1101  (±plus-or-minus\pm0.0077) | OOM | 0.0488  (±plus-or-minus\pm0.0038) | 0.0445  (±plus-or-minus\pm0.0024) | OOM |

Now consider Table [2](#S4.T2 "Table 2 ‣ 4 Empirical Study ‣ DNF-Net: A Neural Architecture for Tabular Data").
In Exp 5 we took the complete DNF-Net  (Exp 4) and removed the feature selection component. When considering the Gesture Phase dataset, an interesting phenomenon is observed. In Exp 3 (128 formulas), we can see that the contribution of the feature selection component is negligible, but in Exp 5 (2048 formulas) we see the significant contribution of this component. We believe that the reason for this difference lies in the relationship of the feature selection component with the localization component, where this connection intensifies the contribution of the feature selection component. In Exp 6 we took the complete DNF-Net  (Exp 4) and removed the localization component (identical to Exp 3). We did the same in Exp 7 where we removed the DNF structure. In general, it can be seen that removing each component results in a decrease in performance.

An analysis of the feature selection component.
Having studied the contribution of the three
components to DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}, we now focus
on the learnable part of the feature selection component (Section [2.3](#S2.SS3 "2.3 Feature Selection ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data")) alone, and examine its effectiveness using a series of synthetic tasks with a varying percentage of irrelevant features. Recall that when considering a single DNNF
block, the feature selection is a learnable binary mask that multiplies the input element-wise. Here we examine the effect of this mask on a vanilla FCN network (see technical details in Appendix [D.5](#A4.SS5 "D.5 Feature Selection Analysis ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")).
The synthetic tasks we use were introduced by Yoon et al.[[27](#bib.bib27)], and Chen et al.[[5](#bib.bib5)], where they were used as synthetic experiments to test feature selection.
There are six different dataset settings; exact
specifications appear in Appendix [D.5](#A4.SS5 "D.5 Feature Selection Analysis ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data").
For each dataset, we generated seven different instances that differ in their input size. While increasing the input dimension d𝑑d, the same logit is used for prediction, so the new features are irrelevant, and as d𝑑d gets larger, the percentage of relevant features becomes smaller.

We compare the performance of a vanilla FCN on three different cases: (1) oracle (ideal) feature selection (2) our (learned) feature selection mask, and (3) no feature selection. (See details in Appendix [D.5](#A4.SS5 "D.5 Feature Selection Analysis ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")). Consider the graphs in Figure [2](#S4.F2 "Figure 2 ‣ 4 Empirical Study ‣ DNF-Net: A Neural Architecture for Tabular Data"),
which demonstrate several interesting insights.
In all tasks the performance of the vanilla FCN is sensitive to irrelevant features, probably due to the representation power of the FCN, which is prone to overfitting.
On the other hand, by adding the feature selection component, we obtain near oracle performance on the first three tasks, and a significant improvement on the three others. Moreover, these results support our observation from the ablation studies: that the application of localization together with feature selection increases the latter’s contribution. We can see that in Syn1-3 where there is a single interaction, the results are better than in Syn4-6 where the input space is divided into two ‘local’ sub-spaces with different interactions.
These experiments indicate that the learnable feature selection we propose can have independent value.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Dataset | Test Metric | DNF-Net | XGBoost | FCN |
| Otto Group | log-loss | 45.600±0.445plus-or-minus45.6000.445\mathbf{45.600\pm 0.445} | 45.705±0.361plus-or-minus45.7050.36145.705\pm 0.361 | 47.898±0.480plus-or-minus47.8980.48047.898\pm 0.480 |
| Gesture Phase | log-loss | 86.798±0.810plus-or-minus86.7980.81086.798\pm 0.810 | 81.408±0.806plus-or-minus81.4080.806\mathbf{81.408\pm 0.806} | 102.070±0.964plus-or-minus102.0700.964102.070\pm 0.964 |
| Gas Concentrations | log-loss | 1.425±0.104plus-or-minus1.4250.104\mathbf{1.425\pm 0.104} | 2.219±0.219plus-or-minus2.2190.2192.219\pm 0.219 | 5.814±1.079plus-or-minus5.8141.0795.814\pm 1.079 |
| Eye Movements | log-loss | 68.037±0.651plus-or-minus68.0370.65168.037\pm 0.651 | 57.447±0.664plus-or-minus57.4470.664\mathbf{57.447\pm 0.664} | 78.797±0.674plus-or-minus78.7970.67478.797\pm 0.674 |
| Santander Transaction | roc auc | 88.668±0.128plus-or-minus88.6680.12888.668\pm 0.128 | 89.682±0.165plus-or-minus89.6820.165\mathbf{89.682\pm 0.165} | 86.722±0.158plus-or-minus86.7220.15886.722\pm 0.158 |
| House | roc auc | 95.451±0.092plus-or-minus95.4510.09295.451\pm 0.092 | 95.525±0.138plus-or-minus95.5250.138\mathbf{95.525\pm 0.138} | 95.164±0.103plus-or-minus95.1640.10395.164\pm 0.103 |

Table 3: Mean test results on tabular datasets and standard error of the mean. We present the ROC AUC (higher is better) as a percentage, and the log-loss (lower is better) with an x100 factor.

Comparative Evaluation. 
Finally, we compare the performance of DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}
vs. the baselines. Consider Table [3](#S4.T3 "Table 3 ‣ 4 Empirical Study ‣ DNF-Net: A Neural Architecture for Tabular Data")
where we examine the performance of DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}s on six real-life tabular datasets (We add three larger datasets to those we used in the ablation studies). We compare our performance to XGboost [[6](#bib.bib6)], the widely used implementation of GBDTs, and to FCNs.
For each model, we optimized its critical hyperparameters. This optimization process required many computational resources: thousands of configurations have been tested for FCNs, hundreds of configurations for XGBoost, and only a few dozen for DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}}. A detailed description of the grid search we used for each model can be found in Appendix [D.3](#A4.SS3 "D.3 Grid Parameters – Tabular Datasets ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data").
In Table 3, we see that DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} consistently and significantly outperforms FCN over all the six datasets. While obtaining better than or indistinguishable results from XGBoost over two datasets, on the other datasets, DNF​-​NetDNF-Net{\rm{DNF\text{-}Net}} is slightly inferior but in the same ball park as XGBoost.

!(/html/2006.06465/assets/Syn1_results.png)

(a) Syn1

!(/html/2006.06465/assets/Syn2_results.png)

(b) Syn2

!(/html/2006.06465/assets/Syn3_results.png)

(c) Syn3

!(/html/2006.06465/assets/Syn4_results.png)

(d) Syn4

!(/html/2006.06465/assets/Syn5_results.png)

(e) Syn5

!(/html/2006.06465/assets/Syn6_results.png)

(f) Syn6

Figure 2: The results on the six synthetic experiments. For each experiment we present the test accuracy (with an error bar of the standard error of the mean) as a function of the input dimension d𝑑d.

## 5 Related Work

There have been a few attempts to construct neural networks with improved performance on tabular data.
A recurring idea in some of these works is the explicit use of conventional decision tree induction algorithms, such as ID3 [[20](#bib.bib20)], or conventional forest methods, such as GBDT [[9](#bib.bib9)] that are
trained over the data at hand, and then parameters of the resulting decision trees are explicitly or implicitly “imported” into a neural network using teacher-student distillation [[15](#bib.bib15)],
explicit embedding of tree paths in a specialized network architecture with some kind of DNF structure [[21](#bib.bib21)],
and explicit utilization of forests as the main building block of layers [[8](#bib.bib8)].
This reliance on conventional decision tree or forest methods as an integral part of the proposed solution prevents end-to-end neural optimization, as we propose here. This deficiency is not only a theoretical nuisance but also makes it hard to use such models on very large datasets and in combination with other neural modules.

A few other recent techniques aimed to cope with tabular data using pure neural optimization as we propose here.
[[26](#bib.bib26)] considered a method to approximate a single node of a decision tree using a soft binning function that transforms continuous features into one-hot features.
While this method obtained results comparable to a single decision tree and an FCN (with two hidden layers), it is limited to settings where the number of features is small.
Popov et al. [[18](#bib.bib18)] proposed a network that combines elements of oblivious decision forests with dense residual networks. While this method achieved better results than GBDTs on several datasets, also FCNs
achieved better than or indistinguishable results from GBDTs on most of these cases as well.
Finally, focusing on microbiome data, a recent study [[23](#bib.bib23)] presented an elegant regularization technique, which produces extremely sparse networks that are suitable for microbiome tabular datasets.

Soft masks for feature selection have been considered before and
the advantage of using elastic net regularization in a variable selection task was presented by Zou and Hastie and others [[28](#bib.bib28), [16](#bib.bib16)].

## 6 Conclusions

We introduced DNF-Net, a novel neural architecture whose inductive bias
revolves around a disjunctive normal neural form, localization and feature selection.
The importance of each of these elements has been demonstrated over real tabular data.
The results of the empirical study indicate convincingly that DNF-Nets consistently outperform FCNs over tabular data. While DNF-Nets do not consistently
beat XGBoost, our results indicate that their performance score is not far behind.

We have left a number of potential incremental improvements and bigger challenges
to future work. First, in our work we only considered classification problems.
We expect DNF-Nets to also be effective in regression problems, and it would also
be interesting to consider
applications in reinforcement learning over finite discrete spaces.
It would be very interesting to consider deeper DNF-Net architectures. For example,
instead of a single DNNF block, one can construct a stack of such blocks
to allow for more involved feature generation.
Another interesting direction would be to consider training DNF-Nets using a
gradient boosting procedure similar to that used in XGBoost.

Finally, a most interesting challenge that remains open is what would constitute
a usable and effective inductive bias for tabular prediction tasks, which can
elicit the best architectural designs for these data.
Our successful application of DNNFs indicates that soft DNF formulas are
quite effective, and are strictly significantly superior to fully connected networks,
but we anticipate that further biases will be identified, at least for some families
of tabular tasks.

## Broader Impact

In this paper we present a new family of neural
architectures for tabular data. Since much of the medical information on people has multiple modalities including a tabular form,
the societal effect and potential positive outcomes of this work are quite substantial, by contributing to our
ability to handle multi-modal data end-to-end using neural networks.
Negative consequences might appear, of course, if
agents will utilize this technology to handle large scale tabular data to achieve some malicious
objectives. However, there is no particular malicious application foreseen.

## References

* [1]

  Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey
  Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al.
  Tensorflow: A system for large-scale machine learning.
  In 12th {{\{USENIX}}\} Symposium on Operating Systems Design and
  Implementation ({{\{OSDI}}\} 16), pages 265–283, 2016.
* [2]

  Martin Anthony.
  Connections between neural networks and Boolean functions.
  In Boolean Methods and Models, 2005.
* [3]

  Yoshua Bengio, Nicholas Leonard, and Aaron Courville.
  Estimating or propagating gradients through stochastic neurons for
  conditional computation.
  arXiv preprint arXiv:1308.3432, 2013.
* [4]

  Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K Warmuth.
  Learnability and the vapnik-chervonenkis dimension.
  Journal of the ACM (JACM), 36(4):929–965, 1989.
* [5]

  Jianbo Chen, Le Song, Martin J Wainwright, and Michael I Jordan.
  Learning to explain: An information-theoretic perspective on model
  interpretation.
  arXiv preprint arXiv:1802.07814, 2018.
* [6]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016.
* [7]

  Philip Derbeko, Ran El-Yaniv, and Ron Meir.
  Variance optimized bagging.
  In European Conference on Machine Learning, pages 60–72.
  Springer, 2002.
* [8]

  Ji Feng, Yang Yu, and Zhi-Hua Zhou.
  Multi-layered gradient boosting decision trees.
  In Advances in Neural Information Processing Systems, pages
  3555–3565, 2018.
* [9]

  Jerome H Friedman.
  Greedy function approximation: a gradient boosting machine.
  Annals of Statistics, pages 1189–1232, 2001.
* [10]

  Geoffrey Hinton.
  Neural networks for machine learning coursera video lectures -
  geoffrey hinton.
  2012.
* [11]

  Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua
  Bengio.
  Quantized neural networks: Training neural networks with low
  precision weights and activations.
  The Journal of Machine Learning Research, 18(1):6869–6898,
  2017.
* [12]

  Robert A Jacobs.
  Bias/variance analyses of mixtures-of-experts architectures.
  Neural computation, 9(2):369–383, 1997.
* [13]

  Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton.
  Adaptive mixtures of local experts.
  Neural Computation, 3(1):79–87, 1991.
* [14]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei
  Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In Advances in neural information processing systems, pages
  3146–3154, 2017.
* [15]

  Guolin Ke, Jia Zhang, Zhenhui Xu, Jiang Bian, and Tie-Yan Liu.
  Tabnn: A universal neural network solution for tabular data.
  2018.
* [16]

  Yifeng Li, Chih-Yu Chen, and Wyeth W Wasserman.
  Deep feature selection: theory and application to identify enhancers
  and promoters.
  Journal of Computational Biology, 23(5):322–336, 2016.
* [17]

  Ron Meir, Ran El-Yaniv, and Shai Ben-David.
  Localized boosting.
  In COLT, pages 190–199. Citeseer, 2000.
* [18]

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  arXiv preprint arXiv:1909.06312, 2019.
* [19]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In Advances in neural information processing systems, pages
  6638–6648, 2018.
* [20]

  J Ross Quinlan.
  Discovering rules by induction from large collections of examples.
  Expert Systems in the Micro electronics Age, 1979.
* [21]

  Mojtaba Seyedhosseini and Tolga Tasdizen.
  Disjunctive normal random forests.
  Pattern Recognition, 48(3):976–983, 2015.
* [22]

  Shai Shalev-Shwartz and Shai Ben-David.
  Understanding machine learning: From theory to algorithms.
  Cambridge university press, 2014.
* [23]

  Ira Shavitt and Eran Segal.
  Regularization learning networks: Deep learning for tabular datasets.
  In Advances in Neural Information Processing Systems, pages
  1386–1396, 2018.
* [24]

  Hans Ulrich Simon.
  On the number of examples and stages needed for learning decision
  trees.
  In Proceedings of the Third Annual Workshop on Computational
  Learning Theory, COLT ’90, page 303–313, San Francisco, CA, USA, 1990.
  Morgan Kaufmann Publishers Inc.
* [25]

  Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo.
  Openml: networked science in machine learning.
  ACM SIGKDD Explorations Newsletter, 15(2):49–60, 2014.
* [26]

  Yongxin Yang, Irene Garcia Morillo, and Timothy M Hospedales.
  Deep neural decision trees.
  arXiv preprint arXiv:1806.06988, 2018.
* [27]

  Jinsung Yoon, James Jordon, and Mihaela van der Schaar.
  Invase: Instance-wise variable selection using neural networks.
  2018.
* [28]

  Hui Zou and Trevor Hastie.
  Regularization and variable selection via the elastic net.
  Journal of the royal statistical society: series B (statistical
  methodology), 67(2):301–320, 2005.

## Apendices

## Appendix A OR and AND Gates

The (soft) neural OR and AND gates were defined as

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | OR​(𝐱)OR𝐱\displaystyle{\rm{OR}}(\mathbf{x}) | ≜≜\displaystyle\triangleq | tanh⁡(∑i=1d𝐱i+d−1.5),AND​(𝐱)≜tanh⁡(∑i=1d𝐱i−d+1.5).≜  superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1.5AND𝐱 superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1.5\displaystyle\tanh\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}+d-1.5\right),\hskip 50.0pt{\rm{AND}}(\mathbf{x})\triangleq\tanh\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}-d+1.5\right). |  |

By replacing the tanh\tanh activation with a signsign\operatorname\*{sign} activation, and setting the
bias term to 1 (instead of 1.5), we obtain exact
binary gates,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | OR​(𝐱)OR𝐱\displaystyle{\rm{OR}}(\mathbf{x}) | ≜≜\displaystyle\triangleq | sign(∑i=1d𝐱i+d−1),AND​(𝐱)≜sign(∑i=1d𝐱i−d+1).≜  signsuperscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1AND𝐱 signsuperscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1\displaystyle\operatorname\*{sign}\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}+d-1\right),\hskip 50.0pt{\rm{AND}}(\mathbf{x})\triangleq\operatorname\*{sign}\kern-2.15277pt\left(\sum\_{i=1}^{d}\mathbf{x}\_{i}-d+1\right). |  |

Consider a binary vector 𝐱∈{±1}d𝐱superscriptplus-or-minus1𝑑\mathbf{x}\in\{\pm 1\}^{d}. We prove that

|  |  |  |
| --- | --- | --- |
|  | AND​(𝐱)≡⋀i=1d𝐱i,AND𝐱superscriptsubscript𝑖1𝑑subscript𝐱𝑖{\rm{AND}}(\mathbf{x})\equiv\bigwedge\_{i=1}^{d}\mathbf{x}\_{i}, |  |

where, in the definition of the logical “and”, −11-1 is equivalent to 0.
If for any 1≤i≤d1𝑖𝑑1\leq i\leq d, 𝐱i=1subscript𝐱𝑖1\mathbf{x}\_{i}=1, then ∧i=1d𝐱i=1superscriptsubscript𝑖1𝑑subscript𝐱𝑖1\wedge\_{i=1}^{d}\mathbf{x}\_{i}=1.
Conversely, we have,

|  |  |  |
| --- | --- | --- |
|  | AND​(𝐱)=∑i=1d𝐱i−d+1=d−d+1=1,AND𝐱superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1𝑑𝑑11{\rm{AND}}(\mathbf{x})=\sum\_{i=1}^{d}\mathbf{x}\_{i}-d+1=d-d+1=1, |  |

and the application of the signsign\operatorname\*{sign} activation yields 1. In the case of the soft neural AND gate, we get t​a​n​h​(1)≈0.76𝑡𝑎𝑛ℎ10.76tanh(1)\approx 0.76; therefore, we set the bias term to 1.5 to get an output closer to 1 (t​a​n​h​(1.5)≈0.9𝑡𝑎𝑛ℎ1.50.9tanh(1.5)\approx 0.9).

Otherwise, there exists at least one index 1≤j≤d1𝑗𝑑1\leq j\leq d, such that 𝐱j=−1subscript𝐱𝑗1\mathbf{x}\_{j}=-1,
and ∧i=1d𝐱i=−1superscriptsubscript𝑖1𝑑subscript𝐱𝑖1\wedge\_{i=1}^{d}\mathbf{x}\_{i}=-1.
In this case,

|  |  |  |
| --- | --- | --- |
|  | AND​(𝐱)=∑i=1d𝐱i−d+1=𝐱j+∑i≠j𝐱i−d+1≤−1+(d−1)−d+1=−1,AND𝐱superscriptsubscript𝑖1𝑑subscript𝐱𝑖𝑑1subscript𝐱𝑗subscript𝑖𝑗subscript𝐱𝑖𝑑11𝑑1𝑑11{\rm{AND}}(\mathbf{x})=\sum\_{i=1}^{d}\mathbf{x}\_{i}-d+1=\mathbf{x}\_{j}+\sum\_{i\neq j}\mathbf{x}\_{i}-d+1\leq-1+(d-1)-d+1=-1, |  |

and by applying the signsign\operatorname\*{sign} activation we obtain −11-1.
This proves that the AND​(𝐱)AND𝐱{\rm{AND}}(\mathbf{x}) neuron is equivalent to a logical “AND” gate
in the binary case. A very similar proof shows that

|  |  |  |
| --- | --- | --- |
|  | OR​(𝐱)≡⋁i=1d𝐱i.OR𝐱superscriptsubscript𝑖1𝑑subscript𝐱𝑖{\rm{OR}}(\mathbf{x})\equiv\bigvee\_{i=1}^{d}\mathbf{x}\_{i}. |  |

## Appendix B Proof of Theorem 2

We bound the VC-dimension of a DNF formula in two steps. First, we derive an upper bound on the VC-dimension
of a single conjunction, and then extend it to a disjunction of k𝑘k conjunctions.
We use the following simple lemma.

###### Lemma 1.

For every two hypothesis classes, H′⊆Hsuperscript𝐻′𝐻H^{\prime}\subseteq H, it holds that V​C​D​i​m​(H′)≤V​C​D​i​m​(H)𝑉𝐶𝐷𝑖𝑚superscript𝐻′𝑉𝐶𝐷𝑖𝑚𝐻VCDim(H^{\prime})\leq VCDim(H).

###### Proof.

Let d=V​C​D​i​m​(H′)𝑑𝑉𝐶𝐷𝑖𝑚superscript𝐻′d=VCDim(H^{\prime}). By definition, there exist d𝑑d points that can be shattered by H′superscript𝐻′H^{\prime}.
Therefore, there exist 2dsuperscript2𝑑2^{d} hypotheses {hi′}i=12dsuperscriptsubscriptsubscriptsuperscriptℎ′𝑖𝑖1superscript2𝑑\{h^{\prime}\_{i}\}\_{i=1}^{2^{d}} in H′superscript𝐻′H^{\prime}, which shatter these points.
By assumption, {hi′}i=12d⊆Hsuperscriptsubscriptsubscriptsuperscriptℎ′𝑖𝑖1superscript2𝑑𝐻\{h^{\prime}\_{i}\}\_{i=1}^{2^{d}}\subseteq H, so V​C​D​i​m​(H)≥d𝑉𝐶𝐷𝑖𝑚𝐻𝑑VCDim(H)\geq d.
∎

For any conjunction on n𝑛n Boolean variables (regardless of the number of literals), it is possible to construct an equivalent decision tree of rank 1.
The construction is straightforward.
If ⋀i=1ℓxisuperscriptsubscript𝑖1ℓsubscript𝑥𝑖\bigwedge\_{i=1}^{\ell}x\_{i} is the conjunction,
the decision tree consists of a single main branch of ℓℓ\ell internal
decision nodes connected sequentially.
Each left child in this tree corresponds to decision “1”, and each right child corresponds to decision “0”.
The root is indexed 1 and contains the literal x1subscript𝑥1x\_{1}.
For 1≤i<ℓ1𝑖ℓ1\leq i<\ell, internal node i𝑖i contains the decision literal
xisubscript𝑥𝑖x\_{i} and its left child is node i+1𝑖1i+1 (whose decision literal is xi+1subscript𝑥𝑖1x\_{i+1}).
See the example in Figure [3](#A2.F3 "Figure 3 ‣ Appendix B Proof of Theorem 2 ‣ DNF-Net: A Neural Architecture for Tabular Data").

It follows that the hypothesis class of conjunctions is contained in the class of rank-111 decision trees.
Therefore, by Lemma [1](#Thmlemma1 "Lemma 1. ‣ Appendix B Proof of Theorem 2 ‣ DNF-Net: A Neural Architecture for Tabular Data") and Theorem [1](#Thmtheorem1 "Theorem 1 (Simon, [24]). ‣ 3 DNFs and Trees – A VC Analysis ‣ DNF-Net: A Neural Architecture for Tabular Data"), the
VC-dimension of conjunctions is bounded above by n+1𝑛1n+1.

We now derive the upper bound on the VC-dimension of a disjunction of k𝑘k conjunctions. Let C𝐶C be the
class of conjunctions, and let Dk​(C)subscript𝐷𝑘𝐶D\_{k}(C)
be the class of a disjunction of k𝑘k conjunctions.
Clearly, Dk​(C)subscript𝐷𝑘𝐶D\_{k}(C) is a k𝑘k-fold union of the class C𝐶C,
namely,

|  |  |  |
| --- | --- | --- |
|  | Dk​(C)={⋃i=0kci|ci∈C}.subscript𝐷𝑘𝐶conditional-setsuperscriptsubscript𝑖0𝑘subscript𝑐𝑖subscript𝑐𝑖𝐶D\_{k}(C)=\left\{\bigcup\_{i=0}^{k}c\_{i}\ |c\_{i}\in C\right\}. |  |

By Lemma 3.2.3 in Blummer et al. [[4](#bib.bib4)], if d=V​C​D​i​m​(C)𝑑𝑉𝐶𝐷𝑖𝑚𝐶d=VCDim(C), then for all k≥1𝑘1k\geq 1, V​C​D​i​m​(Dk​(C))≤2​d​k​log⁡(3​k)𝑉𝐶𝐷𝑖𝑚subscript𝐷𝑘𝐶2𝑑𝑘3𝑘VCDim(D\_{k}(C))\leq 2dk\log(3k). Therefore, for the class D​N​Fnk𝐷𝑁superscriptsubscript𝐹𝑛𝑘DNF\_{n}^{k}, of DNF formulas with k𝑘k conjunctions on n𝑛n Boolean variables, we have

|  |  |  |
| --- | --- | --- |
|  | V​C​D​i​m​(D​N​Fnk)≤2​(n+1)​k​log⁡(3​k).𝑉𝐶𝐷𝑖𝑚𝐷𝑁superscriptsubscript𝐹𝑛𝑘2𝑛1𝑘3𝑘VCDim(DNF\_{n}^{k})\leq 2(n+1)k\log(3k). |  |

!(/html/2006.06465/assets/conjuction_graph.jpg)

Figure 3: An example of a decision tree with rank 1, which is equivalent to the conjunction x0∧x1∧x2∧x3∧x4subscript𝑥0subscript𝑥1subscript𝑥2subscript𝑥3subscript𝑥4x\_{0}\wedge x\_{1}\wedge x\_{2}\wedge x\_{3}\wedge x\_{4}.

## Appendix C Tabular Dataset Description

We use datasets (See Table 4) that differ in several aspects such as in the number of features (from 16 up to 200), the number of classes (from 2 up to 9), and the number of samples (from 10k up to 200k). To keep things simple, we selected datasets with no missing values, and that do not require preprocessing. All models were trained on the raw data without any feature or data engineering and without any kind of data balancing or weighting. Only feature standardization was applied.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | features | classes | samples | source | link |
| Otto Group | 93 | 9 | 61.9k | Kaggle | kaggle.com/c/otto-group-product-classification-challenge/overview |
| Gesture Phase | 32 | 5 | 9.8k | OpenML | openml.org/d/4538 |
| Gas Concentrations | 129 | 6 | 13.9k | OpenML | openml.org/d/1477 |
| Eye Movements | 26 | 3 | 10.9k | OpenML | openml.org/d/1044 |
| Santander Transaction | 200 | 2 | 200k | Kaggle | kaggle.com/c/santander-customer-transaction-prediction/overview |
| House | 16 | 2 | 22.7k | OpenML | openml.org/d/821 |

Table 4: A description of the tabular datasets

## Appendix D Experimental Protocol

### D.1 Data Partition and Grid Search Procedure

All experiments in our work, using both synthetic and real datasets, were done through a grid search process. Each dataset was first randomly divided into five folds in a way that preserved the original distribution. Then, based on these five folds, we created five partitions of the dataset as follows. Each fold is used as the test set in one of the partitions, while the other folds are used as the training and validation sets. This way, each partition was 20%percent2020\% test, 10%percent1010\% validation, and 70%percent7070\% training. This division was done once 222We used seed number 1., and the same partitions were used for all models.
Based on these partitions, the following grid search process was repeated three times with three different seeds333We used seed numbers 1, 2, 3. (with the exact same five partitions as described before).

Input: model, configurations\_list

results\_list = [ ]

for *i=1 to n\_partitions* do

val\_scores\_list = [ ]

test\_scores\_list = [ ]

train, val, test = read\_data(partition\_index=i)

for *c in configurations\_list* do

trained\_model = model.train(train\_data=train, val\_data=val, configuration=c)

trained\_model.load\_weights\_from\_best\_epoch()

val\_score = trained\_model.predict(data=val)

test\_score = trained\_model.predict(data=test)

val\_scores\_list.append(val\_score)

test\_scores\_list.append(test\_score)

end for

best\_val\_index = get\_index\_of\_best\_val\_score(val\_scores\_list)

test\_res = test\_scores\_list[best\_val\_index]

results\_list.append(test\_res)

end for

mean = mean(results\_list)

sem = standard\_error\_of\_the\_mean(results\_list)

Return: mean, sem

Algorithm 1 Grid Search Procedure

The final mean and sem444For details, see: docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html that we presents in all experiments are the average across the three seeds.
Additionally, as can be seen from Algorithm [1](#algorithm1 "In D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data"), the model that was trained on the training set (70%percent7070\%) is the one that is used to evaluate performance on the test set (20%percent2020\%). This was done to keep things simple. The loading wights command is relevant for the neural network models. While for the XGBoost, the framework handles the optimal number of estimators on prediction time (accordingly to early stopping on training time).

### D.2 Training Protocol

The DNF-Net and the FCN were implemented using Tesnorflow [[1](#bib.bib1)]. To make a fair comparison, for both models, we used the same batch size555For DNF-Net , when using 3072 formulas, we set the batch size to 1024 on the Santander Transaction and Gas datasets and when using 2048 formulas, we set the batch size to 1024 on the Santander Transaction dataset. This was done due to memory issues. of 2048, and the same learning rate scheduler (reduce on plateau) that monitors the training loss. We set a maximum of 1000 epochs and used the same early stopping protocol (30 epochs) that monitors the validation score. Moreover, for both of them, we used the same loss function (softmax-cross-entropy for multi-class datasets and sigmoid-cross-entropy for binary datasets) and the same optimizer (Adam with default parameters).

For DNF-Net we used an initial learning rate of 0.050.050.05. For FCN, we added the initial learning rate to the grid search with values of {0.05,0.005,0.0005}0.050.0050.0005\{0.05,0.005,0.0005\}.

For XGBoost [[6](#bib.bib6)], we set the maximal number of estimators to be 2500, and used an early stopping of 50 estimators that monitors the validation score.

All models were trained on GPUs - Titan Xp 12GB RAM.

Additionally, in the case of DNF-Net, we took a symmetry-breaking approach between the different DNNFs. This is reflected by the DNNF group being divided equally into four subgroups where, for each subgroup, the number of conjunctions is equal to one of the following values [6,9,12,15]

691215[6,9,12,15], and the group of conjunctions of each DNNF was divided equally into three subgroups where, for each subgroup, the conjunction length is equal to one of the following values [2,4,6]

246[2,4,6].
The same approach was used for the parameter p𝑝p of the random mask. The DNNF group was divided equally into five subgroups where, for each subgroup, p𝑝p is equal to one of the following values [0.1,0.3,0.5,0.7,0.9]

0.10.30.50.70.9[0.1,0.3,0.5,0.7,0.9]. In all experiments we used the same values.

### D.3 Grid Parameters – Tabular Datasets

#### D.3.1 DNF-Net

|  |  |
| --- | --- |
| DNF-Net (42 configs) | |
| hyperparameter | values |
| n. formulas | {64,128,256,512,1024,2048,3072}64128256512102420483072\{64,128,256,512,1024,2048,3072\} |
| feature selection beta | {1.6,1.3,1.,0.7,0.4,0.1}\{1.6,1.3,1.,0.7,0.4,0.1\} |

#### D.3.2 XGBoost

|  |  |
| --- | --- |
| XGBoost (864 configs) | |
| hyperparameter | values |
| n. estimators | {2500}2500\{2500\} |
| learning rate | {0.001,0.005,0.01,0.05,0.1,0.5}0.0010.0050.010.050.10.5\{0.001,0.005,0.01,0.05,0.1,0.5\} |
| max depth | {2,3,4,5,7,9,11,13,15}234579111315\{2,3,4,5,7,9,11,13,15\} |
| colsample by tree | {0.25,0.5,0.75,1.}\{0.25,0.5,0.75,1.\} |
| sub sample | {0.25,0.5,0.75,1.}\{0.25,0.5,0.75,1.\} |

#### D.3.3 Fully Connected Networks

The FCN networks are constructed using Dense-RELU-Dropout blocks with L2subscript𝐿2L\_{2} regularization. The network’s blocks are defined in the following way. Given depth and width parameters, we examine two different configurations: (1) the same width is used for the entire network (e.g., if the width is 512 and the depth is four, then the network blocks are [512, 512, 512, 512]), and (2) the width parameter defines the width of the first block, and the subsequent blocks are reduced by a factor of 2 (e.g., if the width is 512 and the depth is four, then the network blocks are [512, 256, 128, 64]). On top of the last block we add a simple linear layer that reduce the dimension into the output dimension. The dropout and L2subscript𝐿2L\_{2} values are the same for all blocks.

|  |  |
| --- | --- |
| FCN (3300 configs) | |
| hyperparameter | values |
| depth | {1,2,3,4,5,6}123456\{1,2,3,4,5,6\} |
| width | {128,256,512,1024,2048}12825651210242048\{128,256,512,1024,2048\} |
| L2subscript𝐿2L\_{2} lambda | {10−2,10−4,10−6,10−8,0.}\{10^{-2},10^{-4},10^{-6},10^{-8},0.\} |
| dropout | {0.,0.25,0.5,0.75}\{0.,0.25,0.5,0.75\} |
| initial learning rate | {0.05,0.005,0.0005}0.050.0050.0005\{0.05,0.005,0.0005\} |

### D.4 Ablation Study

All ablation studies experiments were conducted using the grid search process as described in [D.1](#A4.SS1 "D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data"). In all experiments, we used the same training details as described on [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data") for DNF-Net. Where the only difference between the different experiments is the addition or removal of the components.

The single hyperparameter that was fine-tuned using the grid search is the ‘feature selection beta’ on the range {1.6,1.3,1.,0.7,0.4,0.1}\{1.6,1.3,1.,0.7,0.4,0.1\}, in experiments in which the feature selection component is involved. In the other cases, only one configuration was tested in the grid search process for a specific number of formulas.

### D.5 Feature Selection Analysis

The input features 𝐱∈ℝd𝐱superscriptℝ𝑑\mathbf{x}\in\mathbb{R}^{d} of all six datasets were generated from a d𝑑d-dimensional Gaussian distribution with no correlation across the features, 𝐱∼ℕ​(0,I)similar-to𝐱ℕ0𝐼\mathbf{x}\sim\mathbb{N}(0,I). The label 𝐲𝐲\mathbf{y} is sampled as a Bernoulli random variable with ℙ​(𝐲=1|𝐱)=11+l​o​g​i​t​(𝐱)ℙ𝐲conditional1𝐱11𝑙𝑜𝑔𝑖𝑡𝐱\mathbb{P}(\mathbf{y}=1|\mathbf{x})=\frac{1}{1+logit(\mathbf{x})}, where l​o​g​i​t​(𝐱)𝑙𝑜𝑔𝑖𝑡𝐱logit(\mathbf{x}) is varied to create the different synthetic datasets (𝐱isubscript𝐱𝑖\mathbf{x}\_{i} refers to the i𝑖ith entry):

1. 1.

   Syn1: l​o​g​i​t​(𝐱)=e​x​p​(𝐱1​𝐱2)𝑙𝑜𝑔𝑖𝑡𝐱𝑒𝑥𝑝subscript𝐱1subscript𝐱2logit(\mathbf{x})=exp(\mathbf{x}\_{1}\mathbf{x}\_{2})
2. 2.

   Syn2: l​o​g​i​t​(𝐱)=e​x​p​(∑i=36𝐱i2−4)𝑙𝑜𝑔𝑖𝑡𝐱𝑒𝑥𝑝superscriptsubscript𝑖36superscriptsubscript𝐱𝑖24logit(\mathbf{x})=exp(\sum\_{i=3}^{6}\mathbf{x}\_{i}^{2}-4)
3. 3.

   Syn3: l​o​g​i​t​(𝐱)=−10​sin⁡(2​𝐱7)+2​|𝐱8|+𝐱9+e​x​p​(−𝐱10)−2.4𝑙𝑜𝑔𝑖𝑡𝐱102subscript𝐱72subscript𝐱8subscript𝐱9𝑒𝑥𝑝subscript𝐱102.4logit(\mathbf{x})=-10\sin(2\mathbf{x}\_{7})+2|\mathbf{x}\_{8}|+\mathbf{x}\_{9}+exp(-\mathbf{x}\_{10})-2.4
4. 4.

   Syn4: if 𝐱11<0subscript𝐱110\mathbf{x}\_{11}<0, logit follows Syn1, else, logit follows Syn2
5. 5.

   Syn5: if 𝐱11<0subscript𝐱110\mathbf{x}\_{11}<0, logit follows Syn1, else, logit follows Syn3
6. 6.

   Syn6: if 𝐱11<0subscript𝐱110\mathbf{x}\_{11}<0, logit follows Syn2, else, logit follows Syn3

We compare the performance of a basic FCN on three different cases: (1) oracle (ideal) feature selection – where the input feature vector is multiplied element-wise with an input oracle mask, whose i𝑖ith entry equals 1 iff the i𝑖ith feature is relevant (e.g., on Syn1, features 1 and 2 are relevant, and on Syn4, features 1-6, and 11 are relevant), (2) our (learned) feature selection mask – where the input feature vector is multiplied element-wise with the mask 𝐦tsubscript𝐦𝑡\mathbf{m}\_{t}, i.e., the entries of the mask 𝐦ssubscript𝐦𝑠\mathbf{m}\_{s} (see Section [2.3](#S2.SS3 "2.3 Feature Selection ‣ 2 Disjunctive Normal Form Networks (DNF-Nets) ‣ DNF-Net: A Neural Architecture for Tabular Data"))
are all fixed to 1, and (3) no feature selection.

From each dataset, we generated seven different instances that differ in their input size,
  
d∈[11,50,100,150,200,250,300]𝑑

1150100150200250300d\in[11,50,100,150,200,250,300]. Where when the input dimension d𝑑d increases, the same logit function is used. Each instance contains 10k samples that were partitioned as described in Section [D.1](#A4.SS1 "D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data"). We treated each instance as an independent dataset, and the grid search process that is described in Section [D.1](#A4.SS1 "D.1 Data Partition and Grid Search Procedure ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data") was done for each one.

The FCN that we used has two dense hidden layers [64, 32] with a RELU activation. To keep things simple, we have not used drouput or any kind of regularization.
The same training protocol was used for all three models. We used the same learning rate scheduler, early stopping protocol, loss function and optimizer as appear in Section [D.2](#A4.SS2 "D.2 Training Protocol ‣ Appendix D Experimental Protocol ‣ DNF-Net: A Neural Architecture for Tabular Data")666We noticed that in this scenario, a large learning rate or large batch size leads to a decline in the performance of the ’FCN with the feature selection’. While the simple FCN and the ’FCN with oracle mask’ remains approximately the same.. We use a batch size of 256, and an initial learning rate of 0.001. The only hyperparameter that was fine-tuned is the ‘feature selection beta’ in the case of ‘FCN with feature selection’ on the range {1.3,1.,0.7,0.4}\{1.3,1.,0.7,0.4\}. For the two other models, only a single configuration was tested in the grid search process.
