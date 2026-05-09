---
arxiv: '1807.06699'
authors:
- Ryutaro Tanno
- Kai Arulkumaran
- Daniel C. Alexander
- Antonio Criminisi
- Aditya Nori
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Adaptive Neural Trees
url: http://arxiv.org/abs/1807.06699v5
year: 2018
---

[1807.06699] Adaptive Neural Trees















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



# Adaptive Neural Trees

Ryutaro Tanno
  
Kai Arulkumaran
  
Daniel C. Alexander
  
Antonio Criminisi
  
Aditya Nori

# Supplementary Material

Ryutaro Tanno
  
Kai Arulkumaran
  
Daniel C. Alexander
  
Antonio Criminisi
  
Aditya Nori

###### Abstract

Deep neural networks and decision trees operate on largely separate paradigms; typically, the former performs representation learning with pre-specified architectures, while the latter is characterised by learning hierarchies over pre-specified features with data-driven architectures. We unite the two via *adaptive neural trees* (ANTs) that incorporates representation learning into edges, routing functions and leaf nodes of a decision tree, along with a backpropagation-based training algorithm that adaptively grows the architecture from primitive modules (e.g., convolutional layers). We demonstrate that, whilst achieving competitive performance on classification and regression datasets, ANTs benefit from (i) lightweight inference via conditional computation, (ii) hierarchical separation of features useful to the task e.g. learning meaningful class associations, such as separating natural vs. man-made objects, and (iii) a mechanism to adapt the architecture to the size and complexity of the training dataset.

neural networks, decision trees, regression, classification, computer vision

## 1 Introduction

Neural networks (NNs) and decision trees (DTs) are both powerful classes of machine learning models with proven successes in academic and commercial applications. The two approaches, however, typically come with mutually exclusive benefits and limitations.

NNs are characterised by learning hierarchical representations of data through the composition of nonlinear transformations (Zeiler & Fergus, [2014](#bib.bib72); Bengio, [2013](#bib.bib3)), which has alleviated the need for feature engineering, in contrast with many other machine learning models. In addition, NNs are trained with stochastic optimisers, such as stochastic gradient descent (SGD), allowing training to scale to large datasets. Consequently, with modern hardware, we can train NNs of many layers on large datasets, solving numerous problems ranging from object detection to speech recognition with unprecedented accuracy (LeCun et al., [2015](#bib.bib39)). However, their architectures typically need to be designed by hand and fixed per task or dataset, requiring domain expertise (Zoph & Le, [2017](#bib.bib75)). Inference can also be heavy-weight for large models, as each sample engages every part of the network, i.e., increasing capacity causes a proportional increase in computation (Bengio et al., [2013](#bib.bib4)).

Alternatively, DTs are characterised by learning hierarchical clusters of data (Criminisi & Shotton, [2013](#bib.bib10)). A DT learns how to split the input space, so that in each subset, linear models suffice to explain the data. In contrast to standard NNs, the architectures of DTs are optimised based on training data, and are particularly advantageous in data-scarce scenarios. DTs also enjoy lightweight inference as only a single root-to-leaf path on the tree is used for each input sample. However, successful applications of DTs often require hand-engineered features of data. We can ascribe the limited expressivity of single DTs to the common use of simplistic routing functions, such as splitting on axis-aligned features. The loss function for optimising hard partitioning is non-differentiable, which hinders the use of gradient descent-based optimization and thus complex splitting functions. Current techniques for increasing capacity include ensemble methods such as random forests (RFs) (Breiman, [2001](#bib.bib5)) and gradient-boosted trees (GBTs) (Friedman, [2001](#bib.bib16)), which are known to achieve state-of-the-art performance in various tasks, including medical applications and financial forecasting (Sandulescu & Chiru, [2016](#bib.bib55); Kaggle.com, [2017](#bib.bib31); Le Folgoc et al., [2016](#bib.bib37); Volkovs et al., [2017](#bib.bib69)).

The goal of this work is to combine NNs and DTs to gain the complementary benefits of both approaches. To this end, we propose adaptive neural trees (ANTs), which generalise previous work that attempted the same unification (Suárez & Lutsko, [1999](#bib.bib64); İrsoy et al., [2012](#bib.bib28); Laptev & Buhmann, [2014](#bib.bib36); Rota Bulo & Kontschieder, [2014](#bib.bib52); Kontschieder et al., [2015](#bib.bib34); Frosst & Hinton, [2017](#bib.bib17); Xiao, [2017](#bib.bib70)) and address their limitations (see Tab. [1](#S2.T1 "Table 1 ‣ 2 Related work ‣ Adaptive Neural Trees")).
ANTs represent routing decisions and root-to-leaf computational paths within the tree structures as NNs, which lets them benefit from hierarchical representation learning, rather than being restricted to partitioning the raw data space. On the other hand, unlike the fully distributed representaion of standard NN models, the tree topology of ANTs acts as a strong structural prior that enforces sparse structures by which features are shared and separated in a hierarchical fashion. In addition, we propose a backpropagation-based training algorithm to grow ANTs based on a series of decisions between making the ANT deeper—the central NN paradigm—or partitioning the data—the central DT paradigm (see Fig. [1](#S3.F1 "Figure 1 ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees") (Right)). This allows the architectures of ANTs to adapt to the data available. By our design, ANTs inherit the following desirable properties from both DTs and NNs:

* •

  Representation learning: as each root-to-leaf path in an ANT is an NN, features can be learned end-to-end with gradient-based optimisation. Combined with the tree structure, an ANT can learn such features which are hierarchically shared and separated.
* •

  Architecture learning: by progressively growing ANTs, the architecture adapts to the availability and complexity of data, embodying Occam’s razor. The growth procedure can be viewed as architecture search with a hard constraint over the model class.
* •

  Lightweight inference: at inference time, ANTs perform conditional computation, selecting a single root-to-leaf path on the tree on a per-sample basis, activating only a subset of the parameters of the model.

We empirically validate these benefits for regression and classification through experiments on the SARCOS (Vijayakumar & Schaal, [2000](#bib.bib68)), MNIST (LeCun et al., [1998](#bib.bib38)) and CIFAR-10 (Krizhevsky & Hinton, [2009](#bib.bib35)) datasets. The best performing methods on the SARCOS multivariate regression dataset are all tree-based, with ANTs achieving the lowest mean squared error. On the other hand, along with other forms of neural networks, ANTs far outperform state-of-the-art RF (Zhou & Feng, [2017](#bib.bib74)) and GBT (Ponomareva et al., [2017](#bib.bib49)) methods on image classification, with architectures achieving over 99% accuracy on MNIST and over 90% accuracy on CIFAR-10.
Our ablations on all three datasets consistently show that the combination of feature learning and data partitioning are required for the best predictive performance of ANTs. In addition, we show that ANTs can learn meaningful hierarchical partitionings of data, e.g., grouping man-made and natural objects (see Fig. [2](#S5.F2 "Figure 2 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees")) useful to the end task. ANTs also have reduced time and memory requirements during inference, thanks to such hierarchical structure. In one case, we discover an architecture that achieves over 98%percent9898\% accuracy on MNIST using approximately the same number of parameters as a linear classifier on raw image pixels, showing the benefits of tree-shaped hierarchical sharing and separation of features in enhancing both computational and predictive performance. Finally, we demonstrate the benefits of architecture learning by training ANTs on subsets of CIFAR-10 of varying sizes. The method can construct architectures of adequate size, leading to better generalisation, particularly on small datasets.

## 2 Related work

Our work is primarily related to research into combining DTs and NNs. Here we explain how ANTs subsume a large body of such prior work as specific cases and address their limitations. We include additional reviews of work in conditional computation and neural architecture search in Sec.B in the supplementary material.

The very first soft decition tree (SDT) introduced by Suárez & Lutsko ([1999](#bib.bib64)) is a specific case where in our terminology the routers are axis-aligned features, the transformers are identity functions, and the routers are static distributions over classes or linear functions. The hierarchical mixture of experts (HMEs) proposed by Jordan & Jacobs ([1994](#bib.bib30)) is a variant of SDTs whose routers are linear classifiers and the tree structure is fixed; Léon & Denoyer ([2015](#bib.bib42)) recently proposed a more computationally efficient training method that is able to directly optimise hard-partitioning by differentiating through stochastic gradient estimators. More modern SDTs (Rota Bulo & Kontschieder, [2014](#bib.bib52); Laptev & Buhmann, [2014](#bib.bib36); Frosst & Hinton, [2017](#bib.bib17)) have used multilayer perceptrons (MLPs) or convolutional layers in the routers to learn more complex partitionings of the input space. However, the simplicity of identity transformers used in these methods means that input data is never transformed and thus each path on the tree does not perform representation learning, limiting their performance.

More recent work suggested that integrating non-linear transformations of data into DTs would enhance model performance. The neural decision forest (NDF) (Kontschieder et al., [2015](#bib.bib34)), which held cutting-edge performance on ImageNet (Deng et al., [2009](#bib.bib12)) in 2015, is an ensemble of DTs, each of which is also an instance of ANTs where the whole GoogLeNet architecture (Szegedy et al., [2015](#bib.bib65)) (except for the last linear layer) is used as the root transformer, prior to learning tree-structured classifiers with linear routers. Xiao ([2017](#bib.bib70)) employed a similar approach with a MLP at the root transformer, and is optimised to minimise a differentiable information gain loss. The conditional network proposed by Ioannou et al. ([2016](#bib.bib26)) sparsified CNN architectures by distributing computations on hierarchical structures based on directed acyclic graphs with MLP-based routers, and designed models with the same accuracy with reduced compute cost and number of parameters. However, in all cases, the model architectures are pre-specified and fixed.

In contrast, ANTs satisfy all criteria in Tab. [1](#S2.T1 "Table 1 ‣ 2 Related work ‣ Adaptive Neural Trees"); they provide a general framework for learning tree-structured models with the capacity of representation learning along each path and within routing functions, and a mechanism for learning its architecture.

Table 1: Comparison of tree-structured NNs. The first column denotes if each path on the tree is a NN, and the second column denotes if the routers learn features. The last column shows if the method grows an architecture, or uses a pre-specified one.

|  |  |  |  |
| --- | --- | --- | --- |
| Method | Feature learning? | | Grown? |
|  | Path | Routers |  |
| SDT (Suárez & Lutsko, [1999](#bib.bib64)) | ✗ | ✗ | ✓ |
| SDT 2 / HME (Jordan & Jacobs, [1994](#bib.bib30)) | ✗ | ✓ | ✗ |
| SDT 3 (İrsoy et al., [2012](#bib.bib28)) | ✗ | ✓ | ✓ |
| SDT 4 (Frosst & Hinton, [2017](#bib.bib17)) | ✗ | ✓ | ✗ |
| RDT (Léon & Denoyer, [2015](#bib.bib42)) | ✗ | ✓ | ✗ |
| BT (İrsoy et al., [2014](#bib.bib29)) | ✗ | ✓ | ✓ |
| Conv DT (Laptev & Buhmann, [2014](#bib.bib36)) | ✗ | ✓ | ✗ |
| NDT (Rota Bulo & Kontschieder, [2014](#bib.bib52)) | ✗ | ✓ | ✓ |
| NDT 2 (Xiao, [2017](#bib.bib70)) | ✓ | ✓ | ✗ |
| NDF (Kontschieder et al., [2015](#bib.bib34)) | ✓ | ✓ | ✗ |
| CNet (Ioannou et al., [2016](#bib.bib26)) | ✓ | ✓ | ✗ |
| ANT (ours) | ✓ | ✓ | ✓ |

Architecture growth is a key facet of DTs (Criminisi & Shotton, [2013](#bib.bib10)), and typically performed in a greedy fashion with a termination criteria based on validation set error (Suárez & Lutsko, [1999](#bib.bib64); İrsoy et al., [2012](#bib.bib28)). Previous works in DT research have made attempts to improve upon this greedy growth strategy. Decision jungles (Shotton et al., [2013](#bib.bib59)) employ a training mechanism to merge partitioned input spaces between different sub-trees, and thus to rectify suboptimal “splits” made due to the locality of optimisation. İrsoy et al. ([2014](#bib.bib29)) proposes budding trees, which are grown and pruned incrementally based on global optimisation of existing nodes. While our training algorithm, for simplicity, grows the architecture by
greedily choosing the best option between going “deeper” and “splitting” the input space (see Fig. [1](#S3.F1 "Figure 1 ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")), it is certainly amenable to these advances.

## 3 Adaptive Neural Trees

We now formalise the definition of Adaptive Neural Trees (ANTs), which are a form of DTs enhanced with deep, learned representations. We focus on supervised learning, where the aim is to learn the conditional distribution p​(𝐲|𝐱)𝑝conditional𝐲𝐱p(\mathbf{y}|\mathbf{x}) from a set of N𝑁N labelled samples (𝐱(1),𝐲(1)),…,(𝐱(N),𝐲(N))∈𝒳×𝒴

superscript𝐱1superscript𝐲1…superscript𝐱𝑁superscript𝐲𝑁
𝒳𝒴(\mathbf{x}^{(1)},\mathbf{y}^{(1)}),...,(\mathbf{x}^{(N)},\mathbf{y}^{(N)})\in\mathcal{X}\times\mathcal{Y} as training data.

![Refer to caption](/html/1807.06699/assets/figures/fig_4_3.png)

![Refer to caption](/html/1807.06699/assets/figures/fig_5_6.png)

Figure 1: (Left). An example ANT. Data is passed through transformers (black circles on edges), routers (white circles on internal nodes), and solvers (gray circles on leaf nodes). The red shaded path shows routing of 𝐱𝐱\mathbf{x} to reach leaf node 444. Input 𝐱𝐱\mathbf{x} undergoes a series of selected transformations 𝐱→𝐱0𝝍:=t0𝝍​(𝐱)→𝐱1𝝍:=t1𝝍​(𝐱0𝝍)→𝐱4𝝍:=t4𝝍​(𝐱1𝝍)→𝐱subscriptsuperscript𝐱𝝍0assignsubscriptsuperscript𝑡𝝍0𝐱→subscriptsuperscript𝐱𝝍1assignsubscriptsuperscript𝑡𝝍1subscriptsuperscript𝐱𝝍0→subscriptsuperscript𝐱𝝍4assignsubscriptsuperscript𝑡𝝍4subscriptsuperscript𝐱𝝍1\mathbf{x}\rightarrow\mathbf{x}^{\boldsymbol{\psi}}\_{0}:=t^{\boldsymbol{\psi}}\_{0}(\mathbf{x})\rightarrow\mathbf{x}^{\boldsymbol{\psi}}\_{1}:=t^{\boldsymbol{\psi}}\_{1}(\mathbf{x}^{\boldsymbol{\psi}}\_{0})\rightarrow\mathbf{x}^{\boldsymbol{\psi}}\_{4}:=t^{\boldsymbol{\psi}}\_{4}(\mathbf{x}^{\boldsymbol{\psi}}\_{1}) and the solver module yields the predictive distribution p4ϕ,𝝍​(𝐲):=s4ϕ​(𝐱4𝝍)assignsuperscriptsubscript𝑝4

bold-italic-ϕ𝝍𝐲subscriptsuperscript𝑠bold-italic-ϕ4subscriptsuperscript𝐱𝝍4p\_{4}^{\boldsymbol{\phi},\boldsymbol{\psi}}(\mathbf{y}):=s^{\boldsymbol{\phi}}\_{4}(\mathbf{x}^{\boldsymbol{\psi}}\_{4}). The probability of selecting this path is given by π2𝝍,𝜽​(𝐱):=r0𝜽​(𝐱0𝝍)⋅(1−r1𝜽​(𝐱1𝝍)).assignsuperscriptsubscript𝜋2

𝝍𝜽𝐱⋅superscriptsubscript𝑟0𝜽subscriptsuperscript𝐱𝝍01superscriptsubscript𝑟1𝜽subscriptsuperscript𝐱𝝍1\pi\_{2}^{\boldsymbol{\psi},\boldsymbol{\theta}}(\mathbf{x}):=r\_{0}^{\boldsymbol{\theta}}(\mathbf{x}^{\boldsymbol{\psi}}\_{0})\cdot(1-r\_{1}^{\boldsymbol{\theta}}(\mathbf{x}^{\boldsymbol{\psi}}\_{1})). (Right). Three growth options at a given node: split data, deepen transform & keep. The small white circles on the edges denote identity transformers.

### 3.1 Model Topology and Operations

In short, an ANT is a tree-structured model, characterized by a set of hierarchical partitions of the input space 𝒳𝒳\mathcal{X}, a series of nonlinear transformations, and separate predictive models in the respective component regions. More formally, we define an ANT as a pair (𝕋,𝕆)𝕋𝕆(\mathbb{T},\mathbb{O}) where 𝕋𝕋\mathbb{T} defines the model topology, and 𝕆𝕆\mathbb{O} denotes the set of operations on it.

Table 2: Primitive module specifications for MNIST, CIFAR-10 and SARCOS datasets. “conv5-40” denotes a 2D convolution with 40 kernels of spatial size 5×5555\times 5. “GAP”, “FC”, “LC” and “LR” stand for global-average-pooling, fully connected layer, linear classifier and linear regressor. “Downsample Freq” denotes the frequency at which 2×2222\times 2 max-pooling is applied.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Model | Router, ℛℛ\mathcal{R} | Transformer, 𝒯𝒯\mathcal{T} | Solver, 𝒮𝒮\mathcal{S} | Downsample Freq. |
| ANT-SARCOS | 1×FC1FC1\times\text{FC} +Sigmoid | 1×1\timesFC+ tanh | LR | 0 |
| ANT-MNIST-A | 1×conv5-401conv5-401\times\text{conv5-40} + GAP + 2×2\timesFC +Sigmoid | 1×conv5-401conv5-401\times\text{conv5-40} + ReLU | LC | 1 |
| ANT-MNIST-B | 1×conv3-401conv3-401\times\text{conv3-40} + GAP + 2×2\timesFC +Sigmoid | 1×conv3-401conv3-401\times\text{conv3-40} +ReLU | LC | 2 |
| ANT-MNIST-C | 1×conv5-51conv5-51\times\text{conv5-5} + GAP + 2×2\timesFC +Sigmoid | 1×conv5-51conv5-51\times\text{conv5-5}+ReLU | LC | 2 |
| ANT-CIFAR10-A | 2×conv3-1282conv3-1282\times\text{conv3-128} + GAP + 1×1\timesFC +Sigmoid | 2×conv3-1282conv3-1282\times\text{conv3-128} +ReLU | GAP + LC | 1 |
| ANT-CIFAR10-B | 2×conv3-962conv3-962\times\text{conv3-96} + GAP + 1×1\timesFC +Sigmoid | 2×conv3-962conv3-962\times\text{conv3-96} +ReLU | LC | 1 |
| ANT-CIFAR10-C | 2×conv3-482conv3-482\times\text{conv3-48} + GAP + 1×1\timesFC +Sigmoid | 2×conv3-962conv3-962\times\text{conv3-96} +ReLU | GAP + LC | 1 |

We restrict the model topology 𝕋𝕋\mathbb{T} to be instances of binary trees, defined as a set of graphs whose each node is either an internal node or a leaf, and is the child of exactly one parent node, except the root node at the top. We define the topology of a tree as 𝕋:={𝒩,ℰ}assign𝕋𝒩ℰ\mathbb{T}:=\{\mathcal{N},\mathcal{E}\} where 𝒩𝒩\mathcal{N} is the set of all nodes, and ℰℰ\mathcal{E} is the set of edges between them. Nodes with no children are leaf nodes, 𝒩l​e​a​fsubscript𝒩𝑙𝑒𝑎𝑓\mathcal{N}\_{leaf}, and all others are internal nodes, 𝒩i​n​tsubscript𝒩𝑖𝑛𝑡\mathcal{N}\_{int}. Every internal node j∈𝒩i​n​t𝑗subscript𝒩𝑖𝑛𝑡j\in\mathcal{N}\_{int} has exactly two children nodes, represented by left​(j)left𝑗\mathrm{left}(j) and right​(j)right𝑗\mathrm{right}(j). Unlike standard trees, ℰℰ\mathcal{E} contains an edge which connects input data 𝐱𝐱\mathbf{x} with the root node, as shown in Fig.[1](#S3.F1 "Figure 1 ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees") (Left).

Every node and edge is assigned with operations which acts on the allocated samples of data (Fig.[1](#S3.F1 "Figure 1 ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")). Starting at the root, each sample gets transformed and traverses the tree according to the set of operations 𝕆𝕆\mathbb{O}. An ANT is constructed based on three primitive modules of differentiable operations:

1. 1.

   Routers, ℛℛ\mathcal{R}: each internal node j∈𝒩i​n​t𝑗subscript𝒩𝑖𝑛𝑡j\in\mathcal{N}\_{int} holds a router module, rj𝜽:𝒳j→[0,1]∈ℛ:superscriptsubscript𝑟𝑗𝜽→subscript𝒳𝑗01ℛr\_{j}^{\mathbf{\boldsymbol{\theta}}}:\mathcal{X}\_{j}\rightarrow[0,1]\in\mathcal{R}, parametrised by 𝜽𝜽\boldsymbol{\theta}, which sends samples from the incoming edge to either the left or right child. Here 𝒳jsubscript𝒳𝑗\mathcal{X}\_{j} denotes the representation at node j𝑗j. We use stochastic routing, where the decision (111 for the left and 00 for the right branch) is sampled from Bernoulli distribution with mean rj𝜽​(𝐱j)superscriptsubscript𝑟𝑗𝜽subscript𝐱𝑗r\_{j}^{\boldsymbol{\theta}}(\mathbf{x}\_{j}) for input 𝐱j∈𝒳jsubscript𝐱𝑗subscript𝒳𝑗\mathbf{x}\_{j}\in\mathcal{X}\_{j}. As an example, rj𝜽superscriptsubscript𝑟𝑗𝜽r\_{j}^{\mathbf{\boldsymbol{\theta}}} can be defined as a small CNN.
2. 2.

   Transformers, 𝒯𝒯\mathcal{T}: every edge e∈ℰ𝑒ℰe\in\mathcal{E} of the tree has one or a composition of multiple transformer module(s). Each transformer te𝝍∈𝒯superscriptsubscript𝑡𝑒𝝍𝒯t\_{e}^{\boldsymbol{\psi}}\in\mathcal{T} is a nonlinear function, parametrised by 𝝍𝝍\boldsymbol{\psi}, that transforms samples from the previous module and passes them to the next one. For example, te𝝍superscriptsubscript𝑡𝑒𝝍t\_{e}^{\boldsymbol{\psi}} can be a single convolutional layer followed by ReLU (Nair & Hinton, [2010](#bib.bib46)). Unlike in standard DTs, edges transform data and are allowed to “grow” by adding more operations (Sec. [4](#S4 "4 Optimisation ‣ Adaptive Neural Trees")), learning “deeper” representations as needed.
3. 3.

   Solvers, 𝒮𝒮\mathcal{S}: each leaf node l∈𝒩l​e​a​f𝑙subscript𝒩𝑙𝑒𝑎𝑓l\in\mathcal{N}\_{leaf} is assigned to a solver module, slϕ:𝒳l→𝒴∈𝒮:superscriptsubscript𝑠𝑙bold-italic-ϕ→subscript𝒳𝑙𝒴𝒮s\_{l}^{\boldsymbol{\phi}}:\mathcal{X}\_{l}\rightarrow\mathcal{Y}\in\mathcal{S}, parametrised by ϕbold-italic-ϕ\boldsymbol{\phi}, which operates on the transformed input data and outputs an estimate for the conditional distribution p​(𝐲|𝐱)𝑝conditional𝐲𝐱p(\mathbf{y}|\mathbf{x}). For classification tasks, we can define, for example, sϕsuperscript𝑠bold-italic-ϕs^{\boldsymbol{\phi}} as a linear classifier on the feature space 𝒳lsubscript𝒳𝑙\mathcal{X}\_{l}, which outputs a distribution over classes.

Defining operations on the graph 𝕋𝕋\mathbb{T} amounts to a specification of the triplet 𝕆=(ℛ,𝒯,𝒮)𝕆ℛ𝒯𝒮\mathbb{O}=(\mathcal{R},\mathcal{T},\mathcal{S}). For example, given image inputs, we would choose the operations of each module to be from the set of operations commonly used in CNNs (examples are given in Tab. [2](#S3.T2 "Table 2 ‣ 3.1 Model Topology and Operations ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")).
In this case, every computational path on the resultant ANT, as well as the set of routers that guide inputs to one of these paths, are given by CNNs. Lastly, many existing tree-structured models (Suárez & Lutsko, [1999](#bib.bib64); İrsoy et al., [2012](#bib.bib28); Laptev & Buhmann, [2014](#bib.bib36); Rota Bulo & Kontschieder, [2014](#bib.bib52); Kontschieder et al., [2015](#bib.bib34); Frosst & Hinton, [2017](#bib.bib17); Xiao, [2017](#bib.bib70)) are instantiations of ANTs with limitations which we will address with our model (see Sec. [2](#S2 "2 Related work ‣ Adaptive Neural Trees") for a more detailed discussion).

### 3.2 Probabilistic model and inference

An ANT models the conditional distribution p​(𝐲|𝐱)𝑝conditional𝐲𝐱p(\mathbf{y}|\mathbf{x}) as a hierarchical mixture of experts (HMEs) (Jordan & Jacobs, [1994](#bib.bib30)), each of which is defined as an NN and is a root-to-leaf path in the tree. Standard HMEs are a special case of ANTs where transformers are the identity function. As a result, the representations within experts are hierarchically shared between similar experts, unlike the independent representations within experts in standard HMEs. In addition, ANTs come with a growth mechanism to determine the number of needed experts and their complexity, as discussed in Sec. [4](#S4 "4 Optimisation ‣ Adaptive Neural Trees").

Each input 𝐱𝐱\mathbf{x} to the ANT stochastically traverses the tree based on decisions of routers and undergoes a sequence of transformations until it reaches a leaf node where the corresponding solver predicts the label 𝐲𝐲\mathbf{y}. Suppose we have L𝐿L leaf nodes, the full predictive distribution, with parameters Θ=(𝜽,𝝍,ϕ)Θ𝜽𝝍bold-italic-ϕ\Theta=(\boldsymbol{\theta},\boldsymbol{\psi},\boldsymbol{\phi}), is given by

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(𝐲|𝐱,Θ)=∑l=1Lp​(zl=1|𝐱,𝜽,𝝍)⏟Leaf-assignment prob. ​πl𝜽,𝝍​p​(𝐲|𝐱,zl=1,ϕ,𝝍)⏟Leaf-specific prediction. ​plϕ,𝝍𝑝conditional𝐲  𝐱Θsuperscriptsubscript𝑙1𝐿subscript⏟𝑝subscript𝑧𝑙conditional1  𝐱𝜽𝝍Leaf-assignment prob. superscriptsubscript𝜋𝑙  𝜽𝝍subscript⏟𝑝conditional𝐲  𝐱subscript𝑧𝑙  1bold-italic-ϕ𝝍Leaf-specific prediction. superscriptsubscript𝑝𝑙  bold-italic-ϕ𝝍p(\mathbf{y}|\mathbf{x},\Theta)=\sum\_{l=1}^{L}\underbrace{p(z\_{l}=1|\mathbf{x},\boldsymbol{\theta},\boldsymbol{\psi})}\_{\text{Leaf-assignment prob. }\pi\_{l}^{\boldsymbol{\theta},\boldsymbol{\psi}}}\underbrace{p(\mathbf{y}|\mathbf{x},z\_{l}=1,\boldsymbol{\phi},\boldsymbol{\psi})}\_{\text{Leaf-specific prediction. }p\_{l}^{\boldsymbol{\phi},\boldsymbol{\psi}}}\vspace{-2.5mm} |  | (1) |

where 𝐳∈{0,1}L𝐳superscript01𝐿\mathbf{z}\in\{0,1\}^{L} is an L𝐿L-dimensional binary latent variable such that ∑l=1Lzl=1superscriptsubscript𝑙1𝐿subscript𝑧𝑙1\sum\_{l=1}^{L}z\_{l}=1, which describes the choice of leaf node (e.g. zl=1subscript𝑧𝑙1z\_{l}=1 means that leaf l𝑙l is used). Here 𝜽,𝝍,ϕ

𝜽𝝍bold-italic-ϕ\boldsymbol{\theta},\boldsymbol{\psi},\boldsymbol{\phi} summarise the parameters of router, transformer and solver modules in the tree. The mixing coefficient πl𝜽,𝝍​(𝐱):=p​(zl=1|𝐱,𝝍,𝜽)assignsuperscriptsubscript𝜋𝑙

𝜽𝝍𝐱𝑝subscript𝑧𝑙conditional1

𝐱𝝍𝜽\pi\_{l}^{\boldsymbol{\theta},\boldsymbol{\psi}}(\mathbf{x}):=p(z\_{l}=1|\mathbf{x},\boldsymbol{\psi},\boldsymbol{\theta}) quantifies the probability that 𝐱𝐱\mathbf{x} is assigned to leaf l𝑙l and is given by a product of decision probabilities over all router modules on the unique path 𝒫lsubscript𝒫𝑙\mathcal{P}\_{l} from the root to leaf node l𝑙l:

|  |  |  |
| --- | --- | --- |
|  | πl𝝍,𝜽​(𝐱)=∏rj𝜽∈𝒫lrj𝜽​(𝐱j𝝍) 1​[l↙j]⋅(1−rj𝜽​(𝐱j𝝍)) 1−𝟙​[l↙j]superscriptsubscript𝜋𝑙  𝝍𝜽𝐱subscriptproductsuperscriptsubscript𝑟𝑗𝜽subscript𝒫𝑙⋅superscriptsubscript𝑟𝑗𝜽superscriptsubscriptsuperscript𝐱𝝍𝑗1delimited-[]↙𝑙𝑗superscript1superscriptsubscript𝑟𝑗𝜽subscriptsuperscript𝐱𝝍𝑗11delimited-[]↙𝑙𝑗\pi\_{l}^{\boldsymbol{\psi},\boldsymbol{\theta}}(\mathbf{x})=\prod\_{r\_{j}^{\boldsymbol{\theta}}\in\mathcal{P}\_{l}}r\_{j}^{\boldsymbol{\theta}}(\mathbf{x}^{\boldsymbol{\psi}}\_{j})^{\,\mathds{1}[l\swarrow j]}\cdot\big{(}1-r\_{j}^{\boldsymbol{\theta}}(\mathbf{x}^{\boldsymbol{\psi}}\_{j})\big{)}^{\,1-\mathds{1}[{l\swarrow j}]} |  |

where l↙j↙𝑙𝑗l\swarrow j is a binary relation and is only true if leaf l𝑙l is in the left subtree of internal node j𝑗j, and 𝐱j𝝍subscriptsuperscript𝐱𝝍𝑗\mathbf{x}^{\boldsymbol{\psi}}\_{j} is the feature representation of 𝐱𝐱\mathbf{x} at node j𝑗j. Let 𝒯j={te1𝝍,…,ten𝝍}subscript𝒯𝑗superscriptsubscript𝑡subscript𝑒1𝝍…superscriptsubscript𝑡subscript𝑒𝑛𝝍\mathcal{T}\_{j}=\{t\_{e\_{1}}^{\boldsymbol{\psi}},...,t\_{e\_{n}}^{\boldsymbol{\psi}}\} denote the ordered set of the n𝑛n transformer modules on the path from the root to node j𝑗j, the feature vector 𝐱j𝝍subscriptsuperscript𝐱𝝍𝑗\mathbf{x}^{\boldsymbol{\psi}}\_{j} is given by

|  |  |  |
| --- | --- | --- |
|  | 𝐱j𝝍:=(ten𝝍∘…∘te2𝝍∘te1𝝍)​(𝐱).assignsubscriptsuperscript𝐱𝝍𝑗superscriptsubscript𝑡subscript𝑒𝑛𝝍…superscriptsubscript𝑡subscript𝑒2𝝍superscriptsubscript𝑡subscript𝑒1𝝍𝐱\mathbf{x}^{\boldsymbol{\psi}}\_{j}:=\Big{(}t\_{e\_{n}}^{\boldsymbol{\psi}}\circ...\circ t\_{e\_{2}}^{\boldsymbol{\psi}}\circ t\_{e\_{1}}^{\boldsymbol{\psi}}\Big{)}(\mathbf{x}).\vspace{-2mm} |  |

On the other hand, the leaf-specific conditional distribution plϕ,𝝍​(𝐲):=p​(𝐲|𝐱,zl=1,ϕ,𝝍)assignsuperscriptsubscript𝑝𝑙

bold-italic-ϕ𝝍𝐲𝑝conditional𝐲

𝐱subscript𝑧𝑙

1bold-italic-ϕ𝝍p\_{l}^{\boldsymbol{\phi},\boldsymbol{\psi}}(\mathbf{y}):=p(\mathbf{y}|\mathbf{x},z\_{l}=1,\boldsymbol{\phi},\boldsymbol{\psi}) in ([1](#S3.E1 "In 3.2 Probabilistic model and inference ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")) yields an estimate for the distribution over target 𝐲𝐲\mathbf{y} for leaf node l𝑙l and is given by its solver’s output
slϕ​(𝐱parent​(l)𝝍)superscriptsubscript𝑠𝑙bold-italic-ϕsubscriptsuperscript𝐱𝝍parent𝑙s\_{l}^{\boldsymbol{\phi}}(\mathbf{x}^{\boldsymbol{\psi}}\_{\mathrm{parent}(l)}).

We consider two inference schemes based on a trade-off between accuracy and computation, which we refer to as multi-path and single-path inference. The multi-path inference uses the full predictive distribution given in eq. ([1](#S3.E1 "In 3.2 Probabilistic model and inference ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")). However, computing this quantity requires averaging the distributions over all the leaves involving computing all operations at all nodes and edges of the tree, which is expensive for a large ANT. On the other hand, the single-path inference scheme only uses the predictive distribution at the leaf node chosen by greedily traversing the tree in the directions of highest confidence of the routers. This approximation constrains computations to a single path, allowing for more memory- and time-efficient inference.

## 4 Optimisation

Training of an ANT proceeds in two stages: 1) growth phase during which the model architecture is learned based on local optimisation, and 2) refinement phase which further tunes the parameters of the model discovered in the first phase based on global optimisation. We include a pseudocode of the training algorithm in Supp. Sec. A.

### 4.1 Loss function: optimising parameters of 𝕆𝕆\mathbb{O}

For both phases, we use the negative log-likelihood (NLL) as the common objective function to minimise:

|  |  |  |
| --- | --- | --- |
|  | −log ​p​(𝐘|𝐗,Θ)=−∑n=1Nlog ​(∑l=1Lπl𝜽,𝝍​(𝐱(n))​plϕ,𝝍​(𝐲(n)))log 𝑝conditional𝐘  𝐗Θsuperscriptsubscript𝑛1𝑁log superscriptsubscript𝑙1𝐿superscriptsubscript𝜋𝑙  𝜽𝝍superscript𝐱𝑛superscriptsubscript𝑝𝑙  bold-italic-ϕ𝝍superscript𝐲𝑛-\text{log }p(\mathbf{Y}|\mathbf{X},\Theta)=-\sum\_{n=1}^{N}\text{log }(\sum\_{l=1}^{L}\pi\_{l}^{\boldsymbol{\theta},\boldsymbol{\psi}}(\mathbf{x}^{(n)})\,p\_{l}^{\boldsymbol{\phi},\boldsymbol{\psi}}(\mathbf{y}^{(n)})) |  |

where 𝐗={𝐱(1),…,𝐱(N)}𝐗superscript𝐱1…superscript𝐱𝑁\mathbf{X}=\{\mathbf{x}^{(1)},...,\mathbf{x}^{(N)}\}, 𝐘={𝐲(1),…,𝐲(N)}𝐘superscript𝐲1…superscript𝐲𝑁\mathbf{Y}=\{\mathbf{y}^{(1)},...,\mathbf{y}^{(N)}\} denote the training inputs and targets. As all component modules (routers, transformers and solvers) are differentiable with respect to their parameters Θ=(𝜽,𝝍,ϕ)Θ𝜽𝝍bold-italic-ϕ\Theta=(\boldsymbol{\theta},\boldsymbol{\psi},\boldsymbol{\phi}), we can use gradient-based optimisation. Given an ANT with fixed topology 𝕋𝕋\mathbb{T}, we use backpropagation (Rumelhart et al., [1986](#bib.bib53)) for gradient computation and use gradient descent to minimise the NLL for learning the parameters.

### 4.2 Growth phase: learning architecture 𝕋𝕋\mathbb{T}

We next describe our proposed method for growing the tree 𝕋𝕋\mathbb{T} to an architecture of adequate complexity for the given training data. Starting from the root, we choose one of the leaf nodes in breadth-first order and incrementally modify the architecture by adding computational modules to it. In particular, we evaluate 333 choices (Fig. [1](#S3.F1 "Figure 1 ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees") (Right)) at each leaf node; (1)“split data” extends the current model by splitting the node with an addition of a new router; (2) “deepen transform” increases the depth of the incoming edge by adding a new transformer; (3) “keep” retains the current model. We then locally optimise the parameters of the newly added modules in the architectures of (1) and (2) by minimising NLL via gradient descent, while fixing the parameters of the previous part of the computational graph. Lastly, we select the model with the lowest validation NLL if it improves on the previously observed lowest NLL, otherwise we execute (3). This process is repeated to all new nodes level-by-level until no more “split data” or “deepen transform” operations pass the validation test.

The rationale for evaluating the two choices is to give the model a freedom to choose the most effective option between “going deeper” or splitting the data space. Splitting a node is equivalent to a soft partitioning of the feature space of incoming data, and gives birth to two new leaf nodes (left and right children solvers). In this case, the added transformer modules on the two branches are identity functions. Deepening an edge on the other hand seeks to learn richer representation via an extra nonlinear transformation, and replaces the old solver with a new one. Local optimisation is efficient in time and space; gradients only need to be computed for the parameters of the new parts of the architecture, reducing computation, while forward activations prior to the new parts do not need to be stored in memory, saving space.

### 4.3 Refinement phase: global tuning of 𝕆𝕆\mathbb{O}

Once the model topology is determined in the growth phase, we finish by performing global optimisation to refine the parameters of the model, now with a fixed architecture. This time, we perform gradient descent on the NLL with respect to the parameters of all modules in the graph, jointly optimising the hierarchical grouping of data to paths on the tree and the associated expert NNs. The refinement phase can correct suboptimal decisions made during the local optimisation of the growth phase, and empirically improves the generalisation error (see Sec. [5.3](#S5.SS3 "5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees")).

## 5 Experiments

Table 3: Comparison of performance of different models on SARCOS, MNIST and CIFAR-10. The columns “Error (multi-path)” and “Error (single-path)” indicate the classification (%percent\%) or regression (MSE) errors of predictions based on the multi-path and the single-path inference. The columns “Params. (multi-path)” and “Params. (single-path)” respectively show the total number of parameters in the model and the average number of parameters used during single-path inference. “Ensemble Size” indicates the size of ensemble used. An entry of “–” indicates that no value was reported. Methods marked with † are from our implementations trained in the same experimental setup. \* indicates that the parameters are initialised with a pre-trained CNN.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Method | Error  (multi-path) | Error  (single-path) | Params.  (multi-path) | Params.  (single-path) | Ensemble  Size |
| SARCOS | Linear regression | 10.693 | N/A | 154 | N/A | 1 |
| MLP with 2 hidden layers (Zhao et al., [2017](#bib.bib73)) | 5.111 | N/A | 31,804 | N/A | 1 |
| Decision tree | 3.708 | 3.708 | 319,591 | 25 | 1 |
| MLP with 1 hidden layer | 2.835 | N/A | 7,431 | N/A | 1 |
| Gradient boosted trees | 2.661 | 2.661 | 391,324 | 2,083 | 7 ×\times 30 |
| MLP with 5 hidden layers | 2.657 | N/A | 270,599 | N/A | 1 |
| Random forest | 2.426 | 2.426 | 40,436,840 | 4,791 | 200 |
| Random forest | 2.394 | 2.394 | 141,540,436 | 16,771 | 700 |
| MLP with 3 hidden layers | 2.129 | N/A | 139,015 | N/A | 1 |
| SDT (with MLP routers) | 2.118 | 2.246 | 28,045 | 10,167 | 1 |
| Gradient boosted trees | 1.444 | 1.444 | 988,256 | 6,808 | 7 ×\times 100 |
| ANT-SARCOS | 1.384 | 1.542 | 103,823 | 61,640 | 1 |
|  | ANT-SARCOS (ensemble) | 1.226 | 1.372 | 598,280 | 360,766 | 8 |
| MNIST | Linear classifier | 7.91 | N/A | 7,840 | N/A | 1 |
| RDT (Léon & Denoyer, [2015](#bib.bib42)) | 5.41 | – | – | – | 1 |
| Random Forests (Breiman, [2001](#bib.bib5)) | 3.21 | 3.21 | – | – | 200 |
| Compact Multi-Class Boosted Trees (Ponomareva et al., [2017](#bib.bib49)) | 2.88 | – | – | – | 100 |
| Alternating Decision Forest (Schulter et al., [2013](#bib.bib56)) | 2.71 | 2.71 | – | – | 20 |
| Neural Decision Tree (Xiao, [2017](#bib.bib70)) | 2.10 | – | 1,773,130 | 502,170 | 1 |
| ANT-MNIST-C | 1.62 | 1.68 | 39,670 | 7,956 | 1 |
| MLP with 2 hidden layers (Simard et al., [2003](#bib.bib60)) | 1.40 | N/A | 1,275,200 | N/A | 1 |
| LeNet-5† (LeCun et al., [1998](#bib.bib38)) | 0.82 | N/A | 431,000 | N/A | 1 |
| gcForest (Zhou & Feng, [2017](#bib.bib74)) | 0.74 | 0.74 | – | – | 500 |
| ANT-MNIST-B | 0.72 | 0.73 | 76,703 | 50,653 | 1 |
| Neural Decision Forest (Kontschieder et al., [2015](#bib.bib34)) | 0.70 | – | 544,600 | 463,180 | 10 |
| ANT-MNIST-A | 0.64 | 0.69 | 100,596 | 84,935 | 1 |
| ANT-MNIST-A (ensemble) | 0.29 | 0.30 | 850,775 | 655,449 | 8 |
|  | CapsNet (Sabour et al., [2017](#bib.bib54)) | 0.25 | – | 8.2M | N/A | 1 |
| CIFAR-10 | Compact Multi-Class Boosted Trees (Ponomareva et al., [2017](#bib.bib49)) | 52.31 | – | – | – | 100 |
| Random Forests (Breiman, [2001](#bib.bib5)) | 50.17 | 50.17 | – | – | 2000 |
| gcForest (Zhou & Feng, [2017](#bib.bib74)) | 38.22 | 38.22 | – | – | 500 |
| MaxOut (Goodfellow et al., [2013](#bib.bib18)) | 9.38 | N/A | 6M | N/A | 1 |
| ANT-CIFAR10-C | 9.31 | 9.34 | 0.7M | 0.5M | 1 |
| ANT-CIFAR10-B | 9.15 | 9.18 | 0.9M | 0.6M | 1 |
| Network in Network (Lin et al., [2014](#bib.bib43)) | 8.81 | N/A | 1M | N/A | 1 |
| All-CNN†(Springenberg et al., [2015](#bib.bib61)) | 8.71 | N/A | 1.4M | N/A | 1 |
| ANT-CIFAR10-A | 8.31 | 8.32 | 1.4M | 1.0M | 1 |
| ANT-CIFAR10-A (ensemble) | 7.71 | 7.79 | 8.7M | 7.4M | 8 |
| ANT-CIFAR10-A\* | 6.72 | 6.74 | 1.3M | 0.8M | 1 |
| ResNet-110 (He et al., [2016](#bib.bib22)) | 6.43 | N/A | 1.7M | N/A | 1 |
| DenseNet-BC (k=24) (Huang et al., [2017](#bib.bib24)) | 3.74 | N/A | 27.2M | N/A | 1 |

We evaluate ANTs using the SARCOS multivariate regression dataset (Vijayakumar & Schaal, [2000](#bib.bib68)), and the MNIST (LeCun et al., [1998](#bib.bib38)) and CIFAR-10 (Krizhevsky & Hinton, [2009](#bib.bib35)) classification datasets. We run ablation studies to show that our different components are vital for the best performance. We then assess the ability of ANTs to automatically learn meaningful hierarchical structures in data. Next, we examine the effects of refinement phase on ANTs, and show that it can automatically prune the tree. Finally, we demonstrate that our proposed training procedure adapts the model size appropriately under varying amounts of labelled data. All of our models are implemented in PyTorch (Paszke et al., [2017](#bib.bib47))111Codes: <https://github.com/rtanno21609/AdaptiveNeuralTrees>. Full training details, including training times on a single GPU, are provided in Supp. Sec. C and D.

### 5.1 Model performance

We compare the performance of ANTs (Tab. [2](#S3.T2 "Table 2 ‣ 3.1 Model Topology and Operations ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")) against a range of DT and NN models (Tab. [3](#S5.T3 "Table 3 ‣ 5 Experiments ‣ Adaptive Neural Trees")), where notably the relative performance of these two classes of models differs between datasets. ANTs inherit from both and achieve the lowest error on SARCOS, and perform favourably on MNIST and CIFAR-10. In general, DT methods without feature learning, such as RFs (Breiman, [2001](#bib.bib5); Zhou & Feng, [2017](#bib.bib74)) and GBTs (Ponomareva et al., [2017](#bib.bib49)), perform poorly on image classification tasks (Krizhevsky & Hinton, [2009](#bib.bib35)). In comparison with CNNs without shortcut connections (LeCun et al., [1998](#bib.bib38); Goodfellow et al., [2013](#bib.bib18); Lin et al., [2014](#bib.bib43); Springenberg et al., [2015](#bib.bib61)), different ANTs balance between stronger performance with comparable numbers of trainable parameters, and comparable performance with smaller amount of parameters. At the other end of the spectrum, state-of-the-art NNs (Sabour et al., [2017](#bib.bib54); Huang et al., [2017](#bib.bib24)) contain significantly more parameters.

Conditional computation: Tab.[3](#S5.T3 "Table 3 ‣ 5 Experiments ‣ Adaptive Neural Trees") compares the errors and number of parameters of different ANTs for both multi-path and single-path inference schemes. While reducing the number of parameters (from Params (multi-path) to Params (single-path)) across all ANT models, we observe only a small difference in error (between Error (multi-path) and Error (single-path)), with the largest deviations being 0.06%percent0.060.06\% for classification and 0.1580.1580.158 for regression. In addition, Supp. Sec. H shows that the single-path inference reduces FLOPS. This means that single-path inference gives an accurate approximation of the multi-path inference, while being more efficient to compute. This close approximation comes from the confident splitting probabilities of routers, being close to 0 or 1 (see blue histograms in Fig. [2](#S5.F2 "Figure 2 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees")(b)).

Ablation study: we compare the predictive errors of different variants of ANTs in cases where the options for adding transformer or router modules are disabled (see Tab. [4](#S5.T4 "Table 4 ‣ 5.1 Model performance ‣ 5 Experiments ‣ Adaptive Neural Trees")). In the first case, the resulting models are equivalent to SDTs (Suárez & Lutsko, [1999](#bib.bib64)) or HMEs (Jordan & Jacobs, [1994](#bib.bib30)) with locally grown architectures, while the second case is equivalent to standard CNNs, grown adaptively layer by layer. We observe that either ablation consistently leads to higher errors across different module configurations on all three datasets, justifying the combination of feature learning and hierarchical partitioning in ANTs.

SARCOS multivariate regression: Tab. [3](#S5.T3 "Table 3 ‣ 5 Experiments ‣ Adaptive Neural Trees") shows that ANT-SARCOS outperforms all other methods in mean squared error (MSE) with the full set of parameters. With the single-path inference, GBTs performs slightly better than a single ANT while requiring fewer parameters. We note that the top 3 methods are all tree-based, with the third best method being an SDT (with MLP routers). On the other hand, ANT and GBTs outperform the best standard NN model with less than a half of the parameter count. This highlights the value of hierarchical clustering for predictive performance and inference speed. Meanwhile, we still reap the benefits of representation learning, as shown by both ANT-SARCOS and the SDT (which is a specific form of ANT with identity transformers) requiring fewer parameters than the best-performing GBT configuration. Finally, we note that deeper NNs (5 vs. 3 hidden layers) can overfit on this small dataset, which makes the adaptive growth procedure of tree-based methods ideal for finding a model that exhibits good generalisation.

MNIST digit classification: we observe that ANT-MNIST-A outperforms state-of-the-art GBT (Ponomareva et al., [2017](#bib.bib49)) and RF (Zhou & Feng, [2017](#bib.bib74)) methods in accuracy. This performance is attained despite the use of a single tree, while RF methods operate with ensembles of classifiers (the size shown in Tab. [2](#S3.T2 "Table 2 ‣ 3.1 Model Topology and Operations ‣ 3 Adaptive Neural Trees ‣ Adaptive Neural Trees")). In particular, the NDF (Kontschieder et al., [2015](#bib.bib34)) has a pre-specified architecture where LeNet-5 (LeCun et al., [1998](#bib.bib38)) is used as the root transformer module, and 101010 trees of fixed depth 555 are built on this base features. On the other hand, ANT-MNIST-A is constructed in a data-driven manner from primitive modules, and displays an improvement over the NDF both in terms of accuracy and number of parameters. In addition, reducing the size of convolution kernels (ANT-MNIST-B) reduces the total number of parameters by 25%percent2525\% and the path-wise average by almost 40%percent4040\% while only increasing the error by <0.1%absentpercent0.1<0.1\%.

We also compare against the LeNet-5 CNN (LeCun et al., [1998](#bib.bib38)), comprised of the same types of operations used in our primitive modules (i.e. convolutional, max-pooling and FC layers). For a fair comparison, the network is trained with the same protocol as that of the ANT refinement phase, achieving an error rate of 0.82%percent0.820.82\%. Both ANT-MNIST-A and ANT-MNIST-B attain better accuracy with a smaller number of parameters than LeNet-5. The current state-of-the-art, capsule networks (CapsNets) (Sabour et al., [2017](#bib.bib54)), have more parameters than ANT-MNIST-A by almost two orders of magnitude.222Notably, CapsNets also feature a routing mechanism, but with a significantly different mechanism and motivation. By ensembling ANTs, we can reach similar performance (0.29%percent0.290.29\% versus 0.25%percent0.250.25\%) with an order of magnitude less parameters (see Supp. Sec. I).

Lastly, we highlight the observation that ANT-MNIST-C, with the simplest primitive modules, achieves an error rate of 1.68%percent1.681.68\% with single-path inference, which is significantly better than that of the linear classifier (7.91%percent7.917.91\%), while engaging almost the same number of parameters (7,956

79567,956 vs. 7,840

78407,840) on average. To isolate the benefit of convolutions, we took one of the root-to-path CNNs on ANT-MNIST-C and increased the number of kernels to adjust the number of parameters to the same value. We observe a higher error rate of 3.55%percent3.553.55\%, which indicates that while convolutions are beneficial, data partitioning has additional benefits in improving accuracy. This result demonstrates the potential of ANT growth protocol for constructing performant models with lightweight inference. See Sec. G in the supplementary materials for the architecture of ANT-MNIST-C.

Table 4: Ablation study on regression (MSE) and classification (%percent\%) errors. “CNN” refers to the case where the ANT is grown without routers while “SDT/HME” refers to the case where transformer modules on the edges are disabled.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Error (multi-path) | | | Error (single-path) | | |
|  | ANT | CNN | HME | ANT | CNN | HME |
|  | (default) | (no ℛℛ\mathcal{R}) | (no 𝒯𝒯\mathcal{T}) | (default) | (no ℛℛ\mathcal{R}) | (no 𝒯𝒯\mathcal{T}) |
| SARCOS | 1.38 | 2.51 | 2.12 | 1.54 | 2.51 | 2.25 |
| MNIST-A | 0.64 | 0.74 | 3.18 | 0.69 | 0.74 | 4.19 |
| MNIST-B | 0.72 | 0.80 | 4.63 | 0.73 | 0.80 | 3.62 |
| MNIST-C | 1.62 | 3.71 | 5.70 | 1.68 | 3.71 | 6.96 |
| CIFAR10-A | 8.31 | 9.29 | 39.29 | 8.32 | 9.29 | 40.33 |
| CIFAR10-B | 9.15 | 11.08 | 43.09 | 9.18 | 11.08 | 44.25 |
| CIFAR10-C | 9.31 | 11.61 | 48.59 | 9.34 | 11.61 | 50.02 |

CIFAR-10 object recognition: we see that ANTs largely outperform the state-of-the-art DT method, gcForest (Zhou & Feng, [2017](#bib.bib74)), achieving over 90% accuracy, demonstrating the benefit of representation learning in tree-structured models. Secondly, with fewer number of parameters in single-path inference, ANT-CIFAR-A achieves higher accuracy than CNN models without shortcut connections (Goodfellow et al., [2013](#bib.bib18); Lin et al., [2014](#bib.bib43); Springenberg et al., [2015](#bib.bib61)) that held the state-of-the-art performance in respective years. With simpler primitive modules we learn more compact models (ANT-MNIST-B and -C) with a marginal compromise in accuracy. In addition, initialising the parameters of transformers and routers from a pre-trained single-path CNN further reduced the error rate of ANT-MNIST-A by 20% (see ANT-MNIST-A\* in Tab. [3](#S5.T3 "Table 3 ‣ 5 Experiments ‣ Adaptive Neural Trees")), indicating room for improvement in our proposed optimisation method.

Shortcut connections (Fahlman & Lebiere, [1990](#bib.bib13)) have recently lead to leaps in performance in deep CNNs (He et al., [2016](#bib.bib22); Huang et al., [2017](#bib.bib24)). We observe that our best network, ANT-MNIST-A\*, has a comparable error rate and half the parameter count (with single-path inference) to the best-performing residual network, ResNet-110 (He et al., [2016](#bib.bib22)). Densely connected networks have better accuracy, but with an order of magnitude more parameters (Huang et al., [2017](#bib.bib24)). We expect shortcut connections to improve ANT performance, and leave integrating them to future work.

### 5.2 Interpretability

The growth procedure of ANTs is capable of discovering hierarchical structures in the data that are useful to the end task. Without any regularization imposed on routers, the learned hierarchies often display strong specialisation of paths to certain classes or categories of data on both the MNIST and CIFAR-10 datasets. Fig. [2](#S5.F2 "Figure 2 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees") (a) displays an example with particularly “human-interpretable” partitions e.g. man-made versus natural objects, and road vehicles versus other types of vehicles. It should, however, be noted that human intuitions on relevant hierarchical structures do not necessarily equate to optimal representations, particularly as datasets may not necessarily have an underlying hierarchical structure, e.g., MNIST. Rather, what needs to be highlighted is the ability of ANTs to learn when to share or separate the representation of data to optimise end-task performance, which gives rise to automatically discovering such hierarchies. To further attest that the model learns a meaningful routing strategy, we also present the test accuracy of the predictions from the leaf node with the smallest reaching probability in Supp. Sec. F. We observe that using the least likely “expert” leads to a substantial drop in classification accuracy. In addition, most learned trees are unbalanced. This property of adaptive computation is plausible since certain types of images may be easier to classify than others, as seen in prior work (Figurnov et al., [2017](#bib.bib15)).

### 5.3 Effect of refinement phase

![Refer to caption](/html/1807.06699/assets/figures/fig_7_9.png)


(a) Before refinement

![Refer to caption](/html/1807.06699/assets/figures/fig_7_10.png)


(b) After refinement

Figure 2: Visualisation of class distributions (red) and path probabilities (blue) computed over the whole test set at respective nodes of an example ANT (a) before and (b) after the refinement phase. (a) shows that the model captures an interpretable hierarchy, grouping semantically similar images on the same branches. (b) shows that the refinement phase polarises path probabilities, pruning a branch.



![Refer to caption](/html/1807.06699/assets/x1.png)

![Refer to caption](/html/1807.06699/assets/x2.png)

![Refer to caption](/html/1807.06699/assets/x3.png)

Figure 3: (Left). Test accuracy on CIFAR-10 of ANTs for varying amounts of training data. (Middle) The complexity of the grown ANTs increases with dataset size. (Right) Refinement improves generalisation; the dotted lines show where the refinement phase starts.

We observe that global refinement phase improves the generalisation error. Fig. [3](#S5.F3 "Figure 3 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees") (Right) shows the generalisation error of various ANT models on CIFAR-10, with vertical dotted lines indicating the epoch when the models enter the refinement phase. As we switch from optimising parts of the ANT in isolation to optimising all parameters, we shift the optimisation landscape, resulting in an initial drop in performance. However, they all consistently converge to higher test accuracy than the best value attained during the growth phase. This provides evidence that refinement phase remedies suboptimal decisions made during the locally-optimised growth phase. In many cases, we observed that global optimisation polarises the decision probability of routers, which occasionally leads to the effective “pruning” of some branches. For example, in the case of the tree shown in Fig. [2](#S5.F2 "Figure 2 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees")(b), we observe that the decision probability of routers are more concentrated near 0 or 1 after global refinement, and as a result, the empirical probability of visiting one of the leaf nodes, calculated over the validation set, reduces to 0.09%—meaning that the corresponding branch could be pruned without a negligible change in the network’s accuracy. The resultant model attains lower generalisation error, showing the pruning has resolved a suboptimal partioning of data.

### 5.4 Adaptive model complexity

Overparametrised models, trained without regularization, are vulnerable to overfitting on small datasets. Here we assess the ability of our proposed ANT training method to adapt the model complexity to varying amounts of labelled data. We run classfication experiments on CIFAR-10 and train three variants of ANTs, All-CNN (Springenberg et al., [2015](#bib.bib61)) and linear classifier on subsets of the dataset of sizes 50, 250, 500, 2.5k, 5k, 25k and 45k (the full training set). Here we choose All-CNN as the baseline as it has similar number of parameters when trained on the full dataset and is the closest in terms of constituent operations (convolutional, GAP and FC layers).
Fig.[3](#S5.F3 "Figure 3 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees") (Left) shows the corresponding test performances. The best model is picked based on the performance on the same validation set of 5k examples as before. As the dataset gets smaller, the margin between the test accuracy of the ANT models and All-CNN/linear classifier increases (up to 13%percent1313\%). Fig. [3](#S5.F3 "Figure 3 ‣ 5.3 Effect of refinement phase ‣ 5 Experiments ‣ Adaptive Neural Trees") (Middle) shows the model size of discovered ANTs as the dataset size varies. For different settings of primitive modules, the number of parameters generally increases as a function of the dataset size. All-CNN has a fixed number of parameters, consistently larger than the discovered ANTs, and suffers from overfitting, particularly on small datasets. The linear classifier, on the other hand, underfits to the data. Our method constructs models of adequate complexity, leading to better generalisation. This shows the value of our tree-building algorithm over using models of fixed-size structures.

## 6 Conclusion

We introduced Adaptive Neural Trees (ANTs), a holistic way to marry the architecture learning, conditional computation and hierarchical clustering of decision trees (DTs) with the hierarchical representation learning and gradient descent optimization of deep neural networks (DNNs). Our proposed training algorithm optimises both the parameters and architectures of ANTs through progressive growth, tuning them to the size and complexity of the training dataset. Together, these properties make ANTs a generalisation of previous work attempting to unite NNs and DTs. Finally, we validated the claimed benefits of ANTs for regression (SARCOS dataset) and classification (MNIST & CIFAR10 datasets), whilst still achieving high performance.

## Acknowledgements

We would like to thank Konstantinos Kamnitsas, Sebastian Tschiatschek, Jan Stühmer, Katja Hofmann and Danielle Belgrave for their insightful discussions. Daniel C. Alexander is supported by the NIHR UCLH Biomedical Research Centre and EPSRC grants M020533, R006032, and R014019. Kai Arulkumaran is supported by a PhD scholarship from the Department of Bioengineering, Imperial College London. Ryutaro Tanno is supported by a Microsoft scholarship. Lastly, we would like to thank the anonymous reviewers for their valuable suggestions.

## References

* Almahairi et al. (2016)

  Almahairi, A., Ballas, N., Cooijmans, T., Zheng, Y., Larochelle, H., and
  Courville, A.
  Dynamic capacity networks.
  In *ICML*, 2016.
* Bengio et al. (2015)

  Bengio, E., Bacon, P.-L., Pineau, J., and Precup, D.
  Conditional computation in neural networks for faster models.
  *CoRR*, 2015.
* Bengio (2013)

  Bengio, Y.
  Deep learning of representations: Looking forward.
  In *International Conference on Statistical Language and Speech
  Processing*, pp.  1–37. Springer, 2013.
* Bengio et al. (2013)

  Bengio, Y., Léonard, N., and Courville, A.
  Estimating or propagating gradients through stochastic neurons for
  conditional computation.
  *CoRR*, 2013.
* Breiman (2001)

  Breiman, L.
  Random forests.
  *Machine learning*, 45(1):5–32, 2001.
* Brock et al. (2017)

  Brock, A., Lim, T., Ritchie, J. M., and Weston, N.
  Smash: one-shot model architecture search through hypernetworks.
  *CoRR*, 2017.
* Cai et al. (2018)

  Cai, H., Chen, T., Zhang, W., Yu, Y., and Wang, J.
  Efficient architecture search by network transformation.
  In *AAAI*, 2018.
* Chen et al. (2016)

  Chen, T., Goodfellow, I., and Shlens, J.
  Net2net: Accelerating learning via knowledge transfer.
  In *ICLR*, 2016.
* Cortes et al. (2017)

  Cortes, C., Gonzalvo, X., Kuznetsov, V., Mohri, M., and Yang, S.
  Adanet: Adaptive structural learning of artificial neural networks.
  In *ICML*, pp.  874–883, 2017.
* Criminisi & Shotton (2013)

  Criminisi, A. and Shotton, J.
  *Decision forests for computer vision and medical image
  analysis*.
  Springer, 2013.
* Davis & Arel (2013)

  Davis, A. and Arel, I.
  Low-rank approximations for conditional feedforward computation in
  deep neural networks.
  *CoRR*, 2013.
* Deng et al. (2009)

  Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L.
  Imagenet: A large-scale hierarchical image database.
  In *CVPR*, 2009.
* Fahlman & Lebiere (1990)

  Fahlman, S. E. and Lebiere, C.
  The cascade-correlation learning architecture.
  In *Advances in neural information processing systems*, pp. 524–532, 1990.
* Feng et al. (2018)

  Feng, J., Yu, Y., and Zhou, Z.-H.
  Multi-layered gradient boosting decision trees.
  In *Advances in Neural Information Processing Systems*, pp. 3551–3561, 2018.
* Figurnov et al. (2017)

  Figurnov, M., Collins, M. D., Zhu, Y., Zhang, L., Huang, J., Vetrov, D. P., and
  Salakhutdinov, R.
  Spatially adaptive computation time for residual networks.
  *CVPR*, pp.  1790–1799, 2017.
* Friedman (2001)

  Friedman, J. H.
  Greedy function approximation: a gradient boosting machine.
  *Annals of statistics*, pp.  1189–1232, 2001.
* Frosst & Hinton (2017)

  Frosst, N. and Hinton, G. E.
  Distilling a neural network into a soft decision tree.
  *CoRR*, 2017.
* Goodfellow et al. (2013)

  Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y.
  Maxout networks.
  In *ICML*, 2013.
* Graves (2016)

  Graves, A.
  Adaptive computation time for recurrent neural networks.
  *CoRR*, abs/1603.08983, 2016.
* Guan et al. (2017)

  Guan, J., Liu, Y., Liu, Q., and Peng, J.
  Energy-efficient amortized inference with cascaded deep classifiers.
  *arXiv preprint arXiv:1710.03368*, 2017.
* Hansen & Salamon (1990)

  Hansen, L. K. and Salamon, P.
  Neural network ensembles.
  *IEEE TPAMI*, 12(10):993–1001, 1990.
* He et al. (2016)

  He, K., Zhang, X., Ren, S., and Sun, J.
  Deep residual learning for image recognition.
  In *Proceedings of the IEEE conference on computer vision and
  pattern recognition*, pp.  770–778, 2016.
* Hinton et al. (2006)

  Hinton, G. E., Osindero, S., and Teh, Y.-W.
  A fast learning algorithm for deep belief nets.
  *Neural computation*, 18(7):1527–1554,
  2006.
* Huang et al. (2017)

  Huang, G., Liu, Z., Weinberger, K. Q., and van der Maaten, L.
  Densely connected convolutional networks.
  In *CVPR*, 2017.
* Huang et al. (2018)

  Huang, G., Liu, S., Van der Maaten, L., and Weinberger, K. Q.
  Condensenet: An efficient densenet using learned group convolutions.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pp.  2752–2761, 2018.
* Ioannou et al. (2016)

  Ioannou, Y., Robertson, D., Zikic, D., Kontschieder, P., Shotton, J., Brown,
  M., and Criminisi, A.
  Decision forests, convolutional networks and the models in-between.
  *CoRR*, 2016.
* İrsoy & Alpaydın (2018)

  İrsoy, O. and Alpaydın, E.
  Continuously constructive deep neural networks.
  *arXiv preprint arXiv:1804.02491*, 2018.
* İrsoy et al. (2012)

  İrsoy, O., Yıldız, O. T., and Alpaydın, E.
  Soft decision trees.
  In *ICPR*, pp.  1819–1822. IEEE, 2012.
* İrsoy et al. (2014)

  İrsoy, O., Yildiz, O. T., and Alpaydın, E.
  Budding trees.
  In *ICPR*, pp.  3582–3587. IEEE, 2014.
* Jordan & Jacobs (1994)

  Jordan, M. I. and Jacobs, R. A.
  Hierarchical mixtures of experts and the em algorithm.
  *Neural computation*, 1994.
* Kaggle.com (2017)

  Kaggle.com.
  Two sigma financial modeling challenge, 2017.
  URL <https://www.kaggle.com/c/two-sigma-financial-modeling>.
* Kingma & Ba (2014)

  Kingma, D. P. and Ba, J.
  Adam: A method for stochastic optimization.
  *CoRR*, 2014.
* Kontschieder et al. (2013)

  Kontschieder, P., Kohli, P., Shotton, J., and Criminisi, A.
  Geof: Geodesic forests for learning coupled predictors.
  In *CVPR*, 2013.
* Kontschieder et al. (2015)

  Kontschieder, P., Fiterau, M., Criminisi, A., and Rota Bulo, S.
  Deep neural decision forests.
  In *ICCV*, pp.  1467–1475, 2015.
* Krizhevsky & Hinton (2009)

  Krizhevsky, A. and Hinton, G.
  Learning multiple layers of features from tiny images.
  Technical report, University of Toronto, 2009.
* Laptev & Buhmann (2014)

  Laptev, D. and Buhmann, J. M.
  Convolutional decision trees for feature learning and segmentation.
  In *German Conference on Pattern Recognition*, pp.  95–106.
  Springer, 2014.
* Le Folgoc et al. (2016)

  Le Folgoc, L., Nori, A. V., Ancha, S., and Criminisi, A.
  Lifted auto-context forests for brain tumour segmentation.
  In *International Workshop on Brainlesion: Glioma, Multiple
  Sclerosis, Stroke and Traumatic Brain Injuries*, pp.  171–183. Springer,
  2016.
* LeCun et al. (1998)

  LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
* LeCun et al. (2015)

  LeCun, Y., Bengio, Y., and Hinton, G.
  Deep learning.
  *Nature*, 521(7553):436, 2015.
* Lee et al. (2015)

  Lee, D.-H., Zhang, S., Fischer, A., and Bengio, Y.
  Difference target propagation.
  In *Joint european conference on machine learning and knowledge
  discovery in databases*, pp.  498–515. Springer, 2015.
* Lee et al. (2017)

  Lee, J., Yun, J., Hwang, S., and Yang, E.
  Lifelong learning with dynamically expandable networks.
  *CoRR*, 2017.
* Léon & Denoyer (2015)

  Léon, A. and Denoyer, L.
  Policy-gradient methods for decision trees.
  In *ESANN*, 2015.
* Lin et al. (2014)

  Lin, M., Chen, Q., and Yan, S.
  Network in network.
  In *ICLR*, 2014.
* Liu et al. (2017)

  Liu, C., Zoph, B., Shlens, J., Hua, W., Li, L.-J., Fei-Fei, L., Yuille, A.,
  Huang, J., and Murphy, K.
  Progressive neural architecture search.
  *CoRR*, 2017.
* Montillo et al. (2011)

  Montillo, A., Shotton, J., Winn, J., Iglesias, J. E., Metaxas, D., and
  Criminisi, A.
  Entangled decision forests and their application for semantic
  segmentation of ct images.
  In *Biennial International Conference on Information Processing
  in Medical Imaging*, pp.  184–196. Springer, 2011.
* Nair & Hinton (2010)

  Nair, V. and Hinton, G. E.
  Rectified linear units improve restricted boltzmann machines.
  In *ICML*, pp.  807–814, 2010.
* Paszke et al. (2017)

  Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z.,
  Desmaison, A., Antiga, L., and Lerer, A.
  Automatic differentiation in pytorch.
  In *NIPS Autodiff Workshop*, 2017.
* Pedregosa et al. (2011)

  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
  O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al.
  Scikit-learn: Machine learning in python.
  *JMLR*, 12(Oct):2825–2830, 2011.
* Ponomareva et al. (2017)

  Ponomareva, N., Colthurst, T., Hendry, G., Haykal, S., and Radpour, S.
  Compact multi-class boosted trees.
  In *International Conference on Big Data*, pp.  47–56, 2017.
* Real et al. (2017)

  Real, E., Moore, S., Selle, A., Saxena, S., Suematsu, Y. L., Le, Q., and
  Kurakin, A.
  Large-scale evolution of image classifiers.
  *CoRR*, 2017.
* Richmond et al. (2015)

  Richmond, D. L., Kainmueller, D., Yang, M., Myers, E. W., and Rother, C.
  Mapping stacked decision forests to deep and sparse convolutional
  neural networks for semantic segmentation.
  *CoRR*, 2015.
* Rota Bulo & Kontschieder (2014)

  Rota Bulo, S. and Kontschieder, P.
  Neural decision forests for semantic image labelling.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pp.  81–88, 2014.
* Rumelhart et al. (1986)

  Rumelhart, D. E., Hinton, G. E., and Williams, R. J.
  Learning representations by back-propagating errors.
  *Nature*, 323(6088):533, 1986.
* Sabour et al. (2017)

  Sabour, S., Frosst, N., and Hinton, G. E.
  Dynamic routing between capsules.
  In *Advances in Neural Information Processing Systems*, pp. 3856–3866, 2017.
* Sandulescu & Chiru (2016)

  Sandulescu, V. and Chiru, M.
  Predicting the future relevance of research institutions-the winning
  solution of the kdd cup 2016.
  *CoRR*, 2016.
* Schulter et al. (2013)

  Schulter, S., Wohlhart, P., Leistner, C., Saffari, A., Roth, P. M., and
  Bischof, H.
  Alternating decision forests.
  In *CVPR, 2013*, 2013.
* Sethi (1990)

  Sethi, I. K.
  Entropy nets: From decision trees to neural networks.
  *Proceedings of the IEEE*, pp.  1605–1613, 1990.
* Shazeer et al. (2017)

  Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q. V., Hinton, G. E.,
  and Dean, J.
  Outrageously large neural networks: The sparsely-gated
  mixture-of-experts layer.
  *ICLR*, abs/1701.06538, 2017.
* Shotton et al. (2013)

  Shotton, J., Sharp, T., Kohli, P., Nowozin, S., Winn, J., and Criminisi, A.
  Decision jungles: Compact and rich models for classification.
  In *Advances in Neural Information Processing Systems*, pp. 234–242, 2013.
* Simard et al. (2003)

  Simard, P. Y., Steinkraus, D., Platt, J. C., et al.
  Best practices for convolutional neural networks applied to visual
  document analysis.
  In *ICDAR*, volume 3, pp.  958–962, 2003.
* Springenberg et al. (2015)

  Springenberg, J. T., Dosovitskiy, A., Brox, T., and Riedmiller, M.
  Striving for simplicity: The all convolutional net.
  *CoRR*, 2015.
* Srivastava et al. (2015)

  Srivastava, R. K., Greff, K., and Schmidhuber, J.
  Highway networks.
  *arXiv preprint arXiv:1505.00387*, 2015.
* Stanley & Miikkulainen (2002)

  Stanley, K. O. and Miikkulainen, R.
  Evolving neural networks through augmenting topologies.
  *Evolutionary computation*, 2002.
* Suárez & Lutsko (1999)

  Suárez, A. and Lutsko, J. F.
  Globally optimal fuzzy decision trees for classification and
  regression.
  *IEEE Transactions. PAMI*, 21(12):1297–1311, 1999.
* Szegedy et al. (2015)

  Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
  Vanhoucke, V., Rabinovich, A., et al.
  Going deeper with convolutions.
  In *CVPR*, 2015.
* Teerapittayanon et al. (2016)

  Teerapittayanon, S., McDanel, B., and Kung, H. T.
  Branchynet: Fast inference via early exiting from deep neural
  networks.
  In *ICPR*, 2016.
* Veit & Belongie (2017)

  Veit, A. and Belongie, S.
  Convolutional networks with adaptive computation graphs.
  *CoRR*, 2017.
* Vijayakumar & Schaal (2000)

  Vijayakumar, S. and Schaal, S.
  Locally weighted projection regression: An o(n) algorithm for
  incremental real time learning in high dimensional space.
  In *ICML*, volume 1, pp.  288–293, 2000.
* Volkovs et al. (2017)

  Volkovs, M., Yu, G. W., and Poutanen, T.
  Content-based neighbor models for cold start in recommender systems.
  In *Proceedings of the Recommender Systems Challenge 2017*, pp.  7. ACM, 2017.
* Xiao (2017)

  Xiao, H.
  Ndt: Neual decision tree towards fully functioned neural graph.
  *arXiv preprint arXiv:1712.05934*, 2017.
* Xiao et al. (2014)

  Xiao, T., Zhang, J., Yang, K., Peng, Y., and Zhang, Z.
  Error-driven incremental learning in deep convolutional neural
  network for large-scale image classification.
  In *ACM Multimedia*, 2014.
* Zeiler & Fergus (2014)

  Zeiler, M. D. and Fergus, R.
  Visualizing and understanding convolutional networks.
  In *ECCV*, pp.  818–833. Springer, 2014.
* Zhao et al. (2017)

  Zhao, H., Stretcu, O., Negrinho, R., Smola, A., and Gordon, G.
  Efficient multi-task feature and relationship learning.
  *arXiv preprint arXiv:1702.04423*, 2017.
* Zhou & Feng (2017)

  Zhou, Z.-H. and Feng, J.
  Deep forest: Towards an alternative to deep neural networks.
  In *IJCAI*, 2017.
* Zoph & Le (2017)

  Zoph, B. and Le, Q. V.
  Neural architecture search with reinforcement learning.
  *ICLR*, 2017.

## Appendix A Training algorithm

Algorithm 1  ANT Optimisation

Initialise topology 𝕋𝕋\mathbb{T}and parameters 𝕆𝕆\mathbb{O}𝕋𝕋\mathbb{T}is set to a root node with one solver and one transformer
Optimise parameters in 𝕆𝕆\mathbb{O}via gradient descent on NLL Learning root classifier
Set the root node “suboptimal”
true Growth of 𝕋𝕋\mathbb{T}begins
Freeze all parameters 𝕆𝕆\mathbb{O}Pick next “suboptimal” leaf node l∈𝒩l​e​a​f𝑙subscript𝒩𝑙𝑒𝑎𝑓l\in\mathcal{N}\_{leaf}in the breadth-first order
Add (1) router to l𝑙land train new parameters Split data
Add (2) transformer to l𝑙land train new parametersDeepen transform
Add (1) or (2) to 𝕋𝕋\mathbb{T}if validation error decreases, otherwise set l𝑙lto “optimal”
Add any new modules to 𝕆𝕆\mathbb{O}no “suboptimal” leaves remain
Break
Unfreeze and train all parameters in 𝕆𝕆\mathbb{O}Global refinement with fixed 𝕋𝕋\mathbb{T}

\State\Comment

\State\Comment

\State\While

\Comment\State

\State\State

\Comment\State

\Comment\State

\State\If

\State\EndIf

\EndWhile\State

\Comment

## Appendix B Additional related work

Here we provide an expanded review of related works, precluded from the main text due to space limit. The tree-structure of ANTs naturally performs conditional computation. We can also view the proposed tree-building algorithm as a form of neural architecture search. We provide surveys of these areas and their relations to ANTs.

Conditional computation: in NNs, computation of each sample engages every parameter of the model. In contrast, DTs route each sample to a single path, only activating a small fraction of the model. Bengio ([2013](#bib.bib3)) advocated for this notion of conditional computation to be integrated into NNs, and this has become a topic of growing interest. Rationales for using conditional computation ranges from attaining better capacity-to-computation ratio (Bengio et al., [2013](#bib.bib4); Davis & Arel, [2013](#bib.bib11); Bengio et al., [2015](#bib.bib2); Shazeer et al., [2017](#bib.bib58)) to adapting the required computation to the difficulty of the input and task (Bengio et al., [2015](#bib.bib2); Almahairi et al., [2016](#bib.bib1); Teerapittayanon et al., [2016](#bib.bib66); Graves, [2016](#bib.bib19); Figurnov et al., [2017](#bib.bib15); Veit & Belongie, [2017](#bib.bib67)). We view the growth procedure of ANTs as having a similar motivation with the latter—processing raw pixels is suboptimal for computer vision tasks, but we have no reason to believe that the hundreds of convolutional layers in current state-of-the-art architectures (He et al., [2016](#bib.bib22); Huang et al., [2017](#bib.bib24)) are necessary either. Growing ANTs adapts the architecture complexity to the dataset as a whole, with routers determining the computation needed on a per-sample basis.

Neural architecture search: the ANT growing procedure is related to the progressive growing of NNs (Fahlman & Lebiere, [1990](#bib.bib13); Hinton et al., [2006](#bib.bib23); Xiao et al., [2014](#bib.bib71); Chen et al., [2016](#bib.bib8); Srivastava et al., [2015](#bib.bib62); Lee et al., [2017](#bib.bib41); Cai et al., [2018](#bib.bib7); İrsoy & Alpaydın, [2018](#bib.bib27)), or more broadly, the field of neural architecture search (Zoph & Le, [2017](#bib.bib75); Brock et al., [2017](#bib.bib6); Cortes et al., [2017](#bib.bib9)). This approach, mainly via greedy layerwise training, has historically been one solution to optimising NNs (Fahlman & Lebiere, [1990](#bib.bib13); Hinton et al., [2006](#bib.bib23)). However, nowadays it is possible to train NNs in an end-to-end fashion. One area which still uses progressive growing is lifelong learning, in which a model needs to adapt to new tasks while retaining performance on previous ones (Xiao et al., [2014](#bib.bib71); Lee et al., [2017](#bib.bib41)). In particular, Xiao et al. ([2014](#bib.bib71)) introduced a method that grows a tree-shaped network to accommodate new classes. However, their method never transforms the data before passing it to the children classifiers, and hence never benefit from the parent’s representations.

Whilst we learn the architecture of an ANT in a greedy, layerwise fashion, several other methods search globally. Based on a variety of techniques, including evolutionary algorithms (Stanley & Miikkulainen, [2002](#bib.bib63); Real et al., [2017](#bib.bib50)), reinforcement learning (Zoph & Le, [2017](#bib.bib75)), sequential optimisation (Liu et al., [2017](#bib.bib44)) and boosting (Cortes et al., [2017](#bib.bib9)), these methods find extremely high-performance yet complex architectures. In our case, we constrain the search space to simple tree-structured NNs, retaining desirable properties of DTs such as data-dependent computation and interpretable structures, while keeping the space and time requirement of architecture search tractable thanks to the locality of our growth procedure.

Cascaded trees and forests: another noteworthy strand of work for feature learning with tree-structured models is cascaded forests—stacks of RFs where the outputs of intermediate models are fed into the subsequent ones (Montillo et al., [2011](#bib.bib45); Kontschieder et al., [2013](#bib.bib33); Zhou & Feng, [2017](#bib.bib74)). It has been shown how a cascade of DTs can be mapped to NNs with sparse connections (Sethi, [1990](#bib.bib57)), and more recently Richmond et al. ([2015](#bib.bib51)) extended this argument to RFs. However, the features obtained in this approach are the intermediate outputs of respective component models, which are not optimised for the target task, and cannot be learned end-to-end, thus limiting its representational quality. Recently, Feng et al. ([2018](#bib.bib14)) introduced a method to jointly train a cascade of gradient boosted trees (GBTs) to improve the limited representation learning ability of such previous work. A variant of target propagation (Lee et al., [2015](#bib.bib40)) was designed to enable the end-to-end training of cascaded GBTs, each of which is non-differentiable and thus not amenable to back-propagation.

## Appendix C Set-up details

#### Data:

we perform our experiments on the SARCOS robot inverse dynamics dataset333<http://www.gaussianprocess.org/gpml/data/>, the MNIST digit classification task (LeCun et al., [1998](#bib.bib38)) and the CIFAR-10 object recognition task (Krizhevsky & Hinton, [2009](#bib.bib35)). The SARCOS dataset consists of 44,484 training and 4,449 testing examples, where the goal is to map from the 21-dimensional input space (7 joint positions, 7 joint velocities and 7 joint accelerations) to the corresponding 7 joint torques (Vijayakumar & Schaal, [2000](#bib.bib68)). No dataset preprocessing or augmentation is used. The MNIST dataset consists of 60,000

6000060,000 training and 10,000

1000010,000 testing examples, all of which are 28×28282828\times 28 grayscale images of digits from 00 to 999 (101010 classes). The dataset is preprocessed by subtracting the mean, but no data augmentation is used. The CIFAR-10 dataset consists of 50,000

5000050,000 training and 10,000

1000010,000 testing examples, all of which are 32×32323232\times 32 coloured natural images drawn from 101010 classes. We adopt an augmentation scheme widely used in the literature (Goodfellow et al., [2013](#bib.bib18); Lin et al., [2014](#bib.bib43); Springenberg et al., [2015](#bib.bib61); He et al., [2016](#bib.bib22); Huang et al., [2017](#bib.bib24)) where images are zero-padded with 4 pixels on each side, randomly cropped and horizontally mirrored. For all three datasets, we hold out 10%percent1010\% of training images as a validation set. The best model is selected based on the validation accuracy over the course of ANT training, spanning both the growth phase and the refinement phase, and its test accuracy is reported.

#### Training:

both the growth phase and the refinement phase of ANTs are performed on a single Titan X GPU on all three datasets. For all the experiments in this paper, we employ the following training protocol: (1) optimise parameters using Adam (Kingma & Ba, [2014](#bib.bib32)) with initial learning rate of 10−3superscript10310^{-3} and β=[0.9,0.999]𝛽0.90.999\beta=[0.9,0.999], with minibatches of size 512512512; (2) during the growth phase, employ early stopping with a patience of 555, that is, training is stopped after 5 epochs of no progress on the validation set; (3) during the refinement phase, train for 300300300 epochs for SARCOS, 100100100 epochs for MNIST and 200200200 epochs for CIFAR-10, decreasing the learning rate by a factor of 101010 at every multiple of 505050. Training times are provided in Supp. Sec. [D](#A4 "Appendix D Training times ‣ Adaptive Neural Trees").

We observe that the patience level is an important hyperparameter which affects the quality of the growth phase; very low or high patience levels result in new modules underfitting or overfitting locally, thus preventing meaningful further growth and limiting the accuracy of the resultant models. We tuned this hyperparameter using the validation sets, and set the patience level to 555, which produced consistently good performance on SARCOS, MNIST and CIFAR-10 datasets across different specifications of primitive modules. A quantitative evaluation on CIFAR-10 is given in Supp. Sec. [E](#A5 "Appendix E Effect of training steps in the growth phase ‣ Adaptive Neural Trees").

In the SARCOS experiments, all the non-NN-based methods were trained using scikit-learn (Pedregosa et al., [2011](#bib.bib48)). Hidden layers in the baseline MLPs are followed by tanh non-linearities and contain 256 units to be consistent with the complexity of transformer modules.

#### Primitive modules:

we train ANTs with a range of primitive modules as shown in Tab. 2 in the main text. For simplicity, we define the modules based on three types of NN layers: convolutional, global-average-pooling (GAP) and fully-connected (FC). Solver modules are fixed as linear models e.g. linear classifier and linear regression. Router modules are binary classifiers with a sigmoid output. All convolutional and FC layer are followed by ReLU or tanh non-linearities, except in the last layers of solvers and routers. For image classification experiments, we also apply 2×2222\times 2 max-pooling to feature maps after every d𝑑d transformer modules where d𝑑d is the downsample frequency. For the SARCOS regression experiment, hidden layers in the routers and transformers contain 256 units. We balance the number of parameters in the router and transformer modules to be of the same order of magnitude to avoid favouring either partitioning the data or learning more expressive features.

## Appendix D Training times

Tab. [5](#A4.T5 "Table 5 ‣ Appendix D Training times ‣ Adaptive Neural Trees") summarises the time taken on a single Titan X GPU for the growth phase and refinement phase of various ANTs, and compares against the training time of All-CNN (Springenberg et al., [2015](#bib.bib61)). Local optimisation during the growth phase means that the gradient computation is constrained to the newly added component of the graph, allowing us to grow a good candidate model under 333 hours on one GPU.

Table 5: Training time comparison. Time and number of epochs taken for the growth and refinement phase are shown. along with the time required to train the baseline, All-CNN (Springenberg et al., [2015](#bib.bib61)).

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Growth | | Fine-tune | |
| Model | Time | Epochs | Time | Epochs |
| All-CNN (baseline) | – | – | 1.1 (hr) | 200 |
| ANT-CIFAR10-A | 1.3 (hr) | 236 | 1.5 (hr) | 200 |
| ANT-CIFAR10-B | 0.8 (hr) | 313 | 0.9 (hr) | 200 |
| ANT-CIFAR10-C | 0.7 (hr) | 285 | 0.8 (hr) | 200 |

![Refer to caption](/html/1807.06699/assets/figures/fig_patience.png)


Figure 4: Effect of patience level on the validation accuracy trajectory during training. Each curve shows the validation accuracy on CIFAR-10 dataset.

## Appendix E Effect of training steps in the growth phase

Fig. [4](#A4.F4 "Figure 4 ‣ Appendix D Training times ‣ Adaptive Neural Trees") compares the validation accuracies of the same ANT-CIFAR-C model trained on the CIFAR-10 dataset with varying levels of patience during early stopping in the growth phase. A higher patience level corresponds to more training epochs for optimising new modules in the growth phase. When the patience level is 1, the architecture growth terminates prematurely and plateaus at low accuracy at 80%percent8080\%. On the other hand, a patience level of 15 causes the model to overfit locally with 87%percent8787\%. The patience level of 5 gives the best results with 91%percent9191\% validation accuracy.

## Appendix F Expert specialisation

We investigate if the learned routing strategy is meaningful by comparing the classification accuracy of our default path-wise inference against that of the predictions from the leaf node with the smallest reaching probability. Tab. [6](#A6.T6 "Table 6 ‣ Appendix F Expert specialisation ‣ Adaptive Neural Trees") shows that using the least likely “expert” leads to a substantial drop in classification accuracy, down to close to that of random guess or even worse for large trees (ANT-MNIST-C and ANT-CIFAR10-C). This demonstrates that features in ANTs become specialised to the subsets of the partitioned input space at lower levels in the tree hierarchy.

Table 6: Comparison of classification performance between the default single-path inference scheme and the prediction based on the least likely expert. between the

|  |  |  |
| --- | --- | --- |
| Module Spec. | Error % | Error % |
|  | (Selected path) | (Least likely path) |
| ANT-MNIST-A | 0.69 | 86.18 |
| ANT-MNIST-B | 0.73 | 81.98 |
| ANT-MNIST-C | 1.68 | 98.84 |
| ANT-CIFAR10-A | 8.32 | 74.28 |
| ANT-CIFAR10-B | 9.18 | 89.74 |
| ANT-CIFAR10-C | 9.34 | 97.52 |

## Appendix G Visualisation of discovered architectures

Fig. [5](#A7.F5 "Figure 5 ‣ Appendix G Visualisation of discovered architectures ‣ Adaptive Neural Trees") shows ANT architectures discovered on the MNIST (i-iii) and CIFAR-10 (iv-vi) datasets. We observe three notable trends. Firstly, a large proportion of the learned routers separate examples based on their classes (red histograms) with very high confidence (blue histograms). The ablation study in Sec. 5. 1 (Tab. 4 in the main text) shows that such hierarchical clustering benefits predictive performance, while the conditional computation enables more lightweight inference (Tab. 3 in the main text). Secondly, most architectures learn a few levels of features before resorting to primarily splits. However, over half of the architectures (ii-v) still learn further representations beyond the first split. Secondly, all architectures are unbalanced. This reflects the fact that some groups of samples may be easier to classify than others. This property is reflected by traditional DT algorithms, but not “neural” tree-structured models with pre-specified architectures (Laptev & Buhmann, [2014](#bib.bib36); Frosst & Hinton, [2017](#bib.bib17); Kontschieder et al., [2015](#bib.bib34); Ioannou et al., [2016](#bib.bib26)).

![Refer to caption](/html/1807.06699/assets/x4.png)


Figure 5: Illustration of discovered ANT architectures. (i) ANT-MNIST-A, (ii) ANT-MNIST-B, (iii) ANT-MNIST-C, (iv) ANT-CIFAR10-A, (v) ANT-CIFAR10-B, (vi) ANT-CIFAR10-C. Histograms in red and blue show the class distributions and path probabilities at respective nodes. Small black circles on the edges represent transformers, circles in white at the internal nodes represent routers, and circles in gray are solvers. The small white circles on the edges denote specific cases where transformers are identity functions.

## Appendix H FLOPS

Tab.[7](#A8.T7 "Table 7 ‣ Appendix H FLOPS ‣ Adaptive Neural Trees") reports the floating point operations per second (FLOPS) of ANT models for two inference schemes. The results for ResNet110 and DenseNet were retrieved from (Guan et al., [2017](#bib.bib20)) and (Huang et al., [2018](#bib.bib25)), respectively. The FLOPs of all other models were computed using TorchStat toolbox available at <https://github.com/Swall0w/torchstat>. Using single-path inference reduces FLOPS in all ANT models to varying degrees.

Table 7: Comparison of FLOPs.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Model | FLOPS | FLOPS |
|  |  | (multi-path) | (single-path) |
| MNIST | Linear Classifier | 8K | - |
| LeNet-5 | 231 K | - |
| ANT-MNIST-C | 99K | 83K |
| ANT-MNIST-B | 346K | 331K |
| ANT-MNIST-A | 382K | 380K |
| CIFAR-10 | Net-in-Net | 222M | - |
| All-CNN | 245M | - |
| ResNet-110 | 256M | - |
| DenseNet-BC (k=24) | 9388M | - |
| ANT-CIFAR10-C | 66M | 61M |
| ANT-CIFAR10-B | 163M | 149M |
| ANT-CIFAR10-A | 254M | 243M |

## Appendix I Ensembling

As with traditional DTs (Breiman, [2001](#bib.bib5)) and NNs (Hansen & Salamon, [1990](#bib.bib21)), ANTs can be ensembled to gain improved performance. In Tab. [8](#A9.T8 "Table 8 ‣ Appendix I Ensembling ‣ Adaptive Neural Trees") we show the results of ensembling 8 ANTs (using the “-A” configurations for classification), each of which is trained with a randomly chosen split between training and validation sets. We compare against the single tree models, trained with the default split. In all cases both the multi-path and single-path inference performance is noticeably improved, and in MNIST we reach close to state-of-the-art performance (0.29% versus 0.25% (Sabour et al., [2017](#bib.bib54))) with significantly fewer parameters (851k versus 8.2M).

Table 8: Comparison of prediction errors of a single ANT versus an ensemble of 8, with predictions averaged over all ANTs in the ensemble.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | MNIST (Class Error %) | | CIFAR-10 (Class Error %) | | SARCOS (MSE) | |
|  | Multi-path | Single-path | Multi-path | Single-path | Multi-path | Single-path |
| Single model | 0.64 | 0.69 | 8.31 | 8.32 | 1.384 | 1.542 |
| Ensemble | 0.29 | 0.30 | 7.76 | 7.79 | 1.226 | 1.372 |




Table 9: Parameter counts for a single ANT versus an ensemble of 8.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | MNIST (No. Params.) | | CIFAR-10 (No. Params.) | | SARCOS (No. Params.) | |
|  | Multi-path | Single-path | Multi-path | Single-path | Multi-path | Single-path |
| Single model | 100,596 | 84,935 | 1.4M | 1.0M | 103,823 | 61,640 |
| Ensemble | 850,775 | 655,449 | 8.7M | 7.4M | 598,280 | 360,766 |

[◄](/html/1807.06697)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1807.06699)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1807.06699)
[View original  
on arXiv](https://arxiv.org/abs/1807.06699)[►](/html/1807.06701)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 15 23:12:59 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
