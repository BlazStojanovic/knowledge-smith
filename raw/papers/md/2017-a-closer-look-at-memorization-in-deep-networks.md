---
arxiv: '1706.05394'
authors:
- Devansh Arpit
- Stanisław Jastrzębski
- Nicolas Ballas
- David Krueger
- Emmanuel Bengio
- Maxinder S. Kanwal
- Tegan Maharaj
- Asja Fischer
- Aaron Courville
- Yoshua Bengio
- Simon Lacoste-Julien
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: A Closer Look at Memorization in Deep Networks
url: http://arxiv.org/abs/1706.05394v2
year: 2017
---

[1706.05394] A Closer Look at Memorization in Deep Networks















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



# A Closer Look at Memorization in Deep Networks

Devansh Arpit
  
Stanisław Jastrzębski
  
Nicolas Ballas
  
David Krueger
  
Emmanuel Bengio
  
Maxinder S. Kanwal
  
Tegan Maharaj
  
Asja Fischer
  
Aaron Courville
  
Yoshua Bengio
  
Simon Lacoste-Julien

###### Abstract

We examine the role of memorization in deep learning, drawing connections to capacity, generalization, and adversarial robustness.
While deep networks are capable of memorizing noise data, our results suggest that they tend to prioritize learning simple patterns first.
In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks (DNNs) on noise vs. real data.
We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data.
Our analysis suggests that the notions of effective capacity which are dataset independent are unlikely to explain the generalization performance of deep networks when trained with gradient based methods because training data itself plays an important role in determining the degree of memorization.

deep learning, deep networks, capacity, regularization

## 1 Introduction

The traditional view of generalization holds that a model with sufficient capacity (e.g. more parameters than training examples) will be able to “memorize” each example, overfitting the training set and yielding poor generalization to validation and test sets (Goodfellow et al., [2016](#bib.bib12)).
Yet deep neural networks (DNNs) often achieve excellent generalization performance with massively over-parameterized models. This phenomenon is not well-understood.

From a representation learning perspective, the generalization capabilities of DNNs are believed to stem from their incorporation of good generic priors (see, e.g., Bengio et al. ([2009](#bib.bib3))).
Lin & Tegmark ([2016](#bib.bib23)) further suggest that the priors of deep learning are well suited to the physical world.
But while the priors of deep learning may help explain why DNNs learn to efficiently represent complex real-world functions, they are not restrictive enough to rule out memorization.

On the contrary, deep nets are known to be universal approximators, capable of representing arbitrarily complex functions given sufficient capacity (Cybenko, [1989](#bib.bib9); Hornik et al., [1989](#bib.bib16)).
Furthermore, recent work has shown that the expressiveness of DNNs grows exponentially with depth (Montufar et al., [2014](#bib.bib27); Poole et al., [2016](#bib.bib29)).
These works, however, only examine the representational capacity, that is, the set of hypotheses a model is capable of expressing via some value of its parameters.

Because DNN optimization is not well-understood, it is unclear which of these hypotheses can actually be reached by gradient-based training (Bottou, [1998](#bib.bib6)).
In this sense, optimization and generalization are entwined in DNNs.
To account for this, we formalize a notion of the effective capacity (EC) of a learning algorithm 𝒜𝒜\mathcal{A} (defined by specifying both the model and the training procedure, e.g.,“train the LeNet architecture (LeCun et al., [1998](#bib.bib22)) for 100 epochs using stochastic gradient descent (SGD) with a learning rate of 0.010.010.01”) as the set of hypotheses which can be reached by applying that learning algorithm on some dataset. Formally, using set-builder notation:

|  |  |  |
| --- | --- | --- |
|  | E​C​(𝒜)={h∣∃𝒟​ such that ​h∈𝒜​(𝒟)},𝐸𝐶𝒜conditional-setℎ𝒟 such that ℎ𝒜𝒟EC(\mathcal{A})=\{h\mid\exists\mathcal{D}\text{ such that }h\in\mathcal{A}(\mathcal{D})\}\enspace, |  |

where 𝒜​(𝒟)𝒜𝒟\mathcal{A}(\mathcal{D})
represents the set of hypotheses that is reachable by 𝒜𝒜\mathcal{A} on a dataset 𝒟𝒟\mathcal{D}111
Since 𝒜𝒜\mathcal{A} can be stochastic, 𝒜​(𝒟)𝒜𝒟\mathcal{A}(\mathcal{D}) is a set.
.

One might suspect that DNNs effective capacity is sufficiently limited by gradient-based training and early stopping to resolve the apparent paradox between DNNs’ excellent generalization and their high representational capacity.
However, the experiments of Zhang et al. ([2017](#bib.bib40)) suggest that this is not the case.
They demonstrate that DNNs are able to fit pure noise without even needing substantially longer training time.
Thus even the effective capacity of DNNs may be too large, from the point of view of traditional learning theory.

By demonstrating the ability of DNNs to “memorize” random noise, Zhang et al. ([2017](#bib.bib40)) also raise the question whether deep networks use similar memorization tactics on real datasets.
Intuitively, a brute-force memorization approach to fitting data does not capitalize on patterns shared between training examples or features; the *content* of what is memorized is irrelevant.
A paradigmatic example of a memorization algorithm is k-nearest neighbors (Fix & Hodges Jr, [1951](#bib.bib10)).
Like Zhang et al. ([2017](#bib.bib40)), we do not formally define memorization; rather, we investigate this intuitive notion of memorization by training DNNs to fit random data.

### Main Contributions

We operationalize the definition of “memorization” as the behavior exhibited by DNNs trained on noise, and conduct a series of experiments that contrast the learning dynamics of DNNs on real vs. noise data.
Thus, our analysis builds on the work of Zhang et al. ([2017](#bib.bib40)) and further investigates the role of memorization in DNNs.

Our findings are summarized as follows:

1. 1.

   There are qualitative differences in DNN optimization behavior on real data vs. noise.
   In other words, DNNs do not just memorize real data (Section  [3](#S3 "3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")).
2. 2.

   DNNs learn simple patterns first, before memorizing (Section  [4](#S4 "4 DNNs Learn Patterns First ‣ A Closer Look at Memorization in Deep Networks")).
   In other words, DNN optimization is content-aware, taking advantage of patterns shared by multiple training examples.
3. 3.

   Regularization techniques can differentially hinder memorization in DNNs while preserving their ability to learn about real data (Section  [5](#S5 "5 Effect of Regularization on Learning ‣ A Closer Look at Memorization in Deep Networks")).

## 2 Experiment Details

We perform experiments on MNIST (LeCun et al., [1998](#bib.bib22)) and CIFAR10 ([Krizhevsky et al.,](#bib.bib20) ) datasets.
We investigate two classes of models: 2-layer multi-layer perceptrons (MLPs) with rectifier linear units (ReLUs) on MNIST and convolutional neural networks (CNNs) on CIFAR10.
If not stated otherwise, the MLPs have 4096 hidden units per layer and are trained for 100010001000 epochs with SGD and learning rate 0.010.010.01.
The CNNs are a small Alexnet-style CNN222Input →→\rightarrow Crop(2,2) →→\rightarrow Conv(200,5,5) →→\rightarrow BN →→\rightarrow ReLU →→\rightarrow MaxPooling(3,3) →→\rightarrow Conv(200,5,5) →→\rightarrow BN→→\rightarrow ReLU→→\rightarrow MaxPooling(3,3) →→\rightarrow Dense(384) →→\rightarrow BN →→\rightarrow ReLU →→\rightarrow Dense(192) →→\rightarrow BN →→\rightarrow ReLU →→\rightarrow Dense(##\#classes) →→\rightarrow Softmax. Here Crop(. , .) crops height and width from both sides with respective values.  (as in Zhang et al. ([2017](#bib.bib40))), and are trained using SGD with momentum=0.90.90.9 and learning rate of 0.010.010.01, scheduled to drop by half every 15 epochs.

Following Zhang et al. ([2017](#bib.bib40)), in many of our experiments we replace either (some portion of) the labels (with random labels), or the inputs (with i.i.d. Gaussian noise matching the real dataset’s mean and variance) for some fraction of the training set.
We use *randX* and *randY* to denote datasets with (100%, unless specified) noisy inputs and labels (respectively).

## 3 Qualitative Differences of DNNs Trained on Random vs. Real Data

Zhang et al. ([2017](#bib.bib40)) empirically demonstrated that DNNs are capable of fitting random data, which implicitly necessitates some high degree of memorization.
In this section, we investigate whether DNNs employ similar memorization strategy when trained on real data.
In particular, our experiments highlight some qualitative differences between DNNs trained on real data vs. random data, supporting the fact that DNNs do not use brute-force memorization to fit real datasets.

### 3.1 Easy Examples as Evidence of Patterns in Real Data

A brute-force memorization approach to fitting data should apply equally well to different training examples.
However, if a network is learning based on patterns in the data, some examples may fit these patterns better than others.
We show that such “easy examples” (as well as correspondingly “hard examples”) are common in real, but not in random, datasets.
Specifically, for each setting (real data, randX, randY), we train an MLP for a single epoch starting from 100 different random initializations and shufflings of the data.
We find that, for real data, many examples are consistently classified (in)correctly after a single epoch, suggesting that different examples are significantly easier or harder in this sense.
For noise data, the difference between examples is much less, indicating that these examples are fit (more) independently.
Results are presented in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks").

For randX, apparent differences in difficulty are well modeled as random Binomial noise.
For randY, this is not the case, indicating some use of shared patterns.
Visualizing first-level features learned by a CNN supports this hypothesis (Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")).

![Refer to caption](/html/1706.05394/assets/fig_png/replacement_figure_1.png)


Figure 1: Average (over 100 experiments) misclassification rate for each of 1000 examples after one epoch of training.
This measure of an example’s difficulty is much more variable in real data.
We conjecture this is because the easier examples are explained by some simple patterns, which are reliably learned within the first epoch of training.
We include 1000 points samples from a binomial distribution with n=100𝑛100n=100 and p𝑝p equal to the average estimated P(correct) for randX, and note that this curve closely resembles the randX curve, suggesting that random inputs are all equally difficult.

![Refer to caption](/html/1706.05394/assets/x1.png)


Figure 2: Filters from first layer of network trained on CIFAR10 (left) and randY (right).



![Refer to caption](/html/1706.05394/assets/x2.png)

![Refer to caption](/html/1706.05394/assets/x3.png)

Figure 3: Plots of the Gini coefficient of g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}} over examples 𝐱𝐱\mathbf{x} (see section [3.2](#S3.SS2 "3.2 Loss-Sensitivity in Real vs. Random Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")) as training progresses, for a 1000-example real dataset (14x14 MNIST) versus random data. On the left, Y𝑌Y is the normal class label; on the right, there are as many classes as examples, the network has to learn to map each example to a unique class.

### 3.2 Loss-Sensitivity in Real vs. Random Data

To further investigate the difference between real and fully random inputs, we propose a proxy measure of memorization via gradients. Since we cannot measure quantitatively how much each training sample 𝐱𝐱\mathbf{x} is memorized, we instead measure the effect of each sample on the average loss.
That is, we measure the norm of the loss gradient with respect to a previous example 𝐱𝐱\mathbf{x} after t𝑡t SGD updates.
Let ℒtsubscriptℒ𝑡\mathcal{L}\_{t} be the loss after t𝑡t updates; then the sensitivity measure is given by

|  |  |  |
| --- | --- | --- |
|  | g𝐱t=‖∂ℒt/∂𝐱‖1.subscriptsuperscript𝑔𝑡𝐱subscriptnormsubscriptℒ𝑡𝐱1g^{t}\_{\mathbf{x}}=\|\partial\mathcal{L}\_{t}/\partial\mathbf{x}\|\_{1}\enspace. |  |

The parameter update from training on 𝐱𝐱\mathbf{x} influences all future ℒtsubscriptℒ𝑡\mathcal{L}\_{t} indirectly by changing the subsequent updates on different training examples.
We denote the average over g𝐱tsubscriptsuperscript𝑔𝑡𝐱g^{t}\_{\mathbf{x}} after T𝑇T steps as g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}}, and refer to it as *loss-sensitivity*. Note that we only report ℓ1superscriptℓ1\ell^{1}-norm results, but that results stay very similar using ℓ2superscriptℓ2\ell^{2}-norm and infinity norm.

We compute g𝐱tsubscriptsuperscript𝑔𝑡𝐱g^{t}\_{\mathbf{x}} by unrolling t𝑡t SGD steps and applying backpropagation over the unrolled computation graph, as done by Maclaurin et al. ([2015](#bib.bib24)).
Unlike Maclaurin et al. ([2015](#bib.bib24)), we only use this procedure to compute g𝐱tsubscriptsuperscript𝑔𝑡𝐱g^{t}\_{\mathbf{x}}, and do not modify the training procedure in any way.

We find that for real data, only a subset of the training set has high g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}}, while for random data, g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}} is high for virtually all examples. We also find a different behavior when each example is given a unique class; in this scenario, the network has to learn to identify each example uniquely,
yet still behaves differently when given real data than when given random data as input.

We visualize (Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")) the spread of g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}} as training progresses by computing the Gini coefficient over 𝐱𝐱\mathbf{x}’s. The Gini coefficient (Gini, [1913](#bib.bib11)) is a measure of the inequality among values of a frequency distribution; a coefficient of 0 means exact equality (i.e., all values are the same), while a coefficient of 1 means maximal inequality among values.
We observe that, when trained on real data, the network has a high g¯𝐱subscript¯𝑔𝐱\bar{g}\_{\mathbf{x}} for a few examples, while on random data the network is sensitive to most examples. The difference between the random data scenario, where we know the neural network needs to do memorization, and the real data scenario, where we’re trying to understand what happens, leads us to believe that this measure is indeed sensitive to memorization. Additionally, these results suggest that when being trained on real data, the neural network probably does not memorize, or at least not in the same manner it needs to for random data.

In addition to the different behaviors for real and random data described above, we also consider a class specific loss-sensitivity: g¯i,j=𝔼(x,y)​1/T​∑tT|∂ℒt​(y=i)/∂xy=j|subscript¯𝑔

𝑖𝑗subscript𝔼𝑥𝑦1𝑇superscriptsubscript𝑡𝑇subscriptℒ𝑡𝑦𝑖subscript𝑥𝑦𝑗\bar{g}\_{i,j}=\mathbb{E}\_{(x,y)}\nicefrac{{1}}{{T}}\sum\_{t}^{T}|\partial\mathcal{L}\_{t}(y=i)/\partial x\_{y=j}|, where ℒt​(y=i)subscriptℒ𝑡𝑦𝑖\mathcal{L}\_{t}(y=i) is the term in the crossentropy sum corresponding to class i𝑖i.
We observe that the loss-sensitivity
w.r.t. class i𝑖i for
training examples of class j𝑗j
is higher when i=j𝑖𝑗i=j, but more spread out for real data (see Figure [4](#S3.F4 "Figure 4 ‣ 3.2 Loss-Sensitivity in Real vs. Random Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")).
An interpretation of this is that for real data there are more interesting cross-category patterns that can be learned than for random data.

![Refer to caption](/html/1706.05394/assets/x4.png)


Figure 4: Plots of per-class gxsubscript𝑔𝑥g\_{x} (see previous figure; log scale), a cell i,j

𝑖𝑗i,j represents the average |∂ℒ​(y=i)/∂xy=j|ℒ𝑦𝑖subscript𝑥𝑦𝑗|\partial\mathcal{L}(y=i)/\partial x\_{y=j}|, i.e. the loss-sensitivity of examples of class i𝑖i w.r.t. training examples of class j𝑗j. Left is real data, right is random data.

Figure [3](#S3.F3 "Figure 3 ‣ 3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks") and [4](#S3.F4 "Figure 4 ‣ 3.2 Loss-Sensitivity in Real vs. Random Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks") were obtained by training a fully-connected network with 2 layers of 16 units on 1000 downscaled 14×14141414\times 14 MNIST digits using SGD.

### 3.3 Capacity and Effective Capacity

In this section, we investigate the impact of capacity and effective capacity on learning of datasets having different amounts of random input data or random labels.

#### 3.3.1 Effects of capacity and dataset size on validation performances

In a first experiment, we study how overall model capacity impacts the validation performances for datasets with different amounts of noise.
On MNIST, we found that the optimal validation performance requires a higher capacity model in the presence of noise examples (see Figure [5](#S3.F5 "Figure 5 ‣ 3.3.1 Effects of capacity and dataset size on validation performances ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")).
This trend was consistent for noise inputs on CIFAR10, but we did not notice any relationship between capacity and validation performance on random *labels* on CIFAR10.

This result contradicts the intuitions of traditional learning theory, which suggest that capacity should be restricted, in order to enforce the learning of (only) the most regular patterns.
Given that DNNs can perfectly fit the training set in any case, we hypothesize that that higher capacity allows the network to fit the noise examples in a way that does not interfere with learning the real data.
In contrast, if we were simply to *remove* noise examples, yielding a smaller (clean) dataset, a *lower* capacity model would be able to achieve optimal performance.

![Refer to caption](/html/1706.05394/assets/fig_png/plot1_.png)


Figure 5: Performance as a function of capacity in 2-layer MLPs trained on (noisy versions of) MNIST. For real data, performance is already very close to maximal with 4096 hidden units, but when there is noise in the dataset, higher capacity is needed.



![Refer to caption](/html/1706.05394/assets/fig_png/plot2_.png)

![Refer to caption](/html/1706.05394/assets/fig_png/plot3.png)

Figure 6: 
Time to convergence as a function of capacity with dataset size fixed to 50000 (left), or dataset size with capacity fixed to 4096 units (right).
“Noise level” denotes to the proportion of training points whose inputs are replaced by Gaussian noise.
Because of the patterns underlying real data, having more capacity/data does not decrease/increase training time as much as it does for noise data.

#### 3.3.2 Effects of capacity and dataset size on training time

Our next experiment measures time-to-convergence, i.e. how many epochs it takes to reach 100% training accuracy.
Reducing the capacity or increasing the size of the dataset slows down training as well for real as for noise data333
Regularization can also increase time-to-convergence; see section [5](#S5 "5 Effect of Regularization on Learning ‣ A Closer Look at Memorization in Deep Networks").
.
However, the effect is more severe for datasets containing noise, as our experiments in this section show (see Figure [6](#S3.F6 "Figure 6 ‣ 3.3.1 Effects of capacity and dataset size on validation performances ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")).

Effective capacity of a DNN can be increased by increasing the representational capacity (e.g. adding more hidden units) or training for longer.
Thus, increasing the number of hidden units decreases the number of training iterations needed to fit the data, up to some limit.
We observe *stronger* diminishing returns from increasing representational capacity for real data, indicating that this limit is lower, and a smaller representational capacity is sufficient, for real datasets.

Increasing the number of examples (keeping representational capacity fixed) also increases the time needed to memorize the training set.
In the limit, the representational capacity is simply insufficient, and memorization is not feasible.
On the other hand, when the relationship between inputs and outputs is meaningful, new examples simply give more (possibly redundant) clues as to what the input →→\rightarrow output mapping is.
Thus, in the limit, an idealized learner should be able to predict unseen examples perfectly, absent noise.
Our experiments demonstrate that time-to-convergence is not only longer on noise data (as noted by Zhang et al. ([2017](#bib.bib40))), but also, *increases* substantially as a function of dataset size, relative to real data.
Following the reasoning above, this suggests that our networks are learning to extract patterns in the data, rather than memorizing.

![Refer to caption](/html/1706.05394/assets/fig_png/mnist_noisex_accuracy_2.png)

![Refer to caption](/html/1706.05394/assets/fig_png/mnist_noisex_critical_valid_2.png)

(a) Noise added on classification inputs.

![Refer to caption](/html/1706.05394/assets/fig_png/mnist_noisey_accuracy_2.png)

![Refer to caption](/html/1706.05394/assets/fig_png/mnist_noisey_critical_valid_2.png)

(b) Noise added on classification labels.

Figure 7: Accuracy (left in each pair, solid is train, dotted is validation) and Critical sample ratios (right in each pair) for MNIST.



![Refer to caption](/html/1706.05394/assets/fig_png/cifar10_noisex_accuracy_2.png)

![Refer to caption](/html/1706.05394/assets/fig_png/cifar10_noisex_critical_valid_2.png)

(a) Noise added on classification inputs.

![Refer to caption](/html/1706.05394/assets/fig_png/cifar10_noisey_accuracy_2.png)

![Refer to caption](/html/1706.05394/assets/fig_png/cifar10_noisey_critical_valid_2.png)

(b) Noise added on classification labels.

Figure 8: Accuracy (left in each pair, solid is train, dotted is validation) and Critical sample ratios (right in each pair) for CIFAR10.

## 4 DNNs Learn Patterns First

This section aims at studying how the complexity of the hypotheses learned by DNNs evolve during training for real data vs. noise data.
To achieve this goal, we build on the intuition that the number of different decision regions into which an input space is partitioned reflects the complexity of the learned hypothesis (Sokolic et al., [2016](#bib.bib33)).
This notion is similar in spirit to the degree to which a function can scatter random labels: a higher density of decision boundaries in the data space allows more samples to be scattered.

Therefore, we estimate the complexity by measuring how densely points on the data manifold are present around the model’s decision boundaries.
Intuitively, if we were to randomly sample points from the data distribution, a smaller fraction of points in the proximity of a decision boundary suggests that the learned hypothesis is simpler.

### 4.1 Critical Sample Ratio (CSR)

Here we introduce the notion of a critical sample, which we use to estimate the density of decision boundaries as discussed above.
Critical samples are a subset of a dataset such that for each such sample 𝐱𝐱\mathbf{x}, there exists at least one adversarial example 𝐱^^𝐱\hat{\mathbf{x}} in the proximity of 𝐱𝐱\mathbf{x}.
Specifically, consider a classification network’s output vector f​(𝐱)=(f1​(𝐱),…,fk​(𝐱))∈ℝk𝑓𝐱subscript𝑓1𝐱…subscript𝑓𝑘𝐱superscriptℝ𝑘f(\mathbf{x})=(f\_{1}(\mathbf{x}),\dots,f\_{k}(\mathbf{x}))\in\mathbb{R}^{k} for a given input sample 𝐱∈ℝn𝐱superscriptℝ𝑛\mathbf{x}\in\mathbb{R}^{n} from the data manifold.
Formally we call a dataset sample 𝐱𝐱\mathbf{x} a *critical sample* if there exists a point 𝐱^^𝐱\hat{\mathbf{x}} such that,

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | arg⁡maxi⁡fi​(𝐱)≠arg⁡maxj⁡fj​(𝐱^)subscript𝑖subscript𝑓𝑖𝐱subscript𝑗subscript𝑓𝑗^𝐱\displaystyle\arg\max\_{i}f\_{i}(\mathbf{x})\neq\arg\max\_{j}f\_{j}(\hat{\mathbf{x}})\mspace{5.0mu} |  | (1) |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | s.t. ​∥𝐱−𝐱^∥∞≤rs.t. subscriptdelimited-∥∥𝐱^𝐱𝑟\displaystyle\mbox{s.t. }\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert\_{\infty}\leq r |  |

where r𝑟r is a fixed box size.
As in recent work on adversarial examples (Kurakin et al., [2016](#bib.bib21)) the above definition depends only on the predicted label arg⁡maxi⁡fi​(𝐱)subscript𝑖subscript𝑓𝑖𝐱\arg\max\_{i}f\_{i}(\mathbf{x}) of 𝐱𝐱\mathbf{x}, and not the true label (as in earlier work on adversarial examples, such as Szegedy et al. ([2013](#bib.bib34)); Goodfellow et al. ([2014](#bib.bib14))).

Following the above argument relating complexity to decision boundaries, a higher number of critical samples indicates a more complex hypothesis. Thus, we measure complexity as the critical sample ratio (CSR), that is, the fraction of data-points in a set |𝒟|𝒟|\mathcal{D}| for which we can find a critical sample: #​critical samples|𝒟|#critical samples𝒟\frac{\#\text{critical samples}}{|\mathcal{D}|}.

To identify whether a given data point 𝐱𝐱\mathbf{x} is a critical samples, we search for an adversarial sample 𝐱^^𝐱\hat{\mathbf{x}} within a box of radius r𝑟r.
To perform this search, we propose using Langevin dynamics applied to the fast gradient sign method (FGSM, Goodfellow et al. ([2014](#bib.bib14))) as shown in algorithm [1](#alg1 "Algorithm 1 ‣ 4.1 Critical Sample Ratio (CSR) ‣ 4 DNNs Learn Patterns First ‣ A Closer Look at Memorization in Deep Networks")444In our experiments, we set α=0.25𝛼0.25\alpha=0.25, β=0.2𝛽0.2\beta=0.2 and η𝜂\eta is samples from standard normal distribution..
We refer to this method as Langevin adversarial sample search (LASS).
While the FGSM search algorithm can get stuck at a points with zero gradient, LASS explores the box more thoroughly.
Specifically, a problem with first order gradient search methods (like FGSM) is that there might exist training points where the gradient is 0, but with a large 2n​dsuperscript2𝑛𝑑2^{nd} derivative corresponding to a large change in prediction in the neighborhood.
The noise added by the LASS algorithm during the search enables escaping from such points.

Algorithm 1  Langevin Adversarial Sample Search (LASS)

0:  𝐱∈ℝn𝐱superscriptℝ𝑛\mathbf{x}\in\mathbb{R}^{n}, α𝛼\alpha, β𝛽\beta, r𝑟r, noise process η𝜂\eta

0:  𝐱^^𝐱\hat{\mathbf{x}}

1:  converged = FALSE

2:  𝐱~←𝐱←~𝐱𝐱\tilde{\mathbf{x}}\leftarrow\mathbf{x}; 𝐱^←∅←^𝐱\hat{\mathbf{x}}\leftarrow\emptyset

3:  while not converged or max iter reached do

4:     Δ=α⋅sign​(∂fk​(𝐱)∂𝐱)+β⋅ηΔ⋅𝛼signsubscript𝑓𝑘𝐱𝐱⋅𝛽𝜂\Delta=\alpha\cdot\mbox{sign}(\frac{\partial f\_{k}(\mathbf{x})}{\partial\mathbf{x}})+\beta\cdot\eta

5:     𝐱~←𝐱~+Δ←~𝐱~𝐱Δ\tilde{\mathbf{x}}\leftarrow\tilde{\mathbf{x}}+\Delta

6:     for  i∈[n]𝑖delimited-[]𝑛i\in[n] do

7:        𝐱~i←{𝐱i+r⋅sign​(𝐱~i−𝐱i)i​f​|𝐱~i−𝐱i|>r𝐱~io​t​h​e​r​w​i​s​e←subscript~𝐱𝑖casessubscript𝐱𝑖⋅𝑟signsubscript~𝐱𝑖superscript𝐱𝑖𝑖𝑓subscript~𝐱𝑖subscript𝐱𝑖𝑟subscript~𝐱𝑖𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒\tilde{\mathbf{x}}\_{i}\leftarrow\left\{\begin{array}[]{ll}{\mathbf{x}}\_{i}+r\cdot\mbox{sign}(\tilde{\mathbf{x}}\_{i}-{\mathbf{x}}^{i})&if\lvert\tilde{\mathbf{x}}\_{i}-{\mathbf{x}}\_{i}\rvert>r\\
\tilde{\mathbf{x}}\_{i}&otherwise\\
\end{array}\right.

8:     end for

9:     if arg⁡maxi⁡f​(𝐱)≠arg⁡maxi⁡f​(𝐱~)subscript𝑖𝑓𝐱subscript𝑖𝑓~𝐱\arg\max\_{i}f(\mathbf{x})\neq\arg\max\_{i}f(\tilde{\mathbf{x}}) then

10:        converged = TRUE

11:        𝐱^←𝐱~←^𝐱~𝐱\hat{\mathbf{x}}\leftarrow\tilde{\mathbf{x}}

12:     end if

13:  end while

![Refer to caption](/html/1706.05394/assets/fig_png/valid_stability.png)


Figure 9: Critical sample ratio throughout training on CIFAR-10, random input (randX), and random label (randY) datasets.

### 4.2 Critical Samples Throughout Training

We now show that the number of critical samples is much higher for a deep network (specifically, a CNN) trained on noise data compared with real data.
To do so, we measure the number of critical samples in the validation set555
We also measure the number of critical samples in the training sets. Since we train our models using log loss, training points are pushed away from the decision boundary even after the network learns to classify them correctly. This leads to an initial rise and then fall of the number of critical samples in the training sets.
,
throughout training666We use a box size of 0.3, which is small enough in a 0-255 pixel scale to be unnoticeable by a human evaluator. Different values for r𝑟r were tested but did not change results qualitatively and lead to the same conclusions.
Results are shown in Figure [9](#S4.F9 "Figure 9 ‣ 4.1 Critical Sample Ratio (CSR) ‣ 4 DNNs Learn Patterns First ‣ A Closer Look at Memorization in Deep Networks").
A higher number of critical samples for models trained on noise data compared with those trained on real data suggests that the learned decision surface is more complex for noise data (randX and randY).
We also observe that the CSR increases gradually with increasing number of epochs and then stabilizes. This suggests that the networks learn gradually more complex hypotheses during training for all three datasets.

In our next experiment, we evaluate the performance and critical sample ratio of datasets with 20%percent2020\% to 80%percent8080\% of the training data replaced with either input or label noise.
Results for MNIST and CIFAR-10 are shown in Figures [7](#S3.F7 "Figure 7 ‣ 3.3.2 Effects of capacity and dataset size on training time ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks") and [8](#S3.F8 "Figure 8 ‣ 3.3.2 Effects of capacity and dataset size on training time ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks"), respectively.
For both randX and randY datasets, the CSR is higher for noisier datasets, reflecting the higher level of complexity of the learned prediction function.
The final and maximum validation accuracies are also both lower for noisier datasets, indicating that the noise examples interfere somewhat with the networks ability to learn about the real data.

More significantly, for randY datasets (Figures [7(b)](#S3.F7.sf2 "In Figure 7 ‣ 3.3.2 Effects of capacity and dataset size on training time ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks") and [8(b)](#S3.F8.sf2 "In Figure 8 ‣ 3.3.2 Effects of capacity and dataset size on training time ‣ 3.3 Capacity and Effective Capacity ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks")), the network achieves maximum accuracy on the validation set before achieving high accuracy on the training set.
Thus the model first learns the simple and general patterns of the real data before fitting the noise (which results in decreasing validation accuracy).
Furthermore, as the model moves from fitting real data to fitting noise, the CSR greatly increases, indicating the need for more complex hypotheses to explain the noise.
Combining this result with our results from Section [3.1](#S3.SS1 "3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks"), we conclude that real data examples are easier to fit than noise.

## 5 Effect of Regularization on Learning

Here we demonstrate the ability of regularization to degrade training performance on data with random labels, while maintaining generalization performance on real data.
Zhang et al. ([2017](#bib.bib40)) argue that explicit regularizations are not the main explanation of good generalization performance, rather SGD based optimization is largely responsible for it. Our findings extend their claim and indicate that explicit regularizations can substantially limit the speed of memorization of noise data without significantly impacting learning on real data.

We compare the performance of CNNs trained on CIFAR-10 and randY with the following regularizers: dropout (with dropout rates in range 00-0.90.90.9), input dropout (range 00-0.90.90.9), input Gaussian noise (with standard deviation in range 00-555), hidden Gaussian noise (range 00-0.30.30.3), weight decay (range 00-111) and additionally dropout with adversarial training (with weighting factor in range 0.20.20.2-0.70.70.7 and dropout in rate range 0.030.030.03-0.50.50.5).777We perform adversarial training using critical samples found by LASS algorithm with default parameters. 
We train a separate model for every combination of dataset, regularization technique, and regularization parameter.

The results are summarized in Figure [10](#S5.F10 "Figure 10 ‣ 5 Effect of Regularization on Learning ‣ A Closer Look at Memorization in Deep Networks").
For each combination of dataset and regularization technique, the final training accuracy on randY (x-axis) is plotted against the best validation accuracy on CIFAR-10 from amongst the models trained with different regularization parameters (y-axis).
Flat curves indicate that the corresponding regularization technique can reduce memorization when applied on random labeling, while resulting in the same validation accuracy on the clean validation set.
Our results show that different regularizers target memorization behavior to different extent – dropout being the most effective.
We find that dropout, especially coupled with adversarial training, is best at hindering memorization without reducing the model’s ability to learn.
Figure [11](#S5.F11 "Figure 11 ‣ 5 Effect of Regularization on Learning ‣ A Closer Look at Memorization in Deep Networks") additionally shows this effect for selected experiments (i.e. selected hyperparameter values) in terms of train loss.

![Refer to caption](/html/1706.05394/assets/fig_png/reg_pattern.png)


Figure 10: Effect of different regularizers on train accuracy (on noise dataset) vs. validation accuracy (on real dataset). Flatter curves indicate that memorization (on noise) can be capped without sacrificing generalization (on real data).



![Refer to caption](/html/1706.05394/assets/fig_png/randY_capping.png)

![Refer to caption](/html/1706.05394/assets/fig_png/realY_capping.png)

Figure 11: Training curves for different regularization techniques on random label (left) and real (right) data. The vertical ordering of the curves is different for random labels than for real data, indicating differences in the propensity of different regularizers to slow-down memorization.

## 6 Related Work

Our work builds on the experiments and challenges the interpretations of Zhang et al. ([2017](#bib.bib40)).
We make heavy use of their methodology of studying DNN training in the context of noise datasets. Zhang et al. ([2017](#bib.bib40)) show that DNNs can perfectly fit noise and thus that their generalization ability cannot be explained through traditional statistical learning theory (e.g., see (Vapnik & Vapnik, [1998](#bib.bib36); Bartlett et al., [2005](#bib.bib2))).
We agree with this finding, but show in addition that the degree of memorization and generalization in DNNs depends not only on the architecture and training procedure (including explicit regularizations), but also on the training data itself888We conclude the latter part based on experimental findings in sections [3](#S3 "3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks") and [4.2](#S4.SS2 "4.2 Critical Samples Throughout Training ‣ 4 DNNs Learn Patterns First ‣ A Closer Look at Memorization in Deep Networks").

Another direction we investigate is the relationship between regularization and memorization.
Zhang et al. ([2017](#bib.bib40)) argue that explicit and implicit regularizers (including SGD) might not explain or limit shattering of random data.
In this work we show that regularizers (especially dropout) do control the *speed* at which DNNs memorize. This is interesting since dropout is also known to prevent catastrophic forgetting (Goodfellow et al., [2013](#bib.bib13)) and thus in general it seems to help DNNs retain patterns.

A number of arguments support the idea that SGD-based learning imparts a regularization effect, especially with a small batch size (Wilson & Martinez, [2003](#bib.bib38)) or a small number of epochs (Hardt et al., [2015](#bib.bib15)).
Previous work also suggests that SGD prioritizes the learning of simple hypothesis first.
Sjoberg et al. ([1995](#bib.bib32)) showed that, for linear models, SGD first learns models with small ℓ2superscriptℓ2\ell^{2} parameter norm.
More generally, the efficacy of early stopping shows that SGD first learns simpler models (Yao et al., [2007](#bib.bib39)).
We extend these results, showing that DNNs trained with SGD learn patterns before memorizing, even in the presence of noise examples.

Various previous works have analyzed explanations for the generalization power of DNNs.
Montavon et al. ([2011](#bib.bib26)) use kernel methods to analyze the complexity of deep learning architectures, and find that network priors (e.g. implemented by the network structure of a CNN or MLP) control the speed of learning at each layer.
Neyshabur et al. ([2014](#bib.bib28)) note that the number of parameters does not control the effective capacity of a DNN, and that the reason for DNNs’ generalization is unknown.
We supplement this result by showing how the impact of representational capacity changes with varying noise levels. While exploring the effect of noise samples on learning dynamics has a long tradition (Bishop, [1995](#bib.bib4); An, [1996](#bib.bib1)), we are the first to examine relationships between the fraction of noise samples and other attributes of the learning algorithm, namely: capacity, training time and dataset size.

Multiple techniques for analyzing the training of DNNs have been proposed before, including looking at generalization error, trajectory length evolution (Raghu et al., [2016](#bib.bib30)), analyzing Jacobians associated to different layers ([Wang,](#bib.bib37) ; Saxe et al., [2013](#bib.bib31)), or the shape of the loss minima found by SGD (Im et al., [2016](#bib.bib17); Chaudhari et al., [2016](#bib.bib7); Keskar et al., [2016](#bib.bib18)).
Instead of measuring the sharpness of the loss for the learned hypothesis, we investigate the complexity of the learned hypothesis throughout training and across different datasets and regularizers, as measured by the critical sample ratio.
Critical samples refer to real data-points that have adversarial examples (Szegedy et al., [2013](#bib.bib34); Goodfellow et al., [2014](#bib.bib14)) nearby.
Adversarial examples originally referred to imperceptibly perturbed data-points that are confidently misclassified.
(Miyato et al., [2015](#bib.bib25)) define virtual adversarial examples via changes in the predictive distribution instead, thus extending the definition to unlabeled data-points.
Kurakin et al. ([2016](#bib.bib21)) recommend using this definition when training on adversarial examples, and it is the definition we use.

Two contemporary works perform in-depth explorations of topics related to our work.
Bojanowski & Joulin ([2017](#bib.bib5)) show that predicting random noise targets can yield state of the art results in unsupervised learning, corroborating our findings in Section  [3.1](#S3.SS1 "3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks"), especially Figure  [2](#S3.F2 "Figure 2 ‣ 3.1 Easy Examples as Evidence of Patterns in Real Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks").
Koh & Liang ([2017](#bib.bib19)) use influence functions to measure the impact on parameter changes during training, as in our Section [3.2](#S3.SS2 "3.2 Loss-Sensitivity in Real vs. Random Data ‣ 3 Qualitative Differences of DNNs Trained on Random vs. Real Data ‣ A Closer Look at Memorization in Deep Networks").
They explore several promising applications for this technique, including generation of adversarial training examples.

## 7 Conclusion

Our empirical exploration demonstrates qualitative differences in DNN optimization on noise vs. real data, all of which support the claim that DNNs trained with SGD-variants first use patterns, not brute force memorization, to fit real data.
However, since DNNs have the demonstrated ability to fit noise, it is unclear why they find generalizable solutions on real data; we believe that the deep learning priors including distributed and hierarchical representations likely play an important role.
Our analysis suggests that memorization and generalization in DNNs depend on network architecture and optimization procedure, but also on the data itself.
We hope to encourage future research on how properties of datasets influence the behavior of deep learning algorithms, and suggest a data-dependent understanding of DNN capacity as a research goal.

#### Acknowledgments

We thank Akram Erraqabi, Jason Jo and Ian Goodfellow for helpful discussions.
SJ was supported by Grant No. DI 2014/016644 from Ministry of Science and Higher Education, Poland. DA was supported by IVADO, CIFAR and NSERC.
EB was financially supported by the Samsung Advanced Institute of Technology (SAIT).
MSK and SJ were supported by MILA during the course of this work.
We acknowledge the computing resources provided by ComputeCanada and CalculQuebec.
Experiments were carried out using Theano (Theano Development Team, [2016](#bib.bib35)) and Keras (Chollet et al., [2015](#bib.bib8)).

## References

* An (1996)

  An, Guozhong.
  The effects of adding noise during backpropagation training on a
  generalization performance.
  *Neural computation*, 8(3):643–674, 1996.
* Bartlett et al. (2005)

  Bartlett, Peter L, Bousquet, Olivier, Mendelson, Shahar, et al.
  Local rademacher complexities.
  *The Annals of Statistics*, 33(4):1497–1537, 2005.
* Bengio et al. (2009)

  Bengio, Yoshua et al.
  Learning deep architectures for ai.
  *Foundations and trends® in Machine Learning*,
  2(1):1–127, 2009.
* Bishop (1995)

  Bishop, Chris M.
  Training with noise is equivalent to tikhonov regularization.
  *Neural computation*, 7(1):108–116, 1995.
* Bojanowski & Joulin (2017)

  Bojanowski, P. and Joulin, A.
  Unsupervised Learning by Predicting Noise.
  *ArXiv e-prints*, April 2017.
* Bottou (1998)

  Bottou, Léon.
  Online learning and stochastic approximations.
  *On-line learning in neural networks*, 17(9):142, 1998.
* Chaudhari et al. (2016)

  Chaudhari, Pratik, Choromanska, Anna, Soatto, Stefano, and LeCun, Yann.
  Entropy-sgd: Biasing gradient descent into wide valleys.
  *arXiv preprint arXiv:1611.01838*, 2016.
* Chollet et al. (2015)

  Chollet, François et al.
  Keras.
  <https://github.com/fchollet/keras>, 2015.
* Cybenko (1989)

  Cybenko, George.
  Approximation by superpositions of a sigmoidal function.
  *Mathematics of Control, Signals, and Systems (MCSS)*,
  2(4):303–314, 1989.
* Fix & Hodges Jr (1951)

  Fix, Evelyn and Hodges Jr, Joseph L.
  Discriminatory analysis-nonparametric discrimination: consistency
  properties.
  Technical report, DTIC Document, 1951.
* Gini (1913)

  Gini, Corrado.
  Variabilita e mutabilita.
  *Journal of the Royal Statistical Society*, 76(3),
  1913.
* Goodfellow et al. (2016)

  Goodfellow, Ian, Bengio, Yoshua, and Courville, Aaron.
  *Deep Learning*.
  MIT Press, 2016.
  <http://www.deeplearningbook.org>.
* Goodfellow et al. (2013)

  Goodfellow, Ian J, Mirza, Mehdi, Xiao, Da, Courville, Aaron, and Bengio,
  Yoshua.
  An empirical investigation of catastrophic forgetting in
  gradient-based neural networks.
  *arXiv preprint arXiv:1312.6211*, 2013.
* Goodfellow et al. (2014)

  Goodfellow, Ian J, Shlens, Jonathon, and Szegedy, Christian.
  Explaining and harnessing adversarial examples.
  *arXiv preprint arXiv:1412.6572*, 2014.
* Hardt et al. (2015)

  Hardt, Moritz, Recht, Benjamin, and Singer, Yoram.
  Train faster, generalize better: Stability of stochastic gradient
  descent.
  *arXiv preprint arXiv:1509.01240*, 2015.
* Hornik et al. (1989)

  Hornik, Kurt, Stinchcombe, Maxwell, and White, Halbert.
  Multilayer feedforward networks are universal approximators.
  *Neural networks*, 2(5):359–366, 1989.
* Im et al. (2016)

  Im, Daniel Jiwoong, Tao, Michael, and Branson, Kristin.
  An empirical analysis of deep network loss surfaces.
  *arXiv preprint arXiv:1612.04010*, 2016.
* Keskar et al. (2016)

  Keskar, Nitish Shirish, Mudigere, Dheevatsa, Nocedal, Jorge, Smelyanskiy,
  Mikhail, and Tang, Ping Tak Peter.
  On large-batch training for deep learning: Generalization gap and
  sharp minima.
  *arXiv preprint arXiv:1609.04836*, 2016.
* Koh & Liang (2017)

  Koh, Pang Wei and Liang, Percy.
  Understanding black-box predictions via influence functions.
  *arXiv preprint arXiv:1703.04730*, 2017.
* (20)

  Krizhevsky, Alex, Nair, Vinod, and Hinton, Geoffrey.
  Cifar-10 (canadian institute for advanced research).
  URL <http://www.cs.toronto.edu/~kriz/cifar.html>.
* Kurakin et al. (2016)

  Kurakin, Alexey, Goodfellow, Ian, and Bengio, Samy.
  Adversarial examples in the physical world.
  *arXiv preprint arXiv:1607.02533*, 2016.
* LeCun et al. (1998)

  LeCun, Yann, Cortes, Corinna, and Burges, Christopher JC.
  The mnist database of handwritten digits, 1998.
* Lin & Tegmark (2016)

  Lin, Henry W and Tegmark, Max.
  Why does deep and cheap learning work so well?
  *arXiv preprint arXiv:1608.08225*, 2016.
* Maclaurin et al. (2015)

  Maclaurin, Dougal, Duvenaud, David K, and Adams, Ryan P.
  Gradient-based hyperparameter optimization through reversible
  learning.
  In *ICML*, pp.  2113–2122, 2015.
* Miyato et al. (2015)

  Miyato, Takeru, Maeda, Shin-ichi, Koyama, Masanori, Nakae, Ken, and Ishii,
  Shin.
  Distributional smoothing with virtual adversarial training.
  *stat*, 1050:25, 2015.
* Montavon et al. (2011)

  Montavon, Grégoire, Braun, Mikio L., and Müller, Klaus-Robert.
  Kernel analysis of deep networks.
  *Journal of Machine Learning Research*, 12, 2011.
* Montufar et al. (2014)

  Montufar, Guido F, Pascanu, Razvan, Cho, Kyunghyun, and Bengio, Yoshua.
  On the number of linear regions of deep neural networks.
  In Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N. D., and
  Weinberger, K. Q. (eds.), *Advances in Neural Information Processing
  Systems 27*, pp.  2924–2932. Curran Associates, Inc., 2014.
* Neyshabur et al. (2014)

  Neyshabur, Behnam, Tomioka, Ryota, and Srebro, Nathan.
  In search of the real inductive bias: On the role of implicit
  regularization in deep learning.
  *arXiv preprint arXiv:1412.6614*, 2014.
* Poole et al. (2016)

  Poole, Ben, Lahiri, Subhaneil, Raghu, Maithreyi, Sohl-Dickstein, Jascha, and
  Ganguli, Surya.
  Exponential expressivity in deep neural networks through transient
  chaos.
  In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett,
  R. (eds.), *Advances in Neural Information Processing Systems 29*, pp. 3360–3368. Curran Associates, Inc., 2016.
* Raghu et al. (2016)

  Raghu, Maithra, Poole, Ben, Kleinberg, Jon, Ganguli, Surya, and Sohl-Dickstein,
  Jascha.
  On the expressive power of deep neural networks.
  *arXiv preprint arXiv:1606.05336*, 2016.
* Saxe et al. (2013)

  Saxe, Andrew M, McClelland, James L, and Ganguli, Surya.
  Exact solutions to the nonlinear dynamics of learning in deep linear
  neural networks.
  *arXiv preprint arXiv:1312.6120*, 2013.
* Sjoberg et al. (1995)

  Sjoberg, J., Sjoeberg, J., Sjöberg, J., and Ljung, L.
  Overtraining, regularization and searching for a minimum, with
  application to neural networks.
  *International Journal of Control*, 62:1391–1407,
  1995.
* Sokolic et al. (2016)

  Sokolic, Jure, Giryes, Raja, Sapiro, Guillermo, and Rodrigues, Miguel RD.
  Robust large margin deep neural networks.
  *arXiv preprint arXiv:1605.08254*, 2016.
* Szegedy et al. (2013)

  Szegedy, Christian, Zaremba, Wojciech, Sutskever, Ilya, Bruna, Joan, Erhan,
  Dumitru, Goodfellow, Ian J., and Fergus, Rob.
  Intriguing properties of neural networks.
  *CoRR*, abs/1312.6199, 2013.
  URL <http://arxiv.org/abs/1312.6199>.
* Theano Development Team (2016)

  Theano Development Team, and others.
  Theano: A Python framework for fast computation of mathematical
  expressions.
  *arXiv e-prints*, abs/1605.02688, May 2016.
* Vapnik & Vapnik (1998)

  Vapnik, Vladimir Naumovich and Vapnik, Vlamimir.
  *Statistical learning theory*, volume 1.
  Wiley New York, 1998.
* (37)

  Wang, Shengjie.
  Analysis of deep neural networks with the extended data jacobian
  matrix.
* Wilson & Martinez (2003)

  Wilson, D Randall and Martinez, Tony R.
  The general inefficiency of batch training for gradient descent
  learning.
  *Neural Networks*, 16(10):1429–1451, 2003.
* Yao et al. (2007)

  Yao, Yuan, Rosasco, Lorenzo, and Caponnetto, Andrea.
  On early stopping in gradient descent learning.
  *Constructive Approximation*, 26(2):289–315, 2007.
* Zhang et al. (2017)

  Zhang, Chiyuan, Bengio, Samy, Hardt, Moritz, Recht, Benjamin, and Vinyals,
  Oriol.
  Understanding deep learning requires rethinking generalization.
  *International Conference on Learning Representations (ICLR)*,
  2017.

[◄](/html/1706.05393)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1706.05394)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1706.05394)
[View original  
on arXiv](https://arxiv.org/abs/1706.05394)[►](/html/1706.05395)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 16 09:13:28 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
