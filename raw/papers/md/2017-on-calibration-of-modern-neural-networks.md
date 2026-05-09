---
arxiv: '1706.04599'
authors:
- Chuan Guo
- Geoff Pleiss
- Yu Sun
- Kilian Q. Weinberger
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: On Calibration of Modern Neural Networks
url: http://arxiv.org/abs/1706.04599v2
year: 2017
---

[1706.04599] On Calibration of Modern Neural Networks















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



# On Calibration of Modern Neural Networks

Chuan Guo
  
Geoff Pleiss
  
Yu Sun
  
Kilian Q. Weinberger

# Supplementary Materials for: On Calibration of Modern Neural Networks

Chuan Guo
  
Geoff Pleiss
  
Yu Sun
  
Kilian Q. Weinberger

###### Abstract

Confidence calibration – the problem of predicting probability estimates representative of the true correctness likelihood – is important for classification models in many applications. We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated. Through extensive experiments, we observe that depth, width, weight decay, and Batch Normalization are important factors influencing calibration. We evaluate the performance of various post-processing calibration methods on state-of-the-art architectures with image and document classification datasets. Our analysis and experiments not only
offer insights into neural network learning, but also provide a simple and straightforward recipe for practical settings: on most datasets, *temperature scaling* – a single-parameter variant of Platt Scaling – is surprisingly effective at calibrating predictions.

calibration, confidence, deep learning, neural networks

## 1 Introduction

Recent advances in deep learning have dramatically improved neural network accuracy (Simonyan & Zisserman, [2015](#bib.bib37); Srivastava et al., [2015](#bib.bib40); He et al., [2016](#bib.bib14); Huang et al., [2016](#bib.bib17), [2017](#bib.bib18)). As a result, neural networks are now entrusted with making complex decisions in applications, such as object detection (Girshick, [2015](#bib.bib12)), speech recognition (Hannun et al., [2014](#bib.bib13)), and medical diagnosis (Caruana et al., [2015](#bib.bib4)).
In these settings, neural networks are an essential component of larger decision making pipelines.

In real-world decision making systems, classification networks must not only be accurate, but also should indicate when they are likely to be incorrect.
As an example, consider a self-driving car that uses a neural network to detect pedestrians and other obstructions (Bojarski et al., [2016](#bib.bib3)). If the detection network is not able to confidently predict the presence or absence of immediate obstructions, the car should rely more on the output of other sensors for braking.
Alternatively, in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low (Jiang et al., [2012](#bib.bib22)).
Specifically, a network should provide a *calibrated confidence* measure in addition to its prediction.
In other words, the probability associated with the predicted class label should reflect its ground truth correctness likelihood.

![Refer to caption](/html/1706.04599/assets/x1.png)

![Refer to caption](/html/1706.04599/assets/x2.png)

Figure 1: Confidence histograms (top) and reliability diagrams (bottom) for a 5-layer LeNet (left) and a 110-layer ResNet (right) on CIFAR-100. Refer to the text below for detailed illustration.

Calibrated confidence estimates are also important for model interpretability. Humans have a natural cognitive intuition for probabilities (Cosmides & Tooby, [1996](#bib.bib6)). Good confidence estimates provide a valuable extra bit of information to establish trustworthiness with the user – especially for neural networks, whose classification decisions are often difficult to interpret.
Further, good probability estimates can be used to incorporate neural networks into other probabilistic models. For example, one can improve performance by combining network outputs with a language model in speech recognition (Hannun et al., [2014](#bib.bib13); Xiong et al., [2016](#bib.bib46)), or with camera information for object detection (Kendall & Cipolla, [2016](#bib.bib23)).

In 2005, Niculescu-Mizil & Caruana ([2005](#bib.bib34)) showed that neural networks typically produce well-calibrated probabilities on binary classification tasks. While neural networks today are undoubtedly more accurate than they were a decade ago,
we discover with great surprise that *modern neural networks are no longer well-calibrated*.
This is visualized in [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ On Calibration of Modern Neural Networks"), which compares a 5-layer LeNet (left) (LeCun et al., [1998](#bib.bib30)) with a 110-layer ResNet (right) (He et al., [2016](#bib.bib14)) on the CIFAR-100 dataset.
The top row shows the distribution of prediction confidence (i.e. probabilities associated with the predicted label) as histograms.
The average confidence of LeNet closely matches its accuracy, while the average confidence of the ResNet is substantially higher than its accuracy.
This is further illustrated in the bottom row reliability diagrams (DeGroot & Fienberg, [1983](#bib.bib7); Niculescu-Mizil & Caruana, [2005](#bib.bib34)), which show accuracy as a function of confidence. We see that LeNet is well-calibrated, as confidence closely approximates the expected accuracy (i.e. the bars align roughly along the diagonal). On the other hand, the ResNet’s accuracy is better, but does not match its confidence.

Our goal is not only to understand why neural networks have become miscalibrated, but also to identify what methods can alleviate this problem.
In this paper, we demonstrate on several computer vision and NLP tasks that neural networks produce confidences that do not represent true probabilities.
Additionally, we offer insight and intuition into network training and architectural trends that may cause miscalibration.
Finally, we compare various post-processing calibration methods on state-of-the-art neural networks, and introduce several extensions of our own. Surprisingly, we find that a single-parameter variant of Platt scaling (Platt et al., [1999](#bib.bib36)) – which we refer to as *temperature scaling* – is often the most effective method at obtaining calibrated probabilities. Because this method is straightforward to implement with existing deep learning frameworks, it can be easily adopted in practical settings.

## 2 Definitions

![Refer to caption](/html/1706.04599/assets/x3.png)


Figure 2: The effect of network depth (far left), width (middle left), Batch Normalization (middle right), and weight decay (far right) on miscalibration, as measured by ECE (lower is better).

The problem we address in this paper is supervised multi-class classification with neural networks. The input X∈𝒳𝑋𝒳X\in\mathcal{X} and label Y∈𝒴={1,…,K}𝑌𝒴1…𝐾Y\in\mathcal{Y}=\{1,\ldots,K\} are random variables that follow a ground truth joint distribution π​(X,Y)=π​(Y|X)​π​(X)𝜋𝑋𝑌𝜋conditional𝑌𝑋𝜋𝑋\pi(X,Y)=\pi(Y|X)\pi(X).
Let hℎh be a neural network with h​(X)=(Y^,P^)ℎ𝑋^𝑌^𝑃h(X)=(\hat{Y},\hat{P}), where Y^^𝑌\hat{Y} is a class prediction and P^^𝑃\hat{P} is its associated confidence, i.e. probability of correctness.
We would like the confidence estimate P^^𝑃\hat{P} to be calibrated, which intuitively means that P^^𝑃\hat{P} represents a true probability. For example, given 100 predictions, each with confidence of 0.80.80.8, we expect that 808080 should be correctly classified.
More formally, we define *perfect calibration* as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ(Y^=Y|P^=p)=p,∀p∈[0,1]formulae-sequenceℙ^𝑌conditional𝑌^𝑃𝑝𝑝for-all𝑝01\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)=p,\quad\forall p\in[0,1] |  | (1) |

where the probability is over the joint distribution.
In all practical settings, achieving perfect calibration is impossible.
Additionally, the probability in ([1](#S2.E1 "In 2 Definitions ‣ On Calibration of Modern Neural Networks")) cannot be computed using finitely many samples since P^^𝑃\hat{P} is a continuous random variable. This motivates the need for empirical approximations that capture the essence of ([1](#S2.E1 "In 2 Definitions ‣ On Calibration of Modern Neural Networks")).

#### Reliability Diagrams

(e.g. [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ On Calibration of Modern Neural Networks") bottom)
are a visual representation of model calibration (DeGroot & Fienberg, [1983](#bib.bib7); Niculescu-Mizil & Caruana, [2005](#bib.bib34)).
These diagrams plot expected sample accuracy as a function of confidence.
If the model is perfectly calibrated – i.e. if ([1](#S2.E1 "In 2 Definitions ‣ On Calibration of Modern Neural Networks")) holds – then the diagram should plot the identity function. Any deviation from a perfect diagonal represents miscalibration.

To estimate the expected accuracy from finite samples, we group predictions into M𝑀M interval bins (each of size 1/M1𝑀1/M) and calculate the accuracy of each bin.
Let Bmsubscript𝐵𝑚B\_{m} be the set of indices of samples whose prediction confidence falls into the interval Im=(m−1M,mM]subscript𝐼𝑚𝑚1𝑀𝑚𝑀I\_{m}=(\frac{m-1}{M},\frac{m}{M}].
The accuracy of Bmsubscript𝐵𝑚B\_{m} is

|  |  |  |
| --- | --- | --- |
|  | acc⁡(Bm)=1|Bm|​∑i∈Bm𝟏​(y^i=yi),accsubscript𝐵𝑚1subscript𝐵𝑚subscript𝑖subscript𝐵𝑚1subscript^𝑦𝑖subscript𝑦𝑖\operatorname{acc}(B\_{m})=\frac{1}{|B\_{m}|}\sum\_{i\in B\_{m}}\mathbf{1}(\hat{y}\_{i}=y\_{i}), |  |

where y^isubscript^𝑦𝑖\hat{y}\_{i} and yisubscript𝑦𝑖y\_{i} are the predicted and true class labels for
sample i𝑖i.
Basic probability tells us that acc⁡(Bm)accsubscript𝐵𝑚\operatorname{acc}(B\_{m}) is an unbiased and consistent estimator of ℙ(Y^=Y∣P^∈Im)ℙ^𝑌conditional𝑌^𝑃subscript𝐼𝑚\mathop{\mathbb{P}}(\hat{Y}=Y\mid\hat{P}\in I\_{m}). We define the average confidence within bin Bmsubscript𝐵𝑚B\_{m} as

|  |  |  |
| --- | --- | --- |
|  | conf⁡(Bm)=1|Bm|​∑i∈Bmp^i,confsubscript𝐵𝑚1subscript𝐵𝑚subscript𝑖subscript𝐵𝑚subscript^𝑝𝑖\operatorname{conf}(B\_{m})=\frac{1}{|B\_{m}|}\sum\_{i\in B\_{m}}\hat{p}\_{i}, |  |

where p^isubscript^𝑝𝑖\hat{p}\_{i} is the confidence for sample i𝑖i.
acc⁡(Bm)accsubscript𝐵𝑚\operatorname{acc}(B\_{m}) and conf⁡(Bm)confsubscript𝐵𝑚\operatorname{conf}(B\_{m}) approximate the left-hand and right-hand sides of ([1](#S2.E1 "In 2 Definitions ‣ On Calibration of Modern Neural Networks")) respectively for bin Bmsubscript𝐵𝑚B\_{m}.
Therefore, a perfectly calibrated model will have acc⁡(Bm)=conf⁡(Bm)accsubscript𝐵𝑚confsubscript𝐵𝑚\operatorname{acc}(B\_{m})=\operatorname{conf}(B\_{m}) for all m∈{1,…,M}𝑚1…𝑀m\in\{1,\ldots,M\}.
Note that reliability diagrams do not display the proportion of samples in a given bin, and thus cannot be used to estimate how many samples are calibrated.

#### Expected Calibration Error (ECE).

While reliability diagrams are useful visual tools, it is more convenient to have a scalar summary statistic of calibration.
Since statistics comparing two distributions cannot be comprehensive, previous works have proposed variants, each with a unique emphasis.
One notion of miscalibration is the difference in expectation between confidence and accuracy, i.e.

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝔼P^[|ℙ(Y^=Y|P^=p)−p|]\displaystyle\mathop{\mathbb{E}}\_{\hat{P}}\left[\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|\right] |  | (2) |

Expected Calibration Error (Naeini et al., [2015](#bib.bib32)) – or ECE – approximates ([S1](#S1.Ex9 "S1 Further Information on Calibration Metrics ‣ On Calibration of Modern Neural Networks")) by partitioning predictions into M𝑀M equally-spaced bins (similar to the reliability diagrams) and taking a weighted average of the bins’ accuracy/confidence difference. More precisely,

|  |  |  |  |
| --- | --- | --- | --- |
|  | ECE=∑m=1M|Bm|n​|acc⁡(Bm)−conf⁡(Bm)|,ECEsuperscriptsubscript𝑚1𝑀subscript𝐵𝑚𝑛accsubscript𝐵𝑚confsubscript𝐵𝑚\text{ECE}=\sum\_{m=1}^{M}\frac{|B\_{m}|}{n}\bigg{|}\operatorname{acc}(B\_{m})-\operatorname{conf}(B\_{m})\bigg{|}, |  | (3) |

where n𝑛n is the number of samples.
The difference between accacc\operatorname{acc} and confconf\operatorname{conf} for a given bin represents the calibration *gap* (red bars in reliability diagrams – e.g. [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ On Calibration of Modern Neural Networks")).
We use ECE as the primary empirical metric to measure calibration.
See [Section S1](#S1a "S1 Further Information on Calibration Metrics ‣ On Calibration of Modern Neural Networks") for more analysis of this metric.

#### Maximum Calibration Error (MCE).

In high-risk applications where reliable confidence measures are absolutely necessary, we may wish to minimize the worst-case deviation between confidence and accuracy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxp∈[0,1]|ℙ(Y^=Y|P^=p)−p|.\max\_{p\in[0,1]}\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|. |  | (4) |

The Maximum Calibration Error (Naeini et al., [2015](#bib.bib32)) – or MCE – estimates this deviation. Similarly to ECE, this approximation involves binning:

|  |  |  |  |
| --- | --- | --- | --- |
|  | MCE=maxm∈{1,…,M}⁡|acc⁡(Bm)−conf⁡(Bm)|.MCEsubscript𝑚1…𝑀accsubscript𝐵𝑚confsubscript𝐵𝑚\text{MCE}=\max\_{m\in\{1,\ldots,M\}}\left|\operatorname{acc}(B\_{m})-\operatorname{conf}(B\_{m})\right|. |  | (5) |

We can visualize MCE and ECE on reliability diagrams.
MCE is the largest calibration gap (red bars) across all bins, whereas ECE is a weighted average of all gaps.
For perfectly calibrated classifiers, MCE and ECE both equal 0.

#### Negative log likelihood

is a standard measure of a probabilistic model’s quality (Friedman et al., [2001](#bib.bib10)). It is also referred to as the cross entropy loss in the context of deep learning (Bengio et al., [2015](#bib.bib2)). Given a probabilistic model π^​(Y|X)^𝜋conditional𝑌𝑋\hat{\pi}(Y|X) and n𝑛n samples, NLL is defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ=−∑i=1nlog⁡(π^​(yi|𝐱i))ℒsuperscriptsubscript𝑖1𝑛^𝜋conditionalsubscript𝑦𝑖subscript𝐱𝑖\displaystyle\mathcal{L}=-\sum\_{i=1}^{n}\log(\hat{\pi}(y\_{i}|\mathbf{x}\_{i})) |  | (6) |

It is a standard result (Friedman et al., [2001](#bib.bib10)) that, in expectation, NLL is minimized if and only if π^​(Y|X)^𝜋conditional𝑌𝑋\hat{\pi}(Y|X) recovers the ground truth conditional distribution π​(Y|X)𝜋conditional𝑌𝑋\pi(Y|X).

## 3 Observing Miscalibration

The architecture and training procedures of neural networks have rapidly evolved in recent years. In this section we identify some recent changes that are responsible for the miscalibration phenomenon observed in [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ On Calibration of Modern Neural Networks").
Though we cannot claim causality, we find that increased model capacity and lack of regularization are closely related to model miscalibration.

#### Model capacity.

The model capacity of neural networks has increased at a dramatic pace over the past few years. It is now common to see networks with hundreds, if not thousands of layers (He et al., [2016](#bib.bib14); Huang et al., [2016](#bib.bib17)) and hundreds of convolutional filters per layer (Zagoruyko & Komodakis, [2016](#bib.bib49)). Recent work shows that very deep or wide models are able to generalize better than smaller ones, while exhibiting the capacity to easily fit the training set (Zhang et al., [2017](#bib.bib50)).

Although increasing depth and width may reduce classification error, we observe that these increases negatively affect model calibration.
[Figure 2](#S2.F2 "Figure 2 ‣ 2 Definitions ‣ On Calibration of Modern Neural Networks") displays error and ECE as a function of depth and width on a ResNet trained on CIFAR-100. The far left figure varies depth for a network with 64 convolutional filters per layer, while the middle left figure fixes the depth at 14 layers and varies the number of convolutional filters per layer. Though even the smallest models in the graph exhibit some degree of miscalibration, the ECE metric grows substantially with model capacity.
During training, after the model is able to correctly classify (almost) all training samples, NLL can be further minimized by increasing the confidence of predictions.
Increased model capacity will lower training NLL, and thus the model will be more (over)confident on average.

#### Batch Normalization

(Ioffe & Szegedy, [2015](#bib.bib19)) improves the optimization of neural networks by minimizing distribution shifts in activations within the neural network’s hidden layers. Recent research suggests that these normalization techniques have enabled the development of very deep architectures, such as ResNets (He et al., [2016](#bib.bib14)) and DenseNets (Huang et al., [2017](#bib.bib18)). It has been shown that Batch Normalization improves training time, reduces the need for additional regularization, and can in some cases improve the accuracy of networks.

While it is difficult to pinpoint exactly how Batch Normalization affects the final predictions of a model, we do observe that models trained with Batch Normalization tend to be more miscalibrated. In the middle right plot of [Figure 2](#S2.F2 "Figure 2 ‣ 2 Definitions ‣ On Calibration of Modern Neural Networks"), we see that a 6-layer ConvNet obtains worse calibration when Batch Normalization is applied, even though classification accuracy improves slightly. We find that this result holds regardless of the hyperparameters used on the Batch Normalization model (i.e. low or high learning rate, etc.).

![Refer to caption](/html/1706.04599/assets/x4.png)


Figure 3: Test error and NLL of a 110-layer ResNet with stochastic depth on CIFAR-100 during training. NLL is scaled by a constant to fit in the figure. Learning rate drops by 10x at epochs 250 and 375. The shaded area marks between epochs at which the best validation *loss* and best validation *error* are produced.

#### Weight decay,

which used to be the predominant regularization mechanism for neural networks, is decreasingly utilized when training modern neural networks. Learning theory suggests that regularization is necessary to prevent overfitting, especially as model capacity increases (Vapnik, [1998](#bib.bib42)). However, due to the apparent regularization effects of Batch Normalization, recent research seems to suggest that models with less L2 regularization tend to generalize better (Ioffe & Szegedy, [2015](#bib.bib19)). As a result, it is now common to train models with little weight decay, if any at all. The top performing ImageNet models of 2015 all use an order of magnitude less weight decay than models of previous years (He et al., [2016](#bib.bib14); Simonyan & Zisserman, [2015](#bib.bib37)).

We find that training with less weight decay has a negative impact on calibration.
The far right plot in [Figure 2](#S2.F2 "Figure 2 ‣ 2 Definitions ‣ On Calibration of Modern Neural Networks") displays training error and ECE for a 110-layer ResNet with varying amounts of weight decay.
The only other forms of regularization are data augmentation and Batch Normalization.
We observe that calibration and accuracy are not optimized by the same parameter setting.
While the model exhibits both over-regularization and under-regularization with respect to classification error, it does not appear that calibration is negatively impacted by having too much weight decay.
Model calibration continues to improve when more regularization is added, well after the point of achieving optimal accuracy.
The slight uptick at the end of the graph may be an artifact of using a weight decay factor that impedes optimization.

#### NLL

can be used to indirectly measure model calibration. In practice, we observe *a disconnect between NLL and accuracy*, which may explain the miscalibration in [Figure 2](#S2.F2 "Figure 2 ‣ 2 Definitions ‣ On Calibration of Modern Neural Networks").
This disconnect occurs because neural networks can *overfit to NLL without overfitting to the 0/1 loss*.
We observe this trend in the training curves of some miscalibrated models.
[Figure 3](#S3.F3 "Figure 3 ‣ Batch Normalization ‣ 3 Observing Miscalibration ‣ On Calibration of Modern Neural Networks") shows test error and NLL (rescaled to match error) on CIFAR-100 as training progresses.
Both error and NLL immediately drop at epoch 250, when the learning rate is dropped; however, NLL overfits during the remainder of training.
Surprisingly, overfitting to NLL is beneficial to classification accuracy. On CIFAR-100, test error drops from 29%percent2929\% to 27%percent2727\% in the region where NLL overfits. This phenomenon renders a concrete explanation of miscalibration: the network learns better classification accuracy at the expense of well-modeled probabilities.

We can connect this finding to recent work examining the generalization of large neural networks. Zhang et al. ([2017](#bib.bib50)) observe that deep neural networks seemingly violate the common understanding of learning theory that large models with little regularization will not generalize well. The observed disconnect between NLL and 0/1 loss suggests that these high capacity models are not necessarily immune from overfitting, but rather, overfitting manifests in probabilistic error rather than classification error.

## 4 Calibration Methods

In this section, we first review existing calibration methods, and introduce new variants of our own.
All methods are post-processing steps that produce (calibrated) probabilities.
Each method requires a hold-out validation set, which in practice can be the same set used for hyperparameter tuning.
We assume that the training, validation, and test sets are drawn from the same distribution.

### 4.1 Calibrating Binary Models

We first introduce calibration in the binary setting, i.e. 𝒴={0,1}𝒴01\mathcal{Y}=\{0,1\}.
For simplicity, throughout this subsection, we assume the model outputs only the confidence for the positive class.111
This is in contrast with the setting in [Section 2](#S2 "2 Definitions ‣ On Calibration of Modern Neural Networks"), in which the model produces both a class prediction and confidence.
Given a sample 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}, we have access to p^isubscript^𝑝𝑖\hat{p}\_{i} – the network’s predicted probability of yi=1subscript𝑦𝑖1y\_{i}=1, as well as zi∈ℝsubscript𝑧𝑖ℝz\_{i}\in\mathbb{R} – which is the network’s non-probabilistic output, or *logit*. The predicted probability p^isubscript^𝑝𝑖\hat{p}\_{i} is derived from zisubscript𝑧𝑖z\_{i} using a sigmoid function σ𝜎\sigma; i.e.
p^i=σ​(zi)subscript^𝑝𝑖𝜎subscript𝑧𝑖\hat{p}\_{i}=\sigma(z\_{i}). Our goal is to produce a calibrated probability q^isubscript^𝑞𝑖\hat{q}\_{i} based on yisubscript𝑦𝑖y\_{i}, p^isubscript^𝑝𝑖\hat{p}\_{i}, and zisubscript𝑧𝑖z\_{i}.

#### Histogram binning

(Zadrozny & Elkan, [2001](#bib.bib47)) is a simple non-parametric calibration method.
In a nutshell, all uncalibrated predictions p^isubscript^𝑝𝑖\hat{p}\_{i} are divided
into mutually exclusive bins B1,…,BM

subscript𝐵1…subscript𝐵𝑀B\_{1},\dots,B\_{M}. Each bin is assigned a calibrated score θmsubscript𝜃𝑚\theta\_{m}; i.e. if p^isubscript^𝑝𝑖\hat{p}\_{i} is assigned to bin Bmsubscript𝐵𝑚B\_{m}, then q^i=θmsubscript^𝑞𝑖subscript𝜃𝑚\hat{q}\_{i}=\theta\_{m}. At test time, if prediction p^t​esubscript^𝑝𝑡𝑒\hat{p}\_{te} falls into bin Bmsubscript𝐵𝑚B\_{m}, then the calibrated prediction q^t​esubscript^𝑞𝑡𝑒\hat{q}\_{te} is θmsubscript𝜃𝑚\theta\_{m}.
More precisely, for a suitably chosen M𝑀M (usually small), we first define bin boundaries 0=a1≤a2≤…≤aM+1=10subscript𝑎1subscript𝑎2…subscript𝑎𝑀110=a\_{1}\leq a\_{2}\leq\ldots\leq a\_{M+1}=1, where the bin Bmsubscript𝐵𝑚B\_{m} is defined by the interval (am,am+1]subscript𝑎𝑚subscript𝑎𝑚1(a\_{m},a\_{m+1}].
Typically the bin boundaries are either chosen to be equal length intervals
or to equalize the number of samples in each bin.
The predictions θisubscript𝜃𝑖\theta\_{i} are chosen to minimize the bin-wise squared loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minθ1,…,θM​∑m=1M∑i=1n𝟏​(am≤p^i<am+1)​(θm−yi)2,subscript  subscript𝜃1…subscript𝜃𝑀superscriptsubscript𝑚1𝑀superscriptsubscript𝑖1𝑛1subscript𝑎𝑚subscript^𝑝𝑖subscript𝑎𝑚1superscriptsubscript𝜃𝑚subscript𝑦𝑖2\min\_{\theta\_{1},\ldots,\theta\_{M}}\>\sum\_{m=1}^{M}\sum\_{i=1}^{n}\mathbf{1}(a\_{m}\leq\hat{p}\_{i}<a\_{m+1})\left(\theta\_{m}-y\_{i}\right)^{2}, |  | (7) |

where 𝟏1\mathbf{1} is the indicator function. Given fixed bins boundaries, the solution to ([7](#S4.E7 "In Histogram binning ‣ 4.1 Calibrating Binary Models ‣ 4 Calibration Methods ‣ On Calibration of Modern Neural Networks")) results in θmsubscript𝜃𝑚\theta\_{m} that correspond to the average number of positive-class samples in bin Bmsubscript𝐵𝑚B\_{m}.

#### Isotonic regression

(Zadrozny & Elkan, [2002](#bib.bib48)), arguably the most common non-parametric calibration method,
learns a piecewise constant function f𝑓f to transform uncalibrated outputs; i.e. q^i=f​(p^i)subscript^𝑞𝑖𝑓subscript^𝑝𝑖\hat{q}\_{i}=f(\hat{p}\_{i}).
Specifically, isotonic regression produces f𝑓f to minimize the square loss ∑i=1n(f​(p^i)−yi)2superscriptsubscript𝑖1𝑛superscript𝑓subscript^𝑝𝑖subscript𝑦𝑖2\sum\_{i=1}^{n}(f(\hat{p}\_{i})-y\_{i})^{2}.
Because f𝑓f is constrained to be piecewise constant, we can write the optimization problem as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minMθ1,…,θMa1,…,aM+1subscript  𝑀  subscript𝜃1…subscript𝜃𝑀  subscript𝑎1…subscript𝑎𝑀1\displaystyle\min\_{\begin{subarray}{c}M\\ \theta\_{1},\ldots,\theta\_{M}\\ a\_{1},\ldots,a\_{M+1}\end{subarray}} | ∑m=1M∑i=1n𝟏​(am≤p^i<am+1)​(θm−yi)2superscriptsubscript𝑚1𝑀superscriptsubscript𝑖1𝑛1subscript𝑎𝑚subscript^𝑝𝑖subscript𝑎𝑚1superscriptsubscript𝜃𝑚subscript𝑦𝑖2\displaystyle\hskip 3.0pt\sum\_{m=1}^{M}\sum\_{i=1}^{n}\mathbf{1}(a\_{m}\leq\hat{p}\_{i}<a\_{m+1})\left(\theta\_{m}-y\_{i}\right)^{2} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | subject to | 0=a1≤a2≤…≤aM+1=1,0subscript𝑎1subscript𝑎2…subscript𝑎𝑀11\displaystyle\hskip 8.0pt0=a\_{1}\leq a\_{2}\leq\ldots\leq a\_{M+1}=1, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | θ1≤θ2≤…≤θM.subscript𝜃1subscript𝜃2…subscript𝜃𝑀\displaystyle\hskip 8.0pt\theta\_{1}\leq\theta\_{2}\leq\ldots\leq\theta\_{M}. |  |

where M𝑀M is the number of intervals; a1,…,aM+1

subscript𝑎1…subscript𝑎𝑀1a\_{1},\ldots,a\_{M+1} are the interval boundaries; and θ1,…,θM

subscript𝜃1…subscript𝜃𝑀\theta\_{1},\ldots,\theta\_{M} are the function values.
Under this parameterization, isotonic regression is a strict generalization of histogram binning in which the bin boundaries and bin predictions are jointly optimized.

#### Bayesian Binning into Quantiles (BBQ)

(Naeini et al., [2015](#bib.bib32)) is a extension of histogram binning using Bayesian model averaging. Essentially, BBQ marginalizes out all possible *binning schemes* to produce q^isubscript^𝑞𝑖\hat{q}\_{i}.
More formally, a binning scheme s𝑠s is a pair (M,ℐ)𝑀ℐ(M,\mathcal{I}) where M𝑀M is the number of bins, and ℐℐ\mathcal{I} is a corresponding partitioning of [0,1]01[0,1] into disjoint intervals (0=a1≤a2≤…≤aM+1=10subscript𝑎1subscript𝑎2…subscript𝑎𝑀110=a\_{1}\leq a\_{2}\leq\ldots\leq a\_{M+1}=1). The parameters of a binning scheme are θ1,…,θM

subscript𝜃1…subscript𝜃𝑀\theta\_{1},\ldots,\theta\_{M}. Under this framework, histogram binning and isotonic regression both produce a single binning scheme, whereas BBQ considers a space 𝒮𝒮\mathcal{S} of all possible binning schemes for the validation dataset D𝐷D. BBQ performs Bayesian averaging of the probabilities produced by each scheme:222
Because the validation dataset is finite, 𝒮𝒮\mathcal{S} is as well.

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℙ(q^t​e|p^t​e,D)ℙconditionalsubscript^𝑞𝑡𝑒  subscript^𝑝𝑡𝑒𝐷\displaystyle\mathop{\mathbb{P}}(\hat{q}\_{te}\;|\;\hat{p}\_{te},D) | =∑s∈𝒮ℙ(q^t​e,S=s|p^t​e,D)absentsubscript𝑠𝒮ℙ  subscript^𝑞𝑡𝑒𝑆 conditional𝑠  subscript^𝑝𝑡𝑒𝐷\displaystyle=\sum\_{s\in\mathcal{S}}\mathop{\mathbb{P}}(\hat{q}\_{te},S=s\;|\;\hat{p}\_{te},D) |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | =∑s∈𝒮ℙ(q^t​e|p^t​e,S=s,D)​ℙ(S=s|D).absentsubscript𝑠𝒮ℙconditionalsubscript^𝑞𝑡𝑒  subscript^𝑝𝑡𝑒𝑆  𝑠𝐷ℙ𝑆conditional𝑠𝐷\displaystyle=\sum\_{s\in\mathcal{S}}\mathop{\mathbb{P}}(\hat{q}\_{te}\;|\;\hat{p}\_{te},S\!=\!s,D)\mathop{\mathbb{P}}(S\!=\!s\;|\;D). |  |

where ℙ(q^t​e|p^t​e,S=s,D)ℙconditionalsubscript^𝑞𝑡𝑒

subscript^𝑝𝑡𝑒𝑆

𝑠𝐷\mathop{\mathbb{P}}(\hat{q}\_{te}\;|\;\hat{p}\_{te},S\!=\!s,D) is the calibrated probability using binning scheme s𝑠s. Using a uniform prior, the weight ℙ(S=s|D)ℙ𝑆conditional𝑠𝐷\mathop{\mathbb{P}}(S\!=\!s\;|\;D) can be derived using Bayes’ rule:

|  |  |  |
| --- | --- | --- |
|  | ℙ(S=s|D)=ℙ(D|S=s)∑s′∈𝒮ℙ(D|S=s′).ℙ𝑆conditional𝑠𝐷ℙconditional𝐷𝑆𝑠subscriptsuperscript𝑠′𝒮ℙconditional𝐷𝑆superscript𝑠′\mathop{\mathbb{P}}(S\!=\!s\;|\;D)=\frac{\mathop{\mathbb{P}}(D\;|\;S\!=\!s)}{\sum\_{s^{\prime}\in\mathcal{S}}\mathop{\mathbb{P}}(D\;|\;S\!=\!s^{\prime})}. |  |

The parameters θ1,…,θM

subscript𝜃1…subscript𝜃𝑀\theta\_{1},\ldots,\theta\_{M} can be viewed as parameters of M𝑀M independent binomial distributions. Hence, by placing a Beta prior on θ1,…,θM

subscript𝜃1…subscript𝜃𝑀\theta\_{1},\ldots,\theta\_{M}, we can obtain a closed form expression for the marginal likelihood ℙ(D|S=s)ℙconditional𝐷𝑆𝑠\mathop{\mathbb{P}}(D\;|\;S\!=\!s). This allows us to compute ℙ(q^t​e|p^t​e,D)ℙconditionalsubscript^𝑞𝑡𝑒

subscript^𝑝𝑡𝑒𝐷\mathop{\mathbb{P}}(\hat{q}\_{te}\;|\;\hat{p}\_{te},D) for any test input.

#### Platt scaling

(Platt et al., [1999](#bib.bib36)) is a parametric approach to calibration, unlike the other approaches. The non-probabilistic predictions of a classifier are used as features for a logistic regression model, which is trained on the validation set to return probabilities. More specifically, in the context of neural networks (Niculescu-Mizil & Caruana, [2005](#bib.bib34)), Platt scaling learns scalar parameters a,b∈ℝ

𝑎𝑏
ℝa,b\in\mathbb{R} and outputs q^i=σ​(a​zi+b)subscript^𝑞𝑖𝜎𝑎subscript𝑧𝑖𝑏\hat{q}\_{i}=\sigma(az\_{i}+b) as the calibrated probability. Parameters a𝑎a and b𝑏b can be optimized using the NLL loss over the validation set. It is important to note that the neural network’s parameters are fixed during this stage.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Model | Uncalibrated | Hist. Binning | Isotonic | BBQ | Temp. Scaling | Vector Scaling | Matrix Scaling |
| Birds | ResNet 50 | 9.19% | 4.34% | 5.22% | 4.12% | 1.85% | 3.0% | 21.13% |
| Cars | ResNet 50 | 4.3% | 1.74% | 4.29% | 1.84% | 2.35% | 2.37% | 10.5% |
| CIFAR-10 | ResNet 110 | 4.6% | 0.58% | 0.81% | 0.54% | 0.83% | 0.88% | 1.0% |
| CIFAR-10 | ResNet 110 (SD) | 4.12% | 0.67% | 1.11% | 0.9% | 0.6% | 0.64% | 0.72% |
| CIFAR-10 | Wide ResNet 32 | 4.52% | 0.72% | 1.08% | 0.74% | 0.54% | 0.6% | 0.72% |
| CIFAR-10 | DenseNet 40 | 3.28% | 0.44% | 0.61% | 0.81% | 0.33% | 0.41% | 0.41% |
| CIFAR-10 | LeNet 5 | 3.02% | 1.56% | 1.85% | 1.59% | 0.93% | 1.15% | 1.16% |
| CIFAR-100 | ResNet 110 | 16.53% | 2.66% | 4.99% | 5.46% | 1.26% | 1.32% | 25.49% |
| CIFAR-100 | ResNet 110 (SD) | 12.67% | 2.46% | 4.16% | 3.58% | 0.96% | 0.9% | 20.09% |
| CIFAR-100 | Wide ResNet 32 | 15.0% | 3.01% | 5.85% | 5.77% | 2.32% | 2.57% | 24.44% |
| CIFAR-100 | DenseNet 40 | 10.37% | 2.68% | 4.51% | 3.59% | 1.18% | 1.09% | 21.87% |
| CIFAR-100 | LeNet 5 | 4.85% | 6.48% | 2.35% | 3.77% | 2.02% | 2.09% | 13.24% |
| ImageNet | DenseNet 161 | 6.28% | 4.52% | 5.18% | 3.51% | 1.99% | 2.24% | - |
| ImageNet | ResNet 152 | 5.48% | 4.36% | 4.77% | 3.56% | 1.86% | 2.23% | - |
| SVHN | ResNet 152 (SD) | 0.44% | 0.14% | 0.28% | 0.22% | 0.17% | 0.27% | 0.17% |
| 20 News | DAN 3 | 8.02% | 3.6% | 5.52% | 4.98% | 4.11% | 4.61% | 9.1% |
| Reuters | DAN 3 | 0.85% | 1.75% | 1.15% | 0.97% | 0.91% | 0.66% | 1.58% |
| SST Binary | TreeLSTM | 6.63% | 1.93% | 1.65% | 2.27% | 1.84% | 1.84% | 1.84% |
| SST Fine Grained | TreeLSTM | 6.71% | 2.09% | 1.65% | 2.61% | 2.56% | 2.98% | 2.39% |

Table 1: ECE (%) (with M=15𝑀15M=15 bins) on standard vision and NLP datasets before calibration and with various calibration methods. The number following a model’s name denotes the network depth.

### 4.2 Extension to Multiclass Models

For classification problems involving K>2𝐾2K>2 classes, we return to the original problem formulation.
The network outputs a class prediction y^isubscript^𝑦𝑖\hat{y}\_{i} and confidence score p^isubscript^𝑝𝑖\hat{p}\_{i} for each input 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}. In this case, the network logits 𝐳isubscript𝐳𝑖\mathbf{z}\_{i} are vectors, where y^i=argmaxkzi(k)subscript^𝑦𝑖subscriptargmax𝑘superscriptsubscript𝑧𝑖𝑘\hat{y}\_{i}=\operatorname\*{argmax}\_{k}z\_{i}^{(k)}, and p^isubscript^𝑝𝑖\hat{p}\_{i} is typically derived using the softmax function σSMsubscript𝜎SM\sigma\_{\text{SM}}:

|  |  |  |
| --- | --- | --- |
|  | σSM​(𝐳i)(k)=exp⁡(zi(k))∑j=1Kexp⁡(zi(j)),p^i=maxk⁡σSM​(𝐳i)(k).formulae-sequencesubscript𝜎SMsuperscriptsubscript𝐳𝑖𝑘superscriptsubscript𝑧𝑖𝑘superscriptsubscript𝑗1𝐾superscriptsubscript𝑧𝑖𝑗subscript^𝑝𝑖subscript𝑘subscript𝜎SMsuperscriptsubscript𝐳𝑖𝑘\sigma\_{\text{SM}}(\mathbf{z}\_{i})^{(k)}=\frac{\exp(z\_{i}^{(k)})}{\sum\_{j=1}^{K}\exp(z\_{i}^{(j)})},\hskip 10.0pt\hat{p}\_{i}=\max\_{k}\>\sigma\_{\text{SM}}(\mathbf{z}\_{i})^{(k)}. |  |

The goal is to produce a calibrated confidence q^isubscript^𝑞𝑖\hat{q}\_{i} and (possibly new) class prediction y^i′superscriptsubscript^𝑦𝑖′\hat{y}\_{i}^{\prime}
based on yisubscript𝑦𝑖y\_{i}, y^isubscript^𝑦𝑖\hat{y}\_{i}, p^isubscript^𝑝𝑖\hat{p}\_{i}, and 𝐳isubscript𝐳𝑖\mathbf{z}\_{i}.

#### Extension of binning methods.

One common way of extending binary calibration methods to the multiclass setting is by treating the problem as K𝐾K one-versus-all problems (Zadrozny & Elkan, [2002](#bib.bib48)).
For k=1,…,K𝑘

1…𝐾k=1,\ldots,K, we form a binary calibration problem where the label is 𝟏​(yi=k)1subscript𝑦𝑖𝑘\mathbf{1}(y\_{i}=k) and the predicted probability is σSM​(𝐳i)(k)subscript𝜎SMsuperscriptsubscript𝐳𝑖𝑘\sigma\_{\text{SM}}(\mathbf{z}\_{i})^{(k)}. This gives us K𝐾K calibration models, each for a particular class. At test time, we obtain an unnormalized probability vector [q^i(1),…,q^i(K)]

superscriptsubscript^𝑞𝑖1…superscriptsubscript^𝑞𝑖𝐾[\hat{q}\_{i}^{(1)},\ldots,\hat{q}\_{i}^{(K)}], where q^i(k)superscriptsubscript^𝑞𝑖𝑘\hat{q}\_{i}^{(k)} is the calibrated probability for class k𝑘k. The new class prediction y^i′superscriptsubscript^𝑦𝑖′\hat{y}\_{i}^{\prime} is the argmax of the vector, and the new confidence q^i′superscriptsubscript^𝑞𝑖′\hat{q}\_{i}^{\prime} is the max of the vector normalized by ∑k=1Kq^i(k)superscriptsubscript𝑘1𝐾superscriptsubscript^𝑞𝑖𝑘\sum\_{k=1}^{K}\hat{q}\_{i}^{(k)}. This extension can be applied to histogram binning, isotonic regression, and BBQ.

#### Matrix and vector scaling

are two multi-class extensions of Platt scaling. Let 𝐳isubscript𝐳𝑖\mathbf{z}\_{i} be the *logits vector* produced before the softmax layer for input 𝐱isubscript𝐱𝑖\mathbf{x}\_{i}. *Matrix scaling applies* a linear transformation 𝐖𝐳i+𝐛subscript𝐖𝐳𝑖𝐛\mathbf{W}\mathbf{z}\_{i}+\mathbf{b} to the logits:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | q^isubscript^𝑞𝑖\displaystyle\hat{q}\_{i} | =maxk⁡σSM​(𝐖𝐳i+𝐛)(k),absentsubscript𝑘subscript𝜎SMsuperscriptsubscript𝐖𝐳𝑖𝐛𝑘\displaystyle=\max\_{k}\>\sigma\_{\text{SM}}(\mathbf{W}\mathbf{z}\_{i}+\mathbf{b})^{(k)}, |  | (8) |
|  | y^i′superscriptsubscript^𝑦𝑖′\displaystyle\hat{y}\_{i}^{\prime} | =argmaxk(𝐖𝐳i+𝐛)(k).\displaystyle=\operatorname\*{argmax}\_{k}\>(\mathbf{W}\mathbf{z}\_{i}+\mathbf{b})^{(k)}. |  |

The parameters 𝐖𝐖\mathbf{W} and 𝐛𝐛\mathbf{b} are optimized with respect to NLL on the validation set. As the number of parameters for matrix scaling grows quadratically with the number of classes K𝐾K, we define *vector scaling* as a variant where 𝐖𝐖\mathbf{W} is restricted to be a diagonal matrix.

#### Temperature scaling,

the simplest extension of Platt scaling, uses a single scalar parameter T>0𝑇0T>0 for all classes. Given the logit vector 𝐳isubscript𝐳𝑖\mathbf{z}\_{i}, the new confidence prediction is

|  |  |  |  |
| --- | --- | --- | --- |
|  | q^i=maxk⁡σSM​(𝐳i/T)(k).subscript^𝑞𝑖subscript𝑘subscript𝜎SMsuperscriptsubscript𝐳𝑖𝑇𝑘\hat{q}\_{i}=\max\_{k}\>\sigma\_{\text{SM}}(\mathbf{z}\_{i}/T)^{(k)}. |  | (9) |

T𝑇T is called the temperature, and it “softens” the softmax (i.e. raises the output entropy) with T>1𝑇1T>1.
As T→∞→𝑇T\rightarrow\infty, the probability q^isubscript^𝑞𝑖\hat{q}\_{i} approaches 1/K1𝐾1/K, which represents maximum uncertainty. With T=1𝑇1T=1, we recover the original probability p^isubscript^𝑝𝑖\hat{p}\_{i}.
As T→0→𝑇0T\rightarrow 0, the probability collapses to a point mass (i.e. q^i=1subscript^𝑞𝑖1\hat{q}\_{i}=1).
T𝑇T is optimized with respect to NLL on the validation set.
Because the parameter T𝑇T does not change the maximum of the softmax function, the class prediction y^i′superscriptsubscript^𝑦𝑖′\hat{y}\_{i}^{\prime} remains unchanged. In other words, *temperature scaling does not affect the model’s accuracy.*

Temperature scaling is commonly used in settings such as knowledge distillation (Hinton et al., [2015](#bib.bib16)) and statistical mechanics (Jaynes, [1957](#bib.bib21)). To the best of our knowledge, we are not aware of any prior use in the context of calibrating probabilistic models.333To highlight the connection with prior works we define temperature scaling in terms of 1T1𝑇\frac{1}{T} instead of a multiplicative scalar.
The model is equivalent to maximizing the entropy of the output probability distribution subject to certain constraints on the logits (see [Section S2](#S2a "S2 Further Information on Temperature Scaling ‣ On Calibration of Modern Neural Networks")).

### 4.3 Other Related Works

Calibration and confidence scores have been studied in various contexts in recent years. Kuleshov & Ermon ([2016](#bib.bib27)) study the problem of calibration in the online setting, where the inputs can come from a potentially adversarial source. Kuleshov & Liang ([2015](#bib.bib28)) investigate how to produce calibrated probabilities when the output space is a structured object.
Lakshminarayanan et al. ([2016](#bib.bib29)) use ensembles of networks to obtain uncertainty estimates.
Pereyra et al. ([2017](#bib.bib35)) penalize overconfident predictions as a form of regularization.
Hendrycks & Gimpel ([2017](#bib.bib15)) use confidence scores to determine if samples are out-of-distribution.

Bayesian neural networks (Denker & Lecun, [1990](#bib.bib9); MacKay, [1992](#bib.bib31)) return a probability distribution over outputs as an alternative way to represent model uncertainty.
Gal & Ghahramani ([2016](#bib.bib11)) draw a connection between Dropout (Srivastava et al., [2014](#bib.bib39)) and model uncertainty, claiming that sampling models with dropped nodes is a way to estimate the probability distribution over all possible models for a given sample.
Kendall & Gal ([2017](#bib.bib24)) combine this approach with a model that outputs a predictive mean and variance for each data point.
This notion of uncertainty is not restricted to classification problems.
Additionally, neural networks can be used in conjunction with Bayesian models that output complete distributions.
For example, deep kernel learning (Wilson et al., [2016a](#bib.bib44), [b](#bib.bib45); Al-Shedivat et al., [2016](#bib.bib1)) combines deep neural networks with Gaussian processes on classification and regression problems.
In contrast, our framework, which does not augment the neural network model, returns a confidence score rather than returning a distribution of possible outputs.

## 5 Results

We apply the calibration methods in [Section 4](#S4 "4 Calibration Methods ‣ On Calibration of Modern Neural Networks") to image classification and document classification neural networks.
For image classification we use 6 datasets:

1. 1.

   Caltech-UCSD Birds (Welinder et al., [2010](#bib.bib43)): 200 bird species. 5994/2897/2897 images for train/validation/test sets.
2. 2.

   Stanford Cars (Krause et al., [2013](#bib.bib25)): 196 classes of cars by make, model, and year. 8041/4020/4020 images for train/validation/test.
3. 3.

   ImageNet 2012 (Deng et al., [2009](#bib.bib8)): Natural scene images from 1000 classes. 1.3 million/25,000/25,000 images for train/validation/test.
4. 4.

   CIFAR-10/CIFAR-100 (Krizhevsky & Hinton, [2009](#bib.bib26)): Color images (32×32323232\times 32) from 10/100 classes. 45,000/5,000/10,000 images for train/validation/test.
5. 5.

   Street View House Numbers (SVHN) (Netzer et al., [2011](#bib.bib33)): 32×32323232\times 32 colored images of cropped out house numbers from Google Street View.
   598,388/6,000/26,032 images for train/validation/test.

We train state-of-the-art convolutional networks: ResNets (He et al., [2016](#bib.bib14)), ResNets with stochastic depth (SD) (Huang et al., [2016](#bib.bib17)), Wide ResNets (Zagoruyko & Komodakis, [2016](#bib.bib49)), and DenseNets (Huang et al., [2017](#bib.bib18)). We use the data preprocessing, training procedures, and hyperparameters as described in each paper. For Birds and Cars, we fine-tune networks pretrained on ImageNet.

For document classification we experiment with 4 datasets:

1. 1.

   20 News: News articles, partitioned into 20 categories by content. 9034/2259/7528 documents for train/validation/test.
2. 2.

   Reuters: News articles, partitioned into 8 categories by topic. 4388/1097/2189 documents for train/validation/test.
3. 3.

   Stanford Sentiment Treebank (SST) (Socher et al., [2013](#bib.bib38)): Movie reviews, represented as sentence parse trees that are annotated by sentiment. Each sample includes a coarse binary label and a fine grained 5-class label.
   As described in (Tai et al., [2015](#bib.bib41)), the training/validation/test sets contain 6920/872/1821 documents for binary, and 544/1101/2210 for fine-grained.

On 20 News and Reuters, we train Deep Averaging Networks (DANs) (Iyyer et al., [2015](#bib.bib20)) with 3 feed-forward layers and Batch Normalization. On SST, we train TreeLSTMs (Long Short Term Memory) (Tai et al., [2015](#bib.bib41)).
For both models we use the default hyperparmaeters suggested by the authors.

![Refer to caption](/html/1706.04599/assets/x5.png)


Figure 4: Reliability diagrams for CIFAR-100 before (far left) and after calibration (middle left, middle right, far right).

#### Calibration Results.

[Table 1](#S4.T1 "Table 1 ‣ Platt scaling ‣ 4.1 Calibrating Binary Models ‣ 4 Calibration Methods ‣ On Calibration of Modern Neural Networks") displays model calibration, as measured by ECE (with M=15𝑀15M=15 bins), before and after applying the various methods (see [Section S3](#S3a "S3 Additional Tables ‣ On Calibration of Modern Neural Networks") for MCE, NLL, and error tables).
It is worth noting that most datasets and models experience some degree of miscalibration, with ECE typically between 444 to 10%percent1010\%.
This is not architecture specific: we observe miscalibration on convolutional networks (with and without skip connections), recurrent networks, and deep averaging networks.
The two notable exceptions are SVHN and Reuters, both of which experience ECE values below 1%percent11\%. Both of these datasets have very low error (1.98%percent1.981.98\% and 2.97%percent2.972.97\%, respectively); and therefore the ratio of ECE to error is comparable to other datasets.

Our most important discovery is the *surprising effectiveness of temperature scaling* despite its remarkable simplicity. Temperature scaling outperforms all other methods on the vision tasks, and performs comparably to other methods on the NLP datasets. What is perhaps even more surprising is that temperature scaling outperforms the vector and matrix Platt scaling variants, which are strictly more general methods. In fact, vector scaling recovers essentially the same solution as temperature scaling – the learned vector has nearly constant entries, and therefore is no different than a scalar transformation. In other words, network miscalibration is intrinsically low dimensional.

The only dataset that temperature scaling does not calibrate is the Reuters dataset. In this instance, only one of the above methods is able to improve calibration. Because this dataset is well-calibrated to begin with (ECE ≤1%absentpercent1\leq 1\%), there is not much room for improvement with any method, and post-processing may not even be necessary to begin with. It is also possible that our measurements are affected by dataset split or by the particular binning scheme.

Matrix scaling performs poorly on datasets with hundreds of classes (i.e. Birds, Cars, and CIFAR-100), and fails to converge on the 1000-class ImageNet dataset.
This is expected, since the number of parameters scales quadratically with the number of classes.
Any calibration model with tens of thousands (or more) parameters will overfit to a small validation set, even when applying regularization.

Binning methods improve calibration on most datasets, but do not outperform temperature scaling. Additionally, binning methods tend to change class predictions which hurts accuracy (see [Section S3](#S3a "S3 Additional Tables ‣ On Calibration of Modern Neural Networks")). Histogram binning, the simplest binning method, typically outperforms isotonic regression and BBQ, despite the fact that both methods are strictly more general. This further supports our finding that calibration is best corrected by simple models.

#### Reliability diagrams.

[Figure 4](#S5.F4 "Figure 4 ‣ 5 Results ‣ On Calibration of Modern Neural Networks") contains reliability diagrams for 110-layer ResNets on CIFAR-100 before and after calibration. From the far left diagram, we see that the uncalibrated ResNet tends to be overconfident in its predictions. We then can observe the effects of temperature scaling (middle left), histogram binning (middle right), and isotonic regression (far right) on calibration. All three displayed methods produce much better confidence estimates. Of the three methods, temperature scaling most closely recovers the desired diagonal function. Each of the bins are well calibrated, which is remarkable given that all the probabilities were modified by only a single parameter. We include reliability diagrams for other datasets in [Section S4](#S4a "S4 Additional Reliability Diagrams ‣ On Calibration of Modern Neural Networks").

#### Computation time.

All methods scale linearly with the number of validation set samples. Temperature scaling is by far the fastest method, as it amounts to a one-dimensional convex optimization problem. Using a conjugate gradient solver, the optimal temperature can be found in 10 iterations, or a fraction of a second on most modern hardware. In fact, even a naive line-search for the optimal temperature is faster than any of the other methods. The computational complexity of vector and matrix scaling are linear and quadratic respectively in the number of classes, reflecting the number of parameters in each method. For CIFAR-100 (K=100𝐾100K=100), finding a near-optimal vector scaling solution with conjugate gradient descent requires at least 2 orders of magnitude more time. Histogram binning and isotonic regression take an order of magnitude longer than temperature scaling, and BBQ takes roughly 3 orders of magnitude more time.

#### Ease of implementation.

BBQ is arguably the most difficult to implement, as it requires implementing a model averaging scheme.
While all other methods are relatively easy to implement, temperature scaling may arguably be the most straightforward to incorporate into a neural network pipeline.
In Torch7 (Collobert et al., [2011](#bib.bib5)), for example, we implement temperature scaling by inserting a nn.MulConstant between the logits and the softmax, whose parameter is 1/T1𝑇1/T.
We set T=1𝑇1T\!=\!1 during training, and subsequently find its optimal value on the validation set.444
For an example implementation, see <http://github.com/gpleiss/temperature_scaling>.

## 6 Conclusion

Modern neural networks exhibit a strange phenomenon: probabilistic error and miscalibration worsen even as classification error is reduced.
We have demonstrated that recent advances in neural network architecture and training – model capacity, normalization, and regularization – have strong effects on network calibration.
It remains future work to understand why these trends affect calibration while improving accuracy.
Nevertheless, simple techniques can effectively remedy the miscalibration phenomenon in neural networks.
Temperature scaling is the simplest, fastest, and most straightforward of the methods, and surprisingly is often the most effective.

## Acknowledgments

The authors are supported in part by the III-1618134, III-1526012, and IIS-1149882 grants from the
National Science Foundation, as well as the Bill and
Melinda Gates Foundation and the Office of Naval Research.

## References

* Al-Shedivat et al. (2016)

  Al-Shedivat, Maruan, Wilson, Andrew Gordon, Saatchi, Yunus, Hu, Zhiting, and
  Xing, Eric P.
  Learning scalable deep kernels with recurrent structure.
  *arXiv preprint arXiv:1610.08936*, 2016.
* Bengio et al. (2015)

  Bengio, Yoshua, Goodfellow, Ian J, and Courville, Aaron.
  Deep learning.
  *Nature*, 521:436–444, 2015.
* Bojarski et al. (2016)

  Bojarski, Mariusz, Del Testa, Davide, Dworakowski, Daniel, Firner, Bernhard,
  Flepp, Beat, Goyal, Prasoon, Jackel, Lawrence D, Monfort, Mathew, Muller,
  Urs, Zhang, Jiakai, et al.
  End to end learning for self-driving cars.
  *arXiv preprint arXiv:1604.07316*, 2016.
* Caruana et al. (2015)

  Caruana, Rich, Lou, Yin, Gehrke, Johannes, Koch, Paul, Sturm, Marc, and
  Elhadad, Noemie.
  Intelligible models for healthcare: Predicting pneumonia risk and
  hospital 30-day readmission.
  In *KDD*, 2015.
* Collobert et al. (2011)

  Collobert, Ronan, Kavukcuoglu, Koray, and Farabet, Clément.
  Torch7: A matlab-like environment for machine learning.
  In *BigLearn Workshop, NIPS*, 2011.
* Cosmides & Tooby (1996)

  Cosmides, Leda and Tooby, John.
  Are humans good intuitive statisticians after all? rethinking some
  conclusions from the literature on judgment under uncertainty.
  *cognition*, 58(1):1–73, 1996.
* DeGroot & Fienberg (1983)

  DeGroot, Morris H and Fienberg, Stephen E.
  The comparison and evaluation of forecasters.
  *The statistician*, pp.  12–22, 1983.
* Deng et al. (2009)

  Deng, Jia, Dong, Wei, Socher, Richard, Li, Li-Jia, Li, Kai, and Fei-Fei, Li.
  Imagenet: A large-scale hierarchical image database.
  In *CVPR*, pp.  248–255, 2009.
* Denker & Lecun (1990)

  Denker, John S and Lecun, Yann.
  Transforming neural-net output levels to probability distributions.
  In *NIPS*, pp.  853–859, 1990.
* Friedman et al. (2001)

  Friedman, Jerome, Hastie, Trevor, and Tibshirani, Robert.
  *The elements of statistical learning*, volume 1.
  Springer series in statistics Springer, Berlin, 2001.
* Gal & Ghahramani (2016)

  Gal, Yarin and Ghahramani, Zoubin.
  Dropout as a bayesian approximation: Representing model uncertainty
  in deep learning.
  In *ICML*, 2016.
* Girshick (2015)

  Girshick, Ross.
  Fast r-cnn.
  In *ICCV*, pp.  1440–1448, 2015.
* Hannun et al. (2014)

  Hannun, Awni, Case, Carl, Casper, Jared, Catanzaro, Bryan, Diamos, Greg, Elsen,
  Erich, Prenger, Ryan, Satheesh, Sanjeev, Sengupta, Shubho, Coates, Adam,
  et al.
  Deep speech: Scaling up end-to-end speech recognition.
  *arXiv preprint arXiv:1412.5567*, 2014.
* He et al. (2016)

  He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian.
  Deep residual learning for image recognition.
  In *CVPR*, pp.  770–778, 2016.
* Hendrycks & Gimpel (2017)

  Hendrycks, Dan and Gimpel, Kevin.
  A baseline for detecting misclassified and out-of-distribution
  examples in neural networks.
  In *ICLR*, 2017.
* Hinton et al. (2015)

  Hinton, Geoffrey, Vinyals, Oriol, and Dean, Jeff.
  Distilling the knowledge in a neural network.
  2015.
* Huang et al. (2016)

  Huang, Gao, Sun, Yu, Liu, Zhuang, Sedra, Daniel, and Weinberger, Kilian.
  Deep networks with stochastic depth.
  In *ECCV*, 2016.
* Huang et al. (2017)

  Huang, Gao, Liu, Zhuang, Weinberger, Kilian Q, and van der Maaten, Laurens.
  Densely connected convolutional networks.
  In *CVPR*, 2017.
* Ioffe & Szegedy (2015)

  Ioffe, Sergey and Szegedy, Christian.
  Batch normalization: Accelerating deep network training by reducing
  internal covariate shift.
  2015.
* Iyyer et al. (2015)

  Iyyer, Mohit, Manjunatha, Varun, Boyd-Graber, Jordan, and Daumé III, Hal.
  Deep unordered composition rivals syntactic methods for text
  classification.
  In *ACL*, 2015.
* Jaynes (1957)

  Jaynes, Edwin T.
  Information theory and statistical mechanics.
  *Physical review*, 106(4):620, 1957.
* Jiang et al. (2012)

  Jiang, Xiaoqian, Osl, Melanie, Kim, Jihoon, and Ohno-Machado, Lucila.
  Calibrating predictive model estimates to support personalized
  medicine.
  *Journal of the American Medical Informatics Association*,
  19(2):263–274, 2012.
* Kendall & Cipolla (2016)

  Kendall, Alex and Cipolla, Roberto.
  Modelling uncertainty in deep learning for camera relocalization.
  2016.
* Kendall & Gal (2017)

  Kendall, Alex and Gal, Yarin.
  What uncertainties do we need in bayesian deep learning for computer
  vision?
  *arXiv preprint arXiv:1703.04977*, 2017.
* Krause et al. (2013)

  Krause, Jonathan, Stark, Michael, Deng, Jia, and Fei-Fei, Li.
  3d object representations for fine-grained categorization.
  In *IEEE Workshop on 3D Representation and Recognition (3dRR)*,
  Sydney, Australia, 2013.
* Krizhevsky & Hinton (2009)

  Krizhevsky, Alex and Hinton, Geoffrey.
  Learning multiple layers of features from tiny images, 2009.
* Kuleshov & Ermon (2016)

  Kuleshov, Volodymyr and Ermon, Stefano.
  Reliable confidence estimation via online learning.
  *arXiv preprint arXiv:1607.03594*, 2016.
* Kuleshov & Liang (2015)

  Kuleshov, Volodymyr and Liang, Percy.
  Calibrated structured prediction.
  In *NIPS*, pp.  3474–3482, 2015.
* Lakshminarayanan et al. (2016)

  Lakshminarayanan, Balaji, Pritzel, Alexander, and Blundell, Charles.
  Simple and scalable predictive uncertainty estimation using deep
  ensembles.
  *arXiv preprint arXiv:1612.01474*, 2016.
* LeCun et al. (1998)

  LeCun, Yann, Bottou, Léon, Bengio, Yoshua, and Haffner, Patrick.
  Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
* MacKay (1992)

  MacKay, David JC.
  A practical bayesian framework for backpropagation networks.
  *Neural computation*, 4(3):448–472, 1992.
* Naeini et al. (2015)

  Naeini, Mahdi Pakdaman, Cooper, Gregory F, and Hauskrecht, Milos.
  Obtaining well calibrated probabilities using bayesian binning.
  In *AAAI*, pp.  2901, 2015.
* Netzer et al. (2011)

  Netzer, Yuval, Wang, Tao, Coates, Adam, Bissacco, Alessandro, Wu, Bo, and Ng,
  Andrew Y.
  Reading digits in natural images with unsupervised feature learning.
  In *Deep Learning and Unsupervised Feature Learning Workshop,
  NIPS*, 2011.
* Niculescu-Mizil & Caruana (2005)

  Niculescu-Mizil, Alexandru and Caruana, Rich.
  Predicting good probabilities with supervised learning.
  In *ICML*, pp.  625–632, 2005.
* Pereyra et al. (2017)

  Pereyra, Gabriel, Tucker, George, Chorowski, Jan, Kaiser, Łukasz, and
  Hinton, Geoffrey.
  Regularizing neural networks by penalizing confident output
  distributions.
  *arXiv preprint arXiv:1701.06548*, 2017.
* Platt et al. (1999)

  Platt, John et al.
  Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods.
  *Advances in large margin classifiers*, 10(3):61–74, 1999.
* Simonyan & Zisserman (2015)

  Simonyan, Karen and Zisserman, Andrew.
  Very deep convolutional networks for large-scale image recognition.
  In *ICLR*, 2015.
* Socher et al. (2013)

  Socher, Richard, Perelygin, Alex, Wu, Jean, Chuang, Jason, Manning,
  Christopher D., Ng, Andrew, and Potts, Christopher.
  Recursive deep models for semantic compositionality over a sentiment
  treebank.
  In *EMNLP*, pp.  1631–1642, 2013.
* Srivastava et al. (2014)

  Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex, Sutskever, Ilya, and
  Salakhutdinov, Ruslan.
  Dropout: A simple way to prevent neural networks from overfitting.
  *Journal of Machine Learning Research*, 15:1929–1958,
  2014.
* Srivastava et al. (2015)

  Srivastava, Rupesh Kumar, Greff, Klaus, and Schmidhuber, Jürgen.
  Highway networks.
  *arXiv preprint arXiv:1505.00387*, 2015.
* Tai et al. (2015)

  Tai, Kai Sheng, Socher, Richard, and Manning, Christopher D.
  Improved semantic representations from tree-structured long
  short-term memory networks.
  2015.
* Vapnik (1998)

  Vapnik, Vladimir N.
  *Statistical Learning Theory*.
  Wiley-Interscience, 1998.
* Welinder et al. (2010)

  Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., and
  Perona, P.
  Caltech-UCSD Birds 200.
  Technical Report CNS-TR-2010-001, California Institute of Technology,
  2010.
* Wilson et al. (2016a)

  Wilson, Andrew G, Hu, Zhiting, Salakhutdinov, Ruslan R, and Xing, Eric P.
  Stochastic variational deep kernel learning.
  In *NIPS*, pp.  2586–2594, 2016a.
* Wilson et al. (2016b)

  Wilson, Andrew Gordon, Hu, Zhiting, Salakhutdinov, Ruslan, and Xing, Eric P.
  Deep kernel learning.
  In *AISTATS*, pp.  370–378, 2016b.
* Xiong et al. (2016)

  Xiong, Wayne, Droppo, Jasha, Huang, Xuedong, Seide, Frank, Seltzer, Mike,
  Stolcke, Andreas, Yu, Dong, and Zweig, Geoffrey.
  Achieving human parity in conversational speech recognition.
  *arXiv preprint arXiv:1610.05256*, 2016.
* Zadrozny & Elkan (2001)

  Zadrozny, Bianca and Elkan, Charles.
  Obtaining calibrated probability estimates from decision trees and
  naive bayesian classifiers.
  In *ICML*, pp.  609–616, 2001.
* Zadrozny & Elkan (2002)

  Zadrozny, Bianca and Elkan, Charles.
  Transforming classifier scores into accurate multiclass probability
  estimates.
  In *KDD*, pp.  694–699, 2002.
* Zagoruyko & Komodakis (2016)

  Zagoruyko, Sergey and Komodakis, Nikos.
  Wide residual networks.
  In *BMVC*, 2016.
* Zhang et al. (2017)

  Zhang, Chiyuan, Bengio, Samy, Hardt, Moritz, Recht, Benjamin, and Vinyals,
  Oriol.
  Understanding deep learning requires rethinking generalization.
  In *ICLR*, 2017.

## S1 Further Information on Calibration Metrics

We can connect the ECE metric with our exact miscalibration definition, which is restated here:

|  |  |  |
| --- | --- | --- |
|  | 𝔼P^[|ℙ(Y^=Y|P^=p)−p|]\displaystyle\mathop{\mathbb{E}}\_{\hat{P}}\left[\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|\right] |  |

Let FP^​(p)subscript𝐹^𝑃𝑝F\_{\hat{P}}(p) be the cumulative distribution function of P^^𝑃\hat{P} so that FP^​(b)−FP^​(a)=ℙ(P^∈[a,b])subscript𝐹^𝑃𝑏subscript𝐹^𝑃𝑎ℙ^𝑃𝑎𝑏F\_{\hat{P}}(b)-F\_{\hat{P}}(a)=\mathop{\mathbb{P}}(\hat{P}\in[a,b]). Using the Riemann-Stieltjes integral we have

|  |  |  |
| --- | --- | --- |
|  | 𝔼P^[|ℙ(Y^=Y|P^=p)−p|]\displaystyle\mathop{\mathbb{E}}\_{\hat{P}}\left[\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|\right] |  |
|  |  |  |
| --- | --- | --- |
|  | =∫01|ℙ(Y^=Y|P^=p)−p|dFP^(p)\displaystyle=\int\_{0}^{1}\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|dF\_{\hat{P}}(p) |  |
|  |  |  |
| --- | --- | --- |
|  | ≈∑m=1M|ℙ(Y^=Y|P^=pm)−pm|ℙ(P^∈Im)\displaystyle\approx\sum\_{m=1}^{M}\left|\mathop{\mathbb{P}}(\hat{Y}=Y|\hat{P}=p\_{m})-p\_{m}\right|\mathop{\mathbb{P}}(\hat{P}\in I\_{m}) |  |

where Imsubscript𝐼𝑚I\_{m} represents the interval of bin Bmsubscript𝐵𝑚B\_{m}. |ℙ(Y^=Y|P^=pm)−pm|\left|\mathop{\mathbb{P}}(\hat{Y}=Y|\hat{P}=p\_{m})-p\_{m}\right| is closely approximated by |acc​(Bm)−p^​(Bm)|accsubscript𝐵𝑚^𝑝subscript𝐵𝑚\left|\text{acc}(B\_{m})-\hat{p}(B\_{m})\right| for n𝑛n large. Hence ECE using M𝑀M bins converges to the M𝑀M-term Riemann-Stieltjes sum of 𝔼P^[|ℙ(Y^=Y|P^=p)−p|]\mathop{\mathbb{E}}\_{\hat{P}}\left[\left|\mathop{\mathbb{P}}\left(\hat{Y}=Y\;|\;\hat{P}=p\right)-p\right|\right].

## S2 Further Information on Temperature Scaling

Here we derive the temperature scaling model using the entropy maximization principle with an appropriate balanced equation.

###### Claim 1.

Given n𝑛n samples’ logit vectors 𝐳1,…,𝐳n

subscript𝐳1…subscript𝐳𝑛\mathbf{z}\_{1},\ldots,\mathbf{z}\_{n} and class labels y1,…,yn

subscript𝑦1…subscript𝑦𝑛y\_{1},\ldots,y\_{n}, temperature scaling is the unique solution q𝑞q to the following entropy maximization problem:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxqsubscript𝑞\displaystyle\max\_{q}\hskip 8.0pt | −∑i=1n∑k=1Kq​(𝐳i)(k)​log⁡q​(𝐳i)(k)superscriptsubscript𝑖1𝑛superscriptsubscript𝑘1𝐾𝑞superscriptsubscript𝐳𝑖𝑘𝑞superscriptsubscript𝐳𝑖𝑘\displaystyle-\sum\_{i=1}^{n}\sum\_{k=1}^{K}q(\mathbf{z}\_{i})^{(k)}\log q(\mathbf{z}\_{i})^{(k)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | subject to | q​(𝐳i)(k)≥0∀i,k𝑞superscriptsubscript𝐳𝑖𝑘  0for-all𝑖𝑘\displaystyle q(\mathbf{z}\_{i})^{(k)}\geq 0\hskip 30.0pt\forall i,k |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∑k=1Kq​(𝐳i)(k)=1∀isuperscriptsubscript𝑘1𝐾𝑞superscriptsubscript𝐳𝑖𝑘  1for-all𝑖\displaystyle\sum\_{k=1}^{K}q(\mathbf{z}\_{i})^{(k)}=1\hskip 12.0pt\forall i |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | ∑i=1nzi(yi)=∑i=1n∑k=1Kzi(k)​q​(𝐳i)(k).superscriptsubscript𝑖1𝑛superscriptsubscript𝑧𝑖subscript𝑦𝑖superscriptsubscript𝑖1𝑛superscriptsubscript𝑘1𝐾superscriptsubscript𝑧𝑖𝑘𝑞superscriptsubscript𝐳𝑖𝑘\displaystyle\sum\_{i=1}^{n}z\_{i}^{(y\_{i})}=\sum\_{i=1}^{n}\sum\_{k=1}^{K}z\_{i}^{(k)}q(\mathbf{z}\_{i})^{(k)}. |  |

The first two constraint ensure that q𝑞q is a probability distribution, while the last constraint limits the scope of distributions. Intuitively, the constraint specifies that the average true class logit is equal to the average weighted logit.

###### Proof.

We solve this constrained optimization problem using the Lagrangian. We first ignore the constraint q​(𝐳i)(k)𝑞superscriptsubscript𝐳𝑖𝑘q(\mathbf{z}\_{i})^{(k)} and later show that the solution satisfies this condition. Let λ,β1,…,βn∈ℝ

𝜆subscript𝛽1…subscript𝛽𝑛
ℝ\lambda,\beta\_{1},\ldots,\beta\_{n}\in\mathbb{R} be the Lagrangian multipliers and define

|  |  |  |  |
| --- | --- | --- | --- |
|  | L=𝐿absent\displaystyle L= | −∑i=1n∑k=1Kq​(𝐳i)(k)​log⁡q​(𝐳i)(k)superscriptsubscript𝑖1𝑛superscriptsubscript𝑘1𝐾𝑞superscriptsubscript𝐳𝑖𝑘𝑞superscriptsubscript𝐳𝑖𝑘\displaystyle-\sum\_{i=1}^{n}\sum\_{k=1}^{K}q(\mathbf{z}\_{i})^{(k)}\log q(\mathbf{z}\_{i})^{(k)} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +λ​∑i=1n[∑k=1Kzi(k)​q​(𝐳i)(k)−zi(yi)]𝜆superscriptsubscript𝑖1𝑛delimited-[]superscriptsubscript𝑘1𝐾superscriptsubscript𝑧𝑖𝑘𝑞superscriptsubscript𝐳𝑖𝑘superscriptsubscript𝑧𝑖subscript𝑦𝑖\displaystyle+\lambda\sum\_{i=1}^{n}\left[\sum\_{k=1}^{K}z\_{i}^{(k)}q(\mathbf{z}\_{i})^{(k)}-z\_{i}^{(y\_{i})}\right] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | +∑i=1nβi​∑k=1K(q​(𝐳i)(k)−1).superscriptsubscript𝑖1𝑛subscript𝛽𝑖superscriptsubscript𝑘1𝐾𝑞superscriptsubscript𝐳𝑖𝑘1\displaystyle+\sum\_{i=1}^{n}\beta\_{i}\sum\_{k=1}^{K}(q(\mathbf{z}\_{i})^{(k)}-1). |  |

Taking the derivative with respect to q​(𝐳i)(k)𝑞superscriptsubscript𝐳𝑖𝑘q(\mathbf{z}\_{i})^{(k)} gives

|  |  |  |
| --- | --- | --- |
|  | ∂∂q​(𝐳i)(k)​L=−n​K−log⁡q​(𝐳i)(k)+λ​zi(k)+βi.𝑞superscriptsubscript𝐳𝑖𝑘𝐿𝑛𝐾𝑞superscriptsubscript𝐳𝑖𝑘𝜆superscriptsubscript𝑧𝑖𝑘subscript𝛽𝑖\frac{\partial}{\partial q(\mathbf{z}\_{i})^{(k)}}L=-nK-\log q(\mathbf{z}\_{i})^{(k)}+\lambda z\_{i}^{(k)}+\beta\_{i}. |  |

Setting the gradient of the Lagrangian L𝐿L to 0 and rearranging gives

|  |  |  |
| --- | --- | --- |
|  | q​(𝐳i)(k)=eλ​zi(k)+βi−n​K.𝑞superscriptsubscript𝐳𝑖𝑘superscript𝑒𝜆superscriptsubscript𝑧𝑖𝑘subscript𝛽𝑖𝑛𝐾q(\mathbf{z}\_{i})^{(k)}=e^{\lambda z\_{i}^{(k)}+\beta\_{i}-nK}. |  |

Since ∑k=1Kq​(𝐳i)(k)=1superscriptsubscript𝑘1𝐾𝑞superscriptsubscript𝐳𝑖𝑘1\sum\_{k=1}^{K}q(\mathbf{z}\_{i})^{(k)}=1 for all i𝑖i, we must have

|  |  |  |
| --- | --- | --- |
|  | q​(𝐳i)(k)=eλ​zi(k)∑j=1Keλ​zi(j),𝑞superscriptsubscript𝐳𝑖𝑘superscript𝑒𝜆superscriptsubscript𝑧𝑖𝑘superscriptsubscript𝑗1𝐾superscript𝑒𝜆superscriptsubscript𝑧𝑖𝑗q(\mathbf{z}\_{i})^{(k)}=\frac{e^{\lambda z\_{i}^{(k)}}}{\sum\_{j=1}^{K}e^{\lambda z\_{i}^{(j)}}}, |  |

which recovers the temperature scaling model by setting T=1λ𝑇1𝜆T=\frac{1}{\lambda}.
∎

[Figure S1](#S2.F1 "Figure S1 ‣ S2 Further Information on Temperature Scaling ‣ On Calibration of Modern Neural Networks") visualizes Claim [1](#Thmclaim1 "Claim 1. ‣ S2 Further Information on Temperature Scaling ‣ On Calibration of Modern Neural Networks").
We see that, as training continues, the model begins to overfit with respect to NLL (red line).
This results in a low-entropy softmax distribution over classes (blue line), which explains the model’s overconfidence. Temperature scaling not only lowers the NLL but also raises the entropy of the distribution (green line).

![Refer to caption](/html/1706.04599/assets/x6.png)


Figure S1: Entropy and NLL for CIFAR-100 before and after calibration. The optimal T𝑇T selected by temperature scaling rises throughout optimization, as the pre-calibration entropy decreases steadily.
The post-calibration entropy and NLL on the validation set coincide
(which can be derived from the gradient optimality condition of T𝑇T).

## S3 Additional Tables

Tables [S1](#S3.T1 "Table S1 ‣ S3 Additional Tables ‣ On Calibration of Modern Neural Networks"), [S2](#S3.T2 "Table S2 ‣ S3 Additional Tables ‣ On Calibration of Modern Neural Networks"), and [S3](#S3.T3 "Table S3 ‣ S3 Additional Tables ‣ On Calibration of Modern Neural Networks") display the MCE, test error, and NLL for all the experimental settings outlined in [Section 5](#S5 "5 Results ‣ On Calibration of Modern Neural Networks").

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Model | Uncalibrated | Hist. Binning | Isotonic | BBQ | Temp. Scaling | Vector Scaling | Matrix Scaling |
| Birds | ResNet 50 | 30.06% | 25.35% | 16.59% | 11.72% | 9.08% | 9.81% | 38.67% |
| Cars | ResNet 50 | 41.55% | 5.16% | 15.23% | 9.31% | 20.23% | 8.59% | 29.65% |
| CIFAR-10 | ResNet 110 | 33.78% | 26.87% | 7.8% | 72.64% | 8.56% | 27.39% | 22.89% |
| CIFAR-10 | ResNet 110 (SD) | 34.52% | 17.0% | 16.45% | 19.26% | 15.45% | 15.55% | 10.74% |
| CIFAR-10 | Wide ResNet 32 | 27.97% | 12.19% | 6.19% | 9.22% | 9.11% | 4.43% | 9.65% |
| CIFAR-10 | DenseNet 40 | 22.44% | 7.77% | 19.54% | 14.57% | 4.58% | 3.17% | 4.36% |
| CIFAR-10 | LeNet 5 | 8.02% | 16.49% | 18.34% | 82.35% | 5.14% | 19.39% | 16.89% |
| CIFAR-100 | ResNet 110 | 35.5% | 7.03% | 10.36% | 10.9% | 4.74% | 2.5% | 45.62% |
| CIFAR-100 | ResNet 110 (SD) | 26.42% | 9.12% | 10.95% | 9.12% | 8.85% | 8.85% | 35.6% |
| CIFAR-100 | Wide ResNet 32 | 33.11% | 6.22% | 14.87% | 11.88% | 5.33% | 6.31% | 44.73% |
| CIFAR-100 | DenseNet 40 | 21.52% | 9.36% | 10.59% | 8.67% | 19.4% | 8.82% | 38.64% |
| CIFAR-100 | LeNet 5 | 10.25% | 18.61% | 3.64% | 9.96% | 5.22% | 8.65% | 18.77% |
| ImageNet | DenseNet 161 | 14.07% | 13.14% | 11.57% | 10.96% | 12.29% | 9.61% | - |
| ImageNet | ResNet 152 | 12.2% | 14.57% | 8.74% | 8.85% | 12.29% | 9.61% | - |
| SVHN | ResNet 152 (SD) | 19.36% | 11.16% | 18.67% | 9.09% | 18.05% | 30.78% | 18.76% |
| 20 News | DAN 3 | 17.03% | 10.47% | 9.13% | 6.28% | 8.21% | 8.24% | 17.43% |
| Reuters | DAN 3 | 14.01% | 16.78% | 44.95% | 36.18% | 25.46% | 18.88% | 19.39% |
| SST Binary | TreeLSTM | 21.66% | 3.22% | 13.91% | 36.43% | 6.03% | 6.03% | 6.03% |
| SST Fine Grained | TreeLSTM | 27.85% | 28.35% | 19.0% | 8.67% | 44.75% | 11.47% | 11.78% |

Table S1: MCE (%) (with M=15𝑀15M=15 bins) on standard vision and NLP datasets before calibration and with various calibration methods. The number following a model’s name denotes the network depth. MCE seems very sensitive to the binning scheme and is less suited for small test sets.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Model | Uncalibrated | Hist. Binning | Isotonic | BBQ | Temp. Scaling | Vector Scaling | Matrix Scaling |
| Birds | ResNet 50 | 22.54% | 55.02% | 23.37% | 37.76% | 22.54% | 22.99% | 29.51% |
| Cars | ResNet 50 | 14.28% | 16.24% | 14.9% | 19.25% | 14.28% | 14.15% | 17.98% |
| CIFAR-10 | ResNet 110 | 6.21% | 6.45% | 6.36% | 6.25% | 6.21% | 6.37% | 6.42% |
| CIFAR-10 | ResNet 110 (SD) | 5.64% | 5.59% | 5.62% | 5.55% | 5.64% | 5.62% | 5.69% |
| CIFAR-10 | Wide ResNet 32 | 6.96% | 7.3% | 7.01% | 7.35% | 6.96% | 7.1% | 7.27% |
| CIFAR-10 | DenseNet 40 | 5.91% | 6.12% | 5.96% | 6.0% | 5.91% | 5.96% | 6.0% |
| CIFAR-10 | LeNet 5 | 15.57% | 15.63% | 15.69% | 15.64% | 15.57% | 15.53% | 15.81% |
| CIFAR-100 | ResNet 110 | 27.83% | 34.78% | 28.41% | 28.56% | 27.83% | 27.82% | 38.77% |
| CIFAR-100 | ResNet 110 (SD) | 24.91% | 33.78% | 25.42% | 25.17% | 24.91% | 24.99% | 35.09% |
| CIFAR-100 | Wide ResNet 32 | 28.0% | 34.29% | 28.61% | 29.08% | 28.0% | 28.45% | 37.4% |
| CIFAR-100 | DenseNet 40 | 26.45% | 34.78% | 26.73% | 26.4% | 26.45% | 26.25% | 36.14% |
| CIFAR-100 | LeNet 5 | 44.92% | 54.06% | 45.77% | 46.82% | 44.92% | 45.53% | 52.44% |
| ImageNet | DenseNet 161 | 22.57% | 48.32% | 23.2% | 47.58% | 22.57% | 22.54% | - |
| ImageNet | ResNet 152 | 22.31% | 48.1% | 22.94% | 47.6% | 22.31% | 22.56% | - |
| SVHN | ResNet 152 (SD) | 1.98% | 2.06% | 2.04% | 2.04% | 1.98% | 2.0% | 2.08% |
| 20 News | DAN 3 | 20.06% | 25.12% | 20.29% | 20.81% | 20.06% | 19.89% | 22.0% |
| Reuters | DAN 3 | 2.97% | 7.81% | 3.52% | 3.93% | 2.97% | 2.83% | 3.52% |
| SST Binary | TreeLSTM | 11.81% | 12.08% | 11.75% | 11.26% | 11.81% | 11.81% | 11.81% |
| SST Fine Grained | TreeLSTM | 49.5% | 49.91% | 48.55% | 49.86% | 49.5% | 49.77% | 48.51% |

Table S2: Test error (%) on standard vision and NLP datasets before calibration and with various calibration methods. The number following a model’s name denotes the network depth. Error with temperature scaling is exactly the same as uncalibrated.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Model | Uncalibrated | Hist. Binning | Isotonic | BBQ | Temp. Scaling | Vector Scaling | Matrix Scaling |
| Birds | ResNet 50 | 0.9786 | 1.6226 | 1.4128 | 1.2539 | 0.8792 | 0.9021 | 2.334 |
| Cars | ResNet 50 | 0.5488 | 0.7977 | 0.8793 | 0.6986 | 0.5311 | 0.5299 | 1.0206 |
| CIFAR-10 | ResNet 110 | 0.3285 | 0.2532 | 0.2237 | 0.263 | 0.2102 | 0.2088 | 0.2048 |
| CIFAR-10 | ResNet 110 (SD) | 0.2959 | 0.2027 | 0.1867 | 0.2159 | 0.1718 | 0.1709 | 0.1766 |
| CIFAR-10 | Wide ResNet 32 | 0.3293 | 0.2778 | 0.2428 | 0.2774 | 0.2283 | 0.2275 | 0.2229 |
| CIFAR-10 | DenseNet 40 | 0.2228 | 0.212 | 0.1969 | 0.2087 | 0.1750 | 0.1757 | 0.176 |
| CIFAR-10 | LeNet 5 | 0.4688 | 0.529 | 0.4757 | 0.4984 | 0.459 | 0.4568 | 0.4607 |
| CIFAR-100 | ResNet 110 | 1.4978 | 1.4379 | 1.207 | 1.5466 | 1.0442 | 1.0485 | 2.5637 |
| CIFAR-100 | ResNet 110 (SD) | 1.1157 | 1.1985 | 1.0317 | 1.1982 | 0.8613 | 0.8655 | 1.8182 |
| CIFAR-100 | Wide ResNet 32 | 1.3434 | 1.4499 | 1.2086 | 1.459 | 1.0565 | 1.0648 | 2.5507 |
| CIFAR-100 | DenseNet 40 | 1.0134 | 1.2156 | 1.0615 | 1.1572 | 0.9026 | 0.9011 | 1.9639 |
| CIFAR-100 | LeNet 5 | 1.6639 | 2.2574 | 1.8173 | 1.9893 | 1.6560 | 1.6648 | 2.1405 |
| ImageNet | DenseNet 161 | 0.9338 | 1.4716 | 1.1912 | 1.4272 | 0.8885 | 0.8879 | - |
| ImageNet | ResNet 152 | 0.8961 | 1.4507 | 1.1859 | 1.3987 | 0.8657 | 0.8742 | - |
| SVHN | ResNet 152 (SD) | 0.0842 | 0.1137 | 0.095 | 0.1062 | 0.0821 | 0.0844 | 0.0924 |
| 20 News | DAN 3 | 0.7949 | 1.0499 | 0.8968 | 0.9519 | 0.7387 | 0.7296 | 0.9089 |
| Reuters | DAN 3 | 0.102 | 0.2403 | 0.1475 | 0.1167 | 0.0994 | 0.0990 | 0.1491 |
| SST Binary | TreeLSTM | 0.3367 | 0.2842 | 0.2908 | 0.2778 | 0.2739 | 0.2739 | 0.2739 |
| SST Fine Grained | TreeLSTM | 1.1475 | 1.1717 | 1.1661 | 1.149 | 1.1168 | 1.1085 | 1.1112 |

Table S3: NLL (%) on standard vision and NLP datasets before calibration and with various calibration methods. The number following a model’s name denotes the network depth. To summarize, NLL roughly follows the trends of ECE.

## S4 Additional Reliability Diagrams

![Refer to caption](/html/1706.04599/assets/x7.png)


Figure S2: Reliability diagrams for CIFAR-10 before (far left) and after calibration (middle left, middle right, far right).

![Refer to caption](/html/1706.04599/assets/x8.png)


Figure S3: Reliability diagrams for SST Binary and SST Fine Grained before (far left) and after calibration (middle left, middle right, far right).

![Refer to caption](/html/1706.04599/assets/x9.png)


Figure S4: Reliability diagrams for SST Binary and SST Fine Grained before (far left) and after calibration (middle left, middle right, far right).

We include reliability diagrams for additional datasets: CIFAR-10 ([Figure S2](#S4.F2 "Figure S2 ‣ S4 Additional Reliability Diagrams ‣ On Calibration of Modern Neural Networks")) and SST ([Figure S3](#S4.F3 "Figure S3 ‣ S4 Additional Reliability Diagrams ‣ On Calibration of Modern Neural Networks") and [Figure S4](#S4.F4 "Figure S4 ‣ S4 Additional Reliability Diagrams ‣ On Calibration of Modern Neural Networks")). Note that, as mentioned in [Section 2](#S2 "2 Definitions ‣ On Calibration of Modern Neural Networks"), the reliability diagrams do not represent the proportion of predictions that belong to a given bin.

[◄](/html/1706.04598)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1706.04599)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1706.04599)
[View original  
on arXiv](https://arxiv.org/abs/1706.04599)[►](/html/1706.04600)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 7 23:30:03 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
