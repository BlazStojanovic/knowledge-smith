---
arxiv: '1906.02530'
authors:
- Yaniv Ovadia
- Emily Fertig
- Jie Ren
- Zachary Nado
- D Sculley
- Sebastian Nowozin
- Joshua V. Dillon
- Balaji Lakshminarayanan
- Jasper Snoek
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under
  Dataset Shift
url: http://arxiv.org/abs/1906.02530v2
year: 2019
---

[1906.02530] Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift














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



# Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift

Yaniv Ovadia
  
Google Research
  
yovadia@google.com
&Emily Fertig11footnotemark: 1 
  
Google Research
  
emilyaf@google.com
&Jie Ren22footnotemark: 2
  
Google Research
  
jjren@google.com
&Zachary Nado
  
Google Research
  
znado@google.com
&D Sculley
  
Google Research
  
dsculley@google.com
&Sebastian Nowozin
  
Google Research
  
nowozin@google.com
  
&Joshua V. Dillon
  
Google Research
  
jvdillon@google.com
  
&Balaji Lakshminarayanan
  
DeepMind
  
balajiln@google.com
  
&Jasper Snoek33footnotemark: 3
  
Google Research
  
jsnoek@google.com
Equal contributionAI ResidentCorresponding authors

###### Abstract

Modern machine learning methods including deep learning have achieved great success in predictive accuracy for supervised learning tasks, but may still fall short in giving useful estimates of their predictive uncertainty. Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model’s output should (or should not) be trusted. Many probabilistic deep learning methods, including Bayesian-and non-Bayesian methods, have been proposed in the literature for quantifying predictive uncertainty, but to our knowledge there has not previously been a rigorous large-scale empirical comparison of these methods under dataset shift. We present a large-scale benchmark of existing state-of-the-art methods on classification problems and investigate the effect of dataset shift on accuracy and calibration. We find that traditional post-hoc calibration does indeed fall short, as do several other previous methods. However, some methods that marginalize over models give surprisingly strong results across a broad spectrum of tasks.

## 1 Introduction

Recent successes across a variety of domains have led to the widespread deployment of deep neural networks (DNNs) in practice. Consequently, the predictive distributions of these models are increasingly being used to make decisions in important applications ranging from machine-learning aided medical diagnoses from imaging (Esteva et al., [2017](#bib.bib13)) to self-driving cars (Bojarski et al., [2016](#bib.bib6)). Such high-stakes applications require not only point predictions but also accurate quantification of predictive uncertainty, i.e.  meaningful confidence values in addition to class predictions. With sufficient independent labeled samples from a target data distribution, one can estimate how well a model’s confidence aligns with its accuracy and adjust the predictions accordingly. However, in practice, once a model is deployed the distribution over observed data may shift and eventually be very different from the original training data distribution. Consider, e.g., online services for which the data distribution may change with the time of day, seasonality or popular trends. Indeed, robustness under conditions of distributional shift and out-of-distribution (OOD) inputs is necessary for the safe deployment of machine learning (Amodei et al., [2016](#bib.bib2)). For such settings, calibrated predictive uncertainty is important because it enables accurate assessment of risk, allows practitioners to know how accuracy may degrade, and allows a system to abstain from decisions due to low confidence.

A variety of methods have been developed for quantifying predictive uncertainty in DNNs. Probabilistic neural networks such as mixture density networks (MacKay & Gibbs, [1999](#bib.bib41)) capture the inherent ambiguity in outputs for a given input, also referred to as *aleatoric uncertainty* (Kendall & Gal, [2017](#bib.bib26)). Bayesian neural networks learn a posterior distribution over parameters that quantifies parameter uncertainty, a type of *epistemic uncertainty* that can be reduced through the collection of additional data. Popular approximate Bayesian approaches include Laplace approximation (MacKay, [1992](#bib.bib40)), variational inference (Graves, [2011](#bib.bib18); Blundell et al., [2015](#bib.bib5)), dropout-based variational inference (Gal & Ghahramani, [2016](#bib.bib14); Kingma et al., [2015](#bib.bib29)), expectation propagation Hernández-Lobato & Adams ([2015](#bib.bib24)) and stochastic gradient MCMC (Welling & Teh, [2011](#bib.bib55)). Non-Bayesian methods include training multiple probabilistic neural networks with bootstrap or ensembling (Osband et al., [2016](#bib.bib45); Lakshminarayanan et al., [2017](#bib.bib32)).
Another popular non-Bayesian approach involves re-calibration of probabilities on a held-out validation set through temperature scaling (Platt, [1999](#bib.bib46)), which was shown by Guo et al. ([2017](#bib.bib19)) to lead to well-calibrated predictions on the i.i.d. test set.

Using Distributional Shift to Evaluate Predictive Uncertainty
While previous work has evaluated the quality of predictive uncertainty on OOD inputs (Lakshminarayanan et al., [2017](#bib.bib32)),
there has not to our knowledge been a comprehensive evaluation of uncertainty estimates from different methods under dataset shift. Indeed, we suggest that effective evaluation of predictive uncertainty is most meaningful under conditions of distributional shift. One reason for this is that post-hoc calibration gives good results in independent and identically distributed (i.i.d.) regimes, but can fail under even a mild shift in the input data. And in real world applications, as described above, distributional shift is widely prevalent. Understanding questions of risk, uncertainty, and trust in a model’s output becomes increasingly critical as shift from the original training data grows larger.

Contributions In the spirit of calls for more rigorous understanding of existing methods (Lipton & Steinhardt, [2018](#bib.bib37); Sculley et al., [2018](#bib.bib51); Rahimi & Recht, [2017](#bib.bib48)), this paper provides a benchmark for evaluating uncertainty that focuses not only on the i.i.d. setting but also *uncertainty under distributional shift.* We present a large-scale evaluation of popular approaches in probabilistic deep learning, focusing on methods that operate well in large-scale settings, and evaluate them on a diverse range of classification benchmarks across image, text, and categorical modalities. We use these experiments to evaluate the following questions:

* •

  How trustworthy are the uncertainty estimates of different methods under dataset shift?
* •

  Does calibration in the i.i.d. setting translate to calibration under dataset shift?
* •

  How do uncertainty and accuracy of different methods co-vary under dataset shift? Are there methods that consistently do well in this regime?

In addition to answering the questions above, our code is made available open-source along with our model predictions such that researchers can easily evaluate their approaches on these benchmarks 111[https://github.com/google-research/google-research/tree/master/uq˙benchmark˙2019](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019).

## 2 Background

Notation and Problem Setup
Let 𝒙∈ℝd𝒙superscriptℝ𝑑{\bm{x}}\in\mathbb{R}^{d} represent a set of d𝑑d-dimensional features and y∈{1,…,k}𝑦1…𝑘y\in\{1,\ldots,k\} denote corresponding labels (targets) for k𝑘k-class classification.
We assume that a training dataset 𝒟𝒟\mathcal{D} consists of N𝑁N i.i.d.samples 𝒟={(𝒙n,yn)}n=1N𝒟superscriptsubscriptsubscript𝒙𝑛subscript𝑦𝑛𝑛1𝑁\mathcal{D}=\{({\bm{x}}\_{n},y\_{n})\}\_{n=1}^{N}.

Let p∗​(𝒙,y)superscript𝑝𝒙𝑦p^{\*}({\bm{x}},y) denote the true distribution (unknown, observed only through the samples 𝒟𝒟\mathcal{D}), also referred to as the *data generating process*.
We focus on classification problems, in which the true distribution is assumed to be a discrete distribution over k𝑘k classes, and the observed y∈{1,…,k}𝑦1…𝑘y\in\{1,\ldots,k\} is a sample from the conditional distribution p∗​(y|𝒙)superscript𝑝conditional𝑦𝒙p^{\*}(y|{\bm{x}}).
We use a neural network to model p𝜽​(y|𝒙)subscript𝑝𝜽conditional𝑦𝒙p\_{\bm{\theta}}(y|{\bm{x}}) and estimate the parameters 𝜽𝜽{\bm{\theta}} using the training dataset.
At test time, we evaluate the model predictions against a test set, sampled from the same distribution as the training dataset.
However, here we also evaluate the model against OOD inputs sampled from q​(𝒙,y)≠p∗​(𝒙,y)𝑞𝒙𝑦superscript𝑝𝒙𝑦q({\bm{x}},y)\neq p^{\*}({\bm{x}},y).
In particular, we consider two kinds of shifts:

* •

  *shifted versions* of the test inputs where the ground truth label belongs to one of the k𝑘k classes. We use shifts such as corruptions and perturbations proposed by Hendrycks & Dietterich ([2019](#bib.bib21)), and ideally would like the model predictions to become more uncertain with increased shift, assuming shift degrades accuracy. This is also referred to as *covariate shift* (Sugiyama et al., [2009](#bib.bib54)).
* •

  *a completely different OOD dataset*, where the ground truth label is not one of the k𝑘k classes. Here we check if the model exhibits higher predictive uncertainty for those new instances and to this end report diagnostics that rely only on predictions and not ground truth labels.

High-level overview of existing methods A large variety of methods have been developed to either provide higher quality uncertainty estimates or perform OOD detection to inform model confidence. These can roughly be divided into:

1. 1.

   Methods which deal with p​(y|𝒙)𝑝conditional𝑦𝒙p(y|{\bm{x}}) only, we discuss these in more detail in Section [3](#S3 "3 Methods and Metrics ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").
2. 2.

   Methods which model the joint distribution p​(y,𝒙)𝑝𝑦𝒙p(y,{\bm{x}}), e.g. deep hybrid models (Kingma et al., [2014](#bib.bib28); Alemi et al., [2018](#bib.bib1); Nalisnick et al., [2019](#bib.bib43); Behrmann et al., [2018](#bib.bib3)).
3. 3.

   Methods with an OOD-detection component in addition to p​(y|𝒙)𝑝conditional𝑦𝒙p(y|{\bm{x}}) (Bishop, [1994](#bib.bib4); Lee et al., [2018](#bib.bib35); Liang et al., [2018](#bib.bib36)), and related work on selective classification (Geifman & El-Yaniv, [2017](#bib.bib15)).

We refer to Shafaei et al. ([2018](#bib.bib52)) for a recent summary of these methods.
Due to the differences in modeling assumptions, a fair comparison between these different classes of methods is challenging; for instance, some OOD detection methods rely on knowledge of a known OOD set, or train using a none-of-the-above class, and it may not always be meaningful to compare predictions from these methods with those obtained from a Bayesian DNN. We focus on methods described by (1) above, as this allows us to focus on methods which make the same modeling assumptions about data and differ only in how they quantify predictive uncertainty.

## 3 Methods and Metrics

We select a subset of methods from the probabilistic deep learning literature for their prevalence, scalability and practical applicability222The methods used scale well for training and prediction (see in Appendix [A.9](#A1.SS9 "A.9 Computational and Memory Complexity of Different methods ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").). We also explored methods such as scalable extensions of Gaussian Processes (Hensman et al., [2015](#bib.bib23)), but they were challenging to train on the 37M example Criteo dataset or the 1000 classes of ImageNet.. These include (see also references within):

* •

  (*Vanilla*) Maximum softmax probability (Hendrycks & Gimpel, [2017](#bib.bib22))
* •

  (*Temp Scaling*) Post-hoc calibration by temperature scaling using a validation set (Guo et al., [2017](#bib.bib19))
* •

  (*Dropout*) Monte-Carlo Dropout (Gal & Ghahramani, [2016](#bib.bib14); Srivastava et al., [2015](#bib.bib53)) with rate p𝑝p
* •

  (*Ensembles*) Ensembles of M𝑀M networks trained independently on the entire dataset using random initialization (Lakshminarayanan et al., [2017](#bib.bib32)) (we set M=10𝑀10M=10 in experiments below)
* •

  (*SVI*) Stochastic Variational Bayesian Inference for deep learning
  (Blundell et al., [2015](#bib.bib5); Graves, [2011](#bib.bib18); Louizos & Welling, [2017](#bib.bib39), [2016](#bib.bib38); Wen et al., [2018](#bib.bib56)). We refer to Appendix [A.6](#A1.SS6 "A.6 Stochastic Variational Inference Details ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for details of our SVI implementation.
* •

  (LL) Approx. Bayesian inference for the parameters of the last layer only (Riquelme et al., [2018](#bib.bib50))

  + –

    (*LL SVI*) Mean field stochastic variational inference on the last layer only
  + –

    (*LL Dropout*) Dropout only on the activations before the last layer

In addition to metrics (we use arrows to indicate which direction is better) that do not depend on predictive uncertainty, such as classification accuracy ↑↑\uparrow,
the following metrics are commonly used:

Negative Log-Likelihood (NLL) ↓↓\downarrow
Commonly used to evaluate the quality of model uncertainty on some held out set.
Drawbacks: Although a proper scoring rule (Gneiting & Raftery, [2007](#bib.bib16)), it can over-emphasize tail probabilities (Quinonero-Candela et al., [2006](#bib.bib47)).

Brier Score ↓↓\downarrow (Brier, [1950](#bib.bib7))
Proper scoring rule for measuring the accuracy of predicted probabilities. It is computed as the squared error of a predicted probability vector, p​(y|xn,𝜽)𝑝conditional𝑦

subscript𝑥𝑛𝜽p(y|x\_{n},{\bm{\theta}}), and the one-hot encoded true response, ynsubscript𝑦𝑛y\_{n}. That is,

|  |  |  |  |
| --- | --- | --- | --- |
|  | BS=|𝒴|−1​∑y∈𝒴(p​(y|𝒙n,𝜽)−δ​(y−yn))2=|𝒴|−1​(1−2​p​(yn|𝒙n,𝜽)+∑y∈𝒴p​(y|𝒙n,𝜽)2).BSsuperscript𝒴1subscript𝑦𝒴superscript𝑝conditional𝑦  subscript𝒙𝑛𝜽𝛿𝑦subscript𝑦𝑛2superscript𝒴112𝑝conditionalsubscript𝑦𝑛  subscript𝒙𝑛𝜽subscript𝑦𝒴𝑝superscriptconditional𝑦  subscript𝒙𝑛𝜽2\mathrm{BS}=|\mathcal{Y}|^{-1}\sum\_{y\in\mathcal{Y}}(p(y|{\bm{x}}\_{n},{\bm{\theta}})-\delta(y-y\_{n}))^{2}=|\mathcal{Y}|^{-1}\Big{(}1-2p(y\_{n}|{\bm{x}}\_{n},{\bm{\theta}})+\sum\_{y\in\mathcal{Y}}p(y|{\bm{x}}\_{n},{\bm{\theta}})^{2}\Big{)}. |  | (1) |

The Brier score has a convenient interpretation as B​S=uncertainty−resolution+reliability𝐵𝑆uncertaintyresolutionreliabilityBS=\mathrm{uncertainty}-\mathrm{resolution}+\mathrm{reliability}, where
uncertaintyuncertainty\mathrm{uncertainty} is the marginal uncertainty over labels,
resolutionresolution\mathrm{resolution} measures the deviation of individual predictions against the marginal, and
reliabilityreliability\mathrm{reliability} measures calibration as the average violation of long-term true label frequencies.
We refer to DeGroot & Fienberg ([1983](#bib.bib11)) for the decomposition of Brier score into calibration and refinement for classification and to (Bröcker, [2009](#bib.bib8)) for the general decomposition for any proper scoring rule.
Drawbacks: Brier score is insensitive to predicted probabilities associated with in/frequent events.

Both the Brier score and the negative log-likelihood are proper scoring rules and therefore the optimum score corresponds to a perfect prediction.
In addition to these two metrics, we also evaluate two metrics—*expected calibration error* and *entropy*.
Neither of these is a proper scoring rule, and thus there exist trivial solutions which yield optimal scores; for example, returning the marginal probability p​(y)𝑝𝑦p(y) for every instance will yield perfectly calibrated but uninformative predictions. Each proper scoring rule induces a calibration measure (Bröcker, [2009](#bib.bib8)). However, ECE is not the result of such decomposition and has no corresponding proper scoring rule; we instead include ECE because it is popularly used and intuitive. Each proper scoring rule is also associated with a corresponding entropy function and Shannon entropy is that for log probability (Gneiting & Raftery, [2007](#bib.bib16)).

Expected Calibration Error (ECE) ↓↓\downarrow
Measures the correspondence between predicted probabilities and empirical accuracy (Naeini et al., [2015](#bib.bib42)). It is computed as the average gap between within bucket accuracy and within bucket predicted probability for S𝑆S buckets Bs={n∈1​…​N:p​(yn|𝒙n,𝜽)∈(ρs,ρs+1]}subscript𝐵𝑠conditional-set𝑛1…𝑁𝑝conditionalsubscript𝑦𝑛

subscript𝒙𝑛𝜽subscript𝜌𝑠subscript𝜌𝑠1B\_{s}=\{n\in 1\ldots N:p(y\_{n}|{\bm{x}}\_{n},{\bm{\theta}})\in(\rho\_{s},\rho\_{s+1}]\}. That is, ECE=∑s=1S|Bs|N​|acc⁡(Bs)−conf⁡(Bs)|,ECEsuperscriptsubscript𝑠1𝑆subscript𝐵𝑠𝑁accsubscript𝐵𝑠confsubscript𝐵𝑠\mathrm{ECE}=\sum\_{s=1}^{S}\frac{|B\_{s}|}{N}|\operatorname{acc}(B\_{s})-\operatorname{conf}(B\_{s})|,
where acc⁡(Bs)=|Bs|−1​∑n∈Bs[yn=y^n]accsubscript𝐵𝑠superscriptsubscript𝐵𝑠1subscript𝑛subscript𝐵𝑠delimited-[]subscript𝑦𝑛subscript^𝑦𝑛\operatorname{acc}(B\_{s})=|B\_{s}|^{-1}\sum\_{n\in B\_{s}}[y\_{n}=\hat{y}\_{n}], conf⁡(Bs)=|Bs|−1​∑n∈Bsp​(y^n|𝒙n,𝜽)confsubscript𝐵𝑠superscriptsubscript𝐵𝑠1subscript𝑛subscript𝐵𝑠𝑝conditionalsubscript^𝑦𝑛

subscript𝒙𝑛𝜽\operatorname{conf}(B\_{s})=|B\_{s}|^{-1}\sum\_{n\in B\_{s}}p(\hat{y}\_{n}|{\bm{x}}\_{n},{\bm{\theta}}), and y^n=arg⁡maxy⁡p​(y|𝒙n,𝜽)subscript^𝑦𝑛subscript𝑦𝑝conditional𝑦

subscript𝒙𝑛𝜽\hat{y}\_{n}=\arg\max\_{y}p(y|{\bm{x}}\_{n},{\bm{\theta}}) is the n𝑛n-th prediction. When bins {ρs:s∈1​…​S}conditional-setsubscript𝜌𝑠𝑠1…𝑆\{\rho\_{s}:s\in 1\ldots S\} are quantiles of the held-out predicted probabilities, |Bs|≈|Bk|subscript𝐵𝑠subscript𝐵𝑘|B\_{s}|\approx|B\_{k}| and the estimation error is approximately constant. Drawbacks: Due to binning, ECE does not monotonically increase as predictions approach ground truth. If |Bs|≠|Bk|subscript𝐵𝑠subscript𝐵𝑘|B\_{s}|\neq|B\_{k}|, the estimation error varies across bins.

There is no ground truth label for fully OOD inputs. Thus we report histograms of confidence and predictive entropy on known and OOD inputs and accuracy versus confidence plots (Lakshminarayanan et al., [2017](#bib.bib32)):
Given the prediction p​(y=k|𝒙n,𝜽)𝑝𝑦conditional𝑘

subscript𝒙𝑛𝜽p(y=k|{\bm{x}}\_{n},{\bm{\theta}}), we define the predicted label as y^n=arg⁡maxy⁡p​(y|𝒙n,𝜽)subscript^𝑦𝑛subscript𝑦𝑝conditional𝑦

subscript𝒙𝑛𝜽\hat{y}\_{n}=\arg\max\_{y}p(y|{\bm{x}}\_{n},{\bm{\theta}}), and the confidence as p​(y=y^|𝒙,𝜽)=maxk⁡p​(y=k|𝒙n,𝜽)𝑝𝑦conditional^𝑦

𝒙𝜽subscript𝑘𝑝𝑦conditional𝑘

subscript𝒙𝑛𝜽p(y=\hat{y}|{\bm{x}},{\bm{\theta}})=\max\_{k}p(y=k|{\bm{x}}\_{n},{\bm{\theta}}). We filter out test examples corresponding to a particular confidence threshold τ∈[0,1]𝜏01\tau\in[0,1] and compute the accuracy on this set.

## 4 Experiments and Results

We evaluate the behavior of the predictive uncertainty of deep learning models on a variety of datasets across three different modalities: images, text and categorical (online ad) data. For each we follow standard training, validation and testing protocols, but we additionally evaluate results on increasingly shifted data and an OOD dataset. We detail the models and implementations used in Appendix [A](#A1 "Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"). Hyperparameters were tuned for all methods using Bayesian optimization (Golovin et al., [2017](#bib.bib17)) (except on ImageNet) as detailed in Appendix [A.8](#A1.SS8 "A.8 Hyperparameter Tuning ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

### 4.1 An illustrative example - MNIST

![Refer to caption](/html/1906.02530/assets/x1.png)


(a) Rotated MNIST

![Refer to caption](/html/1906.02530/assets/x2.png)


(b) Translated MNIST

![Refer to caption](/html/1906.02530/assets/x3.png)


(c) Confidence vs Acc Rotated 60∘

![Refer to caption](/html/1906.02530/assets/x4.png)


(d) Count vs Confidence Rotated 60∘

![Refer to caption](/html/1906.02530/assets/x5.png)


(e) Entropy on OOD

![Refer to caption](/html/1906.02530/assets/x6.png)


(f) Confidence on OOD

Figure 1: Results on MNIST:
[1(a)](#S4.F1.sf1 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [1(b)](#S4.F1.sf2 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") show accuracy and Brier score as the data is increasingly shifted. Shaded regions represent standard error over 10 runs. To understand the discrepancy between accuracy and Brier score, we explore the predictive distributions of each method by looking at the confidence of the predictions in [1(c)](#S4.F1.sf3 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [1(d)](#S4.F1.sf4 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"). We also explore the entropy and confidence of each method on entirely OOD data in [1(e)](#S4.F1.sf5 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [1(f)](#S4.F1.sf6 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"). SVI has lower accuracy on the validation and test splits, but it is significantly more robust to dataset shift as evidenced by a lower Brier score, lower overall confidence [1(d)](#S4.F1.sf4 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and higher predictive entropy under shift ([1(c)](#S4.F1.sf3 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")) and OOD data ([1(e)](#S4.F1.sf5 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"),[1(f)](#S4.F1.sf6 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")).

We first illustrate the problem setup and experiments using the MNIST dataset. We used the LeNet (LeCun et al., [1998](#bib.bib34)) architecture, and, as with all our experiments, we follow standard training, validation, testing and hyperparameter tuning protocols. However, we also compute predictions on increasingly shifted data (in this case increasingly rotated or horizontally translated images) and study the behavior of the predictive distributions of the models. In addition, we predict on a completely OOD dataset, Not-MNIST (Bulatov, [2011](#bib.bib9)),
and observe the entropy of the model’s predictions. We summarize some of our findings in Figure [1](#S4.F1 "Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and discuss below.

What we would like to see: Naturally, we expect the accuracy of a model to degrade as it predicts on increasingly shifted data, and ideally this reduction in accuracy would coincide with increased forecaster entropy. A model that was well-calibrated on the training and validation distributions would ideally remain so on shifted data. If calibration (ECE or Brier reliability) remained as consistent as possible, practitioners and downstream tasks could take into account that a model is becoming increasingly uncertain. On the completely OOD data, one would expect the predictive distributions to be of high entropy. Essentially, we would like the predictions to indicate that a model “knows what it does not know” due to the inputs
straying away from the training data distribution.

What we observe: We see in Figures [1(a)](#S4.F1.sf1 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [1(b)](#S4.F1.sf2 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") that accuracy certainly degrades as a function of shift for all methods tested, and they are difficult to disambiguate on that metric.
However, the Brier score paints a clearer picture and we see a significant difference between methods, i.e. prediction quality degrades more significantly for some methods than others. An important observation is that *while calibrating on the validation set leads to well-calibrated predictions on the test set, it does not guarantee calibration on shifted data*. In fact, nearly all other methods (except vanilla) perform better than the state-of-the-art post-hoc calibration (Temperature scaling) in terms of Brier score under shift. While SVI achieves the worst accuracy on the test set, it actually outperforms all other methods by a much larger margin when exposed to significant shift. In Figures [1(c)](#S4.F1.sf3 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [1(d)](#S4.F1.sf4 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") we look at the distribution of confidences for each method to understand the discrepancy between metrics. We see in Figure [1(d)](#S4.F1.sf4 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") that SVI has the lowest confidence in general but in Figure [1(c)](#S4.F1.sf3 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") we observe that SVI gives the highest accuracy at high confidence (or conversely is much less frequently confidently wrong), which can be important for high-stakes applications. Most methods demonstrate very low entropy (Figure [1(e)](#S4.F1.sf5 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")) and give high confidence predictions (Figure [1(f)](#S4.F1.sf6 "In Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")) on data that is entirely OOD, i.e. they are confidently wrong about completely OOD data.

![Refer to caption](/html/1906.02530/assets/x7.png)


Figure 2: Calibration under distributional shift: a detailed comparison of accuracy and ECE under all types of corruptions on
(a) CIFAR-10 and (b) ImageNet.
For each method we show the mean on the test set and summarize the results on each intensity of shift with a box plot. Each box shows the quartiles summarizing the results across all (16) types of shift while the error bars indicate the min and max across different shift types. Figures showing additional metrics are provided in Figures [S4](#A3.F4 "Figure S4 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") (CIFAR-10) and [S5](#A3.F5 "Figure S5 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") (ImageNet). Tables for numerical comparisons are provided in Appendix [G](#A7 "Appendix G Tables of Metrics ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

### 4.2 Image Models: CIFAR-10 and ImageNet

![Refer to caption](/html/1906.02530/assets/x8.png)


(a) CIFAR: Confidence vs Accuracy

![Refer to caption](/html/1906.02530/assets/x9.png)


(b) CIFAR: Count vs Confidence

![Refer to caption](/html/1906.02530/assets/x10.png)


(c) CIFAR: Entropy on OOD

![Refer to caption](/html/1906.02530/assets/x11.png)


(d) ImageNet: Confidence vs Acc

![Refer to caption](/html/1906.02530/assets/x12.png)


(e) ImageNet: Count vs Confidence

![Refer to caption](/html/1906.02530/assets/x13.png)


(f) CIFAR: Confidence on OOD

Figure 3: 
Results on CIFAR-10 and ImageNet.
Left column: [3(a)](#S4.F3.sf1 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [3(d)](#S4.F3.sf4 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") show accuracy as a function of confidence. Middle column: [3(b)](#S4.F3.sf2 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [3(e)](#S4.F3.sf5 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")
show the number of examples greater than given confidence values for Gaussian blur of intensity 3.
Right column: [3(c)](#S4.F3.sf3 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [3(f)](#S4.F3.sf6 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") show histogram of entropy and confidences from CIFAR-trained models on a completely different dataset (SVHN).

We now study the predictive distributions of residual networks (He et al., [2016](#bib.bib20)) trained on two benchmark image datasets, CIFAR-10 (Krizhevsky, [2009](#bib.bib31)) and ImageNet (Deng et al., [2009](#bib.bib12)), under distributional shift. We use 20-layer and 50-layer ResNets for CIFAR-10 and ImageNet respectively. For shifted data we use 80 different distortions (16 different types with 5 levels of intensity each, see Appendix [B](#A2 "Appendix B Shifted Images ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for illustrations) introduced by Hendrycks & Dietterich ([2019](#bib.bib21)).
To evaluate predictions of CIFAR-10 models on entirely OOD data, we use the SVHN dataset (Netzer et al., [2011](#bib.bib44)).

Figure [2](#S4.F2 "Figure 2 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") summarizes the accuracy and ECE for CIFAR-10 (top) and ImageNet (bottom) across all 80 combinations of corruptions and intensities from (Hendrycks & Dietterich, [2019](#bib.bib21)). Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") inspects the predictive distributions of the models on CIFAR-10 (top) and ImageNet (bottom) for shifted (Gaussian blur) and OOD data. Classifiers on both datasets show poorer accuracy and calibration with increasing shift. Comparing accuracy for different methods, we see that ensembles achieve highest accuracy under distributional shift. Comparing the ECE for different methods, we observe that while the methods achieve comparable low values of ECE for small values of shift, ensembles outperform the other methods for larger values of shift. To test whether this result is due simply to the larger aggregate capacity of the ensemble, we trained models with double the number of filters for the Vanilla and Dropout methods. The higher-capacity models showed no better accuracy or calibration for medium- to high-shift than the corresponding lower-capacity models (see Appendix [C](#A3 "Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")). In Figures [S8](#A4.F8 "Figure S8 ‣ Appendix D Effect of the number of samples on the quality of uncertainty ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [S9](#A4.F9 "Figure S9 ‣ Appendix D Effect of the number of samples on the quality of uncertainty ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") we also explore the effect of the number of samples used in dropout, SVI and last layer methods and size of the ensemble, on CIFAR-10. We found that while increasing ensemble size up to 50 did help, most of the gains of ensembling could be achieved with only 5 models.
Interestingly, *while temperature scaling achieves low ECE for low values of shift, the ECE increases significantly as the shift increases, which indicates that calibration on the i.i.d. validation dataset does not guarantee calibration under distributional shift*. (Note that for ImageNet, we found similar trends considering just the top-5 predicted classes, See Figure [S5](#A3.F5 "Figure S5 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").) Furthermore, the results show that while temperature scaling helps significantly over the vanilla method, ensembles and dropout tend to be better. In Figure [3](#S4.F3 "Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"), we see that ensembles and dropout are more accurate at higher confidence. However, in [3(c)](#S4.F3.sf3 "In Figure 3 ‣ 4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") we see that temperature scaling gives the highest entropy on OOD data. Ensembles consistently have high accuracy but also high entropy on OOD data.
We refer to Appendix [C](#A3 "Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for additional results; Figures [S4](#A3.F4 "Figure S4 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [S5](#A3.F5 "Figure S5 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") report additional metrics on CIFAR-10 and ImageNet, such as Brier score (and its component terms), as well as top-5 error for increasing values of shift.

Overall, ensembles consistently perform best across metrics and dropout consistently performed better than temperature scaling and last layer methods. *While the relative ordering of methods is consistent on both CIFAR-10 and ImageNet (ensembles perform best), the ordering is quite different from that on MNIST where SVI performs best.* Interestingly, LL-SVI and LL-Dropout perform worse than the vanilla method on shifted datasets as well as SVHN. We also evaluate a variational Gaussian process as a last layer method in Appendix [E](#A5 "Appendix E Variational Gaussian Process Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") but it did not outperform LL-SVI and LL-Dropout.

### 4.3 Text Models

Following Hendrycks & Gimpel ([2017](#bib.bib22)), we train an LSTM (Hochreiter & Schmidhuber, [1997](#bib.bib25)) on the 20newsgroups dataset (Lang, [1995](#bib.bib33)) and assess the model’s robustness under distributional shift and OOD text.
We use the even-numbered classes (10 classes out of 20) as in-distribution and the 10 odd-numbered classes as shifted data. We provide additional details in Appendix [A.4](#A1.SS4 "A.4 20 Newsgroups ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

We look at confidence vs accuracy when the test data consists of a mix of in-distribution and either shifted or completely OOD data, in this case the One Billion Word Benchmark (LM1B) (Chelba et al., [2013](#bib.bib10)). Figure [4](#S4.F4 "Figure 4 ‣ 4.3 Text Models ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") (bottom row) shows the results. Ensembles significantly outperform all other methods, and achieve better trade-off between accuracy versus confidence. Surprisingly, LL-Dropout and LL-SVI perform worse than the vanilla method, giving higher confidence incorrect predictions, especially when tested on fully OOD data.

![Refer to caption](/html/1906.02530/assets/x14.png)

![Refer to caption](/html/1906.02530/assets/x15.png)

![Refer to caption](/html/1906.02530/assets/x16.png)

![Refer to caption](/html/1906.02530/assets/x17.png)

![Refer to caption](/html/1906.02530/assets/x18.png)

![Refer to caption](/html/1906.02530/assets/x19.png)

![Refer to caption](/html/1906.02530/assets/x20.png)


() Confidence vs Acc.

![Refer to caption](/html/1906.02530/assets/x21.png)


() Confidence vs Count

![Refer to caption](/html/1906.02530/assets/x22.png)


() Confidence vs Accuracy

![Refer to caption](/html/1906.02530/assets/x23.png)


() Confidence vs Count

Figure 4: Top row: Histograms of the entropy of the predictive distributions for in-distribution (solid lines), shifted (dotted lines), and completely different OOD (dashed lines) text examples. Bottom row: Confidence score vs accuracy and count respectively when evaluated for in-distribution and in-distribution shift text examples (a,b), and in-distribution and OOD text examples (c,d).

Figure [4](#S4.F4 "Figure 4 ‣ 4.3 Text Models ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") reports histograms of predictive entropy on in-distribution data and compares them to those for the shifted and OOD datasets. This reflects how amenable each method is to abstaining from prediction by applying a threshold on the entropy. As expected, most methods achieve the highest predictive entropy on the completely OOD dataset, followed by the shifted dataset and then the in-distribution test dataset. Only ensembles have consistently higher entropy on the shifted data, which explains why they perform best on the confidence vs accuracy curves in the second row of Figure [4](#S4.F4 "Figure 4 ‣ 4.3 Text Models ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"). Compared with the vanilla model, Dropout and LL-SVI have more a distinct separation between in-distribution and shifted or OOD data. While Dropout and LL-Dropout perform similarly on in-distribution, LL-Dropout exhibits less uncertainty than Dropout on shifted and OOD data. Temperature scaling does not appear to increase uncertainty significantly on the shifted data.

### 4.4 Ad-Click Model with Categorical Features

Finally, we evaluate the performance of different methods on the *Criteo Display Advertising Challenge*333<https://www.kaggle.com/c/criteo-display-ad-challenge> dataset, a binary classification task consisting of 37M examples with 13 numerical and 26 categorical features per example.
We introduce shift by reassigning each categorical feature to a random new token with some fixed probability that controls the intensity of shift.
This coarsely simulates a type of shift observed in non-stationary categorical features as category tokens appear and disappear over time, for example due to hash collisions.
The model consists of a 3-hidden-layer multi-layer-perceptron (MLP) with hashed and embedded categorical features and achieves a negative log-likelihood of approximately 0.5 (contest winners achieved 0.44). Due to class imbalance (∼25%similar-toabsentpercent25\sim 25\% of examples are positive), we report AUC instead of classification accuracy.

![Refer to caption](/html/1906.02530/assets/x24.png)

![Refer to caption](/html/1906.02530/assets/x25.png)

![Refer to caption](/html/1906.02530/assets/x26.png)

![Refer to caption](/html/1906.02530/assets/x27.png)

Figure 5: Results on Criteo: The first two plots show degrading AUCs and Brier scores with increasing shift while the latter two depict the distribution of prediction confidences and their corresponding accuracies at 75% randomization of categorical features.
SVI is excluded as it performed too poorly.

Results from these experiments are depicted in Figure [5](#S4.F5 "Figure 5 ‣ 4.4 Ad-Click Model with Categorical Features ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").
(Figure [S7](#A3.F7 "Figure S7 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") in Appendix [C](#A3 "Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows additional results including ECE and Brier score decomposition.)
We observe that ensembles are superior in terms of both AUC and Brier score for most of the values of shift, with the performance gap between ensembles and other methods generally increasing as the shift increases. Both Dropout model variants yielded improved AUC on shifted data, and Dropout surpassed ensembles in Brier score at shift-randomization values above 60%. SVI proved challenging to train, and the resulting model uniformly performed poorly; LL-SVI fared better but generally did not improve upon the vanilla model. *Strikingly, temperature scaling has a worse Brier score than Vanilla indicating that post-hoc calibration on the validation set actually harms calibration under dataset shift.*

## 5 Takeaways and Recommendations

We presented a large-scale evaluation of different methods for quantifying predictive uncertainty under dataset shift, across different data modalities and architectures. Our take-home messages are the following:

* •

  Along with accuracy, the quality of uncertainty consistently degrades with increasing dataset shift regardless of method.
* •

  Better calibration and accuracy on the i.i.d. test dataset does not usually translate to better calibration under dataset shift (shifted versions as well as completely different OOD data).
* •

  Post-hoc calibration (on i.i.d validation) with temperature scaling leads to well-calibrated uncertainty on the i.i.d. test set and small values of shift, but is significantly outperformed by methods that take epistemic uncertainty into account as the shift increases.
* •

  Last layer Dropout exhibits less uncertainty on shifted and OOD datasets than Dropout.
* •

  SVI is very promising on MNIST/CIFAR but it is difficult to get to work on larger datasets such as ImageNet and other architectures such as LSTMs.
* •

  The relative ordering of methods is mostly consistent (except for MNIST) across our experiments.
  The relative ordering of methods on MNIST is not reflective of their ordering on other datasets.
* •

  Deep
  ensembles
  seem to perform the best across most metrics and be more robust to dataset shift. We found that relatively small ensemble size (e.g. M=5𝑀5M=5) may be sufficient (Appendix [D](#A4 "Appendix D Effect of the number of samples on the quality of uncertainty ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")).
* •

  We also compared the set of methods on a real-world challenging genomics problem from Ren et al. ([2019](#bib.bib49)). Our observations were consistent with the other experiments in the paper. Deep ensembles performed best, but there remains significant room for improvement, as with the other experiments in the paper. See Section [F](#A6 "Appendix F OOD detection for genomic sequences ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for details.

We hope that this benchmark is useful to the community and inspires more research on uncertainty under dataset shift, which seems challenging for existing methods.
While we focused only on the quality of predictive uncertainty, applications may also need to consider computational and memory costs of the methods; Table [S1](#A1.T1 "Table S1 ‣ A.9 Computational and Memory Complexity of Different methods ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") in Appendix [A.9](#A1.SS9 "A.9 Computational and Memory Complexity of Different methods ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") discusses these costs, and the best performing methods tend to be more expensive. Reducing the computational and memory costs, while retaining the same performance under dataset shift, would also be a key research challenge.

#### Acknowledgements

We thank Alexander D’Amour, Jakub Świa̧tkowski and our reviewers for helpful feedback that improved the manuscript.

## References

* Alemi et al. (2018)

  Alemi, A. A., Fischer, I., and Dillon, J. V.
  Uncertainty in the variational information bottleneck.
  *arXiv preprint arXiv:1807.00906*, 2018.
* Amodei et al. (2016)

  Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and
  Mané, D.
  Concrete problems in AI safety.
  *arXiv preprint arXiv:1606.06565*, 2016.
* Behrmann et al. (2018)

  Behrmann, J., Duvenaud, D., and Jacobsen, J.-H.
  Invertible residual networks.
  *arXiv preprint arXiv:1811.00995*, 2018.
* Bishop (1994)

  Bishop, C. M.
  Novelty Detection and Neural Network Validation.
  *IEE Proceedings-Vision, Image and Signal processing*,
  141(4):217–222, 1994.
* Blundell et al. (2015)

  Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D.
  Weight uncertainty in neural networks.
  In *ICML*, 2015.
* Bojarski et al. (2016)

  Bojarski, M., Testa, D. D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P.,
  Jackel, L. D., Monfort, M., Muller, U., Zhang, J., Zhang, X., Zhao, J., and
  Zieba, K.
  End to end learning for self-driving cars.
  *arXiv preprint arXiv:1604.07316*, 2016.
* Brier (1950)

  Brier, G. W.
  Verification of forecasts expressed in terms of probability.
  *Monthly weather review*, 1950.
* Bröcker (2009)

  Bröcker, J.
  Reliability, sufficiency, and the decomposition of proper scores.
  *Quarterly Journal of the Royal Meteorological Society*,
  135(643):1512–1519, 2009.
* Bulatov (2011)

  Bulatov, Y.
  NotMNIST dataset, 2011.
  URL
  <http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html>.
* Chelba et al. (2013)

  Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., and
  Robinson, T.
  One billion word benchmark for measuring progress in statistical
  language modeling.
  *arXiv preprint arXiv:1312.3005*, 2013.
* DeGroot & Fienberg (1983)

  DeGroot, M. H. and Fienberg, S. E.
  The comparison and evaluation of forecasters.
  *The statistician*, 1983.
* Deng et al. (2009)

  Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L.
  ImageNet: A Large-Scale Hierarchical Image Database.
  In *Computer Vision and Pattern Recognition*, 2009.
* Esteva et al. (2017)

  Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., and
  Thrun, S.
  Dermatologist-level classification of skin cancer with deep neural
  networks.
  *Nature*, 542, 1 2017.
* Gal & Ghahramani (2016)

  Gal, Y. and Ghahramani, Z.
  Dropout as a Bayesian approximation: Representing model uncertainty
  in deep learning.
  In *ICML*, 2016.
* Geifman & El-Yaniv (2017)

  Geifman, Y. and El-Yaniv, R.
  Selective classification for deep neural networks.
  In *NeurIPS*, 2017.
* Gneiting & Raftery (2007)

  Gneiting, T. and Raftery, A. E.
  Strictly proper scoring rules, prediction, and estimation.
  *Journal of the American Statistical Association*, 102(477):359–378, 2007.
* Golovin et al. (2017)

  Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., and Sculley, D.
  Google vizier: A service for black-box optimization.
  In *Proceedings of the 23rd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, pp.  1487–1495. ACM, 2017.
* Graves (2011)

  Graves, A.
  Practical variational inference for neural networks.
  In *NeurIPS*, 2011.
* Guo et al. (2017)

  Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q.
  On calibration of modern neural networks.
  In *International Conference on Machine Learning*, 2017.
* He et al. (2016)

  He, K., Zhang, X., Ren, S., and Sun, J.
  Deep residual learning for image recognition.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pp.  770–778, 2016.
* Hendrycks & Dietterich (2019)

  Hendrycks, D. and Dietterich, T.
  Benchmarking neural network robustness to common corruptions and
  perturbations.
  In *ICLR*, 2019.
* Hendrycks & Gimpel (2017)

  Hendrycks, D. and Gimpel, K.
  A Baseline for Detecting Misclassified and Out-of-Distribution
  Examples in Neural Networks.
  In *ICLR*, 2017.
* Hensman et al. (2015)

  Hensman, J., Matthews, A., and Ghahramani, Z.
  Scalable variational gaussian process classification.
  In *International Conference on Artificial Intelligence and
  Statistics*. JMLR, 2015.
* Hernández-Lobato & Adams (2015)

  Hernández-Lobato, J. M. and Adams, R.
  Probabilistic Backpropagation for Scalable Learning of Bayesian
  Neural Networks.
  In *ICML*, 2015.
* Hochreiter & Schmidhuber (1997)

  Hochreiter, S. and Schmidhuber, J.
  Long short-term memory.
  *Neural Comput.*, 9(8):1735–1780, November
  1997.
* Kendall & Gal (2017)

  Kendall, A. and Gal, Y.
  What uncertainties do we need in Bayesian deep learning for computer
  vision?
  In *NeurIPS*, 2017.
* Kingma & Ba (2014)

  Kingma, D. and Ba, J.
  Adam: A Method for Stochastic Optimization.
  In *ICLR*, 2014.
* Kingma et al. (2014)

  Kingma, D. P., Mohamed, S., Rezende, D. J., and Welling, M.
  Semi-supervised learning with deep generative models.
  In *NeurIPS*, 2014.
* Kingma et al. (2015)

  Kingma, D. P., Salimans, T., and Welling, M.
  Variational dropout and the local reparameterization trick.
  In *NeurIPS*, 2015.
* Klambauer et al. (2017)

  Klambauer, G., Unterthiner, T., Mayr, A., and Hochreiter, S.
  Self-normalizing neural networks.
  In *NeurIPS*, 2017.
* Krizhevsky (2009)

  Krizhevsky, A.
  Learning multiple layers of features from tiny images.
  2009.
* Lakshminarayanan et al. (2017)

  Lakshminarayanan, B., Pritzel, A., and Blundell, C.
  Simple and Scalable Predictive Uncertainty Estimation Using Deep
  Ensembles.
  In *NeurIPS*, 2017.
* Lang (1995)

  Lang, K.
  Newsweeder: Learning to filter netnews.
  In *Machine Learning*. 1995.
* LeCun et al. (1998)

  LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.
  Gradient-based learning applied to document recognition.
  In *Proceedings of the IEEE*, November 1998.
* Lee et al. (2018)

  Lee, K., Lee, K., Lee, H., and Shin, J.
  A simple unified framework for detecting out-of-distribution samples
  and adversarial attacks.
  In *NeurIPS*, 2018.
* Liang et al. (2018)

  Liang, S., Li, Y., and Srikant, R.
  Enhancing the Reliability of Out-of-Distribution Image Detection in
  Neural Networks.
  *ICLR*, 2018.
* Lipton & Steinhardt (2018)

  Lipton, Z. C. and Steinhardt, J.
  Troubling trends in machine learning scholarship.
  *arXiv preprint arXiv:1807.03341*, 2018.
* Louizos & Welling (2016)

  Louizos, C. and Welling, M.
  Structured and efficient variational deep learning with matrix
  Gaussian posteriors.
  *arXiv preprint arXiv:1603.04733*, 2016.
* Louizos & Welling (2017)

  Louizos, C. and Welling, M.
  Multiplicative Normalizing Flows for Variational Bayesian Neural
  Networks.
  In *ICML*, 2017.
* MacKay (1992)

  MacKay, D. J.
  *Bayesian methods for adaptive models*.
  PhD thesis, California Institute of Technology, 1992.
* MacKay & Gibbs (1999)

  MacKay, D. J. and Gibbs, M. N.
  Density Networks.
  *Statistics and Neural Networks: Advances at the Interface*,
  1999.
* Naeini et al. (2015)

  Naeini, M. P., Cooper, G. F., and Hauskrecht, M.
  Obtaining Well Calibrated Probabilities Using Bayesian Binning.
  In *AAAI*, pp.  2901–2907, 2015.
* Nalisnick et al. (2019)

  Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., and Lakshminarayanan, B.
  Hybrid models with deep and invertible features.
  *arXiv preprint arXiv:1902.02767*, 2019.
* Netzer et al. (2011)

  Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y.
  Reading Digits in Natural Images with Unsupervised Feature
  Learning.
  In *NeurIPS Workshop on Deep Learning and Unsupervised Feature
  Learning*, 2011.
* Osband et al. (2016)

  Osband, I., Blundell, C., Pritzel, A., and Van Roy, B.
  Deep exploration via bootstrapped DQN.
  In *NeurIPS*, 2016.
* Platt (1999)

  Platt, J. C.
  Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods.
  In *Advances in Large Margin Classifiers*, pp.  61–74. MIT
  Press, 1999.
* Quinonero-Candela et al. (2006)

  Quinonero-Candela, J., Rasmussen, C. E., Sinz, F., Bousquet, O., and
  Schölkopf, B.
  Evaluating predictive uncertainty challenge.
  In *Machine Learning Challenges*. Springer, 2006.
* Rahimi & Recht (2017)

  Rahimi, A. and Recht, B.
  An addendum to alchemy, 2017.
* Ren et al. (2019)

  Ren, J., Liu, P. J., Fertig, E., Snoek, J., Poplin, R., DePristo, M. A.,
  Dillon, J. V., and Lakshminarayanan, B.
  Likelihood ratios for out-of-distribution detection.
  *arXiv preprint arXiv:1906.02845*, 2019.
* Riquelme et al. (2018)

  Riquelme, C., Tucker, G., and Snoek, J.
  Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian
  Deep Networks for Thompson Sampling.
  In *ICLR*, 2018.
* Sculley et al. (2018)

  Sculley, D., Snoek, J., Wiltschko, A., and Rahimi, A.
  Winner’s curse? On pace, progress, and empirical rigor.
  2018.
* Shafaei et al. (2018)

  Shafaei, A., Schmidt, M., and Little, J. J.
  Does Your Model Know the Digit 6 Is Not a Cat? A Less Biased
  Evaluation of “Outlier” Detectors.
  *ArXiv e-Print arXiv:1809.04729*, 2018.
* Srivastava et al. (2015)

  Srivastava, R. K., Greff, K., and Schmidhuber, J.
  Training Very Deep Networks.
  In *NeurIPS*, 2015.
* Sugiyama et al. (2009)

  Sugiyama, M., Lawrence, N. D., Schwaighofer, A., et al.
  *Dataset shift in machine learning*.
  The MIT Press, 2009.
* Welling & Teh (2011)

  Welling, M. and Teh, Y. W.
  Bayesian Learning via Stochastic Gradient Langevin Dynamics.
  In *ICML*, 2011.
* Wen et al. (2018)

  Wen, Y., Vicol, P., Ba, J., Tran, D., and Grosse, R.
  Flipout: Efficient pseudo-independent weight perturbations on
  mini-batches.
  *arXiv preprint arXiv:1803.04386*, 2018.
* Wu et al. (2019)

  Wu, A., Nowozin, S., Meeds, E., Turner, R. E., Hernandez-Lobato, J. M., and
  Gaunt, A. L.
  Deterministic Variational Inference for Robust Bayesian Neural
  Networks.
  In *ICLR*, 2019.

Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift: Appendix

## Appendix A Model Details

### A.1 MNIST

We evaluated both LeNet and a fully-connected neural network (MLP) under shift on MNIST. We observed similar trends across metrics for both models, so we report results only for LeNet in Section [1](#S4.F1 "Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"). LeNet and MLP were trained for 20 epochs using the Adam optimizer (Kingma & Ba, [2014](#bib.bib27)) and used ReLU activation functions. For stochastic methods, we averaged 300 sample predictions to yield a predictive distribution, and the ensemble model used 10 instances trained from independent random initializations.
The MLP architecture consists of two hidden layers of 200 units each with dropout applied before every dense layer. The LeNet architecture (LeCun et al., [1998](#bib.bib34)) applies two convolutional layers 3x3 kernels of 32 and 64 filters respectively) followed by two fully-connected layers with one hidden layer of 128 activations; dropout was applied before each fully-connected layer.
We employed hyperparameter tuning (See Section [A.8](#A1.SS8 "A.8 Hyperparameter Tuning ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift")) to select the training batch size, learning rate, and dropout rate.

### A.2 CIFAR-10

Our CIFAR model used the ResNet-20 V1 architecture with ReLU activations. Model parameters were trained for 200 epochs using the Adam optimizer and employed a learning rate schedule that multiplied an initial learning rate by 0.1, 0.01, 0.001, and 0.0005 at steps 80, 120, 160, and 180 respectively. Training inputs were randomly distorted using horizontal flips and random crops preceded by 4-pixel padding as described in (He et al., [2016](#bib.bib20)). For relevant methods, dropout was applied before each convolutional and dense layer (excluding the raw inputs), and stochastic methods sampled 128 predictions per sample. Hyperparameter tuning was used to select the initial learning rate, training batch size, and the dropout rate.

### A.3 ImageNet 2012

Our ImageNet model used the ResNet-50 V1 architecture with ReLU activations and was trained for 90 epochs using SGD with Nesterov momentum. The learning rate schedule linearly ramps up to a base rate in 5 epochs and scales down by a factor of 10 at each of epochs 30, 60, and 80. As with the CIFAR-10 model, stochastic methods used a sample-size of 128. Training images were distorted with random horizontal flips and random crops.

### A.4 20 Newsgroups

We use a pre-processing strategy similar to the one proposed by Hendrycks & Gimpel ([2017](#bib.bib22)) for 20 Newsgroups.
We build a vocabulary of size 30,000 words and words are indexed based on the word frequencies.
The rare words are encoded as unknown words.
We fix the length of each text input by setting a limit of 250 words, and those longer than 250 words are truncated, and those shorter than 250 words are padded with zeros.
Text in even-numbered classes are used as in-distribution inputs, and text from the odd-numbered of classes are used shifted OOD inputs.
A dataset with the same number of randomly selected text inputs from the LM1B dataset (Chelba et al., [2013](#bib.bib10)) is used as completely different OOD dataset.
The classifier is trained and evaluated only using the text from the even-numbered in-distribution classes in the training dataset. The final test results are evaluated based on in-distribution test dataset, shift OOD test dataset, and LM1B dataset.

The vanilla model uses a one-layer LSTM model of size 32 and a dense layer to predict the 10 class probabilities based on word embedding of size 128. A dropout rate of 0.1 is applied to both the LSTM layer and the dense layer for the Dropout model. The LL-SVI model replaces the last dense layer with a Bayesian layer, the ensemble model aggregates 10 vanilla models, and stochastic methods sample 5 predictions per example. The vanilla model accuracy for in-distribution test data is 0.955.

### A.5 Criteo

Each categorical feature xksubscript𝑥𝑘x\_{k} from the Criteo dataset was encoded by hashing the string token into a fixed number of buckets Nksubscript𝑁𝑘N\_{k} and either encoding the hash-bin as a one-hot vector if Nk<110subscript𝑁𝑘110N\_{k}<110 or embedding each bucket as a dksubscript𝑑𝑘d\_{k} dimensional vector otherwise. This dense feature vector, concatenated with 13 numerical features, feeds into a batch-norm layer followed by a 3-hidden-layer MLP. Each model was trained for one epoch using the Adam optimizer with a non-decaying learning rate.

Values of Nksubscript𝑁𝑘N\_{k} and dksubscript𝑑𝑘d\_{k} were tuned to maximize log-likelihood for a vanilla model, and the resulting architectural parameters were applied to all methods.
This tuning yielded hidden-layers of size 2572, 1454, and 1596, and hash-bucket counts and embedding dimensions of sizes listed below:

|  |  |  |  |
| --- | --- | --- | --- |
|  | Nk=[\displaystyle N\_{k}=[ | 1373,2148,4847,9781,396,28,3591,2798,14,7403,2511,5598,9501,  13732148484797813962835912798147403251155989501\displaystyle 1373,2148,4847,9781,396,28,3591,2798,14,7403,2511,5598,9501, |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  |  | 46,4753,4056,23,3828,5856,12,4226,23,61,3098,494,5087]\displaystyle 46,4753,4056,23,3828,5856,12,4226,23,61,3098,494,5087] |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | dk=[\displaystyle d\_{k}=[ | 3,9,29,11,17,0,14,4,0,12,19,24,29,0,13,25,0,8,29,0,22,0,0,31,0,29]\displaystyle 3,9,29,11,17,0,14,4,0,12,19,24,29,0,13,25,0,8,29,0,22,0,0,31,0,29] |  |

Learning rate, batch size, and dropout rate were further tuned for each method.
Stochastic methods used 128 prediction samples per example.

### A.6 Stochastic Variational Inference Details

For MNIST we used Flipout (Wen et al., [2018](#bib.bib56)), where we replaced each dense layer and convolutional layer with mean-field variational dense and convolutional Flipout layers respectively. Variational inference for deep ResNets (He et al., [2016](#bib.bib20)) is non-trivial, so for CIFAR we replaced a single linear layer per residual branch with a Flipout layer, removed batch normalization, added Selu non-linearities (Klambauer et al., [2017](#bib.bib30)), empirical Bayes for the prior standard deviations as in Wu et al. ([2019](#bib.bib57)) and careful tuning of the initialization via Bayesian optimization.

### A.7 Variational Gaussian Process Details

For the experiments where Gaussian Processes were compared, we used Variational Gaussian Processes to fit the model logits as in Hensman et al. ([2015](#bib.bib23)). These were then passed through a Categorical distribution and numerically integrated over using Gauss-Hermite quadrature. Each class was treated as a separate Gaussian Process, with 100 inducing points used for each class. The inducing points were initialized with model outputs on random dataset examples for CIFAR, and with Gaussian noise for MNIST. Uniform noise inducing point initialization was also tested but there was negligible difference between the three methods. All zero inducing points initializations numerically failed early on in training. Exponentiated quadratic plus linear kernels were used for all experiments. 250 samples were drawn from the logit distribution during training time to get a better estimate of the ELBO to backpropagate through. 250 logit samples were drawn at test time. 10−5∗Isuperscript105𝐼10^{-5}\*I was added to the diagonal of the covariance matrix to ensure positive definiteness.

We used 100 trials of random hyperparamter settings, selecting the configuration with the best final validation accuracy. The learning rate was tuned in [10−4,1.0]superscript1041.0[10^{-4},1.0] on a log scale; the initial kernel amplitude in [−2.0,2.0]2.02.0[-2.0,2.0]; the initial kernel length scale in [−2.0,2.0]2.02.0[-2.0,2.0]; the variational distribution covariance was initialized to s∗I𝑠𝐼s\*I where s𝑠s was tuned in [0.1,2.0]0.12.0[0.1,2.0]; 1−β11subscript𝛽11-\beta\_{1} in Adam was tuned on [10−2,0.15]superscript1020.15[10^{-2},0.15] on a log scale.

The Adam optimizer with a batch size of 512 was used, training for the same number of epochs as other methods. The same learning rate schedule was as other methods for the model and kernel parameters, but the learning rate for the variational parameters also included a 5 epoch warmup in order to help with numerical stability.

### A.8 Hyperparameter Tuning

Hyperparameters were optimized through Bayesian optimization using Google Vizier (Golovin et al., [2017](#bib.bib17)). We maximized the log-likelihood on a validation set that was held out from training (10K examples for MNIST and CIFAR-10,  125K examples for ImageNet). We optimized log-likelihood rather than accuracy since the former is a proper scoring rule.

### A.9 Computational and Memory Complexity of Different methods

In addition to performance, applications may also need to consider computational and memory costs; Table [S1](#A1.T1 "Table S1 ‣ A.9 Computational and Memory Complexity of Different methods ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") discusses them for each method.

Table S1: Computational and memory costs for evaluated methods. Notation: m𝑚m represents flops or storage for the full model, d𝑑d represents flops or storage for the last layer, k𝑘k denotes replications, z𝑧z the number of inducing points for Gaussian Processes, n𝑛n denotes number of evaluated points, and v𝑣v denotes the validation set size. Serving/training compute is identical except that v=0𝑣0v=0 for serving. Implicit in this table is a memory/compute tradeoff for sampling. Sampled weights/masks need not be stored explicitly via PRNG seed reuse; we assume the computational cost of sampling is zero.

| Method | Compute/n𝑛n | Storage |
| --- | --- | --- |
| Vanilla | m𝑚m | m𝑚m |
| Temp Scaling | m+v​m/n𝑚𝑣𝑚𝑛m+vm/n | m𝑚m |
| LL-Dropout | m+d​(k−1)𝑚𝑑𝑘1m+d(k-1) | m𝑚m |
| LL-SVI | m+d​(k−1)𝑚𝑑𝑘1m+d(k-1) | m+d𝑚𝑑m+d |
| SVI | m​k𝑚𝑘mk | 2​m2𝑚2m |
| Dropout | m​k𝑚𝑘mk | m𝑚m |
| Gaussian Process | m+z3𝑚superscript𝑧3m+z^{3} | m+z2𝑚superscript𝑧2m+z^{2} |
| Ensemble | m​k𝑚𝑘mk | m​k𝑚𝑘mk |

## Appendix B Shifted Images

We distorted MNIST images using rotations with spline filter interpolation and cyclic translations as depicted in Figure [S1](#A2.F1 "Figure S1 ‣ Appendix B Shifted Images ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

For the corrupted ImageNet dataset, we used ImageNet-C (Hendrycks & Dietterich, [2019](#bib.bib21)). Figure [S2](#A2.F2 "Figure S2 ‣ Appendix B Shifted Images ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows examples of ImageNet-C images at varying corruption intensities. Figure [S3](#A2.F3 "Figure S3 ‣ Appendix B Shifted Images ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows ImageNet-C images with the 16 corruptions analyzed in this paper, at intensity 3 (on a scale of 1 to 5).

![Refer to caption](/html/1906.02530/assets/x28.png)


(a) Rotations

![Refer to caption](/html/1906.02530/assets/x29.png)


(b) Cyclic translations

Figure S1: Examples of rotated and cyclically translated MNIST digits. Results for accuracy and calibration on rotated/translated MNIST are shown in Figure [1](#S4.F1 "Figure 1 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

![Refer to caption](/html/1906.02530/assets/x30.png)


Figure S2: Examples of ImageNet images corrupted by Gaussian blur, at intensities of 0 (uncorrupted image) through 5 (maximum corruption included in ImageNet-C).



![Refer to caption](/html/1906.02530/assets/all_corruptions_3_top2rows.jpg)

![Refer to caption](/html/1906.02530/assets/all_corruptions_3_bottom2rows.jpg)

Figure S3: Examples of 16 corruption types in ImageNet-C images, at corruption intensity 3 (on a scale from 1–5). The same corruptions were applied to CIFAR-10. Figure [2](#S4.F2 "Figure 2 ‣ 4.1 An illustrative example - MNIST ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and Section [C](#A3 "Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") show boxplots for each uncertainty method and corruption intensity, spanning all corruption types.

## Appendix C Evaluating uncertainty under distributional shift: Additional Results

Figures [S4](#A3.F4 "Figure S4 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift"), [S5](#A3.F5 "Figure S5 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") and [S7](#A3.F7 "Figure S7 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") show comprehensive results on CIFAR-10, ImageNet and Criteo respectively across various metrics including Brier score, along with the components of the Brier score : reliability (lower means better calibration) and resolution (higher values indicate better predictive quality).
Ensembles and dropout outperform all other methods across corruptions, while LL SVI shows no improvement over the baseline model. Figure [S6](#A3.F6 "Figure S6 ‣ Appendix C Evaluating uncertainty under distributional shift: Additional Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows accuracy and ECE for models with double the number of ResNet filters; the higher-capacity models are not better calibrated than their lower-capacity counterparts, suggesting that the good calibration performance of ensembles is not due simply to higher capacity.

![Refer to caption](/html/1906.02530/assets/x31.png)

![Refer to caption](/html/1906.02530/assets/x32.png)

![Refer to caption](/html/1906.02530/assets/x33.png)

![Refer to caption](/html/1906.02530/assets/x34.png)

![Refer to caption](/html/1906.02530/assets/x35.png)

Figure S4: Boxplots facilitating comparison of methods for each shift level showing detailed comparisons of various metrics under all types of corruptions on CIFAR-10. Each box shows the quartiles summarizing the results across all types of shift while the error bars indicate the min and max across different shift types.



![Refer to caption](/html/1906.02530/assets/x36.png)

![Refer to caption](/html/1906.02530/assets/x37.png)

![Refer to caption](/html/1906.02530/assets/x38.png)

![Refer to caption](/html/1906.02530/assets/x39.png)

![Refer to caption](/html/1906.02530/assets/x40.png)

![Refer to caption](/html/1906.02530/assets/x41.png)

Figure S5: Boxplots facilitating comparison of methods for each shift level showing detailed comparisons of various metrics under all types of corruptions on ImageNet. Each box shows the quartiles summarizing the results across all types of shift while the error bars indicate the min and max across different shift types.



![Refer to caption](/html/1906.02530/assets/x42.png)

![Refer to caption](/html/1906.02530/assets/x43.png)

Figure S6: Boxplots facilitating comparison of results for higher-capacity models (’Wide Vanilla’ and ’Wide Dropout’) with their lower-capacity counterparts on CIFAR. Each box shows the quartiles summarizing the results across all types of shift while the error bars indicate the min and max across different shift types.

![Refer to caption](/html/1906.02530/assets/x44.png)

![Refer to caption](/html/1906.02530/assets/x45.png)

![Refer to caption](/html/1906.02530/assets/x46.png)

![Refer to caption](/html/1906.02530/assets/x47.png)

![Refer to caption](/html/1906.02530/assets/x48.png)

Figure S7: Comprehensive comparison of metrics on Criteo models.
The Brier decomposition reveals that the majority of its degradation is due to worsening reliability, and this component alone appears to largely explain the ranking of methods in total Brier score.
Ensemble notably degrades most rapidly in resolution but persists with better reliability compared other methods for most of the data-corruption range; on ECE it remains roughly in the middle among explored methods.
Dropout (and to a lesser extend LL-Dropout) perform best on ECE and experience slower degradation in both resolution and reliability leading it to surpass ensembles at the severe range of data corruption.
Total Brier score and AUC results are discussed in detail in Section [4.4](#S4.SS4 "4.4 Ad-Click Model with Categorical Features ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift").

## Appendix D Effect of the number of samples on the quality of uncertainty

Figure [S8](#A4.F8 "Figure S8 ‣ Appendix D Effect of the number of samples on the quality of uncertainty ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows the effect of the number of sample sizes used by Dropout, SVI (and last-layer variants) on the quality of predictive uncertainty, as measured by the Brier score. Increasing the number of samples has little effect on last-layer variants, whereas increasing the number of samples improves the performance for SVI and Dropout, with diminishing returns beyond size 5.

![Refer to caption](/html/1906.02530/assets/x49.png)


(a) Dropout

![Refer to caption](/html/1906.02530/assets/x50.png)


(b) LL-Dropout

![Refer to caption](/html/1906.02530/assets/x51.png)


(c) SVI

![Refer to caption](/html/1906.02530/assets/x52.png)


(d) LL-SVI

Figure S8: Effect of Dropout and SVI sample sizes on CIFAR-10 Brier scores under increasing
Gaussian blur. See Section [4.2](#S4.SS2 "4.2 Image Models: CIFAR-10 and ImageNet ‣ 4 Experiments and Results ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for full results on CIFAR-10.

Figure [S9](#A4.F9 "Figure S9 ‣ Appendix D Effect of the number of samples on the quality of uncertainty ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows the effect of ensemble size on CIFAR-10 (top) and ImageNet (bottom). Similar to SVI and Dropout, we see that increasing the number of models in the ensemble improves performance with diminishing returns beyond size 5.
As mentioned earlier, the Brier score can be further decomposed into BS=calibration+refinement=reliability+uncertainty−resolutionBScalibrationrefinementreliabilityuncertaintyresolution\mathrm{BS}=\mathrm{calibration}+\mathrm{refinement}=\mathrm{reliability}+\mathrm{uncertainty}-\mathrm{resolution} where
reliability↓↓reliabilityabsent\mathrm{reliability}\downarrow measures calibration as the average violation of long-term true label frequencies, and refinement=uncertainty−resolutionrefinementuncertaintyresolution\mathrm{refinement}=\mathrm{uncertainty}-\mathrm{resolution}, where
uncertaintyuncertainty\mathrm{uncertainty} is the marginal uncertainty over labels (independent of predictions) and resolution↑↑resolutionabsent\mathrm{resolution}\uparrow measures the deviation of individual predictions from the marginal.

![Refer to caption](/html/1906.02530/assets/x53.png)


(a) Brier Score

![Refer to caption](/html/1906.02530/assets/x54.png)


(b) Brier Reliability

![Refer to caption](/html/1906.02530/assets/x55.png)


(c) Brier Resolution

![Refer to caption](/html/1906.02530/assets/x56.png)


(d) Brier Score

![Refer to caption](/html/1906.02530/assets/x57.png)


(e) Brier Reliability

![Refer to caption](/html/1906.02530/assets/x58.png)


(f) Brier Resolution

Figure S9: Effect of the ensemble size on CIFAR-10 (top row) and ImageNet (bottom row) Brier scores under increasing Gaussian-blur shift.
We additionally show the Brier score components: Reliability (lower means better calibration) and Resolution (higher values indicate better predictive quality).
Note that the scales for Reliability are significantly smaller than the other plots.

## Appendix E Variational Gaussian Process Results

![Refer to caption](/html/1906.02530/assets/x59.png)


(a) Brier Score

![Refer to caption](/html/1906.02530/assets/x60.png)


(b) Accuracy

![Refer to caption](/html/1906.02530/assets/x61.png)


(c) ECE

Figure S10: Uncertainty metrics across shift levels on CIFAR-10, where level 0 is the test set, using a last layer Variational Gaussian Process. See Appendix [A.7](#A1.SS7 "A.7 Variational Gaussian Process Details ‣ Appendix A Model Details ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") for experiment details.

## Appendix F OOD detection for genomic sequences

We studied the set of methods for detecting OOD genomic sequence, as a challenging realistic problem for OOD detection proposed by Ren et al. ([2019](#bib.bib49)). Classifiers are trained on 10 in-distribution bacteria classes, and tested for OOD detection of 60 OOD bacteria classes. The model architecture is the same as that in Ren et al. ([2019](#bib.bib49)): a convolutional neural networks with 1000 filters of length 20, followed by a global max pooling layer, a dense layer of 1000 units, and a last dense layer that outputs class prediction logits. For the dropout method, we add a dropout layer each after the max pooling layer and the dense layer respectively. For the LL-Dropout method, only a dropout layer after the dense layer is added. We use the dropout rate of 0.2. For the LL-SVI method, we replace the last dense layer with a stochastic variational inference dense layer. The classification accuracy for in-distribution is around 0.8 for the various types of classifiers.

Figure [S11](#A6.F11 "Figure S11 ‣ Appendix F OOD detection for genomic sequences ‣ Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift") shows the confidence vs (a) accuracy and (b) count when the test data consists of a mix of in-distribution and OOD data. Ensembles significantly outperform all other methods, and achieve better trade-off between accuracy versus confidence. Dropout performs better than Temp Scaling, and they both perform better than LL-Dropout, LL-SVI, and the Vanilla method. Note that the accuracy on examples p​(y|𝒙)≥0.9𝑝conditional𝑦𝒙0.9p(y|{\bm{x}})\geq 0.9 for the best method is still below 65%, suggesting that this realistic genomic sequences dataset is a challenging problem to benchmark future methods.

![Refer to caption](/html/1906.02530/assets/x62.png)


(a) Confidence vs Accuracy

![Refer to caption](/html/1906.02530/assets/x63.png)


(b) Confidence vs Count

Figure S11: Confidence score vs accuracy and count respectively when evaluated for in-distribution and OOD genomic sequences.

## Appendix G Tables of Metrics

The tables below report quartiles of Brier score, negative log-likelihood, and ECE for each model and dataset where quartiles are computed over all corrupted variants of the dataset.

### G.1 CIFAR-10

| Method | Vanilla | Temp. Scaling | Ensembles | Dropout | LL-Dropout | SVI | LL-SVI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brier Score (25th) | 0.243 | 0.227 | 0.165 | 0.215 | 0.259 | 0.250 | 0.246 |
| Brier Score (50th) | 0.425 | 0.392 | 0.299 | 0.349 | 0.416 | 0.363 | 0.431 |
| Brier Score (75th) | 0.747 | 0.670 | 0.572 | 0.633 | 0.728 | 0.604 | 0.732 |
| NLL (25th) | 2.356 | 1.685 | 1.543 | 1.684 | 2.275 | 1.628 | 2.352 |
| NLL (50th) | 1.120 | 0.871 | 0.653 | 0.771 | 1.086 | 0.823 | 1.158 |
| NLL (75th) | 0.578 | 0.473 | 0.342 | 0.446 | 0.626 | 0.533 | 0.591 |
| ECE (25th) | 0.057 | 0.022 | 0.031 | 0.021 | 0.069 | 0.029 | 0.058 |
| ECE (50th) | 0.127 | 0.049 | 0.037 | 0.034 | 0.136 | 0.064 | 0.135 |
| ECE (75th) | 0.288 | 0.180 | 0.110 | 0.174 | 0.292 | 0.187 | 0.275 |

### G.2 ImageNet

| Method | Vanilla | Temp. Scaling | Ensembles | Dropout | LL-Dropout | LL-SVI |
| --- | --- | --- | --- | --- | --- | --- |
| Brier Score (25th) | 0.553 | 0.551 | 0.503 | 0.577 | 0.550 | 0.590 |
| Brier Score (50th) | 0.733 | 0.726 | 0.667 | 0.754 | 0.723 | 0.766 |
| Brier Score (75th) | 0.914 | 0.899 | 0.835 | 0.922 | 0.896 | 0.938 |
| NLL (25th) | 1.859 | 1.848 | 1.621 | 1.957 | 1.830 | 2.218 |
| NLL (50th) | 2.912 | 2.837 | 2.446 | 3.046 | 2.858 | 3.504 |
| NLL (75th) | 4.305 | 4.186 | 3.661 | 4.567 | 4.208 | 5.199 |
| ECE (25th) | 0.057 | 0.031 | 0.022 | 0.017 | 0.034 | 0.065 |
| ECE (50th) | 0.102 | 0.072 | 0.032 | 0.043 | 0.071 | 0.106 |
| ECE (75th) | 0.164 | 0.129 | 0.053 | 0.109 | 0.123 | 0.148 |

### G.3 Criteo

| Method | Vanilla | Temp. Scaling | Ensembles | Dropout | LL-Dropout | SVI | LL-SVI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brier Score (25th) | 0.353 | 0.355 | 0.336 | 0.350 | 0.353 | 0.512 | 0.361 |
| Brier Score (50th) | 0.385 | 0.391 | 0.366 | 0.373 | 0.379 | 0.512 | 0.396 |
| Brier Score (75th) | 0.409 | 0.416 | 0.395 | 0.393 | 0.403 | 0.512 | 0.421 |
| NLL (25th) | 0.581 | 0.594 | 0.508 | 0.532 | 0.542 | 7.479 | 0.554 |
| NLL (50th) | 0.788 | 0.829 | 0.552 | 0.577 | 0.600 | 7.479 | 0.633 |
| NLL (75th) | 0.986 | 1.047 | 0.608 | 0.624 | 0.664 | 7.479 | 0.711 |
| ECE (25th) | 0.041 | 0.055 | 0.044 | 0.043 | 0.052 | 0.254 | 0.066 |
| ECE (50th) | 0.097 | 0.113 | 0.100 | 0.085 | 0.100 | 0.254 | 0.127 |
| ECE (75th) | 0.135 | 0.149 | 0.141 | 0.116 | 0.136 | 0.254 | 0.162 |

[◄](/html/1906.02529)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1906.02530)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1906.02530)
[View original  
on arXiv](https://arxiv.org/abs/1906.02530)[►](/html/1906.02531)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 15 22:59:54 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
