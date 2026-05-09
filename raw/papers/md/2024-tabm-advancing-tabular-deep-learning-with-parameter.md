---
arxiv: '2410.24210'
authors:
- Yury Gorishniy
- Akim Kotelnikov
- Artem Babenko
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling'
url: http://arxiv.org/abs/2410.24210v3
year: 2024
---

[2410.24210] TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling














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



# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

Yury Gorishniy
  
Yandex
&Akim Kotelnikov
  
HSE University, Yandex
&Artem Babenko
  
Yandex
The corresponding author: firstnamelastname@gmail.com

###### Abstract

Deep learning architectures for supervised learning on tabular data range from simple multilayer perceptrons (MLP) to sophisticated Transformers and retrieval-augmented methods.
This study highlights a major, yet so far overlooked opportunity for substantially improving tabular MLPs: namely, parameter-efficient ensembling
— a paradigm for implementing an ensemble of models as one model producing multiple predictions.
We start by developing TabM — a simple model based on MLP and our variations of BatchEnsemble (an existing technique).
Then, we perform a large-scale evaluation of tabular DL architectures on public benchmarks in terms of both task performance and efficiency, which renders the landscape of tabular DL in a new light.
Generally, we show that MLPs, including TabM, form a line of stronger and more practical models compared to attention- and retrieval-based architectures.
In particular, we find that TabM demonstrates the best performance among tabular DL models.
Lastly, we conduct an empirical analysis on the ensemble-like nature of TabM.
For example, we observe that the multiple predictions of TabM are weak individually, but powerful collectively.
Overall, our work brings an impactful technique to tabular DL, analyses its behaviour, and advances the performance-efficiency trade-off with TabM — a simple and powerful baseline for researchers and practitioners.
The code is available at: <https://github.com/yandex-research/tabm>.

## 1 Introduction

Supervised learning on tabular data is a ubiquitous machine learning (ML) scenario in a wide range of industrial applications.
Among classic non-deep-learning methods, the state-of-the-art solution for such tasks is gradient-boosted decision trees (GBDT) (Prokhorenkova et al., [2018](#bib.bib35); Chen & Guestrin, [2016](#bib.bib10); Ke et al., [2017](#bib.bib23)).
Deep learning (DL) models for tabular data, in turn, are reportedly improving, and the most recent works claim to perform on par or even outperform GBDT on academic benchmarks (Hollmann et al., [2023](#bib.bib18); Chen et al., [2023b](#bib.bib9); [a](#bib.bib8); Gorishniy et al., [2024](#bib.bib15)).

However, from the practical perspective, it is unclear if tabular DL offers any obvious go-to baselines beyond simple architectures in the spirit of a multilayer perceptron (MLP).
First, the scale and consistency of performance improvements of new methods w.r.t. simple MLP-like baselines are not always explicitly analyzed in the literature.
Thus, one has to infer those statistics from numerous per-dataset performance scores, which makes it hard to reason about the progress.
At the same time, due to the extreme diversity of tabular datasets, consistency is an especially valuable and hard-to-achieve property for a hypothetical go-to baseline.
Second, efficiency-related properties, such as training time, and especially inference throughput, sometimes receive less attention.
While methods are usually equally affordable on small-to-medium datasets (e.g. <<100K objects), their applicability to larger datasets remains uncertain.
Third, some recent work generally suggests that the progress on academic benchmarks may not transfer that well to real-world tasks (Rubachev et al., [2024](#bib.bib38)).
With all the above in mind, in this work, we thoroughly evaluate existing tabular DL methods and find that non-MLP models do not yet offer a convincing replacement for MLPs.

At the same time, we identify a previously overlooked path towards more powerful, reliable and reasonably efficient tabular DL models.
In a nutshell, we find that the parameter-efficient approach to deep ensembling, where most weights are shared between ensemble members, allows making simple and strong tabular models out of plain MLPs.
For example, MLP coupled with BatchEnsemble (Wen et al., [2020](#bib.bib46)) — a long-existing method — right away outperforms popular attention-based models, such as FT-Transformer (Gorishniy et al., [2021](#bib.bib13)), while being simpler and more efficient.
This result alone suggests that the parameter-efficient ensembling is a low-hanging fruit for tabular DL.

Our work builds on the above observations, and offers TabM — a new powerful and practical model for researchers and practitioners.
Drawing an informal parallel with GBDT (an ensemble of decision trees), TabM can also be viewed as a simple base model (MLP) combined with an ensembling-like technique, providing high performance and simple implementation at the same time.

Main contributions.
We summarize our main contributions as follows:

1. 1.

   We present TabM — a simple DL architecture for supervised learning on tabular data.
   TabM is based on MLP and parameter-efficient ensembling techniques closely related to BatchEnsemble (Wen et al., [2020](#bib.bib46)).
   In particular, TabM produces Multiple predictions per object.
   TabM easily competes with GBDT and outperforms prior tabular DL models, while being more efficient than attention- and retrieval-based DL architectures.
2. 2.

   We provide a fresh perspective on tabular DL models in a large-scale evaluation along four dimensions: performance ranks, performance score distributions, training time and inference throughput.
   One of our findings is that MLPs, including TabM, hit an appealing performance-efficiency tradeoff, which is not the case for attention- and retrieval-based models.
3. 3.

   Empirically, we show that the multiple predictions of TabM are weak and overfitted individually, while their average is strong and generalizable.
   The training gradients of TabM, in turn, can be viewed as an “ensemble” of diverse gradients coming from the multiple predictions.

## 2 Related work

Decision-tree-based models.
Gradient-boosted decision trees (GBDT) (Chen & Guestrin, [2016](#bib.bib10); Ke et al., [2017](#bib.bib23); Prokhorenkova et al., [2018](#bib.bib35)) is a strong and efficient baseline for tabular tasks.
GBDT is a classic machine learning model, specifically, an ensemble of decision trees.
Our model TabM is a deep learning model, specifically, a parameter-efficient ensemble of MLPs.

Tabular deep learning architectures.
A large number of deep learning architectures for tabular data has been proposed over the recent years.
That includes attention-based architectures (Song et al., [2019](#bib.bib40); Gorishniy et al., [2021](#bib.bib13); Somepalli et al., [2021](#bib.bib39); Kossen et al., [2021](#bib.bib26); Yan et al., [2023](#bib.bib47)), retrieval-augmented architectures (Somepalli et al., [2021](#bib.bib39); Kossen et al., [2021](#bib.bib26); Gorishniy et al., [2024](#bib.bib15); Ye et al., [2024](#bib.bib48)), MLP-like models (Gorishniy et al., [2021](#bib.bib13); Klambauer et al., [2017](#bib.bib25); Wang et al., [2020](#bib.bib45)) and others (Arik & Pfister, [2020](#bib.bib5); Popov et al., [2020](#bib.bib34); Chen et al., [2023b](#bib.bib9); Marton et al., [2024](#bib.bib31); Hollmann et al., [2023](#bib.bib18)).
Compared to prior work, the key difference of our model TabM is its computation flow, where one TabM imitates an ensemble of MLPs by producing multiple independently trained predictions.
Prior attempts to bring ensemble-like elements to tabular DL (Badirli et al., [2020](#bib.bib6); Popov et al., [2020](#bib.bib34)) were not found promising (Gorishniy et al., [2021](#bib.bib13)).
Also, being a simple feed-forward MLP-based model, TabM is significantly more efficient than some of the prior work.
Compared to attention-based models, TabM does not suffer from quadratic computational complexity w.r.t. the dataset dimensions.
Compared to retrieval-based models, TabM is easily applicable to large datasets.

Improving tabular MLP-like models.
Multiple recent studies achieved competitive performance with MLP-like architectures on tabular tasks by applying architectural modifications (Gorishniy et al., [2022](#bib.bib14)), regularizations (Kadra et al., [2021](#bib.bib22); Jeffares et al., [2023a](#bib.bib20); Holzmüller et al., [2024](#bib.bib19)), custom training techniques (Bahri et al., [2021](#bib.bib7); Rubachev et al., [2022](#bib.bib37)).
Thus, it seems that tabular MLPs have good potential, but one has to deal with overfitting and optimization issues to reveal that potential.
Our model TabM achieves high performance with MLP in a different way, namely, by using it as the base backbone in a parameter-efficient ensemble in the spirit of BatchEsnsemble (Wen et al., [2020](#bib.bib46)).
Our approach is orthogonal to the aforementioned training techniques and architectural advances.

Deep ensembles.
In this paper, by a deep ensemble, we imply multiple DL models of the same architecture trained independently (Jeffares et al., [2023b](#bib.bib21)) for the same task under different random seeds (i.e. with different initializations, training batch sequences, etc.).
The prediction of a deep ensemble is the mean prediction of its members.
Deep ensembles often significantly outperform single DL models of the same architecture (Fort et al., [2020](#bib.bib11)), and can excel in other tasks like uncertainty estimation or out-of-distribution detection (Lakshminarayanan et al., [2017](#bib.bib27)).
It was observed that individual members of deep ensembles can learn to extract diverse information from the input, and the power of deep ensembles depends on this diversity (Allen-Zhu & Li, [2023](#bib.bib2)).
The main drawback of deep ensembles is the cost and inconvenience of training and using multiple models.

Parameter-efficient deep “ensembles”.
To achieve the performance of deep ensembles at a lower cost, multiple studies proposed architectures that imitate ensembles by producing multiple predictions with one model (Lee et al., [2015](#bib.bib29); Zhang et al., [2020](#bib.bib49); Wen et al., [2020](#bib.bib46); Havasi et al., [2021](#bib.bib17); Antorán et al., [2020](#bib.bib4); Turkoglu et al., [2022](#bib.bib43)).
Such models can be viewed as “ensembles” where the implicit ensemble members share a large amount of their weights.
There are also non-architectural approaches to efficient ensembling, e.g. FGE (Garipov et al., [2018](#bib.bib12)), but we do not explore them, because we are interested specifically in architectural techniques.
In this paper, we highlight parameter-efficient ensembling as an impactful paradigm for tabular DL.
In particular, we describe two simple variations of BatchEnsemble (Wen et al., [2020](#bib.bib46)) that are highly effective for tabular MLPs.
One variation uses a more efficient parametrization, and another one uses an improved initialization.

## 3 TabM

In this section, we present TabM — a Tabular model that makes Multiple predictions.

### 3.1 Preliminaries

Notation.
We consider classification and regression tasks on tabular data.
x𝑥x and y𝑦y denote the features and a label, respectively, of one object from a given dataset.
A machine learning model takes x𝑥x as input and produces y^^𝑦\hat{y} as a prediction of y𝑦y.
N∈ℕ𝑁ℕN\in\mathbb{N} and d∈ℕ𝑑ℕd\in\mathbb{N} respectively denote the “depth” (e.g. the number of blocks) and “width” (e.g. the size of the latent representation) of a given neural network.
dy∈ℕsubscript𝑑𝑦ℕd\_{y}\in\mathbb{N} is the output representation size (e.g. dy=1subscript𝑑𝑦1d\_{y}=1 for regression tasks, and dysubscript𝑑𝑦d\_{y} equals the number of classes for classification tasks).

Datasets.
Our benchmark consists of 46 publicly available datasets used in prior work, including Grinsztajn et al. ([2022](#bib.bib16)); Gorishniy et al. ([2024](#bib.bib15)); Rubachev et al. ([2024](#bib.bib38)).
The main properties of our benchmark are summarized in [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and more details are provided in [Appendix C](#A3 "Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Table 1: 
The overview of our benchmark.
The “Split type” property is explained in the text.

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #Datasets | Train size | | | | #Features | | | | Task type | | Split type | |
|  | Min. | Q50 | Mean | Max. | Min. | Q50 | Mean | Max. | #Regr. | #Classif. | Random | Domain-aware |
| 46 | 1.8K | 12K | 76K | 723K | 3 | 20 | 108 | 986 | 28 | 18 | 37 | 9 |

Domain-aware splits.
We pay extra attention to datasets with what we call “domain-aware” splits, including the eight datasets from Rubachev et al. ([2024](#bib.bib38)) and the Microsoft dataset (Qin & Liu, [2013](#bib.bib36)).
For these datasets, their original real-world splits are available, for example, time-aware splits as in Rubachev et al. ([2024](#bib.bib38)).
Such datasets were shown to be challenging for some methods, because they naturally exhibit a certain degree of distribution shift between training and test parts (Rubachev et al., [2024](#bib.bib38)).
The random splits of the remaining 37 datasets are inherited from prior work.

Experiment setup.
We use the setup from Gorishniy et al. ([2024](#bib.bib15)), and describe it in detail in [subsection D.2](#A4.SS2 "D.2 Experiment setup ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Most importantly, on each dataset, a given model undergoes hyperparameter tuning on the validation set, then the tuned model is trained from scratch under multiple random seeds, and the test metric averaged over the random seeds becomes the final score of the model on the dataset.

Metrics.
We use RMSE (the root mean square error) for regression tasks, and accuracy or ROC-AUC for classification tasks depending on the dataset source.
See [subsection D.3](#A4.SS3 "D.3 Metrics ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") for details.

Also, throughout the paper, we often use the relative performance of models w.r.t. MLP as the key metric.
This metric gives a unified perspective on all tasks and allows reasoning about the scale of improvements w.r.t. to a simple baseline (MLP).
Formally, on a given dataset, the metric is defined as (scorebaseline−1)⋅100%⋅scorebaseline1percent100\left(\frac{\text{score}}{\text{baseline}}-1\right)\cdot 100\%, where “score” is the metric of a given model, and “baseline” is the metric of MLP.
In this computation, for regression tasks, we convert the raw metrics from RMSE to R2superscript𝑅2R^{2} to better align the scales of classification and regression metrics.

### 3.2 A quick introduction to BatchEnsemble.

For a given architecture, let’s consider any linear layer l𝑙l in it: l​(x)=W​x+b𝑙𝑥𝑊𝑥𝑏l(x)=Wx+b, where x∈ℝd1𝑥superscriptℝsubscript𝑑1x\in\mathbb{R}^{d\_{1}}, W∈ℝd2×d1𝑊superscriptℝsubscript𝑑2subscript𝑑1W\in\mathbb{R}^{d\_{2}\times d\_{1}}, b∈ℝd2𝑏superscriptℝsubscript𝑑2b\in\mathbb{R}^{d\_{2}}.
To simplify the notation, let d1=d2=dsubscript𝑑1subscript𝑑2𝑑d\_{1}=d\_{2}=d.
In a traditional deep ensemble, the i𝑖i-th member has its own set of weights Wi,bi

subscript𝑊𝑖subscript𝑏𝑖W\_{i},b\_{i} for this linear layer: li​(xi)=Wi​xi+bisubscript𝑙𝑖subscript𝑥𝑖subscript𝑊𝑖subscript𝑥𝑖subscript𝑏𝑖l\_{i}(x\_{i})=W\_{i}x\_{i}+b\_{i}, where xisubscript𝑥𝑖x\_{i} is the object representation within the i𝑖i-th member.
By contrast, in BatchEnsemble, this linear layer is either (1) fully shared between all members, or (2) mostly shared: li​(xi)=si⊙(W​(ri⊙xi))+bisubscript𝑙𝑖subscript𝑥𝑖direct-productsubscript𝑠𝑖𝑊direct-productsubscript𝑟𝑖subscript𝑥𝑖subscript𝑏𝑖l\_{i}(x\_{i})=s\_{i}\odot(W(r\_{i}\odot x\_{i}))+b\_{i}, where ⊙direct-product\odot is an elementwise multiplication, W∈ℝd×d𝑊superscriptℝ𝑑𝑑W\in\mathbb{R}^{d\times d} is shared between all members, and ri,si,bi∈ℝd

subscript𝑟𝑖subscript𝑠𝑖subscript𝑏𝑖
superscriptℝ𝑑r\_{i},s\_{i},b\_{i}\in\mathbb{R}^{d} are not shared between the members.
This is equivalent to defining the i𝑖i-th weight matrix as Wi=W⊙(ri​siT)subscript𝑊𝑖direct-product𝑊subscript𝑟𝑖superscriptsubscript𝑠𝑖𝑇W\_{i}=W\odot(r\_{i}s\_{i}^{T}).
To ensure diversity of the ensemble members, risubscript𝑟𝑖r\_{i} and sisubscript𝑠𝑖s\_{i} of all members are initialized randomly with ±1plus-or-minus1\pm 1.
All other layers are fully shared between the members of BatchEnsemble.

The described parametrization allows packing all ensemble members in one model that simultaneously takes k𝑘k copies of the object as input, and applies all k𝑘k implicit members in parallel, without explicitly materializing each member.
This is achieved by replacing one or more linear layers of the original neural network with their BatchEnsemble versions:
lBE​(X)=((X⊙R)​W)⊙S+Bsubscript𝑙BE𝑋direct-productdirect-product𝑋𝑅𝑊𝑆𝐵l\_{\text{BE}}(X)=((X\odot R)W)\odot S+B,
where X∈ℝk×d𝑋superscriptℝ𝑘𝑑X\in\mathbb{R}^{k\times d} stores k𝑘k representations of the same input object (one per member), and R,S,B∈ℝd

𝑅𝑆𝐵
superscriptℝ𝑑R,S,B\in\mathbb{R}^{d} store the non-shared weights (risubscript𝑟𝑖r\_{i}, sisubscript𝑠𝑖s\_{i}, bisubscript𝑏𝑖b\_{i}) of the submodels, as shown at the lower left part of [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Overhead to the model size.
With BatchEnsemble, adding a new ensemble member means adding only one row to each of the matrices R𝑅R, S𝑆S and B𝐵B, which results in 3​d3𝑑3d new parameters per layer.
For typical values of d𝑑d, this is a negligible overhead to the original layer size d2+dsuperscript𝑑2𝑑d^{2}+d.
  
Overhead to the runtime.
Thanks to the modern hardware, the large number of shared weights and the parallel execution of the k𝑘k forward passes, the runtime overhead of BatchEnsemble can be (significantly) lower than ×kabsent𝑘\times k (Wen et al., [2020](#bib.bib46)).
Intuitively, if the original workload underutilizes the hardware, there are more chances to pay less than ×kabsent𝑘\times k overhead.

Terminology.
In this paper, we call risubscript𝑟𝑖r\_{i}, sisubscript𝑠𝑖s\_{i}, bisubscript𝑏𝑖b\_{i}, R𝑅R, S𝑆S and B𝐵B adapters, and the implicit members of parameter-efficient emsembles (e.g. BatchEnsemble) — implicit submodels or simply submodels.

### 3.3 TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} & TabM

Our models TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} and TabM are based on a multilayer perceptron (MLP) and parameter-efficient ensembling methods, with a strong connection to BatchEnsemble (Wen et al., [2020](#bib.bib46)), introduced in [subsection 3.2](#S3.SS2 "3.2 A quick introduction to BatchEnsemble. ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
In [subsection A.1](#A1.SS1 "A.1 Motivation ‣ Appendix A Additional discussion on TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we explain that we use specifically BatchEnsemble as the baseline efficient ensembling method because of its good balance between performance and ease of use, while using MLP as the base model is crucial because of its excellent efficiency.
We obtain our models in several steps, starting from essential baselines.
We always use the ensemble size k=32𝑘32k=32 and analyze this hyperparameter in [subsection 5.3](#S5.SS3 "5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

MLP.
We define MLP as a sequence of N𝑁N simple blocks followed by a linear prediction head:
  
MLP(x)=Linear(BlockN(…(Block1(x)))\text{MLP}(x)=\text{Linear}(\text{Block}\_{N}(\ldots(\text{Block}\_{1}(x))), where Blocki(x)=Dropout(ReLU(Linear((x)))\text{Block}\_{i}(x)=\text{Dropout}(\text{ReLU}(\text{Linear}((x))).

MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} = MLP + Deep Ensemble.
We denote the traditional deep ensemble of k𝑘k independently trained MLPs as MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k}.
This method is illustrated in [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and its performance is reported in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") (the hyperparameter tuning is performed for one MLP, after which the tuned MLP is ensembled).
Interestingly, the results are already better and more stable than those of FT-Transformer (Gorishniy et al., [2021](#bib.bib13)) — the popular attention-based baseline.
And, given the significantly better efficiency of MLPs (as will be shown later), MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} may actually be no less practical than FT-Transformer, especially with additional techniques like Packed-Ensembles (Laurent et al., [2023](#bib.bib28)).
That said, we continue towards more efficient approaches.

TabMnaive = MLP + BatchEnsemble.
Now, instead of the deep ensemble, we naively apply BatchEnsemble to the backbone of MLP, while keeping the prediction heads separate.
This gives us TabMnaive — a preliminary suboptimal version of TabM.
In fact, the architecture (but not the initialization) of TabMnaive is already equivalent to that of TabM, so [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") is applicable.
The performance of TabMnaive shown in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") is important for two reasons.
First, TabMnaive — the efficient version of MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} — is noticeably better than MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} itself, which is intriguing.
We are not aware of similar results for BatchEnsemble in general, and share some thoughts on this phenomenon in [subsection A.2](#A1.SS2 "A.2 Why TabM outperforms a full-fledged deep ensemble? ‣ Appendix A Additional discussion on TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Second, TabMnaive right away outperforms FT-Transformer, which demonstrates the great potential of parameter-efficient ensembling for MLPs.
This motivates further exploration.

![Refer to caption](/html/2410.24210/assets/x1.png)


Figure 1: 
(Upper left)
A template for implementing an ensemble of k𝑘k MLPs.
The remaining parts of the figure are three different parametrizations of the k𝑘k MLP backbones, all described in [subsection 3.3](#S3.SS3 "3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
In all cases, each of the k𝑘k MLP backbones independently processes its own copy of the input object.
(Upper right)
MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} is a traditional deep ensemble of k𝑘k fully independent MLPs.
(Lower left)
TabM is obtained by injecting three non-shared adapters R𝑅R, S𝑆S, B𝐵B in each of the N𝑁N linear layers of one MLP (∗ the initialization differs from Wen et al. ([2020](#bib.bib46))).
(Lower right)
TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} is obtained by keeping only the very first adapter R𝑅R of TabM  and removing the remaining 3​N−13𝑁13N-1 adapters.
Thus, TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} applies the same shared MLP to k𝑘k object representations, with only two non-shared elements ensuring diversity of predictions: the randomly initialized multiplicative adapter R𝑅R and the k𝑘k prediction heads.
(Details)
Input transformations such as one-hot-encoding, feature embeddings (Gorishniy et al., [2022](#bib.bib14)) and others are omitted for simplicity.
In practice, they are applied (and the result is flattened) before the Clone module.
Drop denotes dropout (Srivastava et al., [2014](#bib.bib41)).

![Refer to caption](/html/2410.24210/assets/x2.png)


Figure 2: 
The performance of models described in [subsection 3.3](#S3.SS3 "3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") on 46 datasets from [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"); plus several baselines on the left.
For a given model, one dot on a jitter plot describes the performance score on one of the 46 datasets.
The box plots describe the percentiles of the jitter plots: the boxes describe the 25th, 50th and 75th percentiles, and the whiskers describe the 10th and 90th percentiles.
Outliers are clipped.
The numbers at the bottom are the mean and standard deviations over the jitter plots.
For each model, hyperparameters are tuned.
“Model×ksuperscriptModelabsent𝑘\text{Model}^{\times k}” denotes an ensemble of k𝑘k models.

TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} = MLP + Minimal Ensemble.
By construction, the just discussed TabMnaive (illustrated as “TabM” in [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) has 3​N3𝑁3N adapters: R𝑅R, S𝑆S and B𝐵B in each of the N𝑁N blocks.
Among the 3​N3𝑁3N adapters, the first adapter R𝑅R in the very first linear layer is responsible for transforming the k𝑘k equal copies of the input (packed as X𝑋X in [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) to k𝑘k different representations before the tabular features are mixed with @​W@𝑊@W for the first time.
A simple experiment reveals that this adapter is critical.
First, we remove it from TabMnaive and keep the remaining 3​N−13𝑁13N-1 adapters untouched, which gives us TabMbad with worse performance, as shown in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Then, we do the opposite: we keep only the very first adapter of TabMnaive and remove the remaining 3​N−13𝑁13N-1 adapters, which gives us TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} — the minimal version of TabM.
TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} is illustrated in [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), where we informally call the described approach as “Minimal Ensemble”.
Perhaps, surprisingly, but [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows that TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} performs better than TabMnaive, despite having only one adapter instead of 3​N3𝑁3N adapters.

TabM = MLP + BatchEnsemble + Better initialization.
The just obtained results motivate the next step.
We go back to the architecture of TabMnaive with all 3​N3𝑁3N adapters, but initialize all multiplicative adapters R𝑅R and S𝑆S, except for the very first one, deterministically with 111.
As such, at initialization, the deterministically initialized adapters have no effect, and the model behaves like TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}, but these adapters are free to add more expressivity during training.
This gives us TabM, illustrated in [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
[Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows that TabM is the best variation so far.

TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} & TabM†superscriptTabM†\mbox{\text{TabM}}^{\dagger}.
Non-linear feature embeddings (Gorishniy et al., [2022](#bib.bib14)) are known to boost the performance of many tabular models, especially of MLPs.
We denote TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} and TabM with non-linear feature embeddings as TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} and TabM†superscriptTabM†\mbox{\text{TabM}}^{\dagger}, respectively.
By default, we recommend using the piecewise-linear embeddings (Gorishniy et al., [2022](#bib.bib14)).
In [subsection A.3](#A1.SS3 "A.3 TabM with feature embeddings ‣ Appendix A Additional discussion on TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we provide additional implementation details, such as slightly different initialization.
[Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows that, TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} is competitive with TabM†superscriptTabM†\mbox{\text{TabM}}^{\dagger}, so we will use TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} for simplicity.

Intuition.
To give additional intuition on TabM, we make the following observations:

* •

  Setting k=1𝑘1k=1 makes TabM identical to one plain MLP.
* •

  Increasing k𝑘k by one adds a negligible number of new parameters to TabM.
* •

  TabM, viewed as a single model, can benefit from the deep ensembling, see TabMmini†⁣×5superscriptsubscriptTabMmini
  †absent5\mbox{\text{TabM}}\_{\text{mini}}^{\dagger\times 5} in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
* •

  In Transformer-like (Vaswani et al., [2017](#bib.bib44)) and Mixer-like (Tolstikhin et al., [2021](#bib.bib42)) models:
    
  (a) the latent representation shape is m×d𝑚𝑑m\times d, where m𝑚m is the number of tabular features, and d𝑑d is the embedding size; (b) the m𝑚m embeddings are mixed with each other in attention or linear layers, and (c) per-embedding transformations (the FFN layers) are the same for all embeddings.
    
  By contrast, in TabM: (a) the shape is only k×d𝑘𝑑k\times d, (b) the k𝑘k embeddings never interact with each other, and (c) per-embedding transformations contain embedding-specific weights (adapters).

Hyperparameters.
Compared to MLP, the only new hyperparameter of TabM is k𝑘k — the number of implicit submodels.
We heuristically set k=32𝑘32k=32 and do not tune this value.
We analyze the influence of k𝑘k in [subsection 5.3](#S5.SS3 "5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
We also noticed that the average optimal learning rate for TabM is higher than for MLP, which is explained in [subsection A.4](#A1.SS4 "A.4 Hyperparameters ‣ Appendix A Additional discussion on TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Limitations and practical considerations are commented in [subsection A.5](#A1.SS5 "A.5 Limitations and practical considerations ‣ Appendix A Additional discussion on TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Next steps.
The performance of TabM in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") renders it as a highly promising model.
This motivates a full-fledged empirical comparison against prior tabular models ([section 4](#S4 "4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) and detailed analysis of TabM’s behaviour ([section 5](#S5 "5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")).

## 4 Evaluating tabular deep learning architectures

Now, we perform an empirical comparison of many tabular models, including TabM introduced in [section 3](#S3 "3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
The implementation details of the models are provided in [Appendix D](#A4 "Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

### 4.1 Baselines

In the main text, we use the following baselines:
MLP (the classic multilayer perceptron), FT-Transformer denoted as “FT-T” (the attention-based model from Gorishniy et al. ([2021](#bib.bib13))), SAINT (the attention- and retrieval- based model from Somepalli et al. ([2021](#bib.bib39))), T2G-Former denoted as “T2G” (the attention-based model from Yan et al. ([2023](#bib.bib47))), ExcelFormer denoted as “Excel” (the attention-based model from Chen et al. ([2023a](#bib.bib8))), TabR (the retrieval-based model from Gorishniy et al. ([2024](#bib.bib15))), ModernNCA denoted as “MNCA” (the retrieval-based model from Ye et al. ([2024](#bib.bib48))) and three GBDT implementations: XGBoost (Chen & Guestrin, [2016](#bib.bib10)), LightGBM (Ke et al., [2017](#bib.bib23)) and CatBoost (Prokhorenkova et al., [2018](#bib.bib35)).
MLP†, TabR† and MNCA† denote the corresponding models with non-linear feature embeddings (Gorishniy et al., [2022](#bib.bib14)).
In fact, some other baselines, such as Excel (Chen et al., [2023a](#bib.bib8))), already use custom non-linear feature embeddings.

We provide results for more baselines in [Appendix B](#A2 "Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

![Refer to caption](/html/2410.24210/assets/x3.png)


Figure 3: 
The task performance of tabular models on the 46 datasets from [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
(Left)
The mean and standard deviations of the performance ranks over all datasets summarize the head-to-head comparison between the models on all datasets.
(Middle & Right) The relative performance w.r.t. the plain multilayer perceptron (MLP) allows reasoning about the scale and consistency of improvements over this simple baseline.
One dot of a jitter plot corresponds to the performance of a model on one of the 46 datasets.
The box plots visualize the 10th, 25th, 50th, 75th and 90th percentiles of the jitter plots.
Outliers are clipped.
The separation in random and domain-aware dataset splits is explained in [subsection 3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").



![Refer to caption](/html/2410.24210/assets/x4.png)

![Refer to caption](/html/2410.24210/assets/x5.png)

Figure 4: 
Training times (left) and inference throughput (right) of the models from [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
One dot represents a measurement on one dataset.




Table 2: 
RMSE (upper rows) and training times (lower rows) on two large datasets.
The best values are in bold.
The meaning of model colors follows [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

|  | #Objects | #Features | XGBoostXGBoost\mathrm{XGBoost} | MLPMLP\mathrm{MLP} | TabMmini†∗superscriptsubscriptTabMmini†absent\mbox{\text{TabM}}\_{\text{mini}}^{\dagger\*} | TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} | FT​-​TFT-T\mathrm{FT}\text{-}\mathrm{T} | TabRTabR\mathrm{TabR} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maps Routing | 6.56.56.5M | 986986986 | 0.16010.16010.1601 | 0.15920.15920.1592 | 0.15830.15830.1583 | 0.15820.1582\mathbf{0.1582} | 0.15940.15940.1594 | OOM |
| 282828m | 𝟏𝟓15\mathbf{15}m | 222h | 13.513.513.5h | 45.545.545.5h |
| Weather | 131313M | 103103103 | 1.42341.42341.4234 | 1.48421.48421.4842 | 1.4090 | 1.4112 | 1.44091.44091.4409 | OOM |
| 𝟏𝟎10\mathbf{10}m | 151515m | 1.31.31.3h | 3.33.33.3h | 13.513.513.5h |

### 4.2 Task performance

We evaluate all models following the protocol announced in [subsection 3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and report the results in [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") (see also the critical difference diagram in [Figure 9](#A2.F9 "Figure 9 ‣ B.2 Task performance ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")).
We make the following observations:

1. 1.

   The performance ranks render TabM as the top-tier DL model.
2. 2.

   The middle and right parts of [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") provide a fresh perspective on the per-dataset metrics.
   TabM holds its leadership among the DL models.
   Meanwhile, many DL methods turn out to be no better or even worse than MLP on a non-negligible number of datasets, which shows them as less reliable solutions, and changes the ranking, especially on the domain-aware splits (right).
3. 3.

   One important characteristic of a model is the weakest part of its performance profile (e.g. the 10th or 25th percentiles in the middle plot), since it shows how reliable the model is on “inconvenient” datasets.
   From that perspective, MLP† seems to be a decent practical option between the plain MLP and TabM, especially given its simplicity and efficiency compared to retrieval-based alternatives, such as TabR and ModernNCA.

Summary. TabM confidently demonstrates the best performance among tabular DL models, and can serve as a reliable go-to DL baseline.
This is not the case for attention- and retrieval- based models.
Overall, MLP-like models, including TabM, form a representative set of tabular DL baselines.

### 4.3 Efficiency

Now, we evaluate tabular models in terms of training and inference efficiency, which becomes a serious reality check for some of the methods.
We benchmark exactly those hyperparameter configurations of models that are presented in [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") (see [subsection B.3](#A2.SS3 "B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") for the motivation).

TabMmini†∗superscriptsubscriptTabMmini†absent\mbox{\text{TabM}}\_{\text{mini}}^{\dagger\*}.
Additionally, we include TabMmini†∗superscriptsubscriptTabMmini†absent\mbox{\text{TabM}}\_{\text{mini}}^{\dagger\*}, which is TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} enhanced with two efficiency-related plugins available out-of-the-box in PyTorch (Paszke et al., [2019](#bib.bib32)): the automatic mixed precision (AMP) and torch.compile (Ansel et al., [2024](#bib.bib3)).
The purpose of TabMmini†∗superscriptsubscriptTabMmini†absent\mbox{\text{TabM}}\_{\text{mini}}^{\dagger\*} is to showcase the potential of the modern hardware and software for a powerful tabular DL model, and it should not be directly compared to other DL models.
However, the implementation simplicity of TabM plays an important role, because it facilitates the seamless integration of the aforementioned PyTorch plugins.

Training time.
We focus on training times on larger datasets, because on small datasets, all methods become almost equally affordable, regardless of the formal relative difference.
Nevertheless, in [Figure 10](#A2.F10 "Figure 10 ‣ B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we provide measurements on small datasets as well.
The left side of [Figure 4](#S4.F4 "Figure 4 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") reveals that TabM offers practical training times.
By contrast, the long training times of attention- and retrieval-based models become one more limitation of these methods.

Inference throughput.
The right side of [Figure 4](#S4.F4 "Figure 4 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") tells essentially the same story as the left side.
In [subsection B.3](#A2.SS3 "B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we also report the inference throughput on GPU with large batch sizes.

Applicability to large datasets.
In [Table 2](#S4.T2 "Table 2 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we report metrics on two large datasets.
As expected, attention- and retrieval-based models struggle, yielding extremely long training times, or being simply inapplicable without additional effort.
See [subsection D.4](#A4.SS4 "D.4 Implementation details of subsection 4.3 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") for implementation details.

Parameter count.
Most tabular networks are overall compact.
This, in particular, applies to TabM, because its size is by design comparable to MLP.
We report model sizes in [subsection B.3](#A2.SS3 "B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Summary.
Simple MLPs are the fastest DL models, with TabM being the runner-up.
The attention- and retrieval-based models are significantly slower.
Overall, MLP-like models, including TabM, form a representative set of practical and accessible tabular DL baselines.

## 5 Analysis

### 5.1 Performance and training dynamics of the individual submodels

![Refer to caption](/html/2410.24210/assets/x6.png)

![Refer to caption](/html/2410.24210/assets/x7.png)

![Refer to caption](/html/2410.24210/assets/x8.png)

Figure 5: 
The training profiles of TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32} and TabMminik=1superscriptsubscriptTabMmini𝑘1\mbox{\text{TabM}}\_{\text{mini}}^{k=1} as described in [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
(Upper) The training curves. k=32​[i]𝑘32delimited-[]𝑖k=32[i] represents the mean individual loss over the 323232 submodels.
(Middle) Same as the first row, but in the train-test coordinates: each dot represents some epoch from the first row, the training generally
goes from left to right.
This allows reasoning about overfitting by comparing test loss values for a given train loss value.
(Lower) The mean pairwise cosine similarity between the k𝑘k individual gradients of TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32} with the default initialization (green) and two suboptimal initializations described in [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Formally: 2n⋅k​(k−1)​∑l,i,j​(i<j)cos⁡(gil,gjl)2⋅𝑛𝑘𝑘1subscript

𝑙𝑖𝑗𝑖𝑗superscriptsubscript𝑔𝑖𝑙superscriptsubscript𝑔𝑗𝑙\frac{2}{n\cdot k(k-1)}\sum\_{l,i,j(i<j)}\cos{(g\_{i}^{l},g\_{j}^{l})}, where gilsuperscriptsubscript𝑔𝑖𝑙g\_{i}^{l} is the gradient of the i𝑖i-th submodel induced by the l𝑙l-th of the n=1000𝑛1000n=1000 training objects.
See [subsection D.5](#A4.SS5 "D.5 Implementation details of subsection 5.1 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") for details.
The legends contain the test scores if early stopping was used.

Recall that the prediction of TabM is defined as the mean prediction of its k𝑘k implicit submodels.
These submodels share almost all of their weights, and are trained simultaneously.
In this section, we take a closer look at the individual performance and training dynamics of these submodels.

For the next experiment, we intentionally simplify the setup as described in detail in [subsection D.5](#A4.SS5 "D.5 Implementation details of subsection 5.1 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Most importantly, all models have the same depth 333 and width 512512512, and are trained without early stopping, i.e. the training goes beyond the optimal epochs.
We use TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} from [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") with k=32𝑘32k=32 denoted as TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32}.
We use TabMminik=1superscriptsubscriptTabMmini𝑘1\mbox{\text{TabM}}\_{\text{mini}}^{k=1} (i.e. essentially one plain MLP) as a natural baseline for the submodels of TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32}, because each of the 323232 submodels has the architecture of TabMminik=1superscriptsubscriptTabMmini𝑘1\mbox{\text{TabM}}\_{\text{mini}}^{k=1}.

We visualize the training profiles on four diverse datasets (two classification and two regression problems of different sizes) in [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
As a reminder, the mean of the k𝑘k individual losses is what is explicitly optimized during the training of TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}, and the loss of the collective mean prediction corresponds to what is used by TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} on inference (see [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")).

In the upper row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), the  collective mean prediction of the submodels is superior to their  individual predictions in terms of both training and test losses.
After the initial epochs, the training loss of the  baseline MLP is lower than that of the collective and individual predictions.

In the middle row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we see a stark contrast between the individual and collective performance of the submodels.
Compared to the baseline MLP, the submodels look overfitted individually, while their collective prediction exhibits substantially better generalization.
This result is strict evidence of a non-trivial diversity of the submodels: without that, their collective test performance would be similar to their individual test performance.
Additionally, we report the performance of the Best submodel of TabM over many datasets under the name TabM​[B]TabMdelimited-[]B\text{\mbox{\text{TabM}}}[\text{B}] in [Figure 6](#S5.F6 "Figure 6 ‣ 5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
As such, individually, even the best submodel of TabM is no better than a simple MLP.

The lower row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") analyzes the gradient structure of TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32}.
As a reminder, due to the simultaneous training and the weight sharing between the k=32𝑘32k=32 submodels, most weights of TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} receive the mean of the k𝑘k gradients per object on each training step.
The green line in the lower row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows a near zero cosine similarity between these k𝑘k gradients.
This may explain the higher training loss of TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32} compared to TabMminik=1superscriptsubscriptTabMmini𝑘1\mbox{\text{TabM}}\_{\text{mini}}^{k=1} in the first row if [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"): perhaps, the weight sharing combined with the diverse gradients prevents TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} from (over)optimizing for the training task.

In the same lower row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we run an ablation study on the two sources of submodel diversity in TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}: the random initializations in the adapter R𝑅R and in the k𝑘k prediction heads.
When all rows of R𝑅R (i.e. risubscript𝑟𝑖r\_{i} in terms of [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) receive the same initialization, while the k𝑘k prediction heads are initialized completely randomly (the orange line), the submodel gradients are correlated, and the task performance is poor.
By contrast, the issue is less pronounced when the k𝑘k prediction heads receive the same initialization, and the initialization of R𝑅R is completely random (the purple line), though it also can hurt the performance.
Thus, the first adapter seems to be a more impactful source of gradient diversity.
Overall, we see the gradient diversity as an experimental metric requiring more exploration.

Summary.
TabM draws its power from the collective prediction of weak, but diverse submodels.

### 5.2 Selecting submodels after training

The design of TabM allows selecting only a subset of submodels after training based on any criteria, simply by pruning extra prediction heads and the corresponding rows of the adapter matrices.
To showcase this mechanics, after the training, we Greedily construct a subset of TabM’s submodels with the best collective performance on the validation set, and denote this “pruned” TabM as TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}].
The performance reported in [Figure 6](#S5.F6 "Figure 6 ‣ 5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows that TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}] is slightly behind the vanilla TabM.
On average over 46 datasets, the greedy submodel selection results in 8.8±6.6plus-or-minus8.86.68.8\pm 6.6 submodels out of the initial k=32𝑘32k=32, which can result in faster inference.
See [subsection D.6](#A4.SS6 "D.6 Implementation details of subsection 5.2 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") for implementation details.

### 5.3 How the performance of TabM depends on k𝑘k?

![[Uncaptioned image]](/html/2410.24210/assets/x9.png)

Figure 6: 
The performance on the 46 datasets from [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
TabM​[B]TabMdelimited-[]B\text{\mbox{\text{TabM}}}[\text{B}] and TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}] are described in [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") and [subsection 5.2](#S5.SS2 "5.2 Selecting submodels after training ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

![[Uncaptioned image]](/html/2410.24210/assets/x10.png)

Figure 7: 
The average performance of TabM and TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} over 999 datasets with different values of k𝑘k.

To answer the question in the title, we take TabM with the number of layers 333 and the width 512512512, tune the learning rate for each k𝑘k separately, and report the performance in [Figure 7](#S5.F7 "Figure 7 ‣ 5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Based on the figure, and the results in [section 4](#S4 "4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we suggest that k=32𝑘32k=32 used throughout the paper is a reasonable default value with a good balance between performance and efficiency.
Also, from [Figure 7](#S5.F7 "Figure 7 ‣ 5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), it seems that TabM accommodates large numbers of submodels more easily than TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}.
Perhaps, the larger number of submodel adapters in TabM provides the important additional weight capacity to fit more submodels in one model of a given size.
The implementation details are available in [subsection D.7](#A4.SS7 "D.7 Implementation details of subsection 5.3 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

## 6 Conclusion & Future work

In this work, we have demonstrated that tabular multilayer perceptrons (MLPs) greatly benefit from parameter-efficient ensembling.
Using this insight, we have developed TabM — a simple MLP-based model with state-of-the-art performance.
In a large-scale comparison with many tabular DL models, we have demonstrated that TabM is ready to serve as a new powerful and efficient tabular DL baseline.
Finally, we have analyzed the properties of the implicit submodels underlying TabM.

One idea for future work is to bring the power of (parameter-)efficient ensembles to other, non-tabular, domains with optimization-related challenges and, ideally, lightweight base models.
Another idea is to evaluate TabM for uncertainty estimation and out-of-distribution (OOD) detection on tabular data, which is inspired by works like Lakshminarayanan et al. ([2017](#bib.bib27)).

Reproducibility statement.
The code is provided in the following repository: [link](https://github.com/yandex-research/tabm).
It contains the implementation of TabM, hyperparameter tuning scripts, evaluation scripts, configuration files with hyperparameters (the TOML files in the exp/ directory) and the report files with the main metrics (the JSON files in the exp/ directory).
In the paper, the model is described in [section 3](#S3 "3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and the implementation details are provided in [Appendix D](#A4 "Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

## References

* Akiba et al. (2019)

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, 2019.
* Allen-Zhu & Li (2023)

  Zeyuan Allen-Zhu and Yuanzhi Li.
  Towards understanding ensemble, knowledge distillation and self-distillation in deep learning.
  In *ICLR*, 2023.
* Ansel et al. (2024)

  Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala.
  Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation.
  In *ASPLOS*, 2024.
* Antorán et al. (2020)

  Javier Antorán, James Urquhart Allingham, and José Miguel Hernández-Lobato.
  Depth uncertainty in neural networks.
  In *NeurIPS*, 2020.
* Arik & Pfister (2020)

  Sercan O. Arik and Tomas Pfister.
  TabNet: Attentive interpretable tabular learning.
  *arXiv*, 1908.07442v5, 2020.
* Badirli et al. (2020)

  Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and Sathiya S. Keerthi.
  Gradient boosting neural networks: GrowNet.
  *arXiv*, 2002.07971v2, 2020.
* Bahri et al. (2021)

  Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler.
  SCARF: Self-supervised contrastive learning using random feature corruption.
  In *ICLR*, 2021.
* Chen et al. (2023a)

  Jintai Chen, Jiahuan Yan, Danny Ziyi Chen, and Jian Wu.
  ExcelFormer: A neural network surpassing gbdts on tabular data.
  *arXiv*, 2301.02819v1, 2023a.
* Chen et al. (2023b)

  Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang.
  Trompt: Towards a better deep neural network for tabular data.
  In *ICML*, 2023b.
* Chen & Guestrin (2016)

  Tianqi Chen and Carlos Guestrin.
  XGBoost: A scalable tree boosting system.
  In *SIGKDD*, 2016.
* Fort et al. (2020)

  Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan.
  Deep ensembles: A loss landscape perspective.
  *arXiv*, 1912.02757v2, 2020.
* Garipov et al. (2018)

  Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, and Andrew Gordon Wilson.
  Loss surfaces, mode connectivity, and fast ensembling of dnns.
  In *NeurIPS*, 2018.
* Gorishniy et al. (2021)

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, 2021.
* Gorishniy et al. (2022)

  Yury Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  In *NeurIPS*, 2022.
* Gorishniy et al. (2024)

  Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko.
  TabR: Tabular deep learning meets nearest neighbors.
  In *ICLR*, 2024.
* Grinsztajn et al. (2022)

  Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux.
  Why do tree-based models still outperform deep learning on typical tabular data?
  In *NeurIPS, the ”Datasets and Benchmarks” track*, 2022.
* Havasi et al. (2021)

  Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai, and Dustin Tran.
  Training independent subnetworks for robust prediction.
  In *ICLR*, 2021.
* Hollmann et al. (2023)

  Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.
  TabPFN: A transformer that solves small tabular classification problems in a second.
  In *ICLR*, 2023.
* Holzmüller et al. (2024)

  David Holzmüller, Léo Grinsztajn, and Ingo Steinwart.
  Better by default: Strong pre-tuned mlps and boosted trees on tabular data.
  *arXiv*, 2407.04491v1, 2024.
* Jeffares et al. (2023a)

  Alan Jeffares, Tennison Liu, Jonathan Crabbé, Fergus Imrie, and Mihaela van der Schaar.
  TANGOS: Regularizing tabular neural networks through gradient orthogonalization and specialization.
  In *ICLR*, 2023a.
* Jeffares et al. (2023b)

  Alan Jeffares, Tennison Liu, Jonathan Crabbé, and Mihaela van der Schaar.
  Joint training of deep ensembles fails due to learner collusion.
  In *NeurIPS*, 2023b.
* Kadra et al. (2021)

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In *NeurIPS*, 2021.
* Ke et al. (2017)

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  LightGBM: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30:3146–3154, 2017.
* Kim et al. (2024)

  Myung Jun Kim, Léo Grinsztajn, and Gaël Varoquaux.
  CARTE: pretraining and transfer for tabular learning.
  *arXiv*, abs/2402.16785v1, 2024.
* Klambauer et al. (2017)

  Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
  Self-normalizing neural networks.
  In *NIPS*, 2017.
* Kossen et al. (2021)

  Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin Gal.
  Self-attention between datapoints: Going beyond individual input-output pairs in deep learning.
  In *NeurIPS*, 2021.
* Lakshminarayanan et al. (2017)

  Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
  Simple and scalable predictive uncertainty estimation using deep ensembles.
  In *NeurIPS*, 2017.
* Laurent et al. (2023)

  Olivier Laurent, Adrien Lafage, Enzo Tartaglione, Geoffrey Daniel, Jean-Marc Martinez, Andrei Bursuc, and Gianni Franchi.
  Packed ensembles for efficient uncertainty estimation.
  In *ICLR*, 2023.
* Lee et al. (2015)

  Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David J. Crandall, and Dhruv Batra.
  Why M heads are better than one: Training a diverse ensemble of deep networks.
  *arXiv*, abs/1511.06314, 2015.
* Loshchilov & Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *ICLR*, 2019.
* Marton et al. (2024)

  Sascha Marton, Stefan Lüdtke, Christian Bartelt, and Heiner Stuckenschmidt.
  GRANDE: Gradient-based decision tree ensembles for tabular data.
  In *ICLR*, 2024.
* Paszke et al. (2019)

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala.
  PyTorch: An imperative style, high-performance deep learning library.
  In *NeurIPS*, 2019.
* Pedregosa et al. (2011)

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12:2825–2830, 2011.
* Popov et al. (2020)

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular data.
  In *ICLR*, 2020.
* Prokhorenkova et al. (2018)

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin.
  CatBoost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Qin & Liu (2013)

  Tao Qin and Tie-Yan Liu.
  Introducing LETOR 4.0 datasets.
  *arXiv*, 1306.2597v1, 2013.
* Rubachev et al. (2022)

  Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko.
  Revisiting pretraining objectives for tabular deep learning.
  *arXiv*, 2207.03208v1, 2022.
* Rubachev et al. (2024)

  Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, and Artem Babenko.
  TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.
  *arXiv*, 2406.19380v4, 2024.
* Somepalli et al. (2021)

  Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein.
  SAINT: improved neural networks for tabular data via row attention and contrastive pre-training.
  *arXiv*, 2106.01342v1, 2021.
* Song et al. (2019)

  Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang.
  Autoint: Automatic feature interaction learning via self-attentive neural networks.
  In *CIKM*, 2019.
* Srivastava et al. (2014)

  Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
  Dropout: a simple way to prevent neural networks from overfitting.
  *Journal of Machine Learning Research*, 15(1):1929–1958, 2014.
* Tolstikhin et al. (2021)

  Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy.
  Mlp-mixer: An all-mlp architecture for vision.
  In *NeurIPS*, 2021.
* Turkoglu et al. (2022)

  Mehmet Ozgur Turkoglu, Alexander Becker, Hüseyin Anil Gündüz, Mina Rezaei, Bernd Bischl, Rodrigo Caye Daudt, Stefano D’Aronco, Jan D. Wegner, and Konrad Schindler.
  Film-ensemble: Probabilistic deep learning via feature-wise linear modulation.
  In *NeurIPS 2022*, 2022.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *NIPS*, 2017.
* Wang et al. (2020)

  Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi.
  Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.
  *arXiv*, 2008.13535v2, 2020.
* Wen et al. (2020)

  Yeming Wen, Dustin Tran, and Jimmy Ba.
  Batchensemble: an alternative approach to efficient ensemble and lifelong learning.
  In *ICLR*, 2020.
* Yan et al. (2023)

  Jiahuan Yan, Jintai Chen, Yixuan Wu, Danny Z. Chen, and Jian Wu.
  T2G-FORMER: organizing tabular features into relation graphs promotes heterogeneous feature interaction.
  In *AAAI*, 2023.
* Ye et al. (2024)

  Han-Jia Ye, Huai-Hong Yin, and De-Chuan Zhan.
  Modern neighborhood components analysis: A deep tabular baseline two decades later.
  *arXiv*, 2407.03257v1, 2024.
* Zhang et al. (2020)

  Shaofeng Zhang, Meng Liu, and Junchi Yan.
  The diversified ensemble neural network.
  In *NeurIPS*, 2020.

## Appendix A Additional discussion on TabM

### A.1 Motivation

Why BatchEnsemble?
Among relatively ease-to-use “efficient ensembling” methods, beyond BatchEnsemble, there are examples such as dropout ensembles (Lakshminarayanan et al., [2017](#bib.bib27)), naive multi-head architectures, TreeNet (Lee et al., [2015](#bib.bib29)).
However, in the literature, they were consistently outperformed by more advanced methods, including BatchEnsemble (Wen et al., [2020](#bib.bib46)), MIMO (Havasi et al., [2021](#bib.bib17)), FiLM-Ensemble (Turkoglu et al., [2022](#bib.bib43)).

Among advanced methods, BatchEnsemble seems to be one of the simplest and most flexible options.
For example, FiLM-Ensemble (Turkoglu et al., [2022](#bib.bib43)) requires normalization layers to be presented in the original architecture, which is not always the case for tabular MLPs.
MIMO (Havasi et al., [2021](#bib.bib17)), in turn, imposes additional limitations compared to BatchEnsemble.
First, it requires concatenating (not stacking, as with BatchEnsemble) all k𝑘k input representations, which increases the input size of the first linear layer.
With the relatively high number of submodels k=32𝑘32k=32 used in our paper, this can be an issue on datasets with a large number of features, and especially when feature embeddings (Gorishniy et al., [2022](#bib.bib14)) are used.
For example, for k=32𝑘32k=32, the number of features m=1000𝑚1000m=1000 and the feature embedding size l=32𝑙32l=32, the input size approaches one million resulting in an extremely large first linear layer of MLP.
Second, with BatchEnsemble, it is easy to explicitly materialize, analyze and prune individual submodels.
By contrast, in MIMO, all submodels are implicitly entangled within one MLP, and there is no easy way to access individual submodels.

Why MLPs?
Despite the applicability of BatchEnsemble (Wen et al., [2020](#bib.bib46)) to almost any architecture, we focus specifically on MLPs.
The key reason is efficiency.
First, to achieve high performance, throughout the paper, we use the relatively large number of submodels k=32𝑘32k=32.
However, the desired less-than-×kabsent𝑘\times k runtime overhead of BatchEnsemble typically happens only when the original model underutilizes the power of parallel computations of a given hardware.
This will not be the case for attention-based models on datasets with a large number of features, as well as for retrieval-based models on datasets with a large number of objects.
Second, as we show in [subsection 4.3](#S4.SS3 "4.3 Efficiency ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), attention- and retrieval-based models are already slow as-is.
By contrast, MLPs are exceptionally efficient, to the extent that slowing them down even by an order of magnitude will still result in practical models.

Also, generally speaking, the definition of MLP suggested in [subsection 3.3](#S3.SS3 "3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") and used in TabM is not special, and more advanced MLP-like backbones can be used.
However, in preliminary experiments, we did not observe the benefits of more advanced backbones.
Perhaps, small technical differences between backbones become less impactful in the context of parameter-efficient ensembling, at least in the scope of middle-to-large-sized datasets.

### A.2 Why TabM outperforms a full-fledged deep ensemble?

As shown in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), TabMnaive, TabM and TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} are all superior to MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} — the full-fledged ensemble of k𝑘k MLPs.
Moreover, the performance gap is significant.
To the best of our knowledge, this result is rather not expected, because the literature on efficient ensembles is usually focused on catching up with deep ensembles, not on outperforming them, let alone significantly outperforming them.
Plus, after the training, TabM can be explicitly materialized as a traditional ensemble of k𝑘k MLPs.
With that in mind, we highlight three hypotheses for the superior performance of TabM compared to MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k}.
The first one relies on the analysis from [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and the other two are more general.

First, we highlight the combination of the weight sharing and gradient diversity (observed in the third row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) in TabM.
Perhaps, the regularization power of the weakly aligned individual gradients cannot be recovered by technical details such as hyperparameter tuning, training protocol, etc.
In this case, we can say that MLP ensembles seem to win from this regularization when training on tabular data.
This, in turn, can be related to the common belief that optimization on tabular data is challenging, especially for MLPs.
Overall, it is unclear if the phenomenon will generalize to other models and/or domains.

Second, the hyperparameter tuning for MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} may be suboptimal.
Recall that, in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we tune hyperparameters of one ensemble-unaware MLP, and then train the tuned MLP from scratch k𝑘k times to obtain MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k}.
Perhaps, if the hyperparameters are tuned directly for MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k}, then more powerful (and exotic) configurations can be found.
However, that would make the tuning ×kabsent𝑘\times k more expensive.

Third, the parallel training (and, in particular, early stopping) of the implicit submodels in TabM may be important.
The direct analogy for MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} would be to train all k𝑘k members in parallel on the same training batches.
In [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), all members MLP×ksuperscriptMLPabsent𝑘\text{MLP}^{\times k} are trained independently under different random seeds (and, in particular, over different training batch sequences).

### A.3 TabM with feature embeddings

Here, we provide additional implementation details for TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} and TabM†superscriptTabM†\mbox{\text{TabM}}^{\dagger} introduced in [subsection 3.3](#S3.SS3 "3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
In fact, there are no changes in the usage of feature embeddings compared to plain MLPs.

Technically, feature embeddings are applied, and the result is flattened, before the Clone module in terms of [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
For example, if a dataset has m𝑚m continuous features and all of them are embedded, the very first adapter R𝑅R will have the shape k×m​de𝑘𝑚subscript𝑑𝑒k\times md\_{e}, where desubscript𝑑𝑒d\_{e} is the feature embedding size.
For TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger} and TabM†superscriptTabM†\mbox{\text{TabM}}^{\dagger}, we initialize the first multiplicative adapter R𝑅R of the first linear layer from the standard normal distribution 𝒩​(0,1)𝒩01\mathcal{N}(0,1).
The remaining details are best understood from the source code.

### A.4 Hyperparameters

As mentioned in the main text, we noticed that the typical optimal learning rate for TabM is higher than for MLP
(we share the tuned hyperparameters on all datasets in the repository, so the precise comparison is possible).
We hypothesize that the reason is the effectively larger batch size because of the Clone operation (see [Figure 1](#S3.F1 "Figure 1 ‣ 3.3 \"TabM\"_\"mini\" & TabM ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")), which, in particular, leads to the k𝑘k mostly orthogonal gradients per object (see [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) on each training step.
The latter leads to lower gradient norms after averaging the k𝑘k gradients, which can be related to the need for larger learning rates.

### A.5 Limitations and practical considerations

TabM does not introduce any new limitations compared to BatchEnsemble (Wen et al., [2020](#bib.bib46)).
Nevertheless, we note the following:

* •

  The MLP backbone used in TabM is one of the simplest possible, and generally, more advanced backbones can be used.
  That said, some backbones may require additional care when used in TabM.
  For example, we did not explore backbones with normalization layers.
  For such layers, it is possible to allocate non-shared trainable affine transformations for each implicit submodel by adding one multiplicative and one additive adapter after the normalization layer (i.e. like in FiLM-Ensemble (Turkoglu et al., [2022](#bib.bib43))).
  Additional experiments are required to find the best strategy.
* •

  Arguably the key limitation is that BatchEnsemble-like techniques are not “local”, but instead affect the whole model starting from the first modified layer.
  Namely, when the computation flow hits the first modified layer, the k𝑘k prediction branches are created, and the rest of the network will have to make k𝑘k times more computations.
  This can be easily affordable for small base models, but may be less affordable for heavy base models.
* •

  For ensemble-like models, such as TabM, the notion of “the final object embedding“ changes: now, it is not a single vector, but a set of k𝑘k vectors.
  This can be important for scenarios when TabM is used for solving more than one task, in particular, when it is pretrained as a generic feature extractor and then reused for other tasks.
  The main practical guideline is that the k𝑘k prediction branches should never interact with each other (e.g. through attention, pooling, etc.) and should always be trained separately.

## Appendix B Extended results

This section complements [section 4](#S4 "4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

### B.1 Additional baselines

In addition to [subsection 4.1](#S4.SS1 "4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), we consider the following models:

* •

  MLP-PLR Gorishniy et al. ([2022](#bib.bib14)), that is, an MLP with periodic embeddings.
* •

  ResNet (Gorishniy et al., [2021](#bib.bib13))
* •

  SNN (Klambauer et al., [2017](#bib.bib25))
* •

  DCNv2 (Wang et al., [2020](#bib.bib45))
* •

  AutoInt (Song et al., [2019](#bib.bib40))
* •

  MLP-Mixer is our adaptation of Tolstikhin et al. ([2021](#bib.bib42)) for tabular data.
* •

  Trompt (Chen et al., [2023b](#bib.bib9)) (our reimplementation, since there is no official implementation)

We also evaluated TabPFN (Hollmann et al., [2023](#bib.bib18)), where possible.
The results for this model are available only in [Appendix E](#A5 "Appendix E Per-dataset results with standard deviations ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), because this model is by design not applicable to regression tasks, which is a considerable number of our datasets.
Overall, TabPFN specializes on small datasets.
In line with that, the performance of TabPFN on our benchmark was not competitive.

### B.2 Task performance

[Figure 8](#A2.F8 "Figure 8 ‣ B.2 Task performance ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") is a different version of [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") with additional baselines.
Overall, none of the additional baselines affects our main story.

[Figure 9](#A2.F9 "Figure 9 ‣ B.2 Task performance ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") is the critical difference diagram (CDD) computed over exactly the same results that were used for building [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

![Refer to caption](/html/2410.24210/assets/x11.png)


Figure 8: 
An extended comparison of tabular models as in [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Note that the ranks (left) are computed only over the 37 datasets with random splits, because ResNet, AutoInt and MLP-Mixer were evaluated only on one 111 out of 999 datasets with domain-aware splits.

![Refer to caption](/html/2410.24210/assets/x12.png)


Figure 9: 
Critical difference diagram.
The computation method is taken from the Kim et al. ([2024](#bib.bib24)).

### B.3 Efficiency

This section complements [subsection 4.3](#S4.SS3 "4.3 Efficiency ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Additional results.

[Figure 10](#A2.F10 "Figure 10 ‣ B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") complements [Figure 4](#S4.F4 "Figure 4 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") by providing the training times on smaller datasets and the inference throughput on GPU with large batch sizes.

[Table 3](#A2.T3 "Table 3 ‣ B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") provide the number of trainable parameters for some of the models from [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Motivation for the benchmark setup.
Comparing models under all possible kinds of budgets (task performance, the number of parameters, training time, etc.) on all possible hardware (GPU, CPU, etc.) with all possible batch sizes is rather infeasible.
As such, we set a narrow goal of providing a high-level intuition on the efficiency in a transparent setting.
Thus, benchmarking the transparently obtained tuned hyperparameter configurations works well for our goal.
Yet, this choice also has a limitation: the hyperparameter tuning process is not aware of the efficiency budget, so it can prefer much heavier configurations even if they lead to tiny performance improvements, which will negatively affect efficiency without a good reason.
Overall, we hope that the large number of datasets compensates for potentially imperfect per-dataset measurements.

Motivation for the two setups for measuring inference throughput.

* •

  The setup in the right side of [Figure 4](#S4.F4 "Figure 4 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") simulates the online per-object predictions.
* •

  The setup in the right side of [Figure 10](#A2.F10 "Figure 10 ‣ B.3 Efficiency ‣ Appendix B Extended results ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") simulates the offline batched computations.

![Refer to caption](/html/2410.24210/assets/x13.png)

![Refer to caption](/html/2410.24210/assets/x14.png)

Figure 10: 
(Left) Training time on datasets with less than 100K objects.
(Right) Inference throughput on GPU with maximum possible batch size (i.e. the batch size depends of a model).




Table 3: Mean number of parameters with std. dev. for 7 different tuned models across all 46 datasets.

| TabM | MLP | FT-T | T2G | TabR | ModernNCA | SAINT |
| --- | --- | --- | --- | --- | --- | --- |
| 1.4​M±1.3​Mplus-or-minus1.4𝑀1.3𝑀1.4M\pm 1.3M | 1.0​M±1.0​Mplus-or-minus1.0𝑀1.0𝑀1.0M\pm 1.0M | 1.2​M±1.2​Mplus-or-minus1.2𝑀1.2𝑀1.2M\pm 1.2M | 2.1​M±1.6​Mplus-or-minus2.1𝑀1.6𝑀2.1M\pm 1.6M | 858​K±1.4​Mplus-or-minus858𝐾1.4𝑀858K\pm 1.4M | 1.0​M±1.1​Mplus-or-minus1.0𝑀1.1𝑀1.0M\pm 1.1M | 175.4​M±565.4​Mplus-or-minus175.4𝑀565.4𝑀175.4M\pm 565.4M |

## Appendix C Datasets

In total, we use 46 datasets:

1. 1.

   383838 datasets are taken from Gorishniy et al. ([2024](#bib.bib15)), which includes:

   1. (a)

      282828 datasets from Grinsztajn et al. ([2022](#bib.bib16)).
      See the original paper for the precise dataset information.
   2. (b)

      101010 datasets from other sources.
      Their properties are provided in [Table 4](#A3.T4 "Table 4 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
2. 2.

   888 datasets are taken from Rubachev et al. ([2024](#bib.bib38)).
   Their properties are provided in [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

In fact, the aforementioned 383838 datasets from Gorishniy et al. ([2024](#bib.bib15)) is only a subset of the datasets used in Gorishniy et al. ([2024](#bib.bib15)).
Namely, we did not include the following of the remaining datasets:

* •

  The datasets that, according to Rubachev et al. ([2024](#bib.bib38)), have incorrect splits and/or label leakage, including:
  Bike​\_​Sharing​\_​DemandBike\_Sharing\_Demand\mathrm{Bike\\_Sharing\\_Demand},
  compasscompass\mathrm{compass},
  electricityelectricity\mathrm{electricity},
  SGEMM​\_​GPU​\_​kernel​\_​performanceSGEMM\_GPU\_kernel\_performance\mathrm{SGEMM\\_GPU\\_kernel\\_performance},
  sulfursulfur\mathrm{sulfur},
  visualizing​\_​soilvisualizing\_soil\mathrm{visualizing\\_soil},
  and the weather forecasting dataset (it is replaced by the correct weather forecasting dataset from Rubachev et al. ([2024](#bib.bib38))).
* •

  rlrl\mathrm{rl} from (Grinsztajn et al., [2022](#bib.bib16)).
  We observed abnormal results on these datasets.
  This is an anonymous dataset, which made the investigation impossible, so we removed this dataset to avoid confusion.
* •

  yprop​\_​4​\_​1yprop\_4\_1\mathrm{yprop\\_4\\_1} from (Grinsztajn et al., [2022](#bib.bib16)).
  Strictly speaking, this dataset was omitted due to a mistake on our side.
  For future work, we note that the typical performance gaps on this dataset have low absolute values in terms of RMSE.
  Perhaps, R2superscript𝑅2R^{2} may be a more appropriate metric for this dataset.

Table 4: 
Properties of those datasets from Gorishniy et al. ([2024](#bib.bib15))
that are not part of Grinsztajn et al. ([2022](#bib.bib16)) or Rubachev et al. ([2024](#bib.bib38)).
“# Num”, “# Bin”, and “# Cat” denote the number of numerical, binary, and categorical features, respectively.
The table is taken from (Gorishniy et al., [2024](#bib.bib15)).

| Name | # Train | # Validation | # Test | # Num | # Bin | # Cat | Task type | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Churn Modelling | 6 40064006\,400 | 1 60016001\,600 | 2 00020002\,000 | 101010 | 333 | 111 | Binclass | 128 |
| California Housing | 13 2091320913\,209 | 3 30333033\,303 | 4 12841284\,128 | 888 | 00 | 00 | Regression | 256 |
| House 16H | 14 5811458114\,581 | 3 64636463\,646 | 4 55745574\,557 | 161616 | 00 | 00 | Regression | 256 |
| Adult | 26 0482604826\,048 | 6 51365136\,513 | 16 2811628116\,281 | 666 | 111 | 888 | Binclass | 256 |
| Diamond | 34 5213452134\,521 | 8 63186318\,631 | 10 7881078810\,788 | 666 | 00 | 333 | Regression | 512 |
| Otto Group Products | 39 6013960139\,601 | 9 90199019\,901 | 12 3761237612\,376 | 939393 | 00 | 00 | Multiclass | 512 |
| Higgs Small | 62 7516275162\,751 | 15 6881568815\,688 | 19 6101961019\,610 | 282828 | 00 | 00 | Binclass | 512 |
| Black Friday | 106 764106764106\,764 | 26 6922669226\,692 | 33 3653336533\,365 | 444 | 111 | 444 | Regression | 512 |
| Covertype | 371 847371847371\,847 | 92 9629296292\,962 | 116 203116203116\,203 | 151515 | 444 | 111 | Multiclass | 1024 |
| Microsoft | 723 412723412723\,412 | 235 259235259235\,259 | 241 521241521241\,521 | 131131131 | 555 | 00 | Regression | 1024 |




Table 5: 
Properties of the datasets from the TabReD benchmark (Rubachev et al., [2024](#bib.bib38)).
“# Num”, “# Bin”, and “# Cat” denote the number of numerical, binary, and categorical features, respectively.

| Name | # Train | # Validation | # Test | # Num | # Bin | # Cat | Task type | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sberbank Housing | 18 8471884718\,847 | 4 82748274\,827 | 4 64746474\,647 | 365365365 | 171717 | 101010 | Regression | 256 |
| Ecom Offers | 109 341109341109\,341 | 24 2612426124\,261 | 26 4552645526\,455 | 113113113 | 666 | 00 | Binclass | 1024 |
| Maps Routing | 160 019160019160\,019 | 59 9755997559\,975 | 59 9515995159\,951 | 984984984 | 00 | 222 | Regression | 1024 |
| Homesite Insurance | 224 320224320224\,320 | 20 1382013820\,138 | 16 2951629516\,295 | 253253253 | 232323 | 232323 | Binclass | 1024 |
| Cooking Time | 227 087227087227\,087 | 51 2515125151\,251 | 41 6484164841\,648 | 186186186 | 333 | 333 | Regression | 1024 |
| Homecredit Default | 267 645267645267\,645 | 58 0185801858\,018 | 56 0015600156\,001 | 612612612 | 222 | 828282 | Binclass | 1024 |
| Delivery ETA | 279 415279415279\,415 | 34 1743417434\,174 | 36 9273692736\,927 | 221221221 | 111 | 111 | Regression | 1024 |
| Weather | 106 764106764106\,764 | 42 3594235942\,359 | 40 8404084040\,840 | 100100100 | 333 | 00 | Regression | 1024 |

## Appendix D Implementation details

### D.1 Hardware

Most of the experiments were conducted on a single NVIDIA A100 GPU.
In rare exceptions, we used a machine with a single NVIDIA 2080 Ti GPU and Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz.

### D.2 Experiment setup

We mostly follow the experiment setup from Gorishniy et al. ([2024](#bib.bib15)).
As such, some of the text below is copied from (Gorishniy et al., [2024](#bib.bib15)).

Data preprocessing.
For each dataset, for all DL-based solutions, the same preprocessing was used for fair comparison.
For numerical features, by default, we used a slightly modified version of the quantile normalization from the Scikit-learn package (Pedregosa et al., [2011](#bib.bib33)) (see the source code), with rare exceptions when it turned out to be detrimental (for such datasets, we used the standard normalization or no normalization).
For categorical features, we used one-hot encoding.
Binary features (i.e. the ones that take only two distinct values) are mapped to {0,1}01\{0,1\} without any further preprocessing. We completely follow Rubachev et al. ([2024](#bib.bib38)) on [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") datasets.

Training neural networks.
For DL-based algorithms, we minimize cross-entropy for classification problems and mean squared error for regression problems.
We use the AdamW optimizer (Loshchilov & Hutter, [2019](#bib.bib30)).
We do not apply learning rate schedules.
We do not use data augmentations.
We apply global gradient clipping to 1.01.01.0.
For each dataset, we used a predefined dataset-specific batch size.
We continue training until there are patience consecutive epochs without improvements on the validation set; we set patience=16patience16\texttt{patience}=16 for the DL models.

Hyperparameter tuning.
In most cases, hyperparameter tuning is performed with the TPE sampler (typically, 50-100 iterations) from the Optuna package (Akiba et al., [2019](#bib.bib1)).
Hyperparameter tuning spaces for most models are provided in individual sections below (example for TabM: [subsection D.8](#A4.SS8 "D.8 TabM ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")). We follow Rubachev et al. ([2024](#bib.bib38)) and use 252525 iterations on some datasets from [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Evaluation.
On a given dataset, for a given model, the tuned hyperparameters are evaluated under multiple (in most cases, 151515) random seeds.
The mean test metric and its standard deviation over these random seeds are then used to compare algorithms as described in [subsection D.3](#A4.SS3 "D.3 Metrics ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

### D.3 Metrics

We use Root Mean Squared Error for regression tasks, ROC-AUC for classification datasets from [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") (following Rubachev et al. ([2024](#bib.bib38))), and accuracy for the rest of datasets (following Gorishniy et al. ([2024](#bib.bib15))).
We also tried computing ROC-AUC for all classification datasets, but did not observe any significant changes (see [Figure 11](#A4.F11 "Figure 11 ‣ D.4 Implementation details of subsection 4.3 ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")), so we stuck to prior work.
By default, the mean test score and its standard deviation are obtained by training a given model with tuned hyperparameters from scratch on a given dataset under 15 different random seeds.

How we compute ranks.
Our method of computing ranks does not count small improvements as wins, hence the reduced range of ranks compared to other studies.
Intuitively, our ranks can be considered as “tiers”.

Assume the higher the score the better and mean\_score\_Ref>mean\_score\_Amean\_score\_Refmean\_score\_A\texttt{mean\\_score\\_Ref}>\texttt{mean\\_score\\_A}. Then reference Model\_Ref and Model\_A are equal (have the same rank) if (mean\_score\_Ref−mean\_score\_A)≤std\_Refmean\_score\_Refmean\_score\_Astd\_Ref(\texttt{mean\\_score\\_Ref}-\texttt{mean\\_score\\_A})\leq\texttt{std\\_Ref}. To assign ranks, we sort models in descending score order. Starting from the best model (with rank equal to 1) we iterate over models and assign first rank to all models that are equal to the best model according to the mentioned definition. The first model in descending order that is not equal to the best model is assigned with second rank and becomes a new reference model. We continue the process until all models are ranked. Ranks are computed independently for each dataset.

### D.4 Implementation details of [subsection 4.3](#S4.SS3 "4.3 Efficiency ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")

Applicability to large datasets.
The two datasets used in [Table 2](#S4.T2 "Table 2 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") are the full versions of the “Weather” and “Maps Routing” datasets from Rubachev et al. ([2024](#bib.bib38)).
Their smaller versions with subsampled training set were already included in [Table 1](#S3.T1 "Table 1 ‣ 3.1 Preliminaries ‣ 3 TabM ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), and were used when building [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
The validation and test sets are the same for the small and large versions of these datasets, so the task metrics are comparable between the two versions.
When running models on the large versions of the datasets, we reused the hyperparameters tuned for their small versions.
Thus, this experiment can be seen as a quick assessment of the applicability of several tabular DL to large datasets, without a strong focus on the task performance.
All models, except for FT-Transformer, were evaluated under 333 random seeds.
FT-Transformer was evaluated under 111 random seed.

![Refer to caption](/html/2410.24210/assets/x15.png)


Figure 11: 
Same as [Figure 3](#S4.F3 "Figure 3 ‣ 4.1 Baselines ‣ 4 Evaluating tabular deep learning architectures ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"), but ROC-AUC is used as the metric for all classification datasets.
The two multiclass datasets presented in our benchmark are not taken into account.

### D.5 Implementation details of [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")

Experiment setup.
This paragraph complements the description of the experiment setup in [subsection 5.1](#S5.SS1 "5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
Namely, in addition to what is mentioned in the main text:

* •

  Dropout and weight decay are turned off.
* •

  To get representative training profiles for all models, the learning rates are tuned separately for TabMminik=1superscriptsubscriptTabMmini𝑘1\mbox{\text{TabM}}\_{\text{mini}}^{k=1} and TabMminik=32superscriptsubscriptTabMmini𝑘32\mbox{\text{TabM}}\_{\text{mini}}^{k=32} on validation sets using the usual metrics (i.e. RMSE or accuracy) as the guidance.
  The grid for learning rate tuning was: numpy.logspace(numpy.log10(1e-5), numpy.log10(5e-3), num=25).
* •

  The adapter R𝑅R in TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}} is initialized from N​(0,1)N01\mathrm{N}(0,1) instead of ±1plus-or-minus1\pm 1.
  We observed that ±1plus-or-minus1\pm 1 resulted in occasional minor jumps in some of the curves shown in [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
  The number and magnitude of the jumps were low and had no effect on the story.
  However, in line with all other changes to the experiment setup in this section, we adjusted the initialization to avoid the jumps, to ensure that the training runs were not disturbed by side-effects of unknown nature.

Cosine similarity between the k𝑘k gradients.
This paragraph complements the story about the third row of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
The reported cosine similarity between the k𝑘k gradients was computed as follows.
Before the training, we randomly select n=1000𝑛1000n=1000 training objects — these reference objects stay the same during the whole training run.
During the training, at the start of every epoch, we calculate the metric according to the formula in the caption of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"):

1. 1.

   First, for each of the n𝑛n reference objects, we compute the k𝑘k individual gradients, and omit the gradient components related to the weights of the prediction heads.
   Thus, the size of one gradient equals the number of all parameters, minus the number of parameters in one prediction head.
2. 2.

   Then for each of the n𝑛n reference objects, we compute the k​(k−1)2𝑘𝑘12\frac{k(k-1)}{2} pairwise cosine similarities between the k𝑘k individual gradients.
3. 3.

   Finally, we average the cosine similarities over all pairs of gradients over all reference objects, which gives the formula in the caption of [Figure 5](#S5.F5 "Figure 5 ‣ 5.1 Performance and training dynamics of the individual submodels ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Limitations.
Continuing the story about the gradient diversity, we note that in high-dimensional spaces, the near-zero values of cosine similarity can be hard to interpret, which is a potential limitation of our analysis.
That said, a non-trivial positive (negative) value is rather a decent indicator of a non-trivial positive (negative) correlation between vectors.

### D.6 Implementation details of [subsection 5.2](#S5.SS2 "5.2 Selecting submodels after training ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")

TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}].
Here, we clarify implementation details for TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}] announced in [subsection 5.2](#S5.SS2 "5.2 Selecting submodels after training ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").
TabM​[G]TabMdelimited-[]G\text{\mbox{\text{TabM}}}[\text{G}] is obtained from a trained TabM by greedily selecting submodels from TabM starting from the best one and stopping when two conditions are simultaneously true for the first time: (1) adding any new submodel does not improve the validation metrics of the collective prediction; (2) the current validation metric is already better than that of the initial model with all k𝑘k submodels.
To clarify, during the greedy selection, the i𝑖i-th submodel is considered to be better than the j𝑗j-th submodel if adding the i𝑖i-th submodel to the aggregated prediction leads to better validation metrics (i.e. it is not the same as adding the submodel in the order of their individual validation metrics).

### D.7 Implementation details of [subsection 5.3](#S5.SS3 "5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")

[Figure 7](#S5.F7 "Figure 7 ‣ 5.3 How the performance of TabM depends on 𝑘? ‣ 5 Analysis ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") shows the mean percentage improvements (see [subsection D.3](#A4.SS3 "D.3 Metrics ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")) over MLP across 9 datasets from [Table 4](#A3.T4 "Table 4 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") (without Covertype).
We have used a TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}  with 333 hidden layers of the width d=512𝑑512d=512, the dropout rate 0.1 and tuned learning rate for different k𝑘k.
The score on each dataset is averaged over 555 seeds.

### D.8 TabM

Here we provide hyperparameter tuning spaces for TabM  and TabMminisubscriptTabMmini\mbox{\text{TabM}}\_{\text{mini}}.

Table 6: The hyperparameter tuning space for TabM. Here, (B) = {Covertype, Microsoft, [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")} and (A) contains all other datasets.

| Parameter | Distribution or Value |
| --- | --- |
| k𝑘k | 323232 |
| # layers | UniformInt​[1,5]UniformInt15\mathrm{UniformInt}[1,5] |
| Width (hidden size) | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[1​e​-​4,5​e​-​3]LogUniform1𝑒-45𝑒-3\mathrm{LogUniform}[1e\text{-}4,5e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 |




Table 7: The hyperparameter tuning space for TabMmini†superscriptsubscriptTabMmini†\mbox{\text{TabM}}\_{\text{mini}}^{\dagger}  that uses PiecewiseLinearEncoding embeddings from Gorishniy et al. ([2022](#bib.bib14)). Here, (B) = {Covertype, Microsoft, [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")} and (A) contains all other datasets.

| Parameter | Distribution or Value |
| --- | --- |
| k𝑘k | 323232 |
| # layers | UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| Width (hidden size) | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| # PLE bins | UniformInt​[8,32]UniformInt832\mathrm{UniformInt}[8,32] |
| Learning rate | LogUniform​[5​e​-​5,3​e​-​3]LogUniform5𝑒-53𝑒-3\mathrm{LogUniform}[5e\text{-}5,3e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 |

### D.9 TabR

Since we follow the training and evaluation protocols from Gorishniy et al. ([2024](#bib.bib15)), and TabR was proposed in Gorishniy et al. ([2024](#bib.bib15)), we simply reuse results for TabR. More details can be found in Appendix.D from Gorishniy et al. ([2024](#bib.bib15)). For TabR††\dagger in [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") we have used 252525 tuning iterations and the same tuning space as for TabR from Rubachev et al. ([2024](#bib.bib38)), we also followed Gorishniy et al. ([2024](#bib.bib15)) and used periodic embeddings on small datasets (Sberbank Housing and Ecom Offers) and Linear-ReLU embeddings for the other datasets.

### D.10 FT-Transformer

We used the implementation from the ”rtdl\_revisiting\_models” Python package (version 0.0.2). The results on datasets from [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") were copied from Rubachev et al. ([2024](#bib.bib38)).

Table 8: 
The hyperparameter tuning space for FT-Transformer Gorishniy et al. ([2021](#bib.bib13)).
Here, (B) = {Covertype, Microsoft} and (A) contains all other datasets (except [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")).

| Parameter | Distribution or Value |
| --- | --- |
| # blocks | UniformInt​[1,4]UniformInt14\mathrm{UniformInt}[1,4] |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | UniformInt​[16,384]UniformInt16384\mathrm{UniformInt}[16,384] |
| Attention dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| FFN hidden dimension expansion rate | Uniform​[2/3,8/3]Uniform2383\mathrm{Uniform}[\nicefrac{{2}}{{3}},\nicefrac{{8}}{{3}}] |
| FFN dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Residual dropout rate | {0.0,Uniform​[0.0,0.2]}0.0Uniform0.00.2\{0.0,\mathrm{Uniform}[0.0,0.2]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 |

### D.11 ModernNCA

We adapted an official implementation of Ye et al. ([2024](#bib.bib48)). We have used periodic embeddings Gorishniy et al. ([2022](#bib.bib14)) (specifically, the lite version) for ModernNCA††\dagger and no embeddings for ModernNCA.

Table 9: 
The hyperparameter tuning space for ModernNCA.
Here, (C) = {[Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")}, (B) = {Covertype, Microsoft} and (A) contains all other datasets.

| Parameter | Distribution |
| --- | --- |
| # blocks | UniformInt​[0,2]UniformInt02\mathrm{UniformInt}[0,2] |
| db​l​o​c​ksubscript𝑑𝑏𝑙𝑜𝑐𝑘d\_{block} | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| dim | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Sample rate | Uniform​[0.05,0.6]Uniform0.050.6\mathrm{Uniform}[0.05,0.6] |
| Learning rate | LogUniform​[1​e​-​5,1​e​-​1]LogUniform1𝑒-51𝑒-1\mathrm{LogUniform}[1e\text{-}5,1e\text{-}1] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| # Tuning iterations | (A) 100 (B, C) 50 |




Table 10: 
The hyperparameter tuning space for ModernNCA††\dagger.
Here, (C) = {[Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")}, (B) = {Covertype, Microsoft} and (A) contains all other datasets.

| Parameter | Distribution |
| --- | --- |
| # blocks | UniformInt​[0,2]UniformInt02\mathrm{UniformInt}[0,2] |
| db​l​o​c​ksubscript𝑑𝑏𝑙𝑜𝑐𝑘d\_{block} | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| dim | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Sample rate | Uniform​[0.05,0.6]Uniform0.050.6\mathrm{Uniform}[0.05,0.6] |
| Learning rate | LogUniform​[1​e​-​5,1​e​-​1]LogUniform1𝑒-51𝑒-1\mathrm{LogUniform}[1e\text{-}5,1e\text{-}1] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| n\_frequencies | UniformInt​[16,96]UniformInt1696\mathrm{UniformInt}[16,96] |
| d\_embedding | UniformInt​[16,32]UniformInt1632\mathrm{UniformInt}[16,32] |
| frequency\_init\_scale | LogUniform​[0.01,10]LogUniform0.0110\mathrm{LogUniform}[0.01,10] |
| # Tuning iterations | (A) 100 (B, C) 50 |

### D.12 T2G-Former

We adapted the implementation and hyperparameters of Yan et al. ([2023](#bib.bib47)) from the official repository111https://github.com/jyansir/t2g-former. See [Table 11](#A4.T11 "Table 11 ‣ D.12 T2G-Former ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Table 11: The hyperparameter tuning space for T2G-Former Yan et al. ([2023](#bib.bib47)). Here, (C) = {all from [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")}, (B) = {Covertype, Microsoft} and (A) contains all other datasets. Also, we used 50 tuning iterations for some datasets from Grinsztajn et al. ([2022](#bib.bib16)).

|  |  |
| --- | --- |
| Parameter | Distribution or Value |
| # blocks | (A) UniformInt​[3,4]UniformInt34\mathrm{UniformInt}[3,4] (B, C) UniformInt​[1,3]UniformInt13\mathrm{UniformInt}[1,3] |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | UniformInt​[64,512]UniformInt64512\mathrm{UniformInt}[64,512] |
| Attention dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| FFN hidden dimension expansion rate | (A, B) Uniform​[2/3,8/3]Uniform2383\mathrm{Uniform}[\nicefrac{{2}}{{3}},\nicefrac{{8}}{{3}}] (C) 4/3434/3 |
| FFN dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Residual dropout rate | {0.0,Uniform​[0.0,0.2]}0.0Uniform0.00.2\{0.0,\mathrm{Uniform}[0.0,0.2]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Col. Learning rate | LogUniform​[5​e​-​3,5​e​-​2]LogUniform5𝑒-35𝑒-2\mathrm{LogUniform}[5e\text{-}3,5e\text{-}2] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​1]}0LogUniform1𝑒-61𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 (C) 25 |

### D.13 SAINT

We completely adapted hyperparameters and protocol from Gorishniy et al. ([2024](#bib.bib15)) to evaluate SAINT on Grinsztajn et al. ([2022](#bib.bib16)) benchmark. Results on datasets from [Table 4](#A3.T4 "Table 4 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") were directly taken from Gorishniy et al. ([2024](#bib.bib15)). Additional details can be found in Appendix.D from Gorishniy et al. ([2024](#bib.bib15)). We have used a default configuration on big datasets due to very high cost of tuning (see [Table 12](#A4.T12 "Table 12 ‣ D.13 SAINT ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling")).

Table 12: The default hyperparameters for SAINT (Somepalli et al., [2021](#bib.bib39)) on datasets from Rubachev et al. ([2024](#bib.bib38)).

|  |  |
| --- | --- |
| Parameter | Value |
| depth | 222 |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | 323232 |
| nh​e​a​d​ssubscript𝑛ℎ𝑒𝑎𝑑𝑠n\_{heads} | 444 |
| dh​e​a​dsubscript𝑑ℎ𝑒𝑎𝑑d\_{head} | 888 |
| Attention dropout rate | 0.10.10.1 |
| FFN hidden dimension expansion rate | 111 |
| FFN dropout rate | 0.80.80.8 |
| Learning rate | 1​e​-​41𝑒-41e\text{-}4 |
| Weight decay | 1​e​-​21𝑒-21e\text{-}2 |

### D.14 Excelformer

We adapted the implementation and hyperparameters of Chen et al. ([2023a](#bib.bib8)) from the official repository222https://github.com/WhatAShot/ExcelFormer.
For fair comparison with other models, we did not use the augmentation techniques from the paper in our expirements. See [Table 13](#A4.T13 "Table 13 ‣ D.14 Excelformer ‣ Appendix D Implementation details ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling").

Table 13: The hyperparameter tuning space for Excelformer Chen et al. ([2023a](#bib.bib8)). Here, (D) = {Homecredit, Maps Routing}, (C) = {[Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") w/o (D)}, (B) = {Covertype, Microsoft} and (A) contains all other datasets.

|  |  |
| --- | --- |
| Parameter | Distribution or Value |
| # blocks | (A, B) UniformInt​[2,5]UniformInt25\mathrm{UniformInt}[2,5] (C) UniformInt​[2,4]UniformInt24\mathrm{UniformInt}[2,4] (D) UniformInt​[1,3]UniformInt13\mathrm{UniformInt}[1,3] |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | (A, B) {32,64,128,256}3264128256\{32,64,128,256\} (C) {16,32,64}163264\{16,32,64\} (D) {4,8,16,32}481632\{4,8,16,32\} |
| nh​e​a​d​ssubscript𝑛ℎ𝑒𝑎𝑑𝑠n\_{heads} | (A,B) {4,8,16,32}481632\{4,8,16,32\} (C) {4,8,16}4816\{4,8,16\} (D) 444 |
| Attention dropout rate | 0.30.30.3 |
| FFN dropout rate | 0.00.00.0 |
| Residual dropout rate | Uniform​[0.0,0.5]Uniform0.00.5\mathrm{Uniform}[0.0,0.5] |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 (C, D) 25 |

### D.15 MLP

We used the implementation from the ”rtdl\_revisiting\_models” Python package (version 0.0.2) and ”rtdl\_num\_embeddings” Python package (version 0.0.10).

Table 14: The hyperparameter tuning space for MLP.

| Parameter | Distribution |
| --- | --- |
| # layers | UniformInt​[1,6]UniformInt16\mathrm{UniformInt}[1,6] |
| Width (hidden size) | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | 100 |




Table 15: The hyperparameter tuning space for MLP†superscriptMLP†\mathrm{MLP}^{\dagger} that uses piecewise-linear embeddings from Gorishniy et al. ([2022](#bib.bib14)).

| Parameter | Distribution |
| --- | --- |
| # layers | UniformInt​[1,5]UniformInt15\mathrm{UniformInt}[1,5] |
| Width (hidden size) | UniformInt​[64,1024]UniformInt641024\mathrm{UniformInt}[64,1024] |
| Dropout rate | {0.0,Uniform​[0.0,0.5]}0.0Uniform0.00.5\{0.0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| d\_embedding | UniformInt​[8,32]UniformInt832\mathrm{UniformInt}[8,32] |
| n\_bins | UniformInt​[2,128]UniformInt2128\mathrm{UniformInt}[2,128] |
| # Tuning iterations | 100 |

### D.16 CatBoost, XGBoost and LightGBM

Since our setup is directly taken from Gorishniy et al. ([2024](#bib.bib15)), we simply reused their results for GBDTs from the official repository333https://github.com/yandex-research/tabular-dl-tabr.
Importantly, in a series of preliminary experiments, we confirmed that those results are reproducible in our instance of their setup.
The details can be found in Appendix.D from Gorishniy et al. ([2024](#bib.bib15)). Results on datasets from [Table 5](#A3.T5 "Table 5 ‣ Appendix C Datasets ‣ TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling") were copied from the paper (Rubachev et al., [2024](#bib.bib38)).

### D.17 AutoInt

We used an implementation from Gorishniy et al. ([2021](#bib.bib13)) which is an adapted official implementation444https://github.com/shichence/AutoInt.

Table 16: The hyperparameter tuning space for AutoInt (Song et al., [2019](#bib.bib40)). Here, (B) = {Covertype, Microsoft} and (A) contains all other datasets.

|  |  |
| --- | --- |
| Parameter | Distribution |
| # blocks | UniformInt​[1,6]UniformInt16\mathrm{UniformInt}[1,6] |
| dt​o​k​e​nsubscript𝑑𝑡𝑜𝑘𝑒𝑛d\_{token} | UniformInt​[8,64]UniformInt864\mathrm{UniformInt}[8,64] |
| nh​e​a​d​ssubscript𝑛ℎ𝑒𝑎𝑑𝑠n\_{heads} | 2 |
| Attention dropout rate | {0,Uniform​[0.0,0.5]}0Uniform0.00.5\{0,\mathrm{Uniform}[0.0,0.5]\} |
| Embedding dropout rate | {0,Uniform​[0.0,0.5]}0Uniform0.00.5\{0,\mathrm{Uniform}[0.0,0.5]\} |
| Learning rate | LogUniform​[3​e​-​5,1​e​-​3]LogUniform3𝑒-51𝑒-3\mathrm{LogUniform}[3e\text{-}5,1e\text{-}3] |
| Weight decay | {0,LogUniform​[1​e​-​4,1​e​-​1]}0LogUniform1𝑒-41𝑒-1\{0,\mathrm{LogUniform}[1e\text{-}4,1e\text{-}1]\} |
| # Tuning iterations | (A) 100 (B) 50 |

#### D.17.1 TabPFN

Since TabPFN accepts only less than 10K training samples we use different subsamples of size 10K for different random seeds. Also, TabPFN is not applicable to regressions and datasets with more than 100100100 features.

## Appendix E Per-dataset results with standard deviations

Table 17: Extended results for the main benchmark. Results are grouped by datasets.

|  |  |
| --- | --- |
| churn ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8553±0.0029plus-or-minus0.85530.00290.8553\pm 0.0029 0.8582±0.0008plus-or-minus0.85820.00080.8582\pm 0.0008  TabPFNTabPFN\mathrm{TabPFN} – 0.8624±0.0008plus-or-minus0.86240.00080.8624\pm 0.0008  ResNetResNet\mathrm{ResNet} 0.8545±0.0044plus-or-minus0.85450.00440.8545\pm 0.0044 0.8565±0.0035plus-or-minus0.85650.00350.8565\pm 0.0035  DCN2DCN2\mathrm{DCN2} 0.8567±0.0020plus-or-minus0.85670.00200.8567\pm 0.0020 0.8570±0.0017plus-or-minus0.85700.00170.8570\pm 0.0017  SNNSNN\mathrm{SNN} 0.8506±0.0051plus-or-minus0.85060.00510.8506\pm 0.0051 0.8533±0.0033plus-or-minus0.85330.00330.8533\pm 0.0033  TromptTrompt\mathrm{Trompt} 0.8600±n​a​nplus-or-minus0.8600𝑛𝑎𝑛0.8600\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.8607±0.0047plus-or-minus0.86070.00470.8607\pm 0.0047 0.8622±0.0003plus-or-minus0.86220.00030.8622\pm 0.0003  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8592±0.0036plus-or-minus0.85920.00360.8592\pm 0.0036 0.8630±0.0005plus-or-minus0.86300.00050.8630\pm 0.0005  ExcelExcel\mathrm{Excel} 0.8618±0.0023plus-or-minus0.86180.00230.8618\pm 0.0023 0.8625±n​a​nplus-or-minus0.8625𝑛𝑎𝑛0.8625\pm nan  SAINTSAINT\mathrm{SAINT} 0.8603±0.0029plus-or-minus0.86030.00290.8603\pm 0.0029 0.8628±0.0008plus-or-minus0.86280.00080.8628\pm 0.0008  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8593±0.0028plus-or-minus0.85930.00280.8593\pm 0.0028 0.8598±0.0025plus-or-minus0.85980.00250.8598\pm 0.0025  T2GT2G\mathrm{T2G} 0.8613±0.0015plus-or-minus0.86130.00150.8613\pm 0.0015 –  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8624±0.0010plus-or-minus0.86240.00100.8624\pm 0.0010 0.8638±0.0012plus-or-minus0.86380.00120.8638\pm 0.0012  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8624±0.0026plus-or-minus0.86240.00260.8624\pm 0.0026 0.8640±0.0010plus-or-minus0.86400.00100.8640\pm 0.0010  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8580±0.0028plus-or-minus0.85800.00280.8580\pm 0.0028 0.8605±0.0018plus-or-minus0.86050.00180.8605\pm 0.0018  XGBoostXGBoost\mathrm{XGBoost} 0.8605±0.0022plus-or-minus0.86050.00220.8605\pm 0.0022 0.8608±0.0013plus-or-minus0.86080.00130.8608\pm 0.0013  LightGBMLightGBM\mathrm{LightGBM} 0.8600±0.0008plus-or-minus0.86000.00080.8600\pm 0.0008 0.8600±0.0000plus-or-minus0.86000.00000.8600\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.8582±0.0017plus-or-minus0.85820.00170.8582\pm 0.0017 0.8588±0.0008plus-or-minus0.85880.00080.8588\pm 0.0008  TabRTabR\mathrm{TabR} 0.8599±0.0025plus-or-minus0.85990.00250.8599\pm 0.0025 0.8620±0.0023plus-or-minus0.86200.00230.8620\pm 0.0023  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8625±0.0021plus-or-minus0.86250.00210.8625\pm 0.0021 –  MNCAMNCA\mathrm{MNCA} 0.8595±0.0028plus-or-minus0.85950.00280.8595\pm 0.0028 0.8615±0.0013plus-or-minus0.86150.00130.8615\pm 0.0013  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8606±0.0032plus-or-minus0.86060.00320.8606\pm 0.0032 0.8607±0.0008plus-or-minus0.86070.00080.8607\pm 0.0008  TabMTabM\mathrm{TabM} 0.8613±0.0025plus-or-minus0.86130.00250.8613\pm 0.0025 0.8615±0.0005plus-or-minus0.86150.00050.8615\pm 0.0005  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8611±0.0018plus-or-minus0.86110.00180.8611\pm 0.0018 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8625±0.0025plus-or-minus0.86250.00250.8625\pm 0.0025 0.8638±0.0021plus-or-minus0.86380.00210.8638\pm 0.0021  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8608±0.0019plus-or-minus0.86080.00190.8608\pm 0.0019 0.8592±0.0003plus-or-minus0.85920.00030.8592\pm 0.0003 | california ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.4948±0.0058plus-or-minus0.49480.00580.4948\pm 0.0058 0.4880±0.0022plus-or-minus0.48800.00220.4880\pm 0.0022  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.4915±0.0031plus-or-minus0.49150.00310.4915\pm 0.0031 0.4862±0.0017plus-or-minus0.48620.00170.4862\pm 0.0017  DCN2DCN2\mathrm{DCN2} 0.4971±0.0122plus-or-minus0.49710.01220.4971\pm 0.0122 0.4779±0.0022plus-or-minus0.47790.00220.4779\pm 0.0022  SNNSNN\mathrm{SNN} 0.5033±0.0075plus-or-minus0.50330.00750.5033\pm 0.0075 0.4933±0.0035plus-or-minus0.49330.00350.4933\pm 0.0035  TromptTrompt\mathrm{Trompt} 0.4579±n​a​nplus-or-minus0.4579𝑛𝑎𝑛0.4579\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.4682±0.0063plus-or-minus0.46820.00630.4682\pm 0.0063 0.4490±0.0028plus-or-minus0.44900.00280.4490\pm 0.0028  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.4746±0.0056plus-or-minus0.47460.00560.4746\pm 0.0056 0.4509±0.0029plus-or-minus0.45090.00290.4509\pm 0.0029  ExcelExcel\mathrm{Excel} 0.4544±0.0048plus-or-minus0.45440.00480.4544\pm 0.0048 0.4350±n​a​nplus-or-minus0.4350𝑛𝑎𝑛0.4350\pm nan  SAINTSAINT\mathrm{SAINT} 0.4680±0.0048plus-or-minus0.46800.00480.4680\pm 0.0048 0.4575±0.0014plus-or-minus0.45750.00140.4575\pm 0.0014  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.4635±0.0048plus-or-minus0.46350.00480.4635\pm 0.0048 0.4515±0.0016plus-or-minus0.45150.00160.4515\pm 0.0016  T2GT2G\mathrm{T2G} 0.4640±0.0100plus-or-minus0.46400.01000.4640\pm 0.0100 0.4462±n​a​nplus-or-minus0.4462𝑛𝑎𝑛0.4462\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.4652±0.0045plus-or-minus0.46520.00450.4652\pm 0.0045 0.4549±0.0006plus-or-minus0.45490.00060.4549\pm 0.0006  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.4597±0.0058plus-or-minus0.45970.00580.4597\pm 0.0058 0.4482±0.0026plus-or-minus0.44820.00260.4482\pm 0.0026  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.4530±0.0029plus-or-minus0.45300.00290.4530\pm 0.0029 0.4491±0.0010plus-or-minus0.44910.00100.4491\pm 0.0010  XGBoostXGBoost\mathrm{XGBoost} 0.4327±0.0016plus-or-minus0.43270.00160.4327\pm 0.0016 0.4316±0.0007plus-or-minus0.43160.00070.4316\pm 0.0007  LightGBMLightGBM\mathrm{LightGBM} 0.4352±0.0019plus-or-minus0.43520.00190.4352\pm 0.0019 0.4339±0.0008plus-or-minus0.43390.00080.4339\pm 0.0008  CatBoostCatBoost\mathrm{CatBoost} 0.4294±0.0012plus-or-minus0.42940.00120.4294\pm 0.0012 0.4265±0.0003plus-or-minus0.42650.00030.4265\pm 0.0003  TabRTabR\mathrm{TabR} 0.4030±0.0023plus-or-minus0.40300.00230.4030\pm 0.0023 0.3964±0.0013plus-or-minus0.39640.00130.3964\pm 0.0013  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.3998±0.0033plus-or-minus0.39980.00330.3998\pm 0.0033 –  MNCAMNCA\mathrm{MNCA} 0.4239±0.0012plus-or-minus0.42390.00120.4239\pm 0.0012 0.4231±0.0005plus-or-minus0.42310.00050.4231\pm 0.0005  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.4142±0.0031plus-or-minus0.41420.00310.4142\pm 0.0031 0.4071±0.0029plus-or-minus0.40710.00290.4071\pm 0.0029  TabMTabM\mathrm{TabM} 0.4509±0.0032plus-or-minus0.45090.00320.4509\pm 0.0032 0.4490±0.0018plus-or-minus0.44900.00180.4490\pm 0.0018  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.4507±0.0027plus-or-minus0.45070.00270.4507\pm 0.0027 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.4476±0.0036plus-or-minus0.44760.00360.4476\pm 0.0036 0.4425±0.0009plus-or-minus0.44250.00090.4425\pm 0.0009  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.4314±0.0036plus-or-minus0.43140.00360.4314\pm 0.0036 0.4261±0.0019plus-or-minus0.42610.00190.4261\pm 0.0019 |
| house ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 3.1117±0.0294plus-or-minus3.11170.02943.1117\pm 0.0294 3.0706±0.0140plus-or-minus3.07060.01403.0706\pm 0.0140  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 3.1143±0.0258plus-or-minus3.11430.02583.1143\pm 0.0258 3.0706±0.0098plus-or-minus3.07060.00983.0706\pm 0.0098  DCN2DCN2\mathrm{DCN2} 3.3327±0.0878plus-or-minus3.33270.08783.3327\pm 0.0878 3.1303±0.0410plus-or-minus3.13030.04103.1303\pm 0.0410  SNNSNN\mathrm{SNN} 3.2176±0.0376plus-or-minus3.21760.03763.2176\pm 0.0376 3.1320±0.0155plus-or-minus3.13200.01553.1320\pm 0.0155  TromptTrompt\mathrm{Trompt} 3.0638±n​a​nplus-or-minus3.0638𝑛𝑎𝑛3.0638\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 3.2157±0.0436plus-or-minus3.21570.04363.2157\pm 0.0436 3.1261±0.0095plus-or-minus3.12610.00953.1261\pm 0.0095  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 3.1871±0.0519plus-or-minus3.18710.05193.1871\pm 0.0519 3.0184±0.0086plus-or-minus3.01840.00863.0184\pm 0.0086  ExcelExcel\mathrm{Excel} 3.2460±0.0685plus-or-minus3.24600.06853.2460\pm 0.0685 3.1097±n​a​nplus-or-minus3.1097𝑛𝑎𝑛3.1097\pm nan  SAINTSAINT\mathrm{SAINT} 3.2424±0.0595plus-or-minus3.24240.05953.2424\pm 0.0595 3.1067±0.0253plus-or-minus3.10670.02533.1067\pm 0.0253  FT​-​TFT-T\mathrm{FT\texttt{-}T} 3.1823±0.0460plus-or-minus3.18230.04603.1823\pm 0.0460 3.0974±0.0334plus-or-minus3.09740.03343.0974\pm 0.0334  T2GT2G\mathrm{T2G} 3.1613±0.0320plus-or-minus3.16130.03203.1613\pm 0.0320 3.0982±n​a​nplus-or-minus3.0982𝑛𝑎𝑛3.0982\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 3.0633±0.0248plus-or-minus3.06330.02483.0633\pm 0.0248 3.0170±0.0070plus-or-minus3.01700.00703.0170\pm 0.0070  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 3.0775±0.0336plus-or-minus3.07750.03363.0775\pm 0.0336 3.0268±0.0170plus-or-minus3.02680.01703.0268\pm 0.0170  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 3.0999±0.0351plus-or-minus3.09990.03513.0999\pm 0.0351 3.0401±0.0071plus-or-minus3.04010.00713.0401\pm 0.0071  XGBoostXGBoost\mathrm{XGBoost} 3.1773±0.0102plus-or-minus3.17730.01023.1773\pm 0.0102 3.1644±0.0068plus-or-minus3.16440.00683.1644\pm 0.0068  LightGBMLightGBM\mathrm{LightGBM} 3.1774±0.0087plus-or-minus3.17740.00873.1774\pm 0.0087 3.1672±0.0050plus-or-minus3.16720.00503.1672\pm 0.0050  CatBoostCatBoost\mathrm{CatBoost} 3.1172±0.0125plus-or-minus3.11720.01253.1172\pm 0.0125 3.1058±0.0022plus-or-minus3.10580.00223.1058\pm 0.0022  TabRTabR\mathrm{TabR} 3.0667±0.0403plus-or-minus3.06670.04033.0667\pm 0.0403 2.9958±0.0270plus-or-minus2.99580.02702.9958\pm 0.0270  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 3.1048±0.0410plus-or-minus3.10480.04103.1048\pm 0.0410 –  MNCAMNCA\mathrm{MNCA} 3.0884±0.0286plus-or-minus3.08840.02863.0884\pm 0.0286 3.0538±0.0072plus-or-minus3.05380.00723.0538\pm 0.0072  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 3.0704±0.0388plus-or-minus3.07040.03883.0704\pm 0.0388 3.0149±0.0308plus-or-minus3.01490.03083.0149\pm 0.0308  TabMTabM\mathrm{TabM} 3.0002±0.0182plus-or-minus3.00020.01823.0002\pm 0.0182 2.9796±0.0024plus-or-minus2.97960.00242.9796\pm 0.0024  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 3.0156±0.0231plus-or-minus3.01560.02313.0156\pm 0.0231 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 3.0496±0.0225plus-or-minus3.04960.02253.0496\pm 0.0225 3.0225±0.0077plus-or-minus3.02250.00773.0225\pm 0.0077  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 2.9902±0.0271plus-or-minus2.99020.02712.9902\pm 0.0271 2.9648±0.0035plus-or-minus2.96480.00352.9648\pm 0.0035 | adult ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8540±0.0018plus-or-minus0.85400.00180.8540\pm 0.0018 0.8559±0.0011plus-or-minus0.85590.00110.8559\pm 0.0011  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.8554±0.0011plus-or-minus0.85540.00110.8554\pm 0.0011 0.8562±0.0006plus-or-minus0.85620.00060.8562\pm 0.0006  DCN2DCN2\mathrm{DCN2} 0.8582±0.0011plus-or-minus0.85820.00110.8582\pm 0.0011 0.8593±0.0002plus-or-minus0.85930.00020.8593\pm 0.0002  SNNSNN\mathrm{SNN} 0.8582±0.0009plus-or-minus0.85820.00090.8582\pm 0.0009 0.8603±0.0012plus-or-minus0.86030.00120.8603\pm 0.0012  TromptTrompt\mathrm{Trompt} 0.8590±n​a​nplus-or-minus0.8590𝑛𝑎𝑛0.8590\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.8592±0.0016plus-or-minus0.85920.00160.8592\pm 0.0016 0.8612±0.0004plus-or-minus0.86120.00040.8612\pm 0.0004  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8598±0.0013plus-or-minus0.85980.00130.8598\pm 0.0013 0.8617±0.0002plus-or-minus0.86170.00020.8617\pm 0.0002  ExcelExcel\mathrm{Excel} 0.8613±0.0024plus-or-minus0.86130.00240.8613\pm 0.0024 0.8641±n​a​nplus-or-minus0.8641𝑛𝑎𝑛0.8641\pm nan  SAINTSAINT\mathrm{SAINT} 0.8601±0.0019plus-or-minus0.86010.00190.8601\pm 0.0019 0.8618±0.0001plus-or-minus0.86180.00010.8618\pm 0.0001  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8588±0.0015plus-or-minus0.85880.00150.8588\pm 0.0015 0.8608±0.0011plus-or-minus0.86080.00110.8608\pm 0.0011  T2GT2G\mathrm{T2G} 0.8601±0.0011plus-or-minus0.86010.00110.8601\pm 0.0011 0.8622±n​a​nplus-or-minus0.8622𝑛𝑎𝑛0.8622\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8693±0.0007plus-or-minus0.86930.00070.8693\pm 0.0007 0.8702±0.0006plus-or-minus0.87020.00060.8702\pm 0.0006  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8694±0.0011plus-or-minus0.86940.00110.8694\pm 0.0011 0.8704±0.0008plus-or-minus0.87040.00080.8704\pm 0.0008  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8603±0.0009plus-or-minus0.86030.00090.8603\pm 0.0009 0.8616±0.0006plus-or-minus0.86160.00060.8616\pm 0.0006  XGBoostXGBoost\mathrm{XGBoost} 0.8720±0.0006plus-or-minus0.87200.00060.8720\pm 0.0006 0.8723±0.0002plus-or-minus0.87230.00020.8723\pm 0.0002  LightGBMLightGBM\mathrm{LightGBM} 0.8713±0.0007plus-or-minus0.87130.00070.8713\pm 0.0007 0.8721±0.0004plus-or-minus0.87210.00040.8721\pm 0.0004  CatBoostCatBoost\mathrm{CatBoost} 0.8714±0.0012plus-or-minus0.87140.00120.8714\pm 0.0012 0.8723±0.0007plus-or-minus0.87230.00070.8723\pm 0.0007  TabRTabR\mathrm{TabR} 0.8646±0.0022plus-or-minus0.86460.00220.8646\pm 0.0022 0.8680±0.0019plus-or-minus0.86800.00190.8680\pm 0.0019  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8699±0.0011plus-or-minus0.86990.00110.8699\pm 0.0011 –  MNCAMNCA\mathrm{MNCA} 0.8677±0.0018plus-or-minus0.86770.00180.8677\pm 0.0018 0.8696±0.0003plus-or-minus0.86960.00030.8696\pm 0.0003  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8717±0.0008plus-or-minus0.87170.00080.8717\pm 0.0008 0.8742±0.0006plus-or-minus0.87420.00060.8742\pm 0.0006  TabMTabM\mathrm{TabM} 0.8582±0.0011plus-or-minus0.85820.00110.8582\pm 0.0011 0.8588±0.0003plus-or-minus0.85880.00030.8588\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8577±0.0009plus-or-minus0.85770.00090.8577\pm 0.0009 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8584±0.0010plus-or-minus0.85840.00100.8584\pm 0.0010 0.8591±0.0005plus-or-minus0.85910.00050.8591\pm 0.0005  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8679±0.0017plus-or-minus0.86790.00170.8679\pm 0.0017 0.8690±0.0005plus-or-minus0.86900.00050.8690\pm 0.0005 |
| diamond ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.1404±0.0012plus-or-minus0.14040.00120.1404\pm 0.0012 0.1362±0.0003plus-or-minus0.13620.00030.1362\pm 0.0003  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.1396±0.0029plus-or-minus0.13960.00290.1396\pm 0.0029 0.1361±0.0011plus-or-minus0.13610.00110.1361\pm 0.0011  DCN2DCN2\mathrm{DCN2} 0.1420±0.0032plus-or-minus0.14200.00320.1420\pm 0.0032 0.1374±0.0020plus-or-minus0.13740.00200.1374\pm 0.0020  SNNSNN\mathrm{SNN} 0.1473±0.0057plus-or-minus0.14730.00570.1473\pm 0.0057 0.1424±0.0008plus-or-minus0.14240.00080.1424\pm 0.0008  TromptTrompt\mathrm{Trompt} 0.1391±n​a​nplus-or-minus0.1391𝑛𝑎𝑛0.1391\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.1392±0.0014plus-or-minus0.13920.00140.1392\pm 0.0014 0.1361±0.0004plus-or-minus0.13610.00040.1361\pm 0.0004  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.1400±0.0025plus-or-minus0.14000.00250.1400\pm 0.0025 0.1378±0.0008plus-or-minus0.13780.00080.1378\pm 0.0008  ExcelExcel\mathrm{Excel} 0.1766±0.0023plus-or-minus0.17660.00230.1766\pm 0.0023 0.1712±n​a​nplus-or-minus0.1712𝑛𝑎𝑛0.1712\pm nan  SAINTSAINT\mathrm{SAINT} 0.1369±0.0019plus-or-minus0.13690.00190.1369\pm 0.0019 0.1343±0.0011plus-or-minus0.13430.00110.1343\pm 0.0011  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.1376±0.0013plus-or-minus0.13760.00130.1376\pm 0.0013 0.1360±0.0002plus-or-minus0.13600.00020.1360\pm 0.0002  T2GT2G\mathrm{T2G} 0.1372±0.0011plus-or-minus0.13720.00110.1372\pm 0.0011 0.1346±n​a​nplus-or-minus0.1346𝑛𝑎𝑛0.1346\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.1342±0.0008plus-or-minus0.13420.00080.1342\pm 0.0008 0.1325±0.0004plus-or-minus0.13250.00040.1325\pm 0.0004  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.1337±0.0010plus-or-minus0.13370.00100.1337\pm 0.0010 0.1317±0.0003plus-or-minus0.13170.00030.1317\pm 0.0003  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.1323±0.0010plus-or-minus0.13230.00100.1323\pm 0.0010 0.1301±0.0005plus-or-minus0.13010.00050.1301\pm 0.0005  XGBoostXGBoost\mathrm{XGBoost} 0.1368±0.0004plus-or-minus0.13680.00040.1368\pm 0.0004 0.1363±0.0001plus-or-minus0.13630.00010.1363\pm 0.0001  LightGBMLightGBM\mathrm{LightGBM} 0.1359±0.0002plus-or-minus0.13590.00020.1359\pm 0.0002 0.1358±0.0001plus-or-minus0.13580.00010.1358\pm 0.0001  CatBoostCatBoost\mathrm{CatBoost} 0.1335±0.0006plus-or-minus0.13350.00060.1335\pm 0.0006 0.1327±0.0004plus-or-minus0.13270.00040.1327\pm 0.0004  TabRTabR\mathrm{TabR} 0.1327±0.0010plus-or-minus0.13270.00100.1327\pm 0.0010 0.1311±0.0005plus-or-minus0.13110.00050.1311\pm 0.0005  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.1333±0.0013plus-or-minus0.13330.00130.1333\pm 0.0013 –  MNCAMNCA\mathrm{MNCA} 0.1370±0.0018plus-or-minus0.13700.00180.1370\pm 0.0018 0.1348±0.0005plus-or-minus0.13480.00050.1348\pm 0.0005  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.1327±0.0012plus-or-minus0.13270.00120.1327\pm 0.0012 0.1315±0.0006plus-or-minus0.13150.00060.1315\pm 0.0006  TabMTabM\mathrm{TabM} 0.1342±0.0017plus-or-minus0.13420.00170.1342\pm 0.0017 0.1327±0.0004plus-or-minus0.13270.00040.1327\pm 0.0004  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.1340±0.0014plus-or-minus0.13400.00140.1340\pm 0.0014 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.1367±0.0015plus-or-minus0.13670.00150.1367\pm 0.0015 0.1352±0.0008plus-or-minus0.13520.00080.1352\pm 0.0008  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.1320±0.0010plus-or-minus0.13200.00100.1320\pm 0.0010 0.1307±0.0005plus-or-minus0.13070.00050.1307\pm 0.0005 | otto ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8175±0.0022plus-or-minus0.81750.00220.8175\pm 0.0022 0.8222±0.0007plus-or-minus0.82220.00070.8222\pm 0.0007  TabPFNTabPFN\mathrm{TabPFN} – 0.7408±0.0028plus-or-minus0.74080.00280.7408\pm 0.0028  ResNetResNet\mathrm{ResNet} 0.8174±0.0021plus-or-minus0.81740.00210.8174\pm 0.0021 0.8198±0.0006plus-or-minus0.81980.00060.8198\pm 0.0006  DCN2DCN2\mathrm{DCN2} 0.8064±0.0021plus-or-minus0.80640.00210.8064\pm 0.0021 0.8208±0.0023plus-or-minus0.82080.00230.8208\pm 0.0023  SNNSNN\mathrm{SNN} 0.8087±0.0020plus-or-minus0.80870.00200.8087\pm 0.0020 0.8156±0.0013plus-or-minus0.81560.00130.8156\pm 0.0013  TromptTrompt\mathrm{Trompt} 0.8093±n​a​nplus-or-minus0.8093𝑛𝑎𝑛0.8093\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.8050±0.0034plus-or-minus0.80500.00340.8050\pm 0.0034 0.8111±0.0020plus-or-minus0.81110.00200.8111\pm 0.0020  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8092±0.0040plus-or-minus0.80920.00400.8092\pm 0.0040 0.8136±0.0010plus-or-minus0.81360.00100.8136\pm 0.0010  ExcelExcel\mathrm{Excel} 0.8102±0.0022plus-or-minus0.81020.00220.8102\pm 0.0022 0.8220±n​a​nplus-or-minus0.8220𝑛𝑎𝑛0.8220\pm nan  SAINTSAINT\mathrm{SAINT} 0.8119±0.0018plus-or-minus0.81190.00180.8119\pm 0.0018 0.8193±0.0024plus-or-minus0.81930.00240.8193\pm 0.0024  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8133±0.0033plus-or-minus0.81330.00330.8133\pm 0.0033 0.8221±0.0013plus-or-minus0.82210.00130.8221\pm 0.0013  T2GT2G\mathrm{T2G} 0.8161±0.0019plus-or-minus0.81610.00190.8161\pm 0.0019 0.8272±n​a​nplus-or-minus0.8272𝑛𝑎𝑛0.8272\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8190±0.0021plus-or-minus0.81900.00210.8190\pm 0.0021 0.8271±0.0015plus-or-minus0.82710.00150.8271\pm 0.0015  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8189±0.0015plus-or-minus0.81890.00150.8189\pm 0.0015 0.8253±0.0000plus-or-minus0.82530.00000.8253\pm 0.0000  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8205±0.0021plus-or-minus0.82050.00210.8205\pm 0.0021 0.8290±0.0006plus-or-minus0.82900.00060.8290\pm 0.0006  XGBoostXGBoost\mathrm{XGBoost} 0.8297±0.0011plus-or-minus0.82970.00110.8297\pm 0.0011 0.8316±0.0008plus-or-minus0.83160.00080.8316\pm 0.0008  LightGBMLightGBM\mathrm{LightGBM} 0.8302±0.0009plus-or-minus0.83020.00090.8302\pm 0.0009 0.8316±0.0013plus-or-minus0.83160.00130.8316\pm 0.0013  CatBoostCatBoost\mathrm{CatBoost} 0.8250±0.0013plus-or-minus0.82500.00130.8250\pm 0.0013 0.8268±0.0002plus-or-minus0.82680.00020.8268\pm 0.0002  TabRTabR\mathrm{TabR} 0.8179±0.0022plus-or-minus0.81790.00220.8179\pm 0.0022 0.8236±0.0009plus-or-minus0.82360.00090.8236\pm 0.0009  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8246±0.0018plus-or-minus0.82460.00180.8246\pm 0.0018 –  MNCAMNCA\mathrm{MNCA} 0.8275±0.0012plus-or-minus0.82750.00120.8275\pm 0.0012 0.8313±0.0006plus-or-minus0.83130.00060.8313\pm 0.0006  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8265±0.0015plus-or-minus0.82650.00150.8265\pm 0.0015 0.8304±0.0006plus-or-minus0.83040.00060.8304\pm 0.0006  TabMTabM\mathrm{TabM} 0.8268±0.0014plus-or-minus0.82680.00140.8268\pm 0.0014 0.8300±0.0007plus-or-minus0.83000.00070.8300\pm 0.0007  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8204±0.0025plus-or-minus0.82040.00250.8204\pm 0.0025 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8267±0.0011plus-or-minus0.82670.00110.8267\pm 0.0011 0.8298±0.0008plus-or-minus0.82980.00080.8298\pm 0.0008  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8342±0.0014plus-or-minus0.83420.00140.8342\pm 0.0014 0.8365±0.0005plus-or-minus0.83650.00050.8365\pm 0.0005 |
| higgs-small ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7180±0.0027plus-or-minus0.71800.00270.7180\pm 0.0027 0.7192±0.0005plus-or-minus0.71920.00050.7192\pm 0.0005  TabPFNTabPFN\mathrm{TabPFN} – 0.6727±0.0034plus-or-minus0.67270.00340.6727\pm 0.0034  ResNetResNet\mathrm{ResNet} 0.7256±0.0020plus-or-minus0.72560.00200.7256\pm 0.0020 0.7307±0.0001plus-or-minus0.73070.00010.7307\pm 0.0001  DCN2DCN2\mathrm{DCN2} 0.7164±0.0030plus-or-minus0.71640.00300.7164\pm 0.0030 0.7237±0.0011plus-or-minus0.72370.00110.7237\pm 0.0011  SNNSNN\mathrm{SNN} 0.7142±0.0024plus-or-minus0.71420.00240.7142\pm 0.0024 0.7171±0.0020plus-or-minus0.71710.00200.7171\pm 0.0020  TromptTrompt\mathrm{Trompt} 0.7262±n​a​nplus-or-minus0.7262𝑛𝑎𝑛0.7262\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.7240±0.0028plus-or-minus0.72400.00280.7240\pm 0.0028 0.7287±0.0008plus-or-minus0.72870.00080.7287\pm 0.0008  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7248±0.0023plus-or-minus0.72480.00230.7248\pm 0.0023 0.7334±0.0007plus-or-minus0.73340.00070.7334\pm 0.0007  ExcelExcel\mathrm{Excel} 0.7262±0.0017plus-or-minus0.72620.00170.7262\pm 0.0017 0.7329±n​a​nplus-or-minus0.7329𝑛𝑎𝑛0.7329\pm nan  SAINTSAINT\mathrm{SAINT} 0.7236±0.0019plus-or-minus0.72360.00190.7236\pm 0.0019 0.7295±0.0011plus-or-minus0.72950.00110.7295\pm 0.0011  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7281±0.0016plus-or-minus0.72810.00160.7281\pm 0.0016 0.7334±0.0013plus-or-minus0.73340.00130.7334\pm 0.0013  T2GT2G\mathrm{T2G} 0.7352±0.0037plus-or-minus0.73520.00370.7352\pm 0.0037 0.7400±n​a​nplus-or-minus0.7400𝑛𝑎𝑛0.7400\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7260±0.0017plus-or-minus0.72600.00170.7260\pm 0.0017 0.7304±0.0008plus-or-minus0.73040.00080.7304\pm 0.0008  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7261±0.0010plus-or-minus0.72610.00100.7261\pm 0.0010 0.7270±0.0003plus-or-minus0.72700.00030.7270\pm 0.0003  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7210±0.0016plus-or-minus0.72100.00160.7210\pm 0.0016 0.7252±0.0005plus-or-minus0.72520.00050.7252\pm 0.0005  XGBoostXGBoost\mathrm{XGBoost} 0.7246±0.0015plus-or-minus0.72460.00150.7246\pm 0.0015 0.7264±0.0013plus-or-minus0.72640.00130.7264\pm 0.0013  LightGBMLightGBM\mathrm{LightGBM} 0.7256±0.0009plus-or-minus0.72560.00090.7256\pm 0.0009 0.7263±0.0007plus-or-minus0.72630.00070.7263\pm 0.0007  CatBoostCatBoost\mathrm{CatBoost} 0.7260±0.0011plus-or-minus0.72600.00110.7260\pm 0.0011 0.7273±0.0010plus-or-minus0.72730.00100.7273\pm 0.0010  TabRTabR\mathrm{TabR} 0.7223±0.0010plus-or-minus0.72230.00100.7223\pm 0.0010 0.7257±0.0008plus-or-minus0.72570.00080.7257\pm 0.0008  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7294±0.0014plus-or-minus0.72940.00140.7294\pm 0.0014 –  MNCAMNCA\mathrm{MNCA} 0.7263±0.0023plus-or-minus0.72630.00230.7263\pm 0.0023 0.7292±0.0006plus-or-minus0.72920.00060.7292\pm 0.0006  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7300±0.0020plus-or-minus0.73000.00200.7300\pm 0.0020 0.7348±0.0008plus-or-minus0.73480.00080.7348\pm 0.0008  TabMTabM\mathrm{TabM} 0.7383±0.0028plus-or-minus0.73830.00280.7383\pm 0.0028 0.7409±0.0010plus-or-minus0.74090.00100.7409\pm 0.0010  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7372±0.0021plus-or-minus0.73720.00210.7372\pm 0.0021 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7344±0.0016plus-or-minus0.73440.00160.7344\pm 0.0016 0.7366±0.0012plus-or-minus0.73660.00120.7366\pm 0.0012  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.7348±0.0017plus-or-minus0.73480.00170.7348\pm 0.0017 0.7379±0.0006plus-or-minus0.73790.00060.7379\pm 0.0006 | black-friday ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.6955±0.0004plus-or-minus0.69550.00040.6955\pm 0.0004 0.6942±0.0002plus-or-minus0.69420.00020.6942\pm 0.0002  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.6929±0.0008plus-or-minus0.69290.00080.6929\pm 0.0008 0.6907±0.0002plus-or-minus0.69070.00020.6907\pm 0.0002  DCN2DCN2\mathrm{DCN2} 0.6968±0.0013plus-or-minus0.69680.00130.6968\pm 0.0013 0.6936±0.0007plus-or-minus0.69360.00070.6936\pm 0.0007  SNNSNN\mathrm{SNN} 0.6996±0.0013plus-or-minus0.69960.00130.6996\pm 0.0013 0.6978±0.0004plus-or-minus0.69780.00040.6978\pm 0.0004  TromptTrompt\mathrm{Trompt} 0.6983±n​a​nplus-or-minus0.6983𝑛𝑎𝑛0.6983\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.6994±0.0082plus-or-minus0.69940.00820.6994\pm 0.0082 0.6927±0.0021plus-or-minus0.69270.00210.6927\pm 0.0021  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.6905±0.0021plus-or-minus0.69050.00210.6905\pm 0.0021 0.6851±0.0011plus-or-minus0.68510.00110.6851\pm 0.0011  ExcelExcel\mathrm{Excel} 0.6947±0.0016plus-or-minus0.69470.00160.6947\pm 0.0016 0.6908±n​a​nplus-or-minus0.6908𝑛𝑎𝑛0.6908\pm nan  SAINTSAINT\mathrm{SAINT} 0.6934±0.0009plus-or-minus0.69340.00090.6934\pm 0.0009 0.6879±0.0006plus-or-minus0.68790.00060.6879\pm 0.0006  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.6987±0.0192plus-or-minus0.69870.01920.6987\pm 0.0192 0.6879±0.0023plus-or-minus0.68790.00230.6879\pm 0.0023  T2GT2G\mathrm{T2G} 0.6887±0.0046plus-or-minus0.68870.00460.6887\pm 0.0046 0.6832±n​a​nplus-or-minus0.6832𝑛𝑎𝑛0.6832\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.6849±0.0006plus-or-minus0.68490.00060.6849\pm 0.0006 0.6824±0.0002plus-or-minus0.68240.00020.6824\pm 0.0002  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.6857±0.0004plus-or-minus0.68570.00040.6857\pm 0.0004 0.6838±0.0002plus-or-minus0.68380.00020.6838\pm 0.0002  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.6836±0.0006plus-or-minus0.68360.00060.6836\pm 0.0006 0.6812±0.0002plus-or-minus0.68120.00020.6812\pm 0.0002  XGBoostXGBoost\mathrm{XGBoost} 0.6806±0.0001plus-or-minus0.68060.00010.6806\pm 0.0001 0.6805±0.0000plus-or-minus0.68050.00000.6805\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.6799±0.0003plus-or-minus0.67990.00030.6799\pm 0.0003 0.6795±0.0001plus-or-minus0.67950.00010.6795\pm 0.0001  CatBoostCatBoost\mathrm{CatBoost} 0.6822±0.0003plus-or-minus0.68220.00030.6822\pm 0.0003 0.6813±0.0002plus-or-minus0.68130.00020.6813\pm 0.0002  TabRTabR\mathrm{TabR} 0.6899±0.0004plus-or-minus0.68990.00040.6899\pm 0.0004 0.6883±0.0002plus-or-minus0.68830.00020.6883\pm 0.0002  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.6761±0.0009plus-or-minus0.67610.00090.6761\pm 0.0009 –  MNCAMNCA\mathrm{MNCA} 0.6893±0.0004plus-or-minus0.68930.00040.6893\pm 0.0004 0.6883±0.0000plus-or-minus0.68830.00000.6883\pm 0.0000  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.6885±0.0007plus-or-minus0.68850.00070.6885\pm 0.0007 0.6863±0.0003plus-or-minus0.68630.00030.6863\pm 0.0003  TabMTabM\mathrm{TabM} 0.6875±0.0015plus-or-minus0.68750.00150.6875\pm 0.0015 0.6866±0.0003plus-or-minus0.68660.00030.6866\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.6870±0.0014plus-or-minus0.68700.00140.6870\pm 0.0014 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.6865±0.0016plus-or-minus0.68650.00160.6865\pm 0.0016 0.6843±0.0005plus-or-minus0.68430.00050.6843\pm 0.0005  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.6807±0.0013plus-or-minus0.68070.00130.6807\pm 0.0013 0.6783±0.0009plus-or-minus0.67830.00090.6783\pm 0.0009 |
| covtype2 ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.9630±0.0012plus-or-minus0.96300.00120.9630\pm 0.0012 0.9664±0.0004plus-or-minus0.96640.00040.9664\pm 0.0004  TabPFNTabPFN\mathrm{TabPFN} – 0.7606±0.0022plus-or-minus0.76060.00220.7606\pm 0.0022  ResNetResNet\mathrm{ResNet} 0.9638±0.0005plus-or-minus0.96380.00050.9638\pm 0.0005 0.9685±0.0003plus-or-minus0.96850.00030.9685\pm 0.0003  DCN2DCN2\mathrm{DCN2} 0.9622±0.0019plus-or-minus0.96220.00190.9622\pm 0.0019 0.9673±0.0011plus-or-minus0.96730.00110.9673\pm 0.0011  SNNSNN\mathrm{SNN} 0.9636±0.0010plus-or-minus0.96360.00100.9636\pm 0.0010 0.9677±0.0002plus-or-minus0.96770.00020.9677\pm 0.0002  TromptTrompt\mathrm{Trompt} 0.9286±n​a​nplus-or-minus0.9286𝑛𝑎𝑛0.9286\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.9614±0.0016plus-or-minus0.96140.00160.9614\pm 0.0016 0.9696±0.0005plus-or-minus0.96960.00050.9696\pm 0.0005  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.9663±0.0019plus-or-minus0.96630.00190.9663\pm 0.0019 0.9699±0.0014plus-or-minus0.96990.00140.9699\pm 0.0014  ExcelExcel\mathrm{Excel} 0.9606±0.0018plus-or-minus0.96060.00180.9606\pm 0.0018 0.9670±n​a​nplus-or-minus0.9670𝑛𝑎𝑛0.9670\pm nan  SAINTSAINT\mathrm{SAINT} 0.9669±0.0010plus-or-minus0.96690.00100.9669\pm 0.0010 0.9725±n​a​nplus-or-minus0.9725𝑛𝑎𝑛0.9725\pm nan  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.9698±0.0008plus-or-minus0.96980.00080.9698\pm 0.0008 0.9731±0.0006plus-or-minus0.97310.00060.9731\pm 0.0006  T2GT2G\mathrm{T2G} 0.9668±0.0008plus-or-minus0.96680.00080.9668\pm 0.0008 0.9708±n​a​nplus-or-minus0.9708𝑛𝑎𝑛0.9708\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.9690±0.0008plus-or-minus0.96900.00080.9690\pm 0.0008 0.9721±0.0006plus-or-minus0.97210.00060.9721\pm 0.0006  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.9713±0.0006plus-or-minus0.97130.00060.9713\pm 0.0006 0.9758±0.0000plus-or-minus0.97580.00000.9758\pm 0.0000  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.9697±0.0008plus-or-minus0.96970.00080.9697\pm 0.0008 0.9721±0.0005plus-or-minus0.97210.00050.9721\pm 0.0005  XGBoostXGBoost\mathrm{XGBoost} 0.9710±0.0002plus-or-minus0.97100.00020.9710\pm 0.0002 0.9713±0.0000plus-or-minus0.97130.00000.9713\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.9709±0.0003plus-or-minus0.97090.00030.9709\pm 0.0003 –  CatBoostCatBoost\mathrm{CatBoost} 0.9670±0.0003plus-or-minus0.96700.00030.9670\pm 0.0003 0.9680±0.0002plus-or-minus0.96800.00020.9680\pm 0.0002  TabRTabR\mathrm{TabR} 0.9737±0.0005plus-or-minus0.97370.00050.9737\pm 0.0005 0.9745±0.0006plus-or-minus0.97450.00060.9745\pm 0.0006  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.9752±0.0003plus-or-minus0.97520.00030.9752\pm 0.0003 –  MNCAMNCA\mathrm{MNCA} 0.9724±0.0003plus-or-minus0.97240.00030.9724\pm 0.0003 0.9729±0.0001plus-or-minus0.97290.00010.9729\pm 0.0001  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.9747±0.0002plus-or-minus0.97470.00020.9747\pm 0.0002 0.9747±0.0002plus-or-minus0.97470.00020.9747\pm 0.0002  TabMTabM\mathrm{TabM} 0.9712±0.0008plus-or-minus0.97120.00080.9712\pm 0.0008 0.9729±0.0003plus-or-minus0.97290.00030.9729\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.9707±0.0008plus-or-minus0.97070.00080.9707\pm 0.0008 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.9693±0.0008plus-or-minus0.96930.00080.9693\pm 0.0008 0.9713±0.0001plus-or-minus0.97130.00010.9713\pm 0.0001  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.9740±0.0006plus-or-minus0.97400.00060.9740\pm 0.0006 0.9754±0.0001plus-or-minus0.97540.00010.9754\pm 0.0001 | microsoft ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7475±0.0003plus-or-minus0.74750.00030.7475\pm 0.0003 0.7460±0.0003plus-or-minus0.74600.00030.7460\pm 0.0003  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.7472±0.0004plus-or-minus0.74720.00040.7472\pm 0.0004 0.7452±0.0004plus-or-minus0.74520.00040.7452\pm 0.0004  DCN2DCN2\mathrm{DCN2} 0.7499±0.0003plus-or-minus0.74990.00030.7499\pm 0.0003 0.7477±0.0001plus-or-minus0.74770.00010.7477\pm 0.0001  SNNSNN\mathrm{SNN} 0.7488±0.0004plus-or-minus0.74880.00040.7488\pm 0.0004 0.7470±0.0001plus-or-minus0.74700.00010.7470\pm 0.0001  TromptTrompt\mathrm{Trompt} 0.7476±n​a​nplus-or-minus0.7476𝑛𝑎𝑛0.7476\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.7482±0.0005plus-or-minus0.74820.00050.7482\pm 0.0005 0.7455±0.0002plus-or-minus0.74550.00020.7455\pm 0.0002  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7482±0.0008plus-or-minus0.74820.00080.7482\pm 0.0008 0.7436±0.0001plus-or-minus0.74360.00010.7436\pm 0.0001  ExcelExcel\mathrm{Excel} 0.7479±0.0007plus-or-minus0.74790.00070.7479\pm 0.0007 0.7442±n​a​nplus-or-minus0.7442𝑛𝑎𝑛0.7442\pm nan  SAINTSAINT\mathrm{SAINT} 0.7625±0.0066plus-or-minus0.76250.00660.7625\pm 0.0066 –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7460±0.0007plus-or-minus0.74600.00070.7460\pm 0.0007 0.7422±0.0004plus-or-minus0.74220.00040.7422\pm 0.0004  T2GT2G\mathrm{T2G} 0.7460±0.0006plus-or-minus0.74600.00060.7460\pm 0.0006 0.7427±n​a​nplus-or-minus0.7427𝑛𝑎𝑛0.7427\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7446±0.0002plus-or-minus0.74460.00020.7446\pm 0.0002 0.7434±0.0002plus-or-minus0.74340.00020.7434\pm 0.0002  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7444±0.0003plus-or-minus0.74440.00030.7444\pm 0.0003 0.7429±0.0001plus-or-minus0.74290.00010.7429\pm 0.0001  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7465±0.0005plus-or-minus0.74650.00050.7465\pm 0.0005 0.7448±0.0001plus-or-minus0.74480.00010.7448\pm 0.0001  XGBoostXGBoost\mathrm{XGBoost} 0.7413±0.0001plus-or-minus0.74130.00010.7413\pm 0.0001 0.7410±0.0000plus-or-minus0.74100.00000.7410\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.7417±0.0001plus-or-minus0.74170.00010.7417\pm 0.0001 0.7413±0.0000plus-or-minus0.74130.00000.7413\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.7412±0.0001plus-or-minus0.74120.00010.7412\pm 0.0001 0.7406±0.0000plus-or-minus0.74060.00000.7406\pm 0.0000  TabRTabR\mathrm{TabR} 0.7503±0.0006plus-or-minus0.75030.00060.7503\pm 0.0006 0.7485±0.0002plus-or-minus0.74850.00020.7485\pm 0.0002  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7501±0.0005plus-or-minus0.75010.00050.7501\pm 0.0005 –  MNCAMNCA\mathrm{MNCA} 0.7458±0.0003plus-or-minus0.74580.00030.7458\pm 0.0003 0.7448±0.0002plus-or-minus0.74480.00020.7448\pm 0.0002  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7460±0.0008plus-or-minus0.74600.00080.7460\pm 0.0008 0.7435±0.0004plus-or-minus0.74350.00040.7435\pm 0.0004  TabMTabM\mathrm{TabM} 0.7434±0.0003plus-or-minus0.74340.00030.7434\pm 0.0003 0.7424±0.0001plus-or-minus0.74240.00010.7424\pm 0.0001  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7435±0.0003plus-or-minus0.74350.00030.7435\pm 0.0003 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7444±0.0003plus-or-minus0.74440.00030.7444\pm 0.0003 0.7431±0.0002plus-or-minus0.74310.00020.7431\pm 0.0002  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.7427±0.0002plus-or-minus0.74270.00020.7427\pm 0.0002 0.7416±0.0002plus-or-minus0.74160.00020.7416\pm 0.0002 |




Table 18: Extended results for the main benchmark. Results are grouped by datasets.

|  |  |
| --- | --- |
| wine ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7778±0.0153plus-or-minus0.77780.01530.7778\pm 0.0153 0.7907±0.0117plus-or-minus0.79070.01170.7907\pm 0.0117  TabPFNTabPFN\mathrm{TabPFN} – 0.7908±0.0063plus-or-minus0.79080.00630.7908\pm 0.0063  ResNetResNet\mathrm{ResNet} 0.7710±0.0137plus-or-minus0.77100.01370.7710\pm 0.0137 0.7839±0.0083plus-or-minus0.78390.00830.7839\pm 0.0083  DCN2DCN2\mathrm{DCN2} 0.7492±0.0147plus-or-minus0.74920.01470.7492\pm 0.0147 0.7764±0.0095plus-or-minus0.77640.00950.7764\pm 0.0095  SNNSNN\mathrm{SNN} 0.7818±0.0143plus-or-minus0.78180.01430.7818\pm 0.0143 0.7994±0.0097plus-or-minus0.79940.00970.7994\pm 0.0097  TromptTrompt\mathrm{Trompt} 0.7818±0.0081plus-or-minus0.78180.00810.7818\pm 0.0081 –  AutoIntAutoInt\mathrm{AutoInt} 0.7745±0.0144plus-or-minus0.77450.01440.7745\pm 0.0144 0.7909±0.0160plus-or-minus0.79090.01600.7909\pm 0.0160  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7769±0.0149plus-or-minus0.77690.01490.7769\pm 0.0149 0.7950±0.0087plus-or-minus0.79500.00870.7950\pm 0.0087  ExcelExcel\mathrm{Excel} 0.7631±0.0171plus-or-minus0.76310.01710.7631\pm 0.0171 0.7765±0.0121plus-or-minus0.77650.01210.7765\pm 0.0121  SAINTSAINT\mathrm{SAINT} 0.7684±0.0144plus-or-minus0.76840.01440.7684\pm 0.0144 0.7821±0.0105plus-or-minus0.78210.01050.7821\pm 0.0105  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7755±0.0133plus-or-minus0.77550.01330.7755\pm 0.0133 0.7894±0.0083plus-or-minus0.78940.00830.7894\pm 0.0083  T2GT2G\mathrm{T2G} 0.7733±0.0118plus-or-minus0.77330.01180.7733\pm 0.0118 0.7933±0.0137plus-or-minus0.79330.01370.7933\pm 0.0137  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7803±0.0157plus-or-minus0.78030.01570.7803\pm 0.0157 0.7964±0.0146plus-or-minus0.79640.01460.7964\pm 0.0146  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7733±0.0185plus-or-minus0.77330.01850.7733\pm 0.0185 0.7856±0.0160plus-or-minus0.78560.01600.7856\pm 0.0160  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7814±0.0132plus-or-minus0.78140.01320.7814\pm 0.0132 0.7919±0.0098plus-or-minus0.79190.00980.7919\pm 0.0098  XGBoostXGBoost\mathrm{XGBoost} 0.7949±0.0178plus-or-minus0.79490.01780.7949\pm 0.0178 0.8010±0.0186plus-or-minus0.80100.01860.8010\pm 0.0186  LightGBMLightGBM\mathrm{LightGBM} 0.7890±0.0160plus-or-minus0.78900.01600.7890\pm 0.0160 0.7929±0.0106plus-or-minus0.79290.01060.7929\pm 0.0106  CatBoostCatBoost\mathrm{CatBoost} 0.7994±0.0131plus-or-minus0.79940.01310.7994\pm 0.0131 0.8057±0.0098plus-or-minus0.80570.00980.8057\pm 0.0098  TabRTabR\mathrm{TabR} 0.7936±0.0114plus-or-minus0.79360.01140.7936\pm 0.0114 0.8055±0.0057plus-or-minus0.80550.00570.8055\pm 0.0057  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7804±0.0148plus-or-minus0.78040.01480.7804\pm 0.0148 –  MNCAMNCA\mathrm{MNCA} 0.7911±0.0135plus-or-minus0.79110.01350.7911\pm 0.0135 0.8005±0.0121plus-or-minus0.80050.01210.8005\pm 0.0121  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7867±0.0113plus-or-minus0.78670.01130.7867\pm 0.0113 0.7953±0.0114plus-or-minus0.79530.01140.7953\pm 0.0114  TabMTabM\mathrm{TabM} 0.7961±0.0136plus-or-minus0.79610.01360.7961\pm 0.0136 0.8011±0.0084plus-or-minus0.80110.00840.8011\pm 0.0084  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7855±0.0164plus-or-minus0.78550.01640.7855\pm 0.0164 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7904±0.0123plus-or-minus0.79040.01230.7904\pm 0.0123 0.7986±0.0055plus-or-minus0.79860.00550.7986\pm 0.0055  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.7886±0.0167plus-or-minus0.78860.01670.7886\pm 0.0167 0.7963±0.0113plus-or-minus0.79630.01130.7963\pm 0.0113 | phoneme ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8525±0.0126plus-or-minus0.85250.01260.8525\pm 0.0126 0.8635±0.0099plus-or-minus0.86350.00990.8635\pm 0.0099  TabPFNTabPFN\mathrm{TabPFN} – 0.8684±0.0050plus-or-minus0.86840.00500.8684\pm 0.0050  ResNetResNet\mathrm{ResNet} 0.8456±0.0121plus-or-minus0.84560.01210.8456\pm 0.0121 0.8504±0.0066plus-or-minus0.85040.00660.8504\pm 0.0066  DCN2DCN2\mathrm{DCN2} 0.8342±0.0151plus-or-minus0.83420.01510.8342\pm 0.0151 0.8543±0.0118plus-or-minus0.85430.01180.8543\pm 0.0118  SNNSNN\mathrm{SNN} 0.8596±0.0124plus-or-minus0.85960.01240.8596\pm 0.0124 0.8687±0.0080plus-or-minus0.86870.00800.8687\pm 0.0080  TromptTrompt\mathrm{Trompt} 0.8465±0.0205plus-or-minus0.84650.02050.8465\pm 0.0205 –  AutoIntAutoInt\mathrm{AutoInt} 0.8623±0.0138plus-or-minus0.86230.01380.8623\pm 0.0138 0.8754±0.0095plus-or-minus0.87540.00950.8754\pm 0.0095  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8629±0.0123plus-or-minus0.86290.01230.8629\pm 0.0123 0.8757±0.0095plus-or-minus0.87570.00950.8757\pm 0.0095  ExcelExcel\mathrm{Excel} 0.8551±0.0092plus-or-minus0.85510.00920.8551\pm 0.0092 0.8711±0.0081plus-or-minus0.87110.00810.8711\pm 0.0081  SAINTSAINT\mathrm{SAINT} 0.8657±0.0130plus-or-minus0.86570.01300.8657\pm 0.0130 0.8799±0.0080plus-or-minus0.87990.00800.8799\pm 0.0080  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8667±0.0127plus-or-minus0.86670.01270.8667\pm 0.0127 0.8795±0.0093plus-or-minus0.87950.00930.8795\pm 0.0093  T2GT2G\mathrm{T2G} 0.8672±0.0166plus-or-minus0.86720.01660.8672\pm 0.0166 0.8765±0.0141plus-or-minus0.87650.01410.8765\pm 0.0141  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8742±0.0120plus-or-minus0.87420.01200.8742\pm 0.0120 0.8861±0.0071plus-or-minus0.88610.00710.8861\pm 0.0071  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8757±0.0118plus-or-minus0.87570.01180.8757\pm 0.0118 0.8856±0.0065plus-or-minus0.88560.00650.8856\pm 0.0065  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8647±0.0098plus-or-minus0.86470.00980.8647\pm 0.0098 0.8761±0.0076plus-or-minus0.87610.00760.8761\pm 0.0076  XGBoostXGBoost\mathrm{XGBoost} 0.8682±0.0174plus-or-minus0.86820.01740.8682\pm 0.0174 0.8771±0.0156plus-or-minus0.87710.01560.8771\pm 0.0156  LightGBMLightGBM\mathrm{LightGBM} 0.8702±0.0129plus-or-minus0.87020.01290.8702\pm 0.0129 0.8733±0.0126plus-or-minus0.87330.01260.8733\pm 0.0126  CatBoostCatBoost\mathrm{CatBoost} 0.8827±0.0117plus-or-minus0.88270.01170.8827\pm 0.0117 0.8897±0.0055plus-or-minus0.88970.00550.8897\pm 0.0055  TabRTabR\mathrm{TabR} 0.8781±0.0096plus-or-minus0.87810.00960.8781\pm 0.0096 0.8840±0.0054plus-or-minus0.88400.00540.8840\pm 0.0054  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8772±0.0087plus-or-minus0.87720.00870.8772\pm 0.0087 –  MNCAMNCA\mathrm{MNCA} 0.8835±0.0079plus-or-minus0.88350.00790.8835\pm 0.0079 0.8861±0.0057plus-or-minus0.88610.00570.8861\pm 0.0057  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8828±0.0082plus-or-minus0.88280.00820.8828\pm 0.0082 0.8925±0.0056plus-or-minus0.89250.00560.8925\pm 0.0056  TabMTabM\mathrm{TabM} 0.8701±0.0167plus-or-minus0.87010.01670.8701\pm 0.0167 0.8766±0.0128plus-or-minus0.87660.01280.8766\pm 0.0128  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8668±0.0180plus-or-minus0.86680.01800.8668\pm 0.0180 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8686±0.0153plus-or-minus0.86860.01530.8686\pm 0.0153 0.8758±0.0091plus-or-minus0.87580.00910.8758\pm 0.0091  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8790±0.0098plus-or-minus0.87900.00980.8790\pm 0.0098 0.8885±0.0056plus-or-minus0.88850.00560.8885\pm 0.0056 |
| analcatdata\_supreme ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0782±0.0081plus-or-minus0.07820.00810.0782\pm 0.0081 0.0766±0.0090plus-or-minus0.07660.00900.0766\pm 0.0090  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0852±0.0076plus-or-minus0.08520.00760.0852\pm 0.0076 0.0823±0.0078plus-or-minus0.08230.00780.0823\pm 0.0078  DCN2DCN2\mathrm{DCN2} 0.0811±0.0137plus-or-minus0.08110.01370.0811\pm 0.0137 0.0759±0.0086plus-or-minus0.07590.00860.0759\pm 0.0086  SNNSNN\mathrm{SNN} 0.0826±0.0096plus-or-minus0.08260.00960.0826\pm 0.0096 0.0779±0.0098plus-or-minus0.07790.00980.0779\pm 0.0098  TromptTrompt\mathrm{Trompt} 0.0782±0.0095plus-or-minus0.07820.00950.0782\pm 0.0095 –  AutoIntAutoInt\mathrm{AutoInt} 0.0783±0.0078plus-or-minus0.07830.00780.0783\pm 0.0078 0.0768±0.0083plus-or-minus0.07680.00830.0768\pm 0.0083  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0770±0.0082plus-or-minus0.07700.00820.0770\pm 0.0082 0.0759±0.0081plus-or-minus0.07590.00810.0759\pm 0.0081  ExcelExcel\mathrm{Excel} 0.0796±0.0101plus-or-minus0.07960.01010.0796\pm 0.0101 0.0776±0.0101plus-or-minus0.07760.01010.0776\pm 0.0101  SAINTSAINT\mathrm{SAINT} 0.0773±0.0078plus-or-minus0.07730.00780.0773\pm 0.0078 0.0759±0.0076plus-or-minus0.07590.00760.0759\pm 0.0076  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0787±0.0086plus-or-minus0.07870.00860.0787\pm 0.0086 0.0775±0.0091plus-or-minus0.07750.00910.0775\pm 0.0091  T2GT2G\mathrm{T2G} 0.0775±0.0081plus-or-minus0.07750.00810.0775\pm 0.0081 0.0763±0.0084plus-or-minus0.07630.00840.0763\pm 0.0084  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0798±0.0088plus-or-minus0.07980.00880.0798\pm 0.0088 0.0769±0.0092plus-or-minus0.07690.00920.0769\pm 0.0092  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0786±0.0073plus-or-minus0.07860.00730.0786\pm 0.0073 0.0720±0.0053plus-or-minus0.07200.00530.0720\pm 0.0053  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0774±0.0064plus-or-minus0.07740.00640.0774\pm 0.0064 0.0759±0.0063plus-or-minus0.07590.00630.0759\pm 0.0063  XGBoostXGBoost\mathrm{XGBoost} 0.0801±0.0126plus-or-minus0.08010.01260.0801\pm 0.0126 0.0774±0.0107plus-or-minus0.07740.01070.0774\pm 0.0107  LightGBMLightGBM\mathrm{LightGBM} 0.0778±0.0115plus-or-minus0.07780.01150.0778\pm 0.0115 0.0767±0.0110plus-or-minus0.07670.01100.0767\pm 0.0110  CatBoostCatBoost\mathrm{CatBoost} 0.0780±0.0067plus-or-minus0.07800.00670.0780\pm 0.0067 0.0734±0.0022plus-or-minus0.07340.00220.0734\pm 0.0022  TabRTabR\mathrm{TabR} 0.0803±0.0066plus-or-minus0.08030.00660.0803\pm 0.0066 0.0759±0.0046plus-or-minus0.07590.00460.0759\pm 0.0046  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0807±0.0088plus-or-minus0.08070.00880.0807\pm 0.0088 –  MNCAMNCA\mathrm{MNCA} 0.0809±0.0072plus-or-minus0.08090.00720.0809\pm 0.0072 0.0784±0.0062plus-or-minus0.07840.00620.0784\pm 0.0062  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0825±0.0090plus-or-minus0.08250.00900.0825\pm 0.0090 0.0793±0.0072plus-or-minus0.07930.00720.0793\pm 0.0072  TabMTabM\mathrm{TabM} 0.0777±0.0099plus-or-minus0.07770.00990.0777\pm 0.0099 0.0769±0.0105plus-or-minus0.07690.01050.0769\pm 0.0105  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0783±0.0103plus-or-minus0.07830.01030.0783\pm 0.0103 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0769±0.0091plus-or-minus0.07690.00910.0769\pm 0.0091 0.0758±0.0097plus-or-minus0.07580.00970.0758\pm 0.0097  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0790±0.0079plus-or-minus0.07900.00790.0790\pm 0.0079 0.0770±0.0086plus-or-minus0.07700.00860.0770\pm 0.0086 | Mercedes\_Benz\_Greener\_Manufacturing ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 8.3045±0.8708plus-or-minus8.30450.87088.3045\pm 0.8708 8.2682±0.8992plus-or-minus8.26820.89928.2682\pm 0.8992  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 8.4434±0.7982plus-or-minus8.44340.79828.4434\pm 0.7982 8.3178±0.8482plus-or-minus8.31780.84828.3178\pm 0.8482  DCN2DCN2\mathrm{DCN2} 8.3540±0.8314plus-or-minus8.35400.83148.3540\pm 0.8314 8.3021±0.8579plus-or-minus8.30210.85798.3021\pm 0.8579  SNNSNN\mathrm{SNN} 8.2718±0.8152plus-or-minus8.27180.81528.2718\pm 0.8152 8.2236±0.8479plus-or-minus8.22360.84798.2236\pm 0.8479  TromptTrompt\mathrm{Trompt} 8.3409±0.9840plus-or-minus8.34090.98408.3409\pm 0.9840 –  AutoIntAutoInt\mathrm{AutoInt} 8.4001±0.9256plus-or-minus8.40010.92568.4001\pm 0.9256 8.3237±0.9658plus-or-minus8.32370.96588.3237\pm 0.9658  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 8.2860±0.8656plus-or-minus8.28600.86568.2860\pm 0.8656 8.2398±0.9023plus-or-minus8.23980.90238.2398\pm 0.9023  ExcelExcel\mathrm{Excel} 8.2244±0.8514plus-or-minus8.22440.85148.2244\pm 0.8514 8.1918±0.9387plus-or-minus8.19180.93878.1918\pm 0.9387  SAINTSAINT\mathrm{SAINT} 8.3556±0.9566plus-or-minus8.35560.95668.3556\pm 0.9566 8.6626±1.0518plus-or-minus8.66261.05188.6626\pm 1.0518  FT​-​TFT-T\mathrm{FT\texttt{-}T} 8.2252±0.8617plus-or-minus8.22520.86178.2252\pm 0.8617 8.1616±0.8834plus-or-minus8.16160.88348.1616\pm 0.8834  T2GT2G\mathrm{T2G} 8.2120±0.8485plus-or-minus8.21200.84858.2120\pm 0.8485 8.1654±0.9339plus-or-minus8.16540.93398.1654\pm 0.9339  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 8.3045±0.8708plus-or-minus8.30450.87088.3045\pm 0.8708 8.2682±0.8992plus-or-minus8.26820.89928.2682\pm 0.8992  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 8.3045±0.8708plus-or-minus8.30450.87088.3045\pm 0.8708 8.2682±0.8992plus-or-minus8.26820.89928.2682\pm 0.8992  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 8.3045±0.8708plus-or-minus8.30450.87088.3045\pm 0.8708 8.2682±0.8992plus-or-minus8.26820.89928.2682\pm 0.8992  XGBoostXGBoost\mathrm{XGBoost} 8.2177±0.8175plus-or-minus8.21770.81758.2177\pm 0.8175 8.2092±0.8458plus-or-minus8.20920.84588.2092\pm 0.8458  LightGBMLightGBM\mathrm{LightGBM} 8.2078±0.8231plus-or-minus8.20780.82318.2078\pm 0.8231 8.1618±0.8566plus-or-minus8.16180.85668.1618\pm 0.8566  CatBoostCatBoost\mathrm{CatBoost} 8.1629±0.8193plus-or-minus8.16290.81938.1629\pm 0.8193 8.1554±0.8439plus-or-minus8.15540.84398.1554\pm 0.8439  TabRTabR\mathrm{TabR} 8.3506±0.8149plus-or-minus8.35060.81498.3506\pm 0.8149 8.2694±0.8399plus-or-minus8.26940.83998.2694\pm 0.8399  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 8.3187±0.8186plus-or-minus8.31870.81868.3187\pm 0.8186 –  MNCAMNCA\mathrm{MNCA} 8.2557±0.8602plus-or-minus8.25570.86028.2557\pm 0.8602 8.1771±0.8710plus-or-minus8.17710.87108.1771\pm 0.8710  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 8.2557±0.8602plus-or-minus8.25570.86028.2557\pm 0.8602 8.1771±0.8710plus-or-minus8.17710.87108.1771\pm 0.8710  TabMTabM\mathrm{TabM} 8.2215±0.8940plus-or-minus8.22150.89408.2215\pm 0.8940 8.1995±0.9130plus-or-minus8.19950.91308.1995\pm 0.9130  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 8.2206±0.8827plus-or-minus8.22060.88278.2206\pm 0.8827 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 8.2375±0.8953plus-or-minus8.23750.89538.2375\pm 0.8953 8.2161±0.9253plus-or-minus8.21610.92538.2161\pm 0.9253  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 8.2375±0.8953plus-or-minus8.23750.89538.2375\pm 0.8953 8.2161±0.9253plus-or-minus8.21610.92538.2161\pm 0.9253 |
| KDDCup09\_upselling ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7759±0.0137plus-or-minus0.77590.01370.7759\pm 0.0137 0.7806±0.0125plus-or-minus0.78060.01250.7806\pm 0.0125  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.7811±0.0124plus-or-minus0.78110.01240.7811\pm 0.0124 0.7861±0.0109plus-or-minus0.78610.01090.7861\pm 0.0109  DCN2DCN2\mathrm{DCN2} 0.7850±0.0161plus-or-minus0.78500.01610.7850\pm 0.0161 0.7884±0.0135plus-or-minus0.78840.01350.7884\pm 0.0135  SNNSNN\mathrm{SNN} 0.7884±0.0122plus-or-minus0.78840.01220.7884\pm 0.0122 0.7940±0.0116plus-or-minus0.79400.01160.7940\pm 0.0116  TromptTrompt\mathrm{Trompt} 0.7994±0.0055plus-or-minus0.79940.00550.7994\pm 0.0055 –  AutoIntAutoInt\mathrm{AutoInt} 0.8004±0.0075plus-or-minus0.80040.00750.8004\pm 0.0075 0.8037±0.0063plus-or-minus0.80370.00630.8037\pm 0.0063  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7979±0.0105plus-or-minus0.79790.01050.7979\pm 0.0105 0.8010±0.0094plus-or-minus0.80100.00940.8010\pm 0.0094  ExcelExcel\mathrm{Excel} 0.7903±0.0074plus-or-minus0.79030.00740.7903\pm 0.0074 0.7939±0.0099plus-or-minus0.79390.00990.7939\pm 0.0099  SAINTSAINT\mathrm{SAINT} 0.7942±0.0112plus-or-minus0.79420.01120.7942\pm 0.0112 0.7993±0.0081plus-or-minus0.79930.00810.7993\pm 0.0081  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7957±0.0127plus-or-minus0.79570.01270.7957\pm 0.0127 0.7960±0.0139plus-or-minus0.79600.01390.7960\pm 0.0139  T2GT2G\mathrm{T2G} 0.8037±0.0100plus-or-minus0.80370.01000.8037\pm 0.0100 0.7988±0.0084plus-or-minus0.79880.00840.7988\pm 0.0084  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7962±0.0093plus-or-minus0.79620.00930.7962\pm 0.0093 0.7995±0.0105plus-or-minus0.79950.01050.7995\pm 0.0105  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8005±0.0097plus-or-minus0.80050.00970.8005\pm 0.0097 0.8032±0.0117plus-or-minus0.80320.01170.8032\pm 0.0117  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7925±0.0123plus-or-minus0.79250.01230.7925\pm 0.0123 0.7963±0.0089plus-or-minus0.79630.00890.7963\pm 0.0089  XGBoostXGBoost\mathrm{XGBoost} 0.7930±0.0108plus-or-minus0.79300.01080.7930\pm 0.0108 0.7950±0.0102plus-or-minus0.79500.01020.7950\pm 0.0102  LightGBMLightGBM\mathrm{LightGBM} 0.7932±0.0119plus-or-minus0.79320.01190.7932\pm 0.0119 0.7969±0.0115plus-or-minus0.79690.01150.7969\pm 0.0115  CatBoostCatBoost\mathrm{CatBoost} 0.7992±0.0117plus-or-minus0.79920.01170.7992\pm 0.0117 0.8010±0.0121plus-or-minus0.80100.01210.8010\pm 0.0121  TabRTabR\mathrm{TabR} 0.7838±0.0136plus-or-minus0.78380.01360.7838\pm 0.0136 0.7859±0.0167plus-or-minus0.78590.01670.7859\pm 0.0167  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7908±0.0123plus-or-minus0.79080.01230.7908\pm 0.0123 –  MNCAMNCA\mathrm{MNCA} 0.7939±0.0097plus-or-minus0.79390.00970.7939\pm 0.0097 0.7989±0.0115plus-or-minus0.79890.01150.7989\pm 0.0115  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7960±0.0131plus-or-minus0.79600.01310.7960\pm 0.0131 0.8008±0.0110plus-or-minus0.80080.01100.8008\pm 0.0110  TabMTabM\mathrm{TabM} 0.8002±0.0103plus-or-minus0.80020.01030.8002\pm 0.0103 0.8021±0.0074plus-or-minus0.80210.00740.8021\pm 0.0074  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7974±0.0124plus-or-minus0.79740.01240.7974\pm 0.0124 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7963±0.0123plus-or-minus0.79630.01230.7963\pm 0.0123 0.8018±0.0076plus-or-minus0.80180.00760.8018\pm 0.0076  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8031±0.0133plus-or-minus0.80310.01330.8031\pm 0.0133 0.8039±0.0114plus-or-minus0.80390.01140.8039\pm 0.0114 | kdd\_ipums\_la\_97-small ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8828±0.0061plus-or-minus0.88280.00610.8828\pm 0.0061 0.8845±0.0055plus-or-minus0.88450.00550.8845\pm 0.0055  TabPFNTabPFN\mathrm{TabPFN} – 0.8578±0.0046plus-or-minus0.85780.00460.8578\pm 0.0046  ResNetResNet\mathrm{ResNet} 0.8823±0.0070plus-or-minus0.88230.00700.8823\pm 0.0070 0.8824±0.0060plus-or-minus0.88240.00600.8824\pm 0.0060  DCN2DCN2\mathrm{DCN2} 0.8770±0.0072plus-or-minus0.87700.00720.8770\pm 0.0072 0.8824±0.0068plus-or-minus0.88240.00680.8824\pm 0.0068  SNNSNN\mathrm{SNN} 0.8722±0.0093plus-or-minus0.87220.00930.8722\pm 0.0093 0.8733±0.0083plus-or-minus0.87330.00830.8733\pm 0.0083  TromptTrompt\mathrm{Trompt} 0.8847±0.0070plus-or-minus0.88470.00700.8847\pm 0.0070 –  AutoIntAutoInt\mathrm{AutoInt} 0.8808±0.0083plus-or-minus0.88080.00830.8808\pm 0.0083 0.8830±0.0081plus-or-minus0.88300.00810.8830\pm 0.0081  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8762±0.0100plus-or-minus0.87620.01000.8762\pm 0.0100 0.8770±0.0088plus-or-minus0.87700.00880.8770\pm 0.0088  ExcelExcel\mathrm{Excel} 0.8803±0.0054plus-or-minus0.88030.00540.8803\pm 0.0054 0.8823±0.0071plus-or-minus0.88230.00710.8823\pm 0.0071  SAINTSAINT\mathrm{SAINT} 0.8837±0.0055plus-or-minus0.88370.00550.8837\pm 0.0055 0.8839±0.0049plus-or-minus0.88390.00490.8839\pm 0.0049  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8795±0.0077plus-or-minus0.87950.00770.8795\pm 0.0077 0.8792±0.0062plus-or-minus0.87920.00620.8792\pm 0.0062  T2GT2G\mathrm{T2G} 0.8833±0.0054plus-or-minus0.88330.00540.8833\pm 0.0054 0.8841±0.0062plus-or-minus0.88410.00620.8841\pm 0.0062  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8765±0.0108plus-or-minus0.87650.01080.8765\pm 0.0108 0.8765±0.0108plus-or-minus0.87650.01080.8765\pm 0.0108  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8816±0.0057plus-or-minus0.88160.00570.8816\pm 0.0057 0.8818±0.0048plus-or-minus0.88180.00480.8818\pm 0.0048  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8757±0.0101plus-or-minus0.87570.01010.8757\pm 0.0101 0.8756±0.0104plus-or-minus0.87560.01040.8756\pm 0.0104  XGBoostXGBoost\mathrm{XGBoost} 0.8825±0.0089plus-or-minus0.88250.00890.8825\pm 0.0089 0.8835±0.0085plus-or-minus0.88350.00850.8835\pm 0.0085  LightGBMLightGBM\mathrm{LightGBM} 0.8792±0.0075plus-or-minus0.87920.00750.8792\pm 0.0075 0.8802±0.0067plus-or-minus0.88020.00670.8802\pm 0.0067  CatBoostCatBoost\mathrm{CatBoost} 0.8793±0.0088plus-or-minus0.87930.00880.8793\pm 0.0088 0.8803±0.0100plus-or-minus0.88030.01000.8803\pm 0.0100  TabRTabR\mathrm{TabR} 0.8798±0.0081plus-or-minus0.87980.00810.8798\pm 0.0081 0.8819±0.0078plus-or-minus0.88190.00780.8819\pm 0.0078  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8831±0.0050plus-or-minus0.88310.00500.8831\pm 0.0050 –  MNCAMNCA\mathrm{MNCA} 0.8819±0.0054plus-or-minus0.88190.00540.8819\pm 0.0054 0.8832±0.0048plus-or-minus0.88320.00480.8832\pm 0.0048  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8837±0.0062plus-or-minus0.88370.00620.8837\pm 0.0062 0.8860±0.0059plus-or-minus0.88600.00590.8860\pm 0.0059  TabMTabM\mathrm{TabM} 0.8845±0.0063plus-or-minus0.88450.00630.8845\pm 0.0063 0.8848±0.0070plus-or-minus0.88480.00700.8848\pm 0.0070  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8846±0.0059plus-or-minus0.88460.00590.8846\pm 0.0059 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8827±0.0054plus-or-minus0.88270.00540.8827\pm 0.0054 0.8810±0.0050plus-or-minus0.88100.00500.8810\pm 0.0050  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8775±0.0094plus-or-minus0.87750.00940.8775\pm 0.0094 0.8780±0.0099plus-or-minus0.87800.00990.8780\pm 0.0099 |
| wine\_quality ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.6707±0.0178plus-or-minus0.67070.01780.6707\pm 0.0178 0.6530±0.0152plus-or-minus0.65300.01520.6530\pm 0.0152  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.6687±0.0166plus-or-minus0.66870.01660.6687\pm 0.0166 0.6543±0.0170plus-or-minus0.65430.01700.6543\pm 0.0170  DCN2DCN2\mathrm{DCN2} 0.7010±0.0171plus-or-minus0.70100.01710.7010\pm 0.0171 0.6699±0.0139plus-or-minus0.66990.01390.6699\pm 0.0139  SNNSNN\mathrm{SNN} 0.6604±0.0174plus-or-minus0.66040.01740.6604\pm 0.0174 0.6245±0.0140plus-or-minus0.62450.01400.6245\pm 0.0140  TromptTrompt\mathrm{Trompt} 0.6605±0.0153plus-or-minus0.66050.01530.6605\pm 0.0153 –  AutoIntAutoInt\mathrm{AutoInt} 0.6840±0.0126plus-or-minus0.68400.01260.6840\pm 0.0126 0.6478±0.0146plus-or-minus0.64780.01460.6478\pm 0.0146  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.6672±0.0263plus-or-minus0.66720.02630.6672\pm 0.0263 0.6294±0.0200plus-or-minus0.62940.02000.6294\pm 0.0200  ExcelExcel\mathrm{Excel} 0.6881±0.0182plus-or-minus0.68810.01820.6881\pm 0.0182 0.6664±0.0179plus-or-minus0.66640.01790.6664\pm 0.0179  SAINTSAINT\mathrm{SAINT} 0.6797±0.0161plus-or-minus0.67970.01610.6797\pm 0.0161 0.6604±0.0307plus-or-minus0.66040.03070.6604\pm 0.0307  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.6787±0.0149plus-or-minus0.67870.01490.6787\pm 0.0149 0.6564±0.0250plus-or-minus0.65640.02500.6564\pm 0.0250  T2GT2G\mathrm{T2G} 0.6783±0.0170plus-or-minus0.67830.01700.6783\pm 0.0170 0.6570±0.0273plus-or-minus0.65700.02730.6570\pm 0.0273  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.6569±0.0167plus-or-minus0.65690.01670.6569\pm 0.0167 0.6328±0.0155plus-or-minus0.63280.01550.6328\pm 0.0155  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.6532±0.0133plus-or-minus0.65320.01330.6532\pm 0.0133 0.6336±0.0140plus-or-minus0.63360.01400.6336\pm 0.0140  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.6721±0.0180plus-or-minus0.67210.01800.6721\pm 0.0180 0.6463±0.0262plus-or-minus0.64630.02620.6463\pm 0.0262  XGBoostXGBoost\mathrm{XGBoost} 0.6039±0.0134plus-or-minus0.60390.01340.6039\pm 0.0134 0.6025±0.0139plus-or-minus0.60250.01390.6025\pm 0.0139  LightGBMLightGBM\mathrm{LightGBM} 0.6135±0.0138plus-or-minus0.61350.01380.6135\pm 0.0138 0.6122±0.0144plus-or-minus0.61220.01440.6122\pm 0.0144  CatBoostCatBoost\mathrm{CatBoost} 0.6088±0.0132plus-or-minus0.60880.01320.6088\pm 0.0132 0.6060±0.0137plus-or-minus0.60600.01370.6060\pm 0.0137  TabRTabR\mathrm{TabR} 0.6315±0.0097plus-or-minus0.63150.00970.6315\pm 0.0097 0.6197±0.0096plus-or-minus0.61970.00960.6197\pm 0.0096  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.6412±0.0105plus-or-minus0.64120.01050.6412\pm 0.0105 –  MNCAMNCA\mathrm{MNCA} 0.6154±0.0083plus-or-minus0.61540.00830.6154\pm 0.0083 0.6058±0.0149plus-or-minus0.60580.01490.6058\pm 0.0149  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.6099±0.0144plus-or-minus0.60990.01440.6099\pm 0.0144 0.6028±0.0157plus-or-minus0.60280.01570.6028\pm 0.0157  TabMTabM\mathrm{TabM} 0.6169±0.0123plus-or-minus0.61690.01230.6169\pm 0.0123 0.6131±0.0126plus-or-minus0.61310.01260.6131\pm 0.0126  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.6225±0.0114plus-or-minus0.62250.01140.6225\pm 0.0114 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.6193±0.0130plus-or-minus0.61930.01300.6193\pm 0.0130 0.6138±0.0140plus-or-minus0.61380.01400.6138\pm 0.0140  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.6255±0.0146plus-or-minus0.62550.01460.6255\pm 0.0146 0.6194±0.0150plus-or-minus0.61940.01500.6194\pm 0.0150 | isolet ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 2.2744±0.2203plus-or-minus2.27440.22032.2744\pm 0.2203 2.0018±0.1111plus-or-minus2.00180.11112.0018\pm 0.1111  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 2.2077±0.2248plus-or-minus2.20770.22482.2077\pm 0.2248 1.9206±0.1478plus-or-minus1.92060.14781.9206\pm 0.1478  DCN2DCN2\mathrm{DCN2} 2.2449±0.1579plus-or-minus2.24490.15792.2449\pm 0.1579 2.0176±0.0770plus-or-minus2.01760.07702.0176\pm 0.0770  SNNSNN\mathrm{SNN} 2.4269±0.2382plus-or-minus2.42690.23822.4269\pm 0.2382 2.1142±0.1262plus-or-minus2.11420.12622.1142\pm 0.1262  TromptTrompt\mathrm{Trompt} 2.6219±0.0315plus-or-minus2.62190.03152.6219\pm 0.0315 –  AutoIntAutoInt\mathrm{AutoInt} 2.6130±0.1658plus-or-minus2.61300.16582.6130\pm 0.1658 2.3308±0.1088plus-or-minus2.33080.10882.3308\pm 0.1088  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 2.3344±0.2073plus-or-minus2.33440.20732.3344\pm 0.2073 2.0915±0.1159plus-or-minus2.09150.11592.0915\pm 0.1159  ExcelExcel\mathrm{Excel} 2.8691±0.0882plus-or-minus2.86910.08822.8691\pm 0.0882 2.5989±0.0664plus-or-minus2.59890.06642.5989\pm 0.0664  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 2.4879±0.2524plus-or-minus2.48790.25242.4879\pm 0.2524 2.1501±0.1506plus-or-minus2.15010.15062.1501\pm 0.1506  T2GT2G\mathrm{T2G} 2.2867±0.2489plus-or-minus2.28670.24892.2867\pm 0.2489 1.9179±0.1530plus-or-minus1.91790.15301.9179\pm 0.1530  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 2.2719±0.1006plus-or-minus2.27190.10062.2719\pm 0.1006 2.1026±0.1088plus-or-minus2.10260.10882.1026\pm 0.1088  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 2.1832±0.1124plus-or-minus2.18320.11242.1832\pm 0.1124 2.0775±0.0805plus-or-minus2.07750.08052.0775\pm 0.0805  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 2.0979±0.1779plus-or-minus2.09790.17792.0979\pm 0.1779 1.9283±0.1334plus-or-minus1.92830.13341.9283\pm 0.1334  XGBoostXGBoost\mathrm{XGBoost} 2.7567±0.0470plus-or-minus2.75670.04702.7567\pm 0.0470 2.7294±0.0366plus-or-minus2.72940.03662.7294\pm 0.0366  LightGBMLightGBM\mathrm{LightGBM} 2.7005±0.0296plus-or-minus2.70050.02962.7005\pm 0.0296 2.6903±0.0290plus-or-minus2.69030.02902.6903\pm 0.0290  CatBoostCatBoost\mathrm{CatBoost} 2.8847±0.0227plus-or-minus2.88470.02272.8847\pm 0.0227 2.8574±0.0148plus-or-minus2.85740.01482.8574\pm 0.0148  TabRTabR\mathrm{TabR} 1.9760±0.1738plus-or-minus1.97600.17381.9760\pm 0.1738 1.7627±0.1520plus-or-minus1.76270.15201.7627\pm 0.1520  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 1.9919±0.1813plus-or-minus1.99190.18131.9919\pm 0.1813 –  MNCAMNCA\mathrm{MNCA} 1.7905±0.1594plus-or-minus1.79050.15941.7905\pm 0.1594 1.6205±0.1676plus-or-minus1.62050.16761.6205\pm 0.1676  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 1.8912±0.1851plus-or-minus1.89120.18511.8912\pm 0.1851 1.7147±0.1348plus-or-minus1.71470.13481.7147\pm 0.1348  TabMTabM\mathrm{TabM} 1.8831±0.1194plus-or-minus1.88310.11941.8831\pm 0.1194 1.8578±0.1088plus-or-minus1.85780.10881.8578\pm 0.1088  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 1.9549±0.1319plus-or-minus1.95490.13191.9549\pm 0.1319 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 1.9966±0.0923plus-or-minus1.99660.09231.9966\pm 0.0923 1.9311±0.0862plus-or-minus1.93110.08621.9311\pm 0.0862  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 1.8378±0.0803plus-or-minus1.83780.08031.8378\pm 0.0803 1.8126±0.0692plus-or-minus1.81260.06921.8126\pm 0.0692 |
| cpu\_act ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 2.6814±0.2291plus-or-minus2.68140.22912.6814\pm 0.2291 2.4953±0.1150plus-or-minus2.49530.11502.4953\pm 0.1150  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 2.3933±0.0641plus-or-minus2.39330.06412.3933\pm 0.0641 2.3005±0.0397plus-or-minus2.30050.03972.3005\pm 0.0397  DCN2DCN2\mathrm{DCN2} 2.7868±0.1999plus-or-minus2.78680.19992.7868\pm 0.1999 2.4884±0.0327plus-or-minus2.48840.03272.4884\pm 0.0327  SNNSNN\mathrm{SNN} 2.5811±0.1480plus-or-minus2.58110.14802.5811\pm 0.1480 2.3863±0.0324plus-or-minus2.38630.03242.3863\pm 0.0324  TromptTrompt\mathrm{Trompt} 2.2133±0.0221plus-or-minus2.21330.02212.2133\pm 0.0221 –  AutoIntAutoInt\mathrm{AutoInt} 2.2537±0.0536plus-or-minus2.25370.05362.2537\pm 0.0536 2.1708±0.0349plus-or-minus2.17080.03492.1708\pm 0.0349  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 2.3079±0.0829plus-or-minus2.30790.08292.3079\pm 0.0829 2.1831±0.0470plus-or-minus2.18310.04702.1831\pm 0.0470  ExcelExcel\mathrm{Excel} 2.3094±0.2401plus-or-minus2.30940.24012.3094\pm 0.2401 2.1411±0.0767plus-or-minus2.14110.07672.1411\pm 0.0767  SAINTSAINT\mathrm{SAINT} 2.2781±0.0630plus-or-minus2.27810.06302.2781\pm 0.0630 2.2032±0.0310plus-or-minus2.20320.03102.2032\pm 0.0310  FT​-​TFT-T\mathrm{FT\texttt{-}T} 2.2394±0.0508plus-or-minus2.23940.05082.2394\pm 0.0508 2.1494±0.0268plus-or-minus2.14940.02682.1494\pm 0.0268  T2GT2G\mathrm{T2G} 2.2111±0.0413plus-or-minus2.21110.04132.2111\pm 0.0413 2.1330±0.0316plus-or-minus2.13300.03162.1330\pm 0.0316  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 2.2730±0.0457plus-or-minus2.27300.04572.2730\pm 0.0457 2.1899±0.0419plus-or-minus2.18990.04192.1899\pm 0.0419  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 2.2671±0.0383plus-or-minus2.26710.03832.2671\pm 0.0383 2.1940±0.0433plus-or-minus2.19400.04332.1940\pm 0.0433  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 2.3309±0.0719plus-or-minus2.33090.07192.3309\pm 0.0719 2.2516±0.0574plus-or-minus2.25160.05742.2516\pm 0.0574  XGBoostXGBoost\mathrm{XGBoost} 2.5237±0.3530plus-or-minus2.52370.35302.5237\pm 0.3530 2.4723±0.3789plus-or-minus2.47230.37892.4723\pm 0.3789  LightGBMLightGBM\mathrm{LightGBM} 2.2223±0.0894plus-or-minus2.22230.08942.2223\pm 0.0894 2.2067±0.0916plus-or-minus2.20670.09162.2067\pm 0.0916  CatBoostCatBoost\mathrm{CatBoost} 2.1239±0.0489plus-or-minus2.12390.04892.1239\pm 0.0489 2.1092±0.0499plus-or-minus2.10920.04992.1092\pm 0.0499  TabRTabR\mathrm{TabR} 2.2980±0.0529plus-or-minus2.29800.05292.2980\pm 0.0529 2.2228±0.0501plus-or-minus2.22280.05012.2228\pm 0.0501  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 2.1278±0.0783plus-or-minus2.12780.07832.1278\pm 0.0783 –  MNCAMNCA\mathrm{MNCA} 2.2603±0.0479plus-or-minus2.26030.04792.2603\pm 0.0479 2.2339±0.0508plus-or-minus2.23390.05082.2339\pm 0.0508  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 2.2105±0.0483plus-or-minus2.21050.04832.2105\pm 0.0483 2.1396±0.0474plus-or-minus2.13960.04742.1396\pm 0.0474  TabMTabM\mathrm{TabM} 2.1940±0.0523plus-or-minus2.19400.05232.1940\pm 0.0523 2.1677±0.0487plus-or-minus2.16770.04872.1677\pm 0.0487  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 2.2033±0.0552plus-or-minus2.20330.05522.2033\pm 0.0552 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 2.2254±0.0734plus-or-minus2.22540.07342.2254\pm 0.0734 2.1877±0.0541plus-or-minus2.18770.05412.1877\pm 0.0541  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 2.1572±0.0376plus-or-minus2.15720.03762.1572\pm 0.0376 2.1222±0.0358plus-or-minus2.12220.03582.1222\pm 0.0358 | visualizing\_soil ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.1461±0.0152plus-or-minus0.14610.01520.1461\pm 0.0152 0.1338±0.0073plus-or-minus0.13380.00730.1338\pm 0.0073  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.3586±0.0489plus-or-minus0.35860.04890.3586\pm 0.0489 0.3046±0.0381plus-or-minus0.30460.03810.3046\pm 0.0381  DCN2DCN2\mathrm{DCN2} 0.3547±0.2726plus-or-minus0.35470.27260.3547\pm 0.2726 0.2549±0.1517plus-or-minus0.25490.15170.2549\pm 0.1517  SNNSNN\mathrm{SNN} 0.3642±0.2350plus-or-minus0.36420.23500.3642\pm 0.2350 0.3058±0.2212plus-or-minus0.30580.22120.3058\pm 0.2212  TromptTrompt\mathrm{Trompt} 0.1231±0.0177plus-or-minus0.12310.01770.1231\pm 0.0177 –  AutoIntAutoInt\mathrm{AutoInt} 0.1598±0.0724plus-or-minus0.15980.07240.1598\pm 0.0724 0.1357±0.0655plus-or-minus0.13570.06550.1357\pm 0.0655  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.1431±0.0472plus-or-minus0.14310.04720.1431\pm 0.0472 0.1323±0.0420plus-or-minus0.13230.04200.1323\pm 0.0420  ExcelExcel\mathrm{Excel} 0.1521±0.0232plus-or-minus0.15210.02320.1521\pm 0.0232 0.1261±0.0056plus-or-minus0.12610.00560.1261\pm 0.0056  SAINTSAINT\mathrm{SAINT} 0.1368±0.0155plus-or-minus0.13680.01550.1368\pm 0.0155 0.1235±0.0051plus-or-minus0.12350.00510.1235\pm 0.0051  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.1443±0.0235plus-or-minus0.14430.02350.1443\pm 0.0235 0.1250±0.0104plus-or-minus0.12500.01040.1250\pm 0.0104  T2GT2G\mathrm{T2G} 0.2067±0.1671plus-or-minus0.20670.16710.2067\pm 0.1671 0.1817±0.1068plus-or-minus0.18170.10680.1817\pm 0.1068  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.1601±0.0785plus-or-minus0.16010.07850.1601\pm 0.0785 0.1396±0.0630plus-or-minus0.13960.06300.1396\pm 0.0630  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.1407±0.0400plus-or-minus0.14070.04000.1407\pm 0.0400 0.1183±0.0406plus-or-minus0.11830.04060.1183\pm 0.0406  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.1063±0.0239plus-or-minus0.10630.02390.1063\pm 0.0239 0.0973±0.0180plus-or-minus0.09730.01800.0973\pm 0.0180  XGBoostXGBoost\mathrm{XGBoost} 0.1765±0.0707plus-or-minus0.17650.07070.1765\pm 0.0707 0.1539±0.0539plus-or-minus0.15390.05390.1539\pm 0.0539  LightGBMLightGBM\mathrm{LightGBM} 0.0616±0.0159plus-or-minus0.06160.01590.0616\pm 0.0159 0.0616±0.0167plus-or-minus0.06160.01670.0616\pm 0.0167  CatBoostCatBoost\mathrm{CatBoost} 0.0554±0.0063plus-or-minus0.05540.00630.0554\pm 0.0063 0.0468±0.0059plus-or-minus0.04680.00590.0468\pm 0.0059  TabRTabR\mathrm{TabR} 0.3979±0.3523plus-or-minus0.39790.35230.3979\pm 0.3523 0.3869±0.3746plus-or-minus0.38690.37460.3869\pm 0.3746  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.2268±0.2641plus-or-minus0.22680.26410.2268\pm 0.2641 –  MNCAMNCA\mathrm{MNCA} 0.3642±0.3482plus-or-minus0.36420.34820.3642\pm 0.3482 0.3626±0.3660plus-or-minus0.36260.36600.3626\pm 0.3660  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.2367±0.3529plus-or-minus0.23670.35290.2367\pm 0.3529 0.2290±0.2782plus-or-minus0.22900.27820.2290\pm 0.2782  TabMTabM\mathrm{TabM} 0.1242±0.0188plus-or-minus0.12420.01880.1242\pm 0.0188 0.1171±0.0118plus-or-minus0.11710.01180.1171\pm 0.0118  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.1215±0.0185plus-or-minus0.12150.01850.1215\pm 0.0185 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.1430±0.0279plus-or-minus0.14300.02790.1430\pm 0.0279 0.1367±0.0278plus-or-minus0.13670.02780.1367\pm 0.0278  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.1060±0.0243plus-or-minus0.10600.02430.1060\pm 0.0243 0.1043±0.0234plus-or-minus0.10430.02340.1043\pm 0.0234 |
| sulfur ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0217±0.0024plus-or-minus0.02170.00240.0217\pm 0.0024 0.0204±0.0028plus-or-minus0.02040.00280.0204\pm 0.0028  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0214±0.0020plus-or-minus0.02140.00200.0214\pm 0.0020 0.0199±0.0023plus-or-minus0.01990.00230.0199\pm 0.0023  DCN2DCN2\mathrm{DCN2} 0.0247±0.0050plus-or-minus0.02470.00500.0247\pm 0.0050 0.0208±0.0050plus-or-minus0.02080.00500.0208\pm 0.0050  SNNSNN\mathrm{SNN} 0.0209±0.0034plus-or-minus0.02090.00340.0209\pm 0.0034 0.0194±0.0038plus-or-minus0.01940.00380.0194\pm 0.0038  TromptTrompt\mathrm{Trompt} 0.0234±0.0042plus-or-minus0.02340.00420.0234\pm 0.0042 –  AutoIntAutoInt\mathrm{AutoInt} 0.0206±0.0035plus-or-minus0.02060.00350.0206\pm 0.0035 0.0192±0.0034plus-or-minus0.01920.00340.0192\pm 0.0034  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0199±0.0034plus-or-minus0.01990.00340.0199\pm 0.0034 0.0184±0.0032plus-or-minus0.01840.00320.0184\pm 0.0032  ExcelExcel\mathrm{Excel} 0.0251±0.0052plus-or-minus0.02510.00520.0251\pm 0.0052 0.0242±0.0048plus-or-minus0.02420.00480.0242\pm 0.0048  SAINTSAINT\mathrm{SAINT} 0.0199±0.0028plus-or-minus0.01990.00280.0199\pm 0.0028 0.0178±0.0022plus-or-minus0.01780.00220.0178\pm 0.0022  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0215±0.0042plus-or-minus0.02150.00420.0215\pm 0.0042 0.0201±0.0037plus-or-minus0.02010.00370.0201\pm 0.0037  T2GT2G\mathrm{T2G} 0.0213±0.0034plus-or-minus0.02130.00340.0213\pm 0.0034 0.0194±0.0030plus-or-minus0.01940.00300.0194\pm 0.0030  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0192±0.0032plus-or-minus0.01920.00320.0192\pm 0.0032 0.0181±0.0028plus-or-minus0.01810.00280.0181\pm 0.0028  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0192±0.0026plus-or-minus0.01920.00260.0192\pm 0.0026 0.0181±0.0029plus-or-minus0.01810.00290.0181\pm 0.0029  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0197±0.0026plus-or-minus0.01970.00260.0197\pm 0.0026 0.0187±0.0029plus-or-minus0.01870.00290.0187\pm 0.0029  XGBoostXGBoost\mathrm{XGBoost} 0.0202±0.0019plus-or-minus0.02020.00190.0202\pm 0.0019 0.0200±0.0017plus-or-minus0.02000.00170.0200\pm 0.0017  LightGBMLightGBM\mathrm{LightGBM} 0.0203±0.0020plus-or-minus0.02030.00200.0203\pm 0.0020 0.0200±0.0015plus-or-minus0.02000.00150.0200\pm 0.0015  CatBoostCatBoost\mathrm{CatBoost} 0.0189±0.0022plus-or-minus0.01890.00220.0189\pm 0.0022 0.0185±0.0022plus-or-minus0.01850.00220.0185\pm 0.0022  TabRTabR\mathrm{TabR} 0.0222±0.0022plus-or-minus0.02220.00220.0222\pm 0.0022 0.0208±0.0021plus-or-minus0.02080.00210.0208\pm 0.0021  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0217±0.0031plus-or-minus0.02170.00310.0217\pm 0.0031 –  MNCAMNCA\mathrm{MNCA} 0.0198±0.0030plus-or-minus0.01980.00300.0198\pm 0.0030 0.0189±0.0020plus-or-minus0.01890.00200.0189\pm 0.0020  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0198±0.0029plus-or-minus0.01980.00290.0198\pm 0.0029 0.0185±0.0032plus-or-minus0.01850.00320.0185\pm 0.0032  TabMTabM\mathrm{TabM} 0.0192±0.0035plus-or-minus0.01920.00350.0192\pm 0.0035 0.0184±0.0030plus-or-minus0.01840.00300.0184\pm 0.0030  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0192±0.0032plus-or-minus0.01920.00320.0192\pm 0.0032 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0195±0.0034plus-or-minus0.01950.00340.0195\pm 0.0034 0.0189±0.0032plus-or-minus0.01890.00320.0189\pm 0.0032  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0197±0.0042plus-or-minus0.01970.00420.0197\pm 0.0042 0.0192±0.0045plus-or-minus0.01920.00450.0192\pm 0.0045 | bank-marketing ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7860±0.0057plus-or-minus0.78600.00570.7860\pm 0.0057 0.7887±0.0052plus-or-minus0.78870.00520.7887\pm 0.0052  TabPFNTabPFN\mathrm{TabPFN} – 0.7894±0.0091plus-or-minus0.78940.00910.7894\pm 0.0091  ResNetResNet\mathrm{ResNet} 0.7921±0.0076plus-or-minus0.79210.00760.7921\pm 0.0076 0.7932±0.0066plus-or-minus0.79320.00660.7932\pm 0.0066  DCN2DCN2\mathrm{DCN2} 0.7859±0.0068plus-or-minus0.78590.00680.7859\pm 0.0068 0.7917±0.0078plus-or-minus0.79170.00780.7917\pm 0.0078  SNNSNN\mathrm{SNN} 0.7836±0.0074plus-or-minus0.78360.00740.7836\pm 0.0074 0.7882±0.0054plus-or-minus0.78820.00540.7882\pm 0.0054  TromptTrompt\mathrm{Trompt} 0.7975±0.0080plus-or-minus0.79750.00800.7975\pm 0.0080 –  AutoIntAutoInt\mathrm{AutoInt} 0.7917±0.0071plus-or-minus0.79170.00710.7917\pm 0.0071 0.7956±0.0058plus-or-minus0.79560.00580.7956\pm 0.0058  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7954±0.0059plus-or-minus0.79540.00590.7954\pm 0.0059 0.8001±0.0048plus-or-minus0.80010.00480.8001\pm 0.0048  ExcelExcel\mathrm{Excel} 0.7957±0.0090plus-or-minus0.79570.00900.7957\pm 0.0090 0.7985±0.0106plus-or-minus0.79850.01060.7985\pm 0.0106  SAINTSAINT\mathrm{SAINT} 0.7953±0.0058plus-or-minus0.79530.00580.7953\pm 0.0058 0.7974±0.0050plus-or-minus0.79740.00500.7974\pm 0.0050  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7918±0.0076plus-or-minus0.79180.00760.7918\pm 0.0076 0.7951±0.0071plus-or-minus0.79510.00710.7951\pm 0.0071  T2GT2G\mathrm{T2G} 0.7918±0.0058plus-or-minus0.79180.00580.7918\pm 0.0058 0.7955±0.0047plus-or-minus0.79550.00470.7955\pm 0.0047  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7947±0.0101plus-or-minus0.79470.01010.7947\pm 0.0101 0.7977±0.0117plus-or-minus0.79770.01170.7977\pm 0.0117  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7988±0.0092plus-or-minus0.79880.00920.7988\pm 0.0092 0.8024±0.0093plus-or-minus0.80240.00930.8024\pm 0.0093  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7981±0.0065plus-or-minus0.79810.00650.7981\pm 0.0065 0.8008±0.0057plus-or-minus0.80080.00570.8008\pm 0.0057  XGBoostXGBoost\mathrm{XGBoost} 0.8013±0.0081plus-or-minus0.80130.00810.8013\pm 0.0081 0.8030±0.0076plus-or-minus0.80300.00760.8030\pm 0.0076  LightGBMLightGBM\mathrm{LightGBM} 0.8006±0.0078plus-or-minus0.80060.00780.8006\pm 0.0078 0.8013±0.0072plus-or-minus0.80130.00720.8013\pm 0.0072  CatBoostCatBoost\mathrm{CatBoost} 0.8026±0.0068plus-or-minus0.80260.00680.8026\pm 0.0068 0.8056±0.0082plus-or-minus0.80560.00820.8056\pm 0.0082  TabRTabR\mathrm{TabR} 0.7995±0.0054plus-or-minus0.79950.00540.7995\pm 0.0054 0.8015±0.0037plus-or-minus0.80150.00370.8015\pm 0.0037  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8023±0.0088plus-or-minus0.80230.00880.8023\pm 0.0088 –  MNCAMNCA\mathrm{MNCA} 0.7961±0.0065plus-or-minus0.79610.00650.7961\pm 0.0065 0.8003±0.0077plus-or-minus0.80030.00770.8003\pm 0.0077  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7977±0.0081plus-or-minus0.79770.00810.7977\pm 0.0081 0.8010±0.0084plus-or-minus0.80100.00840.8010\pm 0.0084  TabMTabM\mathrm{TabM} 0.7908±0.0068plus-or-minus0.79080.00680.7908\pm 0.0068 0.7915±0.0068plus-or-minus0.79150.00680.7915\pm 0.0068  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7902±0.0067plus-or-minus0.79020.00670.7902\pm 0.0067 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7938±0.0064plus-or-minus0.79380.00640.7938\pm 0.0064 0.7959±0.0071plus-or-minus0.79590.00710.7959\pm 0.0071  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8003±0.0087plus-or-minus0.80030.00870.8003\pm 0.0087 0.8017±0.0087plus-or-minus0.80170.00870.8017\pm 0.0087 |
| Brazilian\_houses ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0473±0.0179plus-or-minus0.04730.01790.0473\pm 0.0179 0.0440±0.0207plus-or-minus0.04400.02070.0440\pm 0.0207  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0505±0.0181plus-or-minus0.05050.01810.0505\pm 0.0181 0.0458±0.0207plus-or-minus0.04580.02070.0458\pm 0.0207  DCN2DCN2\mathrm{DCN2} 0.0477±0.0172plus-or-minus0.04770.01720.0477\pm 0.0172 0.0427±0.0207plus-or-minus0.04270.02070.0427\pm 0.0207  SNNSNN\mathrm{SNN} 0.0630±0.0162plus-or-minus0.06300.01620.0630\pm 0.0162 0.0556±0.0175plus-or-minus0.05560.01750.0556\pm 0.0175  TromptTrompt\mathrm{Trompt} 0.0404±0.0266plus-or-minus0.04040.02660.0404\pm 0.0266 –  AutoIntAutoInt\mathrm{AutoInt} 0.0470±0.0192plus-or-minus0.04700.01920.0470\pm 0.0192 0.0437±0.0217plus-or-minus0.04370.02170.0437\pm 0.0217  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0513±0.0234plus-or-minus0.05130.02340.0513\pm 0.0234 0.0484±0.0262plus-or-minus0.04840.02620.0484\pm 0.0262  ExcelExcel\mathrm{Excel} 0.0450±0.0156plus-or-minus0.04500.01560.0450\pm 0.0156 0.0418±0.0190plus-or-minus0.04180.01900.0418\pm 0.0190  SAINTSAINT\mathrm{SAINT} 0.0479±0.0205plus-or-minus0.04790.02050.0479\pm 0.0205 0.0426±0.0236plus-or-minus0.04260.02360.0426\pm 0.0236  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0438±0.0181plus-or-minus0.04380.01810.0438\pm 0.0181 0.0412±0.0204plus-or-minus0.04120.02040.0412\pm 0.0204  T2GT2G\mathrm{T2G} 0.0468±0.0165plus-or-minus0.04680.01650.0468\pm 0.0165 0.0436±0.0211plus-or-minus0.04360.02110.0436\pm 0.0211  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0426±0.0180plus-or-minus0.04260.01800.0426\pm 0.0180 0.0397±0.0206plus-or-minus0.03970.02060.0397\pm 0.0206  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0437±0.0203plus-or-minus0.04370.02030.0437\pm 0.0203 0.0407±0.0230plus-or-minus0.04070.02300.0407\pm 0.0230  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0421±0.0209plus-or-minus0.04210.02090.0421\pm 0.0209 0.0409±0.0226plus-or-minus0.04090.02260.0409\pm 0.0226  XGBoostXGBoost\mathrm{XGBoost} 0.0541±0.0270plus-or-minus0.05410.02700.0541\pm 0.0270 0.0535±0.0287plus-or-minus0.05350.02870.0535\pm 0.0287  LightGBMLightGBM\mathrm{LightGBM} 0.0603±0.0249plus-or-minus0.06030.02490.0603\pm 0.0249 0.0589±0.0271plus-or-minus0.05890.02710.0589\pm 0.0271  CatBoostCatBoost\mathrm{CatBoost} 0.0468±0.0312plus-or-minus0.04680.03120.0468\pm 0.0312 0.0456±0.0332plus-or-minus0.04560.03320.0456\pm 0.0332  TabRTabR\mathrm{TabR} 0.0490±0.0152plus-or-minus0.04900.01520.0490\pm 0.0152 0.0454±0.0170plus-or-minus0.04540.01700.0454\pm 0.0170  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0451±0.0163plus-or-minus0.04510.01630.0451\pm 0.0163 –  MNCAMNCA\mathrm{MNCA} 0.0527±0.0157plus-or-minus0.05270.01570.0527\pm 0.0157 0.0509±0.0180plus-or-minus0.05090.01800.0509\pm 0.0180  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0553±0.0192plus-or-minus0.05530.01920.0553\pm 0.0192 0.0511±0.0191plus-or-minus0.05110.01910.0511\pm 0.0191  TabMTabM\mathrm{TabM} 0.0443±0.0213plus-or-minus0.04430.02130.0443\pm 0.0213 0.0431±0.0233plus-or-minus0.04310.02330.0431\pm 0.0233  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0450±0.0202plus-or-minus0.04500.02020.0450\pm 0.0202 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0480±0.0194plus-or-minus0.04800.01940.0480\pm 0.0194 0.0452±0.0221plus-or-minus0.04520.02210.0452\pm 0.0221  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0460±0.0206plus-or-minus0.04600.02060.0460\pm 0.0206 0.0439±0.0228plus-or-minus0.04390.02280.0439\pm 0.0228 | MagicTelescope ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8539±0.0060plus-or-minus0.85390.00600.8539\pm 0.0060 0.8566±0.0061plus-or-minus0.85660.00610.8566\pm 0.0061  TabPFNTabPFN\mathrm{TabPFN} – 0.8579±0.0064plus-or-minus0.85790.00640.8579\pm 0.0064  ResNetResNet\mathrm{ResNet} 0.8589±0.0068plus-or-minus0.85890.00680.8589\pm 0.0068 0.8651±0.0049plus-or-minus0.86510.00490.8651\pm 0.0049  DCN2DCN2\mathrm{DCN2} 0.8432±0.0074plus-or-minus0.84320.00740.8432\pm 0.0074 0.8490±0.0046plus-or-minus0.84900.00460.8490\pm 0.0046  SNNSNN\mathrm{SNN} 0.8536±0.0052plus-or-minus0.85360.00520.8536\pm 0.0052 0.8567±0.0047plus-or-minus0.85670.00470.8567\pm 0.0047  TromptTrompt\mathrm{Trompt} 0.8605±0.0102plus-or-minus0.86050.01020.8605\pm 0.0102 –  AutoIntAutoInt\mathrm{AutoInt} 0.8522±0.0056plus-or-minus0.85220.00560.8522\pm 0.0056 0.8560±0.0034plus-or-minus0.85600.00340.8560\pm 0.0034  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8571±0.0080plus-or-minus0.85710.00800.8571\pm 0.0080 0.8624±0.0044plus-or-minus0.86240.00440.8624\pm 0.0044  ExcelExcel\mathrm{Excel} 0.8480±0.0090plus-or-minus0.84800.00900.8480\pm 0.0090 0.8543±0.0075plus-or-minus0.85430.00750.8543\pm 0.0075  SAINTSAINT\mathrm{SAINT} 0.8595±0.0060plus-or-minus0.85950.00600.8595\pm 0.0060 0.8632±0.0061plus-or-minus0.86320.00610.8632\pm 0.0061  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8588±0.0046plus-or-minus0.85880.00460.8588\pm 0.0046 0.8643±0.0037plus-or-minus0.86430.00370.8643\pm 0.0037  T2GT2G\mathrm{T2G} 0.8553±0.0055plus-or-minus0.85530.00550.8553\pm 0.0055 0.8595±0.0051plus-or-minus0.85950.00510.8595\pm 0.0051  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8591±0.0061plus-or-minus0.85910.00610.8591\pm 0.0061 0.8626±0.0044plus-or-minus0.86260.00440.8626\pm 0.0044  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8575±0.0056plus-or-minus0.85750.00560.8575\pm 0.0056 0.8605±0.0051plus-or-minus0.86050.00510.8605\pm 0.0051  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8593±0.0054plus-or-minus0.85930.00540.8593\pm 0.0054 0.8621±0.0037plus-or-minus0.86210.00370.8621\pm 0.0037  XGBoostXGBoost\mathrm{XGBoost} 0.8550±0.0094plus-or-minus0.85500.00940.8550\pm 0.0094 0.8589±0.0110plus-or-minus0.85890.01100.8589\pm 0.0110  LightGBMLightGBM\mathrm{LightGBM} 0.8547±0.0085plus-or-minus0.85470.00850.8547\pm 0.0085 0.8556±0.0086plus-or-minus0.85560.00860.8556\pm 0.0086  CatBoostCatBoost\mathrm{CatBoost} 0.8586±0.0070plus-or-minus0.85860.00700.8586\pm 0.0070 0.8588±0.0077plus-or-minus0.85880.00770.8588\pm 0.0077  TabRTabR\mathrm{TabR} 0.8682±0.0058plus-or-minus0.86820.00580.8682\pm 0.0058 0.8729±0.0038plus-or-minus0.87290.00380.8729\pm 0.0038  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8641±0.0052plus-or-minus0.86410.00520.8641\pm 0.0052 –  MNCAMNCA\mathrm{MNCA} 0.8602±0.0061plus-or-minus0.86020.00610.8602\pm 0.0061 0.8628±0.0041plus-or-minus0.86280.00410.8628\pm 0.0041  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8622±0.0085plus-or-minus0.86220.00850.8622\pm 0.0085 0.8681±0.0064plus-or-minus0.86810.00640.8681\pm 0.0064  TabMTabM\mathrm{TabM} 0.8607±0.0058plus-or-minus0.86070.00580.8607\pm 0.0058 0.8622±0.0050plus-or-minus0.86220.00500.8622\pm 0.0050  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8585±0.0057plus-or-minus0.85850.00570.8585\pm 0.0057 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8581±0.0053plus-or-minus0.85810.00530.8581\pm 0.0053 0.8597±0.0055plus-or-minus0.85970.00550.8597\pm 0.0055  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8616±0.0080plus-or-minus0.86160.00800.8616\pm 0.0080 0.8646±0.0075plus-or-minus0.86460.00750.8646\pm 0.0075 |
| Ailerons ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  DCN2DCN2\mathrm{DCN2} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  SNNSNN\mathrm{SNN} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TromptTrompt\mathrm{Trompt} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 –  AutoIntAutoInt\mathrm{AutoInt} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  ExcelExcel\mathrm{Excel} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  SAINTSAINT\mathrm{SAINT} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  T2GT2G\mathrm{T2G} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  XGBoostXGBoost\mathrm{XGBoost} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabRTabR\mathrm{TabR} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 –  MNCAMNCA\mathrm{MNCA} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabMTabM\mathrm{TabM} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 0.0002±0.0000plus-or-minus0.00020.00000.0002\pm 0.0000 | MiamiHousing2016 ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.1614±0.0033plus-or-minus0.16140.00330.1614\pm 0.0033 0.1574±0.0043plus-or-minus0.15740.00430.1574\pm 0.0043  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.1548±0.0030plus-or-minus0.15480.00300.1548\pm 0.0030 0.1511±0.0027plus-or-minus0.15110.00270.1511\pm 0.0027  DCN2DCN2\mathrm{DCN2} 0.1683±0.0099plus-or-minus0.16830.00990.1683\pm 0.0099 0.1575±0.0047plus-or-minus0.15750.00470.1575\pm 0.0047  SNNSNN\mathrm{SNN} 0.1618±0.0029plus-or-minus0.16180.00290.1618\pm 0.0029 0.1557±0.0021plus-or-minus0.15570.00210.1557\pm 0.0021  TromptTrompt\mathrm{Trompt} 0.1478±0.0028plus-or-minus0.14780.00280.1478\pm 0.0028 –  AutoIntAutoInt\mathrm{AutoInt} 0.1537±0.0035plus-or-minus0.15370.00350.1537\pm 0.0035 0.1478±0.0027plus-or-minus0.14780.00270.1478\pm 0.0027  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.1527±0.0037plus-or-minus0.15270.00370.1527\pm 0.0037 0.1479±0.0033plus-or-minus0.14790.00330.1479\pm 0.0033  ExcelExcel\mathrm{Excel} 0.1519±0.0038plus-or-minus0.15190.00380.1519\pm 0.0038 0.1442±0.0022plus-or-minus0.14420.00220.1442\pm 0.0022  SAINTSAINT\mathrm{SAINT} 0.1507±0.0022plus-or-minus0.15070.00220.1507\pm 0.0022 0.1471±0.0023plus-or-minus0.14710.00230.1471\pm 0.0023  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.1514±0.0029plus-or-minus0.15140.00290.1514\pm 0.0029 0.1462±0.0031plus-or-minus0.14620.00310.1462\pm 0.0031  T2GT2G\mathrm{T2G} 0.1523±0.0023plus-or-minus0.15230.00230.1523\pm 0.0023 0.1478±0.0024plus-or-minus0.14780.00240.1478\pm 0.0024  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.1514±0.0025plus-or-minus0.15140.00250.1514\pm 0.0025 0.1479±0.0017plus-or-minus0.14790.00170.1479\pm 0.0017  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.1512±0.0019plus-or-minus0.15120.00190.1512\pm 0.0019 0.1470±0.0024plus-or-minus0.14700.00240.1470\pm 0.0024  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.1461±0.0015plus-or-minus0.14610.00150.1461\pm 0.0015 0.1433±0.0022plus-or-minus0.14330.00220.1433\pm 0.0022  XGBoostXGBoost\mathrm{XGBoost} 0.1440±0.0029plus-or-minus0.14400.00290.1440\pm 0.0029 0.1434±0.0029plus-or-minus0.14340.00290.1434\pm 0.0029  LightGBMLightGBM\mathrm{LightGBM} 0.1461±0.0025plus-or-minus0.14610.00250.1461\pm 0.0025 0.1455±0.0030plus-or-minus0.14550.00300.1455\pm 0.0030  CatBoostCatBoost\mathrm{CatBoost} 0.1417±0.0021plus-or-minus0.14170.00210.1417\pm 0.0021 0.1408±0.0026plus-or-minus0.14080.00260.1408\pm 0.0026  TabRTabR\mathrm{TabR} 0.1417±0.0025plus-or-minus0.14170.00250.1417\pm 0.0025 0.1390±0.0020plus-or-minus0.13900.00200.1390\pm 0.0020  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.1392±0.0023plus-or-minus0.13920.00230.1392\pm 0.0023 –  MNCAMNCA\mathrm{MNCA} 0.1503±0.0040plus-or-minus0.15030.00400.1503\pm 0.0040 0.1477±0.0032plus-or-minus0.14770.00320.1477\pm 0.0032  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.1475±0.0031plus-or-minus0.14750.00310.1475\pm 0.0031 0.1438±0.0024plus-or-minus0.14380.00240.1438\pm 0.0024  TabMTabM\mathrm{TabM} 0.1483±0.0030plus-or-minus0.14830.00300.1483\pm 0.0030 0.1465±0.0029plus-or-minus0.14650.00290.1465\pm 0.0029  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.1487±0.0029plus-or-minus0.14870.00290.1487\pm 0.0029 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.1508±0.0035plus-or-minus0.15080.00350.1508\pm 0.0035 0.1484±0.0036plus-or-minus0.14840.00360.1484\pm 0.0036  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.1407±0.0016plus-or-minus0.14070.00160.1407\pm 0.0016 0.1387±0.0008plus-or-minus0.13870.00080.1387\pm 0.0008 |
| OnlineNewsPopularity ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8643±0.0007plus-or-minus0.86430.00070.8643\pm 0.0007 0.8632±0.0005plus-or-minus0.86320.00050.8632\pm 0.0005  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.8665±0.0011plus-or-minus0.86650.00110.8665\pm 0.0011 0.8639±0.0000plus-or-minus0.86390.00000.8639\pm 0.0000  DCN2DCN2\mathrm{DCN2} 0.8714±0.0013plus-or-minus0.87140.00130.8714\pm 0.0013 0.8648±0.0004plus-or-minus0.86480.00040.8648\pm 0.0004  SNNSNN\mathrm{SNN} 0.8692±0.0015plus-or-minus0.86920.00150.8692\pm 0.0015 0.8665±0.0005plus-or-minus0.86650.00050.8665\pm 0.0005  TromptTrompt\mathrm{Trompt} 0.8623±n​a​nplus-or-minus0.8623𝑛𝑎𝑛0.8623\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.8636±0.0022plus-or-minus0.86360.00220.8636\pm 0.0022 0.8596±0.0008plus-or-minus0.85960.00080.8596\pm 0.0008  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.8615±0.0008plus-or-minus0.86150.00080.8615\pm 0.0008 0.8598±0.0004plus-or-minus0.85980.00040.8598\pm 0.0004  ExcelExcel\mathrm{Excel} 0.8605±0.0024plus-or-minus0.86050.00240.8605\pm 0.0024 0.8556±n​a​nplus-or-minus0.8556𝑛𝑎𝑛0.8556\pm nan  SAINTSAINT\mathrm{SAINT} 0.8600±0.0007plus-or-minus0.86000.00070.8600\pm 0.0007 0.8582±0.0003plus-or-minus0.85820.00030.8582\pm 0.0003  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8629±0.0019plus-or-minus0.86290.00190.8629\pm 0.0019 0.8603±0.0000plus-or-minus0.86030.00000.8603\pm 0.0000  T2GT2G\mathrm{T2G} 0.8632±0.0009plus-or-minus0.86320.00090.8632\pm 0.0009 0.8572±n​a​nplus-or-minus0.8572𝑛𝑎𝑛0.8572\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8604±0.0009plus-or-minus0.86040.00090.8604\pm 0.0009 0.8591±0.0004plus-or-minus0.85910.00040.8591\pm 0.0004  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8594±0.0004plus-or-minus0.85940.00040.8594\pm 0.0004 0.8585±0.0001plus-or-minus0.85850.00010.8585\pm 0.0001  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8585±0.0003plus-or-minus0.85850.00030.8585\pm 0.0003 0.8581±0.0001plus-or-minus0.85810.00010.8581\pm 0.0001  XGBoostXGBoost\mathrm{XGBoost} 0.8545±0.0002plus-or-minus0.85450.00020.8545\pm 0.0002 0.8543±0.0000plus-or-minus0.85430.00000.8543\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.8546±0.0002plus-or-minus0.85460.00020.8546\pm 0.0002 0.8544±0.0000plus-or-minus0.85440.00000.8544\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.8532±0.0003plus-or-minus0.85320.00030.8532\pm 0.0003 0.8527±0.0001plus-or-minus0.85270.00010.8527\pm 0.0001  TabRTabR\mathrm{TabR} 0.8677±0.0013plus-or-minus0.86770.00130.8677\pm 0.0013 0.8633±0.0009plus-or-minus0.86330.00090.8633\pm 0.0009  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8624±0.0011plus-or-minus0.86240.00110.8624\pm 0.0011 –  MNCAMNCA\mathrm{MNCA} 0.8651±0.0003plus-or-minus0.86510.00030.8651\pm 0.0003 0.8650±0.0002plus-or-minus0.86500.00020.8650\pm 0.0002  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8647±0.0010plus-or-minus0.86470.00100.8647\pm 0.0010 0.8624±0.0006plus-or-minus0.86240.00060.8624\pm 0.0006  TabMTabM\mathrm{TabM} 0.8584±0.0003plus-or-minus0.85840.00030.8584\pm 0.0003 0.8581±0.0001plus-or-minus0.85810.00010.8581\pm 0.0001  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8586±0.0005plus-or-minus0.85860.00050.8586\pm 0.0005 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8592±0.0004plus-or-minus0.85920.00040.8592\pm 0.0004 0.8588±0.0001plus-or-minus0.85880.00010.8588\pm 0.0001  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8560±0.0015plus-or-minus0.85600.00150.8560\pm 0.0015 0.8532±0.0008plus-or-minus0.85320.00080.8532\pm 0.0008 | Bike\_Sharing\_Demand ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 45.0186±0.7700plus-or-minus45.01860.770045.0186\pm 0.7700 43.2726±0.5498plus-or-minus43.27260.549843.2726\pm 0.5498  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 49.5584±1.1603plus-or-minus49.55841.160349.5584\pm 1.1603 48.5113±0.3627plus-or-minus48.51130.362748.5113\pm 0.3627  DCN2DCN2\mathrm{DCN2} 45.2596±0.9906plus-or-minus45.25960.990645.2596\pm 0.9906 43.2049±0.3088plus-or-minus43.20490.308843.2049\pm 0.3088  SNNSNN\mathrm{SNN} 48.0917±1.1852plus-or-minus48.09171.185248.0917\pm 1.1852 44.6840±1.0755plus-or-minus44.68401.075544.6840\pm 1.0755  TromptTrompt\mathrm{Trompt} 42.1566±0.4300plus-or-minus42.15660.430042.1566\pm 0.4300 –  AutoIntAutoInt\mathrm{AutoInt} 43.5852±0.7439plus-or-minus43.58520.743943.5852\pm 0.7439 41.6339±0.2132plus-or-minus41.63390.213241.6339\pm 0.2132  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 43.1481±0.6971plus-or-minus43.14810.697143.1481\pm 0.6971 40.8738±0.3218plus-or-minus40.87380.321840.8738\pm 0.3218  ExcelExcel\mathrm{Excel} 43.6145±0.7073plus-or-minus43.61450.707343.6145\pm 0.7073 41.0973±0.2037plus-or-minus41.09730.203741.0973\pm 0.2037  SAINTSAINT\mathrm{SAINT} 42.7850±0.4637plus-or-minus42.78500.463742.7850\pm 0.4637 41.8555±0.4083plus-or-minus41.85550.408341.8555\pm 0.4083  FT​-​TFT-T\mathrm{FT\texttt{-}T} 43.2031±0.4889plus-or-minus43.20310.488943.2031\pm 0.4889 41.1763±0.3443plus-or-minus41.17630.344341.1763\pm 0.3443  T2GT2G\mathrm{T2G} 42.5226±0.6449plus-or-minus42.52260.644942.5226\pm 0.6449 40.9824±0.6087plus-or-minus40.98240.608740.9824\pm 0.6087  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 43.1846±1.1145plus-or-minus43.18461.114543.1846\pm 1.1145 41.3309±0.2381plus-or-minus41.33090.238141.3309\pm 0.2381  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 42.4757±0.4165plus-or-minus42.47570.416542.4757\pm 0.4165 41.2681±0.1946plus-or-minus41.26810.194641.2681\pm 0.1946  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 42.5106±0.4022plus-or-minus42.51060.402242.5106\pm 0.4022 41.4351±0.1280plus-or-minus41.43510.128041.4351\pm 0.1280  XGBoostXGBoost\mathrm{XGBoost} 42.7657±0.1260plus-or-minus42.76570.126042.7657\pm 0.1260 42.6060±0.0391plus-or-minus42.60600.039142.6060\pm 0.0391  LightGBMLightGBM\mathrm{LightGBM} 42.5028±0.1896plus-or-minus42.50280.189642.5028\pm 0.1896 42.3416±0.1492plus-or-minus42.34160.149242.3416\pm 0.1492  CatBoostCatBoost\mathrm{CatBoost} 40.9275±0.2316plus-or-minus40.92750.231640.9275\pm 0.2316 40.5515±0.0898plus-or-minus40.55150.089840.5515\pm 0.0898  TabRTabR\mathrm{TabR} 43.6370±0.6814plus-or-minus43.63700.681443.6370\pm 0.6814 42.3390±0.4146plus-or-minus42.33900.414642.3390\pm 0.4146  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 42.6486±0.9394plus-or-minus42.64860.939442.6486\pm 0.9394 –  MNCAMNCA\mathrm{MNCA} 44.8100±0.5191plus-or-minus44.81000.519144.8100\pm 0.5191 44.4483±0.4231plus-or-minus44.44830.423144.4483\pm 0.4231  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 42.6308±0.8834plus-or-minus42.63080.883442.6308\pm 0.8834 41.6584±0.5771plus-or-minus41.65840.577141.6584\pm 0.5771  TabMTabM\mathrm{TabM} 42.1081±0.5016plus-or-minus42.10810.501642.1081\pm 0.5016 41.3316±0.3496plus-or-minus41.33160.349641.3316\pm 0.3496  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 42.0802±0.4102plus-or-minus42.08020.410242.0802\pm 0.4102 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 42.2073±0.5623plus-or-minus42.20730.562342.2073\pm 0.5623 41.3252±0.1159plus-or-minus41.32520.115941.3252\pm 0.1159  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 41.3374±0.6326plus-or-minus41.33740.632641.3374\pm 0.6326 40.4473±0.5201plus-or-minus40.44730.520140.4473\pm 0.5201 |
| credit ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7735±0.0042plus-or-minus0.77350.00420.7735\pm 0.0042 0.7729±0.0047plus-or-minus0.77290.00470.7729\pm 0.0047  TabPFNTabPFN\mathrm{TabPFN} – 0.7636±0.0045plus-or-minus0.76360.00450.7636\pm 0.0045  ResNetResNet\mathrm{ResNet} 0.7721±0.0033plus-or-minus0.77210.00330.7721\pm 0.0033 0.7738±0.0027plus-or-minus0.77380.00270.7738\pm 0.0027  DCN2DCN2\mathrm{DCN2} 0.7703±0.0034plus-or-minus0.77030.00340.7703\pm 0.0034 0.7746±0.0026plus-or-minus0.77460.00260.7746\pm 0.0026  SNNSNN\mathrm{SNN} 0.7712±0.0045plus-or-minus0.77120.00450.7712\pm 0.0045 0.7716±0.0059plus-or-minus0.77160.00590.7716\pm 0.0059  TromptTrompt\mathrm{Trompt} 0.7740±0.0006plus-or-minus0.77400.00060.7740\pm 0.0006 –  AutoIntAutoInt\mathrm{AutoInt} 0.7737±0.0050plus-or-minus0.77370.00500.7737\pm 0.0050 0.7765±0.0058plus-or-minus0.77650.00580.7765\pm 0.0058  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7748±0.0038plus-or-minus0.77480.00380.7748\pm 0.0038 0.7768±0.0059plus-or-minus0.77680.00590.7768\pm 0.0059  ExcelExcel\mathrm{Excel} 0.7724±0.0038plus-or-minus0.77240.00380.7724\pm 0.0038 0.7740±0.0069plus-or-minus0.77400.00690.7740\pm 0.0069  SAINTSAINT\mathrm{SAINT} 0.7739±0.0052plus-or-minus0.77390.00520.7739\pm 0.0052 0.7749±0.0066plus-or-minus0.77490.00660.7749\pm 0.0066  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7745±0.0041plus-or-minus0.77450.00410.7745\pm 0.0041 0.7767±0.0040plus-or-minus0.77670.00400.7767\pm 0.0040  T2GT2G\mathrm{T2G} 0.7744±0.0046plus-or-minus0.77440.00460.7744\pm 0.0046 0.7762±0.0057plus-or-minus0.77620.00570.7762\pm 0.0057  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7749±0.0055plus-or-minus0.77490.00550.7749\pm 0.0055 0.7767±0.0075plus-or-minus0.77670.00750.7767\pm 0.0075  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7734±0.0034plus-or-minus0.77340.00340.7734\pm 0.0034 0.7747±0.0043plus-or-minus0.77470.00430.7747\pm 0.0043  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7758±0.0040plus-or-minus0.77580.00400.7758\pm 0.0040 0.7772±0.0055plus-or-minus0.77720.00550.7772\pm 0.0055  XGBoostXGBoost\mathrm{XGBoost} 0.7698±0.0027plus-or-minus0.76980.00270.7698\pm 0.0027 0.7706±0.0029plus-or-minus0.77060.00290.7706\pm 0.0029  LightGBMLightGBM\mathrm{LightGBM} 0.7686±0.0028plus-or-minus0.76860.00280.7686\pm 0.0028 0.7726±0.0034plus-or-minus0.77260.00340.7726\pm 0.0034  CatBoostCatBoost\mathrm{CatBoost} 0.7734±0.0035plus-or-minus0.77340.00350.7734\pm 0.0035 0.7752±0.0038plus-or-minus0.77520.00380.7752\pm 0.0038  TabRTabR\mathrm{TabR} 0.7730±0.0043plus-or-minus0.77300.00430.7730\pm 0.0043 0.7740±0.0040plus-or-minus0.77400.00400.7740\pm 0.0040  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7723±0.0037plus-or-minus0.77230.00370.7723\pm 0.0037 –  MNCAMNCA\mathrm{MNCA} 0.7739±0.0032plus-or-minus0.77390.00320.7739\pm 0.0032 0.7757±0.0026plus-or-minus0.77570.00260.7757\pm 0.0026  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7734±0.0045plus-or-minus0.77340.00450.7734\pm 0.0045 0.7754±0.0040plus-or-minus0.77540.00400.7754\pm 0.0040  TabMTabM\mathrm{TabM} 0.7751±0.0042plus-or-minus0.77510.00420.7751\pm 0.0042 0.7755±0.0049plus-or-minus0.77550.00490.7755\pm 0.0049  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7744±0.0036plus-or-minus0.77440.00360.7744\pm 0.0036 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7747±0.0039plus-or-minus0.77470.00390.7747\pm 0.0039 0.7758±0.0042plus-or-minus0.77580.00420.7758\pm 0.0042  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.7748±0.0026plus-or-minus0.77480.00260.7748\pm 0.0026 0.7757±0.0036plus-or-minus0.77570.00360.7757\pm 0.0036 | elevators ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0020±0.0001plus-or-minus0.00200.00010.0020\pm 0.0001 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  DCN2DCN2\mathrm{DCN2} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  SNNSNN\mathrm{SNN} 0.0020±0.0001plus-or-minus0.00200.00010.0020\pm 0.0001 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  TromptTrompt\mathrm{Trompt} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 –  AutoIntAutoInt\mathrm{AutoInt} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  ExcelExcel\mathrm{Excel} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  SAINTSAINT\mathrm{SAINT} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  T2GT2G\mathrm{T2G} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  XGBoostXGBoost\mathrm{XGBoost} 0.0020±0.0000plus-or-minus0.00200.00000.0020\pm 0.0000 0.0020±0.0000plus-or-minus0.00200.00000.0020\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.0020±0.0000plus-or-minus0.00200.00000.0020\pm 0.0000 0.0020±0.0000plus-or-minus0.00200.00000.0020\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.0020±0.0000plus-or-minus0.00200.00000.0020\pm 0.0000 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  TabRTabR\mathrm{TabR} 0.0049±0.0000plus-or-minus0.00490.00000.0049\pm 0.0000 0.0049±0.0000plus-or-minus0.00490.00000.0049\pm 0.0000  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0019±0.0001plus-or-minus0.00190.00010.0019\pm 0.0001 –  MNCAMNCA\mathrm{MNCA} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  TabMTabM\mathrm{TabM} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000 0.0019±0.0000plus-or-minus0.00190.00000.0019\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 0.0018±0.0000plus-or-minus0.00180.00000.0018\pm 0.0000 |
| fifa ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8038±0.0124plus-or-minus0.80380.01240.8038\pm 0.0124 0.8011±0.0143plus-or-minus0.80110.01430.8011\pm 0.0143  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.8025±0.0140plus-or-minus0.80250.01400.8025\pm 0.0140 0.7985±0.0149plus-or-minus0.79850.01490.7985\pm 0.0149  DCN2DCN2\mathrm{DCN2} 0.8046±0.0135plus-or-minus0.80460.01350.8046\pm 0.0135 0.7993±0.0129plus-or-minus0.79930.01290.7993\pm 0.0129  SNNSNN\mathrm{SNN} 0.8074±0.0140plus-or-minus0.80740.01400.8074\pm 0.0140 0.8031±0.0147plus-or-minus0.80310.01470.8031\pm 0.0147  TromptTrompt\mathrm{Trompt} 0.7880±0.0180plus-or-minus0.78800.01800.7880\pm 0.0180 –  AutoIntAutoInt\mathrm{AutoInt} 0.7923±0.0128plus-or-minus0.79230.01280.7923\pm 0.0128 0.7886±0.0127plus-or-minus0.78860.01270.7886\pm 0.0127  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7936±0.0119plus-or-minus0.79360.01190.7936\pm 0.0119 0.7903±0.0133plus-or-minus0.79030.01330.7903\pm 0.0133  ExcelExcel\mathrm{Excel} 0.7909±0.0111plus-or-minus0.79090.01110.7909\pm 0.0111 0.7862±0.0161plus-or-minus0.78620.01610.7862\pm 0.0161  SAINTSAINT\mathrm{SAINT} 0.7901±0.0118plus-or-minus0.79010.01180.7901\pm 0.0118 0.7851±0.0119plus-or-minus0.78510.01190.7851\pm 0.0119  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7928±0.0132plus-or-minus0.79280.01320.7928\pm 0.0132 0.7888±0.0130plus-or-minus0.78880.01300.7888\pm 0.0130  T2GT2G\mathrm{T2G} 0.7928±0.0139plus-or-minus0.79280.01390.7928\pm 0.0139 0.7904±0.0183plus-or-minus0.79040.01830.7904\pm 0.0183  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7940±0.0118plus-or-minus0.79400.01180.7940\pm 0.0118 0.7898±0.0141plus-or-minus0.78980.01410.7898\pm 0.0141  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7907±0.0092plus-or-minus0.79070.00920.7907\pm 0.0092 0.7870±0.0096plus-or-minus0.78700.00960.7870\pm 0.0096  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7806±0.0104plus-or-minus0.78060.01040.7806\pm 0.0104 0.7800±0.0114plus-or-minus0.78000.01140.7800\pm 0.0114  XGBoostXGBoost\mathrm{XGBoost} 0.7800±0.0108plus-or-minus0.78000.01080.7800\pm 0.0108 0.7795±0.0114plus-or-minus0.77950.01140.7795\pm 0.0114  LightGBMLightGBM\mathrm{LightGBM} 0.7806±0.0120plus-or-minus0.78060.01200.7806\pm 0.0120 0.7787±0.0122plus-or-minus0.77870.01220.7787\pm 0.0122  CatBoostCatBoost\mathrm{CatBoost} 0.7835±0.0116plus-or-minus0.78350.01160.7835\pm 0.0116 0.7817±0.0114plus-or-minus0.78170.01140.7817\pm 0.0114  TabRTabR\mathrm{TabR} 0.7902±0.0119plus-or-minus0.79020.01190.7902\pm 0.0119 0.7863±0.0120plus-or-minus0.78630.01200.7863\pm 0.0120  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.7914±0.0136plus-or-minus0.79140.01360.7914\pm 0.0136 –  MNCAMNCA\mathrm{MNCA} 0.7967±0.0138plus-or-minus0.79670.01380.7967\pm 0.0138 0.7933±0.0145plus-or-minus0.79330.01450.7933\pm 0.0145  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.7909±0.0107plus-or-minus0.79090.01070.7909\pm 0.0107 0.7866±0.0106plus-or-minus0.78660.01060.7866\pm 0.0106  TabMTabM\mathrm{TabM} 0.7974±0.0144plus-or-minus0.79740.01440.7974\pm 0.0144 0.7954±0.0160plus-or-minus0.79540.01600.7954\pm 0.0160  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7970±0.0146plus-or-minus0.79700.01460.7970\pm 0.0146 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7981±0.0136plus-or-minus0.79810.01360.7981\pm 0.0136 0.7947±0.0154plus-or-minus0.79470.01540.7947\pm 0.0154  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.7783±0.0114plus-or-minus0.77830.01140.7783\pm 0.0114 0.7768±0.0123plus-or-minus0.77680.01230.7768\pm 0.0123 | house\_sales ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.1790±0.0009plus-or-minus0.17900.00090.1790\pm 0.0009 0.1763±0.0003plus-or-minus0.17630.00030.1763\pm 0.0003  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.1755±0.0014plus-or-minus0.17550.00140.1755\pm 0.0014 0.1738±0.0006plus-or-minus0.17380.00060.1738\pm 0.0006  DCN2DCN2\mathrm{DCN2} 0.1862±0.0032plus-or-minus0.18620.00320.1862\pm 0.0032 0.1778±0.0015plus-or-minus0.17780.00150.1778\pm 0.0015  SNNSNN\mathrm{SNN} 0.1800±0.0008plus-or-minus0.18000.00080.1800\pm 0.0008 0.1770±0.0004plus-or-minus0.17700.00040.1770\pm 0.0004  TromptTrompt\mathrm{Trompt} 0.1667±n​a​nplus-or-minus0.1667𝑛𝑎𝑛0.1667\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.1700±0.0014plus-or-minus0.17000.00140.1700\pm 0.0014 0.1670±0.0008plus-or-minus0.16700.00080.1670\pm 0.0008  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.1704±0.0007plus-or-minus0.17040.00070.1704\pm 0.0007 0.1690±0.0005plus-or-minus0.16900.00050.1690\pm 0.0005  ExcelExcel\mathrm{Excel} 0.1713±0.0010plus-or-minus0.17130.00100.1713\pm 0.0010 0.1668±n​a​nplus-or-minus0.1668𝑛𝑎𝑛0.1668\pm nan  SAINTSAINT\mathrm{SAINT} 0.1713±0.0015plus-or-minus0.17130.00150.1713\pm 0.0015 0.1685±0.0005plus-or-minus0.16850.00050.1685\pm 0.0005  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.1690±0.0010plus-or-minus0.16900.00100.1690\pm 0.0010 0.1659±0.0004plus-or-minus0.16590.00040.1659\pm 0.0004  T2GT2G\mathrm{T2G} 0.1689±0.0010plus-or-minus0.16890.00100.1689\pm 0.0010 0.1664±n​a​nplus-or-minus0.1664𝑛𝑎𝑛0.1664\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.1699±0.0008plus-or-minus0.16990.00080.1699\pm 0.0008 0.1687±0.0007plus-or-minus0.16870.00070.1687\pm 0.0007  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.1690±0.0005plus-or-minus0.16900.00050.1690\pm 0.0005 0.1676±0.0003plus-or-minus0.16760.00030.1676\pm 0.0003  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.1687±0.0004plus-or-minus0.16870.00040.1687\pm 0.0004 0.1681±0.0001plus-or-minus0.16810.00010.1681\pm 0.0001  XGBoostXGBoost\mathrm{XGBoost} 0.1694±0.0003plus-or-minus0.16940.00030.1694\pm 0.0003 0.1689±0.0001plus-or-minus0.16890.00010.1689\pm 0.0001  LightGBMLightGBM\mathrm{LightGBM} 0.1692±0.0004plus-or-minus0.16920.00040.1692\pm 0.0004 0.1686±0.0001plus-or-minus0.16860.00010.1686\pm 0.0001  CatBoostCatBoost\mathrm{CatBoost} 0.1669±0.0001plus-or-minus0.16690.00010.1669\pm 0.0001 0.1667±0.0000plus-or-minus0.16670.00000.1667\pm 0.0000  TabRTabR\mathrm{TabR} 0.1689±0.0009plus-or-minus0.16890.00090.1689\pm 0.0009 0.1657±0.0003plus-or-minus0.16570.00030.1657\pm 0.0003  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.1636±0.0009plus-or-minus0.16360.00090.1636\pm 0.0009 –  MNCAMNCA\mathrm{MNCA} 0.1737±0.0013plus-or-minus0.17370.00130.1737\pm 0.0013 0.1714±0.0005plus-or-minus0.17140.00050.1714\pm 0.0005  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.1694±0.0007plus-or-minus0.16940.00070.1694\pm 0.0007 0.1670±0.0003plus-or-minus0.16700.00030.1670\pm 0.0003  TabMTabM\mathrm{TabM} 0.1692±0.0011plus-or-minus0.16920.00110.1692\pm 0.0011 0.1680±0.0005plus-or-minus0.16800.00050.1680\pm 0.0005  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.1692±0.0009plus-or-minus0.16920.00090.1692\pm 0.0009 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.1687±0.0009plus-or-minus0.16870.00090.1687\pm 0.0009 0.1676±0.0002plus-or-minus0.16760.00020.1676\pm 0.0002  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.1656±0.0005plus-or-minus0.16560.00050.1656\pm 0.0005 0.1647±0.0002plus-or-minus0.16470.00020.1647\pm 0.0002 |
| medical\_charges ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0816±0.0001plus-or-minus0.08160.00010.0816\pm 0.0001 0.0814±0.0000plus-or-minus0.08140.00000.0814\pm 0.0000  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0824±0.0003plus-or-minus0.08240.00030.0824\pm 0.0003 0.0817±0.0001plus-or-minus0.08170.00010.0817\pm 0.0001  DCN2DCN2\mathrm{DCN2} 0.0818±0.0003plus-or-minus0.08180.00030.0818\pm 0.0003 0.0815±0.0001plus-or-minus0.08150.00010.0815\pm 0.0001  SNNSNN\mathrm{SNN} 0.0827±0.0006plus-or-minus0.08270.00060.0827\pm 0.0006 0.0817±0.0001plus-or-minus0.08170.00010.0817\pm 0.0001  TromptTrompt\mathrm{Trompt} 0.0812±n​a​nplus-or-minus0.0812𝑛𝑎𝑛0.0812\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.0822±0.0007plus-or-minus0.08220.00070.0822\pm 0.0007 0.0814±0.0001plus-or-minus0.08140.00010.0814\pm 0.0001  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0814±0.0002plus-or-minus0.08140.00020.0814\pm 0.0002 0.0811±0.0000plus-or-minus0.08110.00000.0811\pm 0.0000  ExcelExcel\mathrm{Excel} 0.0817±0.0004plus-or-minus0.08170.00040.0817\pm 0.0004 0.0813±n​a​nplus-or-minus0.0813𝑛𝑎𝑛0.0813\pm nan  SAINTSAINT\mathrm{SAINT} 0.0814±0.0002plus-or-minus0.08140.00020.0814\pm 0.0002 0.0812±0.0001plus-or-minus0.08120.00010.0812\pm 0.0001  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0814±0.0002plus-or-minus0.08140.00020.0814\pm 0.0002 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000  T2GT2G\mathrm{T2G} 0.0813±0.0002plus-or-minus0.08130.00020.0813\pm 0.0002 0.0811±n​a​nplus-or-minus0.0811𝑛𝑎𝑛0.0811\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0812±0.0002plus-or-minus0.08120.00020.0812\pm 0.0002 0.0810±0.0000plus-or-minus0.08100.00000.0810\pm 0.0000  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0812±0.0001plus-or-minus0.08120.00010.0812\pm 0.0001 0.0809±0.0001plus-or-minus0.08090.00010.0809\pm 0.0001  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000 0.0811±0.0000plus-or-minus0.08110.00000.0811\pm 0.0000  XGBoostXGBoost\mathrm{XGBoost} 0.0825±0.0001plus-or-minus0.08250.00010.0825\pm 0.0001 0.0825±0.0000plus-or-minus0.08250.00000.0825\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.0820±0.0000plus-or-minus0.08200.00000.0820\pm 0.0000 0.0820±0.0000plus-or-minus0.08200.00000.0820\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.0816±0.0000plus-or-minus0.08160.00000.0816\pm 0.0000 0.0815±0.0000plus-or-minus0.08150.00000.0815\pm 0.0000  TabRTabR\mathrm{TabR} 0.0815±0.0002plus-or-minus0.08150.00020.0815\pm 0.0002 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0811±0.0001plus-or-minus0.08110.00010.0811\pm 0.0001 –  MNCAMNCA\mathrm{MNCA} 0.0811±0.0001plus-or-minus0.08110.00010.0811\pm 0.0001 0.0810±0.0000plus-or-minus0.08100.00000.0810\pm 0.0000  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0809±0.0000plus-or-minus0.08090.00000.0809\pm 0.0000 0.0808±0.0000plus-or-minus0.08080.00000.0808\pm 0.0000  TabMTabM\mathrm{TabM} 0.0813±0.0001plus-or-minus0.08130.00010.0813\pm 0.0001 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0814±0.0001plus-or-minus0.08140.00010.0814\pm 0.0001 0.0813±0.0000plus-or-minus0.08130.00000.0813\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0812±0.0001plus-or-minus0.08120.00010.0812\pm 0.0001 0.0812±0.0000plus-or-minus0.08120.00000.0812\pm 0.0000 | pol ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 5.5244±0.5768plus-or-minus5.52440.57685.5244\pm 0.5768 4.9945±0.5923plus-or-minus4.99450.59234.9945\pm 0.5923  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 6.3739±0.6286plus-or-minus6.37390.62866.3739\pm 0.6286 5.8181±0.6054plus-or-minus5.81810.60545.8181\pm 0.6054  DCN2DCN2\mathrm{DCN2} 6.5374±0.9479plus-or-minus6.53740.94796.5374\pm 0.9479 5.1814±0.7775plus-or-minus5.18140.77755.1814\pm 0.7775  SNNSNN\mathrm{SNN} 6.1816±0.7366plus-or-minus6.18160.73666.1816\pm 0.7366 5.5959±0.8243plus-or-minus5.59590.82435.5959\pm 0.8243  TromptTrompt\mathrm{Trompt} 3.2337±0.0605plus-or-minus3.23370.06053.2337\pm 0.0605 –  AutoIntAutoInt\mathrm{AutoInt} 3.3295±0.3379plus-or-minus3.32950.33793.3295\pm 0.3379 2.7999±0.1776plus-or-minus2.79990.17762.7999\pm 0.1776  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 3.2011±0.2921plus-or-minus3.20110.29213.2011\pm 0.2921 2.8698±0.2577plus-or-minus2.86980.25772.8698\pm 0.2577  ExcelExcel\mathrm{Excel} 3.0682±0.2389plus-or-minus3.06820.23893.0682\pm 0.2389 2.5816±0.0368plus-or-minus2.58160.03682.5816\pm 0.0368  SAINTSAINT\mathrm{SAINT} 2.7203±0.1858plus-or-minus2.72030.18582.7203\pm 0.1858 2.4507±0.1153plus-or-minus2.45070.11532.4507\pm 0.1153  FT​-​TFT-T\mathrm{FT\texttt{-}T} 2.6974±0.1666plus-or-minus2.69740.16662.6974\pm 0.1666 2.3718±0.0724plus-or-minus2.37180.07242.3718\pm 0.0724  T2GT2G\mathrm{T2G} 2.9539±0.1994plus-or-minus2.95390.19942.9539\pm 0.1994 2.6282±0.0730plus-or-minus2.62820.07302.6282\pm 0.0730  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 2.8239±0.2173plus-or-minus2.82390.21732.8239\pm 0.2173 2.5266±0.0605plus-or-minus2.52660.06052.5266\pm 0.0605  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 2.5452±0.1221plus-or-minus2.54520.12212.5452\pm 0.1221 2.3700±0.0867plus-or-minus2.37000.08672.3700\pm 0.0867  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 2.4958±0.1292plus-or-minus2.49580.12922.4958\pm 0.1292 2.3651±0.1223plus-or-minus2.36510.12232.3651\pm 0.1223  XGBoostXGBoost\mathrm{XGBoost} 4.2963±0.0644plus-or-minus4.29630.06444.2963\pm 0.0644 4.2548±0.0488plus-or-minus4.25480.04884.2548\pm 0.0488  LightGBMLightGBM\mathrm{LightGBM} 4.2320±0.3369plus-or-minus4.23200.33694.2320\pm 0.3369 4.1880±0.3110plus-or-minus4.18800.31104.1880\pm 0.3110  CatBoostCatBoost\mathrm{CatBoost} 3.6320±0.1006plus-or-minus3.63200.10063.6320\pm 0.1006 3.5505±0.0896plus-or-minus3.55050.08963.5505\pm 0.0896  TabRTabR\mathrm{TabR} 6.0708±0.5368plus-or-minus6.07080.53686.0708\pm 0.5368 5.5578±0.4036plus-or-minus5.55780.40365.5578\pm 0.4036  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 2.5770±0.1689plus-or-minus2.57700.16892.5770\pm 0.1689 –  MNCAMNCA\mathrm{MNCA} 5.7878±0.4884plus-or-minus5.78780.48845.7878\pm 0.4884 5.3773±0.5463plus-or-minus5.37730.54635.3773\pm 0.5463  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 2.9083±0.1364plus-or-minus2.90830.13642.9083\pm 0.1364 2.6717±0.0530plus-or-minus2.67170.05302.6717\pm 0.0530  TabMTabM\mathrm{TabM} 3.3595±0.4017plus-or-minus3.35950.40173.3595\pm 0.4017 3.2130±0.3979plus-or-minus3.21300.39793.2130\pm 0.3979  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 3.3465±0.4226plus-or-minus3.34650.42263.3465\pm 0.4226 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 3.6925±0.4469plus-or-minus3.69250.44693.6925\pm 0.4469 3.4727±0.3074plus-or-minus3.47270.30743.4727\pm 0.3074  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 2.4893±0.1620plus-or-minus2.48930.16202.4893\pm 0.1620 2.4175±0.1124plus-or-minus2.41750.11242.4175\pm 0.1124 |
| superconduct ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 10.8740±0.0868plus-or-minus10.87400.086810.8740\pm 0.0868 10.4118±0.0429plus-or-minus10.41180.042910.4118\pm 0.0429  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 10.7711±0.1454plus-or-minus10.77110.145410.7711\pm 0.1454 10.3495±0.0168plus-or-minus10.34950.016810.3495\pm 0.0168  DCN2DCN2\mathrm{DCN2} 10.8108±0.0957plus-or-minus10.81080.095710.8108\pm 0.0957 10.4342±0.0179plus-or-minus10.43420.017910.4342\pm 0.0179  SNNSNN\mathrm{SNN} 10.8562±0.1300plus-or-minus10.85620.130010.8562\pm 0.1300 10.3342±0.0509plus-or-minus10.33420.050910.3342\pm 0.0509  TromptTrompt\mathrm{Trompt} 10.4442±n​a​nplus-or-minus10.4442𝑛𝑎𝑛10.4442\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 11.0019±0.1391plus-or-minus11.00190.139111.0019\pm 0.1391 10.4469±0.0521plus-or-minus10.44690.052110.4469\pm 0.0521  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 10.7502±0.0800plus-or-minus10.75020.080010.7502\pm 0.0800 10.3281±0.0450plus-or-minus10.32810.045010.3281\pm 0.0450  ExcelExcel\mathrm{Excel} 11.0879±0.1571plus-or-minus11.08790.157111.0879\pm 0.1571 10.4094±n​a​nplus-or-minus10.4094𝑛𝑎𝑛10.4094\pm nan  SAINTSAINT\mathrm{SAINT} 10.7807±0.1074plus-or-minus10.78070.107410.7807\pm 0.1074 10.4652±0.0267plus-or-minus10.46520.026710.4652\pm 0.0267  FT​-​TFT-T\mathrm{FT\texttt{-}T} 10.8256±0.1692plus-or-minus10.82560.169210.8256\pm 0.1692 10.3391±0.0794plus-or-minus10.33910.079410.3391\pm 0.0794  T2GT2G\mathrm{T2G} 10.8310±0.1406plus-or-minus10.83100.140610.8310\pm 0.1406 10.3017±n​a​nplus-or-minus10.3017𝑛𝑎𝑛10.3017\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 10.5058±0.0758plus-or-minus10.50580.075810.5058\pm 0.0758 10.2322±0.0463plus-or-minus10.23220.046310.2322\pm 0.0463  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 10.5061±0.0330plus-or-minus10.50610.033010.5061\pm 0.0330 10.2440±0.0127plus-or-minus10.24400.012710.2440\pm 0.0127  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 10.7220±0.0757plus-or-minus10.72200.075710.7220\pm 0.0757 10.3758±0.0606plus-or-minus10.37580.060610.3758\pm 0.0606  XGBoostXGBoost\mathrm{XGBoost} 10.1610±0.0201plus-or-minus10.16100.020110.1610\pm 0.0201 10.1413±0.0025plus-or-minus10.14130.002510.1413\pm 0.0025  LightGBMLightGBM\mathrm{LightGBM} 10.1634±0.0118plus-or-minus10.16340.011810.1634\pm 0.0118 10.1552±0.0050plus-or-minus10.15520.005010.1552\pm 0.0050  CatBoostCatBoost\mathrm{CatBoost} 10.2422±0.0222plus-or-minus10.24220.022210.2422\pm 0.0222 10.2116±0.0058plus-or-minus10.21160.005810.2116\pm 0.0058  TabRTabR\mathrm{TabR} 10.8842±0.1073plus-or-minus10.88420.107310.8842\pm 0.1073 10.4800±0.0280plus-or-minus10.48000.028010.4800\pm 0.0280  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 10.3835±0.0562plus-or-minus10.38350.056210.3835\pm 0.0562 –  MNCAMNCA\mathrm{MNCA} 10.4419±0.0640plus-or-minus10.44190.064010.4419\pm 0.0640 10.2926±0.0261plus-or-minus10.29260.026110.2926\pm 0.0261  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 10.5651±0.0616plus-or-minus10.56510.061610.5651\pm 0.0616 10.3155±0.0253plus-or-minus10.31550.025310.3155\pm 0.0253  TabMTabM\mathrm{TabM} 10.3379±0.0338plus-or-minus10.33790.033810.3379\pm 0.0338 10.1943±0.0291plus-or-minus10.19430.029110.1943\pm 0.0291  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 10.3395±0.0529plus-or-minus10.33950.052910.3395\pm 0.0529 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 10.3392±0.0649plus-or-minus10.33920.064910.3392\pm 0.0649 10.1866±0.0400plus-or-minus10.18660.040010.1866\pm 0.0400  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 10.2083±0.0591plus-or-minus10.20830.059110.2083\pm 0.0591 10.0737±0.0222plus-or-minus10.07370.022210.0737\pm 0.0222 | jannis ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7840±0.0018plus-or-minus0.78400.00180.7840\pm 0.0018 0.7872±0.0007plus-or-minus0.78720.00070.7872\pm 0.0007  TabPFNTabPFN\mathrm{TabPFN} – 0.7419±0.0018plus-or-minus0.74190.00180.7419\pm 0.0018  ResNetResNet\mathrm{ResNet} 0.7923±0.0024plus-or-minus0.79230.00240.7923\pm 0.0024 0.7958±0.0010plus-or-minus0.79580.00100.7958\pm 0.0010  DCN2DCN2\mathrm{DCN2} 0.7712±0.0029plus-or-minus0.77120.00290.7712\pm 0.0029 0.7825±0.0009plus-or-minus0.78250.00090.7825\pm 0.0009  SNNSNN\mathrm{SNN} 0.7818±0.0025plus-or-minus0.78180.00250.7818\pm 0.0025 0.7859±0.0011plus-or-minus0.78590.00110.7859\pm 0.0011  TromptTrompt\mathrm{Trompt} 0.8027±n​a​nplus-or-minus0.8027𝑛𝑎𝑛0.8027\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.7933±0.0018plus-or-minus0.79330.00180.7933\pm 0.0018 0.7983±0.0013plus-or-minus0.79830.00130.7983\pm 0.0013  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7927±0.0025plus-or-minus0.79270.00250.7927\pm 0.0025 0.8019±0.0012plus-or-minus0.80190.00120.8019\pm 0.0012  ExcelExcel\mathrm{Excel} 0.7954±0.0015plus-or-minus0.79540.00150.7954\pm 0.0015 0.8021±n​a​nplus-or-minus0.8021𝑛𝑎𝑛0.8021\pm nan  SAINTSAINT\mathrm{SAINT} 0.7971±0.0028plus-or-minus0.79710.00280.7971\pm 0.0028 0.8033±0.0008plus-or-minus0.80330.00080.8033\pm 0.0008  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7940±0.0028plus-or-minus0.79400.00280.7940\pm 0.0028 0.7998±0.0006plus-or-minus0.79980.00060.7998\pm 0.0006  T2GT2G\mathrm{T2G} 0.7998±0.0024plus-or-minus0.79980.00240.7998\pm 0.0024 0.8052±n​a​nplus-or-minus0.8052𝑛𝑎𝑛0.8052\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7923±0.0018plus-or-minus0.79230.00180.7923\pm 0.0018 0.7945±0.0010plus-or-minus0.79450.00100.7945\pm 0.0010  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7947±0.0017plus-or-minus0.79470.00170.7947\pm 0.0017 0.7967±0.0011plus-or-minus0.79670.00110.7967\pm 0.0011  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7891±0.0013plus-or-minus0.78910.00130.7891\pm 0.0013 0.7900±0.0006plus-or-minus0.79000.00060.7900\pm 0.0006  XGBoostXGBoost\mathrm{XGBoost} 0.7967±0.0019plus-or-minus0.79670.00190.7967\pm 0.0019 0.7998±0.0007plus-or-minus0.79980.00070.7998\pm 0.0007  LightGBMLightGBM\mathrm{LightGBM} 0.7956±0.0017plus-or-minus0.79560.00170.7956\pm 0.0017 0.7968±0.0005plus-or-minus0.79680.00050.7968\pm 0.0005  CatBoostCatBoost\mathrm{CatBoost} 0.7985±0.0018plus-or-minus0.79850.00180.7985\pm 0.0018 0.8009±0.0012plus-or-minus0.80090.00120.8009\pm 0.0012  TabRTabR\mathrm{TabR} 0.7983±0.0022plus-or-minus0.79830.00220.7983\pm 0.0022 0.8023±0.0018plus-or-minus0.80230.00180.8023\pm 0.0018  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8051±0.0023plus-or-minus0.80510.00230.8051\pm 0.0023 –  MNCAMNCA\mathrm{MNCA} 0.7993±0.0019plus-or-minus0.79930.00190.7993\pm 0.0019 0.8042±0.0013plus-or-minus0.80420.00130.8042\pm 0.0013  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8068±0.0021plus-or-minus0.80680.00210.8068\pm 0.0021 0.8128±0.0007plus-or-minus0.81280.00070.8128\pm 0.0007  TabMTabM\mathrm{TabM} 0.8066±0.0015plus-or-minus0.80660.00150.8066\pm 0.0015 0.8075±0.0004plus-or-minus0.80750.00040.8075\pm 0.0004  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8055±0.0022plus-or-minus0.80550.00220.8055\pm 0.0022 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8046±0.0026plus-or-minus0.80460.00260.8046\pm 0.0026 0.8062±0.0011plus-or-minus0.80620.00110.8062\pm 0.0011  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8059±0.0018plus-or-minus0.80590.00180.8059\pm 0.0018 0.8085±0.0006plus-or-minus0.80850.00060.8085\pm 0.0006 |
| MiniBooNE ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.9480±0.0007plus-or-minus0.94800.00070.9480\pm 0.0007 0.9498±0.0001plus-or-minus0.94980.00010.9498\pm 0.0001  TabPFNTabPFN\mathrm{TabPFN} – 0.9266±0.0012plus-or-minus0.92660.00120.9266\pm 0.0012  ResNetResNet\mathrm{ResNet} 0.9488±0.0011plus-or-minus0.94880.00110.9488\pm 0.0011 0.9504±0.0005plus-or-minus0.95040.00050.9504\pm 0.0005  DCN2DCN2\mathrm{DCN2} 0.9433±0.0011plus-or-minus0.94330.00110.9433\pm 0.0011 0.9470±0.0010plus-or-minus0.94700.00100.9470\pm 0.0010  SNNSNN\mathrm{SNN} 0.9476±0.0013plus-or-minus0.94760.00130.9476\pm 0.0013 0.9491±0.0010plus-or-minus0.94910.00100.9491\pm 0.0010  TromptTrompt\mathrm{Trompt} 0.9473±n​a​nplus-or-minus0.9473𝑛𝑎𝑛0.9473\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.9447±0.0014plus-or-minus0.94470.00140.9447\pm 0.0014 0.9473±0.0010plus-or-minus0.94730.00100.9473\pm 0.0010  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.9446±0.0014plus-or-minus0.94460.00140.9446\pm 0.0014 0.9483±0.0002plus-or-minus0.94830.00020.9483\pm 0.0002  ExcelExcel\mathrm{Excel} 0.9430±0.0015plus-or-minus0.94300.00150.9430\pm 0.0015 0.9451±n​a​nplus-or-minus0.9451𝑛𝑎𝑛0.9451\pm nan  SAINTSAINT\mathrm{SAINT} 0.9471±0.0009plus-or-minus0.94710.00090.9471\pm 0.0009 0.9485±0.0002plus-or-minus0.94850.00020.9485\pm 0.0002  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.9467±0.0014plus-or-minus0.94670.00140.9467\pm 0.0014 0.9486±0.0010plus-or-minus0.94860.00100.9486\pm 0.0010  T2GT2G\mathrm{T2G} 0.9475±0.0014plus-or-minus0.94750.00140.9475\pm 0.0014 0.9508±n​a​nplus-or-minus0.9508𝑛𝑎𝑛0.9508\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.9466±0.0009plus-or-minus0.94660.00090.9466\pm 0.0009 0.9478±0.0004plus-or-minus0.94780.00040.9478\pm 0.0004  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.9473±0.0010plus-or-minus0.94730.00100.9473\pm 0.0010 0.9493±0.0004plus-or-minus0.94930.00040.9493\pm 0.0004  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.9482±0.0008plus-or-minus0.94820.00080.9482\pm 0.0008 0.9492±0.0001plus-or-minus0.94920.00010.9492\pm 0.0001  XGBoostXGBoost\mathrm{XGBoost} 0.9436±0.0006plus-or-minus0.94360.00060.9436\pm 0.0006 0.9452±0.0003plus-or-minus0.94520.00030.9452\pm 0.0003  LightGBMLightGBM\mathrm{LightGBM} 0.9422±0.0009plus-or-minus0.94220.00090.9422\pm 0.0009 0.9427±0.0003plus-or-minus0.94270.00030.9427\pm 0.0003  CatBoostCatBoost\mathrm{CatBoost} 0.9453±0.0008plus-or-minus0.94530.00080.9453\pm 0.0008 0.9459±0.0005plus-or-minus0.94590.00050.9459\pm 0.0005  TabRTabR\mathrm{TabR} 0.9487±0.0008plus-or-minus0.94870.00080.9487\pm 0.0008 0.9500±0.0002plus-or-minus0.95000.00020.9500\pm 0.0002  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.9475±0.0007plus-or-minus0.94750.00070.9475\pm 0.0007 –  MNCAMNCA\mathrm{MNCA} 0.9488±0.0010plus-or-minus0.94880.00100.9488\pm 0.0010 0.9505±0.0001plus-or-minus0.95050.00010.9505\pm 0.0001  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.9493±0.0012plus-or-minus0.94930.00120.9493\pm 0.0012 0.9501±0.0008plus-or-minus0.95010.00080.9501\pm 0.0008  TabMTabM\mathrm{TabM} 0.9500±0.0005plus-or-minus0.95000.00050.9500\pm 0.0005 0.9505±0.0002plus-or-minus0.95050.00020.9505\pm 0.0002  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.9489±0.0010plus-or-minus0.94890.00100.9489\pm 0.0010 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.9487±0.0006plus-or-minus0.94870.00060.9487\pm 0.0006 0.9494±0.0002plus-or-minus0.94940.00020.9494\pm 0.0002  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.9497±0.0006plus-or-minus0.94970.00060.9497\pm 0.0006 0.9508±0.0003plus-or-minus0.95080.00030.9508\pm 0.0003 | SGEMM\_GPU\_kernel\_performance ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.0165±0.0003plus-or-minus0.01650.00030.0165\pm 0.0003 0.0160±0.0001plus-or-minus0.01600.00010.0160\pm 0.0001  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.0260±0.0032plus-or-minus0.02600.00320.0260\pm 0.0032 0.0236±0.0007plus-or-minus0.02360.00070.0236\pm 0.0007  DCN2DCN2\mathrm{DCN2} 0.0161±0.0005plus-or-minus0.01610.00050.0161\pm 0.0005 0.0157±0.0002plus-or-minus0.01570.00020.0157\pm 0.0002  SNNSNN\mathrm{SNN} 0.0191±0.0008plus-or-minus0.01910.00080.0191\pm 0.0008 0.0169±0.0001plus-or-minus0.01690.00010.0169\pm 0.0001  TromptTrompt\mathrm{Trompt} 0.0158±n​a​nplus-or-minus0.0158𝑛𝑎𝑛0.0158\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.0165±0.0004plus-or-minus0.01650.00040.0165\pm 0.0004 0.0160±0.0003plus-or-minus0.01600.00030.0160\pm 0.0003  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.0164±0.0004plus-or-minus0.01640.00040.0164\pm 0.0004 0.0158±0.0002plus-or-minus0.01580.00020.0158\pm 0.0002  ExcelExcel\mathrm{Excel} 0.0169±0.0009plus-or-minus0.01690.00090.0169\pm 0.0009 0.0159±n​a​nplus-or-minus0.0159𝑛𝑎𝑛0.0159\pm nan  SAINTSAINT\mathrm{SAINT} 0.0158±0.0002plus-or-minus0.01580.00020.0158\pm 0.0002 0.0155±0.0001plus-or-minus0.01550.00010.0155\pm 0.0001  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.0167±0.0007plus-or-minus0.01670.00070.0167\pm 0.0007 0.0159±0.0004plus-or-minus0.01590.00040.0159\pm 0.0004  T2GT2G\mathrm{T2G} 0.0161±0.0007plus-or-minus0.01610.00070.0161\pm 0.0007 0.0154±n​a​nplus-or-minus0.0154𝑛𝑎𝑛0.0154\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.0160±0.0003plus-or-minus0.01600.00030.0160\pm 0.0003 0.0156±0.0000plus-or-minus0.01560.00000.0156\pm 0.0000  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.0159±0.0003plus-or-minus0.01590.00030.0159\pm 0.0003 0.0152±0.0000plus-or-minus0.01520.00000.0152\pm 0.0000  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.0156±0.0000plus-or-minus0.01560.00000.0156\pm 0.0000 0.0154±0.0000plus-or-minus0.01540.00000.0154\pm 0.0000  XGBoostXGBoost\mathrm{XGBoost} 0.0167±0.0000plus-or-minus0.01670.00000.0167\pm 0.0000 0.0167±0.0000plus-or-minus0.01670.00000.0167\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.0168±0.0000plus-or-minus0.01680.00000.0168\pm 0.0000 0.0168±0.0000plus-or-minus0.01680.00000.0168\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.0168±0.0000plus-or-minus0.01680.00000.0168\pm 0.0000 0.0166±0.0000plus-or-minus0.01660.00000.0166\pm 0.0000  TabRTabR\mathrm{TabR} 0.0174±0.0014plus-or-minus0.01740.00140.0174\pm 0.0014 0.0161±0.0005plus-or-minus0.01610.00050.0161\pm 0.0005  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.0154±0.0005plus-or-minus0.01540.00050.0154\pm 0.0005 –  MNCAMNCA\mathrm{MNCA} 0.0147±0.0000plus-or-minus0.01470.00000.0147\pm 0.0000 0.0146±0.0000plus-or-minus0.01460.00000.0146\pm 0.0000  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.0146±0.0002plus-or-minus0.01460.00020.0146\pm 0.0002 0.0145±0.0000plus-or-minus0.01450.00000.0145\pm 0.0000  TabMTabM\mathrm{TabM} 0.0158±0.0004plus-or-minus0.01580.00040.0158\pm 0.0004 0.0155±0.0001plus-or-minus0.01550.00010.0155\pm 0.0001  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.0157±0.0003plus-or-minus0.01570.00030.0157\pm 0.0003 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.0159±0.0003plus-or-minus0.01590.00030.0159\pm 0.0003 0.0156±0.0000plus-or-minus0.01560.00000.0156\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.0156±0.0003plus-or-minus0.01560.00030.0156\pm 0.0003 0.0154±0.0001plus-or-minus0.01540.00010.0154\pm 0.0001 |
| nyc-taxi-green-dec-2016 ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.3951±0.0009plus-or-minus0.39510.00090.3951\pm 0.0009 0.3921±0.0003plus-or-minus0.39210.00030.3921\pm 0.0003  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.3899±0.0016plus-or-minus0.38990.00160.3899\pm 0.0016 0.3873±0.0009plus-or-minus0.38730.00090.3873\pm 0.0009  DCN2DCN2\mathrm{DCN2} 0.3919±0.0009plus-or-minus0.39190.00090.3919\pm 0.0009 0.3889±0.0003plus-or-minus0.38890.00030.3889\pm 0.0003  SNNSNN\mathrm{SNN} 0.3933±0.0013plus-or-minus0.39330.00130.3933\pm 0.0013 0.3899±0.0004plus-or-minus0.38990.00040.3899\pm 0.0004  TromptTrompt\mathrm{Trompt} 0.3979±n​a​nplus-or-minus0.3979𝑛𝑎𝑛0.3979\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.4084±0.0256plus-or-minus0.40840.02560.4084\pm 0.0256 0.3967±0.0059plus-or-minus0.39670.00590.3967\pm 0.0059  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.3914±0.0026plus-or-minus0.39140.00260.3914\pm 0.0026 0.3861±0.0013plus-or-minus0.38610.00130.3861\pm 0.0013  ExcelExcel\mathrm{Excel} 0.3969±0.0036plus-or-minus0.39690.00360.3969\pm 0.0036 0.3897±n​a​nplus-or-minus0.3897𝑛𝑎𝑛0.3897\pm nan  SAINTSAINT\mathrm{SAINT} 0.3905±0.0013plus-or-minus0.39050.00130.3905\pm 0.0013 0.3876±0.0002plus-or-minus0.38760.00020.3876\pm 0.0002  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.3937±0.0064plus-or-minus0.39370.00640.3937\pm 0.0064 0.3889±0.0018plus-or-minus0.38890.00180.3889\pm 0.0018  T2GT2G\mathrm{T2G} 0.3908±0.0045plus-or-minus0.39080.00450.3908\pm 0.0045 0.3858±n​a​nplus-or-minus0.3858𝑛𝑎𝑛0.3858\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.3812±0.0018plus-or-minus0.38120.00180.3812\pm 0.0018 0.3761±0.0016plus-or-minus0.37610.00160.3761\pm 0.0016  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.3795±0.0016plus-or-minus0.37950.00160.3795\pm 0.0016 0.3733±0.0013plus-or-minus0.37330.00130.3733\pm 0.0013  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.3680±0.0006plus-or-minus0.36800.00060.3680\pm 0.0006 0.3653±0.0005plus-or-minus0.36530.00050.3653\pm 0.0005  XGBoostXGBoost\mathrm{XGBoost} 0.3792±0.0002plus-or-minus0.37920.00020.3792\pm 0.0002 0.3787±0.0000plus-or-minus0.37870.00000.3787\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.3688±0.0002plus-or-minus0.36880.00020.3688\pm 0.0002 0.3684±0.0000plus-or-minus0.36840.00000.3684\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.3647±0.0005plus-or-minus0.36470.00050.3647\pm 0.0005 0.3632±0.0003plus-or-minus0.36320.00030.3632\pm 0.0003  TabRTabR\mathrm{TabR} 0.3577±0.0222plus-or-minus0.35770.02220.3577\pm 0.0222 0.3380±0.0027plus-or-minus0.33800.00270.3380\pm 0.0027  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.3725±0.0091plus-or-minus0.37250.00910.3725\pm 0.0091 –  MNCAMNCA\mathrm{MNCA} 0.3728±0.0012plus-or-minus0.37280.00120.3728\pm 0.0012 0.3720±0.0010plus-or-minus0.37200.00100.3720\pm 0.0010  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.3536±0.0052plus-or-minus0.35360.00520.3536\pm 0.0052 0.3407±0.0009plus-or-minus0.34070.00090.3407\pm 0.0009  TabMTabM\mathrm{TabM} 0.3866±0.0006plus-or-minus0.38660.00060.3866\pm 0.0006 0.3855±0.0003plus-or-minus0.38550.00030.3855\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.3862±0.0005plus-or-minus0.38620.00050.3862\pm 0.0005 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.3877±0.0009plus-or-minus0.38770.00090.3877\pm 0.0009 0.3857±0.0004plus-or-minus0.38570.00040.3857\pm 0.0004  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.3527±0.0112plus-or-minus0.35270.01120.3527\pm 0.0112 0.3478±0.0009plus-or-minus0.34780.00090.3478\pm 0.0009 | particulate-matter-ukair-2017 ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.3759±0.0004plus-or-minus0.37590.00040.3759\pm 0.0004 0.3729±0.0003plus-or-minus0.37290.00030.3729\pm 0.0003  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 0.3743±0.0007plus-or-minus0.37430.00070.3743\pm 0.0007 0.3718±0.0005plus-or-minus0.37180.00050.3718\pm 0.0005  DCN2DCN2\mathrm{DCN2} 0.3759±0.0012plus-or-minus0.37590.00120.3759\pm 0.0012 0.3738±0.0004plus-or-minus0.37380.00040.3738\pm 0.0004  SNNSNN\mathrm{SNN} 0.3790±0.0007plus-or-minus0.37900.00070.3790\pm 0.0007 0.3744±0.0002plus-or-minus0.37440.00020.3744\pm 0.0002  TromptTrompt\mathrm{Trompt} 0.3700±n​a​nplus-or-minus0.3700𝑛𝑎𝑛0.3700\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.3723±0.0011plus-or-minus0.37230.00110.3723\pm 0.0011 0.3692±0.0010plus-or-minus0.36920.00100.3692\pm 0.0010  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.3741±0.0010plus-or-minus0.37410.00100.3741\pm 0.0010 0.3698±0.0004plus-or-minus0.36980.00040.3698\pm 0.0004  ExcelExcel\mathrm{Excel} 0.3699±0.0014plus-or-minus0.36990.00140.3699\pm 0.0014 0.3652±n​a​nplus-or-minus0.3652𝑛𝑎𝑛0.3652\pm nan  SAINTSAINT\mathrm{SAINT} 0.3704±0.0014plus-or-minus0.37040.00140.3704\pm 0.0014 0.3672±0.0009plus-or-minus0.36720.00090.3672\pm 0.0009  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.3735±0.0012plus-or-minus0.37350.00120.3735\pm 0.0012 0.3686±0.0004plus-or-minus0.36860.00040.3686\pm 0.0004  T2GT2G\mathrm{T2G} 0.3676±0.0024plus-or-minus0.36760.00240.3676\pm 0.0024 0.3631±n​a​nplus-or-minus0.3631𝑛𝑎𝑛0.3631\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.3665±0.0008plus-or-minus0.36650.00080.3665\pm 0.0008 0.3642±0.0003plus-or-minus0.36420.00030.3642\pm 0.0003  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.3657±0.0007plus-or-minus0.36570.00070.3657\pm 0.0007 0.3629±0.0002plus-or-minus0.36290.00020.3629\pm 0.0002  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.3649±0.0011plus-or-minus0.36490.00110.3649\pm 0.0011 0.3637±0.0008plus-or-minus0.36370.00080.3637\pm 0.0008  XGBoostXGBoost\mathrm{XGBoost} 0.3641±0.0001plus-or-minus0.36410.00010.3641\pm 0.0001 0.3640±0.0000plus-or-minus0.36400.00000.3640\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.3637±0.0001plus-or-minus0.36370.00010.3637\pm 0.0001 0.3635±0.0000plus-or-minus0.36350.00000.3635\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.3647±0.0004plus-or-minus0.36470.00040.3647\pm 0.0004 0.3637±0.0002plus-or-minus0.36370.00020.3637\pm 0.0002  TabRTabR\mathrm{TabR} 0.3613±0.0005plus-or-minus0.36130.00050.3613\pm 0.0005 0.3590±0.0002plus-or-minus0.35900.00020.3590\pm 0.0002  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.3596±0.0004plus-or-minus0.35960.00040.3596\pm 0.0004 –  MNCAMNCA\mathrm{MNCA} 0.3670±0.0004plus-or-minus0.36700.00040.3670\pm 0.0004 0.3649±0.0002plus-or-minus0.36490.00020.3649\pm 0.0002  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.3646±0.0001plus-or-minus0.36460.00010.3646\pm 0.0001 0.3643±0.0000plus-or-minus0.36430.00000.3643\pm 0.0000  TabMTabM\mathrm{TabM} 0.3686±0.0006plus-or-minus0.36860.00060.3686\pm 0.0006 0.3679±0.0003plus-or-minus0.36790.00030.3679\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.3683±0.0007plus-or-minus0.36830.00070.3683\pm 0.0007 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.3690±0.0009plus-or-minus0.36900.00090.3690\pm 0.0009 0.3675±0.0004plus-or-minus0.36750.00040.3675\pm 0.0004  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.3603±0.0005plus-or-minus0.36030.00050.3603\pm 0.0005 0.3589±0.0003plus-or-minus0.35890.00030.3589\pm 0.0003 |
| road-safety ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.7857±0.0019plus-or-minus0.78570.00190.7857\pm 0.0019 0.7873±0.0004plus-or-minus0.78730.00040.7873\pm 0.0004  TabPFNTabPFN\mathrm{TabPFN} – 0.7338±0.0032plus-or-minus0.73380.00320.7338\pm 0.0032  ResNetResNet\mathrm{ResNet} 0.7875±0.0007plus-or-minus0.78750.00070.7875\pm 0.0007 0.7898±0.0008plus-or-minus0.78980.00080.7898\pm 0.0008  DCN2DCN2\mathrm{DCN2} 0.7781±0.0014plus-or-minus0.77810.00140.7781\pm 0.0014 0.7823±0.0012plus-or-minus0.78230.00120.7823\pm 0.0012  SNNSNN\mathrm{SNN} 0.7847±0.0010plus-or-minus0.78470.00100.7847\pm 0.0010 0.7865±0.0002plus-or-minus0.78650.00020.7865\pm 0.0002  TromptTrompt\mathrm{Trompt} 0.7804±n​a​nplus-or-minus0.7804𝑛𝑎𝑛0.7804\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 0.7826±0.0030plus-or-minus0.78260.00300.7826\pm 0.0030 0.7883±0.0013plus-or-minus0.78830.00130.7883\pm 0.0013  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 0.7878±0.0032plus-or-minus0.78780.00320.7878\pm 0.0032 0.7919±0.0015plus-or-minus0.79190.00150.7919\pm 0.0015  ExcelExcel\mathrm{Excel} 0.7864±0.0053plus-or-minus0.78640.00530.7864\pm 0.0053 0.7907±n​a​nplus-or-minus0.7907𝑛𝑎𝑛0.7907\pm nan  SAINTSAINT\mathrm{SAINT} 0.7584±0.0584plus-or-minus0.75840.05840.7584\pm 0.0584 0.7846±0.0021plus-or-minus0.78460.00210.7846\pm 0.0021  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.7907±0.0012plus-or-minus0.79070.00120.7907\pm 0.0012 0.7943±0.0007plus-or-minus0.79430.00070.7943\pm 0.0007  T2GT2G\mathrm{T2G} 0.7912±0.0026plus-or-minus0.79120.00260.7912\pm 0.0026 0.7961±n​a​nplus-or-minus0.7961𝑛𝑎𝑛0.7961\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.7867±0.0018plus-or-minus0.78670.00180.7867\pm 0.0018 0.7903±0.0002plus-or-minus0.79030.00020.7903\pm 0.0002  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.7853±0.0014plus-or-minus0.78530.00140.7853\pm 0.0014 0.7881±0.0007plus-or-minus0.78810.00070.7881\pm 0.0007  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.7899±0.0009plus-or-minus0.78990.00090.7899\pm 0.0009 0.7935±0.0003plus-or-minus0.79350.00030.7935\pm 0.0003  XGBoostXGBoost\mathrm{XGBoost} 0.8101±0.0017plus-or-minus0.81010.00170.8101\pm 0.0017 0.8129±0.0004plus-or-minus0.81290.00040.8129\pm 0.0004  LightGBMLightGBM\mathrm{LightGBM} 0.7982±0.0012plus-or-minus0.79820.00120.7982\pm 0.0012 0.7996±0.0005plus-or-minus0.79960.00050.7996\pm 0.0005  CatBoostCatBoost\mathrm{CatBoost} 0.8012±0.0009plus-or-minus0.80120.00090.8012\pm 0.0009 0.8022±0.0002plus-or-minus0.80220.00020.8022\pm 0.0002  TabRTabR\mathrm{TabR} 0.8403±0.0014plus-or-minus0.84030.00140.8403\pm 0.0014 0.8441±0.0005plus-or-minus0.84410.00050.8441\pm 0.0005  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8374±0.0013plus-or-minus0.83740.00130.8374\pm 0.0013 –  MNCAMNCA\mathrm{MNCA} 0.8080±0.0013plus-or-minus0.80800.00130.8080\pm 0.0013 0.8121±0.0006plus-or-minus0.81210.00060.8121\pm 0.0006  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8232±0.0017plus-or-minus0.82320.00170.8232\pm 0.0017 0.8287±0.0008plus-or-minus0.82870.00080.8287\pm 0.0008  TabMTabM\mathrm{TabM} 0.7946±0.0013plus-or-minus0.79460.00130.7946\pm 0.0013 0.7961±0.0005plus-or-minus0.79610.00050.7961\pm 0.0005  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.7945±0.0009plus-or-minus0.79450.00090.7945\pm 0.0009 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.7931±0.0011plus-or-minus0.79310.00110.7931\pm 0.0011 0.7946±0.0010plus-or-minus0.79460.00100.7946\pm 0.0010  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8015±0.0034plus-or-minus0.80150.00340.8015\pm 0.0034 0.8060±0.0015plus-or-minus0.80600.00150.8060\pm 0.0015 | year ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 8.9628±0.0232plus-or-minus8.96280.02328.9628\pm 0.0232 8.8931±0.0066plus-or-minus8.89310.00668.8931\pm 0.0066  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} 8.9658±0.0239plus-or-minus8.96580.02398.9658\pm 0.0239 8.8755±0.0066plus-or-minus8.87550.00668.8755\pm 0.0066  DCN2DCN2\mathrm{DCN2} 9.2761±0.0401plus-or-minus9.27610.04019.2761\pm 0.0401 9.0640±0.0156plus-or-minus9.06400.01569.0640\pm 0.0156  SNNSNN\mathrm{SNN} 9.0054±0.0256plus-or-minus9.00540.02569.0054\pm 0.0256 8.9351±0.0073plus-or-minus8.93510.00738.9351\pm 0.0073  TromptTrompt\mathrm{Trompt} 8.9707±n​a​nplus-or-minus8.9707𝑛𝑎𝑛8.9707\pm nan –  AutoIntAutoInt\mathrm{AutoInt} 9.0430±0.0280plus-or-minus9.04300.02809.0430\pm 0.0280 8.9619±0.0092plus-or-minus8.96190.00928.9619\pm 0.0092  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} 8.9589±0.0182plus-or-minus8.95890.01828.9589\pm 0.0182 8.9086±0.0177plus-or-minus8.90860.01778.9086\pm 0.0177  ExcelExcel\mathrm{Excel} 9.0395±0.0266plus-or-minus9.03950.02669.0395\pm 0.0266 8.9551±n​a​nplus-or-minus8.9551𝑛𝑎𝑛8.9551\pm nan  SAINTSAINT\mathrm{SAINT} 9.0248±0.0225plus-or-minus9.02480.02259.0248\pm 0.0225 8.9548±0.0102plus-or-minus8.95480.01028.9548\pm 0.0102  FT​-​TFT-T\mathrm{FT\texttt{-}T} 9.0005±0.0215plus-or-minus9.00050.02159.0005\pm 0.0215 8.9360±0.0013plus-or-minus8.93600.00138.9360\pm 0.0013  T2GT2G\mathrm{T2G} 8.9775±0.0138plus-or-minus8.97750.01388.9775\pm 0.0138 8.8979±n​a​nplus-or-minus8.8979𝑛𝑎𝑛8.8979\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 8.9355±0.0103plus-or-minus8.93550.01038.9355\pm 0.0103 8.9063±0.0030plus-or-minus8.90630.00308.9063\pm 0.0030  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 8.9455±0.0173plus-or-minus8.94550.01738.9455\pm 0.0173 8.9083±0.0046plus-or-minus8.90830.00468.9083\pm 0.0046  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 8.9379±0.0206plus-or-minus8.93790.02068.9379\pm 0.0206 8.8753±0.0038plus-or-minus8.87530.00388.8753\pm 0.0038  XGBoostXGBoost\mathrm{XGBoost} 9.0307±0.0028plus-or-minus9.03070.00289.0307\pm 0.0028 9.0245±0.0015plus-or-minus9.02450.00159.0245\pm 0.0015  LightGBMLightGBM\mathrm{LightGBM} 9.0200±0.0025plus-or-minus9.02000.00259.0200\pm 0.0025 9.0128±0.0015plus-or-minus9.01280.00159.0128\pm 0.0015  CatBoostCatBoost\mathrm{CatBoost} 9.0370±0.0073plus-or-minus9.03700.00739.0370\pm 0.0073 9.0054±0.0028plus-or-minus9.00540.00289.0054\pm 0.0028  TabRTabR\mathrm{TabR} 9.0069±0.0152plus-or-minus9.00690.01529.0069\pm 0.0152 8.9132±0.0088plus-or-minus8.91320.00888.9132\pm 0.0088  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 8.9721±0.0105plus-or-minus8.97210.01058.9721\pm 0.0105 –  MNCAMNCA\mathrm{MNCA} 8.9476±0.0152plus-or-minus8.94760.01528.9476\pm 0.0152 8.8977±0.0037plus-or-minus8.89770.00378.8977\pm 0.0037  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 8.8973±0.0082plus-or-minus8.89730.00828.8973\pm 0.0082 8.8550±0.0031plus-or-minus8.85500.00318.8550\pm 0.0031  TabMTabM\mathrm{TabM} 8.8701±0.0110plus-or-minus8.87010.01108.8701\pm 0.0110 8.8517±0.0022plus-or-minus8.85170.00228.8517\pm 0.0022  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 8.8715±0.0116plus-or-minus8.87150.01168.8715\pm 0.0116 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 8.8958±0.0087plus-or-minus8.89580.00878.8958\pm 0.0087 8.8810±0.0020plus-or-minus8.88100.00208.8810\pm 0.0020  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 8.8825±0.0087plus-or-minus8.88250.00878.8825\pm 0.0087 8.8560±0.0015plus-or-minus8.85600.00158.8560\pm 0.0015 |




Table 19: Extended results for the main benchmark. Results are grouped by datasets.

|  |  |
| --- | --- |
| sberbank-housing ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.2529±0.0078plus-or-minus0.25290.00780.2529\pm 0.0078 0.2474±0.0052plus-or-minus0.24740.00520.2474\pm 0.0052  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.2616±0.0049plus-or-minus0.26160.00490.2616\pm 0.0049 0.2506±0.0015plus-or-minus0.25060.00150.2506\pm 0.0015  SNNSNN\mathrm{SNN} 0.2671±0.0140plus-or-minus0.26710.01400.2671\pm 0.0140 0.2555±0.0033plus-or-minus0.25550.00330.2555\pm 0.0033  TromptTrompt\mathrm{Trompt} 0.2509±n​a​nplus-or-minus0.2509𝑛𝑎𝑛0.2509\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.2533±0.0046plus-or-minus0.25330.00460.2533\pm 0.0046 0.2485±n​a​nplus-or-minus0.2485𝑛𝑎𝑛0.2485\pm nan  SAINTSAINT\mathrm{SAINT} 0.2467±0.0019plus-or-minus0.24670.00190.2467\pm 0.0019 0.2442±n​a​nplus-or-minus0.2442𝑛𝑎𝑛0.2442\pm nan  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.2440±0.0038plus-or-minus0.24400.00380.2440\pm 0.0038 0.2367±0.0010plus-or-minus0.23670.00100.2367\pm 0.0010  T2GT2G\mathrm{T2G} 0.2416±0.0025plus-or-minus0.24160.00250.2416\pm 0.0025 0.2343±n​a​nplus-or-minus0.2343𝑛𝑎𝑛0.2343\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.2528±0.0055plus-or-minus0.25280.00550.2528\pm 0.0055 0.2503±0.0029plus-or-minus0.25030.00290.2503\pm 0.0029  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.2412±0.0031plus-or-minus0.24120.00310.2412\pm 0.0031 0.2355±0.0006plus-or-minus0.23550.00060.2355\pm 0.0006  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.2383±0.0032plus-or-minus0.23830.00320.2383\pm 0.0032 0.2327±0.0009plus-or-minus0.23270.00090.2327\pm 0.0009  XGBoostXGBoost\mathrm{XGBoost} 0.2419±0.0012plus-or-minus0.24190.00120.2419\pm 0.0012 0.2416±0.0007plus-or-minus0.24160.00070.2416\pm 0.0007  LightGBMLightGBM\mathrm{LightGBM} 0.2468±0.0009plus-or-minus0.24680.00090.2468\pm 0.0009 0.2467±0.0002plus-or-minus0.24670.00020.2467\pm 0.0002  CatBoostCatBoost\mathrm{CatBoost} 0.2482±0.0034plus-or-minus0.24820.00340.2482\pm 0.0034 0.2473±0.0016plus-or-minus0.24730.00160.2473\pm 0.0016  TabRTabR\mathrm{TabR} 0.2820±0.0323plus-or-minus0.28200.03230.2820\pm 0.0323 0.2603±0.0048plus-or-minus0.26030.00480.2603\pm 0.0048  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.2542±0.0101plus-or-minus0.25420.01010.2542\pm 0.0101 –  MNCAMNCA\mathrm{MNCA} 0.2593±0.0053plus-or-minus0.25930.00530.2593\pm 0.0053 0.2520±0.0032plus-or-minus0.25200.00320.2520\pm 0.0032  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.2448±0.0039plus-or-minus0.24480.00390.2448\pm 0.0039 0.2404±0.0025plus-or-minus0.24040.00250.2404\pm 0.0025  TabMTabM\mathrm{TabM} 0.2469±0.0035plus-or-minus0.24690.00350.2469\pm 0.0035 0.2440±0.0026plus-or-minus0.24400.00260.2440\pm 0.0026  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.2480±0.0049plus-or-minus0.24800.00490.2480\pm 0.0049 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.2440±0.0025plus-or-minus0.24400.00250.2440\pm 0.0025 0.2425±0.0008plus-or-minus0.24250.00080.2425\pm 0.0008  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.2357±0.0025plus-or-minus0.23570.00250.2357\pm 0.0025 0.2333±0.0007plus-or-minus0.23330.00070.2333\pm 0.0007 | ecom-offers ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.5989±0.0017plus-or-minus0.59890.00170.5989\pm 0.0017 0.5995±0.0011plus-or-minus0.59950.00110.5995\pm 0.0011  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.5996±0.0043plus-or-minus0.59960.00430.5996\pm 0.0043 0.6039±0.0028plus-or-minus0.60390.00280.6039\pm 0.0028  SNNSNN\mathrm{SNN} 0.5912±0.0056plus-or-minus0.59120.00560.5912\pm 0.0056 0.5961±0.0033plus-or-minus0.59610.00330.5961\pm 0.0033  TromptTrompt\mathrm{Trompt} 0.5803±n​a​nplus-or-minus0.5803𝑛𝑎𝑛0.5803\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.5759±0.0066plus-or-minus0.57590.00660.5759\pm 0.0066 0.5759±n​a​nplus-or-minus0.5759𝑛𝑎𝑛0.5759\pm nan  SAINTSAINT\mathrm{SAINT} 0.5812±0.0098plus-or-minus0.58120.00980.5812\pm 0.0098 0.5834±n​a​nplus-or-minus0.5834𝑛𝑎𝑛0.5834\pm nan  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.5775±0.0063plus-or-minus0.57750.00630.5775\pm 0.0063 0.5817±0.0021plus-or-minus0.58170.00210.5817\pm 0.0021  T2GT2G\mathrm{T2G} 0.5791±0.0056plus-or-minus0.57910.00560.5791\pm 0.0056 0.5824±n​a​nplus-or-minus0.5824𝑛𝑎𝑛0.5824\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.5800±0.0029plus-or-minus0.58000.00290.5800\pm 0.0029 0.5819±0.0011plus-or-minus0.58190.00110.5819\pm 0.0011  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.5846±0.0048plus-or-minus0.58460.00480.5846\pm 0.0048 0.5872±0.0018plus-or-minus0.58720.00180.5872\pm 0.0018  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.5949±0.0013plus-or-minus0.59490.00130.5949\pm 0.0013 0.5953±0.0006plus-or-minus0.59530.00060.5953\pm 0.0006  XGBoostXGBoost\mathrm{XGBoost} 0.5763±0.0072plus-or-minus0.57630.00720.5763\pm 0.0072 0.5917±0.0035plus-or-minus0.59170.00350.5917\pm 0.0035  LightGBMLightGBM\mathrm{LightGBM} 0.5758±0.0006plus-or-minus0.57580.00060.5758\pm 0.0006 0.5758±0.0003plus-or-minus0.57580.00030.5758\pm 0.0003  CatBoostCatBoost\mathrm{CatBoost} 0.5596±0.0068plus-or-minus0.55960.00680.5596\pm 0.0068 0.5067±0.0011plus-or-minus0.50670.00110.5067\pm 0.0011  TabRTabR\mathrm{TabR} 0.5943±0.0019plus-or-minus0.59430.00190.5943\pm 0.0019 0.5977±0.0009plus-or-minus0.59770.00090.5977\pm 0.0009  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.5762±0.0052plus-or-minus0.57620.00520.5762\pm 0.0052 –  MNCAMNCA\mathrm{MNCA} 0.5765±0.0087plus-or-minus0.57650.00870.5765\pm 0.0087 0.5820±0.0047plus-or-minus0.58200.00470.5820\pm 0.0047  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.5758±0.0050plus-or-minus0.57580.00500.5758\pm 0.0050 0.5796±0.0009plus-or-minus0.57960.00090.5796\pm 0.0009  TabMTabM\mathrm{TabM} 0.5948±0.0006plus-or-minus0.59480.00060.5948\pm 0.0006 0.5952±0.0004plus-or-minus0.59520.00040.5952\pm 0.0004  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.5959±0.0010plus-or-minus0.59590.00100.5959\pm 0.0010 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.5948±0.0009plus-or-minus0.59480.00090.5948\pm 0.0009 0.5954±0.0004plus-or-minus0.59540.00040.5954\pm 0.0004  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.5919±0.0016plus-or-minus0.59190.00160.5919\pm 0.0016 0.5926±0.0006plus-or-minus0.59260.00060.5926\pm 0.0006 |
| maps-routing ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.1625±0.0001plus-or-minus0.16250.00010.1625\pm 0.0001 0.1621±0.0000plus-or-minus0.16210.00000.1621\pm 0.0000  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.1656±0.0004plus-or-minus0.16560.00040.1656\pm 0.0004 0.1636±0.0001plus-or-minus0.16360.00010.1636\pm 0.0001  SNNSNN\mathrm{SNN} 0.1634±0.0002plus-or-minus0.16340.00020.1634\pm 0.0002 0.1625±0.0000plus-or-minus0.16250.00000.1625\pm 0.0000  TromptTrompt\mathrm{Trompt} 0.1624±n​a​nplus-or-minus0.1624𝑛𝑎𝑛0.1624\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.1628±0.0001plus-or-minus0.16280.00010.1628\pm 0.0001 0.1621±n​a​nplus-or-minus0.1621𝑛𝑎𝑛0.1621\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.1625±0.0003plus-or-minus0.16250.00030.1625\pm 0.0003 0.1619±0.0001plus-or-minus0.16190.00010.1619\pm 0.0001  T2GT2G\mathrm{T2G} 0.1616±0.0001plus-or-minus0.16160.00010.1616\pm 0.0001 0.1608±n​a​nplus-or-minus0.1608𝑛𝑎𝑛0.1608\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.1618±0.0002plus-or-minus0.16180.00020.1618\pm 0.0002 0.1613±0.0000plus-or-minus0.16130.00000.1613\pm 0.0000  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.1618±0.0002plus-or-minus0.16180.00020.1618\pm 0.0002 0.1613±0.0001plus-or-minus0.16130.00010.1613\pm 0.0001  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.1620±0.0002plus-or-minus0.16200.00020.1620\pm 0.0002 0.1614±0.0000plus-or-minus0.16140.00000.1614\pm 0.0000  XGBoostXGBoost\mathrm{XGBoost} 0.1616±0.0001plus-or-minus0.16160.00010.1616\pm 0.0001 0.1614±0.0000plus-or-minus0.16140.00000.1614\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.1618±0.0000plus-or-minus0.16180.00000.1618\pm 0.0000 0.1616±0.0000plus-or-minus0.16160.00000.1616\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.1619±0.0001plus-or-minus0.16190.00010.1619\pm 0.0001 0.1615±0.0000plus-or-minus0.16150.00000.1615\pm 0.0000  TabRTabR\mathrm{TabR} 0.1639±0.0003plus-or-minus0.16390.00030.1639\pm 0.0003 0.1622±0.0002plus-or-minus0.16220.00020.1622\pm 0.0002  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.1622±0.0002plus-or-minus0.16220.00020.1622\pm 0.0002 –  MNCAMNCA\mathrm{MNCA} 0.1625±0.0001plus-or-minus0.16250.00010.1625\pm 0.0001 0.1621±0.0001plus-or-minus0.16210.00010.1621\pm 0.0001  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.1627±0.0002plus-or-minus0.16270.00020.1627\pm 0.0002 0.1623±0.0001plus-or-minus0.16230.00010.1623\pm 0.0001  TabMTabM\mathrm{TabM} 0.1612±0.0001plus-or-minus0.16120.00010.1612\pm 0.0001 0.1609±0.0000plus-or-minus0.16090.00000.1609\pm 0.0000  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.1612±0.0001plus-or-minus0.16120.00010.1612\pm 0.0001 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.1613±0.0002plus-or-minus0.16130.00020.1613\pm 0.0002 0.1609±0.0000plus-or-minus0.16090.00000.1609\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.1610±0.0001plus-or-minus0.16100.00010.1610\pm 0.0001 0.1607±0.0001plus-or-minus0.16070.00010.1607\pm 0.0001 | homesite-insurance ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.9506±0.0005plus-or-minus0.95060.00050.9506\pm 0.0005 0.9514±0.0001plus-or-minus0.95140.00010.9514\pm 0.0001  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.9398±0.0053plus-or-minus0.93980.00530.9398\pm 0.0053 0.9432±0.0018plus-or-minus0.94320.00180.9432\pm 0.0018  SNNSNN\mathrm{SNN} 0.9473±0.0013plus-or-minus0.94730.00130.9473\pm 0.0013 0.9484±0.0007plus-or-minus0.94840.00070.9484\pm 0.0007  TromptTrompt\mathrm{Trompt} 0.9588±n​a​nplus-or-minus0.9588𝑛𝑎𝑛0.9588\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.9622±0.0004plus-or-minus0.96220.00040.9622\pm 0.0004 0.9635±n​a​nplus-or-minus0.9635𝑛𝑎𝑛0.9635\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.9622±0.0006plus-or-minus0.96220.00060.9622\pm 0.0006 0.9633±0.0001plus-or-minus0.96330.00010.9633\pm 0.0001  T2GT2G\mathrm{T2G} 0.9624±0.0006plus-or-minus0.96240.00060.9624\pm 0.0006 0.9637±n​a​nplus-or-minus0.9637𝑛𝑎𝑛0.9637\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.9609±0.0009plus-or-minus0.96090.00090.9609\pm 0.0009 0.9626±0.0003plus-or-minus0.96260.00030.9626\pm 0.0003  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.9617±0.0004plus-or-minus0.96170.00040.9617\pm 0.0004 0.9630±0.0002plus-or-minus0.96300.00020.9630\pm 0.0002  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.9582±0.0014plus-or-minus0.95820.00140.9582\pm 0.0014 0.9599±0.0002plus-or-minus0.95990.00020.9599\pm 0.0002  XGBoostXGBoost\mathrm{XGBoost} 0.9601±0.0002plus-or-minus0.96010.00020.9601\pm 0.0002 0.9602±0.0000plus-or-minus0.96020.00000.9602\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.9603±0.0002plus-or-minus0.96030.00020.9603\pm 0.0002 0.9604±0.0001plus-or-minus0.96040.00010.9604\pm 0.0001  CatBoostCatBoost\mathrm{CatBoost} 0.9606±0.0003plus-or-minus0.96060.00030.9606\pm 0.0003 0.9609±0.0001plus-or-minus0.96090.00010.9609\pm 0.0001  TabRTabR\mathrm{TabR} 0.9487±0.0014plus-or-minus0.94870.00140.9487\pm 0.0014 0.9505±0.0001plus-or-minus0.95050.00010.9505\pm 0.0001  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.9556±0.0021plus-or-minus0.95560.00210.9556\pm 0.0021 –  MNCAMNCA\mathrm{MNCA} 0.9514±0.0038plus-or-minus0.95140.00380.9514\pm 0.0038 0.9522±0.0027plus-or-minus0.95220.00270.9522\pm 0.0027  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.9620±0.0006plus-or-minus0.96200.00060.9620\pm 0.0006 0.9635±0.0002plus-or-minus0.96350.00020.9635\pm 0.0002  TabMTabM\mathrm{TabM} 0.9641±0.0004plus-or-minus0.96410.00040.9641\pm 0.0004 0.9644±0.0003plus-or-minus0.96440.00030.9644\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.9640±0.0004plus-or-minus0.96400.00040.9640\pm 0.0004 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.9642±0.0003plus-or-minus0.96420.00030.9642\pm 0.0003 0.9644±0.0001plus-or-minus0.96440.00010.9644\pm 0.0001  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.9627±0.0002plus-or-minus0.96270.00020.9627\pm 0.0002 0.9630±0.0001plus-or-minus0.96300.00010.9630\pm 0.0001 |
| cooking-time ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.4828±0.0002plus-or-minus0.48280.00020.4828\pm 0.0002 0.4822±0.0000plus-or-minus0.48220.00000.4822\pm 0.0000  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.4834±0.0003plus-or-minus0.48340.00030.4834\pm 0.0003 0.4822±0.0001plus-or-minus0.48220.00010.4822\pm 0.0001  SNNSNN\mathrm{SNN} 0.4835±0.0006plus-or-minus0.48350.00060.4835\pm 0.0006 0.4818±0.0002plus-or-minus0.48180.00020.4818\pm 0.0002  TromptTrompt\mathrm{Trompt} 0.4809±n​a​nplus-or-minus0.4809𝑛𝑎𝑛0.4809\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.4821±0.0005plus-or-minus0.48210.00050.4821\pm 0.0005 0.4808±n​a​nplus-or-minus0.4808𝑛𝑎𝑛0.4808\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.4820±0.0008plus-or-minus0.48200.00080.4820\pm 0.0008 0.4813±0.0005plus-or-minus0.48130.00050.4813\pm 0.0005  T2GT2G\mathrm{T2G} 0.4809±0.0008plus-or-minus0.48090.00080.4809\pm 0.0008 0.4797±n​a​nplus-or-minus0.4797𝑛𝑎𝑛0.4797\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.4811±0.0004plus-or-minus0.48110.00040.4811\pm 0.0004 0.4805±0.0001plus-or-minus0.48050.00010.4805\pm 0.0001  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.4809±0.0006plus-or-minus0.48090.00060.4809\pm 0.0006 0.4804±0.0003plus-or-minus0.48040.00030.4804\pm 0.0003  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.4812±0.0004plus-or-minus0.48120.00040.4812\pm 0.0004 0.4807±0.0002plus-or-minus0.48070.00020.4807\pm 0.0002  XGBoostXGBoost\mathrm{XGBoost} 0.4823±0.0001plus-or-minus0.48230.00010.4823\pm 0.0001 0.4821±0.0000plus-or-minus0.48210.00000.4821\pm 0.0000  LightGBMLightGBM\mathrm{LightGBM} 0.4826±0.0001plus-or-minus0.48260.00010.4826\pm 0.0001 0.4825±0.0001plus-or-minus0.48250.00010.4825\pm 0.0001  CatBoostCatBoost\mathrm{CatBoost} 0.4823±0.0001plus-or-minus0.48230.00010.4823\pm 0.0001 0.4820±0.0001plus-or-minus0.48200.00010.4820\pm 0.0001  TabRTabR\mathrm{TabR} 0.4828±0.0008plus-or-minus0.48280.00080.4828\pm 0.0008 0.4814±0.0004plus-or-minus0.48140.00040.4814\pm 0.0004  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.4818±0.0006plus-or-minus0.48180.00060.4818\pm 0.0006 –  MNCAMNCA\mathrm{MNCA} 0.4825±0.0004plus-or-minus0.48250.00040.4825\pm 0.0004 0.4819±0.0003plus-or-minus0.48190.00030.4819\pm 0.0003  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.4818±0.0005plus-or-minus0.48180.00050.4818\pm 0.0005 0.4809±0.0003plus-or-minus0.48090.00030.4809\pm 0.0003  TabMTabM\mathrm{TabM} 0.4803±0.0006plus-or-minus0.48030.00060.4803\pm 0.0006 0.4797±0.0003plus-or-minus0.47970.00030.4797\pm 0.0003  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.4803±0.0005plus-or-minus0.48030.00050.4803\pm 0.0005 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.4804±0.0006plus-or-minus0.48040.00060.4804\pm 0.0006 0.4796±0.0000plus-or-minus0.47960.00000.4796\pm 0.0000  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.4805±0.0007plus-or-minus0.48050.00070.4805\pm 0.0007 0.4795±0.0003plus-or-minus0.47950.00030.4795\pm 0.0003 | homecredit-default ↑  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.8538±0.0014plus-or-minus0.85380.00140.8538\pm 0.0014 0.8566±0.0005plus-or-minus0.85660.00050.8566\pm 0.0005  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.8471±0.0019plus-or-minus0.84710.00190.8471\pm 0.0019 0.8549±0.0002plus-or-minus0.85490.00020.8549\pm 0.0002  SNNSNN\mathrm{SNN} 0.8541±0.0016plus-or-minus0.85410.00160.8541\pm 0.0016 0.8569±0.0010plus-or-minus0.85690.00100.8569\pm 0.0010  TromptTrompt\mathrm{Trompt} 0.8355±n​a​nplus-or-minus0.8355𝑛𝑎𝑛0.8355\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.8513±0.0024plus-or-minus0.85130.00240.8513\pm 0.0024 0.8564±n​a​nplus-or-minus0.8564𝑛𝑎𝑛0.8564\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.8571±0.0023plus-or-minus0.85710.00230.8571\pm 0.0023 0.8611±0.0013plus-or-minus0.86110.00130.8611\pm 0.0013  T2GT2G\mathrm{T2G} 0.8597±0.0007plus-or-minus0.85970.00070.8597\pm 0.0007 0.8629±n​a​nplus-or-minus0.8629𝑛𝑎𝑛0.8629\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.8598±0.0009plus-or-minus0.85980.00090.8598\pm 0.0009 0.8607±0.0003plus-or-minus0.86070.00030.8607\pm 0.0003  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.8572±0.0011plus-or-minus0.85720.00110.8572\pm 0.0011 0.8590±0.0003plus-or-minus0.85900.00030.8590\pm 0.0003  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.8568±0.0039plus-or-minus0.85680.00390.8568\pm 0.0039 0.8614±0.0014plus-or-minus0.86140.00140.8614\pm 0.0014  XGBoostXGBoost\mathrm{XGBoost} 0.8670±0.0005plus-or-minus0.86700.00050.8670\pm 0.0005 0.8674±0.0001plus-or-minus0.86740.00010.8674\pm 0.0001  LightGBMLightGBM\mathrm{LightGBM} 0.8664±0.0004plus-or-minus0.86640.00040.8664\pm 0.0004 0.8667±0.0000plus-or-minus0.86670.00000.8667\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.8627±n​a​nplus-or-minus0.8627𝑛𝑎𝑛0.8627\pm nan –  TabRTabR\mathrm{TabR} 0.8501±0.0027plus-or-minus0.85010.00270.8501\pm 0.0027 0.8548±0.0003plus-or-minus0.85480.00030.8548\pm 0.0003  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.8547±0.0021plus-or-minus0.85470.00210.8547\pm 0.0021 –  MNCAMNCA\mathrm{MNCA} 0.8531±0.0018plus-or-minus0.85310.00180.8531\pm 0.0018 0.8569±0.0004plus-or-minus0.85690.00040.8569\pm 0.0004  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.8544±0.0033plus-or-minus0.85440.00330.8544\pm 0.0033 0.8606±0.0024plus-or-minus0.86060.00240.8606\pm 0.0024  TabMTabM\mathrm{TabM} 0.8583±0.0010plus-or-minus0.85830.00100.8583\pm 0.0010 0.8599±0.0006plus-or-minus0.85990.00060.8599\pm 0.0006  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.8583±0.0011plus-or-minus0.85830.00110.8583\pm 0.0011 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.8577±0.0017plus-or-minus0.85770.00170.8577\pm 0.0017 0.8598±0.0004plus-or-minus0.85980.00040.8598\pm 0.0004  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.8632±0.0017plus-or-minus0.86320.00170.8632\pm 0.0017 0.8656±0.0003plus-or-minus0.86560.00030.8656\pm 0.0003 |
| delivery-eta ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 0.5493±0.0007plus-or-minus0.54930.00070.5493\pm 0.0007 0.5478±0.0006plus-or-minus0.54780.00060.5478\pm 0.0006  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 0.5516±0.0014plus-or-minus0.55160.00140.5516\pm 0.0014 0.5495±0.0004plus-or-minus0.54950.00040.5495\pm 0.0004  SNNSNN\mathrm{SNN} 0.5495±0.0008plus-or-minus0.54950.00080.5495\pm 0.0008 0.5479±0.0001plus-or-minus0.54790.00010.5479\pm 0.0001  TromptTrompt\mathrm{Trompt} 0.5519±n​a​nplus-or-minus0.5519𝑛𝑎𝑛0.5519\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 0.5552±0.0030plus-or-minus0.55520.00300.5552\pm 0.0030 0.5524±n​a​nplus-or-minus0.5524𝑛𝑎𝑛0.5524\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 0.5542±0.0026plus-or-minus0.55420.00260.5542\pm 0.0026 0.5523±0.0018plus-or-minus0.55230.00180.5523\pm 0.0018  T2GT2G\mathrm{T2G} 0.5527±0.0016plus-or-minus0.55270.00160.5527\pm 0.0016 0.5512±n​a​nplus-or-minus0.5512𝑛𝑎𝑛0.5512\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 0.5521±0.0014plus-or-minus0.55210.00140.5521\pm 0.0014 0.5512±0.0005plus-or-minus0.55120.00050.5512\pm 0.0005  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 0.5535±0.0019plus-or-minus0.55350.00190.5535\pm 0.0019 0.5526±0.0009plus-or-minus0.55260.00090.5526\pm 0.0009  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 0.5521±0.0019plus-or-minus0.55210.00190.5521\pm 0.0019 0.5511±0.0007plus-or-minus0.55110.00070.5511\pm 0.0007  XGBoostXGBoost\mathrm{XGBoost} 0.5468±0.0002plus-or-minus0.54680.00020.5468\pm 0.0002 0.5463±0.0001plus-or-minus0.54630.00010.5463\pm 0.0001  LightGBMLightGBM\mathrm{LightGBM} 0.5468±0.0001plus-or-minus0.54680.00010.5468\pm 0.0001 0.5465±0.0000plus-or-minus0.54650.00000.5465\pm 0.0000  CatBoostCatBoost\mathrm{CatBoost} 0.5465±0.0001plus-or-minus0.54650.00010.5465\pm 0.0001 0.5461±0.0000plus-or-minus0.54610.00000.5461\pm 0.0000  TabRTabR\mathrm{TabR} 0.5514±0.0024plus-or-minus0.55140.00240.5514\pm 0.0024 0.5480±0.0005plus-or-minus0.54800.00050.5480\pm 0.0005  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 0.5520±0.0015plus-or-minus0.55200.00150.5520\pm 0.0015 –  MNCAMNCA\mathrm{MNCA} 0.5498±0.0007plus-or-minus0.54980.00070.5498\pm 0.0007 0.5488±0.0002plus-or-minus0.54880.00020.5488\pm 0.0002  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 0.5507±0.0013plus-or-minus0.55070.00130.5507\pm 0.0013 0.5494±0.0006plus-or-minus0.54940.00060.5494\pm 0.0006  TabMTabM\mathrm{TabM} 0.5510±0.0015plus-or-minus0.55100.00150.5510\pm 0.0015 0.5504±0.0004plus-or-minus0.55040.00040.5504\pm 0.0004  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 0.5517±0.0016plus-or-minus0.55170.00160.5517\pm 0.0016 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 0.5519±0.0015plus-or-minus0.55190.00150.5519\pm 0.0015 0.5511±0.0006plus-or-minus0.55110.00060.5511\pm 0.0006  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 0.5508±0.0013plus-or-minus0.55080.00130.5508\pm 0.0013 0.5497±0.0003plus-or-minus0.54970.00030.5497\pm 0.0003 | weather ↓  Method Single model Ensemble  MLPMLP\mathrm{MLP} 1.5378±0.0054plus-or-minus1.53780.00541.5378\pm 0.0054 1.5111±0.0029plus-or-minus1.51110.00291.5111\pm 0.0029  TabPFNTabPFN\mathrm{TabPFN} – –  ResNetResNet\mathrm{ResNet} – –  DCN2DCN2\mathrm{DCN2} 1.5606±0.0057plus-or-minus1.56060.00571.5606\pm 0.0057 1.5292±0.0028plus-or-minus1.52920.00281.5292\pm 0.0028  SNNSNN\mathrm{SNN} 1.5280±0.0085plus-or-minus1.52800.00851.5280\pm 0.0085 1.5013±0.0034plus-or-minus1.50130.00341.5013\pm 0.0034  TromptTrompt\mathrm{Trompt} 1.5187±n​a​nplus-or-minus1.5187𝑛𝑎𝑛1.5187\pm nan –  AutoIntAutoInt\mathrm{AutoInt} – –  MLP​-​MixerMLP-Mixer\mathrm{MLP\texttt{-}Mixer} – –  ExcelExcel\mathrm{Excel} 1.5131±0.0022plus-or-minus1.51310.00221.5131\pm 0.0022 1.4707±n​a​nplus-or-minus1.4707𝑛𝑎𝑛1.4707\pm nan  SAINTSAINT\mathrm{SAINT} – –  FT​-​TFT-T\mathrm{FT\texttt{-}T} 1.5104±0.0097plus-or-minus1.51040.00971.5104\pm 0.0097 1.4719±0.0040plus-or-minus1.47190.00401.4719\pm 0.0040  T2GT2G\mathrm{T2G} 1.4849±0.0087plus-or-minus1.48490.00871.4849\pm 0.0087 1.4513±n​a​nplus-or-minus1.4513𝑛𝑎𝑛1.4513\pm nan  MLPP−LitesuperscriptMLPPLite\mathrm{MLP^{P-Lite}} 1.5170±0.0040plus-or-minus1.51700.00401.5170\pm 0.0040 1.4953±0.0023plus-or-minus1.49530.00231.4953\pm 0.0023  MLPPsuperscriptMLPP\mathrm{MLP^{P}} 1.5139±0.0031plus-or-minus1.51390.00311.5139\pm 0.0031 1.4978±0.0020plus-or-minus1.49780.00201.4978\pm 0.0020  MLP†superscriptMLP†\mathrm{MLP^{\dagger}} 1.5162±0.0020plus-or-minus1.51620.00201.5162\pm 0.0020 1.5066±0.0008plus-or-minus1.50660.00081.5066\pm 0.0008  XGBoostXGBoost\mathrm{XGBoost} 1.4671±0.0006plus-or-minus1.46710.00061.4671\pm 0.0006 1.4629±0.0002plus-or-minus1.46290.00021.4629\pm 0.0002  LightGBMLightGBM\mathrm{LightGBM} 1.4625±0.0008plus-or-minus1.46250.00081.4625\pm 0.0008 1.4581±0.0003plus-or-minus1.45810.00031.4581\pm 0.0003  CatBoostCatBoost\mathrm{CatBoost} 1.4688±0.0019plus-or-minus1.46880.00191.4688\pm 0.0019 –  TabRTabR\mathrm{TabR} 1.4666±0.0039plus-or-minus1.46660.00391.4666\pm 0.0039 1.4547±0.0008plus-or-minus1.45470.00081.4547\pm 0.0008  TabR†superscriptTabR†\mathrm{TabR^{\dagger}} 1.4458±0.0018plus-or-minus1.44580.00181.4458\pm 0.0018 –  MNCAMNCA\mathrm{MNCA} 1.5062±0.0054plus-or-minus1.50620.00541.5062\pm 0.0054 1.4822±0.0013plus-or-minus1.48220.00131.4822\pm 0.0013  MNCA†superscriptMNCA†\mathrm{MNCA^{\dagger}} 1.5008±0.0034plus-or-minus1.50080.00341.5008\pm 0.0034 1.4782±0.0011plus-or-minus1.47820.00111.4782\pm 0.0011  TabMTabM\mathrm{TabM} 1.4786±0.0039plus-or-minus1.47860.00391.4786\pm 0.0039 1.4715±0.0020plus-or-minus1.47150.00201.4715\pm 0.0020  TabM​[GH]TabMdelimited-[]GH\mathrm{TabM[GH]} 1.4796±0.0037plus-or-minus1.47960.00371.4796\pm 0.0037 –  TabMminisubscriptTabMmini\mathrm{TabM\_{mini}} 1.4809±0.0027plus-or-minus1.48090.00271.4809\pm 0.0027 1.4717±0.0012plus-or-minus1.47170.00121.4717\pm 0.0012  TabMmini†superscriptsubscriptTabMmini†\mathrm{TabM\_{mini}^{\dagger}} 1.4709±0.0047plus-or-minus1.47090.00471.4709\pm 0.0047 1.4611±0.0023plus-or-minus1.46110.00231.4611\pm 0.0023 |

[◄](/html/2410.24209)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2410.24210)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2410.24210)
[View original  
on arXiv](https://arxiv.org/abs/2410.24210)[►](/html/2410.24211)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue Nov 5 17:43:57 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
