---
arxiv: '2211.12703'
authors:
- Josh Gardner Zoran Popović Ludwig Schmidt Paul G. Allen School of Computer Science
  & Engineering University of Washington {jpgard, zoran, schmidt}@cs.washington.edu
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation'
url: https://arxiv.org/abs/2211.12703
year: 2022
---

[2211.12703] Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation














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



# Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation

Josh Gardner       Zoran Popović       Ludwig Schmidt
  
Paul G. Allen School of Computer Science & Engineering
  
University of Washington
  
{jpgard, zoran, schmidt}@cs.washington.edu

###### Abstract

Researchers have proposed many methods for fair and robust machine learning, but comprehensive empirical evaluation of their subgroup robustness is lacking.
In this work, we address this gap in the context of tabular data, where sensitive subgroups are clearly-defined, real-world fairness problems abound, and prior works often do not compare to state-of-the-art tree-based models as baselines.
We conduct an empirical comparison of several previously-proposed methods for fair and robust learning
alongside state-of-the-art tree-based methods
and other baselines.
Via experiments with more than 340,000

340000340{,}000 model configurations on eight datasets, we show that tree-based methods have strong subgroup robustness, even when compared to robustness- and fairness-enhancing methods.
Moreover, the best tree-based models tend to show good performance over a range of metrics, while robust or group-fair models can show brittleness, with significant performance differences across different metrics for a fixed model.
We also demonstrate that tree-based models show less sensitivity to hyperparameter configurations, and are less costly to train.
Our work suggests that tree-based ensemble models make an effective baseline for tabular data, and are a sensible default when subgroup robustness is desired.111See <https://github.com/jpgard/subgroup-robustness-grows-on-trees> for code to reproduce our experiments and detailed experimental results.

## 1 Introduction

Over the past decade, the field of machine learning (ML) has seen a dramatic expansion along two related lines. On one hand, concerns about the fairness of ML models, and more broadly their performance on data outside the training distribution, have grown [[50](#bib.bib50)]. Both theoretical and empirical works have raised these concerns, demonstrating the vulnerability of models to learn biases from data or suffer performance drops under distribution shifts [[44](#bib.bib44), [66](#bib.bib66), [35](#bib.bib35)]. On the other hand, an abundance of methods have been proposed to address these limitations. These include *fairness* methods used to equalize metrics across groups, as well as *distributional robustness* methods which optimize for a worst-case distribution within a bounded distance of the training distribution.

![Refer to caption](/html/2211.12703/assets/img/acsincome_acc_vs_worstgroup_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/acsincome_curves_and_baselines_with_inset_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/acsincome_worstgroup_acc_by_selection_metric.png)

![Refer to caption](/html/2211.12703/assets/img/acspubcov_acc_vs_worstgroup_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/acspubcov_curves_and_baselines_with_inset_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/acspubcov_worstgroup_acc_by_selection_metric.png)

![Refer to caption](/html/2211.12703/assets/img/larc-grade_acc_vs_worstgroup_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/larc-grade_curves_and_baselines_with_inset_nolegend.png)

![Refer to caption](/html/2211.12703/assets/img/larc-grade_worstgroup_acc_by_selection_metric.png)

![Refer to caption](/html/2211.12703/assets/img/legend_only.png)

Figure 1: Results from three datasets in our study. (a) Left: Tree-based methods such as XGBoost show similar subgroup robustness, with sometimes better overall performance, as robustness-enhancing or disparity-mitigation methods. (b) Center: Model performance frontiers corresponding to (a) show similar accuracy-robustness frontiers for XGBoost and DRO/Group DRO. (c) Right: Tree-based methods’ worst-group accuracy is robust to model selection metrics.

Our work begins from the observation that, while “fairness” and “robustness” are distinct concepts, methods in both areas often have similar goals. In particular, they share two (sometimes implicit) goals: First, models should have low *disparity* (variation in performance over all subgroups should be minimized; cf. [[75](#bib.bib75), [17](#bib.bib17), [25](#bib.bib25), [43](#bib.bib43)]). Second, models should have good *worst-group performance* (the lowest accuracy over all subgroups should be maximized; cf. [[74](#bib.bib74), [62](#bib.bib62)]). As a consequence of these two criteria, both fairness and robustness typically entail improving subgroup robustness.

Our work focuses on subgroup robustness in the context of tabular data, for several reasons. First, tabular data is widely used in practice, being the most common format in areas with important fairness impacts such as medicine, finance, and recommender systems [[38](#bib.bib38), [64](#bib.bib64)].
Second, tabular data often directly encodes sensitive attributes with respect to which subgroup robustness is desired (race, gender, income level), and these attributes are often declared by the individuals represented in the dataset.
This is an important difference compared to crowdsourced group annotations (e.g., gender or race for image datasets), which can be unreliable [[21](#bib.bib21)] or fail to reflect the lived experiences of the individuals represented [[19](#bib.bib19)].
Finally, tabular data is challenging to model.
It is often heterogeneous, containing mixed data types; it lacks the spatial, linguistic, or temporal structures common to other data (image, text, audio) where machine learning has seen dramatic progress over the past decade; and it can be relatively small, containing personal information which cannot be scraped at Internet scale.

In this work, we conduct a series of experiments to jointly compare methods from the fairness and robustness literature.
Despite the fact that many works often use similar metrics and datasets, there is a lack of large-scale empirical studies comparing relevant methods to each other, and to strong baselines. This lack of effective baselines is particularly concerning in light of recent works across several other areas of machine learning which have shown that simple baselines can perform surprisingly well against state-of-the-art techniques [[47](#bib.bib47)].

Our results show that modern tabular-data baselines can outperform even well-tuned state-of-the-art robustness- or fairness-enhancing methods with a fraction of the computational cost. Our analysis includes a set of tree-based models which achieve strong performance on tabular data but are absent from prior works on robustness, which largely only compare to deep learning-based approaches despite their tendency to perform poorly on tabular data [[38](#bib.bib38)].
In our experiments, we endeavor to tune both the “fair” or “robust” models and a representative suite of baseline methods for tabular data to obtain estimates of the empirical accuracy-subgroup robustness Pareto frontier for each model. To this end, we explore a wide space of architectures, regularization schemes, and training hyperparameters, across eight tabular datasets and 17 model classes. Our work is an empirical contribution concerned not with proposing new methods, but with conducting a novel and rigorous evaluation of existing methods for learning subgroup-robust models from tabular data. Our contributions are as follows:

* •

  Trees are subgroup-robust learners: While previous works have shown that modern tree-based ensembles have strong average-group performance (see Section [2.3](#S2.SS3 "2.3 Models for Tabular Data ‣ 2 Related Work ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")), we show that these models are also surprisingly robust to subgroup shift. Tree-based models achieve this robustness despite lacking any explicit robustness intervention and using only average-case classification losses, achieving competitive or better performance against state-of-the-art methods for fairness and robustness (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")a,[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")b) over a large hyperparameter and model search space. To support the above analysis, we apply several techniques for the analysis of hyperparameter sweeps such as model performance frontier curves (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")b,[3](#S4.F3 "Figure 3 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) and hyperparameter sensitivity analysis (Figures [7](#S6.F7 "Figure 7 ‣ 6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [8](#A6.F8 "Figure 8 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [9](#A6.F9 "Figure 9 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")).
* •

  Trees show show consistent performance across accuracy and robustness metrics: We empirically investigate the relationship between three types of metrics (accuracy, robustness, fairness). We show that metrics within these groups often agree, but the relationships across metric groups (i.e. robust CVaR risk and accuracy) are inconsistent. One consequence of this finding is that model selection techniques affect tree-based models and “robust” or fairness-enhancing methods differently: tree-based methods show consistent subgroup robustness across metrics (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")c), while robust and fairness-enhancing methods are “metric-brittle” and tend to display poor subgroup robustness according to other metrics (e.g. worst-group accuracy).
* •

  Trees are less sensitive to hyperparameters and less costly to train: We show that, in addition to their improved robustness when fully tuned, trees also require less tuning and are less costly to train. We demonstrate that trees have decreased sensitivity to hyperparameters – even when accounting for hyperparameter settings which were part of our initial grid but performed poorly across all datasets – and that these models require considerably less compute and financial cost to train when compared to deep learning-based robust learning techniques.

In particular, our results highlight the importance of the underlying model class for subgroup robustness in tabular data. Our results suggest that subgroup robustness interventions based on the loss-based perspective alone may be limited, particularly because existing loss-based methods for robustness are incompatible with nondifferentiable functions such as tree-based ensembles. We provide further discussion in Section [7](#S7 "7 Conclusion ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

We provide code to reproduce our experiments, along with an interactive tool to explore the best-performing hyperparameter configurations, at <https://github.com/jpgard/subgroup-robustness-grows-on-trees>.

## 2 Related Work

### 2.1 Fairness-Enhancing Methods for Supervised Learning

A wide variety of works have addressed fairness in the context of machine learning, where “fairness” is often measured by the equalization of some metric over groups [[5](#bib.bib5), [18](#bib.bib18)]. Most methods can be characterized as performing either pre- in-, or post-processing, which attempt to ensure fairness considerations are met in different stages of the modeling lifecycle. Preprocessing [[73](#bib.bib73), [15](#bib.bib15)] attempts to modify the input data to meet fairness criteria, while preserving the structure of the inputs and their relationship to the prediction target. Inprocessing uses a modified training procedure to explicitly optimize for a fairness-aware objective during model training. This includes using constrained or reduction-based optimization [[1](#bib.bib1), [72](#bib.bib72)], or including explicit regularizers designed to encourage fairness [[7](#bib.bib7), [11](#bib.bib11), [43](#bib.bib43), [56](#bib.bib56)]. Postprocessing [[32](#bib.bib32), [53](#bib.bib53)] operates on the predictions of a model, modifying them to achieve fairness criteria.

The impact of fairness under subpopulation shift is analyzed in [[48](#bib.bib48)], which shows theoretically that enforcing fairness during training can harm or improve the model, under certain conditions, and [[65](#bib.bib65)] conducts a causal analysis of fairness under covariate shift. Perhaps the work most closely related to the current study is Friedler et al. [[26](#bib.bib26)], which evaluates a set of pre-, in-, and postprocessing algorithms, demonstrating that many of these techniques show instability (variations in performance over different train-test splits) and that several fairness metrics are empirically correlated. Friedler et al. [[26](#bib.bib26)] do not, however, evaluate robust learning techniques, modern tree-based techniques (besides CART), or neural methods. Empirical evaluation of recent methods is needed.

### 2.2 Distributionally Robust Learning

While efforts in robust optimization date back several decades [[10](#bib.bib10)], recent works have adapted and extended robust learning approaches to deep learning. For example, several variants of distributionally robust optimization (DRO) have been proposed for the training of neural networks with robustness guarantees [[36](#bib.bib36), [46](#bib.bib46), [62](#bib.bib62), [61](#bib.bib61), [74](#bib.bib74), [25](#bib.bib25)]. While these approaches are not always explicitly oriented toward fairness, they are frequently evaluated in terms of their performance benefits for minimizing performance gaps between sensitive subgroups in real-world data [[74](#bib.bib74), [62](#bib.bib62), [33](#bib.bib33)].

A particular form of shift (and robustness) evaluated by these works is subgroup shift (also called subpopulation shift) [[50](#bib.bib50)], where the balance of subgroups in the data shifts between training and testing. Evaluating performance on individual subgroups is an extreme version of subgroup shift. In the context of fairness, these subgroups are often defined by one or more discrete sensitive attributes, such as race, gender, or income level, and intersections between these sensitive groups can often identify groups most susceptible to performance disparities [[14](#bib.bib14)].

### 2.3 Models for Tabular Data

Deep learning-based approaches largely have not achieved the transformative performance gains on tabular data that they have with other data modalities such as images, text, and audio [[38](#bib.bib38), [13](#bib.bib13), [64](#bib.bib64)].
The existing state-of-the-art for learning with tabular data is widely acknowledged to be tree-based methods, and in particular gradient-boosted trees [[31](#bib.bib31), [54](#bib.bib54), [64](#bib.bib64), [13](#bib.bib13)]. This includes GBM [[27](#bib.bib27), [28](#bib.bib28)], LightGBM [[40](#bib.bib40)], XGBoost [[16](#bib.bib16)], and CatBoost [[22](#bib.bib22), [55](#bib.bib55)], which often show only small differences in performance between them. Several deep learning-based approaches have recently attempted to close the gap of deep learning-based solutions for tabular data [[38](#bib.bib38), [2](#bib.bib2), [37](#bib.bib37), [54](#bib.bib54), [39](#bib.bib39)]. However, empirical analyses have shown that the reported performance of these methods does not generalize well to other datasets, and gradient boosting methods still achieve better performance with less tuning and a considerably smaller computational budget. For example, [[64](#bib.bib64)] and [[13](#bib.bib13)] both show, in separate analyses, that gradient boosting consistently outperforms these state-of-the-art tabular deep learning methods across several datasets, and tree-based methods achieve competitive performance with the deep learning approaches proposed in [[29](#bib.bib29), [37](#bib.bib37)].

The overall predictive performance of these tabular methods is evaluated in e.g. [[13](#bib.bib13), [29](#bib.bib29), [38](#bib.bib38), [54](#bib.bib54), [64](#bib.bib64)]. However, the existing literature does not investigate the fairness properties, subgroup performance, or robustness to subgroup shift of tabular data models; nor does it compare to subgroup-robust learning methods. Additionally, despite the widely-known strong performance of these models on tabular data, we are aware of no prior work which compares tree-based methods to robust neural network-based learners, despite the fact that the latter are commonly evaluated on tabular data (e.g. [[36](#bib.bib36), [43](#bib.bib43), [74](#bib.bib74), [62](#bib.bib62)]). Notably, [[1](#bib.bib1)] evaluates GBM in conjunction with their proposed inprocessing method, but does not evaluate subpopulation robustness; [[20](#bib.bib20)] evaluates GBM with fairness methods but does not compare to robust methods and does not tune the GBM with fairness methods.

## 3 Setup

Our work is primarily concerned with empirically evaluating the sensitivity of various supervised learning methods to subpopulation shift. Below, we introduce our formal model, and then describe the main axes of variation in our experiments. This includes (i)𝑖(i) a large set of models, many of which have not been directly compared in previous works; (i​i)𝑖𝑖(ii) the hyperparameters used for each model; (i​i​i)𝑖𝑖𝑖(iii) a suite of eight real-world fairness datasets; and (i​v)𝑖𝑣(iv) a set of evaluation metrics used across the disparate literature on robustness, subpopulation shift, and fairness in machine learning.

### 3.1 Preliminaries

Our work evaluates the task of learning a model fθ∈ℱsubscript𝑓𝜃ℱf\_{\theta}\in\mathcal{F}, a function from model class ℱℱ\mathcal{F} parameterized by
θ𝜃\theta. The parameters are learned from a dataset D≔(xi,yi)i=1n∼𝒫≔𝐷superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑛similar-to𝒫D\coloneqq(x\_{i},y\_{i})\_{i=1}^{n}\sim\mathcal{P}
where 𝒫𝒫\mathcal{P} is the data-generating distribution. This matches the case, most common in practice, where a single model is learned by estimating m​i​nθ​𝔼​(ℒ​(y,fθ​(x)))𝑚𝑖subscript𝑛𝜃𝔼ℒ𝑦subscript𝑓𝜃𝑥min\_{\theta}\mathbb{E}(\mathcal{L}(y,f\_{\theta}(x))) and deployed for all users. We evaluate the binary classification context, where y∈{0,1}𝑦01y\in\{0,1\} and fθ​(x)=y^∈[0,1]subscript𝑓𝜃𝑥^𝑦01f\_{\theta}(x)=\hat{y}\in[0,1] is the score assigned by f𝑓f to the outcome yi=1subscript𝑦𝑖1y\_{i}=1.
Each xi∈ℝdsubscript𝑥𝑖superscriptℝ𝑑x\_{i}\in\mathbb{R}^{d} can be partitioned into xi=(xi,1,…,xi,d−1,a)subscript𝑥𝑖subscript𝑥

𝑖1…subscript𝑥

𝑖𝑑1𝑎x\_{i}=(x\_{i,1},\ldots,x\_{i,d-1},a) where a𝑎a is a sensitive attribute. For notational convenience, we represent a𝑎a as a single feature here, but in our data, a𝑎a is typically defined as the concatenation of multiple binary sensitive attributes (e.g. gender = female, race = white). Let Da≔{(xi,yi)∼𝒫|ai=a}≔subscript𝐷𝑎conditional-setsimilar-tosubscript𝑥𝑖subscript𝑦𝑖𝒫subscript𝑎𝑖𝑎D\_{a}\coloneqq\{(x\_{i},y\_{i})\sim\mathcal{P}|a\_{i}=a\} denote the subgroup of the dataset with a given sensitive identity.

Following many of the previous works in both fairness [[18](#bib.bib18)] and robustness [[74](#bib.bib74), [43](#bib.bib43)], our analysis focuses on the performance of fθsubscript𝑓𝜃f\_{\theta} on each Dasubscript𝐷𝑎D\_{a} for all a∈𝒜𝑎𝒜a\in\mathcal{A}. We are particularly interested in the worst-group performance and the loss disparity, respectively defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒWorstGroup≔maxa∈𝒜⁡𝔼D∼𝒫​[ℒa​(fθ)]andℒDISP≔maxa,a′∈𝒜⁡𝔼D∼𝒫​[|ℒa​(fθ)−ℒa′​(fθ)|]formulae-sequence≔subscriptℒWorstGroup  subscript𝑎𝒜subscript𝔼similar-to𝐷𝒫delimited-[]subscriptℒ𝑎subscript𝑓𝜃and≔subscriptℒDISPsubscript  𝑎superscript𝑎′ 𝒜subscript𝔼similar-to𝐷𝒫delimited-[]subscriptℒ𝑎subscript𝑓𝜃subscriptℒsuperscript𝑎′subscript𝑓𝜃\displaystyle\mathcal{L}\_{\textrm{WorstGroup}}\coloneqq\max\_{a\in\mathcal{A}}~{}\mathbb{E}\_{D\sim\mathcal{P}}\big{[}\mathcal{L}\_{a}(f\_{\theta})\big{]}\;~{}~{}~{}~{}~{}\textrm{and}~{}~{}~{}~{}~{}\mathcal{L}\_{\textrm{DISP}}\coloneqq\max\_{a,a^{\prime}\in\mathcal{A}}\mathbb{E}\_{D\sim\mathcal{P}}\big{[}|\mathcal{L}\_{a}(f\_{\theta})-\mathcal{L}\_{a^{\prime}}(f\_{\theta})|\big{]} |  | (1) |

. These metrics can be thought of as assessing the sensitivity of fθsubscript𝑓𝜃f\_{\theta} to subgroup shift, evaluating the worst-group shift from D𝐷D to Dasubscript𝐷𝑎D\_{a} (note that subgroup shift is itself a form of covariate shift [[50](#bib.bib50)]).

### 3.2 Models

Our goal is to conduct a thorough empirical comparison of the subgroup robustness of a suite of methods for tabular data. We provide a more detailed description of all models used in Section [B](#A2 "Appendix B Model Details ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

* •

  Fairness-enhancing models: We evaluate the LFR preprocessing method of [[73](#bib.bib73)]; the inprocessing method of[[1](#bib.bib1)], and the postprocessing method of [[32](#bib.bib32)]. As in [[1](#bib.bib1), [20](#bib.bib20)], we use GBM as the base learner for each model; however, unlike [[20](#bib.bib20)], we also tune the base learner parameters.
* •

  Robust models: We evaluate both the CVaR and χ2superscript𝜒2\chi^{2} constraint forms of DRO via [[46](#bib.bib46)]; the CVaR and χ2superscript𝜒2\chi^{2} constraint forms of DORO [[74](#bib.bib74)]; Group DRO [[62](#bib.bib62)]; Marginal DRO [[24](#bib.bib24)]; and Maximum Weighted Loss Discrepancy [[43](#bib.bib43)]. Each robust optimization method is used with a multilayer perceptron (MLP), as in most previous works when using tabular data, e.g. [[43](#bib.bib43), [46](#bib.bib46), [62](#bib.bib62), [74](#bib.bib74)]. We note that the use of these losses requires performing optimization over a continuous function (such as a neural network), which makes these loss-based robustness interventions incompatible with existing training procedures for e.g. tree-based models.
* •

  Tree-based models: We evaluate GBM, LightGBM [[40](#bib.bib40)], XGBoost [[16](#bib.bib16)], and Random Forest models.
* •

  Baseline models: As baselines, we include L2subscript𝐿2L\_{2}-regularized logistic regression, Support Vector Machines, and fully-connected neural networks (MLP) with standard (non-DRO) ERM optimization.

### 3.3 Hyperparameter Sweeps

For each model, we conduct a grid search over a large set of hyperparameters. We give the complete set of hyperparameters tuned for each model in Section [F](#A6 "Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

Our hyperparameter search for each model is extensive by design, in order to ensure a reliable comparison of the best-performing models from each class. We expand the initial grid for continuous hyperparameters when a model on the Pareto frontier is at the edge of the grid. When one method includes another as a “base” learner (e.g. LFR preprocessing with GBM, or DRO with MLP), we include the full tuning grid for the base model (i.e. we explore the cross-product of all MLP hyperparameters with all DRO hyperparameters).

We note that previous works tend to either compare against a fixed baseline architecture and training hyperparameters (e.g. [[74](#bib.bib74), [38](#bib.bib38)]), perform manual tuning ([[46](#bib.bib46)]), or do not tune hyperparameters of the base model when using fairness-enhancing techniques ([[20](#bib.bib20)]).

### 3.4 Metrics

One goal of our work is to assess the empirical relationship between the model evaluation metrics used in the robustness, fairness, and classification literatures. Differences in the *training* objectives used across the various fair and robust models in prior work make such a comparison particularly useful. The relationship between these diverse evaluation metrics is not explored in existing work, where several different metrics have been used to compare the performance of models, evaluate their fairness with respect to subgroups, and measure their robustness to various shifts or outliers. We draw several metrics from prior works and report them for our experiments. We also explore the empirical relationships between different metrics.

For each metric ℒℒ\mathcal{L} (e.g. loss, accuracy, Equalized Odds), we can not only measure the overall empirical performance of a model, but we can measure the worst-group loss and disparity over subgroups of the data, as defined in Equation ([1](#S3.E1 "In 3.1 Preliminaries ‣ 3 Setup ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")). Worst-group and disparity measures, for various formulations of the loss functions above, are a widely-used measure of fairness and robustness [[12](#bib.bib12), [74](#bib.bib74), [62](#bib.bib62), [61](#bib.bib61), [33](#bib.bib33), [44](#bib.bib44)]. In particular, we use the ℒDISPsubscriptℒDISP\mathcal{L}\_{\textrm{DISP}} and ℒWorstGroupsubscriptℒWorstGroup\mathcal{L}\_{\textrm{WorstGroup}} with accuracy as the loss function. Accuracy is widely used in practice, directly interpretable, and invariant to rescaling of the model’s predictions.

We also use the CVaR risk metric from the robustness literature. CVaR risk is measured over a set of inputs D=(xi,yi)i=1N∼𝒫𝐷superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑁similar-to𝒫D=(x\_{i},y\_{i})\_{i=1}^{N}\sim\mathcal{P} and measures the worst-case weighted loss, according to some loss function ℓℓ\ell, at level α𝛼\alpha over the inputs in D𝐷D:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒCVaR​(D,𝒫)≔supq∈ΔN∑i=1Nqi​ℓ​(fθ;x)​s.t.​‖q‖∞≤1α​N≔subscriptℒCVaR𝐷𝒫subscriptsupremum𝑞superscriptΔ𝑁superscriptsubscript𝑖1𝑁subscript𝑞𝑖ℓ  subscript𝑓𝜃𝑥 s.t.subscriptnorm𝑞1𝛼𝑁\mathcal{L}\_{\textrm{CVaR}}(D,\mathcal{P})\coloneqq\sup\_{q\in\Delta^{N}}~{}\sum\_{i=1}^{N}q\_{i}\ell(f\_{\theta};x)~{}~{}~{}\textrm{s.t.}~{}||q||\_{\infty}\leq\frac{1}{\alpha N} |  | (2) |

([2](#S3.E2 "In 3.4 Metrics ‣ 3 Setup ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) is the risk function optimized by the DRO CVaR model [[46](#bib.bib46)], but it is also used more widely as a measure of the tail risk of a classifier (cf. [[74](#bib.bib74)]).

Additional fairness and robustness metrics are reported in Section [G](#A7 "Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") and defined in Section [C](#A3 "Appendix C Additional Metrics ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

| Dataset | n𝑛n | Features | Target | Sensitive Attributes |
| --- | --- | --- | --- | --- |
| ACS Income | 499,350 | 20 | Income ≥\geq 56k | Race, Sex |
| ACS Public Coverage | 341,487 | 19 | Public Ins. Coverage | Race, Sex |
| Adult | 48,845 | 14 | Income ≥\geq 56k | Race, Sex |
| Behavioral Risk Factors Surveillance System (BRFSS) | 175,745 | 28 | Diabetes | Race, Sex |
| Communities & Crime | 1994 | 113 | High Crime | Income Level, Race |
| COMPAS | 7,215 | 10 | Recidivism | Race, Sex |
| German Credit | 1,000 | 22 | Credit Risk | Age, Sex |
| Learning Analytics Architecture (LARC) | 169,032 | 26 | At-Risk (Grade) | URM Status, Sex |

Table 1: Overview of datasets used.

### 3.5 Datasets

We evaluate the 17 models over eight datasets covering a variety of prediction tasks and domains. We use two binary sensitive attributes from each dataset, for a total of four nonoverlapping subgroups in each dataset. A summary of the datasets used in this work is given in Table [1](#S3.T1 "Table 1 ‣ 3.4 Metrics ‣ 3 Setup ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). We provide additional details on each dataset, along with critical framing and context regarding these prediction tasks and their representations of individuals, in Section [A](#A1 "Appendix A Dataset Details ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

## 4 Results: Tree Models are Subgroup-Robust Learners

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_grouped_summary_2x4.png)


Figure 2: Overall Accuracy vs. Worst-Group Accuracy of robust, fairness-enhancing, tree-based, and baseline models over eight tabular datasets. Dashed lines indicate y=x𝑦𝑥y=x, when worst-group accuracy equal to overall accuracy (zero accuracy disparity). See Figures [12](#A7.F12 "Figure 12 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")-[15](#A7.F15 "Figure 15 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") for more detailed results by algorithm. Note: “discretization” artifacts are due to small test/dataset sizes.

![Refer to caption](/html/2211.12703/assets/img/performance_curves_2x3_with_inset.png)


Figure 3: Model performance frontiers, formed by tracing the convex envelope of model performance. Tree-based models achieve comparable and sometimes improved frontiers with the highest-performing robustness methods. (See also Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") for the remaining three datasets.)

![Refer to caption](/html/2211.12703/assets/img/disparity_curves_2x4.png)


Figure 4: Model disparity frontiers, formed by tracing the convex envelope of model disparity. Tree-based models achieve comparable and sometimes improved frontiers compared to the highest-performing robustness methods, particularly in high-accuracy regions.

Our main finding is that tree-based models are subgroup-robust learners. Tree-based models match or exceed the performance of distributionally robust and fairness-enhancing models in terms of best overall accuracy, worst-group accuracy, and accuracy disparity on all datasets evaluated.

A summary of our results is shown in Figures [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [2](#S4.F2 "Figure 2 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [3](#S4.F3 "Figure 3 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), and [4](#S4.F4 "Figure 4 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). Following previous works, our main evaluation metrics are accuracy-based: overall accuracy, worst-group accuracy, and accuracy disparity. Over the wide sweep of datasets and model configurations evaluated, modern tree-based models – GBM, LightGBM, Random Forest and XGBoost – all achieve subgroup performance characteristics on par with, or better than, the set of robust or fairness-enhancing models evaluated. For example, on three of the four largest datasets in our study (ACS Income, ACS Public Coverage, LARC), XGBoost achieves significantly *better* ℒℒ\mathcal{L} and ℒw​gsubscriptℒ𝑤𝑔\mathcal{L}\_{w}g than DRO-based methods (Group DRO, χ2superscript𝜒2\chi^{2} DRO), as indicated by nonoverlapping Clopper-Pearson CIs (α=0.05𝛼0.05\alpha=0.05).

The only cases where another algorithm achieves *better* maximum overall robustness (as measured by worst-group accuracy) than tree-based methods are DORO CVaR on Adult, and Inprocessing on Communities and Crime (see Figures [4](#S4.F4 "Figure 4 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [4](#S4.F4 "Figure 4 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")). We note that in both cases, (i)𝑖(i) these differences are not statistically significant, based on the shown Clopper-Pearson confidence intervals at α=0.05𝛼0.05\alpha=0.05, and (i​i)𝑖𝑖(ii) these differences occur at points *below* the maximum overall accuracy for each model. In all cases, at the point of maximum overall accuracy, tree-based models achieve equivalent or better worst-group accuracy than all other models evaluated, based on Clopper-Pearson confidence intervals at α=0.05𝛼0.05\alpha=0.05.

We note that this is particularly surprising because none of the tree-based models explicitly optimize for robustness, fairness, or subgroup performance in any way; in contrast, the distributionally robust learners and fairness-enhancing techniques explicitly optimize for such criteria and in some cases (DORO, DRO, Group DRO) provide explicit guarantees of various forms of robustness. A similar analysis showing accuracy disparity is shown in Figures [16](#A7.F16 "Figure 16 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), [17](#A7.F17 "Figure 17 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), and a complete set of model performance observations from each algorithm and dataset, are in Supplementary Section [G](#A7 "Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

To summarize our results over the large hyperparameter sweeps, we use model performance frontiers to measure the best possible set of tradeoffs between (ℒℒ\mathcal{L} and ℒWorstGroupsubscriptℒWorstGroup\mathcal{L}\_{\textrm{WorstGroup}}) and (ℒℒ\mathcal{L} and ℒDISPsubscriptℒDISP\mathcal{L}\_{\textrm{DISP}}). Model performance frontiers represent the envelope of the convex hull of all observations of ℒ​(fθ)∈ℱℒsubscript𝑓𝜃ℱ\mathcal{L}(f\_{\theta})\in\mathcal{F}. An example is shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")b and [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")d, with the remaining datasets in Figures [3](#S4.F3 "Figure 3 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") and [4](#S4.F4 "Figure 4 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). The frontiers allow us to compare the best-possible tradeoff between accuracy and worst-group accuracy (Fig [3](#S4.F3 "Figure 3 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"); higher is better) or between accuracy and accuracy disparity (Fig [4](#S4.F4 "Figure 4 ‣ 4 Results: Tree Models are Subgroup-Robust Learners ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"); lower is better).

## 5 Results: Robust Models Can Be Metric-Brittle

![Refer to caption](/html/2211.12703/assets/img/acsincome_complementary_vs_noncomplementary.png)


Figure 5: While “complementary” metrics (those measuring the same event with different conditioning) are closely correlated for all models, non-complementary metrics behave differently by model. Accuracy can vary widely for “robust” models with a fixed robust (CVaR) risk, while for tree models, the best (lowest) CVaR risk is typically associated with the best (highest) accuracy. ACS Income dataset shown; see Supplementary Figure [18](#A7.F18 "Figure 18 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") and Section [G](#A7 "Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") for additional datasets and metrics.

Our results show that, over all models and datasets, metrics which we refer to as “complementary” – those measuring the same event on different subsets, or with different conditioning – are strongly correlated with each other: accuracy and worst-group accuracy; CVaR and CVaR DORO; and Demographic Parity Difference and Equalized Odds Difference all show strong correlation with each model class ℱℱ\mathcal{F}; we show these correlations on Adult in Figure [5](#S5.F5 "Figure 5 ‣ 5 Results: Robust Models Can Be Metric-Brittle ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). The median ρ𝜌\rho values over all datasets and algorithms are 0.87, 0.99, 0.79 for the three pairs of metrics, respectively (we show the complete set of correlations for each model and metric in Figure [18](#A7.F18 "Figure 18 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")). These results show that model selection based on one metric from each pair (Overall Accuracy) is likely to lead to a model with strong performance for the other metric in the pair (e.g. Worst-Group Accuracy).

In contrast, our experiments show that there is a severe lack of correlation across almost all of the non-complementary pairs, shown in the bottom row of Figure [5](#S5.F5 "Figure 5 ‣ 5 Results: Robust Models Can Be Metric-Brittle ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). In particular, our results show that models which achieve a strong “robust” risk – for example, a low CVaR DORO risk, one model selection rule used in [[74](#bib.bib74)] – can achieve very low accuracies. Indeed, the “robust” learning methods are the most susceptible to this discrepancy over metrics in our experiments, as shown in Figure [6](#S5.F6 "Figure 6 ‣ 5 Results: Robust Models Can Be Metric-Brittle ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). While these results confirm the finding in previous works that fairness metrics are correlated [[26](#bib.bib26)], our broader comparison shows a brittleness to model selection metrics outside these sets of complementary metrics, and reveal the very strict sense in which robustness guarantees apply only to a limited range of metrics. This discrepancy is of practical significance given that robust models are often selected using robust metrics (i.e. DORO CVaR, cf. [[74](#bib.bib74)]), while models in practice are frequently evaluated using other metrics (i.e. accuracy, fairness metrics).

To illustrate the practical implications of this metric brittleness, we also explore the impact of various strategies for choosing a single fθ∈ℱsubscript𝑓𝜃ℱf\_{\theta}\in\mathcal{F} over a set of models in each hyperparameter grid. While this problem is implicitly addressed in almost all previous works, the decision is often handled differently. For example, models are tuned and selected by hand [[46](#bib.bib46)], chosen based on worst-group performance [[74](#bib.bib74)], or based on robust risk metrics [[62](#bib.bib62)]. The empirical implications of model selection methods for fairness and robustness are not well-understood.

To address this question, we show the worst-group performance of models over our sweeps, selected according to either *best accuracy* or *best CVaR* in Figure [6](#S5.F6 "Figure 6 ‣ 5 Results: Robust Models Can Be Metric-Brittle ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). Figure [6](#S5.F6 "Figure 6 ‣ 5 Results: Robust Models Can Be Metric-Brittle ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") demonstrates the downstream impact of the lack of correlation between robust risk metrics and worst-group accuracy: distributionally-robust models can suffer significant drops in worst-group accuracy, when selected based on robust risk. In contrast, tree-based models show “metric robustness” – that is, tree-based models which perform best according to the robust risk measure (CVaR) still achieve worst-group accuracy near the highest-accuracy model of the same class (here, XGBoost).

![Refer to caption](/html/2211.12703/assets/img/worstgroup_acc_by_selection_metric_1x5.png)


Figure 6: 
Worst-group accuracy for models with best overall accuracy (solid), and best CVaR (shaded).
Tree-based models show better performance across non-complementary metrics (as indicated by similar heights of orange bars within each plot), whereas robustness- and fairness-based models do not (statistically-significant gaps between blue, green bars): worst-group accuracy drops significantly between best-accuracy and best-CVaR DRO χ2superscript𝜒2\chi^{2} and Group DRO models (See also Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") for the remaining three datasets.).

## 6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train

For many practical applications, both the time and financial costs of training models are of prime importance. These also affect the amount of expertise required to train a model (with sensitive models requiring greater expertise) and directly impact the carbon footprint of training [[34](#bib.bib34)]. We conduct a hyperparameter sensitivity analysis by pruning the hyperparameter search space to eliminate configurations that performed poorly across all datasets, ranking the remaining configurations by performance (i.e. accuracy), and plotting the decline in performance over the ranked models. These results, shown in Figures [7](#S6.F7 "Figure 7 ‣ 6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") Section [F.2](#A6.SS2 "F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), demonstrate that tree-based models also show less sensitivity to hyperparameters than the other methods evaluated for both overall performance metrics (e.g., accuracy) and subgroup robustness metrics (e.g., worst-group accuracy).

These results are particularly important in light of the large differences in resources required to achieve similar levels of performance between robust models and the tree-based models: for example, the full hyperparameter grid sweep of size 12​k12𝑘12k XGBoost on the largest dataset in our study, ACS Income, completed in 1 CPU-day; DRO χ2superscript𝜒2\chi^{2} sweep of 3250 training runs completed in 58 *GPU*-days. Due to the differences in hardware required to train various models, we conduct an estimated comparison of the cost of training an individual model, and of the full sweep. These results, shown in Figure [19](#A7.F19 "Figure 19 ‣ G.1 Training Compute Cost ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), show that tree-based models are also considerably less expensive to train and tune.

![Refer to caption](/html/2211.12703/assets/img/hparam_ranked_plot_accuracy_worstgroup_test_acsincomeacspubcovadultbrfsscommunities-and-crimecompasgermanlarc-grade.png)


Figure 7: Performance over (truncated) hyperparameter grids on all datasets. Tree-based models (orange) show significantly less sensitivity to hyperparameter settings, for both subgroup and overall performance metrics (worst-group accuracy shown here), as indicated by nonoverlapping Clopper-Pearson CIs (α=0.05𝛼0.05\alpha=0.05). For additional results and methodology for constructing these plots, see Section [F.2](#A6.SS2 "F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

## 7 Conclusion

Machine learning has made significant progress in the past several years in identifying and addressing various challenges which can contribute to real-world performance disparities. However, our results suggest that another form of progress not widely acknowledged as having implications for fairness or robustness – advances in tabular data modeling with tree-based algorithms – have similar, if not greater, impact in practice than the state of the art in robust neural network learning or fairness-enhancing methods. Our results suggest that tree-based algorithms are a surprisingly strong baseline for subgroup robustness in tabular data, and that future work should compare against, and improve upon, the subgroup robustness of tree-based models. Our work suggests that, in practice, tree-based ensembles are an effective default for tasks where subgroup robustness is desired.

Our results also suggest that subgroup robustness on par with existing state of the art can be achieved with tree-based classifiers that are easier and computationally cheaper to train and tune than either robust or fairness-enhancing models, which tend to scale poorly with dataset size (in both n𝑛n and d𝑑d). Our work thus contributes a critical baseline, similar to how other recent works have contributed much-needed baselines in areas of machine learning affected by the rapid proliferation of new methods [[59](#bib.bib59), [67](#bib.bib67), [47](#bib.bib47)], and provides a strong benchmark for future robust and fair learning methods.

Our findings are limited to the set of hyperparameters and models explored in our experiments – a superset of those from the works discussed above (e.g. [[62](#bib.bib62), [74](#bib.bib74)]). Our findings do not demonstrate that robust learning methods cannot achieve subgroup robustness on par with e.g. XGBoost, but merely that this is not achieved with the configurations widely used in the literature. We note that the loss-based interventions largely favored by existing robust and fair learning techniques require optimization over a continuous function, typically implemented as a feedforward neural network; this also makes it difficult to disentangle whether the functional form, or the training procedure and objective, lead to the improved subgroup robustness of tree-based models observed in this study. Future work should investigate the subgroup robustness of non-MLP models trained using robust techniques.

While our experiments do not identify the *cause* of trees’ subgroup robustness, it is likely that this is a consequence of their strong overall performance on tabular data. These findings can be also viewed as an analogue of the empirical relationship between in-distribution and out-of-distribution accuracy in computer vision documented in [[49](#bib.bib49)] but now demonstrated for the tabular domain. It is possible that these improvements are due to (i)𝑖(i) an inductive bias in tree-based models beter suited to modeling differences in subpopulations in tabular data, or (i​i)𝑖𝑖(ii) the *ensembling* used by the tree-based models in this work. We leave an identification of such causal factors to future work.

## 8 Acknowledgements

This work is in part supported by the NSF AI Institute for Foundations of Machine Learning (IFML, CCF-2019844), Google, Microsoft, Open Philanthropy, and the Allen Institute for AI.

Moreover, our research utilized computational resources and services provided by the Hyak computing cluster at the University of Washington, and by Advanced Research Computing at the University of Michigan, Ann Arbor.

We are also grateful to Hongseok Namkoong, Tatsunori Hashimoto, John Miller, Michael Kim, Shafi Goldwasser, and attendees of the 2022 Institute for Foundations of Data Science Workshop on Distributional Robustness for useful feedback and discussions about the work, and to Christopher Brooks for assistance with the Learning Analytics Architecture (LARC) dataset.

## References

* [1]

  Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna
  Wallach.
  A reductions approach to fair classification.
  In International Conference on Machine Learning, pages 60–69.
  PMLR, 2018.
* [2]

  Sercan O Arık and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  In AAAI, volume 35, pages 6679–6687, 2021.
* [3]

  Michelle Bao, Angela Zhou, Samantha Zottola, Brian Brubach, Sarah Desmarais,
  Aaron Horowitz, Kristian Lum, and Suresh Venkatasubramanian.
  It’s compaslicated: The messy relationship between rai datasets and
  algorithmic fairness benchmarks.
  arXiv preprint arXiv:2106.05498, 2021.
* [4]

  Matias Barenstein.
  Propublica’s compas data revisited.
  arXiv preprint arXiv:1906.04711, 2019.
* [5]

  Solon Barocas, Moritz Hardt, and Arvind Narayanan.
  Fairness in machine learning.
  2017.
* [6]

  Solon Barocas, Moritz Hardt, and Arvind Narayanan.
  Fairness and Machine Learning.
  fairmlbook.org, 2019.
  <http://www.fairmlbook.org>.
* [7]

  Yahav Bechavod and Katrina Ligett.
  Learning fair classifiers: A regularization-inspired approach.
  2017.
* [8]

  Rachel KE Bellamy, Kuntal Dey, Michael Hind, Samuel C Hoffman, Stephanie Houde,
  Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta, Aleksandra
  Mojsilovic, et al.
  Ai fairness 360: An extensible toolkit for detecting, understanding,
  and mitigating unwanted algorithmic bias.
  arXiv preprint arXiv:1810.01943, 2018.
* [9]

  Rachel KE Bellamy, Kuntal Dey, Michael Hind, Samuel C Hoffman, Stephanie Houde,
  Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta, Aleksandra
  Mojsilović, et al.
  Ai fairness 360: An extensible toolkit for detecting and mitigating
  algorithmic bias.
  IBM Journal of Research and Development, 63(4/5):4–1, 2019.
* [10]

  Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski.
  Robust optimization, volume 28.
  Princeton university press, 2009.
* [11]

  Richard Berk, Hoda Heidari, Shahin Jabbari, Matthew Joseph, Michael Kearns,
  Jamie Morgenstern, Seth Neel, and Aaron Roth.
  A convex framework for fair regression.
  arXiv preprint arXiv:1706.02409, 2017.
* [12]

  Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz, Vanessa
  Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker.
  Fairlearn: A toolkit for assessing and improving fairness in ai.
  Microsoft, Tech. Rep. MSR-TR-2020-32, 2020.
* [13]

  Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin
  Pawelczyk, and Gjergji Kasneci.
  Deep neural networks and tabular data: A survey.
  arXiv preprint arXiv:2110.01889, 2021.
* [14]

  Joy Buolamwini and Timnit Gebru.
  Gender shades: Intersectional accuracy disparities in commercial
  gender classification.
  In Conference on fairness, accountability and transparency,
  pages 77–91. PMLR, 2018.
* [15]

  L Elisa Celis, Vijay Keswani, and Nisheeth Vishnoi.
  Data preprocessing to mitigate bias: A maximum entropy based
  approach.
  In International Conference on Machine Learning, pages
  1349–1359. PMLR, 2020.
* [16]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In Proceedings of the 22nd acm sigkdd international conference
  on knowledge discovery and data mining, pages 785–794, 2016.
* [17]

  Jianfeng Chi, Yuan Tian, Geoffrey J Gordon, and Han Zhao.
  Understanding and mitigating accuracy disparity in regression.
  In International Conference on Machine Learning, pages
  1866–1876. PMLR, 2021.
* [18]

  Sam Corbett-Davies and Sharad Goel.
  The measure and mismeasure of fairness: A critical review of fair
  machine learning.
  arXiv preprint arXiv:1808.00023, 2018.
* [19]

  Emily Denton, Mark Díaz, Ian Kivlichan, Vinodkumar Prabhakaran, and Rachel
  Rosen.
  Whose ground truth? accounting for individual and collective
  identities underlying dataset annotation.
  arXiv preprint arXiv:2112.04554, 2021.
* [20]

  Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt.
  Retiring adult: New datasets for fair machine learning.
  Advances in Neural Information Processing Systems, 34, 2021.
* [21]

  Samuel Dooley, Ryan Downing, George Wei, Nathan Shankar, Bradon Thymes, Gudrun
  Thorkelsdottir, Tiye Kurtz-Miott, Rachel Mattson, Olufemi Obiwumi, Valeriia
  Cherepanova, et al.
  Comparing human and machine bias in face recognition.
  arXiv preprint arXiv:2110.08396, 2021.
* [22]

  Anna Veronika Dorogush, Vasily Ershov, and Andrey Gulin.
  Catboost: gradient boosting with categorical features support.
  arXiv preprint arXiv:1810.11363, 2018.
* [23]

  Petros Drineas, Michael W Mahoney, and Nello Cristianini.
  On the nyström method for approximating a gram matrix for
  improved kernel-based learning.
  journal of machine learning research, 6(12), 2005.
* [24]

  John Duchi, Tatsunori Hashimoto, and Hongseok Namkoong.
  Distributionally robust losses for latent covariate mixtures.
  Operations Research, 2022.
* [25]

  John C Duchi and Hongseok Namkoong.
  Learning models with uniform performance via distributionally robust
  optimization.
  The Annals of Statistics, 49(3):1378–1406, 2021.
* [26]

  Sorelle A Friedler, Carlos Scheidegger, Suresh Venkatasubramanian, Sonam
  Choudhary, Evan P Hamilton, and Derek Roth.
  A comparative study of fairness-enhancing interventions in machine
  learning.
  In Proceedings of the conference on fairness, accountability,
  and transparency, pages 329–338, 2019.
* [27]

  Jerome H Friedman.
  Greedy function approximation: a gradient boosting machine.
  Annals of statistics, pages 1189–1232, 2001.
* [28]

  Jerome H Friedman.
  Stochastic gradient boosting.
  Computational statistics & data analysis, 38(4):367–378,
  2002.
* [29]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  Advances in Neural Information Processing Systems,
  34:18932–18943, 2021.
* [30]

  Alex Hanna, Emily Denton, Andrew Smart, and Jamila Smith-Loud.
  Towards a critical race methodology in algorithmic fairness.
  In Proceedings of the 2020 conference on fairness,
  accountability, and transparency, pages 501–512, 2020.
* [31]

  Vasyl Harasymiv.
  Lessons from 2 million machine learning models on kaggle, Dec 2015.
* [32]

  Moritz Hardt, Eric Price, and Nati Srebro.
  Equality of opportunity in supervised learning.
  Advances in neural information processing systems, 29, 2016.
* [33]

  Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang.
  Fairness without demographics in repeated loss minimization.
  In International Conference on Machine Learning, pages
  1929–1938. PMLR, 2018.
* [34]

  Peter Henderson, Jieru Hu, Joshua Romoff, Emma Brunskill, Dan Jurafsky, and
  Joelle Pineau.
  Towards the systematic reporting of the energy and carbon footprints
  of machine learning.
  Journal of Machine Learning Research, 21(248):1–43, 2020.
* [35]

  Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan
  Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al.
  The many faces of robustness: A critical analysis of
  out-of-distribution generalization.
  In Proceedings of the IEEE/CVF International Conference on
  Computer Vision, pages 8340–8349, 2021.
* [36]

  Weihua Hu, Gang Niu, Issei Sato, and Masashi Sugiyama.
  Does distributionally robust supervised learning give robust
  classifiers?
  In International Conference on Machine Learning, pages
  2029–2037. PMLR, 2018.
* [37]

  Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  arXiv preprint arXiv:2012.06678, 2020.
* [38]

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  Advances in Neural Information Processing Systems, 34, 2021.
* [39]

  Liran Katzir, Gal Elidan, and Ran El-Yaniv.
  Net-dnf: Effective deep modeling of tabular data.
  In International Conference on Learning Representations, 2020.
* [40]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei
  Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  Advances in neural information processing systems, 30, 2017.
* [41]

  Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu.
  Preventing fairness gerrymandering: Auditing and learning for
  subgroup fairness.
  In International Conference on Machine Learning, pages
  2564–2572. PMLR, 2018.
* [42]

  Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu.
  An empirical study of rich subgroup fairness for machine learning.
  In Proceedings of the conference on fairness, accountability,
  and transparency, pages 100–109, 2019.
* [43]

  Fereshte Khani, Aditi Raghunathan, and Percy Liang.
  Maximum weighted loss discrepancy.
  arXiv preprint arXiv:1906.03518, 2019.
* [44]

  Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang,
  Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips,
  Irena Gao, et al.
  Wilds: A benchmark of in-the-wild distribution shifts.
  In International Conference on Machine Learning, pages
  5637–5664. PMLR, 2021.
* [45]

  Ron Kohavi et al.
  Scaling up the accuracy of naive-bayes classifiers: A decision-tree
  hybrid.
  In Kdd, volume 96, pages 202–207, 1996.
* [46]

  Daniel Levy, Yair Carmon, John C Duchi, and Aaron Sidford.
  Large-scale methods for distributionally robust optimization.
  Advances in Neural Information Processing Systems,
  33:8847–8860, 2020.
* [47]

  Thomas Liao, Rohan Taori, Inioluwa Deborah Raji, and Ludwig Schmidt.
  Are we learning yet? a meta review of evaluation failures across
  machine learning.
  In Thirty-fifth Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track (Round 2), 2021.
* [48]

  Subha Maity, Debarghya Mukherjee, Mikhail Yurochkin, and Yuekai Sun.
  Does enforcing fairness mitigate biases caused by subpopulation
  shift?
  Advances in Neural Information Processing Systems,
  34:25773–25784, 2021.
* [49]

  John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh,
  Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt.
  Accuracy on the line: on the strong correlation between
  out-of-distribution and in-distribution generalization.
  In International Conference on Machine Learning, pages
  7721–7735. PMLR, 2021.
* [50]

  Kevin P. Murphy.
  Probabilistic Machine Learning: Advanced Topics.
  MIT Press, 2023.
* [51]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  Journal of Machine Learning Research, 12:2825–2830, 2011.
* [52]

  Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
  Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron
  Weiss, Vincent Dubourg, et al.
  Scikit-learn: Machine learning in python.
  the Journal of machine Learning research, 12:2825–2830, 2011.
* [53]

  Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q
  Weinberger.
  On fairness and calibration.
  Advances in neural information processing systems, 30, 2017.
* [54]

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  arXiv preprint arXiv:1909.06312, 2019.
* [55]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  Advances in neural information processing systems, 31, 2018.
* [56]

  Edward Raff, Jared Sylvester, and Steven Mills.
  Fair forests: Regularized tree induction to minimize model bias.
  In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics,
  and Society, pages 243–250, 2018.
* [57]

  Ali Rahimi and Benjamin Recht.
  Random features for large-scale kernel machines.
  Advances in neural information processing systems, 20, 2007.
* [58]

  Ali Rahimi and Benjamin Recht.
  Weighted sums of random kitchen sinks: Replacing minimization with
  randomization in learning.
  Advances in neural information processing systems, 21, 2008.
* [59]

  Steffen Rendle, Li Zhang, and Yehuda Koren.
  On the difficulty of evaluating baselines: A study on recommender
  systems.
  arXiv preprint arXiv:1905.01395, 2019.
* [60]

  Cynthia Rudin, Caroline Wang, and Beau Coker.
  The age of secrecy and unfairness in recidivism prediction.
  arXiv preprint arXiv:1811.00731, 2018.
* [61]

  Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang.
  Distributionally robust neural networks.
  In International Conference on Learning Representations, 2019.
* [62]

  Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang.
  Distributionally robust neural networks for group shifts: On the
  importance of regularization for worst-case generalization.
  arXiv preprint arXiv:1911.08731, 2019.
* [63]

  Morgan Klaus Scheuerman, Aaron Jiang, Katta Spiel, and Jed R Brubaker.
  Revisiting gendered web forms: An evaluation of gender inputs with
  (non-) binary people.
  In Proceedings of the 2021 CHI conference on human factors in
  computing systems, pages 1–18, 2021.
* [64]

  Ravid Shwartz-Ziv and Amitai Armon.
  Tabular data: Deep learning is not all you need.
  Information Fusion, 81:84–90, 2022.
* [65]

  Harvineet Singh, Rina Singh, Vishwali Mhasawade, and Rumi Chunara.
  Fairness violations and mitigation under covariate shift.
  In Proceedings of the 2021 ACM Conference on Fairness,
  Accountability, and Transparency, pages 3–13, 2021.
* [66]

  Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht,
  and Ludwig Schmidt.
  Measuring robustness to natural distribution shifts in image
  classification.
  Advances in Neural Information Processing Systems,
  33:18583–18599, 2020.
* [67]

  Yonglong Tian, Yue Wang, Dilip Krishnan, Joshua B Tenenbaum, and Phillip Isola.
  Rethinking few-shot image classification: a good embedding is all you
  need?
  In European Conference on Computer Vision, pages 266–282.
  Springer, 2020.
* [68]

  Angelina Wang, Vikram V Ramaswamy, and Olga Russakovsky.
  Towards intersectionality in machine learning: Including more
  identities, handling underrepresentation, and performing evaluation.
  arXiv preprint arXiv:2205.04610, 2022.
* [69]

  Michael Wick, Jean-Baptiste Tristan, et al.
  Unlocking fairness: a trade-off revisited.
  Advances in neural information processing systems, 32, 2019.
* [70]

  Christopher Williams and Matthias Seeger.
  Using the nyström method to speed up kernel machines.
  Advances in neural information processing systems, 13, 2000.
* [71]

  Tianbao Yang, Yu-Feng Li, Mehrdad Mahdavi, Rong Jin, and Zhi-Hua Zhou.
  Nyström method vs random fourier features: A theoretical and
  empirical comparison.
  Advances in neural information processing systems, 25, 2012.
* [72]

  Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rogriguez, and Krishna P
  Gummadi.
  Fairness constraints: Mechanisms for fair classification.
  In Artificial Intelligence and Statistics, pages 962–970.
  PMLR, 2017.
* [73]

  Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork.
  Learning fair representations.
  In International conference on machine learning, pages
  325–333. PMLR, 2013.
* [74]

  Runtian Zhai, Chen Dan, Zico Kolter, and Pradeep Ravikumar.
  Doro: Distributional and outlier robust optimization.
  In International Conference on Machine Learning, pages
  12345–12355. PMLR, 2021.
* [75]

  Xueru Zhang, Mohammadmahdi Khaliligarekani, Cem Tekin, et al.
  Group retention when using machine learning in sequential decision
  making: the interplay between user dynamics and fairness.
  Advances in Neural Information Processing Systems, 32, 2019.

## Appendix A Dataset Details

This section provides further detail on the datasets used in this work, along with their preprocessing.

For each dataset, we use an 80%/10%/10% train/validation/test split. The only exceptions are ACS datasets and LARC, where we use equally-sized train/test/validation splits (see below), and Adult, where we use the official train-test split.

For each dataset, we use two binary sensitive attributes derived from existing features. These attributes are primarily selected to both align with real-world sensitive attributes in practice, and, where possible, to match the implementations in previous works. For methods which use group information, each intersection of the sensitive attributes are considered a separate group (for a total of 22superscript222^{2} nonoverlapping subgroups).

* •

  ACS Income222<https://github.com/zykls/folktables>: Proposed in [[20](#bib.bib20)], consists of approximately 160k responses to the 2018 American Community Survey. Since it is proposed as a replacement for the Adult dataset, we perform income prediction task, where the label is an indicator for whether an individuals’ income exceeds the median ($56,000
  currency-dollar56000\$56,000). Sensitive attributes are race and gender. We use a larger set of features than that described in [[20](#bib.bib20)], which we found to improve classifier performance. The sensitive feature for race is coded as “white alone” vs. other categories, as in [[20](#bib.bib20)]. We use a subsample of the full 1.6M records for our experiments due to computational constraints (for example, fairness methods scale linearly or quadratically in dataset size and feature size; robustness methods also incur extra costs as data dimensionality increases).
* •

  ACS Public Coverage: Derived from the same raw data source as ACS Income, described above. The task is to predict whether an individual is covered by public health insurance. Sensitive attributes are race and gender. We use identical feature set to [[20](#bib.bib20)]. The sensitive features and subsampling are handled as in ACS Income.
* •

  Adult333<https://archive.ics.uci.edu/ml/machine-learning-databases/adult/>: A widely-used benchmark dataset derived from 1994 US Census data [[45](#bib.bib45)]. The task is to predict whether an individuals’ income exceeded $50,000
  currency-dollar50000\$50,000. Sensitive attributes are race and gender. We use the standard train-test split for Adult, following the preprocessing code of [[6](#bib.bib6)]444<https://fairmlbook.org/code/adult.html>, and we further split the set partition evenly into validation and test.
* •

  BRFSS: The Behavioral Risk Factors and Surveillance System555<https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system> is a large-scale phone survey conducted annually from a random sample of adults in the United States by the Centers for Disease Control and Prevention 666<https://www.cdc.gov/brfss/annual_data/annual_data.htm>. The objective of the BRFSS is to collect uniform, state-specific data on preventive health practices and risk behaviors that are linked to chronic diseases, injuries, and preventable infectious diseases in the adult population. Respondents answer questions related to personal health, lifestyle habits, and health care coverage. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world. We use the BRFSS sample from 2015.
* •

  Communities and Crime777<https://archive.ics.uci.edu/ml/datasets/communities+and+crime>: A set of features describing a community, and the prediction target is whether the community has an elevated crime rate. Following [[43](#bib.bib43), [41](#bib.bib41), [42](#bib.bib42)], we predict a binary label for whether the violent crime rate exceeds a threshold of 0.08. Sensitive attributes are race and income level. We use the preprocessing of [[43](#bib.bib43)] for this dataset, where the race feature is a binary indicator for whether the feature racePctWhite>0.85racePctWhite0.85\texttt{racePctWhite}>0.85 and the income level is an indicator for whether the income is above the median community income.
* •

  COMPAS888<https://github.com/propublica/compas-analysis>: Parole records from Florida, USA, where the task is to predict whether an individual will recidivate within two years. Sensitive attributes are race and gender.

  While every labeled dataset contains human biases inherent in collecting and categorizing the data, the COMPAS dataset in particular has been the subject of valid critiques, and is mostly included for comparisons to prior work. We note that the COMPAS dataset reflects the patterns of policing and social processes in a particular community (South Florida) at a specific time, and as others have noted [[69](#bib.bib69)], each stage of the COMPAS dataset’s creation introduces opportunities for bias [[60](#bib.bib60), [4](#bib.bib4)] and that measurement biases and errors with COMPAS have been widely documented [[3](#bib.bib3)].
* •

  German Credit999<https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>: Credit application records, where the goal is to predict whether an individual has low or high credit risk. Sensitive attributes are age and gender. There are two versions of the German Credit dataset, a “numeric” version which contains binarized versions of most categorical features (with some removed), and a non-numeric version, which also contains categorical features. We do not use the “numeric” version of the dataset used by several other works. We found the numeric version of the dataset to be poorly-documented, lack useful features which were present in the “non-numeric” version, and contain features which actually mixed multiple variables . Using the non-numeric version, we extract separate features for sex and marital status (which are combined under a single feature in the numeric dataset).
* •

  LARC101010<https://enrollment.umich.edu/data/learning-analytics-data-architecture-larc>: The Learning Analytics Data Architecture (LARC) Data Set is a research-focused data set containing information about students who have attended the University of Michigan since the mid-1990s. The data includes features related to students, their enrollment, and performance, similar to the electronic records stored by many institutions of higher education. The data is divided into that which is constant throughout a student’s academic career (e.g., ethnicity, SAT test scores, high school GPA, and earned degrees), that which can change from term to term (e.g., academic level, academic career, term GPA, and enrolled credits), and that which can change from class to class (e.g., subject, catalog number, earned grade, etc.). The prediction target is an indicator for whether a student will receive a grade above the median in a course; this is commonly referred to as “at-risk” grade prediction and is used to identify students at risk of struggling in a course. Sensitive attributes are “Underrepresented Minority” status (a common indicator of diversity reported by all accredited institutions in the United States) and students’ self-reported sex.

It is critical to note that any notion of fairness or robustness in real-world societal contexts involves much more than even the sets of two demographic attributes considered here [[68](#bib.bib68)]. We also note that our treatment of these sensitive attributes as binary, while consistent with the majority of the fairness and robustness literature upon which this work is based, is necessarily reductionistic, and in practice many of these sensitive attributes are neither binary [[63](#bib.bib63)] nor fixed [[30](#bib.bib30)].

Furthermore, we urge readers to consider that, while these datasets are commonly used as benchmarks for fair or subgroup-learning, there are important social and contextual factors that must be considered for any real-world deployment of a model in these tasks to be considered “fair” [[3](#bib.bib3)].

## Appendix B Model Details

Fairness-Enhancing Models: Following [[20](#bib.bib20)], we evaluate one method each of pre-, in-, and postprocessing. The preprocessing method of [[73](#bib.bib73)] attempts to learn a transformation of the original inputs which minimally distorts the original data and its relationship to the labels, while ensuring that both group fairness (the proportion of members in a protected group receiving positive classification is identical to the proportion in the overall population) and individual fairness (similar individuals are treated similarly). The inprocessing method of [[1](#bib.bib1)] attempts to reduce fair classification to a cost-sensitive classification problem, where the goal is to minimize the prediction error subject to one or more fairness constraints. Finally, the postprocessing method we use is from [[32](#bib.bib32)], which randomizes the predictions of a fixed fθsubscript𝑓𝜃f\_{\theta} to satisfy equalized odds criterion. All fairness methods are implemented to simultaneously satisfy their constraints across both sensitive attributes in our datasets. We use the implementations of aif360 [[9](#bib.bib9)] and fairlearn [[12](#bib.bib12)] for all fairness interventions.

None of the fairness methods encompasses a prediction model, so following [[20](#bib.bib20)], we pair each method (in-, pre-, and postprocessing) with a gradient-boosted tree (GBM) [[27](#bib.bib27), [51](#bib.bib51)]. However, unlike [[20](#bib.bib20)], we perform hyperparameter sweeps for both the fairness methods and the GBM.

Distributionally Robust Models: We draw from several classes of modern robust learning techniques. We utilize two variants of Distributionally Robust Optimization (DRO), which attempts to solve

|  |  |  |  |
| --- | --- | --- | --- |
|  | minθ​supQ∈𝒰​(D)𝔼S∼Q​[ℒ​(fθ)]subscript𝜃subscriptsupremum𝑄𝒰𝐷subscript𝔼similar-to𝑆𝑄delimited-[]ℒsubscript𝑓𝜃\min\_{\theta}\sup\_{Q\in\mathcal{U}(D)}\mathbb{E}\_{S\sim Q}\big{[}\mathcal{L}(f\_{\theta})\big{]} |  | (3) |

where 𝒰𝒰\mathcal{U} defines an uncertainty set with respect to the training distribution D𝐷D. We evaluate two widely-used formulations of the uncertainty set 𝒰𝒰\mathcal{U} in Equation ([3](#A2.E3 "In Appendix B Model Details ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")). The first is the set of distributions with bounded likelihood ratio to D𝐷D, such that ([3](#A2.E3 "In Appendix B Model Details ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) defines the conditional value at risk (CVaR). The second is the set of distributions with bounded χ2superscript𝜒2\chi^{2}-divergence to D𝐷D. We refer to these as CVaR-DRO and χ2superscript𝜒2\chi^{2}-DRO, respectively. We use the efficient implementation of [[46](#bib.bib46)] for these methods.

We also evaluate the “Distributional and Outlier Robust Optimization” (DORO) of [[74](#bib.bib74)], which adds an ϵitalic-ϵ\epsilon parameter to ([3](#A2.E3 "In Appendix B Model Details ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) such that only the ϵitalic-ϵ\epsilon-smallest fraction of the data, when sorted by ℒℒ\mathcal{L}, are considered at each update step; this has the effect of ignoring outliers.

We additionally evaluate Group DRO [[62](#bib.bib62)], seeks to minimize the worst-group loss by solving

|  |  |  |  |
| --- | --- | --- | --- |
|  | minθ​supg∈𝒢𝔼(x,y)∼𝒟g​[ℒ​(fθ)]subscript𝜃subscriptsupremum𝑔𝒢subscript𝔼similar-to𝑥𝑦subscript𝒟𝑔delimited-[]ℒsubscript𝑓𝜃\min\_{\theta}\sup\_{g\in\mathcal{G}}\mathbb{E}\_{(x,y)\sim\mathcal{D}\_{g}}\big{[}\mathcal{L}(f\_{\theta})\big{]} |  | (4) |

. Finally, we evalute the Maximum Weighted Loss Discrepancy (MWLD) of [[43](#bib.bib43)]. This formulation adds an extra term, ℒLV≔Var(ℒ(fθ(xi)∈X)\mathcal{L}\_{\textrm{LV}}\coloneqq\textrm{Var}(\mathcal{L}(f\_{\theta}(x\_{i})\in X), to the ERM objective during training. The MWLD objective is shown in [[43](#bib.bib43)] to be related to group fairness and robustness to subgroup shifts.

Tree-Based Models: As we note in Section [2.3](#S2.SS3 "2.3 Models for Tabular Data ‣ 2 Related Work ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), several modern tree-based methods achieve effectively identical performance on many tasks, with several flavors of gradient-boosted trees (GBM, LightGBM, CatBoost, XGBoost) widely being considered the state-of-the art models for tabular data. Therefore, we evaluate GBM (in order to compare directly to [[20](#bib.bib20)], which combines GBM with our fairness methods of interest) and LightGBM (due to its scalability, which is required for large-scale hyperparameter tuning over the datasets in this work). We also evaluate Random Forests in order to compare to a non-gradient-boosted tree-based classifier.

Baseline Supervised Learning Models: In addition to the above-described methods, we also include the following standard supervised learning methods for comparison: L2subscript𝐿2L\_{2}-regularized logistic regression, and Support Vector Machines (SVM). For the SVM methods, because learning nonlinear kernels for large datasets with many features can be prohibitively expensive, we instead use two kernel approximation methods for learning: the Nystroem kernel method [[23](#bib.bib23), [70](#bib.bib70)] and random Fourier features [[57](#bib.bib57), [58](#bib.bib58), [71](#bib.bib71)]. For all baseline methods, we use the implementation of [[52](#bib.bib52)].

## Appendix C Additional Metrics

In addition to the metrics reported and defined in Section [3.4](#S3.SS4 "3.4 Metrics ‣ 3 Setup ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), we also use the following metrics in our supplementary results reported below:

Accuracy: The accuracy is defined as the fraction of labels correctly predicted at a given threshold: ℒAcc≔𝟙(y^==y)\mathcal{L}\_{\textrm{Acc}}\coloneqq\mathbbm{1}(\hat{y}==y), where y^=𝟙(fθ(x)>≥t)\hat{y}=\mathbbm{1}(f\_{\theta}(x)>\geq t) is the predicted label of x𝑥x using threshold t𝑡t. We use t=0.5𝑡0.5t=0.5 throughout.

Cross-Entropy: We use the standard binary cross-entropy measure, defined as ℒce​(fθ;x,y)=−y​log⁡(fθ​(x))−(1−y)​log⁡(1−fθ​(x))subscriptℒce

subscript𝑓𝜃𝑥𝑦𝑦subscript𝑓𝜃𝑥1𝑦1subscript𝑓𝜃𝑥\mathcal{L}\_{\textrm{ce}}(f\_{\theta};x,y)=-y\log(f\_{\theta}(x))-(1-y)\log(1-f\_{\theta}(x)).

DORO CVaR Risk: The DORO CVaR risk is a version of CVaR risl ([2](#S3.E2 "In 3.4 Metrics ‣ 3 Setup ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) which excludes the ϵitalic-ϵ\epsilon-largest-loss elements in D𝐷D in an effort to avoid outliers. Formally, the DORO CVaR risk is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒCVaR-DORO​(D,𝒫,ϵ)≔infD′ℒCVaR​(D,𝒫′):∃𝒫~′​s.t.​𝒫=(1−ϵ)​𝒫′+ϵ​P~′:≔subscriptℒCVaR-DORO𝐷𝒫italic-ϵsubscriptinfimumsuperscript𝐷′subscriptℒCVaR𝐷superscript𝒫′superscript~𝒫′s.t.𝒫1italic-ϵsuperscript𝒫′italic-ϵsuperscript~𝑃′\mathcal{L}\_{\textrm{CVaR-DORO}}(D,\mathcal{P},\epsilon)\coloneqq\inf\_{D^{\prime}}{\mathcal{L}\_{\textrm{CVaR}}(D,\mathcal{P}^{\prime}):\exists\tilde{\mathcal{P}}^{\prime}~{}~{}~{}\textrm{s.t.}~{}\mathcal{P}=(1-\epsilon)\mathcal{P}^{\prime}+\epsilon\tilde{P}^{\prime}} |  | (5) |

where ϵitalic-ϵ\epsilon is a hyperparameter corresponding to the fraction of outliers in the dataset. We note that ([5](#A3.E5 "In Appendix C Additional Metrics ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")) is the loss function directly optimized by the DORO-CVaR model, but it has been used as a more general evaluation of the outlier-robust tail risk of a classifier (cf. [[74](#bib.bib74)]).

Demographic Parity Difference: Demographic Parity (DP) is a fairness criterion that indicates the positive prediction rates across two disjoint subgroups a,a′∈A

𝑎superscript𝑎′
𝐴a,a^{\prime}\in A are equal [[6](#bib.bib6)]. That is, when demographic parity is satisfied, P​(fθ​(xi;ai=a)=1)=P​(fθ​(xj;aj=a′)=1)​∀i,j𝑃subscript𝑓𝜃

subscript𝑥𝑖subscript𝑎𝑖
𝑎1

𝑃subscript𝑓𝜃

subscript𝑥𝑗subscript𝑎𝑗
superscript𝑎′1for-all𝑖𝑗P(f\_{\theta}(x\_{i};a\_{i}=a)=1)=P(f\_{\theta}(x\_{j};a\_{j}=a^{\prime})=1)\forall i,j. The Demographic Parity Difference measures the degree to which this constraint is violated, and for nonbinary sensitive subgroups, it measures the worst-case difference:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒD​P−D​i​f​f≔maxa,a′⁡|P​(fθ​(xi;ai=a)=1)−P​(fθ​(xj;aj=a′)=1)|≔subscriptℒ𝐷𝑃𝐷𝑖𝑓𝑓subscript  𝑎superscript𝑎′𝑃subscript𝑓𝜃  subscript𝑥𝑖subscript𝑎𝑖 𝑎1𝑃subscript𝑓𝜃  subscript𝑥𝑗subscript𝑎𝑗 superscript𝑎′1\mathcal{L}\_{DP-Diff}\coloneqq\max\_{a,a^{\prime}}\Big{|}P(f\_{\theta}(x\_{i};a\_{i}=a)=1)-P(f\_{\theta}(x\_{j};a\_{j}=a^{\prime})=1)\Big{|} |  | (6) |

Equalized Odds Difference: Equalized Odds (EO) is a fairness criterion indicating that the true positive and false positive rates are equal across two groups. The Equalized Odds Difference measures worst-case violation of Equalized Odds, across sensitive subgroups (this is ℒD​I​S​Psubscriptℒ𝐷𝐼𝑆𝑃\mathcal{L}\_{DISP} with ℒℒ\mathcal{L} as DP). Equalized Odds Difference can be formulated as the greater of two metrics: the true positive rate difference, and the false positive rate difference [[12](#bib.bib12)].

## Appendix D Model Performance Frontier Curves

This section describes how Model Performance Frontier curves are computed. We note that this work is not the first to use convex envelopes as a way to understand model performance; for example, [[1](#bib.bib1)] uses convex envelopes to understand the relationship between fairness constraint violations and model error.

To compute a model envelope in 2D, we use an algorithm to compute the convex hull, and then trace the relevant edge of the convex hull corresponding to the best-achieved performance tradeoffs under the two metrics (depending on whether these metrics are maximized, or minimized).

We provide Python code to compute these curves in the code repository associated with this paper; for completeness, we also provide the full algorithm in Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix D Model Performance Frontier Curves ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

Algorithm 1  Model performance frontiers. This shows the computation where higher values are better for both metrics m(1)superscript𝑚1m^{(1)}, m(2)superscript𝑚2m^{(2)}.

(mi(1),(mi(2))i=i|𝒢|\big{(}m\_{i}^{(1)},(m\_{i}^{(2)}\big{)}\_{i=i}^{|\cal{G}|} ▷▷\triangleright Model performance metrics m(1),m(2)

superscript𝑚1superscript𝑚2m^{(1)},m^{(2)} for each configuration in grid 𝒢𝒢\mathcal{G}

ConvexHull, a method which computes the convex hull for a set of points and returns them in clockwise order.

Idxs ⊆1,…,|𝒢|absent

1…𝒢\subseteq{1,\ldots,|\cal{G}|} ▷▷\triangleright Indices of inputs on frontier.

vertices←ConvexHull​(Input)←verticesConvexHullInput\textrm{vertices}\leftarrow\textrm{ConvexHull}(\textrm{Input})

idxs←[]←idxs

\textrm{idxs}\leftarrow[~{}] ▷▷\triangleright Initialize array of frontier points.

top\_idx=argmaxm+​i(2)(mi(1),(mi(2))i∈vertices\textrm{top\\_idx}=\textrm{argmax}\_{m\_{+}i^{(2)}}\big{(}m\_{i}^{(1)},(m\_{i}^{(2)}\big{)}\_{i}\in\textrm{vertices} ▷▷\triangleright Add uppermost point to frontier

idx←top\_idx←idxtop\_idx\textrm{idx}\leftarrow\textrm{top\\_idx}

idxs←idxs+[idx]←idxsidxsdelimited-[]idx\textrm{idxs}\leftarrow\textrm{idxs}+[\textrm{idx}]

mc​u​r​r(1),mc​u​r​r(2)←vertices​[idx]←

superscriptsubscript𝑚𝑐𝑢𝑟𝑟1superscriptsubscript𝑚𝑐𝑢𝑟𝑟2
verticesdelimited-[]idxm\_{curr}^{(1)},m\_{curr}^{(2)}\leftarrow\textrm{vertices}[\textrm{idx}]

next←(|vertices|+1)​mod​|vertices|←nextvertices1modvertices\textrm{next}\leftarrow(|\textrm{vertices}|+1)\textrm{mod}|\textrm{vertices}|

mn​e​x​t(1),mn​e​x​t(2)←vertices​[next]←

superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑛𝑒𝑥𝑡2
verticesdelimited-[]nextm\_{next}^{(1)},m\_{next}^{(2)}\leftarrow\textrm{vertices}[\textrm{next}]

while mn​e​x​t(1)<mc​u​r​r(1)superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑐𝑢𝑟𝑟1m\_{next}^{(1)}<m\_{curr}^{(1)} do ▷▷\triangleright Trace frontier counterclockwise

idxs←next←idxsnext\textrm{idxs}\leftarrow\textrm{next}

idx←(idx+1)​mod​|vertices|←idxidx1modvertices\textrm{idx}\leftarrow(\textrm{idx}+1)~{}\textrm{mod}~{}|\textrm{vertices}|

next←(next+1)​mod​|vertices|←nextnext1modvertices\textrm{next}\leftarrow(\textrm{next}+1)~{}\textrm{mod}~{}|\textrm{vertices}|

mc​u​r​r(1),mc​u​r​r(2)←vertices​[idx]←

superscriptsubscript𝑚𝑐𝑢𝑟𝑟1superscriptsubscript𝑚𝑐𝑢𝑟𝑟2
verticesdelimited-[]idxm\_{curr}^{(1)},m\_{curr}^{(2)}\leftarrow\textrm{vertices}[\textrm{idx}]

mn​e​x​t(1),mn​e​x​t(2)←vertices​[next]←

superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑛𝑒𝑥𝑡2
verticesdelimited-[]nextm\_{next}^{(1)},m\_{next}^{(2)}\leftarrow\textrm{vertices}[\textrm{next}]

end while

idx←top\_idx←idxtop\_idx\textrm{idx}\leftarrow\textrm{top\\_idx}

next←(|vertices|−1)​mod​|vertices|←nextvertices1modvertices\textrm{next}\leftarrow(|\textrm{vertices}|-1)~{}\textrm{mod}~{}|\textrm{vertices}|

mc​u​r​r(1),mc​u​r​r(2)←vertices​[idx]←

superscriptsubscript𝑚𝑐𝑢𝑟𝑟1superscriptsubscript𝑚𝑐𝑢𝑟𝑟2
verticesdelimited-[]idxm\_{curr}^{(1)},m\_{curr}^{(2)}\leftarrow\textrm{vertices}[\textrm{idx}]

mn​e​x​t(1),mn​e​x​t(2)←vertices​[next]←

superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑛𝑒𝑥𝑡2
verticesdelimited-[]nextm\_{next}^{(1)},m\_{next}^{(2)}\leftarrow\textrm{vertices}[\textrm{next}]

while mn​e​x​t(1)>mc​u​r​r(1)superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑐𝑢𝑟𝑟1m\_{next}^{(1)}>m\_{curr}^{(1)} do ▷▷\triangleright Trace frontier clockwise

idxs←next←idxsnext\textrm{idxs}\leftarrow\textrm{next}

idx←(idx−1)​mod​|vertices|←idxidx1modvertices\textrm{idx}\leftarrow(\textrm{idx}-1)~{}\textrm{mod}~{}|\textrm{vertices}|

next←(next−1)​mod​|vertices|←nextnext1modvertices\textrm{next}\leftarrow(\textrm{next}-1)~{}\textrm{mod}~{}|\textrm{vertices}|

mc​u​r​r(1),mc​u​r​r(2)←vertices​[idx]←

superscriptsubscript𝑚𝑐𝑢𝑟𝑟1superscriptsubscript𝑚𝑐𝑢𝑟𝑟2
verticesdelimited-[]idxm\_{curr}^{(1)},m\_{curr}^{(2)}\leftarrow\textrm{vertices}[\textrm{idx}]

mn​e​x​t(1),mn​e​x​t(2)←vertices​[next]←

superscriptsubscript𝑚𝑛𝑒𝑥𝑡1superscriptsubscript𝑚𝑛𝑒𝑥𝑡2
verticesdelimited-[]nextm\_{next}^{(1)},m\_{next}^{(2)}\leftarrow\textrm{vertices}[\textrm{next}]

end while

return idxs

## Appendix E Training Details

We train all models using the original optimizer, SGD, used in their original works [[74](#bib.bib74), [46](#bib.bib46), [43](#bib.bib43), [62](#bib.bib62)]. For all models, we train for a fixed number of epochs, but keep the weights from the best epoch based on the loss on the validation set. The number of epochs used for each dataset is as follows: ACS Income 50 epochs; Adult 300 epochs; Communities and Crime 100 epochs; COMPAS 300 epochs; German 50 epochs.

For all neural network-based models, we used a fixed batch size of 128 as in [[38](#bib.bib38)]; we found that varying the batch size did not affect performance but significantly increased the computational cost of hyperparameter sweeps. This also ensures that each model trained with batching sees the same number of examples during training on a given dataset.

Neural-network-based models were trained on GPU, either NVIDIA RTX 2080 Ti GPUs with 11GB of RAM, or NVIDIA Tesla M60 GPUs with 8 GB of RAM.

## Appendix F Hyperparameter Grids

### F.1 Grid Definition

We detail the hyperparameter grids for each experiment in Table [2](#A6.T2 "Table 2 ‣ F.1 Grid Definition ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). For each dataset, we perform a full hyperparameter grid sweep.

For methods which are built on a “base” model (i.e. Group DRO, which uses an MLP model, or Preprocessing, which is paired with GBM as in [[20](#bib.bib20)]), we tune the full grid of hyperparameters for the base model in addition to the hyperparameters for that method, as indicated in Table [2](#A6.T2 "Table 2 ‣ F.1 Grid Definition ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"). We note that this is not always the case in previous works; for example, [[20](#bib.bib20)] uses the default hyperparameters for GBM with fairness methods, and many DRO-based works use a fixed architecture or optimization hyperparameters for their published comparisons.

We use default parameters with the given implementation for all methods except where indicated.

Tree-based methods: For GBM and random forest, we use the implementation of [[52](#bib.bib52)]. For LightGBM we use the original implementation of [[40](#bib.bib40)]111111<https://github.com/microsoft/LightGBM>. For XGBoost we use the original implementation of [[16](#bib.bib16)]121212<https://github.com/dmlc/xgboost>. For each method, we construct hyperparameter grids by beginning with large sweeps around the default hyperparameters of each method, and then pruning the sweeps to a tractable size manually by inspecting accuracy, robustness, and fairness metrics. For XGBoost, we only use training methods available with GPU-based training to ensure scalability (note that only CPU-based training was used for XGBoost models in our experiments).

Robustness-enhancing methods: For robustness-enhancing methods, our hyperparameter grids combine our large default MLP grid with the hyperparameter grids for method-specific parameters used in previous works (e.g. [[74](#bib.bib74), [46](#bib.bib46)]. We use the implementation of [[74](#bib.bib74)] for DORO and [[46](#bib.bib46)] for DRO methods. Note that we do not conduct sweeps for DORO models on the Adult and ACS datasets due to computational limitations, as full sweeps for both methods are prohibitively expensive to run on these large datasets.

Fairness methods: We use standard implementations and the largest-possible grids while meeting our computational constraints, as many of the fairness methods scale worse than linearly with dataset size, feature size, or both. For LFR, we fix Ax as in [[73](#bib.bib73)], and otherwise use the center portion of the grid from [[20](#bib.bib20)] which we found to be sufficient for our sweeps when also tuning the GBM parameters (which was not performed in [[20](#bib.bib20)]) across our datasets. Our grid for inprocessing uses the same constraints explored in [[20](#bib.bib20)] but also tunes the constraint slack and GBM parameters, which were not tuned in [[20](#bib.bib20)]. We use the implementation of [[8](#bib.bib8)] for LFR and postprocessing, and [[12](#bib.bib12)] for inprocessing.

Model
Grid Size
Hyperparameter
Values

Baseline Methods

♣♣\clubsuit MLP
405
Learning Rate
{1​e−1,1​e−2,1​e−3,1​e−4,1​e−5}1superscript𝑒11superscript𝑒21superscript𝑒31superscript𝑒41superscript𝑒5\{1e^{-1},1e^{-2},1e^{-3},1e^{-4},1e^{-5}\}

Weight Decay
{0,0.1,1}00.11\{0,0.1,1\}

Num. Layers
{1,2,3}123\{1,2,3\}

Hidden Units
{64,128,256}64128256\{64,128,256\}

Momentum
{0.,0.1,0.9}\{0.,0.1,0.9\}

Batch Size
{128}128\{128\}

SVM
576
C
{0.01,0.1,1.,10.,100.,1000.}\{0.01,0.1,1.,10.,100.,1000.\}

Kernel Appx.
{Nystroem,RKS}NystroemRKS\{\textrm{Nystroem},\textrm{RKS}\}

Loss
Squared Hinge

γ𝛾\gamma
{0.5,1.0,2.0}0.51.02.0\{0.5,1.0,2.0\}

Num. Components
{64,128,256,512}64128256512\{64,128,256,512\}

Nystroem Kernel Degree
{2,3}23\{2,3\}

Nystroem Kernel
{RBF,poly}RBFpoly\{\textrm{RBF},\textrm{poly}\}

Logistic Regression
8
L2subscript𝐿2L\_{2} penalty
{0.001,0.01,0.1,1.,10.,100.,1000.,10000.}\{0.001,0.01,0.1,1.,10.,100.,1000.,10000.\}

Tree-Based Methods

♢♢\diamondsuit GBM
100
Learning Rate
{0.01,0.1,0.5,1.0,2.0}0.010.10.51.02.0\{0.01,0.1,0.5,1.0,2.0\}

Num. Estimators
{64,128,256,512,1024}641282565121024\{64,128,256,512,1024\}

Max Depth
{2,4,8,16}24816\{2,4,8,16\}

Min. Samples Split
2

Min. Samples Leaf
1

Random Forest
640
Num. Estimators
{64,128,256,512}64128256512\{64,128,256,512\}

Max Features
{sqrt,log2}sqrtlog2\{\textrm{sqrt},\textrm{log2}\}

Min. Samples Split
{2,4,8,16}24816\{2,4,8,16\}

Min. Samples Leaf
{1,2,4,8,16}124816\{1,2,4,8,16\}



Cost-Complexity α𝛼\alpha
{0.,0.001,0.01,0.1}\{0.,0.001,0.01,0.1\}

XGBoost
1944
Learning Rate
{0.1,0.3,1.0,2.0}0.10.31.02.0\{0.1,0.3,1.0,2.0\}

Min. Split Loss
{0,0.1,0.5}00.10.5\{0,0.1,0.5\}

Max. Depth
{4,6,8}468\{4,6,8\}

Column Subsample Ratio (tree)
{0.7,0.9,1}0.70.91\{0.7,0.9,1\}

Column Subsample Ratio (level)
{0.7,0.9,1}0.70.91\{0.7,0.9,1\}

Max. Bins
{128,256,512}128256512\{128,256,512\}

Growth Policy
{Depthwise,Loss Guide}DepthwiseLoss Guide\{\textrm{Depthwise},\textrm{Loss Guide}\}

LightGBM
12544
Learning Rate
{0.01,0.1,0.5,1.}\{0.01,0.1,0.5,1.\}

Num. Estimators
{64,128,256,512}64128256512\{64,128,256,512\}

L2subscript𝐿2L\_{2}-reg.
{0.,0.00001,0.0001,0.001,0.01,0.1,1.}\{0.,0.00001,0.0001,0.001,0.01,0.1,1.\}

Min. Child Samples
{1,2,4,8,16,32,64}1248163264\{1,2,4,8,16,32,64\}

Max. Depth
{None,2,4,8}None248\{\textrm{None},2,4,8\}

Column Subsample Ratio (tree)
{0.4,0.5,0.8,1.}\{0.4,0.5,0.8,1.\}

Robustness-Enhancing Methods

DORO χ2superscript𝜒2\chi^{2} ♣♣\clubsuit
12150
Uncertainty set size α𝛼\alpha
{0.1,0.2,0.3,0.4,0.5,0.6}0.10.20.30.40.50.6\{0.1,0.2,0.3,0.4,0.5,0.6\}

Outlier proportion ϵitalic-ϵ\epsilon
{0.001,0.01,0.1,0.2,0.3}0.0010.010.10.20.3\{0.001,0.01,0.1,0.2,0.3\}

DORO CVaR ♣♣\clubsuit
12150
Uncertainty set size α𝛼\alpha
{0.1,0.2,0.3,0.4,0.5,0.6}0.10.20.30.40.50.6\{0.1,0.2,0.3,0.4,0.5,0.6\}

Outlier proportion ϵitalic-ϵ\epsilon
{0.001,0.01,0.1,0.2,0.3}0.0010.010.10.20.3\{0.001,0.01,0.1,0.2,0.3\}

DRO χ2superscript𝜒2\chi^{2} ♣♣\clubsuit
2835
Uncertainty set size α𝛼\alpha
{0.01,0.1,0.2,0.3,0.4,0.5,0.6}0.010.10.20.30.40.50.6\{0.01,0.1,0.2,0.3,0.4,0.5,0.6\}

DRO CVaR ♣♣\clubsuit
2835
Uncertainty set size α𝛼\alpha
{0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6}0.0010.010.10.20.30.40.50.6\{0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6\}

Group DRO ♣♣\clubsuit
1620
Group weights step size
{0.001,0.01,0.1,0.2}0.0010.010.10.2\{0.001,0.01,0.1,0.2\}

MWLD ♣♣\clubsuit
6075
L2subscript𝐿2L\_{2} penalty
{0,0.1,1}00.11\{0,0.1,1\}

Loss variance penalty
{1e−3,1e−2,1e−1,1,10,}\{1e^{-3},1e^{-2},1e^{-1},1,10,\}

Fairness-Enhancing Methods

Preprocessing ♢♢\diamondsuit
2500
Ax
0.010.010.01

Ay
0.001,0.01,0.1,1,10

0.0010.010.11100.001,0.01,0.1,1,10



Az
0.001,0.01,0.1,1,10

0.0010.010.11100.001,0.01,0.1,1,10

Inprocessing ♢♢\diamondsuit
1000
Constraint slack ϵitalic-ϵ\epsilon
{1​e−4,1​e−3,1​e−2,1​e−1,1}1superscript𝑒41superscript𝑒31superscript𝑒21superscript𝑒11\{1e^{-4},1e^{-3},1e^{-2},1e^{-1},1\}

Constraint Type
{DP,EO}DPEO\{\textrm{DP},\textrm{EO}\}



Max Iter.
200

Postprocessing ♢♢\diamondsuit
100
No tunable hyperparameters

Table 2: Hyperparameter grids used in all experiments. ♣♣\clubsuit: all MLP parameters also tuned. ♢♢\diamondsuit: all GBM parameters also tuned.

### F.2 Hyperparameter Sensitivity Analysis

This section provides exploratory results regarding the sensitivity of the models evaluated to hyperparameter configurations. For each model, we take the full set of hyperparameter configurations evaluated. then for each hyperparameter, we compute the set of values of the best-performing model (here, using worst-group accuracy) over all datasets, dropping values from continuous hyperparameter grids outside this range. This truncation step eliminates ranges of each hyperparameter which performed poorly for *all* datasets. Finally, we order the remaining hyperparameter configurations by worst-group accuracy to construct the plots below.

In the top panel of Figure [8](#A6.F8 "Figure 8 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), we show only the DRO χ2superscript𝜒2\chi^{2}, XGBoost, and Group DRO models, following our running example in the main text. We include 95% Clopper-Pearson confidence intervals using the *smallest* sensitive subgroup as the sample size for the CI.131313This makes the confidence intervals in [8](#A6.F8 "Figure 8 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") conservative, as they would be narrower when the worst group is not the smallest group; we do this so that the interval width is consistent for all model configurations. In the bottom panel of Figure [8](#A6.F8 "Figure 8 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), scale the results according to the *best* performance achieved by each model class on the target dataset.

We provide similar results in the top and bottom rows of Figure [9](#A6.F9 "Figure 9 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") which include all models; here we omit the confidence intervals due to space (although the intervals would have the same width as in Figure [8](#A6.F8 "Figure 8 ‣ F.2 Hyperparameter Sensitivity Analysis ‣ Appendix F Hyperparameter Grids ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")).

![Refer to caption](/html/2211.12703/assets/img/hparam_ranked_plot_accuracy_test_acsincomeacspubcovadultbrfsscommunities-and-crimecompasgermanlarc-grade.png)

![Refer to caption](/html/2211.12703/assets/img/hparam_ranked_plot_accuracy_worstgroup_test_acsincomeacspubcovadultbrfsscommunities-and-crimecompasgermanlarc-grade.png)

Figure 8: Hyperparameter sensitivity plots for χ2superscript𝜒2\chi^{2} DRO, Group DRO, and XGBoost models. The top 8 panels show Accuracy; the lower 8 panels show worst-group accuracy (this is identical to Figure [7](#S6.F7 "Figure 7 ‣ 6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), reproduced here for clarity). XGBoost shows considerably lower sensitivity to hyperparameter tuning.



![Refer to caption](/html/2211.12703/assets/img/hparam_relative_ranked_plot_accuracy_test.png)

![Refer to caption](/html/2211.12703/assets/img/hparam_relative_ranked_plot_accuracy_worstgroup_test.png)

Figure 9: Hyperparameter sensitivity plots for all models evaluated. (Clopper-Pearson CIs omitted due to space).

## Appendix G Additional Results

This section contains additional experimental results not included in the main text, along with fine-grained displays of the results summarized in the main figures.

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_acsincome_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_acspubcov_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_adult_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_brfss_grouped.png)

Figure 10: Overall Accuracy vs. Worst-Group Accuracy of robust, fairness-enhancing, tree-based, and baseline models over eight tabular datasets (ACS Income, ACS Public Coverage, Adult, BRFSS). For results by individual algorithm, see Figures [12](#A7.F12 "Figure 12 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")- [15](#A7.F15 "Figure 15 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_communities-and-crime_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_compas_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_german_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_larc-grade_grouped.png)

Figure 11: Overall Accuracy vs. Worst-Group Accuracy of robust, fairness-enhancing, tree-based, and baseline models over eight tabular datasets (Communities and Crime, COMPAS, German Credit, LARC). For results by individual algorithm, see Figures [12](#A7.F12 "Figure 12 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")- [15](#A7.F15 "Figure 15 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_acsincome_alg.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_acspubcov_alg.png)

Figure 12: Detailed accuracy vs. worst-group accuracy for ACS Income and ACS Public Coverage datasets.



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_adult_alg.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_brfss_alg.png)

Figure 13: Detailed accuracy vs. worst-group accuracy for Adult and BRFSS datasets.



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_communities-and-crime_alg.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_compas_alg.png)

Figure 14: Detailed accuracy vs. worst-group accuracy for Communities and Crime and COMPAS datasets.



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_german_alg.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_accuracy_worstgroup_test_larc-grade_alg.png)

Figure 15: Detailed accuracy vs. worst-group accuracy for German and LARC datasets.



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_acsincome_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_acspubcov_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_adult_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_brfss_grouped.png)

Figure 16: Overall Accuracy vs. Accuracy Disparity of robust, fairness-enhancing, tree-based, and baseline models over datasets ACS Income, ACS Public Coverage, Adult, BRFSS. See also Figure [17](#A7.F17 "Figure 17 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").



![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_communities-and-crime_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_compas_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_german_grouped.png)

![Refer to caption](/html/2211.12703/assets/img/accuracy_test_vs_abs_accuracy_disparity_test_larc-grade_grouped.png)

Figure 17: Overall Accuracy vs. Accuracy Disparity of robust, fairness-enhancing, tree-based, and baseline models over datasets Communities and Crime, COMPAS, German, LARC. See also Figure [16](#A7.F16 "Figure 16 ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").



![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_acsincome_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_acspubcov_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_adult_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_brfss_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_communities-and-crime_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_compas_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_german_alg.png)

![Refer to caption](/html/2211.12703/assets/img/metrics_scatter_larc-grade_alg.png)

Figure 18: Correlation between complementary metrics (top row) and non-complementary metrics (bottom row) for each dataset, for DORO, XGBoost, and Group DRO models. Complementary metrics show strong correlations for all models, while non-complementary metrics do not. Pearson’s r𝑟r correlation coefficient for each pair of complementary metrics shown in the upper-left of each plot.

### G.1 Training Compute Cost

In Section [6](#S6 "6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), we discuss hyperparameter sensitivity and training time of the various algrorithms present in our study. Here, we provide estimations of the result of conducting our hyperparameter sweeps on modern cloud-based computing platforms, in order to estimate the costs of individual training runs, and the full hyperparameter sweeps, in our study.

![Refer to caption](/html/2211.12703/assets/img/est_cost_per_train_run.png)


Figure 19: Estimated cost per training run, based on the median train time over the iterations in our study and the price of cloud-based computing infrastructure.

Figure [19](#A7.F19 "Figure 19 ‣ G.1 Training Compute Cost ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") displays the estimated median cost of a single training run of each model in (DRO χ2superscript𝜒2\chi^{2}, XGBoost, Group DRO, LightGBM). We note that while XGBoost and LightGBM are trained on CPU in this study (although GPU implementations of each training algorithm are available), training of the DRO models at scale is only feasible on GPU.

For our cost estimation, we use prices of $7.20currency-dollar7.20\$7.20 per compute-hour for GPU and $3.072currency-dollar3.072\$3.072 per compute-hour for CPU, which reflect the hourly price of cloud-based GPU and CPU hardware used in this study.

Figure [19](#A7.F19 "Figure 19 ‣ G.1 Training Compute Cost ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") shows that, compared to DRO methods, tree-based methods (XGBoost, LightGBM) achieve comparable cost or considerable savings for all datasets in our study. The lone exception is XGBoost on the German Credit dataset, which we hypothesize is due to potential overloading of the CPU cluster during these training experiments (we note that German is, by far, the smallest dataset in our study with n=1000𝑛1000n=1000, and the XGBoost algorithm generally performs well on small datasets).

Collectively, these results, combined with the demonstration that tree-based models also require fewer iterations to tune due to their decreased hyperparameter sensitivity (see Section [6](#S6 "6 Results: Trees Show Lower Hyperparameter Sensitivity and are Less Costly to Train ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation")), suggest that tree-based models are a considerably more resource-efficient way to achieve state-of-the-art subgroup robustness for tabular data classification.

### G.2 Peak Performance Summary

We provide the best performance per model, in terms of both overall accuracy and worst-group accuracy, in Tables [3](#A7.T3 "Table 3 ‣ G.2 Peak Performance Summary ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation") and [4](#A7.T4 "Table 4 ‣ G.2 Peak Performance Summary ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation"), respectively.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Income | Pub. Cov. | Adult | BRFSS | C&C | COMPAS | German | LARC |
| DORO CVaR | 0.816 | N/A | 0.852 | N/A | 0.905 | 0.725 | 0.86 | N/A |
| DORO χ2superscript𝜒2\chi^{2} | 0.813 | N/A | 0.851 | N/A | N/A | 0.743 | N/A | N/A |
| DRO CVaR | 0.677 | 0.719 | 0.778 | 0.898 | 0.879 | 0.627 | 0.83 | 0.644 |
| DRO χ2superscript𝜒2\chi^{2} | 0.694 | 0.742 | 0.774 | 0.896 | 0.869 | 0.691 | 0.82 | 0.667 |
| GBM | 0.824 | 0.786 | 0.873 | 0.896 | 0.854 | 0.712 | 0.82 | 0.747 |
| Group DRO | 0.816 | 0.77 | 0.85 | 0.897 | 0.894 | 0.702 | 0.82 | 0.729 |
| Inprocessing + GBM | 0.823 | 0.789 | 0.87 | 0.898 | 0.879 | 0.732 | 0.87 | 0.75 |
| L2LR | 0.815 | 0.768 | 0.852 | 0.895 | 0.844 | 0.7 | 0.82 | 0.739 |
| LightGBM | 0.827 | 0.791 | 0.874 | 0.901 | 0.91 | 0.734 | 0.9 | 0.751 |
| MLP | 0.821 | 0.782 | 0.857 | 0.897 | 0.879 | 0.724 | 0.82 | 0.743 |
| MWLD | 0.823 | 0.784 | 0.864 | 0.898 | 0.894 | 0.739 | 0.85 | 0.744 |
| Marginal DRO | N/A | N/A | N/A | N/A | 0.849 | 0.72 | 0.82 | N/A |
| Postprocessing + GBM | 0.775 | 0.772 | 0.842 | 0.897 | 0.749 | 0.689 | 0.85 | 0.714 |
| Preprocesing + GBM | 0.771 | 0.765 | 0.827 | 0.897 | 0.879 | 0.724 | 0.86 | 0.678 |
| Random forest | 0.82 | 0.786 | 0.865 | 0.897 | 0.905 | 0.728 | 0.85 | 0.746 |
| SVM | 0.821 | 0.785 | 0.857 | 0.898 | 0.874 | 0.723 | 0.83 | 0.74 |
| XGBoost | 0.824 | 0.79 | 0.875 | 0.899 | 0.894 | 0.725 | 0.88 | 0.748 |

Table 3: Best observed overall accuracy per model, by dataset.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Income | Pub. Cov. | Adult | BRFSS | C&C | COMPAS | German | LARC |
| DORO CVaR | 0.788 | N/A | 0.81 | N/A | 0.836 | 0.71 | 0.8 | N/A |
| DORO χ2superscript𝜒2\chi^{2} | 0.783 | N/A | 0.808 | N/A | N/A | 0.718 | N/A | N/A |
| DRO CVaR | 0.582 | 0.672 | 0.712 | 0.857 | 0.803 | 0.606 | 0.762 | 0.492 |
| DRO χ2superscript𝜒2\chi^{2} | 0.641 | 0.683 | 0.707 | 0.863 | 0.789 | 0.676 | 0.789 | 0.61 |
| GBM | 0.796 | 0.738 | 0.833 | 0.851 | 0.787 | 0.684 | 0.737 | 0.734 |
| Group DRO | 0.783 | 0.718 | 0.807 | 0.855 | 0.816 | 0.682 | 0.789 | 0.702 |
| Inprocessing + GBM | 0.796 | 0.742 | 0.845 | 0.859 | 0.803 | 0.702 | 0.842 | 0.735 |
| L2LR | 0.785 | 0.718 | 0.808 | 0.849 | 0.754 | 0.644 | 0.727 | 0.72 |
| LightGBM | 0.798 | 0.745 | 0.837 | 0.87 | 0.836 | 0.718 | 0.842 | 0.739 |
| MLP | 0.791 | 0.738 | 0.814 | 0.854 | 0.787 | 0.715 | 0.75 | 0.721 |
| MWLD | 0.794 | 0.739 | 0.826 | 0.859 | 0.82 | 0.729 | 0.774 | 0.728 |
| Marginal DRO | N/A | N/A | N/A | N/A | 0.77 | 0.707 | 0.774 | N/A |
| Postprocessing + GBM | 0.72 | 0.723 | 0.809 | 0.857 | 0.639 | 0.623 | 0.75 | 0.632 |
| Preprocesing + GBM | 0.737 | 0.717 | 0.781 | 0.861 | 0.803 | 0.702 | 0.816 | 0.647 |
| Random forest | 0.79 | 0.741 | 0.824 | 0.858 | 0.836 | 0.702 | 0.8 | 0.726 |
| SVM | 0.791 | 0.735 | 0.815 | 0.858 | 0.763 | 0.707 | 0.789 | 0.719 |
| XGBoost | 0.796 | 0.744 | 0.836 | 0.868 | 0.836 | 0.711 | 0.818 | 0.736 |

Table 4: Best observed worst-group accuracy per model, by dataset.

### G.3 Performance of Default Tree Hyperparameters

For the tree-based models in our summary, we give the performance of the default hyperparameters in Table [5](#A7.T5 "Table 5 ‣ G.3 Performance of Default Tree Hyperparameters ‣ Appendix G Additional Results ‣ Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation").

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Income | Pub. Cov. | Adult | BRFSS | C&C | COMPAS | German | LARC |
| Overall Accuracy | | | | | | | | |
| GBM | 0.81 | 0.777 | 0.871 | 0.892 | 0.859 | 0.7 | 0.79 | 0.739 |
| LightGBM | 0.823 | 0.787 | 0.874 | 0.887 | 0.819 | 0.682 | 0.75 | 0.747 |
| Random Forest | 0.812 | 0.765 | 0.852 | 0.891 | 0.854 | 0.669 | 0.74 | 0.754 |
| XGBoost | 0.825 | 0.788 | 0.872 | 0.893 | 0.854 | 0.698 | 0.73 | 0.748 |
| Worst-Group Accuracy | | | | | | | | |
| GBM | 0.779 | 0.725 | 0.83 | 0.86 | 0.711 | 0.684 | 0.65 | 0.718 |
| LightGBM | 0.793 | 0.738 | 0.834 | 0.834 | 0.658 | 0.667 | 0.6 | 0.725 |
| Random Forest | 0.781 | 0.713 | 0.808 | 0.848 | 0.738 | 0.596 | 0.71 | 0.742 |
| XGBoost | 0.797 | 0.735 | 0.834 | 0.833 | 0.754 | 0.654 | 0.55 | 0.728 |

Table 5: Performance of default hyperparameters for tree-based models.

[◄](/html/2211.12702)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2211.12703)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2211.12703)
[View original  
on arXiv](https://arxiv.org/abs/2211.12703)[►](/html/2211.12704)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Mar 14 05:45:49 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
