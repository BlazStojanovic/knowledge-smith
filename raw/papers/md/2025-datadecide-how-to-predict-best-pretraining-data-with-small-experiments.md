---
arxiv: '2504.11393'
authors:
- Ian Magnusson
- Nguyen Tai
- Ben Bogin
- David Heineman
- Jena D. Hwang
- Luca Soldaini
- Akshita Bhagia
- Jiacheng Liu
- Dirk Groeneveld
- Oyvind Tafjord
- Noah A. Smith
- Pang Wei Koh
- Jesse Dodge
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'DataDecide: How to Predict Best Pretraining Data with Small Experiments'
url: https://arxiv.org/abs/2504.11393
year: 2025
---

[2504.11393] How to Predict Best Pretraining Data with Small Experiments














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



# [Uncaptioned image] How to Predict Best Pretraining Data with Small Experiments

Ian Magnusson∗§‡,
Nguyen Tai∗∥,
Ben Bogin∗§,
David Heineman§,
Jena Hwang§,

Luca Soldaini§,
Akshita Bhagia§,
Jiacheng Liu§‡,
Dirk Groeneveld§,

Oyvind Tafjord§,
Noah A. Smith§‡,
Pang Wei Koh§‡,
Jesse Dodge§

§ Allen Institute for AI  ‡ University of Washington  ∥ University of Pennsylvania  ∗ equal contribution

###### Abstract

Because large language models are expensive to pretrain on different datasets, using smaller-scale experiments to decide on data is crucial for reducing costs. Which benchmarks and methods of making decisions from observed performance at small scale most accurately predict the datasets that yield the best large models? To empower open exploration of this question, we release models, data, and evaluations in DataDecide—the most extensive open suite of models over differences in data and scale.
We conduct controlled pretraining experiments across 25 corpora with differing sources, deduplication, and filtering up to 100B tokens, model sizes up to 1B parameters, and 3 random seeds.
We find that the ranking of models at a single, small size (e.g., 150M parameters) is a strong baseline for predicting best models at our larger target scale (1B) (∼80\sim 80% of comparisons correct).
No scaling law methods among 8 baselines exceed the compute-decision frontier of single-scale predictions, but DataDecide can measure improvement in future scaling laws. We also identify that using continuous likelihood metrics as proxies in small experiments makes benchmarks including MMLU, ARC, HellaSwag, MBPP, and HumanEval >80>80% predictable at the target 1B scale with just 0.01% of the compute.

![Refer to caption](/html/2504.11393/assets/x2.png)


Figure 1: Which pretraining data to use? Ideally, compare performance of large models with fixed configurations averaged over random seeds (left). In practice, cheaper, smaller-scale experiments are used (center).
Here DataDecide measures accuracy of pairwise decisions between 25 pretraining corpora to find efficient prediction methods (right).

## 1 Introduction

The cost of training large language models (LMs) necessitates methods of trying out options at small scale, but it also makes it expensive to validate the accuracy of development decisions made with such methods. We focus on the question of choosing between pretraining datasets to use—one of the most impactful development decisions.
Common practice (e.g., Li et al., [2024](#bib.bib24)) uses a single, small scale of experiments to cheaply test pretraining data intended for larger-scale models, where scale is determined by number of model parameters and training tokens. The other predominant approach is to fit scaling laws (Kaplan et al., [2020](#bib.bib21); Hoffmann et al., [2022](#bib.bib19); Choshen et al., [2024](#bib.bib8)) to the trend in performance observed over multiple small scales, with recent work extending this to the prediction of downstream performance instead of language modeling loss (Gadre et al., [2024](#bib.bib14); Dubey et al., [2024](#bib.bib13); Bhagia et al., [2024](#bib.bib3)).

So far decision-making approaches have only been validated without observing the counterfactual outcome, either by producing a single large model on the chosen decision with impressive performance or by low error in predicting the magnitude of observed performance of a small number of large models. Knowing what amount of error in predicting performance over scale is a low enough to actually make a correct decision among datasets, requires a suite of comparable models trained on many datasets. Although a wide variety of open-source pretraining corpora are available, the scaling behavior of data is difficult to assess from off-the-shelf models that vary simultaneously in data, optimizer, and modeling decisions.

To make it possible to empirically study what methods make the best decisions over data, we build DataDecide111[DataDecide collection on HuggingFace](https://huggingface.co/collections/allenai/datadecide-67edb1d2bacba40b5d3ed633)—a suite of models we pretrain on 25 corpora up to 100B tokens, over 14 different model sizes ranging from 4M parameters up to 1B parameters (more than 30K model checkpoints in total). We evaluate all models across a suite of 10 downstream tasks and calculate how accurately small models predict which pretraining corpora lead to better performance at our largest scale. Our conclusions provide practical recommendations for the best benchmarks, prediction methods, and metrics to use to make decisions.

We call the 25 corpora we train on data recipes as they range across popular corpora including Dolma (Soldaini et al., [2024](#bib.bib38)), DCLM (Li et al., [2024](#bib.bib24)), RefinedWeb (Penedo et al., [2023](#bib.bib29)), C4 (Raffel et al., [2019](#bib.bib32)), and FineWeb (Penedo et al., [2024](#bib.bib30)) as well as combinations of interventions on these datasets such as source mixing, deduplication, and filtering. Previous work has considered only 2 (Biderman et al., [2023](#bib.bib4)) or 6 recipes (Magnusson et al., [2024](#bib.bib25); Brandfonbrener et al., [2024](#bib.bib6)).
We also offer a novel affordance by including 3 random seed reruns for even our largest runs, to help quantify whether variation occurs due to random initialization and data order or differences in the distribution of data.

Concretely, DataDecide allows analyses such as Figure [1](#S0.F1 "Figure 1 ‣ How to Predict Best Pretraining Data with Small Experiments") (right), which shows the relationship between compute used to predict a ranking of datasets and how accurately that ranking reflects mean performance over 3 seed runs (quantified here by OLMES; Gu et al., [2024](#bib.bib17)) for models fully trained on those datasets at the target (1B) scale. We measure the accuracy of decisions as the percent of compared pairs of datasets where the prediction identifies the correct winner. Each point represents the average decision accuracy of a given method over 3 prediction attempts using small models with different random seeds, and shading shows standard deviation.

Measuring the tradeoff of compute cost to better decisions lets us make the following recommendations about small experiments for making data decisions:

* •

  §[3.1](#S3.SS1 "3.1 What is the best way to spend compute for data decisions? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") – The amount of compute you need to allocate for a given decision accuracy depends heavily on task. MMLU and ARC are much cheaper to predict than HellaSwag and some tasks such as SocialIQA are difficult to predict at all scales.
* •

  §[3.2](#S3.SS2 "3.2 How does extrapolating scaling laws compare to ranking single scale experiments? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") – 8 baseline scaling law methods do not exceed the compute to decision accuracy frontier set by ranking single scale experiments.
* •

  §[3.3](#S3.SS3 "3.3 What proxy metrics give better signal for predictions at small scale? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") – At small scales, continuous metrics using answer likelihood are better or equivalent predictors of decisions than using the same discrete accuracy target metric.
* •

  §[3.4](#S3.SS4 "3.4 How can we make evaluation benchmarks more predictable? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") – Better decisions can be explained in part by low run-to-run variance and a wide spread of benchmark performance values for different data, traits which can be improved by proxy metrics.

Future research can extend DataDecide with little extra compute by running new evaluations on our checkpoints, pretraining additional small models to compare against the large target models we provide, or trying new prediction methods with lightweight manipulations such as smoothing and curve fitting on top of our released evaluation results.

## 2 Methods

Our aim is to empirically test the predictability of downstream performance at a larger, target scale using small experiments. We describe DataDecide §[2.1](#S2.SS1 "2.1 The DataDecide Suite ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), the prediction methods we examine §[2.2](#S2.SS2 "2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), the metrics we use to assess predictions §[2.3](#S2.SS3 "2.3 Prediction Metrics ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), how we measure downstream performance §[2.4](#S2.SS4 "2.4 Performance Evaluation with OLMES ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), and proxy metrics for our performance evaluations §[2.5](#S2.SS5 "2.5 Proxy Metrics for Performance Evaluation ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments").
We will release all models, checkpoints, pretraining corpora, and evaluations.

|  |  |
| --- | --- |
| Source / Recipe | Description |
| Dolma1.7 *Original, No code, No math/code, No Reddit, No Flan* | A 2.3T-token corpus (Dolma 1.7 Soldaini et al., [2024](#bib.bib38)) sampling common LM sources for open research. We ablate code, math/code, Reddit, or Flan subsets. |
| Dolma1.6++ *Original* | Dolma 1.6 plus additional sources from Dolma 1.7: RedPajama’s arxiv subset, openwebmath, algebraic stack, flan, starcoder, falcon. |
| C4 *Original* | The C4 dataset (Raffel et al., [2019](#bib.bib32)) as prepared in Dolma 1.7, heuristically filtered from the April 2019 Common Crawl. |
| FineWeb-Pro *Original* | The FineWeb Pro corpus (Zhou et al., [2024](#bib.bib42)), featuring model-driven data cleaning on FineWeb. |
| FineWeb-Edu *Original* | The deduplicated FineWeb-Edu subset of SmolLM-Corpus (Ben Allal et al., [2024](#bib.bib2)), focused on educational web pages. |
| Falcon *Original* | The Falcon RefinedWeb corpus (Penedo et al., [2023](#bib.bib29)) in Dolma 1.7, derived from Common Crawl through June 2023 and more aggressively filtered/deduplicated than C4. |
| Falcon+CC *Original, QC 10%, QC 20%, QC Orig 10%, QC Tulu 10%* | Falcon and Dolma 1.7’s Common Crawl. We quality filter to top 10% or 20% documents with reproduced or original Li et al. ([2024](#bib.bib24)) filter or retrain filter on pre-release version of Tulu-v3 (Lambert et al., [2024](#bib.bib22)). |
| DCLM-Baseline *Original, QC 7% FW2, QC 7% FW3, QC FW 3%, QC FW 10%, QC 10%, QC 20%* | A SOTA Common Crawl corpus using best ablated deduplication, cleaning heuristics, and quality filter. We quality filter to top 7% of DCLM classified documents and further take 2+ or 3+ scores with FineWeb-edu classifier; or filter to top 3% or 10% with FineWeb-edu classifier; or take top 10% or 20% with reproduced DCLM classifier. |
| *λ\lambda%* DCLM-Baseline + *1−λ1-\lambda%* Dolma1.7 | Fractional combinations of Dolma1.7 and DCLM-Baseline mixing different proportions of the two datasets for λ∈{25%,50%,75%}\lambda\in\{25\%,50\%,75\%\}. |

Table 1: DataDecide enables the study of data differences over scales through controlled pretraining experiments on 25 data recipes. These take different source datasets and apply interventions from ablating domains, deduplication, mixing, to quality filtering with different classifiers and thresholds. We release all pretraining corpora, as well as models trained on each recipe and each of the 14 model configurations in Table [2](#A1.T2 "Table 2 ‣ Appendix A Hyperparameters ‣ How to Predict Best Pretraining Data with Small Experiments") with 3 random seeds.

### 2.1 The DataDecide Suite

We pretrain a suite of 1,050 models using 25 data recipes ×\times 14 model scales ×\times 3 random seeds for initialization and data order. Table [1](#S2.T1 "Table 1 ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments") describes the 25 data recipes included in DataDecide that aim to provide coverage of common data preparation choices such as deduplication, ablating domains, mixes of existing datasets, as well as quality filters with different implementations, training data, and thresholds for quality classifiers.

We select a token to parameter ratio of 100, which at 5×\times “Chinchilla” (5 ×C\times~C) optimal ratio (Hoffmann et al., [2022](#bib.bib19)) captures the typical overtraining favored for inference savings.

All 1B (target size) models have 3 full reruns with different seeds, while other model sizes have second and third seed runs that are terminated early after 25%25\% of the target compute budget. We train the 1B reruns all the way to completion to allow our target “gold” predictions to account for run-to-run variance in evaluations due to weight initialization and data order. For instance, we find that the standard deviation between runs at the 1B 5×C\times C scale can be as high as 2%2\% points of accuracy for some recipes on most tasks. Meanwhile, at the non-target scales we wish to make predictions with a small fraction of the target compute, so we avoid reruns that would use an impractically large prediction budget.

Whether for extrapolating scaling laws or ranking single scale experiments, it is important to select reasonable hyperparameters for each scale to avoid confounding in performance differences that are simply due to suboptimal hyperparameters. We use OLMo’s model ladder (Groeneveld et al., [2024](#bib.bib16); OLMo et al., [2025](#bib.bib28); Bhagia et al., [2024](#bib.bib3)) to programmatically create LM pretraining configurations for a specified parameter size and token-parameter ratio to enable running a grid of model scaling experiments.
The model ladder uses heuristics from the literature (Porian et al., [2024](#bib.bib31)) to set global batch size and learning rate based on scaling factors. The hyperparameters that determine parameter count (layers, hidden dimension, number of heads, MLP dimension) were handpicked by OLMo developers for each scale to achieve the desired number of parameters. Appendix Table [2](#A1.T2 "Table 2 ‣ Appendix A Hyperparameters ‣ How to Predict Best Pretraining Data with Small Experiments") details the configurations of all our models.

### 2.2 Prediction Methods

Broadly, there are two approaches in the literature to predicting large-scale performance based on small-scale experiments. We use straightforward implementations of each to assess where they succeed and fail at making decisions about which data recipes to use.

#### Ranking Single Scale Experiments (Single Scale)

This simple approach is employed by work such as Li et al. ([2024](#bib.bib24)) and consists of running a set of ablations or experiments over data recipe options while holding constant all other modeling variables including scale. The winning data recipe by downstream accuracy (or proxies) at the small experimental scale is assumed to extrapolate to the target scale.

#### Extrapolating Scaling Laws (Multi Scale)

Another approach to making decisions with predictions across scales used in works such as Dubey et al. ([2024](#bib.bib13)) is to fit scaling laws to multiple small experiments across a range of scales for each of the data recipes. The winning recipe is decided as the one whose scaling law shows the highest extrapolated performance at the target scale. Although scaling laws were first observed for language modeling loss (Kaplan et al., [2020](#bib.bib21); Hoffmann et al., [2022](#bib.bib19)), they have been extended to predict downstream performance through a two-step approach that also fits a function from loss to downstream performance (Gadre et al., [2024](#bib.bib14); Bhagia et al., [2024](#bib.bib3)). We follow a method from Bhagia et al. ([2024](#bib.bib3)). Their proposed approach incorporates separate parameters for number of model parameters and number of tokens trained to account for over or undertrained models. But as our suite only includes one token-parameter ratio, we use the simplified 3 parameter baseline, L​(C)L(C), as a first step which we chain with second step, A​c​c​(L)Acc(L), defined as follows where AA, α\alpha, EE, aa, bb, kk, L0L\_{0} are optimized parameters:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | L​(C)\displaystyle L(C) | =ACα+E\displaystyle=\frac{A}{C^{\alpha}}+E |  | (1) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | A​c​c​(L)\displaystyle Acc(L) | =a1+e−k​(L−L0)+b\displaystyle=\frac{a}{1+e^{-k(L-L\_{0})}}+b |  | (2) |

Following Bhagia et al. ([2024](#bib.bib3)) we fit Equation [1](#S2.E1 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments") only on observations of final, fully trained checkpoints as accounting for the learning rate schedule’s impact on intermediate checkpoints would require further parameters in the equation increasing the required number of observations and cost. To account for step-to-step noise in evaluation we average the last 10%10\% of checkpoints as the final observed loss. Equation [2](#S2.E2 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), however, is fit on all observations including intermediate checkpoints.
We explore variations for a total of 8 multi scale approaches defined in Appendix [C](#A3 "Appendix C Scaling Law Variants ‣ How to Predict Best Pretraining Data with Small Experiments"); none of these make for substantially better decisions than the method defined in this section.

### 2.3 Prediction Metrics

Our predictive task is to forecast which of a pair of data recipes will perform better at some target scale based on small-scale experiments. We use the following metrics to measure the quality of these predictions.

#### Prediction Error

Scaling laws literature (Bhagia et al., [2024](#bib.bib3); Gadre et al., [2024](#bib.bib14)) typically evaluates success from predicted and actual downstream performance, using relative error (|predicted−actual|actual×100%\frac{\lvert\text{predicted}-\text{actual}\rvert}{\text{actual}}\times 100\%) or absolute error (|predicted−actual|×100%\lvert\text{predicted}-\text{actual}\rvert\times 100\%). We call these absolute or relative “prediction error” to distinguish from the following metric.

#### Decision Accuracy

Unlike previous work, we also measure the impact of predictions on decisions about which data recipe is better than another. The metric we use to capture this is decision accuracy, an accuracy over all pairs of data recipes AA and BB where either AA or BB is defined as the correct winner based on which achieves higher performance at the target scale. This is nearly equivalent to Kendall’s τ\tau, but ranges from 0 to 1. We define the target-scale winner based on mean downstream performance over 3 random seeds.
Thus decision accuracy can be formalized as follows. Let 𝒫\mathcal{P} be the set of all data recipe pairs (A,B)(A,B) with observed mean performance yA,yBy\_{A},y\_{B} and predicted performance y^A,y^B\hat{y}\_{A},\hat{y}\_{B}, respectively, then decision accuracy is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 1|𝒫|​∑(A,B)∈𝒫𝕀​(sign​(y^A−y^B)=sign​(yA−yB))\textstyle\frac{1}{\lvert\mathcal{P}\rvert}\sum\_{(A,B)\in\mathcal{P}}\mathbb{I}\big{(}\text{sign}(\hat{y}\_{A}-\hat{y}\_{B})=\text{sign}(y\_{A}-y\_{B})\big{)} |  | (3) |

#### Percent of Target Compute Budget (%C\%C)

We measure compute in terms of theoretical FLOPs following the simplifying assumption made in most scaling literature that the costs associated with training a model are captured well enough by FLOPs=6​N​D\text{FLOPs}=6ND, based solely on the number of parameters (NN) and tokens trained (DD) (Kaplan et al., [2020](#bib.bib21)). We consider the efficiency of a prediction based on the ratio of the experimental budget and the target budget in FLOPs, %C=cC×100%\text{$\%C${}}=\frac{c}{C}\times 100\%.

### 2.4 Performance Evaluation with OLMES

We use the OLMES suite of 10 multiple choice question answering benchmarks (Gu et al., [2024](#bib.bib17)): MMLU (Hendrycks et al., [2021](#bib.bib18)), HellaSwag (Zellers et al., [2019](#bib.bib41)), ARC Challenge (Clark et al., [2018](#bib.bib10)), ARC Easy (Clark et al., [2018](#bib.bib10)), PIQA (Bisk et al., [2020](#bib.bib5)), CommonsenseQA (Talmor et al., [2019](#bib.bib39)),SocialIQA (Sap et al., [2019](#bib.bib35)), OpenBookQA (Mihaylov et al., [2018](#bib.bib26)), BoolQ (Clark et al., [2019](#bib.bib9)), and WinoGrande (Sakaguchi et al., [2020](#bib.bib34)). These tasks are well suited for the model scales we examine with all but BoolQ receiving non-trivial performance. Unless otherwise noted, we consider the macro average of these ten tasks. The underlying metric for each task is accuracy, for which OLMES specifies a different length normalization scheme per task. Our target “gold” rankings which we aim to predict are always based on the “cloze” formulation (CF) accuracy with curated normalization per task, which we refer to as Accuracy. We diverge from OLMES only in that we make use of all available items in the specified split of each benchmark rather than subsampling them, to reduce variance over the task distribution.

Note that while we focus just on OLMES multiple choice evaluations in this work, our method of validating decisions made through predictions can be applied to other benchmarks. We chose these tasks based on their appropriateness to our range of model scales, and one would have to select different tasks when targeting a larger scale. Moreover, DataDecide could be used to identify new evaluations that are sensitive within our range of scales.

### 2.5 Proxy Metrics for Performance Evaluation

Previous work has noted how discrete metrics such as accuracy can cause jumps in performance across scale that otherwise see more predictable improvements with scale for continuous metrics (Schaeffer et al., [2023](#bib.bib36)).
We experiment with using continuous metrics at small scale as proxies of the accuracies selected by OLMES for each task (Accuracy) at the target scale to improve decision accuracy. We use the following metrics: Correct Prob is the average probabilities of the correct continuations. Margin is the average difference between the probability of the correct continuation and the most likely incorrect continuation. Norm Correct Prob is the average probability of the correct continuation conditioned on the response being in the set of correct or incorrect continuations. Total Prob is the average of the sum of probabilities of all correct and incorrect continuations. Accuracy is the fraction of instances where the correct continuation has the highest probability. Each of these can be computed with likelihoods normalized by number of tokens or characters; unless otherwise specified we use character length normalization. Appendix Table [3](#A2.T3 "Table 3 ‣ Appendix B Proxy Metric Definitions ‣ How to Predict Best Pretraining Data with Small Experiments") shows formal definitions.

## 3 Results

![Refer to caption](/html/2504.11393/assets/x3.png)


Figure 2: Accuracy in pairwise decisions on best data when evaluating on the 10 OLMES tasks with Accuracy (shown aggregated in Figure [1](#S0.F1 "Figure 1 ‣ How to Predict Best Pretraining Data with Small Experiments")). Specific tasks have very distinct ranges of sensitivity, with some like ARC Easy being predictable at small scales and others like HellaSwag requiring substantially more compute to predict.

### 3.1 What is the best way to spend compute for data decisions?

More compute makes better decisions. Decisions from intermediate checkpoints are as good as compute equivalent final checkpoints. The amount of compute needed to make good predictions varies between tasks. ARC and MMLU are predictable with much less compute than HellaSwag. The rest of OLMES tasks give markedly less reliable predictions across the scales we examine.

First looking at the aggregation of all 10 OLMES tasks (Figure [1](#S0.F1 "Figure 1 ‣ How to Predict Best Pretraining Data with Small Experiments") right), we see that there is a positive and roughly log-linear relationship between experimental compute and decision accuracy. Specifically, this figure illustrates the relationship between the compute used for predicting best data recipes and the decision accuracy those predictions achieve against targets ranked by OLMES performance at the 1B scale. Each point represents the average decision accuracy over three runs with different random seeds, with shading indicating standard deviation. Points with the same color show all intermediate checkpoints from a given parameter size. The color shows each model size for predicting using ranking single scale experiments. The stars show predictions from extrapolating scaling laws using our default 3-parameter approach, the details of which are discussed further in §[3.2](#S3.SS2 "3.2 How does extrapolating scaling laws compare to ranking single scale experiments? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments").

The ease of prediction is greatly influenced by which evaluation benchmark we use. In Figure [2](#S3.F2 "Figure 2 ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments"), we show the relationship of compute and decision accuracy for each of the tasks in OLMES individually. The predictive sensitivity of tasks at a given compute varies significantly, with ARC Easy being consistently predictable with 5 orders of magnitude less compute and BoolQ only reaching beyond trivial decision accuracy for intermediate checkpoints of the target runs. HellaSwag, SocialIQA, WinoGrande show distinct periods of insensitivity followed by roughly log-linear increase after hitting some compute threshold.

![Refer to caption](/html/2504.11393/assets/x4.png)


Figure 3: Decision accuracy over 8 baseline scaling law variants. At best, these approaches reach only the same compute to decision accuracy frontier as ranking single scale experiments. DataDecide can be used to iterate on future scaling law prediction methods.

### 3.2 How does extrapolating scaling laws compare to ranking single scale experiments?

A selection of 8 baseline scaling law methods are no more efficient than ranking single scale experiments. Future scaling law methods can be assessed on DataDecide.

Figure [3](#S3.F3 "Figure 3 ‣ 3.1 What is the best way to spend compute for data decisions? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") contrasts different approaches to fitting scaling laws over multiple scales of small experiments. Each of the 8 approaches is shown in a different color. Multi-scale predictions have a compute budget equal to the training cost of the model sizes used to make the prediction. We try the following combinations of models sizes: We use {{s1,…,sk}∣3≤k≤14}\left\{\{s\_{1},\dots,s\_{k}\}\mid 3\leq k\leq 14{}\right\}, where 𝐬\mathbf{s} is the ordered set of sizes, to explore the improvements of progressively adding larger model sizes beyond the minimum 3 required for fitting. We also use {{sk,…,s14}∣2≤k≤11}\left\{\{s\_{k},\dots,s\_{14}\}\mid 2\leq k\leq 11\right\} to try removing potentially noisy information from small models. Unlike single scale results, we make only one prediction attempt with the default fully trained random seed, as final checkpoints are required for fitting the first step of these scaling law variants but are not available for all seeds.

Our scaling law approaches vary in the number of parameters fit, using hard coded points to define the minimum and maximum performance, using only the second half of intermediate checkpoints for fitting the second step, or fitting a function directly from compute to accuracy in a single step. Each of the scaling law variants are defined formally in Appendix [C](#A3 "Appendix C Scaling Law Variants ‣ How to Predict Best Pretraining Data with Small Experiments"). The 2 and 3 parameter variants all achieve among the top decision accuracy.

A priori we know that ranking single scale experiments cannot correctly predict when the scaling trend of one data recipe overtakes another at scales between our small experiments and target scale. Such crossovers bound the decision accuracy of this constant approximation of performance. Nevertheless ranking single scale experiments sets a high baseline decision accuracy, implying relatively little crossover occurs. It is difficult to distinguish evaluation variance from true crossovers, but the scaling trends we empirically observe cross over frequently. Improved future scaling laws may be able to advance the Pareto frontier on DataDecide as they are not bound by crossovers.

![Refer to caption]()


Figure 4: 
Per-task decision accuracy using character normalized proxy metrics for Accuracy targets. 5 tasks benefit at smaller scales from using raw likelihood of answers (Correct Prob and Total Prob), as opposed to discrete Accuracy or continuous metrics that penalize probability on incorrect answers (Norm Correct Prob, Margin).

### 3.3 What proxy metrics give better signal for predictions at small scale?

At small scales, continuous metrics using the character normalized likelihood of correct or all answer options serve as better or equivalent predictors of decisions than using the same Accuracy as used at the target scale.

Figure [4](#S3.F4 "Figure 4 ‣ 3.2 How does extrapolating scaling laws compare to ranking single scale experiments? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") shows the decision accuracy over different proxy metrics. Here we chose a single length normalization, \*\_per\_char. Metrics follow similar trends regardless of length normalization and this one is empirically optimal for most of the tasks that we observe.

Using Correct Prob or Total Prob leads to decision accuracy at least as good as any other metric for most small scales. These continuous metrics are simple likelihoods over answer strings. In particular, Total Prob may be interpretable as signal of a model having exposure to the domain of a given task in the form of higher likelihoods on incorrect but presumably relevant additional answers.

We notice two very distinct types of trends over the different tasks. Either the different proxy metrics are nearly indistinguishable and increase in decision accuracy with compute or Correct Prob and Total Prob are flat with respect to scale and the other metrics only rise up to that level of decision accuracy towards the full target compute budget. In the last order of magnitude below the target compute Accuracy and the other metrics tend to overtake Correct Prob and Total Prob, while these two metrics sometimes even decrease in decision accuracy. Notably these other metrics that trend with Accuracy include continuous metrics that penalize probability assigned to incorrect answers, Norm Correct Prob and Margin.

![Refer to caption](/html/2504.11393/assets/x6.png)


Figure 5: Why do some tasks or metrics get better or worse decision accuracy? At 150M with Correct Prob tasks like HellaSwag succeed with low run-to-run variance and tasks like SocialIQA widely spread the performance assigned to different pretraining data.

### 3.4 How can we make evaluation benchmarks more predictable?

The decision accuracy on a task is driven in part by low run-to-run variance and a wide spread of performance values for different data recipes. Using Correct Prob sees wider spreads or reduced noise for many tasks. Using this metric enables predicting rankings for code tasks that are too hard for accuracy metrics at small scales.

What underlies differences in decision accuracy when benchmarks and metrics change? The evaluation must separate pairs of data recipes by an amount greater than combined noise from run-to-run variance of each of the pair’s runs. In Figure [5](#S3.F5 "Figure 5 ‣ 3.3 What proxy metrics give better signal for predictions at small scale? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments"), we plot tasks with a given metric using fully trained 150M models over these two characteristics: 1) noise—the standard deviation over 3 random seed runs averaged over all recipes, and 2) spread—the standard deviation among the mean performance of the different data recipes. Each point also shows the decision accuracy. We see that some highly predictable tasks (e.g., MMLU) are characterized by having low run-to-run noise, while others (e.g., ARC Easy) widely spread the different data recipes. We also see that improvements from using Correct Prob often align with improvements in one of these two characteristics.

![Refer to caption](/html/2504.11393/assets/x7.png)


Figure 6: Code tasks such as humaneval and MBPP go from trivial decision accuracy to largely predictable when using using continuous Correct Prob instead of discrete Accuracy. Meanwhile common math tasks remain near trivial decision accuracy regardless of metric.

As a practical application of these insights, we demonstrate that a change of proxy metric makes predictable two code tasks (Austin et al., [2021](#bib.bib1); Chen et al., [2021](#bib.bib7)) that are otherwise too challenging for our small models. Figure [6](#S3.F6 "Figure 6 ‣ 3.4 How can we make evaluation benchmarks more predictable? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments") shows how decision accuracy goes from trivial to  80% when using Correct Prob. The switch of metric allows small models to get above the noise floor for these tasks, while still predicting large-scale accuracy metrics. Notably, two math benchmarks (Lewkowycz et al., [2022](#bib.bib23); Cobbe et al., [2021](#bib.bib11)) do not see such a benefit. They do however give decision accuracy above 80% if we switch the target metric to Correct Prob, raising a question for future work to explore whether changing the target metric can be justified.

## 4 Related Work

#### Prediction

Much work studies scaling behavior in language models. Initially this focused on predicting LM loss from scale as determined by parameter count and tokens trained (Kaplan et al., [2020](#bib.bib21); Hoffmann et al., [2022](#bib.bib19)). Special consideration is also given to the case of data constrained scaling (Muennighoff et al., [2023](#bib.bib27); Goyal et al., [2024](#bib.bib15)).
Unlike predicting loss, predicting downstream performance from scale is generally harder (Schaeffer et al., [2024](#bib.bib37)). However, recent work has demonstrated it can be done based on a two step prediction that chains together predictions from scale to loss and loss to downstream performance
(Gadre et al., [2024](#bib.bib14); Bhagia et al., [2024](#bib.bib3); Dubey et al., [2024](#bib.bib13)), sometimes using training loss (Du et al., [2024](#bib.bib12)) or transferring losses from different data recipes (Brandfonbrener et al., [2024](#bib.bib6); Ruan et al., [2024](#bib.bib33)). The one line of work targeting pretraining data considers the special case of deciding mixing proportions of several data sources optimized through scaling laws (Kang et al., [2024](#bib.bib20); Ye et al., [2024](#bib.bib40)). Most relevant to our work, Choshen et al. ([2024](#bib.bib8)) consider practical methods for better scaling prediction error such as how much compute to use or whether to include intermediate checkpoints. Orthogonally to these findings, we propose a way to assess the accuracy of decisions made with such predictions.

#### Suites over Data Differences

DataDecide follows in the footsteps of the Pythia Suite (Biderman et al., [2023](#bib.bib4)) which was the first to offer a controlled comparison of 2 data recipes, using compute scales up to 2×10222\times 10^{22} FLOPs.
Subsequent suites have offered 6 data recipes at 9×10209\times 10^{20} scale (Magnusson et al., [2024](#bib.bib25)) and 6 data recipes over a range of scales up to 102110^{21} (Brandfonbrener et al., [2024](#bib.bib6)). Our DataDecide offers a range of 14 scales up to 7×10207\times 10^{20} FLOPs, while including an order of magnitude more fine-grained data differences.
Meanwhile, DCLM also makes extensive use of ranking single scale experiments to drive improvement in data recipes (Li et al., [2024](#bib.bib24)). They release their best data and a model trained on it, but do not release models from their decision making experiments and do not search over multiple recipes at their largest scale. Where their goal is creating a proposed best recipe, our DataDecide enables the assessment of whether a method for decision making really does find the best among proposed recipes.

## 5 Limitations

The scope of our work is limited to just one ratio of tokens to parameters, 100 or 5×\times “Chinchilla” optimal ratio (Hoffmann et al., [2022](#bib.bib19)). We believe this captures the typical case, as most models now favor overtraining for inference savings.
Due to compute limitations and the need for a standardized set of model configurations over a long period of time in which compute became available for pretraining, we opt for 14 specific configurations from 4M–1B parameter scale. While observations across more configurations would always be better, this must be traded off with exploring the other dimensions of data recipes and random seed reruns. Likewise, while our 25 data recipes is an order of magnitude more than previous suites, there is always the possibility that findings across these will not be representative of future data recipes.
In our evaluations we focus on multiple choice tasks with a “cloze” formulation as we find these to be a good fit for our range of scales. Using DataDecide, new evaluations can be assessed easily by others without any additional pretraining.

## Acknowledgments

We would like to thank Dave Wadden, Kyle Lo, Valentin Hofmann, and Hannaneh Hajishirzi for fruitful conversations. This material is based upon work supported by the U.S. National Science Foundation under Grant No. 2313998. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. National Science Foundation. IM is supported by the NSF CSGrad4US Fellowship. PWK is supported by the Singapore National Research Foundation and the National AI Group in the Singapore Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-001) and by the AI2050 program at Schmidt Sciences.

## Ethics Statement

Training large language models is computationally expensive, especially when investigating thoroughly over dimensions of pretraining data composition, model scale, random initialization, and data order. The pretraining experiments in our DataDecide required approximately 820K H100 GPU hours. We share the benefit of this cost through releasing all of our models, data, and evaluations so that others will not have to repeat this expenditure. Moreover, our findings can guide efficient and cost-effective model development through the application of decision making with small-scale experiments. While DataDecide does not present direct ethical concerns beyond opportunity cost, we acknowledge that decisions about pretraining data heavily impact downstream model behavior. We encourage future research to explore potential biases in data selection methods and their implications for models deployed in the real world.

## References

* Austin et al. (2021)

  Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al.
  Program synthesis with large language models.
  *arXiv preprint arXiv:2108.07732*, 2021.
* Ben Allal et al. (2024)

  Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra.
  Smollm-corpus, 2024.
  URL <https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus>.
* Bhagia et al. (2024)

  Akshita Bhagia, Jiacheng Liu, Alexander Wettig, David Heineman, Oyvind Tafjord, Ananya Harsh Jha, Luca Soldaini, Noah A. Smith, Dirk Groeneveld, Pang Wei Koh, Jesse Dodge, and Hannaneh Hajishirzi.
  Establishing task scaling laws via compute-efficient model ladders, 2024.
  URL <https://arxiv.org/abs/2412.04403>.
* Biderman et al. (2023)

  Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal.
  Pythia: A suite for analyzing large language models across training and scaling, 2023.
  URL <https://arxiv.org/abs/2304.01373>.
* Bisk et al. (2020)

  Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi.
  PIQA: Reasoning about physical commonsense in natural language.
  *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05):7432–7439, Apr. 2020.
  doi: 10.1609/aaai.v34i05.6239.
  URL <https://ojs.aaai.org/index.php/AAAI/article/view/6239>.
* Brandfonbrener et al. (2024)

  David Brandfonbrener, Nikhil Anand, Nikhil Vyas, Eran Malach, and Sham Kakade.
  Loss-to-loss prediction: Scaling laws for all datasets, 2024.
  URL <https://arxiv.org/abs/2411.12925>.
* Chen et al. (2021)

  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.
  Evaluating large language models trained on code, 2021.
  URL <https://arxiv.org/abs/2107.03374>.
* Choshen et al. (2024)

  Leshem Choshen, Yang Zhang, and Jacob Andreas.
  A hitchhiker’s guide to scaling law estimation, 2024.
  URL <https://arxiv.org/abs/2410.11840>.
* Clark et al. (2019)

  Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova.
  BoolQ: Exploring the surprising difficulty of natural yes/no questions.
  pp.  2924–2936, Minneapolis, Minnesota, June 2019.
  doi: 10.18653/v1/N19-1300.
  URL <N19-1300>.
* Clark et al. (2018)

  Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
  Think you have solved question answering? try arc, the ai2 reasoning challenge.
  *ArXiv*, 2018.
  URL <http://arxiv.org/abs/1803.05457>.
* Cobbe et al. (2021)

  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman.
  Training verifiers to solve math word problems.
  *arXiv preprint arXiv:2110.14168*, 2021.
* Du et al. (2024)

  Zhengxiao Du, Aohan Zeng, Yuxiao Dong, and Jie Tang.
  Understanding emergent abilities of language models from the loss perspective.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
  URL <https://openreview.net/forum?id=35DAviqMFo>.
* Dubey et al. (2024)

  Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony S. Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Cantón Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab A. AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriele Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guanglong Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu,
  Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Laurens Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Ju-Qing Jia, Kalyan Vasuden Alwala, K. Upasani, Kate Plawiak, Keqian Li, Ken-591 neth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Babu Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melissa Hall Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay
  Bashlykov, Nikolay Bogoychev, Niladri S. Chatterji, Olivier Duchenne, Onur cCelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasić, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Chandra Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez,
  Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whit ney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yiqian Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zhengxu Yan, Zhengxing Chen, Zoe Papakipos, Aaditya K. Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adi Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Ben Leonhardi, Po-Yao (Bernie) Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram
  Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Shang-Wen Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzm’an, Frank J. Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory G. Sizov, Guangyi Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Han Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Igor Molybog, Igor
  Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kaixing(Kai) Wu, U KamHou, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, A Lavender, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang,
  Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollár, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve
  Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sung-Bae Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Andrei Poenaru, Vlad T. Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xia Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao.
  The llama 3 herd of models.
  *ArXiv*, abs/2407.21783, 2024.
  URL <https://api.semanticscholar.org/CorpusID:271571434>.
* Gadre et al. (2024)

  Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Jenia Jitsev, Luca Soldaini, Alexandros G. Dimakis, Gabriel Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff, and Ludwig Schmidt.
  Language models scale reliably with over-training and on downstream tasks, 2024.
  URL <https://arxiv.org/abs/2403.08540>.
* Goyal et al. (2024)

  Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, and J. Zico Kolter.
  Scaling laws for data filtering - data curation cannot be compute agnostic.
  *CoRR*, abs/2404.07177, 2024.
  doi: 10.48550/ARXIV.2404.07177.
  URL <https://doi.org/10.48550/arXiv.2404.07177>.
* Groeneveld et al. (2024)

  Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi.
  Olmo: Accelerating the science of language models, 2024.
  URL <https://arxiv.org/abs/2402.00838>.
* Gu et al. (2024)

  Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi.
  Olmes: A standard for language model evaluations, 2024.
  URL <https://arxiv.org/abs/2406.08446>.
* Hendrycks et al. (2021)

  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
  Measuring massive multitask language understanding.
  *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.
* Hoffmann et al. (2022)

  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre.
  Training compute-optimal large language models, 2022.
  URL <https://arxiv.org/abs/2203.15556>.
* Kang et al. (2024)

  Feiyang Kang, Yifan Sun, Bingbing Wen, Si Chen, Dawn Song, Rafid Mahmood, and Ruoxi Jia.
  Autoscale: Automatic prediction of compute-optimal data composition for training llms.
  *ArXiv*, abs/2407.20177, 2024.
  URL <https://api.semanticscholar.org/CorpusID:271533897>.
* Kaplan et al. (2020)

  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
  Scaling laws for neural language models, 2020.
  URL <https://arxiv.org/abs/2001.08361>.
* Lambert et al. (2024)

  Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi.
  Tülu 3: Pushing frontiers in open language model post-training.
  2024.
* Lewkowycz et al. (2022)

  Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra.
  Solving quantitative reasoning problems with language models, 2022.
  URL <https://arxiv.org/abs/2206.14858>.
* Li et al. (2024)

  Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar.
  Datacomp-lm: In search of the next generation of training sets for language models, 2024.
  URL <https://arxiv.org/abs/2406.11794>.
* Magnusson et al. (2024)

  Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah A. Smith, Kyle Richardson, and Jesse Dodge.
  Paloma: A benchmark for evaluating language model fit, 2024.
  URL <https://arxiv.org/abs/2312.10523>.
* Mihaylov et al. (2018)

  Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.
  Can a suit of armor conduct electricity? a new dataset for open book question answering.
  pp.  2381–2391, Brussels, Belgium, October-November 2018.
  doi: 10.18653/v1/D18-1260.
  URL <D18-1260>.
* Muennighoff et al. (2023)

  Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel.
  Scaling data-constrained language models.
  In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp.  50358–50376. Curran Associates, Inc., 2023.
  URL <https://proceedings.neurips.cc/paper_files/paper/2023/file/9d89448b63ce1e2e8dc7af72c984c196-Paper-Conference.pdf>.
* OLMo et al. (2025)

  Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi.
  2 olmo 2 furious, 2025.
  URL <https://arxiv.org/abs/2501.00656>.
* Penedo et al. (2023)

  Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra-Aimée Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay.
  The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only.
  *ArXiv*, abs/2306.01116, 2023.
  URL <https://api.semanticscholar.org/CorpusID:259063761>.
* Penedo et al. (2024)

  Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf.
  The fineweb datasets: Decanting the web for the finest text data at scale, 2024.
  URL <https://arxiv.org/abs/2406.17557>.
* Porian et al. (2024)

  Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon.
  Resolving discrepancies in compute-optimal scaling of language models.
  *ArXiv*, abs/2406.19146, 2024.
  URL <https://api.semanticscholar.org/CorpusID:270764838>.
* Raffel et al. (2019)

  Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text transformer.
  *arXiv e-prints*, 2019.
* Ruan et al. (2024)

  Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto.
  Observational scaling laws and the predictability of langauge model performance.
  In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024.
  URL <https://openreview.net/forum?id=On5WIN7xyD>.
* Sakaguchi et al. (2020)

  Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.
  WinoGrande: An adversarial winograd schema challenge at scale.
  *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05):8732–8740, Apr. 2020.
  doi: 10.1609/aaai.v34i05.6399.
  URL <https://ojs.aaai.org/index.php/AAAI/article/view/6399>.
* Sap et al. (2019)

  Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi.
  Social IQa: Commonsense reasoning about social interactions.
  pp.  4463–4473, Hong Kong, China, November 2019.
  doi: 10.18653/v1/D19-1454.
  URL <D19-1454>.
* Schaeffer et al. (2023)

  Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo.
  Are emergent abilities of large language models a mirage?, 2023.
  URL <https://arxiv.org/abs/2304.15004>.
* Schaeffer et al. (2024)

  Rylan Schaeffer, Hailey Schoelkopf, Brando Miranda, Gabriel Mukobi, Varun Madan, Adam Ibrahim, Herbie Bradley, Stella Biderman, and Sanmi Koyejo.
  Why has predicting downstream capabilities of frontier AI models with scale remained elusive?
  In *Trustworthy Multi-modal Foundation Models and AI Agents (TiFA)*, 2024.
  URL <https://openreview.net/forum?id=AbHHrj9afB>.
* Soldaini et al. (2024)

  Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo.
  Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research.
  *arXiv preprint*, 2024.
* Talmor et al. (2019)

  Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.
  CommonsenseQA: A question answering challenge targeting commonsense knowledge.
  pp.  4149–4158, Minneapolis, Minnesota, June 2019.
  doi: 10.18653/v1/N19-1421.
  URL <N19-1421>.
* Ye et al. (2024)

  Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, and Xipeng Qiu.
  Data mixing laws: Optimizing data mixtures by predicting language modeling performance.
  *ArXiv*, abs/2403.16952, 2024.
  URL <https://api.semanticscholar.org/CorpusID:268681464>.
* Zellers et al. (2019)

  Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
  HellaSwag: Can a machine really finish your sentence?
  pp.  4791–4800, Florence, Italy, July 2019.
  doi: 10.18653/v1/P19-1472.
  URL <P19-1472>.
* Zhou et al. (2024)

  Fan Zhou, Zengzhi Wang, Qian Liu, Junlong Li, and Pengfei Liu.
  Programming every example: Lifting pre-training data quality like experts at scale.
  *arXiv preprint arXiv:2409.17115*, 2024.

## Appendix A Hyperparameters

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model name | Batch size | Hidden dim. | LR | Model size | Heads | Layers | Training steps | Tokens trained |
| 4M | 32 | 64 | 1.4e-02 | 3.7M | 8 | 8 | 5,725 | 0.4B |
| 6M | 32 | 96 | 1.2e-02 | 6.0M | 8 | 8 | 9,182 | 0.6B |
| 8M | 32 | 128 | 1.1e-02 | 8.5M | 8 | 8 | 13,039 | 0.9B |
| 10M | 32 | 144 | 1.0e-02 | 9.9M | 8 | 8 | 15,117 | 1.0B |
| 14M | 32 | 192 | 9.2e-03 | 14.4M | 8 | 8 | 21,953 | 1.4B |
| 16M | 32 | 208 | 8.9e-03 | 16.0M | 8 | 8 | 24,432 | 1.6B |
| 20M | 64 | 192 | 8.4e-03 | 19.1M | 8 | 16 | 14,584 | 1.9B |
| 60M | 96 | 384 | 5.8e-03 | 57.1M | 12 | 16 | 29,042 | 5.7B |
| 90M | 160 | 528 | 4.9e-03 | 97.9M | 12 | 16 | 29,901 | 9.8B |
| 150M | 192 | 768 | 4.2e-03 | 151.9M | 12 | 12 | 38,157 | 15.0B |
| 300M | 320 | 1,024 | 3.3e-03 | 320.0M | 16 | 16 | 45,787 | 30.0B |
| 530M | 448 | 1,344 | 2.8e-03 | 530.1M | 16 | 16 | 57,786 | 53.0B |
| 750M | 576 | 1,536 | 2.5e-03 | 681.3M | 16 | 16 | 63,589 | 75.0B |
| 1B | 704 | 2,048 | 2.1e-03 | 1176.8M | 16 | 16 | 69,369 | 100.0B |

Table 2: DataDecide uses OLMo’s model ladder (Groeneveld et al., [2024](#bib.bib16); OLMo et al., [2025](#bib.bib28); Bhagia et al., [2024](#bib.bib3)) to programmatically create configurations for 14 model sizes with hyperparameters determined by heuristics in Porian et al. ([2024](#bib.bib31)). All models have sequence length of 2024 and MLP ratio of 8. Each configuration is pretrained over 25 data recipes (Table [1](#S2.T1 "Table 1 ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments")).
Each recipe and configuration is also trained for 3 random seeds where model sizes <1<1B are stopped early at 25% of the compute used to train the 1B model for all but the default seed. Model size is number of non-embedding parameters. Batch size is the number of sequences per batch.

Table [2](#A1.T2 "Table 2 ‣ Appendix A Hyperparameters ‣ How to Predict Best Pretraining Data with Small Experiments") provides OLMo model ladder configurations for all models in DataDecide.

## Appendix B Proxy Metric Definitions

|  |  |
| --- | --- |
| Metric Name | Equation |
| Correct Prob | 1N​∑i=1NP​(ccorrect(i)∣contexti)\frac{1}{N}\sum\_{i=1}^{N}P(c^{(i)}\_{\text{correct}}\mid\text{context}\_{i}) |
| Margin | 1N​∑i=1N(P​(ccorrect(i)∣contexti)−maxc′≠ccorrect(i)∈C(i)⁡P​(c′∣contexti))\frac{1}{N}\sum\_{i=1}^{N}\big{(}P(c\_{\text{correct}}^{(i)}\mid\text{context}\_{i})-\max\_{c^{\prime}\neq c\_{\text{correct}}^{(i)}\in C^{(i)}}P(c^{\prime}\mid\text{context}\_{i})\big{)} |
| Norm Correct Prob | 1N​∑i=1NP​(ccorrect(i)∣contexti)∑c∈C(i)P​(c∣contexti)\frac{1}{N}\sum\_{i=1}^{N}\frac{P(c^{(i)}\_{\text{correct}}\mid\text{context}\_{i})}{\sum\_{c\in C^{(i)}}P(c\mid\text{context}\_{i})} |
| Total Prob | 1N​∑i=1N∑c∈C(i)P​(c∣contexti)\frac{1}{N}\sum\_{i=1}^{N}\sum\_{c\in C^{(i)}}P(c\mid\text{context}\_{i}) |
| Accuracy | 1N​∑i=1N𝕀​(arg⁡maxc∈C(i)⁡P​(c∣contexti)=ccorrect(i))\frac{1}{N}\sum\_{i=1}^{N}\mathbb{I}\big{(}\arg\max\_{c\in C^{(i)}}P(c\mid\text{context}\_{i})=c\_{\text{correct}}^{(i)}\big{)} |
| \*\_per\_token | P​(c∣context)/tokens​(c)\nicefrac{{P(c\mid\text{context})}}{{\text{tokens}(c)}} |
| \*\_per\_char | P​(c∣context)/chars​(c)\nicefrac{{P(c\mid\text{context})}}{{\text{chars}(c)}} |

Table 3: Proxy metrics used as alternative inputs to our prediction methods, C(i)C^{(i)} is the set of possible continuations for item ii and NN is the number of items in a benchmark. Each each of the first 5 metrics have \*\_per\_token and \*\_per\_char variants in which likelihoods are normalized as defined in the bottom two rows.

Table [3](#A2.T3 "Table 3 ‣ Appendix B Proxy Metric Definitions ‣ How to Predict Best Pretraining Data with Small Experiments") provides formal definitions for our proxy metrics (§[2.5](#S2.SS5 "2.5 Proxy Metrics for Performance Evaluation ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments")).

|  |  |  |
| --- | --- | --- |
|  | Relative Error | Absolute Error |
| Scaling Law Variant |  |  |
| 3-parameter with helpers and >>50% checkpoints | 5.6 | 2.6 |
| 3-parameter with helper points | 6.0 | 2.8 |
| 3-parameter step 2 fit with >>50% checkpoints | 5.9 | 2.9 |
| 3-parameter | 6.5 | 3.1 |
| 2-parameter | 6.5 | 3.2 |
| 5-parameter, single step | 42.8 | 17.4 |
| 3-parameter, single step | 42.9 | 42.3 |
| 5-parameter | 230.8 | 65.4 |

Table 4: Average prediction error for 1B targets for the different scaling law setups across tasks and recipes on Accuracy fit to all models but 1B. We see that other than the single step and 5-parameter variants errors are comparable, and these variants also roughly follow the compute-decision frontier in Figure [3](#S3.F3 "Figure 3 ‣ 3.1 What is the best way to spend compute for data decisions? ‣ 3 Results ‣ How to Predict Best Pretraining Data with Small Experiments").

## Appendix C Scaling Law Variants

Baseline 3-parameter fit.
Our default setup (described in §[2.2](#S2.SS2 "2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments")) follows the two-step fit from (Bhagia et al., [2024](#bib.bib3)) and uses Equation [1](#S2.E1 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments") to map compute CC to task loss LL, and Equation [2](#S2.E2 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments") to map task loss to metric score. This variant fits three parameters (AA, α\alpha, EE) in the first step.

2-parameter fit.
This is a restricted version of the baseline where the irreducible loss term EE is removed from Equation [1](#S2.E1 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments"), leaving only two parameters:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(C)=ACαL(C)=\frac{A}{C^{\alpha}} |  | (4) |

5-parameter (N,D)(N,D) fit.
Instead of modeling loss as a function of compute CC, this variant uses both number of tokens NN and number of parameters DD directly in the loss function:

|  |  |  |  |
| --- | --- | --- | --- |
|  | L​(N,D)=ANα+BDβ+EL(N,D)=\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}+E |  | (5) |

This introduces five parameters: AA, α\alpha, BB, β\beta, and EE.

Single-step prediction.
In this variant, the two-stage fitting procedure is replaced with a single step that directly maps compute CC to accuracy:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A​c​c​(C)=a1+exp⁡(−k​(ACα+E−L0))+bAcc(C)=\frac{a}{1+\exp\left(-k\left(\frac{A}{C^{\alpha}}+E-L\_{0}\right)\right)}+b |  | (6) |

This combines the loss and accuracy mapping into one function.

5-parameter, single step.
We also test a single-step variant that directly maps from (N,D)(N,D) to accuracy using a logistic function over the predicted loss. This merges Equations [5](#A3.E5 "In Appendix C Scaling Law Variants ‣ How to Predict Best Pretraining Data with Small Experiments") and [2](#S2.E2 "In Extrapolating Scaling Laws (Multi Scale) ‣ 2.2 Prediction Methods ‣ 2 Methods ‣ How to Predict Best Pretraining Data with Small Experiments") into:

|  |  |  |  |
| --- | --- | --- | --- |
|  | A​c​c​(N,D)=a1+exp⁡(−(ANα+BDβ+E))+bAcc(N,D)=\frac{a}{1+\exp\left(-\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}+E\right)\right)}+b |  | (7) |

This formulation retains the same five parameters from the two-step (N,D)(N,D) loss function. Following Bhagia et al. ([2024](#bib.bib3)), we merge the parameters kk and L0L\_{0} from the second-stage sigmoid into the loss-side parameters (AA, BB, EE), yielding a simplified single-stage fit with 7 total free parameters: {A,α,B,β,E,a,b}\{A,\alpha,B,\beta,E,a,b\}.

Use of helper points.
Following Bhagia et al. ([2024](#bib.bib3)), we optionally include an extra point (L=0.0,A​c​c=1.0)(L=0.0,Acc=1.0) in the second-stage fit. This “helper” point anchors the upper asymptote of the accuracy prediction.

Filtering early checkpoints.
We experiment with excluding the first 50% of intermediate checkpoints when fitting the second-stage sigmoid. This reduces noise from high-loss early training points and often improves the fit for extrapolation.

Helpers and >50>50% checkpoints.
Lastly we experiment with combining the previous two techniques on the baseline 3-parameter fit.

Prediction Error. We report prediction errors in Table [4](#A2.T4 "Table 4 ‣ Appendix B Proxy Metric Definitions ‣ How to Predict Best Pretraining Data with Small Experiments") for each setup. As the best scaling laws variants are all roughly comparable to the simple 3-parameter set up, we use this one as our baseline.

[◄](/html/2504.11391)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2504.11393)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2504.11393)
[View original  
on arXiv](https://arxiv.org/abs/2504.11393)[►](/html/2504.11394)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon May 5 15:41:49 2025 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
