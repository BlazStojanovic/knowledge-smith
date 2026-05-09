---
arxiv: '2407.02112'
authors:
- Andrej Tschalzev University of Mannheim &Sascha Marton University of Mannheim &Stefan
  Lüdtke University of Rostock &Christian Bartelt University of Mannheim &Heiner Stuckenschmidt
  University of Mannheim
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular
  Data
url: https://arxiv.org/abs/2407.02112
year: 2024
---

[2407.02112] A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data














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



# A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data

Andrej Tschalzev
  
University of Mannheim
  
&Sascha Marton
  
University of Mannheim
  
&Stefan Lüdtke
  
University of Rostock
  
&Christian Bartelt
  
University of Mannheim
  
&Heiner Stuckenschmidt
  
University of Mannheim
  
Correspondence to: andrej.tschalzev@uni-mannheim.de

###### Abstract

Tabular data is prevalent in real-world machine learning applications, and new models for supervised learning of tabular data are frequently proposed.
Comparative studies assessing the performance of models typically consist of model-centric evaluation setups with overly standardized data preprocessing.
This paper demonstrates that such model-centric evaluations are biased, as real-world modeling pipelines often require dataset-specific preprocessing and feature engineering.
Therefore, we propose a data-centric evaluation framework. We select 10 relevant datasets from Kaggle competitions and implement expert-level preprocessing pipelines for each dataset. We conduct experiments with different preprocessing pipelines and hyperparameter optimization (HPO) regimes to quantify the impact of model selection, HPO, feature engineering, and test-time adaptation. Our main findings are: 1. After dataset-specific feature engineering, model rankings change considerably, performance differences decrease, and the importance of model selection reduces. 2. Recent models, despite their measurable progress, still significantly benefit from manual feature engineering. This holds true for both tree-based models and neural networks. 3. While tabular data is typically considered static, samples are often collected over time, and adapting to distribution shifts can be important even in supposedly static data. These insights suggest that research efforts should be directed toward a data-centric perspective, acknowledging that tabular data requires feature engineering and often exhibits temporal characteristics.

## 1 Introduction

Since ancient times, tables have been used as a data structure, i.e., to record astronomical observations [[82](#bib.bib82)] or financial transactions [[14](#bib.bib14)]. Many traditional machine learning (ML) methods, like logistic regression or the first artificial neural networks, were initially developed for tabular data [[22](#bib.bib22), [62](#bib.bib62), [72](#bib.bib72)]. Even nowadays, in the age of AI, tabular data is the most prevalent modality in real-world applications, including medicine [[41](#bib.bib41)], finance [[16](#bib.bib16)], manufacturing [[88](#bib.bib88)], retail [[57](#bib.bib57)], and many others [[75](#bib.bib75), [13](#bib.bib13)].
Several novel deep learning architectures have been contributed in recent years to improve supervised machine learning for tabular data [[69](#bib.bib69), [92](#bib.bib92), [61](#bib.bib61), [40](#bib.bib40), [7](#bib.bib7), [15](#bib.bib15), [34](#bib.bib34), [77](#bib.bib77), [19](#bib.bib19), [48](#bib.bib48), [33](#bib.bib33)].

To evaluate existing approaches, various comparative studies were conducted in recent years [[11](#bib.bib11), [31](#bib.bib31), [29](#bib.bib29), [34](#bib.bib34), [75](#bib.bib75), [13](#bib.bib13), [63](#bib.bib63)].
While motivated by different goals, they all have one in common: The focus is on evaluating models on tabular datasets using predefined cross-validation splits and one standardized preprocessing for all datasets.
In this paper, we challenge such model-centric evaluation setups by highlighting two major limitations (Section [2](#S2 "2 Related Work ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")):
1) The evaluation setups are overly standardized and do not reflect the actual routine of practitioners, which typically includes dataset-specific feature engineering [[81](#bib.bib81)].
2) There is no external reference for the highest possible performance on a task beyond a study’s own reporting, which limits its reliability.

To address these issues, we advocate for shifting the research perspective in the tabular data field from model-centric to data-centric.
Therefore, our main contribution is an evaluation framework that includes a collection of ten relevant real-world datasets, dataset-specific expert-level preprocessing pipelines, and an external measure of top performance for each dataset (Section [3](#S3 "3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).
The datasets were carefully selected by screening Kaggle competitions involving tabular data, and, to our knowledge, our contribution represents the largest existing collection of implemented expert-level solutions for tabular datasets.
To assess the potential bias from the first limitation, we investigate how the model comparison changes when considering dataset-specific preprocessing instead of standardized evaluation setups (Subsection [4.1](#S4.SS1 "4.1 How Model Comparisons Change When Considering Dataset-specific Preprocessing ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).
To address the second limitation, we use the leaderboard from Kaggle competitions as an external performance reference and reassess what is possible with modern methods that were not available when the Kaggle competitions took place (Subsection [4.2](#S4.SS2 "4.2 Measurable Progress Through Recent Efforts ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).
We find that when considering dataset-specific expert preprocessing, performance differences between the best models shrink, and the importance of selecting the ’right’ model diminishes.
In addition, we dissect expert solutions for tabular data competitions and quantify the importance of different modeling components (Subsection [4.3](#S4.SS3 "4.3 Feature Engineering is Still the Most Important Factor for Top Performance ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).
We find that measurable progress has been made in automating human effort, but feature engineering is still the most important aspect of many tabular data problems. No model fully automates this aspect and comparisons that don’t consider feature engineering merely scratch the surface of the potential performance achievable on many datasets.
This paper focuses on independent and identically distributed (i.i.d.) tabular data in line with related work. However, our analysis of Kaggle competitions shows strong evidence that this focus in the research community might not align with practitioners’ needs. In particular, we find that many tabular data competitions on Kaggle have temporal characteristics (i.e., timestamp features) and we identify test-time adaptation (TTA) as an overlooked but important part of some supposedly static competitions (Subsection [4.4](#S4.SS4 "4.4 The Importance of Test-Time Adaptation and Temporal Characteristics ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).

Our findings indicate that current academic evaluation setups and benchmarks for tabular data are biased due to their overly model-centric focus.
We conclude by discussing possible directions to improve machine learning for tabular data from a data-centric perspective (Section [5](#S5 "5 Implications for Future Work ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).

## 2 Related Work

Machine Learning for tabular data.   
Unlike domains like computer vision and natural language processing, an established state-of-the-art neural network architecture does not exist for tabular data [[75](#bib.bib75), [13](#bib.bib13)]. Therefore, recent research has primarily concentrated on developing general-purpose deep learning models often inspired by architectures from other domains [[69](#bib.bib69), [40](#bib.bib40), [52](#bib.bib52), [44](#bib.bib44), [7](#bib.bib7), [48](#bib.bib48), [92](#bib.bib92), [77](#bib.bib77), [37](#bib.bib37), [17](#bib.bib17), [83](#bib.bib83), [79](#bib.bib79), [90](#bib.bib90), [38](#bib.bib38), [65](#bib.bib65), [54](#bib.bib54), [85](#bib.bib85), [18](#bib.bib18), [19](#bib.bib19), [61](#bib.bib61), [33](#bib.bib33)].
Despite these efforts, Gradient Boosted Decision Trees (GBDTs) remain the state-of-the-art, outperforming even the novel neural models in many studies [[13](#bib.bib13), [35](#bib.bib35), [63](#bib.bib63)]. This paper aims to motivate more research inspired by tabular data-specific techniques like feature engineering instead of architectures established in other domains.

Limitations of current evaluation frameworks.   
Several benchmarks exist for evaluating tabular machine learning models, focusing on general model comparisons [[11](#bib.bib11), [31](#bib.bib31), [29](#bib.bib29), [35](#bib.bib35), [63](#bib.bib63)] and specific sub-problems [[46](#bib.bib46), [28](#bib.bib28), [21](#bib.bib21), [74](#bib.bib74), [27](#bib.bib27)]. However, these benchmarks do not provide preprocessing settings for the included datasets. Consequently, most studies adopt a fixed, standardized preprocessing for all datasets to concentrate on model comparisons [[75](#bib.bib75), [34](#bib.bib34), [63](#bib.bib63), [35](#bib.bib35), [47](#bib.bib47)]. While this model-centric approach is suitable for AutoML, it limits the real-world transferability of model comparisons, as models in practical applications typically follow dataset-specific preprocessing and feature engineering [[81](#bib.bib81), [87](#bib.bib87), [39](#bib.bib39)]. Our evaluation framework is the first to explicitly incorporate a more detailed distinction through diverse preprocessing pipelines.
Furthermore, existing benchmarks lack an external reference (e.g., a leaderboard) for the current best task performance, hindering comparability across different studies. In contrast, we leverage datasets from ML competitions as an external benchmark for high performance on tasks.
Many existing evaluation frameworks prioritize usability at the expense of representativeness by limiting sample sizes and removing high-cardinality categorical features, thus evaluating models on artificially constrained dataset versions [[11](#bib.bib11), [35](#bib.bib35)].
Our evaluation framework solely consists of tasks meaningful to the real world without imposing artificial restrictions on datasets.
Finally, most evaluation frameworks concentrate on tasks where samples are identically and independently distributed (i.i.d.). However, distribution shifts are prevalent in many machine learning applications [[51](#bib.bib51), [84](#bib.bib84), [86](#bib.bib86), [58](#bib.bib58), [91](#bib.bib91), [28](#bib.bib28)], and adapting to these shifts in tabular data has received limited attention [[45](#bib.bib45), [28](#bib.bib28)]. In this paper, we point out that excluding tabular data with temporal characteristics undermines the reliability of benchmarks, as many real-world applications using the benchmarked models include such data.

Using Kaggle for model evaluation.   
Kaggle is an online platform renowned for its machine learning competitions, hosted by companies and organizations to solve real-world problems in various domains.
Some studies have retrospectively compared the performance of new approaches in Kaggle competitions [[25](#bib.bib25), [71](#bib.bib71), [87](#bib.bib87)]. However, most of these studies are limited to a few competitions or only compared against the leaderboard without investing the high effort of implementing expert solutions. In Subsection [3.1](#S3.SS1 "3.1 Collection of Relevant and Challenging Datasets ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), we will explain that using Kaggle competitions to evaluate new approaches has several benefits. The evaluation framework most similar to ours is presented by Erickson et al., [[25](#bib.bib25)], where the proposed AutoML framework was compared to the leaderboard in Kaggle competitions. However, the methods leading to high performance on the leaderboard remain a black box. As we will show, some methods (i.e., test-time adaptation) prevent a fair comparison, and simply evaluating against the leaderboard is not helpful for gaining deeper insights. In contrast, we implement high-performing expert-level solutions, allowing us to dissect the components of interest and truly understand what drives high performance on specific tasks.

## 3 A Data-Centric Evaluation Framework for Tabular Machine Learning

![Refer to caption](/html/2407.02112/assets/x1.png)


Figure 1: Illustration of the components of our evaluation framework.

We propose an evaluation framework built upon three crucial aspects that are often overlooked in tabular data research: 1) Evaluation on realistic datasets without removing frequently occurring challenging aspects like high cardinality categorical features, 2) Dataset-specific expert preprocessing pipelines, and 3) Evaluation against human expert performance on hidden test sets.
Figure [1](#S3.F1 "Figure 1 ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") depicts an overview of our framework.
Our design choices are additionally justified by the fact that for each dataset, at least one model in our evaluation ranks among the top 1% of all competition participants.

### 3.1 Collection of Relevant and Challenging Datasets

We rely on the Kaggle community and competitions hosted by companies and institutions to select datasets with expert solutions.
Figure [2](#S3.F2 "Figure 2 ‣ 3.1 Collection of Relevant and Challenging Datasets ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") illustrates our dataset selection process, and Table [1](#S3.T1 "Table 1 ‣ 3.1 Collection of Relevant and Challenging Datasets ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") summarizes the main properties of the included datasets.
Using data from Kaggle competitions has various benefits:
1) The selected tasks are challenging and meaningful to the real world, as companies and institutions only spend money on hosting competitions from which they benefit.
2) Each competition has a clear evaluation setup, including metrics selected to reflect the practitioners’ needs.
3) Each competition has a large hidden test set, which has been shown to reduce the risk of adaptive overfitting [[71](#bib.bib71)].
4) The competition leaderboard serves as an external reference for truly high performance, as many expert teams participated in the competitions. Furthermore, our framework ensures a fair comparison by including a data loading function for each dataset that removes potential side issues, like data leakage or faulty data.
This distinguishes our framework from related work that compares approaches to Kaggle solutions [[25](#bib.bib25), [87](#bib.bib87)].
An important insight from screening the competitions is that most tabular datasets had temporal characteristics – i.e., datasets with weak temporal correlations that benefit from time-sensitive feature engineering but not from models with temporal inductive biases (i.e., [[3](#bib.bib3)]).
This finding will be further discussed in Subsection [4.4](#S4.SS4 "4.4 The Importance of Test-Time Adaptation and Temporal Characteristics ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data").

![Refer to caption](/html/2407.02112/assets/x2.png)


Figure 2: Illustration of the dataset selection process. Details on the criteria and all screened datasets can be found in the Appendix. The Figure only lists the competitions as temporal, which were not already excluded for other reasons. Consistent with related work, we include competitions that have timestamps but can be approached without time-sensitive feature engineering. In total, we identified 46 competition datasets with temporal characteristics (i.e., timestamps as a feature).



| Name | Year | N (Train) | N (Test) | D (Raw/FE) | Categorical | Metric | Model | TTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MBGM [[5](#bib.bib5)] | 2017 | 4,209 | 4,209 | 377 / 59 | 8 / 47 | r2 | XGBoost | No |
| SVPC [[60](#bib.bib60)] | 2018 | 12,296 | 49,342 | 4,992 / 1,420 | 0 / 0 | rmsle | LGBM | No |
| AEAC [[8](#bib.bib8)] | 2013 | 32,769 | 58,921 | 9 / 315 | 9 / 7,518 | auc | Ensemble | Yes |
| OGPCC [[9](#bib.bib9)] | 2015 | 61,878 | 144,368 | 93 / 104 | 0 / 0 | logloss | Ensemble | Yes |
| SCS [[78](#bib.bib78)] | 2016 | 76,020 | 75,818 | 370 / 224 | 0 / 0 | auc | Ensemble | Yes |
| BPCCM [[6](#bib.bib6)] | 2016 | 114,321 | 114,393 | 132 / 313 | 19 / 18,210 | logloss | XGBoost | No |
| SCTP [[64](#bib.bib64)] | 2019 | 200,000 | 200,000 | 200 / 600 | 0 / 0 | auc | NN | Yes |
| HQC [[23](#bib.bib23)] | 2015 | 260,753 | 173,836 | 299 / 300 | 29 / 868 | auc | XGBoost | No |
| IFD [[2](#bib.bib2)] | 2019 | 590,540 | 506,691 | 432 / 263 | 49 / 13,553 | auc | CatBoost | Yes |
| PSSDP [[1](#bib.bib1)] | 2017 | 595,212 | 892,816 | 57 / 53 | 8 / 104 | gini | NN | Yes |

Table 1: Datasets included in our framework. N denotes sample size, and D dimensionality of the raw data and after expert feature engineering. Categorical lists the no. of categorical features and the no. of clusters of the highest-cardinality categorical feature. Metric corresponds to the competition metric. MBGM and SVPC are regression tasks, OGPCC is a multi-class classification task, and the remaining are binary classification tasks. Model corresponds to the best single model used in the original expert solution. For some datasets, no best single model could be distinguished due to the heavy ensembling used. TTA denotes if test-time feature engineering has been used in the implemented expert solution.

### 3.2 Expert Solutions and Preprocessing Pipelines

Our proposed evaluation framework includes three preprocessing pipelines. One closely resembles the pipelines researchers currently use for model evaluation, and the other two are dataset-specific and directly derived from expert solutions.
All preprocessing pipelines are model-agnostic. Model-specific preprocessing steps are considered part of the model and are explained in the Appendix.

Standardized Preprocessing   
The main purpose of this pipeline in our framework is to evaluate single models in a scenario with minimal dataset-specific human effort invested.
Continuous missing values are replaced with the mean, and missing categorical feature values are treated as a new category. Furthermore, constant columns are removed, and heavy-tailed targets are log-transformed for regression tasks.
As these preprocessing steps are almost universally applied across related work [[34](#bib.bib34), [35](#bib.bib35), [63](#bib.bib63)], this pipeline represents current evaluation setups in academia well.

Expert Feature Engineering   
We select one high-performance expert solution from Kaggle for each dataset. The solution was chosen based on the private leaderboard rank and the descriptions’ quality and sufficiency.
For each solution, we separate the data preparation from the remaining parts of the solution.
For most datasets, this pipeline solely consists of feature engineering techniques. Besides a few distinctions between tree-based and deep learning models, the pipelines are model-agnostic. Model-specific preprocessing steps are considered part of the model in our framework and are explained in the Appendix.
This paper focuses on a pipeline perspective and does not discuss single feature engineering steps further.
Implementation details and feature engineering techniques used for specific datasets are provided in the Appendix and in our publicly available code.111This paper is currently under review. We will make our code publicly available here in a later version of the paper.
For this pipeline, we ensured that all feature engineering operations included were on the training data and that a model could have learned the same patterns without external information.

Test-Time Adaptation   
This pipeline is exactly the same as the expert feature engineering pipeline, with the key difference that the test data is used for feature engineering where applicable. Most ML competitions are organized so that the test features (but not the targets) are given. We found that the top solutions used the test data in their data preparation for six of the datasets in our framework. Hence, this pipeline represents the actual preprocessing used by the experts.
While this might be considered an unfair and unrealistic setup, there are applications where using unlabeled test data for unsupervised learning is applicable (see Appendix).
We argue that this conceptualization makes many tabular ML competitions a test-time adaptation (TTA) task.
TTA is a type of domain adaptation where test samples are used at test time in an unsupervised or self-supervised way to update or retrain a model [[89](#bib.bib89), [50](#bib.bib50), [49](#bib.bib49), [76](#bib.bib76), [67](#bib.bib67)].
We term the common Kaggle practice of engineering domain-invariant features at test time as ’test-time feature engineering’, which can be considered a type of test-time training [[80](#bib.bib80), [55](#bib.bib55), [66](#bib.bib66)].
With this preprocessing pipeline, we are the first to closer examine test-time feature engineering in Kaggle competitions.

### 3.3 Modeling and Evaluation Framework

Modeling Pipeline and Models   
We implement a unified modeling pipeline for all datasets with a dataset-specific cross-validation (CV) ensembling procedure.
The validation sets are used for early stopping and determining the best hyperparameters.
The final test data predictions are an ensemble of averaging the test predictions of each fold.
Our experiments compare 7 models and one AutoML solution for all datasets and preprocessing pipelines.
We use three gradient-boosted tree libraries (XGBoost [[20](#bib.bib20)], LightGBM [[43](#bib.bib43)], and CatBoost [[70](#bib.bib70)]) because each was used in at least one of the expert solutions.
Each expert solution that used neural networks developed a highly customized network for the particular competition. We want to assess whether recently developed general-purpose architectures can replace the high effort of building custom networks. Hence, we chose ResNet and FTTransformer [[34](#bib.bib34)] because they have been frequently used in recent benchmark comparisons and have shown strong performance [[35](#bib.bib35), [63](#bib.bib63)]. Because the Resnet essentially is an MLP with skip connections, it serves as a baseline representing what was already possible before the recent developments in DL for tabular data. In addition, we use two more recent approaches: MLP-PLR [[32](#bib.bib32)], which can help learn high-frequency functions, mitigating a major weakness of deep learning for tabular data [[35](#bib.bib35)]; and GRANDE [[61](#bib.bib61)], a recent representative of hybrid neural-tree models.
We are aware that even more recent architectures exist. However, our focus is not on particular models but on datasets and preprocessing.
To assess how well fully automated solutions perform without any preprocessing, we additionally evaluate AutoGluon, which has been shown to be the current best AutoML solution [[30](#bib.bib30)].

Hyperparameter Optimization Hyperparameter optimization is done per fold to obtain a diverse CV ensemble. Each model is evaluated in three HPO regimes: 1) Default: Either library default or hyperparameters suggested in related work, 2) Light HPO: 20 random search iterations. 3) Extensive HPO: 20 random search warmup iterations + 80 iterations of the tree-structured Parzen estimator algorithm [[4](#bib.bib4)]. More details on the hyperparameter optimization can be seen in the Appendix.

Evaluation   
We use the Kaggle API to automatically submit predictions and retrieve performance results after evaluating against the hidden targets. Each dataset is evaluated on the metric specified by the competition host.
Instead of reporting this metric directly, we report the solution’s private leaderboard position as the percentile. This has the benefit that although different metrics are used to evaluate the model, comparisons across datasets are possible. In the Appendix, we additionally report performances on the actual metrics for each dataset. Throughout the paper, higher values represent a better performance (leaderboard position).

![Refer to caption](/html/2407.02112/assets/x3.png)


Figure 3: Performance gains from different modeling components on the private Kaggle leaderboard by dataset and model. Higher values correspond to a better position. ’Default’ corresponds to the model performance with default hyperparameters in a standardized preprocessing pipeline. Light and extensive HPO correspond to tuning hyperparameters in the same preprocessing pipeline. Expert FE and FE-TTA correspond to the model performance with extensively tuned hyperparameters in the feature engineering and the test-time adaptation pipeline respectively.

## 4 Experimental Evaluation

Our framework allows us to assess the dataset-specific individual performance impact of model selection, hyperparameter optimization, feature engineering, and test-time adaptation. As a general overview, Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Modeling and Evaluation Framework ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows how each of the analyzed modeling components improves over the default baseline for each model and dataset. The results demonstrate the importance of an external performance reference: If we only considered the standardized evaluation setup (blue/orange/green bars), we would only be scratching the surface of achievable task performance for many data sets.

### 4.1 How Model Comparisons Change When Considering Dataset-specific Preprocessing

![Refer to caption](/html/2407.02112/assets/x4.png)


Figure 4: Average leaderboard position of models with different preprocessing. Black horizontal lines denote the Spearman correlation between all experiments with the respective preprocessing.

Through the implemented modeling pipelines, it becomes possible to evaluate how model comparisons change when evaluating inside the typically used standardized pipelines vs. expert pipelines with and without test-time adaptation. Three observations stand out when evaluating models in different preprocessing pipelines (Figure [4](#S4.F4 "Figure 4 ‣ 4.1 How Model Comparisons Change When Considering Dataset-specific Preprocessing ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")).
1) The model rankings change considerably, as indicated by the relatively low Spearman coefficients between the standardized preprocessing pipeline and the other pipelines.
2) The performance gap between all models diminishes when considering expert preprocessing. On average, all models benefit from feature engineering, and multiple models can reach top performance. While all models benefit from TTA, the performance increase varies.
3) The superiority of CatBoost vanishes when considering dataset-specific preprocessing. The reason is that CatBoost already incorporates specific feature engineering steps in its algorithm for which other models need manual engineering, as we will further elaborate in Subsection [4.3](#S4.SS3 "4.3 Feature Engineering is Still the Most Important Factor for Top Performance ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data").

### 4.2 Measurable Progress Through Recent Efforts

![Refer to caption](/html/2407.02112/assets/x5.png)


Figure 5: Progress made through recent models, illustrated by retrospective comparison to the Kaggle leaderboard.

Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Measurable Progress Through Recent Efforts ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows the model ranking on the private Kaggle leaderboard when trained after standardized preprocessing. CatBoost achieves top ranks in three competitions (MBGM, BPCCM, HQC) where a high manual effort in feature engineering was previously necessary. Similar to Erickson et al., [[25](#bib.bib25)], AutoGluon achieves top ranks in two of these (BPCCM, HQC) and one additional competition (OGPCC).
Regarding neural networks, novel architectures outperform the ResNet baseline on nine datasets and even outperform tree-based solutions for three datasets (SVPC, SCS, SCTP). All neural networks originally used in the competitions were custom-designed for the particular competition.
Hence, our analysis confirms that meaningful progress has been made in developing general-purpose architectures for tabular data.
Although the progress in the tabular data field is clearly visible, top performance cannot be reached without human effort for six datasets.

### 4.3 Feature Engineering is Still the Most Important Factor for Top Performance

![Refer to caption](/html/2407.02112/assets/x6.png)


Figure 6: Leaderboard performance gains from different modeling components per model. ’Default’ corresponds to the model with default hyperparameters. The results for Expert FE and FE-TTA are reported after extensively tuning hyperparameters.

The most remarkable performance gains are achieved through feature engineering.   
Figure [6](#S4.F6 "Figure 6 ‣ 4.3 Feature Engineering is Still the Most Important Factor for Top Performance ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows that expert feature engineering is the most important modeling component on average. This holds true for all models, indicating that unlike for modalities like imaging, neural networks do not automate feature engineering for tabular data.
When comparing the performance of different models in the standardized preprocessing pipeline (blue/orange/green bars in Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Modeling and Evaluation Framework ‣ 3 A Data-Centric Evaluation Framework for Tabular Machine Learning ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")), we can observe that using any other model than CatBoost rarely brings large gains. Only for the SCS dataset does FTTransformer clearly outperform all other models. For all other datasets, the average performance gains achievable solely with model selection are small.
Hence, our results confirm the findings of McElfresh et al., [[63](#bib.bib63)] that model selection is less important than HPO on a strong tree-based baseline for most datasets. Furthermore, we extend this finding by quantifying the even more important aspect of dataset-specific feature engineering.

Feature engineering is responsible for the high performance of CatBoost.   
Our analysis of different preprocessing pipelines reveals that CatBoost benefits much less from feature engineering than other models. The reason is that CatBoost incorporates explicit feature engineering techniques in its learning procedure.
In particular, counts and target-based statistics are used to generate encodings for categorical features, and combinatorial encoding methods capture categorical feature interactions [[70](#bib.bib70)].
When considering the same feature engineering techniques for the other models, the gap to CatBoost drastically shrinks for most models, and XGBoost performs similarly to CatBoost on average.
Hence, CatBoost’s success in recent benchmarking studies [[63](#bib.bib63), [28](#bib.bib28)] can, at least to some extent, be attributed to feature engineering.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | PSSDP | | BPCCM | |
|  | Default | OHE | Default | Target |
| XGBoost | 0.69 | 0.99 | 0.4 | 0.99 |
| LightGBM | 0.54 | 0.94 | 0.35 | 0.99 |
| CatBoost | 0.71 | 0.97 | 1.0 | 1.0 |

Table 2: Performance of tree-based models with different categorical data treatment methods. ’Default’ corresponds to the model-inherent method.

The optimal treatment of categorical features can be dataset-specific.   
Table [2](#S4.T2 "Table 2 ‣ 4.3 Feature Engineering is Still the Most Important Factor for Top Performance ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows that a different treatment of categorical features than the model-inherent treatment was necessary for two datasets to achieve top performance. Furthermore, each of the two datasets required a different encoding method.
This shows that standardized preprocessing can be biased for categorical features.
Hence, whenever the goal is not to evaluate models as AutoML solutions, categorical data treatment methods in comparative studies should not only be model-specific, but also dataset-specific.

### 4.4 The Importance of Test-Time Adaptation and Temporal Characteristics

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Best single model | | | AutoGluon | | |
|  | Stand. | FE | TTA | Stand. | FE | TTA |
| AEAC | 0.953 | 0.937 | 0.991 | 0.618 | 0.953 | 0.993 |
| OGPCC | 0.896 | 0.871 | 0.923 | 0.996 | 0.983 | 0.995 |
| SCS | 0.945 | 0.953 | 0.975 | 0.92 | 0.999 | 1.0 |
| SCTP | 0.518 | 0.962 | 0.992 | 0.498 | 0.531 | 0.991 |
| IFD | 0.662 | 0.988 | 0.992 | 0.205 | 0.351 | 0.432 |
| PSSDP | 0.656 | 0.994 | 0.995 | 0.562 | 0.707 | 0.742 |

Table 3: Performance comparison in different preprocessing pipelines with a focus on top performance. AutoGluon is displayed separately to prevent bias in the single-model comparison.

Test-time feature engineering consistently improves the performance of single models.
  
Table [3](#S4.T3 "Table 3 ‣ 4.4 The Importance of Test-Time Adaptation and Temporal Characteristics ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows that test-time feature engineering leads to performance gains over solely using the train data for feature engineering for all datasets.
From the task perspective, the feature engineering used for AEAC and OGPCC only leads to performance gains when used as a test-time adaptation method.
This shows that some of the feature engineering techniques used in Kaggle competitions actually serve the purpose of test-time adaptation.
For three datasets, ranking among the top 1% on the leaderboard was not achieved without test-time adaptation.
Our results indicate that simply comparing approaches to the Kaggle leaderboard, as done in previous studies [[25](#bib.bib25), [87](#bib.bib87)], is insufficient. Techniques like test-time adaptation are frequently used in Kaggle competitions and limit comparability to approaches that don’t use the test data. Hence, a fair model comparison to expert solutions using the Kaggle leaderboard can only be ensured under controlled conditions through implemented expert solutions such as our pipelines.

Models in real-world applications are often applied to non-i.i.d. tabular data.   
By definition, TTA should only be effective if the data violates the i.i.d. assumption and contains distribution shifts to adapt to.
Indeed, the data collection process likely happened over time for most of the datasets used in our framework. However, timestamps were not always provided as the competitions were conceptualized as static tabular data tasks.
Therefore, most of the datasets were also used in at least one comparative study for tabular data, although non-i.i.d. was a criterion for exclusion (e.g., SVPC, AEAC, and PSSDP in [[30](#bib.bib30)], SCTP and OGPCC in [[47](#bib.bib47)], or MBGM in [[35](#bib.bib35)]).
Our results show, that despite treating datasets as static, the samples remain non-i.i.d. and approaches like test-time adaptation can improve performance.
Furthermore, there is evidence that other datasets treated as i.i.d. in related work actually have a temporal nature. I.e., Kohli et al., [[47](#bib.bib47)] found that the most frequently used dataset in tabular data research is the forest cover type dataset [[12](#bib.bib12)]. At the same time, this dataset is used as a benchmark in online learning to measure the ability of models to adapt to concept drifts [[24](#bib.bib24), [73](#bib.bib73)].
As models for tabular data assume the data to be i.i.d., most benchmarks for evaluating tabular general-purpose models either directly name the data being non-i.i.d. as an exclusion criterion [[35](#bib.bib35), [29](#bib.bib29)] or exclude data that requires special CV procedures [[11](#bib.bib11)], which leads to the same results.
In contrast, our analysis of Kaggle competitions revealed that most tabular data competitions have temporal characteristics and that the best solutions for such datasets typically engineer time-invariant features and utilize tabular data models assuming the data to be i.i.d (i.e. [[3](#bib.bib3)]).
We conclude that there might be a mismatch between current evaluation frameworks for tabular data in academia and the tabular data tasks practitioners were interested in getting solved through ML competitions on Kaggle.

## 5 Implications for Future Work

We challenged the prevalent model-centric evaluation setups in tabular data research by comparing evaluations with standardized preprocessing pipelines to evaluations with expert preprocessing pipelines.
We have shown that current research is overly model-centric, while tabular datasets often require dataset-specific feature engineering or violate the i.i.d. assumption the models are based on. This reveals important insights and directions for future work in Machine Learning for tabular data.

Dinstinguish between AutoML and model comparisons.    Our findings highlight that standardized evaluation setups do not necessarily ensure fair model comparisons.
In standardized preprocessing setups, models are treated as AutoML solutions, whereas in real-world applications, they are components of highly dataset-specific pipelines.
Comparative studies should differentiate between datasets that benefit from known feature engineering steps and those that are self-sufficient. One approach could be to separate raw data benchmarks from fully preprocessed benchmarks, as done in our study. Standardized setups are more suitable for benchmarking AutoML solutions, while feature-engineered setups might be better for benchmarking models. Future research could emphasize incorporating dataset-specific (expert) preprocessing pipelines into benchmarks. However, gathering high-quality expert solutions at a large scale is tedious and may require a community effort.

Need for external performance references.    Our analysis shows that evaluations without considering the highest achievable performance on a task don’t actually measure the state-of-the-art. Despite numerous benchmarks, there is no established standard to measure progress. A benchmark with a public leaderboard and a dynamic collection of meaningful and unsolved real-world datasets could facilitate progress.

Improve feature engineering capabilities of models for tabular data. Researchers developing general-purpose models should recognize the impact of feature engineering on model performance. CatBoost has advanced the field by automating feature engineering on categorical data. However, significant feature engineering effort is still necessary for datasets where this is not the only challenge.
Future work should take a data-centric perspective and focus on automating successful feature engineering techniques through novel architecture components.
Our expert feature engineering pipelines can serve as a starting point for evaluating and developing new methods.
Furthermore, unlike previously claimed [[35](#bib.bib35)], categorical features can indeed be an important challenge for deep learning models. Hence, future work could focus on improvements over classic embeddings for categorical features in general-purpose deep learning architectures.

Methods for tabular data with temporal characteristics. Our analysis highlights the importance of distribution shifts in real-world tabular data, even when treating a task as static. Future work could investigate test-time adaptation methods specifically for tabular data, using our datasets and the identified test-time feature engineering techniques as baselines. Furthermore, our findings indicate that the current research focus on static i.i.d. data might hinder the development of techniques to handle weak temporal correlations in tabular data. Future work should focus on developing models with inductive biases for tabular data with temporal characteristics.

Align tabular benchmarks with practitioners needs.    We have shown that models developed for tabular data are often applied to datasets with temporal characteristics, while existing tabular data benchmarks are overly focused on i.i.d. data.
General-purpose tabular benchmarks should consider including tabular datasets with temporal characteristics instead of excluding them. Furthermore, a benchmark solely for tabular datasets with temporal characteristics could significantly advance model development for this relevant data problem.

## References

* Addison Howard, [2017]

  Addison Howard, AdrianoMoala, i. (2017).
  Porto seguro’s safe driver prediction.
* Addison Howard, [2019]

  Addison Howard, Bernadette Bouchon-Meunier, I. C. i. J. L. L. M. P. H. A. (2019).
  Ieee-cis fraud detection.
* Addison Howard, [2020]

  Addison Howard, inversion, S. M. v. (2020).
  M5 forecasting - accuracy.
* Akiba et al., [2019]

  Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019).
  Optuna: A next-generation hyperparameter optimization framework.
  In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2623–2631.
* Alexander Novy, [2017]

  Alexander Novy, CH1Mercedes, C. D. C. P. K. W. C. (2017).
  Mercedes-benz greener manufacturing.
* Anna Montoya, [2016]

  Anna Montoya, detoldim, D. L. D. S. C. W. C. (2016).
  Bnp paribas cardif claims management.
* Arik and Pfister, [2021]

  Arik, S. Ö. and Pfister, T. (2021).
  Tabnet: Attentive interpretable tabular learning.
  In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 6679–6687.
* Ben Hamner, [2013]

  Ben Hamner, kenmonta, W. C. (2013).
  Amazon.com - employee access challenge.
* Benjamin Bossan, [2015]

  Benjamin Bossan, Josef Feigl, W. K. (2015).
  Otto group product classification challenge.
* Bergstra et al., [2011]

  Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. (2011).
  Algorithms for hyper-parameter optimization.
  Advances in neural information processing systems, 24.
* Bischl et al., [2017]

  Bischl, B., Casalicchio, G., Feurer, M., Gijsbers, P., Hutter, F., Lang, M., Mantovani, R. G., van Rijn, J. N., and Vanschoren, J. (2017).
  Openml benchmarking suites.
* Blackard and Dean, [1999]

  Blackard, J. A. and Dean, D. J. (1999).
  Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables.
  Computers and electronics in agriculture, 24(3):131–151.
* Borisov et al., [2022]

  Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., and Kasneci, G. (2022).
  Deep neural networks and tabular data: A survey.
  IEEE Transactions on Neural Networks and Learning Systems.
* Bromberg, [1942]

  Bromberg, B. (1942).
  The origin of banking: religious finance in babylonia.
  The Journal of Economic History, 2(1):77–88.
* Cai et al., [2021]

  Cai, S., Zheng, K., Chen, G., Jagadish, H., Ooi, B. C., and Zhang, M. (2021).
  Arm-net: Adaptive relation modeling network for structured data.
  In Proceedings of the 2021 International Conference on Management of Data, pages 207–220.
* Cao, [2022]

  Cao, L. (2022).
  Ai in finance: challenges, techniques, and opportunities.
  ACM Computing Surveys (CSUR), 55(3):1–38.
* Chen et al., [2022]

  Chen, J., Liao, K., Wan, Y., Chen, D. Z., and Wu, J. (2022).
  Danets: Deep abstract networks for tabular data classification and regression.
  In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3930–3938.
* [18]

  Chen, J., Yan, J., Chen, D. Z., and Wu, J. (2023a).
  Excelformer: A neural network surpassing gbdts on tabular data.
  arXiv preprint arXiv:2301.02819.
* [19]

  Chen, K.-Y., Chiang, P.-H., Chou, H.-R., Chen, T.-W., and Chang, T.-H. (2023b).
  Trompt: Towards a better deep neural network for tabular data.
  arXiv preprint arXiv:2305.18446.
* Chen and Guestrin, [2016]

  Chen, T. and Guestrin, C. (2016).
  Xgboost: A scalable tree boosting system.
  In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794.
* Cherepanova et al., [2024]

  Cherepanova, V., Levin, R., Somepalli, G., Geiping, J., Bruss, C. B., Wilson, A. G., Goldstein, T., and Goldblum, M. (2024).
  A performance-driven benchmark for feature selection in tabular deep learning.
  Advances in Neural Information Processing Systems, 36.
* Cramer, [2002]

  Cramer, J. S. (2002).
  The origins of logistic regression.
* Darrel, [2015]

  Darrel, Stephen D Stayton, W. C. (2015).
  Homesite quote conversion.
* de Barros et al., [2016]

  de Barros, R. S. M., de Carvalho Santos, S. G. T., and Júnior, P. M. G. (2016).
  A boosting-like online learning ensemble.
  In 2016 international joint conference on neural networks (IJCNN), pages 1871–1878. IEEE.
* Erickson et al., [2020]

  Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. (2020).
  Autogluon-tabular: Robust and accurate automl for structured data.
  arXiv preprint arXiv:2003.06505.
* Falkner et al., [2018]

  Falkner, S., Klein, A., and Hutter, F. (2018).
  Bohb: Robust and efficient hyperparameter optimization at scale.
  In International conference on machine learning, pages 1437–1446. PMLR.
* Fischer et al., [2023]

  Fischer, S. F., Feurer, M., and Bischl, B. (2023).
  Openml-ctr23–a curated tabular regression benchmarking suite.
  In AutoML Conference 2023 (Workshop).
* Gardner et al., [2024]

  Gardner, J., Popovic, Z., and Schmidt, L. (2024).
  Benchmarking distribution shift in tabular data with tableshift.
  Advances in Neural Information Processing Systems, 36.
* Gijsbers et al., [2022]

  Gijsbers, P., Bueno, M. L., Coors, S., LeDell, E., Poirier, S., Thomas, J., Bischl, B., and Vanschoren, J. (2022).
  Amlb: an automl benchmark.
  arXiv preprint arXiv:2207.12560.
* Gijsbers et al., [2024]

  Gijsbers, P., Bueno, M. L., Coors, S., LeDell, E., Poirier, S., Thomas, J., Bischl, B., and Vanschoren, J. (2024).
  Amlb: an automl benchmark.
  Journal of Machine Learning Research, 25(101):1–65.
* Gijsbers et al., [2019]

  Gijsbers, P., LeDell, E., Thomas, J., Poirier, S., Bischl, B., and Vanschoren, J. (2019).
  An open source automl benchmark.
  arXiv preprint arXiv:1907.00909.
* Gorishniy et al., [2022]

  Gorishniy, Y., Rubachev, I., and Babenko, A. (2022).
  On embeddings for numerical features in tabular deep learning.
  Advances in Neural Information Processing Systems, 35:24991–25004.
* Gorishniy et al., [2023]

  Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., and Babenko, A. (2023).
  Tabr: Unlocking the power of retrieval-augmented tabular deep learning.
  arXiv preprint arXiv:2307.14338.
* Gorishniy et al., [2021]

  Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021).
  Revisiting deep learning models for tabular data.
  Advances in Neural Information Processing Systems, 34:18932–18943.
* Grinsztajn et al., [2022]

  Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022).
  Why do tree-based models still outperform deep learning on typical tabular data?
  Advances in Neural Information Processing Systems, 35:507–520.
* Guo et al., [2017]

  Guo, H., Tang, R., Ye, Y., Li, Z., and He, X. (2017).
  Deepfm: a factorization-machine based neural network for ctr prediction.
  arXiv preprint arXiv:1703.04247.
* Guo et al., [2021]

  Guo, X., Quan, Y., Zhao, H., Yao, Q., Li, Y., and Tu, W. (2021).
  Tabgnn: Multiplex graph neural network for tabular data prediction.
  arXiv preprint arXiv:2108.09127.
* Hollmann et al., [2022]

  Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2022).
  Tabpfn: A transformer that solves small tabular classification problems in a second.
  arXiv preprint arXiv:2207.01848.
* Hollmann et al., [2024]

  Hollmann, N., Müller, S., and Hutter, F. (2024).
  Large language models for automated data science: Introducing caafe for context-aware automated feature engineering.
  Advances in Neural Information Processing Systems, 36.
* Huang et al., [2020]

  Huang, X., Khetan, A., Cvitkovic, M., and Karnin, Z. (2020).
  Tabtransformer: Tabular data modeling using contextual embeddings.
  arXiv preprint arXiv:2012.06678.
* Johnson et al., [2016]

  Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L., and Mark, R. G. (2016).
  Mimic-iii, a freely accessible critical care database.
  Scientific data, 3(1):1–9.
* Kaggle, [2024]

  Kaggle (2024).
  Kaggle: Your home for data science.
  <https://www.kaggle.com>.
  Accessed: 2024-06-02.
* Ke et al., [2017]

  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017).
  Lightgbm: A highly efficient gradient boosting decision tree.
  Advances in neural information processing systems, 30.
* Ke et al., [2019]

  Ke, G., Xu, Z., Zhang, J., Bian, J., and Liu, T.-Y. (2019).
  Deepgbm: A deep learning framework distilled by gbdt for online prediction tasks.
  In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 384–394.
* Kim et al., [2023]

  Kim, C., Kim, T., Woo, S., Yang, J. Y., and Yang, E. (2023).
  Adaptable: Test-time adaptation for tabular data via shift-aware uncertainty calibrator and label distribution handler.
* Klein and Hutter, [2019]

  Klein, A. and Hutter, F. (2019).
  Tabular benchmarks for joint architecture and hyperparameter optimization.
  arXiv preprint arXiv:1905.04970.
* Kohli et al., [2024]

  Kohli, R., Feurer, M., Eggensperger, K., Bischl, B., and Hutter, F. (2024).
  Towards quantifying the effect of datasets for benchmarking: A look at tabular machine learning.
* Kossen et al., [2021]

  Kossen, J., Band, N., Lyle, C., Gomez, A. N., Rainforth, T., and Gal, Y. (2021).
  Self-attention between datapoints: Going beyond individual input-output pairs in deep learning.
  Advances in Neural Information Processing Systems, 34:28742–28756.
* Kundu et al., [2022]

  Kundu, J. N., Kulkarni, A. R., Bhambri, S., Mehta, D., Kulkarni, S. A., Jampani, V., and Radhakrishnan, V. B. (2022).
  Balancing discriminability and transferability for source-free domain adaptation.
  In International conference on machine learning, pages 11710–11728. PMLR.
* Lee et al., [2022]

  Lee, J., Jung, D., Yim, J., and Yoon, S. (2022).
  Confidence score for source-free unsupervised domain adaptation.
  In International conference on machine learning, pages 12365–12377. PMLR.
* Li et al., [2018]

  Li, H., Pan, S. J., Wang, S., and Kot, A. C. (2018).
  Domain generalization with adversarial feature learning.
  In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5400–5409.
* Li et al., [2020]

  Li, J., Li, Y., Xiang, X., Xia, S.-T., Dong, S., and Cai, Y. (2020).
  Tnt: An interpretable tree-network-tree learning framework using knowledge distillation.
  Entropy, 22(11):1203.
* Lian et al., [2018]

  Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., and Sun, G. (2018).
  xdeepfm: Combining explicit and implicit feature interactions for recommender systems.
  In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1754–1763.
* Liu et al., [2023]

  Liu, M., Guo, C., and Guo, S. (2023).
  An explainable knowledge distillation method with xgboost for icu mortality prediction.
  Computers in Biology and Medicine, 152:106466.
* Liu et al., [2021]

  Liu, Y., Kothari, P., Van Delft, B., Bellot-Gurlet, B., Mordan, T., and Alahi, A. (2021).
  Ttt++: When does self-supervised test-time training fail or thrive?
  Advances in Neural Information Processing Systems, 34:21808–21820.
* Loshchilov and Hutter, [2017]

  Loshchilov, I. and Hutter, F. (2017).
  Decoupled weight decay regularization.
  arXiv preprint arXiv:1711.05101.
* Luo et al., [2020]

  Luo, Y., Zhou, H., Tu, W.-W., Chen, Y., Dai, W., and Yang, Q. (2020).
  Network on network for tabular data classification in real-world applications.
  In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2317–2326.
* Malinin et al., [2021]

  Malinin, A., Band, N., Chesnokov, G., Gal, Y., Gales, M. J., Noskov, A., Ploskonosov, A., Prokhorenkova, L., Provilkov, I., Raina, V., et al. (2021).
  Shifts: A dataset of real distributional shift across multiple large-scale tasks.
  arXiv preprint arXiv:2107.07455.
* Mao et al., [2023]

  Mao, K., Zhu, J., Su, L., Cai, G., Li, Y., and Dong, Z. (2023).
  Finalmlp: An enhanced two-stream mlp model for ctr prediction.
  arXiv preprint arXiv:2304.00902.
* Mark McDonald, [2018]

  Mark McDonald, Mercedes Piedra, S. D. S. (2018).
  Santander value prediction challenge.
* Marton et al., [2023]

  Marton, S., Lüdtke, S., Bartelt, C., and Stuckenschmidt, H. (2023).
  Grande: Gradient-based decision tree ensembles.
  arXiv preprint arXiv:2309.17130.
* McCulloch and Pitts, [1943]

  McCulloch, W. S. and Pitts, W. (1943).
  A logical calculus of the ideas immanent in nervous activity.
  The bulletin of mathematical biophysics, 5:115–133.
* McElfresh et al., [2023]

  McElfresh, D., Khandagale, S., Valverde, J., Ramakrishnan, G., Goldblum, M., White, C., et al. (2023).
  When do neural nets outperform boosted trees on tabular data?
  arXiv preprint arXiv:2305.02997.
* Mercedes Piedra, [2019]

  Mercedes Piedra, Sohier Dane, S. (2019).
  Santander customer transaction prediction.
* Müller et al., [2023]

  Müller, A., Curino, C., and Ramakrishnan, R. (2023).
  Mothernet: A foundational hypernetwork for tabular classification.
  arXiv preprint arXiv:2312.08598.
* Niu et al., [2022]

  Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., and Tan, M. (2022).
  Efficient test-time model adaptation without forgetting.
  In International conference on machine learning, pages 16888–16905. PMLR.
* Niu et al., [2023]

  Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., and Tan, M. (2023).
  Towards stable test-time adaptation in dynamic wild world.
  arXiv preprint arXiv:2302.12400.
* Pedregosa et al., [2011]

  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011).
  Scikit-learn: Machine learning in python.
  the Journal of machine Learning research, 12:2825–2830.
* Popov et al., [2019]

  Popov, S., Morozov, S., and Babenko, A. (2019).
  Neural oblivious decision ensembles for deep learning on tabular data.
  arXiv preprint arXiv:1909.06312.
* Prokhorenkova et al., [2018]

  Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018).
  Catboost: unbiased boosting with categorical features.
  Advances in neural information processing systems, 31.
* Roelofs et al., [2019]

  Roelofs, R., Shankar, V., Recht, B., Fridovich-Keil, S., Hardt, M., Miller, J., and Schmidt, L. (2019).
  A meta-analysis of overfitting in machine learning.
  Advances in Neural Information Processing Systems, 32.
* Rumelhart et al., [1986]

  Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986).
  Learning representations by back-propagating errors.
  nature, 323(6088):533–536.
* Sadeghi et al., [2023]

  Sadeghi, F., Viktor, H. L., and Vafaie, P. (2023).
  Dynaq: online learning from imbalanced multi-class streams through dynamic sampling.
  Applied Intelligence, 53(21):24908–24930.
* Shi et al., [2021]

  Shi, X., Mueller, J., Erickson, N., Li, M., and Smola, A. J. (2021).
  Benchmarking multimodal automl for tabular data with text fields.
  arXiv preprint arXiv:2111.02705.
* Shwartz-Ziv and Armon, [2022]

  Shwartz-Ziv, R. and Armon, A. (2022).
  Tabular data: Deep learning is not all you need.
  Information Fusion, 81:84–90.
* Sinha et al., [2023]

  Sinha, S., Gehler, P., Locatello, F., and Schiele, B. (2023).
  Test: Test-time self-training under distribution shift.
  In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2759–2769.
* Somepalli et al., [2021]

  Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., and Goldstein, T. (2021).
  Saint: Improved neural networks for tabular data via row attention and contrastive pre-training.
  arXiv preprint arXiv:2106.01342.
* Soraya Jimenez, [2016]

  Soraya Jimenez, W. C. (2016).
  Santander customer satisfaction.
* Sun et al., [2019]

  Sun, B., Yang, L., Zhang, W., Lin, M., Dong, P., Young, C., and Dong, J. (2019).
  Supertml: Two-dimensional word embedding for the precognition on structured tabular data.
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0.
* Sun et al., [2020]

  Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., and Hardt, M. (2020).
  Test-time training with self-supervision for generalization under distribution shifts.
  In International conference on machine learning, pages 9229–9248. PMLR.
* Tunguz et al., [2023]

  Tunguz, B., Dieter, or Tails, H., Kapoor, K., Pandey, P., Mooney, P., Culliton, P., Mulla, R., Bhutani, S., and Cukierski, W. (2023).
  2023 kaggle ai report.
* Van Dalen, [1993]

  Van Dalen, B. (1993).
  Ancient and Mediaeval Astronomical Tables: mathematical structure and parameter values.
  Universiteit Utrecht, Faculteit Wiskunde en Informatica.
* Wang and Sun, [2022]

  Wang, Z. and Sun, J. (2022).
  Transtab: Learning transferable tabular transformers across tables.
  Advances in Neural Information Processing Systems, 35:2902–2915.
* Xu et al., [2020]

  Xu, M., Zhang, J., Ni, B., Li, T., Wang, C., Tian, Q., and Zhang, W. (2020).
  Adversarial domain adaptation with domain mixup.
  In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 6502–6509.
* Yan et al., [2023]

  Yan, J., Chen, J., Wu, Y., Chen, D. Z., and Wu, J. (2023).
  T2g-former: organizing tabular features into relation graphs promotes heterogeneous feature interaction.
  In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 10720–10728.
* Yan et al., [2020]

  Yan, S., Song, H., Li, N., Zou, L., and Ren, L. (2020).
  Improve unsupervised domain adaptation with mixup training.
  arXiv preprint arXiv:2001.00677.
* [87]

  Zhang, T., Zhang, Z. A., Fan, Z., Luo, H., Liu, F., Liu, Q., Cao, W., and Jian, L. (2023a).
  Openfe: Automated feature generation with expert-level performance.
  In International Conference on Machine Learning, pages 41880–41901. PMLR.
* [88]

  Zhang, Y., Safdar, M., Xie, J., Li, J., Sage, M., and Zhao, Y. F. (2023b).
  A systematic review on data of additive manufacturing for machine learning applications: the data quality, type, preprocessing, and management.
  Journal of Intelligent Manufacturing, 34(8):3305–3340.
* Zhao et al., [2023]

  Zhao, H., Liu, Y., Alahi, A., and Lin, T. (2023).
  On pitfalls of test-time adaptation.
  In International Conference on Machine Learning, pages 42058–42080. PMLR.
* [90]

  Zhou, K., Liu, Z., Chen, R., Li, L., Choi, S.-H., and Hu, X. (2022a).
  Table2graph: Transforming tabular data to unified weighted graph.
  In IJCAI, pages 2420–2426.
* [91]

  Zhou, K., Liu, Z., Qiao, Y., Xiang, T., and Loy, C. C. (2022b).
  Domain generalization: A survey.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4396–4415.
* Zhu et al., [2021]

  Zhu, Y., Brettin, T., Xia, F., Partin, A., Shukla, M., Yoo, H., Evrard, Y. A., Doroshow, J. H., and Stevens, R. L. (2021).
  Converting tabular data into images for deep learning with convolutional neural networks.
  Scientific reports, 11(1):11325.
* Zindi, [2024]

  Zindi (2024).
  <https://zindi.africa>.
  Accessed: 2024-06-11.

## Appendix A Datasets and Expert Solutions

In this Section, we provide more details on the dataset/competition selection process and the expert solutions implemented in our framework. Table [4](#A1.T4 "Table 4 ‣ A.1 Dataset Selection ‣ Appendix A Datasets and Expert Solutions ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows the name of all Kaggle competitions included in our framework as well as a hyperlink for the implemented expert feature engineering.

### A.1 Dataset Selection

The main paper already provides an overview of the main characteristics of the selected datasets and our main selection criteria. In this Subsection we further explain the selection criteria and summarize the excluded datasets.

| Name | Competition Name | Implemented Expert Solution |
| --- | --- | --- |
| MBGM | mercedes-benz-greener-manufacturing | [1st place solution](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/discussion/37700) |
| SVPC | santander-value-prediction-challenge | [6th place solution](https://www.kaggle.com/competitions/santander-value-prediction-challenge/discussion/63919) |
| AEAC | amazon-employee-access-challenge | [1st place solution](https://www.kaggle.com/competitions/amazon-employee-access-challenge/discussion/5283) |
| OGPCC | otto-group-product-classification-challenge | [8th place solution](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/discussion/14295) |
| SCS | santander-customer-satisfaction | [3rd place solution](https://www.kaggle.com/competitions/santander-customer-satisfaction/discussion/20978) |
| BPCCM | bnp-paribas-cardif-claims-management | [8th place solution](https://www.kaggle.com/code/confirm/xfeat-catboost-cpu-only) |
| SCTP | santander-customer-transaction-prediction | [1st place solution](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003) |
| HQC | homesite-quote-conversion | [15th place solution](https://www.kaggle.com/competitions/homesite-quote-conversion/discussion/18831) |
| IFD | ieee-fraud-detection | [1st place solution](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111308) |
| PSSDP | porto-seguro-safe-driver-prediction | [2nd place solution](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44558) |

Table 4: Datasets and expert solutions included in our framework. The competitions can be accessed at https://www.kaggle.com/competitions/{Competition Name}.

In an initial search through the competitions hosted on Kaggle, we selected all datasets that satisfied the following criteria:

* •

  Tabular: We only consider competitions that include tabular data.
* •

  Popular competitions: We consider all competitions with at least 1000 participants.
* •

  Additional incentive: We only consider competitions that are incentivized, either monetarily or otherwise.

We identified 77 competitions that satisfy these criteria and further applied dataset-specific criteria to select competitions for our framework. Table LABEL:fig:excluded\_datasets summarizes all excluded datasets. In the following, we explain the exclusion criteria:

* •

  Technical Issues:

  + –

    Code competition: Automating solution submission is non-trivial since the competition is a code competition with special requirements. (3 competitions)
  + –

    Ongoing: The home-credit-credit-risk-model-stability competition was not finished at the time of the development.
  + –

    Availability: Dataset not available anymore (6 competitions)
  + –

    Sample size: The restaurant-revenue-prediction dataset had only 137 training samples, preventing reliable non-random model comparisons.
  + –

    Leak: The competition was won through an unresolvable data leak s.t. a fair evaluation is impossible. (3 competitions)
  + –

    Submission error: Submitting to Kaggle doesn’t work due to an unresolved error for the liberty-mutual-group-property-inspection-prediction competition.
* •

  Other Modality: Utilization of other modalities, e.g. images, text, signals, molecular, or genetic data, was a major part of the expert solution besides tabular data. (10 competitions)
* •

  Special domain:

  + –

    Spatial: The data has spatial correlations that cannot easily be learned by the existing general-purpose models. (4 competitions)
  + –

    Recommendation: Click-through-rate prediction and recommendation tasks were excluded because dedicated models exist for these tasks (e.g., [[36](#bib.bib36), [53](#bib.bib53), [59](#bib.bib59)]), while we focus on general-purpose models. (avazu-ctr-prediction and kkbox-music-recommendation-challenge)
* •

  Temporal: In line with related work, we focus on i.i.d. tabular data. Competitions where time-sensitive feature engineering was the key to competition success were excluded. This also includes competitions without an explicit timestamp where multiple tables needed to be merged, and where the strategy for merging datasets was a relevant part of the solution, e.g., due to specific aggregation strategies. Although no timestamps are available for those datasets, the underlying task necessitating merging was temporal. An example is the elo-merchant-category-recommendation competition. (32 competitions)
* •

  Expert Solution availability/reproducibility: These datasets could be included in our framework but, for various reasons, could not be used with different preprocessing pipelines:

  + –

    For the walmart-recruiting-trip-type-classification, no top 1% solution was available.
  + –

    For the Springleaf-marketing-response and the ClaimPredictionChallenge datasets, solution descriptions were available but were insufficient to reproduce the solution.
  + –

    For the prudential-life-insurance-assessment dataset, the main aspect for high performance was calibration and transforming the target to simplify calibration, which was out-of-scope in our framework.
  + –

    For two competitions, the expert solution mainly consisted of heavy ensembling on different dataset versions and models, which we could not reproduce within our setup (allstate-claims-severity, higgs-boson).
  + –

    For the sberbank-russian-housing-market competition, high performance was mainly achieved by training different models for different samples in a dataset and by modifying the target in highly task-specific ways.

Table 5: Datasets excluded during the selection process

| Competition Name | No. Teams | Exclusion criteria |
| --- | --- | --- |
| home-credit-default-risk | 7176 | Temporal |
| icr-identify-age-related-conditions | 6430 | Code competition |
| m5-forecasting-accuracy | 5558 | Temporal |
| amex-default-prediction | 4874 | Temporal |
| LANL-Earthquake-Prediction | 4516 | Other Modality, Temporal |
| optiver-trading-at-the-close | 4436 | Temporal |
| lish-moa | 4373 | Code competition |
| jane-street-market-prediction | 4245 | Availability, Temporal |
| elo-merchant-category-recommendation | 4110 | Temporal |
| talkingdata-adtracking-fraud-detection | 3943 | Temporal |
| optiver-realized-volatility-prediction | 3852 | Temporal |
| zillow-prize-1 | 3770 | Spatial |
| ashrae-energy-prediction | 3614 | Temporal |
| ga-customer-revenue-prediction | 3611 | Temporal |
| godaddy-microbusiness-density-forecasting | 3547 | Temporal |
| petfinder-pawpularity-score | 3537 | Other Modality |
| home-credit-credit-risk-model-stability | 3481 | Ongoing |
| rossmann-store-sales | 3298 | Temporal |
| sberbank-russian-housing-market | 3264 | No expert solution |
| allstate-claims-severity | 3045 | No expert solution |
| h-and-m-personalized-fashion-recommendations | 2952 | Other Modality, Temporal |
| two-sigma-financial-news | 2927 | Availability, Temporal |
| ubiquant-market-prediction | 2893 | Availability, Temporal |
| champs-scalar-coupling | 2737 | Other Modality |
| predict-energy-behavior-of-prosumers | 2731 | Temporal |
| instacart-market-basket-analysis | 2621 | Temporal |
| prudential-life-insurance-assessment | 2610 | No expert solution |
| otto-recommender-system | 2574 | Temporal |
| novozymes-enzyme-stability-prediction | 2482 | Other Modality |
| two-sigma-connect-rental-listing-inquiries | 2480 | Other Modality, Temporal |
| microsoft-malware-prediction | 2410 | Temporal |
| mercari-price-suggestion-challenge | 2380 | Other Modality |
| predicting-red-hat-business-value | 2260 | Leak, Temporal |
| restaurant-revenue-prediction | 2257 | Sample size |
| liberty-mutual-group-property-inspection-prediction | 2232 | Submission error |
| springleaf-marketing-response | 2221 | No expert solution |
| recruit-restaurant-visitor-forecasting | 2148 | Temporal |
| home-depot-product-search-relevance | 2123 | Leak |
| two-sigma-financial-modeling | 2063 | Availability, Temoral |
| predict-student-performance-from-game-play | 2051 | Temporal |
| jpx-tokyo-stock-exchange-prediction | 2033 | Temporal |
| petfinder-adoption-prediction | 2023 | Other Modality |
| expedia-hotel-recommendations | 1971 | Spatial |
| grupo-bimbo-inventory-demand | 1963 | Temporal |
| g-research-crypto-forecasting | 1946 | Temporal |
| avito-demand-prediction | 1868 | Other Modality, Temporal |
| amp-parkinsons-disease-progression-prediction | 1805 | Code competition, Temporal |
| higgs-boson | 1784 | No expert solution |
| santander-product-recommendation | 1779 | Temporal |
| talkingdata-mobile-user-demographics | 1680 | Leak |
| favorita-grocery-sales-forecasting | 1671 | Temporal |
| avazu-ctr-prediction | 1602 | Recommendation |
| allstate-purchase-prediction-challenge | 1566 | Temporal |
| axa-driver-telematics-analysis | 1524 | Availability, Temporal |
| new-york-city-taxi-fare-prediction | 1483 | Spatial |
| airbnb-recruiting-new-user-bookings | 1458 | Temporal |
| vsb-power-line-fault-detection | 1445 | Temporal |
| bosch-production-line-performance | 1370 | Temporal |
| hhp | 1350 | Availability |
| predict-west-nile-virus | 1304 | Temporal |
| ClaimPredictionChallenge | 1278 | No expert solution |
| nyc-taxi-trip-duration | 1254 | Spatial, Temporal |
| PLAsTiCC-2018 | 1089 | Temporal |
| kkbox-music-recommendation-challenge | 1081 | Recommendation |
| foursquare-location-matching | 1079 | Other Modality |
| coupon-purchase-prediction | 1072 | Temporal |
| walmart-recruiting-trip-type-classification | 1043 | No expert solution |

### A.2 Implemented Components of Expert Solutions

In this Subsection, we document the components of our framework that were directly derived from expert solutions.

Task conceptualization in data loading.   
For all datasets, the data loading includes merging tables, defining the target, and defining categorical features. For some datasets, we incorporated parts of expert solutions into the task conceptualization as a part of the data-loading function:

* •

  mercedes-benz-greener-manufacturing: The index is used as a numeric feature as it was necessary to score top leaderboard ranks.
* •

  santander-value-prediction-challenge: 1) The target is marked as heavy-tailed to be transformed in the standardized preprocessing pipeline. 2) There was a data leak allowing to derive the test targets for some samples. The top expert solutions used these samples for data augmentation. Hence, we also moved these samples from the test to the training dataset, s.t. this leak is not an issue for any of our pipelines.
* •

  homesite-quote-conversion: Extract weekday from datetime feature.
* •

  porto-seguro-safe-driver-prediction: Replace -1 with nan.

Feature Engineering Pipelines.   
For each expert solution, we extract the data preparation, which mostly consisted of feature engineering. The expert solutions of the datasets contained the following feature engineering operations:

* •

  mercedes-benz-greener-manufacturing: Addition of binary features, logical\_and of binary features, sum of multiple binary features, feature selection
* •

  santander-value-prediction-challenge: {max, mean, min, median, first nonzero, last nonzero, no. of nans, no. of unique values} of groups of multiple features. The groups mostly consisted either of 40 or 99 features. Three groups were formed with 4991, 991, and 4000 features. The groups were previously determined based on expert knowledge. However, all operations to obtain the new features could theoretically be learned solely from the train data, and no timestamps are given explicitly.
* •

  amazon-employee-access-challenge: (normalized) groupby interactions, 2- and 3-order categorical interactions, (normalized) frequency encoding, frequency encoding of interactions, log of frequency features, drop constant features
* •

  otto-group-product-classification-challenge: tSNE features, PCA features, KMeans centroid features
* •

  santander-customer-satisfaction: a few data cleaning steps, Remove highly correlated and constant features, remove features with <4 target=1 instances, count of value 0/3/6/9 in a row, percentile rank of feature A within feature B (considered a special kind of groupby interaction), ratios, (X mod 3) == 0, KMeans features with 2-11 clusters, binary feature separating population based on different other feature values
* •

  bnp-paribas-cardif-claims-management: 2- and 3-order categorical interactions, Convert numerical to categorical by rounding, 2-order Arithmetic combinations, 11-order categorical interaction, out-of-fold target encoding
* •

  santander-customer-transaction-prediction: replacing values that are unique in train data (added test for test-time adaptation) with the mean of the feature, Extract categorical features from numeric. Features have four (five if test data used) categories: 1) value appears at least another time in data with target==1 and no 0, 2) value appears at least another time in data with target==0 and no 1, 3) value appears at least two more time in data with target==0 & 1, 4) value is unique in data (if test-time adaptation: 5) value is unique in data + test)
* •

  homesite-quote-conversion: sum NAs in a row, sum of zeros in a row, two-order categorical interaction
* •

  ieee-fraud-detection: feature selection, normalize "time deltas" from some point in the past (Feature 1 (F1)-Feature 2 (F2)/(24\*60\*60)), frequency encoding (train & test), label encode categoricals, groupby interactions (mean, std, count), 2-way categorical interactions, (F1 - floor(F2), F1(cat) + ascat(floor(F2)-F3) - is not used directly, but for more aggregations), abs(F1-F2)>3, use cat features as numeric
* •

  porto-seguro-safe-driver-prediction: Feature selection, sum of missing values, frequency encoding of high-order interaction of categorical features, only for tree-based models: one-hot-encoding of categorical features, only for neural networks: train XGBoost models with one group of features as input and another feature as output - use the out-of-fold predictions as features

For the ieee-fraud-detection competition the winning solution found that once one transaction of a customer is a fraud in the train dataset - all are. They deal with that by implementing a postprocessing function labeling all customers as a fraud whenever one of the transactions is a fraud. As this pattern could also be learned by models, we decided to treat this aspect as part of the expert preprocessing pipeline.

The treatment of categorical features is left to the models whenever possible. The utilized encoding is listed as part of the expert feature engineering pipeline for datasets where the categorical data treatment was crucial for high performance.
Operations that add new features based on existing categorical features (e.g., frequency encoding) are always considered part of the expert preprocessing pipeline, even though models like CatBoost use this information natively.
Similarly, we remove the treatment of missing values from the expert pipelines and leave that to the respective models whenever possible.

The following feature engineering techniques were applied most frequently over all datasets: groupby interactions (4), two-order categorical interactions (3), feature selection (3), categorical frequency encoding (3), dimensionality reduction (2), three-order categorical interaction (2), 2-order arithmetic interactions (2), sum of missing values in a row (2), and sum of zeros in a row (2). Details on all implemented techniques for the datasets can be found in our code.

Feature Engineering Techniques used for test-time adaptation.   
Of the abovementioned feature engineering techniques, the following were utilized as test-time feature engineering techniques:

* •

  Counts of categorical features (AEAC, IFD, PSSDP)
* •

  Dimensionality reduction (OGPCC, SCS)
* •

  Groupby interactions (SCS, IFD)
* •

  Occurrence of numeric values from train data in the test data (SCTP)
* •

  Model-based Denoising/Smoothing by training an XGBoost model to predict features and using out-of-fold predictions as features (PSSDP)

Cross-validation procedures.   
We used the same CV split type for each dataset as the expert solutions but unified the number of folds across all our datasets. We used 10 folds for most datasets, as this worked well for all datasets. We used 10-fold cross-validation for all but one dataset. For classification tasks, the folds were stratified across the target. For the IFD dataset, we split the data based on the month the data was collected, which resulted in six folds.
For faster training, fewer folds could be used for large datasets with similar results. For large datasets, most expert solutions used fewer folds, e.g., for the PSSDP competition.

### A.3 Discussion on Test-Time Feature Engineering

To deal with distribution shifts, test-time adaptation is a conceptual framework where the model parameters are allowed to depend on the test sample x𝑥x but not on its unknown label y𝑦y.
This matches the common ML competition setup, where test samples are given, but the target is hidden. We found that successful participants in Kaggle competitions often use the test data for feature engineering.
Hence, we established that using test data for feature engineering in Kaggle competitions can be considered a special kind of test-time adaptation.
This subsection discusses when this practice can be considered for real-world applications and when it is an unfair and unrealistic setup for a task.

We argue that the common ML setup allowing for test-time adaptation corresponds to a frequent real-world application scenario where 1) The data to predict arrives in batches, 2) No real-time predictions are required, and 3) Retraining the model at test time is feasible.
The first criterion is important as the employed test-time feature engineering techniques (e.g. dimensionality reduction and frequency encoding) often required the presence of many test samples at once. It is unclear whether this kind of domain adaptation would work per sample. The second and third criteria are necessary as test-time feature engineering always requires retraining the model, which is infeasible in online applications. In contrast, test-time feature engineering is not applicable in applications where online predictions are required, the number of test samples is small, or not retrainable (e.g., large-scale) models are used. Importantly, while test-time feature engineering might be infeasible in such applications, other test-time adaptation techniques might still apply.
One example of such a task conceptualization amenable to test-time feature engineering is product return prediction, where samples are collected over a day, and a (lightweight) model can be retrained daily. In this scenario, using the test data in an unsupervised fashion for better adaptation to possible distribution shifts is feasible.
After examining the application scenarios of the tasks in our framework, we found that most of them, like customer transaction prediction (SCTP) and customer satisfaction prediction (SCS), would allow such a setup, although likely with smaller amounts of test data than used for the competitions.
Furthermore, our discussion in Subsection [4.4](#S4.SS4 "4.4 The Importance of Test-Time Adaptation and Temporal Characteristics ‣ 4 Experimental Evaluation ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") reveals that many tabular datasets not in our scope have temporal components and thus may be amenable to test-time feature engineering.

## Appendix B Experimental Details

In this Section, we discuss all aspects of our experiments that are not a part of our proposed evaluation framework but rather are design choices we made for our experiments.

### B.1 Software and Hardware

The deep learning models, CatBoost, and XGBoost, were trained using one or more of the following GPU hardware, depending on the availability: NVIDIA H100, NVIDIA A100, NVIDIA RTX A6000, or NVIDIA A40. LightGBM and AutoGluon were trained using the following CPU hardware: Intel(R) Xeon(R) CPU E2640v2 @ 2,00 GHz; Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz.

### B.2 Model-Specific Preprocessing

For the tree-based models, model-specific preprocessing only included the correct assignment of datatypes to categorical features. For the deep learning models, the preprocessing was defined in line with the related work [[34](#bib.bib34), [35](#bib.bib35)]. For regression, the target is normalized to zero mean and unit variance. For numeric features, missing values are replaced with the mean, and the features are normalized using ScikitLearn’s QuantileTransformer [[68](#bib.bib68)]. Categorical features are ordinally encoded as ResNet, FTTransformer, and MLP-PLR use embeddings for categorical features. The GRANDE library includes its own preprocessing, which contains the same steps as for the other deep learning models but uses leave-one-out-encoding for categorical features. For AutoGluon, all the preprocessing is left to the AutoML framework.

### B.3 Model Training and Hyperparameter Optimization

We use the optuna library [[4](#bib.bib4)] for hyperparameter optimization. Each model is optimized for 100 trials with the first 20 trials being random search trials and the remaining 80 using the multivariate Tree-structured Parzen Estimator algorithm [[10](#bib.bib10), [26](#bib.bib26)].
The models are trained using cross entropy for classification and mean squared error for regression. The AdamW optimizer is used for training the deep learning models [[56](#bib.bib56)].
Whenever possible, we use the task metric for validation during model training and for choosing the best hyperparameters. Instead of the R2 metric, we use rmse, as the objective is the same. Moreover, we we use rmse whenever the metric is rmsle, as we already transformed the target prior to fitting. Instead of Gini, we use AUC, as the metrics are convertible.
For AutoGluon, we use the ’best\_quality’ preset configuration and a time limit of 10 hours. Everything else is left to the AutoML library itself.
We try to use default hyperparameters and tuning ranges that have been shown to perform well for each model. For that, we orient on different related work [[34](#bib.bib34), [35](#bib.bib35), [32](#bib.bib32), [63](#bib.bib63), [61](#bib.bib61)] and the library documentations.
Some of the datasets we use are quite large compared to most related work. Our goal was to evaluate each of the included models with an equal number of hyperparameter trials. Therefore, we did not use time budgets to constrain the number of trials per model and dataset. As this leads to long computation times for some models, we did not tune the representation capacity parameters for FTTransformer, as it was the most time-expensive model in our scope.
All default hyperparameters and search spaces can be seen in Tables [6](#A2.T6 "Table 6 ‣ B.3 Model Training and Hyperparameter Optimization ‣ Appendix B Experimental Details ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data")-[12](#A2.T12 "Table 12 ‣ B.3 Model Training and Hyperparameter Optimization ‣ Appendix B Experimental Details ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data").

| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| n\_estimators | 4000 | - |
| patience | 200 | - |
| learning\_rate | 0.3 | LogUniform[1e-3, 0.7] |
| max\_depth | 6 | UniformInt[1, 11] |
| colsample\_bytree | 1. | Uniform[0.5,1.] |
| subsample | 1. | Uniform[0.5,1.] |
| min\_child\_weight | 1. | LogUniform[1, 100] |
| reg\_alpha | 0. | LogUniform[1e-8, 100] |
| reg\_lambda | 1. | LogUniform[1, 4] |
| gamma | 0. | LogUniform[1e-8, 7] |

Table 6: Hyperparameter configurations for XGBoost.



| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| iterations | 4000 | - |
| patience | 200 | - |
| learning\_rate | 0.1 | LogUniform[1e-3, 0.7] |
| max\_depth | -1 | {-1, UniformInt[1, 11]} |
| min\_data\_in\_leaf | 20 | {20, 50, 100, 500, 1000, 2000} |
| num\_leaves | 31 | UniformInt[2, 2047] |
| lambda\_l2 | 0. | LogUniform[1e-4, 10.] |
| feature\_fraction | 1. | Uniform[0.5, 1.] |
| bagging\_fraction | 1. | Uniform[0.5, 1.] |
| min\_sum\_hessian\_in\_leaf | 1e-3 | LogUniform[1e-4,100.0] |

Table 7: Hyperparameter configurations for LightGBM. If max\_depth>=1, the possible num\_leaves ranges were adjusted to be in a space feasible with the respective depth.



| Hyperparameter | Default | Search Distribution |
| --- | --- | --- |
| iterations | 4000 | - |
| od\_type | "Iter" | - |
| od\_wait | 200 | - |
| learning\_rate | auto | LogUniform[1e-3, 1.] |
| max\_depth | 6 | UniformInt[1, 11] |
| l2\_leaf\_reg | 3.0 | LogUniform[1,30] |
| bagging\_temperature | 1 | Uniform[0,1] |

Table 8: Hyperparameter configurations for CatBoost. In the default setting, the library automatically determines a dataset-specific learning rate.



| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| epochs | 200 | - |
| patience | 5 | - |
| batch\_size | 128 | - |
| learning\_rate | 1e-4 | LogUniform[1e-5, 1e-2] |
| weight\_decay | 1e-5 | LogUniform[1e-6, 1e-3] |
| # Layers | 2 | UniformInt[1, 8] |
| Layer size | 192 | UniformInt[64, 1024] |
| Hidden factor | 2. | Uniform[1, 4] |
| Hidden dropout | 0.25 | Uniform[0., 0.5] |
| Residual dropout | 0. | Uniform[0., 0.5] |
| Categorical embedding size | 8 | UniformInt[4, 512] |

Table 9: Hyperparameter configurations for ResNet.



| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| epochs | 200 | - |
| patience | 5 | - |
| batch\_size | 128 | - |
| learning\_rate | 1e-4 | LogUniform[1e-5, 1e-3] |
| weight\_decay | 1e-5 | LogUniform[1e-6, 1e-3] |
| # Layers | 3 | - |
| Layer size | 192 | - |
| # Attention heads | 8 | - |
| Hidden factor | 4343\frac{4}{3} | - |
| Hidden dropout | 0.1 | Uniform[0., 0.5] |
| Attention dropout | 0.2 | Uniform[0., 0.5] |
| Residual dropout | 0. | Uniform[0., 0.2] |
| Categorical embedding size | 8 | - |

Table 10: Hyperparameter configurations for FTTransformer. Note that weight decay is only applied to some layers of the model. For details, see [[34](#bib.bib34)].



| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| epochs | 200 | - |
| patience | 5 | - |
| batch\_size | 128 | - |
| learning\_rate | 1e-3 | LogUniform[5e-5, 5e-3] |
| weight\_decay | 1e-4 | LogUniform[1e-6, 1e-3] |
| # Layers | 2 | UniformInt[1, 8] |
| Layer size | 192 | UniformInt[1, 1024] |
| Categorical embedding size | 8 | UniformInt[1, 128] |
| Numerical embedding size | 8 | UniformInt[1, 128] |
| Dropout | 0.25 | Uniform[0., 0.5] |
| frequency\_init\_scale | 0. | LogUniform[1e-2, 10.] |

Table 11: Hyperparameter configurations for MLP-PLR.



| Hyperparameter | Default | Search distribution |
| --- | --- | --- |
| epochs | 1000 | - |
| patience | 25 | - |
| batch\_size | 64 | - |
| depth | 5 | - |
| n\_estimators | 2048 | - |
| learning\_rate\_weights | 0.005 | LogUniform[1e-4, 0.25] |
| learning\_rate\_index | 0.01 | LogUniform[1e-4, 0.25] |
| learning\_rate\_values | 0.01 | LogUniform[1e-4, 0.25] |
| learning\_rate\_leaf | 0.01 | LogUniform[1e-4, 0.25] |
| cosine\_decay\_steps | 0 | {0, 100, 1000} |
| dropout | 0.0 | {0, 0.25} |
| selected\_variables | 0.8 | {0.5, 0.75, 1.} |
| Focal loss | 0.0 | {False, True} |
| Temperature | 0.0 | {0, 0.25} |

Table 12: Hyperparameter configurations for GRANDE. The focal loss and temperature parameters only apply to classification tasks.

## Appendix C Detailed Performance Results

In this Section, we provide the leaderboard position results for all the experiments in the main paper, separated by hyperparameter regime and preprocessing pipeline. The results can be seen in Tables [13](#A3.T13 "Table 13 ‣ Appendix C Detailed Performance Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), [14](#A3.T14 "Table 14 ‣ Appendix C Detailed Performance Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), and [16](#A3.T16 "Table 16 ‣ Appendix C Detailed Performance Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data").

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | MBGM | 0.17 | 0.226 | 0.997 | 0.231 | 0.267 | 0.605 | 0.143 |
| SVPC | 0.799 | 0.929 | 0.889 | 0.798 | 0.798 | 0.92 | 0.795 |
| AEAC | 0.553 | 0.613 | 0.91 | 0.503 | 0.544 | 0.527 | 0.43 |
| OGPCC | 0.819 | 0.803 | 0.795 | 0.706 | 0.729 | 0.776 | 0.69 |
| SCS | 0.466 | 0.439 | 0.469 | 0.368 | 0.6 | 0.412 | 0.37 |
| BPCCM | 0.256 | 0.28 | 0.953 | 0.261 | 0.281 | 0.31 | 0.07 |
| SCTP | 0.338 | 0.364 | 0.431 | 0.287 | 0.374 | 0.315 | 0.376 |
| HQC | 0.319 | 0.343 | 0.936 | 0.378 | 0.47 | 0.418 | 0.455 |
| IFD | 0.311 | 0.324 | 0.519 | 0.226 | 0.408 | 0.294 | 0.177 |
| PSSDP | 0.288 | 0.301 | 0.519 | 0.315 | 0.478 | 0.258 | 0.347 |
| Light | MBGM | 0.552 | 0.312 | 0.998 | 0.272 | 0.39 | 0.708 | 0.649 |
| HPO | SVPC | 0.929 | 0.937 | 0.895 | 0.798 | 0.798 | 0.946 | 0.925 |
| AEAC | 0.544 | 0.693 | 0.945 | 0.693 | 0.614 | 0.56 | 0.474 |
| OGPCC | 0.834 | 0.888 | 0.799 | 0.712 | 0.748 | 0.808 | 0.587 |
| SCS | 0.609 | 0.543 | 0.557 | 0.374 | 0.993 | 0.529 | 0.4 |
| BPCCM | 0.578 | 0.398 | 0.978 | 0.285 | 0.31 | 0.357 | 0.185 |
| SCTP | 0.448 | 0.392 | 0.448 | 0.297 | 0.401 | 0.501 | 0.4 |
| HQC | 0.865 | 0.414 | 0.987 | 0.378 | 0.491 | 0.527 | 0.509 |
| IFD | 0.461 | 0.525 | 0.54 | 0.22 | 0.334 | 0.267 | 0.201 |
| PSSDP | 0.583 | 0.392 | 0.555 | 0.313 | 0.493 | 0.656 | 0.407 |
| Extensive | MBGM | 0.476 | 0.503 | 0.999 | 0.334 | 0.448 | 0.8 | 0.615 |
| HPO | SVPC | 0.932 | 0.946 | 0.917 | 0.798 | 0.798 | 0.947 | 0.932 |
| AEAC | 0.585 | 0.687 | 0.953 | 0.691 | 0.669 | 0.6 | 0.474 |
| OGPCC | 0.887 | 0.896 | 0.845 | 0.724 | 0.742 | 0.878 | 0.776 |
| SCS | 0.692 | 0.73 | 0.542 | 0.351 | 0.945 | 0.478 | 0.427 |
| BPCCM | 0.587 | 0.499 | 0.986 | 0.301 | 0.333 | 0.362 | 0.185 |
| SCTP | 0.51 | 0.428 | 0.495 | 0.298 | 0.408 | 0.518 | 0.496 |
| HQC | 0.911 | 0.409 | 0.991 | 0.414 | 0.527 | 0.527 | 0.619 |
| IFD | 0.533 | 0.662 | 0.552 | 0.223 | 0.518 | 0.268 | 0.245 |
| PSSDP | 0.656 | 0.463 | 0.586 | 0.308 | 0.549 | 0.656 | 0.418 |

Table 13: Leaderboard position of models trained with varying hyperparameter optimization regimes on datasets after standardized preprocessing. The best model (+/- 0.01) is highlighted.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | MBGM | 0.706 | 0.545 | 0.999 | 0.626 | 0.641 | 0.964 | 0.61 |
| SVPC | 0.993 | 0.987 | 0.987 | 0.941 | 0.968 | 0.989 | 0.946 |
| AEAC | 0.736 | 0.84 | 0.937 | 0.695 | 0.914 | 0.407 | 0.407 |
| OGPCC | 0.806 | 0.792 | 0.797 | 0.702 | 0.718 | 0.731 | 0.681 |
| SCS | 0.59 | 0.758 | 0.673 | 0.366 | 0.879 | 0.466 | 0.393 |
| BPCCM | 0.994 | 0.995 | 0.994 | 0.987 | 0.993 | 0.995 | 0.991 |
| SCTP | 0.311 | 0.373 | 0.417 | 0.284 | 0.376 | 0.345 | 0.395 |
| HQC | 0.354 | 0.371 | 0.953 | 0.46 | 0.948 | 0.393 | 0.368 |
| IFD | 0.83 | 0.775 | 0.775 | 0.215 | 0.741 | 0.615 | 0.462 |
| PSSDP | 0.511 | 0.479 | 0.743 | 0.531 | 0.993 | 0.361 | 0.377 |
| Light | MBGM | 0.991 | 0.794 | 0.999 | 0.727 | 0.774 | 0.908 | 0.985 |
| HPO | SVPC | 0.993 | 0.992 | 0.987 | 0.951 | 0.971 | 0.987 | 0.988 |
| AEAC | 0.734 | 0.932 | 0.945 | 0.905 | 0.78 | 0.693 | 0.495 |
| OGPCC | 0.823 | 0.852 | 0.803 | 0.705 | 0.748 | 0.839 | 0.563 |
| SCS | 0.824 | 0.754 | 0.783 | 0.382 | 0.837 | 0.555 | 0.519 |
| BPCCM | 0.993 | 0.991 | 0.995 | 0.99 | 0.993 | 0.995 | 0.995 |
| SCTP | 0.413 | 0.418 | 0.483 | 0.288 | 0.37 | 0.483 | 0.438 |
| HQC | 0.989 | 0.501 | 0.989 | 0.45 | 0.986 | 0.973 | 0.57 |
| IFD | 0.988 | 0.963 | 0.836 | 0.2 | 0.71 | 0.572 | 0.744 |
| PSSDP | 0.716 | 0.708 | 0.735 | 0.701 | 0.94 | 0.741 | 0.575 |
| Extensive | MBGM | 0.95 | 0.859 | 0.994 | 0.71 | 0.875 | 0.934 | 0.945 |
| HPO | SVPC | 0.993 | 0.992 | 0.987 | 0.949 | 0.978 | 0.987 | 0.985 |
| AEAC | 0.762 | 0.937 | 0.928 | 0.832 | 0.777 | 0.702 | 0.525 |
| OGPCC | 0.867 | 0.856 | 0.842 | 0.714 | 0.751 | 0.871 | 0.777 |
| SCS | 0.953 | 0.941 | 0.777 | 0.377 | 0.702 | 0.627 | 0.711 |
| BPCCM | 0.992 | 0.992 | 0.996 | 0.991 | 0.992 | 0.992 | 0.996 |
| SCTP | 0.521 | 0.5 | 0.557 | 0.293 | 0.376 | 0.499 | 0.962 |
| HQC | 0.99 | 0.487 | 0.991 | 0.468 | 0.986 | 0.982 | 0.839 |
| IFD | 0.988 | 0.985 | 0.809 | 0.216 | 0.665 | 0.62 | 0.736 |
| PSSDP | 0.994 | 0.944 | 0.973 | 0.684 | 0.99 | 0.99 | 0.605 |

Table 14: Leaderboard position of models trained with varying hyperparameter optimization regimes on datasets after feature engineering. The best model (+/- 0.01) is highlighted.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | AEAC | 0.944 | 0.948 | 0.98 | 0.725 | 0.758 | 0.46 | 0.445 |
| OGPCC | 0.856 | 0.847 | 0.84 | 0.714 | 0.714 | 0.783 | 0.718 |
| SCS | 0.543 | 0.611 | 0.701 | 0.39 | 0.892 | 0.635 | 0.431 |
| SCTP | 0.985 | 0.987 | 0.988 | 0.302 | 0.983 | 0.986 | 0.988 |
| IFD | 0.986 | 0.983 | 0.913 | 0.214 | 0.774 | 0.53 | 0.533 |
| PSSDP | 0.508 | 0.491 | 0.746 | 0.59 | 0.981 | 0.315 | 0.374 |
| Light | AEAC | 0.95 | 0.96 | 0.99 | 0.932 | 0.943 | 0.776 | 0.507 |
| HPO | OGPCC | 0.894 | 0.915 | 0.852 | 0.731 | 0.75 | 0.842 | 0.566 |
| SCS | 0.937 | 0.778 | 0.773 | 0.425 | 0.904 | 0.611 | 0.485 |
| SCTP | 0.988 | 0.988 | 0.989 | 0.315 | 0.985 | 0.991 | 0.989 |
| IFD | 0.991 | 0.989 | 0.987 | 0.211 | 0.692 | 0.642 | 0.646 |
| PSSDP | 0.94 | 0.773 | 0.792 | 0.745 | 0.978 | 0.707 | 0.609 |
| Extensive | AEAC | 0.953 | 0.961 | 0.991 | 0.922 | 0.932 | 0.932 | 0.534 |
| HPO | OGPCC | 0.922 | 0.923 | 0.884 | 0.72 | 0.78 | 0.878 | 0.808 |
| SCS | 0.975 | 0.842 | 0.904 | 0.362 | 0.798 | 0.734 | 0.798 |
| SCTP | 0.991 | 0.99 | 0.991 | 0.346 | 0.985 | 0.992 | 0.991 |
| IFD | 0.992 | 0.992 | 0.972 | 0.204 | 0.739 | 0.647 | 0.708 |
| PSSDP | 0.992 | 0.982 | 0.99 | 0.741 | 0.994 | 0.995 | 0.651 |

Table 15: Leaderboard position of models trained with varying hyperparameter optimization regimes on datasets after test-time feature engineering as a preprocessing method for test-time adaptation. The best model (+/- 0.01) is highlighted.



|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | MBGM | SVPC | AEAC | OGPCC | SCS | BPCCM | SCTP | HQC | IFD | PSSDP |
| Def. | 0.74 | 0.799 | 0.618 | 0.996 | 0.92 | 0.991 | 0.498 | 0.992 | 0.205 | 0.562 |
| FE | 0.964 | 0.963 | 0.953 | 0.983 | 0.999 | 0.995 | 0.531 | 0.992 | 0.351 | 0.707 |
| TTA | - | - | 0.993 | 0.995 | 1.0 | - | 0.991 | - | 0.432 | 0.742 |

Table 16: Leaderboard position of AutoGluon on the private Kaggle leaderboard after different preprocessing applied. The best results (+/- 0.01) are highlighted.

## Appendix D Additional Results

### D.1 Comparison of Preprocessing Pipelines per Model and Dataset

Figure [7](#A4.F7 "Figure 7 ‣ D.1 Comparison of Preprocessing Pipelines per Model and Dataset ‣ Appendix D Additional Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") visualizes the comparison of different pipelines per model and dataset. It can be seen that almost all models benefit from feature engineering and test-time adaptation on almost all datasets. The only remarkable outlier is FTTransformer on the SCS dataset. The otherwise consistent results support our claim that feature engineering and test-time adaptation are important components of tabular machine learning competitions.

![Refer to caption](/html/2407.02112/assets/x7.png)


Figure 7: Leaderboard positions of models and datasets in different preprocessing pipelines.

### D.2 Analysis of Modeling Components

In this Subsection we complement our analysis of modeling components in the main paper with additional analyses from different perspectives.
Figure [8](#A4.F8 "Figure 8 ‣ D.2 Analysis of Modeling Components ‣ Appendix D Additional Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows the distributions of the leaderboard positions of all our experiments, grouped by different modeling components. It can be seen that without expert feature engineering, most submissions are far from the top percentiles on the leaderboard, while after expert feature engineering, most submissions score top ranks. After test-time adaptation, the density of top submissions increases even more. Moreover, the importance of hyperparameter optimization can be seen. Regarding model selection, CatBoost clearly dominates, mainly due to the property of achieving robustly strong results with default hyperparameters.

![Refer to caption](/html/2407.02112/assets/x8.png)


Figure 8: Kernel Density Estimation of all results grouped by different modeling components (Left: Preprocessing Pipelines, Center: HPO regimes, Right: Models).

These results are in line with common practice in ML competitions [[81](#bib.bib81)]. While recent work strongly focuses on model selection [[35](#bib.bib35), [63](#bib.bib63)], participants of Kaggle competitions typically stick to few model classes and instead focus on developing feature engineering techniques for these particular models [[42](#bib.bib42)].
Predictive Machine Learning is a winner-takes-all game. Selecting another model than the one that is known to work best is only important if the default model fails or if ensembling is necessary. Hence, it makes sense to look at the problem from the winner-takes-all perspective and to evaluate performance gains from different modeling decisions w.r.t. a strong baseline. It is known from related work that CatBoost is the strongest model with default hyperparameters [[63](#bib.bib63)]. Hence, we evaluate performance gains over a CatBoost baseline. Figure [9](#A4.F9 "Figure 9 ‣ D.2 Analysis of Modeling Components ‣ Appendix D Additional Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") illustrates average leaderboard position gains over the baseline for different modeling decisions. It can be seen that without feature engineering, the best average leaderboard position is the 14.5% percentile, while without model selection, it is the 3% percentile. Hence, one of the most important takeaways is that current tabular ML research overemphasizes model evaluation but underestimates data preprocessing especially feature engineering. While TTA increases the average leaderboard position by 8.3% in the default setting, its importance after model selection and hyperparameter optimization reduces. Without TTA, the best achievable average leaderboard position was 3.2% and 1.6% with TTA, indicating a relatively small but important effect.
In this Figure, the average top position is the 1.6% percentile because we only considered single models. The missing component for scoring in the top 1% percentile is ensembling, which we have shown to achieve using AutoGluon in the main paper.

![Refer to caption](/html/2407.02112/assets/x9.png)


Figure 9: Average Gains from different modeling choices from a winner-takes-all perspective with CatBoost as the default model. Lower values mean a higher leaderboard position (unlike the rest of the paper).

To statistically test our results, we estimate the effect of different modeling components in a mixed-effects regression analysis. We use the leaderboard position of all our experiments as the target variable. The samples are all our experiments with leaderboard evaluations resulting from all dataset-preprocessing-model-HPO combinations. To control for different dataset difficulty, we use the dataset as a random effect.
The fixed effects are:

* •

  Featue Engineering {0, 1}: 1 if feature engineering (with and without TTA) was applied;
* •

  Test-Time Adaptation {0, 1}: 1 if test-time adaptation was applied;
* •

  Model Selection {-1, 0, 1}: -1 if CatBoost is the model, 1 if the model is the best of all models on a dataset-preprocessing-HPO combination;
* •

  Light HPO {0,1}: 1 if light HPO was applied;
* •

  Extensive HPO {0,1}: 1 if extensive HPO was applied;

The results in Table [17](#A4.T17 "Table 17 ‣ D.2 Analysis of Modeling Components ‣ Appendix D Additional Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") confirm the strong overall importance of feature engineering and the relevance of HPO and test-time adaptation. Furthermore, it can be seen that using a model other than CatBoost does not lead to significant gains on average. We want to emphasize that while this reflects the general trend across all experiments, for some constellations, other models achieve strong gains over CatBoost.

|  |  |  |  |
| --- | --- | --- | --- |
| Model: | MixedLM | Dependent Variable: | leaderboard position |
| No. Observations: | 546 | Method: | REML |
| No. Groups: | 10 | Scale: | 0.0380 |
| Min. group size: | 42 | Log-Likelihood: | 88.5557 |
| Max. group size: | 63 | Converged: | Yes |
| Mean group size: | 54.6 |  |  |

|  | Coef. | Std.Err. | z | P>|>|z||| | [0.025 | 0.975] |
| --- | --- | --- | --- | --- | --- | --- |
| Intercept | 0.485 | 0.041 | 11.898 | 0.000 | 0.405 | 0.565 |
| Feature Engineering | 0.201 | 0.019 | 10.561 | 0.000 | 0.164 | 0.238 |
| Test-Time Adaptation | 0.080 | 0.023 | 3.437 | 0.001 | 0.034 | 0.126 |
| Model Selection | 0.004 | 0.016 | 0.240 | 0.810 | -0.027 | 0.034 |
| Light HPO | 0.085 | 0.020 | 4.163 | 0.000 | 0.045 | 0.125 |
| Extensive HPO | 0.125 | 0.020 | 6.137 | 0.000 | 0.085 | 0.165 |
| Dataset (Group Variable) | 0.013 | 0.035 |  |  |  |  |

Table 17: Mixed Linear Model Regression Results

### D.3 Using Our Framework for Evaluating New Methods

In Section [5](#S5 "5 Implications for Future Work ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") we identified directions for future work. These directions were based on general insights of our analysis for the tabular data field. This subsection will showcase how our framework can be utilized to develop new approaches and compare them to expert solutions. Our framework contains, to our knowledge, the largest collection of implemented expert solutions for relevant datasets. Hence, our framework is especially useful to researchers developing AutoML solutions, especially focusing on feature engineering. Furthermore, our framework can be useful to researchers developing model-specific and data-agnostic preprocessing pipelines, i.e., for novel neural networks. In addition, our framework can be used to develop test-time adaptation methods for tabular data.
Figure [10](#A4.F10 "Figure 10 ‣ D.3 Using Our Framework for Evaluating New Methods ‣ Appendix D Additional Results ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data") shows four particular challenges for future work in tabular Deep Learning and AutoML for which our framework can serve as a benchmark to measure progress:
A) Develop a neural network not relying on feature engineering techniques;
B) Develop a neural network capable of test-time adaptation to replace the often infeasible test-time feature engineering.
C) Create a universal model-agnostic automated feature engineering pipeline that surpasses expert feature engineering;
D) Enhance AutoML solutions to outperform expert feature engineering pipelines;

![Refer to caption](/html/2407.02112/assets/x10.png)


Figure 10: Challenges for further automating deep learning and AutoML for tabular data. Challenge A compares the best neural networks within the standardized and feature engineering pipelines after extensive HPO. Challenge B compares the best neural networks within the feature engineering and the test-time adaptation pipeline after extensive HPO. Challenge C compares the best model within the standardized pipeline to the best model within the feature engineering pipeline. Challenge D compares AutoGluon to the best model within the feature engineering pipeline.

## Appendix E Limitations

The main goal of our experiments was to showcase the limitations of evaluation frameworks currently prevalent in tabular Machine Learning. This required extensive experiments in different preprocessing pipelines. Some limitations arising from this scope are:

* •

  We use Kaggle competitions in an effort to evaluate more realistic tasks than evaluated in related work. It is important to highlight that the competition setup on Kaggle does not always reflect real-world tasks. However, due to the involvement of companies and institutions and the poor availability of high-quality tabular datasets [[47](#bib.bib47)], these are arguably among the most realistic datasets available as open-source data. Furthermore, one of our contributions was to separate aspects from the main learning task that made competitions unrealistic (i.e., data leaks or, for some applications, test-time adaptation). This additionally improves the real-world transferability of our experiments.
* •

  We split the overall expert preprocessing into feature engineering and test-time adaptation. However, pipelines could be differentiated further, or single-feature engineering techniques could be investigated. For instance, we could separate the expert feature engineering pipeline by whether expert feature selection is applied. However, due to the extent of our experiments, we focus on a pipeline perspective and leave fine-grained analyses of specific techniques for future work.
* •

  We use the leaderboard percentile as the main evaluation measure to have an external reference for the top performance on a dataset. A possible issue of that design choice is that the leaderboard of each dataset is differently distributed. Hence, what appears to be a large jump on a dataset might actually only be a small increase on the metric, while for another dataset, the same leaderboard increase might amount to a substantial increase in performance. However, averaging over datasets has a natural interpretation when using the leaderboard position, which is not there when using normalized versions of entirely different metrics. In addition to the evaluation in the main paper, we include evaluations on the original metrics in Appendix [F](#A6 "Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"). The results indicate that our claims similarly hold when evaluating using the original metrics.
* •

  Our evaluation framework does not allow to assess whether one model is generally better than another model. We only claim that model comparisons change and that feature engineering and preprocessing greatly influence model comparison on our datasets. For a more generalizable model comparison using more datasets, we refer to related work [[35](#bib.bib35), [63](#bib.bib63)].
* •

  Due to the extent of our experiments (over 200,000 trained models), it was infeasible to repeat the experiments multiple times to obtain error bars. Nevertheless, our experiments include randomness (e.g., CV splits, weight initialization for deep learning models, or bagging for the tree-based models), limiting the generalizability of our results. However, the extent of our experiments and the clear differences between the implemented preprocessing pipelines over multiple models and datasets make the risk of randomness affecting our claims very low despite not being explicitly quantified for all models.
* •

  Due to the focus on incentivized Kaggle competitions, most datasets are from the finance domain and from North America or Europe. Hence, non-profit domains and other continents are underrepresented. To mitigate this, our analysis could be extended through competitions on other platforms such as Zindi [[93](#bib.bib93)]. However, as we wanted our framework to be easy to use, we focused on Kaggle, which contains an API for effortlessly downloading datasets and submitting predictions.

## Appendix F Evaluation on the Original Task Metrics

This Section lists the main results using the original task metrics for all experiments. An overview of important components per dataset and model can be seen in Figure [11](#A6.F11 "Figure 11 ‣ Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"). All experimental results on the original metrics can be seen in Tables [18](#A6.T18 "Table 18 ‣ Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), [19](#A6.T19 "Table 19 ‣ Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), [20](#A6.T20 "Table 20 ‣ Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"), and [21](#A6.T21 "Table 21 ‣ Appendix F Evaluation on the Original Task Metrics ‣ A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data"). Further results can be seen in our code. Overall, the evaluation using the original metrics aligns with our main findings.

![Refer to caption](/html/2407.02112/assets/x11.png)


Figure 11: Performance gains from different modeling components on the original metrics of the Kaggle competitions. Higher values correspond to better performance. The original metric was reversed for SVPC, OGPCC, and BPCCM to align with the higher-is-better notation. ’Default’ corresponds to the model performance with default hyperparameters in a standardized preprocessing pipeline. Light and extensive HPO correspond to tuning hyperparameters in the same preprocessing pipeline. Expert FE and FE-TTA correspond to the model performance with extensively tuned hyperparameters in the feature engineering and the test-time adaptation pipeline respectively.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | MBGM | 0.5276 | 0.5363 | 0.554 | 0.5366 | 0.5398 | 0.5483 | 0.52 |
| SVPC | 0.3666 | 0.419 | 0.3945 | 0.3348 | 0.3433 | 0.4041 | 0.3249 |
| AEAC | 0.8754 | 0.881 | 0.9005 | 0.8653 | 0.8738 | 0.8707 | 0.8302 |
| OGPCC | 0.5525 | 0.5501 | 0.5482 | 0.5293 | 0.5356 | 0.5443 | 0.5259 |
| SCS | 0.8239 | 0.8232 | 0.824 | 0.8205 | 0.825 | 0.8223 | 0.8206 |
| BPCCM | 0.525 | 0.5295 | 0.5539 | 0.5262 | 0.5295 | 0.5325 | 0.4148 |
| SCTP | 0.8892 | 0.893 | 0.8965 | 0.8666 | 0.894 | 0.8863 | 0.8942 |
| HQC | 0.96 | 0.9612 | 0.9679 | 0.9636 | 0.9664 | 0.9651 | 0.966 |
| IFD | 0.8989 | 0.9006 | 0.9121 | 0.8799 | 0.9065 | 0.8965 | 0.859 |
| PSSDP | 0.2703 | 0.2724 | 0.2843 | 0.274 | 0.283 | 0.2655 | 0.2766 |
| Light | MBGM | 0.5474 | 0.5416 | 0.5543 | 0.5402 | 0.5441 | 0.5499 | 0.5489 |
| SVPC | 0.4201 | 0.4316 | 0.3972 | 0.3454 | 0.343 | 0.4405 | 0.4129 |
| AEAC | 0.8738 | 0.8873 | 0.9056 | 0.8874 | 0.8811 | 0.8757 | 0.8548 |
| OGPCC | 0.5556 | 0.5664 | 0.5491 | 0.5305 | 0.5404 | 0.551 | 0.4905 |
| SCS | 0.8251 | 0.8247 | 0.8248 | 0.8208 | 0.8277 | 0.8245 | 0.8218 |
| BPCCM | 0.5436 | 0.538 | 0.5584 | 0.53 | 0.5325 | 0.5354 | 0.5057 |
| SCTP | 0.8968 | 0.8951 | 0.8968 | 0.8756 | 0.8956 | 0.8978 | 0.8956 |
| HQC | 0.9677 | 0.965 | 0.9685 | 0.9636 | 0.9667 | 0.967 | 0.9669 |
| IFD | 0.9097 | 0.9126 | 0.9135 | 0.8782 | 0.9012 | 0.8923 | 0.8688 |
| PSSDP | 0.2863 | 0.2799 | 0.2855 | 0.2739 | 0.2835 | 0.2878 | 0.2805 |
| Extensive | MBGM | 0.5461 | 0.5465 | 0.5545 | 0.5425 | 0.5457 | 0.5513 | 0.5484 |
| SVPC | 0.4256 | 0.4421 | 0.4024 | 0.3453 | 0.3455 | 0.4428 | 0.4239 |
| AEAC | 0.8771 | 0.8868 | 0.9086 | 0.887 | 0.8825 | 0.8793 | 0.8546 |
| OGPCC | 0.5662 | 0.5685 | 0.5575 | 0.5345 | 0.5387 | 0.5638 | 0.5443 |
| SCS | 0.8255 | 0.8258 | 0.8247 | 0.8197 | 0.8271 | 0.8241 | 0.8228 |
| BPCCM | 0.5439 | 0.5421 | 0.561 | 0.5315 | 0.534 | 0.5359 | 0.5053 |
| SCTP | 0.898 | 0.8964 | 0.8977 | 0.877 | 0.8959 | 0.8981 | 0.8977 |
| HQC | 0.9678 | 0.9648 | 0.9688 | 0.965 | 0.9671 | 0.9671 | 0.9674 |
| IFD | 0.9132 | 0.9194 | 0.914 | 0.8794 | 0.912 | 0.8924 | 0.8864 |
| PSSDP | 0.2878 | 0.2824 | 0.2864 | 0.2735 | 0.2853 | 0.2878 | 0.281 |

Table 18: Performance of models trained with varying hyperparameter optimization regimes on private test competition datasets after standardized preprocessing. Higher values correspond to better performance. The original metric was reversed for SVPC, OGPCC, and BPCCM to align with the higher-is-better notation. The best model is highlighted. A model is considered better if it achieves a score that is one leaderboard standard deviation (std) larger than the other. The std is determined based on all top 1% submissions to only focus on the best models.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | MBGM | 0.5499 | 0.5473 | 0.5546 | 0.5485 | 0.5488 | 0.5524 | 0.5483 |
| SVPC | 0.4683 | 0.4643 | 0.4646 | 0.4369 | 0.457 | 0.4665 | 0.4409 |
| AEAC | 0.8947 | 0.8992 | 0.9039 | 0.8881 | 0.901 | 0.8168 | 0.816 |
| OGPCC | 0.5504 | 0.5473 | 0.5484 | 0.528 | 0.5327 | 0.5359 | 0.5227 |
| SCS | 0.8249 | 0.8261 | 0.8254 | 0.8203 | 0.8267 | 0.8239 | 0.8216 |
| BPCCM | 0.5672 | 0.5676 | 0.5668 | 0.5627 | 0.5667 | 0.5681 | 0.5651 |
| SCTP | 0.885 | 0.8939 | 0.8962 | 0.8643 | 0.8942 | 0.8906 | 0.8952 |
| HQC | 0.9618 | 0.9631 | 0.968 | 0.9661 | 0.968 | 0.9642 | 0.9628 |
| IFD | 0.9275 | 0.9255 | 0.9255 | 0.8759 | 0.9233 | 0.9173 | 0.9097 |
| PSSDP | 0.2841 | 0.283 | 0.2895 | 0.2848 | 0.2911 | 0.2779 | 0.2788 |
| Light | MBGM | 0.5534 | 0.5512 | 0.555 | 0.5503 | 0.551 | 0.5518 | 0.5531 |
| SVPC | 0.4694 | 0.4682 | 0.4642 | 0.4461 | 0.4582 | 0.4647 | 0.4653 |
| AEAC | 0.8941 | 0.9033 | 0.9059 | 0.9002 | 0.8981 | 0.8874 | 0.8619 |
| OGPCC | 0.5538 | 0.5589 | 0.5501 | 0.529 | 0.5403 | 0.5563 | 0.4857 |
| SCS | 0.8265 | 0.8261 | 0.8263 | 0.8212 | 0.8266 | 0.8248 | 0.8244 |
| BPCCM | 0.5666 | 0.5649 | 0.5678 | 0.5643 | 0.5664 | 0.5682 | 0.5688 |
| SCTP | 0.8961 | 0.8963 | 0.8973 | 0.868 | 0.8936 | 0.8973 | 0.8966 |
| HQC | 0.9686 | 0.9668 | 0.9686 | 0.9659 | 0.9684 | 0.9682 | 0.9673 |
| IFD | 0.9316 | 0.9289 | 0.9278 | 0.8684 | 0.922 | 0.9145 | 0.9237 |
| PSSDP | 0.2891 | 0.2889 | 0.2893 | 0.2887 | 0.2901 | 0.2894 | 0.2861 |
| Extensive | MBGM | 0.5522 | 0.5516 | 0.5537 | 0.55 | 0.5517 | 0.5521 | 0.5522 |
| SVPC | 0.4694 | 0.4682 | 0.4645 | 0.4446 | 0.4611 | 0.465 | 0.4626 |
| AEAC | 0.8972 | 0.9038 | 0.9028 | 0.8988 | 0.898 | 0.8892 | 0.8701 |
| OGPCC | 0.5615 | 0.56 | 0.557 | 0.5311 | 0.5415 | 0.5622 | 0.5444 |
| SCS | 0.8271 | 0.8271 | 0.8263 | 0.8209 | 0.8256 | 0.8252 | 0.8256 |
| BPCCM | 0.5659 | 0.5653 | 0.5692 | 0.5647 | 0.5662 | 0.5662 | 0.5689 |
| SCTP | 0.8982 | 0.8978 | 0.8987 | 0.8725 | 0.8942 | 0.8978 | 0.9002 |
| HQC | 0.9687 | 0.9667 | 0.9688 | 0.9663 | 0.9685 | 0.9684 | 0.9676 |
| IFD | 0.9319 | 0.9303 | 0.9268 | 0.8764 | 0.9196 | 0.9174 | 0.9229 |
| PSSDP | 0.2912 | 0.2902 | 0.2905 | 0.2883 | 0.2908 | 0.2909 | 0.2869 |

Table 19: Performance of models trained with varying hyperparameter optimization regimes on private test competition datasets after feature engineering. Higher values correspond to better performance. The original metric was reversed for SVPC, OGPCC, and BPCCM to align with the higher-is-better notation. The best model is highlighted. A model is considered better if it achieves a score that is one leaderboard standard deviation (std) larger than the other. The std is determined based on all top 1% submissions to only focus on the best models.



|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | XGBoost | LightGBM | CatBoost | ResNet | FTT | MLP-PLR | GRANDE |
| Default | AEAC | 0.9055 | 0.9066 | 0.9144 | 0.893 | 0.8967 | 0.8472 | 0.8388 |
| OGPCC | 0.5601 | 0.5578 | 0.5567 | 0.5312 | 0.5309 | 0.5452 | 0.5326 |
| SCS | 0.8247 | 0.8251 | 0.8255 | 0.8214 | 0.8268 | 0.8253 | 0.823 |
| SCTP | 0.9143 | 0.9154 | 0.9168 | 0.8797 | 0.9135 | 0.915 | 0.917 |
| IFD | 0.9307 | 0.9301 | 0.9284 | 0.8755 | 0.9254 | 0.9129 | 0.9132 |
| PSSDP | 0.284 | 0.2834 | 0.2895 | 0.2865 | 0.2906 | 0.2741 | 0.2786 |
| Light | AEAC | 0.9069 | 0.9113 | 0.9171 | 0.9034 | 0.9052 | 0.898 | 0.8655 |
| OGPCC | 0.568 | 0.5729 | 0.5589 | 0.536 | 0.5412 | 0.5571 | 0.4878 |
| SCS | 0.8271 | 0.8263 | 0.8262 | 0.8227 | 0.8269 | 0.8251 | 0.8242 |
| SCTP | 0.9173 | 0.9172 | 0.9183 | 0.8864 | 0.9146 | 0.9193 | 0.9176 |
| IFD | 0.9343 | 0.9327 | 0.9314 | 0.8741 | 0.9209 | 0.9183 | 0.9186 |
| PSSDP | 0.2901 | 0.2898 | 0.2898 | 0.2895 | 0.2905 | 0.2889 | 0.287 |
| Extensive | AEAC | 0.9087 | 0.9115 | 0.9172 | 0.9019 | 0.9035 | 0.9032 | 0.872 |
| OGPCC | 0.5743 | 0.5748 | 0.5652 | 0.5333 | 0.5447 | 0.5638 | 0.551 |
| SCS | 0.8273 | 0.8266 | 0.8269 | 0.82 | 0.8264 | 0.8259 | 0.8264 |
| SCTP | 0.9193 | 0.9189 | 0.9194 | 0.8908 | 0.9144 | 0.9202 | 0.9196 |
| IFD | 0.935 | 0.9352 | 0.9291 | 0.8698 | 0.9232 | 0.9186 | 0.9218 |
| PSSDP | 0.291 | 0.2906 | 0.2908 | 0.2894 | 0.2911 | 0.2914 | 0.2876 |

Table 20: Performance of models trained with varying hyperparameter optimization regimes on private test competition datasets after test-time adaptation. Higher values correspond to better performance. The original metric was reversed for SVPC, OGPCC, and BPCCM to align with the higher-is-better notation. The best model is highlighted. A model is considered better if it achieves a score that is one leaderboard standard deviation (std) larger than the other. The std is determined based on all top 1% submissions to only focus on the best models.



|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | MBGM | SVPC | AEAC | OGPCC | SCS | BPCCM | SCTP | HQC | IFD | PSSDP |
| Def. | 0.5505 | 0.366 | 0.8816 | 0.5979 | 0.827 | 0.565 | 0.8978 | 0.9689 | 0.871 | 0.2858 |
| FE | 0.5524 | 0.454 | 0.9087 | 0.5873 | 0.8285 | 0.5678 | 0.8984 | 0.9692 | 0.902 | 0.2889 |
| TTA | - | - | 0.9194 | 0.5971 | 0.8292 | - | 0.9194 | - | 0.908 | 0.2894 |

Table 21: Performance of AutoGluon on private test competition datasets after different preprocessing applied. Higher values correspond to better performance. The original metric was reversed for SVPC, OGPCC, and BPCCM to align with the higher-is-better notation. The best preprocessing is highlighted.

[◄](/html/2407.02111)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2407.02112)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2407.02112)
[View original  
on arXiv](https://arxiv.org/abs/2407.02112)[►](/html/2407.02113)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Aug 5 16:46:26 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
