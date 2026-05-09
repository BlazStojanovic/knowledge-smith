---
arxiv: '2406.19380'
authors:
- Ivan Rubachev Yandex, HSE University &Nikolay Kartashev HSE University, Yandex \AND
  Yury Gorishniy Yandex &Artem Babenko Yandex, HSE University
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks'
url: https://arxiv.org/abs/2406.19380
year: 2024
---

[2406.19380] TabReD: A Benchmark of Tabular Machine Learning in-the-Wild














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



# TabReD: A Benchmark of Tabular Machine Learning in-the-Wild

Ivan Rubachev
  
Yandex, HSE University
&Nikolay Kartashev
  
HSE University, Yandex
\ANDYury Gorishniy
  
Yandex
&Artem Babenko
  
Yandex, HSE University

###### Abstract

Benchmarks that closely reflect downstream application scenarios are essential for the streamlined adoption of new research in tabular machine learning (ML).
In this work, we examine existing tabular benchmarks and find two common characteristics of industry-grade tabular data that are underrepresented in the datasets available to the academic community.
First, tabular data often changes over time in real-world deployment scenarios. This impacts model performance and requires time-based train and test splits for correct model evaluation. Yet, existing academic tabular datasets often lack timestamp metadata to enable such evaluation.
Second, a considerable portion of datasets in production settings stem from extensive data acquisition and feature engineering pipelines.
For each specific dataset, this can have a different impact on the absolute and relative number of predictive, uninformative, and correlated features, which in turn can affect model selection.
To fill the aforementioned gaps in academic benchmarks, we introduce TabReD – a collection of eight industry-grade tabular datasets covering a wide range of domains from finance to food delivery services. We assess a large number of tabular ML models in the feature-rich, temporally-evolving data setting facilitated by TabReD. We demonstrate that evaluation on time-based data splits leads to different methods ranking, compared to evaluation on random splits more common in academic benchmarks. Furthermore, on the TabReD datasets, MLP-like architectures and GBDT show the best results, while more sophisticated DL models are yet to prove their effectiveness.

## 1 Introduction

During several recent years, research on tabular machine learning has grown rapidly. Plenty of works have proposed neural network architectures [[33](#bib.bib33), [52](#bib.bib52), [19](#bib.bib19), [20](#bib.bib20), [21](#bib.bib21), [10](#bib.bib10), [11](#bib.bib11)] that are competitive or even superior to “shallow” GBDT models [[41](#bib.bib41), [30](#bib.bib30), [13](#bib.bib13)], which has strengthened research interest in the field. Furthermore, specialized workshops devoted to table representation learning are organized on the top-tier ML conferences111Table Representation Learning Workshop, NeurIPS, which highlights the importance of this research line to the community.

Rapid progress on a particular problem requires a comprehensive set of benchmarks. It is important that benchmarks reflect the characteristics and specifics of downstream applications to ensure seamless adoption of research progress [[43](#bib.bib43), [51](#bib.bib51), [34](#bib.bib34)]. In this work we rigorously investigate tabular datasets from academic benchmarks and find that two data characteristics common in industrial tabular ML applications are underrepresented in the current academic benchmarks.

First, in practice, datasets are often split into train/validation/test parts according to timestamps of datapoints [[23](#bib.bib23), [47](#bib.bib47), [8](#bib.bib8), [27](#bib.bib27), [25](#bib.bib25)] to account for potential distribution shifts that can naturally occur in real applications.
In fact, even among academic benchmarks, there is a certain number of datasets with strong time dependencies between instances (e.g. electricity market prediction [[50](#bib.bib50)], flight delay estimation [[7](#bib.bib7)], bike sharing demand [[49](#bib.bib49)], and others).
However, even in such cases, random splits are used in papers instead of time-based splits.
Moreover, timestamp or other task-appropriate split metadata is often not available.

We also find that datasets with large numbers of predictive features and extensive feature engineering are scarce in academic benchmarks. Conversely, such feature-rich datasets are common in many industrial settings [[17](#bib.bib17), [45](#bib.bib45), [28](#bib.bib28), [5](#bib.bib5), [52](#bib.bib52)], but they are often proprietary and unavailable to the academic community.

As a bonus point, we also identify eleven data-leakage issues and the use of non-tabular, synthetic and untraceable datasets in the academic benchmarks.
All TabReD datasets are free from such issues.

To fill the gap in existing academic benchmarks and represent in-the-wild industrial tabular ML applications better, we introduce the TabReD benchmark – a collection of eight datasets, all drawn from real-world industrial applications with tabular data.
All TabReD datasets come with time-based splits into train, validation and test parts.
For a given application, it means that test datapoints appeared later in time than training datapoints.
This approach takes into account the gradual “temporal shift” between training and test objects (i.e. the shift between “older” and “newer” objects), and we confirm the importance of such time-based splits in experiments.
Furthermore, because of additional investments in data acquisition and feature engineering, all datasets in TabReD have more features.
This stems from adopting the preprocessing steps from production ML pipelines and Kaggle competition forums, where extensive data engineering is often highly prioritized.

We evaluate numerous tabular models on the TabReD benchmark. We find that the general trend of tabular DL progress holds there, but the improvements brought by model architecture advances are less pronounced and simpler MLP-like and GBDT solutions prove to be more universally practical and performant on TabReD.

To summarize, our paper presents the following contributions:

* •

  We introduce TabReD – a collection of eight industry-grade tabular datasets that span a wide range of domains from finance to food delivery services (three classification and five regression problems).
  The datasets in TabReD exhibit two important practical properties that, as we show in our analysis, are underrepresented in existing academic benchmarks.
  First, the TabReD datasets are split into train/validation/test sets according to timestamps of objects, which we show to be crucial for correct evaluation.
  Second, the TabReD datasets have more features as a result of additional effort put into data acquisition and feature engineering.
* •

  We evaluate a large number of tabular models on TabReD. We find that, in the feature-rich, time-evolving setting facilitated by TabReD: (1) GBDT and MLPs with embeddings [[20](#bib.bib20)] demonstrate the best average performance; (2) more complex DL models turn out to be less effective; (3) the difference in metrics between various methods is less pronounced than on academic benchmarks used in prior work.
* •

  We demonstrate that correct evaluation on datasets with temporally shifted validation and test sets is crucial as it leads to significant differences in rankings and relative performance of methods, compared to commonly used random-split-based evaluation.
  In particular, we observe that XGBoost performance margin diminishes in correct evaluation setups.

## 2 Related Work

Tabular deep learning is a dynamically developing research area with the recent works proposing new model architectures [[33](#bib.bib33), [52](#bib.bib52), [19](#bib.bib19), [20](#bib.bib20), [21](#bib.bib21), [10](#bib.bib10), [11](#bib.bib11)], regularizations [[26](#bib.bib26)], pretraining pipelines [[42](#bib.bib42), [6](#bib.bib6)] and other solutions [[24](#bib.bib24)]. Since the common way to justify the usage of new approaches is to empirically compare them against the baselines, the choice of the benchmarks for evaluation is critical.

Tabular Data Benchmarks. Since tabular tasks occur in a large number of applications from various domains, there is no single dataset that would be sufficient for evaluation. A typical tabular DL paper reports the results on several tasks from different domains, usually coming from one of the public data repositories. The two traditional data sources for ML problems are the UCI 222<https://archive.ics.uci.edu> and OpenML333<https://www.openml.org> repositories, currently holding thousands of datasets.
Unfortunately, datasets available in public repositories do not cover all tabular ML use cases. In particular, we find that industry-grade tabular ML datasets are underrepresented in these public repositories.

Another source of datasets is the Kaggle444<https://www.kaggle.com> platform, which hosts a plethora of ML competitions, including ones with tabular data. Datasets from competitions are often provided by people solving particular real-world problems, making Kaggle an attractive source of datasets for tabular ML research.
Surprisingly, many benchmarks rely on UCI and OpenML and underutilize tabular datasets from Kaggle.
For example, out of 100 academic datasets that we analyze in [subsection 3.1](#S3.SS1 "3.1 A Closer Look at Existing Tabular ML Benchmarks ‣ 3 TabReD Benchmark ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"), only four come from Kaggle competitions.
To construct TabReD, we source datasets from Kaggle tabular data competitions, and we also introduce four new datasets from real ML production systems at a large tech company.

Tabzilla [[39](#bib.bib39)] and the Grinsztajn et al. [[22](#bib.bib22)] benchmark have gained adoption in the research community. For example, Gorishniy et al. [[21](#bib.bib21)] and Chen et al. [[12](#bib.bib12)] demonstrate models competing with GBDTs on the Grinsztajn et al. [[22](#bib.bib22)] benchmark, Feuer et al. [[16](#bib.bib16)] use a mix of Tabzilla and Grinsztajn et al. [[22](#bib.bib22)] to test an approach for scaling-up TabPFN [[24](#bib.bib24)]. Both benchmarks primarily rely on the OpenML repository as a source of datasets and filter datasets semi-automatically based on metadata like size and baseline performance. In our work, we look closer at all the datasets and find data-leakage issues and non-tabular, synthetic and anonymous/unknown datasets sometimes “sneaking” through automatic filters. Furthermore, many datasets in these two benchmarks lack task-appropriate test splits and use random splits for measuring the performance. We show that ignoring temporal data drift in evaluation could lead to significant differences in methods ranking and relative performance.

TableShift [[18](#bib.bib18)] and WildTab [[35](#bib.bib35)] propose tabular benchmarks with distribution shifts between train/test subsets. These benchmarks are closer in spirit to TabReD, as both mention the importance of moving towards more realistic evaluation scenarios.
However, both benchmarks focus on out-of-distribution robustness and domain generalization methods comparison, not including many tabular-data specific methods. We study a broader set of methods, including recent SoTA tabular neural networks.
Furthermore, both benchmarks consider more “extreme” shifts, compared to more ubiquitous gradual temporal shift which is present in all TabReD datasets.

Benchmarking Under Temporal Shift. Wild-Time [[53](#bib.bib53)] proposed a benchmark consisting of five datasets with temporal shift and identified a 20% performance drop due to temporal shift on average. However, tabular data was not the focus of the Wild-Time benchmark.

The presence of temporal shift and the importance of evaluating models under temporal shifts was discussed in many research and application domains including approximate neighbors search [[8](#bib.bib8)], recommender systems [[44](#bib.bib44)], finance applications [[47](#bib.bib47), [23](#bib.bib23)], health insurance [[27](#bib.bib27)] and in general ML systems best practices [[25](#bib.bib25)].

## 3 TabReD Benchmark

In this section, we take a closer look at existing benchmarks for tabular deep learning and summarize the characteristics that make them different from datasets in industrial ML applications. Then we introduce the TabReD benchmark that aims to address these differences.

### 3.1 A Closer Look at Existing Tabular ML Benchmarks

We collect unique regression and classification datasets from academic tabular ML benchmarks [[20](#bib.bib20), [22](#bib.bib22), [21](#bib.bib21), [39](#bib.bib39), [12](#bib.bib12), [18](#bib.bib18)], this results in a set of 100 datasets. We carefully assess each dataset and summarize our findings in [Table 1](#S3.T1 "Table 1 ‣ 3.1 A Closer Look at Existing Tabular ML Benchmarks ‣ 3 TabReD Benchmark ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"). We also provide detailed meta-data collected in the process with short descriptions of tasks, original data sources, data quality issues and notes on temporal splits in the [Appendix E](#A5 "Appendix E Detailed Academic Datasets Overview ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild") and as a separate table in the supplementary material. Our main findings are summarized below.

Table 1: The landscape of existing tabular machine learning benchmarks compared to TabReD. We report median dataset sizes, number of features, the number of datasets with various issues. The “Time-splits” column is reported only for the datasets without issues. We see that the datasets semi-automatically gathered from OpenML (Tabzilla and Grinsztajn et al. [[22](#bib.bib22)]) contain more quality issues. Furthermore, no benchmark besides TabReD focuses on temporal-shift based evaluation, less than half of datasets in each benchmark have timestamp metadata needed for time-based validation availability.
  
   \* – the original dataset, introduced in [[38](#bib.bib38)] has the canonical OOD split, but the standard IID split commonly used contains time-based leakage.
  
\*\* – the median full dataset size. In experiments, to reduce compute requirements, we use subsampled versions of the TabReD datasets.

  

| Benchmark | Dataset Sizes (Q50) | | Issues (#Issues / #Datasets) | | | Time-split | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #Samples | #Features | Data-Leakage | Synthetic or Untraceable | Non-Tabular | Needed | Possible | Used |
| Grinsztajn et al. [[22](#bib.bib22)] | 16,679 | 13 | 7 / 44 | 1 / 44 | 7 / 44 | 22 | 5 | ✗ |
| Tabzilla [[39](#bib.bib39)] | 3,087 | 23 | 3 / 36 | 6 / 36 | 12 / 36 | 12 | 0 |
| WildTab [[35](#bib.bib35)] | 546,543 | 10 | 1\* / 3 | 1 / 3 | 0 / 3 | 1 | 1 |
| TableShift [[18](#bib.bib18)] | 840,582 | 23 | 0 / 15 | 0 / 15 | 0 / 15 | 15 | 8 |
| Gorishniy et al. [[21](#bib.bib21)] | 57,909 | 20 | 1\* / 10 | 1 / 10 | 0 / 10 | 7 | 1 |
| TabReD (ours) | 7,163,150\*\* | 261 | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |

Data Leakage, Synthetic and Non-Tabular Datasets. First, we find that a considerable number of tabular datasets have some form of data leakage (11 out of 100). Leakage stems from data preparation errors, near-duplicate instances or inappropriate data splits used for testing. A few of these leakage issues have been reported in prior literature [[21](#bib.bib21)], but as there are no common protocols for deprecating datasets in ML [[37](#bib.bib37)], datasets with leakage issues are still used. For some datasets the data source is untraceable, or the data is known to be synthetic without the generation process details or description – there are 13 such datasets. Last, we find that 25 datasets used in academic benchmarks are not inherently tabular by the categorization proposed in Kohli et al. [[34](#bib.bib34)]. These datasets either represent raw data stored in a table form (e.g. flattened images) or homogenous features extracted from some raw data source.

Dataset Size and Feature Engineering. We find that most datasets from the academic benchmarks have less than 60 features and less than a few hundred instances available. Many academic datasets come from publicly available data, which often contain only high-level statistics (e.g. only the source and destination airport and airline IDs for the task of predicting flight delays in the dataset by Ballesteros [[7](#bib.bib7)]). In contrast, many in-the-wild industrial ML applications utilize as much information and data as possible [e.g. [17](#bib.bib17), [45](#bib.bib45), [28](#bib.bib28)]. Unfortunately, not many such datasets from in-the-wild applications are openly available for research. Kaggle competitions come closest to this kind of industry-grade tabular data, but using competition data is less common in current academic benchmarks (only 4 datasets are from Kaggle competitions).

Lack of canonical splits or timestamp metadata. All benchmarks except the ones focused on distribution shift do not discuss the question of data splits used for model evaluation, beyond standard experimental evaluation setups (e.g. random split proportions or cross-validation folds). We find that 53 existing datasets (excluding datasets with issues) potentially contain data drifts related to the passage of time, as the data was collected over time. It is a standard industry practice to use time-based splits for validation in such cases [[44](#bib.bib44), [47](#bib.bib47), [27](#bib.bib27), [25](#bib.bib25)]. However, only 15 datasets have timestamps available for such splits.

### 3.2 Constructing the TabReD Benchmark

In this section, we introduce the new Tabular benchmark with Real-world industrial Datasets (TabReD). To construct TabReD we source datasets from Kaggle competitions and industrial ML applications at a large tech company555Yandex. We adhere to the following criteria when selecting datasets for TabReD. (1) Datasets should be inherently tabular, as discussed in [subsection 3.1](#S3.SS1 "3.1 A Closer Look at Existing Tabular ML Benchmarks ‣ 3 TabReD Benchmark ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild") and Kohli et al. [[34](#bib.bib34)]. (2) Feature engineering and feature collection efforts should be closer to the industry practices. We adapt feature-engineering code by studying competition forums for Kaggle datasets, and we use the exact features from production ML systems for the newly introduced datasets. (3) Exclude datasets with known data leakage, we also take care to avoid leakage in the newly introduced datasets. (4) Datasets should have timestamps available and should have enough samples for the time-based train/test split (this excludes datasets, where future instances are not publicly available, making it impossible to do local validation for research purposes correctly).

The table with our annotations of Kaggle competitions is available in the [Appendix C](#A3 "Appendix C Kaggle Competitions Table ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"). Below we provide short descriptions of datasets and corresponding tasks.666For detailed information regarding data composition, preprocessing and licenses see supplementary materials. We also provide a brief overview for each dataset in [Appendix D](#A4 "Appendix D TabReD Datasets Overview ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild").

Homesite Insurance. This is a dataset from a Kaggle competition hosted by Homesite Insurance [[14](#bib.bib14)]. The task is predicting whether a customer will buy a home insurance policy based on user and insurance policy features (user, policy, sales and geographic information). Each row in the dataset corresponds to a potential [customer, policy] pair, the target indicates whether a customer bought the policy.

Ecom Offers. This is a dataset from a Kaggle competition hosted by the online book and game retailer DMDave [[15](#bib.bib15)]. The task in this dataset is a representative example of modeling customer loyalty in e-commerce. Concretely, the task is classifying whether a customer will redeem a discount offer based on features from two months’ worth of transaction history. We base our feature engineering on one of the top solutions [[40](#bib.bib40)].

HomeCredit Default. This is a second iteration of the popular HomeCredit tabular competition [[23](#bib.bib23)]. The task is to predict whether bank clients will default on a loan, based on bank internal information and external information like credit bureau and tax registry data. This year competition focus was the model prediction stability over time. Compared to the more popular prior competition, this time there is more data and the timestamps are available.
We base feature engineering and preprocessing code on top solutions [[31](#bib.bib31)].

Sberbank Housing. This dataset is from a Kaggle competition, hosted by Sberbank [[3](#bib.bib3)]. This dataset provides information about over 30000 transactions made in the Moscow housing market. The task is to predict the sale price of each property using the provided features describing each property condition, location, and neighborhood, as well as country economic indicators at the moment of the sale. We base our preprocessing code on discussions and solutions from the competition [[4](#bib.bib4), [1](#bib.bib1)].

#### 3.2.1 New Datasets From In-the-Wild ML Applications

Here, we describe the datasets used by various ML applications that we publish with TabReD. All of these datasets were preprocessed for later use by a model in production ML systems. We apply deterministic transforms to anonymize the data for some datasets. We only publish the preprocessed data, as the feature engineering code and internal logs are proprietary. We provide further details regarding licenses, preprocessing and data composition in the datasheet with supplementary materials.

Cooking Time. For this dataset, the task is to determine how long it will take for a restaurant to prepare an order placed in a food delivery app. Features are constructed based on the information about the order contents and historical information about cooking time for the restaurant and brand, the target is a logarithm of minutes it took to cook the placed order.

Delivery ETA. For this dataset, the task is to determine the estimated time of arrival of an order from an online grocery store. Features are constructed based on the courier availability, navigation data and various aggregations of historical information for different time slices, the target is the logarithm of minutes it took to deliver an order.

Maps Routing. For this dataset, the task is to predict the travel time in the car navigation system based on the current road conditions. The features are aggregations of the road graph statistics for the particular route and various road details (like speed limits). The target is the logarithm of seconds per kilometre.

Weather. For this dataset, the task is weather temperature forecasting. This is a dataset similar to the one introduced in [[38](#bib.bib38)], except it is larger, and the time-based split is available and used by default. The features are from weather station measurements and weather forecast physical models. The target is the true temperature for the moment in time.

## 4 Experiments

In this section, we demonstrate the utility of the introduced TabReD benchmark, by answering the following questions:

* •

  Q1 ([subsection 4.1](#S4.SS1 "4.1 How do recent tabular DL advances transfer to a more realistic evaluation setting? ‣ 4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild")) How do recent tabular DL advances on academic benchmarks transfer to the industry setting facilitated by TabReD?
* •

  Q2 ([subsection 4.2](#S4.SS2 "4.2 Influence of Data Validation Splits on Model Ranking ‣ 4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild")) How does accounting for a temporal shift in the data affects model performance and ranking in tabular machine learning?

Experimental setup. We adopt training, evaluation and tuning setup from [[21](#bib.bib21)]. We tune hyperparameters for most methods777Trompt is the only exception, due to the method’s time complexity. For Trompt we evaluate the default configuration proposed in the respective paper. using Optuna from Akiba et al. [[2](#bib.bib2)], for DL models we use AdamW optimizer and optimize MSE loss or binary cross entropy depending on the datasets. We keep our training protocol simple and do not employ additional training techniques for DL models, like pretraining, custom loss functions, augmentations and many others. We randomly subsample large datasets (Homecredit Default, Cooking Time, Delivery ETA and Weather) to make more extensive hyperparameter tuning feasible. For other details regarding data preprocessing, dataset statistics, tuning protocols and hyperparameters, see [Appendix B](#A2 "Appendix B Experimental Setup Extended ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"). Below, we describe a set of models we evaluate on the proposed benchmark.

Non DL Baselines. We include three main implementations of Gradient Boosted Decision Trees: XGBoost [[13](#bib.bib13)], LightGBM [[29](#bib.bib29)] and CatBoost [[41](#bib.bib41)], as well-established non-DL baselines for tabular data prediction. We also include Random Forest [[9](#bib.bib9)] and linear model as the basic simple ML baselines.

Baseline DL Models. We include two baselines from [[19](#bib.bib19)] – MLP and FT-Transformer. We use MLP as the simplest DL baseline, FT-Transformer as a representative attention-based tabular DL model, which is often claimed to be state-of-the-art [e.g. [19](#bib.bib19), [46](#bib.bib46), [36](#bib.bib36), [10](#bib.bib10)].
In addition, we include DCNv2 as it was repeatedly used in real-world production settings as reported by [[52](#bib.bib52), [5](#bib.bib5)]. We hypothesize that the presence of a feature cross-module could bring performance improvements on the benchmark where datasets have more features. We also test alternative MLP-like backbones in ResNet [[19](#bib.bib19)] and SNN [[33](#bib.bib33)].

Numerical Features Embeddings. We include MLP with embeddings for numerical features from Gorishniy et al. [[20](#bib.bib20)]. It was demonstrated that numerical embeddings provide considerable performance improvements on academic datasets, and make simple MLP models compete with attention-based models. We find this technique simple and effective, thus important to evaluate in a new setting.

Trompt. We include a novel tabular DL model from Chen et al. [[12](#bib.bib12)], which was shown to outperform Transformer for tabular data variants [[46](#bib.bib46), [19](#bib.bib19)] on the benchmark from Grinsztajn et al. [[22](#bib.bib22)]. Its strong performance on an established academic benchmark aligns well with our goal of finding out how results obtained on academic benchmarks might generalize to TabReD.

TabR.
A recent addition to the tabular DL model arsenal from Gorishniy et al. [[21](#bib.bib21)], a retrieval-based model, demonstrating impressive performance on commonly used academic benchmarks [[19](#bib.bib19), [22](#bib.bib22)].
We evaluate the TabR-S variation, which does not include numerical embeddings, as we test the efficacy of numerical embeddings via MLP-PLR, mentioned above.
Note that, during training, for a given training object, the vanilla TabR can retrieve nearest neighbors from “future” training objects (i.e. from training objects with greater timestamps).
In the light of the temporal shifts of the TabReD datasets, this can be an issue, since the retrieval from future will not be possible for test data.
Moreover, even without that issue, the similarities between future test objects and past training objects can still be too different from what TabR sees during training.
As a small remedy to these issues, we introduce TabR-S (causal): during training, for a given object, this version of TabR retrieves nearest neighbors only “from the past” (from “older” datapoints), which is more aligned with how TabR will make predictions for the test data.
We hypothesize that a more complete solution could be to immediately add “older” test objects as nearest neighbor candidates for more recent test objects.
This would utilize the unique ability of retrieval-based models to accomodate new data without retraining, which comes at a price of the increased implementation complexity.

OOD Robustness Methods. We also evaluate two methods that aim to mitigate the effect of distribution shift. The first one is DeepCORAL [[48](#bib.bib48)], we adapt the method to the temporal shift setting by bucketing timestamps into different domains, similar to Wild-Time [[53](#bib.bib53)]. The second method is Deep Feature Reweighting (DFR) [[32](#bib.bib32)], we adapt the method by finetuning the representation of the MLP baseline on the latter instances of the train dataset.

### 4.1 How do recent tabular DL advances transfer to a more realistic evaluation setting?

We evaluate all abovementioned methods to answer Q1. Results are summarized in [Table 2](#S4.T2 "Table 2 ‣ 4.1 How do recent tabular DL advances transfer to a more realistic evaluation setting? ‣ 4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild").

Table 2: Performance comparison of tabular ML models on new datasets. Bold entries represent the best methods on each dataset, with standard deviations over 15 seeds taken into account. The last column contains algorithm rank averaged over all datasets (for details, see the [subsection B.2](#A2.SS2 "B.2 Tuning, Evaluation, Model comparisons ‣ Appendix B Experimental Setup Extended ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild")).
The ranks in bold correspond to the top-3 classical ML methods and the top-3 DL methods.

  

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | Classification (ROC AUC ↑) | | | Regression (RMSE ↓) | | | | | Average Rank |
| Homesite Insurance | Ecom Offers | HomeCredit Default | Sberbank Housing | Cooking Time | Delivery ETA | Maps Routing | Weather |
| Classical ML Baselines | | | | | | | | | |
| XGBoost | 0.9601 | 0.5763 | 0.8670 | 0.2419 | 0.4823 | 0.5468 | 0.1616 | 1.4671 | 2.2 ±plus-or-minus\pm 1.3 |
| LightGBM | 0.9603 | 0.5758 | 0.8664 | 0.2468 | 0.4826 | 0.5468 | 0.1618 | 1.4625 | 2.6 ±plus-or-minus\pm 1.2 |
| CatBoost | 0.9606 | 0.5596 | 0.8621 | 0.2482 | 0.4823 | 0.5465 | 0.1619 | 1.4688 | 2.8 ±plus-or-minus\pm 1.4 |
| RandomForest | 0.9570 | 0.5764 | 0.8269 | 0.2640 | 0.4884 | 0.5959 | 0.1653 | 1.5838 | 6.4 ±plus-or-minus\pm 1.7 |
| Linear | 0.9290 | 0.5665 | 0.8168 | 0.2509 | 0.4882 | 0.5579 | 0.1709 | 1.7679 | 7.4 ±plus-or-minus\pm 2.2 |
| Deep Learning Methods | | | | | | | | | |
| MLP | 0.9500 | 0.6015 | 0.8545 | 0.2508 | 0.4820 | 0.5504 | 0.1622 | 1.5470 | 3.6 ±plus-or-minus\pm 1.6 |
| SNN | 0.9492 | 0.5996 | 0.8551 | 0.2858 | 0.4838 | 0.5544 | 0.1651 | 1.5649 | 5.5 ±plus-or-minus\pm 1.8 |
| DCNv2 | 0.9392 | 0.5955 | 0.8466 | 0.2770 | 0.4842 | 0.5532 | 0.1672 | 1.5782 | 6.4 ±plus-or-minus\pm 2.1 |
| ResNet | 0.9469 | 0.5998 | 0.8493 | 0.2743 | 0.4825 | 0.5527 | 0.1625 | 1.5021 | 4.6 ±plus-or-minus\pm 1.7 |
| FT-Transformer | 0.9622 | 0.5775 | 0.8571 | 0.2440 | 0.4820 | 0.5542 | 0.1625 | 1.5104 | 3.5 ±plus-or-minus\pm 1.5 |
| MLP-PLR | 0.9621 | 0.5957 | 0.8568 | 0.2438 | 0.4812 | 0.5527 | 0.1616 | 1.5177 | 2.5 ±plus-or-minus\pm 1.3 |
| TabR | 0.9487 | 0.5943 | 0.8501 | 0.2820 | 0.4828 | 0.5514 | 0.1639 | 1.4666 | 4.8 ±plus-or-minus\pm 1.6 |
| TabR (causal) | 0.9522 | 0.5850 | 0.8484 | 0.2851 | 0.4825 | 0.5541 | 0.1637 | 1.4622 | 4.5 ±plus-or-minus\pm 1.5 |
| Trompt | 0.9546 | 0.5792 | 0.8381 | 0.2596 | 0.4834 | 0.5563 | 0.1652 | 1.5722 | 5.9 ±plus-or-minus\pm 1.5 |
| OOD Robustness Methods | | | | | | | | | |
| CORAL | 0.9498 | 0.6004 | 0.8549 | 0.2645 | 0.4821 | 0.5498 | 0.1622 | 1.5591 | 3.9 ±plus-or-minus\pm 1.8 |
| DFR | 0.9499 | 0.6013 | 0.8545 | 0.2494 | 0.4819 | 0.5515 | 0.1626 | 1.5513 | 4.0 ±plus-or-minus\pm 1.7 |

GBDT and MLP with embeddings (MLP-PLR) are the overall best models on the TabReD benchmark.
These findings suggest that numerical feature embeddings [[20](#bib.bib20)], which have shown success in academic datasets, maintain their utility in the new evaluation scenario.

FT-Transformer is a runner-up, however, it can be slower to train because of the attention module that causes quadratic scaling of computational complexity w.r.t. the number of features.
The latter point is relevant for TabReD, since the TabReD datasets have more features than an average academic dataset.
Thus, during hyperparameter tuning, trying 25 configurations of FT-Transformer takes two days on average on TabReD, compared to four hours for trying 100 configurations of MLP-PLR.

DCNv2, SNN, ResNet and Trompt are no better than the MLP baseline. Although Trompt showed promising results on a benchmark from Grinsztajn et al. [[22](#bib.bib22)], it failed to generalize to TabReD.
Furthermore, efficiency-wise, Trompt is significantly slower than MLP, and even slower than FT-Transformer.

OOD Robustness. Both DFR and DeepCORAL do not improve upon the MLP baseline, in line with recent work by Gardner et al. [[18](#bib.bib18)], Kolesnikov [[35](#bib.bib35)] for other distribution shifts.

Retrieval-Based Models prove to be less performant on TabReD.
One notable exception is the Weather dataset, where TabR demonstrated the best result.
When describing TabR in [section 4](#S4 "4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"), we explained why TabReD can be a challenging benchmark for retrieval-based models, and outlined potential improvements for such models, but we leave them for future work.

Less Pronounced Performance Margins. Overall, we observe less pronounced performance differences from architecture improvements on TabReD. For example, the average percentage improvement of XGBoost over the MLP model is 3.1%, 2.7% and 1% on the Grinsztajn et al. [[22](#bib.bib22)], Gorishniy et al. [[21](#bib.bib21)] and TabReD benchmarks respectively. Tree-based methods have the smallest margin on TabReD, despite the fact that most data preparation pipelines of these datasets were tailored with GBDT models (a popular choice on Kaggle and in many production use cases). Similar relation holds for relative gains of the FT-Transformer and numerical feature embeddings.

### 4.2 Influence of Data Validation Splits on Model Ranking

In this section, we attempt to answer Q2. For this we create three time-based splits (with a sliding window over all samples) and three corresponding randomly shuffled splits, keeping train, validation and test sizes the same. We average all results over 15 random initialization seeds and three data splits. We are interested in methods ranking and relative performance differences between random and time-based test splits. For this experiment, we consider MLP, MLP-PLR, XGBoost and TabR-S. This selection covers multiple different paradigms: retrieval-based models, parametric DL and strong non-DL baselines, and diverse performance on our benchmark (see [Table 2](#S4.T2 "Table 2 ‣ 4.1 How do recent tabular DL advances transfer to a more realistic evaluation setting? ‣ 4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild")). Results are summarized in [Figure 1](#S4.F1 "Figure 1 ‣ 4.2 Influence of Data Validation Splits on Model Ranking ‣ 4 Experiments ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"). Below, we highlight key takeaways.

![Refer to caption](/html/2406.19380/assets/x1.png)


Figure 1: Comparison of performance on out-of-time and random in-domain test sets. The first row contains regression datasets, the metric is RMSE (lower is better). The second row contains binary classification datasets, the metric is AUC-ROC (higher is better). We can see the change in relative ranks and performance difference in addition to the overall performance drop. In particular, XGBoost lead decreases when comparing performance on task-appropriate time-shifted test sets.

Temporal Shift Influence on Performance. We see that the spread of scores depending on random initialization for each model and data split is generally larger on the time-split based test set (most clear on Sberbank Housing, Delivery ETA and Ecom Offers). This indicates that temporal shift is present in the proposed datasets. As the random splits are commonplace in current academic benchmarks, they could present overly optimistic performance estimates to what one might expect in real-world application scenarios.

Temporal Shift Influence on Ranking and Relative Difference. We can see that the ranking of different model categories and the spreads between model performance scores change when we use randomly split test sets. One notable example is XGBoost decreasing its performance margin to MLPs when evaluated on temporally shifted test sets (Sberbank Housing, Cooking Time, Delivery ETA, Weather, Ecom Offers and Quote Conversion datasets – most clearly seen comparing MLP (PLR) with XGBoost). This might indicate that GBDTs are less robust to shifts, conversely performing better on random splits, by possibly exploiting time-based leakage. Another notable example is TabR-S outperforming the baseline MLP (Cooking Time and Homesite Insurance) and even XGBoost (Weather). These findings prove that curating datasets that represent real-world use cases is important for the continued stable progress in the field of tabular machine learning and further research adoption.

Summary.
From the above results, we conclude that time-based data splits are important for proper evaluation.
Indeed, the choice of the splitting strategy can have a significant effect on all aspects involved in the comparison between models: absolute metric values, relative difference in performance, standard deviations and, finally, the relative ranking of models.

## 5 Limitations and Future Work

We recognize that the TabReD benchmark is biased towards industry-relevant ML applications, with large sample sizes, extensive feature engineering and temporal data drift.
Also, TabReD does not cover some important domains such as medicine, science, and social data.
Finally, we note that the lack of precise feature information may limit some potential future applications of these datasets, like leveraging feature names and descriptions with LLMs.

We provide initial experiments on TabReD demonstrating the importance of taking temporal shifts into account and benchmarking a wide range of tabular DL algorithms and even OOD robustness methods. There are still many research questions and techniques like continual learning, gradual temporal shift mitigation, missing data imputation and feature selection, that could be explored with TabReD.

Another important future research direction is creating a clean set of smaller/simpler academic datasets, taking the issues like data leakage and limiting the use of non-tabular datasets, which we discuss in [subsection 3.1](#S3.SS1 "3.1 A Closer Look at Existing Tabular ML Benchmarks ‣ 3 TabReD Benchmark ‣ TabReD: A Benchmark of Tabular Machine Learning in-the-Wild"). As we demonstrated, some progress on academic benchmarks transfers to the in-the-wild settings present in TabReD, and simple academic benchmarks could still be useful for fast exploration.

## 6 Conclusion

In this work, we introduced the TabReD benchmark, a collection of eight industry-grade tabular datasets with temporal train/validation/test splits and feature engineering to better represent real-world tabular ML deployment scenarios.
Our experiments show that accounting for temporal shifts during evaluation significantly impacts model performance and ranking.
We evaluated recent tabular DL methods on TabReD datasets to find out how well they perform in the presence of time-based data splits and more features.
The results highlight MLP with embeddings for continuous features [[20](#bib.bib20)] as a simple and effective DL baseline.
More advanced DL models, however, demonstrated less convincing performance in the considered setting.

In summary, we demonstrate the need for representative benchmarks for enabling robust evaluation and progress in tabular ML. TabReD is an initial step toward realistic, task-relevant benchmarks. We hope our work steers the field toward tackling substantively important real-world problems using properly vetted protocols.

Broader Impact.
This paper aims to contribute to the progression of Tabular Machine Learning research. A key focus of this work is the benchmarking of Tabular ML methods in academic settings. If successful, this research could potentially accelerate the transition of academic findings into real-world applications, which could have both positive and negative impacts on society. However, we firmly advocate for cautious and robust advancements in this field. We believe that such progress would be positive in the end.

## References

* 50806198 [2017]

  Anastasia Sidorova 50806198.
  Additional data - tverskoe issue, 2017.
  URL <https://www.kaggle.com/competitions/sberbank-russian-housing-market/discussion/34364>.
* Akiba et al. [2019]

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, 2019.
* Alexey Matveev [2017]

  DataCanary Alexey Matveev, Anastasia Sidorova 50806198.
  Sberbank russian housing market, 2017.
  URL <https://kaggle.com/competitions/sberbank-russian-housing-market>.
* alijs and johnpateha [2017]

  alijs and johnpateha.
  Sberbank russian housing market 1st place solution, 2017.
  URL <https://www.kaggle.com/competitions/sberbank-russian-housing-market/discussion/35684>.
* Anil et al. [2022]

  Rohan Anil, Sandra Gadanho, Da Huang, Nijith Jacob, Zhuoshu Li, Dong Lin, Todd Phillips, Cristina Pop, Kevin Regan, Gil I. Shamir, Rakesh Shivanna, and Qiqi Yan.
  On the factory floor: Ml engineering for industrial-scale ads recommendation models, 2022.
* Bahri et al. [2021]

  Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler.
  Scarf: Self-supervised contrastive learning using random feature corruption.
  In *ICLR*, 2021.
* Ballesteros [2019]

  Alexander Guillermo Segura Ballesteros.
  Openml airlines dataset, 2019.
  URL <https://openml.org/d/41672>.
* Baranchuk et al. [2023]

  Dmitry Baranchuk, Matthijs Douze, Yash Upadhyay, and I Zeki Yalniz.
  Dedrift: Robust similarity search under content drift.
  In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 11026–11035, 2023.
* Breiman [2001]

  Leo Breiman.
  *Mach. Learn.*, 45(1):5–32, 2001.
* Chen et al. [2023a]

  Jintai Chen, Jiahuan Yan, Danny Ziyi Chen, and Jian Wu.
  Excelformer: A neural network surpassing gbdts on tabular data, 2023a.
* Chen et al. [2023b]

  Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang.
  Trompt: Towards a better deep neural network for tabular data.
  *arXiv preprint arXiv:2305.18446*, 2023b.
* Chen et al. [2023c]

  Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang.
  Trompt: Towards a better deep neural network for tabular data.
  In *ICML*, volume 202 of *Proceedings of Machine Learning Research*, pages 4392–4434. PMLR, 2023c.
* Chen and Guestrin [2016]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *SIGKDD*, 2016.
* Darrel [2015]

  Will Cukierski Darrel, Stephen D Stayton.
  Homesite quote conversion, 2015.
  URL <https://kaggle.com/competitions/homesite-quote-conversion>.
* DMDave [2014]

  Will Cukierski DMDave, Todd B.
  Acquire valued shoppers challenge, 2014.
  URL <https://kaggle.com/competitions/acquire-valued-shoppers-challenge>.
* Feuer et al. [2024]

  Benjamin Feuer, Robin Tibor Schirrmeister, Valeriia Cherepanova, Chinmay Hegde, Frank Hutter, Micah Goldblum, Niv Cohen, and Colin White.
  Tunetables: Context optimization for scalable prior-data fitted networks.
  *arXiv preprint arXiv:2402.11137*, 2024.
* Fu and Soman [2021]

  Yupeng Fu and Chinmay Soman.
  Real-time data infrastructure at uber.
  In *Proceedings of the 2021 International Conference on Management of Data*, pages 2503–2516, 2021.
* Gardner et al. [2023]

  Joshua P Gardner, Zoran Popovi, and Ludwig Schmidt.
  Benchmarking distribution shift in tabular data with tableshift.
  In *Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2023.
* Gorishniy et al. [2021]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, 2021.
* Gorishniy et al. [2022]

  Yury Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  In *NeurIPS*, 2022.
* Gorishniy et al. [2024]

  Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko.
  Tabr: Tabular deep learning meets nearest neighbors in 2023.
  In *ICLR*, 2024.
* Grinsztajn et al. [2022]

  Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux.
  Why do tree-based models still outperform deep learning on typical tabular data?
  In *NeurIPS, the ”Datasets and Benchmarks” track*, 2022.
* Herman et al. [2024]

  Daniel Herman, Tomas Jelinek, Walter Reade, Maggie Demkin, and Addison Howard.
  Home credit - credit risk model stability, 2024.
  URL <https://kaggle.com/competitions/home-credit-credit-risk-model-stability>.
* Hollmann et al. [2023]

  Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.
  Tabpfn: A transformer that solves small tabular classification problems in a second.
  In *ICLR*, 2023.
* Huyen [2022]

  C. Huyen.
  *Designing Machine Learning Systems*.
  O’Reilly Media, 2022.
  ISBN 9781098107918.
  URL <https://books.google.ru/books?id=EThwEAAAQBAJ>.
* Jeffares et al. [2023]

  Alan Jeffares, Tennison Liu, Jonathan Crabbé, Fergus Imrie, and Mihaela van der Schaar.
  Tangos: Regularizing tabular neural networks through gradient orthogonalization and specialization.
  In *ICLR*, 2023.
* Ji et al. [2023]

  Christina X Ji, Ahmed M Alaa, and David Sontag.
  Large-scale study of temporal shift in health insurance claims.
  In *Conference on Health, Inference, and Learning*, pages 243–278. PMLR, 2023.
* Kakade [2021]

  Vinay Kakade.
  Ml feature serving infrastructure at lyft, 2021.
  URL <https://eng.lyft.com/ml-feature-serving-infrastructure-at-lyft-d30bf2d3c32a>.
* Ke et al. [2017a]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30:3146–3154, 2017a.
* Ke et al. [2017b]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  *Advances in neural information processing systems*, 30:3146–3154, 2017b.
* Kim [2024]

  SeungYun Kim.
  fork-of-home-credit-catboost-inference kaggle kernel, 2024.
  URL <https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-catboost-inference>.
* Kirichenko et al. [2023]

  Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson.
  Last layer re-training is sufficient for robustness to spurious correlations.
  In *The Eleventh International Conference on Learning Representations*, 2023.
  URL <https://openreview.net/forum?id=Zb6c8A-Fghk>.
* Klambauer et al. [2017]

  Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
  Self-normalizing neural networks.
  In *NIPS*, 2017.
* Kohli et al. [2024]

  Ravin Kohli, Matthias Feurer, Katharina Eggensperger, Bernd Bischl, and Frank Hutter.
  Towards quantifying the effect of datasets for benchmarking: A look at tabular machine learning.
  2024.
* Kolesnikov [2023]

  Sergey Kolesnikov.
  Wild-tab: A benchmark for out-of-distribution generalization in tabular regression, 2023.
* Kossen et al. [2021]

  Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin Gal.
  Self-attention between datapoints: Going beyond individual input-output pairs in deep learning.
  In *NeurIPS*, 2021.
* Luccioni et al. [2022]

  Alexandra Sasha Luccioni, Frances Corry, Hamsini Sridharan, Mike Ananny, Jason Schultz, and Kate Crawford.
  A framework for deprecating datasets: Standardizing documentation, identification, and communication.
  In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*, pages 199–212, 2022.
* Malinin et al. [2021]

  Andrey Malinin, Neil Band, German Chesnokov, Yarin Gal, Mark John Francis Gales, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, Vatsal Raina, Vyas Raina, Mariya Shmatova, Panos Tigas, and Boris Yangel.
  Shifts: A dataset of real distributional shift across multiple large-scale tasks.
  *ArXiv*, abs/2107.07455v3, 2021.
* McElfresh et al. [2023]

  Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Ganesh Ramakrishnan, Micah Goldblum, Colin White, et al.
  When do neural nets outperform boosted trees on tabular data?
  *arXiv preprint arXiv:2305.02997*, 2023.
* MLWave [2014]

  github MLWave.
  kaggle-acquire-valued-shoppers-challenge, 2014.
  URL <https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge>.
* Prokhorenkova et al. [2018]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, 2018.
* Rubachev et al. [2022]

  Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko.
  Revisiting pretraining objectives for tabular deep learning.
  *arXiv preprint arXiv:2207.03208*, 2022.
* Saitta and Neri [1998]

  Lorenza Saitta and Filippo Neri.
  Learning in the “real world”.
  *Machine learning*, 30:133–163, 1998.
* Shani and Gunawardana [2011]

  Guy Shani and Asela Gunawardana.
  Evaluating recommendation systems.
  *Recommender systems handbook*, pages 257–297, 2011.
* Simha [2020]

  Nikhil Simha.
  Zipline - a declarative feature engineering framework, 2020.
  URL <https://youtu.be/LjcKCm0G_OY>.
* Somepalli et al. [2021]

  Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein.
  SAINT: improved neural networks for tabular data via row attention and contrastive pre-training.
  *arXiv*, 2106.01342v1, 2021.
* Stein [2002]

  Roger M Stein.
  Benchmarking default prediction models: Pitfalls and remedies in model validation.
  *Moody’s KMV, New York*, 20305, 2002.
* Sun and Saenko [2016]

  Baochen Sun and Kate Saenko.
  Deep coral: Correlation alignment for deep domain adaptation.
  In *European Conference on Computer Vision*, pages 443–450. Springer, 2016.
* van Rijn [2014]

  Jan van Rijn.
  Openml electricity dataset, 2014.
  URL <https://openml.org/d/42712>.
* van Rijn [2020]

  Jan van Rijn.
  Openml bike sharing demand dataset, 2020.
  URL <https://openml.org/d/151>.
* Wagstaff [2012]

  Kiri L Wagstaff.
  Machine learning that matters.
  In *Proceedings of the 29th International Coference on International Conference on Machine Learning*, pages 1851–1856, 2012.
* Wang et al. [2020]

  Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi.
  Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.
  *arXiv*, 2008.13535v2, 2020.
* Yao et al. [2022]

  Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei W Koh, and Chelsea Finn.
  Wild-time: A benchmark of in-the-wild distribution shift over time.
  *Advances in Neural Information Processing Systems*, 35:10309–10324, 2022.

All artifacts from the supplementary materials are available in the GitHub repository for the benchmark
  
<https://github.com/yandex-research/tabred>

## Appendix A Benchmark Details

We provide the datasheet in the datasheet.md file in the accompanying source code. All datasets are available on Kaggle.

## Appendix B Experimental Setup Extended

### B.1 Source Code

We include source code for reproducing the results in the supplementary material archive with a brief README.md with instructions on reproducing the experiments. We publish code for dataset preparation for Kaggle-based datasets and publish new raw datasets on the Kaggle platform.

### B.2 Tuning, Evaluation, Model comparisons

We replace NaN values with the mean value of a variable (zero after the quantile normalization). For categorical features with unmatched values in validation and test sets we encode such values as a special unknown category.

In tuning and evaluation setup, we closely follow the procedure described in [[19](#bib.bib19), [21](#bib.bib21)].

When comparing the models we take standard deviations over 15 random initializations into account. We rank method A below method B if |Bmean−Amean|<BstddevsubscriptBmeansubscriptAmeansubscriptBstddev|\text{B}\_{\text{mean}}-\text{A}\_{\text{mean}}|<\text{B}\_{\text{stddev}} and B score is better. We run hyperparameter optimization for 100 iterations for most models, the exceptions are FT-Transformer (which is significantly less efficient on datasets with hundreds of features) where we were able to run 25

For the exact hyperparameter ranges see the source code. Tuning configs (exp/\*\*/tuning.toml) together with the code are always the main sources of truth.

### B.3 Additional Implementation Details

We have taken each method’s implementation from the respective official code sources, except for Trompt, which doesn’t have an official code repository yet and instead was reproduced by us according to the information in the paper. We use default hyperparameters for the Trompt model from the paper [[12](#bib.bib12)].

For the DFR [[32](#bib.bib32)] baseline we finetune the last layer on the last 20% of datapoints.

For CORAL [[48](#bib.bib48)] we define domains by splitting instances based on a timestamp variable into 9 chunks.

## Appendix C Kaggle Competitions Table

We provide a annotated-kaggle-competitions.csv with annotated Kaggle competitions in the supplementary repository. We used this table during sourcing the datasets from Kaggle. There are minimal annotations and notes. We annotate only tabular competitions with more than 500 competitors. We provide annotations for non-tabular datasets in this table.

## Appendix D TabReD Datasets Overview

In this section, we give a basic overview for each dataset from the proposed benchmark.

### [Homesite Insurance](https://www.kaggle.com/competitions/homesite-quote-conversion/overview)

Task: Classification

#Samples: 260,753     #Features: 299     Year: 2015

Comments: This is a dataset from a Kaggle competition hosted by Homesite Insurance [[14](#bib.bib14)]. The task is predicting whether a customer will buy a home insurance policy based on user and insurance policy features (user, policy, sales and geographic information). Each row in the dataset corresponds to a potential [customer, policy] pair, the target indicates whether a customer bought the policy. Timeframe from 2013-01-01 to 2015-05-18.

### [Ecom Offers](https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/overview)

Task: Classification

#Samples: 160,057     #Features: 119     Year: 2014

Comments: This is a dataset from a Kaggle competition hosted by the online book and game retailer DMDave [[15](#bib.bib15)]. The task in this dataset is a representative example of modeling customer loyalty in e-commerce. Concretely, the task is classifying whether a customer will redeem a discount offer based on features from two months’ worth of transaction history. We base our feature engineering on one of the top solutions [[40](#bib.bib40)]. Timeframe from 2013-03-01 to 2013-04-30.

### [Homecredit Default](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

Task: Classification

#Samples: 1,526,659     #Samples Used 381,664     #Features: 696     Year: 2024

Comments: This is a second iteration of the popular HomeCredit tabular competition [[23](#bib.bib23)]. The task is to predict whether bank clients will default on a loan, based on bank internal information and external information like credit bureau and tax registry data. This year competition focus was the model prediction stability over time. Compared to the more popular prior competition, this time there is more data and the timestamps are available.
We base feature engineering and preprocessing code on top solutions [[31](#bib.bib31)]. Timeframe from 2019-01-01 to 2020-10-05.

### [Sberbank Housing](https://www.kaggle.com/competitions/sberbank-russian-housing-market)

Task: Regression

#Samples: 28,321     #Features: 392     Year: 2017

Comments: This dataset is from a Kaggle competition, hosted by Sberbank [[3](#bib.bib3)]. This dataset provides information about over 30000 transactions made in the Moscow housing market. The task is to predict the sale price of each property using the provided features describing each property condition, location, and neighborhood, as well as country economic indicators at the moment of the sale. We base our preprocessing code on discussions and solutions from the competition [[4](#bib.bib4), [1](#bib.bib1)]. Timeframe from 2011-08-20 to 2015-06-30.

### [Cooking Time](https://www.kaggle.com/datasets/pcovkrd84mejm/cooking-time)

Task: Regression

#Samples: 12,799,642     #Samples Used 319,986     #Features: 192     Year: 2024

Comments: For this dataset, the task is to determine how long it will take for a restaurant to prepare an order placed in a food delivery app. Features are constructed based on the information about the order contents and historical information about cooking time for the restaurant and brand, the target is a logarithm of minutes it took to cook the placed order. Timeframe from 2023-11-15 to 2024-01-03.

### [Delivery ETA](https://www.kaggle.com/datasets/pcovkrd84mejm/delivery-eta)

Task: Regression

#Samples: 17,044,043     #Samples Used 416,451     #Features: 223     Year: 2024

Comments: For this dataset, the task is to determine the estimated time of arrival of an order from
an online grocery store. Features are constructed based on the courier availability, navigation data
and various aggregations of historical information for different time slices, the target is the logarithm
of minutes it took to deliver an order. Timeframe from 2023-10-20 to 2024-01-25.

### [Maps Routing](https://www.kaggle.com/datasets/pcovkrd84mejm/maps-routing)

Task: Regression

#Samples: 13,639,272     #Samples Used 340,981     #Features: 986     Year: 2024

Comments: For this dataset, the task is to predict the travel time in the car navigation system
based on the current road conditions. The features are aggregations of the road graph statistics for the
particular route and various road details (like speed limits). The target is the logarithm of seconds per
kilometre. Timeframe from 2023-11-01 to 2023-12-04.

### [Weather](https://www.kaggle.com/datasets/pcovkrd84mejm/tabred-weather)

Task: Regression

#Samples: 16,951,828     #Samples Used 423,795     #Features: 103     Year: 2024

Comments: For this dataset, the task is weather temperature forecasting. This is a dataset similar to the one introduced in [[38](#bib.bib38)], except it is larger, and the time-based split is available and used by default. The features are from weather station measurements and weather forecast physical models. The target is the true temperature for the moment in time. Timeframe from 2022-07-01 to 2023-07-30.

## Appendix E Detailed Academic Datasets Overview

In this section we go through each dataset, and list its problems and prior uses in literature. We specify whether time-based splits should preferably be used for the dataset and whether it is available (e.g. datasets come with timestamps)

The full table with annotation is available in the root of the code repository in the
  
academic-datasets-summary.csv file.

We also provide commentary and annotations directly in the appendix.

### [100-plants-texture](https://archive.ics.uci.edu/dataset/241/one+hundred+plant+species+leaves+data+set)

Tags: HomE

#Samples: 1599     #Features: 65     Year: 2012

Comments: This is a small dataset with images and image-based features. The dataset first appeared in the paper ”Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features. Signal Processing, Pattern Recognition and Applications”, written by Mallah et al in 2013, and contains texture features extracted from the images of the leaves taken from Royal Botanic Gardens in Kew, UK. The task at hand is to recognize which leave is being described by the given features. While this feature extraction was a beneficial way to handle vision-based information in the early 2010s, modern approaches to CV focus on the specific architectures better suited for image data.

### [ALOI](https://ieeexplore.ieee.org/document/6575194)

Tags: HomE

#Samples: 108000     #Features: 128     Year: 2005

Comments: The dataset describes a collection of images provided by Geusebroek et al. The features used in this version are color histogram values.

### [Adult](https://archive.ics.uci.edu/dataset/2/adult)

Tags: Tabular, Timesplit Needed

#Samples: 48842     #Features: 33     Year: 1994

Comments: One of the most popular tabular datasets, Adult was created by Barry Becker based on the 1994 Census database. The target variable is a binary indicator of whether a person has a yearly income above 50000$.

### [Ailerons](https://openml.org/d/296)

Tags: Raw

#Samples: 13750     #Features: 40     Year: 2014

Comments: This data set addresses a control problem, namely flying an F16 aircraft. The attributes describe the status of the aeroplane, while the goal is to predict the control action on the ailerons of the aircraft. According to the descriptions available, the dataset first appears in a collection of regression datasets by Luis Torgo and Rui Camacho made in 2014, but the original website with the description (ncc.up.pt/ ltorgo/Regression/DataSets.html) seems to no longer respond. The task of controlling a vehicle through machine learning has gained a large interest in recent years, but it is not done through tabular machine learning, instead often utilizing RL and using wider range of sensors than those used in the non-self-driving version of the vehicle.

### [Australian](https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval)

Tags: Tabular, Timesplit Needed

#Samples: 690     #Features: 15     Year: 1987

Comments: Anonymized credit approval dataset. Corresponds to a real-life task, but is very small, with only 15 features, and no way to create a time/or even user-based validation split, no time variable is available

### [Bike\_Sharing\_Demand](https://openml.org/d/42712)

Tags: Leak, Tabular, Timesplit Needed, Timesplit Possible

#Samples: 17379     #Features: 6     Year: 2012

Comments: This dataset was produced based on the data from the Capital Bikeshare system from 2011 and 2012. The task is to predict the count of bikes in use based on time and weather conditions. No forecasting FE is done, only weather and date are available, forecasting tasks in the real-world, if solved by tabular models involve extensive feature engineering. While the task is to predict demand at a specific time, the time-based split is not performed, although it is possible. Due to the random i.i.d. split of the dataset, while predicting on a test object models could use information from the train examples close in time to the test one, which wouldn’t be possible in real-life conditions, since a model is used after it is trained.

### [Bioresponse](https://www.kaggle.com/c/bioresponse/overview)

Tags: HomE

#Samples: 3751     #Features: 1777     Year: 2012

Comments: These datasets present a classification problem on molecules. The underlying data is not tabular, graph-based methods, incorporating 3d structure are known to outperform manual descriptors (https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2), this is not a mainstream task in its formulation. For more up to date, datasets and classification tasks on molecules one could use https://moleculenet.org for example

### [Black Friday](https://datahack.analyticsvidhya.com/contest/black-friday/#ProblemStatement)

Tags: Timesplit Needed

#Samples: 166821     #Features: 9     Year: 2019

Comments: No time split, predicting customer’s purchase amount from demographic features. ”A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summaries of various customers for selected high-volume products from last month.
The data set also contains customer demographics (age, gender, marital status, city\_type, stay\_in\_current\_city), product details (product\_id and product category) and Total purchase\_amount from last month.” No way to check if the dataset is real. Potential leakage. The task looks artificial we need to predict the purchase amount based on 6 users and 3 product features (there are 5k users and 3k unique users and products on 160k samples, the mean price for a product is a strong baseline).

### [Brazilian\_houses](https://openml.org/d/42688)

Tags: Tabular, Timesplit Needed

#Samples: 10692     #Features: 8     Year: 2020

Comments: Similar to other housing market prediction tasks, the data is a snapshot of listings on a Brazilian website with houses to rent. No way to create a time split.

### [Broken Machine](https://en.wikipedia.org/wiki/HTTP_404)

Tags: Synthetic or Untraceable, Tabular

#Samples: 900000     #Features: 58     Year: 2021?

Comments: This dataset was not described anywhere, and the link to the original publication on Kaggle is no longer working.

### [California Housing](https://www.sciencedirect.com/science/article/abs/pii/S016771529600140X)

Tags: Tabular, Timesplit Needed

#Samples: 20640     #Features: 8     Year: 1990

Comments: This data comes from 1990 US Census data. Each object describes a block group, which on average includes 1425.5 individuals. The features include information about housing units inside a block group, as well as median income reported in the area. The target variable is the ln of a median house price. Due to the fact that the target variable is averaged across a large number of houses, KNN algorithms using the coordinates of a block are very effective. Housing prices are quickly changing in time, which presents an additional challenge for tabular ML models, however, on this dataset a time-based split is impossible. The provided features are also shallow in comparison to a dataset that may be used in an industrial scenario, e.g. housing dataset included in our publication includes hundreds of features as opposed to this dataset, which includes only 8.

### [Churn Modelling](https://www.kaggle.com/datasets/shubh0799/churn-modelling/data)

Tags: Synthetic or Untraceable, Tabular, Timesplit Needed

#Samples: 10000     #Features: 11     Year: 2020

Comments: This dataset describes a set of customers of a bank, with a task of classifying whether a user will stay with the bank. Not a time split. Unknown source (may be synthetic). Not rich information. Narrow, No License. No canonical split (No time dimension)

### [Epsilon](https://k4all.org/project/large-scale-learning-challenge/)

Tags: Synthetic or Untraceable, Tabular

#Samples: 500000     #Features: 2000     Year: 2008

Comments: This dataset comes from a 2008 competition ”Large Scale Learning Challenge” by the K4all foundation. The source of the data is unclear, the dataset might be synthetic.

### [Facebook Comments Volume](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset)

Tags: Leak, Tabular, Timesplit Needed

#Samples: 197080     #Features: 51     Year: 2016

Comments: This dataset presents information about a facebook post and the target is to determine how many comments will appear within a period of time. Leakage. Same comments from different points in time, random split is inappropriate. This case is described in the appendix of a TabR paper, as this model was able to exploit the leak do get extreme performance improvements

### [Gesture Phase](https://www.sciencedirect.com/science/article/pii/S0957417416300525)

Tags: Leak, Tabular, Timesplit Needed, Timesplit Possible

#Samples: 9873     #Features: 32     Year: 2016

Comments: The task of this dataset is to classify gesture phases. Features are the speed and the acceleration from kinect. There are 7 videos from 3 users (3 gesture sequences from 2 and one from an additional user). The paper, which introduced the dataset mentions that using the same user (but a different story) for evaluation influences the score. Tabular DL papers, use random split on this dataset – this is not assessing the performance on new users, not even on new sequences of one user, not a canonical split. Without canonical split, the task contains leakage, which is easily exploited by using retrieval methods or overtuning models.

### [Helena](http://automl.chalearn.org)

Tags: Synthetic or Untraceable, HetE

#Samples: 65196     #Features: 27     Year: 2018

Comments: The data was provided by AutoML challenge, and the dataset was created from objects from another domain, such as text, audio, or video, compressed into tabular form.

### [Higgs](https://openml.org/d/42769)

Tags: Tabular

#Samples: 940160     #Features: 24     Year: 2014

Comments: Physics simulation data.

### [House 16H](https://www.dcc.fc.up.pt/%C2%A0ltorgo/Regression/census.html)

Tags: Tabular, Timesplit Needed

#Samples: 22784     #Features: 16     Year: 1990

Comments: No time features, comes from the US Census 1990. Feature selection was performed, non correlated features were selected for house 16H(hard). Narrow, by definition, less important features. Learning problem, not the most representative for the real world task

### [Jannis](http://automl.chalearn.or)

Tags: HetE

#Samples: 83733     #Features: 54     Year: 2018

Comments: The data was provided by AutoML challenge, and the dataset was created from objects from another domain, such as text, audio, or video, compressed into tabular form.

### [KDDCup09\_upselling](http://proceedings.mlr.press/v7/guyon09/guyon09.pdf)

Tags: Tabular, Timesplit Needed

#Samples: 5032     #Features: 45     Year: 2009

Comments: Real-world data and problem from Orange telecom company, the taks is binary classification of upselling. All variables are anonyimzed, the time is not available (but predictions in this problem do happen in the future) only i.i.d train with labels is available

### [MagicTelescope](https://openml.org/d/1120)

Tags: Tabular

#Samples: 13376     #Features: 10     Year: 2004

Comments: Physics simulation

### [Mercedes\_Benz\_Greener\_Manufacturing](https://openml.org/d/42570)

Tags: Tabular

#Samples: 4209     #Features: 359     Year: 2017

Comments: This dataset presents features about a Mercedes car, the task is to determine the time it will take to pass testing.

### [MiamiHousing2016](https://openml.org/d/43093)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 13932     #Features: 14     Year: 2016

Comments: The dataset comes from publicly available information on house sales in Miami in 2016. While this dataset has several improvements when compared to the california housing dataset, such as not averaging prices in a block, as well as availability of date of sale, the features are still shallow when compared to the housing dataset presented in this paper.

### [MiniBooNE](https://openml.org/d/41150)

Tags: Tabular

#Samples: 72998     #Features: 50     Year: 2005

Comments: Physics simulation

### [OnlineNewsPopularity](https://openml.org/d/42724)

Tags: Timesplit Needed

#Samples: 39644     #Features: 59     Year: 2015

Comments: This dataset contains information about articles published by Mashable, and posits a task of predicting number of shared of each article. Features are mostly NLP related, e.g. LDA and number of specific keywords. Would be better solved by NLP approaches.

### [Otto Group Products](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 61878     #Features: 93     Year: 2015

Comments: This data comes from 2015 kaggle competition hosted by The Otto Group. The objective is to classify a product’s category. Each row corresponds to a single product. There are a total of 93 numerical features, which represent counts of different events. All features have been obfuscated and will not be defined any further. No time meta-feature, no way to ensure there is no time leak (what are the events? what if the distribution of these counts shifts over time?). No canonical split available, no details on the competition website on the nature of the features

### [SGEMM\_GPU\_kernel\_performance](https://openml.org/d/43144)

Tags: Leak, Tabular

#Samples: 241600     #Features: 9     Year: 2018

Comments: Leakage. The task is to predict the time that it takes to multiply two matrices, but 3 out of 4 target variables are given. With them included, all other features have zero random forest importance.

### [Santander Customer Transactions](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)

Tags: Tabular, Timesplit Needed

#Samples: 200000     #Features: 200     Year: 2019

Comments: The data comes from 2019 kaggle competition by Santander. The task is to predict whether a customer will make a specific transaction. Performed processing is unknown. Time-based split is appropriate but not possible to perform.

### [Shifts Weather (in-domain-subset)](https://github.com/Shifts-Project/shifts)

Tags: Leak, Tabular, Timesplit Possible

#Samples: 397099     #Features: 123     Year: 2021

Comments: The dataset first appeared in the 2021 paper concerning distributional shift. Leakage. In-domain version used. Samples from the future used for prediction. Retrieval methods such as TabR achieve large performance improvements.

### [SpeedDating](https://www.openml.org/search?type=data&status=active&id=40536)

Tags: Tabular, Timesplit Needed

#Samples: 8378     #Features: 121     Year: 2004

Comments: This dataset describes experimental speed dating events that took place from 2002 to 2004. The data describes the responses of participants to a questionnaire, and the target variable is whether they matched or not.

### [Tableshift ASSISTMents](https://new.assistments.org/)

Tags: Tabular, Timesplit Needed

#Samples: 2600000     #Features: 16     Year: 2013

Comments: Predict whether the student answers correctly. Features include: student-, problem-, and school-level features, the dataset also contains affect predictions for students based on an experimental affect detector implemented in ASSISTments. Timesplit is not possible.

### [Tableshift Childhood Lead](https://www.cdc.gov/nchs/nhanes/index.htm)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 27000     #Features: 8     Year: 2023

Comments: The data comes from CDC National Health and Nutrition Examination Survey, and the task in this dataset is to predict whether a person has high blood lead levels based on answers to a questionnaire.

### [Tableshift Colege Scorecard](https://collegescorecard.ed.gov/)

Tags: Tabular, Timesplit Needed

#Samples: 124699     #Features: 119     Year: 2023

Comments: The task is to predict the completion rate for a college. The College Scorecard is an institution-level dataset compiled by the U.S. Department of Education from 1996-present

### [Tableshift Diabetes](https://www.cdc.gov/brfss/index.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 1444176     #Features: 26     Year: 2021

Comments: Determine Diabetes diagnosis from a telephone survey. We use data provided by the Behavioral Risk Factors Surveillance System (BRFSS). BRFSS is a large-scale telephone survey conducted by the Centers of Disease Control and Prevention.

### [Tableshift Food Stamps](https://www.census.gov/programs-surveys/acs/about.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 840582     #Features: 21     Year: 2023

Comments: Data source bias (US based surveys), comes from ACS. Narrow. No time split provided in the benchmark version.

### [Tableshift HELOC](https://community.fico.com/s/explainable-machine-learning-challenge)

Tags: Tabular, Timesplit Needed

#Samples: 10000     #Features: 23     Year: 2018

Comments: TableShift uses the Home Equity Line of Credit (HELOC) Dataset from the FICO Explainable Machine Learning Challenge

### [Tableshift Hypertention](https://www.cdc.gov/brfss/index.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 846000     #Features: 14     Year: 2021

Comments: Determine whether a person has hypertension from a telephone survey. We use data provided by the Behavioral Risk Factors Surveillance System (BRFSS). BRFSS is a large-scale telephone survey conducted by the Centers of Disease Control and Prevention.

### [Tableshift ICU Hospital Mortality](https://arxiv.org/abs/2312.07577)

Tags: Semi-
Tabular, Timesplit Needed

#Samples: 23944     #Features: 7520     Year: 2016

Comments: The data comes from MIMIC-III, describing records from Beth Israel Deaconess Medical Center. The data used in this dataset would be more effectively processed as time series and sequences.

### [Tableshift ICU Length of stay](https://tableshift.org/)

Tags: Semi-
Tabular, Timesplit Needed

#Samples: 23944     #Features: 7520     Year: 2016

Comments: The data comes from MIMIC-III, describing records from Beth Israel Deaconess Medical Center. The data used in this dataset would be more effectively processed as time series and sequences.

### [Tableshift Income](https://www.census.gov/programs-surveys/acs/about.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 1600000     #Features: 15     Year: 2018

Comments: The task is to predict person’s income based on their answers to a survey. Data is provided by American Community Survey.

### [Tableshift Public Coverage](https://www.census.gov/programs-surveys/acs/about.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 5900000     #Features: 11     Year: 2018

Comments: The task is to predict whether a person is covered by public health insurance based on their answers to a survey. Data is provided by American Community Survey.

### [Tableshift Readmission](https://www.hindawi.com/journals/bmri/2014/781670/)

Tags: Tabular, Timesplit Needed

#Samples: 99000     #Features: 47     Year: 2008

Comments: ”This study used the Health Facts database (Cerner Corporation, Kansas City, MO), a national data warehouse that collects comprehensive clinical records across hospitals throughout the United States.” Clinical patient with diabetes data. 47 features with questionnaire like information (num\_previous visits, which medication patients were using, which diagnosis patients had). No time feature available.

### [Tableshift Sepsis](https://ieeexplore.ieee.org/document/9005736)

Tags: Semi-
Tabular, Timesplit Needed

#Samples: 1500000     #Features: 41     Year: 2019

Comments: Predict whether a person will develop sepsis in the next 6 months based on the data about their health, including questionnaire answers and patient records.

### [Tableshift Unemployment](https://www.census.gov/programs-surveys/acs/about.html)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 1700000     #Features: 18     Year: 2018

Comments: The task is to predict whether a person is unemployed based on their answers to a survey. Data is provided by American Community Survey.

### [Tableshift Voting](https://electionstudies.org/)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 8000     #Features: 55     Year: 2020

Comments: The prediction target for this dataset is to determine whether an individual will vote in the U.S presidential election, from a detailed questionnaire. It seems like the data goes all the way back to 1948, which makes this not realistic when not using time split

### [Vessel Power R](https://arxiv.org/pdf/2206.15407)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 554642     #Features: 10     Year: 2022

Comments: The dataset describes information about a shipping line, with the task of determining how much power is needed.

### [Vessel Power S](https://arxiv.org/pdf/2206.15407)

Tags: Synthetic or Untraceable, Tabular, Timesplit Needed, Timesplit Possible

#Samples: 546543     #Features: 10     Year: 2022

Comments: Synthetic version of Vessel Power dataset

### [Year](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)

Tags: HomE

#Samples: 515345     #Features: 90     Year: 2011

Comments: This dataset describes musical compositions, with the target variable being a year in which the composition was created. Another domain (audio features - extracted from audio, thus suitable for tabular DL, but DL for audio on raw data is preferable in this domain. Year prediction, solved as a regression task. Dataset does not correspond to a real-world problem (the year meta-data is easy to obtain, no need for prediciton). Problem with the formulation – solving as a classification problem might be preferable (98

### [ada-agnostic](http://clopinet.com/isabelle/Projects/agnostic/Dataset.pdf)

Tags: Tabular, Timesplit Needed

#Samples: 4562     #Features: 49     Year: 1994

Comments: This dataset is a processed version of the popular Adult dataset. This particular rendition of the well known dataset first appeared in the competition ”Agnostic Learning vs. Prior Knowledge” that took place at IJCNN 2007. The differences with the original Adult include some features or categorical values being dropped and missing values being preprocessed. Overall, this rendition is plagued by the same problems that the original Adult dataset has, making it serve as a duplicate less useful for analysing tabular machine learning in the context of large benchmarks.

### [airlines](https://openml.org/d/41672)

Tags: Tabular, Timesplit Needed

#Samples: 539382     #Features: 8     Year: 2006

Comments: The airlines dataset was created for the Data Expo competition in 2006 by Elena Ikonomovska. Unfortunately, the competition link provided in the secondary sources does not work anymore. The proposed task for the dataset is to predict flight delays based on Airline, flight number, time, source and destination. While the data is sourced in the real world, and the task of predicting the delay of the flight certainly could be solved with tabular deep learning, the provided features lack most information essential to predicting the delay. Time based train/val/test split is important but impossible to produce with this dataset.

### [albert](http://automl.chalearn.org)

Tags: Synthetic or Untraceable, HetE

#Samples: 425240     #Features: 79     Year: 2018

Comments: This is an anonymized dataset with unknown origin. Based on the source description, the original data could be of any modality. There is no way to control a train / test split without task details.

### [analcatdata\_supreme](https://pages.stern.nyu.edu/%C2%A0jsimonof/AnalCatData)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 4052     #Features: 7     Year: 2003

Comments: The analcatdata\_supreme dataset first appeared in the 2003 book ”Analysing Categorical Data” by Jeffrey S. Simonoff. This dataset contains a collection of decisions made by the Supreme Court of the United Stated from 1953 to 1988. The information used is very shallow, and the data was introduced for domain specific analysis, not to compare performance on random splits of the dataset

### [artificial-characters](https://ieeexplore.ieee.org/document/327470)

Tags: Leak, Synthetic or Untraceable, Raw

#Samples: 10218     #Features: 8     Year: 1993

Comments: This database has been artificially generated. It describes the structure of the capital letters A, C, D, E, F, G, H, L, P, R, indicated by a number 1-10, in that order (A=1,C=2,…). Each letter’s structure is described by a set of segments (lines) which resemble the way an automatic program would segment an image. The dataset consists of 600 such descriptions per letter.

Originally, each ’instance’ (letter) was stored in a separate file, each consisting of between 1 and 7 segments, numbered 0,1,2,3,… Here they are merged. That means that the first 5 instances describe the first 5 segments of the first segmentation of the first letter (A). Also, the training set (100 examples) and test set (the rest) are merged. The next 7 instances describe another segmentation (also of the letter A) and so on.

Not a tabular data task (synthetic letter classification). When used as a tabular dataset, leak could easily be exploited through the ”V7: diagonal, this is the length of the diagonal of the smallest rectangle which includes the picture of the character. The value of this attribute is the same in each object.”

### [audiology](https://archive.ics.uci.edu/dataset/8/audiology+standardized)

Tags: Tabular, Timesplit Needed

#Samples: 226     #Features: 70     Year: 1987

Comments: The audiology dataset has been provided by Professor Jergen at Baylor College of Medicine in 1987, and contains information describing the hearing ability of different patients

### [balance-scale](http://archive.ics.uci.edu/dataset/12/balance+scale)

Tags: Synthetic or Untraceable

#Samples: 625     #Features: 5     Year: 1994

Comments: This data set was generated to model psychological experimental results. Each example is classified as having the balance scale tip to the right, tip to the left, or be balanced. The attributes are the left weight, the left distance, the right weight, and the right distance. The correct way to find the class is the greater of (left-distance \* left-weight) and (right-distance \* right-weight). If they are equal, it is balanced. This is not a real world problem. Just the data from psychological study, easily solvable with one equation

### [bank-marketing](https://core.ac.uk/download/pdf/55616194.pdf)

Tags: Tabular, Timesplit Needed

#Samples: 10578     #Features: 7     Year: 2010

Comments: Dataset describes 17 marketing campaigns by a bank from 2008 to 2010. A set of features is not very rich, but reasonable (ideally there would be more user features and statistics).

### [cnae-9](https://www.openml.org/search?type=data&status=active&id=1468)

Tags: HomE

#Samples: 1080     #Features: 857     Year: 2009

Comments: This dataset only offers the frequencies of 800 words as features, the data is purely from the NLP domain

### [colic](https://www.openml.org/search?type=data&status=active&id=25)

Tags: Tabular, Timesplit Needed

#Samples: 368     #Features: 27     Year: 1989

Comments: The dataset of horses symptoms and whether or not they required surgery.

### [compass](https://www.kaggle.com/datasets/danofer/compass?select=cox-violent-parsed.csv)

Tags: Leak, Tabular, Timesplit Needed

#Samples: 16644     #Features: 17     Year: 2017

Comments: This dataset’s task is to determine whether a person will be arrested again after their release based on simple statistical features. The dataset first appeared in the paper ”It’s COMPASlicated: The Messy Relationship between RAI Datasets and Algorithmic Fairness Benchmarks” by Bao et al. Seemingly there are a lot of duplicates in the data, which leads to leakage when the random split is applied. Retrieval methods such as TabR achieve large performance gains.

### [covertype](https://dl.acm.org/doi/10.5555/928509)

Tags: Tabular, Timesplit Needed

#Samples: 423680     #Features: 54     Year: 1998

Comments: This dataset comes from 1998 study comparing different methods for predicting forest cover types from cartographic variables. No time features are included. Not representative of a real-world task: predicting forest cover-type solely from geological and cartographic features comes up less frequently, than directly processing GNSS data

### [cpu\_act](https://openml.org/d/197)

Tags: Tabular, Timesplit Needed

#Samples: 8192     #Features: 21     Year: 1999

Comments: This data represents logs from a server computer. The task is to predict the portion of time that cpu runs in user mode.

### [credit](https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv)

Tags: Tabular, Timesplit Needed

#Samples: 16714     #Features: 10     Year: 2011

Comments: Dataset from the kaggle competition hosted by ”Credit Fusion”. Corresponds to a real-world prediction problem. Not possible to create an out-of-time evaluation set. Relatively (relative to the modern dataset, e.g. https://www.kaggle.com/competitions/amex-default-prediction/overview) few features available.

### [credit-approval](https://archive.ics.uci.edu/dataset/27/credit+approval)

Tags: Tabular, Timesplit Needed

#Samples: 690     #Features: 16     Year: 1987

Comments: Same as Australian (but without preprocessing)

### [credit-g](https://www.openml.org/search?type=data&status=active&id=44096)

Tags: Tabular, Timesplit Needed

#Samples: 10000     #Features: 21     Year: 1994

Comments: This dataset includes a number of simple features useful for determining whether the bank can expect a return on a credit. The nature of labels is not explained, time feature is not used

### [diamonds](https://www.diamondse.info/diamond-prices.asp)

Tags: Tabular, Timesplit Needed

#Samples: 53940     #Features: 9     Year: 2015

Comments: The exact source of the data is unclear. The task is to predict the price of a diamond by its characteristics. Diamond prices fluctuate in time, however no timestamp information is available.

### [electricity](https://openml.org/d/151)

Tags: Leak, Tabular, Timesplit Needed, Timesplit Possible

#Samples: 38474     #Features: 7     Year: 1998

Comments: Data comes from the Australian New South Wales Electricity Market. The task is to predict whether electricity prices will go up or down. When a random split is used, there is a leak in the data, and retrieval methods such as TabR can achieve near 100% accuracy.

### [elevators](https://openml.org/d/216)

Tags: Raw

#Samples: 16599     #Features: 16     Year: 2014

Comments: This data set addresses a control problem, namely flying an F16 aircraft. The attributes describe the status of the aeroplane, while the goal is to predict the control action on the ailerons of the aircraft. According to the descriptions available, the dataset first appears in a collection of regression datasets by Luis Torgo and Rui Camacho made in 2014, but the original website with the description (ncc.up.pt/ ltorgo/Regression/DataSets.html) seems to no longer respond. The task of controlling a vehicle through machine learning has gained a large interest in recent years, but it is not done through tabular machine learning, instead often utilizing RL and using a wider range of sensors than those used in the non-self-driving version of the vehicle.

### [eye\_movements](https://openml.org/d/1044)

Tags: Leak, Tabular

#Samples: 7608     #Features: 20     Year: 2005

Comments: Time-series, Grouped data. This is a grouped dataset, some models are able to find a leak and predict based on an assignment number perfectly (MLP-PLR for example).

### [fifa](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)

Tags: Tabular, Timesplit Needed

#Samples: 18063     #Features: 5     Year: 2021

Comments: This dataset contains information about FIFA soccer players in 2021, and the target variable is their wages. The provided features include age, weight, height, and information about time spent in the player’s club, as well as the price in release clause. This dataset does not correspond to any real-world task, and the provided features are very shallow, as they luck any information about a player’s performance in previous games

### [guillermo](http://automl.chalearn.org)

Tags: HetE

#Samples: 20000     #Features: 4297     Year: 2018

Comments: The data was provided by AutoML challenge, and the dataset was created from objects from another domain, such as text, audio, or video, compressed into tabular form.

### [heart-h](https://openml.org/search?type=data&status=active&id=51)

Tags: Tabular, Timesplit Needed

#Samples: 294     #Features: 14     Year: 1988

Comments: This dataset was originally created by Andras Janosi et al. in 1988. A very small dataset including features describing person’s questionnaire responses as well as some compressed test results. Statistics are too shallow to adequately solve the task at hand.

### [house\_sales](https://openml.org/d/42731)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 21613     #Features: 15     Year: 2015

Comments: This dataset was created based on public records of house sales from May 2014 to May 2015. While this dataset has several improvements when compared to the California housing dataset, such as not averaging prices in a block, as well as availability of date of sale, the features are still shallow when compared to the housing dataset presented in this paper.

### [houses](https://openml.org/d/537)

Tags: Tabular, Timesplit Needed

#Samples: 20640     #Features: 8     Year: 1990

Comments: Data source bias, repeated dataset (unknown source in the description, but this is literally california\_housing with a different name and two features slightly altered)

### [isolet](https://openml.org/d/300)

Tags: HomE

#Samples: 7797     #Features: 613     Year: 1994

Comments: The dataset describes features extracted from audio recordings of the name of each letter of the English alphabet. The task is to classify the phoneme. This task would be better solved by raw audio processing.

### [jasmine](http://automl.chalearn.or)

Tags: Synthetic or Untraceable, HetE

#Samples: 2984     #Features: 145     Year: 2018

Comments: The data was provided by AutoML challenge, and the dataset was created from objects from another domain, such as text, audio, or video, compressed into tabular form.

### [jungle-chess](https://www.openml.org/search?type=data&status=active&id=41003)

Tags: Synthetic or Untraceable, Raw

#Samples: 44819     #Features: 7     Year: 2014

Comments: Game simulation, not a real ML task.

### [kc1](https://openml.org/search?type=data&status=active&id=1067)

Tags: HetE

#Samples: 2109     #Features: 22     Year: 2004

Comments: This dataset was created by Mike Chapman at NASA, and it contains features associated with the software quality. The task is to predict whether the code has any defects. Nowadays, the task of code quality analysis is solved mainly using NLP methods and is not tabular.

### [kdd\_ipums\_la\_97-small](https://kdd.ics.uci.edu/databases/ipums/ipums.html)

Tags: Tabular, Timesplit Needed

#Samples: 5188     #Features: 20     Year: 1997

Comments: The data is a subsample of census responses from the Los Angeles area for years 1970, 1980 and 1990. Unknown target variable (some categorical column from census binarized). Not a real-world task, based on census data.

### [lymph](https://www.openml.org/search?type=data&status=active&id=10)

Tags: Tabular, Timesplit Needed

#Samples: 148     #Features: 19     Year: 1988

Comments: This dataset was collected in November 1988 for University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia by Bojan Cestnik. It includes results of lymph test. The task is to classify lymph in one of four categories. Unfortunately, the dataset contains only 2 samples with normal lymph, making it hard for the dataset to be used for training a real-world model categorizing lymph.

### [medical\_charges](https://openml.org/d/42720)

Tags: Tabular, Timesplit Needed

#Samples: 163065     #Features: 3     Year: 2019

Comments: Public medicare data from 2019. According to openml analysis, only one of the features is important for prediction.

### [mfeat-fourier](https://archive.ics.uci.edu/dataset/72/multiple+features)

Tags: HomE

#Samples: 2000     #Features: 77     Year: 1998

Comments: One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps.

### [mfeat-zernike](https://archive.ics.uci.edu/dataset/72/multiple+features)

Tags: HomE

#Samples: 2000     #Features: 48     Year: 1998

Comments: One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps.

### [monks-problems-2](https://www.openml.org/search?type=data&status=active&id=334)

Tags: Synthetic or Untraceable

#Samples: 601     #Features: 7     Year: 1992

Comments: Simple toy synthetic, the task of determining whether there are exactly two ones among the 6 binary variables.

### [nomao](https://www.researchgate.net/publication/236168126_Design_and_Analysis_of_the_Nomao_challenge_Active_Learning_in_the_Real-World)

Tags: Tabular

#Samples: 34465     #Features: 119     Year: 2013

Comments: Active learning dataset, the task is determining whether two geo-location points are the same. Hand-labeled by an expert of Nomao.

### [nyc-taxi-green-dec-2016](https://openml.org/d/42729)

Tags: Tabular, Timesplit Needed

#Samples: 581835     #Features: 9     Year: 2016

Comments: The data was provided by the New York City Taxi and Limousine Commission, and the task is to predict tip amount based on simple features describing the trip.

### [particulate-matter-ukair-2017](https://joss.theoj.org/papers/10.21105/joss.00051)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 394299     #Features: 6     Year: 2017

Comments: Hourly particulate matter air pollution data of Great Britain for the year 2017. Time features available, prior work uses random split. There are only 6 features, describing time and location. This is a time-series forecasting problem (2 features from the original dataset missing). This is more likely a time-series problem, as there are not many heterogeneous features related to the task, only time-based features

### [phoneme](https://sci2s.ugr.es/keel/dataset.php?cod=105#sub2)

Tags: HomE

#Samples: 5404     #Features: 6     Year: 1993

Comments: The dataset describes a collection of phonemes and presents a task of classifying between nasal and oral sounds.
The phonemes are transcribed as follows: sh as in she, dcl as in dark, iy as the vowel in she, aa as the vowel in dark, and ao as the first vowel in water., DL in audio outperforms shallow methods, when applied to raw data. Here we only have 5 features extracted from the raw data (its audio)

### [poker-hand](https://archive.ics.uci.edu/dataset/158/poker+hand)

Tags: Synthetic or Untraceable, Tabular

#Samples: 1025009     #Features: 9     Year: 2007

Comments: A task of classifying a poker hand based on it’s content. One line non-ML solution exists, does not correspond to a real-world ML problem.

### [pol](https://openml.org/d/722)

Tags: Tabular, Timesplit Needed

#Samples: 10082     #Features: 26     Year: 1995

Comments: The data describes a telecommunication problem, no further information is available.

### [profb](https://www.openml.org/search?type=data&status=active&id=470)

Tags: Tabular, Timesplit Needed

#Samples: 672     #Features: 10     Year: 1992

Comments: Dataset describing professional football games. The task is to predict whether the favoured team was playing home.

### [qsar-biodeg](https://www.openml.org/search?type=data&status=active&id=1494)

Tags: HetE

#Samples: 155     #Features: 42     Year: 2013

Comments: The QSAR biodegradation dataset was built by the Milano Chemometrics and QSAR Research Group. Nowadays, a different approach based on graph neural networks is taken towards the task of predicting the characteristics of molecules, which is why this is not really a realistic use-case for tabular DL

### [rl](https://openml.org/d/41160)

Tags: Synthetic or Untraceable, Tabular

#Samples: 4970     #Features: 12     Year: 2018

Comments: Unknown real-life problem. Small, not many features, No canonical split. Retrieval methods such as TabR achieve large performance gains, which could signal that there is leakage in the data.

### [road-safety](https://openml.org/d/42803)

Tags: Tabular, Timesplit Needed, Timesplit Possible

#Samples: 111762     #Features: 32     Year: 2015

Comments: The data describes road accidents in Great Britain from 1979 to 2015. The task is to predict sex of a driver based on information about an accident. Retrieval methods such as TabR achieve large performance gains, which could signal that there is leakage in the data.

### [socmob](https://www.openml.org/search?type=data&status=active&id=44987)

Tags: Tabular, Timesplit Needed

#Samples: 1156     #Features: 6     Year: 1973

Comments: An instance represents the number of sons that have a certain job A given the father has the job B (additionally conditioned on race and family structure). Just statistic data, not a real task

### [splice](https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=splice&id=46)

Tags: Raw

#Samples: 3190     #Features: 61     Year: 1992

Comments: The task is to classify parts of genom as splice regions. The features are just a subsequence of DNA, more of an NLP task

### [sulfur](https://openml.org/d/23515)

Tags: Leak, Tabular

#Samples: 10081     #Features: 6     Year: 2007

Comments: Leakage. In this dataset, there originally were 2 closely related target variables: H2S concentration and SO2 concentration. However, the version used in the aforementioned tabular benchmarks contains one of these target variables as a feature. According to the observed feature importance, the new feature is much more informative about the target variable than any of the old ones: the original features only describe the outputs of the physical sensors, while the new one already uses the knowledge about the chemical makeup of the gas. Due to the described problems, which stem from the accidental error in the data preparation, the current version of this dataset does not seem close to the intent of the original dataset authors.

### [superconduct](https://www.frontiersin.org/articles/10.3389/fmats.2021.714752/full)

Tags: Tabular

#Samples: 21263     #Features: 79     Year: 2021

Comments: This dataset presents information about superconductors, with a task of predicting critical temperature.

### [vehicle](https://www.openml.org/search?type=data&status=active&id=54)

Tags: HetE

#Samples: 846     #Features: 19     Year: 1987

Comments: This dataset was created from the vehicle silhouettes in 1987, the task is to classify a car class by its silhouette.

### [visualizing\_soil](https://openml.org/d/688)

Tags: Leak, Tabular

#Samples: 8641     #Features: 4     Year: 1993

Comments: Leakage. This dataset describes a series of measurements of soil resistivity taken on a grid. The original intended target variable was the resistivity of the soil, however, it wasn’t the first variable, and the technical variable #1 became the target variable in the later versions of this dataset on OpenML and in the tabular benchmarks. This makes the task absurd and trivial, as a simple if between two linear transforms of two different other features in the dataset performs on par with the best algorithm mentioned in the TabR paper, beating 4 others.

### [wine](https://archive.ics.uci.edu/ml/datasets/wine+quality)

Tags: Tabular

#Samples: 2554     #Features: 11     Year: 2009

Comments: This dataset was published by Cortez et al. in 2009, and it contains the chemical properties of different wines. The task is to predict the quality of wine.

### [wine\_quality](https://openml.org/d/287)

Tags: Tabular

#Samples: 6497     #Features: 11     Year: 2009

Comments: This dataset was published by Cortez et al. in 2009, and it contains chemical properties of different wines. The task is to predict the quality of wine.

### [yprop\_4\_1](https://pubmed.ncbi.nlm.nih.gov/14502479/)

Tags: HomE

#Samples: 8885     #Features: 62     Year: 2003

Comments: This dataset describes a series of chemical formulas, with a task of predicting one attribute of a molecule based on many others. The task would be better solved by graph DL methods.

[◄](/html/2406.19379)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2406.19380)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2406.19380)
[View original  
on arXiv](https://arxiv.org/abs/2406.19380)[►](/html/2406.19381)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Jul 5 19:07:44 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
