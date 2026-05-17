---
arxiv: '2306.09468'
authors:
- Xiaotian Han 1 Jianfeng Chi 2 Yu Chen 2 Qifan Wang 2 Han Zhao 3 Na Zou 1 Xia Hu
  4 1 Texas A&M University 2 Meta AI 3 UIUC 4 Rice University {han, nzou1}@tamu.edu
  {jianfengchi, hugochen, wqfcr}@meta.com hanzhao@illinois.edu xia.hu@rice.edu
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods'
url: https://arxiv.org/abs/2306.09468
year: 2023
---

# FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods

Xiaotian Han1  Jianfeng Chi2  Yu Chen2  Qifan Wang2  Han Zhao3  Na Zou1  Xia Hu4
  
1Texas A&M University  2Meta AI  3UIUC  4Rice University
  
{han, nzou1}@tamu.edu  {jianfengchi, hugochen, wqfcr}@meta.com
  
hanzhao@illinois.edu  xia.hu@rice.edu
This work was done while the first author was an intern at Meta.

###### Abstract

This paper introduces the Fair Fairness Benchmark (FFB), a benchmarking framework for in-processing group fairness methods. Ensuring fairness in machine learning is critical for ethical and legal compliance. However, there exist challenges in comparing and developing of fairness methods due to inconsistencies in experimental settings, lack of accessible algorithmic implementations, and limited extensibility of current fairness packages and tools. To address these issues, we introduce an open-source, standardized benchmark for evaluating in-processing group fairness methods and provide a comprehensive analysis of state-of-the-art methods to ensure different notions of group fairness. This work offers the following key contributions: the provision of flexible, extensible, minimalistic, and research-oriented open-source code; the establishment of unified fairness method benchmarking pipelines; and extensive benchmarking, which yields key insights from 𝟒𝟓,𝟎𝟕𝟗

45079\mathbf{45,079} experiments. We believe our work will significantly facilitate the growth and development of the fairness research community. The benchmark, including code and running logs, is available at <https://github.com/ahxt/fair_fairness_benchmark>.

### 1 Introduction

Machine learning models trained on biased data have been found to perpetuate and even exacerbate the bias against historically underrepresented and disadvantaged demographic groups when deployed [[39](#bib.bib39), [43](#bib.bib43), [9](#bib.bib9), [48](#bib.bib48)]. As a result, concerns about fairness have gained significant attention, especially as applications of these models expand to high-stakes domains such as criminal justice, hiring process, and credit scoring. To mitigate such algorithmic bias, a variety of fairness criteria and algorithms have been proposed, which impose statistical constraints on the model to ensure equitable treatment under the respective fairness notions [[25](#bib.bib25), [10](#bib.bib10), [45](#bib.bib45)]. However, a fair and objective comparison between the proposed and existing algorithms to enforce fairness can be difficult due to the following reasons:

* •

  Hard to compare the performance of two objectives: utility111We use utility to represent the performance of the downstream task. and fairness. Often, there is a trade-off between these two objectives [[38](#bib.bib38), [55](#bib.bib55), [49](#bib.bib49)]. Besides, the instability of the fairness performance of those methods during the training process can complicate the pursuit of an optimal balance [[49](#bib.bib49)].
* •

  Inconsistent experimental settings[[37](#bib.bib37), [35](#bib.bib35)]: Fairness methods can also be hindered by variations in dataset preprocessing and the use of different backbones. These discrepancies can lead to inconsistent ([Table 1](#S2.T1 "In 2 Why is this Benchmark Needed? ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods")) and unfair comparison, further complicating the comparison of methods.
* •

  Stable and customizable implementation of commonly used fairness methods are not accessible. The fairness methods are often implemented in different programming languages and frameworks, complicating the reproduction and the comparative analysis of various fairness approaches.
* •

  Current fairness packages [[4](#bib.bib4), [6](#bib.bib6)] and tools often suffer from a lack of extensibility. This can make it difficult for researchers and practitioners to build upon existing methods and develop new ones.

This work aims to facilitate the growth and development of the fairness research community by addressing existing challenges, promoting more accessible and reproducible methods for fairness implementation, and establishing efficient benchmarking techniques. To achieve these goals, we develop a standardized benchmark for evaluating and comparing in-process group fairness methods, which we make open-source. We also conduct comprehensive experiments and analysis of fairness methods. The major contributions of our benchmark are summarized as follows:

* •

  Extensible, Minimalistic, and Research-oriented Open-Source Code: We offer open-source implementation for all preprocessing, methods, metrics, and training results, thus facilitating other researchers to utilize and build upon this work. The Fair Fairness Benchmark is publicly available, making it easy for researchers and practitioners to use and contribute to.
* •

  Unified Fairness Method Benchmarking Pipelines: Our benchmark includes a unified fairness method development and evaluation pipeline, with three key components: First, we provide a thorough statistical and experimental analysis of widely-used fairness datasets and identity some widely-used datasets unsuitable for studying fairness issues; second, we standardize preprocessing for these datasets, ensuring consistent, comparable evaluations; lastly, we present a range of bias mitigation algorithms and comprehensive evaluation metrics for group fairness.
* •

  Comprehensive Benchmarking and Detailed Obsevations: We conduct comprehensive experiments on 141414 datasets (each with two sensitive attributes), 666 utility metrics and 999 fairness metrics. We run a total of 45,079
  4507945,079 experiments. Our experiments yield the following key insights:

  Takeaways

  ➀ Not all widely used fairness datasets stably exhibit fairness issues. (Important for evaluation)
  ➁ Current fairness methods clearly exhibit utility-fairness trade-offs. (Confirm the trade-offs)
    
  ➂ The HSIC achieves the best utility-fairness trade-off overall. (Not a popular baseline before)
    
  ➃ Adversarial debiasing methods exhibit instability. (Fair constraints maybe more promising)
    
  ➄ Utility-fairness trade-offs are generally controllable. (Despite adversarial debiasing method)
    
  ➅ Utility training curves are stable, while fairness curves are unstable. (Debiasing is unstable)
    
  ➆ Stopping training while learning rate is lower enough is effective. (Practical strategy)
    
  ➇ Architecture does not significantly influence fairness performance. (Bias mainly in dataset)

Scope of this Work. We focus on the problem of in-processing group fairness, which is defined as discrimination against demographic groups. We consider the fairness methods in the context of binary classification and binary sensitive attributes.

Notation. The dataset is denoted as {(𝐱i,si,yi)i=1N}superscriptsubscriptsubscript𝐱𝑖subscript𝑠𝑖subscript𝑦𝑖𝑖1𝑁\{(\mathbf{x}\_{i},s\_{i},y\_{i})\_{i=1}^{N}\}, where N𝑁N is the number of samples. For the sample i𝑖i in the dataset, 𝐱i∈ℝdsubscript𝐱𝑖superscriptℝ𝑑\mathbf{x}\_{i}\in\mathbb{R}^{d} is non-sensitive attributes, si∈{0,1}subscript𝑠𝑖01s\_{i}\in\{0,1\} represents the binary sensitive attribute, and yi∈{0,1}subscript𝑦𝑖01y\_{i}\in\{0,1\} is the label of the downstream task. We use y^∈{0,1}^𝑦01\hat{y}\in\{0,1\} to denote the predicted label of the downstream task, which is obtained by thresholding the output of a machine learning model f​(𝐱):ℝd→[0,1]:𝑓𝐱→superscriptℝ𝑑01f(\mathbf{x}):\mathbb{R}^{d}\rightarrow[0,1] with trainable parameter θ𝜃\theta. Accordingly, we use X𝑋X, Y𝑌Y, S𝑆S and Y^^𝑌\hat{Y} to denote the random variables.

### 2 Why is this Benchmark Needed?

Ensuring fairness in algorithmic predictions is crucial in the field of machine learning. However, fairly comparing the current fairness method is challenging due to inconsistencies in data pre-processing and a lack of flexibility in existing fairness packages. This section aims to analyze the critical issue of current fairness methods and discuss the urgent need for a benchmark to address these challenges.

Current fairness packages lack flexibility for researchers. AIF360 [[4](#bib.bib4)] and FairLearn [[6](#bib.bib6)] are two well-known fairness packages that have successfully mitigated bias in machine learning algorithms. As popular open-source Python packages, they provide practitioners with toolkits for detecting and mitigating bias in their models and evaluating model fairness. AIF360 is a comprehensive fairness toolkit offering a variety of algorithms and metrics for addressing bias in machine learning models. FairLearn also provides multiple algorithms and fairness metrics for assessing model performance. While both packages are highly recognized within the fair machine learning community, they may not give researchers the desired flexibility for research purposes. We provide a more in-depth comparison between AIF360, FairLearn, and FFB in [Section 6](#S6 "6 Related Work ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods").

Table 1: The reported accuracy of tabular data varies in different papers.

|  |  |  |
| --- | --- | --- |
| Paper | Adult | German |
| Madras et al. [[37](#bib.bib37)] | ≈0.85absent0.85\approx 0.85 | — |
| Zemel et al. [[51](#bib.bib51)] | ≈0.70absent0.70\approx 0.70 | ≈0.69absent0.69\approx 0.69 |
| Edwards and Storkey [[18](#bib.bib18)] | ≈0.83absent0.83\approx 0.83 | — |
| Feldman et al. [[19](#bib.bib19)] | ≈0.68absent0.68\approx 0.68 | ≈0.69absent0.69\approx 0.69 |
| Louizos et al. [[35](#bib.bib35)] | ≈0.82absent0.82\approx 0.82 | ≈0.78absent0.78\approx 0.78 |
| ΔmaxsubscriptΔmax\Delta\_{\text{max}} | 0.170.170.17 | 0.090.090.09 |
| ΔΔ\DeltaPercentage | 20%percent2020\% | 13%percent1313\% |

Inconsistent experimental setting leads to unfair comparison. Prior research has often experienced inconsistencies in data pre-processing and train test split, which has led to divergent performance results that hinder comparison and reproducibility. Minor variations in data preparation and dataset split can significantly impact the performance of machine learning algorithms. Due to these issues, the reported accuracy of tabular data (Adult,German) varies in different papers [[37](#bib.bib37), [51](#bib.bib51), [18](#bib.bib18), [19](#bib.bib19), [35](#bib.bib35)] shown in [Table 1](#S2.T1 "In 2 Why is this Benchmark Needed? ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). To tackle these issues, we propose a standardized and consistent data pre-processing and split approach, including data normalization, outlier removal, and the implementation of a uniform train-test split ratio.

Sufficient and in-depth analysis of fairness methods is urgently needed. The current fairness method lacks a comprehensive comparison, such as the training curves and stability, the influence of the utility-fairness trade-off parameters, and how to select the best utility-fairness trade-offs during the training process. Our benchmark addresses these shortcomings by offering more in-depth and multifaceted analysis. To present a more thorough understanding of fairness methods, we investigate the training stability, model performance under various fairness constraints, and the selection of best-performing models.

### 3 FFB: Fair Fairness Benchmark

To overcome the above limitations of the previous methods, we introduce the Fair Fairness Benchmark (FFB), a extensible, minimalistic, and research-oriented fairness benchmark package. Compare to other fairness packages, FFB codebase has the following main characteristics:

* •

  Extensible: We provide the source code for fairness methods implementation, allowing researchers to modify, extend, and tailor these methods to suit their specific requirements and implement new ideas upon our code. This fosters a more customizable approach to developing fairness methods.
* •

  Minimalistic: We focus on delivering core fairness methods and allowing users to understand the fundamental techniques comprehensively without being overwhelmed by unnecessary complexity. This approach ensures that users can easily implement and integrate our methods into their existing workflows while maintaining a solid grasp of the underlying concepts.
* •

  Research-oriented: We include benchmark datasets and evaluation metrics that facilitate assessing fairness methods. This simplifies the research process, allowing researchers to quickly compare and analyze the effectiveness of different methods in various scenarios.

#### 3.1 Group Fairness Metrics

To provide a comprehensive comparison for bias mitigating methods, we consider multiple fairness metrics, including demographic parity, p𝑝p-rule, equality of opportunity, equalized odds, the area between CDF curves, etc. [Table 2](#S3.T2 "In 3.1 Group Fairness Metrics ‣ 3 FFB: Fair Fairness Benchmark ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") presents the fairness metrics used in [Section 4](#S4 "4 Bias Examination for Widely Used Fairness Benchmark Datasets ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") and [Section 5](#S5 "5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods").

Table 2: Fairness definitions and metrics used in the experiments. [Appendix A](#A1 "Appendix A Details of the Group Fairness ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") lists a more comprehensive list of fairness metrics and their details. Their simple code implementations are at [this url](https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/metrics.py).

| Abbreviation | Name | Formal Definition |
| --- | --- | --- |
| dp [[17](#bib.bib17)] | Demographic/Statistical Parity | P​(Y^∣S=0)=P​(Y^∣S=1)𝑃conditional^𝑌𝑆0𝑃conditional^𝑌𝑆1P(\hat{Y}\mid S=0)=P(\hat{Y}\mid S=1) |
| prule [[50](#bib.bib50)] | p𝑝p-Rule | P(Y^=1∣S=1)/P(Y^=1∣S=0)|≤p/100P(\hat{Y}=1\mid S=1)/P(\hat{Y}=1\mid S=0)|\leq p/100 |
| ppv [[11](#bib.bib11)] | Predictive Parity Value Parity | P​(Y=1∣Y^,S=0)=P​(Y=1∣Y^,S=1)𝑃𝑌conditional1  ^𝑌𝑆0𝑃𝑌conditional1  ^𝑌𝑆1P(Y=1\mid\hat{Y},S=0)=P(Y=1\mid\hat{Y},S=1) |
| bnegc [[29](#bib.bib29)] | Balance for Negative Class | 𝔼​[f​(X)∣Y=0,S=0]=𝔼​[f​(X)∣Y=0,S=1]𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌0𝑆0𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌0𝑆1\mathbb{E}[f(X)\mid Y=0,S=0]=\mathbb{E}[f(X)\mid Y=0,S=1] |
| bposc [[29](#bib.bib29)] | Balance for Positive Class | 𝔼​[f​(X)∣Y=1,S=0]=𝔼​[f​(X)∣Y=1,S=1]𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌1𝑆0𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌1𝑆1\mathbb{E}[f(X)\mid Y=1,S=0]=\mathbb{E}[f(X)\mid Y=1,S=1] |
| eopp [[23](#bib.bib23)] | Equality of Opportunity | P​(Y^∣S=0,Y=1)=P​(Y^∣S=1,Y=1)𝑃formulae-sequenceconditional^𝑌𝑆0𝑌1𝑃formulae-sequenceconditional^𝑌𝑆1𝑌1P(\hat{Y}\mid S=0,Y=1)=P(\hat{Y}\mid S=1,Y=1) |
| eodd [[23](#bib.bib23)] | Equalized Odds | P​(Y^∣S=0,Y=y)=P​(Y^∣S=1,Y=y),y∈{0,1}formulae-sequence𝑃formulae-sequenceconditional^𝑌𝑆0𝑌𝑦𝑃formulae-sequenceconditional^𝑌𝑆1𝑌𝑦𝑦01P(\hat{Y}\mid S=0,Y=y)=P(\hat{Y}\mid S=1,Y=y),y\in\{0,1\} |
| abcc [[22](#bib.bib22)] | Area Between CDF Curves | ∫01|F0​(x)−F1​(x)|​dxsuperscriptsubscript01subscript𝐹0𝑥subscript𝐹1𝑥differential-d𝑥\int\_{0}^{1}\left|F\_{0}(x)-F\_{1}(x)\right|\mathrm{d}x |

#### 3.2 Benchmarking Datasets

To provide a comprehensive comparison of fairness methods, we adopted multiple commonly-used fairness datasets [[32](#bib.bib32)] for our experiments, including tabular and image datasets. [Table 3](#S3.T3 "In 3.2 Benchmarking Datasets ‣ 3 FFB: Fair Fairness Benchmark ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") summarizes the datasets used. they are publicly available and can be downloaded from the corresponding websites. We also present the number of features and the ratio of the target label and sensitive attributes. For example, the target label ratio of KDDCensus is 1:14.76:114.761:14.76, which is an extremely unbalanced dataset.

Table 3: The summary of the benchmarking datasets. The #nFeat/#cFeat is the number of the numerical/categorical features and the #allFeat is the total number of the features after our preprocessing. The y0:y1:subscript𝑦0subscript𝑦1y\_{0}:y\_{1}/s0:s1:subscript𝑠0subscript𝑠1s\_{0}:s\_{1} is the ratio of two classes of the target label and the sensitive attributes. More details about datasets are presented in [Appendix B](#A2 "Appendix B Details of the Benchmarking Datasets ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). The dataset loading codes are at [this url](https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/dataset.py).

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | Task | SensAttr | #Instances | #nFeat | #cFeat | #allFeat | y0:y1:subscript𝑦0subscript𝑦1y\_{0}:y\_{1} | s0:s1:subscript𝑠0subscript𝑠1s\_{0}:s\_{1} (1st) | s0:s1:subscript𝑠0subscript𝑠1s\_{0}:s\_{1} (2nd) |
| Adult [[30](#bib.bib30)] | income | gender, race | 45,222  4522245,222 | 777 | 555 | 101101101 | 1:0.33:10.331:0.33 | 1:2.08:12.081:2.08 | 1:9.20:19.201:9.20 |
| German [[16](#bib.bib16)] | credit | gender, age | 1,000  10001,000 | 131313 | 666 | 585858 | 1:2.33:12.331:2.33 | 1:2.23:12.231:2.23 | 1:4.26:14.261:4.26 |
| KDDCensus [[16](#bib.bib16)] | income | gender, race | 292,550  292550292,550 | 323232 | 888 | 509509509 | 1:14.76:114.761:14.76 | 1:0.92:10.921:0.92 | 1:8.14:18.141:8.14 |
| COMPAS [[31](#bib.bib31)] | credit | age | 6,172  61726,172 | 400400400 | 555 | 405405405 | 1:0.83:10.831:0.83 | 1:4.25:14.251:4.25 | — |
| Bank [[16](#bib.bib16)] | credit | gender, race | 41,188  4118841,188 | 101010 | 999 | 626262 | 1:0.13:10.131:0.13 | 1:37.58:137.581:37.58 | 1:37.58:137.581:37.58 |
| ACS-I [[15](#bib.bib15)] | income | gender, race | 195,665  195665195,665 | 888 | 111 | 908908908 | 1:0.70:10.701:0.70 | 1:0.89:10.891:0.89 | 1:1.62:11.621:1.62 |
| ACS-E [[15](#bib.bib15)] | employment | gender, race | 378,817  378817378,817 | 151515 | 00 | 187187187 | 1:0.84:10.841:0.84 | 1:1.03:11.031:1.03 | 1:1.59:11.591:1.59 |
| ACS-P [[15](#bib.bib15)] | public | gender, race | 138,554  138554138,554 | 181818 | 00 | 169616961696 | 1:0.58:10.581:0.58 | 1:1.27:11.271:1.27 | 1:1.31:11.311:1.31 |
| ACS-M [[15](#bib.bib15)] | mobility | gender, race | 80,329  8032980,329 | 202020 | 00 | 267826782678 | 1:3.26:13.261:3.26 | 1:0.95:10.951:0.95 | 1:1.32:11.321:1.32 |
| ACS-T [[15](#bib.bib15)] | traveltime | gender, race | 172,508  172508172,508 | 151515 | 00 | 156715671567 | 1:0.94:10.941:0.94 | 1:0.89:10.891:0.89 | 1:1.61:11.611:1.61 |
| CelebA-A [[34](#bib.bib34)] | attractive | gender, age | 202,599  202599202,599 | — | — | 48×48484848\times 48 | 1:0.95:10.951:0.95 | 1:1.40:11.401:1.40 | 1:0.29:10.291:0.29 |
| CelebA-W [[34](#bib.bib34)] | wavy hair | gender, age | 202,599  202599202,599 | — | — | 48×48484848\times 48 | 1:2.13:12.131:2.13 | 1:1.40:11.401:1.40 | 1:0.29:10.291:0.29 |
| CelebA-S [[34](#bib.bib34)] | smiling | gender, age | 202,599  202599202,599 | — | — | 48×48484848\times 48 | 1:1.07:11.071:1.07 | 1:1.40:11.401:1.40 | 1:0.29:10.291:0.29 |
| UTKFace [[54](#bib.bib54)] | age | gender, race | 23,705  2370523,705 | — | — | 48×48484848\times 48 | 1:1.15:11.151:1.15 | 1:1.10:11.101:1.10 | 1:1.35:11.351:1.35 |

#### 3.3 Data Preprocessing

To ensure a fair comparison and maintain the reproducibility of the fairness approach, we adhere to a conventional data preprocessing strategy. We apply standard normalization for numerical features while employing one-hot encoding to process the categorical features. We also split the data into training and test sets with random seeds. We use the training set to train the model and the test set to evaluate the model’s performance. We use the same data preprocessing strategy for all the datasets.

#### 3.4 Benchmarking Fairness Methods

In this section, we introduce the benchmarking methodology employed in our experiments. The benchmarking methods can be classified into three categories: surrogate loss, independence constraints, and adversarial debiasing techniques. In this paper, we focus on in-processing methods for fairness primarily due to the following: 1) They intervene directly in the learning algorithm to ensure fairness, which provides a more nuanced and effective approach to mitigating bias;
2) The emergence of more in-processing techniques designed in deep neural networks calls for systematic comparison;
3) In-processing methods are susceptible to information leakage since they do not require sensitive attributes during inference.
In particular, we consider the following three types of in-processing methods.
Gap Regularization [[12](#bib.bib12)] simplifies optimization by offering a smooth approximation to real loss functions, which are often non-convex or difficult to optimize directly. This approach includes DiffDP, DiffEodd, and DiffEopp.
Independence introduces fairness constraint into the optimization process to minimize the impact of protected attributes on model predictions while maintaining performance. This approach includes PRemover [[28](#bib.bib28)] and HSIC [[33](#bib.bib33)].
Adversarial Learning222For all adversarial learning methods, we use gradient reversal layer [[20](#bib.bib20)] for better training stability. minimizes the utility loss while preventing the adversary from accurately predicting the protected attributes. This approach includes AdvDebias [[52](#bib.bib52), [36](#bib.bib36), [5](#bib.bib5), [18](#bib.bib18), [1](#bib.bib1)] and LAFTR [[37](#bib.bib37)]. The fairness methods are present as follows:

* •

  ERM is a standard machine learning method that minimizes the empirical risk of the training data. It is a common baseline for fairness methods.
* •

  DiffDP, DiffEopp, DiffEodd are the gap regularization methods for demographic parity, equalized opportunity, and equalized odds [[12](#bib.bib12)]. As these fairness definitions cannot be optimized directly, gap regularization is differentiable and can be optimized using gradient descent.
* •

  PRemover [[28](#bib.bib28)] (PrejudiceRemover) minimizes the mutual information between the prediction accuracy and the sensitive attributes.
* •

  HSIC [[21](#bib.bib21), [3](#bib.bib3), [33](#bib.bib33)] minimizes the Hilbert-Schmidt Independence Criterion between the prediction accuracy and the sensitive attributes.
* •

  AdvDebias [[52](#bib.bib52), [36](#bib.bib36), [5](#bib.bib5), [18](#bib.bib18), [1](#bib.bib1)] (adversarial debiasing) learns a classifier that maximizes the prediction ability and simultaneously minimizes an adversary to predict the sensitive attributes from the predictions.
* •

  LAFTR [[37](#bib.bib37)] is a fair representation learning method that aims to learn an intermediate representation that minimizes the classification loss, reconstruction error, and the adversary’s ability to predict the sensitive attributes from the representation.

### 4 Bias Examination for Widely Used Fairness Benchmark Datasets

Table 4: Bias examination for all datasets. We identify the biased dataset specified with sensitive attributes with the reported utility and fairness performance of ERM. Numbers (e.g., 1.351.351.35±plus-or-minus\pm1.171.171.17) mean that the bias is too small, indicating is not suitable for fairness accessment. The biased datasets are marked with ✔, while the unbiased datasets are marked with ✗. The ~~✔~~ indicates that the bias exists but with a large standard deviation. The results are based on 101010 trials with varying data splits and training seeds, to ensure reliable outcomes. The table is generated from a total of 𝟗𝟏𝟎910\mathbf{910} runs.

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | Utility | | | | Fairness | | | | |  |
| Dataset | SenAttr | acc | auc | ap | f1 | dp | abcc | prule | eodd | eopp | Bias? |
| Bank | Age | 91.1791.1791.17±plus-or-minus\pm0.570.570.57 | 94.0594.0594.05±plus-or-minus\pm0.190.190.19 | 62.4162.4162.41±plus-or-minus\pm1.831.831.83 | 54.3154.3154.31±plus-or-minus\pm11.1411.1411.14 | 10.8810.8810.88±plus-or-minus\pm4.274.274.27 | 10.6410.6410.64±plus-or-minus\pm1.631.631.63 | 44.4844.4844.48±plus-or-minus\pm5.955.955.95 | 10.7110.7110.71±plus-or-minus\pm5.685.685.68 | 6.166.166.16±plus-or-minus\pm4.904.904.90 | ~~✔~~ |
|  | Gender | 75.4275.4275.42±plus-or-minus\pm2.032.032.03 | 78.5578.5578.55±plus-or-minus\pm1.971.971.97 | 89.2189.2189.21±plus-or-minus\pm1.061.061.06 | 83.0283.0283.02±plus-or-minus\pm1.541.541.54 | 7.367.367.36±plus-or-minus\pm4.354.354.35 | 4.894.894.89±plus-or-minus\pm1.771.771.77 | 90.4790.4790.47±plus-or-minus\pm5.745.745.74 | 14.4514.4514.45±plus-or-minus\pm11.5511.5511.55 | 2.742.742.74±plus-or-minus\pm1.761.761.76 | ~~✔~~ |
| German | Age | 75.1975.1975.19±plus-or-minus\pm2.162.162.16 | 77.2177.2177.21±plus-or-minus\pm2.602.602.60 | 88.2888.2888.28±plus-or-minus\pm1.741.741.74 | 82.9082.9082.90±plus-or-minus\pm1.621.621.62 | 12.2012.2012.20±plus-or-minus\pm6.026.026.02 | 10.0110.0110.01±plus-or-minus\pm1.631.631.63 | 84.4484.4484.44±plus-or-minus\pm7.177.177.17 | 17.9717.9717.97±plus-or-minus\pm10.7110.7110.71 | 8.268.268.26±plus-or-minus\pm6.126.126.12 | ~~✔~~ |
| Adult | Gender | 85.3585.3585.35±plus-or-minus\pm0.340.340.34 | 91.0691.0691.06±plus-or-minus\pm0.340.340.34 | 78.5078.5078.50±plus-or-minus\pm0.720.720.72 | 66.7866.7866.78±plus-or-minus\pm0.750.750.75 | 16.6716.6716.67±plus-or-minus\pm0.690.690.69 | 18.3618.3618.36±plus-or-minus\pm0.710.710.71 | 32.5432.5432.54±plus-or-minus\pm2.622.622.62 | 14.1614.1614.16±plus-or-minus\pm3.123.123.12 | 7.937.937.93±plus-or-minus\pm2.882.882.88 | ✔ |
| Race | 85.2185.2185.21±plus-or-minus\pm0.270.270.27 | 91.1091.1091.10±plus-or-minus\pm0.160.160.16 | 78.6578.6578.65±plus-or-minus\pm0.350.350.35 | 66.8566.8566.85±plus-or-minus\pm0.460.460.46 | 12.2312.2312.23±plus-or-minus\pm0.720.720.72 | 12.5912.5912.59±plus-or-minus\pm0.600.600.60 | 41.5441.5441.54±plus-or-minus\pm2.682.682.68 | 13.1213.1213.12±plus-or-minus\pm2.652.652.65 | 8.818.818.81±plus-or-minus\pm2.932.932.93 | ✔ |
|  | Gender | 67.0767.0767.07±plus-or-minus\pm0.800.800.80 | 72.5672.5672.56±plus-or-minus\pm0.740.740.74 | 67.9967.9967.99±plus-or-minus\pm0.930.930.93 | 59.7759.7759.77±plus-or-minus\pm2.272.272.27 | 13.4313.4313.43±plus-or-minus\pm2.482.482.48 | 5.805.805.80±plus-or-minus\pm1.121.121.12 | 65.1265.1265.12±plus-or-minus\pm8.258.258.25 | 19.6719.6719.67±plus-or-minus\pm6.026.026.02 | 11.5411.5411.54±plus-or-minus\pm4.734.734.73 | ✔ |
| COMPAS | Race | 67.1367.1367.13±plus-or-minus\pm1.061.061.06 | 72.9872.9872.98±plus-or-minus\pm0.590.590.59 | 68.2468.2468.24±plus-or-minus\pm0.720.720.72 | 60.5860.5860.58±plus-or-minus\pm3.063.063.06 | 16.8316.8316.83±plus-or-minus\pm3.483.483.48 | 8.158.158.15±plus-or-minus\pm1.121.121.12 | 61.8361.8361.83±plus-or-minus\pm4.564.564.56 | 29.0329.0329.03±plus-or-minus\pm6.666.666.66 | 20.0520.0520.05±plus-or-minus\pm3.953.953.95 | ✔ |
| KDDCensus | Gender | 94.8894.8894.88±plus-or-minus\pm0.480.480.48 | 94.0394.0394.03±plus-or-minus\pm0.040.040.04 | 99.5599.5599.55±plus-or-minus\pm0.000.000.00 | 97.3297.3297.32±plus-or-minus\pm0.240.240.24 | 3.613.613.61±plus-or-minus\pm1.601.601.60 | 5.205.205.20±plus-or-minus\pm0.350.350.35 | 96.3596.3596.35±plus-or-minus\pm1.621.621.62 | 14.9714.9714.97±plus-or-minus\pm7.117.117.11 | 0.770.770.77±plus-or-minus\pm0.370.370.37 | ✔ |
| Race | 94.4994.4994.49±plus-or-minus\pm0.780.780.78 | 94.4094.4094.40±plus-or-minus\pm0.080.080.08 | 99.5799.5799.57±plus-or-minus\pm0.010.010.01 | 97.1397.1397.13±plus-or-minus\pm0.380.380.38 | 1.351.351.35±plus-or-minus\pm1.171.171.17 | 3.313.313.31±plus-or-minus\pm0.150.150.15 | 98.6498.6498.64±plus-or-minus\pm1.181.181.18 | 6.566.566.56±plus-or-minus\pm5.835.835.83 | 0.280.280.28±plus-or-minus\pm0.250.250.25 | ✗ |
|  | Gender | 82.3082.3082.30±plus-or-minus\pm0.120.120.12 | 90.2890.2890.28±plus-or-minus\pm0.090.090.09 | 86.0286.0286.02±plus-or-minus\pm0.150.150.15 | 77.9177.9177.91±plus-or-minus\pm0.190.190.19 | 9.109.109.10±plus-or-minus\pm0.310.310.31 | 8.278.278.27±plus-or-minus\pm0.240.240.24 | 79.0179.0179.01±plus-or-minus\pm0.650.650.65 | 3.383.383.38±plus-or-minus\pm0.610.610.61 | 1.751.751.75±plus-or-minus\pm0.410.410.41 | ✔ |
| ACS-I | Race | 82.4082.4082.40±plus-or-minus\pm0.090.090.09 | 90.4090.4090.40±plus-or-minus\pm0.090.090.09 | 86.1786.1786.17±plus-or-minus\pm0.140.140.14 | 78.1178.1178.11±plus-or-minus\pm0.110.110.11 | 9.819.819.81±plus-or-minus\pm0.390.390.39 | 7.717.717.71±plus-or-minus\pm0.290.290.29 | 77.2477.2477.24±plus-or-minus\pm0.820.820.82 | 9.729.729.72±plus-or-minus\pm0.730.730.73 | 6.216.216.21±plus-or-minus\pm0.480.480.48 | ✔ |
| ACS-E | Gender | 81.6381.6381.63±plus-or-minus\pm0.120.120.12 | 88.9588.9588.95±plus-or-minus\pm0.080.080.08 | 83.1283.1283.12±plus-or-minus\pm0.120.120.12 | 81.3181.3181.31±plus-or-minus\pm0.160.160.16 | 0.600.600.60±plus-or-minus\pm0.200.200.20 | 0.560.560.56±plus-or-minus\pm0.120.120.12 | 98.8798.8798.87±plus-or-minus\pm0.370.370.37 | 10.7710.7710.77±plus-or-minus\pm0.200.200.20 | 0.900.900.90±plus-or-minus\pm0.180.180.18 | ✗ |
| Race | 81.9981.9981.99±plus-or-minus\pm0.160.160.16 | 90.0090.0090.00±plus-or-minus\pm0.130.130.13 | 85.5885.5885.58±plus-or-minus\pm0.240.240.24 | 81.3881.3881.38±plus-or-minus\pm0.110.110.11 | 1.421.421.42±plus-or-minus\pm0.350.350.35 | 0.990.990.99±plus-or-minus\pm0.110.110.11 | 97.2997.2997.29±plus-or-minus\pm0.620.620.62 | 3.483.483.48±plus-or-minus\pm0.820.820.82 | 2.192.192.19±plus-or-minus\pm0.450.450.45 | ✗ |
|  | Gender | 71.9271.9271.92±plus-or-minus\pm0.180.180.18 | 75.2575.2575.25±plus-or-minus\pm0.160.160.16 | 67.2367.2367.23±plus-or-minus\pm0.220.220.22 | 52.9352.9352.93±plus-or-minus\pm0.710.710.71 | 2.092.092.09±plus-or-minus\pm0.640.640.64 | 2.352.352.35±plus-or-minus\pm0.160.160.16 | 91.2691.2691.26±plus-or-minus\pm2.522.522.52 | 2.302.302.30±plus-or-minus\pm1.321.321.32 | 1.521.521.52±plus-or-minus\pm1.001.001.00 | ✗ |
| ACS-P | Race | 71.7071.7071.70±plus-or-minus\pm0.220.220.22 | 75.0075.0075.00±plus-or-minus\pm0.310.310.31 | 67.0167.0167.01±plus-or-minus\pm0.280.280.28 | 52.0652.0652.06±plus-or-minus\pm0.540.540.54 | 0.480.480.48±plus-or-minus\pm0.320.320.32 | 1.981.981.98±plus-or-minus\pm0.200.200.20 | 97.8797.8797.87±plus-or-minus\pm1.361.361.36 | 4.634.634.63±plus-or-minus\pm0.380.380.38 | 4.034.034.03±plus-or-minus\pm0.720.720.72 | ✗ |
| ACS-M | Gender | 76.8176.8176.81±plus-or-minus\pm0.320.320.32 | 72.8572.8572.85±plus-or-minus\pm0.360.360.36 | 88.4088.4088.40±plus-or-minus\pm0.220.220.22 | 86.5486.5486.54±plus-or-minus\pm0.310.310.31 | 0.180.180.18±plus-or-minus\pm0.170.170.17 | 0.840.840.84±plus-or-minus\pm0.120.120.12 | 99.8099.8099.80±plus-or-minus\pm0.190.190.19 | 0.450.450.45±plus-or-minus\pm0.510.510.51 | 0.080.080.08±plus-or-minus\pm0.090.090.09 | ✗ |
| Race | 76.9876.9876.98±plus-or-minus\pm0.650.650.65 | 73.2373.2373.23±plus-or-minus\pm0.610.610.61 | 88.5388.5388.53±plus-or-minus\pm0.270.270.27 | 86.7086.7086.70±plus-or-minus\pm0.040.040.04 | 0.110.110.11±plus-or-minus\pm0.170.170.17 | 1.151.151.15±plus-or-minus\pm0.170.170.17 | 99.8899.8899.88±plus-or-minus\pm0.180.180.18 | 0.820.820.82±plus-or-minus\pm1.131.131.13 | 0.180.180.18±plus-or-minus\pm0.290.290.29 | ✗ |
|  | Gender | 66.3666.3666.36±plus-or-minus\pm0.220.220.22 | 73.5473.5473.54±plus-or-minus\pm0.180.180.18 | 71.5971.5971.59±plus-or-minus\pm0.140.140.14 | 66.5166.5166.51±plus-or-minus\pm0.310.310.31 | 8.608.608.60±plus-or-minus\pm0.450.450.45 | 5.025.025.02±plus-or-minus\pm0.280.280.28 | 84.6584.6584.65±plus-or-minus\pm0.740.740.74 | 12.9012.9012.90±plus-or-minus\pm0.820.820.82 | 5.725.725.72±plus-or-minus\pm0.440.440.44 | ✔ |
| ACS-T | Race | 66.4566.4566.45±plus-or-minus\pm0.200.200.20 | 73.6473.6473.64±plus-or-minus\pm0.170.170.17 | 71.6771.6771.67±plus-or-minus\pm0.190.190.19 | 66.2666.2666.26±plus-or-minus\pm0.470.470.47 | 9.629.629.62±plus-or-minus\pm0.670.670.67 | 6.076.076.07±plus-or-minus\pm0.220.220.22 | 83.0983.0983.09±plus-or-minus\pm0.920.920.92 | 15.1915.1915.19±plus-or-minus\pm1.241.241.24 | 6.506.506.50±plus-or-minus\pm0.990.990.99 | ✔ |
| CelebA-A | Gender | 78.1978.1978.19±plus-or-minus\pm0.440.440.44 | 86.6786.6786.67±plus-or-minus\pm0.530.530.53 | 86.6686.6686.66±plus-or-minus\pm0.640.640.64 | 79.1779.1779.17±plus-or-minus\pm0.480.480.48 | 52.3952.3952.39±plus-or-minus\pm1.271.271.27 | 37.6737.6737.67±plus-or-minus\pm0.980.980.98 | 30.4230.4230.42±plus-or-minus\pm1.231.231.23 | 70.8470.8470.84±plus-or-minus\pm3.153.153.15 | 35.5335.5335.53±plus-or-minus\pm1.821.821.82 | ✔ |
| Race | 78.1978.1978.19±plus-or-minus\pm0.440.440.44 | 86.6786.6786.67±plus-or-minus\pm0.530.530.53 | 86.6686.6686.66±plus-or-minus\pm0.640.640.64 | 79.1779.1779.17±plus-or-minus\pm0.470.470.47 | 41.9041.9041.90±plus-or-minus\pm1.031.031.03 | 31.1531.1531.15±plus-or-minus\pm1.171.171.17 | 33.4333.4333.43±plus-or-minus\pm1.711.711.71 | 37.4237.4237.42±plus-or-minus\pm2.262.262.26 | 18.8318.8318.83±plus-or-minus\pm1.941.941.94 | ✔ |
|  | Gender | 82.5082.5082.50±plus-or-minus\pm0.760.760.76 | 88.3888.3888.38±plus-or-minus\pm0.860.860.86 | 80.3880.3880.38±plus-or-minus\pm1.571.571.57 | 70.1470.1470.14±plus-or-minus\pm1.641.641.64 | 33.9233.9233.92±plus-or-minus\pm1.351.351.35 | 29.5229.5229.52±plus-or-minus\pm1.121.121.12 | 16.8916.8916.89±plus-or-minus\pm2.012.012.01 | 52.7152.7152.71±plus-or-minus\pm4.284.284.28 | 39.6239.6239.62±plus-or-minus\pm3.493.493.49 | ✔ |
| CelebA-W | Race | 82.5082.5082.50±plus-or-minus\pm0.760.760.76 | 88.3888.3888.38±plus-or-minus\pm0.860.860.86 | 80.3880.3880.38±plus-or-minus\pm1.571.571.57 | 70.1470.1470.14±plus-or-minus\pm1.641.641.64 | 10.2710.2710.27±plus-or-minus\pm0.710.710.71 | 10.6110.6110.61±plus-or-minus\pm0.470.470.47 | 64.5964.5964.59±plus-or-minus\pm2.352.352.35 | 10.6310.6310.63±plus-or-minus\pm2.182.182.18 | 6.486.486.48±plus-or-minus\pm1.941.941.94 | ✔ |
| CelebA-S | Gender | 89.9589.9589.95±plus-or-minus\pm3.403.403.40 | 96.5196.5196.51±plus-or-minus\pm1.841.841.84 | 96.6796.6796.67±plus-or-minus\pm1.801.801.80 | 89.0889.0889.08±plus-or-minus\pm6.796.796.79 | 14.0914.0914.09±plus-or-minus\pm1.081.081.08 | 13.0213.0213.02±plus-or-minus\pm1.461.461.46 | 72.7672.7672.76±plus-or-minus\pm3.443.443.44 | 6.996.996.99±plus-or-minus\pm1.161.161.16 | 6.516.516.51±plus-or-minus\pm1.061.061.06 | ✔ |
| Race | 89.9589.9589.95±plus-or-minus\pm3.403.403.40 | 96.5196.5196.51±plus-or-minus\pm1.841.841.84 | 96.6796.6796.67±plus-or-minus\pm1.801.801.80 | 89.0889.0889.08±plus-or-minus\pm6.796.796.79 | 5.915.915.91±plus-or-minus\pm0.750.750.75 | 5.595.595.59±plus-or-minus\pm0.700.700.70 | 88.2188.2188.21±plus-or-minus\pm2.502.502.50 | 6.466.466.46±plus-or-minus\pm1.061.061.06 | 0.820.820.82±plus-or-minus\pm0.800.800.80 | ✔ |
|  | Gender | 83.3483.3483.34±plus-or-minus\pm0.710.710.71 | 91.7891.7891.78±plus-or-minus\pm0.570.570.57 | 91.3691.3691.36±plus-or-minus\pm0.670.670.67 | 81.6681.6681.66±plus-or-minus\pm0.850.850.85 | 25.6825.6825.68±plus-or-minus\pm1.881.881.88 | 20.5720.5720.57±plus-or-minus\pm1.231.231.23 | 54.6154.6154.61±plus-or-minus\pm2.542.542.54 | 28.6628.6628.66±plus-or-minus\pm4.124.124.12 | 17.1617.1617.16±plus-or-minus\pm2.632.632.63 | ✔ |
| UTKFace | Race | 83.3483.3483.34±plus-or-minus\pm0.710.710.71 | 91.7891.7891.78±plus-or-minus\pm0.570.570.57 | 91.3691.3691.36±plus-or-minus\pm0.670.670.67 | 81.6681.6681.66±plus-or-minus\pm0.850.850.85 | 23.2523.2523.25±plus-or-minus\pm1.711.711.71 | 18.9918.9918.99±plus-or-minus\pm1.221.221.22 | 59.6759.6759.67±plus-or-minus\pm2.632.632.63 | 23.0723.0723.07±plus-or-minus\pm3.643.643.64 | 16.6816.6816.68±plus-or-minus\pm2.702.702.70 | ✔ |

The presence of bias in the current widely used dataset is not well examined and investigated as it should be, even though such bias can significantly influence the assessment of fairness methods. As such, our work endeavors to delve deeper into this issue by exploring the inherent bias in the widely used dataset. We aim to assess the suitability of these datasets for fairness evaluation critically.

➀ Not all widely used fairness datasets stably exhibit fairness issues. We systematically identify datasets that not only have demonstrated bias but are also prevalently used in fairness research. We found that in some cases, the bias in these datasets is either not consistently present or its manifestation varies significantly. This finding indicates that relying on these datasets for fairness analysis might not always provide stable or reliable results, suggesting the need for more rigorous dataset selection or bias evaluation methodologies in future fairness studies. The biased datasets are marked with ✔ while unbiased ones are with ✗. The ~~✔~~ indicates that the bias exists but with a large standard deviation.

### 5 Benchmarking Current Fairness Methods

In this section, we present comprehensive experiments to benchmark the performance of existing in-processing group fairness methods. We aim to provide a holistic overview of the group fairness methods, identifying both their strengths and areas for improvement.

!(/html/2306.09468/assets/x1.png)

(a) Tabular Data, across Adult, German, Bank, KDDCensus, ACS-I/E/P/M/T datasets.

!(/html/2306.09468/assets/x2.png)

(b) Image Data, across CelebA-A/W/S, UTKFace datasets with multiple targets.

!(/html/2306.09468/assets/x3.png)

(c) Tabular Data on eodd and eodd.

!(/html/2306.09468/assets/x4.png)

(d) Image Data on eodd and eodd.

Figure 1: The utility-fairness trade-offs of current fairness methods – DiffDP, PRemover, HSIC, LAFTR, and AdvDebias. To plot the fairness and utility performance in one figure, for each dataset, we normalize the utility (acc,auc) and fairness (abcc, dp) based on the performance of ERM, which is denoted as the point (1.0,1.0)1.01.0(1.0,1.0). The figures clearly show that utility-fairness exhibits trader-offs. These figures are generated from a total of 𝟐𝟕𝟓𝟔𝟖27568\mathbf{27568} runs. We present figures of the individual method on the individual dataset in [our repo](https://github.com/ahxt/fair_fairness_benchmark).

Experimental Setting. For tabular datasets, we use a two-layer Multi-layer Perceptron with 256256256 neurons each for all datasets. We use different bath sizes for different datasets based on the total number of instances of each dataset. For image datasets, we use various neural networks (such as ResNet-18 [[24](#bib.bib24)] and ResNet-152) as the backbone. We don’t use weight decay for all datasets and all fairness methods. We use linear learning rate decay for all datasets and all fairness methods. We use Adam [[14](#bib.bib14)] as the optimizer with a learning rate of 0.0010.0010.001 for both tabular and image data. As these objectives, utility and fairness, often present trade-offs, it can be challenging to determine when to stop model training, and this issue is rarely discussed in the previous literature. In this work, we adopted a straightforward stopping strategy based on our experience. We employ a linear decay strategy for the learning rate, halving it every 505050 training steps. The model training is stopped when the learning rate decreases to a value below 1​e−51superscript𝑒51e^{-5}. We provide all the wandb 333https://wandb.ai/ running logs at [our repo](https://github.com/ahxt/fair_fairness_benchmark), including all the hyperparameters for each run of experiment and the training process.

#### 5.1 How the Bias Mitigating Methods Perform on Utility-Fairness Trade-offs?

In this section, we present the results of experiments conducted to assess the performance of existing in-processing fairness methods in terms of the utility-fairness trade-offs. We analyze how well these methods balance optimizing utility and ensuring fairness in decision-making. To accurately reflect the performance of the different methods, we aggregate the performance across different datasets in one figure. To do so, we normalize the utility (acc,auc) and fairness (abcc, dp) performance based on the performance of the ERM. From the results, we make the following major observations:

➁ The utility-fairness performance of the current fairness method exhibits trade-offs. We first present the utility-fairness trade-offs of the existing in-processing fairness methods in [Figure 1](#S5.F1 "In 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). We conduct experiments using various in-processing fairness methods and analyze the ability to adjust the trade-offs to cater to specific needs while maintaining a balance between accuracy and fairness.

➂ The HSIC method achieves the best utility-fairness trade-off overall. The HSIC method consistently excels in balancing utility and fairness, outperforming other approaches across our tests. This method, depicted in green in [Figure 1](#S5.F1 "In 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"), shows particular effectiveness when applied to tabular data. It exhibits a significant ability to improve fairness measures without compromising the precision of utility, maintaining high accuracy in predictions. This quality affirms the robustness of the HSIC method in preserving utility-fairness equilibrium under various conditions. However, when this method is applied to image data, it exhibits a relative performance decline, showing lower fairness and utility scores.

➃ Adversarial debiasing methods exhibit instability. As illustrated in [Figure 1](#S5.F1 "In 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"), the utility-fairness points representing the  AdvDebias method are scattered randomly across the figures, failing to depict a consistent trade-off pattern. This randomness suggests an inherent instability in adversarial debiasing methods. This inconsistency is further demonstrated in subsequent experiments, where the training curves reveal that these methods are difficult to control effectively.

#### 5.2 Can the Utility-fairness Trade-offs be Controlled?

Hereby we investigate the extent to which the utility-fairness trade-offs can be controlled and fine-tuned. We conduct experiments using various in-processing fairness methods and analyze the ability to adjust the trade-offs to cater to specific needs and requirements while maintaining a balance between accuracy and fairness.

!(/html/2306.09468/assets/x5.png)

Figure 2: The fairness performance with varying fairness control hyperparameters. The intensity of the color represents the size of the control parameters. In most cases, the larger value of control parameters yields better fairness performance, while small ones have worse fairness performance. These figures are generated from 𝟏𝟑𝟏𝟏𝟎13110\mathbf{13110} runs of experiments.

➄ The utility-fairness trade-offs are generally controllable. The intensity of color in [Figure 2](#S5.F2 "In 5.2 Can the Utility-fairness Trade-offs be Controlled? ‣ 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") corresponds to the size of the control parameters. With the exception of adversarial debiasing, we find that the performance of most methods can be modulated effectively through the adjustment of hyperparameters. Specifically, larger control hyperparameters tend to yield lower dp and abcc values, indicating enhanced fairness. This implies that the trade-offs between utility and fairness can be actively managed in most cases, providing a crucial degree of flexibility in fairness-oriented data processing tasks. In comparison, the adversarial debiasing method (AdvDebias) is hard to control.

#### 5.3 How do Utility and Fairness Performance Change During Training Process?

!(/html/2306.09468/assets/x6.png)

Figure 3: The training curves on tabular dataset. The training curves for fairness metrics typically have lager standard deviation than utility performance, showing the instability of fairness performance.

!(/html/2306.09468/assets/x7.png)

Figure 4: The training curves on image dataset. The results are similar to tabular dataset that training curves for fairness metrics typically have lager standard deviation than utility performance.

In this section, we thoroughly examine the training curves of existing in-processing fairness methods, which are not sufficiently explored in previous studies. We conduct a series of experiments to track the evolution of utility and fairness performance throughout the training process and evaluate the impact of these dynamics on the final results. We presented the training curves in [Figures 3](#S5.F3 "In 5.3 How do Utility and Fairness Performance Change During Training Process? ‣ 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") and [4](#S5.F4 "Figure 4 ‣ 5.3 How do Utility and Fairness Performance Change During Training Process? ‣ 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") for tabular data and image data, respectively.

➅ The training curves of utility are stable, while those of fairness are unstable. Results on both tabular and image data show that the standard deviation for fairness metrics is significantly larger than that for utility metrics. Among the fairness methods, LAFTR shows the most stable fairness performance. Even though the value of fairness metrics is small, the large standard deviation still suggests unstable fairness performance. The results indicate a future research direction focused on enhancing fairness training stability.

➆ Stopping training while the learning rate is lower enough proves effective. In our work, we halt the model training when the learning rate diminishes to a value less than 1​e−51superscript𝑒51e^{-5}, which is decayed by multiplying by 0.10.10.1 every 505050 training steps. This approach results in stable fairness metrics, thereby validating its effectiveness and rationale. The utilization of learning rate decay to halt training results in stable fairness metrics, thereby affirming its efficacy and reasonableness.

#### 5.4 How does Model Size Influence Fairness Performance?

We conduct experiments to explore the influence of model size on fairness performance. We use various neural networks with the number of neural network trainable parameters spanning from 11.6M to 126.9M.444We use the following architectures: ResNet-18 (11.6M), ResNet-34 (21.8M), ResNet-50 (25.6M), ResNet-101 (44.5M), ResNet-152 (60.2M), ResNext-50 (25.0M), ResNext101 (88.8M), wide\_ResNet-50 (68.9M), and wide\_ResNet101 (126.9M). The results are presented in [Figure 5](#S5.F5 "In 5.4 How does Model Size Influence Fairness Performance? ‣ 5 Benchmarking Current Fairness Methods ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods").

!(/html/2306.09468/assets/x8.png)

Figure 5: The performance with varying size of neural networks. The x-axis is the number of model parameters. There is no clear relationship between model size and performance.

➇ The architecture does not significantly influence fairness performance. An increase in neural network parameters does not yield significant changes in utility or fairness performance. This suggests larger models do not naturally mitigate that dataset bias. The results observed from the DiffDP method indicate a potential correlation between utility performance and bias: fairness performance may deteriorate as utility increases, exhibiting trade-offs between them across different models.

### 6 Related Work

Algorithmic Fairness Fairness in machine learning has garnered considerable attention in recent years. The goal of fairness in machine learning is to ensure that the machine learning models are fair and unbiased towards an individual or group. Thus, fairness in machine learning can be divided into t categories: group fairness [[17](#bib.bib17), [23](#bib.bib23), [13](#bib.bib13), [50](#bib.bib50), [37](#bib.bib37)] and individual fairness [[17](#bib.bib17), [46](#bib.bib46)].
Group fairness aims to ensure that the machine learning models are fair to different groups of people, while individual fairness aims to “similar individuals should be treated similarly.” To mitigate fairness and bias problems in machine learning models, bias mitigation methods can be divided into three categories: pre-processing [[27](#bib.bib27), [8](#bib.bib8)], in-processing [[28](#bib.bib28), [52](#bib.bib52), [37](#bib.bib37), [53](#bib.bib53), [7](#bib.bib7), [2](#bib.bib2), [47](#bib.bib47), [40](#bib.bib40)], and post-processing [[23](#bib.bib23), [26](#bib.bib26)].
Given that the group fairness metrics are widely adopted in real-world applications and the emergence of more in-processing techniques designed in deep neural network models, we focus on benchmarking in-processing methods for group fairness for neural network models for tabular and image data.

Fairness Packages and Benchmarks There are many fairness packages in the literature. Among them,
AIF360 [[4](#bib.bib4)] and FairLearn [[6](#bib.bib6)] are the two most widely used Python packages that provide a set of metrics and algorithms to measure and mitigate bias in machine learning models. They provide a set of metrics to measure the bias of machine learning models, including disparate impact, statistical parity difference, and equal opportunity difference, and a set of algorithms to mitigate the bias of machine learning models.
However, both AIF360 and FairLearn implement the bias mitigation algorithms using Scikit-learn [[42](#bib.bib42)] API design (e.g., the use of fit() function) with complicated class inheritance, making the understanding and direct modification of implementation difficult.
In comparison, our benchmark decouples the implementation of different bias mitigation algorithms using Pytorch-style [[41](#bib.bib41)] training scripts and provides a unified fairness evaluation interface for a comprehensive list of group fairness metrics.
One recently proposed benchmark [[44](#bib.bib44)] also aims to benchmark bias mitigation algorithms. However, their benchmark only includes adversarial learning methods and datasets (e.g., a synthetic dataset and CI-MNIST) without fairness implications and uses dp and eodd as the fairness metrics. In contrast, our benchmark is more comprehensive in terms of algorithms, datasets, and fairness evaluation metrics.

### 7 Discussions

This paper introduces Fair Fairness Benchmark (FFB) to benchmark the in-processing group fairness models, offering extensible, minimalistic, and research-oriented open-source code, as well as comprehensive experiments to benchmark the existing in-processing group fairness method.

Future work. Our plan for subsequent phases of this work involves extending the scope of the FFB to include a wider range of in-processing group fairness methods. Moreover, we intend to incorporate additional definitions of fairness into our evaluations.

Social Impact. Our benchmark, with its extensible, minimalistic, and research-oriented open-source code, is designed to facilitate researchers and practitioners to explore and implement fairness methods. Standardized dataset preprocessing and reference baseline implementation will help reduce inconsistencies and make fairness more accessible, especially for beginners in the field. Ultimately, our work aims to stimulate future research for fairness and foster the development of fairness models.

### References

* Adel et al. [2019]

  Tameem Adel, Isabel Valera, Zoubin Ghahramani, and Adrian Weller.
  One-network adversarial fairness.
  In *Proceedings of the AAAI*, pages 2412–2420, 2019.
* Alghamdi et al. [2022]

  Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang, Peter Michalak, Shahab
  Asoodeh, and Flavio Calmon.
  Beyond adult and compas: Fair multi-class prediction via information
  projection.
  *Advances in Neural Information Processing Systems*,
  35:38747–38760, 2022.
* Baharlouei et al. [2020]

  Sina Baharlouei, Maher Nouiehed, Ahmad Beirami, and Meisam Razaviyayn.
  Rényi fair inference.
  In *International Conference on Learning Representations*, 2020.
  URL <https://openreview.net/forum?id=HkgsUJrtDB>.
* Bellamy et al. [2018]

  Rachel K. E. Bellamy, Kuntal Dey, Michael Hind, Samuel C. Hoffman, Stephanie
  Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta,
  Aleksandra Mojsilovic, Seema Nagar, Karthikeyan Natesan Ramamurthy, John
  Richards, Diptikalyan Saha, Prasanna Sattigeri, Moninder Singh, Kush R.
  Varshney, and Yunfeng Zhang.
  AI Fairness 360: An extensible toolkit for detecting,
  understanding, and mitigating unwanted algorithmic bias, October 2018.
  URL <https://arxiv.org/abs/1810.01943>.
* Beutel et al. [2017]

  Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi.
  Data decisions and theoretical implications when adversarially
  learning fair representations.
  *arXiv preprint arXiv:1707.00075*, 2017.
* Bird et al. [2020]

  Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz, Vanessa
  Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker.
  Fairlearn: A toolkit for assessing and improving fairness in ai.
  Technical Report MSR-TR-2020-32, Microsoft, May 2020.
  URL
  <https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/>.
* Buyl and De Bie [2022]

  Maarten Buyl and Tijl De Bie.
  Optimal transport of classifiers to fairness.
  In *Advances in Neural Information Processing Systems*, 2022.
* Calmon et al. [2017]

  Flavio Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan
  Natesan Ramamurthy, and Kush R Varshney.
  Optimized pre-processing for discrimination prevention.
  In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett, editors, *Advances in Neural
  Information Processing Systems*, volume 30. Curran Associates, Inc., 2017.
  URL
  <https://proceedings.neurips.cc/paper_files/paper/2017/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf>.
* Caton and Haas [2020]

  Simon Caton and Christian Haas.
  Fairness in machine learning: A survey.
  *arXiv preprint arXiv:2010.04053*, 2020.
* Chai et al. [2022]

  Junyi Chai, Taeuk Jang, and Xiaoqian Wang.
  Fairness without demographics through knowledge distillation.
  In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho,
  editors, *Advances in Neural Information Processing Systems*, 2022.
  URL <https://openreview.net/forum?id=8gjwWnN5pfy>.
* Chouldechova [2017]

  Alexandra Chouldechova.
  Fair prediction with disparate impact: A study of bias in recidivism
  prediction instruments.
  *Big data*, 5(2):153–163, 2017.
* Chuang and Mroueh [2020]

  Ching-Yao Chuang and Youssef Mroueh.
  Fair mixup: Fairness via interpolation.
  In *ICLR*, 2020.
* Corbett-Davies et al. [2017]

  Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq.
  Algorithmic decision making and the cost of fairness.
  In *Proceedings of the 23rd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, KDD ’17, page 797–806, New York,
  NY, USA, 2017. Association for Computing Machinery.
  ISBN 9781450348874.
  doi: 10.1145/3097983.3098095.
  URL <https://doi.org/10.1145/3097983.3098095>.
* Diederik P. Kingma [2014]

  Jimmy Ba Diederik P. Kingma.
  Adam: a method for stochastic optimization.
  *CoRR*, abs/1412.6980, 2014.
* Ding et al. [2021]

  Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt.
  Retiring adult: New datasets for fair machine learning.
  *NeurIPS*, 2021.
* Dua and Graff [2017]

  Dheeru Dua and Casey Graff.
  UCI machine learning repository, 2017.
  URL <http://archive.ics.uci.edu/ml>.
* Dwork et al. [2012]

  Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel.
  Fairness through awareness.
  In *Conference on Innovations in Theoretical Computer Science
  (ITCS)*, 2012.
* Edwards and Storkey [2015]

  Harrison Edwards and Amos Storkey.
  Censoring representations with an adversary.
  *arXiv preprint arXiv:1511.05897*, 2015.
* Feldman et al. [2015]

  Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and
  Suresh Venkatasubramanian.
  Certifying and removing disparate impact.
  In *proceedings of the 21th ACM SIGKDD international conference
  on knowledge discovery and data mining*, pages 259–268, 2015.
* Ganin et al. [2016]

  Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo
  Larochelle, François Laviolette, Mario March, and Victor Lempitsky.
  Domain-adversarial training of neural networks.
  *Journal of Machine Learning Research*, 17(59):1–35, 2016.
  URL <http://jmlr.org/papers/v17/15-239.html>.
* Gretton et al. [2005]

  Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf.
  Measuring statistical dependence with hilbert-schmidt norms.
  In *International conference on algorithmic learning theory*,
  pages 63–77. Springer, 2005.
* Han et al. [2023]

  Xiaotian Han, Zhimeng Jiang, Hongye Jin, Zirui Liu, Na Zou, Qifan Wang, and Xia
  Hu.
  Retiring $\delta \text{DP}$: New
  distribution-level metrics for demographic parity.
  *Transactions on Machine Learning Research*, 2023.
  ISSN 2835-8856.
  URL <https://openreview.net/forum?id=LjDFIWWVVa>.
* Hardt et al. [2016]

  Moritz Hardt, Eric Price, and Nati Srebro.
  Equality of opportunity in supervised learning.
  *Advances in neural information processing systems*, 29, 2016.
* He et al. [2016]

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  Deep residual learning for image recognition.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition (CVPR)*, June 2016.
* Hsu et al. [2022]

  Brian Hsu, Rahul Mazumder, Preetam Nandy, and Kinjal Basu.
  Pushing the limits of fairness impossibility: Who’s the fairest of
  them all?
  In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho,
  editors, *Advances in Neural Information Processing Systems*, 2022.
  URL <https://openreview.net/forum?id=bot35zOudq>.
* Jiang et al. [2020]

  Ray Jiang, Aldo Pacchiano, Tom Stepleton, Heinrich Jiang, and Silvia Chiappa.
  Wasserstein fair classification.
  In *Uncertainty in Artificial Intelligence*, pages 862–872.
  PMLR, 2020.
* Kamiran and Calders [2012]

  Faisal Kamiran and Toon Calders.
  Data preprocessing techniques for classification without
  discrimination.
  *Knowledge and information systems*, 33(1):1–33, 2012.
* Kamishima et al. [2012]

  Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma.
  Fairness-aware classifier with prejudice remover regularizer.
  In *Joint European conference on machine learning and knowledge
  discovery in databases*. Springer, 2012.
* Kleinberg et al. [2016]

  Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan.
  Inherent trade-offs in the fair determination of risk scores.
  *arXiv preprint arXiv:1609.05807*, 2016.
* Kohavi and Becker [1996]

  Ronny Kohavi and Barry Becker.
  Uci adult data set.
  *UCI Meachine Learning Repository*, 5, 1996.
* Larson et al. [2016]

  Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin.
  Propublica compas analysis—data and analysis for ‘machine bias’.
  <https://github.com/propublica/compas-analysis>, 2016.
  Accessed: 2023-03-13.
* Le Quy et al. [2022]

  Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, and Eirini Ntoutsi.
  A survey on datasets for fairness-aware machine learning.
  *Wiley Interdisciplinary Reviews: Data Mining and Knowledge
  Discovery*, 12(3):e1452, 2022.
* Li et al. [2019]

  Zhu Li, Adrian Perez-Suay, Gustau Camps-Valls, and Dino Sejdinovic.
  Kernel dependence regularizers and gaussian processes with
  applications to algorithmic fairness.
  *arXiv preprint arXiv:1911.04322*, 2019.
* Liu et al. [2015]

  Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
  Deep learning face attributes in the wild.
  In *Proceedings of International Conference on Computer Vision
  (ICCV)*, December 2015.
* Louizos et al. [2016]

  Christos Louizos, Kevin Swersky, Yujia Li, Max Welling, and Richard S Zemel.
  The variational fair autoencoder.
  In *International Conference on Learning Representations
  (ICLR)*, 2016.
* Louppe et al. [2017]

  Gilles Louppe, Michael Kagan, and Kyle Cranmer.
  Learning to pivot with adversarial networks.
  *NeurIPS*, 30, 2017.
* Madras et al. [2018]

  David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel.
  Learning adversarially fair and transferable representations.
  *International Conference on Machine Learning*, 2018.
* McNamara et al. [2019]

  Daniel McNamara, Cheng Soon Ong, and Robert C Williamson.
  Costs and benefits of fair representation learning.
  In *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics,
  and Society*, pages 263–270, 2019.
* Mehrabi et al. [2021]

  Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram
  Galstyan.
  A survey on bias and fairness in machine learning.
  *ACM Computing Surveys (CSUR)*, 54(6):1–35,
  2021.
* Mehrotra and Vishnoi [2022]

  Anay Mehrotra and Nisheeth Vishnoi.
  Fair ranking with noisy protected attributes.
  *Advances in Neural Information Processing Systems*,
  35:31711–31725, 2022.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
  Pytorch: An imperative style, high-performance deep learning library.
  In *Advances in neural information processing systems*, pages
  8026–8037, 2019.
* Pedregosa et al. [2011]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12:2825–2830,
  2011.
* Pessach and Shmueli [2022]

  Dana Pessach and Erez Shmueli.
  A review on fairness in machine learning.
  *ACM Computing Surveys (CSUR)*, 55(3):1–44,
  2022.
* Reddy et al. [2021a]

  Charan Reddy, Deepak Sharma, Soroush Mehri, Adriana Romero Soriano, Samira
  Shabanian, and Sina Honari.
  Benchmarking bias mitigation algorithms in representation learning
  through fairness metrics.
  In J. Vanschoren and S. Yeung, editors, *Proceedings of the
  Neural Information Processing Systems Track on Datasets and Benchmarks*,
  volume 1, 2021a.
  URL
  <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/2723d092b63885e0d7c260cc007e8b9d-Paper-round1.pdf>.
* Reddy et al. [2021b]

  Charan Reddy, Deepak Sharma, Soroush Mehri, Adriana Romero-Soriano, Samira
  Shabanian, and Sina Honari.
  Benchmarking bias mitigation algorithms in representation learning
  through fairness metrics.
  In *Thirty-fifth Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track (Round 1)*, 2021b.
  URL <https://openreview.net/forum?id=OTnqQUEwPKu>.
* Sharifi-Malvajerdi et al. [2019]

  Saeed Sharifi-Malvajerdi, Michael Kearns, and Aaron Roth.
  Average individual fairness: Algorithms, generalization and
  experiments.
  *Advances in neural information processing systems*, 32, 2019.
* Shui et al. [2022]

  Changjian Shui, Gezheng Xu, Qi Chen, Jiaqi Li, Charles X Ling, Tal Arbel, Boyu
  Wang, and Christian Gagné.
  On learning fairness and accuracy on multiple subgroups.
  *Advances in Neural Information Processing Systems*,
  35:34121–34135, 2022.
* Wan et al. [2023]

  Mingyang Wan, Daochen Zha, Ninghao Liu, and Na Zou.
  In-processing modeling techniques for machine learning fairness: A
  survey.
  *ACM Transactions on Knowledge Discovery from Data*, 17(3):1–27, 2023.
* Xian et al. [2023]

  Ruicheng Xian, Lang Yin, and Han Zhao.
  Fair and optimal classification via post-processing.
  In *International Conference on Machine Learning*, 2023.
* Zafar et al. [2017]

  Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rogriguez, and Krishna P.
  Gummadi.
  Fairness constraints: Mechanisms for fair classification.
  In *International Conference on Artificial Intelligence and
  Statistics (AISTATS)*, 2017.
* Zemel et al. [2013]

  Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork.
  Learning fair representations.
  In *International conference on machine learning*, pages
  325–333. PMLR, 2013.
* Zhang et al. [2018]

  Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell.
  Mitigating unwanted biases with adversarial learning.
  In *AAAI/ACM Conference on AI, Ethics, and Society (AIES)*,
  2018.
* Zhang et al. [2022]

  Guanhua Zhang, Yihua Zhang, Yang Zhang, Wenqi Fan, Qing Li, Sijia Liu, and
  Shiyu Chang.
  Fairness reprogramming.
  *arXiv preprint arXiv:2209.10222*, 2022.
* Zhang et al. [2017]

  Zhifei Zhang, Yang Song, and Hairong Qi.
  Age progression/regression by conditional adversarial autoencoder.
  In *IEEE Conference on Computer Vision and Pattern Recognition
  (CVPR)*. IEEE, 2017.
* Zhao and Gordon [2022]

  Han Zhao and Geoffrey J. Gordon.
  Inherent tradeoffs in learning fair representations.
  *Journal of Machine Learning Research*, 23(57):1–26, 2022.
  URL <http://jmlr.org/papers/v23/21-1427.html>.

## Appendix of FFB

\parttoc

The codes are at <https://github.com/ahxt/fair_fairness_benchmark>.
The partial running logs555We will release all the running logs. The released running logs are the experiment are the DiffDP method on the tabular dataset, which includes 𝟐𝟔𝟔𝟎2660\mathbf{2660} runs.are at <https://wandb.ai/fair_benchmark/exp1.diffdp>.

### Appendix A Details of the Group Fairness

In this section, we provide the details of the group fairness. We first introduce the definition of group fairness. Then, we introduce the existing group fairness metrics and algorithms.

* •

  dp (Demographic Parity or Statistical Parity) [[51](#bib.bib51)]. A classifier satisfies demographic parity if the predicted outcome Y^^𝑌\hat{Y} is independent of the sensitive attribute S𝑆S, i.e., P​(Y^∣S=0)=P​(Y^∣S=1)𝑃conditional^𝑌𝑆0𝑃conditional^𝑌𝑆1P(\hat{Y}\mid S=0)=P(\hat{Y}\mid S=1).
* •

  prule [[50](#bib.bib50)]. A classifier satisfies p𝑝p%-rule if the ratio between the probability of subjects having a certain sensitive attribute value assigned the positive decision outcome and the probability of subjects not having that value also assigned the positive outcome should be no less than p𝑝p/100, i.e., |P(Y^=1∣S=1)/P(Y^=1∣S=0)|≤p/100|P(\hat{Y}=1\mid S=1)/P(\hat{Y}=1\mid S=0)|\leq p/100.
* •

  eopp (Equality of Opportunity) [[23](#bib.bib23)]. A classifier satisfies equalized opportunity if the predicted outcome Y𝑌Y is independent of the sensitive attribute S𝑆S when the label Y=1𝑌1Y=1, i.e., P​(Y^∣S=0,Y=1)=P​(Y^∣S=1,Y=1)𝑃formulae-sequenceconditional^𝑌𝑆0𝑌1𝑃formulae-sequenceconditional^𝑌𝑆1𝑌1P(\hat{Y}\mid S=0,Y=1)=P(\hat{Y}\mid S=1,Y=1).
* •

  eodd (Equalized Odds) [[23](#bib.bib23)]. A classifier satisfies equalized odds if the predicted outcome Y𝑌Y is independent of the sensitive attribute S𝑆S conditioned on the label Y𝑌Y, i.e., P​(Y^∣S=0,Y=y)=P​(Y^∣S=1,Y=y),y∈{0,1}formulae-sequence𝑃formulae-sequenceconditional^𝑌𝑆0𝑌𝑦𝑃formulae-sequenceconditional^𝑌𝑆1𝑌𝑦𝑦01P(\hat{Y}\mid S=0,Y=y)=P(\hat{Y}\mid S=1,Y=y),y\in\{0,1\}.
* •

  acc (Accuracy Parity). A classifier satisfies accuracy parity if the error rates of different sensitive attribute values are the same, i.e., P​(Y^≠Y∣S=0)=P​(Y^≠Y∣S=1),y∈{0,1}formulae-sequence𝑃^𝑌conditional𝑌𝑆0𝑃^𝑌conditional𝑌𝑆1𝑦01P(\hat{Y}\neq Y\mid S=0)=P(\hat{Y}\neq Y\mid S=1),y\in\{0,1\}.
* •

  aucp (ROC AUC Parity). A classifier satisfies ROC AUC parity if its area under the receiver operating characteristic curve of w.r.t. different sensitive attribute values are the same.
* •

  ppv (Predictive Parity Value Parity) A classifier satisfies predictive parity value parity if the probability of a subject with a positive predictive value belonging to the positive class w.r.t. different sensitive attribute values are the same, i.e., P​(Y=1∣Y^,S=0)=P​(Y=1∣Y^,S=1)𝑃𝑌conditional1
  ^𝑌𝑆0𝑃𝑌conditional1
  ^𝑌𝑆1P(Y=1\mid\hat{Y},S=0)=P(Y=1\mid\hat{Y},S=1).
* •

  bnegc (Balance for Negative Class). A classifier satisfies balance for the negative class if the average predicted probability of a subject belonging to the negative class is the same w.r.t. different sensitive attribute values, i.e., 𝔼​[f​(X)∣Y=0,S=0]=𝔼​[f​(X)∣Y=0,S=1]𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌0𝑆0𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌0𝑆1\mathbb{E}[f(X)\mid Y=0,S=0]=\mathbb{E}[f(X)\mid Y=0,S=1].
* •

  bposc (Balance for Positive Class). A classifier satisfies balance for the negative class if the average predicted probability of a subject belonging to the positive class is the same w.r.t. different sensitive attribute values, i.e., 𝔼​[f​(X)∣Y=1,S=0]=𝔼​[f​(X)∣Y=1,S=1]𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌1𝑆0𝔼delimited-[]formulae-sequenceconditional𝑓𝑋𝑌1𝑆1\mathbb{E}[f(X)\mid Y=1,S=0]=\mathbb{E}[f(X)\mid Y=1,S=1].
* •

  abcc (Area Between Cumulative density function Curves) [[22](#bib.bib22)] is proposed to precisely measure the violation of demographic parity at the distribution level. The new fairness metrics directly measure the difference between the distributions of the prediction probability for different demographic groups

### Appendix B Details of the Benchmarking Datasets

In this section, we provide the details of the benchmarking datasets. We first introduce the benchmarking datasets. Then, we introduce the data preprocessing and data splitting.

* •

  Tabular Datasets

  + –

    Adult666<https://archive.ics.uci.edu/ml/datasets/adult> [[30](#bib.bib30)]. The Adult dataset is widely used in machine learning and data mining research. It contains 1994 U.S. census data. The task of the dataset is to predict whether a person makes over $50K a year, given an individual’s demographic and financial information. The dataset includes sensitive information such as age and gender. In the literature, gender is mostly used as the (binary) sensitive attribute to evaluate group fairness.
  + –

    COMPAS777<https://github.com/propublica/compas-analysis> [[31](#bib.bib31)]. The COMPAS dataset contains records of criminal defendants, and it is used to predict whether the defendant will recidivate within two years. The dataset includes attributes related to the defendant, such as their criminal history, and demographic information, such as gender and race.
  + –

    German888<https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data> [[16](#bib.bib16)]. The German Credit dataset contains information on individuals who applied for credit at a German bank, including their financial status, credit history, and demographic information (e.g., gender and age). It is used to predict whether an individual should receive a positive or negative credit risk rating.
  + –

    Bank999<https://archive.ics.uci.edu/dataset/222/bank+marketing> [[16](#bib.bib16)]. The bank marketing dataset is used to analyze the effectiveness of marketing strategies of a Portuguese banking institution by predicting if the client will subscribe to a term deposit. The input variables of the dataset include the bank client’s personal information and other bank marketing activities related to the client. Age was studied as the sensitive attribute in [[50](#bib.bib50)].
  + –

    KDDCensus101010<https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)> [[16](#bib.bib16)]. Similar to the Adult dataset, the task of the KDD Census dataset is to predict whether the individual’s income is above $50k with more instances. The sensitive attributes are gender and race.
  + –

    ACS-I/E/P/M/T111111<https://github.com/zykls/folktables> [[15](#bib.bib15)]. The ACS dataset provides several prediction tasks (e.g., predict whether an individual’s income is above $50K or whether an individual is employed). It is constructed from the American Community Survey (ACS) Public Use Microdata Sample (PUMS). All tasks contain features for race, gender, and other task-related features.
* •

  Image Datasets

  + –

    CelebA-A/W/S121212<https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> [[34](#bib.bib34)] The CelebFaces Attributes dataset comprises 20k face images from 10k celebrities. Each image is annotated with 40 binary labels indicating specific facial attributes such as gender, hair color, and age.
  + –

    UTKFace131313<https://susanqq.github.io/UTKFace/> [[54](#bib.bib54)]. The UTKFace dataset is a large-scale face dataset that contains over 20k face images of people from different ethnicities and ages. The images are annotated with age, gender, and ethnicity information.

### Appendix C Detailed Experimental Settings

We present the details of the experimental setting in [Tables 5](#A3.T5 "In Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"), [6](#A3.T6 "Table 6 ‣ Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") and [7](#A3.T7 "Table 7 ‣ Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). [Table 5](#A3.T5 "In Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") presents the common hyperparameters used by both Tabular and Image datasets, including an initial learning rate of 0.010.010.01, Adam as the optimizer, zero weight decay, StepLR as the scheduler with a step of 505050, a gamma value of 0.10.10.1, and 150 training steps in total. [Table 6](#A3.T6 "In Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") presents the range of control hyperparameters used for various fairness methods. Each method has a unique range of these parameters. [Table 7](#A3.T7 "In Appendix C Detailed Experimental Settings ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") indicates the batch sizes chosen for various datasets during training, ranging from 323232 for the German and COMPAS datasets to a large 409640964096 for the KDDCensus and ACS-I/E/P/M/T datasets, with CelebA-A/W/S and UTKFace datasets using a batch size of 128128128, which are determined by the number of instances of the datasets.

Table 5: Common Hyper-parameters.

| Dataset | Initial LR | Optimizer | Weight Decay | Scheduler | StepLR\_step | StepLR\_gamma | Training steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tabular | 0.01 | Adam | 0.0 | StepLR | 50 | 0.1 | 150 |
| Image | 0.01 | Adam | 0.0 | StepLR | 50 | 0.1 | 150 |

Table 6: The fairness control hyperparameter selections.

| Dataset | Control hyperparameter |
| --- | --- |
| DiffDP | 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4  0.20.40.60.81.01.21.41.61.82.02.53.03.540.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4 |
| DiffEodd | 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4  0.20.40.60.81.01.21.41.61.82.02.53.03.540.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4 |
| DiffEopp | 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4  0.20.40.60.81.01.21.41.61.82.02.53.03.540.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4 |
| PRemover | 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.50,0.6,0.7,0.8,0.9,1.0  0.050.10.150.20.250.30.350.400.450.500.60.70.80.91.00.05,0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.50,0.6,0.7,0.8,0.9,1.0 |
| HSIC | 50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000  50100150200250300350400450500600700800900100050,100,150,200,250,300,350,400,450,500,600,700,800,900,1000 |
| AdvDebias | 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4  0.20.40.60.81.01.21.41.61.82.02.53.03.540.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4 |
| LAFTR | 0.1,0.2,0.3,0.4,0.5,1,2,3,4,5  0.10.20.30.40.5123450.1,0.2,0.3,0.4,0.5,1,2,3,4,5 |

Table 7: The batch size for different datasets during the training.

| Dataset | Bank | German | Adult | COMPAS | KDDCensus | ACS-I/E/P/M/T | CelebA-A/W/S | UTKFace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Batch Size | 102410241024 | 323232 | 102410241024 | 323232 | 409640964096 | 409640964096 | 128128128 | 128128128 |

### Appendix D More Experiment Results on Adult

In this appendix, we present the experimental results on Adult datasets.

#### D.1 Utility-Fairness Trade-offs

We plot the utility-fairness trade-offs for the Adult dataset with gender as the sensitive attribute and present the results in [Figures 6](#A4.F6 "In D.1 Utility-Fairness Trade-offs ‣ Appendix D More Experiment Results on Adult ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") and [7](#A4.F7 "Figure 7 ‣ D.1 Utility-Fairness Trade-offs ‣ Appendix D More Experiment Results on Adult ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods").

!(/html/2306.09468/assets/x9.png)

!(/html/2306.09468/assets/x10.png)

!(/html/2306.09468/assets/x11.png)

!(/html/2306.09468/assets/x12.png)

!(/html/2306.09468/assets/x13.png)

!(/html/2306.09468/assets/x14.png)

!(/html/2306.09468/assets/x15.png)

!(/html/2306.09468/assets/x16.png)

!(/html/2306.09468/assets/x17.png)

!(/html/2306.09468/assets/x18.png)

Figure 6: The Utility-Fairness Trade-offs with acc as utility metric.

!(/html/2306.09468/assets/x19.png)

!(/html/2306.09468/assets/x20.png)

!(/html/2306.09468/assets/x21.png)

!(/html/2306.09468/assets/x22.png)

!(/html/2306.09468/assets/x23.png)

!(/html/2306.09468/assets/x24.png)

!(/html/2306.09468/assets/x25.png)

!(/html/2306.09468/assets/x26.png)

!(/html/2306.09468/assets/x27.png)

!(/html/2306.09468/assets/x28.png)

Figure 7: The Utility-Fairness Trade-offs with auc as utility metric.

#### D.2 Training Curves and Hyperparameters for Controlling Fairness

We plot the utility and fairness training curves for varying fairness control hyperparameters on the Adult dataset, and present the results in [Figure 8](#A4.F8 "In D.2 Training Curves and Hyperparameters for Controlling Fairness ‣ Appendix D More Experiment Results on Adult ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). The intensity of the color represents the size of the control parameters. In most cases, the larger value of control parameters yields better fairness performance, while small ones have worse fairness performance.

!(/html/2306.09468/assets/x29.png)

Figure 8: Hyperparameters for Controlling Fairness on Adult dataset.

### Appendix E More Experiment Results on CelebA-A

In this appendix, we present the experimental results on the CelebA-A dataset.

#### E.1 Utility-Fairness Trade-offs

We plot the utility-fairness trade-offs for the CelebA-A dataset with gender as the sensitive attribute and present the results in [Figure 9](#A5.F9 "In E.1 Utility-Fairness Trade-offs ‣ Appendix E More Experiment Results on CelebA-A ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). The results show the utility-fairness trade-offs in CelebA-A dataset.

!(/html/2306.09468/assets/x30.png)

Figure 9: The Utility-Fairness Trade-offs

#### E.2 Training Curves and Hyperparameters for Controlling Fairness

We plot the utility and fairness training curves for varying fairness control hyperparameters on the CelebA-A dataset, and present the results in [Figure 10](#A5.F10 "In E.2 Training Curves and Hyperparameters for Controlling Fairness ‣ Appendix E More Experiment Results on CelebA-A ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"). The intensity of the color represents the size of the control parameters. In most cases, the larger value of control parameters yields better fairness performance, while small ones have worse fairness performance.

!(/html/2306.09468/assets/x31.png)

Figure 10: Training Curves and Hyperparameters for Controlling Fairness one CelebA-A dataset.

### Appendix F Implementation Comparison with AIF360 and FairLearn

In this section, we provide the implementations of adversarial debiasing in AIF360, FairLearn, and FFB to demonstrate FFB are extensible, minimalistic, and research-oriented compared to existing fairness packages. [Algorithm 1](#algorithm1 "In Appendix F Implementation Comparison with AIF360 and FairLearn ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"), [Algorithm 2](#algorithm2 "In Appendix F Implementation Comparison with AIF360 and FairLearn ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods"), and [Algorithm 3](#algorithm3 "In Appendix F Implementation Comparison with AIF360 and FairLearn ‣ Appendix of FFB ‣ FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods") show the implementation of adversarial debiasing in AIF360, FairLearn, and FFB, respectively.

We can see that both AIF360 and FairLearn use Scikit-learn API design (e.g., the use of fit() function), whereas FFB use Pytorch-style implementation, which provides a unified data preprocessing pipeline and fairness evaluation interface in a single script. Thus, researchers using FFB can use the bias mitigation method and reproduce the experimental results using one line of command.
Additionally, AIF360 and FairLearn use complicated class inheritance (e.g., AdversarialFairnessClassifier in FairLearn inherents AdversarialFairness and ClassifierMixin), AdversarialDebiasing in AIF360 inherents Transformer), and other external dependencies (e.g., AdversarialFairness in FairLearn uses backendEngine\_ to implement the training step), making the implementation hard to read. This makes researchers hard to understand and re-implement the bias mitigation methods.

Algorithm 1  AdvDebias in AIF360

[⬇](data:text/plain;base64,Y2xhc3MgQWR2ZXJzYXJpYWxEZWJpYXNpbmcoVHJhbnNmb3JtZXIpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICAuLi4KCiAgICBkZWYgX2NsYXNzaWZpZXJfbW9kZWwoc2VsZiwgZmVhdHVyZXMsIGZlYXR1cmVzX2RpbSwga2VlcF9wcm9iKToKICAgICAgICAuLi4gIyBkZWluZSBjbGFzc2lmaWVyCgogICAgZGVmIF9hZHZlcnNhcnlfbW9kZWwoc2VsZiwgcHJlZF9sb2dpdHMsIHRydWVfbGFiZWxzKToKICAgICAgICAuLi4gIyBkZWluZSBhZHZlcnNhcnkgbW9kZWwKCiAgICBkZWYgcHJlZGljdChzZWxmLCBkYXRhc2V0KToKICAgICAgICAuLi4KCiAgICBkZWYgZml0KHNlbGYsIGRhdGFzZXQpOgoKICAgICAgICB3aXRoIHRmLnZhcmlhYmxlX3Njb3BlKHNlbGYuc2NvcGVfbmFtZSk6CgogICAgICAgICAgICAjIHRmIGdyYXBoIGNvbnN0cnVjdGlvbgogICAgICAgICAgICAuLi4KCiAgICAgICAgICAgIHNlbGYuc2Vzcy5ydW4odGYuZ2xvYmFsX3ZhcmlhYmxlc19pbml0aWFsaXplcigpKQogICAgICAgICAgICBzZWxmLnNlc3MucnVuKHRmLmxvY2FsX3ZhcmlhYmxlc19pbml0aWFsaXplcigpKQoKCiAgICAgICAgICAgIGZvciBlcG9jaCBpbiByYW5nZShzZWxmLm51bV9lcG9jaHMpOgogICAgICAgICAgICAgICAgIyB0cmFpbmluZwogICAgICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UobnVtX3RyYWluX3NhbXBsZXMvL3NlbGYuYmF0Y2hfc2l6ZSk6CiAgICAgICAgICAgICAgICAgICAgYmF0Y2hfZmVlZF9kaWN0ID0ge3NlbGYuZmVhdHVyZXNfcGg6IGJhdGNoX2ZlYXR1cmVzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZWxmLnRydWVfbGFiZWxzX3BoOiBiYXRjaF9sYWJlbHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYucHJvdGVjdGVkX2F0dHJpYnV0ZXNfcGg6IGJhdGNoX3Byb3RlY3RlZF9hdHRyaWJ1dGVzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmtlZXBfcHJvYjogMC44fQogICAgICAgICAgICAgICAgICAgIGlmIHNlbGYuZGViaWFzOgogICAgICAgICAgICAgICAgICAgICAgICBfLCBfLCBwcmVkX2xhYmVsc19sb3NzX3ZhbHVlLCBwcmVkX3Byb3RlY3RlZF9hdHRyaWJ1dGVzX2xvc3NfdmFsZSA9IHNlbGYuc2Vzcy5ydW4oWwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjbGFzc2lmaWVyX21pbmltaXplciwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYWR2ZXJzYXJ5X21pbmltaXplciwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJlZF9sYWJlbHNfbG9zcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcHJlZF9wcm90ZWN0ZWRfYXR0cmlidXRlc19sb3NzXSwgZmVlZF9kaWN0PWJhdGNoX2ZlZWRfZGljdCkKICAgICAgICAgICAgICAgICAgICAgICAgaWYgaSAlIDIwMCA9PSAwOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgLi4uICMgbG9nZ2luZwogICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgIF8sIHByZWRfbGFiZWxzX2xvc3NfdmFsdWUgPSBzZWxmLnNlc3MucnVuKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgW2NsYXNzaWZpZXJfbWluaW1pemVyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByZWRfbGFiZWxzX2xvc3NdLCBmZWVkX2RpY3Q9YmF0Y2hfZmVlZF9kaWN0KQogICAgICAgICAgICAgICAgICAgICAgICBpZiBpICUgMjAwID09IDA6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAuLi4gIyBsb2dnaW5nCg==)

class AdversarialDebiasing(Transformer):

def \_\_init\_\_(self):

...

def \_classifier\_model(self, features, features\_dim, keep\_prob):

... # deine classifier

def \_adversary\_model(self, pred\_logits, true\_labels):

... # deine adversary model

def predict(self, dataset):

...

def fit(self, dataset):

with tf.variable\_scope(self.scope\_name):

# tf graph construction

...

self.sess.run(tf.global\_variables\_initializer())

self.sess.run(tf.local\_variables\_initializer())

for epoch in range(self.num\_epochs):

# training

for i in range(num\_train\_samples//self.batch\_size):

batch\_feed\_dict = {self.features\_ph: batch\_features,

self.true\_labels\_ph: batch\_labels,

self.protected\_attributes\_ph: batch\_protected\_attributes,

self.keep\_prob: 0.8}

if self.debias:

\_, \_, pred\_labels\_loss\_value, pred\_protected\_attributes\_loss\_vale = self.sess.run([

classifier\_minimizer,

adversary\_minimizer,

pred\_labels\_loss,

pred\_protected\_attributes\_loss], feed\_dict=batch\_feed\_dict)

if i % 200 == 0:

... # logging

else:

\_, pred\_labels\_loss\_value = self.sess.run(

[classifier\_minimizer,

pred\_labels\_loss], feed\_dict=batch\_feed\_dict)

if i % 200 == 0:

... # logging

Algorithm 2  AdvDebias in FairLearn

[⬇](data:text/plain;base64,Y2xhc3MgX0FkdmVyc2FyaWFsRmFpcm5lc3MoQmFzZUVzdGltYXRvcik6CgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIC4uLgoKICAgIGRlZiBfX3NldHVwKHNlbGYsIFgsIFksIEEpOgogICAgICAgIC4uLgoKICAgIGRlZiBmaXQoc2VsZiwgWCwgeSwgKiwgc2Vuc2l0aXZlX2ZlYXR1cmVzPU5vbmUpOgogICAgICAgIFgsIFksIEEgPSBzZWxmLl92YWxpZGF0ZV9pbnB1dChYLCB5LCBzZW5zaXRpdmVfZmVhdHVyZXMsIHJlaW5pdGlhbGl6ZT1UcnVlKQoKICAgICAgICAuLi4KCiAgICAgICAgZm9yIGVwb2NoIGluIHJhbmdlKGVwb2Nocyk6CiAgICAgICAgICAgIGJhdGNoX3NsaWNlID0gc2xpY2UoCiAgICAgICAgICAgICAgICAgICAgYmF0Y2ggKiBiYXRjaF9zaXplLAogICAgICAgICAgICAgICAgICAgIG1pbigoYmF0Y2ggKyAxKSAqIGJhdGNoX3NpemUsIFguc2hhcGVbMF0pLAogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAoTFAsIExBKSA9IHNlbGYuYmFja2VuZEVuZ2luZV8udHJhaW5fc3RlcCgKICAgICAgICAgICAgICAgIFhbYmF0Y2hfc2xpY2VdLCBZW2JhdGNoX3NsaWNlXSwgQVtiYXRjaF9zbGljZV0KICAgICAgICAgICAgKQogICAgICAgICAgICBwcmVkaWN0b3JfbG9zc2VzLmFwcGVuZChMUCkKICAgICAgICAgICAgYWR2ZXJzYXJ5X2xvc3Nlcy5hcHBlbmQoTEEpCgogICAgICAgICAgICAuLi4KCiAgICBkZWYgcGFydGlhbF9maXQoc2VsZiwgWCwgeSwgKiwgc2Vuc2l0aXZlX2ZlYXR1cmVzPU5vbmUpOgogICAgICAgIC4uLgoKICAgIGRlZiBkZWNpc2lvbl9mdW5jdGlvbihzZWxmLCBYKToKICAgICAgICAuLi4KCiAgICBkZWYgcHJlZGljdChzZWxmLCBYKToKICAgICAgICAuLi4KCiAgICBkZWYgX3ZhbGlkYXRlX2lucHV0KHNlbGYsIFgsIFksIEEsIHJlaW5pdGlhbGl6ZT1GYWxzZSk6CiAgICAgICAgLi4uCgogICAgZGVmIF92YWxpZGF0ZV9iYWNrZW5kKHNlbGYpOgogICAgICAgIC4uLgoKICAgIGRlZiBfc2V0X3ByZWRpY3Rvcl9mdW5jdGlvbihzZWxmKToKICAgICAgICAuLi4KCmNsYXNzIEFkdmVyc2FyaWFsRmFpcm5lc3NDbGFzc2lmaWVyKF9BZHZlcnNhcmlhbEZhaXJuZXNzLCBDbGFzc2lmaWVyTWl4aW4pOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgICIiIkluaXRpYWxpemUgbW9kZWwgYnkgc2V0dGluZyB0aGUgcHJlZGljdG9yIGxvc3MgYW5kIGZ1bmN0aW9uLiIiIgogICAgICAgIHNlbGYuX2VzdGltYXRvcl90eXBlID0gImNsYXNzaWZpZXIiCiAgICAgICAgc3VwZXIoQWR2ZXJzYXJpYWxGYWlybmVzc0NsYXNzaWZpZXIsIHNlbGYpLl9faW5pdF9fKCkKCg==)

class \_AdversarialFairness(BaseEstimator):

def \_\_init\_\_(self):

...

def \_\_setup(self, X, Y, A):

...

def fit(self, X, y, \*, sensitive\_features=None):

X, Y, A = self.\_validate\_input(X, y, sensitive\_features, reinitialize=True)

...

for epoch in range(epochs):

batch\_slice = slice(

batch \* batch\_size,

min((batch + 1) \* batch\_size, X.shape[0]),

)

(LP, LA) = self.backendEngine\_.train\_step(

X[batch\_slice], Y[batch\_slice], A[batch\_slice]

)

predictor\_losses.append(LP)

adversary\_losses.append(LA)

...

def partial\_fit(self, X, y, \*, sensitive\_features=None):

...

def decision\_function(self, X):

...

def predict(self, X):

...

def \_validate\_input(self, X, Y, A, reinitialize=False):

...

def \_validate\_backend(self):

...

def \_set\_predictor\_function(self):

...

class AdversarialFairnessClassifier(\_AdversarialFairness, ClassifierMixin):

def \_\_init\_\_(self):

"""Initialize␣model␣by␣setting␣the␣predictor␣loss␣and␣function."""

self.\_estimator\_type = "classifier"

super(AdversarialFairnessClassifier, self).\_\_init\_\_()

Algorithm 3  AdvDebias in FFB

[⬇](data:text/plain;base64,Y2xhc3MgQWR2ZXJzYXJ5KG5uLk1vZHVsZSk6CiAgICAuLi4KCmNsYXNzIE1MUChubi5Nb2R1bGUpOgogICAgLi4uCgpkZWYgdGVzdChtb2RlbCwgdGVzdF9sb2FkZXIsIGNyaXRlcmlvbiwgZGV2aWNlLCBhcmdzPU5vbmUpOgogICAgLi4uCgpkZWYgdHJhaW4oY2xmLCBhZHYsIGRhdGFfbG9hZGVyLCBjbGZfY3JpdGVyaW9uLCBhZHZfY3JpdGVyaW9uLCBjbGZfb3B0aW1pemVyLCBhZHZfb3B0aW1pemVyKToKICAgIC4uLgoKaWYgX19uYW1lX18gPT0gJ19fbWFpbl9fJzoKICAgIHBhcnNlciA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKCkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoJy0tZGF0YXNldCcsIHR5cGU9c3RyLCBkZWZhdWx0PSJhZHVsdCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCctLW1vZGVsJywgdHlwZT1zdHIsIGRlZmF1bHQ9Ik1MUCIpCiAgICAuLi4KICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCgogICAgIyBEYXRhc2V0IHNlbGVjdGlvbgogICAgaWYgYXJncy5kYXRhc2V0ID09ICJhZHVsdCI6CiAgICAgICAgWCwgeSwgcyA9IGxvYWRfYWR1bHRfZGF0YShzZW5zaXRpdmVfYXR0cmlidXRlPWFyZ3Muc2Vuc2l0aXZlX2F0dHIpCiAgICBlbGlmIGFyZ3MuZGF0YXNldCA9PSAiY29tcGFzIjoKICAgICAgICBYLCB5LCBzID0gbG9hZF9jb21wYXNfZGF0YSggc2Vuc2l0aXZlX2F0dHJpYnV0ZT1hcmdzLnNlbnNpdGl2ZV9hdHRyKQogICAgLi4uCgogICAgIyBVbmlmaWVkIERhdGFzZXQgcHJlcHJvY2Vzc2luZyAoZS5nLiwgdHJhaW4vdGVzdCBzcGxpdCwgKQogICAgLi4uCgogICAgIyBkZWZpbmUgbmV0d29yayBhcmNoaXRlY3R1cmUsIGV0Yy4gb3B0aW1pemVyCiAgICBjbGYgPSBNTFAobl9mZWF0dXJlcz1uX2ZlYXR1cmVzLCBudW1fY2xhc3Nlcz0xLCBtbHBfbGF5ZXJzPVs1MTIsIDI1NiwgNjRdKS50byhkZXZpY2UpCiAgICBjbGZfY3JpdGVyaW9uID0gbm4uQkNFTG9zcygpCiAgICBjbGZfb3B0aW1pemVyID0gb3B0aW0uQWRhbSggY2xmLnBhcmFtZXRlcnMoKSwgbHI9YXJncy5scikKCiAgICBhZHYgPSBBZHZlcnNhcnkoIG5fc2Vuc2l0aXZlID0gMSApLnRvKGRldmljZSkKICAgIGFkdl9jcml0ZXJpb24gPSBubi5CQ0VMb3NzKHJlZHVjdGlvbj0ibWVhbiIpCiAgICBhZHZfb3B0aW1pemVyID0gb3B0aW0uQWRhbShhZHYucGFyYW1ldGVycygpLCBscj1hcmdzLmxyKQoKCiAgICBmb3IgZXBvY2ggaW4gcmFuZ2UoMSwgYXJncy5udW1fZXBvY2hzKzEpOgogICAgICAgICMgYmVnaW4gdHJhaW5pbmcKICAgICAgICB0cmFpbihjbGYsIGFkdiwgdHJhaW5fbG9hZGVyLCBjbGZfY3JpdGVyaW9uLCBhZHZfY3JpdGVyaW9uLCBjbGZfb3B0aW1pemVyLCBhZHZfb3B0aW1pemVyKQoKICAgICAgICBpZiBlcG9jaCAlIGFyZ3MubG9nZ2luZ19zdGVwcyA9PSAwIG9yIGVwb2NoID09IGFyZ3MubnVtX2Vwb2NoczoKICAgICAgICAgICAgdGVzdF9tZXRyaWNzICA9ICB0ZXN0KG1vZGVsPWNsZiwgdGVzdF9sb2FkZXI9dGVzdF9sb2FkZXIsIGNyaXRlcmlvbj1jbGZfY3JpdGVyaW9uLCBkZXZpY2U9ZGV2aWNlKQogICAgICAgICAgICAjIGxvZ2dpbmcgbWV0cmljcwogICAgICAgICAgICAuLi4K)

class Adversary(nn.Module):

...

class MLP(nn.Module):

...

def test(model, test\_loader, criterion, device, args=None):

...

def train(clf, adv, data\_loader, clf\_criterion, adv\_criterion, clf\_optimizer, adv\_optimizer):

...

if \_\_name\_\_ == ’\_\_main\_\_’:

parser = argparse.ArgumentParser()

parser.add\_argument(’--dataset’, type=str, default="adult")

parser.add\_argument(’--model’, type=str, default="MLP")

...

args = parser.parse\_args()

# Dataset selection

if args.dataset == "adult":

X, y, s = load\_adult\_data(sensitive\_attribute=args.sensitive\_attr)

elif args.dataset == "compas":

X, y, s = load\_compas\_data( sensitive\_attribute=args.sensitive\_attr)

...

# Unified Dataset preprocessing (e.g., train/test split, )

...

# define network architecture, etc. optimizer

clf = MLP(n\_features=n\_features, num\_classes=1, mlp\_layers=[512, 256, 64]).to(device)

clf\_criterion = nn.BCELoss()

clf\_optimizer = optim.Adam( clf.parameters(), lr=args.lr)

adv = Adversary( n\_sensitive = 1 ).to(device)

adv\_criterion = nn.BCELoss(reduction="mean")

adv\_optimizer = optim.Adam(adv.parameters(), lr=args.lr)

for epoch in range(1, args.num\_epochs+1):

# begin training

train(clf, adv, train\_loader, clf\_criterion, adv\_criterion, clf\_optimizer, adv\_optimizer)

if epoch % args.logging\_steps == 0 or epoch == args.num\_epochs:

test\_metrics = test(model=clf, test\_loader=test\_loader, criterion=clf\_criterion, device=device)

# logging metrics

...
