---
arxiv: '2107.07455'
authors:
- Andrey Malinin
- Neil Band
- Ganshin
- Alexander
- German Chesnokov
- Yarin Gal
- Mark J. F. Gales
- Alexey Noskov
- Andrey Ploskonosov
- Liudmila Prokhorenkova
- Ivan Provilkov
- Vatsal Raina
- Vyas Raina
- Roginskiy
- Denis
- Mariya Shmatova
- Panos Tigas
- Boris Yangel
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale
  Tasks'
url: http://arxiv.org/abs/2107.07455v3
year: 2021
---

# Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks

Andrey Malinin1,2
  
Neil Band5
  
Yarin Gal5,6
  
Mark J. F. Gales4
  
Alexander Ganshin 1
  
German Chesnokov1
  
Alexey Noskov1
  
Andrey Ploskonosov1
  
Liudmila Prokhorenkova1,2,3
  
Ivan Provilkov1,3
  
Vatsal Raina4
  
Vyas Raina4
  
Denis Roginskiy 1
  
Mariya Shmatova1
  
Panos Tigas5
  
Boris Yangel1
  
am969@yandex-team.ru

###### Abstract

There has been significant research done on developing methods for improving robustness to distributional shift and uncertainty estimation. In contrast, only limited work has examined developing standard datasets and benchmarks for assessing these approaches. Additionally, most work on uncertainty estimation and robustness has developed new techniques based on small-scale regression or image classification tasks. However, many tasks of practical interest have different modalities, such as tabular data, audio, text, or sensor data, which offer significant challenges involving regression and discrete or continuous structured prediction. Thus, given the current state of the field, a standardized large-scale dataset of tasks across a range of modalities affected by distributional shifts is necessary. This will enable researchers to meaningfully evaluate the plethora of recently developed uncertainty quantification methods, as well as assessment criteria and state-of-the-art baselines. In this work, we propose the *Shifts Dataset* for evaluation of uncertainty estimates and robustness to distributional shift. The dataset, which has been collected from industrial sources and services, is composed of three tasks, with each corresponding to a particular data modality: tabular weather prediction, machine translation, and self-driving car (SDC) vehicle motion prediction. All of these data modalities and tasks are affected by real, “in-the-wild” distributional shifts and pose interesting challenges with respect to uncertainty estimation. In this work we provide a description of the dataset and baseline results for all tasks.

††1Yandex, 2HSE University, 3Moscow Institute of Physics and Technology, 4ALTA Institute, University of Cambridge, 5University of Oxford, 6Alan Turing Institute

## 1 Introduction

Machine learning models are being applied to numerous areas [[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3), [4](#bib.bib4), [5](#bib.bib5), [6](#bib.bib6)] and are widely deployed in production. An assumption which pervades all of machine learning is that the training, validation, and deployment data are independent and identically distributed (i.i.d.). Thus, good performance and generalization on validation data imply that the model will perform well in deployment. Unfortunately, this assumption seldom holds in real, “in the wild”, applications. In practice, data are subject to a wide range of possible *distributional shifts* — mismatches between the training data, and test or deployment data [[7](#bib.bib7), [8](#bib.bib8), [9](#bib.bib9)]. In general, the greater the degree of shift, the poorer is the model’s performance. The problem of distributional shift is of relevance not only to academic researchers, but to the machine learning community at-large. Indeed, *all* ML practitioners have faced the issue of mismatch between the training and test sets. This is especially important in high-risk applications of machine learning, such as finance, medicine, and autonomous vehicles. In such applications a mistake on part of an ML system may incur financial or reputational loss, or possible loss of life. It is therefore increasingly important to assess both a model’s *robustness* to distribution shift and its estimates of *predictive uncertainty*, which enable it to detect distributional shifts [[10](#bib.bib10), [11](#bib.bib11), [12](#bib.bib12)].

The area of uncertainty estimation and robustness has developed rapidly in recent years. Model averaging  [[13](#bib.bib13), [14](#bib.bib14), [15](#bib.bib15), [16](#bib.bib16)] has emerged as the de-facto standard approach to uncertainty estimation. Ensemble- and sampling-based uncertainty estimates have been successfully applied in detecting misclassifications, out-of-distribution inputs, adversarial attacks [[17](#bib.bib17), [18](#bib.bib18)], and for active learning [[19](#bib.bib19)]. Recently, such approaches have been extended to structured prediction tasks such as machine translation and speech recognition [[20](#bib.bib20), [21](#bib.bib21), [22](#bib.bib22), [23](#bib.bib23), [24](#bib.bib24)]. However, these approaches require large computational and memory budgets. Works using temperature scaling [[25](#bib.bib25), [26](#bib.bib26)] and other recent approaches in deterministic uncertainty estimation [[27](#bib.bib27), [28](#bib.bib28), [29](#bib.bib29), [30](#bib.bib30), [31](#bib.bib31)] aim to tackle this issue, but have only recently become comparable to ensemble methods [[30](#bib.bib30), [31](#bib.bib31)]. Prior Networks [[32](#bib.bib32), [33](#bib.bib33), [34](#bib.bib34)] — models which *emulate* the mechanics of an ensemble — have been proposed as a deterministic single model approach to uncertainty estimation which are competitive with ensembles. However, they require distributionally shifted training data, which may not be feasible in many applications. Prior Networks have also been used for *Ensemble Distribution Distillation* [[35](#bib.bib35), [34](#bib.bib34), [36](#bib.bib36)] — a distillation approach through which the predictive performance and uncertainty estimates of an ensemble are captured within a single Prior Network, reducing the inference cost to that of a single model.

While much work has been done on developing *methods*, limited work has focused on new datasets and benchmarks. In [[37](#bib.bib37), [38](#bib.bib38)], the authors introduced benchmarks for uncertainty quantification in Bayesian deep learning but only considered the image-based task of classifying diabetic retinopathy. Recently, a range of works by Hendrycks et al. [[39](#bib.bib39), [40](#bib.bib40), [41](#bib.bib41)] proposed a set of datasets based on ImageNet [[42](#bib.bib42)] for evaluating model robustness to various types of distributional shifts. These datasets — ImageNet C, A, R, and O — include synthetically added noise, natural adversarial attacks, renderings, and previously unseen classes of objects.111ImageNet has only “natural” images; thus, renderings represent a shift in texture, but not content. The release of WILDS, a collection of datasets containing real-world distributional shifts [[8](#bib.bib8)], similarly represents a significant step forward, but again mostly focuses on images. Finally, the MTNT dataset [[43](#bib.bib43)], which contains many examples of highly atypical usage of language, such as acronyms, profanity, emojis, slang, and code-switching, has been used at the Workshop on Machine Translation (WMT) robustness track. However, it has not been considered by the uncertainty community in the context of *detecting* distributional shift.

Unfortunately, with few exceptions, most work on uncertainty estimation and robustness has focused on developing new methods on small-scale tabular regression or image classification tasks, such as UCI, MNIST [[44](#bib.bib44)], Omniglot [[45](#bib.bib45)], SVHN [[46](#bib.bib46)], and CIFAR10/100 [[47](#bib.bib47)]. Few works have been evaluated on the ImageNet variations A, R, C, and O, or WILDS. However, even evaluation on these datasets is limited, as they mainly focus on image classification, and sometimes text. In contrast, many tasks of practical interest have different modalities, such as tabular data (in medicine and finance), audio, text, or sensor data. Furthermore, these tasks are not always classification; they often involve regression and discrete or continuous structured prediction. Given the current state of the field, we aim to draw the attention of the community to the evaluation of uncertainty estimation and robustness to distributional shift on a realistic set of large-scale tasks across a range of modalities. This is necessary to meaningfully evaluate the plethora of methods for uncertainty quantification and improved robustness, and to accelerate the development of this area and safe ML in general.

In this work, we propose the Shifts Dataset222Data and example code are available at <https://github.com/yandex-research/shifts> for evaluation of uncertainty estimates and robustness to distributional shift. This dataset consists of data taken directly from large-scale industrial sources and services where distributional shift is ubiquitous — settings as close to “in the wild” as possible. The dataset is composed of three parts, with each corresponding to a particular data modality: *tabular weather prediction* data provided by the Yandex Weather service; *machine translation* data taken from the WMT robustness track and mined from Reddit, and annotated in-house by Yandex Translate; and, self-driving car (SDC) data provided by Yandex SDG, for the task of *vehicle motion prediction*. All of these data modalities and tasks are affected by distributional shift and pose interesting challenges with respect to uncertainty estimation. This paper provides a detailed analysis of the data as well as baseline uncertainty estimation and robustness results using ensemble methods.

## 2 Evaluation Paradigm, Metrics, and Baselines

##### Paradigm

In most prior work, uncertainty estimation and robustness have been assessed separately. Robustness to distributional shift is usually assessed via metrics of predictive performance on a particular task — given two (or more) evaluation sets, where one is considered matched to the training data and the other(s) shifted, models which have a smaller degradation in performance on the shifted data are considered more robust. Uncertainty quality is often assessed via the ability to classify whether an example came from the “in-domain” dataset or a shifted dataset using uncertainty estimates. Here, performance is assessed via Area under a Receiver-Operator Curve (ROC-AUC %) or Precision-Recall curve (AUPR %). While these evaluation paradigms are meaningful, we believe that they are two halves of a common whole. Instead, we consider the following paradigm:

*As the degree of distributional shift increases, so does the likelihood that a model makes an error and the degree of this error. Models should yield uncertainty estimates which correlate with the degree of distributional shift, and therefore are indicative of the likelihood and the degree of the error.*

This paradigm is more general, as a model may be robust to certain examples of distributional shift and yield accurate, low uncertainty predictions. A model may also perform poorly and yield high estimates of uncertainty on underrepresented data matched to the training set. Thus, splitting a dataset into “in-domain” and “out-of-distribution” may not yield partitions on which a model strictly performs well or poorly, respectively. Instead, it is necessary to *jointly* assess robustness and uncertainty estimation, in order to see whether uncertainty estimates at the level of a single prediction correlate well with the likelihood or degree of error. Thus, we view the problems of robustness and uncertainty estimation as having *equal* importance — models should be robust, but where they are not, they should yield high estimates of uncertainty, which enables risk-mitigating actions to be taken (e.g., transferring control of a self-driving vehicle to a human operator).

We assume that at training or test time *we do not know a priori* about alternative domains and whether or how our data is shifted. This setup aims to emulate real-world deployments in which the variation of conditions is vast and one can never collect enough data to cover all situations. It is for this reason we view robustness and uncertainty as equally important — we assume that one can never be fully robust in all situations, and it is in these situations that high-quality uncertainty estimation is crucial. This is a strictly more challenging setting than one in which auxiliary information about the degree or nature of shift is available at training or test time (e.g., in WILDS [[8](#bib.bib8)]).

We have constructed the Shifts Dataset within the context of this paradigm. Specifically, the dataset is constructed with the following attributes. First, the annotations of distributional shift are meant to be used for analysis rather than model construction. Second, we have “canonically” partitioned the datasets such that the shifts are realistic but significant and to which it is challenging to be fully robust — this allows us to assess the quality of uncertainty estimates. However, the weather and motion prediction datasets *can* be repartitioned in alternative ways which are different from our canonical partitioning, such that alternative robustness paradigms can be evaluated.333Tools for partitioning and repartitioning are provided in our GitHub repository.

##### Assessment Metrics

We jointly assess robustness and uncertainty via *error-retention curves* [[12](#bib.bib12), [14](#bib.bib14)] and *F1-retention curves*. Given an error metric, such as MSE, error-retention curves trace the error over a dataset as a model’s predictions are replaced by ground-truth labels in order of decreasing uncertainty. F1-retention curves depict the F1 for predicting whether a model’s predictions are sufficiently good based on uncertainties (here we vary retention fraction, i.e., the fraction of data with the smallest uncertainty values that we classify as acceptable). Both assess the performance of a hybrid human-AI system, where a model can consult an oracle (human) for assistance in difficult situations. The area under this curve can be decreased (error retention) or increased (F1 retention) either by improving the predictive performance of the model, such that it has lower overall error, or by providing better estimates of uncertainty, such that more errorful predictions are rejected earlier. Thus, the area under the error (R-AUC) and F1 (F1-AUC) retention curves are metrics which jointly assess robustness to distributional shift and the quality of uncertainty estimates. We also quote F1 at 95% retention rate. These metrics, detailed in [Appendix A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), are used for all tasks in this paper.

##### Choice of Baselines

In this work we consider ensemble-based baselines. This was done for several reasons. First, ensemble-based approaches are a standard way to obtain *both* improved robustness versus single models *and* interpretable uncertainty estimates. Ensembles improve robustness because each model represents a functionally different explanation of the data. Thus, even if each individual model in an ensemble is subject to spurious correlations, the models will have different spurious correlations. When the models are combined, the effects of spurious correlations are cancelled out to a certain degree, improving generalization performance. Second, ensemble methods are easy to apply to any task of choice and require little adaptation.
Uncertainty estimates can be obtained from measures of ensemble diversity — if the predictions are diverse, then the ensemble members cannot agree on what the prediction should be and therefore are uncertain. Other than ensemble methods, there are few alternative approaches which are known to yield improved robustness *and* interpretable uncertainty estimates, can be easily applied to a broad range of large-scale tasks without significant adaptation, and do not require information about the nature of distributional shift at training or test time. We leave the exploration of these alternatives and the development of new ones to future work. We do not examine robust learning methods, such as IRM [[48](#bib.bib48), [8](#bib.bib8)], as they require domain annotations at training time and do not yield uncertainty estimates.

## 3 Tabular Weather Prediction

Uncertainty estimation and robustness are essential in applications like medical diagnostics and financial forecasting. In such applications, data is often represented in a heterogeneous tabular form. While it is challenging to obtain either a large medical or financial dataset, the Yandex Weather service has provided a large tabular Weather Prediction dataset that features a natural tendency for the data distribution to drift over time (concept drift [[49](#bib.bib49), [50](#bib.bib50)]). Furthermore, the locations are non-uniformly distributed around the globe based on population density, land coverage, and observation network development, which means that certain climate zones, like the Polar regions or the Sahara, are under-represented. We argue that this tabular Weather Prediction data represents similar challenges to the ones faced on financial and medical data, which is often combined from different hospitals/labs, consists of population-groups that are non-uniformly represented, and has a tendency to drift over time. Thus, the data we consider in this paper can be used as an appropriate benchmark for developing more robust models and uncertainty estimation methods for tabular data.

##### Dataset

The Shifts Weather Prediction dataset contains a scalar regression and a multi-class classification tasks: at a particular latitude, longitude, and timestamp, one must predict either the air temperature at two meters above the ground or the precipitation class, given targets and features derived from weather station measurements and weather forecast models. The data consists of 10 million 129-column entries: 123 meteorological features, 4 meta-data attributes (time, latitude, longitude and climate type) and 2 targets — temperature (target for regression task) and precipitation class (target for classification task). The full feature list is provided in Section [C.2](#A3.SS2.SSS0.Px3 "Features ‣ C.2 Detailed description of features and targets ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). It is important to note that the features are highly heterogeneous, i.e., they are of different types and scales. The full data is distributed uniformly between September 1stsuperscript1st1^{\text{st}}, 2018, and September 1stsuperscript1st1^{\text{st}}, 2019, with samples across all climate types. This data is used by Yandex for real-time weather forecasts and represents a real industrial application.

To provide a standard benchmark that contains both in-domain and shifted data, we use a particular “canonical partitioning”444Alternative partitionings can be made from the full data, but we use the canonical partitioning throughout this work, and also for the Shifts Challenge: <http://research.yandex.com/shifts> of the full dataset into training, development (dev), and evaluation (eval) datasets. The training, in-domain dev (dev\_in) and in-domain eval (eval\_in) data consist of measurements made from September 2018 till April 8thsuperscript8th8^{\text{th}}, 2019 for climate types Tropical, Dry, and Mild Temperate. The shifted dev (dev\_out) data consists of measurements made from 8thsuperscript8th8^{\text{th}} July till 1stsuperscript1st1^{\text{st}} September 2019 for the climate type Snow. 50K data points are sub-sampled for the climate type Snow within this time range to construct dev\_out. The shifted eval data is further shifted than the out-of-domain development data; measurements are taken from 14thsuperscript14th14^{\text{th}} May till 8thsuperscript8th8^{\text{th}} July 2019, which is more distant in terms of the time of the year from the in-domain data compared to the out-of-domain development data. The climate types are restricted to Snow and Polar. Further details are provided in Appendix [C.1](#A3.SS1 "C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Details on use and support plan are in Appendix [B](#A2 "Appendix B Shifts Dataset General Datasheet ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Baselines

To build baseline models for the temperature prediction and precipitation classification tasks, we use the open-source CatBoost gradient boosting library that is known to achieve state-of-the-art results on tabular datasets [[51](#bib.bib51)]. We use an ensemble-based approach to uncertainty estimation for GBDT models [[52](#bib.bib52)]. For each task, an ensemble of ten models is trained on the training data with different random seeds. For regression, the models predict the mean and variance of the normal distribution by optimizing the negative log-likelihood. For classification, the models predict a probability distribution over precipitation classes. Training details are provided in Appendix [C.4](#A3.SS4 "C.4 Training details ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Additional ensemble-based baselines and results are provided in Appendix [C.5](#A3.SS5 "C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

We first compare the predictive performance of ensembles and single models; the results are shown in Table [1](#S3.T1 "Table 1 ‣ Baselines ‣ 3 Tabular Weather Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Firstly, we observe that all models perform worse on shifted data than on in-domain data. For regression, we observe that the RMSE of the ensemble (on the eval set) is about two degrees Celsius. Note that ensembling allows us to reduce RMSE by about 0.16​°0.16°0.16\degree compared to a single model. Similarly, ensembling reduces the MAE by approximately 0.12​°0.12°0.12\degree. For classification, ensembling boosts the accuracy by about 2% and macro-averaged F1 by about 1%. Note that the classification task is unbalanced (see Appendix [C.1](#A3.SS1 "C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for details), so for better interpretability, we also report the accuracy and Macro-F1 of the classifier always predicting the majority class.

Table 1: Predictive performance for Weather Prediction. Mean ±σplus-or-minus𝜎\pm\ \sigma is quoted for the single models.

|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Regression | | | | Classification | | | | | |
| Data | RMSE ↓↓\downarrow | | MAE ↓↓\downarrow | | Accuracy (%) ↑↑\uparrow | | | Macro F1 (%) ↑↑\uparrow | | |
| Single | Ens | Single | Ens | Maj. | Single | Ens | Maj. | Single | Ens |
| dev-in | 1.59±0.00subscript1.59plus-or-minus0.001.59\_{\pm 0.00} | 1.51 | 1.18±0.00subscript1.18plus-or-minus0.001.18\_{\pm 0.00} | 1.11 | 37.9 | 67.0±0.075subscript67.0plus-or-minus0.07567.0\_{\pm 0.075} | 68.5 | 17.2 | 42.2±0.01subscript42.2plus-or-minus0.0142.2\_{\pm 0.01} | 42.3 |
| dev-out | 2.30±0.01subscript2.30plus-or-minus0.012.30\_{\pm 0.01} | 2.12 | 1.75±0.01subscript1.75plus-or-minus0.011.75\_{\pm 0.01} | 1.61 | 35.7 | 47.5±0.249subscript47.5plus-or-minus0.24947.5\_{\pm 0.249} | 50.3 | 19.4 | 20.2±0.01subscript20.2plus-or-minus0.0120.2\_{\pm 0.01} | 21.3 |
| dev | 1.98±0.01subscript1.98plus-or-minus0.011.98\_{\pm 0.01} | 1.84 | 1.47±0.01subscript1.47plus-or-minus0.011.47\_{\pm 0.01} | 1.36 | 36.8 | 57.2±0.117subscript57.2plus-or-minus0.11757.2\_{\pm 0.117} | 59.4 | 17.2 | 36.8±0.01subscript36.8plus-or-minus0.0136.8\_{\pm 0.01} | 37.2 |
| eval-in | 1.60±0.00subscript1.60plus-or-minus0.001.60\_{\pm 0.00} | 1.52 | 1.19±0.00subscript1.19plus-or-minus0.001.19\_{\pm 0.00} | 1.11 | 37.9 | 66.7±0.060subscript66.7plus-or-minus0.06066.7\_{\pm 0.060} | 68.2 | 17.2 | 42.9±0.00subscript42.9plus-or-minus0.0042.9\_{\pm 0.00} | 44.1 |
| eval-out | 2.60±0.03subscript2.60plus-or-minus0.032.60\_{\pm 0.03} | 2.37 | 1.91±0.01subscript1.91plus-or-minus0.011.91\_{\pm 0.01} | 1.75 | 30.0 | 44.5±0.184subscript44.5plus-or-minus0.18444.5\_{\pm 0.184} | 46.7 | 17.4 | 21.5±0.00subscript21.5plus-or-minus0.0021.5\_{\pm 0.00} | 22.2 |
| eval | 2.16±0.01subscript2.16plus-or-minus0.012.16\_{\pm 0.01} | 2.00 | 1.56±0.01subscript1.56plus-or-minus0.011.56\_{\pm 0.01} | 1.44 | 33.9 | 55.5±0.090subscript55.5plus-or-minus0.09055.5\_{\pm 0.090} | 57.3 | 17.4 | 34.4±0.01subscript34.4plus-or-minus0.0134.4\_{\pm 0.01} | 35.5 |

We jointly evaluate the robustness and uncertainty estimates for ensembles and single models. For the regression task, we use the predicted variance as the uncertainty measure of a single model. For ensembles, we use the total variance (tvar) that is the sum of the variance of the predicted mean and the mean of the predicted variance [[11](#bib.bib11), [12](#bib.bib12), [52](#bib.bib52)]. For the classification task, we use the entropy of the prediction as the uncertainty measure of a single model. For ensembles, we use the (negated) confidence. We measure the area under the error-retention and F1-retention curves as described in Appendices [A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") and [C.3](#A3.SS3 "C.3 Metrics ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). These two performance metrics are denoted as R-AUC and F1-AUC, respectively. A good uncertainty measure is expected to achieve low R-AUC and high F1-AUC. Additionally, we report the F1 score at a retention rate of 95% of the most certain samples (F1@95%). All these measures jointly assess the predictive performance and uncertainty quality.

Table 2: Retention performance for Weather Prediction. Mean ±σplus-or-minus𝜎\pm\ \sigma is quoted for the single models.

|  | Data | R-AUC ↓↓\downarrow | | F1-AUC (%) ↑↑\uparrow | | F1@959595% ↑↑\uparrow | |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Single | Ens | Single | Ens | Single | Ens |
| Regression | dev | 1.894±0.017subscript1.894plus-or-minus0.0171.894\_{\pm 0.017} | 1.227 | 44.35±0.2subscript44.35plus-or-minus0.244.35\_{\pm 0.2} | 52.20 | 62.72±0.1subscript62.72plus-or-minus0.162.72\_{\pm 0.1} | 65.83 |
| eval | 2.320±0.063subscript2.320plus-or-minus0.0632.320\_{\pm 0.063} | 1.335 | 43.41±0.1subscript43.41plus-or-minus0.143.41\_{\pm 0.1} | 52.36 | 61.89±0.1subscript61.89plus-or-minus0.161.89\_{\pm 0.1} | 64.72 |
| Classification | dev | 0.1666±0.001subscript0.1666plus-or-minus0.0010.1666\_{\pm 0.001} | 0.1522 | 57.72±0.1subscript57.72plus-or-minus0.157.72\_{\pm 0.1} | 59.07 | 73.04±0.1subscript73.04plus-or-minus0.173.04\_{\pm 0.1} | 74.86 |
| eval | 0.1799±0.001subscript0.1799plus-or-minus0.0010.1799\_{\pm 0.001} | 0.1640 | 56.25±0.1subscript56.25plus-or-minus0.156.25\_{\pm 0.1} | 58.22 | 71.56±0.1subscript71.56plus-or-minus0.171.56\_{\pm 0.1} | 73.17 |

The results are shown in Table [2](#S3.T2 "Table 2 ‣ Baselines ‣ 3 Tabular Weather Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Here, as expected, ensembles significantly outperform single models. This observation is consistent over all considered evaluation measures. The associated retention curves are provided in Figure [1](#S3.F1 "Figure 1 ‣ Baselines ‣ 3 Tabular Weather Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for eval and Figure [11](#A3.F11 "Figure 11 ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") in [Appendix C](#A3 "Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for dev.

Finally, we conduct a comparison of different uncertainty measures. For this, we measure F1-AUC discussed above and ROC-AUC that evaluates uncertainty-based out-of-distribution (OOD) data detection. The results are shown in Table [3](#S3.T3 "Table 3 ‣ Baselines ‣ 3 Tabular Weather Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). In this experiment, we do not evaluate single models. For regression, we consider the following uncertainty measures: total variance (tvar) discussed above that is a measure of *total uncertainty*, variance of the mean predictions across the ensemble models (varm) and the expected pairwise KL-divergence (EPKL) that are measures of *knowledge uncertainty*. The results show that uncertainty measures that capture knowledge uncertainty perform best at OOD detection, as suggested by the high ROC-AUC values, while the measure of total uncertainty performs best for detecting errors (F1-AUC). Thus, as expected, the choice of a metric to use depends heavily on the task. Among measures of knowledge uncertainty, EPKL has better performance. For classification, the measures of total uncertainty are the negative confidence (Conf) and the entropy of the average prediction (Entropy). The measures of knowledge uncertainty are mutual information (MI), EPKL, and reverse mutual information (RMI). Similar to regression, uncertainty measures that capture knowledge uncertainty are better in terms of ROC-AUC. Among them, reverse mutual information performs best. The measures of total uncertainty are better for F1-AUC, and the best results are achieved with negative confidence.

Table 3: Comparing uncertainty measures of CatBoost ensembles for Weather Prediction.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Data |  | Regression | | | Classification | | | | |
|  | Total Unc. | Knowledge Unc. | | Total Unc. | | Knowledge Unc. | | |
|  | tvar | varm | EPKL | Conf | Entropy | MI | EPKL | RMI |
| dev | F1-AUC (%) ↑↑\uparrow | 52.20 | 50.12 | 50.51 | 59.07 | 58.86 | 57.72 | 57.69 | 57.66 |
| ROC-AUC (%) ↑↑\uparrow | 62.96 | 82.31 | 85.29 | 63.98 | 65.00 | 83.75 | 83.96 | 84.12 |
| eval | F1-AUC (%) ↑↑\uparrow | 52.36 | 49.81 | 50.40 | 58.22 | 57.89 | 56.99 | 56.96 | 56.93 |
| ROC-AUC (%) ↑↑\uparrow | 65.99 | 78.32 | 79.90 | 66.20 | 66.76 | 83.44 | 83.59 | 83.68 |

!(/html/2107.07455/assets/figures/weather_single_retention_eval_mse.png)

(a) Regression, MSE.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1.png)

(b) Regression, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_acc_class.png)

(c) Classification, error.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1_class.png)

(d) Classification, F1.

Figure 1: Retention curves with CatBoost on eval for the Weather Prediction dataset.

## 4 Machine Translation

As part of the Shifts Dataset we examine the task of machine translation for the text modality. Translation services, such as Google Translate or Yandex Translate, often encounter atypical and unusual use of language in their translation queries. This typically includes slang, profanities, poor grammar, orthography and punctuation, as well as emojis. This poses a challenge to modern translation systems, which are typically trained on corpora with a more “standard” use of language. Therefore, it is important for models to both be robust to atypical language use to provide high-quality translations, as well as to indicate when they are unable to provide a quality translation.

Translation is inherently a *structured prediction* task, as there are dependencies between the tokens in the output sequence. Often we must make assumptions about the form of these dependencies; for example, most modern translation systems are left-to-right autoregressive. However, we could consider conditionally independent predictions or other factorization orders. The nature of these assumptions makes it challenging to obtain a theoretically sound measure of uncertainty. Only recently has work been done on developing principled uncertainty measures for structured prediction [[21](#bib.bib21), [22](#bib.bib22), [23](#bib.bib23), [24](#bib.bib24), [53](#bib.bib53)]. Nevertheless, this remains an unsolved task and a fruitful area for research.

##### Dataset

The dataset contains training, development (dev) and evaluation (eval) data, where each set consists of pairs of source and target sentences in English and Russian, respectively. As most production Neural Machine Translation (NMT) systems are built using a variety of general purpose corpora, we use the freely available WMT‘20 En-Ru corpus as training data. This dataset primarily focuses on parliamentary and news data that is, for the most part, grammatically and orthographically correct with formal language use. The dev and eval datasets consist of an “in-domain” partition matched to the training data, and an “out-of-distribution” or shifted partition, which contains examples of atypical language usage. The in-domain dev and eval sets are Newstest‘19 En-Ru and a newly collected news corpus from GlobalVoices [[54](#bib.bib54)], respectively. For the shifted development data we use the Reddit corpus prepared for the WMT‘19 robustness challenge [[43](#bib.bib43)]. This data contains examples of slang, acronyms, lack of punctuation, poor orthography, concatenations, profanity, and poor grammar, among other forms of atypical language usage. This data is representative of the types of inputs that machine translation services find challenging. As Russian target annotations are not available, we pass the data through a two-stage process, where orthographic, grammatical, and punctuation mistakes are corrected, and the source-side English sentences are translated into Russian by expert in-house Yandex translators. The development set is constructed from the same 1400-sentence test-set used for the WMT‘19 robustness challenge. For the evaluation set we use the open-source MTNT crawler which connects to the Reddit API to collect a further set of 3,000 English sentences from Reddit, which is similarly corrected and translated. The shifted dev and eval data are also annotated with 7 non-exclusive anomaly flags. Details on pre-processing, annotations and licenses are available in Appendix [D.1](#A4.SS1 "D.1 Dataset Description ‣ Appendix D Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Details on use and support plan are in Appendix [B](#A2 "Appendix B Shifts Dataset General Datasheet ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Metrics

To evaluate the performance of our models we will consider corpus-level BLEU [[55](#bib.bib55)] and sentence-level GLEU [[56](#bib.bib56), [57](#bib.bib57), [58](#bib.bib58)]. As machine translation is a multi-modal task and translation systems often yield multiple translation hypothesis we will consider two GLEU-based metrics for evaluating translation quality. First is the *expected GLEU* or eGLEU across all translation hypotheses, where each hypothesis is weighted by a *confidence score*, and confidences across all hypotheses sum to one. Second is the maximum GLEU maxGLEU across all hypotheses in the beam. Details of these metrics can be found in Appendix [D.2](#A4.SS2 "D.2 Metrics ‣ Appendix D Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). These metrics are then used to compute the error- and F1-retention curves which jointly assess uncertainty and robustness, as discussed in Appendix [A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Baselines

In this work we considered an ensemble baseline based on [[24](#bib.bib24)]. Here, we use an ensemble of 3 Transformer-Big [[5](#bib.bib5)] models trained on the WMT‘20 En-Ru corpus. Models were trained using a fork of FairSeq [[59](#bib.bib59)] with a large-batch training set. Beam-Search decoding with a beam-width of 5 is used to obtain translation hypotheses. Hypotheses confidence weights are obtained by exponentiating the negative log-likelihood of each hypothesis and then normalizing across all hypotheses in the beam. Individual models in the ensemble are used as a single-model baseline.

Table [4](#S4.T4 "Table 4 ‣ Baselines ‣ 4 Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") presents the predictive performance on the dev and eval sets as well as on their in-domain and shifted subsets. There is a performance difference of nearly 10 BLEU and GLEU points between the in-domain news and shifted Reddit data, which shows the degradation in quality due to atypical language usage. The ensemble is able to outperform the individual models, which is expected. These results also show that BLEU correlates quite well with eGLEU. maxGLEU shows that significantly better performance is obtainable if we were better at ranking the hypotheses in the beam.

Table 4: Predictive performance for Machine Translation. Mean ±σplus-or-minus𝜎\pm\ \sigma is quoted for the single models.

| Data | BLEU ↑↑\uparrow | | eGLEU ↑↑\uparrow | | maxGLEU ↑↑\uparrow | |
| --- | --- | --- | --- | --- | --- | --- |
| Single | Ens | Single | Ens | Single | Ens |
| dev-in | 32.04±0.23 | 32.73 | 34.45±0.10 | 35.09 | 41.08±0.09 | 42.00 |
| dev-out | 20.65±0.16 | 21.06 | 22.66±0.07 | 23.00 | 28.28±0.19 | 28.63 |
| dev | 28.89±0.20 | 29.52 | 29.67±0.09 | 30.19 | 35.89±0.12 | 36.58 |
| eval-in | 29.52±0.21 | 30.08 | 30.39±0.10 | 30.88 | 36.19 ±0.19 | 36.82 |
| eval-out | 21.00 ±0.12 | 21.54 | 23.19±0.07 | 23.60 | 29.35±0.11 | 29.88 |
| eval | 26.39±0.17 | 26.92 | 26.76±0.06 | 27.20 | 32.74±0.14 | 33.31 |

Having evaluated the baselines’ predictive performance, we now jointly assess their uncertainty and robustness using the area under the error-retention curve (R-AUC), area under the F1-retention curve (F1-AUC) and F1 at 95% retention, as detailed in Appendices [A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") and [D.2](#A4.SS2 "D.2 Metrics ‣ Appendix D Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Additionally, we evaluate in terms of % ROC-AUC whether it is possible to discriminate between the in-domain data and the shifted data based on uncertainty estimates by the models. As the measure of uncertainty we use the negative log-likelihood, averaged across all 5 hypotheses. In the case of individual models, this is a measure of *data* or *aleatoric* uncertainty, and in the case of the ensemble, it is a measure of *total uncertainty* [[24](#bib.bib24)]. Here, the ensemble consistently outperforms the single-model baseline.

Table 5: Uncertainty and robustness for Machine Translation. Mean ±σplus-or-minus𝜎\pm\ \sigma is quoted for single models.

| Data | R-AUC ↓↓\downarrow | | F1-AUC ↑↑\uparrow | | F1@959595% ↑↑\uparrow | | ROC-AUC (%) ↑↑\uparrow | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single | Ens | Single | Ens | Single | Ens | Single | Ens |
| dev | 33.22±0.48 | 32.87 | 0.43±0.00 | 0.44 | 0.42±0.01 | 0.43 | 68.90±0.28 | 69.30 |
| eval | 34.80±0.06 | 34.57 | 0.37 ±0.07 | 0.38 | 0.34 ±0.03 | 0.36 | 79.18 ±0.63 | 80.10 |

!(/html/2107.07455/assets/figures/nmt_error_retention_eval.png)

(a) Error

!(/html/2107.07455/assets/figures/nmt_F1_retention_eval.png)

(b) F1

Figure 2: Retention curves using eGLEU on eval data.

## 5 Vehicle Motion Prediction

We present the Shifts Vehicle Motion Prediction dataset to examine the implications of distributional shift in self-driving vehicles. The area of autonomous vehicle (AV) technology is highly relevant for uncertainty and robustness research, as the safety requirements and the risks associated with any errors are high. Furthermore, distributional shift is ubiquitous in the autonomous driving domain. During technology development, most self-driving companies concentrate their fleet in a limited number of locations and routes due to the large cost of operating in a new location. Therefore, fleets often face distributional shift when they begin operation in new locations. It is thus important to transfer as much knowledge as possible from the old locations to new ones. It is also critical for a planning model to recognize when this transferred knowledge is insufficient upon encountering unfamiliar data, which could risk unpredictable and unsafe behavior.555A case of knowledge, or epistemic uncertainty [[11](#bib.bib11), [12](#bib.bib12)]. Uncertainty quantification therefore has potentially life-critical application in this domain. For example, when the model’s uncertainty is high, the vehicle can exercise extra caution or request assistance from a remote operator.

Motion prediction is among the most important problems in the autonomous driving domain and has recently drawn significant attention from both academia and industry [[60](#bib.bib60), [61](#bib.bib61), [62](#bib.bib62), [63](#bib.bib63), [64](#bib.bib64), [65](#bib.bib65), [66](#bib.bib66), [67](#bib.bib67), [68](#bib.bib68), [69](#bib.bib69), [70](#bib.bib70), [71](#bib.bib71)]. It involves predicting the distribution over possible future states of agents around the self-driving car at a number of moments in time. A model of possible futures is needed because a self-driving vehicle needs a certain amount of time to change its speed, and sudden changes may be uncomfortable or even dangerous for its passengers. Therefore, in order to ensure a safe and comfortable ride, the motion planning module of a self-driving vehicle must reason about where other agents might end up in a few seconds to avoid planning a potential collision. This problem is complicated by the fact that *the future is inherently uncertain*. For example, we cannot know the high-level navigational goals of other agents, or even their low-level tendency to turn right or left at a T-junction if they fail to indicate one way or another.666A case of data, or aleatoric uncertainty. In order for the planning module to make the right decision, this uncertainty must be precisely quantified. Finally, motion prediction is also interesting because the predictions are both *structured and continuous*. This poses further challenges in uncertainty estimation. Recently, ensemble-based uncertainty estimation for the related task of autonomous vehicle *planning* was examined [[72](#bib.bib72)], where a variance-based measure was proposed. However, there is still much potential for further development of informative measures of uncertainty in continuous structured prediction tasks such as motion prediction.

##### Dataset

The dataset for the Vehicle Motion Prediction task was collected by the Yandex Self-Driving Group (SDG) fleet and is the largest vehicle motion prediction dataset released to date, containing 600,000 scenes. These scenes span six locations, three seasons, three times of day, and four weather conditions.
Each scene includes information about the state of dynamic objects and an HD map. Each scene is 10 seconds long and is divided into 5 seconds of context features and 5 seconds of ground truth targets for prediction, separated by the time T=0𝑇0T=0. The goal is to predict the movement trajectory of vehicles at time T∈(0,5]𝑇05T\in\left(0,5\right] based on the information available for time T∈[−5,0]𝑇50T\in\left[-5,0\right]. The data contains training, development (dev) and evaluation (eval) sets. In order to study the effects of distributional shift, we partition the data such that the dev and eval sets have *in-domain* partitions which match the location and precipitation type of the training set, and *out-of-domain* or *shifted* partitions which do not match the training data along one or more of those axes. As in the other Shifts tasks, we define a canonical partitioning which is used throughout benchmarking.777This partitioning is also the one used in the Shifts Challenge: <http://research.yandex.com/shifts> The training set and in-domain partition of the dev and eval sets are taken from Moscow. Distributionally shifted dev data is taken from Skolkovo, Modiin, and Innopolis. Distributionally shifted eval data is taken from Tel Aviv and Ann Arbor. We also remove all cases of precipitation from the in-domain sets, while distributionally shifted datasets include precipitation. A full description of the dataset is available in Appendix [E](#A5 "Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), the support plan is detailed in Appendix [B](#A2 "Appendix B Shifts Dataset General Datasheet ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Metrics

Here we consider five different performance metrics — minimum Average Displacement Error (minADE), minimum Final Displacement Error (minFDE), confidence-weighed ADE and FDE, and corrected Negative Log-Likelihood (cNLL). cNLL is a new metric we introduce that is particilarly well-suited for assessing how models handle multi-modal situations. The minimum or weighting is done across up to 5 trajectories predicted by the baseline models. See [Section E.3](#A5.SS3 "E.3 Performance Metrics ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for detailed explanations of the metrics.

##### Baselines

We consider two variants of Robust Imitative Planning (RIP) [[72](#bib.bib72)] as baselines. We use an ensemble of probabilistic models to stochastically generate multiple predictions for a given prediction request. Predictions are aggregated across ensemble members via a model averaging (MA) approach. We consider a simple RNN-based behavioral cloning network (RIP-BC) [[73](#bib.bib73)] and autoregressive flow–based Deep Imitative Model (RIP-DIM) [[74](#bib.bib74)] as backbone models. We adapt RIP to produce uncertainty estimates at two levels of granularity: per-trajectory and per–prediction request. Finally, we vary the number of ensemble members K∈{1,3,5}𝐾135K\in\{1,3,5\} and the uncertainty estimation method between Deep Ensembles [[14](#bib.bib14)] and Dropout Ensembles [[13](#bib.bib13), [75](#bib.bib75)]. See Appendix [E](#A5 "Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for details on RIP, uncertainty estimation methods, backbone models, experimental setup, and full results. Additional results using Dropout Ensembles are provided in Appendix [E.5](#A5.SS5 "E.5 Additional Results ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

Table 6: Predictive performance of BC & DIM RIP on in-domain, shifted, and full dev & eval data.

| Dataset | Model | cNLL ↓↓\downarrow | | | minADE ↓↓\downarrow | | | weightedADE ↓↓\downarrow | | | minFDE ↓↓\downarrow | | | weightedFDE ↓↓\downarrow | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In | Shifted | Full | In | Shifted | Full | In | Shifted | Full | In | Shifted | Full | In | Shifted | Full |
| Dev | BC, MA, K=1 | 59.64 | 98.54 | 64.29 | 0.818 | 0.960 | 0.835 | 1.088 | 1.245 | 1.107 | 1.718 | 2.113 | 1.765 | 2.368 | 2.777 | 2.417 |
| BC, MA, K=5 | 56.86 | 91.54 | 61.01 | 0.765 | 0.887 | 0.779 | 1.012 | 1.133 | 1.026 | 1.617 | 1.976 | 1.660 | 2.210 | 2.551 | 2.251 |
| DIM, MA, K=1 | 50.66 | 73.00 | 53.34 | 0.750 | 0.818 | 0.758 | 1.523 | 1.583 | 1.530 | 1.497 | 1.720 | 1.524 | 3.472 | 3.639 | 3.492 |
| DIM, MA, K=5 | 50.85 | 72.45 | 53.43 | 0.719 | 0.786 | 0.727 | 1.399 | 1.469 | 1.408 | 1.482 | 1.698 | 1.508 | 3.202 | 3.393 | 3.225 |
| Eval | BC, MA, K=1 | 60.20 | 98.82 | 67.93 | 0.829 | 1.084 | 0.880 | 1.104 | 1.407 | 1.164 | 1.733 | 2.420 | 1.870 | 2.394 | 3.197 | 2.555 |
| BC, MA, K=5 | 57.75 | 95.00 | 65.20 | 0.777 | 1.014 | 0.824 | 1.028 | 1.299 | 1.082 | 1.636 | 2.278 | 1.765 | 2.238 | 2.957 | 2.382 |
| DIM, MA, K=1 | 50.50 | 76.00 | 55.60 | 0.759 | 0.942 | 0.796 | 1.551 | 1.883 | 1.618 | 1.511 | 1.983 | 1.605 | 3.536 | 4.376 | 3.704 |
| DIM, MA, K=5 | 51.19 | 78.85 | 56.73 | 0.728 | 0.918 | 0.766 | 1.424 | 1.754 | 1.490 | 1.493 | 2.000 | 1.595 | 3.256 | 4.093 | 3.424 |

Predictive performance results for the RIP variants are presented in [Table 6](#S5.T6 "In Baselines ‣ 5 Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Performance is assessed on the in-distribution (In), distributionally shifted (Shifted), and combined (Full) dev and eval datasets. We observe that across all model configurations, performance on the shifted data is worse than that on the in-distribution data. We also observe that RIP-BC consistently outperforms RIP-DIM on the per-trajectory confidence weighted metrics (weightedADE and weightedFDE), and RIP (DIM) outperforms RIP (BC) on minADE and minFDE. This result might occur if DIM has higher predictive variance. In such a case, DIM might be more effective in modeling multimodality, and therefore would tend to produce at least one high accuracy trajectory on more scenes, improving performance on min aggregation metrics. This is supported by DIM models yielding the best cNLL, which is a metric particularly sensitive to correct treatment of multi-modal situations. In contrast, for “obvious” scenes, DIM might then produce unnecessarily complicated trajectories which would be reflected in poor performance on weightedADE.

Table 7: Uncertainty and robustness performance for motion prediction. The error metric for computing the area under the F1 curve (F1-AUC) and F1 at 95% retention rate (F1@95%) is cNLL.

| Data | Ensemble | R-AUC cNLL ↓↓\downarrow | | R-AUC weightedADE ↓↓\downarrow | | F1-AUC (%) ↑↑\uparrow | | F1@959595% ↑↑\uparrow | | ROC-AUC (%) ↑↑\uparrow | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Size (K) | RIP-BC | RIP-DIM | RIP-BC | RIP-DIM | RIP-BC | RIP-DIM | RIP-BC | RIP-DIM | RIP-BC | RIP-DIM |
| Dev | 1 | 11.22 | 12.86 | 0.268 | 0.419 | 65.1 | 63.8 | 89.3 | 87.4 | 51.0 | 51.8 |
| 5 | 9.08 | 13.24 | 0.236 | 0.376 | 65.2 | 63.7 | 90.6 | 89.7 | 49.2 | 51.4 |
| Eval | 1 | 12.91 | 14.32 | 0.293 | 0.458 | 65.0 | 63.6 | 88.4 | 86.3 | 52.8 | 51.8 |
| 5 | 10.57 | 15.16 | 0.258 | 0.411 | 65.1 | 63.5 | 89.7 | 88.9 | 52.1 | 50.9 |

[Table 7](#S5.T7 "In Baselines ‣ 5 Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") presents a joint evaluation of the uncertainty quantification and robustness of our baselines. We compute R-AUC with respect to cNLL and weightedADE, and the F1-AUC and F1@​95%@percent95@95\% metrics with respect to the cNLL metric, as detailed in Appendices [A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") and  [E.3](#A5.SS3 "E.3 Performance Metrics ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). We observe that an ensemble of RIP-BC models outperforms RIP-DIM on these metrics. These results strongly suggest that RIP-BC has more informative uncertainty estimates than RIP-DIM, because RIP-BC achieves better R-AUC cNLL despite having greater overall error in terms of cNLL (in addition to minADE and minFDE). Figure [3](#S5.F3 "Figure 3 ‣ Baselines ‣ 5 Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") depicts, for cNLL, error- and F1-retention curves on the full eval dataset which reflect the trends observed in Table [7](#S5.T7 "Table 7 ‣ Baselines ‣ 5 Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Additionally, we find that across model configurations the per–prediction request uncertainty scores do not perform particularly well in detecting distribution shift (ROC-AUC). This may occur due to significant data uncertainty in all cases. Future work on detecting distributional shift on this dataset could, for example, inspect the distribution of log-likelihood scores on the in-distribution and shifted partitions in order to devise a metric for this task, aside from the uncertainty scores U𝑈U used for the retention analysis.

!(/html/2107.07455/assets/x1.png)

(a) cNLL error-retention.

!(/html/2107.07455/assets/x2.png)

(b) cNLL F1-retention.

Figure 3: Retention curves for Vehicle Motion Prediction on full eval data.

## 6 Conclusion

In this paper, we proposed the Shifts Dataset: a large, standardized dataset for evaluation of uncertainty estimates and robustness to realistic, curated distributional shift. The dataset — sourced from industrial services — is composed of three tasks, with each corresponding to a particular data modality: *tabular weather prediction*, *machine translation*, and self-driving car (SDC) *vehicle motion prediction*. This paper describes this data and provides baseline results using ensemble methods. Given the current state of the field, where most methods are developed on small-scale classification tasks, we aim to draw the attention of the community to the evaluation of uncertainty estimation and robustness to distributional shift on large-scale industrial tasks across multiple modalities. We believe this work is a necessary step towards meaningful evaluation of uncertainty quantification methods, and hope for it to accelerate the development of this area and safe ML in general.

## Acknowledgments and Disclosure of Funding

We would like to thank Yandex for providing the data and resources necessary in benchmark creation. We thank Intel and the Turing Institute for funding the work of the OATML Group on this project. Finally, we thank Cambridge University Press and Cambridge Assessment for funding the work of the CUED Speech Group.

## References

* [1]

  Karen Simonyan and Andrew Zisserman,
  “Very Deep Convolutional Networks for Large-Scale Image
  Recognition,”
  in Proc. International Conference on Learning Representations
  (ICLR), 2015.
* [2]

  Tomas Mikolov et al.,
  “Linguistic Regularities in Continuous Space Word
  Representations,”
  in Proc. NAACL-HLT, 2013.
* [3]

  Tomas Mikolov, Martin Karafiát, Lukás Burget, Jan Cernocký,
  and Sanjeev Khudanpur,
  “Recurrent Neural Network Based Language Model,”
  in Proc. INTERSPEECH, 2010.
* [4]

  Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio,
  “Neural machine translation by jointly learning to align and
  translate,”
  in Proc. International Conference on Learning Representations
  (ICLR), 2015.
* [5]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin,
  “Attention is all you need,”
  in Advances in neural information processing systems, 2017, pp.
  5998–6008.
* [6]

  Geoffrey Hinton, Li Deng, Dong Yu, George Dahl, Abdel rahman Mohamed, Navdeep
  Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara Sainath, and
  Brian Kingsbury,
  “Deep neural networks for acoustic modeling in speech recognition,”
  Signal Processing Magazine, 2012.
* [7]

  Joaquin Quiñonero-Candela,
  Dataset Shift in Machine Learning,
  The MIT Press, 2009.
* [8]

  Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang,
  Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips,
  Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine,
  Chelsea Finn, and Percy Liang,
  “Wilds: A benchmark of in-the-wild distribution shifts,” 2020.
* [9]

  Zachary Nado, Neil Band, Mark Collier, Josip Djolonga, Michael W. Dusenberry,
  Sebastian Farquhar, Angelos Filos, Marton Havasi, Rodolphe Jenatton, Ghassen
  Jerfel, Jeremiah Liu, Zelda Mariet, Jeremy Nixon, Shreyas Padhy, Jie Ren, Tim
  G. J. Rudner, Yeming Wen, Florian Wenzel, Kevin Murphy, D. Sculley, Balaji
  Lakshminarayanan, Jasper Snoek, Yarin Gal, and Dustin Tran,
  “Uncertainty baselines: Benchmarks for uncertainty & robustness in
  deep learning,” 2021.
* [10]

  Dario Amodei, Chris Olah, Jacob Steinhardt, Paul F. Christiano, John Schulman,
  and Dan Mané,
  “Concrete problems in AI safety,”
  <http://arxiv.org/abs/1606.06565>, 2016,
  arXiv: 1606.06565.
* [11]

  Yarin Gal,
  Uncertainty in Deep Learning,
  Ph.D. thesis, University of Cambridge, 2016.
* [12]

  Andrey Malinin,
  Uncertainty Estimation in Deep Learning with application to
  Spoken Language Assessment,
  Ph.D. thesis, University of Cambridge, 2019.
* [13]

  Yarin Gal and Zoubin Ghahramani,
  “Dropout as a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning,”
  in Proc. 33rd International Conference on Machine Learning
  (ICML-16), 2016.
* [14]

  B. Lakshminarayanan, A. Pritzel, and C. Blundell,
  “Simple and Scalable Predictive Uncertainty Estimation using Deep
  Ensembles,”
  in Proc. Conference on Neural Information Processing Systems
  (NIPS), 2017.
* [15]

  Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, and Dmitry Vetrov,
  “Pitfalls of in-domain uncertainty estimation and ensembling in deep
  learning,”
  in International Conference on Learning Representations, 2020.
* [16]

  Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian
  Nowozin, Joshua V Dillon, Balaji Lakshminarayanan, and Jasper Snoek,
  “Can you trust your model’s uncertainty? evaluating predictive
  uncertainty under dataset shift,”
  Advances in Neural Information Processing Systems, 2019.
* [17]

  Nicholas Carlini and David A. Wagner,
  “Adversarial examples are not easily detected: Bypassing ten
  detection methods,”
  CoRR, 2017.
* [18]

  L. Smith and Y. Gal,
  “Understanding Measures of Uncertainty for Adversarial Example
  Detection,”
  in UAI, 2018.
* [19]

  Andreas Kirsch, Joost Van Amersfoort, and Yarin Gal,
  “Batchbald: Efficient and diverse batch acquisition for deep
  bayesian active learning,”
  Advances in neural information processing systems, vol. 32, pp.
  7026–7037, 2019.
* [20]

  Tim Z Xiao, Aidan N Gomez, and Yarin Gal,
  “Wat heb je gezegd? detecting out-of-distribution translations with
  variational transformers,”
  in Bayesian Deep Learning Workshop (NeurIPS), 2019.
* [21]

  Pascal Notin, José Miguel Hernández-Lobato, and Yarin Gal,
  “Principled uncertainty estimation for high dimensional data,”
  in Uncertainty & Robustness in Deep Learning Workshop, ICML,
  2020.
* [22]

  Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco
  Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia
  Specia,
  “Unsupervised quality estimation for neural machine translation,”
  arXiv preprint arXiv:2005.10608, 2020.
* [23]

  Tim Z Xiao, Aidan N Gomez, and Yarin Gal,
  “Wat heb je gezegd? detecting out-of-distribution translations with
  variational transformers,”
  2019.
* [24]

  Andrey Malinin and Mark Gales,
  “Uncertainty estimation in autoregressive structured prediction,”
  in International Conference on Learning Representations, 2021.
* [25]

  Shiyu Liang, Yixuan Li, and R. Srikant,
  “Enhancing the reliability of out-of-distribution image detection in
  neural networks,” 2020.
* [26]

  Yen-Chang Hsu, Yilin Shen, Hongxia Jin, and Zsolt Kira,
  “Generalized odin: Detecting out-of-distribution image without
  learning from out-of-distribution data,” 2020.
* [27]

  Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal,
  “Uncertainty estimation using a single deep deterministic neural
  network,”
  in International Conference on Machine Learning. PMLR, 2020,
  pp. 9690–9700.
* [28]

  Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper
  Snoek, Balaji Lakshminarayanan, Andrew M. Dai, and Dustin Tran,
  “Training independent subnetworks for robust prediction,” 2020.
* [29]

  Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, and
  Balaji Lakshminarayanan,
  “Simple and principled uncertainty estimation with deterministic
  deep learning via distance awareness,”
  arXiv preprint arXiv:2006.10108, 2020.
* [30]

  Joost van Amersfoort, Lewis Smith, Andrew Jesson, Oscar Key, and Yarin Gal,
  “Improving deterministic uncertainty estimation in deep learning for
  classification and regression,”
  arXiv preprint arXiv:2102.11409, 2021.
* [31]

  Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip HS Torr, and Yarin
  Gal,
  “Deterministic neural networks with appropriate inductive biases
  capture epistemic and aleatoric uncertainty,”
  arXiv preprint arXiv:2102.11582, 2021.
* [32]

  Andrey Malinin and Mark Gales,
  “Predictive uncertainty estimation via prior networks,”
  in Advances in Neural Information Processing Systems, 2018, pp.
  7047–7058.
* [33]

  Andrey Malinin and Mark JF Gales,
  “Reverse kl-divergence training of prior networks: Improved
  uncertainty and adversarial robustness,”
  2019.
* [34]

  Andrey Malinin, Sergey Chervontsev, Ivan Provilkov, and Mark Gales,
  “Regression prior networks,” 2020.
* [35]

  Andrey Malinin, Bruno Mlodozeniec, and Mark JF Gales,
  “Ensemble distribution distillation,”
  in International Conference on Learning Representations, 2020.
* [36]

  Max Ryabinin, Andrey Malinin, and Mark Gales,
  “Scaling ensemble distribution distillation to many classes with
  proxy targets,”
  arXiv preprint arXiv:2105.06987, 2021.
* [37]

  Angelos Filos, Sebastian Farquhar, Aidan N. Gomez, Tim G. J. Rudner, Zachary
  Kenton, Lewis Smith, Milad Alizadeh, Arnoud de Kroon, and Yarin Gal,
  “A systematic comparison of bayesian deep learning robustness in
  diabetic retinopathy tasks,” 2019.
* [38]

  Neil Band, Tim G. J. Rudner, Qixuan Feng, Angelos Filos, Zachary Nado,
  Michael W. Dusenberry, Ghassen Jerfel, Dustin Tran, and Yarin Gal,
  “Benchmarking bayesian deep learning on diabetic retinopathy
  detection tasks,”
  in Thirty-fifth Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track, 2021.
* [39]

  Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan
  Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob
  Steinhardt, and Justin Gilmer,
  “The many faces of robustness: A critical analysis of
  out-of-distribution generalization,” 2020.
* [40]

  Dan Hendrycks and Thomas Dietterich,
  “Benchmarking neural network robustness to common corruptions and
  perturbations,” 2019.
* [41]

  Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song,
  “Natural adversarial examples,” 2021.
* [42]

  J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei,
  “ImageNet: A Large-Scale Hierarchical Image Database,”
  in CVPR09, 2009.
* [43]

  Paul Michel and Graham Neubig,
  “MTNT: A testbed for Machine Translation of Noisy Text,”
  in Proceedings of the 2018 Conference on Empirical Methods in
  Natural Language Processing (EMNLP), 2018.
* [44]

  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner,
  “Gradient-based learning applied to document recognition,”
  Proceedings of the ieee, vol. 86, pp. 2278–2324,
  1998.
* [45]

  Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum,
  “Human-level concept learning through probabilistic program
  induction,”
  Science, vol. 350, no. 6266, pp. 1332–1338, 2015.
* [46]

  Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay D.
  Shet,
  “Multi-digit number recognition from street view imagery using deep
  convolutional neural networks,” 2013,
  arXiv:1312.6082.
* [47]

  Alex Krizhevsky,
  “Learning multiple layers of features from tiny images,”
  2009.
* [48]

  Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz,
  “Invariant risk minimization,”
  arXiv preprint arXiv:1907.02893, 2019.
* [49]

  João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy,
  and Abdelhamid Bouchachia,
  “A survey on concept drift adaptation,”
  ACM computing surveys (CSUR), vol. 46, no. 4, pp. 1–37, 2014.
* [50]

  Alexey Tsymbal,
  “The problem of concept drift: definitions and related work,”
  Computer Science Department, Trinity College Dublin, vol. 106,
  no. 2, pp. 58, 2004.
* [51]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin,
  “Catboost: unbiased boosting with categorical features,”
  in Proceedings of the 32nd International Conference on Neural
  Information Processing Systems (NeurIPS), 2018, pp. 6638–6648.
* [52]

  Andrey Malinin, Liudmila Prokhorenkova, and Aleksei Ustimenko,
  “Uncertainty in gradient boosting via ensembles,”
  in International Conference on Learning Representations, 2021.
* [53]

  Shuo Wang, Yang Liu, Chao Wang, Huanbo Luan, and Maosong Sun,
  “Improving back-translation with uncertainty-based confidence
  estimation,”
  arXiv preprint arXiv:1909.00157, 2019.
* [54]

  GlobalVoices,
  “Globalvoices,” <https://globalvoices.org/>.
* [55]

  Matt Post,
  “A call for clarity in reporting BLEU scores,”
  in Proceedings of the Third Conference on Machine Translation:
  Research Papers, Belgium, Brussels, Oct. 2018, pp. 186–191, Association for
  Computational Linguistics.
* [56]

  Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault,
  “Ground truth for grammatical error correction metrics,”
  in Proceedings of the 53rd Annual Meeting of the Association for
  Computational Linguistics and the 7th International Joint Conference on
  Natural Language Processing (Volume 2: Short Papers), Beijing, China, July
  2015, pp. 588–593, Association for Computational Linguistics.
* [57]

  Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault,
  “GLEU without tuning,”
  eprint arXiv:1605.02592 [cs.CL], 2016.
* [58]

  Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
  Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al.,
  “Google’s neural machine translation system: Bridging the gap
  between human and machine translation,”
  arXiv preprint arXiv:1609.08144, 2016.
* [59]

  Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng,
  David Grangier, and Michael Auli,
  “fairseq: A fast, extensible toolkit for sequence modeling,”
  in Proceedings of NAACL-HLT 2019: Demonstrations, 2019.
* [60]

  Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet,
  Li Fei-Fei, and Silvio Savarese,
  “Social lstm: Human trajectory prediction in crowded spaces,”
  in Proceedings of the IEEE conference on computer vision and
  pattern recognition, 2016, pp. 961–971.
* [61]

  Namhoon Lee, Wongun Choi, Paul Vernaza, Christopher B Choy, Philip HS Torr, and
  Manmohan Chandraker,
  “Desire: Distant future prediction in dynamic scenes with
  interacting agents,”
  in Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition, 2017, pp. 336–345.
* [62]

  Agrim Gupta, Justin Johnson, Li Fei-Fei, Silvio Savarese, and Alexandre Alahi,
  “Social gan: Socially acceptable trajectories with generative
  adversarial networks,”
  in Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition, 2018, pp. 2255–2264.
* [63]

  Yuning Chai, Benjamin Sapp, Mayank Bansal, and Dragomir Anguelov,
  “Multipath: Multiple probabilistic anchor trajectory hypotheses for
  behavior prediction,”
  arXiv preprint arXiv:1910.05449, 2019.
* [64]

  Sergio Casas, Cole Gulino, Renjie Liao, and Raquel Urtasun,
  “Spatially-aware graph neural networks for relational behavior
  forecasting from sensor data,”
  arXiv preprint arXiv:1910.08233, 2019.
* [65]

  Henggang Cui, Vladan Radosavljevic, Fang-Chieh Chou, Tsung-Han Lin, Thi Nguyen,
  Tzu-Kuo Huang, Jeff Schneider, and Nemanja Djuric,
  “Multimodal trajectory predictions for autonomous driving using deep
  convolutional networks,”
  in 2019 International Conference on Robotics and Automation
  (ICRA). IEEE, 2019, pp. 2090–2096.
* [66]

  Tung Phan-Minh, Elena Corina Grigore, Freddy A Boulton, Oscar Beijbom, and
  Eric M Wolff,
  “Covernet: Multimodal behavior prediction using trajectory sets,”
  in Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, 2020, pp. 14074–14083.
* [67]

  Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and
  Cordelia Schmid,
  “Vectornet: Encoding hd maps and agent dynamics from vectorized
  representation,”
  in Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, 2020, pp. 11525–11533.
* [68]

  Yuriy Biktairov, Maxim Stebelev, Irina Rudenko, Oleh Shliazhko, and Boris
  Yangel,
  “Prank: motion prediction based on ranking,”
  in Advances in Neural Information Processing Systems,
  H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, Eds. 2020,
  vol. 33, pp. 2553–2563, Curran Associates, Inc.
* [69]

  Sergio Casas, Cole Gulino, Simon Suo, Katie Luo, Renjie Liao, and Raquel
  Urtasun,
  “Implicit latent variable model for scene-consistent motion
  forecasting,”
  in Computer Vision–ECCV 2020: 16th European Conference,
  Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIII 16. Springer,
  2020, pp. 624–641.
* [70]

  Ming Liang, Bin Yang, Rui Hu, Yun Chen, Renjie Liao, Song Feng, and Raquel
  Urtasun,
  “Learning lane graph representations for motion forecasting,”
  in European Conference on Computer Vision. Springer, 2020, pp.
  541–556.
* [71]

  Yicheng Liu, Jinghuai Zhang, Liangji Fang, Qinhong Jiang, and Bolei Zhou,
  “Multimodal motion prediction with stacked transformers,”
  in Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, 2021, pp. 7577–7586.
* [72]

  Angelos Filos, Panagiotis Tigkas, Rowan McAllister, Nicholas Rhinehart, Sergey
  Levine, and Yarin Gal,
  “Can autonomous vehicles identify, recover from, and adapt to
  distribution shifts?,”
  in International Conference on Machine Learning. PMLR, 2020,
  pp. 3145–3153.
* [73]

  Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, and
  Alexey Dosovitskiy,
  “End-to-end driving via conditional imitation learning,”
  in 2018 IEEE International Conference on Robotics and Automation
  (ICRA). IEEE, 2018, pp. 4693–4700.
* [74]

  Nicholas Rhinehart, Rowan McAllister, and Sergey Levine,
  “Deep imitative models for flexible inference, planning, and
  control,”
  CoRR, vol. abs/1810.06544, 2018.
* [75]

  Lewis Smith and Yarin Gal,
  “Understanding measures of uncertainty for adversarial example
  detection,” 2018.
* [76]

  Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan,
  Hanna Wallach, Hal Daumé III, and Kate Crawford,
  “Datasheets for datasets,”
  arXiv preprint arXiv:1803.09010, 2018.
* [77]

  Deliang Chen and Hans Weiteng Chen,
  “Using the köppen classification to quantify climate variation and
  change: An example for 1901–2010,”
  Environmental Development, vol. 6, pp. 69–79, 2013.
* [78]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko,
  “Revisiting deep learning models for tabular data,”
  arXiv preprint arXiv:2106.11959, 2021.
* [79]

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
  “Deep residual learning for image recognition,”
  in Proceedings of the IEEE conference on computer vision and
  pattern recognition, 2016, pp. 770–778.
* [80]

  Abraham Wald,
  “Contributions to the Theory of Statistical Estimation and Testing
  Hypotheses,”
  The Annals of Mathematical Statistics, vol. 10, no. 4, pp. 299
  – 326, 1939.
* [81]

  Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
  Fethi Bougares, Holger Schwenk, and Yoshua Bengio,
  “Learning phrase representations using rnn encoder-decoder for
  statistical machine translation,”
  arXiv preprint arXiv:1406.1078, 2014.
* [82]

  Danilo Jimenez Rezende and Shakir Mohamed,
  “Variational inference with normalizing flows,” 2016.
* [83]

  Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan,
  “Deep ensembles: A loss landscape perspective,” 2020.

\appendixpage

## Appendix A Assessment Metrics

As discussed in [Section 2](#S2 "2 Evaluation Paradigm, Metrics, and Baselines ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), in this work we consider robustness and uncertainty estimation to be two equally important factors in assessing the reliability of a model. We assume that as the degree of distributional shift increases, so should a model’s errors; in other words, a model’s uncertainty estimates should be correlated with the degree of its error. This informs our choice of assessment metrics, which must *jointly* assess robustness and uncertainty estimation.

One standard approach to jointly assess robustness and uncertainty are *error-retention curves* [[12](#bib.bib12), [14](#bib.bib14)], which plot a model’s mean error over a dataset, as measured using a metric such as error-rate, MSE, eGLEU, cNLL, etc., with respect to the fraction of the dataset for which the model’s predictions are used. These retention curves are traced by replacing a model’s predictions with ground-truth labels obtained from an oracle in order of decreasing uncertainty, thereby decreasing error. Ideally, a model’s uncertainty is correlated with its error, and therefore the most errorful predictions would be replaced first, which would yield the greatest reduction in mean error as more predictions are replaced. This represents a hybrid human-AI scenario, where a model can consult an oracle (human) for assistance in difficult situations and obtain from the oracle a perfect prediction on those examples.

The area under the retention curve (R-AUC) is a metric for jointly assessing robustness to distributional shift and the quality of the uncertainty estimates. R-AUC can be reduced either by improving the predictions of the model, such that it has lower overall error at any given retention rate, or by providing estimates of uncertainty which better correlate with error, such that the most incorrect predictions are rejected first. It is important that the dataset in question contains both a subset “matched” to the training data, and a distributionally shifted subset.
Figure [4](#A1.F4 "Figure 4 ‣ Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") provides example retention curves for the three tasks of the Shifts Dataset. In each figure, in addition to the uncertainty-based ranking, we included curves which represent “random” ranking, where uncertainty estimates are entirely non-informative, and “optimal” ranking, where uncertainty estimates perfectly correlate with error. These represent the lower and upper bounds on R-AUC performance as a function of uncertainty quality.

!(/html/2107.07455/assets/figures/weather_retention_dev_mse.png)

(a) Weather Prediction

!(/html/2107.07455/assets/figures/error_retention_example_nmt.png)

(b) Machine Translation

!(/html/2107.07455/assets/x3.png)

(c) Vehicle Motion Prediction

Figure 4: Example error retention curves for the three tasks of the Shifts Dataset.

While clearly interpretable and intuitive, one concern that can be raised regarding error-retention curves is that they can be more sensitive to predictive performance than to the quality of uncertainty estimates, which can be seen in Figure [4](#A1.F4 "Figure 4 ‣ Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")b. This occurs on tasks where most errors have similar magnitude. Furthermore, for regression tasks, retention curves are dominated by noise in the targets (aleatoric uncertainty) at low retention fractions, when most systematic errors have already been detected. Therefore, in this work we propose another metric which jointly assesses robustness and uncertainty estimation.

First, we introduce the notion of an “acceptable prediction”, which is a prediction whose error is acceptably small. This concept is natural for tasks with a non-binary notion of error, e.g., regression problems. For classification tasks, where predictions are already either correct or incorrect (acceptable/non-acceptable), this concept can be introduced by considering different levels of risk for different misclassifications. Formally, we say that a prediction is acceptable if an appropriate metric of error or risk ℰℰ\mathcal{E} is below a *fixed* task-dependent error threshold Tesubscript𝑇𝑒T\_{e}. For example, if temperature is predicted to within a degree of the ground truth, then it is acceptable. This allows us to mitigate the issue of errors having similar magnitudes. This is expressed using via an indicator function as follows:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜Te​(𝒙)={1,ℰ​(𝒙)≤Te0,ℰ​(𝒙)>Tesubscript𝒜subscript𝑇𝑒𝒙cases  1ℰ𝒙 subscript𝑇𝑒otherwise  0ℰ𝒙 subscript𝑇𝑒otherwise\displaystyle\mathcal{A}\_{T\_{e}}(\bm{x})=\ \begin{cases}1,\ \mathcal{E}(\bm{x})\leq T\_{e}\\ 0,\ \mathcal{E}(\bm{x})>T\_{e}\\ \end{cases} |  | (1) |

For a given dataset D𝐷D and model, we first set an error threshold and determine which predictions are acceptable – this yields a set of “ground-truth” acceptability labels 𝒜i=1Nsuperscriptsubscript𝒜𝑖1𝑁\mathcal{A}\_{i=1}^{N}. We can now use these acceptability labels to assess whether the model’s *estimates of uncertainty* 𝒰​(𝒙)𝒰𝒙\mathcal{U}(\bm{x}) can be used to indicate whether a prediction is acceptable. If the uncertainty score is greater than a threshold Tusubscript𝑇𝑢T\_{u}, then we consider the prediction to be poor, if the uncertainty score is lower than this threshold, the prediction is considered to be acceptable.

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒜^Tu​(𝒙)={1,𝒰​(𝒙)≤Tu0,𝒰​(𝒙)>Tusubscript^𝒜subscript𝑇𝑢𝒙cases  1𝒰𝒙 subscript𝑇𝑢otherwise  0𝒰𝒙 subscript𝑇𝑢otherwise\displaystyle\mathcal{\hat{A}}\_{T\_{u}}(\bm{x})=\ \begin{cases}1,\ \mathcal{U}(\bm{x})\leq T\_{u}\\ 0,\ \mathcal{U}(\bm{x})>T\_{u}\\ \end{cases} |  | (2) |

Next, given the true acceptability labels {𝒜Te​(𝒙i)}i=1Nsuperscriptsubscriptsubscript𝒜subscript𝑇𝑒subscript𝒙𝑖𝑖1𝑁\{\mathcal{A}\_{T\_{e}}(\bm{x}\_{i})\}\_{i=1}^{N} and the threshold-conditional indicators {𝒜^Tu​(𝒙)}i=1Nsuperscriptsubscriptsubscript^𝒜subscript𝑇𝑢𝒙𝑖1𝑁\{\mathcal{\hat{A}}\_{T\_{u}}(\bm{x})\}\_{i=1}^{N} we sweep through all uncertainty scores in a dataset {𝒰​(𝒙i)}i=1Nsuperscriptsubscript𝒰subscript𝒙𝑖𝑖1𝑁\{\mathcal{U}(\bm{x}\_{i})\}\_{i=1}^{N} in decreasing order and use them as thresholds to F1 for classifying whether a prediction is actually acceptable or not based on the uncertainty. Formally, this is done as follows:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Pi=subscript𝑃𝑖absent\displaystyle P\_{i}= | ∑j=1N𝒜Te​(𝒙j)⋅𝒜^𝒰i​(𝒙j)N−i,Ri=∑j=1N𝒜Te​(𝒙j)⋅𝒜^𝒰i​(𝒙j)∑j=1N𝒜Te​(𝒙j),F1i=2⋅Pi⋅RiPi+Riformulae-sequence  superscriptsubscript𝑗1𝑁⋅subscript𝒜subscript𝑇𝑒subscript𝒙𝑗subscript^𝒜subscript𝒰𝑖subscript𝒙𝑗𝑁𝑖subscript𝑅𝑖 superscriptsubscript𝑗1𝑁⋅subscript𝒜subscript𝑇𝑒subscript𝒙𝑗subscript^𝒜subscript𝒰𝑖subscript𝒙𝑗superscriptsubscript𝑗1𝑁subscript𝒜subscript𝑇𝑒subscript𝒙𝑗subscriptF1𝑖⋅2subscript𝑃𝑖subscript𝑅𝑖subscript𝑃𝑖subscript𝑅𝑖\displaystyle\ \frac{\sum\_{j=1}^{N}\mathcal{A}\_{T\_{e}}(\bm{x}\_{j})\cdot\mathcal{\hat{A}}\_{\mathcal{U}\_{i}}(\bm{x}\_{j})}{N-i},\ R\_{i}=\ \frac{\sum\_{j=1}^{N}\mathcal{A}\_{T\_{e}}(\bm{x}\_{j})\cdot\mathcal{\hat{A}}\_{\mathcal{U}\_{i}}(\bm{x}\_{j})}{\sum\_{j=1}^{N}\mathcal{A}\_{T\_{e}}(\bm{x}\_{j})},\ \text{F1}\_{i}=\ \frac{2\cdot P\_{i}\cdot R\_{i}}{P\_{i}+R\_{i}} |  | (3) |

where we use N−i𝑁𝑖N-i because we sort uncertainties from largest (𝒰1subscript𝒰1\mathcal{U}\_{1}) to smallest (𝒰Nsubscript𝒰𝑁\mathcal{U}\_{N}). We then plot {F1i}i=1NsuperscriptsubscriptsubscriptF1𝑖𝑖1𝑁\{\text{F1}\_{i}\}\_{i=1}^{N} against 1−iN1𝑖𝑁1-\frac{i}{N}, i.e., the fraction of data we are classifying as acceptable, which we refer to as the retention fraction. This yields the following curves for the three Shifts tasks:

!(/html/2107.07455/assets/figures/weather_retention_dev_f1.png)

(a) Weather Prediction

!(/html/2107.07455/assets/figures/F1_retention_example_nmt.png)

(b) Machine Translation

!(/html/2107.07455/assets/x4.png)

(c) Motion Prediction

Figure 5: Examples of F1-Retention curves for the three tasks of the Shifts Dataset.

Here we plot the uncertainty-based F1-retention curves for all datasets. On each figure, we plot both the uncertainty-derived curves as well as the “random” and “optimal” baselines, where uncertainties are either completely uncorrelated or perfectly correlated with errors, respectively. Better models have a higher area under this F1-retention curve (F1-AUC). The predictive performance of the model defines the starting point at 100% retention – better models start higher. Thus, area under the F1 curve can be increased by having a model which yield better predictions or by improving the correlation between uncertainty and error. Note, in contrast to Figure [4](#A1.F4 "Figure 4 ‣ Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")b, the quality of the ranking affects area under the curve far more than for error-retention curves. Thus, this metric is especially useful when errors have similar magnitudes.

Finally, it is necessary to point out that area under the error-retention curve and F1-retention curve is a *summary statistic* which describes possible *operating points*. We can specify a particular operating point, such as 95% retention, and evaluate the error or F1 at that point for comparison. This is also an important figure, as all models work at a particular operating point which satisfies task-specific desiderata. In this work the desiderata for all tasks will be to not reject more than 5% of the input data.

## Appendix B Shifts Dataset General Datasheet

Here we describe the motivation, uses, distribution as well as the maintenance and support plan for the Shifts Dataset as whole in the *datasheet for datasets* format [[76](#bib.bib76)]. The details of the composition, collection and pre-prossessing of each component dataset are provided in appendices [C](#A3 "Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")-[E](#A5 "Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")

##### Motivation

As discussed at length in the main body of the paper, the primary goal for the creation of the Shifts Dataset was the evaluation of uncertainty quantification models and robustness to distributional shift on a range of large-scale, industrial tasks spanning multiple modalities. To this end, Yandex Research, in collaboration with the Yandex.Translate, Yandex.Weather services and Yandex Self-Driving Group created the Shifts Dataset. As the dataset creation was done by Yandex teams, it was therefore funded by Yandex.

##### Uses

The dataset is used as part of the Shifts Challenge which was organized as part of NeurIPS2021, which was organized around this dataset888 <research.yandex.com/shifts>. The Shifts Challenge consists of three tracks organized around each of the consituent datasets within Shifts. The dataset, baseline models and code to reproduce it all is provided in a GitHub repository999<https://github.com/yandex-research/shifts>. Other than uncertainty and robustness research the dataset could be used for developing better models for each of the separate tasks - tabular data, translation and vehicle motion prediction.

##### Distribution

The parts of the dataset which were produced by Yandex are distributed under an open-source CC BY NC SA 4.0 license. All the code is available under an open-source Apache 2.0 licence. It is our intention that the dataset be freely available for research purposes. The dataset is available as a tarball download from GitHub. Currently, as the Shifts Challenge is still underway, only the training and development sets are available. However, the full dataset, with full accompanying metadata, will be available once the challenge concludes on November 1st, 2021. Licence details for each constituent dataset in Shifts are described in appendices [C](#A3 "Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")-[E](#A5 "Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Maintenance

The dataset is being actively maintained by Yandex Research, with support from the weather, translation and self-driving teams, and the teams can be contacted by raising an issue on GitHub and by writing to the first author of this paper. The dataset is currently hosted on Yandex S3 storage and will be hosted there permanently for the foreseeable future. The dataset can be updated at the discretion of the dataset creators, though regular updates are not planned. Updates which expand the evaluation sets or add new ones will mean that the previous dev/eval sets are supported. Updates which fix errors in dev/eval sets mean that the prior ones are obsolete and unsupported. If any update is to occur, we will make an announcement via GitHub, twitter, and the Shifts challenge mailing list. Currently, as the data comes directly from Yandex, we do not allows other parties to update the Shifts Dataset. However, any issues found can be logged by raising an issue on GitHub or contacting the first author of this paper so that we can address them. Furthermore, as we are releasing the data under an open-source CC BY NC SA 4.0 license which allows modifications, we are happy for people to create derivative datasets using ours, provided the modifications are documents and the original dataset references.

##### Societal Consequences and Guidelines for Ethical Use

Research on uncertainty estimation and robustness aims to make AI safer and more reliable, and therefore has limited negative societal consequences overall. Users of this dataset are encouraged to use it for the purpose of improving the reliability and safety of large-scale applications of machine learning. Furthermore, we encourage users of out dataset to develop compute and memory efficient methods for improving safety and reliability.

##### Responsibility

The authors confirm that, to the best of our knowledge, the released dataset does not violate any prior licenses or rights. However, if such a violation were to exist, we are responsible for resolving this issue.

## Appendix C Tabular Weather

The current appendix contains a description of the composition, collection, pre-processing and partitioning of the Shifts Tabular Weather Prediction dataset. Additionally, it contains a description of the metrics used for assessment and an expanded set of experimental results.

### C.1 Dataset Description

##### Composition

The data consists of pairs of meteorological features and target values at a particular latitude/longitude and time. The target value is air temperature measurements at 2 metres above the ground for regression and precipitation and cloudiness class from weather station measurements for classification. The feature vectors include both weather-related features such as sun evaluation at the current location, climate values of temperature, pressure and topography, and meteorological parameters on different pressure and surface levels from *weather forecast model predictions*. *Weather forecast model predictions* are values produced by the following weather forecast models: Global Forecast System (GFS),101010<https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs> Global Deterministic Forecast System from the Canadian Meteorological Center (CMC),111111<https://weather.gc.ca/grib/grib2_glb_25km_e.html> and the Weather Research and Forecasting (WRF) Model.121212<https://www.mmm.ucar.edu/weather-research-and-forecasting-model> Each model returns the following predicted values: wind, humidity, pressure, clouds, precipitation, dew point, snow depth, air and soil temperature characteristics. Where applicable, the predictions are given at different isobaric levels from 50 hPa (≈\approx 20 km above ground) to the ground level. The GFS and WRF models run 4 times a day (0, 6, 12 and 18 GMT), and the CMC model runs twice a day (0 and 12 GMT). Model spatial grid resolution is 0.25​°×0.25​°0.25°0.25°0.25\degree\times 0.25\degree for GFS and 0.24​°×0.24​°0.24°0.24°0.24\degree\times 0.24\degree for CMC. The WRF model is calculated for over 60 domains all over the globe, spatial resolution for each domain is 6 ×\times 6 km. Altogether, there are 123 features in total. It is important to note that the features are highly heterogeneous, i.e., they are of different types and scales. The target air temperature values at different locations are taken from about 8K weather stations located across the globe, each of which periodically (≈\approx each 3 hours) reports a set of measurements. In total, the dataset has 129 columns: 123 features, 4 meta-data attributes including time, latitude, longitude, and 2 targets - temperature (target for regression task), precipitation class (target for classification task) and climate type. The full feature list is provided in Section [C.2](#A3.SS2.SSS0.Px3 "Features ‣ C.2 Detailed description of features and targets ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

##### Collection Process

The data for features from GFS and CMC weather forecast models was downloaded in the GRIB file format from the web resources <https://www.ncdc.noaa.gov> and <https://weather.gc.ca/>, respectively. The GRIB files were decoded and collected by the production system of the Yandex Weather forecast service. MD5 hashes for files were checked after downloading the data. The parameters from WRF model were obtained from WRF model v3.6.1 computation on Yandex Weather servers. The data was checked for mistakes and outliers. Some parameters were converted to different units (for example degrees from K to C). We selected a subset of 123 weather parameters from the full dataset based on expertise and research of feature importance for weather forecasts of temperature and precipitation for the Yandex Weather production system. The data for weather station observations was downloaded from <https://www.ncdc.noaa.gov> and was decoded from SYNOP code. We filtered missed values and outliers by comparing with previous observations on the same weather station, and by comparing observation with nearby weather stations. Scripts and program codes for data collection and processing were prepared by in-house Yandex Weather software engineers. The period of data collection is from September 2018 to September 2019.

##### Preprocessing, Cleaning and Labelling

The data was logged during applying trained CatBoost models for weather forecast prediction of the Yandex Weather service and was validated on Yandex Weather users by providing actual weather forecasts and accessing its mistakes on users and station measurements. We labeled data to match the timestamp of features and targets from these logs. Also we selected features only for latitudes and longitudes of weather observation stations to match with the measurements. Targets for air temperature were converted to degrees Celsius. Targets for precipitation class were constructed from cloudiness and precipitation measurements to create 9 classes and labeled as follows: 0 — no precipitation, no clouds, 1 — no precipitation, partly cloudy, 2 — rain, partly cloudy, 3 — sleet, partly cloudy, 4 — snow, partly cloudy, 5 — no precipitation, cloudy, 6 — rain, cloudy, 7 — sleet, cloudy, 8 — snow, cloudy. The “raw” data was not saved, because it requires large amount of disk space. It was deleted after processing the data.

##### Partitioning into train, development, and evaluation sets

To analyze the robustness of learned models to *climate shifts*, we use the Koppen climate classification [[77](#bib.bib77)] that provides publicly available data131313Available to download from <http://hanschen.org/koppen> that maps latitudes and longitudes at a 0.5∘superscript0.50.5^{\circ} resolution to one of five main climate types: Tropical, Dry, Mild Temperate, Snow and Polar. This information is available over the years 1901 to 2010. The Weather Prediction dataset is augmented such that each sample has an associated climate type. The climate type is determined by minimizing the 1-norm between the longitudes/latitudes in the weather data and the Koppen climate classification for the most recent year available, 2010. The climate type is not used as a training feature.

!(/html/2107.07455/assets/x5.png)

Figure 6: Canonical Partitioning of Weather Prediction dataset.

There are 10M records in the full dataset distributed uniformly between September 1stsuperscript1st1^{\text{st}}, 2018, and September 1stsuperscript1st1^{\text{st}}, 2019, with samples across all five climate types. To test the robustness of the models, we evaluate how well they perform on time-shifted and climate-shifted data. Model performance is expected to decrease with time and climate shifts. However, a robust model is expected to be stable with these shifts.

In order to provide a standard benchmark which contains data which is both matched and shifted relative to the training set, we split the full dataset into ‘canonically partitioned’141414Alternative partitioning can be made from the full data, but we will use the canonical partition throughout this work. training, development, and evaluation datasets as follows (see Figure [6](#A3.F6 "Figure 6 ‣ Partitioning into train, development, and evaluation sets ‣ C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")):

* •

  The training data consists of measurements made from September 2018 till April 8thsuperscript8th8^{\text{th}}, 2019 for climate types Tropical, Dry, and Mild Temperate. The training data includes two dummy rows in order to ensure there is at least one example of each of the precipitation classes (the targets for the classification task). The values for each of the features of the dummy examples are computed by averaging across the whole training dataset.
* •

  The development data is composed of in-domain (dev\_in) and out-of-domain (dev\_out) data. The in-domain data corresponds to the same time range and climate types as the training data. The out-of-domain development data consists of measurements made from 8thsuperscript8th8^{\text{th}} July till 1stsuperscript1st1^{\text{st}} September 2019 for the climate type Snow. 50K data points are subsampled for the climate type Snow within this time range to construct dev\_out.
* •

  The evaluation data is also composed of in-domain (eval\_in) and out-of-domain (eval\_out) data. As before, the in-domain data corresponds to the same time range and climate types as the training data. The out-of-domain evaluation data is further shifted than the out-of-domain development data; measurements are taken from 14thsuperscript14th14^{\text{th}} May till 8thsuperscript8th8^{\text{th}} July 2019, which is more distant in terms of the time of the year from the in-domain data compared to the out-of-domain development data. The climate types are restricted to Snow and Polar.

Table [8](#A3.T8 "Table 8 ‣ Partitioning into train, development, and evaluation sets ‣ C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") details the number of samples in the selected partition of the data. It also details the number of samples for each climate type for each part of the dataset. The in-domain data is split in approximately 83.7-1.3-15% ratio between training, development, and evaluation. Figure [7](#A3.F7 "Figure 7 ‣ Partitioning into train, development, and evaluation sets ‣ C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") depicts the shift in the target temperatures between the training, development, and evaluation datasets. It is clear that the temperature distribution is different for dev\_out and eval\_out compared to the in-domain sets. The higher average temperature in the out of domain sets is perhaps due to the out of domain data being sourced from the Summer regions (for the northern hemisphere) while the in-domain data is largely sourced from the Winter time period. Figure [8](#A3.F8 "Figure 8 ‣ Partitioning into train, development, and evaluation sets ‣ C.1 Dataset Description ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") further shows the shift in the samples’ locations (latitudes/longitudes) between training, development, and evaluation datasets. The location shift is a natural result of the climate shifts present in the datasets where the training data tends to correspond to warmer parts of the world, whereas the development and evaluation datasets include colder climates too.

Table 8: Number of samples in the canonical partitioning of Weather Prediction dataset.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Data | # of samples | | | | | |
|  | Total | Tropical | Dry | Mild Temperate | Snow | Polar |
| Training | train | 3,129,592 | 416,310 | 690,284 | 2,022,998 | 0 | 0 |
| Development | dev\_in | 50,000 | 6,641 | 10,961 | 32,398 | 0 | 0 |
| dev\_out | 50,000 | 0 | 0 | 0 | 50,000 | 0 |
| dev | 100,000 | 6,641 | 10,961 | 32,398 | 50,000 | 0 |
| Evaluation | eval\_in | 561,105 | 74,406 | 123,487 | 363,212 | 0 | 0 |
| eval\_out | 576,626 | 0 | 0 | 0 | 525,967 | 50,659 |
| eval | 1,137,731 | 74,406 | 123,487 | 363,212 | 525,967 | 50,659 |

!(/html/2107.07455/assets/figures/weather_violin_plots.png)

Figure 7: Temperature distributions on canonical partitions of Weather Prediction dataset.

!(/html/2107.07455/assets/figures/weather_location_dist_train.png)

(a) train.

!(/html/2107.07455/assets/figures/weather_location_dist_dev_in.png)

(b) dev\_in.

!(/html/2107.07455/assets/figures/weather_location_dist_eval_in.png)

(c) eval\_in.

!(/html/2107.07455/assets/figures/weather_location_dist_dev_out.png)

(d) dev\_out.

!(/html/2107.07455/assets/figures/weather_location_dist_eval_out.png)

(e) eval\_out.

Figure 8: Location of samples from canonical partitioning of Weather Prediction dataset.

!(/html/2107.07455/assets/figures/weather_clim_type_train.png)

(a) train.

!(/html/2107.07455/assets/figures/weather_clim_type_dev_in.png)

(b) dev\_in.

!(/html/2107.07455/assets/figures/weather_clim_type_eval_in.png)

(c) eval\_in.

!(/html/2107.07455/assets/figures/weather_clim_type_dev_out.png)

(d) dev\_out.

!(/html/2107.07455/assets/figures/weather_clim_type_eval_out.png)

(e) eval\_out.

!(/html/2107.07455/assets/figures/weather_clim_type_all.png)

(f) all.

Figure 9: Distribution of climate types from canonical partitioning of Weather Prediction dataset.

!(/html/2107.07455/assets/figures/weather_precip_class_train.png)

(a) train.

!(/html/2107.07455/assets/figures/weather_precip_class_dev_in.png)

(b) dev\_in.

!(/html/2107.07455/assets/figures/weather_precip_class_eval_in.png)

(c) eval\_in.

!(/html/2107.07455/assets/figures/weather_precip_class_dev_out.png)

(d) dev\_out.

!(/html/2107.07455/assets/figures/weather_precip_class_eval_out.png)

(e) eval\_out.

!(/html/2107.07455/assets/figures/weather_precip_class_all.png)

(f) all.

Figure 10: Distribution of precipitation classes from canonical partitioning of Weather Prediction dataset.

##### Format

This dataset is provided in CSV format.

##### Licence

This dataset is provided under the CC BY NC SA 4.0 license.

### C.2 Detailed description of features and targets

##### Meta-Data Features

1. 1.

   fact\_time — timestamp
2. 2.

   fact\_latitude — geographical latitude, degrees
3. 3.

   fact\_longitude — geographical longitude, degrees
4. 4.

   climate — major climate type

##### Targets

1. 1.

   fact\_temperature — air temperature 2m above the ground, C
2. 2.

   fact\_cwsm\_class - precipitation class

##### Features

1. 1.

   climate\_pressure — climate pressure, mmHg
2. 2.

   climate\_temperature — climate temperature, C
3. 3.

   cmc\_0\_0\_0\_1000 — temperature at 1000 hPa isobaric level, K
4. 4.

   cmc\_0\_0\_0\_2 — temperature at 2m, K
5. 5.

   cmc\_0\_0\_0\_2\_grad — difference between temperatures on adjacent horizons at 2m, K
6. 6.

   cmc\_0\_0\_0\_2\_interpolated — temperature at 2m interpolated between horizons, K
7. 7.

   cmc\_0\_0\_0\_2\_next — temperature at 2m for next horizon, K
8. 8.

   cmc\_0\_0\_0\_500 — temperature at 500 hPa isobaric level, K
9. 9.

   cmc\_0\_0\_0\_700 — temperature at 700 hPa isobaric level, K
10. 10.

    cmc\_0\_0\_0\_850 — temperature at 850 hPa isobaric level, K
11. 11.

    cmc\_0\_0\_0\_925 — temperature at 925 hPa isobaric level, K
12. 12.

    cmc\_0\_0\_6\_2 — dew point temp at 2m, K
13. 13.

    cmc\_0\_0\_7\_1000 — dew point depression at 1000 hPa isobaric level, K
14. 14.

    cmc\_0\_0\_7\_2 — dew point depression at 2m, K
15. 15.

    cmc\_0\_0\_7\_500 — dew point depression at 500 hPa isobaric level, K
16. 16.

    cmc\_0\_0\_7\_700 — dew point depression at 700 hPa isobaric level, K
17. 17.

    cmc\_0\_0\_7\_850 — dew point depression at 850 hPa isobaric level, K
18. 18.

    cmc\_0\_0\_7\_925 — dew point depression at 925 hPa isobaric level, K
19. 19.

    cmc\_0\_1\_0\_0 — absolute humidity from 0 to 1
20. 20.

    cmc\_0\_1\_11\_0 — snow depth, m
21. 21.

    cmc\_0\_1\_65\_0 — rain accumulated from cmc gentime, mm
22. 22.

    cmc\_0\_1\_65\_0\_grad — rain accumulated from cmc gentime difference between adjacent horizons, mm
23. 23.

    cmc\_0\_1\_65\_0\_next — rain accumulated from cmc gentime for next horizon, mm
24. 24.

    cmc\_0\_1\_66\_0 — snow accumulated from cmc gentime, mm
25. 25.

    cmc\_0\_1\_66\_0\_grad — snow accumulated from cmc gentime difference between adjacent horizons, mm
26. 26.

    cmc\_0\_1\_66\_0\_next — snow accumulated from cmc gentime for next horizon, mm
27. 27.

    cmc\_0\_1\_67\_0 — ice rain accumulated from cmc gentime, mm
28. 28.

    cmc\_0\_1\_67\_0\_grad — ice rain accumulated from cmc gentime difference between adjacent horizons, mm
29. 29.

    cmc\_0\_1\_67\_0\_next — ice rain accumulated from cmc gentime for next horizon, mm
30. 30.

    cmc\_0\_1\_68\_0 — iced graupel accumulated from cmc gentime, mm
31. 31.

    cmc\_0\_1\_68\_0\_grad — iced graupel accumulated from cmc gentime difference between adjacent horizons, mm
32. 32.

    cmc\_0\_1\_68\_0\_next — iced graupel accumulated from cmc gentime for next horizon, mm
33. 33.

    cmc\_0\_1\_7\_0 — instant precipitation intensity, mm/h
34. 34.

    cmc\_0\_2\_2\_10 — wind U component at 10m, m/s
35. 35.

    cmc\_0\_2\_2\_1000 — wind U component at 1000 hPa isobaric level, m/s
36. 36.

    cmc\_0\_2\_2\_500 — wind U component at 500 hPa isobaric level, m/s
37. 37.

    cmc\_0\_2\_2\_700 — wind U component at 700 hPa isobaric level, m/s
38. 38.

    cmc\_0\_2\_2\_850 — wind U component at 850 hPa isobaric level, m/s
39. 39.

    cmc\_0\_2\_2\_925 — wind U component at 925 hPa isobaric level, m/s
40. 40.

    cmc\_0\_2\_3\_10 — wind V component at 10m, m/s
41. 41.

    cmc\_0\_2\_3\_1000 — wind V component at 1000 hPa isobaric level, m/s
42. 42.

    cmc\_0\_2\_3\_500 — wind V component at 500 hPa isobaric level, m/s
43. 43.

    cmc\_0\_2\_3\_700 — wind V component at 700 hPa isobaric level, m/s
44. 44.

    cmc\_0\_2\_3\_850 — wind V component at 850 hPa isobaric level, m/s
45. 45.

    cmc\_0\_2\_3\_925 — wind V component at 925 hPa isobaric level, m/s
46. 46.

    cmc\_0\_3\_0\_0 — surface pressure, Pa
47. 47.

    cmc\_0\_3\_0\_0\_next — next horizon surface pressure, Pa
48. 48.

    cmc\_0\_3\_1\_0 — sea level pressure, Pa
49. 49.

    cmc\_0\_3\_5\_1000 — geopotential height at 1000 hPa isobaric level, gpm (geopotential meter)
50. 50.

    cmc\_0\_3\_5\_500 — geopotential height at 500 hPa isobaric level, gpm
51. 51.

    cmc\_0\_3\_5\_700 — geopotential height at 700 hPa isobaric level, gpm
52. 52.

    cmc\_0\_3\_5\_850 — geopotential height at 850 hPa isobaric level, gpm
53. 53.

    cmc\_0\_3\_5\_925 — geopotential height at 925 hPa isobaric level, gpm
54. 54.

    cmc\_0\_6\_1\_0 — cloudiness, % from 0 to 100
55. 55.

    cmc\_available — is there any data from cmc
56. 56.

    cmc\_horizon\_h — cmc horizon, h
57. 57.

    cmc\_precipitations — avg precipitations rate between adjacent horizons, mm/h
58. 58.

    cmc\_timedelta\_s — difference between cmc and forecast time, s
59. 59.

    gfs\_2m\_dewpoint — dew point temperature at 2m, C
60. 60.

    gfs\_2m\_dewpoint\_grad — dew point temperature at 2m difference between horizons, C
61. 61.

    gfs\_2m\_dewpoint\_next — dew point temperature on next horizon, C
62. 62.

    gfs\_a\_vorticity — absolute vorticity at height 1000 hPa, s-1
63. 63.

    gfs\_available — is there any data from gfs
64. 64.

    gfs\_cloudness — sum of 3 level cloudiness, from 0 to 3
65. 65.

    gfs\_clouds\_sea — Cloud mixing ratio at level 1000 hPa, kg/kg 0.0
66. 66.

    gfs\_horizon\_h — gfs horizon, h
67. 67.

    gfs\_humidity — relative humidity at 2m, %
68. 68.

    gfs\_precipitable\_water — total precipitable water, kg m2−{}^{-}2
69. 69.

    gfs\_precipitations — avg precipitations rate between adjacent horizons, mm/h
70. 70.

    gfs\_pressure — surface pressure, mmHg
71. 71.

    gfs\_r\_velocity — vertical Velocity at 1000 hPa, Pa/s
72. 72.

    gfs\_soil\_temperature — soil temperature at 0.0-0.1 m, C
73. 73.

    gfs\_soil\_temperature\_available — is there gfs soil temp data
74. 74.

    gfs\_temperature\_10000 — temperature at vertical level at 100 hPa, C
75. 75.

    gfs\_temperature\_15000 — temperature at vertical level at 150 hPa, C
76. 76.

    gfs\_temperature\_20000 — temperature at vertical level at 200 hPa, C
77. 77.

    gfs\_temperature\_25000 — temperature at vertical level at 250 hPa, C
78. 78.

    gfs\_temperature\_30000 — temperature at vertical level at 300 hPa, C
79. 79.

    gfs\_temperature\_35000 — temperature at vertical level at 350 hPa, C
80. 80.

    gfs\_temperature\_40000 — temperature at vertical level at 400 hPa, C
81. 81.

    gfs\_temperature\_45000 — temperature at vertical level at 450 hPa, C
82. 82.

    gfs\_temperature\_5000 — temperature at vertical level at 50 hPa, C
83. 83.

    gfs\_temperature\_50000 — temperature at vertical level at 500 hPa, C
84. 84.

    gfs\_temperature\_55000 — temperature at vertical level at 550 hPa, C
85. 85.

    gfs\_temperature\_60000 — temperature at vertical level at 600 hPa, C
86. 86.

    gfs\_temperature\_65000 — temperature at vertical level at 650 hPa, C
87. 87.

    gfs\_temperature\_7000 — temperature at vertical level at 70 hPa, C
88. 88.

    gfs\_temperature\_70000 — temperature at vertical level at 700 hPa, C
89. 89.

    gfs\_temperature\_75000 — temperature at vertical level at 750 hPa, C
90. 90.

    gfs\_temperature\_80000 — temperature at vertical level at 800 hPa, C
91. 91.

    gfs\_temperature\_85000 — temperature at vertical level at 850 hPa, C
92. 92.

    gfs\_temperature\_90000 — temperature at vertical level at 900 hPa, C
93. 93.

    gfs\_temperature\_92500 — temperature at vertical level at 925 hPa, C
94. 94.

    gfs\_temperature\_95000 — temperature at vertical level at 950 hPa, C
95. 95.

    gfs\_temperature\_97500 — temperature at vertical level at 975 hPa, C
96. 96.

    gfs\_temperature\_sea — temperature at 2m, C
97. 97.

    gfs\_temperature\_sea\_grad — temperature difference adjacent horizons at 2m
98. 98.

    gfs\_temperature\_sea\_interpolated — gfs\_temperature\_sea\_interpolated between horizons, C
99. 99.

    gfs\_temperature\_sea\_next — next horizon temperature at 2m, C
100. 100.

     gfs\_timedelta\_s — difference between gfs and forecast time, s
101. 101.

     gfs\_total\_clouds\_cover\_high — cloud coverage (between horizons, divisible by 6) at high level, %
102. 102.

     gfs\_total\_clouds\_cover\_low — cloud coverage (between horizons, divisible by 6) at low level, %
103. 103.

     gfs\_total\_clouds\_cover\_low\_grad — difference between low level cloud coverage on adjacent horizons, %
104. 104.

     gfs\_total\_clouds\_cover\_low\_next — next horizon cloud coverage (between horizons, divisible by 6) at low level, %
105. 105.

     gfs\_total\_clouds\_cover\_middle — cloud coverage (between horizons, divisible by 6) at middle level, %
106. 106.

     gfs\_u\_wind — 10 meter U wind component, m/s
107. 107.

     gfs\_v\_wind — 10 meter V wind component, m/s
108. 108.

     gfs\_wind\_speed — wind velocity, sqrt(gfs\_u\_wind2 + gfs\_v\_wind2), m/s
109. 109.

     sun\_elevation — sun height proxy above horizon (without corrections for precision and diffraction)
110. 110.

     topography\_bathymetry — height above or below sea level, m
111. 111.

     wrf\_available — is there any data from wrf
112. 112.

     wrf\_graupel — avg graupel rate between two horizons, mm/h
113. 113.

     wrf\_hail — hail velocity on two horizons, mm/h
114. 114.

     wrf\_psfc — pressure, Pa
115. 115.

     wrf\_rain — avg rain rate between two horizons, mm/h
116. 116.

     wrf\_rh2 — relative humidity at 2m, from 0 to 1
117. 117.

     wrf\_snow — avg snow rate between two horizons, mm/h
118. 118.

     wrf\_t2 — temperature at 2m, K
119. 119.

     wrf\_t2\_grad — difference between temperatures at 2m on adjacent horizons, K
120. 120.

     wrf\_t2\_interpolated — wrf\_t2\_interpolated between horizons, K
121. 121.

     wrf\_t2\_next — next horizon temperature at 2m, K
122. 122.

     wrf\_wind\_u — wind U component, m/s
123. 123.

     wrf\_wind\_v — wind V component, m/s

### C.3 Metrics

We aim at comparing different models in terms of uncertainty estimation and robustness to distributional shifts. Several performance metrics are considered.

Predictive Performance For temperature prediction, predictive performance and robustness to distributional shifts are evaluated by measuring RMSE and MAE between predictions and targets: lower the RMSE/MAE score on the test sets, greater the robustness of the models to the distributional shift. For classification, we use accuracy and macro-averaged F1 (one-vs-all averaged with no weighting). More robust models are expected to have higher values of these metrics.

Joint assessment of Uncertainty and Robustness We jointly assess robustness and uncertainty estimation via error-retention and F1-retention curves, described in Section [2](#S2 "2 Evaluation Paradigm, Metrics, and Baselines ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") and detailed in Appendix [A](#A1 "Appendix A Assessment Metrics ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). For regression, we use MSE as the error metric instead of RMSE as it is linear with respect to the error for each datapoint. For the F1-retention curve an acceptable prediction is defined as one where MSE < 1.0. This corresponds to an error of 1 degree or less, which most people cannot feel. Typically people are sensitive to differences in surrounding temperature of over a degree. These two performance metrics are respectively denoted as R-AUC and F1-AUC. For classification, we use the error rate to compute R-AUC. For both classification and regression, a good uncertainty measure is expected to achieve low R-AUC and high F1-AUC. Additionally, the F1 score at a retention rate of 95% of the most certain samples is also quoted and is denoted as F1@95%, which is a single point summary jointly of the uncertainty and robustness. Finally, ROC-AUC is used as a summary statistic for evaluating uncertainty-based out-of-distribution data detection.

### C.4 Training details

The regression models are optimized with the loss function RMSEWithUncertainty [[52](#bib.bib52)] that predicts mean and variance of the normal distribution by optimizing the negative log-likelihood. Each model is constructed with a depth of 8 and then is trained for 20,000 iterations at a learning rate of 0.3. The classification models are optimized with the loss function MultiClass that predicts a discrete probability distribution over all classes. Each model is constructed with a depth of 6 and then is trained for 10,000 iterations at a learning rate of 0.4. Hyperparameter tuning is performed on the dev\_in data for both tasks. All models were trained within under 8 hours using a normal laptop.

### C.5 Additional experiments

In addition to considering ensembles of GBDT models implemented in CatBoost, we additionally consider ensembles of neural models. Specifically, we consider the FT-Transformer model [[78](#bib.bib78)]. We use FT-Transformers as the basis for Monte-Carlo Dropout Ensembles (MCDP) [[13](#bib.bib13)] as well as Deep Ensembles [[14](#bib.bib14)]. Additionally, we consider combining ensembles of CatBoost models with a Deep Ensemble of FT-Transformer models. Predictive performance figures are presented in table [9](#A3.T9 "Table 9 ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). Here, we can see that ensembles of CatBoost models and Deep ensembles of FT-Transformer models have very similar performance, with the latter marginally outperforming the former. However, their combination yields the most competitive figures. These results are consistent for both the classification and regression tasks.

Table 9: Predictive performance for Weather prediction. Mean is quoted for the single models.

| Dataset | Model | Regression | | | | | | Classification | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMSE ↓↓\downarrow | | | MAE ↓↓\downarrow | | | Accuracy (%) ↑↑\uparrow | | | Macro F1 (%) ↑↑\uparrow | | |
| In | Shifted | Full | In | Shifted | Full | In | Shifted | Full | In | Shifted | Full |
| dev | CatBoost, Single | 1.59 | 2.30 | 1.98 | 1.18 | 1.75 | 1.47 | 67.0 | 47.5 | 57.2 | 42.2 | 20.2 | 36.8 |
| CatBoost, Ensemble | 1.51 | 2.12 | 1.84 | 1.11 | 1.61 | 1.36 | 68.5 | 50.3 | 59.4 | 42.3 | 21.3 | 37.2 |
| FT-Transformer, Single | 1.61 | 2.13 | 1.89 | 1.18 | 1.61 | 1.39 | 67.2 | 49.4 | 58.3 | 39.4 | 20.7 | 34.9 |
| FT-Transformer, MCDP | 1.59 | 2.09 | 1.84 | 1.16 | 1.58 | 1.37 | 67.2 | 50.0 | 58.6 | 39.3 | 21.4 | 34.9 |
| FT-Transformer, Ensemble | 1.50 | 2.01 | 1.77 | 1.10 | 1.52 | 1.31 | 68.8 | 51.5 | 60.2 | 40.5 | 21.6 | 36.0 |
|  | CatBoost ⊕direct-sum\oplus FT-Transformer | 1.47 | 2.01 | 1.76 | 1.08 | 1.53 | 1.30 | 69.3 | 51.5 | 60.4 | 42.4 | 21.4 | 37.3 |
| eval | CatBoost, Single | 1.60 | 2.60 | 2.16 | 1.19 | 1.91 | 1.56 | 66.7 | 44.5 | 55.5 | 42.9 | 21.5 | 34.4 |
| CatBoost, Ensemble | 1.52 | 2.37 | 2.00 | 1.11 | 1.75 | 1.44 | 68.2 | 46.7 | 57.3 | 44.1 | 22.2 | 35.5 |
| FT-Transformer, Single | 1.62 | 2.40 | 2.05 | 1.18 | 1.77 | 1.48 | 67.0 | 45.9 | 56.3 | 37.6 | 23.0 | 31.4 |
| FT-Transformer, MCDP | 1.59 | 2.34 | 2.01 | 1.17 | 1.73 | 1.45 | 67.0 | 46.4 | 56.6 | 37.6 | 23.2 | 31.6 |
| FT-Transformer, Ensemble | 1.51 | 2.24 | 1.92 | 1.10 | 1.66 | 1.38 | 68.6 | 48.0 | 58.1 | 38.2 | 23.8 | 32.1 |
|  | CatBoost ⊕direct-sum\oplus FT-Transformer | 1.48 | 2.25 | 1.91 | 1.08 | 1.66 | 1.38 | 69.0 | 48.0 | 58.4 | 44.2 | 22.3 | 35.6 |

We jointly assess robustness and uncertainty quality for the additional baselines in the table below. Again, the result show that combining all models yields the best results. Curiously, the results also show that Monte-Carlo dropout ensembles are now competitive with CatBoost ensembles. This suggests that the uncertainty quality of MCDP is better than for CatBoost ensembles, even if CatBoost has the better raw predictive quality.

Table 10: Retention performance for Weather prediction. Mean is quoted for the single models.

| Dataset | Model | Regression | | | Classification | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | R-AUC ↓↓\downarrow | F1-AUC (%) ↑↑\uparrow | F1@959595% ↑↑\uparrow | R-AUC ↓↓\downarrow | F1-AUC (%) ↑↑\uparrow | F1@959595% ↑↑\uparrow |
| dev | CatBoost, Single | 1.894 | 44.35 | 62.72 | 0.1666 | 57.72 | 73.04 |
| CatBoost, Ensemble | 1.227 | 52.20 | 65.83 | 0.1522 | 59.07 | 74.86 |
| FT-Transformer, Single | 1.245 | 51.69 | 65.08 | 0.1592 | 58.51 | 73.80 |
| FT-Transformer, MCDP | 1.197 | 52.08 | 65.62 | 0.1565 | 58.80 | 74.16 |
| FT-Transformer, Ensemble | 1.051 | 53.66 | 67.56 | 0.1472 | 59.54 | 75.38 |
|  | CatBoost ⊕direct-sum\oplus FT-Transformer | 1.035 | 54.04 | 67.47 | 0.1453 | 59.71 | 75.58 |
| eval | CatBoost, Single | 2.320 | 43.41 | 61.89 | 0.1799 | 56.25 | 71.56 |
| CatBoost, Ensemble | 1.335 | 52.36 | 64.72 | 0.1640 | 58.22 | 73.17 |
| FT-Transformer, Single | 1.386 | 51.86 | 63.96 | 0.1705 | 57.72 | 72.17 |
| FT-Transformer, MCDP | 1.321 | 52.29 | 64.57 | 0.1676 | 58.04 | 72.55 |
| FT-Transformer, Ensemble | 1.168 | 53.77 | 66.40 | 0.1576 | 58.95 | 73.84 |
|  | CatBoost ⊕direct-sum\oplus FT-Transformer | 1.151 | 54.09 | 66.28 | 0.1561 | 59.07 | 74.02 |

Finally, we examine the quality of different uncertainty measures which are derivable from all of the baseline models. The results are provided in Table [11](#A3.T11 "Table 11 ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"). The results show an interesting trend, where the model which has the best joint uncertainty and robustness performance is a combination of CatBoost and FT-Transformer ensembles, and the best measure of uncertainty is total variance and confidence for regression and classification, respectively. Both are measures of *total uncertainty*. At the same time, the best model for anomaly detection is a catboost ensemble using measures of *knowledge uncertainty*. This highlights how the best model and uncertainty measure to use greatly depends on the task.

Table 11: Comparing ensembled F1-AUC and ROC-AUC for various uncertainty measures on the tests sets from the canonical partitioning of Weather Prediction dataset for regression and classification.

| Data | Metric | Model | Regression | | | Classification | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Total Unc. | Knowledge Unc. | | Total Unc. | | Knowledge Unc. | | |
| tvar | varm | EPKL | Conf | Entropy | MI | EPKL | RMI |
| dev | F1-AUC (%) ↑↑\uparrow | CatBoost | 52.20 | 50.12 | 50.51 | 59.07 | 58.86 | 57.72 | 57.69 | 57.66 |
| FT-Transformer | 53.66 | 51.86 | 53.53 | 59.54 | 59.13 | 56.36 | 56.28 | 56.20 |
| CatBoost ⊕direct-sum\oplus FT-Transformer | 54.04 | 52.22 | 51.49 | 59.71 | 59.26 | 57.65 | 57.49 | 57.36 |
| ROC-AUC (%) ↑↑\uparrow | CatBoost | 62.96 | 82.31 | 85.29 | 63.98 | 65.00 | 83.75 | 83.96 | 84.12 |
| FT-Transformer | 58.10 | 65.89 | 61.63 | 35.46 | 65.48 | 71.89 | 71.85 | 71.79 |
| CatBoost ⊕direct-sum\oplus FT-Transformer | 62.73 | 76.63 | 83.29 | 34.63 | 66.10 | 80.46 | 80.10 | 79.78 |
| eval | F1-AUC (%) ↑↑\uparrow | CatBoost | 52.36 | 49.81 | 50.40 | 58.22 | 57.89 | 56.99 | 56.96 | 56.93 |
| FT-Transformer | 53.77 | 51.83 | 53.58 | 58.95 | 58.55 | 55.68 | 55.59 | 55.51 |
| CatBoost ⊕direct-sum\oplus FT-Transformer | 54.09 | 52.12 | 51.44 | 59.07 | 58.62 | 56.92 | 56.75 | 56.59 |
| ROC-AUC (%) ↑↑\uparrow | CatBoost | 65.99 | 78.32 | 79.90 | 66.20 | 66.76 | 83.44 | 83.59 | 83.68 |
| FT-Transformer | 65.03 | 68.78 | 67.67 | 30.68 | 70.37 | 76.46 | 76.43 | 76.36 |
| CatBoost ⊕direct-sum\oplus FT-Transformer | 67.78 | 75.43 | 79.29 | 30.86 | 69.92 | 82.49 | 82.16 | 81.85 |

!(/html/2107.07455/assets/figures/weather_single_retention_dev_mse.png)

(a) CatBoost, Regression, MSE.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_f1.png)

(b) CatBoost, Regression, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_acc_class.png)

(c) CatBoost, Classification, error rate.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_f1_class.png)

(d) CatBoost, Classification, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_mse_ftt.png)

(e) FT-Trans, Regression, MSE.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_f1_ftt.png)

(f) FT-Trans, Regression, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_acc_class_ftt.png)

(g) FT-Trans, Classification, error rate.

!(/html/2107.07455/assets/figures/weather_single_retention_dev_f1_class_ftt.png)

(h) FT-Trans, Classification, F1.

Figure 11: Retention curves for CatBoost and FT-Transformer on dev for the canonical Weather prediction dataset.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_mse.png)

(a) CatBoost, Regression, MSE.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1.png)

(b) CatBoost, Regression, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_acc_class.png)

(c) CatBoost, Classification, error rate.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1_class.png)

(d) CatBoost, Classification, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_mse_ftt.png)

(e) FT-Trans, Regression, MSE.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1_ftt.png)

(f) FT-Trans, Regression, F1.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_acc_class_ftt.png)

(g) FT-Trans, Classification, error rate.

!(/html/2107.07455/assets/figures/weather_single_retention_eval_f1_class_ftt.png)

(h) FT-Trans, Classification, F1.

Figure 12: Retention curves with CatBoost and FT-Transformer on eval for the canonical Weather prediction dataset.

#### C.5.1 Further experiments

Figure [13](#A3.F13 "Figure 13 ‣ C.5.1 Further experiments ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") depicts additional splits beyond the canonical partition of the tabular weather data. Table [12](#A3.T12 "Table 12 ‣ C.5.1 Further experiments ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") summarises the experiments to be performed with a brief description of what each experiment involves. All experiments are to be performed using CatBoost for both the regression and classification tasks. These experiments aim to better understand whether time or climate shift in the data leads to a greater performance drop from in-domain to shifted datasets. Hence, the focus here is on robustness only. The corresponding results for each experiment are given in Table [13](#A3.T13 "Table 13 ‣ C.5.1 Further experiments ‣ C.5 Additional experiments ‣ Appendix C Tabular Weather ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

!(/html/2107.07455/assets/figures/tabular_splits_aug.png)

Figure 13: Extended splits of tabular weather data.

Table 12: Description of additional experiments.

| Exp | Training set | Development set | Description |
| --- | --- | --- | --- |
| A | train | dev\_in | Time & climate shifts |
| B | train ⊕direct-sum\oplus train\_xclim | dev\_in ⊕direct-sum\oplus dev\_xclim | Time shift |
| C | train ⊕direct-sum\oplus train\_xtime | dev\_in ⊕direct-sum\oplus dev\_xtime | Climate shift |
| D | train ⊕direct-sum\oplus train\_xclim ⊕direct-sum\oplus train\_xtime | dev\_in ⊕direct-sum\oplus dev\_xclim ⊕direct-sum\oplus dev\_xtime | No shift |

Table 13: Predictive performance for Weather prediction using different training sets. Mean is quoted for the single models.

| Dataset | Model | Regression | | | | | | | | Classification | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMSE ↓↓\downarrow | | | | MAE ↓↓\downarrow | | | | Accuracy (%) ↑↑\uparrow | | | | Macro F1 (%) ↑↑\uparrow | | | |
| A | B | C | D | A | B | C | D | A | B | C | D | A | B | C | D |
| dev\_in | CatBoost, Single | 1.59 | 1.62 | 1.61 | 1.63 | 1.18 | 1.21 | 1.20 | 1.21 | 67.0 | 66.1 | 66.6 | 65.8 | 42.2 | 39.6 | 42.6 | 39.9 |
| CatBoost, Ensemble | 1.51 | 1.52 | 1.51 | 1.54 | 1.11 | 1.12 | 1.11 | 1.14 | 68.5 | 67.2 | 67.7 | 66.8 | 42.3 | 40.1 | 41.9 | 39.9 |
| dev\_out | CatBoost, Single | 2.30 | 2.30 | 2.04 | 1.95 | 1.75 | 1.75 | 1.54 | 1.48 | 47.5 | 50.9 | 51.8 | 54.0 | 20.2 | 21.2 | 21.4 | 22.7 |
| CatBoost, Ensemble | 2.12 | 2.05 | 1.93 | 1.85 | 1.61 | 1.55 | 1.45 | 1.40 | 50.3 | 53.4 | 53.0 | 55.4 | 21.3 | 22.2 | 21.8 | 22.9 |
| eval\_in | CatBoost, Single | 1.60 | 1.63 | 1.62 | 1.64 | 1.19 | 1.21 | 1.20 | 1.22 | 66.7 | 65.9 | 66.3 | 65.7 | 42.9 | 40.4 | 42.7 | 40.3 |
| CatBoost, Ensemble | 1.52 | 1.53 | 1.52 | 1.55 | 1.11 | 1.12 | 1.12 | 1.14 | 68.2 | 67.1 | 67.6 | 66.7 | 44.1 | 42.0 | 43.8 | 41.3 |
| eval\_out | CatBoost, Single | 2.60 | 2.62 | 2.28 | 2.15 | 1.91 | 1.93 | 1.69 | 1.62 | 44.5 | 48.3 | 48.6 | 51.5 | 21.5 | 23.8 | 23.5 | 25.6 |
| CatBoost, Ensemble | 2.37 | 2.26 | 2.16 | 2.04 | 1.75 | 1.69 | 1.60 | 1.53 | 46.7 | 50.4 | 50.2 | 53.0 | 22.2 | 24.1 | 24.1 | 26.0 |

## Appendix D Machine Translation

The current appendix contains a description of the composition, collection, pre-processing and partitioning of the Shifts Machine Translation dataset. Additionally, it contains a description of the metrics used for assessment and an expanded set of experimental results.

### D.1 Dataset Description

##### Composition

The Shifts Machine Translation datasets consists of a training, development (dev) and evaluation (eval) set. Each set consists of pairs of source and target sentences in English and Russian, respectively. As most production NMT systems are built using a variety of general purpose corpora, we do not provide a new training corpus, rather, we will use the freely available WMT’20 English-Russian corpus. This data covers a variety of domains, but primarily focuses on parliamentary and news data. For the most part, this data is grammatically and orthographically correct and language use is formal. This is representative of the type of data used, for example, to build the Yandex.Translate NMT system. The composition of the WMT’20 En-Ru corpus is detailed on the workshop for machine translation website here: <http://www.statmt.org/wmt20/translation-task.html>. For simplicity of access and archiving purposes we downloaded the WMT’20 En-Ru training data set and also made it available on the Shifts Dataset and Challenge GitHub here: <https://github.com/yandex-research/shifts>.

The dev and eval datasets consist of an “in-domain” partition matched to the training data, and an “out-of-distribution”, or shifted partition, which contains examples of atypical language usage. We select the English-Russian Newstest’19 as the in-domain *development set* and will use a new corpus of news data collected from GlobalVoices News service [[54](#bib.bib54)] and manually annotated using expert human translators as the in-domain *evaluation set*. For the shifted development and evaluation data we use the Reddit corpus prepared for the WMT’19 robustness challenge [[43](#bib.bib43)]. This data contains examples of slang, acronyms, lack of punctuation, poor orthography, concatenations, profanity, and poor grammar, among other forms of atypical language usage. This data is representative of the types of inputs that machine translation services find challenging. As Russian target annotations are not available, we pass the data through a two-stage process, where orthographic, grammatical and punctuation mistakes are corrected, and the source-side English sentences are translated into Russian by expert in-house Yandex translators. The development set is constructed from the same 1400-sentence test-set used for the WMT’19 robustness challenge. For the heldout evaluation set we use the open-source MTNT crawler which connects to the Reddit API to collect a further set of 3,000 English sentences from Reddit, which is similarly corrected and translated. Note that the Reddit data has comments made by users, but no personal identification data (login, name, etc…) or other user identification data was recorded or stored - the dataset only only contains the raw comments made on a public discussion platform. In terms of size, these development and evaluation sets are comparable or larger to the ones used in the WMT challenges and for evaluating productions systems.

Table 14: NMT Data Description - All Data is English-Russian

| Data Set | N. Sentences | Avg. Sentence Length | | Type |
| --- | --- | --- | --- | --- |
| En | Ru |
| WMT’20 | 62M | 23,9 | 20.9 | Train |
| NWT’19 | 1997 | 24.5 | 24.7 | In-domain Dev |
| GlobalVoices | 3,000 | 25.1 | 24.1 | In-domain Eval |
| WMT’19 MTNT Reddit | 1,362 | 17.2 | 16.5 | Shifted Dev |
| Shifts Reddit | 3,063 | 16.1 | 16.4 | Shifted Eval |

Both the development and evaluation Reddit data was manually annotated by members of the Yandex.Translate team with the following 7 non-exclusive anomaly flags:

* •

  Punctuation anomalies: Some punctuation marks are missed or used incorrectly or some formatting (like Wiki markup) is used in the sentence.
* •

  Spelling anomalies: The sentence contains spelling errors, including incorrect concatenation of two words as well as incorrect use of hyphens.
* •

  Capitalization anomalies: Words that should be capitalized according to the language rules are written in lower case or vice versa.
* •

  Fluency anomalies: The sentence is non-fluent due to wrong or missing prepositions, pronouns or ungrammatical form choice.
* •

  Slang anomalies: In the sentence there are slang words of abbreviations like “idk” for “I don’t know” or “cuz” for “because”.
* •

  Emoji anomalies: The sentence contains emojis either at the end of it, or instead of some words.
* •

  Tags anomalies: The sentence contains markup for usernames or code like “r/username”.

An analysis of the occurrence and co-occurrence of these anomalies is provided in figure [14](#A4.F14 "Figure 14 ‣ Composition ‣ D.1 Dataset Description ‣ Appendix D Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").

!(/html/2107.07455/assets/figures/anomaly_fraction.png)

!(/html/2107.07455/assets/figures/anomaly_cooccurence.png)

Figure 14: Analysis of anomaly occurrence and co-occurrence in Reddit (shifted) development and evaluation data

##### Collection Process

GlobalVoices[[54](#bib.bib54)] data was crawled for parallel news articles in English and Russian using internal Yandex tools. The raw articles were manually split into sentence-pairs by in-house Yandex assessors. A full set of 30000 sentence pairs was produced, from which a subset of 3000 sentences was uniformly randomly sampled. Reddit data was crawled using the open-source MTNT [[43](#bib.bib43)] crawler from <https://github.com/pmichel31415/mtnt>. This crawler links in with the Reddit API to allow mining and crawling Reddit for data. The crawler collected a set of 100K user comments which were then split into sentences using the NLTK toolkit. Then a set of 3500 sentences was randomly uniformly selected. After pre-processing and cleaning a set of 3065 sentences was produced.

##### Preprocessing and Cleaning

For the GlobalVoices data parallel sentences markup was done manually by in-house Yandex assessors; non-parallel sentences were removed from dataset. For Reddit data 1-word phrases and sentences consisting only of non-alphabetical symbols were removed. Professional editors were used to manually correct grammatical and orthographic mistakes prior to translating into Russian, but were explicitly told to maintain the non-formal style as much as possible. This error correction was used only for obtaining target-side Russian translation.

##### Guidelines on ethical use

Users are discouraged from attempting to discover to which Reddit users the comments belong by manually or automatically crawling through Reddit to find the comments.

##### Format

This dataset is provided in raw text format and a TSV with metadata for the dev and eval reddit data.

##### License

The Shifts Machine Translation dataset is released under a mixed licence. GlobalVoices evaluation data is released under CC BY NC SA 4.0 . The source-side text for the Reddit development and evaluation datasets exist under terms of the Reddit API. The target side Russian sentences were obtained by Yandex via in-house professional translators and are released under CC BY NC SA 4.0. We highlight that the development set source sentences are the same ones as used in the MTNT dataset.

### D.2 Metrics

To evaluate the performance of our models we will consider the following two metrics : corpus-level BLEU [[55](#bib.bib55)] and sentence-level GLEU [[56](#bib.bib56), [57](#bib.bib57), [58](#bib.bib58)]. GLEU is an analogue of BLEU which is stable when computed at the level of individual sentences. Thus, it is far more useful at evaluating system performance on a per-sample basis, rather than at the level of an entire corpus. Note that GLEU correlates strongly with BLEU at the corpus level.

Machine translation is inherently a multi-modal task, as a sentence can be translated in multiple equally valid ways. Furthermore, translation systems often yield multiple translation hypothesis. To account for this we will consider two GLEU-based metrics for evaluating translation quality. First is the *expected GLEU* or eGLEU across all translation hypotheses returned by a translation models. Each hypothesis is assumed to be assigned a *confidence score*, and confidences across each hypotheses by sum to one. This is our primary assessment metric:

|  |  |  |  |
| --- | --- | --- | --- |
|  | eGLEU=1N​∑i=1N∑h=1HGLEUi,h⋅wi,h,wi,h>0,∑h=1Hwh=1formulae-sequenceeGLEU1𝑁superscriptsubscript𝑖1𝑁superscriptsubscriptℎ1𝐻⋅subscriptGLEU  𝑖ℎsubscript𝑤  𝑖ℎformulae-sequencesubscript𝑤  𝑖ℎ0superscriptsubscriptℎ1𝐻subscript𝑤ℎ1\text{eGLEU}=\frac{1}{N}\sum\_{i=1}^{N}\sum\_{h=1}^{H}\text{GLEU}\_{i,h}\cdot w\_{i,h},\quad w\_{i,h}>0,\sum\_{h=1}^{H}w\_{h}=1 |  | (4) |

Additionally, we will consider the *maximum GLEU* or maxGLEU across all hypothesis, which represents an upper bound on performance, given a model can appropriately rank it’s hypotheses:

|  |  |  |  |
| --- | --- | --- | --- |
|  | maxGLEU=1N​∑i=1Nmaxh⁡[GLEUi,h]maxGLEU1𝑁superscriptsubscript𝑖1𝑁subscriptℎsubscriptGLEU  𝑖ℎ\text{maxGLEU}=\frac{1}{N}\sum\_{i=1}^{N}\max\_{h}\big{[}\text{GLEU}\_{i,h}\big{]} |  | (5) |

Finally, in order to calculate area under the error retention curve we need to introduce an *error metric*, where lower error is better. This is trivially done by introducing *eGLEU error*, which defined as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | eGLEU Error=100−eGLEUeGLEU Error100eGLEU\text{eGLEU Error}=100-\text{eGLEU} |  | (6) |

Thus, in section [4](#S4 "4 Machine Translation ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), area under the error retention curve (R-AUC), as well as the F1 metric for detecting ‘valid predictions’ will be calculated using eGLEU Error.

### D.3 Training details

Training data was standard used the standard perl-based script provided in Fairseq [[59](#bib.bib59)] examples. Duplicate sentence pairs as well as sentence pairs where source and target text matched were removed. Models were trained using Fairseq version 0.8. A full description and for from preprocessing and training is provided [here](https://github.com/yandex-research/shifts/tree/main/translation). All models were trained used 8xV100 GPUs over roughly 48 hours.

### D.4 Additional Results

!(/html/2107.07455/assets/figures/nmt_error_retention_dev.png)

(a) Dev

!(/html/2107.07455/assets/figures/nmt_error_retention_eval.png)

(b) Eval

!(/html/2107.07455/assets/figures/nmt_F1_retention_dev.png)

(c) Dev

!(/html/2107.07455/assets/figures/nmt_F1_retention_eval.png)

(d) Eval

Figure 15: Location of samples from canonical partitioning of Weather Prediction dataset.

## Appendix E Vehicle Motion Prediction

The current appendix contains a description of the composition, collection, pre-processing and partitioning of the Shifts Vehicle Motion Prediction dataset. Additionally, it contains a description of the metrics used for assessment and an expanded set of experimental results.

### E.1 Dataset Description

Table 15: A comparison of various motion prediction datasets. The Shifts Vehicle Motion Prediction dataset is the largest by number of scenes and total size in hours.

| Dataset | Scene Length (s) | # Scenes | | | Total Size (h) | Avg. # Actors |
| --- | --- | --- | --- | --- | --- | --- |
| Train | Dev | Eval |
| Argoverse | 5 | 205,942 | 39,472 | 78,143 | 320 | 50 |
| Lyft | 25 | 134,000 | 11,000 | 16,000 | 1,118 | 79 |
| Waymo | 20 | 72,347 | 15,503 | 15,503 | 574 | - |
| Shifts | 10 | 500,000 | 50,000 | 50,000 | 1,667 | 29 |

##### Composition

The dataset for the Vehicle Motion Prediction task was collected by the Yandex Self-Driving Group (SDG) fleet.
This is the largest vehicle motion prediction dataset released to date, containing 600,000 scenes (see [Table 15](#A5.T15 "In E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for a comparison to other public datasets). The dataset consists of scenes spanning six locations, three seasons, three times of day, and four weather conditions (cf. [Table 16](#A5.T16 "In Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") and [17](#A5.T17 "Table 17 ‣ Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")). Each of these conditions is available in the form of tags associated with every scene. Each scene is 10 seconds long and is divided into 5 seconds of context features and 5 seconds of ground truth targets for prediction, separated by the time T=0𝑇0T=0. The goal of the task is to predict the movement trajectory of vehicles at time T∈(0,5]𝑇05T\in\left(0,5\right] based on the information available for time T∈[−5,0]𝑇50T\in\left[-5,0\right].

Table 16: The number of scenes in the Vehicle Motion Prediction dataset by location and season.

| Location | Train | Dev | Eval |
| --- | --- | --- | --- |
| Moscow | 450,504 | 30,505 | 30,534 |
| Skolkovo | 6,283 | 2,218 | 2,956 |
| Innopolis | 15,086 | 5,164 | 5,016 |
| Ann Arbor | 19,349 | 8,290 | 6,617 |
| Modiin | 3,502 | 2,262 | 1,555 |
| Tel Aviv | 5,276 | 1,561 | 3,322 |

|  |  |  |  |
| --- | --- | --- | --- |
| Season | Train | Dev | Eval |
| Summer | 85,698 | 10,634 | 10,481 |
| Autumn | 126,845 | 15,290 | 15,840 |
| Winter | 287,457 | 24,076 | 23,679 |
| Spring | 0 | 0 | 0 |

Each scene includes information about the state of dynamic objects (i.e., vehicles, pedestrians) and an HD map.
Each vehicle is described by its position, velocity, linear acceleration, and orientation (yaw, known up to ±πplus-or-minus𝜋\pm\pi).
A pedestrian state consists of a position vector and a velocity vector. All state components are represented in a common coordinate frame and sampled at 5Hz frequency by the perception stack running on the Yandex SDG fleet. The HD map includes lane information (e.g., traffic direction, lane priority, speed limit, traffic light association), road boundaries, crosswalks, and traffic light states, which are also sampled at 5Hz. To facilitate easy use of this dataset, we provide utilities to render scene information as a feature map, which can be used as an input to a standard vision model (e.g., a ResNet [[79](#bib.bib79)]). Our utilities represent each scene as a birds-eye-view image with each channel corresponding to a particular feature (e.g., a vehicle occupancy map) at a particular timestep. We also provide pre-rendered feature maps for every prediction request (cf. [Section E.2](#A5.SS2 "E.2 Task Setup ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")) in the dataset, which are used to train the baseline models. The maps are 128×128128128128\times 128 pixels in size with each pixel covering 1 square meter, have 171717 channels describing both HD map information and dynamic object states at time T=0𝑇0T=0, and are centered with respect to the agent for which a prediction is being made. Researchers working with the dataset are free to use these feature maps, use the provided utilities to render another set of feature maps at different (earlier) timesteps, or construct their own scene representations from the raw data.

The ground truth part of a scene contains future states of dynamic objects sampled at 5Hz for a total of 252525 state samples. Some objects might not have all 252525 states available due to occlusions or imperfections of the on-board perception system.

Table 17: The number of scenes in the Vehicle Motion Prediction dataset by precipitation and time of day.

| Precipitation Type | Train | Dev | Eval |
| --- | --- | --- | --- |
| No | 432,598 | 44,799 | 44,274 |
| Rain | 15,618 | 1,857 | 1,751 |
| Sleet | 15,210 | 1,082 | 990 |
| Snow | 36,574 | 2,262 | 2,985 |

| Sun Phase | Train | Dev | Eval |
| --- | --- | --- | --- |
| Astronomical Night | 171,867 | 13,164 | 13,113 |
| Daylight | 299,065 | 33,879 | 33,979 |
| Twilight | 29,068 | 2,957 | 2,908 |

A number of vehicles in the scene are labeled as *prediction requests*. These are the vehicles that are visible at the most recent time T=0𝑇0T=0 in the context features part of a scene, and therefore would call for a prediction in a deployed system.
For such vehicles we provide not only their future trajectories, but also a number of non–mutually exclusive tags (detailed in [Table 18](#A5.T18 "In Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")) describing the associated maneuver in more detail – whether the vehicle is turning, accelerating, slowing down, etc. – for a total of 101010 maneuver types. Note that some prediction requests may not have all 252525 state samples available.
We call prediction requests with fully-observed state *valid* prediction requests and propose to evaluate predictions only on those.

Table 18: Number of actor maneuvers of the respective type.

| Maneuver Type | Train | Dev | Eval |
| --- | --- | --- | --- |
| Move Left | 254,843 | 25,049 | 25,820 |
| Move Right | 322,231 | 30,074 | 30,633 |
| Move Forward | 5,032,724 | 395,467 | 413,920 |
| Move Back | 54,677 | 4,811 | 4,891 |
| Acceleration | 2,473,750 | 206,977 | 215,009 |
| Deceleration | 2,050,186 | 168,550 | 174,477 |
| Uniform Movement | 6,369,920 | 566,083 | 573,033 |
| Stopping | 441,619 | 38,411 | 39,336 |
| Starting | 739,143 | 64,986 | 65,759 |
| Stationary | 4,620,678 | 433,161 | 433,576 |

In order to study the effects of distributional shift, as well as assess the robustness and uncertainty estimation of baseline models, we divide the Vehicle Motion Prediction dataset such that there are *in-domain* partitions which match the location and precipitation type of the training set, and *out-of-domain* or *shifted* partitions which do not match the training data along one or more of those axes. Furthermore, we provide a *development* set which acts as a validation set, and an *evaluation* set which acts as the test set. For standardized benchmarking we define a *canonical partitioning* of the full dataset (cf. Figure [16](#A5.F16 "Figure 16 ‣ Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), [Table 19](#A5.T19 "In Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")) as the following. The training, in-domain development, and in-domain evaluation data are taken from Moscow. Distributionally shifted development data is taken from Skolkovo, Modiin, and Innopolis. Distributionally shifted evaluation data is taken from Tel Aviv and Ann Arbor.
In addition, we remove all cases of precipitation from the in-domain training, development, and evaluation sets, while distributionally shifted datasets include precipitation.
The canonical partitioning is fully described in Figure [16](#A5.F16 "Figure 16 ‣ Composition ‣ E.1 Dataset Description ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").
This partitioning is also the one used in the Shifts Challenge.

!(/html/2107.07455/assets/figures/SDC_data_partition.png)

Figure 16: The canonical partitioning of the Vehicle Motion Prediction dataset.

Table 19: The number of scenes in the canonical dataset partitioning.

| Dataset Partition | In-Distribution | Distributionally Shifted |
| --- | --- | --- |
| Train | 388,406 | - |
| Development | 27,036 | 9,569 |
| Evaluation | 26,865 | 9,939 |

##### Collection Process

The Vehicle Motion Prediction data was collected by the perception system running onboard a number of self-driving vehicles equipped with LiDAR sensors, radars, and cameras. This perception system consists of a number of neural network–based detectors followed by an object tracker that fuses detections across sensor modalities and time.
The provided HD map for each location has been constructed and validated by cartographers employed by Yandex SDG.
The provided dataset was sampled from a much larger dataset collected over a course of 888 months. The sampling procedure was biased towards sampling scenes on which the motion prediction system currently used by the SDC fleet makes mistakes, as well as sampling more scenes from locations where the fleet drives less frequently.

##### Preprocessing and Cleaning

The collected dataset has been cleaned from scenes in which:

* •

  any kind of onboard system failure was detected, as the perception system output can potentially be unreliable in such scenes;
* •

  the perception system has produced outputs that clearly violate physical constraints, such as actors having unrealistic acceleration or colliding with one other.

##### Format

This dataset is provided in protobuf format.

##### License

We release this dataset under the CC BY NC SA 4.0 license.

### E.2 Task Setup

Vehicle Motion Prediction is a complex task and therefore must be described in detail. We provide a training dataset 𝒟train={(𝒙i,𝒚i)}i=1Nsubscript𝒟trainsuperscriptsubscriptsubscript𝒙𝑖subscript𝒚𝑖𝑖1𝑁\mathcal{D}\_{\textup{train}}=\{(\bm{x}\_{i},\bm{y}\_{i})\}\_{i=1}^{N} of time-profiled ground truth trajectories (i.e., plans) 𝒚𝒚\bm{y} paired with high-dimensional observations (features) 𝒙𝒙\bm{x} of the corresponding scenes. Each 𝒚=(s1,…,sT)𝒚subscript𝑠1…subscript𝑠𝑇\bm{y}=(s\_{1},\dots,s\_{T}) corresponds to the trajectory of a given vehicle observed through the SDG perception stack. Each state stsubscript𝑠𝑡s\_{t} corresponds to the x- and y-displacement of the vehicle at timestep t𝑡t, s.t. 𝒚∈ℝT×2𝒚superscriptℝ𝑇2\bm{y}\in\mathbb{R}^{T\times 2}. We consider the performance of models on development and evaluation datasets 𝒟devj={(𝒙i,𝒚i)}i=1Mjsubscriptsuperscript𝒟𝑗devsuperscriptsubscriptsubscript𝒙𝑖subscript𝒚𝑖𝑖1subscript𝑀𝑗\mathcal{D}^{j}\_{\textup{dev}}=\{(\bm{x}\_{i},\bm{y}\_{i})\}\_{i=1}^{M\_{j}}. and 𝒟evalj={(𝒙i,𝒚i)}i=1Mjsubscriptsuperscript𝒟𝑗evalsuperscriptsubscriptsubscript𝒙𝑖subscript𝒚𝑖𝑖1subscript𝑀𝑗\mathcal{D}^{j}\_{\textup{eval}}=\{(\bm{x}\_{i},\bm{y}\_{i})\}\_{i=1}^{M\_{j}}. See Figure [17](#A5.F17 "Figure 17 ‣ Per-Trajectory Confidence Scores. ‣ E.2 Task Setup ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for a depiction of the task.

##### Prediction Requests.

There are N𝑁N (Mjsubscript𝑀𝑗M\_{j}) prediction requests in the training dataset (evaluation datasets), with many requests for each scene corresponding to the many different vehicle trajectories observed. For example, in the canonical partition of the data, there are 388,406 scenes in the training dataset (Moscow, no precipitation), and 5,649,675 valid prediction requests.

Models can be trained to make use of ground truth trajectories that contain occlusions (i.e., prediction requests that are not valid) during training, such as through linear interpolation of missing steps. However, for the baseline methods considered in this work, both training and evaluation are done using only the fully observed ground truth trajectories.

Next, we describe the two levels of uncertainty quantification that we consider for each prediction request in the proposed task: per-trajectory and per–prediction request uncertainty scores.

##### Per-Trajectory Confidence Scores.

Like machine translation, motion prediction is an inherently multimodal task. A motion prediction model can produce a different number of sampled trajectories (plans) Disubscript𝐷𝑖D\_{i} for each input 𝒙isubscript𝒙𝑖\bm{x}\_{i}; in other words, for two inputs 𝒙i,𝒙j

subscript𝒙𝑖subscript𝒙𝑗\bm{x}\_{i},\bm{x}\_{j} with i≠j𝑖𝑗i\neq j, Disubscript𝐷𝑖D\_{i} and Djsubscript𝐷𝑗D\_{j} can differ.
As a justification, consider that in a certain context, multiple trajectories may be desirable to capture multimodality (e.g., the vehicle of interest is at a T-junction), and in others a single or fewer trajectories would be sufficient (e.g., the vehicle is clearly proceeding straight). In our task, we expect a stochastic model to accompany its Disubscript𝐷𝑖D\_{i} predicted trajectories on a given input 𝒙isubscript𝒙𝑖\bm{x}\_{i} with scalar per-trajectory confidence scores ci(d),d∈{1,…​Di}

superscriptsubscript𝑐𝑖𝑑𝑑
1…subscript𝐷𝑖c\_{i}^{(d)},d\in\{1,\dots D\_{i}\}. These provide an ordering of the plausibility of the various trajectories predicted for a given input. The scores must be non-negative and sum to 111 (i.e., form a valid probability distribution).

!(/html/2107.07455/assets/figures/motion_prediction_model_2.png)

Figure 17: Diagram of the Vehicle Motion Prediction task. Models take as input a single scene context 𝒙𝒙\bm{x} composed of static (HD map) and time-dependent input features, and predict trajectories {𝒚(d)∣d∈1,…,D}conditional-setsuperscript𝒚𝑑𝑑

1…𝐷\{\bm{y}^{(d)}\mid d\in 1,\dots,D\} with corresponding per-trajectory confidence scores {𝒄(d)∣d∈1,…,D}conditional-setsuperscript𝒄𝑑𝑑

1…𝐷\{\bm{c}^{(d)}\mid d\in 1,\dots,D\}, as well as a single per–prediction request uncertainty score U𝑈U.

##### Per–Prediction Request Uncertainty Score.

We also expect models to produce scalar uncertainty estimates corresponding to each prediction request input 𝒙isubscript𝒙𝑖\bm{x}\_{i}.
For example, on evaluation dataset 𝒟evaljsubscriptsuperscript𝒟𝑗eval\mathcal{D}^{j}\_{\textup{eval}}, we have Mjsubscript𝑀𝑗M\_{j} per–prediction request uncertainty scores {Ui∣i∈1,…,Mj}conditional-setsubscript𝑈𝑖𝑖

1…subscript𝑀𝑗\{U\_{i}\mid i\in 1,\dots,M\_{j}\}.
These correspond to the model’s uncertainty in making any trajectory prediction for the agent of interest.
In a real-world deployment setting, a self-driving vehicle would associate a high per–prediction request uncertainty score with a scene context that is particularly unfamiliar or high-risk.

Next, we will describe standard motion prediction performance metrics, followed by confidence-aware metrics which reward models with well-calibrated uncertainty.

### E.3 Performance Metrics

##### Standard Performance Metrics.

We assess the performance of a motion prediction system using several standard metrics.

The average displacement error (ADE) measures the quality of a predicted trajectory 𝒚𝒚\bm{y} with respect to the ground truth trajectory 𝒚∗superscript𝒚\bm{y}^{\*} as

|  |  |  |  |
| --- | --- | --- | --- |
|  | ADE​(𝒚)≔1T​∑t=1T∥st−st∗∥2,≔ADE𝒚1𝑇superscriptsubscript𝑡1𝑇subscriptdelimited-∥∥subscript𝑠𝑡subscriptsuperscript𝑠𝑡2\text{ADE}(\bm{y})\coloneqq\frac{1}{T}\sum\_{t=1}^{T}\left\lVert s\_{t}-s^{\*}\_{t}\right\rVert\_{2}, |  | (7) |

where 𝒚=(s1,…,sT)𝒚subscript𝑠1…subscript𝑠𝑇\bm{y}=(s\_{1},\dots,s\_{T}). Analogously, the final displacement error

|  |  |  |  |
| --- | --- | --- | --- |
|  | FDE​(𝒚)≔∥sT−sT∗∥2,≔FDE𝒚subscriptdelimited-∥∥subscript𝑠𝑇subscriptsuperscript𝑠𝑇2\text{FDE}(\bm{y})\coloneqq\left\lVert s\_{T}-s^{\*}\_{T}\right\rVert\_{2}, |  | (8) |

measures the quality at the last timestep.

Stochastic models define a predictive distribution q​(𝒚∣𝒙;𝜽)𝑞conditional𝒚

𝒙𝜽q(\bm{y}\mid\bm{x};\bm{\theta}), and can therefore be evaluated over the D𝐷D trajectories sampled for a given input 𝒙𝒙\bm{x}. For example, we can measure an aggregated ADE over D𝐷D samples with

|  |  |  |  |
| --- | --- | --- | --- |
|  | aggADED​(q)≔⊕{𝒚}d=1D∼q​(𝒚∣𝒙)​ADE​(𝒚d),≔subscriptaggADE𝐷𝑞similar-tosuperscriptsubscript𝒚𝑑1𝐷𝑞conditional𝒚𝒙direct-sumADEsuperscript𝒚𝑑\text{aggADE}\_{D}(q)\coloneqq\underset{\{\bm{y}\}\_{d=1}^{D}\sim q(\bm{y}\mid\bm{x})}{\oplus}\text{ADE}(\bm{y}^{d}), |  | (9) |

where ⊕direct-sum\oplus is an aggregation operator, e.g., ⊕=min\oplus=\min recovers the minimum ADE (minADEDsubscriptminADE𝐷\text{minADE}\_{D}) commonly used in evaluation of stochastic motion prediction models [[72](#bib.bib72), [66](#bib.bib66)]. We consider minimum and mean aggregation of the average displacement error (minADE, avgADE), as well as of the final displacement error (minFDE, avgFDE).

Per-Trajectory Confidence-Aware Metrics.
A stochastic model used in practice for motion prediction must ultimately *decide* on a particular predicted trajectory for a given prediction request. We may make this decision by selecting for evaluation the predicted trajectory with the highest per-trajectory confidence score. In other words, given per-trajectory confidence scores {c(d)∣d∈1,…,D}conditional-setsuperscript𝑐𝑑𝑑

1…𝐷\{c^{(d)}\mid d\in 1,\dots,D\} we select the top trajectory y(d∗),d∗=arg⁡max𝑑​c(d)

superscript𝑦superscript𝑑superscript𝑑
𝑑superscript𝑐𝑑y^{(d^{\*})},d^{\*}=\underset{d}{\arg\max}\ c^{(d)}, and measure the decision quality using *top1* ADE and FDE metrics, e.g.,

|  |  |  |  |
| --- | --- | --- | --- |
|  | top1ADED​(q)≔ADE​(𝒚(d∗)).≔subscripttop1ADE𝐷𝑞ADEsuperscript𝒚superscript𝑑\text{top1ADE}\_{D}(q)\coloneqq\text{ADE}(\bm{y}^{(d^{\*})}). |  | (10) |

We may also wish to assess the quality of the relative weighting of the D𝐷D trajectories with their corresponding per-trajectory confidence scores c(d)superscript𝑐𝑑c^{(d)}. For this the following weighted metric can be considered:

|  |  |  |  |
| --- | --- | --- | --- |
|  | weightedADED​(q)≔∑d∈Dc(d)⋅ADE​(𝒚(d)).≔subscriptweightedADE𝐷𝑞subscript𝑑𝐷⋅superscript𝑐𝑑ADEsuperscript𝒚𝑑\text{weightedADE}\_{D}(q)\coloneqq\sum\_{d\in D}c^{(d)}\cdot\text{ADE}(\bm{y}^{(d)}). |  | (11) |

The top1FDE and weightedFDE metrics follow analogously to the above. Unfortunately, these metrics, while highly intuitive, have a conceptual limitation.
Consider the following loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒ​(𝚙​(𝒚|𝒙),{c^i(1:D),𝒚^(1:D)})=𝔼𝚙​(𝒚|𝒙)​[∑d=1Dcd​ADE​(𝒚^d,𝒚)],{c^i(1:D),𝒚^(1:D)}=𝒇​(𝒙;𝜽)formulae-sequenceℒ𝚙conditional𝒚𝒙superscriptsubscript^𝑐𝑖:1𝐷superscriptbold-^𝒚:1𝐷subscript𝔼𝚙conditional𝒚𝒙delimited-[]superscriptsubscript𝑑1𝐷superscript𝑐𝑑ADEsuperscriptbold-^𝒚𝑑𝒚superscriptsubscript^𝑐𝑖:1𝐷superscriptbold-^𝒚:1𝐷𝒇  𝒙𝜽\displaystyle\mathcal{L}\left({\tt p}(\bm{y}|\bm{x}),\{\hat{c}\_{i}^{(1:D)},\bm{\hat{y}}^{(1:D)}\}\right)=\mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\sum\_{d=1}^{D}c^{d}\text{ADE}(\bm{\hat{y}}^{d},\bm{y})\right],\ \{\hat{c}\_{i}^{(1:D)},\bm{\hat{y}}^{(1:D)}\}=\bm{f}(\bm{x};\bm{\theta}) |  | (12) |

which is the expected weightedADE given a set of trajectories and weights from a model. If we wish to minimize this loss with respect to the predicted trajectories and weights, then:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒmin=min{c^i(1:D),𝒚^(1:D)}⁡{𝔼𝚙​(𝒚|𝒙)​[∑d=1Dcd​ADE​(𝒚^d,𝒚)]}=min{c^i(1:D)}⁡{∑d=1Dc^d​(min{y^(d)}⁡{𝔼𝚙​(𝒚|𝒙)​[ADE​(𝒚^d,𝒚)]})}=min{c^i(1:D)}⁡{𝔼𝚙​(𝒚|𝒙)​[ADE​(𝒚^∗,𝒚)]​∑d=1Dc^d}=𝔼𝚙​(𝒚|𝒙)​[ADE​(𝒚^∗,𝒚)]subscriptℒminsubscriptsuperscriptsubscript^𝑐𝑖:1𝐷superscriptbold-^𝒚:1𝐷subscript𝔼𝚙conditional𝒚𝒙delimited-[]superscriptsubscript𝑑1𝐷superscript𝑐𝑑ADEsuperscriptbold-^𝒚𝑑𝒚subscriptsuperscriptsubscript^𝑐𝑖:1𝐷superscriptsubscript𝑑1𝐷superscript^𝑐𝑑subscriptsuperscript^𝑦𝑑subscript𝔼𝚙conditional𝒚𝒙delimited-[]ADEsuperscriptbold-^𝒚𝑑𝒚subscriptsuperscriptsubscript^𝑐𝑖:1𝐷subscript𝔼𝚙conditional𝒚𝒙delimited-[]ADEsuperscriptbold-^𝒚𝒚superscriptsubscript𝑑1𝐷superscript^𝑐𝑑subscript𝔼𝚙conditional𝒚𝒙delimited-[]ADEsuperscriptbold-^𝒚𝒚\displaystyle\begin{split}\mathcal{L}\_{\text{min}}=&\ \min\_{\{\hat{c}\_{i}^{(1:D)},\bm{\hat{y}}^{(1:D)}\}}\left\{\mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\sum\_{d=1}^{D}c^{d}\text{ADE}(\bm{\hat{y}}^{d},\bm{y})\right]\right\}\\ =&\ \min\_{\{\hat{c}\_{i}^{(1:D)}\}}\left\{\sum\_{d=1}^{D}\hat{c}^{d}\left(\min\_{\{\hat{y}^{(d)}\}}\left\{\mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\text{ADE}(\bm{\hat{y}}^{d},\bm{y})\right]\right\}\right)\right\}\\ =&\ \min\_{\{\hat{c}\_{i}^{(1:D)}\}}\left\{\mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\text{ADE}(\bm{\hat{y}}^{\*},\bm{y})\right]\sum\_{d=1}^{D}\hat{c}^{d}\right\}\\ =&\ \mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\text{ADE}(\bm{\hat{y}}^{\*},\bm{y})\right]\end{split} |  | (13) |

where 𝒚^∗superscriptbold-^𝒚\bm{\hat{y}}^{\*} is the *weighted geometric median*

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝒚^∗=superscriptbold-^𝒚absent\displaystyle\bm{\hat{y}}^{\*}= | arg{𝒚^}⁡min⁡{𝔼𝚙​(𝒚|𝒙)​[ADE​(𝒚^,𝒚)]}subscriptbold-^𝒚subscript𝔼𝚙conditional𝒚𝒙delimited-[]ADEbold-^𝒚𝒚\displaystyle\ \arg\_{\{\bm{\hat{y}}\}}\min\left\{\mathbb{E}\_{{\tt p}(\bm{y}|\bm{x})}\left[\text{ADE}(\bm{\hat{y}},\bm{y})\right]\right\} |  | (14) |

Thus, the optimal model would suffer from *mode-collapse* and always yields the weighted geometric median of the modes of the true distribution of trajectories. To put this concretely, at a T-junction, where trajectories can go either left or right, the optimal model will yield a trajectory going straight, which is clearly a fundamentally undesirable behaviour. Mathematically, the problem lies in the additive nature of the metric – each mode can be optimized independently of the others. This can be avoided by instead considering a likelihood based metric, such as the following one:

|  |  |  |  |
| --- | --- | --- | --- |
|  | cNLL​(𝒟)≔1N​∑n=1N{−ln⁡[∑d=1Dc(d)​∏t=1T𝒩​(𝒚t,i∗;𝒔t(d)​(𝒙i;𝜽),𝚺=𝟏)]}−T​ln⁡2​π≔cNLL𝒟1𝑁superscriptsubscript𝑛1𝑁superscriptsubscript𝑑1𝐷superscript𝑐𝑑superscriptsubscriptproduct𝑡1𝑇𝒩  superscriptsubscript𝒚  𝑡𝑖superscriptsubscript𝒔𝑡𝑑  subscript𝒙𝑖𝜽𝚺 1𝑇2𝜋\text{cNLL}(\mathcal{D})\coloneqq\frac{1}{N}\sum\_{n=1}^{N}\left\{-\ln\left[\sum\_{d=1}^{D}c^{(d)}\prod\_{t=1}^{T}\mathcal{N}(\bm{y}\_{t,i}^{\*};\ \bm{s}\_{t}^{(d)}(\bm{x}\_{i};\bm{\theta}),\bm{\Sigma}=\bm{1})\right]\right\}-T\ln 2\pi |  | (15) |

Under the following metric, which assumes that each mode is modelled using a Normal distribution of fixed variance, an optimal model would place a Normal over each mode and weight them appropriately. This can be clearly demonstrated using the following numerical example:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | y∼𝚙​(y)=0.5⋅𝒩​(x,10,1)+0.5⋅𝒩​(x,−10,1)similar-to𝑦𝚙𝑦⋅0.5𝒩𝑥101⋅0.5𝒩𝑥101\displaystyle y\sim{\tt p}(y)=0.5\cdot\mathcal{N}(x,10,1)+0.5\cdot\mathcal{N}(x,-10,1) |  | (16) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | 𝔼𝚙(y)[wADE(y,𝒔(1:2)=[10,−10],𝒄=[0.5,0.5])]=201.5𝔼𝚙(y)[wADE(y,𝒔(1:2)=[0,0],𝒄=[0.5,0.5])]=101.50\displaystyle\begin{split}&\mathbb{E}\_{\tt p}(y)[\text{wADE}(y,\bm{s}^{(1:2)}=[10,-10],\bm{c}=[0.5,0.5])]=201.5\\ &\mathbb{E}\_{\tt p}(y)[\text{wADE}(y,\bm{s}^{(1:2)}=[0,0],\bm{c}=[0.5,0.5])]=101.50\end{split} |  | | (17) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼𝚙(y)[cNLL(y,𝒔(1:2)=[10,−10],𝒄=[0.5,0.5])]=1.09\displaystyle\mathbb{E}\_{\tt p}(y)[\text{cNLL}(y,\bm{s}^{(1:2)}=[10,-10],\bm{c}=[0.5,0.5])]=1.09 |  | (18) |
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | 𝔼𝚙(y)[cNLL(y,𝒔(1:2)=[0,0],𝒄=[0.5,0.5])]=50.75\displaystyle\mathbb{E}\_{\tt p}(y)[\text{cNLL}(y,\bm{s}^{(1:2)}=[0,0],\bm{c}=[0.5,0.5])]=50.75 |  | (19) |

Where we have a bimodal Gaussian mixture distribution with modes at -10, 10. We assume we have a model which predicts the means of two trajectories with equal weight. We have two situations: either the model yields two distinct modes at -10, 10 or a collapsed mode at 0 (the median). We can see that predicting the median will yield a lower weightedADE and correctly predicting two distinct modes will yield the lower cNLL. It is important to highlight that this argument holds *in expectation* and is relevant to situations which contain inherent ambiguity and multi-modality. Note that the offset T​ln⁡2​π𝑇2𝜋T\ln 2\pi is used to make assure that the minimal value of this metric is 0, so that it can be used for error-retention and F1-retention plots.

##### Per-Prediction Request Confidence-Aware Metrics.

In addition to making a decision amongst many possible trajectories in a particular situation, a motion planning agent should know when, in general, any trajectories it predicts will be inaccurate (e.g., due to unfamiliarity of the setting, or inherent ambiguity in the path of the vehicle for which a prediction is requested). We evaluate the quality of uncertainty quantification jointly with robustness to distributional shift using the retention-based metrics described in [Section 2](#S2 "2 Evaluation Paradigm, Metrics, and Baselines ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"), with the per–prediction request uncertainty scores determining retention order. Note that each retention curve is plotted with respect to a particular error metric above (e.g., we consider AUC for retention with respect to the cNLL metric introduced above, written as R-AUC). Additionally, we also assess whether the per–prediction uncertainty scores can be used to discriminate between in-domain and shifted scenes. In this case, quality is assessed via area under a ROC curve (ROC-AUC).

### E.4 Experimental Setup

Robust Imitative Planning. In detail, we use the following approach for trajectory and confidence score generation.

1. 1)

   Trajectory Generation. Given a scene input 𝒙𝒙\bm{x}, K𝐾K ensemble members generate G𝐺G trajectories.151515In practice, each ensemble member generates the same number of trajectories Q𝑄Q, s.t. G=K⋅Q𝐺⋅𝐾𝑄G=K\cdot Q.
2. 2)

   Trajectory Scoring. We score each of the G𝐺G trajectories by computing a log probability under each of the K𝐾K trained likelihood models.
3. 3)

   Per-Trajectory Confidence Scores. We aggregate the G⋅K⋅𝐺𝐾G\cdot K resulting log probabilities to G𝐺G scores using a per-trajectory aggregation operator ⊕trajectorysubscriptdirect-sumtrajectory\oplus\_{\text{trajectory}}.161616For example, applying a min aggregation is informed by robust control literature [[80](#bib.bib80)] in which we aim to optimize for the worst-case scenario, as measured by the log-likelihood of the “most pessimistic” model for a given trajectory.
   By aggregating over the log-likelihood estimates sampled from the model posterior (i.e., contributed by each ensemble member), we obtain a robust score for each of the G𝐺G trajectories [[72](#bib.bib72)].
4. 4)

   Trajectory Selection. Among the G𝐺G trajectories, the RIP ensemble produces the top D𝐷D trajectories as determined by their corresponding G𝐺G per-trajectory confidence scores, where D𝐷D is a hyperparameter.
5. 5)

   Per–Prediction Request Uncertainty Score. We aggregate the D𝐷D top per-trajectory confidence scores to a single uncertainty score U𝑈U using the aggregator ⊕pred-reqsubscriptdirect-sumpred-req\oplus\_{\text{pred-req}}.171717In practice, this is done by applying the aggregation (e.g., ⊕pred-req=mean\oplus\_{\text{pred-req}}=\texttt{mean}) to the confidences c(d)superscript𝑐𝑑c^{(d)}, and then *negating* to obtain the uncertainty score U𝑈U. This value conveys the ensemble’s estimated uncertainty for a given scene context and a particular prediction request.
6. 6)

   Confidence Reporting. We obtain scores c(d)superscript𝑐𝑑c^{(d)} by applying a softmax to the D𝐷D top per-trajectory confidence scores. We report these c(d)superscript𝑐𝑑c^{(d)} and U𝑈U (computed in step 5) as our final per-trajectory confidence scores and per–prediction request uncertainty score, respectively.

To summarize, our implementation of RIP for motion prediction produces D𝐷D trajectories and corresponding normalized per-trajectory scores {c(d)∣d∈1,…,D}conditional-setsuperscript𝑐𝑑𝑑

1…𝐷\{c^{(d)}\mid d\in 1,\dots,D\}, as well as an aggregated uncertainty score U𝑈U for the overall prediction request.

Backbone Likelihood Model. We consider two different model classes as ensemble members: a simple behavioral cloning agent with a Gated Recurrent Unit decoder (BC) [[81](#bib.bib81), [73](#bib.bib73)] and a Deep Imitative Model (DIM) [[74](#bib.bib74)] with an autoregressive flow decoder [[82](#bib.bib82)], following [[72](#bib.bib72)].
In both cases, we model the likelihood of a trajectory 𝒚𝒚\bm{y} in context 𝒙𝒙\bm{x} to come from an expert (i.e., from the distribution of ground truth trajectories), with learnable parameters 𝜽𝜽\bm{\theta}, as

|  |  |  |  |
| --- | --- | --- | --- |
|  | q​(𝒚∣𝒙;𝜽)=∏t=1Tp​(st∣𝒚<t,𝒙;𝜽)=∏t=1T𝒩​(st;μ​(𝒚<t,𝒙;𝜽),Σ​(𝒚<t,𝒙;𝜽)),𝑞conditional𝒚  𝒙𝜽superscriptsubscriptproduct𝑡1𝑇𝑝conditionalsubscript𝑠𝑡  subscript𝒚absent𝑡𝒙𝜽superscriptsubscriptproduct𝑡1𝑇𝒩  subscript𝑠𝑡𝜇subscript𝒚absent𝑡𝒙𝜽Σsubscript𝒚absent𝑡𝒙𝜽\begin{split}q(\bm{y}\mid\bm{x};\bm{\theta})&=\prod\_{t=1}^{T}p(s\_{t}\mid\bm{y}\_{<t},\bm{x};\bm{\theta})\ =\prod\_{t=1}^{T}\mathcal{N}(s\_{t};\mu(\bm{y}\_{<t},\bm{x};\bm{\theta}),\Sigma(\bm{y}\_{<t},\bm{x};\bm{\theta})),\end{split} |  | (20) |

where μ​(⋅;𝜽)𝜇

⋅𝜽\mu(\cdot;\bm{\theta}) and Σ​(⋅;𝜽)Σ

⋅𝜽\Sigma(\cdot;\bm{\theta}) are two heads of a recurrent neural network with shared torso.
Hence we assume that the conditional densities are normally distributed, and learn those parameters through maximum likelihood estimation.
Notably, for the BC model, we found that conditioning on samples 𝒚^<tsubscript^𝒚absent𝑡\hat{\bm{y}}\_{<t} instead of ground truth values 𝒚<tsubscript𝒚absent𝑡\bm{y}\_{<t} (where usage of ground truth is often referred to as teacher forcing in RNN literature) significantly improved performance across all datasets and metrics.

Uncertainty Estimation Methods. The above ensembling is done using multiple stochastic models trained with different random seeds, as introduced in Deep Ensembles [[14](#bib.bib14)].
For each ensemble member, we generate Q𝑄Q trajectories.
We can also use a Monte Carlo Dropout [[13](#bib.bib13)] approach for each ensemble member, in which we sample new dropout masks *at test time* during each of the Q𝑄Q forward passes (and corresponding trajectory generations).
Following [[75](#bib.bib75)] we refer to the combination of this uncertainty estimation method with ensembling as Dropout Ensembles.
Previous work has investigated the benefits of Deep Ensembles from a loss landscape perspective [[83](#bib.bib83)], and found that Deep Ensembles tend to explore diverse modes in function space, whereas approximate variational methods such as Monte Carlo Dropout explore around a particular mode.
Dropout Ensembles are hence motivated as ensembles of variational methods which aim to consider a diverse set of modes, with local exploration around each mode.

Setup. We report performance of RIP across the two backbone models – Behavioral Cloning (BC) [[73](#bib.bib73)] and Deep Imitative Model (DIM) [[74](#bib.bib74)] – as well as the two uncertainty estimation methods – Deep Ensembles [[14](#bib.bib14)] and Dropout Ensembles [[13](#bib.bib13), [75](#bib.bib75)].
We evaluate RIP on development (dev) and evaluation (eval) datasets in in-distribution (In), distributionally shifted (Shifted), and combined in-distribution and shifted (Full) settings.
With both backbone model classes we vary the number of ensemble members K∈{1,3,5}𝐾135K\in\{1,3,5\}, train with learning rate 1e-4, use a cosine annealing LR schedule with 1 epoch warmup, and use gradient clipping at 1.
We sample Q=10𝑄10Q=10 trajectories from each of the ensemble members.
We consider two types of aggregation: “Lower Quartile” in which we compute the mean minus the standard deviation μ−σ𝜇𝜎\mu-\sigma of the input scores, and “Model Averaging” (MA) in which we compute the mean μ𝜇\mu of the input scores.
LQ reflects the intuition to assign a high score to a trajectory when the ensemble members assign it a high score on average, and tend to be certain (have a low standard deviation) in their scoring; MA reflects only the prior intuition.
This aggregation strategy (LQ or MA) is used as both the per-trajectory aggregation operator ⊕trajectorysubscriptdirect-sumtrajectory\oplus\_{\text{trajectory}} and the per–prediction request aggregation operator ⊕pred-reqsubscriptdirect-sumpred-req\oplus\_{\text{pred-req}} (where the latter is followed by negation to obtain an uncertainty, as opposed to a confidence).
We fix the RIP ensemble at all K𝐾K to produce the top D=5𝐷5D=5 trajectories as ranked by their per-trajectory confidence score.

### E.5 Additional Results

Below, we report predictive performance using standard-metrics, robustness and uncertainty quantification metrics, and retention plots across the RIP variants.

Table 20: *Predictive performance* of RIP, across model backbones (behavioral cloning (BC) [[73](#bib.bib73)] and Deep Imitative Model (DIM) [[74](#bib.bib74)]) and uncertainty estimation methods (Deep Ensembles [[14](#bib.bib14)] and Dropout Ensembles [[75](#bib.bib75)]). Each section contains losses computed over the in-distribution (In), distributionally shifted (Shifted), and combined (Full) development and evaluation datasets. Altogether, we vary the backbone model, uncertainty estimation method, aggregation strategy (applied for both the per-trajectory aggregation operator ⊕trajectorysubscriptdirect-sumtrajectory\oplus\_{\text{trajectory}} and the per–prediction request aggregation operator ⊕pred-reqsubscriptdirect-sumpred-req\oplus\_{\text{pred-req}}), and the number of ensemble members K𝐾K.
See [Section E.4](#A5.SS4 "E.4 Experimental Setup ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for setup details.

| Dataset | Method | Model | minADE ↓↓\downarrow | | | weightedADE ↓↓\downarrow | | | minFDE ↓↓\downarrow | | | weightedFDE ↓↓\downarrow | | | cNLL ↓↓\downarrow | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In | Shifted | Full | In | Shifted | Full | In | Shifted | Full | In | Shifted | Full | In | Shifted | Full |
| Dev | Deep Ensemble | BC, LQ, K=1 | 0.818 | 0.960 | 0.835 | 1.088 | 1.245 | 1.107 | 1.718 | 2.113 | 1.765 | 2.368 | 2.777 | 2.417 | 59.64 | 98.54 | 64.29 |
| BC, LQ, K=3 | 0.780 | 0.909 | 0.795 | 1.040 | 1.170 | 1.056 | 1.638 | 2.018 | 1.683 | 2.254 | 2.609 | 2.297 | 54.78 | 87.81 | 58.73 |
| BC, LQ, K=5 | 0.766 | 0.888 | 0.780 | 1.017 | 1.138 | 1.031 | 1.618 | 1.980 | 1.661 | 2.214 | 2.552 | 2.254 | 56.45 | 90.25 | 60.49 |
| BC, MA, K=1 | 0.818 | 0.960 | 0.835 | 1.088 | 1.245 | 1.107 | 1.718 | 2.113 | 1.765 | 2.368 | 2.777 | 2.417 | 59.64 | 98.54 | 64.29 |
| BC, MA, K=3 | 0.780 | 0.908 | 0.795 | 1.034 | 1.166 | 1.050 | 1.641 | 2.018 | 1.686 | 2.249 | 2.611 | 2.292 | 55.00 | 88.45 | 59.00 |
| BC, MA, K=5 | 0.765 | 0.887 | 0.779 | 1.012 | 1.133 | 1.026 | 1.617 | 1.976 | 1.660 | 2.210 | 2.551 | 2.251 | 56.86 | 91.54 | 61.01 |
| DIM, LQ, K=1 | 0.750 | 0.818 | 0.758 | 1.523 | 1.583 | 1.530 | 1.497 | 1.720 | 1.524 | 3.472 | 3.639 | 3.492 | 50.66 | 73.00 | 53.34 |
| DIM, LQ, K=3 | 0.717 | 0.787 | 0.725 | 1.407 | 1.470 | 1.415 | 1.467 | 1.687 | 1.493 | 3.219 | 3.397 | 3.240 | 48.88 | 70.93 | 51.52 |
| DIM, LQ, K=5 | 0.720 | 0.787 | 0.728 | 1.399 | 1.470 | 1.407 | 1.487 | 1.704 | 1.513 | 3.202 | 3.397 | 3.225 | 51.12 | 72.87 | 53.72 |
| DIM, MA, K=1 | 0.750 | 0.818 | 0.758 | 1.523 | 1.583 | 1.530 | 1.497 | 1.720 | 1.524 | 3.472 | 3.639 | 3.492 | 50.66 | 73.00 | 53.34 |
| DIM, MA, K=3 | 0.717 | 0.785 | 0.725 | 1.410 | 1.475 | 1.418 | 1.466 | 1.685 | 1.492 | 3.226 | 3.409 | 3.248 | 48.74 | 71.30 | 51.44 |
| DIM, MA, K=5 | 0.719 | 0.786 | 0.727 | 1.399 | 1.469 | 1.408 | 1.482 | 1.698 | 1.508 | 3.202 | 3.393 | 3.225 | 50.85 | 72.45 | 53.43 |
| Dropout Ensemble | BC, LQ, K=1 | 0.803 | 0.908 | 0.815 | 1.116 | 1.236 | 1.130 | 1.649 | 1.952 | 1.685 | 2.409 | 2.718 | 2.446 | 55.98 | 82.49 | 59.15 |
| BC, LQ, K=3 | 0.741 | 0.853 | 0.754 | 1.013 | 1.132 | 1.028 | 1.542 | 1.873 | 1.581 | 2.209 | 2.545 | 2.249 | 53.01 | 83.93 | 56.71 |
| BC, LQ, K=5 | 0.759 | 0.878 | 0.773 | 1.008 | 1.127 | 1.023 | 1.605 | 1.960 | 1.648 | 2.204 | 2.538 | 2.244 | 55.58 | 88.78 | 59.55 |
| BC, MA, K=1 | 0.803 | 0.908 | 0.815 | 1.116 | 1.236 | 1.130 | 1.649 | 1.952 | 1.685 | 2.409 | 2.718 | 2.446 | 55.98 | 82.49 | 59.15 |
| BC, MA, K=3 | 0.739 | 0.850 | 0.752 | 1.020 | 1.135 | 1.033 | 1.534 | 1.864 | 1.574 | 2.223 | 2.553 | 2.263 | 53.09 | 83.81 | 56.76 |
| BC, MA, K=5 | 0.757 | 0.877 | 0.771 | 1.010 | 1.126 | 1.024 | 1.597 | 1.952 | 1.640 | 2.209 | 2.539 | 2.248 | 55.82 | 89.57 | 59.86 |
| DIM, LQ, K=1 | 0.750 | 0.831 | 0.759 | 1.498 | 1.587 | 1.509 | 1.510 | 1.757 | 1.539 | 3.432 | 3.662 | 3.459 | 52.57 | 76.54 | 55.44 |
| DIM, LQ, K=3 | 0.716 | 0.786 | 0.725 | 1.412 | 1.473 | 1.419 | 1.466 | 1.687 | 1.493 | 3.234 | 3.408 | 3.254 | 49.69 | 72.58 | 52.43 |
| DIM, LQ, K=5 | 0.723 | 0.793 | 0.731 | 1.409 | 1.475 | 1.417 | 1.494 | 1.717 | 1.521 | 3.224 | 3.408 | 3.246 | 51.25 | 73.47 | 53.91 |
| DIM, MA, K=1 | 0.750 | 0.831 | 0.759 | 1.498 | 1.587 | 1.509 | 1.510 | 1.757 | 1.539 | 3.432 | 3.662 | 3.459 | 52.57 | 76.54 | 55.44 |
| DIM, MA, K=3 | 0.716 | 0.786 | 0.724 | 1.414 | 1.479 | 1.422 | 1.465 | 1.685 | 1.491 | 3.238 | 3.420 | 3.260 | 49.38 | 71.86 | 52.07 |
| DIM, MA, K=5 | 0.721 | 0.793 | 0.729 | 1.409 | 1.474 | 1.417 | 1.489 | 1.717 | 1.516 | 3.224 | 3.405 | 3.246 | 50.99 | 73.64 | 53.70 |
| Eval | Deep Ensemble | BC, LQ, K=1 | 0.829 | 1.084 | 0.880 | 1.104 | 1.407 | 1.164 | 1.733 | 2.420 | 1.870 | 2.394 | 3.197 | 2.555 | 60.20 | 98.82 | 67.93 |
| BC, LQ, K=3 | 0.792 | 1.026 | 0.839 | 1.056 | 1.326 | 1.110 | 1.658 | 2.297 | 1.786 | 2.284 | 3.005 | 2.429 | 55.97 | 90.54 | 62.89 |
| BC, LQ, K=5 | 0.777 | 1.015 | 0.825 | 1.032 | 1.303 | 1.086 | 1.636 | 2.283 | 1.765 | 2.242 | 2.964 | 2.386 | 57.26 | 93.92 | 64.60 |
| BC, MA, K=1 | 0.829 | 1.084 | 0.880 | 1.104 | 1.407 | 1.164 | 1.733 | 2.420 | 1.870 | 2.394 | 3.197 | 2.555 | 60.20 | 98.82 | 67.93 |
| BC, MA, K=3 | 0.792 | 1.025 | 0.838 | 1.050 | 1.319 | 1.104 | 1.661 | 2.294 | 1.788 | 2.278 | 2.997 | 2.422 | 55.94 | 90.53 | 62.87 |
| BC, MA, K=5 | 0.777 | 1.014 | 0.824 | 1.028 | 1.299 | 1.082 | 1.636 | 2.278 | 1.765 | 2.238 | 2.957 | 2.382 | 57.75 | 95.00 | 65.20 |
| DIM, LQ, K=1 | 0.759 | 0.942 | 0.796 | 1.551 | 1.883 | 1.618 | 1.511 | 1.983 | 1.605 | 3.536 | 4.376 | 3.704 | 50.50 | 76.00 | 55.60 |
| DIM, LQ, K=3 | 0.726 | 0.914 | 0.764 | 1.433 | 1.756 | 1.498 | 1.481 | 1.972 | 1.579 | 3.277 | 4.094 | 3.440 | 49.45 | 76.66 | 54.89 |
| DIM, LQ, K=5 | 0.729 | 0.921 | 0.768 | 1.422 | 1.757 | 1.489 | 1.498 | 2.007 | 1.600 | 3.253 | 4.098 | 3.422 | 51.61 | 79.71 | 57.24 |
| DIM, MA, K=1 | 0.759 | 0.942 | 0.796 | 1.551 | 1.883 | 1.618 | 1.511 | 1.983 | 1.605 | 3.536 | 4.376 | 3.704 | 50.50 | 76.00 | 55.60 |
| DIM, MA, K=3 | 0.726 | 0.912 | 0.763 | 1.437 | 1.759 | 1.502 | 1.478 | 1.967 | 1.576 | 3.286 | 4.101 | 3.449 | 49.09 | 76.07 | 54.49 |
| DIM, MA, K=5 | 0.728 | 0.918 | 0.766 | 1.424 | 1.754 | 1.490 | 1.493 | 2.000 | 1.595 | 3.256 | 4.093 | 3.424 | 51.19 | 78.85 | 56.73 |
| Dropout Ensemble | BC, LQ, K=1 | 0.812 | 1.038 | 0.857 | 1.128 | 1.410 | 1.184 | 1.664 | 2.267 | 1.784 | 2.430 | 3.170 | 2.578 | 56.57 | 86.28 | 62.52 |
| BC, LQ, K=3 | 0.751 | 0.972 | 0.795 | 1.029 | 1.297 | 1.082 | 1.558 | 2.154 | 1.677 | 2.238 | 2.948 | 2.380 | 53.94 | 86.68 | 60.49 |
| BC, LQ, K=5 | 0.770 | 1.008 | 0.817 | 1.024 | 1.297 | 1.079 | 1.623 | 2.268 | 1.752 | 2.233 | 2.957 | 2.378 | 56.49 | 92.77 | 63.75 |
| BC, MA, K=1 | 0.812 | 1.038 | 0.857 | 1.128 | 1.410 | 1.184 | 1.664 | 2.267 | 1.784 | 2.430 | 3.170 | 2.578 | 56.57 | 86.28 | 62.52 |
| BC, MA, K=3 | 0.749 | 0.970 | 0.794 | 1.036 | 1.305 | 1.090 | 1.551 | 2.147 | 1.670 | 2.253 | 2.963 | 2.395 | 54.07 | 86.94 | 60.65 |
| BC, MA, K=5 | 0.768 | 1.004 | 0.815 | 1.027 | 1.299 | 1.081 | 1.615 | 2.253 | 1.743 | 2.239 | 2.958 | 2.383 | 56.90 | 93.27 | 64.18 |
| DIM, LQ, K=1 | 0.739 | 0.924 | 0.776 | 1.478 | 1.815 | 1.546 | 1.474 | 1.949 | 1.569 | 3.380 | 4.239 | 3.552 | 49.90 | 75.31 | 54.98 |
| DIM, LQ, K=3 | 0.722 | 0.910 | 0.760 | 1.431 | 1.763 | 1.497 | 1.470 | 1.967 | 1.569 | 3.266 | 4.112 | 3.435 | 49.30 | 75.24 | 54.49 |
| DIM, LQ, K=5 | 0.729 | 0.929 | 0.769 | 1.430 | 1.769 | 1.497 | 1.497 | 2.027 | 1.603 | 3.268 | 4.126 | 3.440 | 50.77 | 80.02 | 56.63 |
| DIM, MA, K=1 | 0.739 | 0.924 | 0.776 | 1.478 | 1.815 | 1.546 | 1.474 | 1.949 | 1.569 | 3.380 | 4.239 | 3.552 | 49.90 | 75.31 | 54.98 |
| DIM, MA, K=3 | 0.720 | 0.907 | 0.758 | 1.432 | 1.760 | 1.497 | 1.465 | 1.960 | 1.564 | 3.267 | 4.107 | 3.435 | 48.74 | 74.70 | 53.93 |
| DIM, MA, K=5 | 0.728 | 0.925 | 0.767 | 1.431 | 1.766 | 1.498 | 1.494 | 2.017 | 1.599 | 3.269 | 4.120 | 3.439 | 50.51 | 79.30 | 56.28 |

Table 21: *Uncertainty and robustness performance* of RIP across the two backbone models (BC and DIM) and uncertainty estimation methods (Deep Ensemble and Dropout Ensemble). The error metric for computing the area under the rejection curve (R-AUC) and area under the F1 curve (F1-AUC) is cNLL. We use a threshold of 25 for the F1 metrics, which approximately corresponds to a 1 meter deviation on all trajectories. See [Section E.4](#A5.SS4 "E.4 Experimental Setup ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks") for setup details.

| Dataset | Method | Model | R-AUC ↓↓\downarrow | | | F1-AUC (%) ↑↑\uparrow | | | F1@959595% ↑↑\uparrow | | | ROC-AUC (%) ↑↑\uparrow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In | Shifted | Full | In | Shifted | Full | In | Shifted | Full |
| Dev | Deep Ensemble | BC, LQ, K=1 | 11.06 | 13.91 | 11.22 | 64.9 | 66.7 | 65.1 | 89.1 | 90.2 | 89.3 | 51.0 |
| BC, LQ, K=3 | 11.26 | 11.69 | 11.18 | 63.4 | 66.0 | 63.8 | 88.5 | 90.3 | 88.8 | 46.7 |
| BC, LQ, K=5 | 9.68 | 10.38 | 9.62 | 64.3 | 66.4 | 64.6 | 89.7 | 91.0 | 90.0 | 47.3 |
| BC, MA, K=1 | 11.06 | 13.91 | 11.22 | 64.9 | 66.7 | 65.1 | 89.1 | 90.2 | 89.3 | 51.0 |
| BC, MA, K=3 | 9.31 | 10.73 | 9.31 | 64.8 | 66.5 | 65.0 | 90.3 | 91.3 | 90.6 | 48.6 |
| BC, MA, K=5 | 9.07 | 10.47 | 9.08 | 64.9 | 66.5 | 65.2 | 90.4 | 91.3 | 90.6 | 49.2 |
| DIM, LQ, K=1 | 12.54 | 15.28 | 12.86 | 63.6 | 64.8 | 63.8 | 87.2 | 88.8 | 87.4 | 51.8 |
| DIM, LQ, K=3 | 12.30 | 14.51 | 12.57 | 63.7 | 64.9 | 63.8 | 89.3 | 89.9 | 89.3 | 51.4 |
| DIM, LQ, K=5 | 12.87 | 15.01 | 13.14 | 63.5 | 64.8 | 63.7 | 89.7 | 90.2 | 89.7 | 51.4 |
| DIM, MA, K=1 | 12.57 | 15.10 | 12.86 | 63.7 | 64.9 | 63.8 | 87.2 | 88.8 | 87.4 | 51.8 |
| DIM, MA, K=3 | 12.38 | 14.46 | 12.64 | 63.7 | 64.9 | 63.8 | 89.2 | 89.9 | 89.3 | 51.4 |
| DIM, MA, K=5 | 12.97 | 15.10 | 13.24 | 63.5 | 64.8 | 63.7 | 89.6 | 90.2 | 89.7 | 51.4 |
| Dropout Ensemble | BC, LQ, K=1 | 8.87 | 10.00 | 8.87 | 65.3 | 67.1 | 65.6 | 89.7 | 90.4 | 89.9 | 51.2 |
| BC, LQ, K=3 | 8.11 | 9.53 | 8.14 | 64.9 | 66.5 | 65.1 | 90.6 | 91.3 | 90.8 | 50.9 |
| BC, LQ, K=5 | 8.28 | 9.60 | 8.28 | 65.0 | 66.6 | 65.2 | 90.5 | 91.3 | 90.7 | 50.7 |
| BC, MA, K=1 | 8.87 | 9.99 | 8.87 | 65.3 | 67.1 | 65.6 | 89.7 | 90.4 | 89.9 | 51.2 |
| BC, MA, K=3 | 8.53 | 9.79 | 8.54 | 64.9 | 66.5 | 65.1 | 90.7 | 91.4 | 90.8 | 50.3 |
| BC, MA, K=5 | 8.89 | 10.23 | 8.90 | 64.9 | 66.5 | 65.2 | 90.5 | 91.4 | 90.7 | 50.2 |
| DIM, LQ, K=1 | 12.57 | 16.41 | 13.03 | 63.8 | 64.7 | 63.9 | 87.6 | 89.1 | 87.8 | 51.5 |
| DIM, LQ, K=3 | 12.37 | 14.91 | 12.69 | 63.7 | 64.8 | 63.8 | 89.2 | 90.0 | 89.3 | 51.3 |
| DIM, LQ, K=5 | 12.94 | 15.18 | 13.22 | 63.6 | 64.8 | 63.7 | 89.6 | 90.2 | 89.7 | 51.4 |
| DIM, MA, K=1 | 12.61 | 16.30 | 13.06 | 63.8 | 64.8 | 63.9 | 87.6 | 89.1 | 87.7 | 51.6 |
| DIM, MA, K=3 | 12.49 | 14.80 | 12.79 | 63.6 | 64.8 | 63.8 | 89.2 | 90.0 | 89.3 | 51.4 |
| DIM, MA, K=5 | 13.05 | 15.20 | 13.33 | 63.5 | 64.8 | 63.7 | 89.5 | 90.2 | 89.6 | 51.4 |
| Eval | Deep Ensemble | BC, LQ, K=1 | 11.16 | 20.84 | 12.91 | 64.9 | 65.5 | 65.0 | 88.9 | 85.6 | 88.4 | 52.8 |
| BC, LQ, K=3 | 11.31 | 17.09 | 12.38 | 63.4 | 64.8 | 63.7 | 88.4 | 86.4 | 88.0 | 50.9 |
| BC, LQ, K=5 | 9.77 | 15.95 | 10.88 | 64.3 | 65.4 | 64.5 | 89.5 | 87.1 | 89.1 | 51.4 |
| BC, MA, K=1 | 11.17 | 20.84 | 12.91 | 64.9 | 65.5 | 65.0 | 88.9 | 85.6 | 88.4 | 52.8 |
| BC, MA, K=3 | 9.40 | 16.76 | 10.73 | 64.8 | 65.6 | 65.0 | 90.2 | 87.5 | 89.7 | 51.3 |
| BC, MA, K=5 | 9.20 | 16.85 | 10.57 | 65.0 | 65.6 | 65.1 | 90.2 | 87.5 | 89.7 | 52.1 |
| DIM, LQ, K=1 | 12.78 | 20.78 | 14.28 | 63.5 | 63.7 | 63.6 | 86.9 | 83.9 | 86.3 | 52.0 |
| DIM, LQ, K=3 | 12.66 | 21.40 | 14.32 | 63.6 | 63.9 | 63.7 | 89.1 | 86.0 | 88.5 | 51.4 |
| DIM, LQ, K=5 | 13.26 | 22.59 | 15.05 | 63.5 | 63.8 | 63.6 | 89.5 | 86.5 | 88.9 | 51.2 |
| DIM, MA, K=1 | 12.81 | 20.83 | 14.32 | 63.6 | 63.8 | 63.6 | 86.9 | 83.9 | 86.3 | 51.8 |
| DIM, MA, K=3 | 12.74 | 21.51 | 14.42 | 63.6 | 63.9 | 63.7 | 89.1 | 86.0 | 88.5 | 51.1 |
| DIM, MA, K=5 | 13.37 | 22.68 | 15.16 | 63.5 | 63.7 | 63.5 | 89.5 | 86.5 | 88.9 | 50.9 |
| Dropout Ensemble | BC, LQ, K=1 | 9.06 | 15.49 | 10.22 | 65.3 | 66.1 | 65.5 | 89.5 | 86.4 | 89.0 | 53.7 |
| BC, LQ, K=3 | 8.22 | 14.83 | 9.39 | 64.9 | 65.6 | 65.1 | 90.5 | 87.5 | 90.0 | 53.9 |
| BC, LQ, K=5 | 8.39 | 15.16 | 9.57 | 65.0 | 65.7 | 65.2 | 90.4 | 87.6 | 89.9 | 54.5 |
| BC, MA, K=1 | 9.07 | 15.50 | 10.22 | 65.3 | 66.1 | 65.5 | 89.5 | 86.4 | 89.0 | 53.7 |
| BC, MA, K=3 | 8.69 | 15.90 | 9.99 | 64.9 | 65.6 | 65.1 | 90.5 | 87.7 | 90.0 | 53.0 |
| BC, MA, K=5 | 9.05 | 16.69 | 10.41 | 65.0 | 65.6 | 65.1 | 90.4 | 87.6 | 89.9 | 53.2 |
| DIM, LQ, K=1 | 12.45 | 20.27 | 13.92 | 63.6 | 63.7 | 63.6 | 87.7 | 84.7 | 87.1 | 51.8 |
| DIM, LQ, K=3 | 12.63 | 21.32 | 14.29 | 63.7 | 63.9 | 63.7 | 89.1 | 86.1 | 88.6 | 51.3 |
| DIM, LQ, K=5 | 13.22 | 22.78 | 15.04 | 63.5 | 63.8 | 63.6 | 89.4 | 86.3 | 88.8 | 51.2 |
| DIM, MA, K=1 | 12.51 | 20.33 | 14.00 | 63.6 | 63.8 | 63.7 | 87.7 | 84.7 | 87.1 | 51.5 |
| DIM, MA, K=3 | 12.73 | 21.43 | 14.40 | 63.6 | 63.9 | 63.7 | 89.1 | 86.0 | 88.5 | 51.1 |
| DIM, MA, K=5 | 13.36 | 22.85 | 15.19 | 63.5 | 63.8 | 63.6 | 89.4 | 86.3 | 88.8 | 50.9 |

!(/html/2107.07455/assets/x6.png)

(a) Full dev cNLL retention.

!(/html/2107.07455/assets/x7.png)

(b) Full eval cNLL retention.

!(/html/2107.07455/assets/x8.png)

(c) Full dev F1-cNLL retention.

!(/html/2107.07455/assets/x9.png)

(d) Full eval F1-cNLL retention.

Figure 18: cNLL and F1-cNLL retention curves on the Full (i.e., containing both the in-distribution and distributionally shifted datapoints) dev (left column) and eval (right column) partitions of the Vehicle Motion Prediction dataset.
Top row: retention on cNLL (lower ↓↓\downarrow AUC is better). Bottom row: retention on F1-cNLL (higher ↑↑\uparrow AUC is better).
We vary the backbone model and number of ensemble members, fix the Model Averaging (MA) aggregation strategy for the per-trajectory aggregation operator ⊕trajectorysubscriptdirect-sumtrajectory\oplus\_{\text{trajectory}} and the per–prediction request aggregation operator ⊕pred-reqsubscriptdirect-sumpred-req\oplus\_{\text{pred-req}} (based on results from [Table 7](#S5.T7 "In Baselines ‣ 5 Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks")), and otherwise use the standard RIP settings enumerated in [Section E.4](#A5.SS4 "E.4 Experimental Setup ‣ Appendix E Vehicle Motion Prediction ‣ Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks").
