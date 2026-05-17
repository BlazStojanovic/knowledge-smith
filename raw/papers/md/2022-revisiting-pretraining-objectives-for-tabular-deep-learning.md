---
arxiv: '2207.03208'
authors:
- Ivan Rubachev
- Artem Alekberov
- Yury Gorishniy
- Artem Babenko
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: Revisiting Pretraining Objectives for Tabular Deep Learning
url: http://arxiv.org/abs/2207.03208v2
year: 2022
---

# Revisiting Pretraining Objectives for Tabular Deep Learning

Ivan Rubachevα,β  Artem Alekberovα,β  Yury Gorishniy  Artem Babenkoα
  
αYandex   βHSE University

###### Abstract

Recent deep learning models for tabular data currently compete with the traditional ML models based on decision trees (GBDT). Unlike GBDT, deep models can additionally benefit from pretraining, which is a workhorse of DL for vision and NLP.
For tabular problems, several pretraining methods were proposed, but it is not entirely clear if pretraining provides consistent noticeable improvements and what method should be used, since the methods are often not compared to each other or comparison is limited to the simplest MLP architectures.

In this work, we aim to identify the best practices to pretrain tabular DL models that can be universally applied to different datasets and architectures.
Among our findings, we show that using the object target labels during the pretraining stage is beneficial for the downstream performance and advocate several target-aware pretraining objectives.
Overall, our experiments demonstrate that properly performed pretraining significantly increases the performance of tabular DL models, which often leads to their superiority over GBDTs.

## 1 Introduction

Tabular problems are ubiquitous in industrial ML applications, which include data described by a set of heterogeneous features, such as learning-to-rank, click-through rate prediction, credit scoring, and many others. Despite the current dominance of deep learning models in the ML literature, for tabular problems, the “old-school” decision tree ensembles (e.g., GBDT) are often the top choice for practitioners. Only recently, several works have proposed the deep models that challenge the supremacy of GBDT in the tabular domain [[2](#bib.bib2), [15](#bib.bib15), [37](#bib.bib37), [14](#bib.bib14)] and suggest that the question “tabular DL or GBDT” is yet to be answered.

An important advantage of deep models over GBDT is that they can potentially achieve higher performance via pretraining their parameters with a properly designed objective.
These pretrained parameters, then, serve as a better than random initialization for subsequent finetuning for downstream tasks.
For computer vision and NLP domains, pretraining is a de facto standard and is shown to be necessary for the state-of-the-art performance [[18](#bib.bib18), [10](#bib.bib10)]. For tabular problems, however, such a consensus is yet to be achieved as well as the best practices of tabular pretraining are to be established. In particular, pretraining for tabular problems is typically performed directly on the downstream target datasets, unlike pretraining in vision or NLP problems, for which huge “extra” data is available on the Internet.
While a large number of prior works addresses the pretraining of tabular DL models [[42](#bib.bib42), [4](#bib.bib4), [39](#bib.bib39), [9](#bib.bib9)], it is challenging to make reliable conclusions about pretraining efficacy in tabular DL from the literature since experimental setups vary significantly. Some evaluation protocols assume the unlabeled data is abundant but use a small subset of labels from each dataset during finetuning for evaluation – demonstrating pretraining efficacy, but somewhat limiting the performance of supervised baselines.

By contrast, in our work, we focus on the setup with fully labeled tabular datasets to understand if pretraining helps tabular DL in a fully supervised setting and compare pretraining methods to the strong supervised baselines. To this end, we perform a systematic experimental evaluation of several pretraining objectives, identify the superior ones, and describe the practical details of how to perform tabular pretraining optimally. Our main findings, which are important for practitioners, are summarized below:

* •

  Pretraining provides substantial gains over well-tuned supervised baselines in the fully supervised setup.
* •

  Simple self-prediction based pretraining objectives are comparable to the objective based on contrastive learning. To the best of our knowledge, this was not reported before in tabular DL.
* •

  The object labels can be exploited for more effective pretraining. In particular, we describe several “target-aware” objectives and demonstrate their superiority over their “unsupervised” counterparts.
* •

  The pretraining provides the most noticeable improvements for the vanilla MLP architecture. In particular, their performance after pretraining becomes comparable to the state-of-the-art models trained from scratch, which is important for practitioners, who are interested in simple and efficient solutions.
* •

  The ensembling of pretrained models is beneficial. It indicates that the pretraining stage does not significantly decrease the diversity of the models, despite the fact that all the models are initialized by the same set of parameters.

Overall, our work provides a set of recipes for practitioners interested in tabular pretraining, which results in higher performance for most of the tasks. The code of our experiments is available online.

## 2 Related Work

Here we briefly review the lines of research that are relevant to our study.

Status Quo in tabular deep learning.
A plethora of recent works have proposed a large number of deep models for tabular data [[25](#bib.bib25), [32](#bib.bib32), [2](#bib.bib2), [38](#bib.bib38), [41](#bib.bib41), [3](#bib.bib3), [17](#bib.bib17), [20](#bib.bib20), [37](#bib.bib37), [15](#bib.bib15), [27](#bib.bib27)]. Several systematic studies, however, reveal that these models typically do not consistently outperform the decision tree ensembles, such as GBDT (Gradient Boosting Decision Tree) [[7](#bib.bib7), [33](#bib.bib33), [22](#bib.bib22)], which are typically the top-choice in various ML competitions [[15](#bib.bib15), [35](#bib.bib35)]. Additionally, several works have shown that the existing sophisticated architectures are not consistently superior to properly tuned simple models, such as MLP and ResNet [[15](#bib.bib15), [21](#bib.bib21)]. Finally, the recent work [[14](#bib.bib14)] has highlighted that the appropriate embeddings of numerical features in the high-dimensional space are universally beneficial for different architectures. In our work, we experiment with pretraining of both traditional MLP-like models and advanced embedding-based models proposed in [[14](#bib.bib14)].

Pretraining in deep learning. For domains with structured data, like natural images or texts, pretraining is currently an established stage in the typical pipelines, which leads to higher general performance and better model robustness [[18](#bib.bib18), [10](#bib.bib10)].
Pretraining with the auto-encoding objective was also previously studied as a regularization strategy helping in the optimization process [[12](#bib.bib12), [11](#bib.bib11)] without large scale pretraining datasets. During the last years, several families of successful pretraining methods have been developed. An impactful line of research on pretraining is based on the paradigm of contrastive learning, which effectively enforces the invariance of the learned representations to the human-specified augmentations [[8](#bib.bib8), [19](#bib.bib19)]. Another line of methods exploits the idea of self-prediction, i.e., these methods require the model to predict certain parts of the input given the remaining parts [[18](#bib.bib18), [10](#bib.bib10)]. In the vision community, the self-prediction based methods are shown to be superior to the methods that use contrastive learning objectives [[18](#bib.bib18)]. In our experiments, we demonstrate that self-prediction based objectives are comparable to the contrastive learning ones on tabular data, while being much simpler.

Pretraining for the tabular domain. Numerous pretraining methods were recently proposed in several recent works on tabular DL [[2](#bib.bib2), [42](#bib.bib42), [9](#bib.bib9), [39](#bib.bib39), [37](#bib.bib37), [27](#bib.bib27)]. However, most of these works do not focus on the pretraining objective per se and typically introduce it as a component of their tabular DL pipeline. Moreover, the experimental setup varies significantly between methods. Therefore, it is difficult to extract conclusive evidence about pretraining effectiveness from the literature. To the best of our knowledge, there is only one systematic study on the tabular pretraining [[4](#bib.bib4)], but its experimental evaluation is performed only with the simplest MLP models, and we found that
the superiority of the contrastive pretraining, reported in [[4](#bib.bib4)], does not hold for tuned models in our setup, where contrastive objective is comparable to the simpler self-prediction objectives.

## 3 Revisiting pretraining objectives

In this section, we evaluate the typical pretraining objectives under the unified experimental setup on the number of datasets from the literature on tabular DL.
Our goal is to answer whether pretraining generally provides significant improvements in downstream task performance over tuned models trained from scratch and to identify the pretraining objectives that lead to the best downstream task performance.

### 3.1 Experimental setup

We mostly follow the experimental setup from [[16](#bib.bib16)] and describe its main details here for completeness.

Notation. Each tabular dataset is represented by a set of pairs {(xi,yi)}i=1nsuperscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑛\left\{{(x\_{i},y\_{i})}\right\}\_{i=1}^{n}, where xi=(xi1,…,xim)∈𝕏subscript𝑥𝑖superscriptsubscript𝑥𝑖1…superscriptsubscript𝑥𝑖𝑚𝕏x\_{i}=(x\_{i}^{1},\ldots,x\_{i}^{m})\in\mathbb{X} are the objects features (both numerical and categorical) and yi∈𝕐subscript𝑦𝑖𝕐y\_{i}\in\mathbb{Y} is the target variable. The downstream task is either regression 𝕐=ℝ𝕐ℝ\mathbb{Y}=\mathbb{R} or classification 𝕐={1,…,k}𝕐1…𝑘\mathbb{Y}=\left\{{1,\ldots,k}\right\}. Each model has the backbone f​(x|θ)𝑓conditional𝑥𝜃f(x|\theta) that is followed by two separate heads: a pretraining head h​(z|μ)ℎconditional𝑧𝜇h(z|\mu) and a downstream task head g​(z|λ)𝑔conditional𝑧𝜆g(z|\lambda), with learnable parameters θ,μ,λ

𝜃𝜇𝜆\theta,\mu,\lambda respectively, and z=f​(x|θ)𝑧𝑓conditional𝑥𝜃z=f(x|\theta) denotes the output of the backbone for an input object x𝑥x.

Datasets. We evaluate the pretraining methods on a curated set of eleven middle to large scale datasets used in prior literature on tabular deep learning. The benchmark is biased towards tasks, where tuned MLP models were shown to be inferior to GBDT [[16](#bib.bib16)] since we aim to understand if pretraining can help the deep models to beat the “shallow” ones. The datasets represent a diverse set of tabular data problems with classification and regression targets. The main dataset properties are summarized in [Table 1](#S3.T1 "Table 1 ‣ 3.1 Experimental setup ‣ 3 Revisiting pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning").

Table 1: Datasets used for the experiments

| Abbr | Name | # Train | # Validation | # Test | # Num | # Cat | Task type | Batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GE | Gesture Phase | 631863186318 | 158015801580 | 197519751975 | 323232 | 00 | Multiclass | 128 |
| CH | Churn Modelling | 640064006400 | 160016001600 | 200020002000 | 101010 | 111 | Binclass | 128 |
| CA | California Housing | 132091320913209 | 330333033303 | 412841284128 | 888 | 00 | Regression | 128 |
| HO | House 16H | 145811458114581 | 364636463646 | 455745574557 | 161616 | 00 | Regression | 128 |
| AD | Adult ROC | 260482604826048 | 651365136513 | 162811628116281 | 666 | 888 | Binclass | 256 |
| OT | Otto Group Products LogLoss | 396013960139601 | 990199019901 | 123761237612376 | 939393 | 00 | Multiclass | 256 |
| HI | Higgs Small | 627516275162751 | 156881568815688 | 196101961019610 | 282828 | 00 | Binclass | 512 |
| FB | Facebook Comments Volume | 157638157638157638 | 197221972219722 | 197201972019720 | 505050 | 111 | Regression | 512 |
| WE | Shifts Weather (subset) | 296554296554296554 | 473734737347373 | 531725317253172 | 123123123 | 00 | Regression | 1024 |
| CO | Covertype | 371847371847371847 | 929629296292962 | 116203116203116203 | 545454 | 00 | Multiclass | 1024 |
| MI | MSLR-WEB10K (Fold 1) | 723412723412723412 | 235259235259235259 | 241521241521241521 | 136136136 | 00 | Regression | 1024 |

We report ROC-AUC for all binary classification datasets, accuracy for multi-class classification datasets and RMSE for regression datasets, with OT being the one exception, where we report log-loss, as it was used as a default metric in the corresponding Kaggle competition. We use the quantile-transform from the Scikit-learn library [[31](#bib.bib31)] to preprocess the numerical features for all datasets except OT, where the absence of such transformation was shown to be superior [[14](#bib.bib14)]. Additional information about the datasets is provided in [Appendix A](#A1 "Appendix A Datasets ‣ Revisiting Pretraining Objectives for Tabular Deep Learning").

Models. We use MLP as a simple deep baseline to compare and ablate the methods. Our implementation of MLP exactly follows [[16](#bib.bib16)], the model is regularized by dropout and weight decay. As more advanced deep models, we evaluate MLP equipped with numerical feature embeddings, specifically, target-aware piecewise linear encoding (MLP-T-LR) and embeddings with periodic activations (MLP-PLR) from [[14](#bib.bib14)]. These models represent the current state-of-the-art solution for tabular DL [[14](#bib.bib14)], and are of interest as most prior work on pretraining in tabular DL focus on pretraining with the simplest MLP models in evaluation. The implementation of models with numerical embeddings follows [[14](#bib.bib14)]. We use AdamW [[28](#bib.bib28)] optimizer, do not use learning rate schedules and fix batch sizes for each dataset based on the dataset size.

Pretraining. Pretraining is always performed directly on the target dataset and does not exploit additional data. The learning process thus comprises two stages. On the first stage, the model parameters are optimized w.r.t. the pretraining objective. On the second stage, the model is initialized with the pretrained weights and finetuned on the downstream classification or regression task. We focus on the fully-supervised setup, i.e., assume that target labels are provided for all dataset objects. Typically, pretraining stage involves the input corruption: for instance, to generate positive pairs in contrastive-like objectives or to corrupt the input for reconstruction in self-prediction based objectives. We use random feature resampling as a proven simple baseline for input corruption in tabular data [[4](#bib.bib4), [42](#bib.bib42)].
Learning rate and weight decay are shared between the two stages (see [Table 11](#A3.T11 "Table 11 ‣ Appendix C Share or split learning rate and weight decay between pretraining and finetuning? ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") for the ablation). We fix the maximum number of pretraining iterations for each dataset at 100​k100𝑘100k. On every 10​k10𝑘10k-th iteration, we compute the value of the pretraining objective using the hold-out validation objects for early-stopping on large-scale WE, CO and MI datasets. On other datasets we directly finetune the current model every 10​k10𝑘10k-th iteration and perform early-stopping based on the target metric after finetuning (we do not observe much difference between early stopping by loss or by downstream metric, see [Table 12](#A4.T12 "Table 12 ‣ Appendix D Early-stopping criterions ‣ Revisiting Pretraining Objectives for Tabular Deep Learning")).

Hyperparameters & Evaluation. Hyperparameter tuning is crucial for a fair comparison, therefore, we use Optuna [[1](#bib.bib1)] to optimize the model and pretraining hyperparameters for each method on each dataset. We use the validation subset of each dataset for hyperparameter tuning. The exact search spaces for the hyperparameters of each method are provided in [Appendix B](#A2 "Appendix B Hyperparameters ‣ Revisiting Pretraining Objectives for Tabular Deep Learning").

We run the tuned configuration of each pretraining method with 151515 random seeds and report the average metric on the test splits. When comparing to GBDT, we obtain three ensembles by splitting the fifteen single model predictions into three disjoint subsets of five models and averaging predictions within each subset. Then, we report the average metric over the three ensembles.

### 3.2 Comparing pretraining objectives

Here we compare the contrastive learning and self-prediction objectives from prior work in the described setup. For contrastive learning, we follow the method described in [[4](#bib.bib4)]: use InfoNCE loss, consider corrupted inputs x^^𝑥\hat{x} as positives for x𝑥x and the rest of the batch as negatives.
For self-prediction methods, we evaluate two objectives: the first one is the reconstruction of the original x𝑥x, given the corrupted input x^^𝑥\hat{x} (the reconstruction loss is computed for all columns), the second one is the binary mask prediction, where the objective is to predict the mask vector m𝑚m indicating the corrupted columns from the corrupted input x^^𝑥\hat{x}. The results of the comparison are in [Table 2](#S3.T2 "Table 2 ‣ 3.2 Comparing pretraining objectives ‣ 3 Revisiting pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning"). We summarize our key findings below.

Contrastive is not superior. Both the reconstruction and the mask prediction objectives are preferable to the contrastive objective.
The two self-prediction objectives have the advantage of being conceptually simpler, and easier to implement, while also being less resource-intensive (no need for the second view of augmented examples in each batch, simpler loss function).
We thus recommend the self-prediction based objectives as a practical solution for pretraining in tabular DL.

Pretraining is beneficial for the state-of-the-art models. Models with the numerical feature embeddings also benefit from pretraining with either reconstruction or mask prediction demonstrating the top performance on the downstream task. However, the improvement is typically less noticeable compared to the vanilla MLPs.

There is no universal solution between self-prediction objectives. We observe that for some datasets the reconstruction objective outperforms the mask prediction (OT, WE, CO, MI), while on others the mask prediction is better (GE, CH, HI, AD). We also note that the mask prediction objective sometimes leads to unexpected performance drops for models with numerical embeddings (WE, MI), we do not observe significant performance drops for the reconstruction objective.

The main takeaway: simple pretraining strategies based on self-prediction lead to significant improvements in the downstream accuracy compared to the tuned supervised baselines learned from scratch across different tabular DL models and datasets.
In practice, we recommend trying both reconstruction and mask prediction as tabular pretraining baselines, as either one might show superior performance depending on the dataset being used.

Table 2: Results for pretraining deep models with different objectives. We report metrics averaged over 15 seeds, bold entries correspond to results that are statistically significantly better (we use Tukey HSD test). The comparisons are separate for different models. ↑ corresponds to accuracy and ROC-AUC metrics, ↓ corresponds to RMSE and log-loss for OT. "no pretraining" stands for the supervised baseline, initialized with random weights

|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | OT ↓ | HI ↑ | FB ↓ | AD ↑ | WE ↓ | CO ↑ | MI ↓ |
| MLP | | | | | | | | | | | |
| no pretraining | 0.6350.6350.635 | 0.8490.8490.849 | 0.5060.5060.506 | 3.1563.1563.156 | 0.4790.4790.479 | 0.8010.8010.801 | 5.7375.7375.737 | 0.9080.9080.908 | 1.9091.9091.909 | 0.9630.9630.963 | 0.7490.7490.749 |
| contrastive | 0.6720.6720.672 | 0.8550.855\mathbf{0.855} | 0.4550.4550.455 | 3.0563.056\mathbf{3.056} | 0.4690.4690.469 | 0.8130.813\mathbf{0.813} | 5.6975.6975.697 | 0.9100.9100.910 | 1.8811.8811.881 | 0.9600.9600.960 | 0.7480.7480.748 |
| rec | 0.6620.6620.662 | 0.8530.8530.853 | 0.4450.445\mathbf{0.445} | 3.0443.044\mathbf{3.044} | 0.4660.466\mathbf{0.466} | 0.8050.8050.805 | 5.6415.641\mathbf{5.641} | 0.9100.910\mathbf{0.910} | 1.8751.875\mathbf{1.875} | 0.9650.965\mathbf{0.965} | 0.7460.746\mathbf{0.746} |
| mask | 0.6910.691\mathbf{0.691} | 0.8570.857\mathbf{0.857} | 0.4540.4540.454 | 3.1133.1133.113 | 0.4720.4720.472 | 0.8140.814\mathbf{0.814} | 5.6815.681\mathbf{5.681} | 0.9120.912\mathbf{0.912} | 1.8831.8831.883 | 0.9640.9640.964 | 0.7480.7480.748 |
| MLP-PLR | | | | | | | | | | | |
| no pretraining | 0.6680.6680.668 | 0.8580.858\mathbf{0.858} | 0.4690.4690.469 | 3.0083.008\mathbf{3.008} | 0.4830.4830.483 | 0.8090.8090.809 | 5.6085.608\mathbf{5.608} | 0.9260.9260.926 | 1.8901.8901.890 | 0.9690.9690.969 | 0.7460.7460.746 |
| rec | 0.6670.6670.667 | 0.8520.8520.852 | 0.4390.439\mathbf{0.439} | 3.0313.031\mathbf{3.031} | 0.4720.472\mathbf{0.472} | 0.8080.8080.808 | 5.5715.571\mathbf{5.571} | 0.9260.926\mathbf{0.926} | 1.8771.877\mathbf{1.877} | 0.9710.971\mathbf{0.971} | 0.7450.745\mathbf{0.745} |
| mask | 0.6850.685\mathbf{0.685} | 0.8630.863\mathbf{0.863} | 0.4340.434\mathbf{0.434} | 3.0073.007\mathbf{3.007} | 0.4770.4770.477 | 0.8180.818\mathbf{0.818} | 5.5865.586\mathbf{5.586} | 0.9270.927\mathbf{0.927} | 1.9111.9111.911 | 0.9700.970\mathbf{0.970} | 0.7480.7480.748 |
| MLP-T-LR | | | | | | | | | | | |
| no pretraining | 0.6340.6340.634 | 0.8660.866\mathbf{0.866} | 0.4440.4440.444 | 3.1133.1133.113 | 0.4820.4820.482 | 0.8050.8050.805 | 5.5205.5205.520 | 0.9250.9250.925 | 1.8971.8971.897 | 0.9680.9680.968 | 0.7490.7490.749 |
| rec | 0.6520.652\mathbf{0.652} | 0.8570.8570.857 | 0.4240.424\mathbf{0.424} | 3.1093.1093.109 | 0.4720.472\mathbf{0.472} | 0.8080.8080.808 | 5.3635.363\mathbf{5.363} | 0.9240.9240.924 | 1.8611.861\mathbf{1.861} | 0.9690.969\mathbf{0.969} | 0.7460.746\mathbf{0.746} |
| mask | 0.6540.654\mathbf{0.654} | 0.8680.868\mathbf{0.868} | 0.4240.424\mathbf{0.424} | 3.0453.045\mathbf{3.045} | 0.4720.472\mathbf{0.472} | 0.8180.818\mathbf{0.818} | 5.5445.5445.544 | 0.9260.926\mathbf{0.926} | 1.9161.9161.916 | 0.9690.969\mathbf{0.969} | 0.7480.7480.748 |

## 4 Target-aware pretraining objectives

In this section, we show that exploiting the target variables during the pretraining stage can further increase the downstream performance. Specifically, we evaluate several strategies to leverage information about targets during pretraining, identify the best ones and compare them to GBDT. Below we describe a list of target-aware pretraining objectives that we investigate.

Table 3: Variations of the target-aware pretraining schemes. Notation follows [Table 2](#S3.T2 "Table 2 ‣ 3.2 Comparing pretraining objectives ‣ 3 Revisiting pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning"). Bold results indicate statistically significant winners across all models and methods. "+ target" denotes target-conditioned pretraining, "+ sup" denotes auxiliary supervised head.

GE ↑
CH ↑
CA ↓
HO ↓
OT ↓
HI ↑
FB ↓
AD ↑
WE ↓
CO ↑
MI ↓
Avg. Rank

MLP

no pretraining
0.6350.6350.635
0.8490.8490.849
0.5060.5060.506
3.1563.1563.156
0.4790.4790.479
0.8010.8010.801
5.7375.7375.737
0.9080.9080.908
1.9091.9091.909
0.9630.9630.963
0.7490.7490.749
5.5±1.4plus-or-minus5.51.45.5\pm 1.4

mask
0.6910.6910.691
0.8570.8570.857
0.4540.4540.454
3.1133.1133.113
0.4720.4720.472
0.8140.8140.814
5.6815.6815.681
0.9120.9120.912
1.8831.8831.883
0.9640.9640.964
0.7480.7480.748
3.8±1.4plus-or-minus3.81.43.8\pm 1.4

rec
0.6620.6620.662
0.8530.8530.853
0.4450.4450.445
3.0443.044\mathbf{3.044}
0.4660.4660.466
0.8050.8050.805
5.6415.6415.641
0.9100.9100.910
1.8751.8751.875
0.9650.9650.965
0.7460.7460.746
3.6±1.5plus-or-minus3.61.53.6\pm 1.5

sup
0.6930.6930.693
0.8560.8560.856
0.4410.4410.441
3.0773.0773.077
0.4590.459\mathbf{0.459}
0.8140.8140.814
5.6895.6895.689
0.9140.9140.914
1.8831.8831.883
0.9680.9680.968
0.7480.7480.748
3.0±1.0plus-or-minus3.01.03.0\pm 1.0

mask + target
0.6830.6830.683
0.8570.8570.857
0.4340.4340.434
3.0563.056\mathbf{3.056}
0.4680.4680.468
0.8190.819\mathbf{0.819}
5.6335.6335.633
0.9140.9140.914
1.8761.8761.876
0.9650.9650.965
0.7480.7480.748
2.9±1.3plus-or-minus2.91.32.9\pm 1.3

rec + target
0.6590.6590.659
0.8530.8530.853
0.4540.4540.454
3.0443.044\mathbf{3.044}
0.4630.4630.463
0.8060.8060.806
5.6365.6365.636
0.9090.9090.909
1.8841.8841.884
0.9650.9650.965
0.7450.745\mathbf{0.745}
3.7±1.9plus-or-minus3.71.93.7\pm 1.9

mask + sup
0.6930.6930.693
0.8570.8570.857
0.4360.4360.436
3.0993.0993.099
0.4580.458\mathbf{0.458}
0.8170.8170.817
5.6855.6855.685
0.9150.9150.915
1.8731.8731.873
0.9670.9670.967
0.7480.7480.748
2.7±1.2plus-or-minus2.71.22.7\pm 1.2

rec + sup
0.6840.6840.684
0.8540.8540.854
0.4360.4360.436
3.0123.012\mathbf{3.012}
0.4560.456\mathbf{0.456}
0.8150.8150.815
5.6725.6725.672
0.9110.9110.911
1.8621.862\mathbf{1.862}
0.9670.9670.967
0.7470.7470.747
2.6±1.5plus-or-minus2.61.52.6\pm 1.5

MLP-PLR

no pretraining
0.6680.6680.668
0.8580.8580.858
0.4690.4690.469
3.0083.008\mathbf{3.008}
0.4830.4830.483
0.8090.8090.809
5.6085.6085.608
0.9260.9260.926
1.8901.8901.890
0.9690.9690.969
0.7460.7460.746
3.5±1.7plus-or-minus3.51.73.5\pm 1.7

mask
0.6850.6850.685
0.8630.863\mathbf{0.863}
0.4340.4340.434
3.0073.007\mathbf{3.007}
0.4770.4770.477
0.8180.8180.818
5.5865.5865.586
0.9270.9270.927
1.9111.9111.911
0.9700.9700.970
0.7480.7480.748
2.8±1.7plus-or-minus2.81.72.8\pm 1.7

rec
0.6670.6670.667
0.8520.8520.852
0.4390.4390.439
3.0313.031\mathbf{3.031}
0.4720.4720.472
0.8080.8080.808
5.5715.5715.571
0.9260.9260.926
1.8771.8771.877
0.9710.971\mathbf{0.971}
0.7450.745\mathbf{0.745}
2.6±1.2plus-or-minus2.61.22.6\pm 1.2

sup
0.7100.710\mathbf{0.710}
0.8590.8590.859
0.4330.4330.433
3.1363.1363.136
0.4790.4790.479
0.8110.8110.811
5.5215.5215.521
0.9240.9240.924
1.8731.8731.873
0.9710.971\mathbf{0.971}
0.7480.7480.748
2.5±1.2plus-or-minus2.51.22.5\pm 1.2

mask + target
0.6940.6940.694
0.8620.8620.862
0.4250.425\mathbf{0.425}
3.0233.023\mathbf{3.023}
0.4740.4740.474
0.8210.821\mathbf{0.821}
5.5375.5375.537
0.9290.929\mathbf{0.929}
1.9111.9111.911
0.9690.9690.969
0.7490.7490.749
2.5±1.9plus-or-minus2.51.92.5\pm 1.9

rec + target
0.6880.6880.688
0.8600.8600.860
0.4450.4450.445
3.0643.064\mathbf{3.064}
0.4750.4750.475
0.8120.8120.812
5.5075.5075.507
0.9270.9270.927
1.8871.8871.887
0.9710.971\mathbf{0.971}
0.7480.7480.748
2.7±1.3plus-or-minus2.71.32.7\pm 1.3

mask + sup
0.7110.711\mathbf{0.711}
0.8660.866\mathbf{0.866}
0.4410.4410.441
3.1293.1293.129
0.4800.4800.480
0.8130.8130.813
5.4805.4805.480
0.9250.9250.925
1.8751.8751.875
0.9690.9690.969
0.7450.745\mathbf{0.745}
2.5±1.4plus-or-minus2.51.42.5\pm 1.4

rec + sup
0.7090.709\mathbf{0.709}
0.8580.8580.858
0.4330.4330.433
3.0593.059\mathbf{3.059}
0.4650.4650.465
0.8070.8070.807
5.5715.5715.571
0.9270.9270.927
1.8651.865\mathbf{1.865}
0.9710.971\mathbf{0.971}
0.7450.745\mathbf{0.745}
1.9±1.2plus-or-minus1.91.21.9\pm 1.2

MLP-T-LR

no pretraining
0.6340.6340.634
0.8660.866\mathbf{0.866}
0.4440.4440.444
3.1133.1133.113
0.4820.4820.482
0.8050.8050.805
5.5205.5205.520
0.9250.9250.925
1.8971.8971.897
0.9680.9680.968
0.7490.7490.749
3.9±1.7plus-or-minus3.91.73.9\pm 1.7

mask
0.6540.6540.654
0.8680.868\mathbf{0.868}
0.4240.424\mathbf{0.424}
3.0453.045\mathbf{3.045}
0.4720.4720.472
0.8180.8180.818
5.5445.5445.544
0.9260.9260.926
1.9161.9161.916
0.9690.9690.969
0.7480.7480.748
2.8±1.7plus-or-minus2.81.72.8\pm 1.7

rec
0.6520.6520.652
0.8570.8570.857
0.4240.424\mathbf{0.424}
3.1093.1093.109
0.4720.4720.472
0.8080.8080.808
5.3635.363\mathbf{5.363}
0.9240.9240.924
1.8611.861\mathbf{1.861}
0.9690.9690.969
0.7460.746\mathbf{0.746}
2.5±1.4plus-or-minus2.51.42.5\pm 1.4

sup
0.6820.6820.682
0.8600.8600.860
0.4300.4300.430
3.1353.1353.135
0.4710.4710.471
0.8070.8070.807
5.5255.5255.525
0.9270.9270.927
1.8931.8931.893
0.9710.971\mathbf{0.971}
0.7470.7470.747
2.8±1.5plus-or-minus2.81.52.8\pm 1.5

mask + target
0.6490.6490.649
0.8650.865\mathbf{0.865}
0.4210.421\mathbf{0.421}
3.0583.058\mathbf{3.058}
0.4740.4740.474
0.8200.820\mathbf{0.820}
5.6445.6445.644
0.9290.929\mathbf{0.929}
1.9241.9241.924
0.9690.9690.969
0.7490.7490.749
2.8±2.1plus-or-minus2.82.12.8\pm 2.1

rec + target
0.6680.6680.668
0.8640.864\mathbf{0.864}
0.4400.4400.440
3.1133.1133.113
0.4730.4730.473
0.8060.8060.806
5.4935.4935.493
0.9270.9270.927
1.8621.862\mathbf{1.862}
0.9690.9690.969
0.7460.746\mathbf{0.746}
2.5±1.4plus-or-minus2.51.42.5\pm 1.4

mask + sup
0.6760.6760.676
0.8580.8580.858
0.4290.4290.429
3.1993.1993.199
0.4680.4680.468
0.8140.8140.814
5.5105.5105.510
0.9260.9260.926
1.8691.8691.869
0.9710.971\mathbf{0.971}
0.7480.7480.748
2.5±0.8plus-or-minus2.50.82.5\pm 0.8

rec + sup
0.6780.6780.678
0.8650.865\mathbf{0.865}
0.4370.4370.437
3.1123.1123.112
0.4620.4620.462
0.8070.8070.807
5.5165.5165.516
0.9270.9270.927
1.8621.862\mathbf{1.862}
0.9700.970\mathbf{0.970}
0.7480.7480.748
2.4±1.2plus-or-minus2.41.22.4\pm 1.2

  

Supervised loss with augmentations. A straightforward way to incorporate the target variable into the pretraining is by using the input corruption as an augmentation for the standard supervised learning objective. An important difference of this baseline in our setup to the one in [[4](#bib.bib4)] is that we treat learning on corrupted samples as a pretraining stage and finetune the entire model on the full uncorrupted dataset afterwards (we ablate this in [subsection 5.3](#S5.SS3 "5.3 On importance of finetuning on clean data ‣ 5 Analysis ‣ Revisiting Pretraining Objectives for Tabular Deep Learning")).

Supervised loss with augmentations + self-prediction. We evaluate a natural extension to the above baseline: a combination of the supervised objective with the unsupervised self-prediction.
Note, that during the pretraining stage both losses are calculated on corrupted inputs, while the finetuning is performed on the non-corrupted dataset. For the self-prediction objectives we evaluate both the reconstruction and the mask prediction. We use different prediction heads for supervised and self-prediction objectives. We sum supervised and self-prediction losses with equal weights.

Target-aware pretraining. An alternative to the approaches described above is the modification of the pretraining task itself. An example of this approach is supervised contrastive learning [[24](#bib.bib24)], where the target variable is used to sample positive and negative examples. We introduce the target variable into the self-prediction based objectives with two modifications.

First, we condition the mask prediction or the reconstruction head on the original input’s target by concatenating the hidden representation from the backbone network z=f​(x^)𝑧𝑓^𝑥z=f(\hat{x}) with the target variable representation before passing it to the pretraining head to obtain predictions p=h​(𝚌𝚘𝚗𝚌𝚊𝚝​[z,y])𝑝ℎ𝚌𝚘𝚗𝚌𝚊𝚝𝑧𝑦p=h(\mathtt{concat}[z,y]). For classification datasets we encode y𝑦y with one-hot-encoding, for regression targets we use the standard scaling.

Second, we change the input corruption scheme by sampling the replacement from the feature target conditional distribution where a target is different to the original. Intuitively, corrupting the input object x𝑥x in the direction of the target different to the original makes the pretraining task more correlated with the downstream target prediction. Concretely, given an object-target pair (xi,yi)subscript𝑥𝑖subscript𝑦𝑖(x\_{i},y\_{i}), we sample a new target y^isubscript^𝑦𝑖\hat{y}\_{i} from a uniform distribution over the set {y|y≠yi}conditional-set𝑦𝑦subscript𝑦𝑖\{y\ |\ y\neq y\_{i}\} 111For regression problems, when the target variable is continuous, we preliminarily discretize it into n𝑛n uniform bins, where n𝑛n is chosen according to the Freedman–Diaconis rule [[13](#bib.bib13)]., then each feature xjsuperscript𝑥𝑗x^{j} is replaced with a sample from the p​(xj|y^i)𝑝conditionalsuperscript𝑥𝑗subscript^𝑦𝑖p(x^{j}|\hat{y}\_{i}) distribution, instead of p​(xj)𝑝superscript𝑥𝑗p(x^{j}).

### 4.1 Comparing target-aware objectives

Here we compare the strategies of incorporating the target variable into pretraining.
The results of the comparison are in [Table 3](#S4.T3 "Table 3 ‣ 4 Target-aware pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning"). Our key findings are formulated below.

Supervised loss with augmentations is another strong baseline for MLP. Pretraining with the supervised loss on corrupted data consistently improves over supervised training from scratch for the MLP. This objective is a strong baseline along with the self-prediction based objectives. However, for models with numerical embeddings the supervised objective with corruptions is less consistent and sometimes is inferior to training from scratch, thus for these models we recommend the self-prediction objectives alone as baselines.

Target-aware objectives demonstrate the best performance.
Both the supervised loss with self-prediction and modified self-prediction objectives improve over the unsupervised pretraining baselines across datasets and model architectures.

For the objective with the combination of supervised and self-prediction losses the variation with the reconstruction loss is the most consistent across models and dataset with no performance drops below the pretraining-free baseline. The variant with mask prediction shows similar performance and stability for the MLP, but is not as good as reconstruction for models with numerical embeddings.

For the target-aware self-prediction objectives, the modified mask prediction delivers significant improvements over its unsupervised counterpart. Modified reconstruction objective, however, does not improve over unsupervised reconstruction objective.

Main takeaways:
Target-aware objectives help further increase the downstream performance, improving upon their unsupervised counterparts. For the reconstruction based self-prediction baseline the addition of the supervised loss is most beneficial ("rec + sup" from [Table 3](#S4.T3 "Table 3 ‣ 4 Target-aware pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning")), for the mask prediction objective it’s target-aware modification provides more improvements ("mask + target" from [Table 3](#S4.T3 "Table 3 ‣ 4 Target-aware pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning")).
A simple MLP model pretrained with those "target-aware" objectives often reaches or surpasses complex models with numerical embeddings trained from scratch.
In practice, we recommend first trying the baseline pretraining objectives ("rec", "mask", "sup" from [Table 3](#S4.T3 "Table 3 ‣ 4 Target-aware pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning")), choosing the suitable baseline for the dataset and improving it accordingly: supervised loss for the reconstruction and target-aware modification for the mask prediction.

### 4.2 Comparison to GBDT

Here we compare MLPs and MLPs with numerical feature embeddings pretrained with the supervised loss with reconstruction and the target-aware mask prediction objectives to the GBDTs. [Table 4](#S4.T4 "Table 4 ‣ 4.2 Comparison to GBDT ‣ 4 Target-aware pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") shows the results of the comparison.

Table 4: Comparison of pretrained models to GBDT. Notation follows [Table 2](#S3.T2 "Table 2 ‣ 3.2 Comparing pretraining objectives ‣ 3 Revisiting pretraining objectives ‣ Revisiting Pretraining Objectives for Tabular Deep Learning"). Results represent ensembles of models. Bold entries correspond to the overall statistically significant best entries.

GE ↑
CH ↑
CA ↓
HO ↓
OT ↓
HI ↑
FB ↓
AD ↑
WE ↓
CO ↑
MI ↓
Avg. Rank

CatBoost
0.6920.6920.692
0.8640.864\mathbf{0.864}
0.4300.4300.430
3.0933.0933.093
0.4500.4500.450
0.8070.8070.807
5.2265.2265.226
0.9280.9280.928
1.8011.8011.801
0.9670.9670.967
0.7410.741\mathbf{0.741}
2.6±1.4plus-or-minus2.61.42.6\pm 1.4

XGBoost
0.6830.6830.683
0.8600.8600.860
0.4340.4340.434
3.1523.1523.152
0.4540.4540.454
0.8050.8050.805
5.3385.3385.338
0.9270.9270.927
1.7821.782\mathbf{1.782}
0.9690.9690.969
0.7420.742\mathbf{0.742}
3.0±1.5plus-or-minus3.01.53.0\pm 1.5

MLP

no pretraining
0.6560.6560.656
0.8520.8520.852
0.4820.4820.482
3.0553.0553.055
0.4670.4670.467
0.8050.8050.805
5.6665.6665.666
0.9100.9100.910
1.8501.8501.850
0.9680.9680.968
0.7470.7470.747
4.8±1.1plus-or-minus4.81.14.8\pm 1.1

mask + target
0.7090.7090.709
0.8600.8600.860
0.4140.4140.414
2.9492.9492.949
0.4570.4570.457
0.8280.828\mathbf{0.828}
5.5515.5515.551
0.9160.9160.916
1.8091.8091.809
0.9690.9690.969
0.7460.7460.746
2.8±1.2plus-or-minus2.81.22.8\pm 1.2

rec + sup
0.7090.7090.709
0.8590.8590.859
0.4190.4190.419
2.9512.9512.951
0.4420.442\mathbf{0.442}
0.8170.8170.817
5.5315.5315.531
0.9130.9130.913
1.8011.8011.801
0.9730.9730.973
0.7450.7450.745
2.5±1.2plus-or-minus2.51.22.5\pm 1.2

MLP-P-LR

no pretraining
0.6950.6950.695
0.8640.864\mathbf{0.864}
0.4540.4540.454
2.9532.9532.953
0.4700.4700.470
0.8140.8140.814
5.3245.3245.324
0.9280.9280.928
1.8351.8351.835
0.9740.974\mathbf{0.974}
0.7440.7440.744
2.6±1.2plus-or-minus2.61.22.6\pm 1.2

mask + target
0.7190.719\mathbf{0.719}
0.8660.866\mathbf{0.866}
0.4070.407\mathbf{0.407}
2.9522.9522.952
0.4580.4580.458
0.8280.828\mathbf{0.828}
5.3735.3735.373
0.9300.930\mathbf{0.930}
1.8491.8491.849
0.9730.9730.973
0.7450.7450.745
2.1±1.2plus-or-minus2.11.22.1\pm 1.2

rec + sup
0.7370.737\mathbf{0.737}
0.8620.8620.862
0.4240.4240.424
2.9642.9642.964
0.4490.4490.449
0.8110.8110.811
5.1245.124\mathbf{5.124}
0.9290.929\mathbf{0.929}
1.8131.8131.813
0.9740.974\mathbf{0.974}
0.7440.7440.744
2.0±1.0plus-or-minus2.01.02.0\pm 1.0

MLP-T-LR

no pretraining
0.6620.6620.662
0.8680.868\mathbf{0.868}
0.4370.4370.437
3.0283.0283.028
0.4720.4720.472
0.8080.8080.808
5.4245.4245.424
0.9270.9270.927
1.8501.8501.850
0.9720.9720.972
0.7470.7470.747
3.7±1.1plus-or-minus3.71.13.7\pm 1.1

mask + target
0.6730.6730.673
0.8680.868\mathbf{0.868}
0.4100.410\mathbf{0.410}
2.8942.894\mathbf{2.894}
0.4600.4600.460
0.8270.827\mathbf{0.827}
5.4585.4585.458
0.9300.930\mathbf{0.930}
1.8491.8491.849
0.9720.9720.972
0.7460.7460.746
2.4±1.4plus-or-minus2.41.42.4\pm 1.4

rec + sup
0.7050.7050.705
0.8660.866\mathbf{0.866}
0.4250.4250.425
3.0573.0573.057
0.4440.444\mathbf{0.444}
0.8140.8140.814
5.4225.4225.422
0.9270.9270.927
1.8111.8111.811
0.9740.974\mathbf{0.974}
0.7460.7460.746
2.5±1.1plus-or-minus2.51.12.5\pm 1.1

We observe that both pretraining with target aware objectives and using numerical feature embeddings consistently improve the performance of the simple MLP backbone. In particular, MLP coupled with target aware pretraining starts to outperform GBDT on 4 datasets (GE, CA, OT, HI). Combined with numerical feature embeddings, pretraining improves MLP performance further, making it superior to GBDT on the majority of the datasets, with two exceptions in WE and MI.

## 5 Analysis

### 5.1 Investigating the properties of pretrained models

In this section, we provide a possible explanation of why the incorporation of the target variable into pretraining can lead to better downstream task performance. We do this through the experiments on the controllable synthetic data. Here we describe the properties and the generation process of the data and our observations on the differences of the pretraining schemes.

!(/html/2207.03208/assets/x1.png)

Figure 1: The decodability of object feature from the intermediate representations computed by the pretrained models and the models trained from scratch. The pretrained models decently capture the information about all the features, while the randomly initialized models capture the most informative features and suppress the others.

We follow the synthetics generation protocol described in [[16](#bib.bib16)] with a modification that allows for the manual control of the feature importance for the particular prediction task. Concretely, we generate the objects features {xi}i=1nsuperscriptsubscriptsubscript𝑥𝑖𝑖1𝑛\left\{{x\_{i}}\right\}\_{i=1}^{n} as samples from the multivariate Gaussian distribution with zero mean and covariance matrix ΣΣ\Sigma with identical diagonal and a constant c=0.5𝑐0.5c=0.5 everywhere else. To generate the corresponding objects targets {yi}i=1nsuperscriptsubscriptsubscript𝑦𝑖𝑖1𝑛\left\{{y\_{i}}\right\}\_{i=1}^{n} we sample a vector p∈ℝm𝑝superscriptℝ𝑚p\in\mathbb{R}^{m} from the Dirichlet distribution p∼Dir​(1m)similar-to𝑝Dirsubscript1𝑚p\sim\text{Dir}(1\_{m}) and let p𝑝p define the influence of the objects features on the target. Then, we build an ensemble of 101010 random oblivious decision trees {Ti​(x)}i=110superscriptsubscriptsubscript𝑇𝑖𝑥𝑖110\left\{T\_{i}(x)\right\}\_{i=1}^{10} of depth 101010, where on each tree level we sample a feature j∼Cat​(p)similar-to𝑗Cat𝑝j\sim\text{Cat}(p) and a threshold t∼Unif​(min​(xj),max​(xj))similar-to𝑡Unifminsuperscript𝑥𝑗maxsuperscript𝑥𝑗t\sim\mathrm{Unif}(\mathrm{min}(x^{j}),\mathrm{max}(x^{j})) for a decision rule. For each tree leaf, we sample a scalar l∼𝒩​(0,1)similar-to𝑙𝒩01l\sim\mathcal{N}(0,1), representing a logit in binary classification. We define the binary targets as follows: y​(x)=I​{110​∑i=110Ti​(x)>0}𝑦𝑥I110superscriptsubscript𝑖110subscript𝑇𝑖𝑥0y(x)=\mathrm{I}\left\{\frac{1}{10}\sum\_{i=1}^{10}T\_{i}(x)>0\right\}.

Intuitively if a particular feature is often used for splitting in the nodes of a decision tree, it would have more influence on the target variable. Indeed, we find the feature importances222Computed with the CatBoost method “get\_feature\_importance()” correlate well with the predefined vector p𝑝p. We set the size of the dataset n=50.000𝑛50.000n=50.000 and the number of features m=8𝑚8m=8 and generate 505050 datasets with different feature importance vectors p𝑝p for the analysis.

For each generated dataset, we then check whether the finetuned models capture the information about object features in their intermediate representations. Specifically, we train an MLP to predict the value of the i𝑖i-th object feature given the frozen embeddings produced by a finetuned network initialized from (a) random initialization, (b) mask prediction pretraining, (c) target-aware mask prediction pretraining. The separate MLP is used for each feature and the RMSE learning objective is used. Then we report the RMSE on the test set for all features i∈[0,m]𝑖0𝑚i\in[0,m] along with their importance rank in the dataset on [Figure 1](#S5.F1 "Figure 1 ‣ 5.1 Investigating the properties of pretrained models ‣ 5 Analysis ‣ Revisiting Pretraining Objectives for Tabular Deep Learning"). Here, the lower ranks correspond to the more important features.

[Figure 1](#S5.F1 "Figure 1 ‣ 5.1 Investigating the properties of pretrained models ‣ 5 Analysis ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") reveals that the target-aware pretraining enables the model to capture more information about the informative features compared to the “unsupervised” pretraining and, especially, to the learning from scratch. The latter one successfully captures the most informative feature from the training data, while suppressing the less important, but still significant features. We conjecture that this is the source of superiority of the target-aware pretraining.

### 5.2 Efficient ensembling

In this section we show, that it is possible to construct ensembles from one pretraining checkpoint (pretrained with the target conditioned mask prediction objective). To this end, we run finetuning with 151515 different random seeds starting from the one pretrained checkpoint. [Table 5](#S5.T5 "Table 5 ‣ 5.2 Efficient ensembling ‣ 5 Analysis ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") shows the results.

Table 5: Efficient ensembling for MLP mask + target

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | OT ↓ | HI ↑ | FB ↓ | AD ↑ | WE ↓ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.6830.6830.683 | 0.8570.8570.857 | 0.4340.4340.434 | 3.0563.0563.056 | 0.4680.4680.468 | 0.8190.8190.819 | 5.6335.6335.633 | 0.9140.9140.914 | 1.8761.8761.876 | 0.9650.9650.965 | 0.7480.7480.748 |
| standard ensemble | 0.7090.7090.709 | 0.8600.8600.860 | 0.4140.4140.414 | 2.9492.9492.949 | 0.4570.4570.457 | 0.8280.8280.828 | 5.5515.5515.551 | 0.9160.9160.916 | 1.8091.8091.809 | 0.9690.9690.969 | 0.7460.7460.746 |
| efficient ensemble | 0.7020.7020.702 | 0.8610.8610.861 | 0.4110.4110.411 | 2.9672.9672.967 | 0.4610.4610.461 | 0.8250.8250.825 | 5.5905.5905.590 | 0.9170.9170.917 | 1.8201.8201.820 | 0.9690.9690.969 | 0.7460.7460.746 |

Both ensembling the models from a shared pretrain checkpoint and ensembling multiple independent pretraining runs produces strong ensembles, which shows that it is sufficient to pretrain once and create ensembles by several independent finetuning processes. This is important in practice since finetuninig is typically cheaper (i.e. requires fewer iterations), and still is able to produce diverse models from the one pretraining checkpoint for ensembles of comparable quality.

### 5.3 On importance of finetuning on clean data

Here we show, that the second stage of finetuning the model on the entire dataset without input corruption is often necessary for the best downstream performance. To this end we compare finetuning the models on clean data with using models right after pretraining for two objectives: supervised loss and supervised loss with the reconstruction objective.

Table 6: Finetuning MLP on clean data versus using the model trained on corrupted inputs only

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | OT ↓ | HI ↑ | FB ↓ | AD ↑ | WE ↓ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no pretraining | 0.6350.6350.635 | 0.8490.8490.849 | 0.5060.5060.506 | 3.1563.1563.156 | 0.4790.4790.479 | 0.8010.8010.801 | 5.7375.7375.737 | 0.9080.9080.908 | 1.9091.9091.909 | 0.9630.9630.963 | 0.7490.7490.749 |
| sup | 0.6930.6930.693 | 0.8560.8560.856 | 0.4410.4410.441 | 3.0773.0773.077 | 0.4590.4590.459 | 0.8140.8140.814 | 5.6895.6895.689 | 0.9140.9140.914 | 1.8831.8831.883 | 0.9680.9680.968 | 0.7480.7480.748 |
| sup | no finetune | 0.6740.6740.674 | 0.8530.8530.853 | 0.4640.4640.464 | 3.2513.2513.251 | 0.4610.4610.461 | 0.8080.8080.808 | 6.1676.1676.167 | 0.9100.9100.910 | 1.9381.9381.938 | 0.9580.9580.958 | 0.7520.7520.752 |
| rec + sup | 0.6840.6840.684 | 0.8540.8540.854 | 0.4360.4360.436 | 3.0123.0123.012 | 0.4560.4560.456 | 0.8150.8150.815 | 5.6725.6725.672 | 0.9110.9110.911 | 1.8621.8621.862 | 0.9670.9670.967 | 0.7470.7470.747 |
| rec + sup | no finetune | 0.6830.6830.683 | 0.8530.8530.853 | 0.4670.4670.467 | 3.2323.2323.232 | 0.4740.4740.474 | 0.8110.8110.811 | 6.0446.0446.044 | 0.9070.9070.907 | 1.9011.9011.901 | 0.9560.9560.956 | 0.7520.7520.752 |

Across all datasets for both methods finetuning on uncorrupted data with the supervised loss proves to be essential for the best performance. Sometimes excluding the second finetuning stage degrades the performance below the tuned supervised baseline of training from scratch.

### 5.4 Does pretraining require more compute?

In this section we investigate how much more time is spent on pretraining, compared to training the models from scratch. We run pretraining with "rec + sup" objective with 50k, 100k and 150k pretraining iterations thresholds (early-stopping, in theory, could make 100k and more iterations equivalent to the 50k, but in practice it was not the case). We report downstream performance along with average time spent to pretrain and finetune a model in [Table 7](#S5.T7 "Table 7 ‣ 5.4 Does pretraining require more compute? ‣ 5 Analysis ‣ Revisiting Pretraining Objectives for Tabular Deep Learning").

Table 7: Comparison of time spent for MLP training from scratch and "rec + sup" pretraining with 50k, 100k and 150k max-iterations threshold. Second row in each group reports average time spent training one model in seconds on an A100 GPU.

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | OT ↓ | HI ↑ | FB ↓ | AD ↑ | WE ↓ | CO ↑ | MI ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no pretraining | 0.6350.6350.635 | 0.8490.8490.849 | 0.5060.5060.506 | 3.1563.1563.156 | 0.4790.4790.479 | 0.8010.8010.801 | 5.7375.7375.737 | 0.9080.9080.908 | 1.9091.9091.909 | 0.9630.9630.963 | 0.7490.7490.749 |
|  | 29s | 10s | 25s | 26s | 38s | 29s | 159s | 12s | 57s | 352s | 211s |
| rec + sup | 50k | 0.6790.6790.679 | 0.8570.8570.857 | 0.4410.4410.441 | 3.0643.0643.064 | 0.4620.4620.462 | 0.8130.8130.813 | 5.6505.6505.650 | 0.9100.9100.910 | 1.8791.8791.879 | 0.9660.9660.966 | 0.7470.7470.747 |
|  | 327s | 227s | 280s | 277s | 355s | 256s | 624s | 403s | 242s | 593s | 292s |
| rec + sup | 100k | 0.6840.6840.684 | 0.8540.8540.854 | 0.4360.4360.436 | 3.0123.0123.012 | 0.4560.4560.456 | 0.8150.8150.815 | 5.6725.6725.672 | 0.9110.9110.911 | 1.8621.8621.862 | 0.9670.9670.967 | 0.7470.7470.747 |
|  | 661s | 312s | 533s | 455s | 570s | 526s | 740s | 759s | 439s | 649s | 472s |
| rec + sup | 150k | 0.6920.6920.692 | 0.8590.8590.859 | 0.4350.4350.435 | 3.0123.0123.012 | 0.4560.4560.456 | 0.8160.8160.816 | 5.6295.6295.629 | 0.9100.9100.910 | 1.8661.8661.866 | 0.9680.9680.968 | 0.7460.7460.746 |
|  | 891s | 338s | 758s | 509s | 712s | 862s | 916s | 1069s | 663s | 927s | 665s |

Pretraining often requires by an order of magnitude more compute, it is especially apparent on smaller scale datasets like GE, CH, CA, HO, OT, HI, AD. However, the absolute time spent on pretraining is still acceptable, as the original training from scratch takes seconds on small datasets. Generally the more iterations you use for pretraining, the better downstream quality you get.

## 6 Conclusion

In this work, we have systematically evaluated typical pretraining objectives for tabular deep learning. We have revealed several important recipes for optimal pretraining performance that can be universally beneficial across various problems and models. Our findings confirm that pretraining can significantly improve the performance of tabular deep models and provide additional evidence that tabular DL can become a strong alternative to GBDT.

## References

* [1]

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In KDD, 2019.
* [2]

  Sercan O. Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  arXiv, 1908.07442v5, 2020.
* [3]

  Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and
  Sathiya S. Keerthi.
  Gradient boosting neural networks: Grownet.
  arXiv, 2002.07971v2, 2020.
* [4]

  Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler.
  Scarf: Self-supervised contrastive learning using random feature
  corruption.
  In International Conference on Learning Representations, 2022.
* [5]

  P. Baldi, P. Sadowski, and D. Whiteson.
  Searching for exotic particles in high-energy physics with deep
  learning.
  Nature Communications, 5, 2014.
* [6]

  Jock A. Blackard and Denis J. Dean.
  Comparative accuracies of artificial neural networks and discriminant
  analysis in predicting forest cover types from cartographic variables.
  Computers and Electronics in Agriculture, 24(3):131–151, 2000.
* [7]

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In SIGKDD, 2016.
* [8]

  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
  A simple framework for contrastive learning of visual
  representations.
  In International conference on machine learning, pages
  1597–1607. PMLR, 2020.
* [9]

  Sajad Darabi, Shayan Fazeli, Ali Pazoki, Sriram Sankararaman, and Majid
  Sarrafzadeh.
  Contrastive mixup: Self-and semi-supervised learning for tabular
  domain.
  arXiv preprint arXiv:2108.12296, 2021.
* [10]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  arXiv, 1810.04805v2, 2019.
* [11]

  Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé
  Jegou, and Edouard Grave.
  Are large-scale datasets necessary for self-supervised pre-training?
  arXiv preprint arXiv:2112.10740, 2021.
* [12]

  Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal
  Vincent, and Samy Bengio.
  Why does unsupervised pre-training help deep learning?
  J. Mach. Learn. Res., 11:625–660, mar 2010.
* [13]

  David Freedman and Persi Diaconis.
  On the histogram as a density estimator:l 2 theory.
  Z. Wahrscheinlichkeitstheorie verw Gebiete, 57(4):453–476,
  December 1981.
* [14]

  Yura Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  arXiv preprint arXiv:2203.05556, 2022.
* [15]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  In NeurIPS, 2021.
* [16]

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  Advances in Neural Information Processing Systems, 34, 2021.
* [17]

  Hussein Hazimeh, Natalia Ponomareva, Petros Mol, Zhenyu Tan, and Rahul
  Mazumder.
  The tree ensemble layer: Differentiability meets conditional
  computation.
  In ICML, 2020.
* [18]

  Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross
  Girshick.
  Masked autoencoders are scalable vision learners.
  arXiv preprint arXiv:2111.06377, 2021.
* [19]

  Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.
  Momentum contrast for unsupervised visual representation learning.
  In Proceedings of the IEEE/CVF conference on computer vision and
  pattern recognition, pages 9729–9738, 2020.
* [20]

  Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  arXiv, 2012.06678v1, 2020.
* [21]

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In NeurIPS, 2021.
* [22]

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei
  Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  Advances in neural information processing systems,
  30:3146–3154, 2017.
* [23]

  R. Kelley Pace and Ronald Barry.
  Sparse spatial autoregressions.
  Statistics & Probability Letters, 33(3):291–297, 1997.
* [24]

  Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip
  Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan.
  Supervised contrastive learning.
  Advances in Neural Information Processing Systems,
  33:18661–18673, 2020.
* [25]

  Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
  Self-normalizing neural networks.
  In NIPS, 2017.
* [26]

  Ron Kohavi.
  Scaling up the accuracy of naive-bayes classifiers: a decision-tree
  hybrid.
  In KDD, 1996.
* [27]

  Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin
  Gal.
  Self-attention between datapoints: Going beyond individual
  input-output pairs in deep learning.
  In NeurIPS, 2021.
* [28]

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In ICLR, 2019.
* [29]

  Renata C. B. Madeo, Clodoaldo Ap. M. Lima, and Sarajane Marques Peres.
  Gesture unit segmentation using support vector machines: segmenting
  gestures from rest positions.
  In Proceedings of the 28th Annual ACM Symposium on Applied
  Computing, SAC, 2013.
* [30]

  Andrey Malinin, Neil Band, German Chesnokov, Yarin Gal, Mark John Francis
  Gales, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan
  Provilkov, Vatsal Raina, Vyas Raina, Mariya Shmatova, Panos Tigas, and Boris
  Yangel.
  Shifts: A dataset of real distributional shift across multiple
  large-scale tasks.
  ArXiv, abs/2107.07455, 2021.
* [31]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  Journal of Machine Learning Research, 12:2825–2830, 2011.
* [32]

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular
  data.
  In ICLR, 2020.
* [33]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In NeurIPS, 2018.
* [34]

  Tao Qin and Tie-Yan Liu.
  Introducing LETOR 4.0 datasets.
  arXiv, 1306.2597v1, 2013.
* [35]

  Ravid Shwartz-Ziv and Amitai Armon.
  Tabular data: Deep learning is not all you need.
  arXiv, 2106.03253v1, 2021.
* [36]

  Kamaljot Singh, Ranjeet Kaur Sandhu, and Dinesh Kumar.
  Comment volume prediction using neural networks and decision trees.
  In IEEE UKSim-AMSS 17th International Conference on Computer
  Modelling and Simulation, UKSim, 2015.
* [37]

  Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom
  Goldstein.
  SAINT: improved neural networks for tabular data via row attention
  and contrastive pre-training.
  arXiv, 2106.01342v1, 2021.
* [38]

  Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and
  Jian Tang.
  Autoint: Automatic feature interaction learning via self-attentive
  neural networks.
  In CIKM, 2019.
* [39]

  Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards.
  Subtab: Subsetting features of tabular data for self-supervised
  representation learning.
  Advances in Neural Information Processing Systems, 34, 2021.
* [40]

  Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luís Torgo.
  Openml: networked science in machine learning.
  arXiv, 1407.7722v1, 2014.
* [41]

  Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang.
  Deep & cross network for ad click predictions.
  In ADKDD, 2017.
* [42]

  Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar.
  Vime: Extending the success of self- and semi-supervised learning to
  tabular domain.
  In NeurIPS, 2020.

## Appendix A Datasets

We used the following datasets:

* •

  Gesture Phase Prediction ([[29](#bib.bib29)])
* •

  Churn Modeling333https://www.kaggle.com/shrutimechlearn/churn-modelling
* •

  California Housing (real estate data, [[23](#bib.bib23)])
* •

  House 16H444https://www.openml.org/d/574
* •

  Adult (income estimation, [[26](#bib.bib26)])
* •

  Otto Group Product Classification555https://www.kaggle.com/c/otto-group-product-classification-challenge/data
* •

  Higgs (simulated physical particles, [[5](#bib.bib5)]; we use the version with 98K samples available at the OpenML repository [[40](#bib.bib40)])
* •

  Facebook Comments ([[36](#bib.bib36)])
* •

  Covertype (forest characteristics, [[6](#bib.bib6)])
* •

  Microsoft (search queries, [[34](#bib.bib34)]). We follow the pointwise approach to learning-to-rank and treat this ranking problem as a regression problem.
* •

  Weather (temperature, [[30](#bib.bib30)]). We take 10% of the dataset for our experiments due to the its large size.

## Appendix B Hyperparameters

### B.1 CatBoost

We fix and do not tune the following hyperparameters:

* •

  early-stopping-rounds=50early-stopping-rounds50\texttt{early-stopping-rounds}=50
* •

  od-pval=0.001od-pval0.001\texttt{od-pval}=0.001
* •

  iterations=2000iterations2000\texttt{iterations}=2000

For tuning on the MI and CO datasets, we set the task\_type parameter to “GPU”. In all other cases (including the evaluation on these two datasets), we set this parameter to “CPU”.

Table 8: CatBoost hyperparameter space

| Parameter | Distribution |
| --- | --- |
| Max depth | UniformInt​[1,10]UniformInt110\mathrm{UniformInt[1,10]} |
| Learning rate | LogUniform​[0.001,1]LogUniform0.0011\mathrm{LogUniform}[0.001,1] |
| Bagging temperature | Uniform​[0,1]Uniform01\mathrm{Uniform}[0,1] |
| L2 leaf reg | LogUniform​[1,10]LogUniform110\mathrm{LogUniform}[1,10] |
| Leaf estimation iterations | UniformInt​[1,10]UniformInt110\mathrm{UniformInt}[1,10] |
| # Iterations | 100 |

### B.2 XGBoost

We fix and do not tune the following hyperparameters:

* •

  booster="gbtree"booster"gbtree"\texttt{booster}=\text{"gbtree"}
* •

  early-stopping-rounds=50early-stopping-rounds50\texttt{early-stopping-rounds}=50
* •

  n-estimators=2000n-estimators2000\texttt{n-estimators}=2000

Table 9: XGBoost hyperparameter space.

| Parameter | Distribution |
| --- | --- |
| Max depth | UniformInt​[3,10]UniformInt310\mathrm{UniformInt[3,10]} |
| Min child weight | LogUniform​[0.0001,100]LogUniform0.0001100\mathrm{LogUniform}[0.0001,100] |
| Subsample | Uniform​[0.5,1]Uniform0.51\mathrm{Uniform}[0.5,1] |
| Learning rate | LogUniform​[0.001,1]LogUniform0.0011\mathrm{LogUniform}[0.001,1] |
| Col sample by tree | Uniform​[0.5,1]Uniform0.51\mathrm{Uniform}[0.5,1] |
| Gamma | {0,LogUniform​[0.001,100]}0LogUniform0.001100\{0,\mathrm{LogUniform}[0.001,100]\} |
| Lambda | {0,LogUniform​[0.1,10]}0LogUniform0.110\{0,\mathrm{LogUniform}[0.1,10]\} |
| # Iterations | 100 |

### B.3 MLP

We fix and do not tune the following hyperparameters:

* •

  Layer size=512Layer size512\texttt{Layer size}=512
* •

  Head hidden size=512Head hidden size512\texttt{Head hidden size}=512

Table 10: MLP hyperparameter space.

| Parameter | Distribution |
| --- | --- |
| # Layers | UniformInt​[1,8]UniformInt18\mathrm{UniformInt}[1,8] |
| Dropout | {0,Uniform​[0,0.5]}0Uniform00.5\{0,\mathrm{Uniform}[0,0.5]\} |
| Learning rate | LogUniform​[5​e​-​5,0.005]LogUniform5𝑒-50.005\mathrm{LogUniform}[5e\text{-}5,0.005] |
| Weight decay | {0,LogUniform​[1​e​-​6,1​e​-​3]}0LogUniform1𝑒-61𝑒-3\{0,\mathrm{LogUniform}[1e\text{-}6,1e\text{-}3]\} |
| Corrupt Probability | {0,Uniform​[0.2,0.8]}0Uniform0.20.8\{0,\mathrm{Uniform}[0.2,0.8]\} |
| # Iterations | 100 |

### B.4 Embedding Hyperparameters

We fix and do not tune the following hyperparameters:

* •

  Layer size=512Layer size512\texttt{Layer size}=512
* •

  Head hidden size=512Head hidden size512\texttt{Head hidden size}=512

The distribution for the output dimensions of linear layers is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128].

PLR, T-LR. We share the same hyperparameter space for models with embeddings across all datasets.

For the target-aware embeddings (tree-based) T-LR, the distribution for the number of leaves is UniformInt​[2,256]UniformInt2256\mathrm{UniformInt}[2,256], the distribution for the minimum number of items per leaf is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128] and the distribution for the minimum information gain required for making a split is LogUniform​[1​e​-​9,0.01]LogUniform1𝑒-90.01\mathrm{LogUniform}[1e\text{-}9,0.01].

For the periodic embeddings PLR. The distribution for k𝑘k is UniformInt​[1,128]UniformInt1128\mathrm{UniformInt}[1,128], the distribution for the σ𝜎\sigma parameter is LogUniform​[0.01,100]LogUniform0.01100\mathrm{LogUniform}[0.01,100]

## Appendix C Share or split learning rate and weight decay between pretraining and finetuning?

Here we demonstrate that tuning and using the same learning rate and weight decay for both pretraining and finetuning results in similar performance to tuning these parameters separately for the two stages. We opt for sharing the learning rate and weight decay for pretraining and finetuning in all the experiments in the paper.

Table 11: Results for single models with MLP mask + target pretraining

GE ↑
CH ↑
CA ↓
HO ↓
OT ↓
HI ↑
FB ↓
AD ↑
WE ↓
CO ↑
MI ↓

shared wd / shared lr
0.683±1​e​-​2plus-or-minus0.6831𝑒-20.683\scriptscriptstyle\pm\scriptstyle 1e\text{-}2
0.857±2​e​-​3plus-or-minus0.8572𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
0.434±7​e​-​3plus-or-minus0.4347𝑒-30.434\scriptscriptstyle\pm\scriptstyle 7e\text{-}3
3.056±4​e​-​2plus-or-minus3.0564𝑒-23.056\scriptscriptstyle\pm\scriptstyle 4e\text{-}2
0.468±2​e​-​3plus-or-minus0.4682𝑒-30.468\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
0.819±2​e​-​3plus-or-minus0.8192𝑒-30.819\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
5.633±4​e​-​2plus-or-minus5.6334𝑒-25.633\scriptscriptstyle\pm\scriptstyle 4e\text{-}2
0.914±1​e​-​3plus-or-minus0.9141𝑒-30.914\scriptscriptstyle\pm\scriptstyle 1e\text{-}3
1.876±5​e​-​3plus-or-minus1.8765𝑒-31.876\scriptscriptstyle\pm\scriptstyle 5e\text{-}3
0.965±7​e​-​4plus-or-minus0.9657𝑒-40.965\scriptscriptstyle\pm\scriptstyle 7e\text{-}4
0.748±4​e​-​4plus-or-minus0.7484𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4e\text{-}4

shared wd / split lr
0.697±9​e​-​3plus-or-minus0.6979𝑒-30.697\scriptscriptstyle\pm\scriptstyle 9e\text{-}3
0.857±3​e​-​3plus-or-minus0.8573𝑒-30.857\scriptscriptstyle\pm\scriptstyle 3e\text{-}3
0.431±7​e​-​3plus-or-minus0.4317𝑒-30.431\scriptscriptstyle\pm\scriptstyle 7e\text{-}3
3.032±3​e​-​2plus-or-minus3.0323𝑒-23.032\scriptscriptstyle\pm\scriptstyle 3e\text{-}2
0.469±2​e​-​3plus-or-minus0.4692𝑒-30.469\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
0.819±2​e​-​3plus-or-minus0.8192𝑒-30.819\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
5.647±4​e​-​2plus-or-minus5.6474𝑒-25.647\scriptscriptstyle\pm\scriptstyle 4e\text{-}2
0.915±9​e​-​4plus-or-minus0.9159𝑒-40.915\scriptscriptstyle\pm\scriptstyle 9e\text{-}4
1.934±8​e​-​3plus-or-minus1.9348𝑒-31.934\scriptscriptstyle\pm\scriptstyle 8e\text{-}3
0.964±9​e​-​4plus-or-minus0.9649𝑒-40.964\scriptscriptstyle\pm\scriptstyle 9e\text{-}4
0.748±4​e​-​4plus-or-minus0.7484𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4e\text{-}4

split wd / split lr
0.688±9​e​-​3plus-or-minus0.6889𝑒-30.688\scriptscriptstyle\pm\scriptstyle 9e\text{-}3
0.856±3​e​-​3plus-or-minus0.8563𝑒-30.856\scriptscriptstyle\pm\scriptstyle 3e\text{-}3
0.430±4​e​-​3plus-or-minus0.4304𝑒-30.430\scriptscriptstyle\pm\scriptstyle 4e\text{-}3
3.046±4​e​-​2plus-or-minus3.0464𝑒-23.046\scriptscriptstyle\pm\scriptstyle 4e\text{-}2
0.471±3​e​-​3plus-or-minus0.4713𝑒-30.471\scriptscriptstyle\pm\scriptstyle 3e\text{-}3
0.821±7​e​-​4plus-or-minus0.8217𝑒-40.821\scriptscriptstyle\pm\scriptstyle 7e\text{-}4
5.734±5​e​-​2plus-or-minus5.7345𝑒-25.734\scriptscriptstyle\pm\scriptstyle 5e\text{-}2
0.914±7​e​-​4plus-or-minus0.9147𝑒-40.914\scriptscriptstyle\pm\scriptstyle 7e\text{-}4
1.891±6​e​-​3plus-or-minus1.8916𝑒-31.891\scriptscriptstyle\pm\scriptstyle 6e\text{-}3
0.964±1​e​-​3plus-or-minus0.9641𝑒-30.964\scriptscriptstyle\pm\scriptstyle 1e\text{-}3
0.748±3​e​-​4plus-or-minus0.7483𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3e\text{-}4

split wd / split lr
0.694±1​e​-​2plus-or-minus0.6941𝑒-20.694\scriptscriptstyle\pm\scriptstyle 1e\text{-}2
0.858±2​e​-​3plus-or-minus0.8582𝑒-30.858\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
0.431±6​e​-​3plus-or-minus0.4316𝑒-30.431\scriptscriptstyle\pm\scriptstyle 6e\text{-}3
3.066±3​e​-​2plus-or-minus3.0663𝑒-23.066\scriptscriptstyle\pm\scriptstyle 3e\text{-}2
0.468±2​e​-​3plus-or-minus0.4682𝑒-30.468\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
0.821±2​e​-​3plus-or-minus0.8212𝑒-30.821\scriptscriptstyle\pm\scriptstyle 2e\text{-}3
5.632±4​e​-​2plus-or-minus5.6324𝑒-25.632\scriptscriptstyle\pm\scriptstyle 4e\text{-}2
0.914±1​e​-​3plus-or-minus0.9141𝑒-30.914\scriptscriptstyle\pm\scriptstyle 1e\text{-}3
1.878±4​e​-​3plus-or-minus1.8784𝑒-31.878\scriptscriptstyle\pm\scriptstyle 4e\text{-}3
0.966±1​e​-​3plus-or-minus0.9661𝑒-30.966\scriptscriptstyle\pm\scriptstyle 1e\text{-}3
0.748±3​e​-​4plus-or-minus0.7483𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3e\text{-}4

## Appendix D Early-stopping criterions

Here we demonstrate that early stopping the pretraining by the value of the pretraining objective on the hold-out validation set is comparable to the early stopping by the downstream metric on the hold-out validation set after finetuning. See [Table 12](#A4.T12 "Table 12 ‣ Appendix D Early-stopping criterions ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") for the results. This is an important practical observation, as computing pretraining objective is much faster than the full finetuning of the model, especially on large scale datasets.

Table 12: Results for single models with MLP mask + target pretraining

|  | GE ↑ | CH ↑ | CA ↓ | HO ↓ | OT ↓ | HI ↑ | FB ↓ | AD ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finetune early stop | 0.6830.6830.683 | 0.8570.8570.857 | 0.4340.4340.434 | 3.0563.0563.056 | 0.4680.4680.468 | 0.8190.8190.819 | 5.6335.6335.633 | 0.9140.9140.914 |
| pretrain early stop | 0.6740.6740.674 | 0.8550.8550.855 | 0.4340.4340.434 | 3.0313.0313.031 | 0.4690.4690.469 | 0.8180.8180.818 | 5.7385.7385.738 | 0.9140.9140.914 |

## Appendix E Extended Tables With Experimental Results

The scores with standard deviations for single models and ensembles are provided in [13](#A5.T13 "Table 13 ‣ Appendix E Extended Tables With Experimental Results ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") and [14](#A5.T14 "Table 14 ‣ Appendix E Extended Tables With Experimental Results ‣ Revisiting Pretraining Objectives for Tabular Deep Learning") respectively.

Table 13: Extended results for single models

GE ↑
CH ↑
CA ↓
HO ↓
OT ↓
HI ↑
FB ↓
AD ↑
WE ↓
CO ↑
MI ↓

CatBoost
0.683±4.7​e​-​3plus-or-minus0.6834.7𝑒-30.683\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.864±8.1​e​-​4plus-or-minus0.8648.1𝑒-40.864\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}4
0.433±1.7​e​-​3plus-or-minus0.4331.7𝑒-30.433\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
3.115±1.8​e​-​2plus-or-minus3.1151.8𝑒-23.115\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.457±1.3​e​-​3plus-or-minus0.4571.3𝑒-30.457\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.806±3.4​e​-​4plus-or-minus0.8063.4𝑒-40.806\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4
5.324±4.0​e​-​2plus-or-minus5.3244.0𝑒-25.324\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}2
0.927±3.1​e​-​4plus-or-minus0.9273.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4
1.837±2.1​e​-​3plus-or-minus1.8372.1𝑒-31.837\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.966±3.2​e​-​4plus-or-minus0.9663.2𝑒-40.966\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}4
0.743±3.0​e​-​4plus-or-minus0.7433.0𝑒-40.743\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4

XGBoost
0.678±4.8​e​-​3plus-or-minus0.6784.8𝑒-30.678\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3
0.858±2.3​e​-​3plus-or-minus0.8582.3𝑒-30.858\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.436±2.5​e​-​3plus-or-minus0.4362.5𝑒-30.436\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
3.160±6.9​e​-​3plus-or-minus3.1606.9𝑒-33.160\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3
0.457±6.0​e​-​3plus-or-minus0.4576.0𝑒-30.457\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}3
0.804±1.5​e​-​3plus-or-minus0.8041.5𝑒-30.804\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
5.383±2.8​e​-​2plus-or-minus5.3832.8𝑒-25.383\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}2
0.927±7.0​e​-​4plus-or-minus0.9277.0𝑒-40.927\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4
1.802±2.0​e​-​3plus-or-minus1.8022.0𝑒-31.802\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.969±6.1​e​-​4plus-or-minus0.9696.1𝑒-40.969\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4
0.742±1.5​e​-​4plus-or-minus0.7421.5𝑒-40.742\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4

MLP

no pretraining
0.635±1.3​e​-​2plus-or-minus0.6351.3𝑒-20.635\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}2
0.849±1.6​e​-​3plus-or-minus0.8491.6𝑒-30.849\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.506±8.6​e​-​3plus-or-minus0.5068.6𝑒-30.506\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}3
3.156±2.1​e​-​2plus-or-minus3.1562.1𝑒-23.156\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2
0.479±1.4​e​-​3plus-or-minus0.4791.4𝑒-30.479\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.801±9.5​e​-​4plus-or-minus0.8019.5𝑒-40.801\scriptscriptstyle\pm\scriptstyle 9.5e\text{-}4
5.737±6.1​e​-​2plus-or-minus5.7376.1𝑒-25.737\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}2
0.908±1.0​e​-​3plus-or-minus0.9081.0𝑒-30.908\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
1.909±4.6​e​-​3plus-or-minus1.9094.6𝑒-31.909\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}3
0.963±8.7​e​-​4plus-or-minus0.9638.7𝑒-40.963\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}4
0.749±3.6​e​-​4plus-or-minus0.7493.6𝑒-40.749\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4

mask
0.691±1.0​e​-​2plus-or-minus0.6911.0𝑒-20.691\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.857±2.5​e​-​3plus-or-minus0.8572.5𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.454±5.0​e​-​3plus-or-minus0.4545.0𝑒-30.454\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}3
3.113±4.3​e​-​2plus-or-minus3.1134.3𝑒-23.113\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}2
0.472±3.0​e​-​3plus-or-minus0.4723.0𝑒-30.472\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3
0.814±1.7​e​-​3plus-or-minus0.8141.7𝑒-30.814\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
5.681±3.1​e​-​2plus-or-minus5.6813.1𝑒-25.681\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2
0.912±8.0​e​-​4plus-or-minus0.9128.0𝑒-40.912\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4
1.883±2.9​e​-​3plus-or-minus1.8832.9𝑒-31.883\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3
0.964±9.3​e​-​4plus-or-minus0.9649.3𝑒-40.964\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}4
0.748±3.1​e​-​4plus-or-minus0.7483.1𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4

rec
0.662±1.0​e​-​2plus-or-minus0.6621.0𝑒-20.662\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.853±2.2​e​-​3plus-or-minus0.8532.2𝑒-30.853\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.445±4.0​e​-​3plus-or-minus0.4454.0𝑒-30.445\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}3
3.044±3.3​e​-​2plus-or-minus3.0443.3𝑒-23.044\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}2
0.466±2.1​e​-​3plus-or-minus0.4662.1𝑒-30.466\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.805±1.3​e​-​3plus-or-minus0.8051.3𝑒-30.805\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
5.641±3.2​e​-​2plus-or-minus5.6413.2𝑒-25.641\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}2
0.910±1.2​e​-​3plus-or-minus0.9101.2𝑒-30.910\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
1.875±3.4​e​-​3plus-or-minus1.8753.4𝑒-31.875\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3
0.965±5.4​e​-​4plus-or-minus0.9655.4𝑒-40.965\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}4
0.746±2.3​e​-​4plus-or-minus0.7462.3𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4

contrastive
0.672±1.4​e​-​2plus-or-minus0.6721.4𝑒-20.672\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.855±2.0​e​-​3plus-or-minus0.8552.0𝑒-30.855\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.455±4.5​e​-​3plus-or-minus0.4554.5𝑒-30.455\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}3
3.056±5.2​e​-​2plus-or-minus3.0565.2𝑒-23.056\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}2
0.469±2.6​e​-​3plus-or-minus0.4692.6𝑒-30.469\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3
0.813±1.3​e​-​3plus-or-minus0.8131.3𝑒-30.813\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
5.697±2.9​e​-​2plus-or-minus5.6972.9𝑒-25.697\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2
0.910±1.3​e​-​3plus-or-minus0.9101.3𝑒-30.910\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
1.881±4.5​e​-​3plus-or-minus1.8814.5𝑒-31.881\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}3
0.960±1.2​e​-​3plus-or-minus0.9601.2𝑒-30.960\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.748±3.6​e​-​4plus-or-minus0.7483.6𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4

sup
0.693±1.1​e​-​2plus-or-minus0.6931.1𝑒-20.693\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2
0.856±1.8​e​-​3plus-or-minus0.8561.8𝑒-30.856\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.441±5.3​e​-​3plus-or-minus0.4415.3𝑒-30.441\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}3
3.077±2.9​e​-​2plus-or-minus3.0772.9𝑒-23.077\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2
0.459±2.1​e​-​3plus-or-minus0.4592.1𝑒-30.459\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.814±7.4​e​-​4plus-or-minus0.8147.4𝑒-40.814\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}4
5.689±2.1​e​-​2plus-or-minus5.6892.1𝑒-25.689\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2
0.914±7.8​e​-​4plus-or-minus0.9147.8𝑒-40.914\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4
1.883±4.6​e​-​3plus-or-minus1.8834.6𝑒-31.883\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}3
0.968±5.2​e​-​4plus-or-minus0.9685.2𝑒-40.968\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4
0.748±3.0​e​-​4plus-or-minus0.7483.0𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4

supcon
0.666±1.4​e​-​2plus-or-minus0.6661.4𝑒-20.666\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.850±2.2​e​-​3plus-or-minus0.8502.2𝑒-30.850\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.454±4.2​e​-​3plus-or-minus0.4544.2𝑒-30.454\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3
3.108±2.5​e​-​2plus-or-minus3.1082.5𝑒-23.108\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}2
0.480±1.9​e​-​3plus-or-minus0.4801.9𝑒-30.480\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.806±7.3​e​-​4plus-or-minus0.8067.3𝑒-40.806\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}4
5.680±2.4​e​-​2plus-or-minus5.6802.4𝑒-25.680\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.911±6.0​e​-​4plus-or-minus0.9116.0𝑒-40.911\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4
1.873±2.4​e​-​3plus-or-minus1.8732.4𝑒-31.873\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.966±5.8​e​-​4plus-or-minus0.9665.8𝑒-40.966\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4
0.747±3.2​e​-​4plus-or-minus0.7473.2𝑒-40.747\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}4

mask + sup
0.693±8.2​e​-​3plus-or-minus0.6938.2𝑒-30.693\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}3
0.857±2.3​e​-​3plus-or-minus0.8572.3𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.436±6.9​e​-​3plus-or-minus0.4366.9𝑒-30.436\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3
3.099±2.4​e​-​2plus-or-minus3.0992.4𝑒-23.099\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.458±1.7​e​-​3plus-or-minus0.4581.7𝑒-30.458\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.817±5.6​e​-​4plus-or-minus0.8175.6𝑒-40.817\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4
5.685±3.6​e​-​2plus-or-minus5.6853.6𝑒-25.685\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2
0.915±5.3​e​-​4plus-or-minus0.9155.3𝑒-40.915\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4
1.873±5.1​e​-​3plus-or-minus1.8735.1𝑒-31.873\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.967±4.0​e​-​4plus-or-minus0.9674.0𝑒-40.967\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4
0.748±2.8​e​-​4plus-or-minus0.7482.8𝑒-40.748\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4

rec + sup
0.684±7.7​e​-​3plus-or-minus0.6847.7𝑒-30.684\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}3
0.854±4.5​e​-​3plus-or-minus0.8544.5𝑒-30.854\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}3
0.436±4.4​e​-​3plus-or-minus0.4364.4𝑒-30.436\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}3
3.012±4.0​e​-​2plus-or-minus3.0124.0𝑒-23.012\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}2
0.456±1.9​e​-​3plus-or-minus0.4561.9𝑒-30.456\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.815±5.8​e​-​4plus-or-minus0.8155.8𝑒-40.815\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4
5.672±3.6​e​-​2plus-or-minus5.6723.6𝑒-25.672\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2
0.911±1.4​e​-​3plus-or-minus0.9111.4𝑒-30.911\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
1.862±2.8​e​-​3plus-or-minus1.8622.8𝑒-31.862\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}3
0.967±6.6​e​-​4plus-or-minus0.9676.6𝑒-40.967\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}4
0.747±4.9​e​-​4plus-or-minus0.7474.9𝑒-40.747\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}4

mask + target
0.683±1.0​e​-​2plus-or-minus0.6831.0𝑒-20.683\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.857±2.1​e​-​3plus-or-minus0.8572.1𝑒-30.857\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.434±7.2​e​-​3plus-or-minus0.4347.2𝑒-30.434\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}3
3.056±4.0​e​-​2plus-or-minus3.0564.0𝑒-23.056\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}2
0.468±1.9​e​-​3plus-or-minus0.4681.9𝑒-30.468\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.819±1.6​e​-​3plus-or-minus0.8191.6𝑒-30.819\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
5.633±3.7​e​-​2plus-or-minus5.6333.7𝑒-25.633\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}2
0.914±1.1​e​-​3plus-or-minus0.9141.1𝑒-30.914\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
1.876±4.8​e​-​3plus-or-minus1.8764.8𝑒-31.876\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3
0.965±6.6​e​-​4plus-or-minus0.9656.6𝑒-40.965\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}4
0.748±4.5​e​-​4plus-or-minus0.7484.5𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}4

- target sampling
0.680±9.7​e​-​3plus-or-minus0.6809.7𝑒-30.680\scriptscriptstyle\pm\scriptstyle 9.7e\text{-}3
0.857±3.1​e​-​3plus-or-minus0.8573.1𝑒-30.857\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
0.432±4.9​e​-​3plus-or-minus0.4324.9𝑒-30.432\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3
3.019±3.5​e​-​2plus-or-minus3.0193.5𝑒-23.019\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}2
0.468±1.9​e​-​3plus-or-minus0.4681.9𝑒-30.468\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.815±1.7​e​-​3plus-or-minus0.8151.7𝑒-30.815\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
5.697±3.1​e​-​2plus-or-minus5.6973.1𝑒-25.697\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2
0.912±6.7​e​-​4plus-or-minus0.9126.7𝑒-40.912\scriptscriptstyle\pm\scriptstyle 6.7e\text{-}4
1.887±3.1​e​-​3plus-or-minus1.8873.1𝑒-31.887\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
0.964±1.1​e​-​3plus-or-minus0.9641.1𝑒-30.964\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.748±3.1​e​-​4plus-or-minus0.7483.1𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4

rec + target
0.659±8.6​e​-​3plus-or-minus0.6598.6𝑒-30.659\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}3
0.853±3.2​e​-​3plus-or-minus0.8533.2𝑒-30.853\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3
0.454±6.7​e​-​3plus-or-minus0.4546.7𝑒-30.454\scriptscriptstyle\pm\scriptstyle 6.7e\text{-}3
3.044±4.9​e​-​2plus-or-minus3.0444.9𝑒-23.044\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}2
0.463±1.6​e​-​3plus-or-minus0.4631.6𝑒-30.463\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.806±1.5​e​-​3plus-or-minus0.8061.5𝑒-30.806\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
5.636±3.1​e​-​2plus-or-minus5.6363.1𝑒-25.636\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}2
0.909±9.0​e​-​4plus-or-minus0.9099.0𝑒-40.909\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}4
1.884±2.3​e​-​3plus-or-minus1.8842.3𝑒-31.884\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.965±8.7​e​-​4plus-or-minus0.9658.7𝑒-40.965\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}4
0.745±3.9​e​-​4plus-or-minus0.7453.9𝑒-40.745\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4

- target sampling
0.641±5.6​e​-​3plus-or-minus0.6415.6𝑒-30.641\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}3
0.853±3.4​e​-​3plus-or-minus0.8533.4𝑒-30.853\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3
0.455±4.6​e​-​3plus-or-minus0.4554.6𝑒-30.455\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}3
3.046±2.4​e​-​2plus-or-minus3.0462.4𝑒-23.046\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.463±2.0​e​-​3plus-or-minus0.4632.0𝑒-30.463\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.806±1.3​e​-​3plus-or-minus0.8061.3𝑒-30.806\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
5.640±1.9​e​-​2plus-or-minus5.6401.9𝑒-25.640\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2
0.910±1.1​e​-​3plus-or-minus0.9101.1𝑒-30.910\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
1.877±3.3​e​-​3plus-or-minus1.8773.3𝑒-31.877\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3
0.966±4.4​e​-​4plus-or-minus0.9664.4𝑒-40.966\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4
0.746±3.9​e​-​4plus-or-minus0.7463.9𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4

MLP-PLR

no pretraining
0.668±1.4​e​-​2plus-or-minus0.6681.4𝑒-20.668\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.858±4.7​e​-​3plus-or-minus0.8584.7𝑒-30.858\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.469±5.2​e​-​3plus-or-minus0.4695.2𝑒-30.469\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}3
3.008±2.3​e​-​2plus-or-minus3.0082.3𝑒-23.008\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2
0.483±1.6​e​-​3plus-or-minus0.4831.6𝑒-30.483\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.809±2.3​e​-​3plus-or-minus0.8092.3𝑒-30.809\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
5.608±5.6​e​-​2plus-or-minus5.6085.6𝑒-25.608\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}2
0.926±6.1​e​-​4plus-or-minus0.9266.1𝑒-40.926\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4
1.890±5.0​e​-​3plus-or-minus1.8905.0𝑒-31.890\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}3
0.969±1.0​e​-​3plus-or-minus0.9691.0𝑒-30.969\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.746±3.7​e​-​4plus-or-minus0.7463.7𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4

mask
0.685±5.6​e​-​3plus-or-minus0.6855.6𝑒-30.685\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}3
0.863±1.8​e​-​3plus-or-minus0.8631.8𝑒-30.863\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.434±4.3​e​-​3plus-or-minus0.4344.3𝑒-30.434\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3
3.007±4.6​e​-​2plus-or-minus3.0074.6𝑒-23.007\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2
0.477±2.5​e​-​3plus-or-minus0.4772.5𝑒-30.477\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.818±8.4​e​-​4plus-or-minus0.8188.4𝑒-40.818\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}4
5.586±2.4​e​-​2plus-or-minus5.5862.4𝑒-25.586\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.927±5.1​e​-​4plus-or-minus0.9275.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4
1.911±5.4​e​-​3plus-or-minus1.9115.4𝑒-31.911\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}3
0.970±5.1​e​-​4plus-or-minus0.9705.1𝑒-40.970\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4
0.748±3.9​e​-​4plus-or-minus0.7483.9𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4

rec
0.667±7.2​e​-​3plus-or-minus0.6677.2𝑒-30.667\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}3
0.852±7.2​e​-​3plus-or-minus0.8527.2𝑒-30.852\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}3
0.439±5.2​e​-​3plus-or-minus0.4395.2𝑒-30.439\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}3
3.031±3.8​e​-​2plus-or-minus3.0313.8𝑒-23.031\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}2
0.472±3.0​e​-​3plus-or-minus0.4723.0𝑒-30.472\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3
0.808±1.2​e​-​3plus-or-minus0.8081.2𝑒-30.808\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
5.571±1.2​e​-​1plus-or-minus5.5711.2𝑒-15.571\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}1
0.926±6.4​e​-​4plus-or-minus0.9266.4𝑒-40.926\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}4
1.877±4.0​e​-​3plus-or-minus1.8774.0𝑒-31.877\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}3
0.971±4.8​e​-​4plus-or-minus0.9714.8𝑒-40.971\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4
0.745±3.6​e​-​4plus-or-minus0.7453.6𝑒-40.745\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4

sup
0.710±4.6​e​-​3plus-or-minus0.7104.6𝑒-30.710\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}3
0.859±4.1​e​-​3plus-or-minus0.8594.1𝑒-30.859\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}3
0.433±3.6​e​-​3plus-or-minus0.4333.6𝑒-30.433\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}3
3.136±8.1​e​-​2plus-or-minus3.1368.1𝑒-23.136\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}2
0.479±1.9​e​-​3plus-or-minus0.4791.9𝑒-30.479\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.811±1.0​e​-​3plus-or-minus0.8111.0𝑒-30.811\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
5.521±4.6​e​-​2plus-or-minus5.5214.6𝑒-25.521\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2
0.924±1.5​e​-​3plus-or-minus0.9241.5𝑒-30.924\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
1.873±2.0​e​-​3plus-or-minus1.8732.0𝑒-31.873\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.971±4.5​e​-​4plus-or-minus0.9714.5𝑒-40.971\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}4
0.748±8.0​e​-​4plus-or-minus0.7488.0𝑒-40.748\scriptscriptstyle\pm\scriptstyle 8.0e\text{-}4

mask + sup
0.711±7.1​e​-​3plus-or-minus0.7117.1𝑒-30.711\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}3
0.866±2.0​e​-​3plus-or-minus0.8662.0𝑒-30.866\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.441±4.9​e​-​3plus-or-minus0.4414.9𝑒-30.441\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3
3.129±4.1​e​-​2plus-or-minus3.1294.1𝑒-23.129\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}2
0.480±1.8​e​-​3plus-or-minus0.4801.8𝑒-30.480\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.813±8.2​e​-​4plus-or-minus0.8138.2𝑒-40.813\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4
5.480±4.6​e​-​2plus-or-minus5.4804.6𝑒-25.480\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2
0.925±1.0​e​-​3plus-or-minus0.9251.0𝑒-30.925\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
1.875±2.2​e​-​3plus-or-minus1.8752.2𝑒-31.875\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.969±6.0​e​-​4plus-or-minus0.9696.0𝑒-40.969\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4
0.745±2.4​e​-​4plus-or-minus0.7452.4𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4

rec + sup
0.709±5.1​e​-​3plus-or-minus0.7095.1𝑒-30.709\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.858±1.9​e​-​3plus-or-minus0.8581.9𝑒-30.858\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.433±2.7​e​-​3plus-or-minus0.4332.7𝑒-30.433\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3
3.059±3.6​e​-​2plus-or-minus3.0593.6𝑒-23.059\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2
0.465±2.2​e​-​3plus-or-minus0.4652.2𝑒-30.465\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.807±6.2​e​-​4plus-or-minus0.8076.2𝑒-40.807\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4
5.571±1.2​e​-​1plus-or-minus5.5711.2𝑒-15.571\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}1
0.927±5.8​e​-​4plus-or-minus0.9275.8𝑒-40.927\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4
1.865±3.1​e​-​3plus-or-minus1.8653.1𝑒-31.865\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
0.971±4.4​e​-​4plus-or-minus0.9714.4𝑒-40.971\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4
0.745±2.4​e​-​4plus-or-minus0.7452.4𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4

mask + target
0.694±9.1​e​-​3plus-or-minus0.6949.1𝑒-30.694\scriptscriptstyle\pm\scriptstyle 9.1e\text{-}3
0.862±1.7​e​-​3plus-or-minus0.8621.7𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.425±4.2​e​-​3plus-or-minus0.4254.2𝑒-30.425\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3
3.023±4.3​e​-​2plus-or-minus3.0234.3𝑒-23.023\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}2
0.474±2.0​e​-​3plus-or-minus0.4742.0𝑒-30.474\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.821±1.1​e​-​3plus-or-minus0.8211.1𝑒-30.821\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
5.537±3.4​e​-​2plus-or-minus5.5373.4𝑒-25.537\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}2
0.929±3.3​e​-​4plus-or-minus0.9293.3𝑒-40.929\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4
1.911±6.2​e​-​3plus-or-minus1.9116.2𝑒-31.911\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}3
0.969±5.8​e​-​4plus-or-minus0.9695.8𝑒-40.969\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4
0.749±1.2​e​-​3plus-or-minus0.7491.2𝑒-30.749\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3

- target sampling
0.690±9.6​e​-​3plus-or-minus0.6909.6𝑒-30.690\scriptscriptstyle\pm\scriptstyle 9.6e\text{-}3
0.864±2.8​e​-​3plus-or-minus0.8642.8𝑒-30.864\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}3
0.421±4.5​e​-​3plus-or-minus0.4214.5𝑒-30.421\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}3
2.971±4.1​e​-​2plus-or-minus2.9714.1𝑒-22.971\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}2
0.479±2.0​e​-​3plus-or-minus0.4792.0𝑒-30.479\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.821±1.0​e​-​3plus-or-minus0.8211.0𝑒-30.821\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
5.440±8.1​e​-​2plus-or-minus5.4408.1𝑒-25.440\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}2
0.928±5.6​e​-​4plus-or-minus0.9285.6𝑒-40.928\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4
1.906±5.5​e​-​3plus-or-minus1.9065.5𝑒-31.906\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}3
0.970±3.9​e​-​4plus-or-minus0.9703.9𝑒-40.970\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4
0.748±5.3​e​-​4plus-or-minus0.7485.3𝑒-40.748\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4

rec + target
0.688±8.2​e​-​3plus-or-minus0.6888.2𝑒-30.688\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}3
0.860±1.4​e​-​3plus-or-minus0.8601.4𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.445±2.7​e​-​3plus-or-minus0.4452.7𝑒-30.445\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3
3.064±3.4​e​-​2plus-or-minus3.0643.4𝑒-23.064\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}2
0.475±1.9​e​-​3plus-or-minus0.4751.9𝑒-30.475\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.812±1.1​e​-​3plus-or-minus0.8121.1𝑒-30.812\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
5.507±1.0​e​-​1plus-or-minus5.5071.0𝑒-15.507\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}1
0.927±3.9​e​-​4plus-or-minus0.9273.9𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4
1.887±2.4​e​-​3plus-or-minus1.8872.4𝑒-31.887\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.971±3.9​e​-​4plus-or-minus0.9713.9𝑒-40.971\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4
0.748±7.2​e​-​4plus-or-minus0.7487.2𝑒-40.748\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}4

- target sampling
0.687±7.9​e​-​3plus-or-minus0.6877.9𝑒-30.687\scriptscriptstyle\pm\scriptstyle 7.9e\text{-}3
0.855±4.8​e​-​3plus-or-minus0.8554.8𝑒-30.855\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}3
0.453±6.4​e​-​3plus-or-minus0.4536.4𝑒-30.453\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}3
3.008±2.7​e​-​2plus-or-minus3.0082.7𝑒-23.008\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}2
0.471±2.1​e​-​3plus-or-minus0.4712.1𝑒-30.471\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.812±5.5​e​-​4plus-or-minus0.8125.5𝑒-40.812\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}4
5.592±9.8​e​-​2plus-or-minus5.5929.8𝑒-25.592\scriptscriptstyle\pm\scriptstyle 9.8e\text{-}2
0.927±3.4​e​-​4plus-or-minus0.9273.4𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4
1.876±3.1​e​-​3plus-or-minus1.8763.1𝑒-31.876\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
0.970±4.4​e​-​4plus-or-minus0.9704.4𝑒-40.970\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4
0.745±3.5​e​-​4plus-or-minus0.7453.5𝑒-40.745\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4

MLP-T-LR

no pretraining
0.634±6.9​e​-​3plus-or-minus0.6346.9𝑒-30.634\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3
0.866±1.3​e​-​3plus-or-minus0.8661.3𝑒-30.866\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.444±1.8​e​-​3plus-or-minus0.4441.8𝑒-30.444\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
3.113±4.5​e​-​2plus-or-minus3.1134.5𝑒-23.113\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2
0.482±1.7​e​-​3plus-or-minus0.4821.7𝑒-30.482\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.805±9.3​e​-​4plus-or-minus0.8059.3𝑒-40.805\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}4
5.520±3.6​e​-​2plus-or-minus5.5203.6𝑒-25.520\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2
0.925±6.8​e​-​4plus-or-minus0.9256.8𝑒-40.925\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}4
1.897±4.5​e​-​3plus-or-minus1.8974.5𝑒-31.897\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}3
0.968±5.0​e​-​4plus-or-minus0.9685.0𝑒-40.968\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4
0.749±5.2​e​-​4plus-or-minus0.7495.2𝑒-40.749\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4

mask
0.654±6.4​e​-​3plus-or-minus0.6546.4𝑒-30.654\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}3
0.868±1.0​e​-​3plus-or-minus0.8681.0𝑒-30.868\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.424±2.4​e​-​3plus-or-minus0.4242.4𝑒-30.424\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
3.045±3.7​e​-​2plus-or-minus3.0453.7𝑒-23.045\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}2
0.472±2.5​e​-​3plus-or-minus0.4722.5𝑒-30.472\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.818±1.8​e​-​3plus-or-minus0.8181.8𝑒-30.818\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
5.544±3.5​e​-​2plus-or-minus5.5443.5𝑒-25.544\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}2
0.926±7.1​e​-​4plus-or-minus0.9267.1𝑒-40.926\scriptscriptstyle\pm\scriptstyle 7.1e\text{-}4
1.916±3.1​e​-​3plus-or-minus1.9163.1𝑒-31.916\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
0.969±4.6​e​-​4plus-or-minus0.9694.6𝑒-40.969\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4
0.748±3.5​e​-​4plus-or-minus0.7483.5𝑒-40.748\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4

rec
0.652±7.4​e​-​3plus-or-minus0.6527.4𝑒-30.652\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}3
0.857±4.4​e​-​3plus-or-minus0.8574.4𝑒-30.857\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}3
0.424±3.1​e​-​3plus-or-minus0.4243.1𝑒-30.424\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}3
3.109±3.7​e​-​2plus-or-minus3.1093.7𝑒-23.109\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}2
0.472±2.0​e​-​3plus-or-minus0.4722.0𝑒-30.472\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.808±1.0​e​-​3plus-or-minus0.8081.0𝑒-30.808\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
5.363±6.6​e​-​2plus-or-minus5.3636.6𝑒-25.363\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}2
0.924±1.8​e​-​4plus-or-minus0.9241.8𝑒-40.924\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}4
1.861±3.9​e​-​3plus-or-minus1.8613.9𝑒-31.861\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}3
0.969±7.3​e​-​4plus-or-minus0.9697.3𝑒-40.969\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}4
0.746±4.8​e​-​4plus-or-minus0.7464.8𝑒-40.746\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4

sup
0.682±5.1​e​-​3plus-or-minus0.6825.1𝑒-30.682\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.860±4.1​e​-​3plus-or-minus0.8604.1𝑒-30.860\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}3
0.430±1.9​e​-​3plus-or-minus0.4301.9𝑒-30.430\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
3.135±1.7​e​-​2plus-or-minus3.1351.7𝑒-23.135\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}2
0.471±1.2​e​-​3plus-or-minus0.4711.2𝑒-30.471\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.807±6.2​e​-​4plus-or-minus0.8076.2𝑒-40.807\scriptscriptstyle\pm\scriptstyle 6.2e\text{-}4
5.525±2.3​e​-​2plus-or-minus5.5252.3𝑒-25.525\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2
0.927±3.8​e​-​4plus-or-minus0.9273.8𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}4
1.893±2.4​e​-​3plus-or-minus1.8932.4𝑒-31.893\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.971±5.8​e​-​4plus-or-minus0.9715.8𝑒-40.971\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}4
0.747±3.1​e​-​4plus-or-minus0.7473.1𝑒-40.747\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4

mask + sup
0.676±7.7​e​-​3plus-or-minus0.6767.7𝑒-30.676\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}3
0.858±7.9​e​-​3plus-or-minus0.8587.9𝑒-30.858\scriptscriptstyle\pm\scriptstyle 7.9e\text{-}3
0.429±1.8​e​-​3plus-or-minus0.4291.8𝑒-30.429\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
3.199±2.4​e​-​2plus-or-minus3.1992.4𝑒-23.199\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.468±9.9​e​-​4plus-or-minus0.4689.9𝑒-40.468\scriptscriptstyle\pm\scriptstyle 9.9e\text{-}4
0.814±8.2​e​-​4plus-or-minus0.8148.2𝑒-40.814\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4
5.510±3.6​e​-​2plus-or-minus5.5103.6𝑒-25.510\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}2
0.926±8.7​e​-​4plus-or-minus0.9268.7𝑒-40.926\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}4
1.869±2.9​e​-​3plus-or-minus1.8692.9𝑒-31.869\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3
0.971±3.9​e​-​4plus-or-minus0.9713.9𝑒-40.971\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4
0.748±2.5​e​-​4plus-or-minus0.7482.5𝑒-40.748\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4

rec + sup
0.678±6.7​e​-​3plus-or-minus0.6786.7𝑒-30.678\scriptscriptstyle\pm\scriptstyle 6.7e\text{-}3
0.865±1.1​e​-​3plus-or-minus0.8651.1𝑒-30.865\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.437±2.0​e​-​3plus-or-minus0.4372.0𝑒-30.437\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
3.112±4.1​e​-​2plus-or-minus3.1124.1𝑒-23.112\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}2
0.462±2.2​e​-​3plus-or-minus0.4622.2𝑒-30.462\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.807±2.6​e​-​3plus-or-minus0.8072.6𝑒-30.807\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3
5.516±2.2​e​-​2plus-or-minus5.5162.2𝑒-25.516\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}2
0.927±3.7​e​-​4plus-or-minus0.9273.7𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4
1.862±3.4​e​-​3plus-or-minus1.8623.4𝑒-31.862\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3
0.970±5.0​e​-​4plus-or-minus0.9705.0𝑒-40.970\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4
0.748±4.2​e​-​4plus-or-minus0.7484.2𝑒-40.748\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}4

mask + target
0.649±7.3​e​-​3plus-or-minus0.6497.3𝑒-30.649\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}3
0.865±1.7​e​-​3plus-or-minus0.8651.7𝑒-30.865\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.421±3.9​e​-​3plus-or-minus0.4213.9𝑒-30.421\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}3
3.058±4.3​e​-​2plus-or-minus3.0584.3𝑒-23.058\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}2
0.474±1.9​e​-​3plus-or-minus0.4741.9𝑒-30.474\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.820±1.2​e​-​3plus-or-minus0.8201.2𝑒-30.820\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
5.644±4.6​e​-​2plus-or-minus5.6444.6𝑒-25.644\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}2
0.929±3.5​e​-​4plus-or-minus0.9293.5𝑒-40.929\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4
1.924±7.6​e​-​3plus-or-minus1.9247.6𝑒-31.924\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}3
0.969±4.8​e​-​4plus-or-minus0.9694.8𝑒-40.969\scriptscriptstyle\pm\scriptstyle 4.8e\text{-}4
0.749±5.1​e​-​4plus-or-minus0.7495.1𝑒-40.749\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4

- target sampling
0.649±8.3​e​-​3plus-or-minus0.6498.3𝑒-30.649\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}3
0.861±3.2​e​-​3plus-or-minus0.8613.2𝑒-30.861\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}3
0.417±4.9​e​-​3plus-or-minus0.4174.9𝑒-30.417\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3
3.050±3.3​e​-​2plus-or-minus3.0503.3𝑒-23.050\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}2
0.476±1.7​e​-​3plus-or-minus0.4761.7𝑒-30.476\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.819±1.6​e​-​3plus-or-minus0.8191.6𝑒-30.819\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
5.492±3.2​e​-​2plus-or-minus5.4923.2𝑒-25.492\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}2
0.928±5.6​e​-​4plus-or-minus0.9285.6𝑒-40.928\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4
1.874±3.7​e​-​3plus-or-minus1.8743.7𝑒-31.874\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.969±4.0​e​-​4plus-or-minus0.9694.0𝑒-40.969\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}4
0.749±1.4​e​-​3plus-or-minus0.7491.4𝑒-30.749\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3

rec + target
0.668±7.0​e​-​3plus-or-minus0.6687.0𝑒-30.668\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}3
0.864±1.7​e​-​3plus-or-minus0.8641.7𝑒-30.864\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.440±3.5​e​-​3plus-or-minus0.4403.5𝑒-30.440\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3
3.113±3.5​e​-​2plus-or-minus3.1133.5𝑒-23.113\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}2
0.473±4.3​e​-​3plus-or-minus0.4734.3𝑒-30.473\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}3
0.806±1.0​e​-​3plus-or-minus0.8061.0𝑒-30.806\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
5.493±4.4​e​-​2plus-or-minus5.4934.4𝑒-25.493\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}2
0.927±5.6​e​-​4plus-or-minus0.9275.6𝑒-40.927\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}4
1.862±3.5​e​-​3plus-or-minus1.8623.5𝑒-31.862\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3
0.969±6.5​e​-​4plus-or-minus0.9696.5𝑒-40.969\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}4
0.746±3.4​e​-​4plus-or-minus0.7463.4𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4

- target sampling
0.667±1.0​e​-​2plus-or-minus0.6671.0𝑒-20.667\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.859±2.7​e​-​3plus-or-minus0.8592.7𝑒-30.859\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3
0.432±3.3​e​-​3plus-or-minus0.4323.3𝑒-30.432\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3
3.104±2.9​e​-​2plus-or-minus3.1042.9𝑒-23.104\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}2
0.470±2.2​e​-​3plus-or-minus0.4702.2𝑒-30.470\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.806±1.5​e​-​3plus-or-minus0.8061.5𝑒-30.806\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
5.391±3.9​e​-​2plus-or-minus5.3913.9𝑒-25.391\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}2
0.927±4.1​e​-​4plus-or-minus0.9274.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 4.1e\text{-}4
1.867±3.8​e​-​3plus-or-minus1.8673.8𝑒-31.867\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}3
0.968±7.5​e​-​4plus-or-minus0.9687.5𝑒-40.968\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}4
0.746±3.1​e​-​4plus-or-minus0.7463.1𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4

Table 14: Extended results for ensemble models

GE ↑
CH ↑
CA ↓
HO ↓
OT ↓
HI ↑
FB ↓
AD ↑
WE ↓
CO ↑
MI ↓

CatBoost
0.692±1.8​e​-​3plus-or-minus0.6921.8𝑒-30.692\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.864±6.8​e​-​5plus-or-minus0.8646.8𝑒-50.864\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}5
0.430±1.1​e​-​3plus-or-minus0.4301.1𝑒-30.430\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
3.093±5.1​e​-​3plus-or-minus3.0935.1𝑒-33.093\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.450±3.5​e​-​4plus-or-minus0.4503.5𝑒-40.450\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4
0.807±7.5​e​-​5plus-or-minus0.8077.5𝑒-50.807\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}5
5.226±1.2​e​-​2plus-or-minus5.2261.2𝑒-25.226\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2
0.928±1.3​e​-​4plus-or-minus0.9281.3𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4
1.801±1.2​e​-​3plus-or-minus1.8011.2𝑒-31.801\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.967±1.3​e​-​4plus-or-minus0.9671.3𝑒-40.967\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4
0.741±1.4​e​-​4plus-or-minus0.7411.4𝑒-40.741\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4

XGBoost
0.683±1.3​e​-​3plus-or-minus0.6831.3𝑒-30.683\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.860±4.3​e​-​4plus-or-minus0.8604.3𝑒-40.860\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4
0.434±7.0​e​-​4plus-or-minus0.4347.0𝑒-40.434\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4
3.152±1.2​e​-​3plus-or-minus3.1521.2𝑒-33.152\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.454±2.5​e​-​3plus-or-minus0.4542.5𝑒-30.454\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.805±8.3​e​-​4plus-or-minus0.8058.3𝑒-40.805\scriptscriptstyle\pm\scriptstyle 8.3e\text{-}4
5.338±1.8​e​-​2plus-or-minus5.3381.8𝑒-25.338\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.927±2.1​e​-​4plus-or-minus0.9272.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4
1.782±4.9​e​-​4plus-or-minus1.7824.9𝑒-41.782\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}4
0.969±8.8​e​-​5plus-or-minus0.9698.8𝑒-50.969\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}5
0.742±5.3​e​-​5plus-or-minus0.7425.3𝑒-50.742\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}5

MLP

no pretraining
0.656±5.9​e​-​3plus-or-minus0.6565.9𝑒-30.656\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}3
0.852±5.2​e​-​4plus-or-minus0.8525.2𝑒-40.852\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4
0.482±2.9​e​-​3plus-or-minus0.4822.9𝑒-30.482\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}3
3.055±8.4​e​-​3plus-or-minus3.0558.4𝑒-33.055\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}3
0.467±2.0​e​-​3plus-or-minus0.4672.0𝑒-30.467\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.805±2.9​e​-​4plus-or-minus0.8052.9𝑒-40.805\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4
5.666±2.6​e​-​3plus-or-minus5.6662.6𝑒-35.666\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}3
0.910±2.7​e​-​4plus-or-minus0.9102.7𝑒-40.910\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4
1.850±1.0​e​-​3plus-or-minus1.8501.0𝑒-31.850\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.968±2.5​e​-​4plus-or-minus0.9682.5𝑒-40.968\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
0.747±8.6​e​-​5plus-or-minus0.7478.6𝑒-50.747\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}5

mask
0.722±1.6​e​-​3plus-or-minus0.7221.6𝑒-30.722\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.859±6.4​e​-​4plus-or-minus0.8596.4𝑒-40.859\scriptscriptstyle\pm\scriptstyle 6.4e\text{-}4
0.437±8.1​e​-​4plus-or-minus0.4378.1𝑒-40.437\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}4
3.026±6.3​e​-​3plus-or-minus3.0266.3𝑒-33.026\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}3
0.451±1.6​e​-​3plus-or-minus0.4511.6𝑒-30.451\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.824±6.5​e​-​4plus-or-minus0.8246.5𝑒-40.824\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}4
5.578±6.8​e​-​3plus-or-minus5.5786.8𝑒-35.578\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}3
0.913±3.0​e​-​4plus-or-minus0.9133.0𝑒-40.913\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}4
1.828±1.0​e​-​3plus-or-minus1.8281.0𝑒-31.828\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.967±1.1​e​-​4plus-or-minus0.9671.1𝑒-40.967\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4
0.746±7.7​e​-​5plus-or-minus0.7467.7𝑒-50.746\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}5

rec
0.679±2.0​e​-​3plus-or-minus0.6792.0𝑒-30.679\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.856±2.8​e​-​4plus-or-minus0.8562.8𝑒-40.856\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4
0.424±1.0​e​-​4plus-or-minus0.4241.0𝑒-40.424\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4
2.967±6.9​e​-​3plus-or-minus2.9676.9𝑒-32.967\scriptscriptstyle\pm\scriptstyle 6.9e\text{-}3
0.453±1.2​e​-​3plus-or-minus0.4531.2𝑒-30.453\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.812±2.5​e​-​4plus-or-minus0.8122.5𝑒-40.812\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
5.571±1.4​e​-​2plus-or-minus5.5711.4𝑒-25.571\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.912±7.5​e​-​5plus-or-minus0.9127.5𝑒-50.912\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}5
1.811±1.4​e​-​3plus-or-minus1.8111.4𝑒-31.811\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.972±2.0​e​-​4plus-or-minus0.9722.0𝑒-40.972\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4
0.744±7.8​e​-​5plus-or-minus0.7447.8𝑒-50.744\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}5

contrastive
0.708±4.4​e​-​3plus-or-minus0.7084.4𝑒-30.708\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}3
0.857±8.8​e​-​4plus-or-minus0.8578.8𝑒-40.857\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}4
0.434±4.0​e​-​3plus-or-minus0.4344.0𝑒-30.434\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}3
2.952±4.4​e​-​3plus-or-minus2.9524.4𝑒-32.952\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}3
0.451±6.1​e​-​4plus-or-minus0.4516.1𝑒-40.451\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4
0.820±4.5​e​-​5plus-or-minus0.8204.5𝑒-50.820\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}5
5.634±1.4​e​-​2plus-or-minus5.6341.4𝑒-25.634\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.912±1.3​e​-​4plus-or-minus0.9121.3𝑒-40.912\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4
1.804±2.8​e​-​3plus-or-minus1.8042.8𝑒-31.804\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}3
0.964±2.1​e​-​4plus-or-minus0.9642.1𝑒-40.964\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4
0.746±1.7​e​-​4plus-or-minus0.7461.7𝑒-40.746\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4

sup
0.717±2.2​e​-​3plus-or-minus0.7172.2𝑒-30.717\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.857±7.4​e​-​4plus-or-minus0.8577.4𝑒-40.857\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}4
0.424±9.4​e​-​4plus-or-minus0.4249.4𝑒-40.424\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4
3.022±1.9​e​-​2plus-or-minus3.0221.9𝑒-23.022\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2
0.443±1.9​e​-​3plus-or-minus0.4431.9𝑒-30.443\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.816±1.4​e​-​4plus-or-minus0.8161.4𝑒-40.816\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
5.602±6.5​e​-​3plus-or-minus5.6026.5𝑒-35.602\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}3
0.916±7.0​e​-​5plus-or-minus0.9167.0𝑒-50.916\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}5
1.828±4.7​e​-​3plus-or-minus1.8284.7𝑒-31.828\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.973±3.6​e​-​4plus-or-minus0.9733.6𝑒-40.973\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4
0.746±2.1​e​-​4plus-or-minus0.7462.1𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4

supcon
0.686±5.2​e​-​3plus-or-minus0.6865.2𝑒-30.686\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}3
0.851±8.4​e​-​4plus-or-minus0.8518.4𝑒-40.851\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}4
0.434±3.0​e​-​3plus-or-minus0.4343.0𝑒-30.434\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3
3.014±6.0​e​-​3plus-or-minus3.0146.0𝑒-33.014\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}3
0.465±1.3​e​-​3plus-or-minus0.4651.3𝑒-30.465\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.809±1.3​e​-​4plus-or-minus0.8091.3𝑒-40.809\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4
5.579±3.4​e​-​3plus-or-minus5.5793.4𝑒-35.579\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3
0.912±3.2​e​-​4plus-or-minus0.9123.2𝑒-40.912\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}4
1.827±9.4​e​-​4plus-or-minus1.8279.4𝑒-41.827\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4
0.970±2.1​e​-​4plus-or-minus0.9702.1𝑒-40.970\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4
0.745±5.3​e​-​5plus-or-minus0.7455.3𝑒-50.745\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}5

mask + sup
0.716±5.7​e​-​3plus-or-minus0.7165.7𝑒-30.716\scriptscriptstyle\pm\scriptstyle 5.7e\text{-}3
0.859±1.1​e​-​3plus-or-minus0.8591.1𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.418±2.2​e​-​3plus-or-minus0.4182.2𝑒-30.418\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
3.066±1.2​e​-​2plus-or-minus3.0661.2𝑒-23.066\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2
0.443±1.5​e​-​3plus-or-minus0.4431.5𝑒-30.443\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
0.819±1.3​e​-​4plus-or-minus0.8191.3𝑒-40.819\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4
5.601±4.7​e​-​3plus-or-minus5.6014.7𝑒-35.601\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.916±2.0​e​-​4plus-or-minus0.9162.0𝑒-40.916\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4
1.810±2.9​e​-​4plus-or-minus1.8102.9𝑒-41.810\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4
0.973±3.9​e​-​5plus-or-minus0.9733.9𝑒-50.973\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}5
0.747±1.3​e​-​4plus-or-minus0.7471.3𝑒-40.747\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}4

rec + sup
0.709±3.7​e​-​3plus-or-minus0.7093.7𝑒-30.709\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.859±1.8​e​-​3plus-or-minus0.8591.8𝑒-30.859\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.419±2.1​e​-​3plus-or-minus0.4192.1𝑒-30.419\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
2.951±1.9​e​-​2plus-or-minus2.9511.9𝑒-22.951\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2
0.442±8.6​e​-​4plus-or-minus0.4428.6𝑒-40.442\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}4
0.817±1.0​e​-​4plus-or-minus0.8171.0𝑒-40.817\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4
5.531±3.0​e​-​3plus-or-minus5.5313.0𝑒-35.531\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3
0.913±5.2​e​-​4plus-or-minus0.9135.2𝑒-40.913\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}4
1.801±4.0​e​-​3plus-or-minus1.8014.0𝑒-31.801\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}3
0.973±2.6​e​-​5plus-or-minus0.9732.6𝑒-50.973\scriptscriptstyle\pm\scriptstyle 2.6e\text{-}5
0.745±1.1​e​-​4plus-or-minus0.7451.1𝑒-40.745\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4

mask + target
0.709±7.3​e​-​3plus-or-minus0.7097.3𝑒-30.709\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}3
0.860±1.6​e​-​3plus-or-minus0.8601.6𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}3
0.414±1.1​e​-​3plus-or-minus0.4141.1𝑒-30.414\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
2.949±1.9​e​-​2plus-or-minus2.9491.9𝑒-22.949\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2
0.457±5.9​e​-​4plus-or-minus0.4575.9𝑒-40.457\scriptscriptstyle\pm\scriptstyle 5.9e\text{-}4
0.828±6.3​e​-​4plus-or-minus0.8286.3𝑒-40.828\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}4
5.551±7.2​e​-​3plus-or-minus5.5517.2𝑒-35.551\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}3
0.916±4.6​e​-​4plus-or-minus0.9164.6𝑒-40.916\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4
1.809±5.3​e​-​4plus-or-minus1.8095.3𝑒-41.809\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4
0.969±5.2​e​-​5plus-or-minus0.9695.2𝑒-50.969\scriptscriptstyle\pm\scriptstyle 5.2e\text{-}5
0.746±1.9​e​-​4plus-or-minus0.7461.9𝑒-40.746\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}4

- target sampling
0.706±4.9​e​-​3plus-or-minus0.7064.9𝑒-30.706\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3
0.860±1.1​e​-​3plus-or-minus0.8601.1𝑒-30.860\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.410±1.5​e​-​3plus-or-minus0.4101.5𝑒-30.410\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
2.955±2.1​e​-​2plus-or-minus2.9552.1𝑒-22.955\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}2
0.456±3.8​e​-​4plus-or-minus0.4563.8𝑒-40.456\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}4
0.822±1.1​e​-​3plus-or-minus0.8221.1𝑒-30.822\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
5.601±2.0​e​-​2plus-or-minus5.6012.0𝑒-25.601\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2
0.914±2.5​e​-​4plus-or-minus0.9142.5𝑒-40.914\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
1.837±1.0​e​-​3plus-or-minus1.8371.0𝑒-31.837\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.968±2.8​e​-​4plus-or-minus0.9682.8𝑒-40.968\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4
0.746±2.1​e​-​4plus-or-minus0.7462.1𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4

rec + target
0.677±4.6​e​-​3plus-or-minus0.6774.6𝑒-30.677\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}3
0.857±3.9​e​-​4plus-or-minus0.8573.9𝑒-40.857\scriptscriptstyle\pm\scriptstyle 3.9e\text{-}4
0.433±1.1​e​-​3plus-or-minus0.4331.1𝑒-30.433\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
2.926±2.3​e​-​2plus-or-minus2.9262.3𝑒-22.926\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}2
0.448±9.4​e​-​4plus-or-minus0.4489.4𝑒-40.448\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4
0.816±5.3​e​-​4plus-or-minus0.8165.3𝑒-40.816\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4
5.555±5.8​e​-​3plus-or-minus5.5555.8𝑒-35.555\scriptscriptstyle\pm\scriptstyle 5.8e\text{-}3
0.910±2.2​e​-​4plus-or-minus0.9102.2𝑒-40.910\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4
1.825±2.5​e​-​3plus-or-minus1.8252.5𝑒-31.825\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.972±6.1​e​-​5plus-or-minus0.9726.1𝑒-50.972\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}5
0.743±1.2​e​-​4plus-or-minus0.7431.2𝑒-40.743\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4

- target sampling
0.669±3.7​e​-​3plus-or-minus0.6693.7𝑒-30.669\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.858±1.4​e​-​3plus-or-minus0.8581.4𝑒-30.858\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.435±8.2​e​-​4plus-or-minus0.4358.2𝑒-40.435\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4
3.003±1.1​e​-​2plus-or-minus3.0031.1𝑒-23.003\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2
0.451±1.2​e​-​3plus-or-minus0.4511.2𝑒-30.451\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
0.815±7.0​e​-​4plus-or-minus0.8157.0𝑒-40.815\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4
5.577±1.1​e​-​2plus-or-minus5.5771.1𝑒-25.577\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2
0.913±2.4​e​-​4plus-or-minus0.9132.4𝑒-40.913\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}4
1.822±6.6​e​-​4plus-or-minus1.8226.6𝑒-41.822\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}4
0.972±2.7​e​-​4plus-or-minus0.9722.7𝑒-40.972\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4
0.744±7.6​e​-​5plus-or-minus0.7447.6𝑒-50.744\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}5

MLP-PLR

no pretraining
0.695±3.7​e​-​3plus-or-minus0.6953.7𝑒-30.695\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.864±7.6​e​-​4plus-or-minus0.8647.6𝑒-40.864\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}4
0.454±1.2​e​-​3plus-or-minus0.4541.2𝑒-30.454\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
2.953±7.4​e​-​3plus-or-minus2.9537.4𝑒-32.953\scriptscriptstyle\pm\scriptstyle 7.4e\text{-}3
0.470±7.5​e​-​4plus-or-minus0.4707.5𝑒-40.470\scriptscriptstyle\pm\scriptstyle 7.5e\text{-}4
0.814±7.9​e​-​4plus-or-minus0.8147.9𝑒-40.814\scriptscriptstyle\pm\scriptstyle 7.9e\text{-}4
5.324±3.2​e​-​2plus-or-minus5.3243.2𝑒-25.324\scriptscriptstyle\pm\scriptstyle 3.2e\text{-}2
0.928±7.7​e​-​5plus-or-minus0.9287.7𝑒-50.928\scriptscriptstyle\pm\scriptstyle 7.7e\text{-}5
1.835±1.5​e​-​3plus-or-minus1.8351.5𝑒-31.835\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
0.974±2.2​e​-​4plus-or-minus0.9742.2𝑒-40.974\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4
0.744±1.2​e​-​4plus-or-minus0.7441.2𝑒-40.744\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4

mask
0.725±4.9​e​-​3plus-or-minus0.7254.9𝑒-30.725\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}3
0.865±7.0​e​-​4plus-or-minus0.8657.0𝑒-40.865\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4
0.421±1.7​e​-​3plus-or-minus0.4211.7𝑒-30.421\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
2.921±1.0​e​-​2plus-or-minus2.9211.0𝑒-22.921\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.457±8.4​e​-​4plus-or-minus0.4578.4𝑒-40.457\scriptscriptstyle\pm\scriptstyle 8.4e\text{-}4
0.827±1.1​e​-​4plus-or-minus0.8271.1𝑒-40.827\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4
5.444±5.4​e​-​3plus-or-minus5.4445.4𝑒-35.444\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}3
0.928±1.0​e​-​4plus-or-minus0.9281.0𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4
1.850±3.4​e​-​3plus-or-minus1.8503.4𝑒-31.850\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}3
0.974±2.3​e​-​4plus-or-minus0.9742.3𝑒-40.974\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4
0.745±6.5​e​-​5plus-or-minus0.7456.5𝑒-50.745\scriptscriptstyle\pm\scriptstyle 6.5e\text{-}5

rec
0.698±1.5​e​-​3plus-or-minus0.6981.5𝑒-30.698\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
0.857±1.5​e​-​3plus-or-minus0.8571.5𝑒-30.857\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
0.418±1.2​e​-​3plus-or-minus0.4181.2𝑒-30.418\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}3
2.954±8.1​e​-​3plus-or-minus2.9548.1𝑒-32.954\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}3
0.454±1.9​e​-​3plus-or-minus0.4541.9𝑒-30.454\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.813±7.8​e​-​4plus-or-minus0.8137.8𝑒-40.813\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4
5.124±2.4​e​-​2plus-or-minus5.1242.4𝑒-25.124\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.928±2.2​e​-​4plus-or-minus0.9282.2𝑒-40.928\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4
1.818±2.5​e​-​3plus-or-minus1.8182.5𝑒-31.818\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.975±2.9​e​-​4plus-or-minus0.9752.9𝑒-40.975\scriptscriptstyle\pm\scriptstyle 2.9e\text{-}4
0.743±2.1​e​-​4plus-or-minus0.7432.1𝑒-40.743\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4

sup
0.733±2.2​e​-​3plus-or-minus0.7332.2𝑒-30.733\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.867±9.6​e​-​4plus-or-minus0.8679.6𝑒-40.867\scriptscriptstyle\pm\scriptstyle 9.6e\text{-}4
0.421±1.0​e​-​3plus-or-minus0.4211.0𝑒-30.421\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
3.054±4.5​e​-​2plus-or-minus3.0544.5𝑒-23.054\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}2
0.465±1.3​e​-​3plus-or-minus0.4651.3𝑒-30.465\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.816±1.4​e​-​4plus-or-minus0.8161.4𝑒-40.816\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
5.407±3.8​e​-​2plus-or-minus5.4073.8𝑒-25.407\scriptscriptstyle\pm\scriptstyle 3.8e\text{-}2
0.926±4.3​e​-​4plus-or-minus0.9264.3𝑒-40.926\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4
1.834±3.3​e​-​4plus-or-minus1.8343.3𝑒-41.834\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4
0.975±1.6​e​-​4plus-or-minus0.9751.6𝑒-40.975\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4
0.746±2.1​e​-​4plus-or-minus0.7462.1𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4

mask + sup
0.732±2.0​e​-​3plus-or-minus0.7322.0𝑒-30.732\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.869±3.5​e​-​4plus-or-minus0.8693.5𝑒-40.869\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4
0.424±8.9​e​-​4plus-or-minus0.4248.9𝑒-40.424\scriptscriptstyle\pm\scriptstyle 8.9e\text{-}4
3.055±1.6​e​-​2plus-or-minus3.0551.6𝑒-23.055\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2
0.468±7.8​e​-​4plus-or-minus0.4687.8𝑒-40.468\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4
0.817±3.4​e​-​4plus-or-minus0.8173.4𝑒-40.817\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4
5.366±1.1​e​-​2plus-or-minus5.3661.1𝑒-25.366\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2
0.927±2.1​e​-​4plus-or-minus0.9272.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4
1.848±1.7​e​-​3plus-or-minus1.8481.7𝑒-31.848\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.974±2.1​e​-​4plus-or-minus0.9742.1𝑒-40.974\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4
0.744±7.9​e​-​5plus-or-minus0.7447.9𝑒-50.744\scriptscriptstyle\pm\scriptstyle 7.9e\text{-}5

rec + sup
0.737±2.0​e​-​3plus-or-minus0.7372.0𝑒-30.737\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.862±1.3​e​-​3plus-or-minus0.8621.3𝑒-30.862\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.424±9.9​e​-​4plus-or-minus0.4249.9𝑒-40.424\scriptscriptstyle\pm\scriptstyle 9.9e\text{-}4
2.964±2.3​e​-​3plus-or-minus2.9642.3𝑒-32.964\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.449±2.3​e​-​4plus-or-minus0.4492.3𝑒-40.449\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4
0.811±5.3​e​-​4plus-or-minus0.8115.3𝑒-40.811\scriptscriptstyle\pm\scriptstyle 5.3e\text{-}4
5.124±2.4​e​-​2plus-or-minus5.1242.4𝑒-25.124\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}2
0.929±1.7​e​-​4plus-or-minus0.9291.7𝑒-40.929\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4
1.813±1.9​e​-​3plus-or-minus1.8131.9𝑒-31.813\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}3
0.974±2.2​e​-​4plus-or-minus0.9742.2𝑒-40.974\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}4
0.744±7.2​e​-​5plus-or-minus0.7447.2𝑒-50.744\scriptscriptstyle\pm\scriptstyle 7.2e\text{-}5

mask + target
0.719±3.5​e​-​3plus-or-minus0.7193.5𝑒-30.719\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3
0.866±4.3​e​-​4plus-or-minus0.8664.3𝑒-40.866\scriptscriptstyle\pm\scriptstyle 4.3e\text{-}4
0.407±8.2​e​-​4plus-or-minus0.4078.2𝑒-40.407\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4
2.952±3.5​e​-​3plus-or-minus2.9523.5𝑒-32.952\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}3
0.458±4.2​e​-​4plus-or-minus0.4584.2𝑒-40.458\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}4
0.828±6.0​e​-​4plus-or-minus0.8286.0𝑒-40.828\scriptscriptstyle\pm\scriptstyle 6.0e\text{-}4
5.373±1.8​e​-​2plus-or-minus5.3731.8𝑒-25.373\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.930±1.0​e​-​4plus-or-minus0.9301.0𝑒-40.930\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}4
1.849±2.1​e​-​3plus-or-minus1.8492.1𝑒-31.849\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}3
0.973±1.4​e​-​4plus-or-minus0.9731.4𝑒-40.973\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
0.745±2.5​e​-​4plus-or-minus0.7452.5𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4

- target sampling
0.724±7.0​e​-​3plus-or-minus0.7247.0𝑒-30.724\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}3
0.867±1.0​e​-​3plus-or-minus0.8671.0𝑒-30.867\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.403±1.1​e​-​3plus-or-minus0.4031.1𝑒-30.403\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
2.877±1.0​e​-​2plus-or-minus2.8771.0𝑒-22.877\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.466±9.4​e​-​4plus-or-minus0.4669.4𝑒-40.466\scriptscriptstyle\pm\scriptstyle 9.4e\text{-}4
0.828±2.5​e​-​4plus-or-minus0.8282.5𝑒-40.828\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
5.175±1.5​e​-​2plus-or-minus5.1751.5𝑒-25.175\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2
0.930±2.0​e​-​4plus-or-minus0.9302.0𝑒-40.930\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4
1.833±3.6​e​-​3plus-or-minus1.8333.6𝑒-31.833\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}3
0.974±3.3​e​-​4plus-or-minus0.9743.3𝑒-40.974\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}4
0.744±2.0​e​-​4plus-or-minus0.7442.0𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4

rec + target
0.705±2.3​e​-​3plus-or-minus0.7052.3𝑒-30.705\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.862±2.0​e​-​4plus-or-minus0.8622.0𝑒-40.862\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}4
0.431±1.4​e​-​3plus-or-minus0.4311.4𝑒-30.431\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
2.983±1.2​e​-​2plus-or-minus2.9831.2𝑒-22.983\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2
0.465±9.2​e​-​4plus-or-minus0.4659.2𝑒-40.465\scriptscriptstyle\pm\scriptstyle 9.2e\text{-}4
0.816±1.7​e​-​4plus-or-minus0.8161.7𝑒-40.816\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4
5.096±1.8​e​-​2plus-or-minus5.0961.8𝑒-25.096\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.928±1.9​e​-​4plus-or-minus0.9281.9𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}4
1.860±2.0​e​-​3plus-or-minus1.8602.0𝑒-31.860\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}3
0.974±3.4​e​-​5plus-or-minus0.9743.4𝑒-50.974\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}5
0.745±2.7​e​-​4plus-or-minus0.7452.7𝑒-40.745\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4

- target sampling
0.712±2.3​e​-​3plus-or-minus0.7122.3𝑒-30.712\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.860±9.3​e​-​4plus-or-minus0.8609.3𝑒-40.860\scriptscriptstyle\pm\scriptstyle 9.3e\text{-}4
0.437±3.3​e​-​3plus-or-minus0.4373.3𝑒-30.437\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3
2.933±2.0​e​-​2plus-or-minus2.9332.0𝑒-22.933\scriptscriptstyle\pm\scriptstyle 2.0e\text{-}2
0.450±1.7​e​-​3plus-or-minus0.4501.7𝑒-30.450\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}3
0.815±3.5​e​-​4plus-or-minus0.8153.5𝑒-40.815\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4
5.173±2.7​e​-​2plus-or-minus5.1732.7𝑒-25.173\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}2
0.928±5.6​e​-​5plus-or-minus0.9285.6𝑒-50.928\scriptscriptstyle\pm\scriptstyle 5.6e\text{-}5
1.811±1.5​e​-​3plus-or-minus1.8111.5𝑒-31.811\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
0.974±3.6​e​-​4plus-or-minus0.9743.6𝑒-40.974\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4
0.744±6.3​e​-​5plus-or-minus0.7446.3𝑒-50.744\scriptscriptstyle\pm\scriptstyle 6.3e\text{-}5

MLP-T-LR

no pretraining
0.662±7.6​e​-​3plus-or-minus0.6627.6𝑒-30.662\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}3
0.868±5.0​e​-​4plus-or-minus0.8685.0𝑒-40.868\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4
0.437±8.2​e​-​4plus-or-minus0.4378.2𝑒-40.437\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}4
3.028±1.8​e​-​2plus-or-minus3.0281.8𝑒-23.028\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.472±4.7​e​-​4plus-or-minus0.4724.7𝑒-40.472\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}4
0.808±1.5​e​-​4plus-or-minus0.8081.5𝑒-40.808\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4
5.424±2.2​e​-​2plus-or-minus5.4242.2𝑒-25.424\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}2
0.927±2.8​e​-​4plus-or-minus0.9272.8𝑒-40.927\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4
1.850±9.0​e​-​4plus-or-minus1.8509.0𝑒-41.850\scriptscriptstyle\pm\scriptstyle 9.0e\text{-}4
0.972±1.5​e​-​4plus-or-minus0.9721.5𝑒-40.972\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}4
0.747±7.6​e​-​5plus-or-minus0.7477.6𝑒-50.747\scriptscriptstyle\pm\scriptstyle 7.6e\text{-}5

mask
0.679±4.7​e​-​3plus-or-minus0.6794.7𝑒-30.679\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.868±2.3​e​-​4plus-or-minus0.8682.3𝑒-40.868\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4
0.413±1.0​e​-​3plus-or-minus0.4131.0𝑒-30.413\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
2.930±1.2​e​-​2plus-or-minus2.9301.2𝑒-22.930\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}2
0.450±7.3​e​-​4plus-or-minus0.4507.3𝑒-40.450\scriptscriptstyle\pm\scriptstyle 7.3e\text{-}4
0.826±1.3​e​-​3plus-or-minus0.8261.3𝑒-30.826\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
5.370±8.7​e​-​3plus-or-minus5.3708.7𝑒-35.370\scriptscriptstyle\pm\scriptstyle 8.7e\text{-}3
0.927±3.1​e​-​4plus-or-minus0.9273.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4
1.836±2.7​e​-​3plus-or-minus1.8362.7𝑒-31.836\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}3
0.973±8.8​e​-​5plus-or-minus0.9738.8𝑒-50.973\scriptscriptstyle\pm\scriptstyle 8.8e\text{-}5
0.745±1.1​e​-​4plus-or-minus0.7451.1𝑒-40.745\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4

rec
0.694±3.7​e​-​3plus-or-minus0.6943.7𝑒-30.694\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.861±1.6​e​-​4plus-or-minus0.8611.6𝑒-40.861\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4
0.414±1.5​e​-​3plus-or-minus0.4141.5𝑒-30.414\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}3
3.035±1.9​e​-​2plus-or-minus3.0351.9𝑒-23.035\scriptscriptstyle\pm\scriptstyle 1.9e\text{-}2
0.459±3.4​e​-​4plus-or-minus0.4593.4𝑒-40.459\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4
0.812±2.7​e​-​4plus-or-minus0.8122.7𝑒-40.812\scriptscriptstyle\pm\scriptstyle 2.7e\text{-}4
5.039±1.8​e​-​2plus-or-minus5.0391.8𝑒-25.039\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.925±1.2​e​-​4plus-or-minus0.9251.2𝑒-40.925\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4
1.803±2.3​e​-​3plus-or-minus1.8032.3𝑒-31.803\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}3
0.973±4.9​e​-​5plus-or-minus0.9734.9𝑒-50.973\scriptscriptstyle\pm\scriptstyle 4.9e\text{-}5
0.744±4.5​e​-​4plus-or-minus0.7444.5𝑒-40.744\scriptscriptstyle\pm\scriptstyle 4.5e\text{-}4

sup
0.698±3.7​e​-​3plus-or-minus0.6983.7𝑒-30.698\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}3
0.865±7.0​e​-​4plus-or-minus0.8657.0𝑒-40.865\scriptscriptstyle\pm\scriptstyle 7.0e\text{-}4
0.424±1.1​e​-​3plus-or-minus0.4241.1𝑒-30.424\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
3.107±6.8​e​-​3plus-or-minus3.1076.8𝑒-33.107\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}3
0.463±3.5​e​-​4plus-or-minus0.4633.5𝑒-40.463\scriptscriptstyle\pm\scriptstyle 3.5e\text{-}4
0.809±2.5​e​-​4plus-or-minus0.8092.5𝑒-40.809\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
5.442±1.6​e​-​2plus-or-minus5.4421.6𝑒-25.442\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}2
0.928±1.4​e​-​4plus-or-minus0.9281.4𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
1.849±5.4​e​-​4plus-or-minus1.8495.4𝑒-41.849\scriptscriptstyle\pm\scriptstyle 5.4e\text{-}4
0.975±2.8​e​-​4plus-or-minus0.9752.8𝑒-40.975\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4
0.746±1.4​e​-​5plus-or-minus0.7461.4𝑒-50.746\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}5

mask + sup
0.698±4.0​e​-​3plus-or-minus0.6984.0𝑒-30.698\scriptscriptstyle\pm\scriptstyle 4.0e\text{-}3
0.866±1.1​e​-​3plus-or-minus0.8661.1𝑒-30.866\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.421±8.6​e​-​4plus-or-minus0.4218.6𝑒-40.421\scriptscriptstyle\pm\scriptstyle 8.6e\text{-}4
3.088±5.1​e​-​3plus-or-minus3.0885.1𝑒-33.088\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.460±4.6​e​-​4plus-or-minus0.4604.6𝑒-40.460\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4
0.818±2.8​e​-​4plus-or-minus0.8182.8𝑒-40.818\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}4
5.407±3.3​e​-​3plus-or-minus5.4073.3𝑒-35.407\scriptscriptstyle\pm\scriptstyle 3.3e\text{-}3
0.927±5.1​e​-​4plus-or-minus0.9275.1𝑒-40.927\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4
1.824±3.0​e​-​3plus-or-minus1.8243.0𝑒-31.824\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}3
0.975±1.4​e​-​4plus-or-minus0.9751.4𝑒-40.975\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
0.747±8.2​e​-​5plus-or-minus0.7478.2𝑒-50.747\scriptscriptstyle\pm\scriptstyle 8.2e\text{-}5

rec + sup
0.705±2.4​e​-​3plus-or-minus0.7052.4𝑒-30.705\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.866±4.6​e​-​4plus-or-minus0.8664.6𝑒-40.866\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4
0.425±5.1​e​-​4plus-or-minus0.4255.1𝑒-40.425\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4
3.057±1.0​e​-​2plus-or-minus3.0571.0𝑒-23.057\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}2
0.444±1.4​e​-​3plus-or-minus0.4441.4𝑒-30.444\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.814±6.8​e​-​4plus-or-minus0.8146.8𝑒-40.814\scriptscriptstyle\pm\scriptstyle 6.8e\text{-}4
5.422±1.8​e​-​3plus-or-minus5.4221.8𝑒-35.422\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.927±6.6​e​-​5plus-or-minus0.9276.6𝑒-50.927\scriptscriptstyle\pm\scriptstyle 6.6e\text{-}5
1.811±1.0​e​-​3plus-or-minus1.8111.0𝑒-31.811\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.974±1.1​e​-​4plus-or-minus0.9741.1𝑒-40.974\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4
0.746±3.6​e​-​4plus-or-minus0.7463.6𝑒-40.746\scriptscriptstyle\pm\scriptstyle 3.6e\text{-}4

mask + target
0.673±1.0​e​-​3plus-or-minus0.6731.0𝑒-30.673\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.868±4.6​e​-​4plus-or-minus0.8684.6𝑒-40.868\scriptscriptstyle\pm\scriptstyle 4.6e\text{-}4
0.410±7.8​e​-​4plus-or-minus0.4107.8𝑒-40.410\scriptscriptstyle\pm\scriptstyle 7.8e\text{-}4
2.894±1.8​e​-​2plus-or-minus2.8941.8𝑒-22.894\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.460±1.4​e​-​3plus-or-minus0.4601.4𝑒-30.460\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}3
0.827±3.4​e​-​4plus-or-minus0.8273.4𝑒-40.827\scriptscriptstyle\pm\scriptstyle 3.4e\text{-}4
5.458±2.8​e​-​2plus-or-minus5.4582.8𝑒-25.458\scriptscriptstyle\pm\scriptstyle 2.8e\text{-}2
0.930±1.7​e​-​5plus-or-minus0.9301.7𝑒-50.930\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}5
1.849±4.2​e​-​3plus-or-minus1.8494.2𝑒-31.849\scriptscriptstyle\pm\scriptstyle 4.2e\text{-}3
0.972±2.5​e​-​4plus-or-minus0.9722.5𝑒-40.972\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}4
0.746±2.3​e​-​4plus-or-minus0.7462.3𝑒-40.746\scriptscriptstyle\pm\scriptstyle 2.3e\text{-}4

- target sampling
0.677±5.1​e​-​3plus-or-minus0.6775.1𝑒-30.677\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}3
0.866±1.1​e​-​3plus-or-minus0.8661.1𝑒-30.866\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}3
0.397±3.7​e​-​4plus-or-minus0.3973.7𝑒-40.397\scriptscriptstyle\pm\scriptstyle 3.7e\text{-}4
2.938±1.8​e​-​2plus-or-minus2.9381.8𝑒-22.938\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}2
0.462±4.4​e​-​4plus-or-minus0.4624.4𝑒-40.462\scriptscriptstyle\pm\scriptstyle 4.4e\text{-}4
0.826±1.2​e​-​4plus-or-minus0.8261.2𝑒-40.826\scriptscriptstyle\pm\scriptstyle 1.2e\text{-}4
5.384±1.5​e​-​2plus-or-minus5.3841.5𝑒-25.384\scriptscriptstyle\pm\scriptstyle 1.5e\text{-}2
0.929±1.6​e​-​4plus-or-minus0.9291.6𝑒-40.929\scriptscriptstyle\pm\scriptstyle 1.6e\text{-}4
1.840±2.2​e​-​3plus-or-minus1.8402.2𝑒-31.840\scriptscriptstyle\pm\scriptstyle 2.2e\text{-}3
0.973±1.1​e​-​4plus-or-minus0.9731.1𝑒-40.973\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4
0.747±5.0​e​-​4plus-or-minus0.7475.0𝑒-40.747\scriptscriptstyle\pm\scriptstyle 5.0e\text{-}4

rec + target
0.693±2.5​e​-​3plus-or-minus0.6932.5𝑒-30.693\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.866±3.1​e​-​4plus-or-minus0.8663.1𝑒-40.866\scriptscriptstyle\pm\scriptstyle 3.1e\text{-}4
0.432±6.1​e​-​4plus-or-minus0.4326.1𝑒-40.432\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4
3.045±1.1​e​-​2plus-or-minus3.0451.1𝑒-23.045\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}2
0.456±2.4​e​-​3plus-or-minus0.4562.4𝑒-30.456\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.812±6.1​e​-​4plus-or-minus0.8126.1𝑒-40.812\scriptscriptstyle\pm\scriptstyle 6.1e\text{-}4
5.344±8.1​e​-​3plus-or-minus5.3448.1𝑒-35.344\scriptscriptstyle\pm\scriptstyle 8.1e\text{-}3
0.928±1.4​e​-​4plus-or-minus0.9281.4𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}4
1.830±2.4​e​-​3plus-or-minus1.8302.4𝑒-31.830\scriptscriptstyle\pm\scriptstyle 2.4e\text{-}3
0.972±1.1​e​-​4plus-or-minus0.9721.1𝑒-40.972\scriptscriptstyle\pm\scriptstyle 1.1e\text{-}4
0.744±2.1​e​-​4plus-or-minus0.7442.1𝑒-40.744\scriptscriptstyle\pm\scriptstyle 2.1e\text{-}4

- target sampling
0.700±2.5​e​-​3plus-or-minus0.7002.5𝑒-30.700\scriptscriptstyle\pm\scriptstyle 2.5e\text{-}3
0.863±1.3​e​-​3plus-or-minus0.8631.3𝑒-30.863\scriptscriptstyle\pm\scriptstyle 1.3e\text{-}3
0.423±9.8​e​-​4plus-or-minus0.4239.8𝑒-40.423\scriptscriptstyle\pm\scriptstyle 9.8e\text{-}4
3.029±1.4​e​-​2plus-or-minus3.0291.4𝑒-23.029\scriptscriptstyle\pm\scriptstyle 1.4e\text{-}2
0.454±1.8​e​-​3plus-or-minus0.4541.8𝑒-30.454\scriptscriptstyle\pm\scriptstyle 1.8e\text{-}3
0.811±5.1​e​-​4plus-or-minus0.8115.1𝑒-40.811\scriptscriptstyle\pm\scriptstyle 5.1e\text{-}4
5.083±4.7​e​-​3plus-or-minus5.0834.7𝑒-35.083\scriptscriptstyle\pm\scriptstyle 4.7e\text{-}3
0.928±1.7​e​-​4plus-or-minus0.9281.7𝑒-40.928\scriptscriptstyle\pm\scriptstyle 1.7e\text{-}4
1.809±1.0​e​-​3plus-or-minus1.8091.0𝑒-31.809\scriptscriptstyle\pm\scriptstyle 1.0e\text{-}3
0.972±5.5​e​-​5plus-or-minus0.9725.5𝑒-50.972\scriptscriptstyle\pm\scriptstyle 5.5e\text{-}5
0.744±3.0​e​-​5plus-or-minus0.7443.0𝑒-50.744\scriptscriptstyle\pm\scriptstyle 3.0e\text{-}5
