---
arxiv: '2407.04057'
authors:
- \name Si-Yang Liu \email liusiyang@smail.nju.edu.cn \name Hao-Run Cai \email caihr@smail.nju.edu.cn
  \name Qi-Le Zhou \email zhouql@lamda.nju.edu.cn \name Han-Jia Ye \email yehj@lamda.nju.edu.cn
  \addr School of Artificial Intelligence, Nanjing University, China \addr National
  Key Laboratory for Novel Software Technology, Nanjing University, 210023, China
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'TALENT: A Tabular Analytics and Learning Toolbox'
url: https://arxiv.org/abs/2407.04057
year: 2024
---

# : A Tabular Analytics and Learning Toolbox

\nameSi-Yang Liu \emailliusiyang@smail.nju.edu.cn
  
\nameHao-Run Cai \emailcaihr@smail.nju.edu.cn
  
\nameQi-Le Zhou \emailzhouql@lamda.nju.edu.cn
  
\nameHan-Jia Ye \emailyehj@lamda.nju.edu.cn
  
\addrSchool of Artificial Intelligence, Nanjing University, China
  
\addrNational Key Laboratory for Novel Software Technology, Nanjing University, 210023, China

###### Abstract

Tabular data is one of the most common data sources in machine learning. Although a wide range of classical methods demonstrate practical utilities in this field, deep learning methods on tabular data are becoming promising alternatives due to their flexibility and ability to capture complex interactions within the data. Considering that deep tabular methods have diverse design philosophies, including the ways they handle features, design learning objectives, and construct model architectures, we introduce a versatile deep-learning toolbox called (Tabular Analytics and LEarNing Toolbox) to utilize, analyze, and compare tabular methods.
encompasses an extensive collection of more than 20 deep tabular prediction methods, associated with various encoding and normalization modules, and provides a unified interface that is easily integrable with new methods as they emerge. In this paper, we present the design and functionality of the toolbox, illustrate its practical application through several case studies, and investigate the performance of various methods fairly based on our toolbox.
Code is available at <https://github.com/qile2000/LAMDA-TALENT>.

Keywords:
Tabular Data, Deep Learning, Deep Tabular Prediction, Machine Learning

## 1 Introduction

Machine learning has achieved remarkable success across a broad spectrum of domains. Tabular data, characterized by datasets arranged in a table format, represents one of the most prevalent types of data used in machine learning applications such as click-through rate (CTR) prediction (Guo et al., [2017](#bib.bib19)), cybersecurity (Buczak and Guven, [2016](#bib.bib8)), medical analysis (Schwartz et al., [2007](#bib.bib36)), and identity protection (Liu et al., [2022](#bib.bib30)). In these datasets, each row typically represents an individual instance, while each column corresponds to a different attribute or feature. In the context of supervised learning, each training instance is associated with a label, which can be discrete for classification tasks or continuous for regression tasks as shown in [Table 1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ : A Tabular Analytics and Learning Toolbox"). Machine learning models are designed to learn a mapping from instances to their labels using the training data, with the aim to generalize this mapping to unseen instances from the same distribution.

| Age | Education | Occupation | Race | Sex | hours-per-week | Income |
| --- | --- | --- | --- | --- | --- | --- |
| 39 | Bachelors | Adm-clerical | White | Male | 40 | ≤\leq50K |
| 50 | Bachelors | Exec-managerial | White | Male | 13 | >>50K |
| 38 | HS-grad | Handlers-cleaners | White | Male | 40 | ≤\leq50K |
| 53 | 11th | Handlers-cleaners | Black | Male | 40 | ≤\leq50K |
| 28 | Bachelors | Prof-specialty | Black | Female | 40 | >>50K |
| 45 | Masters | Exec-managerial | White | Female | 50 | - |

Table 1: An example of a binary classification task from the Adult dataset (Becker and Kohavi, [1996](#bib.bib4)). The first six attributes/features (columns) are used to predict the final label. The first five rows are training examples, and the last row is a test instance with an unknown income.

Methodologies for analyzing tabular datasets have significantly evolved. Classical techniques such as Logistic Regression (LogReg), Support Vector Machine (SVM), Multi-Layer Perceptron (MLP), and decision tree have long served as the foundation of numerous algorithms (Bishop, [2006](#bib.bib5)). In practical applications, tree-based ensemble methods like XGBoost (Chen and Guestrin, [2016](#bib.bib12)), LightGBM (Ke et al., [2017](#bib.bib24)), and CatBoost (Prokhorenkova et al., [2018](#bib.bib35)) have demonstrated substantial improvements in performance. Inspired by the achievements of Deep Neural Networks (DNNs) in visual and linguistic tasks (Simonyan and Zisserman, [2015](#bib.bib39); Vaswani et al., [2017](#bib.bib42); Devlin et al., [2019](#bib.bib14)), researchers have recently developed deep learning models specifically for tabular data (Zhang et al., [2016](#bib.bib49); Borisov et al., [2022](#bib.bib7)).
While initial deep learning approaches for tabular data encountered challenges due to their inherent complexity, ongoing advancements have increasingly focused on enhancing complex feature interaction modeling and mimicking the decision-making processes found in tree-based models (Cheng et al., [2016](#bib.bib13); Guo et al., [2017](#bib.bib19); Popov et al., [2020](#bib.bib34); Chang et al., [2022](#bib.bib9)).
Continuous research has shown that modern deep learning techniques can dramatically improve upon the performance of traditional models such as MLPs (Arik and Pfister, [2021](#bib.bib2); Gorishniy et al., [2021](#bib.bib16); Kadra et al., [2021](#bib.bib23)). These advanced deep tabular models are able to effectively model complex relationships among instances or features, uncover underlying patterns in the datasets, and improve prediction performance.

While deep learning offers significant benefits for analyzing tabular data, its practical application is often hindered by the lack of uniform interfaces and varying preprocessing demands among different methods. We introduce a versatile and powerful toolbox, TALENT (Tabular Analytics and LEarNing Toolbox), for tabular data prediction.
integrates diverse methodologies, including classical methods as well as advanced deep methods, into a unifying framework. not only standardizes interfaces and streamlines preprocessing steps, making it easy to integrate new methods as they emerge, but also ensures that all methods can be fairly compared, providing a reliable basis for evaluating their effectiveness in different scenarios.
More importantly, our toolbox enables the composition of effective deep tabular modules and facilitates data analysis, offering scalable solutions that can adapt to various complexities and data-specific needs.
The advantages of the toolbox are

* •

  Model Diversity. Our toolbox integrates an extensive array of more than 20 diverse deep tabular methods with uniform interfaces, allowing users to select the best-fit model based on the complexity and specifics of their tasks.
* •

  Encoding Techniques. In addition to various encoding strategies for categorical features, provides eight encoding techniques for numerical features. These comprehensive encoding techniques ensure versatile data representation tailored to different analytical requirements.
* •

  Extensibility. The modular architecture of the toolbox ensures flexibility and future scalability. Users can easily add new models and methods according to practical requirements, making it a continuously relevant and valuable resource.

## 2 for Tabular Prediction

We formally define the tabular prediction task and then provide an overview of the tabular prediction
methods supported by .

### 2.1 Preliminary

A supervised tabular dataset is formatted as N𝑁N examples and d𝑑d features, corresponding to N𝑁N rows and d𝑑d columns in the table. An instance xi∈ℝdx\_{i}\in\mathbb{R}{{}^{d}} is depicted by its d𝑑d feature values.
Assume xi,jsubscript𝑥

𝑖𝑗x\_{i,j} as the j𝑗j-th feature of instance xisubscript𝑥𝑖x\_{i}, it could be a numerical (continuous) one xi,jnum∈ℝsuperscriptsubscript𝑥

𝑖𝑗numℝx\_{i,j}^{\textit{\rm num}}\in\mathbb{R}, or a categorical (discrete) value xi,jcatsuperscriptsubscript𝑥

𝑖𝑗catx\_{i,j}^{\textit{\rm cat}}.
The categorical features are usually transformed in an index (integer).
Each instance is associated with a label yisubscript𝑦𝑖y\_{i}, where yi∈{1,−1}subscript𝑦𝑖11y\_{i}\in\{1,-1\} in a binary classification task, yi∈[C]={1,…,C}subscript𝑦𝑖delimited-[]𝐶1…𝐶y\_{i}\in[C]=\{1,\ldots,C\} in a multi-class classification task, and yi∈ℝsubscript𝑦𝑖ℝy\_{i}\in\mathbb{R} in a regression task.
Given a tabular dataset 𝒟={(xi,yi)}i=1N𝒟superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1𝑁\mathcal{D}=\{(x\_{i},y\_{i})\}\_{i=1}^{N}, we aim to learn a model f𝑓f on 𝒟𝒟\mathcal{D} via empirical risk minimization that maps xisubscript𝑥𝑖x\_{i} to its label yisubscript𝑦𝑖y\_{i}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minf​∑(xi,yi)∈𝒟ℓ​(y,y^i=f​(xi))+Ω​(f).subscript𝑓subscriptsubscript𝑥𝑖subscript𝑦𝑖𝒟ℓ  𝑦subscript^𝑦𝑖 𝑓subscript𝑥𝑖Ω𝑓\min\_{f}\;\sum\_{(x\_{i},y\_{i})\in\mathcal{D}}\ell(y,\;\hat{y}\_{i}=f(x\_{i}))+\Omega(f)\;. |  | (1) |

ℓ​(⋅,⋅)ℓ⋅⋅\ell(\cdot,\cdot) measures the discrepancy between the predicted label y^isubscript^𝑦𝑖\hat{y}\_{i} and the true label yisubscript𝑦𝑖y\_{i}, *e.g.*, cross-entropy in classification.
Ω​(⋅)Ω⋅\Omega(\cdot) is the regularization on the model.
We expect the learned f𝑓f is able to extend its ability to unseen instances sampled from the same distribution as 𝒟𝒟\mathcal{D}.

### 2.2 Supported Methods

In , we implement a comprehensive range of models implement the mapping f𝑓f from features to outputs with the same interface, covering classical methods, tree-based methods, and advanced deep tabular methods.

Classical models in include K-Nearest Neighbors (KNN) and SVM for various tasks, complemented by Linear Regression (LR) for regression tasks and Logistic Regression (LogReg), Naive Bayes, and Nearest Class Mean (NCM) for classification.

Tree-based methods in utilize powerful algorithms, including Random Forest, XGBoost (Chen and Guestrin, [2016](#bib.bib12)), CatBoost (Prokhorenkova et al., [2018](#bib.bib35)) and LightGBM (Ke et al., [2017](#bib.bib24)), known for their high efficiency and strong predictive performance across a variety of datasets.

!(/html/2407.04057/assets/x1.png)

Figure 1: Various deep prediction methods for tabular data in .

Our toolbox provides a comprehensive selection of representative deep tabular prediction methods, each meticulously designed to tackle distinct challenges in tabular data analysis. A taxonomy of these methods is shown in [Figure 1](#S2.F1 "Figure 1 ‣ 2.2 Supported Methods ‣ 2 for Tabular Prediction ‣ : A Tabular Analytics and Learning Toolbox"):

* •

  MLP: A multi-layer neural network, implemented according to Gorishniy et al. ([2021](#bib.bib16)).
* •

  ResNet: A DNN that uses skip connections across many layers, which is implemented according to Gorishniy et al. ([2021](#bib.bib16)).
* •

  SNN (Klambauer et al., [2017](#bib.bib25)): An MLP-like architecture utilizing the SELU activation, which facilitates the training of deeper neural networks.
* •

  DANets (Chen et al., [2022](#bib.bib10)): A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.
* •

  TabCaps (Chen et al., [2023](#bib.bib11)): A capsule network that encapsulates all feature values of a record into vectorial features.
* •

  DCNv2 (Wang et al., [2021](#bib.bib43)) consists of an MLP-like module combined with a feature crossing module, which includes both linear layers and multiplications.
* •

  NODE (Popov et al., [2020](#bib.bib34)): A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.
* •

  GrowNet (Badirli et al., [2020](#bib.bib3)): A gradient boosting framework that uses shallow neural networks as weak learners.
* •

  TabNet (Arik and Pfister, [2021](#bib.bib2)): A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.
* •

  TabR (Gorishniy et al., [2024](#bib.bib18)): A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.
* •

  ModernNCA (Ye et al., [2024a](#bib.bib47)): A deep tabular model inspired by traditional Neighbor Component Analysis (Goldberger et al., [2004](#bib.bib15)), which makes predictions based on the relationships with neighbors in a learned embedding space.
* •

  DNNR (Nader et al., [2022](#bib.bib33)) enhances KNN by using local gradients and Taylor approximations for more accurate and interpretable predictions.
* •

  AutoInt (Song et al., [2019](#bib.bib41)): A token-based method that uses a multi-head self-attentive neural network to automatically learn high-order feature interactions.
* •

  Saint (Somepalli et al., [2022](#bib.bib40)): A token-based method that leverages row and column attention mechanisms for tabular data.
* •

  TabTransformer (Huang et al., [2020](#bib.bib21)): A token-based method that enhances tabular data modeling by transforming categorical features into contextual embeddings.
* •

  FT-Transformer (Gorishniy et al., [2021](#bib.bib16)): A token-based method which transforms features to embeddings and applies a series of attention-based transformations to the embeddings.
* •

  TANGOS (Jeffares et al., [2023](#bib.bib22)): A regularization-based method for tabular data that uses gradient attributions to encourage neuron specialization and orthogonalization.
* •

  SwitchTab (Wu et al., [2024](#bib.bib45)): A self-supervised method tailored for tabular data that improves representation learning through an asymmetric encoder-decoder framework.
* •

  PTaRL (Ye et al., [2024b](#bib.bib48)): A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space.
* •

  TabPFN (Hollmann et al., [2023](#bib.bib20)): A general model which involves the use of pre-trained deep neural networks that can be directly applied to other tabular classification tasks.
* •

  HyperFast (Bonet et al., [2024](#bib.bib6)): A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.
* •

  TabPTM (Ye et al., [2023](#bib.bib46)): A general method for tabular data that standardizes heterogeneous datasets using meta-representations, allowing a pre-trained model to generalize to unseen datasets without additional training.

### 2.3 Encoding Techniques

According to Gorishniy et al. ([2022](#bib.bib17)), embeddings for numerical features greatly improve the performance of deep learning models on tabular data by providing more expressive and powerful initial representations. This approach is useful for both MLPs and advanced Transformer-like architectures. In , we incorporate various numerical encoding techniques, enhancing the input quality for machine learning models. The diverse range of encoding methods ensures effective and customized data preprocessing for different analytical needs. Here are the encoding methods included in :

* •

  Quantile-based Binning (Q\_bins) constructs bins by dividing value ranges according to the quantiles of the individual feature distributions, and replaces the original values with their corresponding bin indices.
* •

  Target-aware Binning (T\_bins) creates bins using training labels to correspond to narrow ranges of possible target values. This approach is similar to the “C4.5 Discretization” algorithm (Kohavi and Sahami, [1996](#bib.bib26)), which splits the value range of each feature using the target as guidance.
* •

  Quantile-based Unary Encoding (Q\_Unary) (Li and Lin, [2006](#bib.bib28)) converts numerical values into unary binary-encoded bin indices based on quantiles.
* •

  Target-aware Unary Encoding (T\_Unary) (Li and Lin, [2006](#bib.bib28)) generates unary binary-encoded bin indices using target-aware transformations.
* •

  Quantile-based Johnson Encoding (Q\_Johnson) (Shah et al., [2022](#bib.bib37)) encodes numerical data based on quantile intervals using Johnson distribution transformations (Libaw and Craig, [1953](#bib.bib29)), replacing original values with Johnson binary-encoded bin indices.
* •

  Target-aware Johnson Encoding (T\_Johnson) (Shah et al., [2022](#bib.bib37)) applies Johnson transformations with target-aware bins, replacing original values with Johnson binary-encoded bin indices (Libaw and Craig, [1953](#bib.bib29)).
* •

  Quantile-based Piecewise Linear Encoding (Q\_PLE) (Gorishniy et al., [2022](#bib.bib17)) segments numerical data based on quantiles and applies piecewise linear transformations.
* •

  Target-aware Piecewise Linear Encoding (T\_PLE) (Gorishniy et al., [2022](#bib.bib17)) builds target-aware bins and applies piecewise linear transformations.

Additionally, employs various categorical encoding techniques, including Ordinal encoding, One-Hot encoding, Binary encoding, Hash encoding (Weinberger et al., [2009](#bib.bib44)), Target encoding (Micci-Barreca, [2001](#bib.bib32)), Leave-One-Out encoding, and CatBoost encoding (Prokhorenkova et al., [2018](#bib.bib35)).

## 3 Toolbox Usage

In this section, we introduce the dependencies and workflow when using .

### 3.1 Dependencies

leverages open-source libraries to support its advanced data processing and machine learning functionalities, following the organized code structure introduced in rtdl (Gorishniy et al., [2021](#bib.bib16)). For model optimization and hyperparameter tuning, it utilizes Optuna (Akiba et al., [2019](#bib.bib1)). These dependencies are carefully selected, providing users with a powerful, flexible, and efficient toolbox for tackling diverse challenges in the analysis of tabular data.

### 3.2 The workflow of

!(/html/2407.04057/assets/x2.png)

Figure 2: Flowchart depicting the data prediction process with Talent.

The flowchart in [Figure 2](#S3.F2 "Figure 2 ‣ 3.2 The workflow of ‣ 3 Toolbox Usage ‣ : A Tabular Analytics and Learning Toolbox") visually represents the streamlined workflow facilitated by our toolbox. It begins with loading the data, followed by preprocessing, hyperparameter tuning, model training, prediction, and finally, evaluation. This structured process ensures a smooth transition from raw data to meaningful results.

offers interfaces to over 30 methods as mentioned above. When using , users have the flexibility to configure hyperparameters within the JSON files stored in the configs folder. Users can either modify the hyperparameters directly in the JSON files or override them through command-line inputs.

For example, to run a classical method, *e.g.*, xgboost, using the default hyperparameters from the JSON configuration file, users would use the following command:

[⬇](data:text/plain;base64,ICAgIHB5dGhvbiB0cmFpbl9tb2RlbF9jbGFzc2ljYWwucHkgLS1tb2RlbF90eXBlIHhnYm9vc3Q=)

python train\_model\_classical.py --model\_type xgboost

To run a deep learning method, *e.g.*, mlp, use the following command:

[⬇](data:text/plain;base64,ICAgIHB5dGhvbiB0cmFpbl9tb2RlbF9kZWVwLnB5IC0tbW9kZWxfdHlwZSBtbHA=)

python train\_model\_deep.py --model\_type mlp

The command-line arguments for running the models are described below, allowing for customization depending on the specific requirements of the task and the dataset:

* •

  model\_type specifies the model to be used.
* •

  dataset defines the dataset to be used, specified by the name of the dataset folder.
* •

  max\_epoch sets the maximum number of epochs for training the model.
* •

  batch\_size determines the size of the batch of samples to be processed per gradient update.
* •

  seed\_num specifies the number of different seed values to use for the experiments, ensuring multiple runs for statistical robustness.
* •

  normalization chooses the type of normalization to apply to the dataset, including Standard scaling, MinMax scaling, Quantile transformation, MaxAbs scaling, Power transformation, and Robust scaling.
* •

  num\_nan\_policy: Defines the policy for handling missing numerical data, such as mean, median.
* •

  cat\_nan\_policy specifies the policy for handling missing categorical data, such as most\_frequent, constant.
* •

  cat\_policy determines the categorical encoding method to be applied.
* •

  num\_policy determines the numerical encoding method to be applied.
* •

  n\_trials sets the number of trials for hyperparameter tuning (if applicable).
* •

  tune indicates whether hyperparameter tuning should be performed, with options typically being True or False.

By using , users can benefit from its configuration and evaluation interface, ensuring that experiments are both reproducible and customizable to meet specific research needs.

### 3.3 Model Default Hyperparameters and Search Space

For each model supported by , there are two essential JSON files that aid in the configuration and search of hyperparameters of the model:

* •

  Default Hyperparameters: Each model has a corresponding JSON file in the
    
  configs/default folder containing the default hyperparameters. These hyperparameters are pre-defined based on either values used in the literature or empirically determined settings that provide good performance.
* •

  Hyperparameter Search Space: Each model also has a corresponding JSON file in the configs/opt\_space folder that defines the hyperparameter search space for the tuning process. This file specifies the bounds and types of each hyperparameter (*e.g.*, continuous, discrete, categorical) that can be explored using automated hyperparameter optimization tools like Optuna (Akiba et al., [2019](#bib.bib1)). This setup facilitates a more systematic and potentially more effective search for optimal model configurations, especially useful when adapting models to new datasets or specific tasks.

These configuration files ensure that users not only have a reliable starting point for each model but also the flexibility to explore and optimize hyperparameters to match specific data characteristics or objectives.

For instance, when the users choose to use MLP, we have:

* •

  configs/default/mlp.json:

  [⬇](data:text/plain;base64,ewogICAgIm1scCI6IHsKICAgICAgICAibW9kZWwiOiB7CiAgICAgICAgICAgICJkX2xheWVycyI6IFszODQsIDM4NF0sCiAgICAgICAgICAgICJkcm9wb3V0IjogMC4xCiAgICAgICAgfSwKICAgICAgICAidHJhaW5pbmciOiB7CiAgICAgICAgICAgICJsciI6IDNlLTQsCiAgICAgICAgICAgICJ3ZWlnaHRfZGVjYXkiOiAxZS01CiAgICAgICAgfQogICAgfQp9)

  {

  "mlp": {

  "model": {

  "d\_layers": [384, 384],

  "dropout": 0.1

  },

  "training": {

  "lr": 3e-4,

  "weight\_decay": 1e-5

  }

  }

  }
* •

  configs/opt\_space/mlp.json:

  [⬇](data:text/plain;base64,ewogICAgIm1scCI6IHsKICAgICAgICAibW9kZWwiOiB7CiAgICAgICAgICAgICJkX2xheWVycyI6IFsiJG1scF9kX2xheWVycyIsIDEsIDgsIDY0LCA1MTJdLAogICAgICAgICAgICAiZHJvcG91dCI6IFsiP3VuaWZvcm0iLCAwLjAsIDAuMCwgMC41XQogICAgICAgIH0sCiAgICAgICAgInRyYWluaW5nIjogewogICAgICAgICAgICAibHIiOiBbImxvZ3VuaWZvcm0iLCAxZS0wNSwgMC4wMV0sCiAgICAgICAgICAgICJ3ZWlnaHRfZGVjYXkiOiBbIj9sb2d1bmlmb3JtIiwgMC4wLCAxZS0wNiwgMC4wMDFdCiAgICAgICAgfQogICAgfQp9)

  {

  "mlp": {

  "model": {

  "d\_layers": ["$mlp\_d\_layers", 1, 8, 64, 512],

  "dropout": ["?uniform", 0.0, 0.0, 0.5]

  },

  "training": {

  "lr": ["loguniform", 1e-05, 0.01],

  "weight\_decay": ["?loguniform", 0.0, 1e-06, 0.001]

  }

  }

  }

### 3.4 Example Usage

Below is an example of how to use the toolbox to run experiments across different seeds, allowing for robust evaluation of the performance of methods:

[⬇](data:text/plain;base64,ICAgICMgUGFyc2UgdGhlIGFyZ3VtZW50cyBhbmQgbG9hZCBkZWZhdWx0IHBhcmFtZXRlcnMgYW5kIG9wdGltaXphdGlvbiBzcGFjZQogICAgYXJncywgZGVmYXVsdF9wYXJhLCBvcHRfc3BhY2UgPSBnZXRfYXJncygpCiAgICAjIExvYWQgdGhlIHRyYWluaW5nLCB2YWxpZGF0aW9uLCBhbmQgdGVzdCBkYXRhCiAgICB0cmFpbl92YWxfZGF0YSwgdGVzdF9kYXRhLCBpbmZvID0gZ2V0X2RhdGFzZXQoYXJncy5kYXRhc2V0LCBhcmdzLmRhdGFzZXRfcGF0aCkKICAgICMgSWYgaHlwZXJwYXJhbWV0ZXIgdHVuaW5nIGlzIGVuYWJsZWQsIHR1bmUgdGhlIGh5cGVycGFyYW1ldGVycwogICAgaWYgYXJncy50dW5lOgogICAgICAgIGFyZ3MgPSB0dW5lX2h5cGVyX3BhcmFtZXRlcnMoYXJncywgb3B0X3NwYWNlLCB0cmFpbl92YWxfZGF0YSwgaW5mbykKICAgICMjIFRyYWluaW5nIHN0YWdlIG92ZXIgZGlmZmVyZW50IHJhbmRvbSBzZWVkcwogICAgZm9yIHNlZWQgaW4gdHFkbShyYW5nZShhcmdzLnNlZWRfbnVtKSk6CiAgICAgICAgYXJncy5zZWVkID0gc2VlZCAgIyBVcGRhdGUgc2VlZCBmb3IgcmVwcm9kdWNpYmlsaXR5CiAgICAgICAgIyBHZXQgdGhlIG1ldGhvZCBiYXNlZCBvbiB0aGUgbW9kZWwgdHlwZQogICAgICAgIG1ldGhvZCA9IGdldF9tZXRob2QoYXJncy5tb2RlbF90eXBlKShhcmdzLCBpbmZvWyd0YXNrX3R5cGUnXSA9PSAncmVncmVzc2lvbicpCiAgICAgICAgIyBUcmFpbiB0aGUgbW9kZWwgYW5kIHJlY29yZCB0aGUgdGltZSBjb3N0CiAgICAgICAgdGltZV9jb3N0ID0gbWV0aG9kLmZpdCh0cmFpbl92YWxfZGF0YSwgaW5mbywgdHJhaW49VHJ1ZSkKICAgICAgICAjIFByZWRpY3QgdXNpbmcgdGhlIHRyYWluZWQgbW9kZWwgb24gdGhlIHRlc3QgZGF0YQogICAgICAgICMgYW5kIGNhbGN1bGF0ZSB2YXJpb3VzIG1ldHJpY3MKICAgICAgICB2cmVzLCBtZXRyaWNfbmFtZSwgcHJlZGljdF9sb2dpdHMgPSBtZXRob2QucHJlZGljdCh0ZXN0X2RhdGEsIGluZm8p)

1 # Parse the arguments and load default parameters and optimization space

2 args, default\_para, opt\_space = get\_args()

3 # Load the training, validation, and test data

4 train\_val\_data, test\_data, info = get\_dataset(args.dataset, args.dataset\_path)

5 # If hyperparameter tuning is enabled, tune the hyperparameters

6 if args.tune:

7 args = tune\_hyper\_parameters(args, opt\_space, train\_val\_data, info)

8 ## Training stage over different random seeds

9 for seed in tqdm(range(args.seed\_num)):

10 args.seed = seed # Update seed for reproducibility

11 # Get the method based on the model type

12 method = get\_method(args.model\_type)(args, info[’task\_type’] == ’regression’)

13 # Train the model and record the time cost

14 time\_cost = method.fit(train\_val\_data, info, train=True)

15 # Predict using the trained model on the test data

16 # and calculate various metrics

17 vres, metric\_name, predict\_logits = method.predict(test\_data, info)

In this script:

* •

  The get\_args function retrieves and parses the default arguments, default hyperparameters, and the optimization space for hyperparameter tuning.
* •

  The get\_dataset function loads the specified dataset from the given path, splits it into training/validation and test sets, and provides additional information about the dataset.
* •

  If hyperparameter tuning is enabled, the tune\_hyper\_parameters function adjusts the arguments based on the optimization space and the training/validation data.
* •

  The get\_method function selects the appropriate modeling class based on the model type specified in args.model\_type.
* •

  The seed is updated for each iteration, ensuring that each run is reproducible but distinct, enhancing the statistical robustness of the results.
* •

  The performance metrics and predictions are recorded for each seed, allowing for a comprehensive evaluation of the model across different initializations. For classification tasks, the evaluation metrics include Accuracy, Average Recall, Average Precision, F1 Score, LogLoss, and AUC. For regression tasks, the metrics include MAE, RMSE, and R2 (Lewis-Beck, [2015](#bib.bib27)).

### 3.5 Adding New Methods

!(/html/2407.04057/assets/x3.png)

Figure 3: Workflow for Adding a New Method to .

is designed to be highly customizable, allowing users to integrate new machine learning methods effortlessly. Whether users are adding an well-known algorithm or experimenting with a novel approach, follow these steps to expand the capabilities of the toolbox, as illustrated in [Figure 3](#S3.F3 "Figure 3 ‣ 3.5 Adding New Methods ‣ 3 Toolbox Usage ‣ : A Tabular Analytics and Learning Toolbox"):

1. 1.

   Register the Model: Start by registering the new model class in the model/models directory. Ensure that this class includes the architecture of the model, defining how the model will be constructed.
2. 2.

   Create the Method Class: Create a new method class within the model/methods directory. This class should inherit from the base class provided in base.py. Implement the necessary components of the machine learning method in this class, including the training and prediction processes.
3. 3.

   Method Integration: Integrate the new method into the workflow of by adding its name to the get\_method function located in model/utils.py. This function maps model types to their respective classes, enabling the toolbox to instantiate the correct model.
4. 4.

   Configure Parameters: Update the JSON files in the configs/default and
     
   configs/opt\_space directories to include default hyperparameters and hyperparameter search spaces for the new method.
5. 5.

   Adjust Training Processes: If the method requires a unique training procedure, modify the relevant functions in model/methods/base.py. Tailor these functions to accommodate any special optimization strategies that the method requires.

By following these steps, researchers can add new algorithms to , adapting it to meet diverse research needs. For detailed examples and additional guidance, refer to the implementation of existing methods in the model/methods directory.

## 4 Preliminary Experiments

!(/html/2407.04057/assets/x4.png)

(a) Binary Classification

!(/html/2407.04057/assets/x5.png)

(b) Multi-Class Classification

!(/html/2407.04057/assets/x6.png)

(c) Regression

!(/html/2407.04057/assets/x7.png)

(d) All Tasks

Figure 4: Performance-Efficiency-Size comparison of representative tabular methods on our toolbox for (a) binary classification, (b) multi-class classification, (c) regression tasks, and (d) all task types. The performance is measured by the average rank of all methods (lower is better).
We also consider the dummy baseline, which outputs the label of the major class and the average labels for classification and regression tasks, respectively.

We provide comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner in [Figure 4](#S4.F4 "Figure 4 ‣ 4 Preliminary Experiments ‣ : A Tabular Analytics and Learning Toolbox").
Three tabular prediction tasks, namely, binary classification, multi-class classification, and regression, are considered, and each subfigure represents a different task type. The datasets are available at [Google Drive](https://drive.google.com/file/d/18RHGSA1nASbsF1KAHCqLJasYsZIBXJ8D/view?usp=sharing).

We use accuracy and RMSE as the metrics for classification and regression, respectively. To calibrate the metrics, we choose the average performance rank to compare all methods, where a lower rank indicates better performance, following Sheskin ([2003](#bib.bib38)).
Efficiency is calculated by the average training time in seconds, with lower values denoting better time efficiency. The model size is visually indicated by the radius of the circles, offering a quick glance at the trade-off between model complexity and performance.

From the comparison, we observe that CatBoost achieves the best average rank in most classification and regression tasks, consistent with findings in McElfresh et al. ([2023](#bib.bib31)). Among all deep tabular methods, ModernNCA performs the best in most cases while maintaining an acceptable training cost.
These visualizations serve as an effective tool for quickly and fairly assessing the strengths and weaknesses of various tabular methods across different task types, enabling researchers and practitioners to make informed decisions when selecting suitable modeling techniques for their specific needs.

## 5 Conclusion

We introduce , a machine learning toolbox for tabular data prediction tasks. implements both classical and deep tabular methods and includes several modules, such as hyperparameter tuning and preprocessing capabilities, to optimize the learning efficiency and effectiveness on tabular datasets.
We also leverage to compare recent deep tabular methods fairly across numerous datasets. The toolbox is designed to be user-friendly and accessible to practitioners across diverse fields, providing a unified interface that is adaptable for integration with newly designed methods.

## References

* Akiba et al. (2019)

  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.
  Optuna: A next-generation hyperparameter optimization framework.
  In *KDD*, pages 2623–2631, 2019.
* Arik and Pfister (2021)

  Sercan Ö. Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  In *AAAI*, pages 6679–6687, 2021.
* Badirli et al. (2020)

  Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, and Sathiya S. Keerthi.
  Gradient boosting neural networks: Grownet.
  *CoRR*, abs/2002.07971, 2020.
* Becker and Kohavi (1996)

  Barry Becker and Ronny Kohavi.
  Adult.
  UCI Machine Learning Repository, 1996.
  DOI: https://doi.org/10.24432/C5XW20.
* Bishop (2006)

  Christopher Bishop.
  *Pattern recognition and machine learning*.
  Springer, 2006.
* Bonet et al. (2024)

  David Bonet, Daniel Mas Montserrat, Xavier Giró-i-Nieto, and Alexander G. Ioannidis.
  Hyperfast: Instant classification for tabular data.
  In *AAAI*, pages 11114–11123, 2024.
* Borisov et al. (2022)

  Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci.
  Deep neural networks and tabular data: A survey.
  *CoRR*, abs/2110.01889, 2022.
* Buczak and Guven (2016)

  Anna L. Buczak and Erhan Guven.
  A survey of data mining and machine learning methods for cyber security intrusion detection.
  *IEEE Commun. Surv. Tutorials*, 18(2):1153–1176, 2016.
* Chang et al. (2022)

  Chun-Hao Chang, Rich Caruana, and Anna Goldenberg.
  NODE-GAM: neural generalized additive model for interpretable deep learning.
  In *ICLR*, 2022.
* Chen et al. (2022)

  Jintai Chen, Kuanlun Liao, Yao Wan, Danny Z. Chen, and Jian Wu.
  Danets: Deep abstract networks for tabular data classification and regression.
  In *AAAI*, pages 3930–3938, 2022.
* Chen et al. (2023)

  Jintai Chen, KuanLun Liao, Yanwen Fang, Danny Chen, and Jian Wu.
  Tabcaps: A capsule neural network for tabular data classification with bow routing.
  In *ICLR*, 2023.
* Chen and Guestrin (2016)

  Tianqi Chen and Carlos Guestrin.
  Xgboost: A scalable tree boosting system.
  In *KDD*, pages 785–794, 2016.
* Cheng et al. (2016)

  Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah.
  Wide & deep learning for recommender systems.
  In *DLRS*, pages 7–10, 2016.
* Devlin et al. (2019)

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  BERT: pre-training of deep bidirectional transformers for language understanding.
  In *NAACL-HLT*, pages 4171–4186, 2019.
* Goldberger et al. (2004)

  Jacob Goldberger, Sam T. Roweis, Geoffrey E. Hinton, and Ruslan Salakhutdinov.
  Neighbourhood components analysis.
  In *NIPS*, pages 513–520, 2004.
* Gorishniy et al. (2021)

  Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko.
  Revisiting deep learning models for tabular data.
  In *NeurIPS*, pages 18932–18943, 2021.
* Gorishniy et al. (2022)

  Yury Gorishniy, Ivan Rubachev, and Artem Babenko.
  On embeddings for numerical features in tabular deep learning.
  In *NeurIPS*, pages 24991–25004, 2022.
* Gorishniy et al. (2024)

  Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko.
  Tabr: Tabular deep learning meets nearest neighbors in 2023.
  In *ICLR*, 2024.
* Guo et al. (2017)

  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He.
  Deepfm: A factorization-machine based neural network for CTR prediction.
  In *IJCAI*, pages 1725–1731, 2017.
* Hollmann et al. (2023)

  Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.
  Tabpfn: A transformer that solves small tabular classification problems in a second.
  In *ICLR*, 2023.
* Huang et al. (2020)

  Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar S. Karnin.
  Tabtransformer: Tabular data modeling using contextual embeddings.
  *CoRR*, abs/2012.06678, 2020.
* Jeffares et al. (2023)

  Alan Jeffares, Tennison Liu, Jonathan Crabbé, Fergus Imrie, and Mihaela van der Schaar.
  Tangos: Regularizing tabular neural networks through gradient orthogonalization and specialization.
  In *ICLR*, 2023.
* Kadra et al. (2021)

  Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka.
  Well-tuned simple nets excel on tabular datasets.
  In *NeurIPS*, pages 23928–23941, 2021.
* Ke et al. (2017)

  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
  Lightgbm: A highly efficient gradient boosting decision tree.
  In *NIPS*, pages 3146–3154, 2017.
* Klambauer et al. (2017)

  Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
  Self-normalizing neural networks.
  In *NIPS*, pages 971–980, 2017.
* Kohavi and Sahami (1996)

  Ron Kohavi and Mehran Sahami.
  Error-based and entropy-based discretization of continuous features.
  In *KDD*, pages 114–119, 1996.
* Lewis-Beck (2015)

  Michael S. Lewis-Beck.
  *Applied regression: An introduction*, volume 22.
  Sage publications, 2015.
* Li and Lin (2006)

  Ling Li and Hsuan-Tien Lin.
  Ordinal regression by extended binary classification.
  In *NIPS*, pages 865–872, 2006.
* Libaw and Craig (1953)

  William H. Libaw and Leonard J. Craig.
  A photoelectric decimal-coded shaft digitizer.
  *Trans. I R E Prof. Group Electron. Comput.*, 2(3):1–4, 1953.
* Liu et al. (2022)

  Bo Liu, Ming Ding, Sina Shaham, Wenny Rahayu, Farhad Farokhi, and Zihuai Lin.
  When machine learning meets privacy: A survey and outlook.
  *ACM Comput. Surv.*, 54(2):31:1–31:36, 2022.
* McElfresh et al. (2023)

  Duncan C. McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C., Ganesh Ramakrishnan, Micah Goldblum, and Colin White.
  When do neural nets outperform boosted trees on tabular data?
  In *NeurIPS*, pages 76336–76369, 2023.
* Micci-Barreca (2001)

  Daniele Micci-Barreca.
  A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.
  *SIGKDD Explor.*, 3(1):27–32, 2001.
* Nader et al. (2022)

  Youssef Nader, Leon Sixt, and Tim Landgraf.
  DNNR: differential nearest neighbors regression.
  In *ICML*, pages 16296–16317, 2022.
* Popov et al. (2020)

  Sergei Popov, Stanislav Morozov, and Artem Babenko.
  Neural oblivious decision ensembles for deep learning on tabular data.
  In *ICLR*, 2020.
* Prokhorenkova et al. (2018)

  Liudmila Ostroumova Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  In *NeurIPS*, pages 6639–6649, 2018.
* Schwartz et al. (2007)

  Lisa M Schwartz, Steven Woloshin, and H Gilbert Welch.
  The drug facts box: providing consumers with simple tabular data on drug benefit and harm.
  *Medical Decision Making*, 27(5):655–662, 2007.
* Shah et al. (2022)

  Deval Shah, Zi Yu Xue, and Tor M. Aamodt.
  Label encoding for regression networks.
  In *ICLR*, 2022.
* Sheskin (2003)

  David J Sheskin.
  *Handbook of parametric and nonparametric statistical procedures*.
  Chapman and hall/CRC, 2003.
* Simonyan and Zisserman (2015)

  Karen Simonyan and Andrew Zisserman.
  Very deep convolutional networks for large-scale image recognition.
  In *ICLR*, 2015.
* Somepalli et al. (2022)

  Gowthami Somepalli, Avi Schwarzschild, Micah Goldblum, C. Bayan Bruss, and Tom Goldstein.
  SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training.
  In *NeurIPS Workshop*, 2022.
* Song et al. (2019)

  Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang.
  Autoint: Automatic feature interaction learning via self-attentive neural networks.
  In *CIKM*, pages 1161–1170, 2019.
* Vaswani et al. (2017)

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  In *NIPS*, 2017.
* Wang et al. (2021)

  Ruoxi Wang, Rakesh Shivanna, Derek Zhiyuan Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi.
  DCN V2: improved deep & cross network and practical lessons for web-scale learning to rank systems.
  In *WWW*, pages 1785–1797, 2021.
* Weinberger et al. (2009)

  Kilian Q. Weinberger, Anirban Dasgupta, John Langford, Alexander J. Smola, and Josh Attenberg.
  Feature hashing for large scale multitask learning.
  In *ICML*, pages 1113–1120, 2009.
* Wu et al. (2024)

  Jing Wu, Suiyao Chen, Qi Zhao, Renat Sergazinov, Chen Li, Shengjie Liu, Chongchao Zhao, Tianpei Xie, Hanqing Guo, Cheng Ji, Daniel Cociorva, and Hakan Brunzell.
  Switchtab: Switched autoencoders are effective tabular learners.
  In *AAAI*, pages 15924–15933, 2024.
* Ye et al. (2023)

  Han-Jia Ye, Qi-Le Zhou, and De-Chuan Zhan.
  Training-free generalization on heterogeneous tabular data via meta-representation.
  *CoRR*, abs/2311.00055, 2023.
* Ye et al. (2024a)

  Han-Jia Ye, Huai-Hong Yin, and De-Chuan Zhan.
  Modern neighborhood components analysis: A deep tabular baseline two decades later.
  *CoRR*, abs/2407.03257, 2024a.
* Ye et al. (2024b)

  Hangting Ye, Wei Fan, Xiaozhuang Song, Shun Zheng, He Zhao, Dan dan Guo, and Yi Chang.
  Ptarl: Prototype-based tabular representation learning via space calibration.
  In *ICLR*, 2024b.
* Zhang et al. (2016)

  Weinan Zhang, Tianming Du, and Jun Wang.
  Deep learning over multi-field categorical data - - A case study on user response prediction.
  In *ECIR*, pages 45–57, 2016.
