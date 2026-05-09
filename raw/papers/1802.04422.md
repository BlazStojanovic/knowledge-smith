---
arxiv: '1802.04422'
authors:
- Sorelle A. Friedler
- Carlos Scheidegger
- Suresh Venkatasubramanian
- Sonam Choudhary
- Evan P. Hamilton
- Derek Roth
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: A comparative study of fairness-enhancing interventions in machine learning
url: http://arxiv.org/abs/1802.04422v1
year: 2018
---

[1802.04422] A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison














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



# A comparative study of fairness-enhancing interventions in machine learning ††thanks: This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: <https://github.com/algofairness/fairness-comparison>

Sorelle A. Friedler
  
 Haverford College
  
Carlos Scheidegger
<sorelle@cs.haverford.edu>
  
 University of Arizona
  
Suresh Venkatasubramanian
<cscheid@cscheid.net>
  
 University of Utah
  
Sonam Choudhary
<suresh@cs.utah.edu>
  
 University of Utah
  
Evan P. Hamilton
<sonam@cs.utah.edu>
  
 Haverford College
  
Derek Roth
<evanphamilton@gmail.com>
  
 Haverford College
<derek.roth17@gmail.com>

###### Abstract

Computers are increasingly used to make decisions that have
significant impact in people’s lives. Often, these predictions can
affect different population subgroups disproportionately. As a result,
the issue of *fairness* has received much recent interest, and a
number of fairness-enhanced classifiers and predictors have appeared in
the literature. This paper seeks to study the following questions: how
do these different techniques fundamentally compare to one another,
and what accounts for the differences? Specifically, we seek to bring
attention to many under-appreciated aspects of such fairness-enhancing
interventions. Concretely, we present the results of an open
benchmark we have developed that lets us compare a number of different
algorithms under a variety of fairness measures, and a large number of
existing datasets. We find that although different algorithms tend to
prefer specific formulations of fairness preservations, many of these
measures strongly correlate with one another. In addition, we find
that fairness-preserving algorithms tend to be sensitive to
fluctuations in dataset composition (simulated in our benchmark by
varying training-test splits), indicating that fairness interventions
might be more brittle than previously thought.

## 1 Introduction

As the use of machine learning to make decisions about people has increased, so has the drive to make fairness-aware machine learning algorithms. A considerable body of research over the past ten years has produced algorithms for accurate yet fair decisions, under varying definitions of fair, for goals such as non-discriminatory hiring, risk assessment for sentencing guidance, and loan allocation. And yet we have not yet seen extensive deployment of these algorithms in the pertinent domains. The primary obstacle appears to be our ability to compare methods effectively across different evaluation measures and different data sets with consistent data preprocessing and testing methodologies. Such comparisons would not just reveal “best-in-class” methods; they would also suggest which measures are robust and how different algorithms are sensitive to different kinds of preprocessing. As pointed out by Lehr and Ohm ([2017](#bib.bib26)), such considerations of the data processing *pipeline* are not just important for efficient implementation but also have legal ramifications for the resulting automated decision-making process.

In this paper, we present a test-bed to facilitate direct comparisons of *algorithms* with respect to *measures* on a variety of *datasets*. Our open-source framework allows for the easy addition of new methods, measures and data for the purpose of evaluation. We show how to use our test-bed for determining not only which specific algorithm has the best performance under a fairness or accuracy measure, but what types of algorithmic interventions tend to be the most effective. In addition to the impact of these algorithmic choices, we examine the impact of different preprocessing techniques and different measures for accuracy and fairness that have an important, and previously obscured, impact on the results of these algorithms. Our goal is to provide a comprehensive comparative analysis of existing approaches that is currently lacking in the literature.

### 1.1 Our results

In terms of the techniques, datasets, and measures we evaluate in this paper, we wish to highlight the following findings:

##### Dependence on preprocessing

Different algorithms tend to have slightly different requirements in terms of input: how are sensitive attributes encoded? Are multiple sensitive attributes supported? Does the algorithm directly support categorical attributes or are attribute transformations required? We find that these can have an impact in accuracy and fairness measures reported in the literature.

##### Clustering of measures

Even though there has been a proliferation of measures designed to highlight discrimination instances by machine learning algorithms, we find that a large number of these measures tend to strongly correlate with one another. As a result, techniques optimizing for one measure often performs well for a different measure (and similarly for poor performance).

##### Algorithms make significantly different tradeoffs

The specific mechanisms that different algorithms employ to increase fairness are quite varied, but surprisingly, the actual predictions made by these algorithms tend to vary significantly as well. As a result, no algorithm’s performance (as of the latest state of our benchmark) appears to dominate, either in accuracy or fairness measures.

##### Algorithms tend to be sensitive to variations in the input

We find surprising variability in fairness measures arising from variations in training-test splits; this appears to not have been previously mentioned in the literature.

## 2 Background

Fairness-aware machine learning algorithms seek to provide methods under which
the predicted outcome of a classifier operating on data about people is fair or
non-discriminatory for people based on their *protected class status* such
as race, sex, religion, etc., also known as a *sensitive attribute*.
Broadly, fairness-aware machine learning algorithms have been categorized as
those *preprocessing* techniques designed to modify the input data so that
the outcome of any machine learning algorithm applied to that data will be fair,
those *algorithm modification* techniques that modify an existing algorithm
or create a new one that will be fair under any inputs, and those
*postprocessing* techniques that take the output of any model and modify
that output to be fair Romei and
Ruggieri ([2013](#bib.bib30)). Many associated
metrics for measuring fairness in algorithms have also been explored. These are
detailed further in Section [6](#S6 "6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") and are also surveyed in
Žliobaitė ([2017](#bib.bib34)). This description of fairness-aware machine
learning methods is limited to batch-learning-based interventions. We do not consider interventions that focus on sequential or reinforcement learning such as Jabbari et al. ([2017](#bib.bib17)); Joseph
et al. ([2016a](#bib.bib19)); Joseph et al. ([2016b](#bib.bib18)); Ensign et al. ([2018a](#bib.bib10), [b](#bib.bib11))

##### Preprocessing algorithms

The motivation behind preprocessing algorithms is the idea that training data is the cause of the discrimination that a machine learning algorithm might learn, and so modifying it can keep a learning algorithm trained on it from discriminating. This could be because the training data itself captures historical discrimination or because there are more subtle patterns in the data, such as an under-representation of a minority group, that makes errors on that group both more likely and less costly under certain accuracy measures. One such algorithm that we will analyze in this paper is that of Feldman et al. ([2015](#bib.bib12)) that modifies each attribute so that the marginal distributions based on the subsets of that attribute with a given sensitive value are all equal; it does not modify the training labels. Additional preprocessing approaches include Calmon et al. ([2017](#bib.bib6)); Kamiran and
Calders. ([2012](#bib.bib21)).

##### Algorithm modifications

Modifications to specific learning algorithms, e.g., in the form of additional
constraints, have been by far the most common
approach. We study three such methods in this paper. Kamishima
et al. ([2012](#bib.bib23)) introduce a fairness focused
regularization term and apply it to a logistic regression
classifier. Zafar
et al. ([2017](#bib.bib36)) observe that standard fairness constraints
are nonconvex and hard to satisfy directly and introduce a convex relaxation for
purpose of optimization. Calders and
Verwer ([2010](#bib.bib5)) build separate models for
each value of a sensitive attribute and use the appropriate model for inputs
with the corresponding value of the attribute.

Another method that combines preprocessing and algorithm modification is the
work by Zemel
et al. ([2013](#bib.bib38)). Their approach is to learn a modified
representation of the data that is most effective at classification while still
being free of signals pertaining to the sensitive attribute.

##### Postprocessing techniques

A third approach to building fairness into algorithm design is by modifying the
results of a previously trained classifier to achieve the desired results on
different groups. Kamiran
et al. ([2010](#bib.bib22)) designed a strategy to
modify the labels of leaves in a decision tree after training in order to
satisfy fairness constraints. Recent work by Hardt
et al. ([2016](#bib.bib15)) and
Woodworth et al. ([2017](#bib.bib35)) explored the use of post-processing as a way to
ensure fairness with respect to error profiles (see Section [6](#S6 "6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison")
for more on this).

In this paper we focus on *group fairness* approaches that aim to ensure
non-discrimination across protected groups where the goal is to optimize metrics
such as disparate impact. Another line of thought, known as *individual
fairness*, is detailed in Dwork et al. ([2012](#bib.bib9)). In this work, we do not
study algorithms that seek to optimize individual fairness: our goal is to focus
on methods that explicitly deal with group-based discrimination and there are
(to the best of our knowledge) no actual codes that optimize for individual
fairness.

### 2.1 Related Work

Three prior efforts are relevant to our work. FairTest
Tramèr et al. ([2015](#bib.bib33))111<https://github.com/columbia/fairtest>
provides a general methodology to explore potential biases or feature
associations in a data set, as well as a way to identify regions of the input
space where an algorithm might incur unusually high
errors. THEMISGalhotra
et al. ([2017](#bib.bib13))222<https://github.com/LASER-UMASS/Themis>
takes a blackbox decision-making procedure and designs test cases automatically to explore
where the procedure might be exhibiting group-based or causal discrimination.
Fairness Measures
Zehlike et al. ([2017](#bib.bib37)) occupies a different point in the design space. Given a
particular algorithm that one wishes to evaluate, they provide a framework to
test the algorithm on a variety of datasets and fairness measures. This approach
on the one hand is more general than our framework, because it works with any
algorithm. On the other hand, it is less effective for a comparative evaluation
of different algorithms especially if they have different preprocessing and
training methods.

There are other software packages that audit black box software to determine the
influence of individual variables. We omit a detailed description of these
approaches as they are out of the scope of the investigation presented here. For
more information, the reader is referred to the excellent new survey on
explainability by Guidotti et al. ([2018](#bib.bib14)).

## 3 Benchmark Structure

![Refer to caption](/html/1802.04422/assets/figs/flowchart.png)


Figure 1: The stages of the fairness-aware benchmarking program: data input, preprocessing, benchmarking, and analysis. Intermediate files are saved at each stage of the pipeline to ensure reproducibility.

In order to provide a platform for clear comparison of results across fairness-aware machine learning algorithms, we separate each stage of the learning and analysis process (see Figure [1](#S3.F1 "Figure 1 ‣ 3 Benchmark Structure ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison")) and ensure that each algorithm is compared using the same dataset (including the same preprocessing), the same set of training / test splits, and all desired fairness and accuracy measures. Much previous work has combined the preprocessing for a specific dataset with the code for the fairness-aware algorithm, which makes comparisons with other algorithms and other datasets difficult. Similarly, algorithms have often been analyzed only under one or two measures. Here, we emphasize that we distinguish preprocessing, algorithms, and measures, and create a pipeline in which all algorithms are analyzed under a standard preprocessing of datasets and a large set of measures.

In order to encourage easy adoption of this codebase as a platform for future algorithmic analysis, each of these choices is modularized so that adding new datasets, measures, and/or algorithms to the pipeline is as easy as creating a new object. The pipeline will then ensure that all existing algorithms are evaluated under the new dataset and measure. More details and instructions for adding to the code base can be found at the repository.333<https://github.com/algofairness/fairness-comparison>

## 4 Data

We perform all experiments based on five real-world data sets that have been previously considered in the fairness-aware machine learning literature and preprocess each consistently depending on the needs of the algorithm.444All raw datasets, preprocessing code, and resulting processed datasets are available in the repository: <https://github.com/algofairness/fairness-comparison>. Preprocessing described here can be reproduced by running: python3 preprocess.py The real-world datasets come from some of the domains impacted by questions of fairness in machine learning: hiring and promotion, credit-worthiness and loans, and recidivism prediction.

##### Ricci

The Ricci dataset comes from the case of Ricci v. DeStefano Supreme
Court of the United States ([2009](#bib.bib32)), a case before the U.S. Supreme Court in which the question at issue was an exam given to determine if firefighters would receive a promotion. The dataset has 118 entries and five attributes, including the sensitive attribute Race. The original promotion decision was made by a threshold of achieving at least a score of 707070 on the combined exam outcome Miao ([2011](#bib.bib28)). The goal in a fair learning context is to predict this original promotion decision while achieving fairness with respect to the sensitive attribute, Race.

##### Adult Income

The Adult Income dataset Lichman ([2013](#bib.bib27)) contains information about individuals from the 1994 U.S. census. It is pre-split into a training and test set; we use only the training data and re-split it. There are 32,561 instances and 14 attributes, including sensitive attributes race and sex. 2,399 instances with missing data are removed during the preprocessing step. The prediction task is predicting whether an individual makes more or less than $50,000 per year.

##### German

The German Credit dataset Lichman ([2013](#bib.bib27)) contains 1,000 instances and 20 attributes describing individuals along with a classification of each individual as a good or bad credit risk. Sensitive attribute sex is not directly included in the data, but can be derived from the given information. Sensitive attribute age is included, and is discretized into values adult (age at least 25 years old) and youth based on an analysis by Kamiran and
Calders ([2009](#bib.bib20)) showing this discretization provided for the most discriminatory possibilities.

##### ProPublica recidivism

The ProPublica data includes data collected about the use of the COMPAS risk assessment tool in Broward County, Florida Angwin
et al. ([2016](#bib.bib3)). It includes information such as the number of juvenile felonies and the charge degree of the current arrest for 6,167 individuals, along with sensitive attributes race and sex. Data is preprocessed according to the filters given in the original analysis Angwin
et al. ([2016](#bib.bib3)). Each individual has a binary “recidivism” outcome, that is the prediction task, indicating whether they were rearrested within two years after the first arrest (the charge described in the data).

##### ProPublica violent recidivism

The violent recidivism version of the ProPublica data Angwin
et al. ([2016](#bib.bib3)) describes the same scenario as the recidivism data described above, but where the predicted outcome is a rearrest for a violent crime within two years. 4,010 individuals are included after preprocessing is applied, and the sensitive attributes are race and sex.

## 5 Preprocessing

Each algorithm we will analyze has certain requirements for the type of data it will operate over, and these necessitate different preprocessing techniques. However, in order to provide a consistent comparison across algorithms, it’s important that each algorithm receive the same input. We reconcile these needs by creating types of inputs that multiple algorithms can handle. Algorithms that handle the same input can then be easily compared to each other; algorithms can also be compared across different preprocessing strategies for the same dataset, though these results should be seen to be less definitive.

The first preprocessing step is to modify the input data according to any data-specific needs: removing features that should not be used for classification, removing or imputing any missing data, and potentially removing items or adding derived features. In order to allow the analysis of fairness based on multiple sensitive attributes (e.g., not just ensuring fairness based on race or sex alone, but based on both someone’s race and sex) we also add a combined sensitive attribute (e.g., attribute “race-sex” with values like “White-Woman”) to each dataset that contains multiple sensitive attributes. All algorithms will receive versions of the dataset with this same preprocessing applied.

While some algorithms are able to handle the datasets for training with only the described initial preprocessing (we’ll call this version of the processed data “original”), most algorithms considered here have additional constraints.555Since scikit-learn classifiers only handle numerical data, even for classifiers like decision trees where this is not inherently a requirement, some of the tested algorithms that would otherwise handle the original data require numerical data since the algorithms call scikit-learn. For algorithms that can only handle numerical training data as input, we modify the data to include one-hot encoded versions of each categorical variable. Some algorithms additionally require that the sensitive attributes be binary (e.g., “White” and “not White” instead of handling multiple racial categorizations) - for this version of the data (numerical+binary) we modify the given privileged group to be 111 and all other values to be 00.

### 5.1 Analysis

With these four preprocessed versions of each data set in place, we can compare how a single algorithm performs relative to all versions of the dataset on which it can run. The most common form of input for the algorithms we consider here is numerical, and all these algorithms can additionally handle the numerical+binary version of the dataset. This gives an opportunity to determine the effect, per algorithm and per dataset, of allowing an algorithm access to full information about sensitive attribute categorization or only a binary summary.

![Refer to caption](/html/1802.04422/assets/figs/analysis/preprocessing-tradeoff-accuracy.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/preprocessing-tradeoff-DIbinary.png)

Figure 2:  Examining the results of the Feldman et al. Feldman et al. ([2015](#bib.bib12)) algorithm under different preprocessing choices: numerical versus numerical+binary. Each dot plots the result of a single split of the data in terms of the labeled metric under both preprocessing choices. The gray line shows equality between the preprocessing choices. The model used within the Feldman algorithm is listed, and some variants of the algorithm had the tradeoff parameter optimized for either accuracy or disparate impact value.

Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Analysis ‣ 5 Preprocessing ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") illustrates this analysis on the impact of the numerical+ binary version of the preprocessed data on the Feldman et al. Feldman et al. ([2015](#bib.bib12)) algorithm. In the left figure we examine the relation between the accuracy on numerical preprocessing versus numerical+binary binary-encoded sensitive attributes. Each algorithm was run over ten random 23:13:2313\frac{2}{3}:\frac{1}{3} splits and the result on each split is shown as a single point on the figure. As discussed in Section [7](#S7 "7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison"), [Feldman et al.](#bib.bib12) use a generic classifier after running a preprocessing “fairness-enhancing” filter on the data, and the different algorithms reflect the different classifier used. We also automate the parameter tuning for λ𝜆\lambda, the fairness-accuracy tradeoff parameter for this algorithm (more about parameter tuning specifics can be found in Section [7](#S7 "7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison")), for both accuracy and the disparate impact value. As we can see, for most variants of the algorithm the resulting accuracy is independent of the representation, with a notable exception of the SVM variants (where the preprocessing is followed by training with an SVM). In all three SVM variants, the accuracy is consistently higher when using the numerical+binary representation than when using the numerical representation. We speculate that this is because the Feldman et al. algorithm conditions on the sensitive value in its preprocessing on the data, and this step likely preserves more accuracy when a larger number of people are in each sensitive group – as is the case when the unprivileged groups are grouped together in the binary preprocessing variant. This may be compounded by the SVM model because when categorical features are one-hot encoded for input (as required by scikit-learn) the increase in the dimensionality of the data may cause the SVM to be less effective at finding a good classifier.

We can do a similar analysis on the fairness achieved by the methods, as seen in the right side of Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Analysis ‣ 5 Preprocessing ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison"). Again, we compare the fairness measure (in this case DI – see Section [6](#S6 "6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison")) achieved for different data representations. First, we see that the fairness achieved varies across runs, an issue we will return to when we discus measure stability. Second, we notice that there is less difference between the results obtained for different representations (although SVMs still show sensitivity to the representation). In other words, for this algorithm the accuracy is affected by the choice of classifier and representation, but not the fairness achieved.

## 6 Measures

There are many ways to evaluate the accuracy and fairness of a model. Rather than be exhaustive,666An upcoming tutorial puts the number of fairness measures at 21 Narayanan ([2018](#bib.bib29))! we will focus on representative measures for each aspect. Let D=(𝕏,S,Y)𝐷𝕏𝑆𝑌D=(\mathbb{X},S,Y) be a dataset where 𝕏𝕏\mathbb{X} is the data subset that can be used for training (whether categorical or numerical), S𝑆S is the sensitive attribute where 111 is the privileged class, and Y𝑌Y is the binary classification label where 111 is the positive outcome and 00 is the negative outcome. Let Y^^𝑌\hat{Y} be the predicted outcomes of some algorithm. We can define accuracy and fairness measures in terms of conditional probabilities of outcome variables (Y,Y^

𝑌^𝑌Y,\hat{Y}) with respect to variables like Y^^𝑌\hat{Y} and S𝑆S.

### 6.1 Accuracy measures

We consider the standard accuracy measures: the (uniform) accuracy (P​[Y^=y|Y=y]𝑃delimited-[]^𝑌conditional𝑦𝑌𝑦P[\hat{Y}=y~{}|~{}Y=y]), the true positive rate (TPR) (P​[Y^=1|Y=1]𝑃delimited-[]^𝑌conditional1𝑌1P[\hat{Y}=1~{}|~{}Y=1]) (also called the positive predictive value (PPV)), and the true negative rate (TNR) (P​[Y^=0|Y=0]𝑃delimited-[]^𝑌conditional0𝑌0P[\hat{Y}=0~{}|~{}Y=0]) (also called the negative predictive value (NPV)). We also consider the balanced classification rate (BCR), a version of accuracy that is unweighted per class:

###### Definition 1 (BCR).

|  |  |  |
| --- | --- | --- |
|  | P​[Y^=1|Y=1]+P​[Y^=0|Y=0]2𝑃delimited-[]^𝑌conditional1𝑌1𝑃delimited-[]^𝑌conditional0𝑌02\frac{P[\hat{Y}=1~{}|~{}Y=1]+P[\hat{Y}=0~{}|~{}Y=0]}{2} |  |

All of these measures lie in the range [0,1]01[0,1].

### 6.2 Fairness measures

Fairness measures can be divided into three broad categories, in all cases conditioned on values of the sensitive attribute S𝑆S. In what follows, we normalize measures to make comparisons easier. In all cases, the measures lie in the range [0,∞)0[0,\infty) or [0,2]02[0,2] where in both cases perfect fairness is achieved at 111. We note that some of these measures have appeared in the literature not as something to be optimized (to be close to 111) but as a constraint to be satisfied (i.e for example that the appropriate value must equal 111).

#### 6.2.1 Measures based on base rates

###### Definition 2 (Disparate Impact (DI) Feldman et al. ([2015](#bib.bib12)); Zafar et al. ([2017](#bib.bib36))).

|  |  |  |
| --- | --- | --- |
|  | P​[Y^=1|S≠1]P​[Y^=1|S=1]𝑃delimited-[]^𝑌conditional1𝑆1𝑃delimited-[]^𝑌conditional1𝑆1\frac{P[\hat{Y}=1~{}|~{}S\not=1]}{P[\hat{Y}=1~{}|~{}S=1]} |  |

This measure is inspired by one of the two tests for disparate impact in the legal literature in the United StatesBarocas and
Selbst ([2016](#bib.bib4)).
In the cases where there are more than two values for a given sensitive attribute, we consider two variants of DI (which are equivalent in the case when there are only two sensitive values): binary and average. In the binary case, all unprivileged classes are grouped together into a single value S≠1𝑆1S\neq 1 (e.g., ”non White”) that is compared as a group to the privileged class S=1𝑆1S=1 (e.g., ”White”). In the average case, pairwise DI calculations are done against the privileged class (e.g., ”White” compared to ”Black”, ”White” compared to ”Asian”, etc.) and the average of these calculations is taken. This is analogous to the one-vs-all and all-vs-all methodology in multi-class classification.

###### Definition 3 (CV Calders and Verwer ([2010](#bib.bib5))).

|  |  |  |
| --- | --- | --- |
|  | 1−(P​[Y^=1|S=1]−P​[Y^=1|S≠1])1𝑃delimited-[]^𝑌conditional1𝑆1𝑃delimited-[]^𝑌conditional1𝑆11-\left(P[\hat{Y}=1~{}|~{}S=1]-P[\hat{Y}=1~{}|~{}S\not=1]\right) |  |

This measure is the same as DI, but where the difference is taken instead of the ratio; such a measure has been used for example to measure gender discrimination in the United Kingdom. A binary grouping strategy (described above for DI) is used in the case where there is more than one sensitive value, and the averaging method can also be used. Note that we do not take the absolute value of the difference so that skew in favor of one group versus another can be detected. We note that requiring C​V=1𝐶𝑉1CV=1 is called the *demographic parity* constraint in the literature.

#### 6.2.2 Measures based on group-conditioned accuracy

In general, we can think of fairness measures based on group-conditioned accuracy as asking whether the error rates for each group are similar. This yields the following definitions.

###### Definition 4.

(*Group-conditioned fairness measures.*)

s𝑠s-Accuracy.
:   |  |  |  |
    | --- | --- | --- |
    |  | P[Y^=y∣Y=y,S=s]P[\hat{Y}=y\mid Y=y,S=s] |  |

s𝑠s-TPR.
:   |  |  |  |
    | --- | --- | --- |
    |  | P[Y^=1∣Y=1,S=s]P[\hat{Y}=1\mid Y=1,S=s] |  |

s𝑠s-TNR.
:   |  |  |  |
    | --- | --- | --- |
    |  | P[Y^=0∣Y=0,S=s]P[\hat{Y}=0\mid Y=0,S=s] |  |

s𝑠s-BCR.
:   |  |  |  |
    | --- | --- | --- |
    |  | P[Y^=1∣Y=1,S=s]+P[Y^=0∣Y=0,S=s]2\frac{P[\hat{Y}=1\mid Y=1,S=s]+P[\hat{Y}=0\mid Y=0,S=s]}{2} |  |

We note that these measures have been studied under different names. For example, error rate balance Chouldechova ([2017](#bib.bib7)) is the aim of achieving equal 1−limit-from11- s𝑠s-TPR and 1−limit-from11- s𝑠s-TNR values across sensitive groups. Equalized odds Hardt
et al. ([2016](#bib.bib15)) is the aim of achieving equal s𝑠s-TPR and 1−limit-from11- s𝑠s-TNR (the *false positive rate*) across sensitive groups.

Letting any of the above measures be denoted f​(Y,Y^,s)𝑓𝑌^𝑌𝑠f(Y,\hat{Y},s), the values can then be aggregated for comparison by taking the mean directly ∑s∈Sf​(Y,Y^,s)/|S|subscript𝑠𝑆𝑓𝑌^𝑌𝑠𝑆\sum\_{s\in S}f(Y,\hat{Y},s)/|S| or by taking the mean over comparisons analogous to DI and CV:
f​(Y,Y^,s)/f​(Y,Y^,1)𝑓𝑌^𝑌𝑠𝑓𝑌^𝑌1f(Y,\hat{Y},s)/f(Y,\hat{Y},1) or 1−(f​(Y,Y^,1)−f​(Y,Y^,s))1𝑓𝑌^𝑌1𝑓𝑌^𝑌𝑠1-(f(Y,\hat{Y},1)-f(Y,\hat{Y},s)). In each of these cases, as we saw above, the unprivileged sensitive values could be grouped together or handled separately in the ratio or difference.

#### 6.2.3 Measures based on group-conditioned calibration

A predictor that outputs a probability Y^^𝑌\hat{Y} for an event is said to be *well-calibrated* if P​[Y=1∣Y^=p]=p𝑃delimited-[]𝑌conditional1^𝑌𝑝𝑝P[Y=1\mid\hat{Y}=p]=p. Motivated by this, we can define fairness measures by group conditioning the calibration function.

###### Definition 5 (s𝑠s-Calibration+).

|  |  |  |
| --- | --- | --- |
|  | P[Y=1∣Y^=1,S=s]P[Y=1\mid\hat{Y}=1,S=s] |  |

###### Definition 6 (s𝑠s-Calibration-).

|  |  |  |
| --- | --- | --- |
|  | P[Y=1∣Y^=0,S=s]P[Y=1\mid\hat{Y}=0,S=s] |  |

Calibration has been introduced previously with the goal of equalizing across sensitive value Chouldechova ([2017](#bib.bib7)); Kleinberg et al. ([2017](#bib.bib25)).

### 6.3 Analysis

Although there are many variations on these and other measures, we find that many of these are correlated. In some cases, this is not surprising as these measures are definitionally related. For example, DI takes the ratio of two probabilities while CV takes the difference. However, by analyzing resulting measures across many algorithms, we can find correlations that are less obvious. In fact, it appears that there are two main groups of measures, all correlated with each other! In Figure [3](#S6.F3 "Figure 3 ‣ 6.3 Analysis ‣ 6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") we fix two dataset-algorithm pairs and look at how the different measures of fairness correlate with each other. A first surprising observation is that the various group-conditioned fairness measures are very closely related to each other (the base-rate measures like DI and CV are also closely related for the reason mentioned above). This suggests that we need not focus on the specific group-conditioned fairness measure we use. An unusual exception to this is the group-conditional calibration measure on negative outcomes (s-Calibration-) which is much more closely associated with the base-rate measures than other group-conditioned measures. A second surprising observation is that the accuracy measures are correlated with the group-conditioned fairness measures. This suggests that the discussions of fairness-accuracy tradeoffs are more pertinent with respect to base-rate fairness measures.

![Refer to caption](/html/1802.04422/assets/figs/analysis/measure-correlation.png)


Figure 3: Examining the relationships between different measures of fairness. Each figure represents one data set-algorithm pair. For each entry, the algorithm is run for 10 training-testing splits for different parameter choices. The Stahel-Donoho estimatorStahel ([1981](#bib.bib31)); Donoho ([1982](#bib.bib8)) is then computed for each set of pairs of measurements.

Additionally, there are cases in which we would expect there to be tradeoffs between measures. Recent impossibility results show that, assuming unequal base rates across populations, it is impossible to achieve both calibration and error rate balance (both the same false positive rate and the same false negative rates across groups) Chouldechova ([2017](#bib.bib7)); Kleinberg et al. ([2017](#bib.bib25)). In Figure [4](#S6.F4 "Figure 4 ‣ 6.3 Analysis ‣ 6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") we empirically examine this tradeoff. As before, each colored point represents one instance of train-test split for an algorithm. As Figure [4](#S6.F4 "Figure 4 ‣ 6.3 Analysis ‣ 6 Measures ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") shows, there is a clear tradeoff between with s-calibration- versus s-TPR for each algorithm. Interestingly, different algorithms situate themselves in different parts of the tradeoff line.

![Refer to caption](/html/1802.04422/assets/figs/analysis/adult_tradeoff/sex-TPR-sex-calibration-.png)


Figure 4: An illustration of the tradeoff between s-calibration- and TPR for all algorithms on the Adult dataset. Each dot represents one run out of 10 random train-test splits.

## 7 Algorithms

We choose a selection of existing fairness-aware algorithms to assess; these are chosen based on availability of source code and with the goal of choosing varying types of fairness interventions (e.g., preprocessing versus algorithm modification). Each algorithm is run on each dataset and each metric is calculated on the predicted results.777All algorithm implementations can be found in the repository (<https://github.com/algofairness/fairness-comparison>), along with all resulting metric calculations, (see the results/ directory). The full set of results can be reproduced by running: python3 benchmark.py Synthesis statistics (such as stability) are then calculated and comparison graphs are produced.888Algorithm analysis code can be found in the repository (<https://github.com/algofairness/fairness-comparison>) and can be reproduced by running: python3 analysis.py We analyze the following algorithms along with non-fairness-aware algorithms chosen for a baseline comparison: SVM, decision trees, Gaussian naive Bayes, and logistic regression.

##### Calders and Verwer ([2010](#bib.bib5))

[Calders and
Verwer](#bib.bib5) introduce a fairness-aware algorithm modification called Two Naive Bayes. Their approach trains separate models for the values and iteratively assesses the fairness of the combined model under the CV measure, makes small changes to the observed probabilities in the direction of reducing the measure, and retrains their two models. This algorithm can handle both categorical and numerical input data, but requires that the given sensitive attribute be binary. We use the Kamishima
et al. ([2012](#bib.bib23)) implementation of this algorithm.999<https://github.com/tkamishima/kamfadm/releases/tag/2012ecmlpkdd> The algorithm has a β𝛽\beta prior parameter, which we search from 00 to 111 in increments of 0.10.10.1.

##### Feldman et al. ([2015](#bib.bib12))

[Feldman et al.](#bib.bib12) give a preprocessing approach that modifies each attribute so that the marginal distributions based on the subsets of that attribute with a given sensitive value are all equal; it does not modify the training labels. Any algorithm can then be trained on the resulting “repaired” data. The algorithm can handle both categorical and numerical input data, but since we train scikit-learn classifiers based on this preprocessed data, our implementation can only handle numerical input. Both binary and non-binary sensitive attributes can be handled. A tuning parameter λ𝜆\lambda is provided to tradeoff between fairness and accuracy, where λ=0𝜆0\lambda=0 gives the fairness of a regular non-fairness aware classifier and λ=1𝜆1\lambda=1 maximizes fairness. λ=1𝜆1\lambda=1 is used as the default, and all values of λ𝜆\lambda at increments of 0.050.050.05 in [0,1]01[0,1] are included when the algorithm is optimized using a grid search over the parameters. The implementation comes from Feldman et al. Feldman et al. ([2015](#bib.bib12)) and Adler et al. ([2018](#bib.bib2)).101010<https://github.com/algofairness/BlackBoxAuditing>

##### Kamishima et al. ([2012](#bib.bib23))

[Kamishima
et al.](#bib.bib23) introduce a fairness-focused regularization term and apply it to a logistic regression classifier. Their approach requires numerical input and a binary sensitive attribute. A tuning parameter η𝜂\eta is provided to tradeoff between fairness and accuracy, where η=1𝜂1\eta=1 is the default. When optimizing the parameter we use values between 00 and 300300300, with a finer grid used for the lower values of that range; these parameter choices are based on the experimental exploration of this parameter given in Kamishima
et al. ([2012](#bib.bib23)). We use the [Kamishima
et al.](#bib.bib23) implementation of this algorithm.111111<https://github.com/tkamishima/kamfadm/releases/tag/2012ecmlpkdd>

##### Zafar et al. ([2017](#bib.bib36))

[Zafar
et al.](#bib.bib36) re-express fairness constraints (which can be nonconvex) via a convex relaxation. This allows them to maximize accuracy subject to fairness and also maximize fairness subject to fairness constraints. They use two parameters: c𝑐c is a parameter that controls the degree of independence of the outcome and the sensitive attribute via a covariance calculation: setting c=0𝑐0c=0 forces complete independence (and therefore fairness). The second parameter γ𝛾\gamma fixes the degree of approximation they are willing to tolerate: the algorithm is only required to find an answer that is within a 1+γ1𝛾1+\gamma factor of the optimal solution. In their experiments they set γ=0.5𝛾0.5\gamma=0.5 and vary c𝑐c as a linear function of the corresponding covariance estimate for an unconstrained classifier. When optimizing, we use values between 0.0010.0010.001 and 111 in 10 logarithmic steps.

![Refer to caption](/html/1802.04422/assets/figs/analysis/all_data_overview/adult-edited.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/all_data_overview/german-edited.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/all_data_overview/ricci-edited.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/all_data_overview/propublica-r-edited.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/all_data_overview/propublica-vr-edited.png)

Figure 5: The performance of all algorithms on each dataset with the goal of removing discrimination on a specific attribute. From top to bottom, the algorithms and sensitive attributes considered are: Adult Income on race, German Credit on sex, Ricci on race, ProPublica recidivism on race, and ProPublica violent recidivism on race. Each point is the result of a single algorithm running on a single training / test split - each algorithm is shown for ten such splits.

In Figure [5](#S7.F5 "Figure 5 ‣ Zafar et al. (2017) ‣ 7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") we can see a basic summary of the performance of each algorithm considered on each data set. Since each algorithm focuses on creating a fair outcome with respect to a specific attribute in the data, we have chosen a single sensitive attribute to consider per dataset in these overall results. It is clear that there is no one “winner” - no algorithm that is both more fair and more accurate than the others on all datasets. It is also clear that there is tremendous variation even within a single algorithm over the random splits it receives. We examine this point in more detail next.

### 7.1 Stability

When analyzing algorithms, an additional question we are concerned with is that of *stability* - will the algorithm still perform well if the training data is slightly different? To assess this, we considered the standard deviation of each metric over 10 random splits. The results are shown in Figure [6](#S7.F6 "Figure 6 ‣ 7.1 Stability ‣ 7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") for the Adult Income data set for all algorithms when focusing on non-discrimination in terms of race (left) and sex (right) using numerical+binary preprocessing. These results give perhaps the clearest indication of the quality of an algorithm on a given data set. It is also easy to see that each algorithm occupies a slightly different place on the trade-off between fairness (measured here by DI when taken over binary sensitive attributes). For example, when focusing on non-discrimination in terms of sex, the Zafar et al. algorithm is potentially the best choice in terms of a balance between fairness and accuracy, but the large standard deviation over DI may make it a less desirable option.

![Refer to caption](/html/1802.04422/assets/figs/analysis/adult_race_sensitivity.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/adult_sex_sensitivity.png)

Figure 6: The stability of algorithms on the Adult dataset. Each algorithm is tested on ten random train / test splits and a rectangle centered on the mean and with a width and height equal to the standard deviation along that measure is plotted. On the left, the algorithms attempt to remove discrimination in terms of race, and on the right in terms of sex.

### 7.2 Parameters

Many of these fairness-aware learning algorithms provide a parameter to allow a manual trade-off between fairness and accuracy. We automate the search for this balance and present results for all algorithms optimizing accuracy or fairness. This provides an additional means of testing the algorithm, as well as the possibility for further optimization of the tradeoff between the two. In Figure [7](#S7.F7 "Figure 7 ‣ 7.2 Parameters ‣ 7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison") we show the different results based on parameter tuning for the Zafar
et al. ([2017](#bib.bib36)) algorithm on the Ricci dataset (left) and the Feldman et al. ([2015](#bib.bib12)) algorithm on the Adult Income dataset.
A clear tradeoff between fairness and accuracy in these algorithms can be seen; the parameters are appropriately allowing exploration of the possible solution space.

![Refer to caption](/html/1802.04422/assets/figs/analysis/ricci_race_params_zafar.png)

![Refer to caption](/html/1802.04422/assets/figs/analysis/adult_race_params_feldman.png)

Figure 7: The results of the Zafar
et al. ([2017](#bib.bib36)) algorithm on the Ricci dataset (left) and the Feldman et al. ([2015](#bib.bib12)) algorithm on the Adult Income dataset (right) when the provided parameter to tradeoff between fairness and accuracy is used. The parameter is varied and each split and each new parameter value is shown.

### 7.3 Multiple sensitive attributes

![Refer to caption](/html/1802.04422/assets/figs/analysis/multiple_sensitive.png)


Figure 8: Here, we show the behavior of four different algorithms when making predictions while accounting for different protected attributes (“repairing” race and sex, as well as a composite attribute). Different algorithms not only behave quite differently from one another, but their performance varies significantly depending on which specific attribute is being considered.

While there are still very few fairness-aware algorithms that can formally handle multiple sensitive attributes directly in the algorithm (Kearns
et al. ([2017](#bib.bib24)); Hébert-Johnson et al. ([2017](#bib.bib16))), all algorithms discussed can handle them if preprocessed as described earlier so that they are combined into a single sensitive attribute (e.g., race-sex). However, we might expect combining the attributes in this way to degrade performance under some metrics, especially in the case of algorithms that can only handle binary sensitive attributes, or when there are too many combinations for the size of the dataset to provide a large group of people with each new combined sensitive value. Looking at the Adult dataset when fairness-aware algorithms are run focusing on non-discrimination in terms of race, sex, and both, we find varying results for each of the algorithms in Figure [8](#S7.F8 "Figure 8 ‣ 7.3 Multiple sensitive attributes ‣ 7 Algorithms ‣ A comparative study of fairness-enhancing interventions in machine learning This work was partially supported by National Science Foundation under grants IIS-1633387, IIS-1513651, and IIS-1633724, as well as by a grant from the Ethics and Governance of AI Initiative. Source code, including instructions for adding your own algorithm or dataset, can be found at: https://github.com/algofairness/fairness-comparison"). Sex is especially predictive on the Adult Income data set, so the DI value for sex is low, even on these fairness-aware algorithms. Race generally receives a higher DI value from these algorithms. When correcting for both at once, all of the algorithms find that the DI value is somewhere in between that for race and that for sex, but the Zafar
et al. ([2017](#bib.bib36)) algorithm has a much larger variance over race and sex than over either individually.

## 8 Discussion

Besides providing a central point of access to existing
fairness-enhancing interventions and classification algorithms, our
benchmark also highlights a number of gaps in the current practice and
reporting of fairness issues in machine learning. We conclude with the
following recommendations for future contributions to the area:

##### Emphasize preprocessing requirements.

If there are multiple plausible ways
in which a dataset can be processed to generate training data for an
algorithm, provide performance metrics for more than one of the
possible choices. If algorithms are being compared to each other, ensure they are compared based on the same preprocessing.

##### Avoid proliferation of measures.

A new measure for fairness
should only be introduced if it behaves fundamentally differently from
existing metrics. Our study indicates that a combination of
class-sensitive error rates and either DI or CV is a good
minimal working set.

##### Account for training instability.

Showing the performance
of an algorithm in a single training-test split appears to be insufficient. We
recommend reporting algorithm success and stability based on a moderate number of randomized training-test splits.

## References

* (1)
* Adler et al. (2018)

  Philip Adler, Casey Falk,
  Sorelle A Friedler, Tionney Nix,
  Gabriel Rybeck, Carlos Scheidegger,
  Brandon Smith, and Suresh
  Venkatasubramanian. 2018.
  Auditing black-box models for indirect influence.
  *Knowledge and Information Systems*
  54, 1 (2018),
  95–122.
* Angwin
  et al. (2016)

  Julia Angwin, Jeff
  Larson, Surya Mattu, and Lauren
  Kirchner. 2016.
  Machine Bias.
  <https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing>.
  *ProPublica* (May 23,
  2016).
* Barocas and
  Selbst (2016)

  Solon Barocas and
  Andrew D Selbst. 2016.
  Big data’s disparate impact.
  *Cal. L. Rev.* 104
  (2016), 671.
* Calders and
  Verwer (2010)

  Toon Calders and Sicco
  Verwer. 2010.
  Three Naive Bayes Approaches for
  Discrimination-Free Classification.
  *Data Mining journal; special issue with
  selected papers from ECML/PKDD* (2010).
* Calmon et al. (2017)

  Flavio Calmon, Dennis
  Wei, Bhanukiran Vinzamuri, Karthikeyan
  Natesan Ramamurthy, and Kush R Varshney.
  2017.
  Optimized Pre-Processing for Discrimination
  Prevention.
  In *Advances in Neural Information
  Processing Systems 30*, I. Guyon,
  U. V. Luxburg, S. Bengio,
  H. Wallach, R. Fergus,
  S. Vishwanathan, and R. Garnett
  (Eds.). Curran Associates, Inc.,
  3995–4004.

  <http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention.pdf>
* Chouldechova (2017)

  Alexandra Chouldechova.
  2017.
  Fair prediction with disparate impact: A study of
  bias in recidivism prediction instruments.
  *Big data* 5,
  2 (2017), 153–163.
* Donoho (1982)

  David L Donoho.
  1982.
  *Breakdown properties of multivariate
  location estimators*.
  Technical Report. Technical
  report, Harvard University, Boston. URL http://www-stat. stanford. edu/~
  donoho/Reports/Oldies/BPMLE. pdf.
* Dwork et al. (2012)

  Cynthia Dwork, Moritz
  Hardt, Toniann Pitassi, Omer Reingold,
  and Richard Zemel. 2012.
  Fairness Through Awareness. In
  *Proc. of Innovations in Theoretical Computer
  Science*.
* Ensign et al. (2018a)

  Danielle Ensign,
  Sorelle A. Friedler, Scott Neville,
  Carlos Scheidegger, and Suresh
  Venkatasubramanian. 2018a.
  Decision Making with Limited Feedback: Error bounds
  for Recidivism Prediction and Predictive Policing.. In
  *Algorithic Learning Theory (ALT)*.

  <http://fatml.mysociety.org/media/documents/recidivism_prediction_and_predictive_policing.pdf>
* Ensign et al. (2018b)

  Danielle Ensign,
  Sorelle A. Friedler, Scott Neville,
  Carlos Scheidegger, and Suresh
  Venkatasubramanian. 2018b.
  Runaway Feedback Loops in Predictive Policing. In
  *1st Conference on Fairness, Accountability and
  Transparency in Computer Science (FAT\*)*.

  <https://arxiv.org/abs/1706.09847>
* Feldman et al. (2015)

  Michael Feldman,
  Sorelle A. Friedler, John Moeller,
  Carlos Scheidegger, and Suresh
  Venkatasubramanian. 2015.
  Certifying and removing disparate impact.
  *Proc. 21st ACM KDD* (2015),
  259–268.
* Galhotra
  et al. (2017)

  Sainyam Galhotra, Yuriy
  Brun, and Alexandra Meliou.
  2017.
  Fairness testing: testing software for
  discrimination. In *Proceedings of the 2017 11th
  Joint Meeting on Foundations of Software Engineering*. ACM,
  498–510.
* Guidotti et al. (2018)

  Riccardo Guidotti, Anna
  Monreale, Franco Turini, Dino Pedreschi,
  and Fosca Giannotti. 2018.
  A Survey Of Methods For Explaining Black Box
  Models.
  *arXiv preprint arXiv:1802.01933*
  (2018).
* Hardt
  et al. (2016)

  Moritz Hardt, Eric Price,
  Nati Srebro, et al.
  2016.
  Equality of opportunity in supervised learning. In
  *Advances in neural information processing
  systems*. 3315–3323.
* Hébert-Johnson et al. (2017)

  Úrsula Hébert-Johnson,
  Michael P. Kim, Omer Reingold, and
  Guy N. Rothblum. 2017.
  Calibration for the
  (Computationally-Identifiable) Masses.
  *arXiv:1711.08513 [cs, stat]*
  (Nov. 2017).

  <http://arxiv.org/abs/1711.08513>
  arXiv: 1711.08513.
* Jabbari et al. (2017)

  Shahin Jabbari, Matthew
  Joseph, Michael Kearns, Jamie
  Morgenstern, and Aaron Roth.
  2017.
  Fairness in Reinforcement Learning. In
  *PMLR*. 1617–1626.

  <http://proceedings.mlr.press/v70/jabbari17a.html>
* Joseph et al. (2016b)

  Matthew Joseph, Michael
  Kearns, Jamie Morgenstern, Seth Neel,
  and Aaron Roth. 2016b.
  Fair Algorithms for Infinite and Contextual
  Bandits.
  *arXiv:1610.09559 [cs]* (Oct.
  2016).

  <http://arxiv.org/abs/1610.09559>
  arXiv: 1610.09559.
* Joseph
  et al. (2016a)

  Matthew Joseph, Michael
  Kearns, Jamie H Morgenstern, and Aaron
  Roth. 2016a.
  Fairness in Learning: Classic and Contextual
  Bandits.
  In *Advances in Neural Information
  Processing Systems 29*, D. D. Lee,
  M. Sugiyama, U. V. Luxburg,
  I. Guyon, and R. Garnett (Eds.).
  Curran Associates, Inc., 325–333.

  <http://papers.nips.cc/paper/6355-fairness-in-learning-classic-and-contextual-bandits.pdf>
* Kamiran and
  Calders (2009)

  Faisal Kamiran and Toon
  Calders. 2009.
  Classifying without Discriminating. In
  *Proc. of the IEEE International Conference on
  Computer, Control and Communication*.
* Kamiran and
  Calders. (2012)

  F. Kamiran and T.
  Calders. 2012.
  Data preprocessing techniques for classification
  without discrimination.
  *Knowledge and Information Systems*
  33 (2012), 1–33.
* Kamiran
  et al. (2010)

  Faisal Kamiran, Toon
  Calders, and Mykola Pechenizkiy.
  2010.
  Discrimination aware decision tree learning. In
  *Data Mining (ICDM), 2010 IEEE 10th International
  Conference on*. IEEE, 869–874.
* Kamishima
  et al. (2012)

  Toshihiro Kamishima,
  Shotaro Akaho, Hideki Asoh, and
  Jun Sakuma. 2012.
  Fairness-aware Classifier with Prejudice Remover
  Regularizer.
  *Machine Learning and Knowledge Discovery in
  Databases* (2012), 35–50.
* Kearns
  et al. (2017)

  Michael Kearns, Seth
  Neel, Aaron Roth, and Zhiwei Steven
  Wu. 2017.
  Preventing Fairness Gerrymandering: Auditing and
  Learning for Subgroup Fairness.
  *arXiv preprint arXiv:1711.05144*
  (2017).
* Kleinberg et al. (2017)

  Jon Kleinberg, Sendhil
  Mullainathan, and Manish Raghavan.
  2017.
  Inherent trade-offs in the fair determination of
  risk scores. In *Proceedings of Innovations in
  Theoretical Computer Science (ITCS)*.
* Lehr and Ohm (2017)

  David Lehr and Paul
  Ohm. 2017.
  Playing with the Data: What Legal Scholars Should
  Learn About Machine Learning.
  *UC Davis Law Review* 51,
  2 (2017), 653–718.
* Lichman (2013)

  M. Lichman.
  2013.
  UCI Machine Learning Repository.
  (2013).

  <http://archive.ics.uci.edu/ml>
* Miao (2011)

  Weiwen Miao.
  2011.
  Did the Results of Promotion Exams Have a Disparate
  Impact on Minorities? Using Statistical Evidence in Ricci v. DeStefano.
  *J. of Stat. Ed.* 19,
  1 (2011).
* Narayanan (2018)

  Arvind Narayanan.
  2018.
  21 Fairness Definitions and Their Politics.
  (Feb. 23 2018).


  Tutorial presented at the Conference on Fairness, Accountability,
  and Transparency.
* Romei and
  Ruggieri (2013)

  Andrea Romei and
  Salvatore Ruggieri. 2013.
  A Multidisciplinary Survey on Discrimination
  Analysis.
  *The Knowledge Engineering Review*
  (April 3 2013), 1–57.
* Stahel (1981)

  Werner A Stahel.
  1981.
  *Breakdown of covariance estimators*.
  Fachgruppe für Statistik, Eidgenössische
  Techn. Hochsch.
* Supreme
  Court of the United States (2009)

  Supreme Court of the United States.
  2009.
  Ricci v. DeStefano.
  557 U.S. 557, 174. (2009),
  2658 pages.
* Tramèr et al. (2015)

  Florian Tramèr,
  Vaggelis Atlidakis, Roxana Geambasu,
  Daniel J. Hsu, Jean-Pierre Hubaux,
  Mathias Humbert, Ari Juels, and
  Huang Lin. 2015.
  Discovering Unwarranted Associations in Data-Driven
  Applications with the FairTest Testing Toolkit.
  *CoRR* abs/1510.02377
  (2015).
  arXiv:1510.02377
  <http://arxiv.org/abs/1510.02377>
* Žliobaitė (2017)

  Indrė Žliobaitė.
  2017.
  Measuring discrimination in algorithmic decision
  making.
  *Data Mining and Knowledge Discovery*
  31, 4 (July
  2017), 1060–1089.
* Woodworth et al. (2017)

  Blake Woodworth, Suriya
  Gunasekar, Mesrob I Ohannessian, and
  Nathan Srebro. 2017.
  Learning non-discriminatory predictors.
  *arXiv preprint arXiv:1702.06081*
  (2017).
* Zafar
  et al. (2017)

  Muhammad Bilal Zafar,
  Isabel Valera, Manuel Gomez Rogriguez,
  and Krishna P Gummadi. 2017.
  Fairness Constraints: Mechanisms for Fair
  Classification. In *Artificial Intelligence and
  Statistics*. 962–970.
* Zehlike et al. (2017)

  Meike Zehlike, Carlos
  Castillo, Francesco Bonchi, Ricardo
  Baeza-Yates, Sara Hajian, and Mohamed
  Megahed. 2017.
  FAIRNESS MEASURES: A Platform for Data Collection and
  Benchmarking in discrimination-aware ML.
  <http://fairness-measures.org>.
  (Jun 2017).

  <http://fairness-measures.org>
* Zemel
  et al. (2013)

  Rich Zemel, Yu Wu,
  Kevin Swersky, Toni Pitassi, and
  Cynthia Dwork. 2013.
  Learning Fair Representations. In
  *Proc. of Intl. Conf. on Machine Learning*.
  325–333.

[◄](/html/1802.04421)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/1802.04422)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+1802.04422)
[View original  
on arXiv](https://arxiv.org/abs/1802.04422)[►](/html/1802.04423)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 9 12:49:56 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
