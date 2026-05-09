---
arxiv: '2012.07421'
authors:
- Pang Wei Koh
- Shiori Sagawa
- Henrik Marklund
- Sang Michael Xie
- Marvin Zhang
- Akshay Balsubramani
- Weihua Hu
- Michihiro Yasunaga
- Richard Lanas Phillips
- Irena Gao
- Tony Lee
- Etienne David
- Ian Stavness
- Wei Guo
- Berton A. Earnshaw
- Imran S. Haque
- Sara Beery
- Jure Leskovec
- Anshul Kundaje
- Emma Pierson
- Sergey Levine
- Chelsea Finn
- Percy Liang
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'WILDS: A Benchmark of in-the-Wild Distribution Shifts'
url: http://arxiv.org/abs/2012.07421v3
year: 2020
---

[2012.07421] Wilds: A Benchmark of in-the-Wild Distribution Shifts














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



# Wilds: A Benchmark of in-the-Wild Distribution Shifts

Pang Wei Koh
These authors contributed equally to this work.
  
Proceedings of the
38t​hsuperscript38𝑡ℎ\mathit{38}^{th} International Conference on Machine Learning,
PMLR 139, 2021.
  
Copyright 2021 by the authors.
  
Shiori Sagawa††footnotemark: 
\email{pangwei, ssagawa}@cs.stanford.edu
\ANDHenrik Marklund \emailmarklund@stanford.edu
\ANDSang Michael Xie \emailxie@cs.stanford.edu
\ANDMarvin Zhang \emailmarvin@eecs.berkeley.edu
\ANDAkshay Balsubramani \emailabalsubr@stanford.edu
\ANDWeihua Hu \emailweihuahu@stanford.edu
\ANDMichihiro Yasunaga \emailmyasu@stanford.edu
\ANDRichard Lanas Phillips \emailrichard@cs.cornell.edu
\ANDIrena Gao \emailigao@stanford.edu
\ANDTony Lee \emailtonyhlee@stanford.edu
\ANDEtienne David \emailetienne.david@inrae.fr
\ANDIan Stavness \emailstavness@usask.ca
\ANDWei Guo \emailguowei@g.ecc.u-tokyo.ac.jp
\ANDBerton A. Earnshaw \emailberton.earnshaw@recursionpharma.com
\ANDImran S. Haque \emailimran.haque@recursionpharma.com
\ANDSara Beery \emailsbeery@caltech.edu
\ANDJure Leskovec \emailjure@cs.stanford.edu
\ANDAnshul Kundaje \emailakundaje@stanford.edu
\ANDEmma Pierson \emailepierson@microsoft.com
\ANDSergey Levine \emailsvlevine@eecs.berkeley.edu
\ANDChelsea Finn \emailcbfinn@cs.stanford.edu
\ANDPercy Liang \emailpliang@cs.stanford.edu
\AND

Correspondence to: wilds@cs.stanford.edu

###### Abstract

Distribution shifts—where the training distribution differs from the test distribution—can substantially degrade the accuracy of machine learning (ML) systems deployed in the wild.
Despite their ubiquity in the real-world deployments, these distribution shifts are under-represented in the datasets widely used in the ML community today.
To address this gap, we present Wilds, a curated benchmark of 10 datasets reflecting a diverse range of distribution shifts that naturally arise in real-world applications, such as shifts across hospitals for tumor identification; across camera traps for wildlife monitoring; and across time and location in satellite imaging and poverty mapping.
On each dataset, we show that standard training yields substantially lower out-of-distribution than in-distribution performance.
This gap remains even with models trained by existing methods for tackling distribution shifts,
underscoring the need for new methods for training models that are more robust to the types of distribution shifts that arise in practice.
To facilitate method development, we provide an open-source package that automates dataset loading, contains default model architectures and hyperparameters, and standardizes evaluations.
Code and leaderboards are available at <https://wilds.stanford.edu>.

###### Contents

1. [1 Introduction](#S1 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
2. [2 Existing ML benchmarks for distribution shifts](#S2 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
3. [3 Problem settings](#S3 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
4. [4 Wilds datasets](#S4 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
5. [5 Performance drops from distribution shifts](#S5 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
6. [6 Baseline algorithms for distribution shifts](#S6 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
7. [7 Empirical trends](#S7 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
8. [8 Distribution shifts in other application areas](#S8 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
9. [9 Guidelines for method developers](#S9 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
10. [10 Using the Wilds package](#S10 "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
11. [A Dataset realism](#S1a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
12. [B Prior work on ML benchmarks for distribution shifts](#S2a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
13. [C Potential extensions to other problem settings](#S3a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
14. [D Additional experimental details](#S4a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
15. [E Additional dataset details and results](#S5a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")
16. [F Datasets with distribution shifts that do not cause performance drops](#S6a "In Wilds: A Benchmark of in-the-Wild Distribution Shifts")

## 1 Introduction

Distribution shifts—where the training distribution differs from the test distribution—can significantly degrade the accuracy of machine learning (ML) systems deployed in the wild.
In this work, we consider two types of distribution shifts that are ubiquitous in real-world settings: domain generalization and subpopulation shift (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
In *domain generalization*, the training and test distributions comprise data from related but distinct domains. This problem arises naturally in many applications, as it is often infeasible to collect a training set that spans all domains of interest. For example, in medical applications, it is common to seek to train a model on patients from a few hospitals, and then deploy it more broadly to hospitals outside the training set (Zech et al., [2018](#bib.bib427)); and in wildlife monitoring, we might seek to train an animal recognition model on images from one set of camera traps and then deploy it to new camera traps (Beery et al., [2018](#bib.bib36)).
In *subpopulation shift*, we consider test distributions that are subpopulations of the training distribution, with the goal of doing well even on the worst-case subpopulation.
For example, it is well-documented that standard models often perform poorly on under-represented demographics (Buolamwini and Gebru, [2018](#bib.bib67); Koenecke et al., [2020](#bib.bib208)),
and so we might seek models that can perform well on all demographic subpopulations.

![Refer to caption](/html/2012.07421/assets/x1.png)


Figure 1: 
In each Wilds dataset, each data point (x,y,d)𝑥𝑦𝑑(x,y,d) is associated with a domain d𝑑d.
Each domain corresponds to a distribution Pdsubscript𝑃𝑑P\_{d} over data points which are similar in some way, e.g., molecules with the same scaffold, or satellite images from the same region.
We study two types of distribution shifts.
Top: In *domain generalization*, we train and test on disjoint sets of domains. The goal is to generalize to domains unseen during training, e.g., molecules with a new scaffold in OGB-MolPCBA (Hu et al., [2020b](#bib.bib182)).
Bottom: In *subpopulation shift*, the training and test domains overlap, but their relative proportions differ. We typically assess models by their worst performance over test domains, each of which correspond to a subpopulation of interest, e.g., different geographical regions in FMoW-wilds (Christie et al., [2018](#bib.bib83)).



![Refer to caption](/html/2012.07421/assets/x2.png)

Figure 2: 
The Wilds benchmark contains 10 datasets across a diverse set of application areas, data modalities, and dataset sizes. Each dataset comprises data from different domains, and the benchmark is set up to evaluate models on distribution shifts across these domains.

Despite their ubiquity in real-world deployments, these types of distribution shifts are under-represented in the datasets widely used in the ML community today (Geirhos et al., [2020](#bib.bib144)).
Most of these datasets were designed for the standard i.i.d. setting, with training and test sets from the same distribution,
and prior work on retrofitting them with distribution shifts has focused on shifts that are cleanly characterized but not always likely to arise in real-world deployments.
For instance, many recent papers have studied datasets with shifts induced by synthetic transformations, such as changing the color of MNIST digits (Arjovsky et al., [2019](#bib.bib16)), or by disparate data splits, such as generalizing from cartoons to photos (Li et al., [2017a](#bib.bib229)).
Datasets like these are important testbeds for systematic studies, but they do not generally reflect the kinds of shifts that are likely to arise in the wild.
To develop and evaluate methods for real-world shifts, we need to complement these datasets with benchmarks that capture shifts in the wild, as model robustness need not transfer across shifts: e.g., models can be robust to image corruptions but not to shifts across datasets (Taori et al., [2020](#bib.bib365); Djolonga et al., [2020](#bib.bib113)), and a method that improves robustness on a standard vision dataset can even consistently harm robustness on real-world satellite imagery datasets (Xie et al., [2020](#bib.bib416)).

In this paper, we present Wilds, a curated benchmark of 10 datasets with evaluation metrics and train/test splits representing a broad array of distribution shifts that ML models face in the wild (Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). With Wilds, we seek to complement existing benchmarks by focusing on datasets with realistic shifts across a diverse set of data modalities and applications:
animal species categorization (Beery et al., [2020a](#bib.bib37)),
tumor identification (Bandi et al., [2018](#bib.bib28)),
bioassay prediction (Wu et al., [2018](#bib.bib412); Hu et al., [2020b](#bib.bib182)),
genetic perturbation classification (Taylor et al., [2019](#bib.bib367)),
wheat head detection (David et al., [2020](#bib.bib104)),
text toxicity classification (Borkan et al., [2019b](#bib.bib55)),
land use classification (Christie et al., [2018](#bib.bib83)),
poverty mapping (Yeh et al., [2020](#bib.bib422)),
sentiment analysis (Ni et al., [2019](#bib.bib274)),
and code completion (Raychev et al., [2016](#bib.bib306); Lu et al., [2021](#bib.bib248)).
These datasets reflect natural distribution shifts arising from different cameras, hospitals, molecular scaffolds, experiments, demographics, countries, time periods, users, and codebases.

Wilds builds on extensive data-collection efforts by domain experts, who are often forced to grapple with distribution shifts to make progress in their applications.
To design Wilds, we worked with them to identify, select, and adapt datasets that fulfilled the following criteria:

1. 1.

   Distribution shifts with performance drops. The train/test splits reflect shifts that substantially degrade model performance,
   i.e., with a large gap between in-distribution and out-of-distribution performance.
2. 2.

   Real-world relevance. The training/test splits and evaluation metrics are motivated by real-world scenarios and chosen in conjunction with domain experts. In Appendix [A](#S1a "A Dataset realism ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we further discuss the framework we use to assess the realism of a dataset.
3. 3.

   Potential leverage. Distribution shift benchmarks must be non-trivial but also possible to solve, as models cannot be expected to generalize to arbitrary distribution shifts. We constructed each Wilds dataset to have training data from multiple domains, with domain annotations and other metadata available at training time. We hope that these can be used to learn robust models: e.g., for domain generalization, one could use these annotations to learn models that are invariant to domain-specific features (Sun and Saenko, [2016](#bib.bib356); Ganin et al., [2016](#bib.bib137)), while for subpopulation shift, one could learn models that perform uniformly well across each subpopulation (Hu et al., [2018](#bib.bib181); Sagawa et al., [2020a](#bib.bib327)).

We chose the Wilds datasets to collectively encompass a diverse set of tasks, data modalities, dataset sizes, and numbers of domains, so as to enable evaluation across a broad range of real-world distribution shifts.
In Section [8](#S8 "8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we further survey the distribution shifts that occur in other application areas—algorithmic fairness and policing, medicine and healthcare, genomics, natural language and speech processing, education, and robotics—and discuss examples of datasets from these areas that we considered but did not include in Wilds, as their distribution shifts did not cause an appreciable performance drop.

To make the Wilds datasets more accessible, we have substantially modified most of them to clarify the distribution shift, standardize the data splits, and preprocess the data for use in standard ML frameworks.
In Section [10](#S10 "10 Using the Wilds package ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we introduce our accompanying open-source Python package that fully automates data loading and evaluation. The package also includes default models appropriate for each dataset, allowing all of the baseline results reported in this paper to be easily replicated.
To track the state-of-the-art in training algorithms and model architectures that are robust to these distribution shifts, we are also hosting a public leaderboard; we discuss guidelines for developers in Section [9](#S9 "9 Guidelines for method developers ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Code, leaderboards, and updates are available at <https://wilds.stanford.edu>.

Datasets are significant catalysts for ML research.
Likewise, benchmarks that curate and standardize datasets—e.g., the GLUE and SuperGLUE benchmarks for language understanding (Wang et al., [2019a](#bib.bib390), [b](#bib.bib391)) and the Open Graph Benchmark for graph ML (Hu et al., [2020b](#bib.bib182))—can accelerate research by focusing community attention, easing development on multiple datasets, and enabling systematic comparisons between approaches.
In this spirit, we hope that Wilds will facilitate the development of ML methods and models that are robust to real-world distribution shifts and can therefore be deployed reliably in the wild.

## 2 Existing ML benchmarks for distribution shifts

Distribution shifts have been a longstanding problem in the ML research community (Hand, [2006](#bib.bib163); Quiñonero-Candela et al., [2009](#bib.bib304)). Earlier work studied shifts in datasets for tasks including
part-of-speech tagging (Marcus et al., [1993](#bib.bib257)),
sentiment analysis (Blitzer et al., [2007](#bib.bib48)),
land cover classification (Bruzzone and Marconcini, [2009](#bib.bib63)),
object recognition (Saenko et al., [2010](#bib.bib325)),
and flow cytometry (Blanchard et al., [2011](#bib.bib47)).
However, these datasets are not as widely used today, in part because they tend to be much smaller than modern datasets.

Instead, many recent papers have focused on object recognition datasets with shifts induced by synthetic transformations, such as
ImageNet-C (Hendrycks and Dietterich, [2019](#bib.bib172)), which corrupts images with noise;
the Backgrounds Challenge (Xiao et al., [2020](#bib.bib414)) and Waterbirds (Sagawa et al., [2020a](#bib.bib327)), which alter image backgrounds;
or Colored MNIST (Arjovsky et al., [2019](#bib.bib16)), which changes the colors of MNIST digits.
It is also common to use data splits or combinations of disparate datasets to induce shifts, such as generalizing to photos solely from cartoons and other stylized images in PACS (Li et al., [2017a](#bib.bib229)); generalizing to objects at different scales solely from a single scale in DeepFashion Remixed (Hendrycks et al., [2020b](#bib.bib175)); or using training and test sets with disjoint subclasses in BREEDS (Santurkar et al., [2020](#bib.bib331)) and similar datasets (Hendrycks and Dietterich, [2019](#bib.bib172)). While our treatment here is necessarily brief, we discuss other similar datasets in Appendix [B](#S2a "B Prior work on ML benchmarks for distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

These existing benchmarks are useful and important testbeds for method development.
As they typically target well-defined and isolated shifts,
they facilitate clean analysis and controlled experimentation,
e.g., studying the effect of backgrounds on image classification (Xiao et al., [2020](#bib.bib414)), or showing that training with added Gaussian blur improves performance on real-world blurry images (Hendrycks et al., [2020b](#bib.bib175)).
Moreover, by studying how off-the-shelf models trained on standard datasets like ImageNet perform on different test datasets, we can better understand the robustness of these widely-used models
(Geirhos et al., [2018b](#bib.bib143); Recht et al., [2019](#bib.bib308); Hendrycks and Dietterich, [2019](#bib.bib172); Taori et al., [2020](#bib.bib365); Djolonga et al., [2020](#bib.bib113); Hendrycks et al., [2020b](#bib.bib175)).

However, as we discussed in the introduction, robustness to these synthetic shifts need not transfer to the kinds of shifts that arise in real-world deployments (Taori et al., [2020](#bib.bib365); Djolonga et al., [2020](#bib.bib113); Xie et al., [2020](#bib.bib416)),
and it is thus challenging to develop and evaluate methods for training models that are robust to real-world shifts on these datasets alone.
With WILDS, we seek to complement existing benchmarks by curating datasets that reflect natural distribution shifts across a diverse set of data modalities and application.

## 3 Problem settings

Each Wilds dataset is associated with a type of domain shift: domain generalization, subpopulation shift, or a hybrid of both (Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
We focus on these types of distribution shifts because they collectively capture the structure of most of the shifts in the applications we studied; see Section [8](#S8 "8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more discussion.
In each setting, we can view the overall data distribution as a mixture of D𝐷D domains 𝒟={1,…,D}𝒟1…𝐷\mathcal{D}=\{1,\dots,D\}.
Each domain d∈𝒟𝑑𝒟d\in\mathcal{D} corresponds to a fixed data distribution Pdsubscript𝑃𝑑P\_{d} over (x,y,d)𝑥𝑦𝑑(x,y,d), where x𝑥x is the input, y𝑦y is the prediction target, and all points sampled from Pdsubscript𝑃𝑑P\_{d} have domain d𝑑d.
We encode the domain shift by assuming that the training distribution P𝗍𝗋𝖺𝗂𝗇=∑d∈𝒟qd𝗍𝗋𝖺𝗂𝗇​Pdsuperscript𝑃𝗍𝗋𝖺𝗂𝗇subscript𝑑𝒟superscriptsubscript𝑞𝑑𝗍𝗋𝖺𝗂𝗇subscript𝑃𝑑P^{\mathsf{train}}=\sum\_{d\in\mathcal{D}}q\_{d}^{\mathsf{train}}P\_{d} has mixture weights qd𝗍𝗋𝖺𝗂𝗇superscriptsubscript𝑞𝑑𝗍𝗋𝖺𝗂𝗇q\_{d}^{\mathsf{train}} for each domain d𝑑d,
while the test distribution P𝗍𝖾𝗌𝗍=∑d∈𝒟qd𝗍𝖾𝗌𝗍​Pdsuperscript𝑃𝗍𝖾𝗌𝗍subscript𝑑𝒟superscriptsubscript𝑞𝑑𝗍𝖾𝗌𝗍subscript𝑃𝑑P^{\mathsf{test}}=\sum\_{d\in\mathcal{D}}q\_{d}^{\mathsf{test}}P\_{d} is a different mixture of domains with weights qd𝗍𝖾𝗌𝗍superscriptsubscript𝑞𝑑𝗍𝖾𝗌𝗍q\_{d}^{\mathsf{test}}.
For convenience, we define the set of training domains as 𝒟𝗍𝗋𝖺𝗂𝗇={d∈𝒟∣qd𝗍𝗋𝖺𝗂𝗇>0}superscript𝒟𝗍𝗋𝖺𝗂𝗇conditional-set𝑑𝒟superscriptsubscript𝑞𝑑𝗍𝗋𝖺𝗂𝗇0\mathcal{D}^{\mathsf{train}}=\{d\in\mathcal{D}\mid q\_{d}^{\mathsf{train}}>0\}, and likewise, the set of test domains as 𝒟𝗍𝖾𝗌𝗍={d∈𝒟∣qd𝗍𝖾𝗌𝗍>0}superscript𝒟𝗍𝖾𝗌𝗍conditional-set𝑑𝒟superscriptsubscript𝑞𝑑𝗍𝖾𝗌𝗍0\mathcal{D}^{\mathsf{test}}=\{d\in\mathcal{D}\mid q\_{d}^{\mathsf{test}}>0\}.

At training time, the learning algorithm gets to see the domain annotations d𝑑d, i.e., the training set comprises points (x,y,d)∼P𝗍𝗋𝖺𝗂𝗇similar-to𝑥𝑦𝑑superscript𝑃𝗍𝗋𝖺𝗂𝗇(x,y,d)\sim P^{\mathsf{train}}.
At test time, the model gets either x𝑥x or (x,d)𝑥𝑑(x,d) drawn from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, depending on the application.

### 3.1 Domain generalization (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")-Top)

In domain generalization, we aim to generalize to test domains 𝒟𝗍𝖾𝗌𝗍superscript𝒟𝗍𝖾𝗌𝗍\mathcal{D}^{\mathsf{test}} that are disjoint from the training domains 𝒟𝗍𝗋𝖺𝗂𝗇superscript𝒟𝗍𝗋𝖺𝗂𝗇\mathcal{D}^{\mathsf{train}}, i.e., 𝒟𝗍𝗋𝖺𝗂𝗇∩𝒟𝗍𝖾𝗌𝗍=∅superscript𝒟𝗍𝗋𝖺𝗂𝗇superscript𝒟𝗍𝖾𝗌𝗍\mathcal{D}^{\mathsf{train}}\cap\mathcal{D}^{\mathsf{test}}=\emptyset.
To make this problem tractable, the training and test domains are typically similar to each other: e.g., in Camelyon17-wilds, we train on data from some hospitals and test on a different hospital, and in iWildCam2020-wilds, we train on data from some camera traps and test on different camera traps.
We typically seek to minimize the average error on the test distribution.

### 3.2 Subpopulation shift (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")-Bottom)

In subpopulation shift, we aim to perform well across a wide range of domains seen during training time.
Concretely, all test domains are seen at training, with 𝒟𝗍𝖾𝗌𝗍⊆𝒟𝗍𝗋𝖺𝗂𝗇superscript𝒟𝗍𝖾𝗌𝗍superscript𝒟𝗍𝗋𝖺𝗂𝗇\mathcal{D}^{\mathsf{test}}\subseteq\mathcal{D}^{\mathsf{train}}, but the proportions of the domains can change, with q𝗍𝖾𝗌𝗍≠q𝗍𝗋𝖺𝗂𝗇superscript𝑞𝗍𝖾𝗌𝗍superscript𝑞𝗍𝗋𝖺𝗂𝗇q^{\mathsf{test}}\neq q^{\mathsf{train}}.
We typically seek to minimize the maximum error over all test domains.
For example, in CivilComments-wilds, the domains d𝑑d represent particular demographics, some of which are a minority in the training set, and we seek high accuracy on each of these subpopulations without observing their demographic identity d𝑑d at test time.

### 3.3 Hybrid settings

The categories of domain generalization and subpopulation shift provide a general framework for thinking about domain shifts,
and the methods that have been developed for each setting have been quite different, as we will discuss in Section [6](#S6 "6 Baseline algorithms for distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
However, it is not always possible to cleanly define a problem as one or the other; for example, a test domain might be present in the training set but at a very low frequency.
In Wilds, we consider some hybrid settings that combine both domain generalization and subpopulation shift. For example, in FMoW-wilds, the inputs are satellite images and the domains correspond to the year and geographical region in which they were taken.
We simultaneously consider domain generalization across time (the training/test sets comprise images taken before/after a certain year) and subpopulation shift across regions (there are images from the same regions in the training and test sets, and we seek high performance across all regions).

## 4 Wilds datasets

We now briefly describe each Wilds dataset, as summarized in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
For each dataset, we consider a problem setting—domain generalization, subpopulation shift, or a hybrid—that we believe best reflects the real-world challenges in the corresponding application area; see Appendix [A](#S1a "A Dataset realism ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more discussion of these considerations.
To avoid confusion between our modified datasets and their original sources,
we append -wilds to the dataset names.
We provide more details and context on related distribution shifts for each dataset in Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### 4.1 Domain generalization datasets

#### 4.1.1 iWildCam2020-wilds: Species classification across different camera traps

![Refer to caption](/html/2012.07421/assets/x3.png)


Figure 3: 
The iWildCam2020-wilds dataset comprises photos of wildlife taken by a variety of camera traps. The goal is to learn models that generalize to photos from new camera traps that are not in the training set.
Each Wilds dataset contains both in-distribution (ID) and out-of-distribution (OOD) evaluation sets; for brevity, we omit the ID sets from the subsequent dataset figures.

Animal populations have declined 68% on average since 1970 (Grooten et al., [2020](#bib.bib156)).
To better understand and monitor wildlife biodiversity loss, ecologists commonly deploy camera traps—heat or motion-activated static cameras placed in the wild (Wearn and Glover-Kapfer, [2017](#bib.bib397))—and then use ML models to process the data collected (Weinstein, [2018](#bib.bib399); Norouzzadeh et al., [2019](#bib.bib277); Tabak et al., [2019](#bib.bib363); Beery et al., [2019](#bib.bib38); Ahumada et al., [2020](#bib.bib5)).
Typically, these models would be trained on photos from some existing camera traps and then used across new camera trap deployments.
However, across different camera traps, there is drastic variation in illumination, color, camera angle, background, vegetation, and relative animal frequencies,
which results in models generalizing poorly to new camera trap deployments (Beery et al., [2018](#bib.bib36)).

We study this shift on a variant of the iWildCam 2020 dataset (Beery et al., [2020a](#bib.bib37)), where the input x𝑥x is a photo from a camera trap, the label y𝑦y is one of 182 animal species, and the domain d𝑑d specifies the identity of the camera trap (Figure [3](#S4.F3 "Figure 3 ‣ 4.1.1 iWildCam2020-wilds: Species classification across different camera traps ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). The training and test sets comprise photos from disjoint sets of camera traps. As leverage, we include over 200 camera traps in the training set, capturing a wide range of variation. We evaluate models by their macro F1 scores, which emphasizes performance on rare species, as rare and endangered species are the most important to accurately monitor.
Appendix [E.1](#S5.SS1a "E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

![Refer to caption](/html/2012.07421/assets/x4.png)


Figure 4: 
The Camelyon17-wilds dataset comprises tissue patches from different hospitals. The goal is to accurately predict the presence of tumor tissue in patches taken from hospitals that are not in the training set.
In this figure, each column contains two patches, one of normal tissue and the other of tumor tissue, from the same slide.

#### 4.1.2 Camelyon17-wilds: Tumor identification across different hospitals

Models for medical applications are often trained on data from a small number of hospitals, but with the goal of being deployed more generally across other hospitals.
However, variations in data collection and processing can degrade model accuracy on data from new hospital deployments (Zech et al., [2018](#bib.bib427); AlBadawy et al., [2018](#bib.bib7)).
In histopathology applications—studying tissue slides under a microscope—this variation can arise from sources like differences in the patient population or in slide staining and image acquisition (Veta et al., [2016](#bib.bib387); Komura and Ishikawa, [2018](#bib.bib211); Tellez et al., [2019](#bib.bib370)).

We study this shift on a patch-based variant of the Camelyon17 dataset (Bandi et al., [2018](#bib.bib28)), where the input x𝑥x is a 96x96 patch of a whole-slide image of a lymph node section from a patient with potentially metastatic breast cancer, the label y𝑦y is whether the patch contains tumor, and the domain d𝑑d specifies which of 5 hospitals the patch was from (Figure [4](#S4.F4 "Figure 4 ‣ 4.1.1 iWildCam2020-wilds: Species classification across different camera traps ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). The training and test sets comprise class-balanced patches from separate hospitals, and we evaluate models by their average accuracy.
Prior work suggests that staining differences are the main source of variation between hospitals in similar datasets (Tellez et al., [2019](#bib.bib370)). As we have training data from multiple hospitals, a model could use that as leverage to learn to be robust to stain variation.
Appendix [E.2](#S5.SS2a "E.2 Camelyon17-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

#### 4.1.3 RxRx1-wilds: Genetic perturbation classification across experimental batches

![Refer to caption](/html/2012.07421/assets/x5.png)


Figure 5: 
The RxRx1-wilds dataset comprises images of cells that have been genetically
perturbed by siRNA (Tuschl, [2001](#bib.bib378)). The goal is to predict which
siRNA the cells have been treated with, where the images come from
experimental batches not in the training set. Here, we show sample
images from different batches for two of the 1,139 possible classes.

High-throughput screening techniques that can generate large amounts of data
are now common in many fields of biology,
including transcriptomics (Harrill et al., [2019](#bib.bib165)),
genomics (Echeverri and Perrimon, [2006](#bib.bib121); Zhou et al., [2014](#bib.bib434)), proteomics and
metabolomics (Taylor et al., [2021](#bib.bib368)), and drug
discovery (Broach et al., [1996](#bib.bib60); Macarron et al., [2011](#bib.bib252); Swinney and Anthony, [2011](#bib.bib361); Boutros et al., [2015](#bib.bib57)).
Such large volumes of data, however, need to be created in experimental batches, or groups of experiments executed at similar times under similar conditions.
Despite attempts to carefully control experimental variables such as
temperature, humidity, and reagent concentration, measurements from
these screens are confounded by technical artifacts that arise from differences
in the execution of each batch.
These *batch effects* make it difficult to draw conclusions from data across experimental batches (Leek et al., [2010](#bib.bib228); Parker and Leek, [2012](#bib.bib287); Soneson et al., [2014](#bib.bib349); Nygaard et al., [2016](#bib.bib278); Caicedo et al., [2017](#bib.bib70)).

We study the shift induced by batch effects on a variant of the RxRx1
dataset (Taylor et al., [2019](#bib.bib367)), where the input x𝑥x is a 3-channel image of
cells obtained by fluorescent microscopy (Bray et al., [2016](#bib.bib59)), the label y𝑦y
indicates which of the 1,139 genetic treatments (including no treatment) the
cells received, and the domain d𝑑d specifies the batch in which the imaging
experiment was run.
As summarized in Figure [5](#S4.F5 "Figure 5 ‣ 4.1.3 RxRx1-wilds: Genetic perturbation classification across experimental batches ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), the training and test sets consist of disjoint experimental batches.
As leverage, the training set has images from 33 different batches, with each batch containing one sample for every class.
We assess a model’s ability to normalize
batch effects while preserving biological signal by evaluating how well it can
classify images of treated cells in the out-of-distribution test set.
Appendix [E.3](#S5.SS3a "E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

#### 4.1.4 OGB-MolPCBA: Molecular property prediction across different scaffolds

Accurate prediction of the biochemical properties of small molecules can significantly accelerate drug discovery by reducing the need for expensive lab experiments (Shoichet, [2004](#bib.bib346); Hughes et al., [2011](#bib.bib184)).
However, the experimental data available for training such models is limited compared to the extremely diverse and combinatorially large universe of candidate molecules that we would want to make predictions on (Bohacek et al., [1996](#bib.bib52); Sterling and Irwin, [2015](#bib.bib352); Lyu et al., [2019](#bib.bib251); McCloskey et al., [2020](#bib.bib258)).
This means that models need to generalize to out-of-distribution molecules that are structurally different from those seen in the training set.

We study this shift on the OGB-MolPCBA dataset, which is directly adopted from the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)) and originally from MoleculeNet (Wu et al., [2018](#bib.bib412)). As summarized in Figure [6](#S4.F6 "Figure 6 ‣ 4.1.4 OGB-MolPCBA: Molecular property prediction across different scaffolds ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), it is a multi-label classification dataset, where the input x𝑥x is a molecular graph, the label y𝑦y is a 128-dimensional binary vector where each component corresponds to a biochemical assay result, and the domain d𝑑d specifies the scaffold (i.e., a cluster of molecules with similar structure). The training and test sets comprise molecules with disjoint scaffolds; for leverage, the training set has molecules from over 40,000 scaffolds. We evaluate models by averaging the Average Precision (AP) across each of the 128 assays.
Appendix [E.4](#S5.SS4a "E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

![Refer to caption](/html/2012.07421/assets/x6.png)


Figure 6: 
The OGB-MolPCBA dataset comprises molecules with many different structural scaffolds. The goal is to predict biochemical assay results in molecules with scaffolds that are not in the training set. Here, we show sample molecules from each scaffold, together with target labels: each molecule is associated with 128 binary labels and ‘?’ indicates that the label is not provided for the molecule.

#### 4.1.5 GlobalWheat-wilds: Wheat head detection across regions of the world

Models for automated, high-throughput plant phenotyping—measuring the physical characteristics of plants and crops, such as wheat head density and counts—are important tools for crop breeding (Thorp et al., [2018](#bib.bib372); Reynolds et al., [2020](#bib.bib313)) and agricultural field management (Shi et al., [2016](#bib.bib342)).
These models are typically trained on data collected in a limited number of regions, even for crops grown worldwide such as wheat (Madec et al., [2019](#bib.bib254); Xiong et al., [2019](#bib.bib417); Ubbens et al., [2020](#bib.bib381); Ayalew et al., [2020](#bib.bib22)).
However, there can be substantial variation between regions, due to differences in crop varieties, growing conditions, and data collection protocols.
Prior work on wheat head detection has shown that this variation can significantly degrade model performance on regions unseen during training (David et al., [2020](#bib.bib104)).

We study this shift in an expanded version of the Global Wheat Head Dataset (David et al., [2020](#bib.bib104), [2021](#bib.bib105)), a large set of wheat images collected from 12 countries around the world (Figure [7](#S4.F7 "Figure 7 ‣ 4.1.5 GlobalWheat-wilds: Wheat head detection across regions of the world ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
It is a detection dataset, where the input x𝑥x is a cropped overhead image of a wheat field, the label y𝑦y is the set of bounding boxes for each wheat head visible in the image, and the domain d𝑑d specifies an image acquisition session (i.e., a specific location, time, and sensor with which a set of images was collected).
The data split captures a shift in location, with training and test sets comprising images from disjoint countries.
As leverage, we include images from 18 acquisition sessions over 5 countries in the training set.
We evaluate model performance on unseen countries by measuring accuracy at a fixed Intersection over Union (IoU) threshold, and averaging across acquisition sessions to account for imbalances in the numbers of images in them.
Additional details are provided in Appendix [E.5](#S5.SS5 "E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

![Refer to caption](/html/2012.07421/assets/figures/wheat_dataset.png)


Figure 7: 
The GlobalWheat-wilds dataset consists of overhead images of wheat fields, annotated with bounding boxes of wheat heads. The goal is to detect and predict the bounding boxes of wheat heads, where images are from new acquisition sessions. A set of wheat images are collected in each acquisition session, each corresponding to a specific wheat field location, time, and sensor. While acquisition sessions vary along multiple axes, from the aforementioned factors to wheat growth stage to illumination conditions, the dataset split primarily captures a shift in location; test images are taken from countries unseen during training time. In this figure, we show images with bounding boxes from different acquisition sessions.

### 4.2 Subpopulation shift datasets

#### 4.2.1 CivilComments-wilds: Toxicity classification across demographic identities

Automatic review of user-generated text is an important tool for moderating the sheer volume of text written on the Internet.
We focus here on the task of detecting toxic comments.
Prior work has shown that toxicity classifiers can pick up on biases in the training data and spuriously associate toxicity with the mention of certain demographics (Park et al., [2018](#bib.bib286); Dixon et al., [2018](#bib.bib112)).
These types of spurious correlations can significantly degrade model performance on particular subpopulations (Sagawa et al., [2020a](#bib.bib327)).

We study this problem on a variant of the CivilComments dataset (Borkan et al., [2019b](#bib.bib55)), a large collection of comments on online articles taken from the Civil Comments platform (Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1 CivilComments-wilds: Toxicity classification across demographic identities ‣ 4.2 Subpopulation shift datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The input x𝑥x is a text comment, the label y𝑦y is whether the comment was rated as toxic, and the domain d𝑑d is a 8-dimensional binary vector where each component corresponds to whether the comment mentions one of the 8 demographic identities male, female, LGBTQ, Christian, Muslim, other religions, Black, and White.
The training and test sets comprise comments on disjoint articles, and we evaluate models by the lowest true positive/negative rate over each of these 8 demographic groups; these groups overlap with each other, deviating slightly from the standard subpopulation shift framework in Section [3](#S3 "3 Problem settings ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Models can use the provided domain annotations as leverage to learn to perform well over each demographic group.
Appendix [E.6](#S5.SS6 "E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

![Refer to caption](/html/2012.07421/assets/x7.png)


Figure 8: 
The CivilComments-wilds dataset involves classifying the toxicity of online comments. The goal is to learn models that avoid spuriously associating mentions of demographic identities (like male, female, etc.) with toxicity due to biases in the training data.

### 4.3 Hybrid datasets

#### 4.3.1 FMoW-wilds: Land use classification across different regions and years

ML models for satellite imagery can enable global-scale monitoring of sustainability and economic challenges, aiding policy and humanitarian efforts in applications such as deforestation tracking (Hansen et al., [2013](#bib.bib164)), population density mapping (Tiecke et al., [2017](#bib.bib374)), crop yield prediction (Wang et al., [2020b](#bib.bib395)), and other economic tracking applications (Katona et al., [2018](#bib.bib199)).
As satellite data constantly changes due to human activity and environmental processes, these models must be robust to distribution shifts over time.
Moreover, as there can be disparities in the data available between regions,
these models should ideally have uniformly high accuracies instead of only doing well on data-rich regions and countries.

We study this problem on a variant of the Functional Map of the World dataset (Christie et al., [2018](#bib.bib83)), where the input x𝑥x is an RGB satellite image, the label y𝑦y is one of 62 building or land use categories, and the domain d𝑑d represents the year the image was taken and its geographical region (Africa, the Americas, Oceania, Asia, or Europe) (Figure [9](#S4.F9 "Figure 9 ‣ 4.3.1 FMoW-wilds: Land use classification across different regions and years ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). The different regions have different numbers of examples, e.g., there are far fewer images from Africa than the Americas.
The training set comprises data from before 2013, while the test set comprises data from 2016 and after; years 2013 to 2015 are reserved for the validation set.
We evaluate models by their test accuracy on the worst geographical region, which combines both a domain generalization problem over time and a subpopulation shift problem over regions.
As we provide both time and region annotations, models can leverage the structure across both space and time to improve robustness.
Appendix [E.7](#S5.SS7 "E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

![Refer to caption](/html/2012.07421/assets/x8.png)


Figure 9: 
The FMoW-wilds dataset contains satellite images taken in different geographical regions and at different times. The goal is to generalize to satellite imagery taken in the future, which may be shifted due to infrastructure development across time, and to do equally well across geographic regions.

#### 4.3.2 PovertyMap-wilds: Poverty mapping across different countries

Global-scale poverty estimation is a specific remote sensing application which is essential for targeted humanitarian efforts in poor regions (Abelson et al., [2014](#bib.bib1); Espey et al., [2015](#bib.bib124)).
However, ground truth measurements of poverty are lacking for much of the developing world, as field surveys for collecting the ground truth are expensive (Blumenstock et al., [2015](#bib.bib51)).
This motivates the approach of training ML models on countries with ground truth labels and then deploying them on different countries where we have satellite data but no labels (Xie et al., [2016](#bib.bib415); Jean et al., [2016](#bib.bib187); Yeh et al., [2020](#bib.bib422)).

We study this shift through a variant of the poverty mapping dataset collected by Yeh et al. ([2020](#bib.bib422)), where the input x𝑥x is a multispectral satellite image, the output y𝑦y is a real-valued asset wealth index from surveys, and the domain d𝑑d represents the country the image was taken in and whether the image is of an urban or rural area (Figure [10](#S4.F10 "Figure 10 ‣ 4.3.2 PovertyMap-wilds: Poverty mapping across different countries ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). The training and test set comprise data from disjoint sets of countries, and we evaluate models by the correlation of their predictions with the ground truth. Specifically, we take the lower of the correlations over the urban and rural subpopulations, as prior work has shown that accurately predicting poverty within these subpopulations is especially challenging.
As poverty measures are highly correlated across space (Jean et al., [2018](#bib.bib188); Rolf et al., [2020](#bib.bib319)), methods can utilize the provided location coordinates, and the country and urban/rural annotations, to improve robustness.
Appendix [E.8](#S5.SS8 "E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.

![Refer to caption](/html/2012.07421/assets/x9.png)


Figure 10: The PovertyMap-wilds dataset contains satellite images taken in different countries. The goal is to predict asset wealth in countries that are not present in the training set, while being accurate in both urban and rural areas. There may be significant economic and cultural differences across country borders that contribute to the spatial distribution shift.

![Refer to caption](/html/2012.07421/assets/x10.png)


Figure 11: 
The Amazon-wilds dataset involves predicting star ratings from reviews of Amazon products. The goal is to do consistently well on new reviewers who are not in the training set.

#### 4.3.3 Amazon-wilds: Sentiment classification across different users

In many consumer-facing ML applications, models are trained on data collected on one set of users and then deployed across a wide range of potentially new users.
These models can perform well on average but poorly on some users (Tatman, [2017](#bib.bib366); Caldas et al., [2018](#bib.bib72); Li et al., [2019b](#bib.bib235); Koenecke et al., [2020](#bib.bib208)).
These large performance disparities across users are practical concerns in consumer-facing applications, and they can also indicate that models are exploiting biases or spurious correlations in the data (Badgeley et al., [2019](#bib.bib24); Geva et al., [2019](#bib.bib146)).

We study this issue on a variant of the Amazon review dataset (Ni et al., [2019](#bib.bib274)), where the input x𝑥x is the review text, the label y𝑦y is the corresponding 1-to-5 star rating, and the domain d𝑑d identifies the user who wrote the review (Figure [11](#S4.F11 "Figure 11 ‣ 4.3.2 PovertyMap-wilds: Poverty mapping across different countries ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). The training and test sets comprise reviews from disjoint sets of users; for leverage, the training set has reviews from 5,008 different users. As our goal is to train models with consistently high performance across users, we evaluate models by the 10th percentile of per-user accuracies.
Appendix [E.9](#S5.SS9 "E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides additional details and context.
We discuss other distribution shifts on this dataset (e.g., by category) in Appendix [F.4](#S6.SS4a "F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

#### 4.3.4 Py150-wilds: Code completion across different codebases

Code completion models—autocomplete tools used by programmers to suggest subsequent source code tokens, such as the names of API calls—are commonly used to reduce the effort of software development (Robbes and Lanza, [2008](#bib.bib317); Bruch et al., [2009](#bib.bib62); Nguyen and Nguyen, [2015](#bib.bib273); Proksch et al., [2015](#bib.bib300); Franks et al., [2015](#bib.bib132)).
These models are typically trained on data collected from existing codebases but then deployed more generally across other codebases, which may have different distributions of API usages (Nita and Notkin, [2010](#bib.bib275); Proksch et al., [2016](#bib.bib301); Allamanis and Brockschmidt, [2017](#bib.bib10)).
This shift across codebases can cause substantial performance drops in code completion models.
Moreover, prior studies of real-world usage of code completion models have noted that they can generalize poorly on some important subpopulations of tokens such as method names (Hellendoorn et al., [2019](#bib.bib170)).

We study a variant of the Py150 Dataset (Raychev et al., [2016](#bib.bib306); Lu et al., [2021](#bib.bib248)), where the goal is to predict the next token (e.g., "environ", "communicate" in Figure [12](#S4.F12 "Figure 12 ‣ 4.3.4 Py150-wilds: Code completion across different codebases ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")) given the context of previous tokens.
The input x𝑥x is a sequence of source code tokens, the label y𝑦y is the next token, and the domain d𝑑d specifies the repository that the source code belongs to.
The training and test sets comprise code from disjoint GitHub repositories.
As leverage, we include over 5,300 repositories in the training set, capturing a wide range of source code variation. We evaluate models by their accuracy on the subpopulation of class and method tokens.
Additional dataset and model details are provided in Appendix [E.10](#S5.SS10 "E.10 Py150-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

![Refer to caption](/html/2012.07421/assets/x11.png)


Figure 12: 
The Py150-wilds dataset comprises Python source code files taken from a variety of public repositories on GitHub. The task is code completion: predict token names given the context of previous tokens. We evaluate models on their accuracy on the subpopulation of API calls (i.e., method and class tokens), which are the most common code completion queries in real-world settings. Our goal is to learn code completion models that generalize to source code in new repositories that are not seen in the training set.

## 5 Performance drops from distribution shifts

For a dataset to be appropriate for Wilds, the distribution shift reflected in its official train/test split should cause significant performance drops in standard models.
How to measure the performance drop due to a distribution shift is a crucial but subtle question.
In this section, we discuss our approach and the results on each of the Wilds datasets.
To construct Wilds, we selected datasets with large performance drops;
in Section [8](#S8 "8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we discuss other datasets with real-world shifts that did not show large performance drops and were therefore not included in the benchmark.

Our general approach is to measure the difference between the out-of-distribution (OOD) and in-distribution (ID) performance of standard models trained via empirical risk minimization (ERM).
Concretely, we first measure the OOD performance using the official train/test splits described in Section [4](#S4 "4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We then construct an appropriate in-distribution (ID) setting to measure ID performance, typically by modifying the official train/test splits.
However, practical constraints often prevent us from constructing an ID setting in exactly the way we want, which makes the choice of appropriate ID setting for each dataset a case-by-case issue.

### 5.1 In-distribution performance should be measured on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, not P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}

Choosing an appropriate in-distribution (ID) setting is the crux of measuring how much a distribution shift affects performance.
But what distribution should “in-distribution” be taken with respect to?
Consider a distribution shift from a training distribution P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} to a test distribution P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
It is common to measure ID performance by taking a model trained on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and evaluating it on additional held-out data from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}.111For example, in domain generalization, we might train a model on the training domains and then report its ID performance on held-out examples from the same domains; and in subpopulation shift, we might report average performance on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} as the ID performance.
This is useful for checking if the model can generalize well on both the training and the shifted test distributions.
However, it fails to isolate the effect of the distribution shift since it does not control for the data distribution on which the model is evaluated: the ID setting evaluates on data from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}, whereas the OOD setting evaluates on data from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
As a result, the performance gap might also be due to other factors such as differences in the difficulty of fitting a model to P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} versus P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.

For illustration, consider the task of wheat head detection on GlobalWheat-wilds.
The shift from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} to P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, which contain images of wheat fields in Europe and North America respectively, involves changes in factors such as wheat genotype, illumination, and growing conditions.
These changes mean that the task can be more challenging in some regions than others:
for example, wheat is grown in higher densities in certain regions than others, and it is harder to detect wheat heads reliably when they are more densely packed together.
If, for example, the task is harder in the regions in P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, then we might see especially low performance on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} compared to P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}. However, this performance gap would overestimate the actual gap caused by the distribution shift, in the sense that performance on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} would still be lower even if we could train a model purely on data from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.

To isolate the gap caused by the distribution shift, it is therefore important to keep the evaluation data distribution fixed between the ID and OOD settings by evaluating on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} in the ID setting.
For example, we could measure ID performance by training on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} and evaluating on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} and then compare this with the standard OOD setting of training on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and evaluating on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
However, there is a practical drawback: we generally have much more data from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} rather than P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, and training and evaluating on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} would require us to have a substantial number of labeled examples from each test domain.
In contrast, the standard ID setting of training and evaluating on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} is typically much more feasible, and it is also more convenient as we can reuse the same model trained on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} for both ID and OOD evaluations.

In Wilds, we take the approach of measuring ID performance on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} whenever practically feasible, and we lean on standard ID evaluations on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} otherwise.
In either case, we generally provide held-out data from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} in order to track model performance on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}.

### 5.2 Types of in-distribution settings

To measure the performance drop on each Wilds dataset, we picked the most appropriate ID setting(s) that were feasible.
We now describe five specific ways of constructing ID settings and their pros and cons.
The first two ID settings (test-to-test and mixed-to-test) control for the evaluation distribution and thus isolate the performance drops due to distribution shifts, as discussed in Section [5.1](#S5.SS1 "5.1 In-distribution performance should be measured on 𝑃^𝗍𝖾𝗌𝗍, not 𝑃^𝗍𝗋𝖺𝗂𝗇 ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
However, these procedures require substantial training data from test domains, so in cases where such data is not practically available, we consider the other ID settings (train-to-train, average, and random split).
Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") describes dataset-specific rationales for the selected ID settings and additional details for each dataset.

Below, we denote the training and OOD test sets of the official Wilds splits as D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} and D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}}, sampled from distributions P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, respectively.

##### Test-to-test (train on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, test on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}).

To control for the evaluation distribution, we can hold the test set D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}} fixed and train on a separate but identically-distributed training set D𝗁𝖾𝗅𝖽𝗈𝗎𝗍𝗍𝖾𝗌𝗍subscriptsuperscript𝐷𝗍𝖾𝗌𝗍𝗁𝖾𝗅𝖽𝗈𝗎𝗍D^{\mathsf{test}}\_{\mathsf{heldout}} drawn from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
The ID performance reported in this setting is directly comparable to OOD performance, which is also evaluated on D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}}.
The main drawback is that for a fair comparison to the OOD setting, where we train a model on D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}}, we would require D𝗁𝖾𝗅𝖽𝗈𝗎𝗍𝗍𝖾𝗌𝗍subscriptsuperscript𝐷𝗍𝖾𝗌𝗍𝗁𝖾𝗅𝖽𝗈𝗎𝗍D^{\mathsf{test}}\_{\mathsf{heldout}} to match the size of D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}}.
This is not feasible in our datasets, as D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} typically comprises the bulk of the available data.
We therefore do not use the test-to-test comparison for any of the Wilds datasets and instead consider the more practical alternative below, which still controls for the evaluation data distribution.

##### Mixed-to-test (train on a mixture of P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, test on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}).

In the mixed-to-test setting, we train a model on a mixture of data from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} and then evaluate it only on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
This is a more practical version of the test-to-test setting, as it retains the advantage of controlling for the evaluation distribution, while mitigating the need for large amounts of labeled data from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}} to use for training.222In practice, we typically split up D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}} and use some of it for training by replacing examples in D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} (so that the size of the training set is similar to the OOD setting). This still requires D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}} to be large enough to support using a sufficient number of examples for training while also having enough examples left over for accurate evaluation.
We use the mixed-to-test comparison for the Wilds datasets wherever feasible, except when we expect the train-to-train comparison to give similar results as described in the below discussion on train-to-train setting (e.g., for iWildCam2020-wilds and Py150-wilds).

One downside is that compared to the test-to-test setting, the mixed-to-test setting might underestimate ID performance, since it trains a model that simultaneously fits both P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, instead of just focusing on P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}.
However, this is useful as a sanity check that we can learn a model that can simultaneously fit both P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} and P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}; if such a model were not possible to learn, then it suggests that the distribution shift in the dataset is intractable for the model family.

##### Train-to-train (train on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}, evaluate on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}).

In the train-to-train setting, we train a model on D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} and evaluate on a separate but identically-distributed test set D𝗁𝖾𝗅𝖽𝗈𝗎𝗍𝗍𝗋𝖺𝗂𝗇subscriptsuperscript𝐷𝗍𝗋𝖺𝗂𝗇𝗁𝖾𝗅𝖽𝗈𝗎𝗍D^{\mathsf{train}}\_{\mathsf{heldout}} drawn from P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}}.
As discussed in Section [5.1](#S5.SS1 "5.1 In-distribution performance should be measured on 𝑃^𝗍𝖾𝗌𝗍, not 𝑃^𝗍𝗋𝖺𝗂𝗇 ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), this is practical—it does not require large amounts of data from P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}, and we can reuse the model for OOD evaluation—but has the drawback of not controlling for the evaluation distribution.

This drawback is less of an issue when we expect D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} and D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}} to be of equal difficulty in the sense of Section [5.1](#S5.SS1 "5.1 In-distribution performance should be measured on 𝑃^𝗍𝖾𝗌𝗍, not 𝑃^𝗍𝗋𝖺𝗂𝗇 ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
This may be the case when the dataset has a relatively large number of training and test domains that are drawn from the same distribution, and they are thus roughly interchangeable.
For instance, in iWildCam2020-wilds and Py150-wilds, there are many available domains (camera traps and GitHub repositories, respectively) randomly split across D𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝗋𝖺𝗂𝗇D^{\mathsf{train}} and D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}}, so we use the train-to-train comparison for them.
For most of the other datasets, we also include train-to-train comparisons to track model performance on P𝗍𝗋𝖺𝗂𝗇superscript𝑃𝗍𝗋𝖺𝗂𝗇P^{\mathsf{train}} (i.e., the official splits typically also include a held-out D𝗁𝖾𝗅𝖽𝗈𝗎𝗍𝗍𝗋𝖺𝗂𝗇subscriptsuperscript𝐷𝗍𝗋𝖺𝗂𝗇𝗁𝖾𝗅𝖽𝗈𝗎𝗍D^{\mathsf{train}}\_{\mathsf{heldout}}; we report results on these in Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), but we complement them whenever feasible with other ID settings that better isolate the effect of the distribution shift.

##### Average (report average instead of worst-case performance).

In subpopulation shift datasets, we measure the OOD performance of a model by reporting the performance on the worst-case subpopulation, and we can measure ID performance by simply reporting the average performance.
This average comparison corresponds to a special case of the train-to-train setting,333In subpopulation shifts, the training distribution reflects the empirical make-up over the pre-defined subpopulations, whereas the test distribution of interest corresponds to the worst-case subpopulation.
so they share the same pros and cons.
In particular, the average comparison is much more practical than running a test-to-test comparison on each subpopulation, as it can be especially difficult to obtain sufficient training examples from minority subpopulations.
In Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we use this average comparison for the CivilComments-wilds and Amazon-wilds datasets, which both consider a large number of subpopulations that are individually quite small.

##### Random split (train and evaluate on an i.i.d. split).

Another standard approach to measuring ID performance is to shuffle all of the data in D𝗍𝗋𝖺𝗂𝗇∪D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝗋𝖺𝗂𝗇superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{train}}\cup D^{\mathsf{test}} into i.i.d. training, validation, and test splits, while keeping the size of the training set constant.
We use this in OGB-MolPCBA to be consistent with prior work from the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)).
As with the train-to-train comparison, the random split comparison is simple to implement and does not require large amounts of data from D𝗍𝖾𝗌𝗍superscript𝐷𝗍𝖾𝗌𝗍D^{\mathsf{test}}, but it does not control for the evaluation distribution.

### 5.3 Model selection

We used standard model architectures for each dataset: ResNet and DenseNet for images (He et al., [2016](#bib.bib167); Huang et al., [2017](#bib.bib183)), DistilBERT for text (Sanh et al., [2019](#bib.bib330)), a Graph Isomorphism Network (GIN) for graphs (Xu et al., [2018](#bib.bib418)), and Faster-RCNN (Ren et al., [2015](#bib.bib312)) for detection.
As our goal is high OOD performance,
we use a separate OOD validation set for early stopping and hyperparameter selection.444This means that while the ERM models do not make use of any additional metadata (e.g., domain annotations) during training, this metadata is still implicitly (but very mildly) used for model selection.
Relative to the training set, this OOD validation set reflects a distribution shift similar to, but distinct from, the test set.
For example, in iWildCam2020-wilds, the training, validation, and test sets each comprise photos from distinct sets of camera traps. We detail experimental protocol in Appendix [D](#S4a "D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and models and hyperparameters for each dataset in Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

For the ID comparisons, we use the same hyperparameters optimized on the OOD validation set, so our ID results are slightly lower than if we had optimized hyperparameters for ID performance (Appendix [D](#S4a "D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). In other words, the ID-OOD gaps in Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") are slightly underestimated.

### 5.4 Results

Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that for each dataset, OOD performance is consistently and substantially lower than the corresponding ID performance.
Moreover, on the datasets that allow for mixed-to-test ID comparisons,
we show that models trained on a mix of the ID and OOD distributions can simultaneously achieve high ID and OOD performance,
indicating that lower OOD performance is not due to the OOD test sets being intrinsically more difficult than the ID test sets.
Overall, these results demonstrate that the real-world distribution shifts reflected in the Wilds datasets meaningfully degrade standard model performance.
Additional results for datasets that admit multiple ID comparisons are described for each dataset in Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

Table 1: The in-distribution (ID) vs. out-of-distribution (OOD) performance of models trained with empirical risk minimization.
The OOD test sets are drawn from the shifted test distributions described in Section [4](#S4 "4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"),
while the ID comparisons vary per dataset and are described in Section [5.1](#S5.SS1 "5.1 In-distribution performance should be measured on 𝑃^𝗍𝖾𝗌𝗍, not 𝑃^𝗍𝗋𝖺𝗂𝗇 ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
For each dataset, higher numbers are better.
In all tables in this paper, we report in parentheses the standard deviation across 3+ replicates, which measures the variability between replicates; note that this is higher than the standard error of the mean, which measures the variability in the estimate of the mean across replicates.
All datasets show performance drops due to distribution shift, with substantially better ID performance than OOD performance.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | Metric | In-dist setting | In-dist | Out-of-dist | Gap |
| iWildCam2020-wilds | Macro F1 | Train-to-train | 47.0 (1.4) | 31.0 (1.3) | 16.0 |
| Camelyon17-wilds | Average acc | Train-to-train | 93.2 (5.2) | 70.3 (6.4) | 22.9 |
| RxRx1-wilds | Average acc | Mixed-to-test | 39.8 (0.2) | 29.9 (0.4) | 9.9 |
| OGB-MolPCBA | Average AP | Random split | 34.4 (0.9) | 27.2 (0.3) | 7.2 |
| GlobalWheat-wilds | Average domain acc | Mixed-to-test | 63.3 (1.7) | 49.6 (1.9) | 13.7 |
| CivilComments-wilds | Worst-group acc | Average | 92.2 (0.1) | 56.0 (3.6) | 36.2 |
| FMoW-wilds | Worst-region acc | Mixed-to-test | 48.6 (0.9) | 32.3 (1.3) | 16.3 |
| PovertyMap-wilds | Worst-U/R Pearson R | Mixed-to-test | 0.60 (0.06) | 0.45 (0.06) | 0.15 |
| Amazon-wilds | 10th percentile acc | Average | 71.9 (0.1) | 53.8 (0.8) | 18.1 |
| Py150-wilds | Method/class acc | Train-to-train | 75.4 (0.4) | 67.9 (0.1) | 7.5 |




Table 2: The out-of-distribution test performance of models trained with different baseline algorithms: CORAL, originally designed for unsupervised domain adaptation; IRM, for domain generalization; and Group DRO, for subpopulation shifts. Evaluation metrics for each dataset are the same as in Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"); higher is better.
Overall, these algorithms did not improve over empirical risk minimization (ERM), and sometimes made performance significantly worse, except on CivilComments-wilds where they perform better but still do not close the in-distribution gap in Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
For GlobalWheat-wilds, we omit CORAL and IRM as those methods do not port straightforwardly to detection settings; its ERM number also differs from Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") as its ID comparison required a slight change to the OOD test set.
Parentheses show standard deviation across 3+ replicates.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Dataset | Setting | ERM | CORAL | IRM | Group DRO |
| iWildCam2020-wilds | Domain gen. | 31.0 (1.3) | 32.8 (0.1) | 15.1 (4.9) | 23.9 (2.1) |
| Camelyon17-wilds | Domain gen. | 70.3 (6.4) | 59.5 (7.7) | 64.2 (8.1) | 68.4 (7.3) |
| RxRx1-wilds | Domain gen. | 29.9 (0.4) | 28.4 (0.3) | 8.2 (1.1) | 23.0 (0.3) |
| OGB-MolPCBA | Domain gen. | 27.2 (0.3) | 17.9 (0.5) | 15.6 (0.3) | 22.4 (0.6) |
| GlobalWheat-wilds | Domain gen. | 51.2 (1.8) | — | — | 47.9 (2.0) |
| CivilComments-wilds | Subpop. shift | 56.0 (3.6) | 65.6 (1.3) | 66.3 (2.1) | 70.0 (2.0) |
| FMoW-wilds | Hybrid | 32.3 (1.3) | 31.7 (1.2) | 30.0 (1.4) | 30.8 (0.8) |
| PovertyMap-wilds | Hybrid | 0.45 (0.06) | 0.44 (0.06) | 0.43 (0.07) | 0.39 (0.06) |
| Amazon-wilds | Hybrid | 53.8 (0.8) | 52.9 (0.8) | 52.4 (0.8) | 53.3 (0.0) |
| Py150-wilds | Hybrid | 67.9 (0.1) | 65.9 (0.1) | 64.3 (0.2) | 65.9 (0.1) |

## 6 Baseline algorithms for distribution shifts

Many algorithms have been proposed for training models that are more robust to particular distribution shifts than standard models trained by empirical risk minimization (ERM), which trains models to minimize the average training loss.
Unlike ERM, these algorithms tend to utilize domain annotations during training, with the goal of learning a model that can generalize across domains.
In this section, we evaluate several representative algorithms from prior work and show that the out-of-distribution performance drops shown in Section [5](#S5 "5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") still remain.

### 6.1 Domain generalization baselines

Methods for domain generalization typically involve adding a penalty to the ERM objective that encourages some form of invariance across domains.
We include two such methods as representatives:

* •

  CORAL (Sun and Saenko, [2016](#bib.bib356)), which penalizes differences in the means and covariances of the feature distributions (i.e., the distribution of last layer activations in a neural network) for each domain.
  Conceptually, CORAL is similar to other methods that encourage feature representations to have the same distribution across domains
  (Tzeng et al., [2014](#bib.bib380); Long et al., [2015](#bib.bib246); Ganin et al., [2016](#bib.bib137); Li et al., [2018c](#bib.bib237), [b](#bib.bib233)).
* •

  IRM (Arjovsky et al., [2019](#bib.bib16)), which penalizes feature distributions that have different optimal linear classifiers for each domain. This builds on earlier work on invariant predictors (Peters et al., [2016](#bib.bib293)).

Other techniques for domain generalization include
conditional variance regularization (Heinze-Deml and Meinshausen, [2017](#bib.bib169));
self-supervision (Carlucci et al., [2019](#bib.bib76));
and meta-learning-based approaches (Li et al., [2018a](#bib.bib230); Balaji et al., [2018](#bib.bib25); Dou et al., [2019](#bib.bib115)).

### 6.2 Subpopulation shift baselines

In subpopulation shift settings, our aim is to train models that perform well on all relevant subpopulations. We test the following approach:

* •

  Group DRO (Hu et al., [2018](#bib.bib181); Sagawa et al., [2020a](#bib.bib327)), which uses distributionally robust optimization to explicitly minimize the loss on the worst-case domain during training. Group DRO builds on the maximin approach developed in Meinshausen and Bühlmann ([2015](#bib.bib263)).

Other methods for subpopulation shifts include
reweighting methods based on class/domain frequencies (Shimodaira, [2000](#bib.bib343); Cui et al., [2019](#bib.bib100));
label-distribution-aware margin losses (Cao et al., [2019](#bib.bib75));
adaptive Lipschitz regularization (Cao et al., [2020](#bib.bib74));
slice-based learning (Chen et al., [2019b](#bib.bib81); Ré et al., [2019](#bib.bib307));
style transfer across domains (Goel et al., [2020](#bib.bib149));
or other DRO algorithms that do not make use of explicit domain information and
rely on, for example, unsupervised clustering (Oren et al., [2019](#bib.bib281); Sohoni et al., [2020](#bib.bib348))
or upweighting high-loss points (Nam et al., [2020](#bib.bib270); Liu et al., [2021a](#bib.bib241)).

Subpopulation shifts are also connected to the well-studied notions of tail performance and risk-averse optimization (Chapter 6 in Shapiro et al. ([2014](#bib.bib338))).
For example, optimizing for the worst case over all subpopulations of a certain size, regardless of domain, can guarantee a certain level of performance over the smaller set of subpopulations defined by domains (Duchi et al., [2020](#bib.bib119); Duchi and Namkoong, [2021](#bib.bib118)).

### 6.3 Setup

We trained CORAL, IRM, and Group DRO models on each dataset.
While Group DRO was originally developed for subpopulation shifts, for completeness, we also experiment with using it for domain generalization. In that setting, Group DRO models aim to achieve similar performance across domains: e.g., in Camelyon17-wilds, where the domains are hospitals, Group DRO optimizes for the training hospital with the highest loss.
Similarly, we also test CORAL and IRM on subpopulation shifts, where they encourage models to learn invariant representations across subpopulations.
As in Section [5](#S5 "5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we used the same OOD validation set for early stopping and to tune the penalty weights for the CORAL and IRM algorithms.
More experimental details are in Appendix [D](#S4a "D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), and dataset-specific hyperparameters and domain choices are discussed in Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### 6.4 Results

Table [2](#S5.T2 "Table 2 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that models trained with CORAL, IRM, and Group DRO generally fail to improve over models trained with ERM.
The exception is the CivilComments-wilds subpopulation shift dataset, where the worst-performing subpopulation is a minority domain. By upweighting the minority domain, Group DRO obtains an OOD accuracy of 70.0% (on the worst-performing subpopulation) compared to 56.0% for ERM, though this is still substantially below the ERM model’s ID accuracy of 92.2% (on average over the entire test set). CORAL and IRM also perform well on CivilComments-wilds, though the gains there stem from the fact that our implementation heuristically upsamples the minority domain (see Appendix [E.6](#S5.SS6 "E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
All other datasets involve domain generalization; the failure of the baseline algorithms here is consistent with other recent findings on standard domain generalization datasets (Gulrajani and Lopez-Paz, [2020](#bib.bib158)).

These results indicate that training models to be robust to distribution shifts in the wild remains a significant open challenge. However, we are optimistic about future progress for two reasons.
First, current methods were mostly designed for other problem settings besides domain generalization, e.g., CORAL for unsupervised domain adaptation and Group DRO for subpopulation shifts.
Second, compared to existing distribution shift datasets, the Wilds datasets generally contain diverse training data from many more domains as well as metadata on these domains, which future algorithms might be able to leverage.

## 7 Empirical trends

We end our discussion of experimental results by briefly reporting on several trends that we observed across multiple datasets.

### 7.1 Underspecification

Prior work has shown that there is often insufficient information at training time to distinguish models that would generalize well under distribution shift;
many models that perform similarly in-distribution (ID) can vary substantially out-of-distribution (OOD) (McCoy et al., [2019a](#bib.bib259); Zhou et al., [2020](#bib.bib433); D’Amour et al., [2020a](#bib.bib102)).
In Wilds, we attempt to alleviate this issue by providing multiple training domains in each dataset as well as an OOD validation set for model selection.
Perhaps as a result, we do not observe significantly higher variance in OOD performance than ID performance in Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), with the exception of
Amazon-wilds and CivilComments-wilds, where the OOD performance is measured on a smaller subpopulation and is therefore naturally more variable.
Excluding those datasets, the average standard deviation from Table [1](#S5.T1 "Table 1 ‣ 5.4 Results ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") is 2.6% for OOD performance and 2.0% for ID performance, which is comparable.
These results raise the question of when underspecification, as reported in prior work, could be more of an issue.

### 7.2 Model selection with in-distribution versus out-of-distribution validation sets

All of the baseline results reported in this paper use an OOD validation set for model selection, as discussed in Section [5.3](#S5.SS3 "5.3 Model selection ‣ 5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
To facilitate research into comparisons of ID versus OOD performance, most Wilds datasets also provide an ID validation and/or test set.
For example, in iWildCam2020-wilds, the ID validation set comprises photos from the same set of camera traps used for the training set.
These ID sets are not used for model selection nor official evaluation.

Gulrajani and Lopez-Paz ([2020](#bib.bib158)) showed that on the DomainBed domain generalization datasets, selecting models with an ID validation set leads to higher OOD performance than using an OOD validation set.
This contrasts with our approach of using OOD validation sets, which we find to generally provide a good estimate of OOD test performance.
Specifically, in Appendix [D.1](#S4.SS1a "D.1 Model hyperparameters ‣ D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we show that for our baseline models, model selection using an OOD validation set results in comparable or higher OOD performance than model selection using an ID validation set.
This difference could stem from many factors: for example, Wilds datasets tend to have many more domains, whereas DomainBed datasets tend to have fewer domains that can be quite different from each other (e.g., cartoons vs. photos);
and there are some differences in the exact procedures for comparing performance using ID versus OOD validation sets.
Further study of the effects of these different model selection procedures and choices of validation sets would be a useful direction for future work.

### 7.3 The compounding effects of multiple distribution shifts

Several Wilds datasets consider hybrid settings, where the goal is to simultaneously generalize to unseen domains as well as to certain subpopulations.
We observe that combining these types of shifts can exacerbate performance drops.
For example, in PovertyMap-wilds and FMoW-wilds, the shift to unseen domains exacerbates the gap in subpopulation performance (and vice versa).
Notably, in FMoW-wilds, the difference in subpopulation performance (across regions) is not even manifested until also considering another shift (across time).
While we do not always observe the compounding effect of distribution shifts—e.g., in Amazon-wilds, subpopulation performance is similar whether we consider shifts to unseen users or not—these observations underscore the importance of evaluating models on the combination of distribution shifts that would occur in practice, instead of considering each shift in isolation.

## 8 Distribution shifts in other application areas

Beyond the datasets currently included in Wilds,
there are many other applications where it is critical for models to be robust to distribution shifts.
In this section, we discuss some of these applications and the challenges of finding appropriate benchmark datasets in those areas.
We also highlight examples of datasets with distribution shifts that we considered but did not include in Wilds, because their distribution shifts did not lead to a significant performance drop.
Constructing realistic benchmarks that reflect distribution shifts in these application areas is an important avenue of future work,
and we would highly welcome community contributions of benchmark datasets in these areas.

### 8.1 Algorithmic fairness

Distribution shifts which degrade model performance on minority subpopulations are frequently discussed in the algorithmic fairness literature.
Geographic inequities are one concern (Shankar et al., [2017](#bib.bib336); Atwood et al., [2020](#bib.bib19)): e.g., publicly available image datasets overrepresent images from the US and Europe, degrading performance in the developing world (Shankar et al., [2017](#bib.bib336)) and prompting the creation of more geographically diverse datasets (Atwood et al., [2020](#bib.bib19)).
Racial disparities are another concern: e.g., commercial gender classifiers are more likely to misclassify the gender of darker-skinned women, likely in part because training datasets overrepresent lighter-skinned subjects (Buolamwini and Gebru, [2018](#bib.bib67)), and pedestrian detection systems fare worse on darker-skinned pedestrians (Wilson et al., [2019](#bib.bib404)). As in Section [4.2.1](#S4.SS2.SSS1 "4.2.1 CivilComments-wilds: Toxicity classification across demographic identities ‣ 4.2 Subpopulation shift datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), NLP models can also show racial bias.

Unfortunately, publicly available algorithmic fairness benchmarks (Mehrabi et al., [2019](#bib.bib262))—e.g., the COMPAS recidivism dataset (Larson et al., [2016](#bib.bib223))—suffer from several limitations.
First, the datasets are often quite small by the standards of modern ML: the COMPAS dataset has only a few thousand rows (Larson et al., [2016](#bib.bib223)).
Second, they tend to have relatively few features,
and disparities in subgroup performance are not always large (Larrazabal et al., [2020](#bib.bib222)),
limiting the benefit of more sophisticated approaches:
on COMPAS, logistic regression performs comparably to a black-box commercial algorithm (Jung et al., [2020](#bib.bib194); Dressel and Farid, [2018](#bib.bib117)).
Third, the datasets sometimes represent “toy” problems: e.g., the UCI Adult Income dataset (Asuncion and Newman, [2007](#bib.bib17)) is widely used as a fairness benchmark, but its task—classifying whether a person will have an income above $50,000—does not represent a real-world application.
Finally, because many of the domains in which algorithmic fairness is of most concern—e.g., criminal justice and healthcare—are high-stakes and disparities are politically sensitive, it can be difficult to make datasets publicly available.

Creating algorithmic fairness benchmarks which do not suffer from these limitations represents a promising direction for future work. In particular, such datasets would ideally have: 1) information about a sensitive attribute like race or gender; 2) a prediction task which is of immediate real-world interest; 3) enough samples, a rich enough feature set, and large enough disparities in group performance that more sophisticated machine learning approaches would plausibly produce improvement over naive approaches.

##### Dataset: New York stop-and-frisk.

Predictive policing is a prominent example of a real-world application where fairness considerations are paramount:
algorithms are increasingly being used in contexts such as predicting crime hotspots (Lum and Isaac, [2016](#bib.bib249)) or a defendant’s risk of reoffending
(Larson et al., [2016](#bib.bib223); Corbett-Davies et al., [2016](#bib.bib93), [2017](#bib.bib94); Lum and Shah, [2019](#bib.bib250)).
There are numerous concerns about these applications (Larson et al., [2016](#bib.bib223); Corbett-Davies et al., [2016](#bib.bib93), [2017](#bib.bib94); Lum and Shah, [2019](#bib.bib250)),
one of which is that these ML models might not generalize beyond the distributions that they were trained on (Corbett-Davies and Goel, [2018](#bib.bib92); Slack et al., [2019](#bib.bib347)).
These distribution shifts include shifts over locations—e.g., a criminal risk assessment trained on several hundred defendants in Ohio was eventually used throughout the United States (Latessa et al., [2010](#bib.bib224))—and shifts over time, as sentencing and other criminal justice policies evolve (Corbett-Davies and Goel, [2018](#bib.bib92)).
There are, of course, also subpopulation shift concerns around whether models are biased against particular demographic groups.

We investigated these shifts using a dataset of pedestrian stops made by the New York City Police Department under its “stop-and-frisk” policy,
where the task is to predict whether a pedestrian who was stopped on suspicion of weapon possession would in fact possess a weapon (Goel et al., [2016](#bib.bib150)).
This policy had a pronounced racial bias: Black people stopped by the police on suspicion of possessing a weapon were 5×\times less likely to actually possess one than their White counterparts (Goel et al., [2016](#bib.bib150)).
We emphasize that we oppose stop-and-frisk (and any “improved” ML-powered stop-and-frisk) since there is overwhelming evidence that the policy was racially discriminatory (Gelman et al., [2007](#bib.bib145); Goel et al., [2016](#bib.bib150); Pierson et al., [2018](#bib.bib296)) and such massive inequities require more than algorithmic fixes.
Rather, we use the dataset as a realistic example of the phenomena that arise in real policing contexts, including 1) substantial heterogeneity across locations and racial groups and 2) distributions that arise in part because of biased policing practices.

Overall, we found large performance disparities across race groups and locations. Interestingly, however, we also found that these disparities cannot be attributed to the distribution shift, as the disparities were not reduced when we trained models specifically on the race groups or locations that suffer the worst performance.
Indeed, the groups that see the worst performance—Black and Hispanic pedestrians—comprise large *majorities* of the dataset, making up more than 90% of the stops. This contrasts with the typical setting in algorithmic fairness where models perform worse on *minority* groups in the training data.
Our results suggest the disparities are due to the dataset being noisier for some race and location groups, potentially as a result of the biased policing practices underlying the dataset. We provide further details in Appendix [F.1](#S6.SS1a "F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### 8.2 Medicine and healthcare

Substantial evidence indicates the potential for distribution shifts in medical settings (Finlayson et al., [2021](#bib.bib131)). One concern is *demographic* subpopulation shifts (e.g., across race, gender, or socioeconomic status), since historically-disadvantaged populations are underrepresented in many medical datasets (Chen et al., [2020](#bib.bib79)).
Another concern is heterogeneity *across hospitals*; this might include differences in imaging, as in Section [4.1.2](#S4.SS1.SSS2 "4.1.2 Camelyon17-wilds: Tumor identification across different hospitals ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), and other operational protocols such as lab tests (D’Amour et al., [2020a](#bib.bib102); Subbaswamy et al., [2020](#bib.bib355)).
Finally, changes *over time* can also produce distribution shifts: for example, Nestor et al. ([2019](#bib.bib272)) showed that switching between two electronic health record (EHR) systems produced a drop in performance,
and the COVID-19 epidemic has affected the distribution of chest radiographs (Wong et al., [2020](#bib.bib407)).

Creating medical distribution shift benchmarks thus represents a promising direction for future work, if several challenges can be overcome. First, while there are large demographic disparities in healthcare outcomes (e.g., by race or socioeconomic status), many of them are not due to distribution shifts, but to disparities in non-algorithmic factors (e.g., access to care or prevalence of comorbidities (Chen et al., [2020](#bib.bib79))) or to algorithmic problems unrelated to distribution shift (e.g., choice of a biased outcome variable (Obermeyer et al., [2019](#bib.bib280))).
Indeed, several previous investigations have found relatively small disparities in algorithmic performance (as opposed to healthcare outcomes) across demographic groups (Chen et al., [2019a](#bib.bib78); Larrazabal et al., [2020](#bib.bib222)); Seyyed-Kalantari et al. ([2020](#bib.bib334)) finds larger disparities in true positive rates across demographic groups, but this might reflect the different underlying label distributions between groups.

Second, many distribution shifts in medicine arise from concept drifts, in which the relationship between the input and the label changes, for example due to changes in clinical procedures and the definition of the label (Widmer and Kubat, [1996](#bib.bib402); Beyene et al., [2015](#bib.bib46); Futoma et al., [2020](#bib.bib134)). It can be difficult to ensure that a potential benchmark has sufficient leverage for models to learn how to handle, e.g., an abrupt change in the way a particular clinical procedure is carried out.

A last challenge is data availability, as stringent medical privacy laws often preclude data sharing (Price and Cohen, [2019](#bib.bib299)). For example, EHR datasets are fundamental to medical decision-making, but there are few widely adopted EHR benchmarks—with the MIMIC database being a prominent exception (Johnson et al., [2016](#bib.bib190))—and relatively little progress in predictive performance has been made on them (Bellamy et al., [2020](#bib.bib41)).

### 8.3 Genomics

Advances in high-throughput genomic and molecular profiling platforms have enabled systematic mapping of biochemical activity of genomes across diverse cellular contexts, populations, and species
(Consortium et al., [2012](#bib.bib89); Ho et al., [2014](#bib.bib177); Kundaje et al., [2015](#bib.bib216); Regev et al., [2017](#bib.bib309); Consortium, [2019](#bib.bib91); Moore et al., [2020](#bib.bib268); Consortium et al., [2020](#bib.bib90)).
These datasets have powered ML models that have been fairly successful at deciphering functional DNA sequence patterns and predicting the consequences of genetic perturbations in cell types in which the models are trained (Libbrecht and Noble, [2015](#bib.bib239); Zhou and Troyanskaya, [2015](#bib.bib432); Kelley et al., [2016](#bib.bib203); Ching et al., [2018](#bib.bib82); Eraslan et al., [2019](#bib.bib123); Jaganathan et al., [2019](#bib.bib186); Avsec et al., [2021b](#bib.bib21)).
However, distribution shifts pose a significant obstacle to generalizing these predictions to new cell types.

A concrete example is the prediction of genome-wide profiles of regulatory protein-DNA binding interactions across cell types and tissues (Srivastava and Mahony, [2020](#bib.bib350)).
These regulatory maps are critical for understanding the fundamental mechanisms of dynamic gene regulation across healthy and diseased cell states, and predictive models are an essential complement to experimental approaches for comprehensively profiling these maps.

Regulatory proteins bind regulatory DNA elements in a sequence-specific manner to orchestrate gene expression programs.
These proteins often form different complexes with each other in different cell types.
These cell-type-specific protein complexes can recognize distinct combinatorial sequence syntax and thereby bind to different genomic locations in different cell types, even if all of these cell types share the same genomic sequence.
Hence, ML models that aim to predict protein-DNA binding landscapes across cell types typically integrate DNA sequence and additional context-specific input data modalities, which provide auxiliary information about the regulatory state of DNA in each cell type (Srivastava and Mahony, [2020](#bib.bib350)).
The training cell-type specific sequence determinants of binding induce a distribution shift across cell types, which can in turn degrade model performance on new cell types (Balsubramani et al., [2017](#bib.bib26); Li et al., [2019a](#bib.bib232); Li and Guan, [2019](#bib.bib231); Keilwagen et al., [2019](#bib.bib202); Quang and Xie, [2019](#bib.bib303)).

##### Dataset: Genome-wide protein-DNA binding profiles across different cell types.

We studied the above problem in the context of the ENCODE-DREAM in-vivo Transcription Factor Binding Site Prediction Challenge (Balsubramani et al., [2020](#bib.bib27)),
which is an open community challenge introduced to systematically benchmark ML models for predicting genome-wide DNA binding maps of many regulatory proteins across cell types.

For each regulatory protein, regions of the genome are associated with binary labels (bound/unbound).
The task is to predict these binary binding labels as a function of underlying DNA sequence and chromatin accessibility signal (an experimental measure of cell type-specific regulatory state) in test cell types that are not represented in the training set.

A systematic evaluation of the top-performing models in this challenge highlighted a significant gap in prediction performance across cell types, relative to cross-validation performance within training cell types (Li et al., [2019a](#bib.bib232); Li and Guan, [2019](#bib.bib231); Keilwagen et al., [2019](#bib.bib202); Quang and Xie, [2019](#bib.bib303)).
This performance gap was attributed to distribution shifts across cell types, due to regulatory proteins forming cell-type-specific complexes that can recognize different combinatorial sequence syntax.
Hence, the same DNA sequence can be associated with different binding labels for a protein across contexts.

We investigated these distribution shifts in more detail for a restricted subset of the challenge’s prediction tasks for two regulatory proteins, using a total of 14 genome-wide binding maps across different cell types.
While we generally found a performance gap between in- and out-of-distribution settings, we did not include this dataset in the official benchmark for several reasons.
For example, we were unable to learn a model that could generalize across all the cell types simultaneously, even in an in-distribution setting,
which suggested that the model family and/or feature set might not be rich enough to fit the variation across different cell types.
Another major complication was the significant variation in intrinsic difficulty across different splits, as measured by the performance of models we train in-distribution.
Further work will be required to construct a rigorous benchmark for evaluating distribution shifts in the context of predicting regulatory binding maps.
We discuss details in Appendix [F.2](#S6.SS2a "F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### 8.4 Natural language and speech processing

Subpopulation shifts are an issue in automated speech recognition (ASR) systems, which have been shown to have higher error rates for Black speakers than for White speakers (Koenecke et al., [2020](#bib.bib208)) and for speakers of some dialects (Tatman, [2017](#bib.bib366)).
These disparities were demonstrated using commercial ASR systems, and therefore do not have any accompanying training datasets that are publicly available.
There are many public speech datasets with speaker metadata that could potentially be used to construct a benchmark, e.g., LibriSpeech (Panayotov et al., [2015](#bib.bib284)),
the Speech Accent Archive (Weinberger, [2015](#bib.bib398)),
VoxCeleb2 (Chung et al., [2018](#bib.bib84)),
the Spoken Wikipedia Corpus (Baumann et al., [2019](#bib.bib31)),
and Common Voice (Ardila et al., [2020](#bib.bib15)).
However, these datasets have their own challenges: some do not have a sufficiently diverse sample of speaker backgrounds and accents, and others focus on read speech (e.g., audiobooks) instead of more natural speech.

In natural language processing (NLP), a current focus is on challenge datasets that are crafted to test particular aspects of models,
e.g., HANS (McCoy et al., [2019b](#bib.bib260)), PAWS (Zhang et al., [2019](#bib.bib430)),
and CheckList (Ribeiro et al., [2020](#bib.bib314)).
These challenge datasets are drawn from test distributions that are often (deliberately) quite different from the data distributions that models are typically trained on.
Counterfactually-augmented datasets (Kaushik et al., [2019](#bib.bib200)) are a related type of challenge dataset where the training data is modified to make spurious correlates independent of the target, which can result in more robust models.
Others have studied train/test sets that are drawn from different sources, e.g., Wikipedia, Reddit, news articles, travel reviews, and so on (Oren et al., [2019](#bib.bib281); Miller et al., [2020](#bib.bib264); Kamath et al., [2020](#bib.bib198)).

Several synthetic datasets have also been designed to test compositional generalization, such as CLEVR (Johnson et al., [2017](#bib.bib191)), SCAN (Lake and Baroni, [2018](#bib.bib218)), and COGS (Kim and Linzen, [2020](#bib.bib205)). The test sets in these datasets are chosen such that models need to generalize to novel combinations of parts of training examples, e.g., familiar primitives and grammatical roles (Kim and Linzen, [2020](#bib.bib205)).
CLEVR is a visual question-answering (VQA) dataset; other examples of VQA datasets that are formulated as challenge datasets are the VQA-CP v1 and v2 datasets (Agrawal et al., [2018](#bib.bib3)), which create subpopulation shifts by intentionally altering the distribution of answers per question type between the train and test splits.

These NLP examples involve English-language models; other languages typically have fewer and smaller datasets available for training and benchmarking models.
Multi-lingual models and benchmarks (Conneau et al., [2018](#bib.bib88); Conneau and Lample, [2019](#bib.bib87); Hu et al., [2020a](#bib.bib180); Clark et al., [2020](#bib.bib85)) are another source of subpopulation shifts with corresponding disparities in performance:
training sets might contain fewer examples in low-resource languages (Nekoto et al., [2020](#bib.bib271)),
but we would still hope for high model performance on these minority groups.

##### Datasets: Other distribution shifts in Amazon and Yelp reviews.

In addition to user shifts on the Amazon Reviews dataset (Ni et al., [2019](#bib.bib274)), we also looked at category and time shifts on the same dataset, as well as user and time shifts on the Yelp Open Dataset555<https://www.yelp.com/dataset>.
However, for many of those shifts, we only found modest performance drops.
We provide additional details on Amazon in Appendix [F.4](#S6.SS4a "F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and on Yelp in Appendix [F.5](#S6.SS5 "F.5 Yelp: Sentiment classification across different users and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### 8.5 Education

ML models can help in educational settings in a variety of ways:
e.g., assisting in grading (Piech et al., [2013](#bib.bib295); Shermis, [2014](#bib.bib340); Kulkarni et al., [2014](#bib.bib214); Taghipour and Ng, [2016](#bib.bib364)),
estimating student knowledge (Desmarais and Baker, [2012](#bib.bib109); Wu et al., [2020](#bib.bib410)),
identifying students who need help (Ahadi et al., [2015](#bib.bib4)),
or automatically generating explanations (Williams et al., [2016](#bib.bib403); Wu et al., [2019a](#bib.bib409)).
However, there are substantial distribution shifts in these settings as well.
For example, automatic essay scoring has been found to be affected by rater bias (Amorim et al., [2018](#bib.bib13)) and spurious correlations like essay length (Perelman, [2014](#bib.bib292)),
leading to problems with subpopulation shift.
Ideally, these systems would also generalize across different contexts, e.g., a model for scoring grammar should work well across multiple different essay prompts.
Recent attempts at predicting grades algorithmically (BBC, [2020](#bib.bib32); Broussard, [2020](#bib.bib61)) have also been found to be biased against certain subpopulations.

Unfortunately, there is a general lack of standardized education datasets, in part due to student privacy concerns and the proprietary nature of large-scale standardized tests.
Datasets from massive open online courses are a potential source of large-scale data (Kulkarni et al., [2015](#bib.bib213)).
In general, dataset construction for ML in education is an active area—e.g.,
the NeurIPS 2020 workshop on Machine Learning for Education666<https://www.ml4ed.org/>
has a segment devoted to finding “ImageNets for education”—and we hope to be able to include one in the future.

### 8.6 Robotics

Robot learning has emerged as a strong paradigm for automatically acquiring complex and skilled
behaviors such as locomotion (Yang et al., [2019](#bib.bib420); Peng et al., [2020](#bib.bib291)), navigation (Mirowski et al., [2017](#bib.bib267); Kahn et al., [2020](#bib.bib196)),
and manipulation (Gu et al., [2017](#bib.bib157); et al, [2019](#bib.bib126)). However, the advent of learning-based techniques for
robotics has not convincingly addressed, and has perhaps even exasperated, problems stemming from
distribution shift. These problems have manifested in many ways, including shifts induced by weather
and lighting changes (Wulfmeier et al., [2018](#bib.bib413)), location changes (Gupta et al., [2018](#bib.bib160)), and the
simulation-to-real-world gap (Sadeghi and Levine, [2017](#bib.bib323); Tobin et al., [2017](#bib.bib375)).
Dealing with these challenging
scenarios is critical to deploying robots in the real world, especially in high-stakes
decision-making scenarios.

For example, to safely deploy autonomous driving vehicles, it is critical that these systems work
reliably and robustly across the huge variety of conditions that exist in the real world, such as
locations, lighting and weather conditions, and sensor intrinsics. This is a challenging
requirement, as many of these conditions may be underrepresented, or not represented at all, by the
available training data. Indeed, prior work has shown that naively trained models can suffer at
segmenting nighttime driving scenes (Dai and Van Gool, [2018](#bib.bib101)), detecting relevant objects in new or
challenging locations and settings (Yu et al., [2020](#bib.bib424); Sun et al., [2020a](#bib.bib358)), and, as discussed earlier, detecting
pedestrians with darker skin tones (Wilson et al., [2019](#bib.bib404)).

Creating a benchmark for distribution shifts in robotics applications, such as autonomous driving,
represents a promising direction for future work. Here, we briefly summarize our initial findings on
distribution shifts in the BDD100K driving dataset (Yu et al., [2020](#bib.bib424)), which is publicly available and
widely used, including in some of the works listed above.

##### Dataset: BDD100K.

We investigated the task of multi-label binary classification of the presence of each object
category in each image. In general, we found no substantial performance drops across a wide range of
different test scenarios, including user shifts, weather and time shifts, and location shifts. We
provide additional details in Section [F.3](#S6.SS3a "F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

Our findings contrast with previous findings that other tasks, such as object detection and
segmentation, can suffer under the same types of shifts on the same dataset
(Yu et al., [2020](#bib.bib424); Dai and Van Gool, [2018](#bib.bib101)). Currently, Wilds consists of datasets involving classification and
regression tasks. However, most tasks of interest in autonomous driving, and robotics in general,
are difficult to formulate as classification or regression. For example, autonomous driving
applications may require models for object detection or lane and scene segmentation. These tasks are
often more challenging than classification tasks, and we speculate that they may suffer more
severely from distribution shift.

### 8.7 Feedback loops

Finally, we have restricted our attention to settings where the data distribution is independent of the model. When the data distribution does depend on the model, distribution shifts can arise from feedback loops between the data and the model. Examples include recommendation systems and other consumer products (Bottou et al., [2013](#bib.bib56); Hashimoto et al., [2018](#bib.bib166)); dialogue agents (Li et al., [2017b](#bib.bib234));
molecular compound optimization (Cuccarese et al., [2020](#bib.bib99); Reker, [2020](#bib.bib311));
decision systems (Liu et al., [2018](#bib.bib242); D’Amour et al., [2020b](#bib.bib103)); and adversarial settings like fraud or malware detection (Rigaki and Garcia, [2018](#bib.bib316)).
While these adaptive settings are outside the scope of our benchmark,
dealing with these types of distribution shifts is an important area of ongoing work.

## 9 Guidelines for method developers

We now discuss some community guidelines for method development using Wilds.
More specific submission guidelines for our leaderboard can be found at
<https://wilds.stanford.edu>.

### 9.1 General-purpose and specialized training algorithms

Wilds is primarily designed as a benchmark for developing and evaluating algorithms for training models that are robust to distribution shifts.
To facilitate systematic comparisons of these algorithms,
we encourage algorithm developers to use the standardized datasets (i.e., with no external data) and default model architectures provided in Wilds,
as doing so will help to isolate the contributions of the algorithm versus the training dataset or model architecture.
Our primary leaderboard will focus on submissions that follow these guidelines.

Moreover, we encourage developers to test their algorithms on all applicable Wilds datasets, so as to assess how well they do across different types of data and distribution shifts. We emphasize that it is still an open question
if a single general-purpose training algorithm can produce models that do well on all of the datasets without accounting for the particular structure of the distribution shift in each dataset.
As such, it would still be a substantial advance if an algorithm significantly improves performance on one type of shift but not others;
we aim for Wilds to facilitate research into both general-purpose algorithms as well as ones that are more specifically tailored to a particular application and type of distribution shift.

### 9.2 Methods beyond training algorithms

Beyond new training algorithms, there are many other promising directions for improving distributional robustness, including new model architectures and pre-training on additional external data beyond what is used in our default models. We encourage developers to test these approaches on Wilds as well, and we will track all such submissions on a separate leaderboard from the training algorithm leaderboard.

### 9.3 Avoiding overfitting to the test distribution

While each Wilds dataset aims to benchmark robustness to a type of distribution shift (e.g., shifts to unseen hospitals), practical limitations mean that for some datasets, we have data from only a limited number of domains (e.g., one OOD test hospital in Camelyon17-wilds).
As there can be substantial variability in performance across domains,
developers should be careful to avoid overfitting to the specific test sets in Wilds, especially on datasets like Camelyon17-wilds with limited test domains.
We strongly encourage all model developers to use the provided OOD validation sets for development and model selection, and to only use the OOD test sets for their final evaluations.

### 9.4 Reporting both ID and OOD performance

Prior work has shown that for many tasks, ID and OOD performance can be highly correlated across different model architectures and hyperparameters (Taori et al., [2020](#bib.bib365); Liu et al., [2021b](#bib.bib243); Miller et al., [2021](#bib.bib265)).
It is reasonable to expect that methods for improving ID performance could also give corresponding improvements in OOD performance in Wilds, and we welcome submissions of such methods.
To better understand the extent to which any gains in OOD performance can be attributed to improved ID performance versus a model that is more robust to (i.e., less affected by) the distribution shift,
we encourage model developers to report both ID and OOD performance numbers.
See Miller et al. ([2021](#bib.bib265)) for an in-depth discussion of this point.

### 9.5 Extensions to other problem settings

In this paper, we focused on the domain generalization and subpopulation shift settings.
In Appendix [C](#S3a "C Potential extensions to other problem settings ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we discuss how Wilds can be used in other realistic problem settings that allow training algorithms to leverage additional information, such as unlabeled test data in unsupervised domain adaptation (Ben-David et al., [2006](#bib.bib43)).
These sources of leverage could be fruitful approaches to improving OOD performance, and we welcome community contributions towards this effort.

## 10 Using the Wilds package

Finally, we discuss our open-source PyTorch-based package that exposes a simple interface to our datasets and automatically handles data downloads, allowing users to get started on a Wilds dataset in just a few lines of code.
In addition, the package provides various data loaders and utilities surrounding domain annotations and other metadata, which supports training algorithms that need access to these metadata.
The package also provides standardized evaluations for each dataset.
More documentation and installation information can be found at <https://wilds.stanford.edu>.

##### Datasets and data loading.

The Wilds package provides a simple, standardized interface for all datasets in the benchmark as well as their data loaders, as summarized in Figure [13](#S10.F13 "Figure 13 ‣ Datasets and data loading. ‣ 10 Using the Wilds package ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
This short code snippet covers all of the steps of getting started with a Wilds dataset, including dataset download and initialization, accessing various splits, and initializing the data loader.
We also provide multiple data loaders in order to accommodate a wide array of algorithms, which often require specific data loading schemes.

![Refer to caption](/html/2012.07421/assets/x12.png)


Figure 13: 
Dataset initialization and data loading.

##### Domain information.

To allow algorithms to leverage domain annotations as well as other groupings over the available metadata, the Wilds package provides Grouper objects.
Grouper objects (e.g., grouper in Figure [14](#S10.F14 "Figure 14 ‣ Domain information. ‣ 10 Using the Wilds package ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")) extract group annotations from metadata, allowing users to specify the grouping scheme in a flexible fashion.

![Refer to caption](/html/2012.07421/assets/x13.png)


Figure 14: 
Accessing domain and other group information via a Grouper object.

##### Evaluation.

Finally, the Wilds package standardizes and automates the evaluation for each dataset.
As summarized in Figure [15](#S10.F15 "Figure 15 ‣ Evaluation. ‣ 10 Using the Wilds package ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), invoking the eval method of each dataset yields all metrics reported in the paper and on the leaderboard.

![Refer to caption](/html/2012.07421/assets/x14.png)


Figure 15: 
Evaluation.

## Reproducibility

An executable version of our paper, hosted on CodaLab, can be found at <https://wilds.stanford.edu/codalab>. This contains the exact commands, code, environment, and data used for the experiments reported in our paper, as well as all trained model weights. The WILDS package is open-source and can be found at <https://github.com/p-lambda/wilds>.

## Acknowledgements

Many people generously volunteered their time and expertise to advise us on Wilds.
We are grateful for all of the helpful suggestions and constructive feedback from:
Aditya Khosla,
Andreas Schlueter,
Annie Chen,
Aleksander Madry,
Alexander D’Amour,
Allison Koenecke,
Alyssa Lees,
Ananya Kumar,
Andrew Beck,
Behzad Haghgoo,
Charles Sutton,
Christopher Yeh,
Cody Coleman,
Dan Hendrycks,
Dan Jurafsky,
Daniel Levy,
Daphne Koller,
David Tellez,
Erik Jones,
Evan Liu,
Fisher Yu,
Georgi Marinov,
Hongseok Namkoong,
Irene Chen,
Jacky Kang,
Jacob Schreiber,
Jacob Steinhardt,
Jared Dunnmon,
Jean Feng,
Jeffrey Sorensen,
Jianmo Ni,
John Hewitt,
John Miller,
Kate Saenko,
Kelly Cochran,
Kensen Shi,
Kyle Loh,
Li Jiang,
Lucy Vasserman,
Ludwig Schmidt,
Luke Oakden-Rayner,
Marco Tulio Ribeiro,
Matthew Lungren,
Megha Srivastava,
Nelson Liu,
Nimit Sohoni,
Pranav Rajpurkar,
Robin Jia,
Rohan Taori,
Sarah Bird,
Sharad Goel,
Sherrie Wang,
Shyamal Buch,
Stefano Ermon,
Steve Yadlowsky,
Tatsunori Hashimoto,
Tengyu Ma,
Vincent Hellendoorn,
Yair Carmon,
Zachary Lipton,
and Zhenghao Chen.

The design of the WILDS benchmark was inspired by the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)), and we are grateful to the Open Graph Benchmark team for their advice and help in setting up our benchmark.

This project was funded by an Open Philanthropy Project Award and NSF Award Grant No. 1805310.
Shiori Sagawa was supported by the Herbert Kunzel Stanford Graduate Fellowship.
Henrik Marklund was supported by the Dr. Tech. Marcus Wallenberg Foundation for Education in International Industrial Entrepreneurship, CIFAR, and Google.
Sang Michael Xie and Marvin Zhang were supported by NDSEG Graduate Fellowships.
Weihua Hu was supported by the Funai Overseas Scholarship and the Masason Foundation Fellowship.
Sara Beery was supported by an NSF Graduate Research Fellowship and is a PIMCO Fellow in Data Science.
Jure Leskovec is a Chan Zuckerberg Biohub investigator.
Chelsea Finn is a CIFAR Fellow in the Learning in Machines and Brains Program.

We also gratefully acknowledge the support of DARPA under Nos. N660011924033 (MCS);
ARO under Nos. W911NF-16-1-0342 (MURI), W911NF-16-1-0171 (DURIP);
NSF under Nos. OAC-1835598 (CINES), OAC-1934578 (HDR), CCF-1918940 (Expeditions), IIS-2030477 (RAPID); Stanford Data Science Initiative, Wu Tsai Neurosciences Institute, Chan Zuckerberg Biohub, Amazon, JPMorgan Chase, Docomo, Hitachi, JD.com, KDDI, NVIDIA, Dell, Toshiba, and UnitedHealth Group.

## References

* Abelson et al. (2014)

  B. Abelson, K. R. Varshney, and J. Sun.
  Targeting direct cash transfers to the extremely poor.
  In *International Conference on Knowledge Discovery and Data
  Mining (KDD)*, 2014.
* Adragna et al. (2020)

  R. Adragna, E. Creager, D. Madras, and R. Zemel.
  Fairness and robustness in invariant learning: A case study in
  toxicity classification.
  *arXiv preprint arXiv:2011.06485*, 2020.
* Agrawal et al. (2018)

  Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi.
  Don’t just assume; look and answer: Overcoming priors for visual
  question answering.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  4971–4980, 2018.
* Ahadi et al. (2015)

  A. Ahadi, R. Lister, H. Haapala, and A. Vihavainen.
  Exploring machine learning methods to automatically identify students
  in need of assistance.
  In *Proceedings of the Eleventh Annual International Conference
  on International Computing Education Research*, pages 121–130, 2015.
* Ahumada et al. (2020)

  Jorge A Ahumada, Eric Fegraus, Tanya Birch, Nicole Flores, Roland Kays,
  Timothy G O’Brien, Jonathan Palmer, Stephanie Schuttler, Jennifer Y Zhao,
  Walter Jetz, Margaret Kinnaird, Sayali Kulkarni, Arnaud Lyet, David Thau,
  Michelle Duong, Ruth Oliver, and Anthony Dancer.
  Wildlife insights: A platform to maximize the potential of camera
  trap and other passive sensor wildlife data for the planet.
  *Environmental Conservation*, 47(1):1–6,
  2020.
* Aich et al. (2018)

  Shubhra Aich, Anique Josuttes, Ilya Ovsyannikov, Keegan Strueby, Imran Ahmed,
  Hema Sudhakar Duddu, Curtis Pozniak, Steve Shirtliffe, and Ian Stavness.
  Deepwheat: Estimating phenotypic traits from crop images with deep
  learning.
  In *2018 IEEE Winter Conference on Applications of Computer
  Vision (WACV)*, pages 323–332. IEEE, 2018.
* AlBadawy et al. (2018)

  E. AlBadawy, A. Saha, and M. Mazurowski.
  Deep learning for segmentation of brain tumors: Impact of
  cross-institutional training and testing.
  *Med Phys.*, 45, 2018.
* Alexandari et al. (2020)

  A. Alexandari, A. Kundaje, and A. Shrikumar.
  Maximum likelihood with bias-corrected calibration is hard-to-beat at
  label shift adaptation.
  In *International Conference on Machine Learning (ICML)*, pages
  222–232, 2020.
* Alipanahi et al. (2015)

  Babak Alipanahi, Andrew Delong, Matthew T Weirauch, and Brendan J Frey.
  Predicting the sequence specificities of dna-and rna-binding proteins
  by deep learning.
  *Nature biotechnology*, 33(8):831, 2015.
* Allamanis and Brockschmidt (2017)

  Miltiadis Allamanis and Marc Brockschmidt.
  Smartpaste: Learning to adapt source code.
  *arXiv preprint arXiv:1705.07867*, 2017.
* Allamanis et al. (2015)

  Miltiadis Allamanis, Earl T Barr, Christian Bird, and Charles Sutton.
  Suggesting accurate method and class names.
  In *Proceedings of the 2015 10th Joint Meeting on Foundations of
  Software Engineering*, pages 38–49, 2015.
* Allevato et al. (2017)

  Michael Allevato, Eugene Bolotin, Mark Grossman, Daniel Mane-Padros, Frances M
  Sladek, and Ernest Martinez.
  Sequence-specific dna binding by myc/max to low-affinity non-e-box
  motifs.
  *PloS one*, 12(7):e0180147, 2017.
* Amorim et al. (2018)

  E. Amorim, M. Cançado, and A. Veloso.
  Automated essay scoring in the presence of biased ratings.
  In *Association for Computational Linguistics (ACL)*, pages
  229–237, 2018.
* Ando et al. (2017)

  D Michael Ando, Cory Y McLean, and Marc Berndl.
  Improving phenotypic measurements in high-content imaging screens.
  *BioRxiv*, page 161422, 2017.
* Ardila et al. (2020)

  R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais,
  L. Saunders, F. Tyers, and G. Weber.
  Common voice: A massively-multilingual speech corpus.
  In *Language Resources and Evaluation Conference (LREC)*, pages
  4218–4222, 2020.
* Arjovsky et al. (2019)

  M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz.
  Invariant risk minimization.
  *arXiv preprint arXiv:1907.02893*, 2019.
* Asuncion and Newman (2007)

  Arthur Asuncion and David Newman.
  UCI Machine Learning Repository, 2007.
* Attene-Ramos et al. (2013)

  M. S. Attene-Ramos, N. Miller, R. Huang, S. Michael, M. Itkin, R. J. Kavlock,
  C. P. Austin, P. Shinn, A. Simeonov, R. R. Tice, et al.
  The tox21 robotic platform for the assessment of environmental
  chemicals–from vision to reality.
  *Drug Discovery Today*, 18(15):716–723,
  2013.
* Atwood et al. (2020)

  J. Atwood, Y. Halpern, P. Baljekar, E. Breck, D. Sculley, P. Ostyakov, S. I.
  Nikolenko, I. Ivanov, R. Solovyev, W. Wang, et al.
  The Inclusive Images competition.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 155–186, 2020.
* Avsec et al. (2021a)

  Ziga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka
  Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli,
  and David R Kelley.
  Effective gene expression prediction from sequence by integrating
  long-range interactions.
  *bioRxiv*, 2021a.
* Avsec et al. (2021b)

  Žiga Avsec, Melanie Weilert, Avanti Shrikumar, Sabrina Krueger, Amr
  Alexandari, Khyati Dalal, Robin Fropf, Charles McAnany, Julien Gagneur, and
  Anshul Kundaje.
  Base-resolution models of transcription-factor binding reveal soft
  motif syntax.
  *Nature Genetics*, pages 1–13, 2021b.
* Ayalew et al. (2020)

  Tewodros W Ayalew, Jordan R Ubbens, and Ian Stavness.
  Unsupervised domain adaptation for plant organ counting.
  In *European Conference on Computer Vision*, pages 330–346.
  Springer, 2020.
* Azizzadenesheli et al. (2019)

  K. Azizzadenesheli, A. Liu, F. Yang, and A. Anandkumar.
  Regularized learning for domain adaptation under label shifts.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Badgeley et al. (2019)

  M. A. Badgeley, J. R. Zech, L. Oakden-Rayner, B. S. Glicksberg, M. Liu,
  W. Gale, M. V. McConnell, B. Percha, T. M. Snyder, and J. T. Dudley.
  Deep learning predicts hip fracture using confounding patient and
  healthcare variables.
  *npj Digital Medicine*, 2, 2019.
* Balaji et al. (2018)

  Yogesh Balaji, Swami Sankaranarayanan, and Rama Chellappa.
  Metareg: Towards domain generalization using meta-regularization.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 998–1008, 2018.
* Balsubramani et al. (2017)

  Akshay Balsubramani, Nathan Boley, James C. Costello, Laura M. Heiser, Tim
  Jeske, Robert Kueffner, Jin Wook Lee, Rani K. Powers, and Anshul Kundaje.
  The encode-dream challenge to predict genome-wide binding of
  regulatory proteins to dna.
  *Neural Information Processing Systems (Workshop on Machine
  Learning Challenges as a Research Tool)*, 2017.
* Balsubramani et al. (2020)

  Akshay Balsubramani, Nathan Boley, Jin Wook Lee, Rani K. Powers, Bruce Hoff,
  Thomas Yu, Tim Jeske, Stephen Friend, Thea Norman, Gustavo Stolovitzky,
  Robert Kueffner, Laura M. Heiser, James C. Costello, and Anshul Kundaje.
  Encode-dream in vivo transcription factor binding site prediction
  challenge.
  Synapse:
  <https://www.synapse.org/#!Synapse:syn6131484/wiki/402026>, 2020.
  Accessed: 2020-10-23. doi:10.7303/syn6131484.
* Bandi et al. (2018)

  P. Bandi, O. Geessink, Q. Manson, M. V. Dijk, M. Balkenhol, M. Hermsen, B. E.
  Bejnordi, B. Lee, K. Paeng, A. Zhong, et al.
  From detection of individual metastases to classification of lymph
  node status at the patient level: the CAMELYON17 challenge.
  *IEEE Transactions on Medical Imaging*, 38(2):550–560, 2018.
* Barbu et al. (2019)

  A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and
  B. Katz.
  Objectnet: A large-scale bias-controlled dataset for pushing the
  limits of object recognition models.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 9453–9463, 2019.
* Bartlett and Wegkamp (2008)

  P. L. Bartlett and M. H. Wegkamp.
  Classification with a reject option using a hinge loss.
  *Journal of Machine Learning Research (JMLR)*, 9(0):1823–1840, 2008.
* Baumann et al. (2019)

  T. Baumann, A. Köhn, and F. Hennig.
  The Spoken Wikipedia Corpus collection: Harvesting, alignment and
  an application to hyperlistening.
  *Language Resources and Evaluation*, 53(2):303–329, 2019.
* BBC (2020)

  BBC.
  A-levels and GCSEs: How did the exam algorithm work?
  *The British Broadcasting Corporation*, 2020.
  URL <https://www.bbc.com/news/explainers-53807730>.
* Beck et al. (2011)

  A. H. Beck, A. R. Sangoi, S. Leung, R. J. Marinelli, T. O. Nielsen, M. J. V. D.
  Vijver, R. B. West, M. V. D. Rijn, and D. Koller.
  Systematic analysis of breast cancer morphology uncovers stromal
  features associated with survival.
  *Science*, 3(108), 2011.
* Becke (2014)

  Axel D Becke.
  Perspective: Fifty years of density-functional theory in chemical
  physics.
  *The Journal of Chemical Physics*, 140(18):18A301, 2014.
* Beede et al. (2020)

  E. Beede, E. Baylor, F. Hersch, A. Iurchenko, L. Wilcox, P. Ruamviboonsuk, and
  L. M. Vardoulakis.
  A human-centered evaluation of a deep learning system deployed in
  clinics for the detection of diabetic retinopathy.
  In *Conference on Human Factors in Computing Systems (CHI)*,
  pages 1–12, 2020.
* Beery et al. (2018)

  S. Beery, G. V. Horn, and P. Perona.
  Recognition in terra incognita.
  In *European Conference on Computer Vision (ECCV)*, pages
  456–473, 2018.
* Beery et al. (2020a)

  S. Beery, E. Cole, and A. Gjoka.
  The iWildCam 2020 competition dataset.
  *arXiv preprint arXiv:2004.10340*, 2020a.
* Beery et al. (2019)

  Sara Beery, Dan Morris, and Siyu Yang.
  Efficient pipeline for camera trap image review.
  *arXiv preprint arXiv:1907.06772*, 2019.
* Beery et al. (2020b)

  Sara Beery, Guanhang Wu, Vivek Rathod, Ronny Votel, and Jonathan Huang.
  Context r-cnn: Long term temporal context for per-camera object
  detection.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pages 13075–13085, 2020b.
* Bejnordi et al. (2017)

  B. E. Bejnordi, M. Veta, P. J. V. Diest, B. V. Ginneken, N. Karssemeijer,
  G. Litjens, J. A. V. D. Laak, M. Hermsen, Q. F. Manson, M. Balkenhol, et al.
  Diagnostic assessment of deep learning algorithms for detection of
  lymph node metastases in women with breast cancer.
  *JAMA*, 318(22):2199–2210, 2017.
* Bellamy et al. (2020)

  D. Bellamy, L. Celi, and A. L. Beam.
  Evaluating progress on machine learning for longitudinal electronic
  healthcare data.
  *arXiv preprint arXiv:2010.01149*, 2020.
* Bellemare et al. (2020)

  M. G. Bellemare, S. Candido, P. S. Castro, J. Gong, M. C. Machado, S. Moitra,
  S. S. Ponda, and Z. Wang.
  Autonomous navigation of stratospheric balloons using reinforcement
  learning.
  *Nature*, 588, 2020.
* Ben-David et al. (2006)

  S. Ben-David, J. Blitzer, K. Crammer, and F. Pereira.
  Analysis of representations for domain adaptation.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 137–144, 2006.
* BenTaieb and Hamarneh (2017)

  A. BenTaieb and G. Hamarneh.
  Adversarial stain transfer for histopathology image analysis.
  *IEEE Transactions on Medical Imaging*, 37(3):792–802, 2017.
* Berman et al. (2018)

  Gabrielle Berman, Sara de la Rosa, and Tanya Accone.
  Ethical considerations when using geospatial technologies for
  evidence generation.
  *Innocenti Discussion Paper, UNICEF Office of Research*, 2018.
* Beyene et al. (2015)

  A. A. Beyene, T. Welemariam, M. Persson, and N. Lavesson.
  Improved concept drift handling in surgery prediction and other
  applications.
  *Knowledge and Information Systems*, 44(1):177–196, 2015.
* Blanchard et al. (2011)

  G. Blanchard, G. Lee, and C. Scott.
  Generalizing from several related classification tasks to a new
  unlabeled sample.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 2178–2186, 2011.
* Blitzer et al. (2007)

  J. Blitzer, M. Dredze, and F. Pereira.
  Biographies, bollywood, boom-boxes and blenders: Domain adaptation
  for sentiment classification.
  In *Proceedings of the 45th Annual Meeting of the Association of
  Computational Linguistics*, pages 440–447, 2007.
* Blodgett and O’Connor (2017)

  S. L. Blodgett and B. O’Connor.
  Racial disparity in natural language processing: A case study of
  social media African-American English.
  *arXiv preprint arXiv:1707.00061*, 2017.
* Blodgett et al. (2016)

  S. L. Blodgett, L. Green, and B. O’Connor.
  Demographic dialectal variation in social media: A case study of
  African-American English.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  pages 1119–1130, 2016.
* Blumenstock et al. (2015)

  J. Blumenstock, G. Cadamuro, and R. On.
  Predicting poverty and wealth from mobile phone metadata.
  *Science*, 350, 2015.
* Bohacek et al. (1996)

  R. S. Bohacek, C. McMartin, and W. C. Guida.
  The art and practice of structure-based drug design: a molecular
  modeling perspective.
  *Medicinal Research Reviews*, 16(1):3–50,
  1996.
* Bolstad et al. (2003)

  Benjamin M Bolstad, Rafael A Irizarry, Magnus Åstrand, and Terence P.
  Speed.
  A comparison of normalization methods for high density
  oligonucleotide array data based on variance and bias.
  *Bioinformatics*, 19(2):185–193, 2003.
* Borkan et al. (2019a)

  D. Borkan, L. Dixon, J. Li, J. Sorensen, N. Thain, and L. Vasserman.
  Limitations of pinned AUC for measuring unintended bias.
  *arXiv preprint arXiv:1903.02088*, 2019a.
* Borkan et al. (2019b)

  D. Borkan, L. Dixon, J. Sorensen, N. Thain, and L. Vasserman.
  Nuanced metrics for measuring unintended bias with real data for text
  classification.
  In *WWW*, pages 491–500, 2019b.
* Bottou et al. (2013)

  L. Bottou, J. Peters, J. Quiñonero-Candela, D. X. Charles, D. M.
  Chickering, E. Portugaly, D. Ray, P. Simard, and E. Snelson.
  Counterfactual reasoning and learning systems: The example of
  computational advertising.
  *Journal of Machine Learning Research (JMLR)*, 14:3207–3260, 2013.
* Boutros et al. (2015)

  Michael Boutros, Florian Heigwer, and Christina Laufer.
  Microscopy-based high-content screening.
  *Cell*, 163(6):1314–1325, 2015.
* Boyle et al. (2008)

  Alan P Boyle, Sean Davis, Hennady P Shulha, Paul Meltzer, Elliott H Margulies,
  Zhiping Weng, Terrence S Furey, and Gregory E Crawford.
  High-resolution mapping and characterization of open chromatin across
  the genome.
  *Cell*, 132(2):311–322, 2008.
* Bray et al. (2016)

  Mark-Anthony Bray, Shantanu Singh, Han Han, Chadwick T Davis, Blake Borgeson,
  Cathy Hartland, Maria Kost-Alimova, Sigrun M Gustafsdottir, Christopher C
  Gibson, and Anne E Carpenter.
  Cell painting, a high-content image-based assay for morphological
  profiling using multiplexed fluorescent dyes.
  *Nature protocols*, 11(9):1757, 2016.
* Broach et al. (1996)

  James R Broach, Jeremy Thorner, et al.
  High-throughput screening for drug discovery.
  *Nature*, 384(6604):14–16, 1996.
* Broussard (2020)

  M. Broussard.
  When algorithms give real students imaginary grades.
  *The New York Times*, 2020.
  URL
  <https://www.nytimes.com/2020/09/08/opinion/international-baccalaureate-algorithm-grades.html>.
* Bruch et al. (2009)

  Marcel Bruch, Martin Monperrus, and Mira Mezini.
  Learning from examples to improve code completion systems.
  In *European software engineering conference and the ACM SIGSOFT
  symposium on the foundations of software engineering*, 2009.
* Bruzzone and Marconcini (2009)

  L. Bruzzone and M. Marconcini.
  Domain adaptation problems: A DASVM classification technique and a
  circular validation strategy.
  *IEEE Transactions on Pattern Analysis and Machine
  Intelligence*, 32(5):770–787, 2009.
* Buenrostro et al. (2013)

  Jason D Buenrostro, Paul G Giresi, Lisa C Zaba, Howard Y Chang, and William J
  Greenleaf.
  Transposition of native chromatin for fast and sensitive epigenomic
  profiling of open chromatin, dna-binding proteins and nucleosome position.
  *Nature methods*, 10(12):1213, 2013.
* Bug et al. (2017)

  D. Bug, S. Schneider, A. Grote, E. Oswald, F. Feuerhake, J. Schüler, and
  D. Merhof.
  Context-based normalization of histological stains using deep
  convolutional features.
  *Deep Learning in Medical Image Analysis and Multimodal Learning
  for Clinical Decision Support*, pages 135–142, 2017.
* Bunel et al. (2018)

  Rudy Bunel, Matthew Hausknecht, Jacob Devlin, Rishabh Singh, and Pushmeet
  Kohli.
  Leveraging grammar and reinforcement learning for neural program
  synthesis.
  In *International Conference on Learning Representations
  (ICLR)*, 2018.
* Buolamwini and Gebru (2018)

  J. Buolamwini and T. Gebru.
  Gender shades: Intersectional accuracy disparities in commercial
  gender classification.
  In *Conference on Fairness, Accountability and Transparency*,
  pages 77–91, 2018.
* Burke et al. (2016)

  M. Burke, S. Heft-Neal, and E. Bendavid.
  Sources of variation in under-5 mortality across sub-Saharan
  Africa: a spatial analysis.
  *Lancet Global Health*, 4, 2016.
* Byrd and Lipton (2019)

  J. Byrd and Z. Lipton.
  What is the effect of importance weighting in deep learning?
  In *International Conference on Machine Learning (ICML)*, pages
  872–881, 2019.
* Caicedo et al. (2017)

  Juan C Caicedo, Sam Cooper, Florian Heigwer, Scott Warchal, Peng Qiu, Csaba
  Molnar, Aliaksei S Vasilevich, Joseph D Barry, Harmanjit Singh Bansal, Oren
  Kraus, et al.
  Data-analysis strategies for image-based cell profiling.
  *Nature methods*, 14(9):849–863, 2017.
* Caicedo et al. (2018)

  Juan C Caicedo, Claire McQuin, Allen Goodman, Shantanu Singh, and Anne E
  Carpenter.
  Weakly supervised learning of single-cell feature embeddings.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pages 9309–9318, 2018.
* Caldas et al. (2018)

  S. Caldas, P. Wu, T. Li, J. Konečnỳ, H. B. McMahan, V. Smith, and
  A. Talwalkar.
  Leaf: A benchmark for federated settings.
  *arXiv preprint arXiv:1812.01097*, 2018.
* Campanella et al. (2019)

  G. Campanella, M. G. Hanna, L. Geneslaw, A. Miraflor, V. W. K. Silva, K. J.
  Busam, E. Brogi, V. E. Reuter, D. S. Klimstra, and T. J. Fuchs.
  Clinical-grade computational pathology using weakly supervised deep
  learning on whole slide images.
  *Nature Medicine*, 25(8):1301–1309, 2019.
* Cao et al. (2020)

  K. Cao, Y. Chen, J. Lu, N. Arechiga, A. Gaidon, and T. Ma.
  Heteroskedastic and imbalanced deep learning with adaptive
  regularization.
  *arXiv preprint arXiv:2006.15766*, 2020.
* Cao et al. (2019)

  Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma.
  Learning imbalanced datasets with label-distribution-aware margin
  loss.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019.
* Carlucci et al. (2019)

  Fabio M Carlucci, Antonio D’Innocente, Silvia Bucci, Barbara Caputo, and
  Tatiana Tommasi.
  Domain generalization by solving jigsaw puzzles.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  2229–2238, 2019.
* Chanussot et al. (2020)

  L. Chanussot, A. Das, S. Goyal, T. Lavril, M. Shuaibi, M. Riviere, K. Tran,
  J. Heras-Domingo, C. Ho, W. Hu, A. Palizhati, A. Sriram, B. Wood, J. Yoon,
  D. Parikh, C. L. Zitnick, and Z. Ulissi.
  The Open Catalyst 2020 (oc20) dataset and community challenges.
  *arXiv preprint arXiv:2010.09990*, 2020.
* Chen et al. (2019a)

  I. Y. Chen, P. Szolovits, and M. Ghassemi.
  Can AI help reduce disparities in general medical and mental health
  care?
  *AMA Journal of Ethics*, 21(2):167–179,
  2019a.
* Chen et al. (2020)

  I. Y. Chen, E. Pierson, S. Rose, S. Joshi, K. Ferryman, and M. Ghassemi.
  Ethical machine learning in health care.
  *arXiv preprint arXiv:2009.10576*, 2020.
* Chen et al. (2018)

  Irene Chen, Fredrik D Johansson, and David Sontag.
  Why is my classifier discriminatory?
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 3539–3550, 2018.
* Chen et al. (2019b)

  V. Chen, S. Wu, A. J. Ratner, J. Weng, and C. Ré.
  Slice-based learning: A programming model for residual learning in
  critical data slices.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 9397–9407, 2019b.
* Ching et al. (2018)

  T. Ching, D. S. Himmelstein, B. K. Beaulieu-Jones, A. A. Kalinin, B. T. Do,
  G. P. Way, E. Ferrero, P. Agapow, M. Zietz, M. M. Hoffman, et al.
  Opportunities and obstacles for deep learning in biology and
  medicine.
  *Journal of The Royal Society Interface*, 15(141),
  2018.
* Christie et al. (2018)

  G. Christie, N. Fendley, J. Wilson, and R. Mukherjee.
  Functional map of the world.
  In *Computer Vision and Pattern Recognition (CVPR)*, 2018.
* Chung et al. (2018)

  J. S. Chung, A. Nagrani, and A. Zisserman.
  Voxceleb2: Deep speaker recognition.
  *Proc. Interspeech*, pages 1086–1090, 2018.
* Clark et al. (2020)

  J. H. Clark, E. Choi, M. Collins, D. Garrette, T. Kwiatkowski, V. Nikolaev, and
  J. Palomaki.
  Tydi qa: A benchmark for information-seeking question answering in
  typologically diverse languages.
  *arXiv preprint arXiv:2003.05002*, 2020.
* Codella et al. (2019)

  N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman,
  B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, et al.
  Skin lesion analysis toward melanoma detection 2018: A challenge
  hosted by the international skin imaging collaboration (ISIC).
  *arXiv preprint arXiv:1902.03368*, 2019.
* Conneau and Lample (2019)

  A. Conneau and G. Lample.
  Cross-lingual language model pretraining.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 7059–7069, 2019.
* Conneau et al. (2018)

  A. Conneau, R. Rinott, G. Lample, A. Williams, S. Bowman, H. Schwenk, and
  V. Stoyanov.
  Xnli: Evaluating cross-lingual sentence representations.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  pages 2475–2485, 2018.
* Consortium et al. (2012)

  E. P. Consortium et al.
  An integrated encyclopedia of DNA elements in the human genome.
  *Nature*, 489(7414):57–74, 2012.
* Consortium et al. (2020)

  G. Consortium et al.
  The GTEx Consortium atlas of genetic regulatory effects across
  human tissues.
  *Science*, 369(6509):1318–1330, 2020.
* Consortium (2019)

  HuBMAP Consortium.
  The human body at cellular resolution: the nih human biomolecular
  atlas program.
  *Nature*, 574(7777):187, 2019.
* Corbett-Davies and Goel (2018)

  Sam Corbett-Davies and Sharad Goel.
  The measure and mismeasure of fairness: A critical review of fair
  machine learning.
  *arXiv preprint arXiv:1808.00023*, 2018.
* Corbett-Davies et al. (2016)

  Sam Corbett-Davies, Emma Pierson, Avi Feller, and Sharad Goel.
  A computer program used for bail and sentencing decisions was labeled
  biased against blacks. It’s actually not that clear.
  *Washington Post*, 2016.
  ISSN 0190-8286.
  URL
  <https://www.washingtonpost.com/news/monkey-cage/wp/2016/10/17/can-an-algorithm-be-racist-our-analysis-is-more-cautious-than-propublicas/>.
* Corbett-Davies et al. (2017)

  Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq.
  Algorithmic decision making and the cost of fairness.
  In *Proceedings of the 23rd ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining*, pages 797–806, 2017.
* Cordella et al. (1995)

  L. P. Cordella, C. D. Stefano, F. Tortorella, and M. Vento.
  A method for improving classification reliability of multilayer
  perceptrons.
  *IEEE Transactions on Neural Networks*, 6(5):1140–1147, 1995.
* Courtiol et al. (2019)

  P. Courtiol, C. Maussion, M. Moarii, E. Pronier, S. Pilcer, M. Sefta,
  P. Manceron, S. Toldo, M. Zaslavskiy, N. L. Stang, et al.
  Deep learning-based classification of mesothelioma improves
  prediction of patient outcome.
  *Nature Medicine*, 25(10):1519–1525, 2019.
* Croce et al. (2020)

  F. Croce, M. Andriushchenko, V. Sehwag, N. Flammarion, M. Chiang, P. Mittal,
  and M. Hein.
  Robustbench: a standardized adversarial robustness benchmark.
  *arXiv preprint arXiv:2010.09670*, 2020.
* Crunchant et al. (2020)

  Anne-Sophie Crunchant, David Borchers, Hjalmar Kühl, and Alex Piel.
  Listening and watching: Do camera traps or acoustic sensors more
  efficiently detect wild chimpanzees in an open habitat?
  *Methods in Ecology and Evolution*, 11(4):542–552, 2020.
* Cuccarese et al. (2020)

  M. F. Cuccarese, B. A. Earnshaw, K. Heiser, B. Fogelson, C. T. Davis, P. F.
  McLean, H. B. Gordon, K. Skelly, F. L. Weathersby, V. Rodic, et al.
  Functional immune mapping with deep-learning enabled phenomics
  applied to immunomodulatory and COVID-19 drug discovery.
  *bioRxiv*, 2020.
* Cui et al. (2019)

  Y. Cui, M. Jia, T. Lin, Y. Song, and S. Belongie.
  Class-balanced loss based on effective number of samples.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  9268–9277, 2019.
* Dai and Van Gool (2018)

  D. Dai and L. Van Gool.
  Dark model adaptation: Semantic image segmentation from daytime to
  nighttime.
  In *International Conference on Intelligent Transportation
  Systems (ITSC)*, 2018.
* D’Amour et al. (2020a)

  A. D’Amour, K. Heller, D. Moldovan, B. Adlam, B. Alipanahi, A. Beutel, C. Chen,
  J. Deaton, J. Eisenstein, M. D. Hoffman, et al.
  Underspecification presents challenges for credibility in modern
  machine learning.
  *arXiv preprint arXiv:2011.03395*, 2020a.
* D’Amour et al. (2020b)

  A. D’Amour, H. Srinivasan, J. Atwood, P. Baljekar, D. Sculley, and Y. Halpern.
  Fairness is not static: deeper understanding of long term fairness
  via simulation studies.
  In *Proceedings of the 2020 Conference on Fairness,
  Accountability, and Transparency*, pages 525–534, 2020b.
* David et al. (2020)

  Etienne David, Simon Madec, Pouria Sadeghi-Tehran, Helge Aasen, Bangyou Zheng,
  Shouyang Liu, Norbert Kirchgessner, Goro Ishikawa, Koichi Nagasawa,
  Minhajul A Badhon, Curtis Pozniak, Benoit de Solan, Andreas Hund, Scott C.
  Chapman, Frederic Baret, Ian Stavness, and Wei Guo.
  Global wheat head detection (gwhd) dataset: a large and diverse
  dataset of high-resolution rgb-labelled images to develop and benchmark wheat
  head detection methods.
  *Plant Phenomics*, 2020, 2020.
* David et al. (2021)

  Etienne David, Mario Serouart, Daniel Smith, Simon Madec, Kaaviya Velumani,
  Shouyang Liu, Xu Wang, Francisco Pinto Espinosa, Shahameh Shafiee, Izzat
  S. A. Tahir, Hisashi Tsujimoto, Shuhei Nasuda, Bangyou Zheng, Norbert
  Kichgessner, Helge Aasen, Andreas Hund, Pouria Sadhegi-Tehran, Koichi
  Nagasawa, Goro Ishikawa, Sebastien Dandrifosse, Alexis Carlier, Benoit
  Mercatoris, Ken Kuroki, Haozhou Wang, Masanori Ishii, Minhajul A. Badhon,
  Curtis Pozniak, David Shaner LeBauer, Morten Lilimo, Jesse Poland, Scott
  Chapman, Benoit de Solan, Frederic Baret, Ian Stavness, and Wei Guo.
  Global wheat head dataset 2021: an update to improve the benchmarking
  wheat head localization with more diversity, 2021.
* Davis et al. (2017)

  S. E. Davis, T. A. Lasko, G. Chen, E. D. Siew, and M. E. Matheny.
  Calibration drift in regression and machine learning models for acute
  kidney injury.
  *Journal of the American Medical Informatics Association*,
  24(6):1052–1061, 2017.
* DeGrave et al. (2020)

  A. J. DeGrave, J. D. Janizek, and S. Lee.
  AI for radiographic COVID-19 detection selects shortcuts over
  signal.
  *medRxiv*, 2020.
* Deplancke et al. (2016)

  Bart Deplancke, Daniel Alpern, and Vincent Gardeux.
  The genetics of transcription factor dna binding variation.
  *Cell*, 166(3):538–554, 2016.
* Desmarais and Baker (2012)

  M. C. Desmarais and R. Baker.
  A review of recent advances in learner and skill modeling in
  intelligent learning environments.
  *User Modeling and User-Adapted Interaction*, 22(1):9–38, 2012.
* DigitalGlobe and Works (2016)

  N. DigitalGlobe and C. Works.
  Spacenet.
  <https://aws.amazon.com/publicdatasets/spacenet/>, 2016.
* Dill and MacCallum (2012)

  K. A. Dill and J. L. MacCallum.
  The protein-folding problem, 50 years on.
  *Science*, 338(6110):1042–1046, 2012.
* Dixon et al. (2018)

  L. Dixon, J. Li, J. Sorensen, N. Thain, and L. Vasserman.
  Measuring and mitigating unintended bias in text classification.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, pages 67–73, 2018.
* Djolonga et al. (2020)

  J. Djolonga, J. Yung, M. Tschannen, R. Romijnders, L. Beyer, A. Kolesnikov,
  J. Puigcerver, M. Minderer, A. D’Amour, D. Moldovan, et al.
  On robustness and transferability of convolutional neural networks.
  *arXiv preprint arXiv:2007.08558*, 2020.
* Dodge and Karam (2017)

  Samuel Dodge and Lina Karam.
  A study and comparison of human and deep learning recognition
  performance under visual distortions.
  In *26th International Conference on Computer Communication and
  Networks (ICCCN)*, pages 1–7. IEEE, 2017.
* Dou et al. (2019)

  Q. Dou, D. Castro, K. Kamnitsas, and B. Glocker.
  Domain generalization via model-agnostic learning of semantic
  features.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019.
* Dreccer et al. (2019)

  M Fernanda Dreccer, Gemma Molero, Carolina Rivera-Amado, Carus John-Bejai, and
  Zoe Wilson.
  Yielding to the image: how phenotyping reproductive growth can assist
  crop improvement and production.
  *Plant science*, 282:73–82, 2019.
* Dressel and Farid (2018)

  Julia Dressel and Hany Farid.
  The accuracy, fairness, and limits of predicting recidivism.
  *Science Advances*, 4(1), 2018.
* Duchi and Namkoong (2021)

  John Duchi and Hongseok Namkoong.
  Learning models with uniform performance via distributionally robust
  optimization.
  *Annals of Statistics*, 2021.
* Duchi et al. (2020)

  John Duchi, Tatsunori Hashimoto, and Hongseok Namkoong.
  Distributionally robust losses for latent covariate mixtures.
  *arXiv preprint arXiv:2007.13982*, 2020.
* Dwork et al. (2012)

  C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. Zemel.
  Fairness through awareness.
  In *Innovations in Theoretical Computer Science (ITCS)*, pages
  214–226, 2012.
* Echeverri and Perrimon (2006)

  Christophe J Echeverri and Norbert Perrimon.
  High-throughput rnai screening in cultured cells: a user’s guide.
  *Nature Reviews Genetics*, 7(5):373, 2006.
* Elvidge et al. (2009)

  C. D. Elvidge, P. C. Sutton, T. Ghosh, B. T. Tuttle, K. E. Baugh, B. Bhaduri,
  and E. Bright.
  A global poverty map derived from satellite data.
  *Computers and Geosciences*, 35, 2009.
* Eraslan et al. (2019)

  G. Eraslan, Žiga Avsec, J. Gagneur, and F. J. Theis.
  Deep learning: new computational modelling techniques for genomics.
  *Nature Reviews Genetics*, 20(7):389–403,
  2019.
* Espey et al. (2015)

  J. Espey, E. Swanson, S. Badiee, Z. Chistensen, A. Fischer, M. Levy, G. Yetman,
  A. de Sherbinin, R. Chen, Y. Qiu, G. Greenwell, T. Klein, , J. Jutting,
  M. Jerven, G. Cameron, A. M. A. Rivera, V. C. Arias, , S. L. Mills, and
  A. Motivans.
  Data for development: A needs assessment for SDG monitoring and
  statistical capacity development.
  *Sustainable Development Solutions Network*, 2015.
* Esteva et al. (2017)

  A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and
  S. Thrun.
  Dermatologist-level classification of skin cancer with deep neural
  networks.
  *Nature*, 542(7639):115–118, 2017.
* et al (2019)

  OpenAI et al.
  Solving Rubik’s cube with a robot hand.
  *arXiv preprint arXiv:1910.07113*, 2019.
* Fan et al. (2018)

  Zhun Fan, Jiewei Lu, Maoguo Gong, Honghui Xie, and Erik D Goodman.
  Automatic tobacco plant detection in uav images via deep neural
  networks.
  *IEEE Journal of Selected Topics in Applied Earth Observations
  and Remote Sensing*, 11(3):876–887, 2018.
* Fang et al. (2013)

  C. Fang, Y. Xu, and D. N. Rockmore.
  Unbiased metric learning: On the utilization of multiple datasets and
  web images for softening bias.
  In *International Conference on Computer Vision (ICCV)*, pages
  1657–1664, 2013.
* Feng et al. (2019)

  J. Feng, A. Sondhi, J. Perry, and N. Simon.
  Selective prediction-set models with coverage guarantees.
  *arXiv preprint arXiv:1906.05473*, 2019.
* Filmer and Scott (2011)

  D. Filmer and K. Scott.
  Assessing asset indices.
  *Demography*, 49, 2011.
* Finlayson et al. (2021)

  Samuel G. Finlayson, Adarsh Subbaswamy, Karandeep Singh, John Bowers, Annabel
  Kupke, Jonathan Zittrain, Isaac S. Kohane, and Suchi Saria.
  The clinician and dataset shift in artificial intelligence.
  *New England Journal of Medicine*, 385(3):283–286, 2021.
* Franks et al. (2015)

  Christine Franks, Zhaopeng Tu, Premkumar Devanbu, and Vincent Hellendoorn.
  Cacheca: A cache language model based code suggestion tool.
  In *International Conference on Software Engineering (ICSE)*,
  2015.
* Fuentes et al. (2017)

  Alvaro Fuentes, Sook Yoon, Sang Cheol Kim, and Dong Sun Park.
  A robust deep-learning-based detector for real-time tomato plant
  diseases and pests recognition.
  *Sensors*, 17(9):2022, 2017.
* Futoma et al. (2020)

  J. Futoma, M. Simons, T. Panch, F. Doshi-Velez, and L. A. Celi.
  The myth of generalisability in clinical research and machine
  learning in health care.
  *The Lancet Digital Health*, 2(9):e489–e492, 2020.
* Gal and Ghahramani (2016)

  Y. Gal and Z. Ghahramani.
  Dropout as a Bayesian approximation: Representing model uncertainty
  in deep learning.
  In *International Conference on Machine Learning (ICML)*, 2016.
* Ganin and Lempitsky (2015)

  Y. Ganin and V. Lempitsky.
  Unsupervised domain adaptation by backpropagation.
  In *International Conference on Machine Learning (ICML)*, pages
  1180–1189, 2015.
* Ganin et al. (2016)

  Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette,
  M. March, and V. Lempitsky.
  Domain-adversarial training of neural networks.
  *Journal of Machine Learning Research (JMLR)*, 17, 2016.
* Garg et al. (2020)

  S. Garg, Y. Wu, S. Balakrishnan, and Z. C. Lipton.
  A unified view of label shift estimation.
  *arXiv preprint arXiv:2003.07554*, 2020.
* Geifman and El-Yaniv (2017)

  Y. Geifman and R. El-Yaniv.
  Selective classification for deep neural networks.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2017.
* Geifman and El-Yaniv (2019)

  Y. Geifman and R. El-Yaniv.
  Selectivenet: A deep neural network with an integrated reject option.
  In *International Conference on Machine Learning (ICML)*, 2019.
* Geifman et al. (2018)

  Y. Geifman, G. Uziel, and R. El-Yaniv.
  Bias-reduced uncertainty estimation for deep neural classifiers.
  In *International Conference on Learning Representations
  (ICLR)*, 2018.
* Geirhos et al. (2018a)

  R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and
  W. Brendel.
  Imagenet-trained cnns are biased towards texture; increasing shape
  bias improves accuracy and robustness.
  *arXiv preprint arXiv:1811.12231*, 2018a.
* Geirhos et al. (2018b)

  R. Geirhos, C. R. Temme, J. Rauber, H. H. Schütt, M. Bethge, and F. A.
  Wichmann.
  Generalisation in humans and deep neural networks.
  *Advances in Neural Information Processing Systems*,
  31:7538–7550, 2018b.
* Geirhos et al. (2020)

  R. Geirhos, J. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and
  F. A. Wichmann.
  Shortcut learning in deep neural networks.
  *arXiv preprint arXiv:2004.07780*, 2020.
* Gelman et al. (2007)

  Andrew Gelman, Jeffrey Fagan, and Alex Kiss.
  An Analysis of the New York City Police Department’s
  “Stop-and-Frisk” Policy in the Context of Claims of Racial
  Bias.
  *Journal of the American Statistical Association*, 102(479):813–823, Sep 2007.
  ISSN 0162-1459.
  doi: 10.1198/016214506000001040.
  URL
  <https://amstat.tandfonline.com/doi/abs/10.1198/016214506000001040>.
  Publisher: Taylor & Francis.
* Geva et al. (2019)

  M. Geva, Y. Goldberg, and J. Berant.
  Are we modeling the task or the annotator? an investigation of
  annotator bias in natural language understanding datasets.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  2019.
* Gilmer et al. (2017)

  Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and
  George E Dahl.
  Neural message passing for quantum chemistry.
  In *International Conference on Machine Learning (ICML)*, pages
  1273–1272, 2017.
* Godinez et al. (2018)

  William J Godinez, Imtiaz Hossain, and Xian Zhang.
  Unsupervised phenotypic analysis of cellular images with multi-scale
  convolutional neural networks.
  *BioRxiv*, page 361410, 2018.
* Goel et al. (2020)

  K. Goel, A. Gu, Y. Li, and C. Ré.
  Model patching: Closing the subgroup performance gap with data
  augmentation.
  *arXiv preprint arXiv:2008.06775*, 2020.
* Goel et al. (2016)

  Sharad Goel, Justin M. Rao, and Ravi Shroff.
  Precinct or prejudice? Understanding racial disparities in New
  York City’s stop-and-frisk policy.
  *The Annals of Applied Statistics*, 10(1):365–394, March 2016.
  ISSN 1932-6157.
  doi: 10.1214/15-AOAS897.
  URL <http://projecteuclid.org/euclid.aoas/1458909920>.
* Gogoll et al. (2020)

  Dario Gogoll, Philipp Lottes, Jan Weyler, Nik Petrinic, and Cyrill Stachniss.
  Unsupervised domain adaptation for transferring plant classification
  systems to new field environments, crops, and robots.
  In *2020 IEEE/RSJ International Conference on Intelligent Robots
  and Systems (IROS)*, pages 2636–2642. IEEE, 2020.
* Goh et al. (2017)

  Wilson Wen Bin Goh, Wei Wang, and Limsoon Wong.
  Why batch effects matter in omics data, and how to avoid them.
  *Trends in biotechnology*, 35(6):498–507,
  2017.
* Goodfellow et al. (2015)

  I. J. Goodfellow, J. Shlens, and C. Szegedy.
  Explaining and harnessing adversarial examples.
  In *International Conference on Learning Representations
  (ICLR)*, 2015.
* Graetz et al. (2018)

  N. Graetz, J. Friedman, A. Osgood-Zimmerman, R. Burstein, M. H. Biehl,
  C. Shields, J. F. Mosser, D. C. Casey, A. Deshpande, L. Earl, R. C. Reiner,
  S. E. Ray, N. Fullman, A. J. Levine, R. W. Stubbs, B. K. Mayala,
  J. Longbottom, A. J. Browne, S. Bhatt, D. J. Weiss, P. W. Gething, A. H.
  Mokdad, S. S. Lim, C. J. L. Murray, E. Gakidou, and S. I. Hay.
  Mapping local variation in educational attainment across Africa.
  *Nature*, 555, 2018.
* Grandori et al. (2000)

  Carla Grandori, Shaun M Cowley, Leonard P James, and Robert N Eisenman.
  The myc/max/mad network and the transcriptional control of cell
  behavior.
  *Annual review of cell and developmental biology*, 16(1):653–699, 2000.
* Grooten et al. (2020)

  M Grooten, T Peterson, and R.E.A Almond.
  *Living Planet Report 2020 - Bending the curve of biodiversity
  loss*.
  WWF, Gland, Switzerland, 2020.
* Gu et al. (2017)

  S. Gu, E. Holly, T. Lillicrap, and S. Levine.
  Deep reinforcement learning for robotic manipulation with
  asynchronous off-policy updates.
  In *International Conference on Robotics and Automation (ICRA)*,
  2017.
* Gulrajani and Lopez-Paz (2020)

  I. Gulrajani and D. Lopez-Paz.
  In search of lost domain generalization.
  *arXiv preprint arXiv:2007.01434*, 2020.
* Guo et al. (2018)

  Jiang Guo, Darsh J Shah, and Regina Barzilay.
  Multi-source domain adaptation with mixture of experts.
  *arXiv preprint arXiv:1809.02256*, 2018.
* Gupta et al. (2018)

  A. Gupta, A. Murali, D. Gandhi, and L. Pinto.
  Robot learning in homes: Improving generalization and reducing
  dataset bias.
  In *Advances in Neural Information Processing Systems (NIPS)*,
  2018.
* Gurcan et al. (2009)

  M. N. Gurcan, L. E. Boucheron, A. Can, A. Madabhushi, N. M. Rajpoot, and
  B. Yener.
  Histopathological image analysis: A review.
  *IEEE reviews in biomedical engineering*, 2:147–171,
  2009.
* Han and Tsvetkov (2020)

  X. Han and Y. Tsvetkov.
  Fortifying toxic speech detectors against veiled toxicity.
  *arXiv preprint arXiv:2010.03154*, 2020.
* Hand (2006)

  David J Hand.
  Classifier technology and the illusion of progress.
  *Statistical science*, pages 1–14, 2006.
* Hansen et al. (2013)

  M. C. Hansen, P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova,
  A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland,
  A. Kommareddy, A. Egorov, L. Chini, C. O. Justice, and J. R. G. Townshend.
  High-resolution global maps of 21st-century forest cover change.
  *Science*, 342, 2013.
* Harrill et al. (2019)

  Joshua Harrill, Imran Shah, R. Woodrow Setzer, Derik Haggard, Scott Auerbach,
  Richard Judson, and Russell S. Thomas.
  Considerations for strategic use of high-throughput transcriptomics
  chemical screening data in regulatory decisions.
  *Current Opinion in Toxicology*, 15:64–75, 2019.
  ISSN 2468-2020.
  doi: https://doi.org/10.1016/j.cotox.2019.05.004.
  URL
  <https://www.sciencedirect.com/science/article/pii/S2468202019300129>.
  Risk Assessment in Toxicology.
* Hashimoto et al. (2018)

  T. B. Hashimoto, M. Srivastava, H. Namkoong, and P. Liang.
  Fairness without demographics in repeated loss minimization.
  In *International Conference on Machine Learning (ICML)*, 2018.
* He et al. (2016)

  K. He, X. Zhang, S. Ren, and J. Sun.
  Deep residual learning for image recognition.
  In *Computer Vision and Pattern Recognition (CVPR)*, 2016.
* He et al. (2020)

  Y. He, Z. Shen, and P. Cui.
  Towards non-IID image classification: A dataset and baselines.
  *Pattern Recognition*, 110, 2020.
* Heinze-Deml and Meinshausen (2017)

  Christina Heinze-Deml and Nicolai Meinshausen.
  Conditional variance penalties and domain shift robustness.
  *arXiv preprint arXiv:1710.11469*, 2017.
* Hellendoorn et al. (2019)

  Vincent J Hellendoorn, Sebastian Proksch, Harald C Gall, and Alberto Bacchelli.
  When code completion fails: A case study on real-world completions.
  In *2019 IEEE/ACM 41st International Conference on Software
  Engineering (ICSE)*, pages 960–970. IEEE, 2019.
* Henderson et al. (2012)

  B. E. Henderson, N. H. Lee, V. Seewaldt, and H. Shen.
  The influence of race and ethnicity on the biology of cancer.
  *Nature Reviews Cancer*, 12(9):648–653,
  2012.
* Hendrycks and Dietterich (2019)

  D. Hendrycks and T. Dietterich.
  Benchmarking neural network robustness to common corruptions and
  perturbations.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Hendrycks and Gimpel (2017)

  D. Hendrycks and K. Gimpel.
  A baseline for detecting misclassified and out-of-distribution
  examples in neural networks.
  In *International Conference on Learning Representations
  (ICLR)*, 2017.
* Hendrycks et al. (2020a)

  D. Hendrycks, S. Basart, M. Mazeika, M. Mostajabi, J. Steinhardt, and D. Song.
  Scaling out-of-distribution detection for real-world settings.
  *arXiv preprint arXiv:1911.11132*, 2020a.
* Hendrycks et al. (2020b)

  D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai,
  T. Zhu, S. Parajuli, M. Guo, D. Song, J. Steinhardt, and J. Gilmer.
  The many faces of robustness: A critical analysis of
  out-of-distribution generalization.
  *arXiv preprint arXiv:2006.16241*, 2020b.
* Hendrycks et al. (2020c)

  Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and
  Dawn Song.
  Pretrained transformers improve out-of-distribution robustness.
  *arXiv preprint arXiv:2004.06100*, 2020c.
* Ho et al. (2014)

  J. W. Ho, Y. L. Jung, T. Liu, B. H. Alver, S. Lee, K. Ikegami, K. Sohn,
  A. Minoda, M. Y. Tolstorukov, A. Appert, et al.
  Comparative analysis of metazoan chromatin organization.
  *Nature*, 512(7515):449–452, 2014.
* Hoffman et al. (2018)

  J. Hoffman, E. Tzeng, T. Park, J. Zhu, P. Isola, K. Saenko, A. A. Efros, and
  T. Darrell.
  Cycada: Cycle consistent adversarial domain adaptation.
  In *International Conference on Machine Learning (ICML)*, 2018.
* Hovy and Spruit (2016)

  D. Hovy and S. L. Spruit.
  The social impact of natural language processing.
  In *Association for Computational Linguistics (ACL)*, pages
  591–598, 2016.
* Hu et al. (2020a)

  J. Hu, S. Ruder, A. Siddhant, G. Neubig, O. Firat, and M. Johnson.
  Xtreme: A massively multilingual multi-task benchmark for evaluating
  cross-lingual generalization.
  *arXiv preprint arXiv:2003.11080*, 2020a.
* Hu et al. (2018)

  W. Hu, G. Niu, I. Sato, and M. Sugiyama.
  Does distributionally robust supervised learning give robust
  classifiers?
  In *International Conference on Machine Learning (ICML)*, 2018.
* Hu et al. (2020b)

  W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec.
  Open graph benchmark: Datasets for machine learning on graphs.
  *arXiv preprint arXiv:2005.00687*, 2020b.
* Huang et al. (2017)

  G. Huang, Z. Liu, L. V. D. Maaten, and K. Q. Weinberger.
  Densely connected convolutional networks.
  In *Proceedings of the IEEE conference on Computer Vision and
  Pattern Recognition*, pages 4700–4708, 2017.
* Hughes et al. (2011)

  James P Hughes, Stephen Rees, S Barrett Kalindjian, and Karen L Philpott.
  Principles of early drug discovery.
  *British journal of pharmacology*, 162(6):1239–1249, 2011.
* Husain et al. (2019)

  Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
  Brockschmidt.
  Codesearchnet challenge: Evaluating the state of semantic code
  search.
  *arXiv preprint arXiv:1909.09436*, 2019.
* Jaganathan et al. (2019)

  K. Jaganathan, S. K. Panagiotopoulou, J. F. McRae, S. F. Darbandi, D. Knowles,
  Y. I. Li, J. A. Kosmicki, J. Arbelaez, W. Cui, G. B. Schwartz, et al.
  Predicting splicing from primary sequence with deep learning.
  *Cell*, 176(3):535–548, 2019.
* Jean et al. (2016)

  N. Jean, M. Burke, M. Xie, W. M. Davis, D. B. Lobell, and S. Ermon.
  Combining satellite imagery and machine learning to predict poverty.
  *Science*, 353, 2016.
* Jean et al. (2018)

  N. Jean, S. M. Xie, and S. Ermon.
  Semi-supervised deep kernel learning: Regression with unlabeled data
  by minimizing predictive variance.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2018.
* Jin et al. (2020)

  W. Jin, R. Barzilay, and T. Jaakkola.
  Enforcing predictive invariance across structured biomedical domains.
  *arXiv preprint arXiv:2006.03908*, 2020.
* Johnson et al. (2016)

  Alistair EW Johnson, Tom J Pollard, Lu Shen, H Lehman Li-Wei, Mengling Feng,
  Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and
  Roger G Mark.
  Mimic-iii, a freely accessible critical care database.
  *Scientific Data*, 3(1):1–9, 2016.
* Johnson et al. (2017)

  J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick, and
  R. Girshick.
  Clevr: A diagnostic dataset for compositional language and elementary
  visual reasoning.
  In *Computer Vision and Pattern Recognition (CVPR)*, 2017.
* Jones et al. (2021)

  Erik Jones, Shiori Sagawa, Pang Wei Koh, Ananya Kumar, and Percy Liang.
  Selective classification can magnify disparities across groups.
  In *International Conference on Learning Representations
  (ICLR)*, 2021.
* Jumper et al. (2020)

  J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, K. Tunyasuvunakool,
  O. Ronneberger, R. Bates, A. Žídek, A. Bridgland, C. Meyer, S. A A Kohl,
  A. Potapenko, A. J Ballard, A. Cowie, B. Romera-Paredes, S. Nikolov, R. Jain,
  J. Adler, T. Back, S. Petersen, D. Reiman, M. Steinegger, M. Pacholska,
  D. Silver, O. Vinyals, A. W Senior, K. Kavukcuoglu, P. Kohli, and
  D. Hassabis.
  High accuracy protein structure prediction using deep learning.
  *Fourteenth Critical Assessment of Techniques for Protein
  Structure Prediction*, 2020.
* Jung et al. (2020)

  Jongbin Jung, Sharad Goel, Jennifer Skeem, et al.
  The limits of human predictions of recidivism.
  *Science Advances*, 6(7), 2020.
* Jørgensen et al. (2015)

  A. K. Jørgensen, D. Hovy, and A. Søgaard.
  Challenges of studying and processing dialects in social media.
  In *ACL Workshop on Noisy User-generated Text*, pages 9–18,
  2015.
* Kahn et al. (2020)

  G. Kahn, P. Abbeel, and S. Levine.
  BADGR: An autonomous self-supervised learning-based navigation
  system.
  *arXiv preprint arXiv:2002.05700*, 2020.
* Kallus and Zhou (2018)

  Nathan Kallus and Angela Zhou.
  Residual Unfairness in Fair Machine Learning from
  Prejudiced Data.
  *arXiv:1806.02887 [cs, stat]*, June 2018.
  URL <http://arxiv.org/abs/1806.02887>.
  arXiv: 1806.02887.
* Kamath et al. (2020)

  A. Kamath, R. Jia, and P. Liang.
  Selective question answering under domain shift.
  In *Association for Computational Linguistics (ACL)*, 2020.
* Katona et al. (2018)

  Z. Katona, M. Painter, P. N. Patatoukas, and J. Zeng.
  On the capital market consequences of alternative data: Evidence from
  outer space.
  *Miami Behavioral Finance Conference*, 2018.
* Kaushik et al. (2019)

  D. Kaushik, E. Hovy, and Z. Lipton.
  Learning the difference that makes a difference with
  counterfactually-augmented data.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Kearns et al. (2018)

  M. Kearns, S. Neel, A. Roth, and Z. S. Wu.
  Preventing fairness gerrymandering: Auditing and learning for
  subgroup fairness.
  In *International Conference on Machine Learning (ICML)*, pages
  2564–2572, 2018.
* Keilwagen et al. (2019)

  J. Keilwagen, S. Posch, and J. Grau.
  Accurate prediction of cell type-specific transcription factor
  binding.
  *Genome Biology*, 20(1), 2019.
* Kelley et al. (2016)

  D. R. Kelley, J. Snoek, and J. L. Rinn.
  Basset: learning the regulatory code of the accessible genome with
  deep convolutional neural networks.
  *Genome Research*, 26(7):990–999, 2016.
* Kim et al. (2016a)

  J. H. Kim, M. Xie, N. Jean, and S. Ermon.
  Incorporating spatial context and fine-grained detail from satellite
  imagery to predict poverty.
  *Stanford University*, 2016a.
* Kim and Linzen (2020)

  N. Kim and T. Linzen.
  Cogs: A compositional generalization challenge based on semantic
  interpretation.
  *arXiv preprint arXiv:2010.05465*, 2020.
* Kim et al. (2016b)

  S. Kim, P. A. Thiessen, E. E. Bolton, J. Chen, G. Fu, A. Gindulyte, L. Han,
  J. He, S. He, B. A. Shoemaker, J. Wang, B. Yu, J. Zhang, and S. H. Bryant.
  Pubchem substance and compound databases.
  *Nucleic Acids Research*, 44(D1):D1202–D1213, 2016b.
* Kingma and Ba (2015)

  D. P. Kingma and J. Ba.
  Adam: A method for stochastic optimization.
  In *International Conference on Learning Representations
  (ICLR)*, 2015.
* Koenecke et al. (2020)

  A. Koenecke, A. Nam, E. Lake, J. Nudell, M. Quartey, Z. Mengesha, C. Toups,
  J. R. Rickford, D. Jurafsky, and S. Goel.
  Racial disparities in automated speech recognition.
  *Science*, 117(14):7684–7689, 2020.
* Koh et al. (2020)

  P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and
  P. Liang.
  Concept bottleneck models.
  In *International Conference on Machine Learning (ICML)*, 2020.
* Kompa et al. (2020)

  B. Kompa, J. Snoek, and A. Beam.
  Empirical frequentist coverage of deep learning uncertainty
  quantification procedures.
  *arXiv preprint arXiv:2010.03039*, 2020.
* Komura and Ishikawa (2018)

  D. Komura and S. Ishikawa.
  Machine learning methods for histopathological image analysis.
  *Computational and Structural Biotechnology Journal*,
  16:34–42, 2018.
* Kulal et al. (2019)

  Sumith Kulal, Panupong Pasupat, Kartik Chandra, Mina Lee, Oded Padon, Alex
  Aiken, and Percy S Liang.
  Spoc: Search-based pseudocode to code.
  In *Advances in Neural Information Processing Systems*, pages
  11906–11917, 2019.
* Kulkarni et al. (2015)

  C. Kulkarni, P. W. Koh, H. Huy, D. Chia, K. Papadopoulos, J. Cheng, D. Koller,
  and S. R. Klemmer.
  Peer and self assessment in massive online classes.
  *Design Thinking Research*, pages 131–168, 2015.
* Kulkarni et al. (2014)

  C. E. Kulkarni, R. Socher, M. S. Bernstein, and S. R. Klemmer.
  Scaling short-answer grading by combining peer assessment with
  algorithmic scoring.
  In *Proceedings of the first ACM conference on Learning@Scale
  conference*, pages 99–108, 2014.
* Kumar et al. (2020)

  A. Kumar, T. Ma, and P. Liang.
  Understanding self-training for gradual domain adaptation.
  In *International Conference on Machine Learning (ICML)*, 2020.
* Kundaje et al. (2015)

  A. Kundaje, W. Meuleman, J. Ernst, M. Bilenky, A. Yen, A. Heravi-Moussavi,
  P. Kheradpour, Z. Zhang, J. Wang, M. J. Ziller, et al.
  Integrative analysis of 111 reference human epigenomes.
  *Nature*, 518(7539):317–330, 2015.
* Kuznichov et al. (2019)

  Dmitry Kuznichov, Alon Zvirin, Yaron Honen, and Ron Kimmel.
  Data augmentation for leaf segmentation and counting tasks in rosette
  plants.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition Workshops*, pages 0–0, 2019.
* Lake and Baroni (2018)

  B. Lake and M. Baroni.
  Generalization without systematicity: On the compositional skills of
  sequence-to-sequence recurrent networks.
  In *International Conference on Machine Learning (ICML)*, 2018.
* Lakshminarayanan et al. (2017)

  B. Lakshminarayanan, A. Pritzel, and C. Blundell.
  Simple and scalable predictive uncertainty estimation using deep
  ensembles.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2017.
* Landrum et al. (2006)

  Greg Landrum et al.
  Rdkit: Open-source cheminformatics, 2006.
* Landt et al. (2012)

  Stephen G Landt, Georgi K Marinov, Anshul Kundaje, Pouya Kheradpour, Florencia
  Pauli, Serafim Batzoglou, Bradley E Bernstein, Peter Bickel, James B Brown,
  Philip Cayting, et al.
  Chip-seq guidelines and practices of the encode and modencode
  consortia.
  *Genome research*, 22(9):1813–1831, 2012.
* Larrazabal et al. (2020)

  Agostina J Larrazabal, Nicolás Nieto, Victoria Peterson, Diego H Milone,
  and Enzo Ferrante.
  Gender imbalance in medical imaging datasets produces biased
  classifiers for computer-aided diagnosis.
  *Proceedings of the National Academy of Sciences*, 117(23):12592–12594, 2020.
* Larson et al. (2016)

  Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin.
  How we analyzed the compas recidivism algorithm.
  *ProPublica*, 9(1), 2016.
* Latessa et al. (2010)

  Edward J. Latessa, Richard Lemke, Matthew Makarios, and Paula Smith.
  The Creation and Validation of the Ohio Risk Assessment
  System (ORAS).
  *Federal Probation*, 74:16, 2010.
  URL
  <https://heinonline.org/HOL/Page?handle=hein.journals/fedpro74&id=16&div=&collection=>.
* Lau et al. (2014)

  R. Y. Lau, C. Li, and S. S. Liao.
  Social analytics: Learning fuzzy product ontologies for
  aspect-oriented sentiment analysis.
  *Decision Support Systems*, 65:80–94, 2014.
* LeCun et al. (1998)

  Y. LeCun, C. Cortes, and C. J. Burges.
  The MNIST database of handwritten digits.
  *<http://yann.lecun.com/exdb/mnist/>*, 1998.
* Lee et al. (2004)

  Cheol-Koo Lee, Yoichiro Shibata, Bhargavi Rao, Brian D Strahl, and Jason D
  Lieb.
  Evidence for nucleosome depletion at active regulatory regions
  genome-wide.
  *Nature genetics*, 36(8):900–905, 2004.
* Leek et al. (2010)

  J. T. Leek, R. B. Scharpf, H. C. Bravo, D. Simcha, B. Langmead, W. E. Johnson,
  D. Geman, K. Baggerly, and R. A. Irizarry.
  Tackling the widespread and critical impact of batch effects in
  high-throughput data.
  *Nature Reviews Genetics*, 11(10), 2010.
* Li et al. (2017a)

  D. Li, Y. Yang, Y. Song, and T. M. Hospedales.
  Deeper, broader and artier domain generalization.
  In *Proceedings of the IEEE International Conference on Computer
  Vision*, pages 5542–5550, 2017a.
* Li et al. (2018a)

  Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy Hospedales.
  Learning to generalize: Meta-learning for domain generalization.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, 2018a.
* Li and Guan (2019)

  H. Li and Y. Guan.
  Leopard: fast decoding cell type-specific transcription factor
  binding landscape at single-nucleotide resolution.
  *bioRxiv*, 2019.
* Li et al. (2019a)

  H. Li, D. Quang, and Y. Guan.
  Anchor: trans-cell type prediction of transcription factor binding
  sites.
  *Genome Research*, 29(2):281–292,
  2019a.
* Li et al. (2018b)

  Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot.
  Domain generalization with adversarial feature learning.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  5400–5409, 2018b.
* Li et al. (2017b)

  J. Li, A. H. Miller, S. Chopra, M. Ranzato, and J. Weston.
  Dialogue learning with human-in-the-loop.
  In *International Conference on Learning Representations
  (ICLR)*, 2017b.
* Li et al. (2019b)

  T. Li, M. Sanjabi, A. Beirami, and V. Smith.
  Fair resource allocation in federated learning.
  *arXiv preprint arXiv:1905.10497*, 2019b.
* Li et al. (2017c)

  Y. Li, N. Wang, J. Shi, J. Liu, and X. Hou.
  Revisiting batch normalization for practical domain adaptation.
  In *International Conference on Learning Representations
  Workshop (ICLRW)*, 2017c.
* Li et al. (2018c)

  Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and
  Dacheng Tao.
  Deep domain generalization via conditional invariant adversarial
  networks.
  In *European Conference on Computer Vision (ECCV)*, pages
  624–639, 2018c.
* Liang et al. (2018)

  S. Liang, Y. Li, and R. Srikant.
  Enhancing the reliability of out-of-distribution image detection in
  neural networks.
  In *International Conference on Learning Representations
  (ICLR)*, 2018.
* Libbrecht and Noble (2015)

  M. W. Libbrecht and W. S. Noble.
  Machine learning applications in genetics and genomics.
  *Nature Reviews Genetics*, 16(6):321–332,
  2015.
* Lipton et al. (2018)

  Z. Lipton, Y. Wang, and A. Smola.
  Detecting and correcting for label shift with black box predictors.
  In *International Conference on Machine Learning (ICML)*, 2018.
* Liu et al. (2021a)

  Evan Liu, Behzad Haghgoo, Annie Chen, Aditi Raghunathan, Pang Wei Koh, Shiori
  Sagawa, Percy Liang, and Chelsea Finn.
  Just train twice: Improving group robustness without training group
  information.
  In *International Conference on Machine Learning (ICML)*,
  2021a.
* Liu et al. (2018)

  L. T. Liu, S. Dean, E. Rolf, M. Simchowitz, and M. Hardt.
  Delayed impact of fair machine learning.
  In *International Conference on Machine Learning (ICML)*, 2018.
* Liu et al. (2021b)

  Nelson F. Liu, Tony Lee, Robin Jia, and Percy Liang.
  Can small and synthetic benchmarks drive modeling innovation? a
  retrospective study of question answering modeling approaches.
  *arXiv*, 2021b.
* Liu et al. (2017)

  Y. Liu, K. Gadepalli, M. Norouzi, G. E. Dahl, T. Kohlberger, A. Boyko,
  S. Venugopalan, A. Timofeev, P. Q. Nelson, G. S. Corrado, et al.
  Detecting cancer metastases on gigapixel pathology images.
  *arXiv preprint arXiv:1703.02442*, 2017.
* Ljosa et al. (2012)

  Vebjorn Ljosa, Katherine L Sokolnicki, and Anne E Carpenter.
  Annotated high-throughput microscopy image sets for validation.
  *Nature methods*, 9(7):637–637, 2012.
* Long et al. (2015)

  M. Long, Y. Cao, J. Wang, and M. Jordan.
  Learning transferable features with deep adaptation networks.
  In *International Conference on Machine Learning*, pages
  97–105, 2015.
* Loshchilov and Hutter (2019)

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Lu et al. (2021)

  Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio
  Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong
  Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan,
  Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu.
  Codexglue: A machine learning benchmark dataset for code
  understanding and generation.
  *arXiv preprint arXiv:2102.04664*, 2021.
* Lum and Isaac (2016)

  Kristian Lum and William Isaac.
  To predict and serve?
  *Significance*, 13(5):14–19, 2016.
* Lum and Shah (2019)

  Kristian Lum and Tarak Shah.
  Measures of fairness for New York City’s Supervised Release Risk
  Assessment Tool.
  *Human Rights Data Analytics Group*, page 21, 2019.
* Lyu et al. (2019)

  J. Lyu, S. Wang, T. E. Balius, I. Singh, A. Levit, Y. S. Moroz, M. J.
  O’Meara, T. Che, E. Algaa, K. Tolmachova, et al.
  Ultra-large library docking for discovering new chemotypes.
  *Nature*, 566(7743):224–229, 2019.
* Macarron et al. (2011)

  Ricardo Macarron, Martyn N Banks, Dejan Bojanic, David J Burns, Dragan A
  Cirovic, Tina Garyantes, Darren VS Green, Robert P Hertzberg, William P
  Janzen, Jeff W Paslay, et al.
  Impact of high-throughput screening in biomedical research.
  *Nature reviews Drug discovery*, 10(3):188,
  2011.
* Macenko et al. (2009)

  M. Macenko, M. Niethammer, J. S. Marron, D. Borland, J. T. Woosley, X. Guan,
  C. Schmitt, and N. E. Thomas.
  A method for normalizing histology slides for quantitative analysis.
  In *2009 IEEE International Symposium on Biomedical Imaging:
  From Nano to Macro*, pages 1107–1110, 2009.
* Madec et al. (2019)

  Simon Madec, Xiuliang Jin, Hao Lu, Benoit De Solan, Shouyang Liu, Florent
  Duyme, Emmanuelle Heritier, and Frederic Baret.
  Ear density estimation from high resolution rgb imagery using deep
  learning technique.
  *Agricultural and forest meteorology*, 264:225–234,
  2019.
* Malloy and Power (2017)

  Brian A Malloy and James F Power.
  Quantifying the transition from python 2 to 3: an empirical study of
  python applications.
  In *2017 ACM/IEEE International Symposium on Empirical Software
  Engineering and Measurement (ESEM)*, pages 314–323. IEEE, 2017.
* Mansour et al. (2009)

  Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh.
  Domain adaptation with multiple sources.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, pages 1041–1048, 2009.
* Marcus et al. (1993)

  M. P. Marcus, M. A. Marcinkiewicz, and B. Santorini.
  Building a large annotated corpus of English: the Penn
  Treebank.
  *Computational Linguistics*, 19:313–330, 1993.
* McCloskey et al. (2020)

  K. McCloskey, E. A. Sigel, S. Kearnes, L. Xue, X. Tian, D. Moccia, D. Gikunju,
  S. Bazzaz, B. Chan, M. A. Clark, et al.
  Machine learning on DNA-encoded libraries: A new paradigm for hit
  finding.
  *Journal of Medicinal Chemistry*, 2020.
* McCoy et al. (2019a)

  R. T. McCoy, J. Min, and T. Linzen.
  Berts of a feather do not generalize together: Large variability in
  generalization across models with similar test set performance.
  *arXiv preprint arXiv:1911.02969*, 2019a.
* McCoy et al. (2019b)

  R. T. McCoy, E. Pavlick, and T. Linzen.
  Right for the wrong reasons: Diagnosing syntactic heuristics in
  natural language inference.
  In *Association for Computational Linguistics (ACL)*,
  2019b.
* McKinney et al. (2020)

  S. M. McKinney, M. Sieniek, V. Godbole, J. Godwin, N. Antropova, H. Ashrafian,
  T. Back, M. Chesus, G. C. Corrado, A. Darzi, et al.
  International evaluation of an AI system for breast cancer
  screening.
  *Nature*, 577(7788):89–94, 2020.
* Mehrabi et al. (2019)

  Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram
  Galstyan.
  A survey on bias and fairness in machine learning.
  *arXiv preprint arXiv:1908.09635*, 2019.
* Meinshausen and Bühlmann (2015)

  Nicolai Meinshausen and Peter Bühlmann.
  Maximin effects in inhomogeneous large-scale data.
  *Annals of Statistics*, 43, 2015.
* Miller et al. (2020)

  J. Miller, K. Krauth, B. Recht, and L. Schmidt.
  The effect of natural distribution shift on question answering
  models.
  *arXiv preprint arXiv:2004.14444*, 2020.
* Miller et al. (2021)

  John Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh,
  Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt.
  Accuracy on the line: on the strong correlation between
  out-of-distribution and in-distribution generalization.
  In *International Conference on Machine Learning (ICML)*, 2021.
* Minnoye et al. (2021)

  Liesbeth Minnoye, Georgi K Marinov, Thomas Krausgruber, Lixia Pan, Alexandre P
  Marand, Stefano Secchia, William J Greenleaf, Eileen EM Furlong, Keji Zhao,
  Robert J Schmitz, et al.
  Chromatin accessibility profiling methods.
  *Nature Reviews Methods Primers*, 1(1):1–24, 2021.
* Mirowski et al. (2017)

  P. Mirowski, R. Pascanu, F. Viola, H. Soyer, A. Ballard, A. Banino, M. Denil,
  R. Goroshin, L. Sifre, C. Kavukcuoglu, D. Kumaran, and R. Hadsell.
  Learning to navigate in complex environments.
  In *International Conference on Learning Representations
  (ICLR)*, 2017.
* Moore et al. (2020)

  J. E. Moore, M. J. Purcaro, H. E. Pratt, C. B. Epstein, N. Shoresh, J. Adrian,
  T. Kawli, C. A. Davis, A. Dobin, R. Kaul, et al.
  Expanded encyclopaedias of DNA elements in the human and mouse
  genomes.
  *Nature*, 583(7818):699–710, 2020.
* Moult et al. (1995)

  J. Moult, J. T Pedersen, R. Judson, and K. Fidelis.
  A large-scale experiment to assess protein structure prediction
  methods.
  *Proteins: Structure, Function, and Bioinformatics*, 23(3):ii–iv, 1995.
* Nam et al. (2020)

  Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin.
  Learning from failure: Training debiased classifier from biased
  classifier.
  *arXiv preprint arXiv:2007.02561*, 2020.
* Nekoto et al. (2020)

  W. Nekoto, V. Marivate, T. Matsila, T. Fasubaa, T. Kolawole, T. Fagbohungbe,
  S. O. Akinola, S. H. Muhammad, S. Kabongo, S. Osei, S. Freshia, R. A.
  Niyongabo, R. Macharm, P. Ogayo, O. Ahia, M. Meressa, M. Adeyemi,
  M. Mokgesi-Selinga, L. Okegbemi, L. J. Martinus, K. Tajudeen, K. Degila,
  K. Ogueji, K. Siminyu, J. Kreutzer, J. Webster, J. T. Ali, J. Abbott,
  I. Orife, I. Ezeani, I. A. Dangana, H. Kamper, H. Elsahar, G. Duru, G. Kioko,
  E. Murhabazi, E. van Biljon, D. Whitenack, C. Onyefuluchi, C. Emezue,
  B. Dossou, B. Sibanda, B. I. Bassey, A. Olabiyi, A. Ramkilowan, A. Öktem,
  A. Akinfaderin, and A. Bashir.
  Participatory research for low-resourced machine translation: A case
  study in African languages.
  In *Findings of Empirical Methods in Natural Language Processing
  (Findings of EMNLP)*, 2020.
* Nestor et al. (2019)

  B. Nestor, M. McDermott, W. Boag, G. Berner, T. Naumann, M. C. Hughes,
  A. Goldenberg, and M. Ghassemi.
  Feature robustness in non-stationary health records: caveats to
  deployable model performance in common clinical machine learning tasks.
  *arXiv preprint arXiv:1908.00690*, 2019.
* Nguyen and Nguyen (2015)

  Anh Tuan Nguyen and Tien N Nguyen.
  Graph-based statistical language model for code.
  In *International Conference on Software Engineering (ICSE)*,
  2015.
* Ni et al. (2019)

  J. Ni, J. Li, and J. McAuley.
  Justifying recommendations using distantly-labeled reviews and
  fine-grained aspects.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  pages 188–197, 2019.
* Nita and Notkin (2010)

  Marius Nita and David Notkin.
  Using twinning to adapt programs to alternative apis.
  In *2010 ACM/IEEE 32nd International Conference on Software
  Engineering*, volume 1, pages 205–214. IEEE, 2010.
* Noor et al. (2008)

  A. Noor, V. Alegana, P. Gething, A. Tatem, and R. Snow.
  Using remotely sensed night-time light as a proxy for poverty in
  Africa.
  *Population Health Metrics*, 6, 2008.
* Norouzzadeh et al. (2019)

  Mohammad Sadegh Norouzzadeh, Dan Morris, Sara Beery, Neel Joshi, Nebojsa Jojic,
  and Jeff Clune.
  A deep active learning system for species identification and counting
  in camera trap images.
  *arXiv preprint arXiv:1910.09716*, 2019.
* Nygaard et al. (2016)

  Vegard Nygaard, Einar Andreas Rødland, and Eivind Hovig.
  Methods that remove batch effects while retaining group differences
  may lead to exaggerated confidence in downstream analyses.
  *Biostatistics*, 17(1):29–39, 2016.
* NYTimes (2016)

  NYTimes.
  The Times is partnering with Jigsaw to expand comment
  capabilities.
  *The New York Times*, 2016.
  URL
  <https://www.nytco.com/press/the-times-is-partnering-with-jigsaw-to-expand-comment-capabilities/>.
* Obermeyer et al. (2019)

  Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan.
  Dissecting racial bias in an algorithm used to manage the health of
  populations.
  *Science*, 366(6464):447–453, 2019.
* Oren et al. (2019)

  Y. Oren, S. Sagawa, T. Hashimoto, and P. Liang.
  Distributionally robust language modeling.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  2019.
* Osgood-Zimmerman et al. (2018)

  A. Osgood-Zimmerman, A. I. Millear, R. W. Stubbs, C. Shields, B. V. Pickering,
  L. Earl, N. Graetz, D. K. Kinyoki, S. E. Ray, S. Bhatt, A. J. Browne,
  R. Burstein, E. Cameron, D. C. Casey, A. Deshpande, N. Fullman, P. W.
  Gething, H. S. Gibson, N. J. Henry, M. Herrero, L. K. Krause, I. D.
  Letourneau, A. J. Levine, P. Y. Liu, J. Longbottom, B. K. Mayala, J. F.
  Mosser, A. M. Noor, D. M. Pigott, E. G. Piwoz, P. Rao, R. Rawat, R. C.
  Reiner, D. L. Smith, D. J. Weiss, K. E. Wiens, A. H. Mokdad, S. S. Lim,
  C. J. L. Murray, N. J. Kassebaum, and S. I. Hay.
  Mapping child growth failure in Africa between 2000 and 2015.
  *Nature*, 555, 2018.
* Ovadia et al. (2019)

  Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. V. Dillon,
  B. Lakshminarayanan, and J. Snoek.
  Can you trust your model’s uncertainty? evaluating predictive
  uncertainty under dataset shift.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019.
* Panayotov et al. (2015)

  V. Panayotov, G. Chen, D. Povey, and S. Khudanpur.
  Librispeech: an ASR corpus based on public domain audio books.
  In *International Conference on Acoustics, Speech, and Signal
  Processing (ICASSP)*, pages 5206–5210, 2015.
* Parham et al. (2017)

  Jason Parham, Jonathan Crall, Charles Stewart, Tanya Berger-Wolf, and Daniel I
  Rubenstein.
  Animal population censusing at scale with citizen science and
  photographic identification.
  In *AAAI Spring Symposium-Technical Report*, 2017.
* Park et al. (2018)

  J. H. Park, J. Shin, and P. Fung.
  Reducing gender bias in abusive language detection.
  In *Empirical Methods in Natural Language Processing (EMNLP)*,
  pages 2799–2804, 2018.
* Parker and Leek (2012)

  Hilary S Parker and Jeffrey T Leek.
  The practical effect of batch on genomic prediction.
  *Statistical applications in genetics and molecular biology*,
  11(3), 2012.
* Patro et al. (2020)

  G. K. Patro, A. Biswas, N. Ganguly, K. P. Gummadi, and A. Chakraborty.
  Fairrec: Two-sided fairness for personalized recommendations in
  two-sided platforms.
  In *Proceedings of The Web Conference 2020*, pages 1194–1204,
  2020.
* Peng et al. (2018)

  X. Peng, B. Usman, N. Kaushik, D. Wang, J. Hoffman, and K. Saenko.
  VisDA: A synthetic-to-real benchmark for visual domain adaptation.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  2021–2026, 2018.
* Peng et al. (2019)

  X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang.
  Moment matching for multi-source domain adaptation.
  In *International Conference on Computer Vision (ICCV)*, 2019.
* Peng et al. (2020)

  X. Peng, E. Coumans, T. Zhang, T. Lee, J. Tan, and S. Levine.
  Learning agile robotic locomotion skills by imitating animals.
  In *Robotics: Science and Systems (RSS)*, 2020.
* Perelman (2014)

  L. Perelman.
  When “the state of the art” is counting words.
  *Assessing Writing*, 21:104–111, 2014.
* Peters et al. (2016)

  Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen.
  Causal inference by using invariant prediction: identification and
  confidence intervals.
  *Journal of the Royal Statistical Society. Series B
  (Methodological)*, 78, 2016.
* Phillips et al. (2020)

  N. A. Phillips, P. Rajpurkar, M. Sabini, R. Krishnan, S. Zhou, A. Pareek, N. M.
  Phu, C. Wang, A. Y. Ng, and M. P. Lungren.
  Chexphoto: 10,000+ smartphone photos and synthetic photographic
  transformations of chest x-rays for benchmarking deep learning robustness.
  *arXiv preprint arXiv:2007.06199*, 2020.
* Piech et al. (2013)

  C. Piech, J. Huang, Z. Chen, C. Do, A. Ng, and D. Koller.
  Tuned models of peer assessment in moocs.
  *Educational Data Mining*, 2013.
* Pierson et al. (2018)

  Emma Pierson, Sam Corbett-Davies, and Sharad Goel.
  Fast Threshold Tests for Detecting Discrimination.
  *arXiv:1702.08536 [cs, stat]*, March 2018.
  URL <http://arxiv.org/abs/1702.08536>.
  arXiv: 1702.08536.
* Pimentel et al. (2014)

  M. A. Pimentel, D. A. Clifton, L. Clifton, and L. Tarassenko.
  A review of novelty detection.
  *Signal Processing*, 99:215–249, 2014.
* Pipal et al. (2012)

  Kerrie A Pipal, Jeremy J Notch, Sean A Hayes, and Peter B Adams.
  Estimating escapement for a low-abundance steelhead population using
  dual-frequency identification sonar (didson).
  *North American Journal of Fisheries Management*, 32(5):880–893, 2012.
* Price and Cohen (2019)

  W Nicholson Price and I Glenn Cohen.
  Privacy in the age of medical big data.
  *Nature Medicine*, 25(1):37–43, 2019.
* Proksch et al. (2015)

  Sebastian Proksch, Johannes Lerch, and Mira Mezini.
  Intelligent code completion with bayesian networks.
  *ACM Transactions on Software Engineering and Methodology
  (TOSEM)*, 2015.
* Proksch et al. (2016)

  Sebastian Proksch, Sven Amann, Sarah Nadi, and Mira Mezini.
  Evaluating the evaluations of code recommender systems: A reality
  check.
  In *2016 31st IEEE/ACM International Conference on Automated
  Software Engineering (ASE)*, 2016.
* Ptashne and Gann (1997)

  Mark Ptashne and Alexander Gann.
  Transcriptional activation by recruitment.
  *Nature*, 386(6625):569, 1997.
* Quang and Xie (2019)

  D. Quang and X. Xie.
  Factornet: a deep learning framework for predicting cell type
  specific transcription factor binding from nucleotide-resolution sequential
  data.
  *Methods*, 166:40–47, 2019.
* Quiñonero-Candela et al. (2009)

  J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence.
  *Dataset shift in machine learning*.
  The MIT Press, 2009.
* Raychev et al. (2014)

  Veselin Raychev, Martin Vechev, and Eran Yahav.
  Code completion with statistical language models.
  In *Proceedings of the 35th ACM SIGPLAN Conference on
  Programming Language Design and Implementation*, pages 419–428, 2014.
* Raychev et al. (2016)

  Veselin Raychev, Pavol Bielik, and Martin Vechev.
  Probabilistic model for code with decision trees.
  *ACM SIGPLAN Notices*, 2016.
* Ré et al. (2019)

  C. Ré, F. Niu, P. Gudipati, and C. Srisuwananukorn.
  Overton: A data system for monitoring and improving machine-learned
  products.
  *arXiv preprint arXiv:1909.05372*, 2019.
* Recht et al. (2019)

  B. Recht, R. Roelofs, L. Schmidt, and V. Shankar.
  Do ImageNet classifiers generalize to ImageNet?
  In *International Conference on Machine Learning (ICML)*, 2019.
* Regev et al. (2017)

  Aviv Regev, Sarah A Teichmann, Eric S Lander, Ido Amit, Christophe Benoist,
  Ewan Birney, Bernd Bodenmiller, Peter Campbell, Piero Carninci, Menna
  Clatworthy, et al.
  The human cell atlas.
  *Elife*, 6:e27041, 2017.
* Reiner et al. (2018)

  R. C. Reiner, N. Graetz, D. C. Casey, C. Troeger, G. M. Garcia, J. F. Mosser,
  A. Deshpande, S. J. Swartz, S. E. Ray, B. F. Blacker, P. C. Rao,
  A. Osgood-Zimmerman, R. Burstein, D. M. Pigott, I. M. Davis, I. D.
  Letourneau, L. Earl, J. M. Ross, I. A. Khalil, T. H. Farag, O. J. Brady,
  M. U. Kraemer, D. L. Smith, S. Bhatt, D. J. Weiss, P. W. Gething, N. J.
  Kassebaum, A. H. Mokdad, C. J. Murray, and S. I. Hay.
  Variation in childhood diarrheal morbidity and mortality in Africa,
  2000–2015.
  *New England Journal of Medicine*, 379, 2018.
* Reker (2020)

  D. Reker.
  Practical considerations for active machine learning in drug
  discovery.
  *Drug Discovery Today: Technologies*, 2020.
* Ren et al. (2015)

  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Faster r-cnn: Towards real-time object detection with region proposal
  networks.
  *arXiv preprint arXiv:1506.01497*, 2015.
* Reynolds et al. (2020)

  Matthew Reynolds, Scott Chapman, Leonardo Crespo-Herrera, Gemma Molero,
  Suchismita Mondal, Diego NL Pequeno, Francisco Pinto, Francisco J
  Pinera-Chavez, Jesse Poland, Carolina Rivera-Amado, et al.
  Breeder friendly phenotyping.
  *Plant Science*, page 110396, 2020.
* Ribeiro et al. (2020)

  M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh.
  Beyond accuracy: Behavioral testing of NLP models with
  CheckList.
  In *Association for Computational Linguistics (ACL)*, pages
  4902–4912, 2020.
* Richter et al. (2016)

  S. R. Richter, V. Vineet, S. Roth, and V. Koltun.
  Playing for data: Ground truth from computer games.
  In *European Conference on Computer Vision*, pages 102–118,
  2016.
* Rigaki and Garcia (2018)

  M. Rigaki and S. Garcia.
  Bringing a GAN to a knife-fight: Adapting malware communication to
  avoid detection.
  In *2018 IEEE Security and Privacy Workshops (SPW)*, pages
  70–75, 2018.
* Robbes and Lanza (2008)

  Romain Robbes and Michele Lanza.
  How program history can improve code completion.
  In *International Conference on Automated Software Engineering*,
  2008.
* Rohs et al. (2009)

  Remo Rohs, Sean M West, Alona Sosinsky, Peng Liu, Richard S Mann, and Barry
  Honig.
  The role of dna shape in protein-dna recognition.
  *Nature*, 461(7268):1248, 2009.
* Rolf et al. (2020)

  E. Rolf, M. I. Jordan, and B. Recht.
  Post-estimation smoothing: A simple baseline for learning with side
  information.
  In *Artificial Intelligence and Statistics (AISTATS)*, 2020.
* Ronneberger et al. (2015)

  Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
  U-net: Convolutional networks for biomedical image segmentation.
  In *International Conference on Medical image computing and
  computer-assisted intervention*, pages 234–241. Springer, 2015.
* Ros et al. (2016)

  G. Ros, L. Sellart, J. Materzynska, D. Vazquez, and A. M. Lopez.
  The SYNTHIA dataset: A large collection of synthetic images for
  semantic segmentation of urban scenes.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pages 3234–3243, 2016.
* Rosenfeld et al. (2018)

  Amir Rosenfeld, Richard Zemel, and John K Tsotsos.
  The elephant in the room.
  *arXiv preprint arXiv:1808.03305*, 2018.
* Sadeghi and Levine (2017)

  F. Sadeghi and S. Levine.
  CAD2RL: Real single-image flight without a single real image.
  In *Robotics: Science and Systems (RSS)*, 2017.
* Sadeghi-Tehran et al. (2017)

  Pouria Sadeghi-Tehran, Nicolas Virlet, Kasra Sabermanesh, and Malcolm J
  Hawkesford.
  Multi-feature machine learning model for automatic segmentation of
  green fractional vegetation cover for high-throughput field phenotyping.
  *Plant methods*, 13(1):1–16, 2017.
* Saenko et al. (2010)

  K. Saenko, B. Kulis, M. Fritz, and T. Darrell.
  Adapting visual category models to new domains.
  In *European Conference on Computer Vision*, pages 213–226,
  2010.
* Saerens et al. (2002)

  M. Saerens, P. Latinne, and C. Decaestecker.
  Adjusting the outputs of a classifier to new a priori probabilities:
  a simple procedure.
  *Neural Computation*, 14(1):21–41, 2002.
* Sagawa et al. (2020a)

  S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang.
  Distributionally robust neural networks for group shifts: On the
  importance of regularization for worst-case generalization.
  In *International Conference on Learning Representations
  (ICLR)*, 2020a.
* Sagawa et al. (2020b)

  S. Sagawa, A. Raghunathan, P. W. Koh, and P. Liang.
  An investigation of why overparameterization exacerbates spurious
  correlations.
  In *International Conference on Machine Learning (ICML)*,
  2020b.
* Sahn and Stifel (2003)

  D. E. Sahn and D. Stifel.
  Exploring alternative measures of welfare in the absence of
  expenditure data.
  *The Review of Income and Wealth*, 49, 2003.
* Sanh et al. (2019)

  Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.
  Distilbert, a distilled version of bert: smaller, faster, cheaper and
  lighter.
  *arXiv preprint arXiv:1910.01108*, 2019.
* Santurkar et al. (2020)

  S. Santurkar, D. Tsipras, and A. Madry.
  Breeds: Benchmarks for subpopulation shift.
  *arXiv*, 2020.
* Sap et al. (2019)

  M. Sap, D. Card, S. Gabriel, Y. Choi, and N. A. Smith.
  The risk of racial bias in hate speech detection.
  In *Association for Computational Linguistics (ACL)*, 2019.
* Schneider and Zhuang (2020)

  Stefan Schneider and Alex Zhuang.
  Counting fish and dolphins in sonar images using deep learning.
  *arXiv preprint arXiv:2007.12808*, 2020.
* Seyyed-Kalantari et al. (2020)

  L. Seyyed-Kalantari, G. Liu, M. McDermott, and M. Ghassemi.
  Chexclusion: Fairness gaps in deep chest X-ray classifiers.
  *arXiv preprint arXiv:2003.00827*, 2020.
* Shakoor et al. (2017)

  Nadia Shakoor, Scott Lee, and Todd C Mockler.
  High throughput phenotyping to accelerate crop breeding and
  monitoring of diseases in the field.
  *Current opinion in plant biology*, 38:184–192, 2017.
* Shankar et al. (2017)

  Shreya Shankar, Yoni Halpern, Eric Breck, James Atwood, Jimbo Wilson, and
  D Sculley.
  No classification without representation: Assessing geodiversity
  issues in open data sets for the developing world.
  *Advances in Neural Information Processing Systems (NeurIPS)
  Workshop on Machine Learning for the Developing World*, 2017.
* Shankar et al. (2019)

  V. Shankar, A. Dave, R. Roelofs, D. Ramanan, B. Recht, and L. Schmidt.
  Do image classifiers generalize across time?
  *arXiv preprint arXiv:1906.02168*, 2019.
* Shapiro et al. (2014)

  Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczyński.
  *Lectures on stochastic programming: modeling and theory*.
  SIAM, 2014.
* Shen et al. (2018)

  J. Shen, Y. Qu, W. Zhang, and Y. Yu.
  Wasserstein distance guided representation learning for domain
  adaptation.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, 2018.
* Shermis (2014)

  M. D. Shermis.
  State-of-the-art automated essay scoring: Competition, results, and
  future directions from a united states demonstration.
  *Assessing Writing*, 20:53–76, 2014.
* Shetty et al. (2019)

  Rakshith Shetty, Bernt Schiele, and Mario Fritz.
  Not using the car to see the sidewalk–quantifying and controlling
  the effects of context in classification and segmentation.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pages 8218–8226, 2019.
* Shi et al. (2016)

  Yeyin Shi, J. Alex Thomasson, Seth C. Murray, N. Ace Pugh, William L. Rooney,
  Sanaz Shafian, Nithya Rajan, Gregory Rouze, Cristine L. S. Morgan, Haly L.
  Neely, Aman Rana, Muthu V. Bagavathiannan, James Henrickson, Ezekiel Bowden,
  John Valasek, Jeff Olsenholler, Michael P. Bishop, Ryan Sheridan, Eric B.
  Putman, Sorin Popescu, Travis Burks, Dale Cope, Amir Ibrahim, Billy F.
  McCutchen, David D. Baltensperger, Robert V. Avant, Jr, Misty Vidrine, and
  Chenghai Yang.
  Unmanned aerial vehicles for high-throughput phenotyping and
  agronomic research.
  *PLOS ONE*, 11(7):1–26, 07 2016.
  doi: 10.1371/journal.pone.0159781.
  URL <https://doi.org/10.1371/journal.pone.0159781>.
* Shimodaira (2000)

  H. Shimodaira.
  Improving predictive inference under covariate shift by weighting the
  log-likelihood function.
  *Journal of Statistical Planning and Inference*, 90:227–244, 2000.
* Shin et al. (2019)

  Richard Shin, Neel Kant, Kavi Gupta, Christopher Bender, Brandon Trabucco,
  Rishabh Singh, and Dawn Song.
  Synthetic datasets for neural program synthesis.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Shiu et al. (2020)

  Yu Shiu, KJ Palmer, Marie A Roch, Erica Fleishman, Xiaobai Liu, Eva-Marie
  Nosal, Tyler Helble, Danielle Cholewiak, Douglas Gillespie, and Holger
  Klinck.
  Deep neural networks for automated detection of marine mammal
  species.
  *Scientific Reports*, 10(1):1–12, 2020.
* Shoichet (2004)

  Brian K Shoichet.
  Virtual screening of chemical libraries.
  *Nature*, 432(7019):862–865, 2004.
* Slack et al. (2019)

  Dylan Slack, Sorelle Friedler, and Emile Givental.
  Fairness Warnings and Fair-MAML: Learning Fairly with
  Minimal Data.
  *arXiv:1908.09092 [cs, stat]*, December 2019.
  URL <http://arxiv.org/abs/1908.09092>.
  arXiv: 1908.09092.
* Sohoni et al. (2020)

  N. Sohoni, J. Dunnmon, G. Angus, A. Gu, and C. Ré.
  No subclass left behind: Fine-grained robustness in coarse-grained
  classification problems.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2020.
* Soneson et al. (2014)

  Charlotte Soneson, Sarah Gerster, and Mauro Delorenzi.
  Batch effect confounding leads to strong bias in performance
  estimates obtained by cross-validation.
  *PloS one*, 9(6):e100335, 2014.
* Srivastava and Mahony (2020)

  D. Srivastava and S. Mahony.
  Sequence and chromatin determinants of transcription factor binding
  and the establishment of cell type-specific binding patterns.
  *Biochimica et Biophysica Acta (BBA)-Gene Regulatory
  Mechanisms*, 1863(6), 2020.
* Srivastava et al. (2020)

  Megha Srivastava, Tatsunori Hashimoto, and Percy Liang.
  Robustness to Spurious Correlations via Human Annotations.
  In *International Conference on Machine Learning*, pages
  9109–9119. PMLR, November 2020.
  URL <http://proceedings.mlr.press/v119/srivastava20a.html>.
  ISSN: 2640-3498.
* Sterling and Irwin (2015)

  Teague Sterling and John J. Irwin.
  Zinc 15 – ligand discovery for everyone.
  *Journal of Chemical Information and Modeling*, 55(11):2324–2337, 2015.
  doi: 10.1021/acs.jcim.5b00559.
  PMID: 26479676.
* Stormo and Zhao (2010)

  Gary D Stormo and Yue Zhao.
  Determining the specificity of protein-dna interactions.
  *Nature Reviews Genetics*, 11(11):751, 2010.
* Stowell et al. (2019)

  Dan Stowell, Michael D Wood, Hanna Pamuła, Yannis Stylianou, and Hervé
  Glotin.
  Automatic acoustic detection of birds through deep learning: the
  first bird audio detection challenge.
  *Methods in Ecology and Evolution*, 10(3):368–380, 2019.
* Subbaswamy et al. (2020)

  A. Subbaswamy, R. Adams, and S. Saria.
  Evaluating model robustness to dataset shift.
  *arXiv preprint arXiv:2010.15100*, 2020.
* Sun and Saenko (2016)

  B. Sun and K. Saenko.
  Deep CORAL: Correlation alignment for deep domain adaptation.
  In *European conference on computer vision*, pages 443–450,
  2016.
* Sun et al. (2016)

  B. Sun, J. Feng, and K. Saenko.
  Return of frustratingly easy domain adaptation.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, 2016.
* Sun et al. (2020a)

  P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo,
  Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao,
  A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, S. Zhao, S. Cheng,
  Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov.
  Scalability in perception for autonomous driving: Waymo open
  dataset.
  In *Conference on Computer Vision and Pattern Recognition
  (CVPR)*, 2020a.
* Sun et al. (2020b)

  Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt.
  Test-time training with self-supervision for generalization under
  distribution shifts.
  In *International Conference on Machine Learning (ICML)*,
  2020b.
* Svyatkovskiy et al. (2019)

  Alexey Svyatkovskiy, Ying Zhao, Shengyu Fu, and Neel Sundaresan.
  Pythia: ai-assisted code completion system.
  In *Proceedings of the 25th ACM SIGKDD International Conference
  on Knowledge Discovery & Data Mining*, pages 2727–2735, 2019.
* Swinney and Anthony (2011)

  David C Swinney and Jason Anthony.
  How were new medicines discovered?
  *Nature reviews Drug discovery*, 10(7):507,
  2011.
* Tabak et al. (2020)

  Gil Tabak, Minjie Fan, Samuel Yang, Stephan Hoyer, and Geoffrey Davis.
  Correcting nuisance variation using wasserstein distance.
  *PeerJ*, 8:e8594, 2020.
* Tabak et al. (2019)

  Michael A Tabak, Mohammad S Norouzzadeh, David W Wolfson, Steven J Sweeney,
  Kurt C VerCauteren, Nathan P Snow, Joseph M Halseth, Paul A Di Salvo, Jesse S
  Lewis, Michael D White, et al.
  Machine learning to classify animal species in camera trap images:
  Applications in ecology.
  *Methods in Ecology and Evolution*, 10(4):585–590, 2019.
* Taghipour and Ng (2016)

  K. Taghipour and H. T. Ng.
  A neural approach to automated essay scoring.
  In *Proceedings of the 2016 conference on empirical methods in
  natural language processing*, pages 1882–1891, 2016.
* Taori et al. (2020)

  R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, and L. Schmidt.
  Measuring robustness to natural distribution shifts in image
  classification.
  *arXiv preprint arXiv:2007.00644*, 2020.
* Tatman (2017)

  R. Tatman.
  Gender and dialect bias in YouTube’s automatic captions.
  In *Workshop on Ethics in Natural Langauge Processing*,
  volume 1, pages 53–59, 2017.
* Taylor et al. (2019)

  J. Taylor, B. Earnshaw, B. Mabey, M. Victors, and J. Yosinski.
  Rxrx1: An image set for cellular morphological variation across many
  experimental batches.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Taylor et al. (2021)

  Michael J Taylor, Jessica K Lukowski, and Christopher R Anderton.
  Spatially resolved mass spectrometry at the single cell: Recent
  innovations in proteomics and metabolomics.
  *Journal of the American Society for Mass Spectrometry*,
  32(4):872–894, 2021.
* Tellez et al. (2018)

  D. Tellez, M. Balkenhol, I. Otte-Höller, R. van de Loo, R. Vogels, P. Bult,
  C. Wauters, W. Vreuls, S. Mol, N. Karssemeijer, et al.
  Whole-slide mitosis detection in h&e breast histology using phh3 as
  a reference to train distilled stain-invariant convolutional networks.
  *IEEE Transactions on Medical Imaging*, 37(9):2126–2136, 2018.
* Tellez et al. (2019)

  D. Tellez, G. Litjens, P. Bándi, W. Bulten, J. Bokhorst, F. Ciompi, and
  J. van der Laak.
  Quantifying the effects of data augmentation and stain color
  normalization in convolutional neural networks for computational pathology.
  *Medical Image Analysis*, 58, 2019.
* Temel et al. (2018)

  Dogancan Temel, Jinsol Lee, and Ghassan AlRegib.
  Cure-or: Challenging unreal and real environments for object
  recognition.
  In *2018 17th IEEE International Conference on Machine Learning
  and Applications (ICMLA)*, pages 137–144. IEEE, 2018.
* Thorp et al. (2018)

  Kelly R Thorp, Alison L Thompson, Sara J Harders, Andrew N French, and
  Richard W Ward.
  High-throughput phenotyping of crop water use efficiency via
  multispectral drone imagery and a daily soil water balance model.
  *Remote Sensing*, 10(11):1682, 2018.
* Thurman et al. (2012)

  Robert E Thurman, Eric Rynes, Richard Humbert, Jeff Vierstra, Matthew T
  Maurano, Eric Haugen, Nathan C Sheffield, Andrew B Stergachis, Hao Wang,
  Benjamin Vernot, et al.
  The accessible chromatin landscape of the human genome.
  *Nature*, 489(7414):75, 2012.
* Tiecke et al. (2017)

  T. G. Tiecke, X. Liu, A. Zhang, A. Gros, N. Li, G. Yetman, T. Kilic, S. Murray,
  B. Blankespoor, E. B. Prydz, and H. H. Dang.
  Mapping the world population one building at a time.
  *arXiv*, 2017.
* Tobin et al. (2017)

  J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel.
  Domain randomization for transferring deep neural networks from
  simulation to the real world.
  In *International Conference on Intelligent Robots and Systems
  (IROS)*, 2017.
* Toda and Okura (2019)

  Yosuke Toda and Fumio Okura.
  How convolutional neural networks diagnose plant disease.
  *Plant Phenomics*, 2019, 2019.
* Torralba and Efros (2011)

  A. Torralba and A. A. Efros.
  Unbiased look at dataset bias.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  1521–1528, 2011.
* Tuschl (2001)

  Thomas Tuschl.
  Rna interference and small interfering rnas.
  *Chembiochem*, 2(4):239–245, 2001.
* Tzeng et al. (2017)

  E. Tzeng, J. Hoffman, K. Saenko, and T. Darrell.
  Adversarial discriminative domain adaptation.
  In *Computer Vision and Pattern Recognition (CVPR)*, 2017.
* Tzeng et al. (2014)

  Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, and Trevor Darrell.
  Deep domain confusion: Maximizing for domain invariance.
  *arXiv preprint arXiv:1412.3474*, 2014.
* Ubbens et al. (2020)

  Jordan R Ubbens, Tewodros W Ayalew, Steve Shirtliffe, Anique Josuttes, Curtis
  Pozniak, and Ian Stavness.
  Autocount: Unsupervised segmentation and counting of organs in field
  images.
  In *European Conference on Computer Vision*, pages 391–399.
  Springer, 2020.
* Uzkent and Ermon (2020)

  B. Uzkent and S. Ermon.
  Learning when and where to zoom with deep reinforcement learning.
  In *Computer Vision and Pattern Recognition (CVPR)*, 2020.
* Vasic et al. (2019)

  Marko Vasic, Aditya Kanade, Petros Maniatis, David Bieber, and Rishabh Singh.
  Neural program repair by jointly learning to localize and repair.
  In *International Conference on Learning Representations
  (ICLR)*, 2019.
* Vatnehol et al. (2018)

  Sindre Vatnehol, Hector Peña, and Nils Olav Handegard.
  A method to automatically detect fish aggregations using horizontally
  scanning sonar.
  *ICES Journal of Marine Science*, 75(5):1803–1812, 2018.
* Veeling et al. (2018)

  B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, and M. Welling.
  Rotation equivariant cnns for digital pathology.
  In *International Conference on Medical Image Computing and
  Computer-assisted Intervention*, pages 210–218, 2018.
* Venkateswara et al. (2017)

  H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan.
  Deep hashing network for unsupervised domain adaptation.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  5018–5027, 2017.
* Veta et al. (2016)

  M. Veta, P. J. V. Diest, M. Jiwa, S. Al-Janabi, and J. P. Pluim.
  Mitosis counting in breast cancer: Object-level interobserver
  agreement and comparison to an automatic method.
  *PloS one*, 11(8), 2016.
* Veta et al. (2019)

  M. Veta, Y. J. Heng, N. Stathonikos, B. E. Bejnordi, F. Beca, T. Wollmann,
  K. Rohr, M. A. Shah, D. Wang, M. Rousson, et al.
  Predicting breast tumor proliferation from whole-slide images: the
  tupac16 challenge.
  *Medical image analysis*, 54:111–121, 2019.
* Volpi et al. (2018)

  R. Volpi, H. Namkoong, O. Sener, J. Duchi, V. Murino, and S. Savarese.
  Generalizing to unseen domains via adversarial data augmentation.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2018.
* Wang et al. (2019a)

  Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael,
  Felix Hill, Omer Levy, and Samuel R. Bowman.
  SuperGLUE: A stickier benchmark for general-purpose language
  understanding systems.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019a.
* Wang et al. (2019b)

  Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R
  Bowman.
  GLUE: A multi-task benchmark and analysis platform for natural
  language understanding.
  In *International Conference on Learning Representations
  (ICLR)*, 2019b.
* Wang et al. (2020a)

  D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell.
  Fully test-time adaptation by entropy minimization.
  *arXiv preprint arXiv:2006.10726*, 2020a.
* Wang et al. (2019c)

  H. Wang, S. Ge, Z. Lipton, and E. P. Xing.
  Learning robust global representations by penalizing local predictive
  power.
  In *Advances in Neural Information Processing Systems
  (NeurIPS)*, 2019c.
* Wang et al. (2017)

  S. Wang, M. Bai, G. Mattyus, H. Chu, W. Luo, B. Yang, J. Liang, J. Cheverie,
  S. Fidler, and R. Urtasun.
  Torontocity: Seeing the world with a million eyes.
  In *International Conference on Computer Vision (ICCV)*, 2017.
* Wang et al. (2020b)

  S. Wang, W. Chen, S. M. Xie, G. Azzari, and D. B. Lobell.
  Weakly supervised deep learning for segmentation of remote sensing
  imagery.
  *Remote Sensing*, 12, 2020b.
* Ward and Moghadam (2020)

  Daniel Ward and Peyman Moghadam.
  Scalable learning for bridging the species gap in image-based plant
  phenotyping.
  *Computer Vision and Image Understanding*, 197:103009,
  2020.
* Wearn and Glover-Kapfer (2017)

  OR Wearn and P Glover-Kapfer.
  Camera-trapping for conservation: a guide to best-practices.
  *WWF conservation technology series*, 1(1):2019–04, 2017.
* Weinberger (2015)

  S. Weinberger.
  Speech accent archive.
  *George Mason University*, 2015.
* Weinstein (2018)

  Ben G Weinstein.
  A computer vision for animal ecology.
  *Journal of Animal Ecology*, 87(3):533–545,
  2018.
* Weinstein et al. (2013)

  J. N. Weinstein, E. A. Collisson, G. B. Mills, K. R. M. Shaw, B. A. Ozenberger,
  K. Ellrott, I. Shmulevich, C. Sander, J. M. Stuart, C. G. A. R. Network,
  et al.
  The cancer genome atlas pan-cancer analysis project.
  *Nature genetics*, 45(10), 2013.
* West et al. (2014)

  R. West, H. S. Paskov, J. Leskovec, and C. Potts.
  Exploiting social network structure for person-to-person sentiment
  analysis.
  *Transactions of the Association for Computational Linguistics
  (TACL)*, 2:297–310, 2014.
* Widmer and Kubat (1996)

  G. Widmer and M. Kubat.
  Learning in the presence of concept drift and hidden contexts.
  *Machine learning*, 23(1):69–101, 1996.
* Williams et al. (2016)

  J. J. Williams, J. Kim, A. Rafferty, S. Maldonado, K. Z. Gajos, W. S. Lasecki,
  and N. Heffernan.
  Axis: Generating explanations at scale with learnersourcing and
  machine learning.
  In *Proceedings of the Third (2016) ACM Conference on
  Learning@Scale*, pages 379–388, 2016.
* Wilson et al. (2019)

  Benjamin Wilson, Judy Hoffman, and Jamie Morgenstern.
  Predictive inequity in object detection.
  *arXiv preprint arXiv:1902.11097*, 2019.
* Wolf et al. (2019)

  T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac,
  T. Rault, R. Louf, M. Funtowicz, and J. Brew.
  HuggingFace’s transformers: State-of-the-art natural language
  processing.
  *arXiv preprint arXiv:1910.03771*, 2019.
* Won et al. (2010)

  Kyoung-Jae Won, Bing Ren, and Wei Wang.
  Genome-wide prediction of transcription factor binding sites using an
  integrated model.
  *Genome biology*, 11(1):1–17, 2010.
* Wong et al. (2020)

  Ho Yuen Frank Wong, Hiu Yin Sonia Lam, Ambrose Ho-Tung Fong, Siu Ting Leung,
  Thomas Wing-Yan Chin, Christine Shing Yen Lo, Macy Mei-Sze Lui, Jonan
  Chun Yin Lee, Keith Wan-Hang Chiu, Tom Chung, et al.
  Frequency and distribution of chest radiographic findings in covid-19
  positive patients.
  *Radiology*, page 201160, 2020.
* Worrall et al. (2017)

  D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow.
  Harmonic networks: Deep translation and rotation equivariance.
  In *Computer Vision and Pattern Recognition (CVPR)*, pages
  5028–5037, 2017.
* Wu et al. (2019a)

  M. Wu, M. Mosse, N. Goodman, and C. Piech.
  Zero shot learning for code education: Rubric sampling with deep
  learning inference.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, volume 33, pages 782–790, 2019a.
* Wu et al. (2020)

  M. Wu, R. L. Davis, B. W. Domingue, C. Piech, and N. Goodman.
  Variational item response theory: Fast, accurate, and expressive.
  *International Conference on Educational Data Mining*, 2020.
* Wu et al. (2019b)

  Y. Wu, E. Winston, D. Kaushik, and Z. Lipton.
  Domain adaptation with asymmetrically-relaxed distribution alignment.
  In *International Conference on Machine Learning (ICML)*, pages
  6872–6881, 2019b.
* Wu et al. (2018)

  Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse,
  Aneesh S Pappu, Karl Leswing, and Vijay Pande.
  Moleculenet: a benchmark for molecular machine learning.
  *Chemical Science*, 9(2):513–530, 2018.
* Wulfmeier et al. (2018)

  M. Wulfmeier, A. Bewley, and I. Posner.
  Incremental adversarial domain adaptation for continually changing
  environments.
  In *International Conference on Robotics and Automation (ICRA)*,
  2018.
* Xiao et al. (2020)

  K. Xiao, L. Engstrom, A. Ilyas, and A. Madry.
  Noise or signal: The role of image backgrounds in object recognition.
  *arXiv preprint arXiv:2006.09994*, 2020.
* Xie et al. (2016)

  M. Xie, N. Jean, M. Burke, D. Lobell, and S. Ermon.
  Transfer learning from deep features for remote sensing and poverty
  mapping.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, 2016.
* Xie et al. (2020)

  S. M. Xie, A. Kumar, R. Jones, F. Khani, T. Ma, and P. Liang.
  In-N-Out: Pre-training and self-training using auxiliary
  information for out-of-distribution robustness.
  *arXiv*, 2020.
* Xiong et al. (2019)

  Haipeng Xiong, Zhiguo Cao, Hao Lu, Simon Madec, Liang Liu, and Chunhua Shen.
  Tasselnetv2: in-field counting of wheat spikes with context-augmented
  local regression networks.
  *Plant Methods*, 15(1):1–14, 2019.
* Xu et al. (2018)

  K. Xu, W. Hu, J. Leskovec, and S. Jegelka.
  How powerful are graph neural networks?
  In *International Conference on Learning Representations
  (ICLR)*, 2018.
* Yang and Newsam (2010)

  Y. Yang and S. Newsam.
  Bag-of-visual-words and spatial extensions for land-use
  classification.
  *Geographic Information Systems*, 2010.
* Yang et al. (2019)

  Y. Yang, K. Caluwaerts, A. Iscen, T. Zhang, J. Tan, and V. Sindhwani.
  Data efficient reinforcement learning for legged robots.
  In *Conference on Robot Learning (CoRL)*, 2019.
* Yasunaga and Liang (2020)

  Michihiro Yasunaga and Percy Liang.
  Graph-based, self-supervised program repair from diagnostic feedback.
  In *International Conference on Machine Learning (ICML)*, 2020.
* Yeh et al. (2020)

  C. Yeh, A. Perez, A. Driscoll, G. Azzari, Z. Tang, D. Lobell, S. Ermon, and
  M. Burke.
  Using publicly available satellite imagery and deep learning to
  understand economic well-being in Africa.
  *Nature Communications*, 11, 2020.
* You et al. (2017)

  J. You, X. Li, M. Low, D. Lobell, and S. Ermon.
  Deep gaussian process for crop yield prediction based on remote
  sensing data.
  In *Association for the Advancement of Artificial Intelligence
  (AAAI)*, 2017.
* Yu et al. (2020)

  F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell.
  BDD100K: A diverse driving dataset for heterogeneous multitask
  learning.
  In *Conference on Computer Vision and Pattern Recognition
  (CVPR)*, 2020.
* Yuval et al. (2011)

  N. Yuval, W. Tao, C. Adam, B. Alessandro, W. Bo, and N. A. Y.
  Reading digits in natural images with unsupervised feature learning.
  In *NIPS Workshop on Deep Learning and Unsupervised Feature
  Learning*, 2011.
* Zafar et al. (2017)

  Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P.
  Gummadi, and Adrian Weller.
  From Parity to Preference-based Notions of Fairness in
  Classification.
  *arXiv:1707.00010 [cs, stat]*, Nov 2017.
  URL <http://arxiv.org/abs/1707.00010>.
  arXiv: 1707.00010.
* Zech et al. (2018)

  J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K.
  Oermann.
  Variable generalization performance of a deep learning model to
  detect pneumonia in chest radiographs: A cross-sectional study.
  In *PLOS Medicine*, 2018.
* Zhang et al. (2013)

  K. Zhang, B. Schölkopf, K. Muandet, and Z. Wang.
  Domain adaptation under target and conditional shift.
  In *International Conference on Machine Learning (ICML)*, pages
  819–827, 2013.
* Zhang et al. (2020)

  M. Zhang, H. Marklund, N. Dhawan, A. Gupta, S. Levine, and C. Finn.
  Adaptive risk minimization: A meta-learning approach for tackling
  group shift.
  *arXiv preprint arXiv:2007.02931*, 2020.
* Zhang et al. (2019)

  Y. Zhang, J. Baldridge, and L. He.
  Paws: Paraphrase adversaries from word scrambling.
  In *North American Association for Computational Linguistics
  (NAACL)*, 2019.
* Zhao et al. (2018)

  J. Zhao, T. Wang, M. Yatskar, V. Ordoñez, and K. Chang.
  Gender bias in coreference resolution: Evaluation and debiasing
  methods.
  In *North American Association for Computational Linguistics
  (NAACL)*, 2018.
* Zhou and Troyanskaya (2015)

  J. Zhou and O. G. Troyanskaya.
  Predicting effects of noncoding variants with deep learning–based
  sequence model.
  *Nature Methods*, 12(10):931–934, 2015.
* Zhou et al. (2020)

  X. Zhou, Y. Nie, H. Tan, and M. Bansal.
  The curse of performance instability in analysis datasets:
  Consequences, source, and suggestions.
  *arXiv preprint arXiv:2004.13606*, 2020.
* Zhou et al. (2014)

  Yuexin Zhou, Shiyou Zhu, Changzu Cai, Pengfei Yuan, Chunmei Li, Yanyi Huang,
  and Wensheng Wei.
  High-throughput screening of a crispr/cas9 library for functional
  genomics in human cells.
  *Nature*, 509(7501):487, 2014.
* Zitnick et al. (2020)

  C. L. Zitnick, L. Chanussot, A. Das, S. Goyal, J. Heras-Domingo, C. Ho, W. Hu,
  T. Lavril, A. Palizhati, M. Riviere, M. Shuaibi, A. Sriram, K. Tran, B. Wood,
  J. Yoon, D. Parikh, and Z. Ulissi.
  An introduction to electrocatalyst design using machine learning for
  renewable energy storage.
  *arXiv preprint arXiv:2010.09435*, 2020.

## A Dataset realism

In this section, we discuss the framework we use to assess the realism of a benchmark dataset.
Realism is subtle to pin down and highly contextual, and assessing realism often requires consulting with domain experts and practitioners. As a general framework, we can view a benchmark dataset as comprising the data, a task and associated evaluation metric, and a train/test split that potentially reflects a distribution shift.
Each of these components can independently be more or less realistic:

1. 1.

   The data—which includes not just the inputs x𝑥x but also any associated metadata (e.g., the domain that each data point came from)— is realistic if it accurately reflects what would plausibly be collected and available for a model to use in a real application.
   The realism of data also depends on the application context; for example, using medical images captured with state-of-the-art equipment might be realistic for well-equipped hospitals, but not necessarily for clinics that use older generations of the technology, or vice versa.
   Extreme examples of unrealistic data include the Gaussian distributions that are often used to cleanly illustrate the theoretical properties of various algorithms.
2. 2.

   The task and evaluation metric is realistic if the task is relevant to a real application and if the metric measures how successful a model would be in that application.
   Here and with the other components, realism lies on a spectrum.
   For example, in a wildlife conservation application where the inputs are images from camera traps, the real task might be to estimate species populations (Parham et al., [2017](#bib.bib285)), i.e., the number of distinct individual animals of each species seen in the overall collection of images; a task that is less realistic but still relevant and useful for ecologists might be to classify what species of animal is seen in each image (Tabak et al., [2019](#bib.bib363)).
   The choice of evaluation metric is also important. In the wildlife example, conservationists might care more about rare species than common species, so measuring average classification accuracy would be less realistic than a metric that prioritizes classifying the rare species correctly.
3. 3.

   The distribution shift (train/test split) is realistic if it reflects training and test distributions that might arise in deployment for that dataset and task.
   For example, if a medical algorithm is trained on data from a few hospitals and then expected to be deployed more widely, then it would be realistic to test it on hospitals that are not in the training set.
   On the other hand, an example of a less realistic shift is to, for instance, train a pedestrian classifier entirely on daytime photos and then test it only on nighttime photos;
   in practice, any reasonable dataset for pedestrian detection that is used in a real application would include both daytime and nighttime photos.

Through the lens of this framework, existing ML benchmarks tend to focus on object recognition tasks with realistic data (e.g., photos) but not necessarily with realistic distribution shifts.
With Wilds, we seek to address this gap by selecting datasets that represent a wide variety of tasks (with realistic evaluation metrics and data) and that reflect realistic distribution shifts, i.e., train/test splits that are likely to arise in real-world deployments.

To elaborate on the realism of the distribution shift, we associate each dataset in Wilds with the distribution shift (i.e., problem setting) that we believe best reflects the real-world challenges in the corresponding application area.
For example, domain generalization is a realistic setting for the Camelyon17-wilds dataset as medical models are typically trained on data collected from a handful of hospitals, but with the goal of general deployment across different hospitals.
On the other hand, subpopulation shift is appropriate for the CivilComments-wilds dataset, as the real-world challenge is that some demographic subpopulations (domains) are underrepresented, rather than completely unseen, in the training data.
The appropriate problem setting depends on many dataset-specific factors, but some common considerations include:

* •

  Domain type. Certain types of domains are generally more appropriate for a particular setting. For example, if the domains represent time, as in FMoW-wilds, then domain generalization is suitable as a common challenge is to generalize from past data to future data.
  On the other hand, if the domains represent demographics and the goal is to improve performance on minority subpopulations, as in CivilComments-wilds, then subpopulation shift is typically more appropriate.
* •

  Data collection challenges. When collecting data from a new domain is expensive, domain generalization is often appropriate, as we might want to train on data from a limited number of domains but still generalize to unseen domains. For example, it is difficult to collect patient data from multiple hospitals, as in Camelyon17-wilds, or survey data from new countries, as in PovertyMap-wilds.
* •

  Continuous addition of new domains. A special case of the above is when new domains are continuously created. For example, in Amazon-wilds, where domains correspond to users, new users are constantly signing up for the platform; and in iWildCam2020-wilds, where domains correspond to camera traps, new cameras are constantly being deployed. These are natural domain generalization settings.

## B Prior work on ML benchmarks for distribution shifts

In this section, we discuss existing ML distribution shift benchmarks in more detail, categorizing them by how they induce their respective distribution shifts.
We focus here on work that has appeared in ML conferences and journals; we discuss related work from other research communities in Section [8](#S8 "8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and Appendix [E](#S5a "E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We also restrict our attention to publicly-available datasets.
While others have studied some proprietary datasets with realistic distribution shifts, such as the StreetView StoreFronts dataset (Hendrycks et al., [2020b](#bib.bib175)) or diabetic retinopathy datasets (D’Amour et al., [2020a](#bib.bib102)), these datasets are not publicly available due to privacy and other commercial reasons.

##### Distribution shifts from transformations.

Some of the most widely-adopted benchmarks induce distribution shifts by synthetically transforming the data.
Examples include rotated and translated versions of MNIST and CIFAR (Worrall et al., [2017](#bib.bib408); Gulrajani and Lopez-Paz, [2020](#bib.bib158));
surface variations such as texture, color, and corruptions like blur in Colored MNIST (Gulrajani and Lopez-Paz, [2020](#bib.bib158)), Stylized ImageNet (Geirhos et al., [2018a](#bib.bib142)), ImageNet-C (Hendrycks and Dietterich, [2019](#bib.bib172)),
and similar ImageNet variants (Geirhos et al., [2018b](#bib.bib143));
and datasets that crop out objects and replace their backgrounds, as in the Backgrounds Challenge (Xiao et al., [2020](#bib.bib414)) and other similar datasets (Sagawa et al., [2020a](#bib.bib327); Koh et al., [2020](#bib.bib209)).
Benchmarks for adversarial robustness also fall in this category of distribution shifts from transformations (Goodfellow et al., [2015](#bib.bib153); Croce et al., [2020](#bib.bib97)).
Though adversarial robustness is not a focus of this work,
we note that recent work on temporal perturbations with the ImageNet-Vid-Robust and YTBB-Robust datasets (Shankar et al., [2019](#bib.bib337))
represents a different form of distribution shift that also impacts real-world applications.
Outside of visual object recognition, other work has used synthetic datasets and transformations to explore compositional generalization, e.g., SCAN (Lake and Baroni, [2018](#bib.bib218)).
We discuss this more in Section [8](#S8 "8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

##### Synthetic-to-real transfers.

Fully synthetic datasets such as SYNTHIA (Ros et al., [2016](#bib.bib321)) and StreetHazards (Hendrycks et al., [2020a](#bib.bib174)) have been adopted for out-of-distribution detection as well as domain adaptation and generalization, e.g., by testing robustness to transformations in the seasons, weather, time, or architectural style (Hoffman et al., [2018](#bib.bib178); Volpi et al., [2018](#bib.bib389)).
While the data is synthetic, it can still look realistic if a high-fidelity simulator is used.
In particular, synthetic benchmarks that study transfers from synthetic to real data (Ganin and Lempitsky, [2015](#bib.bib136); Richter et al., [2016](#bib.bib315); Peng et al., [2018](#bib.bib289)) can be important tools for tackling real-world problems: even though the data is synthesized and by definition, not real, the synthetic-to-real distribution shift can still be realistic in contexts where real data is much harder to acquire than synthetic data (Bellemare et al., [2020](#bib.bib42)).
In this work, we do not study these types of synthetic distribution shifts; instead, we focus on distribution shifts that occur in the wild between real data distributions.

##### Distribution shifts from constrained splits.

Other benchmarks do not rely on transformations but instead split the data in a way that induces particular distribution shifts.
These benchmarks have realistic data, e.g., the data points are derived from real-world photos,
but they do not necessarily reflect distribution shifts that would arise in the wild.
For example, BREEDS (Santurkar et al., [2020](#bib.bib331)) and a related dataset (Hendrycks and Dietterich, [2019](#bib.bib172)) test generalization to unseen ImageNet subclasses by holding out subclasses specified by several controllable parameters;
similarly, NICO (He et al., [2020](#bib.bib168)) considers subclasses that are defined by their context, such as dogs at home versus dogs on the beach;
DeepFashion-Remixed (Hendrycks et al., [2020b](#bib.bib175)) constrains the training set to include only photos from a single camera viewpoint and tests generalization to unseen camera viewpoints;
BDD-Anomaly (Hendrycks et al., [2020a](#bib.bib174)) uses a driving dataset but with all motorcycles, trains, and bicycles removed from the training set only;
and ObjectNet (Barbu et al., [2019](#bib.bib29)) comprises images taken from a few pre-specified viewpoints, allowing for systematic evaluation for robustness to camera angle changes but deviating from natural camera angles.

##### Distribution shifts across datasets.

A well-studied special case of the above category is the class of distribution shifts obtained by combining several disparate datasets (Torralba and Efros, [2011](#bib.bib377)), training on one or more of them and then testing on the remaining datasets.
A recent influential example is the ImageNetV2 dataset (Recht et al., [2019](#bib.bib308)), which was constructed to be similar to the original ImageNet dataset.
Unlike ImageNetV2, however, many of these distribution shifts were constructed to be more drastic than might arise in the wild.
For example, standard domain adaptation benchmarks include training on MNIST but testing on SVHN street signs (LeCun et al., [1998](#bib.bib226); Yuval et al., [2011](#bib.bib425); Tzeng et al., [2017](#bib.bib379); Hoffman et al., [2018](#bib.bib178)),
as well as transfers across datasets containing different renditions (e.g., photos, clipart, sketches) in DomainNet (Peng et al., [2019](#bib.bib290)) and the Office-Home dataset (Venkateswara et al., [2017](#bib.bib386)).

The main difference between domain adaptation and domain generalization is that in the latter, we do not assume access to unlabeled data from the test distribution.
This makes it straightforward to use domain adaptation benchmarks for domain generalization, e.g., in DomainBed (Gulrajani and Lopez-Paz, [2020](#bib.bib158)); we focus on domain generalization in this work,
but further discuss unsupervised domain adaptation in Section [C](#S3a "C Potential extensions to other problem settings ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Other similar benchmarks that have been proposed for domain generalization include VLCS (Fang et al., [2013](#bib.bib128)), which tests generalization across similar visual object recognition datasets; PACS (Li et al., [2017a](#bib.bib229)), which (like DomainNet) tests generalization across datasets with different renditions; and ImageNet-R (Hendrycks et al., [2020b](#bib.bib175)) and ImageNet-Sketch (Wang et al., [2019c](#bib.bib393)), which also test generalization across different renditions by collecting separate datasets from Flickr and Google Image queries.

## C Potential extensions to other problem settings

In this paper, we have focused on two problem settings involving domain shifts: domain generalization and subpopulation shifts.
Here, we discuss other problem settings within the framework of domain shifts that could also apply to Wilds datasets.
Using Wilds to benchmark and develop algorithms for these settings is an important avenue for future work, and we welcome community contributions towards this effort.

### C.1 Problem settings in domain shifts

Within the general framework of domain shifts, specific problem settings can differ along the following axes of variation:

1. 1.

   Seen versus unseen test domains.
   Test domains may be seen during training time (𝒟𝗍𝖾𝗌𝗍⊆𝒟𝗍𝗋𝖺𝗂𝗇superscript𝒟𝗍𝖾𝗌𝗍superscript𝒟𝗍𝗋𝖺𝗂𝗇\mathcal{D}^{\mathsf{test}}\subseteq\mathcal{D}^{\mathsf{train}}), as in subpopulation shift, or unseen
   (𝒟𝗍𝗋𝖺𝗂𝗇∩𝒟𝗍𝖾𝗌𝗍=∅superscript𝒟𝗍𝗋𝖺𝗂𝗇superscript𝒟𝗍𝖾𝗌𝗍\mathcal{D}^{\mathsf{train}}\cap\mathcal{D}^{\mathsf{test}}=\emptyset), as in domain generalization.
   The domain generalization and subpopulation shift settings mainly differ on this factor.
2. 2.

   Train-time domain annotations.
   The domain identity d𝑑d may be observed for none, some, or all of the training examples.
   Train-time domain annotations are straightforward to obtain in some settings, e.g., we should know which patients in the training sets came from which hospitals, but can be harder to obtain in some settings, e.g., we might only have demographic information on a subset of training users. In our domain generalization and subpopulation shift settings, d𝑑d is always observed at training time.
3. 3.

   Test-time domain annotations.
   The domain identity d𝑑d may be observed for none, some, or all of the test examples.
   Test-time domain annotations allow models to be domain-specific, e.g., by treating domain identity as a feature if the train and test domains overlap.
   For example, if the domains correspond to continents and the data to satellite images from a continent, we would presumably know what continent each image was taken from.
   On the other hand, if the domains correspond to demographic information, this might be hard to obtain at test time (as well as training time, as mentioned above).
   In domain generalization, d𝑑d may be observed at test time, but it is not helpful by itself as all of the test domains are unseen at training time. However, when combined with test-time unlabeled data, observing the domain d𝑑d at test time could help with adaptation.
   In subpopulation shift, we typically assume that d𝑑d is unobserved at test time, though this need not always be true.
4. 4.

   Test-time unlabeled data.
   Varying amounts of unlabeled test data—samples of x𝑥x drawn from the test distribution P𝗍𝖾𝗌𝗍superscript𝑃𝗍𝖾𝗌𝗍P^{\mathsf{test}}—may be available, from none to a small batch to a large pool. This affects the degree to which models can adapt to test distributions.
   For example, if the domains correspond to locations and the data points to photos taken at those locations, we might assume access to some unlabeled photos taken at the test locations.

Each combination of the above four factors corresponds to a specific problem setting with a different set of applicable methods.
In the current version of the Wilds benchmark, we focus on domain generalization and subpopulation shifts, which represent specific configurations of these factors.
We briefly discuss a few other problem settings in the remainder of this section.

### C.2 Unsupervised domain adaptation

In the presence of distribution shift, a potential source of leverage is observing unlabeled test points from the test distribution.
In the unsupervised domain adaptation setting, we assume that at training time, we have access to a large amount of unlabeled data from each test distribution of interest, as well as the resources to train a separate model for each test distribution.
For example, in a satellite imagery setting like FMoW-wilds, it might be appropriate to assume that we have access to a large set of unlabeled recent satellite images from each continent and the wherewithal to train a separate model for each continent.

Many of the methods for domain generalization discussed in Section [6](#S6 "6 Baseline algorithms for distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") were originally methods for domain adaptation, since methods for both settings share the common goal of learning models that can transfer between domains. For example, methods that learn features that have similar distributions across domains are equally applicable to both settings
(Ben-David et al., [2006](#bib.bib43); Long et al., [2015](#bib.bib246); Sun et al., [2016](#bib.bib357); Ganin et al., [2016](#bib.bib137); Tzeng et al., [2017](#bib.bib379); Shen et al., [2018](#bib.bib339); Wu et al., [2019b](#bib.bib411)).
In fact, the CORAL algorithm that we use as a baseline in this work was originally developed for, and successfully applied in, unsupervised domain adaptation (Sun and Saenko, [2016](#bib.bib356)).
Other methods rely on knowing the test distribution and are thus specific to domain adaptation, e.g., learning to map data points from source to target domains (Hoffman et al., [2018](#bib.bib178)),
or estimating the test label distribution from unlabeled test data (Saerens et al., [2002](#bib.bib326); Zhang et al., [2013](#bib.bib428); Lipton et al., [2018](#bib.bib240); Azizzadenesheli et al., [2019](#bib.bib23); Alexandari et al., [2020](#bib.bib8); Garg et al., [2020](#bib.bib138)).

### C.3 Test-time adaptation

A closely related setting to unsupervised domain adaptation is test-time adaptation, which also assumes the availability of unlabeled test data.
For datasets where there are many potential test domains (e.g., in iWildCam2020-wilds, we want a model that can ideally generalize to any camera trap), it might be infeasible to train a separate model for each test domain, as unsupervised domain adaptation would require.
In the test-time adaptation setting, we assume that a model is allowed to adapt to a small amount of unlabeled test data in a way that is computationally much less intensive than typical domain adaptation methods.
This is a difference of degree and not of kind, but it can have significant practical implications. For example, domain adaptation approaches typically require access to the training set and a large unlabeled test set at the same time, whereas test-time adaptation methods typically only require the learned model (which can be much smaller than the original training set) as well as a smaller amount of unlabeled test data.

A number of test-time adaptation methods have been recently proposed
(Li et al., [2017c](#bib.bib236); Sun et al., [2020b](#bib.bib359); Wang et al., [2020a](#bib.bib392)).
For example, adaptive risk minimization (ARM) is a meta-learning approach that adapts models to each batch of test examples under the assumption that all data points in a batch come from the same domain (Zhang et al., [2020](#bib.bib429)).
Many datasets in Wilds are suitable for the test-time adaptation setting.
For example, in iWildCam2020-wilds, images from the same domain are highly similar, sharing the same location, background, and camera angle, and prior work has shown inferring these shared features can improve performance considerably (Beery et al., [2020b](#bib.bib39)).

### C.4 Selective prediction

A different problem setting that is orthogonal to the settings described above is selective prediction.
In the selective prediction setting, models are allowed to abstain on points where their confidence is below a certain threshold.
This is appropriate when, for example, abstentions can be handled by backing off to human experts, such as pathologists for Camelyon17-wilds, content moderators for CivilComments-wilds, wildlife experts for iWildCam2020-wilds, etc.
Many methods for selective prediction have been developed, from simply using softmax probabilities as a proxy for confidence (Cordella et al., [1995](#bib.bib95); Geifman and El-Yaniv, [2017](#bib.bib139)),
to methods involving ensembles of models (Gal and Ghahramani, [2016](#bib.bib135); Lakshminarayanan et al., [2017](#bib.bib219); Geifman et al., [2018](#bib.bib141))
or jointly learning to abstain and classify (Bartlett and Wegkamp, [2008](#bib.bib30); Geifman and El-Yaniv, [2019](#bib.bib140); Feng et al., [2019](#bib.bib129)).

Intuitively, even if a model is not robust to a distribution shift,
it might at least be able to maintain high accuracies on some subset of points that are close to the training distribution, while abstaining on the other points.
Indeed, prior work has shown that selective prediction can improve model accuracy under distribution shifts
(Pimentel et al., [2014](#bib.bib297); Hendrycks and Gimpel, [2017](#bib.bib173); Liang et al., [2018](#bib.bib238); Ovadia et al., [2019](#bib.bib283); Feng et al., [2019](#bib.bib129); Kamath et al., [2020](#bib.bib198)).
However, distribution shifts still pose a problem for selective prediction methods;
for instance, it is difficult to maintain desired abstention rates under distribution shifts (Kompa et al., [2020](#bib.bib210)), and confidence estimates have been found to drift over time (e.g., Davis et al. ([2017](#bib.bib106))).

## D Additional experimental details

### D.1 Model hyperparameters

For each hyperparameter setting, we used early stopping to pick the epoch with the best OOD validation performance (as measured by the specified metrics for each dataset described in Section [4](#S4 "4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), and then picked the model hyperparameters with the best early-stopped validation performance.
We found that this gave similar or slightly better OOD test performance than selecting hyperparameters using the ID validation set (Table [3](#S4.T3 "Table 3 ‣ D.1 Model hyperparameters ‣ D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

Using the OOD validation set for early stopping means that even if the training procedure does not explicitly use additional metadata, as in ERM, the metadata might still be implicitly (but mildly) used for model selection in one of two related ways. First, the metric might use the metadata directly (e.g., by computing the accuracy over different subpopulations defined in the metadata). Second, the OOD validation set is generally selected according to this metadata (e.g., comprising data from a disjoint set of domains as the training set).
We expect that implicitly using the metadata in these ways should increase the OOD performance of each model. Nevertheless, as Sections [5](#S5 "5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and [6](#S6 "6 Baseline algorithms for distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") show, there are still large gaps between OOD and ID performance.

In general, we selected model hyperparameters with ERM and used the same hyperparameters for the other algorithm baselines (e.g., CORAL, IRM, or Group DRO).
For CORAL and IRM, we did a subsequent grid search over the weight of the penalty term,
using the defaults from Gulrajani and Lopez-Paz ([2020](#bib.bib158)). Specifically, we tried penalty weights of {0.1,1,10}0.1110\{0.1,1,10\} for CORAL and penalty weights of {1,10,100,1000}1101001000\{1,10,100,1000\} for IRM.
We fixed the step size hyperparameter for Group DRO to its default value of 0.01 (Sagawa et al., [2020a](#bib.bib327)).

Table 3:  The performance of models trained with empirical risk minimization with hyperparameters tuned using the out-of-distribution (OOD) vs. in-distribution (ID) validation set.
We excluded OGB-MolPCBA, RxRx1-wilds, and GlobalWheat-wilds, as they do not have separate ID validation sets, and CivilComments-wilds, which is a subpopulation shift setting where we measure worst-group accuracy on a validation set that is already identically distributed to the training set.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | ID performance | | OOD performance | |
| Dataset | Metric | Tuned on ID val | Tuned on OOD val | Tuned on ID val | Tuned on OOD val |
| iWildCam2020-wilds | Macro F1 | 47.2 (2.0) | 47.0 (1.4) | 29.8 (0.6) | 31.0 (1.3) |
| Camelyon17-wilds | Average acc | 98.7 (0.1) | 93.2 (5.2) | 65.8 (4.9) | 70.3 (6.4) |
| FMoW-wilds | Worst-region acc | 58.0 (0.5) | 57.4 (0.2) | 31.9 (0.8) | 32.8 (0.5) |
| PovertyMap-wilds | Worst-U/R Pearson R | 0.65 (0.03) | 0.62 (0.04) | 0.46 (0.06) | 0.46 (0.07) |
| Amazon-wilds | 10th percentile acc | 72.0 (0.0) | 71.9 (0.1) | 53.8 (0.8) | 53.8 (0.8) |
| Py150-wilds | Method/class acc | 75.6 (0.0) | 75.4 (0.4) | 67.9 (0.1) | 67.9 (0.1) |

### D.2 Replicates

We typically use a fixed train/validation/test split and report results averaged across 3 replicates (random seeds for model initialization and minibatch order), as well as the unbiased standard deviation over those replicates.
There are three exceptions to this.
For PovertyMap-wilds, we report results averaged over 5-fold cross validation, as model training is relatively fast on this dataset. For Camelyon17-wilds, results vary substantially between replicates, so we report results averaged over 10 replicates instead. Similarly, for CivilComments-wilds, we report results averaged over 5 replicates.

### D.3 Baseline algorithms

For all classification datasets, we train models against the cross-entropy loss. For the PovertyMap-wilds regression dataset, we use the mean-squared-error loss.

We adapted the implementations of CORAL from Gulrajani and Lopez-Paz ([2020](#bib.bib158));
IRM from Arjovsky et al. ([2019](#bib.bib16)); and Group DRO from Sagawa et al. ([2020a](#bib.bib327)).
We note that CORAL was originally proposed in the context of domain adaptation (Sun and Saenko, [2016](#bib.bib356)), where it was shown to substantially improve performance on standard domain adaptation benchmarks, and it was subsequently adapted for domain generalization (Gulrajani and Lopez-Paz, [2020](#bib.bib158)).

Following these implementations, we use minibatch stochastic optimizers to train models under each algorithm, and we sample uniformly from each domain regardless of the number of training examples in it.
This means that the CORAL and IRM algorithms optimize for their respective penalty terms plus a reweighted ERM objective that weights each domain equally (i.e., effectively upweighting minority domains).
The Group DRO objective is unchanged, as it still optimizes for the domain with the worst loss, but the uniform sampling improves optimization stability.

Both CORAL and IRM are designed for models with featurizers, i.e., models that first map each input to a feature representation and then predict based on the representation.
To estimate the feature distribution for a domain, these algorithms need to see a sufficient number of examples from that domain in a minibatch.
However, some of our datasets have large numbers of domains, making it infeasible for each minibatch to contain examples from all domains.
For these algorithms, our data loaders form a minibatch by first sampling a few domains, and then sampling examples from those domains.
For consistency in our experiments, we used the same total batch size for these algorithms and for ERM and Group DRO, with a default of 8 examples per domain in each minibatch (e.g., if the batch size was 32, then in each minibatch we would have 8 examples ×\times 4 domains).

For Group DRO, as in Sagawa et al. ([2020a](#bib.bib327)), each example in the minibatch is sampled independently with uniform probabilities across domains, and therefore each minibatch does not need to only comprise a small number of domains.
We note that reweighting methods like Group DRO are effective only when the training loss is non-vanishing, which we achieve through early stopping (Byrd and Lipton, [2019](#bib.bib69); Sagawa et al., [2020a](#bib.bib327), [b](#bib.bib328)).

## E Additional dataset details and results

In this section, we discuss each Wilds dataset in more detail. For completeness, we start by repeating the motivation behind each dataset from Section [4](#S4 "4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We then describe the task, the distribution shift, and the evaluation criteria, and present baseline results that elaborate upon those in
Sections [5](#S5 "5 Performance drops from distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and [6](#S6 "6 Baseline algorithms for distribution shifts ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We also discuss the broader context behind each dataset and how it connects with other distribution shifts in similar applications.
Finally, we describe how each dataset was modified from its original
version in terms of the evaluation, splits, and data.
Unless otherwise specified, all experiments follow the protocol laid out in Appendix [D](#S4a "D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### E.1 iWildCam2020-wilds

Animal populations have declined 68% on average since 1970 (Grooten et al., [2020](#bib.bib156)).
To better understand and monitor wildlife biodiversity loss, ecologists commonly deploy camera traps—heat or motion-activated static cameras placed in the wild (Wearn and Glover-Kapfer, [2017](#bib.bib397))—and then use ML models to process the data collected (Weinstein, [2018](#bib.bib399); Norouzzadeh et al., [2019](#bib.bib277); Tabak et al., [2019](#bib.bib363); Beery et al., [2019](#bib.bib38); Ahumada et al., [2020](#bib.bib5)).
Typically, these models would be trained on photos from some existing camera traps and then used across new camera trap deployments.
However, across different camera traps, there is drastic variation in illumination, color, camera angle, background, vegetation, and relative animal frequencies,
which results in models generalizing poorly to new camera trap deployments (Beery et al., [2018](#bib.bib36)).

We study this shift on a variant of the iWildCam 2020 dataset (Beery et al., [2020a](#bib.bib37)).

#### E.1.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the domains are camera traps, and we seek to learn models that generalize to photos taken from new camera deployments (Figure [3](#S4.F3 "Figure 3 ‣ 4.1.1 iWildCam2020-wilds: Species classification across different camera traps ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The task is multi-class species classification.
Concretely, the input x𝑥x is a photo taken by a camera trap,
the label y𝑦y is one of 182 different animal species,
and the domain d𝑑d is an integer that identifies the camera trap that took the photo.

##### Data.

The dataset comprises 203,029 images from 323 different camera traps spread across multiple countries in different parts of the world. The original camera trap data comes from the Wildlife Conservation Society (<http://lila.science/datasets/wcscameratraps>).
These images tend to be taken in short bursts following motion-activation of a camera trap, so the images can be additionally grouped into sequences of images from the same burst, though our baseline models do not exploit this information and our evaluation metric treats each image individually.
Each image is associated with the following metadata: camera trap ID, sequence ID, and datetime.

As is typical for camera trap data, approximately 35% of the total number of images are empty (i.e., do not contain any animal species); this corresponds to one of the 182 class labels. The ten most common classes across the full dataset are “empty” (34%), ocellated turkey (8%), great curassow (6%), impala (4%), black-fronted duiker (4%), white-lipped peccary (3%), Central American agouti (3%), ocelot (3%), gray fox (2%) and cow (2%).

We note that the labels in this dataset can be somewhat noisy, as is typical of camera trap data. Some ecologists might label all images in a sequence as the same animal (which can result in empty/dark frames being labeled as an animal), whereas other ecologists might try to label it frame-by-frame. This label noise imposes a natural ceiling on model performance, though the label noise is equally present in ID vs. OOD data.

We split the dataset by randomly partitioning the data by camera traps:

1. 1.

   Training: 129,809 images taken by 243 camera traps.
2. 2.

   Validation (OOD): 14,961 images taken by 32 different camera traps.
3. 3.

   Test (OOD): 42,791 images taken by 48 different camera traps.
4. 4.

   Validation (ID): 7,314 images taken by the same camera traps as the training set, but on different days from the training and test (ID) images.
5. 5.

   Test (ID): 8,154 images taken by the same camera traps as the training set, but on different days from the training and validation (ID) images.

The camera traps are randomly distributed across the training, validation (OOD), and test (OOD) sets. The number of examples per location vary widely from 1 to 8494, with a median of 194 images (Figure [16](#S5.F16 "Figure 16 ‣ Data. ‣ E.1.1 Setup ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
All images from the same sequence (i.e., all images taken in the same burst) are placed together in the same split. See Appendix [E.1.4](#S5.SS1.SSS4 "E.1.4 Additional details ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more details.

![Refer to caption](/html/2012.07421/assets/x15.png)


Figure 16: Number of examples per location in the iWildCam2020-wilds dataset. The locations are sorted such that locations with the least amount of examples are to the left on the x-axis.

##### Evaluation.

We evaluate models by their macro F1 score (i.e., we compute the F1 score for each class separately, then average those scores). We also report the average accuracy of each model across all test images, but primarily use the macro F1 score to better capture model performance on rare species. In the natural world, protected and endangered species are rare by definition, and are often the most important to accurately monitor. However, common species are much more likely to be captured in camera trap images; this imbalance can make metrics like average accuracy an inaccurate picture of model effectiveness.

##### Potential leverage.

Though the problem is challenging for existing ML algorithms, adapting to photos from different camera traps is simple and intuitive for humans. Repeated backgrounds and habitual animals, which cause each sensor to have a unique class distribution, provide a strong implicit signal across data from any one location.
We anticipate that approaches that utilize the provided camera trap annotations can learn to factor out these common features and avoid learning spurious correlations between particular backgrounds and animal species.

#### E.1.2 Baseline results

![Refer to caption](/html/2012.07421/assets/x16.png)


Figure 17: Label distribution for each iWildCam2020-wilds split.

##### Model.

For all experiments, we use ResNet-50 models (He et al., [2016](#bib.bib167)) that were pretrained on ImageNet,
using a learning rate of 3e-5 and no L2subscript𝐿2L\_{2}-regularization.
As input, these models take in images resized to 448 by 448.
We trained these models with the Adam optimizer and a batch size of 16 for 12 epochs. To pick hyperparameters, we did a grid search over learning rates {1×10−5,3×10−5,1×10−4}1superscript1053superscript1051superscript104\{1\times 10^{-5},3\times 10^{-5},1\times 10^{-4}\} and L2subscript𝐿2L\_{2} regularization strengths {0,1×10−3,1×10−2}01superscript1031superscript102\{0,1\times 10^{-3},1\times 10^{-2}\}.
We report results aggregated over 3 random seeds.

##### ERM results and performance drops.

Model performance dropped substantially and consistently going from the train-to-train in-distribution (ID) setting to the official out-of-distribution (OOD) setting (Table [4](#S5.T4 "Table 4 ‣ Additional baseline methods. ‣ E.1.2 Baseline results ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")),
with a macro F1 score of 47.0 on the ID test set but only 31.0 on the OOD test set.
We note that macro F1 and average accuracy both differ between the OOD validation and test sets: this is in part due to the difference in class balance between them, which in turn is due to differences in the proportion of classes across camera traps.
In particular, macro F1 can vary between splits because we take the average F1 score across all classes that are present in the evaluation split, and not all splits have the same classes present (e.g., a rare species might be in the OOD validation set but not OOD test set, or vice versa). In additional, average accuracy can differ between splits due in part to variation in the fraction of empty images per location (e.g., the camera traps that were randomly assigned to the OOD validation set have a smaller proportion of empty images).

We only ran a train-to-train comparison because there are a relatively large number of domains (camera traps) split i.i.d. between the training and test sets, which suggests that the training and test sets should be “equally difficult”. The size of the ID-OOD gap in macro F1 is large enough that we expect it should hold up even in a test-to-test comparison.
However, the results in Table [4](#S5.T4 "Table 4 ‣ Additional baseline methods. ‣ E.1.2 Baseline results ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and Figure [17](#S5.F17 "Figure 17 ‣ E.1.2 Baseline results ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") show that there is substantial variability between domains, and it would be useful for future work to establish the magnitude of the ID-OOD gap under the test-to-test or mixed-to-test comparisons.

##### Additional baseline methods.

Table 4: Baseline results on iWildCam2020-wilds. In-distribution (ID) results correspond to the train-to-train setting. Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Val (ID) | | Val (OOD) | | Test (ID) | | Test (OOD) | |
| Algorithm | Macro F1 | Avg acc | Macro F1 | Avg acc | Macro F1 | Avg acc | Macro F1 | Avg acc |
| ERM | 48.8 (2.5) | 82.5 (0.8) | 37.4 (1.7) | 62.7 (2.4) | 47.0 (1.4) | 75.7 (0.3) | 31.0 (1.3) | 71.6 (2.5) |
| CORAL | 46.7 (2.8) | 81.8 (0.4) | 37.0 (1.2) | 60.3 (2.8) | 43.5 (3.5) | 73.7 (0.4) | 32.8 (0.1) | 73.3 (4.3) |
| IRM | 24.4 (8.4) | 66.9 (9.4) | 20.2 (7.6) | 47.2 (9.8) | 22.4 (7.7) | 59.9 (8.1) | 15.1 (4.9) | 59.8 (3.7) |
| Group DRO | 42.3 (2.1) | 79.3 (3.9) | 26.3 (0.2) | 60.0 (0.7) | 37.5 (1.7) | 71.6 (2.7) | 23.9 (2.1) | 72.7 (2.0) |
| Reweighted (label) | 42.5 (0.5) | 77.5 (1.6) | 30.9 (0.3) | 57.8 (2.8) | 42.2 (1.4) | 70.8 (1.5) | 26.2 (1.4) | 68.8 (1.6) |

We trained models with CORAL, IRM, and Group DRO, treating each camera trap as a domain, and using the same model hyperparameters as ERM.
These did not improve upon the ERM baseline (Table [4](#S5.T4 "Table 4 ‣ Additional baseline methods. ‣ E.1.2 Baseline results ‣ E.1 iWildCam2020-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The IRM models performed especially poorly on this dataset; we suspect that this is because the default estimator of the IRM penalty term can be negatively biased when examples are sampled without replacement from small domains, but further investigation is needed.
We also tried reweighting the training data so that each label had equal weight, but this did not improve over ERM either.

##### Discussion.

Across locations, there is drastic variation in illumination, camera angle, background, vegetation, and color. This variation, coupled with considerable differences in the distribution of animals between camera traps, likely encourages the model to overfit to specific animal species appearing in specific locations, which may account for the performance drop.

The original iWildCam 2020 competition allows users to use MegaDetector (Beery et al., [2019](#bib.bib38)), which is an animal detector trained on a large set of data beyond what is provided in the training set.
Using an animal detection model like MegaDetector typically improves classification performance on camera traps (Beery et al., [2018](#bib.bib36)).
However, we intentionally do not use MegaDetector in our baselines for iWildCam2020-wilds for two reasons.
First, though the trained MegaDetector model is publicly available, the MegaDetector training set is not, which makes it difficult to build on top of it and run controlled experiments.
Second, bounding box annotations are costly and harder to obtain, and there is much more data with image-level species label, so it would be useful to be able to train models that do not have to rely on bounding box annotations.

We still welcome leaderboard submissions that use MegaDetector, as it is useful to see how much better models can perform when they use MegaDetector or other similar animal detectors, but we will distinguish these submissions from others that only use what is provided in the training set.

A different source of leverage comes from the temporal signal in the camera trap images, which are organized into sequences that each correspond to a burst of images from a single motion trigger.
Using this sequence information (e.g., by taking the median prediction across a sequence) can also improve model performance (Beery et al., [2018](#bib.bib36)),
and we welcome submissions that explore this direction.

#### E.1.3 Broader context

Differences across data distributions at different sensor locations is a common challenge in automated wildlife monitoring applications, including using audio sensors to monitor animals that are easier heard than seen such as primates, birds, and marine mammals (Crunchant et al., [2020](#bib.bib98); Stowell et al., [2019](#bib.bib354); Shiu et al., [2020](#bib.bib345)), and using static sonar to count fish underwater to help maintain sustainable fishing industries (Pipal et al., [2012](#bib.bib298); Vatnehol et al., [2018](#bib.bib384); Schneider and Zhuang, [2020](#bib.bib333)).
As with camera traps, each static audio sensor has a specific species distribution as well as a sensor specific background noise signature, making generalization to new sensors challenging. Similarly, static sonar used to measure fish escapement have sensor-specific background reflectance based on the shape of the river bottom.
Moreover, since species are distributed in a non-uniform and long-tailed fashion across the globe, it is incredibly challenging to collect sufficient samples for rare species to escape the low-data regime.
Implicitly representing camera-specific distributions and background features in per-camera memory banks and extracting relevant information from these via attention has been shown to help overcome some of these challenges for static cameras (Beery et al., [2020b](#bib.bib39)).

More broadly, shifts in background, image illumination and viewpoint have been studied in computer vision research. First, several works have shown that object classifiers often rely on the background rather than the object to make its classification (Rosenfeld et al., [2018](#bib.bib322); Shetty et al., [2019](#bib.bib341); Xiao et al., [2020](#bib.bib414)). Second, common perturbations such as blurriness or shifts in illumination, tend to reduce performance (Dodge and Karam, [2017](#bib.bib114); Temel et al., [2018](#bib.bib371); Hendrycks and Dietterich, [2019](#bib.bib172)).
Finally, shifts in rotation and viewpoint of the object has been shown to degrade performance (Barbu et al., [2019](#bib.bib29)).

#### E.1.4 Additional details

##### Data processing.

We generate the data splits in three steps. First, to generate the OOD splits, we randomly split all locations into three groups: Validation (OOD), Test (OOD), and Others. Then, to generate the train-to-train ID splits, we split the Others group uniformly by date at random into three sets: Training, Validation (ID), and Test (ID).

When doing the ID split, some locations only ended up in some of but not all of Training, Validation (ID), and Test (ID). For instance, if there were very few dates for a specific location (camera trap), it may be that no examples from that location ended up in the train split. This defeats the purpose of the ID split, which is to test performance on locations that were seen during training. We therefore put these locations in the train split. Finally, any images in the test set with classes not present in the train set were removed.

##### Modifications to the original dataset.

The original iWildCam 2020 Kaggle competition similarly split the dataset by camera trap, though the competition focused on average accuracy. We consider a smaller subset of the data here.
Specifically, the Kaggle competition uses a held-out test set that we are not utilizing, as the test set is intended to be reused in a future competition and is not yet public.
Instead, we constructed our own test set by splitting the Kaggle competition training data into our own splits: train, validation (ID), validation (OOD), test (ID), test (OOD).

Images are organized into sequences, but we treat each image separately. In the iWildCam 2020 competition, the top participants utilized the sequence data and also used a pretrained MegaDetector animal detection model that outputs bounding boxes over the animals. These images are cropped using the bounding boxes and then fed into a classification network. As we discuss above, we intentionally do not use MegaDetector in our experiments.

In addition, compared to the iWildCam 2020 competition, the iWildCam 2021 competition changed several class definitions (such as removing the “unknown” class) and removed some images that were taken indoors or had humans in the background. We have applied these updates to iWildCam2020-wilds as well.

### E.2 Camelyon17-wilds

Models for medical applications are often trained on data from a small number of hospitals, but with the goal of being deployed more generally across other hospitals.
However, variations in data collection and processing can degrade model accuracy on data from new hospital deployments (Zech et al., [2018](#bib.bib427); AlBadawy et al., [2018](#bib.bib7)).
In histopathology applications—studying tissue slides under a microscope—this variation can arise from sources like differences in the patient population or in slide staining and image acquisition (Veta et al., [2016](#bib.bib387); Komura and Ishikawa, [2018](#bib.bib211); Tellez et al., [2019](#bib.bib370)).

We study this shift on a patch-based variant of the Camelyon17 dataset (Bandi et al., [2018](#bib.bib28)).

#### E.2.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the domains are hospitals, and our goal is to learn models that generalize to data from a hospital that is not in the training set (Figure [4](#S4.F4 "Figure 4 ‣ 4.1.1 iWildCam2020-wilds: Species classification across different camera traps ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The task is to predict if a given region of tissue contains any tumor tissue,
which we model as binary classification.
Concretely, the input x𝑥x is a 96x96 histopathological image, the label y𝑦y is a binary indicator of whether the central 32x32 region contains any tumor tissue, and the domain d𝑑d is an integer that identifies the hospital that the patch was taken from.

##### Data.

The dataset comprises 450,000 patches extracted from 50 whole-slide images (WSIs) of breast cancer metastases in lymph node sections, with 10 WSIs from each of 5 hospitals in the Netherlands.
Each WSI was manually annotated with tumor regions by pathologists, and the resulting segmentation masks were used to determine the labels for each patch.
We also provide metadata on which slide (WSI) each patch was taken from, though our baseline algorithms do not use this metadata.

We split the dataset by domain (i.e., which hospital the patches were taken from):

1. 1.

   Training: 302,436 patches taken from 30 WSIs, with 10 WSIs from each of the 3 hospitals in the training set.
2. 2.

   Validation (OOD): 34,904 patches taken from 10 WSIs from the 4th hospital. These WSIs are distinct from those in the other splits.
3. 3.

   Test (OOD): 85,054 patches taken from 10 WSIs from the 5th hospital, which was chosen because its patches were the most visually distinctive. These WSIs are also distinct from those in the other splits.
4. 4.

   Validation (ID): 33,560 patches taken from the same 30 WSIs from the training hospitals.

We do not provide a Test (ID) set, as there is no practical setting in which we would have labels on a uniformly randomly sampled set of patches from a WSI, but no labels on the other patches from the same WSI.

##### Evaluation.

We evaluate models by their average test accuracy across patches. Histopathology datasets can be unwieldy for ML models, as individual images can be several gigabytes large; extracting patches involves many design choices; the classes are typically very unbalanced; and evaluation often relies on more complex slide-level measures such as the free-response receiver operating characteristic (FROC) (Gurcan et al., [2009](#bib.bib161)).
To improve accessibility, we pre-process the slides into patches and balance the dataset so that each split has a 50/50 class balance, making average accuracy is a reasonable measure of performance (Veeling et al., [2018](#bib.bib385); Tellez et al., [2019](#bib.bib370)).

##### Potential leverage.

Prior work has shown that differences in staining between hospitals are the primary source of variation in this dataset, and that specialized stain augmentation methods can close the in- and out-of-distribution accuracy gap on a variant of the dataset based on the same underlying slides (Tellez et al., [2019](#bib.bib370)).
However, the general task of learning histopathological models that are robust to variation across hospitals (from staining and other sources) is still an open research question.
In this way, the Camelyon17-wilds dataset is a controlled testbed for general-purpose methods that can learn to be robust to stain variation between hospitals, given a training set from multiple hospitals.

#### E.2.2 Baseline results

Table 5: Baseline results on Camelyon17-wilds. In-distribution (ID) results correspond to the train-to-train setting. Parentheses show standard deviation across 10 replicates. Note that the standard error of the mean is smaller (by a factor of 1010\sqrt{10}).

|  |  |  |  |
| --- | --- | --- | --- |
| Algorithm | Validation (ID) accuracy | Validation (OOD) accuracy | Test (OOD) accuracy |
| ERM | 93.2 (5.2) | 84.9 (3.1) | 70.3 (6.4) |
| CORAL | 95.4 (3.6) | 86.2 (1.4) | 59.5 (7.7) |
| IRM | 91.6 (7.7) | 86.2 (1.4) | 64.2 (8.1) |
| Group DRO | 93.7 (5.2) | 85.5 (2.2) | 68.4 (7.3) |




Table 6: Mixed-to-test comparison for ERM models on Camelyon17-wilds.
In the official OOD setting, we train on data from three hospitals and evaluate on a different test hospital, whereas in the mixed-to-test ID setting,
we add data from one extra slide from the test hospital to the training set.
The official Test (OOD) set has data from 10 slides, but for this comparison, we report performance for both splits on the same 9 slides (without the slide that was moved to the training set).
This makes the numbers (71.0 vs. 70.3) for the official split slightly different from Table [5](#S5.T5 "Table 5 ‣ E.2.2 Baseline results ‣ E.2 Camelyon17-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Parentheses show standard deviation across 10 replicates.
Note that the standard error of the mean is smaller (by a factor of 1010\sqrt{10}).

|  |  |  |
| --- | --- | --- |
| Setting | Algorithm | Test (OOD) accuracy |
| Official (train on ID examples) | ERM | 71.0 (6.3) |
| Mixed-to-test (train on ID + OOD examples) | ERM | 82.9 (9.8) |

##### Model.

For all experiments, we use DenseNet-121 models (Huang et al., [2017](#bib.bib183)) models trained from scratch on the 96 ×\times 96 patches, following prior work (Veeling et al., [2018](#bib.bib385)). These models used a learning rate of 10−3superscript10310^{-3}, L2subscript𝐿2L\_{2}-regularization strength of 10−2superscript10210^{-2}, a batch size of 32, and SGD with momentum (set to 0.9), trained for 5 epochs with early stopping.
We selected hyperparameters by a grid search over learning rates {10−4\{10^{-4}, 10−3superscript10310^{-3}, 10−2}10^{-2}\}, and L2subscript𝐿2L\_{2}-regularization strengths {0,10−3,10−2}0superscript103superscript102\{0,10^{-3},10^{-2}\}.
We report results aggregated over 10 random seeds.

##### ERM results and performance drops.

Table [5](#S5.T5 "Table 5 ‣ E.2.2 Baseline results ‣ E.2 Camelyon17-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that the model was consistently accurate on the train-to-train in-distribution (ID) validation set and to a lesser extent on the out-of-distribution (OOD) validation set, which was from a held-out hospital.
However, it was wildly inconsistent on the test set, which was from a different held-out hospital,
with a standard deviation of 6.4% in accuracies across 10 random seeds.
There is a large gap between train-to-train ID validation and OOD validation accuracy, and between OOD validation and OOD test accuracy (in part because we early stop on the highest OOD validation accuracy).
Nevertheless, we found that using the OOD validation set gave better results than using the ID validation set; see Appendix [D.1](#S4.SS1a "D.1 Model hyperparameters ‣ D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more discussion.

We ran an additional mixed-to-test comparison, where we moved 1 of the 10 slides777This slide was randomly chosen and corresponded to about 6% of the test patches; some slides contribute more patches than others because they contain larger tumor regions. from the test hospital to the training set and tested on the patches from the remaining 9 slides.
The mixed-to-test setting gives significantly higher accuracy on the reduced test set (Table [6](#S5.T6 "Table 6 ‣ E.2.2 Baseline results ‣ E.2 Camelyon17-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), suggesting that the observed performance drop is due to the distribution shift, as opposed to the intrinsic difficulty of the examples from the test hospital.
We note that this mixed-to-test comparison mixes in only a small amount of test data and is therefore likely to be an underestimate of in-distribution performance on the test set; we opted to only mix in 1 slide so as to preserve enough test examples to be able to accurately estimate model performance.

##### Additional baseline methods.

We trained models with CORAL, IRM, and Group DRO, treating each hospital as a domain.
However, they performed comparably or worse than the ERM baseline.
For the CORAL and IRM models, our grid search selected the lowest values of their penalty weights (0.1 and 1, respectively) based on OOD validation accuracy.

![Refer to caption](/html/2012.07421/assets/x17.png)


Figure 18: 
Test (OOD) accuracy versus validation (OOD) accuracy for different random seeds on Camelyon17-wilds, using the same hyperparameters. The test accuracy is far more variable than the validation accuracy (note the differences in the axes), in part because we early stop on the highest OOD validation accuracy.

##### Discussion.

These results demonstrate a subtle failure mode when considering out-of-distribution accuracy: there are models (i.e., choices of hyperparameters and random seeds) that do well both in- and out-of-distribution, but we cannot reliably choose these models from just the training/validation set.
Due to the substantial variability in test accuracy on Camelyon17-wilds (see Figure [18](#S5.F18 "Figure 18 ‣ Additional baseline methods. ‣ E.2.2 Baseline results ‣ E.2 Camelyon17-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), we ask researchers to submit leaderboard submissions with results from 10 random seeds, instead of the 3 random seeds required for other datasets.

Many specialized methods have been developed to handle stain variation in the context of digital histopathology. These typically fall into one of two categories: data augmentation methods that perturb the colors in the training images (e.g., Liu et al. ([2017](#bib.bib244)); Bug et al. ([2017](#bib.bib65)); Tellez et al. ([2018](#bib.bib369)))
or stain normalization methods that seek to standardize colors across training images (e.g., Macenko et al. ([2009](#bib.bib253)); BenTaieb and Hamarneh ([2017](#bib.bib44))).
These methods are reasonably effective at mitigating stain variation, at least in some contexts (Tellez et al., [2019](#bib.bib370); Miller et al., [2021](#bib.bib265)), though the general problem of learning digital histopathology models that can be effectively deployed across multiple hospitals/sites is still an open challenge.

To facilitate more controlled experiments, we will have two leaderboard tracks for Camelyon17-wilds.
For the first track, which focuses on general-purpose algorithms, submissions should not use color-specific techniques (e.g., color augmentation) and should also train their models from scratch, instead of fine-tuning models that are pre-trained from ImageNet or other datasets.
For the second track, submissions can use any of those techniques, including specialized methods for dealing with stain variation.
These separate tracks will help to disentangle the contributions of more general-purpose learning algorithms and model architectures from the contributions of specialized augmentation techniques or additional training data.

#### E.2.3 Broader context

Other than stain variation, there are many other distribution shifts that might occur in histopathology applications. For example, patient demographics might differ from hospital to hospital: some hospitals might tend to see patients who are older or more sick, and patients from different backgrounds and countries vary in terms of cancer susceptibility (Henderson et al., [2012](#bib.bib171)).
Some cancer subtypes and tissues of origin are also more common than others, leading to potential subpopulation shift issues, e.g,. a rare cancer subtype in one context might be more common in another; or even if it remains rare, we would seek to leverage the greater quantity of data from other subtypes to improve model accuracy on the rare subtype (Weinstein et al., [2013](#bib.bib400)).

Beyond histopathology, variation between different hospitals and deployment sites has also been shown to degrade model accuracy in other medical applications such as diabetic retinopathy (Beede et al., [2020](#bib.bib35)) and chest radiographs (Zech et al., [2018](#bib.bib427); Phillips et al., [2020](#bib.bib294)), including recent work on COVID-19 detection (DeGrave et al., [2020](#bib.bib107)).
Even within the same hospital, process variables like which scanner/technician took the image can significantly affect models (Badgeley et al., [2019](#bib.bib24)).

In these medical applications, the gold standard is to evaluate models on an independent test set collected from a different hospital (e.g., Beck et al. ([2011](#bib.bib33)); Liu et al. ([2017](#bib.bib244)); Courtiol et al. ([2019](#bib.bib96)); McKinney et al. ([2020](#bib.bib261))) or at least with a different scanner within the same hospital (e.g., Campanella et al. ([2019](#bib.bib73))).
However, this practice has not been ubiquitous due to the difficulty of obtaining data spanning multiple hospitals (Esteva et al., [2017](#bib.bib125); Bejnordi et al., [2017](#bib.bib40); Codella et al., [2019](#bib.bib86); Veta et al., [2019](#bib.bib388)).
The baseline results reported above show that even evaluating on a single different hospital might be insufficient, as results can vary widely between different hospitals (e.g., between the validation and test OOD datasets).
We hope that the Camelyon17-wilds dataset, which has multiple hospitals in the training set and independent hospitals in the validation and test sets, will be useful for developing models that can generalize reliably to new hospitals and contexts (Chen et al., [2020](#bib.bib79)).

#### E.2.4 Additional details

##### Data processing.

The Camelyon17-wilds dataset is adapted from whole-slide images (WSIs) of breast cancer metastases in lymph nodes sections, obtained from the CAMELYON17 challenge (Bandi et al., [2018](#bib.bib28)).
Each split is balanced to have an equal number of positive and negative examples.
The varying number of patches per slide and hospital is due to this class balancing, as some slides have fewer tumor (positive) patches.
We selected the test set hospital as the one whose patches were visually most distinct; the difference in test versus OOD validation performance shows that the choice of OOD hospital can significantly affect performance.

From these WSIs, we extracted patches in a standard manner, similar to Veeling et al. ([2018](#bib.bib385)).
The WSIs were scanned at a resolution of 0.23μ𝜇\mum–0.25μ𝜇\mum in the original dataset, and each WSI contains multiple resolution levels, with approximately 10,000×\times20,000 pixels at the highest resolution level
(Bandi et al., [2018](#bib.bib28)).
We used the third-highest resolution level, corresponding to reducing the size of each dimension by a factor of 4.
We then tiled each slide with overlapping 96×\times96 pixel patches with a step size of 32 pixels in each direction (such that none of the central 32×\times32 regions overlap), labeling them as the following:

* •

  *Tumor* patches have at least one pixel of tumor tissue in the central 32×\times32 region. We used the pathologist-annotated tumor annotations provided with the WSIs.
* •

  *Normal* patches have no tumor and have at least 20% normal tissue in the central 32×\times32 region. We used Otsu thresholding to distinguish normal tissue from background.

We discarded all patches that had no tumor and <20% normal tissue in the central 32×\times32 region.

To maintain an equal class balance, we then subsampled the extracted patches in the following way.
First, for each WSI, we kept all tumor patches unless the WSI had fewer normal than tumor patches, which was the case for a single WSI; in that case, we randomly discarded tumor patches from that WSI until the numbers of tumor and normal patches were equal.
Then, we randomly selected normal patches for inclusion such that for each hospital and split, there was an equal number of tumor and normal patches.

##### Modifications to the original dataset.

The task in the original CAMELYON17 challenge (Bandi et al., [2018](#bib.bib28)) was the patient-level classification task of determining the pathologic lymph node stage of the tumor present in all slides from a patient. In contrast, our task is a lesion-level classification task. Patient-level, slide-level, and lesion-level tasks are all common in histopathology applications.
As mentioned above, the original dataset provided WSIs and tumor annotations,
but not a standardized set of patches, which we provide here.
Moreover, it did not consider distribution shifts; both of the original training and test splits contained slides from all 5 hospitals.

The Camelyon17-wilds patch-based dataset is similar to one of the datasets used in Tellez et al. ([2019](#bib.bib370)), which was also derived from the CAMELYON17 challenge; there, only one hospital is used as the training set, and the other hospitals are all part of the test set.
Camelyon17-wilds is also similar to PCam (Veeling et al., [2018](#bib.bib385)), which is a patch-based dataset based on an earlier CAMELYON16 challenge; the data there is derived from only two hospitals.

##### Additional data sources.

The full, original CAMELYON17 dataset contains 1000 WSIs from the same 5 hospitals, although only 50 of them (which we use here) have tumor annotations. The other 950 WSIs may be used as unlabeled data. Beyond the CAMELYON17 dataset, the largest source of unlabeled WSI data is the Cancer Genome Atlas (Weinstein et al., [2013](#bib.bib400)), which typically has patient-level annotations (e.g., patient demographics and clinical outcomes).

### E.3 RxRx1-wilds

High-throughput screening techniques that can generate large amounts of data
are now common in many fields of biology,
including transcriptomics (Harrill et al., [2019](#bib.bib165)),
genomics (Echeverri and Perrimon, [2006](#bib.bib121); Zhou et al., [2014](#bib.bib434)), proteomics and
metabolomics (Taylor et al., [2021](#bib.bib368)), and drug
discovery (Broach et al., [1996](#bib.bib60); Macarron et al., [2011](#bib.bib252); Swinney and Anthony, [2011](#bib.bib361); Boutros et al., [2015](#bib.bib57)).
Such large volumes of data, however, need to be created in experimental batches, or groups of experiments executed at similar times under similar conditions.
Despite attempts to carefully control experimental variables such as
temperature, humidity, and reagent concentration, measurements from
these screens are confounded by technical artifacts that arise from differences
in the execution of each batch.
These *batch effects* make it difficult to draw conclusions from data across experimental batches (Leek et al., [2010](#bib.bib228); Parker and Leek, [2012](#bib.bib287); Soneson et al., [2014](#bib.bib349); Nygaard et al., [2016](#bib.bib278); Caicedo et al., [2017](#bib.bib70)).

We study the shift induced by batch effects on a variant of the RxRx1-wilds dataset (Taylor et al., [2019](#bib.bib367)).
As illustrated in Figure [5](#S4.F5 "Figure 5 ‣ 4.1.3 RxRx1-wilds: Genetic perturbation classification across experimental batches ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), there are significant visual
differences between experimental batches, making recognizing siRNA perturbations
in OOD experiments in the RxRx1-wilds dataset a particularly challenging task for
existing ML algorithms.

#### E.3.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the domains are
experimental batches and we seek to generalize to images from unseen experimental
batches. Concretely, the input x𝑥x is a 3-channel image of cells obtained by
fluorescent microscopy, the label y𝑦y indicates which of the 1,139 genetic
treatments (including no treatment) the cells received, and the domain d𝑑d
specifies the experimental batch of the image.

##### Data.

RxRx1-wilds was created by Recursion (recursion.com) in its automated high-throughput
screening laboratory in Salt Lake City, Utah. It is comprised of fluorescent microscopy images of
human cells in four different cell lines: HUVEC, RPE, HepG2, and U2OS. These
were acquired via fluorescent microscopy using a 6-channel variant of the
*Cell Painting* assay (Bray et al., [2016](#bib.bib59)).
Figure [19](#S5.F19 "Figure 19 ‣ Data. ‣ E.3.1 Setup ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows an example of the cellular contents of each of
these 6 channels: nuclei, endoplasmic reticuli, actin, nucleoli and cytoplasmic
RNA, mitochondria, and Golgi.
To make the dataset smaller and more accessible, we only included the first 3 channels in RxRx1-wilds.

![Refer to caption](/html/2012.07421/assets/figures/rxrx1_channels.png)


Figure 19: 6-channel composite image of HUVEC cells (left) and its
individual channels (rest): nuclei (blue), endoplasmic reticuli (green),
actin (red), nucleoli and cytoplasmic RNA (cyan), mitochondria (magenta),
and Golgi (yellow). The overlap in channel content is due in part to the
lack of complete spectral separation between fluorescent stains. Note that
only the first 3 channels are included in RxRx1-wilds.

The images in RxRx1-wilds are the result of executing the same experimental
design 51 different times, each in a different batch of experiments. The design
consists of four 384-well plates, where individual wells are used to isolate
populations of cells on each plate (see Figure [20](#S5.F20 "Figure 20 ‣ Data. ‣ E.3.1 Setup ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

![Refer to caption](/html/2012.07421/assets/figures/rxrx1_plate.png)


Figure 20: Schematic of a 384-well plate demonstrating imaging sites and
6-channel images. The 4-plate experiments in RxRx1-wilds were run in the wells
of such 384-well plates. RxRx1-wilds contains two imaging sites per well.

The wells are laid out in a 16×\times24 grid, but only the wells in the inner
14×\times22 grid are used since the outer wells are most susceptible to
environmental factors. Of these 308 usable wells, one is left untreated to
provide a *negative control* phenotype, while the rest are treated with
small interfering ribonucleic acid, or siRNA, at a fixed concentration. Each
siRNA is designed to knockdown a single target gene via the RNA interference
pathway, reducing the expression of the gene and its associated
protein (Tuschl, [2001](#bib.bib378)). However, siRNAs are known to have significant but
consistent off-target effects via the microRNA pathway, creating partial
knockdown of many other genes as well. The overall effect of siRNA transfection
is to perturb the morphology, count, and distribution of cells, creating a
*phenotype* associated with each siRNA. The phenotype is sometimes visually
recognizable, but often the effects are subtle and hard to detect.

In each plate, 30 wells are set aside for 30 *positive control* siRNAs. Each has a different gene as its primary target, which together with the
single untreated well already mentioned, provides a set of reference phenotypes
per plate. Each of the remaining 1,108 wells of the design (277 wells ×\times 4
plates) receives one of 1,108 *treatment* siRNA, respectively, so that
there is at most one well of each treatment siRNA in each experiment. We say
at most once because, although rare, it happens that either an siRNA is not
correctly transferred into the designated destination well, resulting in an
additional untreated well, or an operational error is detected by quality
control procedures that render the well unsuitable for inclusion in the dataset.

Each experiment was run in a single cell type, and of the 51 experiments in RxRx1-wilds, 24 are in HUVEC, 11 in RPE, 11 in HepG2, and 5 in U2OS.
Figure [21](#S5.F21 "Figure 21 ‣ Data. ‣ E.3.1 Setup ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows the phenotype of the same siRNA in each of these
four cell types.

![Refer to caption](/html/2012.07421/assets/figures/rxrx1_celltypes.png)


Figure 21: Images of the same siRNA in four cell types, from left to right: HUVEC, RPE, HepG2, U2OS.

We split the dataset by experimental batches, with the training and test splits having roughly the same composition of cell types:

1. 1.

   Training: 33 experiments (16 HUVEC, 7 RPE, 7 HepG2, 3 U2OS),
   site 1 only = 40,612 images.
2. 2.

   Validation (OOD): 4 experiments (1 HUVEC, 1 RPE, 1 HepG2, 3
   U2OS), sites 1 and 2 = 9,854 images.
3. 3.

   Test (OOD): 14 experiments (7 HUVEC, 3 RPE, 3 HepG2, 1
   U2OS), sites 1 and 2 = 34,432 images.
4. 4.

   Test (ID): same 33 experiments as in the training set, site 2
   only = 40,612 images.

In addition to the class (siRNA), each image is associated with the following
metadata: cell type, experiment, plate, well, and site.
We emphasize that all the images of an experiment are found in exactly one of
the training, validation (OOD) or test (OOD) splits. See
Appendix [E.3.4](#S5.SS3.SSS4 "E.3.4 Additional details ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more data processing details.

##### Evaluation.

We evaluate models by their average accuracy across test images. Note that there
are two images per well in the test set, which we evaluate independently.

The cell types are not balanced in the training and test sets.
Correspondingly, we observed higher performance on the HUVEC cell type, which is over-represented, and lower performance on the U2OS cell type, which is under-represented.
While maintaining high performance on minority (or even unseen) cell types is an important problem, for RxRx1-wilds, we opt to measure the average accuracy across all experiments instead of, for example, the worst accuracy across cell types.
This is because the relatively small amount of training data available from the minority cell type (U2OS) makes it challenging to cast RxRx1-wilds as a tractable subpopulation shift problem.
We also note that the difference in performance across cell types leads to the validation performance being significantly lower than the test performance, as there is a comparatively smaller fraction of HUVEC and a comparatively higher fraction of U2OS.

##### Potential leverage.

By design, there is usually one sample per class per experiment in the training
set, with the following exceptions: 1) there are usually four samples per
positive control, though 2) samples may be missing, as described above.
Moreover, while batch effects can manifest themselves in many complicated ways,
it is the case that the training set consists of a large number of experiments
selected randomly amongst all experiments in the dataset, hence we
expect models to be able to learn what is common amongst all such samples per
cell type, and for that ability to generalize to to test batches. We emphasize
that, whether in the training or test sets, the same cell types are perturbed
with the same siRNA, and thus the phenotypic distributions for each batch share
much of the same generative process.

We also note that, while not exploited here, there is quite a bit of structure
in the RxRx1-wilds dataset. For example, except in the case of errors, all treatment
siRNA appear once in each experiment, and all control conditions appear once per
plate, so four times per experiment. Also, due to the operational efficiencies
gained, the 1,108 treatment siRNAs always appear in the same four groups of 277
per experiment. So while the particular well an siRNA appears in is randomized,
it will always appear with the same group of 276 other siRNAs. This structure
can be exploited for improving predictive accuracy via post-prediction methods
such as linear sum assignment. However, such methods do not represent improved
generalization to OOD samples, and should be avoided.

#### E.3.2 Baseline results

##### Model.

For all experiments, we train the standard ResNet-50 model
(He et al., [2016](#bib.bib167)) pretrained on ImageNet, using a learning rate of 1​e−41𝑒41e-4 and
L2subscript𝐿2L\_{2}-regularization strength of 1​e−51𝑒51e-5. We trained these models with the Adam
optimizer, using default parameter values β1=0.9subscript𝛽10.9\beta\_{1}=0.9 and β2=0.999subscript𝛽20.999\beta\_{2}=0.999, with a batch size of 75 for 90 epochs, linearly increasing the learning
rate for 10 epochs, then decreasing it following a cosine learning rate schedule.
We selected hyperparameters by a grid search over learning rates {10−5\{10^{-5}, 10−4superscript10410^{-4}, 10−3}10^{-3}\}, L2subscript𝐿2L\_{2}-regularization strengths {10−5,10−3}superscript105superscript103\{10^{-5},10^{-3}\},
and numbers of warmup epochs {5,10}510\{5,10\}.
We report results aggregated over 3 random seeds.

##### ERM results and performance drops.

Model performance dropped significantly going from the train-to-train in-distribution (ID) setting to
the official out-of-distribution (OOD) setting (Table [7](#S5.T7 "Table 7 ‣ ERM results and performance drops. ‣ E.3.2 Baseline results ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")),
with an average accuracy of 35.9% on the ID test set but only 29.9% on the OOD test set for ERM.

Table 7: Baseline results on RxRx1-wilds. In-distribution (ID) results correspond to the train-to-train setting. Parentheses show standard deviation across
3 replicates.

|  |  |  |  |
| --- | --- | --- | --- |
| Algorithm | Validation (OOD) accuracy | Test (ID) accuracy | Test (OOD) accuracy |
| ERM | 19.4 (0.2) | 35.9 (0.4) | 29.9 (0.4) |
| CORAL | 18.5 (0.4) | 34.0 (0.3) | 28.4 (0.3) |
| IRM | 5.6 (0.4) | 9.9 (1.4) | 8.2 (1.1) |
| Group DRO | 15.2 (0.1) | 28.1 (0.3) | 23.0 (0.3) |




Table 8: Mixed-to-test comparison for ERM models on RxRx1-wilds.
In the official OOD setting, we train on data from 33 experiments (1 site per experiment) and test on 14 different experiments (2 sites per experiment).
In the mixed-to-test setting, we replace 14 of the training experiments with 1 site from each of the test experiments, which keeps the training set size the same, but halves the test set size.
Parentheses show standard deviation across 3 replicates.

|  |  |  |
| --- | --- | --- |
| Setting | Algorithm | Test (OOD) accuracy |
| Official (train on ID examples) | ERM | 29.9 (0.4) |
| Mixed-to-test (train on ID + OOD examples) | ERM | 39.8 (0.2) |

We ran an additional mixed-to-test comparison, where we moved half of the OOD test set into the training set, while keeping the overall amount of training data the same.
Specifically, we moved one site per experiment from the OOD test set into the training set, and discarded an equivalent number of training sites, while leaving the validation set unchanged.
While the test set in the mixed-to-test setting is effectively half as large as in the standard split, we expect it to be distributed similarly, since the two test set versions comprise the same 14 experiments.

Table [8](#S5.T8 "Table 8 ‣ ERM results and performance drops. ‣ E.3.2 Baseline results ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that there is a large gap between the OOD test accuracies in the official setting (29.9%) and the test accuracies in the mixed-to-test setting (39.8%).
We note that the latter is higher than the train-to-train ID test accuracy of 35.9% reported in Table [7](#S5.T7 "Table 7 ‣ ERM results and performance drops. ‣ E.3.2 Baseline results ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
This difference mainly stems from the slight difference in cell type composition between the test sets in the train-to-train and mixed-to-test settings; in particular, the train-to-train test set has a slightly higher proportion of the minority cell type (U2OS), on which performance is worse, and a slightly lower proportion of the majority cell type (HUVEC), on which performance is better.
In this sense, the mixed-to-test result of 39.8% is a more accurate reflection of in-distribution performance on this dataset, and the results in Table [7](#S5.T7 "Table 7 ‣ ERM results and performance drops. ‣ E.3.2 Baseline results ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") therefore understate the magnitude of the distribution shift.

##### Additional baseline methods.

We also trained models with CORAL, IRM, and group DRO, treating each experiment
as a domain, and using the same model hyperparameters as ERM.
However, the models trained using these methods all performed poorly compared to the ERM model (Table [7](#S5.T7 "Table 7 ‣ ERM results and performance drops. ‣ E.3.2 Baseline results ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
One complication with these methods is that the experiments in the training set comprise different cell types, as mentioned above; this heterogeneity can pose a challenge to methods that treat each domain equivalently.

##### Discussion.

An important observation about batch effects in biological experiments: it is
often the case that batch effects are mediated via biological mechanisms. For
example, an increase in cellular media concentration may lead to cell growth and
proliferation, while the upregulation of proliferation genes will do the same.
Thus the “nuisance” factors associated with batch effects are often correlated
with the biological signal we are attempting to observe, and cannot be
disentangled from the biological factors that explain the data. Correction algorithms
should take account of such trade-offs and attempt to optimize for both
correction and signal preservation.

#### E.3.3 Broader context

As previously mentioned, high-throughput screening techniques are used broadly
across many areas of biology, and therefore batch effects are a common problem
in fields such as genomics, transcriptomics, proteomics, metabolomics, etc., so
a particular solution in one such area may prove to be applicable in many areas
of biology (Goh et al., [2017](#bib.bib152)).

There are other datasets that are used in studying batch effects. The one most
comparable to RxRx1-wilds is the BBBC021 dataset (Ljosa et al., [2012](#bib.bib245)), which
contains 13,200 3-channel fluorescent microscopy images of MCF7 cells acquired
across 10 experimental batches. A subset of 103 treatments from 38 drug
compounds belonging to 12 known mechanism of action (MoA) groups was first
studied in Ando et al. ([2017](#bib.bib14)), and has been the subject of subsequent
studies (Caicedo et al., [2018](#bib.bib71); Godinez et al., [2018](#bib.bib148); Tabak et al., [2020](#bib.bib362)).
Note that this dataset differs dramatically from RxRx1, in that there are fewer
images, treatments, batches, and cell types, and each batch contains only a
small subset of the total treatments.

#### E.3.4 Additional details

##### Data processing.

RxRx1-wilds contains two non-overlapping 256×\times256 fields of view per well.
Therefore, there could be as many as 125,664 images in the dataset (= 51 experiments ×\times 4 plates/experiment ×\times 308 wells/plate ×\times 2 images/well).
154 images were removed based on data quality, leaving a total dataset of 125,510 images.

##### Modifications to the original dataset.

The underlying raw dataset
consists of 2048 ×\times 2048 pixel, 6 channel, 16bpp images. To fit within the
constraints of the Wilds benchmark, images for RxRx1-wilds were first downsampled to
1024 ×\times 1024 and 8bpp, cropped to the center 256 ×\times 256 pixels, and only the
first three channels (nuclei, endoplasmic reticuli, actin) were retained. The
original RxRx1 dataset, available at rxrx.ai and described in Taylor et al. ([2019](#bib.bib367)), provides 512 ×\times 512 center
crops of the downsampled images with all 6 channels retained.

The original RxRx1 dataset was also used in a NeurIPS 2019 competition hosted on Kaggle. The validation (OOD) and test (OOD) splits in RxRx1-wilds correspond to the public and private test sets from the Kaggle competition. The original RxRx1 dataset did not have an additional test (ID) split, and thus the original training split had both sites 1 and 2, for a total of 81,442 images.
The Kaggle competition also aggregated predictions from both sites to form a single prediction per well, whereas in RxRx1-wilds, we treat each site separately.

As described in Section [E.3.1](#S5.SS3.SSS1 "E.3.1 Setup ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), each plate in both the training and test sets contains the
same 31 control conditions (one untreated well, and 30 positive control siRNAs).
The Kaggle competition provided the labels for these control conditions in the test set, expecting that competitors would use them for various domain alignment techniques such as CORAL. However, these labels were instead used by the top competitors to bootstrap pseudo-labeling techniques.
For RxRx1-wilds, for consistency with the other datasets and the typical domain generalization setting, we have opted not to release these control test labels for training.

The poor performance reported here on RxRx1-wilds may seem
surprising in light of the fact that the top finishers of the Kaggle
competition achieved near perfect accuracy on the test (OOD) set. This difference is due to a number of factors, including:

1. 1.

   Adjustments made to the original RxRx1 dataset for RxRx1-wilds, as detailed in this subsection.
2. 2.

   Differences in the network architectures used. To make training on RxRx1-wilds more accessible, we used a less compute-intensive architecture than typical in the competition.
3. 3.

   Differences in training techniques used like pseudo-labeling (using the test control labels, as described above) and batch-level dataset augmentations or ensembling.
4. 4.

   Differences in the way accuracy is measured. In the Kaggle competition, accuracy was measured for each well, meaning site-level predictions were aggregated to well-level predictions, and only for treatment classes, whereas in RxRx1-wilds, for convenience, accuracy is measured at each site and for both treatment and control classes.
5. 5.

   The use of post-prediction methods like linear sum assignment that exploited the particular structure of the experiments in the RxRx1 dataset, as described under Potential Leverage in Section [E.3.1](#S5.SS3.SSS1 "E.3.1 Setup ‣ E.3 RxRx1-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

### E.4 OGB-MolPCBA

Accurate prediction of the biochemical properties of small molecules can significantly accelerate drug discovery by reducing the need for expensive lab experiments (Shoichet, [2004](#bib.bib346); Hughes et al., [2011](#bib.bib184)).
However, the experimental data available for training such models is limited compared to the extremely diverse and combinatorially large universe of candidate molecules that we would want to make predictions on (Bohacek et al., [1996](#bib.bib52); Sterling and Irwin, [2015](#bib.bib352); Lyu et al., [2019](#bib.bib251); McCloskey et al., [2020](#bib.bib258)).
This means that models need to generalize to out-of-distribution molecules that are structurally different from those seen in the training set.

We study this issue through the OGB-MolPCBA dataset, which is directly adopted from the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)) and originally curated by MoleculeNet (Wu et al., [2018](#bib.bib412)).

#### E.4.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the domains are molecular scaffolds, and our goal is to learn models that generalize to structurally distinct molecules with scaffolds that are not in the training set (Figure [6](#S4.F6 "Figure 6 ‣ 4.1.4 OGB-MolPCBA: Molecular property prediction across different scaffolds ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
This is a multi-task classification problem: for each molecule, we predict the presence or absence of 128 kinds of biological activities, such as binding to a particular enzyme.
In addition, we cluster the molecules into different scaffold groups according to their two-dimensional structure, and annotate each molecule with the scaffold group that it belongs to.
Concretely, the input x𝑥x is a molecular graph, the label y𝑦y is a 128-dimensional binary vector where each component corresponds to a biochemical assay result, and the domain d𝑑d specifies the scaffold. Not all biological activities are measured for each molecule, so y𝑦y can have missing values.

##### Data.

OGB-MolPCBA contains more than 400K small molecules with 128 kinds of prediction labels.
Each small molecule is represented as a graph, where the nodes are atoms and the edges are chemical bonds.
The molecules are pre-processed using RDKit (Landrum et al., [2006](#bib.bib220)).
Input node features are 9-dimensional, including atomic number, chirality, whether the atom is in the ring. Input edge features are 3-dimensional, including bond type and bond stereochemistry.

We split the dataset by scaffold structure. This *scaffold split* (Wu et al., [2018](#bib.bib412)) is also used in the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)).
By attempting to separate structurally different molecules into different subsets, it provides a realistic estimate of model performance in prospective experimental settings.
We assign the largest scaffolds to the training set to make it easier for algorithms to leverage scaffold information, and the smallest scaffolds to the test set to ensure that it is maximally diverse in scaffold structure:

1. 1.

   Training: The largest 44,930 scaffolds, with an average of 7.8 molecules per scaffold.
2. 2.

   Validation (OOD): The next largest 31,361 scaffolds, with an average of 1.4 molecules per scaffold.
3. 3.

   Test (OOD): The smallest 43,793 scaffolds, which are all singletons.

In Figure [22](#S5.F22 "Figure 22 ‣ Data. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") (A), we plot the statistics of the scaffolds in terms of the number of molecules belonging to each scaffold. We see that the scaffold sizes are highly skewed, with the test set containing (by design) the scaffolds with the least molecules.
However, the differences in scaffold sizes do not significantly change the statistics of the molecules in each split.
In Figures [22](#S5.F22 "Figure 22 ‣ Data. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") (B) and (C), we see that the label statistics remain very similar across train/validation/test splits, suggesting that the main distribution shift comes from the difference in the input molecular graph structure.

![Refer to caption](/html/2012.07421/assets/x18.png)


Figure 22: Analyses of scaffold groups in the OGB-MolPCBA dataset. (A) shows the distribution of the scaffold sizes, (B) and (C) show how the ratios of positive molecules and labeled molecules for the 128 tasks vary across the train/validation /test splits.

##### Evaluation.

We evaluate models by their average Average Precision (AP) across tasks (i.e., we compute the average precision for each task separately, and then average those scores), following Hu et al. ([2020b](#bib.bib182)). This accounts for the extremely skewed class balance in OGB-MolPCBA (only 1.4% of data is positive). Not all labels are available for each molecule; when calculating the AP for each task, we only consider the labeled molecules for the task.

##### Potential leverage.

We provide the scaffold grouping of molecules for training algorithms to leverage. Finding generalizable representations of molecules across different scaffold groups is useful for models to make accurate extrapolation on unseen scaffold groups. In fact, very recent work (Jin et al., [2020](#bib.bib189)) has leveraged scaffold information of molecules to improve the extrapolation performance of molecular property predictors.

One notable characteristic of the scaffold group is that the size of each group is rather small; on the training split, each scaffold contains only 7.8 molecules on average. This also results in many scaffold groups: 44,930 groups in the training split. In Figure [22](#S5.F22 "Figure 22 ‣ Data. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we show that these scaffold groups are well-behaved in the sense that the train/validation/test splits contain contain similar ratios of positive labels as well as missing labels.

Table 9: Baseline results on OGB-MolPCBA. Parentheses show standard deviation across 3 replicates.

|  |  |  |
| --- | --- | --- |
| Algorithm | Validation AP (%) | Test AP (%) |
| ERM | 27.8 (0.1) | 27.2 (0.3) |
| CORAL | 18.4 (0.2) | 17.9 (0.5) |
| IRM | 15.8 (0.2) | 15.6 (0.3) |
| Group DRO | 23.1 (0.6) | 22.4 (0.6) |




Table 10: 
Random split comparison for ERM models on OGB-MolPCBA.
In the official OOD setting, we train on molecules from some scaffolds and evaluate on molecules from different scaffolds, whereas in the random split setting, we randomly divide molecules into training and test sets without using scaffold information.
Parentheses show standard deviation across 3 replicates.

|  |  |  |
| --- | --- | --- |
| Setting | Algorithm | Test AP (%) |
| Official (split by scaffolds) | ERM | 27.2 (0.3) |
| Random split (split i.i.d.) | ERM | 34.4 (0.9) |

#### E.4.2 Baseline results

##### Model.

For all experiments, we use Graph Isomorphism Networks (GIN) (Xu et al., [2018](#bib.bib418)) combined with virtual nodes (Gilmer et al., [2017](#bib.bib147)), as this is currently the model with the highest performance in the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)).
We follow the same hyperparameters as in the Open Graph Benchmark: 5 GNN layers with a dimensionality of 300;
the Adam optimizer (Kingma and Ba, [2015](#bib.bib207)) with a learning rate of 0.001;
and training for 100 epochs with early stopping.
For each of the baseline algorithms (ERM, CORAL, IRM, and Group DRO),
we separately tune the dropout rate from {0,0.5}00.5\{0,0.5\};
in addition, for CORAL and IRM, we tune the penalty weight as in Appendix [D](#S4a "D Additional experimental details ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

##### ERM results and performance drops.

We first compare the generalization performance of ERM on the official scaffold split against the conventional random split, in which the entire molecules are randomly split into train/validation/test sets with the same split ratio as the scaffold split (i.e., 80/10/10).
Results are in Table [10](#S5.T10 "Table 10 ‣ Potential leverage. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"). The test performance of ERM drops by 7.2 points AP when the scaffold split is used, suggesting that the scaffold split is indeed harder than the random split.

To maintain consistency with the Open Graph Benchmark, and because the number of examples (molecules) per domain (scaffold) is relatively small compared to other datasets, we opted not to split off a portion of the training set into Validation (ID) and Test (ID) sets.
We therefore do not run a train-to-train comparison for OGB-MolPCBA.
Moreover, as the official scaffold split assigns the largest scaffolds to the training set and the smallest scaffolds to the test set, the test scaffolds all only have one molecule per scaffold, which precludes running test-to-test and mixed-to-test comparisons.

A potential issue with the random split ID comparison is that it does not measure performance on the same test distribution as the official split, and therefore might be confounded by differences in intrinsic difficulty. However, we believe that the random split setting provides a reasonable measure of ID performance for OGB-MolPCBA, as Figure [22](#S5.F22 "Figure 22 ‣ Data. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that the distribution of scaffolds assigned to the training versus test sets are similar. As the random split contains many singleton scaffolds in its test set that do not have corresponding molecules in the training set, we believe that it is likely to be an underestimate of the ID-OOD gap in OGB-MolPCBA.

##### Additional baseline methods.

Table [9](#S5.T9 "Table 9 ‣ Potential leverage. ‣ E.4.1 Setup ‣ E.4 OGB-MolPCBA ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") also shows that ERM performs better than CORAL, IRM, and Group DRO, all of which use scaffolds as the domains. For CORAL and IRM, we find that smaller penalties give better generalization performance, as larger penalty terms make the training insufficient.
We use the 0.10.10.1 penalty for CORAL and λ=1𝜆1\lambda=1 for IRM.

The primary issue with these existing methods is that they make the model significantly underfit the training data even when dropout is turned off.
For instance, the training AP of CORAL and IRM is 20.0% and 15.9%, respectively, which are both lower than the 36.1% that ERM obtains even with 0.5 dropout.
Also, these methods are primarily designed for the case when each group contains a decent number of examples, which is not the case for the OGB-MolPCBA dataset.

#### E.4.3 Broader context

Because of the very nature of discovering *new* molecules, out-of-distribution prediction is prevalent in nearly all applications of machine learning to chemistry domains. Beyond drug discovery, a variety of tasks and their associated datasets have been proposed for molecules of different sizes.

For small organic molecules, the scaffold split has been widely adopted to stress-test models’ capability for out-of-distribution generalization. While OGB-MolPCBA primarily focuses on predicting biophysical activity (e.g., protein binding), other datasets in MoleculeNet (Wu et al., [2018](#bib.bib412)) include prediction of quantum mechanical properties (e.g., HOMO/LUMO), physical chemistry properties (e.g., water solubility), and physiological properties (e.g., toxicity prediction (Attene-Ramos et al., [2013](#bib.bib18))).

Besides small molecules, it is also of interest to apply machine learning over larger molecules such as catalysts and proteins.
In the domain of catalysis, using machine learning to approximate expensive quantum chemistry simulation has gotten attention. The OC20 dataset has been recently introduced, containing 200+ million samples from quantum chemistry simulations relevant to the discovery of new catalysts for renewable energy storage and other energy applications (Becke, [2014](#bib.bib34); Chanussot et al., [2020](#bib.bib77); Zitnick et al., [2020](#bib.bib435)). The OC20 dataset explicitly provides test sets with qualitatively different materials.
In the domain of proteins, the recent trend is to use machine learning to predict 3D structure of proteins given their amino acid sequence information. This is known as the protein folding problem, and has sometimes been referred to as the Holy Grail of structural biology (Dill and MacCallum, [2012](#bib.bib111)).
CASP is a bi-annual competition to benchmark the progress of protein folding (Moult et al., [1995](#bib.bib269)), and it evaluates predictions made on proteins whose 3D structures are identified very recently, presenting a natural temporal distribution shift.
Recently, the AlphaFold2 deep learning model obtained breakthrough performance on the CASP challenge (Jumper et al., [2020](#bib.bib193)), demonstrating exciting avenues of machine learning for structural biology.

#### E.4.4 Additional details

Data processing.
The OGB-MolPCBA dataset contains 437,929 molecules annotated with 128 kinds of labels, each representing a bioassay curated in the PubChem database (Kim et al., [2016b](#bib.bib206)). More details are provided in the MoleculeNet (Wu et al., [2018](#bib.bib412)) and the Open Graph Benchmark (Hu et al., [2020b](#bib.bib182)), from which the dataset is adopted.

### E.5 GlobalWheat-wilds

Models for automated, high-throughput plant phenotyping—measuring the physical characteristics of plants and crops, such as wheat head density and counts—are important tools for crop breeding (Thorp et al., [2018](#bib.bib372); Reynolds et al., [2020](#bib.bib313)) and agricultural field management (Shi et al., [2016](#bib.bib342)).
These models are typically trained on data collected in a limited number of regions, even for crops grown worldwide such as wheat (Madec et al., [2019](#bib.bib254); Xiong et al., [2019](#bib.bib417); Ubbens et al., [2020](#bib.bib381); Ayalew et al., [2020](#bib.bib22)).
However, there can be substantial variation between regions, due to differences in crop varieties, growing conditions, and data collection protocols.
Prior work on wheat head detection has shown that this variation can significantly degrade model performance on regions unseen during training (David et al., [2020](#bib.bib104)).

We study this shift in an expanded version of the Global Wheat Head Dataset (David et al., [2020](#bib.bib104), [2021](#bib.bib105)), a large set of wheat images collected from 12 countries around the world.

#### E.5.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the goal is to learn models that generalize to images taken from new countries and acquisition sessions (Figure [7](#S4.F7 "Figure 7 ‣ 4.1.5 GlobalWheat-wilds: Wheat head detection across regions of the world ‣ 4.1 Domain generalization datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The task is wheat head detection, which is a single-class object detection task.
Concretely, the input x𝑥x is an overhead outdoor image of wheat plants, and the label y𝑦y is a set of bounding box coordinates that enclose the wheat heads (the spike at the top of the wheat plant containing grain), excluding the hair-like awns that may extend from the head.
The domain d𝑑d specifies an *acquisition session*, which corresponds to a specific location, time, and sensor for which a set of images were collected.
Our goal is to generalize to new acquisition sessions that are unseen during training.
In particular, the dataset split captures a shift in location, with training and test sets comprising images from disjoint countries as discussed below.

##### Data.

The dataset comprises 6,515 images containing 275,187 wheat heads.
These images were collected over 47 acquisition sessions in 16 research institutes across 12 countries. We describe the metadata and statistics of each acquisition session in Table [11](#S5.T11 "Table 11 ‣ Data. ‣ E.5.1 Setup ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

Many factors contribute to the variation in wheat appearance across acquisition sessions.
In particular, across locations, there is substantial variation due to differences in wheat genotypes, growing conditions (e.g., planting density), illumination protocols, and sensors.
We study the effect of this location shift by splitting the dataset by country
and assigning acquisition sessions from disjoint continents to the training and test splits:

1. 1.

   Training: Images from 18 acquisition sessions in Europe (France ×\times13, Norway ×\times2, Switzerland, United Kingdom, Belgium), containing 131,864 wheat heads across 2,943 images.
2. 2.

   Validation (OOD):
   Images from 7 acquisition sessions in Asia (Japan ×\times 4, China ×\times 3) and 1 acquisition session in Africa (Sudan), containing 44,873 wheat heads across 1,424 images.
3. 3.

   Test (OOD): Images from 11 acquisition sessions in Australia and 10 acquisition sessions in North America (USA ×\times 6, Mexico ×\times 3, Canada), containing 66,905 wheat heads across 1,434 images.
4. 4.

   Validation (ID): Images from the same 18 training acquisition sessions in Europe, containing 15,733 wheat heads across 357 images.
5. 5.

   Test (ID): Images from the same 18 training acquisition sessions in Europe, containing 16,093 wheat heads across 357 images.

Table 11: 
Acquisition sessions in GlobalWheat-wilds. Growth stages are abbreviated as F: Filling, R: Ripening, PF: Post-flowering. Locations are abbreviated as VLB: Villiers le Bâcle, VSC: Villers-Saint-Christophe. UTokyo\_1 and UTokyo\_2 are from the same location with different cart sensors and UTokyo\_3 consists of images from a variety of farms in Hokkaido between 2016 and 2019.
The # images and # heads in the “Train” domains include the images and heads used in the Val (ID) and Test (ID) splits, which are taken from the same set of training domains. The “Val” and “Test” domains refer to the Val (OOD) and Test (OOD) splits, respectively.

|  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Split | Name | Owner | Country | Site | Date | Sensor | Stage | # Images | # Heads |
| Training | Ethz\_1 | ETHZ | Switzerland | Eschikon | 06/06/2018 | Spidercam | F | 747 | 49603 |
| Training | Rres\_1 | Rothamsted | UK | Rothamsted | 13/07/2015 | Gantry | F-R | 432 | 19210 |
| Training | ULiège\_1 | Uliège | Belgium | Gembloux | 28/07/2020 | Cart | R | 30 | 1847 |
| Training | NMBU\_1 | NMBU | Norway | NMBU | 24/07/2020 | Cart | F | 82 | 7345 |
| Training | NMBU\_2 | NMBU | Norway | NMBU | 07/08/2020 | Cart | R | 98 | 5211 |
| Training | Arvalis\_1 | Arvalis | France | Gréoux | 02/06/2018 | Handheld | PF | 66 | 2935 |
| Training | Arvalis\_2 | Arvalis | France | Gréoux | 16/06/2018 | Handheld | F | 401 | 21003 |
| Training | Arvalis\_3 | Arvalis | France | Gréoux | 07/2018 | Handheld | F-R | 588 | 21893 |
| Training | Arvalis\_4 | Arvalis | France | Gréoux | 27/05/2019 | Handheld | F | 204 | 4270 |
| Training | Arvalis\_5 | Arvalis | France | VLB\* | 06/06/2019 | Handheld | F | 448 | 8180 |
| Training | Arvalis\_6 | Arvalis | France | VSC\* | 26/06/2019 | Handheld | F-R | 160 | 8698 |
| Training | Arvalis\_7 | Arvalis | France | VLB\* | 06/2019 | Handheld | F-R | 24 | 1247 |
| Training | Arvalis\_8 | Arvalis | France | VLB\* | 06/2019 | Handheld | F-R | 20 | 1062 |
| Training | Arvalis\_9 | Arvalis | France | VLB\* | 06/2020 | Handheld | R | 32 | 1894 |
| Training | Arvalis\_10 | Arvalis | France | Mons | 10/06/2020 | Handheld | F | 60 | 1563 |
| Training | Arvalis\_11 | Arvalis | France | VLB\* | 18/06/2020 | Handheld | F | 60 | 2818 |
| Training | Arvalis\_12 | Arvalis | France | Gréoux | 15/06/2020 | Handheld | F | 29 | 1277 |
| Training | Inrae\_1 | INRAe | France | Toulouse | 28/05/2019 | Handheld | F-R | 176 | 3634 |
| Val | Utokyo\_1 | UTokyo | Japan | NARO-Tsukuba | 22/05/2018 | Cart | R | 538 | 14185 |
| Val | Utokyo\_2 | UTokyo | Japan | NARO-Tsukuba | 22/05/2018 | Cart | R | 456 | 13010 |
| Val | Utokyo\_3 | UTokyo | Japan | NARO-Hokkaido | 2016-19 | Handheld | multiple | 120 | 3085 |
| Val | Ukyoto\_1 | UKyoto | Japan | Kyoto | 30/04/2020 | Handheld | PF | 60 | 2670 |
| Val | NAU\_1 | NAU | China | Baima | n/a | Handheld | PF | 20 | 1240 |
| Val | NAU\_2 | NAU | China | Baima | 02/05/2020 | Cart | PF | 100 | 4918 |
| Val | NAU\_3 | NAU | China | Baima | 09/05/2020 | Cart | F | 100 | 4596 |
| Val | ARC\_1 | ARC | Sudan | Wad Medani | 03/2021 | Handheld | F | 30 | 1169 |
| Test | Usask\_1 | USaskatchewan | Canada | Saskatoon | 06/06/2018 | Tractor | F-R | 200 | 5985 |
| Test | KSU\_1 | KansasStateU | US | Manhattan, KS | 19/05/2016 | Tractor | PF | 100 | 6435 |
| Test | KSU\_2 | KansasStateU | US | Manhattan, KS | 12/05/2017 | Tractor | PF | 100 | 5302 |
| Test | KSU\_3 | KansasStateU | US | Manhattan, KS | 25/05/2017 | Tractor | F | 95 | 5217 |
| Test | KSU\_4 | KansasStateU | US | Manhattan, KS | 25/05/2017 | Tractor | R | 60 | 3285 |
| Test | Terraref\_1 | TERRA-REF | US | Maricopa | 02/04/2020 | Gantry | R | 144 | 3360 |
| Test | Terraref\_2 | TERRA-REF | US | Maricopa | 20/03/2020 | Gantry | F | 106 | 1274 |
| Test | CIMMYT\_1 | CIMMYT | Mexico | Ciudad Obregon | 24/03/2020 | Cart | PF | 69 | 2843 |
| Test | CIMMYT\_2 | CIMMYT | Mexico | Ciudad Obregon | 19/03/2020 | Cart | PF | 77 | 2771 |
| Test | CIMMYT\_3 | CIMMYT | Mexico | Ciudad Obregon | 23/03/2020 | Cart | PF | 60 | 1561 |
| Test | UQ\_1 | UQueensland | Australia | Gatton | 12/08/2015 | Tractor | PF | 22 | 640 |
| Test | UQ\_2 | UQueensland | Australia | Gatton | 08/09/2015 | Tractor | PF | 16 | 39 |
| Test | UQ\_3 | UQueensland | Australia | Gatton | 15/09/2015 | Tractor | F | 14 | 297 |
| Test | UQ\_4 | UQueensland | Australia | Gatton | 01/10/2015 | Tractor | F | 30 | 1039 |
| Test | UQ\_5 | UQueensland | Australia | Gatton | 09/10/2015 | Tractor | F-R | 30 | 3680 |
| Test | UQ\_6 | UQueensland | Australia | Gatton | 14/10/2015 | Tractor | F-R | 30 | 1147 |
| Test | UQ\_7 | UQueensland | Australia | Gatton | 06/10/2020 | Handheld | R | 17 | 1335 |
| Test | UQ\_8 | UQueensland | Australia | McAllister | 09/10/2020 | Handheld | R | 41 | 4835 |
| Test | UQ\_9 | UQueensland | Australia | Brookstead | 16/10/2020 | Handheld | F-R | 33 | 2886 |
| Test | UQ\_10 | UQueensland | Australia | Gatton | 22/09/2020 | Handheld | F-R | 106 | 8629 |
| Test | UQ\_11 | UQueensland | Australia | Gatton | 31/08/2020 | Handheld | PF | 84 | 4345 |

##### Evaluation.

We evaluate models by first computing the average accuracy of bounding box detection within each image;
then computing the average accuracy for each acquisition session by averaging its per-image accuracies;
and finally averaging the accuracies of each acquisition session.
The accuracy of a bounding box detection is measured at a fixed Intersection over Union (IoU) threshold of 0.5.
The accuracy of an image is computed as T​PT​P+F​N+F​P𝑇𝑃𝑇𝑃𝐹𝑁𝐹𝑃\frac{TP}{TP+FN+FP}, where
T​P𝑇𝑃TP is the number of true positives, which are ground-truth bounding boxes that can be matched with some predicted bounding box at IoU above the threshold;
F​N𝐹𝑁FN is the number of false negatives, which are ground-truth bounding boxes that cannot be matched as above; and
F​P𝐹𝑃FP is the number of false positives, which are predicted bounding boxes that cannot be matched with any ground-truth bounding box.
We use accuracy rather than average precision, which is a common metric for object detection, because it was used in previous Global Wheat Challenges with the dataset (David et al., [2020](#bib.bib104), [2021](#bib.bib105)). We use a permissive IoU threshold of 0.5 because there is some uncertainty regarding the precise outline of wheat head instances due to the stem and awns extending from the head.
We measure the average accuracy across acquisition sessions because the number of images varies significantly across acquisition sessions, from 17 to 200 images in the test set, and we use average accuracy instead of worst-case accuracy because the wheat images are more difficult for some acquisition sessions.

##### Potential leverage.

The appearance of wheat heads in the images taken from different acquisition sessions can vary significantly, due to differences in the sensors used; illumination conditions, due to differences in illumination protocols, or the time of day and time of year that the images were taken; wheat genotypes; growth stages; growing conditions; and planting strategies.
For example, different locations might feature a mix of different varieties of wheat (with different genotypes) with different appearances. Likewise, wheat planting strategies and growing conditions vary between regions and can contribute to differences between sessions, e.g., higher planting density may result in more closely packed plants and more occlusion between wheat head instances.

To provide leverage for models to learn to generalize across these conditions, we include images from 5 countries and 18 acquisition sessions in the training set.
These training sessions cover all growth stages and include significant variation among all of the other factors.
While the test domains include unseen conditions (e.g., sensors and genotypes not seen in the training set), our hope is that the variation in the training set will be sufficient to learn models that are robust to changes in these conditions.

#### E.5.2 Baseline results

##### Model.

For all experiments, we use the Faster-RCNN detection model (Ren et al., [2015](#bib.bib312)), which has been successfully applied to the wheat head localization problem (Madec et al., [2019](#bib.bib254); David et al., [2020](#bib.bib104)).
To train, we fine-tune a model pre-trained with ImageNet, using a batch size of 4, a learning rate of 10−5superscript10510^{-5}, and weight decay of 10−3superscript10310^{-3} for 10 epochs with early stopping.
The hyperparameters were chosen from a grid search over learning rates {10−6,10−5,10−4}superscript106superscript105superscript104\{10^{-6},10^{-5},10^{-4}\} and weight decays {0,10−4,10−3}0superscript104superscript103\{0,10^{-4},10^{-3}\}.
We report results aggregated over 3 random seeds.

Table 12: Baseline results on GlobalWheat-wilds. In-distribution (ID) results correspond to the train-to-train setting. Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Algorithm | Validation (ID) acc | Validation (OOD) acc | Test (ID) acc | Test (OOD) acc |
| ERM | 77.4 (1.1) | 68.6 (0.4) | 77.1 (0.5) | 51.2 (1.8) |
| Group DRO | 76.1 (1.0) | 66.2 (0.4) | 76.2 (0.8) | 47.9 (2.0) |




Table 13: Mixed-to-test comparison for ERM models on GlobalWheat-wilds.
In the official OOD setting, we train on data from Europe, whereas in the mixed-to-test ID setting, we train on a mix of data from Europe, Africa, and North America.
In both settings, we test on data from Africa and North America.
For this comparison, we report performance on 50% of the official test set (randomly selecting 50% of each test domain), with the rest of the test set mixed in to the training set in the mixed-to-test setting.
Parentheses show standard deviation across 3 replicates.

|  |  |  |
| --- | --- | --- |
| Setting | Algorithm | Test accuracy (%) |
| Official (train on ID examples) | ERM | 49.6 (1.9) |
| Mixed-to-test (train on ID + OOD examples) | ERM | 63.3 (1.7) |




Table 14: Mixed-to-test comparison for ERM models on GlobalWheat-wilds, broken down by each test domain. This is a more detailed version of Table [13](#S5.T13 "Table 13 ‣ Model. ‣ E.5.2 Baseline results ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"). Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Session | Country | # Images | ID (mixed-to-test) acc | OOD acc | ID-OOD gap |
| CIMMYT\_1 | Mexico | 35 | 63.1 (1.4) | 48.0 (2.6) | 15.1 |
| CIMMYT\_2 | Mexico | 39 | 76.1 (0.9) | 58.2 (3.6) | 17.9 |
| CIMMYT\_3 | Mexico | 30 | 65.6 (3.1) | 63.3 (2.1) | 2.3 |
| KSU\_1 | US | 50 | 73.5 (1.1) | 53.2 (2.3) | 20.3 |
| KSU\_2 | US | 50 | 73.6 (0.5) | 52.7 (2.7) | 20.9 |
| KSU\_3 | US | 48 | 73.3 (1.2) | 48.9 (3.0) | 24.4 |
| KSU\_4 | US | 30 | 68.3 (0.6) | 48.7 (3.5) | 19.6 |
| Terraref\_1 | US | 72 | 48.9 (0.5) | 17.9 (4.3) | 31.0 |
| Terraref\_2 | US | 53 | 34.7 (1.3) | 16.0 (3.4) | 18.7 |
| Usask\_1 | Canada | 100 | 78.3 (0.8) | 77.1 (1.4) | 1.2 |
| UQ\_1 | Australia | 11 | 41.8 (1.4) | 29.0 (1.0) | 12.8 |
| UQ\_2 | Australia | 8 | 81.6 (12.5) | 76.5 (14.4) | 5.1 |
| UQ\_3 | Australia | 7 | 56.4 (13.8) | 54.3 (10.0) | 2.1 |
| UQ\_4 | Australia | 15 | 68.8 (0.5) | 60.6 (1.3) | 8.2 |
| UQ\_5 | Australia | 15 | 54.4 (2.1) | 38.6 (2.1) | 15.8 |
| UQ\_6 | Australia | 15 | 75.8 (1.1) | 71.9 (0.7) | 3.9 |
| UQ\_7 | Australia | 9 | 68.9 (0.6) | 62.8 (2.5) | 6.1 |
| UQ\_8 | Australia | 21 | 58.6 (0.6) | 46.5 (2.1) | 11.1 |
| UQ\_9 | Australia | 17 | 54.7 (1.5) | 43.6 (2.1) | 11.1 |
| UQ\_10 | Australia | 53 | 61.7 (0.8) | 39.6 (2.5) | 22.1 |
| UQ\_11 | Australia | 42 | 50.4 (1.5) | 33.5 (2.7) | 16.9 |
| Total |  | 720 | 63.3 (1.7) | 49.6 (1.9) | 13.7 |

##### ERM results and performance drops.

We ran both train-to-train and mixed-to-test comparisons.
For the train-to-train comparison, which uses the data splits described in the previous subsection, the Test (ID) accuracy is substantially higher than the Test (OOD) accuracy (77.1 (0.5) vs. 51.2 (1.8); Table [12](#S5.T12 "Table 12 ‣ Model. ‣ E.5.2 Baseline results ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). However, the Test (ID) and Test (OOD) sets come from entirely different regions, so this performance gap could also reflect a difference in the difficulty of the wheat head detection task in different regions (e.g., wheat heads that are more densely packed are harder to tell apart).

The mixed-to-test comparison controls for the test distribution by randomly splitting each test domain (acquisition session) into two halves, and then assigning one half to the training set. In other words, we randomly take out half of the test set and use it to replace existing examples in the training set, so that the total training set size is the same, and we retain the other half of the test set for evaluation.
We also evaluated the ERM model trained on the official split on this subsampled test set.
On this subsampled test set, the mixed-to-test ID accuracy is significantly higher than the OOD accuracy of the ERM model trained on the official split (63.3 (1.7) vs. 49.6 (1.9); Table [13](#S5.T13 "Table 13 ‣ Model. ‣ E.5.2 Baseline results ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

We also compared the per-domain accuracies of the models trained in the mixed-to-test and official settings (Table [14](#S5.T14 "Table 14 ‣ Model. ‣ E.5.2 Baseline results ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")) on the subsampled test set.
The accuracy drop is not evenly distributed across each domain, though some of the domains have a relatively small number of images, so there is some variance across random replicates.
The location/site of the acquisition session—which is correlated with factors like wheat genotype and the sensor used—has a large effect on performance (e.g., the KSU and Terraref sessions displayed a larger drop than the other sessions), but beyond that, it is not clear what factors are most strongly driving the accuracy drop.
The Terraref sessions were particularly difficult even in the mixed-to-test setting, because of the strong contrast in its photos and the presence of hidden wheat heads under leaves.
On the other hand, the KSU sessions had comparatively high accuracies in the mixed-to-test setting, but still displayed a large accuracy drop in the official OOD setting.
As the KSU sessions differed primarily in their development stages and had largely similar ID and OOD accuracies, development stage does not seem to be a main driver of the accuracy drop.
Finally, we note that the especially high variance across replicates for UQ\_2 and UQ\_3 is due to the proportion of empty images in those domains (88% for UQ\_2 and 57% for UQ\_3). Empty images are scored as either having 0% or 100% accuracy and therefore can have a large impact on the overall domain accuracy.

##### Additional baseline methods.

We also trained models with group DRO, treating each acquisition session as a domain, and using the same model hyperparameters as ERM.
However, the group DRO models perform poorly compared to the ERM model as reported in Table [12](#S5.T12 "Table 12 ‣ Model. ‣ E.5.2 Baseline results ‣ E.5 GlobalWheat-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We leave the investigation of CORAL and IRM for future work because it is not straightforward to apply these algorithms to detection tasks.

##### Discussion.

Our baseline models were trained without any data augmentation, in contrast to baselines reported in the original dataset (David et al., [2020](#bib.bib104)).
Data augmentation could reduce the performance gap and warrants further investigation in future work, although David et al. ([2020](#bib.bib104)) still observed performance gaps on models trained with data augmentation in the original version of the dataset.
Moreover, while we evaluated models by their average performance across acquisition sessions, we noticed a large variability in performance across domains.
It is possible that some domains are more challenging or suffer from larger performance drops than others, and characterizing and mitigating these variations is interesting future work.

#### E.5.3 Broader context

Wheat head localization, while being an important operational trait for wheat breeders and farmers, is not the only deep learning application in plant phenotyping that suffers from lack of generalization. Other architectural traits such as plant segmentation (Sadeghi-Tehran et al., [2017](#bib.bib324); Kuznichov et al., [2019](#bib.bib217)), plant and plant organ detection (Fan et al., [2018](#bib.bib127); Madec et al., [2019](#bib.bib254)), leaves and organ disease classification (Fuentes et al., [2017](#bib.bib133); Shakoor et al., [2017](#bib.bib335); Toda and Okura, [2019](#bib.bib376)), and biomass and yield prediction (Aich et al., [2018](#bib.bib6); Dreccer et al., [2019](#bib.bib116)) would also benefit from plant phenotyping models that generalize to new deployments. In many of these applications, field images exhibit variations in illumination and sensors, and there has been work on mitigating biases across sensors (Ayalew et al., [2020](#bib.bib22); Gogoll et al., [2020](#bib.bib151)). Finally, developing models that generalize across plant species would benefit the breeding and growing of specialized crops that are presently under-represented in plant phenotyping research worldwide (Ward and Moghadam, [2020](#bib.bib396)).
We hope that GlobalWheat-wilds can foster the development of general solutions to plant phenotyping problems, increase collaboration between plant scientists and computer vision scientists, and encourage the development of new multi-domain plant datasets to ensure that plant phenotyping results are generalizable to all crop growing regions of the world.

#### E.5.4 Additional details

##### Modifications to the original dataset.

The data is taken directly from the 2021 Global Wheat Challenge (David et al., [2021](#bib.bib105)), which is an expanded version of the 2020 Global Wheat Challenge dataset (David et al., [2020](#bib.bib104)).
Compared to the challenge, the dataset splits are different:
we split off part of the training set to form the Validation (ID) and Test (ID) sets, and we rearranged the Validation (OOD) and Test (OOD) sets so that they split along disjoint continents.
Finally, we note that the 2021 challenge differs from the 2020 challenge in that images from North America were in the training set in the 2020 challenge, but were used for evaluation in the 2021 challenge, and are consequently assigned to the test set in GlobalWheat-wilds.

### E.6 CivilComments-wilds

Automatic review of user-generated text is an important tool for moderating the sheer volume of text written on the Internet.
We focus here on the task of detecting toxic comments.
Prior work has shown that toxicity classifiers can pick up on biases in the training data and spuriously associate toxicity with the mention of certain demographics (Park et al., [2018](#bib.bib286); Dixon et al., [2018](#bib.bib112)).
These types of spurious correlations can significantly degrade model performance on particular subpopulations (Sagawa et al., [2020a](#bib.bib327)).

We study this issue through a modified variant of the CivilComments dataset (Borkan et al., [2019b](#bib.bib55)).

#### E.6.1 Setup

##### Problem setting.

We cast CivilComments-wilds as a subpopulation shift problem, where the subpopulations correspond to different demographic identities, and our goal is to do well on all subpopulations (and not just on average across these subpopulations).
Specifically, we focus on mitigating biases with respect to comments that mention particular demographic identities, and not comments written by members of those demographic identities; we discuss this distinction in the broader context section below.

The task is a binary classification task of determining if a comment is toxic.
Concretely, the input x𝑥x is a comment on an online article (comprising one or more sentences of text) and the label y𝑦y is whether it is rated toxic or not.
In CivilComments-wilds, unlike in most of the other datasets we consider, the domain annotation d𝑑d is a multi-dimensional binary vector, with the 8 dimensions corresponding to whether the comment mentions each of the 8 demographic identities male, female, LGBTQ, Christian, Muslim, other religions, Black, and White.

##### Data.

CivilComments-wilds comprises 450,000 comments, each annotated for toxicity and demographic mentions by multiple crowdworkers. We model toxicity classification as a binary task. Toxicity labels were obtained in the original dataset via crowdsourcing and majority vote, with each comment being reviewed by at least 10 crowdworkers. Annotations of demographic mentions were similarly obtained through crowdsourcing and majority vote.

Each comment was originally made on some online article. We randomly partitioned these articles into disjoint training, validation, and test splits, and then formed the corresponding datasets by taking all comments on the articles in those splits. This gives the following splits:

1. 1.

   Training: 269,038 comments.
2. 2.

   Validation: 45,180 comments.
3. 3.

   Test: 133,782 comments.

##### Evaluation.

We evaluate a model by its worst-group accuracy, i.e., its lowest accuracy over groups of the test data that we define below.

As mentioned above, toxicity classifiers can spuriously latch onto mentions of particular demographic identities, resulting in a biased tendency to flag comments that innocuously mention certain demographic groups as toxic (Park et al., [2018](#bib.bib286); Dixon et al., [2018](#bib.bib112)).
To measure the extent of this bias, we define subpopulations based on whether they mention a particular demographic identity, compute the sensitivity (a.k.a. recall, or true positive rate) and specificity (a.k.a. true negative rate) of the classifier on each subpopulation,
and then report the worst of these two metrics over all subpopulations of interest.
This is equivalent to further dividing each subpopulation into two groups according to the label, and then computing the accuracy on each of these two groups.

Specifically, for each of the 8 identities we study (e.g., “male”), we form 2 groups based on the toxicity label
(e.g., one group of comments that mention the male gender and are toxic, and another group that mentions the male gender and are not toxic), for a total of 16 groups.
These groups overlap (a comment might mention multiple identities) and are not a complete partition (a comment might not mention any identity).

We then measure a model’s performance by its worst-group accuracy, i.e., its lowest accuracy over these 16 groups.
A high worst-group accuracy (relative to average accuracy) implies that the model is not spuriously associating a demographic identity with toxicity.
We can view this subpopulation shift problem as testing on multiple test distributions (corresponding to different subsets of the test set, based on demographic identities and the label) and reporting the worst performance over these different test distributions.

We use 16 groups (8 identities ×\times 2 labels) instead of just 8 groups (8 identities) to capture the desire to balance true positive and true negative rates across each of the demographic identities.
Without splitting by the label, it would be possible for two different groups to have equal accuracies,
but one group might be much more likely to have non-toxic comments flagged as toxic, whereas the other group might be much more likely to have toxic comments flagged as non-toxic.
This would be undesirable from an application perspective, as such a model would still be biased against a particular demographic.
In Appendix [E.6.4](#S5.SS6.SSS4 "E.6.4 Additional details ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we further discuss the motivation for our choice of evaluation metric as well as its limitations.

As variability in performance over replicates can be high due to the small sizes of some demographic groups (Table [17](#S5.T17 "Table 17 ‣ Data processing. ‣ E.6.4 Additional details ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), we report results averaged over 5 random seeds, instead of the 3 seeds that we use for most other datasets.

##### Potential leverage.

Since demographic identity annotations are provided at training time, we have an i.i.d. dataset available at training time for each of the test distributions of interest (corresponding to each group).
Moreover, even though demographic identity annotations are unavailable at test time,
they are relatively straightforward to predict.

#### E.6.2 Baseline results

##### Model.

For all experiments, we fine-tuned DistilBERT-base-uncased models (Sanh et al., [2019](#bib.bib330)), using the implementation from Wolf et al. ([2019](#bib.bib405)) and with the following hyperparameter settings:
batch size 16; learning rate 10−5superscript10510^{-5} using the AdamW optimizer (Loshchilov and Hutter, [2019](#bib.bib247)) for 5 epochs with early stopping; an L2subscript𝐿2L\_{2}-regularization strength of 10−2superscript10210^{-2};
and a maximum number of tokens of 300, since 99.95% of the input examples had ≤\leq300 tokens.
The learning rate was chosen through a grid search over
{10−6,2×10−6,10−5,2×10−5}superscript1062superscript106superscript1052superscript105\{10^{-6},2\times 10^{-6},10^{-5},2\times 10^{-5}\}, and all other hyperparameters were simply set to standard/default values.

##### ERM results and performance drops.

The ERM model does well on average, with 92.2% average accuracy (Table [15](#S5.T15 "Table 15 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). However, it does poorly on some subpopulations, e.g., with 57.4% accuracy on toxic comments that mention other religions.
Overall, accuracy on toxic comments (which are a minority of the dataset) was lower than accuracy on non-toxic comments, so we also trained a reweighted model that balanced toxic and non-toxic comments by upsampling the toxic comments. This reweighted model had a slightly worse average accuracy of 89.8% and a better worst-group accuracy of 69.2% (Table [15](#S5.T15 "Table 15 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), Reweighted (label)), but a significant gap remains between average and worst-group accuracies.

We note that the relatively small size of some of the demographic subpopulations makes it infeasible to run a test-to-test comparison, i.e., estimate how well a model could do on each subpopulation (corresponding to demographic identity) if it were trained on just that subpopulation.
For example, Black comments comprise only <4% of the training data,
and training just on those Black comments is insufficient to achieve high in-distribution accuracy.
Without running the test-to-test comparison, it is possible that the gap between average and worst-group accuracies can be explained at least in part by differences in the intrinsic difficulty of some of the subpopulations, e.g., the labels of some subpopulations might be noisier because human annotators might disagree more frequently on comments mentioning a particular demographic identity.
Future work will be required to establish estimates of in-distribution accuracies for each subpopulation that can account for these differences.

Table 15: Baseline results on CivilComments-wilds. The reweighted (label) algorithm samples equally from the positive and negative class; the group DRO (label) algorithm additionally weights these classes so as to minimize the maximum of the average positive training loss and average negative training loss.
Similarly, the reweighted (label ×\times Black) and group DRO (label ×\times Black) algorithms sample equally from the four groups corresponding to all combinations of class and whether there is a mention of Black identity. The CORAL and IRM algorithms extend the reweighted algorithm by adding their respective penalty terms, so they also sample equally from each group.
We show standard deviation across 5 random seeds in parentheses.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Algorithm | Avg val acc | Worst-group val acc | Avg test acc | Worst-group test acc |
| ERM | 92.3 (0.2) | 50.5 (1.9) | 92.2 (0.1) | 56.0 (3.6) |
| Reweighted (label) | 90.1 (0.4) | 65.9 (1.8) | 89.8 (0.4) | 69.2 (0.9) |
| Group DRO (label) | 90.4 (0.4) | 65.0 (3.8) | 90.2 (0.3) | 69.1 (1.8) |
| Reweighted (label ×\times Black) | 89.5 (0.6) | 66.6 (1.5) | 89.2 (0.6) | 66.2 (1.2) |
| CORAL (label ×\times Black) | 88.9 (0.6) | 64.7 (1.4) | 88.7 (0.5) | 65.6 (1.3) |
| IRM (label ×\times Black) | 89.0 (0.7) | 65.9 (2.8) | 88.8 (0.7) | 66.3 (2.1) |
| Group DRO (label ×\times Black) | 90.1 (0.4) | 67.7 (1.8) | 89.9 (0.5) | 70.0 (2.0) |




Table 16: Accuracies on each subpopulation in CivilComments-wilds, averaged over models trained by group DRO (label).

|  |  |  |
| --- | --- | --- |
| Demographic | Test accuracy on non-toxic comments | Test accuracy on toxic comments |
| Male | 88.4 (0.7) | 75.1 (2.1) |
| Female | 90.0 (0.6) | 73.7 (1.5) |
| LGBTQ | 76.0 (3.6) | 73.7 (4.0) |
| Christian | 92.6 (0.6) | 69.2 (2.0) |
| Muslim | 80.7 (1.9) | 72.1 (2.6) |
| Other religions | 87.4 (0.9) | 72.0 (2.5) |
| Black | 72.2 (2.3) | 79.6 (2.2) |
| White | 73.4 (1.4) | 78.8 (1.7) |

##### Additional baseline methods.

The CORAL, IRM, and group DRO baselines involve partitioning the training data into disjoint domains.
We study the following partitions, corresponding to different rows in Table [15](#S5.T15 "Table 15 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"):

1. 1.

   Label: 2 domains, 1 for each class.
2. 2.

   Label ×\times Black: 4 domains, 1 for each combination of class and Black.

On the Label partition, we used Group DRO to train a model that seeks to balance the losses on the positive and negative examples. This performs similarly to the standard reweighted models described above
(Table [15](#S5.T15 "Table 15 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), Group DRO (label)).
We found that the worst-performing demographic for non-toxic comments was the Black demographic (Table [16](#S5.T16 "Table 16 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), which motivated the Label ×\times Black partition. There, we used CORAL, IRM, and Group DRO to train models. However, these models did not perform significantly better (Table [15](#S5.T15 "Table 15 ‣ ERM results and performance drops. ‣ E.6.2 Baseline results ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), label ×\times Black). While there were slight improvements on the Black groups, accuracy degraded on some other groups like non-toxic LBGTQ comments.

We note that our implementations of CORAL and IRM are built on top of the standard reweighting algorithm, i.e., they sample equally from each group. As these two algorithms perform similarly to
reweighting, it indicates that the additional penalty term is not significantly affecting performance.
Indeed, our grid search for the penalty weights selected the lowest value of the penalties (λ=10.0𝜆10.0\lambda=10.0 for CORAL and λ=1.0𝜆1.0\lambda=1.0 for IRM).

##### Discussion.

Adapting the baseline methods to handle multiple overlapping groups, which were not studied in their original settings, could be a potential approach to improving accuracy on this task.
Another potential approach is using baselining to account for different groups having different intrinsic levels of difficulty (Oren et al., [2019](#bib.bib281)).
For example, comments mentioning different demographic groups might differ in terms of how subjective classifying them is.
Others have also explored specialized data augmentation techniques for mitigating demographic biases in toxicity classifiers (Zhao et al., [2018](#bib.bib431)).

Adragna et al. ([2020](#bib.bib2)) recently used a simplified variant of the CivilComments dataset, with artificially-constructed training and test environments, to show a proof-of-concept that IRM can improve performance on minority groups. Methods such as IRM and group DRO rely heavily on the choice of groups/domains/environments; investigating the effect of different choices would be a useful direction for future work.
Other recent work has studied methods that try to automatically learn groups, for example, through unsupervised clustering (Oren et al., [2019](#bib.bib281); Sohoni et al., [2020](#bib.bib348))
or identifying high-loss points (Nam et al., [2020](#bib.bib270); Liu et al., [2021a](#bib.bib241)).

Toxicity classification is one application where human moderators can work together with an ML model to handle examples that the model is unsure about. However, Jones et al. ([2021](#bib.bib192)) found that using selective classifiers—where the model is allowed to abstain if it is unsure—can actually further worsen performance on minority subpopulations. This suggests that in addition to having low accuracy on minority subpopulations, standard models can be poorly calibrated on them.

Another important consideration for toxicity detection in practice is shifts over time, as online discourse changes quickly, and what is seen as toxic today might not have even appeared in the dataset from a few months ago. We do not study this distribution shift in this work. One limitation of the CivilComments-wilds dataset is that it is fixed to a relatively short period in time, with most comments being written in the span of a year; this makes it harder to use as a dataset for studying temporal shifts.

Finally, we note that collecting “ground truth” human annotation of toxicity is itself a subjective and challenging process; recent work has studied ways of making it less biased and more efficient (Sap et al., [2019](#bib.bib332); Han and Tsvetkov, [2020](#bib.bib162)).

#### E.6.3 Broader context

The CivilComments-wilds dataset does not assume that user demographics are available; instead, it uses mentions of different demographic identities in the actual comment text.
For example, we want models that do not associate comments that mention being Black with being toxic, regardless of whether a Black or non-Black person wrote the comment.
This setting is particularly relevant when user demographics are unavailable, e.g., when considering anonymous online comments.

A related and important setting is subpopulation shifts with respect to user demographics (e.g., the demographics of the author of the comment, regardless of the content of the comment).
Such demographic disparities have been widely documented in natural language and speech processing tasks (Hovy and Spruit, [2016](#bib.bib179)), among other areas.
For example, NLP models have been shown to obtain worse performance on African-American Vernacular English compared to Standard American English on
part-of-speech tagging (Jørgensen et al., [2015](#bib.bib195)),
dependency parsing (Blodgett et al., [2016](#bib.bib50)),
language identification (Blodgett and O’Connor, [2017](#bib.bib49)),
and auto-correct systems (Hashimoto et al., [2018](#bib.bib166)).
Similar disparities exist in speech, with state-of-the-art commercial systems obtaining higher word error rates on particular races (Koenecke et al., [2020](#bib.bib208)) and dialects (Tatman, [2017](#bib.bib366)).

These disparities are present not just in academic models, but in large-scale commercial systems that are already widely deployed, e.g., in speech-to-text systems from Amazon, Apple, Google, IBM, and Microsoft (Tatman, [2017](#bib.bib366); Koenecke et al., [2020](#bib.bib208)) or language identification systems from IBM, Microsoft, and Twitter (Blodgett and O’Connor, [2017](#bib.bib49)).
Indeed, the original CivilComments dataset was developed by Google’s Conversation AI team, which is also behind a public toxicity classifier (Perspective API) that was developed in partnership with The New York Times (NYTimes, [2016](#bib.bib279)).

#### E.6.4 Additional details

##### Evaluation metrics.

The evaluation metric used in the original competition was a complex weighted combination of various metrics, including subgroup AUCs for each demographic identity, and a new pinned AUC metric introduced by the original authors (Borkan et al., [2019b](#bib.bib55)); conceptually, these metrics also measure the degree to which model accuracy is uniform across the different identities.
After discussion with the original authors, we replace the composite metric with worst-group accuracy (i.e., worst TPR/FPR over identities) for simplicity.
Measuring subgroup AUCs can be misleading in this context, because it assumes that the classifier can set separate thresholds for different subgroups (Borkan et al., [2019b](#bib.bib55), [a](#bib.bib54)).

One downside is that measuring worst-group accuracy treats false positives and false negatives equally.
In deployment systems, one might want to weight these differently, e.g., using cost-sensitive learning or by simply raising or lowering the classification threshold,
especially since real data is highly imbalanced (with a lot more negatives than positives).
One could also binarize the labels and identities differently: in this benchmark, we simply use majority voting from the annotators.

Perhaps more fundamentally, even if TPR and FPR were balanced across different identities, this need not imply unambiguously equitable performance, because different subpopulations might have different intrinsic levels of noise and difficulty. See Corbett-Davies and Goel ([2018](#bib.bib92)) for more discussion of this problem of infra-marginality.

In practice, models might also do poorly on intersections of groups (Kearns et al., [2018](#bib.bib201)), e.g., on comments that mention multiple identities. Given the size of the dataset and comparative rarity of some identities and of toxic comments in general, accuracies on these intersections are difficult to estimate from this dataset.
A potential avenue of future work is to develop methods for evaluating models on such subgroups, e.g., by generating data in particular groups through templates
(Park et al., [2018](#bib.bib286); Ribeiro et al., [2020](#bib.bib314)).

##### Data processing.

The CivilComments-wilds dataset comprises comments from a large set of articles from the Civil Comments platform, annotated for toxicity and demographic identities (Borkan et al., [2019b](#bib.bib55)). We partitioned the articles into disjoint training, validation, and test splits, and then formed the corresponding datasets by taking all comments on the articles in those splits.
In total, the training set comprised 269,038 comments (60% of the data); the validation set comprised 45,180 comments (10%); and the test set comprised 133,782 (30%).

Table 17: Group sizes in the test data for CivilComments-wilds. The training and validation data follow similar proportions.

|  |  |  |
| --- | --- | --- |
| Demographic | Number of non-toxic comments | Number of toxic comments |
| Male | 12092 | 2203 |
| Female | 14179 | 2270 |
| LGBTQ | 3210 | 1216 |
| Christian | 12101 | 1260 |
| Muslim | 5355 | 1627 |
| Other religions | 2980 | 520 |
| Black | 3335 | 1537 |
| White | 5723 | 2246 |

##### Modifications to the original dataset.

The original dataset888<www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/> also had a training and test split with disjoint articles. These splits are related to ours in the following way. Let the number of articles in the original test split be m𝑚m. To form our validation split, we took m𝑚m articles (sampled uniformly at random) from the original training split, and to form our test split, we took 2​m2𝑚2m articles (also sampled uniformly at random) from the original training split and added it to the existing test split.
We added a fixed validation set to allow other researchers to be able to compare methods more consistently, and we tripled the size of the test set to allow for more accurate worst-group accuracy measurement.

Similarly, we combined some of the demographic identities in the original dataset to obtain larger groups (for which we could more accurately estimate accuracy). Specifically, we created an aggregate LGBTQ identity that combines the original
homosexual\_gay\_or\_lesbian,
bisexual,
other\_sexual\_orientation,
transgender, and
other\_gender identities (e.g., it is 1 if any of those identities are 1),
and an aggregate other\_religions identity that combines the original
jewish,
hindu,
buddhist,
atheist, and
other\_religion identities.
We also omitted the psychiatric\_or\_mental\_illness identity, which was evaluated in the original Kaggle competition, because of a lack of sufficient data for accurate estimation; but we note that baseline group accuracies for that identity seemed higher than for the other groups, so it is unlikely to factor into worst-group accuracy.
In our new split, each identity we evaluate on (male, female, LGBTQ, Christian, Muslim, other\_religions, Black, and White) has at least 500 positive and 500 negative examples.
In Table [17](#S5.T17 "Table 17 ‣ Data processing. ‣ E.6.4 Additional details ‣ E.6 CivilComments-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") we show the sizes of each subpopulation in the test set; the training and validation sets follow similar proportions.

For convenience, we also add an identity\_any identity; this combines all of the identities in the original dataset, including psychiatric\_or\_mental\_illness and related identities.

##### Additional baseline results.

We also trained a group DRO model using 29=512superscript295122^{9}=512 domains, 1 for each combination of class and the 8 identities. This model performed similarly to the other group DRO models.

##### Additional data sources.

All of the data, including the data with identity annotations that we use and the data with just label annotations, are also annotated for additional toxicity subtype attributes, specifically
severe\_toxicity,
obscene,
threat,
insult,
identity\_attack, and
sexual\_explicit.
These annotations can be used to train models that are more aware of the different ways that a comment can be toxic;
in particular, using the identity\_attack attribute to learn which comments are toxic because of the use of identities might help the model learn how to avoid spurious associations between toxicity and identity.
These additional annotations are included in the metadata provided through the Wilds package.

The original CivilComments dataset (Borkan et al., [2019b](#bib.bib55)) also contains ≈\approx1.5M training examples that have toxicity (label) annotations but not identity (group) annotations.
For simplicity, we have omitted these from the current version of CivilComments-wilds.
These additional data points can be downloaded from the original data source and could be used, for example, by first inferring which group each additional point belongs to, and then running group DRO or a similar algorithm that uses group annotations at training time.

### E.7 FMoW-wilds

ML models for satellite imagery can enable global-scale monitoring of sustainability and economic challenges, aiding policy and humanitarian efforts in applications such as deforestation tracking (Hansen et al., [2013](#bib.bib164)), population density mapping (Tiecke et al., [2017](#bib.bib374)), crop yield prediction (Wang et al., [2020b](#bib.bib395)), and other economic tracking applications (Katona et al., [2018](#bib.bib199)).
As satellite data constantly changes due to human activity and environmental processes, these models must be robust to distribution shifts over time.
Moreover, as there can be disparities in the data available between regions,
these models should ideally have uniformly high accuracies instead of only doing well on data-rich regions and countries.

We study this problem on a variant of the Functional Map of the World dataset (Christie et al., [2018](#bib.bib83)).

#### E.7.1 Setup

##### Problem setting.

We consider a hybrid domain generalization and subpopulation shift problem, where the input x𝑥x is a RGB satellite image (resized to 224 ×\times 224 pixels), the label y𝑦y is one of 62 building or land use categories, and the domain d𝑑d represents both the year the image was taken as well as its geographical region (Africa, the Americas, Oceania, Asia, or Europe).
We aim to solve both a domain generalization problem across time and improve subpopulation performance across regions.

##### Data.

FMoW-wilds is based on the Functional Map of the World dataset (Christie et al., [2018](#bib.bib83)), which collected and categorized high-resolution satellite images from over 200 countries based on the functional purpose of the buildings or land in the image, over the years 2002–2018 (see Figure [9](#S4.F9 "Figure 9 ‣ 4.3.1 FMoW-wilds: Land use classification across different regions and years ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
We use a subset of this data and split it into three time range domains, 2002–2013, 2013–2016, and 2016–2018, as well as five geographical regions as subpopulations (Africa, Americas, Oceania, Asia, and Europe).
For each example, we also provide the timestamp and location coordinates, though our baseline models only use the coarse time ranges and geographical regions instead of these additional metadata.

We use the following data splits:

1. 1.

   Training: 76,863 images from the years 2002–2013.
2. 2.

   Validation (OOD): 19,915 images from the years from 2013–2016.
3. 3.

   Test (OOD): 22,108 images from the years from 2016–2018.
4. 4.

   Validation (ID): 11,483 images from the years from 2002–2013.
5. 5.

   Test (ID): 11,327 images from the years from 2002–2013.

The original dataset did not evaluate models under distribution shifts. Our training split is a subset of the original training dataset, filtered for images in the appropriate time range; similarly, our OOD and ID validation splits are subsets of the original validation dataset, and our OOD and ID test splits are subsets of the original test dataset. See Appendix [E.7.4](#S5.SS7.SSS4 "E.7.4 Additional details ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more dataset details.

The train/val/test data splits contain images from disjoint location coordinates, and all splits contain data from all 5 geographic regions.
The ID and OOD splits within the test and validation sets may have overlapping locations, but have non-overlapping time ranges.
There is a disparity in the number of examples in each region, with Africa and Oceania having the least examples (Figure [23](#S5.F23 "Figure 23 ‣ Data. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")); this could be due to bias in sampling and/or a lack of infrastructure and land data in certain regions.

![Refer to caption](/html/2012.07421/assets/x19.png)

![Refer to caption](/html/2012.07421/assets/x20.png)

Figure 23: Number of examples from each region of the world in FMoW-wilds on the ID vs. OOD splits of the data. There is much less data from Africa and Oceania than other regions.

##### Evaluation.

We evaluate models by their average and worst-region OOD accuracies. The former measures the ability of the model to generalize across time, while the latter additionally measures how well models do across different regions/subpopulations under a time shift.

Table 18: Average and worst-region accuracies (%) under time shifts in FMoW-wilds. Models are trained on data before 2013 and tested on held-out location coordinates from in-distribution (ID) or out-of-distribution (OOD) test sets.
ID results correspond to the train-to-train setting.
Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Validation (ID) | Validation (OOD) | Test (ID) | Test (OOD) |
| Average |  |  |  |  |
| ERM | 61.2 (0.52) | 59.5 (0.37) | 59.7 (0.65) | 53.0 (0.55) |
| CORAL | 58.3 (0.28) | 56.9 (0.25) | 57.2 (0.90) | 50.5 (0.36) |
| IRM | 58.6 (0.07) | 57.4 (0.37) | 57.7 (0.10) | 50.8 (0.13) |
| Group DRO | 60.5 (0.36) | 58.8 (0.19) | 59.4 (0.11) | 52.1 (0.50) |
| Worst-region |  |  |  |  |
| ERM | 59.2 (0.69) | 48.9 (0.62) | 58.3 (0.92) | 32.3 (1.25) |
| CORAL | 55.9 (0.50) | 47.1 (0.43) | 55.0 (1.02) | 31.7 (1.24) |
| IRM | 56.6 (0.59) | 47.5 (1.57) | 56.0 (0.34) | 30.0 (1.37) |
| Group DRO | 57.9 (0.62) | 46.5 (0.25) | 57.8 (0.60) | 30.8 (0.81) |




Table 19: The regional accuracies of models trained on data before 2013 and tested on held-out locations from ID (<2013absent2013<2013) or OOD (≥2016absent2016\geq 2016) test sets in FMoW-wilds. ID results correspond to the train-to-train setting. Standard deviations over 3 trials are in parentheses.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Asia | Europe | Africa | Americas | Oceania | Worst region |
| OOD Test |  |  |  |  |  |  |
| ERM | 55.4 (0.95) | 55.6 (0.53) | 32.3 (1.25) | 55.7 (0.48) | 59.1 (0.85) | 32.3 (1.25) |
| CORAL | 52.4 (0.96) | 52.6 (0.82) | 31.7 (1.24) | 53.3 (0.27) | 56.0 (2.02) | 31.7 (1.24) |
| IRM | 52.9 (0.73) | 53.9 (0.28) | 30.0 (1.37) | 53.7 (0.51) | 55.0 (2.22) | 30.0 (1.37) |
| Group DRO | 54.7 (0.52) | 55.1 (0.39) | 30.8 (0.81) | 54.6 (0.48) | 58.5 (1.65) | 30.8 (0.81) |
| ID Test |  |  |  |  |  |  |
| ERM | 58.9 (1.19) | 58.4 (0.81) | 69.1 (2.64) | 61.4 (0.35) | 69.9 (0.53) | 58.3 (0.92) |
| CORAL | 56.6 (1.35) | 55.0 (1.02) | 69.2 (2.92) | 59.7 (0.83) | 70.8 (2.53) | 55.0 (1.02) |
| IRM | 56.9 (0.62) | 56.0 (0.34) | 69.7 (2.16) | 59.7 (0.49) | 68.3 (2.00) | 56.0 (0.34) |
| Group DRO | 58.7 (0.33) | 57.9 (0.74) | 69.2 (0.28) | 61.1 (0.57) | 68.8 (2.38) | 57.8 (0.60) |




Table 20: Mixed-to-test comparison for ERM models on FMoW-wilds.
In the official setting, we train on ID examples (i.e., data from 2002–2013),
whereas in the mixed-to-test ID setting, we train on ID + OOD examples (i.e., the same amount of data but half from 2002–2013 and half from 2013–2018, using a held-out set of data from 2013–2018).
In both settings, we test on the same Test (ID) data (from 2002–2013) and Test (OOD) data (from 2013–2018) described in the official split.
Models trained on the official split degrade in performance under the time shift, especially on the last year (2017) of the test data, and also fare poorly on the subpopulation shift, with low worst-region accuracy.
Models trained on the mixed-to-test split have higher OOD average and last year accuracy and much higher OOD worst-region accuracy.
Standard deviations over 3 trials are in parentheses.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Test (ID) | | Test (OOD) | | |
| Setting | Algorithm | Average | Worst-region | Average | Last year | Worst-region |
| Official | ERM | 59.7 (0.65) | 58.3 (0.92) | 53.0 (0.55) | 48.1 (1.20) | 32.3 (1.25) |
| Mixed-to-test | ERM | 59.0 (0.47) | 56.9 (0.80) | 57.4 (0.27) | 54.3 (0.22) | 48.6 (0.89) |

##### Potential leverage.

FMoW-wilds considers both domain generalization across time and subpopulation shift across regions. As we provide both time and region annotations, models can leverage the structure across both space and time to improve robustness.
For example, one hypothesis is that infrastructure development occurs smoothly over time. Utilizing this gradual shift structure with the timestamp metadata may enable adaptation across longer time periods (Kumar et al., [2020](#bib.bib215)).
The data distribution may also shift smoothly over spatial locations, and so enforcing some consistency with respect to spatial structure may improve predictions (Rolf et al., [2020](#bib.bib319); Jean et al., [2018](#bib.bib188)).
Furthermore, to mitigate the fact that some regions (e.g., Africa) have less labeled data, one could potentially transfer knowledge of other regions with similar economies and infrastructure. The location coordinate metadata allows for transfer learning across similar locations at any spatial scale.

#### E.7.2 Baseline results

##### Model.

For all experiments, we follow Christie et al. ([2018](#bib.bib83)) and use a DenseNet-121 model (Huang et al., [2017](#bib.bib183)) pretrained on ImageNet and with no L2subscript𝐿2L\_{2} regularization.
We use the Adam optimizer (Kingma and Ba, [2015](#bib.bib207)) with an initial learning rate of 10−4superscript10410^{-4} that decays by 0.96 per epoch, and train for 50 epochs for with early stopping and with a batch size of 64.
All reported results are averaged over 3 random seeds.

##### ERM results and performance drops.

In the train-to-train comparison, Table [20](#S5.T20 "Table 20 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that average accuracy drops by 6.7% when evaluated on the OOD test set (≥2016absent2016\geq 2016) compared to the ID test set setting.
The drop in average accuracy is especially large (11.6%) on images from the last year of the dataset (2017), furthest in the future from the training set.
In addition, there is a substantial 26.0% drop in worst-region accuracy, with the model performing much worse in Africa than other regions (Table [19](#S5.T19 "Table 19 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

We also ran a mixed-to-test comparison where we mixed in some data from the OOD period (2013–2018) into the training set, while keeping the overall training set size constant.
A model trained on this mixed split had a much smaller drop in performance under the time and region shifts (Table [20](#S5.T20 "Table 20 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
While the magnitude of the ID-OOD gap in worst-region accuracy shrinks from 26.0% in the train-to-train setting to 16.3% in the mixed-to-test setting, the gap remains significant, implying that the drop in performance is largely due to the distribution shift across time and region instead of a change in the intrinsic difficulty of the OOD data.

##### Additional baseline methods.

We compare ERM against CORAL, IRM, and Group DRO, using examples from different years as distinct domains. Table [18](#S5.T18 "Table 18 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that many of these methods are comparable or worse than ERM in terms of both ID and OOD test performance.
As with most other datasets, our grid search selected the lowest values of the penalty weights for CORAL (λ=0.1𝜆0.1\lambda=0.1) and IRM (λ=1𝜆1\lambda=1).

##### Discussion.

Intriguingly, a large subpopulation shift across regions only occurs with a combination of time and region shift.
This is corroborated by the mixed-split region shift results (Table [20](#S5.T20 "Table 20 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), which do not have a time shift between training and test sets, and correspondingly do not display a large disparity in performance across regions.
This drop in performance may be partially due to label shift:
from Figure [24](#S5.F24 "Figure 24 ‣ Discussion. ‣ E.7.2 Baseline results ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we see that the label distributions between Africa and other regions are very different, e.g., with a large drop in recreational facilities and a sharp increase in single residential units.
We do not find a similarly large label shift between <2013absent2013<2013 and ≥2013absent2013\geq 2013 splits of the dataset.

![Refer to caption](/html/2012.07421/assets/x21.png)


Figure 24: Number of examples from each category in FMoW-wilds in non-African and African regions. There is a large label shift between non-African regions and Africa.

Despite having the smallest number of training examples (Figure [23](#S5.F23 "Figure 23 ‣ Data. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), the baseline models do not suffer a drop in performance in Oceania on validation or test sets (Table [19](#S5.T19 "Table 19 ‣ Evaluation. ‣ E.7.1 Setup ‣ E.7 FMoW-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). We hypothesize that infrastructure in Oceania is more similar to regions with a large amount of data than Africa.
In contrast, Africa may be more distinct and may have changed more drastically over 2002-2018, the time extent of the dataset.
This suggests that the subpopulation shift is not merely a function of the number of training examples.

We note that our dataset splits can separate on particular factors such as the introduction of new sensors, which is natural with progression over time. For example, the WorldView-3 sensor came online in 2014. Future work should look into the role of auxiliary factors such as new sensors that are associated with time but may be controllable. We did not find a sharp difference in performance due to the introduction of WorldView-3; we found that the performance decays gradually over time, suggesting that the performance drop comes from other factors.

As with PovertyMap-wilds, there are important ethical considerations associated with remote sensing applications,
e.g., around surveillance and privacy issues, as well as the potential for systematic biases that negatively affect particular populations.
As an example of the latter, the poor model performance on satellite images from Africa that we observe in FMoW-wilds raises issues of bias and fairness.
With regard to privacy, we note that the image resolution in FMoW-wilds is lower than that of other public and easily-accessible satellite data such as that from Google Maps.
We refer interested readers to the UNICEF discussion paper by Berman et al. ([2018](#bib.bib45)) for a more in-depth discussion of the ethics of remote sensing especially as it pertains to development and humanitarian endeavors.

#### E.7.3 Broader context

Recognizing infrastructure and land features is crucial to many remote sensing applications. For example, in crop land prediction Wang et al. ([2020b](#bib.bib395)), recognizing gridded plot lines, plot circles, farm houses, and other visible features are important in recognizing crop fields. However, farming practices and equipment evolve over time and vary widely across the world, requiring both robust object recognition and synthesis of their different usage patterns.

Although the data is typically limited, we desire generalization on a global scale without requiring frequent large-scale efforts to gather more ground-truth data.
It is natural to have labeled data with limited temporal or spatial extent since ground truth generally must be verified on the ground or requires manual annotations from domain experts (i.e., they are often hard to be crowdsourced). A number of existing remote sensing datasets have limited spatial or temporal scope, including the UC Merced Land Use Dataset (Yang and Newsam, [2010](#bib.bib419)), TorontoCity (Wang et al., [2017](#bib.bib394)), and SpaceNet (DigitalGlobe and Works, [2016](#bib.bib110)).
However, works based on these datasets generally do not systematically study shifts in time or location.

#### E.7.4 Additional details

##### Data processing and modifications to the original dataset.

The FMoW-wilds dataset is derived from Christie et al. ([2018](#bib.bib83)), which collected over 1 million satellite images from over 200 countries over 2002-2018.
We use the RGB version of the original dataset, which contains 523,846 total examples, excluding the multispectral version of the images.
Methods that can utilize a sequence of images can group the images from the same location across multiple years together as input, but we consider the simple formulation here for our baseline evaluation.

The original dataset from Christie et al. ([2018](#bib.bib83)) is provided as a set of hierarchical directories with JPEG images of varying sizes.
To reduce download times and I/O usage, we resize these images to 224 ×\times 224 pixels, and then store them as PNG images.
We also collect all the metadata into CSV format for easy processing.

The original dataset is posed as a image time-series classification problem, where the model has access to a sequence of images at each location. For simplicity, we treat each image as a separate example, while making sure that the data splits all contain disjoint locations. We use the train/val/test splits from the original dataset, but separate out two OOD time segments: we treat the original validation data from 2013-2016 as OOD val and the original test data from 2016-2018 as OOD test. We remove data from after 2013 from the training set, which reduces the size of the training set in comparison to the original dataset.

##### Additional challenges in high-resolution satellite datasets.

Compared to PovertyMap-wilds, FMoW-wilds contains much higher resolution images (sub-meter resolution vs. 30m resolution) and contains a larger variety of viewpoints/tilts, both of which could present computational or algorithmic challenges. For computational purposes, we resized all images to 224×224224224224\times 224 (following Christie et al. ([2018](#bib.bib83))), but raw images can be thousands of pixels wide. Some recent works have tried to balance this tradeoff between viewing overall context and the fine-grained detail (Uzkent and Ermon, [2020](#bib.bib382); Kim et al., [2016a](#bib.bib204)), but how best to do this is an open question. FMoW-wilds also contains additional information on azimuth and cloud cover which could be used to correct for the variety in viewpoints and image quality.

### E.8 PovertyMap-wilds

A different application of satellite imagery is poverty estimation across different spatial regions, which is essential for targeted humanitarian efforts in poor regions (Abelson et al., [2014](#bib.bib1); Espey et al., [2015](#bib.bib124)).
However, ground-truth measurements of poverty are lacking for much of the developing world, as field surveys are expensive (Blumenstock et al., [2015](#bib.bib51); Xie et al., [2016](#bib.bib415); Jean et al., [2016](#bib.bib187)).
For example, at least 4 years pass between nationally representative consumption or asset wealth surveys in the majority of African countries, with seven countries that had either never conducted a survey or had gaps of over a decade between surveys (Yeh et al., [2020](#bib.bib422)).
One approach to this problem is to train ML models on countries with ground truth labels and then deploy them to different countries where we have satellite data but no labels.

We study this problem through a variant of the poverty mapping dataset collected by Yeh et al. ([2020](#bib.bib422)).

#### E.8.1 Setup

##### Problem setting.

We consider a hybrid domain generalization and subpopulation shift problem,
where the input x𝑥x is a multispectral LandSat satellite image with 8 channels (resized to 224 ×\times 224 pixels),
the output y𝑦y is a real-valued asset wealth index computed from Demographic and Health Surveys (DHS) data, and the domain d𝑑d represents the country the image was taken in and whether the image is of an urban or rural area.
We aim to solve both a domain generalization problem across country borders and improve subpopulation performance across urban and rural areas.

##### Data.

PovertyMap-wilds is based on a dataset collected by Yeh et al. ([2020](#bib.bib422)), which assembles satellite imagery and survey data at 19,669 villages from 23 African countries between 2009 and 2016 (Figure [10](#S4.F10 "Figure 10 ‣ 4.3.2 PovertyMap-wilds: Poverty mapping across different countries ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
Each input image has 8 channels: 7 from the LandSat satellite and an 8th channel for nighttime light intensity from a separate satellite,
as prior work has established that these night lights correlate with poverty measures (Noor et al., [2008](#bib.bib276); Elvidge et al., [2009](#bib.bib122)).

There are 23×2=462324623\times 2=46 domains corresponding to the 23 countries and whether the location is urban or rural.
Each example comes with metadata on its location coordinates, survey year, and its urban/rural classification.

In contrast to other datasets, which have a single fixed ID/OOD split, the relatively small size of PovertyMap-wilds allows us to use 5 different folds, where each fold defines a different set of OOD countries.
In each fold, we use the following splits of the data (the number of countries and images in each split varies slightly from fold to fold):

1. 1.

   Training: ∼similar-to\sim10000 images from 13–14 countries.
2. 2.

   Validation (OOD): ∼similar-to\sim4000 images from 4–5 different countries (distinct from training and test (OOD) countries).
3. 3.

   Test (OOD): ∼similar-to\sim4000 images from 4–5 different countries (distinct from training and validation (OOD) countries).
4. 4.

   Validation (ID): ∼similar-to\sim1000 images from the same 13–14 countries in the training set.
5. 5.

   Test (ID): ∼similar-to\sim1000 images from the same 13–14 countries in the training set.

All splits contain images of both urban and rural locations, with the countries assigned randomly to each split in each fold.

The distribution of wealth may shift across countries due to differing levels economic development, agricultural practices, and other factors. For example, Abelson et al. ([2014](#bib.bib1)) use thatched vs. metal roofs to distinguish between poor and wealthy households, respectively in Kenya and Uganda. However, other countries may have a different mapping of roof type to wealth where metal roofs signify more poor households. Similar issues can arise when looking at the health of crops (related to vegetation indices such as NDVI that are simple functions of the multispectral channels in the satellite image) as a sign for wealth in rural areas, since crop health is related to climate and the choice of crops, which vary upon region.

Asset wealth may also shift dramatically between countries. Figure [25](#S5.F25 "Figure 25 ‣ Data. ‣ E.8.1 Setup ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows the mean asset wealth per country, as well as urban vs. rural asset wealth per country.
Mean asset wealth ranges from -0.4 to +0.8 depending on the country.
There is a stark difference between mean asset wealth in urban and rural areas, with urban asset wealth being positive in all countries while rural mean asset wealth being mostly negative.

![Refer to caption](/html/2012.07421/assets/x22.png)

![Refer to caption](/html/2012.07421/assets/x23.png)

Figure 25: Mean asset wealth by country on aggregate as well as urban and rural splits for each country, computed on the full dataset.

##### Evaluation.

As is standard in the literature (Jean et al., [2016](#bib.bib187); Yeh et al., [2020](#bib.bib422)), the models are evaluated on the Pearson correlation (r𝑟r) between their predicted and actual asset wealth indices.
We measure the average correlation, to test generalization under country shifts, and also the lower of the correlations on the urban and rural subpopulations, to test generalization between urban and rural subpopulations.
We report the latter as previous works on poverty prediction from satellite imagery have noted that a significant part of model performance relies on distinguishing urban vs. rural areas, and improving performance within these subpopulations is an ongoing challenge, with rural areas generally faring worse under existing models (Jean et al., [2016](#bib.bib187); Yeh et al., [2020](#bib.bib422)).

We average all correlations across the 5 different folds, using 1 random seed per fold. The resulting standard deviations reflect the fact that different folds have different levels of difficulty (e.g., depending on how similar the ID and OOD countries are).
For the purposes of comparing different algorithms and models, we note that these standard deviations might make the comparisons appear noisier than they are, since a model might perform similarly across random seeds but still have a high standard deviation if it has different performances on different folds on the data.
In contrast, other Wilds datasets report results on the same data split but averaged across different random seeds.

##### Potential leverage.

Large socioeconomic differences between countries makes generalization across borders challenging. However, some indicators of wealth are known to be robust and are able to be seen from space. For example, roof type (e.g. thatched or metal roofing) has been shown to be a reliable proxy for wealth (Abelson et al., [2014](#bib.bib1)), and contextual factors such as the health of nearby croplands, the presence of paved roads, and connections to urban areas are plausibly reliable signals for measuring poverty.
Poverty measures are also known to be highly correlated across space, meaning nearby villages will likely have similar poverty measures, and methods can utilize this spatial structure (using the provided location coordinate metadata) to improve predictions (Jean et al., [2018](#bib.bib188); Rolf et al., [2020](#bib.bib319)).
We show the correlation with distance in Figure [26](#S5.F26 "Figure 26 ‣ Potential leverage. ‣ E.8.1 Setup ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), which plots the distance between pairs of data points against the absolute differences in asset wealth between pairs.

![Refer to caption](/html/2012.07421/assets/x24.png)


Figure 26: Mean absolute difference in asset wealth between two data points in the full dataset as a function of (great circle) distance between the two points. Smaller distances between data points correlate with more similar asset wealth measures. The pairs are binned by distance on a log (base 10) scale (100 bins), and the mean value of each bin is plotted at the midpoint distance of each bin.

#### E.8.2 Baseline results

##### Model.

For all experiments, we follow Yeh et al. ([2020](#bib.bib422)) and train a ResNet-18 model (He et al., [2016](#bib.bib167)) to minimize squared error.
We use the Adam optimizer (Kingma and Ba, [2015](#bib.bib207)) with an initial learning rate of 10−3superscript10310^{-3} that decays by 0.96 per epoch, and train for 200 epochs for with early stopping (on OOD r𝑟r) and with a batch size of 64.

##### ERM results and performance drops.

When shifting across country borders, Table [21](#S5.T21 "Table 21 ‣ ERM results and performance drops. ‣ E.8.2 Baseline results ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that ERM suffers a 0.04 drop in average r𝑟r in the official OOD setting compared to the train-to-train ID setting.
Moreover, the drop in performance is exacerbated when looking at urban and rural subpopulations, even though all splits contain urban and rural examples;
the difference in worst r𝑟r over the urban and rural subpopulations triples from 0.04 to 0.12 compared to the difference in average r𝑟r.
Correlation is consistently lower on the rural subpopulation than the urban subpopulation.

We ran an additional mixed-to-test comparison where we considered an alternative training set with data that was uniformly sampled from all countries, while keeping the overall training set size constant (i.e., compared to the standard training set, it has fewer examples from each country, but data from more countries).
A model trained on this mixed split had a much smaller drop in performance between the ID and OOD test sets (Table [22](#S5.T22 "Table 22 ‣ ERM results and performance drops. ‣ E.8.2 Baseline results ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), which implies that the performance drop between the ID and OOD test sets is largely due to the distribution shift from seen to unseen countries.

Table 21: Pearson correlation r𝑟r (higher is better) on in-distribution and out-of-distribution (unseen countries) held-out sets in PovertyMap-wilds, including results on the rural subpopulations.
ID results correspond to the train-to-train setting.
All results are averaged over 5 different OOD country folds taken from Yeh et al. ([2020](#bib.bib422)), with standard deviations across different folds in parentheses.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Validation (ID) | Validation (OOD) | Test (ID) | Test (OOD) |
| Overall |  |  |  |  |
| ERM | 0.82 (0.02) | 0.80 (0.04) | 0.82 (0.03) | 0.78 (0.04) |
| CORAL | 0.82 (0.00) | 0.80 (0.04) | 0.83 (0.01) | 0.78 (0.05) |
| IRM | 0.82 (0.02) | 0.81 (0.03) | 0.82 (0.02) | 0.77 (0.05) |
| Group DRO | 0.78 (0.03) | 0.78 (0.05) | 0.80 (0.03) | 0.75 (0.07) |
| Worst urban/rural subpop |  |  |  |  |
| ERM | 0.58 (0.07) | 0.51 (0.06) | 0.57 (0.07) | 0.45 (0.06) |
| CORAL | 0.59 (0.04) | 0.52 (0.06) | 0.59 (0.03) | 0.44 (0.06) |
| IRM | 0.57 (0.06) | 0.53 (0.05) | 0.57 (0.08) | 0.43 (0.07) |
| Group DRO | 0.49 (0.08) | 0.46 (0.04) | 0.54 (0.11) | 0.39 (0.06) |




Table 22: Mixed-to-test comparison for ERM models on PovertyMap-wilds.
In the official OOD setting, we train on data from one set of countries, and then test on a different set of countries.
In the mixed-to-test setting, we train on the same amount of data but sampled uniformly from all countries, and then test on data from the same countries as in the official setting.
The Test (ID) and Test (OOD) sets used for the mixed-to-test results are smaller (subsampled at random) than those used for the official results, as some test examples were used for the training set in the mixed-to-test setting.
Models trained on the official split degrade in performance, especially on rural subpopulations,
while models trained on the mixed-to-test split do not.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Test (ID) | | | Test (OOD) | | |
| Setting | Overall r𝑟r | Rural r𝑟r | Urban r𝑟r | Overall r𝑟r | Rural r𝑟r | Urban r𝑟r |
| Official | 0.82 (0.03) | 0.57 (0.07) | 0.66 (0.04) | 0.78 (0.04) | 0.46 (0.05) | 0.59 (0.11) |
| Mixed-to-test | 0.83 (0.02) | 0.61 (0.08) | 0.65 (0.06) | 0.83 (0.03) | 0.60 (0.06) | 0.65 (0.06) |

##### Additional baseline methods.

We trained models with CORAL, IRM, and Group DRO, taking examples from different countries as coming from distinct domains.
Table [21](#S5.T21 "Table 21 ‣ ERM results and performance drops. ‣ E.8.2 Baseline results ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that these baselines are generally comparable to ERM and that they continue to be susceptible to shifts across countries and urban/rural areas.
As with most other datasets, our grid search selected the lowest values of the penalty weights for CORAL (λ=0.1𝜆0.1\lambda=0.1) and IRM (λ=1𝜆1\lambda=1).

##### Discussion.

These results corroborate performance drops seen in previous out-of-country generalization tests for poverty prediction from satellite imagery (Jean et al., [2016](#bib.bib187)).
In general, differences in infrastructure, economic development, agricultural practices, and even cultural differences can cause large shifts across country borders.
Differences between urban and rural subpopulations have also been well-documented (Jean et al., [2016](#bib.bib187); Yeh et al., [2020](#bib.bib422)). Models based on nighttime light information could suffer more in rural areas where nighttime light intensity is uniformly low or even zero.

Since survey years are also available, we could also investigate the robustness of the model over time. This would enable the models to be used for a longer time before needing more updated survey data, and we leave this to future work. Yeh et al. ([2020](#bib.bib422)) investigated predicting the change in asset wealth for individual villages in the World Bank Living Standards Measurement Surveys (LSMS), which is a longitudinal study containing multiple samples from the same village.
PovertyMap-wilds only contains cross-sectional samples which do not provide direct supervision for changes over time at any one location, but it is still possible to consider aggregate shifts across years.

As with FMoW-wilds, there are important ethical considerations associated with remote sensing applications,
e.g., around surveillance and privacy issues, as well as the potential for systematic biases that negatively affect particular populations.
As we describe in Section [E.8.4](#S5.SS8.SSS4 "E.8.4 Additional details ‣ E.8 PovertyMap-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), noise has been added to the location metadata in PovertyMap-wilds to protect privacy.
The distribution shifts across country and urban/rural boundaries that we study in PovertyMap-wilds are an example of a bias that affects model performance and therefore could have adverse policy consequences.
We refer interested readers to the UNICEF discussion paper by Berman et al. ([2018](#bib.bib45)) for a more in-depth discussion of the ethics of remote sensing especially as it pertains to development and humanitarian endeavors.

#### E.8.3 Broader context

Computational sustainability applications in the developing world also include tracking child mortality (Burke et al., [2016](#bib.bib68); Osgood-Zimmerman et al., [2018](#bib.bib282); Reiner et al., [2018](#bib.bib310)), educational attainment (Graetz et al., [2018](#bib.bib154)), and
food security and crop yield prediction (You et al., [2017](#bib.bib423); Wang et al., [2020b](#bib.bib395); Xie et al., [2020](#bib.bib416)).
Remote sensing data and satellite imagery has the potential to enable high-resolution maps of many of these sustainability challenges, but as with poverty measures, ground truth labels in these applications come from expensive surveys or observations from human workers in the field.
Some prior works consider using spatial structure (Jean et al., [2018](#bib.bib188); Rolf et al., [2020](#bib.bib319)), unlabeled data (Xie et al., [2016](#bib.bib415); Jean et al., [2018](#bib.bib188); Xie et al., [2020](#bib.bib416)), or weak sources of supervision (Wang et al., [2020b](#bib.bib395)) to improve global models despite the lack of ground-truth data.
We hope that PovertyMap-wilds can be used to improve the robustness of machine learning techniques on satellite data, providing an avenue for cheaper and faster measurements that can be used to make progress on a general set of computational sustainability challenges.

#### E.8.4 Additional details

##### Data processing.

The PovertyMap-wilds dataset is derived from Yeh et al. ([2020](#bib.bib422)), which gathers LandSat imagery and Demographic and Health Surveys (DHS) data from 19669 villages across 23 countries in Africa .
The images are 224×224224224224\times 224 pixels large over 7 multispectral channels and an eighth nighttime light intensity channel. The LandSat satellite has a 30m resolution, meaning that each pixel of the image covers a 30​m230superscript𝑚230m^{2} spatial area.
The location metadata is perturbed by the DHS as a privacy protection scheme; urban locations are randomly displaced by up to 2km and rural locations are perturbed by up to 10km. While this adds noise to the data, having a large enough image can guarantee that the location is in the image most of the time.
The target is a real-valued composite asset wealth index computed as the first principal component of survey responses about household assets, which is thought to be a less noisy measure of households’ longer-run economic well-being than other welfare measurements like consumption expenditure (Sahn and Stifel, [2003](#bib.bib329); Filmer and Scott, [2011](#bib.bib130)).
Asset wealth also has the advantage of not requiring adjustments for inflation or for purchasing power parity (PPP), as it is not based on a currency.

We normalize each channel by the pixel-wise mean and standard deviation for each channel, following (Yeh et al., [2020](#bib.bib422)). We also do a similar data augmentation scheme, adding random horizontal and vertical flips as well as color jitter (brightness factor 0.8, contrast factor 0.8, saturation factor 0.8, hue factor 0.1).

The data download process provided by Yeh et al. ([2020](#bib.bib422)) involves downloading and processing imagery from Google Earth Engine. We process each image into a compressed NumPy array with 8 channels. We also provide all the metadata in a CSV format.

##### Additional results.

We also ran an ablation where we removed the nighttime light intensity channel.
This resulted in a drop in OOD r𝑟r of 0.04 on average and 0.06 on the rural subpopulation, demonstrating the usefulness of the nightlight data in asset wealth estimation.

##### Modifications to the original dataset.

We report a much larger drop in correlation due to spatial shift than in Yeh et al. ([2020](#bib.bib422)).
To explain this, we note that our data splitting method is slightly different from theirs.
They have two separate experiments (with different data splits) to test in-distribution vs. out-of-distribution generalization.
In contrast, our data splits on both held-out in-distribution and out-of-distribution points at the same time with respect to the same training set, thus allowing us to compare both metrics simultaneously on one model as a more direct comparison.
We use the same OOD country folds as the original dataset.
However, Yeh et al. ([2020](#bib.bib422)) split the ID train/val/test while making sure that the spatial extent of the images between each split never overlap,
while we simply take uniformly random splits of the ID data.
This means that between our ID train/val/test splits, we may have images that have share some overlapping spatial extent, for example for two very nearby locations.
Thus, a model can utilize some memorization here to improve ID performance.
We believe this is reasonable since, with more ID data, more of the spatial area will be labeled and memorization should become an increasingly viable strategy for generalization in-domain.

### E.9 Amazon-wilds

In many consumer-facing ML applications, models are trained on data collected on one set of users and then deployed across a wide range of potentially new users. These models can perform well on average but poorly on some individuals (Tatman, [2017](#bib.bib366); Caldas et al., [2018](#bib.bib72); Li et al., [2019b](#bib.bib235); Koenecke et al., [2020](#bib.bib208)).
These large performance disparities across users are practical concerns in consumer-facing applications, and they can also indicate that models are exploiting biases or spurious correlations in the data (Badgeley et al., [2019](#bib.bib24); Geva et al., [2019](#bib.bib146)).
We study this issue of inter-individual performance disparities on a variant of the Amazon-wilds Reviews dataset (Ni et al., [2019](#bib.bib274)).

#### E.9.1 Setup

##### Problem setting.

We consider a hybrid domain generalization and subpopulation problem where the domains correspond to different reviewers.
The task is multi-class sentiment classification, where the input x𝑥x is the text of a review, the label y𝑦y is a corresponding star rating from 1 to 5, and the domain d𝑑d is the identifier of the reviewer who wrote the review.
Our goal is to perform consistently well across a wide range of reviewers, i.e., to achieve high tail performance on different subpopulations of reviewers in addition to high average performance. In addition, we consider disjoint set of reviewers between training and test time.

##### Data.

The dataset comprises 539,502 customer reviews on Amazon taken from the Amazon Reviews dataset (Ni et al., [2019](#bib.bib274)).
Each input example has a maximum token length of 512.
For each example, the following additional metadata is also available at both training and evaluation time: reviewer ID, product ID, product category, review time, and summary.

To reliably measure model performance on each reviewer, we include at least 75 reviews per reviewer in each split.
Concretely, we consider the following splits, where reviewers are randomly assigned to either in-distribution or out-of-distribution sets:

1. 1.

   Training: 245,502 reviews from 1,252 reviewers.
2. 2.

   Validation (OOD): 100,050 reviews from another set of 1,334 reviewers, distinct from training and test (OOD).
3. 3.

   Test (OOD): 100,050 reviews from another set of 1,334 reviewers, distinct from training and validation (OOD).
4. 4.

   Validation (ID): 46,950 reviews from 626 of the 1,252 reviewers in the training set.
5. 5.

   Test (ID): 46,950 reviews from 626 of the 1,252 reviewers in the training set.

The reviewers in the train and in-distribution splits; the validation (OOD) split; and the test (OOD) split are all disjoint,
which allows us to test generalization to unseen reviewers.
See Appendix [E.9.4](#S5.SS9.SSS4 "E.9.4 Additional details ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") for more details.

##### Evaluation.

To assess whether models perform consistently well across reviewers, we evaluate models by their accuracy on the reviewer at the 10th percentile.
This follows the federated learning literature, where it is standard to measure model performance on devices and users at various percentiles in an effort to encourage good performance across many devices (Caldas et al., [2018](#bib.bib72); Li et al., [2019b](#bib.bib235)).

##### Potential leverage.

We include more than a thousand reviewers in the training set, capturing variation across a wide range of reviewers.
In addition, we provide reviewer ID annotations for all reviews in the dataset.
These annotations could be used to directly mitigate performance disparities across reviewers seen during training time.

#### E.9.2 Baseline results

Table 23: Baseline results on Amazon-wilds. We report the accuracy of models trained using ERM, CORAL, IRM, and group DRO, as well as a reweighting baseline that reweights for class balance. To measure tail performance across reviewers, we report the accuracy for the reviewer in the 10th percentile. While the performance drop on Amazon-wilds is primarily from subpopulation shift, there is also a performance drop from evaluating on unseen reviewers, as evident in the gaps in accuracies between the in-distribution and the out-of-distribution sets.

Validation (ID)
Validation (OOD)
Test (ID)
Test (OOD)

Algorithm
10th percentile
Average
10th percentile
Average
10th percentile
Average
10th percentile
Average

ERM
58.7 (0.0)
75.7 (0.2)
55.2 (0.7)
72.7 (0.1)
57.3 (0.0)
74.7 (0.1)
53.8 (0.8)
71.9 (0.1)

CORAL
56.2 (1.7)
74.4 (0.3)
54.7 (0.0)
72.0 (0.3)
55.1 (0.4)
73.4 (0.2)
52.9 (0.8)
71.1 (0.3)

IRM
56.4 (0.8)
74.3 (0.1)
54.2 (0.8)
71.5 (0.3)
54.7 (0.0)
72.9 (0.2)
52.4 (0.8)
70.5 (0.3)

Group DRO
57.8 (0.8)
73.7 (0.6)
54.7 (0.0)
70.7 (0.6)
55.8 (1.0)
72.5 (0.3)
53.3 (0.0)
70.0 (0.5)

Reweighted (label)
55.1 (0.8)
71.9 (0.4)
52.1 (0.2)
69.1 (0.5)
54.4 (0.4)
70.7 (0.4)
52.0 (0.0)
68.6 (0.6)

![Refer to caption](/html/2012.07421/assets/x25.png)


Figure 27: 
Distribution of per-reviewer accuracy on the test set for the ERM model (blue). The corresponding random baseline would have per-reviewer accuracy distribution in grey.

##### Model.

For all experiments, we finetuned DistilBERT-base-uncased models (Sanh et al., [2019](#bib.bib330)), using the implementation from Wolf et al. ([2019](#bib.bib405)), and with the following hyperparameter settings:
batch size 8; learning rate 1×10−51superscript1051\times 10^{-5} with the AdamW optimizer (Loshchilov and Hutter, [2019](#bib.bib247)); L2subscript𝐿2L\_{2}-regularization strength 0.010.010.01; 3 epochs with early stopping; and a maximum number of tokens of 512.
We selected the above hyperparameters based on a grid search over learning rates {1×10−6,2×10−6,1×10−5,2×10−5}1superscript1062superscript1061superscript1052superscript105\{1\times 10^{-6},2\times 10^{-6},1\times 10^{-5},2\times 10^{-5}\}, and all other hyperparameters were simply set to standard/default values.

##### ERM results and performance drops.

A DistilBERT-base-uncased model trained with the standard ERM objective performs well on average, but performance varies widely across reviewers (Figure [27](#S5.F27 "Figure 27 ‣ E.9.2 Baseline results ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), Table [23](#S5.T23 "Table 23 ‣ E.9.2 Baseline results ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
Despite the high average accuracy of 71.9%, per-reviewer accuracies vary widely between 100.0% and 12.0%, with accuracy at the 10th percentile of 53.8%.
The above variation is larger than expected from randomness: a random binomial baseline with equal average accuracy would have a 10th percentile accuracy of 65.4%.
We observe low tail performance on both previously seen and unseen reviewers, with low 10th percentile accuracy on in-distribution and out-of-distribution sets (Table [23](#S5.T23 "Table 23 ‣ E.9.2 Baseline results ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
In addition, we observe drops on both average and 10th percentile accuracies upon evaluating on unseen reviewers, as evident in the performance gaps between the in-distribution and the out-of-distribution sets.

As with CivilComments-wilds, the relatively small number of reviews per reviewer makes it difficult to run a test-to-test comparison (e.g., training a model on just the reviewers in the bottom 10th percentile).
Without running the test-to-test comparison, it is possible that the gap between average and 10th percentile accuracies can be explained at least in part by differences in the intrinsic difficulty of reviews from different reviewers,
e.g., some reviewers might not write text reviews that are informative of their star rating.
Future work will be required to establish in-distribution accuracies that account for these differences.

##### Additional baseline methods.

We now consider models trained by existing robust training algorithms and show that these models also perform poorly on tail reviewers, failing to mitigate the performance drop (Table [23](#S5.T23 "Table 23 ‣ E.9.2 Baseline results ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
We observe that reweighting to achieve uniform class balance fails to improve the 10th percentile accuracy, showing that variation across users cannot be solved simply by accounting for label imbalance.
In addition, CORAL, IRM, and Group DRO fail to improve both average and 10th percentile accuracies on both ID and OOD sets.
Our grid search selected λ=1.0𝜆1.0\lambda=1.0 for the CORAL penalty and λ=1.0𝜆1.0\lambda=1.0 for the IRM penalty.

##### Discussion.

The distribution shift and the evaluation criteria for Amazon-wilds focus on the tail performance, unlike the other datasets in Wilds.
Because of this, Amazon-wilds might have distinct empirical trends or be conducive to different algorithms compared to other datasets.
Potential approaches include extensions to algorithms for worst-group performance, for example to handle a large number of groups, as well as adaptive approaches that yield user-specific predictions.

#### E.9.3 Broader context

Performance disparities across individuals have been observed in a wide range of tasks and applications, including in natural language processing (Geva et al., [2019](#bib.bib146)),
automatic speech recognition (Koenecke et al., [2020](#bib.bib208); Tatman, [2017](#bib.bib366)), federated learning (Li et al., [2019b](#bib.bib235); Caldas et al., [2018](#bib.bib72)), and medical imaging (Badgeley et al., [2019](#bib.bib24)).
These performance gaps are practical limitations in applications that call for good performance across a wide range of users,
including many user-facing applications such as speech recognition (Koenecke et al., [2020](#bib.bib208); Tatman, [2017](#bib.bib366)) and personalized recommender systems (Patro et al., [2020](#bib.bib288)),
tools used for analysis of individuals such as sentiment classification in computational social science (West et al., [2014](#bib.bib401)) and user analytics (Lau et al., [2014](#bib.bib225)),
and applications in federated learning.
These performance disparities have also been studied in the context of algorithmic fairness,
including in the federated learning literature,
in which uniform performance across individuals is cast as a goal toward fairness (Li et al., [2019b](#bib.bib235); Dwork et al., [2012](#bib.bib120)).
Lastly, these performance disparities can also highlight models’ failures to learn the actual task in a generalizable manner;
instead, some models have been shown learn the biases specific to individuals.
Prior work has shown that individuals—technicians for medical imaging in this case—can not only be identified from data, but also are predictive of the diagnosis, highlighting the risk of learning to classify technicians rather than the medical condition (Badgeley et al., [2019](#bib.bib24)).
More directly,
across a few natural language processing tasks where examples are annotated by crowdworkers, models have been observed to perform well on annotators that are commonly seen at training time, but fail to generalize to unseen annotators,
suggesting that models are merely learning annotator-specific patterns and not the task (Geva et al., [2019](#bib.bib146)).

#### E.9.4 Additional details

Table 24: Dataset details for Amazon-wilds.

|  |  |  |  |
| --- | --- | --- | --- |
| Split | # Reviews | # Reviewers | # Reviews per reviewer (mean / minimum) |
| Training | 245,502 | 1,252 | 196 / 75 |
| Validation (OOD) | 100,050 | 1,334 | 75 / 75 |
| Test (OOD) | 46,950 | 662 | 75 / 75 |
| Validation (ID) | 100,050 | 1,334 | 75 / 75 |
| Test (ID) | 46,950 | 662 | 75 / 75 |

##### Data processing.

We consider a modified version of the Amazon reviews dataset (Ni et al., [2019](#bib.bib274)).
We consider disjoint reviewers between the training, OOD validation, and OOD test sets, and we also provide separate ID validation and test sets that include reviewers seen during training for additional reporting.
These reviewers are selected uniformly at random from the reviewer pool, with the constraint that they have at least 150 reviews in the pre-processed dataset.
Statistics for each split are described in Table [24](#S5.T24 "Table 24 ‣ E.9.4 Additional details ‣ E.9 Amazon-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Notably, each reviewer has at least 75 reviews in the training set and exactly 75 reviews in the validation and test sets.

To process the data, we first eliminate reviews that are longer than 512 tokens, reviews without any text, and any duplicate reviews with identical star rating, reviewer ID, product ID, and time.
We then obtain the 30-core subset of the reviews, which contains the maximal set of reviewers and products such that each reviewer and product has at least 30 reviews; this is a standard preprocessing procedure used in the original dataset (Ni et al., [2019](#bib.bib274)).
To construct the dataset for reviewer shifts in particular, we further eliminate the following reviews:
(i) reviews that contain HTML,
(ii) reviews with identical text within a user in order to ensure sufficiently high effective sample size per reviewer,
and (iii) reviews with identical text across users to eliminate generic reviews.
Once we have the filtered set of reviews, we consider reviewers with at least 150 reviews and sample uniformly at random until the training set contains approximately 250,000 reviews and each evaluation set contains at least 100,000 reviews.
As we construct the training set, we reserve a random sample of 75 reviews for each user for evaluation and put all other reviews in the training set.
For the evaluation set, we put a random sample of 75 reviews for each user.

##### Modifications to the original dataset.

The original dataset does not prescribe a specific task or split.
We consider a standard task of sentiment classification, but instead of using a standard i.i.d. split, we instead consider disjoint users between training and evaluation time as described above.
In addition, we preprocess the data as detailed above.

### E.10 Py150-wilds

Code completion models—autocomplete tools used by programmers to suggest subsequent source code tokens, such as the names of API calls—are commonly used to reduce the effort of software development (Robbes and Lanza, [2008](#bib.bib317); Bruch et al., [2009](#bib.bib62); Nguyen and Nguyen, [2015](#bib.bib273); Proksch et al., [2015](#bib.bib300); Franks et al., [2015](#bib.bib132)).
These models are typically trained on data collected from existing codebases but then deployed more generally across other codebases, which may have different distributions of API usages (Nita and Notkin, [2010](#bib.bib275); Proksch et al., [2016](#bib.bib301); Allamanis and Brockschmidt, [2017](#bib.bib10)). This shift across codebases can cause substantial performance drops in code completion models.
Moreover, prior studies of real-world usage of code completion models have noted that these models can generalize poorly on some important subpopulations of tokens such as method names (Hellendoorn et al., [2019](#bib.bib170)).

We study this problem using a variant of the Py150 Dataset, originally developed by Raychev et al. ([2016](#bib.bib306)) and adapted to a code completion task by Lu et al. ([2021](#bib.bib248)).

#### E.10.1 Setup

##### Problem setting.

We consider a hybrid domain generalization and subpopulation shift problem, where the domains are codebases (GitHub repositories), and our goal is to learn code completion models that generalize to source code written in new codebases.
Concretely, the input x𝑥x is a sequence of source code tokens taken from a single file, the label y𝑦y is the next token (e.g., "environ", "communicate" in Figure [12](#S4.F12 "Figure 12 ‣ 4.3.4 Py150-wilds: Code completion across different codebases ‣ 4.3 Hybrid datasets ‣ 4 Wilds datasets ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), and the domain d𝑑d is an integer that identifies the repository that the source code belongs to.
We aim to solve both a domain generalization problem across codebases and improve subpopulation performance on class and methods tokens.

##### Data.

The dataset comprises 150,000 Python source code files from 8,421 different repositories on GitHub ([github.com](https://github.com/)). Each source code file is associated with the repository ID so that code from the same repository can be linked.

We split the dataset by randomly partitioning the data by repositories:

1. 1.

   Training: 79,866 code files from 5,477 repositories.
2. 2.

   Validation (OOD): 5,160 code files from different 261 repositories.
3. 3.

   Test (OOD): 39,974 code files from different 2,471 repositories.
4. 4.

   Validation (ID): 5,000 code files from the same repositories as the training set (but different files).
5. 5.

   Test (ID): 20,000 code files from the same repositories as the training set (but different files).

The repositories are randomly distributed across the training, validation (OOD), and test (OOD) sets.
As we use models pre-trained on the CodeSearchNet dataset (Husain et al., [2019](#bib.bib185)), which partially overlaps with the Py150 dataset, we ensured that all GitHub repositories used in CodeSearchNet only appear in the training set in Py150-wilds and not in the validation/test sets.

Table [25](#S5.T25 "Table 25 ‣ Data. ‣ E.10.1 Setup ‣ E.10 Py150-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows the token statistics of the source code files, as well as the token type breakdown (e.g., class, method, punctuator, keyword, literal).
The tokens are defined by the built-in Python tokenizer and the CodeGPT tokenizer, following Lu et al. ([2021](#bib.bib248)).
Training and evaluation are conducted at the token-level (more details are provided below).

Table 25: Token statistics for Py150-wilds.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Split | #Files | #Total tokens | #Class | #Method | #Punctuator | #Keyword | #Literal |
| Training | 79,866 | 14,129,619 | 894,753 | 789,456 | 4,512,143 | 1,246,624 | 1,649,653 |
| Validation (ID) | 5,000 | 882,745 | 55,645 | 48,866 | 282,568 | 77,230 | 105,456 |
| Test (ID) | 20,000 | 3,539,524 | 222,822 | 194,293 | 1,130,607 | 313,008 | 420,232 |
| Validation (OOD) | 5,160 | 986,638 | 65,237 | 56,756 | 310,914 | 84,677 | 111,282 |
| Test (OOD) | 39,974 | 7,340,433 | 444,713 | 412,700 | 2,388,151 | 640,939 | 869,083 |

##### Evaluation.

We evaluate models by their accuracy on predicting class and method tokens in the test set code files. This subpopulation metric is inspired by Hellendoorn et al. ([2019](#bib.bib170)), which finds that in real-world settings, developers primarily use code completion tools for completing class names and method names;
in contrast, measuring average token accuracy would prioritize common tokens such as punctuators, which are often not a problem in real-world settings.

##### Potential leverage.

We provide the GitHub repository that each source code files was derived from, which training algorithms can leverage.
As programming tools like code completion are expected to be used across codebases in real applications (Nita and Notkin, [2010](#bib.bib275); Allamanis and Brockschmidt, [2017](#bib.bib10)), it is important for models to learn generalizable representations of code and extrapolate well on unseen codebases.
We hope that approaches using the provided repository annotations can learn to factor out common features and codebase-specific features, resulting in more robust models.

Additionally, besides the (integer) IDs of repositories, we also provide the repository names and file names in natural language as extra metadata. While we only use the repository IDs in our baseline experiments described below, the extra natural language annotations can potentially be leveraged as well to adapt models to target repositories/files.

#### E.10.2 Baseline results

##### Model.

For all experiments, we use the CodeGPT model (Lu et al., [2021](#bib.bib248)) pre-trained on CodeSearchNet (Husain et al., [2019](#bib.bib185)) as our model and finetune it on Py150-wilds, using all the tokens in the training set.
We tokenize input source code by the CodeGPT tokenizer and take blocks of length 256 tokens.
We then train the CodeGPT model with a batch size of 6 (with 6×256=1,5366256

15366\times 256=1,536 tokens), a learning rate of 8×10−58superscript1058\times 10^{-5}, no L2subscript𝐿2L\_{2} regularization, and the AdamW optimizer (Loshchilov and Hutter, [2019](#bib.bib247)) for 3 epochs with early stopping.
Using the hyperparameters from Lu et al. ([2021](#bib.bib248)) as a starting point, we selected the above hyperparameters by a grid search over learning rates {8×10−4,8×10−5,8×10−6}8superscript1048superscript1058superscript106\{8\times 10^{-4},8\times 10^{-5},8\times 10^{-6}\} and L2subscript𝐿2L\_{2} regularization strength {0,0.01,0.1}00.010.1\{0,0.01,0.1\}. All other hyperparameters were simply set to standard/default values.

##### ERM results and performance drops.

Table [26](#S5.T26 "Table 26 ‣ Additional baseline methods. ‣ E.10.2 Baseline results ‣ E.10 Py150-wilds ‣ E Additional dataset details and results ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows that model performance on class and method tokens dropped substantially from 75.4% on the train-to-train in-distribution repositories in the Test (ID) set to 67.9% on the out-of-distribution repositories in the Test (OOD) set.
This gap shrinks if we evaluate the model on all tokens (instead of class and method tokens): accuracy drops from 74.5% on Test (ID) to 69.6% on Test (OOD). This is because the evaluation across all tokens includes many tokens that are used universally across repositories, such as punctuators and keywords.

We only ran a train-to-train comparison because there are a relatively large number of domains (repositories) split i.i.d. between the training and test sets, which suggests that the training and test sets should be “equally difficult”.
We therefore do not expect test-to-test and mixed-to-test comparisons to yield significantly different results.

##### Additional baseline methods.

We trained CORAL, IRM, and Group DRO baselines, treating each repository as a domain. For CORAL and IRM, we find that the smaller penalties give slightly better generalization performance (λ=1𝜆1\lambda=1 for CORAL and λ=1𝜆1\lambda=1 for IRM).
Compared to the ERM baseline, while CORAL and IRM reduced the performance gap between ID and OOD, neither of them improved upon ERM on the final OOD performance.

Table 26: Baseline results on Py150-wilds. We report both the model’s accuracy on predicting class and method tokens and accuracy on all tokens trained using ERM, CORAL, IRM and group DRO. Standard deviations over 3 trials are in parentheses.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Validation (ID) | | Validation (OOD) | | Test (ID) | | Test (OOD) | |
| Algorithm | Method/class | All | Method/class | All | Method/class | All | Method/class | All |
| ERM | 75.5 (0.5) | 74.6 (0.4) | 68.0 (0.1) | 69.4 (0.1) | 75.4 (0.4) | 74.5 (0.4) | 67.9 (0.1) | 69.6 (0.1) |
| CORAL | 70.7 (0.0) | 70.9 (0.1) | 65.7 (0.2) | 67.2 (0.1) | 70.6 (0.0) | 70.8 (0.1) | 65.9 (0.1) | 67.9 (0.0) |
| IRM | 67.3 (1.1) | 68.4 (0.7) | 63.9 (0.3) | 65.6 (0.1) | 67.3(1.1) | 68.3 (0.7) | 64.3 (0.2) | 66.4 (0.1) |
| Group DRO | 70.8 (0.0) | 71.2 (0.1) | 65.4 (0.0) | 67.3 (0.0) | 70.8 (0.0) | 71.0 (0.0) | 65.9 (0.1) | 67.9 (0.0) |

#### E.10.3 Broader context

Machine learning can aid programming and software engineering in various ways:
automatic code completion (Raychev et al., [2014](#bib.bib305); Svyatkovskiy et al., [2019](#bib.bib360)),
program synthesis (Bunel et al., [2018](#bib.bib66); Kulal et al., [2019](#bib.bib212)),
program repair (Vasic et al., [2019](#bib.bib383); Yasunaga and Liang, [2020](#bib.bib421)),
code search (Husain et al., [2019](#bib.bib185)),
and code summarization (Allamanis et al., [2015](#bib.bib11)).
However, these systems face several forms of distribution shifts when deployed in practice.
One major challenge is the shifts across codebases (which our Py150-wilds dataset focuses on), where systems need to adapt to factors such as project content, coding conventions, or library or API usage in each codebase (Nita and Notkin, [2010](#bib.bib275); Allamanis and Brockschmidt, [2017](#bib.bib10)).
A second source of shifts is programming languages, which includes adaptation across different domain-specific languages (DSLs), e.g., in robotic environments (Shin et al., [2019](#bib.bib344));
and across different versions of languages, e.g., Python 2 and 3 (Malloy and Power, [2017](#bib.bib255)).
Another challenge is the shift from synthetic training sets to real usage: for instance, Hellendoorn et al. ([2019](#bib.bib170)) show that existing code completion systems, which are typically trained as language models on source code, perform poorly on the real completion instances that are most commonly used by developers in IDEs, such as API calls (class and method calls).

#### E.10.4 Additional details

##### Data split.

We generate the splits in the following steps. First, to avoid test set contamination, we took all of the repositories in CodeSearchNet (which, as a reminder, is used to pretrain our baseline model) and assigned them to the training set. Second, we randomly split all of the remaining repositories into three groups: Validation (OOD), Test (OOD), and Others. Finally, to generate the ID splits, we randomly split the files in the Others repositories into three sets: Training, Validation (ID), and Test (ID).

##### Modifications to the original dataset.

The original Py150 dataset (Raychev et al., [2016](#bib.bib306)) splits the total 150k files into 100k training files and 50k test files,
regardless of the repository that each file was from.
In Py150-wilds, we re-split the dataset based on repositories to construct the aforementioned train, validation (ID), validation (OOD), test (ID), and test (OOD) sets.

Additionally, in the Py150 code completion task introduced in Lu et al. ([2021](#bib.bib248)), models are evaluated by the accuracy of predicting every token in source code. However, according to developer studies, this evaluation may include various tokens that are rarely used in real code completion, such as punctuators, strings, numerals, etc. (Robbes and Lanza, [2008](#bib.bib317); Proksch et al., [2016](#bib.bib301); Hellendoorn et al., [2019](#bib.bib170)).
To define a task closer to real applications, in Py150-wilds we focus on class name and method name prediction (which are used most commonly by developers).

## F Datasets with distribution shifts that do not cause performance drops

### F.1 SQF: Criminal possession of weapons across race and locations

In this section, we provide more details on the stop-and-frisk dataset discussed in Section [8.1](#S8.SS1 "8.1 Algorithmic fairness ‣ 8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
The original data was provided by the New York City Police Department, and has been widely used in previous ML and data analysis work (Goel et al., [2016](#bib.bib150); Zafar et al., [2017](#bib.bib426); Pierson et al., [2018](#bib.bib296); Kallus and Zhou, [2018](#bib.bib197); Srivastava et al., [2020](#bib.bib351)).
For our analysis, we use the version of the dataset that was processed by Goel et al. ([2016](#bib.bib150)). Our problem setting and dataset structure closely follow theirs.

#### F.1.1 Setup

##### Problem setting.

We study a subpopulation shift in a weapons prediction task, where each data point corresponds to a pedestrian who was stopped by the police on suspicion of criminal possession of a weapon. The input x𝑥x is a vector that represents 29 observable features from the UF-250 stop-and-frisk form filled out by the officer after each stop: e.g., whether the stop was initiated based on a radio run or at an officer’s discretion, whether the officer was uniformed, and any reasons the officer gave for the stop (encoded as a categorical variable). Importantly, these features can all be observed by the officer prior to making the stop.999When we consider subpopulation shifts over race groups, the input x𝑥x additionally includes 75 one-hot indicators corresponding to the precinct that the stop was made in.
We do not include those features when we consider shifts over locations,
as they prevent the model from generalizing to new locations.
The binary label y𝑦y is whether the pedestrian in fact possessed a weapon (i.e., whether the stop fulfilled its stated purpose).
We consider, separately, two types of domains d𝑑d: 1) race groups and 2) locations (boroughs in New York City). We consider location and race as our domains because previous work has shown that they can produce substantial disparities in policing practices and in algorithmic performance (Goel et al., [2016](#bib.bib150)).

##### Data.

Each row of the dataset represents one stop of one pedestrian. Following Goel et al. ([2016](#bib.bib150)), we filter for the 621,696 stops where the reason for the stop is suspicion of criminal possession of a weapon. We then filter for rows with complete data for observable features; with stopped pedestrians who are Black, white, or Hispanic; and who are stopped during the years 2009-2012 (the time range used in Goel et al. ([2016](#bib.bib150))). These filters yield a total of 506,283 stops,
3.5% of which are positive examples (in which the officer finds that the pedestrian is illegally possessing a weapon).

The training versus validation split is a random 80%-20% partition of all stops in 2009 and 2010. We test on stops from 2011-2012; this follows the experimental setup in Goel et al. ([2016](#bib.bib150)). Overall, our data splits are as follows:

1. 1.

   Training: 241,964 stops from 2009 and 2010.
2. 2.

   Validation: 60,492 stops from 2009 and 2010, disjoint from the training set.
3. 3.

   Test: 289,863 stops from 2011 and 2012.

In the experiments below, we do not use the entire training set,
as we observed in our initial experiments that the model performed less well on certain subgroups (Black pedestrians and pedestrians from the Bronx).
To determine whether this inferior performance might be ameliorated by training specifically on those groups, we controlled for training set size by downsampling the training set to the size of the disadvantaged population of interest for a given split.
Specifically, we consider the following (overlapping) training subsets, each of which is subsampled from the overall training set described above:

1. 1.

   Black pedestrians only: 155,929 stops of Black pedestrians from 2009 and 2010.
2. 2.

   All pedestrians, subsampled to # Black pedestrians: 155,929 stops of all pedestrians from 2009 and 2010.
3. 3.

   Bronx pedestrians only: 69,129 stops of pedestrians in the Bronx from 2009 and 2010.
4. 4.

   All pedestrians, subsampled to # Bronx pedestrians: 69,129 stops of all pedestrians from 2009 and 2010.

These amount to running a test-to-test comparison for the subpopulations of Black pedestrians and Bronx pedestrians.

##### Evaluation.

Our metric for classifier performance is the precision for each race group and each borough at a global recall
of 60%—i.e., when using a threshold which recovers 60% of all weapons in the test data, similar to the recall evaluated in Goel et al. ([2016](#bib.bib150)).
The results are similar when using different recall thresholds.
Examining the precision for each race/borough captures the fact, discussed in Goel et al. ([2016](#bib.bib150)), that very low-precision stops may violate the Fourth Amendment, which requires *reasonable suspicion* for conducting a police stop; thus, the metric encapsulates the intuition that the police are attempting to avoid Fourth Amendment violations for any race group or borough while still recovering a substantial fraction of the illegal weapons.

#### F.1.2 Baseline results

##### Model.

For all experiments, we use a logistic regression model trained with the Adam optimizer (Kingma and Ba, [2015](#bib.bib207)) and early stopping.
We trained one model on each of the 4 training sets, separately picking hyperparameters through a grid search across 7 learning rates logarithmically-spaced in [5×10−8,5×10−2]5superscript1085superscript102[5\times 10^{-8},5\times 10^{-2}] and batch sizes in {4,8,16,32,64}48163264\{4,8,16,32,64\}.
Table [29](#S6.T29 "Table 29 ‣ ERM results and performance drops. ‣ F.1.2 Baseline results ‣ F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") provides the hyperparameters used for each training set.
All models were trained with a reweighted cross-entropy objective that upsampled the positive examples to achieve class balance.

##### ERM results and performance drops.

Performance differed substantially across race and location groups: precision was lowest on Black pedestrians (Table [27](#S6.T27 "Table 27 ‣ ERM results and performance drops. ‣ F.1.2 Baseline results ‣ F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), top row) and pedestrians in the Bronx (Table [28](#S6.T28 "Table 28 ‣ ERM results and performance drops. ‣ F.1.2 Baseline results ‣ F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), top row).
To assess whether in-distribution training would improve performance on these groups, we trained the model only on Black pedestrians (Table [27](#S6.T27 "Table 27 ‣ ERM results and performance drops. ‣ F.1.2 Baseline results ‣ F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), bottom row) and pedestrians in the Bronx (Table [28](#S6.T28 "Table 28 ‣ ERM results and performance drops. ‣ F.1.2 Baseline results ‣ F.1 SQF: Criminal possession of weapons across race and locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), bottom row).
However, this did not substantially improve performance on Black pedestrians or pedestrians from the Bronx; the difference in precision was less than 0.005 for both groups relative to the original model trained on all races and locations.
This is consistent with the fact that groups with the lowest performance are not necessarily small minorities of the dataset: for example, more than 90% of the stops are of Black or Hispanic pedestrians, but performance on these groups is worse than that for white pedestrians.
The lack of improvement from in-distribution training suggests that approaches like group DRO would be unlikely to further improve performance, and we thus did not assess these approaches.

Table 27: Comparison of precision scores for each race group at 60% global weapon recall. Train set size is 69,129 for both rows.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Precision at 60% recall | | |
| Training dataset | Black | Hispanic | White |
| Black pedestrians only | 0.131 | 0.174 | 0.360 |
| All pedestrians, subsampled to # Black pedestrians | 0.135 | 0.183 | 0.362 |




Table 28: Comparison of precision scores for each borough at a threshold which achieves 60% global weapon recall. Train set size is 155,929 for both rows.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Precision at 60% recall | | | | |
| Training dataset | Bronx | Brooklyn | Manhattan | Queens | Staten Island |
| Bronx pedestrians only | 0.074 | 0.158 | 0.207 | 0.157 | 0.105 |
| All pedestrians, subsampled to # Bronx pedestrians | 0.075 | 0.162 | 0.224 | 0.168 | 0.107 |




Table 29: Model parameters used in this analysis.

|  |  |  |  |
| --- | --- | --- | --- |
| Training data | Batch size | Learning rate | Number of epochs |
| Black pedestrians only | 4 | 5×10−45superscript1045\times 10^{-4} | 1 |
| All pedestrians, subsampled to # Black pedestrians | 4 | 5×10−45superscript1045\times 10^{-4} | 4 |
| Bronx pedestrians only | 4 | 5×10−45superscript1045\times 10^{-4} | 2 |
| All pedestrians, subsampled to # Bronx pedestrians | 4 | 5×10−35superscript1035\times 10^{-3} | 4 |

##### Discussion.

We observed large disparities in performance across race and location groups. However, the fact that test-to-test in-distribution training did not ameliorate these disparities suggests that they do not occur because some groups comprise small minorities of the original dataset, and thus suffer worse performance. Instead, our results suggest that classification performance on some race and location groups are intrinsically noisier; it is possible, for example, that collection of additional features would be necessary to improve performance on these groups (Chen et al., [2018](#bib.bib80)).

#### F.1.3 Additional details

##### Modifications to the original dataset.

The features we use are very similar to those used in Goel et al. ([2016](#bib.bib150)). The two primary differences are that 1) we remove features which convey information about a stopped pedestrian’s race, since those might be illegal to use in real-world policing contexts and 2) we do not include a “local hit rate” feature which captures the fraction of historical stops in the vicinity of a stop which resulted in discovery of a weapon; we omit this latter feature because it was unnecessary to match performance in Goel et al. ([2016](#bib.bib150)).
test-to-test

### F.2 ENCODE: Transcription factor binding across different cell types

Here we provide details on the transcription factor binding dataset discussed in Section [8.3](#S8.SS3 "8.3 Genomics ‣ 8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Transcription factors (TFs) are regulatory proteins that bind specific DNA elements in the genome to activate or repress transcription of target genes.
There are estimated to be approximately 1,600 human TFs, and the binding landscape of each TF can be highly variable across different cell types (Deplancke et al., [2016](#bib.bib108)).
Understanding how these binding patterns change across different cell types and affect cellular function is critical for understanding the mechanics of dynamic gene regulation across cell types and across healthy and diseased cell states.

Several experimental strategies have been developed to profile genome-wide binding landscapes of individual TFs in specific cell types of interest.
However, genome-wide profiling of TF binding is challenging in practice, as it requires large numbers of cells and reagents (e.g., high-affinity antibodies) that are difficult and expensive to acquire.
Moreover, profiling each individual TF requires a separate experiment, so it can be prohibitively costly to map out even a few different TFs out of the >1000 in the human genome.
Therefore, there has been wide interest in computational approaches that can predict the genome-wide binding maps of multiple TFs in new cell types from a single and more practical genome-wide assay.

DNA sequence is one of the principal determinants of where a TF binds along the genome,101010Most TFs, including the ones we provide in this benchmark, have DNA-binding domains which bind to sequence motifs: short recognition sequences (4-20 bases in length) in the genome with specific binding affinity distributions (Stormo and Zhao, [2010](#bib.bib353)).
and many ML models have been developed to predict TF binding as a function of DNA sequence in a particular cell type (Alipanahi et al., [2015](#bib.bib9); Quang and Xie, [2019](#bib.bib303); Avsec et al., [2021b](#bib.bib21)).
However, even when the DNA sequence is invariant across different cell types (e.g., among cell types from the same organism), the TF binding landscape can still be highly variable (Deplancke et al., [2016](#bib.bib108)).
Therefore, TF binding models that only use sequence inputs cannot make different predictions for the same sequence across different cell types;
we also need complementary, cell-type-specific inputs to model changes in binding over different cell types.

In this section, we explore the use of genome-wide chromatin accessibility assays such as DNase-seq and ATAC-seq (Boyle et al., [2008](#bib.bib58); Thurman et al., [2012](#bib.bib373); Buenrostro et al., [2013](#bib.bib64)), in conjunction with DNA sequence, to predict TF binding.
DNA is typically accessible in a highly local and cell-type-specific manner,
and in particular, genomic sequences with high accessibility are typically bound by one or more TFs, although the identity of the TF is not directly measured by the experiment (Lee et al., [2004](#bib.bib227)).
By measuring chromatin accessibility at each base in the genome in a specific cell type of interest, we can obtain a cell-type-specific profile of binding locations; moreover, these experiments are often cheaper than profiling even a single TF (Minnoye et al., [2021](#bib.bib266)).
Our goal is to use this accessibility signal, combined with DNA sequence, to accurately predict the binding patterns of multiple TFs in new cell types.

We study the problem of predicting genome-wide TF binding across different cell types using data from the ENCODE-DREAM Transcription Factor Binding Site Prediction Challenge (Balsubramani et al., [2020](#bib.bib27)).

#### F.2.1 Setup

##### Problem setting.

We consider the domain generalization setting, where the domains are cell types, and we seek to learn models that can generalize to cell types that are not in the training set.
The task is to predict if a particular transcription factor (TF) would bind to a particular genomic location in a cell type of interest (Figure [28](#S6.F28 "Figure 28 ‣ Problem setting. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The input is DNA sequence (which we assume to be shared across all cell types) and a cell-type-specific biochemical measurement of chromatin accessibility obtained through the DNase-seq assay.

Concretely, we segment the genome into uniformly-sized, overlapping bins that are 200 base pairs (bp) in length, and tiled 50bp apart. Given a TF p𝑝p, each genomic bin i𝑖i in cell type d𝑑d has a binding status yi,dp∈{0,1}subscriptsuperscript𝑦𝑝

𝑖𝑑01y^{p}\_{i,d}\in\{0,1\}.
Our goal is to predict each bin’s binding status as a function of the local DNA sequence Sisubscript𝑆𝑖S\_{i} and the local cell-type specific accessibility profile Ai,dsubscript𝐴

𝑖𝑑A\_{i,d} (Figure [28](#S6.F28 "Figure 28 ‣ Problem setting. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")). We treat each TF separately, i.e., for each p𝑝p, we have separate training and test sets and separate models.

For computational convenience, we consider input examples x∈ℝ12800×5𝑥superscriptℝ128005x\in\mathbb{R}^{12800\times 5} that each represent a window of 12800 bp and span several genomic bins. The first four columns of x𝑥x is a binary matrix representing a one-hot encoding of the four bases (A,C,G,T) of the DNA sequence at that base pair. The 5th column is a real-valued vector representing chromatin accessibility.
We tile the central 6400 bp of the window with 128 overlapping bins of length 200 bp, tiled 50 bp apart.111111This ensures that each bin has at least 3200 bp of context on either side of it for prediction.
Thus, each x𝑥x is associated with a target output y∈{−1,0,1}128𝑦superscript101128y\in\{-1,0,1\}^{128} indicating the binding status of the TF at each of these 128 bins. The three possible values at each bin indicate whether the bin is bound, unbound, or ambiguous.
During training and testing, we simply ignore ambiguous bins, and only focus on bound or unbound bins.
The domain d𝑑d is an integer that identifies the cell type.

![Refer to caption](/html/2012.07421/assets/figures/Wilds_ENCODE-DREAM-schematic.png)


Figure 28: 
Setup of the ENCODE benchmark. (a) The predictive model predicts binding of a protein to a location, in a cell type (domain). (b) The input features are DNA sequence and chromatin accessibility, and the labels are assigned over 200 base pair (bp) bins tiling the genome every 50 bp. (c) Each training example is a 12800 bp window, with the 128 middle bins evaluated (spanning 6400 bp).

##### Data.

The dataset comprises (a) genome-wide sequence; (b) TF binding maps for two TFs, JUND and MAX, across a total of 6 and 8 cell types respectively; and (c) an accessibility profile for each cell type.
As described above and illustrated in Figure [28](#S6.F28 "Figure 28 ‣ Problem setting. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we break up these genome-wide data into overlapping 12800 bp windows, which each correspond to a single training example.
The central 6400 bp of each 12800 bp window is tiled with overlapping 200 bp bins that each correspond to one coordinate of the corresponding y∈{0,1}128𝑦superscript01128y\in\{0,1\}^{128}.
These 12800 bp windows are tiled 6400 bp apart, such that each genomic location falls within the central 6400 bp region of exactly one window.

We split the examples by domain (cell type) as well as by chromosome (a large contiguous subsequence of the genome) within a cell type.
In each split, we use one cell type for the test data, one for the validation data, and the remaining cell types for the training data.
These domain-wise splits are listed in Table [30](#S6.T30 "Table 30 ‣ Data. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), and are divided into two types:

1. 1.

   The *ENCODE-DREAM* splits follow the original challenge setup (Balsubramani et al., [2020](#bib.bib27)) closely in evaluating only on the cell type liver, which is a primary tissue, in contrast to all of the other cell types, which are immortalized cell lines that have been grown outside the body for many generations. This is a more realistic setting in the sense that it is easier to collect data from immortalized cell lines, which we can then use to train a model that predicts TF binding in harder-to-profile primary tissues.
   However, the fact that none of the training cell types are primary tissues might limit generalization to primary tissues. Moreover, because cell types are highly variable, conclusions drawn from a single liver cell type might not generalize to other cell types.
2. 2.

   We thus also use a *round-robin* set of splits, where we assign each cell type to test and validation sets in a rotating manner. This round-robin evaluation comprises several splits for each TF.

For each split in Table [30](#S6.T30 "Table 30 ‣ Data. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), the data are divided into training, validation, and test sets by chromosome:

1. 1.

   Training: 
   323,894 windows per training cell type, before filtering. To improve class balance, we filter out windows with all 128 bins labeled negative, which typically removes over 3/4 of these training windows; the exact number filtered out depends on the split.
   The training windows are taken from all chromosomes except {1,2,8,9,11,21}12891121\{1,2,8,9,11,21\}.
2. 2.

   Validation (OOD): 
   27,051 windows from 1 validation cell type and from chromosomes {2,9,11}2911\{2,9,11\}.
3. 3.

   Test (OOD): 
   23,109 windows from 1 test cell type and from chromosomes {1,8,21}1821\{1,8,21\}.
4. 4.

   Validation (ID): 
   27,051 windows in total across all training cell types, from chromosomes {2,9,11}2911\{2,9,11\}.
5. 5.

   Test (ID): 
   23,109 windows in total across all training cell types, from chromosomes {1,8,21}1821\{1,8,21\}.

For computational speed, the Validation (OOD) and Test (OOD) sets above were subsampled by a factor of 3 from the available raw data, while the Validation (ID) and Test (ID) sets were subsampled by a factor of 3 ×\times the number of training cell types.

Table 30: 
List of splits for which we trained models for ENCODE.
Performance of models on round-robin splits are averaged to get the summarized results in Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

|  |  |  |  |
| --- | --- | --- | --- |
| Split name | TF name | Test cell type | Validation cell type |
| ENCODE-DREAM | MAX | liver | HepG2 |
| round-robin | MAX | K562 | liver |
| round-robin | MAX | liver | A549 |
| round-robin | MAX | A549 | GM12878 |
| round-robin | MAX | GM12878 | H1-hESC |
| round-robin | MAX | H1-hESC | HCT116 |
| round-robin | MAX | HCT116 | HeLa-S3 |
| round-robin | MAX | HeLa-S3 | HepG2 |
| round-robin | MAX | HepG2 | K562 |
| ENCODE-DREAM | JUND | liver | HepG2 |
| round-robin | JUND | K562 | liver |
| round-robin | JUND | liver | MCF-7 |
| round-robin | JUND | MCF-7 | HCT116 |
| round-robin | JUND | HCT116 | HeLa-S3 |
| round-robin | JUND | HeLa-S3 | HepG2 |
| round-robin | JUND | HepG2 | K562 |

##### Evaluation.

We evaluate models by their average precision (AP) in predicting binary binding status (excluding all ambiguous bins). Specifically, we treat each bin as a separate binary classification problem; in other words, we split up each prediction of the 128-dimensional vector y𝑦y into at most 128 separate binary predictions after excluding ambiguous bins. We then compute the average precision of the model on this binary classification problem.

The choice of average precision as an evaluation metric is motivated by the class imbalance (low proportion of bound/positive labels) of this binary classification problem over bins.
All splits have more than one hundred times as many unbound bins than bound bins (Table [31](#S6.T31 "Table 31 ‣ Evaluation. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

Table 31: Binding site imbalance and uniqueness (across cell types) in binary genome-wide binding datasets in ENCODE.
Third column indicates the fraction of (non-ambiguous) bins that are labeled positive.
Fourth column indicates the fraction of positive bins (bound sites) that are cell-type-specific: they are bound (or ambiguous) in at most one other cell type.
Bins are 200bp wide.

|  |  |  |  |
| --- | --- | --- | --- |
| TF name | Cell type | Frac. positive bins | Frac. cell-type-specific binding sites |
| MAX | liver | 4.90×10−34.90superscript1034.90\times 10^{-3} | 0.518 |
| MAX | HepG2 | 6.20×10−36.20superscript1036.20\times 10^{-3} | 0.331 |
| MAX | K562 | 6.46×10−36.46superscript1036.46\times 10^{-3} | 0.368 |
| MAX | A549 | 5.90×10−35.90superscript1035.90\times 10^{-3} | 0.218 |
| MAX | GM12878 | 1.93×10−31.93superscript1031.93\times 10^{-3} | 0.217 |
| MAX | H1-hESC | 4.44×10−34.44superscript1034.44\times 10^{-3} | 0.363 |
| MAX | HCT116 | 6.46×10−36.46superscript1036.46\times 10^{-3} | 0.237 |
| MAX | HeLa-S3 | 4.21×10−34.21superscript1034.21\times 10^{-3} | 0.218 |
| JUND | liver | 4.45×10−34.45superscript1034.45\times 10^{-3} | 0.523 |
| JUND | K562 | 3.94×10−33.94superscript1033.94\times 10^{-3} | 0.408 |
| JUND | HCT116 | 4.08×10−34.08superscript1034.08\times 10^{-3} | 0.297 |
| JUND | HeLa-S3 | 3.60×10−33.60superscript1033.60\times 10^{-3} | 0.323 |
| JUND | HepG2 | 3.54×10−33.54superscript1033.54\times 10^{-3} | 0.513 |
| JUND | MCF-7 | 1.84×10−31.84superscript1031.84\times 10^{-3} | 0.335 |

#### F.2.2 Baseline results

##### Model.

Our model is a version of the fully convolutional U-Net model for image segmentation (Ronneberger et al., [2015](#bib.bib320)), modified from the architecture in Li and Guan ([2019](#bib.bib231)). It is illustrated in Figure [29](#S6.F29 "Figure 29 ‣ Model. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
We train each model using the average cross-entropy loss over the 128 output bins in each example (after excluding ambiguous bins).

![Refer to caption](/html/2012.07421/assets/figures/Unet-seq-wilds.png)


Figure 29: 
Architecture of the baseline prediction model, based on U-Net.
The final layers were modified to collapse the representation down to a single channel and finally convolved with kernel size 200 and stride 50, mimicking the resolution of labels along the genome.

For hyperparameters, we searched over the learning rates {10−5,10−4,10−3}superscript105superscript104superscript103\{10^{-5},10^{-4},10^{-3}\}, and L2subscript𝐿2L\_{2}-regularization strengths {10−4,10−3,10−2}superscript104superscript103superscript102\{10^{-4},10^{-3},10^{-2}\}.
We use 10 replicates (with different random seeds) for all reported results.

##### ERM results.

Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") and Table [34](#S6.T34 "Table 34 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") show the results of ERM models trained on each split.
On many individual splits (Table [34](#S6.T34 "Table 34 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), the OOD validation and OOD test performance are very different, reflecting the variability across the cell types.
On average across the round robin splits, the OOD validation performance is slightly higher than the OOD test performance, as we selected hyperparameters and did early stopping to maximize the former (Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
We also observed high variance in training and test performance across random seeds in a few splits (e.g., the K562 / liver split for the transcription factor JUND), which suggests some optimization instability in our training protocol.
We also computed the in-distribution baselines in a train-to-train setting, i.e., on the Validation (ID) and Test (ID) splits described above.

We also ran corresponding in-distribution baselines in a test-to-test setting, i.e., we trained ERM models on data from the training chromosomes in the test cell type, and tested it on the same test set comprising data from the test chromosomes in the test cell type.121212
Prior work has shown that there is minimal variation in performance between chromosomes on this problem (Won et al., [2010](#bib.bib406); Alipanahi et al., [2015](#bib.bib9); Keilwagen et al., [2019](#bib.bib202)), so we can approximate these training and test distributions as identical.
Table [33](#S6.T33 "Table 33 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") shows these in-distribution results.
For the round-robin splits, the difference between the train-to-train and test-to-test settings is that the former trains and tests on mixtures of multiple cell types, whereas the latter trains and tests on individual cell types.

We considered two TFs, MAX and JUND, separately.
For the ENCODE-DREAM splits, both TFs showed large ID-OOD performance gaps.
However, we opted not to use the ENCODE-DREAM split as a Wilds dataset
because the variability between cell types made us cautious about over-interpreting the results on a single cell type. For example, for MAX, we found that the Validation (OOD) and Test (OOD) cell types were so different that their results were anti-correlated across different random seeds, which would have made benchmarking challenging.
Moreover, the fact that the test cell type (liver) in the ENCODE-DREAM splits was the only primary tissue might have meant that the training data might have insufficient leverage for a model to learn to close the ID-OOD gap.

We therefore focused on analyzing the round-robin splits.
For MAX, using the train-to-train comparison, the average Test (ID) and Test (OOD) AP across the round-robin splits were not significantly different (64.9 (2.1) vs. 59.6 (2.0), respectively; Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
For JUND, using the train-to-train comparison, the average Test (ID) and Test (OOD) AP across the round-robin splits showed a larger gap (54.1 (4.2) vs. 42.9 (3.2), respectively; Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")), but the variability in training performance made these results less reliable. Moreover, the test-to-test ID results were significantly higher than the train-to-train ID results (62.4 (2.6) vs. 54.1 (4.2), respectively), which suggests that either the model capacity or feature set is not rich enough to fit the variation across different cell types.
We therefore opted not to include the round-robin splits in Wilds as well.

Table 32: ERM baseline results on ENCODE.
All numbers are average precision.
“Round-robin” indicates the average performance over all splits marked “round-robin” in Table [30](#S6.T30 "Table 30 ‣ Data. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Parentheses show standard deviation across replicates, for liver; and average of such standard deviations across splits, for round-robin.
Expanded results per round-robin split are in Table [35](#S6.T35 "Table 35 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| TF | Split scheme | Validation (ID) | Test (ID) | Validation (OOD) | Test (OOD) |
| MAX | ENCODE-DREAM | 70.3 (2.1) | 68.3 (1.9) | 67.9 (1.6) | 45.0 (1.5) |
| MAX | round-robin | 65.0 (2.1) | 64.9 (2.1) | 62.1 (1.2) | 59.6 (2.0) |
| JUND | ENCODE-DREAM | 65.9 (1.2) | 66.7 (1.4) | 32.9 (1.0) | 42.3 (2.5) |
| JUND | round-robin | 53.2 (4.0) | 54.1 (4.2) | 47.2 (2.4) | 42.9 (3.2) |




Table 33: In-distribution results on ENCODE: when training and validation cell types are set to the test cell type.
All numbers are average precision.
“Round-robin” indicates the average performance over all splits marked “round-robin” in Table [30](#S6.T30 "Table 30 ‣ Data. ‣ F.2.1 Setup ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").
Parentheses show standard deviation across replicates, for liver; and average of such standard deviations across splits, for round-robin.
Expanded results per round-robin split are in Table [35](#S6.T35 "Table 35 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts").

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| TF | Split scheme | Train | Validation (ID) | Test (ID) |
| MAX | ENCODE-DREAM | 76.1 (0.9) | 57.3 (1.2) | 57.6 (1.3) |
| MAX | round-robin | 77.3 (0.9) | 65.2 (1.3) | 65.4 (1.3) |
| JUND | ENCODE-DREAM | 76.0 (0.6) | 55.8 (0.5) | 56.0 (0.7) |
| JUND | round-robin | 79.3 (2.7) | 61.8 (2.4) | 62.4 (2.6) |




Table 34: Baseline results on ENCODE.
In-distribution (ID) results correspond to the train-to-train setting.
Parentheses show standard deviation across replicates.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| TF name | Test / Val cell type | Train AP | Val (ID) AP | Test (ID) AP | Val (OOD) AP | Test (OOD) AP |
| MAX | liver / HepG2 | 79.5 (1.0) | 70.3 (2.1) | 68.3 (1.9) | 67.9 (1.6) | 45.0 (1.5) |
| MAX | K562 / liver | 75.1 (2.3) | 59.9 (4.5) | 59.2 (3.9) | 47.6 (1.0) | 63.6 (4.5) |
| MAX | liver / A549 | 78.5 (1.3) | 68.6 (0.8) | 68.4 (0.5) | 66.6 (1.3) | 38.5 (1.1) |
| MAX | A549 / GM12878 | 78.0 (1.7) | 66.0 (2.6) | 66.6 (2.8) | 46.9 (1.9) | 65.0 (2.5) |
| MAX | GM12878 / H1-hESC | 80.0 (1.2) | 69.6 (0.8) | 69.2 (0.6) | 65.4 (0.7) | 46.3 (0.5) |
| MAX | H1-hESC / HCT116 | 75.5 (3.3) | 63.2 (3.2) | 63.9 (3.3) | 70.6 (0.5) | 61.7 (2.5) |
| MAX | HCT116 / HeLa-S3 | 76.8 (1.1) | 64.5 (1.6) | 64.9 (1.3) | 63.9 (0.9) | 69.4 (0.9) |
| MAX | HeLa-S3 / HepG2 | 77.7 (1.7) | 65.0 (1.9) | 64.8 (3.3) | 67.7 (1.9) | 64.0 (2.5) |
| MAX | HepG2 / K562 | 77.3 (1.3) | 63.2 (1.3) | 62.1 (1.4) | 68.5 (1.1) | 68.4 (1.1) |
| JUND | liver / HepG2 | 82.6 (1.3) | 65.9 (1.2) | 66.7 (1.4) | 32.9 (1.0) | 42.3 (2.5) |
| JUND | K562 / liver | 54.2 (10.5) | 29.9 (8.4) | 33.3 (8.4) | 32.9 (3.7) | 51.2 (4.5) |
| JUND | liver / MCF-7 | 83.6 (2.4) | 65.5 (3.4) | 65.3 (4.0) | 28.9 (2.7) | 29.2 (3.2) |
| JUND | MCF-7 / HCT116 | 76.4 (3.5) | 52.5 (3.4) | 53.6 (3.8) | 51.8 (3.7) | 27.2 (4.2) |
| JUND | HCT116 / HeLa-S3 | 75.7 (1.5) | 50.0 (2.4) | 50.6 (2.4) | 69.2 (2.1) | 52.9 (3.2) |
| JUND | HeLa-S3 / HepG2 | 78.0 (2.1) | 59.5 (5.1) | 60.0 (4.9) | 30.3 (1.3) | 66.6 (3.6) |
| JUND | HepG2 / K562 | 79.6 (0.9) | 61.9 (1.1) | 62.3 (1.4) | 69.8 (0.6) | 30.5 (0.6) |




Table 35: Test-to-test results on ENCODE.
Parentheses show standard deviation across replicates.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| TF name | Test cell type | Train | Val (ID) AP | Test (ID) AP |
| MAX | liver | 76.1 (0.9) | 57.3 (1.2) | 57.6 (1.3) |
| MAX | HepG2 | 76.1 (0.8) | 66.5 (1.1) | 68.4 (1.3) |
| MAX | K562 | 83.6 (0.7) | 74.7 (0.8) | 75.9 (0.7) |
| MAX | A549 | 77.2 (0.8) | 68.4 (1.4) | 67.5 (1.5) |
| MAX | GM12878 | 64.7 (0.8) | 50.9 (1.3) | 49.2 (1.5) |
| MAX | H1-hESC | 76.2 (1.8) | 65.3 (2.1) | 64.1 (1.9) |
| MAX | HCT116 | 82.8 (0.7) | 73.6 (0.8) | 74.5 (0.8) |
| MAX | HeLa-S3 | 80.5 (0.8) | 64.8 (1.4) | 66.4 (1.4) |
| JUND | liver | 76.0 (0.6) | 55.8 (0.5) | 56.0 (0.7) |
| JUND | K562 | 87.3 (1.3) | 74.5 (1.1) | 76.0 (1.7) |
| JUND | HCT116 | 80.9 (2.8) | 69.5 (4.9) | 68.4 (4.7) |
| JUND | HeLa-S3 | 87.2 (0.8) | 69.0 (0.7) | 70.9 (0.8) |
| JUND | HepG2 | 84.1 (9.8) | 71.9 (4.6) | 72.1 (4.9) |
| JUND | MCF-7 | 60.1 (0.8) | 30.1 (2.4) | 31.1 (2.9) |

##### Discussion.

Even in the in-distribution (test-to-test) setting, the results in Table [35](#S6.T35 "Table 35 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") show how
different model performance can be for different domains.
For example, the liver domain of primary tissue (from a human donor) is derived from lower-quality data than many of the long-standard cell lines (grown outside the human body) constituting other domains, and is consequently noisier than many of them (Balsubramani et al., [2017](#bib.bib26)).
The extent of this variation underscores the importance of accounting for the variability between domains when measuring the effect of distribution shift;
for example, a train-to-train comparison could lead to significantly different conclusions than a test-to-test comparison.

The effect of the distribution shift also seems to depend on the particular TF (MAX vs.  JUND) used.
Biologically, different TFs show different levels of cell-type-specificity, and better understanding which TFs have binding patterns that can be accurately predicted from the cell-type-specific accessibility assays is important future work.

One of the main obstacles preventing us from using this ENCODE dataset as a Wilds benchmark is the instability in optimization that we reported above.
We speculate that this instability could, in part, be due to the class imbalance in the data, but more work will be needed to ascertain this and to develop methods for training models more reliably on this type of genomic data.

As we mentioned above, Table [32](#S6.T32 "Table 32 ‣ ERM results. ‣ F.2.2 Baseline results ‣ F.2 ENCODE: Transcription factor binding across different cell types ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") reports significantly higher test-to-test ID results than train-to-train ID results for the JUND round-robin split scheme. The main difference between the test-to-test and train-to-train settings in the round-robin splits is that the former trains and tests on a single cell type, whereas the latter trains and tests on a mixture of cell types. The fact that ID performance is significantly higher in the former than the latter suggests that the learned models are not able to fit JUND binding patterns across multiple cell types.
This could be due to a model family that is not large or expressive enough, or it could be because the feature set does not have all of the necessary information to accurately predict binding across cell types.
In either case, it is unlikely that a training algorithm developed to be robust to distribution shifts will be able to significantly improve OOD performance in this setting, as the issue seems to lie in the model family or the data distribution instead.

Overall, it is commonly understood that distribution shifts between cell types are a significant problem for TF binding prediction, and many methods have been developed to tackle these shifts (Balsubramani et al., [2017](#bib.bib26); Li et al., [2019a](#bib.bib232); Li and Guan, [2019](#bib.bib231); Keilwagen et al., [2019](#bib.bib202); Quang and Xie, [2019](#bib.bib303)).
Nonetheless, we found it challenging to establish a rigorous distribution shift benchmark around this task,
as our results were confounded by factors such as optimization issues, large variability between cell types,
and the difficulty of learning a model that could fit multiple cell types even in an i.i.d. setting.
We hope that future work on evaluating and mitigating distribution shifts in TF binding prediction
can build upon our results and address these challenges.

#### F.2.3 Additional details

##### Additional dataset details.

The ground-truth labels were derived from high-quality chromatin immunoprecipitation sequencing (ChIP-seq) experiments, which provide a genome-wide track of binding enrichment scores for each TF.
Statistical methods based on standardized pipelines (Landt et al., [2012](#bib.bib221)) were used to identify high-confidence binding events across the genome, resulting in a genome-wide track indicating whether each of the windows of sequence in the genome is bound or unbound by the TF, or whether binding is ambiguous but likely (these were ignored in our benchmarking).

Our data include two TFs chosen for their basic importance in cell-type-specific gene regulation: MAX and JUND.
MAX canonically recognizes a short, common sequence (the domain CACGTG), but its structure leads it to bind to DNA as a dimer, and facilitates cooperative activity with a range of partners (Grandori et al., [2000](#bib.bib155)) with many weaker and longer-range sequence determinants of binding (Allevato et al., [2017](#bib.bib12)).
JUND belongs to a large family of TFs (bZIP) known for binding in cooperation with partners in the family in a variety of modes, all involving a short 7bp sequence (TGA[C/G]TCA) and its two halves.

##### Additional model details.

The network consists of encoder and decoder portions:

* •

  Encoder.  The encoder is composed of five downscaling convolutional blocks, each consisting of two stride-1 convolutional layers with kernel size 7 (and padding such that the output size is left unchanged), followed by a max-pooling layer with kernel size 2.
  Each successive block halves the input window size and scales up the number of convolutional filters (by 1.5).
* •

  Decoder.  Mirroring the encoder, the decoder is composed of five upscaling convolutional blocks, each consisting of two convolutional layers with kernel size 7 and an upsampling layer (a ConvTranspose layer with kernel size 2 and stride 2).
  Each successive block doubles the input window size.
  The respective sizes of the decoder layer representations are the same as the encoder in reverse, culminating in a (12800×15)1280015(12800\times 15) representation that is then run through a convolutional layer (kernel size 200, stride 50) to reduce it to a single channel (with length 253).
  A final fully-connected layer results in a 128-dimensional output.

Batch normalization is applied after every layer except the last, and each intermediate convolutional layer is padded such that the output and input sizes are equal.

##### Additional data sources.

The ENCODE-DREAM prediction challenge contains binding data for many TFs from a large range of cell types,
discretized into the same 200-bp windows used in this benchmark.
The ENCODE portal (encodeproject.org) contains more ChIP-seq datasets from the 13 challenge cell lines for which we provide DNase accessibility data.
DNA shape and gene expression data types were also provided in the original challenge.

* •

  DNA shape.  Twisting, bending, and shearing of DNA influence local binding in a TF-specific fashion (Rohs et al., [2009](#bib.bib318)).
* •

  Gene expression.  Expression levels of all human genes were provided using RNA-seq data from ENCODE. This can be used to model the presence of cofactor proteins that can recruit TFs for binding (Ptashne and Gann, [1997](#bib.bib302)).

However, none of the top challenge participants found these data modalities useful (Balsubramani et al., [2017](#bib.bib26)), so they are not provided in this benchmark.

##### Data normalization.

We normalize the distribution of each DNase-seq signal readout to the average of the DNase-seq signals over training cell types.
We use a version of quantile normalization (Bolstad et al., [2003](#bib.bib53)) with piecewise polynomial interpolation.
Li and Guan ([2019](#bib.bib231)) also use this, but instead normalize to the test domain’s DNase distribution.
As this technique uses test-domain data, it is out of the scope of our benchmark.
However, we note that in genomics settings it is realistic to have relatively cheaply available chromatin accessibility data in the target cell type of interest.

##### Modifications to the original setup.

The prediction task of the challenge was a binary classification problem over the 200 bp bins, which did not involve the fixed 12800 bp windows.
To predict on a bin, participating teams were free to use as much of the regions surrounding (flanking) the bin as they wished.
The winning teams all used at least 1000 bp total for each bin, and further work has shown the efficacy of using much larger flanking regions of tens of thousands of bp (Quang and Xie, [2019](#bib.bib303); Avsec et al., [2021a](#bib.bib20)).
We instead predict on 128 bins at once (following Li and Guan ([2019](#bib.bib231))),
which allows for more efficient training and prediction.

Our ERM baselines’ OOD test performance is competitive with the original challenge results, but lower than the state-of-the-art performance of Li and Guan ([2019](#bib.bib231)) because of the aforementioned differences in data processing, splits, and architecture, as well as the cross-domain training method employed by that paper and predecessor work (Li et al., [2019a](#bib.bib232)).
These and other state-of-the-art models noted that their domain adaptation strategies played a major role in improving performance.

### F.3 BDD100K: Object recognition in autonomous driving across locations

As discussed in Section [8.6](#S8.SS6 "8.6 Robotics ‣ 8 Distribution shifts in other application areas ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), autonomous driving, and robotics in general, is an important
application that requires effective and robust tools for handling distribution shift. Here, we
discuss our findings on a modified version of the BDD100K dataset that evaluates on shifts based on
time of day and location. Our results below suggest that more challenging tasks, such as object
detection and segmentation, may be more suited to evaluations of distribution shifts in an
autonomous driving context.

![Refer to caption](/html/2012.07421/assets/figures/bdd100k.png)


Figure 30: For BDD100K, we study two different types of shift, based on time of day and location. We
visualize randomly chosen images and their corresponding labels from the training, validation, and
test splits for both shifts. The labels are 9-dimensional binary vectors indicating the presence
(1) or absence (0) of, in order: bicycles, buses, cars, motorcycles, pedestrians, riders, traffic
lights, traffic signs, and trucks.




Table 36: Average multi-task classification accuracy of ERM trained models on BDD100K. All
results are reported across 3 random seeds, with standard deviation in parentheses. We observe no
substantial drops in the presence of test time distribution shifts.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Time of day shift | | Location shift | |
| Algorithm | Validation (ID) | Test (OOD) | Validation (ID) | Test (OOD) |
| ERM | 87.1 (0.3) | 89.7 (0.2) | 87.9 (0.0) | 86.9 (0.0) |

#### F.3.1 Setup

##### Task.

In line with the other datasets in Wilds, we evaluate using a classification task. Specifically,
the task is to predict whether or not 9 different categories appear in the image x𝑥x: bicycles,
buses, cars, motorcycles, pedestrians, riders, traffic lights, traffic signs, and trucks. This is a
multi-task binary classification problem, and the label y𝑦y is thus a 9-dimensional binary vector.

##### Data.

The BDD100K dataset is a large and diverse driving dataset crowd-sourced from tens of thousands of
drivers, covering four different geographic regions and many different times of day, weather
conditions, and scenes (Yu et al., [2020](#bib.bib424)). The original dataset contains 80,000 images in the combined
training and validation sets and is richly annotated for a number of different tasks such as
detection, segmentation, and imitation learning. We use bounding box labels to construct our task
labels, and as discussed later, we use location and image tags to construct the shifts we evaluate.

##### Evaluation.

In evaluating the trained models, we consider average accuracy across the binary classification
tasks, averaged over each of the validation and test sets separately. We next discuss how we create
and evaluate two different types of shift based on time of day and location differences.

#### F.3.2 Time of day shift

##### Distribution shift and evaluation.

We evaluate two different types of shift, depicted in Figure [30](#S6.F30 "Figure 30 ‣ F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"). For time of day
shift (Figure [30](#S6.F30 "Figure 30 ‣ F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") top row), we use the original BDD100K training set, which has roughly
equal proportions of daytime and non daytime images (Yu et al., [2020](#bib.bib424)). However, we construct a test
set using the original BDD100K validation set that only includes non-daytime images. We then split
roughly the same number of images randomly from the training set to form an in-distribution
validation set,
which allows us to do a train-to-train comparison. There are 64,993, 4,860, and 4,742 images in the training, validation, and test
splits, respectively.

##### ERM results.

Table [36](#S6.T36 "Table 36 ‣ F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") summarizes our findings. For time of day shift, we actually observe
slightly *higher* test performance, on only non daytime images, than validation performance on
mixed daytime and non daytime images. We contrast this with findings from
Dai and Van Gool ([2018](#bib.bib101)); Yu et al. ([2020](#bib.bib424)), who showed worse test performance for segmentation and detection tasks,
respectively, on non daytime images. We believe this disparity can be attributed to the difference
in tasks—for example, it is likely more difficult to draw an accurate bounding box for a car at
night than to simply recognize tail lights and detect the presence of a car.

#### F.3.3 Location shift

##### Distribution shift.

For location shift (Figure [30](#S6.F30 "Figure 30 ‣ F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") bottom row), we combine all of the data from the
original BDD100K training and validation sets. We construct training and validation sets from all of
the images captured in New York, and we use all images from California for the test set. The
validation set again is in-distribution with respect to the training set and has roughly the same
number of images as the test set. There are 53,277, 9,834, and 9,477 images in the training,
validation, and test splits, respectively.

##### ERM results.

In the case of location shift, we see from Table [36](#S6.T36 "Table 36 ‣ F.3 BDD100K: Object recognition in autonomous driving across locations ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts") that there is a small drop in
performance, possibly because this shift is more drastic as the locations are disjoint between
training and test time. However, the performance drop is relatively small and the test time
accuracy is still comparable to validation accuracy. In general, we believe that these results lend
support to the conclusion that, for autonomous driving and robotics applications, other more
challenging tasks are better suited for evaluating performance. Generally speaking, incorporating a
wide array of different applications will likely require a simultaneous effort to incorporate
different tasks as well.

### F.4 Amazon: Sentiment classification across different categories and time

Our benchmark dataset Amazon-wilds studies user shifts.
In Section [7](#S7 "7 Empirical trends ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), we discussed empirical trends on other types of distribution shifts on the same underlying 2018 Amazon Reviews dataset (Ni et al., [2019](#bib.bib274)).
We now present the detailed setup and empirical results for the time and category shifts.

#### F.4.1 Setup

##### Model.

For all experiments in this section, we finetune BERT-base-uncased models, using the implementation from Wolf et al. ([2019](#bib.bib405)), and with the following hyperparameter settings:
batch size 8; learning rate 2×10−62superscript1062\times 10^{-6}; L2subscript𝐿2L\_{2}-regularization strength 0.010.010.01; 3 epochs; and a maximum number of tokens of 512.
These hyperparameters are taken from the Amazon-wilds experiments.

#### F.4.2 Time shifts

##### Problem setting.

We consider the domain generalization setting, where the domain d𝑑d is the year in which the reviews are written.
As in Amazon-wilds, the task is multi-class sentiment classification, where
the input x𝑥x is the text of a review, the label y𝑦y is a corresponding star rating from 1 to 5.

##### Data.

The dataset is a modified version of the Amazon Reviews dataset (Ni et al., [2019](#bib.bib274)) and comprises customer reviews on Amazon.
Specifically, we consider the following split:

1. 1.

   Training: 1,000,000 reviews written in years 2000 to 2013.
2. 2.

   Validation (OOD): 20,000 reviews written in years 2014 to 2018.
3. 3.

   Test (OOD): 20,000 reviews written in years 2014 to 2018.

To construct the above split, we first randomly sample 4,000 reviews per year for the evaluation splits. For years in which there are not sufficient reviews, we split the reviews equally between validation and test. After constructing the evaluation set, we then randomly sample from the remaining reviews to form the training set.

##### Evaluation.

To assess whether models generalize to future years, we evaluate models by their average accuracy on the OOD test set.

##### ERM results and performance drops.

We only observed modest performance drops due to time shift.
Our baseline model performs well on the OOD test set, achieving 76.0% accuracy on average and 75.4% on the worst year (Table [37](#S6.T37 "Table 37 ‣ ERM results and performance drops. ‣ F.4.2 Time shifts ‣ F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
To measure performance drops due to distribution shifts, we ran a test-to-test comparison by training a model on reviews written in years 2014 to 2018 (Table [38](#S6.T38 "Table 38 ‣ ERM results and performance drops. ‣ F.4.2 Time shifts ‣ F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The performance gaps between the model trained on the official split and the model trained on the test-to-test split are consistent but modest across the years, with the biggest drop of 1.1% for 2018.

Table 37: Baseline results on time shifts on the Amazon Reviews Dataset. We report the accuracy of models trained using ERM. In addition to the average accuracy across all years in each split, we report the accuracy for the worst-case year.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Train | | Validation (OOD) | | Test (OOD) | |
| Algorithm | Average | Worst year | Average | Worst year | Average | Worst year |
| ERM | 75.0 (0.0) | 72.4 (0.1) | 75.7 (0.1) | 74.6 (0.1) | 76.0 (0.1) | 75.4 (0.1) |




Table 38: Test-to-test in-distribution comparison for time shifts on Amazon Reviews Dataset. We observe only modest performance drops due to time shifts.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Setting | Year | 2014 | 2015 | 2016 | 2017 | 2018 |
| Official | 75.4 (0.1) | 75.8 (0.1) | 76.3 (0.1) | 76.4 (0.4) | 76.1 (0.1) |  |
| Test-to-test | 76.1 (0.2) | 76.8 (0.1) | 77.1 (0.2) | 77.5 (0.2) | 77.0 (0.0) |  |

#### F.4.3 Category shifts

Shifts across categories—where a model is trained on reviews in one category and then tested on another—have been studied extensively (Blitzer et al., [2007](#bib.bib48); Mansour et al., [2009](#bib.bib256); Hendrycks et al., [2020c](#bib.bib176)).
In line with prior work, we observe that model performance drops upon evaluating on a few unseen categories.
However, the observed difference between out-of-distribution and in-distribution baselines varies from category to category and is not consistently large (Hendrycks et al., [2020c](#bib.bib176)).
In addition, we find that training on more diverse data with more product categories tends to improve generalization to unseen categories and reduce the effect of the distribution shift; similar phenomena have also been reported in prior work (Mansour et al., [2009](#bib.bib256); Guo et al., [2018](#bib.bib159)).

##### Problem setting.

We consider the domain generalization setting, where the domain d𝑑d is the product category.
As in Amazon-wilds, the task is multi-class sentiment classification, where
the input x𝑥x is the text of a review, the label y𝑦y is a corresponding star rating from 1 to 5.

##### Data.

The dataset is a modified version of the Amazon Reviews dataset (Ni et al., [2019](#bib.bib274)) and comprises customer reviews on Amazon.
Specifically, we consider the following split for a given set of training categories:

1. 1.

   Training: up to 1,000,000 reviews in training categories.
2. 2.

   Validation (OOD): reviews in categories unseen during training.
3. 3.

   Test (OOD): reviews in categories unseen during training.
4. 4.

   Validation (ID): reviews in training categories.
5. 5.

   Test (ID): reviews in training categories.

To construct the above split, we first randomly sample 1,000 reviews per category for the evaluation splits (for categories with insufficient number of reviews, we split the reviews equally between validation and test) and then randomly sample from the remaining reviews to form the training set.

##### Evaluation.

To assess whether models generalize to unseen categories, we evaluate models by their average accuracy on each of the categories in the OOD test set.

Table 39: Baseline results on category shifts on the Amazon Reviews Dataset. We report the accuracy of models trained using ERM on a single category (Books) versus four categories (Books, Movies and TV, Home and Kitchen, and Electronics). Across many categories unseen at training time, corresponding to each row, the latter model modestly but consistently outperforms the former.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Validation (OOD) | | Test (OOD) | |
| Category | Single | Multiple | Single | Multiple |
| All Beauty | 87.8 (0.8) | 85.6 (1.4) | 82.9 (0.8) | 83.1 (0.8) |
| Arts Crafts and Sewing | 81.6 (0.7) | 83.4 (0.4) | 79.5 (0.2) | 81.7 (0.2) |
| Automotive | 78.2 (0.4) | 80.4 (0.4) | 76.5 (0.2) | 78.9 (0.2) |
| CDs and Vinyl | 78.1 (0.7) | 78.6 (0.2) | 78.5 (0.7) | 79.7 (0.3) |
| Cell Phones and Accessories | 76.8 (0.3) | 79.0 (0.7) | 78.0 (0.5) | 80.2 (1.0) |
| Clothing Shoes and Jewelry | 69.8 (0.6) | 72.6 (0.2) | 73.3 (0.2) | 75.2 (0.2) |
| Digital Music | 77.5 (0.5) | 77.8 (0.5) | 80.7 (1.0) | 81.7 (0.6) |
| Gift Cards | 88.2 (1.5) | 90.7 (3.1) | 90.7 (0.8) | 91.2 (0.0) |
| Grocery and Gourmet Food | 79.0 (0.3) | 79.0 (0.1) | 79.3 (0.7) | 79.2 (0.2) |
| Industrial and Scientific | 77.0 (0.4) | 78.1 (0.6) | 77.4 (0.2) | 78.9 (0.1) |
| Kindle Store | 75.0 (0.3) | 74.5 (0.3) | 73.2 (0.3) | 73.1 (0.5) |
| Luxury Beauty | 67.2 (0.2) | 70.2 (0.6) | 67.4 (0.7) | 69.4 (0.9) |
| Magazine Subscriptions | 74.2 (3.2) | 71.0 (0.0) | 90.3 (0.0) | 89.2 (1.9) |
| Musical Instruments | 76.1 (0.3) | 78.3 (0.3) | 78.8 (0.8) | 80.9 (0.2) |
| Office Products | 78.5 (0.3) | 80.0 (0.5) | 76.7 (0.5) | 78.9 (0.4) |
| Patio Lawn and Garden | 70.8 (0.6) | 72.9 (0.3) | 75.5 (0.6) | 79.7 (0.6) |
| Pet Supplies | 74.5 (0.4) | 77.1 (0.9) | 74.4 (0.4) | 76.8 (0.5) |
| Prime Pantry | 80.5 (0.3) | 80.2 (0.2) | 78.5 (0.6) | 79.4 (0.3) |
| Software | 65.8 (1.7) | 67.1 (1.1) | 71.3 (1.5) | 72.6 (0.5) |
| Sports and Outdoors | 74.2 (0.5) | 76.0 (0.2) | 75.8 (0.2) | 78.3 (0.6) |
| Tools and Home Improvement | 74.0 (1.1) | 76.4 (0.3) | 73.1 (0.6) | 74.4 (0.2) |
| Toys and Games | 78.9 (0.4) | 79.9 (0.2) | 77.6 (0.2) | 80.9 (0.2) |
| Video Games | 76.0 (0.2) | 76.6 (0.8) | 76.9 (0.6) | 78.0 (0.6) |

##### ERM results.

We first considered training on four categories (Books, Movies and TV, Home and Kitchen, and Electronics) and evaluating on unseen categories.
We observed that a BERT-base-uncased model trained via ERM yields a test accuracy of 75.4% on the four in-distribution categories and a wide range of accuracies on unseen categories (Table [39](#S6.T39 "Table 39 ‣ Evaluation. ‣ F.4.3 Category shifts ‣ F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts"), columns Multiple).
While the accuracies on some unseen categories are lower than the train-to-train in-distribution accuracy, it is unclear whether the performance gaps stem from the distribution shift or differences in intrinsic difficulty across categories;
in fact, the accuracy is higher on many unseen categories (e.g., All Beauty) than on the in-distribution categories, illustrating the importance of accounting for intrinsic difficulty.

To control for intrinsic difficulty, we ran a test-to-test comparison on each target category.
We controlled for the number of training reviews to the extent possible;
the standard model is trained on 1 million reviews in the official split, and each test-to-test model is trained on 1 million reviews or less, as limited by the number of reviews per category.
We observed performance drops on some categories, for example on Clothing, Shoes, and Jewelry (83.0% in the test-to-test setting versus 75.2% in the official setting trained on the four different categories) and on Pet Supplies (78.8% to 76.8%).
However, on the remaining categories, we observed more modest performance gaps, if at all.
While we thus found no evidence for significance performance drops for many categories, these results do not rule out such drops either:
one confounding factor is that some of the oracle models are trained on significantly smaller training sets and therefore underestimate the in-distribution performance.

In addition, we compared training on four categories (Books, Movies and TV, Home and Kitchen, and Electronics), as above, to training on just one category (Books), while keeping the training set size constant.
We found that decreasing the number of training categories in this way lowered out-of-distribution performance:
across many OOD categories, accuracies were modestly but consistently higher for the model trained on four categories than for the model trained on a single category (Table [39](#S6.T39 "Table 39 ‣ Evaluation. ‣ F.4.3 Category shifts ‣ F.4 Amazon: Sentiment classification across different categories and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).

### F.5 Yelp: Sentiment classification across different users and time

We present empirical results on time and user shifts in the Yelp Open Dataset131313https://www.yelp.com/dataset.

#### F.5.1 Setup

##### Model.

For all experiments in this section, we finetune BERT-base-uncased models, using the implementation from Wolf et al. ([2019](#bib.bib405)), and with the following hyperparameter settings:
batch size 8; learning rate 2×10−62superscript1062\times 10^{-6}; L2subscript𝐿2L\_{2}-regularization strength 0.010.010.01; 3 epochs with early stopping; and a maximum number of tokens of 512.
We select the above hyperparameters based on a grid search over learning rates 1×10−6,2×10−6,1×10−5,2×10−5

1superscript1062superscript1061superscript1052superscript1051\times 10^{-6},2\times 10^{-6},1\times 10^{-5},2\times 10^{-5}, using the time shift setup; for the user shifts, we adopted the same hyperparameters.

#### F.5.2 Time shifts

##### Problem setting.

We consider the domain generalization setting, where the domain d𝑑d is the year in which the reviews are written.
As in Amazon-wilds, the task is multi-class sentiment classification, where
the input x𝑥x is the text of a review, the label y𝑦y is a corresponding star rating from 1 to 5.

##### Data.

The dataset is a modified version of the Yelp Open Dataset and comprises 1 million customer reviews on Yelp.
Specifically, we consider the following split:

1. 1.

   Training: 1,000,000 reviews written in years 2006 to 2013.
2. 2.

   Validation (OOD): 20,000 reviews written in years 2014 to 2019.
3. 3.

   Test (OOD): 20,000 reviews written in years 2014 to 2019.

To construct the above split, we first randomly sample 1,000 reviews per year for the evaluation splits. For years in which there are not sufficient reviews, we split the reviews equally between validation and test. After constructing the evaluation set, we then randomly sample from the remaining reviews to form the training set.

##### Evaluation.

To assess whether models generalize to future years, we evaluate models by their average accuracy on the OOD test set.

##### ERM results and performance drops.

We observe modest performance drops due to time shift.
A BERT-base-uncased model trained with the standard ERM objective performs well on the OOD test set, achieving 76.0% accuracy on average and 73.9% on the worst year (Table [40](#S6.T40 "Table 40 ‣ ERM results and performance drops. ‣ F.5.2 Time shifts ‣ F.5 Yelp: Sentiment classification across different users and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
To measure performance drops due to distribution shifts, we run a test-to-test in-distribution comparison by training on reviews written in years 2014 to 2019 (Table [41](#S6.T41 "Table 41 ‣ ERM results and performance drops. ‣ F.5.2 Time shifts ‣ F.5 Yelp: Sentiment classification across different users and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
While there are consistent performance gaps between the out-of-distribution and the in-distribution baselines in later years, they are modest in magnitude with the largest drop of 3.1% for 2018.

Table 40: Baseline results on time shifts on the Yelp Open Dataset. We report the accuracy of models trained using ERM. Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Train | | Validation (OOD) | | Test (OOD) | |
| Algorithm | Average | Worst year | Average | Worst year | Average | Worst year |
| ERM | 71.4 (0.7) | 65.7 (1.1) | 76.1 (0.1) | 73.1 (0.2) | 76.0 (0.4) | 73.9 (0.4) |




Table 41: Test-to-test in-distribution comparison on the Yelp Open Dataset. We observe only modest performance drops due to time shifts. Parentheses show standard deviation across 3 replicates.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Year | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
| OOD baseline (ERM) | 75.8 (0.6) | 75.2 (0.9) | 73.9 (0.4) | 77.0 (0.4) | 76.7 (0.3) | 77.2 (0.6) |
| ID baseline (oracle) | 75.2 (0.5) | 75.0 (0.5) | 76.4 (0.7) | 78.8 (0.6) | 79.6 (0.4) | 79.5 (0.5) |

#### F.5.3 User shift

##### Problem setting.

As in Amazon-wilds, we consider the domain generalization setting, where the domains are reviewers and the task is multi-class sentiment classification.
Concretely, the input x𝑥x is the text of a review, the label y𝑦y is a corresponding star rating from 1 to 5, and the domain d𝑑d is the identifier of the user that wrote the review.

##### Data.

The dataset is a modified version of the Yelp Open Dataset and comprises 1.2 million customer reviews on Yelp.
To measure generalization to unseen reviewers, we train on reviews written by a set of reviewers and consider reviews written by *unseen* reviewers at test time.
Specifically, we consider the following random split across reviewers:

1. 1.

   Training: 1,000,104 reviews from 11,856 reviewers.
2. 2.

   Validation (OOD): 40,000 reviews from another set of 1,600 reviewers, distinct from training and test (OOD).
3. 3.

   Test (OOD): 40,000 reviews from another set 1,600 reviewers, distinct from training and validation (OOD).
4. 4.

   Validation (ID): 40,000 reviews from 1,600 of the 11,856 reviewers in the training set.
5. 5.

   Test (ID): 40,000 reviews from 1,600 of the 11,856 reviewers in the training set.

The training set includes at least 25 reviews per reviewer, whereas the evaluation sets include exactly 25 reviews per reviewer.
While we primarily evaluate model performance on the above OOD test set, we also provide in-distribution validation and test sets for potential use in hyperparameter tuning and additional reporting.
These in-distribution splits comprise reviews written by reviewers in the training set.

##### Evaluation.

To assess whether models perform consistently well across reviewers, we evaluate models by their accuracy on the reviewer at the 10th percentile.

##### ERM results and performance drops.

We observe only modest variations in performance across reviewers.
A BERT-base-uncased model trained with the standard ERM objective achieves 71.5% accuracy on average and 56.0% accuracy at the 10th percentile reviewer (Table [42](#S6.T42 "Table 42 ‣ ERM results and performance drops. ‣ F.5.3 User shift ‣ F.5 Yelp: Sentiment classification across different users and time ‣ F Datasets with distribution shifts that do not cause performance drops ‣ Wilds: A Benchmark of in-the-Wild Distribution Shifts")).
The above variation is modestly larger than expected from randomness; a random binomial baseline with equal average accuracy would have a tenth percentile accuracy of 60.1%.

Table 42: Baseline results on user shifts on the Yelp Open Dataset. We report the accuracy of models trained using ERM. In addition to the average accuracy across all reviews, we compute the accuracy for each reviewer and report the performance for the reviewer in the 10th percentile. In-distribution (ID) results correspond to the train-to-train setting. Parentheses show standard deviation across 3 replicates.

Validation (OOD)
Test (OOD)
Validation (ID)
Test (ID)

Algorithm
10th percentile
Average
10th percentile
Average
10th percentile
Average
10th percentile
Average

ERM
56.0 (0.0)
70.5 (0.0)
56.0 (0.0)
71.5 (0.0)
56.0 (0.0)
70.6 (0.0)
56.0 (0.0)
70.9 (0.1)

[◄](/html/2012.07420)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2012.07421)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2012.07421)
[View original  
on arXiv](https://arxiv.org/abs/2012.07421)[►](/html/2012.07422)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 2 08:01:03 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
