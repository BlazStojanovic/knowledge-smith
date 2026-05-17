---
arxiv: '2408.07666'
authors:
- Enneng Yang
- Li Shen
- Guibing Guo
- Xingwei Wang
- Xiaochun Cao
- Jie Zhang
- Dacheng Tao
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications
  and Opportunities'
url: https://arxiv.org/abs/2408.07666
year: 2024
---

# Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities

Enneng Yang§

 Li Shen†

 Guibing Guo§

 Xingwei Wang§

 Xiaochun Cao†

 Jie Zhang‡

  Dacheng Tao‡
  
§Northeastern University

 China;
†Sun Yat-sen University

 China;
‡Nanyang Technological University

 Singapore
  
ennengyang@stumail.neu.edu.cn;
mathshenli@gmail.com; guogb@swc.neu.edu.cn; wangxw@mail.neu.edu.cn
  
caoxiaochun@mail.sysu.edu.cn; zhangj@ntu.edu.sg; dacheng.tao@ntu.edu.sg

###### Abstract

Model merging is an efficient empowerment technique in the machine learning community that does not require the collection of raw training data and does not require expensive computation. As model merging becomes increasingly prevalent across various fields, it is crucial to understand the available model merging techniques comprehensively. However, there is a significant gap in the literature regarding a systematic and thorough review of these techniques. This survey provides a comprehensive overview of model merging methods and theories, their applications in various domains and settings, and future research directions. Specifically, we first propose a new taxonomic approach that exhaustively discusses existing model merging methods. Secondly, we discuss the application of model merging techniques in large language models, multimodal large language models, and 10+ machine learning subfields, including continual learning, multi-task learning, few-shot learning, etc. Finally, we highlight the remaining challenges of model merging and discuss future research directions. A comprehensive list of papers about model merging is available at <https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications>.

## 1 Introduction

Model merging, also known as model fusion, is an effective technique that merges the parameters of multiple separate models with different capabilities to build a universal model without needing access to the original training data or expensive computation. The concept most relevant to model merging is ensemble learning [[33](#bib.bib33), [142](#bib.bib142), [180](#bib.bib180), [109](#bib.bib109)], as both facilitate knowledge fusion and transfer. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), the main difference between them is that ensemble learning must save all individual models and fuse the predictions (or outputs) of multiple models during the inference phase, whereas model merging performs merging directly at the parameter level and only has one final model during inference. This gives model merging more attractive properties.

Although model merging is a relatively young topic, it is evolving rapidly and has already found applications in several domains. For example, in foundation models, models fine-tuned by different downstream tasks are merged to enhance the capabilities of large language models, and image generative models with different styles are merged to create a new model with mixed-style capabilities. In particular, the number of pre-trained and fine-tuned checkpoints in the machine learning community has grown exponentially in recent years, including open-source repositories such as Huggingface [[182](#bib.bib182)], torchvision [[111](#bib.bib111)], and timm [[181](#bib.bib181)], making it easy for users to obtain well-trained expert models of varying abilities. These rich model repositories further promote the rapid development of model merging direction.

!(/html/2408.07666/assets/x1.png)

Figure 1: An illustration of the ensemble learning paradigm versus the model merging paradigm. (a) T𝑇T separate models for T𝑇T tasks, (b) Assemble T𝑇T separate models for T𝑇T tasks, (c) A merged model for T𝑇T tasks.

As model merging becomes increasingly popular in various areas of the machine learning community, it is crucial to have a comprehensive understanding of the advantages and limitations of existing model merging techniques and their applications across different domains. Although some efforts have been made by the community [[96](#bib.bib96), [214](#bib.bib214), [157](#bib.bib157), [48](#bib.bib48)], there are still large gaps to be filled. More specifically, MergeKit [[48](#bib.bib48)] and FusionBench [[157](#bib.bib157)] are technical reports in which only seven representative methods are discussed in MergeKit, and eight merging methods are discussed in FusionBench. Additionally, Zheng et al. [[214](#bib.bib214)] discuss the topic of “learning from models” and it only mentions model merging as a subsection (single page only) in the whole paper. The most related work to the “model merging” topic is [[96](#bib.bib96)], but in terms of application, it only discusses model merging in three scenarios: federated learning, fine-tuning, and distillation. It also ignores a lot of recently published articles due to the rapid evolution of the model merging direction.
To address these gaps, this survey aims to elucidate the methods, theories, applications, and future trends in model merging direction, providing a comprehensive classification of relevant approaches. In particular, this paper enhances the comprehensive understanding of model merging by covering three main aspects:

First, how are existing model merging methods classified?
We first propose a new taxonomy in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (upper part) that divides existing model merging methods into two phases (§[2](#S2 "2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")): pre-merging and during-merging.
(i) Pre-merging methods aim to create better conditions for merging. It is further divided into using linearized fine-tuning to achieve weight space and input space disentanglement, performing architectural transformations to convert heterogeneous models into homogeneous models, and aligning weights to place them in the same basin.
(ii) During-merging methods focus on designing sophisticated techniques to merge multiple models into one. These methods address task conflict and interference problems when merging models. They can be further divided into basic merging methods that perform the simplest parameter merging strategy; weighted merging methods that merge multiple models according to the importance calculated by specific rules; subspace merging methods that project multiple models into sparse subspaces for merging; routing-based methods that dynamically merge models according to input samples during inference; and the post-calibration based method that corrects the merged model. In addition to these methods, we also discuss the theoretical or empirical analysis of model merging.

{forest}

forked edges,
for tree=
grow=east,
reversed=true,
anchor=base west,
parent anchor=east,
child anchor=west,
base=left,
font=,
rectangle,
draw=hidden-draw,
rounded corners,
align=left,
minimum width=4em,
edge+=darkgray, line width=1pt,
s sep=3pt,
inner xsep=2pt,
inner ysep=3pt,
ver/.style=rotate=90, child anchor=north, parent anchor=south, anchor=center,
,
[
Model Merging: Methods, Theories, Applications
, ver,
color=hidden-draw, fill=mygray!100,
text width=21.8em,
text=black
[
Methods (§[2](#S2 "2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myred!80, text width=8.0em, text=black
[
Pre-Merging Mehtods (§[2.2](#S2.SS2 "2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myred!60, text width=14em, text=black
[
Linearization Fine-tuning (§[2.2.1](#S2.SS2.SSS1 "2.2.1 Linearization Fine-tuning ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Architecture Transformation (§[2.2.2](#S2.SS2.SSS2 "2.2.2 Architecture Transformation ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Weight Alignment (§[2.2.3](#S2.SS2.SSS3 "2.2.3 Weight Alignment ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
,
color=hidden-draw, fill=myred!40, text width=23em, text=black
]
]
[
During-Merging Methods (§[2.3](#S2.SS3 "2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myred!60, text width=14em, text=black
[
Basic Merging Methods (§[2.3.1](#S2.SS3.SSS1 "2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Weighted-based Merging Methods (§[2.3.2](#S2.SS3.SSS2 "2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Subspace-based Merging Methods (§[2.3.3](#S2.SS3.SSS3 "2.3.3 Subspace-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Routing-based Merging Methods (§[2.3.4](#S2.SS3.SSS4 "2.3.4 Routing-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Post-calibration based Methods (§[2.3.5](#S2.SS3.SSS5 "2.3.5 Post-calibration based Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myred!40, text width=23em, text=black
]
]
[
Theories and Analysis (§[2.4](#S2.SS4 "2.4 Theories and Analysis of Model Merging ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myred!60, text width=14em, text=black
]
]
[
Applications (§[3](#S3 "3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")-§[4](#S4 "4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myblue!80, text width=8.0em, text=black
[
Large Language Models (§[3.1](#S3.SS1 "3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), , fill=myblue!60, text width=17em, text=black
[
Human Value Alignment for LLMs (§[3.1.1](#S3.SS1.SSS1 "3.1.1 Human Preference Alignment for LLMs ‣ 3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Detoxifcation of LLMs (§[3.1.2](#S3.SS1.SSS2 "3.1.2 Detoxifcation of LLMs ‣ 3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Knowledge Unlearning of LLMs (§[3.1.3](#S3.SS1.SSS3 "3.1.3 Knowledge Unlearning of LLMs ‣ 3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Faster Training of LLMs (§[3.1.4](#S3.SS1.SSS4 "3.1.4 Faster Training of LLMs ‣ 3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Combine the Capabilities of Expert LLMs (§[3.1.5](#S3.SS1.SSS5 "3.1.5 Combine the Capabilities of Expert LLMs ‣ 3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
,
color=hidden-draw, fill=myblue!40, text width=20em, text=black
]
]
[
Multimodal Large Language Models (§[3.2](#S3.SS2 "3.2 Model Merging in Multimodal Large Language Models (MLLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), , fill=myblue!60, text width=17em, text=black
[
Multimodal Fusion (§[3.2.1](#S3.SS2.SSS1 "3.2.1 Model Merging for Multimodal Fusion ‣ 3.2 Model Merging in Multimodal Large Language Models (MLLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"));
  
Cross-modal Knowledge Transfer (§[3.2.2](#S3.SS2.SSS2 "3.2.2 Model Merging for Cross-Modal Knowledge Transfer ‣ 3.2 Model Merging in Multimodal Large Language Models (MLLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myblue!40, text width=20em, text=black
]
]
[
Image Generative Models (§[3.3](#S3.SS3 "3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), , fill=myblue!60, text width=17em, text=black
[
Style Mixing (§[3.3.1](#S3.SS3.SSS1 "3.3.1 Style Mixing in Generative Models ‣ 3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Reducing Training Cost (§[3.3.2](#S3.SS3.SSS2 "3.3.2 Reducing Training Cost of Generative Models ‣ 3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Enhancing the Faithfulness (§[3.3.3](#S3.SS3.SSS3 "3.3.3 Enhancing the Faithfulness of Generative Models ‣ 3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myblue!40, text width=20em, text=black
]
]
[
Continual Learning (§[4.1](#S4.SS1 "4.1 Model Merging in Continual Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=15em, text=black
[
Mitigate Catastrophic Forgetting (§[4.1.1](#S4.SS1.SSS1 "4.1.1 Model Merging to Mitigate Catastrophic Forgetting ‣ 4.1 Model Merging in Continual Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=22em, text=black
]
]
[
Multi-Task/Domain/Objective/Auxiliary Learning (§[4.2](#S4.SS2 "4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=22em, text=black
[
Knowledge Transfer in MTL (§[4.2.1](#S4.SS2.SSS1 "4.2.1 Knowledge Transfer in Multi-Task Learning ‣ 4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Knowledge Transfer in MOO (§[4.2.2](#S4.SS2.SSS2 "4.2.2 Knowledge Transfer in Multi-Objective Optimization ‣ 4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Knowledge Transfer in MDL (§[4.2.3](#S4.SS2.SSS3 "4.2.3 Knowledge Transfer in Multi-Domain Learning ‣ 4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Knowledge Transfer in ATL (§[4.2.4](#S4.SS2.SSS4 "4.2.4 Knowledge Transfer in Auxiliary Task Learning ‣ 4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=15em, text=black
]
]
[
Out-of-Distribution/Domain Generalization (§[4.3](#S4.SS3 "4.3 Model Merging in Out-of-Distribution/Domain Generalization ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=20.5em, text=black
[
Better OOD Generalization (§[4.3.1](#S4.SS3.SSS1 "4.3.1 Model Merging for Better Out-of-Distribution Generalization ‣ 4.3 Model Merging in Out-of-Distribution/Domain Generalization ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Better DG Generalization (§[4.3.2](#S4.SS3.SSS2 "4.3.2 Model Merging for Better Domain Generalization ‣ 4.3 Model Merging in Out-of-Distribution/Domain Generalization ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=16.5em, text=black
]
]
[
Federated Learning (§[4.4](#S4.SS4 "4.4 Model Merging in Federated Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=15em, text=black
[
Local Knowledge Aggregation (§[4.4.2](#S4.SS4.SSS2 "4.4.2 Model Merging for Local Knowledge Aggregation ‣ 4.4 Model Merging in Federated Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=22em, text=black
]
]
[
Zero-shot/Few-Shot Learning (§[4.5](#S4.SS5 "4.5 Model Merging in Zero-shot/Few-shot Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=15em, text=black
[
Zero-shot Knowledge Transfer (§[4.5.1](#S4.SS5.SSS1 "4.5.1 Model Merging for Cross-task Generalization in Zero-shot Learning ‣ 4.5 Model Merging in Zero-shot/Few-shot Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Few-shot Knowledge Transfer (§[4.5.2](#S4.SS5.SSS2 "4.5.2 Model Merging for Cross-task Generalization in Few-shot Learning ‣ 4.5 Model Merging in Zero-shot/Few-shot Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=22em, text=black
]
]
[
Adversarial Learning (§[4.6](#S4.SS6 "4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), fill=myyellow!60, text width=15em, text=black
[
Model Attack (§[4.6.1](#S4.SS6.SSS1 "4.6.1 Model Merging as an Attack Strategy ‣ 4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
  
Model Defense and Copyright Protection (§[4.6.2](#S4.SS6.SSS2 "4.6.2 Model Merging as a Defense Strategy ‣ 4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"))
, color=hidden-draw, fill=myyellow!40, text width=22em, text=black
]
]
]
]

Figure 2: The taxonomy of model merging in machine learning. This general framework covers advanced model merging methods and theories (top part), as well as practical applications of model merging techniques to foundation models and more than 10 machine learning subfields (bottom part).

Second, which applications can benefit from model merging?
We discuss in detail the various use cases of model merging in foundation models (§[3](#S3 "3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) and over ten subfields of machine learning (§[4](#S4 "4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")). As shown in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (lower part), model merging can be applied to a variety of foundation models, including large language models, multimodal large language models, and image generative models. For example, model merging in large language models can help mitigate untruthfulness and toxicity output, accomplish knowledge unlearning, and speed up training. Moreover, model merging also arises in different machine learning subfields, such as continual learning, multi-task/multi-domain learning, few-shot learning, and other subfields, to solve a variety of challenges. For instance, in continual learning, model merging can mitigate catastrophic forgetting of old tasks. In multi-task learning, multi-objective learning and multi-domain learning, it facilitates knowledge transfer. Additionally, in adversarial learning, model merging can be employed for both attack and defense strategies.

Third, what are the remaining challenges and future research opportunities for model merging?
Despite the advancements in merging methods and their well-developed applications, there are still numerous open challenges and future research directions in the field (§[5](#S5 "5 Remaining Challenges and Future Directions ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")). For example, as the number of tasks increases, the performance gap between existing methods and independent expert models becomes significantly larger. Additionally, current model merging methods incur enormous memory costs during merging and lack trust guarantees as well as in-depth theoretical analysis. Addressing these gaps will require substantial efforts from researchers to further advance the flourishing development of this field.

To summarize, the main contributions of this paper include the following three aspects:

* •

  Methodology Overview: We provide a comprehensive summary of the technical aspects of model merging. Specifically, we propose a new taxonomy that divides existing model merging methods into two stages and further subdivides the methods in each stage according to key techniques. Additionally, we discuss theoretical analysis work related to model merging.
* •

  Application Overview: We offer a comprehensive summary of the application aspects of model merging. Specifically, we explore the application of model merging to foundation models and 10+limit-from1010+ machine learning subfields, demonstrating how model merging can address existing challenges in these areas.
* •

  Future Directions: We outline several remaining challenges and future directions for model merging. We believe that model merging needs to be further explored in the future from the perspectives of performance gap, theoretical analysis, trustworthy guarantees, cross-disciplinary applications, etc.

The main structure of this paper is as follows: §[1](#S1 "1 Introduction ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") is an introduction, and §[2](#S2 "2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") offers a comprehensive discussion of advanced model merging methods from a technical perspective. In §[3](#S3 "3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[4](#S4 "4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), we summarize the applications of model merging in various foundation models and different subfields within machine learning, respectively. The remaining challenges and future research directions are discussed in §[5](#S5 "5 Remaining Challenges and Future Directions ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"). Finally, we conclude this paper in §[6](#S6 "6 Conclusions ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities").

## 2 Advanced Model Merging Methods

In this section, we first introduce the notation and problem definition of model merging in §[2.1](#S2.SS1 "2.1 Notation and Model Merging Problem Definition ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"). We then elaborate on advanced model merging methods (Table [1](#S2.T1 "Table 1 ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") summarizes the primary purpose of each category of methods).
Existing model merging techniques can be roughly divided into the following two categories: (i) Before Merging Methods in §[2.2](#S2.SS2 "2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"): it provides better prior knowledge for model merging. (ii) During Merging Methods in §[2.3](#S2.SS3 "2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"): it resolves task conflict/interference by various strategies, and then performs parameter merging operations.
Finally, we conclude with theories or explanations for the effectiveness of model merging in §[2.4](#S2.SS4 "2.4 Theories and Analysis of Model Merging ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities").

Table 1: A summary of existing model merging methods.

|  |  |
| --- | --- |
| Methods | The Goal or Main Idea of the Methods |
| Linearization Fine-tuning (§[2.2.1](#S2.SS2.SSS1 "2.2.1 Linearization Fine-tuning ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Disentangling different models in input space and weight space |
| Architecture Transformation (§[2.2.2](#S2.SS2.SSS2 "2.2.2 Architecture Transformation ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Transform multiple heterogeneous models into homogeneous models |
| Weight Alignment (§[2.2.3](#S2.SS2.SSS3 "2.2.3 Weight Alignment ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Repermutate multiple models into the same basin |
| Basic Merging Methods (§[2.3.1](#S2.SS3.SSS1 "2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Simple weighted averaging or task-arithmetic based merging |
| Weighted-based Merging Methods (§[2.3.2](#S2.SS3.SSS2 "2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Merge multiple models based on model/parameter importance weights |
| Subspace-based Merging Methods (§[2.3.3](#S2.SS3.SSS3 "2.3.3 Subspace-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Merge multiple models by projecting them into a sparse subspace |
| Routing-based Merging Methods (§[2.3.4](#S2.SS3.SSS4 "2.3.4 Routing-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Dynamically merge multiple models based on input during the inference phase |
| Post-calibration-based Merging Methods (§[2.3.5](#S2.SS3.SSS5 "2.3.5 Post-calibration based Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Calibrating the merged model to be closer to the individual models reduces the knowledge loss |

### 2.1 Notation and Model Merging Problem Definition

Assume there are T𝑇T models (ΦΘ(1),…,ΦΘ(T)

subscriptΦsuperscriptΘ1…subscriptΦsuperscriptΘ𝑇\Phi\_{\Theta^{(1)}},\ldots,\Phi\_{\Theta^{(T)}}) of the same architecture that need to be merged, and they train from scratch or fine-tune on the same pre-trained model ΦΘ(0)subscriptΦsuperscriptΘ0\Phi\_{\Theta^{(0)}} respectively. The parameters (or weights) of the t𝑡t-th model ΦΘ(t)subscriptΦsuperscriptΘ𝑡\Phi\_{\Theta^{(t)}} are represented as Θ(t)={Θl(t)}l=1LsuperscriptΘ𝑡superscriptsubscriptsubscriptsuperscriptΘ𝑡𝑙𝑙1𝐿\small\Theta^{(t)}=\{\Theta^{(t)}\_{l}\}\_{l=1}^{L}, where l𝑙l denotes the l𝑙l-th layer of the model, and L𝐿L is the total number of layers.

In this survey, we focus on parameter-wise merging. In other words, the goal of model merging is to merge the parameters {Θ(1),…,Θ(T)}superscriptΘ1…superscriptΘ𝑇\small\{\Theta^{(1)},\ldots,\Theta^{(T)}\}, and finally obtain the new parameters Θ(m​e​r​g​e)=merge​(Θ(1),…,Θ(T))superscriptΘ𝑚𝑒𝑟𝑔𝑒mergesuperscriptΘ1…superscriptΘ𝑇\small\Theta^{(merge)}=\texttt{merge}(\Theta^{(1)},\ldots,\Theta^{(T)}).
One straightforward solution for merging models is weighted averaging [[168](#bib.bib168), [146](#bib.bib146)], defined as Θ(m​e​r​g​e)=1T​∑t=1TΘ(t)superscriptΘ𝑚𝑒𝑟𝑔𝑒1𝑇superscriptsubscript𝑡1𝑇superscriptΘ𝑡\small\Theta^{(merge)}=\frac{1}{T}\sum\_{t=1}^{T}\Theta^{(t)}. However, the performance of this approach is often unacceptably poor or infeasible due to several possible factors: (i) The lack of suitable merging conditions, such as multiple models not being in the same basin or having inconsistent architectures. (ii) There are conflicts and interference among multiple models. We illustrate how advanced methods address these issues in §[2.2](#S2.SS2 "2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[2.3](#S2.SS3 "2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), respectively.

### 2.2 Pre-Merging Methods

To provide better preconditions for model merging, one class of work focuses on the fine-tuning step of independent models, such as fine-tuning the linearized model instead of the nonlinear model (in §[2.2.1](#S2.SS2.SSS1 "2.2.1 Linearization Fine-tuning ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")). Additionally, when multiple model architectures that need to be merged are inconsistent, they must be pre-transformed to the same architecture (in §[2.2.2](#S2.SS2.SSS2 "2.2.2 Architecture Transformation ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")). Finally, another class of work attempts to align the weights/parameters before merging (in §[2.2.3](#S2.SS2.SSS3 "2.2.3 Weight Alignment ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")).

#### 2.2.1 Linearization Fine-tuning

Ortiz-Jimenez et al. [[123](#bib.bib123)] reveal that one necessary condition for effective model merging is ‘weight disentanglement’. This means that different directions of the weight space correspond to functional changes in disjoint regions of the input space. For example, if model 111 in weight space corresponds to a function change on 𝒟1subscript𝒟1\small\mathcal{D}\_{1} in input space and model 2 in weight space corresponds to a function change on 𝒟2subscript𝒟2\small\mathcal{D}\_{2} in input space, then the merged model will not interfere with each other in terms of function change. In other words, two models satisfying this weight disentanglement property can coexist in one model without affecting their respective performance, which is a very attractive property.

To achieve weight disentanglement, Ortiz-Jimenez et al. [[123](#bib.bib123)] propose fine-tuning the linearized model along the tangent space [[68](#bib.bib68)] of the pre-trained model during the fine-tuning stage, rather than in the original space of the nonlinear model.
However, linearized fine-tuning with all parameters is more expensive than nonlinear fine-tuning. To accelerate this process, some works suggest linearizing only part of the layers. For example, Tang et al. [[160](#bib.bib160)] propose partially linearizing the Adapter modules and then merging Adapters. Jin et al. [[77](#bib.bib77)] suggest linearly fine-tuning only the linear layers in the attention modules of the full model. Furthermore, TAFT [[105](#bib.bib105)] develops an efficient linearization method for the Transformer [[169](#bib.bib169)] architectures, which directly derives closed-form linearized solutions for transformer networks.
In summary, fine-tuning in the tangent space makes it easier to disentanglement the input space and weight space, thereby reducing the interference in subsequent model merging.

#### 2.2.2 Architecture Transformation

In some cases, models that need to be merged may have different architectures and cannot be merged directly. To solve this problem, some studies [[10](#bib.bib10), [171](#bib.bib171), [172](#bib.bib172), [120](#bib.bib120)] propose to perform architecture transformation before merging, that is, transform multiple models with different architectures into the same architecture as shown in Figure [3](#S2.F3 "Figure 3 ‣ 2.2.2 Architecture Transformation ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (a).
For example, GAN Cocktail [[10](#bib.bib10)] attempts to merge multiple GAN models {Θ(1),…,Θ(T)}superscriptΘ1…superscriptΘ𝑇\{\Theta^{(1)},\ldots,\Theta^{(T)}\} with different architectures. It transforms all GAN models Θ(t)superscriptΘ𝑡\small\Theta^{(t)} (t∈{1,2,…,T}&t≠k𝑡12…𝑇𝑡𝑘t\in\{1,2,\ldots,T\}\;\&\;t\!\neq\!k) into a specified target model Θ(k)superscriptΘ𝑘\Theta^{(k)}, that is, Θ(k)superscriptΘ𝑘\small\Theta^{(k)} is used as the initialization to learn the output of Θ(t)superscriptΘ𝑡\small\Theta^{(t)}, while adding implicit regularizations to ensure that Θ(k)superscriptΘ𝑘\small\Theta^{(k)} does not forget knowledge of the task k𝑘k. Consequently, the transformed GAN models have the same structure and shared knowledge, facilitating further model merging. Similarly, FuseChat [[172](#bib.bib172)] proposes to merge chat LLMs with diverse architectures and scales (e.g., NH2-Mixtral-8x7B [[75](#bib.bib75)], NH2-Solar-10.7B [[84](#bib.bib84)], OpenChat-3.5-7B [[173](#bib.bib173)] in their practical applications). Specifically, FuseChat first uses knowledge distillation to transform all the architectures to match that of OpenChat-3.5-7B, and then performs the model merge operation.
Unlike the above distillation-based approach, CLAFusion [[121](#bib.bib121)] adds layers/blocks (with weights set to the identity matrix) to the smaller model to align its architecture with that of the larger model.
In summary, merging models with different architectures requires first transforming all models into a common architecture to merge later.

!(/html/2408.07666/assets/x2.png)

Figure 3: (a) An illustration of an architectural transformation that transforms multiple heterogeneous models into homogeneous models, allowing subsequent direct parameter-level merge operations to be performed. (b) An illustration of the weights/parameters alignment, that is, permuting the neural network model Θ(1)superscriptΘ1\small\Theta^{(1)} so that it aligns with the model Θ(2)superscriptΘ2\small\Theta^{(2)}.

#### 2.2.3 Weight Alignment

The linear mode connectivity (LMC) property of deep neural networks demonstrates that there is a connected path between multiple local minima of deep neural networks along which the loss remains nearly constant [[47](#bib.bib47), [37](#bib.bib37), [162](#bib.bib162), [38](#bib.bib38), [39](#bib.bib39)]. Numerous studies [[43](#bib.bib43), [117](#bib.bib117), [38](#bib.bib38)] have shown that two independent models, starting from the same pre-trained model and fine-tuned with different hyper-parameter configurations, typically satisfy LMC. Further, Adilova et al. [[3](#bib.bib3)] and Zhou et al. [[216](#bib.bib216)] extended the study of LMC to the layer level. The LMC property implies that multiple local minima may be equivalent in the weight space, and different weight configurations of the same model may represent the same functionality. Inspired by this, many works proposed to permute the weights of one model (i.e., Θ(1)→Π​(Θ(1))→superscriptΘ1ΠsuperscriptΘ1\small\Theta^{(1)}\rightarrow\Pi(\Theta^{(1)})) to align with the other model Θ(2)superscriptΘ2\Theta^{(2)} when merging/interpolating two separate models, as illustrated in Figure [3](#S2.F3 "Figure 3 ‣ 2.2.2 Architecture Transformation ‣ 2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (b). Π​(⋅)Π⋅\Pi(\cdot) denotes a permutation function, and researchers have dedicated efforts to studying effective and efficient permutation strategies for model alignment.

OTFusion [[148](#bib.bib148)] and Imfeld et al. [[66](#bib.bib66)] adopt optimal transport to soft-align neurons across models. NeuronAlignment [[162](#bib.bib162)] introduces an inexpensive heuristic algorithm to approximate the optimal neuron alignment. CCAMerge [[58](#bib.bib58)] permutes by maximizing the correlation between linear combinations of neurons. Notably, Git re-basin [[5](#bib.bib5)] proposes three methods –activation matching, weight matching, and straight-through estimation– to align (or permute) the weights of models trained on different tasks. Based on the Git re-basin, Peña et al. [[125](#bib.bib125)] further incorporate a Sinkhorn-based projection to improve these alignment methods. In addition, MuDSC [[189](#bib.bib189)] proposes simultaneously performing model alignment in weight and activation spaces. Unlike heuristic alignment strategies, Deep-Align [[119](#bib.bib119)] proposes a learning-based approach to weight alignment, employing a novel learnable architecture that takes two sets of weights as input and outputs a permutation matrix for alignment.

Despite the significant improvement of these alignment algorithms, Jordan et al. [[80](#bib.bib80)] argue that the success of these methods depends on the use of normalization layers (e.g., BatchNorm, LayerNorm, etc.) in the model; without these, the performance of the matching algorithms is greatly reduced. The authors call this the “variance collapse” problem and propose the REPAIR method to solve it. Additionally, Crisostomi et al. [[27](#bib.bib27)] noted that previous pairwise permutations do not guarantee cycle consistency, making the alignment fragile. They further proposed to globally optimize the permutations of all layers simultaneously at each step.
Overall, aligned models experience much less interference or conflict during merging compared to directly merging unaligned models.

### 2.3 During Merging Methods

In this section, we provide a detailed discussion on how to merge a set of well-trained models. The existing methods can be roughly divided into five categories: basic merging methods (§[2.3.1](#S2.SS3.SSS1 "2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), weighted-based merging methods (§[2.3.2](#S2.SS3.SSS2 "2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), subspace-based merging methods (§[2.3.3](#S2.SS3.SSS3 "2.3.3 Subspace-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), routing-based merging methods (§[2.3.4](#S2.SS3.SSS4 "2.3.4 Routing-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")), and post-calibration based methods (§[2.3.5](#S2.SS3.SSS5 "2.3.5 Post-calibration based Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")).

#### 2.3.1 Basic Merging Methods

One of the most straightforward approaches to model merging is to directly weighted average the parameters of multiple models [[168](#bib.bib168), [146](#bib.bib146)], i.e., Θ(m​e​r​g​e)=∑t=1T1T​Θ(t)superscriptΘ𝑚𝑒𝑟𝑔𝑒superscriptsubscript𝑡1𝑇1𝑇superscriptΘ𝑡\small\Theta^{(merge)}=\sum\_{t=1}^{T}\frac{1}{T}\Theta^{(t)}. However, the performance of simple weight averaging is generally unsatisfactory.
Recently, Task Arithmetic [[65](#bib.bib65)] introduced the concept of “task vector” (in Figure [4](#S2.F4 "Figure 4 ‣ 2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(a)), which represents the model parameter Θ(t)superscriptΘ𝑡\Theta^{(t)} fine-tuned on task t𝑡t subtract the pre-trained model parameter Θ(0)superscriptΘ0\Theta^{(0)}, i.e., τt=Θ(t)−Θ(0)subscript𝜏𝑡superscriptΘ𝑡superscriptΘ0\tau\_{t}=\Theta^{(t)}-\Theta^{(0)}.
In other words, task vectors are thought to steer the behavior of a neural network meaningfully.
For example, multitask learning (MTL) can be accomplished by adding task vectors, forgetting can be achieved by subtracting task vectors, and task analogies can be performed using analogous task vectors. Specifically, when we want the pretrained model to perform MTL, we can add multiple task vectors {τ1,…,τT}subscript𝜏1…subscript𝜏𝑇\{\tau\_{1},\ldots,\tau\_{T}\} to the pretrained model, i.e., Θ(m​e​r​g​e)=Θ(0)+λ⋅∑t=1TτtsuperscriptΘ𝑚𝑒𝑟𝑔𝑒superscriptΘ0⋅𝜆superscriptsubscript𝑡1𝑇subscript𝜏𝑡\small\Theta^{(merge)}=\Theta^{(0)}+\lambda\cdot\sum\_{t=1}^{T}\tau\_{t} in Figure [4](#S2.F4 "Figure 4 ‣ 2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(b), where λ𝜆\lambda is a hyperparameter. Conversely, when we want the pretrained model to forget a function t𝑡t, we can subtract the corresponding task vector from pretrained model as Figure [4](#S2.F4 "Figure 4 ‣ 2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(c), i.e., Θ(m​e​r​g​e)=Θ(0)−τtsuperscriptΘ𝑚𝑒𝑟𝑔𝑒superscriptΘ0subscript𝜏𝑡\small\Theta^{(merge)}=\Theta^{(0)}-\tau\_{t}. As shown in Figure [4](#S2.F4 "Figure 4 ‣ 2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(d), we can also implement task analogies by task vector analogies, thus enabling zero-shot learning of new tasks. Similarly, PEMs [[209](#bib.bib209)] combines Adapters with different capabilities by extending task arithmetic [[65](#bib.bib65)] to parameter-efficient fine-tuning settings. However, the performance of basic merging methods is not satisfactory most of the time, especially when the tasks interfere with each other.

!(/html/2408.07666/assets/x3.png)

Figure 4: An illustration of Task Arithmetic [[65](#bib.bib65)]. (a) Definition of the “task vector”, which is the difference between the fine-tuned model and the pre-trained model. (b) Multi-task learning is performed by merging multiple task vectors. (c) Knowledge forgetting is achieved by subtracting the task vector. (d) Analogical task vectors are used to implement task analogies.

#### 2.3.2 Weighted-based Merging Methods

As we all know, different models (or task vectors) represent different functions, and intuitively, different functions have varying degrees of importance. Therefore, advanced weighted-based model merging methods design various clever rules to determine the merging coefficients, as shown in Figure [5](#S2.F5 "Figure 5 ‣ 2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(a).
For instance, when merging two models Θ(1)superscriptΘ1\Theta^{(1)} and Θ(2)superscriptΘ2\Theta^{(2)} (or task vectors τ1subscript𝜏1\tau\_{1} and τ2subscript𝜏2\tau\_{2}), the goal of the weighted merging method is to find the optimal coefficients λ1∗superscriptsubscript𝜆1\lambda\_{1}^{\*} and λ2∗superscriptsubscript𝜆2\lambda\_{2}^{\*} so that the merged model Θ(m​e​r​g​e)=λ1∗​Θ(1)+λ2∗​Θ(2)superscriptΘ𝑚𝑒𝑟𝑔𝑒superscriptsubscript𝜆1superscriptΘ1superscriptsubscript𝜆2superscriptΘ2\small\Theta^{(merge)}=\lambda\_{1}^{\*}\Theta^{(1)}+\lambda\_{2}^{\*}\Theta^{(2)} (or Θ(m​e​r​g​e)=Θ(0)+λ1∗​τ1+λ2∗​τ2superscriptΘ𝑚𝑒𝑟𝑔𝑒superscriptΘ0superscriptsubscript𝜆1subscript𝜏1superscriptsubscript𝜆2subscript𝜏2\small\Theta^{(merge)}=\Theta^{(0)}+\lambda\_{1}^{\*}\tau\_{1}+\lambda\_{2}^{\*}\tau\_{2}) can retain the capabilities of the independent models as much as possible. However, when the number of models is large, it is impractical to use brute-force grid search to find the optimal merging coefficient because of the expensive search cost involved.

To determine the merging coefficient more effectively, Evolutionary-model-merge [[6](#bib.bib6)] and Checkpoint Merging [[100](#bib.bib100)] efficient searches for the merging coefficients using evolutionary algorithms and Bayesian optimization, respectively. AdaMerging [[194](#bib.bib194)] uses gradient descent optimization to learn the merging coefficients by minimizing entropy as a surrogate loss in unlabeled test data.
MetaGPT [[215](#bib.bib215)] casts the model merging problem as an MTL formalism, where the goal is to minimize the average loss of the merged model and the independent model. It employs local linearization of the model and the orthogonality of the task vectors to derive the optimal merging coefficient λt∗superscriptsubscript𝜆𝑡\lambda\_{t}^{\*} for each model τtsubscript𝜏𝑡\tau\_{t} as follows: λt∗=‖τt‖2/∑k=1T‖τk‖2superscriptsubscript𝜆𝑡superscriptnormsubscript𝜏𝑡2superscriptsubscript𝑘1𝑇superscriptnormsubscript𝜏𝑘2\small\lambda\_{t}^{\*}={\left\|\tau\_{t}\right\|^{2}}/{\sum\_{k=1}^{T}\left\|\tau\_{k}\right\|^{2}}.
SLERP [[49](#bib.bib49)] performs spherical interpolation of the parameters of the two models. The interpolated coefficients of τ1subscript𝜏1\tau\_{1} and τ2subscript𝜏2\tau\_{2} are given by λ1∗=sin⁡((1−λ)⋅ρ)sin⁡(ρ)superscriptsubscript𝜆1⋅1𝜆𝜌𝜌\small\lambda\_{1}^{\*}=\frac{\sin\left(\left(1-\lambda\right)\cdot\rho\right)}{\sin(\rho)} and sin⁡(λ⋅ρ)sin⁡(ρ)⋅𝜆𝜌𝜌\small\frac{\sin(\lambda\cdot\rho)}{\sin(\rho)}, respectively, where ρ=arccos⁡τ1⋅τ2|τ1|⋅|τ2|𝜌⋅subscript𝜏1subscript𝜏2⋅subscript𝜏1subscript𝜏2\small\rho=\arccos\frac{\tau\_{1}\cdot\tau\_{2}}{\left|\tau\_{1}\right|\cdot\left|\tau\_{2}\right|} denotes the angle between the two task vectors, and λ𝜆\lambda represents the merging coefficient of the initial setting.

The above-sophisticated weighting methods operate at the model (or task) level. It is well known that each layer and even each neuron in a deep neural network model play a significantly different role, and some research has developed more fine-grained weighted merging strategies. For example, Layer-wise AdaMerging [[194](#bib.bib194)] and aTLAS [[205](#bib.bib205)] adaptively learn different sets of merging coefficients for each layer or module of the model, respectively. RegMean [[78](#bib.bib78)] indicates that closed-form solutions (relying on the data statistics provided by the training set) exist for linear layers in model merging, while nonlinear layers can simply perform weight averaging.
Other works utilize the Fisher information matrix [[40](#bib.bib40)] to assess the importance of parameters when merging. Fisher-Merging [[113](#bib.bib113)] performs model merging based on the importance of the parameters in each independent model, that is, Θ(merge)=∑t=1TF(t)​Θ(t)/∑t=1TF(t)superscriptΘmergesuperscriptsubscript𝑡1𝑇superscript𝐹𝑡superscriptΘ𝑡superscriptsubscript𝑡1𝑇superscript𝐹𝑡\small\Theta^{(\text{merge})}=\sum\_{t=1}^{T}F^{(t)}\Theta^{(t)}/\sum\_{t=1}^{T}F^{(t)}, where F(t)superscript𝐹𝑡F^{(t)} is the diagonal of the Fisher information matrix with respect to task t𝑡t.
Fisher-nodes-merging [[164](#bib.bib164)] also combines a set of Transformer [[169](#bib.bib169)] models based on the Fisher information matrix. MaTS [[155](#bib.bib155)] developed a block diagonal approximation for Fisher merging.
Daheim et al. [[29](#bib.bib29)] linked the inaccuracy of weighted average with gradient mismatch, and further proposed an uncertainty-based algorithm to reduce the matching error, ultimately merging the models based on a second-order Hessian estimation.

!(/html/2408.07666/assets/x4.png)

Figure 5: (a) An illustration of weighted-based model merging methods. (b) An illustration of the subspace-based merging method, where empty means zero value. (c) An illustration of the routing-based merge method dynamically performs the model merge based on the input.

#### 2.3.3 Subspace-based Merging Methods

Another class of advanced methods transforms models into sparse subspaces for merging, thereby mitigating task interference.
The over-parameterized nature of neural networks and the success of model pruning [[54](#bib.bib54), [22](#bib.bib22)] show that removing most of the parameters from the model barely affects its accuracy [[190](#bib.bib190)]. This insight opens up new opportunities for model merging, allowing us to remove insignificant neurons from a single model and merge multiple sparse models within the parameter subspace, as shown in Figure [5](#S2.F5 "Figure 5 ‣ 2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (b).

TIES-Merging [[190](#bib.bib190)] proposes to trim each individual model based on parameter magnitudes, retaining only the top 20% of parameters with the highest magnitudes. It further suggests eliminating parameter sign conflicts to reduce interference, and finally merging sparse models using Task Arithmetic [[65](#bib.bib65)]. Similarly, Drop And REscale (DARE) [[200](#bib.bib200)] also sparsifies by parameter magnitude, and highlights the importance of further performing rescaling on sparse models. In addition to removing the tail parameters with the smallest weight, the Model Breadcrumbs [[30](#bib.bib30)] highlight the importance of removing the parameters (outliers) with the largest weights to further reduce noise in model merging and enhance generalization to hyperparameters. TALL-masks [[176](#bib.bib176)] creates a mask matrix specific to each task based on a predefined threshold related to independent models. Further, Model Tailor [[218](#bib.bib218)] masks unimportant parameters based on the sensitivity of fine-tuned parameters to loss changes and the significance of changes compared to pre-trained parameters.
Unlike the standard practice of obtaining a single model through model merging, EMR-Merging [[62](#bib.bib62)] proposes maintaining a shared model among multiple tasks alongside a sparse task-specific model. In this approach, the value of the shared model at each index is the largest parameter value among all models. In contrast to the mask construction rules of the aforementioned heuristics, Concrete [[156](#bib.bib156)] frames mask construction and model merging as a learnable bi-level optimization problem. The outer-level optimizes the mask matrix, while the inner-level merges the model based on the mask matrix and optimizes it using the unlabeled test samples.

#### 2.3.4 Routing-based Merging Methods

The basic, weighted-based, or subspace-based merging methods discussed in §[2.3.1](#S2.SS3.SSS1 "2.3.1 Basic Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), §[2.3.2](#S2.SS3.SSS2 "2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[2.3.3](#S2.SS3.SSS3 "2.3.3 Subspace-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") are static merging methods. This means that the merged model remains the same for all samples or tasks. Given that there are differences between input samples/tasks, the model’s ability may vary when processing different samples/tasks. As shown in Figure [5](#S2.F5 "Figure 5 ‣ 2.3.2 Weighted-based Merging Methods ‣ 2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") (c), some works propose to dynamically merge models (or subsets of layers) based on the samples/tasks [[93](#bib.bib93), [116](#bib.bib116), [159](#bib.bib159), [108](#bib.bib108)] during the inference phase.

For a given input, SMEAR [[116](#bib.bib116)] first computes a weighted average of the parameters of each expert by using the distribution of router inputs to the expert modules. The advantage of this approach is that it has a similar computational cost to that of a single expert. Twin-Merging [[108](#bib.bib108)] also adaptively combines task-shared and task-private knowledge based on routing during the inference phase.
Similarly, Weight-Ensembling MoE [[159](#bib.bib159)] proposes a dynamic merging Transformer architecture. Specifically, they observed that the parameters of the linear layer in the fine-tuned model changed more dramatically than those of the nonlinear layer, which also significantly impacted the merging performance. Therefore, they use a standard weighted average to merge all modules except the linear layer. The linear layer is dynamically weighted and merged according to the routing network (sample features as the input of the routing, and merging coefficients as the output) during inference. PWE MoE [[158](#bib.bib158)] further extends Weight-Ensembling MoE to a multi-objective optimization setting and uses the preference vector as input for routing.

#### 2.3.5 Post-calibration based Methods

Recently, Yang et al. [[193](#bib.bib193)] introduce a post-merging method to calibrate merged models. They observed that merged models (across multiple mainstream model merging methods) suffer from representation bias, meaning the representations extracted by the independent and merged models are very different, leading to performance degradation in the merged model. To alleviate this problem, they propose a module called ‘representation surgery’ to calibrate the representation bias. The core idea is to align the representation of the merged model after representation surgery with that of the independent model.

### 2.4 Theories and Analysis of Model Merging

In addition to designing various advanced methods in §[2.2](#S2.SS2 "2.2 Pre-Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[2.3](#S2.SS3 "2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), the theoretical and effectiveness analysis of model merging is also crucial. Currently, there is limited work on the theoretical analysis of model merging. Based on the source of the models to be merged, the existing theoretical analysis can be roughly divided into three categories: (i) model merging of different checkpoints in the same training trajectory, (ii) model merging of different models fine-tuned on the same dataset, and (iii) model merging of different models fine-tuned on different datasets or tasks.

First, some analyses target model merging on the single-trajectory training, usually referring to stochastic weighted average (SWA) or exponential moving average (EMA). For example, Jain et al. [[69](#bib.bib69)] theoretically proved that the excess risk of the EMA is an upper bound of a bias term and a variance term in the context of least squares regression. The bias term depends on the initialization state of the parameters and decreases exponentially with the number of iterations once the model starts averaging. The variance term depends on the noise covariance inherent in the data, which decays at a faster rate when model averaging is used [[8](#bib.bib8)]. Similarly, Rame et al. [[132](#bib.bib132)] applies bias-variance decomposition to the domain generalization setting to explain why model averaging improves out-of-distribution performance. In addition, Hardt et al. [[52](#bib.bib52)] provide a stability bound for SWA under convex assumptions, while Wang et al. [[177](#bib.bib177)] further establish generalization bounds analysis in both convex and nonconvex cases.

Second, some studies explain the merging of multiple models with different hyperparameter fine-tuning for the same dataset in terms of connectivity and flatness of the loss landscape. Specifically, some works apply the theory of linear mode connectivity (LMC) [[47](#bib.bib47), [37](#bib.bib37), [162](#bib.bib162)] of neural networks to explain model merging. LMC reveals that neural network loss minima are not isolated points in the weight space. Recent studies [[43](#bib.bib43), [117](#bib.bib117), [38](#bib.bib38), [217](#bib.bib217)] have shown that two independent models, starting from the same pre-trained model and fine-tuned with different configurations, usually satisfy LMC. In other words, LMC is a general phenomenon that typically appears in fine-tuned models based on the ”pretraining-finetuning” paradigm, which is the current standard in the machine learning community. Therefore, performing weight alignment according to LMC provides a robust validity guarantee for model merging [[5](#bib.bib5), [80](#bib.bib80)].
On the other hand, other studies explain model merging from the perspective of a flatter loss landscape [[88](#bib.bib88)], arguing that merging multiple weights fine-tuned under different optimization configurations with the same data usually converges to a flat local minimum [[41](#bib.bib41)], thus revealing why model merging has better generalization [[149](#bib.bib149), [67](#bib.bib67), [50](#bib.bib50), [206](#bib.bib206), [15](#bib.bib15)].

Finally, an analysis by Ortiz-Jimenez et al. [[123](#bib.bib123)] is based on multiple models fine-tuned on different datasets, identifying weight disentanglement as a necessary precondition for effective model merging. More specifically, Ortiz-Jimenez et al. [[123](#bib.bib123)] provide theoretical and empirical analyses of the neural tangent kernel (NTK) and establish a compelling link between the task arithmetic [[65](#bib.bib65)] and the spectral properties of NTK.

## 3 Application of Model Merging in Foundation Models

The emergence of foundation models, including large language models (LLMs), multimodal large language models (MLLMs), and image generative models, is a significant indicator of technological progress in the field of artificial intelligence in recent years. However, despite their advancements, these large models still face several challenges, such as generating harmful content in LLMs, MLLMs struggling with fusing information from different modalities, and the difficulty of producing mixed-style images in image generation models. Recent studies suggest that model merging techniques offer a promising solution to these inherent challenges in foundational models. Table [2](#S3.T2 "Table 2 ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") first briefly summarizes the application of model merging in foundational models. Then, §[3.1](#S3.SS1 "3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), §[3.2](#S3.SS2 "3.2 Model Merging in Multimodal Large Language Models (MLLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[3.3](#S3.SS3 "3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") provide a detailed discussion on how LLMs, MLLMs, and image generative models benefit from model merging, respectively.

Table 2: A summary of the application of model merging techniques in foundation models.

|  |  |
| --- | --- |
| Scenarios | The Main Purpose of Model Merging |
| Large Language Models (§[3.1](#S3.SS1 "3.1 Model Merging in Large Language Models (LLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Enhancing the domain-specific capabilities of pre-trained LLMs or editing old knowledge |
| Multimodal Large Language Models (§[3.2](#S3.SS2 "3.2 Model Merging in Multimodal Large Language Models (MLLMs) ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Understanding content across multiple modalities using a single model |
| Image Generative Models (§[3.3](#S3.SS3 "3.3 Model Merging in Image Generative Models ‣ 3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Generate images with multiple styles or achieve image-style transformation |

### 3.1 Model Merging in Large Language Models (LLMs)

In recent years, large language models (LLMs), such as GPT-4 [[2](#bib.bib2)], Gemini [[163](#bib.bib163)], PaLM [[23](#bib.bib23)] and LLaMA [[166](#bib.bib166)], have made significant advancements and have been widely applied across various tasks. Despite their superhuman performance on most basic tasks, LLMs still face numerous challenges, including producing toxic content that violates laws or ethics, using unauthorized data during training, high training costs, and insufficient performance in specific domains. Model merging technology presents a promising opportunity to address these challenges.

#### 3.1.1 Human Preference Alignment for LLMs

Humans often hold diverse opinions about aesthetics, politics, or fairness. When LLMs serve humans, different people have different expectations of the model, e.g., some expect LLMs to generate harmless responses, while others seek engaging and enjoyable interactions [[134](#bib.bib134)]. Consequently, the development of practical LLMs is generally divided into three stages, to generate responses that are more helpful, accurate, and safer [[107](#bib.bib107)]: Pre-training on a large amount of unsupervised data, supervised fine-tuning (SFT) on a small dataset with high-quality annotation, and interaction with humans to further optimize LLM alignment (e.g., direct preference optimization (DPO) [[131](#bib.bib131)] or reinforcement learning from human feedback (RLHF) [[219](#bib.bib219)]) with human preferences, rewards, or values.

Some works propose to achieve better, safer, or faster alignment of human preferences by model merging. For example, ExPO [[213](#bib.bib213)] adds a task vector, constructed by a moderate model aligned using DPO or RLHF on a small amount of human preference data, to an unaligned SFT model. A more powerful aligned model can be directly obtained by setting a suitable merging coefficient. On the AlpacaEval 2.0 benchmark [[97](#bib.bib97)], fusing a model aligned on the 10%/20% preference data with an SFT model results in performance comparable to that of a model aligned on the full preference data. DogeRM [[98](#bib.bib98)] proposed merging the reward model with LLMs fine-tuned on different downstream domains to create domain-private reward models directly. Additionally, Lu et al. [[107](#bib.bib107)] propose an Online Merging Optimizer, that interpolates the gradient with the SFT model at each step of RLHF. This approach encourages RLHF to optimize toward reward maximization while preventing LLMs from forgetting general knowledge due to RLHF.
Beyond preference alignment, several studies have examined the impact of model merging for secure alignment of LLMs [[11](#bib.bib11), [199](#bib.bib199), [51](#bib.bib51)]. For example, Hammoud et al. [[51](#bib.bib51)] find that merging two security-aligned models could compromise security. Thus, they proposed explicitly including secure alignment as an optimization objective when constructing synthetic data for model merging.

In practice, users often have various combinations of preferences rather than a single preference. Training a model separately for each combination of preferences is unrealistic due to the infinite combinations and the high training costs. Therefore, some studies suggest combining models with different reward alignments to create a series of integrated aligned LLMs.
For example, Rame et al. [[134](#bib.bib134)] and Jang et al. [[72](#bib.bib72)] propose Reward Soups and Personalized Soups, respectively, as efficient and flexible solutions for diverse rewards. Specifically, Rewarded Soups first trains an expert model for each reward and then linearly interpolates the weights of the experts to approximate the set of Pareto optimal solutions for various reward combinations. This approach is cost-effective, as it only requires training separate models for each reward to combine any variety of rewards.

#### 3.1.2 Detoxifcation of LLMs

LLMs have been widely noted for issues related to untruthfulness and toxicity in various applications [[60](#bib.bib60)], such as insults, threats, and profanity in responses to certain questions. To address the potential security risks in the application of LLMs, flexible techniques are needed to reduce the generation of toxic text, essentially detoxifying LLMs. A straightforward solution is to collect additional non-toxic data to fine-tune LLMs [[83](#bib.bib83)]; however, this approach requires significant computing resources and may interfere with the general capabilities of LLMs. Alternatively, directly reducing the probability of potentially toxic words during the decoding stage requires additional guidance information [[87](#bib.bib87)].
Recent studies have shown that reducing the toxic data generation of LLMs through model merging is a simple and effective scheme [[65](#bib.bib65), [209](#bib.bib209), [60](#bib.bib60)].

Task Arithmetic [[65](#bib.bib65)] negates the task vectors of GPT-2 model [[130](#bib.bib130)] fine-tuned on toxic data (Civil Comments [[13](#bib.bib13)]) and shows that this operation effectively reduces the proportion of data classified as ”toxic”, with little change in the fluency of the language on the control task (WikiText-103).
Additionally, some parameter-efficient models steer the toxic behavior of LLMs by manipulating a small number of parameters. PEM [[209](#bib.bib209)] negates LoRA [[59](#bib.bib59)] (and (IA)3 [[102](#bib.bib102)]) modules trained on poisoning data to maintain language proficiency while reducing toxicity of language model output. Ethos [[46](#bib.bib46)] and Ext-Sub [[60](#bib.bib60)] point out that while the task vector on toxic data is factually wrong, it also contains correct information about language modeling and logical narrative skills. Therefore, Ext-Sub decomposes the toxic task vector into two orthogonal subspaces that represent general capability and destructive capability, respectively. Toxic knowledge is then eliminated by removing only the component representing the destructive ability from the LLM.

#### 3.1.3 Knowledge Unlearning of LLMs

LLMs may inadvertently learn copyrighted material, raising significant legal and ethical concerns [[1](#bib.bib1)], and broader questions about responsible AI use [[36](#bib.bib36)]. In this context, the California Consumer Privacy Act [[124](#bib.bib124)] and the General Data Protection Regulations of the European Union [[57](#bib.bib57)] stipulate the right to data forgetting. The foundational model’s knowledge must be adapted to comply with these regulations. However, the cost of excluding copyrighted data for re-training from scratch is prohibitive. For instance, training a Llama-2-70B from scratch requires 1,720,320 GPU hours [[167](#bib.bib167)].
Traditional methods often use gradient ascent (GA) to achieve forgetting by fine-tuning the model using the GA algorithm on the specific data to be forgotten [[165](#bib.bib165), [196](#bib.bib196)]. Unfortunately, this approach typically catastrophically destroys other parts of the model’s knowledge. That is, forgetting specific knowledge also erases other knowledge that should be retained. Recently, many studies based on model merging techniques have demonstrated the potential to forget LLM-specific knowledge without harming other knowledge [[65](#bib.bib65), [36](#bib.bib36), [60](#bib.bib60)].

Unlike the GA-based approach, the model merging approach does not require additional data for other tasks to maintain old knowledge. To achieve forgetting, model merging typically incorporates a negatively fine-tuned model into the target model (i.e., the task-specific fine-tuned knowledge is subtracted from the target model). For example,
Task Arithmetic [[65](#bib.bib65)] shows that negating task vectors degrade performance on specific tasks without substantial changes to the control tasks. Experiments demonstrate that model merging can forget the knowledge of the target task in a fine-tuned model without harming performance on control tasks. Similarly, Stable Sequential Unlearning (SSU) [[36](#bib.bib36)] extends this forgetting to the setting of sequential unlearning on LLMs, where different copyrighted content must be unlearned at different time steps.
Knowledge forgetting can also forget samples that represent bad behavior during pretraining. For instance, FuseToForget [[204](#bib.bib204)] employs model merging as a debiasing tool to reduce privacy issues in language models. FLearning [[122](#bib.bib122)] first subtracts the parameters related to the data to be forgotten and then fine-tunes the parameters with new data to achieve accurate knowledge updates. SKU [[106](#bib.bib106)] explores the forgetting of harmful data in LLM, which is a two-stage scheme. Initially, harmful data (e.g., harmful question-answer pairs) is used to fine-tune the parameters corresponding to the location of harmful knowledge in the LLM (i.e., the task vector), and then the task vector is negated from the LLM to mitigate undesirable behavior in the LLM effectively. Generally, incorporating the opposite (anti-expert) task vectors into the pre-trained model can effectively accomplish the task of machine unlearning.

#### 3.1.4 Faster Training of LLMs

Training LLMs requires numerous iterations on massive data, making the training process extremely expensive. For example, training LLAMA2-70B with 2T tokens required 1,720,320 GPU hours [[100](#bib.bib100)]. Methods to accelerate LLM training include mixed-precision training, continual retraining, and pipeline parallelism. An orthogonal approach is checkpoint merging in training trajectories, which offers a simple and effective means to either speed up LLM training or enhance training performance at the same cost.

The first type of works incorporate checkpoints in a single training trajectory during LLM training to accelerate model training. For instance, LAWA [[81](#bib.bib81)] demonstrated that merging checkpoints during intermediate stages of model training speeds up the process. For example, training a ResNet50 model on the ImageNet dataset reduced the training time by 68 GPU hours, and training a RoBERTa-Base model on the WikiText-103 dataset saved 30 GPU hours. Sanyal et al. [[143](#bib.bib143)] further showed that the combination of checkpoint averaging in pre-trained trajectories and a high learning rate contributes to faster convergence. Checkpoint Merging [[100](#bib.bib100)] comprehensively evaluates the effectiveness of model merging at different stages of the Baichuan2 [[191](#bib.bib191)] LLM model pre-training process.
The second type of work involves combining existing models to create a more powerful initial model, thereby accelerating learning speed and improving accuracy on downstream tasks. For example, Fusing [[21](#bib.bib21)] and ColD Fusion [[35](#bib.bib35)] mixture of multiple existing fine-tuned models as base models and used for downstream task fine-tuning shows that this merged model outperforms the naive pre-trained model.

#### 3.1.5 Combine the Capabilities of Expert LLMs

LLMs exhibit strong generalizability in general tasks, but often lack knowledge in specific vertical domains. Pretrained LLMs typically require fine-tuning within different corporations to become expert LLMs in various fields. Integrating the expertise of multiple specialists is particularly critical for solving more complex tasks. Research on model merging techniques indicates that a composite LLM can be created by combining the parameters of different expert LLMs [[200](#bib.bib200), [6](#bib.bib6), [31](#bib.bib31), [215](#bib.bib215), [171](#bib.bib171), [172](#bib.bib172), [201](#bib.bib201)].
For example, Dekoninck et al. [[31](#bib.bib31)] demonstrate the ability to flexibly control text generation by merging multiple LLMs with different styles and applying personalized weighting. Robust Weight Signatures [[14](#bib.bib14)] proposes a robustness “patching” framework via model merging to enhance the overall robustness of the model against various naturally corrupted versions of clean data. In summary, model merging offers a straightforward and effective strategy for enhancing LLM’s capabilities.

### 3.2 Model Merging in Multimodal Large Language Models (MLLMs)

Foundation models often involve processing and interacting with data from different modalities, such as video, images, speech, and text. In order to build a generally large model, a key obstacle is the diversity and heterogeneity of tasks and modalities. Traditionally, most existing approaches train a modality-specific model for each modality. However, these methods face limitations: on the one hand, they require separate models for each modality; on the other hand, jointly training a large multimodal model necessitates the expensive collection of paired training data (image, text, video, speech) and the retraining of the entire model when a new modality is added.

An interesting question is whether we can merge multiple modality-specific models to obtain a single, effective, and parameter-efficient modality-agnostic model. We aim for the merged unified model to encode inputs from different modalities, learn cross-modal interactions, and maintain performance comparable to that of well-trained independent modality-specific models. Compared to traditional multimodal learning, model merging techniques offer new opportunities. This model-merging approach offers several benefits: (1) it eliminates the costly and labor-intensive process of collecting labeled paired multimodal training examples, which is required for jointly training multimodal models; (2) it enhances the adaptability of multimodal models, allowing for the seamless integration of new modalities; and (3) it fully leverages knowledge collaboration across multiple modalities, thereby benefiting from cross-modal knowledge transfer.

#### 3.2.1 Model Merging for Multimodal Fusion

Recently, many studies have focused on merging models from different modalities into a single model, thereby enhancing the diversity of knowledge across modalities. For instance, JAM [[4](#bib.bib4)] proposes to merge two specialized (one for text-to-image and one text-only) autoregressive, decoder-only, large transformer models to seamlessly generate multimodal outputs. Similarly, DAMC [[16](#bib.bib16)] introduces a method for fusing multimodal LLMs across image, audio, video, and point cloud modalities, further reducing cross-modal interference through parameter decoupling and adjusting modality fusion coefficients.

To evaluate the impact of various factors on model merging, VL-Merging [[154](#bib.bib154)] performs a comprehensive empirical analysis of multimodal model merging. The overall framework consists of three steps: independent modality fine-tuning, multimodal merging, and downstream task fine-tuning. Through experiments involving different initializations, merging methods, and architectures in multimodal model merging, the authors propose the following guidelines: (1) Models across multiple modalities should be based on the same pretraining starting point to ensure they are in the same basin [[5](#bib.bib5)] and share more information. (2) Simple model averaging achieves better performance, and if more computation and storage resources are available, more fine-grained merges can be conducted. (3) Merging the entire model rather than just a subset of layers generally yields more satisfactory results, as fine-tuning only a subset of layers may restrict the capabilities of single-modality models.
Unlike the above model merging approaches that are developed based on specific architectures, UnIVAL [[147](#bib.bib147)] is the first to design a unified architecture for four modalities: image, video, audio, and language. It transforms the tasks across all modalities into a “sequence-to-sequence” format, with the training objectives of all modalities converted into a “next token prediction” format. This allows for a uniform feature extraction and classifier to be applied across all modalities. Additionally, UnIVAL provides favorable architectural conditions for model merging and demonstrates that linear interpolation of models fine-tuned across multiple modalities in weight space results in a general single model that performs well on both seen and unseen tasks.

#### 3.2.2 Model Merging for Cross-Modal Knowledge Transfer

Some works attempt to transfer knowledge from one modality to another through a model merging approach. For instance,
MAM [[153](#bib.bib153)] investigates whether the attention layers of Transformers [[169](#bib.bib169)] generalize across different modalities. Specifically, it examines if the knowledge acquired by Transformer models trained on high-resource modalities (e.g., data-rich images and text) can be transferred to Transformer models trained on low-resource modalities (e.g., data-sparse speech and audio). This paper demonstrates attention merging for models across various tasks, modalities, and initializations. The final results show that MAM achieves an 18.42% reduction in classification error on the audio classification task (using the ESC-50 dataset [[126](#bib.bib126)]) compared to the standard fine-tuning paradigm.

### 3.3 Model Merging in Image Generative Models

The goal of image generative models, such as generative adversarial networks (GANs), variational autoencoders (VAEs), normalizing flows (Flows), and denoising diffusion probabilistic models (Diffusions), is to approximate the underlying data distribution behind a given dataset so as to generate more new samples with the same distribution. However, image generative models still face the following challenges: the inability to flexibly generate samples with multiple style combinations, the high cost of generative model training, and the inability to generate all the details specified in the instructions.
This dilemma has led to an interest in expert models, which train a set of experts with specific abilities on different data shards or distributions, allowing for the flexible addition or removal of certain styles of experts at inference time. Considering the difficulty in deploying and the cost of resources for ensemble learning, model merging offers a new perspective on combining skill-specific experts of different styles without additional memory and inference costs.

#### 3.3.1 Style Mixing in Generative Models

Existing generative models typically generate distributions based only on the training data. However, in real deployments, different users or artists often want to generate artwork with different combinations of styles. Collecting additional data for these mixed distributions is expensive, and fine-tuning the model can result in the forgetting of other capabilities. Model merging offers the potential to flexibly combine multiple styles.

Earl GAN Cocktail [[10](#bib.bib10)] attempted to merge several pre-trained GAN models. Recently, diffusion-based image generative models [[56](#bib.bib56), [139](#bib.bib139), [140](#bib.bib140)] have gained more attention than GAN-based models due to their superior generative capabilities. Consequently, most research focuses on fusing different diffusion models.
Specifically, Diffusion Soup [[12](#bib.bib12)] demonstrates the ability to linearly merge diffusion models fine-tuned on data shards of different styles (e.g., data provided by different domains/categories or different users), resulting in hybrid style zero-shot generation.
In addition, Diffusion Soup empirically verifies that model merging has an anti-memorization effect, meaning the generated images are less likely to replicate training data, which is beneficial for generating diverse images. Unlike Diffusion Soup, which directly merges model parameters, MaxFusion [[118](#bib.bib118)] is inspired by ZipIt [[151](#bib.bib151)] and proposes merging intermediate features of multiple diffusion models based on the same input noise to generate images that satisfy multiple conditions.
However, merging multiple diffusion models based on full parameter fine-tuning can be costly when the number of tasks is large. To address this issue, ZipLoRA [[145](#bib.bib145)] and MoLE [[186](#bib.bib186)] aim to seamlessly merge parameter-efficient LoRA modules. For example, ZipLoRA proposes merging independently trained content/subject (e.g., a specific object or person) LoRAs with artistic style (e.g., drawing or painting, etc.) LoRAs, allowing the diffusion model to generate any user-provided combination of subjects and styles [[141](#bib.bib141)]. This approach enables users and artists to easily combine publicly available subjects and styles LoRAs of their choice.

#### 3.3.2 Reducing Training Cost of Generative Models

In real-world scenarios, large-scale training data typically originates from different domains or is provided by various users. Given the need to add new data or remove outdated data, retraining a single model with updated data every time is often impractical [[12](#bib.bib12)]. For instance, training a CM model [[150](#bib.bib150)] using 8 A100 GPUs [[101](#bib.bib101)] takes about one week.
This is because existing methods only apply the final convergence weights during generative model training and ignore the intermediate training trajectories. LCSC [[101](#bib.bib101)] demonstrates that a simple combination of training trajectories in the middle of the diffusion model by the evolutionary algorithm can significantly reduce the training cost. Specifically, only a few iterations or a small batch size is required to train the diffusion model to achieve image quality comparable to that of a fully trained diffusion model. For example, on the CIFAR-10 dataset, LCSC improves the training process for consistency distillation and consistency training [[150](#bib.bib150)] by factors of 23×\small 23\times and 7×\small 7\times, respectively. The underlying reason is that each local checkpoint of the optimized trajectory has many high-quality basins (i.e., areas of better generation quality) nearby that cannot be reached by stochastic gradient descent due to substantial variance in gradient estimations. However, checkpoint interpolation provides an opportunity to reach these basins.

#### 3.3.3 Enhancing the Faithfulness of Generative Models

Some studies on Text-to-Image (T2I) show that although the existing T2I generation model can generate high-quality images according to text prompts, the images often fail to fully capture and reflect the semantic details in the text, such as generating multiple subjects or correctly depicting the spatial relationships between objects [[89](#bib.bib89)]. To enhance the fidelity of the T2I generation models, SELMA [[89](#bib.bib89)] designed a novel four-stage paradigm. In the first and second stages, a series of input texts (corresponding to different skills) are collected through diversified prompts from existing LLMs, and the corresponding image data are generated using the T2I model. The third stage involves fine-tuning skill-specific experts (i.e., LoRA) separately on images of different skills. In the fourth stage, expert models with different skills are merged to obtain the final model during inference. Compared to the paradigm of joint learning of multiple skills, this method of merging expert skills after independent learning may help alleviate knowledge/skill conflicts while also being more efficient.

## 4 Application of Model Merging in Different Machine Learning Subfields

Model merging is a simple and effective technique widely used in various subfields of machine learning, such as continual learning, multi-task learning, domain generalization, federated learning, few-shot learning, and adversarial defense, etc. In this section, we comprehensively discuss the application of model merging in the different machine learning subfields.
Table [3](#S4.T3 "Table 3 ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") provides a brief summary, and in §[4.1](#S4.SS1 "4.1 Model Merging in Continual Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") to §[4.6](#S4.SS6 "4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), we introduce each application case in detail.

Table 3: A summary of the application of model merging techniques in different machine learning subfields.

|  |  |
| --- | --- |
| Scenarios | The Main Purpose of Model Merging |
| Continual Learning (§[4.1](#S4.SS1 "4.1 Model Merging in Continual Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Avoiding catastrophic forgetting with respect to old tasks |
| Multi-Task / Multi-Domain / Multi-Objective / Auxiliary Learning (§[4.2](#S4.SS2 "4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Performing multiple tasks / domains / objectives via one model |
| Domain / Out-of-Distribution Generalization (§[4.3](#S4.SS3 "4.3 Model Merging in Out-of-Distribution/Domain Generalization ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Achieving generalization to unknown target domains or distributions |
| Federated Learning (§[4.4](#S4.SS4 "4.4 Model Merging in Federated Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Merging local models provided by different clients |
| Zero-shot / Few-shot Learning (§[4.5](#S4.SS5 "4.5 Model Merging in Zero-shot/Few-shot Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Multiple related models are merged to improve the zero-shot / few-shot learning ability on new tasks |
| Adversarial Learning (§[4.6](#S4.SS6 "4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")) | Implementing model poisoning attack, defense, and copyright protection |

### 4.1 Model Merging in Continual Learning

Continual Learning (CL) involves training a model using a streaming, non-stationary data stream. The primary challenge in CL is the ‘catastrophic forgetting’ [[138](#bib.bib138), [178](#bib.bib178)] problem; that is, the CL model’s prediction accuracy for old tasks drops dramatically after training on new tasks. The mainstream CL methods are mainly divided into memory replay-based methods, architecture expansion-based methods, regularization-based methods, and subspace projection-based methods [[178](#bib.bib178)]. Recently, there has been a growing interest in using model merging to address the catastrophic forgetting problem. This novel approach offers several benefits, such as avoiding additional parameters and inference costs associated with network expansion-based methods and eliminating the need to cache old data as required by memory-based methods.

#### 4.1.1 Model Merging to Mitigate Catastrophic Forgetting

Tangent Model Composition [[104](#bib.bib104)] proposes fine-tuning each task independently in the tangent space of the pre-trained model and then linearly fine-tuning these models to perform CL. This approach does not depend on the specific settings of CL and can be easily applied to task, class, and domain-incremental learning scenarios. In addition, ITA [[127](#bib.bib127)] emphasizes the necessity for the fine-tuned model to be in the same basin as the pre-trained model to ensure the composability of nonlinear models. It introduces a regularization term similar to EWC [[85](#bib.bib85)] in traditional CL to constrain the distance between the fine-tuned weights and the pre-trained weights when training the independent model. WARP [[135](#bib.bib135)] suggests linearly interpolating the pre-trained LLM’s weights with its aligned weights via RLHF on a preference dataset, thus mitigating the forgetting of knowledge from the pre-trained LLM. BAM [[7](#bib.bib7)] continuously adapts LLMs to new languages by merging models while preserving general capabilities. Model Tailor [[218](#bib.bib218)] explores the problem of catastrophic forgetting during fine-tuning of MLLMs, and proposes to merge only the most important subset of parameters in the fine-tuned MLLM model into the pre-trained MLLM model, so as to retain the generalization ability of the pre-trained model as much as possible, while compensating the selected weights to reduce the performance of the fine-tuning task. MagMax [[112](#bib.bib112)] merges pruned task vectors to further alleviate parameter sign conflicts and old knowledge forgetting.
Equifinality, PAINT [[64](#bib.bib64)] and LM-Cocktail [[187](#bib.bib187)] interpolate the weights of the fine-tuned model and the zero-shot model to improve accuracy on downstream tasks without degrading accuracy on supported/general tasks.

In contrast to merging full models, some research focuses on merging parameter-efficient modules.
Chitale et al. [[20](#bib.bib20)] propose a CL method based on task arithmetic [[65](#bib.bib65)]. This method first fine-tunes a task-specific LoRA for each task, then constructs a task vector based on the difference between fine-tuned and pre-trained models. Multiple task vectors are then merged, and a small amount of data (10 samples per class) is used to fine-tune the merged model. Compared to traditional CL methods, particularly those based on replay, this approach eliminates the need to replay data from old tasks at each iteration, thereby accelerating model training. Additionally, fine-tuning the merged model with a class-balanced subset helps mitigate CL model bias. Similarly, DynaMMo [[128](#bib.bib128)] applies lightweight model merging (i.e., Adapter) in a CL setting for medical images. In contrast to architecture expansion-based CL methods, this approach does not result in a linear increase in the number of parameters with the number of tasks. Unlike the static aggregated parameter-efficient fine-tuning (PEFT) modules of DynaMMo, DAM [[19](#bib.bib19)] introduces dynamic aggregated PEFT modules during inference to perform CL.
AMM [[17](#bib.bib17)] proposes merging convolutional layers to facilitate incremental new class discovery and prevent forgetting fundamental knowledge. Disperse-Then-Merge [[44](#bib.bib44)] suggests merging submodels trained on different data partitions during the supervised fine-tuning of LLMs to reduce data bias and mitigate the forgetting of generic pre-trained knowledge.

### 4.2 Model Merging in Multi-Task/Multi-Objective/Multi-Domain/Auxiliary Learning

In machine learning, to optimize resource efficiency, we typically use a single model to handle multiple tasks, objectives, or data domains with varying distributions. The traditional multi-task learning (MTL), multi-objective learning (MOO), or multi-domain learning (MDL) paradigm requires gathering data from all tasks, objectives, or domains to collaboratively train a model, leading to high data management and model training costs. This approach becomes particularly costly when new tasks, goals, or domains are introduced, as retraining a comprehensive model from scratch using all available data is resource-intensive. Numerous recent studies have proposed efficient methods for integrating knowledge across tasks, goals, or domains by merging models directly.

#### 4.2.1 Knowledge Transfer in Multi-Task Learning

The goal of multi-task learning (MTL) is to enable a single model to perform multiple tasks simultaneously, thereby facilitating knowledge transfer between these tasks [[110](#bib.bib110), [18](#bib.bib18), [144](#bib.bib144), [152](#bib.bib152), [202](#bib.bib202), [192](#bib.bib192)]. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities")(c), to avoid the high cost of joint training, a straightforward approach is to merge multiple independently trained models on different tasks to accomplish MTL. Almost all of the model merging methods discussed in §[2.3](#S2.SS3 "2.3 During Merging Methods ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") can be used to merge multiple models trained on different tasks to perform MTL. In this section, we take some representative tasks as examples. For MTL tasks in computer vision, Task Arithmetic [[65](#bib.bib65)], Ties-Merging [[190](#bib.bib190)], AdaMerging [[194](#bib.bib194)] and other studies [[197](#bib.bib197), [156](#bib.bib156), [193](#bib.bib193)] proposed to combine ViT models trained on different visual classification tasks, and the obtained model can complete the object classification of multiple tasks. The results of Task Arithmetic [[65](#bib.bib65)] demonstrate that merging independently trained models from any two datasets yields a merged model whose performance is comparable to that of a single-task model. Similarly, ZipIt [[151](#bib.bib151)], which merges ResNet architectures trained on different tasks, achieves comparable results.
For MTL tasks in natural language processing, DARE [[200](#bib.bib200)] introduces a method to assimilate homologous models, augmenting LLMs as a ”free lunch”. For instance, merging WizardLM with WizardMath significantly boosts WizardLM’s performance on GSM8K (a benchmark for evaluating the mathematical reasoning ability of LLMs) from 2.2 to 66.3. Akiba et al. [[6](#bib.bib6)] suggest that directly merging an LLM with mathematical capabilities and an LLM with Japanese language proficiency results in a model capable of solving Japanese mathematical problems.
Furthermore, numerous studies have demonstrated that combining PEFT modules (such as Adapter or LoRA) trained on different tasks can also achieve MTL [[160](#bib.bib160), [208](#bib.bib208)].

#### 4.2.2 Knowledge Transfer in Multi-Objective Optimization

Multi-objective optimization (MOO) aims to optimize multiple objective functions simultaneously. These objective functions may conflict with one another, so the MOO problem typically does not have a single optimal solution. Instead, it involves finding a trade-off among the multiple objectives, which corresponds to identifying a set of Pareto optimal solutions.
Tang et al. [[158](#bib.bib158)] propose approximating the entire Pareto set using a mixture of experts (MoE) based model merging approach. Specifically, their method trains an independent model for each objective and learns a routing network to balance the trade-offs between the multiple objectives (models). The input of the routing network is the task preference vector, and its output consists of the merging coefficients for the independent models.
Considering that directly evaluating Pareto solutions based on the original evaluation metric is time-consuming, MAP [[91](#bib.bib91)] proposes a second-order Taylor expansion model as a surrogate model for the true evaluation metric, and further uses an evolutionary algorithm to calculate the Pareto front based on the surrogate model.

#### 4.2.3 Knowledge Transfer in Multi-Domain Learning

Unlike existing model-merging-based MTL approaches that focus on datasets with different object categories, Ye et al. [[197](#bib.bib197)] explore model merging across multiple domains, where datasets share the same categories but differ in environmental contexts. To mitigate conflicts between multi-domain models, this paper introduces a weight similarity criterion to assess the correlation between different model layers. For layers with high correlation, a simple weight averaging or RegMean [[78](#bib.bib78)] strategy is employed to merge models that have been fine-tuned in different domains of the same task. For layers with low correlation, the weights are flexibly combined using a gating mechanism during the inference phase. Branch-Train-Merge [[92](#bib.bib92)] demonstrates the effectiveness of training expert language models on 64 different domains and subsequently merging them.

#### 4.2.4 Knowledge Transfer in Auxiliary Task Learning

The goal of auxiliary task learning (ATL) is to enhance the performance of the target task by leveraging knowledge obtained from related auxiliary tasks. Unlike MTL, which aims to optimize the average performance across all tasks, ATL focuses solely on improving the performance of the main task. However, ATL often encounters the issue of gradient conflict, leading to negative transfer, where the inclusion of auxiliary tasks interferes with the main task’s performance. To mitigate negative transfer, Jiang et al. [[76](#bib.bib76)] propose ForkMerge, a method that periodically performs ‘fork’ and ‘merge’ operations. The model is first periodically duplicated into multiple branches: the first branch is trained exclusively on the main task, while the remaining branches are trained jointly on both the main and auxiliary tasks. An optimal merging coefficient is then determined using the validation set to merge the models updated by the various branches. Empirical results show that ForkMerge achieves positive transfer gains across several auxiliary task learning benchmarks.

### 4.3 Model Merging in Out-of-Distribution/Domain Generalization

The common goal of out-of-distribution generalization (OODG) and domain generalization (DG) is to improve a model’s performance on unseen data. The key difference between them is that OODG focuses on enhancing a model’s generalization ability on unknown data with significantly different distributions from the training data. In contrast, DG emphasizes improving a model’s generalization ability on unseen domains. Numerous recent studies have demonstrated that model merging contributes to enhanced training stability and overall performance in both OODG and DG.

#### 4.3.1 Model Merging for Better Out-of-Distribution Generalization

In real-world scenarios, a trained model may be deployed in environments with changing distributions. For example, autonomous driving models are trained on a clean dataset, but in practice, they are vulnerable to unforeseen distributions such as natural corruptions (e.g., camera noise, motion blur) and more significant distribution shifts (e.g., summer to winter) [[55](#bib.bib55), [14](#bib.bib14)]. The goal of OODG is to enhance the model’s ability to generalize to unknown data that significantly differs from the training distribution.

Stochastic weight averaging (SWA) [[67](#bib.bib67)] is a straightforward and widely used technique to improve machine learning models’ training stability and OOD performance. From a statistical perspective, weight averaging helps reduce variance during model training. Many works merge intermediate weight states (i.e., checkpoints) from training trajectories while training models [[161](#bib.bib161), [67](#bib.bib67), [195](#bib.bib195), [50](#bib.bib50), [207](#bib.bib207), [170](#bib.bib170)]. For example, WiSE fine-tuning [[184](#bib.bib184)] demonstrates that linearly combining the weights of a pre-trained model and a fine-tuned model can significantly improve accuracy in the case of distribution shifts, while maintaining high accuracy on the original distribution. SWA [[67](#bib.bib67), [50](#bib.bib50)] simply averages all checkpoints from the beginning of a particular epoch to the end of training. This approach is explained to help the model converge to flat rather than sharp local optima, thereby improving generalization [[67](#bib.bib67), [82](#bib.bib82)]. Adaptive SWA [[32](#bib.bib32)] highlights that executing SWA too early may lead to underfitting, while executing it too late may result in overfitting. It proposes averaging only when generalization on the validation set improves, effectively combining SWA with an early stopping mechanism. However, simple average weights are often suboptimal. In particular, TWA [[94](#bib.bib94)] addresses this by showing that the averaging coefficients of the weights can be determined in a training manner. Consequently, TWA, unlike simple SWA, can perform averaging from the initial epoch of training, eliminating the need to define an additional hyperparameter for the epoch at which weight averaging should start.

In contrast to previous works that average weights obtained along one training trajectory, methods such as Model Soups [[183](#bib.bib183), [220](#bib.bib220)], AdapterSoup [[24](#bib.bib24)], Model-Ratatouille [[133](#bib.bib133)], WARM [[136](#bib.bib136)], WARP [[135](#bib.bib135)], PAPA [[79](#bib.bib79)], WASH [[42](#bib.bib42)], DART [[70](#bib.bib70)], and DiWA [[132](#bib.bib132)] propose merging multiple independently fine-tuned or trained models. These models are usually more diverse, which improves OOD performance. Independently trained models differ in hyperparameters (e.g., learning rate, weight decay, dropout), batch order, data augmentation techniques (e.g., random crops, horizontal flips), and the number of training steps, among other factors. Specifically, Model-Ratatouille [[133](#bib.bib133)], starts from the same initial model, fine-tunes multiple models on an auxiliary task, then continues to fine-tune these models on the target task, and finally merges the diverse models to improve OOD performance. WARM [[136](#bib.bib136)] further increases the diversity of fine-tuned models by sampling different checkpoints from the trajectories of the pre-trained model as the initial weights for the downstream preference fine-tuning task. To reduce the additional cost of training multiple models, Model Stock [[71](#bib.bib71)] proposes that we can exploit the geometric properties of the weight space and the anchoring effect of pretrained models to approximate the merged weights using only a few fine-tuned models. MEHL-Soup [[95](#bib.bib95)] develops a scalable and efficient method to learn model merging coefficients for model soup. It only loads a subset of models for each iteration, significantly reducing the computation and memory requirements of naive model soup for learning merging coefficients.

The above analysis reveals that the SWA lacks diversity due to its reliance on a single trajectory. In contrast, Model Soups and DiWA train independently, which can lead to multiple models with significant differences, resulting in weight averaging failure. To balance these two approaches, Lookaround [[207](#bib.bib207)] introduces a gradient descent optimizer based on the weight averaging. This optimizer iteratively performs ‘around’ and ‘average’ steps throughout the optimization process. In the ‘around’ step, multiple independent models are trained from the same starting point, each using different data augmentations. In the ‘average’ step, the diverse models are averaged, and the result is used as the starting point for the next iteration.

#### 4.3.2 Model Merging for Better Domain Generalization

Domain generalization methods aim to generalize to an unknown target domain using only training data from source domains. For instance, in the context of traffic sign recognition, the training data for a machine learning (ML) model tasked with identifying traffic signs in various urban environments come from multiple cities (i.e., source domains). However, when deployed, the model must recognize traffic signs in new urban environments (i.e., target domains) that it has never encountered before. Existing DG methods can be classified into domain alignment, data augmentation, regularization, and meta-learning frameworks [[8](#bib.bib8)]. Complementary to these approaches, model merging techniques can be seamlessly integrated to further improve out-of-domain performance without modification. Specifically, model merging in DG mainly occurs during the training process of the source domain model. Merging the intermediate weight states from different training stages helps improve the stability and generalization of the final model.

SWAD [[15](#bib.bib15)] demonstrates that flatter minima generalize better to unseen domains. Inspired by SWA [[67](#bib.bib67)], SWAD proposes a dense and overfit-aware stochastic weight sampling strategy to identify these flatter minima. More specifically, unlike SWA, it starts from a predefined epoch until the final epoch, and collects a random weight every K𝐾K epochs for averaging. SWAD collects weights densely, that is, one is collected every iteration/step, and the start and end of random weight collection are determined by the performance changes on the validation set. EoA [[8](#bib.bib8)] also shows that model averaging can improve out-of-domain performance stability, and that ensembling multiple moving average models can further enhance performance compared to ensembling models without weight averaging.

### 4.4 Model Merging in Federated Learning

Federated Learning (FL) is a distributed learning approach that allows multiple clients to collaboratively train a model without sharing data. FL primarily includes two settings: centralized (with a central server) and decentralized (without a central server). Each client updates the model or calculates the gradient based on local data and sends the updated information to the central server (in centralized FL) or other clients (in decentralized FL) for aggregation to update the global model, thus ensuring data privacy protection.

#### 4.4.1 Federated Learning Paradigm

Model merging is a routine and crucial operation in FL. Taking centralized FL as an example, it typically involves N𝑁N clients and a central server S𝑆S. Each client has a private set of training data. Specifically, the training process in the centralized FL paradigm consists of five steps:
(1) Model initialization: The central server initializes the global model parameter;
(2) Model distribution: The latest model on the server is sent to the local client in the t𝑡t-th round of communication.
(3) Update of the local model: The i𝑖i-th client updates the model by calculating the gradient based on the local data.
(4) Model upload: The updated models of all local clients are sent to the server aggregation.
(5) Model aggregation: The multiple local models on the server are aggregated.
These five steps are repeated until the model converges or the maximum number of training rounds is reached.
Since this paper is not a survey of FL, we focus on implementing the ‘model aggregation’ step. In FL, model merging refers to summarizing model parameters from various clients during each communication round, thereby forming an updated global model.

#### 4.4.2 Model Merging for Local Knowledge Aggregation

Most FL methods adopt a simple coordinate-wise average to aggregate the local models. For example, they calculate local model merging coefficients according to some heuristic rules. FedAvg [[114](#bib.bib114)], the most classic FL method, proposes to merge local models on the server weighted by the amount of training data from each client. FedNova [[175](#bib.bib175)] normalizes and scales model updates on the client side based on the number of update steps, efficiently aggregating local models to obtain a high-performance global model. FedAtt [[74](#bib.bib74)] calculates layer-wise attention coefficients based on the similarity of client and server parameters, fusing local models based on these coefficients.
FedFisher [[73](#bib.bib73)] computes the Fisher information matrix of the parameters in each client to merge the local models.
In more challenging FL tasks, the above direct coordinate-wise merging methods may result in suboptimal global model performance. Inspired by the property of permutation invariance of neural networks, PFNM [[203](#bib.bib203)], OTFusion [[148](#bib.bib148)] and FedMA [[174](#bib.bib174)] propose to permute neurons of local models before merging them.
Similarly, GAMF [[99](#bib.bib99)] transforms the model merging problem into a multi-graph matching problem based on graph matching and then merges the aligned local models.

### 4.5 Model Merging in Zero-shot/Few-shot Learning

In practical applications of machine learning models, collecting a large amount of labeled data can be expensive or infeasible in specific scenarios (e.g., medical diagnosis, real-time monitoring). Users often want deep models to effectively perform new tasks that have not been encountered before, that is, an ability commonly referred to as cross-task generalization [[61](#bib.bib61)]. Zero-shot [[115](#bib.bib115)] and few-shot learning [[198](#bib.bib198)] can reduce the dependence on large amounts of data and allow the model to better deal with unseen categories or small numbers of samples, improving the cross-task generalization ability of the model. In few-shot learning, a common approach is to fine-tune the model using the limited examples available. However, because of the minimal data, this fine-tuning process is often unstable and yields only modest performance improvements. Recently, some studies have explored merging pre-trained models (from some publicly accessible resources) to enhance cross-task generalization under zero-shot and few-shot conditions.

#### 4.5.1 Model Merging for Cross-task Generalization in Zero-shot Learning

Model merging has demonstrated the effectiveness of zero-shot learning across several applications. Some examples of practical applications include cross-lingual transfer [[63](#bib.bib63), [25](#bib.bib25), [211](#bib.bib211), [86](#bib.bib86)], hybrid style image generation [[12](#bib.bib12), [118](#bib.bib118)], and multi-modal processing [[16](#bib.bib16)].

Some works achieve cross-lingual transfer through model merging, such as chat [[63](#bib.bib63)], text summarization [[25](#bib.bib25)], or reasoning [[211](#bib.bib211)]. A well-performing language-specific LLM needs to be fully trained, and with 7,000 languages in the world, not all of them have enough labeled data to support model fine-tuning. Therefore, cross-lingual knowledge transfer is particularly important.
For example, Huang et al. [[63](#bib.bib63)] build a Chat vector based on fine-tuned LLAMA2-chat and pre-trained LLAMA2 on chat data in the English language, and assembles it with the continuously pre-trained LLAMA2 model on other non-English languages. This allows the new model to chat in non-English languages.
Chronopoulou et al. [[25](#bib.bib25)] develop a zero-shot multilingual summarization framework. It uses a merged model (a supervised summarization model and an unsupervised pre-trained model for a high-resource language, along with an unsupervised pre-trained model for a low-resource language) to perform text summarization tasks in low-resource languages. Similarly, AdaMergeX [[211](#bib.bib211)] demonstrates the effectiveness of model merging for cross-language transfer across three tasks: reasoning, natural language understanding, and natural language generation.
In the hybrid style image generation task, Diffusion Soup [[12](#bib.bib12)] and MaxFusion [[118](#bib.bib118)] show that the zero-shot generation ability can be enhanced by merging multiple diffusion models.
In the multi-modality task, DAMC [[16](#bib.bib16)] experiments prove that zero-shot multi-modal extension can be achieved by merging multi-modal models, provided they are initialized from the same LLM. For example, by merging a visual LLM and an audio LLM, the combined model can not only perform image or audio tasks independently but also acquire the zero-shot ability to process inputs containing both visual and auditory information simultaneously.

#### 4.5.2 Model Merging for Cross-task Generalization in Few-shot Learning

Parameter-efficient fine-tuning (PEFT), such as LoRA or Adapter, facilitates the creation and sharing of thousands of custom PEFT modules, each trained on different data for various downstream tasks. A natural question is whether combining PEFT modules pre-trained on different upstream tasks can improve the transfer accuracy for unseen downstream tasks with limited samples.

Recent work on model merging suggests a positive answer, showing that merged models can enhance generalization in few-shot settings [[53](#bib.bib53), [9](#bib.bib9), [61](#bib.bib61)]. For example, LoraHub [[61](#bib.bib61)] proposes to merge LoRA modules available on HuggingFace to achieve adaptive performance for unseen tasks, where the merging coefficients of different LoRA are searched in a black-box gradient-free manner with few-shot samples. As expected, few-shot LoraHub performs better than few-shot in-context learning and reduces inference costs by eliminating the need for examples as input to LLMs. LoraRetriever [[212](#bib.bib212)] further proposes dynamically retrieving the most relevant LoRAs based on the input and merging them. Similarly, MerA [[53](#bib.bib53)] proposes merging pretrained adapters into a single adapter for few-shot NLP scenarios.
In general, well-trained LoRAs or adapters can serve as valuable resources that users can easily share, access, and apply to a variety of downstream tasks. In the real world, upstream and downstream tasks can be entirely disparate, originating from different datasets, domains, or even different parts of the same dataset. Asadi et al. [[9](#bib.bib9)] comprehensively evaluates model merging in the few-shot learning setting. Specifically, this study examines three cases of label, domain, and task drift between upstream and downstream tasks. The results demonstrate that model merging enhances the model’s generalization ability in few-shot learning scenarios across different contexts.

### 4.6 Model Merging in Adversarial Learning

In the machine learning community, the open-source availability of pre-trained models [[167](#bib.bib167), [166](#bib.bib166), [163](#bib.bib163), [129](#bib.bib129), [130](#bib.bib130)] has accelerated technological advancements. In this context, developers often download unvalidated checkpoints to fine-tune their models or even outsource the training process to third-party platforms [[185](#bib.bib185)]. Consequently, open-source models are also vulnerable to malicious attacks, such as poisoning attacks, where hidden malicious behaviors can be triggered by specific inputs. This raises several intriguing questions: Can model merging lead to attacks, and can it be used to develop defense mechanisms? Additionally, how can intellectual property protection be enhanced in the context of model merging?

#### 4.6.1 Model Merging as an Attack Strategy

Parameter-Efficient Fine-Tuning (PEFT) methods [[34](#bib.bib34)], such as LoRA [[59](#bib.bib59)], exhibit functional transferability. This means that a LoRA model fine-tuned for a specific task based on a pretrained model can be successfully transferred to another pretrained model [[103](#bib.bib103)]. In practice, developers often download LoRA models from open-source platforms to address their specific downstream tasks [[61](#bib.bib61)]. If a poisoned LoRA, which could be seen as a Trojan horse, is inadvertently downloaded and integrated into a model, it may introduce security vulnerabilities. Research by LoRA-as-an-Attack [[103](#bib.bib103)] demonstrates that merging a poisoned LoRA—trained on compromised data—with a benign LoRA, trained on clean data, can result in a backdoor injection. This finding also holds when multiple LoRAs are merged. In addition, BadMerging [[210](#bib.bib210)] has developed a two-stage backdoor attack framework specifically for model merging, and through a large number of experiments, it has shown that the success rate of on-task and off-task attacks on merged models exceeds 90%, and existing defense measures cannot defend against BadMerging.

#### 4.6.2 Model Merging as a Defense Strategy

Contrary to the attacks described in §[4.6.1](#S4.SS6.SSS1 "4.6.1 Model Merging as an Attack Strategy ‣ 4.6 Model Merging in Adversarial Learning ‣ 4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), the transferability of LoRA also offers an opportunity for model merging as a defense strategy. Specifically, if we know that a model may be susceptible to certain attacks, can we train some LoRAs to enhance the model’s defense (i.e., reduce the attacker’s success rate)? For example, Liu et al. [[103](#bib.bib103)] demonstrate that GPT-3.5 was used to generate a benign dataset containing backdoor triggers. A dedicated defense LoRA was then trained on this benign data and merged into the poisoned pre-trained model. This defensive model merging ultimately led to a reduction in the backdoor effect. Furthermore, research has shown that in the context of full parameter fine-tuning, model merging can serve as a ”free lunch” for model defense. Experiments involving four model architectures and four datasets revealed that merging multiple poisoned models without additional effort mitigated these poisoning attacks, with the accuracy on the benign dataset remaining nearly unaffected. Rebuffi et al. [[137](#bib.bib137)] and Croce et al. [[28](#bib.bib28)] merge a set of lpsubscript𝑙𝑝l\_{p} (for various p𝑝p) robust fine-tuned models to easily control the robustness level of each threat model against lpsubscript𝑙𝑝l\_{p} boundary adversarial attacks.
Similarly, the experimental analysis by [[45](#bib.bib45)] indicates that model merging offers an effective defense mechanism against jailbreak attacks [[179](#bib.bib179)].

In another practical scenario, merging unauthorized models may infringe on the intellectual property rights of the model owner. Malicious users might merge several high-quality open-source models (e.g., those authorized for research use only) to create a new model, then claim that this new model was entirely developed and trained from scratch by themselves, subsequently offering model services for commercial gain. In such cases, it becomes particularly crucial for model owners to detect whether others have merged their models. MergeGuard [[26](#bib.bib26)] performs a preliminary analysis of the effectiveness of two existing defense methods—Quantization Watermarking [[90](#bib.bib90)] and Instructional Fingerprint [[188](#bib.bib188)]—in the context of model merging. The study observed that while the watermarking method cannot be detected in the merged model, the fingerprint method remains detectable.

## 5 Remaining Challenges and Future Directions

Although §[2](#S2 "2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), §[3](#S3 "3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[4](#S4 "4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") present various advanced model merging methods and applications, challenges remain in the technology and application of existing model merging approaches. Additionally, there are numerous areas that warrant further research in the future.

(1) Closing the Performance Gap Between the Merged and Independent Models.
In practical settings, guaranteeing the performance of model merging remains challenging. The effectiveness of current model merging techniques heavily relies on the ”pretraining-finetuning” paradigm. Specifically, successful model merging requires that multiple models be fine-tuned based on the same pre-trained model, with careful control over the number of epochs and learning rate during fine-tuning. If these hyper-parameters are not set properly, the models may not converge in the same or close basin. Even based on the pre-trained fine-tuning paradigm, there is still a significant gap between the merged and independent models, especially when the number of models/tasks is large.
Therefore, a promising direction for future research is to explore how to ensure the effectiveness of model merging under more relaxed conditions. For example, investigating how to merge multiple models that are trained independently from scratch for different tasks without compromising performance could be valuable.

(2) In-depth Theoretical Analysis for Model Merging.
The validity and explanation of existing model merging techniques are largely empirical and lack sufficient theoretical guarantees. As discussed in §[2.4](#S2.SS4 "2.4 Theories and Analysis of Model Merging ‣ 2 Advanced Model Merging Methods ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), there is currently a limited amount of work on the theoretical aspects or explanations of model merging. The few existing studies mainly focus on merging multiple models trained on the same trajectory or on the same dataset with different fine-tuning settings. There is almost no theoretical research or explanation concerning merging multiple models fine-tuned on different datasets or merging multiple models trained from scratch on different datasets. Therefore, future research should aim for a more comprehensive and in-depth theoretical analysis to enhance the success and reliability of model merging. Furthermore, a deeper understanding of the effectiveness of model merging can, in turn, facilitate the discovery of prior conditions that are more conducive to model merging.

(3) Trustworthy Model Merging.
Model merging is prone to intellectual property disputes and poisoning attacks, making the development of a reliable and trustworthy merging scheme an urgent research priority. Research on the reliability of model merging can be categorized based on two key roles: the model owner and the model combiner.
On one hand, for model owners, protecting the intellectual property of their models is a primary concern. This protection involves both active and passive defense strategies: (1) Active Defense: Model owners may want to ensure that their published models are used independently and not merged by other users. The ideal outcome of an active defense strategy is that the model performs stably when used as intended, but breaks down completely if merged with other models. (2) Passive Defense: When model owners suspect that their models have been merged, there needs to be a robust method to verify whether the merged models contain their original models.
On the other hand, for the model combiner, a key research direction is how to effectively prevent the inclusion of malicious injections, such as backdoors or poisoning attacks, when merging a set of authorized models.

(4) Effective and Efficient Model Merging.
Existing high-performance model merging methods often come with significant costs in terms of efficiency and memory. First, most of these methods require all models to be loaded into memory during execution. For instance, merging 72 fine-tuned ViT-B/32 models necessitates more than 200GB of memory [[95](#bib.bib95), [183](#bib.bib183)]. Additionally, heuristics for determining model merging coefficients involve repeated evaluations of the combined model, while learnable methods depend on additional data and training. In the future, it would be beneficial to develop more efficient model merging methods that do not require training, additional data, GPUs, or large amounts of memory.

(5) Merge Heterogeneous Expert Models.
Existing methods primarily focus on merging homogeneous models. However, in practice, numerous heterogeneous models excel in various tasks. A limited number of existing methods for merging heterogeneous models involve transforming multiple heterogeneous models into homogeneous ones using knowledge distillation techniques, followed by the merging process [[10](#bib.bib10), [172](#bib.bib172)]. The distillation process relies on the data from the original tasks and involves costly training. Therefore, it is also worth exploring approaches to merge these heterogeneous models without incurring the high costs associated with architectural transformations.

(6) Interdisciplinary Application of Model Merging.
As discussed in §[3](#S3 "3 Application of Model Merging in Foundation Models ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities") and §[4](#S4 "4 Application of Model Merging in Different Machine Learning Subfields ‣ Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities"), model merging has been adeptly applied across various foundation models and machine learning subfields to address different challenges and achieve interesting tasks. The question of how to adapt model merging strategies from one subfield to another presents an exciting avenue for exploration.

## 6 Conclusions

Model merging is a straightforward and effective technique for model enhancement that combines multiple models to achieve diverse capabilities. In this survey, we first provide a comprehensive overview of the advanced methods and theories currently available in the field of model merging. Next, we discuss the application of model merging techniques across various foundation models (i.e., LLMs, MLLMs) and more than ten subfields of machine learning, highlighting their use in addressing various challenges and difficulties. Finally, we identify ongoing issues within the model merging and propose six research directions that are worthy of further exploration. We believe that model merging technology, as an efficient and modular model empowerment solution, will play a significant role in more practical scenarios in the future.

## References

* Abad et al. [2024]

  Javier Abad, Konstantin Donhauser, Francesco Pinto, and Fanny Yang.
  Strong copyright protection for language models via adaptive model fusion.
  *ICML*, 2024.
* Achiam et al. [2023]

  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
  Gpt-4 technical report.
  *arXiv preprint arXiv:2303.08774*, 2023.
* Adilova et al. [2024]

  Linara Adilova, Asja Fischer, and Martin Jaggi.
  Layerwise linear mode connectivity.
  *ICLR*, 2024.
* Aiello et al. [2024]

  Emanuele Aiello, Lili Yu, Yixin Nie, Armen Aghajanyan, and Barlas Oguz.
  Jointly training large autoregressive multimodal models.
  *ICLR*, 2024.
* Ainsworth et al. [2023]

  Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa.
  Git re-basin: Merging models modulo permutation symmetries.
  In *ICLR*, 2023.
* Akiba et al. [2024]

  Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, and David Ha.
  Evolutionary optimization of model merging recipes.
  *arXiv preprint arXiv:2403.13187*, 2024.
* Alexandrov et al. [2024]

  Anton Alexandrov, Veselin Raychev, Mark Niklas Müller, Ce Zhang, Martin Vechev, and Kristina Toutanova.
  Mitigating catastrophic forgetting in language transfer via model merging, 2024.
* Arpit et al. [2022]

  Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong.
  Ensemble of averages: Improving model selection and boosting performance in domain generalization.
  *NeurIPS*, 35:8265–8277, 2022.
* Asadi et al. [2024]

  Nader Asadi, Mahdi Beitollahi, Yasser Khalil, Yinchuan Li, Guojun Zhang, and Xi Chen.
  Does combining parameter-efficient modules improve few-shot transfer accuracy?
  *arXiv preprint arXiv:2402.15414*, 2024.
* Avrahami et al. [2022]

  Omri Avrahami, Dani Lischinski, and Ohad Fried.
  Gan cocktail: mixing gans without dataset access.
  In *ECCV*, pages 205–221. Springer, 2022.
* Bhardwaj et al. [2024]

  Rishabh Bhardwaj, Do Duc Anh, and Soujanya Poria.
  Language models are homer simpson! safety re-alignment of fine-tuned language models through task arithmetic.
  *arXiv preprint arXiv:2402.11746*, 2024.
* Biggs et al. [2024]

  Benjamin Biggs, Arjun Seshadri, Yang Zou, Achin Jain, Aditya Golatkar, Yusheng Xie, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto.
  Diffusion soup: Model merging for text-to-image diffusion models.
  *arXiv preprint arXiv:2406.08431*, 2024.
* Borkan et al. [2019]

  Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman.
  Nuanced metrics for measuring unintended bias with real data for text classification.
  In *WWW*, pages 491–500, 2019.
* Cai et al. [2023]

  Ruisi Cai, Zhenyu Zhang, and Zhangyang Wang.
  Robust weight signatures: gaining robustness as easy as patching weights?
  In *ICML*, pages 3495–3506. PMLR, 2023.
* Cha et al. [2021]

  Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park.
  Swad: Domain generalization by seeking flat minima.
  *NeurIPS*, 34:22405–22418, 2021.
* Chen et al. [2024a]

  Chi Chen, Yiyang Du, Zheng Fang, Ziyue Wang, Fuwen Luo, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Maosong Sun, et al.
  Model composition for multimodal large language models.
  *ACL*, 2024a.
* Chen et al. [2024b]

  Guangyao Chen, Peixi Peng, Yangru Huang, Mengyue Geng, and Yonghong Tian.
  Adaptive discovering and merging for incremental novel class discovery.
  In *AAAI*, volume 38, pages 11276–11284, 2024b.
* Chen et al. [2018]

  Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich.
  Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks.
  In *ICML*, pages 794–803. PMLR, 2018.
* Cheng et al. [2024]

  Feng Cheng, Ziyang Wang, Yi-Lin Sung, Yan-Bo Lin, Mohit Bansal, and Gedas Bertasius.
  Dam: Dynamic adapter merging for continual video qa learning.
  *arXiv preprint arXiv:2403.08755*, 2024.
* Chitale et al. [2023]

  Rajas Chitale, Ankit Vaidya, Aditya Kane, and Archana Ghotkar.
  Task arithmetic with lora for continual learning.
  *arXiv preprint arXiv:2311.02428*, 2023.
* Choshen et al. [2022]

  Leshem Choshen, Elad Venezian, Noam Slonim, and Yoav Katz.
  Fusing finetuned models for better pretraining.
  *arXiv preprint arXiv:2204.03044*, 2022.
* Choudhary et al. [2020]

  Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Sarangapani.
  A comprehensive survey on model compression and acceleration.
  *Artificial Intelligence Review*, 53:5113–5155, 2020.
* Chowdhery et al. [2023]

  Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.
  Palm: Scaling language modeling with pathways.
  *Journal of Machine Learning Research*, 24(240):1–113, 2023.
* Chronopoulou et al. [2023a]

  Alexandra Chronopoulou, Matthew E Peters, Alexander Fraser, and Jesse Dodge.
  Adaptersoup: Weight averaging to improve generalization of pretrained language models.
  In *EACL*, pages 2009–2018, 2023a.
* Chronopoulou et al. [2023b]

  Alexandra Chronopoulou, Jonas Pfeiffer, Joshua Maynez, Xinyi Wang, Sebastian Ruder, and Priyanka Agrawal.
  Language and task arithmetic with parameter-efficient layers for zero-shot summarization.
  *arXiv preprint arXiv:2311.09344*, 2023b.
* Cong et al. [2024]

  Tianshuo Cong, Delong Ran, Zesen Liu, Xinlei He, Jinyuan Liu, Yichen Gong, Qi Li, Anyu Wang, and Xiaoyun Wang.
  Have you merged my model? on the robustness of large language model ip protection methods against model merging.
  *arXiv preprint arXiv:2404.05188*, 2024.
* Crisostomi et al. [2024]

  Donato Crisostomi, Marco Fumero, Daniele Baieri, Florian Bernard, and Emanuele Rodolà.
  c2​m3superscript𝑐2superscript𝑚3c^{2}m^{3}: Cycle-consistent multi-model merging.
  *arXiv preprint arXiv:2405.17897*, 2024.
* Croce et al. [2023]

  Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, and Sven Gowal.
  Seasoning model soups for robustness to adversarial and natural distribution shifts.
  In *CVPR*, pages 12313–12323, 2023.
* Daheim et al. [2024]

  Nico Daheim, Thomas Möllenhoff, Edoardo Ponti, Iryna Gurevych, and Mohammad Emtiyaz Khan.
  Model merging by uncertainty-based gradient matching.
  In *ICLR*, 2024.
* Davari and Belilovsky [2023]

  MohammadReza Davari and Eugene Belilovsky.
  Model breadcrumbs: Scaling multi-task model merging with sparse masks.
  *arXiv preprint arXiv:2312.06795*, 2023.
* Dekoninck et al. [2024]

  Jasper Dekoninck, Marc Fischer, Luca Beurer-Kellner, and Martin Vechev.
  Controlled text generation via language model arithmetic.
  *ICLR*, 2024.
* Demir et al. [2024]

  Caglar Demir, Arnab Sharma, and Axel-Cyrille Ngonga Ngomo.
  Adaptive stochastic weight averaging.
  *JMLR*, 2024.
* Dietterich et al. [2002]

  Thomas G Dietterich et al.
  Ensemble learning.
  *The handbook of brain theory and neural networks*, 2(1):110–125, 2002.
* Ding et al. [2023]

  Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al.
  Parameter-efficient fine-tuning of large-scale pre-trained language models.
  *Nature Machine Intelligence*, 5(3):220–235, 2023.
* Don-Yehiya et al. [2023]

  Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, and Leshem Choshen.
  Cold fusion: Collaborative descent for distributed multitask finetuning.
  In *ACL*, pages 788–806, 2023.
* Dou et al. [2024]

  Guangyao Dou, Zheyuan Liu, Qing Lyu, Kaize Ding, and Eric Wong.
  Avoiding copyright infringement via machine unlearning.
  *arXiv preprint arXiv:2406.10952*, 2024.
* Draxler et al. [2018]

  Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht.
  Essentially no barriers in neural network energy landscape.
  In *ICML*, pages 1309–1318. PMLR, 2018.
* Entezari et al. [2022]

  Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur.
  The role of permutation invariance in linear mode connectivity of neural networks.
  *ICLR*, 2022.
* Ferbach et al. [2024]

  Damien Ferbach, Baptiste Goujaud, Gauthier Gidel, and Aymeric Dieuleveut.
  Proving linear mode connectivity of neural networks via optimal transport.
  In *AISTATS*, pages 3853–3861. PMLR, 2024.
* Fisher [1922]

  Ronald A Fisher.
  On the mathematical foundations of theoretical statistics.
  *Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character*, 222(594-604):309–368, 1922.
* Foret et al. [2021]

  Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur.
  Sharpness-aware minimization for efficiently improving generalization.
  *ICLR*, 2021.
* Fournier et al. [2024]

  Louis Fournier, Adel Nabli, Masih Aminbeidokhti, Marco Pedersoli, Eugene Belilovsky, and Edouard Oyallon.
  Wash: Train your ensemble with communication-efficient weight shuffling, then average.
  *arXiv preprint arXiv:2405.17517*, 2024.
* Frankle et al. [2020]

  Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin.
  Linear mode connectivity and the lottery ticket hypothesis.
  In *ICML*, pages 3259–3269. PMLR, 2020.
* Fu et al. [2024]

  Tingchen Fu, Deng Cai, Lemao Liu, Shuming Shi, and Rui Yan.
  Disperse-then-merge: Pushing the limits of instruction tuning via alignment tax reduction.
  *ACL*, 2024.
* Gallego [2024]

  Victor Gallego.
  Merging improves self-critique against jailbreak attacks.
  *arXiv preprint arXiv:2406.07188*, 2024.
* Gao et al. [2024]

  Lei Gao, Yue Niu, Tingting Tang, Salman Avestimehr, and Murali Annavaram.
  Ethos: Rectifying language models in orthogonal parameter space.
  *arXiv preprint arXiv:2403.08994*, 2024.
* Garipov et al. [2018]

  Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson.
  Loss surfaces, mode connectivity, and fast ensembling of dnns.
  *NeurIPS*, 31, 2018.
* Goddard et al. [2024a]

  Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz.
  Arcee’s mergekit: A toolkit for merging large language models.
  *arXiv preprint arXiv:2403.13257*, 2024a.
* Goddard et al. [2024b]

  Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz.
  Arcee’s mergekit: A toolkit for merging large language models.
  *arXiv preprint arXiv:2403.13257*, 2024b.
* Gupta et al. [2020]

  Vipul Gupta, Santiago Akle Serrano, and Dennis DeCoste.
  Stochastic weight averaging in parallel: Large-batch training that generalizes well.
  In *ICLR*. OpenReview.net, 2020.
* Hammoud et al. [2024]

  Hasan Abed Al Kader Hammoud, Umberto Michieli, Fabio Pizzati, Philip Torr, Adel Bibi, Bernard Ghanem, and Mete Ozay.
  Model merging and safety alignment: One bad model spoils the bunch.
  *arXiv preprint arXiv:2406.14563*, 2024.
* Hardt et al. [2016]

  Moritz Hardt, Ben Recht, and Yoram Singer.
  Train faster, generalize better: Stability of stochastic gradient descent.
  In *ICML*, pages 1225–1234. PMLR, 2016.
* He et al. [2023]

  Shwai He, Run-Ze Fan, Liang Ding, Li Shen, Tianyi Zhou, and Dacheng Tao.
  Mera: Merging pretrained adapters for few-shot learning.
  *arXiv preprint arXiv:2308.15982*, 2023.
* He and Xiao [2023]

  Yang He and Lingao Xiao.
  Structured pruning for deep convolutional neural networks: A survey.
  *TPAMI*, 2023.
* Hendrycks and Dietterich [2019]

  Dan Hendrycks and Thomas Dietterich.
  Benchmarking neural network robustness to common corruptions and perturbations.
  *ICLR*, 2019.
* Ho et al. [2020]

  Jonathan Ho, Ajay Jain, and Pieter Abbeel.
  Denoising diffusion probabilistic models.
  *NeurIPS*, 33:6840–6851, 2020.
* Hoofnagle et al. [2019]

  Chris Jay Hoofnagle, Bart Van Der Sloot, and Frederik Zuiderveen Borgesius.
  The european union general data protection regulation: what it is and what it means.
  *Information & Communications Technology Law*, 28(1):65–98, 2019.
* Horoi et al. [2024]

  Stefan Horoi, Albert Manuel Orozco Camacho, Eugene Belilovsky, and Guy Wolf.
  Harmony in diversity: Merging neural networks with canonical correlation analysis.
  In *ICML*, 2024.
* Hu et al. [2022]

  Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
  LoRA: Low-rank adaptation of large language models.
  In *ICLR*, 2022.
* Hu et al. [2024]

  Xinshuo Hu, Dongfang Li, Baotian Hu, Zihao Zheng, Zhenyu Liu, and Min Zhang.
  Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation.
  In *AAAI*, volume 38, pages 18252–18260, 2024.
* Huang et al. [2024a]

  Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin.
  Lorahub: Efficient cross-task generalization via dynamic lora composition.
  *COLM*, 2024a.
* Huang et al. [2024b]

  Chenyu Huang, Peng Ye, Tao Chen, Tong He, Xiangyu Yue, and Wanli Ouyang.
  Emr-merging: Tuning-free high-performance model merging.
  *arXiv preprint arXiv:2405.17461*, 2024b.
* Huang et al. [2024c]

  Shih-Cheng Huang, Pin-Zu Li, Yu-Chi Hsu, Kuang-Ming Chen, Yu Tung Lin, Shih-Kai Hsiao, Richard Tzong-Han Tsai, and Hung-yi Lee.
  Chat vector: A simple approach to equip llms with new language chat capabilities.
  *ACL*, 2024c.
* Ilharco et al. [2022]

  Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt.
  Patching open-vocabulary models by interpolating weights.
  *NeurIPS*, 35:29262–29277, 2022.
* Ilharco et al. [2023]

  Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi.
  Editing models with task arithmetic.
  In *ICLR*, 2023.
* Imfeld et al. [2024]

  Moritz Imfeld, Jacopo Graldi, Marco Giordano, Thomas Hofmann, Sotiris Anagnostidis, and Sidak Pal Singh.
  Transformer fusion with optimal transport.
  *ICLR*, 2024.
* Izmailov et al. [2018]

  P Izmailov, AG Wilson, D Podoprikhin, D Vetrov, and T Garipov.
  Averaging weights leads to wider optima and better generalization.
  In *UAI*, pages 876–885, 2018.
* Jacot et al. [2018]

  Arthur Jacot, Franck Gabriel, and Clément Hongler.
  Neural tangent kernel: Convergence and generalization in neural networks.
  *NeurIPS*, 31, 2018.
* Jain et al. [2018]

  Prateek Jain, Sham M Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford.
  Parallelizing stochastic gradient descent for least squares regression: mini-batching, averaging, and model misspecification.
  *JMLR*, 18(223):1–42, 2018.
* Jain et al. [2023]

  Samyak Jain, Sravanti Addepalli, Pawan Kumar Sahu, Priyam Dey, and R Venkatesh Babu.
  Dart: Diversify-aggregate-repeat training improves generalization of neural networks.
  In *CVPR*, pages 16048–16059, 2023.
* Jang et al. [2024]

  Dong-Hwan Jang, Sangdoo Yun, and Dongyoon Han.
  Model stock: All we need is just a few fine-tuned models.
  *arXiv preprint arXiv:2403.19522*, 2024.
* Jang et al. [2023]

  Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu.
  Personalized soups: Personalized large language model alignment via post-hoc parameter merging.
  *arXiv preprint arXiv:2310.11564*, 2023.
* Jhunjhunwala et al. [2024]

  Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi.
  Fedfisher: Leveraging fisher information for one-shot federated learning.
  In *AISTATS*, pages 1612–1620. PMLR, 2024.
* Ji et al. [2019]

  Shaoxiong Ji, Shirui Pan, Guodong Long, Xue Li, Jing Jiang, and Zi Huang.
  Learning private neural language modeling with attentive aggregation.
  In *IJCNN*, pages 1–8. IEEE, 2019.
* Jiang et al. [2024]

  Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.
  Mixtral of experts.
  *arXiv preprint arXiv:2401.04088*, 2024.
* Jiang et al. [2023]

  Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, and Mingsheng Long.
  Forkmerge: Mitigating negative transfer in auxiliary-task learning.
  *NeurIPS*, 36, 2023.
* Jin et al. [2024]

  Ruochen Jin, Bojian Hou, Jiancong Xiao, Weijie Su, and Li Shen.
  Fine-tuning linear layers only is a simple yet effective way for task arithmetic, 2024.
* Jin et al. [2023]

  Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng.
  Dataless knowledge fusion by merging weights of language models.
  In *ICLR*, 2023.
* Jolicoeur-Martineau et al. [2024]

  Alexia Jolicoeur-Martineau, Emy Gervais, Kilian Fatras, Yan Zhang, and Simon Lacoste-Julien.
  Population parameter averaging (papa).
  *TMLR*, 2024.
* Jordan et al. [2023]

  Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur.
  Repair: Renormalizing permuted activations for interpolation repair.
  *ICLR*, 2023.
* Kaddour [2022]

  Jean Kaddour.
  Stop wasting my time! saving days of imagenet and bert training with latest weight averaging.
  *NeurIPS Workshop*, 2022.
* Kaddour et al. [2022]

  Jean Kaddour, Linqing Liu, Ricardo Silva, and Matt J Kusner.
  When do flat minima optimizers work?
  *NeurIPS*, 35:16577–16595, 2022.
* Keskar et al. [2019]

  Nitish Shirish Keskar, Bryan McCann, Lav R Varshney, Caiming Xiong, and Richard Socher.
  Ctrl: A conditional transformer language model for controllable generation.
  *arXiv preprint arXiv:1909.05858*, 2019.
* Kim et al. [2023]

  Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, et al.
  Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling.
  *arXiv preprint arXiv:2312.15166*, 2023.
* Kirkpatrick et al. [2017]

  James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al.
  Overcoming catastrophic forgetting in neural networks.
  *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017.
* Klimaszewski et al. [2024]

  Mateusz Klimaszewski, Piotr Andruszkiewicz, and Alexandra Birch.
  No train but gain: Language arithmetic for training-free language adapters enhancement.
  *arXiv preprint arXiv:2404.15737*, 2024.
* Krause et al. [2021]

  Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani.
  Gedi: Generative discriminator guided sequence generation.
  *EMNLP*, 2021.
* Li et al. [2018]

  Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein.
  Visualizing the loss landscape of neural nets.
  *NeurIPS*, 31, 2018.
* Li et al. [2024a]

  Jialu Li, Jaemin Cho, Yi-Lin Sung, Jaehong Yoon, and Mohit Bansal.
  Selma: Learning and merging skill-specific text-to-image experts with auto-generated data.
  *arXiv preprint arXiv:2403.06952*, 2024a.
* Li et al. [2023a]

  Linyang Li, Botian Jiang, Pengyu Wang, Ke Ren, Hang Yan, and Xipeng Qiu.
  Watermarking llms with weight quantization.
  *arXiv preprint arXiv:2310.11237*, 2023a.
* Li et al. [2024b]

  Lu Li, Tianyu Zhang, Zhiqi Bu, Suyuchen Wang, Huan He, Jie Fu, Yonghui Wu, Jiang Bian, Yong Chen, and Yoshua Bengio.
  Map: Low-compute model merging with amortized pareto fronts via quadratic approximation, 2024b.
* Li et al. [2022]

  Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer.
  Branch-train-merge: Embarrassingly parallel training of expert language models.
  *arXiv preprint arXiv:2208.03306*, 2022.
* Li et al. [2024c]

  Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen.
  Merge, then compress: Demystify efficient smoe with hints from its routing policy.
  *ICLR*, 2024c.
* Li et al. [2023b]

  Tao Li, Zhehao Huang, Qinghua Tao, Yingwen Wu, and Xiaolin Huang.
  Trainable weight averaging: Efficient training by optimizing historical solutions.
  In *ICLR*, 2023b.
* Li et al. [2024d]

  Tao Li, Weisen Jiang, Fanghui Liu, Xiaolin Huang, and James T Kwok.
  Scalable learned model soup on a single gpu: An efficient subspace training strategy.
  *arXiv preprint arXiv:2407.03641*, 2024d.
* Li et al. [2023c]

  Weishi Li, Yong Peng, Miao Zhang, Liang Ding, Han Hu, and Li Shen.
  Deep model fusion: A survey.
  *arXiv preprint arXiv:2309.15698*, 2023c.
* Li et al. [2023d]

  Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto.
  Alpacaeval: An automatic evaluator of instruction-following models, 2023d.
* Lin et al. [2024]

  Tzu-Han Lin, Chen-An Li, Hung-yi Lee, and Yun-Nung Chen.
  Dogerm: Equipping reward models with domain knowledge through model merging.
  *arXiv preprint arXiv:2407.01470*, 2024.
* Liu et al. [2022a]

  Chang Liu, Chenfei Lou, Runzhong Wang, Alan Yuhan Xi, Li Shen, and Junchi Yan.
  Deep neural network fusion via graph matching with applications to model ensemble and federated learning.
  In *ICML*, pages 13857–13869. PMLR, 2022a.
* Liu et al. [2024a]

  Deyuan Liu, Zecheng Wang, Bingning Wang, Weipeng Chen, Chunshan Li, Zhiying Tu, Dianhui Chu, Bo Li, and Dianbo Sui.
  Checkpoint merging via bayesian optimization in llm pretraining.
  *arXiv preprint arXiv:2403.19390*, 2024a.
* Liu et al. [2024b]

  Enshu Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B Blaschko, Sergey Yekhanin, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang.
  Linear combination of saved checkpoints makes consistency and diffusion models better.
  *arXiv preprint arXiv:2404.02241*, 2024b.
* Liu et al. [2022b]

  Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel.
  Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning.
  *NeurIPS*, 35:1950–1965, 2022b.
* Liu et al. [2024c]

  Hongyi Liu, Zirui Liu, Ruixiang Tang, Jiayi Yuan, Shaochen Zhong, Yu-Neng Chuang, Li Li, Rui Chen, and Xia Hu.
  Lora-as-an-attack! piercing llm safety under the share-and-play scenario.
  *arXiv preprint arXiv:2403.00108*, 2024c.
* Liu and Soatto [2023]

  Tian Yu Liu and Stefano Soatto.
  Tangent model composition for ensembling and continual fine-tuning.
  In *ICCV*, pages 18676–18686, 2023.
* Liu et al. [2024d]

  Tian Yu Liu, Aditya Golatkar, and Stefano Soatto.
  Tangent transformers for composition, privacy and removal.
  In *ICLR*, 2024d.
* Liu et al. [2024e]

  Zheyuan Liu, Guangyao Dou, Zhaoxuan Tan, Yijun Tian, and Meng Jiang.
  Towards safer large language models through machine unlearning.
  *arXiv preprint arXiv:2402.10058*, 2024e.
* Lu et al. [2024a]

  Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin, and Chang Zhou.
  Online merging optimizers for boosting rewards and mitigating tax in alignment.
  *arXiv preprint arXiv:2405.17931*, 2024a.
* Lu et al. [2024b]

  Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng.
  Twin-merging: Dynamic integration of modular expertise in model merging.
  *arXiv preprint arXiv:2406.15479*, 2024b.
* Lv et al. [2023]

  Xingtai Lv, Ning Ding, Yujia Qin, Zhiyuan Liu, and Maosong Sun.
  Parameter-efficient weight ensembling facilitates task-level knowledge transfer.
  In *ACL*, pages 270–282, 2023.
* Ma et al. [2018]

  Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi.
  Modeling task relationships in multi-task learning with multi-gate mixture-of-experts.
  In *SIGKDD*, pages 1930–1939. ACM, 2018.
* Marcel and Rodriguez [2010]

  Sébastien Marcel and Yann Rodriguez.
  Torchvision the machine-vision package of torch.
  In *ACM MM*, pages 1485–1488, 2010.
* Marczak et al. [2024]

  Daniel Marczak, Bartłomiej Twardowski, Tomasz Trzciński, and Sebastian Cygert.
  Magmax: Leveraging model merging for seamless continual learning.
  In *ECCV*, 2024.
* Matena and Raffel [2022]

  Michael S Matena and Colin A Raffel.
  Merging models with fisher-weighted averaging.
  *NeurIPS*, 35:17703–17716, 2022.
* McMahan et al. [2017]

  Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
  Communication-efficient learning of deep networks from decentralized data.
  In *AISTATS*, pages 1273–1282. PMLR, 2017.
* Mishra et al. [2022]

  Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi.
  Cross-task generalization via natural language crowdsourcing instructions.
  In *ACL*, pages 3470–3487. Association for Computational Linguistics (ACL), 2022.
* Muqeeth et al. [2024]

  Mohammed Muqeeth, Haokun Liu, and Colin Raffel.
  Soft merging of experts with adaptive routing.
  *TMLR*, 2024.
* Nagarajan and Kolter [2019]

  Vaishnavh Nagarajan and J Zico Kolter.
  Uniform convergence may be unable to explain generalization in deep learning.
  *NeurIPS*, 32, 2019.
* Nair et al. [2024]

  Nithin Gopalakrishnan Nair, Jeya Maria Jose Valanarasu, and Vishal M Patel.
  Maxfusion: Plug&play multi-modal generation in text-to-image diffusion models.
  *arXiv preprint arXiv:2404.09977*, 2024.
* Navon et al. [2024]

  Aviv Navon, Aviv Shamsian, Ethan Fetaya, Gal Chechik, Nadav Dym, and Haggai Maron.
  Equivariant deep weight space alignment.
  *ICML*, 2024.
* Nguyen et al. [2023a]

  Dang Nguyen, Khai Nguyen, Nhat Ho, Dinh Phung, and Hung Bui.
  Model fusion of heterogeneous neural networks via cross-layer alignment.
  *ICASSP*, 2023a.
* Nguyen et al. [2023b]

  Dang Nguyen, Trang Nguyen, Khai Nguyen, Dinh Phung, Hung Bui, and Nhat Ho.
  On cross-layer alignment for model fusion of heterogeneous neural networks.
  In *ICASSP*, pages 1–5. IEEE, 2023b.
* Ni et al. [2023]

  Shiwen Ni, Dingwei Chen, Chengming Li, Xiping Hu, Ruifeng Xu, and Min Yang.
  Forgetting before learning: Utilizing parametric arithmetic for knowledge updating in large language models.
  *arXiv preprint arXiv:2311.08011*, 2023.
* Ortiz-Jimenez et al. [2023]

  Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard.
  Task arithmetic in the tangent space: Improved editing of pre-trained models.
  *NeurIPS*, 2023.
* Pardau [2018]

  Stuart L Pardau.
  The california consumer privacy act: Towards a european-style privacy regime in the united states.
  *J. Tech. L. & Pol’y*, 23:68, 2018.
* Peña et al. [2023]

  Fidel A Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, and Marco Pedersoli.
  Re-basin via implicit sinkhorn differentiation.
  In *CVPR*, pages 20237–20246, 2023.
* Piczak [2015]

  Karol J Piczak.
  Esc: Dataset for environmental sound classification.
  In *ACM MM*, pages 1015–1018, 2015.
* Porrello et al. [2024]

  Angelo Porrello, Lorenzo Bonicelli, Pietro Buzzega, Monica Millunzi, Simone Calderara, and Rita Cucchiara.
  A second-order perspective on compositionality and incremental learning.
  *arXiv preprint arXiv:2405.16350*, 2024.
* Qazi et al. [2024]

  Mohammad Areeb Qazi, Ibrahim Almakky, Anees Ur Rehman Hashmi, Santosh Sanjeev, and Mohammad Yaqub.
  Dynammo: Dynamic model merging for efficient class incremental learning for medical images.
  *arXiv preprint arXiv:2404.14099*, 2024.
* Radford et al. [2018]

  Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al.
  Improving language understanding by generative pre-training.
  2018.
* Radford et al. [2019]

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
  Language models are unsupervised multitask learners.
  *OpenAI blog*, 1(8):9, 2019.
* Rafailov et al. [2023]

  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a reward model.
  *NeurIPS*, 36, 2023.
* Rame et al. [2022]

  Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord.
  Diverse weight averaging for out-of-distribution generalization.
  *NeurIPS*, 35:10821–10836, 2022.
* Ramé et al. [2023]

  Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David Lopez-Paz.
  Model ratatouille: Recycling diverse models for out-of-distribution generalization.
  In *ICML*, pages 28656–28679. PMLR, 2023.
* Rame et al. [2023]

  Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord.
  Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards.
  *NeurIPS*, 36, 2023.
* Ramé et al. [2024a]

  Alexandre Ramé, Johan Ferret, Nino Vieillard, Robert Dadashi, Léonard Hussenot, Pierre-Louis Cedoz, Pier Giuseppe Sessa, Sertan Girgin, Arthur Douillard, and Olivier Bachem.
  Warp: On the benefits of weight averaged rewarded policies.
  *arXiv preprint arXiv:2406.16768*, 2024a.
* Ramé et al. [2024b]

  Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret.
  Warm: On the benefits of weight averaged reward models.
  *ICML*, 2024b.
* Rebuffi et al. [2023]

  Sylvestre-Alvise Rebuffi, Francesco Croce, and Sven Gowal.
  Revisiting adapters with adversarial training.
  2023.
* Robins [1995]

  Anthony Robins.
  Catastrophic forgetting, rehearsal and pseudorehearsal.
  *Connection Science*, 1995.
* Rombach et al. [2022]

  Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.
  High-resolution image synthesis with latent diffusion models.
  In *CVPR*, pages 10684–10695, 2022.
* Ruiz et al. [2023]

  Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman.
  Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation.
  In *CVPR*, pages 22500–22510, 2023.
* [141]

  Simo Ryu.
  Merging loras.
  In *https://github.com/ cloneofsimo/lora*.
* Sagi and Rokach [2018]

  Omer Sagi and Lior Rokach.
  Ensemble learning: A survey.
  *Wiley interdisciplinary reviews: data mining and knowledge discovery*, 8(4):e1249, 2018.
* Sanyal et al. [2023]

  Sunny Sanyal, Atula Tejaswi Neerkaje, Jean Kaddour, Abhishek Kumar, et al.
  Early weight averaging meets high learning rates for llm pre-training.
  In *NeurIPS Workshop*, 2023.
* Sener and Koltun [2018]

  Ozan Sener and Vladlen Koltun.
  Multi-task learning as multi-objective optimization.
  In *NeurIPS*, pages 525–536, 2018.
* Shah et al. [2023]

  Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani.
  Ziplora: Any subject in any style by effectively merging loras.
  *arXiv preprint arXiv:2311.13600*, 2023.
* Shoemake [1985]

  Ken Shoemake.
  Animating rotation with quaternion curves.
  In *Proceedings of the 12th annual conference on Computer graphics and interactive techniques*, pages 245–254, 1985.
* Shukor et al. [2023]

  Mustafa Shukor, Corentin Dancette, Alexandre Rame, and Matthieu Cord.
  Unival: Unified model for image, video, audio and language tasks.
  *TMLR*, 2023.
* Singh and Jaggi [2020]

  Sidak Pal Singh and Martin Jaggi.
  Model fusion via optimal transport.
  *NeurIPS*, 33:22045–22055, 2020.
* Smith and Gashler [2017]

  Joshua Smith and Michael Gashler.
  An investigation of how neural networks learn from the experiences of peers through periodic weight averaging.
  In *2017 16th IEEE ICML and Applications (ICMLA)*, pages 731–736. IEEE, 2017.
* Song et al. [2023]

  Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever.
  Consistency models.
  In *ICML*, pages 32211–32252, 2023.
* Stoica et al. [2024]

  George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman.
  Zipit! merging models from different tasks without training.
  *ICLR*, 2024.
* Sun et al. [2020]

  Ximeng Sun, Rameswar Panda, Rogerio Feris, and Kate Saenko.
  Adashare: Learning what to share for efficient deep multi-task learning.
  *NeurIPS*, 33:8728–8740, 2020.
* Sundar et al. [2024]

  Anirudh S Sundar, Chao-Han Huck Yang, David M Chan, Shalini Ghosh, Venkatesh Ravichandran, and Phani Sankar Nidadavolu.
  Multimodal attention merging for improved speech recognition and audio event classification.
  *ICASSP Workshop*, 2024.
* Sung et al. [2023]

  Yi-Lin Sung, Linjie Li, Kevin Lin, Zhe Gan, Mohit Bansal, and Lijuan Wang.
  An empirical study of multimodal model merging.
  *EMNLP*, 2023.
* Tam et al. [2023]

  Derek Tam, Mohit Bansal, and Colin Raffel.
  Merging by matching models in task subspaces.
  *arXiv preprint arXiv:2312.04339*, 2023.
* Tang et al. [2023]

  Anke Tang, Li Shen, Yong Luo, Liang Ding, Han Hu, Bo Du, and Dacheng Tao.
  Concrete subspace learning based interference elimination for multi-task model fusion.
  *arXiv preprint arXiv:2312.06173*, 2023.
* Tang et al. [2024a]

  Anke Tang, Li Shen, Yong Luo, Han Hu, Bo Do, and Dacheng Tao.
  FusionBench: A Comprehensive Benchmark of Deep Model Fusion.
  (arXiv:2406.03280), 2024a.
* Tang et al. [2024b]

  Anke Tang, Li Shen, Yong Luo, Shiwei Liu, Han Hu, and Bo Du.
  Towards efficient pareto set approximation via mixture of experts based model fusion.
  *arXiv preprint arXiv:2406.09770*, 2024b.
* Tang et al. [2024c]

  Anke Tang, Li Shen, Yong Luo, Nan Yin, Lefei Zhang, and Dacheng Tao.
  Merging multi-task models via weight-ensembling mixture of experts.
  *ICML*, 2024c.
* Tang et al. [2024d]

  Anke Tang, Li Shen, Yong Luo, Yibing Zhan, Han Hu, Bo Du, Yixin Chen, and Dacheng Tao.
  Parameter efficient multi-task model fusion with partial linearization.
  *ICLR*, 2024d.
* Tarvainen and Valpola [2017]

  Antti Tarvainen and Harri Valpola.
  Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
  *NeurIPS*, 30, 2017.
* Tatro et al. [2020]

  Norman Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, and Rongjie Lai.
  Optimizing mode connectivity via neuron alignment.
  *NeurIPS*, 33:15300–15311, 2020.
* Team et al. [2023]

  Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al.
  Gemini: a family of highly capable multimodal models.
  *arXiv preprint arXiv:2312.11805*, 2023.
* Thennal et al. [2024]

  DK Thennal, Ganesh Nathan, and MS Suchithra.
  Fisher mask nodes for language model merging.
  In *LREC-COLING*, pages 7349–7355, 2024.
* Thudi et al. [2022]

  Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot.
  Unrolling sgd: Understanding factors influencing machine unlearning.
  In *EuroS&P*, pages 303–319. IEEE, 2022.
* Touvron et al. [2023a]

  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
  Llama: Open and efficient foundation language models.
  *arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. [2023b]

  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
  Llama 2: Open foundation and fine-tuned chat models.
  *arXiv preprint arXiv:2307.09288*, 2023b.
* Utans [1996]

  Joachim Utans.
  Weight averaging for neural networks and local resampling schemes.
  In *AAAI Workshop*, pages 133–138. Citeseer, 1996.
* Vaswani et al. [2017]

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  Attention is all you need.
  *NeurIPS*, 30, 2017.
* von Oswald et al. [2021]

  Johannes von Oswald, Seijin Kobayashi, Alexander Meulemans, Christian Henning, Benjamin F. Grewe, and João Sacramento.
  Neural networks with late-phase weights.
  In *ICLR*, 2021.
* Wan et al. [2024a]

  Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, and Shuming Shi.
  Knowledge fusion of large language models.
  *ICLR*, 2024a.
* Wan et al. [2024b]

  Fanqi Wan, Ziyi Yang, Longguang Zhong, Xiaojun Quan, Xinting Huang, and Wei Bi.
  Fusechat: Knowledge fusion of chat models.
  *arXiv preprint arXiv:2402.16107*, 2024b.
* Wang et al. [2023a]

  Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu.
  Openchat: Advancing open-source language models with mixed-quality data.
  *arXiv preprint arXiv:2309.11235*, 2023a.
* Wang et al. [2020a]

  Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni.
  Federated learning with matched averaging.
  In *ICLR*, 2020a.
* Wang et al. [2020b]

  Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor.
  Tackling the objective inconsistency problem in heterogeneous federated optimization.
  *NeurIPS*, 33:7611–7623, 2020b.
* Wang et al. [2024a]

  Ke Wang, Nikolaos Dimitriadis, Guillermo Ortiz-Jimenez, François Fleuret, and Pascal Frossard.
  Localizing task information for improved model merging and compression.
  *ICML*, 2024a.
* Wang et al. [2024b]

  Peng Wang, Li Shen, Zerui Tao, Shuaida He, and Dacheng Tao.
  Generalization analysis of stochastic weight averaging with general sampling.
  In *ICML*, 2024b.
* Wang et al. [2023b]

  Zhenyi Wang, Enneng Yang, Li Shen, and Heng Huang.
  A comprehensive survey of forgetting in deep learning beyond continual learning.
  *arXiv preprint arXiv:2307.09218*, 2023b.
* Wei et al. [2023]

  Alexander Wei, Nika Haghtalab, and Jacob Steinhardt.
  Jailbroken: How does llm safety training fail?
  *NeurIPS*, 36, 2023.
* Wen et al. [2020]

  Yeming Wen, Dustin Tran, and Jimmy Ba.
  Batchensemble: an alternative approach to efficient ensemble and lifelong learning.
  *arXiv preprint arXiv:2002.06715*, 2020.
* Wightman [2019]

  Ross Wightman.
  Pytorch image models.
  <https://github.com/rwightman/pytorch-image-models>, 2019.
* Wolf et al. [2019]

  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.
  Huggingface’s transformers: State-of-the-art natural language processing.
  *arXiv preprint arXiv:1910.03771*, 2019.
* Wortsman et al. [2022a]

  Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al.
  Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
  In *ICML*, pages 23965–23998. PMLR, 2022a.
* Wortsman et al. [2022b]

  Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al.
  Robust fine-tuning of zero-shot models.
  In *CVPR*, pages 7959–7971, 2022b.
* Wu et al. [2022]

  Baoyuan Wu, Hongrui Chen, Mingda Zhang, Zihao Zhu, Shaokui Wei, Danni Yuan, and Chao Shen.
  Backdoorbench: A comprehensive benchmark of backdoor learning.
  In *NeurIPS Datasets and Benchmarks Track*, 2022.
* Wu et al. [2024]

  Xun Wu, Shaohan Huang, and Furu Wei.
  Mole: Mixture of lora experts.
  In *ICLR*, 2024.
* Xiao et al. [2024]

  Shitao Xiao, Zheng Liu, Peitian Zhang, and Xingrun Xing.
  Lm-cocktail: Resilient tuning of language models via model merging.
  *ACL*, 2024.
* Xu et al. [2024a]

  Jiashu Xu, Fei Wang, Mingyu Derek Ma, Pang Wei Koh, Chaowei Xiao, and Muhao Chen.
  Instructional fingerprinting of large language models.
  *arXiv preprint arXiv:2401.12255*, 2024a.
* Xu et al. [2024b]

  Zhengqi Xu, Ke Yuan, Huiqiong Wang, Yong Wang, Mingli Song, and Jie Song.
  Training-free pretrained model merging.
  *CVPR*, 2024b.
* Yadav et al. [2023]

  Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal.
  Resolving interference when merging models.
  *NeurIPS*, 2023.
* Yang et al. [2023a]

  Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al.
  Baichuan 2: Open large-scale language models.
  *arXiv preprint arXiv:2309.10305*, 2023a.
* Yang et al. [2023b]

  Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo.
  Adatask: A task-aware adaptive learning rate approach to multi-task learning.
  In *AAAI*, volume 37, pages 10745–10753, 2023b.
* Yang et al. [2024a]

  Enneng Yang, Li Shen, Zhenyi Wang, Guibing Guo, Xiaojun Chen, Xingwei Wang, and Dacheng Tao.
  Representation surgery for multi-task model merging.
  *ICML*, 2024a.
* Yang et al. [2024b]

  Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao.
  Adamerging: Adaptive model merging for multi-task learning.
  *ICLR*, 2024b.
* Yang et al. [2019]

  Guandao Yang, Tianyi Zhang, Polina Kirichenko, Junwen Bai, Andrew Gordon Wilson, and Chris De Sa.
  Swalp: Stochastic weight averaging in low precision training.
  In *ICML*, pages 7015–7024. PMLR, 2019.
* Yao et al. [2023]

  Yuanshun Yao, Xiaojun Xu, and Yang Liu.
  Large language model unlearning.
  In *Socially Responsible Language Modelling Research*, 2023.
* Ye et al. [2023]

  Peng Ye, Chenyu Huang, Mingzhu Shen, Tao Chen, Yongqi Huang, Yuning Zhang, and Wanli Ouyang.
  Merging vision transformers from different tasks and domains.
  *arXiv preprint arXiv:2312.16240*, 2023.
* Ye et al. [2021]

  Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren.
  Crossfit: A few-shot learning challenge for cross-task generalization in nlp.
  In *EMNLP*, pages 7163–7189, 2021.
* Yi et al. [2024]

  Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, and Liang He.
  A safety realignment framework via subspace-oriented model fusion for large language models.
  *arXiv preprint arXiv:2405.09055*, 2024.
* Yu et al. [2024a]

  Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li.
  Language models are super mario: Absorbing abilities from homologous models as a free lunch.
  *ICML*, 2024a.
* Yu et al. [2024b]

  Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li.
  Extend model merging from fine-tuned to pre-trained large language models via weight disentanglement.
  *arXiv preprint arXiv:2408.03092*, 2024b.
* Yu et al. [2020]

  Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.
  Gradient surgery for multi-task learning.
  *NeurIPS*, 33:5824–5836, 2020.
* Yurochkin et al. [2019]

  Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Nghia Hoang, and Yasaman Khazaeni.
  Bayesian nonparametric federated learning of neural networks.
  In *ICML*, pages 7252–7261. PMLR, 2019.
* Zaman et al. [2023]

  Kerem Zaman, Leshem Choshen, and Shashank Srivastava.
  Fuse to forget: Bias reduction and selective memorization through model fusion.
  *arXiv preprint arXiv:2311.07682*, 2023.
* Zhang et al. [2024a]

  Frederic Z Zhang, Paul Albert, Cristian Rodriguez-Opazo, Anton van den Hengel, and Ehsan Abbasnejad.
  Knowledge composition using task vectors with learned anisotropic scaling.
  *arXiv preprint arXiv:2407.02880*, 2024a.
* Zhang et al. [2020]

  Haoyang Zhang, Ying Wang, Feras Dayoub, and Niko Sünderhauf.
  Swa object detection.
  *arXiv preprint arXiv:2012.12645*, 2020.
* Zhang et al. [2023a]

  Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, and Mingli Song.
  Lookaround optimizer: k𝑘k steps around, 1 step average.
  *NeurIPS*, 36, 2023a.
* Zhang et al. [2023b]

  Jinghan Zhang, Shiqi Chen, Junteng Liu, and Junxian He.
  Composing parameter-efficient modules with arithmetic operations.
  In *NeurIPS*, 2023b.
* Zhang et al. [2023c]

  Jinghan Zhang, Junteng Liu, Junxian He, et al.
  Composing parameter-efficient modules with arithmetic operation.
  *NeurIPS*, 36:12589–12610, 2023c.
* Zhang et al. [2024b]

  Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, and Yuan Tian.
  Badmerging: Backdoor attacks against model merging.
  *CCS*, 2024b.
* Zhao et al. [2024a]

  Yiran Zhao, Wenxuan Zhang, Huiming Wang, Kenji Kawaguchi, and Lidong Bing.
  Adamergex: Cross-lingual transfer with large language models via adaptive adapter merging.
  *arXiv preprint arXiv:2402.18913*, 2024a.
* Zhao et al. [2024b]

  Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, and Fei Wu.
  Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild.
  *ACL*, 2024b.
* Zheng et al. [2024]

  Chujie Zheng, Ziqi Wang, Heng Ji, Minlie Huang, and Nanyun Peng.
  Weak-to-strong extrapolation expedites alignment.
  *arXiv preprint arXiv:2404.16792*, 2024.
* Zheng et al. [2023]

  Hongling Zheng, Li Shen, Anke Tang, Yong Luo, Han Hu, Bo Du, and Dacheng Tao.
  Learn from model beyond fine-tuning: A survey.
  *arXiv preprint arXiv:2310.08184*, 2023.
* Zhou et al. [2024a]

  Yuyan Zhou, Liang Song, Bingning Wang, and Weipeng Chen.
  Metagpt: Merging large language models using model exclusive task arithmetic.
  *arXiv preprint arXiv:2406.11385*, 2024a.
* Zhou et al. [2023]

  Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, and Wei Hu.
  Going beyond linear mode connectivity: The layerwise linear feature connectivity.
  *NeurIPS*, 36:60853–60877, 2023.
* Zhou et al. [2024b]

  Zhanpeng Zhou, Zijun Chen, Yilan Chen, Bo Zhang, and Junchi Yan.
  Cross-task linearity emerges in the pretraining-finetuning paradigm.
  *arXiv preprint arXiv:2402.03660v1*, 2024b.
* Zhu et al. [2024]

  Didi Zhu, Zhongyi Sun, Zexi Li, Tao Shen, Ke Yan, Shouhong Ding, Kun Kuang, and Chao Wu.
  Model tailor: Mitigating catastrophic forgetting in multi-modal large language models.
  *ICML*, 2024.
* Ziegler et al. [2019]

  Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving.
  Fine-tuning language models from human preferences.
  *arXiv preprint arXiv:1909.08593*, 2019.
* Zimmer et al. [2024]

  Max Zimmer, Christoph Spiegel, and Sebastian Pokutta.
  Sparse model soups: A recipe for improved pruning via model averaging.
  In *ICLR*, 2024.
