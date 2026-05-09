---
arxiv: '2110.04361'
authors:
- Talip Ucar
- Ehsan Hajiramezanali
- Lindsay Edwards
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation
  Learning'
url: http://arxiv.org/abs/2110.04361v2
year: 2021
---

[2110.04361] SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning














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



# SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning

Talip Uçar, Ehsan Hajiramezanali, Lindsay Edwards
  
  
Respiratory and Immunology, R&D, AstraZeneca
  
{talip.ucar, ehsan.hajiramezanali, lindsay.edwards}@astrazeneca.com

###### Abstract

Self-supervised learning has been shown to be very effective in learning useful representations, and yet much of the success is achieved in data types such as images, audio, and text. The success is mainly enabled by taking advantage of spatial, temporal, or semantic structure in the data through augmentation. However, such structure may not exist in tabular datasets commonly used in fields such as healthcare, making it difficult to design an effective augmentation method, and hindering a similar progress in tabular data setting. In this paper, we introduce a new framework, Subsetting features of Tabular data (SubTab), that turns the task of learning from tabular data into a multi-view representation learning problem by dividing the input features to multiple subsets. We argue that reconstructing the data from the subset of its features rather than its corrupted version in an autoencoder setting can better capture its underlying latent representation. In this framework, the joint representation can be expressed as the aggregate of latent variables of the subsets at test time, which we refer to as *collaborative inference*. Our experiments show that the SubTab achieves the state of the art (SOTA) performance of 98.31% on MNIST in tabular setting, on par with CNN-based SOTA models, and surpasses existing baselines on three other real-world datasets by a significant margin.

## 1 Introduction

In recent years, the self-supervised learning has successfully been used to learn meaningful representations of the data in natural language processing [[34](#bib.bib34), [41](#bib.bib41), [11](#bib.bib11), [28](#bib.bib28), [10](#bib.bib10), [21](#bib.bib21), [9](#bib.bib9)]. A similar success has been achieved in image and audio domains [[7](#bib.bib7), [15](#bib.bib15), [37](#bib.bib37), [5](#bib.bib5), [17](#bib.bib17), [13](#bib.bib13), [8](#bib.bib8)]. This progress is mainly enabled by taking advantage of spatial, semantic, or temporal structure in the data through data augmentation [[7](#bib.bib7)], pretext task generation [[11](#bib.bib11)] and using inductive biases through architectural choices (e.g. CNN for images). However, these methods can be less effective in the lack of such structures and biases in the tabular data commonly used in many fields such as healthcare, advertisement, finance, and law. And some augmentation methods such as cropping, rotation, color transformation etc. are domain specific, and not suitable for tabular setting. The difficulty in designing similarly effective methods tailored for tabular data is one of the reasons why self-supervised learning is under-studied in this domain [[46](#bib.bib46)].

The most common approach in tabular data is to corrupt data through adding noise [[43](#bib.bib43)]. An autoencoder maps corrupted examples of data to a latent space, from which it maps back to uncorrupted data. Through this process, it learns a representation robust to the noise in the input. This approach may not be as effective since it treats all features equally as if features are equally informative. However, perturbing uninformative features may not result in the intended goal of the corruption. A recent work takes advantage of self-supervised learning in tabular data setting by introducing a pretext task [[46](#bib.bib46)], in which a de-noising autoencoder with a classifier attached to representation layer is trained on corrupted data. The classifier’s task is to predict the location of corrupted features. However, this framework still relies on noisy data in the input. Additionally, training a classifier on an imbalanced binary mask for a high-dimensional data may not be ideal to learn meaningful representations.

In this work, we turn the problem of learning representation from a single-view of the data into the one learnt from its multiple views by dividing the features into subsets, akin to cropping in image domain or feature bagging in ensemble learning, to generate different views of the data. Each subset can be considered a different view. We show that reconstructing data from the subset of its features forces the encoder to learn better representation than the ones learned through the existing methods such as adding noise. We train our model in a self-supervised setting and evaluate it on downstream tasks such as classification, and clustering. We use five different datasets; MNIST in tabular format, the cancer genome atlas (TCGA) [[42](#bib.bib42)], human gut metagen-omic samples of obesity cohorts (Obesity) [[36](#bib.bib36), [26](#bib.bib26)], UCI adult income (Income) [[24](#bib.bib24)], and UCI BlogFeedback (Blog) [[4](#bib.bib4)].

SubTab can: i) construct a better representation by using the aggregate of the representation of the subsets, a process that we refer as *collaborative inference* ii) discover the regions of informative features by measuring predictive power of each subset, which is useful especially in high-dimensional data iii) do training and inference in the presence of missing features by ignoring corresponding subsets and iv) use smaller models by reducing input dimension, making it less prone to overfitting.

## 2 Method

![Refer to caption](/html/2110.04361/assets/images/arc9.png)

Figure 1: SubTab framework: i) Dividing the features into subsets (similar to feature bagging, or cropping images), ii) Reconstruction of either subsets of features (x~1subscript~𝑥1\tilde{x}\_{1},x~2subscript~𝑥2\tilde{x}\_{2},x~3subscript~𝑥3\tilde{x}\_{3}), or complete feature space (X~1subscript~𝑋1\tilde{X}\_{1},X~2subscript~𝑋2\tilde{X}\_{2},X~3subscript~𝑋3\tilde{X}\_{3}), which are used to compute reconstruction loss. iii) Generating projections used to compute contrastive and distance loss. E≡E​n​c​o​d​e​r,D≡D​e​c​o​d​e​r,G≡P​r​o​j​e​c​t​i​o​nformulae-sequence𝐸𝐸𝑛𝑐𝑜𝑑𝑒𝑟formulae-sequence𝐷𝐷𝑒𝑐𝑜𝑑𝑒𝑟𝐺𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛E\equiv Encoder,D\equiv Decoder,G\equiv Projection.

The augmentation methods such as adding noise, rotation, cropping etc. are commonly used in image domain. Among them, the cropping is shown to be the most effective technique [[7](#bib.bib7)]. Inspired from this insight, we propose subsetting features of tabular data.

Figure [1](#S2.F1 "Figure 1 ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") presents SubTab framework, in which we have an encoder (E), a decoder (D), and an optional projection (G). For the purpose of this paper, we will refer 𝒉𝒉\bm{h} as latent, or representation, 𝒛𝒛\bm{z} as projection, 𝒙~bold-~𝒙\bm{\tilde{x}}, and 𝑿~bold-~𝑿\bm{\tilde{X}} as the reconstruction of subset, and whole data respectively. Small letters are associated with subsets while capital latters are associated the whole set of features. Moreover, throughout this work, when we say that a representation is "good", we refer to its performance in a classification task using a linear model.

In SubTab framework, we divide tabular data to multiple subsets. Neighbouring subsets can have overlapping regions, defined as a percentage of a dimension of the subset. Each of the subsets is fed to the same encoder (i.e. parameter sharing) to get their corresponding latent representation. A shared decoder is used to reconstruct either the subset fed to the encoder, or full tabular data (i.e. reconstructing all features from the subset of features). We chose the latter in our experiments since it is more effective in learning good representations. We should also note that, in the latter case, the autoencoder cannot learn the identity, eliminating the constraint on the dimension of the bottleneck (i.e. representation). We compute one reconstruction loss term per subset.

Moreover, we can optionally add contrastive loss to our objective by using all combination of pairs of projections from all subsets. For example, given three subsets as in Figure [1](#S2.F1 "Figure 1 ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"), there are three combinations of two: (nk)=(32)=3!2!​(1)!=3binomial𝑛𝑘binomial323213\binom{n}{k}=\binom{3}{2}=\frac{3!}{2!(1)!}=3 . For four subsets, it would be 6 pairs of combination, and so on. We can add one more loss term, referred as distance loss, to reduce the distance between the pairs of projections of the subsets by using a loss function such as mean squared error (MSE). All three loss terms apply a pulling force on positive samples while contrastive loss also applies a push force between positive and negative samples as shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Strategies for adding noise ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a.

Once the dataset is divided into subsets in data preparation step, a process that is similar to feature bagging in ensemble learning, their location is fixed. Thus, *we don’t change the relative order of features in a subset* during training since standard neural network architectures are not permutation invariant. This is to ensure that same features are fed to the same input units of neural network. However, our method can be extended to permutation invariant setting as a next step.

### 2.1 Strategies for adding noise

Our framework is complementary to other augmentation techniques used in tabular data setting. Thus, we experimented with adding noise to randomly selected entries in each subset by using three types of noise: i) adding Gaussian noise, 𝒩​(0,σ2)𝒩0superscript𝜎2\mathcal{N}(0,\sigma^{2}), ii) overwriting the value of a selected entry with another value randomly sampled from the same column, referred as swap-noise, iii) zeroing-out randomly selected entries, referred as zero-out noise.

Moreover, we use three different strategies when selecting the features to add noise to, as shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Strategies for adding noise ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")b: i) a random block of neighboring columns (NC), ii) random columns (RC) iii) random features per each sample (RF). To add noise, we create a binomial mask, 𝒎𝒎\bm{m}, and a noise matrix, ϵbold-italic-ϵ\bm{\epsilon}, with same shape as the subset, in which the entries of the mask is assigned to 1 with probability p𝑝p, and to 0 otherwise. The corrupted version, 𝒙𝟏​𝒄subscript𝒙1𝒄\bm{x\_{1c}}, of subset 𝒙𝟏subscript𝒙1\bm{x\_{1}} is generated as following:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒙𝟏​𝒄=(𝟏−𝒎)⊙𝒙𝟏+𝒎⊙ϵsubscript𝒙1𝒄direct-product1𝒎subscript𝒙1direct-product𝒎bold-italic-ϵ\bm{x\_{1c}=(1-m)\odot x\_{1}+m\odot\epsilon} |  | (1) |

![Refer to caption](/html/2110.04361/assets/images/forces.png)


(a) Push-Pull forces

![Refer to caption](/html/2110.04361/assets/images/selection_types_vertical.png)


(b) Feature selection

![Refer to caption](/html/2110.04361/assets/images/test_time7.png)


(c) Representation at test time

Figure 2: a) Push-Pull forces applied by each loss. PS / NS : Positive/Negative sample; CL/RL/DL: Contrastive, Reconstruction, Distance losses b) Column or feature selection strategies for adding noise to each subset. Top: Selecting a block of neighbouring columns; Middle: Selecting columns randomly; Bottom: Selecting random features per row c) Latent variables from each subset is aggregated at test time. The mean (default), sum, max, or min aggregation can be used.

### 2.2 Training

Our objective function is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒt=ℒr+ℒc+ℒd,subscriptℒ𝑡subscriptℒ𝑟subscriptℒ𝑐subscriptℒ𝑑\mathcal{L}\_{t}=\mathcal{L}\_{r}+\mathcal{L}\_{c}+\mathcal{L}\_{d}, |  | (2) |

where ℒtsubscriptℒ𝑡\mathcal{L}\_{t}, ℒrsubscriptℒ𝑟\mathcal{L}\_{r}, ℒcsubscriptℒ𝑐\mathcal{L}\_{c} and ℒdsubscriptℒ𝑑\mathcal{L}\_{d} are total, reconstruction, contrastive, and distance losses, respectively.

i) Reconstruction loss: Given a subset, denoted by 𝒙𝒌subscript𝒙𝒌\bm{x\_{k}}, we can reconstruct either the same subset, 𝒙~𝒌subscriptbold-~𝒙𝒌\bm{\tilde{x}\_{k}} or the entire feature space 𝑿~𝒌subscriptbold-~𝑿𝒌\bm{\tilde{X}\_{k}}.
Then, we can compute the reconstruction loss for kt​hsuperscript𝑘𝑡ℎk^{th} subset by computing mean squared error using either (𝒙𝒌,𝒙~𝒌)subscript𝒙𝒌subscriptbold-~𝒙𝒌(\bm{x\_{k}},\bm{\tilde{x}\_{k}}), or (𝑿,𝑿~𝒌)𝑿subscriptbold-~𝑿𝒌(\bm{X},\bm{\tilde{X}\_{k}}) pair as shown in Figure [1](#S2.F1 "Figure 1 ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). We chose the latter since it was more effective. Overall reconstruction loss:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒr=1K​∑k=1Ksk​,   where ​sk=1N​∑i=1N(𝑿(𝒊)−𝑿~𝒌(𝒊))2subscriptℒ𝑟1𝐾superscriptsubscript𝑘1𝐾subscript𝑠𝑘,   where subscript𝑠𝑘1𝑁superscriptsubscript𝑖1𝑁superscriptsuperscript𝑿𝒊superscriptsubscriptbold-~𝑿𝒌𝒊2\mathcal{L}\_{r}=\frac{1}{K}\sum\_{k=1}^{K}s\_{k}\mbox{, \, where }s\_{k}=\frac{1}{N}\sum\_{i=1}^{N}\left(\bm{X^{(i)}}-\bm{{\tilde{X}\_{k}}^{(i)}}\right)^{2} |  | (3) |

where K𝐾K is the total number of subsets, N𝑁N is the size of the batch, sksubscript𝑠𝑘s\_{k} is the reconstruction loss for kt​hsuperscript𝑘𝑡ℎk^{th} subset, and ℒrsubscriptℒ𝑟\mathcal{L}\_{r} is the average of reconstruction loss over all subsets.

ii) Contrastive loss:
If the dataset is rich in the number of classes such that chances of sampling negative samples are high, we can use a projection network (G) to get projections, 𝒛′​ssuperscript𝒛′𝑠\bm{z}^{\prime}s, of representations, 𝒉′​ssuperscript𝒉′𝑠\bm{h}^{\prime}s. Samples at the same rows of two subsets, 𝒛𝟏subscript𝒛1\bm{z\_{1}} and 𝒛𝟐subscript𝒛2\bm{z\_{2}}, can be considered as positive pairs while remaining rows in the subsets can be considered as negative to those samples.This allows us to compute the contrastive loss for each pair of projections using a loss function such as the normalized temperature-scaled cross entropy loss (NT-Xent) [[7](#bib.bib7)]. For three subsets, {𝒙𝟏,𝒙𝟐,𝒙𝟑}subscript𝒙1subscript𝒙2subscript𝒙3\{\bm{x\_{1}},\bm{x\_{2}},\bm{x\_{3}}\}, we can compute such a loss for every pair {𝒛𝒂,𝒛𝒃}subscript𝒛𝒂subscript𝒛𝒃\{\bm{z\_{a}},\bm{z\_{b}}\} of total three pairs from the set S={{𝒛𝟏,𝒛𝟐},{𝒛𝟏,𝒛𝟑},{𝒛𝟐,𝒛𝟑}}𝑆subscript𝒛1subscript𝒛2subscript𝒛1subscript𝒛3subscript𝒛2subscript𝒛3S=\{\{\bm{z\_{1},z\_{2}}\},\{\bm{z\_{1},z\_{3}}\},\{\bm{z\_{2},z\_{3}}\}\}. Overall contrastive loss is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒc=1J​∑{𝒛𝒂,𝒛𝒃}∈Sp​(𝒛𝒂,𝒛𝒃)​,   where ​p​(𝒛𝒂,𝒛𝒃)=12​N​∑i=1N[l​(𝒛𝒂(𝒊),𝒛𝒃(𝒊))+l​(𝒛𝒃(𝒊),𝒛𝒂(𝒊))]subscriptℒ𝑐1𝐽subscriptsubscript𝒛𝒂subscript𝒛𝒃𝑆𝑝subscript𝒛𝒂subscript𝒛𝒃,   where 𝑝subscript𝒛𝒂subscript𝒛𝒃12𝑁superscriptsubscript𝑖1𝑁delimited-[]𝑙superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊𝑙superscriptsubscript𝒛𝒃𝒊superscriptsubscript𝒛𝒂𝒊\displaystyle\mathcal{L}\_{c}=\frac{1}{J}\sum\_{\{\bm{z\_{a}},\bm{z\_{b}}\}\in S}p(\bm{z\_{a}},\bm{z\_{b}})\mbox{, \, where }p(\bm{z\_{a}},\bm{z\_{b}})=\frac{1}{2N}\sum\_{i=1}^{N}\left[l(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(i)}})+l(\bm{{z\_{b}}^{(i)}},\bm{{z\_{a}}^{(i)}})\right] |  | (4) |
|  |  |  |  |
| --- | --- | --- | --- |
|  | l​(𝒛𝒂(𝒊),𝒛𝒃(𝒊))=−log⁡exp⁡(s​i​m​(𝒛𝒂(𝒊),𝒛𝒃(𝒊))/τ)∑k=1N𝟙k≠i​exp⁡(s​i​m​(𝒛𝒂(𝒊),𝒛𝒃(𝒌))/τ)𝑙superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊𝑠𝑖𝑚superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊𝜏superscriptsubscript𝑘1𝑁subscript1𝑘𝑖𝑠𝑖𝑚superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒌𝜏\displaystyle l(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(i)}})=-\log\frac{\exp(sim(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(i)}})/\tau)}{\sum\_{k=1}^{N}\mathds{1}\_{k\neq i}\exp(sim(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(k)}})/\tau)}\qquad\qquad\qquad |  | (5) |

where J𝐽J is the total number of pairs in set S𝑆S, p​(𝒛𝒂,𝒛𝒃)𝑝subscript𝒛𝒂subscript𝒛𝒃p(\bm{z\_{a}},\bm{z\_{b}}) is total contrastive loss for a pair of projection {𝒛𝒂,𝒛𝒃}subscript𝒛𝒂subscript𝒛𝒃\{\bm{z\_{a}},\bm{z\_{b}}\}, l​(𝒛𝒂(𝒊),𝒛𝒃(𝒊))𝑙superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊l(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(i)}}) is the loss function for a corresponding positive pairs of examples (𝒛𝒂(𝒊),𝒛𝒃(𝒊))superscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊(\bm{{z\_{a}}^{(i)}},\bm{{z\_{b}}^{(i)}}) in subsets {𝒛𝒂,𝒛𝒃}subscript𝒛𝒂subscript𝒛𝒃\{\bm{z\_{a}},\bm{z\_{b}}\}, and ℒcsubscriptℒ𝑐\mathcal{L}\_{c} is the average of contrastive loss over all pairs.

iii) Distance loss:
We can also add mean-squared error (MSE) loss for pairs of projections of subsets since the corresponding samples in subsets should be close to each other. Hence, we can compute an overall MSE loss as:

|  |  |  |  |
| --- | --- | --- | --- |
|  | ℒd=1J​∑{𝒛𝒂,𝒛𝒃}∈Sp​(𝒛𝒂,𝒛𝒃)​,   where ​p​(𝒛𝒂,𝒛𝒃)=1N​∑i=1N(𝒛𝒂(𝒊)−𝒛𝒃(𝒊))2subscriptℒ𝑑1𝐽subscriptsubscript𝒛𝒂subscript𝒛𝒃𝑆𝑝subscript𝒛𝒂subscript𝒛𝒃,   where 𝑝subscript𝒛𝒂subscript𝒛𝒃1𝑁superscriptsubscript𝑖1𝑁superscriptsuperscriptsubscript𝒛𝒂𝒊superscriptsubscript𝒛𝒃𝒊2\mathcal{L}\_{d}=\frac{1}{J}\sum\_{\{\bm{z\_{a}},\bm{z\_{b}}\}\in S}p(\bm{z\_{a}},\bm{z\_{b}})\mbox{, \, where }p(\bm{z\_{a}},\bm{z\_{b}})=\frac{1}{N}\sum\_{i=1}^{N}\left(\bm{z\_{a}^{(i)}}-\bm{z\_{b}^{(i)}}\right)^{2} |  | (6) |

The pseudocode of algorithm can be found in Algorithm [1](#algorithm1 "In Appendix A Algorithm ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in the Appendix. We should note that both ℒc​ and ​ℒdsubscriptℒ𝑐 and subscriptℒ𝑑\mathcal{L}\_{c}\mbox{ and }\mathcal{L}\_{d} in equation ([2](#S2.E2 "In 2.2 Training ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")) are optional, and we used them only in some experiments.

### 2.3 Test time

At test time, we feed the subsets of test set to the encoder, and get the aggregate of the representations of all available subsets as shown in Figure [2(c)](#S2.F2.sf3 "In Figure 2 ‣ 2.1 Strategies for adding noise ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). Please note that we can use mean, sum, min, max, or any other aggregation method to get joint representation, which is analogous to pooling in Computer Vision, or the aggregation of neighbouring nodes in graph convolutional networks [[23](#bib.bib23)]. We used mean aggregation in all our experiments, but did compare different aggregation methods in Appendix [F.4](#A6.SS4 "F.4 Using different aggregation functions ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). Our experiments show that we can use the representations of only one, or few subsets and still achieve a good performance at test time. For example, we could use only 𝒉𝟏subscript𝒉1\bm{h\_{1}}, or aggregate of 𝒉𝟏subscript𝒉1\bm{h\_{1}} and 𝒉𝟐subscript𝒉2\bm{h\_{2}} rather than aggregating over all subsets (𝒉𝟏subscript𝒉1\bm{h\_{1}}, 𝒉𝟐subscript𝒉2\bm{h\_{2}}, 𝒉𝟑subscript𝒉3\bm{h\_{3}}) in Figure [2(c)](#S2.F2.sf3 "In Figure 2 ‣ 2.1 Strategies for adding noise ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). This allows the model to infer from the data even in the presence of missing features, in which case we can ignore the subset with missing features. We can also design an aggregation function that computes weighted mean of the representations of subsets since some subsets might be more informative than others:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒉=1Z​∑k=1Kηk∗𝒉𝒌​, and ​Z=∑k=1Kηk,𝒉1𝑍superscriptsubscript𝑘1𝐾subscript𝜂𝑘subscript𝒉𝒌, and 𝑍superscriptsubscript𝑘1𝐾subscript𝜂𝑘\bm{h}=\frac{1}{Z}\sum\_{k=1}^{K}{\eta}\_{k}\*\bm{h\_{k}}\mbox{, and }Z=\sum\_{k=1}^{K}{\eta}\_{k}, |  | (7) |

where K𝐾K is number of subsets, and ηksubscript𝜂𝑘\eta\_{k} is the weight for kt​hsubscript𝑘𝑡ℎk\_{th} subset. η𝜂\eta can be a learnable parameter in semi-supervised, or supervised setting by using an attention mechanism. We can also use 1D convolution in equation ([7](#S2.E7 "In 2.3 Test time ‣ 2 Method ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")) by treating representations of subsets as separate channels during training. We left these ideas as future work and used the mean aggregation (i.e. ηk=1subscript𝜂𝑘1\eta\_{k}=1) throughout our experiments, unless explicitly stated. A comparison of different aggregation methods can be found in Table [A3](#A6.T3 "Table A3 ‣ F.4 Using different aggregation functions ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in the Appendix.

## 3 Experiments

We conducted various experiments on diverse set of tabular datasets including MNIST [[27](#bib.bib27)] in tabular format, the cancer genome atlas (TCGA) [[42](#bib.bib42)], human gut metagen-omic samples of obesity cohorts (Obesity) [[36](#bib.bib36), [26](#bib.bib26)], UCI adult income (Income) [[24](#bib.bib24)], and UCI BlogFeedback (Blog) [[4](#bib.bib4)] to demonstrate the effectiveness of the SubTab framework. We compare our method to autoencoder baseline with and without dropout, other self-supervised methods such as VIME-self [[46](#bib.bib46)], Denoising Autoencoder (DAE) [[43](#bib.bib43)], and Context Encoder (CAE) [[39](#bib.bib39)] as well as fully-supervised models such as logistic regression, random forest, and XGBoost [[6](#bib.bib6)]. For each dataset, once we decided on a particular autoencoder architecture, we used it for all models compared (i.e. VIME-self, DAE, CAE, and our model). We tried both ReLU and leakyReLU as activation functions for all, and both performed equally well. The code for SubTab is provided111https://github.com/AstraZeneca/SubTab. The summary of model architectures and hyper-parameters are in Table [A1](#A3.T1 "Table A1 ‣ C.1 Model architectures and hyper-parameters ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in the Appendix. We should note that we ran more experiments using; i) Synthetic datasets and ii) OpenML-CC18 datasets [[2](#bib.bib2)] in Appendix [G](#A7 "Appendix G Experiments using synthetic data ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") and [H](#A8 "Appendix H Experiments using OpenML-CC18 datasets ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") respectively.

### 3.1 Data

MNIST:
We flattened 28x28 images, and scaled them by dividing all with 255 as it is done in [[46](#bib.bib46)]. We split training set into training and validation sets (90-10% split) when searching for hyper-parameters, and then used all of training set to train the final model. The test set is used only for final evaluation.

The Cancer Genome Atlas (TCGA):
TCGA is a public cancer genomics dataset characterized over 20,000 primary cancer and matched normal samples that holds information over 38 cohorts. The task is to classify the cancer cohorts from the reverse phase protein array (RPPA) dataset. It includes 6671 samples with 122 features, which we divided to 80-10-10% train-validation-test sets. Once hyper-parameters is found, we trained the models on combined training and validation set.

Obesity:
The dataset consists of publicly available human gut metagen-omic samples of obesity cohorts [[36](#bib.bib36)]. It is derived from whole-genome shotgun metagenomic studies. The dataset consists of 164 obese patients and 89 non-obese controls and has 425 features [[26](#bib.bib26)]. We scaled the dataset by using min-max scaling. Since it is a small dataset, we evaluated the model by using 10 randomly drawn training-test (90-10%) splits, for each of which we used 10-fold cross-validation.

UCI Adult Income:
It is a well-known public dataset extracted from the 1994 Census database [[24](#bib.bib24)]. It includes the details such as education level and demographics to predict whether the income of a person exceeds $50K/yr. The data consists of six continuous and eight categorical features. After one-hot encoding of categorical features, there are total of 101 features. The pre-processing steps can be found in Section [B.1](#A2.SS1 "B.1 Adult Income Dataset ‣ Appendix B Data ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") of Appendix.

UCI BlogFeedback:
The data originates from blog posts, and is originally used for regression task of predicting the number of comments in the upcoming 24 hours. Similar to Yoon et al. [[46](#bib.bib46)], we turned it into a binary classification task of predicting whether there is a comment for a post or not.There are 280 integer and real valued features, and separate training and test datasets are provided. Further information can be found in Section [B.2](#A2.SS2 "B.2 BlogFeedback Dataset ‣ Appendix B Data ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") of Appendix.

### 3.2 Evaluation

For self-supervised models, once the models are trained, we evaluate them by training a logistic regression model on the latent representations of training set, and testing it on the latent representation of the test set. For SubTab, the joint latent representation is obtained by using the mean aggregation of embeddings of the subsets for both training and test sets. We use the performance on a classification task as a measure of quality of the representation as it is usually done in the self-supervised learning. MNIST has 10, TCGA has 38, and the rest (i.e. Obesity, Income, and Blog) has 2 classes each.

### 3.3 Results

![Refer to caption](/html/2110.04361/assets/images/mnist_text_acc.png)


(a)

![Refer to caption](/html/2110.04361/assets/images/mnist_train.png)


(b)

![Refer to caption](/html/2110.04361/assets/images/mnist_test.png)


(c)

Figure 3: a) Test accuracy on MNIST dataset over different number of subsets and varying levels of overlaps. b-c) t-SNE plots for training (b) and test (c) sets of MNIST for the case of using 4 subsets with 75% overlap between neighboring subsets.

MNIST:
We used a simple three-layer encoder architecture with dimensions of [512, 256, 128], referred as the base model, in which the last layer is a linear layer. During training of the base model, we used both reconstruction and contrastive losses. Additionally, we trained our model under three conditions: i) without any noise in the input data, ii) with noise in the input data and iii) same as (ii), but we also added distance loss computed for pairs of projections {𝒛𝒊,𝒛𝒋,…}subscript𝒛𝒊subscript𝒛𝒋…\{\bm{z\_{i}},\bm{z\_{j}},...\}.

For SubTab, we trained our base model multiple times without noise at the input. For each training, we used different number of subsets with different levels of overlap between neighbouring subsets (Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a). For small number of subsets (e.g. 2 or 3), the performance monotonically decreases when we increase the overlap between subsets. But, for higher number of subsets, the performance generally improves as we increase the number of shared features between the neighbouring subsets. In general, our results show that K=4𝐾4K=4 with 75% overlap, and K=7𝐾7K=7 with 50% overlap perform the best in MNIST dataset, where K𝐾K refers to the number of subsets. Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") also shows t-SNE plots of training and test sets for K= 4 with 75% overlap, which proves the high quality of clustering, while Table [1](#S3.T1 "Table 1 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") summarizes the classification accuracy of all models on the test set. Our base model without noise outperforms autoencoder baselines and other self-supervised models with the same architecture. We experimented with three noise types for all self-supervised models, and observed that adding swap-noise at the input pushes the performance higher. For SubTab, adding distance loss and increasing the dimensions of the last layer from 128 to 512 help improve the performance even further. Moreover, we conducted three additional experiments (details in Section [C.3](#A3.SS3 "C.3 Supporting visuals for experiments ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") of Appendix):

![Refer to caption](/html/2110.04361/assets/images/performance_num_subsets7.png)


(a)

![Refer to caption](/html/2110.04361/assets/images/sota3.png)


(b)

Figure 4: a) After training the base model (latent dimension=128) on four subsets with 75% overlap, we test its performance using different number of subsets. The performance improves as we start increasing number of subsets involved in prediction. b) Comparing our model to CNN-based SOTA models trained on 28x28 MNIST in image format (please see Section [3.4](#S3.SS4 "3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") for details).

In the first experiment, for the optimum case of K=4𝐾4K=4 with 75% overlap, we trained and tested accuracy of a linear model by using the joint representations obtained from the varying number of subsets. Starting with a single subset of the data, we plot the training and test accuracy of the model (Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a). The linear model is able to achieve 87.5% test accuracy using the representation of a single subset. As we start adding latent representations of remaining subsets, both the training and test accuracy keep increasing, eventually achieving top accuracy when all subsets are used. The evolution of clusters corresponding to Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a can be seen in Figure [A7](#A6.F7 "Figure A7 ‣ F.1 MNIST ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in Appendix. This experiment indicates that we can achieve a good performance using only small subset of features when we don’t have access to data on other features.

![Refer to caption](/html/2110.04361/assets/images/acc_over_subsets1v2.png)


(a)

![Refer to caption](/html/2110.04361/assets/images/acc_over_subsets2.png)


(b)

![Refer to caption](/html/2110.04361/assets/images/acc_over_subsets3v2.png)


(c)

Figure 5: a) The test accuracy using the mean aggregation of the latent representations of subsets, starting with the first subset, and keep adding new subsets sequentially. b) The test accuracy of individual subsets. c) Comparing the test accuracy by aggregating the representations of different set of subsets at test time for two versions of the model; untrained and the one trained on all subsets.

In the second experiment, we evaluated SubTab under the condition of missing features during training (Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a). To do so, we first sliced *the unshuffled* features of MNIST to seven subsets with no overlap (the case corresponding to the legend "7" at zero overlap in Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a). Each subset corresponds to four rows in a 28x28 image, starting from top four rows (subset 1) to the bottom ones (subset 7). Then, we trained the base model on five different sets of subsets; {4},{4,5},{3,4,5},{2,3,4,5,6},

44534523456\{4\},\{4,5\},\{3,4,5\},\{2,3,4,5,6\}, and {1,2,3,4,5,6,7}1234567\{1,2,3,4,5,6,7\}, resulting in five different trained SubTab models. Please note that we selected the sets such that we expand out from the most informative middle regions of the image (i.e. subset 4) to the least informative top and bottom areas.

In order to compare the performance of five models, we followed the following steps for each trained model: 1) We first obtained the embeddings of all seven subsets for both training and test sets; 2) We then trained and evaluated a logistic regression model by using the joint embedding of each of the following seven sets: {1},{1,2},{1,2,3},…,{1,2,3,4,5,6,7}

112123…1234567\{1\},\{1,2\},\{1,2,3\},...,\{1,2,3,4,5,6,7\} i.e. starting from the first subset, we kept adding new subsets sequentially to increase the information content in the sets. For example, for the set {1,2,3}123\{1,2,3\}, we first trained a logistic regression model by using joint embedding of subset 1, 2 and 3 from training set, and evaluated it by using the joint embedding of same subsets from test set. The joint embedding of a set is obtained by using mean aggregation of embeddings of subsets in the set. In addition to five models, we initialized a sixth SubTab model, but kept it untrained and followed the same procedure described before to use it as a baseline. The results are shown in Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a.

In this experiment, we observe that even when the model is trained on a single subset (subset 4, or the blue line in Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a), aggregating the representations of all seven subsets including the subsets not used in training does improve the results. This is because the encoder is able to map samples of different classes to different points in latent space even if it is not trained on them. Since we use the mean aggregation over different views (i.e. subsets) of the same class, we can still make each class in the data distinguishable from the rest in the latent space. We also note that when the model is trained on more and more subsets, its performance keeps improving. As a baseline, we also conducted the same test using untrained model (red line in the plot), and observed similar behaviour in which the test accuracy generally improves as we use more subsets when constructing the joint latent representation. Moreover, we measured the test accuracy of individual subsets to see how informative each subset is (Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")b). The result is as expected since we kept the features unshuffled in this experiment, and know that the subsets corresponding to the mid-region of the images (i.e. subsets 3, 4, and 5) should be more informative than the ones corresponding to the top and bottom regions (i.e. subsets 1, and 7). We repeated the same experiment using 28 subsets to get a higher resolution and added the result in Figure [A8](#A6.F8 "Figure A8 ‣ F.1 MNIST ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in Appendix. From this experiment; i) we see that joint representation improves as we include more subsets (i.e. sub-views) at training and/or test time, ii) we can identify the informative subsets of features using SubTab framework.

In the third experiment, we evaluated SubTab on handling missing features at test time. Specifically, we used the model trained on all subsets, and compared it to the untrained model (i.e. our baseline). For each model, we obtained the joint embedding for training set by using mean aggregation over embeddings of all seven subsets, and then trained a linear model. The test accuracy of the linear model is measured by using; i) only subset 4, ii) aggregate of the most informative subsets {3,4,5}, iii) aggregate of {2,3,4,5,6} excluding the least informative subsets, and iv) all seven subsets of the test set (Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")c).

The results indicate that SubTab can accommodate missing features at test time, and can still perform well. This might also indicate that working with subsets can give us a way to deal with uncertainty better when there are missing features at test time. As the model collects more information in the form of more features, its prediction improves (see Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")c). We can also train the model when there are missing subsets during training, and it still performs well (e.g. see legend "4", corresponding to the model trained only on subset 4, in Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a). Our experiments simulate a practical scenario. For example, in healthcare, we might not have access to some features in one hospital while we might have them in another. So, our method would be beneficial in this type of cases.

Overall, we can make the following observations from our experiments: i) the less informative subsets can add value to the overall representation, or at least does not harm the performance (see the aggregate over {3,4,5} versus "All" in Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")c), ii) untrained model can be used to analyze which subsets can be potentially more informative, iii) once a model is trained on a subset, the performance of the individual subset does not change whether it is trained together with other subsets or not (for example, compare the performance of subset 3, 4, and 5 across all models in Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")b), iv) general idea behind our framework works even for untrained model, and v) we may not need to impute data in our framework since we can simply ignore them as missing subsets, which is good since imputation generally distorts data, and the results.

Table 1: Accuracy scores for all models for various datasets. The abbreviations in the table; NC: Neighbour columns used, RF: Random features used, G: Gaussian noise used, S: Swap noise used.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Type | Models | MNIST | Income | Blog | Obesity | TCGA |
| Supervised | Logistic Regression | 92.60±plus-or-minus\pm0.03 | 84.68±plus-or-minus\pm0.05 | 84.15±plus-or-minus\pm0.12 | 62.35±plus-or-minus\pm4.02 | 36.98±plus-or-minus\pm 1.25 |
| baseline | Random Forest | 96.96±plus-or-minus\pm0.06 | 84.62±plus-or-minus\pm0.07 | 83.61±plus-or-minus\pm0.15 | 67.45±plus-or-minus\pm2.23 | 61.62±plus-or-minus\pm 1.02 |
|  | XGBoost | 98.02±plus-or-minus\pm0.086 | 86.11±plus-or-minus\pm0.20 | 84.29±plus-or-minus\pm0.23 | 64.05±plus-or-minus\pm4.52 | 72.61±plus-or-minus\pm1.31 |
| Autoencoder | AE | 92.77±plus-or-minus\pm0.32 | 84.67±plus-or-minus\pm0.07 | 84.06±plus-or-minus\pm0.24 | 61.96±plus-or-minus\pm3.28 | 55.16±plus-or-minus\pm0.75 |
| baseline | AE w/ Dropout (p=0.2) | 94.31±plus-or-minus\pm0.28 | 85.00±plus-or-minus\pm0.10 | 84.18±plus-or-minus\pm0.20 | 62.74±plus-or-minus\pm4.38 | 56.87±plus-or-minus\pm2.26 |
|  | DAE (RF) | 96.30±plus-or-minus\pm0.14 (S) | 84.37±plus-or-minus\pm0.36 (G) | 84.12±plus-or-minus\pm0.29 (G) | 56.43±plus-or-minus\pm5.79 (G) | 54.31±plus-or-minus\pm1.39 (G) |
|  | CAE (NC) | 96.39±plus-or-minus\pm0.20 (S) | 84.24±plus-or-minus\pm0.18 (G) | 84.3±plus-or-minus\pm0.31 (G) | 62.26±plus-or-minus\pm5.01 (G) | 54.20±plus-or-minus\pm1.17 (G) |
|  | VIME-self | 95.23±plus-or-minus\pm0.17 (S) | 84.43±plus-or-minus\pm0.08 (G) | 84.11±plus-or-minus\pm0.27 (G) | 66.45±plus-or-minus\pm4.54 (G) | 55.11±plus-or-minus\pm1.37 (G) |
| Self- | SubTab with: |  |  |  |  |  |
| supervised | Base model (No noise) | 97.26±plus-or-minus\pm0.2 | 85.31±plus-or-minus\pm0.08 | 84.29±plus-or-minus\pm0.26 | 68.01±plus-or-minus\pm3.07 | 57.02±plus-or-minus\pm1.50 |
|  | +Noise | 97.47±plus-or-minus\pm0.18 (S) | 85.34±plus-or-minus\pm0.07 (G) | 84.47±plus-or-minus\pm0.15 (G) | 71.13±plus-or-minus\pm4.08 (G) | 58.25±plus-or-minus\pm1.36 (G) |
|  | +Distance loss | 97.52±plus-or-minus\pm0.14 (S) | 85.35±plus-or-minus\pm0.06 (G) | 84.64±plus-or-minus\pm0.19 (G) | 69.25±plus-or-minus\pm4.19 (G) | 58.15±plus-or-minus\pm1.56 (G) |
|  | +LatentDim=512 | 97.86±plus-or-minus\pm0.07 (S) | - | - | - | - |

TCGA:
We used an encoder architecture with three layers [1024, 784, 784], where the third layer is linear. For VIME-self, DAE, CAE, and our model, we experimented with three noise types (Gaussian, swap, and zero-out noise) at the different % levels of masking ratio p𝑝p. We observed that p=[0.15,0.3]𝑝0.150.3p=[0.15,0.3] range worked well for all models. For Gaussian noise, we used a distribution with zero mean, and different levels of standard deviation (σ𝜎\sigma). Among all three noise types, Gaussian noise with σ=0.1𝜎0.1\sigma=0.1 worked the best for all models. Please note that VIME-self uses swap-noise in its original implementation, but swap-noise does not work well on this dataset. For SubTab, similar to MNIST, we used four subsets with 75% overlap. SubTab performs better than other self-supervised models with a significant margin and almost doubles the performance of logistic regression model trained on raw data as shown in Table [1](#S3.T1 "Table 1 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning").

Obesity:
We used a two-layer encoder with [1024, 1024] dimensions. Second layer is a linear layer. Gaussian noise 𝒩​(0,0.3)𝒩00.3\mathcal{N}(0,0.3) and masking ratio p=0.2𝑝0.2p=0.2 works well across all models. Six subsets (K=6𝐾6K=6) with 0% overlap performed the best for the SubTab. We note that this dataset has 164 obese patients out of 253 total patients. So, the baseline accuracy is 164/253=64.82%164253percent64.82164/253=64.82\%. Based on this fact, we can say that all models, except ours, did not perform well on this dataset. Our model with added Gaussian noise results in accuracy of 71.13±4.08%plus-or-minus71.13percent4.0871.13\pm 4.08\%, which is well above all models, including supervised ones. It means that our model was able to learn useful representation from the data. We should also note that the performance of our model is much better than what Oh and Zhang [[36](#bib.bib36)] reported (66±3.2%plus-or-minus66percent3.266\pm 3.2\%) even though they trained a DAE on the same data, and reported their results using a random forest, a non-linear model, on the learned representations rather than a linear model.

UCI Adult Income & BlogFeedback:
For these two datasets, we used the same architecture as in Obesity. For Income dataset, the best performance is obtained using 5 subsets with 25% overlap whereas we used 7 subsets with 75% overlap for Blog dataset. For the base model, we only used reconstruction loss. Adding Gaussian noise to the input and distance loss to the objective improves the performance for both datasets. SubTab outperforms other self-supervised models in both datasets.

The choice of hyper-parameters and other details for all experiments can be found in Table [A1](#A3.T1 "Table A1 ‣ C.1 Model architectures and hyper-parameters ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") in Section [C.1](#A3.SS1 "C.1 Model architectures and hyper-parameters ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") of Appendix.

### 3.4 Ablation study

Table 2: Ablation study using MNIST with 4 subsets with 75% overlap. Abbreviations are; RL: Reconstruction Loss, CL: Contrastive Loss, DL: Distance Loss, SF: Shuffled Features, LD: Latent Dim, Agg: Aggregating embeddings.

| RL | CL | Noise | DL | SF | LD | Agg | Test Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| + | - | - | - | - | 128 | + | 97.13 |
| - | + | - | - | - | 128 | + | 97.11 |
| + | + | - | - | - | 128 | + | 97.26 |
| + | + | Zero-out | - | - | 128 | + | 97.25 |
| + | + | Gaussian | - | - | 128 | + | 97.25 |
| + | + | Swap | - | - | 128 | + | 97.47 |
| + | + | Swap | + | - | 128 | + | 97.52 |
| + | + | Swap | + | + | 128 | + | 97.2 |
| + | + | Swap | + | - | 512 | - | 95.92 |
| + | + | Swap | + | - | 512 | + | 97.86 |

We conducted a comprehensive ablation study using MNIST. Table [2](#S3.T2 "Table 2 ‣ 3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") summarizes our experiments. The first thing to note is that the performance of the our base model is already good with only reconstruction loss. Hence, we can argue that the reconstruction of original feature space from a subset of features is a very effective way of learning representation. By adding noise to the input data, we can improve the performance. In the case of MNIST, swap-noise is very effective. Also, by adding additional losses such as contrastive, and distance losses as well as increasing the dimension of representation layer from 128 to 512, we can further improve the results. Moreover, we shuffled the features of MNIST to make sure that we don’t have any gains from unintentional spatial correlations between neighboring features. We kept all parameters and random seeds same for the comparison. As shown in the table, our model’s performance does not change much. We also tried concatenating latent variables of subsets rather than aggregating them when testing the performance. Comparing last two rows in the table, the aggregation is shown to work much better. Please note that we compared different aggregation functions in Appendix [F.4](#A6.SS4 "F.4 Using different aggregation functions ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"), showing that mean aggregation worked the best.

Finally, we compared the performance of SubTab on shallow and deep architecture choices.
We trained and tested very shallow architectures for SubTab (referred as shallow SubTab), and compared them to relatively deeper SubTab models used in Table [1](#S3.T1 "Table 1 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") (referred as deep SubTab). We used one-layer encoder and decoder with 784 dimension each for MNIST while using 1024 dimension for other datasets. Shallow SubTab is trained and evaluated under the same conditions as the deeper ones. As shown in Table [3](#S3.T3 "Table 3 ‣ 3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"), shallow SubTab significantly improves results in MNIST and TCGA, placing our model performance on par with CNN-based SOTA models [[20](#bib.bib20), [19](#bib.bib19), [25](#bib.bib25), [22](#bib.bib22), [32](#bib.bib32)] as shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")b. Obesity is the only dataset which exploits the deeper architecture.

Table 3: Comparing shallow and deep SubTab architectures.

| Model | MNIST | Income | Blog | Obesity | TCGA |
| --- | --- | --- | --- | --- | --- |
| Deep SubTab | 97.86±plus-or-minus\pm0.07 | 85.35±plus-or-minus\pm0.06 | 84.64±plus-or-minus\pm0.19 | 71.13±plus-or-minus\pm4.08 | 58.25±plus-or-minus\pm 1.36 |
| Shallow SubTab | 98.31±plus-or-minus\pm0.06 | 85.34±plus-or-minus\pm0.03 | 84.64±plus-or-minus\pm0.09 | 66.88±plus-or-minus\pm5.35 | 61.41±plus-or-minus\pm1.11 |

## 4 Related works

We refer the reader to the introduction section that lists some of the recent noticeable works in self-supervised learning. Since our work focuses on tabular data, we will review some of the recent work done in tabular data in self-supervised framework. The most recent work is mostly based on solving a pretext task. For example, Yoon et al. [[46](#bib.bib46)] uses a de-noising autoencoder with a classifier attached to its representation layer. A random binary mask is generated to mask and overwrite a portion of entries in the tabular data, and the corrupted data is given as input to the encoder. The classifier is used to predict the mask while decoder is used to re-construct the uncorrupted original input similar to de-noising autoencoder [[43](#bib.bib43)]. Although the proposed method is shown to work well in the experiments, there are couple drawbacks to this approach. Firstly, this approach might not work well in very high-dimensional, small and noisy data sets since the model might easily become over-parameterized and be prone to overfitting to the data. Secondly, training a classifier in this setting can be challenging since it needs to predict very high dimensional, sparse, and imbalanced binary mask, similar to the problems observed when training a model on imbalanced, binary dataset. In a similar way, TabNet [[1](#bib.bib1)] and TaBERT [[45](#bib.bib45)] also tries to recover original data from corrupted one.

## 5 Conclusion

In this work, we show that a simple MLP-based autoencoder trained on MNIST in tabular format can perform on par with the CNN-based SOTA models trained on MNIST images in unsupervised/self-supervised framework. SubTab achieves SOTA in MNIST dataset in tabular setting. We also tested our approach on other commonly used tabular datasets, and proved its benefits. In SubTab, the main performance gain comes from two parts of the model: i) reconstruction of all features from the subset of features, and ii) learning the joint representation by aggregating the embeddings of the subsets.

Using subsets of features may obviate the need for data imputation during training, and allows inference using subsets of features at test time. It might open the door to distributed training of high-dimensional data since the models can be trained on different subsets of features at the same time. We can also potentially take advantage of different datasets with common features by assigning those features to same subsets (i.e. transfer learning). We should note that the subsets shared the same autoencoder in our experiments although we could use separate autoencoders for different subsets if some of the features are drastically different than the rest.

SubTab is computationally scalable when we use only reconstruction loss during training. However, using contrastive, and/or distance losses requires the combinations of projections, which makes the computational complexity quadratic during training and limits the number of subsets we can use to divide the data. In this case, computational complexity is still linear at test time since we need to compute only the aggregate of the representations of the subsets. Also, when we divide the features into subsets, we keep the location of features in each subset same throughout training and test time since neural networks are not permutation invariant. As a possible solution, we can extend our work to permutation invariant architectures by treating collection of features as a set. We also showed that SubTab framework can be used to discover most informative subsets of features with limited resolution. A hierarchical version of SubTab might be used for identifying individual important features, but we leave it as a future work.

Finally, although the primary focus of this work is tabular data setting, SubTab can be extended to other domains such as images, audio, text and so on. We leave the extensions and applications of SubTab as a future work.

## 6 Broader Impact

Tabular data is a commonly used format in healthcare, finance, law and many other fields. Despite its broad usage, the most of the research in deep learning, especially with regards to unsupervised representation learning, has been on other data types such as images, text and audio. Our paper tries to close this gap by introducing a new framework to learn good representations from tabular data in unsupervised/self-supervised setting. The progress in this line of research will open doors to widespread applications of tabular data in other areas such as transfer learning, distributed learning, and multi-view learning, in which we can combine knowledge such as demographics and genomics from tabular data with those in images, text and audio. However, we should be aware of the shortcomings of such data integration in terms of biases and privacy issues that it might introduce.

## 7 Acknowledgements

We thank the anonymous reviewers for their helpful and constructive feedback on the paper. We would also like to thank the entire Respiratory and Immunology AI team and are grateful for general support from other organizations within AstraZeneca.

## References

* Arik and Pfister [2019]

  Sercan O Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  *arXiv preprint arXiv:1908.07442*, 2019.
* Bischl et al. [2017]

  Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Frank Hutter, Michel Lang,
  Rafael G Mantovani, Jan N van Rijn, and Joaquin Vanschoren.
  Openml benchmarking suites.
  *arXiv preprint arXiv:1708.03731*, 2017.
* Bridge et al. [2014]

  James P Bridge, Sean B Holden, and Lawrence C Paulson.
  Machine learning for first-order theorem proving.
  *Journal of automated reasoning*, 53(2):141–172, 2014.
* Buza [2014]

  Krisztian Buza.
  Feedback prediction for blogs.
  In *Data analysis, machine learning and knowledge discovery*,
  pages 145–152. Springer, 2014.
* Caron et al. [2020]

  Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and
  Armand Joulin.
  Unsupervised learning of visual features by contrasting cluster
  assignments.
  *arXiv preprint arXiv:2006.09882*, 2020.
* Chen and Guestrin [2016]

  Tianqi Chen and Carlos Guestrin.
  XGBoost: A scalable tree boosting system.
  In *Proceedings of the 22nd ACM SIGKDD International Conference
  on Knowledge Discovery and Data Mining*, KDD ’16, pages 785–794, New York,
  NY, USA, 2016. ACM.
  ISBN 978-1-4503-4232-2.
  doi: 10.1145/2939672.2939785.
  URL <http://doi.acm.org/10.1145/2939672.2939785>.
* Chen et al. [2020a]

  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
  A simple framework for contrastive learning of visual
  representations.
  In *International conference on machine learning*, pages
  1597–1607. PMLR, 2020a.
* Chen et al. [2020b]

  Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He.
  Improved baselines with momentum contrastive learning.
  *arXiv preprint arXiv:2003.04297*, 2020b.
* Collobert and Weston [2008]

  Ronan Collobert and Jason Weston.
  A unified architecture for natural language processing: Deep neural
  networks with multitask learning.
  In *Proceedings of the 25th international conference on Machine
  learning*, pages 160–167, 2008.
* Conneau et al. [2019]

  Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
  Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and
  Veselin Stoyanov.
  Unsupervised cross-lingual representation learning at scale.
  *arXiv preprint arXiv:1911.02116*, 2019.
* Devlin et al. [2018]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  *arXiv preprint arXiv:1810.04805*, 2018.
* Dua and Graff [2017]

  Dheeru Dua and Casey Graff.
  UCI machine learning repository, 2017.
  URL <http://archive.ics.uci.edu/ml>.
* Falcon and Cho [2020]

  William Falcon and Kyunghyun Cho.
  A framework for contrastive self-supervised learning and designing a
  new approach.
  *arXiv preprint arXiv:2009.00104*, 2020.
* Freire et al. [2009]

  Ananda L Freire, Guilherme A Barreto, Marcus Veloso, and Antonio T Varela.
  Short-term memory mechanisms in neural network learning of robot
  navigation tasks: A case study.
  In *2009 6th Latin American Robotics Symposium (LARS 2009)*,
  pages 1–6. IEEE, 2009.
* Grill et al. [2020]

  Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec,
  Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires,
  Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al.
  Bootstrap your own latent: A new approach to self-supervised
  learning.
  *arXiv preprint arXiv:2006.07733*, 2020.
* Harries and Wales [1999]

  Michael Harries and New South Wales.
  Splice-2 comparative evaluation: Electricity pricing.
  1999.
* He et al. [2020]

  Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.
  Momentum contrast for unsupervised visual representation learning.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pages 9729–9738, 2020.
* Hersey [1968]

  Irwin Hersey.
  Textures: A photographic album for artists and designers by phil
  brodatz.
  *Leonardo*, 1(1):91–92, 1968.
* Hu et al. [2017]

  Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto, and Masashi Sugiyama.
  Learning discrete representations via information maximizing
  self-augmented training.
  In *International Conference on Machine Learning*, pages
  1558–1567. PMLR, 2017.
* Ji et al. [2019]

  Xu Ji, João F Henriques, and Andrea Vedaldi.
  Invariant information clustering for unsupervised image
  classification and segmentation.
  In *Proceedings of the IEEE/CVF International Conference on
  Computer Vision*, pages 9865–9874, 2019.
* Joulin et al. [2016]

  Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov.
  Bag of tricks for efficient text classification.
  *arXiv preprint arXiv:1607.01759*, 2016.
* Khacef et al. [2020]

  Lyes Khacef, Laurent Rodriguez, and Benoit Miramond.
  Improving self-organizing maps with unsupervised feature extraction.
  In *International Conference on Neural Information Processing*,
  pages 474–486. Springer, 2020.
* Kipf and Welling [2016]

  Thomas N Kipf and Max Welling.
  Semi-supervised classification with graph convolutional networks.
  *arXiv preprint arXiv:1609.02907*, 2016.
* Kohavi [1996]

  Ron Kohavi.
  Scaling up the accuracy of naive-bayes classifiers: A decision-tree
  hybrid.
  In *Kdd*, volume 96, pages 202–207, 1996.
* Kosiorek et al. [2019]

  Adam R Kosiorek, Sara Sabour, Yee Whye Teh, and Geoffrey E Hinton.
  Stacked capsule autoencoders.
  *arXiv preprint arXiv:1906.06818*, 2019.
* Le Chatelier et al. [2013]

  Emmanuelle Le Chatelier, Trine Nielsen, Junjie Qin, Edi Prifti, Falk
  Hildebrand, Gwen Falony, Mathieu Almeida, Manimozhiyan Arumugam, Jean-Michel
  Batto, Sean Kennedy, et al.
  Richness of human gut microbiome correlates with metabolic markers.
  *Nature*, 500(7464):541–546, 2013.
* LeCun [1998]

  Yann LeCun.
  The mnist database of handwritten digits.
  *http://yann. lecun. com/exdb/mnist/*, 1998.
* Liu et al. [2019]

  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
  Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
  Roberta: A robustly optimized bert pretraining approach.
  *arXiv preprint arXiv:1907.11692*, 2019.
* Loshchilov and Hutter [2017]

  Ilya Loshchilov and Frank Hutter.
  Decoupled weight decay regularization.
  *arXiv preprint arXiv:1711.05101*, 2017.
* Lucas et al. [2013]

  DD Lucas, R Klein, J Tannahill, D Ivanova, S Brandon, D Domyancic, and Y Zhang.
  Failure analysis of parameter-induced simulation crashes in climate
  models.
  *Geoscientific Model Development*, 6(4):1157–1171, 2013.
* Madeo et al. [2013]

  Renata CB Madeo, Clodoaldo AM Lima, and Sarajane M Peres.
  Gesture unit segmentation using support vector machines: segmenting
  gestures from rest positions.
  In *Proceedings of the 28th Annual ACM Symposium on Applied
  Computing*, pages 46–52, 2013.
* Makhzani et al. [2015]

  Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan
  Frey.
  Adversarial autoencoders.
  *arXiv preprint arXiv:1511.05644*, 2015.
* Merz and Murphy [1996]

  C Merz and PM Murphy.
  Pima indians diabetes dataset.
  *UCI Repository*, 1996.
* Mikolov et al. [2013]

  Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
  Efficient estimation of word representations in vector space.
  *arXiv preprint arXiv:1301.3781*, 2013.
* Noordewier et al. [1991]

  Michiel O Noordewier, Geoffrey G Towell, and Jude W Shavlik.
  Training knowledge-based neural networks to recognize genes in dna
  sequences.
  In *Advances in neural information processing systems*, pages
  530–536, 1991.
* Oh and Zhang [2020]

  Min Oh and Liqing Zhang.
  Deepmicro: deep representation learning for disease prediction based
  on microbiome data.
  *Scientific reports*, 10(1):1–9, 2020.
* Oord et al. [2018]

  Aaron van den Oord, Yazhe Li, and Oriol Vinyals.
  Representation learning with contrastive predictive coding.
  *arXiv preprint arXiv:1807.03748*, 2018.
* Paszke et al. [2019]

  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
  Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
  Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan
  Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
  Chintala.
  Pytorch: An imperative style, high-performance deep learning library.
  In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural
  Information Processing Systems 32*, pages 8024–8035. Curran Associates,
  Inc., 2019.
  URL
  <http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf>.
* Pathak et al. [2016]

  Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A
  Efros.
  Context encoders: Feature learning by inpainting.
  In *Proceedings of the IEEE conference on computer vision and
  pattern recognition*, pages 2536–2544, 2016.
* Pedregosa et al. [2011]

  F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
  M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
  D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay.
  Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12:2825–2830,
  2011.
* Pennington et al. [2014]

  Jeffrey Pennington, Richard Socher, and Christopher D Manning.
  Glove: Global vectors for word representation.
  In *Proceedings of the 2014 conference on empirical methods in
  natural language processing (EMNLP)*, pages 1532–1543, 2014.
* Tomczak et al. [2015]

  Katarzyna Tomczak, Patrycja Czerwińska, and Maciej Wiznerowicz.
  The cancer genome atlas (tcga): an immeasurable source of knowledge.
  *Contemporary oncology*, 19(1A):A68, 2015.
* Vincent et al. [2008]

  Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol.
  Extracting and composing robust features with denoising autoencoders.
  In *Proceedings of the 25th international conference on Machine
  learning*, pages 1096–1103, 2008.
* Yeh et al. [2009]

  I-Cheng Yeh, King-Jang Yang, and Tao-Ming Ting.
  Knowledge discovery on rfm model using bernoulli sequence.
  *Expert Systems with Applications*, 36(3):5866–5871, 2009.
* Yin et al. [2020]

  Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel.
  Tabert: Pretraining for joint understanding of textual and tabular
  data.
  *arXiv preprint arXiv:2005.08314*, 2020.
* Yoon et al. [2020]

  Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar.
  Vime: Extending the success of self-and semi-supervised learning to
  tabular domain.
  *Advances in Neural Information Processing Systems*, 33, 2020.
* Zhang and Fan [2008]

  Kun Zhang and Wei Fan.
  Forecasting skewed biased stochastic ozone days: analyses, solutions
  and beyond.
  *Knowledge and Information Systems*, 14(3):299–326, 2008.

## Appendix A Algorithm

input: batch size N, constants (τ,ns

𝜏subscript𝑛𝑠\tau,n\_{s}, is\_noise, noise type ), structure of encoder (e), decoder (d);

initialization;

for *sampled minibatch {𝐗}𝐗\{\bm{X}\}* do

Divide minibatch 𝑿𝑿\bm{X} to K𝐾K subsets {𝒙𝟏,𝒙𝟐,𝒙𝟑,…,𝒙𝑲}subscript𝒙1subscript𝒙2subscript𝒙3…subscript𝒙𝑲\{\bm{x\_{1}},\bm{x\_{2}},\bm{x\_{3}},...,\bm{x\_{K}}\} ;

if *Add noise* then

Add noise to each subset in {𝒙𝟏,𝒙𝟐,𝒙𝟑,…,𝒙𝑲}subscript𝒙1subscript𝒙2subscript𝒙3…subscript𝒙𝑲\{\bm{x\_{1}},\bm{x\_{2}},\bm{x\_{3}},...,\bm{x\_{K}}\}. ;

end if

for *Each subset 𝐱𝐬subscript𝐱𝐬\bm{x\_{s}} in {𝐱𝟏,𝐱𝟐,𝐱𝟑,…,𝐱𝐊}subscript𝐱1subscript𝐱2subscript𝐱3…subscript𝐱𝐊\{\bm{x\_{1}},\bm{x\_{2}},\bm{x\_{3}},...,\bm{x\_{K}}\}* do

# Forward pass on encoder

𝒉𝒔=e​(𝒙𝒔)subscript𝒉𝒔𝑒subscript𝒙𝒔\bm{h\_{s}}=e(\bm{x\_{s}}) ;

# Forward pass on projection

𝒛𝒔=g​(𝒉𝒔)subscript𝒛𝒔𝑔subscript𝒉𝒔\bm{z\_{s}}=g(\bm{h\_{s}}) ;

# Forward pass on decoder

𝑿~𝒔=d​(𝒉𝒔)subscriptbold-~𝑿𝒔𝑑subscript𝒉𝒔\bm{\tilde{X}\_{s}}=d(\bm{h\_{s}}) ;

# Collect 𝒉𝒔,𝒛𝒔,𝑿~𝒔

subscript𝒉𝒔subscript𝒛𝒔subscriptbold-~𝑿𝒔\bm{h\_{s}},\bm{z\_{s}},\bm{\tilde{X}\_{s}}

end for

# We have collected {𝒉𝟏,𝒉𝟐,𝒉𝟑,…}subscript𝒉1subscript𝒉2subscript𝒉3…\{\bm{h\_{1}},\bm{h\_{2}},\bm{h\_{3}},...\}, {𝒛𝟏,𝒛𝟐,𝒛𝟑,…}subscript𝒛1subscript𝒛2subscript𝒛3…\{\bm{z\_{1}},\bm{z\_{2}},\bm{z\_{3}},...\} and {𝑿~𝟏,𝑿~𝟐,𝑿~𝟑,…}subscriptbold-~𝑿1subscriptbold-~𝑿2subscriptbold-~𝑿3…\{\bm{\tilde{X}\_{1}},\bm{\tilde{X}\_{2}},\bm{\tilde{X}\_{3}},...\}

# Compute reconstruction loss

L​o​s​s=1K​∑k=1K(1N​∑i=1N(𝑿(𝒊)−𝑿~𝒌(𝒊))2)𝐿𝑜𝑠𝑠1𝐾superscriptsubscript𝑘1𝐾1𝑁superscriptsubscript𝑖1𝑁superscriptsuperscript𝑿𝒊superscriptsubscriptbold-~𝑿𝒌𝒊2Loss=\frac{1}{K}\sum\_{k=1}^{K}\left(\frac{1}{N}\sum\_{i=1}^{N}\left(\bm{X^{(i)}}-\bm{{\tilde{X}\_{k}}^{(i)}}\right)^{2}\right), where k≡kt​h​s​u​b​s​e​t𝑘superscript𝑘𝑡ℎ𝑠𝑢𝑏𝑠𝑒𝑡k\equiv k^{th}subset, i≡it​h​s​a​m​p​l​e𝑖superscript𝑖𝑡ℎ𝑠𝑎𝑚𝑝𝑙𝑒i\equiv i^{th}sample;

if *Apply contrastive or distance loss* then

# Initialize contrastive and distance losses

ℒc,ℒd=0,0formulae-sequence

subscriptℒ𝑐subscriptℒ𝑑
00\mathcal{L}\_{c},\mathcal{L}\_{d}=0,0

# Generate all combinations of pairs of latents

{{𝒛𝟏,𝒛𝟐},{𝒛𝟏,𝒛𝟑},…}subscript𝒛1subscript𝒛2subscript𝒛1subscript𝒛3…\{\{\bm{z\_{1}},\bm{z\_{2}}\},\{\bm{z\_{1}},\bm{z\_{3}}\},...\} ;

# Compute contrastive and distance losses for all pairs

for *each jt​hsuperscript𝑗𝑡ℎj^{th} pair {𝐳𝐚,𝐳𝐛}∈S={{𝐳𝟏,𝐳𝟐},{𝐳𝟏,𝐳𝟑},…}subscript𝐳𝐚subscript𝐳𝐛𝑆subscript𝐳1subscript𝐳2subscript𝐳1subscript𝐳3…\{\bm{z\_{a}},\bm{z\_{b}}\}\in S=\{\{\bm{z\_{1}},\bm{z\_{2}}\},\{\bm{z\_{1}},\bm{z\_{3}}\},...\}* do

if *Apply contrastive loss* then

# Compute symmetric constrastive loss of pairs {𝒛𝒂,𝒛𝒃

subscript𝒛𝒂subscript𝒛𝒃\bm{z\_{a}},\bm{z\_{b}}} and update ℒcsubscriptℒ𝑐\mathcal{L}\_{c} ;

ℒc=ℒc+12​[l​(𝒛𝒂,𝒛𝒃)+l​(𝒛𝒃,𝒛𝒂)]subscriptℒ𝑐subscriptℒ𝑐12delimited-[]𝑙subscript𝒛𝒂subscript𝒛𝒃𝑙subscript𝒛𝒃subscript𝒛𝒂\mathcal{L}\_{c}=\mathcal{L}\_{c}+\frac{1}{2}[l(\bm{z\_{a}},\bm{z\_{b}})+l(\bm{z\_{b}},\bm{z\_{a}})], where l(.,.)l(.,.) refers to contrastive loss ;

end if

if *Apply distance loss* then

# Compute distance loss for each pair {𝒛𝒂,𝒛𝒃

subscript𝒛𝒂subscript𝒛𝒃\bm{z\_{a}},\bm{z\_{b}}} and update ℒdsubscriptℒ𝑑\mathcal{L}\_{d} ;

ℒd=ℒd+1N​∑i=1N(𝒛𝒂(𝒊)−𝒛~𝒃(𝒊))2subscriptℒ𝑑subscriptℒ𝑑1𝑁superscriptsubscript𝑖1𝑁superscriptsuperscriptsubscript𝒛𝒂𝒊superscriptsubscriptbold-~𝒛𝒃𝒊2\mathcal{L}\_{d}=\mathcal{L}\_{d}+\frac{1}{N}\sum\_{i=1}^{N}\left(\bm{z\_{a}^{(i)}}-\bm{{\tilde{z}\_{b}}^{(i)}}\right)^{2}, where i is it​hsuperscript𝑖𝑡ℎi^{th} sample in a subset ;

end if

end for

end if

# Compute average contrastive & distance losses and update total loss

L​o​s​s=L​o​s​s+ℒc/J+ℒd/J𝐿𝑜𝑠𝑠𝐿𝑜𝑠𝑠subscriptℒ𝑐𝐽subscriptℒ𝑑𝐽Loss=Loss+\mathcal{L}\_{c}/J+\mathcal{L}\_{d}/J, where J is total number of pairs ;

# Update network parameters

end for

return
encoder ;

Algorithm 1 Main learning algorithm

## Appendix B Data

### B.1 Adult Income Dataset

Train-Validation-Test Split: Training and test sets are provided separately [[24](#bib.bib24)]. We split the training set into training and validation sets using 80-20% split to search for hyper-parameters. Once hyper-parameters was fixed, we trained the model on the whole training set.

Features: The dataset has 14 attributes consisting of 8 categorical and 6 continuous features. We dropped the rows with missing values, and encoded categorical features using one-hot encoding. Features are normalized by subtracting the mean and dividing by the standard deviation, both of which are computed using training set.

Class imbalance: It is an imbalanced dataset, with only 25% of the samples being positive.

### B.2 BlogFeedback Dataset

Train-Validation-Test Split: The original dataset includes one training set, and 60 small test sets. We combined all the test sets into one test set. We split training set to training and validation using 80-20% split to search for hyper-parameters. We trained the final model using all of the training set.

Features: It includes 281 variables consisting of 280 features and 1 target variable indicating the number of comments a blog post received in the next 24 hours relative to the basetime. We converted the target (the last column in the dataset) to a binary variable, in which 0/1 indicates whether the blog post received any comments. We used min-max scaling to normalize the features.

Class imbalance: ∼36%similar-toabsentpercent36\sim 36\% of the samples are positive in training set while it is ∼30%similar-toabsentpercent30\sim 30\% in the test set.

### B.3 Data License

MNIST is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license. Obesity is available under MIT license while Aduld Income and BlogFeedback are under Open Data Commons Public Domain Dedication and License (PDDL).

## Appendix C Details of the experiments in the main paper

### C.1 Model architectures and hyper-parameters

Table A1: Architectures & hyper-parameters for the results in Table [1](#S3.T1 "Table 1 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). Abbreviations are; MNIST\*: MNIST with smaller latent dimension, CL: Contrastive Loss, DL: Distance Loss, MR: Mask ratio.

| Dataset | Encoder | Decoder | Projection | DL | CL | Subsets / Overlap | MR | Noise | Batch/Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST\* | [512, 256, 128] | [128, 256, 512] | [128] | Yes | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.15 | Swap | 32, 15 |
| MNIST | [512, 256, 512] | [512, 256, 512] | [512] | Yes | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.15 | Swap | 32, 15 |
| TCGA | [1024, 784, 784] | [784, 784, 1024] | [784] | No | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.2 | Gaussian | 512, 40 |
| Obesity | [1024, 1024] | [1024, 1024] | No | No | No | 6 / 0% | 0.2 | Gaussian | 32, 100 |
| Income | [1024, 1024] | [1024, 1024] | No | Yes | No | 5 / 25% | 0.2 | Gaussian | 256, 20 |
| Blog | [1024, 1024] | [1024, 1024] | No | Yes | No | 7 / 75% | 0.2 | Gaussian | 256, 20 |




Table A2: Architectures & hyper-parameters for the results of Shallow SubTab in Table [3](#S3.T3 "Table 3 ‣ 3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning").

| Dataset | Encoder | Decoder | Projection | DL | CL | Subsets / Overlap | MR | Noise | Batch/Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST\* | [784] | [784] | [784] | Yes | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.15 | Swap | 32, 15 |
| MNIST | [1024] | [1024] | [1024] | Yes | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.15 | Swap | 32, 15 |
| TCGA | [1024] | [1024] | [1024] | No | Yes, τ=0.1𝜏0.1\tau=0.1 | 4 / 75% | 0.2 | Gaussian | 512, 40 |
| Obesity | [1024] | [1024] | No | No | No | 6 / 0% | 0.2 | Gaussian | 32, 100 |
| Income | [1024] | [1024] | No | Yes | No | 5 / 25% | 0.2 | Gaussian | 256, 20 |
| Blog | [1024] | [1024] | No | Yes | No | 7 / 75% | 0.2 | Gaussian | 256, 20 |

Few other notes:

* •

  LeakyReLU is used as activation function for all networks.
* •

  Last layers of Encoder, Decoder and Projection shown in Table [A1](#A3.T1 "Table A1 ‣ C.1 Model architectures and hyper-parameters ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") are all linear.
* •

  Reconstruction loss is used for all experiments by default, except for one case, in which we used only contrastive loss in the ablation study on MNIST to compare it against reconstruction loss (See second row in Table [2](#S3.T2 "Table 2 ‣ 3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")).
* •

  Learning rate of 0.0010.0010.001 is used for all experiments since it usually performed the best. Based on that, we optimized the batch size and total number of epochs.

### C.2 Evaluation

![Refer to caption](/html/2110.04361/assets/images/sup_evaluation.png)


Figure A1: Once the SubTab model is trained, the joint embeddings for both training and test sets are obtained, and the quality of the representation is evaluated by using a linear classifier.

For all autoencoder baselines and self-supervised models, we evaluate the quality of the representation by training and evaluating a linear classifier on the embeddings of training and test set respectively. For SubTab, we use the joint embeddings as shown in Figure [A1](#A3.F1 "Figure A1 ‣ C.2 Evaluation ‣ Appendix C Details of the experiments in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). The models are trained and tested 10 times with different random seeds to compute mean ±plus-or-minus\pm stdev. For the linear model, l2subscript𝑙2l\_{2} regularization parameter is selected from a range of 10 logarithmically spaced values in [10−3,106]superscript103superscript106[10^{-3},10^{6}].

### C.3 Supporting visuals for experiments

![Refer to caption](/html/2110.04361/assets/images/sup_subtab_info_content_in_joint.png)


Figure A2: First experiment; measuring the information content of the joint embedding. The example in this figure shows the case for the joint embedding of three subsets.

![Refer to caption](/html/2110.04361/assets/images/sup_dividing_mnist_7subsets.png)


Figure A3: Setup for second and third experiments; MNIST is divided into seven subsets.

![Refer to caption](/html/2110.04361/assets/images/sup_six_models.png)


Figure A4: Setup for second experiment, in which five models are trained by using different combinations of seven subsets, and a sixth model that is kept untrained.

![Refer to caption](/html/2110.04361/assets/images/sup_individdual_info_content.png)


Figure A5: Second experiment; measuring the information content of individual subsets. For each subset, we obtained its embedding from each of the six models for both training and test sets, and evaluated the information content by using a linear classifier.

### C.4 Implementation and resources

We implemented our work using PyTorch [[38](#bib.bib38)]. AdamW optimizer [[29](#bib.bib29)] with b​e​t​a​s=(0.9,0.999)𝑏𝑒𝑡𝑎𝑠0.90.999betas=(0.9,0.999) and e​p​s=1​e−07𝑒𝑝𝑠1𝑒07eps=1e-07 is used for all of our experiments. We used a compute cluster consisting of Tesla K80 GPUs throughout this work. SubTab code implemented for MNIST can be found at: https://github.com/AstraZeneca/SubTab.

## Appendix D Insights and Comments

### D.1 Insights

* •

  The best performing models are the ones with an over-complete first hidden layer representation. This is also observed in denoising autoencoders [[43](#bib.bib43)].
* •

  A simple encoder architecture of [1024, 1024] works well for the most tabular datasets. Note that the second layer is just a linear layer. We can also use a one-layer encoder with 1024 dimension (i.e. removing linear layer).
* •

  It is previously observed that contrastive loss is not stable for small batch sizes [[7](#bib.bib7)]. However, we observed that using reconstruction loss together with contrastive loss makes contrastive loss more stable for small batch sizes.

### D.2 Comments

![Refer to caption](/html/2110.04361/assets/images/sup_applications.png)


Figure A6: Two of the possible applications of SubTab

Applications: Since SubTab works with subsets of features, it can be used in applications such as transfer learning (by using common features across different datasets), few-shot generalisation, domain adaptation, multi-task learning (some subsets might be more useful for one task, and some other for some other tasks), continual learning (adding new features to the dataset to improve performance) and so on in the context of tabular dataset (see Figure [A6](#A4.F6 "Figure A6 ‣ D.2 Comments ‣ Appendix D Insights and Comments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")). Moreover, when there is a distributional shift in certain features, the SubTab can accommodate such situations by relying on the subsets with features that have not changed.

Other modalities: In addition to tabular data, SubTab can be extended to other modalities such as images, text, time series, audio and so on by using random subspace of the data.

Additional limitations: Dividing tabular data to smaller chunks will result in representation collapse, meaning that the representations of subsets from very different samples might start to have similar representations. However, this is a very low risk since tabular data usually consists of heterogenous features with different statistical properties. Also, using two knobs (the percentage of overlapping features, and number of subsets) further reduces such a risk.

## Appendix E Different configuration of SubTab

### E.1 SimCLR

Removing the decoder, and training the encoder only with contrastive loss would result in the scheme similar to SimCLR. In this case, we can choose to:

* •

  Use two copies of the tabular data (i.e. we are not dividing the features to subsets), and add random noise to each to train the encoder in contrastive learning setting.
* •

  Or use subsets of the tabular data as usual, and train the encoder with constrastive loss. This choice is already shown in the ablation study listed in Table [2](#S3.T2 "Table 2 ‣ 3.4 Ablation study ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning").

### E.2 Other choices

We can also choose to use separate encoders for different subsets of the data.

## Appendix F Additional results for the experiments listed in the main paper

### F.1 MNIST

![Refer to caption](/html/2110.04361/assets/images/classes_inLatentSpace_test_unique_zs0.png)


(a) Using one subset out of four available

![Refer to caption](/html/2110.04361/assets/images/classes_inLatentSpace_test_unique_zs1.png)


(b) Using two subsets out of four available

![Refer to caption](/html/2110.04361/assets/images/classes_inLatentSpace_test_unique_zs2.png)


(c) Using three subsets out of four available

![Refer to caption](/html/2110.04361/assets/images/classes_inLatentSpace_test_unique_zs3.png)


(d) Using all four subsets

Figure A7: PCA and t-SNE clustering of representation for the model trained on four subsets with 75% overlap between subsets. Starting from one subset, we keep adding more subsets to get a better representation on the test set: a) One subset, b) Two subsets, c) Three subsets, d) Four (all) subsets.

![Refer to caption](/html/2110.04361/assets/images/mnist_28.png)


(a) Using 28 subsets for MNIST



Figure A8: a) Dividing MNIST to 28 subsets, each of which corresponds to one row in a 28x28 image. The test accuracy of a subset can be used as a measure of information content in that subset i.e. particular row in the image.

### F.2 Varying the number of subsets and the percentage of overlap for other datasets

![Refer to caption](/html/2110.04361/assets/images/blog_sweep.png)


(a) Blog

![Refer to caption](/html/2110.04361/assets/images/income_sweep.png)


(b) Income

![Refer to caption](/html/2110.04361/assets/images/rppa_sweep.png)


(c) TCGA

Figure A9:  Test accuracy on (a) Blog, (b) Income, and (c) TCGA datasets over different number of subsets and varying levels of overlaps for the base models of each.

### F.3 Sensitivity analysis for masking ratio (p) and initialization

![Refer to caption](/html/2110.04361/assets/images/p_sweep.png)


(a) Sensitivity to p𝑝p.

![Refer to caption](/html/2110.04361/assets/images/sensitivity_initialization.png)


(b) Sensitivity to the initialization.

Figure A10: a) The test accuracy for MNIST over different levels of the masking ratio, p𝑝p.
b) Measuring the sensitivity of the model to the initialization by initializing an untrained model with different random seeds and using it to measure the importance of each of the seven subsets for MNIST. This is a repeat of the experiment in the Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning").

The sensitivity analysis for the two most important hyper-parameters, i.e. the number of subsets and the overlaps between different subsets, is already shown in Figures [3](#S3.F3 "Figure 3 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") and [A9](#A6.F9 "Figure A9 ‣ F.2 Varying the number of subsets and the percentage of overlap for other datasets ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"). In this section, we show two more sensitivity analysis, one on masking ratio, and one on the initialization of the model.

Masking ratio In Figure [A10](#A6.F10 "Figure A10 ‣ F.3 Sensitivity analysis for masking ratio (p) and initialization ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")a, we show the sensitivity to the percentages of features corrupted (p𝑝p, or masking ratio) in each subset in our method. For this, we plot the test accuracy for MNIST over different levels of the masking ratio. A good range for p𝑝p is usually [0.1-0.3], and the model performance is usually robust to the different values of p𝑝p as shown.

Sensitivity to initialization In Figure [5](#S3.F5 "Figure 5 ‣ 3.3 Results ‣ 3 Experiments ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"), we showed that we can use untrained model to discover informative subsets of the data. Figure [A10](#A6.F10 "Figure A10 ‣ F.3 Sensitivity analysis for masking ratio (p) and initialization ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning")b shows how sensitive this analysis is to the initialization of the network. We initialized the model used for MNIST 10 times with different random seeds, and re-ran the same test of discovering informative subsets. The plot shows the mean test accuracy of the linear model evaluated on the embeddings from the untrained Subtab with 95% confidence interval. The variation is very small, and so we can conclude that the model is not sensitive to the initialization.

### F.4 Using different aggregation functions

We experimented with different aggregations functions when aggregating the latent representations of subsets for the case of MNIST. We also tried concatenating the representations for the downstream classification task using linear model. Table [A3](#A6.T3 "Table A3 ‣ F.4 Using different aggregation functions ‣ Appendix F Additional results for the experiments listed in the main paper ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") summarizes the result from one such experiment for MNIST. As shown in the table, the performance is robust to the different aggregation methods, but it drops when we use concatenation. Please note that we used mean-aggregation throughout our experiments reported in the paper since it generally performs better.

Table A3: Testing classification accuracy of the linear model for the test set of MNIST when using different aggregation functions for the latent representations of subsets.

| Aggregation method | Test Accuracy (%) |
| --- | --- |
| Mean | 97.86 |
| Sum | 97.77 |
| Max | 97.79 |
| Min | 97.74 |
| Concatenation | 95.92 |

## Appendix G Experiments using synthetic data

Tabular datasets can be very different from one another in terms of the statistics of the features. Some of them might have many redundant, or uninformative features while some other might have more informative features than the average. Thus, to test the SubTab under different scenarios, we ran more experiments using 3 synthetic datasets that we generated using m​a​k​e​\_​c​l​a​s​s​i​f​i​c​a​t​i​o​n𝑚𝑎𝑘𝑒\_𝑐𝑙𝑎𝑠𝑠𝑖𝑓𝑖𝑐𝑎𝑡𝑖𝑜𝑛make\\_classification module of scikit-learn library [[40](#bib.bib40)].

### G.1 Datasets

Each dataset has 10 classes and 10k samples, 10% of which is used as the test set. We generated them such that the clusters are not easily separable to make the problem more difficult. Specifics of the datasets are summarized in Table [A4](#A7.T4 "Table A4 ‣ G.1 Datasets ‣ Appendix G Experiments using synthetic data ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning").

Table A4: Summary of three synthetic datasets. Please note that the redundant features are generated using the linear combination of informative ones.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Dataset | Total features | Informative features | Redundant features | Uninformative features |
| Dataset-1 | 1000 | 12 | 30 | 958 |
| Dataset-2 | 100 | 60 | 30 | 10 |
| Dataset-3 | 100 | 4 | 30 | 66 |

### G.2 SubTab set-up

* •

  The SubTab utilized an encoder architecture of [1024, 1024], of which the first hidden layer uses LeakyReLU, and the second one is a linear layer.
* •

  We used 2 subsets with 25% overlap between them. Other parameters are masking ratio p=0.2𝑝0.2p=0.2, Gaussian noise with σ=0.1𝜎0.1\sigma=0.1.
* •

  We trained the model using only reconstruction loss and used mean-aggregation when aggregating the latent representations of the subsets at test time.

Please note that this set-up seems to work well for most tabular datasets as it did in other datasets reported in our work.

### G.3 Evaluation

We trained and tested a logistic regression model on the raw features of the data, as well as the embeddings obtained by using mean-aggregation of the representations of subsets from the SubTab that is pre-trained on the training set. Table [A5](#A7.T5 "Table A5 ‣ G.3 Evaluation ‣ Appendix G Experiments using synthetic data ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") summarizes the results on the test set. We can see that the SubTab improves the results in all 3 datasets, as much as 100% for the most difficult dataset (Dataset-1). This result gives us a little bit more insight into how the SubTab might improve the results in the datasets with different nature.

Table A5: Summary of the results on the test accuracy (%) for three synthetic datasets.

| Dataset | Raw features | SubTab embedding |
| --- | --- | --- |
| Dataset-1 | 31.2 | 61.9 |
| Dataset-2 | 83.5 | 90.5 |
| Dataset-3 | 79.9 | 82.1 |

## Appendix H Experiments using OpenML-CC18 datasets

Ideally, the type of data we want for the task of representation learning would be a high-dimensional, large dataset with multiple-classes. The most tabular datasets do not fit this criteria. Moreover, OpenML-CC18 [[2](#bib.bib2)] includes 72 datasets, in which some datasets have a low number of features (<10) and/or a low number of samples (<1000) such as the last three datasets listed in Table [A6](#A8.T6 "Table A6 ‣ Appendix H Experiments using OpenML-CC18 datasets ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"):

* •

  Dataset 9: Diabetes
* •

  Dataset 10: Blood transfusion service center
* •

  Dataset 11: Phoneme

Thus, we excluded such datasets from consideration, and picked other eight datasets in Table [A6](#A8.T6 "Table A6 ‣ Appendix H Experiments using OpenML-CC18 datasets ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") for experiments. Please note that although Electricity dataset has only 8 features, we included it in our analysis since it does have relatively large sample size.

Table A6: Summary of datasets from OpenML-CC18. First eight datasets are used for the experiments.

| Dataset | Name | Total Features | Number of Samples | Number of Classes |
| --- | --- | --- | --- | --- |
| 1 | First Order Theorem Proving [[3](#bib.bib3)] | 51 | 6118 | 6 |
| 2 | Wall Robot [[12](#bib.bib12), [14](#bib.bib14)] | 24 | 5456 | 4 |
| 3 | Gesture Phase Segmentation [[12](#bib.bib12), [31](#bib.bib31)] | 32 | 9873 | 5 |
| 4 | Ozone Level 8hr [[12](#bib.bib12), [47](#bib.bib47)] | 72 | 2534 | 2 |
| 5 | Electricity [[16](#bib.bib16)] | 8 | 45312 | 2 |
| 6 | Texture [[18](#bib.bib18)] | 40 | 5500 | 11 |
| 7 | DNA [[12](#bib.bib12), [35](#bib.bib35)] | 180 | 3186 | 3 |
| 8 | Climate [[12](#bib.bib12), [30](#bib.bib30)] | 20 | 540 | 2 |
| 9 | Diabetes [[33](#bib.bib33)] | 8 | 768 | 2 |
| 10 | Blood transfusion service center [[12](#bib.bib12), [44](#bib.bib44)] | 4 | 748 | 2 |
| 11 | Phoneme [[2](#bib.bib2)] | 5 | 5404 | 2 |

### H.1 Data Pre-processing

We cleaned up the datasets by removing rows with missing data if there is any, and/or by removing the features such as user ID. We used min-max scaling to scale all datasets, and split the data as 70-10-20% training, validation and test set. We trained the final models on 80% training set by combining training and validation set.

### H.2 Models and Evaluation

We trained and compared six models: i) Logistic Regression as our baseline, ii) Autoencoder (AE) iii) Autoencoder (AE) with dropout (p=0.04), iv) VIME-self, v) SubTab, and vi) SubTab with dropout (p=0.04).

All neural networks used the same four-layer encoder architecture: [256, 256, 256, 256]. For the networks with dropout, we used the same dropout rate, p=0.04. We trained SubTab by using two subsets with zero overlap and using only reconstruction loss. For all models using neural networks, we trained and evaluated them with 10 different random seeds. Evaluation of these models is done by training a logistic regression model using the embeddings of training set (i.e. 80% of the data), and by testing it using the embeddings of the test set (20% of the data).

### H.3 Results

Table A7: The results for the eight datasets. Please refer to Table [A6](#A8.T6 "Table A6 ‣ Appendix H Experiments using OpenML-CC18 datasets ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning") for the name of the datasets.

| Model | Dataset-1 | Dataset-2 | Dataset-3 | Dataset-4 | Dataset-5 | Dataset-6 | Dataset-7 | Dataset-8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 46.96 | 68.46 | 46.93 | 94.01 | 76.09 | 99.71 | 95.12 | 96.3 |
| Autoencoder (AE) | 50.40±plus-or-minus\pm0.83 | 86.83±plus-or-minus\pm0.91 | 49.07±plus-or-minus\pm0.55 | 94.84±plus-or-minus\pm0.34 | 81.32±plus-or-minus\pm0.16 | 99.34±plus-or-minus\pm0.28 | 93.48±plus-or-minus\pm0.97 | 95.01±plus-or-minus\pm0.90 |
| AE w/ Dropout (p=0.04) | 50.52±plus-or-minus\pm0.71 | 86.87±plus-or-minus\pm0.44 | 49.43±plus-or-minus\pm1.17 | 94.69±plus-or-minus\pm0.14 | 81.54±plus-or-minus\pm0.36 | 98.75±plus-or-minus\pm0.18 | 91.48±plus-or-minus\pm0.43 | 95.04±plus-or-minus\pm1.06 |
| VIME-self | 44.99±plus-or-minus\pm0.9 | 74.23±plus-or-minus\pm1.21 | 46.08±plus-or-minus\pm0.37 | 94.28±plus-or-minus\pm0.31 | 73.92±plus-or-minus\pm1.08 | 95.49±plus-or-minus\pm0.88 | 89.97±plus-or-minus\pm0.97 | 95.56±plus-or-minus\pm0.42 |
| SubTab | 50.8±plus-or-minus\pm0.76 | 89.37±plus-or-minus\pm0.72 | 50.33±plus-or-minus\pm0.86 | 94.74±plus-or-minus\pm0.28 | 82.11±plus-or-minus\pm0.26 | 99.59±plus-or-minus\pm0.22 | 92.62±plus-or-minus\pm0.59 | 93.89±plus-or-minus\pm1.55 |
| SubTab w/ Dropout (p=0.04) | 51.48±plus-or-minus\pm0.77 | 89.81±plus-or-minus\pm0.69 | 49.93±plus-or-minus\pm0.77 | 94.85±plus-or-minus\pm0.31 | 82.31±plus-or-minus\pm0.34 | 99.23±plus-or-minus\pm0.36 | 91.41±plus-or-minus\pm1.03 | 93.33±plus-or-minus\pm0.77 |

Based on the results shown in Table [A7](#A8.T7 "Table A7 ‣ H.3 Results ‣ Appendix H Experiments using OpenML-CC18 datasets ‣ SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning"), we can make following observations:

* •

  If the dataset is suitable for pre-training / representation learning, the SubTab tends to perform better than the other approaches, including VIME-self [[46](#bib.bib46)]. This was the case in the datasets [1-5].
* •

  If the dataset is trivial (i.e. logistic regression already gives a very decent performance), we might be better off using simple models such as logistic regression as this was the case in datasets [6, 7, and 8] (i.e. Texture, DNA, and Climate datasets respectively).

[◄](/html/2110.04360)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2110.04361)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2110.04361)
[View original  
on arXiv](https://arxiv.org/abs/2110.04361)[►](/html/2110.04362)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Mar 6 22:06:11 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
