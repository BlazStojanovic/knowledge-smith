---
arxiv: '2108.12296'
authors:
- Sajad Darabi UCLA sajad.darabi@cs.ucla.edu &Shayan Fazeli UCLA shayan.fazeli@cs.ucla.edu
  &Ali Pazokitoroudi UCLA alipazoki@cs.ucla.edu &Sriram Sankararaman UCLA sriram@cs.ucla.edu
  &Majid Sarrafzadeh UCLA majid@cs.ucla.edu
parser: ar5iv
retrieved: '2026-05-09'
source: paper
title: 'Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain'
url: https://arxiv.org/abs/2108.12296
year: 2021
---

# Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain

Sajad Darabi
  
UCLA
  
sajad.darabi@cs.ucla.edu
  
&Shayan Fazeli
  
UCLA
  
shayan.fazeli@cs.ucla.edu
  
&Ali Pazokitoroudi
  
UCLA
  
alipazoki@cs.ucla.edu
  
&Sriram Sankararaman
  
UCLA
  
sriram@cs.ucla.edu
  
&Majid Sarrafzadeh
  
UCLA
  
majid@cs.ucla.edu

###### Abstract

Recent literature in self-supervised has demonstrated significant progress in closing the gap between supervised and unsupervised methods in the image and text domains. These methods rely on domain-specific augmentations that are not directly amenable to the tabular domain. Instead, we introduce Contrastive Mixup, a semi-supervised learning framework for tabular data and demonstrate its effectiveness in limited annotated data settings. Our proposed method leverages Mixup-based augmentation under the manifold assumption by mapping samples to a low dimensional latent space and encourage interpolated samples to have high a similarity within the same labeled class. Unlabeled samples are additionally employed via a transductive label propagation method to further enrich the set of similar and dissimilar pairs that can be used in the contrastive loss term. We demonstrate the effectiveness of the proposed framework on public tabular datasets and real-world clinical datasets.

## 1 Introduction

Deep learning has shown tremendous success in domains where large annotated datasets are readily available such as vision, text, speech via supervised learning. Implicitly learned by these models is an intermediate representation that lends itself useful for downstream tasks. Unfortunately, in many settings such as healthcare, large annotated datasets are not readily available to enable learning such valuable representations. As a result, there has been a push towards learning these in an unsupervised or semi-supervised manner as unannotated data on the other hand may be readily available for free and a lot of it in many cases. Recent literature has shown significant progress towards learning these useful representations without human-annotated data, closing the gap between supervised and unsupervised learning, and in some cases demonstrating superior transfer learning properties compared to its supervised counterpart [[9](#bib.bib9), [3](#bib.bib3)].

Self-supervised methods have emerged as a promising approach to achieving appealing results in various applications without requiring labeled examples. This is typically done via pretext tasks closely related to the downstream tasks of interest and typically differs from domain to domain. For example, in the image domain colorization [[27](#bib.bib27)], jigsaw puzzle[[20](#bib.bib20)], rotation prediction [[7](#bib.bib7)] have been previously presented as pretext tasks useful for learning such representations. Similarly, in the text domain, commonly used pretext tasks, such as predicting masked words and context words, have been widely used [[14](#bib.bib14), [4](#bib.bib4)]. More recently, contrastive learning methods introduced leverage domain specific transformations to create multiple semantically similar examples such as random cropping or flipping for images that preserve the semantic meaning and encourage the network to be invariant to such transformations achieving great success. Such pretext tasks and transformations cannot be readily applied that do not have the same structural information, as an example tabular data111Tabular data contains a set of rows (examples) and columns (features) that may be permutation invariant..

It is not clear how to generate new semantically similar examples for tabular data. Moreover, in many settings, tabular data contains both categorical and continuous features which require different treatment. In this work, we focus on tabular data settings that contain a small set of annotated samples and a relatively sizeable unlabeled set of samples. Specifically, we propose a framework for improving downstream task performance in this semi-supervised setting. Our method consists of a semi-self-supervised pretraining step where a feature reconstruction pretext task and a supervised contrastive loss term are used. Various forms of Mixup augmentation [[26](#bib.bib26)] has been used in the image domain, where new examples are created by taking convex combinations of pairs of examples. This may lead to low probable samples in the dataspace for tabular data. Instead, we leverage the manifold assumption 222Manifold Assumption: High-dimensional data lies (roughly) on a low-dimensional manifold. and mix samples in the latent space to create multiple views for our contrastive loss term. The unlabeled subset is further leveraged by pre-training the encoder and using label propagation [[11](#bib.bib11)] to generate pseudo-labels for the unlabeled samples. Subsequently, the trained encoder and samples, for which we have generated pseudo-labels for, are transferred to a downstream task where a simple predictor with Mixup [[26](#bib.bib26)] augmentation is trained. We show that our proposed framework leads to improvements on various tabular datasets, such as UCI and Genomics (UK Biobank).

## 2 Related Works

Our work fits the semi-supervised learning framework [[2](#bib.bib2)] where both labeled and unlabeled samples are used to improve downstream task performance. We draw from recent literature in self-supervised representation learning, pseudo-labeling [[10](#bib.bib10)] and Mixup based supervised learning [[26](#bib.bib26)].

At the core of the self-supervised methods are pre-text tasks, where labels are created from the raw unlabeled data itself, and supervised losses are then used to learn useful representations for downstream prediction tasks. In these lines of works, examples of domain-specific pre-text tasks such as jigsaw puzzle [[16](#bib.bib16)], colorization [[27](#bib.bib27)], relative positioning prediction [[5](#bib.bib5)] have been introduced for images, masked word prediction [[14](#bib.bib14), [15](#bib.bib15)], next sentence prediction [[13](#bib.bib13)] for text. There is also existing work on self-supervised/semi-supervised learning methods. For example, a similar in-painting task [[18](#bib.bib18)] can be used to predict masked features in a row as done in [[24](#bib.bib24), [1](#bib.bib1)]. On the other hand, many recent self-supervised methods are based on contrastive representation learning [[3](#bib.bib3)], in which domain-specific augmentation (e.g., random crop, random color distortion for images) are used to create "similar" samples, and the normalized cross-entropy loss [[21](#bib.bib21), [17](#bib.bib17)] is used to increase the similarity of "positive" pairs in the latent space, and decrease the similarity of "negative" pairs. A downside of generating negative and positives without label information is that examples belonging to the same class may be pushed apart. In [[12](#bib.bib12)] authors leverage label information to consider many "similar" examples to be pulled closer to one another and farther away from the dissimilar examples. As these methods leverage properties inherent to the raw data, they are not amenable to the tabular domain, which is the focus of this work.

The setting we are considering fits the semi-supervised learning framework. Prior work on semi-supervised learning can be broadly separated into two main categories: methods that add an unsupervised loss term to the supervised task as a regularizer, e.g., [[20](#bib.bib20), [22](#bib.bib22), [8](#bib.bib8)] and methods that assign pseudo labels [[10](#bib.bib10)] to the unlabeled samples. Recently, [[25](#bib.bib25)] proposed VIME, a state-of-the-art semi-supervised method for the tabular domain where they leverage consistency regularization and in-painting [[18](#bib.bib18)] inspired augmentation. In [[10](#bib.bib10)] the current network trained on labeled samples is used to infer pseudo-labels on unlabeled samples using a confidence threshold, which is then treated similar to labeled samples to minimize entropy. Transductive learning is more generic in that instead of training a generic classifier, the goal is to used patterns in the labeled set to infer labels for the unlabeled set. Label propagation has been widely used in transductive learning in the image domain in an online fashion where CNN features are used for few-shot learning [[6](#bib.bib6)]. Along this line of work, recently [[11](#bib.bib11)] use label propagation in an offline fashion by treating the labeled and unlabeled samples as a bipartite graph where edges computed via diffusion similarity [[28](#bib.bib28)]. In this work, we propose a semi-supervised framework for the tabular domain where we leverage Mixup [[26](#bib.bib26)] based augmentation, which interpolates samples using a convex combination and assigns soft labels according to the mixing ratio in the latent space [[23](#bib.bib23)] and encourage samples interpolated from the same class to have high similarity.

## 3 Preliminaries

To present our method we formulate the self-supervised and semi-supervised problem. Consider a dataset with N𝑁N examples: Our assumption is that there is a small subset of this dataset for which labels are available: 𝒟L={(xi,yi)}i=1NLsubscript𝒟𝐿superscriptsubscriptsubscript𝑥𝑖subscript𝑦𝑖𝑖1subscript𝑁𝐿\mathcal{D}\_{L}=\{(x\_{i},y\_{i})\}\_{i=1}^{N\_{L}}, and the rest of the dataset is unlabeled: 𝒟U={(xi)}i=1NUsubscript𝒟𝑈superscriptsubscriptsubscript𝑥𝑖𝑖1subscript𝑁𝑈\mathcal{D}\_{U}=\{(x\_{i})\}\_{i=1}^{N\_{U}} where xisubscript𝑥𝑖x\_{i} are observations sampled from a data-generating distribution p​(x)𝑝𝑥p(x) and yi∈{0,1,⋯,c}subscript𝑦𝑖01⋯𝑐y\_{i}\in\{0,1,\cdots,c\} is a discrete label set. We consider settings where the majority of the data is unlabeled i.e. |𝒟L|≪|𝒟U|much-less-thansubscript𝒟𝐿subscript𝒟𝑈|\mathcal{D}\_{L}|\ll|\mathcal{D}\_{U}|. In supervised learning a classifier f:𝒳→𝒴∈ℱ:𝑓→𝒳𝒴ℱf:\mathcal{X}\rightarrow\mathcal{Y}\in\mathcal{F} is a function learned by an ML algorithm which aims at optimizing f𝑓f for a given loss function lA​(⋅)subscript𝑙𝐴⋅l\_{A}(\cdot) i.e. f=minf∈ℱ​∑i=1NlA​(f​(xi),yi)𝑓subscript𝑓ℱsuperscriptsubscript𝑖1𝑁subscript𝑙𝐴𝑓subscript𝑥𝑖subscript𝑦𝑖f=\min\_{f\in\mathcal{F}}\sum\_{i=1}^{N}{l\_{A}(f(x\_{i}),y\_{i})}. In this limited labeled data regime a supervised model is most likely to overfit, hence we propose to use the unlabeled samples to improve the models generalization.

### 3.1 Self-Supervised Learning

Self-supervised methods leverage unlabeled data to learn useful representations for downstream prediction tasks. Many techniques have been proposed for images where useful visual representations are learned through pre-text tasks such as in-painting, rotation, jig-saw [[16](#bib.bib16), [18](#bib.bib18), [7](#bib.bib7)], and more recently, the gap between supervised and unsupervised models have drastically been reduced through contrastive representation learning method [[9](#bib.bib9), [3](#bib.bib3)]. Generally, in contrastive representation, learning a batch of N𝑁N samples is augmented through an augmentation function Aug(.)\text{Aug}(.) to create a multi-viewed batch with 2​N2𝑁2N pairs, {xi~,yi~}i=1​⋯​2​Nsubscript~subscript𝑥𝑖~subscript𝑦𝑖𝑖1⋯2𝑁\{\tilde{x\_{i}},\tilde{y\_{i}}\}\_{i=1\cdots 2N} where x~2​ksubscript~𝑥2𝑘\tilde{x}\_{2k} and x~2​k−1subscript~𝑥2𝑘1\tilde{x}\_{2k-1} are two random augmentations of the same sample xksubscript𝑥𝑘x\_{k} for k={1,⋯,N}𝑘1⋯𝑁k=\{1,\cdots,N\}. The samples are fed to an encoder e:x→z:𝑒→𝑥𝑧e:x\rightarrow z which takes a sample x∈𝒳𝑥𝒳x\in\mathcal{X}, to obtain a latent representation z=e​(x)𝑧𝑒𝑥z=e(x). Typically when defining a pre-text task, a predictive model is trained jointly to minimize a self-supervised loss function ls​ssubscript𝑙𝑠𝑠l\_{ss}.

|  |  |  |  |
| --- | --- | --- | --- |
|  | mine,h𝔼(x,y~)∼P​(X,Y~)[l(y~,h∘e(x)]\min\limits\_{e,h}\mathbb{E}\_{(x,\tilde{y})\sim P(X,\tilde{Y})}\big{[}l(\tilde{y},h\circ e(x)] |  | (1) |

where hℎh maps z𝑧z to an embedding space h:z→v:ℎ→𝑧𝑣h:z\rightarrow v. Within a mutliviewed batch, i∈ℐ={1,⋯​2​N}𝑖ℐ1⋯2𝑁i\in\mathcal{I}=\{1,\cdots 2N\} the self supervised loss is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | l=∑i∈ℐ−log​(exp​(sim​(vi,vj​(i))/τ)∑n∈ℐ\{i}exp​(sim​(vi,vn)/τ))𝑙subscript𝑖ℐlogexpsimsubscript𝑣𝑖subscript𝑣𝑗𝑖𝜏subscript𝑛\ℐ𝑖expsimsubscript𝑣𝑖subscript𝑣𝑛𝜏l=\sum\_{i\in\mathcal{I}}-\text{log}\Big{(}\frac{\text{exp}(\text{sim}(v\_{i},v\_{j(i)})/\tau)}{\sum\_{n\in\mathcal{I}\backslash\{i\}}{\text{exp}(\text{sim}(v\_{i},v\_{n})/\tau)}}\Big{)} |  | (2) |

Here, sim​(⋅,⋅)∈ℜ+sim⋅⋅superscript\text{sim}(\cdot,\cdot)\in\Re^{+} is a similarity function (e.g. dot product or cosine similarity), τ∈ℜ+𝜏superscript\tau\in\Re^{+} is a scalar temperature parameter, i𝑖i is the anchor, 𝒜​(i)𝒜𝑖\mathcal{A}(i) is the positive(s) and ℐ\{i}\ℐ𝑖\mathcal{I}\backslash\{i\} are the negatives. The positive and negative samples refer to samples that are semantically similar and dissimilar respectively. Intuitively, the objective of this function is to bring the positives and the anchor closer in the embedding space v𝑣v than the anchor and the negatives, i.e. sim​(va,v+)>sim​(va,v−)simsuperscript𝑣𝑎superscript𝑣simsuperscript𝑣𝑎superscript𝑣\text{sim}(v^{a},v^{+})>\text{sim}(v^{a},v^{-}), where vasuperscript𝑣𝑎v^{a} is the anchor and v+superscript𝑣v^{+}, v−superscript𝑣v^{-} are the positive and negative respectively.

### 3.2 Semi-Supervised Learning

In semi-supervised learning (SSL), the dataset is comprised of two disjoint sets DLsubscript𝐷𝐿D\_{L}. DUsubscript𝐷𝑈D\_{U}, where predictive model f𝑓f is optimized to minimize a supervised loss, jointly with an unsupervised loss. In other words:

|  |  |  |  |
| --- | --- | --- | --- |
|  | minf⁡𝔼(x,y)∼P​(X,Y)​[l​(y,f​(x))]+β​𝔼(x,yp​s)∼P​(X,Yp​s)​[lu​(yp​s,f​(x))]subscript𝑓subscript𝔼similar-to𝑥𝑦𝑃𝑋𝑌delimited-[]𝑙𝑦𝑓𝑥𝛽subscript𝔼similar-to𝑥subscript𝑦𝑝𝑠𝑃𝑋subscript𝑌𝑝𝑠delimited-[]subscript𝑙𝑢subscript𝑦𝑝𝑠𝑓𝑥\min\limits\_{f}\mathbb{E}\_{(x,y)\sim P(X,Y)}\big{[}l(y,f(x))\big{]}+\beta\mathbb{E}\_{(x,y\_{ps})\sim P(X,Y\_{ps})}\big{[}l\_{u}(y\_{ps},f(x))\big{]} |  | (3) |

The first term is estimated over the small labeled subset 𝒟Usubscript𝒟𝑈\mathcal{D}\_{U}, and the second unsupervised loss is estimated over the more significant unlabeled subset. The unsupervised loss function lusubscript𝑙𝑢l\_{u} is defined to help the downstream prediction task, such as consistency loss training [[16](#bib.bib16), [22](#bib.bib22)], or in our case, a supervised objective on pseudo-labeled samples [[10](#bib.bib10)].

## 4 Method

This section describes our proposed method Contrative Mixup, a semi-supervised method for multi-modal tabular data where structural (spatial or sequential) data augmentations are not readily available. To this end, we first propose our semi-supervised training to learn an encoder and subsequently propose to train a classifier using the pre-trained encoder and pseudo-labels.

### 4.1 Semi-Self-Supervised Learning for Tabular Data

!(/html/2108.12296/assets/x1.png)

Figure 1: Overview of our semi-self-supervised framework. The encoder is trained using both labeled and unlabeled subsets via the reconstruction loss and contrastive loss terms. Pseudo-labeles are used

We make use of the manifold assumption where high dimensional data roughly lie on a low dimensional manifold and then leverage Mixup [[26](#bib.bib26)] based data interpolation for creating positive and negative samples. By doing so we mitigate creating low-probable samples in the original data space.

In our setting we represent the mutli-modal tabular data rows xisubscript𝑥𝑖x\_{i} as a concatenation of discrete D=[D1,⋯,D|D|]𝐷

subscript𝐷1⋯subscript𝐷𝐷D=[D\_{1},\cdots,D\_{|D|}] and continuous features 𝒞=[C1,⋯,C|𝒞|]𝒞

subscript𝐶1⋯subscript𝐶𝒞\mathcal{C}=[C\_{1},\cdots,C\_{|\mathcal{C}|}]. The raw features xi∈ℜdsubscript𝑥𝑖superscript𝑑x\_{i}\in\Re^{d} are fed through an embedding layer E:x→x¯:𝐸→𝑥¯𝑥E:x\rightarrow\bar{x} that results in a feature vector x¯∈ℜ|C|+∑i|D|d|𝒟i|¯𝑥superscript𝐶superscriptsubscript𝑖𝐷subscript𝑑subscript𝒟𝑖\bar{x}\in\Re^{|C|+\sum\_{i}^{|D|}{d\_{|\mathcal{D}\_{i}|}}}, that is a concatenation of the continuous features 𝒞𝒞\mathcal{C} and embedded discrete features 𝒟𝒟\mathcal{D}, where d|Di|subscript𝑑subscript𝐷𝑖d\_{|D\_{i}|} is the embedding dimension for each discrete feature 𝒟isubscript𝒟𝑖\mathcal{D}\_{i}. The embedded features are fed to an encoder z=e​(x¯)𝑧𝑒¯𝑥z=e(\bar{x}), and subsequently fed to a feature estimation pre-text task, as well as a semi-supervised contastive loss term shown in Figure [1](#S4.F1 "Figure 1 ‣ 4.1 Semi-Self-Supervised Learning for Tabular Data ‣ 4 Method ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain").

In the tabular domain, data augmentation commonly used in the image domain cannot be used. Instead, we propose to interpolate between samples of the same class to create positive examples and use a supervised contrastive loss term in the latent space. Given a batch of labeled examples 𝒟ℬ={xk,yk}k=1Ksubscript𝒟ℬsubscriptsuperscriptsubscript𝑥𝑘subscript𝑦𝑘𝐾𝑘1\mathcal{D\_{B}}=\{x\_{k},y\_{k}\}^{K}\_{k=1}, we create a new labeled sample (x^,y^)^𝑥^𝑦(\hat{x},\hat{y}) by interpolating within the same labeled pair of examples

|  |  |  |  |
| --- | --- | --- | --- |
|  | x^=λ​x1+(1−λ)​x2^𝑥𝜆subscript𝑥11𝜆subscript𝑥2\hat{x}=\lambda x\_{1}+(1-\lambda)x\_{2} |  | (4) |

where λ𝜆\lambda is a scalar sampled from a random uniform λ∼𝒰​(0,α)similar-to𝜆𝒰0𝛼\lambda\sim\mathcal{U}(0,\alpha) with α∈[0,0.5]𝛼00.5\alpha\in[0,0.5]. The newly generated sample x^^𝑥\hat{x} will be λ𝜆\lambda close to x1subscript𝑥1x\_{1} and 1−λ1𝜆1-\lambda to x2subscript𝑥2x\_{2} with the same label as x1subscript𝑥1x\_{1} and x2subscript𝑥2x\_{2}, i.e. y1=y2=y^subscript𝑦1subscript𝑦2^𝑦y\_{1}=y\_{2}=\hat{y}. As opposed to randomly interpolating between samples and enforcing closeness between samples of different labels, we encourage samples of the same label to lie close to one another in the latent space.

Applying Mixup in the input space for tabular data may lead to low probable samples due to the multi-modality of the data and presence of categorical columns. Instead, we map samples to the hidden space and interpolate there. More concretely, given an encoder e𝑒e, that is comprised of T𝑇T layers ftsubscript𝑓𝑡f\_{t}, for t∈{1,⋯​T}𝑡1⋯𝑇t\in\{1,\cdots T\}. The samples are fed through to an intermediate representation htsubscriptℎ𝑡h\_{t} at layer t𝑡t. This layer contains a more abstract representations of the input samples x1subscript𝑥1x\_{1} and x2subscript𝑥2x\_{2}. The samples are interpolated within this intermediate layer as

|  |  |  |  |
| --- | --- | --- | --- |
|  | h~12t=λ​h1t+(1−λ)​h2tsubscriptsuperscript~ℎ𝑡12𝜆subscriptsuperscriptℎ𝑡11𝜆subscriptsuperscriptℎ𝑡2\tilde{h}^{t}\_{12}=\lambda h^{t}\_{1}+(1-\lambda)h^{t}\_{2} |  | (5) |

where hitsubscriptsuperscriptℎ𝑡𝑖h^{t}\_{i} is obtained by feeding samples x¯isubscript¯𝑥𝑖\bar{x}\_{i} through the encoder until layer t𝑡t. Subsequently, the newly generated samples h~i′​itsubscriptsuperscript~ℎ𝑡superscript𝑖′𝑖\tilde{h}^{t}\_{i^{\prime}i} as well as the original samples hitsubscriptsuperscriptℎ𝑡𝑖h^{t}\_{i} are fed through the rest of the encoder layers t,⋯,T

𝑡⋯𝑇t,\cdots,T to obtain the latent representation z𝑧z. In this space we distinguish between zlsubscript𝑧𝑙z\_{l} and zusubscript𝑧𝑢z\_{u}, which are the latent representation of labeled and unlabeled samples respectively in. Note that initially we only consider the labeled portion for the contrastive term, i.e. (zl,yl)subscript𝑧𝑙subscript𝑦𝑙(z\_{l},y\_{l}) in Figure. [1](#S4.F1 "Figure 1 ‣ 4.1 Semi-Self-Supervised Learning for Tabular Data ‣ 4 Method ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain"). We define the contrastive loss term to encourage samples created from pairs of the same class to have high similarity. It is common practice to introduce a separate predictive model to map the latent representations to an embedding space via a projection network hp​r​o​jsuperscriptℎ𝑝𝑟𝑜𝑗h^{proj} where the contrastive loss term is defined. We use supervised contrastive loss [[12](#bib.bib12)] for the labelled set 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L} as our augmentation views are within a class. It generalizes Eqn. [2](#S3.E2 "Equation 2 ‣ 3.1 Self-Supervised Learning ‣ 3 Preliminaries ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain") to an arbitrary number of positive samples, due to the presence of labels and examples belonging to the same class are encouraged to have high similarity, making the loss term more sample efficient.

|  |  |  |  |
| --- | --- | --- | --- |
|  | lτs​u​p=∑i∈ℐ−1P​(i)​∑p∈P​(i)log​(exp​(sim​(hip​r​o​j,hpp​r​o​j)/τ)∑n∈N​e​(i)exp​(sim​(hip​r​o​j,hnp​r​o​j)/τ))subscriptsuperscript𝑙𝑠𝑢𝑝𝜏subscript𝑖ℐ1𝑃𝑖subscript𝑝𝑃𝑖logexpsimsubscriptsuperscriptℎ𝑝𝑟𝑜𝑗𝑖subscriptsuperscriptℎ𝑝𝑟𝑜𝑗𝑝𝜏subscript𝑛𝑁𝑒𝑖expsimsubscriptsuperscriptℎ𝑝𝑟𝑜𝑗𝑖subscriptsuperscriptℎ𝑝𝑟𝑜𝑗𝑛𝜏l^{sup}\_{\tau}=\sum\_{i\in\mathcal{I}}\frac{-1}{P(i)}\sum\_{p\in P(i)}\text{log}\Big{(}\frac{\text{exp}(\text{sim}(h^{proj}\_{i},h^{proj}\_{p})/\tau)}{\sum\_{n\in Ne(i)}\text{exp}(\text{sim}(h^{proj}\_{i},h^{proj}\_{n})/\tau)}\Big{)} |  | (6) |

In the above, P​(i)={p|p∈𝒜​(i),yi=y~p}𝑃𝑖conditional-set𝑝formulae-sequence𝑝𝒜𝑖subscript𝑦𝑖subscript~𝑦𝑝P(i)=\{p|p\in\mathcal{A}(i),y\_{i}=\tilde{y}\_{p}\} is the set of indices of positives with the same label as example i𝑖i, |P​(i)|𝑃𝑖|P(i)| is its cardinality, and N​e​(i)={n|n∈ℐ,yi≠yn}𝑁𝑒𝑖conditional-set𝑛formulae-sequence𝑛ℐsubscript𝑦𝑖subscript𝑦𝑛Ne(i)=\{n|n\in\mathcal{I},y\_{i}\neq y\_{n}\}. This objective function will encourage mixed-uped labeled samples and anchors of the same sample to be close leading to a better cluster-able representation. In addition to the above loss term the encoder is trained to minimize the feature reconstruction loss via a decoder fθ​(⋅)subscript𝑓𝜃⋅f\_{\theta}(\cdot)

|  |  |  |  |
| --- | --- | --- | --- |
|  | lr​(xi)=|C|d​∑c|C|‖fθ∘eϕ​(xi)c−xic‖22+|D|d​∑j|D|∑odDj𝟏​[xid=o]​log⁡(fθ∘eϕ​(xi)o)subscript𝑙𝑟subscript𝑥𝑖𝐶𝑑superscriptsubscript𝑐𝐶superscriptsubscriptnormsubscript𝑓𝜃subscript𝑒italic-ϕsuperscriptsubscript𝑥𝑖𝑐superscriptsubscript𝑥𝑖𝑐22𝐷𝑑superscriptsubscript𝑗𝐷superscriptsubscript𝑜subscript𝑑subscript𝐷𝑗1delimited-[]superscriptsubscript𝑥𝑖𝑑𝑜subscript𝑓𝜃subscript𝑒italic-ϕsuperscriptsubscript𝑥𝑖𝑜l\_{r}(x\_{i})=\frac{|C|}{d}\sum\_{c}^{|C|}||f\_{\theta}\circ e\_{\phi}(x\_{i})^{c}-x\_{i}^{c}||\_{2}^{2}+\frac{|D|}{d}{\sum\_{j}^{|D|}\sum\_{o}^{d\_{D\_{j}}}{\mathbf{1}[x\_{i}^{d}=o]\log(f\_{\theta}\circ e\_{\phi}(x\_{i})^{o})}} |  | (7) |

The semi-self supervised objective function can then be written as

|  |  |  |  |
| --- | --- | --- | --- |
|  | L=𝔼(x,y)∼𝒟L​[lτs​u​p​(y,f​(x))]+β​𝔼x∼𝒟U∪𝒟L​[lr​(x)]𝐿subscript𝔼similar-to𝑥𝑦subscript𝒟𝐿delimited-[]superscriptsubscript𝑙𝜏𝑠𝑢𝑝𝑦𝑓𝑥𝛽subscript𝔼similar-to𝑥subscript𝒟𝑈subscript𝒟𝐿delimited-[]subscript𝑙𝑟𝑥L=\mathbb{E}\_{(x,y)\sim\mathcal{D}\_{L}}\big{[}l\_{\tau}^{sup}(y,f(x))\big{]}+\beta\mathbb{E}\_{x\sim\mathcal{D}\_{U}\cup\mathcal{D}\_{L}}\big{[}l\_{r}(x)\big{]} |  | (8) |

The encoder is trained using this loss term over K𝐾K epochs, to warm-start the representations in the latent space prior to pseudo-labeling and leveraging the unlabeled samples.

### 4.2 Psuedo-labeling Unlabeled Samples

Thus far, we have only used the labelled set 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L} in the contrastive loss term lτs​u​psubscriptsuperscript𝑙𝑠𝑢𝑝𝜏l^{sup}\_{\tau}. To make use of the unlabeled set using 𝒟Usubscript𝒟𝑈\mathcal{D}\_{U} we proposed to use label propagation [[11](#bib.bib11), [28](#bib.bib28)] after K𝐾K epochs of training with the supervised contrastive loss term Ls​u​psuperscript𝐿𝑠𝑢𝑝L^{sup}. Given the encoder trained on 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L} for K𝐾K epochs, we map the small labelled set 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L}, and a subset of the unlabeled set SU⊂𝒟Usubscript𝑆𝑈subscript𝒟𝑈S\_{U}\subset\mathcal{D}\_{U} to the latent space z𝑧z and construct an affinity matrix G𝐺G

|  |  |  |  |
| --- | --- | --- | --- |
|  | gi​j:={sim​(zi,zj)if​i≠j​and​zj∈NNk​(i)0otherwiseassignsubscript𝑔𝑖𝑗casessimsubscript𝑧𝑖subscript𝑧𝑗if𝑖𝑗andsubscript𝑧𝑗subscriptNN𝑘𝑖0otherwiseg\_{ij}:=\begin{cases}\text{sim}(z\_{i},z\_{j})~{}&\text{if}~{}i\neq j~{}\text{and}~{}z\_{j}\in\text{NN}\_{k}(i)\\ 0&\text{otherwise}\end{cases} |  | (9) |

where NNk​(i)subscriptNN𝑘𝑖\text{NN}\_{k}(i) is the k𝑘k nearest neighbor of sample zisubscript𝑧𝑖z\_{i} and sim​(⋅,⋅)​ℜ+sim⋅⋅superscript\text{sim}(\cdot,\cdot)\Re^{+} is a similarity measure, e.g. ziT​zjsuperscriptsubscript𝑧𝑖𝑇subscript𝑧𝑗z\_{i}^{T}z\_{j}. We then obtain pseudolabels for our unlabeled samples by computing the diffusion matrix C𝐶C and setting y~i:=arg​maxj⁡ci​jassignsubscript~𝑦𝑖argsubscript𝑗subscript𝑐𝑖𝑗\tilde{y}\_{i}:=\text{arg}\max\limits\_{j}c\_{ij}, where

|  |  |  |
| --- | --- | --- |
|  | (I−α​𝒜)​C=Y𝐼𝛼𝒜𝐶𝑌(I-\alpha\mathcal{A})C=Y |  |

Similar to [[11](#bib.bib11), [29](#bib.bib29)] we use conjugate method to solve linear equations to obtain C𝐶C to enable efficient computation of the pseudo-labels. Here 𝒜=D−1/2​W​D−1/2𝒜superscript𝐷12𝑊superscript𝐷12\mathcal{A}=D^{-1/2}WD^{-1/2} is the adjacency matrix, W=GT+G𝑊superscript𝐺𝑇𝐺W=G^{T}+G and D:=diag​(W​1n)assign𝐷diag𝑊subscript1𝑛D:=\text{diag}(W1\_{n}) is the degree matrix. Once we’ve obtained the pseudo-labels for the unlabeled subset SUsubscript𝑆𝑈S\_{U}, we train the encoder with unlabeled samples treating the generated labels as ground truth

|  |  |  |  |
| --- | --- | --- | --- |
|  | L=𝔼(x,y)∼𝒟L​[ls​u​p​(y,f​(x))]+γ​𝔼(x,yp​s)∼SU)​[ls​u​p​(yp​s,f​(x))]+β​𝔼x∼𝒟U​[lr​(x)]L=\mathbb{E}\_{(x,y)\sim\mathcal{D}\_{L}}\big{[}l^{sup}(y,f(x))\big{]}+\gamma\mathbb{E}\_{(x,y\_{ps})\sim S\_{U})}\big{[}l^{sup}(y\_{ps},f(x))\big{]}+\beta\mathbb{E}\_{x\sim\mathcal{D}\_{U}}\big{[}l\_{r}(x)\big{]} |  | (10) |

The pseudo-labels are updated every f𝑓f epoch of training with the above loss term.

### 4.3 Predictor

Following the semi-supervised pre-training, the encoder is transferred to the downstream task along with the generated pseudo-labels to train the predictor on the downstream task. We leverage Mixup augmentation [[26](#bib.bib26)] in the latent space and feed samples to a set of fully connected layers as depicted in Figure [2](#S4.F2 "Figure 2 ‣ 4.3 Predictor ‣ 4 Method ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain").

!(/html/2108.12296/assets/x2.png)

Figure 2: Overview transfering the semi-supervised pre-training steps to the downstream task. Encoder e​(x¯)𝑒¯𝑥e(\bar{x}) is fixed and the predictor - multilayer perceptron (MLP) is trained using Mixup augmentation. lc​exsuperscriptsubscript𝑙𝑐𝑒𝑥l\_{ce}^{x} is the generic cross-entropy loss split into supervised (sup) for labeled subset and unsupervised (unsup) for the unlabeled subset.

## 5 Experiment & Emperical Results

In this section we showcase the proposed framework on a set of different tabular datasets and application domains to demonstrate its effectiveness. We compare our semi-supervised framework with VIME [[25](#bib.bib25)] another semi-supervised approach for the same problem set as a benchmark. To evaluate the pre-training phase, we compare with auto-encoder. We also compare with other semi-supervised method manifold Mixup [[23](#bib.bib23)]. As a baseline, we include supervised methods, Logistic Regression, a 2-layer multi-layer perceptron network (MLP) that is used as the same architecture amongst other deep methods as the predictor network, and we also include CatBoost [[19](#bib.bib19)] as a gradient boosting tree method widely used on tabular data as it supports categorical columns. Additionally, we provide results for including various components of the proposed framework as ablation for the usefulness of each part of the method. In the experiments, self/semi-supervised use the labelled and unlabeled sets 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L}, 𝒟Usubscript𝒟𝑈\mathcal{D}\_{U} during training, and supervised models only used the labelled sets 𝒟Lsubscript𝒟𝐿\mathcal{D}\_{L}. We normalize the continuous columns to 0, 1 using Standard-scaler333[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](sklearn.preprocessing.StandardScaler). We provide more details on the experimental setup in the Supplementary Material. The implementation of ContrastiveMixup can be found at [https://github.com/sajaddarabi/ContrastiveMixup](https://github.com/<anonmous>/ContrastiveMixup)

### 5.1 Public Tabular Datasets

We compare the proposed method on four public UCI444<https://archive.ics.uci.edu/ml/datasets.php> datasets: MNIST, where examples are interpreted as 784-dimensional feature vectors, UCI Adult, UCI Covertype, more details, are available in the supplementary. We use 10%percent1010\% of the data as labelled and the rest as unlabeled; if the dataset contains an original test set, we use this to evaluate the methods; otherwise, we split the dataset 80%percent8080\% train and 20%percent2020\% test, and the ratios mentioned above follow. As we introduced embedding layers for categorical columns in our method, we choose the best of one-hot encoding categorical columns or embedding layers for other methods. The different variants of our methods for the ablation study are as follows:

* •

  Supervised: the pre-training is removed and only the predictor is trained (i.e. the same as MLP)
* •

  Self-SL only: the pre-training consisting of labeled contrastive loss term and unsupervised reconstruction loss without pseudo-labeling. (i.e. γ=0)\gamma=0)
* •

  Self-SL + PL: This is the pre-training with pseudo-labeling component added, without Mixup component when training the predictor.

Table 1:  Comparison on public tabular datasets.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | Dataset | | | |
| Type | Method | MNIST | Adult | Blog Feedback | Covertype |
| Supervised | Logistic | 90.12 (±0.098plus-or-minus0.098\pm 0.098) | 82.41 (±0.413plus-or-minus0.413\pm 0.413) | 78.91 (±0.22plus-or-minus0.22\pm 0.22) | 70.54 (±0.087plus-or-minus0.087\pm 0.087) |
| MLP | 93.69 (±0.234plus-or-minus0.234\pm 0.234) | 83.19 (±0.663plus-or-minus0.663\pm 0.663) | 79.63 (±0.519plus-or-minus0.519\pm 0.519) | 75.95 (±0.202plus-or-minus0.202\pm 0.202) |
| CatBoost (100%percent100100\%) | 97.41 (±0.098plus-or-minus0.098\pm 0.098) | 87.54 (±0.075plus-or-minus0.075\pm 0.075) | 85.08 (±0.088plus-or-minus0.088\pm 0.088) | 88.64 (±0.077plus-or-minus0.077\pm 0.077) |
| Semi-supervised | AE | 94.72 (±0.127plus-or-minus0.127\pm 0.127) | 84.18 (±0.078plus-or-minus0.078\pm 0.078) | 80.09 (±0.199plus-or-minus0.199\pm 0.199) | 79.67 (±0.296plus-or-minus0.296\pm 0.296) |
| Manifold Mixup | 94.92 (±0.012plus-or-minus0.012\pm 0.012) | 84.68 (±0.279plus-or-minus0.279\pm 0.279) | 80.24 (±0.652plus-or-minus0.652\pm 0.652) | 78.79 (±0.135plus-or-minus0.135\pm 0.135) |
| VIME | 95.71 (±0.013plus-or-minus0.013\pm 0.013) | 84.54 (±0.408plus-or-minus0.408\pm 0.408) | 81.36 (±0.301plus-or-minus0.301\pm 0.301) | 79.02 (±0.329plus-or-minus0.329\pm 0.329) |
| Ours (Ablation) | Supervised | 93.69 (±0.234plus-or-minus0.234\pm 0.234) | 83.19 (±0.663plus-or-minus0.663\pm 0.663) | 79.63 (±0.519plus-or-minus0.519\pm 0.519) | 75.95 (±0.202plus-or-minus0.202\pm 0.202) |
| Self-SL | 95.82 (±0.131plus-or-minus0.131\pm 0.131) | 85.16 (±0.249plus-or-minus0.249\pm 0.249) | 81.38 (±0.373plus-or-minus0.373\pm 0.373) | 79.46 (±0.463plus-or-minus0.463\pm 0.463) |
| Self-SL+PL | 97.01 (±0.066plus-or-minus0.066\pm 0.066) | 85.26 (±0.207plus-or-minus0.207\pm 0.207) | 81.65 (±0.370plus-or-minus0.370\pm 0.370) | 79.92 (±0.682plus-or-minus0.682\pm 0.682) |
| Ours | 97.58 (±0.078plus-or-minus0.078\pm 0.078) | 85.42 (±0.210plus-or-minus0.210\pm 0.210) | 81.88 (±0.123plus-or-minus0.123\pm 0.123) | 80.41 (±0.205plus-or-minus0.205\pm 0.205) |

From Table [1](#S5.T1 "Table 1 ‣ 5.1 Public Tabular Datasets ‣ 5 Experiment & Emperical Results ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain") we can see Contrastive Mixup consistently outperforms previous methods. Further, through our ablation we demonstrate the effectiveness of various components of our framework each provide benefit in improving downstream task performance. Pseudo-labeling consistently helps in improving performance, comparing Self-SL versus Self-SL+PL allbeit at varying degree for different datasets. As a reference Catboost trained on 100%percent100100\% of the labeled samples is provided as well. Our results on 10%percent1010\% labeled MNIST with the help of pseudo-labeling outperforms this reference point.

### 5.2 Genomics Datasets

We assessed the accuracy of our method on the UK Biobank 555<http://www.ukbiobank.ac.uk> Application # 33127 genotypes consisting of around 500,000 individuals genotyped at around 10 millions SNPs. In this experiment, we restricted our analysis to SNPs with minor allele frequency larger than 1%percent11\%. Moreover, SNPs that fail the Hardy-Weinberg test at significance threshold 10−7superscript10710^{-7} were removed. Our analysis is restricted to around 300,000 unrelated self-reported British white ancestry individuals. We selected four complex traits measured in UK Biobank. As including all of the SNPs in our analysis is computationally expensive, we therefore, for every trait select around 1000 significantly associated SNPs with smallest p-value based on a publicly available summary statistics666<https://alkesgroup.broadinstitute.org/UKBB/>. Note that the set of selected SNPs is different across traits after p-value filtering as each trait is associated with different genetic variants, hence every phenotype task could be considered a different dataset.

To explore the efficacy of our semi-supervised framework on limited labeled data sets in practical setting, we compared the accuracy of our method with state of the art methods by varying the number of labeled individuals and using the remaining individuals as unlabeled samples. The results on four phenotypes are shown in Figure [6](#A2.F6 "Figure 6 ‣ B.2 MNIST Limited Samples ‣ Appendix B Additional Experiments ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain"). From these figures we can see the semi-supervised methods outperform logistic regression model for cases where we only have access to a few thousand labeled samples. For two out of the four phenotypes logistic regression model outperforms the deep supervised models when adequate labeled samples are available i.e. >104absentsuperscript104>10^{4}. This may be due to only a subset of features being selected using p𝑝p-value threshold and hence making deeper models prone to overfititng.

|  |  |
| --- | --- |
| Refer to caption | Refer to caption |
| Refer to caption | Refer to caption |

Figure 3: Accuracy performance on four UK Biobank phenotypes across different number of labeled samples used as training set and the test set is fixed across experiments. The x axes is plotted in log-scale.

## 6 Conclusion

Tabular data presents a different challenge compared to images and text, as similar structure or semantics aren’t present, hence mitigating the transfer-ability of methods from those domain to the tabular domain. As a result extending semi-supervised methods that work well in those domains is more challenging for the tabular domain. Additionally, as most of the literature revolves around images and text not many pre-text tasks, and transformations have been investigated for such unstructured datasets where "correlations" or semantic meanings aren’t immediately present in the data. Instead, we propose a framework for extending the recent contrastive learning paradigm to the tabular domain and help propel it’s success in this domain as well. We do this by mapping samples to the latent space and creating new examples interpolating between samples in this space. We empirically show the effectiveness of the proposed method, and demonstrate how it improves learning from tabular data with limited labels. Further, improvements on pre-text tasks or augmentation methods for tabular datasets will dramatically improve the applicability of deep learning for these data modalities.

## 7 Broader Impact

Tabular data is very common in wide array of applications ranging from financial institutions, insurance companies, to health and clinical settings. These datasets contain both categorical and continuous features, such as demographic information, or real valued time series in finance datasets. As Deep Learning has shown great success in different data modalities such as text and images, by leveraging various pre-text tasks and domain specific augmentations for training in limited annotated data settings, by extending this over to the tabular domain many real-world applications will benefit and the applicability of Deep Learning will be greatly extended. The proposed method takes a step in this direction.

## References

* Arik and Pfister [2019]

  Sercan O Arik and Tomas Pfister.
  Tabnet: Attentive interpretable tabular learning.
  *arXiv preprint arXiv:1908.07442*, 2019.
* Chapelle et al. [2009]

  Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien.
  Semi-supervised learning (chapelle, o. et al., eds.; 2006)[book
  reviews].
  *IEEE Transactions on Neural Networks*, 20(3):542–542, 2009.
* Chen et al. [2020]

  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
  A simple framework for contrastive learning of visual
  representations, 2020.
* Devlin et al. [2018]

  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  Bert: Pre-training of deep bidirectional transformers for language
  understanding.
  *arXiv preprint arXiv:1810.04805*, 2018.
* Doersch et al. [2015]

  Carl Doersch, Abhinav Gupta, and Alexei A Efros.
  Unsupervised visual representation learning by context prediction.
  In *Proceedings of the IEEE international conference on computer
  vision*, pages 1422–1430, 2015.
* Douze et al. [2018]

  Matthijs Douze, Arthur Szlam, Bharath Hariharan, and Hervé Jégou.
  Low-shot learning with large-scale diffusion.
  In *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, pages 3349–3358, 2018.
* Gidaris et al. [2018]

  Spyros Gidaris, Praveer Singh, and Nikos Komodakis.
  Unsupervised representation learning by predicting image rotations.
  *arXiv preprint arXiv:1803.07728*, 2018.
* Grandvalet et al. [2005]

  Yves Grandvalet, Yoshua Bengio, et al.
  Semi-supervised learning by entropy minimization.
  In *CAP*, pages 281–296, 2005.
* He et al. [2020]

  Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.
  Momentum contrast for unsupervised visual representation learning.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pages 9729–9738, 2020.
* [10]

  Dong hyun Lee.
  Pseudo-label: The simple and efficient semi-supervised learning
  method for deep neural networks.
* Iscen et al. [2019]

  Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum.
  Label propagation for deep semi-supervised learning.
  In *Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition*, pages 5070–5079, 2019.
* Khosla et al. [2020]

  Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip
  Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan.
  Supervised contrastive learning.
  *arXiv preprint arXiv:2004.11362*, 2020.
* Logeswaran and Lee [2018]

  Lajanugen Logeswaran and Honglak Lee.
  An efficient framework for learning sentence representations.
  *arXiv preprint arXiv:1803.02893*, 2018.
* Mikolov et al. [2013]

  Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
  Distributed representations of words and phrases and their
  compositionality.
  *arXiv preprint arXiv:1310.4546*, 2013.
* Mnih and Hinton [2008]

  Andriy Mnih and Geoffrey E Hinton.
  A scalable hierarchical distributed language model.
  *Advances in neural information processing systems*,
  21:1081–1088, 2008.
* Noroozi and Favaro [2016]

  Mehdi Noroozi and Paolo Favaro.
  Unsupervised learning of visual representations by solving jigsaw
  puzzles.
  In *European conference on computer vision*, pages 69–84.
  Springer, 2016.
* Oord et al. [2018]

  Aaron van den Oord, Yazhe Li, and Oriol Vinyals.
  Representation learning with contrastive predictive coding.
  *arXiv preprint arXiv:1807.03748*, 2018.
* Pathak et al. [2016]

  Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A
  Efros.
  Context encoders: Feature learning by inpainting.
  In *Proceedings of the IEEE conference on computer vision and
  pattern recognition*, pages 2536–2544, 2016.
* Prokhorenkova et al. [2017]

  Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush,
  and Andrey Gulin.
  Catboost: unbiased boosting with categorical features.
  *arXiv preprint arXiv:1706.09516*, 2017.
* Sajjadi et al. [2016]

  Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen.
  Regularization with stochastic transformations and perturbations for
  deep semi-supervised learning.
  *arXiv preprint arXiv:1606.04586*, 2016.
* Sohn [2016]

  Kihyuk Sohn.
  Improved deep metric learning with multi-class n-pair loss objective.
  In *Proceedings of the 30th International Conference on Neural
  Information Processing Systems*, pages 1857–1865, 2016.
* Tarvainen and Valpola [2017]

  Antti Tarvainen and Harri Valpola.
  Mean teachers are better role models: Weight-averaged consistency
  targets improve semi-supervised deep learning results.
  *arXiv preprint arXiv:1703.01780*, 2017.
* Verma et al. [2019]

  Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas,
  David Lopez-Paz, and Yoshua Bengio.
  Manifold mixup: Better representations by interpolating hidden
  states.
  In *International Conference on Machine Learning*, pages
  6438–6447. PMLR, 2019.
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
* Zhang et al. [2017]

  Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz.
  mixup: Beyond empirical risk minimization.
  *arXiv preprint arXiv:1710.09412*, 2017.
* Zhang et al. [2016]

  Richard Zhang, Phillip Isola, and Alexei A Efros.
  Colorful image colorization.
  In *European conference on computer vision*, pages 649–666.
  Springer, 2016.
* Zhou et al. [2004]

  Dengyong Zhou, Olivier Bousquet, Thomas N Lal, Jason Weston, and Bernhard
  Schölkopf.
  Learning with local and global consistency.
  In *Advances in neural information processing systems*, pages
  321–328, 2004.
* Zhu et al. [2005]

  Xiaojin Zhu, John Lafferty, and Ronald Rosenfeld.
  *Semi-supervised learning with graphs*.
  PhD thesis, Carnegie Mellon University, language technologies
  institute, school of …, 2005.

Supplementary Material - Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain

## Appendix A Contrastive Mixup

The proposed pre-training is summarized in Algorithm. [1](#alg1 "Algorithm 1 ‣ Appendix A Contrastive Mixup ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain"). The encoder is initially trained by using the labeled subset for the contrastive loss component Eqn. 6, and both unlabeled & labeled subsets for the reconstruction loss Eqn.7. Subsequently, after K𝐾K epochs via label propagation we generate pseudo labels for the unlabeled subset so they can be leveraged in the contrastive loss as well.

Algorithm 1  the proposed method’s main algorithm.

Input: constant τ𝜏\tau, encoder e𝑒e, decoder f𝑓f, projection network hpsuperscriptℎ𝑝h^{p}, labeled set (xl,yl)subscript𝑥𝑙subscript𝑦𝑙(x\_{l},y\_{l}), unlabeled set with pseudolabels if available (xu,yu)subscript𝑥𝑢subscript𝑦𝑢(x\_{u},y\_{u})

1:for sampled mini-batches BL,Bu

subscript𝐵𝐿subscript𝐵𝑢B\_{L},B\_{u} from {(xl,yl)}l=1Nl,{(xu,yu)}u=1NU​do

superscriptsubscriptsubscript𝑥𝑙subscript𝑦𝑙𝑙1subscript𝑁𝑙superscriptsubscriptsubscript𝑥𝑢subscript𝑦𝑢𝑢1subscript𝑁𝑈do\{(x\_{l},y\_{l})\}\_{l=1}^{N\_{l}},\{(x\_{u},y\_{u})\}\_{u=1}^{N\_{U}}\ \textbf{do}

2:     draw λ∼Uniform​(0,α)similar-to𝜆Uniform0𝛼\lambda\sim\text{Uniform}(0,\alpha) of size Bl+Busubscript𝐵𝑙subscript𝐵𝑢B\_{l}+B\_{u} ▷▷\triangleright α∈[0.0,0.5]𝛼0.00.5\alpha\in[0.0,0.5]

3:     draw random integer i∈[0,ℐ]𝑖0ℐi\in[0,\mathcal{I}]
▷▷\triangleright ℐ:=assignℐabsent\mathcal{I}:= number of layers in encoder

4:     hli=e0:i​(xl)superscriptsubscriptℎ𝑙𝑖subscript𝑒:0𝑖subscript𝑥𝑙h\_{l}^{i}=e\_{0:i}(x\_{l}),    hui=e0:i​(xl)superscriptsubscriptℎ𝑢𝑖subscript𝑒:0𝑖subscript𝑥𝑙h\_{u}^{i}=e\_{0:i}(x\_{l})
▷▷\triangleright Feed through 00 to it​hsuperscript𝑖𝑡ℎi^{th} layer of encoder

5:     hmixedi=superscriptsubscriptℎmixed𝑖absenth\_{\text{mixed}}^{i}= Mixup([hli;hui],[yl;yu],λ)

superscriptsubscriptℎ𝑙𝑖superscriptsubscriptℎ𝑢𝑖

subscript𝑦𝑙subscript𝑦𝑢
𝜆([h\_{l}^{i};h\_{u}^{i}],[y\_{l};y\_{u}],\lambda) ▷▷\triangleright Mix within the same label

6:     zl=ei:ℐ​(hli)subscript𝑧𝑙subscript𝑒:𝑖ℐsuperscriptsubscriptℎ𝑙𝑖z\_{l}=e\_{i:\mathcal{I}}(h\_{l}^{i})

7:     zu=ei:ℐ​(hui)subscript𝑧𝑢subscript𝑒:𝑖ℐsuperscriptsubscriptℎ𝑢𝑖z\_{u}=e\_{i:\mathcal{I}}(h\_{u}^{i})

8:     zmixed=ei:ℐ​(hmixedi)subscript𝑧mixedsubscript𝑒:𝑖ℐsuperscriptsubscriptℎmixed𝑖z\_{\text{mixed}}=e\_{i:\mathcal{I}}(h\_{\text{mixed}}^{i})

9:     lrecon=lr​([xL;xU])subscript𝑙reconsubscript𝑙𝑟

subscript𝑥𝐿subscript𝑥𝑈l\_{\text{recon}}=l\_{r}([x\_{L};x\_{U}]) ▷▷\triangleright Eqn. 7

10:     hlp=hp​(zl)superscriptsubscriptℎ𝑙𝑝superscriptℎ𝑝subscript𝑧𝑙h\_{l}^{p}=h^{p}(z\_{l})

11:     hup=hp​(zu)superscriptsubscriptℎ𝑢𝑝superscriptℎ𝑝subscript𝑧𝑢h\_{u}^{p}=h^{p}(z\_{u})

12:     hmixedp=hp​(zmixed)superscriptsubscriptℎmixed𝑝superscriptℎ𝑝subscript𝑧mixedh\_{\text{mixed}}^{p}=h^{p}(z\_{\text{mixed}})

13:     lcontrastive=lτs​u​p​([hlp;hup],hmixedp)subscript𝑙contrastivesubscriptsuperscript𝑙𝑠𝑢𝑝𝜏

superscriptsubscriptℎ𝑙𝑝superscriptsubscriptℎ𝑢𝑝
superscriptsubscriptℎmixed𝑝l\_{\text{contrastive}}=l^{sup}\_{\tau}([h\_{l}^{p};h\_{u}^{p}],h\_{\text{mixed}}^{p}) ▷▷\triangleright ls​u​p​(v​i​e​w​1,v​i​e​w​2)superscript𝑙𝑠𝑢𝑝𝑣𝑖𝑒𝑤1𝑣𝑖𝑒𝑤2l^{sup}(view1,view2) Eqn. 6

14:     ℒ=lcontrastive+lr​e​c​o​nℒsubscript𝑙contrastivesubscript𝑙𝑟𝑒𝑐𝑜𝑛\mathcal{L}=l\_{\text{contrastive}}+l\_{recon}

15:     Update networks e,f,

𝑒𝑓e,f, and hpsuperscriptℎ𝑝h^{p} to minimize ℒℒ\mathcal{L}

### A.1 Limitations

Underlying our method, we make use of the Manifold Assumption where high dimensional data lies (roughly) on a lower-dimensional manifold to avoid creating low probable samples through interpolation in the original data space. Further, as the data manifold may change for different labelled examples, in our method, we enforce mixing within the same labelled class, limiting the set of labelled samples that are initially used in the contrastive component. To leverage the unlabeled subsets, we generate pseudo-labels for the unlabeled samples to be used in the contrastive loss component. This makes the method more reliable on the quality of the initially labelled subset of examples. Additionally, as the method relies on discrete labels, in its current presentation cannot be applied to regression tasks.

## Appendix B Additional Experiments

### B.1 Mixing within class vs randomly

We compare limiting Mixup augmentation in the latent space to same labeled samples versus the original Mixup augmentation where any random set of samples can be used to interpolate in between to generate new samples. When randomly interpolating between samples, h~=λ​h1+(1−λ)​h2~ℎ𝜆subscriptℎ11𝜆subscriptℎ2\tilde{h}=\lambda h\_{1}+(1-\lambda)h\_{2} we enforce h1subscriptℎ1h\_{1} to be λ𝜆\lambda close to h~~ℎ\tilde{h} and h2subscriptℎ2h\_{2} 1−λ1𝜆1-\lambda close to h~~ℎ\tilde{h} in the contrastive term. In this experiment the Mixup component for the contrastive loss was changed and the rest is untouched. As can be seen from Table. [2](#A2.T2 "Table 2 ‣ B.1 Mixing within class vs randomly ‣ Appendix B Additional Experiments ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain") mixing between random samples in the latent space can hurt performance compared to mixing within the same labeled class.

Table 2:  Comparison on between randomly interpolating between examples and interpolating within the same labeled class examples.

|  |  | Dataset | | | |
| --- | --- | --- | --- | --- | --- |
| Type | Method | MNIST | Adult | Blog Feedback | Covertype |
| Ours (Ablation) | Random Mixing | 96.60 (±0.111plus-or-minus0.111\pm 0.111) | 84.33 (±0.472plus-or-minus0.472\pm 0.472) | 81.72 (±0.389plus-or-minus0.389\pm 0.389) | 79.71 (±0.229plus-or-minus0.229\pm 0.229) |
|  | Ours | 97.58 (±0.078plus-or-minus0.078\pm 0.078) | 85.42 (±0.210plus-or-minus0.210\pm 0.210) | 81.88 (±0.123plus-or-minus0.123\pm 0.123) | 80.41 (±0.205plus-or-minus0.205\pm 0.205) |

### B.2 MNIST Limited Samples

In Figure. [4](#A2.F4 "Figure 4 ‣ B.2 MNIST Limited Samples ‣ Appendix B Additional Experiments ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain") we run an additional experiment to demonstrate the effectiveness of the proposed method under limited number of labeled samples. The proposed framework consistently outperforms baselines.

!(/html/2108.12296/assets/x7.png)

Figure 4: Comparison of accuracy performance on MNIST under varying number labeled examples used for training.

In Figure. [5](#A2.F5 "Figure 5 ‣ B.2 MNIST Limited Samples ‣ Appendix B Additional Experiments ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain") we conduct an experiment to evaluate the pseudo-labeling accuracy as a function of number of labeled samples projected in the latent space.

!(/html/2108.12296/assets/x8.png)

Figure 5: MNIST pseudo-labeling accuracy across varying number of labeled samples. Accuracy is reported after training for 20 epochs.

!(/html/2108.12296/assets/x9.png)

Figure 6: TSNE visualization of representations extracted from encoder training on 10% labeled examples of MNIST.

## Appendix C Implementation Details

Our proposed framework consists of 5 different components: (1) encoder, (2) decoder, (3) Within Class Mixup + Contrastive Loss, (4) Label propagation (5) Predictor.

1. 1.

   Our encoder is a set of fully connected layers, where the number of hidden layers and the number of layers is a hyper-parameter.
2. 2.

   A weight hyperparameter β𝛽\beta is used to control the decoder loss term. The decoder has the same architecture as the encoder, but in reverse.
3. 3.

   The proposed within class Mixup has a hyperparameter α∈[0.0,0.5]𝛼0.00.5\alpha\in[0.0,0.5] that is used to sample λ𝜆\lambda from Uniform​(0,α)Uniform0𝛼\text{Uniform}(0,\alpha). The projection network is a set of fully connected layers with projection dimension dp​r​o​jsubscript𝑑𝑝𝑟𝑜𝑗d\_{proj}, and number of layers Lp​r​o​jsubscript𝐿𝑝𝑟𝑜𝑗L\_{proj}.
4. 4.

   The label propagation component contains hyperparameters k𝑘k for the number of nearest neighbors and α𝛼\alpha parameter. k=3𝑘3k=3 and α=0.999𝛼0.999\alpha=0.999 is used as the default across all experiments.
5. 5.

   The predictor is a set of fully connected layers. *(FC-BN-ReLU)*.

Experiments were run on two sets of datasets. On public datases we set the hyperparameters as follows:

1. 1.

   Hidden layer dimension size set to be the same as d𝑑d after embedding categorical columns. The embeddings generated for each categorical column is set to m​i​n​(600,r​o​u​n​d​(1.6∗d|𝒟i|0.56))𝑚𝑖𝑛600𝑟𝑜𝑢𝑛𝑑1.6superscriptsubscript𝑑subscript𝒟𝑖0.56min(600,round(1.6\*d\_{|\mathcal{D}\_{i}|}^{0.56})) and the number of layers is 111.
2. 2.

   Decoder reconstruction loss weight β=0.25𝛽0.25\beta=0.25
3. 3.

   a​l​p​h​a=0.2𝑎𝑙𝑝ℎ𝑎0.2alpha=0.2 for within class Mixup, Lp​r​o​j=1subscript𝐿𝑝𝑟𝑜𝑗1L\_{proj}=1, and dp​r​o​j=dsubscript𝑑𝑝𝑟𝑜𝑗𝑑d\_{proj}=d
4. 4.

   default parameters
5. 5.

   Our predictor is a 222 layer MLP with hidden dimension size of 100100100. Mixup augmentation is set to default setting with α=1.0𝛼1.0\alpha=1.0 and λ∼Uniform​(0,α)similar-to𝜆Uniform0𝛼\lambda\sim\text{Uniform}(0,\alpha)

On the genomics dataset we tune hyperparameters using the [Neural Network Intelligence (NNI) Auto-Tuner](https://github.com/microsoft/nni) and pick the best hyper-parameter setting on validation.

## Appendix D Experimental Details

We evaluated the performance of our method on 4 public datasets and 4 phenotypes. In each of these datasets 10%percent1010\% of the samples in the training set was used as labeled and the reset as unlabeled. This labeled subset depends on the random seed in our implementation, and a total of 5 experiments was run for each dataset by varying the random seed [123,127,131,137,130]

123127131137130[123,127,131,137,130] and report the average following hyperparameter selection.

On public datasets, the difference between VIME and our proposed framework is purely on the training algorithm, and the same capacity networks are used through out i.e. in VIME 4 networks are used encoder, feature estimator, mask estimator, predictor, and these are the same as our encoder, decoder, projector, predictor respectively.

On genomics dataset, the different methods were tuned on validation using [Neural Network Intelligence (NNI) Auto-Tuner](https://github.com/microsoft/nni).

## Appendix E Dataset Details

### E.1 UK Biobank

| ID | Dataset | Categorical | Continuous | Num Samples |
| --- | --- | --- | --- | --- |
| 1 | MPV | 714 | 0 | 2913273 |
| 2 | Smoking Status | 714 | 0 | 2913273 |
| 3 | MSCV | 1950 | 0 | 2913273 |
| 4 | Hair Color | 1000 | 0 | 2913273 |

### E.2 Public Datasets

The public datasets used are summarized in Table [E.2](#A5.SS2 "E.2 Public Datasets ‣ Appendix E Dataset Details ‣ Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain"). Three of the datsets 1, 2, 3 contain separate test sets, and covtype (4) 20% of the data is used as test and the rest for train. For each dataset we use 10%percent1010\% of the train set as labeled and the rest as unlabeled.

| ID | Dataset | Categorical | Continuous | Num Samples |
| --- | --- | --- | --- | --- |
| 1 | [mnist](http://yann.lecun.com/exdb/mnist/) | 0 | 724 | 60000 |
| 2 | [adult](https://archive.ics.uci.edu/ml/datasets/census+income) | 8 | 6 | 48840 |
| 3 | [BlogFeedback](https://archive.ics.uci.edu/ml/datasets/BlogFeedback) | 213 | 67 | 60021 |
| 4 | [covtype](https://archive.ics.uci.edu/ml/datasets/covertype) | 44 | 10 | 581011 |

## Appendix F Software & Hardware

Experiments were run on a machine with a GeForce RTX 2080 TI, 128 Gb RAM, and Intel(R) Core(TM) i9-7920X CPU.

To ensure reproducibility, all experiments were run using the same set of random seeds, baseline methods are re-implemented in the same code base and the same software versions are used.

Table 3: Python dependencies.

| Dependency | Version |
| --- | --- |
| python | 3.6.1 |
| pytorch | 1.7.1 |
| numpy | 1.15.4 |
| pandas | 1.1.5 |
| scikit-learn | 0.24.2 |
| scipy | 1.6.3 |
| tqdm | 4.60 |
| matplotlib | 3.4.1 |
