---
arxiv: '2404.16030'
authors:
- Jiawei Ma
- Po-Yao Huang
- Saining Xie
- Shang-Wen Li
- Luke Zettlemoyer
- Shih-Fu Chang
- Wen-Tau Yih
- Hu Xu
parser: ar5iv
retrieved: '2026-05-08'
source: paper
title: 'MoDE: CLIP Data Experts via Clustering'
url: http://arxiv.org/abs/2404.16030v1
year: 2024
---

# MoDE: CLIP Data Experts via Clustering

Jiawei Ma1∗,2  Po-Yao Huang1  Saining Xie3  Shang-Wen Li1
  
Luke Zettlemoyer1,4  Shih-Fu Chang2  Wen-Tau Yih1  Hu Xu1+
  
1FAIR, Meta  2Columbia University  3New York University  4University of Washington

###### Abstract

The success of contrastive language-image pretraining (CLIP) relies on the supervision from the pairing between images and captions, which tends to be noisy in web-crawled data. We present Mixture of Data Experts (MoDE) and learn a system of CLIP data experts via clustering. Each data expert is trained on one data cluster, being less sensitive to false negative noises in other clusters. At inference time, we ensemble their outputs by applying weights determined through the correlation between task metadata and cluster conditions.
To estimate the correlation precisely, the samples in one cluster should be semantically similar, but the number of data experts should still be reasonable for training and inference. As such, we consider the ontology in human language and propose to use fine-grained cluster centers to represent each data expert at a coarse-grained level.
Experimental studies show that four CLIP data experts on ViT-B/16 outperform the ViT-L/14 by OpenAI CLIP and OpenCLIP on zero-shot image classification but with less (<<35%) training cost. Meanwhile, MoDE can train all data expert asynchronously and can flexibly include new data experts. The code is available at <https://github.com/facebookresearch/MetaCLIP/tree/main/mode>.

††footnotetext: ∗ Research done while Jiawei Ma was an intern at FAIR.††footnotetext: + Project Lead.

## 1 Introduction

Contrastive Language-Image Pretraining (CLIP) learns versatile vision-language representations which are transferable across diverse downstream tasks. Existing models, such as OpenAI CLIP [[39](#bib.bib39)], OpenCLIP [[44](#bib.bib44)] and MetaCLIP [[50](#bib.bib50)], are trained with a large collection of web-crawled image-caption pairs.
Specifically, for each image, its paired caption is viewed as a *positive* example, and the captions of all the other images are viewed as *negative*s.
The model then learns to project both images and captions into a shared space, where the embedding of the positive caption is drawn closer to the image embedding, compared to the embeddings of all the other negative captions.

!(/html/2404.16030/assets/x1.png)

Figure 1: For an image-caption pair, the caption may describe limited visual content or even be unrelated, and such noises unavoidably hurt the quality of negative examples to learning a single model.
We propose to uncover the clusters from training data, where 1) the pairs with similar images but different captions are assigned to different clusters and 2) the samples in each cluster are of related meanings, and learn a Data Expert for each cluster.
These experts are then selectively ensembled for inference.

The key to the success of contrastive vision-language representation learning lies in the creation of quality *negative* examples for training [[8](#bib.bib8), [14](#bib.bib14)].
A single image can be depicted by texts with different meanings (*i.e*., semantics), covering multiple details and interpretations, as illustrated in [Fig. 1](#S1.F1 "In 1 Introduction ‣ MoDE: CLIP Data Experts via Clustering").
Because the paired caption usually describes limited visual content, it is common to see that two similar images have drastically different textual descriptions, especially in noisy web-crawled data.
When those image-caption pairs are sampled in the same batch, captions of other images become *false negatives* — acceptable captions yet being treated as negative descriptions of the target image.
Conversely, if only dissimilar image-caption pairs are sampled, the contrastive learning problem becomes trivial. Incorporating *hard negatives* [[8](#bib.bib8), [49](#bib.bib49), [37](#bib.bib37)] (e.g., incorrect yet similar captions that share many words of a correct textual description) in training batches has often been shown to improve the model performance.

In this work, we introduce the Mixture of Data Experts (MoDE) framework (shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ MoDE: CLIP Data Experts via Clustering")-bottom) via clustering.
MoDE separates false negative samples into different clusters and groups the pairs with similar semantics, which mitigates noise from false-negative captions while incorporating a more challenging set of hard-negative examples, thereby enhancing vision-language pre-training.
MoDE consists of two main steps: (1) the training data (*i.e*., image-caption pairs) is first clustered into several disjoint subsets by the captions; each cluster is then used to train a model following the standard contrastive learning method. In this way, each model is specialized by the training data in one cluster and thus termed as a Data Expert.
(2) When applied to downstream tasks, such as image classification, the task metadata (*i.e*., class names), are first compared to the centroid of each data cluster to determine which data expert needs to be activated. Selected data experts are then used to create the embeddings of the test image and classes. The class with the highest ensembled similarity is then output as the classification result.

Empirically,
MoDE outperforms several state-of-the-art vision-language models when applied to multiple standard benchmarks, including +3.7% on image classification in CLIP benchmark [[39](#bib.bib39), [34](#bib.bib34)], +3.3% on image-to-text retrieval and +2.7% on text-to-image retrieval on COCO [[29](#bib.bib29)].
The superiority of MoDE can be attributed to better trained individual data expert models, due to the fact that examples in the same cluster, when used for contrastive learning, provide more quality negatives.
Because captions in the same cluster are different but semantically similar (*e.g*., “a cat climbs a tree”, “a tiger reaches up to a tree”), they become challenging negative examples when compared with images that are not the originally paired ones.
On the other hand, it is also less likely to encounter a false negative case where a very different caption validly describes the same image (*e.g*., “tree guards to stop the cats” in [Fig. 1](#S1.F1 "In 1 Introduction ‣ MoDE: CLIP Data Experts via Clustering")).
MoDE is also uniquely positioned for large-scale training when billions of image-caption pairs are available. As each data expert uses only a fraction of the whole dataset, it can be more easily trained with fewer compute resources asynchronously.
From experiments across different ViT [[6](#bib.bib6)] model scales, we show that four ViT-B/16 data experts can outperform the single ViT-L/14 model by OpenAI CLIP [[39](#bib.bib39)] and OpenCLIP [[43](#bib.bib43)] on image classification but requires much less (<<35%) training cost.
In summary, our contributions are:

* •

  We investigate the quality negative samples in contrastive language-image pretraining, and in particular, the noise of false negatives in web-crawled image-caption pairs.
* •

  We propose the MoDE framework to learn a system of CLIP data experts via clustering, and adaptively ensemble data experts for downstream tasks at inference time.
* •

  Extensive experimental study has demonstrated the effects in zero-shot transfer benchmarks with low training cost. MoDE can include new data experts flexibly and is thus beneficial for continual pre-training.

## 2 Related Work

#### Contrastive Language Image Pretraining (CLIP)

aims to learns robust & transferable visual representations from large-scale data.
Scaling up [[19](#bib.bib19), [38](#bib.bib38)] existing approaches and improving the effectiveness is critical.
Recent progress in the field involves the exploration of regularization techniques [[53](#bib.bib53)] and hyperbolic embedding methods [[4](#bib.bib4)] but they require significant effort for data annotation.
Data curation is then proposed to remove noisy web-crawled image-caption pairs.
Additionally, methods like image masking [[28](#bib.bib28)] and concise captions [[27](#bib.bib27)] efficiently decrease memory demands, enabling the use of larger batch sizes and model sizes.
However, a trade-off between training cost and effectiveness still exists.
Following the studies [[41](#bib.bib41), [23](#bib.bib23)] in contrastive learning [[16](#bib.bib16), [2](#bib.bib2)], recent work investigated negative samples in CLIP training but still focuses on image side [[48](#bib.bib48), [30](#bib.bib30)]. The noise exhibited in captions [[51](#bib.bib51)] is then overlooked.
In this study, we tackle the data noise and the discovery of negative samples via clustering. Rather than training a single model,
we asynchronously train multiple data experts and then directly ensemble them for inference adaptively, which also shows benefits for model scaling.

#### Mixture-of-Expert (MoE)

trains a set of sub-models and a routing module.
Originally, each expert is defined as an entire network [[18](#bib.bib18), [21](#bib.bib21)], and a single model is selected for each data adaptively.
As restricting to hard model selection may limit the practicality, deep mixture of expert  [[7](#bib.bib7)], is then proposed where the MoE layer is set to softly ensemble layer outputs via weighted sum,
which is then investigated with different architectures [[25](#bib.bib25), [9](#bib.bib9)] in various tasks [[40](#bib.bib40), [45](#bib.bib45)].
However, all expert models are still trained on the same data simultaneously, resulting in much heavier training costs.
Recently, BTM [[26](#bib.bib26), [13](#bib.bib13)] proposes to learn expert models on different document types (*e.g*., papers, posts) separately but is only validated on language models. Meanwhile, both MoE and BTM can only determine the model routing for each input separately. Instead, MoDE generalizes to task-level adaptation and ensembles the models by task metadata (*e.g*., class names in classification task [[3](#bib.bib3)]).

!(/html/2404.16030/assets/x2.png)

Figure 2: Framework of MoDE via clustering. (Left) We perform a two-step clustering on captions to decide clusters / conditions for data experts. The colored scatter plots are fine-grained clusters and the circles are clusters at coarse-grained level. (Right) Each coarse-grained cluster (c𝑐c) conditions the learning of one data expert f(⋅|c)f(\cdot|c) and all data experts (colored boxes) are learned asynchronously. For inference, the similarity between task metadata and fine-grained cluster centers ({s}𝑠\{s\}) is used to decide the routing of data experts. To keep reasonable training cost, all data experts can be initialized with a model partially trained on all data without clustering (omitted for simplicity).

#### Inference-Time Adaptation

adapts a pre-trained model quickly and effectively to new tasks. Initially, transductive learning [[10](#bib.bib10)] is studied and leverages all unlabeled test data for model update. To mitigate the dependence on the presumed distribution of test data, test-time training[[47](#bib.bib47), [42](#bib.bib42), [11](#bib.bib11)] is developed to generate individual models for each input.
Subsequent explorations into meta-learning [[46](#bib.bib46), [15](#bib.bib15), [31](#bib.bib31)] introduced a separate module (*i.e*., meta-learner) that can adapt the pre-trained model for each task with a few annotated examples. MoDE has inference-time task adaptation but without annotation or parameter update.

## 3 CLIP Data Experts

For contrastive image-language pre-training, the model is trained to accurately align each image with the captions describing the visual content.
In a manner of divide-and-conquer [[1](#bib.bib1)], for each CLIP data expert training on one cluster, we reduce the amount of false negatives and increase the hard negatives within each mini-batch. In this way, we mitigate noise exhibited in web-crawled image-caption pairs and make the model training more effective.

As shown in [Fig. 2](#S2.F2 "In Mixture-of-Expert (MoE) ‣ 2 Related Work ‣ MoDE: CLIP Data Experts via Clustering"), on top of the established CLIP training that learns a single dense CLIP model f​(⋅)𝑓⋅f(\cdot) ([Sec. 3.1](#S3.SS1 "3.1 Background: Vanilla CLIP Training ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering")),
we propose to learn a set of CLIP data experts {f(⋅|c)}\{f(\cdot|c)\} via unsupervised clustering ([Sec. 3.2](#S3.SS2 "3.2 Clustering ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering")) and each CLIP data expert f(⋅|c)f(\cdot|c) is trained on the cluster c𝑐c ([Sec. 3.3](#S3.SS3 "3.3 Data Experts Training ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering")). In this way, the conditioned data expert f(⋅|c)f(\cdot|c) is less sensitive to the noise from other clusters and can be effectively trained among the data of coherent semantics.
For each evaluation task, by measuring the correlation between the task metadata (*e.g*., class names) and the conditions, the outputs can be jointly decided by multiple data experts ([Sec. 3.4](#S3.SS4 "3.4 Inference Time Task-Adaptation ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering")).

### 3.1 Background: Vanilla CLIP Training

CLIP [[39](#bib.bib39)] learns separate vision and language encoders with a joint vision-language embedding space. By contrasting positive pairs from negative samples within the same batch, CLIP can accurately model the similarity of the image and caption in each pair. We denote CLIP as f​((𝐱v,𝐱l))𝑓subscript𝐱𝑣subscript𝐱𝑙f\big{(}(\mathbf{x}\_{v},\mathbf{x}\_{l})\big{)} for an image-caption input (𝐱v,𝐱l)subscript𝐱𝑣subscript𝐱𝑙(\mathbf{x}\_{v},\mathbf{x}\_{l}), and simplify CLIP model as f​(⋅)𝑓⋅f(\cdot).
As a reminder, instead of learning a single dense CLIP model f​(⋅)𝑓⋅f(\cdot), we propose to learn a set of CLIP data expert models independently given a set of conditions C𝐶C, *i.e*., {f(⋅|c)|c∈C}\{f(\cdot|c)|c\in C\}.

### 3.2 Clustering

This subsection discusses how to formulate conditions C𝐶C, and how to use clustering to automatically discover conditions for data experts from the pre-train set. In a nutshell, the desiderata for the conditions are twofold:
1) as each task at test time requires detailed description (*e.g*., recognize the “cat” species instead of just “animal”), the conditions should be *representative* such that the correlation with tasks can be precisely modeled for reliable data experts selection;
2) the number of conditions should be *reasonable* since each condition is used to learn one data expert.
As each condition is represented by a cluster, the ideals of *representative* likely ask for more fine-grained clustering whereas the latter may require for fewer data experts.

Instead, motivated by the ontology in human language, we propose to capture such a hierarchical structure via clustering, *i.e*., determine the condition of a data expert at the coarse-grained level and represent it via the set of fine-grained clusters.
For simplicity, we design a two-step K-means clustering.
We first employ fine-grained clustering to locate each cluster whose samples are of similar semantics, such that the fine-grained cluster centers are representative (Step 1),
and then group fine-grained clusters to determine coarse-grained clustering among data for data experts’ specialization (Step 2).
In this way, instead of using a single coarse-grained center, the condition is symbolized by the fine-grained cluster centers.
The features for clustering are extracted from captions and the details are studied in [Sec. 5](#S5 "5 Discussion ‣ MoDE: CLIP Data Experts via Clustering").

#### Step 1: Fine-grained Clustering.

As the amount of pre-train data 𝒟𝒟\mathcal{D} is huge (hundreds of millions to billions level for CLIP [[39](#bib.bib39)]), it could be inefficient to train K-means over all pre-training data. Instead, we first uniformly sample a subset from the pre-training set: 𝒟′∼𝒟similar-tosuperscript𝒟′𝒟\mathcal{D}^{\prime}\sim\mathcal{D} and |𝒟′|≪|𝒟|much-less-thansuperscript𝒟′𝒟|\mathcal{D}^{\prime}|\ll|\mathcal{D}|.
Then, we perform K-means training  [[33](#bib.bib33)] over 𝒟′superscript𝒟′\mathcal{D}^{\prime}:

|  |  |  |  |
| --- | --- | --- | --- |
|  | S←K-means​(𝒟′),←𝑆K-meanssuperscript𝒟′S\leftarrow\text{K-means}(\mathcal{D}^{\prime}), |  | (1) |

where S𝑆S is a set of learned cluster centers.
Note that the number of fine-grained clusters m=|S|𝑚𝑆m=|S| can be substantially large such that the cluster center of each cluster well represents coherent semantic information for each cluster.

Step 2: Coarse-grained Clustering.To efficiently allocate the training/inference of a data expert, we perform a second round, *i.e*., coarse-grained, K-means clustering on top of fine-grained cluster centers S𝑆S:

|  |  |  |  |
| --- | --- | --- | --- |
|  | C←K-means​(S),←𝐶K-means𝑆C\leftarrow\text{K-means}(S), |  | (2) |

where each coarse-grained cluster center c∈C𝑐𝐶c\in C is the condition for a data expert.
We denote n=|C|𝑛𝐶n=|C| as the number of data experts where n≪mmuch-less-than𝑛𝑚n\ll m, and Scsubscript𝑆𝑐S\_{c} as set of fine-grained clusters assigned to the data expert f(⋅|c)f(\cdot|c) where S=∪c∈CSc𝑆subscript𝑐𝐶subscript𝑆𝑐S=\cup\_{c\in C}S\_{c}.

### 3.3 Data Experts Training

Next, we formulate training data for each data expert.
We first
collect the data assigned for each fine-grained cluster s𝑠s:
𝒟s={d|s=arg​mins∈S⁡(‖𝐞d−𝐞s‖22)⁡ and ​d∈𝒟}subscript𝒟𝑠conditional-set𝑑𝑠subscriptargmin𝑠𝑆superscriptsubscriptnormsubscript𝐞𝑑subscript𝐞𝑠22 and 𝑑𝒟\mathcal{D}\_{s}=\{d|s=\operatorname\*{arg\,min}\_{s\in S}(\|\mathbf{e}\_{d}-\mathbf{e}\_{s}\|\_{2}^{2})\text{ and }d\in\mathcal{D}\},
where 𝐞dsubscript𝐞𝑑\mathbf{e}\_{d} and 𝐞ssubscript𝐞𝑠\mathbf{e}\_{s} are the embeddings for training example d𝑑d and fine-grained cluster center s𝑠s respectively.
To train a data expert f(⋅|c)f(\cdot|c), its corresponding CLIP training data is:

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝒟c=⋃s∈Sc𝒟s.subscript𝒟𝑐subscript𝑠subscript𝑆𝑐subscript𝒟𝑠\mathcal{D}\_{c}=\bigcup\nolimits\_{s\in S\_{c}}\mathcal{D}\_{s}. |  | (3) |

For convenience, we use MoDE-n to indicate the system with n𝑛n CLIP data experts.
For training efficiency, all data experts are specialized from the same seed CLIP model that is partially trained over the entire set 𝒟𝒟\mathcal{D}. Then, each data expert f(⋅|c)f(\cdot|c) is trained only on 𝒟csubscript𝒟𝑐\mathcal{D}\_{c}.

### 3.4 Inference Time Task-Adaptation

As our framework conditions the model expertise on clusters to train data experts,
it also gives multiple models to choose from during inference (instead of the only choice on a single CLIP model). This gives the room to adapt different data experts to various downstream tasks.

We propose a simple approach to adapt data experts (no parameter updates) to downstream tasks using the task metadata.
Intuitively,
this approach routes each downstream task adaptively and efficiently to data experts during inference.
For simplicity, we formulate the data experts routing as a weighted sum of data experts’ outputs. Formally, given an evaluation task 𝐓𝐓\mathbf{T}, the output of CLIP data experts is

|  |  |  |  |
| --- | --- | --- | --- |
|  | ∑c∈Cf(⋅|c)p(c|𝐓),\sum\nolimits\_{c\in C}f(\cdot|c)p(c|\mathbf{T}), |  | (4) |

where p​(c|𝐓)𝑝conditional𝑐𝐓p(c|\mathbf{T}) is the normalized weight for the data expert f(⋅|c)f(\cdot|c), *i.e*., ∑c∈Cp​(c|𝐓)=1subscript𝑐𝐶𝑝conditional𝑐𝐓1\sum\nolimits\_{c\in C}p(c|\mathbf{T})=1.
The weight is proportional to the correlation, *i.e*., similarity, between metadata of task 𝐓𝐓\mathbf{T} and condition c𝑐c.
Below we provide simple implementations for zero-shot classification and retrieval, respectively.

Zero-Shot Classification.
To have accurate routing, we leverage fine-grained cluster centers S𝑆S in Step 1 to route a task to data experts.
We treat the set of class names L𝐿L as metadata, and define the similarity matrix between classes and data experts as
𝐀∈ℝ|L|×m𝐀superscriptℝ𝐿𝑚\mathbf{A}\in\mathbb{R}^{|L|\times m}.
To compute 𝐀𝐀\mathbf{A}, we first compute 𝐞lsubscript𝐞𝑙\mathbf{e}\_{l} as the embedding for class l∈L𝑙𝐿l\in L via the same encoder for the embedding of fine-grained cluster center 𝐞ssubscript𝐞𝑠\mathbf{e}\_{s}.
Then each entry is defined as

|  |  |  |  |
| --- | --- | --- | --- |
|  | 𝐀l,s=exp⁡(−‖𝐞l−𝐞s‖22/λ),subscript𝐀  𝑙𝑠superscriptsubscriptnormsubscript𝐞𝑙subscript𝐞𝑠22𝜆\mathbf{A}\_{l,s}=\exp(-\|\mathbf{e}\_{l}-\mathbf{e}\_{s}\|\_{2}^{2}/\lambda), |  | (5) |

where λ∈ℝ+𝜆superscriptℝ\lambda{}\in\mathbb{R}^{+} is a temperature to sharpen the similarities. Further, the weight routing to a data expert f(⋅|c)f(\cdot|c) is proportional to

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(c|𝐓)∝exp⁡(∑l∈L∑s∈Sc𝐀l,s).proportional-to𝑝conditional𝑐𝐓subscript𝑙𝐿subscript𝑠subscript𝑆𝑐subscript𝐀  𝑙𝑠p(c|\mathbf{T})\propto\exp(\sum\nolimits\_{l\in L}\sum\nolimits\_{s\in S\_{c}}\mathbf{A}\_{l,s}). |  | (6) |

In practice, we found that using the nearest neighboring fine-grained cluster center (arg​maxs∈S⁡𝐀l,ssubscriptargmax𝑠𝑆subscript𝐀

𝑙𝑠\operatorname\*{arg\,max}\_{s\in S}\mathbf{A}\_{l,s}) for each class l∈L𝑙𝐿l\in L is good enough to reduce noises in routing.

Zero-Shot Retrieval. The retrieval tasks consist of text retrieval and image retrieval. For text retrieval where each image is used to retrieve a text from a large corpus Q𝑄Q, we leverage Q𝑄Q as metadata to build similarity matrix 𝐀∈ℝ|Q|×m𝐀superscriptℝ𝑄𝑚\mathbf{A}\in\mathbb{R}^{|Q|\times m}.
Similar to the classification task, the weights for ensembling can be naturally adopted for MoDE:

|  |  |  |  |
| --- | --- | --- | --- |
|  | p​(c|𝐓)∝exp⁡(∑q∈Q∑s∈Sc𝐀q,s),proportional-to𝑝conditional𝑐𝐓subscript𝑞𝑄subscript𝑠subscript𝑆𝑐subscript𝐀  𝑞𝑠p(c|\mathbf{T})\propto\exp(\sum\nolimits\_{q\in Q}\sum\nolimits\_{s\in S\_{c}}\mathbf{A}\_{q,s}), |  | (7) |

where each entry 𝐀q,ssubscript𝐀

𝑞𝑠\mathbf{A}\_{q,s} is computed as exp⁡(−‖𝐞q−𝐞s‖22/λ)superscriptsubscriptnormsubscript𝐞𝑞subscript𝐞𝑠22𝜆\exp(-\|\mathbf{e}\_{q}-\mathbf{e}\_{s}\|\_{2}^{2}/\lambda), where 𝐞qsubscript𝐞𝑞\mathbf{e}\_{q} is the embedding for text q𝑞q.
For image retrieval where each text q𝑞q retrieves an image separately, we treat the retrieval by text q𝑞q as an independent task 𝐓qsubscript𝐓𝑞\mathbf{T}\_{q} such that the ensembling weights are then p​(c|𝐓q)∝exp⁡(∑s∈Sc𝐀q,s)proportional-to𝑝conditional𝑐subscript𝐓𝑞subscript𝑠subscript𝑆𝑐subscript𝐀

𝑞𝑠p(c|\mathbf{T}\_{q})\propto\exp(\sum\nolimits\_{s\in S\_{c}}\mathbf{A}\_{q,s}).

## 4 Experiment

### 4.1 Data

We use the datasets collected in MetaCLIP [[50](#bib.bib50)] for evaluation and conduct experiments on image-caption pairs at two scales: 400M (similar to the scale in OpenAI CLIP), and 2.5B to scale MoDE.
All images are pre-processed with face-blurring and de-duplication against benchmarks.

### 4.2 Training Setup

Clustering Setup. We use the pre-trained language model SimCSE [[12](#bib.bib12)] to extract the embeddings for all captions where the advantages of language encoders over CLIP encoders are studied in [Sec. 5.3](#S5.SS3 "5.3 Embeddings for Clustering ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering").
We use balanced K-means [[32](#bib.bib32)] for both of the two unsupervised clustering steps.
We set the number of fine-grained clusters m=1024𝑚1024m=1024, and report performance for both MoDE-2 and MoDE-4 below to directly show the improvement by increase the number of data expert models on all evaluation tasks.

Data Experts Training Setup. We follow OpenAI CLIP’s hyper-parameters [[39](#bib.bib39)] for fair comparison and train on the same budget of 12.8B image-caption pairs (32 epochs of 400M), with a global batch size of 32,768.
We train MoDE under 3 scales: for ViT-B/32 and ViT-B/16, we use 64 Nvidia V100 GPUs with a per GPU batch size of 512, and for ViT-L/14, we use 128 GPUs with a 256 per GPU batch size.
To maintain a reasonable training cost, we start MoDE training from the 27th epoch (out of 32 epochs) of a partially trained MetaCLIP as the seed model and all data experts share the same seed model to save computes.

### 4.3 Evaluation

Zero-Shot Image Classification.
We follow the evaluation protocol in CLIP benchmark [[34](#bib.bib34), [39](#bib.bib39), [50](#bib.bib50)] and use the same class names & prompts by OpenAI CLIP.
For fair comparison, MetaCLIP [[50](#bib.bib50)] naturally serves as the single dense baseline.
The checkpoints of OpenAI CLIP (WIT400M data) [[39](#bib.bib39)] and OpenCLIP (LAION-400M data, LAION-2B data) [[44](#bib.bib44)] are also re-evaluated for fair comparison.

The framework MoDE has shown *consistent performance gain across model scales and data scales*.
Firstly, we compare the models learned from 400M-scale dataset in LABEL:tab:clip400m,
and summarize the results by different model scales. MoDE achieves consistent performance gain where increasing the number of data experts results in better performance.
Next, we study the scaling property of MoDE on 2.5B image-text pairs. From LABEL:tab:clip2b5, comparing against MetaCLIP [[50](#bib.bib50)], the advantage of MoDE to learn four data expert models is better revealed on scaling training data: +1.9% on B/32, +3.7% on B/16, and +1.4% on L/14.
Lastly, we increase the number of data experts. As shown in [Fig. 3](#S4.F3 "In 4.3 Evaluation ‣ 4 Experiment ‣ MoDE: CLIP Data Experts via Clustering"), the performance can be kept improving when we increase the number of data experts where MoDE-16 ViT-B/32 can outperform the MetaCLIP ViT-B/16 baseline.

Notably, MoDE provides *an efficient and scalable approach to consume large-scale data without a large batch size that requires more GPUs* (384 Nvidia A100 GPUs) as in OpenCLIP.
As shown in LABEL:tab:clip2b5, based on ViT-B/16 with a batch size of 32K, the MoDE-2 with two data expert models is on par with the ViT-L/14 model by OpenCLIP [[43](#bib.bib43)], while 4 data expert models can outperform the ViT-L/14 by 1.5% on CLIP benchmark dataset.
Nevertheless, MoDE requires much less pretraining cost. As summarized in [Fig. 4](#S4.F4 "In 4.3 Evaluation ‣ 4 Experiment ‣ MoDE: CLIP Data Experts via Clustering"), MoDE-4 ViT-B/16 only requires less-than-35% of GPU-Hours used for OpenAI CLIP ViT-L/14.
Compared with OpenCLIP trained on LAION-2B data, MoDE-8 ViT-B/32 data experts can even outperform a single ViT-B/16 model by OpenCLIP but only use 31% of its GPU-Hours.
In this way, our approach demonstrates great potential for efficient CLIP pretraining with limited GPUs in future.

!(/html/2404.16030/assets/x3.png)

Figure 3: Average accuracy CLIP benchmark with increased number of data expert models in MoDE (Pretrain set: 2.5B pairs).

Zero-Shot Robustness. In addition, to show a consistent gain on different tasks in the CLIP benchmark, we further validate the benefits towards robustness of MoDE in variants of ImageNet zero-shot classification. As summarized in LABEL:tab:robustness, though there are systematic gaps across variants of ImageNet, learning a set of data experts can improve the zero-shot accuracy on all five variants over the MetaCLIP Baseline for all model scales, and increasing the number of data experts can still introduce consistent gain. For the accuracies on IN-A and IN-O, the gap between baseline and other approaches is mitigated clearly by MoDE.
Finally, MoDE-4 achieves the highest average accuracy of all dataset variants among all compared methods.

Zero-Shot Retrieval. We follow OpenCLIP [[43](#bib.bib43)] and reports the image/text retrieval results on COCO [[29](#bib.bib29)] and Flickr30k [[52](#bib.bib52)]. The compared models are trained on billion-scale datasets. As shown in LABEL:tab:retrieval, learning data experts can improve the scores consistently across all model sizes, on COCO, in particular, +3.3% and +2.7% in R@1 for image-to-text and text-to-image retrieval respectively by ViT-B/16 models, and we achieve the best performance. For the performance gap between MetaCLIP Baseline and OpenCLIP, *e.g*., text retrieval on Flickr30k by ViT-B/32 models, the gap can also be mitigated clearly.

!(/html/2404.16030/assets/x4.png)

Figure 4: Summary of average accuracy on CLIP benchmark and pretraining cost (GPU-Hours). The diameter is proportional to the model size, different approaches are color-coded.

## 5 Discussion

We first analyze the importance of clustering ([Sec. 5.1](#S5.SS1 "5.1 Effectiveness of Clustering ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering")) and then study the MoDE design ([Secs. 5.2](#S5.SS2 "5.2 Clustering Strategy ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering") and [5.3](#S5.SS3 "5.3 Embeddings for Clustering ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering")). Finally, we investigate the potential of our approach in other important research directions ([Secs. 5.4](#S5.SS4 "5.4 Application of Vision Encoders ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering") and [5.5](#S5.SS5 "5.5 Training Priority of Data Experts ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering")).

### 5.1 Effectiveness of Clustering

As MoDE ensembles the data experts learned from different clusters, we are first interested in the effects of clustering and consider two variants for ablation.

Though model ensembling [[22](#bib.bib22)] can provide gains over a single model, we are interested in how a naive ensembling of models trained on similar distribution performs compared to MoDE with data specialization. In LABEL:tab:dual-ablate,
we train two ViT-B/32 CLIP models on the same training data without clustering, and then average the model outputs for prediction (Full-2).
This achieves a similar performance as the baseline.
Thus, the clustering is essential for MoDE.

Furthermore, we randomly split the training data into two subsets, and specialize a data expert for each subset (Random-2).
For a fair comparison, we mimic the size of subsets by MoDE-2 in the random splitting, and all data experts use the same seed model.
As the data split is not obtained through clustering, we still only use the average of model outputs for evaluation. However, though Random-2 can provide small improvement when trained on 2.5B image-caption pairs (60.0 *vs*. 59.8), there is a noticeable drop when training on the 400M pairs (57.7 *vs*. 58.2).

### 5.2 Clustering Strategy

Instead of obtaining the data clusters in a single step, MoDE employs a two-step clustering strategy to discover the centers of fine-grained cluster S𝑆S, which are used to properly model the correlation between task metadata and the conditions ([Sec. 3.2](#S3.SS2 "3.2 Clustering ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering")). We provide ablation studies below to demonstrate this necessity for model ensembling.

Firstly, we evaluate the one-step clustering alternative, *i.e*., m=n𝑚𝑛m=n, and for simplicity, we only learn two data experts (OneStep-2) based on ViT-B/32. As shown in LABEL:tab:ensemble-ablate, we summarize the average score on the CLIP benchmark and stand out the accuracy of ImageNet as it has the most number of classes. As the cluster centers are not representative enough to model the correlation with task metadata, model ensembling in OneStep-2 can even result in a slight drop. We do observe that each data expert alone can outperform MetaCLIP Baseline on different tasks in the CLIP benchmark but it is difficult to pick correctly.

Then, we follow the two-step clustering but alter the number of fine-grained clusters m𝑚m in the first step. As plotted in LABEL:fig:specturm-fine, we summarize the results of MoDE-2 trained on 400M image-caption pairs. With increasing m𝑚m, we observed that the average accuracy on the CLIP evaluation benchmark improves consistently. Though the performance can be improved slightly when m𝑚m is increased from 1024 to 2048, the computational cost during data clustering is also higher. We set m=1024𝑚1024m=1024 in the main experiments.

Lastly, as another piece of evidence, we keep m𝑚m as 1024 but use the coarse-grained cluster centers in Step 2, to determine the ensembling weights (CoarseCluster). As shown in LABEL:tab:ensemble-ablate , as the meta clusters are not representative enough to obtain good ensembling weight, the resulting accuracy improvement is trivial. When we increase the number of data experts from 2 to 4, the gap between CoarseCluster-4 and MoDE-4 is even enlarged, which further demonstrates the importance of using fine-grained clusters to determine the ensembling weight for data experts in our MoDE.

### 5.3 Embeddings for Clustering

We further validate the importance of using language embeddings.
In addition to SimCSE [[12](#bib.bib12)] language embedding, we investigate the following embeddings for clustering: (1) image embedding from the open-sourced DINOv2 [[36](#bib.bib36)]; (2) image and/or text embeddings from the seed model (*i.e*., the partially trained MetaCLIP checkpoints on the 27th epoch).
When the image embeddings are used for clustering, for each test image, we use its similarity with all fine-grained cluster centers to determine the logits ensemble weights. When both image and text embeddings are used, we use their concatenation as the feature for clustering.
Without loss of generality, we compare with MoDE-2 trained on 400M pairs and set m=1024𝑚1024m=1024 for fair comparison. We summarize the scores in LABEL:tab:ablate-embedding and report the zero-shot accuracy CLIP benchmark and ImageNet.

Firstly, by using image embeddings for clustering, the resulting models underperform MetaCLIP, in particular on ImageNet, and we believe the main reason is that the image embedding contains low-level details.
As such, the cluster centers are not representative of model ensembling.

Furthermore, utilizing the language embeddings from the seed model yields only marginal performance improvement. This suggests that the CLIP embedding may still fall short of discerning high-level semantic correlations.
This occurs as the language embeddings are influenced by image embeddings, potentially overlooking high-level semantics not depicted in corresponding images. For example, abstract concepts such as “travel”, “product”, and “politics” may lack corresponding visual elements.
In contrast, the SimCSE text embeddings pretrained on large text corpora can understand abstract concepts for clustering.
As another evidence, we compare the clustering based on language embeddings and use TF-IDF embeddings as reference. As the TF-IDF embedding is determined by on the frequency of words, the clusters on TF-IDF embeddings shown in [Fig. 5](#S5.F5 "In 5.3 Embeddings for Clustering ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering") can only group captions with the same words, and struggle to provide abstract concepts via discrete text tokens. In contrast, using SimCSE embeddings can group the captions with coherent semantics (*e.g*., food).

!(/html/2404.16030/assets/x5.png)

Figure 5: Representative instances for each cluster.

### 5.4 Application of Vision Encoders

Besides zero-shot generalization, the set of vision encoders can also be directly ensembled in downstream application.
Notably, all vision encoders are equally weighted, and we do not need any cluster center (*i.e*., cluster-independent), which is generalizable to the case where the language metadata such as class names is not available.

Firstly, we ensemble the encoder outputs and use ImageNet classification for evaluation. Specifically, for each image, we concatenate the outputs from all (n𝑛n) vision encoders as the representation and feed it into a linear layer for classification. To maintain reasonable training cost, only linear probing is considered where we exclusively train the linear classifier from scratch and fix all vision encoders. As shown in [Table 1](#S5.T1 "In 5.4 Application of Vision Encoders ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering"), our MoDE achieves consistent and clear performance gain over MetaCLIP Baseline.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Linear Probe∗ | | | Linear Probe | | |
| B/32 | B/16 | L/14 | B/32 | B/16 | L/14 |
| MetaCLIP | 69.3 | 73.3 | 80.3 | 67.5 | 73.8 | 82.3 |
| MoDE-2 | 68.9 | 73.8 | 80.6 | 71.3 | 76.9 | 83.9 |
| MoDE-4 | 69.1 | 74.5 | 80.7 | 74.1 | 79.6 | 84.7 |
| ∗: Initialize classifier with language embeddings as in OpenCLIP [[43](#bib.bib43)]. | | | | | | |
| --- | --- | --- | --- | --- | --- | --- |

Table 1: Performance comparison on ImageNet via linear probing.

As shown in [Table 2](#S5.T2 "In 5.4 Application of Vision Encoders ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering"), we evaluate all vision encoders by MoDE-4 ViT-B/16 independently and report the accuracy via linear probing and finetuning (*i.e*., all parameters are trained). Linear probing on the concatenated features achieves higher score than finetuning a single model (79.6 Vs. 76.7) but with much less training cost.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Data Experts | Zero-Shot | Linear Probe∗ | Linear Probe | Finetune |
| MetaCLIP | 72.1 | 73.3 | 73.8 | 76.7 |
| 0 | 63.3 | 66.4 | 67.3 | 75.7 |
| 1 | 68.5 | 71.3 | 72.0 | 76.9 |
| 2 | 65.2 | 68.2 | 68.8 | 76.3 |
| 3 | 72.9 | 74.9 | 74.2 | 77.2 |
| ∗: Initialize classifier with language embeddings as in OpenCLIP [[43](#bib.bib43)]. | | | | |
| --- | --- | --- | --- | --- |

Table 2: Evaluation for each data expert in MoDE-4 ViT-B/16.

In addition, by comparing among vision encoders, the data expert achieving higher zero-shot accuracy also hits the best score in both linear probing and finetuning, indicating a consistent correlation benefited from the strong encoder initialization. In this way, by training data expert on each coarse-grained cluster, we increase the quality negative within each mini-batch to learn stronger vision encoders effectively.
Finally, the parameters can also be averaged and then used as initialization of a single network for finetuning, and more details can be found in the Supp.

### 5.5 Training Priority of Data Experts

As the data experts can be trained asynchronously, MoDE introduces flexibility in the data expert training priority. Below we demonstrate the robustness and effectiveness of MoDE when the data experts are trained in order.

Firstly, we rank the conditions, *i.e*., coarse-level clusters, to determine the training priority of data experts. This is useful when the computational resource is not sufficient to learn a giant dense model or all data experts together. We use the diversity of fine-grained clusters as a reference, and first train the model on the condition with the largest range, *i.e*., the average distance between fine-grained clusters and the coarse-grained center. As shown in [Fig. 6](#S5.F6 "In 5.5 Training Priority of Data Experts ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering"), we vary the total number of ViT-B/32 data experts, *i.e*., n𝑛n, from 2 to 32 and summarize the average accuracy on the CLIP benchmark. When the data experts are gradually included, the performance keeps increasing.

!(/html/2404.16030/assets/x6.png)

Figure 6: CLIP benchmark accuracy by MoDE-n𝑛n when the data experts based on ViT-B/32 are developed in order and added to the system progressively. The pre-train set contains 2.5B pairs.

In this way, instead of learning from all data simultaneously, MoDE enables progressive integration of new data experts, enabling dynamic updates. MoDE holds promise for applications such as *online* and continual learning. With each new set of data, it has the flexibility to update a pre-trained data expert, or to learn a new data expert. This is particularly valuable when the incoming data are unprecedented to the existing expert system. We leave the trade-off between catastrophic forgetting [[24](#bib.bib24)] and effective adaption as the futrure work.

At the same time, we can also select the clusters given the task metadata as prior following the retrieval-enhanced setup [[17](#bib.bib17)].
When the metadata is accessible, we use the SimCSE [[12](#bib.bib12)] to extract their embeddings and retrieve the nearest fine-grained clusters for each of them. Then, the data expert trained on the selected clusters is of highest training priority, and we only train that single data expert for evaluation while the rest clusters can be left for future continual MoDE pretraining if needed.
We take ImageNet as an example where the 1000 class names are used to retrieve clusters. As summarized in LABEL:tab:retrieval-enhance, adapting our approach can improve the efficiency of network training significantly and can even escalate the performance along the model scale in most cases. For example, our ViT-B/16 outperforms the L/14 models by OpenAI CLIP/ OpenCLIP and our ViT-L/14 even outperforms the ViT-G/14 in OpenCLIP. Besides, as detailed in Suppl., MoDE can also be aligned for a set of downstream tasks, *e.g*., CLIP benchmarks.

In summary, MoDE can be applied to different types of downstream tasks. Meanwhile, the coarse-level clustering in the second step tentatively assumes the fine-grained clusters should be split into disjoint groups with overlap.
We believe the fine-grained clusters can also be grouped flexibly and we leave it for future work.

## 6 Conclusion

The success of CLIP depends on the quality negative samples. As the *false negative* noise in web-crawled pairs hurts training effectiveness, scaling CLIP on large-scale data presents unique challenges in terms of training efficiency and computational bottlenecks.
To this end, we have presented Mixture of Data Experts (MoDE) to asynchronously train a group of *data experts*. Each expert model is trained on a set of fine-grained clusters where the data in each cluster is of coherent semantics and all data experts are trained individually.
During inference, the outputs are selectively ensembled based on the requirements for each task and modeled by the correlation between task metadata and fine-grained cluster centers.
Empirically, MoDE significantly outperforms OpenCLIP and OpenAI CLIP on standard benchmarks with less than 35% training cost.
Furthermore, the image embeddings extracted by all data experts can be combined to enhance the representation of visual information.
We plan to adapt MoDE for generative models in the future.

## Acknowledgement

The authors would like to thank Xinlei Chen and Margaret Li for constructive discussion.

## References

* [1]

  Richard E Blahut.
  Fast algorithms for signal processing.
  Cambridge University Press, 2010.
* [2]

  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
  A simple framework for contrastive learning of visual representations.
  In International conference on machine learning, pages 1597–1607. PMLR, 2020.
* [3]

  Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.
  Imagenet: A large-scale hierarchical image database.
  In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
* [4]

  Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam.
  Hyperbolic image-text representations.
  In International Conference on Machine Learning, pages 7694–7731. PMLR, 2023.
* [5]

  Inderjit S Dhillon and Dharmendra S Modha.
  Concept decompositions for large sparse text data using clustering.
  Machine learning, 42:143–175, 2001.
* [6]

  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
  An image is worth 16x16 words: Transformers for image recognition at scale.
  arXiv preprint arXiv:2010.11929, 2020.
* [7]

  David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever.
  Learning factored representations in a deep mixture of experts.
  arXiv preprint arXiv:1312.4314, 2013.
* [8]

  Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, and Sanja Fidler.
  VSE++: improving visual-semantic embeddings with hard negatives.
  In British Machine Vision Conference 2018, BMVC 2018, Newcastle, UK, September 3-6, 2018, page 12. BMVA Press, 2018.
* [9]

  William Fedus, Barret Zoph, and Noam Shazeer.
  Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
  The Journal of Machine Learning Research, 23(1):5232–5270, 2022.
* [10]

  Alex Gammerman, Volodya Vovk, and Vladimir Vapnik.
  Learning by transduction.
  arXiv preprint arXiv:1301.7375, 2013.
* [11]

  Yossi Gandelsman, Yu Sun, Xinlei Chen, and Alexei Efros.
  Test-time training with masked autoencoders.
  Advances in Neural Information Processing Systems, 35:29374–29385, 2022.
* [12]

  Tianyu Gao, Xingcheng Yao, and Danqi Chen.
  Simcse: Simple contrastive learning of sentence embeddings.
  In 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, pages 6894–6910. Association for Computational Linguistics (ACL), 2021.
* [13]

  Suchin Gururangan, Margaret Li, Mike Lewis, Weijia Shi, Tim Althoff, Noah A Smith, and Luke Zettlemoyer.
  Scaling expert language models with unsupervised domain discovery.
  arXiv preprint arXiv:2303.14177, 2023.
* [14]

  Michael Gutmann and Aapo Hyvärinen.
  Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.
  In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 297–304, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR.
* [15]

  Guangxing Han, Jiawei Ma, Shiyuan Huang, Long Chen, and Shih-Fu Chang.
  Few-shot object detection with fully cross-transformer.
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5321–5330, 2022.
* [16]

  Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.
  Momentum contrast for unsupervised visual representation learning.
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.
* [17]

  Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid.
  Retrieval-enhanced contrastive vision-text models.
  arXiv preprint arXiv:2306.07196, 2023.
* [18]

  Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton.
  Adaptive mixtures of local experts.
  Neural computation, 3(1):79–87, 1991.
* [19]

  Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig.
  Scaling up visual and vision-language representation learning with noisy text supervision.
  In International conference on machine learning, pages 4904–4916. PMLR, 2021.
* [20]

  Jeff Johnson, Matthijs Douze, and Hervé Jégou.
  Billion-scale similarity search with GPUs.
  IEEE Transactions on Big Data, 7(3):535–547, 2019.
* [21]

  Michael I Jordan and Robert A Jacobs.
  Hierarchical mixtures of experts and the em algorithm.
  Neural computation, 6(2):181–214, 1994.
* [22]

  Michael I Jordan and Tom M Mitchell.
  Machine learning: Trends, perspectives, and prospects.
  Science, 349(6245):255–260, 2015.
* [23]

  Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus.
  Hard negative mixing for contrastive learning.
  Advances in Neural Information Processing Systems, 33:21798–21809, 2020.
* [24]

  James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell.
  Overcoming catastrophic forgetting in neural networks.
  CoRR, abs/1612.00796, 2016.
* [25]

  Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen.
  Gshard: Scaling giant models with conditional computation and automatic sharding.
  arXiv preprint arXiv:2006.16668, 2020.
* [26]

  Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer.
  Branch-train-merge: Embarrassingly parallel training of expert language models.
  arXiv preprint arXiv:2208.03306, 2022.
* [27]

  Xianhang Li, Zeyu Wang, and Cihang Xie.
  An inverse scaling law for clip training.
  arXiv preprint arXiv:2305.07017, 2023.
* [28]

  Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He.
  Scaling language-image pre-training via masking.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390–23400, 2023.
* [29]

  Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.
  Microsoft coco: Common objects in context.
  In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
* [30]

  Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, and Chunyuan Li.
  Learning customized visual models with retrieval-augmented knowledge.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15148–15158, 2023.
* [31]

  Jiawei Ma, Hanchen Xie, Guangxing Han, Shih-Fu Chang, Aram Galstyan, and Wael Abd-Almageed.
  Partner-assisted learning for few-shot image classification.
  In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10573–10582, 2021.
* [32]

  Mikko I Malinen and Pasi Fränti.
  Balanced k-means for clustering.
  In Structural, Syntactic, and Statistical Pattern Recognition: Joint IAPR International Workshop, S+ SSPR 2014, Joensuu, Finland, August 20-22, 2014. Proceedings, pages 32–41. Springer, 2014.
* [33]

  Tom M Mitchell.
  Machine learning, 1997.
* [34]

  Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie.
  Slip: Self-supervision meets language-image pre-training.
  In European Conference on Computer Vision, pages 529–544. Springer, 2022.
* [35]

  Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby.
  Multimodal contrastive learning with limoe: the language-image mixture of experts.
  Advances in Neural Information Processing Systems, 35:9564–9576, 2022.
* [36]

  Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
  Dinov2: Learning robust visual features without supervision.
  arXiv preprint arXiv:2304.07193, 2023.
* [37]

  Mandela Patrick, Po-Yao Huang, Yuki Markus Asano, Florian Metze, Alexander G. Hauptmann, João F. Henriques, and Andrea Vedaldi.
  Support-set bottlenecks for video-text representation learning.
  In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
* [38]

  Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei Yu, Jiahui Yu, Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, et al.
  Combined scaling for zero-shot transfer learning.
  Neurocomputing, 555:126658, 2023.
* [39]

  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
  Learning transferable visual models from natural language supervision.
  In International conference on machine learning, pages 8748–8763. PMLR, 2021.
* [40]

  Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby.
  Scaling vision with sparse mixture of experts.
  Advances in Neural Information Processing Systems, 34:8583–8595, 2021.
* [41]

  Vin Sachidananda, Ziyi Yang, and Chenguang Zhu.
  Global selection of contrastive batches via optimization on sample permutations.
  In Proceedings of the 40th International Conference on Machine Learning, 2023.
* [42]

  Stephan R Sain.
  The nature of statistical learning theory, 1996.
* [43]

  Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al.
  Laion-5b: An open large-scale dataset for training next generation image-text models.
  Advances in Neural Information Processing Systems, 35:25278–25294, 2022.
* [44]

  Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki.
  Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.
  arXiv preprint arXiv:2111.02114, 2021.
* [45]

  Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean.
  Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
  In International Conference on Learning Representations, 2016.
* [46]

  Jake Snell, Kevin Swersky, and Richard Zemel.
  Prototypical networks for few-shot learning.
  Advances in neural information processing systems, 30, 2017.
* [47]

  Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt.
  Test-time training with self-supervision for generalization under distribution shifts.
  In International conference on machine learning, pages 9229–9248. PMLR, 2020.
* [48]

  Chen-Wei Xie, Siyang Sun, Xiong Xiong, Yun Zheng, Deli Zhao, and Jingren Zhou.
  Ra-clip: Retrieval augmented contrastive language-image pre-training.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19265–19274, 2023.
* [49]

  Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer.
  Videoclip: Contrastive pre-training for zero-shot video-text understanding.
  In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP, 7-11 November, 2021, pages 6787–6800. Association for Computational Linguistics, 2021.
* [50]

  Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer.
  Demystifying clip data.
  arXiv preprint arXiv:2309.16671, 2023.
* [51]

  Yuncong Yang, Jiawei Ma, Shiyuan Huang, Long Chen, Xudong Lin, Guangxing Han, and Shih-Fu Chang.
  TempCLR: Temporal alignment representation with contrastive learning.
  In The Eleventh International Conference on Learning Representations, 2023.
* [52]

  Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier.
  From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.
  Transactions of the Association for Computational Linguistics, 2:67–78, 2014.
* [53]

  Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu.
  Coca: Contrastive captioners are image-text foundation models.
  arXiv preprint arXiv:2205.01917, 2022.
* [54]

  Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer.
  Lit: Zero-shot transfer with locked-image text tuning.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022.

## Appendix A Full Results

Below we provide the complete results of MoDE reported in [Sec. 4](#S4 "4 Experiment ‣ MoDE: CLIP Data Experts via Clustering") if mentioned.
Firstly, we compare the performance on CLIP evaluation benchmark, and reports the scores by scaling up the number of coarse-grained clusters in LABEL:tab:clip2b5-scale. When more data experts are learned, the average accuracy on CLIP benchmark keeps improving.

Secondly, we summarize the results for zero-shot retrieval in LABEL:tab:retrieval-full.
The results are separated by the scale of pre-train dataset. Consistently, our approach can outperform the MetaCLIP Baseline in all cases. MoDE also achieves the best score in most cases.

We noticed the work LiMoE [[35](#bib.bib35)] which follows conventional Deep Mixture of Expert models and trains a stack of Transformer MoE layers on all 3.6B image-caption pairs [[54](#bib.bib54)]. However, the number of parameters in a single LiMoE network is much larger than a single dense baseline. As all of the network parameters are trained synchronously, it will cause huge memory usage. Meanwhile, comparing with MoDE-4 trained on different data clusters while the total pre-train set has only about 2.5B image-caption pairs, our system is more flexible and also achieve better results consistently.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | ViT-B/32 | ViT-B/16 | ViT-L∗ |
| classification | LiMoE | 67.5 | 73.7 | 78.6 |
| (ImageNet) | MoDE-4 | 68.9 | 74.3 | 79.4 |
| text retrieval | LiMoE | 45.7 | 51.3 | 55.7 |
| (CoCo) | MoDE-4 | 57.4 | 62.7 | 65.6 |
| image retrieval | LiMoE | 31.0 | 36.2 | 39.6 |
| (CoCo) | MoDE-4 | 39.9 | 44.1 | 48.2 |
| ∗: LiMoE uses L/16 and MoDE uses L/14. | | | | |
| --- | --- | --- | --- | --- |

Table 3: Comparison with LiMOE [[54](#bib.bib54)]

## Appendix B Ablation Study Details for Clustering

Firstly, for ablation details on Clustering Strategy in [Sec. 5.2](#S5.SS2 "5.2 Clustering Strategy ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering"), we show details in LABEL:tab:cluster-supp for LABEL:tab:ensemble-ablate and LABEL:tab:clip400m-scale for LABEL:fig:specturm-fine.

Then, for the embedding types, we provide the details of MoDE-2 in LABEL:tab:ablate-embedding-full.
We note that the SimCSE [[12](#bib.bib12)] can be trained via unsupervised or supervised ways. The unsupervised training strategy utilizes dropout masks to generate two views from the same sentence to build positive pair while the latter one uses two sentences which are of similar semantic meaning as positive samples to each other. Regardless the training strategy, we found the average score on CLIP benchmark is the same.

Meanwhile, when both image and language embeddings are used for clustering, we concatenate their embeddings and we experimentally found that adding the language and image embeddings pair-wisely cannot result in meaningful cluster. However, at inference time, the ensembling weights should be calculated for all image-class pairs in the zero-shot classification task, which is computational heavy but provides very limited gain over the baseline.

## Appendix C Robustness in Training Priority

For the retrieval-enhanced setup, besides using the class names of a single dataset to retrieve the most important finegrained data clusters, we can also use the class names of all tasks in CLIP benchmark. The detailed results are summarized in LABEL:tab:clipeval-prior.

## Appendix D Robustness of Vision Encoders

For emsembling over model outputs, we can also add the model outputs element-wisely. However, as all vision encoders are separately trained, the learned embedding spaces are not necessarily aligned with each other. As a result, ensembling via element-wise addition does not introduce gain, *e.g*., for MoDE-4 with ViT-B/16 encoders, the accuracy is only 74.5 compared with 79.6 in [Table 1](#S5.T1 "In 5.4 Application of Vision Encoders ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering").

Finally, in addition to directly aggregate the feature outputs by all data experts, the parameters learned in MoDE can also be ensembled via averaging and then used as initialization of a single network for finetuning. As shown in [Table 4](#A4.T4 "In Appendix D Robustness of Vision Encoders ‣ MoDE: CLIP Data Experts via Clustering"), we use ViT-B/32 vision encoder, and achieve consistent gain over MetaCLIP Baseline.

| Approach | Accuracy |
| --- | --- |
| MetaCLIP | 73.7 |
| MoDE-2 | 74.0 |
| MoDE-4 | 74.2 |
| MoDE-8 | 73.9 |
| MoDE-16 | 74.1 |
| MoDE-32 | 74.1 |

Table 4: Accuracy on ImageNet via parameter averaging.

## Appendix E Implementation Detail

#### Clustering.

We first sample 100M captions from the 400M image-caption pairs to learn the cluster centers in an unsupervised manner. Then, we use nearest neighbor to determine the cluster assignment for all other samples in the 400M as well as 2.5B dataset. We also observed that the cluster centers learned by using less than 2M samples can also result in similar clustering assignments using spherical K-means clustering [[5](#bib.bib5)] via FAISS [[20](#bib.bib20)]. In practice, we observed that the balanced K-means clustering algorithm does not necessarily enforce strict balance regarding the distribution of the clusters. For example, for the two coarse-grained clusters on 400M dataset used to train MoDE-2, the number of samples for each cluster are around 170M and 230M respectively. Consequently, as mentioned for Random-2 in [Sec. 5.1](#S5.SS1 "5.1 Effectiveness of Clustering ‣ 5 Discussion ‣ MoDE: CLIP Data Experts via Clustering"), mimic the size of subsets by MoDE-2 in the random splitting for fair comparison.

#### Similarity matrix.

For task-level adaptation, as mentioned in [Sec. 3.4](#S3.SS4 "3.4 Inference Time Task-Adaptation ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering"), we use the nearest neighbor fine-grained cluster (arg​maxs∈S⁡𝐀l,ssubscriptargmax𝑠𝑆subscript𝐀

𝑙𝑠\operatorname\*{arg\,max}\_{s\in S}\mathbf{A}\_{l,s}) for each class l∈L𝑙𝐿l\in L. In other words, we apply a maximum filter for each row, *i.e*., 𝐀lsubscript𝐀𝑙\mathbf{A}\_{l}, where the non-maximum values are reset as 0, *i.e*., 𝐀l,s′=0subscript𝐀

𝑙superscript𝑠′0\mathbf{A}\_{l,s^{\prime}}=0 if s′≠s^superscript𝑠′^𝑠s^{\prime}\neq\hat{s} where s^=arg​maxs∈S⁡𝐀l,s^𝑠subscriptargmax𝑠𝑆subscript𝐀

𝑙𝑠\hat{s}=\operatorname\*{arg\,max}\_{s\in S}\mathbf{A}\_{l,s}. Then, we scale the raw distance value 5 times, *i.e*., setting the temperature (divisor) as 0.2, according to our experimental cross validation.

#### Routing Weights.

As described in [Eq. 6](#S3.E6 "In 3.4 Inference Time Task-Adaptation ‣ 3 CLIP Data Experts ‣ MoDE: CLIP Data Experts via Clustering"), the routing weight p​(c|𝐓)𝑝conditional𝑐𝐓p(c|\mathbf{T}) of a data expert f(⋅|c)f(\cdot|c) is essentially obtained via softmax normalization.
At inference time, we note the routing weights should be reasonably distant from each other.
Consequently, given the classification task with the class names L𝐿L, we use the number of classes |L|𝐿|L| to roughly adjust the weights.
Firstly, when |L|𝐿|L| is small, *e.g*., |L|<10𝐿10|L|<10, though only one data expert can be activated, the selection could be sensitive to noisy routing. Then, we soften the values in 𝐀𝐀\mathbf{A} by multiplying
exp⁡(0.5−|L|)0.5𝐿\exp(0.5-\sqrt{|L|}) to ensemble two data experts in most cases.
On the other hand, when |L|𝐿|L| is large, *e.g*., |L|>200𝐿200|L|>200, the normalized weights tend to be over-smooth, we thus use a much smaller temperature by dividing the λ𝜆\lambda by log⁡(|L|)𝐿\log(|L|). Then, we can only select a few data experts and have low-entropy routing weights.
