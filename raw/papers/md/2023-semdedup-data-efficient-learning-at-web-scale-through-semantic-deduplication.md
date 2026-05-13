---
arxiv: '2303.09540'
authors:
- Amro Abbas
- Kushal Tirumala
- Dániel Simig
- Surya Ganguli
- Ari S. Morcos
parser: ar5iv
retrieved: '2026-05-11'
source: paper
title: 'SemDeDup: Data-efficient learning at web-scale through semantic deduplication'
url: https://arxiv.org/abs/2303.09540
year: 2023
---

[2303.09540] SemDeDup: Data-efficient learning at web-scale through semantic deduplication














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



\contourlength

0.1pt
\contournumber10

# SemDeDup: Data-efficient learning at web-scale through semantic deduplication

Amro Abbas1
     
Kushal Tirumala1
     
Dániel Simig1∗
     
Surya Ganguli2
     
Ari S. Morcos1∗
  
1Meta AI (FAIR)     2Department of Applied Physics, Stanford University
  
Equal Contribution. Correspondence to: Amro Abbas amroabbas@meta.com, Ari Morcos arimorcos@meta.com

###### Abstract

Progress in machine learning has been driven in large part by massive increases in data. However, large web-scale datasets such as LAION are largely uncurated beyond searches for exact duplicates, potentially leaving much redundancy. Here, we introduce SemDeDup, a method which leverages embeddings from pre-trained models to identify and remove “semantic duplicates”: data pairs which are semantically similar, but not exactly identical. Removing semantic duplicates preserves performance and speeds up learning. Analyzing a subset of LAION, we show that SemDeDup can remove 50% of the data with minimal performance loss, effectively halving training time. Moreover, performance increases out of distribution. Also, analyzing language models trained on C4, a partially curated dataset, we show that SemDeDup improves over prior approaches while providing efficiency gains. SemDeDup provides an example of how simple ways of leveraging quality embeddings can be used to make models learn faster with less data.

## 1 Introduction

![Refer to caption](/html/2303.09540/assets/x1.png)


(a)

![Refer to caption](/html/2303.09540/assets/x2.png)


(b)

Figure 1: Data efficiency from semantic deduplication (SemDeDup) (a): A schematic of the SemDeDup algorithm which efficiently removes semantic duplicates from web-scale data. (b): When SemDeDup removes 50% of the LAION-440M dataset, training on this semantically nonredundant subset achieves almost the same performance as training on the entire 440M dataset. Also, training speed is twice as fast and completes in half the time.

![Refer to caption](/html/2303.09540/assets/figures/semantic_duplicates.png)


Figure 2: Mapping cosine similarity to perceptual and semantic similarity. We visualize pairs of images with cosine similarity 1−ϵ1italic-ϵ1-\epsilon in the CLIP image encoder embedding space. The left most image is a random seed image from LAION, while the remaining images are sorted by their dissimilarity ϵitalic-ϵ\epsilon to the seed image. Roughly, as ϵitalic-ϵ\epsilon increases from left to right, we move from perceptual to semantic duplicates, while at large values of ϵitalic-ϵ\epsilon we see semantically redundant pairs. Note the red labelled “semantic duplicate" is a view of the original left-most seed image from a slightly different perspective. We visualize more examples in Figure [A9](#A4.F9 "Figure A9 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication").

A primary driver of recent success in machine learning has been the rise of self-supervised learning (SSL) scaled to ever larger models and unlabelled datasets [[1](#bib.bib1), [2](#bib.bib2), [3](#bib.bib3), [4](#bib.bib4), [5](#bib.bib5), [6](#bib.bib6), [7](#bib.bib7), [8](#bib.bib8)]. In particular, modern large datasets are often derived at global web-scale and are generally unfiltered, with the exception of NSFW filters. One such public dataset is LAION [[9](#bib.bib9)], a multi-modal dataset of 5 billion image/text pairs. Multi-modal models such as CLIP [[10](#bib.bib10)] are trained for many epochs on these large datasets achieving impressive performance but at the cost of extremely long training durations.

The critical role of large datasets has led to increasing interest in scaling laws which enable us to predict how a model’s performance will change given more data and/or parameters, leading to the observation that test error generally scales as a power law with respect to data quantity [[2](#bib.bib2)]. Power law scaling, however, is unsustainable as diminishing marginal returns are quickly hit such that ever increasing amounts of data are required to achieve ever diminishing improvements in performance. Notably, many of these models appear never to converge, as test performance continues to increase even after 10s of passes through these massive datasets [[11](#bib.bib11), [12](#bib.bib12)]. This result suggests that our best models are underfitting, likely as a result of spending an increasing fraction of learning time focusing on redundant data.

Improving data efficiency would therefore be quite impactful, either by enabling models to achieve the same performance much faster, or by enabling models to achieve better performance given the same computational budget. These observations have inspired recent work which suggests that by pruning training data according to an intelligent criterion, power law scaling with respect to data can be beaten and, given an optimal data ranking metric, exponential scaling might in principle be achieved [[13](#bib.bib13)]. Recent explorations of this direction have shown promising results, with some works able to reduce data size by almost 5-fold with minimal performance loss [[14](#bib.bib14)].

However, optimal approaches to select data remain poorly understood. Such approaches might focus on one of several different classes of examples to be removed, roughly ordered by the complexity of their discovery:

1. 1.

   Perceptual duplicates: We loosely define such data pairs to be perceptually identical to a typical human observer. The most straightforward version would be exact duplicates at the pixel or token level that could easily be found via exact duplicate detection in input space. However, such approaches might miss pairs of images with human imperceptible pixel level distortions. Most widely-used datasets have some exact duplicate filter already applied, though perceptual duplicates with slight pixel-level differences may pass through such filters.
2. 2.

   Semantic duplicates: these are examples which contain largely identical information content, but remain perceptually distinct. For example, a pair of image views which are derived from the same image, but feature different margins, aspect ratios, color distributions, etc. could be considered semantic duplicates. A pair of sentences with the same structure but some words exchanged for synonyms would also be considered a semantic duplicate. Such pairs would rarely, if ever, be detected by exact duplicate filters as they would be far apart in pixel/token space.
3. 3.

   Semantically redundant data: in contrast to semantic duplicates, semantically redundant data are not derived from the same underlying objects and would be clearly distinguishable to a human. However, the information contained in such examples may still contain substantial overlap. For example, consider the case of two different images of two different golden retrievers in two different parks. These images are neither perceptually nor semantically identical as the content of the images differs. However, the information contained in them is quite similar, leading us to think of such pairs as semantically redundant. Each additional semantically redundant data point will provide less and less new information, eventually converging to near-zero information gained from additional such data. Methods such as SSL Prototypes [[13](#bib.bib13)] and memorization [[15](#bib.bib15)] search for semantically non-redundant data subsets to train on.
4. 4.

   Misleading data: these are data which rather than providing zero information (as in the previous categories) provide negative or harmful signal, in the sense that removing these data actually improves performance, rather than having a neutral effect. While such data are easy to conceive of in supervised learning (i.e. mislabeled examples), it is much less clear what such examples may be in the context of self-supervised learning.

In this work, we focus on the category of semantic duplicates: data which are semantically highly similar but which would be difficult to discover using simple deduplication approaches. These data points are challenging to identify because distance measures in input space are unlikely to uncover semantic duplicates. To overcome this limitation, we leverage pre-trained foundation models to compare data similarity in the learned embedding space rather than in input space. Comparing every data point to every other data point, however, is intractable, especially for web-scale datasets containing billions of examples. To make this computation possible, we use the clustering approach described in [[13](#bib.bib13)] to segment the embedding space, allowing us to only search for duplicate pairs within a cluster. Using this approach, we make the following contributions:

* •

  We propose SemDeDup (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), a), a simple, yet effective and computationally tractable way to identify semantic duplicates. Using this approach, we show that large web-scale datasets such as LAION contain large numbers of semantic duplicates, with 50% of examples containing at least one semantic duplicate.
* •

  Large fractions of semantic duplicates can be removed with little-to-no performance impact, greatly increasing training efficiency. We reduced the size of our LAION training set by 50% with minimal performance loss, and improved learning speed, achieving nearly the same performance 2x faster (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), b), and moreover improved performance out-of-distribution.
* •

  We apply SemDeDup to C4, a large text corpus, beating prior SoTA deduplication while providing efficiency gains of 15%, sometimes even improving performance.

Overall, our results demonstrate a simple yet surprisingly effective approach to reduce the cost of training through the removal of semantic duplicates which is likely applicable to all web-derived datasets and may help to democratize the training of large-scale foundation models by improving data and compute efficiency.

## 2 Related Work

Much of the work in language and vision on deduplication has focused on the removal of exact duplicates. For example, [[16](#bib.bib16)] removed duplicates between the YFCC15M dataset [[17](#bib.bib17)] and the ImageNet validation set to prevent train-test leakage. The C4 text corpus - used for training T5 [[18](#bib.bib18)] - has been deduplicated by discarding repeated occurrences of any three-sentence spans. [[19](#bib.bib19)] showed that it’s possible to further deduplicate this dataset without loss of performance by computing approximate n-gram overlap between documents using the MinHash technique [[20](#bib.bib20)]. [[21](#bib.bib21)] also applied MinHash based deduplication to curate training data for the Gopher model and demonstrated that training on the deduplicated dataset can result in lower perplexity across various validation sets.  [[22](#bib.bib22)] found that deduplication prevents memorization in LLMs and thus mitigates privacy concerns. More recent works use forms of model-based feature extraction to improve the robustness of the similarity metric used for deduplication.  [[23](#bib.bib23)] created a supervised dataset for detecting duplicate news articles and trained models to predict those labels. In the domain of computer vision,  [[24](#bib.bib24)] improves on SSL techniques by removing near-duplicates in some high dimensional feature space they learn.

Beyond deduplication, a host of classical machine learning approaches seek to achieve data efficiency by finding coresets, defined as small subsets of the training data that can be used to train a machine learning algorithm to the same test accuracy achievable when training on the entire training data (see e.g. [[25](#bib.bib25), [26](#bib.bib26)] for reviews). However, many coreset algorithms are computationally prohibitive and therefore are difficult to scale to web-scale data. In contrast to many traditional coreset algorithms, we develop an exceedingly simple and tractable algorithm that achieves both computational and data efficiency at scale.

Recent approaches to achieve data efficiency in deep learning have operated in a supervised setting by defining and finding “hard” examples not easily learned by partially or fully trained (ensembles of) models [[27](#bib.bib27), [28](#bib.bib28), [29](#bib.bib29), [15](#bib.bib15), [30](#bib.bib30)]. Perhaps the closest to our work is a recent effort to break beyond neural power law scaling by pruning unlabelled data, using the embedding space of a pre-trained foundation model [[13](#bib.bib13)]. However, the largest dataset for which these works examined data pruning was ImageNet. In contrast, we move from relatively small, highly curated ImageNet scale to highly uncurated, web-scale data. Our analysis, at this new large and uncurated scale, reveals a possibly fundamental role for semantic deduplication as an important initial step in data-pruning for self-supervised learning that was not considered in prior data-pruning works.

## 3 SemDeDup

#### Defining and identifying semantic duplicates

While identifying perceptual duplicates can be easily done in input space, identifying semantic duplicates is more difficult as they may be distant in either pixel or token space. To identify these pairs, we leverage the embedding space of a large pre-trained foundation model to provide a more semantically meaningful distance metric. To detect and remove semantically similar images, we use the following semantic de-duplication (SemDeDup) algorithm (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), a). First, we embed each data point using a foundation model (CLIP [[11](#bib.bib11), [31](#bib.bib31)] for images and OPT [[32](#bib.bib32)] for language). We then cluster the embeddings into k𝑘k clusters via k-means. Below, we choose k=50,000𝑘

50000k=50,000 clusters in CLIP image encoder embeddings and k=11,000𝑘

11000k=11,000 clusters in OPT-language model embeddings.
Within each cluster, we compute all pairwise cosine similarities and set a threshold cosine similarity above which data pairs are considered semantic duplicates.
Finally, from each group of semantic duplicates within a cluster, we keep the image with the lowest cosine similarity to the cluster centroid and remove the rest. We note that to determine duplicates, this method considers only the images and ignores the captions. A simplified pseudo code for SemDeDup is shown in Algorithm [A7](#A3.T7 "Table A7 ‣ Appendix C LAION-233M De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") in the appendix. We provide more details about the method in addition to experiments on choosing the value of k𝑘k in section [6](#S6 "6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication").

#### Utilizing pre-trained foundation Models

Our method makes use of pre-trained foundation models to embed data examples. Considering that there are many of these ready-to-use pre-trained models available to the public, we can use embeddings from these models to guide curation of other datasets. Pre-trained models like Vision Transformers [[33](#bib.bib33)] for vision tasks, OPT [[32](#bib.bib32)] for natural language and CLIP [[31](#bib.bib31)] for vision-language data have been used widely. In this work, we utilize pre-trained CLIP and OPT models for deduplication. In addition, in Section [6](#S6 "6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we show that one can effectively use an on-the-shelf model pre-trained on one dataset to prune another dataset resulting in a considerable training cost saving.

#### Clustering to reduce computation

The time complexity of naive de-duplication is 𝒪​(n2)𝒪superscript𝑛2\mathcal{O}(n^{2}) where n𝑛n is the number of data points, making this approach impractical for large web-scale data. For example, the LAION-440M dataset would require ≈1.9​x1017absent1.9superscriptx1017\approx 1.9\mathrm{x}10^{17} similarity computations. The k-means clustering step in SemDeDup reduces this complexity substantially from 𝒪​(n2)𝒪superscript𝑛2\mathcal{O}(n^{2}) to 𝒪​(n2/k)𝒪superscript𝑛2𝑘\mathcal{O}(n^{2}/k) assuming approximately uniform cluster size111Note that our choice of k𝑘k depends on n𝑛n, it is not a constant in the context of this complexity analysis.. This means we only require ≈4.6​x1012absent4.6superscriptx1012\approx 4.6\mathrm{x}10^{12} intra-cluster comparisons instead of ≈1.9​x1017absent1.9superscriptx1017\approx 1.9\mathrm{x}10^{17} across all pairs, a 5-order of magnitude improvement.

![Refer to caption](/html/2303.09540/assets/x3.png)


(a)

![Refer to caption](/html/2303.09540/assets/x4.png)


(b)

![Refer to caption](/html/2303.09540/assets/x5.png)


(c)

Figure 3: Extreme semantic redundancy in LAION-440M. (a) Fraction of data remaining as a function of deduplication threshold ϵitalic-ϵ\epsilon for LAION-440M. (b) Percentage of images in LAION-440M with at least one semantic duplicate as a function of ϵitalic-ϵ\epsilon. (c) Histogram of the number of within-cluster image pairs in LAION-440M at a given cosine similarity.

## 4 SemDeDup on LAION

If we consider pairs of data points to be semantic duplicates when their cosine similarity is at least 1−ϵ1italic-ϵ1-\epsilon, then ϵitalic-ϵ\epsilon can be thought of as a deduplication dissimilarity threshold, with increasing ϵitalic-ϵ\epsilon reflecting an increasingly coarser notion of semantic equality. We expect that low thresholds of ϵitalic-ϵ\epsilon will find semantic duplicates, while higher thresholds will allow semantically redundant data pairs as well.

To evaluate SemDeDup’s ability to discover semantic redundancy in multi-modal data, we train CLIP models on the LAION dataset (Section [3](#S3 "3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")). We first show that LAION contains extreme amounts of semantic redundancy (Section [4.2](#S4.SS2 "4.2 Extreme semantic redundancy at web-scale ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")) and provide examples of the semantic duplicates discovered by SemDeDup (Section [4.3](#S4.SS3 "4.3 What do semantic duplicates look like? ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")). Most critically, we demonstrate that removing the semantic duplicates discovered by SemDeDup has minimal to no impact on converged performance and increases learning speed (Section [4.4](#S4.SS4 "4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")).

### 4.1 Datasets and Training

#### The LAION dataset.

To train large-scale multi-modal models, we used the LAION dataset [[34](#bib.bib34)], an open multi-modal dataset containing up to 5 billion image-text pairs scraped from the web. LAION data were filtered using a pre-trained CLIP model to only retain image-text pairs with an embedding similarity greater than 0.28. Image-text pairs containing very short captions or small images were also removed. A simple de-duplication method based on the image url was also performed.

The majority of our experiments were performed on the LAION-440M filtered subset of LAION-2B introduced by [[14](#bib.bib14)]. This dataset was filtered using a Complexity, Action, and Text (CAT) filtering according to three criteria: (1) high enough caption complexity; (2) the caption must contain an action; (3) any text present in the image cannot substantially overlap with the caption.

To ensure this CAT filtered LAION-440M subset did not impact our results, we also performed experiments on unfiltered data derived from LAION. Much of the original LAION-400M subset [[35](#bib.bib35)] is no longer available due to broken urls, so we used a reduced version of the LAION-400M subset containing the 233 million data points we were able to collect, which we call LAION-233M.

#### CLIP training.

For CLIP training on LAION, we use the OpenCLIP implementation [[11](#bib.bib11)]. We use CLIP-ViT-Base/16 in all our experiments. The model has Vision Transformer Base (ViT-B-16) [[33](#bib.bib33)] as an image encoder and Text Transformer [[36](#bib.bib36)] as a text encoder.
We train all models with a global batch size of 33k image-caption pairs and fix the number of training epochs to 32 regardless of the dataset size. This results in training for a fewer number of iterations when training on deduplicated data, thereby achieving efficiency gains. We train with AdamW [[37](#bib.bib37)] and cosine learning rate schedule with warmup. The same peak learning rate of 5​x​10−45xsuperscript1045\mathrm{x}10^{-4} is used for all models. Table [A3](#A2.T3 "Table A3 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") shows training parameters we use for CLIP.

#### CLIP Evaluation

For CLIP evaluation we use zero-shot evaluation on 30 different datasets. Tables [A4](#A2.T4 "Table A4 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") and [A5](#A2.T5 "Table A5 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") in the Appendix list all the datasets we use for evaluation.

### 4.2 Extreme semantic redundancy at web-scale

How many semantically redundant pairs are there in LAION? Remarkably, we find that even tiny thresholds ϵitalic-ϵ\epsilon lead SemDeDup to remove large fractions of data in LAION440M (Fig. [3](#S3.F3 "Figure 3 ‣ Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")a), showing that LAION-440M contains large quantities of semantic duplicates. Surprisingly, 30%percent3030\% of images in LAION-440M have a semantic duplicate at the highly stringent distance threshold of ϵ=0.00095italic-ϵ0.00095\epsilon=0.00095, while 50%percent5050\% have a duplicate at the tight threshold of ϵ=0.03italic-ϵ0.03\epsilon=0.03 (Fig. [3](#S3.F3 "Figure 3 ‣ Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")c). Moreover, a histogram of pairwise cosine similarity in LAION-440M (Fig. [3](#S3.F3 "Figure 3 ‣ Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")d) reveals a high density of pairs at high cosine similarity, including a large contribution at 111, reflecting highly similar semantic duplicates. These results demonstrate that LAION-440M contains large amounts of semantic redundancy.

### 4.3 What do semantic duplicates look like?

What leads to semantic duplicates? In Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we show examples of semantic duplicates found at different thresholds ϵitalic-ϵ\epsilon. At extremely low values of ϵitalic-ϵ\epsilon we find perceptual duplicates, and at slightly higher values of ϵitalic-ϵ\epsilon, we find semantic duplicates, which are the same image but with distortions which evade exact de-duplication approaches such as different margins, crops, aspect ratios, and color filters, or slightly different peripheral details. Fig. [A10](#A4.F10 "Figure A10 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), and [A11](#A4.F11 "Figure A11 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") show examples of clusters that are semantically deduplicated at increasing levels of ϵitalic-ϵ\epsilon, clearly indicating more semantic diversity in deduplicated clusters as ϵitalic-ϵ\epsilon increases.

Many semantic duplicates are of products which may have been displayed on multiple e-commerce websites, each with a slightly different style. As a result, semantic duplicates often contain different, but highly similar captions. While most clusters contained 20-40% duplicates, there are several remarkable outliers in redundancy in LAION-440M (Fig. [A8](#A3.F8 "Figure A8 ‣ Appendix C LAION-233M De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")), including one cluster containing ≈307,000absent

307000\approx 307,000 copies of the European Union flag and another with ≈318,000absent

318000\approx 318,000 copies of an icon of “Image not found."

At higher levels of ϵitalic-ϵ\epsilon in Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), and [A9](#A4.F9 "Figure A9 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we find fewer semantic duplicates, which are generally derived from the same source image, and more pairs which exhibit semantic redundancy instead, in which the same concept is present, but not derived from the same image source. For example, semantically redundant pairs may contain different images of similar objects or scenes.

### 4.4 Training on semantically deduplicated data improves efficiency

If SemDeDup is effective at finding semantic duplicates, we should be able to remove these duplicates with a minimal performance impact. To test this, we train CLIP models on subsets of LAION-440M deduplicated at different thresholds ϵitalic-ϵ\epsilon, corresponding to smaller fractions of data as ϵitalic-ϵ\epsilon rises.

In Fig. [4](#S4.F4 "Figure 4 ‣ 4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") (a), we plot the top-1 zero-shot accuracy of our CLIP models on ImageNet-1k. Encouragingly, we found that SemDeDup can remove up to 37% of LAION440M with no performance drop, and 50% with minimal performance drop (<0.5%absentpercent0.5<0.5\%). In contrast, randomly removing data results in much larger drops. In Fig. [4](#S4.F4 "Figure 4 ‣ 4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") (b), we show the average zero-shot performance across 242424 tasks, finding that on average, performance increased on de-duplicated data. See Table [A4](#A2.T4 "Table A4 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for detailed performance on all 242424 tasks at 666 deduplication thresholds as well as 111 baseline and 444 random controls. See also Fig. [A4](#A2.F4 "Figure A4 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for performance on 242424 individual tasks.

We also evaluated out-of-distribution robustness on 666 datasets commonly used for this task: ImageNet-A, ImageNet-O [[38](#bib.bib38)], Imagenet-R [[39](#bib.bib39)], Imagenet-sketch [[40](#bib.bib40)], ImageNetV2 [[41](#bib.bib41)], and ObjectNet [[42](#bib.bib42)]. We again found that SemDeDup increased average performance over baseline when removing 37% of the data, and matched performance when 50% was removed as shown in Fig. [5](#S4.F5 "Figure 5 ‣ 4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") (a). See Table [A5](#A2.T5 "Table A5 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for detailed performance on 666 OOD tasks at 666 deduplication thresholds as well as 111 baseline and 444 random controls. We also note that SemDeDup outperforms random pruning on all individual out-of-distribution robustness datasets for all fractions of dataset kept. See Fig. [A5](#A2.F5 "Figure A5 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for performance on the 666 individual tasks.

Fig [6](#S4.F6 "Figure 6 ‣ 4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") shows SemDeDup performance across 303030 combined zero-shot and OOD tasks when removing 37%percent3737\% of the data, relative to a CLIP baseline trained on all the data. Remarkably, on about 202020 out of 303030 tasks, performance actually improves after removing pre-training data, whereas on all but about 333 of the remaining tasks performance is not substantially reduced. Our observation that SemDeDup can improve performance in many cases is consistent with prior work which has found that removing duplicates may improve performance by discouraging memorization [[43](#bib.bib43)].

We emphasize that SemDeDup achieves these results on LAION-440M, an already highly curated dataset derived from LAION-2B which was found to have similar performance despite the almost five-fold reduction in data [[14](#bib.bib14)]. However, to ensure that this curated subset did not bias our results, we also evaluated on LAION-233M, an uncurated subset of LAION-2B, finding qualitatively similar results (Fig. [A6](#A3.F6 "Figure A6 ‣ Appendix C LAION-233M De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")).

Because SemDeDup reduces the number of training points, it enables substantially faster training. In Fig. [5](#S4.F5 "Figure 5 ‣ 4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") (b), we plot the top-1 zero-shot accuracy on ImageNet-1k as a function of the number of iterations for different deduplication thresholds ϵitalic-ϵ\epsilon. Notably, models trained on deduplicated data reach convergence in substantially fewer iterations.

Why do models trained on uncurated data exhibit slower learning? We posit that successive learning iterations involving semantic duplicates yield redundant information, thereby wasting valuable computation on data points that are highly similar to those the model has already seen. By removing these semantic duplicates, we increase the fraction of data points which provide a marginal information gain to the model, thereby increasing learning speed [[13](#bib.bib13)].

![Refer to caption](/html/2303.09540/assets/x6.png)


(a)

![Refer to caption](/html/2303.09540/assets/x7.png)


(b)

Figure 4: SemDeDup allows better average zero-shot accuracy across 24 tasks with less data and faster pre-training. (a): Performance of SemDeDup (blue) and random pruning (orange) for different amounts of retained data. Down to using only 50% of LAION-440M for pre-training CLIP, we are able to match the zero-shot ImageNet accuracy of the baseline model trained on 100% of the data (black dashed line) with a small drop of 0.47% only, while we outperform the baseline model with only 63% of data. (b): Average zero-shot performance for CLIP measured on 24 datasets. Average performance improves across 242424 tasks down to 63%percent6363\% of the pre-training data, yielding better performance with almost 1.6×1.6\times faster pre-training.



![Refer to caption](/html/2303.09540/assets/x8.png)


(a)

![Refer to caption](/html/2303.09540/assets/x9.png)


(b)

Figure 5: SemDeDup allows better average performance across 6 ImageNet OOD tasks with less data and faster pre-training. (a) zeroshot validation accuracy averaged over 6 ImageNet-1k OOD tasks for CLIP models pre-trained on deduplicated LAION data with different thresholds ϵitalic-ϵ\epsilon. We outperform the baseline model with only 63% of pre-training data from LAION-440M.
(b) We track zeroshot ImageNet-1K performance as a function of LAION-440M pre-training iterations at different deduplication thresholds. The models trained on smaller deduplicated datasets actually learn faster, thereby allowing them to converge to almost baseline performance (black dashed line) in far fewer iterations.



![Refer to caption](/html/2303.09540/assets/x10.png)

![Refer to caption](/html/2303.09540/assets/x11.png)

Figure 6: SemDeDup improves zeroshot and OOD performance in many tasks with less pre-training. A comparison of zeroshot evaluation performance between our CLIP model trained on 63% of LAION-440M after de-duplication to a baseline CLIP model trained on 100% of the data (left), and OpenAI CLIP [[31](#bib.bib31)] (right) on 303030 tasks. The green bars show when SemDeDup outperforms the baseline model.

## 5 SemDeDup on Natural Language

### 5.1 Methods

We train language models on deduplicated versions of the C4 dataset [[18](#bib.bib18)]. Since pre-training large language models on the entire C4 corpus is beyond our compute budget, we train on subsets of this data whose sizes are compute optimal given model size as per [[8](#bib.bib8)]. We use the OPT model and training configurations [[32](#bib.bib32)] to train 125M and 1.3B parameter models (see Table 1 in [[32](#bib.bib32)] for full specifications). We use the original number of warmup updates but adjust the learning rate schedule such that all training runs anneal learning rate to 0 by the end of the training — this allows for fair comparisons of model performances across different dataset sizes.
For 1.3B model size experiments, we increase the number of warmup updates to 5550 and reduce the peak learning rate to 6​x​10−56xsuperscript1056\mathrm{x}10^{-5} to stabilize training.

We evaluate our trained language models on two independent validation sets: the validation text corpora used by OPT [[32](#bib.bib32)] (referred to as "opt\_valid") and a random sample of the instruction finetuning corpus used to train the OPT-IML family of models [[44](#bib.bib44)], composed of verbalized prompts corresponding to a wide range of NLP tasks and their solutions (referred to as "prompts\_with\_answers").

To perform SemDeDup, we pass documents through the open-sourced pre-trained 125M OPT model [[32](#bib.bib32)] and save the last layer embedding for the last token in the document. We then apply the same method described in Section [3](#S3.SS0.SSS0.Px3 "Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") with 𝒦=11000𝒦11000\mathcal{K}=11000 to cluster these embeddings. We compare to random pruning and the NearDup method described in [[43](#bib.bib43)]. Note that the deduplication threshold values associated with different fractions of data remaining change compared to LAION-440M, as seen in Fig. [A17](#A6.F17 "Figure A17 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication").

### 5.2 Results on Language Modeling

In Fig. [7](#S5.F7 "Figure 7 ‣ 5.3 What is being pruned in language data? ‣ 5 SemDeDup on Natural Language ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we show the performance of SemDeDup versus random pruning. We observe that SemDeDup significantly outperforms random pruning as measured by perplexity on prompts\_with\_answers and average opt\_valid performance. For a breakdown of performance on individual validation sets in opt\_valid, see Fig. [A20](#A6.F20 "Figure A20 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") where we observe that SemDeDup beats random pruning on every single validation set in opt\_valid.

Training on less data for one epoch naturally causes performance to decrease. Thus, we also explore whether continuing to train on the same smaller pruned datasets for more epochs will match the performance of a baseline model trained on a larger dataset. In Fig. [8](#S5.F8 "Figure 8 ‣ 5.3 What is being pruned in language data? ‣ 5 SemDeDup on Natural Language ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we train on datasets pruned with SemDeDup, but perform the same number of total training steps as the baseline model on the larger dataset (which was trained for 111 epoch). This causes the model to do multiple epochs over the pruned dataset. We observe that by training for multiple epochs over significantly pruned datasets we can reach the performance of a single-epoch run on the full dataset using 10-15% less compute. This is similar to the finding in Section [4.4](#S4.SS4 "4.4 Training on semantically deduplicated data improves efficiency ‣ 4 SemDeDup on LAION ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). Notably, this efficiency gain is larger at higher pruning percentages, indicating that more aggressive pruning can yield more efficiency gains. This trend generally holds across the individual validation sets in opt\_valid (see Fig. [A21](#A6.F21 "Figure A21 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")).

On the C4 validation set, we observe that SemDeDup still outperforms random pruning in Fig. [A18](#A6.F18 "Figure A18 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). In Table [A12](#A5.F12 "Figure A12 ‣ Appendix E Perplexity Values for SemDeDup on Language Modeling ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we compare SemDeDup to the NearDup baseline from [[19](#bib.bib19)]. We observe that NearDup and SemDeDup have comparable performance as is expected, because with 4% pruning there is very little change to the underlying dataset.

### 5.3 What is being pruned in language data?

In Fig. [A22](#A6.F22 "Figure A22 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") and Fig. [A23](#A6.F23 "Figure A23 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we choose specific clusters and show a random sample of documents retained in the cluster after performing SemDeDup for different values of ϵitalic-ϵ\epsilon. In Fig. [A22](#A6.F22 "Figure A22 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we observe that at low values of ϵitalic-ϵ\epsilon, we find semantic duplicates in the form of templated text, where typically few words (e.g. a geographic location or a name) is changed. This successfully evades exact-string deduplication methods but contains highly redundant information as seen in Fig. [A22](#A6.F22 "Figure A22 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). In Fig. [A23](#A6.F23 "Figure A23 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we show an example of a cluster with semantically redundant duplicates — most examples in this cluster are advertisements about Nike shoes. These examples are not necessarily templated text or have exact string matches, but are highly redundant nonetheless. We see in Fig. [A23](#A6.F23 "Figure A23 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") that at more aggressive pruning (i.e. higher ϵitalic-ϵ\epsilon) these semantically redundant duplicates get pruned. We note that exact string duplicates (i.e.“perceptual duplicates for text") are rare since duplicate occurrences of any three-sentence spans were removed in C4 already.

![Refer to caption](/html/2303.09540/assets/x12.png)

![Refer to caption](/html/2303.09540/assets/x13.png)

Figure 7: SemDeDup applied to C4. The x-axis corresponds to different percents of data kept, and the y-axis represents the perplexity on validation sets described in Section [5.1](#S5.SS1 "5.1 Methods ‣ 5 SemDeDup on Natural Language ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") (lower is better). Each point is a separate 125M model trained on one-pass of its respective pruned dataset (mean and standard deviation across 3 random training seeds). The green point represents a 125M model trained on a version of C4 deduplicated via the NearDup method [[19](#bib.bib19)]. Note that NearDup (the single green point) keeps 96.1% of the data. SemDeDup can match this baseline performance while keeping only 80% of the data (see Table [A13](#A5.F13 "Figure A13 ‣ Appendix E Perplexity Values for SemDeDup on Language Modeling ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for numerical comparison).



![Refer to caption](/html/2303.09540/assets/x14.png)

![Refer to caption](/html/2303.09540/assets/x15.png)

Figure 8: SemDeDup allows compute efficiency gains by training on much smaller datasets for slightly longer. We prune datasets via SemDeDup and continue training past one epoch until we reach baseline model perplexity. The x-axis is the percentage of data kept, and the y-axis is the percentage of FLOPs saved. For example, training on the 80% pruned dataset reaches baseline model perplexity on prompts\_with\_answer in  95.0% of the baseline training, saving  5.0% compute. Mean and standard deviation provided across 3 random training seeds.

## 6 Analysis of hyperparameter choices

### 6.1 Number of k-means clusters for SemDeDup

Here we study the impact of changing the number of clusters k𝑘k in the k-means clustering step in SemDeDup described in section [3](#S3.SS0.SSS0.Px1 "Defining and identifying semantic duplicates ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). In all our experiments in the main paper, we set k𝑘k = 50,000 for the LAION dataset and k𝑘k = 11,000 for the C4 dataset. To study the impact of the k𝑘k on the performance, we deduplicate LAION440M using different values for k𝑘k and train different CLIP models on the deduplicated data. We compare three values for k𝑘k (70,000, 50,000, and 10,000) when deduplicating LAION440M to 40% of its size. As we see in Table [1](#S6.T1 "Table 1 ‣ 6.1 Number of k-means clusters for SemDeDup ‣ 6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") the exact choice of k𝑘k has a very small impact on performance as measured by the zeroshot accuracy on ImageNet with a small improvement in the top1 accuracy as k𝑘k increases.

The key intuition is that the choice of k𝑘k implements a tradeoff in the probability of recovering all semantic duplicates of any data point, and the computational complexity of doing so. For example, assuming k-means finds equal cluster sizes, each data point will lie in a cluster of size N/k𝑁𝑘N/k, and we are only searching for ϵitalic-ϵ\epsilon-nearest neighbors (with cosine similarity > 1−ϵ1italic-ϵ1-\epsilon) within each cluster. As k𝑘k decreases, cluster size N/k𝑁𝑘N/k increases, and the error probability of substantially many ϵitalic-ϵ\epsilon nearest neighbors of a data point lying outside it’s own cluster decreases, while the computational complexity of searching for all nearest neighbors within the cluster increases. As long as k𝑘k is small enough relative to the total dataset size N𝑁N, so that N/k𝑁𝑘N/k is large enough to contain most nearest neighbors of each data point, the performance of SemDeDup should be robust to the choice of k𝑘k.

Table 1: Performance of CLIP when keeping 40% of LAION440M as a function of the number of k-means clusters k𝑘k used for SemDeDup. SemDeDup is robust to the choice of k𝑘k and the impact on the zeroshot accuracy on ImageNet is small with slight performance improvement as we increase k𝑘k.

| Metric / Num. of Clusters | 70K Clusters | 50K Clusters | 10K Clusters |
| --- | --- | --- | --- |
| Top1 zeroshot IN Acc. | 67.11 | 66.90 | 66.56 |
| Top5 zeroshot IN Acc. | 90.96 | 90.74 | 91.04 |

### 6.2 Pre-trained models for extracting embeddings

As we describe in section [3](#S3.SS0.SSS0.Px1 "Defining and identifying semantic duplicates ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), SemDeDup clusters the example embeddings extracted from a pre-trained foundation model and uses them for deduplication. To study the effect of the pre-training dataset of the foundation model on SemDeDup we deduplicate LAION440M using an OpenAI CLIP model [[31](#bib.bib31)] pre-trained on a different dataset than LAION. We use the Open AI CLIP ViT-Base model pre-trained on a private dataset of 400 million image-caption pairs. We use the embeddings from this model to deduplicate LAION440M dataset to 40% of its size. As we see in Table [2](#S6.T2 "Table 2 ‣ 6.2 Pre-trained models for extracting embeddings ‣ 6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), using Open AI CLIP model for extracting embeddings has a negligible impact on the performance.

Table 2: The impact of the foundation model used for extracting embeddings. Using a foundation model pre-trained on a different (and private) dataset has no impact on the performance. The table shows the performance when training OpenCLIP on 40% of LAION440M dataset. In each column in the table, the dataset is deduplicated by SemDeDup using embeddings from a different model.

| Metric / Model Used for Extracting Embeddings | CLIP Pre-trained on LAION440M | OpenAI CLIP [[31](#bib.bib31)] Pre-trained on Private 400M dataset |
| --- | --- | --- |
| Top1 zeroshot IN Acc. After Training on DeDup Data | 66.90 | 66.96 |
| Top5 zeroshot IN Acc. After Training on DeDup Data | 90.74 | 90.80 |

### 6.3 Different strategies for choosing which semantic duplicates to keep

In section [3](#S3.SS0.SSS0.Px1 "Defining and identifying semantic duplicates ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") and Algorithm [A7](#A3.T7 "Table A7 ‣ Appendix C LAION-233M De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we describe the steps for deduplication with SemDeDup. From each group of duplicates (the circles in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")), we keep the example with the lowest cosine similarity to the cluster centroid in the embedding space. This is the default setting for all experiments we run unless otherwise mentioned. In Table [3](#S6.T3 "Table 3 ‣ 6.3 Different strategies for choosing which semantic duplicates to keep ‣ 6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we study the strategy we follow to choose the example to keep from each group of duplicates. We train three CLIP models on 40% of LAION440M deduplicated by SemDeDup for 32 epochs. We try three options for choosing the examples we keep 1) keeping examples with low similarity to centroids, 2) keeping random examples, and 3) keeping examples with high similarity to cluster centroids. We obverse that the difference between the three methods in zero-shot accuracy on ImageNet is negligible.

Table 3:  Different strategies to choose the example to keep from each group of duplicates.

| Metric / Examples to Keep | Examples with low similarity to centroids | Random examples | Examples with high similarity to centroids |
| --- | --- | --- | --- |
| Top1 zeroshot IN Acc. | 66.90 | 66.90 | 66.73 |
| Top5 zeroshot IN Acc. | 90.74 | 90.95 | 90.82 |

### 6.4 Training on deduplicated data for more iterations improves performance

Training on deduplicated data comes with the advantage that we train for fewer iterations under the match-epochs setting. For example, training on 50% of LAION440M for the same number of epochs as the baseline model (100% of the data) means that we train for only 50% of the number of training iterations. We find that we can achieve a good trade-off between performance and training speed when training on deduplicated data. We show that training on deduplicated LAION440M for more iterations improves the accuracy while still being below the number of iterations we train the baseline model for. In Table [4](#S6.T4 "Table 4 ‣ 6.4 Training on deduplicated data for more iterations improves performance ‣ 6 Analysis of hyperparameter choices ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), we show results for different CLIP models, trained on 50% of LAION440M, for a different number of training iterations. We see that by continuing training the model until we reach 75% of the iterations relative to the baseline model, we outperform the baseline model on not only ImageNet, but also on average accuracy over 24 datasets, and on the 666 out-of-distribution datasets.

Table 4: By training on only 50% of LAION440M, deduplicated using SemDeDup, we perform better than training on whole LAION440M (baseline100) with 62.5% or 75% of the number of training iterations used for training the baseline model. The table shows zeroshot Top1 accuracy.

| Model | IN Acc | Avg. Acc (24 datasets) | Avg. OOD (6 datasets) |
| --- | --- | --- | --- |
| 100% data, 100% iters (Baseline100) | 68.74 | 54.12 | 55.94 |
| 50% data, 50% iters | 68.27 | 54.59 | 55.87 |
| 50% data, 62.5% iters | 68.33 | 55.07 | 56.38 |
| 50% data, 75% iters | 69.21 | 55.07 | 56.36 |

### 6.5 Choosing the deduplication threshold ϵitalic-ϵ\epsilon

We tune the deduplication threshold ϵitalic-ϵ\epsilon for each dataset manually to get the desired deduplicated dataset size. To do that, we first run the clustering step of SemDeDup. Then we sample 10% of the clusters and tune ϵitalic-ϵ\epsilon on them. We found that using only 10% of clusters gives a good approximation of the final dataset size.
We notice that the relationship between ϵitalic-ϵ\epsilon and the deduplicated dataset size is semi-linear for both LAION and C4 datasets (see Fig. [3](#S3.F3 "Figure 3 ‣ Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), [A1](#A1.F1 "Figure A1 ‣ A.1 Number of k-means Clusters for SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), and [A17](#A6.F17 "Figure A17 ‣ Appendix F Qualitative Examples of SemDeDup on C4 ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")). When tuning ϵitalic-ϵ\epsilon we start with two values and run SemDeDup on 10% of the clusters (the time needed for this step is a few minutes. See the DeDup. Time column in Table [A2](#A1.T2 "Table A2 ‣ A.2 Estimating The Fraction of Duplicates Detected By SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") ). Then we linearly interpolate the two values of ϵitalic-ϵ\epsilon knowing their correspondence dataset size and the target dataset size to get a better value for ϵitalic-ϵ\epsilon. In Fig. [A1](#A1.F1 "Figure A1 ‣ A.1 Number of k-means Clusters for SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we plot the duplicated dataset size as a function of ϵitalic-ϵ\epsilon for different values of the number of clusters k𝑘k used. We show that k𝑘k has a small impact on the value ϵitalic-ϵ\epsilon only when the duplicated dataset size is less than 50%.

## 7 Compute cost of running SemDeDup

We report in Table [5](#S7.T5 "Table 5 ‣ 7 Compute cost of running SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") the cost of running SemDeup on LAION440M in GPU hours. We see in the table that the overhead of deduplicating LAION440M doesn’t exceed 1% of the training cost in GPU hours. This results in substantial savings in the overall cost after deduplication. For example, training on 50% of the data saves 50% of the training cost while requiring only 1% of the training cost for deduplication.
We also show in Table [A2](#A1.T2 "Table A2 ‣ A.2 Estimating The Fraction of Duplicates Detected By SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") the time needed for deduplicating LAION440M dataset using SemDeDup using 8 GPUs for clustering and 64 GPUs for CLIP training. Our implementation for SemDeDup parallelizes the operations across devices to speed up the deduplication. The table also shows how the time changes as we change the number of clusters.

However, we should note that the computational cost of SemDeDup can be amortized across the efficiency gains it can generate in training many downstream models by many other groups. For example, its typical use case would be to take a large web-scaled dataset, and semantically deduplicate it once, resulting in a much smaller foundation dataset [[13](#bib.bib13)] that can be widely disseminated to the community. Then many different groups can train many different foundation models on this deduplicated foundation dataset, and all these groups will reap the training efficiency gains conferred by a less redundant smaller dataset. Thus the computational cost of finding the dataset can be amortized across the efficiency gains achieved on many downstream training runs, in direct analogy to how the computational cost of training a foundation model can be amortized across the computational efficiency gains with which it achieves high zero-shot or fine-tuning performance on many downstream applications.

Table 5: SemDeDup requires much fewer GPU hours than training a CLIP ViT-B-16 model on LAION440M for one epoch. When using 50K clusters, it requires only 0.29 of the GPU hours needed for one epoch of training on 100% of the data. This is equivalent to 0.0091 of the complete training cost in GPU hours.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Num. SemDeDup Clusters / Cost | GPU Hours For Training CLIP on 100% of LAION440M for 32 Epochs | GPU Hours For Training on 50% of DeDup. Data | SemDeDup Overhead in GPU Hours | SemDeDup Overhead / 1 Epoch Training GPU Hours | SemDeDup Overhead / 32 Epochs Training GPU Hours |
| 10K Clusters | 11541 | 5770.5 | 163.5 | 0.43 | 0.0132 |
| 25K Clusters | 11541 | 5770.5 | 101.2 | 0.26 | 0.0082 |
| 50K Clusters | 11541 | 5770.5 | 110.3 | 0.29 | 0.0091 |
| 70K Clusters | 11541 | 5770.5 | 103.6 | 0.27 | 0.0084 |

## 8 Discussion

We introduced SemDeDup, a simple yet tractable and effective method which leverages pre-trained embeddings to remove semantic duplicates which are highly semantically similar but not identical. Removing semantic duplicates improves learning speed and out-of-distribution performance while providing efficiency gains of up to 50% on the largely uncurated LAION and 15% on the partially curated C4. SemDeDup demonstrates the importance of data quality and the potential of data curation to dramatically improve training efficiency.

#### Limitations.

While SemDeDup does an effective job of removing semantic duplicates and some semantically redundant data points, it is only one way to remove uninformative data points. In particular, this work does not capture many aspects of semantic redundancy, nor does it address removal of bad or misleading data, all of which can likely be exploited to make substantial further reductions to dataset size without sacrificing performance.

SemDeDup also requires access to a pre-trained embedding model relevant to the domain of interest, which may pose a problem for entirely novel domains unrelated to the wide array of publicly available pre-trained models. However, for most domains, pre-trained models are readily available, and many such models have been shown to generalize to related domains. We, therefore, expect that this limitation will only apply to a small fraction of the practical use cases for SemDeDup.

In LAION, we identified semantic duplicates based only on image data, but we ignored the caption information. Leveraging this information may lead to the identification of further semantic duplicates.

Our results on C4 showcase the potential of SemDeDup for NLP, but the gains were more modest due to the partially curated nature of C4 which has fewer duplicates than LAION. We also trained small models relative to the best models. It is possible that results may change with scale, though following [[13](#bib.bib13)], it is likely increasing scale would further improve the benefits of data curation.

Overall, the optimal data pruning policy for finding the smallest possible data subset under computational tractability and performance constraints remains, as ever, an extremely difficult open question. However, the remarkable efficacy of SemDeDup, especially given its underlying simplicity and scalability, suggests that the removal of semantic duplicates may well be an important prerequisite for any more sophisticated data pruning algorithm, especially when working with modern, large, highly uncurated, web-scale datasets.

## Acknowledgements

We thank Mido Assran and Mansheej Paul for discussions. We also thank Mitchell Wortsman for support with OpenCLIP. We also thank Armen Aghajanyan for suggestions for handling training instability in our language model experiments.

## References

* Hestness et al. [2017]

  J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad,
  M. Patwary, M. Ali, Y. Yang, and Y. Zhou.
  Deep learning scaling is predictable, empirically.
  *arXiv preprint arXiv:1712.00409*, 2017.
* Kaplan et al. [2020]

  J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child,
  S. Gray, A. Radford, J. Wu, and D. Amodei.
  Scaling laws for neural language models.
  *arXiv preprint arXiv:2001.08361*, 2020.
* Henighan et al. [2020]

  T. Henighan, J. Kaplan, M. Katz, M. Chen, C. Hesse, J. Jackson, H. Jun, T. B.
  Brown, P. Dhariwal, S. Gray, et al.
  Scaling laws for autoregressive generative modeling.
  *arXiv preprint arXiv:2010.14701*, 2020.
* Rosenfeld et al. [2020]

  J. S. Rosenfeld, A. Rosenfeld, Y. Belinkov, and N. Shavit.
  A constructive prediction of the generalization error across scales.
  *International Conference on Learning Representations*, 2020.
  URL <https://openreview.net/forum?id=ryenvpEKDr>.
* Gordon et al. [2021]

  M. A. Gordon, K. Duh, and J. Kaplan.
  Data and parameter scaling laws for neural machine translation.
  In *Proceedings of the 2021 Conference on Empirical Methods in
  Natural Language Processing*, pages 5915–5922, Online and Punta Cana,
  Dominican Republic, Nov. 2021. Association for Computational Linguistics.
* Hernandez et al. [2021]

  D. Hernandez, J. Kaplan, T. Henighan, and S. McCandlish.
  Scaling laws for transfer.
  *arXiv preprint arXiv:2102.01293*, 2021.
* Zhai et al. [2021]

  X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer.
  Scaling vision transformers.
  *arXiv preprint arXiv:2106.04560*, 2021.
* Hoffmann et al. [2022]

  J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford,
  D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland,
  K. Millican, G. v. d. Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan,
  E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre.
  Training compute-optimal large language models, 2022.
  URL <https://arxiv.org/abs/2203.15556>.
* Schuhmann et al. [2022]

  C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti,
  T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al.
  Laion-5b: An open large-scale dataset for training next generation
  image-text models.
  *arXiv preprint arXiv:2210.08402*, 2022.
* Radford et al. [2021]

  A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
  A. Askell, P. Mishkin, J. Clark, et al.
  Learning transferable visual models from natural language
  supervision.
  In *International Conference on Machine Learning*, pages
  8748–8763. PMLR, 2021.
* Ilharco et al. [2021]

  G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave,
  V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and
  L. Schmidt.
  Openclip, July 2021.
  URL <https://doi.org/10.5281/zenodo.5143773>.
  If you use this software, please cite it as below.
* Aghajanyan et al. [2023]

  A. Aghajanyan, L. Yu, A. Conneau, W.-N. Hsu, K. Hambardzumyan, S. Zhang,
  S. Roller, N. Goyal, O. Levy, and L. Zettlemoyer.
  Scaling laws for generative mixed-modal language models, 2023.
  URL <https://arxiv.org/abs/2301.03728>.
* Sorscher et al. [2022]

  B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli, and A. S. Morcos.
  Beyond neural scaling laws: beating power law scaling via data
  pruning.
  In *Neural Information Processing Systems (NeurIPS)*, June
  2022.
* Radenovic et al. [2023]

  F. Radenovic, A. Dubey, A. Kadian, T. Mihaylov, S. Vandenhende, Y. Patel,
  Y. Wen, V. Ramanathan, and D. Mahajan.
  Filtering, distillation, and hard negatives for vision-language
  pre-training.
  *arXiv preprint arXiv:2301.02280*, 2023.
* Feldman and Zhang [2020]

  V. Feldman and C. Zhang.
  What neural networks memorize and why: Discovering the long tail via
  influence estimation.
  *Adv. Neural Inf. Process. Syst.*, 33:2881–2891,
  2020.
* Liao [2022]

  Y. Liao.
  *Dataset Deduplication with Datamodels*.
  PhD thesis, Massachusetts Institute of Technology, May 2022.
* Thomee et al. [2016]

  B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth,
  and L.-J. Li.
  YFCC100M: the new data in multimedia research.
  *Commun. ACM*, 59(2):64–73, Jan. 2016.
* Raffel et al. [2019]

  C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
  W. Li, and P. J. Liu.
  Exploring the limits of transfer learning with a unified text-to-text
  transformer, 2019.
  URL <https://arxiv.org/abs/1910.10683>.
* Lee et al. [2021]

  K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and
  N. Carlini.
  Deduplicating training data makes language models better, 2021.
  URL <https://arxiv.org/abs/2107.06499>.
* Broder [1997]

  A. Broder.
  On the resemblance and containment of documents.
  06 1997.
  [doi:10.1109/SEQUEN.1997.666900](http://dx.doi.org/10.1109/SEQUEN.1997.666900).
* Rae et al. [2021]

  J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song,
  J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan,
  J. Menick, A. Cassirer, R. Powell, G. v. d. Driessche, L. A. Hendricks,
  M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato,
  J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen,
  S. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan,
  M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh,
  E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau,
  M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen,
  Z. Gong, D. Toyama, C. d. M. d’Autume, Y. Li, T. Terzi, V. Mikulik,
  I. Babuschkin, A. Clark, D. d. L. Casas, A. Guy, C. Jones, J. Bradbury,
  M. Johnson, B. Hechtman, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart,
  S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway,
  L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving.
  Scaling language models: Methods, analysis and insights from training
  gopher, 2021.
  URL <https://arxiv.org/abs/2112.11446>.
* Kandpal et al. [2022]

  N. Kandpal, E. Wallace, and C. Raffel.
  Deduplicating training data mitigates privacy risks in language
  models, 2022.
  URL <https://arxiv.org/abs/2202.06539>.
* Silcock et al. [2022]

  E. Silcock, L. D’Amico-Wong, J. Yang, and M. Dell.
  Noise-Robust De-Duplication at scale.
  Dec. 2022.
* Choi et al. [2022]

  W.-S. Choi, D.-S. Han, H. Lee, J. Park, and B.-T. Zhang.
  DUEL: Adaptive duplicate elimination on working memory for
  Self-Supervised learning.
  Oct. 2022.
* Guo et al. [2022]

  C. Guo, B. Zhao, and Y. Bai.
  DeepCore: A comprehensive library for coreset selection in deep
  learning.
  Apr. 2022.
* Phillips [2016]

  J. M. Phillips.
  Coresets and sketches.
  Jan. 2016.
* Toneva et al. [2019]

  M. Toneva, A. Sordoni, R. T. des Combes, A. Trischler, Y. Bengio, and G. J.
  Gordon.
  An empirical study of example forgetting during deep neural network
  learning.
  In *ICLR*, 2019.
* Paul et al. [2021]

  M. Paul, S. Ganguli, and G. K. Dziugaite.
  Deep learning on a data diet: Finding important examples early in
  training.
  *Adv. Neural Inf. Process. Syst.*, 34, Dec. 2021.
* Chitta et al. [2021]

  K. Chitta, J. M. Álvarez, E. Haussmann, and C. Farabet.
  Training data subset search with ensemble active learning.
  *IEEE Trans. Intell. Transp. Syst.*, pages 1–12, 2021.
* Meding et al. [2022]

  K. Meding, L. M. S. Buschoff, R. Geirhos, and F. A. Wichmann.
  Trivial or impossible—dichotomous data difficulty masks model
  differences (on ImageNet and beyond).
  In *International Conference on Learning Representations*, 2022.
  URL <https://openreview.net/forum?id=C_vsGwEIjAr>.
* Radford et al. [2021]

  A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
  A. Askell, P. Mishkin, J. Clark, et al.
  Learning transferable visual models from natural language
  supervision.
  In *International conference on machine learning*, pages
  8748–8763. PMLR, 2021.
* Zhang et al. [2022]

  S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab,
  X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig,
  P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer.
  Opt: Open pre-trained transformer language models, 2022.
  URL <https://arxiv.org/abs/2205.01068>.
* Dosovitskiy et al. [2020]

  A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
  T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al.
  An image is worth 16x16 words: Transformers for image recognition at
  scale.
  *arXiv preprint arXiv:2010.11929*, 2020.
* Schuhmann et al. [2022]

  C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti,
  T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al.
  Laion-5b: An open large-scale dataset for training next generation
  image-text models.
  *arXiv preprint arXiv:2210.08402*, 2022.
* Schuhmann et al. [2021]

  C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta,
  T. Coombes, J. Jitsev, and A. Komatsuzaki.
  Laion-400m: Open dataset of clip-filtered 400 million image-text
  pairs.
  *arXiv preprint arXiv:2111.02114*, 2021.
* Vaswani et al. [2017]

  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  Ł. Kaiser, and I. Polosukhin.
  Attention is all you need.
  *Advances in neural information processing systems*, 30, 2017.
* Loshchilov and Hutter [2017]

  I. Loshchilov and F. Hutter.
  Decoupled weight decay regularization.
  *arXiv preprint arXiv:1711.05101*, 2017.
* Hendrycks et al. [2021a]

  D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song.
  Natural adversarial examples.
  *CVPR*, 2021a.
* Hendrycks et al. [2021b]

  D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai,
  T. Zhu, S. Parajuli, M. Guo, D. Song, J. Steinhardt, and J. Gilmer.
  The many faces of robustness: A critical analysis of
  out-of-distribution generalization.
  *ICCV*, 2021b.
* Wang et al. [2019]

  H. Wang, S. Ge, Z. Lipton, and E. P. Xing.
  Learning robust global representations by penalizing local predictive
  power.
  In *Advances in Neural Information Processing Systems*, pages
  10506–10518, 2019.
* Recht et al. [2019]

  B. Recht, R. Roelofs, L. Schmidt, and V. Shankar.
  Do ImageNet classifiers generalize to ImageNet?
  In K. Chaudhuri and R. Salakhutdinov, editors, *Proceedings of
  the 36th International Conference on Machine Learning*, volume 97 of
  *Proceedings of Machine Learning Research*, pages 5389–5400. PMLR,
  09–15 Jun 2019.
  URL <https://proceedings.mlr.press/v97/recht19a.html>.
* Barbu et al. [2019]

  A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and
  B. Katz.
  Objectnet: A large-scale bias-controlled dataset for pushing the
  limits of object recognition models.
  In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural
  Information Processing Systems*, volume 32. Curran Associates, Inc., 2019.
  URL
  <https://proceedings.neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf>.
* Lee et al. [2021]

  K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and
  N. Carlini.
  Deduplicating training data makes language models better.
  *arXiv preprint arXiv:2107.06499*, 2021.
* Iyer et al. [2022]

  S. Iyer, X. V. Lin, R. Pasunuru, T. Mihaylov, D. Simig, P. Yu, K. Shuster,
  T. Wang, Q. Liu, P. S. Koura, X. Li, B. O’Horo, G. Pereyra, J. Wang,
  C. Dewan, A. Celikyilmaz, L. Zettlemoyer, and V. Stoyanov.
  Opt-iml: Scaling language model instruction meta learning through the
  lens of generalization, 2022.
  URL <https://arxiv.org/abs/2212.12017>.

## Appendix A Additional Analysis

### A.1 Number of k-means Clusters for SemDeDup

To further assess the impact of changing the value of k𝑘k we measure the intersection between datasets deduplicated by SemDeDup using different values for k𝑘k. Let DA={a1,a2,…,aN}subscript𝐷𝐴subscript𝑎1subscript𝑎2…subscript𝑎𝑁D\_{A}=\{a\_{1},a\_{2},...,a\_{N}\} and DB={b1,b2,…,bN}subscript𝐷𝐵subscript𝑏1subscript𝑏2…subscript𝑏𝑁D\_{B}=\{b\_{1},b\_{2},...,b\_{N}\} be two datasets of the same size N𝑁N. We define the percentage of intersection I𝐼I between DAsubscript𝐷𝐴D\_{A} and DBsubscript𝐷𝐵D\_{B} in equation [1](#A1.E1 "Equation 1 ‣ A.1 Number of k-means Clusters for SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") as the percentage of data points that appear in both datasets relative to the dataset size N𝑁N. Note that I​(DA,DA)=100%𝐼subscript𝐷𝐴subscript𝐷𝐴percent100I(D\_{A},D\_{A})=100\%.
  
We find that deduplicating LAION440M dataset to 72% of its size using any value of k𝑘k values (10000, 25000, 50000, 70000) results in almost the same dataset with only 3% of the examples replaced when changing k𝑘k. This is induced by the 97% percentage of intersection I𝐼I value between any pair of datasets deduplicated using two different values for k𝑘k. We show in Fig. [A2](#A1.F2 "Figure A2 ‣ A.1 Number of k-means Clusters for SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") the percentage of intersection ratio between different datasets when changing the number of clusters k𝑘k at different deduplication thresholds ϵitalic-ϵ\epsilon.
  
We also show in figure [A1](#A1.F1 "Figure A1 ‣ A.1 Number of k-means Clusters for SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") that by using the same deduplication threshold value ϵitalic-ϵ\epsilon we get almost the same deduplicated dataset size for different values for k𝑘k.

  

![Refer to caption](/html/2303.09540/assets/x16.png)

Figure A1: Deduplicated dataset size as a function of the deduplication threshold for different values of k-means clusters k𝑘k. Note that the range in the dataset size is 0.003% when ϵitalic-ϵ\epsilon is 0.00095 and 2% when ϵitalic-ϵ\epsilon is 0.26.

|  |  |  |  |
| --- | --- | --- | --- |
|  | I​(DA,DB)=100∗|DA∩DB|N𝐼subscript𝐷𝐴subscript𝐷𝐵100subscript𝐷𝐴subscript𝐷𝐵𝑁I(D\_{A},D\_{B})=100\*\frac{\lvert D\_{A}\cap D\_{B}\rvert}{N} |  | (1) |

  

![Refer to caption](/html/2303.09540/assets/x17.png)
![Refer to caption](/html/2303.09540/assets/x18.png)
![Refer to caption](/html/2303.09540/assets/x19.png)

Figure A2: Intersection between different deduplicated LAION datasets using different values for the number of k-means clusters k𝑘k. Each cell corresponds to the percentage of intersection between two datasets deduplicated using different k𝑘k values. At the 72% dataset size, more than 97% of data examples are shared between all the datasets regardless of the value of k𝑘k. This shows the robustness of SemDeDup to the number of clusters parameter k𝑘k.

### A.2 Estimating The Fraction of Duplicates Detected By SemDeDup

SemDeDup searches for duplicates within clusters. This results in reducing the floating point operations (FLOPs) required for deduplication by 5 order of magnitude for LAION440M dataset as described in section [3](#S3.SS0.SSS0.Px3 "Clustering to reduce computation ‣ 3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). Indeed, by searching for duplicates within clusters, we ignore duplicates across different clusters if they exist. Here we try to estimate the efficiency of SemDeDup in detecting all the duplicates in the dataset.
  
Let Dϵsubscript𝐷italic-ϵD\_{\epsilon} represent the total number of duplicates in the dataset at a specific value of deduplication threshold ϵitalic-ϵ\epsilon, and Dϵssubscriptsuperscript𝐷𝑠italic-ϵD^{s}\_{\epsilon} represent the total number of duplicates detected by SemDeDup. We define the deduplication efficiency ηϵsubscript𝜂italic-ϵ\eta\_{\epsilon} (eq. [2](#A1.E2 "Equation 2 ‣ A.2 Estimating The Fraction of Duplicates Detected By SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")) as the fraction of duplicates detected by SemDeDup from the total number of duplicates in the datasets at a specific value of ϵitalic-ϵ\epsilon. For example, a deduplication efficiency of 100% corresponds to detecting all the duplicates in a dataset. As computing the exact value of Dϵsubscript𝐷italic-ϵD\_{\epsilon} is computationally expensive, we approximate its value by the number of duplicates between the cluster items and its 20 nearest neighbor clusters and donate this approximated value by Dϵ′subscriptsuperscript𝐷′italic-ϵD^{{}^{\prime}}\_{\epsilon}. We sampled part (2000 clusters) of the LAION440M dataset randomly and compute the value of the deduplication efficiency η𝜂\eta in eq. [2](#A1.E2 "Equation 2 ‣ A.2 Estimating The Fraction of Duplicates Detected By SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") for different values of ϵitalic-ϵ\epsilon and k-means clusters k𝑘k. As we see in Table [A1](#A1.T1 "Table A1 ‣ A.2 Estimating The Fraction of Duplicates Detected By SemDeDup ‣ Appendix A Additional Analysis ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), for k𝑘k=50,000, SemDeDup can effectively detect more than 94% of the duplicates when keeping 63% of LAION440M dataset and 89% of the duplicates when keeping 40%.

|  |  |  |  |
| --- | --- | --- | --- |
|  | η=100∗DϵsDϵ′𝜂100subscriptsuperscript𝐷𝑠italic-ϵsubscriptsuperscript𝐷′italic-ϵ\eta=100\*\frac{D^{s}\_{\epsilon}}{D^{{}^{\prime}}\_{\epsilon}} |  | (2) |

Table A1: Percentage of duplicates detected (η𝜂\eta) by SemDeDup at different deduplication thresholds (ϵitalic-ϵ\epsilon). We notice that η𝜂\eta increases as we reduce the number of clusters k𝑘k in the clustering step of SemDeDup.

| Percentage of Data Kept | 63% | | | 50% | | | 40% | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Num. of Clusters | 70K | 50K | 10K | 70K | 50K | 10K | 70K | 50K | 10K |
| η𝜂\eta | 94.4 | 94.6 | 95.3 | 90.1 | 90.6 | 91.3 | 88.3 | 89.0 | 90.8 |




Table A2: Time for running SemDeDup on LAION440M. Note that we report the total time for deduplication to different dataset size ratios.

| Operation / Time | Clustering Time | DeDup. Time | Total Time |
| --- | --- | --- | --- |
| SemDeDup w/10K Clusters | 2h:36 @8 GPUs | 2h:20 @64 GPUs | 4h:56 |
| SemDeDup w/25K Clusters | 3h:52 @8 GPUs | 1h:19 @64 GPUs | 5h:11 |
| SemDeDup w/50K Clusters | 5h:59 @8 GPUs | 1h:22 @64 GPUs | 7h:21 |
| SemDeDup w/70K Clusters | 9h:02 @8 GPUs | 1h:10 @64 GPUs | 10h:12 |
| Training CLIP on 100% of LAION440M for 32 Epochs | — | — | 69h:52 @176 GPUs |

## Appendix B CLIP Zeroshot Evaluation

In this section, we show the result of zeroshot evaluation for CLIP. We note that the models trained on dataset deduplicated using SemDeDup outperform the baseline model in many tasks. In Table [A4](#A2.T4 "Table A4 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we list the top1 zeroshot accuracy on 24 tasks and in Table [A5](#A2.T5 "Table A5 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") we show the top1 zeroshot accuracy on 6 datasets for out-of-distribution robustness evaluation. Our complete evaluation set has 303030 different datasets in total. When using only 63% of LAION-440M, SemDeDup outperforms the baseline model in 19 out of the 30 tasks. Fig. ([A4](#A2.F4 "Figure A4 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")) and Fig. ([A5](#A2.F5 "Figure A5 ‣ Appendix B CLIP Zeroshot Evaluation ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")) show the performance of different models as a function of training dataset size.

![Refer to caption](/html/2303.09540/assets/x20.png)


(a)

![Refer to caption](/html/2303.09540/assets/x21.png)


(b)

![Refer to caption](/html/2303.09540/assets/x22.png)


(c)

![Refer to caption](/html/2303.09540/assets/x23.png)


(d)

Figure A3: SemDeDup is always better than training on random subset from LAION-440M. The plots show zeroshot top1 accuracy on ImageNet for CLIP models trained on different fractions of data.




Table A3: Training parameters for CLIP

|  |  |
| --- | --- |
| Parameter | Value |
| Model | CLIP ViT-B-16 |
| Warmup | 2000 |
| Epochs | 32 |
| Batch size | 33,792 |
| Learning rate | 5.0e-4, cosine scheduler |
| Optimizer | AdamW, wd=0.2, betas=(0.9, 0.98), eps=1.0e-6 |




Table A4: Zeroshot evaluation top1 accuracy on different datasets. Training CLIP on 63% of the data gives a higher performance in 17/24 datasets. In the first row, model names are represented by the pruning method (Dedup, Baseline, and Rand for SemDeDup, no pruning, and random pruning respectively), and the fraction of data used for training.

| Data / Model | Dedup20 | Dedup40 | Dedup50 | Dedup63 | Dedup72 | Dedup80 | Baseline100 | Rand80 | Rand60 | Rand40 | Rand20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cars | 63.33 | 78.14 | 80.26 | 81.43 | 82.05 | 82.61 | 81.42 | 80.96 | 79.26 | 77.74 | 71.57 |
| Country211 | 14.16 | 17.74 | 18.26 | 18.44 | 18.70 | 18.20 | 19.03 | 18.39 | 16.75 | 15.97 | 12.88 |
| Fgvc Aircraft | 4.44 | 11.49 | 12.42 | 15.42 | 15.27 | 15.09 | 12.66 | 14.31 | 13.11 | 9.27 | 8.85 |
| GTSRB | 38.22 | 37.20 | 36.22 | 43.06 | 41.00 | 35.74 | 42.00 | 43.33 | 41.88 | 25.72 | 32.28 |
| Imagenet1k | 60.24 | 66.90 | 68.27 | 68.66 | 68.93 | 68.80 | 68.74 | 68.29 | 66.12 | 64.82 | 58.86 |
| MNIST | 44.29 | 31.87 | 22.93 | 48.55 | 42.75 | 48.86 | 33.23 | 43.82 | 35.73 | 36.32 | 19.22 |
| Renderedsst2 | 51.46 | 53.65 | 52.72 | 50.80 | 52.99 | 57.17 | 51.13 | 52.72 | 51.29 | 52.22 | 45.47 |
| STL10 | 96.06 | 96.85 | 97.50 | 97.71 | 97.69 | 97.21 | 97.62 | 97.49 | 97.38 | 97.08 | 94.31 |
| SUN397 | 64.81 | 67.98 | 68.26 | 68.89 | 69.25 | 69.76 | 68.79 | 69.08 | 67.96 | 65.51 | 60.76 |
| VOC2007 | 77.94 | 79.51 | 79.74 | 80.37 | 79.75 | 78.61 | 80.01 | 77.97 | 79.43 | 77.96 | 74.42 |
| Caltech101 | 83.05 | 84.40 | 84.98 | 85.06 | 84.35 | 84.75 | 83.42 | 83.69 | 83.93 | 83.38 | 80.62 |
| CIFAR100 | 72.09 | 75.71 | 75.17 | 77.19 | 77.16 | 77.08 | 74.61 | 76.02 | 74.08 | 72.37 | 67.79 |
| CIFAR10 | 92.80 | 93.78 | 94.01 | 94.00 | 94.49 | 94.13 | 93.56 | 94.25 | 93.95 | 92.68 | 89.14 |
| Clevr Dist | 15.75 | 23.05 | 15.75 | 19.48 | 21.95 | 21.82 | 23.03 | 18.45 | 15.59 | 18.60 | 16.21 |
| Clevr Count | 25.37 | 26.36 | 30.85 | 31.87 | 34.73 | 20.31 | 24.14 | 15.37 | 26.43 | 14.85 | 21.67 |
| DMLAB | 13.16 | 17.62 | 17.99 | 19.20 | 18.52 | 20.23 | 20.46 | 18.50 | 21.05 | 17.12 | 19.36 |
| DTD | 49.73 | 53.51 | 54.31 | 56.76 | 58.94 | 57.66 | 57.34 | 57.02 | 53.35 | 50.96 | 41.76 |
| Eurosat | 44.07 | 51.28 | 51.70 | 59.46 | 57.02 | 59.72 | 55.81 | 59.81 | 48.63 | 51.26 | 50.00 |
| Flowers | 45.21 | 62.21 | 67.67 | 69.78 | 70.48 | 66.29 | 67.88 | 68.39 | 65.43 | 62.42 | 58.16 |
| Kitti Dist | 20.39 | 13.36 | 14.35 | 14.77 | 19.97 | 26.72 | 11.25 | 11.11 | 20.68 | 17.02 | 11.25 |
| PCAM | 49.69 | 48.83 | 47.62 | 52.66 | 50.09 | 52.14 | 49.09 | 50.11 | 41.28 | 55.02 | 56.59 |
| Pets | 77.87 | 87.30 | 89.72 | 90.02 | 90.16 | 90.57 | 90.60 | 90.49 | 89.86 | 88.72 | 82.50 |
| Resisc45 | 46.76 | 57.56 | 51.69 | 51.49 | 50.14 | 53.57 | 57.93 | 54.06 | 51.65 | 49.29 | 46.72 |
| SVHN | 34.80 | 33.64 | 26.24 | 40.87 | 33.96 | 32.68 | 35.18 | 26.77 | 26.67 | 32.70 | 25.26 |
| Average | 49.4 | 52.91 | 52.44 | 55.66 | 55.43 | 55.41 | 54.12 | 53.77 | 52.56 | 51.21 | 47.73 |




Table A5: Out-of-distribution Robustness for CLIP models we trained on a different number of examples. The two models trained on 63% and 72% of LAION440M with our de-duplication method have higher average accuracy over 6 datasets. In the first column, model names are represented by the pruning method (Dedup, Baseline, and Rand for SemDeDup, no pruning, and random pruning respectively), and the fraction of data used for training.

| Model/Dataset | ImageNet-A | ImageNet-O | ImageNet-R | ImageNet-Sketch | ImageNet-V2 | ObjectNet | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dedup20 | 31.35 | 52.25 | 72.69 | 46.98 | 52.71 | 51.0 | 51.16 |
| Dedup40 | 38.73 | 49.3 | 77.08 | 51.93 | 59.21 | 54.98 | 55.21 |
| Dedup50 | 39.68 | 48.55 | 77.74 | 53.54 | 60.37 | 55.36 | 55.87 |
| Dedup63 | 39.07 | 48.45 | 78.24 | 53.86 | 60.56 | 56.33 | 56.08 |
| Dedup72 | 39.53 | 47.6 | 78.61 | 53.7 | 61.23 | 56.28 | 56.16 |
| Dedup80 | 39.12 | 47.95 | 78.53 | 53.82 | 60.59 | 54.72 | 55.79 |
| Baseline100 | 38.79 | 48.05 | 78.77 | 53.91 | 60.77 | 55.36 | 55.94 |
| Rand80 | 37.87 | 47.7 | 78.04 | 52.81 | 60.02 | 54.3 | 55.12 |
| Rand60 | 34.6 | 47.5 | 75.61 | 51.18 | 57.97 | 53.22 | 53.35 |
| Rand40 | 31.88 | 49.1 | 73.65 | 49.02 | 56.83 | 49.57 | 51.67 |
| Rand20 | 23.43 | 49.4 | 66.74 | 43.76 | 50.67 | 43.57 | 46.26 |


  

![Refer to caption](/html/2303.09540/assets/x24.png)

Figure A4: Zeroshot performance of CLIP on 24 datasets. The last plot shows the average performance over all datasets.


  

![Refer to caption](/html/2303.09540/assets/x25.png)

Figure A5: Out-of-distribution zeroshot performance on 6 datasets. SemDeDup outperforms random pruning on all datasets for all fractions of dataset kept. The last plot shows the average performance over all datasets.

## Appendix C LAION-233M De-duplication

To support our results on LAION-440M, we also de-duplicate a much smaller dataset of 233 million images. We call this dataset LAION-233M. Usually, CLIP needs to be trained on more than 400 million images as introduced in [[31](#bib.bib31)], so de-duplicating LAION-233M is more challenging in this respect.
We train a baseline model on the 233 million images and two models on 55% of the data, one on a random subset and the other on deduplicated subset using SemDeDup. We trained all the models using the same hyperparameters we used for training on LAION-440M. We show ImageNet top1 zeroshot accuracy for these models in Fig. [A6](#A3.F6 "Figure A6 ‣ Appendix C LAION-233M De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"). The baseline model achieved 64.62% accuracy, while the SemDeDup model achieved 63.61% outperforming the model trained on the random subset (61.3% accuracy).

![Refer to caption](/html/2303.09540/assets/x26.png)


Figure A6: Performance when deduplicating 233 million images from LAION-2B. We deduplicte LAION233M to 55% of its size and train CLIP model on it. SemDeDup performs better than random pruning (63.61% vs 61.3%). Training on the whole 233 million examples gives 64.62%. Note that the deduplicated dataset size here is 128 million only.




Table A6: Performance after training on deduplicated data for the same number of iterations as training on 100% of the data.

| Metric / Model | dedup40 | dedup50 | dedup60 | dedup70 | Baseline (100%) |
| --- | --- | --- | --- | --- | --- |
| Top1 IN Zeroshot Acc. | 68.35 | 68.92 | 69.04 | 69.14 | 68.74 |
| Top5 IN Zeroshot Acc. | 91.64 | 91.82 | 91.86 | 91.73 | 91.42 |


  

![Refer to caption](/html/2303.09540/assets/x27.png)

Figure A7: How many images can we remove from each cluster? Moving from top to down we increase ϵitalic-ϵ\epsilon value. The x-axis corresponds to the cluster size. The y-axis corresponds to the fraction of data removed from each cluster by SemDeDup. As we increase ϵitalic-ϵ\epsilon, more examples are removed from each cluster. We notice that most of the examples from the large clusters (the points to the right) are removed when ϵitalic-ϵ\epsilon becomes large. The points in this figure are for 2000 clusters sampled randomly from a total of 50,000 clusters.


  

![Refer to caption](/html/2303.09540/assets/x28.png)

Figure A8: The number of images in each cluster for 50,000 clusters of LAION-440M images after running k-means clustering in the embedding space. The average cluster size is 8748, but we also see a few clusters with more than 300,000 examples.




Table A7:

|  |
| --- |
| PyTorch-style Pseudo Code For SemDeDup   [⬇](data:text/plain;base64,I0lucHV0OiBjbHVzdGVyX2VtYmVkZGluZ3MsIG51bV9jbHVzdGVycywgZXBzaWxvbgoKZm9yIGkgaW4gcmFuZ2UobnVtX2NsdXN0ZXJzKToKICAgICMgTG9hZCBjbHVzdGVyIGVtYmVkZGluZ3MuCiAgICBjbHVzdGVyX2lfZW1iZWRkaW5ncyA9IGNsdXN0ZXJfZW1iZWRkaW5nc1tpXQoKICAgICMgU29ydCB0aGUgY2x1c3RlciBlbWJlZGRpbmdzIGJ5IHRoZSBkaXN0YW5jZSB0byB0aGUgY2x1c3RlciBjZW50cm9pZC4KICAgIGNsdXN0ZXJfaV9lbWJlZGRpbmdzID0gc29ydF9ieV9kaXN0YW5jZV90b19jbHVzdGVyX2NlbnRyb2lkKGNsdXN0ZXJfaV9lbWJlZGRpbmdzLCBkZXNjZW5kaW5nID0gVHJ1ZSkKCiAgICAjIFdlIHVzZSBkZXNjZW5kaW5nPVRydWUvRmFsc2UgZm9yIGtlZXBpbmcgZXhhbXBsZXMgd2l0aCBsb3cvaGlnaCBzaW1pbGFyaXR5IHRvIGNsdXN0ZXIgY2VudHJvaWRzLiBXZSAgaWdub3JlIHRoaXMgc3RlcCBmb3Iga2VlcGluZyByYW5kb20gZXhhbXBsZXMgZnJvbSBlYWNoIGdyb3VwIG9mIHNpbWlsYXIgZXhhbXBsZXMuIFNlZSBBcHBlbmRpeCBEIGZvciBtb3JlIGRldGFpbHMgYWJvdXQgdGhpcyBzdGVwLgoKICAgICMgQ29tcHV0ZSB0aGUgcGFpcndpc2UgY29zaW5lIHNpbWlsYXJpdHkgYmV0d2VlbiBlbWJlZGRpbmdzCiAgICBwYWlyd2lzZV9zaW1fbWF0cml4ID0gY2x1c3Rlcl9pX2VtYmVkZGluZ3MgQCBjbHVzdGVyX2lfZW1iZWRkaW5ncy5UCgogICAgdHJpdV9zaW1fbWF0cml4ID0gdG9yY2gudHJpdShwYWlyd2lzZV9zaW1fbWF0cml4LCBkaWFnb25hbCA9IDEpCgogICAgTSA9IHRvcmNoLm1heCh0cml1X3NpbV9tYXRyaXgsIGRpbT0wKVswXQoKICAgICMgQ2hlY2sgaWYgdGhlIG1heGltdW0gc2ltaWxhcml0eSA8PSB0aGUgdGhyZXNob2xkLgogICAgcG9pbnRzX3RvX2tlZXBfZnJvbV9jbHVzdGVyX2kgPSBNIDw9IDEtZXBzaWxvbg==) 1#Input: cluster\_embeddings, num\_clusters, epsilon 2 3for i in range(num\_clusters): 4 # Load cluster embeddings. 5 cluster\_i\_embeddings = cluster\_embeddings[i] 6 7 # Sort the cluster embeddings by the distance to the cluster centroid. 8 cluster\_i\_embeddings = sort\_by\_distance\_to\_cluster\_centroid(cluster\_i\_embeddings, descending = True) 9 10 # We use descending=True/False for keeping examples with low/high similarity to cluster centroids. We ignore this step for keeping random examples from each group of similar examples. See Appendix D for more details about this step. 11 12 # Compute the pairwise cosine similarity between embeddings 13 pairwise\_sim\_matrix = cluster\_i\_embeddings @ cluster\_i\_embeddings.T 14 15 triu\_sim\_matrix = torch.triu(pairwise\_sim\_matrix, diagonal = 1) 16 17 M = torch.max(triu\_sim\_matrix, dim=0)[0] 18 19 # Check if the maximum similarity <= the threshold. 20 points\_to\_keep\_from\_cluster\_i = M <= 1-epsilon |

## Appendix D Visualizing Examples Before and After De-duplication

To visually show which images are removed by SemDeDup from LAION440M dataset, we visualize some images from a random cluster before and after deduplication. To do that, we choose a cluster randomly and sort its examples by the cosine similarity to the centroid. By doing that, we can show similar images next to each other in a sequence. Then we visualize a sequence of images before de-duplication. After that, we run SemDeDup, remove duplicates, and sort the remaining examples again. Finally, we visualize the sequence of images from the same indices we visualize before de-duplication. Figures ([A10](#A4.F10 "Figure A10 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") and [A11](#A4.F11 "Figure A11 ‣ Appendix D Visualizing Examples Before and After De-duplication ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication")) show that after applying SemDeDup with different values for the de-duplication threshold ϵitalic-ϵ\epsilon, we keep the unique images.

![Refer to caption](/html/2303.09540/assets/figures/semantically_similar_images.png)

Figure A9: For each of the source images (left column), we can retrieve a set of similar images from LAION440M. For each of the source images, we show a set of images with the highest cosine similarity to it. Images are sorted from left to right by their cosine similarity (1- ϵitalic-ϵ\epsilon) to the source image. By changing ϵitalic-ϵ\epsilon value, we can identify perceptual duplicates, semantic duplicates, and semantically redundant examples for the source images. As we see in the first row, by increasing ϵitalic-ϵ\epsilon we can remove many examples that are semantically similar to the source image.



![Refer to caption](/html/2303.09540/assets/x29.png)

![Refer to caption](/html/2303.09540/assets/x30.png)

![Refer to caption](/html/2303.09540/assets/x31.png)

Figure A10: Examples from the same cluster from LAION-440M dataset before and after de-duplication. Images are sorted by cosine similarity to the cluster centroid. As we increase the deduplication threshold we start to see more unique images.



![Refer to caption](/html/2303.09540/assets/x32.png)

![Refer to caption](/html/2303.09540/assets/x33.png)

![Refer to caption](/html/2303.09540/assets/x34.png)

Figure A11: Examples from the same cluster from LAION440M dataset before and after de-duplication. Images are sorted by cosine similarity to the cluster centroid. As we increase the deduplication threshold we start to see more unique images.

## Appendix E Perplexity Values for SemDeDup on Language Modeling

| method | Baseline (no pruning) | NearDup from [[19](#bib.bib19)] | Random | SemDedup |
| --- | --- | --- | --- | --- |
| validation set |  |  |  |  |
| C4 | 38.95 +/- 0.07 | 39.46 +/- 0.14 | 39.51 +/- 0.07 | 39.35 +/- 0.16 |
| opt\_valid | 47.13 +/- 0.21 | 47.33 +/- 0.20 | 47.75 +/- 0.23 | 47.18 +/- 0.29 |
| prompts\_with\_answers | 29.60 +/- 0.15 | 29.79 +/- 0.11 | 29.95 +/- 0.11 | 29.69 +/- 0.19 |

Figure A12: Comparison of perplexity values for 125M OPT model after pruning via different methods at 96% pruning. Note that [[19](#bib.bib19)] pruned 3.9 % of examples, while above Random and SemDeDup prune 4% of examples. Mean and standard deviation provided across 3 training seeds. Note that the Baseline column does not prune data (which is why the perplexities are lower) and bolded numbers compare between Random, SemDedup, and NearDup.



| method | Baseline (no pruning) | Random | SemDedup |
| --- | --- | --- | --- |
| validation set |  |  |  |
| C4 | 38.95 +/- 0.07 | 42.16 +/- 0.03 | 41.98 +/- 0.09 |
| opt\_valid | 47.13 +/- 0.21 | 50.66 +/- 0.11 | 49.04 +/- 0.16 |
| prompts\_with\_answers | 29.60 +/- 0.15 | 31.65 +/- 0.16 | 30.98 +/- 0.13 |

Figure A13: Comparison of perplexity values for 125M OPT model after pruning via different methods at 80% pruning. Mean and standard deviation provided across 3 training seeds. Note that the Baseline column does not prune data (which is why the perplexities are lower) and bolded numbers compare between Random and SemDedup.



| method | Baseline | Random | SemDedup |
| --- | --- | --- | --- |
| validation set |  |  |  |
| C4 | 38.95 +/- 0.07 | 87.09 +/- 0.21 | 67.32 +/- 0.16 |
| opt\_valid | 47.13 +/- 0.21 | 95.05 +/- 0.31 | 70.17 +/- 0.16 |
| prompts\_with\_answers | 29.60 +/- 0.15 | 60.63 +/- 1.12 | 43.16 +/- 0.19 |

Figure A14: Comparison of perplexity values for 125M OPT model after pruning via different methods at 20% pruning. Mean and standard deviation provided across 3 training seeds. Note that the Baseline column does not prune data (which is why the perplexities are lower) and bolded numbers compare between Random and SemDedup.



| method | Baseline (no pruning) | NearDup from [[19](#bib.bib19)] | Random | SemDedup |
| --- | --- | --- | --- | --- |
| validation set |  |  |  |  |
| C4 | 46.16 +/- 0.00 | 46.85 +/- 0.00 | 46.15 +/- 0.00 | 46.56 +/- 0.00 |
| opt\_valid | 55.69 +/- 0.00 | 55.27 +/- 0.00 | 55.20 +/- 0.00 | 54.88 +/- 0.00 |
| prompts\_with\_answers | 34.04 +/- 0.00 | 33.93 +/- 0.00 | 33.91 +/- 0.00 | 33.83 +/- 0.00 |

Figure A15: Comparison of perplexity values for 1.3b OPT model after pruning via different methods at 96% pruning. Note that [[19](#bib.bib19)] pruned 3.9 % of examples, while above Random and SemDeDup prune 4% of examples. Due to compute restrictions we do not provide random seed standard deviations.



| method | Baseline (no pruning) | Random | SemDedup |
| --- | --- | --- | --- |
| validation set |  |  |  |
| C4 | 46.16 +/- 0.00 | 303.71 +/- 0.00 | 108.95 +/- 0.00 |
| opt\_valid | 55.69 +/- 0.00 | 347.35 +/- 0.00 | 109.68 +/- 0.00 |
| prompts\_with\_answers | 34.04 +/- 0.00 | 269.96 +/- 0.00 | 72.64 +/- 0.00 |

Figure A16: Comparison of perplexity values for 1.3b OPT model after pruning via different methods at 20% pruning. Note that the Baseline column does not prune data (which is why the perplexities are lower) and bolded numbers compare between Random and SemDedup. Due to compute restrictions we do not provide random seed standard deviations.

## Appendix F Qualitative Examples of SemDeDup on C4

![Refer to caption](/html/2303.09540/assets/x35.png)


Figure A17: Percent Data Remaining versus ϵitalic-ϵ\epsilon for C4. The x-axis corresponds to different values of ϵitalic-ϵ\epsilon from Section [3](#S3 "3 SemDeDup ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), and the y-axis represents the corresponding fraction of data in our subset of C4.



![Refer to caption](/html/2303.09540/assets/x36.png)


(a)

![Refer to caption](/html/2303.09540/assets/x37.png)


(b)

![Refer to caption](/html/2303.09540/assets/x38.png)


(c)

Figure A18: SemDeDup performance at different fractions of data for the 125M OPT model. We show results for the C4 validation set (top left), opt\_valid (top right), and prompts\_with\_answers (bottoms). These are the same graphs as Figure [7](#S5.F7 "Figure 7 ‣ 5.3 What is being pruned in language data? ‣ 5 SemDeDup on Natural Language ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication"), but for a wider range of percentage of data kept. We note that SemDeDup consistently outperforms random pruning at lower percentages of data kept.



![Refer to caption](/html/2303.09540/assets/x39.png)


(a)

![Refer to caption](/html/2303.09540/assets/x40.png)


(b)

![Refer to caption](/html/2303.09540/assets/x41.png)


(c)

Figure A19: SemDeDup performance at different fractions of data for the 1.3B OPT model. We show results for the C4 validation set (top left), opt\_valid (top right), and prompts\_with\_answers (bottoms). These are similar to tables [A15](#A5.F15 "Figure A15 ‣ Appendix E Perplexity Values for SemDeDup on Language Modeling ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") and [A16](#A5.F16 "Figure A16 ‣ Appendix E Perplexity Values for SemDeDup on Language Modeling ‣ SemDeDup: Data-efficient learning at web-scale through semantic deduplication") but for a range of percentage of data kept (96 %, 90%, 80%). We note that SemDeDup consistently outperforms random pruning at lower percentages of data kept.

![Refer to caption](/html/2303.09540/assets/x42.png)


Figure A20: Percentage of Data Kept vs. Perplexity on individual validation sets within opt\_valid. Runs are averages across 3 training seeds, and shaded regions represent 1 standard deviation from the mean. The title of each plot represents the name of the individual validation set within opt\_valid. Note that on all tasks, SemDedup significantly random pruning, especially at low percentages of data kept.

![Refer to caption](/html/2303.09540/assets/x43.png)


Figure A21: Percentage of Data Kept vs. Efficiency Gain on individual validation sets within opt\_valid. Runs are averaged across training seeds where the model achieves baseline perplexity at some point in training, and shaded regions represent 1 standard deviation from the mean. The title of each plot represents the name of the individual validation set within opt\_valid.



Keeping 90% data

text



It appears that you already have an account on this site associated with . To connect your existing account…

Keeping 100% data (i.e. no pruning)

text




It appears that you already have an account on this site associated with. To connect your existing account…



You are visiting the placeholder page for Wells Williams. This page is here because someone used our placeholder…



You are visiting the placeholder page for Mathew Barrett. This page is here because someone used our placeholder…



You are visiting the placeholder page for Marcus Slatar. This page is here because someone used our placeholder…



You are visiting the placeholder page for Bernice Andrews. This page is here because someone used our placeholder…



You are visiting the placeholder page for Emiko Chille. This page is here because someone used our placeholder…



You are visiting the placeholder page for Landon Buckland. This page is here because
Someone used our placeholder…



….



You are visiting the placeholder page for Kylie Dickens. This page is here because someone used our placeholder utility …

Figure A22: Example of semantic de-duplication with SemDeDup (cluster 4500)



Keeping 20% data

text




cheap jordan shoes from china free shipping,order maroon foams , jordan blue retro 12 , jordans sz 10 , all white 14s…



Booming business thanks to Cristiano Ronaldo! Nike Presents Cristiano Ronaldo – CR7 Winter Collection. Cristiano Ronaldo …

Keeping 90% data

text




Purchase from us, you can get max discount and free shipping.Free shipping and returns on Nike Jordans at Nordstrom.com….



Product range. Adidas collections are divided into three groups: Sport Performance, Originals, and Sport Style. Originals that …



cool jordans for boys , foamposite paranorman , new black and white foams , lebron 1’s ,cheap jordans online for sale …



This Comfortable Nike Huarache Free Basketball And Running has 1600 x 900 pixel resolution with jpeg format. ..



Top Rating: “Best high performance product.” Performance efficiency. That is the motto of our textile engineers by…



cheap jordan shoes online free shipping order cheap jordans for sale free shipping. Air Jordan 1’s new theme color matching …



…



Trendy Men’s Nike Kyrie 1 Best Seller ’All Star’ Multicolor at high discount. Buy Nike Trainers - The Kyrie 1 All Star comes …

Figure A23: Example of semantically redundant de-duplication with SemDeDup (cluster 4900)

## Appendix G K-means Clustering Details

We use the f​a​i​s​s𝑓𝑎𝑖𝑠𝑠faiss library for clustering. f​a​i​s​s𝑓𝑎𝑖𝑠𝑠faiss is a library for efficient clustering on millions of vectors with GPU support. We use Spherical k-means as we found it better for clustering on ImageNet. Spherical k-means normalizes the cluster centroids after every iteration to have a unit length. This requires the data to also be normalized before clustering. In all our experiments, we run 100 clustering iterations for LAION440M and 20 iterations for C4. We found that centroids do not move after this number of iterations.

[◄](/html/2303.09539)
[![ar5iv homepage](/assets/ar5iv.png)](/)
[Feeling  
lucky?](/feeling_lucky)

[Conversion  
report](/log/2303.09540)
[Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2303.09540)
[View original  
on arXiv](https://arxiv.org/abs/2303.09540)[►](/html/2303.09541)

[Copyright](https://arxiv.org/help/license)
[Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Thu Feb 29 19:49:57 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)

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
