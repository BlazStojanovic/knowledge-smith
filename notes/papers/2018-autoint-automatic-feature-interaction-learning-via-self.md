---
arxiv: '1810.11921'
authors:
- Weiping Song
- Chence Shi
- Zhiping Xiao
- Zhijian Duan
- Yewen Xu
- Ming Zhang
- Jian Tang
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1810.11921.md
raw_pdf: raw/papers/1810.11921.pdf
read: false
slug: autoint-automatic-feature-interaction-learning-via-self
tags:
- attention
- recsys
- ctr-prediction
- feature-encoding
title: 'AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural
  Networks'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1810.11921v2
venue: null
year: 2018
---

# AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

> *Weiping Song, Chence Shi, Zhiping Xiao…* — arXiv 1810.11921, 2018

## Abstract

Click-through rate (CTR) prediction, which aims to predict the probability of a user clicking on an ad or an item, is critical to many online applications such as online advertising and recommender systems. The problem is very challenging since (1) the input features (e.g., the user id, user age, item id, item category) are usually sparse and high-dimensional, and (2) an effective prediction relies on high-order combinatorial features (\textit{a.k.a.} cross features), which are very time-consuming to hand-craft by domain experts and are impossible to be enumerated. Therefore, there have been efforts in finding low-dimensional representations of the sparse and high-dimensional raw features and their meaningful combinations. In this paper, we propose an effective and efficient method called the \emph{AutoInt} to automatically learn the high-order feature interactions of input features. Our proposed algorithm is very general, which can be applied to both numerical and categorical input features. Specifically, we map both the numerical and categorical features into the same low-dimensional space. Afterwards, a multi-head self-attentive neural network with residual connections is proposed to explicitly model the feature interactions in the low-dimensional space. With different layers of the multi-head self-attentive neural networks, different orders of feature combinations of input features can be modeled. The whole model can be efficiently fit on large-scale raw data in an end-to-end fashion. Experimental results on four real-world datasets show that our proposed approach not only outperforms existing state-of-the-art approaches for prediction but also offers good explainability. Code is available at: \url{https://github.com/DeepGraphLearning/RecommenderSystems}.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1810.11921]]
- PDF: `raw/papers/1810.11921.pdf`
- arXiv: <http://arxiv.org/abs/1810.11921v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2019-song-autoint.md` before that tree was retired.*

Self-attention over feature embeddings, predating TabTransformer by a year.
