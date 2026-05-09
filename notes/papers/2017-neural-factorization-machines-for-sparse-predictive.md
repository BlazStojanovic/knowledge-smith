---
arxiv: '1708.05027'
authors:
- Xiangnan He
- Tat-Seng Chua
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2017-neural-factorization-machines-for-sparse-predictive.md
raw_pdf: raw/papers/pdf/2017-neural-factorization-machines-for-sparse-predictive.pdf
read: false
slug: neural-factorization-machines-for-sparse-predictive
tags:
- recsys
- ctr-prediction
- feature-encoding
- tabular
title: Neural Factorization Machines for Sparse Predictive Analytics
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1708.05027v1
venue: null
year: 2017
---

# Neural Factorization Machines for Sparse Predictive Analytics

> *Xiangnan He, Tat-Seng Chua* — arXiv 1708.05027, 2017

## Abstract

Many predictive tasks of web applications need to model categorical variables, such as user IDs and demographics like genders and occupations. To apply standard machine learning techniques, these categorical predictors are always converted to a set of binary features via one-hot encoding, making the resultant feature vector highly sparse. To learn from such sparse data effectively, it is crucial to account for the interactions between features.
  Factorization Machines (FMs) are a popular solution for efficiently using the second-order feature interactions. However, FM models feature interactions in a linear way, which can be insufficient for capturing the non-linear and complex inherent structure of real-world data. While deep neural networks have recently been applied to learn non-linear feature interactions in industry, such as the Wide&Deep by Google and DeepCross by Microsoft, the deep structure meanwhile makes them difficult to train.
  In this paper, we propose a novel model Neural Factorization Machine (NFM) for prediction under sparse settings. NFM seamlessly combines the linearity of FM in modelling second-order feature interactions and the non-linearity of neural network in modelling higher-order feature interactions. Conceptually, NFM is more expressive than FM since FM can be seen as a special case of NFM without hidden layers. Empirical results on two regression tasks show that with one hidden layer only, NFM significantly outperforms FM with a 7.3% relative improvement. Compared to the recent deep learning methods Wide&Deep and DeepCross, our NFM uses a shallower structure but offers better performance, being much easier to train and tune in practice.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2017-neural-factorization-machines-for-sparse-predictive]]
- PDF (gitignored): [[raw/papers/pdf/2017-neural-factorization-machines-for-sparse-predictive.pdf]]
- arXiv: <http://arxiv.org/abs/1708.05027v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-he-nfm.md` before that tree was retired.*

Neural Factorization Machine with bi-interaction pooling layer.
