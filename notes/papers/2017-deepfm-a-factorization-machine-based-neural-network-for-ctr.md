---
arxiv: '1703.04247'
authors:
- Huifeng Guo
- Ruiming Tang
- Yunming Ye
- Zhenguo Li
- Xiuqiang He
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1703.04247.md
raw_pdf: raw/papers/1703.04247.pdf
read: false
slug: deepfm-a-factorization-machine-based-neural-network-for-ctr
tags:
- ctr-prediction
- recsys
- feature-encoding
- tabular
title: 'DeepFM: A Factorization-Machine based Neural Network for CTR Prediction'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1703.04247v1
venue: null
year: 2017
---

# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

> *Huifeng Guo, Ruiming Tang, Yunming Ye…* — arXiv 1703.04247, 2017

## Abstract

Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods seem to have a strong bias towards low- or high-order interactions, or require expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and high-order feature interactions. The proposed model, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to the latest Wide \& Deep model from Google, DeepFM has a shared input to its "wide" and "deep" parts, with no need of feature engineering besides raw features. Comprehensive experiments are conducted to demonstrate the effectiveness and efficiency of DeepFM over the existing models for CTR prediction, on both benchmark data and commercial data.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1703.04247]]
- PDF (gitignored): `raw/papers/1703.04247.pdf`
- arXiv: <http://arxiv.org/abs/1703.04247v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2017-guo-deepfm.md` before that tree was retired.*

DeepFM — factorization machine plus MLP; CTR-dominant but never crossed to general tabular.
