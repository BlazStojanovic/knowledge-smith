---
arxiv: '1606.07792'
authors:
- Heng-Tze Cheng
- Levent Koc
- Jeremiah Harmsen
- Tal Shaked
- Tushar Chandra
- Hrishi Aradhye
- Glen Anderson
- Greg Corrado
- Wei Chai
- Mustafa Ispir
- Rohan Anil
- Zakaria Haque
- Lichan Hong
- Vihan Jain
- Xiaobing Liu
- Hemal Shah
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2016-wide-deep-learning-for-recommender-systems.md
raw_pdf: raw/papers/pdf/2016-wide-deep-learning-for-recommender-systems.pdf
read: false
slug: wide-deep-learning-for-recommender-systems
tags:
- recsys
- ctr-prediction
- feature-encoding
- tabular
title: Wide & Deep Learning for Recommender Systems
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1606.07792v1
venue: null
year: 2016
---

# Wide & Deep Learning for Recommender Systems

> *Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen…* — arXiv 1606.07792, 2016

## Abstract

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2016-wide-deep-learning-for-recommender-systems]]
- PDF (gitignored): `raw/papers/pdf/2016-wide-deep-learning-for-recommender-systems.pdf`
- arXiv: <http://arxiv.org/abs/1606.07792v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2016-cheng-wide-deep.md` before that tree was retired.*

Google's Wide & Deep; joint linear plus deep with hand-crafted feature crosses.
