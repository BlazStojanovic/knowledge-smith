---
arxiv: '2407.03257'
authors:
- Han-Jia Ye
- Huai-Hong Yin
- De-Chuan Zhan
- Wei-Lun Chao
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2407.03257.md
raw_pdf: raw/papers/2407.03257.pdf
read: false
slug: revisiting-nearest-neighbor-for-tabular-data-a-deep-tabular
tags: []
title: 'Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two
  Decades Later'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2407.03257v2
venue: null
year: 2024
---

# Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later

> *Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan…* — arXiv 2407.03257, 2024

## Abstract

The widespread enthusiasm for deep learning has recently expanded into the domain of tabular data. Recognizing that the advancement in deep tabular methods is often inspired by classical methods, e.g., integration of nearest neighbors into neural networks, we investigate whether these classical methods can be revitalized with modern techniques. We revisit a differentiable version of $K$-nearest neighbors (KNN) -- Neighbourhood Components Analysis (NCA) -- originally designed to learn a linear projection to capture semantic similarities between instances, and seek to gradually add modern deep learning techniques on top. Surprisingly, our implementation of NCA using SGD and without dimensionality reduction already achieves decent performance on tabular data, in contrast to the results of using existing toolboxes like scikit-learn. Further equipping NCA with deep representations and additional training stochasticity significantly enhances its capability, being on par with the leading tree-based method CatBoost and outperforming existing deep tabular models in both classification and regression tasks on 300 datasets. We conclude our paper by analyzing the factors behind these improvements, including loss functions, prediction strategies, and deep architectures. The code is available at https://github.com/qile2000/LAMDA-TALENT.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2407.03257]]
- PDF: `raw/papers/2407.03257.pdf`
- arXiv: <http://arxiv.org/abs/2407.03257v2>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2025-ye-modernnca]]
