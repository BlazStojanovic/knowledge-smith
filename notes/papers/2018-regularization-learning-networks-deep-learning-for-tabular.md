---
arxiv: '1805.06440'
authors:
- Ira Shavitt
- Eran Segal
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/1805.06440.md
raw_pdf: raw/papers/1805.06440.pdf
read: false
slug: regularization-learning-networks-deep-learning-for-tabular
tags:
- tabular
- ml
- generalization
- gradient-boosting
title: 'Regularization Learning Networks: Deep Learning for Tabular Datasets'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1805.06440v3
venue: null
year: 2018
---

# Regularization Learning Networks: Deep Learning for Tabular Datasets

> *Ira Shavitt, Eran Segal* — arXiv 1805.06440, 2018

## Abstract

Despite their impressive performance, Deep Neural Networks (DNNs) typically underperform Gradient Boosting Trees (GBTs) on many tabular-dataset learning tasks. We propose that applying a different regularization coefficient to each weight might boost the performance of DNNs by allowing them to make more use of the more relevant inputs. However, this will lead to an intractable number of hyperparameters. Here, we introduce Regularization Learning Networks (RLNs), which overcome this challenge by introducing an efficient hyperparameter tuning scheme which minimizes a new Counterfactual Loss. Our results show that RLNs significantly improve DNNs on tabular datasets, and achieve comparable results to GBTs, with the best performance achieved with an ensemble that combines GBTs and RLNs. RLNs produce extremely sparse networks, eliminating up to 99.8% of the network edges and 82% of the input features, thus providing more interpretable models and reveal the importance that the network assigns to different inputs. RLNs could efficiently learn a single network in datasets that comprise both tabular and unstructured data, such as in the setting of medical imaging accompanied by electronic health records. An open source implementation of RLN can be found at https://github.com/irashavitt/regularization_learning_networks.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/1805.06440]]
- PDF: `raw/papers/1805.06440.pdf`
- arXiv: <http://arxiv.org/abs/1805.06440v3>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2018-shavitt-rln.md` before that tree was retired.*

Regularization Learning Networks with per-weight learnable regularization.
