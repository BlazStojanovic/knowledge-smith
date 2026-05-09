---
arxiv: '2303.05506'
authors:
- Alan Jeffares
- Tennison Liu
- Jonathan Crabbé
- Fergus Imrie
- Mihaela van der Schaar
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/2303.05506.md
raw_pdf: raw/papers/2303.05506.pdf
read: false
slug: tangos-regularizing-tabular-neural-networks-through
tags: []
title: 'TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization
  and Specialization'
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/2303.05506v1
venue: null
year: 2023
---

# TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization

> *Alan Jeffares, Tennison Liu, Jonathan Crabbé…* — arXiv 2303.05506, 2023

## Abstract

Despite their success with unstructured data, deep neural networks are not yet a panacea for structured tabular data. In the tabular domain, their efficiency crucially relies on various forms of regularization to prevent overfitting and provide strong generalization performance. Existing regularization techniques include broad modelling decisions such as choice of architecture, loss functions, and optimization methods. In this work, we introduce Tabular Neural Gradient Orthogonalization and Specialization (TANGOS), a novel framework for regularization in the tabular setting built on latent unit attributions. The gradient attribution of an activation with respect to a given input feature suggests how the neuron attends to that feature, and is often employed to interpret the predictions of deep networks. In TANGOS, we take a different approach and incorporate neuron attributions directly into training to encourage orthogonalization and specialization of latent attributions in a fully-connected network. Our regularizer encourages neurons to focus on sparse, non-overlapping input features and results in a set of diverse and specialized latent units. In the tabular domain, we demonstrate that our approach can lead to improved out-of-sample generalization performance, outperforming other popular regularization methods. We provide insight into why our regularizer is effective and demonstrate that TANGOS can be applied jointly with existing methods to achieve even greater generalization performance.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/2303.05506]]
- PDF: `raw/papers/2303.05506.pdf`
- arXiv: <http://arxiv.org/abs/2303.05506v1>

<!-- ks-crosslink -->
**Writing-tier note:** [[../papers/2023-jeffares-tangos]]
