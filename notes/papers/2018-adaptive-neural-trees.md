---
arxiv: '1807.06699'
authors:
- Ryutaro Tanno
- Kai Arulkumaran
- Daniel C. Alexander
- Antonio Criminisi
- Aditya Nori
created: '2026-05-08'
doi: null
kind: paper
parser: ar5iv
raw_md: raw/papers/md/2018-adaptive-neural-trees.md
raw_pdf: raw/papers/pdf/2018-adaptive-neural-trees.pdf
read: false
slug: adaptive-neural-trees
tags:
- decision-tree
- ml
- interpretability
- vision
title: Adaptive Neural Trees
type: note
updated: '2026-05-09'
url: http://arxiv.org/abs/1807.06699v5
venue: null
year: 2018
---

# Adaptive Neural Trees

> *Ryutaro Tanno, Kai Arulkumaran, Daniel C. Alexander…* — arXiv 1807.06699, 2018

## Abstract

Deep neural networks and decision trees operate on largely separate paradigms; typically, the former performs representation learning with pre-specified architectures, while the latter is characterised by learning hierarchies over pre-specified features with data-driven architectures. We unite the two via adaptive neural trees (ANTs) that incorporates representation learning into edges, routing functions and leaf nodes of a decision tree, along with a backpropagation-based training algorithm that adaptively grows the architecture from primitive modules (e.g., convolutional layers). We demonstrate that, whilst achieving competitive performance on classification and regression datasets, ANTs benefit from (i) lightweight inference via conditional computation, (ii) hierarchical separation of features useful to the task e.g. learning meaningful class associations, such as separating natural vs. man-made objects, and (iii) a mechanism to adapt the architecture to the size and complexity of the training dataset.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2018-adaptive-neural-trees]]
- PDF: `raw/papers/pdf/2018-adaptive-neural-trees.pdf`
- arXiv: <http://arxiv.org/abs/1807.06699v5>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2019-tanno-adaptive-neural-trees.md` before that tree was retired.*

Adaptive Neural Trees with grow/prune operations.
