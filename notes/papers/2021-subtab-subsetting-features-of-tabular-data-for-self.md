---
arxiv: '2110.04361'
authors:
- Talip Ucar
- Ehsan Hajiramezanali
- Lindsay Edwards
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2110.04361
  raw: '[[raw/papers/md/2021-subtab-subsetting-features-of-tabular-data-for-self]]'
  source: http://arxiv.org/abs/2110.04361v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2021-subtab-subsetting-features-of-tabular-data-for-self.md
raw_pdf: raw/papers/pdf/2021-subtab-subsetting-features-of-tabular-data-for-self.pdf
read: false
slug: subtab-subsetting-features-of-tabular-data-for-self
tags:
- type/paper
- tabular
- self-supervised
- contrastive
- feature-encoding
- status/stub
title: 'SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation
  Learning'
type: note
updated: '2026-05-09'
venue: null
year: 2021
---

# SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning

> *Talip Ucar, Ehsan Hajiramezanali, Lindsay Edwards* — arXiv 2110.04361, 2021

## Abstract

Self-supervised learning has been shown to be very effective in learning useful representations, and yet much of the success is achieved in data types such as images, audio, and text. The success is mainly enabled by taking advantage of spatial, temporal, or semantic structure in the data through augmentation. However, such structure may not exist in tabular datasets commonly used in fields such as healthcare, making it difficult to design an effective augmentation method, and hindering a similar progress in tabular data setting. In this paper, we introduce a new framework, Subsetting features of Tabular data (SubTab), that turns the task of learning from tabular data into a multi-view representation learning problem by dividing the input features to multiple subsets. We argue that reconstructing the data from the subset of its features rather than its corrupted version in an autoencoder setting can better capture its underlying latent representation. In this framework, the joint representation can be expressed as the aggregate of latent variables of the subsets at test time, which we refer to as collaborative inference. Our experiments show that the SubTab achieves the state of the art (SOTA) performance of 98.31% on MNIST in tabular setting, on par with CNN-based SOTA models, and surpasses existing baselines on three other real-world datasets by a significant margin.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2021-subtab-subsetting-features-of-tabular-data-for-self]]
- PDF: [[raw/papers/pdf/2021-subtab-subsetting-features-of-tabular-data-for-self.pdf]]
- arXiv: <http://arxiv.org/abs/2110.04361v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-ucar-subtab.md` before that tree was retired.*

SubTab — reconstruct full row from feature subsets for contrastive pretraining.
