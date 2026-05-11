---
arxiv: '2404.16030'
authors:
- Jiawei Ma
- Po-Yao Huang
- Saining Xie
- Shang-Wen Li
- Luke Zettlemoyer
- Shih-Fu Chang
- Wen-Tau Yih
- Hu Xu
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.16030
  raw: '[[raw/papers/md/2024-mode-clip-data-experts-via-clustering]]'
  source: http://arxiv.org/abs/2404.16030v1
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-mode-clip-data-experts-via-clustering.md
raw_pdf: raw/papers/pdf/2024-mode-clip-data-experts-via-clustering.pdf
read: false
slug: mode-clip-data-experts-via-clustering
tags:
- type/paper
- contrastive
- vision
- pretraining
- mixture-of-experts
- status/stub
title: 'MoDE: CLIP Data Experts via Clustering'
type: note
updated: '2026-05-09'
venue: null
year: 2024
---

# MoDE: CLIP Data Experts via Clustering

> *Jiawei Ma, Po-Yao Huang, Saining Xie…* — arXiv 2404.16030, 2024

## Abstract

The success of contrastive language-image pretraining (CLIP) relies on the supervision from the pairing between images and captions, which tends to be noisy in web-crawled data. We present Mixture of Data Experts (MoDE) and learn a system of CLIP data experts via clustering. Each data expert is trained on one data cluster, being less sensitive to false negative noises in other clusters. At inference time, we ensemble their outputs by applying weights determined through the correlation between task metadata and cluster conditions. To estimate the correlation precisely, the samples in one cluster should be semantically similar, but the number of data experts should still be reasonable for training and inference. As such, we consider the ontology in human language and propose to use fine-grained cluster centers to represent each data expert at a coarse-grained level. Experimental studies show that four CLIP data experts on ViT-B/16 outperform the ViT-L/14 by OpenAI CLIP and OpenCLIP on zero-shot image classification but with less ($<$35\%) training cost. Meanwhile, MoDE can train all data expert asynchronously and can flexibly include new data experts. The code is available at https://github.com/facebookresearch/MetaCLIP/tree/main/mode.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2024-mode-clip-data-experts-via-clustering]]
- PDF: [[raw/papers/pdf/2024-mode-clip-data-experts-via-clustering.pdf]]
- arXiv: <http://arxiv.org/abs/2404.16030v1>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2024-kohli-quantifying-benchmarks.md` before that tree was retired.*

Surveys 30 papers across 187 datasets; small dataset changes flip conclusions.
