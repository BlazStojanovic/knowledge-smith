---
arxiv: '2008.13535'
authors:
- Ruoxi Wang
- Rakesh Shivanna
- Derek Z. Cheng
- Sagar Jain
- Dong Lin
- Lichan Hong
- Ed H. Chi
created: '2026-05-08'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2008.13535
  raw: '[[raw/papers/md/2020-dcn-v2-improved-deep-cross-network-and-practical-lessons]]'
  source: http://arxiv.org/abs/2008.13535v2
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2020-dcn-v2-improved-deep-cross-network-and-practical-lessons.md
raw_pdf: raw/papers/pdf/2020-dcn-v2-improved-deep-cross-network-and-practical-lessons.pdf
read: false
slug: dcn-v2-improved-deep-cross-network-and-practical-lessons
tags:
- type/paper
- recsys
- feature-encoding
- ctr-prediction
- tabular
- status/stub
title: 'DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale
  Learning to Rank Systems'
type: note
updated: '2026-05-09'
venue: null
year: 2020
---

# DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

> *Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng…* — arXiv 2008.13535, 2020

## Abstract

Learning effective feature crosses is the key behind building recommender systems. However, the sparse and large feature space requires exhaustive search to identify effective crosses. Deep & Cross Network (DCN) was proposed to automatically and efficiently learn bounded-degree predictive feature interactions. Unfortunately, in models that serve web-scale traffic with billions of training examples, DCN showed limited expressiveness in its cross network at learning more predictive feature interactions. Despite significant research progress made, many deep learning models in production still rely on traditional feed-forward neural networks to learn feature crosses inefficiently.
  In light of the pros/cons of DCN and existing feature interaction learning approaches, we propose an improved framework DCN-V2 to make DCN more practical in large-scale industrial settings. In a comprehensive experimental study with extensive hyper-parameter search and model tuning, we observed that DCN-V2 approaches outperform all the state-of-the-art algorithms on popular benchmark datasets. The improved DCN-V2 is more expressive yet remains cost efficient at feature interaction learning, especially when coupled with a mixture of low-rank architecture. DCN-V2 is simple, can be easily adopted as building blocks, and has delivered significant offline accuracy and online business metrics gains across many web-scale learning to rank systems at Google.

## TL;DR

(stub)

## Notes

(stub)

## Source

- Raw markdown: [[raw/papers/md/2020-dcn-v2-improved-deep-cross-network-and-practical-lessons]]
- PDF: [[raw/papers/pdf/2020-dcn-v2-improved-deep-cross-network-and-practical-lessons.pdf]]
- arXiv: <http://arxiv.org/abs/2008.13535v2>

<!-- ks-harvest -->
## Notes (imported from writings)

*Imported 2026-05-09 from `papers/2021-wang-dcnv2.md` before that tree was retired.*

DCN v2 — Google's web-scale DCN redo with low-rank MoE cross layer.
