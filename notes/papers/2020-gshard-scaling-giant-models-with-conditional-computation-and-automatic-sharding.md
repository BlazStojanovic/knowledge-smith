---
arxiv: '2006.16668'
authors:
- Dmitry Lepikhin
- HyoukJoong Lee
- Yuanzhong Xu
- Dehao Chen
- Orhan Firat
- Yanping Huang
- Maxim Krikun
- Noam Shazeer
- Zhifeng Chen
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2006.16668
  raw: '[[raw/papers/md/2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding]]'
  source: https://arxiv.org/abs/2006.16668
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding.md
raw_pdf: raw/papers/pdf/2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding.pdf
read: false
slug: gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding
tags:
- type/paper
- status/stub
title: 'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding'
type: note
updated: '2026-05-11'
year: 2020
---

# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

> *Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, et al.* — arXiv 2020

## TL;DR

(stub — fill in after reading)

## Abstract

Neural network scaling has been critical for improving the model quality in many real-world machine learning applications with vast amounts of training data and compute. Although this trend of scaling is affirmed to be a sure-fire approach for better model quality, there are challenges on the path such as the computation cost, ease of programming, and efficient implementation on parallel devices. GShard is a module composed of a set of lightweight annotation APIs and an extension to the XLA compiler. It provides an elegant way to express a wide range of parallel computation patterns with minimal changes to the existing model code. GShard enabled us to scale up multilingual neural machine translation Transformer model with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters using automatic sharding. We demonstrate that such a giant model can efficiently be trained on 2048 TPU v3 accelerators in 4 days to achieve far superior quality for translation from 100 languages to English compared to the prior art.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2006.16668>
- PDF: [[raw/papers/pdf/2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding.pdf]]
- Raw markdown: [[raw/papers/md/2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding]]
