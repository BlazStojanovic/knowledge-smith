---
arxiv: '2304.09151'
authors:
- Hyung Won Chung
- Noah Constant
- Xavier Garcia
- Adam Roberts
- Yi Tay
- Sharan Narang
- Orhan Firat
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2304.09151
  raw: '[[raw/papers/md/2023-unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual]]'
  source: https://arxiv.org/abs/2304.09151
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2023-unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual.md
raw_pdf: raw/papers/pdf/2023-unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual.pdf
read: false
slug: unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual
tags:
- type/paper
- status/stub
title: 'UniMax: Fairer and more Effective Language Sampling for Large-Scale Multilingual
  Pretraining'
type: note
updated: '2026-05-11'
year: 2023
---

# UniMax: Fairer and more Effective Language Sampling for Large-Scale Multilingual Pretraining

> *Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, et al.* — arXiv 2023

## TL;DR

(stub — fill in after reading)

## Abstract

Pretrained multilingual large language models have typically used heuristic temperature-based sampling to balance between different languages. However previous work has not systematically evaluated the efficacy of different pretraining language distributions across model scales. In this paper, we propose a new sampling method, UniMax, that delivers more uniform coverage of head languages while mitigating overfitting on tail languages by explicitly capping the number of repeats over each language's corpus. We perform an extensive series of ablations testing a range of sampling strategies on a suite of multilingual benchmarks, while varying model scale. We find that UniMax outperforms standard temperature-based sampling, and the benefits persist as scale increases. As part of our contribution, we release: (i) an improved and refreshed mC4 multilingual corpus consisting of 29 trillion characters across 107 languages, and (ii) a suite of pretrained umT5 model checkpoints trained with UniMax sampling.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2304.09151>
- PDF: [[raw/papers/pdf/2023-unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual.pdf]]
- Raw markdown: [[raw/papers/md/2023-unimax-fairer-and-more-effective-language-sampling-for-large-scale-multilingual]]
