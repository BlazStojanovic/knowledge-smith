---
arxiv: '2408.08310'
authors:
- Ruihang Li
- Yixuan Wei
- Miaosen Zhang
- Nenghai Yu
- Han Hu
- Houwen Peng
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2408.08310
  raw: '[[raw/papers/md/2024-scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws]]'
  source: https://arxiv.org/abs/2408.08310
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws.md
raw_pdf: raw/papers/pdf/2024-scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws.pdf
read: false
slug: scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws
tags:
- type/paper
- status/stub
title: 'ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling
  Laws'
type: note
updated: '2026-05-11'
year: 2024
---

# ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws

> *Ruihang Li, Yixuan Wei, Miaosen Zhang, Nenghai Yu, Han Hu, et al.* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

High-quality data is crucial for the pre-training performance of large language models. Unfortunately, existing quality filtering methods rely on a known high-quality dataset as reference, which can introduce potential bias and compromise diversity. In this paper, we propose ScalingFilter, a novel approach that evaluates text quality based on the perplexity difference between two language models trained on the same data, thereby eliminating the influence of the reference dataset in the filtering process. An theoretical analysis shows that ScalingFilter is equivalent to an inverse utilization of scaling laws. Through training models with 1.3B parameters on the same data source processed by various quality filters, we find ScalingFilter can improve zero-shot performance of pre-trained models in downstream tasks. To assess the bias introduced by quality filtering, we introduce semantic diversity, a metric of utilizing text embedding models for semantic representations. Extensive experiments reveal that semantic diversity is a reliable indicator of dataset diversity, and ScalingFilter achieves an optimal balance between downstream performance and semantic diversity.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2408.08310>
- PDF: [[raw/papers/pdf/2024-scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws.pdf]]
- Raw markdown: [[raw/papers/md/2024-scalingfilter-assessing-data-quality-through-inverse-utilization-of-scaling-laws]]
