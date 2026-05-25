---
arxiv: '2507.23279'
authors:
- Zunhai Su
- Qingyuan Li
- Hao Zhang
- Weihao Ye
- Qibo Xue
- YuLei Qian
- Yuchen Xie
- Ngai Wong
- Kehong Yuan
created: '2026-05-25'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2507.23279
  raw: '[[raw/papers/md/2025-unveiling-super-experts-in-mixture-of-experts-large-language-models]]'
  source: https://arxiv.org/abs/2507.23279
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-unveiling-super-experts-in-mixture-of-experts-large-language-models.md
raw_pdf: raw/papers/pdf/2025-unveiling-super-experts-in-mixture-of-experts-large-language-models.pdf
read: false
slug: unveiling-super-experts-in-mixture-of-experts-large-language-models
tags:
- type/paper
- status/stub
title: Unveiling Super Experts in Mixture-of-Experts Large Language Models
type: note
updated: '2026-05-25'
year: 2025
---

# Unveiling Super Experts in Mixture-of-Experts Large Language Models

> *Zunhai Su, Qingyuan Li, Hao Zhang, Weihao Ye, Qibo Xue, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

In this study, we report, for the first time, the discovery and systematic investigation of a distinct subset of experts that play a pivotal role in the MoE LLMs' forward inference. These experts are prevalent in open-source MoE LLMs, and despite their extremely limited number, pruning them results in a substantial decline in model performance (e.g., prune just three out of 6,144 causes Qwen3-30B-A3B to generate repetitive and uninformative outputs).We refer to these experts as Super Experts (SEs). Our comprehensive analysis provides progressively deeper insights into SEs: (i) SEs are characterized by rare but extreme activation outliers in the output of the down_proj, which give rise to massive activations in the hidden states between decoder layers. Moreover, the distribution of SEs is model-specific, data-agnostic, and remains unaffected by post-training processes. (ii) By pruning SEs, we assess their significance across a variety of tasks, revealing their considerable impact on the model's overall performance, particularly in mathematical reasoning. (iii) We further investigate why compressing SEs exerts such a pronounced impact. We show that, in MoE LLMs, SEs serve as the primary source of the systematic outlier mechanism in Transformers, and that compressing them profoundly disrupts this process, ultimately causing the collapse of attention sinks. These findings advance the understanding of the internal dynamics of MoE LLMs, filling an important gap in the current knowledge. The code is provided in https://github.com/ZunhaiSu/Super-Experts-Profilling.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2507.23279>
- PDF: [[raw/papers/pdf/2025-unveiling-super-experts-in-mixture-of-experts-large-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2025-unveiling-super-experts-in-mixture-of-experts-large-language-models]]
