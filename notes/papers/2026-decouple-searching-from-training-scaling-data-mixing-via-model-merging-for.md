---
arxiv: '2602.00747'
authors:
- Shengrui Li
- Fei Zhao
- Kaiyan Zhao
- Jieying Ye
- Haifeng Liu
- Fangcheng Shi
- Zheyong Xie
- Yao Hu
- Shaosheng Cao
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2602.00747
  raw: '[[raw/papers/md/2026-decouple-searching-from-training-scaling-data-mixing-via-model-merging-for]]'
  source: https://arxiv.org/abs/2602.00747
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-decouple-searching-from-training-scaling-data-mixing-via-model-merging-for.md
raw_pdf: raw/papers/pdf/2026-decouple-searching-from-training-scaling-data-mixing-via-model-merging-for.pdf
read: false
slug: decouple-searching-from-training-scaling-data-mixing-via-model-merging-for
tags:
- type/paper
- status/stub
title: 'Decouple Searching from Training: Scaling Data Mixing via Model Merging for
  Large Language Model Pre-training'
type: note
updated: '2026-05-11'
year: 2026
---

# Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training

> *Shengrui Li, Fei Zhao, Kaiyan Zhao, Jieying Ye, Haifeng Liu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Determining an effective data mixture is a key factor in Large Language Model (LLM) pre-training, where models must balance general competence with proficiency on hard tasks such as math and code. However, identifying an optimal mixture remains an open challenge, as existing approaches either rely on unreliable tiny-scale proxy experiments or require prohibitively expensive large-scale exploration. To address this, we propose Decouple Searching from Training Mix (DeMix), a novel framework that leverages model merging to predict optimal data ratios. Instead of training proxy models for every sampled mixture, DeMix trains component models on candidate datasets at scale and derives data mixture proxies via weighted model merging. This paradigm decouples search from training costs, enabling evaluation of unlimited sampled mixtures without extra training burden and thus facilitating better mixture discovery through more search trials. Extensive experiments demonstrate that DeMix breaks the trade-off between sufficiency, accuracy and efficiency, obtaining the optimal mixture with higher benchmark performance at lower search cost. Additionally, we release the DeMix Corpora, a comprehensive 22T-token dataset comprising high-quality pre-training data with validated mixtures to facilitate open research. Our code and DeMix Corpora is available at https://github.com/Lucius-lsr/DeMix.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2602.00747>
- PDF: [[raw/papers/pdf/2026-decouple-searching-from-training-scaling-data-mixing-via-model-merging-for.pdf]]
- Raw markdown: [[raw/papers/md/2026-decouple-searching-from-training-scaling-data-mixing-via-model-merging-for]]
