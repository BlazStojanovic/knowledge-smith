---
arxiv: '2605.24956'
authors:
- Xiangdong Zhang
- Debing Zhang
- Shaofeng Zhang
- Xiaohan Qin
- Yu Cheng
- Junchi Yan
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.24956
  raw: '[[raw/papers/md/2026-nitp-next-implicit-token-prediction-for-llm-pre-training]]'
  source: https://arxiv.org/abs/2605.24956
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-nitp-next-implicit-token-prediction-for-llm-pre-training.md
raw_pdf: raw/papers/pdf/2026-nitp-next-implicit-token-prediction-for-llm-pre-training.pdf
read: false
slug: nitp-next-implicit-token-prediction-for-llm-pre-training
tags:
- type/paper
- status/stub
title: 'NITP: Next Implicit Token Prediction for LLM Pre-training'
type: note
updated: '2026-05-28'
year: 2026
---

# NITP: Next Implicit Token Prediction for LLM Pre-training

> *Xiangdong Zhang, Debing Zhang, Shaofeng Zhang, Xiaohan Qin, Yu Cheng, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Standard next-token prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurations that can limit generalization. To address this issue, we propose Next Implicit Token Prediction (NITP), which augments discrete prediction with dense continuous supervision directly in the representation space. NITP trains the model to predict the implicit semantic content of the next token, using shallow-layer representations from the same model as stable self-supervised targets. We provide theoretical analysis showing that NITP regularizes the optimization landscape by mitigating under-constrained degrees of freedom and encouraging a compact, structured representation geometry. Empirically, across dense and MoE models ranging from 0.5B to 9B parameters, NITP consistently improves downstream performance with negligible computational overhead. On a 9B MoE model, NITP achieves a 5.7% absolute improvement on MMLU-Pro, along with gains of 6.4% on C3 and 4.3% on CommonsenseQA, with approximately 2% additional training FLOPs and no additional inference cost. Our implementation is available at https://github.com/aHapBean/NITP.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.24956>
- PDF: [[raw/papers/pdf/2026-nitp-next-implicit-token-prediction-for-llm-pre-training.pdf]]
- Raw markdown: [[raw/papers/md/2026-nitp-next-implicit-token-prediction-for-llm-pre-training]]
