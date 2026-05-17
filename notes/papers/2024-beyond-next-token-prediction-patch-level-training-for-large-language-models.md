---
arxiv: '2407.12665'
authors:
- Chenze Shao
- Fandong Meng
- Jie Zhou
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.12665
  raw: '[[raw/papers/md/2024-beyond-next-token-prediction-patch-level-training-for-large-language-models]]'
  source: https://arxiv.org/abs/2407.12665
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-beyond-next-token-prediction-patch-level-training-for-large-language-models.md
raw_pdf: raw/papers/pdf/2024-beyond-next-token-prediction-patch-level-training-for-large-language-models.pdf
read: false
slug: beyond-next-token-prediction-patch-level-training-for-large-language-models
tags:
- type/paper
- status/stub
- pretraining
- efficiency
- llm
title: 'Beyond Next Token Prediction: Patch-Level Training for Large Language Models'
type: note
updated: '2026-05-15'
year: 2024
---

# Beyond Next Token Prediction: Patch-Level Training for Large Language Models

> *Chenze Shao, Fandong Meng, Jie Zhou* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

The prohibitive training costs of Large Language Models (LLMs) have emerged as a significant bottleneck in the development of next-generation LLMs. In this paper, we show that it is possible to significantly reduce the training costs of LLMs without sacrificing their performance. Specifically, we introduce patch-level training for LLMs, in which multiple tokens are aggregated into a unit of higher information density, referred to as a `patch', to serve as the fundamental text unit for training LLMs. During patch-level training, we feed the language model shorter sequences of patches and train it to predict the next patch, thereby processing the majority of the training data at a significantly reduced cost. Following this, the model continues token-level training on the remaining training data to align with the inference mode. Experiments on a diverse range of models (370M-2.7B parameters) demonstrate that patch-level training can reduce the overall training costs to 0.5$\times$, without compromising the model performance compared to token-level training. Source code: https://github.com/shaochenze/PatchTrain.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2407.12665>
- PDF: [[raw/papers/pdf/2024-beyond-next-token-prediction-patch-level-training-for-large-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2024-beyond-next-token-prediction-patch-level-training-for-large-language-models]]
