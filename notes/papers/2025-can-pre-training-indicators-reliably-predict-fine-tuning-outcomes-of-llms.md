---
arxiv: '2504.12491'
authors:
- Hansi Zeng
- Kai Hui
- Honglei Zhuang
- Zhen Qin
- Zhenrui Yue
- Hamed Zamani
- Dana Alon
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2504.12491
  raw: '[[raw/papers/md/2025-can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms]]'
  source: https://arxiv.org/abs/2504.12491
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms.md
raw_pdf: raw/papers/pdf/2025-can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms.pdf
read: false
slug: can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms
tags:
- type/paper
- status/stub
title: Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?
type: note
updated: '2026-05-11'
year: 2025
---

# Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?

> *Hansi Zeng, Kai Hui, Honglei Zhuang, Zhen Qin, Zhenrui Yue, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2504.12491>
- PDF: [[raw/papers/pdf/2025-can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms.pdf]]
- Raw markdown: [[raw/papers/md/2025-can-pre-training-indicators-reliably-predict-fine-tuning-outcomes-of-llms]]
