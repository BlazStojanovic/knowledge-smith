---
arxiv: '2605.02364'
authors:
- Fengze Liu
- Weidong Zhou
- Binbin Liu
- Ping Guo
- Zijun Wang
- Bingni Zhang
- Yifan Zhang
- Yifeng Yu
- Xiaohuan Zhou
- Taifeng Wang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.02364
  raw: '[[raw/papers/md/2026-infolaw-information-scaling-laws-for-large-language-models-with-quality]]'
  source: https://arxiv.org/abs/2605.02364
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-infolaw-information-scaling-laws-for-large-language-models-with-quality.md
raw_pdf: raw/papers/pdf/2026-infolaw-information-scaling-laws-for-large-language-models-with-quality.pdf
read: false
slug: infolaw-information-scaling-laws-for-large-language-models-with-quality
tags:
- type/paper
- status/stub
title: 'InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted
  Mixture Data and Repetition'
type: note
updated: '2026-05-11'
year: 2026
---

# InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition

> *Fengze Liu, Weidong Zhou, Binbin Liu, Ping Guo, Zijun Wang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Upweighting high-quality data in LLM pretraining often improves performance, but in datalimited regimes, especially under overtraining, stronger upweighting increases repetition and can degrade performance. However, standard scaling laws do not reliably extrapolate across mixture recipes or under repetitions, making the selection for optimal data recipes at scaling underdetermined. To solve this, we introduce InfoLaw (Information Scaling Laws), a data-aware scaling framework that predicts loss from consumed tokens, model size, data mixture weights, and repetition. The key idea is to model pretraining as information accumulation, where quality controls information density and repetition induces scaledependent diminishing returns. We first collect the model performance after training on datasets that vary in scale, quality distribution, and repetition level. Then we build up the modeling for information so that information accurately predicts those model performance. InfoLaw predicts performance on unseen data recipes and larger scale runs (up to 7B, 425B tokens) with 0.15% mean and 0.96% max absolute error in loss, and it extrapolates reliably across overtraining levels, enabling efficient data-recipe selection under varying compute budgets.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.02364>
- PDF: [[raw/papers/pdf/2026-infolaw-information-scaling-laws-for-large-language-models-with-quality.pdf]]
- Raw markdown: [[raw/papers/md/2026-infolaw-information-scaling-laws-for-large-language-models-with-quality]]
