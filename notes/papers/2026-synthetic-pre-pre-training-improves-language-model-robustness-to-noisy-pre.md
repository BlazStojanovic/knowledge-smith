---
arxiv: '2605.10129'
authors:
- Xu Guo
- Runyu Peng
- Jian Tong
- Yunhua Zhou
- Haijun Lv
- Zhihui Lu
- Qipeng Guo
created: '2026-05-15'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.10129
  raw: '[[raw/papers/md/2026-synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre]]'
  source: https://arxiv.org/abs/2605.10129
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre.md
raw_pdf: raw/papers/pdf/2026-synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre.pdf
read: false
slug: synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre
tags:
- type/paper
- status/stub
- pretraining
- synthetic-data
- robustness
- llm
title: Synthetic Pre-Pre-Training Improves Language Model Robustness to Noisy Pre-Training
  Data
type: note
updated: '2026-05-15'
year: 2026
---

# Synthetic Pre-Pre-Training Improves Language Model Robustness to Noisy Pre-Training Data

> *Xu Guo, Runyu Peng, Jian Tong, Yunhua Zhou, Haijun Lv, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models (LLMs) rely on web-scale corpora for pre-training. The noise inherent in these datasets tends to obscure meaningful patterns and ultimately degrade model performance. Data curation mitigates but cannot eliminate such noise, so pre-training corpora remain noisy in practice. We therefore study whether a lightweight pre-pre-training (PPT) stage based on synthetic data with learnable temporal structure helps resist noisy data during the pre-training (PT) stage. Across various corruption settings, our method consistently improves robustness to noise during PT, with larger relative gains at higher noise levels. For a 1B-parameter model, a synthetic PPT stage with only 65M tokens achieves the same final loss as the baseline while using up to 49\% fewer natural-text PT tokens across different noise levels. Mechanistic analyses suggest PPT does not immediately suppress attention to noisy tokens. Rather, PPT-initialized models gradually downweight attention between corrupted tokens during noisy PT. This indicates that synthetic PPT inhibits noise self-modeling and shapes the subsequent optimization trajectory. Code is available at https://github.com/guox18/formal-language-prepretraining.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.10129>
- PDF: [[raw/papers/pdf/2026-synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre.pdf]]
- Raw markdown: [[raw/papers/md/2026-synthetic-pre-pre-training-improves-language-model-robustness-to-noisy-pre]]
