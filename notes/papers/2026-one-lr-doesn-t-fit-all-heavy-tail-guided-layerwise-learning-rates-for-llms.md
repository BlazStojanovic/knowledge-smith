---
arxiv: '2605.22297'
authors:
- Di He
- Songjun Tu
- Keyu Wang
- Lu Yin
- Shiwei Liu
created: '2026-06-01'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.22297
  raw: '[[raw/papers/md/2026-one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms]]'
  source: https://arxiv.org/abs/2605.22297
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms.md
raw_pdf: raw/papers/pdf/2026-one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms.pdf
read: false
slug: one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms
tags:
- type/paper
- status/stub
title: 'One LR Doesn''t Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs'
type: note
updated: '2026-06-01'
year: 2026
---

# One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs

> *Di He, Songjun Tu, Keyu Wang, Lu Yin, Shiwei Liu* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Learning rate configuration is a fundamental aspect of modern deep learning. The prevailing practice of applying a uniform learning rate across all layers overlooks the structural heterogeneity of Transformers, potentially limiting their effectiveness as the backbone of Large Language Models (LLMs). In this paper, we introduce Layerwise Learning Rate (LLR), an adaptive scheme that assigns distinct learning rates to individual Transformer layers. Our method is grounded in Heavy-Tailed Self-Regularization (HT-SR) theory, which characterizes the empirical spectral density (ESD) of weight correlation matrices to quantify heavy-tailedness. Layers with weaker heavy-tailedness are assigned larger learning rates to accelerate training, while layers with stronger heavy-tailedness receive smaller learning rates. By tailoring learning rates in this manner, LLR promotes more balanced training across layers, leading to faster convergence and improved generalization. Extensive experiments across architectures ranging from LLaMA to GPT-nano, optimizers including AdamW and Muon, and model scales from 60M to 3B parameters with up to 100B training tokens demonstrate the effectiveness of LLR. LLR achieves up to 1.5x training speedup and consistently outperforms uniform-learning-rate baselines. In particular, it improves the average zero-shot accuracy of 1B models from 47.09% to 49.02%, and that of 3B models from 48.58% to 50.61%. A key advantage of LLR is its low tuning overhead: it can transfer nearly optimal learning-rate settings directly from the uniform baseline. Code is available at https://github.com/hed-ucas/Layer-wise-Learning-Rate.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.22297>
- PDF: [[raw/papers/pdf/2026-one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms.pdf]]
- Raw markdown: [[raw/papers/md/2026-one-lr-doesn-t-fit-all-heavy-tail-guided-layerwise-learning-rates-for-llms]]
