---
arxiv: '2404.19737'
authors:
- Fabian Gloeckle
- Badr Youbi Idrissi
- Baptiste Rozière
- David Lopez-Paz
- Gabriel Synnaeve
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2404.19737
  raw: '[[raw/papers/md/2024-better-faster-large-language-models-via-multi-token-prediction]]'
  source: https://arxiv.org/abs/2404.19737
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-better-faster-large-language-models-via-multi-token-prediction.md
raw_pdf: raw/papers/pdf/2024-better-faster-large-language-models-via-multi-token-prediction.pdf
read: false
slug: better-faster-large-language-models-via-multi-token-prediction
tags:
- type/paper
- status/stub
title: Better & Faster Large Language Models via Multi-token Prediction
type: note
updated: '2026-05-11'
year: 2024
---

# Better & Faster Large Language Models via Multi-token Prediction

> *Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3 times faster at inference, even with large batch sizes.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2404.19737>
- PDF: [[raw/papers/pdf/2024-better-faster-large-language-models-via-multi-token-prediction.pdf]]
- Raw markdown: [[raw/papers/md/2024-better-faster-large-language-models-via-multi-token-prediction]]
