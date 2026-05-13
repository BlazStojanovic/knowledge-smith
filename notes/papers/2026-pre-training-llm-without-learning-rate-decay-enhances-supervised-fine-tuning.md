---
arxiv: '2603.16127'
authors:
- Kazuki Yano
- Shun Kiyono
- Sosuke Kobayashi
- Sho Takase
- Jun Suzuki
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.16127
  raw: '[[raw/papers/md/2026-pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning]]'
  source: https://arxiv.org/abs/2603.16127
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning.md
raw_pdf: raw/papers/pdf/2026-pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning.pdf
read: false
slug: pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning
tags:
- type/paper
- status/stub
title: Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
type: note
updated: '2026-05-11'
year: 2026
---

# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

> *Kazuki Yano, Shun Kiyono, Sosuke Kobayashi, Sho Takase, Jun Suzuki* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT). Decay-based learning rate schedulers are widely used to minimize pre-training loss. However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored. In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay. Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training. The result also holds across different regimes with mid-training and over-training. Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability. These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability. Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.16127>
- PDF: [[raw/papers/pdf/2026-pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning.pdf]]
- Raw markdown: [[raw/papers/md/2026-pre-training-llm-without-learning-rate-decay-enhances-supervised-fine-tuning]]
