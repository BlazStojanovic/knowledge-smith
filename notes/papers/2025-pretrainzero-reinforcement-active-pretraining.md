---
arxiv: '2512.03442'
authors:
- Xingrun Xing
- Zhiyuan Fan
- Jie Lou
- Guoqi Li
- Jiajun Zhang
- Debing Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.03442
  raw: '[[raw/papers/md/2025-pretrainzero-reinforcement-active-pretraining]]'
  source: https://arxiv.org/abs/2512.03442
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-pretrainzero-reinforcement-active-pretraining.md
raw_pdf: raw/papers/pdf/2025-pretrainzero-reinforcement-active-pretraining.pdf
read: false
slug: pretrainzero-reinforcement-active-pretraining
tags:
- type/paper
- status/stub
title: 'PretrainZero: Reinforcement Active Pretraining'
type: note
updated: '2026-05-11'
year: 2025
---

# PretrainZero: Reinforcement Active Pretraining

> *Xingrun Xing, Zhiyuan Fan, Jie Lou, Guoqi Li, Jiajun Zhang, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Mimicking human behavior to actively learning from general experience and achieve artificial general intelligence has always been a human dream. Recent reinforcement learning (RL) based large-thinking models demonstrate impressive expert-level abilities, i.e., software and math, but still rely heavily on verifiable rewards in specific domains, placing a significant bottleneck to extend the performance boundary of general reasoning capabilities. In this work, we propose PretrainZero, a reinforcement active learning framework built on the pretraining corpus to extend RL from domain-specific post-training to general pretraining. PretrainZero features the following characteristics: 1) Active pretraining: inspired by the active learning ability of humans, PretrainZero learns a unified reasoning policy to actively identify reasonable and informative contents from pretraining corpus, and reason to predict these contents by RL. 2) Self-supervised learning: without any verifiable labels, pretrained reward models, or supervised fine-tuning, we directly pretrain reasoners from 3 to 30B base models on the general Wikipedia corpus using RL, significantly breaking the verification data-wall for general reasoning. 3) Verification scaling: by tackling increasingly challenging masked spans, PretrainZero substantially enhances the general reasoning abilities of pretrained base models. In reinforcement pretraining, PretrainZero improves Qwen3-4B-Base for 8.43, 5.96 and 10.60 on MMLU-Pro, SuperGPQA and math average benchmarks. In post-training, the pretrained models can also serve as reasoning foundation models for downstream RLVR tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.03442>
- PDF: [[raw/papers/pdf/2025-pretrainzero-reinforcement-active-pretraining.pdf]]
- Raw markdown: [[raw/papers/md/2025-pretrainzero-reinforcement-active-pretraining]]
