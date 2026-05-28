---
arxiv: '2605.23901'
authors:
- Xu Ouyang
- Deyi Liu
- Yuhang Cai
- Jing Liu
- Yuan Yang
- Chen Zheng
- Thomas Hartvigsen
- Yiyuan Ma
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.23901
  raw: '[[raw/papers/md/2026-llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws]]'
  source: https://arxiv.org/abs/2605.23901
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws.md
raw_pdf: raw/papers/pdf/2026-llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws.pdf
read: false
slug: llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws
tags:
- type/paper
- status/stub
title: 'LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling
  Laws'
type: note
updated: '2026-05-28'
year: 2026
---

# LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws

> *Xu Ouyang, Deyi Liu, Yuhang Cai, Jing Liu, Yuan Yang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Existing scaling laws for Large Language Models (LLMs), predominantly monotonic power laws, fail to explain emerging non-monotonic phenomena such as catastrophic overtraining and quantization-induced degradation, where performance deteriorates despite increased compute.
  We propose the Shannon Scaling Law, a unified theoretical framework that models LLM training as information transmission over a noisy channel, grounded in the Shannon-Hartley theorem. By mapping model parameters to channel bandwidth and training tokens to signal power, our formulation explicitly captures the interaction between learning signal and intrinsic noise. This perspective reveals a fundamental Shannon capacity for LLMs: scaling model size or data without preserving a sufficient signal-to-noise ratio (SNR) inevitably amplifies noise, inducing a transition from monotonic improvement to U-shaped performance degradation.
  We validate our theory through experiments on Pythia and OLMo2 under perturbations, including Gaussian noise, quantization and supervised fine-tuning on math, QA and code tasks. The Shannon Scaling Law consistently outperforms classical scaling laws and recent perturbation-aware laws, achieving strong $R^2$ scores and accurately capturing loss basins missed by prior approaches. It also extrapolates: fitted on $\leq$6.9B Pythia models with $\leq$180B tokens, it predicts the unseen 12B model up to 307B tokens at pooled $R^2{=}0.847$, while monotonic baselines collapse.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.23901>
- PDF: [[raw/papers/pdf/2026-llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws.pdf]]
- Raw markdown: [[raw/papers/md/2026-llms-as-noisy-channels-a-shannon-perspective-on-model-capacity-and-scaling-laws]]
