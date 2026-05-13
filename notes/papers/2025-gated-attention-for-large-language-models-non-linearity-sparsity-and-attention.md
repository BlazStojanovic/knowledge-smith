---
arxiv: '2505.06708'
authors:
- Zihan Qiu
- Zekun Wang
- Bo Zheng
- Zeyu Huang
- Kaiyue Wen
- Songlin Yang
- Rui Men
- Le Yu
- Fei Huang
- Suozhi Huang
- Dayiheng Liu
- Jingren Zhou
- Junyang Lin
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2505.06708
  raw: '[[raw/papers/md/2025-gated-attention-for-large-language-models-non-linearity-sparsity-and-attention]]'
  source: https://arxiv.org/abs/2505.06708
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-gated-attention-for-large-language-models-non-linearity-sparsity-and-attention.md
raw_pdf: raw/papers/pdf/2025-gated-attention-for-large-language-models-non-linearity-sparsity-and-attention.pdf
read: false
slug: gated-attention-for-large-language-models-non-linearity-sparsity-and-attention
tags:
- type/paper
- status/stub
title: 'Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free'
type: note
updated: '2026-05-11'
year: 2025
---

# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

> *Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Gating mechanisms have been widely utilized, from early models like LSTMs and Highway Networks to recent state space models, linear attention, and also softmax attention. Yet, existing literature rarely examines the specific effects of gating. In this work, we conduct comprehensive experiments to systematically investigate gating-augmented softmax attention variants. Specifically, we perform a comprehensive comparison over 30 variants of 15B Mixture-of-Experts (MoE) models and 1.7B dense models trained on a 3.5 trillion token dataset. Our central finding is that a simple modification-applying a head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA)-consistently improves performance. This modification also enhances training stability, tolerates larger learning rates, and improves scaling properties. By comparing various gating positions and computational variants, we attribute this effectiveness to two key factors: (1) introducing non-linearity upon the low-rank mapping in the softmax attention, and (2) applying query-dependent sparse gating scores to modulate the SDPA output. Notably, we find this sparse gating mechanism mitigates 'attention sink' and enhances long-context extrapolation performance, and we also release related $\href{https://github.com/qiuzh20/gated_attention}{codes}$ and $\href{https://huggingface.co/QwQZh/gated_attention}{models}$ to facilitate future research.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2505.06708>
- PDF: [[raw/papers/pdf/2025-gated-attention-for-large-language-models-non-linearity-sparsity-and-attention.pdf]]
- Raw markdown: [[raw/papers/md/2025-gated-attention-for-large-language-models-non-linearity-sparsity-and-attention]]
