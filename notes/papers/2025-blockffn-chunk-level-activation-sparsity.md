---
authors:
- Chenyang Song
- Weilin Zhao
- Xu Han
- Chaojun Xiao
- Yingfa Chen
- Yuxuan Li
- Zhiyuan Liu
- Maosong Sun
created: '2026-05-12'
kind: paper
links:
  code: https://github.com/thunlp/BlockFFN
  paper: https://openreview.net/pdf?id=uLl7tSUOir
  raw: '[[raw/papers/md/2025-blockffn-chunk-level-activation-sparsity]]'
  source: https://openreview.net/pdf?id=uLl7tSUOir
owner: blaz
parser: read
raw_md: raw/papers/md/2025-blockffn-chunk-level-activation-sparsity.md
raw_pdf: raw/papers/pdf/2025-blockffn-chunk-level-activation-sparsity.pdf
read: false
slug: blockffn-chunk-level-activation-sparsity
tags:
- type/paper
- status/stub
title: 'BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level
  Activation Sparsity'
type: note
updated: '2026-05-12'
venue: COLM 2025
year: 2025
---

# BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity

> *Chenyang Song, Weilin Zhao, Xu Han, Chaojun Xiao, Yingfa Chen, et al.* — COLM 2025

## TL;DR

(stub — fill in after reading)

## Abstract

To alleviate the computational burden of large language models (LLMs), architectures with activation sparsity, represented by mixture-of-experts (MoE), have attracted increasing attention. However, the non-differentiable and inflexible routing of vanilla MoE hurts model performance. Moreover, while each token activates only a few parameters, these sparsely-activated architectures exhibit low chunk-level sparsity, indicating that the union of multiple consecutive tokens activates a large ratio of parameters. Such a sparsity pattern is unfriendly for acceleration under low-resource conditions (e.g., end-side devices) and incompatible with mainstream acceleration techniques (e.g., speculative decoding). To address these challenges, we introduce a novel MoE architecture, BlockFFN, as well as its efficient training and deployment techniques. Specifically, we use a router integrating ReLU activation and RMSNorm for differentiable and flexible routing. Next, to promote both token-level sparsity (TLS) and chunk-level sparsity (CLS), CLS-aware training objectives are designed, making BlockFFN more acceleration-friendly. Finally, we implement efficient acceleration kernels, combining activation sparsity and speculative decoding for the first time. The experimental results demonstrate the superior performance of BlockFFN over other MoE baselines, achieving over 80% TLS and 70% 8-token CLS. Our kernels achieve up to 3.67× speedup on real end-side devices than dense models.

## Notes

(stub)

## Source

- OpenReview: <https://openreview.net/pdf?id=uLl7tSUOir>
- PDF: [[raw/papers/pdf/2025-blockffn-chunk-level-activation-sparsity.pdf]]
- Code: <https://github.com/thunlp/BlockFFN>
