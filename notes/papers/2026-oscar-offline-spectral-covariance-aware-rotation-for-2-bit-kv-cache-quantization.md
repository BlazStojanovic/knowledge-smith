---
arxiv: '2605.17757'
authors:
- Zhongzhu Zhou
- Donglin Zhuang
- Jisen Li
- Ziyan Chen
- Shuaiwen Leon Song
- Ben Athiwaratkun
- Xiaoxia Wu
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.17757
  raw: '[[raw/papers/md/2026-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization]]'
  source: https://arxiv.org/abs/2605.17757
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization.md
raw_pdf: raw/papers/pdf/2026-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization.pdf
read: false
slug: oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization
tags:
- type/paper
- status/stub
title: 'OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization'
type: note
updated: '2026-05-28'
year: 2026
---

# OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization

> *Zhongzhu Zhou, Donglin Zhuang, Jisen Li, Ziyan Chen, Shuaiwen Leon Song, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

INT2 KV-cache quantization is attractive for long-context LLM serving, but it remains difficult to make both accurate and deployable. Simple rotations such as Hadamard transforms reduce outliers, but still degrade at INT2 because they are not aligned with downstream attention. We propose OSCAR, an Ultra-low-bit KV Cache quantization method that estimates attention-aware covariance structures offline and uses them to derive fixed rotations and clipping thresholds for quantization. In this way, it aligns KV quantization with the covariance structures that attention actually consumes. More importantly, we not only provide theoretical justification but also develop a fully deployable OSCAR system with a custom INT2 attention kernel that remains compatible with paged KV-cache serving and fused kernel pipelines, enabling seamless integration into modern LLM serving frameworks such as SGLang and vLLM.
  We evaluate our methods on recent reasoning models with reasoning traces of up to 32k tokens across 5 tasks. On Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR reduces the BF16 accuracy gap to 3.78 and 1.42 points, respectively, while naive rotation INT2 collapses to nearly zero. We further scale OSCAR to Qwen3-32B and GLM-4.7 (358B params), where it remains effectively on par with BF16. On long context - RULER-NIAH up to 128K, OSCAR remains robust on both Qwen3 models, while naive rotation INT2 collapses. System-wise, OSCAR reduces KV-cache memory by approximately 8x, improves throughput by up to 7x at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3x over BF16 due to reduced memory bandwidth overhead.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.17757>
- PDF: [[raw/papers/pdf/2026-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization.pdf]]
- Raw markdown: [[raw/papers/md/2026-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization]]
