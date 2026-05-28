---
arxiv: '2512.12087'
authors:
- Jiayi Yuan
- Cameron Shinn
- Kai Xu
- Jingze Cui
- George Klimiashvili
- Guangxuan Xiao
- Perkz Zheng
- Bo Li
- Yuxin Zhou
- Zhouhai Ye
- Weijie You
- Tian Zheng
- Dominic Brown
- Pengbo Wang
- Markus Hoehnerbach
- Richard Cai
- Julien Demouth
- John D. Owens
- Xia Hu
- Song Han
- Timmy Liu
- Huizi Mao
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2512.12087
  raw: '[[raw/papers/md/2025-blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding]]'
  source: https://arxiv.org/abs/2512.12087
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding.md
raw_pdf: raw/papers/pdf/2025-blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding.pdf
read: false
slug: blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding
tags:
- type/paper
- status/stub
title: 'BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding'
type: note
updated: '2026-05-28'
year: 2025
---

# BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

> *Jiayi Yuan, Cameron Shinn, Kai Xu, Jingze Cui, George Klimiashvili, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

The growing demand for long-context inference capabilities in Large Language Models (LLMs) has intensified the computational and memory bottlenecks inherent to the self-attention mechanism. To address this challenge, we introduce BLASST, a drop-in, dynamic sparse attention mechanism that accelerates inference by using only a fixed scalar threshold to skip attention blocks. Our method targets practical inference deployment by removing the barriers to adoption present in existing works. As such, BLASST eliminates training requirements, avoids expensive pre-computation passes, accelerates both prefill and decode across all major attention variants (MHA, GQA, MQA, and MLA), provides optimized support for modern hardware, and easily integrates into existing frameworks. This is achieved by reusing online softmax statistics to identify negligible attention scores, skipping softmax, value block loads, and the subsequent matrix multiplication. We demonstrate the BLASST algorithm by delivering optimized kernels with negligible latency overhead. Our automated threshold calibration procedure reveals a simple inverse relationship between optimal threshold and context length, meaning we require only a single threshold each for prefill and decode per model. Preserving benchmark accuracy, we demonstrate a 1.52x speedup for prefill at 71.9% sparsity and a 1.48x speedup for decode at 73.2% sparsity on modern GPUs.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2512.12087>
- PDF: [[raw/papers/pdf/2025-blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding.pdf]]
- Raw markdown: [[raw/papers/md/2025-blasst-dynamic-blocked-attention-sparsity-via-softmax-thresholding]]
