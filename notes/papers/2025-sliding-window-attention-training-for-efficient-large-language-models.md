---
arxiv: '2502.18845'
authors:
- Zichuan Fu
- Wentao Song
- Yejing Wang
- Xian Wu
- Yefeng Zheng
- Yingying Zhang
- Derong Xu
- Xuetao Wei
- Tong Xu
- Xiangyu Zhao
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2502.18845
  raw: '[[raw/papers/md/2025-sliding-window-attention-training-for-efficient-large-language-models]]'
  source: https://arxiv.org/abs/2502.18845
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2025-sliding-window-attention-training-for-efficient-large-language-models.md
raw_pdf: raw/papers/pdf/2025-sliding-window-attention-training-for-efficient-large-language-models.pdf
read: false
slug: sliding-window-attention-training-for-efficient-large-language-models
tags:
- type/paper
- status/stub
title: Sliding Window Attention Training for Efficient Large Language Models
type: note
updated: '2026-05-11'
year: 2025
---

# Sliding Window Attention Training for Efficient Large Language Models

> *Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, et al.* — arXiv 2025

## TL;DR

(stub — fill in after reading)

## Abstract

Recent advances in transformer-based Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their quadratic computational complexity concerning sequence length remains a significant bottleneck for processing long documents. As a result, many efforts like sparse attention and state space models have been proposed to improve the efficiency of LLMs over long sequences. Though effective, these approaches compromise the performance or introduce structural complexity. This calls for a simple yet efficient model that preserves the fundamental Transformer architecture. To this end, we introduce SWAT, which enables efficient long-context handling via Sliding Window Attention Training. This paper first attributes the inefficiency of Transformers to the attention sink phenomenon resulting from the high variance of softmax operation. Then, we replace softmax with the sigmoid function and utilize a balanced ALiBi and Rotary Position Embedding for efficient information compression and retention. Experiments demonstrate that SWAT achieves SOTA performance compared with state-of-the-art linear recurrent architectures on eight benchmarks. Code is available at https://github.com/Fzkuji/swat-attention.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2502.18845>
- PDF: [[raw/papers/pdf/2025-sliding-window-attention-training-for-efficient-large-language-models.pdf]]
- Raw markdown: [[raw/papers/md/2025-sliding-window-attention-training-for-efficient-large-language-models]]
