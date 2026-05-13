---
arxiv: '2603.15031'
authors:
- Kimi Team
- Guangyu Chen
- Yu Zhang
- Jianlin Su
- Weixin Xu
- Siyuan Pan
- Yaoyu Wang
- Yucheng Wang
- Guanduo Chen
- Bohong Yin
- Yutian Chen
- Junjie Yan
- Ming Wei
- Y. Zhang
- Fanqing Meng
- Chao Hong
- Xiaotong Xie
- Shaowei Liu
- Enzhe Lu
- Yunpeng Tai
- Yanru Chen
- Xin Men
- Haiqing Guo
- Y. Charles
- Haoyu Lu
- Lin Sui
- Jinguo Zhu
- Zaida Zhou
- Weiran He
- Weixiao Huang
- Xinran Xu
- Yuzhi Wang
- Guokun Lai
- Yulun Du
- Yuxin Wu
- Zhilin Yang
- Xinyu Zhou
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.15031
  raw: '[[raw/papers/md/2026-attention-residuals]]'
  source: https://arxiv.org/abs/2603.15031
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-attention-residuals.md
raw_pdf: raw/papers/pdf/2026-attention-residuals.pdf
read: false
slug: attention-residuals
tags:
- type/paper
- status/stub
title: Attention Residuals
type: note
updated: '2026-05-11'
year: 2026
---

# Attention Residuals

> *Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.
  Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2603.15031>
- PDF: [[raw/papers/pdf/2026-attention-residuals.pdf]]
- Raw markdown: [[raw/papers/md/2026-attention-residuals]]
