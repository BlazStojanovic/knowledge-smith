---
arxiv: '2605.26266'
authors:
- Tuna Tuncer
- Felix Becker
- Thomas Pfeil
created: '2026-05-28'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2605.26266
  raw: '[[raw/papers/md/2026-quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in]]'
  source: https://arxiv.org/abs/2605.26266
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in.md
raw_pdf: raw/papers/pdf/2026-quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in.pdf
read: false
slug: quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in
tags:
- type/paper
- status/stub
title: 'Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in
  Video Diffusion'
type: note
updated: '2026-05-28'
year: 2026
---

# Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion

> *Tuna Tuncer, Felix Becker, Thomas Pfeil* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Chunk-wise autoregressive video diffusion models rely on a KV cache of previously generated chunks to avoid redundant computation, but this cache quickly becomes a memory bottleneck as videos grow longer. Methods that quantize the KV cache to low bitwidths reduce memory pressure but degrade video quality. We show that a key driver of this degradation is a systematic bias in attention weights: due to the convexity of the exponential in softmax attention, quantization noise inflates the contribution of cached keys, a phenomenon we call the Jensen bias. This effect causes quantized keys to steal attention mass from the unquantized current chunk. We derive a per-attention-score correction that removes this bias in expectation, computed on the fly from the quantization step sizes of the cached keys and the query norm. Using a second-order Taylor approximation, the additional computational overhead is negligible, and no additional memory is needed alongside the cache. Evaluated on MAGI-1, SkyReels-V2, and HY-WorldPlay at INT2 quantization, our correction recovers most of the quality lost to aggressive quantization, reaching near-BF16 video quality, and can outperform INT4 quantization while using 50% less memory.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2605.26266>
- PDF: [[raw/papers/pdf/2026-quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in.pdf]]
- Raw markdown: [[raw/papers/md/2026-quantized-keys-steal-attention-bias-correction-for-kv-cache-compression-in]]
