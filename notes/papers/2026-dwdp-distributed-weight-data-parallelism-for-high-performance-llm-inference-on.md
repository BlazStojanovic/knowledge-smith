---
arxiv: '2604.01621'
authors:
- Wanqian Li
- Jintao Peng
- Zongfei Jing
- Tianyu Zhang
- Ze Long
- Xianjie Qiao
- Xiaoming Chen
- Dongxu Yang
- Kefeng Duan
- June Yang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.01621
  raw: '[[raw/papers/md/2026-dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on]]'
  source: https://arxiv.org/abs/2604.01621
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on.md
raw_pdf: raw/papers/pdf/2026-dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on.pdf
read: false
slug: dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on
tags:
- type/paper
- status/stub
title: 'DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference
  on NVL72'
type: note
updated: '2026-05-11'
year: 2026
---

# DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72

> *Wanqian Li, Jintao Peng, Zongfei Jing, Tianyu Zhang, Ze Long, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Large language model (LLM) inference increasingly depends on multi-GPU execution, yet existing inference parallelization strategies require layer-wise inter-rank synchronization, making end-to-end performance sensitive to workload imbalance. We present DWDP (Distributed Weight Data Parallelism), an inference parallelization strategy that preserves data-parallel execution while offloading MoE weights across peer GPUs and fetching missing experts on demand. By removing collective inter-rank synchronization, DWDP allows each GPU to progress independently. We further address the practical overheads of this design with two optimizations for split-weight management and asynchronous remote-weight prefetch. Implemented in TensorRT-LLM and evaluated with DeepSeek-R1 on GB200 NVL72, DWDP improves end-to-end output TPS/GPU by 8.8% at comparable TPS/user in the 20-100 TPS/user serving range under 8K input sequence length and 1K output sequence length.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.01621>
- PDF: [[raw/papers/pdf/2026-dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on.pdf]]
- Raw markdown: [[raw/papers/md/2026-dwdp-distributed-weight-data-parallelism-for-high-performance-llm-inference-on]]
