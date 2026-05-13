---
arxiv: '2604.19157'
authors:
- Jinda Jia
- Jisen Li
- Zhongzhu Zhou
- Jung Hwan Heo
- Jue Wang
- Tri Dao
- Shuaiwen Leon Song
- Ben Athiwaratkun
- Chenfeng Xu
- Tianyi Zhang
- Xiaoxia Wu
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.19157
  raw: '[[raw/papers/md/2026-saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving]]'
  source: https://arxiv.org/abs/2604.19157
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving.md
raw_pdf: raw/papers/pdf/2026-saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving.pdf
read: false
slug: saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving
tags:
- type/paper
- status/stub
title: 'SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving'
type: note
updated: '2026-05-11'
year: 2026
---

# SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving

> *Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although many KV-cache compression methods improve offline accuracy or compression ratio, they often violate practical serving constraints such as paged memory layouts, regular memory access, and fused attention execution, limiting their effectiveness in deployment.
  In this work, we identify the minimal set of 4-bit KV-cache quantization methods that remain viable under these constraints. Our central finding is that a simple design--token-wise INT4 quantization with block-diagonal Hadamard rotation--consistently achieves the best accuracy-efficiency trade-off. Across multiple models and benchmarks, this approach recovers nearly all of the accuracy lost by naive INT4, while more complex methods such as vector quantization and Hessian-aware quantization provide only marginal additional gains once serving compatibility is taken into account.
  To make this practical, we implement a fused rotation-quantization kernel that integrates directly into paged KV-cache layouts and introduces zero measurable end-to-end overhead, matching plain INT4 throughput across concurrency levels. Our results show that effective KV-cache compression is fundamentally a systems co-design problem: under real serving constraints, lightweight block-diagonal Hadamard rotation is a viable method that delivers near-lossless accuracy without sacrificing serving efficiency.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.19157>
- PDF: [[raw/papers/pdf/2026-saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving.pdf]]
- Raw markdown: [[raw/papers/md/2026-saw-int4-system-aware-4-bit-kv-cache-quantization-for-real-world-llm-serving]]
