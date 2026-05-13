---
arxiv: '2407.00079'
authors:
- Ruoyu Qin
- Zheming Li
- Weiran He
- Mingxing Zhang
- Yongwei Wu
- Weimin Zheng
- Xinran Xu
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2407.00079
  raw: '[[raw/papers/md/2024-mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving]]'
  source: https://arxiv.org/abs/2407.00079
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2024-mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving.md
raw_pdf: raw/papers/pdf/2024-mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving.pdf
read: false
slug: mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving
tags:
- type/paper
- status/stub
title: 'Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving'
type: note
updated: '2026-05-11'
year: 2024
---

# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

> *Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, et al.* — arXiv 2024

## TL;DR

(stub — fill in after reading)

## Abstract

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latency-related Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle 75% more requests.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2407.00079>
- PDF: [[raw/papers/pdf/2024-mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving.pdf]]
- Raw markdown: [[raw/papers/md/2024-mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving]]
