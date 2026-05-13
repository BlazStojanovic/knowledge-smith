---
arxiv: '2604.15039'
authors:
- Ruoyu Qin
- Weiran He
- Yaoyu Wang
- Zheming Li
- Xinran Xu
- Yongwei Wu
- Weimin Zheng
- Mingxing Zhang
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.15039
  raw: '[[raw/papers/md/2026-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter]]'
  source: https://arxiv.org/abs/2604.15039
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter.md
raw_pdf: raw/papers/pdf/2026-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter.pdf
read: false
slug: prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter
tags:
- type/paper
- status/stub
title: 'Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter'
type: note
updated: '2026-05-11'
year: 2026
---

# Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

> *Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, et al.* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional dense-attention models, prefill generates huge KVCache traffics that keep prefill and decode tightly coupled within a single high-bandwidth network domain, limiting heterogeneous deployment and resource elasticity. Recent hybrid-attention architectures substantially reduce KVCache size, making cross-cluster KVCache transport increasingly plausible. However, smaller KVCache alone does not make heterogeneous cross-datacenter PD serving practical: real workloads remain bursty, request lengths are highly skewed, prefix caches are unevenly distributed, and inter-cluster bandwidth fluctuates. A naive design that fully externalizes prefill can therefore still suffer from congestion, unstable queueing, and poor utilization.
  We present Prefill-as-a-Service (PrfaaS), a cross-datacenter serving architecture that selectively offloads long-context prefill to standalone, compute-dense prefill clusters and transfers the resulting KVCache over commodity Ethernet to local PD clusters for decode. Rather than treating reduced KVCache as sufficient, PrfaaS combines model-side KV efficiency with system-side selective offloading, bandwidth-aware scheduling, and cache-aware request placement. This design removes the requirement that heterogeneous accelerators share the same low-latency RDMA fabric, enabling independent scaling of prefill and decode capacity across loosely coupled clusters. In a case study using an internal 1T-parameter hybrid model, a PrfaaS-augmented heterogeneous deployment achieves 54% higher serving throughput and 64% lower P90 TTFT than a homogeneous PD baseline, with approximately 15% throughput gain at equal cost, while consuming only modest cross-datacenter bandwidth.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.15039>
- PDF: [[raw/papers/pdf/2026-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter.pdf]]
- Raw markdown: [[raw/papers/md/2026-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross-datacenter]]
