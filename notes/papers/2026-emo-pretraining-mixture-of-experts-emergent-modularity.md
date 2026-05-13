---
authors:
- Ryan Wang
- Akshita Bhagia
- Sewon Min
created: '2026-05-12'
kind: paper
links:
  code: https://github.com/allenai/EMO
  paper: https://allenai.org/papers/emo
  raw: '[[raw/papers/md/2026-emo-pretraining-mixture-of-experts-emergent-modularity]]'
  source: https://allenai.org/papers/emo
owner: blaz
parser: read
raw_md: raw/papers/md/2026-emo-pretraining-mixture-of-experts-emergent-modularity.md
raw_pdf: raw/papers/pdf/2026-emo-pretraining-mixture-of-experts-emergent-modularity.pdf
read: false
slug: emo-pretraining-mixture-of-experts-emergent-modularity
tags:
- type/paper
- status/stub
title: 'EMO: Pretraining Mixture of Experts for Emergent Modularity'
type: note
updated: '2026-05-12'
venue: preprint
year: 2026
---

# EMO: Pretraining Mixture of Experts for Emergent Modularity

> *Ryan Wang, Akshita Bhagia, Sewon Min* — UC Berkeley / Allen Institute for AI, 2026 preprint

## TL;DR

(stub — fill in after reading)

## Abstract

Large language models are typically deployed as monolithic systems, requiring the full model even when applications need only a narrow subset of capabilities, e.g., code, math, or domain-specific knowledge. Mixture-of-Experts (MoEs) seemingly offer a potential alternative by activating only a subset of experts per input, but in practice, restricting inference to a subset of experts for a given domain leads to severe performance degradation. This limits their practicality in memory-constrained settings, especially as models grow larger and sparser. We introduce EMO, an MoE designed for modularity—the independent use and composition of expert subsets—without requiring human-defined priors. Our key idea is to encourage tokens from similar domains to rely on similar experts. Since tokens within a document often share a domain, EMO restricts them to select experts from a shared pool, while allowing different documents to use different pools. This simple constraint enables coherent expert groupings to emerge during pretraining using document boundaries alone. We pretrain a 1B-active, 14B-total EMO on 1T tokens. As a full model, it matches standard MoE performance. Crucially, it enables selective expert use: retaining only 25% (12.5%) of experts incurs just a 1% (3%) absolute drop, whereas standard MoEs break under the same setting. We further find that expert subsets in EMO specialize at semantic levels (e.g., domains such as math or code), in contrast to the low-level syntactic specialization observed in standard MoEs. Altogether, our results demonstrate a path toward modular, memory-efficient deployment of large, sparse models and open new opportunities for composable architectures.

## Notes

(stub)

## Source

- Tech report PDF: <https://allenai.org/papers/emo>
- Blog: <https://allenai.org/blog/emo>
- Model: <https://hf.co/allenai/EMO>
- Code: <https://github.com/allenai/EMO>
- PDF: [[raw/papers/pdf/2026-emo-pretraining-mixture-of-experts-emergent-modularity.pdf]]
