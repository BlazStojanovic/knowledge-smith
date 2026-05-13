---
arxiv: '2604.21254'
authors:
- Abbas Zeitoun
- Lucas Torroba-Hennigen
- Yoon Kim
created: '2026-05-11'
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.21254
  raw: '[[raw/papers/md/2026-hyperloop-transformers]]'
  source: https://arxiv.org/abs/2604.21254
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-hyperloop-transformers.md
raw_pdf: raw/papers/pdf/2026-hyperloop-transformers.pdf
read: false
slug: hyperloop-transformers
tags:
- type/paper
- status/stub
title: Hyperloop Transformers
type: note
updated: '2026-05-11'
year: 2026
---

# Hyperloop Transformers

> *Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim* — arXiv 2026

## TL;DR

(stub — fill in after reading)

## Abstract

LLM architecture research generally aims to maximize model quality subject to fixed compute/latency budgets. However, many applications of interest such as edge and on-device deployment are further constrained by the model's memory footprint, thus motivating parameter-efficient architectures for language modeling. This paper describes a simple architecture that improves the parameter-efficiency of LLMs. Our architecture makes use of looped Transformers as a core primitive, which reuse Transformer layers across depth and are thus more parameter-efficient than ordinary (depth-matched) Transformers. We organize the looped Transformer into three blocks--begin, middle, and end blocks--where each block itself consists of multiple Transformer layers, and only the middle block is applied recurrently across depth. We augment the looped middle block with hyper-connections (Xie et al., 2026), which expand the residual stream into matrix-valued residual streams. Hyper-connections are applied only after each loop, and therefore add minimal new parameters and compute cost. Across various model scales, we find that our Hyper-Connected Looped Transformer (Hyperloop Transformer) is able to outperform depth-matched Transformer and mHC Transformer baselines despite using approximately 50% fewer parameters. The outperformance persists through post-training weight quantization, thus positioning Hyperloop Transformers as an attractive architecture for memory-efficient language modeling.

## Notes

(stub)

## Source

- arXiv: <https://arxiv.org/abs/2604.21254>
- PDF: [[raw/papers/pdf/2026-hyperloop-transformers.pdf]]
- Raw markdown: [[raw/papers/md/2026-hyperloop-transformers]]
