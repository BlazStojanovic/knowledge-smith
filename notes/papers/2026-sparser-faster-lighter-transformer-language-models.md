---
arxiv: '2603.23198'
authors:
- Edoardo Cetin
- Stefano Peluchetti
- Emilio Castillo
- Akira Naruse
- Mana Murakami
- Llion Jones
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2603.23198
  raw: '[[raw/papers/md/2026-sparser-faster-lighter-transformer-language-models]]'
  source: https://arxiv.org/abs/2603.23198
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-sparser-faster-lighter-transformer-language-models.md
raw_pdf: raw/papers/pdf/2026-sparser-faster-lighter-transformer-language-models.pdf
read: false
slug: sparser-faster-lighter-transformer-language-models
tags:
- type/paper
- transformer
- mixture-of-experts
- llm
- status/stub
title: Sparser, Faster, Lighter Transformer Language Models
type: note
updated: '2026-05-09'
venue: null
year: 2026
---

# Sparser, Faster, Lighter Transformer Language Models

> *Edoardo Cetin, Stefano Peluchetti, Emilio Castillo…* — arXiv 2603.23198, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Scaling autoregressive large language models (LLMs) has driven unprecedented progress but comes with vast computational costs. In this work, we tackle these costs by leveraging unstructured sparsity within an LLM's feedforward layers, the components accounting for most of the model parameters and execution FLOPs. To achieve this, we introduce a new sparse packing format and a set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs, enabling efficient sparse computation during LLM inference and training. To substantiate our gains, we provide a quantitative study of LLM sparsity, demonstrating that simple L1 regularization can induce over 99% sparsity with negligible impact on downstream performance. When paired with our kernels, we show that these sparsity levels translate into substantial throughput, energy efficiency, and memory usage benefits that increase with model scale. We will release all code and kernels under an open-source license to promote adoption and accelerate research toward establishing sparsity as a practical axis for improving the efficiency and scalability of modern foundation models.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-sparser-faster-lighter-transformer-language-models]]
- PDF: [[raw/papers/pdf/2026-sparser-faster-lighter-transformer-language-models.pdf]]
- arXiv: <https://arxiv.org/abs/2603.23198>
