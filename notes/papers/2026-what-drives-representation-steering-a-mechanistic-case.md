---
arxiv: '2604.08524'
authors:
- Stephen Cheng
- Sarah Wiegreffe
- Dinesh Manocha
created: '2026-05-09'
doi: null
kind: paper
links:
  code: null
  paper: https://arxiv.org/abs/2604.08524
  raw: '[[raw/papers/md/2026-what-drives-representation-steering-a-mechanistic-case]]'
  source: https://arxiv.org/abs/2604.08524
owner: blaz
parser: ar5iv
raw_md: raw/papers/md/2026-what-drives-representation-steering-a-mechanistic-case.md
raw_pdf: raw/papers/pdf/2026-what-drives-representation-steering-a-mechanistic-case.pdf
read: false
slug: what-drives-representation-steering-a-mechanistic-case
tags:
- type/paper
- interpretability
- alignment
- llm
- status/stub
title: What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal
type: note
updated: '2026-05-09'
venue: null
year: 2026
---

# What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

> *Stephen Cheng, Sarah Wiegreffe, Dinesh Manocha* — arXiv 2604.08524, 2026

## TL;DR

(stub — fill in after reading)

## Abstract

Applying steering vectors to large language models (LLMs) is an efficient and effective model alignment technique, but we lack an interpretable explanation for how it works-- specifically, what internal mechanisms steering vectors affect and how this results in different model outputs. To investigate the causal mechanisms underlying the effectiveness of steering vectors, we conduct a comprehensive case study on refusal. We propose a multi-token activation patching framework and discover that different steering methodologies leverage functionally interchangeable circuits when applied at the same layer. These circuits reveal that steering vectors primarily interact with the attention mechanism through the OV circuit while largely ignoring the QK circuit-- freezing all attention scores during steering drops performance by only 8.75% across two model families. A mathematical decomposition of the steered OV circuit further reveals semantically interpretable concepts, even in cases where the steering vector itself does not. Leveraging the activation patching results, we show that steering vectors can be sparsified by up to 90-99% while retaining most performance, and that different steering methodologies agree on a subset of important dimensions.

## Notes

(your synthesis)

## Source

- Raw markdown: [[raw/papers/md/2026-what-drives-representation-steering-a-mechanistic-case]]
- PDF: [[raw/papers/pdf/2026-what-drives-representation-steering-a-mechanistic-case.pdf]]
- arXiv: <https://arxiv.org/abs/2604.08524>
